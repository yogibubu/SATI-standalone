from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile

import numpy as np

from matrix_chem import preprocess_to_enriched_xyz, write_validation_section
from matrix_chem.topology.elements import atomic_symbol
from matrix_core import (
    read_sectioned_lines,
    section_content,
)
from matrix_numerics import eigh_arrays
from matrix_smith import (
    CV_RADIAL_SIGMA_SCALE,
    CV_RADIAL_WEIGHT_THRESHOLD,
    FrozenGIC,
    GICDefinition,
    GICPrimitive,
    apply_cv_radial_geometry_correction,
    build_gic_b_matrix,
    read_gic_definition_from_xyzin,
    topology_bond_order_components_from_lines,
    topology_bond_orders_from_lines,
    write_gicforge_build_sections,
)

from .harmonic import solve_wilson_gf
from .large_amplitude import (
    GFLargeAmplitudeAnalysis,
    LargeAmplitudeTopologyContext,
    classify_gic_anharmonic_model,
    large_amplitude_analysis_from_gf_matrices,
)
from .large_amplitude import large_amplitude_topology_context_from_arrays
from .models import HessianInput


@dataclass(frozen=True)
class RedundantPrimitiveHessian:
    """Cartesian curvature represented on a redundant primitive pool.

    The transformation is performed only after translations and rotations
    have been removed in mass-weighted Cartesian space.  Consequently the
    rank is the molecular vibrational rank, even when fixed-frame SOOP rows
    contain Cartesian external components.
    """

    force_constants: np.ndarray
    vibrational_basis: np.ndarray
    primitive_jacobian: np.ndarray
    primitive_rank: int


def redundant_primitive_hessian_from_cartesian(
    cartesian_hessian: np.ndarray,
    primitive_b_matrix: np.ndarray,
    coordinates_bohr: np.ndarray,
    masses_amu: np.ndarray,
    *,
    rcond: float = 1.0e-10,
) -> RedundantPrimitiveHessian:
    """Transform a Cartesian Hessian onto a redundant primitive pool.

    ``primitive_b_matrix`` is the derivative with respect to bohr Cartesian
    coordinates.  The returned matrix is the minimum-norm redundant
    representation of the projected internal Cartesian curvature.
    """

    hessian = np.asarray(cartesian_hessian, dtype=float)
    b_matrix = np.asarray(primitive_b_matrix, dtype=float)
    xyz = np.asarray(coordinates_bohr, dtype=float)
    masses = np.asarray(masses_amu, dtype=float).reshape(-1)
    size = 3 * len(masses)
    if hessian.shape != (size, size) or b_matrix.ndim != 2 or b_matrix.shape[1] != size:
        raise ValueError("inconsistent Cartesian Hessian or primitive B matrix")
    if xyz.shape != (len(masses), 3) or np.any(masses <= 0.0):
        raise ValueError("inconsistent geometry or masses")

    center = np.average(xyz, axis=0, weights=masses)
    centered = xyz - center
    sqrt_mass = np.sqrt(masses)[:, None]
    external = [
        (sqrt_mass * np.tile(axis, (len(masses), 1))).reshape(-1)
        for axis in np.eye(3)
    ]
    external.extend(
        (sqrt_mass * np.cross(np.tile(axis, (len(masses), 1)), centered)).reshape(-1)
        for axis in np.eye(3)
    )
    external_matrix = np.column_stack(external)
    u_ext, singular, _vh = np.linalg.svd(external_matrix, full_matrices=True)
    tolerance = rcond * (singular[0] if singular.size else 1.0)
    external_rank = int(np.sum(singular > tolerance))
    vibrational_basis = u_ext[:, external_rank:]

    inverse_sqrt_mass = np.repeat(1.0 / np.sqrt(masses), 3)
    cartesian_internal = inverse_sqrt_mass[:, None] * vibrational_basis
    primitive_jacobian = b_matrix @ cartesian_internal
    internal_hessian = cartesian_internal.T @ (0.5 * (hessian + hessian.T)) @ cartesian_internal
    backtransform = np.linalg.pinv(primitive_jacobian, rcond=rcond)
    force_constants = backtransform.T @ internal_hessian @ backtransform
    force_constants = 0.5 * (force_constants + force_constants.T)
    return RedundantPrimitiveHessian(
        force_constants=force_constants,
        vibrational_basis=vibrational_basis,
        primitive_jacobian=primitive_jacobian,
        primitive_rank=int(np.linalg.matrix_rank(primitive_jacobian, tol=tolerance)),
    )


BOHR_TO_ANGSTROM = 0.52917721092


@dataclass(frozen=True)
class PEDTable:
    """Potential-energy distribution in non-redundant GIC coordinates."""

    values: np.ndarray
    labels: tuple[str, ...]


@dataclass(frozen=True)
class GFLocalOptions:
    """Pulay-style local force-field filtering options."""

    enabled: bool = False


@dataclass(frozen=True)
class InternalGFResult:
    frequencies_cm: np.ndarray
    force_constants: np.ndarray
    g_matrix: np.ndarray
    b_matrix: np.ndarray
    u_matrix: np.ndarray
    modes_internal: np.ndarray
    ped: PEDTable
    primitive_labels: tuple[str, ...]
    gic_labels: tuple[str, ...]
    gic_names: tuple[str, ...] = ()
    gic_irreps: tuple[str, ...] = ()
    gic_families: tuple[str, ...] = ()
    gic_anharmonic_classes: tuple[str, ...] = ()
    gic_zeroth_order_models: tuple[str, ...] = ()
    gic_cross_coupling_policies: tuple[str, ...] = ()
    gic_anharmonic_reasons: tuple[str, ...] = ()
    point_group: str = "UNKNOWN"
    symmetrized_gics: bool = False
    scaling_factors: np.ndarray | None = None
    coordinate_source: str = "frozen-gic-definition"
    matrix_model: str = "FULL"
    block_labels: tuple[str, ...] = ()
    force_threshold: float | None = None
    hessian_correction: str = "NONE"
    large_amplitude: GFLargeAmplitudeAnalysis | None = None


def primitive_label(primitive: object) -> str:
    function = getattr(primitive, "function", "")
    atoms = tuple(int(atom) for atom in getattr(primitive, "atoms"))
    mode = int(getattr(primitive, "mode", 0))
    atoms_text = ",".join(str(atom) for atom in atoms)
    if function == "R":
        return f"R({atoms_text})"
    if function == "A":
        return f"A({atoms_text})"
    if function == "D":
        return f"D({atoms_text})"
    if function == "U":
        return f"U({atoms_text})"
    if function == "H":
        return f"H({atoms_text})"
    if function == "IMPD":
        center, n1, n2, n3 = atoms
        return f"D({n1},{center},{n3},{n2})"
    if function == "L":
        return f"L({atoms_text},0,{mode})"
    if function == "RPCK":
        return getattr(primitive, "gaussian_expression")()
    suffix = f":{mode}" if mode else ""
    return f"{function}{suffix}({atoms_text})"


def gic_labels_from_u(
    u_matrix: np.ndarray,
    primitive_labels: tuple[str, ...],
    *,
    threshold: float = 0.15,
) -> tuple[str, ...]:
    labels: list[str] = []
    for col in range(u_matrix.shape[1]):
        terms = []
        for row, coeff in enumerate(u_matrix[:, col]):
            if abs(coeff) < threshold:
                continue
            sign = "+" if coeff >= 0.0 else "-"
            terms.append(f"{sign}{abs(coeff):.3f}*{primitive_labels[row]}")
        labels.append(" ".join(terms) if terms else f"GIC{col + 1}")
    return tuple(labels)


def _mass_inverse(masses_amu: np.ndarray) -> np.ndarray:
    weights = np.repeat(1.0 / np.asarray(masses_amu, dtype=float), 3)
    return np.diag(weights)


def _internal_backtransform(
    bq: np.ndarray, masses_amu: np.ndarray, g_matrix: np.ndarray
) -> np.ndarray:
    minv = _mass_inverse(masses_amu)
    return minv @ bq.T @ np.linalg.pinv(g_matrix, rcond=1.0e-10)


def _ped(
    force_constants: np.ndarray, modes_internal: np.ndarray, eigenvalues: np.ndarray
) -> np.ndarray:
    ped = np.zeros((force_constants.shape[0], modes_internal.shape[1]), dtype=float)
    for mode in range(modes_internal.shape[1]):
        lam = eigenvalues[mode]
        if abs(lam) < 1.0e-14:
            continue
        vector = modes_internal[:, mode]
        raw = vector * (force_constants @ vector) / lam
        total = float(np.sum(np.abs(raw)))
        if total > 0.0:
            ped[:, mode] = 100.0 * np.abs(raw) / total
    return ped


def pulay_scale_internal_hessian(
    force_constants: np.ndarray,
    diagonal_factors: np.ndarray | None,
) -> np.ndarray:
    """Scale an internal-coordinate Hessian with Pulay-style factors."""
    f_mat = np.asarray(force_constants, dtype=float)
    if diagonal_factors is None:
        return np.array(f_mat, dtype=float, copy=True)
    factors = np.asarray(diagonal_factors, dtype=float)
    if factors.shape != (f_mat.shape[0],):
        raise ValueError(f"Scaling factors must have length {f_mat.shape[0]}")
    if np.any(factors < 0.0):
        raise ValueError("Pulay scaling factors must be non-negative")
    scale = np.sqrt(np.outer(factors, factors))
    return 0.5 * (f_mat * scale + (f_mat * scale).T)


def gf_from_cartesian_hessian_and_gic_b_matrix(
    cartesian_hessian: np.ndarray,
    b_matrix_internal: np.ndarray,
    masses_amu: np.ndarray,
    *,
    b_matrix_hessian_transform: np.ndarray | None = None,
    gic_labels: tuple[str, ...],
    primitive_labels: tuple[str, ...] = (),
    u_matrix: np.ndarray | None = None,
    gic_names: tuple[str, ...] = (),
    gic_irreps: tuple[str, ...] = (),
    point_group: str = "UNKNOWN",
    symmetrized_gics: bool = False,
    scaling_factors: np.ndarray | None = None,
    local_mask: np.ndarray | None = None,
    force_threshold: float | None = None,
    block_by_irrep: bool = False,
    cartesian_hessian_correction: np.ndarray | None = None,
    cartesian_hessian_correction_label: str = "NONE",
    coordinate_source: str = "frozen-gic-definition",
    large_amplitude_frequency_cutoff_cm: float | None = None,
    large_amplitude_frequency_model: str = "projected",
    large_amplitude_topology_context=None,
) -> InternalGFResult:
    """Run Wilson GF from a Cartesian Hessian and fixed non-redundant B matrices.

    ``b_matrix_internal`` defines the metric geometry used for the Wilson G
    matrix and final GF diagonalization.  ``b_matrix_hessian_transform`` may be
    supplied when an external Cartesian Hessian belongs to a different source
    geometry; in that case the Hessian is transformed with the source-geometry
    B matrix while the G matrix is built from ``b_matrix_internal``.  A radial
    Cartesian correction is transformed independently and subtracted only in
    the non-redundant SONIC basis supplied by SMITH and consumed by TRINITY.
    """
    hessian = np.asarray(cartesian_hessian, dtype=float)
    correction = None
    if cartesian_hessian_correction is not None:
        correction = np.asarray(cartesian_hessian_correction, dtype=float)
        if correction.shape != hessian.shape:
            raise ValueError("Cartesian Hessian correction has inconsistent dimensions")
    bq = np.asarray(b_matrix_internal, dtype=float)
    bq_hessian = (
        bq
        if b_matrix_hessian_transform is None
        else np.asarray(b_matrix_hessian_transform, dtype=float)
    )
    masses = np.asarray(masses_amu, dtype=float)
    if hessian.shape != (3 * len(masses), 3 * len(masses)):
        raise ValueError("Cartesian Hessian has inconsistent dimensions")
    hessian = 0.5 * (hessian + hessian.T)
    if correction is not None:
        correction = 0.5 * (correction + correction.T)
    if bq.ndim != 2 or bq.shape[1] != hessian.shape[0]:
        raise ValueError("Internal B matrix has inconsistent dimensions")
    if bq_hessian.shape != bq.shape:
        raise ValueError("Hessian-transform B matrix must match the metric B matrix shape")
    if len(gic_labels) != bq.shape[0]:
        raise ValueError("GIC label count does not match B matrix rows")

    minv = _mass_inverse(masses)
    g_matrix = bq @ minv @ bq.T
    hessian_g_matrix = bq_hessian @ minv @ bq_hessian.T
    backtransform = _internal_backtransform(bq_hessian, masses, hessian_g_matrix)
    qm_force_constants = backtransform.T @ hessian @ backtransform
    correction_force_constants = (
        np.zeros_like(qm_force_constants)
        if correction is None
        else backtransform.T @ correction @ backtransform
    )
    force_constants = qm_force_constants - correction_force_constants
    force_constants = 0.5 * (force_constants + force_constants.T)
    force_constants = pulay_scale_internal_hessian(force_constants, scaling_factors)
    matrix_model = "FULL"
    if local_mask is not None:
        mask = np.asarray(local_mask, dtype=bool)
        if mask.shape != force_constants.shape:
            raise ValueError("Local force-constant mask has inconsistent dimensions")
        force_constants = np.where(mask, force_constants, 0.0)
        force_constants = 0.5 * (force_constants + force_constants.T)
        matrix_model = "LOCAL"
    if force_threshold is not None:
        threshold = float(force_threshold)
        if threshold < 0.0:
            raise ValueError("Force-constant threshold must be non-negative")
        force_constants = np.where(np.abs(force_constants) < threshold, 0.0, force_constants)
        force_constants = 0.5 * (force_constants + force_constants.T)

    effective_block_by_irrep = bool(block_by_irrep)
    if gic_irreps:
        if len(gic_irreps) != force_constants.shape[0]:
            raise ValueError("GIC irrep count does not match internal-matrix dimensions")
        symmetry_mask = np.equal.outer(np.asarray(gic_irreps), np.asarray(gic_irreps))
        force_constants = np.where(symmetry_mask, force_constants, 0.0)
        g_matrix = np.where(symmetry_mask, g_matrix, 0.0)
        force_constants = 0.5 * (force_constants + force_constants.T)
        g_matrix = 0.5 * (g_matrix + g_matrix.T)
        matrix_model = f"{matrix_model}+SYMMETRY_PROJECTED"
        effective_block_by_irrep = True

    gf, g_matrix, force_constants, block_labels = _solve_internal_gf(
        force_constants,
        g_matrix,
        gic_irreps=tuple(gic_irreps),
        block_by_irrep=effective_block_by_irrep,
    )
    if block_labels:
        matrix_model = f"{matrix_model}+IRREP_BLOCKS"
    g_eval, g_vec = eigh_arrays(0.5 * (g_matrix + g_matrix.T))
    g_inv_half = (g_vec * (1.0 / np.sqrt(np.clip(g_eval, 1.0e-14, None)))) @ g_vec.T
    modes_internal = g_inv_half @ gf.normal_modes
    ped = _ped(force_constants, modes_internal, gf.eigenvalues)
    assignment_names = tuple(
        gic_names[idx] if idx < len(gic_names) else f"GIC{idx + 1:03d}"
        for idx in range(len(gic_labels))
    )
    assignments = tuple(
        classify_gic_anharmonic_model(
            assignment_names[idx],
            label,
            topology_context=large_amplitude_topology_context,
        )
        for idx, label in enumerate(gic_labels)
    )
    large_amplitude = large_amplitude_analysis_from_gf_matrices(
        force_constants=force_constants,
        g_matrix=g_matrix,
        frequencies_cm=gf.frequencies_cm,
        ped=ped,
        gic_labels=tuple(gic_labels),
        gic_names=tuple(gic_names),
        gic_irreps=tuple(gic_irreps),
        frequency_cutoff_cm=large_amplitude_frequency_cutoff_cm,
        topology_context=large_amplitude_topology_context,
        block_frequency_model=large_amplitude_frequency_model,
        primitive_labels=tuple(primitive_labels),
        u_matrix=u_matrix,
    )
    stored_u = np.array(
        u_matrix if u_matrix is not None else np.eye(bq.shape[0]),
        dtype=float,
        copy=True,
    )
    if stored_u.ndim != 2 or stored_u.shape[1] != bq.shape[0]:
        raise ValueError("U matrix must have shape (n_primitives, n_gics)")
    if primitive_labels and len(primitive_labels) != stored_u.shape[0]:
        raise ValueError("Primitive label count does not match U matrix rows")

    return InternalGFResult(
        frequencies_cm=gf.frequencies_cm,
        force_constants=force_constants,
        g_matrix=g_matrix,
        b_matrix=bq,
        u_matrix=stored_u,
        modes_internal=modes_internal,
        ped=PEDTable(ped, gic_labels),
        primitive_labels=tuple(primitive_labels),
        gic_labels=tuple(gic_labels),
        gic_names=tuple(gic_names),
        gic_irreps=tuple(gic_irreps),
        gic_families=tuple(item.family for item in assignments),
        gic_anharmonic_classes=tuple(item.anharmonic_class for item in assignments),
        gic_zeroth_order_models=tuple(item.zeroth_order_model for item in assignments),
        gic_cross_coupling_policies=tuple(item.cross_coupling_policy for item in assignments),
        gic_anharmonic_reasons=tuple(item.reason for item in assignments),
        point_group=point_group,
        symmetrized_gics=bool(symmetrized_gics),
        scaling_factors=None
        if scaling_factors is None
        else np.asarray(scaling_factors, dtype=float),
        coordinate_source=coordinate_source,
        matrix_model=matrix_model,
        block_labels=block_labels,
        force_threshold=force_threshold,
        hessian_correction=cartesian_hessian_correction_label,
        large_amplitude=large_amplitude,
    )


def gf_from_hessian_input_and_gic_definition(
    input_data: HessianInput,
    definition: GICDefinition,
    *,
    coordinates_angstrom: np.ndarray | None = None,
    scaling_factors: np.ndarray | None = None,
    topology_bonds: tuple[tuple[int, int], ...] = (),
    local_options: GFLocalOptions | None = None,
    force_threshold: float | None = None,
    block_by_irrep: bool = False,
    cartesian_hessian_correction: np.ndarray | None = None,
    cartesian_hessian_correction_label: str = "NONE",
    large_amplitude_frequency_cutoff_cm: float | None = None,
    large_amplitude_frequency_model: str = "projected",
    cv_correction: bool = False,
    cv_correction_sigma_scale: float = CV_RADIAL_SIGMA_SCALE,
    cv_correction_threshold: float = CV_RADIAL_WEIGHT_THRESHOLD,
) -> InternalGFResult:
    """Run TRINITY harmonic analysis using a frozen SONIC definition and Hessian input."""
    input_data.validate()
    if len(definition.reference_coordinates_angstrom) != len(input_data.atomic_numbers):
        raise ValueError("GIC definition atom count does not match Hessian input")
    coords_for_b = (
        np.asarray(input_data.cartesian_coordinates_bohr, dtype=float) * BOHR_TO_ANGSTROM
        if coordinates_angstrom is None
        else np.asarray(coordinates_angstrom, dtype=float)
    )
    source_b_matrix = build_gic_b_matrix(definition, coordinates_angstrom=coords_for_b)
    coordinate_source = f"frozen-gic-definition:{definition.point_group}"
    if cv_correction:
        coords_for_b, cv_report = apply_cv_radial_geometry_correction(
            definition,
            input_data.atomic_numbers,
            coords_for_b,
            sigma_scale=cv_correction_sigma_scale,
            weight_threshold=cv_correction_threshold,
        )
        coordinate_source = f"{coordinate_source}+{cv_report.label}"
    b_matrix = build_gic_b_matrix(definition, coordinates_angstrom=coords_for_b)
    local_mask = None
    if local_options is not None and local_options.enabled:
        local_mask = local_force_constant_mask(definition, topology_bonds)
    large_context = large_amplitude_topology_context_from_arrays(
        atomic_numbers=input_data.atomic_numbers,
        coordinates_angstrom=coords_for_b,
    )
    result = gf_from_cartesian_hessian_and_gic_b_matrix(
        input_data.cartesian_hessian,
        np.asarray(b_matrix.rows, dtype=float),
        input_data.masses_amu,
        b_matrix_hessian_transform=np.asarray(source_b_matrix.rows, dtype=float),
        gic_labels=_gic_display_labels(definition),
        primitive_labels=tuple(primitive_label(primitive) for primitive in definition.primitives),
        u_matrix=_u_matrix_from_definition(definition),
        gic_names=b_matrix.coordinate_names,
        gic_irreps=b_matrix.irreps,
        point_group=definition.point_group,
        symmetrized_gics=definition.symmetrize,
        scaling_factors=scaling_factors,
        local_mask=local_mask,
        force_threshold=force_threshold,
        block_by_irrep=block_by_irrep,
        cartesian_hessian_correction=cartesian_hessian_correction,
        cartesian_hessian_correction_label=cartesian_hessian_correction_label,
        coordinate_source=coordinate_source,
        large_amplitude_frequency_cutoff_cm=large_amplitude_frequency_cutoff_cm,
        large_amplitude_frequency_model=large_amplitude_frequency_model,
        large_amplitude_topology_context=large_context,
    )
    if (
        coordinates_angstrom is None
        and scaling_factors is None
        and not (local_options is not None and local_options.enabled)
        and force_threshold is None
        and cartesian_hessian_correction is None
        and not cv_correction
    ):
        _validate_exact_hessian_ped_contract(input_data, result, definition.rank)
    return result


def gf_from_hessian_input_and_xyzin(
    input_data: HessianInput,
    xyzin_path: Path,
    *,
    scaling_factors: np.ndarray | None = None,
    local_options: GFLocalOptions | None = None,
    force_threshold: float | None = None,
    block_by_irrep: bool = False,
    cartesian_hessian_correction: np.ndarray | None = None,
    cartesian_hessian_correction_label: str = "NONE",
    large_amplitude_frequency_cutoff_cm: float | None = None,
    large_amplitude_frequency_model: str = "projected",
    cv_correction: bool = False,
    cv_correction_sigma_scale: float = CV_RADIAL_SIGMA_SCALE,
    cv_correction_threshold: float = CV_RADIAL_WEIGHT_THRESHOLD,
) -> InternalGFResult:
    """Run TRINITY harmonic analysis using the frozen #GIC section in a MATRIX xyzin."""
    input_data.validate()
    definition = read_gic_definition_from_xyzin(Path(xyzin_path))
    coords_for_b = (
        np.asarray(input_data.cartesian_coordinates_bohr, dtype=float) * BOHR_TO_ANGSTROM
    )
    source_b_matrix = build_gic_b_matrix(
        definition,
        coordinates_angstrom=coords_for_b,
    )
    coordinate_source = f"xyzin-frozen-gic:{Path(xyzin_path)}"
    if cv_correction:
        coords_for_b, cv_report = apply_cv_radial_geometry_correction(
            definition,
            input_data.atomic_numbers,
            coords_for_b,
            sigma_scale=cv_correction_sigma_scale,
            weight_threshold=cv_correction_threshold,
        )
        coordinate_source = f"{coordinate_source}+{cv_report.label}"
    b_matrix = build_gic_b_matrix(
        definition,
        coordinates_angstrom=coords_for_b,
    )
    local_mask = None
    if local_options is not None and local_options.enabled:
        local_mask = local_force_constant_mask(
            definition,
            topology_bonds_from_xyzin(Path(xyzin_path)),
        )
    large_context = _large_amplitude_topology_context_from_xyzin(
        Path(xyzin_path),
        atomic_numbers=input_data.atomic_numbers,
        coordinates_angstrom=coords_for_b,
    )
    result = gf_from_cartesian_hessian_and_gic_b_matrix(
        input_data.cartesian_hessian,
        np.asarray(b_matrix.rows, dtype=float),
        input_data.masses_amu,
        b_matrix_hessian_transform=np.asarray(source_b_matrix.rows, dtype=float),
        gic_labels=_gic_display_labels(definition),
        primitive_labels=tuple(primitive_label(primitive) for primitive in definition.primitives),
        u_matrix=_u_matrix_from_definition(definition),
        gic_names=b_matrix.coordinate_names,
        gic_irreps=b_matrix.irreps,
        point_group=definition.point_group,
        symmetrized_gics=definition.symmetrize,
        scaling_factors=scaling_factors,
        local_mask=local_mask,
        force_threshold=force_threshold,
        block_by_irrep=block_by_irrep,
        cartesian_hessian_correction=cartesian_hessian_correction,
        cartesian_hessian_correction_label=cartesian_hessian_correction_label,
        coordinate_source=coordinate_source,
        large_amplitude_frequency_cutoff_cm=large_amplitude_frequency_cutoff_cm,
        large_amplitude_frequency_model=large_amplitude_frequency_model,
        large_amplitude_topology_context=large_context,
    )
    if (
        scaling_factors is None
        and not (local_options is not None and local_options.enabled)
        and force_threshold is None
        and cartesian_hessian_correction is None
        and not cv_correction
    ):
        _validate_exact_hessian_ped_contract(input_data, result, definition.rank)
    return result


def _validate_exact_hessian_ped_contract(
    input_data: HessianInput,
    result: InternalGFResult,
    expected_mode_count: int,
) -> None:
    from .invariants import validate_hessian_ped_invariants

    validate_hessian_ped_invariants(
        input_data.cartesian_hessian,
        input_data.masses_amu,
        input_data.cartesian_coordinates_bohr,
        result,
        expected_mode_count=expected_mode_count,
        reported_cartesian_frequencies_cm=input_data.harmonic_frequencies_cm,
    )


def gf_from_acquired_force_constants_and_xyzin(
    acquired_force_field,
    hessian_geometry_input: HessianInput,
    xyzin_path: Path,
    *,
    local_options: GFLocalOptions | None = None,
    force_threshold: float | None = None,
    block_by_irrep: bool = False,
    large_amplitude_frequency_cutoff_cm: float | None = None,
    large_amplitude_frequency_model: str = "projected",
    cv_correction: bool = False,
    cv_correction_sigma_scale: float = CV_RADIAL_SIGMA_SCALE,
    cv_correction_threshold: float = CV_RADIAL_WEIGHT_THRESHOLD,
) -> InternalGFResult:
    """Run GF/PED from an acquired L1 internal force field.

    The Cartesian Hessian geometry supplies masses and the starting geometry for
    the Wilson metric.  When requested, CV-radial correction is applied only to
    the geometry used for B/G; the acquired force field is already expressed in
    the frozen coordinate basis.
    """

    hessian_geometry_input.validate()
    definition = read_gic_definition_from_xyzin(Path(xyzin_path))
    force_constants = np.asarray(
        getattr(acquired_force_field, "force_constants", acquired_force_field),
        dtype=float,
    )
    if force_constants.shape != (len(definition.gics), len(definition.gics)):
        raise ValueError("Acquired force-constant matrix does not match frozen #GIC dimension")
    coords_for_b = (
        np.asarray(hessian_geometry_input.cartesian_coordinates_bohr, dtype=float)
        * BOHR_TO_ANGSTROM
    )
    coordinate_source = f"xyzin-frozen-gic:{Path(xyzin_path)}+ACQUIRED_L1"
    if cv_correction:
        coords_for_b, cv_report = apply_cv_radial_geometry_correction(
            definition,
            hessian_geometry_input.atomic_numbers,
            coords_for_b,
            sigma_scale=cv_correction_sigma_scale,
            weight_threshold=cv_correction_threshold,
        )
        coordinate_source = f"{coordinate_source}+{cv_report.label}"
    b_matrix = build_gic_b_matrix(definition, coordinates_angstrom=coords_for_b)
    bq = np.asarray(b_matrix.rows, dtype=float)
    minv = _mass_inverse(np.asarray(hessian_geometry_input.masses_amu, dtype=float))
    # SMITH differentiates every internal coordinate with respect to Cartesian
    # coordinates in angstrom, whereas the harmonic conversion constant used
    # by solve_wilson_gf assumes bohr.  For an acquired force matrix expressed
    # in the natural SONIC units, dq/dx_bohr = a0 * dq/dx_angstrom.
    g_matrix = BOHR_TO_ANGSTROM**2 * (bq @ minv @ bq.T)
    if local_options is not None and local_options.enabled:
        mask = local_force_constant_mask(definition, topology_bonds_from_xyzin(Path(xyzin_path)))
        force_constants = np.where(mask, force_constants, 0.0)
    if force_threshold is not None:
        threshold = float(force_threshold)
        if threshold < 0.0:
            raise ValueError("Force-constant threshold must be non-negative")
        force_constants = np.where(np.abs(force_constants) < threshold, 0.0, force_constants)
    force_constants = 0.5 * (force_constants + force_constants.T)
    gf, g_matrix, force_constants, block_labels = _solve_internal_gf(
        force_constants,
        g_matrix,
        gic_irreps=b_matrix.irreps,
        block_by_irrep=block_by_irrep,
    )
    g_eval, g_vec = eigh_arrays(0.5 * (g_matrix + g_matrix.T))
    g_inv_half = (g_vec * (1.0 / np.sqrt(np.clip(g_eval, 1.0e-14, None)))) @ g_vec.T
    modes_internal = g_inv_half @ gf.normal_modes
    ped = _ped(force_constants, modes_internal, gf.eigenvalues)
    labels = _gic_display_labels(definition)
    primitive_labels = tuple(primitive_label(primitive) for primitive in definition.primitives)
    large_context = _large_amplitude_topology_context_from_xyzin(
        Path(xyzin_path),
        atomic_numbers=hessian_geometry_input.atomic_numbers,
        coordinates_angstrom=coords_for_b,
    )
    assignments = tuple(
        classify_gic_anharmonic_model(
            b_matrix.coordinate_names[idx] if idx < len(b_matrix.coordinate_names) else f"GIC{idx + 1:03d}",
            label,
            topology_context=large_context,
        )
        for idx, label in enumerate(labels)
    )
    u_matrix = _u_matrix_from_definition(definition)
    large_amplitude = large_amplitude_analysis_from_gf_matrices(
        force_constants=force_constants,
        g_matrix=g_matrix,
        frequencies_cm=gf.frequencies_cm,
        ped=ped,
        gic_labels=labels,
        gic_names=b_matrix.coordinate_names,
        gic_irreps=b_matrix.irreps,
        frequency_cutoff_cm=large_amplitude_frequency_cutoff_cm,
        topology_context=large_context,
        block_frequency_model=large_amplitude_frequency_model,
        primitive_labels=primitive_labels,
        u_matrix=u_matrix,
    )
    scaling = getattr(getattr(acquired_force_field, "scaling", None), "factors", None)
    matrix_model = "ACQUIRED_L1"
    if block_labels:
        matrix_model = f"{matrix_model}+IRREP_BLOCKS"
    return InternalGFResult(
        frequencies_cm=gf.frequencies_cm,
        force_constants=force_constants,
        g_matrix=g_matrix,
        b_matrix=bq,
        u_matrix=u_matrix,
        modes_internal=modes_internal,
        ped=PEDTable(ped, labels),
        primitive_labels=primitive_labels,
        gic_labels=labels,
        gic_names=b_matrix.coordinate_names,
        gic_irreps=b_matrix.irreps,
        gic_families=tuple(item.family for item in assignments),
        gic_anharmonic_classes=tuple(item.anharmonic_class for item in assignments),
        gic_zeroth_order_models=tuple(item.zeroth_order_model for item in assignments),
        gic_cross_coupling_policies=tuple(item.cross_coupling_policy for item in assignments),
        gic_anharmonic_reasons=tuple(item.reason for item in assignments),
        point_group=definition.point_group,
        symmetrized_gics=definition.symmetrize,
        scaling_factors=None if scaling is None else np.asarray(scaling, dtype=float),
        coordinate_source=coordinate_source,
        matrix_model=matrix_model,
        block_labels=block_labels,
        force_threshold=force_threshold,
        hessian_correction="NONE",
        large_amplitude=large_amplitude,
    )


def gf_from_cartesian_hessian_and_matrix_gics(
    cartesian_hessian: np.ndarray,
    coordinates_bohr: np.ndarray,
    atomic_numbers: np.ndarray,
    masses_amu: np.ndarray,
    *,
    symmetrize: bool = True,
) -> InternalGFResult:
    """Run Wilson GF from a Cartesian Hessian and freshly generated MATRIX GICs.

    Fresh SONIC contracts are symmetry adapted by default.  This is required
    for complete non-Abelian labels (including the partner rows of degenerate
    irreps) and avoids rank defects that can be hidden by an unsymmetrized
    reduction.  ``symmetrize=False`` remains available for explicit native-GIC
    diagnostics and legacy comparisons.
    """
    with tempfile.TemporaryDirectory(prefix="matrix-gf-") as tmp:
        tmpdir = Path(tmp)
        xyz = tmpdir / "geometry.xyz"
        xyzin = tmpdir / "geometry.xyzin"
        _write_xyz_from_hessian_geometry(xyz, atomic_numbers, coordinates_bohr)
        preprocess_to_enriched_xyz(xyz, xyzin)
        write_validation_section(xyzin)
        write_gicforge_build_sections(xyzin, symmetrize=symmetrize)
        result = gf_from_hessian_input_and_xyzin(
            HessianInput(
                atomic_numbers=np.asarray(atomic_numbers, dtype=int),
                cartesian_coordinates_bohr=np.asarray(coordinates_bohr, dtype=float),
                masses_amu=np.asarray(masses_amu, dtype=float),
                cartesian_hessian=np.asarray(cartesian_hessian, dtype=float),
                harmonic_frequencies_cm=np.array((), dtype=float),
                source="generated-matrix-gics",
            ),
            xyzin,
        )
    return InternalGFResult(
        frequencies_cm=result.frequencies_cm,
        force_constants=result.force_constants,
        g_matrix=result.g_matrix,
        b_matrix=result.b_matrix,
        u_matrix=result.u_matrix,
        modes_internal=result.modes_internal,
        ped=result.ped,
        primitive_labels=result.primitive_labels,
        gic_labels=result.gic_labels,
        gic_names=result.gic_names,
        gic_irreps=result.gic_irreps,
        gic_families=result.gic_families,
        gic_anharmonic_classes=result.gic_anharmonic_classes,
        gic_zeroth_order_models=result.gic_zeroth_order_models,
        gic_cross_coupling_policies=result.gic_cross_coupling_policies,
        gic_anharmonic_reasons=result.gic_anharmonic_reasons,
        point_group=result.point_group,
        symmetrized_gics=result.symmetrized_gics,
        scaling_factors=result.scaling_factors,
        coordinate_source="generated-matrix-gics",
        large_amplitude=result.large_amplitude,
    )


def gf_from_hessian_input_with_matrix_gics(
    input_data: HessianInput,
    *,
    symmetrize: bool = True,
) -> InternalGFResult:
    """Run TRINITY harmonic analysis from canonical Hessian input and generated SONIC coordinates."""
    input_data.validate()
    return gf_from_cartesian_hessian_and_matrix_gics(
        input_data.cartesian_hessian,
        input_data.cartesian_coordinates_bohr,
        input_data.atomic_numbers,
        input_data.masses_amu,
        symmetrize=symmetrize,
    )


def gf_from_gaussian_fchk_with_matrix_gics(
    path: Path,
    *,
    symmetrize: bool = True,
) -> InternalGFResult:
    """Gaussian adapter: read FCHK, then run the diagnostic MATRIX GF path."""
    from matrix_gaussian import hessian_input_from_gaussian_fchk

    return gf_from_hessian_input_with_matrix_gics(
        hessian_input_from_gaussian_fchk(path),
        symmetrize=symmetrize,
    )


gf_from_cartesian_hessian_and_oracle_gics = gf_from_cartesian_hessian_and_matrix_gics
gf_from_hessian_input_with_oracle_gics = gf_from_hessian_input_with_matrix_gics
gf_from_gaussian_fchk_with_oracle_gics = gf_from_gaussian_fchk_with_matrix_gics


def _large_amplitude_topology_context_from_xyzin(
    path: Path,
    *,
    atomic_numbers: np.ndarray,
    coordinates_angstrom: np.ndarray,
):
    lines = read_sectioned_lines(Path(path))
    topology_bonds = topology_bonds_from_xyzin(Path(path))
    topology_bond_orders = topology_bond_orders_from_lines(
        lines,
        natoms=int(len(atomic_numbers)),
    )
    topology_bond_components = topology_bond_order_components_from_lines(
        lines,
        natoms=int(len(atomic_numbers)),
    )
    if topology_bonds:
        return LargeAmplitudeTopologyContext(
            atomic_numbers=tuple(int(value) for value in np.asarray(atomic_numbers).reshape(-1)),
            bonds=topology_bonds,
            bond_orders=topology_bond_orders,
            bond_order_components=topology_bond_components,
            synthon_signatures=_synthon_signatures_from_lines(lines),
            synthon_zeff=_synthon_zeff_from_lines(lines),
            coordinates_angstrom=tuple(
                tuple(float(value) for value in row)
                for row in np.asarray(coordinates_angstrom, dtype=float)
            ),
            bond_order_source=_topology_bond_order_source(lines),
        )
    try:
        from matrix_chem.link import gaussian_topology_overrides_from_xyzin

        gaussian = gaussian_topology_overrides_from_xyzin(Path(path))
    except Exception:
        gaussian = {
            "bond_orders": {},
            "bond_order_source": "ORACLE Pauling estimate",
        }
    return large_amplitude_topology_context_from_arrays(
        atomic_numbers=atomic_numbers,
        coordinates_angstrom=coordinates_angstrom,
        bond_order_overrides=topology_bond_orders or gaussian.get("bond_orders") or None,
        bond_order_source=str(
            gaussian.get("bond_order_source") or "ORACLE Pauling estimate"
        ),
    )


def _synthon_signatures_from_lines(lines: list[str]) -> dict[int, tuple[object, ...]]:
    synthons = section_content(lines, "SYNTHONS")
    signatures: dict[int, tuple[object, ...]] = {}
    for line in synthons:
        text = line.strip()
        if (
            not text
            or text.startswith("[")
            or text.upper().startswith(
                ("SCHEMA", "ALIAS_", "INDEXING", "CHARGE_", "BOND_", "COLUMNS")
            )
        ):
            continue
        parts = text.split()
        if len(parts) < 8:
            continue
        try:
            atom = int(parts[0])
        except ValueError:
            continue
        signatures[atom] = _parse_synthon_signature_token(parts[-1])
    return signatures


def _synthon_zeff_from_lines(lines: list[str]) -> dict[int, float]:
    synthons = section_content(lines, "SYNTHONS")
    zeff: dict[int, float] = {}
    for line in synthons:
        text = line.strip()
        if not text or text.startswith("[") or text.upper().startswith(("SCHEMA", "ALIAS_", "INDEXING", "CHARGE_", "BOND_", "COLUMNS")):
            continue
        parts = text.split()
        if len(parts) < 3:
            continue
        try:
            atom = int(parts[0])
            zeff[atom] = float(parts[2])
        except ValueError:
            continue
    return zeff


def _parse_synthon_signature_token(token: str) -> tuple[object, ...]:
    values: list[object] = []
    for part in token.split(","):
        text = part.strip()
        if not text:
            continue
        try:
            values.append(int(text))
            continue
        except ValueError:
            pass
        try:
            values.append(float(text))
            continue
        except ValueError:
            values.append(text)
    return tuple(values)


def _topology_bond_order_source(lines: list[str]) -> str:
    for line in section_content(lines, "TOPOLOGY"):
        text = line.strip()
        if text.startswith("BOND_ORDER_SOURCE "):
            return text.split(" ", 1)[1].strip()
    return "Frozen #TOPOLOGY"


def topology_bonds_from_xyzin(path: Path) -> tuple[tuple[int, int], ...]:
    """Read one-based topology bonds from a MATRIX xyzin file."""
    lines = read_sectioned_lines(Path(path))
    topology = section_content(lines, "TOPOLOGY")
    bond_lines = _subsection(topology, "BONDS")
    bonds: list[tuple[int, int]] = []
    for line in bond_lines:
        text = line.strip()
        if not text or text.upper() == "NONE":
            continue
        parts = text.split()
        if len(parts) < 2:
            continue
        left, right = int(parts[0]), int(parts[1])
        bonds.append(tuple(sorted((left, right))))
    return tuple(dict.fromkeys(bonds))


def local_force_constant_mask(
    definition: GICDefinition,
    topology_bonds: tuple[tuple[int, int], ...],
) -> np.ndarray:
    """Return a local force-field mask in the frozen GIC basis."""
    primitive_by_id = {primitive.identifier: primitive for primitive in definition.primitives}
    sources = tuple(_gic_source_primitives(gic, primitive_by_id) for gic in definition.gics)
    graph = _bond_graph(topology_bonds)
    ncoord = len(definition.gics)
    mask = np.eye(ncoord, dtype=bool)
    for i in range(ncoord):
        for j in range(i + 1, ncoord):
            allowed = any(
                _local_primitive_pair_allowed(left, right, graph)
                for left in sources[i]
                for right in sources[j]
            )
            mask[i, j] = allowed
            mask[j, i] = allowed
    return mask


def _solve_internal_gf(
    force_constants: np.ndarray,
    g_matrix: np.ndarray,
    *,
    gic_irreps: tuple[str, ...],
    block_by_irrep: bool,
):
    if not block_by_irrep:
        return (
            solve_wilson_gf(force_constants, g_matrix, scale_to_cm=True),
            g_matrix,
            force_constants,
            (),
        )
    blocks = _irrep_blocks(gic_irreps)
    if len(blocks) <= 1:
        return (
            solve_wilson_gf(force_constants, g_matrix, scale_to_cm=True),
            g_matrix,
            force_constants,
            (),
        )
    # The acquired Hessian and the Wilson metric are built from finite
    # precision geometries and displaced points.  The symmetry contract of a
    # block GF calculation is therefore an explicit projection: retain every
    # diagonal irrep block and set all inter-irrep couplings to zero in both F
    # and G before solving.  The non-totally-symmetric displaced points have
    # already contributed to their own diagonal blocks; this step removes
    # only couplings forbidden by the requested SONIC symmetry.
    same_block = np.zeros_like(force_constants, dtype=bool)
    for _irrep, indices in blocks:
        index = np.asarray(indices, dtype=int)
        same_block[np.ix_(index, index)] = True
    force_constants = np.where(same_block, force_constants, 0.0)
    g_matrix = np.where(same_block, g_matrix, 0.0)

    f_block = np.zeros_like(force_constants, dtype=float)
    g_block = np.zeros_like(g_matrix, dtype=float)
    eigenvalues: list[float] = []
    frequencies: list[float] = []
    modes: list[np.ndarray] = []
    block_labels: list[str] = []
    ncoord = force_constants.shape[0]
    for irrep, indices in blocks:
        index = np.asarray(indices, dtype=int)
        f_sub = force_constants[np.ix_(index, index)]
        g_sub = g_matrix[np.ix_(index, index)]
        gf_sub = solve_wilson_gf(f_sub, g_sub, scale_to_cm=True)
        f_block[np.ix_(index, index)] = f_sub
        g_block[np.ix_(index, index)] = g_sub
        for col in range(gf_sub.normal_modes.shape[1]):
            mode = np.zeros(ncoord, dtype=float)
            mode[index] = gf_sub.normal_modes[:, col]
            modes.append(mode)
            eigenvalues.append(float(gf_sub.eigenvalues[col]))
            frequencies.append(float(gf_sub.frequencies_cm[col]))
            block_labels.append(irrep)
    normal_modes = np.column_stack(modes) if modes else np.zeros((ncoord, 0), dtype=float)
    from .harmonic import GFResult

    return (
        GFResult(
            eigenvalues=np.asarray(eigenvalues, dtype=float),
            frequencies_cm=np.asarray(frequencies, dtype=float),
            normal_modes=normal_modes,
        ),
        g_block,
        f_block,
        tuple(block_labels),
    )




def _irrep_blocks(gic_irreps: tuple[str, ...]) -> tuple[tuple[str, tuple[int, ...]], ...]:
    blocks: dict[str, list[int]] = {}
    for idx, irrep in enumerate(gic_irreps):
        label = (irrep or "UNASSIGNED").strip()
        if label.upper() in {"", "UNK", "UNKNOWN", "UNASSIGNED"}:
            return ()
        blocks.setdefault(label, []).append(idx)
    return tuple((label, tuple(indices)) for label, indices in blocks.items())


def _gic_source_primitives(
    gic: FrozenGIC,
    primitive_by_id: dict[str, GICPrimitive],
) -> tuple[GICPrimitive, ...]:
    coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
    sources: list[GICPrimitive] = []
    for primitive_id, coefficient in coefficients:
        if abs(float(coefficient)) <= 1.0e-14:
            continue
        primitive = primitive_by_id.get(primitive_id)
        if primitive is not None:
            sources.append(primitive)
    return tuple(sources)


def _local_primitive_pair_allowed(
    left: GICPrimitive,
    right: GICPrimitive,
    graph: dict[int, set[int]],
) -> bool:
    left_kind = _primitive_local_kind(left)
    right_kind = _primitive_local_kind(right)
    if left_kind is None or right_kind is None:
        return False
    kinds = {left_kind, right_kind}
    if kinds == {"STRETCH"}:
        return bool(set(left.atoms) & set(right.atoms))
    if kinds == {"STRETCH", "BEND"}:
        stretch = left if left_kind == "STRETCH" else right
        bend = right if left_kind == "STRETCH" else left
        return set(stretch.atoms).issubset(set(bend.atoms))
    if kinds == {"BEND"}:
        return left.atoms[1] == right.atoms[1] or len(set(left.atoms) & set(right.atoms)) >= 2
    if kinds == {"STRETCH", "TORSION"}:
        stretch = left if left_kind == "STRETCH" else right
        torsion = right if left_kind == "STRETCH" else left
        return _sets_are_vicinal(set(stretch.atoms), set(torsion.atoms), graph)
    if kinds == {"BEND", "TORSION"}:
        bend = left if left_kind == "BEND" else right
        torsion = right if left_kind == "BEND" else left
        return len(set(bend.atoms) & set(torsion.atoms)) >= 2 or _sets_are_vicinal(
            set(bend.atoms),
            set(torsion.atoms[1:3] if len(torsion.atoms) == 4 else torsion.atoms),
            graph,
        )
    return False


def _primitive_local_kind(primitive: GICPrimitive) -> str | None:
    if primitive.function == "R":
        return "STRETCH"
    if primitive.function in {"A", "L"}:
        return "BEND"
    if primitive.function in {"D", "RPCK"}:
        return "TORSION"
    return None


def _sets_are_vicinal(left: set[int], right: set[int], graph: dict[int, set[int]]) -> bool:
    if left & right:
        return True
    return any(atom in graph.get(other, set()) for atom in left for other in right)


def _bond_graph(bonds: tuple[tuple[int, int], ...]) -> dict[int, set[int]]:
    graph: dict[int, set[int]] = {}
    for left, right in bonds:
        graph.setdefault(left, set()).add(right)
        graph.setdefault(right, set()).add(left)
    return graph


def _subsection(section_lines: list[str], name: str) -> list[str]:
    header = f"[{name.upper()}]"
    start = None
    for idx, line in enumerate(section_lines):
        if line.strip().upper() == header:
            start = idx + 1
            break
    if start is None:
        return []
    end = len(section_lines)
    for idx in range(start, len(section_lines)):
        text = section_lines[idx].strip()
        if text.startswith("[") and text.endswith("]"):
            end = idx
            break
    return list(section_lines[start:end])


def _write_xyz_from_hessian_geometry(
    path: Path,
    atomic_numbers: np.ndarray,
    coordinates_bohr: np.ndarray,
) -> None:
    atoms = tuple(atomic_symbol(int(number)) for number in atomic_numbers)
    coords = np.asarray(coordinates_bohr, dtype=float) * BOHR_TO_ANGSTROM
    lines = [str(len(atoms)), "MATRIX GF generated geometry"]
    for atom, (x, y, z) in zip(atoms, coords):
        lines.append(f"{atom:2s} {x:16.10f} {y:16.10f} {z:16.10f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _gic_display_labels(definition: GICDefinition) -> tuple[str, ...]:
    labels: list[str] = []
    for gic in definition.gics:
        if gic.gaussian_expression and gic.gaussian_expression != "NONE":
            labels.append(f"{gic.identifier} {gic.gaussian_expression}")
        else:
            labels.append(gic.identifier)
    return tuple(labels)


def _u_matrix_from_definition(definition: GICDefinition) -> np.ndarray:
    primitive_index = {
        primitive.identifier: index for index, primitive in enumerate(definition.primitives)
    }
    matrix = np.zeros((len(definition.primitives), len(definition.gics)), dtype=float)
    for col, gic in enumerate(definition.gics):
        coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
        for primitive_id, coefficient in coefficients:
            row = primitive_index.get(primitive_id)
            if row is not None:
                matrix[row, col] += float(coefficient)
    return matrix
