"""Symmetry-reduced SONIC Hessians acquired from electronic energies."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from matrix_chem import atomic_mass, read_xyzin_geometry, write_xyz
from matrix_chem.topology.elements import atomic_number
from matrix_gf import gf_from_acquired_force_constants_and_xyzin, solve_wilson_gf
from matrix_gf import cartesian_normal_modes_from_sonic_hessian
from matrix_qm.hessians import HessianInput
from matrix_smith import (
    build_gic_b_matrix,
    is_total_symmetric_irrep,
    read_gic_definition_from_xyzin,
)

from .optimizer import (
    GeometryEvaluationService,
    OptimizerEvaluation,
    OptimizerSettings,
    coordinate_model_from_xyzin,
)
from .scan import BOHR_TO_ANGSTROM, QMScanBackend, read_point_result


SONIC_HESSIAN_PLAN_SCHEMA = "matrix.link.sonic_hessian_plan.v1"
SONIC_HESSIAN_RESULT_SCHEMA = "matrix.link.sonic_hessian_result.v1"
SONIC_SYMMETRY_CURVATURE_SCHEMA = "matrix.link.symmetry_curvature_check.v1"
SONIC_SYMMETRY_STABILITY_PROTOCOL = "matrix.link.symmetry_stability.standard.v1"
DEFAULT_SYMMETRY_BREAKING_FREQUENCY_CUTOFF_CM = -10.0
DEFAULT_SYMMETRY_BREAKING_MAX_DISPLACEMENT_ANGSTROM = 0.05

_LENGTH_FAMILIES = {
    "STRETCH",
    "HBOND_DISTANCE",
    "SPECIAL_HBOND_DISTANCE",
    "FRAGMENT_CENTER_DISTANCE",
    "FRAGMENT_CENTER_ATOM_DISTANCE",
    "FRAG_CENTER_ATOM_DISTANCE",
    "CENTER_ATOM_DISTANCE",
    "FRAG_TRANSLATION",
}


@dataclass(frozen=True)
class SonicHessianCoordinate:
    index: int
    label: str
    family: str
    irrep: str
    unit: str
    step: float


@dataclass(frozen=True)
class SonicHessianPoint:
    key: str
    role: str
    displacement: tuple[float, ...]
    active_coordinates: tuple[int, ...]


@dataclass(frozen=True)
class SonicHessianPlan:
    point_group: str
    coordinates: tuple[SonicHessianCoordinate, ...]
    symmetry_blocks: tuple[tuple[str, tuple[int, ...]], ...]
    points: tuple[SonicHessianPoint, ...]
    property_source: str = "energy"
    displacement_basis: str = "sonic"
    mixed_stencil: str = "one-sided"
    schema: str = SONIC_HESSIAN_PLAN_SCHEMA

    @property
    def coordinate_count(self) -> int:
        return len(self.coordinates)

    @property
    def mixed_coupling_count(self) -> int:
        return sum(point.role == "mixed-plus-plus" for point in self.points)

    @property
    def energy_point_count(self) -> int:
        return len(self.points)

    def to_json(self) -> dict[str, object]:
        payload = asdict(self)
        payload.update(
            {
                "coordinate_count": self.coordinate_count,
                "mixed_coupling_count": self.mixed_coupling_count,
                "energy_point_count": self.energy_point_count,
                "cross_irrep_policy": "zero by symmetry",
            }
        )
        return payload


@dataclass(frozen=True)
class SonicHessianResult:
    plan: SonicHessianPlan
    force_constants: np.ndarray
    energies_hartree: Mapping[str, float]
    frequencies_cm: np.ndarray
    g_matrix: np.ndarray
    state_audit: Mapping[str, Mapping[str, object]]
    result_path: Path


def parallel_sycart_hessian(
    calculator,
    atoms,
    coordinates_angstrom: np.ndarray,
    *,
    basis_cartesian_from_sycart: np.ndarray,
    irreps: tuple[str, ...],
    sign_symmetric: tuple[int, ...] = (),
    step_bohr: float = 1.0e-3,
    workers: int = 1,
):
    """Freeze, displace and assemble a symmetry-adapted SYCART Hessian."""
    from matrix_gfn2 import build_sycart_hessian_plan, sycart_numerical_hessian

    plan = build_sycart_hessian_plan(
        atoms,
        coordinates_angstrom,
        basis_cartesian_from_sycart=basis_cartesian_from_sycart,
        irreps=irreps,
        sign_symmetric=sign_symmetric,
        step_bohr=step_bohr,
    )
    result = sycart_numerical_hessian(calculator, plan, workers=workers)
    from matrix_core.owner_evidence import emit_owner_validation
    matrix = result.hessian_hartree_per_coordinate2
    emit_owner_validation('LINK', 'SYCART_HESSIAN_ASSEMBLY', passed=bool(np.all(np.isfinite(matrix))),
        metrics={'symmetry_error': float(np.max(np.abs(matrix-matrix.T))), 'dimension': float(matrix.shape[0])},
        units={'symmetry_error': 'hartree/bohr^2', 'dimension': 'dimensionless'},
        context={'scope': 'Frozen SYCART assembly; does not certify QM derivative accuracy'})
    return result


def _positive_symmetry_curvature_jobs(
    dimension: int,
    selected_indices: tuple[int, ...],
    steps: Mapping[int, float],
) -> tuple[tuple[str, np.ndarray], ...]:
    """Return the canonical plus-only gradient grid for a symmetry check."""

    jobs = []
    for index in selected_indices:
        displacement = np.zeros(int(dimension), dtype=float)
        displacement[index] = float(steps[index])
        jobs.append((f"q{index + 1:03d}-plus", displacement))
    return tuple(jobs)


def acquire_sonic_symmetry_curvature(
    xyzin_path: Path | str,
    *,
    run_dir: Path | str,
    backend: QMScanBackend,
    irreps: tuple[str, ...] = (),
    workers: int = 1,
    stretch_step_angstrom: float = 0.01,
    angular_step_radian: float = 0.01,
    center_point_result: Path | str | None = None,
    negative_frequency_cutoff_cm: float = DEFAULT_SYMMETRY_BREAKING_FREQUENCY_CUTOFF_CM,
    symmetry_breaking_max_displacement_angstrom: float = (
        DEFAULT_SYMMETRY_BREAKING_MAX_DISPLACEMENT_ANGSTROM
    ),
    timeout: float | None = None,
) -> dict[str, object]:
    """Check selected SONIC symmetry blocks using the standard MATRIX protocol.

    With no explicit ``irreps``, every non-totally-symmetric block is checked.
    Only the positive displacement is evaluated for every selected coordinate;
    the negative-side response is generated from the parent symmetry, directly
    by a group operation for one-dimensional irreps and in the linearized irrep
    tangent space otherwise. Stability is decided from the Wilson GF eigenproblem
    rather than the unit-dependent raw internal-coordinate Hessian.
    """

    if workers < 1:
        raise ValueError("symmetry-curvature workers must be positive")
    source = Path(xyzin_path).expanduser().resolve()
    target = Path(run_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    definition = read_gic_definition_from_xyzin(source)
    requested_irreps = tuple(
        dict.fromkeys(str(label).strip() for label in irreps if str(label).strip())
    )
    selection_policy = "explicit_irreps"
    if not requested_irreps:
        requested_irreps = tuple(
            dict.fromkeys(
                str(gic.irrep)
                for gic in definition.gics
                if not is_total_symmetric_irrep(definition.point_group, str(gic.irrep))
            )
        )
        selection_policy = "all_non_totally_symmetric_irreps"
    if not requested_irreps:
        raise ValueError("the SONIC chart has no non-totally-symmetric block to check")
    if float(negative_frequency_cutoff_cm) > 0.0:
        raise ValueError("negative-frequency cutoff must be non-positive")
    if float(symmetry_breaking_max_displacement_angstrom) <= 0.0:
        raise ValueError("symmetry-breaking Cartesian displacement must be positive")
    selected_indices = tuple(
        index for index, gic in enumerate(definition.gics) if str(gic.irrep) in requested_irreps
    )
    if not selected_indices:
        raise ValueError(
            "selected symmetry species are absent from the SONIC chart: "
            + ", ".join(requested_irreps)
        )
    labels = tuple(str(gic.name or gic.identifier) for gic in definition.gics)
    model = coordinate_model_from_xyzin(source, kind="sonic", coordinates=labels)
    geometry = read_xyzin_geometry(source)
    service = GeometryEvaluationService(
        xyzin_path=source,
        run_dir=target / "evaluation-service",
        coordinate_model=model,
        backend=backend,
        timeout=timeout,
        settings=OptimizerSettings(
            freeze_inactive_sonic=False,
            symmetrize_analytic_gradients=False,
            resume=True,
            coordinate_parallel_workers=max(1, int(workers)),
        ),
    )
    zero = np.zeros(len(definition.gics), dtype=float)
    service.initialize_coordinate_projector(zero, geometry.coordinates_angstrom)
    steps = {
        index: (
            float(stretch_step_angstrom)
            if sonic_coordinate_unit(definition.gics[index].family) == "angstrom"
            else float(angular_step_radian)
        )
        for index in selected_indices
    }

    def evaluate(key: str, displacement: np.ndarray):
        return key, service.evaluate(
            displacement,
            tag=f"symmetry-curvature-{key}",
            requested_properties=("energy", "gradient"),
        )

    if center_point_result is None:
        center_key, center = evaluate("center", zero)
        center_was_reused = False
    else:
        center_key = "center"
        center = OptimizerEvaluation(
            q=zero.copy(),
            coordinates_angstrom=np.asarray(geometry.coordinates_angstrom, dtype=float),
            result=read_point_result(Path(center_point_result).expanduser().resolve()),
        )
        center_was_reused = True
    service.accept_electronic_state(center)
    from matrix_smith import irrep_characters_for_operations, irrep_dimension

    symmetry = service._assigned_cartesian_symmetry
    operation_by_irrep = {}
    if symmetry is not None:
        character_rows = irrep_characters_for_operations(
            [operation.label for operation in symmetry.operations],
            point_group=symmetry.point_group,
            operation_matrices=[operation.rotation for operation in symmetry.operations],
        )
        for irrep, characters in character_rows:
            if irrep_dimension(irrep) != 1:
                continue
            operation_by_irrep[str(irrep)] = next(
                (
                    operation
                    for operation, character in zip(
                        symmetry.operations, characters, strict=True
                    )
                    if float(character) < -0.5
                ),
                None,
            )
    jobs = _positive_symmetry_curvature_jobs(
        len(definition.gics), selected_indices, steps
    )
    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        displaced = tuple(executor.map(lambda item: evaluate(*item), jobs))
    evaluations = {center_key: center, **dict(displaced)}

    def internal_gradient(coordinates, gradient) -> np.ndarray:
        if gradient is None:
            raise RuntimeError("backend returned no Cartesian gradient")
        b_matrix = np.asarray(
            build_gic_b_matrix(
                definition,
                coordinates_angstrom=np.asarray(coordinates, dtype=float),
            ).rows,
            dtype=float,
        )
        internal, *_ = np.linalg.lstsq(
            b_matrix.T,
            np.asarray(gradient, dtype=float).reshape(-1) / BOHR_TO_ANGSTROM,
            rcond=1.0e-10,
        )
        return np.asarray(internal, dtype=float)

    internal_gradients = {
        key: internal_gradient(
            evaluation.coordinates_angstrom,
            evaluation.result.gradient_hartree_per_bohr,
        )
        for key, evaluation in evaluations.items()
    }
    block = np.zeros((len(selected_indices), len(selected_indices)), dtype=float)
    for column, index in enumerate(selected_indices):
        minus_key = f"q{index + 1:03d}-minus"
        plus = internal_gradients[f"q{index + 1:03d}-plus"]
        if minus_key in internal_gradients:
            minus = internal_gradients[minus_key]
        elif operation_by_irrep.get(str(definition.gics[index].irrep)) is not None:
            operation = operation_by_irrep[str(definition.gics[index].irrep)]
            plus_gradient = np.asarray(
                evaluations[f"q{index + 1:03d}-plus"].result.gradient_hartree_per_bohr,
                dtype=float,
            ).reshape((-1, 3))
            mapping = np.asarray(
                [int(atom) - 1 for atom in operation.permutation], dtype=int
            )
            transformed_gradient = plus_gradient[np.argsort(mapping)] @ np.asarray(
                operation.rotation, dtype=float
            )
            minus_displacement = zero.copy()
            minus_displacement[index] = -steps[index]
            minus_coordinates = service.coordinates_from_q(minus_displacement)
            minus = internal_gradient(minus_coordinates, transformed_gradient.reshape(-1))
        else:
            # In multidimensional non-total irreps a single group operation
            # need not map an individual chosen component onto its negative.
            # The Hessian is nevertheless the symmetry-linear response at the
            # parent geometry: its missing negative-side gradient is fixed by
            # the odd tangent response around the symmetry-forced central
            # gradient.  No -delta backend evaluation is required.
            minus = 2.0 * internal_gradients[center_key] - plus
        block[:, column] = (plus - minus)[list(selected_indices)] / (2.0 * steps[index])
    block = 0.5 * (block + block.T)
    raw_eigenvalues, raw_eigenvectors = np.linalg.eigh(block)
    b_matrix_at_center = np.asarray(
        build_gic_b_matrix(
            definition,
            coordinates_angstrom=np.asarray(geometry.coordinates_angstrom, dtype=float),
        ).rows,
        dtype=float,
    )
    masses = np.asarray(
        [atomic_mass(atomic_number(atom)) for atom in geometry.atoms], dtype=float
    )
    mass_inverse = np.diag(np.repeat(1.0 / masses, 3))
    full_g_matrix = BOHR_TO_ANGSTROM**2 * (
        b_matrix_at_center @ mass_inverse @ b_matrix_at_center.T
    )
    g_block = full_g_matrix[np.ix_(selected_indices, selected_indices)]
    gf = solve_wilson_gf(block, g_block, scale_to_cm=True)
    frequencies_cm = np.asarray(gf.frequencies_cm, dtype=float)
    negative = np.flatnonzero(
        frequencies_cm < float(negative_frequency_cutoff_cm)
    )
    g_values, g_vectors = np.linalg.eigh(0.5 * (g_block + g_block.T))
    g_inverse_half = (g_vectors * (1.0 / np.sqrt(g_values))) @ g_vectors.T
    physical_modes = g_inverse_half @ np.asarray(gf.normal_modes, dtype=float)
    symmetry_breaking_seed: Path | None = None
    if negative.size:
        mode = np.zeros(len(definition.gics), dtype=float)
        mode[list(selected_indices)] = physical_modes[:, int(negative[0])]
        probe_scale = min(steps.values())
        probe = service.coordinates_from_q(probe_scale * mode)
        displacement = np.asarray(probe, dtype=float) - np.asarray(
            geometry.coordinates_angstrom, dtype=float
        )
        maximum = float(np.max(np.linalg.norm(displacement, axis=1)))
        if maximum <= 1.0e-12:
            raise RuntimeError("negative mode produced no Cartesian symmetry breaking")
        scale = probe_scale * float(symmetry_breaking_max_displacement_angstrom) / maximum
        lowered = service.coordinates_from_q(scale * mode)
        symmetry_breaking_seed = target / "symmetry-lowered-seed.xyz"
        write_xyz(
            symmetry_breaking_seed,
            geometry.atoms,
            lowered,
            comment=(
                f"MATRIX symmetry lowering from {definition.point_group}; "
                f"mode {int(negative[0]) + 1}; rebuild ORACLE and SONIC contracts"
            ),
        )
    state_audit = {
        key: {
            field: value
            for field, value in evaluation.result.execution.items()
            if field.startswith("state_") or field == "selected_electronic_state"
        }
        for key, evaluation in evaluations.items()
    }
    payload = {
        "schema": SONIC_SYMMETRY_CURVATURE_SCHEMA,
        "protocol": SONIC_SYMMETRY_STABILITY_PROTOCOL,
        "point_group": str(definition.point_group),
        "selected_irreps": list(requested_irreps),
        "selection_policy": selection_policy,
        "coordinate_indices_zero_based": list(selected_indices),
        "coordinate_labels": [labels[index] for index in selected_indices],
        "coordinate_families": [str(definition.gics[index].family) for index in selected_indices],
        "steps": [steps[index] for index in selected_indices],
        "force_constant_block": block.tolist(),
        "g_matrix_block": g_block.tolist(),
        "eigenvalues": raw_eigenvalues.tolist(),
        "eigenvectors_columns": raw_eigenvectors.tolist(),
        "raw_force_constant_eigenvalues": raw_eigenvalues.tolist(),
        "raw_force_constant_eigenvectors_columns": raw_eigenvectors.tolist(),
        "gf_eigenvalues": np.asarray(gf.eigenvalues, dtype=float).tolist(),
        "frequencies_cm": frequencies_cm.tolist(),
        "negative_frequency_cutoff_cm": float(negative_frequency_cutoff_cm),
        "negative_eigenvalue_count": int(negative.size),
        "minimum_eigenvalue": float(gf.eigenvalues[0]),
        "minimum_frequency_cm": float(frequencies_cm[0]),
        "minimum_mode_selected_coordinates": physical_modes[:, 0].tolist(),
        "symmetry_stability": (
            "lower_symmetry_required" if negative.size else "stable_in_selected_subspace"
        ),
        "symmetry_breaking_seed": (
            None if symmetry_breaking_seed is None else str(symmetry_breaking_seed)
        ),
        "next_protocol_stage": (
            "KEYMAKER_ORACLE_SMITH_REBUILD"
            if symmetry_breaking_seed is not None
            else "KEYMAKER_EXPLOITATION"
        ),
        "energy_gradient_points": len(evaluations),
        "new_gradient_evaluations": len(displaced),
        "center_gradient_reused": center_was_reused,
        "symmetry_generated_negative_displacements": sum(
            1 for _index in selected_indices
        ),
        "negative_displacement_policy": "generated_from_parent_symmetry_no_backend_run",
        "exact_operation_generated_negative_displacements": sum(
            operation_by_irrep.get(str(definition.gics[index].irrep)) is not None
            for index in selected_indices
        ),
        "tangent_representation_generated_negative_displacements": sum(
            operation_by_irrep.get(str(definition.gics[index].irrep)) is None
            for index in selected_indices
        ),
        "state_audit": state_audit,
    }
    result_path = target / "symmetry-curvature-check.json"
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {**payload, "result_path": str(result_path)}


def sonic_coordinate_unit(family: str) -> str:
    """Return the finite-difference unit implied by a frozen SONIC family."""

    return "angstrom" if str(family).strip().upper() in _LENGTH_FAMILIES else "radian"


def plan_sonic_hessian(
    xyzin_path: Path | str,
    *,
    stretch_step_angstrom: float = 0.01,
    angular_step_radian: float = 0.01,
    property_source: str = "energy",
    mixed_stencil: str = "one-sided",
    include_mixed: bool = False,
) -> SonicHessianPlan:
    """Build the minimal symmetry-block SONIC energy grid for a Hessian.

    Totally symmetric energy diagonals use the central three-point
    ``(-h_i,0,+h_i)`` derivative.  For every non-totally-symmetric coordinate,
    parent symmetry makes the two displaced energies identical, so only the
    positive point is evaluated and its negative partner is generated exactly.
    Optional mixed derivatives add one ``(+h_i,+h_j)`` point only within the
    same irrep; cross-irrep force constants are exactly zero.
    """

    if stretch_step_angstrom <= 0.0 or angular_step_radian <= 0.0:
        raise ValueError("SONIC Hessian displacement steps must be positive")
    source = str(property_source).strip().casefold().replace("_", "-")
    if source not in {"energy", "cartesian-gradient"}:
        raise ValueError("property_source must be 'energy' or 'cartesian-gradient'")
    stencil = str(mixed_stencil).strip().casefold().replace("_", "-")
    if stencil not in {"one-sided"}:
        raise ValueError("mixed_stencil must be 'one-sided'")

    definition = read_gic_definition_from_xyzin(Path(xyzin_path))
    if definition.rank != len(definition.gics):
        raise ValueError(
            "direct SONIC Hessian acquisition requires a frozen nonredundant GIC contract"
        )
    coordinates = tuple(
        SonicHessianCoordinate(
            index=index,
            label=str(gic.name or gic.identifier),
            family=str(gic.family).strip().upper(),
            irrep=str(gic.irrep).strip(),
            unit=sonic_coordinate_unit(gic.family),
            step=(
                float(stretch_step_angstrom)
                if sonic_coordinate_unit(gic.family) == "angstrom"
                else float(angular_step_radian)
            ),
        )
        for index, gic in enumerate(definition.gics)
    )
    if not coordinates:
        raise ValueError("the frozen SONIC contract contains no coordinates")

    block_map: dict[str, list[int]] = {}
    for coordinate in coordinates:
        block_map.setdefault(coordinate.irrep, []).append(coordinate.index)
    symmetry_blocks = tuple(
        (irrep, tuple(indices)) for irrep, indices in block_map.items()
    )

    ncoord = len(coordinates)
    zero = np.zeros(ncoord, dtype=float)
    points = [
        SonicHessianPoint(
            key="center",
            role="center",
            displacement=tuple(zero),
            active_coordinates=(),
        )
    ]
    for coordinate in coordinates:
        total_symmetric = is_total_symmetric_irrep(
            definition.point_group, coordinate.irrep
        )
        signed_points = (
            ((1.0, "plus"),)
            if source == "energy" and not total_symmetric
            else ((-1.0, "minus"), (1.0, "plus"))
        )
        for sign, suffix in signed_points:
            displacement = zero.copy()
            displacement[coordinate.index] = sign * coordinate.step
            points.append(
                SonicHessianPoint(
                    key=f"q{coordinate.index + 1:03d}-{suffix}",
                    role=f"diagonal-{suffix}",
                    displacement=tuple(displacement),
                    active_coordinates=(coordinate.index,),
                )
            )
    if source == "energy" and include_mixed:
        for _irrep, indices in symmetry_blocks:
            for position, left in enumerate(indices):
                for right in indices[position + 1 :]:
                    displacement = zero.copy()
                    displacement[left] = coordinates[left].step
                    displacement[right] = coordinates[right].step
                    points.append(
                        SonicHessianPoint(
                            key=f"q{left + 1:03d}-q{right + 1:03d}-plus-plus",
                            role="mixed-plus-plus",
                            displacement=tuple(displacement),
                            active_coordinates=(left, right),
                        )
                    )
    return SonicHessianPlan(
        point_group=str(definition.point_group),
        coordinates=coordinates,
        symmetry_blocks=symmetry_blocks,
        points=tuple(points),
        property_source=source,
        mixed_stencil=stencil,
    )


def sonic_hessian_from_energies(
    plan: SonicHessianPlan,
    energies_hartree: Mapping[str, float],
) -> np.ndarray:
    """Reconstruct a symmetry-block Hessian from a completed plan."""

    if plan.property_source != "energy":
        raise ValueError("the SONIC Hessian plan does not acquire energies")

    missing = tuple(point.key for point in plan.points if point.key not in energies_hartree)
    if missing:
        raise ValueError("missing SONIC Hessian energies: " + ", ".join(missing))
    center = float(energies_hartree["center"])
    hessian = np.zeros((plan.coordinate_count, plan.coordinate_count), dtype=float)
    for coordinate in plan.coordinates:
        plus = float(energies_hartree[f"q{coordinate.index + 1:03d}-plus"])
        minus_key = f"q{coordinate.index + 1:03d}-minus"
        # Absence of a negative point is deliberate only for a non-total
        # symmetry coordinate: E(-q)=E(+q) follows from the parent operation
        # that changes the sign of q.
        minus = float(energies_hartree.get(minus_key, plus))
        hessian[coordinate.index, coordinate.index] = (
            plus - 2.0 * center + minus
        ) / coordinate.step**2
    for point in plan.points:
        if point.role != "mixed-plus-plus":
            continue
        left, right = point.active_coordinates
        left_energy = float(energies_hartree[f"q{left + 1:03d}-plus"])
        right_energy = float(energies_hartree[f"q{right + 1:03d}-plus"])
        value = (
            float(energies_hartree[point.key]) - left_energy - right_energy + center
        ) / (plan.coordinates[left].step * plan.coordinates[right].step)
        hessian[left, right] = value
        hessian[right, left] = value
    return hessian


def sonic_hessian_from_cartesian_gradients(
    plan: SonicHessianPlan,
    definition,
    point_coordinates_angstrom: Mapping[str, np.ndarray],
    gradients_hartree_per_bohr: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Differentiate backend Cartesian gradients in the displaced SONIC basis."""

    if plan.property_source != "cartesian-gradient":
        raise ValueError("the SONIC Hessian plan does not acquire Cartesian gradients")
    missing = tuple(
        point.key
        for point in plan.points
        if point.key not in point_coordinates_angstrom
        or point.key not in gradients_hartree_per_bohr
    )
    if missing:
        raise ValueError("missing SONIC Hessian gradients: " + ", ".join(missing))
    internal_gradients: dict[str, np.ndarray] = {}
    for point in plan.points:
        b_matrix = np.asarray(
            build_gic_b_matrix(
                definition,
                coordinates_angstrom=np.asarray(point_coordinates_angstrom[point.key], dtype=float),
            ).rows,
            dtype=float,
        )
        cartesian = np.asarray(
            gradients_hartree_per_bohr[point.key], dtype=float
        ).reshape(-1)
        if cartesian.size != b_matrix.shape[1]:
            raise ValueError(f"Cartesian gradient size is invalid at {point.key}")
        gradient_per_angstrom = cartesian / BOHR_TO_ANGSTROM
        internal, *_ = np.linalg.lstsq(
            b_matrix.T,
            gradient_per_angstrom,
            rcond=1.0e-10,
        )
        internal_gradients[point.key] = np.asarray(internal, dtype=float)
    hessian = np.zeros((plan.coordinate_count, plan.coordinate_count), dtype=float)
    for coordinate in plan.coordinates:
        minus = internal_gradients[f"q{coordinate.index + 1:03d}-minus"]
        plus = internal_gradients[f"q{coordinate.index + 1:03d}-plus"]
        hessian[:, coordinate.index] = (plus - minus) / (2.0 * coordinate.step)
    hessian = 0.5 * (hessian + hessian.T)
    for left in range(plan.coordinate_count):
        for right in range(plan.coordinate_count):
            if plan.coordinates[left].irrep != plan.coordinates[right].irrep:
                hessian[left, right] = 0.0
    return hessian


def acquire_sonic_hessian(
    xyzin_path: Path | str,
    *,
    run_dir: Path | str,
    backend: QMScanBackend,
    workers: int = 1,
    stretch_step_angstrom: float = 0.01,
    angular_step_radian: float = 0.01,
    property_source: str = "energy",
    mixed_stencil: str = "one-sided",
    include_mixed: bool = False,
    timeout: float | None = None,
) -> SonicHessianResult:
    """Execute a planned SONIC Hessian and directly perform the GF analysis."""

    if workers <= 0:
        raise ValueError("SONIC Hessian workers must be positive")
    source = Path(xyzin_path).expanduser().resolve()
    target = Path(run_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    plan = plan_sonic_hessian(
        source,
        stretch_step_angstrom=stretch_step_angstrom,
        angular_step_radian=angular_step_radian,
        property_source=property_source,
        mixed_stencil=mixed_stencil,
        include_mixed=include_mixed,
    )
    (target / "sonic-hessian-plan.json").write_text(
        json.dumps(plan.to_json(), indent=2) + "\n", encoding="utf-8"
    )

    if (
        plan.property_source == "energy"
        and str(backend.name).strip().casefold() in {"gaussian", "g16", "gdv", "junchs-gdv"}
    ):
        backend = replace(
            backend,
            gaussian_native_checkpoint_restart=True,
            restart_reuse_for_displacements=True,
        )

    labels = tuple(coordinate.label for coordinate in plan.coordinates)
    model = coordinate_model_from_xyzin(source, kind="sonic", coordinates=labels)
    geometry = read_xyzin_geometry(source)
    service = GeometryEvaluationService(
        xyzin_path=source,
        run_dir=target / "evaluation-service",
        coordinate_model=model,
        backend=backend,
        timeout=timeout,
        settings=OptimizerSettings(
            # Hessian rows span every irrep.  Full reference-symmetry
            # projection is valid only for totally symmetric displacements;
            # non-total rows must be allowed to retain their proper subgroup.
            freeze_inactive_sonic=False,
            symmetrize_analytic_gradients=False,
            resume=True,
            coordinate_parallel_workers=max(1, int(workers)),
        ),
    )
    zero = np.zeros(plan.coordinate_count, dtype=float)
    service.initialize_coordinate_projector(zero, geometry.coordinates_angstrom)

    requested_properties = (
        ("energy", "gradient")
        if plan.property_source == "cartesian-gradient"
        else ("energy",)
    )

    def evaluate(point: SonicHessianPoint):
        return point, service.evaluate(
            np.asarray(point.displacement, dtype=float),
            tag=f"sonic-hessian-{point.key}",
            requested_properties=requested_properties,
        )

    center_point = plan.points[0]
    _point, center_evaluation = evaluate(center_point)
    service.accept_electronic_state(center_evaluation)
    evaluated = [(center_point, center_evaluation)]
    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        evaluated.extend(executor.map(evaluate, plan.points[1:]))

    energies = {
        point.key: float(evaluation.result.energy_hartree)
        for point, evaluation in evaluated
    }
    state_audit = {
        point.key: {
            key: value
            for key, value in evaluation.result.execution.items()
            if key.startswith("state_") or key in {"selected_electronic_state"}
        }
        for point, evaluation in evaluated
    }
    point_coordinates = {
        point.key: np.asarray(evaluation.coordinates_angstrom, dtype=float)
        for point, evaluation in evaluated
    }
    if plan.property_source == "cartesian-gradient":
        gradients = {}
        for point, evaluation in evaluated:
            gradient = evaluation.result.gradient_hartree_per_bohr
            if gradient is None:
                raise RuntimeError(
                    f"backend returned no Cartesian gradient at {point.key}"
                )
            gradients[point.key] = np.asarray(gradient, dtype=float)
        definition = read_gic_definition_from_xyzin(source)
        force_constants = sonic_hessian_from_cartesian_gradients(
            plan,
            definition,
            point_coordinates,
            gradients,
        )
    else:
        force_constants = sonic_hessian_from_energies(plan, energies)

    numbers = np.asarray([atomic_number(atom) or 0 for atom in geometry.atoms], dtype=int)
    masses = np.asarray([atomic_mass(int(number)) for number in numbers], dtype=float)
    coordinates_bohr = np.asarray(geometry.coordinates_angstrom, dtype=float) / BOHR_TO_ANGSTROM
    ncart = coordinates_bohr.size
    geometry_input = HessianInput(
        atomic_numbers=numbers,
        cartesian_coordinates_bohr=coordinates_bohr,
        masses_amu=masses,
        cartesian_hessian=np.zeros((ncart, ncart), dtype=float),
        harmonic_frequencies_cm=np.array((), dtype=float),
        source="LINK direct SONIC energy Hessian",
        provenance={
            "BACKEND": backend.name,
            "METHOD": backend.method,
            "BASIS": backend.basis,
        },
    )
    gf = gf_from_acquired_force_constants_and_xyzin(
        force_constants,
        geometry_input,
        source,
        block_by_irrep=True,
    )
    b_matrix = np.asarray(
        build_gic_b_matrix(
            read_gic_definition_from_xyzin(source),
            coordinates_angstrom=np.asarray(geometry.coordinates_angstrom, dtype=float),
        ).rows,
        dtype=float,
    )
    cartesian_modes = cartesian_normal_modes_from_sonic_hessian(
        force_constants,
        b_matrix,
        masses,
        coordinates_bohr.reshape(len(geometry.atoms), 3),
        source="LINK SONIC Hessian",
    )
    result_path = target / "sonic-hessian-gf.json"
    payload = {
        "schema": SONIC_HESSIAN_RESULT_SCHEMA,
        "xyzin": str(source),
        "plan": plan.to_json(),
        "backend": backend.name,
        "method": backend.method,
        "basis": backend.basis,
        "electronic_state": int(backend.electronic_state),
        "state_tracking": backend.state_tracking,
        "energies_hartree": energies,
        "state_audit": state_audit,
        "force_constants_hartree_per_sonic_unit2": force_constants.tolist(),
        "g_matrix": gf.g_matrix.tolist(),
        "frequencies_cm-1": gf.frequencies_cm.tolist(),
        "cartesian_mass_weighted_normal_modes": cartesian_modes.mass_weighted_modes.tolist(),
        "normal_mode_basis_contract": "mass-weighted orthonormal Cartesian modes; 3N-6 rows",
        "gf_block_labels": list(gf.block_labels),
    }
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return SonicHessianResult(
        plan=plan,
        force_constants=force_constants,
        energies_hartree=energies,
        frequencies_cm=gf.frequencies_cm,
        g_matrix=gf.g_matrix,
        state_audit=state_audit,
        result_path=result_path,
    )
