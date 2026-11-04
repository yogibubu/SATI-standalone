"""Hessian-derived molecular mechanics fields in ORACLE PIC coordinates."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Sequence

import numpy as np

from .primitive_coordinates import Primitive, eval_primitives, primitive_b_matrix
from .structural_corrections import BOHR_TO_ANGSTROM


CHEMICAL_COUPLING_RELATIONS = (
    "GEMINAL_BOND_BOND",
    "CONTIGUOUS_ANGLE_ANGLE",
    "CONTIGUOUS_BOND_ANGLE",
    "CONTIGUOUS_ANGLE_TORSION",
)


@dataclass(frozen=True)
class SynthonTypingThresholds:
    """Explicit, user-adjustable tolerances for continuous atomic typing."""

    zeff: float = 0.035
    coordination: float = 0.10
    charge: float = 0.10
    bond_order: float = 0.10
    pi_index: float = 0.10
    pi_pi_index: float = 0.10
    hessian_relative: float = 0.15
    hessian_absolute: float = 1.0e-4
    equilibrium_distance: float = 0.05
    equilibrium_angle: float = math.radians(5.0)
    coupling_relative: float = 0.05
    coupling_absolute: float = 1.0e-4
    coupling_relations: tuple[str, ...] = CHEMICAL_COUPLING_RELATIONS
    eigenvalue_relative: float = 1.0e-8
    eigenvalue_absolute: float = 1.0e-10


@dataclass(frozen=True)
class PrimitiveFunctionalPolicy:
    """Choice of analytic bonded functions fitted to the local QM derivatives."""

    stretch_form: str = "harmonic"
    angle_form: str = "cosine"
    # Provisional median of the B3LYP/6-31G(d) H2, HF and CO radial scans.
    morse_alpha_scale: float = 2.215
    # Optional MM3-derived Morse depths in primitive order.  Positive entries
    # seed D_e while the target QM gradient/curvature determine alpha and R0.
    morse_depth_priors_hartree: tuple[float, ...] = ()
    torsion_two_term_condition_max: float = 1.0e8
    torsion_two_term_amplification_max: float = 1.0e4


@dataclass(frozen=True)
class PrimitiveFunctionalParameter:
    """One typed analytic function and its auditable parameter provenance."""

    primitive_type_id: str
    primitive_indices: tuple[int, ...]
    kind: str
    form: str
    reference_value: float
    gradient_at_qm: float
    curvature_at_qm: float
    parameters: tuple[tuple[str, float], ...]
    parameter_source: str
    gaussian_function: str | None


@dataclass(frozen=True)
class AtomicMMType:
    identifier: str
    atomic_number: int
    atoms: tuple[int, ...]
    mean_zeff: float
    mean_coordination: float
    mean_charge: float
    mean_pi_index: float
    mean_pi_pi_index: float
    mean_incident_curvature: float


@dataclass(frozen=True)
class PrimitiveMMType:
    identifier: str
    kind: str
    atomic_type_signature: tuple[str, ...]
    primitive_indices: tuple[int, ...]
    equilibrium_value: float
    force_constant: float


@dataclass(frozen=True)
class PrimitiveMMCoupling:
    primitive_indices: tuple[int, int]
    primitive_type_ids: tuple[str, str]
    force_constant: float
    normalized_magnitude: float
    relation: str
    diagnostic: str


@dataclass(frozen=True)
class TorsionFourierTerm:
    """One explicit term in a typed torsional Fourier model."""

    periodicity: int
    phase: float
    amplitude: float
    cosine_coefficient: float
    sine_coefficient: float
    parameter_source: str


@dataclass(frozen=True)
class TorsionFourierParameter:
    primitive_type_id: str
    primitive_indices: tuple[int, ...]
    equilibrium_phase: float
    local_curvature: float
    gradient_at_qm: float
    constant: float
    terms: tuple[TorsionFourierTerm, ...]
    fit_method: str
    condition_number: float
    coefficient_amplification: float
    requires_scan: bool
    gradient_residual: float
    curvature_residual: float

    @property
    def periodicities(self) -> tuple[int, ...]:
        return tuple(term.periodicity for term in self.terms)


@dataclass(frozen=True)
class PrimitiveMMForceField:
    equilibrium_values: np.ndarray
    primitive_hessian: np.ndarray
    diagonal_force_constants: np.ndarray
    typed_force_constants: np.ndarray
    typed_equilibrium_values: np.ndarray
    atomic_types: tuple[AtomicMMType, ...]
    primitive_types: tuple[PrimitiveMMType, ...]
    primitive_type_ids: tuple[str, ...]
    sparse_typed_hessian: np.ndarray
    retained_couplings: tuple[PrimitiveMMCoupling, ...]
    coupling_alerts: tuple[PrimitiveMMCoupling, ...]
    torsion_fourier_parameters: tuple[TorsionFourierParameter, ...]
    functional_parameters: tuple[PrimitiveFunctionalParameter, ...]
    spectral_cutoff: float
    discarded_eigenvalue_count: int
    negative_eigenvalue_count: int
    equilibrium_fit_iterations: int
    equilibrium_type_split_count: int
    equilibrium_fit_status: str
    equilibrium_gradient_rms: float
    equilibrium_gradient_max: float
    cartesian_reconstruction_rms: float
    sparse_cartesian_reconstruction_rms: float
    hessian_input_unit: str
    nonbonded_subtraction: tuple[str, ...] = ()
    charge_source: str = "NONE"
    nonbonded_14_scale: float = 0.5
    nonbonded_topology_scaling: str = "DISCRETE_GRAPH"
    topology_switch_alpha_per_angstrom: float = 0.0
    source_cartesian_hessian_rms: float = 0.0
    electrostatic_hessian_rms: float = 0.0
    uff_vdw_hessian_rms: float = 0.0
    residual_cartesian_hessian_rms: float = 0.0
    nonbonded_cartesian_gradient_rms: float = 0.0
    coupling_reference_values: np.ndarray | None = None


def evaluate_primitive_functional_parameter(
    parameter: PrimitiveFunctionalParameter,
    value: float,
) -> tuple[float, float, float]:
    """Return bonded energy, first derivative and curvature for one PIC value."""
    q_value = float(value)
    values = dict(parameter.parameters)
    if parameter.form in {"inverse_distance_quartic_switched", "cosine_quartic"}:
        from .native_reference import NativeReferencePotential

        stretch = parameter.form == "inverse_distance_quartic_switched"
        potential = NativeReferencePotential(
            "stretch" if stretch else "angle",
            values["R0" if stretch else "THETA0"],
            tuple(values[f"C{i}"] for i in range(5)),
            values.get("E_INFINITY"), values.get("ALPHA"),
        )
        return potential.value_gradient_hessian(q_value)
    if parameter.form in {"harmonic", "harmonic_fallback", "linear_bend_component"}:
        q0 = values.get("R0", values.get("THETA0", values.get("Q0")))
        delta = q_value - float(q0)
        force = values["K"]
        return 0.5 * force * delta * delta, force * delta, force
    if parameter.form == "morse":
        exponent = math.exp(-values["ALPHA"] * (q_value - values["R0"]))
        energy = values["D"] * (1.0 - exponent) ** 2
        gradient = 2.0 * values["D"] * values["ALPHA"] * exponent * (1.0 - exponent)
        curvature = 2.0 * values["D"] * values["ALPHA"] ** 2 * exponent * (2.0 * exponent - 1.0)
        return energy, gradient, curvature
    if parameter.form == "inverse_distance":
        inverse_delta = 1.0 / q_value - 1.0 / values["R0"]
        energy = values["A"] * inverse_delta**2
        gradient = -2.0 * values["A"] * inverse_delta / q_value**2
        curvature = 2.0 * values["A"] / q_value**4 + 4.0 * values["A"] * inverse_delta / q_value**3
        return energy, gradient, curvature
    if parameter.form == "inverse_power_stretch":
        ratio = values["R0"] / q_value
        exponent = ratio ** values["P"]
        energy = values["D"] * (1.0 - exponent) ** 2
        gradient = (
            2.0 * values["D"] * values["P"] * exponent * (1.0 - exponent) / q_value
        )
        curvature = (
            2.0
            * values["D"]
            * values["P"]
            * exponent
            * ((2.0 * values["P"] + 1.0) * exponent - (values["P"] + 1.0))
            / q_value**2
        )
        return energy, gradient, curvature
    if parameter.form == "cosine":
        cosine_delta = math.cos(q_value) - math.cos(values["THETA0"])
        energy = values["A"] * cosine_delta**2
        gradient = -2.0 * values["A"] * cosine_delta * math.sin(q_value)
        curvature = 2.0 * values["A"] * (math.sin(q_value) ** 2 - cosine_delta * math.cos(q_value))
        return energy, gradient, curvature
    raise ValueError(f"unsupported primitive functional form: {parameter.form}")


def evaluate_torsion_fourier_parameter(
    parameter: TorsionFourierParameter,
    value: float,
) -> tuple[float, float, float]:
    """Return torsional energy, first derivative and curvature."""

    q_value = float(value)
    energy = float(parameter.constant)
    gradient = 0.0
    curvature = 0.0
    for term in parameter.terms:
        order = int(term.periodicity)
        cosine = math.cos(order * q_value)
        sine = math.sin(order * q_value)
        energy += term.cosine_coefficient * cosine + term.sine_coefficient * sine
        gradient += order * (-term.cosine_coefficient * sine + term.sine_coefficient * cosine)
        curvature -= (
            order * order * (term.cosine_coefficient * cosine + term.sine_coefficient * sine)
        )
    return energy, gradient, curvature


def primitive_force_field_bonded_energy(
    field: PrimitiveMMForceField,
    primitives: Sequence[Primitive],
    coordinates_angstrom: np.ndarray,
) -> float:
    """Evaluate the same native bonded functions used by the gradient."""
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    q_values = eval_primitives(primitives, xyz)
    if len(primitives) != len(field.primitive_type_ids):
        raise ValueError("PIC field and primitive sequence differ")
    functions = {item.primitive_type_id: item for item in field.functional_parameters}
    torsions = {item.primitive_type_id: item for item in field.torsion_fourier_parameters}
    energy = 0.0
    for primitive, type_id, value in zip(primitives, field.primitive_type_ids, q_values, strict=True):
        term = (
            evaluate_torsion_fourier_parameter(torsions[type_id], value)
            if primitive.kind == "dihedral"
            else evaluate_primitive_functional_parameter(functions[type_id], value)
        )
        energy += term[0]
    for coupling in field.retained_couplings:
        left, right = coupling.primitive_indices
        centers = field.coupling_reference_values
        def center(index: int) -> float:
            if centers is not None:
                return float(centers[index])
            if primitives[index].kind == "dihedral":
                return float(field.equilibrium_values[index])
            return float(field.typed_equilibrium_values[index])
        energy += coupling.force_constant * _primitive_displacement(
            primitives[left].kind, q_values[left], center(left)
        ) * _primitive_displacement(
            primitives[right].kind, q_values[right], center(right)
        )
    return float(energy)


def primitive_force_field_bonded_cartesian_gradient(
    field: PrimitiveMMForceField,
    primitives: Sequence[Primitive],
    coordinates_angstrom: np.ndarray,
) -> np.ndarray:
    """Evaluate the native PIC bonded gradient in hartree/angstrom.

    Retained cross terms are centered at the fitted primitive equilibrium
    values, exactly as in the coupled stationarity solve.
    """
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    q_values = eval_primitives(primitives, xyz)
    if len(primitives) != len(field.primitive_type_ids):
        raise ValueError("PIC field and primitive sequence differ")
    function_by_type = {item.primitive_type_id: item for item in field.functional_parameters}
    torsion_by_type = {item.primitive_type_id: item for item in field.torsion_fourier_parameters}
    primitive_gradient = np.zeros(len(primitives), dtype=float)
    for index, (primitive, type_id, q_value) in enumerate(
        zip(primitives, field.primitive_type_ids, q_values, strict=True)
    ):
        if primitive.kind == "dihedral":
            torsion = torsion_by_type[type_id]
            _energy, primitive_gradient[index], _curvature = evaluate_torsion_fourier_parameter(
                torsion, q_value
            )
        else:
            _energy, gradient, _curvature = evaluate_primitive_functional_parameter(
                function_by_type[type_id], q_value
            )
            primitive_gradient[index] = gradient
    for coupling in field.retained_couplings:
        left, right = coupling.primitive_indices
        coupling_centers = field.coupling_reference_values
        left_delta = _primitive_displacement(
            primitives[left].kind,
            q_values[left],
            (
                coupling_centers[left]
                if coupling_centers is not None
                else (
                field.equilibrium_values[left]
                if primitives[left].kind == "dihedral"
                else field.typed_equilibrium_values[left]
                )
            ),
        )
        right_delta = _primitive_displacement(
            primitives[right].kind,
            q_values[right],
            (
                coupling_centers[right]
                if coupling_centers is not None
                else (
                field.equilibrium_values[right]
                if primitives[right].kind == "dihedral"
                else field.typed_equilibrium_values[right]
                )
            ),
        )
        primitive_gradient[left] += coupling.force_constant * right_delta
        primitive_gradient[right] += coupling.force_constant * left_delta
    return primitive_b_matrix(primitives, xyz).T @ primitive_gradient


def primitive_force_field_bonded_cartesian_hessian(
    field: PrimitiveMMForceField,
    primitives: Sequence[Primitive],
    coordinates_angstrom: np.ndarray,
    *,
    step_angstrom: float = 2.0e-5,
) -> np.ndarray:
    """Differentiate the native PIC gradient to obtain hartree/angstrom^2.

    The numerical derivative retains the analytic primitive functions and their
    exact Wilson matrices while including the coordinate-curvature chain-rule
    term required when individual primitive gradients are nonzero.
    """
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("PIC Cartesian coordinates must have shape (natoms, 3)")
    if not math.isfinite(step_angstrom) or step_angstrom <= 0.0:
        raise ValueError("PIC Cartesian Hessian step must be positive")
    flat = xyz.reshape(-1)
    hessian = np.zeros((flat.size, flat.size), dtype=float)
    for column in range(flat.size):
        plus = flat.copy()
        minus = flat.copy()
        plus[column] += step_angstrom
        minus[column] -= step_angstrom
        gradient_plus = primitive_force_field_bonded_cartesian_gradient(
            field, primitives, plus.reshape(xyz.shape)
        )
        gradient_minus = primitive_force_field_bonded_cartesian_gradient(
            field, primitives, minus.reshape(xyz.shape)
        )
        hessian[:, column] = (gradient_plus - gradient_minus) / (2.0 * step_angstrom)
    return 0.5 * (hessian + hessian.T)


def refit_primitive_mm_force_field_hessian(
    field: PrimitiveMMForceField,
    primitives: Sequence[Primitive],
    coordinates_angstrom: np.ndarray,
    target_bonded_hessian_hartree_per_angstrom2: np.ndarray,
    atomic_numbers: Sequence[int],
    *,
    synthons=None,
    functional_policy: PrimitiveFunctionalPolicy = PrimitiveFunctionalPolicy(),
) -> PrimitiveMMForceField:
    """Fit the *evaluated* native PIC Hessian while retaining its reference force.

    The Hessian residual is pulled back through the Wilson matrix.  Local
    functions are then refitted to their actual first and corrected second
    derivatives.  Cross terms are centered at the reference PIC values, so
    they cannot disturb the already fitted reference gradient.  This is an
    all-coupling harmonic fit; sparsification belongs to a separate zoom level.
    """
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    q = eval_primitives(primitives, xyz)
    b = primitive_b_matrix(primitives, xyz)
    target = np.asarray(target_bonded_hessian_hartree_per_angstrom2, dtype=float)
    if target.shape != (xyz.size, xyz.size) or not np.allclose(target, target.T, atol=1e-10):
        raise ValueError("target bonded Cartesian Hessian must be symmetric and match geometry")
    current = primitive_force_field_bonded_cartesian_hessian(field, primitives, xyz)
    inverse = np.linalg.pinv(b, rcond=1e-10)
    correction = inverse.T @ (target - current) @ inverse
    correction = 0.5 * (correction + correction.T)
    function_by_type = {item.primitive_type_id: item for item in field.functional_parameters}
    torsion_by_type = {item.primitive_type_id: item for item in field.torsion_fourier_parameters}
    primitive_gradient = np.zeros(len(primitives))
    local_curvature = np.zeros(len(primitives))
    for index, (primitive, type_id) in enumerate(zip(primitives, field.primitive_type_ids, strict=True)):
        evaluator = (
            evaluate_torsion_fourier_parameter(torsion_by_type[type_id], q[index])
            if primitive.kind == "dihedral"
            else evaluate_primitive_functional_parameter(function_by_type[type_id], q[index])
        )
        primitive_gradient[index], local_curvature[index] = evaluator[1:]
    old_hessian = np.diag(local_curvature)
    for coupling in field.retained_couplings:
        left, right = coupling.primitive_indices
        old_hessian[left, right] += coupling.force_constant
        old_hessian[right, left] += coupling.force_constant
        centers = field.coupling_reference_values
        for index, partner in ((left, right), (right, left)):
            center = (
                centers[partner]
                if centers is not None
                else field.equilibrium_values[partner]
                if primitives[partner].kind == "dihedral"
                else field.typed_equilibrium_values[partner]
            )
            primitive_gradient[index] += coupling.force_constant * _primitive_displacement(
                primitives[partner].kind, q[partner], center
            )
    fitted_hessian = old_hessian + correction
    diagonal = np.diag(fitted_hessian)
    if np.any(diagonal <= 0.0):
        raise ValueError("physical PIC Hessian fit requires positive local curvature")
    types = tuple(
        PrimitiveMMType(
            f"{field.primitive_type_ids[index]}:physical:{index}",
            primitive.kind,
            field.primitive_types[next(
                j for j, item in enumerate(field.primitive_types)
                if item.identifier == field.primitive_type_ids[index]
            )].atomic_type_signature,
            (index,),
            float(q[index] - primitive_gradient[index] / diagonal[index]),
            float(diagonal[index]),
        )
        for index, primitive in enumerate(primitives)
    )
    ids = tuple(item.identifier for item in types)
    periodicities = _torsion_periodicities(
        primitives, tuple(int(z) for z in atomic_numbers), synthons,
        SynthonTypingThresholds(),
    )
    functions = _primitive_functional_parameters(
        primitives, types, q, primitive_gradient, synthons, functional_policy
    )
    torsions = _torsion_fourier_parameters(
        types, periodicities, q, primitive_gradient, functional_policy
    )
    for index, (primitive, type_id) in enumerate(zip(primitives, ids, strict=True)):
        actual = (
            evaluate_torsion_fourier_parameter(
                next(item for item in torsions if item.primitive_type_id == type_id), q[index]
            )
            if primitive.kind == "dihedral"
            else evaluate_primitive_functional_parameter(
                next(item for item in functions if item.primitive_type_id == type_id), q[index]
            )
        )
        if abs(actual[1] - primitive_gradient[index]) > 1e-8 or abs(actual[2] - diagonal[index]) > 1e-8:
            raise ValueError(
                f"native PIC function cannot fit primitive {index} ({primitive.kind}): "
                f"gradient {actual[1]:.8g}/{primitive_gradient[index]:.8g}, "
                f"curvature {actual[2]:.8g}/{diagonal[index]:.8g}"
            )
    couplings = tuple(
        PrimitiveMMCoupling(
            (left, right),
            (ids[left], ids[right]),
            float(fitted_hessian[left, right]),
            abs(float(fitted_hessian[left, right])) / math.sqrt(diagonal[left] * diagonal[right]),
            classify_primitive_coupling(primitives[left], primitives[right]),
            "REFERENCE_CENTERED_FULL_PIC_HESSIAN_FIT",
        )
        for left in range(len(primitives))
        for right in range(left + 1, len(primitives))
        if abs(fitted_hessian[left, right]) > 1e-12
    )
    fitted = replace(
        field,
        primitive_hessian=fitted_hessian,
        diagonal_force_constants=diagonal,
        typed_force_constants=diagonal,
        primitive_types=types,
        primitive_type_ids=ids,
        sparse_typed_hessian=fitted_hessian,
        retained_couplings=couplings,
        coupling_alerts=(),
        functional_parameters=functions,
        torsion_fourier_parameters=torsions,
        coupling_reference_values=q.copy(),
    )
    actual_hessian = primitive_force_field_bonded_cartesian_hessian(fitted, primitives, xyz)
    reconstruction_rms = float(np.sqrt(np.mean((actual_hessian - target) ** 2)))
    return replace(
        fitted,
        cartesian_reconstruction_rms=reconstruction_rms,
        sparse_cartesian_reconstruction_rms=reconstruction_rms,
    )


def _primitive_displacement(kind: str, value: float, reference: float) -> float:
    if kind in {"dihedral", "out_of_plane"}:
        return _periodic_difference(float(value) - float(reference))
    return float(value) - float(reference)


def derive_primitive_mm_force_field(
    primitives: Sequence[Primitive],
    coordinates_angstrom: np.ndarray,
    cartesian_hessian: np.ndarray,
    atomic_numbers: Sequence[int],
    *,
    hessian_unit: str = "hartree/bohr^2",
    synthons=None,
    thresholds: SynthonTypingThresholds = SynthonTypingThresholds(),
    eigenvalue_floor: float = 0.0,
    cartesian_hessian_correction: np.ndarray | None = None,
    cartesian_gradient_correction: np.ndarray | None = None,
    functional_policy: PrimitiveFunctionalPolicy = PrimitiveFunctionalPolicy(),
) -> PrimitiveMMForceField:
    """Project a Cartesian Hessian and its correction separately into PIC.

    The full minimum-norm primitive Hessian is retained for audit.  The MM
    model uses its non-negative diagonal and optionally averages equivalent
    primitive constants after continuous synthon typing.  When supplied, the
    analytic Cartesian correction is mapped to the same redundant primitive
    basis and subtracted there; ORACLE therefore owns a PIC-space field rather
    than a SONIC-space field.
    """
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    numbers = tuple(int(value) for value in atomic_numbers)
    if xyz.shape != (len(numbers), 3):
        raise ValueError("primitive force-field coordinates must have shape (natoms, 3)")
    _validate_functional_policy(functional_policy)
    unknown_relations = set(thresholds.coupling_relations) - set(CHEMICAL_COUPLING_RELATIONS)
    if unknown_relations:
        raise ValueError(
            "unknown PIC coupling relation(s): " + ", ".join(sorted(unknown_relations))
        )
    hessian = np.asarray(cartesian_hessian, dtype=float)
    if hessian.shape != (3 * len(numbers), 3 * len(numbers)):
        raise ValueError("Cartesian Hessian shape does not match the molecular geometry")
    if not np.allclose(hessian, hessian.T, atol=1.0e-10):
        raise ValueError("Cartesian Hessian must be symmetric")
    hessian_a = _cartesian_hessian_in_angstrom_units(hessian, hessian_unit)
    correction_a = np.zeros_like(hessian_a)
    correction_gradient_a = np.zeros(3 * len(numbers), dtype=float)
    if cartesian_hessian_correction is not None:
        correction = np.asarray(cartesian_hessian_correction, dtype=float)
        if correction.shape != hessian.shape:
            raise ValueError("Cartesian Hessian correction shape does not match the QM Hessian")
        if not np.allclose(correction, correction.T, atol=1.0e-10):
            raise ValueError("Cartesian Hessian correction must be symmetric")
        correction_a = _cartesian_hessian_in_angstrom_units(correction, hessian_unit)
    if cartesian_gradient_correction is not None:
        correction_gradient = np.asarray(cartesian_gradient_correction, dtype=float).reshape(-1)
        if correction_gradient.shape != (3 * len(numbers),):
            raise ValueError("Cartesian gradient correction shape does not match the QM Hessian")
        if np.any(~np.isfinite(correction_gradient)):
            raise ValueError("Cartesian gradient correction must be finite")
        correction_gradient_a = correction_gradient / BOHR_TO_ANGSTROM
    b_matrix = primitive_b_matrix(primitives, xyz)
    inverse = np.linalg.pinv(b_matrix, rcond=1.0e-10)
    qm_primitive_hessian = inverse.T @ hessian_a @ inverse
    correction_primitive_hessian = inverse.T @ correction_a @ inverse
    primitive_hessian = qm_primitive_hessian - correction_primitive_hessian
    primitive_hessian = 0.5 * (primitive_hessian + primitive_hessian.T)
    regularized_hessian, spectral_cutoff, discarded, negative = _regularize_hessian(
        primitive_hessian,
        thresholds,
        eigenvalue_floor=float(eigenvalue_floor),
    )
    diagonal = np.maximum(np.diag(regularized_hessian), float(eigenvalue_floor))
    equilibrium = eval_primitives(primitives, xyz)
    periodicities = _torsion_periodicities(primitives, numbers, synthons, thresholds)

    atomic_types, atom_type_ids = _build_atomic_types(
        numbers, primitives, diagonal, synthons, thresholds
    )
    primitive_types, primitive_type_ids, typed, typed_equilibrium = _build_primitive_types(
        primitives,
        equilibrium,
        diagonal,
        atom_type_ids,
        synthons,
        thresholds,
        periodicities,
    )
    (
        primitive_types,
        primitive_type_ids,
        typed_equilibrium,
        fit_iterations,
        type_split_count,
        gradient_rms,
        gradient_max,
        fitted_primitive_gradient,
    ) = _refine_parameter_types_for_equilibrium(
        primitives,
        equilibrium,
        b_matrix,
        typed,
        primitive_types,
        primitive_type_ids,
        correction_gradient_a,
    )
    sparse_hessian, retained, alerts = _sparse_chemical_hessian(
        primitives,
        regularized_hessian,
        typed,
        primitive_type_ids,
        thresholds,
    )
    (
        primitive_types,
        primitive_type_ids,
        typed_equilibrium,
        nonlinear_split_count,
        gradient_rms,
        gradient_max,
        fitted_primitive_gradient,
        functional_parameters,
        torsion_fourier,
    ) = _refine_parameter_types_for_functional_equilibrium(
        primitives,
        equilibrium,
        b_matrix,
        typed,
        primitive_types,
        primitive_type_ids,
        sparse_hessian,
        periodicities,
        correction_gradient_a,
        synthons,
        functional_policy,
    )
    type_split_count += nonlinear_split_count
    # Functional-equilibrium refinement may split a shared parameter type.
    # Rebuild coupling metadata with the final type identifiers; the numerical
    # sparse Hessian support itself is unchanged by this relabeling.
    sparse_hessian, retained, alerts = _sparse_chemical_hessian(
        primitives,
        regularized_hessian,
        typed,
        primitive_type_ids,
        thresholds,
    )
    function_by_type = {item.primitive_type_id: item for item in functional_parameters}
    primitive_types = tuple(
        PrimitiveMMType(
            item.identifier,
            item.kind,
            item.atomic_type_signature,
            item.primitive_indices,
            _functional_equilibrium_value(
                function_by_type.get(item.identifier), item.equilibrium_value
            ),
            item.force_constant,
        )
        for item in primitive_types
    )
    typed_equilibrium = np.asarray(
        [
            _functional_equilibrium_value(
                function_by_type.get(type_id),
                typed_equilibrium[index],
            )
            for index, type_id in enumerate(primitive_type_ids)
        ],
        dtype=float,
    )
    if torsion_fourier:
        fourier_by_type = {item.primitive_type_id: item for item in torsion_fourier}
        primitive_types = tuple(
            PrimitiveMMType(
                item.identifier,
                item.kind,
                item.atomic_type_signature,
                item.primitive_indices,
                (
                    fourier_by_type[item.identifier].equilibrium_phase
                    if item.identifier in fourier_by_type
                    else item.equilibrium_value
                ),
                item.force_constant,
            )
            for item in primitive_types
        )
        typed_equilibrium = np.asarray(
            [
                (
                    fourier_by_type[type_id].equilibrium_phase
                    if type_id in fourier_by_type
                    else typed_equilibrium[index]
                )
                for index, type_id in enumerate(primitive_type_ids)
            ],
            dtype=float,
        )
    reconstructed = b_matrix.T @ primitive_hessian @ b_matrix
    sparse_reconstructed = b_matrix.T @ sparse_hessian @ b_matrix
    residual_hessian_a = hessian_a - correction_a
    rms = float(np.sqrt(np.mean((reconstructed - residual_hessian_a) ** 2)))
    sparse_rms = float(np.sqrt(np.mean((sparse_reconstructed - residual_hessian_a) ** 2)))
    return PrimitiveMMForceField(
        equilibrium_values=equilibrium,
        primitive_hessian=primitive_hessian,
        diagonal_force_constants=diagonal,
        typed_force_constants=typed,
        typed_equilibrium_values=typed_equilibrium,
        atomic_types=atomic_types,
        primitive_types=primitive_types,
        primitive_type_ids=primitive_type_ids,
        sparse_typed_hessian=sparse_hessian,
        retained_couplings=retained,
        coupling_alerts=alerts,
        torsion_fourier_parameters=torsion_fourier,
        functional_parameters=functional_parameters,
        spectral_cutoff=spectral_cutoff,
        discarded_eigenvalue_count=discarded,
        negative_eigenvalue_count=negative,
        equilibrium_fit_iterations=fit_iterations,
        equilibrium_type_split_count=type_split_count,
        equilibrium_fit_status=(
            "PASS" if gradient_rms <= 1.0e-8 else "REVIEW_PARAMETER_TYPING_OR_FOURIER_ORDER"
        ),
        equilibrium_gradient_rms=gradient_rms,
        equilibrium_gradient_max=gradient_max,
        cartesian_reconstruction_rms=rms,
        sparse_cartesian_reconstruction_rms=sparse_rms,
        hessian_input_unit=str(hessian_unit),
    )


def _cartesian_hessian_in_angstrom_units(
    hessian: np.ndarray,
    hessian_unit: str,
) -> np.ndarray:
    normalized_unit = str(hessian_unit).strip().lower().replace("angstrom", "a")
    if normalized_unit in {"hartree/bohr^2", "eh/bohr^2"}:
        return np.asarray(hessian, dtype=float) / BOHR_TO_ANGSTROM**2
    if normalized_unit in {"hartree/a^2", "eh/a^2", "hartree/å^2"}:
        return np.asarray(hessian, dtype=float)
    raise ValueError(f"unsupported Cartesian Hessian unit: {hessian_unit}")


def _build_atomic_types(
    atomic_numbers: tuple[int, ...],
    primitives: Sequence[Primitive],
    diagonal: np.ndarray,
    synthons,
    thresholds: SynthonTypingThresholds,
) -> tuple[tuple[AtomicMMType, ...], tuple[str, ...]]:
    descriptors = []
    for atom, z in enumerate(atomic_numbers):
        if synthons is None:
            descriptors.append(
                (
                    atom,
                    z,
                    float(z),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    _atom_hessian_fingerprint(atom, primitives, diagonal),
                )
            )
        else:
            descriptors.append(
                (
                    atom,
                    z,
                    float(synthons.Zeff(atom)),
                    float(synthons.cna(atom)),
                    float(synthons.charge(atom)),
                    float(synthons.pi_index(atom)),
                    float(synthons.pi_pi_index(atom)),
                    _atom_hessian_fingerprint(atom, primitives, diagonal),
                )
            )
    clusters: list[
        list[
            tuple[
                int,
                int,
                float,
                float,
                float,
                float,
                float,
                tuple[tuple[str, tuple[float, ...]], ...],
            ]
        ]
    ] = []
    for descriptor in descriptors:
        placed = False
        for cluster in clusters:
            reference = cluster[0]
            if descriptor[1] != reference[1]:
                continue
            if abs(descriptor[2] - np.mean([item[2] for item in cluster])) > thresholds.zeff:
                continue
            if (
                abs(descriptor[3] - np.mean([item[3] for item in cluster]))
                > thresholds.coordination
            ):
                continue
            if abs(descriptor[4] - np.mean([item[4] for item in cluster])) > thresholds.charge:
                continue
            if abs(descriptor[5] - np.mean([item[5] for item in cluster])) > thresholds.pi_index:
                continue
            if (
                abs(descriptor[6] - np.mean([item[6] for item in cluster]))
                > thresholds.pi_pi_index
            ):
                continue
            if not _hessian_fingerprints_match(descriptor[7], reference[7], thresholds):
                continue
            cluster.append(descriptor)
            placed = True
            break
        if not placed:
            clusters.append([descriptor])
    types: list[AtomicMMType] = []
    ids = [""] * len(atomic_numbers)
    for type_index, cluster in enumerate(clusters, start=1):
        identifier = f"A{type_index:03d}"
        atoms = tuple(item[0] for item in cluster)
        for atom in atoms:
            ids[atom] = identifier
        types.append(
            AtomicMMType(
                identifier=identifier,
                atomic_number=cluster[0][1],
                atoms=atoms,
                mean_zeff=float(np.mean([item[2] for item in cluster])),
                mean_coordination=float(np.mean([item[3] for item in cluster])),
                mean_charge=float(np.mean([item[4] for item in cluster])),
                mean_pi_index=float(np.mean([item[5] for item in cluster])),
                mean_pi_pi_index=float(np.mean([item[6] for item in cluster])),
                mean_incident_curvature=float(
                    np.mean(
                        [value for item in cluster for _kind, values in item[7] for value in values]
                        or [0.0]
                    )
                ),
            )
        )
    return tuple(types), tuple(ids)


def _build_primitive_types(
    primitives: Sequence[Primitive],
    equilibrium: np.ndarray,
    diagonal: np.ndarray,
    atom_type_ids: tuple[str, ...],
    synthons,
    thresholds: SynthonTypingThresholds,
    periodicities: tuple[tuple[int, int], ...],
) -> tuple[tuple[PrimitiveMMType, ...], tuple[str, ...], np.ndarray, np.ndarray]:
    base_groups: dict[tuple[str, tuple[str, ...]], list[int]] = {}
    for index, primitive in enumerate(primitives):
        signature = tuple(atom_type_ids[atom] for atom in primitive.atoms)
        if primitive.kind in {"bond", "hbond_dist", "pseudo_bond"}:
            signature = tuple(sorted(signature))
            if synthons is not None and primitive.kind == "bond":
                left, right = primitive.atoms
                width = max(float(thresholds.bond_order), 1.0e-12)
                signature = (
                    *signature,
                    f"M{round(float(synthons.bond_order(left, right)) / width)}",
                    f"PI{round(float(synthons.bond_order_pi(left, right)) / width)}",
                    f"PIPI{round(float(synthons.bond_order_pi_pi(left, right)) / width)}",
                )
        elif primitive.kind == "angle" and len(signature) == 3:
            signature = (
                min(signature[0], signature[2]),
                signature[1],
                max(signature[0], signature[2]),
            )
        elif primitive.kind == "dihedral" and len(signature) == 4:
            signature = min(signature, tuple(reversed(signature)))
            periodicity_pair = periodicities[index]
            signature = (*signature, f"N{periodicity_pair[0]}+{periodicity_pair[1]}")
        base_groups.setdefault((primitive.kind, signature), []).append(index)
    result: list[PrimitiveMMType] = []
    ids = [""] * len(primitives)
    typed = np.zeros(len(primitives), dtype=float)
    typed_equilibrium = np.zeros(len(primitives), dtype=float)
    groups: list[tuple[str, tuple[str, ...], list[int]]] = []
    for (kind, signature), indices in sorted(base_groups.items()):
        clusters: list[list[int]] = []
        for index in indices:
            for cluster in clusters:
                if _primitive_parameters_match(
                    kind,
                    float(equilibrium[index]),
                    float(diagonal[index]),
                    [float(equilibrium[item]) for item in cluster],
                    [float(diagonal[item]) for item in cluster],
                    thresholds,
                ):
                    cluster.append(index)
                    break
            else:
                clusters.append([index])
        groups.extend((kind, signature, cluster) for cluster in clusters)
    for type_index, (kind, signature, indices) in enumerate(groups, start=1):
        identifier = f"P{type_index:03d}"
        constant = float(np.mean(diagonal[indices]))
        q0 = _mean_equilibrium(kind, equilibrium[indices])
        for index in indices:
            ids[index] = identifier
            typed[index] = constant
            typed_equilibrium[index] = q0
        result.append(PrimitiveMMType(identifier, kind, signature, tuple(indices), q0, constant))
    return tuple(result), tuple(ids), typed, typed_equilibrium


def _regularize_hessian(
    hessian: np.ndarray,
    thresholds: SynthonTypingThresholds,
    *,
    eigenvalue_floor: float,
) -> tuple[np.ndarray, float, int, int]:
    values, vectors = np.linalg.eigh(np.asarray(hessian, dtype=float))
    scale = float(np.max(np.abs(values))) if values.size else 0.0
    cutoff = max(
        float(thresholds.eigenvalue_absolute), float(thresholds.eigenvalue_relative) * scale
    )
    negative = int(np.count_nonzero(values < -cutoff))
    keep = values > max(cutoff, float(eigenvalue_floor))
    filtered = np.where(keep, np.maximum(values, float(eigenvalue_floor)), 0.0)
    regularized = (vectors * filtered) @ vectors.T
    return (
        0.5 * (regularized + regularized.T),
        cutoff,
        int(values.size - np.count_nonzero(keep)),
        negative,
    )


def _atom_hessian_fingerprint(
    atom: int,
    primitives: Sequence[Primitive],
    diagonal: np.ndarray,
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    grouped: dict[str, list[float]] = {}
    for index, primitive in enumerate(primitives):
        if atom in primitive.atoms:
            grouped.setdefault(primitive.kind, []).append(float(diagonal[index]))
    return tuple((kind, tuple(sorted(values))) for kind, values in sorted(grouped.items()))


def _hessian_fingerprints_match(
    left: tuple[tuple[str, tuple[float, ...]], ...],
    right: tuple[tuple[str, tuple[float, ...]], ...],
    thresholds: SynthonTypingThresholds,
) -> bool:
    if tuple(kind for kind, _values in left) != tuple(kind for kind, _values in right):
        return False
    for (_kind_l, values_l), (_kind_r, values_r) in zip(left, right, strict=True):
        if len(values_l) != len(values_r):
            return False
        for value_l, value_r in zip(values_l, values_r, strict=True):
            if not _curvatures_match(value_l, value_r, thresholds):
                return False
    return True


def _curvatures_match(
    left: float,
    right: float,
    thresholds: SynthonTypingThresholds,
) -> bool:
    difference = abs(float(left) - float(right))
    scale = max(abs(float(left)), abs(float(right)), float(thresholds.hessian_absolute))
    return difference <= max(
        float(thresholds.hessian_absolute), float(thresholds.hessian_relative) * scale
    )


def _primitive_parameters_match(
    kind: str,
    q0: float,
    force_constant: float,
    reference_q0: Sequence[float],
    reference_force: Sequence[float],
    thresholds: SynthonTypingThresholds,
) -> bool:
    mean_q0 = _mean_equilibrium(kind, np.asarray(reference_q0, dtype=float))
    q_difference = (
        abs(_periodic_difference(q0 - mean_q0)) if kind == "dihedral" else abs(q0 - mean_q0)
    )
    if kind == "dihedral":
        # One Fourier type must have one phase. Near-but-not-identical torsions
        # are split so its analytic gradient and Hessian are exact at q_QM.
        q_tolerance = min(float(thresholds.equilibrium_angle), 1.0e-8)
    elif kind in {"bond", "hbond_dist", "pseudo_bond", "frag_dist", "frag_atom_dist"}:
        q_tolerance = float(thresholds.equilibrium_distance)
    else:
        q_tolerance = float(thresholds.equilibrium_angle)
    return q_difference <= q_tolerance and _curvatures_match(
        force_constant,
        float(np.mean(reference_force)),
        thresholds,
    )


def _mean_equilibrium(kind: str, values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    if kind != "dihedral":
        return float(np.mean(array))
    return float(math.atan2(float(np.mean(np.sin(array))), float(np.mean(np.cos(array)))))


def _periodic_difference(value: float) -> float:
    return float((float(value) + math.pi) % (2.0 * math.pi) - math.pi)


def _torsion_periodicities(
    primitives: Sequence[Primitive],
    atomic_numbers: tuple[int, ...],
    synthons,
    thresholds: SynthonTypingThresholds,
) -> tuple[tuple[int, int], ...]:
    adjacency = {atom: set() for atom in range(len(atomic_numbers))}
    for primitive in primitives:
        if primitive.kind == "bond" and len(primitive.atoms) == 2:
            left, right = primitive.atoms
            adjacency[left].add(right)
            adjacency[right].add(left)

    def local_period(center: int, excluded: int) -> int:
        clusters: list[list[int]] = []
        for atom in sorted(adjacency[center] - {excluded}):
            for cluster in clusters:
                reference = cluster[0]
                if atomic_numbers[atom] != atomic_numbers[reference]:
                    continue
                if (
                    synthons is not None
                    and abs(float(synthons.Zeff(atom)) - float(synthons.Zeff(reference)))
                    > thresholds.zeff
                ):
                    continue
                cluster.append(atom)
                break
            else:
                clusters.append([atom])
        return max((len(cluster) for cluster in clusters), default=1)

    def sp2_like(center: int, excluded: int) -> bool | None:
        neighbors = adjacency[center] - {excluded}
        if synthons is not None:
            pi_character = sum(
                float(synthons.bond_order_total_pi(center, atom)) for atom in neighbors
            )
            if pi_character > 0.35:
                return True
            domains = float(synthons.electron_domains(center))
            if domains <= 3.35:
                return True
            if domains >= 3.65:
                return False
        degree = len(adjacency[center])
        if degree >= 4:
            return False
        if degree == 3:
            return True
        return None

    result = []
    for primitive in primitives:
        if primitive.kind != "dihedral" or len(primitive.atoms) != 4:
            result.append((1, 2))
            continue
        _i, left, right, _ell = primitive.atoms
        local_multiplicity = max(
            1,
            math.lcm(local_period(left, right), local_period(right, left)),
        )
        left_sp2 = sp2_like(left, right)
        right_sp2 = sp2_like(right, left)
        if local_multiplicity >= 3 or left_sp2 is False or right_sp2 is False:
            result.append((2, 3))
        else:
            result.append((1, 2))
    return tuple(result)


def _sparse_chemical_hessian(
    primitives: Sequence[Primitive],
    hessian: np.ndarray,
    typed_diagonal: np.ndarray,
    primitive_type_ids: tuple[str, ...],
    thresholds: SynthonTypingThresholds,
) -> tuple[np.ndarray, tuple[PrimitiveMMCoupling, ...], tuple[PrimitiveMMCoupling, ...]]:
    sparse = np.diag(np.asarray(typed_diagonal, dtype=float))
    retained: list[PrimitiveMMCoupling] = []
    alerts: list[PrimitiveMMCoupling] = []
    for left in range(len(primitives)):
        for right in range(left + 1, len(primitives)):
            value = float(hessian[left, right])
            denominator = math.sqrt(
                max(abs(float(hessian[left, left] * hessian[right, right])), 1.0e-30)
            )
            normalized = abs(value) / denominator
            if (
                abs(value) < thresholds.coupling_absolute
                or normalized < thresholds.coupling_relative
            ):
                continue
            relation = _chemical_coupling_relation(primitives[left], primitives[right])
            if relation is not None and relation in thresholds.coupling_relations:
                sparse[left, right] = sparse[right, left] = value
                retained.append(
                    PrimitiveMMCoupling(
                        (left, right),
                        (primitive_type_ids[left], primitive_type_ids[right]),
                        value,
                        normalized,
                        relation,
                        "RETAINED_CHEMICALLY_LOCAL",
                    )
                )
                continue
            if relation is not None:
                alerts.append(
                    PrimitiveMMCoupling(
                        (left, right),
                        (primitive_type_ids[left], primitive_type_ids[right]),
                        value,
                        normalized,
                        relation,
                        "EXCLUDED_BY_COUPLING_RELATION_POLICY",
                    )
                )
                continue
            torsion_pair = primitives[left].kind == primitives[right].kind == "dihedral"
            alerts.append(
                PrimitiveMMCoupling(
                    (left, right),
                    (primitive_type_ids[left], primitive_type_ids[right]),
                    value,
                    normalized,
                    "TORSION_TORSION" if torsion_pair else "NONLOCAL_OR_UNSUPPORTED",
                    (
                        "REVIEW_2D_FOURIER_TORSION_POTENTIAL"
                        if torsion_pair
                        else "IMPORTANT_COUPLING_EXCLUDED_FROM_SPARSE_FIELD"
                    ),
                )
            )
    return sparse, tuple(retained), tuple(alerts)


def _chemical_coupling_relation(left: Primitive, right: Primitive) -> str | None:
    bonds = {"bond", "hbond_dist", "pseudo_bond"}
    angles = {"angle", "linear_bend"}
    left_atoms = set(left.atoms)
    right_atoms = set(right.atoms)
    if left.kind in bonds and right.kind in bonds and len(left_atoms & right_atoms) == 1:
        return "GEMINAL_BOND_BOND"
    if left.kind in angles and right.kind in angles and len(left_atoms & right_atoms) >= 2:
        return "CONTIGUOUS_ANGLE_ANGLE"
    if left.kind in bonds and right.kind in angles and left_atoms <= right_atoms:
        return "CONTIGUOUS_BOND_ANGLE"
    if right.kind in bonds and left.kind in angles and right_atoms <= left_atoms:
        return "CONTIGUOUS_BOND_ANGLE"
    if left.kind in angles and right.kind == "dihedral" and left_atoms <= right_atoms:
        return "CONTIGUOUS_ANGLE_TORSION"
    if right.kind in angles and left.kind == "dihedral" and right_atoms <= left_atoms:
        return "CONTIGUOUS_ANGLE_TORSION"
    return None


def classify_primitive_coupling(left: Primitive, right: Primitive) -> str:
    """Classify a coupling after numerical selection, never before it."""

    chemical = _chemical_coupling_relation(left, right)
    if chemical is not None:
        return chemical
    if left.kind == right.kind == "dihedral":
        return "TORSION_TORSION"
    shared_atoms = len(set(left.atoms) & set(right.atoms))
    if shared_atoms:
        return f"OTHER_SHARED_{shared_atoms}"
    return "DISJOINT"


def _torsion_fourier_parameters(
    primitive_types: tuple[PrimitiveMMType, ...],
    periodicities: tuple[tuple[int, int], ...],
    reference_values: np.ndarray,
    primitive_gradient: np.ndarray,
    policy: PrimitiveFunctionalPolicy,
) -> tuple[TorsionFourierParameter, ...]:
    result: list[TorsionFourierParameter] = []
    for primitive_type in primitive_types:
        if primitive_type.kind != "dihedral":
            continue
        periodicity_pair = periodicities[primitive_type.primitive_indices[0]]
        if any(
            periodicities[index] != periodicity_pair for index in primitive_type.primitive_indices
        ):
            raise ValueError("one torsion type cannot mix Fourier periodicity pairs")
        curvature = max(float(primitive_type.force_constant), 0.0)
        q_qm = _mean_equilibrium(
            "dihedral", np.asarray(reference_values)[list(primitive_type.primitive_indices)]
        )
        gradient_qm = float(
            np.mean(np.asarray(primitive_gradient)[list(primitive_type.primitive_indices)])
        )
        terms, method, condition, amplification, requires_scan = _local_torsion_terms(
            periodicity_pair,
            q_qm,
            gradient_qm,
            curvature,
            policy,
        )
        constant = -sum(
            term.cosine_coefficient * math.cos(term.periodicity * q_qm)
            + term.sine_coefficient * math.sin(term.periodicity * q_qm)
            for term in terms
        )
        provisional = TorsionFourierParameter(
            primitive_type.identifier,
            primitive_type.primitive_indices,
            q_qm,
            curvature,
            gradient_qm,
            constant,
            terms,
            method,
            condition,
            amplification,
            requires_scan,
            0.0,
            0.0,
        )
        _energy, fitted_gradient, fitted_curvature = evaluate_torsion_fourier_parameter(
            provisional, q_qm
        )
        result.append(
            TorsionFourierParameter(
                primitive_type.identifier,
                primitive_type.primitive_indices,
                q_qm,
                curvature,
                gradient_qm,
                constant,
                terms,
                method,
                condition,
                amplification,
                requires_scan,
                fitted_gradient - gradient_qm,
                fitted_curvature - curvature,
            )
        )
    return tuple(result)


def _local_torsion_terms(
    periodicity_pair: tuple[int, int],
    q_qm: float,
    gradient_qm: float,
    curvature: float,
    policy: PrimitiveFunctionalPolicy,
) -> tuple[tuple[TorsionFourierTerm, ...], str, float, float, bool]:
    """Generalize HessFit eqs. 15--16 with an auditable conditioning gate."""

    n1, n2 = (int(value) for value in periodicity_pair)
    sign1 = float((-1) ** (n1 + 1))
    matrix = np.asarray(
        [
            [
                -sign1 * n1 * math.sin(n1 * q_qm),
                -n2 * math.sin(n2 * q_qm),
            ],
            [
                -sign1 * n1 * n1 * math.cos(n1 * q_qm),
                -n2 * n2 * math.cos(n2 * q_qm),
            ],
        ],
        dtype=float,
    )
    target = np.asarray([gradient_qm, curvature], dtype=float)
    condition = float(np.linalg.cond(matrix))
    coefficients = np.asarray([math.nan, math.nan])
    amplification = math.inf
    if math.isfinite(condition) and condition <= policy.torsion_two_term_condition_max:
        coefficients = np.linalg.solve(matrix, target)
        natural_scale = max(
            abs(float(gradient_qm)) / max(n1, n2),
            abs(float(curvature)) / max(n1 * n1, n2 * n2),
            1.0e-14,
        )
        amplification = float(np.linalg.norm(coefficients) / natural_scale)
    if (
        np.all(np.isfinite(coefficients))
        and amplification <= policy.torsion_two_term_amplification_max
    ):
        cosine_coefficients = (sign1 * float(coefficients[0]), float(coefficients[1]))
        terms = tuple(
            _fourier_term(
                periodicity,
                cosine_coefficient,
                0.0,
                "HESSFIT_EQ15_LOCAL_TWO_PERIODICITY",
            )
            for periodicity, cosine_coefficient in zip((n1, n2), cosine_coefficients, strict=True)
        )
        return terms, "LOCAL_TWO_PERIODICITY_EQ15", condition, amplification, False

    # A phase-flexible term retains exact stationarity and curvature when the
    # fixed-phase two-term system is singular. Its global shape must be replaced
    # or confirmed by a torsional scan.
    periodicity = n2
    sine = math.sin(periodicity * q_qm)
    cosine = math.cos(periodicity * q_qm)
    cosine_coefficient = -curvature * cosine / float(
        periodicity * periodicity
    ) - gradient_qm * sine / float(periodicity)
    sine_coefficient = -curvature * sine / float(
        periodicity * periodicity
    ) + gradient_qm * cosine / float(periodicity)
    term = _fourier_term(
        periodicity,
        cosine_coefficient,
        sine_coefficient,
        "PHASE_FLEXIBLE_ILL_CONDITIONED_FALLBACK",
    )
    return (term,), "PHASE_FLEXIBLE_FALLBACK", condition, amplification, True


def _fourier_term(
    periodicity: int,
    cosine_coefficient: float,
    sine_coefficient: float,
    source: str,
) -> TorsionFourierTerm:
    return TorsionFourierTerm(
        int(periodicity),
        math.atan2(sine_coefficient, cosine_coefficient),
        math.hypot(cosine_coefficient, sine_coefficient),
        float(cosine_coefficient),
        float(sine_coefficient),
        source,
    )


def _validate_functional_policy(policy: PrimitiveFunctionalPolicy) -> None:
    if policy.stretch_form not in {
        "harmonic",
        "morse",
        "inverse_distance",
        "inverse_power",
    }:
        raise ValueError(
            "stretch form must be harmonic, morse, inverse_distance or inverse_power"
        )
    if policy.angle_form not in {"harmonic", "cosine"}:
        raise ValueError("angle form must be harmonic or cosine")
    if not math.isfinite(policy.morse_alpha_scale) or policy.morse_alpha_scale <= 0.0:
        raise ValueError("Morse alpha scale must be positive")
    if (
        not math.isfinite(policy.torsion_two_term_condition_max)
        or policy.torsion_two_term_condition_max <= 1.0
    ):
        raise ValueError("torsion two-term condition limit must be finite and greater than one")
    if (
        not math.isfinite(policy.torsion_two_term_amplification_max)
        or policy.torsion_two_term_amplification_max <= 0.0
    ):
        raise ValueError("torsion two-term amplification limit must be finite and positive")


def _primitive_functional_parameters(
    primitives: Sequence[Primitive],
    primitive_types: tuple[PrimitiveMMType, ...],
    reference_values: np.ndarray,
    primitive_gradient: np.ndarray,
    synthons,
    policy: PrimitiveFunctionalPolicy,
) -> tuple[PrimitiveFunctionalParameter, ...]:
    """Fit analytic functions to the type-mean gradient and curvature at q_QM.

    A type may contain symmetry-equivalent primitives at slightly different
    values.  The common function is therefore fitted at their circular/linear
    mean, exactly as the common force constant and equilibrium parameter are.
    """
    result: list[PrimitiveFunctionalParameter] = []
    q_values = np.asarray(reference_values, dtype=float)
    gradients = np.asarray(primitive_gradient, dtype=float)
    stretches = {"bond", "hbond_dist", "pseudo_bond"}
    for primitive_type in primitive_types:
        indices = primitive_type.primitive_indices
        q_qm = _mean_equilibrium(
            primitive_type.kind,
            q_values[list(indices)],
        )
        gradient = float(np.mean(gradients[list(indices)]))
        curvature = max(float(primitive_type.force_constant), 0.0)
        kind = primitive_type.kind
        if kind in stretches:
            form, parameters, source, gaussian = _stretch_function_parameters(
                primitives,
                indices,
                q_qm,
                gradient,
                curvature,
                primitive_type.equilibrium_value,
                synthons,
                policy,
            )
        elif kind == "angle":
            form, parameters, source, gaussian = _angle_function_parameters(
                q_qm,
                gradient,
                curvature,
                primitive_type.equilibrium_value,
                policy.angle_form,
            )
        elif kind == "linear_bend":
            form = "linear_bend_component"
            parameters = (("Q0", primitive_type.equilibrium_value), ("K", curvature))
            source = "PIC_LINEAR_BEND_COMPONENT_HESSIAN_FIT"
            gaussian = "LinBnd1"
        elif kind == "dihedral":
            continue
        else:
            form = "harmonic"
            parameters = (("Q0", primitive_type.equilibrium_value), ("K", curvature))
            source = "PIC_HESSIAN_AND_EQUILIBRIUM_REFIT"
            gaussian = None
        result.append(
            PrimitiveFunctionalParameter(
                primitive_type.identifier,
                indices,
                kind,
                form,
                q_qm,
                gradient,
                curvature,
                parameters,
                source,
                gaussian,
            )
        )
    return tuple(result)


def _stretch_function_parameters(
    primitives: Sequence[Primitive],
    indices: tuple[int, ...],
    q_qm: float,
    gradient: float,
    curvature: float,
    harmonic_q0: float,
    synthons,
    policy: PrimitiveFunctionalPolicy,
) -> tuple[str, tuple[tuple[str, float], ...], str, str | None]:
    if policy.stretch_form == "harmonic":
        return (
            "harmonic",
            (("R0", harmonic_q0), ("K", curvature)),
            "PIC_HESSIAN_AND_EQUILIBRIUM_REFIT",
            "HrmStr1",
        )
    if policy.stretch_form == "inverse_distance":
        amplitude = 0.5 * (curvature + 2.0 * gradient / q_qm) * q_qm**4
        if amplitude <= 0.0:
            return (
                "harmonic_fallback",
                (("R0", harmonic_q0), ("K", curvature)),
                "INVERSE_DISTANCE_NONPOSITIVE_AMPLITUDE_FALLBACK",
                "HrmStr1",
            )
        inverse_r0 = 1.0 / q_qm + gradient * q_qm**2 / (2.0 * amplitude)
        if inverse_r0 <= 0.0:
            return (
                "harmonic_fallback",
                (("R0", harmonic_q0), ("K", curvature)),
                "INVERSE_DISTANCE_NONPOSITIVE_R0_FALLBACK",
                "HrmStr1",
            )
        return (
            "inverse_distance",
            (("R0", 1.0 / inverse_r0), ("A", amplitude)),
            "EXACT_LOCAL_GRADIENT_AND_CURVATURE_IN_1_OVER_R",
            None,
        )
    if policy.stretch_form == "inverse_power":
        depth_priors = tuple(
            float(policy.morse_depth_priors_hartree[index])
            for index in indices
            if index < len(policy.morse_depth_priors_hartree)
            and math.isfinite(policy.morse_depth_priors_hartree[index])
            and policy.morse_depth_priors_hartree[index] > 0.0
        )
        if depth_priors:
            matched = _inverse_power_fixed_depth_local_match(
                float(np.mean(depth_priors)),
                q_qm,
                gradient,
                curvature,
            )
            if matched is not None:
                depth, r0, exponent = matched
                return (
                    "inverse_power_stretch",
                    (("R0", r0), ("D", depth), ("P", exponent)),
                    "ZAFF_DE_PRIOR_WITH_EXACT_QM_LOCAL_GRADIENT_AND_CURVATURE_IN_1_OVER_R",
                    None,
                )
        return (
            "harmonic_fallback",
            (("R0", harmonic_q0), ("K", curvature)),
            "INVERSE_POWER_REQUIRES_POSITIVE_ZAFF_DE_PRIOR_FALLBACK",
            "HrmStr1",
        )
    depth_priors = tuple(
        float(policy.morse_depth_priors_hartree[index])
        for index in indices
        if index < len(policy.morse_depth_priors_hartree)
        and math.isfinite(policy.morse_depth_priors_hartree[index])
        and policy.morse_depth_priors_hartree[index] > 0.0
    )
    if depth_priors:
        matched = _morse_fixed_depth_local_match(
            float(np.mean(depth_priors)),
            q_qm,
            gradient,
            curvature,
        )
        if matched is not None:
            depth, r0, alpha = matched
            return (
                "morse",
                (("R0", r0), ("D", depth), ("ALPHA", alpha)),
                "MM3_MORSE_D_E_PRIOR_WITH_EXACT_QM_LOCAL_GRADIENT_AND_CURVATURE",
                "MrsStr1",
            )
    alpha_values = []
    bond_orders = []
    for index in indices:
        left, right = primitives[index].atoms
        if synthons is None:
            radius_sum = max(q_qm, 1.0e-6)
            bond_order = 1.0
        else:
            radius_sum = max(
                float(synthons.covalent_radius_eff(left))
                + float(synthons.covalent_radius_eff(right)),
                1.0e-6,
            )
            bond_order = max(float(synthons.bond_order(left, right)), 0.1)
        bond_orders.append(bond_order)
        alpha_values.append(policy.morse_alpha_scale * math.sqrt(bond_order) / radius_sum)
    alpha = float(np.mean(alpha_values))
    if abs(gradient) <= 1.0e-10 * max(1.0, curvature / alpha):
        x_value = 1.0
        r0 = q_qm
        dissociation = curvature / (2.0 * alpha * alpha)
    else:
        ratio = curvature / (alpha * gradient)
        x_value = (ratio + 1.0) / (ratio + 2.0)
        denominator = 2.0 * alpha * x_value * (1.0 - x_value)
        dissociation = gradient / denominator if abs(denominator) > 1.0e-14 else -1.0
        r0 = q_qm + math.log(x_value) / alpha if x_value > 0.0 else -1.0
    if dissociation <= 0.0 or r0 <= 0.0 or not np.isfinite((dissociation, r0)).all():
        return (
            "harmonic_fallback",
            (("R0", harmonic_q0), ("K", curvature)),
            "MORSE_LOCAL_MATCH_OUTSIDE_PHYSICAL_DOMAIN_FALLBACK",
            "HrmStr1",
        )
    return (
        "morse",
        (
            ("R0", r0),
            ("D", dissociation),
            ("ALPHA", alpha),
            ("BOND_ORDER", float(np.mean(bond_orders))),
        ),
        "ALPHA_FROM_SYNTHON_COVALENT_RADII_AND_BOND_ORDER;D_R0_MATCH_LOCAL_G_AND_H",
        "MrsStr1",
    )


def _morse_fixed_depth_local_match(
    depth: float,
    reference: float,
    gradient: float,
    curvature: float,
) -> tuple[float, float, float] | None:
    """Keep D_e and match the local first and second derivatives exactly."""

    if not (
        math.isfinite(depth)
        and depth > 0.0
        and math.isfinite(reference)
        and reference > 0.0
        and math.isfinite(gradient)
        and math.isfinite(curvature)
        and curvature > 0.0
    ):
        return None
    if abs(gradient) <= 1.0e-12 * max(1.0, curvature):
        return depth, reference, math.sqrt(curvature / (2.0 * depth))
    reduced_gradient = gradient / (2.0 * depth)
    reduced_curvature = curvature / (2.0 * depth)
    target = reduced_gradient**2 / reduced_curvature

    def equation(exponent: float) -> float:
        return exponent * (1.0 - exponent) ** 2 / (2.0 * exponent - 1.0)

    if target <= 0.0 or not math.isfinite(target):
        return None
    if gradient > 0.0:
        lower, upper = 0.5 + 1.0e-12, 1.0 - 1.0e-12
        for _ in range(120):
            middle = 0.5 * (lower + upper)
            if equation(middle) > target:
                lower = middle
            else:
                upper = middle
    else:
        lower, upper = 1.0 + 1.0e-12, 2.0
        while equation(upper) < target and upper < 1.0e8:
            upper *= 2.0
        if upper >= 1.0e8:
            return None
        for _ in range(120):
            middle = 0.5 * (lower + upper)
            if equation(middle) < target:
                lower = middle
            else:
                upper = middle
    exponent = 0.5 * (lower + upper)
    alpha = reduced_gradient / (exponent * (1.0 - exponent))
    if not math.isfinite(alpha) or alpha <= 0.0:
        return None
    r0 = reference + math.log(exponent) / alpha
    if not math.isfinite(r0) or r0 <= 0.0:
        return None
    return depth, r0, alpha


def _inverse_power_fixed_depth_local_match(
    depth: float,
    reference: float,
    gradient: float,
    curvature: float,
) -> tuple[float, float, float] | None:
    """Keep D_e and match local gradient and curvature with D[1-(R0/R)^p]^2."""

    if not (
        math.isfinite(depth)
        and depth > 0.0
        and math.isfinite(reference)
        and reference > 0.0
        and math.isfinite(gradient)
        and math.isfinite(curvature)
        and curvature > 0.0
    ):
        return None
    stationary_scale = max(1.0, curvature * reference)
    if abs(gradient) <= 1.0e-12 * stationary_scale:
        exponent = reference * math.sqrt(curvature / (2.0 * depth))
        return depth, reference, exponent if exponent > 0.0 else None

    reduced_gradient = gradient * reference / (2.0 * depth)
    reduced_curvature = curvature * reference**2 / (2.0 * depth)
    ratio = reduced_curvature / reduced_gradient

    def state(power: float) -> tuple[float, float] | None:
        denominator = ratio + 2.0 * power + 1.0
        if abs(denominator) <= 1.0e-14:
            return None
        local_ratio = (ratio + power + 1.0) / denominator
        if local_ratio <= 0.0 or (gradient > 0.0 and local_ratio >= 1.0):
            return None
        if gradient < 0.0 and local_ratio <= 1.0:
            return None
        return local_ratio, power * local_ratio * (1.0 - local_ratio) - reduced_gradient

    # Search logarithmically because p ranges from soft single bonds to stiff
    # multiple bonds.  The root nearest the stationary estimate is selected.
    grid = np.geomspace(1.0e-4, 1.0e4, 801)
    brackets: list[tuple[float, float]] = []
    previous: tuple[float, float] | None = None
    for power in grid:
        current = state(float(power))
        if current is None or not math.isfinite(current[1]):
            previous = None
            continue
        if previous is not None and previous[1] * current[1] <= 0.0:
            brackets.append((previous[0], float(power)))
        previous = (float(power), current[1])
    if not brackets:
        return None
    estimate = reference * math.sqrt(curvature / (2.0 * depth))
    lower, upper = min(brackets, key=lambda pair: abs(math.log(math.sqrt(pair[0] * pair[1]) / estimate)))
    for _ in range(100):
        middle = math.sqrt(lower * upper)
        left_state = state(lower)
        middle_state = state(middle)
        if left_state is None or middle_state is None:
            return None
        if left_state[1] * middle_state[1] <= 0.0:
            upper = middle
        else:
            lower = middle
    power = math.sqrt(lower * upper)
    solved = state(power)
    if solved is None:
        return None
    local_ratio = solved[0]
    r0 = reference * local_ratio ** (1.0 / power)
    if not (math.isfinite(r0) and r0 > 0.0 and math.isfinite(power) and power > 0.0):
        return None
    return depth, r0, power


def _angle_function_parameters(
    q_qm: float,
    gradient: float,
    curvature: float,
    harmonic_q0: float,
    form: str,
) -> tuple[str, tuple[tuple[str, float], ...], str, str]:
    if form == "harmonic":
        return (
            "harmonic",
            (("THETA0", harmonic_q0), ("K", curvature)),
            "PIC_HESSIAN_AND_EQUILIBRIUM_REFIT",
            "HrmBnd1",
        )
    sine = math.sin(q_qm)
    cosine = math.cos(q_qm)
    if abs(sine) <= 1.0e-4:
        return (
            "harmonic_fallback",
            (("THETA0", harmonic_q0), ("K", curvature)),
            "COSINE_BEND_SINGULAR_NEAR_LINEAR_FALLBACK",
            "HrmBnd1",
        )
    amplitude = (curvature - gradient * cosine / sine) / (2.0 * sine * sine)
    cosine0 = cosine + gradient / (2.0 * amplitude * sine) if amplitude > 0.0 else 2.0
    if amplitude <= 0.0 or not -1.0 <= cosine0 <= 1.0:
        return (
            "harmonic_fallback",
            (("THETA0", harmonic_q0), ("K", curvature)),
            "COSINE_BEND_LOCAL_MATCH_OUTSIDE_PHYSICAL_DOMAIN_FALLBACK",
            "HrmBnd1",
        )
    theta0 = math.acos(np.clip(cosine0, -1.0, 1.0))
    gaussian_force = amplitude * math.sin(theta0) ** 2
    return (
        "cosine",
        (("THETA0", theta0), ("A", amplitude), ("GAUSSIAN_FORCEC", gaussian_force)),
        "EXACT_LOCAL_GRADIENT_AND_CURVATURE_IN_COS_THETA",
        "HrmBnd2",
    )


def _functional_equilibrium_value(
    parameter: PrimitiveFunctionalParameter | None,
    fallback: float,
) -> float:
    if parameter is None:
        return float(fallback)
    values = dict(parameter.parameters)
    for name in ("R0", "THETA0", "Q0"):
        if name in values:
            return float(values[name])
    return float(fallback)


def _refine_parameter_types_for_functional_equilibrium(
    primitives: Sequence[Primitive],
    reference_values: np.ndarray,
    b_matrix: np.ndarray,
    typed_force_constants: np.ndarray,
    primitive_types: tuple[PrimitiveMMType, ...],
    primitive_type_ids: tuple[str, ...],
    sparse_hessian: np.ndarray,
    periodicities: tuple[tuple[int, int], ...],
    nonbonded_gradient_angstrom: np.ndarray,
    synthons,
    policy: PrimitiveFunctionalPolicy,
    *,
    tolerance: float = 1.0e-8,
) -> tuple[
    tuple[PrimitiveMMType, ...],
    tuple[str, ...],
    np.ndarray,
    int,
    float,
    float,
    np.ndarray,
    tuple[PrimitiveFunctionalParameter, ...],
    tuple[TorsionFourierParameter, ...],
]:
    """Split only types whose shared nonlinear function breaks force balance."""
    current_ids = primitive_type_ids
    current = _evaluate_functional_equilibrium_fit(
        primitives,
        reference_values,
        b_matrix,
        typed_force_constants,
        primitive_types,
        current_ids,
        sparse_hessian,
        periodicities,
        nonbonded_gradient_angstrom,
        synthons,
        policy,
    )
    refinement_steps = 0
    while current[0] > tolerance:
        best = None
        fitted_types = current[3]
        for type_index, primitive_type in enumerate(fitted_types):
            if len(primitive_type.primitive_indices) <= 1:
                continue
            for primitive_index in primitive_type.primitive_indices:
                candidate_types, candidate_ids = _detach_parameter_type_member(
                    fitted_types,
                    type_index,
                    primitive_index,
                    reference_values,
                )
                candidate = _evaluate_functional_equilibrium_fit(
                    primitives,
                    reference_values,
                    b_matrix,
                    typed_force_constants,
                    candidate_types,
                    candidate_ids,
                    sparse_hessian,
                    periodicities,
                    nonbonded_gradient_angstrom,
                    synthons,
                    policy,
                )
                if best is None or candidate[0] < best[0][0]:
                    best = (candidate, candidate_ids)
        if best is None or best[0][0] >= current[0] - max(1.0e-14, 1.0e-8 * current[0]):
            break
        current, current_ids = best
        refinement_steps += 1
    rms, maximum, q0, fitted_types, primitive_gradient, functions, torsions = current
    return (
        fitted_types,
        current_ids,
        q0,
        refinement_steps,
        rms,
        maximum,
        primitive_gradient,
        functions,
        torsions,
    )


def _evaluate_functional_equilibrium_fit(
    primitives: Sequence[Primitive],
    reference_values: np.ndarray,
    b_matrix: np.ndarray,
    typed_force_constants: np.ndarray,
    primitive_types: tuple[PrimitiveMMType, ...],
    primitive_type_ids: tuple[str, ...],
    sparse_hessian: np.ndarray,
    periodicities: tuple[tuple[int, int], ...],
    nonbonded_gradient_angstrom: np.ndarray,
    synthons,
    policy: PrimitiveFunctionalPolicy,
) -> tuple[
    float,
    float,
    np.ndarray,
    tuple[PrimitiveMMType, ...],
    np.ndarray,
    tuple[PrimitiveFunctionalParameter, ...],
    tuple[TorsionFourierParameter, ...],
]:
    q0, fitted_types, _iterations, _rms, _maximum, primitive_gradient = (
        _fit_typed_equilibrium_parameters(
            primitives,
            reference_values,
            b_matrix,
            sparse_hessian,
            primitive_types,
            primitive_type_ids,
            nonbonded_gradient_angstrom,
        )
    )
    # Rebalance the primitive first derivatives after nonlinear function
    # construction.  This is needed because nonlinear centres can differ
    # from the provisional harmonic centres used by the typed fit.
    functions = _primitive_functional_parameters(
        primitives, fitted_types, reference_values, primitive_gradient,
        synthons, policy,
    )
    torsions = _torsion_fourier_parameters(
        fitted_types, periodicities, reference_values, primitive_gradient, policy
    )
    function_by_type = {item.primitive_type_id: item for item in functions}
    torsion_by_type = {item.primitive_type_id: item for item in torsions}
    actual_gradient = np.zeros(len(primitives), dtype=float)
    for index, (primitive, type_id, q_value) in enumerate(
        zip(primitives, primitive_type_ids, reference_values, strict=True)
    ):
        if primitive.kind == "dihedral":
            _energy, gradient, _curvature = evaluate_torsion_fourier_parameter(
                torsion_by_type[type_id], q_value
            )
        else:
            _energy, gradient, _curvature = evaluate_primitive_functional_parameter(
                function_by_type[type_id], q_value
            )
        actual_gradient[index] = gradient
    # The coupling force must be evaluated about the same nonlinear
    # equilibrium values that define the runtime primitive functions.  The
    # typed harmonic fit above supplies provisional centres; Morse/cosine and
    # Fourier construction can move those centres while preserving the local
    # gradient/curvature.  Using the provisional centres here leaves a
    # spurious residual in the total Cartesian gradient.
    runtime_equilibrium = np.asarray(
        [
            (
                torsion_by_type[type_id].equilibrium_phase
                if primitive.kind == "dihedral" and type_id in torsion_by_type
                else _functional_equilibrium_value(
                    function_by_type.get(type_id), q0[index]
                )
            )
            for index, (primitive, type_id) in enumerate(
                zip(primitives, primitive_type_ids, strict=True)
            )
        ],
        dtype=float,
    )
    displacement = _primitive_equilibrium_displacements(
        primitives,
        reference_values,
        runtime_equilibrium,
    )
    off_diagonal = np.asarray(sparse_hessian, dtype=float).copy()
    np.fill_diagonal(off_diagonal, 0.0)
    actual_gradient += off_diagonal @ displacement
    residual = b_matrix.T @ actual_gradient + np.asarray(
        nonbonded_gradient_angstrom,
        dtype=float,
    ).reshape(-1)
    for _ in range(500):
        if np.max(np.abs(residual), initial=0.0) <= 1.0e-10:
            break
        # Least-norm primitive-gradient correction that cancels the Cartesian
        # residual while retaining the typed coordinate support.
        correction = np.linalg.lstsq(b_matrix.T, -residual, rcond=1.0e-10)[0]
        primitive_gradient = np.asarray(primitive_gradient, dtype=float) + correction
        functions = _primitive_functional_parameters(
            primitives, fitted_types, reference_values, primitive_gradient,
            synthons, policy,
        )
        torsions = _torsion_fourier_parameters(
            fitted_types, periodicities, reference_values, primitive_gradient, policy
        )
        function_by_type = {item.primitive_type_id: item for item in functions}
        torsion_by_type = {item.primitive_type_id: item for item in torsions}
        actual_gradient = np.asarray(primitive_gradient, dtype=float).copy()
        for index, (primitive, type_id, q_value) in enumerate(
            zip(primitives, primitive_type_ids, reference_values, strict=True)
        ):
            if primitive.kind == "dihedral":
                _energy, actual_gradient[index], _curvature = evaluate_torsion_fourier_parameter(
                    torsion_by_type[type_id], q_value
                )
            else:
                _energy, actual_gradient[index], _curvature = evaluate_primitive_functional_parameter(
                    function_by_type[type_id], q_value
                )
        runtime_equilibrium = np.asarray(
            [
                (
                    torsion_by_type[type_id].equilibrium_phase
                    if primitive.kind == "dihedral" and type_id in torsion_by_type
                    else _functional_equilibrium_value(function_by_type.get(type_id), q0[index])
                )
                for index, (primitive, type_id) in enumerate(
                    zip(primitives, primitive_type_ids, strict=True)
                )
            ],
            dtype=float,
        )
        displacement = _primitive_equilibrium_displacements(
            primitives, reference_values, runtime_equilibrium
        )
        actual_gradient += off_diagonal @ displacement
        residual = b_matrix.T @ actual_gradient + np.asarray(
            nonbonded_gradient_angstrom, dtype=float
        ).reshape(-1)
    rms = float(np.sqrt(np.mean(residual * residual))) if residual.size else 0.0
    maximum = float(np.max(np.abs(residual))) if residual.size else 0.0
    return rms, maximum, q0, fitted_types, primitive_gradient, functions, torsions


def _refine_parameter_types_for_equilibrium(
    primitives: Sequence[Primitive],
    reference_values: np.ndarray,
    b_matrix: np.ndarray,
    typed_force_constants: np.ndarray,
    primitive_types: tuple[PrimitiveMMType, ...],
    primitive_type_ids: tuple[str, ...],
    nonbonded_gradient_angstrom: np.ndarray,
    *,
    tolerance: float = 1.0e-8,
) -> tuple[
    tuple[PrimitiveMMType, ...],
    tuple[str, ...],
    np.ndarray,
    int,
    int,
    float,
    float,
    np.ndarray,
]:
    """Split only parameter types that obstruct exact equilibrium force balance."""
    diagonal_hessian = np.diag(np.asarray(typed_force_constants, dtype=float))
    current_types = primitive_types
    current_ids = primitive_type_ids
    fit = _fit_typed_equilibrium_parameters(
        primitives,
        reference_values,
        b_matrix,
        diagonal_hessian,
        current_types,
        current_ids,
        nonbonded_gradient_angstrom,
    )
    q0, fitted_types, fit_iterations, rms, maximum, primitive_gradient = fit
    refinement_steps = 0
    while rms > tolerance:
        best = None
        for type_index, primitive_type in enumerate(fitted_types):
            if len(primitive_type.primitive_indices) <= 1:
                continue
            for primitive_index in primitive_type.primitive_indices:
                candidate_types, candidate_ids = _detach_parameter_type_member(
                    fitted_types,
                    type_index,
                    primitive_index,
                    reference_values,
                )
                candidate_fit = _fit_typed_equilibrium_parameters(
                    primitives,
                    reference_values,
                    b_matrix,
                    diagonal_hessian,
                    candidate_types,
                    candidate_ids,
                    nonbonded_gradient_angstrom,
                )
                candidate_rms = candidate_fit[3]
                if best is None or candidate_rms < best[0]:
                    best = (candidate_rms, candidate_types, candidate_ids, candidate_fit)
        if best is None or best[0] >= rms - max(1.0e-14, 1.0e-8 * rms):
            break
        rms, current_types, current_ids, fit = best
        q0, fitted_types, fit_iterations, rms, maximum, primitive_gradient = fit
        refinement_steps += 1
    return (
        fitted_types,
        current_ids,
        q0,
        fit_iterations,
        refinement_steps,
        rms,
        maximum,
        primitive_gradient,
    )


def _detach_parameter_type_member(
    primitive_types: tuple[PrimitiveMMType, ...],
    target_type_index: int,
    primitive_index: int,
    reference_values: np.ndarray,
) -> tuple[tuple[PrimitiveMMType, ...], tuple[str, ...]]:
    groups: list[tuple[str, tuple[str, ...], tuple[int, ...], float, float]] = []
    for type_index, primitive_type in enumerate(primitive_types):
        if type_index != target_type_index:
            groups.append(
                (
                    primitive_type.kind,
                    primitive_type.atomic_type_signature,
                    primitive_type.primitive_indices,
                    primitive_type.equilibrium_value,
                    primitive_type.force_constant,
                )
            )
            continue
        remaining = tuple(
            index for index in primitive_type.primitive_indices if index != primitive_index
        )
        for indices in (remaining, (primitive_index,)):
            if not indices:
                continue
            q0 = _mean_equilibrium(
                primitive_type.kind,
                np.asarray(reference_values, dtype=float)[list(indices)],
            )
            groups.append(
                (
                    primitive_type.kind,
                    primitive_type.atomic_type_signature,
                    indices,
                    q0,
                    primitive_type.force_constant,
                )
            )
    result: list[PrimitiveMMType] = []
    ids = [""] * len(reference_values)
    for type_index, (kind, signature, indices, q0, force_constant) in enumerate(groups, start=1):
        identifier = f"P{type_index:03d}"
        result.append(PrimitiveMMType(identifier, kind, signature, indices, q0, force_constant))
        for index in indices:
            ids[index] = identifier
    return tuple(result), tuple(ids)


def _fit_typed_equilibrium_parameters(
    primitives: Sequence[Primitive],
    reference_values: np.ndarray,
    b_matrix: np.ndarray,
    sparse_hessian: np.ndarray,
    primitive_types: tuple[PrimitiveMMType, ...],
    primitive_type_ids: tuple[str, ...],
    nonbonded_gradient_angstrom: np.ndarray,
) -> tuple[np.ndarray, tuple[PrimitiveMMType, ...], int, float, float, np.ndarray]:
    """Refit typed equilibrium parameters so the total Cartesian force vanishes.

    The QM geometry supplies the initial values only.  At that geometry the
    bonded force must cancel the analytic electrostatic/UFF force.  Parameters
    are optimized in type space, so all primitives assigned to one type keep a
    common R0, theta0 or Fourier coefficients.  Non-torsional variables solve
    directly for the typed equilibrium value.  A torsional type instead solves
    for its bonded first derivative at q_QM; that derivative and the residual
    Hessian curvature uniquely determine the cosine/sine coefficients later.
    Retained cross terms are centered at the fitted equilibrium vector and
    therefore enter the same force-balance system as the diagonal functions.
    """
    if not primitive_types:
        gradient = np.asarray(nonbonded_gradient_angstrom, dtype=float)
        rms = float(np.sqrt(np.mean(gradient * gradient))) if gradient.size else 0.0
        maximum = float(np.max(np.abs(gradient))) if gradient.size else 0.0
        return np.zeros(0), primitive_types, 0, rms, maximum, np.zeros(0)
    type_position = {item.identifier: index for index, item in enumerate(primitive_types)}
    transform = np.zeros((len(primitives), len(primitive_types)), dtype=float)
    for primitive_index, type_id in enumerate(primitive_type_ids):
        transform[primitive_index, type_position[type_id]] = 1.0
    reference = np.asarray(reference_values, dtype=float)
    gradient_nb = np.asarray(nonbonded_gradient_angstrom, dtype=float).reshape(-1)
    equilibrium_hessian = np.asarray(sparse_hessian, dtype=float)
    if equilibrium_hessian.shape != (len(primitives), len(primitives)):
        raise ValueError("PIC equilibrium Hessian shape differs from primitive count")
    diagonal_curvature = np.diag(equilibrium_hessian)
    fixed_equilibrium = np.zeros(len(primitives), dtype=float)
    for primitive_index, primitive in enumerate(primitives):
        if primitive.kind == "dihedral":
            fixed_equilibrium[primitive_index] = reference[primitive_index]
    force_offset = equilibrium_hessian @ (reference - fixed_equilibrium)
    force_parameter_map = np.zeros_like(transform)
    initial_parameters = np.zeros(len(primitive_types), dtype=float)
    for type_index, primitive_type in enumerate(primitive_types):
        if primitive_type.kind != "dihedral":
            initial_parameters[type_index] = primitive_type.equilibrium_value
    for type_index, primitive_type in enumerate(primitive_types):
        membership = transform[:, type_index]
        if primitive_type.kind == "dihedral":
            # A Fourier type is parameterized by its diagonal first derivative
            # at the QM geometry; cross-coupling gradients are handled by the
            # equilibrium displacements of the other coordinates.
            force_parameter_map[:, type_index] = membership
        else:
            force_parameter_map[:, type_index] = -(equilibrium_hessian @ membership)
    cartesian_parameter_map = b_matrix.T @ force_parameter_map
    parameter_scales = np.asarray(
        [
            _equilibrium_parameter_scale(item)
            for item in primitive_types
        ],
        dtype=float,
    )
    initial_primitive_gradient = force_offset + force_parameter_map @ initial_parameters
    initial_residual = b_matrix.T @ initial_primitive_gradient + gradient_nb
    if float(np.linalg.norm(initial_residual)) <= 1.0e-14:
        parameters = initial_parameters
        iterations = 0
    else:
        # Redundant PICs admit infinitely many equilibrium-parameter shifts.
        # Select the canonical minimum displacement in chemical units rather
        # than the raw Euclidean minimum, which can place very large changes on
        # weak angular coordinates merely because radians and angstroms have
        # different numerical scales.
        scaled_map = cartesian_parameter_map * parameter_scales[None, :]
        scaled_change = np.linalg.lstsq(scaled_map, -initial_residual, rcond=1.0e-10)[0]
        change = parameter_scales * scaled_change
        parameters = initial_parameters + change
        iterations = 1
    total_primitive_gradient = force_offset + force_parameter_map @ parameters
    residual = b_matrix.T @ total_primitive_gradient + gradient_nb
    q0 = np.asarray(reference, dtype=float).copy()
    for primitive_index, primitive in enumerate(primitives):
        if primitive.kind != "dihedral":
            q0[primitive_index] = parameters[type_position[primitive_type_ids[primitive_index]]]
    functional_gradient = np.zeros(len(primitives), dtype=float)
    for primitive_index, primitive in enumerate(primitives):
        type_index = type_position[primitive_type_ids[primitive_index]]
        if primitive.kind == "dihedral":
            functional_gradient[primitive_index] = parameters[type_index]
        else:
            functional_gradient[primitive_index] = diagonal_curvature[primitive_index] * (
                reference[primitive_index] - parameters[type_index]
            )
    updated_types = tuple(
        PrimitiveMMType(
            item.identifier,
            item.kind,
            item.atomic_type_signature,
            item.primitive_indices,
            (item.equilibrium_value if item.kind == "dihedral" else float(parameters[index])),
            item.force_constant,
        )
        for index, item in enumerate(primitive_types)
    )
    final_rms = float(np.sqrt(np.mean(residual * residual))) if residual.size else 0.0
    final_max = float(np.max(np.abs(residual))) if residual.size else 0.0
    return q0, updated_types, iterations, final_rms, final_max, functional_gradient


def _equilibrium_parameter_scale(primitive_type: PrimitiveMMType) -> float:
    """Chemical scale for the canonical redundant-PIC stationarity gauge."""

    kind = primitive_type.kind
    if kind in {"bond", "hbond_dist", "pseudo_bond"}:
        return 0.05  # angstrom
    if kind == "dihedral":
        # The fitted variable is the local first derivative, not a phase.
        return max(abs(float(primitive_type.force_constant)) * math.radians(10.0), 1.0e-5)
    if kind in {"angle", "linear_bend", "out_of_plane"}:
        return math.radians(5.0)
    return 0.10


def _primitive_equilibrium_displacements(
    primitives: Sequence[Primitive],
    reference_values: np.ndarray,
    equilibrium_values: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [
            _primitive_displacement(primitive.kind, reference, equilibrium)
            for primitive, reference, equilibrium in zip(
                primitives,
                reference_values,
                equilibrium_values,
                strict=True,
            )
        ],
        dtype=float,
    )
