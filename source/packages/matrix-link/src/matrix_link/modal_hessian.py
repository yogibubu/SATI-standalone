"""Energy/gradient Hessians acquired along a lower-level normal-mode basis."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from matrix_chem import (
    MolecularGeometry,
    SymmetryThresholds,
    analyze_molecular_symmetry,
    atomic_mass,
    expected_vibrational_mode_count,
    is_linear_geometry,
    read_xyzin_geometry,
)
from matrix_chem.symmetry import symmetrize_cartesian_gradient
from matrix_gf import cartesian_normal_modes_from_hessian
from matrix_qm.sections import read_normal_modes_section
from matrix_chem.topology.elements import atomic_number

from .optimizer import (
    ElectronicStateResolutionError,
    GeometryEvaluationService,
    OptimizerCoordinateModel,
    OptimizerSettings,
)
from .scan import BOHR_TO_ANGSTROM, QMScanBackend


MODAL_HESSIAN_PLAN_SCHEMA = "matrix.link.modal_hessian_plan.v2"
MODAL_HESSIAN_RESULT_SCHEMA = "matrix.link.modal_hessian_result.v2"
MODAL_PARITY_PROTOCOL = "matrix.link.modal_parity.v1"
NORMAL_MODE_BASIS_PROTOCOL = "matrix.link.mass_weighted_normal_modes.v1"
MODE_PARITY_TOLERANCE = 1.0e-7
MODE_PARITY_AMBIGUITY_TOLERANCE = 1.0e-4
MODE_SYMMETRY_PROJECTION_TOLERANCE = 1.0e-6
MODAL_TARGET_ENERGY_HARTREE = 1.0e-4
MODAL_MIN_STEP_SQRT_AMU_BOHR = 0.02
MODAL_MAX_STEP_SQRT_AMU_BOHR = 0.20
MODAL_MAX_ATOM_DISPLACEMENT_ANGSTROM = 0.04


@dataclass(frozen=True)
class ModalHessianPoint:
    key: str
    role: str
    mode_indices: tuple[int, ...]
    displacement: tuple[float, ...]


@dataclass(frozen=True)
class ModalModeStencil:
    mode_index: int
    stencil: str
    symmetry_operation: str | None
    parity_residual: float
    operation_rotation: tuple[tuple[float, float, float], ...] | None = None
    operation_permutation: tuple[int, ...] | None = None


@dataclass(frozen=True)
class ModalHessianPlan:
    mode_count: int
    active_mode_indices: tuple[int, ...]
    steps_sqrt_amu_bohr: tuple[float, ...]
    points: tuple[ModalHessianPoint, ...]
    mode_stencils: tuple[ModalModeStencil, ...]
    point_group: str
    symmetry_operation_count: int
    parity_protocol: str = MODAL_PARITY_PROTOCOL
    normal_mode_basis_protocol: str = NORMAL_MODE_BASIS_PROTOCOL
    property_source: str = "energy"
    include_mixed: bool = False
    mixed_stencil: str = "one-sided"
    mode_subspace: str = "complete_vibrational"
    step_policy: str = "fixed"
    curvature_source: str = "none"
    curvatures_hartree_per_q2: tuple[float, ...] = ()
    target_energy_change_hartree: float | None = None
    curvature_floor_hartree_per_q2: float | None = None
    maximum_atom_displacement_angstrom: float | None = None
    linear_molecule: bool = False
    schema: str = MODAL_HESSIAN_PLAN_SCHEMA

    def to_json(self) -> dict[str, object]:
        payload = asdict(self)
        payload.update({"energy_or_gradient_points": len(self.points)})
        return payload


@dataclass(frozen=True)
class ModalHessianConvergenceDiagnostics:
    column_relative_changes: tuple[float, ...]
    maximum_column_relative_change: float
    antisymmetric_relative_residual: float
    status: str
    coarse_scale: float
    fine_scale: float

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def classify_modal_stencils(
    geometry: MolecularGeometry,
    modes: np.ndarray,
) -> tuple[tuple[ModalModeStencil, ...], str, int]:
    """Classify each mode from the molecular symmetry action itself.

    A mode uses one displaced point only when an operation of the retained
    molecular point group maps its +Q Cartesian displacement onto -Q,
    including the associated permutation of indistinguishable atoms.  The
    decision is therefore independent of mode number, frequency ordering,
    irrep labels, backend, and molecule-specific tables.  Ambiguous numerical
    matches fail before any electronic-structure input is generated.
    """
    vectors = np.asarray(modes, dtype=float)
    natoms = len(geometry.atoms)
    if vectors.ndim != 2 or vectors.shape[1] != 3 * natoms:
        raise ValueError("normal modes must have shape mode_count x 3N")
    thresholds = SymmetryThresholds()
    symmetry = analyze_molecular_symmetry(
        geometry,
        distance_tolerance=thresholds.distance_angstrom,
        inertia_tolerance=thresholds.inertia_relative,
        max_rotation_order=thresholds.max_rotation_order,
    )
    stencils: list[ModalModeStencil] = []
    for index, flattened in enumerate(vectors):
        mode = flattened.reshape(natoms, 3)
        scale = float(np.linalg.norm(mode))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"normal mode {index + 1} has invalid norm")
        matches: list[tuple[float, object]] = []
        for operation in symmetry.operations:
            rotation = np.asarray(operation.rotation, dtype=float)
            mapping = np.asarray([int(atom) - 1 for atom in operation.permutation], dtype=int)
            if mapping.shape != (natoms,) or set(mapping.tolist()) != set(range(natoms)):
                raise ValueError(
                    f"invalid atom permutation in symmetry operation {operation.label}"
                )
            inverse_mapping = np.argsort(mapping)
            transformed = mode[inverse_mapping] @ rotation
            residual = float(np.linalg.norm(transformed + mode) / scale)
            matches.append((residual, operation))
        best_residual, best_operation = min(matches, key=lambda item: item[0])
        if MODE_PARITY_TOLERANCE < best_residual < MODE_PARITY_AMBIGUITY_TOLERANCE:
            raise ValueError(
                f"normal mode {index + 1} has ambiguous +Q/-Q symmetry equivalence "
                f"(best residual {best_residual:.3e})"
            )
        equivalent = best_residual <= MODE_PARITY_TOLERANCE
        stencils.append(
            ModalModeStencil(
                mode_index=index,
                stencil="symmetry-one-sided" if equivalent else "central",
                symmetry_operation=(str(best_operation.label) if equivalent else None),
                parity_residual=best_residual,
                operation_rotation=(
                    tuple(tuple(float(value) for value in row) for row in best_operation.rotation)
                    if equivalent
                    else None
                ),
                operation_permutation=(
                    tuple(int(atom) for atom in best_operation.permutation) if equivalent else None
                ),
            )
        )
    return tuple(stencils), str(symmetry.point_group), len(symmetry.operations)


def totally_symmetric_mode_indices(
    geometry: MolecularGeometry,
    modes: np.ndarray,
) -> tuple[int, ...]:
    """Return lower-level normal modes in the totally symmetric subspace.

    MATRIX-CHEM already owns the Cartesian group-average projector used for
    gradient symmetrization.  Applying that same projector to each stored
    mass-weighted mode avoids a second irrep-classification implementation and
    gives the exact active subspace needed by a symmetry-constrained minimum
    optimization.  Ambiguous numerical projections fail before any QM point
    is generated.
    """

    vectors = np.asarray(modes, dtype=float)
    natoms = len(geometry.atoms)
    if vectors.ndim != 2 or vectors.shape[1] != 3 * natoms:
        raise ValueError("normal modes must have shape mode_count x 3N")
    thresholds = SymmetryThresholds()
    symmetry = analyze_molecular_symmetry(
        geometry,
        distance_tolerance=thresholds.distance_angstrom,
        inertia_tolerance=thresholds.inertia_relative,
        max_rotation_order=thresholds.max_rotation_order,
    )
    selected: list[int] = []
    for index, vector in enumerate(vectors):
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError(f"normal mode {index + 1} has invalid norm")
        projected = np.asarray(
            symmetrize_cartesian_gradient(vector, symmetry),
            dtype=float,
        ).reshape(-1)
        residual = float(np.linalg.norm(projected - vector) / norm)
        if (
            MODE_SYMMETRY_PROJECTION_TOLERANCE
            < residual
            < MODE_PARITY_AMBIGUITY_TOLERANCE
        ):
            raise ValueError(
                f"normal mode {index + 1} has ambiguous totally-symmetric projection "
                f"(residual {residual:.3e})"
            )
        if residual <= MODE_SYMMETRY_PROJECTION_TOLERANCE:
            selected.append(index)
    if not selected:
        raise ValueError("normal-mode basis contains no totally symmetric vibration")
    return tuple(selected)


def _symmetry_transform_cartesian(
    vector: np.ndarray,
    stencil: ModalModeStencil,
) -> np.ndarray:
    if stencil.operation_rotation is None or stencil.operation_permutation is None:
        raise ValueError(
            f"mode {stencil.mode_index + 1} has no operation for symmetry reconstruction"
        )
    cartesian = np.asarray(vector, dtype=float).reshape((-1, 3))
    rotation = np.asarray(stencil.operation_rotation, dtype=float)
    mapping = np.asarray([int(atom) - 1 for atom in stencil.operation_permutation], dtype=int)
    inverse_mapping = np.argsort(mapping)
    return (cartesian[inverse_mapping] @ rotation).reshape(-1)


def plan_modal_hessian(
    xyzin_path: Path | str,
    *,
    step_sqrt_amu_bohr: float = 0.10,
    property_source: str = "energy",
    include_mixed: bool = False,
    mixed_stencil: str = "one-sided",
    curvatures_hartree_per_q2: tuple[float, ...] | list[float] | np.ndarray | None = None,
    curvature_floor_hartree_per_q2: float | None = None,
    curvature_source: str = "lower_level_normal_modes",
    target_energy_change_hartree: float = MODAL_TARGET_ENERGY_HARTREE,
    minimum_step_sqrt_amu_bohr: float = MODAL_MIN_STEP_SQRT_AMU_BOHR,
    maximum_step_sqrt_amu_bohr: float = MODAL_MAX_STEP_SQRT_AMU_BOHR,
    maximum_atom_displacement_angstrom: float = MODAL_MAX_ATOM_DISPLACEMENT_ANGSTROM,
    step_scale: float = 1.0,
    totally_symmetric_only: bool = False,
) -> tuple[ModalHessianPlan, np.ndarray, np.ndarray]:
    """Build the symmetry-preserving normal-mode finite-difference grid.

    The ``#NORMAL_MODES`` section is a lower-level reference and must contain
    exactly the complete 3N-6 (nonlinear) or 3N-5 (linear) set of
    mass-weighted, orthonormal modes.  The finite-difference
    coordinate is Q in sqrt(amu)*bohr; LINK converts it to Cartesian points.
    """
    if step_sqrt_amu_bohr <= 0.0 or step_scale <= 0.0:
        raise ValueError("normal-mode step must be positive")
    source = str(property_source).strip().casefold().replace("_", "-")
    if source not in {"energy", "cartesian-gradient"}:
        raise ValueError("property_source must be 'energy' or 'cartesian-gradient'")
    if str(mixed_stencil).strip().casefold() != "one-sided":
        raise ValueError("mixed_stencil must be 'one-sided'")
    geometry = read_xyzin_geometry(Path(xyzin_path))
    section = read_normal_modes_section(Path(xyzin_path))
    modes = np.asarray(section.modes, dtype=float)
    linear = is_linear_geometry(np.asarray(geometry.coordinates_angstrom, dtype=float))
    expected = expected_vibrational_mode_count(
        np.asarray(geometry.coordinates_angstrom, dtype=float)
    )
    if modes.shape != (expected, 3 * len(geometry.atoms)):
        raise ValueError(
            f"#NORMAL_MODES must contain shape ({expected}, {3 * len(geometry.atoms)})"
        )
    if not np.all(np.isfinite(modes)):
        raise ValueError("#NORMAL_MODES contains non-finite values")
    gram = modes @ modes.T
    if not np.allclose(gram, np.eye(expected), atol=1.0e-6, rtol=1.0e-6):
        raise ValueError("#NORMAL_MODES must be orthonormal mass-weighted modes")
    masses = np.asarray([atomic_mass(int(atomic_number(atom))) for atom in geometry.atoms])
    if np.any(~np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("invalid atomic masses for normal-mode Hessian")
    if curvatures_hartree_per_q2 is None:
        steps = np.full(expected, float(step_sqrt_amu_bohr))
        curvature_values = np.array((), dtype=float)
        step_policy = "fixed"
        active_curvature_floor = None
        active_target_energy = None
        active_maximum_atom_displacement = None
        active_curvature_source = "none"
    else:
        curvature_values = np.asarray(curvatures_hartree_per_q2, dtype=float).reshape(-1)
        if curvature_values.shape != (expected,) or not np.all(np.isfinite(curvature_values)):
            raise ValueError("adaptive modal curvatures must contain one finite value per mode")
        if curvature_floor_hartree_per_q2 is None:
            raise ValueError(
                "adaptive modal steps require an explicitly calibrated curvature floor"
            )
        floor = float(curvature_floor_hartree_per_q2)
        target_energy = float(target_energy_change_hartree)
        minimum_step = float(minimum_step_sqrt_amu_bohr)
        maximum_step = float(maximum_step_sqrt_amu_bohr)
        maximum_atom = float(maximum_atom_displacement_angstrom)
        if (
            floor <= 0.0
            or target_energy <= 0.0
            or minimum_step <= 0.0
            or maximum_step < minimum_step
            or maximum_atom <= 0.0
        ):
            raise ValueError("invalid adaptive modal displacement parameters")
        steps = np.sqrt(2.0 * target_energy / np.maximum(np.abs(curvature_values), floor))
        steps = np.clip(steps, minimum_step, maximum_step)
        mass_weighted_modes = modes.reshape(expected, len(geometry.atoms), 3)
        cartesian_per_q = mass_weighted_modes / np.sqrt(masses)[None, :, None]
        maximum_per_mode = (
            np.max(np.linalg.norm(cartesian_per_q, axis=2), axis=1) * BOHR_TO_ANGSTROM
        )
        atom_caps = np.divide(
            maximum_atom,
            maximum_per_mode,
            out=np.full(expected, np.inf),
            where=maximum_per_mode > np.finfo(float).tiny,
        )
        steps = np.minimum(steps, atom_caps)
        if np.any(steps <= 0.0) or not np.all(np.isfinite(steps)):
            raise ValueError("adaptive modal Cartesian cap produced an invalid step")
        step_policy = "curvature_energy_and_cartesian_cap"
        active_curvature_floor = floor
        active_target_energy = target_energy
        active_maximum_atom_displacement = maximum_atom
        active_curvature_source = str(curvature_source).strip()
        if not active_curvature_source:
            raise ValueError("adaptive modal steps require curvature provenance")
    steps *= float(step_scale)
    mode_stencils, point_group, operation_count = classify_modal_stencils(geometry, modes)
    active_mode_indices = (
        totally_symmetric_mode_indices(geometry, modes)
        if totally_symmetric_only
        else tuple(range(expected))
    )
    zero = np.zeros(expected)
    points = [ModalHessianPoint("center", "center", (), tuple(zero))]
    for index in active_mode_indices:
        stencil = mode_stencils[index]
        signs = (
            ((1.0, "plus"),)
            if stencil.stencil == "symmetry-one-sided"
            else ((-1.0, "minus"), (1.0, "plus"))
        )
        for sign, suffix in signs:
            displacement = zero.copy()
            displacement[index] = sign * steps[index]
            points.append(
                ModalHessianPoint(
                    f"q{index + 1:03d}-{suffix}",
                    f"diagonal-{suffix}",
                    (index,),
                    tuple(displacement),
                )
            )
    if source == "energy" and include_mixed:
        for left_offset, left in enumerate(active_mode_indices):
            for right in active_mode_indices[left_offset + 1 :]:
                displacement = zero.copy()
                displacement[left] = steps[left]
                displacement[right] = steps[right]
                points.append(
                    ModalHessianPoint(
                        f"q{left + 1:03d}-q{right + 1:03d}-plus-plus",
                        "mixed-plus-plus",
                        (left, right),
                        tuple(displacement),
                    )
                )
    return (
        ModalHessianPlan(
            mode_count=expected,
            active_mode_indices=active_mode_indices,
            steps_sqrt_amu_bohr=tuple(steps),
            points=tuple(points),
            mode_stencils=mode_stencils,
            point_group=point_group,
            symmetry_operation_count=operation_count,
            parity_protocol=MODAL_PARITY_PROTOCOL,
            property_source=source,
            include_mixed=bool(include_mixed),
            mixed_stencil=str(mixed_stencil),
            mode_subspace=(
                "totally_symmetric" if totally_symmetric_only else "complete_vibrational"
            ),
            step_policy=step_policy,
            curvature_source=active_curvature_source,
            curvatures_hartree_per_q2=tuple(float(value) for value in curvature_values),
            target_energy_change_hartree=active_target_energy,
            curvature_floor_hartree_per_q2=active_curvature_floor,
            maximum_atom_displacement_angstrom=active_maximum_atom_displacement,
            linear_molecule=bool(linear),
        ),
        modes,
        masses,
    )


def modal_hessian_from_energies(plan: ModalHessianPlan, energies: dict[str, float]) -> np.ndarray:
    if plan.property_source != "energy":
        raise ValueError("modal plan does not acquire energies")
    center = float(energies["center"])
    hessian = np.zeros((plan.mode_count, plan.mode_count))
    for index in plan.active_mode_indices:
        step = plan.steps_sqrt_amu_bohr[index]
        stencil = plan.mode_stencils[index]
        plus = float(energies[f"q{index + 1:03d}-plus"])
        if stencil.stencil == "symmetry-one-sided":
            hessian[index, index] = 2.0 * (plus - center) / step**2
        else:
            hessian[index, index] = (
                plus - 2.0 * center + float(energies[f"q{index + 1:03d}-minus"])
            ) / step**2
    for point in plan.points:
        if point.role != "mixed-plus-plus":
            continue
        left, right = point.mode_indices
        hessian[left, right] = hessian[right, left] = (
            float(energies[point.key])
            - float(energies[f"q{left + 1:03d}-plus"])
            - float(energies[f"q{right + 1:03d}-plus"])
            + center
        ) / (plan.steps_sqrt_amu_bohr[left] * plan.steps_sqrt_amu_bohr[right])
    return 0.5 * (hessian + hessian.T)


def modal_hessian_convergence_diagnostics(
    coarse_hessian: np.ndarray,
    fine_hessian: np.ndarray,
    *,
    coarse_scale: float = 1.0,
    fine_scale: float = 0.5,
) -> ModalHessianConvergenceDiagnostics:
    """Compare two complete modal Hessians using the approved provisional gates."""

    coarse = np.asarray(coarse_hessian, dtype=float)
    fine = np.asarray(fine_hessian, dtype=float)
    if coarse.ndim != 2 or coarse.shape[0] != coarse.shape[1] or fine.shape != coarse.shape:
        raise ValueError("modal Hessian convergence requires equal square matrices")
    if not np.all(np.isfinite(coarse)) or not np.all(np.isfinite(fine)):
        raise ValueError("modal Hessian convergence received non-finite values")
    denominator = np.maximum(np.linalg.norm(fine, axis=0), np.finfo(float).tiny)
    changes = np.linalg.norm(fine - coarse, axis=0) / denominator
    maximum_change = float(np.max(changes)) if changes.size else 0.0
    norm = max(float(np.linalg.norm(fine)), np.finfo(float).tiny)
    antisymmetric = float(np.linalg.norm(fine - fine.T) / norm)
    if maximum_change <= 0.02 and antisymmetric <= 1.0e-3:
        status = "accepted"
    elif maximum_change <= 0.05 and antisymmetric <= 5.0e-3:
        status = "warning"
    else:
        status = "refine_or_fail"
    return ModalHessianConvergenceDiagnostics(
        column_relative_changes=tuple(float(value) for value in changes),
        maximum_column_relative_change=maximum_change,
        antisymmetric_relative_residual=antisymmetric,
        status=status,
        coarse_scale=float(coarse_scale),
        fine_scale=float(fine_scale),
    )


def acquire_modal_hessian(
    xyzin_path: Path | str,
    *,
    run_dir: Path | str,
    backend: QMScanBackend,
    workers: int = 1,
    step_sqrt_amu_bohr: float = 0.10,
    property_source: str = "energy",
    include_mixed: bool = False,
    mixed_stencil: str = "one-sided",
    timeout: float | None = None,
    curvatures_hartree_per_q2: tuple[float, ...] | list[float] | np.ndarray | None = None,
    curvature_floor_hartree_per_q2: float | None = None,
    curvature_source: str = "lower_level_normal_modes",
    verify_step_convergence: bool = False,
    totally_symmetric_only: bool = False,
) -> dict[str, object]:
    """Acquire a Hessian in a stored lower-level normal-mode basis."""
    if workers <= 0:
        raise ValueError("normal-mode Hessian workers must be positive")
    source = Path(xyzin_path).expanduser().resolve()
    target = Path(run_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    plan, modes, masses = plan_modal_hessian(
        source,
        step_sqrt_amu_bohr=step_sqrt_amu_bohr,
        property_source=property_source,
        include_mixed=include_mixed,
        mixed_stencil=mixed_stencil,
        curvatures_hartree_per_q2=curvatures_hartree_per_q2,
        curvature_floor_hartree_per_q2=curvature_floor_hartree_per_q2,
        curvature_source=curvature_source,
        totally_symmetric_only=totally_symmetric_only,
    )
    geometry = read_xyzin_geometry(source)
    # Q is in sqrt(amu)*bohr, while the service consumes Cartesian angstrom.
    mode_vectors = modes.reshape(plan.mode_count, len(geometry.atoms), 3)
    directions = mode_vectors / np.sqrt(masses)[None, :, None]
    directions = directions.reshape(plan.mode_count, -1) * BOHR_TO_ANGSTROM
    model = OptimizerCoordinateModel(
        kind="cartesian",
        labels=tuple(f"NM{index + 1:03d}" for index in range(plan.mode_count)),
        directions_angstrom=directions,
        metric_diagonal=np.ones(plan.mode_count),
    )
    service = GeometryEvaluationService(
        xyzin_path=source,
        run_dir=target / "evaluation-service",
        coordinate_model=model,
        backend=backend,
        timeout=timeout,
        settings=OptimizerSettings(resume=True, coordinate_parallel_workers=max(1, int(workers))),
    )

    def build_plan(scale: float) -> ModalHessianPlan:
        active, _active_modes, _active_masses = plan_modal_hessian(
            source,
            step_sqrt_amu_bohr=step_sqrt_amu_bohr,
            property_source=property_source,
            include_mixed=include_mixed,
            mixed_stencil=mixed_stencil,
            curvatures_hartree_per_q2=curvatures_hartree_per_q2,
            curvature_floor_hartree_per_q2=curvature_floor_hartree_per_q2,
            curvature_source=curvature_source,
            step_scale=scale,
            totally_symmetric_only=totally_symmetric_only,
        )
        return active

    def evaluate_complete(active_plan: ModalHessianPlan, scale: float):
        requested = (
            ("energy", "gradient")
            if active_plan.property_source == "cartesian-gradient"
            else ("energy",)
        )

        def evaluate(point: ModalHessianPoint):
            return point, service.evaluate(
                np.asarray(point.displacement),
                tag=f"modal-hessian-s{scale:.6g}-{point.key}",
                requested_properties=requested,
            )

        center_point, center_eval = evaluate(active_plan.points[0])
        service.accept_electronic_state(center_eval)
        active_evaluated = [(center_point, center_eval)]
        with ThreadPoolExecutor(max_workers=int(workers)) as executor:
            active_evaluated.extend(executor.map(evaluate, active_plan.points[1:]))
        active_energies = {
            point.key: float(evaluation.result.energy_hartree)
            for point, evaluation in active_evaluated
        }
        if active_plan.property_source == "energy":
            active_force = modal_hessian_from_energies(active_plan, active_energies)
            return active_evaluated, active_energies, active_force
        gradients: dict[str, np.ndarray] = {}
        for point, evaluation in active_evaluated:
            if evaluation.result.gradient_hartree_per_bohr is None:
                raise RuntimeError(f"backend returned no Cartesian gradient at {point.key}")
            gradients[point.key] = np.asarray(evaluation.result.gradient_hartree_per_bohr).reshape(
                -1
            )
        active_force = np.zeros((active_plan.mode_count, active_plan.mode_count))
        d_bohr = mode_vectors / np.sqrt(masses)[None, :, None]
        projector = d_bohr.reshape(active_plan.mode_count, -1)
        active_indices = np.asarray(active_plan.active_mode_indices, dtype=int)
        active_projector = projector[active_indices]
        projected = {
            key: active_projector @ gradient for key, gradient in gradients.items()
        }
        for index in active_plan.active_mode_indices:
            step = active_plan.steps_sqrt_amu_bohr[index]
            stencil = active_plan.mode_stencils[index]
            plus_key = f"q{index + 1:03d}-plus"
            if stencil.stencil == "symmetry-one-sided":
                reconstructed_minus = _symmetry_transform_cartesian(gradients[plus_key], stencil)
                minus_projection = active_projector @ reconstructed_minus
            else:
                minus_projection = projected[f"q{index + 1:03d}-minus"]
            active_force[active_indices, index] = (
                projected[plus_key] - minus_projection
            ) / (2.0 * step)
        return (
            active_evaluated,
            active_energies,
            0.5 * (active_force + active_force.T),
        )

    scale = 1.0
    maximum_state_halvings = (
        int(backend.state_tracking_max_displacement_halvings)
        if int(backend.electronic_state) > 0
        else 0
    )
    state_halvings = 0
    while True:
        plan = build_plan(scale)
        try:
            evaluated, energies, modal_force = evaluate_complete(plan, scale)
            break
        except ElectronicStateResolutionError:
            if state_halvings >= maximum_state_halvings:
                raise
            state_halvings += 1
            scale *= 0.5

    convergence: list[dict[str, object]] = []
    if verify_step_convergence:
        fine_scale = scale * 0.5
        fine_plan = build_plan(fine_scale)
        fine_evaluated, fine_energies, fine_force = evaluate_complete(fine_plan, fine_scale)
        diagnostic = modal_hessian_convergence_diagnostics(
            modal_force,
            fine_force,
            coarse_scale=scale,
            fine_scale=fine_scale,
        )
        convergence.append(diagnostic.to_json())
        plan, evaluated, energies, modal_force = (
            fine_plan,
            fine_evaluated,
            fine_energies,
            fine_force,
        )
        scale = fine_scale
        if diagnostic.status == "refine_or_fail":
            final_scale = scale * 0.5
            final_plan = build_plan(final_scale)
            final_evaluated, final_energies, final_force = evaluate_complete(
                final_plan, final_scale
            )
            final_diagnostic = modal_hessian_convergence_diagnostics(
                modal_force,
                final_force,
                coarse_scale=scale,
                fine_scale=final_scale,
            )
            convergence.append(final_diagnostic.to_json())
            if final_diagnostic.status == "refine_or_fail":
                raise RuntimeError(
                    "modal Hessian has no finite-difference plateau after one further halving"
                )
            plan, evaluated, energies, modal_force = (
                final_plan,
                final_evaluated,
                final_energies,
                final_force,
            )
            scale = final_scale
    sqrt_m = np.sqrt(np.repeat(masses, 3))
    cartesian_hessian = (modes.T * sqrt_m[:, None]) @ modal_force @ (modes * sqrt_m[None, :])
    normal = cartesian_normal_modes_from_hessian(
        cartesian_hessian,
        masses,
        np.asarray(geometry.coordinates_angstrom, dtype=float) / BOHR_TO_ANGSTROM,
    )
    payload = {
        "schema": MODAL_HESSIAN_RESULT_SCHEMA,
        "xyzin": str(source),
        "plan": plan.to_json(),
        "normal_mode_source": read_normal_modes_section(source).source,
        "backend": backend.name,
        "method": backend.method,
        "basis": backend.basis,
        "energies_hartree": energies,
        "frequencies_cm-1": normal.frequencies_cm.tolist(),
        "modal_force_constants": modal_force.tolist(),
        "cartesian_hessian": cartesian_hessian.tolist(),
        "step_convergence": convergence,
        "selected_step_scale": scale,
        "state_displacement_halvings": state_halvings,
        "state_audit": {
            point.key: {
                k: v
                for k, v in evaluation.result.execution.items()
                if k.startswith("state_") or k == "selected_electronic_state"
            }
            for point, evaluation in evaluated
        },
    }
    result_path = target / "modal-hessian.json"
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (target / "modal-hessian-plan.json").write_text(
        json.dumps(plan.to_json(), indent=2) + "\n", encoding="utf-8"
    )
    return payload | {"result_path": result_path, "frequencies_cm": normal.frequencies_cm}


__all__ = [
    "ModalHessianConvergenceDiagnostics",
    "ModalHessianPlan",
    "ModalHessianPoint",
    "ModalModeStencil",
    "MODAL_PARITY_PROTOCOL",
    "classify_modal_stencils",
    "totally_symmetric_mode_indices",
    "plan_modal_hessian",
    "modal_hessian_from_energies",
    "modal_hessian_convergence_diagnostics",
    "acquire_modal_hessian",
]
