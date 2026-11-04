"""LINK production optimization of mass-normalized Bowman ``Q_im`` paths."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize


QIM_LINK_SCHEMA = "matrix.link.qim_constrained_optimization.v2"
ANGSTROM_TO_BOHR = 1.8897261246257702


@dataclass(frozen=True)
class TransitionVectorContract:
    """Legacy Cartesian transition vector accepted at the LINK boundary."""

    vector: np.ndarray
    imaginary_frequency: float
    source: str = "transition_state_hessian"

    def __post_init__(self) -> None:
        values = np.asarray(self.vector, dtype=float).reshape(-1)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("transition vector must be finite and non-empty")
        if not np.isfinite(self.imaginary_frequency):
            raise ValueError("imaginary frequency must be finite")
        if not np.isclose(np.linalg.norm(values), 1.0, atol=1.0e-6):
            raise ValueError("legacy transition vector must have unit Euclidean norm")


@dataclass(frozen=True)
class ConstrainedQimPoint:
    q_im: float
    coordinates_angstrom: np.ndarray
    energy_hartree: float
    orthogonal_gradient_norm: float
    iterations: int
    converged: bool
    orthogonal_coordinates: np.ndarray | None = None
    dipole_au: np.ndarray | None = None
    energy_standard_error_hartree: float | None = None
    dipole_covariance_au2: np.ndarray | None = None
    function_evaluations: int = 0
    optimizer_message: str = ""


@dataclass(frozen=True)
class ConstrainedQimResult:
    schema: str
    contract: object
    points: tuple[ConstrainedQimPoint, ...]
    orthogonal_basis: np.ndarray
    optimizer: str = "continuation-l-bfgs-b"
    all_converged: bool = False


def orthogonal_basis(transition_vector: np.ndarray, count: int | None = None) -> np.ndarray:
    """Return a deterministic Euclidean transverse basis (legacy convention)."""

    return _orthogonal_basis(np.asarray(transition_vector, dtype=float), None, count)


def mass_orthogonal_basis(
    transition_vector: np.ndarray,
    cartesian_masses: Sequence[float],
    count: int | None = None,
    *,
    reference_coordinates_angstrom: np.ndarray | None = None,
) -> np.ndarray:
    """Return mass-orthonormal transverse vibrational Cartesian rows."""

    vector = np.asarray(transition_vector, dtype=float).reshape(-1)
    masses = np.asarray(cartesian_masses, dtype=float).reshape(-1)
    if reference_coordinates_angstrom is None:
        return _orthogonal_basis(vector, masses, count)
    geometry = np.asarray(reference_coordinates_angstrom, dtype=float).reshape((-1, 3))
    if masses.shape != (geometry.size,) or not np.allclose(
        masses.reshape((-1, 3)),
        masses.reshape((-1, 3))[:, :1],
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError("rigid-mode projection requires one common mass per atom")
    atom_masses = masses.reshape((-1, 3))[:, 0]
    centered = geometry - np.sum(atom_masses[:, None] * geometry, axis=0) / np.sum(atom_masses)
    constraints = [np.sqrt(masses) * vector]
    for axis in range(3):
        translation = np.zeros_like(geometry)
        translation[:, axis] = 1.0
        constraints.append((np.sqrt(atom_masses)[:, None] * translation).reshape(-1))
    unit_axes = np.eye(3)
    for axis in unit_axes:
        rotation = np.cross(np.broadcast_to(axis, centered.shape), centered)
        constraints.append((np.sqrt(atom_masses)[:, None] * rotation).reshape(-1))
    constraint_matrix = np.asarray(constraints, dtype=float)
    _left, singular, right = np.linalg.svd(constraint_matrix, full_matrices=True)
    threshold = 1e-10 * max(float(singular[0]), 1.0)
    rank = int(np.count_nonzero(singular > threshold))
    weighted_basis = right[rank:]
    if count is not None:
        weighted_basis = weighted_basis[:count]
    # Fix SVD row signs so checkpoint identities do not depend on BLAS choices.
    for row in weighted_basis:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0.0:
            row *= -1.0
    return weighted_basis / np.sqrt(masses)[None, :]


def _orthogonal_basis(
    transition_vector: np.ndarray,
    masses: np.ndarray | None,
    count: int | None,
) -> np.ndarray:
    vector = np.asarray(transition_vector, dtype=float).reshape(-1)
    scale = np.ones_like(vector) if masses is None else np.sqrt(masses.reshape(-1))
    if scale.shape != vector.shape or np.any(scale <= 0.0):
        raise ValueError("Cartesian masses must match the transition vector")
    weighted = scale * vector
    weighted /= np.linalg.norm(weighted)
    basis: list[np.ndarray] = []
    for index in range(vector.size):
        trial = np.zeros_like(vector)
        trial[index] = 1.0
        for previous in [weighted, *basis]:
            trial -= np.dot(trial, previous) * previous
        norm = np.linalg.norm(trial)
        if norm > 1.0e-10:
            basis.append(trial / norm)
        if count is not None and len(basis) == count:
            break
    return np.asarray(basis) / scale[None, :]


def build_qim_contract_from_transition_state(
    hessian_hartree_per_bohr2: np.ndarray,
    masses_amu: Sequence[float],
    q_values: Sequence[float],
    *,
    reference_coordinates_angstrom: np.ndarray | None = None,
    oracle_point_group: str = "unknown",
    oracle_symmetry_actions: Mapping[str, np.ndarray] | None = None,
    imaginary_frequency_cm1: float | None = None,
):
    """Create the SMITH coordinate directly from a certified LINK TS result."""

    from matrix_smith import build_tunneling_coordinate_contract

    return build_tunneling_coordinate_contract(
        np.asarray(hessian_hartree_per_bohr2, dtype=float),
        masses_amu,
        q_values,
        periodic=False,
        reference_coordinates_angstrom=reference_coordinates_angstrom,
        imaginary_frequency_cm1=imaginary_frequency_cm1,
        oracle_point_group=oracle_point_group,
        oracle_symmetry_actions=oracle_symmetry_actions,
    )


def qim_contract_from_link_transition_state_artifact(
    artifact: Path | str,
    masses_amu: Sequence[float],
    q_values: Sequence[float],
    *,
    reference_coordinates_angstrom: np.ndarray | None = None,
    oracle_point_group: str = "unknown",
    oracle_symmetry_actions: Mapping[str, np.ndarray] | None = None,
):
    """Read LINK's certified final TS Hessian and construct the SMITH contract."""

    source = Path(artifact)
    if source.is_dir():
        candidates = (
            source / "final_cartesian_hessian.json",
            source / "optimizer_hessian.json",
        )
        source = next((path for path in candidates if path.exists()), candidates[0])
    if not source.exists():
        raise FileNotFoundError(f"LINK transition-state Hessian not found: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        raw = None
        for key in (
            "cartesian_hessian_hartree_per_bohr2",
            "hessian_hartree_per_bohr2",
            "cartesian_hessian",
            "hessian",
        ):
            if key in payload:
                raw = payload[key]
                break
        if raw is None:
            raise ValueError("LINK transition-state artifact contains no Cartesian Hessian")
    else:
        raw = payload
    hessian = np.asarray(raw, dtype=float)
    return build_qim_contract_from_transition_state(
        hessian,
        masses_amu,
        q_values,
        reference_coordinates_angstrom=reference_coordinates_angstrom,
        oracle_point_group=oracle_point_group,
        oracle_symmetry_actions=oracle_symmetry_actions,
    )


def _contract_arrays(contract: object) -> tuple[np.ndarray, np.ndarray | None]:
    if isinstance(contract, TransitionVectorContract):
        return np.asarray(contract.vector, dtype=float).reshape(-1), None
    direction = np.asarray(getattr(contract, "cartesian_direction"), dtype=float).reshape(-1)
    masses = np.asarray(getattr(contract, "cartesian_masses"), dtype=float).reshape(-1)
    if getattr(contract, "coordinate_unit", "angstrom_sqrt_amu") != "angstrom_sqrt_amu":
        raise ValueError("LINK requires Qim in angstrom_sqrt_amu")
    if direction.shape != masses.shape:
        raise ValueError("SMITH Qim direction and masses have inconsistent dimensions")
    return direction, masses


def _property(evaluation: object, names: Sequence[str]) -> object | None:
    for name in names:
        value = getattr(evaluation, name, None)
        if value is not None:
            return value
    return None


def optimize_constrained_qim(
    session: object,
    reference_coordinates_angstrom: np.ndarray,
    contract: object,
    q_values: Sequence[float] | None = None,
    *,
    perpendicular_modes: np.ndarray | None = None,
    max_iterations: int = 200,
    gradient_tolerance: float = 1.0e-5,
    trust_radius: float = 0.35,
    step_size: float | None = None,
    checkpoint_path: Path | str | None = None,
    require_convergence: bool = True,
    sample_dipole: bool = False,
    dipole_evaluator: Callable[[np.ndarray, object], Sequence[float]] | None = None,
    symmetry_actions: Mapping[str, np.ndarray] | None = None,
    transverse_coordinate_scales: Sequence[float] | None = None,
) -> ConstrainedQimResult:
    """Relax transverse modes at fixed Qim with checkpointed branch continuation.

    ``step_size`` retains the historical displacement-budget spelling and maps to
    ``trust_radius`` in the canonical optimizer.
    """

    if step_size is not None:
        # The canonical optimizer uses bounded L-BFGS-B with correct units.
        # Retain the old caller's explicit displacement budget as its trust radius.
        if not np.isfinite(step_size) or step_size <= 0.0:
            raise ValueError("step_size must be finite and positive")
        if trust_radius != 0.35 and trust_radius != step_size:
            raise ValueError("step_size and trust_radius specify conflicting budgets")
        trust_radius = float(step_size)

    reference = np.asarray(reference_coordinates_angstrom, dtype=float).reshape(-1)
    vector, masses = _contract_arrays(contract)
    if reference.size != vector.size:
        raise ValueError("reference and SMITH Qim dimensions differ")
    values = np.asarray(
        tuple(getattr(contract, "points")) if q_values is None else q_values,
        dtype=float,
    )
    if values.ndim != 1 or values.size < 1 or not np.all(np.isfinite(values)):
        raise ValueError("Qim grid must be a non-empty finite sequence")
    if len(np.unique(values)) != len(values):
        raise ValueError("Qim grid values must be unique")
    if max_iterations < 1 or gradient_tolerance <= 0.0 or trust_radius <= 0.0:
        raise ValueError("Qim optimizer limits and tolerances must be positive")

    if perpendicular_modes is None:
        basis = (
            orthogonal_basis(vector)
            if masses is None
            else mass_orthogonal_basis(
                vector,
                masses,
                reference_coordinates_angstrom=reference.reshape((-1, 3)),
            )
        )
    else:
        basis = np.asarray(perpendicular_modes, dtype=float)
        if basis.ndim != 2 or basis.shape[1] != vector.size:
            raise ValueError("perpendicular_modes must contain Cartesian row vectors")
        metric = np.ones_like(vector) if masses is None else masses
        if not np.allclose((basis * metric[None, :]) @ basis.T, np.eye(len(basis)), atol=1e-7):
            raise ValueError("perpendicular modes must be mass-metric orthonormal")
        if not np.allclose((basis * metric[None, :]) @ vector, 0.0, atol=1e-7):
            raise ValueError("perpendicular modes are not transverse to Qim")
    coordinate_scales = (
        np.ones(basis.shape[0])
        if transverse_coordinate_scales is None
        else np.asarray(transverse_coordinate_scales, dtype=float)
    )
    if coordinate_scales.shape != (basis.shape[0],) or np.any(coordinate_scales <= 0.0):
        raise ValueError("transverse coordinate scales must be positive with one value per mode")

    restored = dict(_read_checkpoint(checkpoint_path, values, basis))
    completed = {q_im: point for q_im, point in restored.items() if point.converged}
    coefficients_by_q = {
        q: np.asarray(point.orthogonal_coordinates, dtype=float)
        for q, point in restored.items()
        if point.orthogonal_coordinates is not None
    }
    center = float(values[int(np.argmin(np.abs(values)))])
    reversing_action: tuple[str, np.ndarray] | None = None
    characters = dict(getattr(contract, "symmetry_characters", ()))
    if symmetry_actions and any(character == -1 for character in characters.values()):
        if abs(center) > 1.0e-12:
            raise ValueError("a symmetry-generated Qim path requires a zero grid point")
        reflected = {-float(value) for value in values if value < 0.0}
        positive = {float(value) for value in values if value > 0.0}
        if reflected != positive:
            raise ValueError("a symmetry-generated Qim grid must contain matching +/- points")
        reversing_labels = [
            label
            for label, character in sorted(characters.items())
            if character == -1 and label in symmetry_actions
        ]
        if not reversing_labels:
            raise ValueError("no supplied ORACLE Cartesian action reverses Qim")
        label = reversing_labels[0]
        action = np.asarray(symmetry_actions[label], dtype=float)
        if action.shape != (reference.size, reference.size):
            raise ValueError("the Qim-reversing Cartesian symmetry action has the wrong shape")
        if not np.allclose(action @ reference, reference, atol=1.0e-7, rtol=0.0):
            raise ValueError("the Qim-reversing action does not preserve the reference geometry")
        if not np.allclose(action @ vector, -vector, atol=1.0e-7, rtol=0.0):
            raise ValueError("the selected ORACLE action does not reverse Qim")
        reversing_action = (label, action)
    branches = (
        (center,),
        tuple(sorted(float(q) for q in values if q > center)),
        (() if reversing_action is not None else tuple(
            sorted((float(q) for q in values if q < center), reverse=True)
        )),
    )
    center_coefficients = coefficients_by_q.get(center, np.zeros(basis.shape[0]))
    for branch in branches:
        coefficients = center_coefficients.copy()
        for q_im in branch:
            if q_im in completed:
                coefficients = coefficients_by_q.get(q_im, coefficients).copy()
                if q_im == center:
                    center_coefficients = coefficients.copy()
                continue
            point = _optimize_point(
                session,
                reference,
                vector,
                basis,
                q_im,
                coefficients_by_q.get(q_im, coefficients),
                max_iterations=max_iterations,
                gradient_tolerance=gradient_tolerance,
                trust_radius=trust_radius,
                sample_dipole=sample_dipole,
                dipole_evaluator=dipole_evaluator,
                coordinate_scales=coordinate_scales,
            )
            completed[q_im] = point
            coefficients = np.asarray(point.orthogonal_coordinates, dtype=float)
            coefficients_by_q[q_im] = coefficients.copy()
            if q_im == center:
                center_coefficients = coefficients.copy()
            _write_checkpoint(checkpoint_path, contract, values, completed, basis)
            if require_convergence and not point.converged:
                raise RuntimeError(
                    f"Qim transverse optimization failed at q={q_im:g}: {point.optimizer_message}"
                )

    if reversing_action is not None:
        label, action = reversing_action
        metric = np.ones_like(vector) if masses is None else masses
        for q_im in sorted(float(value) for value in values if value > 0.0):
            source = completed[q_im]
            partner_q = -q_im
            partner_flat = action @ np.asarray(source.coordinates_angstrom, dtype=float).reshape(-1)
            displacement = partner_flat - reference - partner_q * vector
            coefficients = (basis * metric[None, :]) @ displacement
            if source.dipole_au is not None:
                raise ValueError(
                    "symmetry-generated Qim dipoles require an explicit molecular-vector action"
                )
            completed[partner_q] = ConstrainedQimPoint(
                q_im=partner_q,
                coordinates_angstrom=partner_flat.reshape((-1, 3)),
                energy_hartree=source.energy_hartree,
                orthogonal_gradient_norm=source.orthogonal_gradient_norm,
                iterations=0,
                converged=source.converged,
                orthogonal_coordinates=coefficients,
                energy_standard_error_hartree=source.energy_standard_error_hartree,
                function_evaluations=0,
                optimizer_message=f"generated by ORACLE symmetry operation {label}",
            )
        _write_checkpoint(checkpoint_path, contract, values, completed, basis)

    points = tuple(completed[float(q)] for q in sorted(values))
    return ConstrainedQimResult(
        QIM_LINK_SCHEMA,
        contract,
        points,
        basis,
        all_converged=all(point.converged for point in points),
    )


def _optimize_point(
    session: object,
    reference: np.ndarray,
    vector: np.ndarray,
    basis: np.ndarray,
    q_im: float,
    initial: np.ndarray,
    *,
    max_iterations: int,
    gradient_tolerance: float,
    trust_radius: float,
    sample_dipole: bool,
    dipole_evaluator: Callable[[np.ndarray, object], Sequence[float]] | None,
    coordinate_scales: np.ndarray,
    mobile_trust_region: bool = True,
) -> ConstrainedQimPoint:
    cache: dict[bytes, tuple[float, np.ndarray, object, np.ndarray]] = {}

    def evaluate(coefficients: np.ndarray) -> tuple[float, np.ndarray, object, np.ndarray]:
        key = np.asarray(coefficients, dtype=np.float64).tobytes()
        if key not in cache:
            flat = reference + q_im * vector + coefficients @ basis
            geometry = flat.reshape((-1, 3))
            result = session.evaluate(geometry)
            gradient = np.asarray(result.gradient_hartree_per_bohr, dtype=float).reshape(-1)
            if gradient.shape != flat.shape or not np.all(np.isfinite(gradient)):
                raise ValueError("LINK Qim evaluation returned an invalid Cartesian gradient")
            cache[key] = (
                float(result.energy_hartree),
                ANGSTROM_TO_BOHR * (basis @ gradient),
                result,
                geometry,
            )
        return cache[key]

    def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
        energy, projected, _result, _geometry = evaluate(coefficients)
        return energy, projected

    initial = np.asarray(initial, dtype=float)
    scales = np.asarray(coordinate_scales, dtype=float)
    scaled_coordinates = initial / scales
    fixed_bound_center = scaled_coordinates.copy()
    iterations = 0
    messages: list[str] = []
    optimized = None
    while iterations < max_iterations:
        stage_iterations = min(20, max_iterations - iterations)
        center = scaled_coordinates.copy()
        bound_center = center if mobile_trust_region else fixed_bound_center

        def scaled_objective(values: np.ndarray) -> tuple[float, np.ndarray]:
            energy, gradient = objective(values * scales)
            return energy, gradient * scales

        optimized = minimize(
            scaled_objective,
            center,
            method="L-BFGS-B",
            jac=True,
            bounds=tuple(
                zip(
                    bound_center - trust_radius / scales,
                    bound_center + trust_radius / scales,
                    strict=True,
                )
            ),
            options={
                "maxiter": int(stage_iterations),
                # L-BFGS-B applies gtol to the infinity norm, whereas LINK's
                # convergence contract uses the Euclidean transverse norm.
                "gtol": (
                    float(gradient_tolerance)
                    * float(np.min(scales))
                    / np.sqrt(max(initial.size, 1))
                ),
                # Total electronic energies are hundreds or thousands of hartree.
                # SciPy's default relative-energy stop can therefore pre-empt the
                # Cartesian-gradient criterion after a physically tiny first step.
                "ftol": 1.0e-14,
                "maxls": 30,
            },
        )
        step_iterations = int(getattr(optimized, "nit", 0))
        iterations += max(step_iterations, 1)
        scaled_coordinates = np.asarray(optimized.x, dtype=float)
        coefficients = scaled_coordinates * scales
        _energy, stage_gradient, _evaluation, _geometry = evaluate(coefficients)
        messages.append(str(optimized.message))
        if np.linalg.norm(stage_gradient) <= gradient_tolerance:
            break
        if np.linalg.norm(scaled_coordinates - center) <= 1.0e-12:
            break
    if optimized is None:  # pragma: no cover - limits are validated by the caller.
        raise RuntimeError("Qim optimizer did not execute")
    coefficients = scaled_coordinates * scales
    energy, projected, evaluation, geometry = evaluate(coefficients)
    gradient_norm = float(np.linalg.norm(projected))
    converged = bool(gradient_norm <= gradient_tolerance)
    dipole = None
    if sample_dipole:
        raw = (
            dipole_evaluator(geometry.copy(), evaluation)
            if dipole_evaluator is not None
            else _property(evaluation, ("dipole_au", "molecular_dipole_au", "electric_dipole_au"))
        )
        if raw is None:
            raise ValueError("the selected QM backend did not return a molecular dipole")
        dipole = np.asarray(raw, dtype=float).reshape(-1)
        if dipole.shape != (3,) or not np.all(np.isfinite(dipole)):
            raise ValueError("QM dipole samples must contain three finite components")
    energy_sigma = _property(
        evaluation, ("energy_standard_error_hartree", "energy_uncertainty_hartree")
    )
    covariance = _property(evaluation, ("dipole_covariance_au2",))
    dipole_covariance = None if covariance is None else np.asarray(covariance, dtype=float)
    if dipole_covariance is not None and dipole_covariance.shape != (3, 3):
        raise ValueError("QM dipole covariance must have shape 3 x 3")
    return ConstrainedQimPoint(
        q_im=float(q_im),
        coordinates_angstrom=geometry.copy(),
        energy_hartree=energy,
        orthogonal_gradient_norm=gradient_norm,
        iterations=iterations,
        converged=converged,
        orthogonal_coordinates=coefficients.copy(),
        dipole_au=None if dipole is None else dipole.copy(),
        energy_standard_error_hartree=None if energy_sigma is None else float(energy_sigma),
        dipole_covariance_au2=None if dipole_covariance is None else dipole_covariance.copy(),
        function_evaluations=len(cache),
        optimizer_message="; ".join(messages),
    )


def _read_checkpoint(
    path: Path | str | None,
    q_values: np.ndarray,
    basis: np.ndarray,
) -> tuple[tuple[float, ConstrainedQimPoint], ...]:
    if path is None or not Path(path).exists():
        return ()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != QIM_LINK_SCHEMA:
        raise ValueError("Qim checkpoint uses an unsupported schema")
    stored_basis = np.asarray(payload.get("orthogonal_basis"), dtype=float)
    if stored_basis.shape != basis.shape or not np.allclose(
        stored_basis, basis, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("Qim checkpoint transverse basis differs from the current basis")
    stored_values = np.asarray(payload.get("q_values"), dtype=float)
    if stored_values.ndim != 1 or any(
        not np.any(np.isclose(value, q_values, rtol=0.0, atol=1.0e-14))
        for value in stored_values
    ):
        raise ValueError("Qim checkpoint contains points outside the requested grid")
    rows = []
    for item in payload.get("points", ()):
        coefficients = np.asarray(item["orthogonal_coordinates"], dtype=float)
        if coefficients.shape != (basis.shape[0],):
            raise ValueError("Qim checkpoint transverse dimension changed")
        point = _point_from_dict(item)
        rows.append((point.q_im, point))
    return tuple(rows)


def _write_checkpoint(
    path: Path | str | None,
    contract: object,
    q_values: np.ndarray,
    completed: Mapping[float, ConstrainedQimPoint],
    basis: np.ndarray,
) -> None:
    if path is None:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": QIM_LINK_SCHEMA,
        "q_values": [float(value) for value in q_values],
        "coordinate_schema": getattr(contract, "schema", "legacy"),
        "orthogonal_basis": basis.tolist(),
        "points": [_point_to_dict(completed[key]) for key in sorted(completed)],
    }
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


def _point_to_dict(point: ConstrainedQimPoint) -> dict[str, object]:
    return {
        "q_im": point.q_im,
        "coordinates_angstrom": np.asarray(point.coordinates_angstrom).tolist(),
        "energy_hartree": point.energy_hartree,
        "orthogonal_gradient_norm": point.orthogonal_gradient_norm,
        "iterations": point.iterations,
        "converged": point.converged,
        "orthogonal_coordinates": None
        if point.orthogonal_coordinates is None
        else np.asarray(point.orthogonal_coordinates).tolist(),
        "dipole_au": None if point.dipole_au is None else np.asarray(point.dipole_au).tolist(),
        "energy_standard_error_hartree": point.energy_standard_error_hartree,
        "dipole_covariance_au2": None
        if point.dipole_covariance_au2 is None
        else np.asarray(point.dipole_covariance_au2).tolist(),
        "function_evaluations": point.function_evaluations,
        "optimizer_message": point.optimizer_message,
    }


def _point_from_dict(payload: Mapping[str, object]) -> ConstrainedQimPoint:
    return ConstrainedQimPoint(
        q_im=float(payload["q_im"]),
        coordinates_angstrom=np.asarray(payload["coordinates_angstrom"], dtype=float),
        energy_hartree=float(payload["energy_hartree"]),
        orthogonal_gradient_norm=float(payload["orthogonal_gradient_norm"]),
        iterations=int(payload["iterations"]),
        converged=bool(payload["converged"]),
        orthogonal_coordinates=np.asarray(payload["orthogonal_coordinates"], dtype=float),
        dipole_au=None
        if payload.get("dipole_au") is None
        else np.asarray(payload["dipole_au"], dtype=float),
        energy_standard_error_hartree=None
        if payload.get("energy_standard_error_hartree") is None
        else float(payload["energy_standard_error_hartree"]),
        dipole_covariance_au2=None
        if payload.get("dipole_covariance_au2") is None
        else np.asarray(payload["dipole_covariance_au2"], dtype=float),
        function_evaluations=int(payload.get("function_evaluations", 0)),
        optimizer_message=str(payload.get("optimizer_message", "restored from checkpoint")),
    )


def optimize_qim_isotopologues(
    session_factory: Callable[[str], object],
    reference_coordinates_angstrom: np.ndarray,
    hessian_hartree_per_bohr2: np.ndarray,
    isotopologue_masses_amu: Mapping[str, Sequence[float]],
    q_values: Sequence[float],
    **optimizer_options: object,
) -> dict[str, ConstrainedQimResult]:
    """Rebuild and relax Qim independently for every isotopologue."""

    if not isotopologue_masses_amu:
        raise ValueError("at least one isotopologue mass vector is required")
    results: dict[str, ConstrainedQimResult] = {}
    for label, masses in isotopologue_masses_amu.items():
        contract = build_qim_contract_from_transition_state(
            hessian_hartree_per_bohr2,
            masses,
            q_values,
            reference_coordinates_angstrom=reference_coordinates_angstrom,
        )
        options = dict(optimizer_options)
        checkpoint_root = options.pop("checkpoint_root", None)
        if checkpoint_root is not None:
            options["checkpoint_path"] = Path(checkpoint_root) / str(label) / "qim-checkpoint.json"
        results[str(label)] = optimize_constrained_qim(
            session_factory(str(label)),
            reference_coordinates_angstrom,
            contract,
            **options,
        )
    return results


def convert_path_direction(
    direction: np.ndarray,
    *,
    representation: str,
    internal_to_cartesian: np.ndarray | None = None,
) -> np.ndarray:
    """Convert a Cartesian or internal-coordinate path direction to Cartesian space."""

    key = str(representation).strip().casefold()
    values = np.asarray(direction, dtype=float).reshape(-1)
    if key == "cartesian":
        return values
    if key != "internal":
        raise ValueError("Qim representation must be cartesian or internal")
    if internal_to_cartesian is None:
        raise ValueError("internal Qim paths require an internal_to_cartesian Jacobian")
    return np.linalg.lstsq(np.asarray(internal_to_cartesian, dtype=float), values, rcond=1e-10)[0]


def mass_weighted_path_distances(
    coordinates: Sequence[np.ndarray], masses: Sequence[float]
) -> np.ndarray:
    """Recompute cumulative path distance in Cartesian mass-weighted space."""

    values = [np.asarray(item, dtype=float).reshape(-1) for item in coordinates]
    mass = np.repeat(np.sqrt(np.asarray(masses, dtype=float).reshape(-1)), 3)
    distances = [0.0]
    for previous, current in zip(values, values[1:], strict=False):
        if previous.size != mass.size or current.size != mass.size:
            raise ValueError("coordinates and masses have incompatible dimensions")
        distances.append(distances[-1] + float(np.linalg.norm((current - previous) * mass)))
    return np.asarray(distances)


def write_constrained_qim_result(path: Path | str, result: ConstrainedQimResult) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    to_dict = getattr(result.contract, "to_dict", None)
    contract_payload = (
        to_dict()
        if callable(to_dict)
        else {
            "imaginary_frequency": getattr(result.contract, "imaginary_frequency", None),
            "source": getattr(result.contract, "source", "legacy"),
        }
    )
    target.write_text(
        json.dumps(
            {
                "schema": result.schema,
                "optimizer": result.optimizer,
                "all_converged": result.all_converged,
                "coordinate_contract": contract_payload,
                "orthogonal_basis": result.orthogonal_basis.tolist(),
                "points": [_point_to_dict(point) for point in result.points],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return target


__all__ = [
    "ConstrainedQimPoint",
    "ConstrainedQimResult",
    "QIM_LINK_SCHEMA",
    "TransitionVectorContract",
    "build_qim_contract_from_transition_state",
    "convert_path_direction",
    "mass_orthogonal_basis",
    "mass_weighted_path_distances",
    "qim_contract_from_link_transition_state_artifact",
    "optimize_constrained_qim",
    "optimize_qim_isotopologues",
    "orthogonal_basis",
    "write_constrained_qim_result",
]
