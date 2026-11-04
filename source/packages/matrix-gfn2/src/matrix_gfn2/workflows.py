"""Autonomous geometry-optimization and harmonic-analysis workflows."""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
import multiprocessing as mp
from pathlib import Path

import numpy as np

from .calculator import ANGSTROM_TO_BOHR, GFN2Calculator


BOHR_TO_ANGSTROM = 1.0 / ANGSTROM_TO_BOHR
FREQUENCY_FACTOR_CM1 = 5140.487143715828


def _sycart_plan_identity(
    atoms: tuple[str, ...],
    irreps: tuple[str, ...],
    coordinates: np.ndarray,
    basis: np.ndarray,
    rigid_basis: np.ndarray,
    parity: np.ndarray,
) -> str:
    atom_bytes = "\0".join(atoms).encode("utf-8")
    irrep_bytes = "\0".join(irreps).encode("utf-8")
    payload = b"matrix-gfn2/frozen-sycart-hessian-plan/v2\0" + atom_bytes + b"\0"
    payload += irrep_bytes + b"\0"
    payload += np.concatenate(
        (coordinates.reshape(-1), basis.reshape(-1), rigid_basis.reshape(-1), parity.astype(float))
    ).tobytes()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class OptimizationStep:
    iteration: int
    energy_hartree: float
    gradient_max_hartree_per_bohr: float
    coordinates_angstrom: np.ndarray


@dataclass(frozen=True)
class OptimizationResult:
    coordinates_angstrom: np.ndarray
    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray
    iterations: int
    converged: bool
    trajectory: tuple[OptimizationStep, ...]


@dataclass(frozen=True)
class HarmonicResult:
    hessian_hartree_per_bohr2: np.ndarray
    frequencies_cm1: np.ndarray
    mass_weighted_modes: np.ndarray
    rigid_mode_count: int
    mode_irreps: tuple[str, ...] = ()
    mode_irrep_weights: tuple[float, ...] = ()


@dataclass(frozen=True)
class HessianVectorResult:
    product_hartree_per_bohr2: np.ndarray
    direction_bohr: np.ndarray
    step: float
    stencil_order: int
    gradient_evaluations: int
    method: str = "directional-derivative-of-analytic-gradient"


@dataclass(frozen=True)
class FrozenSYCARTHessianPlan:
    """Immutable translation/rotation-free Cartesian Hessian plan.

    ``basis`` has orthonormal Cartesian columns and is frozen read-only.  The
    optional ``sign_symmetric`` mask identifies SYCART coordinates for which
    the frozen local surface obeys ``g(+q)=-g(-q)``; those columns require one
    displaced gradient instead of a plus/minus pair.  The mask is explicit so
    the workflow never guesses parity from a changing topology.
    """

    atoms: tuple[str, ...]
    irreps: tuple[str, ...]
    coordinates_angstrom: np.ndarray
    basis: np.ndarray
    rigid_basis: np.ndarray
    sign_symmetric: np.ndarray
    step_bohr: float
    identity: str

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.coordinates_angstrom, dtype=float)
        basis = np.asarray(self.basis, dtype=float)
        parity = np.asarray(self.sign_symmetric, dtype=bool)
        if coordinates.shape != (len(self.atoms), 3):
            raise ValueError("SYCART plan coordinates have the wrong shape")
        rigid = np.asarray(self.rigid_basis, dtype=float)
        if basis.shape[0] != coordinates.size or basis.shape[1] == 0:
            raise ValueError("SYCART basis has the wrong shape")
        if rigid.shape[0] != coordinates.size:
            raise ValueError("SYCART rigid completion has the wrong shape")
        if parity.shape != (basis.shape[1],):
            raise ValueError("SYCART parity mask has the wrong shape")
        if len(self.irreps) != basis.shape[1] or any(not label.strip() for label in self.irreps):
            raise ValueError("SYCART requires one non-empty frozen irrep label per coordinate")
        if not np.isfinite(self.step_bohr) or self.step_bohr <= 0.0:
            raise ValueError("SYCART Hessian step must be positive and finite")
        if not np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=2.0e-12):
            raise ValueError("SYCART basis is not orthonormal")
        if not np.allclose(rigid.T @ rigid, np.eye(rigid.shape[1]), atol=2.0e-12):
            raise ValueError("SYCART rigid completion is not orthonormal")
        if not np.allclose(basis.T @ rigid, 0.0, atol=2.0e-12):
            raise ValueError("SYCART and rigid spaces are not orthogonal")
        for value in (coordinates, basis, rigid, parity):
            value.setflags(write=False)
        object.__setattr__(self, "coordinates_angstrom", coordinates)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "rigid_basis", rigid)
        object.__setattr__(self, "sign_symmetric", parity)


@dataclass(frozen=True)
class SYCARTHessianResult:
    """Hessian and audit information returned by the immutable SYCART path."""

    hessian_hartree_per_coordinate2: np.ndarray
    plan: FrozenSYCARTHessianPlan
    gradient_evaluations: int
    one_point_columns: tuple[int, ...]
    two_point_columns: tuple[int, ...]

    def __post_init__(self) -> None:
        hessian = np.asarray(self.hessian_hartree_per_coordinate2, dtype=float)
        hessian.setflags(write=False)
        object.__setattr__(self, "hessian_hartree_per_coordinate2", hessian)


def build_sycart_hessian_plan(
    atoms: tuple[str, ...] | list[str],
    coordinates_angstrom: np.ndarray,
    *,
    basis_cartesian_from_sycart: np.ndarray,
    irreps: tuple[str, ...],
    sign_symmetric: tuple[int, ...] = (),
    step_bohr: float = 1.0e-3,
) -> FrozenSYCARTHessianPlan:
    """Freeze a symmetry-adapted, TR-free SYCART basis and parity contract."""

    symbols = tuple(str(atom) for atom in atoms)
    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    if coordinates.shape != (len(symbols), 3):
        raise ValueError("SYCART coordinates must have shape natoms x 3")
    if len(symbols) < 2:
        raise ValueError("SYCART requires at least two atoms")
    flat = coordinates.reshape(-1)
    centered = coordinates - np.mean(coordinates, axis=0, keepdims=True)
    rigid = np.zeros((flat.size, 6), dtype=float)
    rigid[:, :3] = np.tile(np.eye(3), (len(symbols), 1))
    for atom, vector in enumerate(centered):
        offset = 3 * atom
        rigid[offset:offset + 3, 3:] = np.array(
            ((0.0, -vector[2], vector[1]),
             (vector[2], 0.0, -vector[0]),
             (-vector[1], vector[0], 0.0)),
            dtype=float,
        )
    _, singular, vh = np.linalg.svd(rigid.T, full_matrices=True)
    rank = int(np.count_nonzero(singular > max(rigid.shape) * np.finfo(float).eps * singular[0]))
    rigid_basis = vh[:rank].T
    default_basis = vh[rank:].T
    basis = np.asarray(basis_cartesian_from_sycart, dtype=float)
    if basis.shape != default_basis.shape:
        raise ValueError("frozen symmetry-adapted SYCART basis has the wrong shape")
    parity = np.zeros(basis.shape[1], dtype=bool)
    for index in sign_symmetric:
        if not 0 <= int(index) < basis.shape[1]:
            raise ValueError("SYCART parity index is outside the active basis")
        parity[int(index)] = True
    labels = tuple(str(label) for label in irreps)
    if len(labels) != basis.shape[1] or any(not label.strip() for label in labels):
        raise ValueError("SYCART requires one non-empty frozen irrep label per coordinate")
    identity = _sycart_plan_identity(symbols, labels, coordinates, basis, rigid_basis, parity)
    return FrozenSYCARTHessianPlan(
        symbols, labels, coordinates.copy(), basis, rigid_basis, parity, float(step_bohr), identity
    )


def sycart_hessian_plan_to_dict(plan: FrozenSYCARTHessianPlan) -> dict:
    """Serialize the frozen SYCART contract without losing its audit hash."""

    return {
        "schema": "matrix-gfn2/frozen-sycart-hessian-plan/v2",
        "identity": plan.identity,
        "atoms": list(plan.atoms),
        "irreps": list(plan.irreps),
        "coordinates_angstrom": plan.coordinates_angstrom.tolist(),
        "basis": plan.basis.tolist(),
        "rigid_basis": plan.rigid_basis.tolist(),
        "sign_symmetric": [int(value) for value in plan.sign_symmetric],
        "step_bohr": float(plan.step_bohr),
    }


def sycart_hessian_plan_from_dict(payload: dict) -> FrozenSYCARTHessianPlan:
    """Load and validate a frozen plan; reject any tampered identity."""

    if payload.get("schema") != "matrix-gfn2/frozen-sycart-hessian-plan/v2":
        raise ValueError("unsupported SYCART Hessian plan schema")
    plan = FrozenSYCARTHessianPlan(
        tuple(str(atom) for atom in payload["atoms"]),
        tuple(str(label) for label in payload.get("irreps", ())),
        np.asarray(payload["coordinates_angstrom"], dtype=float),
        np.asarray(payload["basis"], dtype=float),
        np.asarray(payload["rigid_basis"], dtype=float),
        np.asarray(payload["sign_symmetric"], dtype=bool),
        float(payload["step_bohr"]),
        str(payload["identity"]),
    )
    expected = _sycart_plan_identity(
        plan.atoms,
        plan.irreps,
        plan.coordinates_angstrom,
        plan.basis,
        plan.rigid_basis,
        plan.sign_symmetric,
    )
    if expected != plan.identity:
        raise ValueError("SYCART Hessian plan identity mismatch")
    return plan


def write_sycart_hessian_plan(path: Path | str, plan: FrozenSYCARTHessianPlan) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(sycart_hessian_plan_to_dict(plan), indent=2) + "\n", encoding="utf-8")
    return target


def read_sycart_hessian_plan(path: Path | str) -> FrozenSYCARTHessianPlan:
    source = Path(path).expanduser().resolve()
    return sycart_hessian_plan_from_dict(json.loads(source.read_text(encoding="utf-8")))


def sycart_to_cartesian(plan: FrozenSYCARTHessianPlan, displacement: np.ndarray) -> np.ndarray:
    """Expand an active SYCART displacement into Cartesian components."""

    vector = np.asarray(displacement, dtype=float)
    if vector.shape != (plan.basis.shape[1],):
        raise ValueError("SYCART displacement has the wrong dimension")
    return plan.basis @ vector


def cartesian_to_sycart(plan: FrozenSYCARTHessianPlan, displacement: np.ndarray) -> np.ndarray:
    """Project a Cartesian displacement onto the frozen TR-free SYCART basis."""

    vector = np.asarray(displacement, dtype=float).reshape(-1)
    if vector.shape != (plan.basis.shape[0],):
        raise ValueError("Cartesian displacement has the wrong dimension")
    return plan.basis.T @ vector


def sycart_to_sonic_matrix(
    plan: FrozenSYCARTHessianPlan, sonic_b_cartesian: np.ndarray
) -> np.ndarray:
    """Compose a Cartesian SONIC B matrix with the frozen SYCART basis."""

    matrix = np.asarray(sonic_b_cartesian, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != plan.basis.shape[0]:
        raise ValueError("SONIC B matrix must have 3N Cartesian columns")
    return matrix @ plan.basis


def sycart_to_sonic(
    plan: FrozenSYCARTHessianPlan,
    displacement: np.ndarray,
    sonic_b_cartesian: np.ndarray,
) -> np.ndarray:
    """Map a SYCART displacement through the Cartesian SONIC Jacobian."""

    return sycart_to_sonic_matrix(plan, sonic_b_cartesian) @ np.asarray(displacement, dtype=float)


def sonic_to_sycart(
    plan: FrozenSYCARTHessianPlan,
    displacement: np.ndarray,
    sonic_b_cartesian: np.ndarray,
) -> np.ndarray:
    """Recover the least-norm SYCART displacement from a SONIC displacement."""

    reduced = sycart_to_sonic_matrix(plan, sonic_b_cartesian)
    values = np.asarray(displacement, dtype=float)
    if values.shape != (reduced.shape[0],):
        raise ValueError("SONIC displacement has the wrong dimension")
    return np.linalg.pinv(reduced, rcond=1.0e-12) @ values


def sycart_hessian_to_sonic(
    plan: FrozenSYCARTHessianPlan,
    hessian_hartree_per_sycart_coordinate2: np.ndarray,
    sonic_b_cartesian: np.ndarray,
    *,
    rcond: float = 1.0e-12,
) -> np.ndarray:
    """Transform a stationary-point Hessian from SYCART to SONIC.

    With ``dq_sonic = J dq_sycart`` and
    ``J = B_sonic_cartesian @ B_sycart_cartesian``, force constants transform
    as ``J^{+T} H_sycart J^+``.  The pseudoinverse also covers redundant
    SONIC sets.  At a non-stationary point the coordinate-curvature (B-prime)
    term must additionally be supplied by the optimizer; harmonic analysis is
    deliberately restricted to stationary points.
    """

    hessian = np.asarray(hessian_hartree_per_sycart_coordinate2, dtype=float)
    dimension = plan.basis.shape[1]
    if hessian.shape != (dimension, dimension):
        raise ValueError("SYCART Hessian has the wrong dimension")
    if rcond <= 0.0 or not np.isfinite(rcond):
        raise ValueError("SONIC Hessian pseudoinverse threshold must be positive")
    jacobian = sycart_to_sonic_matrix(plan, sonic_b_cartesian)
    inverse = np.linalg.pinv(jacobian, rcond=rcond)
    transformed = inverse.T @ hessian @ inverse
    return 0.5 * (transformed + transformed.T)


def _translation_free(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=float).reshape(-1, 3).copy()
    values -= np.mean(values, axis=0)
    return values.reshape(-1)


def optimize_geometry(
    calculator: GFN2Calculator,
    coordinates_angstrom: np.ndarray,
    *,
    gradient_tolerance_hartree_per_bohr: float = 3.0e-4,
    maximum_iterations: int = 100,
    maximum_atom_step_bohr: float = 0.20,
) -> OptimizationResult:
    """Minimize with translation-free Cartesian inverse-BFGS and line search."""

    coordinates = np.asarray(coordinates_angstrom, dtype=float) * ANGSTROM_TO_BOHR
    if coordinates.shape != (len(calculator.atoms), 3):
        raise ValueError("optimizer coordinates have the wrong shape")
    dimension = coordinates.size
    inverse_hessian = np.eye(dimension) * 4.0
    trajectory: list[OptimizationStep] = []
    result = (
        calculator.evaluate(coordinates * BOHR_TO_ANGSTROM)
        if calculator.nn_enabled else calculator._single_point(coordinates)
    )
    symmetry_adapter = None
    symmetry_projector = np.eye(dimension)
    if calculator.symmetry_contract is not None:
        diagnostics = calculator._model(coordinates)[-1]
        symmetry_adapter = diagnostics["compiled_adapter"]
        if symmetry_adapter is not None:
            manifold = symmetry_adapter.geometry_manifold(coordinates * BOHR_TO_ANGSTROM)
            symmetry_projector = manifold.basis @ manifold.basis.T
    for iteration in range(maximum_iterations + 1):
        gradient = _translation_free(result.gradient_hartree_per_bohr).reshape(
            coordinates.shape
        )
        gradient_max = float(np.max(np.linalg.norm(gradient, axis=1)))
        trajectory.append(
            OptimizationStep(
                iteration,
                result.energy_hartree,
                gradient_max,
                coordinates.copy() * BOHR_TO_ANGSTROM,
            )
        )
        if gradient_max <= gradient_tolerance_hartree_per_bohr:
            return OptimizationResult(
                coordinates * BOHR_TO_ANGSTROM,
                result.energy_hartree,
                result.gradient_hartree_per_bohr,
                iteration,
                True,
                tuple(trajectory),
            )
        if iteration == maximum_iterations:
            break
        flat_gradient = gradient.reshape(-1)
        direction = symmetry_projector @ _translation_free(
            -(inverse_hessian @ flat_gradient)
        )
        if float(direction @ flat_gradient) >= 0.0:
            inverse_hessian = np.eye(dimension) * 4.0
            direction = symmetry_projector @ _translation_free(-flat_gradient)
        atom_norms = np.linalg.norm(direction.reshape(-1, 3), axis=1)
        largest = float(np.max(atom_norms))
        if largest > maximum_atom_step_bohr:
            direction *= maximum_atom_step_bohr / largest
        directional_derivative = float(flat_gradient @ direction)
        scale = 1.0
        accepted = None
        while scale >= 1.0e-4:
            trial_coordinates = coordinates + scale * direction.reshape(-1, 3)
            if symmetry_adapter is not None:
                trial_coordinates = (
                    symmetry_adapter.project_geometry(
                        trial_coordinates * BOHR_TO_ANGSTROM
                    )
                    * ANGSTROM_TO_BOHR
                )
            trial = (
                calculator.evaluate(trial_coordinates * BOHR_TO_ANGSTROM)
                if calculator.nn_enabled else calculator._single_point(trial_coordinates)
            )
            if trial.energy_hartree <= result.energy_hartree + 1.0e-4 * scale * directional_derivative:
                accepted = (trial_coordinates, trial)
                break
            scale *= 0.5
        if accepted is None:
            break
        trial_coordinates, trial = accepted
        step = (trial_coordinates - coordinates).reshape(-1)
        next_gradient = _translation_free(trial.gradient_hartree_per_bohr)
        difference = next_gradient - flat_gradient
        curvature = float(step @ difference)
        if curvature > 1.0e-10 * np.linalg.norm(step) * np.linalg.norm(difference):
            rho = 1.0 / curvature
            identity = np.eye(dimension)
            left = identity - rho * np.outer(step, difference)
            inverse_hessian = (
                left @ inverse_hessian @ left.T + rho * np.outer(step, step)
            )
            inverse_hessian = 0.5 * (inverse_hessian + inverse_hessian.T)
        else:
            inverse_hessian = np.eye(dimension) * 4.0
        coordinates, result = trial_coordinates, trial
    return OptimizationResult(
        coordinates * BOHR_TO_ANGSTROM,
        result.energy_hartree,
        result.gradient_hartree_per_bohr,
        len(trajectory) - 1,
        False,
        tuple(trajectory),
    )


def numerical_hessian(
    calculator: GFN2Calculator,
    coordinates_angstrom: np.ndarray,
    *,
    step_bohr: float = 1.0e-3,
    workers: int | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    node_index: int = 0,
    node_count: int = 1,
) -> np.ndarray:
    """Differentiate the analytic gradient with resumable node parallelism.

    For NN-xTB this compatibility API returns the analytic NN Hessian.

    ``node_index``/``node_count`` partition Cartesian columns deterministically;
    each node may use ``workers`` local processes.  A checkpoint stores the
    completed columns, allowing interrupted node jobs to resume without
    recomputing gradients.
    """

    if calculator.nn_enabled:
        if not np.isfinite(step_bohr) or step_bohr <= 0.0:
            raise ValueError("Hessian compatibility step must be positive and finite")
        if workers is not None and int(workers) < 1:
            raise ValueError("Hessian worker count must be positive")
        if checkpoint_path is not None or node_index != 0 or node_count != 1:
            raise ValueError(
                "NN-xTB analytic Hessian does not support distributed checkpoints"
            )
        return calculator.evaluate_nn_hessian_analytic(
            coordinates_angstrom,
        ).hessian_hartree_per_bohr2

    coordinates = np.asarray(coordinates_angstrom, dtype=float) * ANGSTROM_TO_BOHR
    dimension = coordinates.size
    hessian = np.zeros((dimension, dimension))
    requested_workers = (
        min(dimension, max(1, min(os.cpu_count() or 1, 8)))
        if workers is None
        else int(workers)
    )
    if requested_workers < 1:
        raise ValueError("Hessian worker count must be positive")
    if node_count < 1 or not 0 <= node_index < node_count:
        raise ValueError("node_index must be in [0, node_count)")
    if calculator.diagonalizer_backend in {"gpu", "cupy", "torch-cuda", "torch-mps"}:
        requested_workers = 1
    payload = {
        "atoms": calculator.atoms,
        "charge": calculator.charge,
        "multiplicity": calculator.multiplicity,
        "temperature_kelvin": calculator.temperature_kelvin,
        "scf_energy_tolerance_hartree": calculator.scf_energy_tolerance_hartree,
        "scf_convergence_mode": calculator.scf_convergence_mode,
        "scf_mixer_tolerance": calculator.scf_mixer_tolerance,
        "scf_wavefunction_tolerance_electron": (
            calculator.scf_wavefunction_tolerance_electron
        ),
        "maximum_scf_iterations": calculator.maximum_scf_iterations,
        "point_charges_electron": calculator.point_charges_electron,
        "point_charge_coordinates_angstrom": (
            calculator.point_charge_coordinates_bohr * BOHR_TO_ANGSTROM
        ),
        "diagonalizer_backend": calculator.diagonalizer_backend,
        "density_solver": calculator.solver_policy.mode,
        "dense_sp2_min_orbitals": calculator.solver_policy.dense_sp2_min_orbitals,
        "sparse_sp2_min_orbitals": calculator.solver_policy.sparse_sp2_min_orbitals,
        "sparse_sp2_maximum_fill_fraction": (
            calculator.solver_policy.sparse_sp2_maximum_fill_fraction
        ),
        "orthogonalizer_method": calculator.orthogonalizer_method,
        "symmetry_contract": calculator.symmetry_contract,
    }
    checkpoint = None if checkpoint_path is None else Path(checkpoint_path)
    completed = np.zeros(dimension, dtype=bool)
    if checkpoint is not None and checkpoint.exists():
        saved = np.load(checkpoint, allow_pickle=False)
        if tuple(saved["coordinates_shape"].tolist()) != coordinates.shape:
            raise ValueError("Hessian checkpoint geometry shape does not match")
        if not np.allclose(saved["coordinates"], coordinates):
            raise ValueError("Hessian checkpoint coordinates do not match")
        hessian = np.asarray(saved["hessian"], dtype=float)
        completed = np.asarray(saved["completed"], dtype=bool)
        if hessian.shape != (dimension, dimension) or completed.shape != (dimension,):
            raise ValueError("invalid Hessian checkpoint dimensions")
    tasks = [
        (payload, coordinates, coordinate, step_bohr)
        for coordinate in range(node_index, dimension, node_count)
        if not completed[coordinate]
    ]
    if requested_workers == 1:
        columns = map(_hessian_column, tasks)
    else:
        context = mp.get_context("fork" if os.name == "posix" else "spawn")
        executor = ProcessPoolExecutor(
            max_workers=requested_workers, mp_context=context
        )
        columns = executor.map(_hessian_column, tasks)
    try:
        for coordinate, column in columns:
            hessian[:, coordinate] = column
            completed[coordinate] = True
            if checkpoint is not None:
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    checkpoint,
                    coordinates=coordinates,
                    coordinates_shape=np.asarray(coordinates.shape, dtype=int),
                    hessian=hessian,
                    completed=completed,
                )
    finally:
        if requested_workers != 1:
            executor.shutdown()
    if node_count > 1 and not np.all(completed):
        return hessian
    hessian = 0.5 * (hessian + hessian.T)
    if calculator.symmetry_contract is not None:
        model = calculator._model(coordinates)
        diagnostics = model[-1]
        adapter = diagnostics["compiled_adapter"]
        if adapter is not None:
            hessian = adapter.symmetrize_hessian(hessian)
    return hessian


def sycart_numerical_hessian(
    calculator: GFN2Calculator,
    plan: FrozenSYCARTHessianPlan,
    *,
    workers: int = 1,
) -> SYCARTHessianResult:
    """Evaluate a frozen SYCART Hessian with independent gradient workers.

    NN-xTB projects the complete analytic Cartesian Hessian directly.

    The plan is immutable: no worker may change the geometry, TR-free basis,
    or parity mask.  Generic coordinates use a central plus/minus stencil;
    explicitly sign-symmetric coordinates use one positive displacement and
    the reference gradient.  The returned Hessian is in the active SYCART
    basis, so translations and rotations are absent by construction.
    """

    if tuple(calculator.atoms) != plan.atoms:
        raise ValueError("SYCART plan atom frame differs from calculator")
    workers = int(workers)
    if workers < 1:
        raise ValueError("SYCART Hessian worker count must be positive")
    if calculator.nn_enabled:
        cartesian = calculator.evaluate_nn_hessian_analytic(
            plan.coordinates_angstrom,
        ).hessian_hartree_per_bohr2
        hessian = plan.basis.T @ cartesian @ plan.basis
        hessian = 0.5 * (hessian + hessian.T)
        for left, left_irrep in enumerate(plan.irreps):
            for right, right_irrep in enumerate(plan.irreps):
                if left_irrep != right_irrep:
                    hessian[left, right] = 0.0
        # Analytic contraction: no displaced-gradient evaluations.
        return SYCARTHessianResult(hessian, plan, 0, (), ())
    coordinates_bohr = plan.coordinates_angstrom * ANGSTROM_TO_BOHR
    base = calculator._single_point(coordinates_bohr).gradient_hartree_per_bohr
    payload = {
        "atoms": calculator.atoms,
        "charge": calculator.charge,
        "multiplicity": calculator.multiplicity,
        "temperature_kelvin": calculator.temperature_kelvin,
        "scf_energy_tolerance_hartree": calculator.scf_energy_tolerance_hartree,
        "scf_convergence_mode": calculator.scf_convergence_mode,
        "scf_mixer_tolerance": calculator.scf_mixer_tolerance,
        "scf_wavefunction_tolerance_electron": calculator.scf_wavefunction_tolerance_electron,
        "maximum_scf_iterations": calculator.maximum_scf_iterations,
        "point_charges_electron": calculator.point_charges_electron,
        "point_charge_coordinates_angstrom": calculator.point_charge_coordinates_bohr * BOHR_TO_ANGSTROM,
        "diagonalizer_backend": calculator.diagonalizer_backend,
        "density_solver": calculator.solver_policy.mode,
        "dense_sp2_min_orbitals": calculator.solver_policy.dense_sp2_min_orbitals,
        "sparse_sp2_min_orbitals": calculator.solver_policy.sparse_sp2_min_orbitals,
        "sparse_sp2_maximum_fill_fraction": calculator.solver_policy.sparse_sp2_maximum_fill_fraction,
        "orthogonalizer_method": calculator.orthogonalizer_method,
        "symmetry_contract": calculator.symmetry_contract,
    }
    tasks = [
        (payload, coordinates_bohr, plan.basis, coordinate, plan.step_bohr,
         bool(plan.sign_symmetric[coordinate]), base)
        for coordinate in range(plan.basis.shape[1])
    ]
    if workers == 1:
        columns = map(_sycart_hessian_column, tasks)
    else:
        context = mp.get_context("fork" if os.name == "posix" else "spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            columns = tuple(executor.map(_sycart_hessian_column, tasks))
    if workers == 1:
        columns = tuple(columns)
    hessian = np.column_stack([item[1] for item in columns])
    hessian = 0.5 * (hessian + hessian.T)
    for left, left_irrep in enumerate(plan.irreps):
        for right, right_irrep in enumerate(plan.irreps):
            if left_irrep != right_irrep:
                hessian[left, right] = 0.0
    one_point = tuple(item[0] for item in columns if item[2])
    two_point = tuple(item[0] for item in columns if not item[2])
    evaluations = 1 + len(one_point) + 2 * len(two_point)
    return SYCARTHessianResult(hessian, plan, evaluations, one_point, two_point)


def _sycart_hessian_column(
    task: tuple[dict[str, object], np.ndarray, np.ndarray, int, float, bool, np.ndarray]
) -> tuple[int, np.ndarray, bool]:
    payload, coordinates, basis, coordinate, step_bohr, one_point, base = task
    calculator = GFN2Calculator(**payload)
    direction = basis[:, coordinate] * step_bohr
    positive = coordinates.reshape(-1) + direction
    positive_gradient = calculator._single_point(
        positive.reshape(coordinates.shape)
    ).gradient_hartree_per_bohr.reshape(-1)
    if one_point:
        derivative = (positive_gradient - base.reshape(-1)) / step_bohr
    else:
        negative = coordinates.reshape(-1) - direction
        negative_gradient = calculator._single_point(
            negative.reshape(coordinates.shape)
        ).gradient_hartree_per_bohr.reshape(-1)
        derivative = (positive_gradient - negative_gradient) / (2.0 * step_bohr)
    # A Hessian column is the derivative of the complete gradient.  Project
    # it onto every active SYCART direction, not only onto the displaced one.
    return coordinate, basis.T @ derivative, one_point


def hessian_vector_product(
    calculator: GFN2Calculator,
    coordinates_angstrom: np.ndarray,
    direction_bohr: np.ndarray,
    *,
    step: float = 5.0e-4,
    stencil_order: int = 4,
) -> HessianVectorResult:
    """Apply the full Hessian by differentiating the analytic gradient.

    NN-xTB instead contracts its complete analytic Cartesian Hessian.

    This matrix-free route includes Pulay, SCC, CN, multipolar, D4, embedding,
    and repulsion responses already assembled by the analytic gradient.  It is
    independent of Cartesian dimension: one product costs two gradients for a
    second-order stencil or four for a fourth-order stencil.
    """

    coordinates = np.asarray(coordinates_angstrom, dtype=float) * ANGSTROM_TO_BOHR
    direction = np.asarray(direction_bohr, dtype=float)
    if coordinates.shape != (len(calculator.atoms), 3):
        raise ValueError("Hessian-vector coordinates have the wrong shape")
    if direction.shape not in {coordinates.shape, (coordinates.size,)}:
        raise ValueError("Hessian-vector direction has the wrong shape")
    direction = direction.reshape(coordinates.shape)
    if not np.all(np.isfinite(direction)) or not np.any(direction):
        raise ValueError("Hessian-vector direction must be finite and nonzero")
    if step <= 0.0 or not np.isfinite(step):
        raise ValueError("Hessian-vector step must be positive and finite")
    if stencil_order not in {2, 4}:
        raise ValueError("Hessian-vector stencil order must be two or four")
    if calculator.nn_enabled:
        hessian = calculator.evaluate_nn_hessian_analytic(coordinates_angstrom)
        product = hessian.hessian_hartree_per_bohr2 @ direction.ravel()
        return HessianVectorResult(
            product.reshape(coordinates.shape), direction.copy(), 0.0, 0, 0,
            "analytic-NN-CPSCF-hessian",
        )

    def gradient(offset: float) -> np.ndarray:
        displaced = coordinates + offset * direction
        return (
            calculator.evaluate(displaced * BOHR_TO_ANGSTROM)
            if calculator.nn_enabled else calculator._single_point(displaced)
        ).gradient_hartree_per_bohr.reshape(-1)

    if stencil_order == 2:
        product = (gradient(step) - gradient(-step)) / (2.0 * step)
        evaluations = 2
    else:
        product = (
            -gradient(2.0 * step)
            + 8.0 * gradient(step)
            - 8.0 * gradient(-step)
            + gradient(-2.0 * step)
        ) / (12.0 * step)
        evaluations = 4
    return HessianVectorResult(
        product.reshape(coordinates.shape),
        direction.copy(),
        float(step),
        stencil_order,
        evaluations,
        ("directional-derivative-of-numerical-gradient"
         if calculator.nn_enabled else "directional-derivative-of-analytic-gradient"),
    )


def _hessian_column(
    task: tuple[dict[str, object], np.ndarray, int, float]
) -> tuple[int, np.ndarray]:
    payload, coordinates, coordinate, step_bohr = task
    calculator = GFN2Calculator(**payload)
    shape = coordinates.shape
    positive = coordinates.copy().reshape(-1)
    negative = coordinates.copy().reshape(-1)
    positive[coordinate] += step_bohr
    negative[coordinate] -= step_bohr
    positive_gradient = calculator._single_point(
        positive.reshape(shape)
    ).gradient_hartree_per_bohr.reshape(-1)
    negative_gradient = calculator._single_point(
        negative.reshape(shape)
    ).gradient_hartree_per_bohr.reshape(-1)
    return coordinate, (positive_gradient - negative_gradient) / (2.0 * step_bohr)


def harmonic_analysis(
    hessian_hartree_per_bohr2: np.ndarray,
    coordinates_angstrom: np.ndarray,
    masses_amu: np.ndarray,
    *,
    symmetry_adapter=None,
) -> HarmonicResult:
    """Project rigid motions and diagonalize a Cartesian harmonic Hessian."""

    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    masses = np.asarray(masses_amu, dtype=float)
    natoms = len(masses)
    dimension = 3 * natoms
    hessian = np.asarray(hessian_hartree_per_bohr2, dtype=float)
    if hessian.shape != (dimension, dimension):
        raise ValueError("harmonic Hessian and masses differ in dimension")
    if coordinates.shape != (natoms, 3) or np.any(masses <= 0.0):
        raise ValueError("harmonic coordinates or masses are invalid")
    if symmetry_adapter is not None:
        hessian = symmetry_adapter.symmetrize_hessian(hessian)
    centered = coordinates - np.sum(coordinates * masses[:, None], axis=0) / np.sum(masses)
    sqrt_mass = np.sqrt(masses)
    rigid = []
    for axis in range(3):
        vector = np.zeros((natoms, 3))
        vector[:, axis] = sqrt_mass
        rigid.append(vector.reshape(-1))
    axes = np.eye(3)
    for axis in axes:
        displacement = np.cross(np.broadcast_to(axis, centered.shape), centered)
        rigid.append((sqrt_mass[:, None] * displacement).reshape(-1))
    rigid_matrix = np.stack(rigid, axis=1)
    singular_values = np.linalg.svd(rigid_matrix, compute_uv=False)
    rank = int(np.sum(singular_values > singular_values[0] * 1.0e-10))
    rigid_basis = np.linalg.svd(rigid_matrix, full_matrices=True)[0][:, :rank]
    projector = np.eye(dimension) - rigid_basis @ rigid_basis.T
    inverse_sqrt_mass = 1.0 / np.repeat(sqrt_mass, 3)
    mass_weighted = hessian * inverse_sqrt_mass[:, None] * inverse_sqrt_mass[None, :]
    projected = projector @ mass_weighted @ projector
    projected = 0.5 * (projected + projected.T)
    irrep_for_vector: tuple[str, ...] = ()
    if symmetry_adapter is None:
        eigenvalues, eigenvectors = np.linalg.eigh(projected)
    else:
        transformation, blocks, labels = symmetry_adapter.cartesian_blocks()
        values: list[float] = []
        vectors: list[np.ndarray] = []
        vector_labels: list[str] = []
        for label, block in zip(labels, blocks, strict=True):
            basis = transformation[:, block]
            block_values, block_vectors = np.linalg.eigh(basis.T @ projected @ basis)
            values.extend(float(value) for value in block_values)
            vectors.extend(basis @ vector for vector in block_vectors.T)
            vector_labels.extend([label] * len(block_values))
        eigenvalues = np.asarray(values)
        eigenvectors = np.column_stack(vectors)
        irrep_for_vector = tuple(vector_labels)
    order = np.argsort(np.abs(eigenvalues))[rank:]
    values = eigenvalues[order]
    frequencies = np.sign(values) * np.sqrt(np.abs(values)) * FREQUENCY_FACTOR_CM1
    mode_vectors = eigenvectors[:, order].T
    labels: tuple[str, ...] = ()
    weights: tuple[float, ...] = ()
    if symmetry_adapter is not None:
        labels = tuple(irrep_for_vector[index] for index in order)
        weights = tuple(1.0 for _ in order)
    return HarmonicResult(hessian, frequencies, mode_vectors, rank, labels, weights)


__all__ = [
    "HarmonicResult",
    "HessianVectorResult",
    "OptimizationResult",
    "OptimizationStep",
    "harmonic_analysis",
    "hessian_vector_product",
    "numerical_hessian",
    "optimize_geometry",
]
