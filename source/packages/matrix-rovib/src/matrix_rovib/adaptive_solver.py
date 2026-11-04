"""Adaptive, restartable and parallel eigensolver routing for LOCK blocks."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Callable, Hashable, Sequence

import numpy as np
from matrix_numerics import diagonalize_hermitian


VectorAction = Callable[[np.ndarray], np.ndarray]


def _validate_hermitian_matrix(matrix):
    if not np.all(np.isfinite(matrix)):
        raise ValueError("LOCK eigenproblem matrix must be finite")
    scale = max(1., float(np.max(np.abs(matrix))))
    if np.max(np.abs(matrix - matrix.conj().T)) > 1.e-11 * scale:
        raise ValueError("LOCK eigenproblem matrix must be Hermitian")


@dataclass(frozen=True)
class AdaptiveSolverPolicy:
    dense_max_dimension: int = 384
    dense_root_fraction: float = 0.35
    tolerance: float = 1.0e-10
    maximum_iterations: int | None = None
    memory_budget_bytes: int = 2 * 1024**3
    shift_invert: bool = True

    def __post_init__(self) -> None:
        if self.dense_max_dimension < 1 or not 0.0 < self.dense_root_fraction <= 1.0:
            raise ValueError("invalid adaptive eigensolver crossover")
        if not math.isfinite(self.tolerance) or self.tolerance <= 0.0 or self.memory_budget_bytes < 1:
            raise ValueError("invalid adaptive eigensolver tolerance or memory budget")
        if self.maximum_iterations is not None and self.maximum_iterations < 1:
            raise ValueError("maximum_iterations must be positive")


@dataclass(frozen=True)
class EigenproblemBlock:
    key: Hashable
    dimension: int
    roots: int
    matrix: np.ndarray | None = None
    matvec: VectorAction | None = None
    shift_invert_matvec: VectorAction | None = None
    diagonal: np.ndarray | None = None
    target: float | None = None

    def __post_init__(self) -> None:
        if self.dimension < 1 or not 1 <= self.roots <= self.dimension:
            raise ValueError("invalid LOCK eigenproblem dimension or root count")
        if self.matrix is None and self.matvec is None:
            raise ValueError("an eigenproblem requires a matrix or matrix-vector action")
        if self.matrix is not None:
            matrix = np.asarray(self.matrix)
            if matrix.shape != (self.dimension, self.dimension):
                raise ValueError("eigenproblem matrix has an incompatible shape")
            _validate_hermitian_matrix(matrix)
        if self.diagonal is not None and np.asarray(self.diagonal).shape != (self.dimension,):
            raise ValueError("eigenproblem diagonal has an incompatible shape")
        if self.target is not None and not math.isfinite(self.target):
            raise ValueError("eigenproblem target must be finite")
        if self.shift_invert_matvec is not None and self.target is None:
            raise ValueError("shift_invert_matvec requires a finite target")


@dataclass(frozen=True)
class AdaptiveEigenResult:
    key: Hashable
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    residual_norms: np.ndarray
    backend: str
    reused_subspace: bool
    fallback: str | None = None


@dataclass
class EigensolverWorkspace:
    """Warm-start vectors retained between nonlinear-fit iterations."""

    vectors: dict[Hashable, np.ndarray] = field(default_factory=dict)

    def seed(self, key: Hashable, dimension: int) -> np.ndarray | None:
        value = self.vectors.get(key)
        if value is None or value.ndim != 2 or value.shape[0] != dimension:
            return None
        return np.asarray(value[:, 0], dtype=complex)

    def update(self, key: Hashable, vectors: np.ndarray) -> None:
        array = np.asarray(vectors, dtype=complex)
        if array.ndim != 2:
            raise ValueError("warm-start vectors must be a column matrix")
        self.vectors[key] = array.copy()


@dataclass(frozen=True)
class EigenstateTracking:
    """Stable correspondence between two eigensystems."""

    permutation: tuple[int, ...]
    squared_overlaps: tuple[float, ...]
    principal_overlaps: tuple[tuple[float, ...], ...]
    minimum_principal_overlap: float


@dataclass(frozen=True)
class AlignedEigenstates:
    """Current eigenvectors reordered and gauge-aligned to a reference."""

    tracking: EigenstateTracking
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray


def track_eigenstates_by_subspace(
    reference_values: np.ndarray,
    reference_vectors: np.ndarray,
    current_values: np.ndarray,
    current_vectors: np.ndarray,
    *,
    degeneracy_tolerance: float = 1.0e-8,
) -> EigenstateTracking:
    """Track states with Hungarian assignment and degenerate-subspace SVDs."""

    from scipy.optimize import linear_sum_assignment

    left_values = np.asarray(reference_values, dtype=float)
    right_values = np.asarray(current_values, dtype=float)
    left = np.asarray(reference_vectors, dtype=complex)
    right = np.asarray(current_vectors, dtype=complex)
    if (
        left_values.ndim != 1
        or right_values.shape != left_values.shape
        or left.shape != right.shape
        or left.shape[1] != left_values.size
    ):
        raise ValueError("eigenstate tracking requires equal complete eigensystems")
    if degeneracy_tolerance < 0.0:
        raise ValueError("degeneracy_tolerance must be nonnegative")
    squared = np.abs(left.conj().T @ right) ** 2
    rows, columns = linear_sum_assignment(-squared)
    initial = {int(row): int(column) for row, column in zip(rows, columns, strict=True)}
    permutation = np.empty(left_values.size, dtype=int)
    assigned_overlap = np.empty(left_values.size, dtype=float)
    principal_blocks = []
    minimum = 1.0
    for block in _degenerate_blocks(left_values, degeneracy_tolerance):
        mapped = tuple(initial[index] for index in block)
        cross = left[:, block].conj().T @ right[:, mapped]
        singular = np.linalg.svd(cross, compute_uv=False)
        principal_blocks.append(tuple(float(value) for value in singular))
        if singular.size:
            minimum = min(minimum, float(np.min(singular)))
        local_rows, local_columns = linear_sum_assignment(-np.abs(cross) ** 2)
        for row, column in zip(local_rows, local_columns, strict=True):
            reference_index = block[int(row)]
            current_index = mapped[int(column)]
            permutation[reference_index] = current_index
            assigned_overlap[reference_index] = squared[reference_index, current_index]
    return EigenstateTracking(
        permutation=tuple(int(value) for value in permutation),
        squared_overlaps=tuple(float(value) for value in assigned_overlap),
        principal_overlaps=tuple(principal_blocks),
        minimum_principal_overlap=minimum,
    )


def align_eigenstates_by_subspace(
    reference_values: np.ndarray,
    reference_vectors: np.ndarray,
    current_values: np.ndarray,
    current_vectors: np.ndarray,
    *,
    degeneracy_tolerance: float = 1.0e-8,
) -> AlignedEigenstates:
    """Reorder states and remove arbitrary phases or degenerate rotations.

    Degenerate manifolds use the unitary polar factor of their overlap matrix.
    This produces continuous eigenvectors for finite differences and nonlinear
    fits without assigning a physically meaningless phase to an eigenstate.
    """

    tracking = track_eigenstates_by_subspace(
        reference_values,
        reference_vectors,
        current_values,
        current_vectors,
        degeneracy_tolerance=degeneracy_tolerance,
    )
    values = np.asarray(current_values, dtype=float)[list(tracking.permutation)].copy()
    vectors = np.asarray(current_vectors, dtype=complex)[:, list(tracking.permutation)].copy()
    reference = np.asarray(reference_vectors, dtype=complex)
    for block in _degenerate_blocks(np.asarray(reference_values, dtype=float), degeneracy_tolerance):
        indices = list(block)
        overlap = reference[:, indices].conj().T @ vectors[:, indices]
        left, _singular, right_h = np.linalg.svd(overlap)
        # If C = R^H V, rotate V by the adjoint polar factor so R^H V is positive.
        rotation = right_h.conj().T @ left.conj().T
        vectors[:, indices] = vectors[:, indices] @ rotation
    return AlignedEigenstates(tracking, values, vectors)


def _degenerate_blocks(values: np.ndarray, tolerance: float) -> tuple[tuple[int, ...], ...]:
    blocks: list[list[int]] = []
    for index, value in enumerate(values):
        if not blocks:
            blocks.append([index])
            continue
        previous = values[blocks[-1][-1]]
        scale = max(1.0, abs(float(value)), abs(float(previous)))
        if abs(float(value - previous)) <= tolerance * scale:
            blocks[-1].append(index)
        else:
            blocks.append([index])
    return tuple(tuple(block) for block in blocks)


def write_eigensolver_workspace(path: Path | str, workspace: EigensolverWorkspace) -> Path:
    """Write warm-start vectors atomically as a portable compressed NPZ."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    arrays = {}
    for index, (key, vectors) in enumerate(workspace.vectors.items()):
        try:
            encoded_key = json.dumps(key, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("workspace keys must be JSON serializable") from exc
        name = f"vectors_{index}"
        manifest.append({"key": encoded_key, "array": name})
        arrays[name] = np.asarray(vectors, dtype=complex)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, manifest=np.asarray(json.dumps(manifest)), **arrays)
    temporary.replace(target)
    return target


def read_eigensolver_workspace(path: Path | str) -> EigensolverWorkspace:
    """Restore a warm-start workspace written by :func:`write_eigensolver_workspace`."""

    workspace = EigensolverWorkspace()
    with np.load(Path(path), allow_pickle=False) as archive:
        manifest = json.loads(str(archive["manifest"].item()))
        for item in manifest:
            key = json.loads(item["key"])
            if isinstance(key, list):
                key = tuple(key)
            workspace.update(key, np.asarray(archive[item["array"]], dtype=complex))
    return workspace


def solve_eigenproblem(
    problem: EigenproblemBlock,
    *,
    policy: AdaptiveSolverPolicy = AdaptiveSolverPolicy(),
    workspace: EigensolverWorkspace | None = None,
    diagonalizer_backend: str | None = None,
) -> AdaptiveEigenResult:
    """Choose dense, partial dense, Lanczos, or shift-invert automatically."""

    matrix = None if problem.matrix is None else np.asarray(problem.matrix, dtype=complex)
    dense_bytes = problem.dimension * problem.dimension * np.dtype(complex).itemsize
    dense = (
        problem.roots == problem.dimension
        or problem.dimension <= policy.dense_max_dimension
        or problem.roots / problem.dimension >= policy.dense_root_fraction
    )
    if matrix is None and dense and dense_bytes <= policy.memory_budget_bytes:
        identity = np.eye(problem.dimension, dtype=complex)
        matrix = np.column_stack(
            [problem.matvec(identity[:, index]) for index in range(problem.dimension)]
        )
        _validate_hermitian_matrix(matrix)
        matrix = 0.5 * (matrix + matrix.conj().T)
    if problem.roots == problem.dimension and (
        matrix is None or dense_bytes > policy.memory_budget_bytes
    ):
        raise MemoryError(
            "all eigenpairs require dense storage; increase memory_budget_bytes "
            "or request fewer roots"
        )
    if matrix is not None and dense and dense_bytes <= policy.memory_budget_bytes:
        result = diagonalize_hermitian(
            matrix,
            backend=diagonalizer_backend,
            subset_by_index=(0, problem.roots - 1) if problem.roots < problem.dimension else None,
        )
        if result.eigenvectors is None:
            raise RuntimeError("adaptive dense diagonalizer did not return eigenvectors")
        values = np.asarray(result.eigenvalues)
        vectors = np.asarray(result.eigenvectors)
        residuals = np.linalg.norm(matrix @ vectors - vectors * values[None, :], axis=0)
        answer = AdaptiveEigenResult(
            problem.key,
            values,
            vectors,
            residuals,
            f"dense-{result.backend}",
            False,
        )
    else:
        answer = _solve_sparse(problem, policy=policy, workspace=workspace)
        threshold = policy.tolerance * np.maximum(1.0, np.abs(answer.eigenvalues))
        if np.any(answer.residual_norms > threshold):
            if matrix is None or dense_bytes > policy.memory_budget_bytes:
                raise RuntimeError(
                    f"LOCK iterative eigensolver failed residual certification for {problem.key!r}"
                )
            result = diagonalize_hermitian(
                matrix,
                backend=diagonalizer_backend,
                subset_by_index=(0, problem.roots - 1),
            )
            if result.eigenvectors is None:
                raise RuntimeError("adaptive fallback did not return eigenvectors")
            values, vectors = np.asarray(result.eigenvalues), np.asarray(result.eigenvectors)
            answer = AdaptiveEigenResult(
                problem.key,
                values,
                vectors,
                _residuals(problem, values, vectors),
                f"dense-{result.backend}",
                answer.reused_subspace,
                fallback=answer.backend,
            )
    if workspace is not None:
        workspace.update(problem.key, answer.eigenvectors)
    return answer


def _solve_sparse(
    problem: EigenproblemBlock,
    *,
    policy: AdaptiveSolverPolicy,
    workspace: EigensolverWorkspace | None,
) -> AdaptiveEigenResult:
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import LinearOperator, eigsh

    k = min(problem.roots, problem.dimension - 1)
    if k < 1:
        raise ValueError("iterative diagonalization requires a dimension above one")
    seed = None if workspace is None else workspace.seed(problem.key, problem.dimension)
    if seed is None:
        if problem.diagonal is None:
            seed = np.ones(problem.dimension, dtype=complex)
        else:
            index = int(np.argmin(np.abs(np.asarray(problem.diagonal) - (problem.target or 0.0))))
            seed = np.zeros(problem.dimension, dtype=complex)
            seed[index] = 1.0
    seed /= np.linalg.norm(seed)
    kwargs = {
        "k": k,
        "tol": policy.tolerance,
        "maxiter": policy.maximum_iterations or max(1000, 20 * problem.dimension),
        "v0": seed,
    }
    backend = "scipy-eigsh"
    if problem.matrix is not None:
        operator = csc_matrix(np.asarray(problem.matrix, dtype=complex))
        if policy.shift_invert and problem.target is not None:
            values, vectors = eigsh(operator, sigma=problem.target, which="LM", **kwargs)
            backend += "-shift-invert"
        else:
            values, vectors = eigsh(operator, which="SA", **kwargs)
    else:
        operator = LinearOperator(
            (problem.dimension, problem.dimension),
            matvec=problem.matvec,
            dtype=np.complex128,
        )
        if policy.shift_invert and problem.target is not None and problem.shift_invert_matvec:
            inverse = LinearOperator(
                (problem.dimension, problem.dimension),
                matvec=problem.shift_invert_matvec,
                dtype=np.complex128,
            )
            values, vectors = eigsh(
                operator, sigma=problem.target, which="LM", OPinv=inverse, **kwargs
            )
            backend += "-matrix-free-shift-invert"
        else:
            values, vectors = eigsh(operator, which="SA", **kwargs)
    order = np.argsort(values, kind="stable")
    values, vectors = values[order], vectors[:, order]
    return AdaptiveEigenResult(
        problem.key,
        values,
        vectors,
        _residuals(problem, values, vectors),
        backend,
        workspace is not None and workspace.seed(problem.key, problem.dimension) is not None,
    )


def _residuals(problem: EigenproblemBlock, values: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    if problem.matrix is not None:
        products = np.asarray(problem.matrix) @ vectors
    else:
        products = np.column_stack(
            [problem.matvec(vectors[:, index]) for index in range(vectors.shape[1])]
        )
    return np.linalg.norm(products - vectors * values[None, :], axis=0)


def solve_eigenblocks(
    problems: Sequence[EigenproblemBlock],
    *,
    policy: AdaptiveSolverPolicy = AdaptiveSolverPolicy(),
    workspace: EigensolverWorkspace | None = None,
    workers: int = 1,
    diagonalizer_backend: str | None = None,
) -> tuple[AdaptiveEigenResult, ...]:
    """Solve independent symmetry blocks in parallel with stable output order."""

    ordered = tuple(problems)
    if workers < 1:
        raise ValueError("LOCK eigensolver workers must be positive")
    if len({problem.key for problem in ordered}) != len(ordered):
        raise ValueError("parallel LOCK eigenproblem keys must be unique")

    def solve(problem: EigenproblemBlock) -> AdaptiveEigenResult:
        local_workspace = None
        if workspace is not None:
            local_workspace = EigensolverWorkspace()
            seed = workspace.vectors.get(problem.key)
            if seed is not None:
                local_workspace.update(problem.key, seed)
        return solve_eigenproblem(
            problem,
            policy=policy,
            workspace=local_workspace,
            diagonalizer_backend=diagonalizer_backend,
        )

    if workers == 1 or len(ordered) < 2:
        results = tuple(solve(problem) for problem in ordered)
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(ordered))) as executor:
            futures = [executor.submit(solve, problem) for problem in ordered]
            results = tuple(future.result() for future in futures)
    if workspace is not None:
        # Commit warm starts in canonical input order after workers finish.
        for result in results:
            workspace.update(result.key, result.eigenvectors)
    return results


__all__ = [
    "AdaptiveEigenResult",
    "AdaptiveSolverPolicy",
    "AlignedEigenstates",
    "EigenstateTracking",
    "EigenproblemBlock",
    "EigensolverWorkspace",
    "read_eigensolver_workspace",
    "solve_eigenblocks",
    "solve_eigenproblem",
    "align_eigenstates_by_subspace",
    "track_eigenstates_by_subspace",
    "write_eigensolver_workspace",
]
