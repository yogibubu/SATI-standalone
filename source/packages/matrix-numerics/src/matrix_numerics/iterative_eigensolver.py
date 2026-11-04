"""Shared deterministic matrix-free eigensolver routing for MATRIX."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import importlib.util
import os
from pathlib import Path
from typing import Callable, Literal

import numpy as np


VectorAction = Callable[[np.ndarray], np.ndarray]
BlockAction = Callable[[np.ndarray], np.ndarray]
IterativeBackend = Literal["auto", "python-davidson", "diaglib-davidson"]


@dataclass(frozen=True)
class IterativeEigenResult:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    residual_norms: np.ndarray
    operator_batches: int
    converged: bool
    backend: str

    def __post_init__(self) -> None:
        values = np.array(self.eigenvalues, dtype=float, copy=True)
        vectors = np.array(self.eigenvectors, dtype=float, copy=True)
        residuals = np.array(self.residual_norms, dtype=float, copy=True)
        if values.ndim != 1 or vectors.shape[0] != len(values):
            raise ValueError("iterative eigenpair dimensions are inconsistent")
        if residuals.shape != values.shape or vectors.ndim != 2:
            raise ValueError("iterative eigenpair residual dimensions are inconsistent")
        if not all(np.all(np.isfinite(item)) for item in (values, vectors, residuals)):
            raise ValueError("iterative eigenpairs must be finite")
        for item in (values, vectors, residuals):
            item.setflags(write=False)
        object.__setattr__(self, "eigenvalues", values)
        object.__setattr__(self, "eigenvectors", vectors)
        object.__setattr__(self, "residual_norms", residuals)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _diaglib_library_candidates() -> tuple[Path, ...]:
    configured = os.environ.get("MATRIX_DIAGLIB_C_LIBRARY") or os.environ.get(
        "DIAGLIB_C_LIBRARY"
    )
    candidates = [] if not configured else [Path(configured).expanduser()]
    root = _repository_root() / "third_party" / "diaglib"
    names = ("libdiaglib_c.dylib", "libdiaglib_c.so", "diaglib_c.dll")
    for build in (root / "build", root / "build-release"):
        for name in names:
            candidates.extend((build / "src" / "c_interface" / name, build / "lib" / name))
    return tuple(path.resolve() for path in candidates)


@lru_cache(maxsize=1)
def _load_diaglib():
    try:
        import pyDiaglib  # type: ignore[import-not-found]

        module = pyDiaglib
    except ImportError:
        source = (
            _repository_root()
            / "third_party"
            / "diaglib"
            / "src"
            / "python_interface"
            / "src"
            / "pyDiaglib.py"
        )
        if not source.is_file():
            return None
        specification = importlib.util.spec_from_file_location("_matrix_vendored_pydiaglib", source)
        if specification is None or specification.loader is None:
            return None
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
    library = next((path for path in _diaglib_library_candidates() if path.is_file()), None)
    if library is None:
        return None
    return module, library


def diaglib_available() -> bool:
    """Return whether the pinned native DiagLib backend can be loaded."""

    return _load_diaglib() is not None


def iterative_eigensolver_backends() -> tuple[str, ...]:
    available = ["python-davidson"]
    if diaglib_available():
        available.append("diaglib-davidson")
    return tuple(available)


def _apply_block(action: VectorAction, vectors: np.ndarray) -> np.ndarray:
    return np.column_stack([action(vectors[:, column]) for column in range(vectors.shape[1])])


def _certify_residuals(
    block_action: BlockAction,
    eigenvalues: np.ndarray,
    eigenvectors_by_row: np.ndarray,
    metric_action: BlockAction | None,
) -> np.ndarray:
    vectors = eigenvectors_by_row.T
    product = block_action(vectors)
    metric_vectors = vectors if metric_action is None else metric_action(vectors)
    return np.linalg.norm(product - metric_vectors * eigenvalues[None, :], axis=0)


def _python_davidson(
    block_action: BlockAction,
    diagonal: np.ndarray,
    roots: int,
    *,
    tolerance: float,
    maximum_iterations: int,
    maximum_subspace: int,
    initial_vectors: np.ndarray,
) -> IterativeEigenResult:
    basis, _ = np.linalg.qr(initial_vectors, mode="reduced")
    products = block_action(basis)
    batches = 1
    eigenvalues = diagonal[np.argsort(diagonal, kind="stable")[:roots]].copy()
    ritz = basis[:, :roots].copy()
    residual_norms = np.full(roots, np.inf)
    for iteration in range(1, maximum_iterations + 1):
        projected = basis.T @ products
        projected = 0.5 * (projected + projected.T)
        values, vectors = np.linalg.eigh(projected)
        order = np.argsort(values, kind="stable")[:roots]
        eigenvalues = values[order]
        coefficients = vectors[:, order]
        ritz = basis @ coefficients
        sigma = products @ coefficients
        residuals = sigma - ritz * eigenvalues
        residual_norms = np.linalg.norm(residuals, axis=0)
        if np.all(residual_norms <= tolerance):
            return IterativeEigenResult(
                eigenvalues, ritz.T, residual_norms, batches, True, "python-davidson"
            )
        corrections = []
        for root in range(roots):
            if residual_norms[root] <= tolerance:
                continue
            denominator = eigenvalues[root] - diagonal
            safe = np.where(
                np.abs(denominator) < 1.0e-10,
                np.copysign(1.0e-10, denominator + 1.0e-300),
                denominator,
            )
            trial = residuals[:, root] / safe
            trial -= basis @ (basis.T @ trial)
            for previous in corrections:
                trial -= previous * np.dot(previous, trial)
            norm = float(np.linalg.norm(trial))
            if norm > 1.0e-11:
                corrections.append(trial / norm)
        if not corrections:
            break
        if basis.shape[1] + len(corrections) > maximum_subspace:
            basis, _ = np.linalg.qr(ritz, mode="reduced")
            products = block_action(basis)
        else:
            added = np.column_stack(corrections)
            basis = np.column_stack((basis, added))
            products = np.column_stack((products, block_action(added)))
        batches += 1
    return IterativeEigenResult(
        eigenvalues, ritz.T, residual_norms, batches, False, "python-davidson"
    )


def _diaglib_davidson(
    block_action: BlockAction,
    diagonal: np.ndarray,
    roots: int,
    *,
    tolerance: float,
    maximum_iterations: int,
    maximum_subspace: int,
    initial_vectors: np.ndarray,
    metric_action: BlockAction | None,
) -> IterativeEigenResult:
    loaded = _load_diaglib()
    if loaded is None:
        raise RuntimeError("the vendored DiagLib native library is not built")
    module, library = loaded
    dimension = len(diagonal)
    nmax = min(max(roots + 4, 2 * roots), maximum_subspace, (dimension - 1) // 2)
    if nmax < roots or 2 * nmax >= dimension:
        raise ValueError("DiagLib expansion arrays do not fit this small eigenproblem")
    eigenvalues = np.zeros(nmax, dtype=np.float64)
    eigenvectors = np.zeros((dimension, nmax), dtype=np.float64, order="F")
    deterministic_seeds = np.argsort(diagonal, kind="stable")[:nmax]
    eigenvectors[deterministic_seeds, np.arange(nmax)] = 1.0
    columns = min(initial_vectors.shape[1], nmax)
    eigenvectors[:, :columns] = initial_vectors[:, :columns]
    batches = 0

    def as_matrix(pointer, rows, columns):
        return np.ctypeslib.as_array(pointer, shape=(rows * columns,)).reshape(
            (rows, columns), order="F"
        )

    def matvec(n_pointer, m_pointer, x_pointer, y_pointer):
        nonlocal batches
        rows, columns = n_pointer[0], m_pointer[0]
        x = as_matrix(x_pointer, rows, columns)
        as_matrix(y_pointer, rows, columns)[:, :] = block_action(x)
        batches += 1

    def precondition(n_pointer, m_pointer, shift_pointer, r_pointer, z_pointer):
        rows, columns = n_pointer[0], m_pointer[0]
        residual = as_matrix(r_pointer, rows, columns)
        denominator = diagonal[:, None] + shift_pointer[0]
        safe = np.where(
            np.abs(denominator) < 1.0e-10,
            np.copysign(1.0e-10, denominator + 1.0e-300),
            denominator,
        )
        as_matrix(z_pointer, rows, columns)[:, :] = residual / safe

    def metric(n_pointer, m_pointer, x_pointer, y_pointer):
        rows, columns = n_pointer[0], m_pointer[0]
        x = as_matrix(x_pointer, rows, columns)
        as_matrix(y_pointer, rows, columns)[:, :] = metric_action(x)

    solver = module.diaglib(
        str(library),
        dimension,
        roots,
        nmax,
        tol=tolerance / np.sqrt(dimension),
        max_iter=maximum_iterations,
        dav_iter=max(4, maximum_subspace // nmax),
        memory=max(80, int(8 * dimension * maximum_subspace / 1024**2) + 32),
    )
    ok = solver.dgl_davidson_driver(
        eigenvalues,
        eigenvectors,
        matvec,
        precondition,
        metvec=None if metric_action is None else metric,
    )
    values = eigenvalues[:roots].copy()
    vectors = eigenvectors[:, :roots].T.copy()
    residuals = _certify_residuals(block_action, values, vectors, metric_action)
    return IterativeEigenResult(
        values,
        vectors,
        residuals,
        batches,
        bool(ok and np.all(residuals <= tolerance)),
        "diaglib-davidson",
    )


def solve_symmetric_matrix_free(
    matvec: VectorAction,
    diagonal: np.ndarray,
    *,
    roots: int = 1,
    block_matvec: BlockAction | None = None,
    metric_matvec: VectorAction | None = None,
    block_metric_matvec: BlockAction | None = None,
    backend: IterativeBackend = "auto",
    tolerance: float = 1.0e-8,
    maximum_iterations: int = 100,
    maximum_subspace: int | None = None,
    initial_vectors: np.ndarray | None = None,
    diaglib_minimum_dimension: int = 256,
) -> IterativeEigenResult:
    """Compute the lowest eigenpairs with deterministic backend routing."""

    diagonal_array = np.asarray(diagonal, dtype=float)
    if diagonal_array.ndim != 1 or not len(diagonal_array) or not np.all(
        np.isfinite(diagonal_array)
    ):
        raise ValueError("matrix-free diagonal must be a finite nonempty vector")
    dimension = len(diagonal_array)
    nroots = int(roots)
    if nroots < 1 or nroots > dimension:
        raise ValueError("requested roots must lie in [1, dimension]")
    if tolerance <= 0.0 or maximum_iterations < 1:
        raise ValueError("iterative eigensolver tolerances and limits must be positive")
    maximum = max(2 * nroots + 4, 20) if maximum_subspace is None else int(maximum_subspace)
    if maximum <= nroots:
        raise ValueError("maximum subspace must exceed the number of roots")
    action = block_matvec or (lambda vectors: _apply_block(matvec, vectors))
    metric_action = block_metric_matvec
    if metric_action is None and metric_matvec is not None:
        metric_action = lambda vectors: _apply_block(metric_matvec, vectors)
    if initial_vectors is None:
        seeds = np.argsort(diagonal_array, kind="stable")[:nroots]
        guesses = np.eye(dimension)[:, seeds]
    else:
        guesses = np.asarray(initial_vectors, dtype=float)
        if guesses.ndim != 2 or guesses.shape[0] != dimension or guesses.shape[1] < nroots:
            raise ValueError("initial eigenvectors have incompatible dimensions")
        if not np.all(np.isfinite(guesses)):
            raise ValueError("initial eigenvectors must be finite")
    normalized = str(backend).strip().casefold()
    if normalized not in {"auto", "python-davidson", "diaglib-davidson"}:
        raise ValueError(f"unsupported iterative eigensolver backend {backend!r}")
    diaglib_eligible = (
        diaglib_available()
        and dimension >= int(diaglib_minimum_dimension)
        and 2 * min(max(nroots + 4, 2 * nroots), maximum) < dimension
    )
    if normalized == "diaglib-davidson" and not diaglib_eligible:
        raise RuntimeError("DiagLib is unavailable or the eigenproblem is too small")
    # A dimension-only crossover is not valid for matrix-free operators: its
    # location depends primarily on callback cost and convergence behavior.
    # Keep standard problems on the resident solver unless the caller selects
    # DiagLib explicitly.  Generalized problems require its metric-aware path.
    use_diaglib = normalized == "diaglib-davidson" or (
        normalized == "auto" and metric_action is not None and diaglib_eligible
    )
    if use_diaglib:
        return _diaglib_davidson(
            action,
            diagonal_array,
            nroots,
            tolerance=tolerance,
            maximum_iterations=maximum_iterations,
            maximum_subspace=maximum,
            initial_vectors=guesses,
            metric_action=metric_action,
        )
    if metric_action is not None:
        raise RuntimeError("generalized matrix-free Davidson requires the DiagLib backend")
    return _python_davidson(
        action,
        diagonal_array,
        nroots,
        tolerance=tolerance,
        maximum_iterations=maximum_iterations,
        maximum_subspace=maximum,
        initial_vectors=guesses,
    )


__all__ = [
    "IterativeEigenResult",
    "diaglib_available",
    "iterative_eigensolver_backends",
    "solve_symmetric_matrix_free",
]
