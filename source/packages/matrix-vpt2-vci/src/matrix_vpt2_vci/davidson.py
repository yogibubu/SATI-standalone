from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from matrix_numerics import solve_symmetric_matrix_free


MatVec = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class DavidsonResult:
    """Lowest eigenpairs from the shared MATRIX matrix-free solver."""

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    residual_norms: np.ndarray
    iterations: int
    converged: bool
    backend: str = "python-davidson"


def davidson_lowest(
    matvec: MatVec,
    diagonal: np.ndarray,
    *,
    n_roots: int,
    max_subspace: int = 80,
    max_iter: int = 200,
    convergence: float = 1.0e-8,
    block_matvec: Callable[[np.ndarray], np.ndarray] | None = None,
    backend: str = "auto",
) -> DavidsonResult:
    """Compute the lowest VCI eigenpairs through the common native router.

    The default is the deterministic in-tree Davidson implementation.  The
    pinned DiagLib backend is available explicitly for large or generalized
    problems, with the same matrix-free Hamiltonian contract and independently
    certified residuals.
    """
    solved = solve_symmetric_matrix_free(
        matvec,
        diagonal,
        roots=n_roots,
        block_matvec=block_matvec,
        backend=backend,
        tolerance=convergence,
        maximum_iterations=max_iter,
        maximum_subspace=max_subspace,
    )
    return DavidsonResult(
        eigenvalues=solved.eigenvalues,
        eigenvectors=solved.eigenvectors.T,
        residual_norms=solved.residual_norms,
        iterations=solved.operator_batches,
        converged=solved.converged,
        backend=solved.backend,
    )
