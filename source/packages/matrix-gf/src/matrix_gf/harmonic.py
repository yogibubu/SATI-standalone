from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CM_PER_HARTREE = 219474.6313705


def __getattr__(name):
    # A prepared vibronic model needs the conversion constant alone. Preserve
    # these historical re-exports without importing the full QM stack merely
    # to contract an already supplied harmonic model.
    if name in {'HESSIAN_EIGENVALUE_TO_CM','mass_weighted_cartesian_hessian'}:
        import matrix_qm
        value=getattr(matrix_qm,name);globals()[name]=value
        return value
    raise AttributeError(name)


@dataclass(frozen=True)
class GFResult:
    """Wilson GF harmonic result in an already non-redundant coordinate basis."""

    eigenvalues: np.ndarray
    frequencies_cm: np.ndarray
    normal_modes: np.ndarray


def solve_wilson_gf(
    force_constants: np.ndarray,
    g_matrix: np.ndarray,
    *,
    scale_to_cm: bool = False,
) -> GFResult:
    """Solve the symmetric Wilson GF eigenproblem from independent matrices."""
    from matrix_numerics import eigh_arrays
    f_mat = np.asarray(force_constants, dtype=float)
    g_mat = np.asarray(g_matrix, dtype=float)
    if f_mat.shape != g_mat.shape or f_mat.ndim != 2 or f_mat.shape[0] != f_mat.shape[1]:
        raise ValueError("F and G must be square matrices with the same shape")

    g_eval, g_vec = eigh_arrays((g_mat + g_mat.T) * 0.5)
    if np.any(g_eval <= 0.0):
        raise ValueError("G matrix must be positive definite")
    g_half = (g_vec * np.sqrt(g_eval)) @ g_vec.T
    sym = g_half @ ((f_mat + f_mat.T) * 0.5) @ g_half
    eig, vec = eigh_arrays((sym + sym.T) * 0.5)
    order = np.argsort(eig)
    eig = eig[order]
    vec = vec[:, order]
    freqs = np.sign(eig) * np.sqrt(np.abs(eig))
    if scale_to_cm:
        freqs = freqs * HESSIAN_EIGENVALUE_TO_CM
    return GFResult(eigenvalues=eig, frequencies_cm=freqs, normal_modes=vec)
