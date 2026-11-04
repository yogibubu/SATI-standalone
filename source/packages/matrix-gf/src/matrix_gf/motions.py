"""Cartesian normal-mode motions derived from Cartesian or SONIC Hessians."""

from __future__ import annotations

import numpy as np

from matrix_qm import CartesianNormalModeSet, cartesian_normal_modes_from_hessian

from .internal import BOHR_TO_ANGSTROM


def cartesian_normal_modes_from_sonic_hessian(
    sonic_hessian: np.ndarray,
    sonic_b_matrix_per_angstrom: np.ndarray,
    masses_amu: np.ndarray,
    coordinates_bohr: np.ndarray,
    *,
    sonic_gradient: np.ndarray | None = None,
    gradient_curvature_cartesian_au: np.ndarray | None = None,
    source: str = "SONIC Hessian",
) -> CartesianNormalModeSet:
    """Transform a frozen-SONIC Hessian and obtain invariant Cartesian modes.

    The exact Hessian transformation away from a stationary point contains the
    coordinate-curvature term ``sum_k g_k d2q_k/dx2``.  A nonzero SONIC gradient
    is therefore rejected unless that Cartesian correction is supplied.
    """
    force = np.asarray(sonic_hessian, dtype=float)
    b_matrix = np.asarray(sonic_b_matrix_per_angstrom, dtype=float)
    if force.ndim != 2 or force.shape[0] != force.shape[1]:
        raise ValueError("SONIC Hessian must be a square matrix")
    if b_matrix.ndim != 2 or b_matrix.shape[0] != force.shape[0]:
        raise ValueError("SONIC Hessian order must match the frozen SONIC B matrix")
    if not np.allclose(force, force.T, atol=1.0e-10):
        raise ValueError("SONIC Hessian must be symmetric")
    gradient = None if sonic_gradient is None else np.asarray(sonic_gradient, dtype=float)
    correction = gradient_curvature_cartesian_au
    if gradient is not None:
        if gradient.shape != (force.shape[0],):
            raise ValueError("SONIC gradient length must match the SONIC Hessian")
        if np.linalg.norm(gradient) > 1.0e-10 and correction is None:
            raise ValueError(
                "a non-stationary SONIC Hessian requires the gradient-curvature Cartesian term"
            )
    # B is evaluated with Angstrom Cartesians.  Convert H_x from hartree/A^2
    # to hartree/bohr^2 before the common mass-weighted diagonalization.
    cartesian = (b_matrix.T @ force @ b_matrix) * BOHR_TO_ANGSTROM**2
    if correction is not None:
        correction_array = np.asarray(correction, dtype=float)
        if correction_array.shape != cartesian.shape:
            raise ValueError("gradient-curvature correction has the wrong Cartesian shape")
        cartesian = cartesian + correction_array
    return cartesian_normal_modes_from_hessian(
        cartesian,
        masses_amu,
        coordinates_bohr,
        source=source,
    )
