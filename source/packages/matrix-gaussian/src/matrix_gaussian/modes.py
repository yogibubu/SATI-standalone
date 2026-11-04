"""Canonical conversion of Gaussian normal coordinates.

Gaussian prints and stores Cartesian displacement vectors with unit Cartesian
norm.  MATRIX stores normal modes as orthonormal vectors in mass-weighted
Cartesian space.  This module is the only conversion boundary between those
two conventions.
"""

from __future__ import annotations

import numpy as np


GAUSSIAN_MODE_CONVERSION_PROTOCOL = "matrix.gaussian.normal_modes.v1"
MASS_WEIGHTED_ORTHONORMALITY_TOLERANCE = 1.0e-6


def gaussian_cartesian_modes_to_mass_weighted(
    modes,
    masses_amu,
    *,
    coordinate_count: int | None = None,
) -> np.ndarray:
    """Return Gaussian Cartesian modes as mass-weighted orthonormal rows.

    If ``L`` contains Gaussian's Cartesian displacement rows and ``M`` is the
    diagonal atomic-mass matrix, the canonical MATRIX rows are
    ``Q_i = M**(1/2) L_i / ||M**(1/2) L_i||``.  The strict Gram-matrix check is
    deliberate: malformed, transposed, or already reweighted input must fail
    before any displaced electronic-structure calculation is generated.
    """
    values = np.asarray(modes, dtype=float)
    masses = np.asarray(masses_amu, dtype=float).reshape(-1)
    if masses.size < 1 or np.any(~np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("Gaussian normal-mode conversion requires positive atomic masses")
    expected_coordinates = 3 * masses.size
    if coordinate_count is None:
        coordinate_count = expected_coordinates
    if int(coordinate_count) != expected_coordinates:
        raise ValueError(
            f"Gaussian normal-mode coordinate count must be 3N={expected_coordinates}"
        )
    if values.size == 0 or values.size % expected_coordinates != 0:
        raise ValueError("Gaussian normal-mode array is not divisible by 3N")
    cartesian = values.reshape((-1, expected_coordinates))
    if np.any(~np.isfinite(cartesian)):
        raise ValueError("Gaussian normal-mode array contains non-finite values")
    sqrt_masses = np.sqrt(np.repeat(masses, 3))
    mass_weighted = cartesian * sqrt_masses[None, :]
    norms = np.linalg.norm(mass_weighted, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise ValueError("Gaussian normal-mode array contains a zero-norm mode")
    mass_weighted /= norms[:, None]
    gram = mass_weighted @ mass_weighted.T
    if not np.allclose(
        gram,
        np.eye(mass_weighted.shape[0]),
        atol=MASS_WEIGHTED_ORTHONORMALITY_TOLERANCE,
        rtol=MASS_WEIGHTED_ORTHONORMALITY_TOLERANCE,
    ):
        deviation = float(np.max(np.abs(gram - np.eye(mass_weighted.shape[0]))))
        raise ValueError(
            "Gaussian modes do not become mass-weighted orthonormal under the "
            f"canonical conversion (maximum Gram deviation {deviation:.3e})"
        )
    return mass_weighted
