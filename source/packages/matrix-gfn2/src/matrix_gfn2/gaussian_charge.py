"""Coulomb interaction of normalized spherical Gaussian charge densities.

Widths are defined by rho(r) = q exp(-r**2/sigma**2)/(pi**1.5 sigma**3).
A zero width denotes a point charge. This radial utility does not change
the GFN2 Hamiltonian or its default interaction model.
"""

from __future__ import annotations

from math import erf

import numpy as np


def gaussian_charge_coulomb(
    distance_bohr,
    width_left_bohr,
    width_right_bohr=0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return interaction per unit charge product and two radial derivatives.

    All inputs broadcast. The two widths combine as sqrt(sigma_a**2+sigma_b**2).
    Values use atomic units; the first and second derivatives are with respect
    to the nonnegative intercenter distance. At coincidence the analytic limit
    is used, including a zero first derivative and a finite second derivative.
    """
    r, left, right = np.broadcast_arrays(
        np.asarray(distance_bohr, dtype=float),
        np.asarray(width_left_bohr, dtype=float),
        np.asarray(width_right_bohr, dtype=float),
    )
    if any(np.any(~np.isfinite(x)) for x in (r, left, right)):
        raise ValueError("Gaussian charge distances and widths must be finite")
    if np.any(r < 0) or np.any(left < 0) or np.any(right < 0):
        raise ValueError("Gaussian charge distances and widths must be nonnegative")
    sigma = np.hypot(left, right)
    if np.any(sigma == 0):
        raise ValueError("at least one Gaussian charge width must be positive")
    t = r / sigma
    value, first, second = (np.empty_like(t) for _ in range(3))
    small = t <= 0.01
    # A common series avoids cancellation in both radial derivatives.
    x = t[small] ** 2
    prefactor = 2 / (np.sqrt(np.pi) * sigma[small])
    value[small] = prefactor * (1 + x * (-1/3 + x * (1/10 + x * (-1/42 + x/216))))
    first[small] = prefactor * r[small] / sigma[small]**2 * (
        -2/3 + x * (2/5 + x * (-1/7 + x/27))
    )
    second[small] = prefactor / sigma[small]**2 * (
        -2/3 + x * (6/5 + x * (-5/7 + 7*x/27))
    )
    large = ~small
    rr, ss, tt = r[large], sigma[large], t[large]
    error_function = np.fromiter(
        (erf(float(v)) for v in tt.flat), dtype=float, count=tt.size,
    ).reshape(tt.shape)
    exponential = 2 / (np.sqrt(np.pi) * ss) * np.exp(-tt**2)
    value[large] = error_function / rr
    first[large] = exponential / rr - error_function / rr**2
    second[large] = (
        -2 * exponential / ss**2 - 2 * exponential / rr**2
        + 2 * error_function / rr**3
    )
    return value, first, second
