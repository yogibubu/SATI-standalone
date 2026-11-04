from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SemiDiagonalCubicField:
    """TRINITY normal-coordinate cubic sector Phi(j,k,k)."""

    cubic_fjkk_qmw: np.ndarray
    steps_sqrt_amu_bohr: np.ndarray
    source: str = "central-second-differences-of-analytic-gradients"


def semidiagonal_cubic_from_modal_gradient_stencil(
    gradient_reference_cartesian,
    gradients_plus_cartesian,
    gradients_minus_cartesian,
    modes_mw,
    masses_amu,
    steps_sqrt_amu_bohr,
) -> SemiDiagonalCubicField:
    """Recover Phi(j,k,k) from the LINK-evaluated 0,+/-h_k gradients.

    The reference gradient is retained explicitly, so the stencil remains
    correct when a geometry optimized at another level is not stationary for
    the electronic-structure level supplying these gradients.  The output is
    the semidiagonal cubic sector needed by both VPT2 improvements and
    vibration-rotation corrections.
    """

    modes = np.asarray(modes_mw, dtype=float)
    masses = np.asarray(masses_amu, dtype=float).reshape(-1)
    steps = np.asarray(steps_sqrt_amu_bohr, dtype=float).reshape(-1)
    nvib = modes.shape[0]
    dimension = 3 * masses.size
    if modes.shape != (nvib, masses.size, 3) or steps.shape != (nvib,):
        raise ValueError("normal modes and finite-difference steps disagree")
    if np.any(steps <= 0.0) or not np.all(np.isfinite(steps)):
        raise ValueError("normal-mode finite-difference steps must be positive and finite")
    g0 = np.asarray(gradient_reference_cartesian, dtype=float).reshape(dimension)
    gp = np.asarray(gradients_plus_cartesian, dtype=float)
    gm = np.asarray(gradients_minus_cartesian, dtype=float)
    if gp.shape != (nvib, dimension) or gm.shape != (nvib, dimension):
        raise ValueError("one Cartesian +/- gradient is required per normal mode")
    if not all(np.all(np.isfinite(item)) for item in (g0, gp, gm)):
        raise ValueError("gradient stencil contains non-finite values")
    cartesian_per_q = modes.reshape((nvib, dimension)).T / np.sqrt(
        np.repeat(masses, 3)
    )[:, None]
    g0_q = cartesian_per_q.T @ g0
    gp_q = gp @ cartesian_per_q
    gm_q = gm @ cartesian_per_q
    cubic = ((gp_q + gm_q - 2.0 * g0_q[None, :]) / steps[:, None] ** 2).T
    return SemiDiagonalCubicField(cubic, steps)


__all__ = ["SemiDiagonalCubicField", "semidiagonal_cubic_from_modal_gradient_stencil"]
