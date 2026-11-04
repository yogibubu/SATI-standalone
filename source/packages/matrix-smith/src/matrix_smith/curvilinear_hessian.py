"""Explicit curvilinear Hessian transformation for SONIC coordinates."""

from __future__ import annotations

import numpy as np


def curvilinear_hessian(
    cartesian_hessian: np.ndarray,
    back_transform: np.ndarray,
    cartesian_gradient: np.ndarray,
    *,
    second_back_transform: np.ndarray | None = None,
    stationary_tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Transform a Cartesian Hessian with the full curvilinear correction.

    ``back_transform`` is ``A = dx/dq``.  If the Cartesian gradient is not
    stationary, ``second_back_transform`` with shape ``(n_cart, n_q, n_q)`` is
    mandatory and supplies ``d²x/dq²``.  A stationary point may omit it because
    the gradient-times-curvature term vanishes exactly in the formula.
    """

    hessian = np.asarray(cartesian_hessian, dtype=float)
    transform = np.asarray(back_transform, dtype=float)
    gradient = np.asarray(cartesian_gradient, dtype=float).reshape(-1)
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError("Cartesian Hessian must be square")
    if transform.ndim != 2 or transform.shape[0] != hessian.shape[0]:
        raise ValueError("back_transform shape is incompatible with Cartesian Hessian")
    if gradient.shape != (hessian.shape[0],):
        raise ValueError("Cartesian gradient shape is incompatible with Hessian")
    if not all(np.all(np.isfinite(array)) for array in (hessian, transform, gradient)):
        raise ValueError("curvilinear Hessian inputs must be finite")
    tolerance = float(stationary_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("stationary tolerance must be finite and non-negative")
    if second_back_transform is None:
        if float(np.linalg.norm(gradient)) > tolerance:
            raise ValueError("nonstationary Hessian requires second_back_transform")
        curvature = np.zeros((transform.shape[1], transform.shape[1]), dtype=float)
    else:
        second = np.asarray(second_back_transform, dtype=float)
        expected = (hessian.shape[0], transform.shape[1], transform.shape[1])
        if second.shape != expected:
            raise ValueError(f"second_back_transform must have shape {expected}")
        if not np.all(np.isfinite(second)):
            raise ValueError("second_back_transform must be finite")
        curvature = np.einsum("a,aqr->qr", gradient, second)
    transformed = transform.T @ hessian @ transform + curvature
    return 0.5 * (transformed + transformed.T)
