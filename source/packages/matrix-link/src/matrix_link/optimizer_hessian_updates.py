"""Secant updates and Hessian validity checks for the LINK optimizer."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class OptimizerHessianSettings(Protocol):
    min_hessian_eigenvalue: float
    stationary_point: str


def stored_hessian_is_numerically_usable(
    hessian: np.ndarray,
    settings: OptimizerHessianSettings,
) -> bool:
    """Validate storage safety without imposing a stationary-point index."""

    del settings
    matrix = np.asarray(hessian, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    if not np.all(np.isfinite(matrix)):
        return False
    try:
        np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    except np.linalg.LinAlgError:
        return False
    # Do not impose an absolute spectral ceiling on the covariant stored
    # Hessian. Its scale changes under valid coordinate recombinations; the
    # ephemeral step model is where conditioning and index control belong.
    return True


def bfgs_update(
    hessian: np.ndarray, step: np.ndarray, y: np.ndarray, *, damp: bool
) -> tuple[np.ndarray, str]:
    """Apply the direct-Hessian BFGS update used for minimum searches."""

    s = np.asarray(step, dtype=float).reshape(-1)
    yvec = np.asarray(y, dtype=float).reshape(-1)
    h = 0.5 * (np.asarray(hessian, dtype=float) + np.asarray(hessian, dtype=float).T)
    hs = h @ s
    s_h_s = float(s @ hs)
    s_y = float(s @ yvec)
    if damp and s_h_s > 0.0 and s_y < 0.2 * s_h_s:
        theta = 0.8 * s_h_s / (s_h_s - s_y)
        yvec = theta * yvec + (1.0 - theta) * hs
        s_y = float(s @ yvec)
        status = "bfgs_damped"
    else:
        status = "bfgs"
    if s_y <= 1.0e-12 or s_h_s <= 1.0e-12:
        return h, "bfgs_skipped_curvature"
    updated = h - np.outer(hs, hs) / s_h_s + np.outer(yvec, yvec) / s_y
    return 0.5 * (updated + updated.T), status


def bofill_update(
    hessian: np.ndarray, step: np.ndarray, y: np.ndarray, *, damp: bool | None = None
) -> tuple[np.ndarray, str]:
    """Apply the canonical Murtagh--Sargent--Powell/Bofill TS update."""

    del damp
    h = 0.5 * (np.asarray(hessian, dtype=float) + np.asarray(hessian, dtype=float).T)
    s = np.asarray(step, dtype=float).reshape(-1)
    yvec = np.asarray(y, dtype=float).reshape(-1)
    residual = yvec - h @ s
    step2 = float(s @ s)
    residual2 = float(residual @ residual)
    if step2 <= 1.0e-24 or residual2 <= 1.0e-24:
        return h, "bofill_msp_skipped_null_secant"
    residual_step = float(residual @ s)
    cosine2 = float(np.clip(residual_step**2 / (residual2 * step2), 0.0, 1.0))
    powell_weight = 1.0 - cosine2
    # cosine2 * (r r.T)/(r.T s), evaluated in cancellation-free form so the
    # orthogonal-secant limit is finite.
    weighted_ms = residual_step * np.outer(residual, residual) / (residual2 * step2)
    psb = (np.outer(residual, s) + np.outer(s, residual)) / step2 - residual_step * np.outer(
        s, s
    ) / (step2 * step2)
    updated = h + weighted_ms + powell_weight * psb
    return 0.5 * (updated + updated.T), f"bofill_msp_phi={powell_weight:.3f}"


def bfgs_bofill_update(
    hessian: np.ndarray, step: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, str]:
    """Blend secant-exact BFGS and Bofill updates using secant alignment.

    BFGS is used unchanged while the observed secant curvature is at least
    20 percent of the current model curvature, matching the usual Powell
    safeguard boundary.  Only weaker-curvature secants activate a continuous
    BFGS--Bofill blend.  Because both components obey the same undamped secant
    equation, their convex combination does as well.
    """

    s = np.asarray(step, dtype=float).reshape(-1)
    yvec = np.asarray(y, dtype=float).reshape(-1)
    s2 = float(s @ s)
    sy = float(s @ yvec)
    base = 0.5 * (np.asarray(hessian, dtype=float) + np.asarray(hessian, dtype=float).T)
    model_curvature = float(s @ (base @ s))
    if s2 <= 1.0e-24 or float(yvec @ yvec) <= 1.0e-24:
        return base, "bfgs_bofill_skipped_null_secant"
    if model_curvature > 1.0e-12 and sy >= 0.2 * model_curvature:
        bfgs_weight = 1.0
    elif model_curvature > 1.0e-12 and sy > 0.0:
        bfgs_weight = float(np.clip(sy / (0.2 * model_curvature), 0.0, 1.0))
    else:
        bfgs_weight = 0.0
    bfgs, bfgs_status = bfgs_update(hessian, s, yvec, damp=False)
    bofill, _bofill_status = bofill_update(hessian, s, yvec)
    if "skipped" in bfgs_status:
        bfgs_weight = 0.0
    updated = bfgs_weight * bfgs + (1.0 - bfgs_weight) * bofill
    return (
        0.5 * (updated + updated.T),
        f"bfgs_bofill_bfgs_weight={bfgs_weight:.6f}",
    )


def sella_ts_bfgs_update(
    hessian: np.ndarray, step: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, str]:
    """Apply the one-secant TS-BFGS update used by Sella 2.3.2."""

    s = np.asarray(step, dtype=float).reshape(-1)
    yvec = np.asarray(y, dtype=float).reshape(-1)
    base = 0.5 * (np.asarray(hessian, dtype=float) + np.asarray(hessian, dtype=float).T)
    if float(s @ s) <= 1.0e-24 or float(yvec @ yvec) <= 1.0e-24:
        return base, "sella_ts_bfgs_skipped_null_secant"
    eigenvalues, eigenvectors = np.linalg.eigh(base)
    abs_h_s = eigenvectors @ (np.abs(eigenvalues) * (eigenvectors.T @ s))
    xvec = float(s @ yvec) * yvec + float(s @ abs_h_s) * abs_h_s
    denominator = float(xvec @ s)
    scale = max(float(np.linalg.norm(xvec) * np.linalg.norm(s)), 1.0)
    if abs(denominator) <= 1.0e-14 * scale:
        return base, "sella_ts_bfgs_skipped_singular_weight"
    uvec = xvec / denominator
    residual = yvec - base @ s
    updated = (
        base
        + np.outer(uvec, residual)
        + np.outer(residual, uvec)
        - float(residual @ s) * np.outer(uvec, uvec)
    )
    return 0.5 * (updated + updated.T), "sella_ts_bfgs"


def sr1_update(hessian: np.ndarray, step: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, str]:
    """Apply a symmetric-rank-one direct-Hessian update."""

    s = np.asarray(step, dtype=float).reshape(-1)
    yvec = np.asarray(y, dtype=float).reshape(-1)
    h = 0.5 * (np.asarray(hessian, dtype=float) + np.asarray(hessian, dtype=float).T)
    residual = yvec - h @ s
    denom = float(residual @ s)
    threshold = 1.0e-8 * float(np.linalg.norm(residual) * np.linalg.norm(s))
    if abs(denom) <= threshold:
        return h, "sr1_skipped_curvature"
    updated = h + np.outer(residual, residual) / denom
    return 0.5 * (updated + updated.T), "sr1"


def hessian_is_usable(hessian: np.ndarray, settings: OptimizerHessianSettings) -> bool:
    """Return whether a physical Hessian is usable by the selected search."""

    matrix = 0.5 * (np.asarray(hessian, dtype=float) + np.asarray(hessian, dtype=float).T)
    if not np.all(np.isfinite(matrix)):
        return False
    try:
        eig = np.linalg.eigvalsh(matrix)
    except np.linalg.LinAlgError:
        return False
    if eig.size == 0:
        return True
    negative_count = int(np.count_nonzero(eig < -settings.min_hessian_eigenvalue))
    if settings.stationary_point == "minimum" and negative_count:
        return False
    # A TS Hessian with index 0 or >1 is still a valid physical model. The
    # ephemeral P-RFO model retains only the tracked reaction direction as
    # negative; final classification reports the observed physical index.
    return True


def optimizer_hessian_index(
    hessian: np.ndarray,
    settings: OptimizerHessianSettings,
) -> int:
    """Count negative physical-Hessian eigenvalues above the numerical floor."""

    matrix = 0.5 * (np.asarray(hessian, dtype=float) + np.asarray(hessian, dtype=float).T)
    if matrix.size == 0:
        return 0
    eigenvalues = np.linalg.eigvalsh(matrix)
    return int(np.count_nonzero(eigenvalues < -settings.min_hessian_eigenvalue))
