"""Shared More--Hebden SVD trust-region kernel, originally in MORPHEUS.

The model is min 0.5 ||r - J p||^2. Callers with r + J p pass -r.
No molecular data or spectroscopy-specific stopping rules enter this layer.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

TRUST_REGION_MIN_RADIUS = 1.0e-10
DAMPING_MIN = 1.0e-14
DAMPING_MAX = 1.0e12

@dataclass(frozen=True)
class TrustRegionLMStep:
    step: np.ndarray
    shift: float
    on_boundary: bool
    solver: str


def limit_step(step, max_norm):
    norm = float(np.linalg.norm(step))
    return step if max_norm <= 0.0 or norm <= max_norm else step * (max_norm / norm)


def svd_trust_region_lm_step(
    jac_weighted: np.ndarray,
    weighted_residual: np.ndarray,
    damping: float,
    trust_radius: float,
) -> TrustRegionLMStep:
    """Solve the rank-revealing LM trust-region subproblem in SVD coordinates.

    The local model is ``min 0.5 ||r - J p||^2`` with ``||p|| <= Delta``.
    When the Gauss-Newton step is inside the region, the minimum-norm
    rank-revealing step is used. Otherwise the Levenberg shift is found by
    robust bisection of the secular equation ``||p(mu)|| = Delta``. This is
    preferable to clipping a full LM step because the returned step is the
    solution of the regularized model for the active trust radius.
    """
    jac = np.asarray(jac_weighted, dtype=float)
    residual = np.asarray(weighted_residual, dtype=float)
    ncols = jac.shape[1]
    if ncols == 0:
        return TrustRegionLMStep(
            np.zeros((0,), dtype=float), max(float(damping), 0.0), False, "empty"
        )
    delta = float(trust_radius)
    if delta <= 0.0 or not np.isfinite(delta):
        step = _rank_revealing_lm_step(jac, residual, damping)
        return TrustRegionLMStep(step, max(float(damping), 0.0), False, "svd_rank_revealing_lm")
    delta = max(delta, TRUST_REGION_MIN_RADIUS)
    try:
        u_matrix, singular, vh = np.linalg.svd(jac, full_matrices=False)
    except np.linalg.LinAlgError:
        step = limit_step(_augmented_qr_lm_step(jac, residual, damping), delta)
        return TrustRegionLMStep(step, max(float(damping), 0.0), True, "augmented_qr_lm_fallback")
    if not singular.size:
        return TrustRegionLMStep(np.zeros(ncols, dtype=float), 0.0, False, "zero_jacobian")
    s0 = max(float(singular[0]), 1.0)
    tol = max(jac.shape) * np.finfo(float).eps * s0 * 100.0
    beta = u_matrix.T @ residual

    def step_for_shift(shift: float) -> np.ndarray:
        mu = max(float(shift), 0.0)
        factors = np.zeros_like(singular)
        keep = singular > tol
        factors[keep] = singular[keep] / (singular[keep] * singular[keep] + mu)
        return vh.T @ (factors * beta)

    unconstrained = step_for_shift(0.0)
    unconstrained_norm = float(np.linalg.norm(unconstrained))
    if np.all(np.isfinite(unconstrained)) and unconstrained_norm <= delta:
        return TrustRegionLMStep(unconstrained, 0.0, False, "svd_more_hebden_trust_region")

    low = 0.0
    high = max(float(damping), s0 * s0 * 1.0e-12, DAMPING_MIN)
    high_step = step_for_shift(high)
    high_norm = float(np.linalg.norm(high_step))
    while (not np.isfinite(high_norm) or high_norm > delta) and high < DAMPING_MAX:
        low = high
        high = min(high * 4.0, DAMPING_MAX)
        high_step = step_for_shift(high)
        high_norm = float(np.linalg.norm(high_step))
    if not np.all(np.isfinite(high_step)):
        step = limit_step(_cauchy_step(jac, residual), delta)
        return TrustRegionLMStep(step, high, True, "cauchy_fallback")
    if high_norm > delta:
        step = limit_step(high_step, delta)
        return TrustRegionLMStep(step, high, True, "svd_more_hebden_trust_region_clipped")

    best_shift = high
    best_step = high_step
    for _ in range(80):
        mid = 0.5 * (low + high)
        trial = step_for_shift(mid)
        trial_norm = float(np.linalg.norm(trial))
        if not np.isfinite(trial_norm):
            low = mid
            continue
        best_shift = mid
        best_step = trial
        if abs(trial_norm - delta) <= max(1.0e-10 * delta, 1.0e-12):
            break
        if trial_norm > delta:
            low = mid
        else:
            high = mid
    if float(np.linalg.norm(best_step)) > delta * (1.0 + 1.0e-8):
        best_step = limit_step(best_step, delta)
    return TrustRegionLMStep(best_step, best_shift, True, "svd_more_hebden_trust_region")


def _rank_revealing_lm_step(
    jac_weighted: np.ndarray,
    weighted_residual: np.ndarray,
    damping: float,
) -> np.ndarray:
    jac = np.asarray(jac_weighted, dtype=float)
    residual = np.asarray(weighted_residual, dtype=float)
    ncols = jac.shape[1]
    mu = max(float(damping), 0.0)
    try:
        u_matrix, singular, vh = np.linalg.svd(jac, full_matrices=False)
    except np.linalg.LinAlgError:
        return _augmented_qr_lm_step(jac, residual, mu)
    if not singular.size:
        return np.zeros(ncols, dtype=float)
    tol = max(jac.shape) * np.finfo(float).eps * max(float(singular[0]), 1.0) * 100.0
    projected = u_matrix.T @ residual
    factors = np.zeros_like(singular)
    keep = singular > tol
    factors[keep] = singular[keep] / (singular[keep] * singular[keep] + mu)
    step = vh.T @ (factors * projected)
    if step.shape[0] != ncols or not np.all(np.isfinite(step)):
        return _augmented_qr_lm_step(jac, residual, mu)
    return step


def _augmented_qr_lm_step(
    jac_weighted: np.ndarray, weighted_residual: np.ndarray, damping: float
) -> np.ndarray:
    jac = np.asarray(jac_weighted, dtype=float)
    residual = np.asarray(weighted_residual, dtype=float)
    ncols = jac.shape[1]
    if ncols == 0:
        return np.zeros((0,), dtype=float)
    mu = max(float(damping), 0.0)
    if mu > 0.0:
        lhs = np.vstack([jac, np.sqrt(mu) * np.eye(ncols)])
        rhs = np.concatenate([residual, np.zeros(ncols, dtype=float)])
    else:
        lhs = jac
        rhs = residual
    try:
        step = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    except np.linalg.LinAlgError:
        return _cauchy_step(jac, residual)
    return np.asarray(step, dtype=float)


def _cauchy_step(jac_weighted: np.ndarray, weighted_residual: np.ndarray) -> np.ndarray:
    gradient = jac_weighted.T @ weighted_residual
    if gradient.size == 0:
        return gradient
    jg = jac_weighted @ gradient
    denom = float(jg @ jg)
    if denom <= 0.0 or not np.isfinite(denom):
        norm = float(np.linalg.norm(gradient))
        return gradient / norm if norm > 0.0 else gradient
    alpha = float((gradient @ gradient) / denom)
    return alpha * gradient



def bounded_svd_trust_region_lm_step(jac_weighted, weighted_residual, damping,
                                     trust_radius, lower, upper):
    """Solve the same LM subproblem with box bounds containing the zero step.

    If the unbounded More--Hebden step is feasible it is returned unchanged.
    Otherwise bounded augmented least squares solves each secular-equation
    evaluation. This imposes the box on the subproblem instead of clipping
    a completed step. No parameter is frozen based on its name or magnitude.
    """
    from scipy.optimize import lsq_linear
    jac = np.asarray(jac_weighted, dtype=float)
    residual = np.asarray(weighted_residual, dtype=float)
    lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    n = jac.shape[1]
    if (lo.shape != (n,) or hi.shape != (n,) or residual.shape != (jac.shape[0],)
            or np.any(np.isnan(lo)) or np.any(np.isnan(hi)) or np.any(lo > 0.)
            or np.any(hi < 0.) or np.any(lo >= hi)):
        raise ValueError('step bounds must contain zero and have positive width')
    original = svd_trust_region_lm_step(jac, residual, damping, trust_radius)
    if np.all(original.step >= lo) and np.all(original.step <= hi):
        return original
    def solve(shift):
        matrix = np.vstack((jac, np.sqrt(shift)*np.eye(n))) if shift > 0 else jac
        rhs = np.r_[residual, np.zeros(n)] if shift > 0 else residual
        fit = lsq_linear(matrix, rhs, bounds=(lo,hi), method='bvls',tol=1.e-12,max_iter=1000)
        if not fit.success:
            raise RuntimeError('bounded LM linear subproblem did not converge')
        return fit.x
    step = solve(0.)
    delta = float(trust_radius)
    if delta <= 0. or not np.isfinite(delta):
        mu = max(float(damping),0.)
        return TrustRegionLMStep(solve(mu),mu,False,'bounded_svd_lm')
    if np.linalg.norm(step) <= delta:
        return TrustRegionLMStep(step,0.,False,'bounded_svd_more_hebden_trust_region')
    low, high = 0., max(float(original.shift),float(damping),DAMPING_MIN)
    for _ in range(160):
        step = solve(high)
        if np.linalg.norm(step) <= delta:break
        low, high = high, high*4.
    else:
        raise RuntimeError('bounded LM secular equation could not be bracketed')
    best = step
    for _ in range(100):
        mid = .5*(low+high); trial = solve(mid); length=float(np.linalg.norm(trial))
        if length <= delta:
            high, best = mid, trial
        else:
            low = mid
        if abs(length-delta) <= max(1.e-10*delta,1.e-12):break
    return TrustRegionLMStep(best,high,True,'bounded_svd_more_hebden_trust_region')
