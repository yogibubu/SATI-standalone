"""Controlled GDIIS, Farkas & Schlegel, PCCP 4 (2002) 11–15.

Equations (3), (6), (8), (9) and criteria (a)–(d), DOI 10.1039/B108658H.
The caller supplies ONE current effective inverse Hessian, including its RFO
shift, for the reference, all historical residuals, and the final relaxation.
The optional line search mentioned in the paper is not used: rejection falls
back to the reference step. No energy/force monotonicity rule is added here.
"""

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


COSINE_CUTOFFS = (0.97, 0.84, 0.71, 0.67, 0.62, 0.56, 0.49, 0.41)


@dataclass(frozen=True)
class ControlledStep:
    step: np.ndarray | None
    status: str
    used_points: int
    discarded_points: int


def coefficients(errors: np.ndarray) -> tuple[np.ndarray | None, str]:
    """Solve Eq. (9); scale ALL residuals by the smallest residual norm."""
    errors = np.asarray(errors, dtype=float)
    norms = np.linalg.norm(errors, axis=1)
    if not np.all(np.isfinite(norms)) or np.min(norms) <= 0:
        return None, "zero_or_nonfinite_residual"
    scaled = errors / np.min(norms)
    overlap = scaled @ scaled.T
    try:
        # u = c / |r|^2, rather than the augmented Pulay system.
        u = np.linalg.solve(overlap, np.ones(len(errors)))
    except np.linalg.LinAlgError:
        return None, "singular_overlap"
    if not np.all(np.isfinite(u)) or np.max(np.abs(u)) > 1.e8:
        return None, "criterion_d"
    total = float(np.sum(u))
    if total <= 0 or not np.isfinite(total):
        return None, "invalid_residual_norm"
    c = u / total
    if max(float(c[c > 0].sum()), float(-c[c < 0].sum())) > 15.:
        return None, "criterion_c"
    return c, "ok"


def controlled_step(
    history: Sequence[tuple[np.ndarray, float, np.ndarray]],
    current_q: np.ndarray,
    current_gradient: np.ndarray,
    inverse_action: Callable[[np.ndarray], np.ndarray],
    *,
    max_history: int,
    vector_map: Callable[[np.ndarray], np.ndarray] = np.asarray,
) -> ControlledStep:
    """Grow newest-first history, retaining the last admissible candidate.

    vector_map expresses displacements in the single current Euclidean chart
    used for both the residual overlaps and direction/length comparisons.
    Only criterion (a) with an angle >90 degrees permanently removes points.
    """
    records = list(history[-max_history:])
    dropped = len(history) - len(records)
    reference = -np.asarray(inverse_action(np.asarray(current_gradient)), dtype=float)
    reference_vector = vector_map(reference)
    reference_norm = float(np.linalg.norm(reference_vector))
    if not np.isfinite(reference_norm) or reference_norm == 0:
        return ControlledStep(None, "zero_or_nonfinite_reference", 1, dropped)
    errors, points, mapped_errors = [], [], []
    best = None
    used = 1
    status = "reference"
    for point, _energy, gradient in reversed(records):
        error = -np.asarray(inverse_action(np.asarray(gradient)), dtype=float)
        errors.append(error)
        points.append(np.asarray(point, dtype=float))
        mapped_errors.append(vector_map(error))
        n = len(errors)
        if n < 2:
            continue
        weights, reason = coefficients(np.asarray(mapped_errors))
        if weights is None:
            status = reason
            break
        candidate = weights @ (np.asarray(points) + np.asarray(errors)) - current_q
        vector = vector_map(candidate)
        length = float(np.linalg.norm(vector))
        if not np.isfinite(length) or length == 0:
            status = "zero_or_nonfinite_candidate"
            break
        cosine = float(vector @ reference_vector) / (length * reference_norm)
        cutoff = COSINE_CUTOFFS[n - 2] if n < 10 else 0.
        if cosine < cutoff:
            status = f"criterion_a:n={n}:cosine={cosine:.8g}:cutoff={cutoff}"
            if cosine < 0:
                dropped = len(history) - (n - 1)
            break
        if length > 10. * reference_norm:
            status = "criterion_b"
            break
        best, used = candidate, n
        status = "accepted"
    return ControlledStep(best, f"farkas_schlegel:n={used}:{status}", used, dropped)
