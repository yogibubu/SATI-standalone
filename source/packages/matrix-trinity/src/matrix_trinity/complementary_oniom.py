"""Frequency-level ONIOM, DOI 10.1021/acs.jctc.6c00816, Eqs. 28--29."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ComplementaryFrequencies:
    mode_ids: tuple[str, ...]
    harmonic_cm: np.ndarray
    normalized_weights: np.ndarray
    fundamentals_cm: np.ndarray | None


def complementary_oniom_frequencies(frequencies_cm, model_character, mode_ids, *,
                                    low_harmonic_cm=None, low_fundamentals_cm=None):
    """Blend already matched parent modes with their model-system characters.

    Arrays are partition x mode. Rows must have identical explicit mode IDs,
    established from eigenvectors/character by the caller, not frequency order.
    Characters must share one convention (fractions or percentages). This
    combines scalar frequencies only, not Hessians, intensities, mode maps or
    resonances. Optional LL shifts are assumed matched to the same mode order.
    """
    f = np.asarray(frequencies_cm, dtype=float)
    p = np.asarray(model_character, dtype=float)
    ids = tuple(tuple(row) for row in mode_ids)
    if f.ndim != 2 or min(f.shape) < 1 or p.shape != f.shape:
        raise ValueError('frequencies/characters need matching partition x mode shapes')
    if not np.all(np.isfinite(f)) or np.any(f <= 0):
        raise ValueError('frequency blending requires finite positive harmonic frequencies')
    if not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError('model characters must be finite and nonnegative')
    if (len(ids) != f.shape[0] or any(len(row) != f.shape[1] for row in ids)
            or any(row != ids[0] for row in ids) or len(set(ids[0])) != f.shape[1]
            or any(not s for s in ids[0])):
        raise ValueError('provide identical unique physically matched mode IDs for every partition')
    sums = p.sum(axis=0)
    if np.any(sums <= 0):
        raise ValueError('every mode needs nonzero total model character')
    weights = p / sums
    harmonic = np.sum(weights * f, axis=0)
    if (low_harmonic_cm is None) != (low_fundamentals_cm is None):
        raise ValueError('supply both LL harmonics and fundamentals to transfer shifts')
    fundamentals = None
    if low_harmonic_cm is not None:
        low_h = np.asarray(low_harmonic_cm, dtype=float)
        low_a = np.asarray(low_fundamentals_cm, dtype=float)
        if (low_h.shape != harmonic.shape or low_a.shape != harmonic.shape
                or not np.all(np.isfinite(low_h)) or not np.all(np.isfinite(low_a))
                or np.any(low_h <= 0) or np.any(low_a <= 0)):
            raise ValueError('LL frequencies must be positive finite and in matched mode order')
        fundamentals = harmonic + low_a - low_h
        if np.any(fundamentals <= 0):
            raise ValueError('transferred shifts yield nonpositive fundamentals')
    return ComplementaryFrequencies(ids[0], harmonic, weights, fundamentals)
