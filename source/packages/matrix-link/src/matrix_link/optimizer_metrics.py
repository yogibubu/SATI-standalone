"""Small numerical metrics shared by LINK optimizer components."""

from __future__ import annotations

import numpy as np


def rms(values: np.ndarray) -> float:
    """Return the root-mean-square of a flattened numerical array."""

    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(array * array)))
