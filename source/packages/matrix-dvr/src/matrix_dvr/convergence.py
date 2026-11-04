"""Grid/domain convergence diagnostics for DVR calculations."""

from __future__ import annotations

import numpy as np


def assess_convergence(level_sets: list[np.ndarray] | tuple[np.ndarray, ...], *, tolerance_cm: float = 0.01) -> dict[str, object]:
    if len(level_sets) < 2:
        raise ValueError("at least two level sets are required")
    arrays = [np.asarray(item, dtype=float) for item in level_sets]
    count = min(array.size for array in arrays)
    differences = [float(np.max(np.abs(arrays[i][:count] - arrays[i - 1][:count]), initial=0.0)) for i in range(1, len(arrays))]
    return {"converged": differences[-1] <= tolerance_cm, "tolerance_cm": float(tolerance_cm), "max_differences_cm": differences, "levels_compared": count, "trend_decreasing": all(right <= left for left, right in zip(differences, differences[1:], strict=False))}


__all__ = ["assess_convergence"]
