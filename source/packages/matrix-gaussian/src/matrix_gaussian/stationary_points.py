"""Backend-neutral stationary-point classification for harmonic Hessians."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class StationaryPointClassification:
    """Auditable minimum/transition-state classification."""

    kind: str
    imaginary_frequency_count: int
    lowest_frequency_cm1: float
    rescue_required: bool
    rescue_displacements: tuple[int, ...]


def classify_stationary_point(
    frequencies_cm1: Sequence[float], *, imaginary_cutoff_cm1: float = -10.0
) -> StationaryPointClassification:
    """Classify a Hessian and specify the deterministic rescue action.

    Frequencies above the cutoff are treated as numerically non-imaginary.
    One significant imaginary mode is a transition-state candidate and gets
    both-sign displacement rescue; two or more significant imaginary modes are
    rejected as a higher-order saddle.
    """

    values = np.asarray(tuple(frequencies_cm1), dtype=float).reshape(-1)
    cutoff = float(imaginary_cutoff_cm1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("stationary-point classification needs finite frequencies")
    if not np.isfinite(cutoff) or cutoff >= 0.0:
        raise ValueError("imaginary cutoff must be finite and negative")
    imaginary = np.flatnonzero(values < cutoff)
    count = int(imaginary.size)
    if count == 0:
        kind = "minimum"
    elif count == 1:
        kind = "transition_state_candidate"
    else:
        kind = "higher_order_saddle"
    return StationaryPointClassification(
        kind=kind,
        imaginary_frequency_count=count,
        lowest_frequency_cm1=float(values.min()),
        rescue_required=count == 1,
        rescue_displacements=(-1, 1) if count == 1 else (),
    )


__all__ = ["StationaryPointClassification", "classify_stationary_point"]
