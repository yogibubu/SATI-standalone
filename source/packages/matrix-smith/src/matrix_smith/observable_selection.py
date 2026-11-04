"""Observable-driven coordinate sensitivity for SONIC selection."""

from __future__ import annotations

import numpy as np

from .adaptive_selection import SonicSelectionPlan, select_sonic_coordinates
from .models import GICBMatrix


def select_for_observable(
    b_matrix: GICBMatrix | np.ndarray,
    observable_gradient_q: tuple[float, ...] | list[float] | np.ndarray,
    *,
    identifiers: tuple[str, ...] | list[str] | None = None,
    protected: tuple[str, ...] | list[str] = (),
    max_count: int | None = None,
    role: str = "OBSERVABLE",
    rank_tolerance: float = 1.0e-8,
) -> SonicSelectionPlan:
    """Select coordinates by the declared gradient of an observable in SONIC q.

    The gradient must already be expressed in the frozen SONIC coordinate
    basis.  This function does not infer an observable from Cartesian data and
    therefore cannot silently mix coordinate conventions.
    """

    gradient = np.asarray(observable_gradient_q, dtype=float)
    labels = tuple(identifiers or (b_matrix.coordinate_labels if isinstance(b_matrix, GICBMatrix) else ()))
    if gradient.ndim != 1 or not np.all(np.isfinite(gradient)):
        raise ValueError("observable SONIC gradient must be a finite vector")
    if len(labels) != gradient.size:
        raise ValueError("observable gradient length must match SONIC coordinate labels")
    sensitivities = {label: abs(float(value)) for label, value in zip(labels, gradient)}
    return select_sonic_coordinates(
        b_matrix,
        identifiers=labels,
        sensitivities=sensitivities,
        protected=protected,
        max_count=max_count,
        role=role,
        rank_tolerance=rank_tolerance,
    )
