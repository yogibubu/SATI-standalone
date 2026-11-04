"""Mass-aware rank and row selection for SONIC coordinate spaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .adaptive_selection import SonicSelectionPlan, select_sonic_coordinates
from .models import GICBMatrix


@dataclass(frozen=True)
class MetricRankReport:
    rank: int
    maximum_rank: int
    singular_values: tuple[float, ...]
    cutoff: float
    condition_number: float
    mass_weighted: bool


def metric_aware_rank(
    b_matrix: GICBMatrix | np.ndarray,
    masses_amu: tuple[float, ...] | list[float] | np.ndarray,
    *,
    tolerance: float = 1.0e-8,
) -> MetricRankReport:
    """Compute the rank of ``B M^-1/2`` rather than raw Cartesian B."""

    matrix = _as_matrix(b_matrix)
    masses = _as_masses(masses_amu, matrix.shape[1])
    relative = float(tolerance)
    if not np.isfinite(relative) or relative <= 0.0:
        raise ValueError("metric rank tolerance must be positive and finite")
    weighted = matrix * np.repeat(1.0 / np.sqrt(masses), 3)
    singular = np.asarray(np.linalg.svd(weighted, compute_uv=False), dtype=float)
    maximum = float(singular[0]) if singular.size else 0.0
    cutoff = relative * max(maximum, 1.0)
    rank = int(np.count_nonzero(singular > cutoff))
    minimum = float(singular[rank - 1]) if rank else 0.0
    condition = float(maximum / minimum) if minimum else float("inf")
    return MetricRankReport(
        rank=rank,
        maximum_rank=min(weighted.shape),
        singular_values=tuple(float(value) for value in singular),
        cutoff=cutoff,
        condition_number=condition,
        mass_weighted=True,
    )


def select_metric_aware_sonic_coordinates(
    b_matrix: GICBMatrix | np.ndarray,
    masses_amu: tuple[float, ...] | list[float] | np.ndarray,
    **selection_kwargs: object,
) -> SonicSelectionPlan:
    """Select coordinates using a mass-weighted row-space rank test."""

    matrix = _as_matrix(b_matrix)
    masses = _as_masses(masses_amu, matrix.shape[1])
    weighted = matrix * np.repeat(1.0 / np.sqrt(masses), 3)
    return select_sonic_coordinates(weighted, **selection_kwargs)


def _as_matrix(b_matrix: GICBMatrix | np.ndarray) -> np.ndarray:
    matrix = np.asarray(b_matrix.rows if isinstance(b_matrix, GICBMatrix) else b_matrix, dtype=float)
    if matrix.ndim != 2 or not matrix.size or not np.all(np.isfinite(matrix)):
        raise ValueError("metric reduction requires a finite nonempty B matrix")
    return matrix


def _as_masses(masses_amu: tuple[float, ...] | list[float] | np.ndarray, columns: int) -> np.ndarray:
    masses = np.asarray(tuple(masses_amu), dtype=float)
    if masses.ndim != 1 or masses.size * 3 != columns:
        raise ValueError("one positive mass is required for each B-matrix atom")
    if not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("metric reduction masses must be finite and positive")
    return masses
