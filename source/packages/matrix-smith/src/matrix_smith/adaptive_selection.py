"""Deterministic SONIC coordinate selection for local-field workflows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import GICBMatrix


ADAPTIVE_SELECTION_SCHEMA = "matrix.smith.adaptive_coordinate_selection.v1"


@dataclass(frozen=True)
class SonicSelectionRecord:
    identifier: str
    index: int
    score: float
    sensitivity: float
    row_norm: float
    selected: bool
    protected: bool
    reason: str


@dataclass(frozen=True)
class SonicSelectionPlan:
    schema: str
    role: str
    policy: str
    selected: tuple[str, ...]
    records: tuple[SonicSelectionRecord, ...]
    achieved_rank: int
    target_count: int | None
    rank_tolerance: float


def select_sonic_coordinates(
    b_matrix: GICBMatrix | np.ndarray,
    *,
    identifiers: tuple[str, ...] | list[str] | None = None,
    sensitivities: dict[str, float] | None = None,
    protected: tuple[str, ...] | list[str] = (),
    max_count: int | None = None,
    role: str = "UNSPECIFIED",
    rank_tolerance: float = 1.0e-8,
) -> SonicSelectionPlan:
    """Select rows by observable sensitivity while preserving row-space rank.

    The method is deliberately conservative: protected coordinates are always
    considered first, but a dependent row is still reported rather than
    silently duplicated.  All scores are derived from declared inputs and the
    B matrix; no training or hidden chemical rule is involved.
    """

    matrix, labels = _matrix_and_labels(b_matrix, identifiers)
    tolerance = float(rank_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("adaptive selection rank_tolerance must be positive and finite")
    if max_count is not None and int(max_count) < 1:
        raise ValueError("adaptive selection max_count must be positive")
    sensitivity_map = {str(key): float(value) for key, value in (sensitivities or {}).items()}
    if any(not np.isfinite(value) or value < 0.0 for value in sensitivity_map.values()):
        raise ValueError("coordinate sensitivities must be finite and non-negative")
    protected_set = {str(value) for value in protected}
    if not protected_set.issubset(set(labels)):
        missing = sorted(protected_set.difference(labels))
        raise ValueError(f"protected SONIC coordinates are not present: {missing}")

    norms = np.linalg.norm(matrix, axis=1)
    candidates = []
    for index, identifier in enumerate(labels):
        sensitivity = sensitivity_map.get(identifier, 1.0)
        score = sensitivity * float(norms[index])
        candidates.append((identifier not in protected_set, -score, identifier, index, sensitivity))
    candidates.sort()

    selected_indices: list[int] = []
    selected_labels: list[str] = []
    records: list[SonicSelectionRecord] = []
    for _priority, _negative_score, identifier, index, sensitivity in candidates:
        protected_flag = identifier in protected_set
        if max_count is not None and len(selected_indices) >= int(max_count) and not protected_flag:
            records.append(
                SonicSelectionRecord(
                    identifier, index, float(sensitivity * norms[index]), sensitivity,
                    float(norms[index]), False, protected_flag, "MAX_COUNT_REACHED",
                )
            )
            continue
        before = _row_rank(matrix[selected_indices], tolerance) if selected_indices else 0
        trial = selected_indices + [index]
        after = _row_rank(matrix[trial], tolerance)
        if after > before:
            selected_indices.append(index)
            selected_labels.append(identifier)
            reason = "PROTECTED_RANK_GAIN" if protected_flag else "SENSITIVITY_RANK_GAIN"
            selected = True
        else:
            reason = "DEPENDENT_ROW"
            selected = False
        records.append(
            SonicSelectionRecord(
                identifier, index, float(sensitivity * norms[index]), sensitivity,
                float(norms[index]), selected, protected_flag, reason,
            )
        )
    return SonicSelectionPlan(
        schema=ADAPTIVE_SELECTION_SCHEMA,
        role=str(role).strip().upper() or "UNSPECIFIED",
        policy="PROTECTED_THEN_SENSITIVITY_GREEDY_ROW_RANK",
        selected=tuple(selected_labels),
        records=tuple(records),
        achieved_rank=_row_rank(matrix[selected_indices], tolerance) if selected_indices else 0,
        target_count=None if max_count is None else int(max_count),
        rank_tolerance=tolerance,
    )


def _matrix_and_labels(
    b_matrix: GICBMatrix | np.ndarray,
    identifiers: tuple[str, ...] | list[str] | None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if isinstance(b_matrix, GICBMatrix):
        matrix = np.asarray(b_matrix.rows, dtype=float)
        labels = tuple(identifiers or b_matrix.coordinate_labels)
    else:
        matrix = np.asarray(b_matrix, dtype=float)
        labels = tuple(identifiers or (f"Q{index + 1:04d}" for index in range(matrix.shape[0])))
    if matrix.ndim != 2 or not matrix.size or not np.all(np.isfinite(matrix)):
        raise ValueError("SONIC adaptive selection requires a finite nonempty B matrix")
    if len(labels) != matrix.shape[0] or len(set(labels)) != len(labels):
        raise ValueError("SONIC selection identifiers must be unique and match B rows")
    return matrix, labels


def _row_rank(matrix: np.ndarray, tolerance: float) -> int:
    if matrix.size == 0:
        return 0
    singular = np.linalg.svd(matrix, compute_uv=False)
    return int(np.count_nonzero(singular > tolerance * max(float(singular[0]), 1.0)))
