"""Deterministic rank-revealing selection kernels."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RankRevealingRowSelection:
    """Result of a rank-revealing row selection.

    ``indices`` records the deterministic pivot order. ``orthonormal_basis``
    spans the selected row space and is produced by twice-reorthogonalized
    modified Gram--Schmidt.
    """

    indices: tuple[int, ...]
    orthonormal_basis: np.ndarray
    tolerance: float

    @property
    def rank(self) -> int:
        return len(self.indices)


def select_rank_revealing_rows(
    matrix: np.ndarray,
    *,
    target_rank: int | None = None,
    tolerance: float = 1.0e-10,
    priorities: Sequence[int] | None = None,
    tie_tolerance: float = 1.0e-12,
) -> RankRevealingRowSelection:
    """Select a stable independent row basis in deterministic pivot order.

    The kernel performs maximum-residual pivoted modified Gram--Schmidt with
    two reorthogonalization passes. Rows in lower-valued priority classes are
    exhausted before later classes; priorities affect selection order only,
    never the linear-dependence criterion. Equal pivots are resolved by input
    row order.

    Parameters
    ----------
    matrix
        Candidate rows. The dependence threshold is applied to their
        orthogonal residuals, so callers that require scale-independent
        selection should normalize rows before calling this function.
    target_rank
        Maximum number of rows to select. By default the full available rank
        is revealed.
    tolerance
        Absolute residual-norm threshold for linear independence.
    priorities
        Optional integer priority class for every row. Lower values are
        processed first.
    tie_tolerance
        Absolute score tolerance used to preserve input order for effectively
        equal maximum-residual pivots.
    """

    rows = np.asarray(matrix, dtype=float)
    if rows.ndim != 2:
        raise ValueError("rank-revealing input must be a two-dimensional matrix")
    if not np.all(np.isfinite(rows)):
        raise ValueError("rank-revealing input contains non-finite values")
    threshold = float(tolerance)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("rank-revealing tolerance must be finite and non-negative")
    tie = float(tie_tolerance)
    if not np.isfinite(tie) or tie < 0.0:
        raise ValueError("rank-revealing tie tolerance must be finite and non-negative")

    row_count, column_count = rows.shape
    maximum_rank = min(row_count, column_count)
    if target_rank is None:
        requested_rank = maximum_rank
    else:
        requested_rank = int(target_rank)
        if requested_rank < 0:
            raise ValueError("rank-revealing target rank must be non-negative")
        requested_rank = min(requested_rank, maximum_rank)

    if priorities is None:
        priority_array = np.zeros(row_count, dtype=np.int64)
    else:
        if len(priorities) != row_count:
            raise ValueError("rank-revealing priorities must match the row count")
        priority_array = np.asarray(priorities, dtype=np.int64)

    if requested_rank == 0 or row_count == 0 or column_count == 0:
        return RankRevealingRowSelection(
            indices=(),
            orthonormal_basis=np.zeros((0, column_count), dtype=float),
            tolerance=threshold,
        )

    residuals = np.array(rows, dtype=float, copy=True)
    residual_norms = np.linalg.norm(residuals, axis=1)
    available = np.ones(row_count, dtype=bool)
    selected: list[int] = []
    basis: list[np.ndarray] = []

    for priority in np.unique(priority_array):
        while len(selected) < requested_rank:
            eligible = np.flatnonzero(available & (priority_array == priority))
            if not eligible.size:
                break
            scores = residual_norms[eligible]
            maximum_score = float(np.max(scores, initial=0.0))
            if maximum_score <= threshold:
                available[eligible] = False
                break
            tied = eligible[scores >= maximum_score - tie]
            pivot = int(tied[0])

            residual = _twice_orthogonalized_residual(rows[pivot], basis)
            residual_norm = float(np.linalg.norm(residual))
            if residual_norm <= threshold:
                available[pivot] = False
                residual_norms[pivot] = 0.0
                continue

            new_basis_row = residual / residual_norm
            basis.append(new_basis_row)
            selected.append(pivot)
            available[pivot] = False
            residual_norms[pivot] = 0.0

            active = np.flatnonzero(available)
            if active.size:
                active_residuals = residuals[active]
                for _pass in range(2):
                    projections = active_residuals @ new_basis_row
                    active_residuals -= projections[:, None] * new_basis_row[None, :]
                residuals[active] = active_residuals
                residual_norms[active] = np.linalg.norm(active_residuals, axis=1)

        if len(selected) == requested_rank:
            break

    return RankRevealingRowSelection(
        indices=tuple(selected),
        orthonormal_basis=(
            np.vstack(basis) if basis else np.zeros((0, column_count), dtype=float)
        ),
        tolerance=threshold,
    )


def _twice_orthogonalized_residual(
    row: np.ndarray,
    basis: list[np.ndarray],
) -> np.ndarray:
    residual = np.array(row, dtype=float, copy=True)
    if not basis:
        return residual
    orthonormal = np.vstack(basis)
    for _pass in range(2):
        residual -= (orthonormal @ residual) @ orthonormal
    return residual
