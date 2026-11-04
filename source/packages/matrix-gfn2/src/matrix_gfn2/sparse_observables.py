"""Lazy observables assembled directly from localized sparse AO matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .localized_solver import SparseSymmetricMatrix

try:
    from . import _gfn2_native
except ImportError:  # pragma: no cover
    _gfn2_native = None


@dataclass(frozen=True)
class LazyAtomicPairMatrix:
    """Symmetric atomic pair property with deferred dense materialization."""

    size: int
    rows: np.ndarray
    columns: np.ndarray
    values: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return (self.size, self.size)

    def to_dense(self) -> np.ndarray:
        result = np.zeros(self.shape, dtype=float)
        result[self.rows, self.columns] = self.values
        result[self.columns, self.rows] = self.values
        return result

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        dense = self.to_dense()
        if dtype is not None:
            dense = dense.astype(dtype, copy=False)
        if copy:
            dense = dense.copy()
        return dense

    def __getitem__(self, key):
        return self.to_dense()[key]


def sparse_mayer_bond_orders(
    density: SparseSymmetricMatrix,
    overlap: SparseSymmetricMatrix,
    orbital_atom: Sequence[int],
) -> LazyAtomicPairMatrix:
    """Mayer bond orders on the AO locality graph, returned lazily.

    Both directional entries of ``P S`` are evaluated explicitly.  This is
    essential because ``P S`` is generally nonsymmetric even though ``P`` and
    ``S`` are symmetric.
    """

    if density.size != overlap.size:
        raise ValueError("density and overlap dimensions differ")
    assignment = np.asarray(tuple(orbital_atom), dtype=np.intp)
    if assignment.shape != (density.size,):
        raise ValueError("orbital assignment does not match AO dimension")
    if _gfn2_native is not None and hasattr(
        _gfn2_native, "sparse_mayer_bond_orders"
    ):
        rows, columns, values = _gfn2_native.sparse_mayer_bond_orders(
            density.size,
            density.rows,
            density.columns,
            density.values,
            overlap.rows,
            overlap.columns,
            overlap.values,
            assignment.astype(np.int64, copy=False),
        )
        return LazyAtomicPairMatrix(
            int(np.max(assignment, initial=-1)) + 1,
            np.asarray(rows, dtype=np.intp),
            np.asarray(columns, dtype=np.intp),
            np.asarray(values, dtype=float),
        )

    def adjacency(matrix: SparseSymmetricMatrix) -> list[dict[int, float]]:
        result: list[dict[int, float]] = [dict() for _ in range(matrix.size)]
        for row, column, value in zip(
            matrix.rows, matrix.columns, matrix.values, strict=True
        ):
            i, j, scalar = int(row), int(column), float(value)
            result[i][j] = scalar
            result[j][i] = scalar
        return result

    p_rows = adjacency(density)
    s_rows = adjacency(overlap)
    atomic: dict[tuple[int, int], float] = {}
    for left, right in zip(overlap.rows, overlap.columns, strict=True):
        left = int(left)
        right = int(right)
        left_atom = int(assignment[left])
        right_atom = int(assignment[right])
        if left_atom == right_atom:
            continue
        common_lr = p_rows[left].keys() & s_rows[right].keys()
        common_rl = p_rows[right].keys() & s_rows[left].keys()
        ps_lr = sum(p_rows[left][index] * s_rows[right][index] for index in common_lr)
        ps_rl = sum(p_rows[right][index] * s_rows[left][index] for index in common_rl)
        atom_left, atom_right = max(left_atom, right_atom), min(left_atom, right_atom)
        key = (atom_left, atom_right)
        atomic[key] = atomic.get(key, 0.0) + ps_lr * ps_rl
    items = sorted((left, right, value) for (left, right), value in atomic.items())
    return LazyAtomicPairMatrix(
        int(np.max(assignment, initial=-1)) + 1,
        np.asarray([item[0] for item in items], dtype=np.intp),
        np.asarray([item[1] for item in items], dtype=np.intp),
        np.asarray([item[2] for item in items], dtype=float),
    )


__all__ = ["LazyAtomicPairMatrix", "sparse_mayer_bond_orders"]
