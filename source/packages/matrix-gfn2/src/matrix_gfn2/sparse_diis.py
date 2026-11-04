"""Bounded sparse fixed-point DIIS for localized SCF iterations."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .localized_solver import SparseSymmetricMatrix


@dataclass
class SparseDIIS:
    maximum_size: int = 8
    store_residual_float32: bool = False
    compact_indices: bool = False
    focks: list[SparseSymmetricMatrix] = field(default_factory=list)
    residuals: list[SparseSymmetricMatrix] = field(default_factory=list)
    _topology_rows: np.ndarray | None = field(default=None, init=False, repr=False)
    _topology_columns: np.ndarray | None = field(default=None, init=False, repr=False)
    _native_topology: tuple[np.ndarray, np.ndarray, np.ndarray] | None = field(
        default=None, init=False, repr=False
    )
    _gram_workspace: np.ndarray = field(init=False, repr=False)
    _system_workspace: np.ndarray = field(init=False, repr=False)
    _rhs_workspace: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.maximum_size < 2:
            raise ValueError("DIIS history size must be at least two")
        self._gram_workspace = np.empty(
            (self.maximum_size, self.maximum_size), dtype=np.float64
        )
        self._system_workspace = np.empty(
            (self.maximum_size + 1, self.maximum_size + 1), dtype=np.float64
        )
        self._rhs_workspace = np.zeros(self.maximum_size + 1, dtype=np.float64)

    def clear(self) -> None:
        self.focks.clear()
        self.residuals.clear()
        self._topology_rows = None
        self._topology_columns = None
        self._native_topology = None

    def append(
        self, fock: SparseSymmetricMatrix, residual: SparseSymmetricMatrix
    ) -> None:
        if fock.size != residual.size:
            raise ValueError("DIIS Fock and residual dimensions differ")
        self.focks.append(self._share_topology(fock))
        if self.store_residual_float32 and residual.values.dtype != np.float32:
            residual = SparseSymmetricMatrix(
                residual.size,
                residual.rows,
                residual.columns,
                residual.values.astype(np.float32),
            )
        self.residuals.append(self._share_topology(residual))
        if len(self.focks) <= self.maximum_size:
            return
        # Keep the two newest points and the best older fixed-point residuals.
        worst = max(
            range(len(self.residuals) - 2),
            key=lambda index: self.residuals[index].frobenius_inner(
                self.residuals[index]
            ),
        )
        del self.focks[worst]
        del self.residuals[worst]

    def _share_topology(
        self,
        matrix: SparseSymmetricMatrix,
    ) -> SparseSymmetricMatrix:
        """Intern and optionally compact a common DIIS sparse pattern."""

        if self._topology_rows is None:
            index_dtype = (
                np.int32
                if self.compact_indices and matrix.size <= np.iinfo(np.int32).max
                else np.intp
            )
            self._topology_rows = np.asarray(matrix.rows, dtype=index_dtype)
            self._topology_columns = np.asarray(matrix.columns, dtype=index_dtype)
            # Build native row adjacency once and retain it with the interned
            # pattern instead of reconstructing it for every history vector.
            self._native_topology = matrix.native_topology()
            return SparseSymmetricMatrix(
                matrix.size,
                self._topology_rows,
                self._topology_columns,
                matrix.values,
                self._native_topology,
            )
        if (
            matrix.size > 0
            and np.array_equal(matrix.rows, self._topology_rows)
            and np.array_equal(matrix.columns, self._topology_columns)
        ):
            return SparseSymmetricMatrix(
                matrix.size,
                self._topology_rows,
                self._topology_columns,
                matrix.values,
                self._native_topology,
            )
        return matrix

    def extrapolate(self, *, threshold: float = 0.0) -> SparseSymmetricMatrix:
        if not self.focks:
            raise ValueError("cannot extrapolate an empty DIIS history")
        if len(self.focks) < 2:
            return self.focks[-1]
        count = len(self.focks)
        gram = self._gram_workspace[:count, :count]
        for left_index, left in enumerate(self.residuals):
            for right_index in range(left_index + 1):
                value = left.frobenius_inner(self.residuals[right_index])
                gram[left_index, right_index] = value
                gram[right_index, left_index] = value
        scale = max(float(np.max(np.abs(gram))), 1.0)
        gram.flat[:: count + 1] += scale * 1.0e-12
        system = self._system_workspace[: count + 1, : count + 1]
        system[:count, :count] = gram
        system[count, :count] = -1.0
        system[:count, count] = -1.0
        system[count, count] = 0.0
        rhs = self._rhs_workspace[: count + 1]
        rhs.fill(0.0)
        rhs[count] = -1.0
        try:
            coefficients = np.linalg.solve(system, rhs)[:count]
        except np.linalg.LinAlgError:
            return self.focks[-1]
        if (
            not np.all(np.isfinite(coefficients))
            or np.max(np.abs(coefficients)) > 8.0
            or np.sum(np.abs(coefficients)) > 16.0
        ):
            return self.focks[-1]
        result = SparseSymmetricMatrix(
            self.focks[0].size,
            np.empty(0, dtype=np.intp),
            np.empty(0, dtype=np.intp),
            np.empty(0, dtype=float),
        )
        for coefficient, fock in zip(coefficients, self.focks, strict=True):
            result, _ = result.scaled_add(
                fock, other_scale=float(coefficient), threshold=threshold
            )
        return result


__all__ = ["SparseDIIS"]
