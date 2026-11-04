from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import numpy as np


SPARSE_ZERO_TOLERANCE = 1.0e-14


@dataclass(frozen=True)
class SparseBRow:
    """Sparse Wilson-B row with deterministic sorted entries.

    The row is a runtime object, not part of coordinate construction.  Frozen
    coordinate contracts define which rows must be evaluated; this class stores
    the evaluated derivative row efficiently for iterative optimization and
    refinement steps.
    """

    size: int
    entries: tuple[tuple[int, float], ...] = ()

    def __post_init__(self) -> None:
        size = int(self.size)
        if size < 0:
            raise ValueError("SparseBRow size must be non-negative")
        merged: dict[int, float] = {}
        for raw_index, raw_value in self.entries:
            index = int(raw_index)
            if index < 0 or index >= size:
                raise ValueError(f"SparseBRow index {index} outside row size {size}")
            value = float(raw_value)
            if not np.isfinite(value):
                raise ValueError("SparseBRow entries must be finite")
            if abs(value) <= SPARSE_ZERO_TOLERANCE:
                continue
            merged[index] = merged.get(index, 0.0) + value
        canonical = tuple(
            (index, value)
            for index, value in sorted(merged.items())
            if abs(value) > SPARSE_ZERO_TOLERANCE
        )
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "entries", canonical)

    @classmethod
    def zeros(cls, size: int) -> "SparseBRow":
        return cls(size=int(size), entries=())

    @classmethod
    def from_dense(
        cls,
        row: Iterable[float] | np.ndarray,
        *,
        zero_tolerance: float = SPARSE_ZERO_TOLERANCE,
    ) -> "SparseBRow":
        dense = np.asarray(tuple(row) if not isinstance(row, np.ndarray) else row, dtype=float)
        if dense.ndim != 1:
            raise ValueError("SparseBRow.from_dense expects a one-dimensional row")
        entries = tuple(
            (int(index), float(value))
            for index, value in enumerate(dense)
            if abs(float(value)) > float(zero_tolerance)
        )
        return cls(size=int(dense.size), entries=entries)

    @classmethod
    def from_atom_gradients(
        cls,
        natoms: int,
        gradients: Iterable[tuple[int, Iterable[float]]],
        *,
        one_based: bool = True,
    ) -> "SparseBRow":
        """Create a row from atom-indexed 3-vector gradients."""

        size = 3 * int(natoms)
        entries: list[tuple[int, float]] = []
        for atom_index, gradient in gradients:
            atom = int(atom_index) - 1 if one_based else int(atom_index)
            if atom < 0 or atom >= int(natoms):
                raise ValueError(f"atom index {atom_index} outside 1..{natoms}")
            values = tuple(float(value) for value in gradient)
            if len(values) != 3:
                raise ValueError("atom gradient must contain three Cartesian components")
            start = 3 * atom
            entries.extend((start + offset, value) for offset, value in enumerate(values))
        return cls(size=size, entries=tuple(entries))

    @property
    def nnz(self) -> int:
        return len(self.entries)

    def to_dense(self) -> np.ndarray:
        dense = np.zeros(self.size, dtype=float)
        for index, value in self.entries:
            dense[index] = value
        return dense

    def scaled(self, coefficient: float) -> "SparseBRow":
        factor = float(coefficient)
        if abs(factor) <= SPARSE_ZERO_TOLERANCE:
            return SparseBRow.zeros(self.size)
        return SparseBRow(self.size, tuple((index, factor * value) for index, value in self.entries))

    def add(self, other: "SparseBRow", *, coefficient: float = 1.0) -> "SparseBRow":
        if self.size != other.size:
            raise ValueError("cannot add SparseBRow objects with different sizes")
        return SparseBRow.combine((1.0, self), (float(coefficient), other))

    def dot_dense(self, vector: Iterable[float] | np.ndarray) -> float:
        dense = np.asarray(vector, dtype=float)
        if dense.shape != (self.size,):
            raise ValueError(f"dense vector shape {dense.shape} does not match row size {self.size}")
        return float(sum(value * float(dense[index]) for index, value in self.entries))

    def norm(self) -> float:
        return float(np.sqrt(sum(value * value for _index, value in self.entries)))

    @classmethod
    def combine(cls, *weighted_rows: tuple[float, "SparseBRow"]) -> "SparseBRow":
        rows = tuple((float(coefficient), row) for coefficient, row in weighted_rows)
        if not rows:
            return cls.zeros(0)
        size = rows[0][1].size
        accumulator: dict[int, float] = {}
        for coefficient, row in rows:
            if row.size != size:
                raise ValueError("cannot combine SparseBRow objects with different sizes")
            if abs(coefficient) <= SPARSE_ZERO_TOLERANCE:
                continue
            for index, value in row.entries:
                accumulator[index] = accumulator.get(index, 0.0) + coefficient * value
        return cls(size=size, entries=tuple(accumulator.items()))


class CSRBackedRows(Sequence[SparseBRow]):
    """Read-only compatibility view over a native CSR Wilson matrix."""

    def __init__(self, data, indices, indptr, column_count: int):
        self.data = np.ascontiguousarray(data, dtype=float)
        self.indices = np.ascontiguousarray(indices, dtype=np.intp)
        self.indptr = np.ascontiguousarray(indptr, dtype=np.intp)
        self.column_count = int(column_count)
        if self.indptr.ndim != 1 or self.indptr.size == 0:
            raise ValueError("CSR row pointer must be a nonempty vector")
        if self.data.ndim != 1 or self.indices.shape != self.data.shape:
            raise ValueError("CSR data and column indices must be equal vectors")
        if self.indptr[0] != 0 or self.indptr[-1] != self.data.size:
            raise ValueError("CSR row pointer does not cover the stored entries")
        if np.any(np.diff(self.indptr) < 0):
            raise ValueError("CSR row pointer must be nondecreasing")
        if np.any(self.indices < 0) or np.any(self.indices >= self.column_count):
            raise ValueError("CSR column index is outside the matrix")
        if not np.all(np.isfinite(self.data)):
            raise ValueError("CSR Wilson entries must be finite")
        for array in (self.data, self.indices, self.indptr):
            array.setflags(write=False)

    def __len__(self) -> int:
        return int(self.indptr.size - 1)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(len(self))))
        position = int(index)
        if position < 0:
            position += len(self)
        if position < 0 or position >= len(self):
            raise IndexError("CSR Wilson row index out of range")
        start, stop = int(self.indptr[position]), int(self.indptr[position + 1])
        return SparseBRow(
            self.column_count,
            tuple(
                (int(column), float(value))
                for column, value in zip(
                    self.indices[start:stop], self.data[start:stop], strict=True
                )
            ),
        )

    def iter_entries(self) -> Iterator[tuple[tuple[int, float], ...]]:
        for row in range(len(self)):
            start, stop = int(self.indptr[row]), int(self.indptr[row + 1])
            yield tuple(
                (int(column), float(value))
                for column, value in zip(
                    self.indices[start:stop], self.data[start:stop], strict=True
                )
            )


@dataclass(frozen=True)
class SparseBMatrix:
    """Sparse Wilson-B matrix with dense conversion for audits."""

    rows: tuple[SparseBRow, ...]
    column_count: int
    row_labels: tuple[str, ...] = ()
    backend: str = "oracle-sparse-bmatrix.v1"

    def __post_init__(self) -> None:
        column_count = int(self.column_count)
        if column_count < 0:
            raise ValueError("SparseBMatrix column_count must be non-negative")
        if not isinstance(self.rows, CSRBackedRows):
            for row in self.rows:
                if row.size != column_count:
                    raise ValueError("SparseBMatrix row size does not match column_count")
        elif self.rows.column_count != column_count:
            raise ValueError("CSR Wilson row size does not match column_count")
        if self.row_labels and len(self.row_labels) != len(self.rows):
            raise ValueError("SparseBMatrix row_labels length must match rows")
        object.__setattr__(self, "column_count", column_count)

    @classmethod
    def from_csr(
        cls,
        data,
        indices,
        indptr,
        *,
        column_count: int,
        row_labels: tuple[str, ...] = (),
        backend: str = "oracle-sparse-bmatrix.v1",
    ) -> "SparseBMatrix":
        """Construct without expanding native CSR rows into Python objects."""

        rows = CSRBackedRows(data, indices, indptr, column_count)
        return cls(rows=rows, column_count=column_count, row_labels=row_labels, backend=backend)

    def csr_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return canonical CSR arrays without a dense or row-object round trip."""

        if isinstance(self.rows, CSRBackedRows):
            return self.rows.data, self.rows.indices, self.rows.indptr
        data: list[float] = []
        indices: list[int] = []
        indptr = [0]
        for row in self.rows:
            for column, value in row.entries:
                indices.append(int(column))
                data.append(float(value))
            indptr.append(len(data))
        return (
            np.asarray(data, dtype=float),
            np.asarray(indices, dtype=np.intp),
            np.asarray(indptr, dtype=np.intp),
        )

    def csr_payload(self) -> dict[str, object]:
        """Return a compact, JSON-safe ORACLE→SMITH CSR handoff payload."""

        data, indices, indptr = self.csr_arrays()
        return {
            "schema": "matrix.smith.sparse_b_csr.v1",
            "column_count": self.column_count,
            "data": data.tolist(),
            "indices": indices.tolist(),
            "indptr": indptr.tolist(),
            "row_labels": list(self.row_labels),
            "backend": self.backend,
        }

    @classmethod
    def from_csr_payload(cls, payload: dict[str, object]) -> "SparseBMatrix":
        """Validate and restore a :meth:`csr_payload` record without densifying."""

        if payload.get("schema") != "matrix.smith.sparse_b_csr.v1":
            raise ValueError("unsupported sparse-B CSR payload schema")
        return cls.from_csr(
            np.asarray(payload["data"], dtype=float),
            np.asarray(payload["indices"], dtype=np.intp),
            np.asarray(payload["indptr"], dtype=np.intp),
            column_count=int(payload["column_count"]),
            row_labels=tuple(str(value) for value in payload.get("row_labels", ())),
            backend=str(payload.get("backend", "oracle-sparse-bmatrix.v1")),
        )

    def iter_csr_entries(self):
        data, indices, indptr = self.csr_arrays()
        for row in range(self.row_count):
            start, stop = int(indptr[row]), int(indptr[row + 1])
            yield tuple(
                (int(column), float(value))
                for column, value in zip(indices[start:stop], data[start:stop], strict=True)
            )

    @classmethod
    def from_dense(
        cls,
        matrix: Iterable[Iterable[float]] | np.ndarray,
        *,
        row_labels: tuple[str, ...] = (),
        backend: str = "oracle-sparse-bmatrix.v1",
        zero_tolerance: float = SPARSE_ZERO_TOLERANCE,
    ) -> "SparseBMatrix":
        dense = np.asarray(matrix, dtype=float)
        if dense.ndim != 2:
            raise ValueError("SparseBMatrix.from_dense expects a two-dimensional matrix")
        rows = tuple(
            SparseBRow.from_dense(row, zero_tolerance=zero_tolerance) for row in dense
        )
        return cls(rows=rows, column_count=int(dense.shape[1]), row_labels=row_labels, backend=backend)

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def nnz(self) -> int:
        if isinstance(self.rows, CSRBackedRows):
            return int(self.rows.data.size)
        return sum(row.nnz for row in self.rows)

    @property
    def density(self) -> float:
        total = self.row_count * self.column_count
        return 0.0 if total == 0 else float(self.nnz) / float(total)

    def to_dense(self) -> np.ndarray:
        """Materialize a dense matrix for an explicit audit or legacy export.

        This is deliberately not used by the iterative LINK path. Callers
        that only need products with ``B`` must use :meth:`matvec` or
        :meth:`transpose_matvec`.
        """
        if not self.rows:
            return np.zeros((0, self.column_count), dtype=float)
        return np.vstack([row.to_dense() for row in self.rows])

    def assert_finite(self) -> None:
        """Validate stored values without expanding the CSR representation."""

        data, _indices, _indptr = self.csr_arrays()
        if not np.all(np.isfinite(data)):
            raise ValueError("SparseBMatrix contains non-finite Wilson entries")

    def rows_touching_atoms(self, atom_indices_zero_based: Iterable[int]) -> tuple[int, ...]:
        """Rows whose Cartesian support touches an invalidated atom domain."""

        columns = {
            3 * int(atom) + component
            for atom in atom_indices_zero_based
            for component in range(3)
        }
        return tuple(
            row_index
            for row_index, entries in enumerate(self.iter_csr_entries())
            if any(column in columns for column, _value in entries)
        )

    def replace_rows(self, replacements: dict[int, SparseBRow]) -> "SparseBMatrix":
        """Return a CSR matrix with only explicitly invalidated rows replaced."""

        if any(index < 0 or index >= self.row_count for index in replacements):
            raise IndexError("sparse-B replacement row is outside the matrix")
        rows = tuple(
            replacements.get(index, row)
            for index, row in enumerate(self.rows)
        )
        return SparseBMatrix(
            rows=rows,
            column_count=self.column_count,
            row_labels=self.row_labels,
            backend=self.backend,
        )

    def structural_rank(self) -> int:
        """Return the exact bipartite support rank without densification.

        This is a *structural* certificate: it detects missing or disconnected
        derivative support, whereas SMITH's frozen numerical rank certificate
        remains the authority for near-linear dependencies.
        """

        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import maximum_bipartite_matching

        data, indices, indptr = self.csr_arrays()
        support = csr_matrix(
            (np.ones(data.size, dtype=np.int8), indices, indptr),
            shape=(self.row_count, self.column_count),
        )
        matching = maximum_bipartite_matching(support, perm_type="column")
        return int(np.count_nonzero(matching >= 0))

    def matvec(self, vector: Iterable[float] | np.ndarray) -> np.ndarray:
        """Apply ``B`` in time proportional to the stored nonzero entries."""

        values = np.asarray(vector, dtype=float).reshape(-1)
        if values.shape != (self.column_count,):
            raise ValueError("SparseBMatrix matvec input has the wrong dimension")
        if isinstance(self.rows, CSRBackedRows):
            try:
                from matrix_chem import compiled_csr_matvec
                return compiled_csr_matvec(
                    self.rows.data, self.rows.indices, self.rows.indptr, values,
                    column_count=self.column_count,
                )
            except (ImportError, RuntimeError, AttributeError):
                pass
            result = np.zeros(self.row_count, dtype=float)
            counts = np.diff(self.rows.indptr)
            if self.rows.data.size:
                row_indices = np.repeat(np.arange(self.row_count), counts)
                np.add.at(
                    result,
                    row_indices,
                    self.rows.data * values[self.rows.indices],
                )
            return result
        return np.asarray([row.dot_dense(values) for row in self.rows], dtype=float)

    def transpose_matvec(self, vector: Iterable[float] | np.ndarray) -> np.ndarray:
        """Apply ``B.T`` without materializing either sparse or dense CSC data."""

        values = np.asarray(vector, dtype=float).reshape(-1)
        if values.shape != (self.row_count,):
            raise ValueError("SparseBMatrix transpose matvec input has the wrong dimension")
        result = np.zeros(self.column_count, dtype=float)
        if isinstance(self.rows, CSRBackedRows):
            try:
                from matrix_chem import compiled_csr_matvec
                return compiled_csr_matvec(
                    self.rows.data, self.rows.indices, self.rows.indptr, values,
                    column_count=self.column_count, transpose=True,
                )
            except (ImportError, RuntimeError, AttributeError):
                pass
            counts = np.diff(self.rows.indptr)
            if self.rows.data.size:
                row_indices = np.repeat(np.arange(self.row_count), counts)
                np.add.at(
                    result,
                    self.rows.indices,
                    values[row_indices] * self.rows.data,
                )
            return result
        for coefficient, row in zip(values, self.rows, strict=True):
            if coefficient == 0.0:
                continue
            for index, value in row.entries:
                result[index] += float(coefficient) * value
        return result
