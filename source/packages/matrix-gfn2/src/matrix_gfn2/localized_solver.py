"""Sparse local algebra, natural orbitals, and restart contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from . import _gfn2_native
except ImportError:  # pragma: no cover
    _gfn2_native = None

from .locality import LocalizedOrbitalDomains


@dataclass(frozen=True)
class SparseSymmetricMatrix:
    """Lower-triangular sparse symmetric matrix with deterministic ordering."""

    size: int
    rows: np.ndarray
    columns: np.ndarray
    values: np.ndarray
    _native_topology: tuple[np.ndarray, np.ndarray, np.ndarray] | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.size < 0 or self.size > np.iinfo(np.int32).max:
            raise ValueError("sparse matrix size exceeds 32-bit indexing")
        rows = np.asarray(self.rows, dtype=np.int32)
        columns = np.asarray(self.columns, dtype=np.int32)
        values = np.asarray(self.values)
        if rows.ndim != 1 or columns.shape != rows.shape or values.shape != rows.shape:
            raise ValueError("sparse matrix arrays must be equally sized vectors")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "values", values)

    @classmethod
    def from_dense(
        cls,
        matrix: np.ndarray,
        domains: LocalizedOrbitalDomains,
        *,
        threshold: float,
    ) -> SparseSymmetricMatrix:
        dense = np.asarray(matrix, dtype=float)
        if dense.shape != (domains.norbitals, domains.norbitals):
            raise ValueError("matrix shape does not match localized domains")
        if _gfn2_native is not None:
            allowed_left, allowed_right = domains.allowed_pair_arrays()
            rows, columns, values = _gfn2_native.sparse_from_dense(
                dense, allowed_left, allowed_right, float(threshold)
            )
            return cls(
                domains.norbitals,
                np.asarray(rows, dtype=np.int32),
                np.asarray(columns, dtype=np.int32),
                np.asarray(values, dtype=float),
            )
        if not np.allclose(dense, dense.T, atol=1.0e-13):
            raise ValueError("sparse symmetric matrix requires symmetric input")
        entries = [
            (left, right, dense[left, right])
            for left, right in domains.allowed_pairs
            if left == right or abs(dense[left, right]) >= threshold
        ]
        return cls(
            domains.norbitals,
            np.asarray([item[0] for item in entries], dtype=np.intp),
            np.asarray([item[1] for item in entries], dtype=np.intp),
            np.asarray([item[2] for item in entries], dtype=float),
        )

    @property
    def nonzero_count(self) -> int:
        return int(np.count_nonzero(self.values) * 2 - np.count_nonzero(
            self.values[self.rows == self.columns]
        ))

    @property
    def shape(self) -> tuple[int, int]:
        return (self.size, self.size)

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        dense = self.to_dense()
        if dtype is not None:
            dense = dense.astype(dtype, copy=False)
        if copy:
            dense = dense.copy()
        return dense

    def __getitem__(self, key):
        # Compatibility boundary only: internal sparse code never enters it.
        return self.to_dense()[key]

    def copy(self) -> SparseSymmetricMatrix:
        return SparseSymmetricMatrix(
            self.size, self.rows.copy(), self.columns.copy(), self.values.copy()
        )

    def native_topology(self):
        """Return cached row adjacency for repeated native contractions."""

        if self._native_topology is not None:
            return self._native_topology
        if _gfn2_native is None or not hasattr(_gfn2_native, "sparse_topology"):
            return None
        raw = _gfn2_native.sparse_topology(
            int(self.size), self.rows, self.columns
        )
        topology = (
            np.asarray(raw[0], dtype=np.int64),
            np.asarray(raw[1], dtype=np.int32),
            np.asarray(raw[2], dtype=np.int32),
        )
        object.__setattr__(self, "_native_topology", topology)
        return topology

    def to_dense(self) -> np.ndarray:
        result = np.zeros((self.size, self.size), dtype=float)
        result[self.rows, self.columns] = self.values
        result[self.columns, self.rows] = self.values
        return result

    def matvec(self, vector: np.ndarray) -> np.ndarray:
        operand = np.asarray(vector, dtype=float)
        if operand.shape != (self.size,):
            raise ValueError("vector shape does not match sparse matrix")
        if _gfn2_native is not None:
            return np.asarray(
                _gfn2_native.sparse_matvec(
                    int(self.size), self.rows, self.columns, self.values, operand
                ),
                dtype=float,
            )
        result = np.zeros(self.size, dtype=float)
        np.add.at(result, self.rows, self.values * operand[self.columns])
        off_diagonal = self.rows != self.columns
        np.add.at(
            result,
            self.columns[off_diagonal],
            self.values[off_diagonal] * operand[self.rows[off_diagonal]],
        )
        return result

    def values_on(
        self, rows: np.ndarray, columns: np.ndarray
    ) -> np.ndarray:
        """Gather values on a sorted lower-triangular pair list in O(nnz)."""

        target_rows = np.asarray(rows, dtype=np.int64)
        target_columns = np.asarray(columns, dtype=np.int64)
        if target_rows.ndim != 1 or target_columns.shape != target_rows.shape:
            raise ValueError("target sparse pairs must be equally sized vectors")
        if _gfn2_native is not None and hasattr(_gfn2_native, "sparse_gather"):
            return np.asarray(
                _gfn2_native.sparse_gather(
                    int(self.size), self.rows, self.columns, self.values,
                    target_rows, target_columns,
                ),
                dtype=float,
            )
        keys = self.rows.astype(np.int64) * self.size + self.columns
        targets = target_rows * self.size + target_columns
        positions = np.searchsorted(keys, targets)
        result = np.zeros(len(targets), dtype=float)
        valid = positions < len(keys)
        matched = np.zeros(len(targets), dtype=bool)
        matched[valid] = keys[positions[valid]] == targets[valid]
        result[matched] = self.values[positions[matched]]
        return result

    def scaled_add(
        self,
        other: SparseSymmetricMatrix,
        *,
        self_scale: float = 1.0,
        other_scale: float = 1.0,
        threshold: float = 0.0,
    ) -> tuple[SparseSymmetricMatrix, float]:
        """Return a sparse linear combination and its discarded Frobenius norm."""

        if self.size != other.size:
            raise ValueError("sparse matrix dimensions differ")
        if _gfn2_native is not None:
            rows, columns, result_values, discarded = _gfn2_native.sparse_scaled_add(
                int(self.size),
                self.rows, self.columns, self.values,
                other.rows, other.columns, other.values,
                float(self_scale), float(other_scale), float(threshold),
            )
            return (
                SparseSymmetricMatrix(
                    self.size,
                    np.asarray(rows, dtype=np.intp),
                    np.asarray(columns, dtype=np.intp),
                    np.asarray(result_values, dtype=float),
                ),
                float(discarded),
            )
        values: dict[tuple[int, int], float] = {}
        for matrix, scale in ((self, self_scale), (other, other_scale)):
            for row, column, value in zip(
                matrix.rows, matrix.columns, matrix.values, strict=True
            ):
                key = (int(row), int(column))
                values[key] = values.get(key, 0.0) + float(scale) * float(value)
        kept: list[tuple[int, int, float]] = []
        discarded_squared = 0.0
        for (row, column), value in sorted(values.items()):
            if row == column or abs(value) >= threshold:
                kept.append((row, column, value))
            else:
                discarded_squared += (2.0 if row != column else 1.0) * value**2
        return (
            SparseSymmetricMatrix(
                self.size,
                np.asarray([item[0] for item in kept], dtype=np.intp),
                np.asarray([item[1] for item in kept], dtype=np.intp),
                np.asarray([item[2] for item in kept], dtype=float),
            ),
            float(np.sqrt(discarded_squared)),
        )

    def square(
        self,
        domains: LocalizedOrbitalDomains,
        *,
        threshold: float = 0.0,
    ) -> tuple[SparseSymmetricMatrix, float]:
        """Square within localized domains without constructing a dense matrix."""

        if self.size != domains.norbitals:
            raise ValueError("sparse matrix and domain dimensions differ")
        if _gfn2_native is not None:
            allowed_left, allowed_right = domains.allowed_pair_arrays()
            rows, columns, result_values, discarded = _gfn2_native.sparse_square(
                int(self.size), self.rows, self.columns, self.values,
                allowed_left, allowed_right, float(threshold),
            )
            return (
                SparseSymmetricMatrix(
                    self.size,
                    np.asarray(rows, dtype=np.intp),
                    np.asarray(columns, dtype=np.intp),
                    np.asarray(result_values, dtype=float),
                ),
                float(discarded),
            )
        rows: list[dict[int, float]] = [dict() for _ in range(self.size)]
        for row, column, value in zip(self.rows, self.columns, self.values, strict=True):
            i, j, scalar = int(row), int(column), float(value)
            rows[i][j] = scalar
            rows[j][i] = scalar
        entries: list[tuple[int, int, float]] = []
        discarded_squared = 0.0
        for left, right in domains.allowed_pairs:
            common = rows[left].keys() & rows[right].keys()
            value = sum(rows[left][index] * rows[right][index] for index in common)
            if left == right or abs(value) >= threshold:
                entries.append((left, right, value))
            else:
                discarded_squared += (2.0 if left != right else 1.0) * value**2
        return (
            SparseSymmetricMatrix(
                self.size,
                np.asarray([item[0] for item in entries], dtype=np.intp),
                np.asarray([item[1] for item in entries], dtype=np.intp),
                np.asarray([item[2] for item in entries], dtype=float),
            ),
            float(np.sqrt(discarded_squared)),
        )

    def product(
        self,
        other: SparseSymmetricMatrix,
        domains: LocalizedOrbitalDomains,
        *,
        threshold: float = 0.0,
    ) -> tuple[SparseSymmetricMatrix, float]:
        """Return ``(AB+BA)/2`` without leaving the localized domain.

        Matrix polynomials commute in exact arithmetic.  The explicit Jordan
        symmetrization prevents truncation roundoff from creating a hidden
        nonsymmetric component during Newton--Schulz iterations.
        """

        if self.size != other.size or self.size != domains.norbitals:
            raise ValueError("sparse product dimensions differ")
        if _gfn2_native is not None and hasattr(_gfn2_native, "sparse_product"):
            allowed_left, allowed_right = domains.allowed_pair_arrays()
            rows, columns, values, discarded = _gfn2_native.sparse_product(
                int(self.size),
                self.rows, self.columns, self.values,
                other.rows, other.columns, other.values,
                allowed_left, allowed_right, float(threshold),
                int(domains.native_threads),
            )
            return (
                SparseSymmetricMatrix(
                    self.size,
                    np.asarray(rows, dtype=np.intp),
                    np.asarray(columns, dtype=np.intp),
                    np.asarray(values, dtype=float),
                ),
                float(discarded),
            )
        product = 0.5 * (
            self.to_dense() @ other.to_dense()
            + other.to_dense() @ self.to_dense()
        )
        return SparseSymmetricMatrix.from_dense(
            product, domains, threshold=threshold
        ), float(np.linalg.norm(
            product
            - SparseSymmetricMatrix.from_dense(
                product, domains, threshold=threshold
            ).to_dense()
        ))

    def congruence(
        self,
        matrix: SparseSymmetricMatrix,
        domains: LocalizedOrbitalDomains,
        *,
        threshold: float = 0.0,
    ) -> tuple[SparseSymmetricMatrix, float]:
        """Return ``self @ matrix @ self`` without a nonsymmetric intermediate."""

        if self.size != matrix.size or self.size != domains.norbitals:
            raise ValueError("sparse congruence dimensions differ")
        if _gfn2_native is not None and hasattr(_gfn2_native, "sparse_congruence"):
            allowed_left, allowed_right = domains.allowed_pair_arrays()
            topology = self.native_topology()
            topology_args = () if topology is None else tuple(topology)
            rows, columns, values, discarded = _gfn2_native.sparse_congruence(
                int(self.size),
                self.rows, self.columns, self.values,
                matrix.rows, matrix.columns, matrix.values,
                allowed_left, allowed_right, float(threshold),
                *topology_args,
                int(domains.native_threads),
            )
            return (
                SparseSymmetricMatrix(
                    self.size,
                    np.asarray(rows, dtype=np.intp),
                    np.asarray(columns, dtype=np.intp),
                    np.asarray(values, dtype=float),
                ),
                float(discarded),
            )
        dense = self.to_dense() @ matrix.to_dense() @ self.to_dense()
        result = SparseSymmetricMatrix.from_dense(dense, domains, threshold=threshold)
        return result, float(np.linalg.norm(dense - result.to_dense()))

    def frobenius_norm(self) -> float:
        weights = np.where(self.rows == self.columns, 1.0, 2.0)
        return float(np.sqrt(np.sum(weights * self.values * self.values)))

    def frobenius_inner(self, other: SparseSymmetricMatrix) -> float:
        """Frobenius inner product without aligning through a dense matrix."""

        if self.size != other.size:
            raise ValueError("sparse matrix dimensions differ")
        other_values = other.values_on(self.rows, self.columns)
        weights = np.where(self.rows == self.columns, 1.0, 2.0)
        return float(np.sum(weights * self.values * other_values))

    def maximum_absolute_difference(self, other: SparseSymmetricMatrix) -> float:
        """Maximum entrywise difference over the union of sparse patterns."""

        if self.size != other.size:
            raise ValueError("sparse matrix dimensions differ")
        difference, _ = self.scaled_add(other, other_scale=-1.0)
        return float(np.max(np.abs(difference.values), initial=0.0))

    def trace(self) -> float:
        diagonal = self.rows == self.columns
        return float(np.sum(self.values[diagonal]))


@dataclass(frozen=True)
class SparseInverseSqrtResult:
    inverse_sqrt: SparseSymmetricMatrix
    iterations: int
    residual: float
    accumulated_error_bound: float
    converged: bool


def sparse_newton_schulz_inverse_sqrt(
    overlap: SparseSymmetricMatrix,
    domains: LocalizedOrbitalDomains,
    *,
    tolerance: float = 1.0e-8,
    maximum_iterations: int = 100,
    truncation_threshold: float = 1.0e-12,
    input_error_bound: float = 0.0,
) -> SparseInverseSqrtResult:
    """Domain-sparse inverse square root with a propagated error bound."""

    if overlap.size != domains.norbitals:
        raise ValueError("overlap and localized domains differ")
    if tolerance <= 0.0 or truncation_threshold < 0.0:
        raise ValueError("invalid sparse inverse-square-root tolerance")
    # Newton--Schulz converges much more slowly if we normalize by the full
    # Frobenius norm: for a local matrix that norm grows as sqrt(N), even when
    # the spectral radius stays bounded.  The maximum absolute row sum is a
    # Gershgorin upper bound on the spectral radius, is available in O(nnz),
    # and keeps the initial spectrum close to the unit interval as system size
    # grows.
    row_sums = np.zeros(overlap.size, dtype=float)
    absolute_values = np.abs(overlap.values)
    np.add.at(row_sums, overlap.rows, absolute_values)
    off_diagonal = overlap.rows != overlap.columns
    np.add.at(
        row_sums,
        overlap.columns[off_diagonal],
        absolute_values[off_diagonal],
    )
    scale = float(np.max(row_sums, initial=0.0))
    if scale <= np.finfo(float).tiny:
        raise np.linalg.LinAlgError("overlap matrix is singular")
    size = overlap.size
    indices = np.arange(size, dtype=np.intp)
    identity = SparseSymmetricMatrix(
        size, indices, indices.copy(), np.ones(size, dtype=float)
    )
    y = SparseSymmetricMatrix(
        size, overlap.rows.copy(), overlap.columns.copy(), overlap.values / scale
    )
    z = identity
    y_bound = float(input_error_bound) / scale
    z_bound = 0.0

    def product_with_bound(left, left_bound, right, right_bound, threshold):
        value, discarded = left.product(right, domains, threshold=threshold)
        if np.isfinite(left_bound) and np.isfinite(right_bound):
            with np.errstate(over="ignore", invalid="ignore"):
                bound = float(
                    left.frobenius_norm() * right_bound
                    + right.frobenius_norm() * left_bound
                    + left_bound * right_bound
                    + discarded
                )
            if not np.isfinite(bound):
                bound = float("inf")
        else:
            bound = float("inf")
        return value, bound

    residual = float("inf")
    residual_bound = float("inf")
    candidate = z
    for iteration in range(1, int(maximum_iterations) + 1):
        zy, zy_bound = product_with_bound(
            z, z_bound, y, y_bound, truncation_threshold
        )
        zy_difference, _ = identity.scaled_add(
            zy, self_scale=1.0, other_scale=-1.0, threshold=0.0
        )
        zy_residual = zy_difference.frobenius_norm()
        correction, discarded = identity.scaled_add(
            zy,
            self_scale=1.5,
            other_scale=-0.5,
            threshold=truncation_threshold,
        )
        correction_bound = 0.5 * zy_bound + discarded
        y, y_bound = product_with_bound(
            y, y_bound, correction, correction_bound, truncation_threshold
        )
        z, z_bound = product_with_bound(
            correction, correction_bound, z, z_bound, truncation_threshold
        )
        candidate = SparseSymmetricMatrix(
            size,
            z.rows.copy(),
            z.columns.copy(),
            z.values / np.sqrt(scale),
        )
        candidate_bound = z_bound / np.sqrt(scale)
        # X S X is two additional sparse products.  The coupled
        # Newton--Schulz residual I-ZY is already available and is a necessary
        # convergence condition, so defer the expensive a-posteriori check
        # until the iteration is in its convergence basin.
        if zy_residual > max(1.0e-4, np.sqrt(tolerance)):
            continue
        xs, xs_bound = product_with_bound(
            candidate, candidate_bound, overlap, input_error_bound,
            truncation_threshold,
        )
        xsx, xsx_bound = product_with_bound(
            xs, xs_bound, candidate, candidate_bound, truncation_threshold,
        )
        difference, discarded = identity.scaled_add(
            xsx, self_scale=1.0, other_scale=-1.0, threshold=0.0
        )
        residual = difference.frobenius_norm()
        residual_bound = xsx_bound + discarded
        # The recursive bound counts the same discarded domain complement at
        # every polynomial multiplication and eventually becomes vacuous for
        # long localized systems.  The explicitly evaluated X S X-I residual
        # is the relevant a-posteriori certificate; the accumulated bound is
        # retained as diagnostic telemetry, not added repeatedly to it.
        residual_tolerance = tolerance * (
            1.0 if size < 128 else np.sqrt(size)
        )
        if residual <= residual_tolerance:
            return SparseInverseSqrtResult(
                candidate, iteration, residual, residual_bound, True
            )
        # The recursively propagated truncation bound can overflow for large
        # systems even while the directly evaluated AO orthogonality residual
        # is finite and still converging.  Keep that bound as diagnostic
        # telemetry; only a non-finite measured residual is a reason to stop.
        if not np.isfinite(residual):
            break
    return SparseInverseSqrtResult(
        candidate, iteration, residual, residual_bound, False
    )


def local_natural_orbitals(
    density: np.ndarray,
    orbital_indices: Sequence[int],
    *,
    occupation_threshold: float = 1.0e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize a local density block and retain occupied natural orbitals."""

    indices = np.asarray(tuple(orbital_indices), dtype=np.intp)
    block = np.asarray(density, dtype=float)[np.ix_(indices, indices)]
    occupations, coefficients = np.linalg.eigh(0.5 * (block + block.T))
    order = np.argsort(occupations)[::-1]
    occupations = occupations[order]
    coefficients = coefficients[:, order]
    keep = occupations >= occupation_threshold
    return occupations[keep], coefficients[:, keep]


def geometry_fingerprint(
    atomic_numbers: Sequence[int],
    coordinates_bohr: np.ndarray,
    orbital_atom: Sequence[int],
) -> str:
    numbers = np.asarray(tuple(atomic_numbers), dtype="<i4")
    coordinates = np.asarray(coordinates_bohr, dtype="<f8")
    assignment = np.asarray(tuple(orbital_atom), dtype="<i4")
    digest = sha256()
    for array in (numbers, coordinates, assignment):
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class LocalizedRestart:
    fingerprint: str
    density: np.ndarray
    shell_charges: np.ndarray
    electrons: int
    multiplicity: int

    def validate(self, fingerprint: str, norbitals: int) -> None:
        if self.fingerprint != fingerprint:
            raise ValueError("restart geometry/basis fingerprint mismatch")
        if self.density.shape != (norbitals, norbitals):
            raise ValueError("restart density dimension mismatch")
        if not np.all(np.isfinite(self.density)) or not np.all(
            np.isfinite(self.shell_charges)
        ):
            raise ValueError("restart contains non-finite values")

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            Path(path),
            fingerprint=np.asarray(self.fingerprint),
            density=np.asarray(self.density, dtype=float),
            shell_charges=np.asarray(self.shell_charges, dtype=float),
            electrons=np.asarray(self.electrons, dtype=np.int64),
            multiplicity=np.asarray(self.multiplicity, dtype=np.int64),
        )

    @classmethod
    def load(cls, path: str | Path) -> LocalizedRestart:
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(
                fingerprint=str(data["fingerprint"].item()),
                density=np.asarray(data["density"], dtype=float),
                shell_charges=np.asarray(data["shell_charges"], dtype=float),
                electrons=int(data["electrons"].item()),
                multiplicity=int(data["multiplicity"].item()),
            )


__all__ = [
    "LocalizedRestart",
    "SparseInverseSqrtResult",
    "SparseSymmetricMatrix",
    "geometry_fingerprint",
    "local_natural_orbitals",
    "sparse_newton_schulz_inverse_sqrt",
]
