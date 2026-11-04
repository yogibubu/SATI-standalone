"""Pair-packed SCF adjoints for localized spin-flip TDA."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations, product
from typing import Callable

import numpy as np

from . import _gfn2_native
from .localized_solver import SparseSymmetricMatrix
from .locality import LocalizedOrbitalDomains
from .sf_tda_gradient import _matrix_free_gmres
from .sf_tda_local import SparseSFTDAOperator


@dataclass(frozen=True)
class SparseRootMatrixBatch:
    """Several symmetric AO matrices sharing one lower-triangle pattern."""

    size: int
    rows: np.ndarray
    columns: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        rows = np.asarray(self.rows, dtype=np.int64)
        columns = np.asarray(self.columns, dtype=np.int64)
        values = np.asarray(self.values, dtype=float)
        if rows.ndim != 1 or columns.shape != rows.shape:
            raise ValueError("sparse root matrix pattern is inconsistent")
        if values.ndim != 2 or values.shape[1] != len(rows):
            raise ValueError("sparse root matrix values must be nroot x npair")
        if np.any(rows < columns) or np.any(rows >= self.size) or np.any(columns < 0):
            raise ValueError("sparse root matrix uses invalid lower-triangle indices")
        if not np.all(np.isfinite(values)):
            raise ValueError("sparse root matrix values must be finite")
        for array in (rows, columns, values):
            array.setflags(write=False)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "values", values)

    @property
    def nroots(self) -> int:
        return int(self.values.shape[0])

    def root(self, index: int) -> SparseSymmetricMatrix:
        return SparseSymmetricMatrix(self.size, self.rows, self.columns, self.values[index])


@dataclass(frozen=True)
class SparseOrbitalCoefficients:
    """Column-compressed localized orbital coefficients."""

    nao: int
    offsets: np.ndarray
    rows: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        offsets = np.asarray(self.offsets, dtype=np.int64)
        rows = np.asarray(self.rows, dtype=np.int64)
        values = np.asarray(self.values, dtype=float)
        if offsets.ndim != 1 or len(offsets) < 2 or offsets[0] != 0:
            raise ValueError("sparse orbital offsets are inconsistent")
        if offsets[-1] != len(rows) or rows.shape != values.shape:
            raise ValueError("sparse orbital coefficient arrays are inconsistent")
        if np.any(np.diff(offsets) < 1) or np.any(rows < 0) or np.any(rows >= self.nao):
            raise ValueError("sparse orbital coefficient indices are invalid")
        for orbital in range(len(offsets) - 1):
            start, stop = offsets[orbital : orbital + 2]
            if np.any(np.diff(rows[start:stop]) <= 0):
                raise ValueError("AO rows within each sparse orbital must be sorted and unique")
        if not np.all(np.isfinite(values)):
            raise ValueError("sparse orbital coefficients must be finite")
        for array in (offsets, rows, values):
            array.setflags(write=False)
        object.__setattr__(self, "offsets", offsets)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "values", values)

    @property
    def nmo(self) -> int:
        return len(self.offsets) - 1

    @classmethod
    def from_dense(
        cls, coefficients: np.ndarray, *, threshold: float = 0.0
    ) -> "SparseOrbitalCoefficients":
        dense = np.asarray(coefficients, dtype=float)
        if dense.ndim != 2 or threshold < 0.0:
            raise ValueError("dense orbital coefficients or threshold are invalid")
        offsets = [0]
        rows: list[int] = []
        values: list[float] = []
        for orbital in range(dense.shape[1]):
            selected = np.flatnonzero(np.abs(dense[:, orbital]) > threshold)
            if not len(selected):
                raise ValueError("orbital truncation removed an entire orbital")
            rows.extend(int(value) for value in selected)
            values.extend(float(value) for value in dense[selected, orbital])
            offsets.append(len(rows))
        return cls(
            dense.shape[0],
            np.asarray(offsets, dtype=np.int64),
            np.asarray(rows, dtype=np.int64),
            np.asarray(values),
        )


@dataclass(frozen=True)
class SparseOrbitalMetricBatch:
    """Root metric values on one orbital-rotation graph plus its diagonal."""

    pair_values: np.ndarray
    diagonal_values: np.ndarray

    def __post_init__(self) -> None:
        pairs = np.asarray(self.pair_values, dtype=float)
        diagonal = np.asarray(self.diagonal_values, dtype=float)
        if pairs.ndim != 2 or diagonal.ndim != 2 or pairs.shape[0] != diagonal.shape[0]:
            raise ValueError("sparse orbital metric batch dimensions are inconsistent")
        if not np.all(np.isfinite(pairs)) or not np.all(np.isfinite(diagonal)):
            raise ValueError("sparse orbital metric values must be finite")
        pairs.setflags(write=False)
        diagonal.setflags(write=False)
        object.__setattr__(self, "pair_values", pairs)
        object.__setattr__(self, "diagonal_values", diagonal)

    @property
    def nroots(self) -> int:
        return self.pair_values.shape[0]


@dataclass(frozen=True)
class SparseSFTDAOperatorAdjoint:
    """Root derivatives with respect to the stored sparse SF Hamiltonian."""

    diagonal_values: np.ndarray
    pair_values: np.ndarray
    rows: np.ndarray
    columns: np.ndarray

    def __post_init__(self) -> None:
        diagonal = np.asarray(self.diagonal_values, dtype=float)
        pairs = np.asarray(self.pair_values, dtype=float)
        rows = np.asarray(self.rows, dtype=np.int64)
        columns = np.asarray(self.columns, dtype=np.int64)
        if diagonal.ndim != 2 or pairs.shape != (diagonal.shape[0], len(rows)):
            raise ValueError("sparse SF adjoint dimensions are inconsistent")
        for array in (diagonal, pairs, rows, columns):
            array.setflags(write=False)
        object.__setattr__(self, "diagonal_values", diagonal)
        object.__setattr__(self, "pair_values", pairs)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "columns", columns)


@dataclass(frozen=True)
class SparseSylvesterResult:
    solution: SparseSymmetricMatrix
    residual_norm: float
    iterations: int
    converged: bool


def sparse_square_root_adjoint(
    overlap: SparseSymmetricMatrix,
    inverse_square_root: SparseSymmetricMatrix,
    domains: LocalizedOrbitalDomains,
    source: SparseSymmetricMatrix,
    *,
    tolerance: float = 1.0e-10,
    maximum_iterations: int = 200,
    truncation_threshold: float = 0.0,
) -> SparseSylvesterResult:
    """Solve ``sqrt(S) X + X sqrt(S) = source`` on one sparse pattern.

    This is the reverse Fréchet derivative of the symmetric square root.  The
    solve uses a diagonally preconditioned conjugate gradient in the weighted
    lower-triangle metric and never materializes an AO-square array.
    """

    if overlap.size != inverse_square_root.size or overlap.size != source.size:
        raise ValueError("sparse square-root adjoint dimensions differ")
    square_root, discarded = overlap.product(
        inverse_square_root, domains, threshold=truncation_threshold
    )
    if discarded > tolerance:
        return SparseSylvesterResult(source.copy(), float(discarded), 0, False)
    rows, columns = domains.allowed_pair_arrays()
    template = SparseSymmetricMatrix(
        overlap.size,
        rows,
        columns,
        np.zeros(len(rows), dtype=float),
    )
    right_hand_side = source.values_on(rows, columns)
    multiplicity = np.where(rows == columns, 1.0, 2.0)

    def dot(left: np.ndarray, right: np.ndarray) -> float:
        return float(np.dot(multiplicity * left, right))

    def apply(values: np.ndarray) -> np.ndarray:
        trial = SparseSymmetricMatrix(overlap.size, rows, columns, np.asarray(values, dtype=float))
        left, _ = square_root.product(trial, domains, threshold=truncation_threshold)
        right, _ = trial.product(square_root, domains, threshold=truncation_threshold)
        summed, _ = left.scaled_add(right, threshold=truncation_threshold)
        return summed.values_on(rows, columns)

    diagonal = square_root.values_on(rows, rows) + square_root.values_on(columns, columns)
    diagonal = np.where(np.abs(diagonal) > 1.0e-12, diagonal, 1.0)
    solution = np.zeros_like(right_hand_side)
    residual = right_hand_side - apply(solution)
    preconditioned = residual / diagonal
    direction = preconditioned.copy()
    scalar = dot(residual, preconditioned)
    target = tolerance * max(1.0, np.sqrt(dot(right_hand_side, right_hand_side)))
    residual_norm = np.sqrt(dot(residual, residual))
    if residual_norm <= target:
        return SparseSylvesterResult(template, residual_norm, 0, True)
    for iteration in range(1, int(maximum_iterations) + 1):
        image = apply(direction)
        denominator = dot(direction, image)
        if denominator <= np.finfo(float).tiny:
            break
        step = scalar / denominator
        solution += step * direction
        residual -= step * image
        residual_norm = np.sqrt(max(0.0, dot(residual, residual)))
        if residual_norm <= target:
            return SparseSylvesterResult(
                SparseSymmetricMatrix(overlap.size, rows, columns, solution),
                residual_norm,
                iteration,
                True,
            )
        preconditioned = residual / diagonal
        updated_scalar = dot(residual, preconditioned)
        direction = preconditioned + (updated_scalar / scalar) * direction
        scalar = updated_scalar
    return SparseSylvesterResult(
        SparseSymmetricMatrix(overlap.size, rows, columns, solution),
        residual_norm,
        int(maximum_iterations),
        False,
    )


def sparse_sf_tda_operator_adjoint(
    operator: SparseSFTDAOperator, amplitudes: np.ndarray
) -> SparseSFTDAOperatorAdjoint:
    """Reverse the eigenvalue contraction without a dense CSF outer product."""

    roots = np.asarray(amplitudes, dtype=float)
    if roots.ndim != 2 or roots.shape[1] != operator.size:
        raise ValueError("localized SF amplitudes and operator dimensions differ")
    diagonal = roots * roots
    pairs = 2.0 * roots[:, operator.rows] * roots[:, operator.columns]
    return SparseSFTDAOperatorAdjoint(diagonal, pairs, operator.rows, operator.columns)


def sparse_ao_from_mo(
    coefficients: np.ndarray,
    matrices_mo: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
) -> SparseRootMatrixBatch:
    """Evaluate only allowed AO pairs of ``C M C.T`` for every root."""

    coefficients = np.asarray(coefficients, dtype=float)
    matrices = np.asarray(matrices_mo, dtype=float)
    row = np.asarray(rows, dtype=np.int64)
    column = np.asarray(columns, dtype=np.int64)
    if matrices.ndim != 3 or matrices.shape[1:] != (
        coefficients.shape[1],
        coefficients.shape[1],
    ):
        raise ValueError("MO matrix batch and coefficient dimensions differ")
    values = np.einsum(
        "pi,rij,pj->rp",
        coefficients[row],
        matrices,
        coefficients[column],
        optimize=True,
    )
    return SparseRootMatrixBatch(coefficients.shape[0], row, column, values)


def sparse_mo_from_ao(matrix: SparseSymmetricMatrix, coefficients: np.ndarray) -> np.ndarray:
    """Contract ``C.T A C`` directly from a pair-packed AO matrix."""

    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.ndim != 2 or coefficients.shape[0] != matrix.size:
        raise ValueError("sparse AO matrix and coefficients differ")
    left = coefficients[matrix.rows]
    right = coefficients[matrix.columns]
    result = np.einsum("p,pi,pj->ij", matrix.values, left, right, optimize=True)
    off = matrix.rows != matrix.columns
    result += np.einsum(
        "p,pi,pj->ij",
        matrix.values[off],
        right[off],
        left[off],
        optimize=True,
    )
    return result


def _sparse_ao_from_rotation_values(
    coefficients: np.ndarray | SparseOrbitalCoefficients,
    left: np.ndarray,
    right: np.ndarray,
    values: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    *,
    diagonal_values: np.ndarray | None = None,
) -> SparseSymmetricMatrix:
    """Transform only stored MO rotations to one AO pair pattern."""

    if isinstance(coefficients, SparseOrbitalCoefficients):
        diagonal = (
            np.zeros(coefficients.nmo)
            if diagonal_values is None
            else np.asarray(diagonal_values, dtype=float)
        )
        transformed = _gfn2_native.sparse_rotation_to_ao(
            coefficients.nao,
            coefficients.offsets,
            coefficients.rows,
            coefficients.values,
            np.asarray(left, dtype=np.int64),
            np.asarray(right, dtype=np.int64),
            np.asarray(values, dtype=float),
            np.asarray(rows, dtype=np.int64),
            np.asarray(columns, dtype=np.int64),
            diagonal,
        )
        return SparseSymmetricMatrix(
            coefficients.nao, rows, columns, np.asarray(transformed, dtype=float)
        )
    result = np.zeros(len(rows), dtype=float)
    for p, q, value in zip(left, right, values, strict=True):
        result += float(value) * (
            coefficients[rows, p] * coefficients[columns, q]
            + coefficients[rows, q] * coefficients[columns, p]
        )
    if diagonal_values is not None:
        result += np.einsum(
            "i,pi,pi->p",
            np.asarray(diagonal_values, dtype=float),
            coefficients[rows],
            coefficients[columns],
            optimize=True,
        )
    return SparseSymmetricMatrix(coefficients.shape[0], rows, columns, result)


def _sparse_mo_rotation_values(
    matrix: SparseSymmetricMatrix,
    coefficients: np.ndarray | SparseOrbitalCoefficients,
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Contract selected off-diagonal MO pairs and every MO diagonal."""

    if isinstance(coefficients, SparseOrbitalCoefficients):
        selected, diagonal = _gfn2_native.sparse_ao_to_rotation(
            coefficients.nao,
            coefficients.offsets,
            coefficients.rows,
            coefficients.values,
            matrix.rows,
            matrix.columns,
            matrix.values,
            np.asarray(left, dtype=np.int64),
            np.asarray(right, dtype=np.int64),
        )
        return np.asarray(selected, dtype=float), np.asarray(diagonal, dtype=float)
    row_coefficients = coefficients[matrix.rows]
    column_coefficients = coefficients[matrix.columns]
    off_ao = matrix.rows != matrix.columns
    diagonal = np.einsum(
        "p,pi,pi->i",
        matrix.values,
        row_coefficients,
        column_coefficients,
        optimize=True,
    )
    diagonal += np.einsum(
        "p,pi,pi->i",
        matrix.values[off_ao],
        column_coefficients[off_ao],
        row_coefficients[off_ao],
        optimize=True,
    )
    selected = np.empty(len(left), dtype=float)
    for index, (p, q) in enumerate(zip(left, right, strict=True)):
        value = np.dot(
            matrix.values,
            row_coefficients[:, p] * column_coefficients[:, q],
        )
        value += np.dot(
            matrix.values[off_ao],
            column_coefficients[off_ao, p] * row_coefficients[off_ao, q],
        )
        selected[index] = value
    return selected, diagonal


@dataclass(frozen=True)
class SparseMatrixFreeZVectorResult:
    orbital_source: np.ndarray
    orbital_z_vector: np.ndarray
    response_density: SparseRootMatrixBatch
    response_weighted_density: SparseRootMatrixBatch
    residual_norms: np.ndarray
    iterations: np.ndarray
    discarded_source_norms: np.ndarray
    rotation_space: "LocalizedOrbitalRotationSpace | None" = None
    certificates: tuple["SparseZVectorCertificate", ...] = ()
    coefficient_backend: str = "dense"


@dataclass(frozen=True)
class LocalizedRotationPolicy:
    """Adaptive spatial policy and rigorous Z-vector error target."""

    initial_cutoff_bohr: float = 8.0
    maximum_cutoff_bohr: float = 32.0
    expansion_factor: float = 1.75
    orbital_tail_tolerance: float = 1.0e-7
    error_tolerance: float = 1.0e-8

    def __post_init__(self) -> None:
        if self.initial_cutoff_bohr <= 0.0:
            raise ValueError("rotation cutoff must be positive")
        if self.maximum_cutoff_bohr < self.initial_cutoff_bohr:
            raise ValueError("maximum rotation cutoff is smaller than the initial cutoff")
        if self.expansion_factor <= 1.0:
            raise ValueError("rotation-domain expansion factor must exceed one")
        if not 0.0 <= self.orbital_tail_tolerance < 1.0:
            raise ValueError("orbital tail tolerance must lie in [0,1)")
        if self.error_tolerance <= 0.0:
            raise ValueError("rotation error tolerance must be positive")


@dataclass(frozen=True)
class LocalizedOrbitalRotationSpace:
    """Spatially bounded canonical-orbital rotations."""

    left: np.ndarray
    right: np.ndarray
    cutoff_bohr: float
    full_pair_count: int
    orbital_centers_bohr: np.ndarray
    orbital_tail_norms: np.ndarray

    def __post_init__(self) -> None:
        left = np.asarray(self.left, dtype=np.int64)
        right = np.asarray(self.right, dtype=np.int64)
        centers = np.asarray(self.orbital_centers_bohr, dtype=float)
        tails = np.asarray(self.orbital_tail_norms, dtype=float)
        if left.shape != right.shape or np.any(left >= right):
            raise ValueError("localized rotation pairs must satisfy left < right")
        if centers.ndim != 2 or centers.shape[1] != 3 or tails.shape != (len(centers),):
            raise ValueError("localized rotation diagnostics are inconsistent")
        if np.any(tails < 0.0) or not np.all(np.isfinite(centers)):
            raise ValueError("localized rotation diagnostics must be finite")
        for value in (left, right, centers, tails):
            value.setflags(write=False)
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)
        object.__setattr__(self, "orbital_centers_bohr", centers)
        object.__setattr__(self, "orbital_tail_norms", tails)

    @property
    def size(self) -> int:
        return len(self.left)

    @property
    def complete(self) -> bool:
        return self.size == self.full_pair_count


@dataclass(frozen=True)
class SparseZVectorCertificate:
    """A posteriori error certificate for one truncated external Z-vector."""

    full_residual_norm: float
    omitted_residual_norm: float
    response_frobenius_bound: float
    inverse_jacobian_bound: float
    z_error_bound: float
    qualified: bool


def build_localized_rotation_space(
    lowdin_coefficients: np.ndarray | SparseOrbitalCoefficients,
    ao_atom: np.ndarray,
    coordinates_bohr: np.ndarray,
    *,
    cutoff_bohr: float,
    tail_tolerance: float,
) -> LocalizedOrbitalRotationSpace:
    """Build a rotation graph from atom populations without an MO-square mask."""

    assignment = np.asarray(ao_atom, dtype=np.int64)
    coordinates = np.asarray(coordinates_bohr, dtype=float)
    supports: list[set[int]] = []
    if isinstance(lowdin_coefficients, SparseOrbitalCoefficients):
        sparse = lowdin_coefficients
        nmo = sparse.nmo
        centers = np.zeros((nmo, 3))
        tails = np.empty(nmo, dtype=float)
        for orbital in range(nmo):
            start, stop = sparse.offsets[orbital : orbital + 2]
            atoms = assignment[sparse.rows[start:stop]]
            populations: dict[int, float] = {}
            for atom, value in zip(atoms, sparse.values[start:stop], strict=True):
                populations[int(atom)] = populations.get(int(atom), 0.0) + value * value
            total = sum(populations.values())
            normalized = sorted(
                ((atom, value / total) for atom, value in populations.items()),
                key=lambda item: (-item[1], item[0]),
            )
            centers[orbital] = sum(
                (weight * coordinates[atom] for atom, weight in normalized),
                start=np.zeros(3),
            )
            cumulative = 0.0
            support: set[int] = set()
            for atom, weight in normalized:
                support.add(atom)
                cumulative += weight
                if cumulative >= 1.0 - tail_tolerance:
                    break
            supports.append(support)
            tails[orbital] = np.sqrt(max(0.0, 1.0 - cumulative))
    else:
        lowdin = np.asarray(lowdin_coefficients, dtype=float)
        nmo = lowdin.shape[1]
        populations = np.zeros((nmo, len(coordinates)), dtype=float)
        for atom in range(len(coordinates)):
            populations[:, atom] = np.sum(lowdin[assignment == atom] ** 2, axis=0)
        totals = np.sum(populations, axis=1)
        populations /= np.where(totals > 0.0, totals, 1.0)[:, None]
        centers = populations @ coordinates
        tails = np.empty(nmo, dtype=float)
        for orbital in range(nmo):
            order = np.argsort(-populations[orbital], kind="stable")
            count = int(
                np.searchsorted(np.cumsum(populations[orbital, order]), 1.0 - tail_tolerance) + 1
            )
            support = set(int(value) for value in order[: max(1, count)])
            supports.append(support)
            tails[orbital] = np.sqrt(
                max(0.0, 1.0 - float(np.sum(populations[orbital, list(support)])))
            )
    pairs: set[tuple[int, int]] = set()
    support_owners: dict[int, list[int]] = {}
    for orbital, support in enumerate(supports):
        for atom in support:
            support_owners.setdefault(atom, []).append(orbital)
    for owners in support_owners.values():
        pairs.update(combinations(owners, 2))
    if np.isinf(cutoff_bohr):
        pairs.update(combinations(range(nmo), 2))
    else:
        width = float(cutoff_bohr)
        cells: dict[tuple[int, int, int], list[int]] = {}
        for orbital, center in enumerate(centers):
            key = tuple(int(value) for value in np.floor(center / width))
            cells.setdefault(key, []).append(orbital)
        neighbor_offsets = tuple(product((-1, 0, 1), repeat=3))
        for key, members in cells.items():
            for offset in neighbor_offsets:
                adjacent = tuple(key[axis] + offset[axis] for axis in range(3))
                for first in members:
                    for second in cells.get(adjacent, ()):
                        if first >= second:
                            continue
                        if np.linalg.norm(centers[first] - centers[second]) <= width:
                            pairs.add((first, second))
    ordered_pairs = sorted(pairs)
    return LocalizedOrbitalRotationSpace(
        np.asarray([pair[0] for pair in ordered_pairs], dtype=np.int64),
        np.asarray([pair[1] for pair in ordered_pairs], dtype=np.int64),
        float(cutoff_bohr),
        nmo * (nmo - 1) // 2,
        centers,
        tails,
    )


def certify_localized_z_vector(
    coefficients: np.ndarray,
    orbital_energies_hartree: np.ndarray,
    occupations: np.ndarray,
    full_source: np.ndarray,
    result: SparseMatrixFreeZVectorResult,
    ao_rows: np.ndarray,
    ao_columns: np.ndarray,
    fock_density_response: Callable[[SparseSymmetricMatrix], SparseSymmetricMatrix],
    *,
    error_tolerance: float,
) -> tuple[SparseZVectorCertificate, ...]:
    """Certify a local rotation solve from its complete matrix-free residual."""

    space = result.rotation_space
    if space is None:
        raise ValueError("Z-vector certification requires a localized rotation space")
    coefficients = np.asarray(coefficients, dtype=float)
    energies = np.asarray(orbital_energies_hartree, dtype=float)
    occupation = np.asarray(occupations, dtype=float)
    full_left, full_right = np.triu_indices(len(energies), 1)
    gaps = energies[full_right] - energies[full_left]
    if np.any(np.abs(gaps) <= 1.0e-8):
        raise ValueError("Z-vector certificate encountered a canonical degeneracy")
    occupation_difference = occupation[full_right] - occupation[full_left]
    keys = full_left.astype(np.int64) * len(energies) + full_right
    local_keys = space.left.astype(np.int64) * len(energies) + space.right
    local_positions = np.searchsorted(keys, local_keys)

    def apply_full(vector: np.ndarray) -> np.ndarray:
        scaled = np.asarray(vector, dtype=float) / gaps
        trial = np.zeros((len(energies), len(energies)), dtype=float)
        trial[full_left, full_right] = 0.5 * scaled
        trial[full_right, full_left] = 0.5 * scaled
        packed = sparse_ao_from_mo(coefficients, trial[None], ao_rows, ao_columns).root(0)
        response = fock_density_response(packed)
        response_mo = sparse_mo_from_ao(response, coefficients)
        return np.asarray(vector, dtype=float) - (
            2.0 * occupation_difference * response_mo[full_left, full_right]
        )

    omitted = np.ones(len(full_left), dtype=bool)
    omitted[local_positions] = False
    if space.complete:
        response_bound = 0.0
        inverse_bound = 0.0
    else:
        response_squared = 0.0
        unit = np.zeros(len(full_left), dtype=float)
        for column in range(len(full_left)):
            unit[column] = 1.0
            response_column = unit - apply_full(unit)
            response_squared += float(np.dot(response_column, response_column))
            unit[column] = 0.0
        response_bound = float(np.sqrt(response_squared))
        inverse_bound = 1.0 / (1.0 - response_bound) if response_bound < 1.0 else float("inf")
    certificates: list[SparseZVectorCertificate] = []
    for root in range(len(full_source)):
        embedded = np.zeros(len(full_left), dtype=float)
        embedded[local_positions] = result.orbital_z_vector[root]
        residual = apply_full(embedded) - full_source[root]
        full_norm = float(np.linalg.norm(residual))
        omitted_norm = float(np.linalg.norm(residual[omitted]))
        if space.complete:
            error_bound = 0.0
            qualified = True
        else:
            error_bound = float(inverse_bound * full_norm)
            qualified = bool(np.isfinite(error_bound) and error_bound <= error_tolerance)
        certificates.append(
            SparseZVectorCertificate(
                full_norm,
                omitted_norm,
                response_bound,
                inverse_bound,
                error_bound,
                qualified,
            )
        )
    return tuple(certificates)


@dataclass(frozen=True)
class SparseLocalizedSFTDALagrangian:
    """Complete pair-packed Lagrangian for localized SF-TDA roots."""

    response_density: SparseRootMatrixBatch
    response_weighted_density: SparseRootMatrixBatch
    explicit_kernel_gradient_hartree_per_bohr: np.ndarray
    z_vector: SparseMatrixFreeZVectorResult
    lowdin_residual_norms: np.ndarray


def _sparse_matmat(matrix: SparseSymmetricMatrix, vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=float)
    if values.ndim != 2 or values.shape[0] != matrix.size:
        raise ValueError("sparse matrix/multivector dimensions differ")
    return np.column_stack(tuple(matrix.matvec(values[:, i]) for i in range(values.shape[1])))


def localized_sparse_sf_tda_lagrangian(
    state,
    operator: SparseSFTDAOperator,
    amplitudes: np.ndarray,
    overlap: SparseSymmetricMatrix,
    inverse_square_root: SparseSymmetricMatrix,
    domains: LocalizedOrbitalDomains,
    fock_density_response: Callable[[SparseSymmetricMatrix], SparseSymmetricMatrix],
    *,
    z_tolerance: float = 1.0e-10,
    maximum_z_iterations: int = 200,
    lowdin_tolerance: float = 1.0e-10,
    rotation_policy: LocalizedRotationPolicy | None = None,
) -> SparseLocalizedSFTDALagrangian:
    """Reverse localized SF-TDA through a fully pair-packed SCF adjoint.

    Only the bounded configuration graph and the common AO pair pattern are
    traversed.  No CSF-square, AO-square, atom-square, or CP-SCF matrix is
    materialized.
    """

    roots = np.asarray(amplitudes, dtype=float)
    adjoint = sparse_sf_tda_operator_adjoint(operator, roots)
    space = operator.space
    nroot = len(roots)
    nao = state.nao
    nmo = state.nmo
    coordinates = np.asarray(state.coordinates_bohr, dtype=float)
    coefficients = np.asarray(state.coefficients, dtype=float)
    occupations = np.asarray(state.alpha_occupations + state.beta_occupations)
    energies = np.asarray(state.orbital_energies_hartree)
    rows, columns = domains.allowed_pair_arrays()
    square_root, discarded = overlap.product(inverse_square_root, domains, threshold=0.0)
    if discarded > lowdin_tolerance:
        raise RuntimeError("sparse Lowdin square root lost a non-negligible product tail")
    lowdin = _sparse_matmat(square_root, coefficients)
    orbital_adjoint = np.zeros((nroot, nmo), dtype=float)
    lowdin_adjoint = np.zeros((nroot, nao, nmo), dtype=float)
    explicit = np.zeros((nroot, len(coordinates), 3), dtype=float)
    numbers = np.asarray(state.atomic_numbers, dtype=np.int64)

    from .sf_tda import GRIMME_XTB_SF_STDA, grimme_stda_atomic_hardnesses

    hardness = grimme_stda_atomic_hardnesses()[numbers - 1]
    screened = GRIMME_XTB_SF_STDA.hardness_scale * 0.5 * (hardness[:, None] + hardness[None, :])
    exponent = GRIMME_XTB_SF_STDA.distance_exponent

    entries = [(i, i, adjoint.diagonal_values[:, i]) for i in range(space.size)]
    entries.extend(
        (int(i), int(j), adjoint.pair_values[:, p])
        for p, (i, j) in enumerate(zip(operator.rows, operator.columns, strict=True))
    )
    for left, right, weights in entries:
        oi, oj = int(space.occupied_orbitals[left]), int(space.occupied_orbitals[right])
        va, vb = int(space.virtual_orbitals[left]), int(space.virtual_orbitals[right])
        if left == right:
            orbital_adjoint[:, va] += weights
            orbital_adjoint[:, oi] -= weights
        atoms = np.union1d(space.support(left), space.support(right))
        occupied_pair = np.empty(len(atoms), dtype=float)
        virtual_pair = np.empty(len(atoms), dtype=float)
        masks: list[np.ndarray] = []
        for position, atom in enumerate(atoms):
            mask = np.flatnonzero(state.ao_atom == atom)
            masks.append(mask)
            occupied_pair[position] = np.dot(lowdin[mask, oi], lowdin[mask, oj])
            virtual_pair[position] = np.dot(lowdin[mask, va], lowdin[mask, vb])
        displacement = coordinates[atoms][:, None, :] - coordinates[atoms][None, :, :]
        distance = np.linalg.norm(displacement, axis=2)
        local_screened = screened[np.ix_(atoms, atoms)]
        gamma = 1.0 / (distance**exponent + local_screened ** (-exponent)) ** (1.0 / exponent)
        occupied_factor = -(gamma @ virtual_pair)
        virtual_factor = -(gamma.T @ occupied_pair)
        for atom_position, mask in enumerate(masks):
            for root, weight in enumerate(weights):
                factor = weight * occupied_factor[atom_position]
                lowdin_adjoint[root, mask, oi] += factor * lowdin[mask, oj]
                lowdin_adjoint[root, mask, oj] += factor * lowdin[mask, oi]
                factor = weight * virtual_factor[atom_position]
                lowdin_adjoint[root, mask, va] += factor * lowdin[mask, vb]
                lowdin_adjoint[root, mask, vb] += factor * lowdin[mask, va]
        for a in range(len(atoms)):
            for b in range(a):
                radius = distance[a, b]
                if radius <= 0.0:
                    continue
                # Both ordered gamma entries contribute to the symmetric kernel.
                gamma_weight = -(
                    occupied_pair[a] * virtual_pair[b] + occupied_pair[b] * virtual_pair[a]
                )
                radial = gamma[a, b] ** (exponent + 1.0) * radius ** (exponent - 2.0)
                vector = radial * displacement[a, b]
                atom_a, atom_b = int(atoms[a]), int(atoms[b])
                for root, weight in enumerate(weights):
                    force = -weight * gamma_weight * vector
                    explicit[root, atom_a] += force
                    explicit[root, atom_b] -= force

    density_values = np.einsum(
        "ri,pi,pi->rp",
        orbital_adjoint,
        coefficients[rows],
        coefficients[columns],
        optimize=True,
    )
    coefficient_adjoint = np.empty_like(lowdin_adjoint)
    weighted_values = np.empty_like(density_values)
    lowdin_residuals = np.empty(nroot, dtype=float)
    for root in range(nroot):
        coefficient_adjoint[root] = _sparse_matmat(square_root, lowdin_adjoint[root])
        source_values = 0.5 * (
            np.einsum("pi,pi->p", lowdin_adjoint[root, rows], coefficients[columns])
            + np.einsum("pi,pi->p", lowdin_adjoint[root, columns], coefficients[rows])
        )
        source = SparseSymmetricMatrix(nao, rows, columns, source_values)
        lowdin_reverse = sparse_square_root_adjoint(
            overlap,
            inverse_square_root,
            domains,
            source,
            tolerance=lowdin_tolerance,
        )
        if not lowdin_reverse.converged:
            raise RuntimeError(
                "sparse Lowdin adjoint did not converge: "
                f"residual={lowdin_reverse.residual_norm:.3e}"
            )
        lowdin_residuals[root] = lowdin_reverse.residual_norm
        energy_metric = np.einsum(
            "i,pi,pi->p",
            orbital_adjoint[root] * energies,
            coefficients[rows],
            coefficients[columns],
            optimize=True,
        )
        weighted_values[root] = energy_metric - lowdin_reverse.solution.values_on(rows, columns)

    frozen_density = SparseRootMatrixBatch(nao, rows, columns, density_values)
    # The SCF adjoint is not restricted to the excitation window: rotations
    # through external orbitals contribute even when their TDA amplitudes are
    # exactly zero.  AO contractions remain pair-packed.  A future bounded
    # external-orbital domain may replace this complete list only together
    # with a certified discarded-source bound.
    active = np.arange(nmo, dtype=np.int64)
    local_coefficients = coefficients[:, active]
    left, right = np.triu_indices(len(active), 1)
    source = np.empty((nroot, len(left)), dtype=float)
    metric = np.empty((nroot, len(active), len(active)), dtype=float)
    occupation_difference = occupations[active][right] - occupations[active][left]
    for root in range(nroot):
        kernel = fock_density_response(frozen_density.root(root))
        kernel_mo = sparse_mo_from_ao(kernel, local_coefficients)
        coefficient_mo = coefficient_adjoint[root].T @ local_coefficients
        coefficient_mo = coefficient_mo[np.ix_(active, np.arange(len(active)))]
        source[root] = (
            coefficient_mo[right, left]
            - coefficient_mo[left, right]
            + 2.0 * occupation_difference * kernel_mo[left, right]
        )
        metric[root] = 0.25 * (coefficient_mo + coefficient_mo.T)
        metric[root] += 0.5 * (occupations[active, None] + occupations[active][None, :]) * kernel_mo
    rotation = rotation_policy or LocalizedRotationPolicy()
    full_left, full_right = np.triu_indices(nmo, 1)
    exact_sparse_coefficients = SparseOrbitalCoefficients.from_dense(coefficients, threshold=0.0)
    coefficient_fill = len(exact_sparse_coefficients.values) / max(1, coefficients.size)
    response_coefficients = exact_sparse_coefficients if coefficient_fill <= 0.35 else coefficients
    cutoff = rotation.initial_cutoff_bohr
    while True:
        rotation_space = build_localized_rotation_space(
            lowdin,
            state.ao_atom,
            coordinates,
            cutoff_bohr=cutoff,
            tail_tolerance=rotation.orbital_tail_tolerance,
        )
        keys = full_left.astype(np.int64) * nmo + full_right
        local_keys = rotation_space.left.astype(np.int64) * nmo + rotation_space.right
        local_positions = np.searchsorted(keys, local_keys)
        local_metric = SparseOrbitalMetricBatch(
            metric[:, rotation_space.left, rotation_space.right],
            np.diagonal(metric, axis1=1, axis2=2),
        )
        z_vector = matrix_free_sparse_z_vector(
            response_coefficients,
            energies,
            occupations,
            source[:, local_positions],
            local_metric,
            rows,
            columns,
            fock_density_response,
            active_orbitals=active,
            rotation_space=rotation_space,
            tolerance=z_tolerance,
            maximum_iterations=maximum_z_iterations,
        )
        certificates = certify_localized_z_vector(
            coefficients,
            energies,
            occupations,
            source,
            z_vector,
            rows,
            columns,
            fock_density_response,
            error_tolerance=rotation.error_tolerance,
        )
        omitted = np.ones(len(full_left), dtype=bool)
        omitted[local_positions] = False
        discarded_source = np.linalg.norm(source[:, omitted], axis=1)
        z_vector = replace(
            z_vector,
            discarded_source_norms=discarded_source,
            certificates=certificates,
        )
        if all(item.qualified for item in certificates):
            break
        if cutoff >= rotation.maximum_cutoff_bohr:
            cutoff = float("inf")
        else:
            cutoff = min(rotation.maximum_cutoff_bohr, cutoff * rotation.expansion_factor)
    response_density = SparseRootMatrixBatch(
        nao, rows, columns, density_values + z_vector.response_density.values
    )
    response_weighted = SparseRootMatrixBatch(
        nao, rows, columns, weighted_values + z_vector.response_weighted_density.values
    )
    return SparseLocalizedSFTDALagrangian(
        response_density,
        response_weighted,
        explicit,
        z_vector,
        lowdin_residuals,
    )


def matrix_free_sparse_z_vector(
    coefficients: np.ndarray | SparseOrbitalCoefficients,
    orbital_energies_hartree: np.ndarray,
    occupations: np.ndarray,
    orbital_source: np.ndarray,
    metric_weighted_mo: np.ndarray | SparseOrbitalMetricBatch,
    ao_rows: np.ndarray,
    ao_columns: np.ndarray,
    fock_density_response: Callable[[SparseSymmetricMatrix], SparseSymmetricMatrix],
    *,
    active_orbitals: np.ndarray | None = None,
    rotation_space: LocalizedOrbitalRotationSpace | None = None,
    tolerance: float = 1.0e-10,
    maximum_iterations: int = 200,
) -> SparseMatrixFreeZVectorResult:
    """Solve a bounded orbital adjoint without materializing any AO matrix."""

    sparse_coefficients = isinstance(coefficients, SparseOrbitalCoefficients)
    if sparse_coefficients:
        coefficient_count = coefficients.nmo
        nao = coefficients.nao
    else:
        coefficients = np.asarray(coefficients, dtype=float)
        coefficient_count = coefficients.shape[1]
        nao = coefficients.shape[0]
    energies = np.asarray(orbital_energies_hartree, dtype=float)
    occupation = np.asarray(occupations, dtype=float)
    active = (
        np.arange(len(energies), dtype=np.int64)
        if active_orbitals is None
        else np.asarray(active_orbitals, dtype=np.int64)
    )
    if sparse_coefficients and not np.array_equal(
        active, np.arange(coefficient_count, dtype=np.int64)
    ):
        raise ValueError("sparse orbital coefficients require the complete active index map")
    local_coefficients = coefficients if sparse_coefficients else coefficients[:, active]
    local_energies = energies[active]
    local_occupations = occupation[active]
    if rotation_space is None:
        left, right = np.triu_indices(len(active), 1)
    else:
        if len(active) != len(energies) or not np.array_equal(
            active, np.arange(len(energies), dtype=np.int64)
        ):
            raise ValueError("localized rotations currently require the complete MO index map")
        left, right = rotation_space.left, rotation_space.right
    gaps = local_energies[right] - local_energies[left]
    if np.any(np.abs(gaps) <= 1.0e-8):
        raise ValueError("localized sparse Z-vector contains a canonical degeneracy")
    source = np.asarray(orbital_source, dtype=float)
    sparse_metric = isinstance(metric_weighted_mo, SparseOrbitalMetricBatch)
    weighted_metric = (
        metric_weighted_mo if sparse_metric else np.asarray(metric_weighted_mo, dtype=float)
    )
    if source.ndim != 2:
        raise ValueError("sparse Z-vector source must be nroot x npair")
    if source.shape[1] != len(left):
        raise ValueError("sparse Z-vector source and active rotation space differ")
    if sparse_metric:
        if weighted_metric.pair_values.shape != (source.shape[0], len(left)) or (
            weighted_metric.diagonal_values.shape != (source.shape[0], len(active))
        ):
            raise ValueError("pair-packed Z-vector metric has incompatible dimensions")
    elif weighted_metric.shape != (source.shape[0], len(active), len(active)):
        raise ValueError("sparse Z-vector metric adjoint has incompatible dimensions")
    occupation_difference = local_occupations[right] - local_occupations[left]

    def response_rotations(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        packed = _sparse_ao_from_rotation_values(
            local_coefficients,
            left,
            right,
            values,
            ao_rows,
            ao_columns,
        )
        response = fock_density_response(packed)
        if not isinstance(response, SparseSymmetricMatrix):
            raise TypeError("sparse density response must return SparseSymmetricMatrix")
        return _sparse_mo_rotation_values(response, local_coefficients, left, right)

    def apply_transpose(vector: np.ndarray) -> np.ndarray:
        scaled = np.asarray(vector, dtype=float) / gaps
        response, _diagonal = response_rotations(0.5 * scaled)
        return np.asarray(vector, dtype=float) - (2.0 * occupation_difference * response)

    roots = len(source)
    z_vectors = np.empty_like(source)
    residuals = np.empty(roots, dtype=float)
    iterations = np.empty(roots, dtype=np.int64)
    density_values = np.empty((roots, len(ao_rows)), dtype=float)
    weighted_values = np.empty_like(density_values)
    energy_sum = local_energies[left] + local_energies[right]
    for root, rhs in enumerate(source):
        z, residual, count = _matrix_free_gmres(
            apply_transpose,
            rhs,
            tolerance=tolerance,
            maximum_iterations=maximum_iterations,
        )
        z_vectors[root] = z
        residuals[root] = residual
        iterations[root] = count
        scaled = z / gaps
        density = _sparse_ao_from_rotation_values(
            local_coefficients, left, right, 0.5 * scaled, ao_rows, ao_columns
        )
        density_values[root] = density.values
        response_off, response_diagonal = response_rotations(0.5 * scaled)
        kernel_off = 0.5 * (local_occupations[left] + local_occupations[right]) * response_off
        kernel_off += 0.25 * scaled * energy_sum
        kernel_diagonal = local_occupations * response_diagonal
        correction = _sparse_ao_from_rotation_values(
            local_coefficients,
            left,
            right,
            kernel_off,
            ao_rows,
            ao_columns,
            diagonal_values=kernel_diagonal,
        )
        if sparse_metric:
            base_values = _sparse_ao_from_rotation_values(
                local_coefficients,
                left,
                right,
                weighted_metric.pair_values[root],
                ao_rows,
                ao_columns,
                diagonal_values=weighted_metric.diagonal_values[root],
            ).values
        elif np.any(weighted_metric[root]):
            base_values = sparse_ao_from_mo(
                local_coefficients,
                weighted_metric[root : root + 1],
                ao_rows,
                ao_columns,
            ).values[0]
        else:
            base_values = np.zeros(len(ao_rows), dtype=float)
        weighted_values[root] = base_values + correction.values
    density = SparseRootMatrixBatch(nao, ao_rows, ao_columns, density_values)
    weighted = SparseRootMatrixBatch(nao, ao_rows, ao_columns, weighted_values)
    return SparseMatrixFreeZVectorResult(
        orbital_source=source.copy(),
        orbital_z_vector=z_vectors,
        response_density=density,
        response_weighted_density=weighted,
        residual_norms=residuals,
        iterations=iterations,
        discarded_source_norms=np.zeros(roots, dtype=float),
        rotation_space=rotation_space,
        coefficient_backend=("csr-native" if sparse_coefficients else "dense"),
    )


__all__ = [
    "SparseOrbitalCoefficients",
    "SparseOrbitalMetricBatch",
    "SparseLocalizedSFTDALagrangian",
    "SparseMatrixFreeZVectorResult",
    "SparseRootMatrixBatch",
    "SparseSFTDAOperatorAdjoint",
    "SparseSylvesterResult",
    "matrix_free_sparse_z_vector",
    "localized_sparse_sf_tda_lagrangian",
    "sparse_ao_from_mo",
    "sparse_mo_from_ao",
    "sparse_sf_tda_operator_adjoint",
    "sparse_square_root_adjoint",
]
