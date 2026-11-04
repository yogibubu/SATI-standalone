"""Spatial domains for linear-scaling MATRIX-GFN2 algebra.

GFN2 already uses atom-centred minimal orbitals.  Linear scaling is obtained by
retaining that locality in the density/Fock algebra and by truncating only with
an auditable error bound.  Pair-natural compression may subsequently reduce a
domain, but it is not used as a replacement for the occupied density matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Sequence as SequenceABC
from itertools import product
from typing import Iterable, Sequence

import numpy as np

try:
    from . import _gfn2_native
except ImportError:  # pragma: no cover
    _gfn2_native = None


# Bound persistent intersections; compact domains use on-the-fly CSR products.
_SPARSE_SP2_PRECOMPUTE_MAX_BYTES = 1024 * 1024 * 1024


@dataclass(frozen=True)
class LocalityPolicy:
    interaction_cutoff_bohr: float = 24.0
    buffer_hops: int = 1
    truncation_threshold: float = 1.0e-10
    minimum_gap_hartree: float = 1.0e-3

    def __post_init__(self) -> None:
        if self.interaction_cutoff_bohr <= 0.0:
            raise ValueError("interaction_cutoff_bohr must be positive")
        if self.buffer_hops < 0:
            raise ValueError("buffer_hops must be nonnegative")
        if self.truncation_threshold <= 0.0:
            raise ValueError("truncation_threshold must be positive")
        if self.minimum_gap_hartree <= 0.0:
            raise ValueError("minimum_gap_hartree must be positive")


@dataclass(frozen=True)
class EnclosingEllipsoid:
    """PCA-oriented ellipsoid enclosing all atoms with an isotropic margin."""

    center_bohr: np.ndarray
    axes_bohr: np.ndarray
    orientation: np.ndarray

    @property
    def aspect_ratio(self) -> float:
        return float(self.axes_bohr[0] / self.axes_bohr[-1])


def enclosing_pca_ellipsoid(
    coordinates_bohr: np.ndarray, *, margin_bohr: float = 2.0
) -> EnclosingEllipsoid:
    """Find conservative principal axes, including for thin sheets and chains."""

    points = np.asarray(coordinates_bohr, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("coordinates must have shape (natoms, 3)")
    if not np.all(np.isfinite(points)) or margin_bohr <= 0.0:
        raise ValueError("coordinates and positive margin must be finite")
    center = 0.5 * (np.min(points, axis=0) + np.max(points, axis=0))
    centered = points - center
    covariance = centered.T @ centered / len(points)
    _, orientation = np.linalg.eigh(covariance)
    orientation = orientation[:, ::-1]
    projected = centered @ orientation
    axes = np.sqrt(3.0) * (np.max(np.abs(projected), axis=0) + margin_bohr)
    order = np.argsort(axes)[::-1]
    return EnclosingEllipsoid(center, axes[order], orientation[:, order])


class ArrayBackedPairs(SequenceABC):
    """Sequence view over native pair arrays without Python tuple materialization."""

    def __init__(self, left: np.ndarray, right: np.ndarray) -> None:
        self.left = left
        self.right = right

    def __len__(self) -> int:
        return len(self.left)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(zip(self.left[index].tolist(), self.right[index].tolist()))
        return int(self.left[index]), int(self.right[index])

    def __iter__(self):
        return ((int(left), int(right)) for left, right in zip(self.left, self.right))


@dataclass(frozen=True)
class LocalizedOrbitalDomains:
    atom_neighbors: tuple[tuple[int, ...], ...]
    orbital_atom: tuple[int, ...]
    allowed_pairs: Sequence[tuple[int, int]]
    _pair_left: np.ndarray | None = field(default=None, repr=False, compare=False)
    _pair_right: np.ndarray | None = field(default=None, repr=False, compare=False)
    _native_topology: tuple[np.ndarray, np.ndarray, np.ndarray] | None = field(
        default=None, repr=False, compare=False
    )
    _native_sp2_topology: tuple[np.ndarray, ...] | None = field(
        default=None, repr=False, compare=False
    )
    native_threads: int = field(default=1, repr=False, compare=False)

    @property
    def natoms(self) -> int:
        return len(self.atom_neighbors)

    @property
    def norbitals(self) -> int:
        return len(self.orbital_atom)

    @property
    def fill_fraction(self) -> float:
        if not self.orbital_atom:
            return 0.0
        full = len(self.orbital_atom) ** 2
        if self._pair_left is not None and self._pair_right is not None:
            return float(
                (2 * len(self._pair_left)
                 - np.count_nonzero(self._pair_left == self._pair_right)) / full
            )
        off_diagonal = sum(2 for left, right in self.allowed_pairs if left != right)
        diagonal = sum(1 for left, right in self.allowed_pairs if left == right)
        return (off_diagonal + diagonal) / full

    def dense_mask(self) -> np.ndarray:
        mask = np.zeros((self.norbitals, self.norbitals), dtype=bool)
        for left, right in self.allowed_pairs:
            mask[left, right] = True
            mask[right, left] = True
        return mask

    def allowed_pair_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if self._pair_left is not None and self._pair_right is not None:
            return self._pair_left, self._pair_right
        if not self.allowed_pairs:
            empty = np.empty(0, dtype=np.int32)
            return empty, empty
        pairs = np.asarray(self.allowed_pairs, dtype=np.int32)
        return pairs[:, 0], pairs[:, 1]

    def native_topology(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Return a cached contiguous CSR-like adjacency for native kernels."""

        if self._native_topology is not None:
            return self._native_topology
        if _gfn2_native is None or not hasattr(_gfn2_native, "sparse_topology"):
            return None
        left, right = self.allowed_pair_arrays()
        raw = _gfn2_native.sparse_topology(int(self.norbitals), left, right)
        topology = (
            np.asarray(raw[0], dtype=np.int64),
            np.asarray(raw[1], dtype=np.int32),
            np.asarray(raw[2], dtype=np.int32),
        )
        object.__setattr__(self, "_native_topology", topology)
        return topology  # type: ignore[return-value]

    def native_sp2_topology(self) -> tuple[np.ndarray, ...] | None:
        """Return cached CSR rows and, when small enough, SP2 intersections."""

        if self._native_sp2_topology is not None:
            return self._native_sp2_topology
        if _gfn2_native is None or not hasattr(_gfn2_native, "sparse_topology"):
            return None
        left, right = self.allowed_pair_arrays()
        degrees = np.bincount(left, minlength=self.norbitals).astype(
            np.int64, copy=False
        )
        off_diagonal = left != right
        degrees += np.bincount(
            right[off_diagonal], minlength=self.norbitals
        ).astype(np.int64, copy=False)
        # min(deg(i), deg(j)) bounds the row intersection for each allowed
        # pair.  The two packed index arrays use 8 bytes per intersection
        # together, before accounting for the native call's temporary copies.
        intersection_upper_bound = int(
            np.minimum(degrees[left], degrees[right]).sum(dtype=np.int64)
        )
        include_intersections = (
            (self.norbitals < 128 or self.fill_fraction < 0.5)
            and 8 * intersection_upper_bound <= _SPARSE_SP2_PRECOMPUTE_MAX_BYTES
        )
        raw = _gfn2_native.sparse_topology(
            int(self.norbitals), left, right, include_intersections
        )
        topology = tuple(
            np.asarray(item, dtype=np.int64 if index in (0, 3) else np.int32)
            for index, item in enumerate(raw)
        )
        object.__setattr__(self, "_native_sp2_topology", topology)
        return topology


def _cell_key(point: np.ndarray, width: float) -> tuple[int, int, int]:
    return tuple(int(value) for value in np.floor(point / width))


def _spatial_neighbors(coordinates: np.ndarray, cutoff: float) -> list[set[int]]:
    cells: dict[tuple[int, int, int], list[int]] = {}
    for index, point in enumerate(coordinates):
        cells.setdefault(_cell_key(point, cutoff), []).append(index)
    neighbors = [{index} for index in range(len(coordinates))]
    offsets = tuple(product((-1, 0, 1), repeat=3))
    for key, members in cells.items():
        for offset in offsets:
            adjacent = tuple(key[axis] + offset[axis] for axis in range(3))
            for left in members:
                for right in cells.get(adjacent, ()):
                    if right <= left:
                        continue
                    if np.linalg.norm(coordinates[left] - coordinates[right]) <= cutoff:
                        neighbors[left].add(right)
                        neighbors[right].add(left)
    return neighbors


def _expand_neighbors(neighbors: list[set[int]], hops: int) -> list[set[int]]:
    expanded = [set(items) for items in neighbors]
    for _ in range(hops):
        expanded = [
            set().union(*(neighbors[item] for item in current))
            for current in expanded
        ]
    return expanded


def build_localized_domains(
    coordinates_bohr: np.ndarray,
    orbital_atom: Sequence[int],
    policy: LocalityPolicy = LocalityPolicy(),
) -> LocalizedOrbitalDomains:
    coordinates = np.asarray(coordinates_bohr, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates_bohr must have shape natoms x 3")
    assignment = tuple(int(value) for value in orbital_atom)
    if any(value < 0 or value >= len(coordinates) for value in assignment):
        raise ValueError("orbital_atom contains an invalid atom index")
    if _gfn2_native is not None:
        offsets, neighbor_indices, pair_left, pair_right = _gfn2_native.localized_domains(
            coordinates,
            np.asarray(assignment, dtype=np.int64),
            float(policy.interaction_cutoff_bohr),
            int(policy.buffer_hops),
        )
        neighbors_tuple = tuple(
            tuple(int(value) for value in neighbor_indices[offsets[atom]:offsets[atom + 1]])
            for atom in range(len(coordinates))
        )
        pair_left = np.asarray(pair_left, dtype=np.int32)
        pair_right = np.asarray(pair_right, dtype=np.int32)
        return LocalizedOrbitalDomains(
            neighbors_tuple,
            assignment,
            ArrayBackedPairs(pair_left, pair_right),
            pair_left,
            pair_right,
        )
    neighbors = _expand_neighbors(
        _spatial_neighbors(coordinates, policy.interaction_cutoff_bohr),
        policy.buffer_hops,
    )
    pairs = []
    for left, left_atom in enumerate(assignment):
        for right in range(left + 1):
            if assignment[right] in neighbors[left_atom]:
                pairs.append((left, right))
    return LocalizedOrbitalDomains(
        atom_neighbors=tuple(tuple(sorted(items)) for items in neighbors),
        orbital_atom=assignment,
        allowed_pairs=tuple(pairs),
    )


def localized_domains_from_atom_neighbors(
    atom_neighbors: Sequence[Sequence[int]], orbital_atom: Sequence[int],
) -> LocalizedOrbitalDomains:
    """Build matching AO pair sets after residual-guided atom-pair growth."""
    neighbors = tuple(tuple(sorted(set(int(value) for value in row)))
                      for row in atom_neighbors)
    natom = len(neighbors)
    assignment = tuple(int(value) for value in orbital_atom)
    if (not neighbors or any(atom not in row for atom, row in enumerate(neighbors))
            or any(value < 0 or value >= natom for value in assignment)
            or any(other < 0 or other >= natom or atom not in neighbors[other]
                   for atom, row in enumerate(neighbors) for other in row)):
        raise ValueError("invalid symmetric atom-neighbor domains")
    rows = []
    columns = []
    for left, left_atom in enumerate(assignment):
        admitted = neighbors[left_atom]
        for right in range(left + 1):
            if assignment[right] in admitted:
                rows.append(left)
                columns.append(right)
    pair_left = np.asarray(rows, dtype=np.int32)
    pair_right = np.asarray(columns, dtype=np.int32)
    return LocalizedOrbitalDomains(
        neighbors, assignment, ArrayBackedPairs(pair_left, pair_right),
        pair_left, pair_right,
    )


def truncate_symmetric_matrix(
    matrix: np.ndarray,
    domains: LocalizedOrbitalDomains,
    *,
    threshold: float,
) -> tuple[np.ndarray, float]:
    """Apply a domain/absolute cutoff and return its Frobenius error bound."""

    values = np.asarray(matrix, dtype=float)
    if values.shape != (domains.norbitals, domains.norbitals):
        raise ValueError("matrix shape does not match localized orbital domains")
    if not np.allclose(values, values.T, atol=1.0e-13):
        raise ValueError("localized truncation requires a symmetric matrix")
    keep = domains.dense_mask() & (np.abs(values) >= float(threshold))
    keep[np.diag_indices_from(keep)] = True
    result = np.where(keep, values, 0.0)
    discarded = values - result
    return result, float(np.linalg.norm(discarded))


def domain_indices(
    domains: LocalizedOrbitalDomains, atoms: Iterable[int]
) -> np.ndarray:
    selected_atoms = set(int(atom) for atom in atoms)
    expanded = set().union(*(domains.atom_neighbors[atom] for atom in selected_atoms))
    return np.asarray(
        [index for index, atom in enumerate(domains.orbital_atom) if atom in expanded],
        dtype=np.intp,
    )


__all__ = [
    "LocalizedOrbitalDomains",
    "LocalityPolicy",
    "build_localized_domains",
    "localized_domains_from_atom_neighbors",
    "domain_indices",
    "truncate_symmetric_matrix",
]
