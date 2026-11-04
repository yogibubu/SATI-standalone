"""Localized sparse excitation spaces for resident spin-flip TDA."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from matrix_numerics import solve_symmetric_matrix_free

from .electronic_state import GFN2OrbitalState
from .sf_tda import (
    GRIMME_XTB_SF_STDA,
    SFTDAResult,
    SpinFlipExcitationSpace,
    SFTDASelectionPolicy,
    grimme_stda_atomic_hardnesses,
)


@dataclass(frozen=True)
class LocalizedSFTDAPolicy:
    """Auditable locality policy for an extensive SF configuration graph."""

    configuration_population_threshold: float = 1.0e-5
    support_tail_tolerance: float = 1.0e-8
    support_radius_bohr: float = 6.0
    center_cutoff_bohr: float = 18.0
    maximum_configurations_per_atom: int = 24
    truncation_error_tolerance_hartree: float = 1.0e-6

    def __post_init__(self) -> None:
        if not 0.0 <= self.configuration_population_threshold < 1.0:
            raise ValueError("localized SF population threshold must lie in [0,1)")
        if not 0.0 <= self.support_tail_tolerance < 1.0:
            raise ValueError("localized SF support-tail tolerance must lie in [0,1)")
        if self.support_radius_bohr <= 0.0:
            raise ValueError("localized SF support radius must be positive")
        if self.center_cutoff_bohr <= 2.0 * self.support_radius_bohr:
            raise ValueError(
                "localized SF center cutoff must exceed twice the support radius"
            )
        if self.maximum_configurations_per_atom < 1:
            raise ValueError("localized SF per-atom configuration cap must be positive")
        if self.truncation_error_tolerance_hartree <= 0.0:
            raise ValueError("localized SF error tolerance must be positive")


@dataclass(frozen=True)
class LocalizedSpinFlipSpace:
    """Nonrectangular configurations and bounded atom supports."""

    parent: SpinFlipExcitationSpace | None
    parent_indices: np.ndarray
    occupied_orbitals: np.ndarray
    virtual_orbitals: np.ndarray
    orbital_gaps_hartree: np.ndarray
    centers_bohr: np.ndarray
    support_offsets: np.ndarray
    support_atoms: np.ndarray
    discarded_population_bound: float

    def __post_init__(self) -> None:
        indices = np.asarray(self.parent_indices, dtype=np.int64)
        occupied = np.asarray(self.occupied_orbitals, dtype=np.int64)
        virtual = np.asarray(self.virtual_orbitals, dtype=np.int64)
        gaps = np.asarray(self.orbital_gaps_hartree, dtype=float)
        centers = np.asarray(self.centers_bohr, dtype=float)
        offsets = np.asarray(self.support_offsets, dtype=np.int64)
        atoms = np.asarray(self.support_atoms, dtype=np.int64)
        count = len(indices)
        if (
            count < 1
            or occupied.shape != (count,)
            or virtual.shape != (count,)
            or gaps.shape != (count,)
        ):
            raise ValueError("localized SF configuration arrays are inconsistent")
        if centers.shape != (count, 3) or offsets.shape != (count + 1,):
            raise ValueError("localized SF centers or support offsets are inconsistent")
        if offsets[0] != 0 or offsets[-1] != len(atoms) or np.any(np.diff(offsets) < 1):
            raise ValueError("every localized SF configuration needs atom support")
        if self.parent is not None and (
            np.any(indices < 0) or np.any(indices >= self.parent.size)
        ):
            raise ValueError("localized SF configuration lies outside the parent space")
        if self.parent is None and np.any(indices != -1):
            raise ValueError("direct localized configurations have no parent indices")
        if not np.all(np.isfinite(centers)) or np.any(~np.isfinite(gaps)):
            raise ValueError("localized SF diagnostics must be finite")
        if np.isnan(self.discarded_population_bound) or self.discarded_population_bound < 0.0:
            raise ValueError("localized SF discarded bound must be nonnegative")
        for array in (indices, occupied, virtual, gaps, centers, offsets, atoms):
            array.setflags(write=False)
        object.__setattr__(self, "parent_indices", indices)
        object.__setattr__(self, "occupied_orbitals", occupied)
        object.__setattr__(self, "virtual_orbitals", virtual)
        object.__setattr__(self, "orbital_gaps_hartree", gaps)
        object.__setattr__(self, "centers_bohr", centers)
        object.__setattr__(self, "support_offsets", offsets)
        object.__setattr__(self, "support_atoms", atoms)

    @property
    def size(self) -> int:
        return len(self.parent_indices)

    def support(self, configuration: int) -> np.ndarray:
        start, stop = self.support_offsets[configuration : configuration + 2]
        return self.support_atoms[start:stop]


@dataclass(frozen=True)
class SparseSFTDAOperator:
    """Symmetric pair-packed SF Hamiltonian on a localized graph."""

    space: LocalizedSpinFlipSpace
    rows: np.ndarray
    columns: np.ndarray
    values_hartree: np.ndarray
    diagonal_hartree: np.ndarray
    omitted_coupling_bound_hartree: float

    def __post_init__(self) -> None:
        rows = np.asarray(self.rows, dtype=np.int64)
        columns = np.asarray(self.columns, dtype=np.int64)
        values = np.asarray(self.values_hartree, dtype=float)
        diagonal = np.asarray(self.diagonal_hartree, dtype=float)
        if rows.shape != columns.shape or values.shape != rows.shape:
            raise ValueError("sparse SF operator pair arrays are inconsistent")
        if diagonal.shape != (self.space.size,):
            raise ValueError("sparse SF operator diagonal has the wrong size")
        if np.any(rows <= columns) or np.any(rows >= self.space.size) or np.any(columns < 0):
            raise ValueError("sparse SF operator stores strict lower-triangle pairs")
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(diagonal)):
            raise ValueError("sparse SF operator must be finite")
        for array in (rows, columns, values, diagonal):
            array.setflags(write=False)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "values_hartree", values)
        object.__setattr__(self, "diagonal_hartree", diagonal)

    @property
    def size(self) -> int:
        return self.space.size

    def apply(self, vector: np.ndarray) -> np.ndarray:
        operand = np.asarray(vector, dtype=float)
        if operand.shape != (self.size,):
            raise ValueError("localized SF vector has the wrong size")
        result = self.diagonal_hartree * operand
        np.add.at(result, self.rows, self.values_hartree * operand[self.columns])
        np.add.at(result, self.columns, self.values_hartree * operand[self.rows])
        return result

    def apply_block(self, vectors: np.ndarray) -> np.ndarray:
        operands = np.asarray(vectors, dtype=float)
        if operands.ndim != 2 or operands.shape[0] != self.size:
            raise ValueError("localized SF vector block has the wrong size")
        result = self.diagonal_hartree[:, None] * operands
        for column in range(operands.shape[1]):
            np.add.at(
                result[:, column],
                self.rows,
                self.values_hartree * operands[self.columns, column],
            )
            np.add.at(
                result[:, column],
                self.columns,
                self.values_hartree * operands[self.rows, column],
            )
        return result


@dataclass(frozen=True)
class LocalizedSFTDAResult:
    result: SFTDAResult
    space: LocalizedSpinFlipSpace
    operator: SparseSFTDAOperator
    operator_nonzero_count: int
    omitted_coupling_bound_hartree: float
    qualified: bool


def _lowdin_coefficients(state: GFN2OrbitalState) -> np.ndarray:
    values, vectors = np.linalg.eigh(state.overlap)
    if np.min(values) <= 0.0:
        raise ValueError("localized SF-TDA requires a positive-definite overlap")
    return (vectors * np.sqrt(values)) @ vectors.T @ state.coefficients


def build_localized_spin_flip_space(
    state: GFN2OrbitalState,
    parent: SpinFlipExcitationSpace,
    policy: LocalizedSFTDAPolicy,
) -> LocalizedSpinFlipSpace:
    """Select configurations by atom-resolved transition population."""

    lowdin = _lowdin_coefficients(state)
    occupied, virtual = parent.configuration_orbitals()
    atom_count = len(state.atomic_numbers)
    configuration_count = parent.size
    weights = np.empty((configuration_count, atom_count), dtype=float)
    for atom in range(atom_count):
        mask = state.ao_atom == atom
        products = (
            lowdin[mask][:, occupied] * lowdin[mask][:, virtual]
        )
        weights[:, atom] = np.sum(np.abs(products), axis=0)
    totals = np.sum(weights, axis=1)
    valid = totals > np.finfo(float).tiny
    normalized = np.zeros_like(weights)
    normalized[valid] = weights[valid] / totals[valid, None]
    centers = normalized @ state.coordinates_bohr
    dominant = np.argmax(normalized, axis=1)
    keep = valid & (
        np.max(normalized, axis=1) >= policy.configuration_population_threshold
    )
    retained: list[int] = []
    for atom in range(atom_count):
        candidates = np.flatnonzero(keep & (dominant == atom))
        order = sorted(
            candidates,
            key=lambda index: (
                float(parent.diagonal_hartree[index]),
                -float(normalized[index, atom]),
                int(index),
            ),
        )
        retained.extend(order[: policy.maximum_configurations_per_atom])
    indices = np.asarray(sorted(set(retained)), dtype=np.int64)
    if not len(indices):
        raise ValueError("localized SF policy removed every configuration")
    support_parts: list[np.ndarray] = []
    discarded = 0.0
    for index in indices:
        order = np.argsort(-normalized[index], kind="stable")
        cumulative = np.cumsum(normalized[index, order])
        count = int(np.searchsorted(cumulative, 1.0 - policy.support_tail_tolerance) + 1)
        candidate = order[: max(1, count)]
        spatial = np.linalg.norm(
            state.coordinates_bohr[candidate] - centers[index], axis=1
        ) <= policy.support_radius_bohr
        support = np.sort(candidate[spatial])
        if not len(support):
            support = np.asarray((int(order[0]),), dtype=np.int64)
        support_parts.append(support)
        discarded += max(0.0, 1.0 - float(np.sum(normalized[index, support])))
    offsets = np.concatenate(
        (np.asarray((0,), dtype=np.int64), np.cumsum([len(x) for x in support_parts]))
    )
    return LocalizedSpinFlipSpace(
        parent,
        indices,
        occupied[indices],
        virtual[indices],
        parent.diagonal_hartree[indices],
        centers[indices],
        offsets,
        np.concatenate(support_parts),
        float(discarded),
    )


def build_extensive_localized_spin_flip_space(
    state: GFN2OrbitalState,
    selection: SFTDASelectionPolicy,
    policy: LocalizedSFTDAPolicy,
    *,
    occupation_tolerance: float = 1.0e-7,
) -> LocalizedSpinFlipSpace:
    """Build bounded local configurations without an occupied x virtual array."""

    energies = np.asarray(state.orbital_energies_hartree)
    occupied = np.flatnonzero(state.alpha_occupations >= 1.0 - occupation_tolerance)
    virtual = np.flatnonzero(state.beta_occupations <= occupation_tolerance)
    if not len(occupied) or not len(virtual):
        raise ValueError("localized SF-TDA needs occupied-alpha and virtual-beta orbitals")
    alpha_occupied = np.flatnonzero(state.alpha_occupations >= 1.0 - occupation_tolerance)
    beta_occupied = np.flatnonzero(state.beta_occupations >= 1.0 - occupation_tolerance)
    alpha_virtual = np.flatnonzero(state.alpha_occupations <= occupation_tolerance)
    occupied_all = np.concatenate((alpha_occupied, beta_occupied))
    virtual_all = np.concatenate((alpha_virtual, virtual))
    half_width = 2.0 * (1.0 + 0.8 * GRIMME_XTB_SF_STDA.exact_exchange_fraction) * (
        selection.spectral_range_hartree
    )
    occupied = occupied[
        energies[occupied] > float(np.min(energies[virtual_all])) - half_width
    ]
    virtual = virtual[
        energies[virtual] < float(np.max(energies[occupied_all])) + half_width
    ]
    if not len(occupied) or not len(virtual):
        raise ValueError("no localized configurations survive the Grimme window")
    lowdin = _lowdin_coefficients(state)
    atom_count = len(state.atomic_numbers)
    orbital_population = np.zeros((state.nmo, atom_count), dtype=float)
    for atom in range(atom_count):
        mask = state.ao_atom == atom
        orbital_population[:, atom] = np.sum(lowdin[mask] ** 2, axis=0)
    normalization = np.sum(orbital_population, axis=1)
    orbital_population /= np.where(normalization > 0.0, normalization, 1.0)[:, None]
    pairs: set[tuple[int, int]] = set()
    cap_hit = False
    for atom in range(atom_count):
        local_occupied = occupied[
            orbital_population[occupied, atom] >= policy.configuration_population_threshold
        ]
        local_virtual = virtual[
            orbital_population[virtual, atom] >= policy.configuration_population_threshold
        ]
        candidates = [
            (int(i), int(a)) for i in local_occupied for a in local_virtual
        ]
        candidates.sort(
            key=lambda pair: (
                float(energies[pair[1]] - energies[pair[0]]),
                -float(
                    orbital_population[pair[0], atom]
                    * orbital_population[pair[1], atom]
                ),
                pair,
            )
        )
        cap_hit |= len(candidates) > policy.maximum_configurations_per_atom
        pairs.update(candidates[: policy.maximum_configurations_per_atom])
    ordered = sorted(pairs, key=lambda pair: (energies[pair[1]] - energies[pair[0]], pair))
    if not ordered:
        raise ValueError("localized SF policy removed every configuration")
    occupied_config = np.asarray([pair[0] for pair in ordered], dtype=np.int64)
    virtual_config = np.asarray([pair[1] for pair in ordered], dtype=np.int64)
    centers = np.empty((len(ordered), 3), dtype=float)
    supports: list[np.ndarray] = []
    discarded = 0.0
    for index, (i, a) in enumerate(ordered):
        weights = np.zeros(atom_count, dtype=float)
        for atom in range(atom_count):
            mask = state.ao_atom == atom
            weights[atom] = np.sum(np.abs(lowdin[mask, i] * lowdin[mask, a]))
        if np.sum(weights) <= np.finfo(float).tiny:
            weights = np.sqrt(orbital_population[i] * orbital_population[a])
        weights /= np.sum(weights)
        centers[index] = weights @ state.coordinates_bohr
        order = np.argsort(-weights, kind="stable")
        count = int(
            np.searchsorted(np.cumsum(weights[order]), 1.0 - policy.support_tail_tolerance)
            + 1
        )
        candidate = order[: max(1, count)]
        spatial = np.linalg.norm(
            state.coordinates_bohr[candidate] - centers[index], axis=1
        ) <= policy.support_radius_bohr
        support = np.sort(candidate[spatial])
        if not len(support):
            support = np.asarray((int(order[0]),), dtype=np.int64)
        supports.append(support)
        discarded += max(0.0, 1.0 - float(np.sum(weights[support])))
    complete = len(ordered) == len(occupied) * len(virtual) and not cap_hit
    if not complete:
        discarded = float("inf")
    offsets = np.concatenate(
        (np.asarray((0,), dtype=np.int64), np.cumsum([len(x) for x in supports]))
    )
    return LocalizedSpinFlipSpace(
        None,
        np.full(len(ordered), -1, dtype=np.int64),
        occupied_config,
        virtual_config,
        energies[virtual_config] - energies[occupied_config],
        centers,
        offsets,
        np.concatenate(supports),
        discarded,
    )


def build_sparse_sf_tda_operator(
    state: GFN2OrbitalState,
    space: LocalizedSpinFlipSpace,
    policy: LocalizedSFTDAPolicy,
) -> SparseSFTDAOperator:
    """Assemble only spatially adjacent localized configuration couplings."""

    lowdin = _lowdin_coefficients(state)
    numbers = np.asarray(state.atomic_numbers, dtype=np.int64)
    hardness = grimme_stda_atomic_hardnesses()[numbers - 1]
    screened = GRIMME_XTB_SF_STDA.hardness_scale * 0.5 * (
        hardness[:, None] + hardness[None, :]
    )
    exponent = GRIMME_XTB_SF_STDA.distance_exponent

    def gamma_block(atoms: np.ndarray) -> np.ndarray:
        coordinates = state.coordinates_bohr[atoms]
        displacement = coordinates[:, None, :] - coordinates[None, :, :]
        distance = np.linalg.norm(displacement, axis=2)
        local_screened = screened[np.ix_(atoms, atoms)]
        return 1.0 / (
            distance**exponent + local_screened ** (-exponent)
        ) ** (1.0 / exponent)

    def coupling(left: int, right: int) -> float:
        left_support = space.support(left)
        right_support = space.support(right)
        support = np.union1d(left_support, right_support)
        oi, oj = space.occupied_orbitals[left], space.occupied_orbitals[right]
        va, vb = space.virtual_orbitals[left], space.virtual_orbitals[right]
        occupied_pair = np.empty(len(support), dtype=float)
        virtual_pair = np.empty(len(support), dtype=float)
        for position, atom in enumerate(support):
            mask = state.ao_atom == atom
            occupied_pair[position] = np.dot(lowdin[mask, oi], lowdin[mask, oj])
            virtual_pair[position] = np.dot(lowdin[mask, va], lowdin[mask, vb])
        return -float(
            occupied_pair @ gamma_block(support) @ virtual_pair
        )

    diagonal = np.empty(space.size, dtype=float)
    for index in range(space.size):
        diagonal[index] = space.orbital_gaps_hartree[index] + coupling(index, index)
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    cells: dict[tuple[int, int, int], list[int]] = {}
    width = policy.center_cutoff_bohr
    for index, center in enumerate(space.centers_bohr):
        key = tuple(int(value) for value in np.floor(center / width))
        cells.setdefault(key, []).append(index)
    offsets = tuple(product((-1, 0, 1), repeat=3))
    visited: set[tuple[int, int]] = set()
    neighbor_counts = np.zeros(space.size, dtype=np.int64)
    for key, members in cells.items():
        for offset in offsets:
            adjacent = tuple(key[axis] + offset[axis] for axis in range(3))
            for row in members:
                for column in cells.get(adjacent, ()):
                    if row <= column or (row, column) in visited:
                        continue
                    visited.add((row, column))
                    radius = np.linalg.norm(
                        space.centers_bohr[row] - space.centers_bohr[column]
                    )
                    if radius > policy.center_cutoff_bohr:
                        continue
                    value = coupling(row, column)
                    neighbor_counts[row] += 1
                    neighbor_counts[column] += 1
                    if value != 0.0:
                        rows.append(row)
                        columns.append(column)
                        values.append(value)
    separation = policy.center_cutoff_bohr - 2.0 * policy.support_radius_bohr
    omitted_per_row = np.maximum(0, space.size - 1 - neighbor_counts)
    omitted_bound = float(np.max(omitted_per_row, initial=0)) / separation
    omitted_bound += space.discarded_population_bound * float(
        np.max(np.abs(diagonal))
    )
    return SparseSFTDAOperator(
        space,
        np.asarray(rows, dtype=np.int64),
        np.asarray(columns, dtype=np.int64),
        np.asarray(values, dtype=float),
        diagonal,
        float(omitted_bound),
    )


def localized_sf_tda(
    state: GFN2OrbitalState,
    parent: SpinFlipExcitationSpace | None = None,
    *,
    roots: int,
    policy: LocalizedSFTDAPolicy,
    tolerance_hartree: float = 1.0e-8,
    maximum_iterations: int = 100,
    maximum_subspace: int | None = None,
    backend: str = "auto",
    selection_policy: SFTDASelectionPolicy | None = None,
) -> LocalizedSFTDAResult:
    """Solve a bounded localized SF graph without a global CSF rectangle."""

    space = (
        build_localized_spin_flip_space(state, parent, policy)
        if parent is not None
        else build_extensive_localized_spin_flip_space(
            state, selection_policy or SFTDASelectionPolicy(), policy
        )
    )
    operator = build_sparse_sf_tda_operator(state, space, policy)
    solved = solve_symmetric_matrix_free(
        operator.apply,
        operator.diagonal_hartree,
        roots=min(int(roots), operator.size),
        block_matvec=operator.apply_block,
        backend=backend,
        tolerance=tolerance_hartree,
        maximum_iterations=maximum_iterations,
        maximum_subspace=maximum_subspace,
    )
    result = SFTDAResult(
        solved.eigenvalues,
        solved.eigenvectors,
        solved.residual_norms,
        solved.operator_batches,
        solved.converged,
        solver_backend=solved.backend,
    )
    bound = float(operator.omitted_coupling_bound_hartree)
    return LocalizedSFTDAResult(
        result,
        space,
        operator,
        int(2 * len(operator.values_hartree) + operator.size),
        bound,
        bool(solved.converged and bound <= policy.truncation_error_tolerance_hartree),
    )


__all__ = [
    "LocalizedSFTDAPolicy",
    "LocalizedSFTDAResult",
    "LocalizedSpinFlipSpace",
    "SparseSFTDAOperator",
    "build_localized_spin_flip_space",
    "build_extensive_localized_spin_flip_space",
    "build_sparse_sf_tda_operator",
    "localized_sf_tda",
]
