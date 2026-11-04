"""Linear-scaling closed-shell sTDA from sparse SCF projectors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _gfn2_native
from .electronic_state import GFN2SparseProjectorState
from .sf_tda import grimme_stda_atomic_hardnesses


@dataclass(frozen=True)
class LocalProjectedOrbitals:
    size: int
    offsets: np.ndarray
    rows: np.ndarray
    values: np.ndarray
    centers_bohr: np.ndarray
    atom_populations: np.ndarray
    discarded_projector_bound: float

    @property
    def count(self) -> int:
        return len(self.offsets) - 1

    def to_dense(self) -> np.ndarray:
        result = np.zeros((self.size, self.count))
        for orbital in range(self.count):
            start, stop = self.offsets[orbital : orbital + 2]
            result[self.rows[start:stop], orbital] = self.values[start:stop]
        return result


@dataclass(frozen=True)
class SparseExcitationBasis:
    occupied: LocalProjectedOrbitals
    virtual: LocalProjectedOrbitals
    inherited_projector_bound: float


@dataclass(frozen=True)
class SparseExcitationSpace:
    occupied_orbitals: np.ndarray
    virtual_orbitals: np.ndarray
    diagonal_gaps_hartree: np.ndarray
    centers_bohr: np.ndarray
    dominant_atoms: np.ndarray
    omitted_excitation_count: int

    @property
    def size(self) -> int:
        return len(self.occupied_orbitals)


@dataclass(frozen=True)
class PairPackedSTDAOperator:
    diagonal_hartree: np.ndarray
    rows: np.ndarray
    columns: np.ndarray
    values_hartree: np.ndarray
    omitted_coupling_bound_hartree: float

    @property
    def size(self) -> int:
        return len(self.diagonal_hartree)

    @property
    def nonzero_count(self) -> int:
        return self.size + 2 * len(self.values_hartree)

    def apply(self, vector: np.ndarray) -> np.ndarray:
        operand = np.asarray(vector)
        if operand.shape[0] != self.size:
            raise ValueError("sTDA operator vector has the wrong leading dimension")
        result = self.diagonal_hartree * operand
        np.add.at(result, self.rows, self.values_hartree * operand[self.columns])
        np.add.at(result, self.columns, self.values_hartree * operand[self.rows])
        return result


@dataclass(frozen=True)
class LinearScalingSTDAResult:
    excitation_energies_hartree: np.ndarray
    amplitudes: np.ndarray
    residual_norms_hartree: np.ndarray
    excitation_space_size: int
    operator_nonzero_count: int
    omitted_coupling_bound_hartree: float
    omitted_excitation_count: int
    projector_bound: float
    solver_converged: bool
    qualified: bool


def _factor_projector(
    state: GFN2SparseProjectorState,
    *,
    rank: int,
    complement: bool,
    threshold: float,
) -> LocalProjectedOrbitals:
    if threshold < 0.0:
        raise ValueError("local projected-orbital threshold must be nonnegative")
    offsets, rows, values, discarded = _gfn2_native.localized_projected_orbitals(
        int(state.size),
        state.projector_rows,
        state.projector_columns,
        state.projector_values,
        state.domain_rows,
        state.domain_columns,
        int(rank),
        float(threshold),
        bool(complement),
    )
    offsets = np.asarray(offsets, dtype=np.int64)
    rows = np.asarray(rows, dtype=np.int64)
    values = np.asarray(values, dtype=float)
    populations = np.zeros((rank, len(state.atomic_numbers)))
    for orbital in range(rank):
        start, stop = offsets[orbital : orbital + 2]
        np.add.at(
            populations[orbital],
            state.ao_atom[rows[start:stop]],
            values[start:stop] ** 2,
        )
    normalization = np.sum(populations, axis=1)
    populations /= np.where(normalization > 0.0, normalization, 1.0)[:, None]
    centers = populations @ state.coordinates_bohr
    return LocalProjectedOrbitals(
        int(state.size),
        offsets,
        rows,
        values,
        centers,
        populations,
        float(discarded),
    )


def build_sparse_excitation_basis(
    state: GFN2SparseProjectorState,
    *,
    truncation_threshold: float = 1.0e-10,
) -> SparseExcitationBasis:
    """Factor ``P`` and ``I-P`` directly into atom-local sparse orbitals."""

    occupied = _factor_projector(
        state,
        rank=state.occupied_orbitals,
        complement=False,
        threshold=truncation_threshold,
    )
    virtual = _factor_projector(
        state,
        rank=state.size - state.occupied_orbitals,
        complement=True,
        threshold=truncation_threshold,
    )
    return SparseExcitationBasis(
        occupied,
        virtual,
        float(
            state.truncation_bound
            + occupied.discarded_projector_bound
            + virtual.discarded_projector_bound
        ),
    )


def _orbital_energies(
    state: GFN2SparseProjectorState, orbitals: LocalProjectedOrbitals
) -> np.ndarray:
    return np.asarray(
        _gfn2_native.sparse_orbital_quadratic_forms(
            int(state.size),
            state.fock_rows,
            state.fock_columns,
            state.fock_values,
            orbitals.offsets,
            orbitals.rows,
            orbitals.values,
        ),
        dtype=float,
    )


def build_sparse_excitation_space(
    state: GFN2SparseProjectorState,
    basis: SparseExcitationBasis,
    *,
    center_cutoff_bohr: float = 12.0,
    maximum_gap_hartree: float = 2.1,
) -> SparseExcitationSpace:
    """Build an extensive local occupied–virtual graph by cell lists."""

    if center_cutoff_bohr <= 0.0 or maximum_gap_hartree <= 0.0:
        raise ValueError("localized excitation cutoffs must be positive")
    occupied_energy = _orbital_energies(state, basis.occupied)
    virtual_energy = _orbital_energies(state, basis.virtual)
    width = float(center_cutoff_bohr)
    cells: dict[tuple[int, int, int], list[int]] = {}
    for virtual, center in enumerate(basis.virtual.centers_bohr):
        key = tuple(int(value) for value in np.floor(center / width))
        cells.setdefault(key, []).append(virtual)
    offsets = tuple((x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1))
    pairs: list[tuple[int, int, float]] = []
    for occupied, center in enumerate(basis.occupied.centers_bohr):
        key = tuple(int(value) for value in np.floor(center / width))
        candidates: set[int] = set()
        for offset in offsets:
            candidates.update(cells.get(tuple(key[axis] + offset[axis] for axis in range(3)), ()))
        for virtual in sorted(candidates):
            if np.linalg.norm(center - basis.virtual.centers_bohr[virtual]) > width:
                continue
            gap = virtual_energy[virtual] - occupied_energy[occupied]
            if 0.0 < gap <= maximum_gap_hartree:
                pairs.append((occupied, virtual, float(gap)))
    if not pairs:
        raise ValueError("localized policy removed every closed-shell excitation")
    occupied = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
    virtual = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
    gaps = np.asarray([pair[2] for pair in pairs])
    weights = basis.occupied.atom_populations[occupied] * basis.virtual.atom_populations[virtual]
    normalization = np.sum(weights, axis=1)
    normalized = weights / np.where(normalization > 0.0, normalization, 1.0)[:, None]
    centers = normalized @ state.coordinates_bohr
    dominant = np.argmax(normalized, axis=1).astype(np.int64)
    excluded = basis.occupied.count * basis.virtual.count - len(pairs)
    return SparseExcitationSpace(
        occupied,
        virtual,
        gaps,
        centers,
        dominant,
        excluded,
    )


def build_pair_packed_stda_operator(
    state: GFN2SparseProjectorState,
    basis: SparseExcitationBasis,
    space: SparseExcitationSpace,
    *,
    coupling_cutoff_bohr: float = 18.0,
    value_threshold_hartree: float = 1.0e-10,
) -> PairPackedSTDAOperator:
    """Assemble local one-electron, J, and K terms without a global CSF matrix."""

    if coupling_cutoff_bohr <= 0.0 or value_threshold_hartree < 0.0:
        raise ValueError("sparse sTDA coupling cutoffs are invalid")
    hardness = grimme_stda_atomic_hardnesses()[state.atomic_numbers - 1]
    width = float(coupling_cutoff_bohr)
    cells: dict[tuple[int, int, int], list[int]] = {}
    for index, center in enumerate(space.centers_bohr):
        key = tuple(int(value) for value in np.floor(center / width))
        cells.setdefault(key, []).append(index)
    offsets = tuple((x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1))
    rows: list[int] = []
    columns: list[int] = []
    neighbor_count = np.zeros(space.size, dtype=np.int64)
    for key, members in cells.items():
        for offset in offsets:
            adjacent = tuple(key[axis] + offset[axis] for axis in range(3))
            for row in members:
                for column in cells.get(adjacent, ()):
                    if row <= column:
                        continue
                    if np.linalg.norm(space.centers_bohr[row] - space.centers_bohr[column]) > width:
                        continue
                    neighbor_count[row] += 1
                    neighbor_count[column] += 1
                    rows.append(row)
                    columns.append(column)
    edge_rows = np.asarray(rows, dtype=np.int64)
    edge_columns = np.asarray(columns, dtype=np.int64)
    diagonal, edge_values = _gfn2_native.localized_stda_pairs(
        {
            "ao_atom": state.ao_atom,
            "atomic_numbers": state.atomic_numbers,
            "coordinates": state.coordinates_bohr,
            "hardness": hardness,
            "fock_rows": state.fock_rows,
            "fock_columns": state.fock_columns,
            "fock_values": state.fock_values,
            "occupied_offsets": basis.occupied.offsets,
            "occupied_rows": basis.occupied.rows,
            "occupied_values": basis.occupied.values,
            "virtual_offsets": basis.virtual.offsets,
            "virtual_rows": basis.virtual.rows,
            "virtual_values": basis.virtual.values,
            "configuration_occupied": space.occupied_orbitals,
            "configuration_virtual": space.virtual_orbitals,
            "edge_rows": edge_rows,
            "edge_columns": edge_columns,
            "gaps": space.diagonal_gaps_hartree,
            "value_threshold": np.asarray(value_threshold_hartree),
        }
    )
    edge_values = np.asarray(edge_values)
    keep = edge_values != 0.0
    omitted = np.maximum(0, space.size - 1 - neighbor_count)
    occupied_l1 = np.zeros(basis.occupied.count)
    virtual_l1 = np.zeros(basis.virtual.count)
    np.add.at(
        occupied_l1,
        np.repeat(np.arange(basis.occupied.count), np.diff(basis.occupied.offsets)),
        np.abs(basis.occupied.values),
    )
    np.add.at(
        virtual_l1,
        np.repeat(np.arange(basis.virtual.count), np.diff(basis.virtual.offsets)),
        np.abs(basis.virtual.values),
    )
    maximum_transition_l1 = float(
        np.max(
            occupied_l1[space.occupied_orbitals] * virtual_l1[space.virtual_orbitals],
            initial=0.0,
        )
    )
    maximum_occupied_pair_l1 = float(np.max(occupied_l1, initial=0.0) ** 2)
    maximum_virtual_pair_l1 = float(np.max(virtual_l1, initial=0.0) ** 2)
    fock_row_sums = np.zeros(state.size)
    np.add.at(fock_row_sums, state.fock_rows, np.abs(state.fock_values))
    unequal = state.fock_rows != state.fock_columns
    np.add.at(
        fock_row_sums,
        state.fock_columns[unequal],
        np.abs(state.fock_values[unequal]),
    )
    maximum_hardness = float(np.max(hardness, initial=0.0))
    maximum_fock_norm = float(np.max(fock_row_sums, initial=0.0))
    # For every screened interaction gamma_p(R), gamma_p <= eta_bar.
    # Atomic transition/pair charges are bounded by products of AO one-norms;
    # the Fock bilinear form is bounded by its infinity norm.  The resulting
    # row-sum bound is deliberately conservative but cannot understate an
    # omitted 2K-J plus one-electron coupling.
    maximum_omitted_edge = (
        2.0 * maximum_hardness * maximum_transition_l1**2
        + 0.5 * maximum_hardness * maximum_occupied_pair_l1 * maximum_virtual_pair_l1
        + 2.0 * maximum_fock_norm * max(maximum_occupied_pair_l1, maximum_virtual_pair_l1)
    )
    omitted_bound = float(np.max(omitted, initial=0)) * maximum_omitted_edge
    return PairPackedSTDAOperator(
        np.asarray(diagonal),
        edge_rows[keep],
        edge_columns[keep],
        edge_values[keep],
        omitted_bound,
    )


def linear_scaling_stda(
    state: GFN2SparseProjectorState,
    *,
    roots: int = 5,
    excitation_cutoff_bohr: float = 12.0,
    coupling_cutoff_bohr: float = 18.0,
    truncation_threshold: float = 1.0e-10,
    qualification_tolerance_hartree: float = 1.0e-5,
) -> LinearScalingSTDAResult:
    """Solve the local closed-shell sTDA graph and certify its residuals."""

    basis = build_sparse_excitation_basis(state, truncation_threshold=truncation_threshold)
    space = build_sparse_excitation_space(state, basis, center_cutoff_bohr=excitation_cutoff_bohr)
    operator = build_pair_packed_stda_operator(
        state,
        basis,
        space,
        coupling_cutoff_bohr=coupling_cutoff_bohr,
        value_threshold_hartree=truncation_threshold,
    )
    if roots < 1:
        raise ValueError("at least one localized sTDA root is required")
    count = min(int(roots), operator.size - 1)
    if operator.size == 1:
        total_bound = basis.inherited_projector_bound + operator.omitted_coupling_bound_hartree
        return LinearScalingSTDAResult(
            operator.diagonal_hartree.copy(),
            np.ones((1, 1)),
            np.zeros(1),
            1,
            1,
            operator.omitted_coupling_bound_hartree,
            space.omitted_excitation_count,
            basis.inherited_projector_bound,
            True,
            bool(
                space.omitted_excitation_count == 0
                and total_bound <= qualification_tolerance_hartree
            ),
        )
    energies, vectors, converged, _batches = _gfn2_native.sparse_symmetric_davidson(
        operator.diagonal_hartree,
        operator.rows,
        operator.columns,
        operator.values_hartree,
        count,
        1.0e-9,
    )
    energies = np.asarray(energies)
    vectors = np.asarray(vectors).T
    order = np.argsort(energies)
    energies = energies[order]
    amplitudes = vectors[:, order].T
    residuals = np.asarray(
        [
            np.linalg.norm(operator.apply(vector) - energy * vector)
            for energy, vector in zip(energies, amplitudes, strict=True)
        ]
    )
    total_bound = basis.inherited_projector_bound + operator.omitted_coupling_bound_hartree
    return LinearScalingSTDAResult(
        energies,
        amplitudes,
        residuals,
        space.size,
        operator.nonzero_count,
        operator.omitted_coupling_bound_hartree,
        space.omitted_excitation_count,
        basis.inherited_projector_bound,
        bool(converged),
        bool(
            bool(converged)
            and np.max(residuals, initial=0.0) <= 1.0e-8
            and space.omitted_excitation_count == 0
            and total_bound <= qualification_tolerance_hartree
        ),
    )


__all__ = [
    "LocalProjectedOrbitals",
    "SparseExcitationBasis",
    "SparseExcitationSpace",
    "PairPackedSTDAOperator",
    "LinearScalingSTDAResult",
    "build_sparse_excitation_basis",
    "build_sparse_excitation_space",
    "build_pair_packed_stda_operator",
    "linear_scaling_stda",
]
