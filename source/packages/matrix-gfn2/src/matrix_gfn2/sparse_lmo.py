"""Experimental local occupied/virtual orbitals for a sparse GFN2 solver.

This module builds an orthogonal Lewis-type orbital seed without a global
eigensolve. It does not by itself solve the SCC equations or establish linear
scaling of a complete energy/gradient calculation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from .hybrid_basis import _directional_d
from .localized_solver import SparseSymmetricMatrix
from .occupied_local_subspace import orthogonalize_local_occupied_subspace
from .locality import LocalizedOrbitalDomains


def symmetric_to_csr(matrix: SparseSymmetricMatrix) -> sparse.csr_matrix:
    """Expose the stored lower triangle to SciPy without an AO-square array."""
    off = matrix.rows != matrix.columns
    return sparse.coo_matrix((
        np.concatenate((matrix.values, matrix.values[off])),
        (np.concatenate((matrix.rows, matrix.columns[off])),
         np.concatenate((matrix.columns, matrix.rows[off]))),
    ), shape=matrix.shape).tocsr()


@dataclass(frozen=True)
class SparseLewisOrbitalSeed:
    """Full-rank Euclidean-orthogonal LMO basis in orthogonal AO coordinates."""

    coefficients: sparse.csc_matrix
    orbital_atoms: np.ndarray
    occupied: int
    bonding_orbitals: int
    atom_hybrid_nonzeros: int

    @property
    def fill_fraction(self) -> float:
        n = self.coefficients.shape[0]
        return self.coefficients.nnz / (n * n)

    def orthogonality_error(self) -> float:
        n = self.coefficients.shape[1]
        gram = self.coefficients.T @ self.coefficients - sparse.identity(n)
        return float(np.max(abs(gram.data), initial=0.0))

    def centers_angstrom(self, coordinates_angstrom: np.ndarray) -> np.ndarray:
        xyz = np.asarray(coordinates_angstrom, dtype=float)
        weights = self.coefficients.power(2)
        centers = np.asarray(weights.T @ xyz[self.orbital_atoms])
        normalization = np.asarray(weights.sum(axis=0)).ravel()
        return centers / normalization[:, None]


def local_orbital_support_mask(
    coordinates_angstrom: np.ndarray,
    orbital_atoms: Sequence[int],
    orbital_centers_angstrom: np.ndarray,
    radius_angstrom: float,
) -> sparse.csc_matrix:
    """Bound every LMO column to nearby AO atoms with a spatial index."""
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    centers = np.asarray(orbital_centers_angstrom, dtype=float)
    atoms = np.asarray(orbital_atoms, dtype=np.intp)
    if (xyz.ndim != 2 or xyz.shape[1] != 3
            or centers.ndim != 2 or centers.shape[1] != 3
            or np.any(atoms < 0) or np.any(atoms >= len(xyz))
            or radius_angstrom <= 0.0 or not np.all(np.isfinite(centers))):
        raise ValueError("invalid LMO support geometry")
    atom_orbitals = [np.flatnonzero(atoms == atom) for atom in range(len(xyz))]
    neighbors = cKDTree(xyz).query_ball_point(centers, radius_angstrom)
    rows: list[int] = []
    columns: list[int] = []
    for orbital, nearby in enumerate(neighbors):
        selected = np.concatenate([atom_orbitals[atom] for atom in nearby])
        rows.extend(selected)
        columns.extend([orbital] * len(selected))
    return sparse.coo_matrix((
        np.ones(len(rows), dtype=bool), (rows, columns),
    ), shape=(len(atoms), len(centers))).tocsc()


def _atomic_hybrids(
    coordinates_angstrom: np.ndarray,
    orbital_atoms: np.ndarray,
    spherical_shell: Sequence[str],
    bonds: Sequence[tuple[int, int]],
) -> tuple[sparse.csc_matrix, dict[tuple[int, int], int], int]:
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    assignment = np.asarray(orbital_atoms, dtype=np.intp)
    size = len(assignment)
    if (xyz.ndim != 2 or xyz.shape[1] != 3
            or len(spherical_shell) != size or np.any(assignment < 0)
            or np.any(assignment >= len(xyz)) or not np.all(np.isfinite(xyz))):
        raise ValueError("invalid local hybrid inputs")
    adjacency = [set() for _ in xyz]
    for left, right in bonds:
        if (not 0 <= left < len(xyz) or not 0 <= right < len(xyz)
                or left == right or right in adjacency[left]):
            raise ValueError("invalid or repeated Lewis bond")
        adjacency[left].add(right)
        adjacency[right].add(left)
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    directed: dict[tuple[int, int], int] = {}
    for atom in range(len(xyz)):
        indices = np.flatnonzero(assignment == atom)
        if not len(indices):
            raise ValueError("atom has no AO functions")
        local_shell = [spherical_shell[index] for index in indices]
        angular = {
            symbol: [i for i, label in enumerate(local_shell) if label.endswith(symbol)]
            for symbol in ("s", "p", "d")
        }
        s_index = angular["s"][-1] if angular["s"] else None
        p_indices = angular["p"][-3:] if len(angular["p"]) >= 3 else []
        d_indices = angular["d"][-5:] if len(angular["d"]) >= 5 else []
        neighbors = sorted(adjacency[atom])
        if len(neighbors) > len(indices):
            raise ValueError("Lewis coordination exceeds the atomic AO basis")
        candidates: list[np.ndarray] = []
        for neighbor in neighbors:
            vector = xyz[neighbor] - xyz[atom]
            length = float(np.linalg.norm(vector))
            if length <= 1.0e-8:
                raise ValueError("coincident atoms cannot define hybrids")
            direction = vector / length
            candidate = np.zeros(len(indices))
            if s_index is not None:
                candidate[s_index] = 1.0
            if p_indices:
                candidate[p_indices] = np.sqrt(3.0) * direction
            if d_indices:
                candidate[d_indices] = np.sqrt(5.0) * _directional_d(direction)
            if not np.any(candidate):
                raise ValueError("bond direction has no local AO support")
            candidates.append(candidate)
        columns_local: list[np.ndarray] = []
        for candidate_index, candidate in enumerate(
            candidates + [np.eye(len(indices))[:, i] for i in range(len(indices))]
        ):
            vector = candidate.copy()
            for _ in range(2):
                for previous in columns_local:
                    vector -= float(previous @ vector) * previous
            norm = float(np.linalg.norm(vector))
            if norm <= 1.0e-10:
                if candidate_index < len(candidates):
                    raise ValueError("directed hybrid is linearly dependent")
                continue
            vector /= norm
            column = len(columns_local)
            if candidate_index < len(candidates):
                directed[(atom, neighbors[candidate_index])] = int(indices[column])
            columns_local.append(vector)
            if len(columns_local) == len(indices):
                break
        if len(columns_local) != len(indices):
            raise RuntimeError("atomic hybrid basis lost rank")
        block = np.column_stack(columns_local)
        row_local, col_local = np.nonzero(block)
        rows.extend(indices[row_local])
        columns.extend(indices[col_local])
        values.extend(block[row_local, col_local])
    transform = sparse.coo_matrix(
        (values, (rows, columns)), shape=(size, size),
    ).tocsc()
    return transform, directed, transform.nnz


def oracle_carbonyl_pi_bonds(
    plan: object, atomic_numbers: Sequence[int],
) -> tuple[tuple[int, int], ...]:
    """Pick unambiguous carbonyl pi pairs from an ORACLE region plan."""
    numbers = np.asarray(atomic_numbers, dtype=np.intp)
    degree = np.zeros(len(numbers), dtype=np.intp)
    for left, right in plan.bonds:
        degree[left] += 1
        degree[right] += 1
    selected = []
    for region in plan.regions:
        if region.kind != "DELOCALIZED_PI" or len(region.atoms) != 2:
            continue
        left, right = (int(atom) for atom in region.atoms)
        if (numbers[left] == 6 and numbers[right] == 8
                and degree[left] == 3 and degree[right] == 1):
            selected.append((left, right))
        elif (numbers[right] == 6 and numbers[left] == 8
                and degree[right] == 3 and degree[left] == 1):
            selected.append((right, left))
    return tuple(sorted(set(selected)))


def build_sparse_lewis_orbital_seed(
    coordinates_angstrom: np.ndarray,
    orbital_atoms: Sequence[int],
    spherical_shell: Sequence[str],
    bonds: Sequence[tuple[int, int]],
    orthogonal_core: SparseSymmetricMatrix,
    occupied_orbitals: int,
    *,
    site_charges: np.ndarray | None = None,
    atomic_numbers: Sequence[int] | None = None,
    pi_bonds: Sequence[tuple[int, int]] = (),
) -> SparseLewisOrbitalSeed:
    """Make local occupied and virtual columns from Lewis bonds and AO hybrids.

    Each bond makes a disjoint 2x2 orthogonal bonding/antibonding rotation.
    Remaining occupied columns are selected from local core expectations
    without diagonalizing the complete Hamiltonian. The resulting seed is
    only an initial determinant and must be relaxed self-consistently.
    """
    atoms = np.asarray(orbital_atoms, dtype=np.intp)
    size = len(atoms)
    if (orthogonal_core.size != size or not 0 < occupied_orbitals <= size
            or len(bonds) > occupied_orbitals):
        raise ValueError("invalid Lewis occupied space")
    charges = (
        np.zeros(len(coordinates_angstrom)) if site_charges is None
        else np.asarray(site_charges, dtype=float)
    )
    if charges.shape != (len(coordinates_angstrom),) or not np.all(np.isfinite(charges)):
        raise ValueError("invalid Lewis site charges")
    hybrid, directed, hybrid_nnz = _atomic_hybrids(
        coordinates_angstrom, atoms, spherical_shell, bonds,
    )
    transform = sparse.identity(size, format="lil")
    used = np.zeros(size, dtype=bool)
    bond_columns: list[int] = []
    for left, right in bonds:
        a = directed[(left, right)]
        b = directed[(right, left)]
        if used[a] or used[b]:
            raise ValueError("atomic hybrid was assigned to multiple bonds")
        used[a] = used[b] = True
        population = np.maximum(0.05, 1.0 - charges[[left, right]])
        population *= 2.0 / float(np.sum(population))
        alpha, beta = np.sqrt(population / 2.0)
        transform[a, a] = alpha
        transform[b, a] = beta
        transform[a, b] = -beta
        transform[b, b] = alpha
        bond_columns.append(a)
    coefficients = (hybrid @ transform.tocsc()).tocsc()
    core = symmetric_to_csr(orthogonal_core)
    pi_columns: list[int] = []
    if atomic_numbers is not None:
        numbers = np.asarray(atomic_numbers, dtype=np.intp)
        if numbers.shape != (len(coordinates_angstrom),):
            raise ValueError("Lewis atom identities do not match geometry")
        local_core = (hybrid.T @ (core @ hybrid)).tocsr()
        for carbon, oxygen in pi_bonds:
            carbon_options = np.flatnonzero((atoms == carbon) & ~used)
            oxygen_options = np.flatnonzero((atoms == oxygen) & ~used)
            if not len(carbon_options) or not len(oxygen_options):
                raise ValueError("carbonyl has no unused local pi hybrids")
            candidates = np.asarray([
                abs(local_core[left, right])
                for left in carbon_options for right in oxygen_options
            ])
            best = int(np.argmax(candidates))
            left = int(carbon_options[best // len(oxygen_options)])
            right = int(oxygen_options[best % len(oxygen_options)])
            used[left] = used[right] = True
            population = np.maximum(0.05, 1.0 - charges[[carbon, oxygen]])
            population *= 2.0 / float(np.sum(population))
            alpha, beta = np.sqrt(population / 2.0)
            transform[left, left] = alpha
            transform[right, left] = beta
            transform[left, right] = -beta
            transform[right, right] = alpha
            pi_columns.append(left)
        coefficients = (hybrid @ transform.tocsc()).tocsc()
    projected_diagonal = np.asarray(
        coefficients.multiply(core @ coefficients).sum(axis=0)
    ).ravel()
    if atomic_numbers is None:
        candidate = np.ones(size, dtype=bool)
        candidate[bond_columns] = False
        candidate_indices = np.flatnonzero(candidate)
        remaining = occupied_orbitals - len(bond_columns)
        chosen = (
            candidate_indices[np.argpartition(
                projected_diagonal[candidate_indices], remaining - 1,
            )[:remaining]] if remaining else np.empty(0, dtype=np.intp)
        )
    else:
        degree = np.zeros(len(numbers), dtype=np.intp)
        pi_degree = np.zeros(len(numbers), dtype=np.intp)
        for left, right in bonds:
            degree[left] += 1
            degree[right] += 1
        for left, right in pi_bonds:
            pi_degree[left] += 1
            pi_degree[right] += 1
        valence = {1: 1, 6: 4, 7: 5, 8: 6}
        chosen_list: list[int] = []
        for atom, number in enumerate(numbers):
            if int(number) not in valence:
                raise ValueError("Lewis lone-pair policy does not cover this element")
            remaining_electrons = valence[int(number)] - degree[atom] - pi_degree[atom]
            if remaining_electrons < 0 or remaining_electrons % 2:
                raise ValueError("Lewis bond orders do not define lone pairs")
            lone_pairs = remaining_electrons // 2
            options = np.flatnonzero((atoms == atom) & ~used)
            if lone_pairs > len(options):
                raise ValueError("Lewis lone pairs exceed local AO support")
            if lone_pairs:
                chosen_list.extend(options[np.argpartition(
                    projected_diagonal[options], lone_pairs - 1,
                )[:lone_pairs]])
        chosen = np.asarray(chosen_list, dtype=np.intp)
    occupied_indices = np.concatenate((
        bond_columns, pi_columns, np.sort(chosen),
    )).astype(np.intp)
    if len(occupied_indices) != occupied_orbitals:
        raise ValueError("Lewis occupancy does not match the electron count")
    virtual_indices = np.flatnonzero(~np.isin(np.arange(size), occupied_indices))
    ordered = coefficients[:, np.concatenate((occupied_indices, virtual_indices))]
    result = SparseLewisOrbitalSeed(
        ordered.tocsc(), atoms.copy(), occupied_orbitals, len(bond_columns),
        hybrid_nnz,
    )
    if result.orthogonality_error() > 1.0e-10:
        raise RuntimeError("Lewis orbital rotations lost Euclidean orthogonality")
    return result


@dataclass(frozen=True)
class SparseLocalRotationSweep:
    coefficients: sparse.csc_matrix
    selected_pairs: int
    maximum_residual_before: float
    maximum_residual_after: float
    energy_before: float
    energy_after: float
    orthogonality_error: float

    @property
    def fill_fraction(self) -> float:
        size = self.coefficients.shape[0]
        return self.coefficients.nnz / (size * size)


def sparse_local_orbital_sweep(
    coefficients: sparse.csc_matrix,
    orthogonal_fock: SparseSymmetricMatrix | sparse.spmatrix,
    occupied: int,
    *,
    maximum_rotation: float = 0.25,
    residual_threshold: float = 1.0e-7,
    coefficient_threshold: float = 1.0e-12,
    support_mask: sparse.csc_matrix | None = None,
    orthogonality_tolerance: float = 1.0e-8,
) -> SparseLocalRotationSweep:
    """Apply disjoint occupied/virtual Jacobi rotations without dense arrays.

    All nonzero occupied/virtual Fock couplings are inspected. The Fock matrix
    is fixed over one sweep; an SCC caller must rebuild it and check energy
    and convergence after accepting the new determinant.
    """
    q = sparse.csc_matrix(coefficients, dtype=float)
    size = q.shape[0]
    if (q.shape != (size, size) or not 0 < occupied < size
            or not 0.0 < maximum_rotation <= np.pi / 4.0
            or residual_threshold <= 0.0 or coefficient_threshold < 0.0
            or orthogonality_tolerance <= 0.0):
        raise ValueError("invalid sparse local orbital sweep")
    if support_mask is not None and support_mask.shape != q.shape:
        raise ValueError("LMO support mask and orbital dimensions differ")
    if isinstance(orthogonal_fock, SparseSymmetricMatrix):
        fock = symmetric_to_csr(orthogonal_fock)
    else:
        fock = orthogonal_fock.tocsr()
    if fock.shape != q.shape:
        raise ValueError("sparse Fock and orbital dimensions differ")
    projected = (q.T @ (fock @ q)).tocsr()
    coupling = projected[occupied:, :occupied].tocoo()
    maximum_before = float(np.max(abs(coupling.data), initial=0.0))
    diagonal = projected.diagonal()
    before = 2.0 * float(np.sum(diagonal[:occupied]))
    if maximum_before < residual_threshold:
        gram = q.T @ q - sparse.identity(size)
        return SparseLocalRotationSweep(
            q, 0, maximum_before, maximum_before, before, before,
            float(np.max(abs(gram.data), initial=0.0)),
        )
    delta = diagonal[occupied + coupling.row] - diagonal[coupling.col]
    angles = 0.5 * np.arctan2(-2.0 * coupling.data, delta)
    angles = np.clip(angles, -maximum_rotation, maximum_rotation)
    eligible = abs(coupling.data) >= residual_threshold
    candidates = np.flatnonzero(eligible)
    order = candidates[np.argsort(-abs(angles[candidates]), kind="stable")]
    used_occupied = np.zeros(occupied, dtype=bool)
    used_virtual = np.zeros(size - occupied, dtype=bool)
    rotation = sparse.identity(size, format="lil")
    count = 0
    for index in order:
        virtual = int(coupling.row[index])
        orbital = int(coupling.col[index])
        if used_occupied[orbital] or used_virtual[virtual]:
            continue
        used_occupied[orbital] = True
        used_virtual[virtual] = True
        angle = float(angles[index])
        cosine, sine = np.cos(angle), np.sin(angle)
        right = occupied + virtual
        rotation[orbital, orbital] = cosine
        rotation[right, orbital] = sine
        rotation[orbital, right] = -sine
        rotation[right, right] = cosine
        count += 1
    trial = (q @ rotation.tocsc()).tocsc()
    if support_mask is not None:
        trial = trial.multiply(support_mask).tocsc()
    if coefficient_threshold:
        trial.data[abs(trial.data) < coefficient_threshold] = 0.0
        trial.eliminate_zeros()
    gram = trial.T @ trial - sparse.identity(size)
    error = float(np.max(abs(gram.data), initial=0.0))
    if error > orthogonality_tolerance:
        corrected = orthogonalize_local_occupied_subspace(
            trial, sparse.identity(size, format="csc"),
            np.arange(size), truncation_threshold=min(
                coefficient_threshold, 1.0e-10,
            ),
            orthogonality_tolerance=orthogonality_tolerance,
        )
        trial = corrected.subspace.coefficients
        error = corrected.maximum_orthogonality_error
        if support_mask is not None:
            trial = trial.multiply(support_mask).tocsc()
            gram = trial.T @ trial - sparse.identity(size)
            error = float(np.max(abs(gram.data), initial=0.0))
            if error > orthogonality_tolerance:
                raise ValueError("LMO support truncation lost orthogonality")
    projected_after = (trial.T @ (fock @ trial)).tocsr()
    residual_after = projected_after[occupied:, :occupied]
    after = 2.0 * float(np.sum(projected_after.diagonal()[:occupied]))
    return SparseLocalRotationSweep(
        trial, count, maximum_before,
        float(np.max(abs(residual_after.data), initial=0.0)),
        before, after, error,
    )


def sparse_projector_from_local_orbitals(
    coefficients: sparse.csc_matrix,
    occupied: int,
    domains: LocalizedOrbitalDomains,
    *,
    threshold: float = 1.0e-12,
    maximum_discarded_norm: float = 1.0e-5,
) -> tuple[SparseSymmetricMatrix, float]:
    """Form a domain-restricted occupied projector and audit its omission."""
    q = sparse.csc_matrix(coefficients)
    size = q.shape[0]
    if (q.shape != (size, size) or domains.norbitals != size
            or not 0 < occupied < size or threshold < 0.0
            or maximum_discarded_norm < 0.0):
        raise ValueError("invalid local occupied projector")
    block = q[:, :occupied]
    projector = (block @ block.T).tocoo()
    lower = projector.row >= projector.col
    rows = projector.row[lower]
    columns = projector.col[lower]
    values = projector.data[lower]
    allowed_left, allowed_right = domains.allowed_pair_arrays()
    keys = rows.astype(np.int64) * size + columns
    allowed_keys = allowed_left.astype(np.int64) * size + allowed_right
    allowed_keys.sort()
    found = np.searchsorted(allowed_keys, keys)
    retained = (found < len(allowed_keys))
    retained[retained] &= allowed_keys[found[retained]] == keys[retained]
    retained &= ((rows == columns) | (abs(values) >= threshold))
    rejected = ~retained
    discarded = float(np.sqrt(np.sum(
        np.where(rows[rejected] == columns[rejected], 1.0, 2.0)
        * values[rejected] ** 2,
    )))
    if discarded > maximum_discarded_norm:
        raise ValueError(
            "local occupied projector exceeds the domain truncation budget "
            f"({discarded:.3e} > {maximum_discarded_norm:.3e})"
        )
    rows = rows[retained].astype(np.int32)
    columns = columns[retained].astype(np.int32)
    values = values[retained]
    order = np.lexsort((columns, rows))
    return SparseSymmetricMatrix(
        size, rows[order], columns[order], values[order],
    ), discarded
