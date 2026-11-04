"""Persistent atom-local spd hybrids and exact AO-operator congruences.

This module defines the full-rank basis change. SCC shell projectors and
integral operators must be transformed with it; it is not a density-only guess.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class HybridAtomicBasis:
    atom_orbitals: tuple[np.ndarray, ...]
    blocks: tuple[np.ndarray, ...]
    neutral_density: np.ndarray
    bond_hybrids_by_atom: tuple[int, ...]
    bond_columns_by_atom: tuple[dict[int, int], ...]

    @property
    def size(self) -> int:
        return self.neutral_density.shape[0]

    def transform_operator(self, operator: np.ndarray) -> np.ndarray:
        """Return U.T A U for an AO operator A."""
        values = np.asarray(operator, dtype=float)
        if values.shape != (self.size, self.size):
            raise ValueError("hybrid operator has incompatible dimensions")
        if self.size >= 32:
            transform = self._sparse_transformation()
            result = np.asarray(transform.T @ values @ transform)
            return 0.5 * (result + result.T)
        result = np.empty_like(values)
        for left, (left_indices, left_block) in enumerate(zip(self.atom_orbitals, self.blocks)):
            for right in range(left + 1):
                right_indices = self.atom_orbitals[right]
                block = left_block.T @ values[np.ix_(left_indices, right_indices)] @ self.blocks[right]
                result[np.ix_(left_indices, right_indices)] = block
                if right != left:
                    result[np.ix_(right_indices, left_indices)] = block.T
        return result

    def _sparse_transformation(self):
        rows = []
        columns = []
        values = []
        for indices, block in zip(self.atom_orbitals, self.blocks, strict=True):
            left, right = np.nonzero(block)
            rows.extend(indices[left])
            columns.extend(indices[right])
            values.extend(block[left, right])
        return sparse.coo_matrix((
            values, (rows, columns),
        ), shape=(self.size, self.size)).tocsr()

    def transform_sparse_operator(self, operator):
        """Apply the same atom-local congruence without an AO-square array."""
        from .localized_solver import SparseSymmetricMatrix

        if not isinstance(operator, SparseSymmetricMatrix) or operator.size != self.size:
            raise ValueError("hybrid sparse operator has incompatible dimensions")
        rows = np.asarray(operator.rows, dtype=np.intp)
        columns = np.asarray(operator.columns, dtype=np.intp)
        values = np.asarray(operator.values, dtype=float)
        off_diagonal = rows != columns
        source = sparse.coo_matrix((
            np.concatenate((values, values[off_diagonal])),
            (np.concatenate((rows, columns[off_diagonal])),
             np.concatenate((columns, rows[off_diagonal]))),
        ), shape=(self.size, self.size)).tocsr()
        transform = self._sparse_transformation()
        rotated = (transform.T @ source @ transform).tocsr()
        rotated = (0.5 * (rotated + rotated.T)).tocoo()
        keep = rotated.row >= rotated.col
        order = np.lexsort((rotated.col[keep], rotated.row[keep]))
        return SparseSymmetricMatrix(
            self.size,
            rotated.row[keep][order].astype(np.int32),
            rotated.col[keep][order].astype(np.int32),
            rotated.data[keep][order],
        )

    def density_to_ao(self, hybrid_density: np.ndarray) -> np.ndarray:
        """Return U P_h U.T for boundary-only AO contractions/derivatives."""
        values = np.asarray(hybrid_density, dtype=float)
        if values.shape != (self.size, self.size):
            raise ValueError("hybrid density has incompatible dimensions")
        result = np.empty_like(values)
        for left, (left_indices, left_block) in enumerate(zip(self.atom_orbitals, self.blocks)):
            for right in range(left + 1):
                right_indices = self.atom_orbitals[right]
                block = left_block @ values[np.ix_(left_indices, right_indices)] @ self.blocks[right].T
                result[np.ix_(left_indices, right_indices)] = block
                if right != left:
                    result[np.ix_(right_indices, left_indices)] = block.T
        return result

    def shell_populations(
        self, hybrid_density: np.ndarray, hybrid_overlap: np.ndarray,
        orbital_shell: np.ndarray, nshells: int,
    ) -> np.ndarray:
        """Evaluate transformed Mulliken projectors without AO occupations."""
        density = np.asarray(hybrid_density)
        overlap = np.asarray(hybrid_overlap)
        if density.shape != (self.size, self.size) or overlap.shape != density.shape:
            raise ValueError("hybrid SCC density/overlap have incompatible dimensions")
        populations = np.zeros(nshells)
        labels = np.asarray(orbital_shell, dtype=np.intp)
        for indices, block in zip(self.atom_orbitals, self.blocks, strict=True):
            # Only atom-diagonal blocks of P S enter the shell projectors.
            # Avoid a full dense matrix product on every SCC cycle.
            local = density[indices, :] @ overlap[:, indices]
            for shell in np.unique(labels[indices]):
                selector = (labels[indices] == shell).astype(float)
                projector = block.T @ np.diag(selector) @ block
                populations[shell] += float(np.trace(projector @ local))
        return populations

    def monopole_fock(
        self, hybrid_overlap: np.ndarray, shell_potential: np.ndarray,
        orbital_shell: np.ndarray,
    ) -> np.ndarray:
        """Exact transformed GFN2 monopole Fock, entirely in hybrid space.

        Shell projectors are block-local but generally non-diagonal after
        spd hybridization. Treating hybrid columns as if they retained one
        AO shell label would change the energy and SCC equations.
        """
        overlap = np.asarray(hybrid_overlap, dtype=float)
        potential = np.asarray(shell_potential, dtype=float)
        labels = np.asarray(orbital_shell, dtype=np.intp)
        if (
            overlap.shape != (self.size, self.size)
            or labels.shape != (self.size,)
            or np.any(labels < 0) or np.any(labels >= len(potential))
        ):
            raise ValueError("hybrid SCC potential has incompatible dimensions")
        left_product = np.zeros_like(overlap)
        for indices, block in zip(self.atom_orbitals, self.blocks, strict=True):
            local_potential = (
                block.T @ np.diag(potential[labels[indices]]) @ block
            )
            left_product[indices, :] = local_potential @ overlap[indices, :]
        return -0.5 * (left_product + left_product.T)


def _directional_d(direction: np.ndarray) -> np.ndarray:
    x, y, z = direction
    root3 = np.sqrt(3.0)
    return np.asarray((z * z - 0.5 * (x * x + y * y), root3 * x * z,
                       root3 * y * z, 0.5 * root3 * (x * x - y * y), root3 * x * y))


def build_hybrid_atomic_basis(
    coordinates_angstrom: np.ndarray,
    orbital_atoms: np.ndarray,
    spherical_shell: tuple[str, ...],
    orbital_shell: np.ndarray,
    reference_population: np.ndarray,
    overlap: np.ndarray,
    bonds: tuple[tuple[int, int], ...],
) -> HybridAtomicBasis:
    """Construct directed spd hybrids and project the neutral-atom density."""
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    assignment = np.asarray(orbital_atoms, dtype=np.intp)
    labels = np.asarray(orbital_shell, dtype=np.intp)
    reference = np.asarray(reference_population, dtype=float)
    overlap = np.asarray(overlap, dtype=float)
    natoms = len(xyz)
    norbitals = len(assignment)
    if (
        xyz.shape != (natoms, 3) or not np.all(np.isfinite(xyz))
        or len(spherical_shell) != norbitals or labels.shape != (norbitals,)
        or overlap.shape != (norbitals, norbitals)
        or np.any(assignment < 0) or np.any(assignment >= natoms)
        or np.any(labels < 0) or np.any(labels >= len(reference))
    ):
        raise ValueError("invalid atom-local hybrid model")
    adjacency = [set() for _ in range(natoms)]
    for left, right in bonds:
        if not (0 <= left < natoms and 0 <= right < natoms and left != right):
            raise ValueError("invalid hybrid covalent bond")
        adjacency[left].add(right)
        adjacency[right].add(left)
    atom_orbitals = tuple(np.flatnonzero(assignment == atom) for atom in range(natoms))
    blocks = []
    hybrid_counts = []
    directed_columns = []
    for atom, indices in enumerate(atom_orbitals):
        size = len(indices)
        if size == 0:
            raise ValueError("atom has no GFN2 orbitals")
        local_shells = [spherical_shell[index] for index in indices]
        angular_indices = {
            angular: [local for local, shell in enumerate(local_shells) if shell.endswith(angular)]
            for angular in ("s", "p", "d")
        }
        # Use the outermost complete angular shell for each directed seed.
        s = angular_indices["s"][-1] if angular_indices["s"] else None
        p = angular_indices["p"][-3:] if len(angular_indices["p"]) >= 3 else []
        d = angular_indices["d"][-5:] if len(angular_indices["d"]) >= 5 else []
        candidates = []
        if s is not None and p:
            for neighbor in sorted(adjacency[atom]):
                vector = xyz[neighbor] - xyz[atom]
                norm = float(np.linalg.norm(vector))
                if norm <= 1.0e-8:
                    raise ValueError("coincident atoms cannot define hybrids")
                direction = vector / norm
                candidate = np.zeros(size)
                candidate[s] = 1.0
                candidate[p] = np.sqrt(3.0) * direction
                if d:
                    candidate[d] = np.sqrt(5.0) * _directional_d(direction)
                candidates.append(candidate)
        elif s is not None and size == 1 and len(adjacency[atom]) == 1:
            candidates.append(np.ones(1))
        columns = []
        directed = 0
        bond_columns: dict[int, int] = {}
        neighbors = sorted(adjacency[atom])
        for candidate_index, candidate in enumerate(
            candidates + [np.eye(size)[:, index] for index in range(size)]
        ):
            vector = candidate.copy()
            for column in columns:
                vector -= np.dot(column, vector) * column
            norm = float(np.linalg.norm(vector))
            if norm <= 1.0e-10:
                continue
            vector /= norm
            if candidate_index < len(candidates):
                directed += 1
                bond_columns[neighbors[candidate_index]] = len(columns)
            columns.append(vector)
            if len(columns) == size:
                break
        block = np.column_stack(columns)
        if block.shape != (size, size):
            raise RuntimeError("hybrid basis is rank deficient")
        blocks.append(block)
        hybrid_counts.append(directed)
        directed_columns.append(bond_columns)
    width = np.bincount(labels, minlength=len(reference))
    if np.any(width[labels] == 0) or np.any(np.diag(overlap) <= 0.0):
        raise ValueError("invalid neutral-shell hybrid population")
    neutral_ao_diagonal = reference[labels] / width[labels] / np.diag(overlap)
    neutral_hybrid = np.zeros((norbitals, norbitals))
    for indices, block in zip(atom_orbitals, blocks, strict=True):
        neutral_hybrid[np.ix_(indices, indices)] = (
            block.T @ np.diag(neutral_ao_diagonal[indices]) @ block
        )
    return HybridAtomicBasis(atom_orbitals, tuple(blocks), neutral_hybrid,
                             tuple(hybrid_counts), tuple(directed_columns))
