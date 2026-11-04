"""Atom-granular orbital supports with fail-closed compression.

Each accepted orbital update may be tidied by removing small *whole-atom*
coefficient blocks. Candidate compression is retained only if the occupied
projector, S-orthogonality, and full GFN2 energy pass explicit budgets. This
Accepted supports can be held as CSR operands across macro-iterations. The
GFN2 energy callback still consumes a dense density and returns a dense Fock;
this is an experimental sparse algebra route, not a sparse native SCC kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix


@dataclass(frozen=True)
class SupportPolicy:
    atom_population_threshold: float = 1.0e-12
    maximum_orthogonality_error: float = 1.0e-8
    maximum_energy_error_hartree: float = 1.0e-8
    maximum_density_error: float = 1.0e-6
    maximum_reorthogonalization_steps: int = 8


@dataclass(frozen=True)
class SupportUpdate:
    coefficients: np.ndarray
    density: np.ndarray
    fock: np.ndarray
    energy: float
    accepted: bool
    changed: bool
    orbital_atom_supports: tuple[tuple[int, ...], ...]
    coefficient_nnz: int
    maximum_orthogonality_error: float
    maximum_density_error: float
    energy_error_hartree: float


@dataclass(frozen=True)
class SparseOrbitalSupport:
    """Persistent CSR coefficients for density and MO-Fock projection."""

    coefficients: csr_matrix
    occupied: int

    @classmethod
    def from_dense(cls, coefficients: np.ndarray, occupied: int):
        dense = np.asarray(coefficients, dtype=float)
        if (dense.ndim != 2 or dense.shape[0] != dense.shape[1]
                or not 0 < occupied < len(dense)
                or not np.all(np.isfinite(dense))):
            raise ValueError("invalid sparse orbital coefficients")
        return cls(csr_matrix(dense), occupied)

    @property
    def density(self) -> np.ndarray:
        block = self.coefficients[:, :self.occupied]
        return (2.0 * (block @ block.T)).toarray()

    def project_fock(self, fock: np.ndarray) -> np.ndarray:
        matrix = np.asarray(fock, dtype=float)
        if matrix.shape != self.coefficients.shape:
            raise ValueError("invalid Fock matrix shape")
        return np.asarray(self.coefficients.T @ (matrix @ self.coefficients))


@dataclass(frozen=True)
class FragmentSupportWorkspace:
    """Disjoint fragment MO supports used before global orthogonalization."""

    coefficients: csc_matrix
    occupations: np.ndarray
    blocks: tuple[tuple[np.ndarray, np.ndarray, slice], ...]
    packed_rows: np.ndarray
    ao_offsets: np.ndarray
    mo_offsets: np.ndarray
    packed_values: np.ndarray

    @classmethod
    def from_states(cls, states: tuple[object, ...], size: int):
        if size < 1 or not states:
            raise ValueError("invalid fragment support model")
        row_parts = []
        column_parts = []
        value_parts = []
        occupations = []
        blocks = []
        ao_offsets = [0]
        mo_offsets = [0]
        covered = np.zeros(size, dtype=bool)
        column_offset = 0
        for state in states:
            rows = np.asarray(state.global_orbitals, dtype=np.intp)
            block = np.asarray(state.coefficients_on_real_atoms, dtype=float)
            occupation = (
                np.asarray(state.alpha_occupations, dtype=float)
                + np.asarray(state.beta_occupations, dtype=float)
            )
            if (block.shape != (len(rows), len(occupation))
                    or len(rows) == 0 or np.any(rows < 0)
                    or np.any(rows >= size) or np.any(covered[rows])
                    or len(np.unique(rows)) != len(rows)
                    or not np.all(np.isfinite(block))
                    or not np.all(np.isfinite(occupation))):
                raise ValueError("incompatible fragment orbital supports")
            covered[rows] = True
            local_rows, local_columns = np.nonzero(block)
            row_parts.append(rows[local_rows])
            column_parts.append(column_offset + local_columns)
            value_parts.append(block[local_rows, local_columns])
            occupations.extend(occupation)
            blocks.append((rows.copy(), block.copy(),
                           slice(column_offset, column_offset + len(occupation))))
            column_offset += len(occupation)
            ao_offsets.append(ao_offsets[-1] + len(rows))
            mo_offsets.append(column_offset)
        if not np.all(covered):
            raise ValueError("fragment orbital supports do not cover the AO basis")
        coefficients = csc_matrix((
            np.concatenate(value_parts),
            (np.concatenate(row_parts), np.concatenate(column_parts)),
        ), shape=(size, column_offset))
        return cls(
            coefficients, np.asarray(occupations, dtype=float), tuple(blocks),
            np.concatenate([item[0] for item in blocks]).astype(np.int64),
            np.asarray(ao_offsets, dtype=np.int64),
            np.asarray(mo_offsets, dtype=np.int64),
            np.concatenate([item[1].ravel() for item in blocks]),
        )

    def native_operands(self, fock: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Build fragment density and projected Fock using native block kernels."""
        from . import _gfn2_native

        density, projected = _gfn2_native.block_sparse_fragment_operands(
            np.asarray(fock, dtype=float), self.packed_rows,
            self.ao_offsets, self.mo_offsets, self.packed_values,
            self.occupations,
        )
        return np.asarray(density), np.asarray(projected)

    @property
    def density(self) -> np.ndarray:
        result = np.zeros((self.coefficients.shape[0],) * 2)
        for rows, block, columns in self.blocks:
            result[np.ix_(rows, rows)] = (
                block * self.occupations[columns][None, :]
            ) @ block.T
        return result

    def project_fock(self, fock: np.ndarray) -> np.ndarray:
        matrix = np.asarray(fock, dtype=float)
        if matrix.shape != (self.coefficients.shape[0],) * 2:
            raise ValueError("incompatible fragment Fock matrix")
        result = np.empty((self.coefficients.shape[1],) * 2)
        for left_rows, left_block, left_columns in self.blocks:
            for right_rows, right_block, right_columns in self.blocks:
                result[left_columns, right_columns] = (
                    left_block.T @ matrix[np.ix_(left_rows, right_rows)]
                    @ right_block
                )
        return result


def _supports(coefficients: np.ndarray, orbital_atoms: np.ndarray):
    atoms = np.asarray(orbital_atoms, dtype=np.intp)
    return tuple(
        tuple(int(atom) for atom in np.unique(atoms[np.flatnonzero(column)]))
        for column in coefficients.T
    )


def tidy_orbital_support(
    coefficients: np.ndarray,
    overlap: np.ndarray,
    orbital_atoms: np.ndarray,
    occupied: int,
    fock_and_energy: Callable[[np.ndarray], tuple[np.ndarray, float]],
    *,
    reference_fock: np.ndarray,
    reference_energy: float,
    policy: SupportPolicy = SupportPolicy(),
) -> SupportUpdate:
    """Attempt local block pruning; keep the exact orbitals if uncertified."""
    c = np.asarray(coefficients, dtype=float)
    s = np.asarray(overlap, dtype=float)
    atoms = np.asarray(orbital_atoms, dtype=np.intp)
    size = len(c)
    if (
        c.shape != (size, size) or s.shape != c.shape
        or atoms.shape != (size,) or np.any(atoms < 0)
        or not 0 < occupied < size
        or policy.atom_population_threshold <= 0.0
        or policy.maximum_orthogonality_error <= 0.0
        or policy.maximum_energy_error_hartree < 0.0
        or policy.maximum_density_error < 0.0
        or policy.maximum_reorthogonalization_steps < 0
    ):
        raise ValueError("invalid persistent orbital support model")
    natoms = int(np.max(atoms)) + 1
    candidate = c.copy()
    for orbital in range(size):
        populations = np.bincount(
            atoms, weights=np.square(c[:, orbital]), minlength=natoms,
        )
        largest = int(np.argmax(populations))
        keep_atoms = populations >= (
            policy.atom_population_threshold * float(np.sum(populations))
        )
        keep_atoms[largest] = True
        candidate[~keep_atoms[atoms], orbital] = 0.0
    changed = bool(np.any(candidate != c))
    original_density = 2.0 * c[:, :occupied] @ c[:, :occupied].T
    if changed:
        mask = candidate != 0.0
        for _ in range(policy.maximum_reorthogonalization_steps + 1):
            norms = np.einsum("ij,ij->j", candidate, s @ candidate)
            if np.min(norms) <= 1.0e-12:
                break
            candidate /= np.sqrt(norms)[None, :]
            gram_error = candidate.T @ s @ candidate - np.eye(size)
            if float(np.max(abs(gram_error))) <= policy.maximum_orthogonality_error:
                break
            candidate -= 0.5 * candidate @ gram_error
            candidate[~mask] = 0.0
        orthogonality = float(np.max(abs(
            candidate.T @ s @ candidate - np.eye(size)
        )))
        if np.count_nonzero(candidate) * 2 <= candidate.size:
            density = SparseOrbitalSupport.from_dense(candidate, occupied).density
        else:
            density = 2.0 * candidate[:, :occupied] @ candidate[:, :occupied].T
        density_error = float(np.max(abs(density - original_density)))
        if (orthogonality <= policy.maximum_orthogonality_error
                and density_error <= policy.maximum_density_error):
            fock, energy = fock_and_energy(density)
            energy_error = abs(float(energy) - reference_energy)
            if energy_error <= policy.maximum_energy_error_hartree:
                return SupportUpdate(
                    candidate, density, np.asarray(fock), float(energy), True, True,
                    _supports(candidate, atoms), int(np.count_nonzero(candidate)),
                    orthogonality, density_error, energy_error,
                )
        else:
            energy_error = np.inf
    else:
        orthogonality = float(np.max(abs(c.T @ s @ c - np.eye(size))))
        density_error = 0.0
        energy_error = 0.0
    return SupportUpdate(
        c.copy(), original_density, np.asarray(reference_fock),
        float(reference_energy), False, changed,
        _supports(c, atoms), int(np.count_nonzero(c)),
        orthogonality, density_error, energy_error,
    )
