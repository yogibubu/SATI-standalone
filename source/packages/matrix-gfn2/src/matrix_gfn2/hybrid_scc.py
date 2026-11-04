"""Exact dense GFN2 SCC observables and Fock in a persistent hybrid basis.

The atom-local basis change preserves the atom assignment of each function,
so cumulative atomic multipoles retain their endpoint form. Shell labels do
not: the shell projectors are transformed explicitly and never inferred from
the hybrid column index.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hybrid_basis import HybridAtomicBasis
from .multipole import (
    AtomicMultipoles,
    CumulativeMultipoleOperators,
    dense_scc_fock,
    dense_scc_observables,
)


def _packed_to_dense(values: np.ndarray, size: int) -> np.ndarray:
    lower, upper = np.tril_indices(size)
    result = np.empty((size, size), dtype=float)
    result[lower, upper] = values
    result[upper, lower] = values
    return result


def _dense_to_packed(values: np.ndarray) -> np.ndarray:
    lower, upper = np.tril_indices(len(values))
    return np.ascontiguousarray(values[lower, upper])


@dataclass(frozen=True)
class HybridShellProjectors:
    blocks: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]
    size: int
    nshells: int

    @classmethod
    def build(
        cls, hybrid: HybridAtomicBasis, orbital_shell: np.ndarray,
        nshells: int,
    ) -> HybridShellProjectors:
        labels = np.asarray(orbital_shell, dtype=np.intp)
        blocks = []
        for indices, rotation in zip(hybrid.atom_orbitals, hybrid.blocks, strict=True):
            shells = np.unique(labels[indices])
            projectors = np.asarray([
                rotation.T @ np.diag((labels[indices] == shell).astype(float)) @ rotation
                for shell in shells
            ])
            blocks.append((indices, shells, projectors))
        return cls(tuple(blocks), hybrid.size, nshells)

    def populations(self, density: np.ndarray, overlap: np.ndarray) -> np.ndarray:
        result = np.zeros(self.nshells)
        for indices, shells, projectors in self.blocks:
            local = density[indices, :] @ overlap[:, indices]
            result[shells] += np.einsum("sij,ji->s", projectors, local)
        return result

    def fock(self, overlap: np.ndarray, shell_potential: np.ndarray) -> np.ndarray:
        left_product = np.zeros_like(overlap)
        for indices, shells, projectors in self.blocks:
            local_potential = np.tensordot(
                np.asarray(shell_potential)[shells], projectors, axes=(0, 0)
            )
            left_product[indices, :] = local_potential @ overlap[indices, :]
        return -0.5 * (left_product + left_product.T)


@dataclass(frozen=True)
class HybridDenseSCC:
    hybrid: HybridAtomicBasis
    overlap: np.ndarray
    operators: CumulativeMultipoleOperators
    orbital_shell: np.ndarray
    reference_population: np.ndarray
    valence: np.ndarray
    shell_projectors: HybridShellProjectors

    @classmethod
    def from_ao(
        cls,
        hybrid: HybridAtomicBasis,
        operators: CumulativeMultipoleOperators,
        orbital_shell: np.ndarray,
        reference_population: np.ndarray,
        valence: np.ndarray,
    ) -> HybridDenseSCC:
        """Transform all model operators once, before SCC iterations."""
        size = hybrid.size
        if len(operators.atom_index) != size:
            raise ValueError("hybrid and AO multipole sizes differ")
        transformation = np.zeros((size, size))
        for indices, block in zip(hybrid.atom_orbitals, hybrid.blocks, strict=True):
            transformation[np.ix_(indices, indices)] = block
        left_transform = transformation.T

        def rotate(packed):
            return _dense_to_packed(
                left_transform @ _packed_to_dense(packed, size) @ transformation
            )

        overlap = left_transform @ _packed_to_dense(
            operators.pair_overlap, size
        ) @ transformation
        pair_dipole = np.column_stack([
            rotate(operators.pair_dipole[:, component]) for component in range(3)
        ])
        pair_second = np.empty_like(operators.pair_second)
        for row in range(3):
            for column in range(3):
                pair_second[:, row, column] = rotate(
                    operators.pair_second[:, row, column]
                )
        transformed = CumulativeMultipoleOperators(
            operators.atom_index.copy(), operators.centers.copy(),
            _dense_to_packed(overlap), pair_dipole, pair_second,
        )
        return cls(
            hybrid, overlap, transformed,
            np.asarray(orbital_shell, dtype=np.intp),
            np.asarray(reference_population, dtype=float),
            np.asarray(valence, dtype=float),
            HybridShellProjectors.build(
                hybrid, orbital_shell, len(reference_population),
            ),
        )

    def observables(self, density: np.ndarray) -> tuple[np.ndarray, AtomicMultipoles]:
        """SCC shell charges and atom multipoles without an AO density."""
        # The native endpoint scan still gives the exact atom multipoles.
        # Its shell charge is intentionally discarded: hybrids mix AO shells.
        _, multipoles = dense_scc_observables(
            density, self.operators, np.zeros(self.hybrid.size, dtype=np.intp),
            np.zeros(1), self.valence,
        )
        population = self.shell_projectors.populations(density, self.overlap)
        return self.reference_population - population, multipoles

    def fock(
        self,
        core: np.ndarray,
        shell_potential: np.ndarray,
        charge_potential: np.ndarray,
        dipole_potential: np.ndarray,
        quadrupole_potential: np.ndarray,
        d4_charge_derivative: np.ndarray,
    ) -> np.ndarray:
        """Assemble the same GFN2 Fock directly in hybrid coordinates."""
        other = dense_scc_fock(
            core, self.overlap, self.operators,
            np.zeros(self.hybrid.size, dtype=np.intp), np.zeros(1),
            charge_potential, dipole_potential, quadrupole_potential,
            d4_charge_derivative,
        )
        return other + self.shell_projectors.fock(self.overlap, shell_potential)
