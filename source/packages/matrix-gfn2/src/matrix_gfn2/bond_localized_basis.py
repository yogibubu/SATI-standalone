"""Exact 2x2 polarized bond/antibond rotations of atom-local hybrids.

Each rotation is Euclidean-orthogonal and touches only the two directed
hybrids of one internal covalent bond. Antibonding partners and unused
lone-pair/core hybrids remain in the full-rank basis. Boundary bonds are
deferred until interfragment coupling; no link function is silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hybrid_basis import HybridAtomicBasis
from .hybrid_domains import PersistentHybridDomains
from .hybrid_region_orbitals import HybridRegionSeeds


@dataclass(frozen=True)
class PolarizedBondRotation:
    bond: tuple[int, int]
    bonding_index: int
    antibonding_index: int
    left_coefficient: float
    right_coefficient: float


@dataclass(frozen=True)
class BondLocalizedBasis:
    size: int
    rotations: tuple[PolarizedBondRotation, ...]
    deferred_boundary_bonds: tuple[tuple[int, int], ...]

    def transformation(self) -> np.ndarray:
        result = np.eye(self.size)
        for item in self.rotations:
            left, right = item.bonding_index, item.antibonding_index
            alpha, beta = item.left_coefficient, item.right_coefficient
            result[left, left] = alpha
            result[left, right] = -beta
            result[right, left] = beta
            result[right, right] = alpha
        return result

    def transform_operator(self, hybrid_operator: np.ndarray) -> np.ndarray:
        """Apply all pairwise T.T A T rotations without a dense T product."""
        result = np.asarray(hybrid_operator, dtype=float).copy()
        if result.shape != (self.size, self.size):
            raise ValueError("bond-localized operator shape differs from basis")
        for item in self.rotations:
            left, right = item.bonding_index, item.antibonding_index
            alpha, beta = item.left_coefficient, item.right_coefficient
            old_left = result[left, :].copy()
            old_right = result[right, :].copy()
            result[left, :] = alpha * old_left + beta * old_right
            result[right, :] = -beta * old_left + alpha * old_right
            old_left = result[:, left].copy()
            old_right = result[:, right].copy()
            result[:, left] = alpha * old_left + beta * old_right
            result[:, right] = -beta * old_left + alpha * old_right
        return result


def build_bond_localized_basis(
    hybrid: HybridAtomicBasis,
    seeds: HybridRegionSeeds,
    domains: PersistentHybridDomains,
) -> BondLocalizedBasis:
    """Keep an exact full-rank basis while making directed hybrid bond pairs."""
    used = np.zeros(hybrid.size, dtype=bool)
    rotations = []
    deferred = []
    for seed in seeds.bonds:
        left_atom, right_atom = seed.atoms
        if domains.atom_fragment[left_atom] != domains.atom_fragment[right_atom]:
            deferred.append(seed.atoms)
            continue
        left_column = hybrid.bond_columns_by_atom[left_atom].get(right_atom)
        right_column = hybrid.bond_columns_by_atom[right_atom].get(left_atom)
        if left_column is None or right_column is None:
            continue
        left = int(hybrid.atom_orbitals[left_atom][left_column])
        right = int(hybrid.atom_orbitals[right_atom][right_column])
        if used[left] or used[right]:
            raise ValueError("hybrid orbital was assigned to two covalent bonds")
        used[left] = used[right] = True
        alpha = float(np.sqrt(seed.site_populations[0] / 2.0))
        beta = float(np.sqrt(seed.site_populations[1] / 2.0))
        rotations.append(PolarizedBondRotation(
            seed.atoms, left, right, alpha, beta,
        ))
    return BondLocalizedBasis(hybrid.size, tuple(rotations), tuple(deferred))
