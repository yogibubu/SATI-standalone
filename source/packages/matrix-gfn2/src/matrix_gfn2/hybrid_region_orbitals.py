"""Charge-polarized bond seeds and complete local pi spaces in hybrid coordinates.

ORACLE charges only polarize the initial vectors. They are not substituted for
SCF occupations, and the exact local generalized eigenproblem uses the GFN2
operators without a charge-dependent Hamiltonian modification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh

from .hybrid_basis import HybridAtomicBasis


@dataclass(frozen=True)
class BondOrbitalSeed:
    atoms: tuple[int, int]
    coefficients: np.ndarray
    site_populations: tuple[float, float]


@dataclass(frozen=True)
class PiRegionSeed:
    atoms: tuple[int, ...]
    site_orbitals: np.ndarray
    site_populations: np.ndarray
    polarized_orbital: np.ndarray

    def diagonalize(self, core: np.ndarray, overlap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Solve all pi-region orbitals in the physical GFN2 H/S projection."""
        h = self.site_orbitals.T @ core @ self.site_orbitals
        s = self.site_orbitals.T @ overlap @ self.site_orbitals
        values, vectors = eigh(h, s)
        return values, self.site_orbitals @ vectors


@dataclass(frozen=True)
class HybridRegionSeeds:
    bonds: tuple[BondOrbitalSeed, ...]
    pi_regions: tuple[PiRegionSeed, ...]


def _positive_site_populations(charges: np.ndarray, charge_weight: float) -> np.ndarray:
    centered = charges - np.mean(charges)
    raw = np.maximum(0.05, 1.0 - centered / charge_weight)
    return len(charges) * raw / np.sum(raw)


def _pi_components(plan) -> tuple[tuple[int, ...], ...]:
    pi_edges: set[tuple[int, int]] = set()
    for region in plan.regions:
        if region.kind == "DELOCALIZED_AROMATIC":
            atoms = set(region.atoms)
            pi_edges.update(
                (left, right) for left, right in plan.bonds
                if left in atoms and right in atoms
            )
        elif region.kind == "DELOCALIZED_PI" and len(region.atoms) == 2:
            pi_edges.add(tuple(region.atoms))
    adjacency: dict[int, set[int]] = {}
    for left, right in pi_edges:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    components = []
    while adjacency:
        start = min(adjacency)
        seen = {start}
        stack = [start]
        while stack:
            for neighbor in adjacency.pop(stack.pop(), ()):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(tuple(sorted(seen)))
    return tuple(components)


def _pi_direction(
    atom: int, component: tuple[int, ...], xyz: np.ndarray,
    adjacency: tuple[set[int], ...],
) -> np.ndarray | None:
    # Use all covalent neighbours to define the local plane, including H and
    # sigma substituents. A terminal pi site inherits its partner's plane.
    neighbors = sorted(adjacency[atom])
    if len(neighbors) < 2:
        partner = next((other for other in component if other in adjacency[atom]), None)
        if partner is not None:
            neighbors = sorted(adjacency[partner] - {atom})
            neighbors = [partner] + neighbors
            origin = xyz[partner]
        else:
            return None
    else:
        origin = xyz[atom]
    directions = xyz[neighbors] - origin
    _, singular, vh = np.linalg.svd(directions, full_matrices=True)
    if len(singular) < 2 or singular[1] < 1.0e-8:
        return None
    return vh[-1]


def build_hybrid_region_seeds(
    hybrid: HybridAtomicBasis,
    plan,
    coordinates_angstrom: np.ndarray,
    spherical_shell: tuple[str, ...],
    hybrid_overlap: np.ndarray,
    *,
    charge_weight: float = 1.0,
) -> HybridRegionSeeds:
    """Construct ORACLE-polarized initial bond and pi vectors in hybrid basis.

    The charge weight is a tunable *initialization* parameter, not an SCF
    threshold or an energy correction. A site with positive ORACLE charge has
    less initial electronic weight. Occupations remain the global SCF's job.
    """
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    charges = np.asarray(plan.electronegativity_charges, dtype=float)
    overlap = np.asarray(hybrid_overlap, dtype=float)
    if (
        charge_weight <= 0.0 or not np.isfinite(charge_weight)
        or xyz.shape != (len(plan.atomic_numbers), 3)
        or charges.shape != (len(plan.atomic_numbers),)
        or not np.all(np.isfinite(charges))
        or overlap.shape != (hybrid.size, hybrid.size)
        or len(spherical_shell) != hybrid.size
    ):
        raise ValueError("invalid charge-polarized hybrid region inputs")
    bond_seeds = []
    for left, right in plan.bonds:
        left_col = hybrid.bond_columns_by_atom[left].get(right)
        right_col = hybrid.bond_columns_by_atom[right].get(left)
        if left_col is None or right_col is None:
            continue
        left_index = int(hybrid.atom_orbitals[left][left_col])
        right_index = int(hybrid.atom_orbitals[right][right_col])
        populations = _positive_site_populations(charges[[left, right]], charge_weight)
        vector = np.zeros(hybrid.size)
        vector[left_index] = np.sqrt(populations[0])
        vector[right_index] = np.sqrt(populations[1])
        vector /= np.sqrt(vector @ overlap @ vector)
        bond_seeds.append(BondOrbitalSeed((left, right), vector, tuple(populations)))
    adjacency = tuple(set() for _ in plan.atomic_numbers)
    for left, right in plan.bonds:
        adjacency[left].add(right)
        adjacency[right].add(left)
    pi_regions = []
    for atoms in _pi_components(plan):
        site_vectors = []
        reference_direction = None
        for atom in atoms:
            direction = _pi_direction(atom, atoms, xyz, adjacency)
            indices = hybrid.atom_orbitals[atom]
            p_indices = [local for local, index in enumerate(indices)
                         if spherical_shell[index].endswith("p")]
            if direction is None or len(p_indices) < 3:
                break
            if reference_direction is not None and direction @ reference_direction < 0:
                direction = -direction
            if reference_direction is None:
                reference_direction = direction
            ao_vector = np.zeros(len(indices))
            ao_vector[p_indices[-3:]] = direction
            local_vector = hybrid.blocks[atom].T @ ao_vector
            full_vector = np.zeros(hybrid.size)
            full_vector[indices] = local_vector
            site_vectors.append(full_vector)
        if len(site_vectors) != len(atoms):
            continue  # Unsupported local angular topology; keep full hybrid basis.
        sites = np.column_stack(site_vectors)
        populations = _positive_site_populations(charges[list(atoms)], charge_weight)
        polarized = sites @ np.sqrt(populations)
        polarized /= np.sqrt(polarized @ overlap @ polarized)
        pi_regions.append(PiRegionSeed(atoms, sites, populations, polarized))
    return HybridRegionSeeds(tuple(bond_seeds), tuple(pi_regions))


@dataclass(frozen=True)
class BondPolarizedGuess:
    density: np.ndarray
    occupied_orbitals: np.ndarray
    bond_orbitals: int
    pi_orbitals: int


def bond_polarized_density_guess(
    seeds: HybridRegionSeeds,
    core: np.ndarray,
    overlap: np.ndarray,
    electrons: int,
) -> BondPolarizedGuess:
    """Fill polarized bonds, pi seeds, then residual lone-pair/core space.

    This is only the initial determinant. Global occupations and charge
    transfer are subsequently optimized by the full SCC fixed point.
    """
    size = len(overlap)
    occupied = electrons // 2
    if electrons % 2 or not 0 < occupied <= size:
        raise ValueError("bond-polarized guess requires a closed-shell system")
    metric = np.asarray(overlap, dtype=float)
    columns: list[np.ndarray] = []

    def append(candidate: np.ndarray) -> bool:
        vector = np.asarray(candidate, dtype=float).copy()
        for _ in range(2):
            for previous in columns:
                vector -= (previous @ metric @ vector) * previous
        norm2 = float(vector @ metric @ vector)
        if norm2 <= 1.0e-10:
            return False
        columns.append(vector / np.sqrt(norm2))
        return True

    for seed in seeds.bonds:
        if len(columns) >= occupied:
            break
        append(seed.coefficients)
    bond_count = len(columns)
    for region in seeds.pi_regions:
        if len(columns) >= occupied:
            break
        append(region.polarized_orbital)
    pi_count = len(columns) - bond_count
    _, canonical = eigh(np.asarray(core, dtype=float), metric)
    for index in range(size):
        if len(columns) >= occupied:
            break
        append(canonical[:, index])
    if len(columns) != occupied:
        raise RuntimeError("bond-polarized guess could not span occupied space")
    orbitals = np.column_stack(columns)
    return BondPolarizedGuess(
        2.0 * orbitals @ orbitals.T, orbitals, bond_count, pi_count,
    )
