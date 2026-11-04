"""Consume an ORACLE region plan to make a local-H/S dense SCF seed.

The region plan is owned by ORACLE. This module performs only electronic
operations and never infers bonds, aromaticity, or lone-pair identities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg

from .scc import fermi_occupations


@dataclass(frozen=True)
class LocalOrbitalRegion:
    """Persistent ORACLE support and its first-step generalized eigenbasis."""

    region_id: int
    kind: str
    atoms: tuple[int, ...]
    orbitals: tuple[int, ...]
    eigenvalues_hartree: np.ndarray
    coefficients: np.ndarray
    overlap: np.ndarray
    occupations: np.ndarray
    assembly_weights: np.ndarray


def oracle_region_density_guess(
    core: np.ndarray,
    overlap: np.ndarray,
    orbital_atoms: np.ndarray,
    orbital_shell: np.ndarray,
    reference_population: np.ndarray,
    atomic_numbers: np.ndarray,
    electrons: int,
    temperature_kelvin: float,
    plan: object,
    *,
    maximum_region_orbitals: int = 64,
    minimum_large_component_orbitals: int = 96,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    dict[str, int], tuple[LocalOrbitalRegion, ...],
]:
    """Solve regional generalized eigenproblems and blend positive densities.

    Each region inherits population from a neutral-shell AO seed. The same
    orbital may occur in several overlapping regions, so a symmetric
    partition of unity prevents double counting. The resulting density is
    *only* a guess; the complete global Fock and SCF follow unchanged.
    """
    core = np.asarray(core, dtype=float)
    overlap = np.asarray(overlap, dtype=float)
    orbital_atoms = np.asarray(orbital_atoms, dtype=np.intp)
    orbital_shell = np.asarray(orbital_shell, dtype=np.intp)
    populations = np.asarray(reference_population, dtype=float)
    numbers = tuple(int(z) for z in atomic_numbers)
    if (
        core.shape != overlap.shape or core.ndim != 2
        or core.shape[0] != core.shape[1]
        or len(orbital_atoms) != len(core)
        or len(orbital_shell) != len(core)
        or getattr(plan, "provider", None) != "ORACLE_GEOMETRY_TOPOLOGY"
        or tuple(getattr(plan, "atomic_numbers", ())) != numbers
    ):
        raise ValueError("ORACLE region plan does not match the GFN2 AO model")
    if maximum_region_orbitals < 2 or minimum_large_component_orbitals < 1:
        raise ValueError("invalid ORACLE region-size policy")
    if np.any(orbital_atoms < 0) or np.any(orbital_atoms >= len(numbers)):
        raise ValueError("ORACLE region orbital-to-atom map is invalid")
    if np.any(orbital_shell < 0) or np.any(orbital_shell >= len(populations)):
        raise ValueError("ORACLE region orbital-to-shell map is invalid")

    parent = list(range(len(numbers)))

    def root(atom: int) -> int:
        while parent[atom] != atom:
            parent[atom] = parent[parent[atom]]
            atom = parent[atom]
        return atom

    for left, right in plan.bonds:
        if not (0 <= left < len(numbers) and 0 <= right < len(numbers)):
            raise ValueError("ORACLE region bond index is outside the molecule")
        parent[root(int(right))] = root(int(left))
    component_sizes: dict[int, int] = {}
    for atom in orbital_atoms:
        key = root(int(atom))
        component_sizes[key] = component_sizes.get(key, 0) + 1
    largest_component = max(component_sizes.values(), default=0)
    if largest_component <= minimum_large_component_orbitals:
        raise ValueError("all ORACLE covalent components fit the fragment-SCF route")

    widths = np.bincount(orbital_shell, minlength=len(populations))
    diagonal = populations[orbital_shell] / widths[orbital_shell]
    seed = np.diag(diagonal / np.diag(overlap))
    coverage = np.zeros(len(core), dtype=np.int32)
    atom_orbitals = [[] for _ in numbers]
    for orbital, atom in enumerate(orbital_atoms):
        atom_orbitals[int(atom)].append(orbital)
    supports: list[tuple[int, str, tuple[int, ...], np.ndarray]] = []
    kinds: dict[str, int] = {}
    for region_id, region in enumerate(plan.regions):
        atoms = tuple(int(a) for a in region.atoms)
        if not atoms or any(atom < 0 or atom >= len(numbers) for atom in atoms):
            raise ValueError("ORACLE region contains an invalid atom")
        orbitals = np.asarray(
            sorted(orbital for atom in set(atoms) for orbital in atom_orbitals[atom]),
            dtype=np.intp,
        )
        if len(orbitals) > maximum_region_orbitals:
            raise ValueError("an ORACLE delocalized region exceeds the AO size cap")
        if len(orbitals) == 0:
            continue
        supports.append((region_id, str(region.kind), atoms, orbitals))
        coverage[orbitals] += 1
        kinds[str(region.kind)] = kinds.get(str(region.kind), 0) + 1
    if not supports or np.any(coverage == 0):
        raise ValueError("ORACLE region plan does not cover the AO basis")

    weights = 1.0 / np.sqrt(coverage.astype(float))
    density = np.zeros_like(core)
    regional_orbitals: list[LocalOrbitalRegion] = []
    for region_id, kind, atoms, orbitals in supports:
        local_overlap = overlap[np.ix_(orbitals, orbitals)]
        values, vectors = linalg.eigh(
            core[np.ix_(orbitals, orbitals)], local_overlap,
            driver="gvd", check_finite=False,
        )
        local_seed = seed[np.ix_(orbitals, orbitals)]
        projected = local_overlap @ local_seed @ local_overlap
        occupations = np.clip(
            np.einsum("ij,ij->j", vectors, projected @ vectors), 0.0, 2.0
        )
        local_density = (vectors * occupations) @ vectors.T
        local_weights = weights[orbitals]
        density[np.ix_(orbitals, orbitals)] += (
            local_weights[:, None] * local_density * local_weights[None, :]
        )
        stored = [
            np.asarray(array, dtype=float).copy()
            for array in (values, vectors, local_overlap, occupations, local_weights)
        ]
        for array in stored:
            array.setflags(write=False)
        regional_orbitals.append(LocalOrbitalRegion(
            region_id, kind, atoms, tuple(int(a) for a in orbitals), *stored,
        ))
    density = 0.5 * (density + density.T)
    population = float(np.sum(density * overlap))
    if not np.isfinite(population) or population <= 0.0:
        raise ValueError("ORACLE regional density has invalid electron population")
    density *= electrons / population
    if not np.all(np.isfinite(density)):
        raise ValueError("ORACLE regional density is not finite")
    # These initialize only the policy's gap/temperature estimate. The first
    # full Fock solve supplies actual molecular orbital energies and vectors.
    energies = np.sort(np.diag(core).copy())
    initial_occupations = fermi_occupations(
        energies, electrons, multiplicity=1,
        temperature_kelvin=temperature_kelvin,
    )
    return density, energies, np.eye(len(core)), initial_occupations, {
        "regions": len(supports),
        "largest_component_orbitals": largest_component,
        **{f"kind_{kind}": count for kind, count in kinds.items()},
    }, tuple(regional_orbitals)
