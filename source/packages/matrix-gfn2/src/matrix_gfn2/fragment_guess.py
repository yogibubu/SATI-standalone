"""Topology-based initial densities for full-system SCC experiments.

No fragment charge is imposed: a single global Fermi occupation distributes
the electrons across the block eigenstates.  The full Fock and overlap are
restored in the first SCC step, so this only changes the starting density.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg as scipy_linalg
from scipy.spatial import cKDTree

from .scc import fermi_occupations


def covalent_components(
    atomic_numbers: np.ndarray,
    coordinates_bohr: np.ndarray,
    covalent_radii_bohr: np.ndarray,
) -> list[np.ndarray]:
    """Partition atoms by covalent links, without using a water-specific rule."""

    numbers = np.asarray(atomic_numbers, dtype=np.intp)
    coordinates = np.asarray(coordinates_bohr, dtype=float)
    radii = np.asarray(covalent_radii_bohr, dtype=float)[numbers - 1]
    if len(numbers) != len(coordinates):
        raise ValueError("atom numbers and coordinates differ in length")
    parent = np.arange(len(numbers), dtype=np.intp)

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    if len(numbers):
        candidates = cKDTree(coordinates).query_pairs(
            2.5 * float(np.max(radii)), output_type="ndarray"
        )
        for left, right in candidates:
            if np.linalg.norm(coordinates[left] - coordinates[right]) > 1.25 * (
                radii[left] + radii[right]
            ):
                continue
            left_root, right_root = root(int(left)), root(int(right))
            if left_root != right_root:
                parent[right_root] = left_root
    groups: dict[int, list[int]] = {}
    for index in range(len(numbers)):
        groups.setdefault(root(index), []).append(index)
    return [np.asarray(group, dtype=np.intp) for group in groups.values()]


def topology_block_density_guess(
    core: np.ndarray,
    overlap: np.ndarray,
    orbital_atoms: np.ndarray,
    atomic_numbers: np.ndarray,
    coordinates_bohr: np.ndarray,
    covalent_radii_bohr: np.ndarray,
    electrons: int,
    multiplicity: int,
    temperature_kelvin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Diagonalize covalent blocks, then fill all states at one chemical potential."""

    components = covalent_components(
        atomic_numbers, coordinates_bohr, covalent_radii_bohr
    )
    orbital_atoms = np.asarray(orbital_atoms, dtype=np.intp)
    size = len(orbital_atoms)
    if len(components) < 2:
        raise ValueError("no separable topological blocks")
    blocks = [np.flatnonzero(np.isin(orbital_atoms, atoms)) for atoms in components]
    if sum(len(block) for block in blocks) != size or max(map(len, blocks)) >= 0.8 * size:
        raise ValueError("topological blocks do not cover a separable basis")
    block_solutions: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    eigenvalues = np.empty(size)
    cursor = 0
    for orbitals in blocks:
        local_core = core[np.ix_(orbitals, orbitals)]
        local_overlap = overlap[np.ix_(orbitals, orbitals)]
        values, vectors = scipy_linalg.eigh(
            local_core, local_overlap, check_finite=False, driver="gvd"
        )
        eigenvalues[cursor : cursor + len(values)] = values
        block_solutions.append((orbitals, values, vectors))
        cursor += len(values)
    order = np.argsort(eigenvalues, kind="stable")
    sorted_energies = eigenvalues[order]
    sorted_occupations = fermi_occupations(
        sorted_energies, electrons, multiplicity=multiplicity,
        temperature_kelvin=temperature_kelvin,
    )
    unsorted_occupations = np.empty_like(sorted_occupations)
    unsorted_occupations[order] = sorted_occupations
    density = np.zeros_like(core)
    coefficients = np.zeros_like(core)
    cursor = 0
    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(size)
    for orbitals, values, vectors in block_solutions:
        width = len(values)
        local_occ = unsorted_occupations[cursor : cursor + width]
        density[np.ix_(orbitals, orbitals)] = (
            vectors * local_occ
        ) @ vectors.T
        coefficients[np.ix_(orbitals, inverse_order[cursor : cursor + width])] = vectors
        cursor += width
    return density, sorted_energies, coefficients, sorted_occupations, len(blocks)


def overlapping_fragment_density_guess(
    core: np.ndarray,
    overlap: np.ndarray,
    orbital_atoms: np.ndarray,
    atomic_numbers: np.ndarray,
    coordinates_bohr: np.ndarray,
    covalent_radii_bohr: np.ndarray,
    valence_electrons_by_atom: np.ndarray,
    electrons: int,
    multiplicity: int,
    temperature_kelvin: float,
    *,
    neighbor_cutoff_bohr: float = 6.5,
    maximum_fragment_orbitals: int = 96,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Blend overlapping local core densities with an AO partition of unity.

    Every covalent component is a center.  Nearby components enter its local
    eigenproblem but are shared with their own neighborhoods; symmetric AO
    weights prevent a component appearing multiple times in the assembled
    density.  The global S-trace is then normalized to the requested electron
    count.  This is only an initial guess: SCC uses the unmodified full model.
    """

    core = np.asarray(core, dtype=float)
    overlap = np.asarray(overlap, dtype=float)
    orbital_atoms = np.asarray(orbital_atoms, dtype=np.intp)
    numbers = np.asarray(atomic_numbers, dtype=np.intp)
    coordinates = np.asarray(coordinates_bohr, dtype=float)
    valence = np.asarray(valence_electrons_by_atom, dtype=float)
    if (
        core.shape != overlap.shape
        or core.ndim != 2
        or core.shape[0] != core.shape[1]
        or len(orbital_atoms) != len(core)
        or len(numbers) != len(coordinates)
        or len(valence) != len(numbers)
        or np.any(orbital_atoms < 0)
        or np.any(orbital_atoms >= len(numbers))
    ):
        raise ValueError("incompatible overlapping-fragment model dimensions")
    if neighbor_cutoff_bohr <= 0.0 or maximum_fragment_orbitals < 2:
        raise ValueError("invalid overlapping-fragment locality policy")
    if electrons <= 0 or electrons > 2 * len(core):
        raise ValueError("invalid overlapping-fragment electron count")

    components = covalent_components(numbers, coordinates, covalent_radii_bohr)
    if len(components) < 2:
        raise ValueError("no separable covalent components for overlapping guess")
    atom_component = np.empty(len(numbers), dtype=np.intp)
    for index, atoms in enumerate(components):
        atom_component[atoms] = index
    orbital_component = atom_component[orbital_atoms]
    component_orbitals = [
        np.flatnonzero(orbital_component == index)
        for index in range(len(components))
    ]
    if any(len(item) > maximum_fragment_orbitals for item in component_orbitals):
        raise ValueError("a covalent component exceeds the local fragment limit")

    # Retain the nonoverlapping block spectrum for the existing frontier-gap
    # and temperature schedule.  Only its initial density is replaced below.
    _, energies, coefficients, occupations, _ = topology_block_density_guess(
        core, overlap, orbital_atoms, numbers, coordinates, covalent_radii_bohr,
        electrons, multiplicity, temperature_kelvin,
    )
    neighbors: list[dict[int, float]] = [
        {} for _ in range(len(components))
    ]
    for left_atom, right_atom in cKDTree(coordinates).query_pairs(
        neighbor_cutoff_bohr, output_type="ndarray"
    ):
        left = int(atom_component[left_atom])
        right = int(atom_component[right_atom])
        if left == right:
            continue
        distance = float(np.linalg.norm(
            coordinates[left_atom] - coordinates[right_atom]
        ))
        previous = neighbors[left].get(right, np.inf)
        if distance < previous:
            neighbors[left][right] = distance
            neighbors[right][left] = distance

    fragments: list[tuple[np.ndarray, np.ndarray]] = []
    for center, central_atoms in enumerate(components):
        selected = [center]
        width = len(component_orbitals[center])
        for neighbor, _ in sorted(neighbors[center].items(), key=lambda item: (item[1], item[0])):
            extra = len(component_orbitals[neighbor])
            if width + extra <= maximum_fragment_orbitals:
                selected.append(neighbor)
                width += extra
        atoms = np.concatenate([components[index] for index in selected])
        orbitals = np.sort(np.concatenate([component_orbitals[index] for index in selected]))
        fragments.append((atoms, orbitals))
    if not any(len(atoms) > len(components[index]) for index, (atoms, _) in enumerate(fragments)):
        raise ValueError("no overlapping neighboring fragments were found")

    coverage = np.zeros(len(core), dtype=np.int32)
    for _, orbitals in fragments:
        coverage[orbitals] += 1
    if np.any(coverage == 0):
        raise ValueError("fragment neighborhoods do not cover the AO basis")
    weights = 1.0 / np.sqrt(coverage.astype(float))
    density = np.zeros_like(core)
    for atoms, orbitals in fragments:
        local_electrons = int(round(float(np.sum(valence[atoms]))))
        if not 0 < local_electrons <= 2 * len(orbitals):
            raise ValueError("invalid local fragment electron count")
        local_multiplicity = 1 if local_electrons % 2 == 0 else 2
        values, vectors = scipy_linalg.eigh(
            core[np.ix_(orbitals, orbitals)],
            overlap[np.ix_(orbitals, orbitals)],
            check_finite=False, driver="gvd",
        )
        local_occupations = fermi_occupations(
            values, local_electrons, multiplicity=local_multiplicity,
            temperature_kelvin=temperature_kelvin,
        )
        occupied = local_occupations > 0.0
        weighted_vectors = vectors[:, occupied] * np.sqrt(local_occupations[occupied])
        local_density = weighted_vectors @ weighted_vectors.T
        local_weights = weights[orbitals]
        density[np.ix_(orbitals, orbitals)] += (
            local_weights[:, None] * local_density * local_weights[None, :]
        )
    density = 0.5 * (density + density.T)
    population = float(np.sum(density * overlap))
    if not np.isfinite(population) or population <= np.finfo(float).tiny:
        raise ValueError("overlapping-fragment density has invalid population")
    density *= electrons / population
    if not np.all(np.isfinite(density)):
        raise ValueError("overlapping-fragment density is not finite")
    return density, energies, coefficients, occupations, len(fragments)
