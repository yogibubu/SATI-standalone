"""Capped, charge-embedded fragment SCF as a full-system starting state.

The fragment orbitals are retained for later coupling, but the final global
SCF and analytic gradient remain unchanged. This route is experimental.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .parameters import ELEMENT_SYMBOLS
from .scc import fermi_occupations
from .fragment_rigid_cache import (
    fragment_shape_key,
    rigid_equivalence_rotation,
    rotate_fragment_aos,
)
from .persistent_orbital_support import FragmentSupportWorkspace


@dataclass(frozen=True)
class FragmentOrbitalState:
    atoms: tuple[int, ...]
    boundary_bonds: tuple[tuple[int, int], ...]
    global_orbitals: tuple[int, ...]
    coefficients_on_real_atoms: np.ndarray
    orbital_energies_hartree: np.ndarray
    alpha_occupations: np.ndarray
    beta_occupations: np.ndarray
    fragment_iterations: int


def embedded_fragment_scf_guess(
    core: np.ndarray,
    overlap: np.ndarray,
    orbital_atoms: np.ndarray,
    electrons: int,
    temperature_kelvin: float,
    frames: tuple[object, ...],
    *,
    fragment_mixer_tolerance: float = 2.0e-3,
    fragment_energy_tolerance_hartree: float = 1.0e-4,
    maximum_fragment_cycles: int = 4,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    dict[str, float | int], tuple[FragmentOrbitalState, ...],
]:
    """Solve neutral capped fragments and project real-atom densities globally."""
    from .calculator import ANGSTROM_TO_BOHR, GFN2Calculator

    core = np.asarray(core, dtype=float)
    overlap = np.asarray(overlap, dtype=float)
    orbital_atoms = np.asarray(orbital_atoms, dtype=np.intp)
    if (
        core.ndim != 2 or core.shape != overlap.shape
        or core.shape[0] != core.shape[1]
        or len(orbital_atoms) != len(core)
        or not frames or maximum_fragment_cycles < 1
    ):
        raise ValueError("invalid embedded fragment SCF model")
    atom_count = max(int(np.max(orbital_atoms, initial=-1)) + 1,
                     max(max(frame.real_atoms, default=-1) for frame in frames) + 1)
    global_atom_orbitals = [np.flatnonzero(orbital_atoms == atom) for atom in range(atom_count)]
    states: list[FragmentOrbitalState] = []
    maximum_local_orbitals = 0
    total_fragment_cycles = 0
    reused_fragments = 0
    cached: dict[tuple[object, ...], list[tuple[object, np.ndarray, np.ndarray, np.ndarray,
                                             np.ndarray, np.ndarray, np.ndarray,
                                             tuple[str, ...], int]]] = {}
    used_orbitals = np.zeros(len(core), dtype=bool)
    for frame in frames:
        key = fragment_shape_key(frame)
        reuse = next(
            ((item, rotation)
             for item in cached.get(key, ())
             if (rotation := rigid_equivalence_rotation(item[0], frame)) is not None),
            None,
        )
        if reuse is None:
            symbols = tuple(ELEMENT_SYMBOLS[number - 1] for number in frame.atomic_numbers)
            calculator = GFN2Calculator(
                symbols,
                scf_convergence_mode="tblite",
                scf_mixer_tolerance=fragment_mixer_tolerance,
                scf_energy_tolerance_hartree=fragment_energy_tolerance_hartree,
                maximum_scf_iterations=maximum_fragment_cycles,
                point_charges_electron=frame.embedding_charges_electron,
                point_charge_coordinates_angstrom=frame.embedding_coordinates_angstrom,
                temperature_kelvin=temperature_kelvin,
            )
            calculator._fragment_scf_child = True
            # A fragment is only a preparatory environment, not the final
            # energy model. The complete parent SCC certifies convergence.
            result = calculator._single_point(
                frame.coordinates_angstrom * ANGSTROM_TO_BOHR,
                compute_bond_orders=False,
                collect_telemetry=False,
                fragment_orbitals_only=True,
            )
            local_map = np.asarray(calculator._basis_template.spherical_atom, dtype=np.intp)
            local_shell = calculator._basis_template.spherical_shell
            local_density = np.asarray(result.density, dtype=float)
            full_coefficients = np.asarray(result.coefficients, dtype=float)
            eigenvalues = np.asarray(result.orbital_energies_hartree, dtype=float)
            alpha = 0.5 * np.asarray(result.occupations, dtype=float)
            beta = alpha
            fragment_iterations = result.iterations
            cached.setdefault(key, []).append((
                frame, local_map, local_density.copy(), full_coefficients.copy(),
                eigenvalues.copy(), alpha.copy(), beta.copy(), local_shell,
                fragment_iterations,
            ))
        else:
            source, rotation = reuse
            _, local_map, source_density, source_coefficients, eigenvalues, alpha, beta, local_shell, _ = source
            ao_rotation = rotate_fragment_aos(rotation, local_map, local_shell)
            local_density = ao_rotation @ source_density @ ao_rotation.T
            full_coefficients = ao_rotation @ source_coefficients
            fragment_iterations = 0
            reused_fragments += 1
        local_real = np.flatnonzero(local_map < len(frame.real_atoms))
        global_real = np.concatenate([global_atom_orbitals[atom] for atom in frame.real_atoms])
        if len(local_real) != len(global_real) or np.any(used_orbitals[global_real]):
            raise ValueError("embedded fragment AO mapping is not disjoint")
        used_orbitals[global_real] = True
        coefficients = np.asarray(full_coefficients[local_real], dtype=float).copy()
        eigenvalues = np.asarray(eigenvalues, dtype=float).copy()
        alpha = np.asarray(alpha, dtype=float).copy()
        beta = np.asarray(beta, dtype=float).copy()
        for array in (coefficients, eigenvalues, alpha, beta):
            array.setflags(write=False)
        states.append(FragmentOrbitalState(
            frame.real_atoms, frame.link_bonds,
            tuple(int(orbital) for orbital in global_real),
            coefficients, eigenvalues, alpha, beta, fragment_iterations,
        ))
        maximum_local_orbitals = max(maximum_local_orbitals, len(local_map))
        total_fragment_cycles += fragment_iterations
    if not np.all(used_orbitals):
        raise ValueError("embedded fragments do not cover the global AO basis")
    support = FragmentSupportWorkspace.from_states(tuple(states), len(core))
    density, _ = support.native_operands(core)
    density = 0.5 * (density + density.T)
    population = float(np.sum(density * overlap))
    if not np.isfinite(population) or population <= 0.0:
        raise ValueError("embedded fragment density has invalid population")
    density *= electrons / population
    energies = np.sort(np.diag(core).copy())
    occupations = fermi_occupations(
        energies, electrons, multiplicity=1,
        temperature_kelvin=temperature_kelvin,
    )
    return density, energies, np.eye(len(core)), occupations, {
        "fragments": len(frames),
        "maximum_fragment_orbitals": maximum_local_orbitals,
        "total_fragment_cycles": total_fragment_cycles,
        "reused_equivalent_fragments": reused_fragments,
        "fragment_support_nnz": int(support.coefficients.nnz),
        "fragment_support_density_nnz": int(np.count_nonzero(density)),
        "initial_population_before_normalization": population,
    }, tuple(states)
