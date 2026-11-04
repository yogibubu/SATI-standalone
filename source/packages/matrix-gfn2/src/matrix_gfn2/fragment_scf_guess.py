"""Neutral covalent-fragment SCF in the full system's SCC embedding field.

This constructs only a starting density. The subsequent global SCF retains all
interfragment couplings, charge redistribution, and the original convergence
criteria. Fragment electron counts are fixed only during the guess stage.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import linalg

from .fragment_guess import covalent_components
from .scc import fermi_occupations


def neutral_fragment_scf_guess(
    core: np.ndarray,
    overlap: np.ndarray,
    orbital_atoms: np.ndarray,
    atomic_numbers: np.ndarray,
    coordinates_bohr: np.ndarray,
    covalent_radii_bohr: np.ndarray,
    valence_electrons_by_atom: np.ndarray,
    temperature_kelvin: float,
    fock_and_energy: Callable[[np.ndarray], tuple[np.ndarray, float]],
    *,
    plan: object | None = None,
    maximum_cycles: int = 40,
    density_tolerance: float = 1.0e-5,
    energy_tolerance_hartree: float = 1.0e-6,
    maximum_fragment_orbitals: int = 96,
    maximum_interfragment_overlap: float = 0.10,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    """Converge block densities self-consistently in one shared SCC field.

    Each block is solved with its neutral valence electron count. Off-block AO
    density is zero during preparation, but the Fock builder sees all blocks
    together and includes their electrostatic, multipolar, and D4 response.
    """
    core = np.asarray(core, dtype=float)
    overlap = np.asarray(overlap, dtype=float)
    orbital_atoms = np.asarray(orbital_atoms, dtype=np.intp)
    valence = np.asarray(valence_electrons_by_atom, dtype=float)
    if core.shape != overlap.shape or core.ndim != 2 or core.shape[0] != core.shape[1]:
        raise ValueError("fragment SCF requires matching square Fock and overlap")
    if len(orbital_atoms) != len(core) or len(valence) != len(atomic_numbers):
        raise ValueError("fragment SCF atom and orbital dimensions disagree")
    if (maximum_cycles < 1 or density_tolerance <= 0.0
            or energy_tolerance_hartree <= 0.0
            or not 0.0 < maximum_interfragment_overlap < 1.0):
        raise ValueError("invalid fragment SCF convergence controls")
    if plan is None:
        components = covalent_components(
            atomic_numbers, coordinates_bohr, covalent_radii_bohr
        )
    else:
        if (
            getattr(plan, "provider", None) != "ORACLE_GEOMETRY_TOPOLOGY"
            or tuple(getattr(plan, "atomic_numbers", ()))
            != tuple(int(z) for z in atomic_numbers)
        ):
            raise ValueError("ORACLE fragment plan does not match the atom frame")
        parent = list(range(len(atomic_numbers)))

        def root(atom: int) -> int:
            while parent[atom] != atom:
                parent[atom] = parent[parent[atom]]
                atom = parent[atom]
            return atom

        for left, right in plan.bonds:
            left, right = int(left), int(right)
            if not (0 <= left < len(parent) and 0 <= right < len(parent)):
                raise ValueError("ORACLE fragment bond is outside the atom frame")
            parent[root(right)] = root(left)
        groups: dict[int, list[int]] = {}
        for atom in range(len(parent)):
            groups.setdefault(root(atom), []).append(atom)
        components = [np.asarray(atoms, dtype=np.intp) for atoms in groups.values()]
    if len(components) < 2:
        raise ValueError("neutral fragment SCF needs multiple covalent components")
    blocks: list[tuple[np.ndarray, np.ndarray, int]] = []
    for atoms in components:
        orbitals = np.flatnonzero(np.isin(orbital_atoms, atoms))
        electrons_real = float(np.sum(valence[atoms]))
        electrons = int(round(electrons_real))
        if (
            not np.isclose(electrons_real, electrons, atol=1.0e-8)
            or electrons < 2 or electrons % 2
            or len(orbitals) > maximum_fragment_orbitals
            or electrons > 2 * len(orbitals)
        ):
            raise ValueError("a covalent component is not an eligible neutral closed-shell fragment")
        blocks.append((orbitals, overlap[np.ix_(orbitals, orbitals)], electrons))

    # Neutral block occupations are a useful starting point only when the
    # component AO spaces are weakly coupled. Strong hydrogen-bond overlap can
    # otherwise leave the global SCC mixer on a long charge-transfer plateau.
    orbital_component = np.empty(len(orbital_atoms), dtype=np.intp)
    for index, (orbitals, _, _) in enumerate(blocks):
        orbital_component[orbitals] = index
    largest_cross_overlap = 0.0
    for index, (orbitals, _, _) in enumerate(blocks):
        other = np.flatnonzero(orbital_component != index)
        if len(other):
            largest_cross_overlap = max(
                largest_cross_overlap,
                float(np.max(np.abs(overlap[np.ix_(orbitals, other)]))),
            )
        if largest_cross_overlap > maximum_interfragment_overlap:
            raise ValueError(
                "neutral fragment guess rejected: interfragment AO overlap "
                f"{largest_cross_overlap:.3f} exceeds "
                f"{maximum_interfragment_overlap:.3f}"
            )

    density = np.zeros_like(core)

    def local_density(fock: np.ndarray) -> np.ndarray:
        result = np.zeros_like(core)
        for orbitals, local_overlap, electrons in blocks:
            values, vectors = linalg.eigh(
                fock[np.ix_(orbitals, orbitals)], local_overlap,
                driver="gvd", check_finite=False,
            )
            occupations = fermi_occupations(
                values, electrons, multiplicity=1,
                temperature_kelvin=temperature_kelvin,
            )
            result[np.ix_(orbitals, orbitals)] = (
                vectors * occupations
            ) @ vectors.T
        return result

    density = local_density(core)
    previous_energy = np.inf
    previous_residual = np.inf
    for cycle in range(1, maximum_cycles + 1):
        fock, energy = fock_and_energy(density)
        output = local_density(fock)
        residual = float(np.max(np.abs(output - density)))
        energy_change = abs(float(energy) - previous_energy)
        if residual < density_tolerance and energy_change < energy_tolerance_hartree:
            return output, {
                "converged": True,
                "cycles": cycle,
                "components": len(blocks),
                "density_residual": residual,
                "energy_residual_hartree": energy_change,
            }
        damping = 0.4 if residual > 1.1 * previous_residual else 0.7
        density = (1.0 - damping) * density + damping * output
        previous_residual = residual
        previous_energy = float(energy)
    # A guess need not satisfy the final SCF thresholds.  Return the last
    # neutral block solution; the subsequent *global* SCF still certifies the
    # requested energy and multipole tolerances.
    return output, {
        "converged": False,
        "cycles": maximum_cycles,
        "components": len(blocks),
        "density_residual": residual,
        "energy_residual_hartree": energy_change,
    }
