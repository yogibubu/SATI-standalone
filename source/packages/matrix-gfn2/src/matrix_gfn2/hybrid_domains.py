"""Persistent ORACLE fragment supports shared by orbital rotations and PNO work.

The domains are geometric supports, not pair natural orbitals. Actual PNOs
require a correlated pair density and are deliberately not fabricated here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PersistentHybridDomains:
    atoms: tuple[tuple[int, ...], ...]
    orbitals: tuple[np.ndarray, ...]
    centers_angstrom: np.ndarray
    orbital_fragment: np.ndarray
    boundary_bonds: tuple[tuple[tuple[int, int], ...], ...]
    coordinates_angstrom: np.ndarray
    atom_fragment: np.ndarray

    def nearby_fragments(self, fragment: int, radius_angstrom: float) -> tuple[int, ...]:
        if fragment < 0 or fragment >= len(self.atoms) or radius_angstrom < 0:
            raise ValueError("invalid persistent domain query")
        # The minimum atom-atom distance, not centroid separation, controls
        # locality for elongated covalent fragments and anisotropic systems.
        members = self.coordinates_angstrom[list(self.atoms[fragment])]
        distance = np.linalg.norm(
            self.coordinates_angstrom[:, None, :] - members[None, :, :], axis=2
        ).min(axis=1)
        return tuple(int(index) for index in np.unique(self.atom_fragment[
            distance <= radius_angstrom + 1.0e-12
        ]))

    def pair_domain(
        self, left_fragment: int, right_fragment: int, radius_angstrom: float,
    ) -> np.ndarray:
        """AO/hybrid index support for a later correlated pair density."""
        chosen = set(self.nearby_fragments(left_fragment, radius_angstrom))
        chosen.update(self.nearby_fragments(right_fragment, radius_angstrom))
        return np.sort(np.concatenate([self.orbitals[index] for index in sorted(chosen)]))


def build_persistent_hybrid_domains(
    fragments,
    orbital_atoms: np.ndarray,
    coordinates_angstrom: np.ndarray,
) -> PersistentHybridDomains:
    """Freeze ORACLE fragment identities for SCF and subsequent pair domains."""
    assignment = np.asarray(orbital_atoms, dtype=np.intp)
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    atoms = tuple(tuple(int(atom) for atom in item.atoms) for item in fragments)
    if not atoms or xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("invalid ORACLE persistent fragment inputs")
    atom_owner = np.full(len(xyz), -1, dtype=np.intp)
    for fragment, members in enumerate(atoms):
        for atom in members:
            if atom < 0 or atom >= len(xyz) or atom_owner[atom] >= 0:
                raise ValueError("fragments overlap or contain an invalid atom")
            atom_owner[atom] = fragment
    if np.any(atom_owner < 0) or np.any(assignment < 0) or np.any(assignment >= len(xyz)):
        raise ValueError("fragments do not cover all orbital atoms")
    owner = atom_owner[assignment]
    orbitals = tuple(np.flatnonzero(owner == index) for index in range(len(atoms)))
    centers = np.asarray([np.mean(xyz[list(members)], axis=0) for members in atoms])
    return PersistentHybridDomains(
        atoms, orbitals, centers, owner,
        tuple(tuple(item.boundary_bonds) for item in fragments),
        xyz, atom_owner,
    )
