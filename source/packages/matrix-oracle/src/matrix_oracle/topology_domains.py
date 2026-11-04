"""ORACLE ownership of locally certain and uncertain topology domains.

The optimization layer must not infer chemical regions from optimizer
failures.  This module turns ORACLE's graph, ring, and reactive-site evidence
into an explicit atom partition that SMITH/LINK can consume mechanically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


ORACLE_TOPOLOGY_DOMAIN_SCHEMA = "matrix.oracle.topology_domain.v1"


@dataclass(frozen=True)
class TopologyDomainAssessment:
    """One geometry-local ORACLE partition; atom indices are zero based."""

    task_regime: str
    stable_atoms: tuple[int, ...]
    uncertain_atoms: tuple[int, ...]
    boundary_atoms: tuple[int, ...]
    topology_well_defined: bool
    geometry_near_stationary: bool | None
    gradient_inf_hartree_per_bohr: float | None
    reasons: tuple[str, ...]
    bonds: tuple[tuple[int, int], ...]
    schema: str = ORACLE_TOPOLOGY_DOMAIN_SCHEMA

    def __post_init__(self) -> None:
        regime = str(self.task_regime).strip().upper().replace("-", "_")
        if regime not in {"MINIMUM", "TRANSITION_STATE"}:
            raise ValueError("topology domains require MINIMUM or TRANSITION_STATE")
        stable = tuple(sorted({int(atom) for atom in self.stable_atoms}))
        uncertain = tuple(sorted({int(atom) for atom in self.uncertain_atoms}))
        boundary = tuple(sorted({int(atom) for atom in self.boundary_atoms}))
        if set(stable).intersection(uncertain):
            raise ValueError("stable and uncertain topology atoms must be disjoint")
        if not set(boundary).issubset(stable):
            raise ValueError("the topology boundary must belong to the stable domain")
        if any(atom < 0 for atom in (*stable, *uncertain, *boundary)):
            raise ValueError("topology-domain atom indices must be non-negative")
        object.__setattr__(self, "task_regime", regime)
        object.__setattr__(self, "stable_atoms", stable)
        object.__setattr__(self, "uncertain_atoms", uncertain)
        object.__setattr__(self, "boundary_atoms", boundary)
        object.__setattr__(
            self,
            "bonds",
            tuple(sorted({tuple(sorted(map(int, bond))) for bond in self.bonds})),
        )
        object.__setattr__(self, "reasons", tuple(str(item) for item in self.reasons))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "task_regime": self.task_regime,
            "stable_atoms_zero_based": list(self.stable_atoms),
            "uncertain_atoms_zero_based": list(self.uncertain_atoms),
            "boundary_atoms_zero_based": list(self.boundary_atoms),
            "topology_well_defined": self.topology_well_defined,
            "geometry_near_stationary": self.geometry_near_stationary,
            "gradient_inf_hartree_per_bohr": self.gradient_inf_hartree_per_bohr,
            "reasons": list(self.reasons),
            "bonds_zero_based": [list(pair) for pair in self.bonds],
        }


def assess_topology_domains(
    atomic_numbers: Sequence[int],
    coordinates_angstrom: np.ndarray,
    *,
    task_regime: str = "MINIMUM",
    gradient_hartree_per_bohr: np.ndarray | Sequence[float] | None = None,
    reactive_atom_indices: Iterable[int] = (),
    additional_uncertain_atom_indices: Iterable[int] = (),
    gradient_stationary_threshold: float = 0.50,
    boundary_hops: int = 1,
) -> TopologyDomainAssessment:
    """Partition a molecular graph using local ORACLE chemical evidence.

    Partial metal--ring hapticity is the first important geometry-first case:
    individual metal--donor spokes are not a stable topology, whereas a fully
    perceived eta ring is.  Explicit reactive atoms use the identical contract
    for transition-state work.  Callers may add atoms identified by another
    ORACLE uncertainty provider without changing downstream code.
    """

    numbers = tuple(int(value) for value in atomic_numbers)
    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    if (
        not numbers
        or coordinates.shape != (len(numbers), 3)
        or not np.all(np.isfinite(coordinates))
    ):
        raise ValueError("topology-domain coordinates must be finite natoms by 3")
    hops = int(boundary_hops)
    if hops < 0:
        raise ValueError("topology-domain boundary_hops must be non-negative")
    if float(gradient_stationary_threshold) <= 0.0:
        raise ValueError("gradient stationary threshold must be positive")

    from matrix_chem import build_topology_objects
    from matrix_chem.topology.metals import is_metal_atomic_number

    _continuous, discrete, rings, _synthons, _aromaticity = build_topology_objects(
        coordinates, numbers
    )
    bonds = tuple(sorted(tuple(sorted(map(int, pair))) for pair in discrete.bonds))
    bond_set = set(bonds)
    reactive = tuple(int(atom) for atom in reactive_atom_indices)
    additional = tuple(int(atom) for atom in additional_uncertain_atom_indices)
    uncertain = {int(atom) for atom in (*reactive, *additional)}
    if any(atom < 0 or atom >= len(numbers) for atom in uncertain):
        raise ValueError("topology-domain seed atom lies outside the molecular frame")
    reasons: list[str] = []
    if reactive:
        reasons.append("ORACLE_REACTIVE_DOMAIN")
    if additional:
        reasons.append("ORACLE_ADDITIONAL_UNCERTAINTY_PROVIDER")
    if len(numbers) > 1 and not bonds:
        # A topology-free aggregate (e.g. a Lennard-Jones cluster) is a valid
        # optimization target, but it has no unique covalent internal chart.
        # Keep it in the topology-independent route rather than manufacturing
        # bonds merely to force a SONIC terminal stage.
        uncertain.update(range(len(numbers)))
        reasons.append("NO_COVALENT_TOPOLOGY")

    metals = tuple(index for index, number in enumerate(numbers) if is_metal_atomic_number(number))
    for ring in rings:
        ring_atoms = tuple(int(atom) for atom in ring.atoms)
        if len(ring_atoms) not in {3, 4, 5, 6}:
            continue
        center = np.mean(coordinates[list(ring_atoms)], axis=0)
        for metal in metals:
            center_distance = float(np.linalg.norm(coordinates[metal] - center))
            donor_distances = np.linalg.norm(
                coordinates[list(ring_atoms)] - coordinates[metal], axis=1
            )
            # A generous local screen includes slipped eta_n geometries while
            # excluding ordinary remote aromatic rings.  The actual ambiguity
            # criterion below remains discrete and auditable.
            if center_distance > 3.25 or float(np.min(donor_distances)) > 3.20:
                continue
            spokes = sum(tuple(sorted((metal, atom))) in bond_set for atom in ring_atoms)
            if 0 < spokes < len(ring_atoms):
                uncertain.update((metal, *ring_atoms))
                reasons.append(
                    f"PARTIAL_METAL_RING_HAPTICITY:M{metal + 1}:R{int(ring.index) + 1}:"
                    f"{spokes}/{len(ring_atoms)}"
                )

    adjacency = [set() for _ in numbers]
    for left, right in bonds:
        adjacency[left].add(right)
        adjacency[right].add(left)
    boundary: set[int] = set()
    frontier = set(uncertain)
    visited = set(uncertain)
    for _ in range(hops):
        next_frontier = {
            neighbor
            for atom in frontier
            for neighbor in adjacency[atom]
            if neighbor not in visited
        }
        boundary.update(next_frontier)
        visited.update(next_frontier)
        frontier = next_frontier

    gradient_inf: float | None = None
    near_stationary: bool | None = None
    if gradient_hartree_per_bohr is not None:
        gradient = np.asarray(gradient_hartree_per_bohr, dtype=float).reshape(-1)
        if gradient.shape != (coordinates.size,) or not np.all(np.isfinite(gradient)):
            raise ValueError("topology-domain gradient must contain 3N finite values")
        gradient_inf = float(np.max(np.abs(gradient), initial=0.0))
        near_stationary = gradient_inf <= float(gradient_stationary_threshold)
        reasons.append(
            "CARTESIAN_GRADIENT_NEAR_STATIONARY"
            if near_stationary
            else "CARTESIAN_GRADIENT_FAR_FROM_STATIONARY"
        )

    stable = tuple(atom for atom in range(len(numbers)) if atom not in uncertain)
    return TopologyDomainAssessment(
        task_regime=task_regime,
        stable_atoms=stable,
        uncertain_atoms=tuple(uncertain),
        boundary_atoms=tuple(boundary),
        topology_well_defined=not uncertain,
        geometry_near_stationary=near_stationary,
        gradient_inf_hartree_per_bohr=gradient_inf,
        reasons=tuple(dict.fromkeys(reasons or ("ORACLE_TOPOLOGY_WELL_DEFINED",))),
        bonds=bonds,
    )


__all__ = [
    "ORACLE_TOPOLOGY_DOMAIN_SCHEMA",
    "TopologyDomainAssessment",
    "assess_topology_domains",
]
