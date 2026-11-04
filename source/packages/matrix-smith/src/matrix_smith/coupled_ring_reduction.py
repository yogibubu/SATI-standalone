"""SMITH reduction domains derived from the frozen ORACLE ring topology.

ORACLE owns every chemical decision in the input topology.  This module only
turns its ordered rings, bonds, SOOP records and symmetry permutations into
the numerical blocks on which SMITH may subsequently perform an irrep-wise
rank-revealing reduction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


SMITH_COUPLED_RING_REDUCTION_SCHEMA = "matrix.smith.coupled_ring_reduction.v1"


@dataclass(frozen=True)
class CoupledRingCenterReduction:
    atom: int
    coordination: int
    angular_rank: int
    ring_neighbors: tuple[int, ...]
    substituent_neighbors: tuple[int, ...]


@dataclass(frozen=True)
class CoupledRingReductionDomain:
    domain_id: str
    component_index: int
    symmetry_orbit_id: str
    ring_ids: tuple[int, ...]
    core_atoms: tuple[int, ...]
    core_bonds: tuple[tuple[int, int], ...]
    attachment_atoms: tuple[int, ...]
    centers: tuple[CoupledRingCenterReduction, ...]
    endocyclic_soop_ids: tuple[str, ...]
    attached_exocyclic_soop_ids: tuple[str, ...]


@dataclass(frozen=True)
class CoupledRingReductionPlan:
    schema: str
    atom_index_base: int
    domains: tuple[CoupledRingReductionDomain, ...]
    acyclic_soop_ids: tuple[str, ...]


def build_coupled_ring_reduction_plan(
    topology,
    soop_primitives: Iterable[object] = (),
) -> CoupledRingReductionPlan:
    """Build component-local, symmetry-orbit-aware cyclic reduction domains.

    Atom and ring identifiers remain one based, exactly as serialized in the
    ORACLE SONIC contract.  A component is maximal under ring atom overlap;
    symmetry-related but disconnected components remain distinct domains and
    receive the same ``symmetry_orbit_id``.  This is the information needed to
    form SALCs without allowing an unconstrained molecular-global SVD.
    """

    natoms = int(topology.natoms)
    rings = tuple(tuple(int(atom) for atom in ring) for ring in topology.rings)
    adjacency = {atom: set() for atom in range(1, natoms + 1)}
    for bond in topology.bonds:
        left, right = (int(atom) for atom in bond[:2])
        if left == right or left not in adjacency or right not in adjacency:
            raise ValueError(f"invalid frozen topology bond ({left}, {right})")
        adjacency[left].add(right)
        adjacency[right].add(left)
    for ring in rings:
        if len(ring) < 3 or len(set(ring)) != len(ring):
            raise ValueError("invalid frozen ORACLE ring")
        if any(atom not in adjacency for atom in ring):
            raise ValueError("frozen ORACLE ring contains an invalid atom")

    component_ring_indices = _ring_overlap_components(rings)
    core_sets = tuple(
        frozenset(atom for index in indices for atom in rings[index])
        for indices in component_ring_indices
    )
    orbit_indices = _component_symmetry_orbits(
        core_sets,
        tuple(tuple(int(atom) for atom in permutation)
              for permutation in topology.symmetry_permutations),
        natoms,
    )
    orbit_by_component = {
        component: orbit
        for orbit, components in enumerate(orbit_indices, start=1)
        for component in components
    }

    records = tuple(soop_primitives)
    identifiers = tuple(str(record.identifier) for record in records)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate frozen SOOP identifier")
    endocyclic_by_component: dict[int, list[str]] = {
        index: [] for index in range(len(core_sets))
    }
    exocyclic_by_component: dict[int, list[str]] = {
        index: [] for index in range(len(core_sets))
    }
    acyclic: list[str] = []
    ring_to_component = {
        ring_index + 1: component_index
        for component_index, indices in enumerate(component_ring_indices)
        for ring_index in indices
    }
    for record in records:
        ring_ids = tuple(int(value) for value in record.endocyclic_ring_ids)
        if ring_ids:
            owners = {ring_to_component.get(ring_id) for ring_id in ring_ids}
            if None in owners or len(owners) != 1:
                raise ValueError(
                    f"SOOP {record.identifier} crosses unrelated cyclic components"
                )
            owner = owners.pop()
            if (
                int(record.center_atom) not in core_sets[owner]
                or int(record.moving_atom) not in core_sets[owner]
            ):
                raise ValueError(
                    f"SOOP {record.identifier} endocyclic support disagrees with its ring"
                )
            endocyclic_by_component[owner].append(str(record.identifier))
            continue
        center = int(record.center_atom)
        matching = [
            index for index, atoms in enumerate(core_sets) if center in atoms
        ]
        if len(matching) > 1:
            raise ValueError(
                f"SOOP {record.identifier} belongs to multiple cyclic components"
            )
        if matching:
            exocyclic_by_component[matching[0]].append(str(record.identifier))
        else:
            acyclic.append(str(record.identifier))

    domains = []
    for component_index, (ring_indices, core_atoms) in enumerate(
        zip(component_ring_indices, core_sets, strict=True)
    ):
        core_bonds = tuple(sorted({
            tuple(sorted((ring[position], ring[(position + 1) % len(ring)])))
            for ring_index in ring_indices
            for ring in (rings[ring_index],)
            for position in range(len(ring))
        }))
        attachments = tuple(
            sorted(
                {
                    neighbor
                    for atom in core_atoms
                    for neighbor in adjacency[atom]
                    if neighbor not in core_atoms
                }
            )
        )
        centers = tuple(
            CoupledRingCenterReduction(
                atom=atom,
                coordination=len(adjacency[atom]),
                angular_rank=max(0, 2 * len(adjacency[atom]) - 3),
                ring_neighbors=tuple(sorted(adjacency[atom].intersection(core_atoms))),
                substituent_neighbors=tuple(sorted(adjacency[atom] - core_atoms)),
            )
            for atom in sorted(core_atoms)
        )
        orbit = orbit_by_component[component_index]
        domains.append(
            CoupledRingReductionDomain(
                domain_id=f"RING_COMPONENT_{component_index + 1:04d}",
                component_index=component_index,
                symmetry_orbit_id=f"RING_COMPONENT_ORBIT_{orbit:04d}",
                ring_ids=tuple(index + 1 for index in ring_indices),
                core_atoms=tuple(sorted(core_atoms)),
                core_bonds=core_bonds,
                attachment_atoms=attachments,
                centers=centers,
                endocyclic_soop_ids=tuple(endocyclic_by_component[component_index]),
                attached_exocyclic_soop_ids=tuple(exocyclic_by_component[component_index]),
            )
        )
    return CoupledRingReductionPlan(
        schema=SMITH_COUPLED_RING_REDUCTION_SCHEMA,
        atom_index_base=1,
        domains=tuple(domains),
        acyclic_soop_ids=tuple(acyclic),
    )


def coupled_ring_reduction_plan_from_oracle_contract(contract):
    """Convenience entry point for a validated frozen ORACLE SONIC contract."""

    return build_coupled_ring_reduction_plan(
        contract.primary_topology,
        contract.selective_out_of_plane_primitives,
    )


def _ring_overlap_components(rings):
    ring_sets = tuple(frozenset(ring) for ring in rings)
    pending = set(range(len(rings)))
    result = []
    while pending:
        seed = min(pending)
        pending.remove(seed)
        component = {seed}
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            joined = {
                other
                for other in pending
                if ring_sets[current].intersection(ring_sets[other])
            }
            pending.difference_update(joined)
            component.update(joined)
            frontier.extend(sorted(joined, reverse=True))
        result.append(tuple(sorted(component)))
    return tuple(result)


def _component_symmetry_orbits(core_sets, permutations, natoms):
    count = len(core_sets)
    parents = list(range(count))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    identity = tuple(range(1, natoms + 1))
    operations = permutations or (identity,)
    lookup = {atoms: index for index, atoms in enumerate(core_sets)}
    for permutation in operations:
        if len(permutation) != natoms or sorted(permutation) != list(identity):
            raise ValueError("invalid frozen topology symmetry permutation")
        for index, atoms in enumerate(core_sets):
            mapped = frozenset(permutation[atom - 1] for atom in atoms)
            partner = lookup.get(mapped)
            if partner is None:
                raise ValueError("symmetry permutation does not preserve cyclic components")
            union(index, partner)
    grouped: dict[int, list[int]] = {}
    for index in range(count):
        grouped.setdefault(find(index), []).append(index)
    return tuple(tuple(grouped[root]) for root in sorted(grouped))


__all__ = [
    "SMITH_COUPLED_RING_REDUCTION_SCHEMA",
    "CoupledRingCenterReduction",
    "CoupledRingReductionDomain",
    "CoupledRingReductionPlan",
    "build_coupled_ring_reduction_plan",
    "coupled_ring_reduction_plan_from_oracle_contract",
]
