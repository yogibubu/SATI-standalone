"""Public ORACLE ring-perception contract.

ORACLE owns molecular cycle perception.  SMITH consumes the resulting ordered
cycles to construct ring coordinates; it does not define a second ring model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from matrix_chem.topology.ringset import RingSet


ORACLE_RING_PERCEPTION_SCHEMA = "matrix.oracle.rings.v1"
ORACLE_COUPLED_RING_TOPOLOGY_SCHEMA = "matrix.oracle.rings.v2"


@dataclass(frozen=True)
class OracleRingPerception:
    schema: str
    rings: tuple[tuple[int, ...], ...]
    atom_to_rings: tuple[tuple[int, ...], ...]
    bond_to_rings: tuple[tuple[tuple[int, int], tuple[int, ...]], ...]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OracleRingCenter:
    """Local environment retained inside a coupled cyclic component.

    ``substituent_neighbors`` is deliberately part of the cyclic contract:
    SMITH must not reduce an endocyclic center before seeing the first-shell
    exocyclic directions that constrain its angular and out-of-plane space.
    Roles are non-exclusive topology annotations, not reduction rules.
    """

    atom: int
    coordination: int
    neighbors: tuple[int, ...]
    ring_neighbors: tuple[int, ...]
    substituent_neighbors: tuple[int, ...]
    ring_memberships: tuple[int, ...]
    roles: tuple[str, ...]


@dataclass(frozen=True)
class OracleCoupledRingComponent:
    """A maximal atom-overlap component of the perceived cycle basis."""

    index: int
    ring_indices: tuple[int, ...]
    core_atoms: tuple[int, ...]
    core_bonds: tuple[tuple[int, int], ...]
    attachment_atoms: tuple[int, ...]
    centers: tuple[OracleRingCenter, ...]
    topology_flags: tuple[str, ...]


@dataclass(frozen=True)
class OracleCoupledRingTopology:
    """Versioned ORACLE handoff for coupled ring-coordinate construction."""

    schema: str
    atom_index_base: int
    perception: OracleRingPerception
    components: tuple[OracleCoupledRingComponent, ...]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def perceive_rings(
    graph,
    coordinates=None,
    *,
    maximum_size: int | None = None,
) -> OracleRingPerception:
    """Perceive a complete non-metal minimum cycle basis.

    ``maximum_size=None`` is the production default and admits macrocycles.
    A finite maximum is an explicit user truncation; completeness is exposed
    in the returned diagnostics and is never silently assumed.
    """

    if maximum_size is not None and int(maximum_size) < 3:
        raise ValueError("maximum ring size must be at least three")
    ring_set = RingSet(
        graph,
        coords=coordinates,
        ring_max=None if maximum_size is None else int(maximum_size),
    )
    natoms = int(getattr(graph, "natoms", getattr(graph, "n_atoms", 0)))
    diagnostics = asdict(ring_set.cycle_basis_diagnostics)
    diagnostics["maximum_size_requested"] = maximum_size
    return OracleRingPerception(
        schema=ORACLE_RING_PERCEPTION_SCHEMA,
        rings=tuple(tuple(int(atom) for atom in ring.atoms) for ring in ring_set.rings),
        atom_to_rings=tuple(
            tuple(int(index) for index in ring_set.rings_of_atom(atom))
            for atom in range(natoms)
        ),
        bond_to_rings=tuple(
            (tuple(int(atom) for atom in bond), tuple(int(index) for index in indices))
            for bond, indices in sorted(ring_set.bond_to_rings.items())
        ),
        diagnostics=diagnostics,
    )


def build_coupled_ring_topology(
    graph,
    perception: OracleRingPerception,
) -> OracleCoupledRingTopology:
    """Build maximal coupled cyclic domains, including first-shell attachments.

    Rings are connected by atom overlap rather than only by a shared bond.  The
    resulting domains therefore cover fused, spiro, bridged and more general
    condensed systems without making the numerical treatment depend on an
    ambiguous named-ring classification.
    """

    rings = tuple(tuple(int(atom) for atom in ring) for ring in perception.rings)
    natoms = int(getattr(graph, "natoms", getattr(graph, "n_atoms", 0)))
    adjacency = _graph_adjacency(graph, natoms)
    ring_sets = tuple(frozenset(ring) for ring in rings)
    pending = set(range(len(rings)))
    ring_components: list[tuple[int, ...]] = []
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
        ring_components.append(tuple(sorted(component)))

    components = tuple(
        _coupled_component(index, indices, rings, adjacency)
        for index, indices in enumerate(ring_components)
    )
    return OracleCoupledRingTopology(
        schema=ORACLE_COUPLED_RING_TOPOLOGY_SCHEMA,
        atom_index_base=0,
        perception=perception,
        components=components,
        diagnostics={
            "component_count": len(components),
            "cyclic_atom_count": len(
                {atom for component in components for atom in component.core_atoms}
            ),
            "attachment_atom_count": len(
                {atom for component in components for atom in component.attachment_atoms}
            ),
            "component_rule": "cycle_basis_atom_overlap",
        },
    )


def perceive_coupled_ring_topology(
    graph,
    coordinates=None,
    *,
    maximum_size: int | None = None,
) -> OracleCoupledRingTopology:
    """Perceive rings and construct their coupled ORACLE v2 handoff."""

    perception = perceive_rings(graph, coordinates, maximum_size=maximum_size)
    return build_coupled_ring_topology(graph, perception)


def _graph_adjacency(graph, natoms: int) -> tuple[frozenset[int], ...]:
    if natoms < 0:
        raise ValueError("number of atoms cannot be negative")
    adjacency = [set() for _ in range(natoms)]
    bonds = getattr(graph, "bonds", ())
    for bond in bonds:
        if hasattr(bond, "left") and hasattr(bond, "right"):
            left, right = int(bond.left), int(bond.right)
        else:
            left, right = (int(atom) for atom in bond[:2])
        if left == right or not (0 <= left < natoms and 0 <= right < natoms):
            raise ValueError(f"invalid graph bond ({left}, {right})")
        adjacency[left].add(right)
        adjacency[right].add(left)
    if not bonds and hasattr(graph, "neighbors"):
        for atom in range(natoms):
            adjacency[atom].update(int(neighbor) for neighbor in graph.neighbors(atom))
    return tuple(frozenset(neighbors) for neighbors in adjacency)


def _ring_edges(ring: tuple[int, ...]) -> frozenset[tuple[int, int]]:
    if len(ring) < 3:
        raise ValueError("a perceived ring must contain at least three atoms")
    return frozenset(
        tuple(sorted((ring[index], ring[(index + 1) % len(ring)])))
        for index in range(len(ring))
    )


def _coupled_component(index, ring_indices, rings, adjacency):
    selected = tuple(rings[ring_index] for ring_index in ring_indices)
    invalid_atoms = sorted(
        {
            atom
            for ring in selected
            for atom in ring
            if atom < 0 or atom >= len(adjacency)
        }
    )
    if invalid_atoms:
        raise ValueError(
            "perceived ring atoms are outside the graph: "
            + ", ".join(str(atom) for atom in invalid_atoms)
        )
    selected_sets = tuple(frozenset(ring) for ring in selected)
    selected_edges = tuple(_ring_edges(ring) for ring in selected)
    core_atoms = frozenset(atom for ring in selected for atom in ring)
    core_bonds = frozenset(edge for edges in selected_edges for edge in edges)
    attachments = frozenset(
        neighbor
        for atom in core_atoms
        for neighbor in adjacency[atom]
        if neighbor not in core_atoms
    )
    centers = []
    for atom in sorted(core_atoms):
        memberships = tuple(
            ring_indices[local_index]
            for local_index, atoms in enumerate(selected_sets)
            if atom in atoms
        )
        ring_neighbors = tuple(sorted(adjacency[atom].intersection(core_atoms)))
        substituents = tuple(sorted(adjacency[atom] - core_atoms))
        roles = ["ring_center"]
        if substituents:
            roles.append("substituted")
        if len(memberships) > 1:
            incident_shared_edge = any(
                atom in edge and sum(edge in edges for edges in selected_edges) > 1
                for edge in core_bonds
            )
            if incident_shared_edge:
                roles.append("shared_edge_junction")
            else:
                roles.append("shared_atom_junction")
            if len(ring_neighbors) >= 3:
                roles.append("high_connectivity_junction")
        centers.append(
            OracleRingCenter(
                atom=atom,
                coordination=len(adjacency[atom]),
                neighbors=tuple(sorted(adjacency[atom])),
                ring_neighbors=ring_neighbors,
                substituent_neighbors=substituents,
                ring_memberships=memberships,
                roles=tuple(roles),
            )
        )

    flags = []
    if len(selected) == 1:
        flags.append("isolated")
    else:
        flags.append("polycyclic")
        if any(
            selected_edges[left].intersection(selected_edges[right])
            for left in range(len(selected))
            for right in range(left + 1, len(selected))
        ):
            flags.append("shared_edge")
        if any(
            len(selected_sets[left].intersection(selected_sets[right])) == 1
            for left in range(len(selected))
            for right in range(left + 1, len(selected))
        ):
            flags.append("spiro_overlap")

    return OracleCoupledRingComponent(
        index=int(index),
        ring_indices=tuple(int(value) for value in ring_indices),
        core_atoms=tuple(sorted(core_atoms)),
        core_bonds=tuple(sorted(core_bonds)),
        attachment_atoms=tuple(sorted(attachments)),
        centers=tuple(centers),
        topology_flags=tuple(flags),
    )


__all__ = [
    "ORACLE_COUPLED_RING_TOPOLOGY_SCHEMA",
    "ORACLE_RING_PERCEPTION_SCHEMA",
    "OracleCoupledRingComponent",
    "OracleCoupledRingTopology",
    "OracleRingCenter",
    "OracleRingPerception",
    "build_coupled_ring_topology",
    "perceive_coupled_ring_topology",
    "perceive_rings",
]
