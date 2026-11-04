"""Label-preserving automorphisms exposed by the ORACLE topology contract."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

from matrix_core import read_sectioned_lines, section_content


LEGACY_MATRIX_XYZ_TOPOLOGY_AUTOMORPHISMS_SCHEMA = "matrix.xyz.topology_automorphisms.v1"
MATRIX_XYZ_TOPOLOGY_AUTOMORPHISMS_SCHEMA = "matrix.xyz.topology_automorphisms.v2"
SUPPORTED_MATRIX_XYZ_TOPOLOGY_AUTOMORPHISMS_SCHEMAS = (
    LEGACY_MATRIX_XYZ_TOPOLOGY_AUTOMORPHISMS_SCHEMA,
    MATRIX_XYZ_TOPOLOGY_AUTOMORPHISMS_SCHEMA,
)
TOPOLOGY_AUTOMORPHISM_LABEL_POLICY = (
    "ELEMENT+SYNTHON_SIGNATURE+EXTERNAL_ATTACHMENTS/"
    "BOND_CLASS+FULL_GRAPH_EXTENSION_VF2+CYCLE_BASIS_INDEPENDENT"
)


@dataclass(frozen=True)
class TopologyAutomorphismOrbit:
    """One labelled ring or ring-system orbit in local atom ordering."""

    kind: str
    index: int
    atoms: tuple[int, ...]
    permutations: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().upper()
        if kind not in {"RING", "RING_SYSTEM"}:
            raise ValueError("automorphism orbit kind must be RING or RING_SYSTEM")
        atoms = tuple(int(atom) for atom in self.atoms)
        if not atoms or len(set(atoms)) != len(atoms) or min(atoms) < 0:
            raise ValueError("automorphism orbit atoms must be unique non-negative indices")
        expected = tuple(range(len(atoms)))
        permutations = tuple(tuple(int(value) for value in item) for item in self.permutations)
        if not permutations or any(tuple(sorted(item)) != expected for item in permutations):
            raise ValueError("automorphism permutations must cover the local atom ordering")
        if int(self.index) < 1:
            raise ValueError("automorphism orbit index must be positive")
        if len(set(permutations)) != len(permutations) or expected not in permutations:
            raise ValueError("automorphism permutations must be unique and contain identity")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "index", int(self.index))
        object.__setattr__(self, "atoms", atoms)
        object.__setattr__(self, "permutations", permutations)


def enumerate_labelled_graph_automorphisms(
    adjacency: Sequence[Sequence[bool | int]] | np.ndarray,
    *,
    vertex_labels: Sequence[Hashable] | None = None,
    edge_labels: Mapping[tuple[int, int], Hashable] | None = None,
    max_automorphisms: int = 4096,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate automorphisms of an arbitrary finite labelled graph."""

    graph = np.asarray(adjacency, dtype=bool)
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1] or graph.shape[0] == 0:
        raise ValueError("graph adjacency must be a nonempty square matrix")
    if not np.array_equal(graph, graph.T) or np.any(np.diag(graph)):
        raise ValueError("graph adjacency must be symmetric with a zero diagonal")
    size = graph.shape[0]
    labels = tuple(0 for _ in range(size)) if vertex_labels is None else tuple(vertex_labels)
    if len(labels) != size:
        raise ValueError("vertex_labels must match graph size")
    limit = int(max_automorphisms)
    if limit < 1:
        raise ValueError("max_automorphisms must be positive")

    normalized_edges: dict[tuple[int, int], Hashable] = {}
    for pair, label in (edge_labels or {}).items():
        left, right = int(pair[0]), int(pair[1])
        if left < 0 or right < 0 or left >= size or right >= size or left == right:
            raise ValueError("edge label key is outside the graph")
        key = tuple(sorted((left, right)))
        if not graph[key]:
            raise ValueError("edge labels may be assigned only to graph edges")
        normalized_edges[key] = label

    def edge_label(left: int, right: int) -> Hashable | None:
        return normalized_edges.get(tuple(sorted((left, right))))

    labelled_graph = _networkx_labelled_graph(graph, labels, edge_label)
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        labelled_graph,
        labelled_graph,
        node_match=nx.algorithms.isomorphism.categorical_node_match("label", None),
        edge_match=nx.algorithms.isomorphism.categorical_edge_match("label", None),
    )
    output: list[tuple[int, ...]] = []
    for mapping in matcher.isomorphisms_iter():
        output.append(tuple(int(mapping[index]) for index in range(size)))
        if len(output) > limit:
            raise RuntimeError(
                f"labelled graph has more than {limit} admitted automorphisms"
            )
    return tuple(sorted(output))


def geometry_labelled_graph_automorphisms(
    atoms: Sequence[str],
    coordinates_angstrom: Sequence[Sequence[float]] | np.ndarray,
    *,
    max_automorphisms: int = 4096,
) -> tuple[tuple[int, ...], ...]:
    """Return element-labelled graph automorphisms perceived for one geometry."""

    from .elements import atomic_number
    from .pipeline import build_topology_objects

    numbers = tuple(int(atomic_number(atom) or 0) for atom in atoms)
    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    if coordinates.shape != (len(numbers), 3):
        raise ValueError("automorphism geometry must have shape (natoms, 3)")
    _continuous, graph, _rings, _synthons, _aromaticity = build_topology_objects(
        coordinates,
        numbers,
    )
    adjacency = np.zeros((len(numbers), len(numbers)), dtype=bool)
    for left, right in graph.bonds:
        adjacency[int(left), int(right)] = True
        adjacency[int(right), int(left)] = True
    return enumerate_labelled_graph_automorphisms(
        adjacency,
        vertex_labels=numbers,
        max_automorphisms=max_automorphisms,
    )


def topology_automorphism_edge_labels(
    discrete_graph,
    synthons,
    *,
    aromaticity=None,
) -> dict[tuple[int, int], str]:
    """Return ORACLE's canonical edge labels for full-graph automorphisms.

    This is the shared, enumeration-free label provider for consumers that
    already have a candidate atom permutation.  It keeps quasi-symmetry and
    other graph-covariance checks on the same SINGLE/DOUBLE/TRIPLE/AROMATIC
    contract used by ORACLE's ring and ring-system automorphisms.
    """

    aromatic_bonds = _cycle_basis_independent_aromatic_bonds(
        discrete_graph,
        aromaticity,
    )
    return {
        pair: _bond_class(
            synthons,
            pair[0],
            pair[1],
            pair in aromatic_bonds,
        )
        for pair in (
            tuple(sorted((int(left), int(right))))
            for left, right in discrete_graph.bonds
        )
    }


def _networkx_labelled_graph(graph, labels, edge_label):
    output = nx.Graph()
    output.add_nodes_from(
        (index, {"label": labels[index]}) for index in range(graph.shape[0])
    )
    output.add_edges_from(
        (
            left,
            right,
            {"label": edge_label(left, right)},
        )
        for left in range(graph.shape[0])
        for right in range(left + 1, graph.shape[0])
        if graph[left, right]
    )
    return output


def build_topology_automorphism_orbits(
    discrete_graph,
    ringset,
    synthons,
    atomic_numbers: Sequence[int],
    *,
    aromaticity=None,
) -> tuple[TopologyAutomorphismOrbit, ...]:
    """Build labelled automorphism orbits for rings and ring systems."""

    rings = tuple(ringset.rings)
    if not rings:
        return ()
    components = _ring_components(rings)
    system_orbits: list[TopologyAutomorphismOrbit] = []
    system_by_ring: dict[int, TopologyAutomorphismOrbit] = {}
    for system_index, component in enumerate(components, start=1):
        atoms = tuple(sorted({atom for ring_index in component for atom in rings[ring_index].atoms}))
        system_orbit = _build_orbit(
            "RING_SYSTEM",
            system_index,
            atoms,
            discrete_graph,
            ringset,
            synthons,
            atomic_numbers,
            aromaticity=aromaticity,
        )
        system_orbits.append(system_orbit)
        for ring_index in component:
            system_by_ring[int(ring_index)] = system_orbit
    ring_orbits = tuple(
        _restrict_system_automorphisms_to_ring(
            system_by_ring[ring_index - 1],
            ring_index,
            tuple(int(atom) for atom in ring.atoms),
        )
        for ring_index, ring in enumerate(rings, start=1)
    )
    return tuple(system_orbits) + ring_orbits


def _restrict_system_automorphisms_to_ring(
    system_orbit: TopologyAutomorphismOrbit,
    ring_index: int,
    ring_atoms: tuple[int, ...],
) -> TopologyAutomorphismOrbit:
    """Derive one ring stabilizer from its already extended system group."""

    system_local = {atom: index for index, atom in enumerate(system_orbit.atoms)}
    ring_local = {atom: index for index, atom in enumerate(ring_atoms)}
    ring_set = set(ring_atoms)
    permutations: set[tuple[int, ...]] = set()
    for system_permutation in system_orbit.permutations:
        mapped_atoms = tuple(
            system_orbit.atoms[system_permutation[system_local[atom]]]
            for atom in ring_atoms
        )
        if set(mapped_atoms) != ring_set:
            continue
        permutations.add(tuple(ring_local[atom] for atom in mapped_atoms))
    return TopologyAutomorphismOrbit(
        "RING",
        int(ring_index),
        ring_atoms,
        tuple(sorted(permutations)),
    )


def topology_automorphism_lines(
    orbits: Sequence[TopologyAutomorphismOrbit],
) -> list[str]:
    lines = [
        "[AUTOMORPHISMS]",
        f"SCHEMA {MATRIX_XYZ_TOPOLOGY_AUTOMORPHISMS_SCHEMA}",
        f"LABEL_POLICY {TOPOLOGY_AUTOMORPHISM_LABEL_POLICY}",
        f"ORBIT_COUNT {len(orbits)}",
    ]
    for orbit_number, orbit in enumerate(orbits, start=1):
        atoms = ",".join(str(atom + 1) for atom in orbit.atoms)
        lines.append(
            f"ORBIT {orbit_number} KIND={orbit.kind} INDEX={orbit.index} "
            f"ATOMS={atoms} COUNT={len(orbit.permutations)}"
        )
        lines.extend(
            f"MAP {map_index} PERMUTATION="
            + ",".join(str(value + 1) for value in permutation)
            for map_index, permutation in enumerate(orbit.permutations, start=1)
        )
    if not orbits:
        lines.append("NONE")
    return lines


def read_topology_automorphism_orbits(path: Path | str) -> tuple[TopologyAutomorphismOrbit, ...]:
    topology = section_content(read_sectioned_lines(Path(path)), "TOPOLOGY")
    block = _subsection(topology, "AUTOMORPHISMS")
    if not block:
        return ()
    schema = next(
        (line.split(None, 1)[1].strip() for line in block if line.upper().startswith("SCHEMA ")),
        "",
    )
    if schema not in SUPPORTED_MATRIX_XYZ_TOPOLOGY_AUTOMORPHISMS_SCHEMAS:
        raise ValueError(f"unsupported topology automorphism schema: {schema or 'missing'}")
    output: list[TopologyAutomorphismOrbit] = []
    declared_orbit_count = next(
        (
            int(line.split()[1])
            for line in block
            if line.upper().startswith("ORBIT_COUNT ")
        ),
        0,
    )
    current: dict[str, object] | None = None
    for line in block:
        if line.startswith("ORBIT "):
            if current is not None:
                output.append(_parsed_orbit(current))
            parts = line.split()
            values = _inline_values(parts[2:])
            current = {
                "kind": values["KIND"],
                "index": int(values["INDEX"]),
                "atoms": tuple(int(value) - 1 for value in values["ATOMS"].split(",")),
                "declared_count": int(values["COUNT"]),
                "permutations": [],
            }
        elif line.startswith("MAP ") and current is not None:
            values = _inline_values(line.split()[2:])
            current["permutations"].append(
                tuple(int(value) - 1 for value in values["PERMUTATION"].split(","))
            )
    if current is not None:
        output.append(_parsed_orbit(current))
    if len(output) != declared_orbit_count:
        raise ValueError(
            f"topology automorphism orbit count is {len(output)}, expected {declared_orbit_count}"
        )
    return tuple(output)


def select_topology_automorphism_orbit(
    orbits: Sequence[TopologyAutomorphismOrbit],
    *,
    kind: str,
    index: int,
) -> TopologyAutomorphismOrbit:
    target_kind = str(kind).strip().upper()
    matches = tuple(
        orbit for orbit in orbits if orbit.kind == target_kind and orbit.index == int(index)
    )
    if len(matches) != 1:
        raise ValueError(
            f"expected one {target_kind} automorphism orbit with index {index}, found {len(matches)}"
        )
    return matches[0]


def _build_orbit(
    kind,
    index,
    atoms,
    discrete_graph,
    ringset,
    synthons,
    atomic_numbers,
    *,
    aromaticity,
) -> TopologyAutomorphismOrbit:
    atom_to_local = {atom: local for local, atom in enumerate(atoms)}
    size = len(atoms)
    adjacency = np.zeros((size, size), dtype=bool)
    edge_labels: dict[tuple[int, int], Hashable] = {}
    aromatic_bonds = _cycle_basis_independent_aromatic_bonds(
        discrete_graph,
        aromaticity,
    )
    for left_local, left in enumerate(atoms):
        for right in discrete_graph.adjacency[left]:
            right_local = atom_to_local.get(right)
            if right_local is None or right_local <= left_local:
                continue
            adjacency[left_local, right_local] = adjacency[right_local, left_local] = True
            pair = tuple(sorted((left, right)))
            edge_labels[(left_local, right_local)] = _bond_class(
                synthons, left, right, pair in aromatic_bonds
            )
    labels = tuple(
        _vertex_label(
            atom,
            set(atoms),
            discrete_graph,
            ringset,
            synthons,
            atomic_numbers,
            aromatic_bonds,
        )
        for atom in atoms
    )
    permutations = enumerate_labelled_graph_automorphisms(
        adjacency,
        vertex_labels=labels,
        edge_labels=edge_labels,
    )
    if len(atoms) != len(atomic_numbers):
        permutations = tuple(
            permutation
            for permutation in permutations
            if _mapping_extends_to_full_graph(
                atoms,
                permutation,
                discrete_graph,
                ringset,
                synthons,
                atomic_numbers,
                aromatic_bonds,
            )
        )
    return TopologyAutomorphismOrbit(kind, index, atoms, permutations)


def _mapping_extends_to_full_graph(
    orbit_atoms,
    permutation,
    graph,
    ringset,
    synthons,
    atomic_numbers,
    aromatic_bonds,
) -> bool:
    """Return whether a local ring map extends to the complete labelled graph."""

    size = len(atomic_numbers)
    adjacency = tuple(frozenset(int(value) for value in row) for row in graph.adjacency)

    def vertex_label(atom: int):
        return (
            int(atomic_numbers[atom]),
            tuple(synthons.canonical_signature(atom)),
        )

    def edge_label(left: int, right: int):
        pair = tuple(sorted((left, right)))
        return _bond_class(synthons, left, right, pair in aromatic_bonds)

    labels = tuple(vertex_label(atom) for atom in range(size))
    graph_array = np.zeros((size, size), dtype=bool)
    for left in range(size):
        for right in adjacency[left]:
            graph_array[left, right] = True

    source_anchors: dict[int, int] = {}
    target_anchors: dict[int, int] = {}
    for anchor, (source_local, image_local) in enumerate(
        zip(range(len(permutation)), permutation, strict=True)
    ):
        source = int(orbit_atoms[source_local])
        image = int(orbit_atoms[image_local])
        if source in source_anchors or image in target_anchors:
            return False
        source_anchors[source] = anchor
        target_anchors[image] = anchor

    source_labels = tuple(
        (labels[atom], source_anchors.get(atom)) for atom in range(size)
    )
    target_labels = tuple(
        (labels[atom], target_anchors.get(atom)) for atom in range(size)
    )
    source_graph = _networkx_labelled_graph(graph_array, source_labels, edge_label)
    target_graph = _networkx_labelled_graph(graph_array, target_labels, edge_label)
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        source_graph,
        target_graph,
        node_match=nx.algorithms.isomorphism.categorical_node_match("label", None),
        edge_match=nx.algorithms.isomorphism.categorical_edge_match("label", None),
    )
    return bool(matcher.is_isomorphic())


def _vertex_label(atom, orbit_atoms, graph, ringset, synthons, atomic_numbers, aromatic_bonds):
    external = []
    for neighbor in graph.adjacency[atom]:
        if neighbor in orbit_atoms:
            continue
        pair = tuple(sorted((atom, neighbor)))
        external.append(
            (
                int(atomic_numbers[neighbor]),
                _bond_class(synthons, atom, neighbor, pair in aromatic_bonds),
            )
        )
    return (
        int(atomic_numbers[atom]),
        tuple(synthons.canonical_signature(atom)),
        tuple(sorted(external)),
    )


def _bond_class(synthons, left: int, right: int, aromatic: bool) -> str:
    if aromatic:
        return "AROMATIC"
    components = synthons.bond_order_components(left, right)
    if float(components.pi_pi) >= 0.20:
        return "TRIPLE"
    if float(components.pi) >= 0.20 or float(components.total) >= 1.25:
        return "DOUBLE"
    return "SINGLE"


def _cycle_basis_independent_aromatic_bonds(graph, aromaticity) -> set[tuple[int, int]]:
    """Return delocalized aromatic edges without depending on one cycle basis.

    An edge between aromatic atoms is delocalized only when it is not a bridge
    of the aromatic-atom-induced graph.  This retains every cage/fused-system
    edge, including edges of a face omitted from a minimum cycle basis, while
    excluding inter-ring links such as the central bond of biphenyl.
    """

    aromatic_atoms = set(int(atom) for atom in getattr(aromaticity, "aromatic_atoms", ()))
    if not aromatic_atoms:
        return set()
    aromatic_graph = nx.Graph()
    aromatic_graph.add_nodes_from(aromatic_atoms)
    aromatic_graph.add_edges_from(
        tuple(sorted((int(left), int(right))))
        for left in aromatic_atoms
        for right in graph.adjacency[left]
        if int(right) in aromatic_atoms and int(left) < int(right)
    )
    bridges = {tuple(sorted(pair)) for pair in nx.bridges(aromatic_graph)}
    return {
        tuple(sorted((int(left), int(right))))
        for left, right in aromatic_graph.edges
        if tuple(sorted((int(left), int(right)))) not in bridges
    }


def _ring_components(rings) -> tuple[tuple[int, ...], ...]:
    remaining = set(range(len(rings)))
    output = []
    while remaining:
        seed = min(remaining)
        stack = [seed]
        component = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            current_atoms = set(rings[current].atoms)
            stack.extend(
                other
                for other in remaining
                if other not in component and current_atoms.intersection(rings[other].atoms)
            )
        remaining.difference_update(component)
        output.append(tuple(sorted(component)))
    return tuple(output)


def _subsection(lines: Sequence[str], name: str) -> tuple[str, ...]:
    marker = f"[{name}]"
    try:
        start = tuple(lines).index(marker) + 1
    except ValueError:
        return ()
    output = []
    for line in tuple(lines)[start:]:
        if line.startswith("["):
            break
        output.append(line)
    return tuple(output)


def _inline_values(tokens: Sequence[str]) -> dict[str, str]:
    output = {}
    for token in tokens:
        if "=" in token:
            key, value = token.split("=", 1)
            output[key] = value
    return output


def _parsed_orbit(payload: dict[str, object]) -> TopologyAutomorphismOrbit:
    orbit = TopologyAutomorphismOrbit(
        str(payload["kind"]),
        int(payload["index"]),
        tuple(payload["atoms"]),
        tuple(payload["permutations"]),
    )
    if len(orbit.permutations) != int(payload["declared_count"]):
        raise ValueError(
            f"topology automorphism map count is {len(orbit.permutations)}, "
            f"expected {payload['declared_count']}"
        )
    return orbit
