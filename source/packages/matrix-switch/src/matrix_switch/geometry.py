"""Deterministic Cartesian realization of a SWITCH molecular graph."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from matrix_chem import (
    HydrogenCompletion,
    complete_valence_hydrogens,
    release_completed_hydrogen_clashes,
)
from matrix_chem.geometry import MolecularGeometry
from matrix_chem.topology.elements import atomic_number
from matrix_chem.topology.pykko_radii import covalent_radius
from matrix_numerics import eigh_arrays

from .model import SwitchBond, SwitchMolecularGraph
from .validation import validate_switch_geometry


SWITCH_GEOMETRY_SCHEMA = "matrix.switch.cartesian_seed.v1"
DEFAULT_DENSE_LAYOUT_MAX_ATOMS = 256
_GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))
_DIRECTION_COUNT = 96
_DIRECTION_INDEX = np.arange(_DIRECTION_COUNT, dtype=float)
_DIRECTION_Z = 1.0 - 2.0 * (_DIRECTION_INDEX + 0.5) / _DIRECTION_COUNT
_DIRECTION_RADIUS = np.sqrt(np.maximum(0.0, 1.0 - _DIRECTION_Z * _DIRECTION_Z))
_DIRECTIONS = np.column_stack(
    (
        _DIRECTION_RADIUS * np.cos(_GOLDEN_ANGLE * _DIRECTION_INDEX),
        _DIRECTION_Z,
        _DIRECTION_RADIUS * np.sin(_GOLDEN_ANGLE * _DIRECTION_INDEX),
    )
)


def build_cartesian_seed(
    graph: SwitchMolecularGraph,
    *,
    title: str = "",
    multiplicity: int | None = None,
    complete_hydrogens: bool = True,
) -> MolecularGeometry:
    """Build a reproducible non-random 3D seed without a force-field dependency."""

    coordinates = np.zeros((len(graph.atoms), 3), dtype=float)
    dense_layout = all(
        len(component) <= DEFAULT_DENSE_LAYOUT_MAX_ATOMS for component in graph.components
    )
    x_offset = 0.0
    for component in graph.components:
        local = _component_coordinates(graph, component)
        local[:, 0] -= float(np.min(local[:, 0]))
        coordinates[np.asarray(component, dtype=int)] = local + np.asarray((x_offset, 0.0, 0.0))
        x_offset += float(np.ptp(local[:, 0])) + 4.0
    coordinates -= np.mean(coordinates, axis=0)
    applied_coordination = _apply_coordination_stereochemistry(graph, coordinates)
    applied_double_bonds = _apply_directional_double_bonds(graph, coordinates)
    seed = MolecularGeometry(
        atoms=tuple(atom.symbol for atom in graph.atoms),
        coordinates_angstrom=coordinates,
        comment=title or graph.source_smiles,
        source_format="smiles_switch",
        charge=graph.total_formal_charge,
        multiplicity=multiplicity,
        metadata={
            "schema": SWITCH_GEOMETRY_SCHEMA,
            "smiles": graph.source_smiles,
            "smiles_parser": "MATRIX SWITCH",
            "cartesian_seed": (
                "compiled shortest paths + accelerated classical MDS"
                if dense_layout
                else "linear deterministic graph growth"
            ),
            "switch_applied_coordination_centers": applied_coordination,
            "switch_applied_directional_double_bonds": applied_double_bonds,
        },
    )
    if not complete_hydrogens:
        return _enforce_seed_contract(graph, seed)
    completion = complete_graph_hydrogens(graph, seed)
    completed_coordinates = np.asarray(
        completion.geometry.coordinates_angstrom,
        dtype=float,
    ).copy()
    if _graph_prefers_planar_seed(graph):
        for local_index in _cycle_atom_indices(graph, tuple(range(len(graph.atoms)))):
            completed_coordinates[local_index, 2] = 0.0
        _restore_projected_hydrogen_lengths(
            completed_coordinates,
            completion.additions,
            parent_symbols=graph.atoms,
            bonds=completion.bonds,
        )
        _release_projected_hydrogen_clashes(
            completed_coordinates,
            tuple((addition.parent, addition.atom) for addition in completion.additions),
        )
        grouped_parents: dict[int, list] = {}
        for addition in completion.additions:
            grouped_parents.setdefault(int(addition.parent), []).append(addition)
        multi_hydrogen_additions = tuple(
            (addition.parent, addition.atom)
            for additions in grouped_parents.values()
            if len(additions) > 1
            for addition in additions
        )
        completion = release_completed_hydrogen_clashes(
            _completion_with_coordinates(completion, completed_coordinates),
            atom_indices=tuple(hydrogen for _, hydrogen in multi_hydrogen_additions),
        )
        completed_coordinates = np.asarray(
            completion.geometry.coordinates_angstrom,
            dtype=float,
        ).copy()
    applied_chirality = _apply_tetrahedral_chirality(
        graph,
        completed_coordinates,
        tuple((addition.parent, addition.atom) for addition in completion.additions),
    )
    result = MolecularGeometry(
        atoms=completion.geometry.atoms,
        coordinates_angstrom=completed_coordinates,
        comment=seed.comment,
        source_format=seed.source_format,
        charge=seed.charge,
        multiplicity=seed.multiplicity,
        metadata={
            **dict(completion.geometry.metadata),
            "switch_graph_atom_count": len(graph.atoms),
            "switch_added_hydrogen_count": len(completion.additions),
            "switch_applied_tetrahedral_centers": applied_chirality,
        },
    )
    return _enforce_seed_contract(graph, result)


def _enforce_seed_contract(
    graph: SwitchMolecularGraph,
    geometry: MolecularGeometry,
) -> MolecularGeometry:
    """Reject malformed SWITCH seeds before they can enter a QM launcher."""
    validation = validate_switch_geometry(graph, geometry)
    if validation.errors:
        raise ValueError("invalid SWITCH seed: " + "; ".join(validation.errors))
    return geometry


def complete_graph_hydrogens(
    graph: SwitchMolecularGraph,
    geometry: MolecularGeometry,
    *,
    release_clashes: bool = True,
    minimum_separation_angstrom: float = 1.35,
) -> HydrogenCompletion:
    """Apply exactly the implicit-hydrogen contract of a SWITCH graph.

    The supplied Cartesian geometry must describe the explicit graph atoms in
    the original order.  Only implicit hydrogens declared or inferred by the
    parsed graph are appended; the explicit constitution and all heavy-atom
    coordinates remain immutable.
    """

    graph_symbols = tuple(atom.symbol for atom in graph.atoms)
    if tuple(geometry.atoms) != graph_symbols:
        raise ValueError("SWITCH hydrogen completion requires the exact graph atom order")
    if int(geometry.charge or 0) != int(graph.total_formal_charge):
        raise ValueError("SWITCH hydrogen completion requires the graph formal charge")
    contracted = MolecularGeometry(
        atoms=geometry.atoms,
        coordinates_angstrom=geometry.coordinates_angstrom,
        comment=geometry.comment,
        source_format=geometry.source_format,
        source_path=geometry.source_path,
        charge=geometry.charge,
        multiplicity=geometry.multiplicity,
        fixed_parameters=geometry.fixed_parameters,
        metadata={
            **dict(geometry.metadata),
            "switch_graph_schema": graph.schema,
            "switch_graph_source_smiles": graph.source_smiles,
            "switch_graph_atom_count": len(graph.atoms),
            "switch_graph_bond_count": len(graph.bonds),
            "switch_graph_formal_charge": graph.total_formal_charge,
        },
    )
    completion = complete_valence_hydrogens(
        contracted,
        tuple(bond.key for bond in graph.bonds),
        bond_orders={bond.key: bond.order for bond in graph.bonds},
        requested_counts=tuple(atom.hydrogen_count for atom in graph.atoms),
    )
    coordinates = np.asarray(completion.geometry.coordinates_angstrom, dtype=float).copy()
    aromatic_hydrogens: set[int] = set()
    for addition in completion.additions:
        parent = int(addition.parent)
        hydrogen = int(addition.atom)
        if not bool(graph.atoms[parent].aromatic) or len(graph.neighbors(parent)) != 2:
            continue
        neighbours = tuple(int(neighbour) for neighbour in graph.neighbors(parent))
        first = coordinates[neighbours[0]] - coordinates[parent]
        second = coordinates[neighbours[1]] - coordinates[parent]
        normal = np.cross(first, second)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1.0e-12:
            raise ValueError("cannot construct a plane for an aromatic hydrogen")
        normal /= normal_norm
        # The pre-completion Cartesian direction can point into the ring when
        # a shortest-path layout is spectrally degenerate.  For an aromatic
        # atom with two ring neighbours the chemically valid direction is the
        # outward bisector, independent of that provisional direction.
        unit_first = first / np.linalg.norm(first)
        unit_second = second / np.linalg.norm(second)
        direction = -(unit_first + unit_second)
        direction -= float(np.dot(direction, normal)) * normal
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1.0e-12:
            raise ValueError("cannot construct an in-plane aromatic hydrogen direction")
        direction /= direction_norm
        bond_length = float(np.linalg.norm(coordinates[hydrogen] - coordinates[parent]))
        coordinates[hydrogen] = coordinates[parent] + bond_length * direction
        aromatic_hydrogens.add(hydrogen)
    if aromatic_hydrogens:
        completion = HydrogenCompletion(
            geometry=MolecularGeometry(
                atoms=completion.geometry.atoms,
                coordinates_angstrom=coordinates,
                comment=completion.geometry.comment,
                source_format=completion.geometry.source_format,
                source_path=completion.geometry.source_path,
                charge=completion.geometry.charge,
                multiplicity=completion.geometry.multiplicity,
                fixed_parameters=completion.geometry.fixed_parameters,
                metadata={
                    **dict(completion.geometry.metadata),
                    "planar_aromatic_hydrogens": tuple(sorted(aromatic_hydrogens)),
                },
            ),
            bonds=completion.bonds,
            additions=completion.additions,
            schema=completion.schema,
        )
    if not release_clashes:
        return completion
    protected = set(
        int(index)
        for index in completion.geometry.metadata.get("planar_amide_hydrogens", ())
    )
    protected.update(
        int(index)
        for index in completion.geometry.metadata.get("planar_aromatic_hydrogens", ())
    )
    movable = tuple(
        int(addition.atom)
        for addition in completion.additions
        if int(addition.atom) not in protected
    )
    if not movable:
        return completion
    return release_completed_hydrogen_clashes(
        completion,
        minimum_separation_angstrom=minimum_separation_angstrom,
        atom_indices=movable,
    )


def _completion_with_coordinates(
    completion: HydrogenCompletion,
    coordinates: np.ndarray,
) -> HydrogenCompletion:
    geometry = completion.geometry
    return HydrogenCompletion(
        geometry=MolecularGeometry(
            atoms=geometry.atoms,
            coordinates_angstrom=coordinates,
            comment=geometry.comment,
            source_format=geometry.source_format,
            source_path=geometry.source_path,
            charge=geometry.charge,
            multiplicity=geometry.multiplicity,
            fixed_parameters=geometry.fixed_parameters,
            metadata=geometry.metadata,
        ),
        bonds=completion.bonds,
        additions=completion.additions,
        schema=completion.schema,
    )


def _restore_projected_hydrogen_lengths(
    coordinates: np.ndarray,
    additions,
    *,
    parent_symbols=None,
    bonds=(),
) -> None:
    """Keep completed X--H bonds physical after exact planar projection.

    A projected seed must not preserve a degenerate provisional direction.  In
    particular, an O--H generated as a continuation of the O--C vector gives
    a 180-degree bend and can make Gaussian/GDV redundant coordinates
    singular.  For a single H on a centre with one heavy neighbour, rebuild
    the direction from the coordination-specific valence angle while retaining
    the existing in-plane side whenever it is defined.
    """
    grouped: dict[int, list] = {}
    for addition in additions:
        grouped.setdefault(int(addition.parent), []).append(addition)
    adjacency: dict[int, list[int]] = {}
    for left, right in bonds:
        left, right = int(left), int(right)
        adjacency.setdefault(left, []).append(right)
        adjacency.setdefault(right, []).append(left)
    for parent, parent_additions in grouped.items():
        count = len(parent_additions)
        preserve_projected = count == 1
        for rank, addition in enumerate(parent_additions):
            hydrogen = int(addition.atom)
            if count > 1 and not preserve_projected:
                symbol = (
                    str(getattr(parent_symbols[parent], "symbol", parent_symbols[parent]))
                    if parent_symbols is not None
                    else "C"
                )
                separation = {
                    "O": np.deg2rad(104.5),
                    "N": np.deg2rad(107.0),
                }.get(symbol, np.deg2rad(109.5))
                if count > 2:
                    separation = 2.0 * np.pi / 3.0
                angle = (
                    parent * 2.399963229728653 + (rank - 0.5 * (count - 1)) * separation + 0.37
                ) % (2.0 * np.pi)
                vector = np.asarray((np.cos(angle), np.sin(angle), 0.0))
            elif len(adjacency.get(parent, ())) == 1:
                neighbour = adjacency[parent][0]
                axis = coordinates[neighbour] - coordinates[parent]
                axis[2] = 0.0
                axis_norm = float(np.linalg.norm(axis))
                if axis_norm < 1.0e-8:
                    axis = np.asarray((1.0, 0.0, 0.0))
                else:
                    axis /= axis_norm
                symbol = str(
                    getattr(parent_symbols[parent], "symbol", parent_symbols[parent])
                    if parent_symbols is not None
                    else "C"
                )
                target = {
                    "O": np.deg2rad(104.5),
                    "N": np.deg2rad(107.0),
                }.get(symbol, np.deg2rad(109.47122063449069))
                current = coordinates[hydrogen] - coordinates[parent]
                current[2] = 0.0
                cross_z = float(axis[0] * current[1] - axis[1] * current[0])
                if abs(cross_z) < 1.0e-8:
                    sign = 1.0 if (parent + hydrogen) % 2 == 0 else -1.0
                else:
                    sign = 1.0 if cross_z > 0.0 else -1.0
                perpendicular = np.asarray((-axis[1], axis[0], 0.0))
                vector = np.cos(target) * axis + sign * np.sin(target) * perpendicular
            else:
                vector = coordinates[hydrogen] - coordinates[parent]
                vector[2] = 0.0
            norm = float(np.linalg.norm(vector))
            if norm < 1.0e-8:
                angle = (parent * 2.399963229728653 + hydrogen * 0.618033988749895) % (2.0 * np.pi)
                vector = np.asarray((np.cos(angle), np.sin(angle), 0.0))
                norm = 1.0
            coordinates[hydrogen] = (
                coordinates[parent] + float(addition.bond_length_angstrom) * vector / norm
            )


def _release_projected_hydrogen_clashes(
    coordinates: np.ndarray,
    additions: tuple[tuple[int, int], ...],
) -> None:
    """Resolve post-projection X--H clashes while retaining a planar seed."""

    angles = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
    directions = np.column_stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)))
    for parent, hydrogen in additions:
        origin = coordinates[parent]
        length = float(np.linalg.norm(coordinates[hydrogen] - origin))
        others = np.asarray(
            [
                coordinates[index]
                for index in range(len(coordinates))
                if index not in {parent, hydrogen}
            ],
            dtype=float,
        )
        if not len(others):
            continue
        candidates = origin + max(length, 0.85) * directions
        minimum_separations = np.min(
            np.linalg.norm(candidates[:, np.newaxis, :] - others[np.newaxis, :, :], axis=2),
            axis=1,
        )
        current_minimum = float(np.min(np.linalg.norm(coordinates[hydrogen] - others, axis=1)))
        if current_minimum < 1.35:
            coordinates[hydrogen] = candidates[int(np.argmax(minimum_separations))]


def _component_coordinates(
    graph: SwitchMolecularGraph,
    component: tuple[int, ...],
) -> np.ndarray:
    count = len(component)
    if count == 1:
        return np.zeros((1, 3), dtype=float)
    if count > DEFAULT_DENSE_LAYOUT_MAX_ATOMS:
        return _linear_component_coordinates(graph, component)
    lookup = {atom: local for local, atom in enumerate(component)}
    rows: list[int] = []
    columns: list[int] = []
    lengths: list[float] = []
    for bond in graph.bonds:
        if bond.left not in lookup or bond.right not in lookup:
            continue
        value = _target_bond_length(graph, bond)
        left, right = lookup[bond.left], lookup[bond.right]
        rows.extend((left, right))
        columns.extend((right, left))
        lengths.extend((value, value))
    adjacency = csr_matrix((lengths, (rows, columns)), shape=(count, count))
    distances = np.asarray(
        shortest_path(adjacency, directed=False, unweighted=False),
        dtype=float,
    )
    if not np.all(np.isfinite(distances)):
        raise ValueError("SWITCH component is not connected")
    centering = np.eye(count) - np.full((count, count), 1.0 / count)
    gram = -0.5 * centering @ (distances * distances) @ centering
    values, vectors = eigh_arrays(gram)
    order = np.argsort(values)[::-1]
    coordinates = np.zeros((count, 3), dtype=float)
    for axis, eigen_index in enumerate(order[:3]):
        if values[eigen_index] > 1.0e-10:
            coordinates[:, axis] = vectors[:, eigen_index] * np.sqrt(values[eigen_index])
            pivot = int(np.argmax(np.abs(coordinates[:, axis])))
            if coordinates[pivot, axis] < 0.0:
                coordinates[:, axis] *= -1.0
    rank = int(np.count_nonzero(np.asarray(values) > 1.0e-10))
    if rank < 3 and count > 2:
        indices = np.arange(count, dtype=float)
        coordinates[:, 2] += 0.06 * np.sin(indices * 2.399963229728653)
        if rank < 2:
            coordinates[:, 1] += 0.06 * np.cos(indices * 1.618033988749895)
    refined = _refine_graph_geometry(graph, component, coordinates)
    refined = _repair_collapsed_aromatic_rings(graph, component, refined)
    if _graph_prefers_planar_seed(graph, component):
        for local_index in _cycle_atom_indices(graph, component):
            refined[local_index, 2] = 0.0
    return refined


def _repair_collapsed_aromatic_rings(
    graph: SwitchMolecularGraph,
    component: tuple[int, ...],
    coordinates: np.ndarray,
) -> np.ndarray:
    """Replace singular aromatic cycles by a deterministic regular polygon.

    The graph-distance/MDS layout is intentionally generic, but a short
    aromatic cycle can become spectrally singular when two equivalent atoms
    have the same shortest-path distances.  Repairing the cycle before
    hydrogen completion is essential: otherwise an aromatic H may be
    perceived as bonded to two neighbouring nuclei.  This is a graph-level
    rule and therefore applies to every aromatic heterocycle, not only to
    histidine.
    """

    selected = set(int(atom) for atom in component)
    aromatic = {int(atom) for atom in component if graph.atoms[atom].aromatic}
    aromatic_adjacency: dict[int, list[int]] = {atom: [] for atom in aromatic}
    for bond in graph.bonds:
        left, right = int(bond.left), int(bond.right)
        if left in aromatic and right in aromatic and bond.aromatic:
            aromatic_adjacency[left].append(right)
            aromatic_adjacency[right].append(left)
    repaired = np.asarray(coordinates, dtype=float).copy()
    local = {int(atom): index for index, atom in enumerate(component)}
    visited: set[int] = set()
    for start in sorted(aromatic):
        if start in visited or len(aromatic_adjacency.get(start, ())) != 2:
            continue
        cycle = [start]
        previous: int | None = None
        current = start
        while True:
            neighbours = sorted(aromatic_adjacency[current])
            next_atom = neighbours[0] if neighbours[0] != previous else neighbours[1]
            if next_atom == start:
                break
            if next_atom in cycle or len(cycle) > len(aromatic):
                cycle = []
                break
            cycle.append(next_atom)
            previous, current = current, next_atom
        if len(cycle) < 3 or any(len(aromatic_adjacency[a]) != 2 for a in cycle):
            continue
        visited.update(cycle)
        before_ring_repair = repaired.copy()
        cycle_local = np.asarray([local[a] for a in cycle], dtype=int)
        current_xyz = repaired[cycle_local]
        center = np.mean(current_xyz, axis=0)
        # Aromatic seeds are projected onto the XY plane immediately after
        # this routine.  Build the repair in that same plane; using the
        # provisional MDS normal here would make the later projection collapse
        # the ring again.
        normal = np.asarray((0.0, 0.0, 1.0))
        attachment = None
        cycle_set = set(cycle)
        for atom in cycle:
            external = [
                int(neighbour)
                for neighbour in graph.neighbors(atom)
                if int(neighbour) in selected and int(neighbour) not in cycle_set
            ]
            if external:
                attachment = (atom, external[0])
                break
        if attachment is not None:
            atom, external = attachment
            direction = repaired[local[external]] - center
        else:
            direction = current_xyz[0] - center
        direction -= float(np.dot(direction, normal)) * normal
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1.0e-10:
            direction = np.asarray((1.0, 0.0, 0.0))
            direction -= float(np.dot(direction, normal)) * normal
            direction_norm = float(np.linalg.norm(direction))
        e1 = direction / direction_norm
        e2 = np.cross(normal, e1)
        e2 /= max(float(np.linalg.norm(e2)), 1.0e-12)
        bond_lengths = []
        for left, right in zip(cycle, cycle[1:] + cycle[:1]):
            bond = next(
                item for item in graph.bonds
                if {int(item.left), int(item.right)} == {left, right}
            )
            bond_lengths.append(_target_bond_length(graph, bond))
        bond_length = float(np.mean(bond_lengths))
        radius = bond_length / (2.0 * np.sin(np.pi / len(cycle)))
        for index, atom in enumerate(cycle):
            angle = 2.0 * np.pi * index / len(cycle)
            repaired[local[atom]] = center + radius * (np.cos(angle) * e1 + np.sin(angle) * e2)
        # Preserve every acyclic substituent attached through one ring atom.
        # Moving the regular polygon without transporting these components
        # stretches the exocyclic constitutional bond even though the graph
        # refinement immediately above had already established its target
        # length.  A rigid translation by the anchor displacement preserves
        # all internal distances and applies to any singly anchored branch.
        remaining = selected - cycle_set
        while remaining:
            seed = min(remaining)
            stack = [seed]
            branch: set[int] = set()
            while stack:
                branch_atom = stack.pop()
                if branch_atom not in remaining:
                    continue
                remaining.remove(branch_atom)
                branch.add(branch_atom)
                stack.extend(
                    int(neighbour)
                    for neighbour in graph.neighbors(branch_atom)
                    if int(neighbour) in remaining
                )
            anchors = {
                int(neighbour)
                for branch_atom in branch
                for neighbour in graph.neighbors(branch_atom)
                if int(neighbour) in cycle_set
            }
            if len(anchors) != 1:
                continue
            anchor = next(iter(anchors))
            shift = repaired[local[anchor]] - before_ring_repair[local[anchor]]
            branch_local = np.asarray(sorted(local[atom] for atom in branch), dtype=int)
            repaired[branch_local] = before_ring_repair[branch_local] + shift
    return repaired


def _cycle_atom_indices(
    graph: SwitchMolecularGraph,
    component: tuple[int, ...],
) -> set[int]:
    """Return local atom indices that belong to at least one graph cycle."""

    lookup = {atom: local for local, atom in enumerate(component)}
    edges = {
        tuple(sorted((lookup[bond.left], lookup[bond.right])))
        for bond in graph.bonds
        if bond.left in lookup and bond.right in lookup
    }
    adjacency: list[list[int]] = [[] for _ in component]
    for left, right in edges:
        adjacency[left].append(right)
        adjacency[right].append(left)
    cyclic: set[int] = set()
    for left, right in edges:
        parent: dict[int, int | None] = {left: None}
        queue = [left]
        while queue and right not in parent:
            current = queue.pop(0)
            for neighbor in adjacency[current]:
                if (current == left and neighbor == right) or (
                    current == right and neighbor == left
                ):
                    continue
                if neighbor not in parent:
                    parent[neighbor] = current
                    queue.append(neighbor)
        if right in parent:
            current: int | None = right
            while current is not None:
                cyclic.add(current)
                current = parent[current]
    return cyclic


def _graph_prefers_planar_seed(
    graph: SwitchMolecularGraph,
    component: tuple[int, ...] | None = None,
) -> bool:
    selected = set(range(len(graph.atoms)) if component is None else component)
    bonds = [bond for bond in graph.bonds if bond.left in selected and bond.right in selected]
    component_count = sum(1 for item in graph.components if set(item).issubset(selected)) or 1
    cycle_rank = len(bonds) - len(selected) + component_count
    return cycle_rank > 0 and (
        any(bond.aromatic for bond in bonds)
        or (cycle_rank == 1 and sum(bond.order >= 1.75 for bond in bonds) >= 2)
    )


def _refine_graph_geometry(
    graph: SwitchMolecularGraph,
    component: tuple[int, ...],
    initial: np.ndarray,
) -> np.ndarray:
    """Remove spectral-layout degeneracies while preserving every graph edge.

    Classical multidimensional scaling is an excellent deterministic global
    layout, but symmetry-equivalent vertices can coincide and its least-square
    compromise does not enforce individual bond lengths.  A short analytic
    spring refinement makes the seed safe for force-field relaxation.  It is
    deliberately only a graph realization: no chemical energy is evaluated.
    """

    count = len(component)
    lookup = {atom: local for local, atom in enumerate(component)}
    edge_terms: list[tuple[int, int, float]] = []
    edge_keys: set[tuple[int, int]] = set()
    edge_lengths: dict[tuple[int, int], float] = {}
    incident_bonds: list[list[SwitchBond]] = [[] for _ in component]
    for bond in graph.bonds:
        if bond.left not in lookup or bond.right not in lookup:
            continue
        left, right = lookup[bond.left], lookup[bond.right]
        target = _target_bond_length(graph, bond)
        key = tuple(sorted((left, right)))
        edge_terms.append((left, right, target))
        edge_keys.add(key)
        edge_lengths[key] = target
        incident_bonds[left].append(bond)
        incident_bonds[right].append(bond)

    # A shortest-path distance matrix alone makes every acyclic three-atom
    # path prefer a straight line.  Encode local valence before hydrogens are
    # materialized by constraining the corresponding 1--3 chord.  Ring-only
    # angles are left to their cycle geometry; exocyclic pairs retain the
    # local-valence constraint.
    bridge_keys = _bridge_edge_keys(count, edge_keys)
    angle_chords: list[tuple[int, int, float]] = []
    adjacency: list[list[int]] = [[] for _ in component]
    for left, right in edge_keys:
        adjacency[left].append(right)
        adjacency[right].append(left)
    for center, neighbors in enumerate(adjacency):
        target_angle = _preferred_valence_angle(
            graph,
            component[center],
            incident_bonds[center],
        )
        if target_angle is None or len(neighbors) < 2:
            continue
        cosine = float(np.cos(target_angle))
        ordered = sorted(neighbors)
        for position, left in enumerate(ordered):
            center_left = tuple(sorted((center, left)))
            for right in ordered[position + 1 :]:
                center_right = tuple(sorted((center, right)))
                if tuple(sorted((left, right))) in edge_keys:
                    continue
                if center_left not in bridge_keys and center_right not in bridge_keys:
                    continue
                left_length = edge_lengths[center_left]
                right_length = edge_lengths[center_right]
                chord = np.sqrt(
                    max(
                        1.0e-12,
                        left_length * left_length
                        + right_length * right_length
                        - 2.0 * left_length * right_length * cosine,
                    )
                )
                angle_chords.append((left, right, float(chord)))

    # Separate nonbonded nuclei enough for xTB to accept the seed.  The tiny,
    # deterministic three-dimensional offset resolves exact spectral
    # degeneracies without introducing a random-state contract.
    coordinates = np.asarray(initial, dtype=float).copy()
    indices = np.arange(count, dtype=float)
    coordinates += 1.0e-3 * np.column_stack(
        (
            np.sin(indices * 1.3247179572447458),
            np.cos(indices * 1.618033988749895),
            np.sin(indices * 2.399963229728653),
        )
    )
    # Classical MDS can place branch atoms almost exactly on top of one
    # another (notably the two oxygens of a carboxyl group). A local
    # optimizer started at that singularity may never recover a physical
    # bond length, and hydrogen completion then sees spurious bonds. Seed
    # those degeneracies apart deterministically; elastic refinement below
    # restores the graph-specific target lengths.
    for left in range(count):
        for right in range(left + 1, count):
            delta = coordinates[right] - coordinates[left]
            distance = float(np.linalg.norm(delta))
            if distance >= 0.75:
                continue
            angle = (left * 2.399963229728653 + right * 1.618033988749895) % (2.0 * np.pi)
            direction = np.asarray((np.cos(angle), np.sin(angle), 0.17))
            direction /= np.linalg.norm(direction)
            midpoint = 0.5 * (coordinates[left] + coordinates[right])
            separation = 0.85
            coordinates[left] = midpoint - 0.5 * separation * direction
            coordinates[right] = midpoint + 0.5 * separation * direction
    reference = coordinates.copy()
    atom_radii = []
    for atom_index in component:
        number = atomic_number(graph.atoms[atom_index].symbol)
        atom_radii.append(float(covalent_radius(number) or 0.77) if number else 0.77)
    nonbonded: list[tuple[int, int, float]] = []
    for left in range(count):
        for right in range(left + 1, count):
            if (left, right) in edge_keys:
                continue
            nonbonded.append(
                (left, right, max(0.85, 1.30 * (atom_radii[left] + atom_radii[right])))
            )

    def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        xyz = flat.reshape(count, 3)
        gradient = np.zeros_like(xyz)
        value = 0.0
        for left, right, target in edge_terms:
            delta = xyz[left] - xyz[right]
            distance = max(float(np.linalg.norm(delta)), 1.0e-10)
            residual = distance - target
            value += 32.0 * residual * residual
            force = (64.0 * residual / distance) * delta
            gradient[left] += force
            gradient[right] -= force
        for left, right, target in angle_chords:
            delta = xyz[left] - xyz[right]
            distance = max(float(np.linalg.norm(delta)), 1.0e-10)
            residual = distance - target
            value += 12.0 * residual * residual
            force = (24.0 * residual / distance) * delta
            gradient[left] += force
            gradient[right] -= force
        for left, right, minimum in nonbonded:
            delta = xyz[left] - xyz[right]
            distance = max(float(np.linalg.norm(delta)), 1.0e-10)
            if distance >= minimum:
                continue
            residual = minimum - distance
            value += 4.0 * residual * residual
            force = (-8.0 * residual / distance) * delta
            gradient[left] += force
            gradient[right] -= force
        displacement = xyz - reference
        value += 1.0e-3 * float(np.sum(displacement * displacement))
        gradient += 2.0e-3 * displacement
        # Remove the translational null mode explicitly.
        gradient -= np.mean(gradient, axis=0)
        return value, gradient.ravel()

    result = minimize(
        objective,
        coordinates.ravel(),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 1000, "ftol": 0.0, "gtol": 1.0e-8, "maxls": 50},
    )
    refined = np.asarray(result.x, dtype=float).reshape(count, 3)
    # L-BFGS can terminate at a singular local layout when two bonded atoms
    # start nearly coincident. Repair only these pathological edges with a
    # deterministic SHAKE-like projection; ordinary optimized distances are
    # left untouched.
    for _ in range(6):
        repaired = False
        for left, right, target in edge_terms:
            delta = refined[right] - refined[left]
            distance = float(np.linalg.norm(delta))
            if distance >= 0.75:
                continue
            angle = (left * 2.399963229728653 + right * 1.618033988749895) % (2.0 * np.pi)
            direction = np.asarray((np.cos(angle), np.sin(angle), 0.17))
            direction /= np.linalg.norm(direction)
            if distance > 1.0e-8:
                direction = delta / distance
            midpoint = 0.5 * (refined[left] + refined[right])
            refined[left] = midpoint - 0.5 * target * direction
            refined[right] = midpoint + 0.5 * target * direction
            repaired = True
        if not repaired:
            break
    refined -= np.mean(refined, axis=0)
    return refined


def _bridge_edge_keys(
    count: int,
    edge_keys: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    """Return graph bridges in local component indexing."""

    adjacency: list[list[int]] = [[] for _ in range(count)]
    for left, right in edge_keys:
        adjacency[left].append(right)
        adjacency[right].append(left)
    discovery = [-1] * count
    low = [0] * count
    bridges: set[tuple[int, int]] = set()
    clock = 0

    def visit(atom: int, parent: int) -> None:
        nonlocal clock
        discovery[atom] = low[atom] = clock
        clock += 1
        for neighbor in adjacency[atom]:
            if neighbor == parent:
                continue
            if discovery[neighbor] < 0:
                visit(neighbor, atom)
                low[atom] = min(low[atom], low[neighbor])
                if low[neighbor] > discovery[atom]:
                    bridges.add(tuple(sorted((atom, neighbor))))
            else:
                low[atom] = min(low[atom], discovery[neighbor])

    for atom in range(count):
        if discovery[atom] < 0:
            visit(atom, -1)
    return bridges


def _preferred_valence_angle(
    graph: SwitchMolecularGraph,
    atom_index: int,
    incident: list[SwitchBond],
) -> float | None:
    """Return a basic-knowledge valence angle in radians.

    The rule uses total local valence, including hydrogens still implicit in
    the SMILES graph.  It supplies a deterministic 1--3 seed constraint, not a
    force-field parameter.
    """

    atom = graph.atoms[atom_index]
    number = atomic_number(atom.symbol)
    if number is None or len(incident) < 2:
        return None
    if atom.aromatic or any(bond.aromatic for bond in incident):
        return 2.0 * np.pi / 3.0
    maximum_order = max(float(bond.order) for bond in incident)
    if maximum_order >= 2.5:
        return np.pi
    if maximum_order >= 1.75:
        return 2.0 * np.pi / 3.0

    coordination = len(incident) + _seed_hydrogen_count(
        graph,
        atom_index,
        incident,
    )
    if coordination > 4:
        return None
    if number in {5, 6, 14, 32}:
        if coordination >= 4:
            return float(np.arccos(-1.0 / 3.0))
        if coordination == 3:
            return 2.0 * np.pi / 3.0
        if coordination == 2:
            return np.pi
    if number in {7, 8, 15, 16, 33, 34} and coordination >= 2:
        return float(np.arccos(-1.0 / 3.0))
    if coordination >= 4:
        return float(np.arccos(-1.0 / 3.0))
    if coordination == 3:
        return 2.0 * np.pi / 3.0
    return None


def _linear_component_coordinates(
    graph: SwitchMolecularGraph,
    component: tuple[int, ...],
) -> np.ndarray:
    """O(V+E) seed for large graphs, intended for subsequent FF relaxation."""

    lookup = {atom: local for local, atom in enumerate(component)}
    adjacency: list[list[tuple[int, float]]] = [[] for _ in component]
    for bond in graph.bonds:
        left = lookup.get(bond.left)
        right = lookup.get(bond.right)
        if left is None or right is None:
            continue
        length = _target_bond_length(graph, bond)
        adjacency[left].append((right, length))
        adjacency[right].append((left, length))
    for neighbors in adjacency:
        neighbors.sort(key=lambda item: item[0])

    coordinates = np.full((len(component), 3), np.nan, dtype=float)
    coordinates[0] = 0.0
    queue = [0]
    cursor = 0
    while cursor < len(queue):
        parent = queue[cursor]
        cursor += 1
        occupied = []
        for neighbor, _ in adjacency[parent]:
            if np.all(np.isfinite(coordinates[neighbor])):
                vector = coordinates[neighbor] - coordinates[parent]
                norm = float(np.linalg.norm(vector))
                if norm > 1.0e-12:
                    occupied.append(vector / norm)
        for ordinal, (neighbor, length) in enumerate(adjacency[parent]):
            if np.all(np.isfinite(coordinates[neighbor])):
                continue
            offset = (parent * 37 + neighbor * 17 + ordinal * 13) % _DIRECTION_COUNT
            candidates = np.roll(_DIRECTIONS, -offset, axis=0)
            if occupied:
                score = np.max(candidates @ np.asarray(occupied).T, axis=1)
                direction = candidates[int(np.argmin(score))]
            else:
                direction = candidates[0]
            coordinates[neighbor] = coordinates[parent] + float(length) * direction
            occupied.append(direction)
            queue.append(neighbor)
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("SWITCH component is not connected")
    coordinates -= np.mean(coordinates, axis=0)
    return coordinates


def _target_bond_length(graph: SwitchMolecularGraph, bond: SwitchBond) -> float:
    left_radius = _seed_covalent_radius(graph, bond.left)
    right_radius = _seed_covalent_radius(graph, bond.right)
    base = float(left_radius or 0.77) + float(right_radius or 0.77)
    if bond.order >= 3.5:
        factor = 0.78
    elif bond.order >= 2.5:
        factor = 0.82
    elif bond.order >= 1.75:
        factor = 0.88
    elif bond.aromatic or bond.order > 1.25:
        factor = 0.92
    else:
        factor = 1.0
    return max(0.65, base * factor)


def _seed_covalent_radius(graph: SwitchMolecularGraph, atom_index: int) -> float | None:
    """Use Pyykkö's coordination-specific radius for a SMILES atom."""

    atom = graph.atoms[atom_index]
    number = atomic_number(atom.symbol)
    if number is None:
        return None
    incident = [bond for bond in graph.bonds if atom_index in {bond.left, bond.right}]
    degree = len(incident)
    hydrogens = _seed_hydrogen_count(graph, atom_index, incident)
    return covalent_radius(number, max(1, degree + hydrogens))


def _seed_hydrogen_count(
    graph: SwitchMolecularGraph,
    atom_index: int,
    incident: list[SwitchBond] | None = None,
) -> int:
    """Return the explicit-or-inferred hydrogen count used by seed geometry."""

    atom = graph.atoms[atom_index]
    if incident is None:
        incident = [bond for bond in graph.bonds if atom_index in {bond.left, bond.right}]
    degree = len(incident)
    if atom.hydrogen_count is not None:
        return max(0, int(atom.hydrogen_count))
    if atom.aromatic:
        return 1 if atom.symbol in {"B", "C"} and degree == 2 else 0
    number = atomic_number(atom.symbol)
    target_valence = {
        5: 3.0,
        6: 4.0,
        7: 4.0 if atom.formal_charge > 0 else 3.0,
        8: 3.0 if atom.formal_charge > 0 else 2.0,
        14: 4.0,
        15: 3.0,
        16: 2.0,
        32: 4.0,
        33: 3.0,
        34: 2.0,
    }.get(number)
    if target_valence is None:
        return 0
    used_valence = sum(float(bond.order) for bond in incident)
    return max(0, int(round(target_valence - used_valence)))


def _apply_tetrahedral_chirality(
    graph: SwitchMolecularGraph,
    coordinates: np.ndarray,
    added_hydrogens: tuple[tuple[int, int], ...],
) -> int:
    added_by_parent: dict[int, list[int]] = {}
    for parent, hydrogen in added_hydrogens:
        added_by_parent.setdefault(parent, []).append(hydrogen)
    neighbor_order: list[list[int]] = [[] for _ in graph.atoms]
    for bond in graph.bonds:
        neighbor_order[bond.left].append(bond.right)
        neighbor_order[bond.right].append(bond.left)
    applied = 0
    for atom in graph.atoms:
        if atom.chirality not in {"@", "@@", "@TH1", "@TH2"}:
            continue
        neighbors = [
            *neighbor_order[atom.index],
            *added_by_parent.get(atom.index, ()),
        ]
        if len(neighbors) != 4:
            continue
        implicit = list(added_by_parent.get(atom.index, ()))
        explicit_hydrogens = [
            neighbor
            for neighbor in neighbor_order[atom.index]
            if graph.atoms[neighbor].symbol == "H"
        ]
        terminal_neighbors = [
            neighbor
            for neighbor in neighbor_order[atom.index]
            if len(neighbor_order[neighbor]) == 1
        ]
        movable = [*implicit, *explicit_hydrogens]
        if not movable:
            movable = terminal_neighbors[:1]
        if len(movable) == 1:
            ligand = movable[0]
            fixed = [neighbor for neighbor in neighbors if neighbor != ligand]
            if len(fixed) != 3:
                continue
            points = coordinates[np.asarray(fixed, dtype=int)]
            normal = np.cross(points[1] - points[0], points[2] - points[0])
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm < 1.0e-12:
                continue
            normal /= normal_norm
            current_volume = float(
                np.linalg.det(
                    np.column_stack(
                        (
                            points[0] - coordinates[ligand],
                            points[1] - coordinates[ligand],
                            points[2] - coordinates[ligand],
                        )
                    )
                )
            )
            desired_positive = atom.chirality in {"@", "@TH1"}
            original_length = float(np.linalg.norm(coordinates[ligand] - coordinates[atom.index]))
            if (current_volume > 0.0) != desired_positive:
                distance = float(np.dot(coordinates[ligand] - points[0], normal))
                reflected = coordinates[ligand] - 2.0 * distance * normal
                vector = reflected - coordinates[atom.index]
                vector_norm = float(np.linalg.norm(vector))
                if vector_norm > 1.0e-12:
                    coordinates[ligand] = (
                        coordinates[atom.index] + original_length * vector / vector_norm
                    )
            if ligand >= len(graph.atoms) or (
                ligand < len(graph.atoms) and graph.atoms[ligand].symbol == "H"
            ):
                center = coordinates[atom.index]
                fixed_vectors = points - center
                fixed_norms = np.linalg.norm(fixed_vectors, axis=1)
                if np.all(fixed_norms > 1.0e-12):
                    outward = -np.sum(
                        fixed_vectors / fixed_norms[:, np.newaxis],
                        axis=0,
                    )
                    outward_norm = float(np.linalg.norm(outward))
                    if outward_norm > 1.0e-12:
                        outward_point = center + original_length * outward / outward_norm
                        excluded = {atom.index, ligand}
                        others = np.asarray(
                            [
                                coordinates[index]
                                for index in range(len(coordinates))
                                if index not in excluded
                            ]
                        )
                        current_clearance = float(
                            np.min(
                                np.linalg.norm(
                                    others - coordinates[ligand],
                                    axis=1,
                                )
                            )
                        )
                        outward_clearance = float(
                            np.min(
                                np.linalg.norm(
                                    others - outward_point,
                                    axis=1,
                                )
                            )
                        )
                        if outward_clearance > current_clearance + 0.1:
                            coordinates[ligand] = outward_point
            applied += 1
            continue
        # A fully substituted cyclic center has no independently movable
        # ligand.  Its marker is retained in the graph and GFN-FF receives the
        # intact connectivity; tearing several ring bonds merely to impose a
        # tetrahedral template would be a worse seed than deferring this case.
    return applied


def _apply_coordination_stereochemistry(
    graph: SwitchMolecularGraph,
    coordinates: np.ndarray,
) -> int:
    neighbor_order: list[list[int]] = [[] for _ in graph.atoms]
    for bond in graph.bonds:
        neighbor_order[bond.left].append(bond.right)
        neighbor_order[bond.right].append(bond.left)
    applied = 0
    for atom in graph.atoms:
        marker = atom.chirality or ""
        if marker.startswith("@SP"):
            template = np.asarray(
                ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0))
            )
            variant = int(marker[3:] or "1")
            if variant == 2:
                template[[2, 3]] = template[[3, 2]]
            elif variant == 3:
                template[[1, 2]] = template[[2, 1]]
        elif marker.startswith("@TB"):
            template = np.asarray(
                (
                    (0.0, 0.0, 1.0),
                    (0.0, 0.0, -1.0),
                    (1.0, 0.0, 0.0),
                    (-0.5, np.sqrt(3.0) / 2.0, 0.0),
                    (-0.5, -np.sqrt(3.0) / 2.0, 0.0),
                )
            )
            template = np.roll(template, -(int(marker[3:]) - 1) % len(template), axis=0)
        elif marker.startswith("@OH"):
            template = np.asarray(
                (
                    (1.0, 0.0, 0.0),
                    (-1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, -1.0, 0.0),
                    (0.0, 0.0, 1.0),
                    (0.0, 0.0, -1.0),
                )
            )
            template = np.roll(template, -(int(marker[3:]) - 1) % len(template), axis=0)
        else:
            continue
        neighbors = neighbor_order[atom.index]
        if len(neighbors) != len(template):
            continue
        center = coordinates[atom.index].copy()
        for neighbor, direction in zip(neighbors, template, strict=True):
            length = max(0.8, float(np.linalg.norm(coordinates[neighbor] - center)))
            coordinates[neighbor] = center + length * direction
        applied += 1
    return applied


def _apply_directional_double_bonds(
    graph: SwitchMolecularGraph,
    coordinates: np.ndarray,
) -> int:
    adjacency: list[list[SwitchBond]] = [[] for _ in graph.atoms]
    for bond in graph.bonds:
        adjacency[bond.left].append(bond)
        adjacency[bond.right].append(bond)
    applied = 0
    for double in graph.bonds:
        if double.order < 1.75 or double.order >= 2.5:
            continue
        left_bonds = [
            bond
            for bond in adjacency[double.left]
            if bond is not double and bond.direction in {"/", "\\"}
        ]
        right_bonds = [
            bond
            for bond in adjacency[double.right]
            if bond is not double and bond.direction in {"/", "\\"}
        ]
        if not left_bonds or not right_bonds:
            continue
        left_directional = left_bonds[0]
        right_directional = right_bonds[0]
        left_substituent = (
            left_directional.right
            if left_directional.left == double.left
            else left_directional.left
        )
        right_substituent = (
            right_directional.right
            if right_directional.left == double.right
            else right_directional.left
        )
        left_center = coordinates[double.left]
        right_center = coordinates[double.right]
        axis = right_center - left_center
        norm = float(np.linalg.norm(axis))
        if norm < 1.0e-10:
            continue
        axis /= norm
        reference = np.asarray((0.0, 0.0, 1.0))
        if abs(float(np.dot(axis, reference))) > 0.9:
            reference = np.asarray((0.0, 1.0, 0.0))
        perpendicular = np.cross(axis, reference)
        perpendicular /= np.linalg.norm(perpendicular)
        left_length = max(
            0.8,
            float(np.linalg.norm(coordinates[left_substituent] - left_center)),
        )
        right_length = max(
            0.8,
            float(np.linalg.norm(coordinates[right_substituent] - right_center)),
        )
        axial = 0.5
        lateral = np.sqrt(3.0) / 2.0
        same_symbol = left_directional.direction == right_directional.direction
        coordinates[left_substituent] = left_center + left_length * (
            -axial * axis + lateral * perpendicular
        )
        coordinates[right_substituent] = right_center + right_length * (
            axial * axis + (-lateral if same_symbol else lateral) * perpendicular
        )
        applied += 1
    return applied


__all__ = [
    "DEFAULT_DENSE_LAYOUT_MAX_ATOMS",
    "SWITCH_GEOMETRY_SCHEMA",
    "build_cartesian_seed",
]
