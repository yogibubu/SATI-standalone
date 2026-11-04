from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..native_topology import compiled_elementary_cycle_basis, native_topology_backend

_CYCLE_BASIS_CACHE_MAXSIZE = 512
_CYCLE_BASIS_CACHE: dict[
    tuple[tuple[int, ...], tuple[tuple[int, int], ...], int | None],
    tuple[tuple[tuple[int, ...], ...], int, int],
] = {}


class GraphLike(Protocol):
    def neighbors(self, atom: int): ...


@dataclass(frozen=True)
class CycleBasisDiagnostics:
    candidate_cycle_count: int
    selected_cycle_count: int
    cycle_rank: int
    allowed_atom_count: int
    allowed_edge_count: int
    excluded_atoms: tuple[int, ...]
    algorithm: str = "HORTON_UNWEIGHTED_MCB"
    complete: bool = True
    maximum_selected_size: int = 0


def canonical_cycle(cycle: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    atoms = tuple(int(atom) for atom in cycle)
    if not atoms:
        return ()
    start = atoms.index(min(atoms))
    forward = atoms[start:] + atoms[:start]
    backward = (atoms[start],) + tuple(
        atoms[(start - offset) % len(atoms)] for offset in range(1, len(atoms))
    )
    return min(forward, backward)


def elementary_cycle_basis(
    graph: GraphLike,
    *,
    allowed_atoms: set[int],
    ring_max: int | None = None,
) -> tuple[tuple[tuple[int, ...], ...], CycleBasisDiagnostics]:
    edge_index = _allowed_edge_index(graph, allowed_atoms)
    cache_key = (tuple(sorted(allowed_atoms)), tuple(edge_index), ring_max)
    cached = _CYCLE_BASIS_CACHE.get(cache_key)
    if cached is None:
        precomputed = getattr(graph, "native_cycle_basis", None)
        precomputed_bonds = getattr(graph, "native_cycle_bonds", None)
        all_atoms = _all_graph_atoms(graph)
        if (
            precomputed is not None
            and ring_max is None
            and allowed_atoms == all_atoms
            and precomputed_bonds == tuple(edge_index)
            and _cycle_edges_belong_to_graph(precomputed, edge_index)
        ):
            cached = (
                tuple(precomputed),
                int(graph.native_cycle_candidate_count),
                int(graph.native_cycle_rank),
            )
        elif native_topology_backend(
            len(allowed_atoms) + len(edge_index)
        ).accelerated:
            selected, candidate_count, cycle_rank = compiled_elementary_cycle_basis(
                len(_all_graph_atoms(graph)),
                tuple(edge_index),
                allowed_atoms,
                ring_max,
            )
            cached = (selected, candidate_count, cycle_rank)
        else:
            cycle_rank = _cycle_rank(allowed_atoms, edge_index)
            candidate_cycles = (
                _horton_candidates(
                    graph,
                    allowed_atoms=allowed_atoms,
                    ring_max=ring_max,
                )
                if cycle_rank
                else set()
            )
            selected = _minimum_cycle_basis(
                candidate_cycles,
                edge_index,
                target_rank=cycle_rank,
            )
            cached = (selected, len(candidate_cycles), cycle_rank)
        if len(_CYCLE_BASIS_CACHE) >= _CYCLE_BASIS_CACHE_MAXSIZE:
            _CYCLE_BASIS_CACHE.pop(next(iter(_CYCLE_BASIS_CACHE)))
        _CYCLE_BASIS_CACHE[cache_key] = cached
    selected, candidate_cycle_count, cycle_rank = cached
    all_atoms = _all_graph_atoms(graph)
    diagnostics = CycleBasisDiagnostics(
        candidate_cycle_count=candidate_cycle_count,
        selected_cycle_count=len(selected),
        cycle_rank=cycle_rank,
        allowed_atom_count=len(allowed_atoms),
        allowed_edge_count=len(edge_index),
        excluded_atoms=tuple(sorted(all_atoms - allowed_atoms)),
        complete=len(selected) == cycle_rank,
        maximum_selected_size=max((len(cycle) for cycle in selected), default=0),
    )
    return selected, diagnostics


def _cycle_edges_belong_to_graph(
    cycles: tuple[tuple[int, ...], ...] | list[tuple[int, ...]],
    edges: dict[tuple[int, int], int],
) -> bool:
    """Certify that a native cycle basis still belongs to the final graph."""

    final_edges = set(edges)
    return all(
        tuple(sorted((cycle[index], cycle[(index + 1) % len(cycle)]))) in final_edges
        for cycle in cycles
        for index in range(len(cycle))
    )


def is_chordless_cycle(graph: GraphLike, cycle: tuple[int, ...]) -> bool:
    cycle_set = set(cycle)
    ring_edges = {
        tuple(sorted((cycle[index], cycle[(index + 1) % len(cycle)])))
        for index in range(len(cycle))
    }
    for atom in cycle:
        for neighbor in graph.neighbors(atom):
            if neighbor not in cycle_set:
                continue
            edge = tuple(sorted((atom, int(neighbor))))
            if edge not in ring_edges:
                return False
    return True


def _horton_candidates(
    graph: GraphLike,
    *,
    allowed_atoms: set[int],
    ring_max: int | None,
) -> set[tuple[int, ...]]:
    """Generate the polynomial Horton candidate set for an unweighted graph.

    One deterministic shortest-path tree is built from every allowed vertex.
    Every non-tree edge closes a fundamental cycle.  The union contains an
    unweighted minimum cycle basis, while avoiding enumeration of all simple
    or all chordless cycles in highly polycyclic graphs.
    """

    all_edges = tuple(_allowed_edge_index(graph, allowed_atoms))
    core_atoms = _cyclic_core(allowed_atoms, all_edges)
    edges = tuple(
        (left, right) for left, right in all_edges if left in core_atoms and right in core_atoms
    )
    adjacency = tuple(
        tuple(
            sorted(
                int(neighbor) for neighbor in graph.neighbors(atom) if int(neighbor) in core_atoms
            )
        )
        if atom in core_atoms
        else ()
        for atom in range(max(core_atoms, default=-1) + 1)
    )
    candidates: set[tuple[int, ...]] = set()
    processed: set[tuple[int, ...]] = set()
    roots = _horton_roots(core_atoms, adjacency)
    for root in roots:
        parent, depth = _shortest_path_tree_from_adjacency(adjacency, root)
        for left, right in edges:
            if parent[left] == right or parent[right] == left:
                continue
            cycle = _tree_cycle(left, right, parent, depth)
            if len(cycle) < 3:
                continue
            canonical = canonical_cycle(cycle)
            if ring_max is not None and len(canonical) > ring_max:
                continue
            if canonical in processed:
                continue
            processed.add(canonical)
            if _is_chordless_from_adjacency(adjacency, canonical):
                candidates.add(canonical)
            else:
                candidates.update(
                    part
                    for part in _split_at_chords_from_adjacency(adjacency, canonical)
                    if ring_max is None or len(part) <= ring_max
                )
    return candidates


def _horton_roots(
    core_atoms: set[int],
    adjacency: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    """Use branch vertices, plus one root for every isolated simple cycle."""

    roots = {atom for atom in core_atoms if len(adjacency[atom]) != 2}
    unseen = set(core_atoms)
    while unseen:
        seed = min(unseen)
        component = {seed}
        queue = [seed]
        for atom in queue:
            for neighbor in adjacency[atom]:
                if neighbor not in component:
                    component.add(neighbor)
                    queue.append(neighbor)
        unseen.difference_update(component)
        if roots.isdisjoint(component):
            roots.add(seed)
    return tuple(sorted(roots))


def _cyclic_core(
    allowed_atoms: set[int],
    edges: tuple[tuple[int, int], ...],
) -> set[int]:
    """Return the graph 2-core; every cycle is wholly contained in it."""

    adjacency = {atom: set() for atom in allowed_atoms}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    pending = [atom for atom, neighbors in adjacency.items() if len(neighbors) < 2]
    core = set(allowed_atoms)
    while pending:
        atom = pending.pop()
        if atom not in core:
            continue
        core.remove(atom)
        for neighbor in adjacency[atom]:
            if neighbor not in core:
                continue
            adjacency[neighbor].discard(atom)
            if len(adjacency[neighbor]) < 2:
                pending.append(neighbor)
    return core




def _shortest_path_tree_from_adjacency(
    adjacency: tuple[tuple[int, ...], ...],
    root: int,
) -> tuple[list[int], list[int]]:
    parent = [-2] * len(adjacency)
    depth = [-1] * len(adjacency)
    parent[root] = -1
    depth[root] = 0
    queue = [root]
    position = 0
    while position < len(queue):
        atom = queue[position]
        position += 1
        for neighbor in adjacency[atom]:
            if parent[neighbor] != -2:
                continue
            parent[neighbor] = atom
            depth[neighbor] = depth[atom] + 1
            queue.append(neighbor)
    return parent, depth


def _tree_cycle(
    left: int,
    right: int,
    parent: list[int],
    depth: list[int],
) -> tuple[int, ...]:
    if parent[left] == -2 or parent[right] == -2:
        return ()
    left_path: list[int] = []
    right_path: list[int] = []
    a, b = left, right
    while depth[a] > depth[b]:
        left_path.append(a)
        predecessor = parent[a]
        if predecessor < 0:
            return ()
        a = predecessor
    while depth[b] > depth[a]:
        right_path.append(b)
        predecessor = parent[b]
        if predecessor < 0:
            return ()
        b = predecessor
    while a != b:
        left_path.append(a)
        right_path.append(b)
        predecessor_a = parent[a]
        predecessor_b = parent[b]
        if predecessor_a < 0 or predecessor_b < 0:
            return ()
        a, b = predecessor_a, predecessor_b
    left_path.append(a)
    return tuple(left_path + list(reversed(right_path)))




def _split_at_chords_from_adjacency(
    adjacency: tuple[tuple[int, ...], ...],
    cycle: tuple[int, ...],
) -> set[tuple[int, ...]]:
    pending = [canonical_cycle(cycle)]
    chordless: set[tuple[int, ...]] = set()
    while pending:
        current = pending.pop()
        ring_edges = {
            tuple(sorted((current[index], current[(index + 1) % len(current)])))
            for index in range(len(current))
        }
        chord = None
        positions = {atom: index for index, atom in enumerate(current)}
        for atom in current:
            for neighbor in adjacency[atom]:
                edge = tuple(sorted((atom, neighbor)))
                if neighbor in positions and edge not in ring_edges:
                    chord = (positions[atom], positions[neighbor])
                    break
            if chord is not None:
                break
        if chord is None:
            chordless.add(canonical_cycle(current))
            continue
        first, second = sorted(chord)
        part_a = current[first : second + 1]
        part_b = current[second:] + current[: first + 1]
        if len(part_a) >= 3:
            pending.append(canonical_cycle(part_a))
        if len(part_b) >= 3:
            pending.append(canonical_cycle(part_b))
    return chordless


def _is_chordless_from_adjacency(
    adjacency: tuple[tuple[int, ...], ...],
    cycle: tuple[int, ...],
) -> bool:
    positions = {atom: index for index, atom in enumerate(cycle)}
    size = len(cycle)
    for atom, position in positions.items():
        previous = cycle[(position - 1) % size]
        following = cycle[(position + 1) % size]
        for neighbor in adjacency[atom]:
            if neighbor in positions and neighbor not in (previous, following):
                return False
    return True


def _minimum_cycle_basis(
    cycles: set[tuple[int, ...]],
    edge_index: dict[tuple[int, int], int],
    *,
    target_rank: int,
) -> tuple[tuple[int, ...], ...]:
    basis: dict[int, tuple[int, tuple[int, ...]]] = {}
    selected: list[tuple[int, ...]] = []
    for cycle in sorted(cycles, key=lambda item: (len(item), item)):
        vector = _cycle_vector(cycle, edge_index)
        reduced = vector
        for pivot in sorted(basis, reverse=True):
            if reduced & (1 << pivot):
                reduced ^= basis[pivot][0]
        if reduced == 0:
            continue
        pivot = reduced.bit_length() - 1
        basis[pivot] = (reduced, cycle)
        selected.append(cycle)
        if len(selected) >= target_rank:
            break
    return tuple(selected)


def _allowed_edge_index(graph: GraphLike, allowed_atoms: set[int]) -> dict[tuple[int, int], int]:
    graph_bonds = getattr(graph, "bonds", None)
    if graph_bonds is not None:
        if getattr(graph, "bonds_are_canonical_sorted", False):
            edges = [
                (int(left), int(right))
                for left, right in graph_bonds
                if int(left) in allowed_atoms and int(right) in allowed_atoms
            ]
        else:
            edges = sorted(
                (min(int(left), int(right)), max(int(left), int(right)))
                for left, right in graph_bonds
                if int(left) in allowed_atoms and int(right) in allowed_atoms
            )
        return {edge: index for index, edge in enumerate(edges)}
    edges = set()
    for atom in allowed_atoms:
        for neighbor_raw in graph.neighbors(atom):
            neighbor = int(neighbor_raw)
            if neighbor in allowed_atoms and atom < neighbor:
                edges.add((atom, neighbor))
    return {edge: index for index, edge in enumerate(sorted(edges))}


def _cycle_rank(allowed_atoms: set[int], edge_index: dict[tuple[int, int], int]) -> int:
    if not allowed_atoms:
        return 0
    parent = {atom: atom for atom in allowed_atoms}

    def find(atom: int) -> int:
        while parent[atom] != atom:
            parent[atom] = parent[parent[atom]]
            atom = parent[atom]
        return atom

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left, right in edge_index:
        union(left, right)
    components = {find(atom) for atom in allowed_atoms}
    return max(0, len(edge_index) - len(allowed_atoms) + len(components))


def _cycle_vector(cycle: tuple[int, ...], edge_index: dict[tuple[int, int], int]) -> int:
    vector = 0
    for index in range(len(cycle)):
        edge = tuple(sorted((cycle[index], cycle[(index + 1) % len(cycle)])))
        vector ^= 1 << edge_index[edge]
    return vector


def _all_graph_atoms(graph: GraphLike) -> set[int]:
    natoms = getattr(graph, "natoms", getattr(graph, "n_atoms", None))
    if natoms is None:
        return set()
    return set(range(int(natoms)))
