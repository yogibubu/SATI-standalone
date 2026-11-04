"""Versioned canonical graph ranking and SMILES serialization."""

from __future__ import annotations

from collections import defaultdict

from .model import SwitchAtom, SwitchBond, SwitchMolecularGraph


SWITCH_CANONICAL_SCHEMA = "matrix.switch.canonical.v1"


def canonical_atom_ranks(graph: SwitchMolecularGraph) -> tuple[int, ...]:
    """Return stable Weisfeiler--Lehman equivalence ranks."""

    adjacency: list[list[tuple[int, SwitchBond]]] = [[] for _ in graph.atoms]
    for bond in graph.bonds:
        adjacency[bond.left].append((bond.right, bond))
        adjacency[bond.right].append((bond.left, bond))
    signatures = [
        (
            atom.symbol,
            atom.isotope or 0,
            atom.formal_charge,
            -1 if atom.hydrogen_count is None else atom.hydrogen_count,
            atom.aromatic,
            atom.chirality or "",
            len(adjacency[atom.index]),
            tuple(sorted(_bond_code(bond) for _, bond in adjacency[atom.index])),
        )
        for atom in graph.atoms
    ]
    colors = _compress(signatures)
    while True:
        refined = [
            (
                colors[index],
                tuple(
                    sorted(
                        (_bond_code(bond), colors[neighbor])
                        for neighbor, bond in adjacency[index]
                    )
                ),
            )
            for index in range(len(graph.atoms))
        ]
        next_colors = _compress(refined)
        if next_colors == colors:
            return tuple(colors)
        colors = next_colors


def canonical_graph_key(graph: SwitchMolecularGraph) -> str:
    return f"{SWITCH_CANONICAL_SCHEMA}:{canonical_smiles(graph)}"


def canonical_smiles(graph: SwitchMolecularGraph) -> str:
    """Serialize a deterministic canonical constitutional SMILES."""

    ranks = canonical_atom_ranks(graph)
    component_strings = []
    for component in graph.components:
        candidates = [
            _serialize_component(graph, component, root, ranks)
            for root in component
        ]
        component_strings.append(min(candidates))
    return ".".join(sorted(component_strings))


def _serialize_component(
    graph: SwitchMolecularGraph,
    component: tuple[int, ...],
    root: int,
    ranks: tuple[int, ...],
) -> str:
    allowed = set(component)
    adjacency: list[list[tuple[int, SwitchBond]]] = [[] for _ in graph.atoms]
    for bond in graph.bonds:
        if bond.left in allowed and bond.right in allowed:
            adjacency[bond.left].append((bond.right, bond))
            adjacency[bond.right].append((bond.left, bond))
    for neighbors in adjacency:
        neighbors.sort(
            key=lambda item: (
                ranks[item[0]],
                _atom_token(graph.atoms[item[0]]),
                _bond_code(item[1]),
                item[0],
            )
        )
    parent: dict[int, int | None] = {root: None}
    discovery: dict[int, int] = {}
    tree_children: dict[int, list[tuple[int, SwitchBond]]] = defaultdict(list)
    ring_edges: dict[tuple[int, int], SwitchBond] = {}

    def walk(atom: int) -> None:
        discovery[atom] = len(discovery)
        for neighbor, bond in adjacency[atom]:
            if neighbor == parent[atom]:
                continue
            edge = tuple(sorted((atom, neighbor)))
            if neighbor not in discovery:
                parent[neighbor] = atom
                tree_children[atom].append((neighbor, bond))
                walk(neighbor)
            elif discovery[neighbor] < discovery[atom]:
                ring_edges[edge] = bond

    walk(root)
    ordered_rings = sorted(
        ring_edges,
        key=lambda edge: (
            min(discovery[edge[0]], discovery[edge[1]]),
            max(discovery[edge[0]], discovery[edge[1]]),
        ),
    )
    ring_labels = {edge: _ring_label(index + 1) for index, edge in enumerate(ordered_rings)}
    annotations: dict[
        int, list[tuple[str, int, SwitchBond, bool]]
    ] = defaultdict(list)
    for edge in ordered_rings:
        left, right = edge
        first, second = (
            (left, right)
            if discovery[left] < discovery[right]
            else (right, left)
        )
        annotations[first].append((ring_labels[edge], second, ring_edges[edge], True))
        annotations[second].append((ring_labels[edge], first, ring_edges[edge], False))

    def render(atom: int) -> str:
        children = [
            (_bond_symbol(bond, from_atom=atom) + render(child), child)
            for child, bond in tree_children[atom]
        ]
        children.sort(key=lambda item: (item[0], ranks[item[1]]))
        ring_items = sorted(annotations[atom], key=lambda item: item[0])
        ring_text = "".join(
            (
                _bond_symbol(bond, from_atom=atom)
                if first
                else ""
            )
            + label
            for label, _neighbor, bond, first in ring_items
        )
        explicit_order = []
        if parent[atom] is not None:
            explicit_order.append(parent[atom])
        explicit_order.extend(neighbor for _label, neighbor, _bond, _first in ring_items)
        explicit_order.extend(child for _text, child in children)
        chirality = _canonical_chirality(
            graph,
            atom,
            explicit_order,
            root=parent[atom] is None,
        )
        text = _atom_token(graph.atoms[atom], chirality=chirality) + ring_text
        if not children:
            return text
        for child_text, _ in children[:-1]:
            text += f"({child_text})"
        return text + children[-1][0]

    return render(root)


def _atom_token(atom: SwitchAtom, *, chirality: str | None = None) -> str:
    organic = {"B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I"}
    aromatic = {"B", "C", "N", "O", "P", "S", "Se", "As"}
    if (
        not atom.bracketed
        and atom.symbol in organic
        and atom.isotope is None
        and atom.formal_charge == 0
            and (atom.chirality if chirality is None else chirality) is None
        and atom.atom_class is None
    ):
        return atom.symbol.lower() if atom.aromatic and atom.symbol in aromatic else atom.symbol
    symbol = atom.symbol.lower() if atom.aromatic and atom.symbol in aromatic else atom.symbol
    isotope = "" if atom.isotope is None else str(atom.isotope)
    chirality = (atom.chirality if chirality is None else chirality) or ""
    hydrogen = (
        ""
        if not atom.hydrogen_count
        else "H" + (str(atom.hydrogen_count) if atom.hydrogen_count != 1 else "")
    )
    if atom.formal_charge == 0:
        charge = ""
    elif atom.formal_charge == 1:
        charge = "+"
    elif atom.formal_charge == -1:
        charge = "-"
    else:
        charge = f"{'+' if atom.formal_charge > 0 else '-'}{abs(atom.formal_charge)}"
    atom_class = "" if atom.atom_class is None else f":{atom.atom_class}"
    return f"[{isotope}{symbol}{chirality}{hydrogen}{charge}{atom_class}]"


def _bond_symbol(bond: SwitchBond, *, from_atom: int | None = None) -> str:
    if bond.direction:
        if from_atom is not None and from_atom == bond.right:
            return "/" if bond.direction == "\\" else "\\"
        return bond.direction
    if bond.dative:
        if from_atom is not None and from_atom == bond.right:
            return "<-" if bond.dative == "->" else "->"
        return bond.dative
    if bond.aromatic:
        return ":"
    if bond.order >= 3.5:
        return "$"
    if bond.order >= 2.5:
        return "#"
    if bond.order >= 1.75:
        return "="
    if bond.order == 0.0:
        return "~"
    return ""


def _bond_code(bond: SwitchBond) -> tuple[float, bool, str, str]:
    return (
        round(bond.order, 3),
        bond.aromatic,
        bond.direction or "",
        bond.dative or "",
    )


def _compress(values) -> list[int]:
    unique = {value: index for index, value in enumerate(sorted(set(values)))}
    return [unique[value] for value in values]


def _ring_label(index: int) -> str:
    if index < 10:
        return str(index)
    if index < 100:
        return f"%{index:02d}"
    return f"%({index})"


def _canonical_chirality(
    graph: SwitchMolecularGraph,
    atom_index: int,
    explicit_order: list[int],
    *,
    root: bool,
) -> str | None:
    atom = graph.atoms[atom_index]
    if atom.chirality not in {"@", "@@"}:
        return atom.chirality
    hydrogen_count = atom.hydrogen_count or 0
    implicit = [None] * hydrogen_count
    output_order = implicit + explicit_order if root else explicit_order + implicit
    original_order = _source_ligand_order(graph, atom_index)
    if len(output_order) != len(original_order) or set(output_order) != set(
        original_order
    ):
        return atom.chirality
    permutation = [output_order.index(ligand) for ligand in original_order]
    inversions = sum(
        permutation[left] > permutation[right]
        for left in range(len(permutation))
        for right in range(left + 1, len(permutation))
    )
    if inversions % 2 == 0:
        return atom.chirality
    return "@@" if atom.chirality == "@" else "@"


def _source_ligand_order(
    graph: SwitchMolecularGraph,
    atom_index: int,
) -> list[int | None]:
    atom = graph.atoms[atom_index]
    if atom.stereo_neighbors:
        return list(atom.stereo_neighbors)
    explicit = []
    has_incoming = False
    for bond in graph.bonds:
        if bond.left == atom_index:
            explicit.append(bond.right)
        elif bond.right == atom_index:
            explicit.append(bond.left)
            if bond.left < atom_index:
                has_incoming = True
    implicit = [None] * (atom.hydrogen_count or 0)
    return explicit + implicit if has_incoming else implicit + explicit


__all__ = [
    "SWITCH_CANONICAL_SCHEMA",
    "canonical_atom_ranks",
    "canonical_graph_key",
    "canonical_smiles",
]
