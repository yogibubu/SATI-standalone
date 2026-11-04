"""Deterministic subgraph matching for SWITCH molecular graphs."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

from .model import SwitchAtom, SwitchBond, SwitchMolecularGraph


@dataclass(frozen=True)
class CommonSubgraphMatch:
    source_atoms: tuple[int, ...]
    target_atoms: tuple[int, ...]

    @property
    def atom_count(self) -> int:
        return len(self.source_atoms)


def find_substructure_matches(
    target: SwitchMolecularGraph,
    query: SwitchMolecularGraph,
    *,
    use_chirality: bool = True,
    uniquify: bool = True,
    max_matches: int = 10000,
    allow_attachment_hydrogen_mismatch: bool = False,
) -> tuple[tuple[int, ...], ...]:
    """Return query-index ordered embeddings using a bounded VF2-style search."""

    if not query.atoms:
        return ((),)
    target_adjacency, target_bonds = _adjacency(target)
    query_adjacency, query_bonds = _adjacency(query)
    order = tuple(
        sorted(
            range(len(query.atoms)),
            key=lambda atom: (
                -len(query_adjacency[atom]),
                query.atoms[atom].symbol == "*",
                atom,
            ),
        )
    )
    candidates = {
        query_atom: tuple(
            target_atom
            for target_atom in range(len(target.atoms))
            if len(target_adjacency[target_atom]) >= len(query_adjacency[query_atom])
            and _atom_compatible(
                target.atoms[target_atom],
                query.atoms[query_atom],
                use_chirality=use_chirality,
            )
            and _query_hydrogen_compatible(
                target,
                target_atom,
                query.atoms[query_atom],
                query_degree=len(query_adjacency[query_atom]),
                allow_attachment_hydrogen_mismatch=allow_attachment_hydrogen_mismatch,
            )
        )
        for query_atom in order
    }
    mapping: dict[int, int] = {}
    used: set[int] = set()
    matches: list[tuple[int, ...]] = []
    identities: set[tuple[int, ...]] = set()

    def visit(depth: int) -> None:
        if len(matches) >= max_matches:
            return
        if depth == len(order):
            result = tuple(mapping[index] for index in range(len(query.atoms)))
            if use_chirality and not _stereo_mapping_compatible(
                target,
                query,
                mapping,
            ):
                return
            identity = tuple(sorted(result)) if uniquify else result
            if identity not in identities:
                identities.add(identity)
                matches.append(result)
            return
        query_atom = order[depth]
        for target_atom in candidates[query_atom]:
            if target_atom in used:
                continue
            if not _mapping_compatible(
                query_atom,
                target_atom,
                mapping,
                query_bonds,
                target_bonds,
            ):
                continue
            mapping[query_atom] = target_atom
            used.add(target_atom)
            visit(depth + 1)
            used.remove(target_atom)
            del mapping[query_atom]

    visit(0)
    return tuple(matches)


def maximum_common_connected_subgraphs(
    source: SwitchMolecularGraph,
    target: SwitchMolecularGraph,
    *,
    minimum_atoms: int = 1,
    timeout_seconds: float = 5.0,
    max_matches: int = 256,
    hydrogen_mode: str = "ignore",
    induced: bool = False,
) -> tuple[CommonSubgraphMatch, ...]:
    """Find largest connected element/bond-order preserving partial mappings."""

    if minimum_atoms < 1:
        raise ValueError("minimum common subgraph size must be positive")
    if hydrogen_mode not in {"ignore", "interface"}:
        raise ValueError("hydrogen_mode must be 'ignore' or 'interface'")
    deadline = monotonic() + float(timeout_seconds)
    source_adjacency, source_bonds = _adjacency(source)
    target_adjacency, target_bonds = _adjacency(target)
    best_size = 0
    best: list[CommonSubgraphMatch] = []
    identities: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()

    def record(mapping: dict[int, int]) -> None:
        nonlocal best_size
        size = len(mapping)
        if size < minimum_atoms or size < best_size:
            return
        if induced and not _induced_mapping_compatible(
            mapping,
            source_bonds,
            target_bonds,
        ):
            return
        source_atoms = tuple(sorted(mapping))
        target_atoms = tuple(mapping[index] for index in source_atoms)
        identity = (source_atoms, target_atoms)
        if size > best_size:
            best_size = size
            best.clear()
            identities.clear()
        if identity not in identities and len(best) < max_matches:
            identities.add(identity)
            best.append(CommonSubgraphMatch(source_atoms, target_atoms))

    def extend(
        mapping: dict[int, int],
        used_target: set[int],
        processed_source: set[int],
    ) -> None:
        if monotonic() > deadline:
            return
        record(mapping)
        frontier = sorted(
            {
                neighbor
                for source_atom in mapping
                for neighbor in source_adjacency[source_atom]
                if neighbor not in processed_source
            },
            key=lambda atom: (-len(source_adjacency[atom]), atom),
        )
        if not frontier:
            return
        source_atom = frontier[0]
        next_processed = processed_source | {source_atom}
        possible_targets = sorted(
            {
                neighbor
                for target_atom in mapping.values()
                for neighbor in target_adjacency[target_atom]
                if neighbor not in used_target
            }
        )
        for target_atom in possible_targets:
            if not _atom_compatible(
                target.atoms[target_atom],
                source.atoms[source_atom],
                use_chirality=False,
            ):
                continue
            if hydrogen_mode == "interface" and not _mcs_hydrogen_compatible(
                source,
                target,
                source_atom,
                target_atom,
            ):
                continue
            if not _mcs_extension_compatible(
                source_atom,
                target_atom,
                mapping,
                source_bonds,
                target_bonds,
            ):
                continue
            mapping[source_atom] = target_atom
            used_target.add(target_atom)
            extend(mapping, used_target, next_processed)
            used_target.remove(target_atom)
            del mapping[source_atom]
        extend(mapping, used_target, next_processed)

    for source_root in range(len(source.atoms)):
        for target_root in range(len(target.atoms)):
            if monotonic() > deadline:
                break
            if not _atom_compatible(
                target.atoms[target_root],
                source.atoms[source_root],
                use_chirality=False,
            ):
                continue
            if hydrogen_mode == "interface" and not _mcs_hydrogen_compatible(
                source,
                target,
                source_root,
                target_root,
            ):
                continue
            extend(
                {source_root: target_root},
                {target_root},
                {source_root},
            )
    return tuple(
        sorted(
            best,
            key=lambda match: (match.source_atoms, match.target_atoms),
        )
    )


def _mcs_extension_compatible(
    source_atom: int,
    target_atom: int,
    mapping: dict[int, int],
    source_bonds: dict[tuple[int, int], SwitchBond],
    target_bonds: dict[tuple[int, int], SwitchBond],
) -> bool:
    """Accept one or more common connecting edges.

    MCS is an edge-subgraph problem: an additional source edge between two
    mapped vertices may be omitted from the common graph. Requiring an induced
    atom subgraph would incorrectly reject, for example, opening a three-member
    fragment ring onto an acyclic target.
    """

    connected = False
    for mapped_source, mapped_target in mapping.items():
        source_bond = source_bonds.get(
            tuple(sorted((source_atom, mapped_source)))
        )
        target_bond = target_bonds.get(
            tuple(sorted((target_atom, mapped_target)))
        )
        if (
            source_bond is not None
            and target_bond is not None
            and _bond_compatible(target_bond, source_bond)
        ):
            connected = True
    return connected


def _mcs_hydrogen_compatible(
    source: SwitchMolecularGraph,
    target: SwitchMolecularGraph,
    source_index: int,
    target_index: int,
) -> bool:
    """Check H counts, allowing mismatch only at a valence interface."""
    source_adjacency, source_bonds = _adjacency(source)
    target_adjacency, target_bonds = _adjacency(target)

    def count(graph, adjacency, bonds, index):
        atom = graph.atoms[index]
        if atom.hydrogen_count is not None:
            return atom.hydrogen_count
        default = {"B": 3.0, "C": 4.0, "N": 3.0, "O": 2.0}.get(atom.symbol)
        if default is None:
            return None
        valence = sum(
            bond.order
            for bond in bonds.values()
            if bond.left == index or bond.right == index
        )
        return max(0, int(round(default - valence)))

    source_h = count(source, source_adjacency, source_bonds, source_index)
    target_h = count(target, target_adjacency, target_bonds, target_index)
    if source_h is None or target_h is None:
        return False
    if source_h == target_h:
        return True
    source_degree = len(source_adjacency[source_index])
    target_degree = len(target_adjacency[target_index])
    # At a valid interface, a target-side extra heavy-atom attachment replaces
    # one donor H (seed direction), or a donor-side extra heavy-atom attachment
    # replaces one target H (supplement direction).  Equal-degree mismatches
    # are never interfaces and are rejected.
    return (
        source_degree < target_degree and source_h > target_h
    ) or (
        source_degree > target_degree and source_h < target_h
    )


def _induced_mapping_compatible(
    mapping: dict[int, int],
    source_bonds: dict[tuple[int, int], SwitchBond],
    target_bonds: dict[tuple[int, int], SwitchBond],
) -> bool:
    """Require identical internal edge presence and compatible orders."""
    source_atoms = sorted(mapping)
    for left_index, left in enumerate(source_atoms):
        for right in source_atoms[left_index + 1:]:
            source_bond = source_bonds.get(tuple(sorted((left, right))))
            target_bond = target_bonds.get(
                tuple(sorted((mapping[left], mapping[right])))
            )
            if (source_bond is None) != (target_bond is None):
                return False
            if source_bond is not None and not _bond_compatible(target_bond, source_bond):
                return False
    return True


def _adjacency(
    graph: SwitchMolecularGraph,
) -> tuple[list[set[int]], dict[tuple[int, int], SwitchBond]]:
    adjacency = [set() for _ in graph.atoms]
    bonds = {}
    for bond in graph.bonds:
        adjacency[bond.left].add(bond.right)
        adjacency[bond.right].add(bond.left)
        bonds[bond.key] = bond
    return adjacency, bonds


def _atom_compatible(
    target: SwitchAtom,
    query: SwitchAtom,
    *,
    use_chirality: bool,
) -> bool:
    if query.symbol == "*":
        return True
    if target.symbol != query.symbol:
        return False
    if query.isotope is not None and target.isotope != query.isotope:
        return False
    if query.formal_charge != target.formal_charge:
        return False
    if query.aromatic and not target.aromatic:
        return False
    if use_chirality and query.chirality and not target.chirality:
        return False
    return True


def _query_hydrogen_compatible(
    target_graph: SwitchMolecularGraph,
    target_index: int,
    query: SwitchAtom,
    *,
    query_degree: int | None = None,
    allow_attachment_hydrogen_mismatch: bool = False,
) -> bool:
    if query.hydrogen_count is None:
        return True
    target = target_graph.atoms[target_index]
    target_hydrogen_count = target.hydrogen_count
    if target_hydrogen_count is None:
        target_adjacency, target_bonds = _adjacency(target_graph)
        default_valence = {"B": 3.0, "C": 4.0, "N": 3.0, "O": 2.0}.get(
            target.symbol
        )
        if default_valence is None:
            return False
        valence = sum(
            bond.order
            for bond in target_bonds.values()
            if bond.left == target_index or bond.right == target_index
        )
        target_hydrogen_count = max(0, int(round(default_valence - valence)))
    if target_hydrogen_count == query.hydrogen_count:
        return True
    if not allow_attachment_hydrogen_mismatch or query_degree is None:
        return False
    # A seed may end at a target atom that has one additional heavy-atom
    # attachment.  The donor then has one more H at that boundary.  The
    # opposite direction (donor missing H while carrying an extra heavy atom)
    # is deliberately not accepted here: that is a supplementary-fragment
    # interface and must be handled by the MCS stage.
    target_degree = len(_adjacency(target_graph)[0][target_index])
    return (
        query.hydrogen_count is not None
        and target_hydrogen_count is not None
        and query.hydrogen_count > target_hydrogen_count
        and target_degree > query_degree
    )


def _mapping_compatible(
    query_atom: int,
    target_atom: int,
    mapping: dict[int, int],
    query_bonds: dict[tuple[int, int], SwitchBond],
    target_bonds: dict[tuple[int, int], SwitchBond],
) -> bool:
    for mapped_query, mapped_target in mapping.items():
        query_bond = query_bonds.get(tuple(sorted((query_atom, mapped_query))))
        if query_bond is None:
            continue
        target_bond = target_bonds.get(tuple(sorted((target_atom, mapped_target))))
        if target_bond is None or not _bond_compatible(target_bond, query_bond):
            return False
    return True


def _bond_compatible(target: SwitchBond, query: SwitchBond) -> bool:
    if query.aromatic:
        return target.aromatic
    if query.order == 0.0:
        return True
    return abs(target.order - query.order) < 0.2


def _stereo_mapping_compatible(
    target: SwitchMolecularGraph,
    query: SwitchMolecularGraph,
    mapping: dict[int, int],
) -> bool:
    for query_atom in query.atoms:
        if query_atom.chirality not in {"@", "@@"}:
            continue
        target_index = mapping[query_atom.index]
        target_atom = target.atoms[target_index]
        if target_atom.chirality not in {"@", "@@"}:
            return False
        query_order = _smiles_ligand_order(query, query_atom.index)
        target_order = _smiles_ligand_order(target, target_index)
        mapped_order = [
            None if ligand is None else mapping[ligand]
            for ligand in query_order
        ]
        if len(mapped_order) != len(target_order) or set(mapped_order) != set(
            target_order
        ):
            return False
        permutation = [target_order.index(ligand) for ligand in mapped_order]
        parity = _permutation_parity(permutation)
        query_parity = 0 if query_atom.chirality == "@" else 1
        target_parity = 0 if target_atom.chirality == "@" else 1
        if query_parity ^ parity != target_parity:
            return False
    return True


def _smiles_ligand_order(
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
    hydrogens = atom.hydrogen_count or 0
    implicit = [None] * hydrogens
    return explicit + implicit if has_incoming else implicit + explicit


def _permutation_parity(permutation: list[int]) -> int:
    inversions = sum(
        permutation[left] > permutation[right]
        for left in range(len(permutation))
        for right in range(left + 1, len(permutation))
    )
    return inversions % 2


__all__ = [
    "CommonSubgraphMatch",
    "find_substructure_matches",
    "maximum_common_connected_subgraphs",
]
