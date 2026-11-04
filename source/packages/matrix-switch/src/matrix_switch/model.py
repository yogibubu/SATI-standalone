"""Stable, serializable molecular-graph contract used by SWITCH."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


SWITCH_GRAPH_SCHEMA = "matrix.switch.graph.v1"


@dataclass(frozen=True)
class SwitchAtom:
    index: int
    symbol: str
    isotope: int | None = None
    formal_charge: int = 0
    hydrogen_count: int | None = None
    aromatic: bool = False
    chirality: str | None = None
    atom_class: int | None = None
    bracketed: bool = False
    source_span: tuple[int, int] = (0, 0)
    stereo_neighbors: tuple[int | None, ...] = ()


@dataclass(frozen=True)
class SwitchBond:
    left: int
    right: int
    order: float = 1.0
    aromatic: bool = False
    direction: str | None = None
    dative: str | None = None
    ring_label: str | None = None

    @property
    def key(self) -> tuple[int, int]:
        return tuple(sorted((self.left, self.right)))


@dataclass(frozen=True)
class SwitchMolecularGraph:
    atoms: tuple[SwitchAtom, ...]
    bonds: tuple[SwitchBond, ...]
    components: tuple[tuple[int, ...], ...]
    source_smiles: str
    total_formal_charge: int
    schema: str = SWITCH_GRAPH_SCHEMA

    @property
    def natoms(self) -> int:
        return len(self.atoms)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def neighbors(self, atom: int) -> tuple[int, ...]:
        result = []
        for bond in self.bonds:
            if bond.left == atom:
                result.append(bond.right)
            elif bond.right == atom:
                result.append(bond.left)
        return tuple(sorted(result))


def graph_from_topology(
    atoms: Sequence[str],
    bonds: Sequence[tuple[int, int]],
    *,
    bond_orders: Mapping[tuple[int, int], float] | None = None,
    hydrogen_counts: Sequence[int | None] | None = None,
    formal_charges: Sequence[int] | None = None,
    aromatic_atoms: Sequence[int] = (),
    source_smiles: str = "",
) -> SwitchMolecularGraph:
    """Construct the SWITCH contract from an already perceived topology."""

    atom_count = len(atoms)
    if formal_charges is not None and len(formal_charges) != atom_count:
        raise ValueError("formal charges must match the atom count")
    if hydrogen_counts is not None and len(hydrogen_counts) != atom_count:
        raise ValueError("hydrogen counts must match the atom count")
    aromatic = {int(index) for index in aromatic_atoms}
    switch_atoms = tuple(
        SwitchAtom(
            index=index,
            symbol=str(symbol),
            formal_charge=0 if formal_charges is None else int(formal_charges[index]),
            hydrogen_count=(
                None if hydrogen_counts is None else int(hydrogen_counts[index])
            ),
            aromatic=index in aromatic,
            bracketed=True,
        )
        for index, symbol in enumerate(atoms)
    )
    orders = {
        tuple(sorted((int(left), int(right)))): float(order)
        for (left, right), order in (bond_orders or {}).items()
    }
    switch_bonds = []
    adjacency = [[] for _ in range(atom_count)]
    seen: set[tuple[int, int]] = set()
    for raw_left, raw_right in bonds:
        left, right = int(raw_left), int(raw_right)
        key = tuple(sorted((left, right)))
        if left < 0 or right < 0 or left >= atom_count or right >= atom_count or left == right:
            raise ValueError(f"invalid topology bond: {(left, right)}")
        if key in seen:
            continue
        seen.add(key)
        order = orders.get(key, 1.0)
        switch_bonds.append(
            SwitchBond(
                left=left,
                right=right,
                order=order,
                aromatic=order == 1.5 or (left in aromatic and right in aromatic),
            )
        )
        adjacency[left].append(right)
        adjacency[right].append(left)
    components = []
    pending = set(range(atom_count))
    while pending:
        root = min(pending)
        component = []
        queue = [root]
        pending.remove(root)
        for atom in queue:
            component.append(atom)
            for neighbor in sorted(adjacency[atom]):
                if neighbor in pending:
                    pending.remove(neighbor)
                    queue.append(neighbor)
        components.append(tuple(component))
    return SwitchMolecularGraph(
        atoms=switch_atoms,
        bonds=tuple(switch_bonds),
        components=tuple(components),
        source_smiles=source_smiles,
        total_formal_charge=sum(atom.formal_charge for atom in switch_atoms),
    )


__all__ = [
    "SWITCH_GRAPH_SCHEMA",
    "SwitchAtom",
    "SwitchBond",
    "SwitchMolecularGraph",
    "graph_from_topology",
]
