"""ORACLE coordinate-topology decomposition for haptic complexes."""
from __future__ import annotations
from dataclasses import dataclass
from .topology.metals import is_metal_atomic_number
from .topology.elements import atomic_number

@dataclass(frozen=True)
class _CoordinateFragmentRecord:
    identifier: str
    atoms: tuple[int, ...]

def _adjacency(bonds, *, natoms):
    result = {i: set() for i in range(1, natoms + 1)}
    for a, b in bonds:
        result[a].add(b)
        result[b].add(a)
    return result

def _haptic_coordinate_bonds_and_fragments(
    bonds: tuple[tuple[int, int], ...],
    *,
    interaction_centers: object | None,
    atom_symbols: tuple[str, ...],
    natoms: int,
) -> tuple[tuple[tuple[int, int], ...], tuple[_CoordinateFragmentRecord, ...]]:
    """Replace metal--donor spokes by the existing rigid-fragment coordinates.

    ORACLE's chemical topology is deliberately left untouched.  This helper
    only builds SMITH's coordinate graph: an eta interaction is represented by
    its metal--ring/haptic center together with the already established
    fragment translations and exponential-map orientations, rather than by
    one ordinary bond, bend, and torsion family per donor atom.
    """

    if interaction_centers is None or len(atom_symbols) != natoms:
        return bonds, ()
    centers = {
        str(getattr(center, "identifier")): center
        for center in getattr(interaction_centers, "centers", ())
    }
    bond_set = {tuple(sorted((int(left), int(right)))) for left, right in bonds}
    suppressed: set[tuple[int, int]] = set()
    for interaction in getattr(interaction_centers, "interactions", ()):
        atom = int(getattr(interaction, "atom", 0))
        if not 1 <= atom <= natoms:
            continue
        center = centers.get(str(getattr(interaction, "center_id", "")))
        if center is None or str(getattr(center, "kind", "")).upper() not in {
            "RING_CENTER",
            "HAPTIC_CENTER",
        }:
            continue
        if not is_metal_atomic_number(atomic_number(atom_symbols[atom - 1])):
            continue
        for donor in getattr(center, "atoms", ()):
            edge = tuple(sorted((atom, int(donor))))
            if edge in bond_set:
                suppressed.add(edge)
    if not suppressed:
        return bonds, ()

    coordinate_bonds = tuple(
        bond for bond in bonds if tuple(sorted((int(bond[0]), int(bond[1])))) not in suppressed
    )
    adjacency = _adjacency(coordinate_bonds, natoms=natoms)
    unseen = set(range(1, natoms + 1))
    components: list[tuple[int, ...]] = []
    while unseen:
        root = min(unseen)
        pending = [root]
        component: set[int] = set()
        while pending:
            atom = pending.pop()
            if atom in component:
                continue
            component.add(atom)
            pending.extend(sorted(adjacency[atom] - component, reverse=True))
        unseen.difference_update(component)
        components.append(tuple(sorted(component)))
    records = tuple(
        _CoordinateFragmentRecord(
            identifier=f"HAPTIC_COMPONENT_{index:03d}",
            atoms=atoms,
        )
        for index, atoms in enumerate(components, start=1)
    )
    return coordinate_bonds, records
