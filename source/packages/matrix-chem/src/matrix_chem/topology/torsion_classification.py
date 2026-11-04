"""Common endocyclic/exocyclic classification for primitive torsions."""

from __future__ import annotations

from typing import Iterable


def classify_torsion_ring_type(
    atoms: Iterable[int],
    bond_to_rings,
) -> str:
    """Return ``endocyclic`` or ``exocyclic`` from the central edge.

    A torsion is endocyclic exactly when its central bond (the second and
    third atoms) belongs to at least one perceived ring.  This definition is
    graph-based and is shared by ORACLE, SMITH, ARCHITECT and ZAFF; it does not
    depend on atom names, ring size, or a particular coordinate convention.
    """
    quartet = tuple(int(value) for value in atoms)
    if len(quartet) != 4 or len(set(quartet)) != 4:
        raise ValueError("torsion classification requires four distinct atoms")
    central = tuple(sorted((quartet[1], quartet[2])))
    return "endocyclic" if tuple(bond_to_rings.get(central, ())) else "exocyclic"


__all__ = ["classify_torsion_ring_type"]
