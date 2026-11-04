# ring_primitives.py
# ============================================================
# Primitive internal coordinates associated with rings
#
# This module provides functions that, given a Ring object,
# generate cyclic primitive internal coordinates:
#   - cyclic valence angles
#   - cyclic dihedrals
#
# It does NOT:
#   - define Ring
#   - perform pruning
#   - decide aromaticity
#   - mix coordinate spaces
# ============================================================


def ring_valence_angles(ring):
    """
    Generate cyclic valence-angle primitives for a ring.

    Parameters
    ----------
    ring : Ring
        Ring object with ordered atoms.

    Returns
    -------
    list of tuple
        Each tuple is (i, j, k) representing A(i,j,k),
        where j is the central atom.
    """
    angles = []
    atoms = ring.atoms
    n = len(atoms)

    for i in range(n):
        i_prev = atoms[(i - 1) % n]
        j = atoms[i]
        i_next = atoms[(i + 1) % n]
        angles.append((i_prev, j, i_next))

    return angles


def ring_dihedrals(ring):
    """
    Generate cyclic dihedral primitives for a ring.

    Parameters
    ----------
    ring : Ring
        Ring object with ordered atoms.

    Returns
    -------
    list of tuple
        Each tuple is (i, j, k, l) representing D(i,j,k,l).
    """
    dihedrals = []
    atoms = ring.atoms
    n = len(atoms)

    for i in range(n):
        i1 = atoms[i % n]
        i2 = atoms[(i + 1) % n]
        i3 = atoms[(i + 2) % n]
        i4 = atoms[(i + 3) % n]
        dihedrals.append((i1, i2, i3, i4))

    return dihedrals


def ring_bonds(ring):
    """
    Return ring bonds as primitive stretchings.

    Parameters
    ----------
    ring : Ring

    Returns
    -------
    list of tuple
        Each tuple is (i,j) representing R(i,j).
    """
    return list(ring.bonds)


def ring_atom_pairs(ring):
    """
    Return atom pairs along the ring (useful for diagnostics
    or extended ring coordinates).

    Parameters
    ----------
    ring : Ring

    Returns
    -------
    list of tuple
        (i,j) pairs along the cycle.
    """
    atoms = ring.atoms
    n = len(atoms)
    return [(atoms[i], atoms[(i + 1) % n]) for i in range(n)]
