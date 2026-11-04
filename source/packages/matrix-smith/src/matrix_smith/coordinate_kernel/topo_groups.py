from __future__ import annotations

from typing import Dict, List, Tuple


def primitive_signature(p, atom_class, Z, ringset=None, angle_bin_val=None):
    kind = p.kind
    atoms = p.atoms

    def _ring_tag(atoms_tuple):
        if ringset is None:
            return 0
        aset = set(atoms_tuple)
        for r in ringset:
            if aset.issubset(set(r.atoms)):
                return 1
        return 0

    ring_tag = _ring_tag(atoms)

    if kind == "bond":
        i, j = atoms
        h_i = Z[i] == 1
        h_j = Z[j] == 1
        htag = (1 if h_i else 0, 1 if h_j else 0)
        # X–H bonds can only be combined if geminal (same heavy atom).
        if h_i ^ h_j:
            heavy = j if h_i else i
            h = i if h_i else j
            return (kind, atom_class[heavy], atom_class[h], "XH", ring_tag, heavy)
        return (kind, atom_class[i], atom_class[j], htag, ring_tag)
    if kind == "angle":
        i, j, k = atoms
        htag = (1 if Z[i] == 1 else 0, 1 if Z[j] == 1 else 0, 1 if Z[k] == 1 else 0)
        pair = tuple(sorted((atom_class[i], atom_class[k])))
        return (kind, atom_class[j], pair, htag, ring_tag, angle_bin_val)
    if kind == "dihedral":
        i, j, k, l = atoms
        htag = (
            1 if Z[i] == 1 else 0,
            1 if Z[j] == 1 else 0,
            1 if Z[k] == 1 else 0,
            1 if Z[l] == 1 else 0,
        )
        return (kind, atom_class[j], atom_class[k], atom_class[i], atom_class[l], htag, ring_tag)
    if kind == "out_of_plane":
        center, plane1, plane2, out = atoms
        htag = (
            1 if Z[center] == 1 else 0,
            1 if Z[plane1] == 1 else 0,
            1 if Z[plane2] == 1 else 0,
            1 if Z[out] == 1 else 0,
        )
        return (
            kind,
            atom_class[center],
            tuple(sorted((atom_class[plane1], atom_class[plane2], atom_class[out]))),
            htag,
            ring_tag,
        )
    if kind == "linear_bend":
        i, j, k = atoms
        htag = (1 if Z[i] == 1 else 0, 1 if Z[j] == 1 else 0, 1 if Z[k] == 1 else 0)
        pair = tuple(sorted((atom_class[i], atom_class[k])))
        return (kind, atom_class[j], pair, htag, ring_tag, p.mode)
    if kind.startswith("frag_"):
        return (kind, tuple(sorted(atom_class[a] for a in atoms)))
    return (kind, tuple(sorted(atom_class[a] for a in atoms)), ring_tag)


def group_primitives(prims, atom_class, Z, ringset=None, angle_bins=None, idx_map=None):
    groups: Dict[Tuple, List[int]] = {}
    for i, p in enumerate(prims):
        angle_bin_val = None
        if angle_bins is not None and p.kind == "angle":
            orig_i = i if idx_map is None else idx_map[i]
            angle_bin_val = angle_bins.get(orig_i)
        sig = primitive_signature(p, atom_class, Z, ringset=ringset, angle_bin_val=angle_bin_val)
        groups.setdefault(sig, []).append(i)
    return groups
