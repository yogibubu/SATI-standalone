from pathlib import Path

import numpy as np


def _smith_symmetry_api():
    """Load the optional SMITH reporting extension only when it is requested."""

    from matrix_smith.coordinate_kernel.primitives import build_primitives
    from matrix_smith.coordinate_kernel.symmetry_classifier import group_label as _group_label
    from matrix_smith.coordinate_kernel.symmetry_detector import (
        is_linear,
        orient_coords,
        symmetry_elements_from_geometry,
    )
    from matrix_smith.coordinate_kernel.symmetry_global import primitive_permutation

    return (
        build_primitives,
        _group_label,
        is_linear,
        orient_coords,
        symmetry_elements_from_geometry,
        primitive_permutation,
    )


# ============================================================
# Complete periodic table
# ============================================================

_PERIODIC_TABLE = {
    1: "H",
    2: "He",
    3: "Li",
    4: "Be",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    10: "Ne",
    11: "Na",
    12: "Mg",
    13: "Al",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    18: "Ar",
    19: "K",
    20: "Ca",
    21: "Sc",
    22: "Ti",
    23: "V",
    24: "Cr",
    25: "Mn",
    26: "Fe",
    27: "Co",
    28: "Ni",
    29: "Cu",
    30: "Zn",
    31: "Ga",
    32: "Ge",
    33: "As",
    34: "Se",
    35: "Br",
    36: "Kr",
    37: "Rb",
    38: "Sr",
    39: "Y",
    40: "Zr",
    41: "Nb",
    42: "Mo",
    43: "Tc",
    44: "Ru",
    45: "Rh",
    46: "Pd",
    47: "Ag",
    48: "Cd",
    49: "In",
    50: "Sn",
    51: "Sb",
    52: "Te",
    53: "I",
    54: "Xe",
    55: "Cs",
    56: "Ba",
    57: "La",
    58: "Ce",
    59: "Pr",
    60: "Nd",
    61: "Pm",
    62: "Sm",
    63: "Eu",
    64: "Gd",
    65: "Tb",
    66: "Dy",
    67: "Ho",
    68: "Er",
    69: "Tm",
    70: "Yb",
    71: "Lu",
    72: "Hf",
    73: "Ta",
    74: "W",
    75: "Re",
    76: "Os",
    77: "Ir",
    78: "Pt",
    79: "Au",
    80: "Hg",
    81: "Tl",
    82: "Pb",
    83: "Bi",
    84: "Po",
    85: "At",
    86: "Rn",
    87: "Fr",
    88: "Ra",
    89: "Ac",
    90: "Th",
    91: "Pa",
    92: "U",
    93: "Np",
    94: "Pu",
    95: "Am",
    96: "Cm",
    97: "Bk",
    98: "Cf",
    99: "Es",
    100: "Fm",
    101: "Md",
    102: "No",
    103: "Lr",
    104: "Rf",
    105: "Db",
    106: "Sg",
    107: "Bh",
    108: "Hs",
    109: "Mt",
    110: "Ds",
    111: "Rg",
    112: "Cn",
    113: "Nh",
    114: "Fl",
    115: "Mc",
    116: "Lv",
    117: "Ts",
    118: "Og",
}


def _element_symbol(Z):
    """Return element symbol from atomic number."""
    return _PERIODIC_TABLE.get(int(Z), f"Z{int(Z)}")


def _uf_find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def _uf_union(parent, a, b):
    ra = _uf_find(parent, a)
    rb = _uf_find(parent, b)
    if ra != rb:
        parent[rb] = ra


def _primitive_label(p):
    if p.kind == "bond":
        i, j = p.atoms
        return f"({i + 1}-{j + 1})"
    if p.kind == "angle":
        i, j, k = p.atoms
        return f"({i + 1}-{j + 1}-{k + 1})"
    if p.kind == "dihedral":
        i, j, k, l = p.atoms
        return f"({i + 1}-{j + 1}-{k + 1}-{l + 1})"
    return "(" + ",".join(str(a + 1) for a in p.atoms) + ")"


def _symmetry_summary(
    cg,
    dg,
    *,
    symm_tol: float = 1.0e-3,
    symmetrize_coords: bool = False,
):
    """
    Compute point group and symmetry-equivalent primitive classes.
    Uses the same geometry already available in topology.
    """
    (
        build_primitives,
        _group_label,
        is_linear,
        orient_coords,
        symmetry_elements_from_geometry,
        primitive_permutation,
    ) = _smith_symmetry_api()
    Z = np.asarray(cg.Z, dtype=int)
    coords = np.asarray(cg.coords, dtype=float)
    symbols = [_element_symbol(z) for z in Z]

    def _mass_weights(Zvals):
        try:
            from matrix_chem.average_atomic_masses import atomic_mass

            return np.array([atomic_mass(int(z)) for z in Zvals], dtype=float)
        except Exception:
            return np.array(Zvals, dtype=float)

    weights = _mass_weights(Z)

    def _orient_with_frame(x, w):
        x = np.array(x, dtype=float)
        w = np.array(w, dtype=float)
        wsum = float(np.sum(w))
        com = np.sum(x * w[:, None], axis=0) / max(wsum, 1.0e-12)
        x0 = x - com[None, :]
        I = np.zeros((3, 3))
        for i, r in enumerate(x0):
            I += w[i] * ((np.dot(r, r) * np.eye(3)) - np.outer(r, r))
        evals, evecs = np.linalg.eigh(I)
        order = np.argsort(evals)
        V = evecs[:, order]
        if np.linalg.det(V) < 0:
            V[:, -1] *= -1.0
        return x0 @ V, com, V

    coords_oriented, com, V = _orient_with_frame(coords, weights)
    elements, atom_classes, permutations = symmetry_elements_from_geometry(
        symbols,
        coords_oriented,
        tol=float(symm_tol),
        max_n=8,
        auto_max_n=True,
    )
    point_group = _group_label(elements, linear=is_linear(coords_oriented, tol=float(symm_tol)))

    prims = build_primitives(dg, coords)
    nprim = len(prims)
    parent = list(range(nprim))
    for mapping in permutations:
        perm_idx, _ = primitive_permutation(prims, mapping)
        for i, j in enumerate(perm_idx):
            _uf_union(parent, i, j)

    classes = {}
    for i in range(nprim):
        root = _uf_find(parent, i)
        classes.setdefault(root, []).append(i)

    by_kind = {}
    for idxs in classes.values():
        idxs_sorted = sorted(idxs)
        kind = prims[idxs_sorted[0]].kind
        by_kind.setdefault(kind, []).append(idxs_sorted)

    out = {
        "point_group": point_group,
        "atom_classes": [sorted(cls) for cls in atom_classes],
        "primitive_classes_by_kind": by_kind,
        "primitives": prims,
    }

    if symmetrize_coords and permutations and elements:
        coords_symm = np.zeros_like(coords_oriented)
        count = 0
        for _label, R in elements:
            coords_t = coords_oriented @ R.T
            coords_perm = np.zeros_like(coords_t)
            for i, j in enumerate(permutations[count]):
                coords_perm[i] = coords_t[j]
            coords_symm += coords_perm
            count += 1
        if count > 0:
            coords_symm /= float(count)
            coords_symm_world = coords_symm @ V.T + com[None, :]
            # Snap tiny numerical noise for strict symmetry.
            coords_symm_world = np.round(coords_symm_world, decimals=8)
            out["symmetrized_coords"] = coords_symm_world
    return out


# ============================================================
# Topology reporting
# ============================================================


def print_topology_report(
    cg,
    dg,
    synthons,
    arom=None,
    filename="topology.report",
    ringset=None,
    *,
    symm_tol: float = 1.0e-3,
    symmetrize_coords: bool = False,
):
    """
    Write a diagnostic report of the molecular topology to a file.

    Atom indices in the report are 1-based (ORACLE convention).
    """

    Z = cg.Z
    symbols = [_element_symbol(z) for z in Z]
    coords = cg.coords

    # --- write output into cwd/working ---
    outdir = Path.cwd() / "working"
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / filename

    with open(outfile, "w") as fh:
        charge_source = getattr(synthons, "_charge_source", "ORACLE electronegativity estimate")
        bo_source = getattr(synthons, "_bond_order_source", "ORACLE Pauling estimate")

        fh.write("\nTOPOLOGY INPUT SOURCES\n")
        fh.write("----------------------\n")
        fh.write(f"Atomic charges source: {charge_source}\n")
        fh.write(f"Bond-order source: {bo_source}\n")

        # ========================================================
        # Bonded pairs and bond orders
        # ========================================================

        fh.write("\nBOND ORDERS (bonded pairs, heavy atoms only)\n")
        fh.write("------------------------------------------\n")

        for i, j in dg.bonds:
            if Z[i] <= 1 or Z[j] <= 1:
                continue

            rij = np.linalg.norm(coords[i] - coords[j])
            zi = _element_symbol(Z[i])
            zj = _element_symbol(Z[j])

            bo = synthons.bond_order(i, j)
            sigma, pi, pi_pi = synthons.bond_order_indices(i, j)

            fh.write(
                f"({i + 1:2d},{j + 1:2d})  {zi:>2s}–{zj:<2s}   r = {rij:6.3f} Å   "
                f"BO = {bo:6.3f}  "
                f"(sigma,pi,pi-pi)=({sigma:.3f},{pi:.3f},{pi_pi:.3f})\n"
            )

        # ========================================================
        # Atomic synthons
        # ========================================================

        fh.write("\nATOMIC SYNTHONS (heavy atoms)\n")
        fh.write("----------------------------\n")

        for i in range(len(Z)):
            if Z[i] <= 1:
                continue

            zi = _element_symbol(Z[i])

            cna = synthons.cna(i)
            edom = synthons._electron_domains(i)
            cn_strain = int(round(edom))

            fh.write(f"\nAtom {i + 1:2d}  {zi}  (Z={Z[i]})\n")
            fh.write(f"  Zeff              = {synthons.Zeff(i):8.3f}\n")
            fh.write(f"  CNA               = {cna:8.3f}\n")
            fh.write(f"  Electron domains  = {edom:8.3f}\n")
            fh.write(f"  CN (strain)       = {cn_strain:8d}\n")
            fh.write(f"  Charge            = {synthons.charge(i):8.3f}\n")
            fh.write(f"  Covalency         = {synthons.covalency(i):8.3f}\n")
            fh.write(f"  BO product (D)    = {synthons.delocalization(i):8.3f}\n")
            fh.write(f"  Radial dispersion = {synthons.radial_dispersion(i):8.3f}\n")
            fh.write(f"  Strain            = {synthons.strain(i):8.3f}\n")

        # ========================================================
        # Rings (optional)
        # ========================================================
        if ringset is not None:
            fh.write("\nRINGS\n")
            fh.write("-----\n")
            rings = list(getattr(ringset, "rings", []))
            if rings:
                for ring in rings:
                    atoms = [a + 1 for a in getattr(ring, "atoms", [])]
                    fh.write(
                        f"Ring {int(getattr(ring, 'index', 0)) + 1}: "
                        f"size={len(atoms)} atoms={atoms}\n"
                    )
            else:
                fh.write("No rings detected\n")

        # ========================================================
        # Aromaticity (optional)
        # ========================================================

        if arom is not None:
            fh.write("\nAROMATICITY\n")
            fh.write("-----------\n")

            if arom.aromatic_atoms:
                fh.write(f"Aromatic atoms: {sorted([i + 1 for i in arom.aromatic_atoms])}\n")
            else:
                fh.write("Aromatic atoms: none\n")

            if arom.aromatic_bonds:
                fh.write(
                    "Aromatic bonds: "
                    f"{sorted([(i + 1, j + 1) for (i, j) in arom.aromatic_bonds])}\n"
                )
            else:
                fh.write("Aromatic bonds: none\n")

        # ========================================================
        # Representation
        # ========================================================
        try:
            from matrix_core import parse_key_value_section, read_sectioned_lines, section_content

            values = parse_key_value_section(
                section_content(read_sectioned_lines(Path(filename).with_name("xyzin")), "BASIC")
            )
            rep = values.get("REPRESENTATION", "Ir")
        except Exception:
            rep = "Ir"
        fh.write(f"Representation: {rep}\n")

        # ========================================================
        # Global symmetry and equivalent parameters
        # ========================================================
        try:
            sym = _symmetry_summary(
                cg,
                dg,
                symm_tol=symm_tol,
                symmetrize_coords=symmetrize_coords,
            )
            fh.write("\nGLOBAL SYMMETRY\n")
            fh.write("---------------\n")
            fh.write(f"Point group: {sym['point_group']}\n")

            atom_classes = [[i + 1 for i in cls] for cls in sym["atom_classes"] if len(cls) > 1]
            if atom_classes:
                fh.write("Equivalent atom classes (1-based):\n")
                for icls, cls in enumerate(atom_classes, start=1):
                    fh.write(f"  class {icls:2d}: {cls}\n")
            else:
                fh.write("Equivalent atom classes: none (C1-like partition)\n")

            if symmetrize_coords and "symmetrized_coords" in sym:
                try:
                    report_path = Path(filename)
                    out_xyz = report_path.with_name("symmetrized.xyz")
                    coords_symm = sym["symmetrized_coords"]
                    fh.write("\nSymmetrized XYZ: ")
                    fh.write(str(out_xyz) + "\n")
                    lines = [str(len(Z)), "Symmetrized coordinates"]
                    for symb, (x, y, z) in zip(symbols, coords_symm):
                        lines.append(f"{symb} {x: .6f} {y: .6f} {z: .6f}")
                    out_xyz.write_text("\n".join(lines) + "\n", encoding="utf-8")
                except Exception:
                    pass

            fh.write("\nEQUIVALENT INTERNAL-PARAMETER CLASSES\n")
            fh.write("------------------------------------\n")
            prims = sym["primitives"]
            by_kind = sym["primitive_classes_by_kind"]
            for kind in ("bond", "angle", "dihedral", "out_of_plane", "linear_bend"):
                groups = [g for g in by_kind.get(kind, []) if len(g) > 1]
                fh.write(f"{kind}: {len(groups)} equivalent classes\n")
                for ig, g in enumerate(groups, start=1):
                    labels = [_primitive_label(prims[idx]) for idx in g]
                    fh.write(f"  class {ig:2d}: {labels}\n")
        except Exception as exc:
            fh.write("\nGLOBAL SYMMETRY\n")
            fh.write("---------------\n")
            fh.write(f"Symmetry analysis unavailable: {exc}\n")
