from __future__ import annotations

import re
import numpy as np


def highest_cn_axis(labels):
    nmax = 1
    axis = None
    for lab in labels:
        if not lab.startswith("C"):
            continue
        m = re.match(r"C(\d+)", lab)
        if not m:
            continue
        n = int(m.group(1))
        if n > nmax:
            nmax = n
            if "z" in lab:
                axis = "z"
            elif "x" in lab:
                axis = "x"
            elif "y" in lab:
                axis = "y"
    return nmax, axis


def count_cn_axis(labels, axis="z"):
    ns = set()
    for lab in labels:
        if not lab.startswith("C"):
            continue
        if f"{axis}" not in lab:
            continue
        m = re.match(r"C(\d+)", lab)
        if not m:
            continue
        ns.add(int(m.group(1)))
    return len(ns), ns


def highest_sn(labels):
    nmax = 1
    for lab in labels:
        if not lab.startswith("S"):
            continue
        m = re.match(r"S(\d+)", lab)
        if not m:
            continue
        n = int(m.group(1))
        if n > nmax:
            nmax = n
    return nmax


def pick_generators(elements):
    labels = [e[0] for e in elements]
    gens = []
    nmax, axis = highest_cn_axis(labels)
    if nmax > 1 and axis is not None:
        gens.append(f"C{nmax}{axis}^1")
    for cand in ("sigma_xy", "sigma_xz", "sigma_yz"):
        if cand in labels:
            gens.append(cand)
            break
    if "i" in labels:
        gens.append("i")
    return gens


def spherical_top_guess(symbols, coords, center_idx=None, ignore_isotopes=False):
    if len(symbols) < 5:
        return None
    if len(set(symbols)) == 1 and center_idx is None:
        return None
    center = 0 if center_idx is None else int(center_idx)
    ligands = [i for i in range(len(symbols)) if i != center]
    if not ignore_isotopes:
        lig_syms = {symbols[i] for i in ligands}
        if len(lig_syms) > 1:
            return None
    r = np.linalg.norm(coords[ligands] - coords[center], axis=1)
    if np.std(r) < 1e-3:
        if len(ligands) == 4:
            return "Td"
        if len(ligands) == 6:
            return "Oh"
        if len(ligands) == 8:
            return "Oh"
        if len(ligands) == 12:
            return "Ih"
    return None


def group_label(elements, linear=False):
    labels = [e[0] for e in elements]
    nmax, axis = highest_cn_axis(labels)
    ncount_z, _ = count_cn_axis(labels, axis="z")
    snmax = highest_sn(labels)

    has_i = any(lab == "i" for lab in labels)
    has_sigma = any(lab.startswith("sigma") for lab in labels)
    has_s = any(lab.startswith("S") for lab in labels)

    t_ops = [lab for lab in labels if lab.endswith("_t")]
    o_ops = [lab for lab in labels if lab.endswith("_o")]
    i_ops = [lab for lab in labels if lab.endswith("_i")]

    has_poly = any(lab.endswith(("_t", "_o", "_i")) for lab in labels)
    axis_use = axis if axis in {"x", "y", "z"} else "z"
    sigma_h_label = {"x": "sigma_yz", "y": "sigma_xz", "z": "sigma_xy"}[axis_use]
    sigma_v_labels = {
        "x": {"sigma_xy", "sigma_xz"},
        "y": {"sigma_xy", "sigma_yz"},
        "z": {"sigma_xz", "sigma_yz"},
    }[axis_use]
    has_sigma_h = sigma_h_label in labels
    has_sigma_v = any(lab.startswith("sigma_v") for lab in labels) or any(
        lab in sigma_v_labels for lab in labels
    )
    c2_axes = set()
    explicit_c2_axes = set()
    for lab in labels:
        m = re.match(r"C2([xyz])\^", lab)
        if m:
            c2_axes.add(m.group(1))
            explicit_c2_axes.add(m.group(1))
        if lab.startswith("C2_xy"):
            c2_axes.add("xy")
    has_c2_perp = any(ax != axis for ax in c2_axes)
    has_explicit_c2_perp = any(ax != axis for ax in explicit_c2_axes)

    if linear and not has_poly:
        return "Dinfh" if has_i else "Cinfv"
    if not has_poly:
        if (nmax >= 3 and ncount_z >= 2 and has_sigma_v) or (nmax >= 6 and has_sigma_v):
            return "Dinfh" if (has_i or has_c2_perp) else "Cinfv"

    n_t = len(t_ops)
    n_o = len(o_ops)
    n_i = len(i_ops)
    if n_i >= 20:
        return "Ih" if has_i else "I"
    if n_o >= 12:
        return "Oh" if has_i else "O"
    if n_t >= 6:
        return "Th" if has_i else "Td" if has_sigma else "T"

    if has_s and has_c2_perp and not has_sigma:
        n_eff = nmax if nmax > 1 else snmax
        return f"D{n_eff}d"
    if has_s and not has_sigma and not has_i:
        return f"S{snmax}"

    if nmax >= 2:
        if has_sigma_h and has_explicit_c2_perp:
            return f"D{nmax}h"
        if has_sigma_h:
            return f"C{nmax}h"
        if has_sigma:
            return f"C{nmax}v"
        if has_c2_perp:
            return f"D{nmax}"
        return f"C{nmax}"

    if has_i:
        return "Ci"
    if has_sigma:
        return "Cs"
    return "C1"
