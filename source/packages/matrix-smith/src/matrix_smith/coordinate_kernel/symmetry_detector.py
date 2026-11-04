from __future__ import annotations

import numpy as np

from .symmetry_ops import candidate_ops


def match_with_map(
    symbols,
    coords1,
    coords2,
    tol,
    tol_H=None,
    tol_rel=0.0,
    enforce_radial_filter=True,
):
    used = np.zeros(len(coords2), dtype=bool)
    mapping = [-1] * len(coords1)
    max_dev = 0.0

    sym_to_idx2 = {}
    for j, sym in enumerate(symbols):
        sym_to_idx2.setdefault(sym, []).append(j)
    sym_counts = {sym: len(idxs) for sym, idxs in sym_to_idx2.items()}
    radii2 = np.linalg.norm(coords2, axis=1)

    order = sorted(range(len(coords1)), key=lambda i: (sym_counts.get(symbols[i], 0), i))
    for i in order:
        v = coords1[i]
        sym = symbols[i]
        idxs = sym_to_idx2.get(sym, [])
        if not idxs:
            return None, None
        eff_tol = tol
        if tol_H is not None and sym == "H":
            eff_tol = tol_H
        r1 = float(np.linalg.norm(v))
        rad_tol = eff_tol + tol_rel * r1
        found = False
        if enforce_radial_filter:
            candidates = [j for j in idxs if not used[j] and abs(radii2[j] - r1) <= rad_tol]
        else:
            candidates = [j for j in idxs if not used[j]]
        candidates.sort(key=lambda j: abs(radii2[j] - r1))
        for j in candidates:
            d = np.linalg.norm(v - coords2[j])
            if d < eff_tol:
                used[j] = True
                mapping[i] = j
                if d > max_dev:
                    max_dev = float(d)
                found = True
                break
        if not found:
            return None, None
    return mapping, max_dev


def orient_coords(coords, weights=None):
    x = np.array(coords, dtype=float)
    if weights is None:
        w = np.ones(len(x))
    else:
        w = np.array(weights, dtype=float)
    wsum = np.sum(w)
    com = np.sum(x * w[:, None], axis=0) / wsum
    x = x - com

    I = np.zeros((3, 3))
    for i, r in enumerate(x):
        I += w[i] * ((np.dot(r, r) * np.eye(3)) - np.outer(r, r))
    evals, evecs = np.linalg.eigh(I)
    order = np.argsort(evals)
    R = evecs[:, order]
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1.0
    x_oriented = x @ R
    return x_oriented


def is_linear(coords, tol=1.0e-3):
    x = np.array(coords, dtype=float)
    I = np.zeros((3, 3))
    for r in x:
        I += np.dot(r, r) * np.eye(3) - np.outer(r, r)
    evals = np.linalg.eigvalsh(I)
    return bool(evals[0] < tol)


def symmetry_elements_from_geometry(
    symbols,
    coords_oriented,
    tol=1.0e-3,
    max_n=6,
    tol_H=None,
    ignore_isotopes=False,
    op_filter=None,
    tol_rel=0.0,
    auto_max_n=False,
    inertia_tol=1e-3,
    max_radius=None,
    enforce_radial_filter=True,
    perf=None,
):
    coords = np.asarray(coords_oriented, dtype=float)
    if max_radius is None:
        r = np.linalg.norm(coords, axis=1)
        max_radius = float(np.max(r)) if len(r) else 1.0
        if max_radius <= 0.0:
            max_radius = 1.0
    coords_scaled = coords / max_radius
    sym_use = symbols if not ignore_isotopes else [s[0] for s in symbols]
    elements = []
    permutations = []
    if auto_max_n:
        I = np.zeros((3, 3))
        for r in coords:
            I += np.dot(r, r) * np.eye(3) - np.outer(r, r)
        evals = np.linalg.eigvalsh(I)
        maxI = float(np.max(evals)) if len(evals) else 0.0
        if maxI > 0.0:
            d01 = abs(evals[0] - evals[1]) / maxI
            d12 = abs(evals[1] - evals[2]) / maxI
            if d01 > inertia_tol and d12 > inertia_tol:
                max_n = min(max_n, 2)
    if perf is not None:
        perf.setdefault("ops_total", 0)
        perf.setdefault("ops_kept", 0)
        perf.setdefault("ops_time", 0.0)
        perf.setdefault("match_calls", 0)
    for label, R in candidate_ops(max_n=max_n):
        if op_filter is not None and not op_filter(label):
            continue
        if perf is not None:
            perf["ops_total"] += 1
        coords_t = coords_scaled @ R.T
        mapping, max_dev = match_with_map(
            sym_use,
            coords_scaled,
            coords_t,
            tol,
            tol_H=tol_H,
            tol_rel=tol_rel,
            enforce_radial_filter=enforce_radial_filter,
        )
        if mapping is not None:
            if perf is not None:
                perf["ops_kept"] += 1
            elements.append((label, R, max_dev))
            permutations.append(mapping)

    # build equivalence classes from permutations
    n = len(symbols)
    parent = list(range(n))

    def _find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def _union(a, b):
        ra = _find(a)
        rb = _find(b)
        if ra != rb:
            parent[rb] = ra

    for mapping in permutations:
        for i, j in enumerate(mapping):
            _union(i, j)

    classes = {}
    for i in range(n):
        r = _find(i)
        classes.setdefault(r, []).append(i)

    return elements, list(classes.values()), permutations
