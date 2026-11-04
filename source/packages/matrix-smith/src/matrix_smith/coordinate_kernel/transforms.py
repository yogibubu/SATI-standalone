from __future__ import annotations

import os
import numpy as np

from matrix_numerics import numerical_matrix_rank

from .pipeline import b_matrix, eval_primitives, zeff_from_topology
from .symmetry_local import (
    atom_classes,
    match_local_geometry,
    pattern_for_center,
    smooth_geometry_label,
    write_pattern_report,
    write_pattern_report_txt,
)
from .topo_groups import group_primitives, primitive_signature


def mass_matrices(masses):
    m = np.array(masses, dtype=float)
    nat = len(m)
    Minv = np.zeros((3 * nat, 3 * nat))
    Minv_sqrt = np.zeros((3 * nat, 3 * nat))
    for i, mi in enumerate(m):
        for c in range(3):
            idx = 3 * i + c
            Minv[idx, idx] = 1.0 / mi
            Minv_sqrt[idx, idx] = 1.0 / np.sqrt(mi)
    return Minv, Minv_sqrt


def bplus(B, masses=None, mass_weighted=False):
    if mass_weighted:
        if masses is None:
            raise ValueError("masses required for mass_weighted=True")
        Minv, _ = mass_matrices(masses)
        G = B @ Minv @ B.T
        return Minv @ B.T @ np.linalg.pinv(G)
    G = B @ B.T
    return B.T @ np.linalg.pinv(G)


def cart_to_internal_grad(gx, B, U=None):
    gs = B @ gx
    if U is None:
        return gs
    return U.T @ gs


def internal_to_cart_grad(gq_or_gs, B, U=None, masses=None, mass_weighted=False):
    gs = gq_or_gs if U is None else U @ gq_or_gs
    Bp = bplus(B, masses=masses, mass_weighted=mass_weighted)
    return Bp @ gs


def cart_to_internal_coords(coords, prims, U=None):
    from .primitives import eval_primitives

    s = eval_primitives(prims, coords)
    if U is None:
        return s
    return U.T @ s


def _build_mass_inv(masses):
    nat = len(masses)
    Minv = np.zeros((3 * nat, 3 * nat))
    for i, m in enumerate(masses):
        for c in range(3):
            idx = 3 * i + c
            Minv[idx, idx] = 1.0 / m
    return Minv


def _invert_matrix_with_tol(mat, tol=1e-12):
    """SVD-based inverse with tolerance to handle near-singular matrices."""
    U, S, Vt = np.linalg.svd(mat, full_matrices=False)
    S_inv = np.array([1.0 / s if s > tol else 0.0 for s in S])
    return (Vt.T * S_inv) @ U.T


def compute_fortran_update_matrix(B, masses=None, tol=1e-12):
    """Reconstruct G = B M^-1 B^T and return M^-1 B^T G^-1."""
    num_prims, num_coords = B.shape
    if masses is None:
        Minv = np.eye(num_coords)
    else:
        Minv = _build_mass_inv(masses)
    G = B @ Minv @ B.T
    G_inv = _invert_matrix_with_tol(G, tol=tol)
    return Minv @ B.T @ G_inv


def internal_to_cart_coords(
    q_or_s,
    coords0,
    prims,
    U=None,
    masses=None,
    mass_weighted=False,
    fd_step=1e-4,
    max_iter=50,
    tol=1e-8,
    metric_weights=None,
):
    """Iterative back-transform from internal to Cartesian coords.

    Uses Fortran-like linearized updates: x <- x + (B G^-1) (s_target - s(x)),
    where G = B M^-1 B^T and the pseudo-inverse uses SVD with tolerance.
    """
    from .primitives import eval_primitives

    coords = coords0.copy()
    if U is None:
        s_target = np.array(q_or_s, dtype=float)
    else:
        s_target = U @ np.array(q_or_s, dtype=float)
    if metric_weights is not None:
        mw = np.array(metric_weights, dtype=float).reshape(-1)
        if mw.size != len(prims):
            raise ValueError("metric_weights size does not match primitives")
        if np.any(mw < 0.0):
            raise ValueError("metric_weights must be non-negative")
        row_scale = np.sqrt(mw)
    else:
        row_scale = None

    for _ in range(max_iter):
        s = eval_primitives(prims, coords)
        ds = s_target - s
        if np.linalg.norm(ds) < tol:
            break
        B = b_matrix(prims, coords, fd_step)
        if row_scale is not None:
            B = B * row_scale[:, None]
            ds = ds * row_scale
        G1B = compute_fortran_update_matrix(B, masses if mass_weighted else None, tol=1e-5)
        dx = G1B @ ds
        coords = coords + dx.reshape(coords.shape)
    return coords


def cart_to_internal_hess(
    Hx,
    gx,
    coords,
    prims,
    U=None,
    masses=None,
    mass_weighted=False,
    include_curvature=False,
    fd_step=1e-4,
    dq=1e-3,
):
    """Transform Cartesian Hessian to internal coordinates.

    If include_curvature and ||gx||>0, uses finite-difference on g_s along
    internal directions to avoid explicit second derivatives of s(x).
    """
    B = b_matrix(prims, coords, fd_step)

    if mass_weighted:
        if masses is None:
            raise ValueError("masses required for mass_weighted=True")
        _, Minv_sqrt = mass_matrices(masses)
        B_use = B @ Minv_sqrt
        gx_use = Minv_sqrt @ gx
        Hx_use = Minv_sqrt @ Hx @ Minv_sqrt
    else:
        B_use = B
        gx_use = gx
        Hx_use = Hx

    g_s = B_use @ gx_use

    if (not include_curvature) or (np.linalg.norm(gx_use) < 1e-12):
        Hs = B_use @ Hx_use @ B_use.T
        if U is not None:
            Hs = U.T @ Hs @ U
        return 0.5 * (Hs + Hs.T)

    # curvature-including FD in internal space
    Bp = bplus(B_use, masses=masses, mass_weighted=False)
    Hs = np.zeros((B_use.shape[0], B_use.shape[0]))
    for j in range(B_use.shape[0]):
        dxp = Bp[:, j] * dq
        if mass_weighted:
            dx = np.linalg.solve(Minv_sqrt, dxp)
        else:
            dx = dxp
        coords_p = coords + dx.reshape(coords.shape)

        B_p = b_matrix(prims, coords_p, fd_step)
        if mass_weighted:
            B_p = B_p @ Minv_sqrt

        gx_p = gx_use + Hx_use @ dxp
        g_s_p = B_p @ gx_p

        Hs[:, j] = (g_s_p - g_s) / dq

    Hs = 0.5 * (Hs + Hs.T)
    if U is not None:
        Hs = U.T @ Hs @ U
    return Hs


def internal_to_cart_hess(Hs, B, U=None, masses=None, mass_weighted=False):
    """Approximate inverse transform of internal Hessian to Cartesian.

    Curvature term is not included here (assumes stationary or approximate).
    """
    Hs_use = Hs if U is None else U @ Hs @ U.T

    if mass_weighted:
        if masses is None:
            raise ValueError("masses required for mass_weighted=True")
        _, Minv_sqrt = mass_matrices(masses)
        B_use = B @ Minv_sqrt
    else:
        B_use = B

    Bp = bplus(B_use, masses=masses, mass_weighted=False)
    Hx_use = Bp @ Hs_use @ Bp.T

    if mass_weighted:
        _, Minv_sqrt = mass_matrices(masses)
        Hx_use = Minv_sqrt @ Hx_use @ Minv_sqrt
    return 0.5 * (Hx_use + Hx_use.T)


def _ring_index_for_atoms(ringset, atoms):
    if ringset is None:
        return None
    aset = set(atoms)
    best = None
    best_len = None
    for ring in ringset:
        rset = set(ring.atoms)
        if aset.issubset(rset):
            if (
                best is None
                or len(rset) < best_len
                or (len(rset) == best_len and ring.index < best)
            ):
                best = ring.index
                best_len = len(rset)
    return best


def _ring_components(ringset):
    if ringset is None:
        return []
    n = len(ringset)
    seen = [False] * n
    comps = []
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp = [i]
        while stack:
            r = stack.pop()
            for nb in ringset.fused_rings(r):
                if not seen[nb]:
                    seen[nb] = True
                    stack.append(nb)
                    comp.append(nb)
        comps.append(comp)
    return comps


def _ring_component_for_atoms(ringset, atoms):
    if ringset is None:
        return None
    aset = set(atoms)
    comps = _ring_components(ringset)
    for ci, comp in enumerate(comps):
        atom_set = set()
        for r in comp:
            atom_set.update(ringset.get_ring(r).atoms)
        if aset.issubset(atom_set):
            return ci, len(comp)
    return None


def _orthonormal_basis_sumdiff(n):
    if n <= 0:
        return np.zeros((0, 0), dtype=float)
    v0 = np.ones((n, 1), dtype=float) / np.sqrt(n)
    diffs = []
    for i in range(n - 1):
        v = np.zeros(n, dtype=float)
        v[i] = 1.0
        v[i + 1] = -1.0
        diffs.append(v)
    if not diffs:
        return v0
    # Gram-Schmidt
    basis = [v0[:, 0]]
    for v in diffs:
        w = v.copy()
        for b in basis:
            w = w - np.dot(w, b) * b
        nrm = np.linalg.norm(w)
        if nrm > 1e-12:
            basis.append(w / nrm)
    U = np.stack(basis, axis=1)
    return U


def _xy3_basis():
    return np.array(
        [
            [2.0 / np.sqrt(6.0), 0.0],
            [-1.0 / np.sqrt(6.0), 1.0 / np.sqrt(2.0)],
            [-1.0 / np.sqrt(6.0), -1.0 / np.sqrt(2.0)],
        ],
        dtype=float,
    )


def _choose_xy3_unique(neigh, degree, Z=None, atom_class=None):
    neigh = list(neigh)
    if len(neigh) != 3:
        raise ValueError("XY3 selection requires three neighbors")
    if Z is not None:
        h = [int(Z[i] == 1) for i in neigh]
        if sum(h) == 1:
            return neigh[h.index(1)]
    term = [int(degree[i] <= 1) for i in neigh]
    if sum(term) == 1:
        return neigh[term.index(1)]
    if atom_class is not None:
        classes = [atom_class[i] for i in neigh]
        counts = {c: classes.count(c) for c in set(classes)}
        unique = [i for i, c in zip(neigh, classes) if counts[c] == 1]
        if len(unique) == 1:
            return unique[0]
    return min(neigh)


def _xy3_angle_block(prims, center, neigh, degree, Z=None, atom_class=None):
    if len(neigh) != 3:
        return None, None
    unique = _choose_xy3_unique(neigh, degree, Z=Z, atom_class=atom_class)
    others = [i for i in neigh if i != unique]
    others = sorted(others)
    pair_order = [
        tuple(sorted((others[0], others[1]))),
        tuple(sorted((unique, others[0]))),
        tuple(sorted((unique, others[1]))),
    ]
    angle_map = {}
    for idx, p in enumerate(prims):
        if p.kind != "angle" or p.atoms[1] != center:
            continue
        angle_map[tuple(sorted((p.atoms[0], p.atoms[2])))] = idx
    idxs = []
    for pair in pair_order:
        idx = angle_map.get(pair)
        if idx is None:
            return None, None
        idxs.append(idx)
    U = _xy3_basis()
    return idxs, U


def _xy2_angle_block(prims, center, neigh):
    if len(neigh) != 2:
        return None, None
    angle_map = {}
    for idx, p in enumerate(prims):
        if p.kind != "angle" or p.atoms[1] != center:
            continue
        angle_map[tuple(sorted((p.atoms[0], p.atoms[2])))] = idx
    pair = tuple(sorted((neigh[0], neigh[1])))
    idx = angle_map.get(pair)
    if idx is None:
        return None, None
    return [idx], np.eye(1, dtype=float)


def _high_coord_angle_block(prims, center, neigh):
    neigh = set(neigh)
    idxs = []
    for idx, p in enumerate(prims):
        if p.kind != "angle" or p.atoms[1] != center:
            continue
        a, _, b = p.atoms
        if a in neigh and b in neigh:
            idxs.append(idx)
    if len(idxs) <= 1:
        return None, None
    return idxs, np.eye(len(idxs), dtype=float)


def _ring_ordered_indices(prims, ring, kind):
    from matrix_chem.topology.ring_primitives import ring_bonds, ring_dihedrals, ring_valence_angles

    idxs = []
    # build map for fast lookup
    if kind == "bond":
        want = ring_bonds(ring)
        bond_map = {}
        for i, p in enumerate(prims):
            if p.kind != "bond":
                continue
            a, b = p.atoms
            bond_map.setdefault(tuple(sorted((a, b))), i)
        for a, b in want:
            idx = bond_map.get(tuple(sorted((a, b))))
            if idx is not None:
                idxs.append(idx)
    elif kind == "angle":
        want = ring_valence_angles(ring)
        angle_map = {}
        for i, p in enumerate(prims):
            if p.kind != "angle":
                continue
            a, j, b = p.atoms
            key = (j, tuple(sorted((a, b))))
            angle_map.setdefault(key, i)
        for a, j, b in want:
            idx = angle_map.get((j, tuple(sorted((a, b)))))
            if idx is not None:
                idxs.append(idx)
    elif kind == "dihedral":
        want = ring_dihedrals(ring)
        dihed_map = {}
        dihed_map_by_bond = {}
        for i, p in enumerate(prims):
            if p.kind != "dihedral":
                continue
            dihed_map.setdefault(tuple(p.atoms), i)
            a, j, k, b = p.atoms
            key = (min(j, k), max(j, k), min(a, b), max(a, b))
            if key not in dihed_map_by_bond:
                dihed_map_by_bond[key] = i
        for a, j, k, b in want:
            idx = dihed_map.get((a, j, k, b))
            if idx is None:
                idx = dihed_map.get((b, k, j, a))
            if idx is None:
                key = (min(j, k), max(j, k), min(a, b), max(a, b))
                idx = dihed_map_by_bond.get(key)
            if idx is not None:
                idxs.append(idx)
    return idxs


def _planarity_tol(n):
    # size-adaptive tolerance (angstrom): looser for larger rings
    return 0.02 + 0.004 * max(0, n - 3)


def _ring_cyclic_u(prims, ringset, kind, planar_tol=0.05):
    if ringset is None:
        return np.zeros((0, 0), dtype=float), []

    # per ring indices in canonical order
    ring_idx_lists = []
    for ring in ringset:
        if kind == "dihedral":
            try:
                tol = _planarity_tol(len(ring.atoms))
                if ring.planarity() is not None and ring.planarity() < tol:
                    continue
            except Exception:
                pass
        idxs = _ring_ordered_indices(prims, ring, kind)
        if idxs:
            ring_idx_lists.append((ring.index, idxs))

    if not ring_idx_lists:
        return np.zeros((0, 0), dtype=float), []

    # ring cluster reduction (drop symmetric col for one ring per cluster)
    comps = _ring_components(ringset)
    drop_ring = set()
    for comp in comps:
        if len(comp) > 1:
            drop_ring.add(max(comp))

    # assemble block
    idxs_all = []
    blocks = []
    for ring_idx, idxs in ring_idx_lists:
        for i in idxs:
            if i not in idxs_all:
                idxs_all.append(i)
        Uloc = _orthonormal_basis_sumdiff(len(idxs))
        if ring_idx in drop_ring and Uloc.shape[1] > 0:
            # drop symmetric column (first)
            Uloc = Uloc[:, 1:] if Uloc.shape[1] > 1 else np.zeros((len(idxs), 0))
        blocks.append((idxs, Uloc))

    n_rows = len(idxs_all)
    n_nonred = sum(b.shape[1] for _, b in blocks)
    U = np.zeros((n_rows, n_nonred), dtype=float)
    row_map = {idx: r for r, idx in enumerate(idxs_all)}
    col = 0
    for idxs, block in blocks:
        if block.size == 0:
            continue
        rows = [row_map[i] for i in idxs]
        U[np.ix_(rows, range(col, col + block.shape[1]))] = block
        col += block.shape[1]
    return U, idxs_all


def valence_angle_u(
    prims,
    coords,
    tol=1e-8,
    fd_step=1e-4,
    ringset=None,
    Z=None,
    atom_class=None,
    exclude_centers=None,
):
    """Build non-redundant valence-angle transform U via per-center G blocks.

    Returns:
        U_angles: (n_angles, n_nonred_angles) matrix
        angle_indices: indices of angle primitives in the input list
    """
    angle_indices = []
    all_angle_indices = []
    for i, p in enumerate(prims):
        if p.kind != "angle":
            continue
        all_angle_indices.append(i)
        if _ring_index_for_atoms(ringset, p.atoms) is not None:
            continue
        angle_indices.append(i)
    # Fallback: in strongly connected/symmetric synthetic cases, ring perception
    # may absorb all angles. Keep the full angle set to avoid empty valence block.
    fallback_all_angles = False
    if not angle_indices:
        n_rings = len(ringset) if ringset is not None else 0
        # Keep strict ring exclusion for genuine ring systems; relax it only
        # for over-connected synthetic cases where ring perception explodes.
        if ringset is None or n_rings == 0 or n_rings > coords.shape[0]:
            angle_indices = all_angle_indices
            fallback_all_angles = True
    if not angle_indices:
        return np.zeros((0, 0), dtype=float), []

    n_angles = len(angle_indices)
    blocks = []
    if fallback_all_angles:
        prim_block = [prims[i] for i in angle_indices]
        Bc = b_matrix(prim_block, coords, fd_step)
        G = Bc @ Bc.T
        evals, evecs = np.linalg.eigh(G)
        keep = evals > tol
        if np.any(keep):
            blocks.append((angle_indices, evecs[:, keep]))
    else:
        by_center = {}
        for i in angle_indices:
            _, j, _ = prims[i].atoms
            if exclude_centers is not None and j in exclude_centers:
                continue
            by_center.setdefault(j, []).append(i)
        neigh_sets = {}
        for p in prims:
            if p.kind != "bond":
                continue
            a, b = p.atoms
            neigh_sets.setdefault(a, set()).add(b)
            neigh_sets.setdefault(b, set()).add(a)
        neighbors = {j: sorted(v) for j, v in neigh_sets.items()}
        degree = {j: len(v) for j, v in neigh_sets.items()}
        for center, idxs in sorted(by_center.items()):
            if center in neighbors and len(neighbors[center]) > 4:
                hc_idxs, Uhc = _high_coord_angle_block(prims, center, neighbors[center])
                if hc_idxs is not None:
                    blocks.append((hc_idxs, Uhc))
                    continue
            if len(idxs) == 1 and center in neighbors and len(neighbors[center]) == 2:
                xy2_idxs, Uxy2 = _xy2_angle_block(prims, center, neighbors[center])
                if xy2_idxs is not None:
                    blocks.append((xy2_idxs, Uxy2))
                    continue
            if len(idxs) == 3 and center in neighbors and len(neighbors[center]) == 3:
                xy3_idxs, Uxy3 = _xy3_angle_block(
                    prims,
                    center,
                    neighbors[center],
                    degree,
                    Z=Z,
                    atom_class=atom_class,
                )
                if xy3_idxs is not None:
                    blocks.append((xy3_idxs, Uxy3))
                    continue
            prim_block = [prims[i] for i in idxs]
            Bc = b_matrix(prim_block, coords, fd_step)
            G = Bc @ Bc.T
            evals, evecs = np.linalg.eigh(G)
            keep = evals > tol
            if not np.any(keep):
                continue
            blocks.append((idxs, evecs[:, keep]))

    n_nonred = sum(block.shape[1] for _, block in blocks)
    U = np.zeros((n_angles, n_nonred), dtype=float)

    row_map = {idx: r for r, idx in enumerate(angle_indices)}
    col = 0
    for idxs, block in blocks:
        rows = [row_map[i] for i in idxs]
        U[np.ix_(rows, range(col, col + block.shape[1]))] = block
        col += block.shape[1]

    return U, angle_indices


def _ring_u(prims, coords, ringset, kind, tol=1e-8, fd_step=1e-4):
    idxs = [
        i
        for i, p in enumerate(prims)
        if p.kind == kind and _ring_index_for_atoms(ringset, p.atoms) is not None
    ]
    if not idxs:
        return np.zeros((0, 0), dtype=float), []

    # group by ring
    by_ring = {}
    for i in idxs:
        r = _ring_index_for_atoms(ringset, prims[i].atoms)
        if r is None:
            continue
        by_ring.setdefault(r, []).append(i)

    n_rows = len(idxs)
    blocks = []
    for ring_idx, ridxs in sorted(by_ring.items()):
        prim_block = [prims[i] for i in ridxs]
        Bc = b_matrix(prim_block, coords, fd_step)
        G = Bc @ Bc.T
        evals, evecs = np.linalg.eigh(G)
        keep = evals > tol
        if not np.any(keep):
            continue
        blocks.append((ridxs, evecs[:, keep]))

    n_nonred = sum(block.shape[1] for _, block in blocks)
    U = np.zeros((n_rows, n_nonred), dtype=float)
    row_map = {idx: r for r, idx in enumerate(idxs)}
    col = 0
    for ridxs, block in blocks:
        rows = [row_map[i] for i in ridxs]
        U[np.ix_(rows, range(col, col + block.shape[1]))] = block
        col += block.shape[1]
    return U, idxs


def ring_angle_u(prims, coords, ringset, tol=1e-8, fd_step=1e-4):
    return _ring_u(prims, coords, ringset, "angle", tol=tol, fd_step=fd_step)


def ring_dihedral_u(prims, coords, ringset, tol=1e-8, fd_step=1e-4):
    return _ring_u(prims, coords, ringset, "dihedral", tol=tol, fd_step=fd_step)


def _butterfly_groups(prims, ringset):
    if ringset is None:
        return []
    groups = []
    # map ring index -> atom set
    ring_atoms = {r.index: set(r.atoms) for r in ringset}
    # bonds shared by >=2 rings
    for bond, ring_idxs in ringset.bond_to_rings.items():
        if len(ring_idxs) < 2:
            continue
        j, k = bond
        for a_i in range(len(ring_idxs)):
            for b_i in range(a_i + 1, len(ring_idxs)):
                ra = ring_idxs[a_i]
                rb = ring_idxs[b_i]
                ra_atoms = ring_atoms[ra]
                rb_atoms = ring_atoms[rb]
                ra_only = ra_atoms - rb_atoms
                rb_only = rb_atoms - ra_atoms
                if not ra_only or not rb_only:
                    continue
                idxs = []
                for idx, p in enumerate(prims):
                    if p.kind != "dihedral":
                        continue
                    i, j0, k0, l = p.atoms
                    if set((j0, k0)) != set((j, k)):
                        continue
                    if (i in ra_only and l in rb_only) or (i in rb_only and l in ra_only):
                        idxs.append(idx)
                if idxs:
                    groups.append((bond, ra, rb, idxs))
    return groups


def ring_butterfly_u(prims, coords, ringset, tol=1e-8, fd_step=1e-4):
    """Butterfly coordinates for fused rings sharing a bond."""
    groups = _butterfly_groups(prims, ringset)
    if not groups:
        return np.zeros((0, 0), dtype=float), []

    idxs_all = []
    for _, _, _, idxs in groups:
        for i in idxs:
            if i not in idxs_all:
                idxs_all.append(i)

    n_rows = len(idxs_all)
    blocks = []
    for _, _, _, idxs in groups:
        prim_block = [prims[i] for i in idxs]
        Bc = b_matrix(prim_block, coords, fd_step)
        G = Bc @ Bc.T
        evals, evecs = np.linalg.eigh(G)
        keep = evals > tol
        if not np.any(keep):
            continue
        blocks.append((idxs, evecs[:, keep]))

    n_nonred = sum(block.shape[1] for _, block in blocks)
    U = np.zeros((n_rows, n_nonred), dtype=float)
    row_map = {idx: r for r, idx in enumerate(idxs_all)}
    col = 0
    for ridxs, block in blocks:
        rows = [row_map[i] for i in ridxs]
        U[np.ix_(rows, range(col, col + block.shape[1]))] = block
        col += block.shape[1]
    return U, idxs_all


def ring_condensed_dihedral_u(prims, coords, ringset, tol=1e-8, fd_step=1e-4):
    """Dihedral block for fused (condensed) ring systems."""
    if ringset is None:
        return np.zeros((0, 0), dtype=float), []

    comps = _ring_components(ringset)
    if not comps:
        return np.zeros((0, 0), dtype=float), []

    # build atom sets per component
    comp_atoms = []
    for comp in comps:
        atom_set = set()
        for r in comp:
            atom_set.update(ringset.get_ring(r).atoms)
        comp_atoms.append(atom_set)

    # collect dihedrals fully contained in each multi-ring component
    groups = []
    idxs_all = []
    for ci, comp in enumerate(comps):
        if len(comp) < 2:
            continue
        idxs = []
        aset = comp_atoms[ci]
        for i, p in enumerate(prims):
            if p.kind != "dihedral":
                continue
            if set(p.atoms).issubset(aset):
                idxs.append(i)
        if idxs:
            groups.append(idxs)
            for i in idxs:
                if i not in idxs_all:
                    idxs_all.append(i)

    if not groups:
        return np.zeros((0, 0), dtype=float), []

    n_rows = len(idxs_all)
    blocks = []
    for idxs in groups:
        prim_block = [prims[i] for i in idxs]
        Bc = b_matrix(prim_block, coords, fd_step)
        G = Bc @ Bc.T
        evals, evecs = np.linalg.eigh(G)
        keep = evals > tol
        if not np.any(keep):
            continue
        blocks.append((idxs, evecs[:, keep]))

    n_nonred = sum(block.shape[1] for _, block in blocks)
    U = np.zeros((n_rows, n_nonred), dtype=float)
    row_map = {idx: r for r, idx in enumerate(idxs_all)}
    col = 0
    for ridxs, block in blocks:
        rows = [row_map[i] for i in ridxs]
        U[np.ix_(rows, range(col, col + block.shape[1]))] = block
        col += block.shape[1]
    return U, idxs_all


def dihedral_u(
    prims,
    coords,
    Z=None,
    priority=None,
    mode="pick",
    coords_units="auto",
    ringset=None,
    return_info=False,
):
    """Build non-redundant dihedral transform U.

    For each non-terminal bond, either pick a single dihedral based on
    atomic priority, combine all dihedrals with equal weights, or close the
    local symmetry orbit of the selected representative.
    """
    dihedral_indices = []
    for i, p in enumerate(prims):
        if p.kind != "dihedral":
            continue
        if _ring_index_for_atoms(ringset, p.atoms) is not None:
            continue
        comp = _ring_component_for_atoms(ringset, p.atoms)
        if comp is not None and comp[1] > 1:
            continue
        dihedral_indices.append(i)
    if not dihedral_indices:
        if return_info:
            return np.zeros((0, 0), dtype=float), [], {"mode": mode, "bonds": []}
        return np.zeros((0, 0), dtype=float), []

    # build adjacency from bonds
    nat = coords.shape[0]
    neighbors = [set() for _ in range(nat)]
    for p in prims:
        if p.kind != "bond":
            continue
        a, b = p.atoms
        neighbors[a].add(b)
        neighbors[b].add(a)
    degree = np.array([len(n) for n in neighbors], dtype=int)

    atom_class = None
    if mode == "orbit":
        if Z is None:
            mode = "pick"
        else:
            atom_class, _, _, ringset, _ = atom_classes(coords, Z)

    # priority
    if priority is None:
        if Z is not None:
            priority = zeff_from_topology(coords, Z, coords_units=coords_units)
        else:
            priority = np.zeros(nat, dtype=float)
    elif isinstance(priority, str) and priority == "zeff":
        if Z is None:
            raise ValueError("Z array required for Zeff priority")
        priority = zeff_from_topology(coords, Z, coords_units=coords_units)
    else:
        priority = np.array(priority, dtype=float)

    # group dihedrals by central bond (j,k)
    groups = {}
    for idx in dihedral_indices:
        i, j, k, l = prims[idx].atoms
        key = tuple(sorted((j, k)))
        groups.setdefault(key, []).append(idx)

    row_map = {idx: r for r, idx in enumerate(dihedral_indices)}
    cols = []
    info = {
        "mode": mode,
        "bonds": [],
    }

    for (j, k), idxs in groups.items():
        # terminal if either end has degree 1
        if degree[j] <= 1 or degree[k] <= 1:
            for idx in idxs:
                col = np.zeros(len(dihedral_indices), dtype=float)
                col[row_map[idx]] = 1.0
                cols.append(col)
            if return_info:
                info["bonds"].append(
                    {
                        "bond": [int(j), int(k)],
                        "candidates": [int(x) for x in idxs],
                        "selected": [int(x) for x in idxs],
                        "reason": "terminal",
                    }
                )
            continue

        if mode == "combine":
            col = np.zeros(len(dihedral_indices), dtype=float)
            w = 1.0 / len(idxs)
            for idx in idxs:
                col[row_map[idx]] = w
            cols.append(col)
            if return_info:
                info["bonds"].append(
                    {
                        "bond": [int(j), int(k)],
                        "candidates": [int(x) for x in idxs],
                        "selected": [int(x) for x in idxs],
                        "reason": "combine",
                    }
                )
            continue

        if mode == "orbit":
            best_idx = None
            best_score = None
            for idx in idxs:
                i, j0, k0, l = prims[idx].atoms
                mid = 0.5 * (coords[j0] + coords[k0])
                di = float(np.linalg.norm(coords[i] - mid))
                dl = float(np.linalg.norm(coords[l] - mid))
                dist_score = max(di, dl)
                score = (max(priority[i], priority[l]), min(priority[i], priority[l]), dist_score)
                if best_score is None or score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is None:
                continue
            if atom_class is None:
                col = np.zeros(len(dihedral_indices), dtype=float)
                col[row_map[best_idx]] = 1.0
                cols.append(col)
                continue
            ref_sig = primitive_signature(prims[best_idx], atom_class, Z, ringset=ringset)
            orbit = [
                idx
                for idx in idxs
                if primitive_signature(prims[idx], atom_class, Z, ringset=ringset) == ref_sig
            ]
            if not orbit:
                orbit = [best_idx]
            col = np.zeros(len(dihedral_indices), dtype=float)
            w = 1.0 / len(orbit)
            for idx in orbit:
                col[row_map[idx]] = w
            cols.append(col)
            if return_info:
                info["bonds"].append(
                    {
                        "bond": [int(j), int(k)],
                        "candidates": [int(x) for x in idxs],
                        "selected": [int(x) for x in orbit],
                        "reason": "orbit",
                    }
                )
            continue

        # mode == "pick"
        best_idx = None
        best_score = None
        for idx in idxs:
            i, j0, k0, l = prims[idx].atoms
            mid = 0.5 * (coords[j0] + coords[k0])
            di = float(np.linalg.norm(coords[i] - mid))
            dl = float(np.linalg.norm(coords[l] - mid))
            dist_score = max(di, dl)
            score = (max(priority[i], priority[l]), min(priority[i], priority[l]), dist_score)
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx
        col = np.zeros(len(dihedral_indices), dtype=float)
        col[row_map[best_idx]] = 1.0
        cols.append(col)
        if return_info:
            info["bonds"].append(
                {
                    "bond": [int(j), int(k)],
                    "candidates": [int(x) for x in idxs],
                    "selected": [int(best_idx)],
                    "reason": "priority",
                }
            )

    U = np.stack(cols, axis=1) if cols else np.zeros((len(dihedral_indices), 0))
    if return_info:
        return U, dihedral_indices, info
    return U, dihedral_indices


def linear_u(prims):
    indices = [i for i, p in enumerate(prims) if p.kind == "linear_bend"]
    if not indices:
        return np.zeros((0, 0), dtype=float), []
    U = np.eye(len(indices), dtype=float)
    return U, indices


def oop_u(prims, coords, tol=1e-8, fd_step=1e-4, linear_threshold=np.deg2rad(170.0)):
    from .geometry import angle as geom_angle

    indices = [i for i, p in enumerate(prims) if p.kind == "out_of_plane"]
    if not indices:
        return np.zeros((0, 0), dtype=float), []

    by_center = {}
    for i in indices:
        center, a, b, c = prims[i].atoms
        # Skip oop if any valence angle at center is near-linear
        ang_ab = geom_angle(a, center, b, coords)
        ang_ac = geom_angle(a, center, c, coords)
        ang_bc = geom_angle(b, center, c, coords)
        if ang_ab >= linear_threshold or ang_ac >= linear_threshold or ang_bc >= linear_threshold:
            continue
        by_center.setdefault(center, []).append(i)

    kept = []
    for idxs in by_center.values():
        for i in idxs:
            if i not in kept:
                kept.append(i)
    n_rows = len(kept)
    blocks = []
    for center, idxs in sorted(by_center.items()):
        prim_block = [prims[i] for i in idxs]
        Bc = b_matrix(prim_block, coords, fd_step)
        G = Bc @ Bc.T
        evals, evecs = np.linalg.eigh(G)
        keep = evals > tol
        if not np.any(keep):
            continue
        blocks.append((idxs, evecs[:, keep]))

    n_nonred = sum(block.shape[1] for _, block in blocks)
    U = np.zeros((n_rows, n_nonred), dtype=float)
    row_map = {idx: r for r, idx in enumerate(kept)}
    col = 0
    for idxs, block in blocks:
        rows = [row_map[i] for i in idxs]
        U[np.ix_(rows, range(col, col + block.shape[1]))] = block
        col += block.shape[1]
    return U, kept


def build_u_blocks(
    prims,
    coords,
    Z=None,
    ringset=None,
    tol=1e-8,
    fd_step=1e-4,
    linear_threshold=np.deg2rad(170.0),
    dihedral_mode="pick",
    dihedral_priority=None,
    symmetry_mode="hybrid",
    prune_mode="svd",
    zeff_tol=0.05,
    geometry_match_tol=12.0,
    pattern_report_path=None,
    g_prune_tol=1e-8,
):
    """Return list of (label, indices, U_block) for each contribution."""
    if symmetry_mode == "gblock":
        return _build_u_blocks_gblock(
            prims,
            coords,
            Z=Z,
            ringset=ringset,
            tol=tol,
            fd_step=fd_step,
            linear_threshold=linear_threshold,
            dihedral_mode=dihedral_mode,
            dihedral_priority=dihedral_priority,
        )
    return _build_u_blocks_symmetry(
        prims,
        coords,
        Z=Z,
        ringset=ringset,
        tol=tol,
        fd_step=fd_step,
        linear_threshold=linear_threshold,
        dihedral_mode=dihedral_mode,
        dihedral_priority=dihedral_priority,
        symmetry_mode=symmetry_mode,
        prune_mode=prune_mode,
        zeff_tol=zeff_tol,
        geometry_match_tol=geometry_match_tol,
        pattern_report_path=pattern_report_path,
        g_prune_tol=g_prune_tol,
    )


def _build_u_blocks_gblock(
    prims,
    coords,
    Z=None,
    ringset=None,
    tol=1e-8,
    fd_step=1e-4,
    linear_threshold=np.deg2rad(170.0),
    dihedral_mode="pick",
    dihedral_priority=None,
):
    blocks = []

    # bonds (identity)
    bond_idx = [i for i, p in enumerate(prims) if p.kind == "bond"]
    if bond_idx:
        blocks.append(("bond", bond_idx, np.eye(len(bond_idx), dtype=float)))

    # fragment coords (identity)
    frag_idx = [i for i, p in enumerate(prims) if p.kind.startswith("frag_")]
    if frag_idx:
        blocks.append(("fragment", frag_idx, np.eye(len(frag_idx), dtype=float)))

    # linear angles (identity)
    U_lin, idx_lin = linear_u(prims)
    if idx_lin:
        blocks.append(("linear_bend", idx_lin, U_lin))

    nat = coords.shape[0]
    neighbors = [set() for _ in range(nat)]
    for p in prims:
        if p.kind == "bond":
            i, j = p.atoms
            neighbors[i].add(j)
            neighbors[j].add(i)
    # out-of-plane (reduced)
    U_oop, idx_oop = oop_u(
        prims, coords, tol=tol, fd_step=fd_step, linear_threshold=linear_threshold
    )
    if idx_oop:
        blocks.append(("out_of_plane", idx_oop, U_oop))

    # angles
    U_val, idx_val = valence_angle_u(
        prims,
        coords,
        tol=tol,
        fd_step=fd_step,
        ringset=ringset,
        Z=Z,
    )
    if idx_val:
        blocks.append(("angle", idx_val, U_val))
    U_ring_ang, idx_ring_ang = ring_angle_u(prims, coords, ringset, tol=tol, fd_step=fd_step)
    if idx_ring_ang:
        blocks.append(("cyclic_valence_bend", idx_ring_ang, U_ring_ang))

    # dihedrals
    U_dih, idx_dih = dihedral_u(
        prims,
        coords,
        Z=Z,
        priority=dihedral_priority,
        mode=dihedral_mode,
        coords_units="auto",
        ringset=ringset,
    )
    if idx_dih:
        blocks.append(("dihedral", idx_dih, U_dih))
    U_ring_dih, idx_ring_dih = ring_dihedral_u(prims, coords, ringset, tol=tol, fd_step=fd_step)
    if idx_ring_dih:
        blocks.append(("cyclic_torsion", idx_ring_dih, U_ring_dih))
    U_cond, idx_cond = ring_condensed_dihedral_u(prims, coords, ringset, tol=tol, fd_step=fd_step)
    if idx_cond:
        blocks.append(("hinge", idx_cond, U_cond))

    return blocks


def _build_u_blocks_symmetry(
    prims,
    coords,
    Z=None,
    ringset=None,
    tol=1e-8,
    fd_step=1e-4,
    linear_threshold=np.deg2rad(170.0),
    dihedral_mode="pick",
    dihedral_priority=None,
    symmetry_mode="hybrid",
    prune_mode="svd",
    zeff_tol=0.05,
    geometry_match_tol=12.0,
    pattern_report_path=None,
    g_prune_tol=1e-8,
):
    blocks = []

    if Z is None:
        raise ValueError("Z array required for symmetry_mode != 'gblock'")
    if ringset is None:
        import warnings

        warnings.warn("ringset is None: ring-specific symmetry blocks will be skipped")

    atom_class, class_meta, zeff, ringset, quasi = atom_classes(coords, Z, zeff_tol=zeff_tol)

    n = coords.shape[0]
    neighbors = [set() for _ in range(n)]
    for p in prims:
        if p.kind == "bond":
            i, j = p.atoms
            neighbors[i].add(j)
            neighbors[j].add(i)

    ring_clusters = []
    ring_drop = []
    if ringset is not None:
        comps = _ring_components(ringset)
        ring_clusters = [[int(r) for r in comp] for comp in comps]
        ring_drop = [int(max(comp)) for comp in comps if len(comp) > 1]

    pattern_report = {
        "schema_version": "1.0",
        "zeff_tol": zeff_tol,
        "geometry_match_tol": geometry_match_tol,
        "classes": {int(k): class_meta[k].__dict__ for k in class_meta},
        "ring_clusters": ring_clusters,
        "ring_drop": ring_drop,
        "centers": [],
    }

    for j in range(n):
        neigh = sorted(neighbors[j])
        if len(neigh) < 2:
            continue
        vecs = np.array([coords[i] - coords[j] for i in neigh], dtype=float)
        vecs = np.array([v / np.linalg.norm(v) for v in vecs], dtype=float)
        geom = match_local_geometry(vecs, tol_deg=geometry_match_tol)
        geom_label = smooth_geometry_label(j, neigh, atom_class, geom.label)
        geom = geom.__class__(label=geom_label, score=geom.score, angles=geom.angles)
        pattern_report["centers"].append(pattern_for_center(j, neigh, atom_class, geom))

    # Stretchings are already the desired independent coordinates.  Bending,
    # torsion and out-of-plane blocks below are the ones reduced from their
    # redundant primitive sets.
    bond_idx = [i for i, p in enumerate(prims) if p.kind == "bond"]
    if bond_idx:
        blocks.append(("bond", bond_idx, np.eye(len(bond_idx), dtype=float)))

    # fragment coords (identity)
    frag_idx = [i for i, p in enumerate(prims) if p.kind.startswith("frag_")]
    if frag_idx:
        blocks.append(("fragment", frag_idx, np.eye(len(frag_idx), dtype=float)))

    # linear angles (identity)
    U_lin, idx_lin = linear_u(prims)
    if idx_lin:
        blocks.append(("linear_bend", idx_lin, U_lin))

    # out-of-plane (grouped by signature)
    oop_idx = [i for i, p in enumerate(prims) if p.kind == "out_of_plane"]
    if oop_idx:
        groups = group_primitives(
            [prims[i] for i in oop_idx], atom_class, Z, ringset=ringset, idx_map=oop_idx
        )
        for sig, local in groups.items():
            idxs = [oop_idx[i] for i in local]
            U = _orthonormal_basis_sumdiff(len(idxs))
            blocks.append(("out_of_plane", idxs, U))

    # non-ring valence angles
    U_val, idx_val = valence_angle_u(
        prims,
        coords,
        tol=tol,
        fd_step=fd_step,
        ringset=ringset,
        Z=Z,
        atom_class=atom_class,
    )
    if idx_val:
        blocks.append(("angle", idx_val, U_val))

    # ring angles / dihedrals (cyclic)
    U_ring_ang, idx_ring_ang = _ring_cyclic_u(prims, ringset, "angle")
    if idx_ring_ang:
        blocks.append(("cyclic_valence_bend", idx_ring_ang, U_ring_ang))

    U_ring_dih, idx_ring_dih = _ring_cyclic_u(prims, ringset, "dihedral")
    if idx_ring_dih:
        blocks.append(("cyclic_torsion", idx_ring_dih, U_ring_dih))

    # butterfly + condensed (extra ring dihedral modes)
    U_bfly, idx_bfly = ring_butterfly_u(prims, coords, ringset, tol=tol, fd_step=fd_step)
    if idx_bfly:
        blocks.append(("butterfly", idx_bfly, U_bfly))
    U_cond, idx_cond = ring_condensed_dihedral_u(prims, coords, ringset, tol=tol, fd_step=fd_step)
    if idx_cond:
        blocks.append(("hinge", idx_cond, U_cond))

    # dihedrals (non-ring, one per bond)
    U_dih, idx_dih, dihedral_info = dihedral_u(
        prims,
        coords,
        Z=Z,
        priority=dihedral_priority,
        mode=dihedral_mode,
        coords_units="auto",
        ringset=ringset,
        return_info=True,
    )
    if idx_dih:
        blocks.append(("dihedral", idx_dih, U_dih))

    # prune mode (svd/g/none) per block
    if prune_mode != "none":
        pruned = []
        B = None
        G = None
        if prune_mode == "g":
            B = b_matrix(prims, coords, fd_step)
            G = B @ B.T

        for label, idxs, Ub in blocks:
            if Ub.size == 0:
                continue
            Unew = Ub
            if prune_mode == "svd":
                u, s, _ = np.linalg.svd(Ub, full_matrices=False)
                keep = s > g_prune_tol
                if np.any(keep):
                    Unew = u[:, keep]
                else:
                    Unew = np.zeros((Ub.shape[0], 0), dtype=float)
            elif prune_mode == "g":
                rows = np.array(idxs, dtype=int)
                Gsub = G[np.ix_(rows, rows)]
                Gq = Ub.T @ Gsub @ Ub
                evals, evecs = np.linalg.eigh(Gq)
                keep = evals > g_prune_tol
                if np.any(keep):
                    Unew = Ub @ evecs[:, keep]
                else:
                    Unew = np.zeros((Ub.shape[0], 0), dtype=float)

            if label == "bond":
                Unew = Ub

            if Unew.shape[1] > 0:
                pruned.append((label, idxs, Unew))

        blocks = pruned

    if pattern_report_path:
        pattern_report["quasi_equivalent"] = {int(k): [int(x) for x in v] for k, v in quasi.items()}
        pattern_report["dihedrals"] = dihedral_info
        block_reports = []
        for label, idxs, Ub in blocks:
            cols = []
            for j in range(Ub.shape[1]):
                nz = []
                for r, coeff in enumerate(Ub[:, j]):
                    if abs(coeff) > 1e-6:
                        nz.append({"prim_idx": int(idxs[r]), "coeff": float(coeff)})
                cols.append(nz)
            block_reports.append(
                {
                    "label": label,
                    "indices": [int(i) for i in idxs],
                    "ncols": int(Ub.shape[1]),
                    "columns": cols,
                }
            )
        pattern_report["blocks"] = block_reports
        write_pattern_report(pattern_report_path, pattern_report)
        txt_path = (
            pattern_report_path[:-5] + ".txt"
            if pattern_report_path.endswith(".json")
            else pattern_report_path + ".txt"
        )
        write_pattern_report_txt(txt_path, pattern_report)

    return blocks


def build_u(
    prims,
    coords,
    Z=None,
    ringset=None,
    tol=1e-8,
    fd_step=1e-4,
    linear_threshold=np.deg2rad(170.0),
    dihedral_mode="pick",
    dihedral_priority=None,
    symmetry_mode="hybrid",
    prune_mode="svd",
    zeff_tol=0.05,
    geometry_match_tol=12.0,
    pattern_report_path=None,
    g_prune_tol=1e-8,
    symmetrize_global=False,
    keep_a1_only=False,
    symmetry_tol=1.0e-3,
    symmetry_max_n=6,
    a1_tol=1e-6,
    assign_symmetry_labels=False,
    symmetry_quasi_tol=None,
    symmetry_tol_h=None,
    heavy_only_orient=False,
    symmetry_center_idx=None,
    ignore_isotopes=False,
    max_dev_strict=None,
    symmetry_tol_rel=0.0,
    symmetry_auto_max_n=False,
    symmetry_inertia_tol=1e-3,
    symmetry_max_radius=None,
    symmetry_enforce_radial=True,
    symmetry_profile=False,
    symmetry_group_limit=None,
    symmetry_confidence=False,
):
    """Assemble a full non-redundant U from primitive coordinates."""
    import os

    if os.environ.get("ORACLE_GICFORGE_PROFILE") == "1":
        import time

        t0 = time.perf_counter()
    nprim = len(prims)
    blocks = build_u_blocks(
        prims,
        coords,
        Z=Z,
        ringset=ringset,
        tol=tol,
        fd_step=fd_step,
        linear_threshold=linear_threshold,
        dihedral_mode=dihedral_mode,
        dihedral_priority=dihedral_priority,
        symmetry_mode=symmetry_mode,
        prune_mode=prune_mode,
        zeff_tol=zeff_tol,
        geometry_match_tol=geometry_match_tol,
        pattern_report_path=pattern_report_path,
        g_prune_tol=g_prune_tol,
    )
    U_raw, column_labels = _assemble_u_from_blocks(nprim, blocks)
    keep = _rank_pruned_column_indices(prims, coords, U_raw, column_labels, fd_step=fd_step)
    U = U_raw[:, keep]
    if symmetrize_global:
        from .symmetry_global import symmetrize_u

        U, symm_info = symmetrize_u(
            U,
            prims,
            Z,
            coords,
            tol=symmetry_tol,
            max_n=symmetry_max_n,
            keep_a1_only=keep_a1_only,
            a1_tol=a1_tol,
            label_symmetry=assign_symmetry_labels,
            quasi_tol=symmetry_quasi_tol,
            tol_H=symmetry_tol_h,
            heavy_only_orient=heavy_only_orient,
            center_idx=symmetry_center_idx,
            ignore_isotopes=ignore_isotopes,
            max_dev_strict=max_dev_strict,
            tol_rel=symmetry_tol_rel,
            auto_max_n=symmetry_auto_max_n,
            inertia_tol=symmetry_inertia_tol,
            max_radius=symmetry_max_radius,
            enforce_radial_filter=symmetry_enforce_radial,
            profile=symmetry_profile,
            op_filter=symmetry_group_limit,
        )
        if symmetry_confidence:
            if symm_info.get("max_dev", 0.0) > 0.0:
                conf = max(0.0, 1.0 - symm_info.get("max_dev", 0.0) / max(symmetry_tol, 1e-12))
            else:
                conf = 1.0 if symm_info.get("elements") else 0.0
            symm_info["confidence"] = float(conf)
        if assign_symmetry_labels and pattern_report_path and os.path.exists(pattern_report_path):
            import json

            with open(pattern_report_path, "r", encoding="utf-8") as fh:
                rep = json.load(fh)
            rep["global_symmetry"] = symm_info
            with open(pattern_report_path, "w", encoding="utf-8") as fh:
                json.dump(rep, fh, indent=2)
    if os.environ.get("ORACLE_GICFORGE_PROFILE") == "1":
        import time

        t1 = time.perf_counter()
        print(f"build_u: {t1 - t0:.6f}s, nprim={nprim}, ncols={U.shape[1]}")
    return U


def _assemble_u_from_blocks(nprim: int, blocks) -> tuple[np.ndarray, list[str]]:
    ncols = sum(Ub.shape[1] for _, _, Ub in blocks)
    U = np.zeros((nprim, ncols), dtype=float)
    column_labels: list[str] = []
    col = 0
    for label, idxs, Ub in blocks:
        rows = np.array(idxs, dtype=int)
        U[np.ix_(rows, range(col, col + Ub.shape[1]))] = Ub
        column_labels.extend([label] * Ub.shape[1])
        col += Ub.shape[1]
    return U, column_labels


def _rank_pruned_column_indices(prims, coords, U, column_labels, fd_step=1e-4, tol=1e-7):
    """Return a deterministic non-redundant column subset.

    Local blocks preserve chemical meaning.  This final pass only removes
    residual dependencies between already-built columns, using Wilson B rows
    projected onto the vibrational Cartesian subspace.
    """
    if U.size == 0:
        return []
    target = _vibrational_rank(coords)
    if U.shape[1] <= target:
        return list(range(U.shape[1]))
    B = b_matrix(prims, coords, fd_step)
    projector = _vibrational_projector_local(coords)
    rows = U.T @ B @ projector
    max_rank = numerical_matrix_rank(rows, absolute_tolerance=tol)
    if max_rank < target:
        import warnings

        warnings.warn(
            f"non-redundant GIC basis reaches rank {max_rank} instead of vibrational target {target}; "
            "returning the maximal rank-preserving subset"
        )

    priorities = {
        "bond": 0,
        "fragment": 1,
        "linear_bend": 2,
        "angle": 3,
        "high_coord_angle": 4,
        "cyclic_valence_bend": 5,
        "out_of_plane": 6,
        "cyclic_torsion": 7,
        "dihedral": 8,
        "butterfly": 9,
        "hinge": 10,
    }
    candidates = sorted(
        range(U.shape[1]), key=lambda col: (priorities.get(column_labels[col], 99), col)
    )
    basis: list[np.ndarray] = []
    keep: list[int] = []
    for col in candidates:
        row = rows[col].astype(float, copy=True)
        row_norm0 = float(np.linalg.norm(row))
        if row_norm0 <= 1.0e-12:
            continue
        for item in basis:
            row -= np.dot(row, item) * item
        row_norm = float(np.linalg.norm(row))
        if row_norm > tol and row_norm > 1.0e-8 * row_norm0:
            basis.append(row / row_norm)
            keep.append(col)
        if len(keep) == min(target, max_rank):
            break
    if len(keep) != min(target, max_rank):
        return list(range(U.shape[1]))
    return sorted(keep)


def _vibrational_rank(coords) -> int:
    natoms = int(np.asarray(coords).shape[0])
    external = []
    for axis in range(3):
        vec = np.zeros(3 * natoms, dtype=float)
        vec[axis::3] = 1.0
        external.append(vec)
    for axis in np.eye(3):
        vec = np.array(
            [component for coord in coords for component in np.cross(axis, coord)], dtype=float
        )
        external.append(vec)
    rank = numerical_matrix_rank(
        np.vstack(external),
        absolute_tolerance=1.0e-10,
    )
    return max(0, 3 * natoms - rank)


def _vibrational_projector_local(coords) -> np.ndarray:
    natoms = int(np.asarray(coords).shape[0])
    basis = []
    for axis in range(3):
        vec = np.zeros(3 * natoms, dtype=float)
        vec[axis::3] = 1.0
        basis.append(vec)
    for axis in np.eye(3):
        vec = np.array(
            [component for coord in coords for component in np.cross(axis, coord)], dtype=float
        )
        basis.append(vec)
    ortho: list[np.ndarray] = []
    for vec in basis:
        residual = vec.astype(float, copy=True)
        for item in ortho:
            residual -= np.dot(residual, item) * item
        norm = float(np.linalg.norm(residual))
        if norm > 1.0e-10:
            ortho.append(residual / norm)
    if not ortho:
        return np.eye(3 * natoms, dtype=float)
    q_matrix = np.vstack(ortho).T
    return np.eye(3 * natoms, dtype=float) - q_matrix @ q_matrix.T


def _primitive_label(p):
    if p.kind == "bond":
        i, j = p.atoms
        return f"R({i + 1:3d},{j + 1:3d})"
    if p.kind == "angle":
        i, j, k = p.atoms
        return f"A({i + 1:3d},{j + 1:3d},{k + 1:3d})"
    if p.kind == "dihedral":
        i, j, k, l = p.atoms
        return f"D({i + 1:3d},{j + 1:3d},{k + 1:3d},{l + 1:3d})"
    if p.kind == "out_of_plane":
        i, j, k, l = p.atoms
        return f"U({i + 1:3d},{j + 1:3d},{k + 1:3d},{l + 1:3d})"
    if p.kind == "linear_bend":
        i, j, k = p.atoms
        comp = 1 if p.mode == -1 else 2
        return f"L{comp}({i + 1:3d},{j + 1:3d},{k + 1:3d})"
    if p.kind.startswith("frag_"):
        return f"X({','.join(str(a + 1) for a in p.atoms)})"
    return f"{p.kind}"


def build_u_with_names(
    prims,
    coords,
    Z=None,
    ringset=None,
    tol=1e-8,
    fd_step=1e-4,
    linear_threshold=np.deg2rad(170.0),
    dihedral_mode="pick",
    dihedral_priority=None,
    include_frag=False,
    min_coeff=1e-4,
    normalize=True,
    symmetry_mode="hybrid",
    prune_mode="svd",
    zeff_tol=0.05,
    geometry_match_tol=12.0,
    pattern_report_path=None,
    g_prune_tol=1e-8,
    symmetrize_global=False,
    keep_a1_only=False,
    symmetry_tol=1.0e-3,
    symmetry_max_n=6,
    a1_tol=1e-6,
    assign_symmetry_labels=False,
    symmetry_quasi_tol=None,
    symmetry_tol_h=None,
    heavy_only_orient=False,
    symmetry_center_idx=None,
    ignore_isotopes=False,
    max_dev_strict=None,
    symmetry_tol_rel=0.0,
    symmetry_auto_max_n=False,
    symmetry_inertia_tol=1e-3,
    symmetry_max_radius=None,
    symmetry_enforce_radial=True,
    symmetry_profile=False,
    symmetry_group_limit=None,
    symmetry_confidence=False,
):
    """Return U and Gaussian-style ReadGIC lines for non-redundant coords."""
    blocks = build_u_blocks(
        prims,
        coords,
        Z=Z,
        ringset=ringset,
        tol=tol,
        fd_step=fd_step,
        linear_threshold=linear_threshold,
        dihedral_mode=dihedral_mode,
        dihedral_priority=dihedral_priority,
        symmetry_mode=symmetry_mode,
        prune_mode=prune_mode,
        zeff_tol=zeff_tol,
        geometry_match_tol=geometry_match_tol,
        pattern_report_path=pattern_report_path,
        g_prune_tol=g_prune_tol,
    )
    nprim = len(prims)
    U_raw, column_labels = _assemble_u_from_blocks(nprim, blocks)
    keep = _rank_pruned_column_indices(prims, coords, U_raw, column_labels, fd_step=fd_step)
    keep_set = set(keep)
    U = U_raw[:, keep]
    if symmetrize_global:
        from .symmetry_global import symmetrize_u

        U, symm_info = symmetrize_u(
            U,
            prims,
            Z,
            coords,
            tol=symmetry_tol,
            max_n=symmetry_max_n,
            keep_a1_only=keep_a1_only,
            a1_tol=a1_tol,
            label_symmetry=assign_symmetry_labels,
            quasi_tol=symmetry_quasi_tol,
            tol_H=symmetry_tol_h,
            heavy_only_orient=heavy_only_orient,
            center_idx=symmetry_center_idx,
            ignore_isotopes=ignore_isotopes,
            max_dev_strict=max_dev_strict,
            tol_rel=symmetry_tol_rel,
            auto_max_n=symmetry_auto_max_n,
            inertia_tol=symmetry_inertia_tol,
            max_radius=symmetry_max_radius,
            enforce_radial_filter=symmetry_enforce_radial,
            profile=symmetry_profile,
            op_filter=symmetry_group_limit,
        )
        if symmetry_confidence:
            if symm_info.get("max_dev", 0.0) > 0.0:
                conf = max(0.0, 1.0 - symm_info.get("max_dev", 0.0) / max(symmetry_tol, 1e-12))
            else:
                conf = 1.0 if symm_info.get("elements") else 0.0
            symm_info["confidence"] = float(conf)
        if assign_symmetry_labels and pattern_report_path and os.path.exists(pattern_report_path):
            import json

            with open(pattern_report_path, "r", encoding="utf-8") as fh:
                rep = json.load(fh)
            rep["global_symmetry"] = symm_info
            with open(pattern_report_path, "w", encoding="utf-8") as fh:
                json.dump(rep, fh, indent=2)

    # compute values
    s = eval_primitives(prims, coords)
    q_raw = U_raw.T @ s

    # naming
    names = []
    counter = {}
    col = 0
    for label, idxs, Ub in blocks:
        # skip fragment coords for readgic unless asked
        if label == "fragment" and not include_frag:
            col += Ub.shape[1]
            continue
        for j in range(Ub.shape[1]):
            raw_col = col
            col += 1
            if raw_col not in keep_set:
                continue
            vec = Ub[:, j]
            terms = []
            for k, coeff in enumerate(vec):
                if abs(coeff) < min_coeff:
                    continue
                prim_idx = idxs[k]
                terms.append((coeff, _primitive_label(prims[prim_idx])))

            if not terms:
                # fallback: keep largest coefficient to preserve naming
                k_max = int(np.argmax(np.abs(vec)))
                prim_idx = idxs[k_max]
                terms = [(vec[k_max], _primitive_label(prims[prim_idx]))]

            if normalize and len(terms) > 1:
                norm = np.sqrt(sum(c * c for c, _ in terms))
                if norm > 1e-12:
                    terms = [(c / norm, lab) for c, lab in terms]

            counter[label] = counter.get(label, 0) + 1
            tag = counter[label]

            if label == "bond":
                name = f"Stre{tag:04d}"
            elif label == "angle":
                name = f"Bend{tag:04d}"
            elif label == "high_coord_angle":
                name = f"HCAn{tag:04d}"
            elif label == "cyclic_valence_bend":
                name = f"CVB{tag:04d}"
            elif label == "dihedral":
                name = f"Dihe{tag:04d}"
            elif label == "cyclic_torsion":
                name = f"CTor{tag:04d}"
            elif label == "ring_breathing":
                name = f"Brea{tag:04d}"
            elif label == "butterfly":
                name = f"Butt{tag:04d}"
            elif label == "hinge":
                name = f"Hing{tag:04d}"
            elif label == "linear_bend":
                name = f"LinB{tag:04d}"
            elif label == "out_of_plane":
                name = f"OuPl{tag:04d}"
            elif label == "fragment":
                name = f"Frag{tag:04d}"
            else:
                name = f"Coord{tag:04d}"

            if len(terms) == 1:
                expr = terms[0][1]
            else:
                parts = []
                for coeff, lab in terms:
                    parts.append(f"{coeff: .5f}*{lab}")
                expr = "[ " + " ".join(parts) + " ]"
            names.append((name, q_raw[raw_col], expr))

    return U, names


def format_readgic_lines(names):
    """Format (name, value, expr) tuples as Gaussian ReadGIC lines."""
    lines = []
    for name, val, expr in names:
        lines.append(f"{name}(Value={val: .5f})={expr}")
    return lines
