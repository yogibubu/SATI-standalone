from __future__ import annotations

import numpy as np
from pathlib import Path
import hashlib
import os

from matrix_chem.topology.elements import atomic_number
from matrix_chem.topology.pipeline import build_topology_objects

from .primitives import Primitive, build_primitives, eval_primitive, grad_primitive

BOHR_TO_ANG = 0.52917721092
_TOPO_CACHE = []
_TOPO_CACHE_MAX = 4


def _cache_dir():
    return os.environ.get("ORACLE_GICFORGE_CACHE_DIR")


def _hash_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def _prims_signature(prims):
    parts = []
    for p in prims:
        parts.append(
            (
                p.kind,
                tuple(p.atoms),
                int(p.mode),
                tuple(p.ref),
                tuple(getattr(p, "refs", ())),
                str(getattr(p, "chart", "PRINCIPAL")),
                str(getattr(p, "smith_function", "")),
                str(getattr(p, "smith_family", "")),
                tuple(getattr(p, "frame_atoms", ())),
                tuple(getattr(p, "ref_frame_atoms", ())),
                getattr(p, "chart_reference_radian", None),
                tuple(getattr(p, "rotation_reference", ())),
            )
        )
    return _hash_bytes(repr(parts).encode("utf-8"))


def _load_topology_elements():
    return atomic_number


def _coords_to_angstrom(coords: np.ndarray, Z=None, coords_units: str = "auto") -> np.ndarray:
    arr = np.array(coords, dtype=float)
    mode = str(coords_units).strip().lower()
    if mode in {"ang", "angstrom", "a"}:
        return arr
    if mode in {"au", "bohr"}:
        return arr * BOHR_TO_ANG
    if mode != "auto":
        raise ValueError("coords_units must be 'auto', 'au', or 'ang'")

    # Auto-detection heuristic:
    # 1) If hydrogens exist, use H-nearest-neighbor distances (most robust).
    # 2) Fallback to global minimum distance.
    # - typical minimum bonded distance in Angstrom: ~0.7-1.6
    # - same in Bohr: ~1.3-3.0
    # If geometry is in Bohr, dmin is usually larger.
    nat = arr.shape[0]
    if nat < 2:
        return arr

    if Z is not None:
        Zarr = np.array(Z, dtype=int)
        h_idx = np.where(Zarr == 1)[0]
        if h_idx.size > 0:
            h_mins = []
            for i in h_idx:
                dmin_i = None
                for j in range(nat):
                    if i == j:
                        continue
                    d = float(np.linalg.norm(arr[i] - arr[j]))
                    if d <= 1.0e-12:
                        continue
                    if dmin_i is None or d < dmin_i:
                        dmin_i = d
                if dmin_i is not None:
                    h_mins.append(dmin_i)
            if h_mins:
                h_med = float(np.median(h_mins))
                # H-neighbor distances around 0.8-1.2 A vs 1.5-2.3 bohr-like input.
                if h_med > 1.25:
                    return arr * BOHR_TO_ANG
                return arr
        # No hydrogens: use global-distance fallback below.

    dmin = None
    for i in range(nat):
        for j in range(i + 1, nat):
            d = float(np.linalg.norm(arr[i] - arr[j]))
            if d <= 1.0e-12:
                continue
            if dmin is None or d < dmin:
                dmin = d
    if dmin is None:
        return arr
    if dmin > 1.60:
        return arr * BOHR_TO_ANG
    return arr


def build_topology(coords, Z, coords_units="auto"):
    coords_ang = _coords_to_angstrom(coords, Z=Z, coords_units=coords_units)
    cg, dg, ringset, synthons, aromaticity = build_topology_objects(
        coords_ang, np.array(Z, dtype=int)
    )
    return cg, dg, ringset


def build_topology_full(coords, Z, coords_units="auto"):
    key = (
        np.array(Z, dtype=int).tobytes(),
        np.array(coords, dtype=float).tobytes(),
        str(coords_units).lower(),
    )
    for k, v in _TOPO_CACHE:
        if k == key:
            return v
    coords_ang = _coords_to_angstrom(coords, Z=Z, coords_units=coords_units)
    cg, dg, ringset, synthons, aromaticity = build_topology_objects(
        coords_ang, np.array(Z, dtype=int)
    )
    value = (cg, dg, ringset, synthons, aromaticity)
    _TOPO_CACHE.append((key, value))
    if len(_TOPO_CACHE) > _TOPO_CACHE_MAX:
        _TOPO_CACHE.pop(0)
    return value


def zeff_from_topology(coords, Z, coords_units="au"):
    """Return effective atomic numbers (Zeff) from topology synthons."""
    coords_ang = _coords_to_angstrom(coords, Z=Z, coords_units=coords_units)
    cg, dg, ringset, synthons, aromaticity = build_topology_objects(
        coords_ang, np.array(Z, dtype=int)
    )
    return np.array([synthons.Zeff(i) for i in range(dg.natoms)], dtype=float)


def _connected_components(adjacency, nat):
    seen = [False] * nat
    comps = []
    for i in range(nat):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp = [i]
        while stack:
            v = stack.pop()
            for nb in adjacency[v]:
                if not seen[nb]:
                    seen[nb] = True
                    stack.append(nb)
                    comp.append(nb)
        comps.append(comp)
    return comps


def primitives_from_topology(
    coords, Z, linear_threshold, include_fragments=True, coords_units="auto"
):
    _, dg, _ = build_topology(coords, Z, coords_units=coords_units)
    prims = build_primitives(dg, coords, linear_threshold)

    if include_fragments:
        # TRIC-style fragment coordinates (Wang & Song, JCP 144, 214108):
        # translations = centroid (geometric center), rotations = exp-map of quaternion
        comps = _connected_components(dg.adjacency, dg.natoms)
        if len(comps) > 1:
            ref_idx = max(range(len(comps)), key=lambda i: len(comps[i]))
            ref_atoms = tuple(sorted(comps[ref_idx]))
            for frag_idx, comp in enumerate(comps):
                if frag_idx == ref_idx:
                    continue
                frag_atoms = tuple(sorted(comp))
                for axis in range(3):
                    prims.append(Primitive("frag_trans", frag_atoms, mode=axis, ref=ref_atoms))
                for axis in range(3):
                    prims.append(Primitive("frag_rot", frag_atoms, mode=axis, ref=ref_atoms))
    return prims


def eval_primitives(prims, coords):
    coords_b = np.array(coords, dtype=float).tobytes()
    psig = _prims_signature(prims)
    key = (psig, coords_b)
    cache = getattr(eval_primitives, "_cache", None)
    order = getattr(eval_primitives, "_order", None)
    if cache is None:
        cache = {}
        order = []
        eval_primitives._cache = cache
        eval_primitives._order = order
    if key in cache:
        return cache[key].copy()
    cdir = _cache_dir()
    if cdir:
        os.makedirs(cdir, exist_ok=True)
        ckey = _hash_bytes(coords_b)
        path = Path(cdir) / f"eval_primitives_{psig}_{ckey}.npy"
        if path.exists():
            return np.load(path)
    val = np.array([eval_primitive(p, coords) for p in prims], dtype=float)
    cache[key] = val
    order.append(key)
    if len(order) > 8:
        old = order.pop(0)
        cache.pop(old, None)
    if cdir:
        np.save(path, val)
    return val.copy()


def b_matrix(prims, coords, fd_step):
    coords_b = np.array(coords, dtype=float).tobytes()
    psig = _prims_signature(prims)
    key = (psig, float(fd_step), coords_b)
    cache = getattr(b_matrix, "_cache", None)
    order = getattr(b_matrix, "_order", None)
    if cache is None:
        cache = {}
        order = []
        b_matrix._cache = cache
        b_matrix._order = order
    if key in cache:
        return cache[key].copy()
    cdir = _cache_dir()
    if cdir:
        os.makedirs(cdir, exist_ok=True)
        ckey = _hash_bytes(coords_b)
        path = Path(cdir) / f"b_matrix_{psig}_{ckey}_{fd_step:.2e}.npy"
        if path.exists():
            return np.load(path)
    nprim = len(prims)
    nat = coords.shape[0]
    b = np.zeros((nprim, 3 * nat))
    for p_idx, p in enumerate(prims):
        g = grad_primitive(p, coords, fd_step)
        b[p_idx, :] = g.reshape(-1)
    cache[key] = b
    order.append(key)
    if len(order) > 8:
        old = order.pop(0)
        cache.pop(old, None)
    if cdir:
        np.save(path, b)
    return b.copy()


def b_matrix_analytic(prims, coords):
    """Return the analytic primitive B matrix.

    Standard valence primitives use closed-form gradients. Fragment translation
    and rotation primitives retain the finite-difference fallback implemented in
    `grad_primitive`; semiexperimental geometry fits use connected molecular
    GICs, so their B matrix is analytic.
    """
    return b_matrix(prims, coords, fd_step=1e-4)


def load_u_matrix(path, nprim):
    if path is None:
        return np.eye(nprim)
    u = np.load(path)
    return u


def q_from_s(u, s):
    return u.T @ s


def gq_from_gx(u, b, gx):
    gs = b @ gx
    return u.T @ gs


def g_matrix(u, b, masses):
    nat = len(masses)
    m_inv = np.zeros((3 * nat, 3 * nat))
    for i, m in enumerate(masses):
        for c in range(3):
            idx = 3 * i + c
            m_inv[idx, idx] = 1.0 / m
    return u.T @ b @ m_inv @ b.T @ u


def g_matrix_derivs(u, prims, coords, masses, fd_step, dq=1e-3):
    """Finite-difference derivatives of G wrt q using linearized dx from B^+.

    Returns:
        G0, dG (nvib,nvib,nvib), d2G (nvib,nvib,nvib,nvib)
    """
    b0 = b_matrix(prims, coords, fd_step)
    u = np.array(u)
    g0 = g_matrix(u, b0, masses)

    nat = coords.shape[0]
    nvib = u.shape[1]

    # pseudo-inverse for dx = B^+ ds
    bp = np.linalg.pinv(b0)

    dG = np.zeros((nvib, nvib, nvib))
    d2G = np.zeros((nvib, nvib, nvib, nvib))

    for k in range(nvib):
        ds = u[:, k] * dq
        dx = bp @ ds
        coords_p = coords + dx.reshape(nat, 3)
        coords_m = coords - dx.reshape(nat, 3)
        b_p = b_matrix(prims, coords_p, fd_step)
        b_m = b_matrix(prims, coords_m, fd_step)
        g_p = g_matrix(u, b_p, masses)
        g_m = g_matrix(u, b_m, masses)
        dG[:, :, k] = (g_p - g_m) / (2.0 * dq)

    # second derivatives (mixed)
    for k in range(nvib):
        for l in range(k, nvib):
            ds_k = u[:, k] * dq
            ds_l = u[:, l] * dq
            dx_k = bp @ ds_k
            dx_l = bp @ ds_l
            coords_pp = coords + (dx_k + dx_l).reshape(nat, 3)
            coords_pm = coords + (dx_k - dx_l).reshape(nat, 3)
            coords_mp = coords + (-dx_k + dx_l).reshape(nat, 3)
            coords_mm = coords + (-dx_k - dx_l).reshape(nat, 3)
            g_pp = g_matrix(u, b_matrix(prims, coords_pp, fd_step), masses)
            g_pm = g_matrix(u, b_matrix(prims, coords_pm, fd_step), masses)
            g_mp = g_matrix(u, b_matrix(prims, coords_mp, fd_step), masses)
            g_mm = g_matrix(u, b_matrix(prims, coords_mm, fd_step), masses)
            d2 = (g_pp - g_pm - g_mp + g_mm) / (4.0 * dq * dq)
            d2G[:, :, k, l] = d2
            if l != k:
                d2G[:, :, l, k] = d2

    return g0, dG, d2G
