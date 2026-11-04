from __future__ import annotations

from pathlib import Path
import numpy as np

from matrix_gaussian.survibfit_gaussian_log import read_standard_orientation, read_cartesian_force_constants
from .pipeline import primitives_from_topology, b_matrix, g_matrix, build_topology
from .transforms import build_u

BOHR_TO_ANG = 0.52917721092


def _hessian_eig_to_freq_cm1(eigvals):
    # Convert eigenvalues of mass-weighted Hessian (Eh/Bohr^2/amu) to cm^-1
    c = 2.99792458e10  # cm/s
    Eh = 4.3597447222071e-18  # J
    a0 = 5.29177210903e-11  # m
    amu = 1.66053906660e-27  # kg
    factor = (1.0 / (2.0 * np.pi * c)) * np.sqrt(Eh / (a0 * a0 * amu))
    return np.sign(eigvals) * np.sqrt(np.abs(eigvals)) * factor


def _pulay_scale_diag(prims, scale_map=None):
    if scale_map is None:
        scale_map = {}
    legacy = {
        "dihed": "dihedral",
        "oop": "out_of_plane",
        "linear": "linear_bend",
        "frag": "fragment",
    }
    s = np.ones(len(prims), dtype=float)
    for i, p in enumerate(prims):
        key = p.kind
        if key.startswith("frag_"):
            key = "fragment"
        key = legacy.get(key, key)
        if key in scale_map:
            s[i] *= float(scale_map[key])
        elif key == "fragment" and "frag" in scale_map:
            s[i] *= float(scale_map["frag"])
    return s


def hessian_cart_to_internal(
    Hx,
    coords_ang,
    Z,
    masses_amu,
    scale_map=None,
    linear_threshold=np.deg2rad(170.0),
    fd_step=1e-4,
):
    """Transform Cartesian Hessian to non-redundant internal-coordinate Hessian."""
    coords_au = np.array(coords_ang, dtype=float) / BOHR_TO_ANG
    prims = primitives_from_topology(coords_au, Z, linear_threshold)
    _, _, ringset = build_topology(coords_au, Z)

    B = b_matrix(prims, coords_au, fd_step)
    Hs = B @ Hx @ B.T

    # Pulay-type scaling on primitive coordinates
    S = _pulay_scale_diag(prims, scale_map=scale_map)
    Hs = (S[:, None] * Hs) * S[None, :]

    U = build_u(prims, coords_au, Z=Z, ringset=ringset, tol=1e-8, fd_step=fd_step)
    F = U.T @ Hs @ U
    G = g_matrix(U, B, masses_amu)

    return F, G, U, prims


def gf_frequencies(F, G):
    """Solve generalized eigenproblem F L = G L λ and return frequencies/modes."""
    # Symmetric orthogonalization of G
    evals_g, evecs_g = np.linalg.eigh(G)
    keep = evals_g > 1e-12
    if not np.any(keep):
        raise ValueError("G matrix is singular")
    G_inv_sqrt = evecs_g[:, keep] @ np.diag(1.0 / np.sqrt(evals_g[keep])) @ evecs_g[:, keep].T
    A = G_inv_sqrt @ F @ G_inv_sqrt

    evals, evecs = np.linalg.eigh(A)
    order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]

    freqs = _hessian_eig_to_freq_cm1(evals)
    modes_q = G_inv_sqrt @ evecs
    return freqs, modes_q


def modes_from_gaussian_log(
    log_path: Path,
    fchk_path: Path | None = None,
    scale_map=None,
):
    """Read Gaussian log Hessian and compute non-redundant internal modes."""
    Z, coords_ang = read_standard_orientation(Path(log_path))
    Hx = read_cartesian_force_constants(Path(log_path))

    if fchk_path is None:
        cand = Path(log_path).with_suffix(".fchk")
        if cand.exists():
            fchk_path = cand
        else:
            cand = Path(log_path).with_suffix(".fch")
            if cand.exists():
                fchk_path = cand
    if fchk_path is None:
        raise ValueError("fchk_path is required for atomic masses")
    masses_amu = read_fchk_masses(fchk_path)

    F, G, U, prims = hessian_cart_to_internal(Hx, coords_ang, Z, masses_amu, scale_map=scale_map)
    freqs, modes_q = gf_frequencies(F, G)
    return freqs, modes_q, U, prims


def read_fchk_masses(fchk):
    """Read atomic masses from a Gaussian fchk file."""
    with open(fchk, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        l = lines[i]
        if (
            l.startswith("Real atomic weights")
            or "Atomic masses" in l
            or l.startswith("Vib-AtMass")
        ):
            n = int(l.split()[-1])
            vals = []
            i += 1
            while len(vals) < n and i < len(lines):
                vals += [float(x) for x in lines[i].split()]
                i += 1
            return np.array(vals, dtype=float)
        i += 1

    raise ValueError("FCHK masses not found (Real atomic weights / Atomic masses / Vib-AtMass)")
