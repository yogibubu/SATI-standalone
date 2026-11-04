from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from matrix_smith.parameters.bdpcs3 import load_bdpcs3_parameters

from .pipeline import primitives_from_topology, b_matrix
from .primitives import Primitive, eval_primitives
from .transforms import compute_fortran_update_matrix

BOHR_TO_ANG = 0.52917721092
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG
BDPCS3_VERSIONS = ("updated", "legacy")

_BDPCS3_PARAMETERS = load_bdpcs3_parameters()
BDPCS3_DEFAULT_WEIGHT_PROFILE = _BDPCS3_PARAMETERS.weights.profile
BDPCS3_WEIGHT_PROFILES = (BDPCS3_DEFAULT_WEIGHT_PROFILE, "unit")
BDPCS3_HBOND_DISTANCE_CUTOFF_ANG = _BDPCS3_PARAMETERS.hbond.distance_cutoff_ang
BDPCS3_HBOND_DISTANCE_WIDTH_ANG = _BDPCS3_PARAMETERS.hbond.distance_width_ang
BDPCS3_HBOND_SEARCH_CUTOFF_ANG = _BDPCS3_PARAMETERS.hbond.search_cutoff_ang
BDPCS3_HBOND_ANGLE_THRESHOLD_DEG = _BDPCS3_PARAMETERS.hbond.angle_threshold_deg

BDPCS3_WEIGHT_STRETCH = _BDPCS3_PARAMETERS.weights.stretch
BDPCS3_WEIGHT_ANGLE = _BDPCS3_PARAMETERS.weights.angle
BDPCS3_WEIGHT_HBOND = _BDPCS3_PARAMETERS.weights.hbond
BDPCS3_WEIGHT_TORSION_MIN = _BDPCS3_PARAMETERS.weights.torsion_min
BDPCS3_WEIGHT_FRAGMENT = _BDPCS3_PARAMETERS.weights.fragment


@dataclass(frozen=True)
class HydrogenBond:
    donor: int
    hydrogen: int
    acceptor: int
    distance_ang: float
    angle_deg: float


def _load_topology_elements():
    from matrix_chem.topology.elements import atomic_number

    return atomic_number


def _load_covalent_radius():
    from matrix_chem.topology.covalent_radii import covalent_radius

    return covalent_radius


def _load_topology_bond_order():
    from matrix_chem.topology.continuous_graph import pauling_bond_order

    return pauling_bond_order


def _load_isotopes_table():
    from matrix_chem.isotopes_table import ISOTOPES

    return ISOTOPES


def _load_average_atomic_mass_table():
    from matrix_chem.average_atomic_masses import atomic_mass

    return atomic_mass


def rcov_ct(z1: int, z2: int) -> float:
    covrad = _load_covalent_radius()
    r1 = covrad(int(z1)) or 0.0
    r2 = covrad(int(z2)) or 0.0
    return r1 + r2


_BDPCS3_SCALES = None


def _principal_quantum(z: int) -> int:
    Z = int(z)
    if Z <= 2:
        return 1
    if Z <= 10:
        return 2
    return 3


def _is_hydrogen(z: int) -> bool:
    return int(z) == 1


def _is_carbon(z: int) -> bool:
    return int(z) == 6






def _is_sulfur(z: int) -> bool:
    return int(z) == 16


def _is_fluorine(z: int) -> bool:
    return int(z) == 9


def _bdpcs3_delta_bcv(z1: int, z2: int, cov_sum: float) -> float:
    ni = min(_principal_quantum(z1), 3)
    nj = min(_principal_quantum(z2), 3)
    term = ni * nj - 1
    if term <= 0:
        return 0.0
    delta_ch = (_is_carbon(z1) and _is_hydrogen(z2)) or (_is_carbon(z2) and _is_hydrogen(z1))
    scale = -0.0011 * (1.0 + 1.1 * float(delta_ch))
    return scale * np.sqrt(term) * cov_sum


def _bdpcs3_delta_bv(z1: int, z2: int, bond_order: float, delta_bcv: float) -> float:
    if delta_bcv == 0.0:
        return 0.0
    is_c1 = _is_carbon(z1)
    is_c2 = _is_carbon(z2)
    is_s1 = _is_sulfur(z1)
    is_s2 = _is_sulfur(z2)
    indicator_special = (is_s1 and is_c2) or (is_s2 and is_c1) or (is_c1 and is_c2)
    indicator_cf = (is_c1 and _is_fluorine(z2)) or (is_c2 and _is_fluorine(z1))
    bo_term = np.sqrt(abs(bond_order - 2.0)) - 1.0
    delta = 0.0
    if indicator_special:
        delta += delta_bcv * bo_term
    if indicator_cf:
        delta += delta_bcv
    return delta


_BDPCS3_UPDATED_PARAMS = {
    "sigma_coord": 0.057,
}

_BDPCS3_UPDATED_RCOV = {
    1: 0.31,  # H
    6: 0.76,  # C
    8: 0.66,  # O
    16: 1.05,  # S
}


def _bdpcs3_updated_rcov(z: int) -> float:
    if int(z) in _BDPCS3_UPDATED_RCOV:
        return _BDPCS3_UPDATED_RCOV[int(z)]
    covrad = _load_covalent_radius()
    return covrad(int(z)) or 0.0


def _bdpcs3_updated_rcov_ct(z1: int, z2: int) -> float:
    return _bdpcs3_updated_rcov(z1) + _bdpcs3_updated_rcov(z2)


def _bdpcs3_pyykko_single_double_triple(z: int):
    from matrix_chem.topology.pykko_radii import bond_order_reference_radii

    return bond_order_reference_radii(
        int(z),
        fallback_radius=_bdpcs3_updated_rcov(z),
    )




def _bdpcs3_electronegativity_scale(z1: int, z2: int) -> float:
    return 0.0025


def _bdpcs3_delocalization_cc_cs(z1: int, z2: int, r_ang: float, delta_cv: float) -> float:
    is_cc = _is_carbon(z1) and _is_carbon(z2)
    is_cs = (_is_carbon(z1) and _is_sulfur(z2)) or (_is_carbon(z2) and _is_sulfur(z1))
    if not (is_cc or is_cs):
        return 0.0

    r1_single, r1_double, r1_triple = _bdpcs3_pyykko_single_double_triple(z1)
    r2_single, r2_double, r2_triple = _bdpcs3_pyykko_single_double_triple(z2)
    r_single = r1_single + r2_single
    r_double = r1_double + r2_double
    r_triple = r1_triple + r2_triple

    d1 = abs(r_double - r_single)
    d2 = abs(r_triple - r_double)
    sigma = max(1.0e-6, min(d1, d2) / 1.5) if min(d1, d2) > 0.0 else 0.05

    # Compensate CV exactly at r_double.
    return -delta_cv * math.exp(-(((r_ang - r_double) / sigma) ** 2))


def topology_bond_order_for_pair(i: int, j: int, Z, coords_ang, cache=None) -> float:
    """
    Topology-consistent bond order for a specific pair.
    Uses the same continuous coordination and BO model as topology module.
    """
    bo_fn = _load_topology_bond_order()
    return bo_fn(i, j, Z, coords_ang)


def _angle_deg_center(i: int, j: int, k: int, coords_ang) -> float:
    """Angle i-j-k in degrees, with j as vertex."""
    coords_arr = np.array(coords_ang, dtype=float)
    v1 = coords_arr[i] - coords_arr[j]
    v2 = coords_arr[k] - coords_arr[j]
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 <= 1.0e-12 or n2 <= 1.0e-12:
        return 0.0
    c = float(np.dot(v1, v2) / (n1 * n2))
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


def _hbond_base_delta_ang(donor_z: int, acceptor_z: int) -> float:
    """BDPCS3 hydrogen-bond correction before geometric damping."""
    return _BDPCS3_PARAMETERS.hbond.correction_ang(donor_z, acceptor_z)


def bdpcs3_hbond_delta(
    donor_z: int,
    acceptor_z: int,
    distance_ang: float,
    angle_deg: float,
    *,
    angle_threshold_deg: float = BDPCS3_HBOND_ANGLE_THRESHOLD_DEG,
    distance_cutoff_ang: float = BDPCS3_HBOND_DISTANCE_CUTOFF_ANG,
    distance_width_ang: float = BDPCS3_HBOND_DISTANCE_WIDTH_ANG,
) -> float:
    """Hydrogen-bond BDPCS3 correction for the H...Y distance in Angstrom."""
    if angle_deg < angle_threshold_deg:
        return 0.0
    base = _hbond_base_delta_ang(donor_z, acceptor_z)
    if base == 0.0:
        return 0.0
    width = max(float(distance_width_ang), 1.0e-6)
    arg = (float(distance_ang) - float(distance_cutoff_ang)) / width
    damping = 0.5 * (1.0 - math.erf(arg))
    damping = max(0.0, min(1.0, damping))
    return base * damping


def detect_hydrogen_bonds(
    Z,
    coords_ang,
    covalent_bonds,
    *,
    angle_threshold_deg: float = BDPCS3_HBOND_ANGLE_THRESHOLD_DEG,
    distance_cutoff_ang: float = BDPCS3_HBOND_SEARCH_CUTOFF_ANG,
) -> list[HydrogenBond]:
    """Find X-H...Y contacts compatible with GICForge's hydrogen-bond logic."""
    Zarr = np.array(Z, dtype=int)
    coords_arr = np.array(coords_ang, dtype=float)
    nat = len(Zarr)
    adjacency = [set() for _ in range(nat)]
    for i, j in covalent_bonds:
        i = int(i)
        j = int(j)
        adjacency[i].add(j)
        adjacency[j].add(i)

    selected: list[HydrogenBond] = []
    for h in range(nat):
        if Zarr[h] != 1:
            continue
        donors = [
            a for a in adjacency[h] if int(Zarr[a]) in _BDPCS3_PARAMETERS.hbond.donor_atomic_numbers
        ]
        if len(donors) != 1:
            continue
        donor = donors[0]
        best = None
        for acc in range(nat):
            if acc == h or acc == donor:
                continue
            if int(Zarr[acc]) not in _BDPCS3_PARAMETERS.hbond.acceptor_atomic_numbers:
                continue
            if acc in adjacency[h]:
                continue
            if acc in adjacency[donor]:
                continue
            donor_neigh = set(adjacency[donor]) - {h, acc}
            acc_neigh = set(adjacency[acc]) - {h, donor}
            if donor_neigh.intersection(acc_neigh):
                continue
            dist = float(np.linalg.norm(coords_arr[h] - coords_arr[acc]))
            if dist > distance_cutoff_ang:
                continue
            angle = _angle_deg_center(donor, h, acc, coords_arr)
            if angle < angle_threshold_deg:
                continue
            candidate = HydrogenBond(donor, h, acc, dist, angle)
            if best is None or candidate.distance_ang < best.distance_ang:
                best = candidate
        if best is not None:
            selected.append(best)

    unique: list[HydrogenBond] = []
    seen_da = set()
    for hb in sorted(selected, key=lambda item: item.distance_ang):
        key = (hb.donor, hb.acceptor)
        if key in seen_da:
            continue
        seen_da.add(key)
        unique.append(hb)
    return sorted(unique, key=lambda item: (item.donor, item.hydrogen, item.acceptor))


def bdpcs3_metric_weights(
    prims,
    Z,
    coords_ang,
    *,
    hbond_pairs=frozenset(),
    profile: str = BDPCS3_DEFAULT_WEIGHT_PROFILE,
) -> np.ndarray:
    """Physical metric weights for internal-to-Cartesian BDPCS3 back-transform."""
    profile_norm = (profile or BDPCS3_DEFAULT_WEIGHT_PROFILE).strip().lower()
    if profile_norm == "unit":
        return np.ones(len(prims), dtype=float)
    if profile_norm != BDPCS3_DEFAULT_WEIGHT_PROFILE.lower():
        raise ValueError(f"Unknown BDPCS3 weight profile: {profile}")

    hbond_pairs_norm = {tuple(sorted((int(i), int(j)))) for i, j in hbond_pairs}
    weights = np.ones(len(prims), dtype=float)
    bo_cache = {}
    for idx, prim in enumerate(prims):
        kind = prim.kind
        if kind == "bond":
            pair = tuple(sorted((int(prim.atoms[0]), int(prim.atoms[1]))))
            weights[idx] = (
                BDPCS3_WEIGHT_HBOND if pair in hbond_pairs_norm else BDPCS3_WEIGHT_STRETCH
            )
        elif kind in {"angle", "linear_bend", "out_of_plane"}:
            weights[idx] = BDPCS3_WEIGHT_ANGLE
        elif kind == "dihedral":
            _, j, k, _ = prim.atoms
            try:
                bo = topology_bond_order_for_pair(j, k, Z, coords_ang, cache=bo_cache)
            except Exception:
                bo = 1.0
            weights[idx] = BDPCS3_WEIGHT_TORSION_MIN
            if np.isfinite(bo) and bo > 1.0:
                weights[idx] = min(
                    BDPCS3_WEIGHT_ANGLE,
                    BDPCS3_WEIGHT_TORSION_MIN
                    + (float(bo) - 1.0) * (BDPCS3_WEIGHT_ANGLE - BDPCS3_WEIGHT_TORSION_MIN),
                )
        elif kind in {"frag_trans", "frag_rot"}:
            weights[idx] = BDPCS3_WEIGHT_FRAGMENT
        else:
            weights[idx] = 1.0
    return weights


def bdpcs3_delta_and_order_updated(
    z1: int, z2: int, r_ang: float, bond_order_override: float | None = None
):
    params = _BDPCS3_UPDATED_PARAMS
    val0 = _bdpcs3_updated_rcov_ct(z1, z2)
    bo_geom = 0.0 if val0 <= 0.0 else np.exp((val0 - r_ang) / 0.30)
    bond_order = (
        max(float(bond_order_override), bo_geom) if bond_order_override is not None else bo_geom
    )

    ni = min(_principal_quantum(z1), 3)
    nj = min(_principal_quantum(z2), 3)
    nmax = max(ni, nj)
    if nmax <= 1 or val0 <= 0.0:
        return 0.0, bond_order

    sigma_coord = params["sigma_coord"]
    f_coord = 0.5 * (1.0 - math.erf((r_ang - (1.3 * val0)) / sigma_coord))

    scale = _bdpcs3_electronegativity_scale(z1, z2)
    delta_cv = -scale * float(nmax - 1)
    delta_deloc = _bdpcs3_delocalization_cc_cs(z1, z2, r_ang, delta_cv)
    delta_r = (delta_cv + delta_deloc) * f_coord

    return delta_r, bond_order


def bdpcs3_function(version: str):
    version_norm = (version or "updated").strip().lower()
    if version_norm in {"updated", "unified"}:
        return bdpcs3_delta_and_order_updated
    if version_norm == "legacy":
        return bdpcs3_delta_and_order
    raise ValueError(f"Unknown BDPCS3 version: {version}")




def _disable_bdpcs3_fit():
    global _BDPCS3_SCALES
    _BDPCS3_SCALES = {}


def bdpcs3_delta_and_order(
    z1: int, z2: int, r_ang: float, bond_order_override: float | None = None
):
    val0 = rcov_ct(z1, z2)
    bo_geom = np.exp((val0 - r_ang) / 0.30)
    bond_order = (
        max(float(bond_order_override), bo_geom) if bond_order_override is not None else bo_geom
    )

    delta_bcv = _bdpcs3_delta_bcv(z1, z2, val0)
    delta_bv = _bdpcs3_delta_bv(z1, z2, bond_order, delta_bcv)
    delta_r = delta_bcv + delta_bv

    is_cc_or_cs = (
        (_is_carbon(z1) and _is_carbon(z2))
        or (_is_carbon(z1) and _is_sulfur(z2))
        or (_is_sulfur(z1) and _is_carbon(z2))
    )
    if is_cc_or_cs and bond_order < 0.3:
        delta_r = 0.0

    return delta_r, bond_order


def bdpcs3_correct_length(z1: int, z2: int, r_ang: float, version: str = "updated") -> float:
    dlt_r, _ = bdpcs3_function(version)(z1, z2, r_ang)
    return r_ang + dlt_r


def read_xyz(path: Path):
    lines = path.read_text().splitlines()
    nat = int(lines[0].strip())
    comment = lines[1] if len(lines) > 1 else ""
    atoms = []
    coords = []
    for line in lines[2 : 2 + nat]:
        if not line.strip():
            continue
        parts = line.split()
        atoms.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return atoms, np.array(coords, dtype=float), comment


def write_xyz(path: Path, atoms, coords, comment="", extra_lines=None):
    lines = [str(len(atoms)), comment]
    for a, (x, y, z) in zip(atoms, coords):
        lines.append(f"{a} {x: .8f} {y: .8f} {z: .8f}")
    if extra_lines:
        lines.extend(extra_lines)
    path.write_text("\n".join(lines) + "\n")


def find_bond_primitive(prims, i, j):
    for idx, p in enumerate(prims):
        if p.kind != "bond":
            continue
        a, b = p.atoms
        if (a == i and b == j) or (a == j and b == i):
            return idx
    return None


def isotopic_masses_au(Z, isotopes=None, use_average=False):
    ISOTOPES = _load_isotopes_table()
    amu_to_au = 1822.888486209
    masses = []
    used = []
    average_mass_fn = None
    if use_average:
        average_mass_fn = _load_average_atomic_mass_table()
    for idx, z in enumerate(Z):
        if use_average:
            mass = average_mass_fn(int(z))
            used.append((idx + 1, int(z), None, mass))
        elif isotopes and idx in isotopes:
            A = isotopes[idx]
            iso_list = ISOTOPES.get(int(z), [])
            mass = None
            for iso in iso_list:
                if iso.A == A:
                    mass = iso.mass
                    break
            if mass is None:
                raise SystemExit(f"Isotope A={A} not found for Z={z}")
            used.append((idx + 1, int(z), A, mass))
        else:
            iso = ISOTOPES.get(int(z), [])[0]
            mass = iso.mass
            used.append((idx + 1, int(z), iso.A, mass))
        masses.append(mass * amu_to_au)
    return np.array(masses, dtype=float), used




def rotational_constants(coords_au, masses_au):
    # compute rotational constants (GHz) from inertia tensor (au)
    nat = coords_au.shape[0]
    m = np.array(masses_au, dtype=float)
    com = np.sum(coords_au * m[:, None], axis=0) / np.sum(m)
    x = coords_au - com

    I = np.zeros((3, 3))
    for i in range(nat):
        xi, yi, zi = x[i]
        mi = m[i]
        I += mi * np.array(
            [
                [yi**2 + zi**2, -xi * yi, -xi * zi],
                [-xi * yi, xi**2 + zi**2, -yi * zi],
                [-xi * zi, -yi * zi, xi**2 + yi**2],
            ]
        )

    evals = np.linalg.eigvalsh(I)
    # sort so A>=B>=C (smallest I gives largest B)
    evals = np.sort(evals)
    me = 9.1093837015e-31
    bohr = 5.29177210903e-11
    I_SI = evals * me * bohr**2
    h = 6.62607015e-34
    B_hz = h / (8.0 * np.pi**2 * I_SI)
    B_ghz = B_hz * 1.0e-9
    # return sorted A,B,C (largest to smallest)
    return np.sort(B_ghz)[::-1]


def parse_isotopes(arg):
    if not arg:
        return {}
    # format: "1:2,3:13" (atom index : A)
    isotopes = {}
    for item in arg.split(","):
        if not item:
            continue
        i_str, a_str = item.split(":")
        isotopes[int(i_str) - 1] = int(a_str)
    return isotopes


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






def _backtransform_iterative(
    s_target,
    coords0,
    prims,
    masses=None,
    max_iter=20,
    tol=1e-6,
    mass_weighted=False,
    damping=1.0,
    adaptive=False,
    metric_weights=None,
    weights=None,
):
    coords = coords0.copy()
    prev_norm = None
    if metric_weights is not None and weights is not None:
        raise ValueError("Use metric_weights or legacy weights, not both.")
    if metric_weights is not None:
        mw = np.array(metric_weights, dtype=float).reshape(-1)
        if mw.size != len(prims):
            raise ValueError("metric_weights size does not match primitives")
        if np.any(mw < 0.0):
            raise ValueError("metric_weights must be non-negative")
        w = np.sqrt(mw)
    elif weights is not None:
        w = np.array(weights, dtype=float).reshape(-1)
        if w.size != len(prims):
            raise ValueError("weights size does not match primitives")
    else:
        w = None
    for _ in range(max_iter):
        s = eval_primitives(prims, coords)
        ds = s_target - s
        norm = np.linalg.norm(ds)
        if norm < tol:
            break
        B = b_matrix(prims, coords, fd_step=1e-4)
        if w is not None:
            B = B * w[:, None]
            ds = ds * w
        G1B = compute_fortran_update_matrix(B, masses if mass_weighted else None, tol=1e-5)
        dx = G1B @ ds
        step = damping
        if adaptive and prev_norm is not None and norm > prev_norm * 1.05:
            step = 1.0
            for _ in range(4):
                trial = coords + (dx * step).reshape(coords.shape)
                s_trial = eval_primitives(prims, trial)
                norm_trial = np.linalg.norm(s_target - s_trial)
                if norm_trial < norm:
                    coords = trial
                    norm = norm_trial
                    break
                step *= 0.5
            else:
                coords = coords + (dx * step).reshape(coords.shape)
        else:
            if step < 1.0:
                dx = dx * step
            coords += dx.reshape(coords.shape)
        prev_norm = norm
    return coords


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xyz", required=False)
    ap.add_argument("--out", required=False)
    ap.add_argument("--xyz-units", choices=["ang", "au"], default=None)
    ap.add_argument("--out-units", choices=["ang", "au"], default=None)
    ap.add_argument(
        "--bond",
        action="append",
        nargs=3,
        metavar=("I", "J", "R"),
        help="Set bond length between atoms I J to R (units per --xyz-units).",
    )
    ap.add_argument(
        "--rule",
        choices=["bdpcs3"],
        default=None,
        help="Apply automatic bond-length rule to all detected bonds.",
    )
    ap.add_argument(
        "--bdpcs3-fit",
        action="store_true",
        help="Apply fitted per-pair BDPCS3 scale factors if present.",
    )
    ap.add_argument(
        "--bdpcs3-version",
        choices=BDPCS3_VERSIONS,
        default="updated",
        help="Select BDPCS3 formulation when --rule bdpcs3 is enabled.",
    )
    ap.add_argument(
        "--isotopes", default=None, help="Isotope map: i:A (1-based atom index). Example: 1:2,3:13"
    )
    ap.add_argument(
        "--report", action="store_true", help="Print bond report and rotational constants."
    )
    ap.add_argument(
        "--average-masses",
        action="store_true",
        help="Use FilAMS average atomic masses instead of principal isotopes.",
    )
    ap.add_argument("--max-iter", type=int, default=50)
    ap.add_argument("--tol", type=float, default=1e-8)
    args = ap.parse_args()

    if not args.xyz:
        args.xyz = input("XYZ file path: ").strip()
    if not args.out:
        default_out = "out.xyz"
        out_in = input(f"Output XYZ path [{default_out}]: ").strip()
        args.out = out_in or default_out
    if not args.xyz_units:
        u = input("Units for input XYZ (ang/au) [ang]: ").strip().lower()
        args.xyz_units = u or "ang"
    if not args.out_units:
        u = input("Units for output XYZ (ang/au) [ang]: ").strip().lower()
        args.out_units = u or "ang"
    if not args.rule:
        r = input("Apply BDPCS3 rule to all bonds? [Y/n]: ").strip().lower()
        args.rule = "bdpcs3" if r in ("", "y", "yes") else None
    if not args.report:
        r = input("Print report (bond table + rotational constants)? [Y/n]: ").strip().lower()
        args.report = True if r in ("", "y", "yes") else False
    if args.isotopes is None and args.report:
        iso = input("Isotope map (e.g. 1:2,3:13) or blank for defaults: ").strip()
        args.isotopes = iso or None

    atoms, coords, comment = read_xyz(Path(args.xyz))
    if args.xyz_units == "ang":
        coords_au = coords * ANG_TO_BOHR
        coords_ang_for_bo = coords
    else:
        coords_au = coords.copy()
        coords_ang_for_bo = coords * BOHR_TO_ANG

    atomic_number = _load_topology_elements()
    Z = np.array([atomic_number(a) for a in atoms], dtype=int)

    iso_map = parse_isotopes(args.isotopes)
    masses, used_isotopes = isotopic_masses_au(Z, iso_map, use_average=args.average_masses)

    prims_base = primitives_from_topology(coords_au, Z, linear_threshold=np.deg2rad(170.0))
    prims_bond = [p for p in prims_base if p.kind == "bond"]
    covalent_pairs = {tuple(sorted(p.atoms)) for p in prims_bond}
    detected_hbonds = detect_hydrogen_bonds(Z, coords_ang_for_bo, covalent_pairs)
    hbonds = []
    hbond_prims = []
    for hb in detected_hbonds:
        pair = tuple(sorted((hb.hydrogen, hb.acceptor)))
        if pair in covalent_pairs:
            continue
        hbonds.append(hb)
        hbond_prims.append(Primitive("bond", (hb.hydrogen, hb.acceptor)))
    prims_all = prims_base + hbond_prims
    hbond_indices = list(range(len(prims_base), len(prims_all)))
    hbond_pairs = {
        tuple(sorted((hb.hydrogen, hb.acceptor)))
        for hb in hbonds
        if tuple(sorted((hb.hydrogen, hb.acceptor))) not in covalent_pairs
    }
    s = eval_primitives(prims_all, coords_au)
    bond_index_by_pair = {
        tuple(sorted(p.atoms)): idx
        for idx, p in enumerate(prims_all)
        if p.kind == "bond" and idx < len(prims_base)
    }

    report_rows = []
    if args.rule == "bdpcs3":
        if not args.bdpcs3_fit:
            _disable_bdpcs3_fit()
        bdpcs3_fn = bdpcs3_function(args.bdpcs3_version)
        bo_cache = {}
        for idx, p in enumerate(prims_all[: len(prims_base)]):
            if p.kind != "bond":
                continue
            i, j = p.atoms
            r_ang = s[idx] * BOHR_TO_ANG
            bndord_topo = topology_bond_order_for_pair(i, j, Z, coords_ang_for_bo, cache=bo_cache)
            dlt_r, bndord = bdpcs3_fn(int(Z[i]), int(Z[j]), r_ang, bond_order_override=bndord_topo)
            r_corr = r_ang + dlt_r
            report_rows.append((i + 1, j + 1, r_ang, r_corr, bndord))
            s[idx] = r_corr * ANG_TO_BOHR
        for idx, hb in zip(hbond_indices, hbonds):
            r_ang = s[idx] * BOHR_TO_ANG
            dlt_r = bdpcs3_hbond_delta(int(Z[hb.donor]), int(Z[hb.acceptor]), r_ang, hb.angle_deg)
            s[idx] = (r_ang + dlt_r) * ANG_TO_BOHR
            report_rows.append((hb.hydrogen + 1, hb.acceptor + 1, r_ang, r_ang + dlt_r, 0.0))

    if args.bond:
        for i_str, j_str, r_str in args.bond:
            i = int(i_str) - 1
            j = int(j_str) - 1
            r = float(r_str)
            if args.xyz_units == "ang":
                r *= ANG_TO_BOHR
            idx = bond_index_by_pair.get(tuple(sorted((i, j))))
            if idx is None:
                raise SystemExit(f"Bond primitive not found for {i + 1}-{j + 1}")
            r_old = s[idx] * BOHR_TO_ANG
            s[idx] = r
            r_new = r * BOHR_TO_ANG
            report_rows.append((i + 1, j + 1, r_old, r_new, 0.0))

    # Weighted internal-coordinate back-transform.
    metric_weights = bdpcs3_metric_weights(
        prims_all,
        Z,
        coords_ang_for_bo,
        hbond_pairs=hbond_pairs,
        profile=BDPCS3_DEFAULT_WEIGHT_PROFILE,
    )
    coords_new = _backtransform_iterative(
        s,
        coords_au,
        prims_all,
        masses=masses,
        max_iter=args.max_iter,
        tol=args.tol,
        adaptive=True,
        metric_weights=metric_weights,
    )

    if args.out_units == "ang":
        coords_out = coords_new * BOHR_TO_ANG
    else:
        coords_out = coords_new

    extra_lines = []
    report_rows_final = []
    if args.report:
        s_new = eval_primitives(prims_bond, coords_new)
        for i, j, r0, r1, bndord in report_rows:
            idx = find_bond_primitive(prims_bond, i - 1, j - 1)
            if idx is not None:
                r_final = s_new[idx] * BOHR_TO_ANG
            else:
                r_final = np.linalg.norm(coords_new[i - 1] - coords_new[j - 1]) * BOHR_TO_ANG
            err_pct = 0.0 if r1 == 0.0 else abs(r_final - r1) / r1 * 100.0
            report_rows_final.append((i, j, r0, r1, err_pct, bndord))

        B0 = rotational_constants(coords_au, masses)
        B1 = rotational_constants(coords_new, masses)
        B0_mhz = B0 * 1000.0
        B1_mhz = B1 * 1000.0

        extra_lines.append("Bond distances")
        extra_lines.append(
            "(i, j) | Old dist | New dist | Convergence Err (%) | Original Bond Order"
        )
        extra_lines.append("-" * 59)
        for i, j, r0, r1, err_pct, bndord in report_rows_final:
            extra_lines.append(f"({i}, {j}) | {r0:.6f} | {r1:.6f} | {err_pct:.6f} | {bndord:.3f}")
        extra_lines.append("")
        extra_lines.append("Parent rotational constants")
        extra_lines.append("Asymmetric top, nearly prolate")
        extra_lines.append("")
        extra_lines.append("MHz                 | A             | B            | C")
        extra_lines.append("-" * 65)
        extra_lines.append(
            f"Before correcting   | {B0_mhz[0]:.10f} | {B0_mhz[1]:.10f} | {B0_mhz[2]:.10f}"
        )
        extra_lines.append(
            f"After correcting    | {B1_mhz[0]:.10f} | {B1_mhz[1]:.10f} | {B1_mhz[2]:.10f}"
        )
        extra_lines.append("-" * 65)

    write_xyz(Path(args.out), atoms, coords_out, comment=comment, extra_lines=extra_lines)

    if args.report:
        B0 = rotational_constants(coords_au, masses)
        B1 = rotational_constants(coords_new, masses)
        print("\nIsotopes used (atom, Z, A, mass_amu):")
        for idx, z, a, mamu in used_isotopes:
            a_str = f"{a:3d}" if a is not None else "avg"
            print(f"{idx:3d}  Z={z:3d}  A={a_str}  mass={mamu:.6f}")
        print("\nBond report (Angstrom):")
        print(" i   j     r_old       r_new")
        for i, j, r0, r1, _ in report_rows:
            print(f"{i:2d} {j:2d}  {r0:9.5f}  {r1:9.5f}")
        print("\nRotational constants (GHz):")
        print(f"Start: A={B0[0]:.6f}  B={B0[1]:.6f}  C={B0[2]:.6f}")
        print(f"Final: A={B1[0]:.6f}  B={B1[1]:.6f}  C={B1[2]:.6f}")


if __name__ == "__main__":
    main()
