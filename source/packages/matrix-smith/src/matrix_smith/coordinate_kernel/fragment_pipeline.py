from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import dihedral
from .modify_geom import read_xyz
from .pipeline import _load_topology_elements, build_topology_full


@dataclass
class Fragment:
    source_xyz: str
    source_library: str
    molecule_key: str
    fragment_id: str
    atom_indices: list[int]
    feature_vector: np.ndarray
    fragment_kind: str
    core_size: int
    ring_size: int
    ring_coords_norm: np.ndarray
    hetero_frac: float
    aromatic_like_frac: float
    elem_profile: np.ndarray
    radial_pos: float
    local_type: str
    torsion_vec: np.ndarray
    torsion_available: bool


def _xyz_to_au(path: Path):
    atoms, coords_ang, _ = read_xyz(path)
    atomic_number = _load_topology_elements()
    Z = np.array([atomic_number(a) for a in atoms], dtype=int)
    coords_au = np.array(coords_ang, dtype=float) / 0.52917721092
    return atoms, Z, coords_au


def _bo_class(bo: float) -> str:
    if bo >= 2.4:
        return "T"
    if bo >= 1.7:
        return "D"
    if bo >= 1.25:
        return "A"
    return "S"


def _bond_torsion_vec(
    core_atoms: list[int], dg, coords: np.ndarray, Z: np.ndarray
) -> tuple[np.ndarray, bool]:
    if len(core_atoms) != 2:
        return np.zeros(2, dtype=float), False
    a, b = int(core_atoms[0]), int(core_atoms[1])
    na = [int(x) for x in dg.neighbors(a) if int(x) != b]
    nb = [int(x) for x in dg.neighbors(b) if int(x) != a]
    if not na or not nb:
        return np.zeros(2, dtype=float), False
    vec = np.zeros(2, dtype=float)
    wsum = 0.0
    for i in na:
        for l in nb:
            w = (2.0 if int(Z[i]) > 1 else 1.0) * (2.0 if int(Z[l]) > 1 else 1.0)
            phi = float(dihedral(i, a, b, l, coords))
            vec += w * np.array([np.cos(phi), np.sin(phi)], dtype=float)
            wsum += w
    if wsum < 1.0e-12:
        return np.zeros(2, dtype=float), False
    vec /= wsum
    nrm = float(np.linalg.norm(vec))
    if nrm < 1.0e-12:
        return np.zeros(2, dtype=float), False
    return vec / nrm, True


def _build_fragment(
    path: Path,
    source_library: str,
    fragment_id: str,
    fragment_kind: str,
    core_atoms: list[int],
    frag_atoms: list[int],
    coords: np.ndarray,
    Z: np.ndarray,
    synthons,
    dg,
    mol_center: np.ndarray,
    mol_scale: float,
    *,
    planarity: float,
    fused_degree: float,
) -> Fragment:
    syn = np.zeros(5, dtype=float)
    for a in frag_atoms:
        syn[0] += float(synthons.charge(a))
        syn[1] += float(synthons.covalency(a))
        syn[2] += float(synthons.delocalization(a))
        syn[3] += float(synthons.strain(a))
        syn[4] += float(synthons.Zeff(a))
    syn /= max(len(frag_atoms), 1)

    frag_Z = Z[frag_atoms]
    n_frag = float(len(frag_atoms))
    n_heavy = float(np.count_nonzero(frag_Z > 1))
    n_hetero = float(np.count_nonzero((frag_Z != 1) & (frag_Z != 6)))
    n_c = float(np.count_nonzero(frag_Z == 6))
    n_n = float(np.count_nonzero(frag_Z == 7))
    n_o = float(np.count_nonzero(frag_Z == 8))
    n_s = float(np.count_nonzero(frag_Z == 16))
    n_p = float(np.count_nonzero(frag_Z == 15))
    n_hal = float(np.count_nonzero(np.isin(frag_Z, [9, 17, 35, 53])))

    elem = np.array(
        [
            n_heavy / max(n_frag, 1.0),
            n_hetero / max(n_frag, 1.0),
            n_n / max(n_frag, 1.0),
            n_o / max(n_frag, 1.0),
            n_s / max(n_frag, 1.0),
            n_p / max(n_frag, 1.0),
            n_hal / max(n_frag, 1.0),
            float(np.mean(frag_Z) / 20.0),
            float(np.std(frag_Z) / 20.0),
        ],
        dtype=float,
    )
    elem_profile = np.array(
        [
            n_c / max(n_frag, 1.0),
            n_n / max(n_frag, 1.0),
            n_o / max(n_frag, 1.0),
            n_s / max(n_frag, 1.0),
            n_p / max(n_frag, 1.0),
            n_hal / max(n_frag, 1.0),
        ],
        dtype=float,
    )

    bo_vals = []
    for i in range(len(frag_atoms)):
        ai = frag_atoms[i]
        for j in range(i + 1, len(frag_atoms)):
            aj = frag_atoms[j]
            if float(synthons.bond_order(ai, aj)) > 0.1:
                bo_vals.append(
                    (
                        float(synthons.bond_order(ai, aj)),
                        float(synthons.bond_order_total_pi(ai, aj)),
                    )
                )
    if bo_vals:
        bo_arr = np.array([item[0] for item in bo_vals], dtype=float)
        bond_desc = np.array(
            [
                float(np.mean(bo_arr)),
                float(np.max(bo_arr)),
                float(np.count_nonzero([item[1] >= 0.4 for item in bo_vals]) / bo_arr.size),
            ],
            dtype=float,
        )
    else:
        bond_desc = np.zeros(3, dtype=float)

    core_xyz = coords[core_atoms]
    ctr = core_xyz.mean(axis=0)
    core_size = float(len(core_atoms))
    radial_pos = float(np.linalg.norm(ctr - mol_center) / mol_scale)
    core_radius = float(np.sqrt(((core_xyz - ctr) ** 2).sum(axis=1).mean()) / mol_scale)
    frag_size = float(len(frag_atoms))
    vec = np.concatenate(
        [
            syn,
            elem,
            bond_desc,
            np.array(
                [
                    core_size,
                    float(planarity),
                    float(fused_degree),
                    radial_pos,
                    core_radius,
                    frag_size,
                ],
                dtype=float,
            ),
        ]
    )
    core_xyz_centered = core_xyz - ctr
    core_scale = float(np.sqrt((core_xyz_centered**2).sum(axis=1).mean()))
    if core_scale < 1.0e-12:
        core_scale = 1.0
    core_coords_norm = np.array(core_xyz_centered / core_scale, dtype=float)
    if fragment_kind == "ring":
        local_type = (
            f"ring:s{int(core_size)}:h{int(n_hetero > 0)}:a{int(float(bond_desc[2]) > 0.3)}"
        )
        torsion_vec = np.zeros(2, dtype=float)
        torsion_available = False
    else:
        z1 = int(Z[int(core_atoms[0])])
        z2 = int(Z[int(core_atoms[1])])
        bo_core = float(
            synthons.bond_order(int(core_atoms[0]), int(core_atoms[1]))
        )
        local_type = f"bond:{min(z1, z2)}-{max(z1, z2)}:{_bo_class(bo_core)}:h{int(n_hetero > 0)}"
        torsion_vec, torsion_available = _bond_torsion_vec(core_atoms, dg, coords, Z)

    return Fragment(
        source_xyz=str(path),
        source_library=source_library,
        molecule_key=path.stem.lower(),
        fragment_id=fragment_id,
        atom_indices=frag_atoms,
        feature_vector=vec,
        fragment_kind=fragment_kind,
        core_size=int(core_size),
        ring_size=int(core_size if fragment_kind == "ring" else 0),
        ring_coords_norm=core_coords_norm,
        hetero_frac=float(n_hetero / max(n_frag, 1.0)),
        aromatic_like_frac=float(bond_desc[2]),
        elem_profile=elem_profile,
        radial_pos=float(radial_pos),
        local_type=local_type,
        torsion_vec=torsion_vec,
        torsion_available=bool(torsion_available),
    )


def _fragment_features(path: Path, source_library: str) -> list[Fragment]:
    atoms, Z, coords_au = _xyz_to_au(path)
    _cg, dg, ringset, synthons, _arom = build_topology_full(coords_au, Z)

    coords = np.array(coords_au, dtype=float)
    mol_center = coords.mean(axis=0)
    mol_scale = float(np.sqrt(((coords - mol_center) ** 2).sum(axis=1).mean()))
    if mol_scale < 1.0e-12:
        mol_scale = 1.0

    out = []
    ring_atoms_union = set()
    if ringset is not None and len(ringset) > 0:
        for idx, ring in enumerate(ringset.rings):
            ring_atoms = sorted(set(int(a) for a in ring.atoms))
            ring_atoms_union.update(ring_atoms)
            neigh = set(ring_atoms)
            for a in ring_atoms:
                for b in range(len(atoms)):
                    if b == a:
                        continue
                    bo = synthons.bond_order(a, b)
                    if bo > 0.3:
                        neigh.add(int(b))
            frag_atoms = sorted(neigh)
            out.append(
                _build_fragment(
                    path,
                    source_library,
                    f"{path.stem}:ring_{idx + 1}",
                    "ring",
                    ring_atoms,
                    frag_atoms,
                    coords,
                    Z,
                    synthons,
                    dg,
                    mol_center,
                    mol_scale,
                    planarity=float(ring.planarity() or 0.0),
                    fused_degree=float(len(ring.connected_rings)),
                )
            )

    bond_candidates = []
    for i in range(dg.natoms):
        for j in dg.neighbors(i):
            if j <= i:
                continue
            if int(i) in ring_atoms_union and int(j) in ring_atoms_union:
                continue
            bo = float(synthons.bond_order(int(i), int(j)))
            if bo < 0.2:
                continue
            zi = int(Z[int(i)])
            zj = int(Z[int(j)])
            hetero = int((zi != 6 and zi != 1) or (zj != 6 and zj != 1))
            bond_candidates.append((hetero, bo, int(i), int(j)))

    bond_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    max_bond_frags_per_mol = 16
    for _hetero, _bo, i, j in bond_candidates[:max_bond_frags_per_mol]:
        core_atoms = [int(i), int(j)]
        neigh = set(core_atoms)
        for a in core_atoms:
            for b in dg.neighbors(a):
                neigh.add(int(b))
        frag_atoms = sorted(neigh)
        out.append(
            _build_fragment(
                path,
                source_library,
                f"{path.stem}:bond_{int(i) + 1}_{int(j) + 1}",
                "bond",
                core_atoms,
                frag_atoms,
                coords,
                Z,
                synthons,
                dg,
                mol_center,
                mol_scale,
                planarity=0.0,
                fused_degree=0.0,
            )
        )
    return out


def _sim(a: np.ndarray, b: np.ndarray) -> float:
    # Stable bounded similarity in (0,1].
    d = float(np.linalg.norm(a - b))
    dim = max(int(a.size), 1)
    return float(np.exp(-(d / np.sqrt(dim))))


def _standardize_feature_vectors(
    library_frags: list[Fragment],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not library_frags:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)
    x = np.vstack([f.feature_vector for f in library_frags])
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    sigma = np.where(sigma < 1.0e-8, 1.0, sigma)
    # Reliability for uncertainty-weighted similarity: high-variance features contribute less.
    rel = 1.0 / (1.0 + sigma)
    rel = rel / max(float(np.mean(rel)), 1.0e-12)
    return mu, sigma, rel


def _kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape or a.shape[0] == 0:
        return 10.0
    h = a.T @ b
    u, _s, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0.0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    a_rot = a @ r
    diff = a_rot - b
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _ring_shape_similarity(query: Fragment, cand: Fragment) -> tuple[float, float]:
    if query.fragment_kind != cand.fragment_kind:
        return 0.0, 10.0

    if query.core_size != cand.core_size:
        gap = float(abs(query.core_size - cand.core_size))
        sim = float(np.exp(-1.5 * gap))
        return sim, 10.0

    q = query.ring_coords_norm
    c = cand.ring_coords_norm
    n = q.shape[0]
    if n == 0:
        return 0.0, 10.0

    best = None
    if query.fragment_kind == "ring":
        for shift in range(n):
            c_shift = np.roll(c, shift=shift, axis=0)
            rmsd_fwd = _kabsch_rmsd(q, c_shift)
            rmsd_rev = _kabsch_rmsd(q, c_shift[::-1])
            rmsd = min(rmsd_fwd, rmsd_rev)
            if best is None or rmsd < best:
                best = rmsd
    else:
        best = _kabsch_rmsd(q, c)
    assert best is not None
    sim = float(np.exp(-1.5 * best))
    return sim, float(best)


def _chemical_compatibility(query: Fragment, cand: Fragment) -> float:
    diff_hetero = abs(query.hetero_frac - cand.hetero_frac)
    diff_arom = abs(query.aromatic_like_frac - cand.aromatic_like_frac)
    if query.elem_profile.shape == cand.elem_profile.shape and query.elem_profile.size:
        diff_elem = float(
            np.linalg.norm(query.elem_profile - cand.elem_profile)
            / np.sqrt(query.elem_profile.size)
        )
    else:
        diff_elem = 1.0

    if query.hetero_frac >= 0.20 and cand.hetero_frac <= 0.05:
        return 0.0
    if query.hetero_frac <= 0.05 and cand.hetero_frac >= 0.30:
        return 0.0

    return float(np.exp(-(1.8 * diff_hetero + 1.2 * diff_elem + 0.8 * diff_arom)))


def _local_type_similarity(query: Fragment, cand: Fragment) -> float:
    if query.local_type == cand.local_type:
        return 1.0
    q = query.local_type.split(":")
    c = cand.local_type.split(":")
    if not q or not c or q[0] != c[0]:
        return 0.0
    if q[0] == "ring":
        # Same kind with different tags still partially compatible.
        return 0.55
    # bond: compare element pair / bond class / hetero tag
    score = 0.0
    if len(q) > 1 and len(c) > 1 and q[1] == c[1]:
        score += 0.45
    if len(q) > 2 and len(c) > 2 and q[2] == c[2]:
        score += 0.35
    if len(q) > 3 and len(c) > 3 and q[3] == c[3]:
        score += 0.20
    return float(score)


def _calibrated_reliability(sim: float, kind: str) -> float:
    # Simple logistic calibration (placeholder for data-driven fit).
    if kind == "bond":
        k, x0 = 9.0, 0.52
    else:
        k, x0 = 7.5, 0.50
    return float(1.0 / (1.0 + np.exp(-k * (float(sim) - x0))))


def _rank_fragment(
    query_frag: Fragment,
    library_frags: list[Fragment],
    top_k: int,
    *,
    feat_mu: np.ndarray,
    feat_sigma: np.ndarray,
    feat_reliability: np.ndarray,
    max_ring_size_delta: int,
    chemical_min_score: float,
):
    rows = []
    for cand in library_frags:
        if cand.source_xyz == query_frag.source_xyz and cand.fragment_id == query_frag.fragment_id:
            continue
        if query_frag.fragment_kind != cand.fragment_kind:
            continue
        core_size_delta = int(abs(query_frag.core_size - cand.core_size))
        if core_size_delta > int(max_ring_size_delta):
            continue

        if feat_mu.size and feat_sigma.size:
            qf = (query_frag.feature_vector - feat_mu) / feat_sigma
            cf = (cand.feature_vector - feat_mu) / feat_sigma
        else:
            qf = query_frag.feature_vector
            cf = cand.feature_vector
        if feat_reliability.size:
            diff = qf - cf
            d = float(np.sqrt(np.mean(feat_reliability * diff * diff)))
            s_feat = float(np.exp(-d))
        else:
            s_feat = _sim(qf, cf)
        s_3d, rmsd_3d = _ring_shape_similarity(query_frag, cand)
        chem_comp = _chemical_compatibility(query_frag, cand)
        local_sim = _local_type_similarity(query_frag, cand)
        if chem_comp < float(chemical_min_score):
            continue
        if local_sim < 0.25:
            continue
        # Advanced chemical filter: enforce closer element-profile match and aromaticity consistency.
        diff_elem = float(
            np.linalg.norm(query_frag.elem_profile - cand.elem_profile)
            / np.sqrt(max(query_frag.elem_profile.size, 1))
        )
        if diff_elem > 0.35:
            continue
        if abs(query_frag.aromatic_like_frac - cand.aromatic_like_frac) > 0.60:
            continue
        if query_frag.fragment_kind == "bond" and local_sim < 0.55:
            continue
        torsion_sim = 1.0
        if query_frag.fragment_kind == "bond":
            if query_frag.torsion_available and cand.torsion_available:
                cphi = float(np.clip(np.dot(query_frag.torsion_vec, cand.torsion_vec), -1.0, 1.0))
                torsion_sim = 0.5 * (1.0 + cphi)
            else:
                torsion_sim = 0.7
        size_factor = float(np.exp(-0.8 * core_size_delta))
        if core_size_delta == 0:
            w_feat, w_3d = 0.50, 0.50
        elif core_size_delta == 1:
            w_feat, w_3d = 0.65, 0.35
        else:
            w_feat, w_3d = 0.85, 0.15
        s_base = float(w_feat * s_feat + w_3d * s_3d)
        s = float(s_base * size_factor * (0.55 + 0.30 * chem_comp + 0.15 * local_sim))
        if query_frag.fragment_kind == "bond":
            s *= float(0.75 + 0.25 * torsion_sim)
        reliability = _calibrated_reliability(s, query_frag.fragment_kind)
        rows.append(
            {
                "candidate_xyz": cand.source_xyz,
                "candidate_library": cand.source_library,
                "candidate_molecule_key": cand.molecule_key,
                "candidate_fragment_id": cand.fragment_id,
                "candidate_fragment_kind": cand.fragment_kind,
                "candidate_cycle_size": int(cand.ring_size),
                "candidate_fragment_label": (
                    f"{cand.fragment_id} ({cand.fragment_kind}_{int(cand.core_size)})"
                ),
                "similarity": s,
                "similarity_base": float(s_base),
                "similarity_feature": float(s_feat),
                "similarity_3d": float(s_3d),
                "ring_rmsd_3d": float(rmsd_3d),
                "chemical_compatibility": float(chem_comp),
                "local_type_similarity": float(local_sim),
                "torsion_similarity": float(torsion_sim),
                "reliability": float(reliability),
                "ring_size_delta": int(core_size_delta),
                "ring_size_factor": float(size_factor),
                "candidate_ring_size": cand.ring_size,
                "candidate_core_size": cand.core_size,
                "candidate_radial_pos": float(cand.radial_pos),
                "candidate_atom_indices": [int(x) for x in cand.atom_indices],
            }
        )
    # Prefer SE over PCS2 when the same molecule exists in both libraries.
    by_molecule = {}
    for row in rows:
        key = row["candidate_molecule_key"]
        cur = by_molecule.get(key)
        if cur is None:
            by_molecule[key] = row
            continue
        # Same molecule key: keep SE candidate if available, otherwise best score.
        if cur["candidate_library"] != "SE" and row["candidate_library"] == "SE":
            by_molecule[key] = row
            continue
        if cur["candidate_library"] == row["candidate_library"]:
            if row["similarity"] > cur["similarity"]:
                by_molecule[key] = row

    uniq_rows = list(by_molecule.values())
    uniq_rows.sort(
        key=lambda r: (r["similarity"], 1 if r["candidate_library"] == "SE" else 0),
        reverse=True,
    )
    return uniq_rows[: max(1, int(top_k))]


def _pairwise_assembly_consistency(a: dict, b: dict) -> float:
    qa = a["query_fragment"]
    qb = b["query_fragment"]
    ca = a["candidate"]
    cb = b["candidate"]
    q_rad = abs(float(qa.radial_pos) - float(qb.radial_pos))
    c_rad = abs(float(ca["candidate_radial_pos"]) - float(cb["candidate_radial_pos"]))
    s_rad = float(np.exp(-2.0 * abs(q_rad - c_rad)))
    q_ring = abs(int(qa.ring_size) - int(qb.ring_size))
    c_ring = abs(int(ca["candidate_ring_size"]) - int(cb["candidate_ring_size"]))
    s_ring = float(np.exp(-0.8 * abs(q_ring - c_ring)))
    same_key = 1.0 if ca["candidate_molecule_key"] == cb["candidate_molecule_key"] else 0.0
    same_lib = 1.0 if ca["candidate_library"] == cb["candidate_library"] else 0.0
    return float(0.45 * s_rad + 0.35 * s_ring + 0.15 * same_key + 0.05 * same_lib)


def _assemble_global_solution(
    query_frags: list[Fragment],
    report_rows: list[dict],
    per_fragment_top: int = 3,
    reuse_penalty: float = 0.08,
    max_combinations: int = 50000,
):
    bundles = []
    for qf, row in zip(query_frags, report_rows):
        cands = row.get("ranking", [])[: max(1, int(per_fragment_top))]
        if not cands:
            return {
                "available": False,
                "reason": "At least one fragment has no candidates for assembly.",
            }
        bundles.append({"query_fragment": qf, "candidates": cands})

    idx_ranges = [range(len(b["candidates"])) for b in bundles]
    ncomb = 1
    for r in idx_ranges:
        ncomb *= max(len(r), 1)
    if ncomb > int(max_combinations):
        chosen = [
            {"query_fragment": b["query_fragment"], "candidate": b["candidates"][0]}
            for b in bundles
        ]
        ind_mean = float(np.mean([x["candidate"]["similarity"] for x in chosen]))
        pair_terms = []
        for i in range(len(chosen)):
            for j in range(i + 1, len(chosen)):
                pair_terms.append(_pairwise_assembly_consistency(chosen[i], chosen[j]))
        pair_mean = float(np.mean(pair_terms)) if pair_terms else 1.0
        labels = [x["candidate"]["candidate_fragment_label"] for x in chosen]
        uniq = len(set(labels))
        total = max(len(labels), 1)
        reuse_ratio = float((total - uniq) / total)
        reuse_factor = float(max(0.0, 1.0 - float(reuse_penalty) * reuse_ratio))
        best = {
            "score": float(ind_mean * (0.75 + 0.25 * pair_mean) * reuse_factor),
            "individual_mean": ind_mean,
            "pairwise_mean": pair_mean,
            "reuse_ratio": reuse_ratio,
            "reuse_factor": reuse_factor,
            "chosen": chosen,
        }
    else:
        best = None
        for combo_idx in itertools.product(*idx_ranges):
            chosen = []
            for i, cidx in enumerate(combo_idx):
                chosen.append(
                    {
                        "query_fragment": bundles[i]["query_fragment"],
                        "candidate": bundles[i]["candidates"][cidx],
                    }
                )

            ind_mean = float(np.mean([x["candidate"]["similarity"] for x in chosen]))
            pair_terms = []
            for i in range(len(chosen)):
                for j in range(i + 1, len(chosen)):
                    pair_terms.append(_pairwise_assembly_consistency(chosen[i], chosen[j]))
            pair_mean = float(np.mean(pair_terms)) if pair_terms else 1.0
            labels = [x["candidate"]["candidate_fragment_label"] for x in chosen]
            uniq = len(set(labels))
            total = max(len(labels), 1)
            reuse_ratio = float((total - uniq) / total)
            reuse_factor = float(max(0.0, 1.0 - float(reuse_penalty) * reuse_ratio))
            score = float(ind_mean * (0.75 + 0.25 * pair_mean) * reuse_factor)
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "individual_mean": ind_mean,
                    "pairwise_mean": pair_mean,
                    "reuse_ratio": reuse_ratio,
                    "reuse_factor": reuse_factor,
                    "chosen": chosen,
                }

    assert best is not None
    assignment = []
    for x in best["chosen"]:
        c = x["candidate"]
        assignment.append(
            {
                "query_fragment_id": x["query_fragment"].fragment_id,
                "candidate_fragment_label": c["candidate_fragment_label"],
                "candidate_xyz": c["candidate_xyz"],
                "candidate_library": c["candidate_library"],
                "similarity": c["similarity"],
                "chemical_compatibility": c.get("chemical_compatibility", 0.0),
            }
        )
    return {
        "available": True,
        "score": float(best["score"]),
        "individual_mean": float(best["individual_mean"]),
        "pairwise_mean": float(best["pairwise_mean"]),
        "reuse_ratio": float(best["reuse_ratio"]),
        "reuse_factor": float(best["reuse_factor"]),
        "reuse_penalty": float(reuse_penalty),
        "per_fragment_top_used": int(per_fragment_top),
        "assignment": assignment,
    }


def _closest_library_reference(ranking: list[dict]):
    if not ranking:
        return None
    best = None
    best_score = None
    for row in ranking[: min(5, len(ranking))]:
        se_bonus = 0.10 if row.get("candidate_library") == "SE" else 0.0
        score = (
            0.50 * float(row.get("chemical_compatibility", 0.0))
            + 0.30 * float(row.get("similarity_3d", 0.0))
            + 0.20 * float(row.get("similarity", 0.0))
            + se_bonus
        )
        if best is None or score > best_score:
            best = row
            best_score = score
    assert best is not None
    return {
        "candidate_fragment_label": best.get("candidate_fragment_label"),
        "candidate_xyz": best.get("candidate_xyz"),
        "candidate_atom_indices": best.get("candidate_atom_indices", []),
        "candidate_library": best.get("candidate_library"),
    }


def _propose_new_fragment_from_query(qf: Fragment, ranking: list[dict], *, library_label: str):
    ref = _closest_library_reference(ranking)
    return {
        "proposal_type": "new_fragment_from_query_subgraph",
        "query_fragment_id": qf.fragment_id,
        "query_atom_indices": [int(x) for x in qf.atom_indices],
        "query_fragment_kind": qf.fragment_kind,
        "query_core_size": int(qf.core_size),
        "reason": (
            "No sufficiently similar fragment in library: run this target-derived fragment at high level "
            f"and add it to {library_label}."
        ),
        "closest_library_reference": ref,
    }


def build_high_level_curation_queue(report_rows: list[dict]):
    queue = []
    for r in report_rows:
        if not r.get("is_low_score"):
            continue
        s = r.get("suggested_new_fragment") or {}
        ref = s.get("closest_library_reference") or {}
        queue.append(
            {
                "query_fragment_id": r.get("query_fragment_id"),
                "query_xyz": r.get("query_xyz"),
                "query_fragment_kind": s.get("query_fragment_kind", r.get("query_fragment_kind")),
                "query_core_size": s.get("query_core_size", r.get("query_core_size")),
                "query_atom_indices": s.get("query_atom_indices", r.get("query_atom_indices", [])),
                "query_local_type": r.get("query_local_type"),
                "query_elem_profile": r.get("query_elem_profile", []),
                "query_torsion_vec": r.get("query_torsion_vec", []),
                "query_torsion_available": r.get("query_torsion_available", False),
                "best_similarity": r.get("best_similarity"),
                "best_reliability": r.get("best_reliability"),
                "closest_library_reference": {
                    "candidate_fragment_label": ref.get("candidate_fragment_label"),
                    "candidate_xyz": ref.get("candidate_xyz"),
                    "candidate_library": ref.get("candidate_library"),
                },
                "action": "Run this query-derived fragment at high level and add to library.",
            }
        )
    # Deduplicate near-identical suggestions.
    merged = []
    for item in queue:
        merged_into = False
        for m in merged:
            if item.get("query_fragment_kind") != m.get("query_fragment_kind"):
                continue
            if int(item.get("query_core_size", -1)) != int(m.get("query_core_size", -2)):
                continue
            if item.get("query_local_type") and m.get("query_local_type"):
                if item.get("query_local_type") != m.get("query_local_type"):
                    continue
            v1 = np.array(item.get("query_elem_profile", []), dtype=float)
            v2 = np.array(m.get("query_elem_profile", []), dtype=float)
            if v1.size and v2.size:
                if float(np.linalg.norm(v1 - v2)) > 0.10:
                    continue
            if item.get("query_fragment_kind") == "bond":
                if item.get("query_torsion_available") and m.get("query_torsion_available"):
                    t1 = np.array(item.get("query_torsion_vec", []), dtype=float)
                    t2 = np.array(m.get("query_torsion_vec", []), dtype=float)
                    if t1.size == 2 and t2.size == 2:
                        if float(np.dot(t1, t2)) < 0.90:
                            continue
            # Merge: keep the lower similarity (more urgent).
            if float(item.get("best_similarity", 0.0)) < float(m.get("best_similarity", 0.0)):
                m.update(item)
            merged_into = True
            break
        if not merged_into:
            merged.append(item)
    return merged


def run_fragment_pipeline(
    xyz: Path,
    se_dir: Path,
    pcs2_dir: Path,
    out_dir: Path,
    *,
    xyz_glob: str = "*.xyz",
    top_k: int = 5,
    gap_threshold: float = 0.75,
    max_ring_size_delta: int = 2,
    chemical_min_score: float = 0.15,
    reuse_penalty: float = 0.04,
    low_score_threshold: float = 0.45,
    use_pcs2_only: bool = False,
    library_label: str = "SE/PCS2",
):
    out_dir.mkdir(parents=True, exist_ok=True)
    q_frags = _fragment_features(xyz, source_library="QUERY")

    lib_xyz = (
        sorted(pcs2_dir.glob(xyz_glob))
        if use_pcs2_only
        else sorted(se_dir.glob(xyz_glob)) + sorted(pcs2_dir.glob(xyz_glob))
    )
    lib_frags = []
    if not use_pcs2_only:
        for p in sorted(se_dir.glob(xyz_glob)):
            try:
                lib_frags.extend(_fragment_features(p, source_library="SE"))
            except Exception:
                continue
    for p in sorted(pcs2_dir.glob(xyz_glob)):
        try:
            lib_frags.extend(_fragment_features(p, source_library="PCS2"))
        except Exception:
            continue

    feat_mu, feat_sigma, feat_reliability = _standardize_feature_vectors(lib_frags)

    report_rows = []
    to_curate = []
    for qf in q_frags:
        ranking = _rank_fragment(
            qf,
            lib_frags,
            top_k=top_k,
            feat_mu=feat_mu,
            feat_sigma=feat_sigma,
            feat_reliability=feat_reliability,
            max_ring_size_delta=max_ring_size_delta,
            chemical_min_score=chemical_min_score,
        )
        best = ranking[0] if ranking else None
        best_sim = float(best["similarity"]) if best is not None else 0.0
        status = "OK" if best_sim >= gap_threshold else "GAP"
        low_score = bool(best is not None and best_sim < float(low_score_threshold))
        new_frag_suggestion = (
            _propose_new_fragment_from_query(qf, ranking, library_label=library_label)
            if low_score
            else None
        )
        row = {
            "query_fragment_id": qf.fragment_id,
            "query_xyz": qf.source_xyz,
            "query_fragment_kind": qf.fragment_kind,
            "query_core_size": qf.core_size,
            "query_ring_size": qf.ring_size,
            "query_atom_indices": [int(x) for x in qf.atom_indices],
            "query_local_type": qf.local_type,
            "query_elem_profile": [float(x) for x in qf.elem_profile],
            "query_torsion_vec": [float(x) for x in qf.torsion_vec],
            "query_torsion_available": bool(qf.torsion_available),
            "best_similarity": best_sim,
            "best_reliability": float(best.get("reliability", 0.0)) if best is not None else 0.0,
            "status": status,
            "is_low_score": low_score,
            "suggested_new_fragment": new_frag_suggestion,
            "ranking": ranking,
        }
        report_rows.append(row)
        if status == "GAP":
            to_curate.append(
                {
                    "query_fragment_id": qf.fragment_id,
                    "query_xyz": qf.source_xyz,
                    "best_similarity": best_sim,
                    "best_candidate": best,
                    "action": f"Add a closer high-accuracy fragment to {library_label} library.",
                }
            )

    assembly = _assemble_global_solution(
        q_frags,
        report_rows,
        per_fragment_top=min(3, max(1, int(top_k))),
        reuse_penalty=float(reuse_penalty),
    )
    curation_queue = build_high_level_curation_queue(report_rows)

    report = {
        "query_xyz": str(xyz),
        "query_fragments": len(q_frags),
        "library_xyz_count": len(lib_xyz),
        "library_fragments": len(lib_frags),
        "library_mode": "PCS2-only" if use_pcs2_only else "SE+PCS2",
        "library_label": library_label,
        "gap_threshold": float(gap_threshold),
        "max_ring_size_delta": int(max_ring_size_delta),
        "chemical_min_score": float(chemical_min_score),
        "reuse_penalty": float(reuse_penalty),
        "low_score_threshold": float(low_score_threshold),
        "low_score_fragments": int(sum(1 for r in report_rows if r.get("is_low_score"))),
        "high_level_curation_queue": curation_queue,
        "global_assembly": assembly,
        "fragments": report_rows,
        "to_curate": to_curate,
    }

    json_path = out_dir / "fragment_pipeline.json"
    queue_path = out_dir / "high_level_curation_queue.json"
    sr_path = out_dir / "score_reliability.csv"
    gp_path = out_dir / "score_reliability.gnuplot"
    curate_path = out_dir / "to_curate.json"
    md_path = out_dir / "fragment_pipeline.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    queue_path.write_text(json.dumps(curation_queue, indent=2) + "\n", encoding="utf-8")
    curate_path.write_text(json.dumps(to_curate, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# Fragment Pipeline Report: {xyz.name}",
        "",
        f"- Query fragments: `{len(q_frags)}`",
        f"- Library XYZ count: `{len(lib_xyz)}`",
        f"- Library fragments: `{len(lib_frags)}`",
        f"- Library mode: `{'PCS2-only' if use_pcs2_only else 'SE+PCS2'}`",
        f"- Gap threshold: `{gap_threshold:.2f}`",
        f"- Max ring-size delta: `{int(max_ring_size_delta)}`",
        f"- Chemical min score: `{chemical_min_score:.2f}`",
        f"- Reuse penalty: `{reuse_penalty:.2f}`",
        f"- Low-score threshold: `{low_score_threshold:.2f}`",
        f"- Low-score fragments: `{sum(1 for r in report_rows if r.get('is_low_score'))}`",
        "",
        "## Fragment Matches",
        "",
        "| Fragment | Kind | Core size | Best match | Best similarity | Status |",
        "|---|---|---:|---|---:|---|",
    ]
    for r in report_rows:
        best = r["ranking"][0]["candidate_fragment_label"] if r["ranking"] else "-"
        lines.append(
            f"| `{r['query_fragment_id']}` | {r.get('query_fragment_kind', '-')} | {r.get('query_core_size', r['query_ring_size'])} | "
            f"`{best}` | {r['best_similarity']:.6f} | {r['status']} |"
        )
    lines.append("")
    lines.append(f"GAP fragments: `{len(to_curate)}`")
    low_rows = [r for r in report_rows if r.get("is_low_score")]
    lines.append("")
    lines.append("## Suggested High-Level Additions")
    if low_rows:
        lines.append("| Query fragment | Best sim | Suggested candidate | Library |")
        lines.append("|---|---:|---|---|")
        for r in low_rows:
            s = r.get("suggested_new_fragment") or {}
            ref = s.get("closest_library_reference") or {}
            lines.append(
                f"| `{r['query_fragment_id']}` | {r['best_similarity']:.6f} | "
                f"`query-derived {s.get('query_fragment_kind', 'fragment')}_{s.get('query_core_size', '-')}` | "
                f"{ref.get('candidate_library', '-')} ({ref.get('candidate_fragment_label', '-')}) |"
            )
    else:
        lines.append("- None.")
    lines.append("")
    lines.append("## Global Assembly")
    if assembly.get("available"):
        lines.append(f"- Assembly score: `{assembly['score']:.6f}`")
        lines.append(f"- Mean fragment similarity: `{assembly['individual_mean']:.6f}`")
        lines.append(f"- Pairwise consistency: `{assembly['pairwise_mean']:.6f}`")
        lines.append(f"- Reuse ratio: `{assembly.get('reuse_ratio', 0.0):.6f}`")
        lines.append(f"- Reuse factor: `{assembly.get('reuse_factor', 1.0):.6f}`")
        lines.append("")
        lines.append("| Query fragment | Selected candidate | sim | chem |")
        lines.append("|---|---|---:|---:|")
        for a in assembly.get("assignment", []):
            lines.append(
                f"| `{a['query_fragment_id']}` | `{a['candidate_fragment_label']}` | "
                f"{a['similarity']:.6f} | {a['chemical_compatibility']:.6f} |"
            )
    else:
        lines.append(f"- Not available: `{assembly.get('reason', 'unknown')}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Export score vs reliability data for plotting.
    sr_lines = ["fragment_id,kind,core_size,best_similarity,best_reliability"]
    for r in report_rows:
        sr_lines.append(
            f"{r['query_fragment_id']},{r.get('query_fragment_kind', '')},"
            f"{r.get('query_core_size', '')},{r.get('best_similarity', 0.0):.6f},"
            f"{r.get('best_reliability', 0.0):.6f}"
        )
    sr_path.write_text("\n".join(sr_lines) + "\n", encoding="utf-8")

    gp_lines = [
        "set datafile separator ','",
        "set terminal pngcairo size 900,600",
        "set output 'score_reliability.png'",
        "set xlabel 'best_similarity'",
        "set ylabel 'best_reliability'",
        "set grid",
        "plot 'score_reliability.csv' using 4:5 with points pt 7 ps 1.0 title 'fragments'",
    ]
    gp_path.write_text("\n".join(gp_lines) + "\n", encoding="utf-8")
    write_fragment_view_html(xyz, report_rows, out_dir)
    return report


def write_fragment_view_html(
    query_xyz: Path,
    report_rows: list[dict],
    out_dir: Path,
    *,
    corrected_xyz: Path | None = None,
    corrected_label: str = "PCS2",
):
    try:
        atoms, coords, comment = read_xyz(query_xyz)
    except Exception:
        return
    corr_atoms = None
    corr_coords = None
    if corrected_xyz is not None and Path(corrected_xyz).exists():
        try:
            corr_atoms, corr_coords, _c = read_xyz(Path(corrected_xyz))
            if len(corr_atoms) != len(atoms) or any(a != b for a, b in zip(corr_atoms, atoms)):
                corr_atoms = None
                corr_coords = None
        except Exception:
            corr_atoms = None
            corr_coords = None

    max_frags = 24
    palette = [
        "#e41a1c",
        "#377eb8",
        "#4daf4a",
        "#984ea3",
        "#ff7f00",
        "#a65628",
        "#f781bf",
        "#999999",
        "#66c2a5",
        "#fc8d62",
        "#8da0cb",
        "#e78ac3",
        "#a6d854",
        "#ffd92f",
        "#e5c494",
        "#b3b3b3",
    ]

    frags = []
    for row in report_rows[:max_frags]:
        ranking = row.get("ranking", [])
        best = ranking[0] if ranking else {}
        frags.append(
            {
                "id": row.get("query_fragment_id"),
                "best_label": best.get("candidate_fragment_label") or "-",
                "best_library": best.get("candidate_library") or "-",
                "best_similarity": float(best.get("similarity", 0.0)) if best else 0.0,
                "atoms": [int(x) for x in row.get("query_atom_indices", [])],
            }
        )

    xyz_lines = [str(len(atoms)), comment or "query xyz"]
    for a, (x, y, z) in zip(atoms, coords):
        xyz_lines.append(f"{a} {x: .8f} {y: .8f} {z: .8f}")
    xyz_text = "\\n".join(xyz_lines)
    corr_xyz_text = None
    coords_list = [[float(x), float(y), float(z)] for x, y, z in coords]
    corr_list = None
    if corr_coords is not None:
        corr_list = [[float(x), float(y), float(z)] for x, y, z in corr_coords]
        corr_lines = [str(len(atoms)), f"{corrected_label} (corrected)"]
        for a, (x, y, z) in zip(atoms, corr_coords):
            corr_lines.append(f"{a} {x: .8f} {y: .8f} {z: .8f}")
        corr_xyz_text = "\\n".join(corr_lines)

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Fragment Viewer</title>
  <style>
    body {{ font-family: sans-serif; margin: 0; padding: 0; display: flex; height: 100vh; }}
    #viewer {{ flex: 1; min-width: 60%; }}
    #side {{ width: 360px; padding: 12px; overflow: auto; border-left: 1px solid #ddd; }}
    .legend-item {{ display: flex; align-items: center; margin: 6px 0; }}
    .swatch {{ width: 14px; height: 14px; margin-right: 8px; }}
    .title {{ font-weight: 600; margin-bottom: 8px; }}
    .note {{ color: #666; font-size: 12px; margin-top: 8px; }}
    .section {{ margin-top: 12px; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace; }}
    .btn {{ display: inline-block; margin-top: 6px; padding: 4px 8px; border: 1px solid #bbb; border-radius: 4px; cursor: pointer; }}
  </style>
  <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
</head>
<body>
  <div id="viewer"></div>
  <div id="side">
    <div class="title">Fragments (first {len(frags)} shown)</div>
    <div id="legend"></div>
    <div class="note">Labels show query fragment → best selected candidate.</div>
    <div class="section" id="viewmode_section">
      <div class="title">Show</div>
      <div><input type="radio" name="viewmode" id="view_hpcs2" checked> HPCS2</div>
      <div><input type="radio" name="viewmode" id="view_corr"> {corrected_label}</div>
      <div><input type="radio" name="viewmode" id="view_both"> Both</div>
    </div>
    <div class="section">
      <div class="title">Bond length tool</div>
      <div>Select two atoms to measure distances.</div>
      <div class="mono">Atoms: <span id="bond_atoms">-</span></div>
      <div class="mono">HPCS2: <span id="bond_hpcs2">-</span> Å</div>
      <div class="mono">{corrected_label}: <span id="bond_corr">-</span> Å</div>
      <div id="clear" class="btn">Clear selection</div>
    </div>
  </div>
  <script>
    const xyz = `{xyz_text}`;
    const viewer = $3Dmol.createViewer("viewer", {{ backgroundColor: "white" }});
    viewer.addModel(xyz, "xyz");
    viewer.setStyle({{model: 0}}, {{ stick: {{ radius: 0.2, colorscheme: "Jmol" }} }});
    const corrXYZ = {json.dumps(corr_xyz_text) if corr_xyz_text is not None else "null"};
    if (corrXYZ) {{
      viewer.addModel(corrXYZ, "xyz");
      viewer.setStyle({{model: 1}}, {{ stick: {{ radius: 0.12, color: "#444444" }}, sphere: {{ scale: 0.15, color: "#444444" }} }});
    }}

    const fragments = {json.dumps(frags)};
    const colors = {json.dumps(palette)};
    const coords = {json.dumps(coords_list)};
    const corrCoords = {json.dumps(corr_list) if corr_list is not None else "null"};
    const corrLabel = {json.dumps(corrected_label)};
    const legend = document.getElementById("legend");
    const viewSection = document.getElementById("viewmode_section");
    const viewH = document.getElementById("view_hpcs2");
    const viewC = document.getElementById("view_corr");
    const viewB = document.getElementById("view_both");
    const bondAtoms = document.getElementById("bond_atoms");
    const bondH = document.getElementById("bond_hpcs2");
    const bondC = document.getElementById("bond_corr");
    const clearBtn = document.getElementById("clear");
    let picked = [];
    let bondObj = null;

    fragments.forEach((f, i) => {{
      const color = colors[i % colors.length];
      viewer.setStyle({{ index: f.atoms }}, {{
        stick: {{ radius: 0.32, color }},
        sphere: {{ scale: 0.3, color }}
      }});
      const row = document.createElement("div");
      row.className = "legend-item";
      const sw = document.createElement("div");
      sw.className = "swatch";
      sw.style.background = color;
      const label = document.createElement("div");
      const sim = (f.best_similarity || 0).toFixed(3);
      label.textContent = `${{f.id}} → ${{f.best_label}} [${{f.best_library}}; sim=${{sim}}]`;
      row.appendChild(sw);
      row.appendChild(label);
      legend.appendChild(row);
    }});

    function dist(a, b, arr) {{
      const dx = arr[a][0] - arr[b][0];
      const dy = arr[a][1] - arr[b][1];
      const dz = arr[a][2] - arr[b][2];
      return Math.sqrt(dx*dx + dy*dy + dz*dz);
    }}

    function isBonded(a, b) {{
      const atom = viewer.getModel().selectedAtoms({{index: a}})[0];
      if (!atom || !atom.bonds) return false;
      return atom.bonds.includes(b);
    }}

    function updateBondInfo() {{
      if (picked.length < 2) {{
        bondAtoms.textContent = picked.length === 0 ? "-" : `${{picked[0]+1}}`;
        bondH.textContent = "-";
        bondC.textContent = "-";
        return;
      }}
      const a = picked[0];
      const b = picked[1];
      if (!isBonded(a, b)) {{
        bondAtoms.textContent = `${{a+1}}–${{b+1}}`;
        bondH.textContent = "not bonded";
        bondC.textContent = "not bonded";
        return;
      }}
      const d0 = dist(a, b, coords);
      bondAtoms.textContent = `${{a+1}}–${{b+1}}`;
      bondH.textContent = d0.toFixed(4);
      if (corrCoords) {{
        const d1 = dist(a, b, corrCoords);
        bondC.textContent = d1.toFixed(4);
      }} else {{
        bondC.textContent = "-";
      }}
      if (bondObj) {{
        viewer.removeShape(bondObj);
        bondObj = null;
      }}
      bondObj = viewer.addCylinder({{
        start: {{x: coords[a][0], y: coords[a][1], z: coords[a][2]}},
        end: {{x: coords[b][0], y: coords[b][1], z: coords[b][2]}},
        radius: 0.05,
        color: "black",
        dashed: true
      }});
      viewer.render();
    }}

    function atomIndex(atom) {{
      if (atom === undefined || atom === null) return null;
      if (Number.isInteger(atom.index)) return atom.index;
      if (Number.isInteger(atom.serial)) return atom.serial - 1;
      return null;
    }}

    viewer.setClickable({{}}, true, function(atom) {{
      const idx = atomIndex(atom);
      if (idx === null || idx < 0 || idx >= coords.length) {{
        return;
      }}
      if (picked.length === 0 && atom.bonds && atom.bonds.length === 1) {{
        const j = atom.bonds[0];
        if (Number.isInteger(j)) {{
          picked = [idx, j];
          updateBondInfo();
          return;
        }}
      }}
      if (picked.length === 2) picked = [];
      if (!picked.includes(idx)) picked.push(idx);
      updateBondInfo();
    }});

    function setModelVisible(modelIdx, visible) {{
      const mdl = viewer.getModel(modelIdx);
      if (!mdl) return;
      mdl.setStyle({{}}, visible ? (modelIdx === 0
        ? {{ stick: {{ radius: 0.2, colorscheme: "Jmol" }} }}
        : {{ stick: {{ radius: 0.12, color: "#444444" }}, sphere: {{ scale: 0.15, color: "#444444" }} }})
        : {{}}
      );
    }}

    function applyViewMode() {{
      if (!corrXYZ) {{
        setModelVisible(0, true);
        viewer.render();
        return;
      }}
      if (viewH.checked) {{
        setModelVisible(0, true);
        setModelVisible(1, false);
      }} else if (viewC.checked) {{
        setModelVisible(0, false);
        setModelVisible(1, true);
      }} else {{
        setModelVisible(0, true);
        setModelVisible(1, true);
      }}
      viewer.render();
    }}

    if (viewH && viewC && viewB) {{
      viewH.onchange = applyViewMode;
      viewC.onchange = applyViewMode;
      viewB.onchange = applyViewMode;
    }}
    if (!corrXYZ && viewSection) {{
      viewSection.style.display = "none";
    }}
    applyViewMode();

    clearBtn.onclick = () => {{
      picked = [];
      if (bondObj) {{
        viewer.removeShape(bondObj);
        bondObj = null;
        viewer.render();
      }}
      updateBondInfo();
    }};

    viewer.zoomTo();
    viewer.render();
  </script>
</body>
</html>
"""

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fragment_view.html").write_text(html, encoding="utf-8")
    (out_dir / "fragment_view.json").write_text(
        json.dumps({"query_xyz": str(query_xyz), "fragments": frags}, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fragment similarity + GAP detection MVP.")
    ap.add_argument("--xyz", required=True, help="Query molecule XYZ.")
    ap.add_argument("--se-dir", required=True, help="SE library directory.")
    ap.add_argument("--pcs2-dir", required=True, help="PCS2 library directory.")
    ap.add_argument("--out", required=True, help="Output directory.")
    ap.add_argument(
        "--target-pcs2",
        action="store_true",
        help="Target PCS2 level: use PCS2-only matching (disable SE).",
    )
    ap.add_argument(
        "--pcs2-only",
        action="store_true",
        help="Deprecated alias for --target-pcs2.",
    )
    ap.add_argument("--xyz-glob", default="*.xyz")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--gap-threshold", type=float, default=0.75)
    ap.add_argument("--max-ring-size-delta", type=int, default=2)
    ap.add_argument("--chemical-min-score", type=float, default=0.15)
    ap.add_argument("--reuse-penalty", type=float, default=0.04)
    ap.add_argument("--low-score-threshold", type=float, default=0.45)
    args = ap.parse_args(argv)

    target_pcs2 = bool(args.target_pcs2 or args.pcs2_only)
    rep = run_fragment_pipeline(
        Path(args.xyz),
        Path(args.se_dir),
        Path(args.pcs2_dir),
        Path(args.out),
        xyz_glob=args.xyz_glob,
        top_k=args.top_k,
        gap_threshold=args.gap_threshold,
        max_ring_size_delta=args.max_ring_size_delta,
        chemical_min_score=args.chemical_min_score,
        reuse_penalty=args.reuse_penalty,
        low_score_threshold=args.low_score_threshold,
        use_pcs2_only=target_pcs2,
        library_label="PCS2/HPCS2" if target_pcs2 else "SE/PCS2",
    )
    print(f"Fragment report written to: {Path(args.out) / 'fragment_pipeline.json'}")
    print(f"Fragments: {rep['query_fragments']}  GAPs: {len(rep['to_curate'])}")


if __name__ == "__main__":
    main()
