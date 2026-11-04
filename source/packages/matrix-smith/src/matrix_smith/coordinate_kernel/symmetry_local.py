from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple
from itertools import permutations
from hashlib import sha1

import numpy as np

from .pipeline import build_topology_full


@dataclass(frozen=True)
class AtomClassMeta:
    zeff_mean: float
    signature: Tuple[int, int, int, int, int]
    members: Tuple[int, ...]


@dataclass(frozen=True)
class LocalGeometry:
    label: str
    score: float
    angles: Tuple[float, ...]


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _pairwise_angles(vecs: np.ndarray) -> List[float]:
    ang = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            c = float(np.clip(np.dot(vecs[i], vecs[j]), -1.0, 1.0))
            ang.append(math.degrees(math.acos(c)))
    return ang


def _template_vectors() -> Dict[int, Dict[str, np.ndarray]]:
    sqrt3 = math.sqrt(3.0)
    phi = (1.0 + math.sqrt(5.0)) / 2.0

    templates: Dict[int, Dict[str, np.ndarray]] = {
        4: {
            "tetra": np.array(
                [
                    (1, 1, 1),
                    (1, -1, -1),
                    (-1, 1, -1),
                    (-1, -1, 1),
                ],
                dtype=float,
            ),
            "square_planar": np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)], dtype=float),
        },
        5: {
            "tbp": np.array(
                [
                    (0, 0, 1),
                    (0, 0, -1),
                    (1, 0, 0),
                    (-0.5, sqrt3 / 2.0, 0),
                    (-0.5, -sqrt3 / 2.0, 0),
                ],
                dtype=float,
            ),
            "square_pyramidal": np.array(
                [
                    (1, 0, 0),
                    (-1, 0, 0),
                    (0, 1, 0),
                    (0, -1, 0),
                    (0, 0, 1),
                ],
                dtype=float,
            ),
        },
        6: {
            "octa": np.array(
                [
                    (1, 0, 0),
                    (-1, 0, 0),
                    (0, 1, 0),
                    (0, -1, 0),
                    (0, 0, 1),
                    (0, 0, -1),
                ],
                dtype=float,
            ),
            "trig_prism": np.array(
                [
                    (1, 0, 1),
                    (-0.5, sqrt3 / 2.0, 1),
                    (-0.5, -sqrt3 / 2.0, 1),
                    (1, 0, -1),
                    (-0.5, sqrt3 / 2.0, -1),
                    (-0.5, -sqrt3 / 2.0, -1),
                ],
                dtype=float,
            ),
        },
        8: {
            "square_antiprism": np.array(
                [
                    (1, 0, 1),
                    (0, 1, 1),
                    (-1, 0, 1),
                    (0, -1, 1),
                    (math.sqrt(0.5), math.sqrt(0.5), -1),
                    (-math.sqrt(0.5), math.sqrt(0.5), -1),
                    (-math.sqrt(0.5), -math.sqrt(0.5), -1),
                    (math.sqrt(0.5), -math.sqrt(0.5), -1),
                ],
                dtype=float,
            ),
            "cube": np.array(
                [
                    (1, 1, 1),
                    (1, 1, -1),
                    (1, -1, 1),
                    (1, -1, -1),
                    (-1, 1, 1),
                    (-1, 1, -1),
                    (-1, -1, 1),
                    (-1, -1, -1),
                ],
                dtype=float,
            ),
            "dodeca": np.array(
                [
                    (1, 0, phi),
                    (-1, 0, phi),
                    (1, 0, -phi),
                    (-1, 0, -phi),
                    (0, 1, phi),
                    (0, -1, phi),
                    (0, 1, -phi),
                    (0, -1, -phi),
                ],
                dtype=float,
            ),
        },
    }

    # normalize
    for n, geom in templates.items():
        for k, arr in geom.items():
            u = np.array([_unit(v) for v in arr], dtype=float)
            templates[n][k] = u
    return templates


_TEMPLATES = _template_vectors()
_GEOM_CACHE: Dict[Tuple[int, Tuple[int, ...]], str] = {}
_INVARIANT_CACHE: Dict[str, np.ndarray] = {}


def _shape_invariants(vecs: np.ndarray) -> np.ndarray:
    # rotational invariants from second moment tensor
    key = sha1(vecs.tobytes()).hexdigest()
    cached = _INVARIANT_CACHE.get(key)
    if cached is not None:
        return cached
    Q = vecs.T @ vecs
    evals = np.linalg.eigvalsh(Q)
    inv = np.sort(evals)
    _INVARIANT_CACHE[key] = inv
    return inv


def _kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    # a, b: (n,3)
    aa = a - a.mean(axis=0)
    bb = b - b.mean(axis=0)
    H = aa.T @ bb
    u, _, vt = np.linalg.svd(H)
    R = vt.T @ u.T
    if np.linalg.det(R) < 0:
        vt[-1, :] *= -1.0
        R = vt.T @ u.T
    ar = aa @ R
    diff = ar - bb
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _best_kabsch_rmsd(vecs: np.ndarray, tvecs: np.ndarray) -> float:
    n = len(vecs)
    if n > 8:
        return _kabsch_rmsd(vecs, tvecs)
    best = None
    for perm in permutations(range(n)):
        tv = tvecs[list(perm)]
        r = _kabsch_rmsd(vecs, tv)
        if best is None or r < best:
            best = r
    return float(best if best is not None else 1e9)


def match_local_geometry(vecs: np.ndarray, tol_deg: float = 12.0) -> LocalGeometry:
    n = len(vecs)
    if n not in _TEMPLATES:
        return LocalGeometry(label="unknown", score=1e9, angles=tuple(_pairwise_angles(vecs)))

    angles = sorted(_pairwise_angles(vecs))
    inv = _shape_invariants(vecs)
    best = None
    second = None
    for label, tvecs in _TEMPLATES[n].items():
        t_angles = sorted(_pairwise_angles(tvecs))
        if len(t_angles) != len(angles):
            continue
        diff = np.array(angles) - np.array(t_angles)
        angle_score = float(np.sqrt(np.mean(diff * diff)))
        k_score = _best_kabsch_rmsd(vecs, tvecs)
        t_inv = _shape_invariants(tvecs)
        inv_score = float(np.linalg.norm(inv - t_inv))
        # combine: favor kabsch, keep angle scale, add invariants
        score = 0.6 * k_score + 0.3 * angle_score + 0.1 * inv_score
        if best is None or score < best.score:
            second = best
            best = LocalGeometry(label=label, score=score, angles=tuple(angles))
        elif second is None or score < second.score:
            second = LocalGeometry(label=label, score=score, angles=tuple(angles))

    if best is None:
        return LocalGeometry(label="unknown", score=1e9, angles=tuple(angles))
    if best.score > tol_deg:
        return LocalGeometry(label="ambiguous", score=best.score, angles=tuple(angles))
    if second is not None and abs(second.score - best.score) < 0.5 * tol_deg:
        return LocalGeometry(label="ambiguous", score=best.score, angles=tuple(angles))
    return best


def atom_classes(coords, Z, zeff_tol=0.05):
    cg, dg, ringset, synthons, aromaticity = build_topology_full(coords, Z)
    zeff = np.array([synthons.Zeff(i) for i in range(dg.natoms)], dtype=float)

    # discrete signature: (A, NED, D, ring_tag, degree)
    sig = []
    for i in range(dg.natoms):
        _, A, NED, D = synthons.canonical_signature(i)
        ring_tag = 1 if ringset is not None and ringset.rings_of_atom(i) else 0
        degree = len(dg.adjacency[i])
        sig.append((int(A), int(NED), int(D), int(ring_tag), int(degree)))

    # group by discrete signature, then Zeff clustering
    by_sig: Dict[Tuple[int, int, int, int, int], List[int]] = {}
    for i, s in enumerate(sig):
        by_sig.setdefault(s, []).append(i)

    class_id = np.zeros(dg.natoms, dtype=int) - 1
    class_meta: Dict[int, AtomClassMeta] = {}
    next_id = 0

    for s, idxs in by_sig.items():
        idxs_sorted = sorted(idxs, key=lambda i: zeff[i])
        clusters: List[List[int]] = []
        for i in idxs_sorted:
            placed = False
            for c in clusters:
                zbin_i = int(round(zeff[i] / zeff_tol))
                zbin_c = int(round(zeff[c[0]] / zeff_tol))
                if zbin_i == zbin_c and abs(zeff[i] - zeff[c[0]]) < zeff_tol:
                    c.append(i)
                    placed = True
                    break
            if not placed:
                clusters.append([i])

        for c in clusters:
            for i in c:
                class_id[i] = next_id
            mean_zeff = float(np.mean([zeff[i] for i in c]))
            class_meta[next_id] = AtomClassMeta(
                zeff_mean=mean_zeff,
                signature=s,
                members=tuple(c),
            )
            next_id += 1

    # quasi-equivalence among classes (same signature, Zeff within 2*tol)
    quasi = {k: [] for k in class_meta}
    keys = list(class_meta.keys())
    for i in range(len(keys)):
        ki = keys[i]
        for j in range(i + 1, len(keys)):
            kj = keys[j]
            if class_meta[ki].signature != class_meta[kj].signature:
                continue
            if abs(class_meta[ki].zeff_mean - class_meta[kj].zeff_mean) < 2.0 * zeff_tol:
                quasi[ki].append(kj)
                quasi[kj].append(ki)

    return class_id, class_meta, zeff, ringset, quasi


def pattern_for_center(center, neighbors, atom_class, geom: LocalGeometry):
    # count classes of neighbors
    counts: Dict[int, int] = {}
    for nb in neighbors:
        counts[atom_class[nb]] = counts.get(atom_class[nb], 0) + 1

    # map to Y, Z, W... by count then class id
    letters = "YZWVUTSRQPONMLKJIHGFEDCBA"
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    parts = []
    for i, (cls, cnt) in enumerate(items):
        lab = letters[i] if i < len(letters) else f"G{i}"
        if cnt == 1:
            parts.append(lab)
        else:
            parts.append(f"{lab}{cnt}")

    pattern = "X(" + "".join(parts) + ")"
    return {
        "center": int(center),
        "neighbors": [int(n) for n in neighbors],
        "pattern": pattern,
        "geometry": geom.label,
        "geom_score": geom.score,
    }


def angle_bin(angle_deg: float):
    refs = [60.0, 90.0, 109.47, 120.0, 180.0]
    best = min(refs, key=lambda a: abs(angle_deg - a))
    if abs(angle_deg - best) <= 15.0:
        return int(round(best))
    return None


def _pattern_key(center, neighbors, atom_class):
    cls = tuple(sorted(atom_class[n] for n in neighbors))
    return (int(center), cls)


def smooth_geometry_label(center, neighbors, atom_class, geom_label):
    key = _pattern_key(center, neighbors, atom_class)
    if geom_label == "ambiguous" and key in _GEOM_CACHE:
        return _GEOM_CACHE[key]
    _GEOM_CACHE[key] = geom_label
    return geom_label


def write_pattern_report(path, report):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)


def write_pattern_report_txt(path, report):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("Local symmetry pattern report\n")
        fh.write(f"zeff_tol = {report.get('zeff_tol')}\n")
        fh.write(f"geometry_match_tol = {report.get('geometry_match_tol')}\n\n")
        ring_clusters = report.get("ring_clusters", [])
        ring_drop = report.get("ring_drop", [])
        if ring_clusters:
            fh.write("Ring clusters:\n")
            for comp in ring_clusters:
                fh.write(f"  cluster: {comp}\n")
            if ring_drop:
                fh.write(f"  reduced rings: {ring_drop}\n")
            fh.write("\n")
        fh.write("Centers:\n")
        for c in report.get("centers", []):
            fh.write(f"  center {c['center']:3d}  geom={c['geometry']}  pattern={c['pattern']}\n")
        fh.write("\nBlocks:\n")
        for b in report.get("blocks", []):
            fh.write(f"  {b['label']}: ncols={b['ncols']} nrows={len(b['indices'])}\n")
        dihed = report.get("dihedrals")
        if dihed:
            fh.write("\nDihedrals:\n")
            fh.write(f"  mode = {dihed.get('mode')}\n")
            for bond in dihed.get("bonds", []):
                fh.write(
                    f"  bond {bond['bond']}  selected={bond['selected']}  reason={bond['reason']}\n"
                )
        glob = report.get("global_symmetry")
        if glob:
            fh.write("\nGlobal symmetry:\n")
            fh.write(f"  point_group = {glob.get('point_group')}\n")
            fh.write(f"  max_dev = {glob.get('max_dev')}\n")
            fh.write(f"  mean_dev = {glob.get('mean_dev')}\n")
            if "confidence" in glob:
                fh.write(f"  confidence = {glob.get('confidence')}\n")
            perf = glob.get("perf")
            if perf:
                fh.write(
                    f"  perf: ops_total={perf.get('ops_total')} ops_kept={perf.get('ops_kept')}\n"
                )
