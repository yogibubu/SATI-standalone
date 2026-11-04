from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from matrix_smith.coordinate_kernel.modify_geom import read_xyz, write_xyz


def _subset_xyz(atoms: list[str], coords: np.ndarray, indices: list[int]):
    idx = [int(i) for i in indices]
    return [atoms[i] for i in idx], np.array([coords[i] for i in idx], dtype=float)


def _kabsch_transform(src: np.ndarray, dst: np.ndarray):
    c_src = src.mean(axis=0)
    c_dst = dst.mean(axis=0)
    x = src - c_src
    y = dst - c_dst
    h = x.T @ y
    u, _s, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0.0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    t = c_dst - c_src @ r
    return r, t


def _align_coords(src: np.ndarray, dst: np.ndarray):
    r, t = _kabsch_transform(src, dst)
    return src @ r + t, r, t


def prepare_delta_workflow(query_xyz: Path, fragment_report_json: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    atoms_q, coords_q, _ = read_xyz(query_xyz)

    report = json.loads(fragment_report_json.read_text(encoding="utf-8"))
    rows = report.get("fragments", [])

    manifest = {
        "query_xyz": str(query_xyz),
        "fragment_report_json": str(fragment_report_json),
        "entries": [],
    }

    for i, row in enumerate(rows, start=1):
        q_idx = [int(x) for x in row.get("query_atom_indices", [])]
        ranking = row.get("ranking", [])
        if not q_idx or not ranking:
            continue
        best = ranking[0]
        new_frag_suggestion = row.get("suggested_new_fragment")
        c_idx = [int(x) for x in best.get("candidate_atom_indices", [])]
        cand_xyz = Path(best["candidate_xyz"])
        if not cand_xyz.exists() or not c_idx:
            continue

        atoms_c, coords_c, _ = read_xyz(cand_xyz)
        q_atoms, q_coords = _subset_xyz(atoms_q, coords_q, q_idx)
        c_atoms, c_coords = _subset_xyz(atoms_c, coords_c, c_idx)

        frag_dir = out_dir / f"frag_{i:02d}_{row['query_fragment_id'].replace(':', '_')}"
        frag_dir.mkdir(parents=True, exist_ok=True)
        q_frag_xyz = frag_dir / "query_fragment.xyz"
        c_frag_xyz = frag_dir / "candidate_fragment.xyz"
        proposed_new_frag_xyz = frag_dir / "proposed_new_library_fragment.xyz"
        ll_xyz = frag_dir / "low_level_result.xyz"
        hl_xyz = frag_dir / "high_level_result.xyz"
        write_xyz(
            q_frag_xyz, q_atoms, q_coords, comment=f"query fragment {row['query_fragment_id']}"
        )
        write_xyz(
            proposed_new_frag_xyz,
            q_atoms,
            q_coords,
            comment="query-derived fragment proposed for new high-level library entry",
        )
        write_xyz(
            c_frag_xyz,
            c_atoms,
            c_coords,
            comment=f"candidate fragment {best.get('candidate_fragment_label', best.get('candidate_fragment_id'))}",
        )

        manifest["entries"].append(
            {
                "query_fragment_id": row["query_fragment_id"],
                "query_atom_indices": q_idx,
                "candidate_fragment_label": best.get("candidate_fragment_label"),
                "candidate_xyz": str(cand_xyz),
                "suggested_low_level_input_xyz": str(q_frag_xyz),
                "suggested_high_level_reference_xyz": str(c_frag_xyz),
                "low_level_result_xyz": str(ll_xyz),
                # High-level fragment is taken directly from the curated library candidate.
                "high_level_result_xyz": str(c_frag_xyz),
                "reserved_high_level_result_xyz": str(hl_xyz),
                "proposed_new_library_fragment_xyz": str(proposed_new_frag_xyz),
                "proposed_new_library_high_level_xyz": str(
                    frag_dir / "proposed_new_library_high_level.xyz"
                ),
                "weight_prior": float(best.get("similarity", 0.0)),
                "is_low_score": bool(row.get("is_low_score", False)),
                "suggested_new_fragment": new_frag_suggestion,
                "status": "pending_low_level_only",
            }
        )

    manifest_path = out_dir / "delta_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def prepare_hpcs2_delta_workflow(
    query_xyz: Path,
    fragment_report_json: Path,
    hpcs2_dir: Path,
    out_dir: Path,
    *,
    xyz_glob: str = "*.xyz",
):
    out_dir.mkdir(parents=True, exist_ok=True)
    atoms_q, coords_q, _ = read_xyz(query_xyz)

    report = json.loads(fragment_report_json.read_text(encoding="utf-8"))
    rows = report.get("fragments", [])

    # Index HPCS2 library by molecule key (stem).
    hpcs2_map = {p.stem.lower(): p for p in sorted(hpcs2_dir.glob(xyz_glob))}

    manifest = {
        "query_xyz": str(query_xyz),
        "fragment_report_json": str(fragment_report_json),
        "hpcs2_dir": str(hpcs2_dir),
        "entries": [],
        "skipped": [],
    }

    for i, row in enumerate(rows, start=1):
        q_idx = [int(x) for x in row.get("query_atom_indices", [])]
        ranking = row.get("ranking", [])
        if not q_idx or not ranking:
            continue
        best = None
        for cand in ranking:
            if cand.get("candidate_library") != "PCS2":
                continue
            c_idx = [int(x) for x in cand.get("candidate_atom_indices", [])]
            if len(c_idx) != len(q_idx):
                continue
            best = cand
            break
        if best is None:
            manifest["skipped"].append(
                {
                    "query_fragment_id": row.get("query_fragment_id"),
                    "reason": "No PCS2 candidate with matching atom count for PCS2->HPCS2 delta.",
                }
            )
            continue

        c_idx = [int(x) for x in best.get("candidate_atom_indices", [])]
        cand_xyz = Path(best["candidate_xyz"])
        if not cand_xyz.exists() or not c_idx:
            continue

        # Try to locate corresponding HPCS2 molecule by stem.
        hpcs2_xyz = None
        key = str(best.get("candidate_molecule_key", "")).lower()
        if key:
            hpcs2_xyz = hpcs2_map.get(key)
        if hpcs2_xyz is None:
            hpcs2_xyz = hpcs2_dir / cand_xyz.name
            if not hpcs2_xyz.exists():
                hpcs2_xyz = None
        if hpcs2_xyz is None or not hpcs2_xyz.exists():
            manifest["skipped"].append(
                {
                    "query_fragment_id": row.get("query_fragment_id"),
                    "reason": "Missing HPCS2 counterpart for PCS2 candidate.",
                    "candidate_xyz": str(cand_xyz),
                }
            )
            continue

        atoms_c, coords_c, _ = read_xyz(cand_xyz)
        atoms_h, coords_h, _ = read_xyz(hpcs2_xyz)
        q_atoms, q_coords = _subset_xyz(atoms_q, coords_q, q_idx)
        c_atoms, c_coords = _subset_xyz(atoms_c, coords_c, c_idx)
        h_atoms, h_coords = _subset_xyz(atoms_h, coords_h, c_idx)
        if len(c_atoms) != len(h_atoms) or any(a != b for a, b in zip(c_atoms, h_atoms)):
            manifest["skipped"].append(
                {
                    "query_fragment_id": row.get("query_fragment_id"),
                    "reason": "PCS2/HPCS2 fragment atom mismatch.",
                    "candidate_xyz": str(cand_xyz),
                    "hpcs2_xyz": str(hpcs2_xyz),
                }
            )
            continue

        frag_dir = out_dir / f"frag_{i:02d}_{row['query_fragment_id'].replace(':', '_')}"
        frag_dir.mkdir(parents=True, exist_ok=True)
        q_frag_xyz = frag_dir / "query_fragment.xyz"
        pcs2_frag_xyz = frag_dir / "pcs2_fragment.xyz"
        hpcs2_frag_xyz = frag_dir / "hpcs2_fragment.xyz"
        write_xyz(
            q_frag_xyz, q_atoms, q_coords, comment=f"query fragment {row['query_fragment_id']}"
        )
        write_xyz(
            pcs2_frag_xyz,
            c_atoms,
            c_coords,
            comment=f"PCS2 fragment {best.get('candidate_fragment_label', best.get('candidate_fragment_id'))}",
        )
        write_xyz(
            hpcs2_frag_xyz,
            h_atoms,
            h_coords,
            comment=f"HPCS2 fragment {best.get('candidate_fragment_label', best.get('candidate_fragment_id'))}",
        )

        manifest["entries"].append(
            {
                "query_fragment_id": row["query_fragment_id"],
                "query_atom_indices": q_idx,
                "candidate_fragment_label": best.get("candidate_fragment_label"),
                "candidate_xyz": str(cand_xyz),
                "hpcs2_xyz": str(hpcs2_xyz),
                "suggested_low_level_input_xyz": str(q_frag_xyz),
                "suggested_high_level_reference_xyz": str(pcs2_frag_xyz),
                # HPCS2 is low-level; PCS2 is high-level (fragment library).
                "low_level_result_xyz": str(hpcs2_frag_xyz),
                "high_level_result_xyz": str(pcs2_frag_xyz),
                "weight_prior": float(best.get("similarity", 0.0)),
                "is_low_score": bool(row.get("is_low_score", False)),
                "status": "library_delta_ready",
            }
        )

    manifest_path = out_dir / "delta_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def prepare_low_level_gaussian_inputs(
    delta_manifest_json: Path,
    *,
    route: str,
    charge: int = 0,
    multiplicity: int = 1,
    nproc: int = 8,
    mem: str = "8GB",
):
    from matrix_gaussian import ensure_gaussian_output_pickett

    manifest = json.loads(delta_manifest_json.read_text(encoding="utf-8"))
    entries = manifest.get("entries", [])
    written = []
    gaussian_route = ensure_gaussian_output_pickett(route)

    for idx, e in enumerate(entries, start=1):
        xyz_path = Path(e.get("suggested_low_level_input_xyz", ""))
        if not xyz_path.exists():
            continue
        atoms, coords, _ = read_xyz(xyz_path)
        gjf_path = xyz_path.with_name("low_level_input.gjf")
        chk_name = f"frag_{idx:02d}.chk"
        lines = [
            f"%chk={chk_name}",
            f"%nprocshared={int(max(1, nproc))}",
            f"%mem={mem}",
            gaussian_route,
            "",
            f"Low-level fragment {e.get('query_fragment_id', idx)}",
            "",
            f"{int(charge)} {int(multiplicity)}",
        ]
        for a, (x, y, z) in zip(atoms, coords):
            lines.append(f"{a:2s} {x: .8f} {y: .8f} {z: .8f}")
        lines.append("")
        gjf_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        e["low_level_gaussian_input"] = str(gjf_path)
        e["status"] = "pending_low_level_gaussian_run"
        written.append(str(gjf_path))

    manifest["low_level_gaussian"] = {
        "route": gaussian_route,
        "charge": int(charge),
        "multiplicity": int(multiplicity),
        "nproc": int(max(1, nproc)),
        "mem": str(mem),
        "generated_inputs": len(written),
    }
    delta_manifest_json.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"generated_inputs": len(written), "inputs": written}


def prepare_high_level_curation_jobs(
    delta_manifest_json: Path,
    out_dir: Path,
    *,
    route: str,
    charge: int = 0,
    multiplicity: int = 1,
    nproc: int = 8,
    mem: str = "16GB",
):
    from matrix_gaussian import ensure_gaussian_output_pickett

    manifest = json.loads(delta_manifest_json.read_text(encoding="utf-8"))
    entries = manifest.get("entries", [])
    out_dir.mkdir(parents=True, exist_ok=True)
    gaussian_route = ensure_gaussian_output_pickett(route)

    queue = []
    for idx, e in enumerate(entries, start=1):
        if not bool(e.get("is_low_score", False)):
            continue
        src_xyz = Path(e.get("proposed_new_library_fragment_xyz", ""))
        if not src_xyz.exists():
            continue
        qid = str(e.get("query_fragment_id", f"frag_{idx}")).replace(":", "_")
        job_dir = out_dir / f"{idx:02d}_{qid}"
        job_dir.mkdir(parents=True, exist_ok=True)
        xyz_out = job_dir / "fragment_for_high_level.xyz"
        shutil.copy2(src_xyz, xyz_out)

        atoms, coords, _ = read_xyz(xyz_out)
        gjf_path = job_dir / "high_level_input.gjf"
        lines = [
            f"%chk={qid}.chk",
            f"%nprocshared={int(max(1, nproc))}",
            f"%mem={mem}",
            gaussian_route,
            "",
            f"High-level curation fragment {e.get('query_fragment_id', idx)}",
            "",
            f"{int(charge)} {int(multiplicity)}",
        ]
        for a, (x, y, z) in zip(atoms, coords):
            lines.append(f"{a:2s} {x: .8f} {y: .8f} {z: .8f}")
        lines.append("")
        gjf_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        q = {
            "query_fragment_id": e.get("query_fragment_id"),
            "job_dir": str(job_dir),
            "fragment_xyz": str(xyz_out),
            "high_level_input_gjf": str(gjf_path),
            "closest_library_reference": (e.get("suggested_new_fragment") or {}).get(
                "closest_library_reference", {}
            ),
        }
        queue.append(q)

    queue_path = out_dir / "high_level_curation_jobs.json"
    queue_path.write_text(json.dumps(queue, indent=2) + "\n", encoding="utf-8")
    return {"jobs": queue, "queue_path": str(queue_path)}


def apply_delta_correction(
    query_xyz: Path,
    delta_manifest_json: Path,
    out_xyz: Path,
    *,
    min_weight: float = 0.05,
):
    atoms_q, coords_q, _ = read_xyz(query_xyz)
    coords_q = np.array(coords_q, dtype=float)
    nat = coords_q.shape[0]

    manifest = json.loads(delta_manifest_json.read_text(encoding="utf-8"))
    entries = manifest.get("entries", [])

    disp_sum = np.zeros_like(coords_q)
    w_sum = np.zeros((nat,), dtype=float)
    used = []

    for e in entries:
        q_idx = [int(x) for x in e.get("query_atom_indices", [])]
        ll_path = Path(e.get("low_level_result_xyz", ""))
        hl_path = Path(e.get("high_level_result_xyz", ""))
        w = float(e.get("weight_prior", 0.0))
        if w < min_weight:
            continue
        if not q_idx or not ll_path.exists() or not hl_path.exists():
            continue

        atoms_ll, coords_ll, _ = read_xyz(ll_path)
        atoms_hl, coords_hl, _ = read_xyz(hl_path)
        if len(atoms_ll) != len(atoms_hl) or len(atoms_ll) != len(q_idx):
            continue

        q_coords = np.array([coords_q[i] for i in q_idx], dtype=float)
        ll = np.array(coords_ll, dtype=float)
        hl = np.array(coords_hl, dtype=float)

        # Build delta in low-level local frame.
        hl_in_ll, _r_hl2ll, _t_hl2ll = _align_coords(hl, ll)
        delta_ll = hl_in_ll - ll

        # Transfer local delta to query fragment frame.
        _ll_in_q, r_ll2q, _t_ll2q = _align_coords(ll, q_coords)
        delta_q = delta_ll @ r_ll2q

        for local_i, atom_i in enumerate(q_idx):
            disp_sum[atom_i] += w * delta_q[local_i]
            w_sum[atom_i] += w

        used.append(
            {
                "query_fragment_id": e.get("query_fragment_id"),
                "weight": w,
                "nat": len(q_idx),
            }
        )

    corrected = np.array(coords_q, dtype=float)
    mask = w_sum > 1.0e-12
    corrected[mask] = coords_q[mask] + disp_sum[mask] / w_sum[mask, None]

    out_xyz.parent.mkdir(parents=True, exist_ok=True)
    write_xyz(
        out_xyz,
        atoms_q,
        corrected,
        comment="delta-corrected geometry from fragment low/high level deltas",
    )
    meta = {
        "query_xyz": str(query_xyz),
        "delta_manifest_json": str(delta_manifest_json),
        "out_xyz": str(out_xyz),
        "entries_total": len(entries),
        "entries_used": len(used),
        "atoms_corrected": int(np.count_nonzero(mask)),
        "used": used,
    }
    meta_path = out_xyz.with_suffix(".delta_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fragment delta-correction workflow (prepare/apply).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_prep = sub.add_parser(
        "prepare", help="Prepare fragment files + manifest for LL/HL calculations."
    )
    ap_prep.add_argument("--query-xyz", required=True)
    ap_prep.add_argument("--fragment-report", required=True, help="fragment_pipeline.json")
    ap_prep.add_argument("--out", required=True, help="Output folder for workflow bundle.")

    ap_hpcs2 = sub.add_parser(
        "prepare-hpcs2",
        help="Prepare PCS2 (HL) - HPCS2 (LL) library delta workflow.",
    )
    ap_hpcs2.add_argument("--query-xyz", required=True)
    ap_hpcs2.add_argument("--fragment-report", required=True, help="fragment_pipeline.json")
    ap_hpcs2.add_argument("--hpcs2-dir", required=True, help="HPCS2 library directory.")
    ap_hpcs2.add_argument("--out", required=True, help="Output folder for workflow bundle.")
    ap_hpcs2.add_argument("--xyz-glob", default="*.xyz")

    ap_ll = sub.add_parser(
        "prepare-low-level", help="Create Gaussian .gjf inputs for fragment low-level calculations."
    )
    ap_ll.add_argument("--manifest", required=True, help="delta_manifest.json")
    ap_ll.add_argument(
        "--route", required=True, help='Gaussian route, e.g. "#p b3lyp/6-31g(d) opt"'
    )
    ap_ll.add_argument("--charge", type=int, default=0)
    ap_ll.add_argument("--multiplicity", type=int, default=1)
    ap_ll.add_argument("--nproc", type=int, default=8)
    ap_ll.add_argument("--mem", default="8GB")

    ap_hl = sub.add_parser(
        "prepare-high-level-queue",
        help="Export low-score query-derived fragments + Gaussian .gjf for high-level curation jobs.",
    )
    ap_hl.add_argument("--manifest", required=True, help="delta_manifest.json")
    ap_hl.add_argument("--out", required=True, help="Output folder for HL job queue")
    ap_hl.add_argument(
        "--route", required=True, help='Gaussian route, e.g. "#p wB97XD/def2TZVP Opt"'
    )
    ap_hl.add_argument("--charge", type=int, default=0)
    ap_hl.add_argument("--multiplicity", type=int, default=1)
    ap_hl.add_argument("--nproc", type=int, default=8)
    ap_hl.add_argument("--mem", default="16GB")

    ap_apply = sub.add_parser("apply", help="Apply LL->HL fragment deltas to full query geometry.")
    ap_apply.add_argument("--query-xyz", required=True)
    ap_apply.add_argument("--manifest", required=True, help="delta_manifest.json")
    ap_apply.add_argument("--out-xyz", required=True)
    ap_apply.add_argument("--min-weight", type=float, default=0.05)

    args = ap.parse_args(argv)
    if args.cmd == "prepare":
        rep = prepare_delta_workflow(
            Path(args.query_xyz),
            Path(args.fragment_report),
            Path(args.out),
        )
        print(f"Delta manifest written to: {Path(args.out) / 'delta_manifest.json'}")
        print(f"Fragments prepared: {len(rep.get('entries', []))}")
    elif args.cmd == "prepare-hpcs2":
        rep = prepare_hpcs2_delta_workflow(
            Path(args.query_xyz),
            Path(args.fragment_report),
            Path(args.hpcs2_dir),
            Path(args.out),
            xyz_glob=args.xyz_glob,
        )
        print(f"Delta manifest written to: {Path(args.out) / 'delta_manifest.json'}")
        print(f"Fragments prepared: {len(rep.get('entries', []))}")
    elif args.cmd == "prepare-low-level":
        out = prepare_low_level_gaussian_inputs(
            Path(args.manifest),
            route=args.route,
            charge=args.charge,
            multiplicity=args.multiplicity,
            nproc=args.nproc,
            mem=args.mem,
        )
        print(f"Low-level Gaussian inputs generated: {out['generated_inputs']}")
    elif args.cmd == "prepare-high-level-queue":
        out = prepare_high_level_curation_jobs(
            Path(args.manifest),
            Path(args.out),
            route=args.route,
            charge=args.charge,
            multiplicity=args.multiplicity,
            nproc=args.nproc,
            mem=args.mem,
        )
        print(f"High-level curation jobs generated: {len(out['jobs'])}")
        print(f"Queue JSON: {out['queue_path']}")
    elif args.cmd == "apply":
        meta = apply_delta_correction(
            Path(args.query_xyz),
            Path(args.manifest),
            Path(args.out_xyz),
            min_weight=float(args.min_weight),
        )
        print(f"Corrected XYZ written to: {args.out_xyz}")
        print(
            f"Used entries: {meta['entries_used']} / {meta['entries_total']}  "
            f"Atoms corrected: {meta['atoms_corrected']}"
        )


if __name__ == "__main__":
    main()
