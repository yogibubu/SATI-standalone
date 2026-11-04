from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .modify_geom import read_xyz
from .pipeline import _load_topology_elements
from .synthon_similarity import compare_against_library
from .symmetry_detector import orient_coords, symmetry_elements_from_geometry, is_linear
from .symmetry_classifier import group_label
from matrix_chem.topology.pipeline import build_topology_objects
from matrix_chem.topology.topology_reporting import print_topology_report


def _point_group_from_xyz(xyz: Path):
    atoms, coords_ang, _ = read_xyz(xyz)
    atomic_number = _load_topology_elements()
    Z = np.array([atomic_number(a) for a in atoms], dtype=int)
    coords_oriented = orient_coords(coords_ang, weights=Z)
    elements, _, _ = symmetry_elements_from_geometry(atoms, coords_oriented, tol=1.0e-3, max_n=10)
    return group_label(elements, linear=is_linear(coords_oriented, tol=1.0e-3))


def _write_topology_report(xyz: Path, out_path: Path):
    atoms, coords_ang, _ = read_xyz(xyz)
    atomic_number = _load_topology_elements()
    Z = np.array([atomic_number(a) for a in atoms], dtype=int)
    cg, dg, ringset, synthons, aromaticity = build_topology_objects(coords_ang, Z)
    print_topology_report(
        cg=cg,
        dg=dg,
        synthons=synthons,
        arom=aromaticity,
        filename=str(out_path),
        ringset=ringset,
    )


def _write_markdown_report(query: Path, similarity_report: dict, point_group: str, out_md: Path):
    lines = [
        f"# Auto Report: {query.name}",
        "",
        f"- Query: `{query}`",
        f"- Point group: `{point_group}`",
        f"- Compared molecules: `{similarity_report['library_size']}`",
        "",
        "## Top Similarity Matches",
        "",
        "| Rank | Molecule | sim_comb | sim_syn | sim_ring |",
        "|---:|---|---:|---:|---:|",
    ]
    for i, row in enumerate(similarity_report["ranking"][:10], start=1):
        lines.append(
            f"| {i} | `{row['xyz']}` | {row['similarity_combined']:.6f} | "
            f"{row['similarity_exp_minus_db']:.6f} | {row['ring_similarity_exp_minus_db']:.6f} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_auto_reports(
    query_dir: Path,
    library_dir: Path,
    out_dir: Path,
    *,
    xyz_glob: str = "*.xyz",
    top_k: int = 10,
    covariance_mode: str = "full",
    regularization: float = 5.0e-2,
    ring_weight: float = 0.25,
):
    queries = sorted(query_dir.glob(xyz_glob))
    library = sorted(library_dir.glob(xyz_glob))
    if not queries:
        raise ValueError("No query XYZ files found.")
    if not library:
        raise ValueError("No library XYZ files found.")

    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for query in queries:
        report = compare_against_library(
            query,
            library,
            covariance_mode=covariance_mode,
            regularization=regularization,
            standardize=True,
            include_ring_comparison=True,
            ring_weight=ring_weight,
        )
        report["ranking"] = report["ranking"][: max(1, int(top_k))]
        pg = "n/a"
        point_group_error = None
        try:
            pg = _point_group_from_xyz(query)
        except Exception as exc:
            point_group_error = str(exc)

        stem = query.stem
        top_path = out_dir / f"{stem}.topology.report"
        json_path = out_dir / f"{stem}.similarity.json"
        md_path = out_dir / f"{stem}.report.md"

        topology_error = None
        try:
            _write_topology_report(query, top_path)
        except Exception as exc:
            topology_error = str(exc)
        json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        _write_markdown_report(query, report, pg, md_path)

        best = report["ranking"][0] if report["ranking"] else None
        index.append(
            {
                "query_xyz": str(query),
                "point_group": pg,
                "best_match": best["xyz"] if best else None,
                "best_similarity_combined": best["similarity_combined"] if best else None,
                "topology_report": str(top_path),
                "similarity_json": str(json_path),
                "markdown_report": str(md_path),
                "point_group_error": point_group_error,
                "topology_error": topology_error,
            }
        )

    index_path = out_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return index


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Auto-report pipeline (topology + symmetry + similarity)."
    )
    ap.add_argument("--query-dir", required=True)
    ap.add_argument("--library-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--xyz-glob", default="*.xyz")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--covariance-mode", choices=("full", "diag"), default="full")
    ap.add_argument("--regularization", type=float, default=5.0e-2)
    ap.add_argument("--ring-weight", type=float, default=0.25)
    args = ap.parse_args(argv)

    index = run_auto_reports(
        Path(args.query_dir),
        Path(args.library_dir),
        Path(args.out_dir),
        xyz_glob=args.xyz_glob,
        top_k=args.top_k,
        covariance_mode=args.covariance_mode,
        regularization=args.regularization,
        ring_weight=args.ring_weight,
    )
    print(f"Auto reports generated: {len(index)}")
    print(f"Index: {Path(args.out_dir) / 'index.json'}")


if __name__ == "__main__":
    main()
