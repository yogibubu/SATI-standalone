from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .modify_geom import read_xyz
from .pipeline import _load_topology_elements, build_topology_full


FEATURE_NAMES = (
    "charge",
    "covalency",
    "delocalization",
    "strain",
    "zeff",
    "pi_index",
    "pi_pi_index",
)
RING_FEATURE_NAMES = (
    "ring_size",
    "ring_planarity",
    "ring_radial_pos",
    "ring_fused_degree",
    "ring_nn_centroid_dist",
    "ring_radius",
)


@dataclass
class GaussianModel:
    mean: np.ndarray
    cov: np.ndarray
    natoms: int


def _feature_explain(
    a: GaussianModel,
    b: GaussianModel,
    names: tuple[str, ...],
):
    sigma = 0.5 * (a.cov + b.cov)
    inv_sigma = np.linalg.pinv(sigma)
    delta = a.mean - b.mean
    mean_terms = 0.125 * delta * (inv_sigma @ delta)

    da = np.clip(np.diag(a.cov), 1.0e-15, None)
    db = np.clip(np.diag(b.cov), 1.0e-15, None)
    ds = np.clip(np.diag(sigma), 1.0e-15, None)
    var_terms = 0.5 * (np.log(ds) - 0.5 * (np.log(da) + np.log(db)))

    rows = []
    for i, nm in enumerate(names):
        total = float(mean_terms[i] + var_terms[i])
        rows.append(
            {
                "feature": nm,
                "mean_term": float(mean_terms[i]),
                "variance_term": float(var_terms[i]),
                "total_term": total,
                "mean_a": float(a.mean[i]),
                "mean_b": float(b.mean[i]),
                "abs_mean_delta": float(abs(delta[i])),
            }
        )
    rows.sort(key=lambda r: abs(r["total_term"]), reverse=True)
    return rows


def _xyz_to_au(path: Path):
    atoms, coords_ang, _ = read_xyz(path)
    atomic_number = _load_topology_elements()
    Z = np.array([atomic_number(a) for a in atoms], dtype=int)
    coords_au = np.array(coords_ang, dtype=float) / 0.52917721092
    return Z, coords_au


def synthon_feature_matrix(Z: np.ndarray, coords_au: np.ndarray) -> np.ndarray:
    _, dg, _, synthons, _ = build_topology_full(coords_au, Z)
    X = np.zeros((dg.natoms, len(FEATURE_NAMES)), dtype=float)
    for i in range(dg.natoms):
        X[i, 0] = synthons.charge(i)
        X[i, 1] = synthons.covalency(i)
        X[i, 2] = synthons.delocalization(i)
        X[i, 3] = synthons.strain(i)
        X[i, 4] = synthons.Zeff(i)
        X[i, 5] = synthons.pi_index(i)
        X[i, 6] = synthons.pi_pi_index(i)
    return X


def ring_feature_matrix(Z: np.ndarray, coords_au: np.ndarray) -> np.ndarray:
    _, dg, ringset, _, _ = build_topology_full(coords_au, Z)
    nrings = len(ringset)
    if nrings == 0:
        return np.zeros((0, len(RING_FEATURE_NAMES)), dtype=float)

    coords = np.array(coords_au, dtype=float)
    mol_center = coords.mean(axis=0)
    mol_scale = float(np.sqrt(((coords - mol_center) ** 2).sum(axis=1).mean()))
    if mol_scale < 1.0e-12:
        mol_scale = 1.0

    centers = []
    ring_sizes = []
    ring_planarity = []
    fused_degree = []
    ring_radius = []

    for ring in ringset.rings:
        xyz_ring = coords[ring.atoms]
        ctr = xyz_ring.mean(axis=0)
        centers.append(ctr)
        ring_sizes.append(float(len(ring)))
        p = ring.planarity()
        ring_planarity.append(float(p) if p is not None else 0.0)
        fused_degree.append(float(len(ring.connected_rings)))
        ring_radius.append(float(np.sqrt(((xyz_ring - ctr) ** 2).sum(axis=1).mean()) / mol_scale))

    centers = np.array(centers, dtype=float)
    radial_pos = np.linalg.norm(centers - mol_center, axis=1) / mol_scale

    nn_dist = np.zeros(nrings, dtype=float)
    if nrings > 1:
        for i in range(nrings):
            d = np.linalg.norm(centers - centers[i], axis=1)
            d[i] = np.inf
            nn_dist[i] = float(np.min(d) / mol_scale)

    X = np.zeros((nrings, len(RING_FEATURE_NAMES)), dtype=float)
    X[:, 0] = np.array(ring_sizes, dtype=float)
    X[:, 1] = np.array(ring_planarity, dtype=float)
    X[:, 2] = np.array(radial_pos, dtype=float)
    X[:, 3] = np.array(fused_degree, dtype=float)
    X[:, 4] = np.array(nn_dist, dtype=float)
    X[:, 5] = np.array(ring_radius, dtype=float)
    return X


def fit_gaussian_model(
    X: np.ndarray,
    *,
    covariance_mode: str = "full",
    regularization: float = 5.0e-2,
) -> GaussianModel:
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("synthon feature matrix must be 2D with at least one row")
    if covariance_mode not in {"full", "diag"}:
        raise ValueError("covariance_mode must be 'full' or 'diag'")
    if regularization <= 0.0:
        raise ValueError("regularization must be positive")

    mean = X.mean(axis=0)
    if X.shape[0] == 1:
        cov = np.eye(X.shape[1], dtype=float) * regularization
    else:
        cov = np.cov(X, rowvar=False, ddof=1)
        cov = np.array(cov, dtype=float)
        if covariance_mode == "diag":
            cov = np.diag(np.diag(cov))
        cov = cov + regularization * np.eye(cov.shape[0], dtype=float)
    return GaussianModel(mean=mean, cov=cov, natoms=int(X.shape[0]))


def bhattacharyya_distance(a: GaussianModel, b: GaussianModel) -> float:
    sigma = 0.5 * (a.cov + b.cov)
    delta = (a.mean - b.mean).reshape(-1, 1)

    inv_sigma = np.linalg.pinv(sigma)
    term1 = 0.125 * float((delta.T @ inv_sigma @ delta)[0, 0])

    sign_s, logdet_s = np.linalg.slogdet(sigma)
    sign_a, logdet_a = np.linalg.slogdet(a.cov)
    sign_b, logdet_b = np.linalg.slogdet(b.cov)
    if sign_s <= 0 or sign_a <= 0 or sign_b <= 0:
        raise ValueError("non-positive definite covariance encountered")
    term2 = 0.5 * (logdet_s - 0.5 * (logdet_a + logdet_b))
    return float(term1 + term2)


def gaussian_similarity(a: GaussianModel, b: GaussianModel) -> float:
    db = bhattacharyya_distance(a, b)
    return float(np.exp(-db))


def compare_molecules(
    xyz_a: Path,
    xyz_b: Path,
    *,
    covariance_mode: str = "full",
    regularization: float = 5.0e-2,
    standardize: bool = True,
    include_ring_comparison: bool = True,
    ring_weight: float = 0.25,
):
    if not (0.0 <= ring_weight <= 1.0):
        raise ValueError("ring_weight must be in [0, 1]")

    Z_a, coords_a = _xyz_to_au(xyz_a)
    Z_b, coords_b = _xyz_to_au(xyz_b)

    X_a = synthon_feature_matrix(Z_a, coords_a)
    X_b = synthon_feature_matrix(Z_b, coords_b)
    R_a = ring_feature_matrix(Z_a, coords_a)
    R_b = ring_feature_matrix(Z_b, coords_b)

    if standardize:
        X_all = np.vstack((X_a, X_b))
        mu = X_all.mean(axis=0)
        sigma = X_all.std(axis=0)
        sigma = np.where(sigma < 1.0e-10, 1.0, sigma)
        X_a = (X_a - mu) / sigma
        X_b = (X_b - mu) / sigma
        if include_ring_comparison and (R_a.shape[0] > 0 or R_b.shape[0] > 0):
            if R_a.shape[0] > 0 and R_b.shape[0] > 0:
                R_all = np.vstack((R_a, R_b))
            elif R_a.shape[0] > 0:
                R_all = R_a
            else:
                R_all = R_b
            r_mu = R_all.mean(axis=0)
            r_sigma = R_all.std(axis=0)
            r_sigma = np.where(r_sigma < 1.0e-10, 1.0, r_sigma)
            if R_a.shape[0] > 0:
                R_a = (R_a - r_mu) / r_sigma
            if R_b.shape[0] > 0:
                R_b = (R_b - r_mu) / r_sigma

    model_a = fit_gaussian_model(
        X_a, covariance_mode=covariance_mode, regularization=regularization
    )
    model_b = fit_gaussian_model(
        X_b, covariance_mode=covariance_mode, regularization=regularization
    )

    db = bhattacharyya_distance(model_a, model_b)
    sim = gaussian_similarity(model_a, model_b)

    ring_db = 0.0
    ring_sim = 1.0
    ring_model_a = None
    ring_model_b = None
    effective_ring_weight = 0.0
    if include_ring_comparison and ring_weight > 0.0:
        effective_ring_weight = ring_weight
        if R_a.shape[0] == 0 and R_b.shape[0] == 0:
            ring_db = 0.0
            ring_sim = 1.0
            effective_ring_weight = 0.0
        elif R_a.shape[0] == 0 or R_b.shape[0] == 0:
            ring_db = 4.0
            ring_sim = float(np.exp(-ring_db))
        else:
            ring_model_a = fit_gaussian_model(
                R_a, covariance_mode=covariance_mode, regularization=regularization
            )
            ring_model_b = fit_gaussian_model(
                R_b, covariance_mode=covariance_mode, regularization=regularization
            )
            ring_db = bhattacharyya_distance(ring_model_a, ring_model_b)
            ring_sim = gaussian_similarity(ring_model_a, ring_model_b)

    combined_sim = float((1.0 - effective_ring_weight) * sim + effective_ring_weight * ring_sim)
    combined_db = float(-np.log(max(combined_sim, 1.0e-15)))
    syn_explain = _feature_explain(model_a, model_b, FEATURE_NAMES)
    if ring_model_a is not None and ring_model_b is not None:
        ring_explain = _feature_explain(ring_model_a, ring_model_b, RING_FEATURE_NAMES)
    else:
        ring_explain = []

    return {
        "xyz_a": str(xyz_a),
        "xyz_b": str(xyz_b),
        "natoms_a": model_a.natoms,
        "natoms_b": model_b.natoms,
        "feature_names": list(FEATURE_NAMES),
        "covariance_mode": covariance_mode,
        "regularization": float(regularization),
        "standardize": bool(standardize),
        "include_ring_comparison": bool(include_ring_comparison),
        "ring_weight": float(ring_weight),
        "effective_ring_weight": float(effective_ring_weight),
        "bhattacharyya_distance": db,
        "similarity_exp_minus_db": sim,
        "ring_bhattacharyya_distance": float(ring_db),
        "ring_similarity_exp_minus_db": float(ring_sim),
        "similarity_combined": combined_sim,
        "combined_distance_neglog": combined_db,
        "mean_a": model_a.mean.tolist(),
        "mean_b": model_b.mean.tolist(),
        "nrings_a": int(R_a.shape[0]),
        "nrings_b": int(R_b.shape[0]),
        "explain": {
            "synthon_feature_terms": syn_explain,
            "ring_feature_terms": ring_explain,
        },
    }


def compare_directory_sets(
    query_xyz: list[Path],
    library_xyz: list[Path],
    *,
    covariance_mode: str = "full",
    regularization: float = 5.0e-2,
    standardize: bool = True,
    include_ring_comparison: bool = True,
    ring_weight: float = 0.25,
    top_k: int | None = None,
):
    reports = []
    for query in sorted(query_xyz):
        rep = compare_against_library(
            query,
            library_xyz,
            covariance_mode=covariance_mode,
            regularization=regularization,
            standardize=standardize,
            include_ring_comparison=include_ring_comparison,
            ring_weight=ring_weight,
        )
        if top_k is not None:
            rep["ranking"] = rep["ranking"][: max(0, int(top_k))]
        reports.append(rep)
    return {
        "queries": [str(p) for p in sorted(query_xyz)],
        "library_size_input": len(library_xyz),
        "nqueries": len(reports),
        "reports": reports,
    }


def compare_against_library(
    query_xyz: Path,
    library_xyz: list[Path],
    *,
    covariance_mode: str = "full",
    regularization: float = 5.0e-2,
    standardize: bool = True,
    include_ring_comparison: bool = True,
    ring_weight: float = 0.25,
):
    results = []
    skipped = []
    for candidate in library_xyz:
        if candidate.resolve() == query_xyz.resolve():
            continue
        try:
            comp = compare_molecules(
                query_xyz,
                candidate,
                covariance_mode=covariance_mode,
                regularization=regularization,
                standardize=standardize,
                include_ring_comparison=include_ring_comparison,
                ring_weight=ring_weight,
            )
        except Exception as exc:  # keep batch mode robust across heterogeneous libraries
            skipped.append({"xyz": str(candidate), "error": str(exc)})
            continue
        results.append(
            {
                "xyz": comp["xyz_b"],
                "natoms": comp["natoms_b"],
                "similarity_exp_minus_db": comp["similarity_exp_minus_db"],
                "bhattacharyya_distance": comp["bhattacharyya_distance"],
                "ring_similarity_exp_minus_db": comp["ring_similarity_exp_minus_db"],
                "ring_bhattacharyya_distance": comp["ring_bhattacharyya_distance"],
                "similarity_combined": comp["similarity_combined"],
                "combined_distance_neglog": comp["combined_distance_neglog"],
                "effective_ring_weight": comp["effective_ring_weight"],
                "nrings": comp["nrings_b"],
            }
        )
    results.sort(key=lambda r: r["similarity_combined"], reverse=True)
    return {
        "query_xyz": str(query_xyz),
        "library_size": len(results),
        "feature_names": list(FEATURE_NAMES),
        "ring_feature_names": list(RING_FEATURE_NAMES),
        "covariance_mode": covariance_mode,
        "regularization": float(regularization),
        "standardize": bool(standardize),
        "include_ring_comparison": bool(include_ring_comparison),
        "ring_weight": float(ring_weight),
        "ranking": results,
        "skipped": skipped,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Molecule similarity with synthon Gaussian models.")
    ap.add_argument("--xyz-a", default=None, help="First XYZ file (pair mode)")
    ap.add_argument("--xyz-b", default=None, help="Second XYZ file (pair mode)")
    ap.add_argument("--query-xyz", default=None, help="Query XYZ (library mode)")
    ap.add_argument("--query-dir", default=None, help="Directory of queries (batch mode)")
    ap.add_argument(
        "--library-dir",
        default=None,
        help="Directory containing library XYZ files (library mode)",
    )
    ap.add_argument(
        "--library-glob",
        default="*.xyz",
        help="Glob pattern for library files (default: *.xyz)",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Print only top-k matches in library mode",
    )
    ap.add_argument(
        "--covariance-mode",
        choices=("full", "diag"),
        default="full",
        help="Covariance model for synthon Gaussian fitting",
    )
    ap.add_argument(
        "--regularization",
        type=float,
        default=5.0e-2,
        help="Diagonal regularization added to covariance matrices",
    )
    ap.add_argument(
        "--json-out",
        default=None,
        help="Optional output path for JSON report",
    )
    ap.add_argument(
        "--csv-out",
        default=None,
        help="Optional output path for CSV report (library or batch mode)",
    )
    ap.add_argument(
        "--no-standardize",
        action="store_true",
        help="Disable global feature standardization before Gaussian fitting",
    )
    ap.add_argument(
        "--ring-weight",
        type=float,
        default=0.25,
        help="Weight of ring-based comparison in final combined similarity [0,1]",
    )
    ap.add_argument(
        "--no-ring-comparison",
        action="store_true",
        help="Disable ring-aware comparison term",
    )
    args = ap.parse_args(argv)
    standardize = not args.no_standardize
    include_ring_comparison = not args.no_ring_comparison

    pair_mode = args.xyz_a is not None and args.xyz_b is not None
    lib_mode = args.query_xyz is not None and args.library_dir is not None
    batch_mode = args.query_dir is not None and args.library_dir is not None
    modes = [pair_mode, lib_mode, batch_mode]
    if sum(bool(x) for x in modes) != 1:
        raise SystemExit(
            "Use exactly one mode: pair (--xyz-a --xyz-b), "
            "library (--query-xyz --library-dir), or batch (--query-dir --library-dir)."
        )

    if pair_mode:
        result = compare_molecules(
            Path(args.xyz_a),
            Path(args.xyz_b),
            covariance_mode=args.covariance_mode,
            regularization=args.regularization,
            standardize=standardize,
            include_ring_comparison=include_ring_comparison,
            ring_weight=args.ring_weight,
        )
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(result, indent=2) + "\n")
        print(f"Synthon Gaussian similarity: {result['similarity_exp_minus_db']:.6f}")
        print(f"Bhattacharyya distance:     {result['bhattacharyya_distance']:.6f}")
        print(f"Ring Gaussian similarity:   {result['ring_similarity_exp_minus_db']:.6f}")
        print(f"Combined similarity:        {result['similarity_combined']:.6f}")
        print(f"Molecule A atoms:           {result['natoms_a']}")
        print(f"Molecule B atoms:           {result['natoms_b']}")
        print(f"Molecule A/B rings:         {result['nrings_a']} / {result['nrings_b']}")
        top_syn = result["explain"]["synthon_feature_terms"][:3]
        if top_syn:
            print("Top synthon contributors:")
            for row in top_syn:
                print(
                    f"  - {row['feature']}: total={row['total_term']:.6f} "
                    f"(mean={row['mean_term']:.6f}, var={row['variance_term']:.6f})"
                )
        return

    if batch_mode:
        query_dir = Path(args.query_dir)
        lib_dir = Path(args.library_dir)
        queries = sorted(query_dir.glob(args.library_glob))
        library = sorted(lib_dir.glob(args.library_glob))
        if not queries:
            raise SystemExit("No query XYZ files found with current --query-dir/--library-glob.")
        if not library:
            raise SystemExit(
                "No library XYZ files found with current --library-dir/--library-glob."
            )
        result = compare_directory_sets(
            queries,
            library,
            covariance_mode=args.covariance_mode,
            regularization=args.regularization,
            standardize=standardize,
            include_ring_comparison=include_ring_comparison,
            ring_weight=args.ring_weight,
            top_k=args.top_k,
        )
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(result, indent=2) + "\n")
        if args.csv_out:
            with open(args.csv_out, "w", newline="", encoding="utf-8") as fh:
                wr = csv.writer(fh)
                wr.writerow(
                    [
                        "query_xyz",
                        "candidate_xyz",
                        "sim_comb",
                        "sim_syn",
                        "sim_ring",
                        "db_comb",
                        "db_syn",
                        "db_ring",
                    ]
                )
                for rep in result["reports"]:
                    q = rep["query_xyz"]
                    for row in rep["ranking"]:
                        wr.writerow(
                            [
                                q,
                                row["xyz"],
                                row["similarity_combined"],
                                row["similarity_exp_minus_db"],
                                row["ring_similarity_exp_minus_db"],
                                row["combined_distance_neglog"],
                                row["bhattacharyya_distance"],
                                row["ring_bhattacharyya_distance"],
                            ]
                        )
        print(f"Batch queries: {result['nqueries']}")
        for rep in result["reports"]:
            top = rep["ranking"][0] if rep["ranking"] else None
            if top is None:
                print(f"- {rep['query_xyz']}: no matches")
            else:
                print(
                    f"- {rep['query_xyz']}: best={top['xyz']} "
                    f"sim_comb={top['similarity_combined']:.6f}"
                )
        return

    query = Path(args.query_xyz)
    lib_dir = Path(args.library_dir)
    library = sorted(lib_dir.glob(args.library_glob))
    if not library:
        raise SystemExit("No library XYZ files found with current --library-dir/--library-glob.")
    result = compare_against_library(
        query,
        library,
        covariance_mode=args.covariance_mode,
        regularization=args.regularization,
        standardize=standardize,
        include_ring_comparison=include_ring_comparison,
        ring_weight=args.ring_weight,
    )
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2) + "\n")
    if args.csv_out:
        with open(args.csv_out, "w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow(
                [
                    "query_xyz",
                    "candidate_xyz",
                    "sim_comb",
                    "sim_syn",
                    "sim_ring",
                    "db_comb",
                    "db_syn",
                    "db_ring",
                ]
            )
            for row in result["ranking"]:
                wr.writerow(
                    [
                        result["query_xyz"],
                        row["xyz"],
                        row["similarity_combined"],
                        row["similarity_exp_minus_db"],
                        row["ring_similarity_exp_minus_db"],
                        row["combined_distance_neglog"],
                        row["bhattacharyya_distance"],
                        row["ring_bhattacharyya_distance"],
                    ]
                )
    ranking = result["ranking"]
    if args.top_k is not None:
        ranking = ranking[: max(0, args.top_k)]
    print(f"Query: {result['query_xyz']}")
    print(f"Compared molecules: {result['library_size']}")
    if result["skipped"]:
        print(f"Skipped molecules: {len(result['skipped'])}")
    for i, row in enumerate(ranking, start=1):
        print(
            f"{i:3d}. sim_comb={row['similarity_combined']:.6f}  "
            f"sim_syn={row['similarity_exp_minus_db']:.6f}  "
            f"sim_ring={row['ring_similarity_exp_minus_db']:.6f}  {row['xyz']}"
        )


if __name__ == "__main__":
    main()
