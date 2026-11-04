"""ORACLE/ESALC scan acquisition for the tunneling DVR workflow."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence

import numpy as np

@dataclass(frozen=True)
class ESALCScanRequest:
    repo_root: Path
    oracle_xyzin: Path
    coordinates: tuple[str, ...]
    outdir: Path
    backend: str = "matrix_gfn2"
    charge: int = 0
    multiplicity: int = 1
    step: float = 0.10
    points_each_side: int = 6
    workers: int = 1
    levels: int = 12
    solver: str = "fourier"
    boundary: str = "periodic"
    resume: bool = True
    adaptive: bool = False
    interpolation_factor: int = 1
    vibin: Path | None = None
    execution_backend: str = "auto"
    multimode: bool = False
    multimode_surface: Path | None = None
    remote_host: str | None = None
    torsional_symmetry_order: int = 1
    nuclear_spin_weights: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not 1 <= len(self.coordinates) <= 6:
            raise ValueError("ESALC scan supports one to six coordinates")
        if len(self.coordinates) > 2 and not self.multimode:
            raise ValueError("more than two coordinates require multimode=True")
        if self.step <= 0.0 or self.points_each_side < 1 or self.workers < 1:
            raise ValueError("step, points_each_side, and workers must be positive")
        if self.interpolation_factor < 1:
            raise ValueError("interpolation_factor must be at least one")
        if self.torsional_symmetry_order < 1:
            raise ValueError("torsional_symmetry_order must be positive")
        if self.nuclear_spin_weights and len(self.nuclear_spin_weights) != self.torsional_symmetry_order:
            raise ValueError("one nuclear-spin weight is required per torsional sector")


def run_esalc_scan(request: ESALCScanRequest) -> Path:
    """Acquire an ESALC scan and write a DVR-ready result manifest."""
    request.outdir.mkdir(parents=True, exist_ok=True)
    scan_dir = request.outdir / "scan"
    scan_dir.mkdir(parents=True, exist_ok=True)
    from .advanced_workflow import (
        adaptive_cells_2d,
        interpolation_error_2d,
        plan_execution,
        write_interactive_viewer,
        write_multimode_contract,
        measure_autotune,
        write_oracle_sync_manifest,
    )

    execution = plan_execution(request.execution_backend, request.workers)
    if len(request.coordinates) > 2:
        contract = write_multimode_contract(
            request.outdir,
            oracle_xyzin=request.oracle_xyzin,
            coordinates=request.coordinates,
        )
        result_path = request.outdir / "esalc_scan_result.json"
        multimode_result = {
            "schema": "matrix.tunneling.esalc_scan_result.v2",
            "dimension": len(request.coordinates),
            "coordinates": list(request.coordinates),
            "multimode_contract": str(contract),
            "execution": execution.__dict__,
            "autotune": measure_autotune(points=0, workers=request.workers, interpolation_factor=request.interpolation_factor, backend=request.execution_backend),
        }
        if request.multimode_surface is not None:
            from .multimode import solve_tensor_dvr_from_surface, write_tensor_dvr_result
            multimode_result["tensor_dvr_result"] = str(write_tensor_dvr_result(
                request.outdir / "multimode_tensor_dvr.json",
                solve_tensor_dvr_from_surface(request.multimode_surface, request.coordinates, levels=request.levels),
            ))
        result_path.write_text(json.dumps(multimode_result, indent=2) + "\n", encoding="utf-8")
        sync = write_oracle_sync_manifest(
            request.outdir,
            oracle_xyzin=request.oracle_xyzin,
            coordinates=request.coordinates,
            artifacts={"multimode_contract": str(contract), "result": str(result_path)},
        )
        multimode_result["oracle_sync"] = str(sync)
        result_path.write_text(json.dumps(multimode_result, indent=2) + "\n", encoding="utf-8")
        return result_path
    if len(request.coordinates) == 1:
        source = _run_1d_scan(request, scan_dir)
        geometry = None
        surface_source = scan_dir / "esalc_scan_surface.csv"
        if not surface_source.is_file():
            surface_source = source
        result = {
            "schema": "matrix.tunneling.esalc_scan_result.v1",
            "dimension": 1,
            "source": str(source),
            "coordinates": list(request.coordinates),
            "backend": request.backend,
            "workers": request.workers,
            "resume": request.resume,
            "adaptive": request.adaptive,
            "execution": execution.__dict__,
        }
    else:
        source, geometry = _run_2d_scan(request, scan_dir)
        surface_source = source
        dvr_result = _run_2d_dvr(request, source, geometry)
        result = {
            "schema": "matrix.tunneling.esalc_scan_result.v1",
            "dimension": 2,
            "source": str(source),
            "geometry_grid": str(geometry),
            "coordinates": list(request.coordinates),
            "backend": request.backend,
            "workers": request.workers,
            "resume": request.resume,
            "adaptive": request.adaptive,
            "execution": execution.__dict__,
            "dvr_result": str(dvr_result),
        }
    from .rovib_diagnostics import (
        validate_scan_artifacts,
        write_human_report,
        write_lock_export,
    )

    diagnostics = validate_scan_artifacts(
        source=surface_source,
        dimension=len(request.coordinates),
        coordinates=request.coordinates,
        geometry_grid=geometry,
    )
    result["diagnostics"] = diagnostics
    result["autotune"] = measure_autotune(
        points=(len(request.coordinates) and (2 * request.points_each_side + 1) ** len(request.coordinates)),
        workers=request.workers,
        interpolation_factor=request.interpolation_factor,
        backend=request.execution_backend,
    )
    result["uncertainty"] = _surface_uncertainty(surface_source)
    result["adaptive_cells"] = _write_adaptive_cells(
        surface_source, request.outdir, adaptive_cells_2d
    )
    result["interpolation_error"] = _surface_interpolation_error(
        surface_source, interpolation_error_2d
    )
    from .lam_metric import lam_rotation_surface_from_scan_files
    from matrix_rovib import write_lam_rotation_surface

    lock_surface_path = request.outdir / "lock_lam_rotation_surface.npz"
    try:
        lock_surface = lam_rotation_surface_from_scan_files(
            surface_source,
            source if len(request.coordinates) == 1 else geometry,
            request.coordinates,
            periodic=(request.boundary == "periodic",) * len(request.coordinates),
            torsional_symmetry_order=request.torsional_symmetry_order,
            nuclear_spin_weights=request.nuclear_spin_weights,
        )
        result["lock_surface"] = str(
            write_lam_rotation_surface(lock_surface_path, lock_surface)
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        result["lock_surface"] = {
            "status": "unavailable",
            "reason": str(exc),
        }
    coriolis_path = _write_coriolis_export(request, request.outdir)
    result["coriolis"] = str(coriolis_path) if coriolis_path is not None else {
        "status": "deferred_to_LOCK",
        "reason": "supply --vibin to evaluate normal-mode Coriolis entries",
    }
    result["lock_input"] = str(
        write_lock_export(
            request.outdir,
            source=surface_source,
            diagnostics=diagnostics,
            dvr_result=Path(result["dvr_result"]) if result.get("dvr_result") else None,
            coriolis_file=coriolis_path,
            lock_surface=(
                Path(result["lock_surface"])
                if isinstance(result.get("lock_surface"), str)
                else None
            ),
        )
    )
    result_path = request.outdir / "esalc_scan_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if request.remote_host:
        from .remote_workflow import RemoteRunRequest, build_remote_bundle, write_remote_plan
        bundle = build_remote_bundle(
            request.outdir,
            request.outdir.parent / f"{request.outdir.name}-remote" / "esalc_run",
        )
        result["remote_plan"] = str(write_remote_plan(
            request.outdir, RemoteRunRequest(host=request.remote_host), bundle
        ))
    result["report"] = str(write_human_report(request.outdir, result=result, diagnostics=diagnostics))
    from .scientific_report import write_scientific_report
    result["scientific_report"] = write_scientific_report(request.outdir, result=result)
    result["viewer"] = str(
        write_interactive_viewer(
            request.outdir, result_json=result_path, surface=surface_source
        )
    )
    result["oracle_sync"] = str(
        write_oracle_sync_manifest(
            request.outdir,
            oracle_xyzin=request.oracle_xyzin,
            coordinates=request.coordinates,
            artifacts={key: value for key, value in result.items() if key in {
                "source", "geometry_grid", "dvr_result", "lock_surface",
                "lock_input", "viewer", "coriolis"
            }},
        )
    )
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result_path


def _run_1d_scan(request: ESALCScanRequest, scan_dir: Path) -> Path:
    from matrix_link import (
        PESExplorationPolicy,
        coordinate_direction_from_gic,
        prepare_coordinate_scan,
    )
    from matrix_chem.xyzin_geometry import read_xyzin_geometry

    direction = coordinate_direction_from_gic(request.oracle_xyzin, request.coordinates[0])
    displacements = tuple(
        i * request.step for i in range(-request.points_each_side, request.points_each_side + 1)
    )
    points = prepare_coordinate_scan(
        request.oracle_xyzin,
        direction,
        displacements,
        run_dir=scan_dir,
        exploration_policy=PESExplorationPolicy(),
    )
    manifest = scan_dir / "scan_manifest.json"
    cache = _load_cached_points(scan_dir) if request.resume else {}
    results = _evaluate_points(request, scan_dir, points, PESExplorationPolicy())
    if request.adaptive:
        energies = np.asarray([float(item.energy_hartree) for item in results], dtype=float)
        curvatures = np.abs(np.diff(energies, n=2))
        scale = max(float(np.ptp(energies)), 1.0e-12)
        candidates = [
            0.5 * (points[index].displacement + points[index + 1].displacement)
            for index, curvature in enumerate(curvatures)
            if curvature > 0.05 * scale
        ]
        if candidates:
            refined = tuple(sorted({point.displacement for point in points} | set(candidates)))
            points = prepare_coordinate_scan(
                request.oracle_xyzin,
                direction,
                refined,
                run_dir=scan_dir,
                exploration_policy=PESExplorationPolicy(),
            )
            results = _evaluate_points(request, scan_dir, points, PESExplorationPolicy())
    atoms = read_xyzin_geometry(request.oracle_xyzin).atoms
    output = scan_dir / "esalc_scan.xyz"
    surface = scan_dir / "esalc_scan_surface.csv"
    surface_rows: list[tuple[float, float, tuple[float, float, float]]] = []
    with output.open("w", encoding="utf-8") as handle:
        for point, result in zip(points, results, strict=True):
            if result.energy_hartree is None:
                raise RuntimeError(f"ESALC point {point.index} has no energy: {result.message}")
            coordinates = result.backend_coordinates_angstrom
            if coordinates is None:
                coordinates = point.coordinates_angstrom
            surface_rows.append(
                (
                    point.displacement,
                    float(result.energy_hartree),
                    _rotational_constants_mhz(atoms, coordinates),
                )
            )
            handle.write(
                f"{len(atoms)}\nenergy_hartree={result.energy_hartree:.16e} "
                f"{direction.label}={point.displacement:.16e} backend={request.backend}\n"
            )
            for atom, xyz in zip(atoms, coordinates, strict=True):
                handle.write(f"{atom:2s} {xyz[0]: .12f} {xyz[1]: .12f} {xyz[2]: .12f}\n")
    with surface.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([direction.label, "energy_hartree", "A_MHz", "B_MHz", "C_MHz"])
        for coordinate, energy, abc in surface_rows:
            writer.writerow([coordinate, energy, *abc])
    if len(surface_rows) >= 3:
        q = np.asarray([row[0] for row in surface_rows], dtype=float)
        abc = np.asarray([row[2] for row in surface_rows], dtype=float)
        d1 = np.gradient(abc, q, axis=0, edge_order=2 if len(q) > 2 else 1)
        d2 = np.gradient(d1, q, axis=0, edge_order=2 if len(q) > 2 else 1)
        with (scan_dir / "rotational_derivatives.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                direction.label, "dA_dq", "dB_dq", "dC_dq",
                "d2A_dq2", "d2B_dq2", "d2C_dq2",
            ])
            for index, coordinate in enumerate(q):
                writer.writerow([coordinate, *d1[index], *d2[index]])
    manifest.write_text(
        json.dumps(
            {
                "schema": "matrix.tunneling.esalc_scan_manifest.v1",
                "dimension": 1,
                "oracle_xyzin": str(request.oracle_xyzin),
                "coordinate": direction.label,
                "points": [point.displacement for point in points],
                "cached_points": sorted(cache),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def _run_2d_scan(request: ESALCScanRequest, scan_dir: Path) -> tuple[Path, Path]:
    from matrix_link import PESExplorationPolicy
    from matrix_link.scan import coordinate_directions_from_gics
    from matrix_chem.xyzin_geometry import read_xyzin_geometry
    from matrix_link.scan import ScanPoint, prepare_pes_exploration_geometry

    directions = coordinate_directions_from_gics(request.oracle_xyzin, request.coordinates)
    geometry = read_xyzin_geometry(request.oracle_xyzin)
    values = tuple(i * request.step for i in range(-request.points_each_side, request.points_each_side + 1))
    policy = PESExplorationPolicy()
    points: list[ScanPoint] = []
    index = 0
    for q1 in values:
        for q2 in values:
            raw = geometry.coordinates_angstrom + q1 * directions[0].vector_angstrom.reshape((-1, 3))
            raw = raw + q2 * directions[1].vector_angstrom.reshape((-1, 3))
            coords, symmetry = prepare_pes_exploration_geometry(
                request.oracle_xyzin, geometry.atoms, raw, policy=policy
            )
            points.append(
                ScanPoint(
                    index=index,
                    displacement=q1,
                    coordinates_angstrom=coords,
                    coordinate_key=f"{request.coordinates[0]},{request.coordinates[1]}",
                    # A 2D grid point is not a finite-difference center for
                    # either coordinate; keep the displacement metadata but
                    # disable the 1D finite-difference sign contract.
                    side="unspecified",
                    status="prepared",
                    symmetry_analyzed=True,
                    point_group=symmetry.point_group,
                    retained_group=policy.retained_group,
                )
            )
            index += 1
    results = _evaluate_points(request, scan_dir, points, policy)
    if request.adaptive:
        energy_grid = np.asarray(
            [float(item.energy_hartree) for item in results], dtype=float
        ).reshape(len(values), len(values))
        from .advanced_workflow import adaptive_cells_2d

        local_cells = adaptive_cells_2d(energy_grid)
        if local_cells:
            local_points: list[ScanPoint] = []
            for local_index, cell in enumerate(local_cells[: min(16, len(local_cells))], start=len(points)):
                i = int(cell["i"])
                j = int(cell["j"])
                q1 = 0.5 * (values[i + 1] + values[i + 2])
                q2 = 0.5 * (values[j + 1] + values[j + 2])
                raw = geometry.coordinates_angstrom + q1 * directions[0].vector_angstrom.reshape((-1, 3))
                raw = raw + q2 * directions[1].vector_angstrom.reshape((-1, 3))
                coords, symmetry = prepare_pes_exploration_geometry(
                    request.oracle_xyzin, geometry.atoms, raw, policy=policy
                )
                local_points.append(ScanPoint(
                    index=local_index,
                    displacement=q1,
                    coordinates_angstrom=coords,
                    coordinate_key=f"{request.coordinates[0]},{request.coordinates[1]}",
                    side="unspecified",
                    status="adaptive_local",
                    symmetry_analyzed=True,
                    point_group=symmetry.point_group,
                    retained_group=policy.retained_group,
                ))
            local_results = _evaluate_points(replace(request, resume=False), scan_dir, local_points, policy)
            with (scan_dir / "adaptive_local_points.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow([request.coordinates[0], request.coordinates[1], "energy_hartree", "curvature_score"])
                for point, local_result, cell in zip(local_points, local_results, local_cells, strict=True):
                    writer.writerow([point.displacement, values[int(cell["j"]) + 1], local_result.energy_hartree, cell["score"]])
        curvature = max(
            float(np.max(np.abs(np.diff(energy_grid, n=2, axis=0)))) if len(values) > 2 else 0.0,
            float(np.max(np.abs(np.diff(energy_grid, n=2, axis=1)))) if len(values) > 2 else 0.0,
        )
        if curvature > 0.05 * max(float(np.ptp(energy_grid)), 1.0e-12):
            refined_request = replace(
                request,
                step=request.step / 2.0,
                points_each_side=request.points_each_side * 2,
                adaptive=False,
                resume=False,
            )
            return _run_2d_scan(refined_request, scan_dir)
    csv_path = scan_dir / "esalc_grid.csv"
    geometry_path = scan_dir / "esalc_geometry_grid.xyz"
    rotational_rows: list[tuple[float, float, tuple[float, float, float]]] = []
    with csv_path.open("w", newline="", encoding="utf-8") as csv_handle, geometry_path.open("w", encoding="utf-8") as xyz_handle:
        writer = csv.writer(csv_handle)
        writer.writerow([
            request.coordinates[0], request.coordinates[1], "energy_hartree",
            "A_MHz", "B_MHz", "C_MHz",
        ])
        for point, result, q1 in zip(points, results, [q1 for q1 in values for _ in values], strict=True):
            q2 = values[point.index % len(values)]
            if result.energy_hartree is None:
                raise RuntimeError(f"ESALC grid point {point.index} has no energy: {result.message}")
            coords = (
                result.backend_coordinates_angstrom
                if result.backend_coordinates_angstrom is not None
                else point.coordinates_angstrom
            )
            abc = _rotational_constants_mhz(geometry.atoms, coords)
            writer.writerow([q1, q2, result.energy_hartree, *abc])
            rotational_rows.append((q1, q2, abc))
            xyz_handle.write(f"{len(geometry.atoms)}\nq1={q1:.12g} q2={q2:.12g}\n")
            for atom, xyz in zip(geometry.atoms, coords, strict=True):
                xyz_handle.write(f"{atom:2s} {xyz[0]: .12f} {xyz[1]: .12f} {xyz[2]: .12f}\n")
    _write_rotational_derivatives(scan_dir / "rotational_derivatives.csv", rotational_rows, values)
    return csv_path, geometry_path


def _run_2d_dvr(request: ESALCScanRequest, csv_path: Path, geometry_path: Path) -> Path:
    dvr_dir = request.outdir / "dvr"
    fig_dir = dvr_dir / "figures"
    dvr_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    script = request.repo_root / "engines" / "puckering_dvr" / "scripts" / "mw_path_dvr.py"
    result_path = dvr_dir / "esalc_2d_dvr_result.json"
    args = [
        sys.executable,
        str(script),
        "--grid2d-csv", str(csv_path),
        "--grid2d-geom-xyz", str(geometry_path),
        "--metric-mode", "esalc",
        "--esalc-xyzin", str(request.oracle_xyzin),
        "--esalc-q1", request.coordinates[0],
        "--esalc-q2", request.coordinates[1],
        "--q1-key", request.coordinates[0],
        "--q2-key", request.coordinates[1],
        "--energy-key", "energy_hartree",
        "--energy-unit", "hartree",
        "--boundary1", request.boundary,
        "--boundary2", request.boundary,
        "--solver", request.solver,
        "--levels", str(request.levels),
        "--interpolation-factor", str(request.interpolation_factor),
        "--property", "A_MHz",
        "--property", "B_MHz",
        "--property", "C_MHz",
        "--outdir", str(dvr_dir),
        "--figdir", str(fig_dir),
        "--prefix", "esalc_2d",
    ]
    completed = subprocess.run(args, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "2D ESALC DVR failed")
    result_path.write_text(
        json.dumps({"command": args, "stdout": completed.stdout, "stderr": completed.stderr}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return result_path


def _rotational_constants_mhz(atoms: Sequence[str], coordinates: object) -> tuple[float, float, float]:
    from matrix_chem import atomic_mass
    from matrix_chem.topology.elements import atomic_number

    coords = np.asarray(coordinates, dtype=float)
    masses = np.asarray([atomic_mass(atomic_number(str(atom)) or 0) for atom in atoms], dtype=float)
    center = np.sum(masses[:, None] * coords, axis=0) / np.sum(masses)
    centered = coords - center
    inertia = np.zeros((3, 3), dtype=float)
    for mass, vector in zip(masses, centered, strict=True):
        inertia += mass * (float(vector @ vector) * np.eye(3) - np.outer(vector, vector))
    moments = np.linalg.eigvalsh(0.5 * (inertia + inertia.T))
    factor = 505379.0090099937
    return tuple(float(factor / moment) for moment in moments)


def _write_coriolis_export(request: ESALCScanRequest, outdir: Path) -> Path | None:
    """Export standard normal-mode Coriolis entries when a vibin is supplied."""
    vibin = getattr(request, "vibin", None)
    if vibin is None:
        return None
    from matrix_chem.xyzin_geometry import read_xyzin_geometry
    from matrix_rovib.coriolis import compute_coriolis_from_vibin

    geometry = read_xyzin_geometry(request.oracle_xyzin)
    abc = _rotational_constants_mhz(geometry.atoms, geometry.coordinates_angstrom)
    result = compute_coriolis_from_vibin(vibin, *abc, units="MHz")
    path = outdir / "coriolis_parameters.json"
    path.write_text(
        json.dumps(
            {
                "schema": "matrix.lock.coriolis_parameters.v1",
                "source_vibin": str(vibin),
                "A_MHz": abc[0], "B_MHz": abc[1], "C_MHz": abc[2],
                "threshold_cm1": result.threshold_cm1,
                "entries": [entry.__dict__ for entry in result.entries],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_rotational_derivatives(
    path: Path,
    rows: Sequence[tuple[float, float, tuple[float, float, float]]],
    values: Sequence[float],
) -> None:
    """Write finite-difference dA/dq, dB/dq, dC/dq for the 2D scan."""
    n = len(values)
    if n < 3 or len(rows) != n * n:
        return
    grid = np.asarray([row[2] for row in rows], dtype=float).reshape(n, n, 3)
    spacing = float(values[1] - values[0])
    d1 = np.gradient(grid, spacing, axis=0, edge_order=2 if n > 2 else 1)
    d2 = np.gradient(grid, spacing, axis=1, edge_order=2 if n > 2 else 1)
    d11 = np.gradient(d1, spacing, axis=0, edge_order=2 if n > 2 else 1)
    d22 = np.gradient(d2, spacing, axis=1, edge_order=2 if n > 2 else 1)
    d12 = np.gradient(d1, spacing, axis=1, edge_order=2 if n > 2 else 1)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "q1", "q2", "dA_dq1", "dB_dq1", "dC_dq1", "dA_dq2", "dB_dq2", "dC_dq2",
            "d2A_dq1q1", "d2B_dq1q1", "d2C_dq1q1", "d2A_dq2q2", "d2B_dq2q2", "d2C_dq2q2",
            "d2A_dq1q2", "d2B_dq1q2", "d2C_dq1q2",
        ])
        for i, q1 in enumerate(values):
            for j, q2 in enumerate(values):
                writer.writerow([
                    q1, q2, *d1[i, j], *d2[i, j],
                    *d11[i, j], *d22[i, j], *d12[i, j],
                ])


def _load_cached_points(scan_dir: Path) -> dict[str, object]:
    path = scan_dir / "esalc_scan_cache.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _evaluate_points(
    request: ESALCScanRequest,
    scan_dir: Path,
    points: Sequence[object],
    policy: object,
) -> tuple[object, ...]:
    """Evaluate only missing points and reuse valid LINK point results."""
    from matrix_link import PESExplorationPolicy, QMScanBackend, run_qm_scan_points
    from matrix_link.scan import PointEvaluationResult, ScanPoint, point_result_to_json, read_point_result
    from matrix_chem.xyzin_geometry import read_xyzin_geometry
    from .advanced_workflow import geometry_cache_key, parallel_map, plan_execution
    from .cache_store import export_parquet, get_cache, open_cache, put_cache

    typed_points = tuple(points)
    cached: dict[int, PointEvaluationResult] = {}
    missing: list[ScanPoint] = []
    cache_dir = scan_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    sqlite_cache = open_cache(scan_dir / "esalc_cache.sqlite")
    atoms = read_xyzin_geometry(request.oracle_xyzin).atoms
    for point in typed_points:
        result_path = scan_dir / f"point_{point.index:04d}" / "point.json"
        cache_path = cache_dir / (
            geometry_cache_key(
                point.coordinates_angstrom,
                atoms=atoms,
                backend=request.backend,
                charge=request.charge,
                multiplicity=request.multiplicity,
                coordinate_labels=request.coordinates,
            )
            + ".json"
        )
        candidates = (cache_path, result_path)
        if request.resume:
            try:
                candidate = next((path for path in candidates if path.is_file()), None)
                if candidate is None:
                    cached_payload = get_cache(sqlite_cache, cache_path.stem)
                    if cached_payload is not None:
                        candidate = cache_path
                        candidate.write_text(json.dumps(cached_payload), encoding="utf-8")
                if candidate is not None:
                    result = read_point_result(candidate, point=point)
                    if (
                        result.status == "completed"
                        and result.energy_hartree is not None
                        and abs(result.displacement - point.displacement) <= 1.0e-12
                    ):
                        cached[point.index] = result
                        continue
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
        missing.append(point)
    if missing:
        execution = plan_execution(request.execution_backend, request.workers)
        backend = QMScanBackend(
            name=request.backend,
            charge=request.charge,
            multiplicity=request.multiplicity,
            processors=request.workers,
            device="cuda" if execution.selected == "gpu" else None,
            properties=("energy",),
        )
        if request.workers > 1 and len(missing) > 1:
            # Each worker owns its resident calculator. This avoids sharing
            # mutable SCF state while still parallelising independent points.
            def evaluate_one(point: ScanPoint):
                return run_qm_scan_points(
                    request.oracle_xyzin,
                    (point,),
                    backend,
                    run_dir=scan_dir,
                    exploration_policy=policy if isinstance(policy, PESExplorationPolicy) else None,
                )[0]

            fresh = parallel_map(evaluate_one, missing, execution)
        else:
            fresh = run_qm_scan_points(
                request.oracle_xyzin,
                missing,
                backend,
                run_dir=scan_dir,
                exploration_policy=policy if isinstance(policy, PESExplorationPolicy) else None,
            )
        for result in fresh:
            if result.status != "completed" or result.energy_hartree is None:
                (scan_dir / "failed_points.json").write_text(
                    json.dumps({"point_index": result.point_index, "message": result.message}, indent=2) + "\n",
                    encoding="utf-8",
                )
                raise RuntimeError(f"ESALC point {result.point_index} failed: {result.message}")
            cached[result.point_index] = result
            point = next(item for item in missing if item.index == result.point_index)
            cache_path = cache_dir / (
                geometry_cache_key(
                    point.coordinates_angstrom,
                    atoms=atoms,
                    backend=request.backend,
                    charge=request.charge,
                    multiplicity=request.multiplicity,
                    coordinate_labels=request.coordinates,
                )
                + ".json"
            )
            cache_path.write_text(json.dumps(point_result_to_json(result), indent=2) + "\n", encoding="utf-8")
            put_cache(sqlite_cache, cache_path.stem, point_result_to_json(result))
    ordered = tuple(cached[point.index] for point in typed_points)
    (scan_dir / "esalc_scan_cache.json").write_text(
        json.dumps(
            {"schema": "matrix.tunneling.esalc_scan_cache.v1", "completed": sorted(cached)},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    sqlite_cache.close()
    cache_connection = open_cache(scan_dir / "esalc_cache.sqlite")
    export_parquet(cache_connection, scan_dir / "esalc_cache.parquet")
    cache_connection.close()
    return ordered


def _surface_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.suffix.casefold() != ".csv":
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _surface_uncertainty(path: Path) -> dict[str, float]:
    from .advanced_workflow import uncertainty_from_energies

    rows = _surface_rows(path)
    key = "energy_hartree" if rows and "energy_hartree" in rows[0] else "relative_energy_cm-1"
    return uncertainty_from_energies([float(row[key]) for row in rows]) if rows else {}


def _write_adaptive_cells(path: Path, outdir: Path, selector) -> str | None:
    rows = _surface_rows(path)
    if not rows or not {"q1", "q2", "energy_hartree"}.issubset(rows[0]):
        return None
    q1 = sorted({float(row["q1"]) for row in rows})
    q2 = sorted({float(row["q2"]) for row in rows})
    energy = np.full((len(q1), len(q2)), np.nan)
    index1 = {value: index for index, value in enumerate(q1)}
    index2 = {value: index for index, value in enumerate(q2)}
    for row in rows:
        energy[index1[float(row["q1"])]][index2[float(row["q2"])]] = float(row["energy_hartree"])
    cells = selector(energy)
    path_out = outdir / "adaptive_cells.json"
    path_out.write_text(json.dumps({"cells": cells, "grid_shape": list(energy.shape)}, indent=2) + "\n", encoding="utf-8")
    return str(path_out)


def _surface_interpolation_error(path: Path, estimator) -> dict[str, float]:
    rows = _surface_rows(path)
    if not rows or not {"q1", "q2", "energy_hartree"}.issubset(rows[0]):
        return {}
    q1 = sorted({float(row["q1"]) for row in rows})
    q2 = sorted({float(row["q2"]) for row in rows})
    energy = np.full((len(q1), len(q2)), np.nan)
    for row in rows:
        energy[q1.index(float(row["q1"]))][q2.index(float(row["q2"]))] = float(row["energy_hartree"])
    return estimator(np.asarray(q1), np.asarray(q2), energy)


SonicScanRequest = ESALCScanRequest
run_sonic_scan = run_esalc_scan


__all__ = [
    "ESALCScanRequest",
    "run_esalc_scan",
]
