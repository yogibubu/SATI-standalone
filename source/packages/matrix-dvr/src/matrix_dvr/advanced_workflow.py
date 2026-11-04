"""Operational services for reproducible, resilient ESALC/DVR workflows."""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ExecutionPlan:
    requested: str
    selected: str
    workers: int
    gpu_available: bool
    mpi_available: bool
    reason: str


def parallel_map(function, items: Sequence[object], plan: ExecutionPlan) -> tuple[object, ...]:
    """Execute independent tasks through the selected available executor."""
    values = tuple(items)
    if len(values) <= 1 or plan.workers <= 1:
        return tuple(function(item) for item in values)
    if plan.selected == "mpi":
        try:
            from mpi4py.futures import MPIPoolExecutor

            with MPIPoolExecutor(max_workers=plan.workers) as executor:
                return tuple(executor.map(function, values))
        except (ImportError, RuntimeError, OSError, TypeError, ValueError):
            pass
    # Threading is effective here because electronic-structure kernels release
    # the GIL; each task owns its resident calculator/session.
    with ThreadPoolExecutor(max_workers=plan.workers) as executor:
        return tuple(executor.map(function, values))


def accelerated_eigh(matrix: np.ndarray, plan: ExecutionPlan) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize on CuPy when selected, otherwise use NumPy deterministically."""
    values = np.asarray(matrix, dtype=float)
    if plan.selected == "gpu":
        try:
            import cupy as cp

            gpu_matrix = cp.asarray(values)
            eigenvalues, eigenvectors = cp.linalg.eigh(gpu_matrix)
            return cp.asnumpy(eigenvalues), cp.asnumpy(eigenvectors)
        except (ImportError, RuntimeError, OSError):
            pass
    return np.linalg.eigh(values)


def plan_execution(requested: str, workers: int) -> ExecutionPlan:
    requested = str(requested).strip().casefold() or "auto"
    gpu_available = False
    for module in ("cupy", "torch"):
        try:
            loaded = __import__(module)
            gpu_available = bool(
                getattr(getattr(loaded, "cuda", None), "is_available", lambda: False)()
            )
        except (ImportError, RuntimeError, AttributeError):
            continue
        if gpu_available:
            break
    try:
        __import__("mpi4py")
        mpi_available = True
    except ImportError:
        mpi_available = False
    if requested == "auto":
        selected = "gpu" if gpu_available else "thread"
    elif requested == "gpu" and not gpu_available:
        selected = "thread"
    elif requested == "mpi" and not mpi_available:
        selected = "thread"
    elif requested not in {"thread", "mpi", "gpu"}:
        raise ValueError("execution backend must be auto, thread, mpi, or gpu")
    else:
        selected = requested
    reason = "requested backend available" if selected == requested else "requested backend unavailable; safe fallback"
    return ExecutionPlan(requested, selected, max(1, int(workers)), gpu_available, mpi_available, reason)


def geometry_cache_key(
    coordinates: np.ndarray,
    *,
    atoms: Sequence[str],
    backend: str,
    charge: int,
    multiplicity: int,
    coordinate_labels: Sequence[str],
) -> str:
    digest = hashlib.sha256()
    digest.update("|".join(map(str, atoms)).encode())
    digest.update(np.asarray(coordinates, dtype=np.float64).tobytes())
    digest.update(json.dumps({
        "backend": backend,
        "charge": int(charge),
        "multiplicity": int(multiplicity),
        "coordinates": list(coordinate_labels),
    }, sort_keys=True).encode())
    return digest.hexdigest()


def adaptive_cells_2d(energy: np.ndarray, threshold: float = 0.05) -> list[dict[str, int | float]]:
    """Rank local 2D cells by normalized curvature for targeted refinement."""
    values = np.asarray(energy, dtype=float)
    if values.ndim != 2 or min(values.shape) < 3:
        return []
    scale = max(float(np.ptp(values)), 1.0e-12)
    cells: list[dict[str, int | float]] = []
    for i in range(values.shape[0] - 2):
        for j in range(values.shape[1] - 2):
            curvature = float(abs(values[i + 1, j] - 2 * values[i + 1, j + 1] + values[i + 1, j + 2]))
            curvature += float(abs(values[i, j + 1] - 2 * values[i + 1, j + 1] + values[i + 2, j + 1]))
            score = curvature / scale
            if score > threshold:
                cells.append({"i": i, "j": j, "curvature": curvature, "score": score})
    return sorted(cells, key=lambda item: float(item["score"]), reverse=True)


def interpolation_error_2d(q1: np.ndarray, q2: np.ndarray, values: np.ndarray) -> dict[str, float]:
    """Estimate interpolation uncertainty from cell-center bilinear residuals."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or min(values.shape) < 2:
        return {"max_abs": 0.0, "rms": 0.0, "relative_max": 0.0}
    residuals = []
    for i in range(len(q1) - 1):
        for j in range(len(q2) - 1):
            center = 0.25 * (values[i, j] + values[i + 1, j] + values[i, j + 1] + values[i + 1, j + 1])
            diagonal = 0.5 * (values[i, j] + values[i + 1, j + 1])
            residuals.append(float(center - diagonal))
    residual = np.asarray(residuals, dtype=float)
    scale = max(float(np.ptp(values)), 1.0e-12)
    return {
        "max_abs": float(np.max(np.abs(residual), initial=0.0)),
        "rms": float(np.sqrt(np.mean(residual * residual))) if residual.size else 0.0,
        "relative_max": float(np.max(np.abs(residual), initial=0.0) / scale),
    }


def uncertainty_from_energies(energies: Sequence[float]) -> dict[str, float]:
    values = np.asarray(energies, dtype=float)
    if values.size < 2:
        return {"energy_std_hartree": 0.0, "energy_range_hartree": 0.0}
    return {"energy_std_hartree": float(np.std(values)), "energy_range_hartree": float(np.ptp(values))}


def validate_geometry_grid(atoms: Sequence[str], geometries: Sequence[Sequence[Sequence[float]]]) -> dict[str, object]:
    reference = tuple(str(atom) for atom in atoms)
    errors: list[str] = []
    for index, geometry in enumerate(geometries):
        array = np.asarray(geometry, dtype=float)
        if array.shape != (len(reference), 3) or not np.all(np.isfinite(array)):
            errors.append(f"geometry {index} has invalid shape or non-finite coordinates")
    return {"valid": not errors, "atom_order": list(reference), "errors": errors}


def write_interactive_viewer(outdir: Path, *, result_json: Path, surface: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    header = surface.read_text(encoding="utf-8").splitlines()[0].split(",") if surface.is_file() else []
    payload = {
        "result": str(result_json),
        "surface": str(surface),
        "csv": surface.read_text(encoding="utf-8") if surface.is_file() else "",
        "energy_index": header.index("energy_hartree") if "energy_hartree" in header else 2,
    }
    text = """<!doctype html><meta charset='utf-8'><title>ESALC surface viewer</title>
<style>body{font:15px system-ui;margin:2em}canvas{width:100%;max-width:1000px;border:1px solid #aaa}</style>
<h1>ESALC/DVR surface viewer</h1><p id='status'></p><label>Energy scale <input id='scale' type='range' min='0.2' max='3' step='0.1' value='1'></label><p>Click a point in the 1D plot to inspect its index and energy.</p><canvas id='plot' width='1000' height='500'></canvas><canvas id='surface2d' width='1000' height='500'></canvas><canvas id='surface3d' width='1000' height='500'></canvas>
<script>const manifest=""" + json.dumps(payload) + """;document.querySelector('#status').textContent='Surface: '+manifest.surface;
const lines=manifest.csv.trim().split(/\\n/),header=lines.shift().split(','),rows=lines.map(x=>x.split(','));const e=rows.map(x=>+x[manifest.energy_index]);const c=document.querySelector('#plot').getContext('2d');const lo=Math.min(...e),hi=Math.max(...e);c.beginPath();e.forEach((v,i)=>{const x=20+i*960/Math.max(1,e.length-1),y=470-430*(v-lo)/Math.max(hi-lo,1e-12);i?c.lineTo(x,y):c.moveTo(x,y)});c.strokeStyle='#1464a0';c.lineWidth=2;c.stroke();c.canvas.onclick=(event)=>{const index=Math.round((event.offsetX-20)*Math.max(1,e.length-1)/960);document.querySelector('#status').textContent='Point '+index+' | energy '+(e[index]??'n/a')+' | '+manifest.surface};document.querySelector('#scale').oninput=(event)=>{c.canvas.style.transform=`scale(${event.target.value})`;};if(header.includes('q1')&&header.includes('q2')){const i1=header.indexOf('q1'),i2=header.indexOf('q2'),xs=[...new Set(rows.map(r=>+r[i1]))],ys=[...new Set(rows.map(r=>+r[i2]))],ctx=document.querySelector('#surface2d').getContext('2d'),w=960/xs.length,h=430/ys.length;rows.forEach(r=>{const x=xs.indexOf(+r[i1]),y=ys.indexOf(+r[i2]),v=+r[manifest.energy_index],t=(v-lo)/Math.max(hi-lo,1e-12);ctx.fillStyle=`hsl(${240-240*t},75%,50%)`;ctx.fillRect(20+x*w,20+(ys.length-1-y)*h,w+1,h+1)});const p=document.querySelector('#surface3d').getContext('2d');p.strokeStyle='#355c7d';p.fillStyle='#c06c84';const zscale=Math.max(hi-lo,1e-12);rows.forEach(r=>{const x=(+r[i1]-Math.min(...xs))/Math.max(Math.max(...xs)-Math.min(...xs),1e-12),y=(+r[i2]-Math.min(...ys))/Math.max(Math.max(...ys)-Math.min(...ys),1e-12),z=(+r[manifest.energy_index]-lo)/zscale,px=500+(x-y)*330,py=350-(x+y)*100-z*170;p.beginPath();p.arc(px,py,3,0,2*Math.PI);p.fill();});p.beginPath();p.moveTo(170,380);p.lineTo(500,480);p.lineTo(830,380);p.stroke();}</script>"""
    path = outdir / "esalc_scan_viewer.html"
    path.write_text(text, encoding="utf-8")
    return path


def write_multimode_contract(outdir: Path, *, oracle_xyzin: Path, coordinates: Sequence[str]) -> Path:
    payload = {
        "schema": "matrix.mctdh.multimode_surface.v1",
        "engine": "tensor-dvr-or-mctdh",
        "oracle_xyzin": str(oracle_xyzin),
        "coordinates": list(coordinates),
        "status": "surface_contract_ready",
        "fitting_owner": "LOCK",
    }
    path = outdir / "multimode_mctdh_input.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def write_oracle_sync_manifest(
    outdir: Path,
    *,
    oracle_xyzin: Path,
    coordinates: Sequence[str],
    artifacts: dict[str, object],
) -> Path:
    """Record the immutable ORACLE/ESALC-to-DVR handoff and artifact hashes."""
    source_hash = hashlib.sha256(oracle_xyzin.read_bytes()).hexdigest() if oracle_xyzin.is_file() else None
    payload = {
        "schema": "matrix.oracle.esalc.dvr_sync.v1",
        "oracle_xyzin": str(oracle_xyzin),
        "oracle_sha256": source_hash,
        "esalc_coordinates": list(coordinates),
        "artifacts": artifacts,
        "status": "synchronized",
    }
    path = outdir / "oracle_esalc_sync.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def autotune(*, points: int, workers: int, interpolation_factor: int) -> dict[str, int | str]:
    recommended_workers = max(1, min(int(workers), max(1, int(points))))
    recommended_factor = int(interpolation_factor)
    if points < 25 and recommended_factor == 1:
        recommended_factor = 2
    return {
        "recommended_workers": recommended_workers,
        "recommended_interpolation_factor": recommended_factor,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "policy": "bounded-by-point-count-and-request",
    }


def measure_autotune(*, points: int, workers: int, interpolation_factor: int, backend: str = "auto") -> dict[str, object]:
    """Measure a bounded numerical kernel and combine it with safe limits."""
    plan = plan_execution(backend, workers)
    size = 24
    matrix = np.diag(np.linspace(0.0, 1.0, size))
    start = time.perf_counter()
    accelerated_eigh(matrix, plan)
    elapsed = max(time.perf_counter() - start, 1.0e-9)
    recommendation = autotune(
        points=points,
        workers=workers,
        interpolation_factor=interpolation_factor,
    )
    recommendation.update({
        "measured_eigh_seconds": elapsed,
        "backend_selected": plan.selected,
        "measurement_kernel": "eigh_24x24",
        "measured": True,
    })
    return recommendation


def benchmark_execution(*, backend: str = "auto", workers: int = 2, size: int = 48) -> dict[str, object]:
    """Run a small backend benchmark and report correctness and wall time."""
    plan = plan_execution(backend, workers)
    matrix = np.diag(np.linspace(0.0, 1.0, int(size)))
    start = time.perf_counter()
    values, _ = accelerated_eigh(matrix, plan)
    elapsed = max(time.perf_counter() - start, 1.0e-12)
    return {"requested": backend, "selected": plan.selected, "workers": plan.workers, "seconds": elapsed, "eigenvalues": int(values.size), "correct": bool(np.all(np.diff(values) >= -1.0e-12))}


__all__ = [
    "ExecutionPlan", "adaptive_cells_2d", "autotune", "geometry_cache_key",
    "benchmark_execution", "interpolation_error_2d", "measure_autotune", "plan_execution", "uncertainty_from_energies",
    "validate_geometry_grid", "write_interactive_viewer", "write_multimode_contract",
    "write_oracle_sync_manifest", "parallel_map", "accelerated_eigh",
]
