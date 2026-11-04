"""Production-grade extensions for fitting, surfaces, scheduling and qualification."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sqlite3
import subprocess
from typing import Callable, Sequence

import numpy as np


def solve_sparse_tensor_dvr(grids: tuple[np.ndarray, ...], potential_cm: np.ndarray, *, levels: int = 12, periodic: bool = True):
    """Solve a tensor DVR without constructing a dense Hamiltonian."""
    from .next_generation import solve_sparse_levels, sparse_laplacian_1d
    try:
        from scipy.sparse import diags, eye, kron
    except ImportError:
        raise RuntimeError("sparse tensor DVR requires scipy") from None
    shape = tuple(len(grid) for grid in grids)
    potential = np.asarray(potential_cm, dtype=float)
    if potential.shape != shape:
        raise ValueError("potential shape must match tensor grids")
    hamiltonian = diags(potential.reshape(-1), format="csr")
    for axis, grid in enumerate(grids):
        values = np.asarray(grid, dtype=float)
        if len(values) < 3 or not np.allclose(np.diff(values), values[1] - values[0]):
            raise ValueError("sparse tensor grids must be uniformly spaced")
        factors = [eye(len(item), format="csr") for item in grids]
        factors[axis] = sparse_laplacian_1d(len(values), float(values[1] - values[0]), periodic=periodic)
        operator = factors[0]
        for factor in factors[1:]:
            operator = kron(operator, factor, format="csr")
        hamiltonian = hamiltonian + operator
    return solve_sparse_levels(hamiltonian, levels)


def fit_wms_rot_nonlinear(model: Callable[[np.ndarray], np.ndarray], initial: Sequence[float], observed: Sequence[float], *, bounds: tuple[Sequence[float], Sequence[float]] | None = None, sigma: Sequence[float] | None = None) -> dict[str, object]:
    """Bounded nonlinear least-squares fit with covariance estimate."""
    target = np.asarray(observed, dtype=float)
    weights = np.ones_like(target) if sigma is None else 1.0 / np.maximum(np.asarray(sigma, dtype=float), 1.0e-15)
    def residual(parameters: np.ndarray) -> np.ndarray:
        return (np.asarray(model(parameters), dtype=float) - target) * weights
    try:
        from scipy.optimize import least_squares
        result = least_squares(residual, np.asarray(initial, dtype=float), bounds=bounds if bounds is not None else (-np.inf, np.inf))
        jacobian = np.asarray(result.jac, dtype=float)
        covariance = np.linalg.pinv(jacobian.T @ jacobian) * (2 * result.cost / max(1, target.size - len(result.x)))
        return {"parameters": result.x.tolist(), "covariance": covariance.tolist(), "sigma": np.sqrt(np.maximum(np.diag(covariance), 0.0)).tolist(), "rmse": float(np.sqrt(np.mean(residual(result.x) ** 2))), "success": bool(result.success), "message": result.message}
    except ImportError:
        return {"parameters": list(initial), "covariance": [], "sigma": [], "rmse": float("nan"), "success": False, "message": "scipy is required for nonlinear fitting"}


def compare_spectroscopic_database(predicted: dict[str, float], observed: dict[str, float], *, tolerances: dict[str, float] | None = None) -> dict[str, object]:
    tolerances = tolerances or {}
    rows = []
    for key in sorted(set(predicted) | set(observed)):
        p, o = predicted.get(key), observed.get(key)
        error = None if p is None or o is None else float(p - o)
        rows.append({"key": key, "predicted": p, "observed": o, "error": error, "pass": error is not None and abs(error) <= tolerances.get(key, np.inf)})
    return {"count": len(rows), "pass": all(row["pass"] for row in rows), "rows": rows}


def select_electronic_surface(energies: np.ndarray, state: int = 0) -> np.ndarray:
    values = np.asarray(energies, dtype=float)
    if values.ndim < 2 or not 0 <= state < values.shape[-1]:
        raise ValueError("energies must have a state axis and a valid state index")
    return values[..., state]


def finite_difference_nac(states: np.ndarray, spacing: float) -> np.ndarray:
    values = np.asarray(states, dtype=float)
    if values.ndim != 3:
        raise ValueError("states must have shape (grid, basis, electronic_state)")
    return np.gradient(values, float(spacing), axis=0, edge_order=1)


def conical_intersection_indicator(energies: np.ndarray, *, gap_threshold: float = 1.0e-4) -> dict[str, object]:
    values = np.asarray(energies, dtype=float)
    gaps = np.abs(values[..., 1:] - values[..., :-1])
    locations = np.argwhere(np.any(gaps <= gap_threshold, axis=-1))
    return {"gap_min": float(np.min(gaps, initial=np.inf)), "threshold": gap_threshold, "candidate_indices": locations.tolist(), "detected": bool(locations.size)}


def monte_carlo_geometry(geometry: np.ndarray, covariance: np.ndarray, *, samples: int = 1000, seed: int = 0) -> np.ndarray:
    coordinates = np.asarray(geometry, dtype=float)
    flat_cov = np.asarray(covariance, dtype=float)
    vector = coordinates.reshape(-1)
    if flat_cov.shape != (vector.size, vector.size):
        raise ValueError("geometry covariance shape mismatch")
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(vector, flat_cov, size=int(samples)).reshape((int(samples),) + coordinates.shape)


def scheduler_script(command: Sequence[str], *, scheduler: str = "slurm", job_name: str = "matrix-sonic", workers: int = 1, walltime: str = "01:00:00") -> str:
    lines = ["#!/bin/bash", f"# {'SBATCH' if scheduler.casefold() == 'slurm' else '#PBS'} -J {job_name}"]
    if scheduler.casefold() == "slurm":
        lines += [f"#SBATCH --cpus-per-task={workers}", f"#SBATCH --time={walltime}"]
    else:
        lines += [f"#PBS -l select=1:ncpus={workers}", f"#PBS -l walltime={walltime}"]
    lines.append(" ".join(command))
    return "\n".join(lines) + "\n"


def submit_scheduler_job(script: Path, *, scheduler: str = "slurm", dry_run: bool = True) -> dict[str, object]:
    executable = "sbatch" if scheduler.casefold() == "slurm" else "qsub"
    command = [executable, str(script)]
    if dry_run:
        return {"submitted": False, "command": command, "status": "dry_run"}
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return {"submitted": result.returncode == 0, "command": command, "stdout": result.stdout, "stderr": result.stderr}


def open_central_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, created_at TEXT, payload TEXT NOT NULL)")
    connection.commit()
    return connection


def register_run(connection: sqlite3.Connection, run_id: str, payload: dict[str, object]) -> None:
    connection.execute("INSERT OR REPLACE INTO runs VALUES (?, ?, ?)", (run_id, datetime.now(timezone.utc).isoformat(), json.dumps(payload, sort_keys=True)))
    connection.commit()


def hardware_capability_report() -> dict[str, object]:
    report = {"platform": platform.platform(), "python": platform.python_version(), "mpi": False, "cuda": False}
    try:
        __import__("mpi4py")
        report["mpi"] = True
    except ImportError:
        pass
    try:
        import cupy
        report["cuda"] = bool(cupy.cuda.is_available())
    except (ImportError, RuntimeError):
        pass
    return report


def reproducible_environment(path: Path) -> Path:
    payload = {"python": platform.python_version(), "platform": platform.platform(), "packages": subprocess.run(["python", "-m", "pip", "freeze"], capture_output=True, text=True, check=False).stdout.splitlines()}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def keymaker_dashboard_contract(outdir: Path) -> Path:
    payload = {
        "schema": "matrix.keymaker.esalc_lock_dashboard.v1",
        "actions": ["launch_esalc_scan", "fit_lock", "view_levels", "export_report"],
        "outdir": str(outdir),
    }
    path = outdir / "keymaker_dashboard.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def certify_calculation(reference: Sequence[float], candidate: Sequence[float], *, tolerance: float = 1.0e-3) -> dict[str, object]:
    ref, got = np.asarray(reference, dtype=float), np.asarray(candidate, dtype=float)
    error = float(np.max(np.abs(ref - got), initial=0.0)) if ref.shape == got.shape else float("inf")
    return {"schema": "matrix.tunneling.certification.v1", "passed": error <= tolerance, "max_error": error, "tolerance": tolerance, "count": int(ref.size) if ref.shape == got.shape else 0}


__all__ = ["certify_calculation", "compare_spectroscopic_database", "conical_intersection_indicator", "finite_difference_nac", "fit_wms_rot_nonlinear", "hardware_capability_report", "keymaker_dashboard_contract", "monte_carlo_geometry", "open_central_database", "register_run", "reproducible_environment", "scheduler_script", "select_electronic_surface", "solve_sparse_tensor_dvr", "submit_scheduler_job"]
