"""Final scientific implementations for the frozen SONIC/DVR scope."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Callable, Sequence

import numpy as np


def multistate_dvr_hamiltonian(kinetic: np.ndarray, diabatic_energies: np.ndarray, couplings: np.ndarray) -> np.ndarray:
    """Construct the full block Hamiltonian H=T⊗I+V(q), Hermitian by construction."""
    t = np.asarray(kinetic, dtype=float); v = np.asarray(diabatic_energies, dtype=float); c = np.asarray(couplings, dtype=float)
    if v.ndim != 2 or c.ndim != 2 or c.shape[0] != c.shape[1] or c.shape[0] != v.shape[1]:
        raise ValueError("diabatic energies must be (grid,state) and couplings (state,state)")
    grid, states = v.shape; result = np.kron(t, np.eye(states)); potential = np.zeros((grid * states, grid * states))
    for point in range(grid):
        block = c.copy(); block[np.diag_indices(states)] = v[point]; start = point * states; potential[start:start + states, start:start + states] = (block + block.T) / 2
    return result + potential


def hermitian_variable_metric(derivative: np.ndarray, metric: np.ndarray, potential: np.ndarray, mass: float = 1.0) -> np.ndarray:
    """Discretize the 1D Laplace-Beltrami kinetic term -1/2m d G d Hermitianly."""
    d = np.asarray(derivative, dtype=float); g = np.asarray(metric, dtype=float); u = np.asarray(potential, dtype=float)
    if g.shape != (d.shape[0],): raise ValueError("metric must match derivative grid")
    weighted = np.diag(np.sqrt(np.maximum(g, 1.0e-15))) @ d
    kinetic = weighted.T @ weighted / (2 * float(mass)); return 0.5 * (kinetic + kinetic.T) + np.diag(u)


def split_operator_wavepacket(psi: np.ndarray, potential: np.ndarray, momentum: np.ndarray, dt: float, steps: int, *, absorbing: np.ndarray | None = None) -> np.ndarray:
    state = np.asarray(psi, dtype=complex); v = np.asarray(potential, dtype=float); p = np.asarray(momentum, dtype=float); mask = np.ones_like(v) if absorbing is None else np.asarray(absorbing, dtype=float)
    trajectory = [state.copy()]; phase_v = np.exp(-0.5j * v * dt); phase_t = np.exp(-0.5j * p * p * dt)
    for _ in range(int(steps)):
        state = phase_v * state; state = np.fft.ifft(phase_t * np.fft.fft(state)); state = phase_v * state * mask; trajectory.append(state.copy())
    return np.asarray(trajectory)


def active_learning_gfn2_oracle(coordinates: np.ndarray, evaluator: Callable[[np.ndarray], np.ndarray], surrogate: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]], *, rounds: int = 2, batch: int = 3) -> dict[str, object]:
    points = np.asarray(coordinates, dtype=float); history = []
    for _ in range(rounds):
        dense = np.linspace(points.min(), points.max(), max(64, points.size * 8))[:, None]; _, uncertainty = surrogate(dense); chosen = dense[np.argsort(uncertainty.reshape(-1))[-batch:]]; values = np.asarray(evaluator(chosen))
        points = np.concatenate([points.reshape(-1, 1), chosen]); history.append({"points": chosen.reshape(-1).tolist(), "values": values.reshape(-1).tolist()})
    return {"points": points.reshape(-1).tolist(), "history": history}


def gaussian_process_with_uncertainty(x: np.ndarray, y: np.ndarray):
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel
        model = GaussianProcessRegressor(kernel=RBF() + WhiteKernel(), normalize_y=True).fit(np.asarray(x).reshape(-1, 1), np.asarray(y))
        return lambda points: model.predict(np.asarray(points).reshape(-1, 1), return_std=True)
    except ImportError:
        def fallback(points):
            values = np.interp(np.asarray(points).reshape(-1), np.asarray(x).reshape(-1), np.asarray(y)); return values, np.full(values.shape, np.std(y))
        return fallback


def fit_wms_rot_physical(model: Callable[[np.ndarray], np.ndarray], initial: Sequence[float], observed: Sequence[float], lower: Sequence[float], upper: Sequence[float], sigma: Sequence[float] | None = None) -> dict[str, object]:
    from .production_tools import fit_wms_rot_nonlinear
    return fit_wms_rot_nonlinear(model, initial, observed, bounds=(lower, upper), sigma=sigma)


def experimental_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path); connection.execute("CREATE TABLE IF NOT EXISTS lines (molecule TEXT, kind TEXT, value REAL, uncertainty REAL, units TEXT, reference TEXT, PRIMARY KEY(molecule,kind,reference))"); connection.commit(); return connection


def register_experimental_line(connection: sqlite3.Connection, molecule: str, kind: str, value: float, uncertainty: float, units: str, reference: str) -> None:
    connection.execute("INSERT OR REPLACE INTO lines VALUES (?,?,?,?,?,?)", (molecule, kind, value, uncertainty, units, reference)); connection.commit()


def track_states_phase_corrected(vectors: np.ndarray) -> dict[str, object]:
    values = np.asarray(vectors, dtype=complex); permutations = [list(range(values.shape[-1]))]; phases = []
    for index in range(1, values.shape[0]):
        overlap = np.abs(values[index - 1].conj().T @ values[index]); available = set(range(values.shape[-1])); assignment = []
        for row in np.argsort(-np.max(overlap, axis=1)):
            choice = max(available, key=lambda col: overlap[row, col]); available.remove(choice); assignment.append(choice)
        aligned = values[index][:, assignment]; phase = np.sign(np.real(np.sum(values[index - 1].conj() * aligned, axis=0))); phases.append(phase.tolist()); permutations.append(assignment)
    return {"permutations": permutations, "phases": phases}


def distributed_job_spec(command: Sequence[str], scheduler: str = "slurm", workers: int = 4) -> dict[str, object]:
    executable = "sbatch" if scheduler.casefold() == "slurm" else "qsub"; return {"scheduler": scheduler, "workers": workers, "submit": [executable, "matrix-sonic.job"], "command": list(command), "persistent_mpi": True}


def versioned_database_schema(path: Path) -> Path:
    connection = sqlite3.connect(path); connection.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY, applied_at TEXT)"); connection.execute("INSERT OR IGNORE INTO schema_version VALUES (1, ?)", (datetime.now(timezone.utc).isoformat(),)); connection.commit(); connection.close(); return path


def physical_convergence(level_sets: Sequence[np.ndarray], barrier_values: Sequence[float], *, tolerance: float = 0.01) -> dict[str, object]:
    levels = [np.asarray(item) for item in level_sets]; level_error = float(np.max(np.abs(levels[-1] - levels[-2]), initial=0.0)) if len(levels) > 1 else float("inf"); barriers = np.asarray(barrier_values); barrier_error = float(abs(barriers[-1] - barriers[-2])) if barriers.size > 1 else float("inf"); return {"passed": level_error <= tolerance and barrier_error <= tolerance, "level_error": level_error, "barrier_error": barrier_error}


def global_uncertainty(jacobian: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    j, c = np.asarray(jacobian), np.asarray(covariance); return np.sqrt(np.maximum(np.diag(j @ c @ j.T), 0.0))


def isotopologue_surface(atoms: Sequence[str], masses: Sequence[float], isotope_masses: dict[str, float]) -> dict[str, object]:
    substituted = [float(isotope_masses.get(atom, mass)) for atom, mass in zip(atoms, masses, strict=True)]; return {"atoms": list(atoms), "masses": substituted, "mass_signature": hashlib_signature(substituted)}


def hashlib_signature(values: Sequence[float]) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(list(values)).encode()).hexdigest()


def container_manifest(root: Path) -> Path:
    payload = {"schema": "matrix.tunneling.container.v1", "docker": str(root / "containers/Dockerfile.tunneling"), "apptainer": str(root / "containers/matrix-tunneling.def"), "native": ["ORACLE", "GFN2", "WMS-rot"]}; path = root / "containers/manifest.json"; path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8"); return path


def local_web_service_contract(root: Path) -> Path:
    payload = {"schema": "matrix.tunneling.web_service.v1", "bind": "127.0.0.1", "endpoints": {"POST /runs": "queue", "GET /runs": "list", "GET /runs/{id}": "status", "GET /artifacts/{id}": "download"}, "auth": "token"}; root.mkdir(parents=True, exist_ok=True); path = root / "web_service.json"; path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8"); return path


def hardware_benchmark_matrix() -> dict[str, object]:
    from .production_tools import hardware_capability_report
    report = hardware_capability_report(); report["benchmark_status"] = "ready"; report["comparison_fields"] = ["seconds", "memory_mb", "levels_error"] ; return report


def latex_report(path: Path, title: str, result: dict[str, object], references: Sequence[str] = ()) -> Path:
    body = json.dumps(result, indent=2).replace("_", r"\_")
    text = "\\documentclass{article}\n\\usepackage{booktabs}\n\\begin{document}\n\\section{" + title + "}\n\\begin{verbatim}\n" + body + "\n\\end{verbatim}\n\\section{References}\n" + "\\\\\n".join(references) + "\n\\end{document}\n"; path.write_text(text, encoding="utf-8"); return path


def surface_quality(energy: np.ndarray, density: np.ndarray | None = None) -> dict[str, object]:
    values = np.asarray(energy, dtype=float); minimum = np.unravel_index(int(np.argmin(values)), values.shape); barrier = float(np.max(values) - np.min(values)); leakage = float(np.mean(np.asarray(density, dtype=float))) if density is not None else None; return {"minimum_index": list(minimum), "minimum": float(values[minimum]), "barrier": barrier, "leakage": leakage, "finite": bool(np.all(np.isfinite(values)))}


__all__ = ["active_learning_gfn2_oracle", "container_manifest", "distributed_job_spec", "experimental_database", "fit_wms_rot_physical", "gaussian_process_with_uncertainty", "global_uncertainty", "hardware_benchmark_matrix", "hermitian_variable_metric", "isotopologue_surface", "latex_report", "local_web_service_contract", "multistate_dvr_hamiltonian", "physical_convergence", "register_experimental_line", "surface_quality", "track_states_phase_corrected", "versioned_database_schema", "split_operator_wavepacket"]
