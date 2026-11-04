"""Backend-independent geometry optimization and finite-difference Hessian."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np


BOHR_TO_ANGSTROM = 0.529177210903
PointEvaluator = Callable[[np.ndarray], tuple[float, np.ndarray]]


@dataclass(frozen=True)
class QMGeometryResult:
    coordinates_angstrom: np.ndarray
    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray
    hessian_hartree_per_bohr2: np.ndarray
    iterations: int
    converged: bool
    manifest_path: Path | None = None


def optimize_and_hessian(
    coordinates_angstrom: object,
    evaluate: PointEvaluator,
    *,
    max_iterations: int = 30,
    gradient_tolerance: float = 1.0e-4,
    initial_step_bohr: float = 0.15,
    hessian_step_bohr: float = 1.0e-3,
    hessian_workers: int = 1,
) -> QMGeometryResult:
    """Use monotonic backtracking gradient descent, then central Hessian FD."""
    coords = np.asarray(coordinates_angstrom, dtype=float).copy()
    if coords.ndim != 2 or coords.shape[1] != 3 or not np.all(np.isfinite(coords)):
        raise ValueError("coordinates must be a finite natoms x 3 array")
    if max_iterations < 0 or gradient_tolerance <= 0.0 or hessian_step_bohr <= 0.0:
        raise ValueError("iteration count and tolerances/steps must be positive")
    if hessian_workers < 1:
        raise ValueError("hessian_workers must be at least one")
    energy, gradient = evaluate(coords)
    converged = False
    iterations = 0
    for iteration in range(1, max_iterations + 1):
        gradient = np.asarray(gradient, dtype=float).reshape(coords.shape)
        norm = float(np.linalg.norm(gradient))
        iterations = iteration
        if norm <= gradient_tolerance:
            converged = True
            break
        direction = -gradient / max(norm, 1.0e-15)
        step = float(initial_step_bohr)
        accepted = False
        for _ in range(12):
            trial = coords + direction * step * BOHR_TO_ANGSTROM
            trial_energy, trial_gradient = evaluate(trial)
            if trial_energy < energy:
                coords, energy, gradient = trial, float(trial_energy), trial_gradient
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
    if not converged:
        final_gradient = np.asarray(gradient, dtype=float).reshape(coords.shape)
        converged = float(np.linalg.norm(final_gradient)) <= gradient_tolerance
    flat = coords.reshape(-1)
    hessian = np.empty((flat.size, flat.size), dtype=float)
    step_angstrom = float(hessian_step_bohr) * BOHR_TO_ANGSTROM
    def hessian_column(column: int) -> tuple[int, np.ndarray]:
        plus = flat.copy(); plus[column] += step_angstrom
        minus = flat.copy(); minus[column] -= step_angstrom
        _, gradient_plus = evaluate(plus.reshape(coords.shape))
        _, gradient_minus = evaluate(minus.reshape(coords.shape))
        values = (
            np.asarray(gradient_plus).reshape(-1) - np.asarray(gradient_minus).reshape(-1)
        ) / (2.0 * float(hessian_step_bohr))
        return column, values
    with ThreadPoolExecutor(max_workers=max(1, int(hessian_workers))) as executor:
        for column, values in executor.map(hessian_column, range(flat.size)):
            hessian[:, column] = values
    return QMGeometryResult(coords, float(energy), np.asarray(gradient).reshape(-1), hessian, iterations, converged)


def write_opt_hessian_manifest(
    workdir: Path | str,
    result: QMGeometryResult,
    *,
    backend: str,
    route: str,
    settings: dict[str, object] | None = None,
) -> Path:
    """Write a compact, reproducible manifest for an Opt+Hessian run."""
    root = Path(workdir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    hessian = np.asarray(result.hessian_hartree_per_bohr2, dtype=float)
    payload = {
        "schema": "matrix.qm.opt_hessian_manifest.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": str(backend).casefold(),
        "route": str(route).casefold(),
        "settings": dict(settings or {}),
        "result": {
            "converged": bool(result.converged),
            "iterations": int(result.iterations),
            "energy_hartree": float(result.energy_hartree),
            "gradient_norm_hartree_per_bohr": float(np.linalg.norm(result.gradient_hartree_per_bohr)),
            "coordinates_shape": list(np.asarray(result.coordinates_angstrom).shape),
            "hessian_shape": list(hessian.shape),
            "coordinates_sha256": hashlib.sha256(np.asarray(result.coordinates_angstrom, dtype="<f8").tobytes()).hexdigest(),
            "hessian_sha256": hashlib.sha256(np.asarray(hessian, dtype="<f8").tobytes()).hexdigest(),
        },
    }
    target = root / "opt-hessian-manifest.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
