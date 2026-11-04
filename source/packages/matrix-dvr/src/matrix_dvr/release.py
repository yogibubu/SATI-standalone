"""Frozen MATRIX tunneling 1.0 release workflow and acceptance gates."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .advanced_extensions import coupled_surface_hamiltonian, propagate_wavepacket
from .convergence import assess_convergence
from .metric_validation import validate_metric_grid
from .next_generation import fit_chebyshev, validate_experimental_reference
from .oracle_extensions import sync_manifest, validate_before_sync
from .production_tools import hardware_capability_report


@dataclass(frozen=True)
class ReleaseRequest:
    outdir: Path
    levels: int = 12
    tolerance_cm: float = 0.01
    experimental_tolerance: float = 1.0
    run_wavepacket: bool = False


def run_release_gates(request: ReleaseRequest, *, level_sets: Sequence[np.ndarray] = (), metric: np.ndarray | None = None, predicted: Sequence[float] = (), observed: Sequence[float] = ()) -> dict[str, object]:
    """Run all deterministic 1.0 acceptance gates and write one manifest."""
    request.outdir.mkdir(parents=True, exist_ok=True)
    gates: dict[str, object] = {"schema": "matrix.tunneling.release.v1", "release": "1.0.0", "gates": {}}
    if metric is not None:
        gates["gates"]["metric"] = validate_metric_grid(metric)
    if len(level_sets) >= 2:
        gates["gates"]["convergence"] = assess_convergence(list(level_sets), tolerance_cm=request.tolerance_cm)
    if predicted and observed:
        gates["gates"]["experimental"] = validate_experimental_reference(predicted, observed)
    gates["gates"]["hardware"] = hardware_capability_report()
    gates["gates"]["inputs"] = validate_before_sync(request.outdir)
    gates["passed"] = all(value.get("valid", value.get("pass", value.get("converged", value.get("passed", True)))) for value in gates["gates"].values() if isinstance(value, dict))
    path = request.outdir / "release_acceptance.json"
    path.write_text(json.dumps(gates, indent=2) + "\n", encoding="utf-8")
    return gates


def active_learning_loop(coordinates: np.ndarray, values: np.ndarray, evaluator: Callable[[np.ndarray], np.ndarray], *, rounds: int = 2, points_per_round: int = 2) -> dict[str, object]:
    """Select high-residual points and extend a Chebyshev surrogate."""
    q = np.asarray(coordinates, dtype=float).reshape(-1)
    y = np.asarray(values, dtype=float).reshape(-1)
    history = []
    for _ in range(int(rounds)):
        model = fit_chebyshev(q, y, degree=min(8, len(q) - 1))
        dense = np.linspace(q.min(), q.max(), max(32, len(q) * 4))
        residual = np.abs(np.asarray(evaluator(dense)) - model(dense))
        chosen = dense[np.argsort(residual)[-int(points_per_round):]]
        q = np.concatenate([q, chosen]); y = np.concatenate([y, np.asarray(evaluator(chosen))])
        history.append({"points": chosen.tolist(), "max_residual": float(np.max(residual))})
    return {"coordinates": q.tolist(), "values": y.tolist(), "history": history}


def build_multistate_wavepacket(energies: np.ndarray, couplings: np.ndarray, initial: np.ndarray, *, dt: float, steps: int) -> np.ndarray:
    hamiltonian = coupled_surface_hamiltonian(energies, couplings)
    return propagate_wavepacket(initial, hamiltonian, dt, steps)


def release_version_manifest(root: Path) -> Path:
    payload = {"schema": "matrix.tunneling.release_manifest.v1", "version": "1.0.0", "scope": ["scientific", "active-learning", "wms-rot", "validation", "oracle", "performance", "reproducibility", "certification"], "root": str(root), "oracle_sync_manifest": str(sync_manifest(root))}
    path = root / "release_manifest.json"; path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8"); return path


__all__ = ["ReleaseRequest", "active_learning_loop", "build_multistate_wavepacket", "release_version_manifest", "run_release_gates"]
