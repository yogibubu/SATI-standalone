"""Local ORACLE synchronization, audit and data-management utilities."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Sequence

import numpy as np


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bidirectional_sync(local: Path, oracle: Path, *, conflict: str = "report") -> dict[str, object]:
    local.mkdir(parents=True, exist_ok=True); oracle.mkdir(parents=True, exist_ok=True)
    conflicts, copied = [], []
    for relative in sorted(set(p.relative_to(local) for p in local.rglob("*") if p.is_file()) | set(p.relative_to(oracle) for p in oracle.rglob("*") if p.is_file())):
        left, right = local / relative, oracle / relative
        if left.is_file() and right.is_file() and file_hash(left) != file_hash(right):
            conflicts.append(str(relative))
            if conflict == "oracle_wins": shutil.copy2(right, left)
            elif conflict == "local_wins": shutil.copy2(left, right)
        elif left.is_file() and not right.exists(): shutil.copy2(left, right); copied.append(str(relative))
        elif right.is_file() and not left.exists(): shutil.copy2(right, left); copied.append(str(relative))
    return {"schema": "matrix.oracle.bidirectional_sync.v1", "conflicts": conflicts, "copied": copied, "clean": not conflicts, "timestamp": datetime.now(timezone.utc).isoformat()}


def sync_manifest(root: Path) -> Path:
    payload = {"schema": "matrix.oracle.sync_manifest.v1", "files": {str(p.relative_to(root)): file_hash(p) for p in root.rglob("*") if p.is_file()}, "timestamp": datetime.now(timezone.utc).isoformat()}
    path = root / "oracle_sync_manifest.json"; path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8"); return path


def job_queue(path: Path, job: dict[str, object] | None = None) -> list[dict[str, object]]:
    queue = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    if job is not None: queue.append({**job, "queued_at": datetime.now(timezone.utc).isoformat()}); path.write_text(json.dumps(queue, indent=2) + "\n", encoding="utf-8")
    return queue


def checkpoint(path: Path, state: dict[str, object]) -> Path:
    target = path.with_suffix(path.suffix + ".checkpoint.json"); target.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8"); return target


def audit_event(path: Path, action: str, payload: dict[str, object]) -> Path:
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), "action": action, "payload": payload}
    with path.open("a", encoding="utf-8") as handle: handle.write(json.dumps(record) + "\n")
    return path


def compare_versions(first: Path, second: Path) -> dict[str, object]:
    left = json.loads(first.read_text(encoding="utf-8")); right = json.loads(second.read_text(encoding="utf-8"))
    keys = sorted(set(left) | set(right)); return {"changed": {key: {"first": left.get(key), "second": right.get(key)} for key in keys if left.get(key) != right.get(key)}, "identical": left == right}


def experimental_record(values: Sequence[float], units: str, reference: str, *, uncertainty: Sequence[float] = ()) -> dict[str, object]:
    return {"schema": "matrix.experimental.observable.v1", "values": list(map(float, values)), "units": units, "reference": reference, "uncertainty": list(map(float, uncertainty))}


def total_numerical_error(errors: Sequence[float]) -> dict[str, float]:
    values = np.abs(np.asarray(errors, dtype=float)); return {"max": float(np.max(values, initial=0.0)), "rms": float(np.sqrt(np.mean(values * values))) if values.size else 0.0, "quadrature": float(np.sqrt(np.sum(values * values)))}


def fit_sensitivity(model, parameters: Sequence[float], perturbation: float = 1.0e-5) -> np.ndarray:
    base = np.asarray(parameters, dtype=float); reference = np.asarray(model(base), dtype=float); jacobian = np.empty((reference.size, base.size))
    for index in range(base.size):
        shifted = base.copy(); shifted[index] += perturbation; jacobian[:, index] = (np.asarray(model(shifted), dtype=float) - reference).reshape(-1) / perturbation
    return jacobian


def isotopologue_records(atoms: Sequence[str], substitutions: dict[int, str]) -> list[str]:
    record = list(atoms)
    for index, isotope in substitutions.items(): record[int(index)] = isotope
    return ["".join(record)]


def propagate_rotational_uncertainty(constants: Sequence[float], sigma: Sequence[float], *, samples: int = 1000, seed: int = 0) -> dict[str, object]:
    rng = np.random.default_rng(seed); values = rng.normal(np.asarray(constants), np.asarray(sigma), size=(samples, len(constants)))
    return {"mean": np.mean(values, axis=0).tolist(), "std": np.std(values, axis=0).tolist(), "p2_5": np.percentile(values, 2.5, axis=0).tolist(), "p97_5": np.percentile(values, 97.5, axis=0).tolist()}


def density_current_viewer(outdir: Path, density: Sequence[float], current: Sequence[float]) -> Path:
    outdir.mkdir(parents=True, exist_ok=True); path = outdir / "density_current_viewer.html"
    path.write_text(f"<html><canvas id='plot' width='1000' height='500'></canvas><script>const d={json.dumps(list(density))},j={json.dumps(list(current))};const c=document.querySelector('canvas').getContext('2d');c.beginPath();d.forEach((v,i)=>{{const x=i*980/Math.max(1,d.length-1),y=480-400*v;c.lineTo(x,y)}});c.strokeStyle='blue';c.stroke();c.beginPath();j.forEach((v,i)=>{{const x=i*980/Math.max(1,j.length-1),y=480-100*v;c.lineTo(x,y)}});c.strokeStyle='red';c.stroke();</script></html>", encoding="utf-8"); return path


def export_scientific_data(path: Path, payload: dict[str, object], *, hdf5: bool = True) -> Path:
    if hdf5:
        try:
            import h5py
            with h5py.File(path.with_suffix(".h5"), "w") as handle: handle.create_dataset("json", data=json.dumps(payload).encode())
            return path.with_suffix(".h5")
        except ImportError: pass
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8"); return path


def validate_before_sync(root: Path) -> dict[str, object]:
    errors = []
    for path in root.rglob("*.json"):
        try: json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error: errors.append(f"{path}: {error}")
    return {"valid": not errors, "errors": errors, "files_checked": len(list(root.rglob("*")))}


def rollback_sync(backup: Path, target: Path) -> Path:
    if target.exists(): shutil.rmtree(target)
    shutil.copytree(backup, target); return target


__all__ = ["audit_event", "bidirectional_sync", "checkpoint", "compare_versions", "density_current_viewer", "experimental_record", "export_scientific_data", "file_hash", "fit_sensitivity", "isotopologue_records", "job_queue", "propagate_rotational_uncertainty", "rollback_sync", "sync_manifest", "total_numerical_error", "validate_before_sync"]
