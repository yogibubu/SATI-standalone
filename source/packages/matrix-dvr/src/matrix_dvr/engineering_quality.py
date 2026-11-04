"""Final software-engineering infrastructure for MATRIX tunneling."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol


class QMBackend(Protocol):
    def evaluate(self, coordinates: Any) -> dict[str, Any]: ...


class BackendRegistry:
    """Dependency-injection registry for QM engines and storage backends."""
    def __init__(self) -> None:
        self._backends: dict[str, Callable[..., QMBackend]] = {}

    def register(self, name: str, factory: Callable[..., QMBackend]) -> None:
        self._backends[name] = factory

    def create(self, name: str, **options: Any) -> QMBackend:
        if name not in self._backends:
            raise KeyError(f"unknown QM backend: {name}")
        return self._backends[name](**options)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._backends))


@dataclass(frozen=True)
class WorkflowPolicy:
    retry_count: int = 2
    timeout_seconds: float = 3600.0
    cache_mode: str = "sqlite"
    log_level: str = "INFO"


def unified_schema(payload: dict[str, Any], schema: str = "matrix.tunneling.artifact.v1") -> dict[str, Any]:
    return {"schema": schema, "payload": payload}


def migrate_api_payload(payload: dict[str, Any], source_version: str, target_version: str = "v1") -> dict[str, Any]:
    result = dict(payload); result["api_version"] = target_version; result["migrated_from"] = source_version; return result


def stream_jsonl(path: Path, rows: Iterator[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    return path


def stream_scientific_table(path: Path, rows: Iterator[dict[str, Any]]) -> Path:
    """Stream to HDF5 when available, otherwise JSONL."""
    try:
        import h5py
        values = list(rows)
        with h5py.File(path.with_suffix(".h5"), "w") as handle:
            handle.create_dataset("rows_json", data=json.dumps(values).encode())
        return path.with_suffix(".h5")
    except ImportError:
        return stream_jsonl(path.with_suffix(".jsonl"), iter(rows))


def authenticated_token(token: str, supplied: str) -> bool:
    import hmac
    return hmac.compare_digest(str(token), str(supplied))


def rest_service_contract(root: Path, token_env: str = "MATRIX_API_TOKEN") -> Path:
    payload = {"schema": "matrix.tunneling.rest.v1", "authentication": "bearer", "token_env": token_env, "routes": {"POST /runs": "submit", "GET /runs": "list", "GET /runs/{id}": "status", "GET /artifacts/{id}": "download"}}
    root.mkdir(parents=True, exist_ok=True); path = root / "rest_service.json"; path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8"); return path


def generate_release_notes(commits: list[str], path: Path) -> Path:
    path.write_text("# Release notes\n\n" + "\n".join(f"- {item}" for item in commits) + "\n", encoding="utf-8"); return path


def coverage_command() -> list[str]:
    return ["python", "-m", "pytest", "--cov=packages/matrix-dvr/src/matrix_dvr", "--cov-report=term-missing"]


def property_test_contract() -> dict[str, Any]:
    return {"properties": ["metric_symmetry", "unit_roundtrip", "wavepacket_norm", "permutation_invariance", "cache_idempotence"]}


def end_to_end_contract() -> dict[str, Any]:
    return {"pipeline": ["oracle_input", "sonic_scan", "dvr_solve", "wms_rot_export", "scientific_report", "oracle_manifest"]}


__all__ = ["BackendRegistry", "QMBackend", "WorkflowPolicy", "authenticated_token", "coverage_command", "end_to_end_contract", "generate_release_notes", "migrate_api_payload", "property_test_contract", "rest_service_contract", "stream_jsonl", "stream_scientific_table", "unified_schema"]
