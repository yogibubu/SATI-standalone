"""Reproducibility manifests for scientific KEYMAKER runs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .manifest import sha256_file
from .keymaker_protocol import FROZEN_KEYMAKER_STAGES


REPRODUCIBILITY_SCHEMA = "matrix.keymaker.reproducibility-manifest.v1"


def _file_records(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"reproducibility artifact not found: {path}")
        records.append({"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return records


def build_reproducibility_manifest(
    *,
    run_id: str,
    seed: int,
    input_paths: Sequence[str | Path] = (),
    output_paths: Sequence[str | Path] = (),
    checkpoints: Sequence[Mapping[str, Any]] = (),
    tool_versions: Mapping[str, str] = (),
    completed_stages: Sequence[str] = (),
) -> dict[str, Any]:
    """Build a deterministic audit payload without copying scientific data."""

    if not str(run_id).strip():
        raise ValueError("run_id must not be empty")
    stages = tuple(completed_stages) or (FROZEN_KEYMAKER_STAGES[0],)
    if tuple(stages) != FROZEN_KEYMAKER_STAGES[: len(stages)]:
        raise ValueError("completed_stages must be an ordered frozen-protocol prefix")
    return {
        "schema": REPRODUCIBILITY_SCHEMA,
        "run_id": str(run_id),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "protocol_stages": list(stages),
        "tool_versions": {str(key): str(value) for key, value in dict(tool_versions).items()},
        "inputs": _file_records(input_paths),
        "outputs": _file_records(output_paths),
        "checkpoints": [dict(item) for item in checkpoints],
    }


def write_reproducibility_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
