from __future__ import annotations
import json
from pathlib import Path

from matrix_core import atomic_json_write

def write_smith_checkpoint(payload: dict[str, object], path: str | Path) -> Path:
    target = Path(path)
    atomic_json_write(target, {"schema": "matrix.smith.checkpoint.v1", **payload})
    return target

def read_smith_checkpoint(path: str | Path) -> dict[str, object] | None:
    target=Path(path)
    if not target.is_file(): return None
    payload=json.loads(target.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) and payload.get("schema")=="matrix.smith.checkpoint.v1" else None
