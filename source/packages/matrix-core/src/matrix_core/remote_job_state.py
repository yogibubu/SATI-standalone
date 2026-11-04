"""Strict reader for the persistent remote QM job-state contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REMOTE_JOB_STATE_SCHEMA = "matrix.remote_job_state.v1"
REMOTE_JOB_STATUSES = ("SUBMITTED", "RUNNING", "COMPLETED", "FAILED", "UNKNOWN")


def read_remote_job_state(path: Path | str) -> dict[str, Any]:
    """Read and validate one remote state sidecar before exposing it to GUI code."""
    target = Path(path).expanduser().resolve()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid remote job state: {target}") from exc
    if not isinstance(payload, dict):
        raise ValueError("remote job state must be a JSON object")
    required = {"schema", "job", "engine", "status", "pid", "workdir", "input", "updated_epoch"}
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError("remote job state missing: " + ", ".join(missing))
    if payload["schema"] != REMOTE_JOB_STATE_SCHEMA:
        raise ValueError("unsupported remote job state schema")
    if payload["status"] not in REMOTE_JOB_STATUSES:
        raise ValueError("unsupported remote job state status")
    if not all(isinstance(payload[key], str) and payload[key].strip() for key in ("job", "engine", "workdir", "input")):
        raise ValueError("remote job state identity fields must be non-empty strings")
    if payload["pid"] is not None and (not isinstance(payload["pid"], int) or payload["pid"] < 1):
        raise ValueError("remote job state pid must be null or a positive integer")
    if not isinstance(payload["updated_epoch"], int) or payload["updated_epoch"] < 0:
        raise ValueError("remote job state updated_epoch must be a non-negative integer")
    return dict(payload)


__all__ = ["REMOTE_JOB_STATE_SCHEMA", "REMOTE_JOB_STATUSES", "read_remote_job_state"]
