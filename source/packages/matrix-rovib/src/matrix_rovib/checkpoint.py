"""Deterministic parallel execution with content-bound LOCK checkpoints."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from typing import Any


LOCK_CHECKPOINT_SCHEMA = "matrix.lock.deterministic_checkpoint.v1"


def _fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def run_checkpointed_blocks(
    keys: Sequence[str],
    solve: Callable[[str], Mapping[str, Any]],
    *,
    input_contract: Mapping[str, Any],
    checkpoint: Path | str,
    workers: int = 1,
    resume: bool = True,
) -> dict[str, Mapping[str, Any]]:
    """Run independent blocks once and persist them in canonical key order.

    Parallel completion timing never changes the returned ordering or checkpoint
    bytes. A resumed run refuses a checkpoint produced from different inputs.
    """

    ordered = tuple(str(key) for key in keys)
    from numbers import Integral
    if (not ordered or len(set(ordered)) != len(ordered)
            or isinstance(workers, bool) or not isinstance(workers, Integral) or workers < 1):
        raise ValueError("checkpointed blocks need unique keys and positive workers")
    contract = json.loads(json.dumps(dict(input_contract), allow_nan=False))
    contract_sha256 = _fingerprint(contract)
    target = Path(checkpoint)
    completed: dict[str, Mapping[str, Any]] = {}
    if target.is_file() and resume:
        payload = json.loads(target.read_text(encoding="utf-8"))
        if payload.get("schema") != LOCK_CHECKPOINT_SCHEMA:
            raise ValueError("unsupported LOCK checkpoint schema")
        if payload.get("input_sha256") != contract_sha256 or payload.get("keys") != list(ordered):
            raise ValueError("LOCK checkpoint does not belong to this problem")
        saved = payload.get("completed")
        if (not isinstance(saved, dict) or set(saved).difference(ordered)
                or any(not isinstance(value, dict) for value in saved.values())):
            raise ValueError("invalid completed block records in LOCK checkpoint")
        # json.loads accepts NaN/Infinity by default; solver outputs never do.
        json.dumps(saved, allow_nan=False)
        completed = saved
    pending = tuple(key for key in ordered if key not in completed)

    def certified(key: str) -> tuple[str, Mapping[str, Any]]:
        result = json.loads(json.dumps(dict(solve(key)), allow_nan=False))
        return key, result

    if workers == 1:
        iterator = map(certified, pending)
        executor = None
    else:
        executor = ThreadPoolExecutor(max_workers=min(workers, len(pending) or 1))
        iterator = executor.map(certified, pending)
    try:
        for key, result in iterator:
            completed[key] = result
            _write_checkpoint(target, ordered, contract_sha256, completed)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    return {key: completed[key] for key in ordered}


def _write_checkpoint(
    target: Path,
    keys: Sequence[str],
    input_sha256: str,
    completed: Mapping[str, Mapping[str, Any]],
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": LOCK_CHECKPOINT_SCHEMA,
        "input_sha256": input_sha256,
        "keys": list(keys),
        "completed": {key: completed[key] for key in keys if key in completed},
    }
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)


__all__ = ["LOCK_CHECKPOINT_SCHEMA", "run_checkpointed_blocks"]
