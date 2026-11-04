"""Content-addressed cache primitives for repeatable ORACLE operations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from matrix_core import (
    atomic_copy,
    atomic_json_write,
    scientific_cache_envelope,
    unwrap_scientific_cache,
)


def cache_key(*parts: object) -> str:
    encoded = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def cache_path(cache_dir: Path | str, key: str, suffix: str = ".json") -> Path:
    target = Path(cache_dir).expanduser() / "oracle" / key[:2] / f"{key}{suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def read_cached(
    cache_dir: Path | str,
    key: str,
    *,
    expected_state_sha256: str | None = None,
) -> dict[str, Any] | None:
    target = cache_path(cache_dir, key)
    if not target.is_file():
        return None
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    if expected_state_sha256 is None:
        # Plain legacy records remain readable. New envelopes unwrap only when
        # the caller explicitly supplies the state it expects.
        return payload
    return unwrap_scientific_cache(
        payload,
        expected_state_sha256=expected_state_sha256,
    )


def write_cached(
    cache_dir: Path | str,
    key: str,
    payload: dict[str, Any],
    *,
    state_sha256: str | None = None,
) -> Path:
    target = cache_path(cache_dir, key)
    stored = (
        payload
        if state_sha256 is None
        else scientific_cache_envelope(payload, state_sha256=state_sha256)
    )
    atomic_json_write(target, stored, allow_nan=False)
    return target


def read_cached_report(
    cache_dir: Path | str,
    key: str,
    *,
    expected_schema: str = "matrix.oracle.analysis.v2",
    expected_state_sha256: str | None = None,
) -> dict[str, Any] | None:
    payload = read_cached(
        cache_dir,
        key,
        expected_state_sha256=expected_state_sha256,
    )
    if payload is None or payload.get("schema") != expected_schema:
        return None
    return payload


def cache_artifact(cache_dir: Path | str, key: str, source: Path | str) -> Path:
    target = cache_path(cache_dir, key, suffix=Path(source).suffix or ".artifact")
    atomic_copy(source, target)
    return target


def restore_artifact(
    cache_dir: Path | str,
    key: str,
    destination: Path | str,
    suffix: str = ".xyzin",
) -> bool:
    cached = cache_path(cache_dir, key, suffix=suffix)
    if not cached.is_file():
        return False
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_copy(cached, target)
    return True
