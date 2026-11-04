"""SMITH JSON cache with optional scientific-state invalidation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

from matrix_core import atomic_json_write, scientific_cache_envelope, unwrap_scientific_cache


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def smith_cache_key(*parts: object) -> str:
    return hashlib.sha256(
        json.dumps(parts, sort_keys=True, default=str).encode()
    ).hexdigest()


def cache_file(cache_dir: str | Path, key: str, suffix: str = ".json") -> Path:
    target = Path(cache_dir) / "smith" / key[:2] / (key + suffix)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def load_json(
    cache_dir: str | Path,
    key: str,
    *,
    expected_state_sha256: str | None = None,
) -> dict[str, object] | None:
    target = cache_file(cache_dir, key)
    if not target.is_file():
        return None
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    if expected_state_sha256 is None:
        return payload
    return unwrap_scientific_cache(
        payload,
        expected_state_sha256=expected_state_sha256,
    )


def store_json(
    cache_dir: str | Path,
    key: str,
    payload: dict[str, object],
    *,
    state_sha256: str | None = None,
) -> Path:
    target = cache_file(cache_dir, key)
    stored = (
        payload
        if state_sha256 is None
        else scientific_cache_envelope(payload, state_sha256=state_sha256)
    )
    atomic_json_write(target, stored, allow_nan=False)
    return target


def store_compiled_plan(cache_dir, key, definition, *, state_sha256: str) -> Path:
    """Persist the native ORACLE plan and GIC combination under scientific identity."""

    from .evaluation import compiled_smith_plan_payload

    return store_json(
        cache_dir,
        key,
        compiled_smith_plan_payload(definition),
        state_sha256=state_sha256,
    )


def load_compiled_plan(cache_dir, key, definition, *, state_sha256: str):
    """Load a compiled plan only when state and frozen-definition identities match."""

    from .evaluation import compiled_smith_plan_from_payload

    payload = load_json(cache_dir, key, expected_state_sha256=state_sha256)
    if payload is None:
        return None
    return compiled_smith_plan_from_payload(payload, definition)


def store_support_matching(cache_dir, key, matcher, *, state_sha256: str) -> Path:
    return store_json(cache_dir, key, matcher.payload(), state_sha256=state_sha256)


def load_support_matching(cache_dir, key, *, state_sha256: str):
    from .minimum_sparse_certificate import IncrementalSupportMatcher

    payload = load_json(cache_dir, key, expected_state_sha256=state_sha256)
    return None if payload is None else IncrementalSupportMatcher.from_payload(payload)


def store_sparse_b_matrix(cache_dir, key, matrix, *, state_sha256: str) -> Path:
    """Persist CSR arrays independently so warm loads can memory-map them."""

    from .bmatrix import SparseBMatrix

    if not isinstance(matrix, SparseBMatrix):
        raise TypeError("sparse cache accepts only SparseBMatrix")
    root = Path(cache_dir) / "smith" / key[:2] / key
    root.mkdir(parents=True, exist_ok=True)
    arrays = matrix.csr_arrays()
    names = ("data", "indices", "indptr")
    checksums: dict[str, str] = {}
    for name, array in zip(names, arrays, strict=True):
        target = root / f"{name}.npy"
        handle, temporary_name = tempfile.mkstemp(
            prefix=f".{name}.", suffix=".npy", dir=root
        )
        os.close(handle)
        temporary = Path(temporary_name)
        try:
            with temporary.open("wb") as stream:
                import numpy as np

                np.save(stream, array, allow_pickle=False)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        checksums[name] = _file_sha256(target)
    manifest = {
        "schema": "matrix.smith.mmap_sparse_b.v1",
        "state_sha256": str(state_sha256),
        "column_count": matrix.column_count,
        "row_labels": list(matrix.row_labels),
        "backend": matrix.backend,
        "checksums": checksums,
    }
    atomic_json_write(root / "manifest.json", manifest, allow_nan=False)
    return root


def load_sparse_b_matrix(cache_dir, key, *, state_sha256: str):
    """Restore read-only memory-mapped CSR buffers after checksum validation."""

    import numpy as np
    from .bmatrix import SparseBMatrix

    root = Path(cache_dir) / "smith" / key[:2] / key
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "matrix.smith.mmap_sparse_b.v1" or (
            manifest.get("state_sha256") != str(state_sha256)
        ):
            return None
        arrays = []
        for name in ("data", "indices", "indptr"):
            target = root / f"{name}.npy"
            if _file_sha256(target) != manifest["checksums"][name]:
                return None
            arrays.append(np.load(target, mmap_mode="r", allow_pickle=False))
        return SparseBMatrix.from_csr(
            *arrays,
            column_count=int(manifest["column_count"]),
            row_labels=tuple(str(value) for value in manifest.get("row_labels", ())),
            backend=str(manifest.get("backend", "oracle-sparse-bmatrix.v1")),
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def prune_smith_cache(cache_dir, *, maximum_bytes: int) -> tuple[int, int]:
    """Apply a deterministic least-recently-used byte budget to cache entries."""

    root = Path(cache_dir) / "smith"
    budget = int(maximum_bytes)
    if budget < 0:
        raise ValueError("cache byte budget must be non-negative")
    if not root.is_dir():
        return 0, 0
    files = tuple(path for path in root.rglob("*") if path.is_file())
    total = sum(path.stat().st_size for path in files)
    removed = 0
    for path in sorted(files, key=lambda item: (item.stat().st_atime_ns, str(item))):
        if total <= budget:
            break
        size = path.stat().st_size
        path.unlink(missing_ok=True)
        total -= size
        removed += 1
    return removed, total
