"""Crash-safe, collision-free publication of local files."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any


def _sync_directory(directory: Path) -> None:
    """Persist a directory entry where the platform supports directory fsync."""
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _temporary_path(target: Path) -> tuple[int, Path]:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    return descriptor, Path(name)


@contextmanager
def atomic_output_path(path: Path | str) -> Iterator[Path]:
    """Yield a unique private path and publish its completed file on success."""
    target = Path(path).expanduser()
    descriptor, temporary = _temporary_path(target)
    os.close(descriptor)
    try:
        yield temporary
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        _sync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_bytes(path: Path | str, payload: bytes) -> Path:
    """Write *payload* and atomically expose it at *path*."""
    target = Path(path).expanduser()
    descriptor, temporary = _temporary_path(target)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        _sync_directory(target.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def atomic_write_text(
    path: Path | str,
    text: str,
    *,
    encoding: str = "utf-8",
) -> Path:
    """Encode and atomically publish text at *path*."""
    return atomic_write_bytes(path, text.encode(encoding))


def atomic_json_write(
    path: Path | str,
    payload: Any,
    *,
    indent: int | None = 2,
    sort_keys: bool = True,
    ensure_ascii: bool = True,
    allow_nan: bool = True,
) -> Path:
    """Serialize one complete JSON document and publish it atomically."""
    text = json.dumps(
        payload,
        indent=indent,
        sort_keys=sort_keys,
        ensure_ascii=ensure_ascii,
        allow_nan=allow_nan,
    )
    return atomic_write_text(path, text + "\n")


def atomic_copy(source: Path | str, destination: Path | str) -> Path:
    """Copy metadata and bytes privately, then atomically expose the copy."""
    target = Path(destination).expanduser()
    descriptor, temporary = _temporary_path(target)
    os.close(descriptor)
    try:
        shutil.copy2(source, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        _sync_directory(target.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


__all__ = [
    "atomic_copy",
    "atomic_json_write",
    "atomic_output_path",
    "atomic_write_bytes",
    "atomic_write_text",
]
