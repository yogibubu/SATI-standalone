"""Artifact integrity helpers."""
from __future__ import annotations
import hashlib
from pathlib import Path

def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def verify_hash(path: str | Path, expected: str) -> bool:
    return Path(path).is_file() and sha256_path(path) == expected
