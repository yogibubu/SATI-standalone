"""Portable artifact digest/signature helpers for MATRIX manifests."""
from __future__ import annotations
import hashlib
from pathlib import Path

def artifact_digest(path: str | Path) -> str:
    digest = hashlib.sha256();
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()

def verify_artifact(path: str | Path, expected_sha256: str) -> bool:
    return artifact_digest(path).casefold() == str(expected_sha256).casefold()
