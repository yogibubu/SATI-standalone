"""Canonical MATRIX monorepo version."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _source_version() -> str:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "VERSION"
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8").strip()
    raise FileNotFoundError("VERSION is not present above matrix_core")


try:
    MATRIX_VERSION = _source_version()
except FileNotFoundError:
    try:
        MATRIX_VERSION = version("matrix-core")
    except PackageNotFoundError:
        raise RuntimeError("MATRIX version is unavailable") from None


__all__ = ["MATRIX_VERSION"]
