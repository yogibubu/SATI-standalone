"""Single read-only compatibility boundary for pre-ZAFF schema identifiers.

The adapter never writes files and never exposes a second runtime API.  It only
normalizes historical schema identifiers while data enters the canonical ZAFF
implementation.
"""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Mapping


LEGACY_ZAFF_SCHEMA_PREFIX = "matrix.zion."
ZAFF_SCHEMA_PREFIX = "matrix.zaff."


def normalize_legacy_zaff_payload(value: Any) -> Any:
    """Return a detached payload with historical schema identifiers upgraded."""

    if isinstance(value, Mapping):
        return {
            key: normalize_legacy_zaff_payload(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [normalize_legacy_zaff_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(normalize_legacy_zaff_payload(item) for item in value)
    if isinstance(value, str) and value.startswith(LEGACY_ZAFF_SCHEMA_PREFIX):
        return ZAFF_SCHEMA_PREFIX + value.removeprefix(LEGACY_ZAFF_SCHEMA_PREFIX)
    return value


def load_zaff_json(path: Path | str) -> Any:
    """Read JSON and normalize only historical ZAFF schema identifiers."""

    source = Path(path).expanduser().resolve()
    return normalize_legacy_zaff_payload(
        json.loads(source.read_text(encoding="utf-8"))
    )


__all__ = [
    "LEGACY_ZAFF_SCHEMA_PREFIX",
    "ZAFF_SCHEMA_PREFIX",
    "load_zaff_json",
    "normalize_legacy_zaff_payload",
]
