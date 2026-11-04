"""Unified electronic-reference lookup for the LCB26 library.

The former LCB25 datasets are promoted into the single LCB26 namespace.
Consumers query one API and do not need historical library labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .lcb26_index import LCB26IndexError, load_lcb26_record, query_lcb26


LCB26_ELECTRONIC_SCHEMA = "matrix.lcb26.unified_electronic_reference.v1"


class LCB26ElectronicError(ValueError):
    """Raised when a unified electronic-reference lookup is invalid."""


def lcb26_electronic_paths(cache_root: Path) -> tuple[Path, ...]:
    root = Path(cache_root).expanduser().resolve() / "enriched"
    if not root.is_dir():
        raise LCB26ElectronicError(f"LCB26 enriched directory is missing: {root}")
    return tuple(sorted(root.rglob("*.cm5_mayer.json")))


def resolve_lcb26_electronic(cache_root: Path, identifier: str) -> dict[str, Any]:
    """Return one raw CM5/Mayer record from unified LCB26 storage.

    The query index accepts the stable identifier, filename stem, display name
    and aliases.  A scan remains as a compatibility fallback for pre-indexed
    caches and test fixtures.
    """

    requested = str(identifier).strip()
    if not requested:
        raise LCB26ElectronicError("LCB26 electronic identifier cannot be empty")
    try:
        indexed = query_lcb26(cache_root, identifier=requested)
    except (LCB26IndexError, OSError):
        indexed = ()
    if len(indexed) == 1:
        return load_lcb26_record(cache_root, indexed[0])
    if len(indexed) > 1:
        raise LCB26ElectronicError(
            "ambiguous LCB26 electronic identifier: "
            + ", ".join(str(row["identifier"]) for row in indexed)
        )
    paths = lcb26_electronic_paths(cache_root)
    matches: list[Path] = []
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LCB26ElectronicError(f"invalid electronic record: {path}") from exc
        record_id = str(record.get("identifier", ""))
        if requested in {record_id, path.stem, path.name} or record_id.endswith(requested):
            matches.append(path)
    if not matches:
        raise LCB26ElectronicError(f"LCB26 electronic record not found: {identifier!r}")
    if len(matches) != 1:
        raise LCB26ElectronicError(
            "ambiguous LCB26 electronic identifier: "
            + ", ".join(str(path) for path in matches)
        )
    return json.loads(matches[0].read_text(encoding="utf-8"))


__all__ = ["LCB26_ELECTRONIC_SCHEMA", "LCB26ElectronicError", "lcb26_electronic_paths", "resolve_lcb26_electronic"]
