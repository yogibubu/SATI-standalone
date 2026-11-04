"""Lossless report compatibility helpers."""
from __future__ import annotations
from typing import Any

def migrate_analysis_report(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    schema = result.get("schema")
    if schema == "matrix.oracle.analysis.v1":
        result["schema"] = "matrix.oracle.analysis.v2"
    result.setdefault("provenance", {"backend": "legacy-unknown"})
    result.setdefault("cache_key", None)
    return result
