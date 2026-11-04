"""Canonical cross-tool paths exposed to Keymaker."""
from __future__ import annotations
from pathlib import Path

def matrix_library_paths(root: str | Path) -> dict[str, Path]:
    base = Path(root).expanduser().resolve()
    return {
        "lcb26_root": base / "data" / "lcb26",
        "lcb26_pl2_geometries": base / "data" / "lcb26" / "geometries" / "PL2",
        "lcb26_pcs2_geometries": base / "data" / "lcb26" / "geometries" / "PCS2",
        "lcb26_electronic": base / "data" / "lcb26" / "enriched",
        "oracle_cache": base / "data" / "oracle" / "cache",
        "architect_provenance": base / "data" / "architect" / "provenance",
        "cypher_diagnostics": base / "data" / "cypher" / "diagnostics",
        "niobe_reports": base / "data" / "niobe" / "reports",
        "keymaker_runs": base / "runs",
    }

__all__ = ["matrix_library_paths"]
