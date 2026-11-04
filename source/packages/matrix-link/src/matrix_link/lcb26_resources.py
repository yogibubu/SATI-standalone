"""Unified LCB26 geometry/electronic resource resolution."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from .lcb26 import resolve_lcb26_geometry
from .lcb26 import LCB26_DATASETS
from .electronic_library import resolve_lcb26_electronic

def resolve_lcb26_resource(cache_root: Path, identifier: str, *, kind: str = "auto") -> Path | dict[str, Any]:
    if kind not in {"auto", "geometry", "electronic"}: raise ValueError("kind must be auto, geometry or electronic")
    if kind in {"auto", "geometry"}:
        for dataset in LCB26_DATASETS:
            try: return resolve_lcb26_geometry(cache_root, identifier, dataset=dataset)
            except Exception: pass
        if kind == "geometry": raise FileNotFoundError(identifier)
    return resolve_lcb26_electronic(cache_root, identifier)
