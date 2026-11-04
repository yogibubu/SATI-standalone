"""Workspace path safety checks."""
from __future__ import annotations
from pathlib import Path

def safe_workspace_path(root: str | Path, candidate: str | Path) -> Path:
    base = Path(root).expanduser().resolve(); target = (base / Path(candidate)).resolve() if not Path(candidate).is_absolute() else Path(candidate).expanduser().resolve()
    if base != target and base not in target.parents: raise ValueError(f"path escapes workspace: {candidate}")
    return target
