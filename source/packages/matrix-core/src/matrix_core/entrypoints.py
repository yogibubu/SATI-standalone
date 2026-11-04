"""Resolve installed MATRIX commands with an explicit source-tree fallback."""

from __future__ import annotations

import shutil
import sys
import os
from pathlib import Path


def _source_tree_command() -> tuple[str, ...] | None:
    """Return the repository launcher when a MATRIX checkout is in scope.

    Keymaker must execute the same nano-MATRIX source tree that it used to
    prepare a job.  Preferring an unrelated globally installed ``matrix``
    executable silently routed LINK jobs to the historical LINK-standalone
    package and made source-tree fixes (including native SONIC back-transform)
    ineffective.
    """

    candidates: list[Path] = []
    configured = os.environ.get("MATRIX_HOME", "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())
    current = Path.cwd()
    candidates.extend((current, *current.parents))
    for candidate in candidates:
        root = candidate if candidate.is_dir() else candidate.parent
        launcher = root / "tools" / "matrix_run.py"
        if (root / "VERSION").is_file() and launcher.is_file():
            return (sys.executable, str(launcher))
    return None


def matrix_command() -> tuple[str, ...]:
    source_command = _source_tree_command()
    if source_command is not None:
        return source_command
    executable = shutil.which("matrix")
    if executable is not None:
        return (executable,)
    for parent in Path(__file__).resolve().parents:
        development_launcher = parent / "tools" / "matrix_run.py"
        if (parent / "VERSION").is_file() and development_launcher.is_file():
            return (sys.executable, str(development_launcher))
    return (sys.executable, "-m", "matrix_cli")


__all__ = ["matrix_command"]
