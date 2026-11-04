"""Compatibility facade for the CLI now owned by :mod:`matrix_cli`.

Importing :mod:`matrix_core` remains dependency-light. New code and installed
commands use the ``matrix-cli`` entry points directly.
"""

from __future__ import annotations

from importlib import import_module
import os
from pathlib import Path
import sys
from typing import Any


def find_repo_root(start: Path | None = None) -> Path:
    env_root = os.environ.get("MATRIX_HOME") or os.environ.get("ORACLE_HOME")
    if env_root:
        return Path(env_root).expanduser().resolve()
    search_from = Path.cwd() if start is None else Path(start).resolve()
    for candidate in (search_from, *search_from.parents):
        if (candidate / "packages").is_dir() and (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd().resolve()


def add_repo_packages_to_path(repo_root: Path | None = None) -> None:
    root = find_repo_root() if repo_root is None else Path(repo_root)
    packages = root / "packages"
    if not packages.is_dir():
        return
    for src in sorted(packages.glob("*/src")):
        text = str(src)
        if text not in sys.path:
            sys.path.insert(0, text)


def _implementation():
    try:
        return import_module("matrix_cli.cli")
    except ModuleNotFoundError as exc:
        if exc.name == "matrix_cli":
            raise RuntimeError(
                "MATRIX command support is not installed; install matrix-cli"
            ) from exc
        raise


def main(*args: Any, **kwargs: Any) -> int:
    return int(_implementation().main(*args, **kwargs))


def matrix_main(*args: Any, **kwargs: Any) -> int:
    return int(_implementation().matrix_main(*args, **kwargs))


def link_main(*args: Any, **kwargs: Any) -> int:
    return int(_implementation().link_main(*args, **kwargs))


def smith_main(*args: Any, **kwargs: Any) -> int:
    return int(_implementation().smith_main(*args, **kwargs))


def __getattr__(name: str) -> Any:
    value = getattr(_implementation(), name)
    globals()[name] = value
    return value
