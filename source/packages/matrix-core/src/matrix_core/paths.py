from __future__ import annotations

from pathlib import Path


def repo_root(start: Path | None = None) -> Path:
    here = Path(start).resolve() if start is not None else Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "packages").is_dir():
            return candidate
    raise RuntimeError("cannot locate MATRIX repository root")
