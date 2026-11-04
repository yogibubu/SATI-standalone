"""Discovery helpers for the vendored puckering DVR Python backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PuckeringDVRLayout:
    root: Path
    engine_dir: Path
    path_analysis_script: Path
    fortran_bridge_script: Path


def puckering_dvr_layout(repo_root: Path | None = None) -> PuckeringDVRLayout:
    root = _repo_root(repo_root)
    engine_dir = root / "engines" / "puckering_dvr"
    return PuckeringDVRLayout(
        root=root,
        engine_dir=engine_dir,
        path_analysis_script=engine_dir / "scripts" / "mw_path_dvr.py",
        fortran_bridge_script=engine_dir / "scripts" / "fortran_bridge" / "run_fortran_dvr.py",
    )


def validate_puckering_dvr_backend(repo_root: Path | None = None) -> tuple[Path, ...]:
    layout = puckering_dvr_layout(repo_root)
    required = (
        layout.path_analysis_script,
        layout.fortran_bridge_script,
        layout.engine_dir / "requirements.txt",
    )
    return tuple(path for path in required if not path.is_file())


def _repo_root(repo_root: Path | None) -> Path:
    if repo_root is not None:
        return Path(repo_root).resolve()
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "engines").is_dir():
            return candidate
    raise RuntimeError("cannot locate MATRIX repository root")
