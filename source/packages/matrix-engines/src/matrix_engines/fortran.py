"""Discovery helpers for ORACLE DVR and VPT2/VCI Fortran backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess


DVR_FORTRAN_FILES = ("path_dvr.f", "dvr_hqrii.f")
VPT2_VCI_FORTRAN_FILES = (
    "gf_core.f",
    "vpt2_core.f",
    "vci_core.f",
    "davidson_core.f",
)


@dataclass(frozen=True)
class DVRFortranLayout:
    root: Path
    source_dir: Path
    compile_script: Path
    source_paths: tuple[Path, ...]
    build_executable: Path
    bin_executable: Path


@dataclass(frozen=True)
class VPT2VCIFortranLayout:
    root: Path
    source_dir: Path
    compile_script: Path
    source_paths: tuple[Path, ...]
    object_dir: Path


def dvr_fortran_layout(repo_root: Path | None = None) -> DVRFortranLayout:
    root = _repo_root(repo_root)
    source_dir = root / "engines" / "fortran" / "dvr"
    return DVRFortranLayout(
        root=root,
        source_dir=source_dir,
        compile_script=source_dir / "compile_MAC",
        source_paths=tuple(source_dir / name for name in DVR_FORTRAN_FILES),
        build_executable=source_dir / "build" / "path_dvr",
        bin_executable=root / "bin" / "path_dvr.x",
    )


def vpt2_vci_fortran_layout(repo_root: Path | None = None) -> VPT2VCIFortranLayout:
    root = _repo_root(repo_root)
    source_dir = root / "engines" / "fortran" / "vpt2_vci"
    return VPT2VCIFortranLayout(
        root=root,
        source_dir=source_dir,
        compile_script=source_dir / "compile_check",
        source_paths=tuple(source_dir / name for name in VPT2_VCI_FORTRAN_FILES),
        object_dir=source_dir / "build",
    )


def dvr_source_paths(repo_root: Path | None = None) -> tuple[Path, ...]:
    return dvr_fortran_layout(repo_root).source_paths


def vpt2_vci_source_paths(repo_root: Path | None = None) -> tuple[Path, ...]:
    return vpt2_vci_fortran_layout(repo_root).source_paths


def validate_dvr_sources(repo_root: Path | None = None) -> tuple[Path, ...]:
    layout = dvr_fortran_layout(repo_root)
    required = (*layout.source_paths, layout.compile_script)
    return tuple(path for path in required if not path.is_file())


def validate_vpt2_vci_sources(repo_root: Path | None = None) -> tuple[Path, ...]:
    layout = vpt2_vci_fortran_layout(repo_root)
    required = (*layout.source_paths, layout.compile_script)
    return tuple(path for path in required if not path.is_file())


def dvr_executable(repo_root: Path | None = None, *, compile_if_missing: bool = True) -> Path:
    """Return the ORACLE DVR executable, compiling the vendored source when needed."""
    layout = dvr_fortran_layout(repo_root)
    candidates = (
        layout.bin_executable,
        layout.build_executable,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    for name in ("path_dvr.x", "path_dvr"):
        found = shutil.which(name)
        if found:
            return Path(found)
    if compile_if_missing and layout.compile_script.is_file():
        subprocess.run(
            [str(layout.compile_script)],
            check=True,
            cwd=layout.source_dir,
            capture_output=True,
            text=True,
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    missing = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"cannot locate ORACLE DVR executable; checked {missing} and PATH")


def _repo_root(repo_root: Path | None) -> Path:
    if repo_root is not None:
        return Path(repo_root).resolve()
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "engines").is_dir():
            return candidate
    raise RuntimeError("cannot locate MATRIX repository root")
