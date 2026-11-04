from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Iterable

from matrix_core import build_run_manifest, sha256_file, write_manifest
from matrix_core.paths import repo_root
from matrix_engines.gicforge_reference import gicforge_reference_executable
from .gic_symmetry import gicsym_requested, symmetry_postprocess_requested, write_gic_symmetry_files


GICFORGE_OUTPUTS = (
    "gicforge.out",
    "provout",
    "gauin",
    "gauin.raw",
    "gauin.symm",
    "gicsym",
    "gic_symmetry_diagnostics.json",
    "sycart.xyz",
    "symmetrized.xyz",
    "msrin",
    "VPT2in",
    "bmat.out",
)


class GICForgeError(RuntimeError):
    """Raised when the GICForge backend cannot be executed successfully."""


@dataclass(frozen=True)
class GICForgeResult:
    workdir: Path
    executable: Path
    logfile: Path
    files: dict[str, Path]
    manifest: Path


def run_gicforge(
    workdir: Path,
    *,
    executable: Path | None = None,
    output_names: Iterable[str] = GICFORGE_OUTPUTS,
    symmetrize: bool = True,
    symmetry_backend: str | None = None,
) -> GICForgeResult:
    """Run GICForge in `workdir` and write a normalized manifest."""
    run_dir = Path(workdir)
    run_dir.mkdir(parents=True, exist_ok=True)
    exe = Path(executable) if executable is not None else gicforge_reference_executable()
    logfile = run_dir / "gicforge.log"

    try:
        with logfile.open("w", encoding="utf-8") as log:
            subprocess.run(
                [str(exe)],
                cwd=run_dir,
                stdout=log,
                stderr=log,
                check=True,
            )
    except FileNotFoundError as exc:
        raise GICForgeError(f"Executable not found: {exe}") from exc
    except subprocess.CalledProcessError as exc:
        raise GICForgeError(f"GICForge failed, see {logfile}") from exc

    _copy_legacy_report(run_dir)
    effective_symmetrize = symmetrize or gicsym_requested(run_dir)
    do_symmetry_post = effective_symmetrize or symmetry_postprocess_requested(run_dir)
    if do_symmetry_post:
        write_gic_symmetry_files(
            run_dir, symmetrize_gics=effective_symmetrize, symmetry_backend=symmetry_backend
        )
    else:
        _remove_symmetry_outputs(run_dir)
    files = _collect_outputs(run_dir, output_names)
    manifest = _write_gicforge_manifest(
        run_dir, exe, logfile, files, symmetrize=effective_symmetrize
    )
    return GICForgeResult(
        workdir=run_dir,
        executable=exe,
        logfile=logfile,
        files=files,
        manifest=manifest,
    )


def _copy_legacy_report(run_dir: Path) -> None:
    legacy_output = run_dir / "provout"
    readable_output = run_dir / "gicforge.out"
    if legacy_output.exists():
        readable_output.write_text(
            legacy_output.read_text(encoding="utf-8", errors="replace"),
            encoding="utf-8",
        )


def _remove_symmetry_outputs(run_dir: Path) -> None:
    for name in (
        "gauin.raw",
        "gauin.symm",
        "gicsym",
        "gic_symmetry_diagnostics.json",
        "sycart.xyz",
        "symmetrized.xyz",
    ):
        path = run_dir / name
        if path.exists():
            path.unlink()


def _collect_outputs(run_dir: Path, output_names: Iterable[str]) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for name in output_names:
        path = run_dir / name
        if path.exists():
            files[name] = path
    return files


def _optional_checksum(path: Path) -> str | None:
    return sha256_file(path) if path.exists() else None


def _write_gicforge_manifest(
    run_dir: Path,
    executable: Path,
    logfile: Path,
    files: dict[str, Path],
    *,
    symmetrize: bool,
) -> Path:
    input_checksums = {
        name: checksum
        for name in ("provin", "xyzin")
        if (checksum := _optional_checksum(run_dir / name)) is not None
    }
    output_checksums = {
        name: sha256_file(path) for name, path in sorted(files.items()) if path.is_file()
    }
    legacy_manifest = {
        "workflow": "gicforge",
        "executable": str(executable),
        "executable_sha256": _optional_checksum(executable),
        "logfile": str(logfile),
        "symmetrize": symmetrize,
        "inputs": input_checksums,
        "outputs": {name: str(path) for name, path in sorted(files.items())},
        "output_sha256": output_checksums,
    }
    manifest = build_run_manifest(
        workflow="gicforge",
        status="completed",
        run_dir=run_dir,
        inputs={name: run_dir / name for name in ("provin", "xyzin") if (run_dir / name).exists()},
        outputs=files,
        backend={
            "name": "gicforge",
            "executable": str(executable),
            "executable_sha256": _optional_checksum(executable),
            "logfile": str(logfile),
            "symmetrize": symmetrize,
            "git_commit": _git_commit(),
            "git_dirty": _git_dirty(),
        },
    ).to_dict()
    symmetry_diagnostics = _read_optional_json(run_dir / "gic_symmetry_diagnostics.json")
    if symmetry_diagnostics:
        manifest.setdefault("parameters", {})["gic_symmetry"] = {
            key: symmetry_diagnostics.get(key)
            for key in (
                "schema",
                "symmetry_backend",
                "point_group",
                "python_point_group",
                "fortran_point_group",
                "operation_order",
                "irreps",
                "targets",
                "counts",
                "b_ranks",
                "class_targets",
                "class_counts",
                "raw_class_targets",
                "sources",
                "strict_clean",
                "tolerances",
            )
            if key in symmetry_diagnostics
        }
    manifest.update({"legacy": legacy_manifest})
    return write_manifest(run_dir / "gicforge_manifest.json", manifest)


def _read_optional_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except Exception:
        return None
    return completed.stdout.strip() or None


def _git_dirty() -> bool | None:
    try:
        completed = subprocess.run(
            ["git", "status", "--short"],
            cwd=repo_root(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except Exception:
        return None
    return bool(completed.stdout.strip())
