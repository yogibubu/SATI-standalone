"""Software-hardening primitives shared by the final MATRIX workflow."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import platform
import sys
import time
from typing import Iterator, Mapping


class MatrixWorkflowError(RuntimeError):
    """Stable diagnostic exception for user-facing workflows."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}"); self.code = code; self.message = message


@dataclass(frozen=True)
class WorkflowConfig:
    schema: str = "matrix.tunneling.config.v1"
    backend: str = "matrix_gfn2"
    workers: int = 1
    levels: int = 12
    output_directory: str = "tunneling-results"


def load_workflow_config(path: Path) -> WorkflowConfig:
    text = path.read_text(encoding="utf-8")
    if path.suffix.casefold() == ".toml":
        import tomllib
        payload = tomllib.loads(text)
    else:
        try:
            import yaml
            payload = yaml.safe_load(text) or {}
        except ImportError:
            payload = json.loads(text)
    return WorkflowConfig(**{key: value for key, value in payload.items() if key in WorkflowConfig.__dataclass_fields__})


def migrate_schema(payload: Mapping[str, object], target: str = "matrix.tunneling.config.v1") -> dict[str, object]:
    output = dict(payload); output["schema"] = target; output.setdefault("workers", 1); output.setdefault("levels", 12); return output


def api_registry() -> dict[str, str]:
    return {
        "scan": "matrix_dvr.esalc_scan.run_esalc_scan",
        "dvr": "matrix_dvr.multimode.solve_tensor_dvr",
        "lock": "matrix_dvr.rovib_diagnostics.write_lock_export",
        "qualification": "tools.qualification.final_qualification",
    }


@contextmanager
def transactional_cache(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    acquired = False
    try:
        lock_path.open("x").close(); acquired = True; yield
    except FileExistsError as error:
        raise MatrixWorkflowError("CACHE_LOCKED", f"cache lock exists: {lock_path}") from error
    finally:
        if acquired: lock_path.unlink(missing_ok=True)


def configure_logging(path: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("matrix.tunneling"); logger.setLevel(level)
    if not logger.handlers:
        handler = logging.FileHandler(path) if path else logging.StreamHandler(); handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s")); logger.addHandler(handler)
    return logger


def profile_call(function, *args, **kwargs):
    started = time.perf_counter(); value = function(*args, **kwargs); return value, {"seconds": time.perf_counter() - started, "function": getattr(function, "__name__", "callable")}


def diagnostic(code: str, message: str, *, hint: str = "") -> dict[str, str]:
    return {"code": code, "message": message, "hint": hint}


def cli_command_registry() -> dict[str, str]:
    return {
        "scan": "esalc-scan",
        "fit": "lock-fit",
        "view": "dashboard",
        "qualify": "final-qualification",
        "sync": "oracle-sync",
    }


def safe_artifact_path(root: Path, requested: str) -> Path:
    target = (root / requested).resolve()
    if root.resolve() not in target.parents and target != root.resolve(): raise MatrixWorkflowError("PATH_ESCAPE", "artifact path leaves workflow directory")
    return target


def offline_package_manifest(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "matrix.tunneling.offline_package.v1", "python": sys.version, "platform": platform.platform(), "files": sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())}
    path = root / "offline_package_manifest.json"; path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8"); return path


def api_documentation(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# MATRIX tunneling API\n\n" + "\n".join(f"- `{key}`: `{value}`" for key, value in api_registry().items()) + "\n", encoding="utf-8"); return path


def compatibility_report() -> dict[str, object]:
    return {"python": platform.python_version(), "platform": platform.platform(), "supported_python": ["3.11", "3.12", "3.13"], "status": "supported" if sys.version_info >= (3, 11) else "unsupported"}


def scientific_artifact_export(path: Path, payload: Mapping[str, object]) -> Path:
    if path.suffix.casefold() in {".h5", ".hdf5"}:
        try:
            import h5py
            with h5py.File(path, "w") as handle: handle.create_dataset("payload_json", data=json.dumps(dict(payload)).encode())
            return path
        except ImportError: pass
    path.write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8"); return path


def semantic_release_manifest(root: Path, version: str = "1.0.0") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    payload = {"name": "nano-MATRIX tunneling", "version": version, "semver": True, "changelog": "CHANGELOG.md", "api": api_registry()}; path = root / "release_manifest.json"; path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8"); return path


__all__ = ["MatrixWorkflowError", "WorkflowConfig", "api_documentation", "api_registry", "cli_command_registry", "compatibility_report", "configure_logging", "diagnostic", "load_workflow_config", "migrate_schema", "offline_package_manifest", "profile_call", "safe_artifact_path", "scientific_artifact_export", "semantic_release_manifest", "transactional_cache"]
