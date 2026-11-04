from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys

from matrix_core import (
    build_run_manifest,
    key_value_section_lines,
    parse_key_value_section,
    read_sectioned_lines,
    replace_section,
    section_content,
)
from matrix_engines import dvr_executable, puckering_dvr_layout


FORTRAN_SOLVERS = {"fortran-sinc-dvr", "fortran-gaussian"}
ORACLE_XYZ_DVR_SCHEMA = "oracle.xyz.dvr.v1"


@dataclass(frozen=True)
class DVRRequest:
    repo_root: Path
    log_path: Path
    outdir: Path
    figdir: Path
    prefix: str = "puckering_dvr"
    boundary: str = "periodic"
    solver: str = "fourier"
    levels: int = 12
    compute_rotconst: bool = True
    label_cremer_pople: bool = True
    check_only: bool = False
    workers: int = 1
    python_executable: str = sys.executable

    @property
    def dvr_root(self) -> Path:
        return puckering_dvr_layout(self.repo_root).engine_dir

    @property
    def script(self) -> Path:
        return puckering_dvr_layout(self.repo_root).path_analysis_script

    @property
    def fortran_bridge(self) -> Path:
        return puckering_dvr_layout(self.repo_root).fortran_bridge_script

    @property
    def normalized_prefix(self) -> str:
        return self.prefix or "puckering_dvr"

    @property
    def grid_csv(self) -> Path:
        return self.outdir / f"{self.normalized_prefix}_grid.csv"


@dataclass(frozen=True)
class DVRSection:
    log_path: Path
    outdir: Path
    figdir: Path
    prefix: str = "puckering_dvr"
    boundary: str = "periodic"
    solver: str = "fourier"
    compute_rotconst: bool = True
    label_cremer_pople: bool = True
    check_only: bool = False
    manifest_path: Path | None = None
    grid_csv: Path | None = None
    summary: Path | None = None
    levels: Path | None = None
    outputs: Mapping[str, Path] | None = None
    status: str = "prepared"
    source: str = "gaussian-log"
    schema: str = ORACLE_XYZ_DVR_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "log_path", Path(self.log_path))
        object.__setattr__(self, "outdir", Path(self.outdir))
        object.__setattr__(self, "figdir", Path(self.figdir))
        for attr in ("manifest_path", "grid_csv", "summary", "levels"):
            value = getattr(self, attr)
            if value is not None:
                object.__setattr__(self, attr, Path(value))
        outputs = {
            _normalize_output_key(name): Path(path)
            for name, path in dict(self.outputs or {}).items()
            if path is not None
        }
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "prefix", self.prefix or "puckering_dvr")
        object.__setattr__(self, "compute_rotconst", bool(self.compute_rotconst))
        object.__setattr__(self, "label_cremer_pople", bool(self.label_cremer_pople))
        object.__setattr__(self, "check_only", bool(self.check_only))

    def with_outputs(
        self, outputs: Mapping[str, Path], *, status: str | None = None
    ) -> "DVRSection":
        return DVRSection(
            log_path=self.log_path,
            outdir=self.outdir,
            figdir=self.figdir,
            prefix=self.prefix,
            boundary=self.boundary,
            solver=self.solver,
            compute_rotconst=self.compute_rotconst,
            label_cremer_pople=self.label_cremer_pople,
            check_only=self.check_only,
            manifest_path=self.manifest_path,
            grid_csv=outputs.get("grid_csv", self.grid_csv),
            summary=outputs.get("summary", self.summary),
            levels=outputs.get("levels", self.levels),
            outputs=outputs,
            status=self.status if status is None else status,
            source=self.source,
            schema=self.schema,
        )


@dataclass(frozen=True)
class DVRCommandResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class DVRRunResult:
    request: DVRRequest
    manifest_path: Path
    python_args: tuple[str, ...]
    bridge_args: tuple[str, ...] = ()
    commands: tuple[DVRCommandResult, ...] = ()
    status: str = "complete"


class DVRRunError(RuntimeError):
    """Raised when a DVR backend command exits unsuccessfully."""


def is_fortran_solver(solver: str) -> bool:
    return solver in FORTRAN_SOLVERS


def effective_python_solver(solver: str) -> str:
    return "sinc-dvr" if is_fortran_solver(solver) else solver


def resolve_dvr_executable(repo_root: Path) -> Path:
    return dvr_executable(repo_root)


def build_path_analysis_args(request: DVRRequest) -> list[str]:
    source_flag = "--xyz" if request.log_path.suffix.casefold() == ".xyz" else "--gaussian-log"
    args = [
        str(request.script),
        source_flag,
        str(request.log_path),
        "--log-selection",
        "last-per-link",
        "--boundary",
        request.boundary,
        "--solver",
        effective_python_solver(request.solver),
        "--outdir",
        str(request.outdir),
        "--figdir",
        str(request.figdir),
        "--prefix",
        request.normalized_prefix,
        "--workers",
        str(max(1, int(request.workers))),
        "--levels",
        str(max(1, int(request.levels))),
    ]
    if request.compute_rotconst:
        args.append("--compute-rotconst")
    if request.label_cremer_pople:
        args.append("--label-cremer-pople")
    if request.check_only:
        args.append("--check-only")
    return args


def build_fortran_bridge_args(request: DVRRequest, fortran_exe: Path) -> list[str]:
    args = [
        str(request.fortran_bridge),
        "--grid-csv",
        str(request.grid_csv),
        "--exe",
        str(fortran_exe),
        "--outdir",
        str(request.outdir),
        "--prefix",
        request.normalized_prefix,
        "--boundary",
        request.boundary,
    ]
    if request.solver == "fortran-gaussian":
        args.extend(["--mode", "gaussian"])
    return args


def build_fortran_shell_command(
    request: DVRRequest,
    python_args: list[str],
    bridge_args: list[str],
) -> str:
    return (
        f"{_quote(request.python_executable)} {' '.join(_quote(arg) for arg in python_args)}"
        f" && {_quote(request.python_executable)} {' '.join(_quote(arg) for arg in bridge_args)}"
    )


def _quote(value: str | Path) -> str:
    text = str(value)
    return "'" + text.replace("'", "'\"'\"'") + "'"


def write_dvr_manifest(request: DVRRequest, args: list[str], *, status: str = "prepared") -> Path:
    """Write a standard MATRIX run manifest for a DVR request."""
    outputs = {
        "grid_csv": request.grid_csv,
        "summary": request.outdir / f"{request.normalized_prefix}_summary.txt",
        "levels": request.outdir / f"{request.normalized_prefix}_levels.csv",
    }
    manifest = build_run_manifest(
        workflow="dvr",
        status=status,
        run_dir=request.outdir,
        inputs={"gaussian_log": request.log_path},
        outputs={name: path for name, path in outputs.items() if path.exists()},
        parameters={
            "prefix": request.normalized_prefix,
            "boundary": request.boundary,
            "solver": request.solver,
            "compute_rotconst": request.compute_rotconst,
            "label_cremer_pople": request.label_cremer_pople,
            "check_only": request.check_only,
        },
        backend={"python_executable": request.python_executable, "args": args},
    )
    return manifest.write(request.outdir / f"{request.normalized_prefix}_manifest.json")


def dvr_request_from_section(
    section: DVRSection,
    *,
    repo_root: Path,
    python_executable: str = sys.executable,
) -> DVRRequest:
    return DVRRequest(
        repo_root=repo_root,
        log_path=section.log_path,
        outdir=section.outdir,
        figdir=section.figdir,
        prefix=section.prefix,
        boundary=section.boundary,
        solver=section.solver,
        compute_rotconst=section.compute_rotconst,
        label_cremer_pople=section.label_cremer_pople,
        check_only=section.check_only,
        python_executable=python_executable,
    )


def run_dvr_request(
    request: DVRRequest,
    *,
    timeout: float | None = None,
    fortran_exe: Path | None = None,
) -> DVRRunResult:
    """Execute the normalized DVR request and write the completed run manifest."""
    request.outdir.mkdir(parents=True, exist_ok=True)
    request.figdir.mkdir(parents=True, exist_ok=True)
    python_args = build_path_analysis_args(request)
    manifest_args = python_args
    commands: list[DVRCommandResult] = []

    try:
        python_result = _run_backend_command(
            [request.python_executable, *python_args],
            timeout=timeout,
        )
    except DVRRunError:
        write_dvr_manifest(request, manifest_args, status="failed")
        raise
    commands.append(python_result)
    if python_result.returncode != 0:
        write_dvr_manifest(request, manifest_args, status="failed")
        raise DVRRunError(_failed_command_message("DVR path analysis", python_result))

    bridge_args: list[str] = []
    if is_fortran_solver(request.solver):
        executable = (
            fortran_exe if fortran_exe is not None else resolve_dvr_executable(request.repo_root)
        )
        bridge_args = build_fortran_bridge_args(request, executable)
        manifest_args = [build_fortran_shell_command(request, python_args, bridge_args)]
        try:
            bridge_result = _run_backend_command(
                [request.python_executable, *bridge_args],
                timeout=timeout,
            )
        except DVRRunError:
            write_dvr_manifest(request, manifest_args, status="failed")
            raise
        commands.append(bridge_result)
        if bridge_result.returncode != 0:
            write_dvr_manifest(request, manifest_args, status="failed")
            raise DVRRunError(_failed_command_message("DVR Fortran bridge", bridge_result))

    manifest_path = write_dvr_manifest(request, manifest_args, status="complete")
    return DVRRunResult(
        request=request,
        manifest_path=manifest_path,
        python_args=tuple(python_args),
        bridge_args=tuple(bridge_args),
        commands=tuple(commands),
    )


def dvr_section_from_request(
    request: DVRRequest,
    *,
    manifest_path: Path | None = None,
    status: str = "prepared",
) -> DVRSection:
    return DVRSection(
        log_path=request.log_path,
        outdir=request.outdir,
        figdir=request.figdir,
        prefix=request.normalized_prefix,
        boundary=request.boundary,
        solver=request.solver,
        compute_rotconst=request.compute_rotconst,
        label_cremer_pople=request.label_cremer_pople,
        check_only=request.check_only,
        manifest_path=manifest_path,
        grid_csv=request.grid_csv,
        summary=request.outdir / f"{request.normalized_prefix}_summary.txt",
        levels=request.outdir / f"{request.normalized_prefix}_levels.csv",
        status=status,
    )


def dvr_section_lines(section: DVRSection) -> list[str]:
    values = {
        "SOURCE": section.source,
        "STATUS": section.status,
        "LOG_PATH": section.log_path,
        "OUTDIR": section.outdir,
        "FIGDIR": section.figdir,
        "PREFIX": section.prefix,
        "BOUNDARY": section.boundary,
        "SOLVER": section.solver,
        "COMPUTE_ROTCONST": int(section.compute_rotconst),
        "LABEL_CREMER_POPLE": int(section.label_cremer_pople),
        "CHECK_ONLY": int(section.check_only),
        "MANIFEST": section.manifest_path,
        "GRID_CSV": section.grid_csv,
        "SUMMARY": section.summary,
        "LEVELS": section.levels,
    }
    values.update({f"OUTPUT_{name.upper()}": path for name, path in section.outputs.items()})
    return key_value_section_lines(
        ORACLE_XYZ_DVR_SCHEMA,
        values,
        key_order=(
            "SOURCE",
            "STATUS",
            "LOG_PATH",
            "OUTDIR",
            "FIGDIR",
            "PREFIX",
            "BOUNDARY",
            "SOLVER",
            "COMPUTE_ROTCONST",
            "LABEL_CREMER_POPLE",
            "CHECK_ONLY",
            "MANIFEST",
            "GRID_CSV",
            "SUMMARY",
            "LEVELS",
        ),
    )


def parse_dvr_section(lines: list[str] | tuple[str, ...]) -> DVRSection:
    values = parse_key_value_section(lines)
    schema = values.get("SCHEMA", ORACLE_XYZ_DVR_SCHEMA)
    if schema != ORACLE_XYZ_DVR_SCHEMA:
        raise ValueError(f"unsupported DVR schema: {schema}")
    outdir = _path_value(values, "OUTDIR", Path("."))
    prefix = values.get("PREFIX", "puckering_dvr") or "puckering_dvr"
    outputs = {
        _normalize_output_key(key[len("OUTPUT_") :]): Path(raw)
        for key, raw in values.items()
        if key.startswith("OUTPUT_") and raw.strip()
    }
    return DVRSection(
        log_path=_path_value(values, "LOG_PATH", Path("")),
        outdir=outdir,
        figdir=_path_value(values, "FIGDIR", outdir / "figures"),
        prefix=prefix,
        boundary=values.get("BOUNDARY", "periodic"),
        solver=values.get("SOLVER", "fourier"),
        compute_rotconst=_bool_value(values, "COMPUTE_ROTCONST", True),
        label_cremer_pople=_bool_value(values, "LABEL_CREMER_POPLE", True),
        check_only=_bool_value(values, "CHECK_ONLY", False),
        manifest_path=_optional_path(values.get("MANIFEST")),
        grid_csv=_optional_path(values.get("GRID_CSV")) or outdir / f"{prefix}_grid.csv",
        summary=_optional_path(values.get("SUMMARY")) or outdir / f"{prefix}_summary.txt",
        levels=_optional_path(values.get("LEVELS")) or outdir / f"{prefix}_levels.csv",
        outputs=outputs,
        status=values.get("STATUS", "prepared"),
        source=values.get("SOURCE", "gaussian-log"),
        schema=schema,
    )


def read_dvr_section(path: Path | str) -> DVRSection:
    content = section_content(read_sectioned_lines(Path(path)), "DVR")
    if not content:
        raise ValueError("missing #DVR section")
    return parse_dvr_section(content)


def write_dvr_section(path: Path | str, section: DVRSection) -> None:
    replace_section(Path(path), "DVR", dvr_section_lines(section))


def _run_backend_command(command: list[str], *, timeout: float | None) -> DVRCommandResult:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise DVRRunError(
            f"DVR backend command timed out after {exc.timeout} s: {command}"
        ) from exc
    return DVRCommandResult(
        args=tuple(str(arg) for arg in command),
        returncode=int(completed.returncode),
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


def _failed_command_message(label: str, result: DVRCommandResult) -> str:
    detail = result.stderr.strip() or result.stdout.strip()
    if detail:
        return f"{label} failed with exit code {result.returncode}: {detail}"
    return f"{label} failed with exit code {result.returncode}"


def _optional_path(raw: str | None) -> Path | None:
    if raw is None or not raw.strip():
        return None
    return Path(raw)


def _path_value(values: dict[str, str], key: str, default: Path) -> Path:
    return _optional_path(values.get(key)) or default


def _bool_value(values: dict[str, str], key: str, default: bool) -> bool:
    raw = values.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _normalize_output_key(value: str) -> str:
    return "_".join(part for part in str(value).lower().replace("-", "_").split("_") if part)
