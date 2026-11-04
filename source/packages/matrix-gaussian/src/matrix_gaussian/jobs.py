from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess
import time

from matrix_core import (
    CalculationLaunchAuthorization,
    CalculationLaunchError,
    CalculationLaunchPlan,
    CalculationResources,
    authorized_parent_plan_from_environment,
    build_calculation_launch_plan,
    require_calculation_launch_or_parent_authorization,
    read_sectioned_lines,
    section_content,
)

from .route_policy import (
    GaussianRouteOverride,
    GaussianRoutePolicyError,
    gaussian_route_digest,
    gaussian_route_violations,
    validate_gaussian_route_policy,
)


GAUSSIAN_EXECUTABLE = "gdv"
FORMCHK_EXECUTABLE = "formchk"
NORMAL_TERMINATION_MARKER = "Normal termination of Gaussian"
ERROR_TERMINATION_MARKER = "Error termination"
LOG_CANDIDATES = ("gauin.log", "gauout.log")
PID_FILE = "gaussian.pid"


class GaussianInputError(RuntimeError):
    """Raised when no runnable Gaussian input can be found."""


@dataclass(frozen=True)
class GaussianJobStatus:
    workdir: Path
    input_path: Path | None
    log_path: Path
    status: str
    normal_termination: bool
    error_termination: bool
    pid: int | None = None
    exit_code: int | None = None
    message: str = ""


@dataclass(frozen=True)
class GaussianRunResult:
    workdir: Path
    input_path: Path
    log_path: Path
    executable: str
    pid: int | None
    exit_code: int | None
    success: bool | None
    message: str


def ensure_gjf_input(
    workdir: Path,
    *,
    input_name: str = "gauin",
    gjf_name: str = "gauin.gjf",
) -> Path:
    """Return a Gaussian input path, copying `gauin` to `gauin.gjf` if needed."""
    workdir = Path(workdir).resolve()
    gauin_gjf = workdir / gjf_name
    gauin_raw = workdir / input_name
    if not gauin_gjf.exists() and gauin_raw.exists():
        gauin_gjf.write_text(
            gauin_raw.read_text(encoding="utf-8", errors="replace"),
            encoding="utf-8",
        )
    if gauin_gjf.exists():
        return gauin_gjf
    if gauin_raw.exists():
        return gauin_raw
    raise GaussianInputError(f"{gjf_name} or {input_name} not found")


def select_latest_log(workdir: Path, *, candidates: tuple[str, ...] = LOG_CANDIDATES) -> Path:
    """Select the most likely active Gaussian log in a work directory."""
    workdir = Path(workdir).resolve()
    existing: list[tuple[float, int, Path]] = []
    for name in candidates:
        path = workdir / name
        if path.exists():
            try:
                stat = path.stat()
            except OSError:
                continue
            existing.append((stat.st_mtime, stat.st_size, path))
    if not existing:
        return workdir / candidates[0]
    existing.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return existing[0][2]


def gaussian_completed_normally(log_path: Path) -> bool:
    if not Path(log_path).exists():
        return False
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    return NORMAL_TERMINATION_MARKER in text


def gaussian_has_error_termination(log_path: Path) -> bool:
    if not Path(log_path).exists():
        return False
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    return ERROR_TERMINATION_MARKER in text


def gaussian_completion_message(workdir: Path, exit_code: int) -> tuple[bool, str]:
    """Return `(success, message)` for a finished Gaussian process."""
    log_path = select_latest_log(workdir)
    if exit_code != 0 or gaussian_has_error_termination(log_path):
        return False, f"Gaussian finished with errors (exit_code={exit_code}; see {log_path.name})"
    if gaussian_completed_normally(log_path):
        return True, "Gaussian completed successfully"
    if log_path.exists():
        return True, f"Gaussian completed (check {log_path.name})"
    return False, f"Gaussian produced no log (exit_code={exit_code})"


def gaussian_job_status(workdir: Path) -> GaussianJobStatus:
    workdir = Path(workdir)
    input_path = _optional_input_path(workdir)
    log_path = select_latest_log(workdir)
    pid = _read_pid(workdir / PID_FILE)
    running = pid is not None and _pid_is_running(pid)
    normal = gaussian_completed_normally(log_path)
    error = gaussian_has_error_termination(log_path)
    if error:
        status = "failed"
        message = "Gaussian error termination detected"
    elif normal:
        status = "completed"
        message = "Gaussian normal termination detected"
    elif running:
        status = "running"
        message = f"Gaussian process appears to be running (pid={pid})"
    elif log_path.exists():
        status = "unknown"
        message = f"Gaussian log exists without termination marker: {log_path.name}"
    elif input_path is not None:
        status = "ready"
        message = f"Gaussian input found: {input_path.name}"
    else:
        status = "missing"
        message = "No Gaussian input or log found"
    return GaussianJobStatus(
        workdir=workdir,
        input_path=input_path,
        log_path=log_path,
        status=status,
        normal_termination=normal,
        error_termination=error,
        pid=pid,
        message=message,
    )


def run_gaussian_job(
    workdir: Path,
    *,
    executable: str | None = None,
    input_path: Path | None = None,
    background: bool = False,
    timeout: float | None = None,
    env: dict[str, str] | None = None,
    route_overrides: tuple[GaussianRouteOverride, ...] = (),
    resources: CalculationResources | None = None,
    launch_authorization: CalculationLaunchAuthorization | None = None,
) -> GaussianRunResult:
    """Run Gaussian from an ORACLE work directory."""
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    executable = executable or os.environ.get("ORACLE_GAUSSIAN_EXE") or GAUSSIAN_EXECUTABLE
    gauin = Path(input_path) if input_path is not None else ensure_gjf_input(workdir)
    if not gauin.is_absolute():
        gauin = workdir / gauin
    if not gauin.exists():
        raise GaussianInputError(f"Gaussian input not found: {gauin}")
    validate_gaussian_input_route_policy(gauin, route_overrides=route_overrides)
    cmd = [executable, str(gauin)]
    log_path = select_latest_log(workdir)
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    parent = authorized_parent_plan_from_environment(run_env)
    effective_resources = resources or (None if parent is None else parent.resources)
    if effective_resources is None:
        raise CalculationLaunchError(
            "Gaussian launch requires explicit process, thread, and memory limits"
        )
    launch_plan = build_calculation_launch_plan(
        backend="Gaussian",
        host=parent.host if parent is not None else None,
        workdir=workdir,
        input_path=gauin,
        command=cmd,
        resources=effective_resources,
    )
    require_calculation_launch_or_parent_authorization(
        launch_plan,
        launch_authorization,
        environment=run_env,
    )
    if background:
        process = subprocess.Popen(cmd, cwd=workdir, env=run_env)
        (workdir / PID_FILE).write_text(str(process.pid) + "\n", encoding="utf-8")
        return GaussianRunResult(
            workdir=workdir,
            input_path=gauin,
            log_path=log_path,
            executable=executable,
            pid=process.pid,
            exit_code=None,
            success=None,
            message=f"Gaussian started in background (pid={process.pid})",
        )
    completed = subprocess.run(cmd, cwd=workdir, env=run_env, timeout=timeout, check=False)
    success, message = gaussian_completion_message(workdir, int(completed.returncode))
    _write_finished_pid_file(workdir / PID_FILE, int(completed.returncode))
    return GaussianRunResult(
        workdir=workdir,
        input_path=gauin,
        log_path=select_latest_log(workdir),
        executable=executable,
        pid=None,
        exit_code=int(completed.returncode),
        success=success,
        message=message,
    )


def prepare_gaussian_launch_plan(
    workdir: Path,
    *,
    resources: CalculationResources,
    executable: str | None = None,
    input_path: Path | None = None,
    env: dict[str, str] | None = None,
    route_overrides: tuple[GaussianRouteOverride, ...] = (),
    host: str | None = None,
) -> CalculationLaunchPlan:
    """Prepare and validate the exact Gaussian launch without executing it."""

    target = Path(workdir).resolve()
    selected_executable = (
        executable
        or (env or {}).get("ORACLE_GAUSSIAN_EXE")
        or os.environ.get("ORACLE_GAUSSIAN_EXE")
        or GAUSSIAN_EXECUTABLE
    )
    gauin = Path(input_path) if input_path is not None else ensure_gjf_input(target)
    if not gauin.is_absolute():
        gauin = target / gauin
    if not gauin.exists():
        raise GaussianInputError(f"Gaussian input not found: {gauin}")
    validate_gaussian_input_route_policy(gauin, route_overrides=route_overrides)
    return build_calculation_launch_plan(
        backend="Gaussian",
        host=host,
        workdir=target,
        input_path=gauin,
        command=(selected_executable, str(gauin)),
        resources=resources,
    )


def gaussian_input_routes(input_path: Path) -> tuple[str, ...]:
    """Extract every route section from a Gaussian input, including Link1 jobs."""

    path = Path(input_path)
    if not path.exists():
        raise GaussianInputError(f"Gaussian input not found: {path}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    routes: list[str] = []
    index = 0
    while index < len(lines):
        if not lines[index].lstrip().startswith("#"):
            index += 1
            continue
        route_lines = [lines[index].strip()]
        index += 1
        while index < len(lines) and lines[index].strip():
            if lines[index].strip().lower() == "--link1--":
                break
            route_lines.append(lines[index].strip())
            index += 1
        routes.append(" ".join(" ".join(route_lines).split()))
    if not routes:
        raise GaussianInputError(f"Gaussian input has no route section: {path}")
    return tuple(routes)


def validate_gaussian_input_route_policy(
    input_path: Path,
    *,
    route_overrides: tuple[GaussianRouteOverride, ...] = (),
) -> tuple[str, ...]:
    """Enforce the frozen route policy immediately before Gaussian execution."""

    routes = gaussian_input_routes(input_path)
    overrides_by_digest = {override.route_sha256: override for override in route_overrides}
    used_digests: set[str] = set()
    for route in routes:
        violations = gaussian_route_violations(route)
        override = None
        if violations:
            digest = gaussian_route_digest(route)
            override = overrides_by_digest.get(digest)
            if override is not None:
                used_digests.add(digest)
        validate_gaussian_route_policy(route, override=override)
    unused = set(overrides_by_digest) - used_digests
    if unused:
        raise GaussianRoutePolicyError(
            "Gaussian launcher received route overrides that do not match any "
            "forbidden route in the exact input"
        )
    return routes


def validate_gaussian_readallgic_input(
    input_path: Path,
    *,
    reference_xyzin: Path | None = None,
    g16_compatibility: bool = False,
) -> tuple[str, ...]:
    """Fail closed on a final ReadAllGIC input immediately before execution.

    This is the execution-side counterpart of the canonical MATRIX writer.  It
    rejects hand-edited or partially serialized GIC blocks whose ``Value=``
    fields do not reproduce the Cartesian input in Gaussian's native units.
    """

    target = Path(input_path)
    routes = validate_gaussian_input_route_policy(target)
    if not any("readallgic" in route.casefold() for route in routes):
        return routes

    from .parsers import _read_gaussian_input_block, read_gaussian_cartesian_input
    from .writers import (
        DEFAULT_GAUSSIAN_GIC_MAX_ADDENDS,
        DEFAULT_GAUSSIAN_GIC_MAX_LABEL_LENGTH,
        GaussianWriteError,
        _gaussian_base_label,
        _gaussian_compact_transport_labels,
        _gaussian_definition,
        _gaussian_factor_long_gic_lines,
        _gaussian_gic_lines,
        _gaussian_gic_lines_with_values,
        _gaussian_native_inactive_improper_helpers,
        validate_gaussian_geometry_identity,
    )

    block = _read_gaussian_input_block(target)
    gic_lines: list[str] = []
    for raw_line in block.tail_lines:
        line = raw_line.strip()
        if not line:
            if gic_lines:
                break
            continue
        if _gaussian_definition(line) is None:
            raise GaussianInputError(
                f"ReadAllGIC input contains a non-definition row in its GIC block: {line}"
            )
        gic_lines.append(line)
    if not gic_lines:
        raise GaussianInputError("ReadAllGIC route requires a non-empty GIC block")
    for line in gic_lines:
        parsed = _gaussian_definition(line)
        assert parsed is not None
        label, expression = parsed
        fragment_match = re.fullmatch(
            r"\s*Fragment\s*\(([^()]*)\)\s*",
            expression,
            flags=re.IGNORECASE,
        )
        if fragment_match is not None and not fragment_match.group(1).strip():
            raise GaussianInputError(
                "ReadAllGIC input contains an empty Gaussian Fragment declaration: "
                f"{label}"
            )

    geometry = read_gaussian_cartesian_input(target)
    if reference_xyzin is not None:
        from matrix_chem import read_enriched_xyz

        reference = read_enriched_xyz(Path(reference_xyzin))
        try:
            validate_gaussian_geometry_identity(
                target,
                reference.atoms,
                reference.coordinates_angstrom,
            )
        except GaussianWriteError as exc:
            raise GaussianInputError(str(exc)) from exc
        if section_content(read_sectioned_lines(Path(reference_xyzin)), "GIC"):
            try:
                from matrix_smith.definition import (
                    read_gic_definition_from_xyzin,
                    validate_frozen_sonic_basis,
                )

                definition = read_gic_definition_from_xyzin(Path(reference_xyzin))
                validate_frozen_sonic_basis(definition, rank_tolerance=1.0e-7)
            except (ValueError, RuntimeError) as exc:
                raise GaussianInputError(
                    f"ReadAllGIC reference fails the frozen SMITH basis gate: {exc}"
                ) from exc
    try:
        canonical = _gaussian_gic_lines_with_values(
            gic_lines,
            geometry.coordinates_angstrom,
        )
    except GaussianWriteError as exc:
        raise GaussianInputError(str(exc)) from exc
    if canonical != gic_lines:
        raise GaussianInputError(
            "ReadAllGIC input is not a finalized MATRIX serialization: "
            "required Value= fields are missing or non-canonical"
        )
    if reference_xyzin is not None and not g16_compatibility:
        expected = _gaussian_gic_lines(Path(reference_xyzin))
        expected = _gaussian_native_inactive_improper_helpers(expected)
        expected = _gaussian_compact_transport_labels(expected)
        expected = _gaussian_gic_lines_with_values(
            expected,
            geometry.coordinates_angstrom,
        )
        expected = _gaussian_factor_long_gic_lines(
            expected,
            max_addends=DEFAULT_GAUSSIAN_GIC_MAX_ADDENDS,
        )
        if expected != gic_lines:
            raise GaussianInputError(
                "ReadAllGIC block is not the canonical native serialization of "
                "the frozen SMITH chart"
            )

    labels = [
        _gaussian_base_label(_gaussian_definition(line)[0])
        for line in gic_lines
    ]
    if len(labels) != len(set(labels)):
        raise GaussianInputError("ReadAllGIC input contains duplicate GIC labels")
    overlong = tuple(
        label for label in labels if len(label) > DEFAULT_GAUSSIAN_GIC_MAX_LABEL_LENGTH
    )
    if overlong:
        raise GaussianInputError(
            "ReadAllGIC input contains transport labels beyond the canonical "
            f"{DEFAULT_GAUSSIAN_GIC_MAX_LABEL_LENGTH}-character capacity: "
            + ",".join(overlong)
        )
    parser_unsafe = tuple(
        label
        for label in labels
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", label) is None
    )
    if parser_unsafe:
        raise GaussianInputError(
            "ReadAllGIC input contains labels rejected by the Gaussian expression "
            "parser: " + ",".join(parser_unsafe)
        )
    return routes


def formchk_checkpoint(
    chk_path: Path,
    fchk_path: Path | None = None,
    *,
    executable: str = FORMCHK_EXECUTABLE,
    timeout: float | None = None,
) -> Path:
    chk = Path(chk_path)
    if fchk_path is None:
        fchk = chk.with_suffix(".fchk")
    else:
        fchk = Path(fchk_path)
    fchk.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([executable, str(chk), str(fchk)], cwd=chk.parent, check=True, timeout=timeout)
    return fchk


def _optional_input_path(workdir: Path) -> Path | None:
    for name in ("gauin.gjf", "gauin"):
        path = workdir / name
        if path.exists():
            return path
    return None


def _read_pid(path: Path) -> int | None:
    try:
        raw = path.read_text(encoding="utf-8").splitlines()[0].strip()
        return int(raw)
    except Exception:
        return None


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _write_finished_pid_file(path: Path, exit_code: int) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    path.write_text(f"finished exit_code={exit_code} at {stamp}\n", encoding="utf-8")
