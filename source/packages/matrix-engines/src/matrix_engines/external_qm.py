"""Shared launch/status helpers for external QM programs."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import time
from collections.abc import Mapping, Sequence

from matrix_core import (
    CalculationLaunchAuthorization,
    CalculationLaunchPlan,
    CalculationResources,
    authorized_parent_plan_from_environment,
    build_calculation_launch_plan,
    require_calculation_launch_or_parent_authorization,
)


class ExternalQMInputError(RuntimeError):
    """Raised when no runnable external-QM input can be found."""


@dataclass(frozen=True)
class ExternalQMProgramSpec:
    name: str
    default_executable: str
    env_var: str
    input_candidates: tuple[str, ...]
    output_suffix: str
    normal_markers: tuple[str, ...] = ()
    error_markers: tuple[str, ...] = ()
    pid_file: str = "matrix_qm.pid"
    capture_output: bool = True
    pass_input_path: bool = True
    pass_input_basename: bool = False
    extra_args_before_input: bool = False
    telemetry_filename: str = "matrix_qm_telemetry.jsonl"


@dataclass(frozen=True)
class ExternalQMJobStatus:
    program: str
    workdir: Path
    input_path: Path | None
    output_path: Path
    status: str
    normal_termination: bool
    error_termination: bool
    pid: int | None = None
    exit_code: int | None = None
    message: str = ""


@dataclass(frozen=True)
class ExternalQMRunResult:
    program: str
    workdir: Path
    input_path: Path
    output_path: Path
    executable: str
    pid: int | None
    exit_code: int | None
    success: bool | None
    message: str


def find_external_qm_input(
    workdir: Path | str,
    candidates: Sequence[str],
) -> Path:
    target = Path(workdir).resolve()
    for name in candidates:
        path = target / name
        if path.exists():
            return path
    raise ExternalQMInputError(
        f"no runnable input found in {target}; checked {', '.join(candidates)}"
    )


def external_qm_output_path(input_path: Path | str, suffix: str) -> Path:
    input_file = Path(input_path)
    if suffix.startswith("."):
        return input_file.with_suffix(suffix)
    return input_file.with_name(f"{input_file.stem}{suffix}")


def external_qm_has_marker(path: Path | str, markers: Sequence[str]) -> bool:
    target = Path(path)
    if not target.exists() or not markers:
        return False
    text = target.read_text(encoding="utf-8", errors="replace")
    return any(marker in text for marker in markers)


def external_qm_completion_message(
    spec: ExternalQMProgramSpec,
    output_path: Path,
    exit_code: int,
) -> tuple[bool, str]:
    error = external_qm_has_marker(output_path, spec.error_markers)
    if exit_code != 0 or error:
        return False, f"{spec.name} finished with errors (exit_code={exit_code}; see {output_path.name})"
    if external_qm_has_marker(output_path, spec.normal_markers):
        return True, f"{spec.name} completed successfully"
    if output_path.exists():
        return True, f"{spec.name} completed (check {output_path.name})"
    return False, f"{spec.name} produced no output (exit_code={exit_code})"


def external_qm_job_status(
    workdir: Path | str,
    spec: ExternalQMProgramSpec,
    *,
    input_path: Path | str | None = None,
    output_path: Path | str | None = None,
) -> ExternalQMJobStatus:
    target = Path(workdir).resolve()
    selected_input = _optional_input_path(target, spec.input_candidates, input_path)
    selected_output = _selected_output_path(target, spec, selected_input, output_path)
    pid = _read_pid(target / spec.pid_file)
    running = pid is not None and _pid_is_running(pid)
    normal = external_qm_has_marker(selected_output, spec.normal_markers)
    error = external_qm_has_marker(selected_output, spec.error_markers)
    if error:
        status = "failed"
        message = f"{spec.name} error termination detected"
    elif normal:
        status = "completed"
        message = f"{spec.name} normal termination detected"
    elif running:
        status = "running"
        message = f"{spec.name} process appears to be running (pid={pid})"
    elif selected_output.exists():
        status = "unknown"
        message = f"{spec.name} output exists without termination marker: {selected_output.name}"
    elif selected_input is not None:
        status = "ready"
        message = f"{spec.name} input found: {selected_input.name}"
    else:
        status = "missing"
        message = f"No {spec.name} input or output found"
    return ExternalQMJobStatus(
        program=spec.name,
        workdir=target,
        input_path=selected_input,
        output_path=selected_output,
        status=status,
        normal_termination=normal,
        error_termination=error,
        pid=pid,
        message=message,
    )


def run_external_qm_job(
    workdir: Path | str,
    spec: ExternalQMProgramSpec,
    *,
    executable: str | None = None,
    input_path: Path | str | None = None,
    output_path: Path | str | None = None,
    background: bool = False,
    timeout: float | None = None,
    env: Mapping[str, str] | None = None,
    extra_args: Sequence[str] = (),
    resources: CalculationResources | None = None,
    launch_authorization: CalculationLaunchAuthorization | None = None,
) -> ExternalQMRunResult:
    target = Path(workdir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    selected_input = _input_path(target, spec.input_candidates, input_path)
    selected_output = _selected_output_path(target, spec, selected_input, output_path)
    selected_output.parent.mkdir(parents=True, exist_ok=True)
    run_env = os.environ.copy()
    if env:
        run_env.update(dict(env))
    selected_executable, cmd = _external_qm_command(
        spec,
        selected_input,
        executable=executable,
        extra_args=extra_args,
        environment=run_env,
    )
    parent = authorized_parent_plan_from_environment(run_env)
    effective_resources = resources or (None if parent is None else parent.resources)
    if effective_resources is None:
        from matrix_core import CalculationLaunchError

        raise CalculationLaunchError(
            f"{spec.name} launch requires explicit process, thread, and memory limits"
        )
    launch_plan = build_calculation_launch_plan(
        backend=spec.name,
        host=parent.host if parent is not None else None,
        workdir=target,
        input_path=selected_input,
        command=cmd,
        resources=effective_resources,
    )
    require_calculation_launch_or_parent_authorization(
        launch_plan,
        launch_authorization,
        environment=run_env,
    )
    telemetry_path = target / spec.telemetry_filename
    _emit_qm_telemetry(
        telemetry_path,
        program=spec.name,
        phase="queued",
        detail=f"{spec.name} job queued",
        workdir=target,
        input_path=selected_input,
        output_path=selected_output,
    )
    executable_path = Path(selected_executable)
    if executable_path.parent != Path("."):
        run_env["PATH"] = f"{executable_path.parent}{os.pathsep}{run_env.get('PATH', '')}"
    if background:
        log_handle = selected_output.open("wb") if spec.capture_output else subprocess.DEVNULL
        try:
            process = subprocess.Popen(
                cmd,
                cwd=target,
                env=run_env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
        finally:
            if hasattr(log_handle, "close"):
                log_handle.close()
        (target / spec.pid_file).write_text(str(process.pid) + "\n", encoding="utf-8")
        _emit_qm_telemetry(
            telemetry_path,
            program=spec.name,
            phase="running",
            detail=f"{spec.name} job started",
            workdir=target,
            input_path=selected_input,
            output_path=selected_output,
            pid=process.pid,
        )
        return ExternalQMRunResult(
            program=spec.name,
            workdir=target,
            input_path=selected_input,
            output_path=selected_output,
            executable=selected_executable,
            pid=process.pid,
            exit_code=None,
            success=None,
            message=f"{spec.name} started in background (pid={process.pid})",
        )
    stdout = selected_output.open("wb") if spec.capture_output else None
    process = subprocess.Popen(
        cmd,
        cwd=target,
        env=run_env,
        start_new_session=(os.name == "posix"),
        stdout=stdout,
        stderr=subprocess.STDOUT if spec.capture_output else None,
    )
    try:
        process.communicate(timeout=timeout)
        return_code = int(process.returncode)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            import signal

            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        try:
            process.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.communicate()
        timeout_message = f"{spec.name} timed out after {timeout}s"
        _emit_qm_telemetry(
            telemetry_path,
            program=spec.name,
            phase="failed",
            detail=timeout_message,
            workdir=target,
            input_path=selected_input,
            output_path=selected_output,
        )
        return ExternalQMRunResult(
            program=spec.name,
            workdir=target,
            input_path=selected_input,
            output_path=selected_output,
            executable=selected_executable,
            pid=None,
            exit_code=None,
            success=False,
            message=timeout_message,
        )
    finally:
        if stdout is not None:
            stdout.close()
    success, message = external_qm_completion_message(spec, selected_output, return_code)
    _write_finished_pid_file(target / spec.pid_file, return_code)
    _emit_qm_telemetry(
        telemetry_path,
        program=spec.name,
        phase="completed" if success else "failed",
        detail=message,
        workdir=target,
        input_path=selected_input,
        output_path=selected_output,
        exit_code=return_code,
    )
    return ExternalQMRunResult(
        program=spec.name,
        workdir=target,
        input_path=selected_input,
        output_path=selected_output,
        executable=selected_executable,
        pid=None,
        exit_code=return_code,
        success=success,
        message=message,
    )


def prepare_external_qm_launch_plan(
    workdir: Path | str,
    spec: ExternalQMProgramSpec,
    *,
    executable: str | None = None,
    input_path: Path | str | None = None,
    extra_args: Sequence[str] = (),
    resources: CalculationResources,
    env: Mapping[str, str] | None = None,
    host: str | None = None,
) -> CalculationLaunchPlan:
    """Prepare the exact external-QM command for display without execution."""

    target = Path(workdir).resolve()
    selected_input = _input_path(target, spec.input_candidates, input_path)
    environment = os.environ.copy()
    if env:
        environment.update(dict(env))
    parent = authorized_parent_plan_from_environment(environment)
    _selected_executable, command = _external_qm_command(
        spec,
        selected_input,
        executable=executable,
        extra_args=extra_args,
        environment=environment,
    )
    return build_calculation_launch_plan(
        backend=spec.name,
        host=host or (None if parent is None else parent.host),
        workdir=target,
        input_path=selected_input,
        command=command,
        resources=resources,
    )


def _external_qm_command(
    spec: ExternalQMProgramSpec,
    input_path: Path,
    *,
    executable: str | None,
    extra_args: Sequence[str],
    environment: Mapping[str, str],
) -> tuple[str, tuple[str, ...]]:
    selected_executable = executable or environment.get(spec.env_var) or spec.default_executable
    command = [selected_executable]
    if spec.extra_args_before_input:
        command.extend(str(arg) for arg in extra_args)
    if spec.pass_input_path:
        command.append(input_path.name if spec.pass_input_basename else str(input_path))
    if not spec.extra_args_before_input:
        command.extend(str(arg) for arg in extra_args)
    return selected_executable, tuple(command)


def _optional_input_path(
    workdir: Path,
    candidates: Sequence[str],
    explicit: Path | str | None = None,
) -> Path | None:
    if explicit is not None:
        path = Path(explicit)
        if not path.is_absolute():
            path = workdir / path
        return path if path.exists() else None
    for name in candidates:
        path = workdir / name
        if path.exists():
            return path
    return None


def _input_path(
    workdir: Path,
    candidates: Sequence[str],
    explicit: Path | str | None = None,
) -> Path:
    selected = _optional_input_path(workdir, candidates, explicit)
    if selected is None:
        raise ExternalQMInputError(
            f"no runnable input found in {workdir}; checked {', '.join(candidates)}"
        )
    return selected


def _selected_output_path(
    workdir: Path,
    spec: ExternalQMProgramSpec,
    input_path: Path | None,
    output_path: Path | str | None,
) -> Path:
    if output_path is not None:
        path = Path(output_path)
        return path if path.is_absolute() else workdir / path
    if input_path is not None:
        return external_qm_output_path(input_path, spec.output_suffix)
    return workdir / f"{spec.name.lower()}{spec.output_suffix}"


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


def _emit_qm_telemetry(
    path: Path,
    *,
    program: str,
    phase: str,
    detail: str,
    workdir: Path,
    input_path: Path,
    output_path: Path,
    pid: int | None = None,
    exit_code: int | None = None,
) -> None:
    """Append execution lifecycle telemetry without interpreting science."""

    payload: dict[str, object] = {
        "schema": "matrix.keymaker.qm_telemetry.v1",
        "program": program,
        "phase": phase,
        "detail": detail,
        "emitted_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "workdir": str(workdir),
        "input_path": str(input_path),
        "output_path": str(output_path),
    }
    if pid is not None:
        payload["pid"] = pid
    if exit_code is not None:
        payload["exit_code"] = exit_code
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    except OSError:
        # Telemetry must never alter the backend calculation outcome.
        return
