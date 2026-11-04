"""SSH/SCP orchestration for remote MATRIX QM jobs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
from collections.abc import Callable, Sequence

from matrix_core import (
    CalculationLaunchAuthorization,
    CalculationLaunchError,
    CalculationLaunchPlan,
    CalculationResources,
    authorized_parent_plan_from_environment,
    build_calculation_launch_plan,
    require_calculation_launch_or_parent_authorization,
)


REMOTE_ENGINES = (
    "gdv32",
    "g16",
    "molpro",
    "orca",
    "mrcc",
    "cfour",
    "xtb",
    "pyscf",
    "et",
)
DEFAULT_REMOTE_HOST = "oracle"
DEFAULT_REMOTE_ROOT = "~/matrix"
PROMOTE_MODES = (
    "none",
    "auto",
    "molpro",
    "gaussian-log-hessian",
    "gaussian-rovib",
    "gaussian-electronic",
    "gaussian-fchk",
    "orca",
    "xtb-hessian",
    "pyscf-hessian",
)


class RemoteQMError(RuntimeError):
    """Raised when a remote QM operation fails."""


SubprocessRunner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class RemoteQMMetadata:
    job: str
    engine: str
    pid: str = ""
    workdir: str = ""
    log: str = ""
    native_output: str = ""
    stdout: str = ""
    input: str = ""

    @classmethod
    def from_text(cls, text: str) -> "RemoteQMMetadata":
        values: dict[str, str] = {}
        for line in text.splitlines():
            key, sep, value = line.partition("=")
            if sep:
                values[key.strip()] = value.strip()
        job = values.get("job")
        engine = values.get("engine")
        if not job or not engine:
            raise RemoteQMError("remote metadata is missing job or engine")
        return cls(
            job=job,
            engine=engine,
            pid=values.get("pid", ""),
            workdir=values.get("workdir", ""),
            log=values.get("log", ""),
            native_output=values.get("native_output") or values.get("log", ""),
            stdout=values.get("stdout", ""),
            input=values.get("input", ""),
        )


@dataclass(frozen=True)
class RemoteQMSubmitResult:
    host: str
    engine: str
    input_path: Path
    remote_input: str
    job: str
    workdir: str
    log: str
    native_output: str
    stdout: str
    raw_output: str


@dataclass(frozen=True)
class RemoteQMJobStatus:
    host: str
    job: str
    state: str
    pid: str = ""
    log: str = ""

    @property
    def canonical_state(self) -> str:
        """Normalize ORACLE/MAC-STUDIO matrix-status vocabulary."""

        state = self.state.strip().upper()
        return {
            "RUNNING": "running",
            "STARTING": "running",
            "FINISHED": "completed",
            "COMPLETED": "completed",
            "FAILED": "failed",
            "ERROR": "failed",
            "CANCELLED": "cancelled",
            "INTERRUPTED": "interrupted",
        }.get(state, "unknown")

    @property
    def terminal(self) -> bool:
        return self.canonical_state in {"completed", "failed", "cancelled", "interrupted"}


@dataclass(frozen=True)
class RemoteQMFetchResult:
    host: str
    job: str
    engine: str
    destination: Path
    output_path: Path
    metadata_path: Path
    manifest_path: Path
    metadata: RemoteQMMetadata
    artifacts: dict[str, Path] = field(default_factory=dict)
    promotion: "RemoteQMPromotionResult | None" = None


@dataclass(frozen=True)
class RemoteQMPromotionResult:
    mode: str
    status: str
    message: str


def remote_qm_submit(
    input_path: Path | str,
    *,
    engine: str,
    host: str = DEFAULT_REMOTE_HOST,
    remote_root: str = DEFAULT_REMOTE_ROOT,
    extra_args: Sequence[str] = (),
    ssh_executable: str = "ssh",
    scp_executable: str = "scp",
    runner: SubprocessRunner = subprocess.run,
    resources: CalculationResources | None = None,
    launch_authorization: CalculationLaunchAuthorization | None = None,
) -> RemoteQMSubmitResult:
    if engine not in REMOTE_ENGINES:
        raise RemoteQMError(f"unsupported remote engine: {engine}")
    source = Path(input_path).expanduser().resolve()
    if not source.is_file():
        raise RemoteQMError(f"input file not found: {source}")
    remote_inputs = f"{remote_root.rstrip('/')}/inputs"
    remote_input = f"{remote_inputs}/{source.name}"
    parent = authorized_parent_plan_from_environment()
    effective_resources = resources or (None if parent is None else parent.resources)
    if effective_resources is None:
        raise CalculationLaunchError(
            "remote QM launch requires explicit process, thread, and memory limits"
        )
    launch_plan = prepare_remote_qm_launch_plan(
        source,
        engine=engine,
        host=host,
        remote_root=remote_root,
        extra_args=extra_args,
        resources=effective_resources,
    )
    require_calculation_launch_or_parent_authorization(
        launch_plan,
        launch_authorization,
    )
    if engine in {"gdv32", "g16"}:
        from matrix_gaussian import validate_gaussian_input_route_policy

        validate_gaussian_input_route_policy(source)
    _run_checked(
        _cmd(ssh_executable, host, _remote_bash(f"mkdir -p {_remote_arg(remote_inputs)}")),
        runner=runner,
        context="create remote MATRIX input directory",
    )
    _run_checked(
        _cmd(scp_executable, str(source), f"{host}:{remote_input}"),
        runner=runner,
        context="copy QM input to remote host",
    )
    resource_environment = _remote_resource_environment(effective_resources)
    submit_cmd = " ".join(
        [
            "env",
            *(_q(item) for item in resource_environment),
            _remote_arg(f"{remote_root.rstrip('/')}/bin/matrix-submit"),
            _q(engine),
            _remote_arg(remote_input),
            *(_q(arg) for arg in extra_args),
        ]
    )
    completed = _run_checked(
        _cmd(ssh_executable, host, _remote_bash(submit_cmd)),
        runner=runner,
        context="submit remote MATRIX QM job",
    )
    parsed = _parse_submit_output(completed.stdout)
    return RemoteQMSubmitResult(
        host=host,
        engine=engine,
        input_path=source,
        remote_input=remote_input,
        job=parsed.get("job", ""),
        workdir=parsed.get("workdir", ""),
        log=parsed.get("log", ""),
        native_output=parsed.get("native_output", ""),
        stdout=parsed.get("stdout", ""),
        raw_output=completed.stdout,
    )


def prepare_remote_qm_launch_plan(
    input_path: Path | str,
    *,
    engine: str,
    host: str,
    remote_root: str = DEFAULT_REMOTE_ROOT,
    extra_args: Sequence[str] = (),
    resources: CalculationResources,
) -> CalculationLaunchPlan:
    """Prepare the exact remote submission envelope without opening SSH."""

    if engine not in REMOTE_ENGINES:
        raise RemoteQMError(f"unsupported remote engine: {engine}")
    source = Path(input_path).expanduser().resolve()
    if not source.is_file():
        raise RemoteQMError(f"input file not found: {source}")
    remote_input = f"{remote_root.rstrip('/')}/inputs/{source.name}"
    command = (
        "env",
        *_remote_resource_environment(resources),
        f"{remote_root.rstrip('/')}/bin/matrix-submit",
        engine,
        remote_input,
        *(str(arg) for arg in extra_args),
    )
    return build_calculation_launch_plan(
        backend=f"remote/{engine}",
        host=host,
        workdir=source.parent,
        input_path=source,
        command=command,
        resources=resources,
    )


def _remote_resource_environment(resources: CalculationResources) -> tuple[str, str]:
    """Translate the authorized per-job envelope into remote launcher settings."""

    memory_gb = float(resources.memory_per_job_gb)
    memory_value = (
        str(int(memory_gb)) if memory_gb.is_integer() else format(memory_gb, ".15g")
    )
    return (
        f"MATRIX_QM_NPROC={resources.threads_per_job}",
        f"MATRIX_QM_MEM={memory_value}GB",
    )


def remote_qm_status(
    *,
    host: str = DEFAULT_REMOTE_HOST,
    remote_root: str = DEFAULT_REMOTE_ROOT,
    ssh_executable: str = "ssh",
    runner: SubprocessRunner = subprocess.run,
) -> str:
    completed = _run_checked(
        _cmd(
            ssh_executable,
            host,
            _remote_bash(_remote_arg(f"{remote_root.rstrip('/')}/bin/matrix-status")),
        ),
        runner=runner,
        context="inspect remote MATRIX QM jobs",
    )
    return completed.stdout


def remote_qm_job_status(
    job: str,
    *,
    host: str = DEFAULT_REMOTE_HOST,
    remote_root: str = DEFAULT_REMOTE_ROOT,
    ssh_executable: str = "ssh",
    runner: SubprocessRunner = subprocess.run,
) -> RemoteQMJobStatus:
    """Return the status of one background job reported by matrix-status."""
    output = remote_qm_status(
        host=host,
        remote_root=remote_root,
        ssh_executable=ssh_executable,
        runner=runner,
    )
    marker = f"job={job}"
    for line in output.splitlines():
        fields = line.split()
        if marker not in fields:
            continue
        values = {
            key: value
            for field in fields[1:]
            for key, separator, value in (field.partition("="),)
            if separator
        }
        return RemoteQMJobStatus(
            host=host,
            job=job,
            state=fields[0].upper(),
            pid=values.get("pid", ""),
            log=values.get("log", ""),
        )
    raise RemoteQMError(f"remote job not found in status output: {job}")


def remote_qm_output_completed_normally(path: Path | str, engine: str) -> bool:
    """Check native completion markers for monitored QM jobs."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    normalized = str(engine).strip().lower()
    if normalized in {"gdv32", "g16"}:
        return "Normal termination of Gaussian" in text
    if normalized == "orca":
        return "ORCA TERMINATED NORMALLY" in text
    if normalized == "xtb":
        return "normal termination of xtb" in text.lower()
    if normalized == "pyscf":
        return "MATRIX PYSCF TERMINATED NORMALLY" in text
    if normalized == "et":
        return "eT terminated successfully!" in text
    return True


def remote_qm_fetch(
    job: str,
    *,
    host: str = DEFAULT_REMOTE_HOST,
    destination: Path | str = Path("remote_qm_runs"),
    remote_root: str = DEFAULT_REMOTE_ROOT,
    promote: str = "none",
    xyzin: Path | str | None = None,
    ssh_executable: str = "ssh",
    scp_executable: str = "scp",
    runner: SubprocessRunner = subprocess.run,
) -> RemoteQMFetchResult:
    if promote not in PROMOTE_MODES:
        raise RemoteQMError(f"unsupported promotion mode: {promote}")
    metadata_text = remote_qm_metadata_text(
        job,
        host=host,
        remote_root=remote_root,
        ssh_executable=ssh_executable,
        runner=runner,
    )
    metadata = RemoteQMMetadata.from_text(metadata_text)
    if not metadata.native_output:
        raise RemoteQMError(f"remote job has no native_output/log path: {job}")
    target_dir = Path(destination).expanduser().resolve() / metadata.job
    target_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = target_dir / "metadata.txt"
    metadata_path.write_text(metadata_text, encoding="utf-8")
    output_path = target_dir / Path(metadata.native_output).name
    _run_checked(
        _cmd(scp_executable, f"{host}:{metadata.native_output}", str(output_path)),
        runner=runner,
        context="copy remote native QM output",
    )
    resolved_promote = _resolved_promote_mode(promote, metadata)
    artifacts: dict[str, Path] = {}
    if resolved_promote == "xtb-hessian":
        artifacts = _fetch_remote_xtb_hessian_bundle(
            metadata,
            host=host,
            target_dir=target_dir,
            scp_executable=scp_executable,
            runner=runner,
        )
    promotion = promote_remote_qm_output(
        output_path,
        metadata,
        promote=resolved_promote,
        xyzin=xyzin,
        artifacts=artifacts,
    )
    manifest_path = _write_fetch_manifest(
        target_dir,
        host=host,
        metadata=metadata,
        output_path=output_path,
        metadata_path=metadata_path,
        artifacts=artifacts,
        promotion=promotion,
    )
    return RemoteQMFetchResult(
        host=host,
        job=metadata.job,
        engine=metadata.engine,
        destination=target_dir,
        output_path=output_path,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        metadata=metadata,
        artifacts=artifacts,
        promotion=promotion,
    )


def remote_qm_metadata_text(
    job: str,
    *,
    host: str = DEFAULT_REMOTE_HOST,
    remote_root: str = DEFAULT_REMOTE_ROOT,
    ssh_executable: str = "ssh",
    runner: SubprocessRunner = subprocess.run,
) -> str:
    remote_metadata = f"{remote_root.rstrip('/')}/jobs/{job}/metadata.txt"
    completed = _run_checked(
        _cmd(ssh_executable, host, _remote_bash(f"cat {_remote_arg(remote_metadata)}")),
        runner=runner,
        context="read remote MATRIX QM metadata",
    )
    return completed.stdout


def promote_remote_qm_output(
    output_path: Path | str,
    metadata: RemoteQMMetadata,
    *,
    promote: str,
    xyzin: Path | str | None = None,
    artifacts: dict[str, Path] | None = None,
) -> RemoteQMPromotionResult | None:
    mode = _resolved_promote_mode(promote, metadata)
    if mode == "none":
        return None
    if xyzin is None:
        return RemoteQMPromotionResult(
            mode=mode,
            status="skipped",
            message="promotion requested but no xyzin path was provided",
        )
    output = Path(output_path)
    xyzin_path = Path(xyzin)
    if mode == "molpro":
        from matrix_molpro import promote_molpro_output_to_xyzin

        result = promote_molpro_output_to_xyzin(output, xyzin_path)
        return RemoteQMPromotionResult(
            mode=mode,
            status="promoted",
            message=f"promoted Molpro output to {result.path}",
        )
    if mode == "gaussian-log-hessian":
        from matrix_gaussian import promote_gaussian_log_hessian_to_xyzin

        result = promote_gaussian_log_hessian_to_xyzin(output, xyzin_path)
        return RemoteQMPromotionResult(
            mode=mode,
            status="promoted",
            message=f"promoted Gaussian log Hessian to {result.xyzin}",
        )
    if mode == "gaussian-rovib":
        from matrix_gaussian import promote_gaussian_rovib_to_xyzin

        result = promote_gaussian_rovib_to_xyzin(output, xyzin_path)
        return RemoteQMPromotionResult(
            mode=mode,
            status="promoted",
            message=f"promoted Gaussian rovib data to {result.xyzin}",
        )
    if mode == "gaussian-electronic":
        from matrix_gaussian import promote_gaussian_electronic_log_to_xyzin

        result = promote_gaussian_electronic_log_to_xyzin(output, xyzin_path)
        return RemoteQMPromotionResult(
            mode=mode,
            status="promoted",
            message=f"promoted Gaussian electronic data to {result.xyzin}",
        )
    if mode == "gaussian-fchk":
        from matrix_gaussian import promote_gaussian_fchk_to_xyzin

        result = promote_gaussian_fchk_to_xyzin(output, xyzin_path)
        return RemoteQMPromotionResult(
            mode=mode,
            status="promoted",
            message=f"promoted Gaussian FCHK data to {result.xyzin}",
        )
    if mode == "orca":
        from matrix_orca import promote_orca_output_to_xyzin

        result = promote_orca_output_to_xyzin(output, xyzin_path)
        sections = ["geometry"]
        if result.wrote_cartesian_hessian:
            sections.append("Cartesian Hessian")
        return RemoteQMPromotionResult(
            mode=mode,
            status="promoted",
            message=f"promoted ORCA {' and '.join(sections)} to {result.xyzin}",
        )
    if mode == "xtb-hessian":
        bundle = artifacts or {}
        required = ("xtb_hessian", "xtb_geometry", "xtb_spectrum")
        missing = [name for name in required if name not in bundle]
        if missing:
            raise RemoteQMError(
                "xTB Hessian promotion is missing fetched artifacts: " + ", ".join(missing)
            )
        from matrix_qm import (
            cartesian_hessian_section_from_hessian_input,
            write_cartesian_hessian_section_atomic,
        )
        from matrix_xtb import hessian_input_from_xtb_files

        data = hessian_input_from_xtb_files(
            bundle["xtb_hessian"],
            geometry=bundle["xtb_geometry"],
            spectrum=bundle["xtb_spectrum"],
            output=output,
        )
        write_cartesian_hessian_section_atomic(
            xyzin_path,
            cartesian_hessian_section_from_hessian_input(
                data,
                source=f"remote xTB Hessian job={metadata.job} host bundle",
            ),
        )
        return RemoteQMPromotionResult(
            mode=mode,
            status="promoted",
            message=f"promoted remote xTB Cartesian Hessian to {xyzin_path}",
        )
    if mode == "pyscf-hessian":
        if metadata.engine != "pyscf":
            raise RemoteQMError("pyscf-hessian promotion requires a PySCF remote job")
        from matrix_pyscf import hessian_input_from_pyscf_output
        from matrix_qm import (
            cartesian_hessian_section_from_hessian_input,
            write_cartesian_hessian_section_atomic,
        )

        data = hessian_input_from_pyscf_output(output)
        write_cartesian_hessian_section_atomic(
            xyzin_path,
            cartesian_hessian_section_from_hessian_input(
                data,
                source=f"remote PySCF Hessian job={metadata.job}",
            ),
        )
        return RemoteQMPromotionResult(
            mode=mode,
            status="promoted",
            message=f"promoted remote PySCF Cartesian Hessian to {xyzin_path}",
        )
    return RemoteQMPromotionResult(
        mode=mode,
        status="skipped",
        message=f"no MATRIX promotion adapter is defined for engine {metadata.engine}",
    )


def _resolved_promote_mode(promote: str, metadata: RemoteQMMetadata) -> str:
    if promote != "auto":
        return promote
    if metadata.engine == "molpro":
        return "molpro"
    if metadata.engine in {"gdv32", "g16"}:
        suffix = Path(metadata.native_output).suffix.lower()
        if suffix in {".fchk", ".fch"}:
            return "gaussian-fchk"
        return "none"
    if metadata.engine == "orca":
        return "orca"
    return "none"


def _write_fetch_manifest(
    target_dir: Path,
    *,
    host: str,
    metadata: RemoteQMMetadata,
    output_path: Path,
    metadata_path: Path,
    artifacts: dict[str, Path],
    promotion: RemoteQMPromotionResult | None,
) -> Path:
    files = {
        "native_output": str(output_path),
        "metadata": str(metadata_path),
        **{name: str(path) for name, path in sorted(artifacts.items())},
    }
    manifest = {
        "schema": "matrix.remote_qm.fetch.v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": host,
        "job": metadata.job,
        "engine": metadata.engine,
        "metadata": asdict(metadata),
        "files": files,
        "sha256": {name: _sha256_file(Path(path)) for name, path in files.items()},
        "promotion": None if promotion is None else asdict(promotion),
    }
    target = target_dir / "remote_qm_fetch_manifest.json"
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def _fetch_remote_xtb_hessian_bundle(
    metadata: RemoteQMMetadata,
    *,
    host: str,
    target_dir: Path,
    scp_executable: str,
    runner: SubprocessRunner,
) -> dict[str, Path]:
    if metadata.engine != "xtb":
        raise RemoteQMError("xtb-hessian promotion requires an xTB remote job")
    if not metadata.workdir or not metadata.input:
        raise RemoteQMError("xTB remote metadata must contain workdir and input paths")
    remote_files = {
        "xtb_hessian": f"{metadata.workdir.rstrip('/')}/hessian",
        "xtb_spectrum": f"{metadata.workdir.rstrip('/')}/vibspectrum",
        "xtb_geometry": metadata.input,
    }
    local_files = {
        "xtb_hessian": target_dir / "hessian",
        "xtb_spectrum": target_dir / "vibspectrum",
        "xtb_geometry": target_dir / Path(metadata.input).name,
    }
    for role, remote_path in remote_files.items():
        _run_checked(
            _cmd(scp_executable, f"{host}:{remote_path}", str(local_files[role])),
            runner=runner,
            context=f"copy remote xTB Hessian artifact {role}",
        )
    return local_files


def _parse_submit_output(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("Submitted "):
            values["job"] = line.split(maxsplit=1)[1].strip()
        elif line.startswith("Workdir:"):
            values["workdir"] = line.partition(":")[2].strip()
        elif line.startswith("Log:"):
            values["log"] = line.partition(":")[2].strip()
        elif line.startswith("Native output:"):
            values["native_output"] = line.partition(":")[2].strip()
        elif line.startswith("Stdout:"):
            values["stdout"] = line.partition(":")[2].strip()
    if "job" not in values:
        raise RemoteQMError(f"could not parse remote submit output:\n{text}")
    return values


def _run_checked(
    argv: Sequence[str],
    *,
    runner: SubprocessRunner,
    context: str,
) -> subprocess.CompletedProcess[str]:
    completed = runner(
        list(argv),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        detail = stderr or stdout or f"exit code {completed.returncode}"
        raise RemoteQMError(f"failed to {context}: {detail}")
    return completed


def _remote_bash(command: str) -> str:
    return "bash -lc " + shlex.quote(command)


def _cmd(executable: str, *args: str) -> list[str]:
    return [*shlex.split(executable), *args]


def _q(value: str) -> str:
    return shlex.quote(str(value))


def _remote_arg(value: str) -> str:
    text = str(value)
    if text.startswith("~/"):
        return "~/" + shlex.quote(text[2:])
    if text.startswith("$HOME/"):
        return "$HOME/" + shlex.quote(text[6:])
    return shlex.quote(text)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remote_fetch_cli_hint(result: RemoteQMFetchResult) -> str:
    if result.promotion is not None:
        return result.promotion.message
    if result.engine in {"gdv32", "g16"}:
        return (
            "Gaussian output fetched. Use --promote gaussian-log-hessian, "
            "--promote gaussian-rovib, --promote gaussian-electronic or "
            "--promote gaussian-fchk with --xyzin when the requested section is present."
        )
    if result.engine == "orca":
        return (
            "ORCA output fetched. Use --promote orca --xyzin to normalize final geometry "
            "and Cartesian Hessian when printed."
        )
    if result.engine == "xtb":
        return (
            "xTB output fetched. For Hessian jobs use --promote xtb-hessian --xyzin "
            "to fetch hessian, input geometry and vibspectrum as one checksummed bundle."
        )
    if result.engine == "pyscf":
        return (
            "PySCF output fetched. For self-reporting Hessian jobs use "
            "--promote pyscf-hessian --xyzin to normalize Hessian and frequencies."
        )
    if result.engine == "et":
        return "eT output fetched; LINK supplies numerical derivatives when required."
    return "Output fetched. No promotion was requested."


def python_cli_command() -> tuple[str, str, str]:
    return (sys.executable, "-m", "matrix")
