"""Qualified execution-node snapshots shared by Keymaker and CLI workflows."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import importlib
import importlib.util
import json
from pathlib import Path
import platform
import shlex
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

from .environment import (
    DEFAULT_CONFIG_PATH,
    MachineLimits,
    MatrixEnvironment,
    RemoteMachine,
    default_environment,
    load_runtime_environment,
    normalize_architecture,
    normalize_operating_system,
)
from .package_registry import (
    PACKAGE_CAPABILITIES,
    PACKAGE_CAPABILITY_REGISTRY_SCHEMA,
)


HOST_CAPABILITY_SNAPSHOT_SCHEMA = "matrix.host-capability-snapshot.v1"
HOST_QUALIFICATION_SCHEMA = "matrix.host-qualification.v1"
ENVIRONMENT_HOST_QUALIFICATION_SCHEMA = "matrix.environment-host-qualification.v1"

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _capability_available_for_host(
    contract: object,
    capability: str,
    *,
    operating_system: str,
    architecture: str,
    limits: MachineLimits,
) -> bool:
    requirements = getattr(contract, "capability_requirements", {}).get(capability, {})
    systems = {normalize_operating_system(str(value)) for value in requirements.get("systems", ())}
    architectures = {
        normalize_architecture(str(value)) for value in requirements.get("architectures", ())
    }
    if systems and operating_system not in systems:
        return False
    if architectures and architecture not in architectures:
        return False
    if limits.gpu_count < int(requirements.get("gpu_count", 0)):
        return False
    if limits.neural_engine_count < int(requirements.get("neural_engine_count", 0)):
        return False
    return True


@dataclass(frozen=True)
class HostCapabilitySnapshot:
    """Observed software and hardware contract for one execution node."""

    name: str
    kind: str
    observed_at_utc: str
    reachable: bool
    operating_system: str
    architecture: str
    python_version: str
    python_abi: str
    matrix_commit: str
    dirty_files: int
    packages: tuple[str, ...]
    capabilities: tuple[str, ...]
    programs: tuple[str, ...]
    native_backends: tuple[str, ...]
    limits: MachineLimits
    host: str = ""
    package_errors: tuple[str, ...] = ()
    package_registry_schema: str = PACKAGE_CAPABILITY_REGISTRY_SCHEMA
    package_count: int = len(PACKAGE_CAPABILITIES)
    error: str = ""
    schema: str = HOST_CAPABILITY_SNAPSHOT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != HOST_CAPABILITY_SNAPSHOT_SCHEMA:
            raise ValueError(f"unsupported host snapshot schema: {self.schema}")
        if self.kind not in {"local", "remote"}:
            raise ValueError("host snapshot kind must be local or remote")
        if not self.name.strip():
            raise ValueError("host snapshot name is required")
        if self.kind == "remote" and not self.host.strip():
            raise ValueError("remote host snapshot requires an SSH host")
        if self.dirty_files < 0:
            raise ValueError("dirty_files cannot be negative")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for key in (
            "packages",
            "capabilities",
            "programs",
            "native_backends",
            "package_errors",
        ):
            payload[key] = list(payload[key])
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "HostCapabilitySnapshot":
        payload = dict(data)
        raw_limits = payload.get("limits", {})
        if not isinstance(raw_limits, Mapping):
            raise ValueError("host snapshot limits must be an object")
        payload["limits"] = MachineLimits(**dict(raw_limits))
        for key in (
            "packages",
            "capabilities",
            "programs",
            "native_backends",
            "package_errors",
        ):
            payload[key] = tuple(payload.get(key, ()))
        return cls(**payload)


@dataclass(frozen=True)
class HostQualification:
    """Eligibility result for a concrete scheduling request."""

    host: str
    eligible: bool
    issues: tuple[str, ...]
    required_packages: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    required_programs: tuple[str, ...] = ()
    required_native_backends: tuple[str, ...] = ()
    schema: str = HOST_QUALIFICATION_SCHEMA

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for key in (
            "issues",
            "required_packages",
            "required_capabilities",
            "required_programs",
            "required_native_backends",
        ):
            payload[key] = list(payload[key])
        return payload


def build_host_capability_snapshot(
    environment: MatrixEnvironment,
    *,
    name: str = "local",
    kind: str = "local",
    host: str = "",
    runner: CommandRunner = subprocess.run,
    observed_at_utc: str | None = None,
) -> HostCapabilitySnapshot:
    """Inspect the current Python/runtime without asserting remote reachability."""

    programs = tuple(
        sorted(item.key for item in environment.programs if item.enabled and item.available)
    )
    packages: list[str] = []
    package_errors: list[str] = []
    capabilities: set[str] = set()
    native_backends: set[str] = set()
    operating_system = normalize_operating_system(platform.system())
    architecture = normalize_architecture(platform.machine())
    limits = environment.local_limits
    for contract in PACKAGE_CAPABILITIES:
        import_error = _package_import_error(contract.import_name)
        if import_error:
            package_errors.append(f"{contract.package}: {import_error}")
            continue
        packages.append(contract.package)
        provider_ready = not contract.provider_programs or bool(
            set(contract.provider_programs) & set(programs)
        )
        if provider_ready:
            capabilities.update(
                capability
                for capability in contract.capabilities
                if _capability_available_for_host(
                    contract,
                    capability,
                    operating_system=operating_system,
                    architecture=architecture,
                    limits=limits,
                )
            )
        native_backends.update(
            backend for backend in contract.native_backends if _native_backend_available(backend)
        )
    root = Path(environment.matrix_root).expanduser().resolve()
    commit = _git_output(root, ("rev-parse", "HEAD"), runner=runner)
    dirty_output = _git_output(root, ("status", "--porcelain"), runner=runner)
    return HostCapabilitySnapshot(
        name=str(name),
        kind=str(kind),
        host=str(host),
        observed_at_utc=observed_at_utc or _utc_now(),
        reachable=True,
        operating_system=operating_system,
        architecture=architecture,
        python_version=platform.python_version(),
        python_abi=sys.implementation.cache_tag or "",
        matrix_commit=commit,
        dirty_files=len(dirty_output.splitlines()) if dirty_output else 0,
        packages=tuple(sorted(packages)),
        capabilities=tuple(sorted(capabilities)),
        programs=programs,
        native_backends=tuple(sorted(native_backends)),
        limits=limits,
        package_errors=tuple(sorted(package_errors)),
    )


def unreachable_host_capability_snapshot(
    *,
    name: str,
    host: str,
    error: str,
    observed_at_utc: str | None = None,
) -> HostCapabilitySnapshot:
    return HostCapabilitySnapshot(
        name=str(name),
        kind="remote",
        host=str(host),
        observed_at_utc=observed_at_utc or _utc_now(),
        reachable=False,
        operating_system="",
        architecture="",
        python_version="",
        python_abi="",
        matrix_commit="",
        dirty_files=0,
        packages=(),
        capabilities=(),
        programs=(),
        native_backends=(),
        limits=MachineLimits(),
        package_count=0,
        error=str(error),
    )


def probe_remote_host_capabilities(
    machine: RemoteMachine,
    *,
    runner: CommandRunner = subprocess.run,
    connect_timeout_seconds: int = 6,
    process_timeout_seconds: float = 10.0,
) -> HostCapabilitySnapshot:
    """Read one configured node's canonical capability snapshot over SSH."""

    expand = shlex.quote(
        "import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))"
    )
    probe = (
        '"$runtime_venv/bin/python" -m matrix_core.host_capabilities '
        '--environment "$config_dir/environment.json" --matrix-root "$root" '
        f"--name {shlex.quote(machine.name)} --kind remote "
        f"--host {shlex.quote(machine.host)}"
    )
    script = "\n".join(
        (
            f"root=$(python3 -c {expand} {shlex.quote(machine.remote_root)})",
            f"runtime_venv=$(python3 -c {expand} {shlex.quote(machine.runtime_venv)})",
            f"config_dir=$(python3 -c {expand} {shlex.quote(machine.runtime_config_dir)})",
            'cd "$root"',
            'export MATRIX_HOME="$root"',
            'export MATRIX_VENV="$runtime_venv"',
            'export MATRIX_CONFIG_DIR="$config_dir"',
            probe,
        )
    )
    try:
        completed = runner(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                f"ConnectTimeout={max(1, int(connect_timeout_seconds))}",
                machine.host,
                f"bash -lc {shlex.quote(script)}",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=max(1.0, float(process_timeout_seconds)),
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr.strip() or "SSH failed")
        start = completed.stdout.find("{")
        if start < 0:
            raise RuntimeError("remote probe returned no JSON snapshot")
        snapshot = HostCapabilitySnapshot.from_dict(
            json.loads(completed.stdout[start:])
        )
        if snapshot.name != machine.name or snapshot.host != machine.host:
            raise RuntimeError(
                "remote snapshot identity does not match the configured node"
            )
        return snapshot
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return unreachable_host_capability_snapshot(
            name=machine.name,
            host=machine.host,
            error=str(exc),
        )


def qualify_host_snapshot(
    snapshot: HostCapabilitySnapshot,
    *,
    expected_commit: str = "",
    expected_python_abi: str = "",
    expected_operating_system: str = "",
    expected_architecture: str = "",
    required_packages: Sequence[str] = (),
    required_capabilities: Sequence[str] = (),
    required_programs: Sequence[str] = (),
    required_native_backends: Sequence[str] = (),
    required_systems: Sequence[str] = (),
    required_architectures: Sequence[str] = (),
    cpu_cores: int = 1,
    memory_gb: float = 0.1,
    gpu_count: int = 0,
    neural_engine_count: int = 0,
    require_clean: bool = True,
    max_age_seconds: float | None = 900.0,
    now: datetime | None = None,
) -> HostQualification:
    """Fail closed when a snapshot cannot satisfy a reproducible request."""

    packages = tuple(
        sorted({_normalize_package_name(str(value)) for value in required_packages if str(value)})
    )
    capabilities = tuple(
        sorted(
            {
                str(value).strip().upper().replace("-", "_")
                for value in required_capabilities
                if str(value)
            }
        )
    )
    programs = tuple(
        sorted({str(value).strip().casefold() for value in required_programs if str(value)})
    )
    native_backends = tuple(
        sorted({str(value).strip() for value in required_native_backends if str(value).strip()})
    )
    issues: list[str] = []
    if not snapshot.reachable:
        issues.append(snapshot.error or "node is unreachable")
    if snapshot.package_registry_schema != PACKAGE_CAPABILITY_REGISTRY_SCHEMA:
        issues.append("package registry schema differs from the scheduler")
    if snapshot.package_count != len(PACKAGE_CAPABILITIES):
        issues.append(
            f"package registry count is {snapshot.package_count}, expected {len(PACKAGE_CAPABILITIES)}"
        )
    if expected_commit and snapshot.matrix_commit != expected_commit:
        issues.append(
            f"MATRIX commit is {snapshot.matrix_commit or 'unknown'}, expected {expected_commit}"
        )
    if expected_python_abi and snapshot.python_abi != expected_python_abi:
        issues.append(
            f"Python ABI is {snapshot.python_abi or 'unknown'}, expected {expected_python_abi}"
        )
    expected_system = normalize_operating_system(expected_operating_system)
    if expected_system and snapshot.operating_system != expected_system:
        issues.append(
            f"operating system is {snapshot.operating_system or 'unknown'}, "
            f"expected {expected_system}"
        )
    expected_machine = normalize_architecture(expected_architecture)
    if expected_machine and snapshot.architecture != expected_machine:
        issues.append(
            f"architecture is {snapshot.architecture or 'unknown'}, expected {expected_machine}"
        )
    if require_clean and snapshot.dirty_files:
        issues.append(f"MATRIX checkout has {snapshot.dirty_files} dirty file(s)")
    if max_age_seconds is not None:
        try:
            observed = datetime.fromisoformat(snapshot.observed_at_utc)
            reference = now or datetime.now(timezone.utc)
            if observed.tzinfo is None:
                observed = observed.replace(tzinfo=timezone.utc)
            age_seconds = (reference - observed).total_seconds()
            if age_seconds > max_age_seconds:
                issues.append("capability snapshot is stale")
            elif age_seconds < -300.0:
                issues.append("capability snapshot clock is more than five minutes ahead")
        except ValueError:
            issues.append("capability snapshot timestamp is invalid")
    systems = {normalize_operating_system(value) for value in required_systems}
    if systems and snapshot.operating_system not in systems:
        issues.append(f"operating system {snapshot.operating_system or 'unknown'} is not supported")
    architectures = {normalize_architecture(value) for value in required_architectures}
    if architectures and snapshot.architecture not in architectures:
        issues.append(f"architecture {snapshot.architecture or 'unknown'} is not supported")
    missing_packages = sorted(set(packages) - set(snapshot.packages))
    if missing_packages:
        issues.append("missing packages: " + ", ".join(missing_packages))
    missing_capabilities = sorted(set(capabilities) - set(snapshot.capabilities))
    if missing_capabilities:
        issues.append("missing capabilities: " + ", ".join(missing_capabilities))
    missing_programs = sorted(set(programs) - set(snapshot.programs))
    if missing_programs:
        issues.append("missing programs: " + ", ".join(missing_programs))
    missing_native_backends = sorted(set(native_backends) - set(snapshot.native_backends))
    if missing_native_backends:
        issues.append("missing native backends: " + ", ".join(missing_native_backends))
    if int(cpu_cores) > snapshot.limits.processors:
        issues.append(
            f"requires {int(cpu_cores)} CPU cores, node exposes {snapshot.limits.processors}"
        )
    if float(memory_gb) > snapshot.limits.memory_gb:
        issues.append(
            f"requires {float(memory_gb):g} GB, node exposes {snapshot.limits.memory_gb:g} GB"
        )
    if int(gpu_count) > snapshot.limits.gpu_count:
        issues.append(f"requires {int(gpu_count)} GPU(s), node exposes {snapshot.limits.gpu_count}")
    if int(neural_engine_count) > snapshot.limits.neural_engine_count:
        issues.append(
            "requires "
            f"{int(neural_engine_count)} Neural Engine device(s), node exposes "
            f"{snapshot.limits.neural_engine_count}"
        )
    return HostQualification(
        host=snapshot.name,
        eligible=not issues,
        issues=tuple(issues),
        required_packages=packages,
        required_capabilities=capabilities,
        required_programs=programs,
        required_native_backends=native_backends,
    )


def qualify_environment_hosts(
    environment: MatrixEnvironment,
    *,
    local_runner: CommandRunner = subprocess.run,
    remote_runner: CommandRunner = subprocess.run,
) -> dict[str, Any]:
    """Qualify local and configured remote nodes against one source contract."""

    local = build_host_capability_snapshot(environment, runner=local_runner)
    observations: list[tuple[HostCapabilitySnapshot, HostQualification]] = []
    local_qualification = qualify_host_snapshot(
        local,
        expected_commit=local.matrix_commit,
        expected_python_abi=local.python_abi,
        expected_operating_system=local.operating_system,
        expected_architecture=local.architecture,
        require_clean=True,
    )
    if not local.matrix_commit:
        local_qualification = replace(
            local_qualification,
            eligible=False,
            issues=(*local_qualification.issues, "MATRIX commit is unavailable"),
        )
    observations.append((local, local_qualification))
    for machine in environment.remote_machines:
        if not machine.enabled:
            continue
        snapshot = probe_remote_host_capabilities(machine, runner=remote_runner)
        qualification = qualify_host_snapshot(
            snapshot,
            expected_commit=local.matrix_commit,
            expected_python_abi=machine.expected_python_abi,
            expected_operating_system=machine.expected_operating_system,
            expected_architecture=machine.expected_architecture,
            require_clean=True,
        )
        observations.append((snapshot, qualification))
    nodes = [
        {
            "name": snapshot.name,
            "snapshot": snapshot.to_dict(),
            "qualification": qualification.to_dict(),
        }
        for snapshot, qualification in observations
    ]
    return {
        "schema": ENVIRONMENT_HOST_QUALIFICATION_SCHEMA,
        "status": (
            "PASS"
            if all(item["qualification"]["eligible"] for item in nodes)
            else "MISMATCH"
        ),
        "nodes": nodes,
    }


def host_snapshot_json(snapshot: HostCapabilitySnapshot, *, indent: int = 2) -> str:
    return json.dumps(snapshot.to_dict(), indent=indent, sort_keys=True)


def host_capability_snapshot_schema_path() -> Path:
    return Path(__file__).resolve().parent / "schemas" / "host-capability-snapshot-v1.schema.json"


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _package_import_error(name: str) -> str:
    try:
        importlib.import_module(name)
    except Exception as exc:
        return f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
    return ""


def _normalize_package_name(value: str) -> str:
    normalized = value.strip().casefold().replace("_", "-")
    return normalized if normalized.startswith("matrix-") else f"matrix-{normalized}"


def _native_backend_available(name: str) -> bool:
    if name == "numpy-scipy":
        return _module_available("numpy") and _module_available("scipy")
    if name == "gpu":
        return _module_available("cupy") or _module_available("torch")
    if "." in name or name == "fmm3d":
        return _module_available(name)
    return False


def _git_output(
    root: Path,
    arguments: Sequence[str],
    *,
    runner: CommandRunner,
) -> str:
    try:
        completed = runner(
            ("git", "-C", str(root), *arguments),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print this MATRIX node capability snapshot")
    parser.add_argument("--environment", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--matrix-root", type=Path)
    parser.add_argument("--name", default="local")
    parser.add_argument("--kind", choices=("local", "remote"), default="local")
    parser.add_argument("--host", default="")
    args = parser.parse_args(argv)
    environment = (
        load_runtime_environment(args.environment)
        if args.environment.is_file()
        else default_environment(matrix_root=args.matrix_root)
    )
    if args.matrix_root is not None:
        environment = replace(
            environment,
            matrix_root=str(args.matrix_root.expanduser().resolve()),
        )
    snapshot = build_host_capability_snapshot(
        environment,
        name=args.name,
        kind=args.kind,
        host=args.host,
    )
    print(host_snapshot_json(snapshot))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "HOST_CAPABILITY_SNAPSHOT_SCHEMA",
    "HOST_QUALIFICATION_SCHEMA",
    "ENVIRONMENT_HOST_QUALIFICATION_SCHEMA",
    "HostCapabilitySnapshot",
    "HostQualification",
    "build_host_capability_snapshot",
    "host_capability_snapshot_schema_path",
    "host_snapshot_json",
    "normalize_architecture",
    "normalize_operating_system",
    "probe_remote_host_capabilities",
    "qualify_host_snapshot",
    "qualify_environment_hosts",
    "unreachable_host_capability_snapshot",
]
