"""Immutable, resource-explicit authorization contract for QM launches."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
from typing import Mapping, Sequence


CALCULATION_LAUNCH_PLAN_SCHEMA = "matrix.calculation-launch-plan.v1"
CALCULATION_LAUNCH_AUTHORIZATION_SCHEMA = "matrix.calculation-launch-authorization.v1"
CALCULATION_AUTHORIZATION_BUNDLE_SCHEMA = "matrix.calculation-authorization-bundle.v1"
DELEGATED_CALCULATION_AUTHORIZATION_BUNDLE_SCHEMA = (
    "matrix.delegated-calculation-authorization-bundle.v1"
)
CALCULATION_AUTHORIZATION_BUNDLE_ENV = "MATRIX_CALCULATION_AUTHORIZATION_BUNDLE"


class CalculationLaunchError(RuntimeError):
    """Raised when a calculation launch is incomplete, stale, or unauthorized."""


@dataclass(frozen=True)
class CalculationResources:
    """Maximum resource envelope for one launch request.

    ``process_count`` and ``threads_per_process`` describe each concurrent job.
    ``memory_per_job_gb`` is the maximum total memory available to one job, not
    memory per MPI process or per thread.
    """

    process_count: int
    threads_per_process: int
    memory_per_job_gb: float
    concurrent_jobs: int = 1

    def __post_init__(self) -> None:
        for label, value in (
            ("process_count", self.process_count),
            ("threads_per_process", self.threads_per_process),
            ("concurrent_jobs", self.concurrent_jobs),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        if (
            isinstance(self.memory_per_job_gb, bool)
            or not isinstance(self.memory_per_job_gb, (int, float))
            or not math.isfinite(float(self.memory_per_job_gb))
            or float(self.memory_per_job_gb) <= 0.0
        ):
            raise ValueError("memory_per_job_gb must be a positive finite number")

    @property
    def threads_per_job(self) -> int:
        return self.process_count * self.threads_per_process

    @property
    def maximum_processes(self) -> int:
        return self.process_count * self.concurrent_jobs

    @property
    def maximum_threads(self) -> int:
        return self.threads_per_job * self.concurrent_jobs

    @property
    def maximum_memory_gb(self) -> float:
        return float(self.memory_per_job_gb) * self.concurrent_jobs

    def to_dict(self) -> dict[str, object]:
        return {
            "process_count_per_job": self.process_count,
            "threads_per_process": self.threads_per_process,
            "threads_per_job": self.threads_per_job,
            "memory_per_job_gb": float(self.memory_per_job_gb),
            "concurrent_jobs": self.concurrent_jobs,
            "maximum_processes": self.maximum_processes,
            "maximum_threads": self.maximum_threads,
            "maximum_memory_gb": self.maximum_memory_gb,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "CalculationResources":
        return cls(
            process_count=int(data["process_count_per_job"]),
            threads_per_process=int(data["threads_per_process"]),
            memory_per_job_gb=float(data["memory_per_job_gb"]),
            concurrent_jobs=int(data.get("concurrent_jobs", 1)),
        )


@dataclass(frozen=True)
class ConcurrentJobEstimate:
    """Auditable resource bound for independent calculations run in parallel."""

    workers: int
    independent_calculations: int
    cpu_cap: int
    memory_cap: int
    job_cap: int
    requested_cap: int | None

    @property
    def limiting_factors(self) -> tuple[str, ...]:
        limits = {
            "independent_calculations": self.independent_calculations,
            "cpu": self.cpu_cap,
            "memory": self.memory_cap,
            "job_limit": self.job_cap,
        }
        if self.requested_cap is not None:
            limits["requested_cap"] = self.requested_cap
        return tuple(name for name, value in limits.items() if value == self.workers)


def estimate_concurrent_jobs(
    *,
    independent_calculations: int,
    threads_per_calculation: int,
    memory_per_calculation_gb: float,
    assigned_threads: int,
    assigned_memory_gb: float,
    assigned_job_limit: int,
    requested_cap: int | None = None,
) -> ConcurrentJobEstimate:
    """Choose the largest safe concurrency inside an assigned resource envelope.

    ``requested_cap`` is an optional user ceiling, never a request to exceed the
    CPU, memory, scheduler, or available-work limits.
    """

    integer_values = {
        "independent_calculations": independent_calculations,
        "threads_per_calculation": threads_per_calculation,
        "assigned_threads": assigned_threads,
        "assigned_job_limit": assigned_job_limit,
    }
    for label, value in integer_values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label} must be a positive integer")
    for label, value in (
        ("memory_per_calculation_gb", memory_per_calculation_gb),
        ("assigned_memory_gb", assigned_memory_gb),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"{label} must be a positive finite number")
    if requested_cap is not None and (
        isinstance(requested_cap, bool)
        or not isinstance(requested_cap, int)
        or requested_cap < 1
    ):
        raise ValueError("requested_cap must be a positive integer when specified")

    cpu_cap = assigned_threads // threads_per_calculation
    memory_cap = int(math.floor(assigned_memory_gb / memory_per_calculation_gb + 1.0e-12))
    if cpu_cap < 1:
        raise ValueError("assigned CPU threads cannot accommodate one energy calculation")
    if memory_cap < 1:
        raise ValueError("assigned memory cannot accommodate one energy calculation")
    caps = [independent_calculations, cpu_cap, memory_cap, assigned_job_limit]
    if requested_cap is not None:
        caps.append(requested_cap)
    return ConcurrentJobEstimate(
        workers=min(caps),
        independent_calculations=independent_calculations,
        cpu_cap=cpu_cap,
        memory_cap=memory_cap,
        job_cap=assigned_job_limit,
        requested_cap=requested_cap,
    )


@dataclass(frozen=True)
class CalculationLaunchPlan:
    """Exact input, command, host, and resource envelope awaiting approval."""

    backend: str
    host: str
    workdir: str
    input_path: str
    input_sha256: str
    command: tuple[str, ...]
    resources: CalculationResources
    schema: str = CALCULATION_LAUNCH_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CALCULATION_LAUNCH_PLAN_SCHEMA:
            raise ValueError(f"unsupported calculation launch-plan schema: {self.schema}")
        if not self.backend.strip():
            raise ValueError("calculation launch plan requires a backend")
        if not self.host.strip():
            raise ValueError("calculation launch plan requires a host")
        if not self.command or any(not str(part) for part in self.command):
            raise ValueError("calculation launch plan requires a non-empty command")
        if len(self.input_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.input_sha256
        ):
            raise ValueError("calculation launch plan requires a SHA-256 input digest")

    def unsigned_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "backend": self.backend,
            "host": self.host,
            "workdir": self.workdir,
            "input_path": self.input_path,
            "input_sha256": self.input_sha256,
            "command": list(self.command),
            "resources": self.resources.to_dict(),
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.unsigned_dict())

    def to_dict(self) -> dict[str, object]:
        return {**self.unsigned_dict(), "plan_sha256": self.sha256}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "CalculationLaunchPlan":
        resources = data.get("resources")
        if not isinstance(resources, Mapping):
            raise ValueError("calculation launch plan resources must be an object")
        raw_command = data.get("command")
        if not isinstance(raw_command, Sequence) or isinstance(raw_command, (str, bytes)):
            raise ValueError("calculation launch plan command must be an array")
        plan = cls(
            backend=str(data["backend"]),
            host=str(data["host"]),
            workdir=str(data["workdir"]),
            input_path=str(data["input_path"]),
            input_sha256=str(data["input_sha256"]),
            command=tuple(str(item) for item in raw_command),
            resources=CalculationResources.from_dict(resources),
            schema=str(data.get("schema", CALCULATION_LAUNCH_PLAN_SCHEMA)),
        )
        expected = str(data.get("plan_sha256", plan.sha256))
        if expected != plan.sha256:
            raise CalculationLaunchError("stored calculation launch-plan digest is invalid")
        return plan


@dataclass(frozen=True)
class CalculationLaunchAuthorization:
    """User authorization bound to one exact calculation launch plan."""

    plan_sha256: str
    authorized_by: str
    approved_at_utc: str
    schema: str = CALCULATION_LAUNCH_AUTHORIZATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CALCULATION_LAUNCH_AUTHORIZATION_SCHEMA:
            raise ValueError(
                f"unsupported calculation authorization schema: {self.schema}"
            )
        if len(self.plan_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.plan_sha256
        ):
            raise ValueError("calculation authorization requires a SHA-256 plan digest")
        if not self.authorized_by.strip():
            raise ValueError("calculation authorization requires an authorizer")
        if not self.approved_at_utc.strip():
            raise ValueError("calculation authorization requires a timestamp")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "CalculationLaunchAuthorization":
        return cls(
            plan_sha256=str(data["plan_sha256"]),
            authorized_by=str(data["authorized_by"]),
            approved_at_utc=str(data["approved_at_utc"]),
            schema=str(data.get("schema", CALCULATION_LAUNCH_AUTHORIZATION_SCHEMA)),
        )


def build_calculation_launch_plan(
    *,
    backend: str,
    input_path: Path | str,
    command: Sequence[str],
    resources: CalculationResources,
    workdir: Path | str | None = None,
    host: str | None = None,
) -> CalculationLaunchPlan:
    """Build a plan from the exact on-disk input and command to be executed."""

    source = Path(input_path).expanduser().resolve()
    if not source.is_file():
        raise CalculationLaunchError(f"calculation input not found: {source}")
    directory = (
        Path(workdir).expanduser().resolve() if workdir is not None else source.parent
    )
    return CalculationLaunchPlan(
        backend=str(backend).strip(),
        host=str(host or platform.node() or "local").strip(),
        workdir=str(directory),
        input_path=str(source),
        input_sha256=_sha256_file(source),
        command=tuple(str(part) for part in command),
        resources=resources,
    )


def calculation_launch_plan_lines(plan: CalculationLaunchPlan) -> tuple[str, ...]:
    """Return the mandatory human-readable approval summary."""

    resources = plan.resources
    return (
        f"Backend: {plan.backend}",
        f"Host: {plan.host}",
        f"Work directory: {plan.workdir}",
        f"Input: {plan.input_path}",
        f"Input SHA-256: {plan.input_sha256}",
        f"Command: {shlex.join(plan.command)}",
        f"Processes per job: {resources.process_count}",
        f"Threads per process: {resources.threads_per_process}",
        f"Threads per job: {resources.threads_per_job}",
        f"Concurrent jobs: {resources.concurrent_jobs}",
        f"Maximum processes: {resources.maximum_processes}",
        f"Maximum CPU threads: {resources.maximum_threads}",
        f"Maximum memory per job: {_format_gb(resources.memory_per_job_gb)} GB",
        f"Maximum aggregate memory: {_format_gb(resources.maximum_memory_gb)} GB",
        f"Launch plan SHA-256: {plan.sha256}",
    )


def authorize_calculation_launch(
    plan: CalculationLaunchPlan,
    *,
    approved_plan_sha256: str,
    authorized_by: str = "user",
    approved_at_utc: str | None = None,
) -> CalculationLaunchAuthorization:
    """Record a specific affirmative decision for the displayed exact plan."""

    if str(approved_plan_sha256).strip().lower() != plan.sha256:
        raise CalculationLaunchError(
            "authorization does not match the exact displayed calculation launch plan"
        )
    return CalculationLaunchAuthorization(
        plan_sha256=plan.sha256,
        authorized_by=str(authorized_by),
        approved_at_utc=approved_at_utc
        or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    )


def require_calculation_launch_authorization(
    plan: CalculationLaunchPlan,
    authorization: CalculationLaunchAuthorization | None,
) -> None:
    """Reject missing, stale, or input-invalid calculation authorization."""

    if authorization is None:
        raise CalculationLaunchError(
            "specific user authorization is required before launching this calculation"
        )
    if authorization.plan_sha256 != plan.sha256:
        raise CalculationLaunchError(
            "calculation authorization does not match the exact launch plan"
        )
    source = Path(plan.input_path)
    if not source.is_file() or _sha256_file(source) != plan.input_sha256:
        raise CalculationLaunchError(
            "calculation input changed after authorization; display and authorize a new plan"
        )


def write_calculation_authorization_bundle(
    path: Path | str,
    plan: CalculationLaunchPlan,
    authorization: CalculationLaunchAuthorization,
) -> Path:
    """Persist an authorized parent workflow for audited descendant QM jobs."""

    require_calculation_launch_authorization(plan, authorization)
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": CALCULATION_AUTHORIZATION_BUNDLE_SCHEMA,
        "plan": plan.to_dict(),
        "authorization": authorization.to_dict(),
    }
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def write_delegated_calculation_authorization_bundle(
    path: Path | str,
    parent_bundle_path: Path | str,
    *,
    delegated_host: str,
    delegated_input_path: Path | str,
) -> Path:
    """Delegate an authorized remote workflow to its byte-identical copied input.

    The original authorization remains bound to the exact Keymaker plan.  This
    envelope only relocates that already-authorized input to the host named in
    the plan; it cannot change backend, command, resources, host, or input
    digest.
    """

    parent, authorization = read_calculation_authorization_bundle(parent_bundle_path)
    host = str(delegated_host).strip()
    if host != parent.host:
        raise CalculationLaunchError(
            "delegated calculation host differs from the authorized workflow"
        )
    remote_input = str(delegated_input_path).strip()
    if not remote_input:
        raise CalculationLaunchError("delegated calculation input path is empty")
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": DELEGATED_CALCULATION_AUTHORIZATION_BUNDLE_SCHEMA,
        "parent_plan": parent.to_dict(),
        "authorization": authorization.to_dict(),
        "delegation": {
            "host": host,
            "input_path": remote_input,
            "input_sha256": parent.input_sha256,
        },
    }
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def read_calculation_authorization_bundle(
    path: Path | str,
) -> tuple[CalculationLaunchPlan, CalculationLaunchAuthorization]:
    source = Path(path).expanduser().resolve()
    payload = _read_bundle_payload(source)
    if payload.get("schema") != CALCULATION_AUTHORIZATION_BUNDLE_SCHEMA:
        raise CalculationLaunchError("unsupported calculation authorization bundle schema")
    if not isinstance(payload.get("plan"), Mapping) or not isinstance(
        payload.get("authorization"), Mapping
    ):
        raise CalculationLaunchError("calculation authorization bundle is incomplete")
    plan = CalculationLaunchPlan.from_dict(payload["plan"])
    authorization = CalculationLaunchAuthorization.from_dict(payload["authorization"])
    require_calculation_launch_authorization(plan, authorization)
    return plan, authorization


def authorized_parent_plan_from_environment(
    environment: Mapping[str, str] | None = None,
) -> CalculationLaunchPlan | None:
    values = os.environ if environment is None else environment
    bundle = str(values.get(CALCULATION_AUTHORIZATION_BUNDLE_ENV, "")).strip()
    if not bundle:
        return None
    source = Path(bundle).expanduser().resolve()
    payload = _read_bundle_payload(source)
    if payload.get("schema") == CALCULATION_AUTHORIZATION_BUNDLE_SCHEMA:
        plan, _authorization = read_calculation_authorization_bundle(source)
        return plan
    if payload.get("schema") == DELEGATED_CALCULATION_AUTHORIZATION_BUNDLE_SCHEMA:
        return _read_delegated_parent_plan(payload)
    raise CalculationLaunchError("unsupported calculation authorization bundle schema")


def require_calculation_launch_or_parent_authorization(
    plan: CalculationLaunchPlan,
    authorization: CalculationLaunchAuthorization | None,
    *,
    environment: Mapping[str, str] | None = None,
) -> None:
    """Accept exact authorization or a validated, resource-bounded parent workflow."""

    if authorization is not None:
        require_calculation_launch_authorization(plan, authorization)
        return
    parent = authorized_parent_plan_from_environment(environment)
    if parent is None:
        raise CalculationLaunchError(
            "specific user authorization is required before launching this calculation"
        )
    if plan.host != parent.host:
        raise CalculationLaunchError(
            "descendant calculation host differs from the authorized parent workflow"
        )
    if plan.resources.maximum_threads > parent.resources.maximum_threads:
        raise CalculationLaunchError(
            "descendant calculation exceeds the authorized CPU-thread limit"
        )
    if plan.resources.maximum_memory_gb > parent.resources.maximum_memory_gb + 1.0e-12:
        raise CalculationLaunchError(
            "descendant calculation exceeds the authorized memory limit"
        )


def require_authorized_descendant_calculation(
    *,
    backend: str,
    input_path: Path | str,
    command: Sequence[str],
    workdir: Path | str | None = None,
    environment: Mapping[str, str] | None = None,
) -> CalculationLaunchPlan:
    """Authorize an internal QM evaluator under one validated parent workflow."""

    parent = authorized_parent_plan_from_environment(environment)
    if parent is None:
        raise CalculationLaunchError(
            "specific parent authorization is required before launching this calculation"
        )
    plan = build_calculation_launch_plan(
        backend=backend,
        host=parent.host,
        workdir=workdir,
        input_path=input_path,
        command=command,
        resources=parent.resources,
    )
    require_calculation_launch_or_parent_authorization(
        plan,
        None,
        environment=environment,
    )
    return plan


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_bundle_payload(source: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalculationLaunchError(
            f"cannot read calculation authorization bundle: {source}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise CalculationLaunchError("calculation authorization bundle must be an object")
    return payload


def _read_delegated_parent_plan(payload: Mapping[str, object]) -> CalculationLaunchPlan:
    raw_plan = payload.get("parent_plan")
    raw_authorization = payload.get("authorization")
    raw_delegation = payload.get("delegation")
    if not isinstance(raw_plan, Mapping) or not isinstance(
        raw_authorization, Mapping
    ) or not isinstance(raw_delegation, Mapping):
        raise CalculationLaunchError("delegated calculation authorization is incomplete")
    parent = CalculationLaunchPlan.from_dict(raw_plan)
    authorization = CalculationLaunchAuthorization.from_dict(raw_authorization)
    if authorization.plan_sha256 != parent.sha256:
        raise CalculationLaunchError(
            "delegated authorization does not match its original launch plan"
        )
    host = str(raw_delegation.get("host", "")).strip()
    input_path = Path(str(raw_delegation.get("input_path", ""))).expanduser().resolve()
    input_sha256 = str(raw_delegation.get("input_sha256", ""))
    if host != parent.host:
        raise CalculationLaunchError(
            "delegated calculation host differs from the authorized workflow"
        )
    if input_sha256 != parent.input_sha256:
        raise CalculationLaunchError(
            "delegated calculation input digest differs from the authorized input"
        )
    if not input_path.is_file() or _sha256_file(input_path) != input_sha256:
        raise CalculationLaunchError(
            "delegated calculation input is missing or differs from the authorized input"
        )
    return CalculationLaunchPlan(
        backend=parent.backend,
        host=host,
        workdir=str(input_path.parent),
        input_path=str(input_path),
        input_sha256=input_sha256,
        command=parent.command,
        resources=parent.resources,
    )


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _format_gb(value: float) -> str:
    return f"{float(value):.6f}".rstrip("0").rstrip(".")
