"""Canonical capability- and load-aware execution routing for MATRIX."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .host_capabilities import HostCapabilitySnapshot, qualify_host_snapshot
from .environment import normalize_architecture, normalize_operating_system


EXECUTION_ROUTING_SCHEMA = "matrix.execution-routing.v1"
WORKLOAD_CLASSES = ("auto", "interactive", "standard", "heavy")


@dataclass(frozen=True)
class ExecutionRoutingPolicy:
    """Host-independent thresholds controlling only placement, never science."""

    heavy_walltime_seconds: float = 300.0
    heavy_local_cpu_fraction: float = 0.5
    heavy_local_memory_fraction: float = 0.5
    prefer_remote_for_heavy: bool = True
    prefer_local_for_standard: bool = True

    def __post_init__(self) -> None:
        if self.heavy_walltime_seconds <= 0:
            raise ValueError("heavy walltime threshold must be positive")
        for value in (
            self.heavy_local_cpu_fraction,
            self.heavy_local_memory_fraction,
        ):
            if not 0 < value <= 1:
                raise ValueError("heavy local resource fractions must be in (0, 1]")


@dataclass(frozen=True)
class ExecutionRoutingDecision:
    target: Mapping[str, Any] | None
    workload_class: str
    reason: str
    rejected_targets: Mapping[str, tuple[str, ...]]
    schema: str = EXECUTION_ROUTING_SCHEMA

    @property
    def target_name(self) -> str:
        return "" if self.target is None else str(self.target.get("name", ""))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["target"] = None if self.target is None else dict(self.target)
        payload["rejected_targets"] = {
            name: list(issues) for name, issues in self.rejected_targets.items()
        }
        payload["target_name"] = self.target_name
        return payload


def classify_workload(
    request: Mapping[str, object],
    targets: Sequence[Mapping[str, object]],
    *,
    policy: ExecutionRoutingPolicy | None = None,
) -> str:
    """Classify placement cost from declared resources and the local capacity."""

    selected_policy = policy or ExecutionRoutingPolicy()
    explicit = str(request.get("workload_class", "auto")).strip().casefold() or "auto"
    if explicit not in WORKLOAD_CLASSES:
        raise ValueError(f"unsupported workload class: {explicit}")
    if explicit != "auto":
        return explicit
    if str(request.get("execution_mode", "")).strip().casefold() == "interactive":
        return "interactive"
    if float(request.get("estimated_walltime_seconds", 0.0) or 0.0) >= (
        selected_policy.heavy_walltime_seconds
    ):
        return "heavy"
    if int(request.get("gpu_count", 0) or 0) > 0 or int(
        request.get("neural_engine_count", 0) or 0
    ) > 0:
        return "heavy"
    local = next(
        (
            target
            for target in targets
            if str(target.get("kind", "")).strip().casefold() == "local"
        ),
        None,
    )
    if local is not None:
        local_cpu = max(1, int(local.get("cpu_cores", 1)))
        local_memory = max(0.1, float(local.get("memory_gb", 0.1)))
        if int(request.get("cpu_cores", 1)) >= (
            selected_policy.heavy_local_cpu_fraction * local_cpu
        ):
            return "heavy"
        if float(request.get("memory_gb", 0.1)) >= (
            selected_policy.heavy_local_memory_fraction * local_memory
        ):
            return "heavy"
    return "standard"


def route_host(
    candidates: Sequence[Mapping[str, object]],
    *,
    request: Mapping[str, object] | None = None,
    allocations: Sequence[Mapping[str, object]] = (),
    policy: ExecutionRoutingPolicy | None = None,
) -> ExecutionRoutingDecision:
    """Select one eligible node using the shared MATRIX placement policy."""

    normalized_request = dict(request or {})
    normalized_request.setdefault("cpu_cores", 1)
    normalized_request.setdefault("memory_gb", 0.1)
    normalized_request.setdefault("execution_preference", "auto")
    normalized_request.setdefault("preferred_targets", ())
    normalized_request.setdefault("required_target", "")
    selected_policy = policy or ExecutionRoutingPolicy()
    workload = classify_workload(normalized_request, candidates, policy=selected_policy)
    rejected: dict[str, tuple[str, ...]] = {}
    ranked: list[tuple[tuple[object, ...], Mapping[str, object], int, float]] = []
    preferred_names = {
        str(value).strip() for value in normalized_request.get("preferred_targets", ())
    }
    preference = str(normalized_request.get("execution_preference", "auto")).casefold()
    required_target = str(normalized_request.get("required_target", "")).strip()
    if preference not in {"auto", "local", "remote"}:
        raise ValueError("execution_preference must be auto, local or remote")

    for target in candidates:
        name = str(target.get("name", "")).strip()
        issues = _target_issues(target, normalized_request)
        kind = str(target.get("kind", "local")).strip().casefold()
        if required_target and name != required_target:
            issues.append(f"request requires execution target {required_target}")
        if preference in {"local", "remote"} and kind != preference:
            issues.append(f"request requires {preference} execution")
        if kind == "remote" and not str(target.get("host", "")).strip():
            issues.append("remote target has no transport host")
        active = tuple(
            allocation
            for allocation in allocations
            if allocation.get("target") == name
            and allocation.get("status") in {"allocated", "running"}
        )
        used_cpu = sum(int(item.get("cpu_cores", 0)) for item in active)
        used_memory = sum(float(item.get("memory_gb", 0.0)) for item in active)
        used_gpu = sum(int(item.get("gpu_count", 0)) for item in active)
        used_neural = sum(int(item.get("neural_engine_count", 0)) for item in active)
        if len(active) >= int(target.get("max_jobs", 1)):
            issues.append("concurrent-job capacity is exhausted")
        if used_cpu + int(normalized_request["cpu_cores"]) > int(
            target.get("cpu_cores", 1)
        ):
            issues.append("CPU capacity is insufficient")
        if used_memory + float(normalized_request["memory_gb"]) > float(
            target.get("memory_gb", 0.1)
        ):
            issues.append("memory capacity is insufficient")
        if used_gpu + int(normalized_request.get("gpu_count", 0)) > int(
            target.get("gpu_count", 0)
        ):
            issues.append("GPU capacity is insufficient")
        if used_neural + int(
            normalized_request.get("neural_engine_count", 0)
        ) > int(target.get("neural_engine_count", 0)):
            issues.append("Neural Engine capacity is insufficient")
        if issues:
            rejected[name or "<unnamed>"] = tuple(issues)
            continue

        preferred_rank = 0 if name in preferred_names else 1
        if workload == "heavy" and selected_policy.prefer_remote_for_heavy:
            placement_rank = 0 if kind == "remote" else 1
        elif workload in {"interactive", "standard"} and (
            workload == "interactive" or selected_policy.prefer_local_for_standard
        ):
            placement_rank = 0 if kind == "local" else 1
        else:
            placement_rank = 0
        cpu_capacity = max(1, int(target.get("cpu_cores", 1)))
        memory_capacity = max(0.1, float(target.get("memory_gb", 0.1)))
        utilization = max(used_cpu / cpu_capacity, used_memory / memory_capacity)
        score = (placement_rank, preferred_rank, len(active), utilization, name)
        ranked.append((score, target, len(active), utilization))

    if not ranked:
        return ExecutionRoutingDecision(
            None,
            workload,
            "no eligible execution target satisfies the declared contract",
            rejected,
        )
    _score, selected, active_count, utilization = min(ranked, key=lambda item: item[0])
    kind = str(selected.get("kind", "local"))
    reason = (
        f"{workload} workload routed to {kind} target {selected.get('name')}; "
        f"{active_count} active job(s), {utilization:.1%} allocated utilization"
    )
    if workload == "heavy" and kind == "local":
        unavailable_remotes = sorted(
            name
            for name in rejected
            if any(
                target.get("name") == name and target.get("kind") == "remote"
                for target in candidates
            )
        )
        if unavailable_remotes:
            reason += "; remote candidates unavailable: " + ", ".join(
                unavailable_remotes
            )
    return ExecutionRoutingDecision(selected, workload, reason, rejected)


def execution_routing_schema_path() -> Path:
    return Path(__file__).resolve().parent / "schemas" / "execution-routing-v1.schema.json"


def _target_issues(
    target: Mapping[str, object],
    request: Mapping[str, object],
) -> list[str]:
    issues: list[str] = []
    if not bool(target.get("reachable", False)):
        issues.append(str(target.get("error") or "target is unreachable"))
    if not bool(target.get("qualified", target.get("reachable", False))):
        issues.extend(str(value) for value in target.get("qualification_issues", ()))
        if not target.get("qualification_issues"):
            issues.append("target is not qualified")
    snapshot_payload = target.get("snapshot")
    if isinstance(snapshot_payload, Mapping):
        qualification = qualify_host_snapshot(
            HostCapabilitySnapshot.from_dict(snapshot_payload),
            expected_commit=str(target.get("qualified_against_commit", "")),
            expected_python_abi=str(target.get("qualified_against_python_abi", "")),
            expected_operating_system=str(
                target.get("qualified_against_operating_system", "")
            ),
            expected_architecture=str(target.get("qualified_against_architecture", "")),
            required_packages=request.get("required_packages", ()),
            required_capabilities=request.get("required_capabilities", ()),
            required_programs=request.get("required_programs", ()),
            required_native_backends=request.get("required_native_backends", ()),
            required_systems=request.get("required_systems", ()),
            required_architectures=request.get("required_architectures", ()),
            cpu_cores=int(request.get("cpu_cores", 1)),
            memory_gb=float(request.get("memory_gb", 0.1)),
            gpu_count=int(request.get("gpu_count", 0)),
            neural_engine_count=int(request.get("neural_engine_count", 0)),
            require_clean=str(target.get("kind", "local")) == "remote",
        )
        issues.extend(qualification.issues)
    else:
        requirements = (
            ("required_packages", "packages"),
            ("required_capabilities", "capabilities"),
            ("required_programs", "programs"),
            ("required_native_backends", "native_backends"),
        )
        for request_key, target_key in requirements:
            missing = set(request.get(request_key, ())) - set(target.get(target_key, ()))
            if missing:
                issues.append(f"missing {target_key}: {', '.join(sorted(missing))}")
        required_systems = {
            normalize_operating_system(str(value))
            for value in request.get("required_systems", ())
        }
        target_system = normalize_operating_system(str(target.get("operating_system", "")))
        if required_systems and target_system not in required_systems:
            issues.append(f"operating system {target_system or 'unknown'} is not supported")
        required_architectures = {
            normalize_architecture(str(value))
            for value in request.get("required_architectures", ())
        }
        target_architecture = normalize_architecture(str(target.get("architecture", "")))
        if required_architectures and target_architecture not in required_architectures:
            issues.append(
                f"architecture {target_architecture or 'unknown'} is not supported"
            )
    return list(dict.fromkeys(issues))


__all__ = [
    "EXECUTION_ROUTING_SCHEMA",
    "WORKLOAD_CLASSES",
    "ExecutionRoutingDecision",
    "ExecutionRoutingPolicy",
    "classify_workload",
    "execution_routing_schema_path",
    "route_host",
]
