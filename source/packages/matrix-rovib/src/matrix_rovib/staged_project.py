"""Content-bound LOCK projects spanning exploration and quantitative refinement."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


LOCK_STAGED_PROJECT_SCHEMA = "matrix.lock.staged_project.v1"
LOCK_STAGE_SEQUENCE = ("explore", "refine", "validate")


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def create_staged_project(
    *,
    name: str,
    molecular_identity: Mapping[str, Any],
    artifact_chain_sha256: str | None = None,
) -> dict[str, Any]:
    """Create a molecule-independent project whose scientific state is immutable."""

    if not name.strip() or not molecular_identity:
        raise ValueError("a staged LOCK project requires a name and molecular identity")
    project = {
        "schema": LOCK_STAGED_PROJECT_SCHEMA,
        "name": name.strip(),
        "molecular_identity": json.loads(json.dumps(dict(molecular_identity), allow_nan=False)),
        "artifact_chain_sha256": artifact_chain_sha256,
        "current_stage": "explore",
        "state_registry_sha256": None,
        "parameter_registry_sha256": None,
        "assignment_registry_sha256": None,
        "stages": [],
        "promotions": [],
    }
    project["project_identity_sha256"] = _digest({
        "name": project["name"],
        "molecular_identity": project["molecular_identity"],
        "artifact_chain_sha256": artifact_chain_sha256,
    })
    return project


def record_stage_result(
    project: Mapping[str, Any],
    *,
    stage: str,
    artifacts: Mapping[str, Any],
    evidence: Mapping[str, Any],
    state_registry: Mapping[str, Any],
    parameter_registry: Mapping[str, Any],
    assignment_registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach a completed stage while freezing all cross-stage registries."""

    result = _validated_copy(project)
    if stage not in LOCK_STAGE_SEQUENCE or stage != result["current_stage"]:
        raise ValueError(f"stage result must match current stage {result['current_stage']!r}")
    if not artifacts or not evidence or not state_registry or not parameter_registry:
        raise ValueError("stage results require artifacts, evidence and state/parameter registries")
    registries = {
        "state_registry_sha256": _digest(state_registry),
        "parameter_registry_sha256": _digest(parameter_registry),
        "assignment_registry_sha256": (
            None if assignment_registry is None else _digest(assignment_registry)
        ),
    }
    if result["stages"]:
        previous_record = result["stages"][-1]
        if previous_record.get("state_registry") != state_registry:
            raise ValueError("state registry changed across LOCK stages")
        for name, current in (
            ("parameter_registry", parameter_registry),
            ("assignment_registry", assignment_registry),
        ):
            previous = previous_record.get(name)
            if previous is not None and current is not None and not _registry_extends(
                previous, current
            ):
                raise ValueError(
                    f"{name} changed existing identities instead of extending the registry"
                )
    result.update({key: value or result.get(key) for key, value in registries.items()})
    record = {
        "stage": stage,
        "artifacts": json.loads(json.dumps(dict(artifacts), allow_nan=False)),
        "evidence": json.loads(json.dumps(dict(evidence), allow_nan=False)),
        "state_registry": json.loads(json.dumps(dict(state_registry), allow_nan=False)),
        "parameter_registry": json.loads(json.dumps(dict(parameter_registry), allow_nan=False)),
        "assignment_registry": (
            None if assignment_registry is None
            else json.loads(json.dumps(dict(assignment_registry), allow_nan=False))
        ),
        **registries,
    }
    record["sha256"] = _digest(record)
    result["stages"].append(record)
    return result


def promotion_assessment(
    project: Mapping[str, Any],
    *,
    target_stage: str,
) -> dict[str, Any]:
    """Evaluate objective gates before exploration is promoted to a fit."""

    validated = _validated_copy(project)
    source = validated["current_stage"]
    expected_index = LOCK_STAGE_SEQUENCE.index(source) + 1
    if expected_index >= len(LOCK_STAGE_SEQUENCE) or LOCK_STAGE_SEQUENCE[expected_index] != target_stage:
        raise ValueError(f"cannot promote LOCK project from {source!r} to {target_stage!r}")
    completed = [item for item in validated["stages"] if item["stage"] == source]
    if not completed:
        raise ValueError(f"stage {source!r} has no recorded result")
    evidence = completed[-1]["evidence"]
    if source == "explore":
        checks = {
            "coordinate_model_selected": bool(evidence.get("coordinate_model_selected")),
            "symmetry_complete": bool(evidence.get("symmetry_complete")),
            "state_registry_stable": bool(evidence.get("state_registry_stable")),
            "surface_qualified": bool(evidence.get("surface_qualified")),
            "parameter_mapping_complete": bool(evidence.get("parameter_mapping_complete")),
        }
    else:
        checks = {
            "fit_converged": bool(evidence.get("fit_converged")),
            "full_parameter_rank": bool(evidence.get("full_parameter_rank")),
            "numerical_convergence": bool(evidence.get("numerical_convergence")),
            "holdout_defined": bool(evidence.get("holdout_defined")),
            "physical_guards_passed": bool(evidence.get("physical_guards_passed")),
        }
    blockers = [name for name, passed in checks.items() if not passed]
    return {
        "source_stage": source,
        "target_stage": target_stage,
        "checks": checks,
        "ready": not blockers,
        "blockers": blockers,
        "source_result_sha256": completed[-1]["sha256"],
    }


def promote_staged_project(
    project: Mapping[str, Any],
    *,
    target_stage: str,
) -> dict[str, Any]:
    """Promote a project only after its stage-specific evidence gates pass."""

    result = _validated_copy(project)
    assessment = promotion_assessment(result, target_stage=target_stage)
    if not assessment["ready"]:
        raise ValueError(
            "LOCK stage promotion is blocked by: " + ", ".join(assessment["blockers"])
        )
    promotion = {
        **assessment,
        "registry_binding": {
            key: result.get(key)
            for key in (
                "state_registry_sha256",
                "parameter_registry_sha256",
                "assignment_registry_sha256",
            )
        },
    }
    promotion["sha256"] = _digest(promotion)
    result["promotions"].append(promotion)
    result["current_stage"] = target_stage
    return result


def write_staged_project(path: Path | str, project: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _validated_copy(project)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def read_staged_project(path: Path | str) -> dict[str, Any]:
    return _validated_copy(json.loads(Path(path).read_text(encoding="utf-8")))


def _validated_copy(project: Mapping[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(dict(project), allow_nan=False))
    if result.get("schema") != LOCK_STAGED_PROJECT_SCHEMA:
        raise ValueError("unsupported LOCK staged-project schema")
    if result.get("current_stage") not in LOCK_STAGE_SEQUENCE:
        raise ValueError("invalid LOCK project stage")
    if not isinstance(result.get("stages"), list) or not isinstance(
        result.get("promotions"), list
    ):
        raise ValueError("LOCK staged-project history is invalid")
    return result


def _registry_extends(previous: object, current: object) -> bool:
    if isinstance(previous, dict) and isinstance(current, dict):
        return all(key in current and _registry_extends(value, current[key])
                   for key, value in previous.items())
    if isinstance(previous, list) and isinstance(current, list):
        return len(current) >= len(previous) and all(
            _registry_extends(left, right)
            for left, right in zip(previous, current, strict=False)
        )
    return previous == current


__all__ = [
    "LOCK_STAGED_PROJECT_SCHEMA",
    "LOCK_STAGE_SEQUENCE",
    "create_staged_project",
    "promotion_assessment",
    "promote_staged_project",
    "read_staged_project",
    "record_stage_result",
    "write_staged_project",
]
