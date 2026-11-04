"""Canonical cross-owner project state for nano-MATRIX workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
from importlib.resources import files
import json
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from .atomic_io import atomic_json_write
from .workspace import WorkspaceLayout, ensure_workspace

if TYPE_CHECKING:
    from .workflow import WorkflowPlan, WorkflowStep


STATE_CONTRACT_SCHEMA = "matrix.state_contract.v1"
STATE_CONTRACT_FILENAME = "matrix-state.json"
STATE_OWNERS = ("KEYMAKER", "ORACLE", "SMITH", "LINK", "ARCHITECT")
STATE_STATUSES = ("inactive", "pending", "blocked", "ready", "running", "completed", "failed", "cancelled")
ARTIFACT_STATUSES = ("expected", "present", "verified")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class StateArtifact:
    id: str
    schema: str
    owner: str
    producer_checkpoint: str
    required: bool = True
    path: str = ""
    status: str = "expected"
    sha256: str = ""

    def __post_init__(self) -> None:
        if not self.id or not self.schema:
            raise ValueError("state artifacts require id and schema")
        if self.owner not in STATE_OWNERS:
            raise ValueError(f"unknown state owner: {self.owner}")
        if self.status not in ARTIFACT_STATUSES:
            raise ValueError(f"unknown artifact status: {self.status}")
        if self.status == "verified" and not self.sha256:
            raise ValueError("verified artifacts require sha256")


@dataclass(frozen=True)
class StateCheckpoint:
    id: str
    owner: str
    stage: str
    status: str
    depends_on: tuple[str, ...] = ()
    required_artifacts: tuple[str, ...] = ()
    produced_artifacts: tuple[str, ...] = ()
    run_id: str = ""
    attempts: int = 0
    last_error: str = ""

    def __post_init__(self) -> None:
        if not self.id or not self.stage:
            raise ValueError("state checkpoints require id and stage")
        if self.owner not in STATE_OWNERS:
            raise ValueError(f"unknown state owner: {self.owner}")
        if self.status not in STATE_STATUSES[1:]:
            raise ValueError(f"unknown checkpoint status: {self.status}")
        if not self.produced_artifacts:
            raise ValueError(f"checkpoint {self.id} must declare a produced artifact")
        if self.attempts < 0:
            raise ValueError("checkpoint attempts cannot be negative")


@dataclass(frozen=True)
class OwnerState:
    owner: str
    status: str
    checkpoints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.owner not in STATE_OWNERS:
            raise ValueError(f"unknown state owner: {self.owner}")
        if self.status not in STATE_STATUSES:
            raise ValueError(f"unknown owner status: {self.status}")


@dataclass(frozen=True)
class MatrixStateContract:
    project_id: str
    workflow: str
    workflow_schema: str
    status: str
    owners: tuple[OwnerState, ...]
    checkpoints: tuple[StateCheckpoint, ...]
    artifacts: tuple[StateArtifact, ...]
    schema: str = STATE_CONTRACT_SCHEMA
    revision: int = 1
    created_utc: str = field(default_factory=_utc_now)
    updated_utc: str = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        if self.schema != STATE_CONTRACT_SCHEMA:
            raise ValueError(f"unsupported state contract schema: {self.schema}")
        if not self.project_id:
            raise ValueError("state contract requires project_id")
        if self.status not in STATE_STATUSES[1:]:
            raise ValueError(f"unknown project status: {self.status}")
        if self.revision < 1:
            raise ValueError("state contract revision must be positive")
        if tuple(owner.owner for owner in self.owners) != STATE_OWNERS:
            raise ValueError("state contract must contain every owner in canonical order")
        checkpoint_ids = [checkpoint.id for checkpoint in self.checkpoints]
        artifact_ids = [artifact.id for artifact in self.artifacts]
        if len(checkpoint_ids) != len(set(checkpoint_ids)):
            raise ValueError("state checkpoint ids must be unique")
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("state artifact ids must be unique")
        known_checkpoints = set(checkpoint_ids)
        known_artifacts = set(artifact_ids)
        owner_checkpoints: set[str] = set()
        for owner in self.owners:
            if not set(owner.checkpoints).issubset(known_checkpoints):
                raise ValueError(f"owner {owner.owner} references unknown checkpoints")
            for checkpoint_id in owner.checkpoints:
                checkpoint = self.checkpoints[checkpoint_ids.index(checkpoint_id)]
                if checkpoint.owner != owner.owner:
                    raise ValueError(
                        f"checkpoint {checkpoint_id} is assigned to the wrong owner"
                    )
                if checkpoint_id in owner_checkpoints:
                    raise ValueError(f"checkpoint {checkpoint_id} has multiple owners")
                owner_checkpoints.add(checkpoint_id)
        if owner_checkpoints != known_checkpoints:
            raise ValueError("every checkpoint must belong to exactly one owner")
        for checkpoint in self.checkpoints:
            if not set(checkpoint.depends_on).issubset(known_checkpoints):
                raise ValueError(f"checkpoint {checkpoint.id} has unknown dependencies")
            if not set((*checkpoint.required_artifacts, *checkpoint.produced_artifacts)).issubset(known_artifacts):
                raise ValueError(f"checkpoint {checkpoint.id} references unknown artifacts")
        for artifact in self.artifacts:
            if artifact.producer_checkpoint not in known_checkpoints:
                raise ValueError(f"artifact {artifact.id} has an unknown producer")
            producer = self.checkpoints[checkpoint_ids.index(artifact.producer_checkpoint)]
            if artifact.id not in producer.produced_artifacts:
                raise ValueError(
                    f"artifact {artifact.id} is absent from its producer checkpoint"
                )
            if artifact.owner != producer.owner:
                raise ValueError(f"artifact {artifact.id} is assigned to the wrong owner")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for key in ("owners", "checkpoints", "artifacts"):
            payload[key] = list(payload[key])
        for owner in payload["owners"]:
            owner["checkpoints"] = list(owner["checkpoints"])
        for checkpoint in payload["checkpoints"]:
            for key in ("depends_on", "required_artifacts", "produced_artifacts"):
                checkpoint[key] = list(checkpoint[key])
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "MatrixStateContract":
        return cls(
            project_id=str(data["project_id"]),
            workflow=str(data["workflow"]),
            workflow_schema=str(data["workflow_schema"]),
            status=str(data["status"]),
            owners=tuple(OwnerState(**dict(item)) for item in data.get("owners", ())),
            checkpoints=tuple(
                StateCheckpoint(
                    **{
                        **dict(item),
                        "depends_on": tuple(dict(item).get("depends_on", ())),
                        "required_artifacts": tuple(dict(item).get("required_artifacts", ())),
                        "produced_artifacts": tuple(dict(item).get("produced_artifacts", ())),
                    }
                )
                for item in data.get("checkpoints", ())
            ),
            artifacts=tuple(StateArtifact(**dict(item)) for item in data.get("artifacts", ())),
            schema=str(data.get("schema", "")),
            revision=int(data.get("revision", 1)),
            created_utc=str(data.get("created_utc", "")),
            updated_utc=str(data.get("updated_utc", "")),
        )


def _owner_for_step(step: "WorkflowStep") -> str:
    return {
        "oracle": "ORACLE",
        "gicforge": "SMITH",
        "smith": "SMITH",
        "link": "LINK",
        "architect": "ARCHITECT",
    }.get(step.tool, "KEYMAKER")


def _state_status(status: str) -> str:
    return {"awaiting_confirmation": "blocked"}.get(status, status)


def _artifact_path(root: Path, relative: str) -> tuple[str, str, str]:
    if not relative:
        return "", "expected", ""
    path = root / relative
    if not path.is_file():
        return relative, "expected", ""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return relative, "verified", digest


def build_state_contract(
    plan: "WorkflowPlan",
    workspace: Path | WorkspaceLayout,
    *,
    previous: MatrixStateContract | None = None,
) -> MatrixStateContract:
    layout = workspace if isinstance(workspace, WorkspaceLayout) else ensure_workspace(Path(workspace))
    layout.ensure()
    root = layout.root
    artifacts: dict[str, StateArtifact] = {}
    checkpoints: list[StateCheckpoint] = []

    fixed = (
        ("keymaker.project", "matrix.keymaker.project.v1", "matrix-project.json"),
        ("keymaker.workflow", plan.schema, "state/matrix-workflow.json"),
    )
    for artifact_id, artifact_schema, relative in fixed:
        path, status, digest = _artifact_path(root, relative)
        artifacts[artifact_id] = StateArtifact(
            id=artifact_id,
            schema=artifact_schema,
            owner="KEYMAKER",
            producer_checkpoint=artifact_id,
            path=path,
            status=status,
            sha256=digest,
        )
        checkpoints.append(
            StateCheckpoint(
                id=artifact_id,
                owner="KEYMAKER",
                stage=artifact_id.rsplit(".", 1)[-1],
                status="completed" if status == "verified" else "pending",
                produced_artifacts=(artifact_id,),
            )
        )

    producer_steps = {
        artifact_id: step
        for step in plan.steps
        for artifact_id in step.produced_artifacts
    }
    external_artifact_ids = tuple(
        dict.fromkeys(
            artifact_id
            for step in plan.steps
            for artifact_id in step.required_artifacts
            if artifact_id not in producer_steps
        )
    )
    if external_artifact_ids:
        for artifact_id in external_artifact_ids:
            artifacts[artifact_id] = StateArtifact(
                id=artifact_id,
                schema=artifact_id,
                owner="KEYMAKER",
                producer_checkpoint="keymaker.inputs",
                status="present" if artifact_id in plan.observed_artifacts else "expected",
            )
        checkpoints.append(
            StateCheckpoint(
                id="keymaker.inputs",
                owner="KEYMAKER",
                stage="inputs",
                status=(
                    "completed"
                    if set(external_artifact_ids).issubset(plan.observed_artifacts)
                    else "pending"
                ),
                depends_on=("keymaker.project",),
                produced_artifacts=external_artifact_ids,
            )
        )

    for step in plan.steps:
        owner = _owner_for_step(step)
        for artifact_id in (*step.required_artifacts, *step.produced_artifacts):
            if artifact_id in artifacts:
                continue
            producer_step = producer_steps[artifact_id]
            artifacts[artifact_id] = StateArtifact(
                id=artifact_id,
                schema=artifact_id,
                owner=_owner_for_step(producer_step),
                producer_checkpoint=producer_step.id,
                status="present" if artifact_id in plan.observed_artifacts else "expected",
            )
        produced = tuple(step.produced_artifacts) or (f"matrix.checkpoint.{step.id}.v1",)
        for artifact_id in produced:
            artifacts.setdefault(
                artifact_id,
                StateArtifact(
                    id=artifact_id,
                    schema=artifact_id,
                    owner=owner,
                    producer_checkpoint=step.id,
                ),
            )
        checkpoints.append(
            StateCheckpoint(
                id=step.id,
                owner=owner,
                stage=step.action,
                status=_state_status(step.status),
                depends_on=tuple(step.depends_on),
                required_artifacts=tuple(step.required_artifacts),
                produced_artifacts=produced,
                run_id=step.run_id,
                attempts=step.attempts,
                last_error=step.last_error,
            )
        )

    owners = tuple(
        OwnerState(
            owner=owner,
            checkpoints=tuple(checkpoint.id for checkpoint in checkpoints if checkpoint.owner == owner),
            status=_aggregate_status(
                tuple(checkpoint.status for checkpoint in checkpoints if checkpoint.owner == owner)
            ),
        )
        for owner in STATE_OWNERS
    )
    return MatrixStateContract(
        project_id=plan.project_id,
        workflow="state/matrix-workflow.json",
        workflow_schema=plan.schema,
        status=_state_status(plan.status),
        owners=owners,
        checkpoints=tuple(checkpoints),
        artifacts=tuple(artifacts.values()),
        revision=1 if previous is None else previous.revision + 1,
        created_utc=_utc_now() if previous is None else previous.created_utc,
        updated_utc=_utc_now(),
    )


def _aggregate_status(statuses: tuple[str, ...]) -> str:
    if not statuses:
        return "inactive"
    for status in ("failed", "running", "blocked", "ready", "pending", "cancelled"):
        if status in statuses:
            return status
    return "completed"


def state_contract_path(workspace: Path | WorkspaceLayout) -> Path:
    layout = workspace if isinstance(workspace, WorkspaceLayout) else ensure_workspace(Path(workspace))
    layout.ensure()
    return layout.state / STATE_CONTRACT_FILENAME


def state_contract_schema_path() -> Path:
    return Path(str(files("matrix_core").joinpath("schemas", "state-contract-v1.schema.json")))


def write_state_contract(contract: MatrixStateContract, workspace: Path | WorkspaceLayout) -> Path:
    path = state_contract_path(workspace)
    atomic_json_write(path, contract.to_dict())
    return path


def read_state_contract(workspace: Path | WorkspaceLayout) -> MatrixStateContract:
    return MatrixStateContract.from_dict(
        json.loads(state_contract_path(workspace).read_text(encoding="utf-8"))
    )


def sync_state_contract(plan: "WorkflowPlan", workspace: Path | WorkspaceLayout) -> Path:
    path = state_contract_path(workspace)
    previous = read_state_contract(workspace) if path.is_file() else None
    return write_state_contract(build_state_contract(plan, workspace, previous=previous), workspace)
