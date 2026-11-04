"""Tamper-evident, append-only provenance for Keymaker projects.

The ledger records orchestration decisions and references scientific run
manifests.  Numerical payloads remain in their native artifacts; this module
stores paths, schemas, identifiers and checksums only.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator, Mapping

from .manifest import sha256_file
from .workspace import WorkspaceLayout


PROVENANCE_SCHEMA = "matrix.keymaker.provenance.event.v1"
PROVENANCE_FILENAME = "keymaker-provenance.jsonl"
PROVENANCE_EVENT_TYPES = frozenset(
    {
        "project.created",
        "intent.compiled",
        "workflow.planned",
        "environment.recorded",
        "qm.resolved",
        "authorization.requested",
        "authorization.granted",
        "authorization.denied",
        "job.queued",
        "job.running",
        "job.completed",
        "job.failed",
        "job.cancelled",
        "job.restarted",
        "artifact.registered",
        "workflow.completed",
        "workflow.failed",
    }
)


def provenance_schema_path() -> Path:
    return Path(__file__).resolve().parent / "schemas" / "provenance-event-v1.schema.json"


@dataclass(frozen=True)
class ProvenanceEvent:
    sequence: int
    event_id: str
    event_type: str
    timestamp_utc: str
    project_id: str
    actor: str
    subject_id: str
    payload: dict[str, Any]
    previous_hash: str
    event_hash: str
    schema: str = PROVENANCE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProvenanceEvent":
        return cls(
            sequence=int(data["sequence"]),
            event_id=str(data["event_id"]),
            event_type=str(data["event_type"]),
            timestamp_utc=str(data["timestamp_utc"]),
            project_id=str(data["project_id"]),
            actor=str(data["actor"]),
            subject_id=str(data.get("subject_id", "")),
            payload=dict(data.get("payload", {})),
            previous_hash=str(data.get("previous_hash", "")),
            event_hash=str(data["event_hash"]),
            schema=str(data.get("schema", PROVENANCE_SCHEMA)),
        )


@dataclass(frozen=True)
class ProvenanceVerification:
    valid: bool
    event_count: int
    last_hash: str
    errors: tuple[str, ...] = ()


class ProvenanceLedger:
    """Append and verify events in one project-local JSONL hash chain."""

    def __init__(self, workspace: WorkspaceLayout | Path | str, *, project_id: str = ""):
        if isinstance(workspace, WorkspaceLayout):
            self.workspace = workspace.ensure()
        else:
            self.workspace = WorkspaceLayout(Path(workspace).expanduser().resolve()).ensure()
        self.path = self.workspace.state / PROVENANCE_FILENAME
        self.lock_path = self.workspace.state / f".{PROVENANCE_FILENAME}.lock"
        self.project_id = project_id.strip() or _project_id(self.workspace.root)

    def append(
        self,
        event_type: str,
        *,
        actor: str = "matrix",
        subject_id: str = "",
        payload: Mapping[str, Any] | None = None,
        timestamp_utc: str | None = None,
    ) -> ProvenanceEvent:
        if event_type not in PROVENANCE_EVENT_TYPES:
            raise ValueError(f"unsupported provenance event type: {event_type}")
        if not self.project_id:
            raise ValueError("provenance project_id must not be empty")
        if not str(actor).strip():
            raise ValueError("provenance actor must not be empty")
        safe_payload = _json_object(payload or {})
        with self._locked():
            events = self._read_unlocked()
            sequence = len(events) + 1
            previous_hash = events[-1].event_hash if events else ""
            event_data: dict[str, Any] = {
                "sequence": sequence,
                "event_id": f"evt-{sequence:08d}",
                "event_type": event_type,
                "timestamp_utc": timestamp_utc or datetime.now(timezone.utc).isoformat(),
                "project_id": self.project_id,
                "actor": str(actor),
                "subject_id": str(subject_id),
                "payload": safe_payload,
                "previous_hash": previous_hash,
                "schema": PROVENANCE_SCHEMA,
            }
            event_data["event_hash"] = _event_hash(event_data)
            event = ProvenanceEvent.from_dict(event_data)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event.to_dict(), sort_keys=True, separators=(",", ":")))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            return event

    def register_artifact(
        self,
        path: Path | str,
        *,
        role: str,
        artifact_schema: str = "",
        subject_id: str = "",
        actor: str = "matrix",
    ) -> ProvenanceEvent:
        target = Path(path).expanduser().resolve()
        if not target.is_file():
            raise FileNotFoundError(target)
        return self.append(
            "artifact.registered",
            actor=actor,
            subject_id=subject_id or target.name,
            payload={
                "artifact": artifact_reference(
                    target,
                    workspace=self.workspace,
                    role=role,
                    artifact_schema=artifact_schema,
                )
            },
        )

    def events(self) -> tuple[ProvenanceEvent, ...]:
        with self._locked():
            return tuple(self._read_unlocked())

    def verify(self) -> ProvenanceVerification:
        errors: list[str] = []
        try:
            events = self.events()
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return ProvenanceVerification(False, 0, "", (f"invalid JSONL: {exc}",))
        if not events:
            return ProvenanceVerification(False, 0, "", ("provenance ledger is empty",))
        previous_hash = ""
        expected_project_id = events[0].project_id if events else self.project_id
        for expected_sequence, event in enumerate(events, start=1):
            if event.schema != PROVENANCE_SCHEMA:
                errors.append(f"event {expected_sequence}: unsupported schema {event.schema}")
            if event.event_type not in PROVENANCE_EVENT_TYPES:
                errors.append(f"event {expected_sequence}: unsupported event_type")
            if event.project_id != expected_project_id:
                errors.append(f"event {expected_sequence}: inconsistent project_id")
            if event.sequence != expected_sequence:
                errors.append(f"event {expected_sequence}: sequence is {event.sequence}")
            if event.event_id != f"evt-{expected_sequence:08d}":
                errors.append(f"event {expected_sequence}: inconsistent event_id")
            if event.previous_hash != previous_hash:
                errors.append(f"event {expected_sequence}: broken previous_hash")
            expected_hash = _event_hash(event.to_dict())
            if event.event_hash != expected_hash:
                errors.append(f"event {expected_sequence}: invalid event_hash")
            previous_hash = event.event_hash
        return ProvenanceVerification(not errors, len(events), previous_hash, tuple(errors))

    def _read_unlocked(self) -> list[ProvenanceEvent]:
        if not self.path.is_file():
            return []
        events: list[ProvenanceEvent] = []
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                events.append(ProvenanceEvent.from_dict(json.loads(line)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid provenance event on line {line_number}: {exc}") from exc
        return events

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def provenance_path(workspace: WorkspaceLayout | Path | str) -> Path:
    if isinstance(workspace, WorkspaceLayout):
        return workspace.state / PROVENANCE_FILENAME
    return Path(workspace).expanduser().resolve() / "state" / PROVENANCE_FILENAME


def artifact_reference(
    path: Path | str,
    *,
    workspace: WorkspaceLayout,
    role: str,
    artifact_schema: str = "",
) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    try:
        display_path = str(target.relative_to(workspace.root.resolve()))
    except ValueError:
        display_path = str(target)
    return {
        "path": display_path,
        "role": str(role),
        "schema": str(artifact_schema),
        "sha256": sha256_file(target),
        "size_bytes": target.stat().st_size,
    }


def _event_hash(data: Mapping[str, Any]) -> str:
    body = {key: value for key, value in data.items() if key != "event_hash"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _json_object(payload: Mapping[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(dict(payload), sort_keys=True, ensure_ascii=False, allow_nan=False)
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise TypeError("provenance payload must be a JSON object")
    return decoded


def _project_id(root: Path) -> str:
    manifest = root / "matrix-project.json"
    if manifest.is_file():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            project_id = str(data.get("project_id", "")).strip()
            if project_id:
                return project_id
            project = data.get("project", {})
            if isinstance(project, dict):
                name = str(project.get("name", "")).strip()
                if name:
                    return name
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    return root.name
