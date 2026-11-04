"""Content-addressed scientific handoffs for the LOCK production chain."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


LOCK_ARTIFACT_CHAIN_SCHEMA = "matrix.lock.artifact_chain.v1"
LOCK_OWNER_SEQUENCE = ("ORACLE", "SMITH", "GF", "TRINITY", "LOCK")


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ScientificArtifact:
    """One immutable, schema-tagged handoff between MATRIX owners."""

    owner: str
    schema: str
    payload: Mapping[str, Any]
    upstream_sha256: str | None = None

    def __post_init__(self) -> None:
        owner = self.owner.strip().upper()
        if owner not in LOCK_OWNER_SEQUENCE:
            raise ValueError(f"unsupported LOCK artifact owner: {self.owner!r}")
        if not self.schema.strip() or not isinstance(self.payload, Mapping):
            raise ValueError("scientific artifacts require a schema and mapping payload")
        if self.upstream_sha256 is not None and (
            len(self.upstream_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.upstream_sha256)
        ):
            raise ValueError("upstream_sha256 must be a lowercase SHA-256 digest")
        object.__setattr__(self, "owner", owner)
        # JSON round-tripping rejects arrays with hidden mutable or nonportable state.
        canonical = json.loads(json.dumps(dict(self.payload), allow_nan=False))
        object.__setattr__(self, "payload", canonical)

    @property
    def sha256(self) -> str:
        return _canonical_digest(self.unsigned_record())

    def unsigned_record(self) -> dict[str, Any]:
        return {
            "owner": self.owner,
            "schema": self.schema,
            "upstream_sha256": self.upstream_sha256,
            "payload": dict(self.payload),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_record(), "sha256": self.sha256}

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "ScientificArtifact":
        if set(record) != {"owner", "schema", "upstream_sha256", "payload", "sha256"}:
            raise ValueError("LOCK artifact envelope has incomplete or unknown fields")
        artifact = cls(
            owner=str(record["owner"]),
            schema=str(record["schema"]),
            upstream_sha256=(
                None if record["upstream_sha256"] is None else str(record["upstream_sha256"])
            ),
            payload=record["payload"],
        )
        if str(record["sha256"]) != artifact.sha256:
            raise ValueError("LOCK artifact content digest does not match its envelope")
        return artifact


def append_artifact(
    chain: Sequence[ScientificArtifact],
    *,
    owner: str,
    schema: str,
    payload: Mapping[str, Any],
) -> tuple[ScientificArtifact, ...]:
    """Append an owner output while binding it to the exact preceding artifact."""

    validated = validate_artifact_chain(chain, require_complete=False)
    upstream = validated[-1].sha256 if validated else None
    artifact = ScientificArtifact(owner, schema, payload, upstream)
    return validate_artifact_chain((*validated, artifact), require_complete=False)


def validate_artifact_chain(
    chain: Sequence[ScientificArtifact | Mapping[str, Any]],
    *,
    require_complete: bool = True,
) -> tuple[ScientificArtifact, ...]:
    """Validate owner order, schemas, content hashes and predecessor links."""

    artifacts = tuple(
        item if isinstance(item, ScientificArtifact) else ScientificArtifact.from_dict(item)
        for item in chain
    )
    if not artifacts:
        if require_complete:
            raise ValueError("a complete LOCK artifact chain cannot be empty")
        return ()
    owners = tuple(item.owner for item in artifacts)
    expected = LOCK_OWNER_SEQUENCE[: len(artifacts)]
    if owners != expected:
        raise ValueError(f"LOCK owner sequence must be {expected}, observed {owners}")
    if require_complete and owners != LOCK_OWNER_SEQUENCE:
        raise ValueError(f"complete LOCK chain requires owners {LOCK_OWNER_SEQUENCE}")
    for index, artifact in enumerate(artifacts):
        expected_upstream = None if index == 0 else artifacts[index - 1].sha256
        if artifact.upstream_sha256 != expected_upstream:
            raise ValueError(f"broken LOCK artifact link at owner {artifact.owner}")
        declared = artifact.payload.get("schema")
        if declared is not None and declared != artifact.schema:
            raise ValueError(f"{artifact.owner} payload schema contradicts its envelope")
    return artifacts


def write_artifact_chain(
    path: Path | str, chain: Sequence[ScientificArtifact | Mapping[str, Any]]
) -> Path:
    artifacts = validate_artifact_chain(chain)
    payload = {
        "schema": LOCK_ARTIFACT_CHAIN_SCHEMA,
        "owners": list(LOCK_OWNER_SEQUENCE),
        "artifacts": [artifact.to_dict() for artifact in artifacts],
        "terminal_sha256": artifacts[-1].sha256,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def read_artifact_chain(path: Path | str) -> tuple[ScientificArtifact, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != LOCK_ARTIFACT_CHAIN_SCHEMA:
        raise ValueError("unsupported LOCK artifact-chain schema")
    artifacts = validate_artifact_chain(payload.get("artifacts", ()))
    if payload.get("owners") != list(LOCK_OWNER_SEQUENCE):
        raise ValueError("LOCK artifact-chain owner declaration changed")
    if payload.get("terminal_sha256") != artifacts[-1].sha256:
        raise ValueError("LOCK artifact-chain terminal digest changed")
    return artifacts


__all__ = [
    "LOCK_ARTIFACT_CHAIN_SCHEMA",
    "LOCK_OWNER_SEQUENCE",
    "ScientificArtifact",
    "append_artifact",
    "read_artifact_chain",
    "validate_artifact_chain",
    "write_artifact_chain",
]
