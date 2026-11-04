"""Architecture-neutral content addressing and scientific-state cache guards."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .atomic_io import atomic_json_write


SCIENTIFIC_STATE_SCHEMA = "matrix.core.scientific_state.v1"
SCIENTIFIC_CACHE_ENVELOPE_SCHEMA = "matrix.core.scientific_cache_envelope.v1"


def canonical_sha256(payload: Any) -> str:
    """Hash one strict, architecture-neutral JSON representation."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def content_key(payload: Any) -> str:
    """Backward-compatible name for the canonical content digest."""

    return canonical_sha256(payload)


@dataclass(frozen=True)
class ScientificStateManifest:
    """One stage identity whose hash invalidates every dependent cache."""

    stage: str
    inputs: Mapping[str, str]
    contracts: Mapping[str, str]
    parameters: Mapping[str, Any] = field(default_factory=dict)
    upstream_states: Mapping[str, str] = field(default_factory=dict)
    schema: str = SCIENTIFIC_STATE_SCHEMA

    def __post_init__(self) -> None:
        stage = str(self.stage).strip()
        inputs = _digest_mapping(self.inputs, "input")
        contracts = _digest_mapping(self.contracts, "contract")
        upstream = _digest_mapping(self.upstream_states, "upstream state")
        parameters = _strict_json_object(self.parameters)
        if not stage:
            raise ValueError("scientific-state stage cannot be empty")
        if not inputs and not upstream:
            raise ValueError("scientific state needs an input or upstream-state digest")
        if not contracts:
            raise ValueError("scientific state needs at least one contract digest")
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "contracts", contracts)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "upstream_states", upstream)

    @property
    def state_sha256(self) -> str:
        return canonical_sha256(self._identity_record())

    def _identity_record(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "stage": self.stage,
            "inputs": dict(self.inputs),
            "contracts": dict(self.contracts),
            "parameters": dict(self.parameters),
            "upstream_states": dict(self.upstream_states),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_record(), "state_sha256": self.state_sha256}

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "ScientificStateManifest":
        if record.get("schema") != SCIENTIFIC_STATE_SCHEMA:
            raise ValueError("unsupported scientific-state schema")
        manifest = cls(
            stage=str(record["stage"]),
            inputs=dict(record.get("inputs", {})),
            contracts=dict(record.get("contracts", {})),
            parameters=dict(record.get("parameters", {})),
            upstream_states=dict(record.get("upstream_states", {})),
        )
        expected = str(record.get("state_sha256", ""))
        if expected and expected != manifest.state_sha256:
            raise ValueError("scientific-state digest does not match its payload")
        return manifest


def scientific_state_key(
    stage: str,
    *,
    inputs: Mapping[str, str],
    contracts: Mapping[str, str],
    parameters: Mapping[str, Any] | None = None,
    upstream_states: Mapping[str, str] | None = None,
) -> str:
    """Return the definitive cache key for one scientific stage."""

    return ScientificStateManifest(
        stage=stage,
        inputs=inputs,
        contracts=contracts,
        parameters=dict(parameters or {}),
        upstream_states=dict(upstream_states or {}),
    ).state_sha256


def scientific_cache_envelope(
    payload: Mapping[str, Any],
    *,
    state_sha256: str,
) -> dict[str, Any]:
    """Wrap a JSON cache value with the exact state that produced it."""

    _require_digest(state_sha256, "scientific state")
    return {
        "schema": SCIENTIFIC_CACHE_ENVELOPE_SCHEMA,
        "state_sha256": state_sha256,
        "payload": _strict_json_object(payload),
    }


def unwrap_scientific_cache(
    record: Mapping[str, Any],
    *,
    expected_state_sha256: str,
) -> dict[str, Any] | None:
    """Return a cache payload only when its scientific state is current."""

    _require_digest(expected_state_sha256, "expected scientific state")
    if record.get("schema") != SCIENTIFIC_CACHE_ENVELOPE_SCHEMA:
        return None
    if record.get("state_sha256") != expected_state_sha256:
        return None
    payload = record.get("payload")
    return dict(payload) if isinstance(payload, Mapping) else None


def put_json(root: str | Path, payload: Any) -> Path:
    path = Path(root) / f"{content_key(payload)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(path, payload, allow_nan=False)
    return path


def get_json(root: str | Path, key: str) -> Any:
    return json.loads((Path(root) / f"{key}.json").read_text(encoding="utf-8"))


def _digest_mapping(values: Mapping[str, str], label: str) -> dict[str, str]:
    result = {str(name).strip(): str(value).strip().lower() for name, value in values.items()}
    if any(not name for name in result):
        raise ValueError(f"scientific-state {label} name cannot be empty")
    for digest in result.values():
        _require_digest(digest, label)
    return dict(sorted(result.items()))


def _require_digest(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _strict_json_object(values: Mapping[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(
        dict(values),
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise TypeError("scientific-state parameters must be a JSON object")
    return decoded


__all__ = [
    "SCIENTIFIC_CACHE_ENVELOPE_SCHEMA",
    "SCIENTIFIC_STATE_SCHEMA",
    "ScientificStateManifest",
    "canonical_sha256",
    "content_key",
    "get_json",
    "put_json",
    "scientific_cache_envelope",
    "scientific_state_key",
    "unwrap_scientific_cache",
]
