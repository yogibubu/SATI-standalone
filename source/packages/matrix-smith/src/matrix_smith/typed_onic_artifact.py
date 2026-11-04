"""Self-contained serialized artifact for composite typed ONIC charts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from matrix_core import read_sectioned_lines, replace_section, section_content

from .definition import sonic_definition_identity_sha256
from .gic_serialization import gic_definition_from_dict, gic_definition_to_dict
from .models import GICDefinition
from .onic_blocks import (
    CompositeOnicDefinition,
    OnicBlockContractError,
    composite_onic_definition_from_dict,
    composite_onic_definition_identity_sha256,
    composite_onic_definition_to_dict,
    read_onic_block_contract_from_xyzin,
    write_onic_block_contract,
)
from .payload_codec import (
    BASE64_CANONICAL_JSON_ENCODING,
    decode_canonical_json_lines,
    encode_canonical_json_lines,
    is_payload_subsection_header,
)
from .relative_pose_blocks import relative_pose_payload_identity_sha256


TYPED_ONIC_ARTIFACT_SCHEMA = "matrix.smith.typed_onic_artifact.v1"
TYPED_ONIC_ARTIFACT_SECTION = "TYPED_ONIC"


@dataclass(frozen=True)
class TypedOnicArtifact:
    """One composite contract plus every delegated frozen GIC payload."""

    definition: CompositeOnicDefinition
    payloads: tuple[tuple[str, GICDefinition], ...]
    schema: str = TYPED_ONIC_ARTIFACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TYPED_ONIC_ARTIFACT_SCHEMA:
            raise OnicBlockContractError(f"unsupported typed ONIC artifact schema: {self.schema}")
        if not isinstance(self.definition, CompositeOnicDefinition):
            raise TypeError("typed ONIC artifact requires a composite definition")
        records = tuple((str(identifier), payload) for identifier, payload in self.payloads)
        identifiers = tuple(identifier for identifier, _payload in records)
        if len(set(identifiers)) != len(identifiers):
            raise OnicBlockContractError("typed ONIC artifact contains duplicate payload ids")
        by_block = {block.identifier: block for block in self.definition.blocks}
        required_order = tuple(
            block.identifier
            for block in self.definition.blocks
            if block.representation
            in {"NATURAL_INTERNAL", "EXPONENTIAL_MAP", "PSEUDO_BOND_CONTACT"}
        )
        required = set(required_order)
        if set(identifiers) != required:
            missing = sorted(required - set(identifiers))
            extra = sorted(set(identifiers) - required)
            raise OnicBlockContractError(
                f"typed ONIC artifact payload mismatch: missing={missing}, extra={extra}"
            )
        payload_by_id = dict(records)
        records = tuple((identifier, payload_by_id[identifier]) for identifier in required_order)
        for identifier, payload in records:
            if not isinstance(payload, GICDefinition):
                raise TypeError(f"typed ONIC payload {identifier} is not a GICDefinition")
            block = by_block[identifier]
            actual_identity = (
                relative_pose_payload_identity_sha256(payload)
                if block.representation == "EXPONENTIAL_MAP"
                else sonic_definition_identity_sha256(payload)
            )
            if block.payload_schema != payload.contract_schema_version:
                raise OnicBlockContractError(
                    f"typed ONIC payload schema mismatch for block {identifier}"
                )
            if block.payload_identity_sha256 != actual_identity:
                raise OnicBlockContractError(
                    f"typed ONIC payload checksum mismatch for block {identifier}"
                )
        object.__setattr__(self, "payloads", records)

    @property
    def payload_by_id(self) -> dict[str, GICDefinition]:
        return dict(self.payloads)


def typed_onic_artifact_to_dict(artifact: TypedOnicArtifact) -> dict[str, Any]:
    return {
        "schema": artifact.schema,
        "composite_identity_sha256": composite_onic_definition_identity_sha256(
            artifact.definition
        ),
        "definition": composite_onic_definition_to_dict(artifact.definition),
        "payloads": [
            {
                "block_identifier": identifier,
                "payload": gic_definition_to_dict(payload),
            }
            for identifier, payload in artifact.payloads
        ],
    }


def typed_onic_artifact_from_dict(payload: Mapping[str, Any]) -> TypedOnicArtifact:
    if not isinstance(payload, Mapping) or payload.get("schema") != TYPED_ONIC_ARTIFACT_SCHEMA:
        raise OnicBlockContractError("unsupported typed ONIC artifact schema")
    try:
        definition = composite_onic_definition_from_dict(payload["definition"])
        artifact = TypedOnicArtifact(
            definition=definition,
            payloads=tuple(
                (
                    str(item["block_identifier"]),
                    gic_definition_from_dict(item["payload"]),
                )
                for item in payload["payloads"]
            ),
        )
    except OnicBlockContractError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise OnicBlockContractError("typed ONIC artifact is incomplete or malformed") from exc
    declared = str(payload.get("composite_identity_sha256", "")).lower()
    actual = composite_onic_definition_identity_sha256(definition)
    if declared != actual:
        raise OnicBlockContractError("typed ONIC artifact composite checksum does not match")
    return artifact


def typed_onic_artifact_json(artifact: TypedOnicArtifact) -> str:
    return json.dumps(
        typed_onic_artifact_to_dict(artifact),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def typed_onic_artifact_identity_sha256(artifact: TypedOnicArtifact) -> str:
    return hashlib.sha256(typed_onic_artifact_json(artifact).encode("utf-8")).hexdigest()


def write_typed_onic_artifact(
    path: Path | str,
    definition: CompositeOnicDefinition,
    *,
    payloads: Mapping[str, GICDefinition] | Sequence[tuple[str, GICDefinition]],
) -> TypedOnicArtifact:
    """Write a byte-stable, self-contained typed ONIC enriched-XYZ artifact."""

    items = tuple(payloads.items()) if isinstance(payloads, Mapping) else tuple(payloads)
    artifact = TypedOnicArtifact(definition=definition, payloads=items)
    target = Path(path)
    write_onic_block_contract(target, definition)
    serialized = typed_onic_artifact_json(artifact)
    replace_section(
        target,
        TYPED_ONIC_ARTIFACT_SECTION,
        [
            f"SCHEMA {TYPED_ONIC_ARTIFACT_SCHEMA}",
            "STATUS BUILT",
            f"ENCODING {BASE64_CANONICAL_JSON_ENCODING}",
            f"IDENTITY_SHA256 {typed_onic_artifact_identity_sha256(artifact)}",
            "[ARTIFACT_JSON]",
            *encode_canonical_json_lines(serialized),
        ],
    )
    return artifact


def read_typed_onic_artifact_from_xyzin(
    path: Path | str,
    *,
    required: bool = True,
) -> TypedOnicArtifact | None:
    target = Path(path)
    section = section_content(read_sectioned_lines(target), TYPED_ONIC_ARTIFACT_SECTION)
    if not section:
        if required:
            raise OnicBlockContractError(f"missing #{TYPED_ONIC_ARTIFACT_SECTION} section")
        return None
    if section[0].strip() != f"SCHEMA {TYPED_ONIC_ARTIFACT_SCHEMA}":
        raise OnicBlockContractError("unsupported typed ONIC artifact schema")
    if (_section_value(section, "STATUS") or "").upper() != "BUILT":
        raise OnicBlockContractError("typed ONIC artifact is not built")
    artifact_lines = _subsection(section, "ARTIFACT_JSON")
    try:
        serialized = decode_canonical_json_lines(
            artifact_lines,
            encoding=_section_value(section, "ENCODING") or "",
        )
        payload = json.loads(serialized)
    except (json.JSONDecodeError, ValueError) as exc:
        raise OnicBlockContractError("typed ONIC artifact contains invalid JSON") from exc
    artifact = typed_onic_artifact_from_dict(payload)
    declared = str(_section_value(section, "IDENTITY_SHA256") or "").lower()
    if declared != typed_onic_artifact_identity_sha256(artifact):
        raise OnicBlockContractError("typed ONIC artifact identity checksum does not match")
    serialized_definition = read_onic_block_contract_from_xyzin(target, required=True)
    if serialized_definition != artifact.definition:
        raise OnicBlockContractError(
            "typed ONIC artifact and #ONIC_BLOCKS definition do not match"
        )
    return artifact


def _section_value(section: Sequence[str], key: str) -> str | None:
    prefix = f"{key} "
    for line in section:
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return None


def _subsection(section: Sequence[str], name: str) -> list[str]:
    marker = f"[{name}]"
    try:
        start = list(section).index(marker) + 1
    except ValueError:
        return []
    result: list[str] = []
    for line in section[start:]:
        if is_payload_subsection_header(line):
            break
        result.append(line)
    return result


__all__ = [
    "TYPED_ONIC_ARTIFACT_SCHEMA",
    "TYPED_ONIC_ARTIFACT_SECTION",
    "TypedOnicArtifact",
    "read_typed_onic_artifact_from_xyzin",
    "typed_onic_artifact_from_dict",
    "typed_onic_artifact_identity_sha256",
    "typed_onic_artifact_json",
    "typed_onic_artifact_to_dict",
    "write_typed_onic_artifact",
]
