"""Frozen, auditable source-pool artifact for Hessian-adapted TS SONICs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .evaluation import build_gic_b_matrix
from .gic_serialization import gic_definition_from_dict, gic_definition_to_dict
from .models import GICDefinition
from .policy import TS_REACTIVE_COORDINATE_CLASS, TS_REACTIVE_FAMILY_MIXING_POLICY


REACTIVE_SONIC_POOL_ARTIFACT_SCHEMA = "matrix.smith.reactive_sonic_pool_artifact.v1"
REACTIVE_SONIC_POOL_SOURCES = frozenset(
    {
        "ENDPOINT_CONNECTIVITY_REACTIVE_POOL",
        "MULTITOPOLOGY_REACTIVE_POOL",
    }
)


@dataclass(frozen=True)
class ReactiveSonicPoolArtifact:
    """A redundant reactive pool plus its invariant conserved prefix."""

    definition: GICDefinition
    common_count: int
    source_topology_ids: tuple[str, ...]
    reactive_source_labels: tuple[str, ...]
    reactive_source_topology_memberships: tuple[tuple[str, ...], ...]
    reactive_primitive_ids: tuple[str, ...]
    reactive_primitive_families: tuple[str, ...]
    reactive_coordinate_class: str = TS_REACTIVE_COORDINATE_CLASS
    family_mixing_policy: str = TS_REACTIVE_FAMILY_MIXING_POLICY
    schema: str = REACTIVE_SONIC_POOL_ARTIFACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != REACTIVE_SONIC_POOL_ARTIFACT_SCHEMA:
            raise ValueError(f"unsupported reactive SONIC pool schema: {self.schema}")
        definition = self.definition
        if not isinstance(definition, GICDefinition):
            raise TypeError("reactive SONIC pool requires a GICDefinition")
        count = int(self.common_count)
        if count < 0 or count >= definition.target_rank:
            raise ValueError("reactive SONIC pool has an invalid conserved-prefix size")
        if definition.rank != definition.target_rank:
            raise ValueError("reactive SONIC pool must span the exact molecular rank")
        if len(definition.gics) < definition.target_rank:
            raise ValueError("reactive SONIC pool contains too few coordinate rows")
        if definition.primitive_source not in REACTIVE_SONIC_POOL_SOURCES:
            raise ValueError("reactive SONIC pool records an unsupported primitive source")
        if self.reactive_coordinate_class != TS_REACTIVE_COORDINATE_CLASS:
            raise ValueError("reactive SONIC pool records an unsupported coordinate class")
        if self.family_mixing_policy != TS_REACTIVE_FAMILY_MIXING_POLICY:
            raise ValueError("reactive SONIC pool records an unsupported family-mixing policy")
        if definition.primitive_source == "ENDPOINT_CONNECTIVITY_REACTIVE_POOL" and not any(
            "CONNECTIVITY_CHANGE=YES" in diagnostic
            for diagnostic in definition.semantic_diagnostics
        ):
            raise ValueError(
                "cross-family reactive mixing requires an endpoint connectivity change"
            )
        if definition.primitive_source == "MULTITOPOLOGY_REACTIVE_POOL" and not any(
            "CONNECTIVITY_CHANGE=YES" in diagnostic
            for diagnostic in definition.semantic_diagnostics
        ):
            raise ValueError(
                "cross-family reactive mixing requires a multi-topology connectivity change"
            )
        identifiers = tuple(gic.identifier for gic in definition.gics)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("reactive SONIC pool contains duplicate coordinate identifiers")
        topology_ids = tuple(str(value).strip() for value in self.source_topology_ids)
        if not topology_ids or any(not value for value in topology_ids):
            raise ValueError("reactive SONIC pool requires named source topologies")
        if len(set(topology_ids)) != len(topology_ids):
            raise ValueError("reactive SONIC pool source topologies must be unique")
        labels = tuple(str(value) for value in self.reactive_source_labels)
        expected_labels = tuple(gic.name or gic.identifier for gic in definition.gics[count:])
        if labels != expected_labels:
            raise ValueError("reactive SONIC source labels do not match the frozen pool")
        memberships = tuple(
            tuple(str(value).strip() for value in membership)
            for membership in self.reactive_source_topology_memberships
        )
        if len(memberships) != len(labels) or any(not membership for membership in memberships):
            raise ValueError("every reactive SONIC source requires topology provenance")
        topology_set = frozenset(topology_ids)
        if any(
            len(set(membership)) != len(membership)
            or any(not value or value not in topology_set for value in membership)
            for membership in memberships
        ):
            raise ValueError("reactive SONIC source topology provenance is invalid")
        expected_primitive_ids, expected_families = _reactive_region_primitives(
            definition,
            count,
        )
        if tuple(self.reactive_primitive_ids) != expected_primitive_ids:
            raise ValueError("reactive SONIC primitive IDs do not match the frozen region")
        if tuple(self.reactive_primitive_families) != expected_families:
            raise ValueError("reactive SONIC primitive families do not match the frozen region")
        _validate_pool_tangent(definition, count)
        object.__setattr__(self, "common_count", count)
        object.__setattr__(self, "source_topology_ids", topology_ids)
        object.__setattr__(self, "reactive_source_labels", labels)
        object.__setattr__(self, "reactive_source_topology_memberships", memberships)


def reactive_sonic_pool_artifact_to_dict(
    artifact: ReactiveSonicPoolArtifact,
) -> dict[str, Any]:
    payload = _reactive_sonic_pool_payload(artifact)
    payload["identity_sha256"] = _canonical_sha256(payload)
    return payload


def reactive_sonic_pool_artifact_from_dict(
    payload: Mapping[str, Any],
) -> ReactiveSonicPoolArtifact:
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != REACTIVE_SONIC_POOL_ARTIFACT_SCHEMA
    ):
        raise ValueError("unsupported reactive SONIC pool artifact schema")
    declared = str(payload.get("identity_sha256", "")).lower()
    unsigned = dict(payload)
    unsigned.pop("identity_sha256", None)
    if declared != _canonical_sha256(unsigned):
        raise ValueError("reactive SONIC pool artifact checksum does not match")
    try:
        return ReactiveSonicPoolArtifact(
            definition=gic_definition_from_dict(payload["definition"]),
            common_count=int(payload["common_count"]),
            source_topology_ids=tuple(str(value) for value in payload["source_topology_ids"]),
            reactive_source_labels=tuple(str(value) for value in payload["reactive_source_labels"]),
            reactive_source_topology_memberships=tuple(
                tuple(str(value) for value in membership)
                for membership in payload["reactive_source_topology_memberships"]
            ),
            reactive_primitive_ids=tuple(str(value) for value in payload["reactive_primitive_ids"]),
            reactive_primitive_families=tuple(
                str(value) for value in payload["reactive_primitive_families"]
            ),
            reactive_coordinate_class=str(payload["reactive_coordinate_class"]),
            family_mixing_policy=str(payload["family_mixing_policy"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("reactive SONIC pool artifact is incomplete or malformed") from exc


def reactive_sonic_pool_artifact_json(artifact: ReactiveSonicPoolArtifact) -> str:
    return json.dumps(
        reactive_sonic_pool_artifact_to_dict(artifact),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def write_reactive_sonic_pool_artifact(
    path: Path | str,
    definition: GICDefinition,
    *,
    common_count: int,
    source_topology_ids: tuple[str, ...],
    reactive_source_topology_memberships: tuple[tuple[str, ...], ...] | None = None,
) -> ReactiveSonicPoolArtifact:
    count = int(common_count)
    reactive_primitive_ids, reactive_primitive_families = _reactive_region_primitives(
        definition,
        count,
    )
    artifact = ReactiveSonicPoolArtifact(
        definition=definition,
        common_count=count,
        source_topology_ids=source_topology_ids,
        reactive_source_labels=tuple(gic.name or gic.identifier for gic in definition.gics[count:]),
        reactive_source_topology_memberships=(
            tuple(tuple(source_topology_ids) for _gic in definition.gics[count:])
            if reactive_source_topology_memberships is None
            else reactive_source_topology_memberships
        ),
        reactive_primitive_ids=reactive_primitive_ids,
        reactive_primitive_families=reactive_primitive_families,
    )
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(reactive_sonic_pool_artifact_json(artifact) + "\n", encoding="utf-8")
    return artifact


def read_reactive_sonic_pool_artifact(path: Path | str) -> ReactiveSonicPoolArtifact:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read reactive SONIC pool artifact: {source}") from exc
    return reactive_sonic_pool_artifact_from_dict(payload)


def _reactive_sonic_pool_payload(
    artifact: ReactiveSonicPoolArtifact,
) -> dict[str, Any]:
    return {
        "schema": artifact.schema,
        "common_count": artifact.common_count,
        "source_topology_ids": list(artifact.source_topology_ids),
        "reactive_source_labels": list(artifact.reactive_source_labels),
        "reactive_source_topology_memberships": [
            list(membership)
            for membership in artifact.reactive_source_topology_memberships
        ],
        "reactive_primitive_ids": list(artifact.reactive_primitive_ids),
        "reactive_primitive_families": list(artifact.reactive_primitive_families),
        "reactive_coordinate_class": artifact.reactive_coordinate_class,
        "family_mixing_policy": artifact.family_mixing_policy,
        "definition": gic_definition_to_dict(artifact.definition),
    }


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reactive_region_primitives(
    definition: GICDefinition,
    common_count: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    referenced = {
        primitive_id
        for gic in definition.gics[int(common_count) :]
        for primitive_id, _coefficient in (
            gic.coefficients or ((gic.primitive_id, 1.0),)
        )
    }
    identifiers = tuple(
        primitive.identifier
        for primitive in definition.primitives
        if primitive.identifier in referenced
    )
    if not identifiers:
        raise ValueError("reactive SONIC region contains no primitive coordinates")
    families = tuple(
        sorted(
            {
                primitive.family
                for primitive in definition.primitives
                if primitive.identifier in referenced
            }
        )
    )
    return identifiers, families


def _validate_pool_tangent(definition: GICDefinition, common_count: int) -> None:
    rows = np.asarray(build_gic_b_matrix(definition).rows, dtype=float)
    if not np.all(np.isfinite(rows)):
        raise ValueError("reactive SONIC pool tangent contains non-finite values")
    singular = np.linalg.svd(rows, compute_uv=False)
    scale = max(float(singular[0]), 1.0) if singular.size else 1.0
    # Use the same scientific rank gate as endpoint-pool construction.  The
    # machine-epsilon threshold incorrectly promotes round-off rigid-body
    # leakage from frozen SOOP duals to additional molecular directions.
    tolerance = 1.0e-9 * scale
    rank = int(np.count_nonzero(singular > tolerance))
    if rank != definition.target_rank:
        raise ValueError(
            "reactive SONIC pool does not span the exact molecular tangent rank: "
            f"observed={rank} target={definition.target_rank} rows={rows.shape[0]} "
            f"threshold={tolerance:.6g}"
        )
    common = rows[:common_count]
    if common_count:
        common_singular = np.linalg.svd(common, compute_uv=False)
        common_rank = int(np.count_nonzero(common_singular > tolerance))
        if common_rank != common_count:
            raise ValueError("reactive SONIC conserved prefix is rank deficient")
        projected = (
            rows[common_count:]
            - (rows[common_count:] @ np.linalg.pinv(common, rcond=np.sqrt(np.finfo(float).eps)))
            @ common
        )
    else:
        projected = rows
    reactive_singular = np.linalg.svd(projected, compute_uv=False)
    reactive_rank = int(np.count_nonzero(reactive_singular > tolerance))
    if reactive_rank != definition.target_rank - common_count:
        raise ValueError("reactive SONIC source rows do not span the conserved complement")


__all__ = [
    "REACTIVE_SONIC_POOL_ARTIFACT_SCHEMA",
    "ReactiveSonicPoolArtifact",
    "reactive_sonic_pool_artifact_from_dict",
    "reactive_sonic_pool_artifact_json",
    "reactive_sonic_pool_artifact_to_dict",
    "read_reactive_sonic_pool_artifact",
    "write_reactive_sonic_pool_artifact",
]
