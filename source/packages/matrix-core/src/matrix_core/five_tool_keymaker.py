"""Exclusive KEYMAKER contract for the consolidated five-tool path.

This module contains routing metadata only.  Scientific implementations remain
in SWITCH, ORACLE, SMITH, LINK, and APOC and are reached through immutable
artifacts rather than package-internal calls from KEYMAKER.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .atomic_io import atomic_json_write
from .manifest import sha256_file


FIVE_TOOL_READINESS_SCHEMA = "matrix.keymaker.five-tool-readiness.v1"
FIVE_TOOL_PLAN_SCHEMA = "matrix.keymaker.five-tool-plan.v1"
FIVE_TOOL_DRY_RUN_SCHEMA = "matrix.keymaker.five-tool-dry-run.v1"
FIVE_TOOL_WORKFLOWS = (
    "minimum",
    "noncovalent_minimum",
    "ts_endpoint_pair",
    "energy_only",
)
FIVE_TOOL_ROUTE = ("SWITCH", "ORACLE", "SMITH", "LINK")


_READINESS = (
    {
        "tool": "SWITCH",
        "version": 1,
        "role": "source-normalization-and-geometry-seeding",
        "requires": ("matrix.keymaker.source.v1",),
        "produces": ("matrix.geometry.seed.v1",),
    },
    {
        "tool": "ORACLE",
        "version": 1,
        "role": "scientific-perception-and-immutable-atlas",
        "requires": ("matrix.geometry.seed.v1",),
        "produces": ("oracle.sonic.contract.v2", "matrix.oracle.coordinate_atlas.v1"),
    },
    {
        "tool": "SMITH",
        "version": 1,
        "role": "sonic-coordinate-selection",
        "requires": ("oracle.sonic.contract.v2", "matrix.oracle.coordinate_atlas.v1"),
        "produces": ("matrix.smith.typed_onic.artifact.v1",),
    },
    {
        "tool": "APOC",
        "version": 1,
        "role": "link-electronic-state-service",
        "requires": ("matrix.qm.result.v1",),
        "produces": ("matrix.apoc.scf-state.v1", "matrix.apoc.analysis.v1"),
        "execution_phase": "LINK",
    },
    {
        "tool": "LINK",
        "version": 1,
        "role": "point-evaluation-and-geometry-optimization",
        "requires": ("matrix.smith.typed_onic.artifact.v1",),
        "optional": (
            "matrix.apoc.scf-state.v1",
            "matrix.epoc.boundary.v1",
            "matrix.epoc.evaluation.v1",
        ),
        "produces": ("matrix.link.workflow-result.v1", "matrix.qm.result.v1"),
        "services": ("APOC",),
    },
)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def five_tool_readiness_contracts() -> tuple[dict[str, Any], ...]:
    """Return caller-safe, versioned readiness contracts for all five tools."""

    contracts: list[dict[str, Any]] = []
    for raw in _READINESS:
        payload = {
            "schema": FIVE_TOOL_READINESS_SCHEMA,
            "access": "KEYMAKER_ONLY",
            **{key: list(value) if isinstance(value, tuple) else value for key, value in raw.items()},
        }
        payload["contract_sha256"] = _canonical_sha256(payload)
        contracts.append(payload)
    return tuple(json.loads(json.dumps(item)) for item in contracts)


def five_tool_readiness_schema_path() -> Path:
    return Path(__file__).resolve().parent / "schemas" / "five-tool-readiness-v1.schema.json"


def five_tool_plan_schema_path() -> Path:
    return Path(__file__).resolve().parent / "schemas" / "five-tool-plan-v1.schema.json"


def immutable_five_tool_artifact(
    path: Path | str,
    *,
    schema: str,
    owner: str,
) -> dict[str, str]:
    """Describe an existing artifact by immutable content identity."""

    target = Path(path).expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(target)
    normalized_owner = str(owner).strip().upper()
    if normalized_owner not in {"KEYMAKER", "SWITCH", "ORACLE", "SMITH", "LINK", "APOC", "EPOC"}:
        raise ValueError(f"unsupported artifact owner: {owner!r}")
    if not str(schema).strip():
        raise ValueError("artifact schema must not be empty")
    return {
        "path": str(target),
        "schema": str(schema),
        "owner": normalized_owner,
        "sha256": sha256_file(target),
    }


def build_five_tool_keymaker_plan(
    workflow: str,
    *,
    source_artifacts: Sequence[Mapping[str, Any]],
    manifest_sha256: Mapping[str, str],
    issuer: str = "KEYMAKER",
) -> dict[str, Any]:
    """Resolve an immutable five-tool route; direct callers fail closed."""

    if str(issuer).strip().upper() != "KEYMAKER":
        raise PermissionError("the consolidated five-tool workflow is KEYMAKER-only")
    workflow_key = str(workflow).strip().casefold()
    if workflow_key not in FIVE_TOOL_WORKFLOWS:
        raise ValueError(f"unsupported five-tool workflow: {workflow!r}")
    artifacts = [dict(item) for item in source_artifacts]
    if not any(item.get("schema") == "matrix.keymaker.source.v1" for item in artifacts):
        raise ValueError("five-tool workflow requires an immutable KEYMAKER source artifact")
    for artifact in artifacts:
        _validate_artifact_reference(artifact)
    manifests = {str(key): str(value) for key, value in sorted(manifest_sha256.items())}
    if not manifests or any(len(value) != 64 for value in manifests.values()):
        raise ValueError("five-tool workflow requires SHA-256 identities for all manifests")
    contracts = five_tool_readiness_contracts()
    contract_by_tool = {item["tool"]: item for item in contracts}
    payload: dict[str, Any] = {
        "schema": FIVE_TOOL_PLAN_SCHEMA,
        "issuer": "KEYMAKER",
        "access": "KEYMAKER_ONLY",
        "workflow": workflow_key,
        "route": list(FIVE_TOOL_ROUTE),
        "link_services": ["APOC"],
        "source_artifacts": artifacts,
        "manifest_sha256": manifests,
        "readiness_contract_sha256": {
            tool: contract_by_tool[tool]["contract_sha256"]
            for tool in ("SWITCH", "ORACLE", "SMITH", "LINK", "APOC")
        },
        "excluded_tools": ["ARCHITECT", "TRINITY"],
    }
    payload["plan_sha256"] = _canonical_sha256(payload)
    return payload


def validate_five_tool_keymaker_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(dict(plan)))
    digest = str(payload.pop("plan_sha256", ""))
    if payload.get("schema") != FIVE_TOOL_PLAN_SCHEMA:
        raise ValueError("unsupported five-tool plan schema")
    if payload.get("issuer") != "KEYMAKER" or payload.get("access") != "KEYMAKER_ONLY":
        raise PermissionError("five-tool plan was not issued by KEYMAKER")
    if tuple(payload.get("route", ())) != FIVE_TOOL_ROUTE:
        raise ValueError("five-tool route is not canonical")
    if tuple(payload.get("link_services", ())) != ("APOC",):
        raise ValueError("APOC must be the exclusive LINK electronic-state service")
    if digest != _canonical_sha256(payload):
        raise ValueError("five-tool plan identity mismatch")
    payload["plan_sha256"] = digest
    return payload


def dry_run_five_tool_keymaker_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Validate routing, source bytes, and the artifact chain without execution."""

    payload = validate_five_tool_keymaker_plan(plan)
    issues: list[str] = []
    for artifact in payload["source_artifacts"]:
        try:
            _validate_artifact_reference(artifact, verify_file=True)
        except (FileNotFoundError, ValueError) as exc:
            issues.append(str(exc))
    available = {item["schema"] for item in payload["source_artifacts"]}
    rows: list[dict[str, Any]] = []
    for contract in five_tool_readiness_contracts():
        if contract["tool"] == "APOC":
            continue
        missing = sorted(set(contract["requires"]) - available)
        rows.append({"tool": contract["tool"], "ready": not missing, "missing": missing})
        if not missing:
            available.update(contract["produces"])
    apoc = next(item for item in five_tool_readiness_contracts() if item["tool"] == "APOC")
    rows.append(
        {
            "tool": "APOC",
            "ready": "matrix.qm.result.v1" in available,
            "missing": [] if "matrix.qm.result.v1" in available else list(apoc["requires"]),
            "execution_phase": "LINK",
        }
    )
    return {
        "schema": FIVE_TOOL_DRY_RUN_SCHEMA,
        "plan_sha256": payload["plan_sha256"],
        "ready": not issues and all(row["ready"] for row in rows),
        "issues": issues,
        "stages": rows,
    }


def write_five_tool_keymaker_plan(path: Path | str, plan: Mapping[str, Any]) -> Path:
    target = Path(path).expanduser().resolve()
    atomic_json_write(target, validate_five_tool_keymaker_plan(plan))
    return target


def read_five_tool_keymaker_plan(path: Path | str) -> dict[str, Any]:
    return validate_five_tool_keymaker_plan(
        json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    )


def _validate_artifact_reference(
    artifact: Mapping[str, Any], *, verify_file: bool = False
) -> None:
    required = {"path", "schema", "owner", "sha256"}
    if required - set(artifact):
        raise ValueError("incomplete immutable artifact reference")
    digest = str(artifact["sha256"])
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("artifact reference has invalid SHA-256")
    if verify_file:
        path = Path(str(artifact["path"]))
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != digest:
            raise ValueError(f"artifact content identity changed: {path}")


__all__ = [
    "FIVE_TOOL_DRY_RUN_SCHEMA",
    "FIVE_TOOL_PLAN_SCHEMA",
    "FIVE_TOOL_READINESS_SCHEMA",
    "FIVE_TOOL_ROUTE",
    "FIVE_TOOL_WORKFLOWS",
    "build_five_tool_keymaker_plan",
    "dry_run_five_tool_keymaker_plan",
    "five_tool_readiness_contracts",
    "five_tool_readiness_schema_path",
    "five_tool_plan_schema_path",
    "immutable_five_tool_artifact",
    "read_five_tool_keymaker_plan",
    "validate_five_tool_keymaker_plan",
    "write_five_tool_keymaker_plan",
]
