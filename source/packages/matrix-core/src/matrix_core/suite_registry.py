"""Unified read model for package, capability, operation and tool contracts."""

from __future__ import annotations

import json
from pathlib import Path

from .operation_registry import operation_contract, operation_registry_issues
from .package_registry import package_registry_issues, package_registry_payload
from .tool_contracts import tool_contracts


SUITE_REGISTRY_SCHEMA = "matrix.suite-registry.v1"


def suite_registry_issues() -> tuple[str, ...]:
    issues = [*package_registry_issues(), *operation_registry_issues()]
    for tool in tool_contracts():
        operation_key = tool.operation_key or tool.key
        try:
            operation_contract(operation_key)
        except KeyError:
            issues.append(f"tool {tool.key}: unknown operation {operation_key}")
    return tuple(issues)


def suite_registry_payload() -> dict[str, object]:
    issues = suite_registry_issues()
    if issues:
        raise ValueError("invalid MATRIX suite registry: " + "; ".join(issues))
    package_payload = package_registry_payload()
    operations: list[dict[str, object]] = []
    for tool in tool_contracts():
        owner = operation_contract(tool.operation_key or tool.key)
        operations.append(
            {
                "tool": tool.key,
                "operation": owner.key,
                "owner_package": owner.owner_package,
                "owner_api": owner.owner_api,
                "contract": tool.to_dict(),
            }
        )
    return {
        "schema": SUITE_REGISTRY_SCHEMA,
        "packages": package_payload["packages"],
        "capability_index": package_payload["capability_index"],
        "operations": operations,
    }


def suite_registry_json(*, indent: int = 2) -> str:
    return json.dumps(suite_registry_payload(), indent=indent, sort_keys=True)


def suite_registry_schema_path() -> Path:
    return Path(__file__).resolve().parent / "schemas" / "suite-registry-v1.schema.json"


__all__ = [
    "SUITE_REGISTRY_SCHEMA",
    "suite_registry_issues",
    "suite_registry_json",
    "suite_registry_payload",
    "suite_registry_schema_path",
]
