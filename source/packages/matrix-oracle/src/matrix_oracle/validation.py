"""Uniform, non-destructive validation for ORACLE JSON artifacts."""
from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any
from pathlib import Path

ORACLE_VALIDATION_SCHEMA = "matrix.oracle.artifact_validation.v1"

def validate_artifact(payload: Mapping[str, Any], *, expected_schema: str | None = None,
                      required_keys: Sequence[str] = ()) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(payload, Mapping):
        errors.append("artifact must be a mapping")
        payload = {}
    schema = payload.get("schema")
    if not isinstance(schema, str) or not schema.startswith("matrix.oracle."):
        errors.append("artifact schema must start with matrix.oracle.")
    if expected_schema is not None and schema != expected_schema:
        errors.append(f"expected schema {expected_schema}")
    errors.extend(f"missing required key: {key}" for key in required_keys if key not in payload)
    return {"schema": ORACLE_VALIDATION_SCHEMA, "valid": not errors,
            "artifact_schema": schema, "errors": errors}

def validate_analysis_report(payload: Mapping[str, Any]) -> dict[str, Any]:
    return validate_artifact(payload, expected_schema="matrix.oracle.analysis.v2",
                             required_keys=("oracle_version", "source_sha256", "output_sha256", "geometry", "topology"))

def validate_xyzin_output(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    errors: list[str] = []
    if not target.is_file() or target.stat().st_size == 0:
        errors.append("output is missing or empty")
    else:
        text = target.read_text(encoding="utf-8", errors="replace")
        for section in ("#BASIC", "#SYMMETRY", "#TOPOLOGY", "#VALIDATION"):
            if section not in text:
                errors.append(f"missing section {section}")
    return {"schema": ORACLE_VALIDATION_SCHEMA, "valid": not errors,
            "path": str(target), "errors": errors}
