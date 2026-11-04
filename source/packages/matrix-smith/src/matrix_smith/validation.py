"""Non-destructive validation of SMITH coordinate artifacts.

The validator deliberately does not rebuild topology or coordinates.  It checks
the frozen artifact contract and reports scientific metadata that downstream
tools need in order to decide whether a SONIC definition is safe to consume.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

from matrix_core import read_sectioned_lines, section_content


VALIDATION_SCHEMA = "matrix.smith.artifact_validation.v2"
_REQUIRED_SECTIONS = ("BASIC", "VALIDATION", "TOPOLOGY", "SYNTHONS", "PRIMITIVES")
_GIC_SECTION_NAMES = ("GIC", "GICS")


def validate_xyzin(path: str | Path) -> dict[str, Any]:
    """Validate the structural and scientific metadata of a SMITH artifact.

    This function is intentionally conservative: it validates the serialized
    contract but does not regenerate topology, rank, or B rows.  Regeneration
    belongs to ORACLE/SMITH construction; this validator is the fail-closed
    boundary before downstream consumers use a frozen definition.
    """

    target = Path(path)
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, bool] = {}
    result: dict[str, Any] = {
        "schema": VALIDATION_SCHEMA,
        "valid": False,
        "path": str(target),
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
    }
    if not target.is_file() or target.stat().st_size == 0:
        errors.append("missing or empty artifact")
        return result

    try:
        lines = read_sectioned_lines(target)
    except (OSError, UnicodeError, ValueError) as exc:
        errors.append(f"cannot parse sectioned XYZ artifact: {exc}")
        return result

    result["geometry_sha256"] = _section_sha256(lines, "BASIC")
    sections = {name for name in _known_section_names(lines)}
    for section in _REQUIRED_SECTIONS:
        present = section in sections
        checks[f"section:{section.lower()}"] = present
        if not present:
            errors.append(f"missing #{section}")

    gic_name = next((name for name in _GIC_SECTION_NAMES if name in sections), None)
    checks["section:gic"] = gic_name is not None
    if gic_name is None:
        errors.append("missing #GIC section")
        return result

    gic = section_content(lines, gic_name)
    metadata = _key_value_metadata(gic)
    _validate_gic_metadata(metadata, errors, warnings, checks)

    validation = _key_value_metadata(section_content(lines, "VALIDATION"))
    status = validation.get("STATUS", "").upper()
    checks["validation:pass"] = status == "PASS"
    if status != "PASS":
        errors.append(f"#VALIDATION status must be PASS; found {status or 'UNKNOWN'}")

    if not errors:
        result["valid"] = True
    result["gic_section"] = gic_name
    result["gic_status"] = metadata.get("STATUS", "UNKNOWN")
    result["rank"] = _integer_or_none(metadata.get("RANK"))
    result["target_rank"] = _integer_or_none(metadata.get("TARGET_RANK"))
    result["derivative_mode"] = metadata.get("B_MATRIX_DERIVATIVE_MODE", "UNKNOWN")
    return result


def _known_section_names(lines: list[str]) -> tuple[str, ...]:
    names: list[str] = []
    for line in lines:
        text = line.strip()
        if text.startswith("#") and len(text) > 1:
            name = text[1:].split()[0].upper()
            if name not in names:
                names.append(name)
    return tuple(names)


def _section_sha256(lines: list[str], name: str) -> str | None:
    content = section_content(lines, name)
    if not content:
        return None
    payload = "\n".join(line.rstrip() for line in content).encode("utf-8")
    return sha256(payload).hexdigest()


def _key_value_metadata(lines: list[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in lines:
        fields = line.strip().split(maxsplit=1)
        if len(fields) == 2 and fields[0].upper() not in {"[FROZEN_GICS]", "[SYCART]"}:
            metadata.setdefault(fields[0].upper(), fields[1].strip())
    return metadata


def _validate_gic_metadata(
    metadata: dict[str, str],
    errors: list[str],
    warnings: list[str],
    checks: dict[str, bool],
) -> None:
    schema = metadata.get("SCHEMA", "")
    checks["gic:schema"] = schema.startswith("oracle.xyz.gic.")
    if not checks["gic:schema"]:
        errors.append("#GIC lacks a supported oracle.xyz.gic schema")

    status = metadata.get("STATUS", "").upper()
    checks["gic:frozen"] = status == "FROZEN"
    if status == "PLANNED":
        errors.append("#GIC is planned, not a frozen SONIC definition")
    elif status != "FROZEN":
        errors.append(f"#GIC status must be FROZEN; found {status or 'UNKNOWN'}")

    for key in ("RANK", "TARGET_RANK", "CANDIDATE_COUNT", "RANK_TOLERANCE"):
        present = key in metadata
        checks[f"gic:{key.lower()}"] = present
        if not present:
            errors.append(f"#GIC lacks {key} metadata")

    derivative_mode = metadata.get("B_MATRIX_DERIVATIVE_MODE", "").upper()
    checks["gic:analytic_b"] = derivative_mode == "ANALYTIC"
    if derivative_mode != "ANALYTIC":
        errors.append("#GIC must declare B_MATRIX_DERIVATIVE_MODE ANALYTIC")

    if metadata.get("PRIMITIVE_B_MATRIX_SHA256", "NONE").upper() == "NONE":
        warnings.append("#GIC has no primitive B-matrix provenance hash")

    rank = _integer_or_none(metadata.get("RANK"))
    target_rank = _integer_or_none(metadata.get("TARGET_RANK"))
    if rank is not None and target_rank is not None and rank != target_rank:
        warnings.append(f"#GIC rank {rank} differs from target rank {target_rank}")


def _integer_or_none(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value.split()[0])
    except (TypeError, ValueError):
        return None
