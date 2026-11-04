"""Lightweight reader for a frozen SMITH SYCART basis."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from matrix_core import read_sectioned_lines, section_content

from .contracts import ORACLE_XYZ_SYCART_SCHEMA, GICForgeContractError
from .models import SYCartDefinition
from .policy import SYCART_BACKEND


def _field(lines: list[str], key: str) -> str | None:
    expected = key.upper()
    for line in lines:
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[0].upper() == expected:
            return parts[1].strip()
    return None


def _integer(lines: list[str], key: str) -> int:
    text = _field(lines, key)
    if text is None:
        return 0
    try:
        return int(text)
    except ValueError as exc:
        raise GICForgeContractError(f"invalid #SYCART integer field {key}: {text}") from exc


def _coordinate_lines(lines: list[str]) -> list[str]:
    start = None
    for index, line in enumerate(lines):
        if line.strip().upper() == "[SYCART]":
            start = index + 1
            break
    if start is None:
        return []
    end = len(lines)
    for index in range(start, len(lines)):
        text = lines[index].strip()
        if text.startswith("[") and text.endswith("]"):
            end = index
            break
    return list(lines[start:end])


def read_sycart_definition_from_xyzin(path: Path) -> SYCartDefinition:
    """Read the serialized basis without regenerating symmetry projectors."""

    target = Path(path)
    lines = read_sectioned_lines(target)
    section = section_content(lines, "SYCART")
    if not section:
        raise GICForgeContractError("missing #SYCART section")
    if section[0].strip() != f"SCHEMA {ORACLE_XYZ_SYCART_SCHEMA}":
        raise GICForgeContractError("invalid #SYCART schema")
    status = _field(section, "STATUS")
    if (status or "").upper() != "BUILT":
        raise GICForgeContractError(
            f"#SYCART status must be BUILT; found {status or 'UNKNOWN'}"
        )
    try:
        natoms = int(lines[0].strip())
    except (IndexError, ValueError) as exc:
        raise GICForgeContractError("invalid atom count before #SYCART") from exc
    if natoms <= 0:
        raise GICForgeContractError("invalid atom count before #SYCART")
    ncart = 3 * natoms
    vectors: list[tuple[float, ...]] = []
    irreps: list[str] = []
    axes = {"X": 0, "Y": 1, "Z": 2}
    for raw in _coordinate_lines(section):
        text = raw.strip()
        if not text or text.upper() == "NONE":
            continue
        parts = text.split()
        expected = f"SYC{len(vectors) + 1:03d}"
        if not parts or parts[0].upper() != expected:
            raise GICForgeContractError(
                f"invalid frozen SYCART coordinate order: expected {expected}"
            )
        fields = {
            key.upper(): value
            for part in parts[1:]
            if "=" in part
            for key, value in (part.split("=", 1),)
        }
        irrep = fields.get("IRREP")
        encoded = fields.get("COMPONENTS")
        if not irrep or encoded is None:
            raise GICForgeContractError(f"malformed frozen SYCART coordinate: {raw}")
        vector = np.zeros(ncart, dtype=float)
        if encoded.upper() != "NONE":
            for component in encoded.split(";"):
                key, separator, value_text = component.partition("=")
                atom_text, axis_separator, axis = key.partition(":")
                if not separator or not axis_separator or axis.upper() not in axes:
                    raise GICForgeContractError(
                        f"malformed frozen SYCART component: {component}"
                    )
                try:
                    atom = int(atom_text)
                    value = float(value_text)
                except ValueError as exc:
                    raise GICForgeContractError(
                        f"malformed frozen SYCART component: {component}"
                    ) from exc
                if atom < 1 or atom > natoms or not np.isfinite(value):
                    raise GICForgeContractError(
                        f"invalid frozen SYCART component: {component}"
                    )
                vector[3 * (atom - 1) + axes[axis.upper()]] = value
        vectors.append(tuple(float(value) for value in vector))
        irreps.append(irrep)
    target_rank = _integer(section, "TARGET_RANK")
    coordinate_count = _integer(section, "COORD_COUNT")
    if target_rank <= 0 or coordinate_count != target_rank or len(vectors) != target_rank:
        raise GICForgeContractError("frozen #SYCART rank/count metadata is inconsistent")
    return SYCartDefinition(
        backend=_field(section, "BACKEND") or SYCART_BACKEND,
        point_group=_field(section, "POINT_GROUP") or "C1",
        target_rank=target_rank,
        vectors=tuple(vectors),
        irreps=tuple(irreps),
        external_mode_count=_integer(section, "EXTERNAL_MODE_COUNT"),
        linearity=_field(section, "LINEARITY") or "",
        gauge_policy=_field(section, "COMPONENT_GAUGE") or "",
    )


__all__ = ["read_sycart_definition_from_xyzin"]
