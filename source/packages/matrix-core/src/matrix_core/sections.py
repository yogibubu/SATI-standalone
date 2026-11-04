from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
import re

from .sectioned_xyz import read_sectioned_lines, replace_section, section_content


ORACLE_XYZ_BASIC_SCHEMA = "oracle.xyz.basic.v1"
SUPPORTED_BASIC_SCHEMAS = (ORACLE_XYZ_BASIC_SCHEMA,)


@dataclass(frozen=True)
class BasicSection:
    charge: int = 0
    multiplicity: int = 1
    point_group: str = "C1"
    temperature_K: float = 298.15
    pressure_atm: float = 1.0
    watson_reduction: str = "S"
    schema: str = ORACLE_XYZ_BASIC_SCHEMA


def parse_key_value_section(lines: Iterable[str]) -> dict[str, str]:
    """Parse permissive MATRIX key-value section lines."""
    values: dict[str, str] = {}
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("["):
            continue
        if line.upper().startswith("SCHEMA "):
            values["SCHEMA"] = line.split(None, 1)[1].strip()
            continue
        if "=" in line:
            key, value = line.split("=", 1)
        else:
            parts = line.split()
            if len(parts) < 2:
                continue
            key, value = " ".join(parts[:-1]), parts[-1]
        values[normalize_key(key)] = value.strip()
    return values


def key_value_section_lines(
    schema: str,
    values: Mapping[str, object],
    *,
    key_order: tuple[str, ...] = (),
) -> list[str]:
    lines = [f"SCHEMA {schema}"]
    emitted: set[str] = set()
    for key in key_order:
        if key in values and values[key] is not None:
            lines.append(f"{key} = {values[key]}")
            emitted.add(key)
    for key, value in sorted(values.items()):
        if key in emitted or value is None:
            continue
        lines.append(f"{key} = {value}")
    return lines


def normalize_key(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", key.strip()).strip("_").upper()


def parse_basic_section(lines: Iterable[str]) -> BasicSection:
    values = parse_key_value_section(lines)
    schema = values.get("SCHEMA", ORACLE_XYZ_BASIC_SCHEMA)
    if schema not in SUPPORTED_BASIC_SCHEMAS:
        raise ValueError(f"unsupported BASIC schema: {schema}")
    return BasicSection(
        charge=_int_value(values, "CHARGE", 0),
        multiplicity=_int_value(values, "MULTIPLICITY", _int_value(values, "SPIN_MULTIPLICITY", 1)),
        point_group=values.get("POINT_GROUP", "C1"),
        temperature_K=_float_value(values, "T_K", 298.15),
        pressure_atm=_float_value(values, "P_ATM", 1.0),
        watson_reduction=values.get("WATSON_REDUCTION", "S"),
        schema=schema,
    )


def basic_section_lines(section: BasicSection) -> list[str]:
    return key_value_section_lines(
        ORACLE_XYZ_BASIC_SCHEMA,
        {
            "CHARGE": int(section.charge),
            "MULTIPLICITY": int(section.multiplicity),
            "POINT_GROUP": section.point_group,
            "WATSON_REDUCTION": section.watson_reduction,
            "T_K": f"{float(section.temperature_K):.6f}",
            "P_ATM": f"{float(section.pressure_atm):.6f}",
        },
        key_order=("CHARGE", "MULTIPLICITY", "POINT_GROUP", "WATSON_REDUCTION", "T_K", "P_ATM"),
    )


def read_basic_section(path: Path) -> BasicSection:
    content = section_content(read_sectioned_lines(Path(path)), "BASIC")
    if not content:
        return BasicSection()
    return parse_basic_section(content)


def write_basic_section(path: Path, section: BasicSection) -> None:
    replace_section(Path(path), "BASIC", basic_section_lines(section))


def _int_value(values: Mapping[str, str], key: str, default: int) -> int:
    raw = values.get(key)
    if raw is None:
        return default
    return int(float(raw.replace("D", "E").replace("d", "e")))


def _float_value(values: Mapping[str, str], key: str, default: float) -> float:
    raw = values.get(key)
    if raw is None:
        return default
    return float(raw.replace("D", "E").replace("d", "e"))
