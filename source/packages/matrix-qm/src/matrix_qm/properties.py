from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import shlex

from matrix_core import read_sectioned_lines, replace_section, section_content


ORACLE_XYZ_PROPERTIES_SCHEMA = "oracle.xyz.properties.v1"
PROPERTY_COLUMNS = (
    "NAME",
    "TARGET",
    "TARGET_ID",
    "ATOM",
    "ISOTOPE",
    "UNIT",
    "AXES",
    "PROGRAM",
    "METHOD",
    "LEVEL",
    "STATUS",
    "CONVERSION",
    "UNCERTAINTY",
    "SOURCE",
    "VALUE",
    "COMMENT",
)


@dataclass(frozen=True)
class PropertyRecord:
    name: str
    value: tuple[float, ...]
    unit: str
    target: str = "MOLECULE"
    target_id: str = ""
    atom: int | None = None
    isotope: str = ""
    axes: str = ""
    program: str = ""
    method: str = ""
    level: str = ""
    source: str = ""
    status: str = "raw"
    conversion: str = ""
    uncertainty: float | None = None
    comment: str = ""

    def __post_init__(self) -> None:
        name = self.name.strip()
        unit = self.unit.strip()
        target = (self.target.strip() or "MOLECULE").upper()
        status = (self.status.strip() or "raw").lower()
        values = tuple(float(value) for value in self.value)
        atom = None if self.atom is None else int(self.atom)
        if not name:
            raise ValueError("property name cannot be empty")
        if not values:
            raise ValueError("property value cannot be empty")
        if not unit:
            raise ValueError("property unit cannot be empty")
        if atom is not None and atom <= 0:
            raise ValueError("property atom index must be one-based")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "value", values)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "atom", atom)
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "uncertainty",
            None if self.uncertainty is None else float(self.uncertainty),
        )


@dataclass(frozen=True)
class PropertiesSection:
    records: tuple[PropertyRecord, ...] = ()
    schema: str = ORACLE_XYZ_PROPERTIES_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))


@dataclass(frozen=True)
class PropertyComparison:
    reference_label: str
    candidate_label: str
    reference: PropertyRecord
    candidate: PropertyRecord
    delta: tuple[float, ...]
    max_abs_delta: float
    rms_delta: float
    compatible: bool


def properties_section_lines(section: PropertiesSection) -> list[str]:
    lines = [
        f"SCHEMA {ORACLE_XYZ_PROPERTIES_SCHEMA}",
        "COLUMNS " + " ".join(PROPERTY_COLUMNS),
    ]
    for record in section.records:
        lines.append(
            " ".join(
                (
                    _token(record.name),
                    _token(record.target),
                    _token(record.target_id),
                    _integer_or_dash(record.atom),
                    _token(record.isotope),
                    _token(record.unit),
                    _token(record.axes),
                    _token(record.program),
                    _token(record.method),
                    _token(record.level),
                    _token(record.status),
                    _token(record.conversion),
                    _float_or_dash(record.uncertainty),
                    _token(record.source),
                    _token(_format_value(record.value)),
                    _token(record.comment),
                )
            )
        )
    return lines


def parse_properties_section(lines: list[str] | tuple[str, ...]) -> PropertiesSection:
    records: list[PropertyRecord] = []
    columns = PROPERTY_COLUMNS
    schema = ORACLE_XYZ_PROPERTIES_SCHEMA
    for raw in lines:
        text = raw.strip()
        if not text:
            continue
        upper = text.upper()
        if upper.startswith("SCHEMA "):
            schema = text.split(maxsplit=1)[1].strip()
            continue
        if upper.startswith("COLUMNS "):
            columns = tuple(item.upper() for item in text.split()[1:])
            continue
        if upper.startswith("["):
            continue
        parts = _split(text)
        if not parts:
            continue
        values = {column: parts[idx] for idx, column in enumerate(columns[: len(parts)])}
        name = _dash_to_empty(values.get("NAME", ""))
        unit = _dash_to_empty(values.get("UNIT", ""))
        value_text = values.get("VALUE", "")
        if not name or not unit or not value_text:
            continue
        records.append(
            PropertyRecord(
                name=name,
                target=_dash_to_empty(values.get("TARGET", "MOLECULE")) or "MOLECULE",
                target_id=_dash_to_empty(values.get("TARGET_ID", "")),
                atom=_optional_int(values.get("ATOM", "-")),
                isotope=_dash_to_empty(values.get("ISOTOPE", "")),
                unit=unit,
                axes=_dash_to_empty(values.get("AXES", "")),
                program=_dash_to_empty(values.get("PROGRAM", "")),
                method=_dash_to_empty(values.get("METHOD", "")),
                level=_dash_to_empty(values.get("LEVEL", "")),
                status=_dash_to_empty(values.get("STATUS", "raw")) or "raw",
                conversion=_dash_to_empty(values.get("CONVERSION", "")),
                uncertainty=_optional_float(values.get("UNCERTAINTY", "-")),
                source=_dash_to_empty(values.get("SOURCE", "")),
                value=_parse_value(value_text),
                comment=_dash_to_empty(values.get("COMMENT", "")),
            )
        )
    return PropertiesSection(tuple(records), schema=schema)


def write_properties_section(path: Path | str, section: PropertiesSection) -> None:
    replace_section(Path(path), "PROPERTIES", properties_section_lines(section))


def read_properties_section(path: Path | str) -> PropertiesSection:
    return parse_properties_section(section_content(read_sectioned_lines(Path(path)), "PROPERTIES"))


def merge_properties_section(
    path: Path | str,
    records: tuple[PropertyRecord, ...],
    *,
    replace_matching: bool = True,
) -> PropertiesSection:
    target = Path(path)
    current = read_properties_section(target)
    if replace_matching:
        merged: dict[
            tuple[str, str, str, int | None, str, str, str, str, str, str, str, str],
            PropertyRecord,
        ]
        merged = {_property_key(record): record for record in current.records}
        for record in records:
            merged[_property_key(record)] = record
        section = PropertiesSection(tuple(merged.values()), schema=current.schema)
    else:
        section = PropertiesSection((*current.records, *records), schema=current.schema)
    write_properties_section(target, section)
    return section


def property_records_by_name(
    section: PropertiesSection,
    name: str,
) -> tuple[PropertyRecord, ...]:
    normalized = name.strip().upper()
    return tuple(record for record in section.records if record.name.upper() == normalized)


def property_records_for_atom(
    section: PropertiesSection,
    atom: int,
) -> tuple[PropertyRecord, ...]:
    target_atom = int(atom)
    return tuple(record for record in section.records if record.atom == target_atom)


def property_record_to_dict(record: PropertyRecord) -> dict[str, object]:
    return {
        "name": record.name,
        "target": record.target,
        "target_id": record.target_id,
        "atom": record.atom,
        "isotope": record.isotope,
        "value": record.value,
        "unit": record.unit,
        "axes": record.axes,
        "program": record.program,
        "method": record.method,
        "level": record.level,
        "source": record.source,
        "status": record.status,
        "conversion": record.conversion,
        "uncertainty": record.uncertainty,
        "comment": record.comment,
    }


def compare_property_records(
    reference: PropertyRecord,
    candidate: PropertyRecord,
    *,
    reference_label: str = "reference",
    candidate_label: str = "candidate",
    atol: float = 0.0,
    rtol: float = 0.0,
) -> PropertyComparison:
    if reference.name.upper() != candidate.name.upper():
        raise ValueError(
            f"cannot compare different properties: {reference.name} vs {candidate.name}"
        )
    if reference.unit != candidate.unit:
        raise ValueError(f"cannot compare different units: {reference.unit} vs {candidate.unit}")
    if len(reference.value) != len(candidate.value):
        raise ValueError(
            f"cannot compare values with different lengths: "
            f"{len(reference.value)} vs {len(candidate.value)}"
        )
    delta = tuple(
        candidate_value - reference_value
        for reference_value, candidate_value in zip(reference.value, candidate.value)
    )
    max_abs_delta = max((abs(value) for value in delta), default=0.0)
    rms_delta = math.sqrt(sum(value * value for value in delta) / len(delta)) if delta else 0.0
    compatible = all(
        abs(candidate_value - reference_value) <= float(atol) + float(rtol) * abs(reference_value)
        for reference_value, candidate_value in zip(reference.value, candidate.value)
    )
    return PropertyComparison(
        reference_label=reference_label,
        candidate_label=candidate_label,
        reference=reference,
        candidate=candidate,
        delta=delta,
        max_abs_delta=max_abs_delta,
        rms_delta=rms_delta,
        compatible=compatible,
    )


def property_comparison_to_dict(comparison: PropertyComparison) -> dict[str, object]:
    return {
        "reference": comparison.reference_label,
        "candidate": comparison.candidate_label,
        "property": comparison.reference.name,
        "unit": comparison.reference.unit,
        "target": comparison.reference.target,
        "target_id": comparison.reference.target_id,
        "atom": comparison.reference.atom,
        "isotope": comparison.reference.isotope,
        "reference_value": comparison.reference.value,
        "candidate_value": comparison.candidate.value,
        "delta": comparison.delta,
        "max_abs_delta": comparison.max_abs_delta,
        "rms_delta": comparison.rms_delta,
        "compatible": comparison.compatible,
    }


def property_comparison_lines(comparisons: tuple[PropertyComparison, ...]) -> list[str]:
    lines = [f"comparisons: {len(comparisons)}"]
    for comparison in comparisons:
        status = "ok" if comparison.compatible else "failed"
        lines.append(
            f"- {comparison.candidate_label} vs {comparison.reference_label}: "
            f"{comparison.reference.name} atom={comparison.reference.atom or '-'} "
            f"max_abs_delta={_format_float(comparison.max_abs_delta)} "
            f"rms_delta={_format_float(comparison.rms_delta)} "
            f"unit={comparison.reference.unit} status={status}"
        )
    return lines


def properties_summary_lines(
    section: PropertiesSection,
    *,
    name: str | None = None,
    atom: int | None = None,
) -> list[str]:
    records = _filtered_records(section, name=name, atom=atom)
    lines = [f"properties: {len(records)}"]
    for record in records:
        target = record.target
        if record.target_id:
            target = f"{target}:{record.target_id}"
        if record.atom is not None:
            target = f"{target}:atom={record.atom}"
        isotope = "" if not record.isotope else f" isotope={record.isotope}"
        program = "" if not record.program else f" program={record.program}"
        method = "" if not record.method else f" method={record.method}"
        conversion = "" if not record.conversion else f" conversion={record.conversion}"
        lines.append(
            f"- {record.name} {target}{isotope} "
            f"{_format_value(record.value)} {record.unit} "
            f"status={record.status}{program}{method}{conversion}"
        )
    return lines


def filtered_property_records(
    section: PropertiesSection,
    *,
    name: str | None = None,
    atom: int | None = None,
) -> tuple[PropertyRecord, ...]:
    return tuple(_filtered_records(section, name=name, atom=atom))


def _filtered_records(
    section: PropertiesSection,
    *,
    name: str | None = None,
    atom: int | None = None,
) -> list[PropertyRecord]:
    records = list(section.records)
    if name:
        normalized = name.strip().upper()
        records = [record for record in records if record.name.upper() == normalized]
    if atom is not None:
        target_atom = int(atom)
        records = [record for record in records if record.atom == target_atom]
    return records


def _property_key(
    record: PropertyRecord,
) -> tuple[str, str, str, int | None, str, str, str, str, str, str, str, str]:
    return (
        record.name.upper(),
        record.target.upper(),
        record.target_id,
        record.atom,
        record.isotope,
        record.unit,
        record.axes,
        record.program,
        record.method,
        record.level,
        record.status,
        record.conversion,
    )


def _token(text: object) -> str:
    value = str(text)
    if not value:
        return "-"
    return shlex.quote(value)


def _integer_or_dash(value: int | None) -> str:
    return "-" if value is None else str(int(value))


def _float_or_dash(value: float | None) -> str:
    return "-" if value is None else _format_float(value)


def _format_value(values: tuple[float, ...]) -> str:
    return ",".join(_format_float(value) for value in values)


def _format_float(value: float) -> str:
    return f"{float(value):.16g}"


def _split(text: str) -> list[str]:
    try:
        return shlex.split(text)
    except ValueError:
        return text.split()


def _parse_value(text: str) -> tuple[float, ...]:
    clean = _dash_to_empty(text).strip().strip("[]()")
    if not clean:
        return ()
    values: list[float] = []
    for token in clean.replace(";", " ").replace(",", " ").split():
        item = token.strip()
        if not item:
            continue
        values.append(float(item.replace("D", "E").replace("d", "e")))
    return tuple(values)


def _optional_float(text: str | None) -> float | None:
    clean = _dash_to_empty(text)
    if not clean:
        return None
    return float(clean.replace("D", "E").replace("d", "e"))


def _optional_int(text: str | None) -> int | None:
    clean = _dash_to_empty(text)
    if not clean:
        return None
    return int(float(clean))


def _dash_to_empty(text: str | None) -> str:
    if text is None:
        return ""
    value = str(text).strip()
    return "" if value == "-" else value
