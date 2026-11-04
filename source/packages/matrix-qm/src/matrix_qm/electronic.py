from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex

from matrix_core import read_sectioned_lines, replace_section, section_content


ORACLE_XYZ_ELECTRONIC_SCHEMA = "oracle.xyz.electronic.v1"
ORACLE_XYZ_TRANSITIONS_SCHEMA = "oracle.xyz.transitions.v1"
ORACLE_XYZ_ORBITALS_SCHEMA = "oracle.xyz.orbitals.v1"


@dataclass(frozen=True)
class ElectronicStateRecord:
    label: str
    energy_hartree: float | None = None
    energy_ev: float | None = None
    multiplicity: str = ""
    symmetry: str = ""
    source: str = ""


@dataclass(frozen=True)
class ElectronicSection:
    states: tuple[ElectronicStateRecord, ...] = ()
    schema: str = ORACLE_XYZ_ELECTRONIC_SCHEMA


@dataclass(frozen=True)
class ElectronicTransitionRecord:
    from_state: str
    to_state: str
    energy_ev: float
    oscillator_strength: float | None = None
    wavelength_nm: float | None = None
    strength: str = "electric-dipole"
    source: str = ""


@dataclass(frozen=True)
class TransitionsSection:
    transitions: tuple[ElectronicTransitionRecord, ...] = ()
    schema: str = ORACLE_XYZ_TRANSITIONS_SCHEMA


@dataclass(frozen=True)
class OrbitalFileRecord:
    kind: str
    format: str
    role: str
    path: Path
    label: str = ""
    source: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", self.kind.upper() or "FILE")
        object.__setattr__(self, "format", self.format.upper())
        object.__setattr__(self, "role", self.role.lower() or "orbitals")
        object.__setattr__(self, "path", Path(self.path))


@dataclass(frozen=True)
class OrbitalsSection:
    files: tuple[OrbitalFileRecord, ...] = ()
    schema: str = ORACLE_XYZ_ORBITALS_SCHEMA


def electronic_section_lines(section: ElectronicSection) -> list[str]:
    lines = [
        f"SCHEMA {ORACLE_XYZ_ELECTRONIC_SCHEMA}",
        "COLUMNS LABEL ENERGY_HARTREE ENERGY_EV MULTIPLICITY SYMMETRY SOURCE",
    ]
    for record in section.states:
        lines.append(
            " ".join(
                (
                    _token(record.label),
                    _float_or_dash(record.energy_hartree),
                    _float_or_dash(record.energy_ev),
                    _token(record.multiplicity),
                    _token(record.symmetry),
                    _token(record.source),
                )
            )
        )
    return lines


def transitions_section_lines(section: TransitionsSection) -> list[str]:
    lines = [
        f"SCHEMA {ORACLE_XYZ_TRANSITIONS_SCHEMA}",
        "COLUMNS FROM TO ENERGY_EV WAVELENGTH_NM OSC STRENGTH SOURCE",
    ]
    for record in section.transitions:
        lines.append(
            " ".join(
                (
                    _token(record.from_state),
                    _token(record.to_state),
                    _format_float(record.energy_ev),
                    _float_or_dash(record.wavelength_nm),
                    _float_or_dash(record.oscillator_strength),
                    _token(record.strength),
                    _token(record.source),
                )
            )
        )
    return lines


def orbitals_section_lines(section: OrbitalsSection) -> list[str]:
    lines = [
        f"SCHEMA {ORACLE_XYZ_ORBITALS_SCHEMA}",
        "COLUMNS KIND FORMAT ROLE PATH LABEL SOURCE",
    ]
    for record in section.files:
        lines.append(
            " ".join(
                (
                    _token(record.kind),
                    _token(record.format),
                    _token(record.role),
                    _token(str(record.path)),
                    _token(record.label),
                    _token(record.source),
                )
            )
        )
    return lines


def parse_electronic_section(lines: list[str] | tuple[str, ...]) -> ElectronicSection:
    records: list[ElectronicStateRecord] = []
    for row in _data_rows(lines):
        parts = _split(row)
        if not parts:
            continue
        if parts[0].upper() == "STATE":
            records.append(_legacy_state_record(parts))
            continue
        if len(parts) >= 6:
            records.append(
                ElectronicStateRecord(
                    label=parts[0],
                    energy_hartree=_optional_float(parts[1]),
                    energy_ev=_optional_float(parts[2]),
                    multiplicity=_dash_to_empty(parts[3]),
                    symmetry=_dash_to_empty(parts[4]),
                    source=_dash_to_empty(parts[5]),
                )
            )
    return ElectronicSection(tuple(records))


def parse_transitions_section(lines: list[str] | tuple[str, ...]) -> TransitionsSection:
    records: list[ElectronicTransitionRecord] = []
    columns: tuple[str, ...] = ()
    for raw in lines:
        text = raw.strip()
        if text.upper().startswith("COLUMNS "):
            columns = tuple(item.upper() for item in text.split()[1:])
            continue
        if not _is_data_line(text):
            continue
        parts = _split(text)
        if not parts:
            continue
        if columns:
            values = {column: parts[idx] for idx, column in enumerate(columns[: len(parts)])}
            from_state = values.get("FROM") or values.get("FROM_STATE") or parts[0]
            to_state = (
                values.get("TO") or values.get("TO_STATE") or (parts[1] if len(parts) > 1 else "")
            )
            energy = values.get("ENERGY_EV") or values.get("EV")
            if energy is None:
                continue
            records.append(
                ElectronicTransitionRecord(
                    from_state=from_state,
                    to_state=to_state,
                    energy_ev=float(energy.replace("D", "E")),
                    wavelength_nm=_optional_float(values.get("WAVELENGTH_NM", "-")),
                    oscillator_strength=_optional_float(values.get("OSC", "-")),
                    strength=_dash_to_empty(values.get("STRENGTH", "electric-dipole"))
                    or "electric-dipole",
                    source=_dash_to_empty(values.get("SOURCE", "")),
                )
            )
        elif len(parts) >= 4:
            records.append(
                ElectronicTransitionRecord(
                    from_state=parts[0],
                    to_state=parts[1],
                    energy_ev=float(parts[2].replace("D", "E")),
                    oscillator_strength=_optional_float(parts[3]),
                )
            )
    return TransitionsSection(tuple(records))


def parse_orbitals_section(lines: list[str] | tuple[str, ...]) -> OrbitalsSection:
    records: list[OrbitalFileRecord] = []
    columns: tuple[str, ...] = ()
    for raw in lines:
        text = raw.strip()
        if text.upper().startswith("COLUMNS "):
            columns = tuple(item.upper() for item in text.split()[1:])
            continue
        if not _is_data_line(text):
            continue
        parts = _split(text)
        if not parts:
            continue
        if columns and len(parts) >= 4:
            values = {column: parts[idx] for idx, column in enumerate(columns[: len(parts)])}
            path = values.get("PATH")
            if path is None:
                continue
            records.append(
                OrbitalFileRecord(
                    kind=values.get("KIND", "FILE"),
                    format=values.get("FORMAT", _format_from_path(path)),
                    role=values.get("ROLE", "orbitals"),
                    path=Path(path),
                    label=_dash_to_empty(values.get("LABEL", "")),
                    source=_dash_to_empty(values.get("SOURCE", "")),
                )
            )
            continue
        if len(parts) >= 2 and parts[0].upper() in {"MOLDEN", "CUBE", "FCHK", "FCH", "XYZ"}:
            path = parts[1]
            records.append(
                OrbitalFileRecord(
                    kind="FILE",
                    format="FCHK" if parts[0].upper() == "FCH" else parts[0],
                    role=_role_from_format(parts[0], path),
                    path=Path(path),
                    label=Path(path).stem,
                    source="legacy",
                )
            )
    return OrbitalsSection(tuple(records))


def read_electronic_section(path: Path | str) -> ElectronicSection:
    return parse_electronic_section(section_content(read_sectioned_lines(Path(path)), "ELECTRONIC"))


def read_transitions_section(path: Path | str) -> TransitionsSection:
    return parse_transitions_section(
        section_content(read_sectioned_lines(Path(path)), "TRANSITIONS")
    )


def read_orbitals_section(path: Path | str) -> OrbitalsSection:
    return parse_orbitals_section(section_content(read_sectioned_lines(Path(path)), "ORBITALS"))


def write_electronic_section(path: Path | str, section: ElectronicSection) -> None:
    replace_section(Path(path), "ELECTRONIC", electronic_section_lines(section))


def write_transitions_section(path: Path | str, section: TransitionsSection) -> None:
    replace_section(Path(path), "TRANSITIONS", transitions_section_lines(section))


def write_orbitals_section(path: Path | str, section: OrbitalsSection) -> None:
    replace_section(Path(path), "ORBITALS", orbitals_section_lines(section))


def merge_orbitals_section(
    path: Path | str, records: tuple[OrbitalFileRecord, ...]
) -> OrbitalsSection:
    target = Path(path)
    current = read_orbitals_section(target)
    merged: dict[tuple[str, str, str], OrbitalFileRecord] = {
        _orbital_key(record): record for record in current.files
    }
    for record in records:
        merged[_orbital_key(record)] = record
    section = OrbitalsSection(tuple(merged.values()))
    write_orbitals_section(target, section)
    return section


def orbital_file_record_from_path(
    path: Path | str,
    *,
    role: str | None = None,
    label: str = "",
    source: str = "",
) -> OrbitalFileRecord:
    target = Path(path)
    fmt = _format_from_path(str(target))
    return OrbitalFileRecord(
        kind="FILE",
        format=fmt,
        role=role or _role_from_format(fmt, str(target)),
        path=target,
        label=label or target.stem,
        source=source,
    )


def _legacy_state_record(parts: list[str]) -> ElectronicStateRecord:
    label = parts[1] if len(parts) > 1 else "S0"
    values: dict[str, str] = {}
    for item in parts[2:]:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        values[key.upper()] = value
    return ElectronicStateRecord(
        label=label,
        energy_hartree=_optional_float(values.get("ENERGY_HARTREE", "-")),
        energy_ev=_optional_float(values.get("ENERGY_EV", "-")),
        multiplicity=values.get("MULTIPLICITY", ""),
        symmetry=values.get("SYMMETRY", ""),
        source=values.get("SOURCE", "legacy"),
    )


def _data_rows(lines: list[str] | tuple[str, ...]) -> list[str]:
    return [line.strip() for line in lines if _is_data_line(line.strip())]


def _is_data_line(text: str) -> bool:
    if not text:
        return False
    upper = text.upper()
    return not (
        upper.startswith("SCHEMA ")
        or upper.startswith("COLUMNS ")
        or text.startswith("[")
        or text.startswith("#")
    )


def _split(text: str) -> list[str]:
    return shlex.split(text, comments=False, posix=True)


def _format_from_path(path: str) -> str:
    suffix = Path(path).suffix.lower().lstrip(".")
    if suffix in {"fchk", "fch"}:
        return "FCHK"
    if suffix in {"molden", "input"} or path.lower().endswith(".molden.input"):
        return "MOLDEN"
    if suffix in {"cube", "cub"}:
        return "CUBE"
    if suffix in {"xyz", "xyzin"}:
        return "XYZ"
    return suffix.upper() if suffix else "UNKNOWN"


def _role_from_format(fmt: str, path: str) -> str:
    text = f"{fmt} {path}".lower()
    if "density" in text or fmt.upper() == "CUBE":
        return "density"
    if fmt.upper() in {"XYZ", "XYZIN"}:
        return "geometry"
    return "orbitals"


def _orbital_key(record: OrbitalFileRecord) -> tuple[str, str, str]:
    return (record.format.upper(), str(record.path), record.role.lower())


def _optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text == "-":
        return None
    return float(text.replace("D", "E").replace("d", "e"))


def _float_or_dash(value: float | None) -> str:
    return "-" if value is None else _format_float(value)


def _format_float(value: float) -> str:
    return f"{float(value):.12g}"


def _dash_to_empty(value: str | None) -> str:
    if value is None or value == "-":
        return ""
    return value


def _token(value: object) -> str:
    text = str(value) if value not in (None, "") else "-"
    return shlex.quote(text)
