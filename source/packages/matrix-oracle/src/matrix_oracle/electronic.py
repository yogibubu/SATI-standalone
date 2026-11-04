from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import shutil

from matrix_core import has_section, read_sectioned_lines, section_content
from matrix_qm import OrbitalFileRecord, read_orbitals_section, read_transitions_section

from .commands import (
    MORBVIS_URL,
    OracleGuiCommand,
    external_viewer_command,
    molden_command,
    morbvis_command,
)
from .publication import PublicationExportResult, export_electronic_spectrum


ELECTRONIC_SECTION_NAMES = ("ELECTRONIC", "TRANSITIONS", "ORBITALS")


@dataclass(frozen=True)
class ElectronicTable:
    title: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class ElectronicGuiState:
    xyzin: Path
    exists: bool
    ready: bool
    sections: ElectronicTable
    transitions: ElectronicTable
    orbitals: ElectronicTable
    orbital_records: tuple[OrbitalFileRecord, ...] = ()
    messages: tuple[str, ...] = ()


class OracleElectronicController:
    def __init__(self, xyzin: Path | str | None = None) -> None:
        self.xyzin = None if xyzin is None else Path(xyzin)

    def set_xyzin(self, xyzin: Path | str | None) -> ElectronicGuiState | None:
        self.xyzin = None if xyzin is None else Path(xyzin)
        if self.xyzin is None:
            return None
        return self.state()

    def state(self) -> ElectronicGuiState:
        if self.xyzin is None:
            raise ValueError("no MATRIX xyzin project is loaded")
        return load_electronic_gui_state(self.xyzin)

    def molden_command(
        self,
        target: Path | str,
        *,
        executable: str = "molden",
    ) -> OracleGuiCommand:
        return molden_command(target, executable=executable)

    def avogadro_command(
        self,
        target: Path | str,
        *,
        executable: str = "avogadro2",
    ) -> OracleGuiCommand:
        return external_viewer_command(target, executable=executable, label="Open in Avogadro")

    def morbvis_command(self, *, url: str = MORBVIS_URL) -> OracleGuiCommand:
        return morbvis_command(url=url)

    def selected_orbital_viewer_command(
        self,
        row: int,
        *,
        molden_executable: str = "molden",
        avogadro_executable: str = "avogadro2",
        morbvis_url: str = MORBVIS_URL,
    ) -> OracleGuiCommand:
        records = self.state().orbital_records
        if row < 0 or row >= len(records):
            raise ValueError("select an #ORBITALS row first")
        return viewer_command_for_orbital_record(
            records[row],
            molden_executable=molden_executable,
            avogadro_executable=avogadro_executable,
            morbvis_url=morbvis_url,
        )

    def export_electronic_publication(
        self,
        outdir: Path | str,
        *,
        formats: tuple[str, ...] = ("csv", "svg", "pdf"),
    ) -> PublicationExportResult:
        if self.xyzin is None:
            raise ValueError("no MATRIX xyzin project is loaded")
        return export_electronic_spectrum(self.xyzin, outdir, formats=formats)


def load_electronic_gui_state(path: Path | str) -> ElectronicGuiState:
    target = Path(path)
    if not target.exists():
        return ElectronicGuiState(
            xyzin=target,
            exists=False,
            ready=False,
            sections=ElectronicTable("Sections", ("Section", "Present", "Lines"), ()),
            transitions=ElectronicTable("Transitions", ("Line", "Record"), ()),
            orbitals=ElectronicTable("Orbitals", ("Line", "Record"), ()),
            messages=(f"Missing file: {target}",),
        )
    section_rows = []
    for name in ELECTRONIC_SECTION_NAMES:
        content = (
            section_content(read_sectioned_lines(target), name) if has_section(target, name) else []
        )
        section_rows.append((name, "yes" if content else "no", str(len(content))))
    ready = any(row[1] == "yes" for row in section_rows)
    messages = () if ready else ("missing #ELECTRONIC/#TRANSITIONS/#ORBITALS sections",)
    return ElectronicGuiState(
        xyzin=target,
        exists=True,
        ready=ready,
        sections=ElectronicTable("Sections", ("Section", "Present", "Lines"), tuple(section_rows)),
        transitions=_transitions_table(target),
        orbitals=_orbitals_table(target),
        orbital_records=read_orbitals_section(target).files,
        messages=messages,
    )


def electronic_gui_state_lines(state: ElectronicGuiState) -> list[str]:
    return [
        f"xyzin: {state.xyzin}",
        f"ready: {int(state.ready)}",
        *state.messages,
    ]


def _content_table(target: Path, section: str) -> ElectronicTable:
    if not has_section(target, section):
        return ElectronicTable(section.title(), ("Line", "Record"), ())
    content = [
        line.strip()
        for line in section_content(read_sectioned_lines(target), section)
        if line.strip() and not line.strip().upper().startswith("SCHEMA ")
    ]
    parsed = _columns_table(section, content)
    if parsed is not None:
        return parsed
    return ElectronicTable(
        section.title(),
        ("Line", "Record"),
        tuple((str(index), line) for index, line in enumerate(content, start=1)),
    )


def _columns_table(section: str, content: list[str]) -> ElectronicTable | None:
    columns: tuple[str, ...] | None = None
    rows: list[tuple[str, ...]] = []
    for raw in content:
        if raw.upper().startswith("COLUMNS "):
            columns = tuple(raw.split()[1:])
            continue
        if columns is None:
            continue
        parts = tuple(raw.split())
        if len(parts) == len(columns):
            rows.append(parts)
    if columns is None:
        return None
    return ElectronicTable(section.title(), columns, tuple(rows))


def _transitions_table(target: Path) -> ElectronicTable:
    if not has_section(target, "TRANSITIONS"):
        return ElectronicTable("Transitions", ("Line", "Record"), ())
    records = read_transitions_section(target).transitions
    if not records:
        return _content_table(target, "TRANSITIONS")
    return ElectronicTable(
        "Transitions",
        ("From", "To", "Energy eV", "Wavelength nm", "Osc", "Source"),
        tuple(
            (
                record.from_state,
                record.to_state,
                _format_float(record.energy_ev),
                "" if record.wavelength_nm is None else _format_float(record.wavelength_nm),
                ""
                if record.oscillator_strength is None
                else _format_float(record.oscillator_strength),
                record.source,
            )
            for record in records
        ),
    )


def _orbitals_table(target: Path) -> ElectronicTable:
    if not has_section(target, "ORBITALS"):
        return ElectronicTable("Orbitals", ("Line", "Record"), ())
    records = read_orbitals_section(target).files
    if not records:
        return _content_table(target, "ORBITALS")
    return ElectronicTable(
        "Orbitals",
        ("Kind", "Format", "Role", "Path", "Label", "Source"),
        tuple(
            (
                record.kind,
                record.format,
                record.role,
                str(record.path),
                record.label,
                record.source,
            )
            for record in records
        ),
    )


def viewer_command_for_orbital_record(
    record: OrbitalFileRecord,
    *,
    molden_executable: str = "molden",
    avogadro_executable: str = "avogadro2",
    morbvis_url: str = MORBVIS_URL,
) -> OracleGuiCommand:
    target = Path(record.path)
    fmt = record.format.upper()
    role = record.role.lower()
    if not target.exists():
        raise FileNotFoundError(f"viewer file not found: {target}")
    if role == "geometry" or fmt in {"XYZ", "XYZIN"}:
        if not _executable_available(avogadro_executable):
            raise ValueError(f"Avogadro executable not found: {avogadro_executable}")
        return external_viewer_command(
            target, executable=avogadro_executable, label="Open in Avogadro"
        )
    if fmt in {"MOLDEN", "CUBE"}:
        if _molden_ready(molden_executable):
            return molden_command(target, executable=molden_executable)
        return morbvis_command(url=morbvis_url)
    if fmt in {"FCHK", "FCH"}:
        if _molden_ready(molden_executable):
            return molden_command(target, executable=molden_executable)
        raise ValueError(
            "FCHK viewing requires Molden with XQuartz; MOrbVis accepts Molden/Cube files"
        )
    raise ValueError(f"no configured viewer for #ORBITALS format {fmt}")


def default_electronic_export_dir(xyzin: Path | str) -> Path:
    target = Path(xyzin)
    return target.with_name(f"{target.stem}.electronic_publication")


def _molden_ready(executable: str) -> bool:
    return _executable_available(executable) and _xquartz_available()


def _executable_available(executable: str) -> bool:
    if os.path.isabs(executable):
        return Path(executable).exists()
    return shutil.which(executable) is not None


def _xquartz_available() -> bool:
    return (
        Path("/Applications/Utilities/XQuartz.app").exists()
        or Path("/opt/X11/bin/Xquartz").exists()
        or bool(os.environ.get("DISPLAY"))
    )


def _format_float(value: float) -> str:
    return f"{float(value):.8g}"
