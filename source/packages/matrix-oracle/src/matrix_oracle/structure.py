from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from matrix_core import read_sectioned_lines, section_content
from matrix_fragments import read_fragment_records

from .commands import OracleGuiCommand, avogadro_command, fragments_command, preprocess_command


@dataclass(frozen=True)
class StructureTable:
    title: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class StructureGuiState:
    xyzin: Path
    exists: bool
    topology_bonds: StructureTable
    topology_rings: StructureTable
    synthons: StructureTable
    fragments: StructureTable
    messages: tuple[str, ...] = ()

    @property
    def has_structure_sections(self) -> bool:
        return bool(self.topology_bonds.rows or self.topology_rings.rows or self.synthons.rows)


class OracleStructureController:
    def __init__(self, xyzin: Path | str | None = None) -> None:
        self.xyzin = None if xyzin is None else Path(xyzin)

    def set_xyzin(self, xyzin: Path | str | None) -> StructureGuiState | None:
        self.xyzin = None if xyzin is None else Path(xyzin)
        if self.xyzin is None:
            return None
        return self.state()

    def state(self) -> StructureGuiState:
        if self.xyzin is None:
            raise ValueError("no MATRIX xyzin project is loaded")
        return load_structure_gui_state(self.xyzin)

    def preprocess_command(
        self,
        source: Path | str,
        output: Path | str | None = None,
        *,
        source_kind: str = "auto",
    ) -> OracleGuiCommand:
        target = Path(output) if output is not None else default_preprocess_output(source)
        return preprocess_command(source, target, source_kind=source_kind)

    def avogadro_command(self, *, executable: str = "avogadro2") -> OracleGuiCommand:
        if self.xyzin is None:
            raise ValueError("no MATRIX xyzin project is loaded")
        return avogadro_command(self.xyzin, executable=executable)

    def fragments_command(self, action: str = "build") -> OracleGuiCommand:
        if self.xyzin is None:
            raise ValueError("no MATRIX xyzin project is loaded")
        return fragments_command(self.xyzin, action)


def default_preprocess_output(source: Path | str) -> Path:
    target = Path(source)
    return target.with_suffix(".xyzin")


def load_structure_gui_state(path: Path | str) -> StructureGuiState:
    target = Path(path)
    if not target.exists():
        empty = _empty_tables()
        return StructureGuiState(
            xyzin=target,
            exists=False,
            topology_bonds=empty[0],
            topology_rings=empty[1],
            synthons=empty[2],
            fragments=empty[3],
            messages=(f"Missing file: {target}",),
        )
    lines = read_sectioned_lines(target)
    messages: list[str] = []
    topology = section_content(lines, "TOPOLOGY")
    synthons = section_content(lines, "SYNTHONS")
    if not topology:
        messages.append("Missing #TOPOLOGY")
    if not synthons:
        messages.append("Missing #SYNTHONS")
    return StructureGuiState(
        xyzin=target,
        exists=True,
        topology_bonds=_topology_bond_table(topology),
        topology_rings=_topology_ring_table(topology),
        synthons=_synthon_table(synthons),
        fragments=_fragment_table(target),
        messages=tuple(messages),
    )


def _empty_tables() -> tuple[StructureTable, StructureTable, StructureTable, StructureTable]:
    return (
        StructureTable("Bonds", ("Atom i", "Atom j"), ()),
        StructureTable("Rings", ("Ring", "Size", "Atoms"), ()),
        StructureTable(
            "Synthons",
            (
                "Atom",
                "Element",
                "Zeff",
                "Charge",
                "Covalency",
                "Delocalization",
                "Strain",
                "Signature",
            ),
            (),
        ),
        StructureTable(
            "Fragments",
            ("ID", "Label", "Size", "Atoms", "Charge", "Multiplicity", "Center"),
            (),
        ),
    )


def _topology_bond_table(section: list[str]) -> StructureTable:
    component_rows: list[tuple[str, ...]] = []
    for line in _subsection(section, "BOND_ORDER_COMPONENTS"):
        upper = line.strip().upper()
        if upper == "NONE" or upper.startswith("COLUMNS "):
            continue
        parts = line.split()
        if len(parts) >= 7 and parts[0].isdigit() and parts[1].isdigit():
            # Read-only migration of the short-lived topology/multiplicity draft.
            component_rows.append(tuple((parts[0], parts[1], *parts[3:7])))
        elif len(parts) >= 6 and parts[0].isdigit() and parts[1].isdigit():
            component_rows.append(tuple(parts[:6]))
    if component_rows:
        return StructureTable(
            "Bonds",
            ("Atom i", "Atom j", "BO", "Sigma", "Pi", "Pi-Pi"),
            tuple(component_rows),
        )
    rows: list[tuple[str, ...]] = []
    for line in _subsection(section, "BONDS"):
        if line.strip().upper() == "NONE":
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            rows.append((parts[0], parts[1]))
    return StructureTable("Bonds", ("Atom i", "Atom j"), tuple(rows))


def _topology_ring_table(section: list[str]) -> StructureTable:
    rows: list[tuple[str, str, str]] = []
    for line in _subsection(section, "RINGS"):
        if line.strip().upper() == "NONE":
            continue
        parts = line.replace(",", " ").split()
        if not parts or not parts[0].isdigit():
            continue
        fields = _fields(parts[1:])
        atoms = fields.get("ATOMS", "")
        if not atoms:
            atoms = " ".join(_atoms_after_marker(parts, "ATOMS"))
        rows.append((parts[0], fields.get("SIZE", ""), atoms))
    return StructureTable("Rings", ("Ring", "Size", "Atoms"), tuple(rows))


def _synthon_table(section: list[str]) -> StructureTable:
    columns = (
        "Atom",
        "Element",
        "Zeff",
        "Charge",
        "Covalency",
        "Delocalization",
        "Strain",
        "Signature",
    )
    for raw in section:
        fields = raw.strip().split()
        if fields and fields[0].upper() == "COLUMNS":
            columns = tuple(field.replace("_", " ").title() for field in fields[1:])
            break
    rows: list[tuple[str, ...]] = []
    for raw in section:
        line = raw.strip()
        if not line or not line.split()[0].isdigit():
            continue
        parts = line.split(maxsplit=len(columns) - 1)
        if len(parts) < len(columns):
            continue
        rows.append(tuple(parts[: len(columns)]))
    return StructureTable(
        "Synthons",
        columns,
        tuple(rows),
    )


def _fragment_table(path: Path) -> StructureTable:
    rows = tuple(
        (
            record.identifier,
            record.label,
            str(len(record.atoms)),
            ",".join(str(atom) for atom in record.atoms),
            "" if record.charge is None else str(record.charge),
            "" if record.multiplicity is None else str(record.multiplicity),
            ", ".join(f"{value:g}" for value in record.center),
        )
        for record in read_fragment_records(path)
    )
    return StructureTable(
        "Fragments",
        ("ID", "Label", "Size", "Atoms", "Charge", "Multiplicity", "Center"),
        rows,
    )


def _subsection(section: list[str], name: str) -> list[str]:
    header = f"[{name.upper()}]"
    start = None
    for idx, line in enumerate(section):
        if line.strip().upper() == header:
            start = idx + 1
            break
    if start is None:
        return []
    out: list[str] = []
    for line in section[start:]:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            break
        if stripped:
            out.append(stripped)
    return out


def _fields(parts: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    idx = 0
    while idx < len(parts):
        token = parts[idx]
        if "=" not in token:
            idx += 1
            continue
        key, value = token.split("=", 1)
        if key.upper() == "ATOMS" and value:
            atoms = [value]
            idx += 1
            while idx < len(parts) and "=" not in parts[idx]:
                atoms.append(parts[idx])
                idx += 1
            result[key.upper()] = " ".join(atoms)
            continue
        result[key.upper()] = value
        idx += 1
    return result


def _atoms_after_marker(parts: list[str], marker: str) -> tuple[str, ...]:
    atoms: list[str] = []
    reading = False
    for token in parts:
        if token.upper().startswith(f"{marker.upper()}="):
            reading = True
            value = token.split("=", 1)[1]
            if value:
                atoms.append(value)
            continue
        if reading and "=" in token:
            break
        if reading:
            atoms.append(token)
    return tuple(atoms)
