from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class OrcaQuadrupolePromotion:
    xyzin: Path
    output_path: Path
    wrote_properties: bool
    property_count: int


def parse_orca_quadrupole_properties(path: Path | str) -> tuple:
    target = Path(path)
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    direct = _direct_nqcc_records(target, lines)
    if direct:
        return direct
    hyperfine = _hyperfine_efg_records(target, lines)
    if hyperfine:
        return hyperfine
    return _efg_records(target, lines)


def promote_orca_quadrupole_properties_to_xyzin(
    output: Path | str,
    xyzin: Path | str,
) -> OrcaQuadrupolePromotion:
    from matrix_qm import merge_properties_section

    source = Path(output)
    target = Path(xyzin)
    records = parse_orca_quadrupole_properties(source)
    if records:
        merge_properties_section(target, records)
    return OrcaQuadrupolePromotion(
        xyzin=target,
        output_path=source,
        wrote_properties=bool(records),
        property_count=len(records),
    )


def _direct_nqcc_records(path: Path, lines: list[str]) -> tuple:
    from matrix_qm import atomic_number_from_isotope_or_atom, quadrupole_property_records_from_nqcc

    records = []
    in_section = False
    for raw in lines:
        text = raw.strip()
        upper = text.upper()
        if "QUADRUPOLE" in upper and ("COUPLING" in upper or "NQCC" in upper):
            in_section = True
            continue
        if in_section and not text:
            continue
        if in_section and _section_ended(upper):
            in_section = False
            continue
        if not in_section and not upper.startswith(("NQCC", "CHI")):
            continue
        row = _parse_tensor_row(text)
        if row is None:
            continue
        number = atomic_number_from_isotope_or_atom(
            isotope=row["isotope"],
            atom_symbol=row["atom_symbol"],
        )
        records.extend(
            quadrupole_property_records_from_nqcc(
                atom=row["atom"],
                atomic_number_value=number,
                nqcc_mhz=row["values"],
                program="ORCA",
                source=path,
                isotope=row["isotope"],
                axes="ORCA:chi_xx,chi_yy,chi_zz,chi_xy,chi_xz,chi_yz",
                status="raw",
                comment="ORCA nuclear quadrupole coupling constants in MHz",
            )
        )
    return tuple(records)


def _hyperfine_efg_records(path: Path, lines: list[str]) -> tuple:
    from matrix_qm import atomic_number_from_isotope_or_atom, quadrupole_property_records_from_efg

    records = []
    current: dict[str, object] | None = None
    nucleus_re = re.compile(
        r"Nucleus\s+(?P<zero>\d+)(?P<symbol>[A-Za-z]{1,2})\b.*?"
        r"Isotope=\s*(?P<mass>\d+)",
        re.I,
    )
    for raw in lines:
        text = raw.strip()
        match = nucleus_re.search(text)
        if match is not None:
            symbol = match.group("symbol").capitalize()
            current = {
                "atom": int(match.group("zero")) + 1,
                "atom_symbol": symbol,
                "isotope": f"{int(match.group('mass'))}{symbol}",
            }
            continue
        if current is None or not text.startswith("V(Tot)"):
            continue
        values = _first_floats(text.split()[1:], count=3)
        if len(values) != 3:
            continue
        number = atomic_number_from_isotope_or_atom(
            isotope=str(current["isotope"]),
            atom_symbol=str(current["atom_symbol"]),
        )
        records.extend(
            quadrupole_property_records_from_efg(
                atom=int(current["atom"]),
                atomic_number_value=number,
                efg_au=tuple(values),
                program="ORCA",
                source=path,
                isotope=str(current["isotope"]),
                axes="ORCA_EFG_PAS:Vxx,Vyy,Vzz",
                status="raw",
                comment="ORCA EPRNMR V(Tot) electric-field gradient converted by MATRIX; "
                "nqcc_convention=Pickett",
                nqcc_sign=1.0,
                nqcc_convention="Pickett/ORCA-EFG",
            )
        )
        current = None
    return tuple(records)


def _efg_records(path: Path, lines: list[str]) -> tuple:
    from matrix_qm import atomic_number_from_isotope_or_atom, quadrupole_property_records_from_efg

    records = []
    in_section = False
    for raw in lines:
        text = raw.strip()
        upper = text.upper()
        if "ELECTRIC FIELD GRADIENT" in upper or "EFG" in upper and "A.U" in upper:
            in_section = True
            continue
        if in_section and not text:
            continue
        if in_section and _section_ended(upper):
            in_section = False
            continue
        if not in_section and not upper.startswith("EFG"):
            continue
        row = _parse_tensor_row(text)
        if row is None:
            continue
        number = atomic_number_from_isotope_or_atom(
            isotope=row["isotope"],
            atom_symbol=row["atom_symbol"],
        )
        records.extend(
            quadrupole_property_records_from_efg(
                atom=row["atom"],
                atomic_number_value=number,
                efg_au=row["values"],
                program="ORCA",
                source=path,
                isotope=row["isotope"],
                axes="ORCA_EFG:xx,yy,zz,xy,xz,yz",
                status="raw",
                comment="ORCA electric-field gradient converted by MATRIX; "
                "nqcc_convention=Pickett",
                nqcc_sign=1.0,
                nqcc_convention="Pickett/ORCA-EFG",
            )
        )
    return tuple(records)


def _parse_tensor_row(text: str) -> dict[str, object] | None:
    clean = (
        text.replace("=", " ")
        .replace(",", " ")
        .replace(":", " ")
        .replace("(", " ")
        .replace(")", " ")
    )
    parts = clean.split()
    if not parts:
        return None
    if parts[0].upper() in {"NQCC", "CHI", "EFG", "ATOM"}:
        parts = parts[1:]
    if len(parts) < 5 or not parts[0].isdigit():
        return None
    atom = int(parts[0])
    isotope = ""
    atom_symbol = ""
    idx = 1
    if idx < len(parts) and _looks_isotope(parts[idx]):
        isotope = _normalize_isotope(parts[idx])
        idx += 1
    elif idx < len(parts) and re.fullmatch(r"[A-Za-z]{1,2}", parts[idx]):
        atom_symbol = parts[idx].capitalize()
        idx += 1
        if idx < len(parts) and parts[idx].isdigit():
            isotope = f"{parts[idx]}{atom_symbol}"
            idx += 1
    values = _first_floats(parts[idx:], count=6)
    if len(values) < 3:
        return None
    return {
        "atom": atom,
        "isotope": isotope,
        "atom_symbol": atom_symbol,
        "values": tuple(values[:6] if len(values) >= 6 else values[:3]),
    }


def _first_floats(parts: list[str], *, count: int) -> list[float]:
    values: list[float] = []
    for part in parts:
        try:
            values.append(float(part.replace("D", "E").replace("d", "e")))
        except ValueError:
            continue
        if len(values) == count:
            break
    return values


def _looks_isotope(text: str) -> bool:
    return re.fullmatch(r"(?:\d+[A-Za-z]{1,2}|[A-Za-z]{1,2}\d+)", text) is not None


def _normalize_isotope(text: str) -> str:
    match = re.fullmatch(r"(\d+)([A-Za-z]{1,2})", text)
    if match is not None:
        return f"{int(match.group(1))}{match.group(2).capitalize()}"
    match = re.fullmatch(r"([A-Za-z]{1,2})(\d+)", text)
    if match is not None:
        return f"{int(match.group(2))}{match.group(1).capitalize()}"
    return text


def _section_ended(upper: str) -> bool:
    return upper.startswith(("----", "====", "CARTESIAN", "VIBRATIONAL", "ORCA TERMINATED"))
