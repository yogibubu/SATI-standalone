from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class GaussianQuadrupolePromotion:
    xyzin: Path
    log_path: Path
    wrote_properties: bool
    property_count: int


def parse_gaussian_quadrupole_properties(path: Path | str) -> tuple:
    from matrix_qm import (
        atomic_number_from_isotope_or_atom,
        quadrupole_moment,
        quadrupole_property_records_from_efg,
        quadrupole_property_records_from_nqcc,
    )

    target = Path(path)
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    records = []
    direct_rows = [*_pickett_quadrupole_rows(lines), *_quadrupole_rows(lines)]
    for row in direct_rows:
        number = atomic_number_from_isotope_or_atom(
            isotope=row["isotope"],
            atom_symbol=row["atom_symbol"],
        )
        records.extend(
            quadrupole_property_records_from_nqcc(
                atom=row["atom"],
                atomic_number_value=number,
                nqcc_mhz=row["values"],
                program="Gaussian",
                source=target,
                isotope=row["isotope"],
                method=_parse_route_method(lines),
                level="",
                axes=row["axes"],
                status="raw",
                comment=row["comment"],
            )
        )
    if records:
        return tuple(records)
    for row in _efg_rows(target, lines):
        number = row["atomic_number"]
        if quadrupole_moment(number) is None:
            continue
        records.extend(
            quadrupole_property_records_from_efg(
                atom=row["atom"],
                atomic_number_value=number,
                efg_au=row["values"],
                program="Gaussian",
                source=target,
                method=_parse_route_method(lines),
                level="",
                axes="GAUSSIAN_EFG_TENSOR:3xx-rr,3yy-rr,3zz-rr,xy,xz,yz",
                status="raw",
                comment="Gaussian electric-field gradient converted by MATRIX; "
                "nqcc_convention=Pickett",
                nqcc_sign=-1.0,
                nqcc_convention="Pickett/Gaussian-EFG",
            )
        )
    return tuple(records)


def promote_gaussian_quadrupole_properties_to_xyzin(
    log_path: Path | str,
    xyzin: Path | str,
) -> GaussianQuadrupolePromotion:
    from matrix_qm import merge_properties_section

    source = Path(log_path)
    target = Path(xyzin)
    records = parse_gaussian_quadrupole_properties(source)
    if records:
        merge_properties_section(target, records)
    return GaussianQuadrupolePromotion(
        xyzin=target,
        log_path=source,
        wrote_properties=bool(records),
        property_count=len(records),
    )


def _quadrupole_rows(lines: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
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
        row = _parse_direct_nqcc_row(text)
        if row is not None:
            rows.append(row)
    return rows


def _pickett_quadrupole_rows(lines: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    idx = 0
    while idx < len(lines):
        upper = lines[idx].upper()
        if "NUCLEAR QUADRUPOLE COUPLING CONSTANTS" not in upper or "CHI" not in upper:
            idx += 1
            continue
        idx += 1
        while idx < len(lines):
            text = lines[idx].strip()
            upper = text.upper()
            if not text:
                idx += 1
                continue
            if upper.startswith(("DIPOLE", "ATOMS WITH", "----")):
                break
            header = re.match(r"^(?P<atom>\d+)\s+(?P<symbol>[A-Za-z]{1,2})\((?P<mass>\d+)\)", text)
            if header is None:
                idx += 1
                continue
            components: dict[str, float] = {}
            for raw_component in lines[idx + 1 : idx + 4]:
                for key, value in re.findall(
                    r"\b([abc][abc])\s*=\s*([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[DEde][-+]?\d+)?)",
                    raw_component,
                ):
                    components[key.lower()] = _float(value)
            if {"aa", "bb", "cc"}.issubset(components):
                symbol = header.group("symbol").capitalize()
                rows.append(
                    {
                        "atom": int(header.group("atom")),
                        "isotope": f"{int(header.group('mass'))}{symbol}",
                        "atom_symbol": symbol,
                        "values": (
                            components["aa"],
                            components["bb"],
                            components["cc"],
                            components.get("ab", components.get("ba", 0.0)),
                            components.get("ac", components.get("ca", 0.0)),
                            components.get("bc", components.get("cb", 0.0)),
                        ),
                        "axes": "PICKETT:chi_aa,chi_bb,chi_cc,chi_ab,chi_ac,chi_bc",
                        "comment": "Gaussian output=pickett nuclear quadrupole coupling constants in MHz",
                    }
                )
            idx += 4
        idx += 1
    return rows


def _efg_rows(path: Path, lines: list[str]) -> list[dict[str, object]]:
    from matrix_chem.geometry_io import GeometryParseError
    from matrix_chem.topology.elements import atomic_number

    from .parsers import read_gaussian_log_geometry

    try:
        atoms = read_gaussian_log_geometry(path).atoms
    except GeometryParseError:
        atoms = ()
    offdiag: dict[int, tuple[float, float, float]] = {}
    diag: dict[int, tuple[float, float, float]] = {}
    idx = 0
    while idx < len(lines):
        upper = lines[idx].upper()
        if "XY" in upper and "XZ" in upper and "YZ" in upper:
            idx = _collect_gaussian_tensor_rows(lines, idx + 1, offdiag)
            continue
        if "3XX-RR" in upper and "3YY-RR" in upper and "3ZZ-RR" in upper:
            idx = _collect_gaussian_tensor_rows(lines, idx + 1, diag)
            continue
        idx += 1
    rows = []
    for atom, diagonal in sorted(diag.items()):
        atom_symbol = atoms[atom - 1] if 0 < atom <= len(atoms) else ""
        number = atomic_number(atom_symbol) if atom_symbol else None
        if number is None:
            continue
        xy, xz, yz = offdiag.get(atom, (0.0, 0.0, 0.0))
        rows.append(
            {
                "atom": atom,
                "atomic_number": int(number),
                "values": (*diagonal, xy, xz, yz),
            }
        )
    return rows


def _collect_gaussian_tensor_rows(
    lines: list[str],
    start: int,
    out: dict[int, tuple[float, float, float]],
) -> int:
    idx = start
    dash_seen = False
    while idx < len(lines):
        text = lines[idx].strip()
        if set(text) == {"-"}:
            if dash_seen and out:
                return idx + 1
            dash_seen = True
            idx += 1
            continue
        row = _parse_gaussian_atom_tensor_row(text)
        if row is not None:
            atom, values = row
            out[atom] = values
        elif out and text:
            return idx
        idx += 1
    return idx


def _parse_gaussian_atom_tensor_row(text: str) -> tuple[int, tuple[float, float, float]] | None:
    parts = text.split()
    if len(parts) < 5 or not parts[0].isdigit() or parts[1].upper() != "ATOM":
        return None
    values = _first_floats(parts[2:], count=3)
    if len(values) != 3:
        return None
    return int(parts[0]), (values[0], values[1], values[2])


def _parse_direct_nqcc_row(text: str) -> dict[str, object] | None:
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
    if parts[0].upper() in {"NQCC", "CHI", "ATOM"}:
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
    axes = "GAUSSIAN:chi_xx,chi_yy,chi_zz"
    if len(values) >= 6:
        axes += ",chi_xy,chi_xz,chi_yz"
    return {
        "atom": atom,
        "isotope": isotope,
        "atom_symbol": atom_symbol,
        "values": tuple(values[:6] if len(values) >= 6 else values[:3]),
        "axes": axes,
        "comment": "Gaussian nuclear quadrupole coupling constants in MHz",
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


def _float(token: str) -> float:
    return float(token.replace("D", "E").replace("d", "e"))


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
    return upper.startswith(("----", "====", "DIPOLE", "THERMOCHEMISTRY", "NORMAL TERMINATION"))


def _parse_route_method(lines: list[str]) -> str:
    for raw in lines:
        text = raw.strip()
        if not text.startswith("#"):
            continue
        route = text.lstrip("#").strip()
        for token in route.split():
            if "/" in token:
                return token
    return ""
