from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re

from matrix_chem import get_default_isotope, get_isotope
from matrix_chem.topology.elements import atomic_number, atomic_symbol

from .properties import PropertyRecord


EFG_AU_TO_NQCC_MHZ_PER_BARN = 234.9647
QUADRUPOLE_MOMENT_SOURCE = "matrix-isotopes-table:femtometer2-to-barn"


@dataclass(frozen=True)
class QuadrupoleMoment:
    atomic_number: int
    mass_number: int
    isotope: str
    spin: float
    quadrupole_barn: float
    source: str = QUADRUPOLE_MOMENT_SOURCE


@dataclass(frozen=True)
class QuadrupoleTensor:
    values: tuple[float, ...]
    unit: str
    axes: str

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.values)
        if len(values) not in {3, 6}:
            raise ValueError("quadrupole tensors need 3 diagonal or 6 symmetric components")
        object.__setattr__(self, "values", values)

    @property
    def padded_values(self) -> tuple[float, float, float, float, float, float]:
        if len(self.values) == 6:
            return (
                self.values[0],
                self.values[1],
                self.values[2],
                self.values[3],
                self.values[4],
                self.values[5],
            )
        return (self.values[0], self.values[1], self.values[2], 0.0, 0.0, 0.0)


def parse_isotope_label(label: str) -> tuple[int, int, str]:
    clean = label.strip()
    if not clean:
        raise ValueError("isotope label cannot be empty")
    match = re.match(r"^(?P<a>\d+)\s*[-_ ]?\s*(?P<symbol>[A-Za-z]{1,2})$", clean)
    if match is None:
        match = re.match(r"^(?P<symbol>[A-Za-z]{1,2})\s*[-_ ]?\s*(?P<a>\d+)$", clean)
    if match is None:
        raise ValueError(f"invalid isotope label: {label}")
    mass_number = int(match.group("a"))
    symbol = match.group("symbol").capitalize()
    number = atomic_number(symbol)
    if number is None:
        raise ValueError(f"unknown isotope element: {label}")
    return int(number), mass_number, f"{mass_number}{atomic_symbol(int(number))}"


def quadrupole_moment(
    atomic_number_value: int,
    *,
    mass_number: int | None = None,
) -> QuadrupoleMoment | None:
    isotope = (
        get_default_isotope(int(atomic_number_value))
        if mass_number is None
        else get_isotope(int(atomic_number_value), int(mass_number))
    )
    if isotope is None or isotope.spin2 < 2:
        return None
    quadrupole_barn = _isotope_table_quadrupole_to_barn(float(isotope.quadrupole))
    if math.isclose(quadrupole_barn, 0.0, abs_tol=1.0e-15):
        return None
    return QuadrupoleMoment(
        atomic_number=int(isotope.Z),
        mass_number=int(isotope.A),
        isotope=f"{int(isotope.A)}{atomic_symbol(int(isotope.Z))}",
        spin=0.5 * float(isotope.spin2),
        quadrupole_barn=quadrupole_barn,
    )


def quadrupole_moment_from_label(label: str) -> QuadrupoleMoment | None:
    number, mass_number, _ = parse_isotope_label(label)
    return quadrupole_moment(number, mass_number=mass_number)


def efg_to_nqcc_mhz(
    efg_au: tuple[float, ...],
    moment: QuadrupoleMoment,
    *,
    sign: float = 1.0,
) -> tuple[float, ...]:
    factor = float(sign) * EFG_AU_TO_NQCC_MHZ_PER_BARN * moment.quadrupole_barn
    return tuple(factor * float(value) for value in efg_au)


def efg_asymmetry_parameter(efg_diagonal_au: tuple[float, float, float]) -> float:
    ordered = sorted((float(value) for value in efg_diagonal_au), key=lambda value: abs(value))
    v_xx, v_yy, v_zz = ordered[0], ordered[1], ordered[2]
    if math.isclose(v_zz, 0.0, abs_tol=1.0e-15):
        return 0.0
    return (v_xx - v_yy) / v_zz


def quadrupole_property_records_from_efg(
    *,
    atom: int | None,
    atomic_number_value: int,
    efg_au: tuple[float, ...],
    program: str,
    source: Path | str,
    isotope: str = "",
    method: str = "",
    level: str = "",
    axes: str = "MOLECULAR:xx,yy,zz,xy,xz,yz",
    status: str = "raw",
    comment: str = "",
    nqcc_sign: float = 1.0,
    nqcc_convention: str = "program",
) -> tuple[PropertyRecord, ...]:
    moment = (
        quadrupole_moment_from_label(isotope)
        if isotope
        else quadrupole_moment(int(atomic_number_value))
    )
    target_id = _target_id(atom=atom, isotope=moment.isotope if moment else isotope)
    source_text = str(source)
    efg_tensor = QuadrupoleTensor(efg_au, unit="a.u.", axes=axes)
    records = [
        PropertyRecord(
            name="ELECTRIC_FIELD_GRADIENT",
            target="ATOM",
            target_id=target_id,
            atom=atom,
            isotope=moment.isotope if moment else isotope,
            value=efg_tensor.values,
            unit=efg_tensor.unit,
            axes=efg_tensor.axes,
            program=program,
            method=method,
            level=level,
            source=source_text,
            status=status,
            comment=comment,
        )
    ]
    if moment is not None:
        records.append(
            PropertyRecord(
                name="NUCLEAR_QUADRUPOLE_COUPLING",
                target="ATOM",
                target_id=target_id,
                atom=atom,
                isotope=moment.isotope,
                value=efg_to_nqcc_mhz(efg_tensor.values, moment, sign=nqcc_sign),
                unit="MHz",
                axes=efg_tensor.axes,
                program=program,
                method=method,
                level=level,
                source=source_text,
                status="converted",
                conversion=(
                    f"EFG_AU_TO_MHZ;factor={EFG_AU_TO_NQCC_MHZ_PER_BARN:.7g};"
                    f"sign={float(nqcc_sign):.12g};Q_barn={moment.quadrupole_barn:.12g};"
                    f"source={moment.source};convention={nqcc_convention}"
                ),
                comment=comment,
            )
        )
        if len(efg_tensor.values) >= 3:
            records.append(
                PropertyRecord(
                    name="NUCLEAR_QUADRUPOLE_ASYMMETRY",
                    target="ATOM",
                    target_id=target_id,
                    atom=atom,
                    isotope=moment.isotope,
                    value=(efg_asymmetry_parameter(efg_tensor.values[:3]),),
                    unit="dimensionless",
                    axes="PAS:eta=(Vxx-Vyy)/Vzz",
                    program=program,
                    method=method,
                    level=level,
                    source=source_text,
                    status="derived",
                    conversion="from-electric-field-gradient-diagonal",
                    comment=comment,
                )
            )
    return tuple(records)


def quadrupole_property_records_from_nqcc(
    *,
    atom: int | None,
    atomic_number_value: int,
    nqcc_mhz: tuple[float, ...],
    program: str,
    source: Path | str,
    isotope: str = "",
    method: str = "",
    level: str = "",
    axes: str = "MOLECULAR:chi_xx,chi_yy,chi_zz,chi_xy,chi_xz,chi_yz",
    status: str = "raw",
    comment: str = "",
) -> tuple[PropertyRecord, ...]:
    moment = (
        quadrupole_moment_from_label(isotope)
        if isotope
        else quadrupole_moment(int(atomic_number_value))
    )
    isotope_label = moment.isotope if moment else isotope
    return (
        PropertyRecord(
            name="NUCLEAR_QUADRUPOLE_COUPLING",
            target="ATOM",
            target_id=_target_id(atom=atom, isotope=isotope_label),
            atom=atom,
            isotope=isotope_label,
            value=QuadrupoleTensor(nqcc_mhz, unit="MHz", axes=axes).values,
            unit="MHz",
            axes=axes,
            program=program,
            method=method,
            level=level,
            source=str(source),
            status=status,
            comment=comment,
        ),
    )


def infer_unique_quadrupolar_atom(atoms: tuple[str, ...] | list[str]) -> tuple[int, int] | None:
    candidates: list[tuple[int, int]] = []
    for idx, atom in enumerate(atoms, start=1):
        number = atomic_number(atom)
        if number is None:
            continue
        if quadrupole_moment(int(number)) is not None:
            candidates.append((idx, int(number)))
    if len(candidates) == 1:
        return candidates[0]
    return None


def atomic_number_from_isotope_or_atom(
    *,
    isotope: str = "",
    atom_symbol: str = "",
) -> int:
    if isotope:
        number, _, _ = parse_isotope_label(isotope)
        return number
    number = atomic_number(atom_symbol)
    if number is None:
        raise ValueError(f"cannot determine atomic number for {atom_symbol or isotope}")
    return int(number)


def _isotope_table_quadrupole_to_barn(value: float) -> float:
    return value / 100.0


def _target_id(*, atom: int | None, isotope: str) -> str:
    parts = []
    if atom is not None:
        parts.append(f"atom{atom}")
    if isotope:
        parts.append(isotope)
    return ":".join(parts)
