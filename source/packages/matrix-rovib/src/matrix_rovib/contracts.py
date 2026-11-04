from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
import re

from matrix_core import (
    key_value_section_lines,
    normalize_key,
    parse_key_value_section,
    read_sectioned_lines,
    replace_section,
    section_content,
)


ORACLE_XYZ_ROTATIONAL_SCHEMA = "oracle.xyz.rotational.v1"
ORACLE_XYZ_VIBRATIONAL_SCHEMA = "oracle.xyz.vibrational.v1"
ORACLE_XYZ_DELTABVIB_SCHEMA = "oracle.xyz.deltabvib.v1"
ORACLE_XYZ_CORIOLIS_SCHEMA = "oracle.xyz.coriolis.v1"
ORACLE_XYZ_QCENT_SCHEMA = "oracle.xyz.qcent.v1"

@dataclass(frozen=True)
class RotationalSection:
    rotor_type: str = ""
    representation: str = ""
    point_group: str = ""
    watson_reduction: str = ""
    symmetry_number: int | None = None
    temperature_K: float | None = None
    pressure_atm: float | None = None
    A_MHz: float | None = None
    B_MHz: float | None = None
    C_MHz: float | None = None
    equilibrium_MHz: tuple[float | None, float | None, float | None] | None = None
    ground_state_MHz: tuple[float | None, float | None, float | None] | None = None
    constant_kind: str = ""
    constant_source: str = ""
    dipole_debye: tuple[float | None, float | None, float | None] | None = None
    q_rot: float | None = None
    delta_vib_MHz: tuple[float | None, float | None, float | None] | None = None
    quartic_distortion_MHz: tuple[float, ...] = ()
    sextic_distortion_MHz: tuple[float, ...] = ()
    octic_distortion_MHz: tuple[float, ...] = ()
    distortion_source: str = ""
    schema: str = ORACLE_XYZ_ROTATIONAL_SCHEMA


@dataclass(frozen=True)
class VibrationalSection:
    linear: bool | None = None
    nvib: int | None = None
    n_imag_like: int | None = None
    symmetry_group: str = ""
    frequencies_cm1: tuple[float, ...] = ()
    anharmonic_frequencies_cm1: tuple[float, ...] = ()
    ir_intensities_km_mol: tuple[float, ...] = ()
    raman_activities_A4_amu: tuple[float, ...] = ()
    vcd_rot_strengths: tuple[float, ...] = ()
    roa_intensities: tuple[float, ...] = ()
    anharmonic_ir_intensities_km_mol: tuple[float, ...] = ()
    anharmonic_raman_activities_A4_amu: tuple[float, ...] = ()
    anharmonic_vcd_rot_strengths: tuple[float, ...] = ()
    anharmonic_roa_intensities: tuple[float, ...] = ()
    chi_cm1: tuple[tuple[int, int, float], ...] = ()
    schema: str = ORACLE_XYZ_VIBRATIONAL_SCHEMA


@dataclass(frozen=True)
class DeltaBVibAlphaRow:
    mode: int
    a_MHz: float
    b_MHz: float
    c_MHz: float


@dataclass(frozen=True)
class DeltaBVibSection:
    delta_A_MHz: float | None = None
    delta_B_MHz: float | None = None
    delta_C_MHz: float | None = None
    available: bool = True
    source: str = ""
    reason: str = ""
    alpha_rows_MHz: tuple[DeltaBVibAlphaRow, ...] = ()
    excluded_modes: tuple[int, ...] = ()
    invert_imaginary_modes: bool = True
    schema: str = ORACLE_XYZ_DELTABVIB_SCHEMA


def parse_rotational_section(lines: Iterable[str]) -> RotationalSection:
    values = parse_key_value_section(lines)
    return RotationalSection(
        rotor_type=values.get("ROTOR_TYPE", ""),
        representation=values.get("REPRESENTATION", ""),
        point_group=values.get("POINT_GROUP", ""),
        watson_reduction=values.get("WATSON_REDUCTION", values.get("REDUCTION", "")),
        symmetry_number=_optional_int(values.get("SYMM_NUMBER") or values.get("SIGMA")),
        temperature_K=_optional_float(values.get("T_K")),
        pressure_atm=_optional_float(values.get("P_ATM")),
        A_MHz=_optional_float(values.get("A_MHZ")),
        B_MHz=_optional_float(values.get("B_MHZ")),
        C_MHz=_optional_float(values.get("C_MHZ")),
        equilibrium_MHz=_parse_axis_triplet(values, ("AE_MHZ", "BE_MHZ", "CE_MHZ")),
        ground_state_MHz=_parse_axis_triplet(values, ("A0_MHZ", "B0_MHZ", "C0_MHZ")),
        constant_kind=values.get("CONSTANT_KIND", ""),
        constant_source=values.get("CONSTANT_SOURCE", ""),
        dipole_debye=_parse_dipole(values),
        q_rot=_optional_float(values.get("Q_ROT")),
        delta_vib_MHz=_parse_delta_vib(values),
        quartic_distortion_MHz=_parse_watson_constants(values, order=4),
        sextic_distortion_MHz=_parse_watson_constants(values, order=6),
        octic_distortion_MHz=_parse_watson_constants(values, order=8),
        distortion_source=values.get("DISTORTION_SOURCE", ""),
        schema=values.get("SCHEMA", ORACLE_XYZ_ROTATIONAL_SCHEMA),
    )


def rotational_section_lines(section: RotationalSection) -> list[str]:
    values: dict[str, object] = {
        "ROTOR_TYPE": section.rotor_type or None,
        "REPRESENTATION": section.representation or None,
        "POINT_GROUP": section.point_group or None,
        "WATSON_REDUCTION": section.watson_reduction or None,
        "SYMM_NUMBER": section.symmetry_number,
        "T_K": _format_float(section.temperature_K),
        "P_ATM": _format_float(section.pressure_atm),
        "A_MHZ": _format_float(section.A_MHz),
        "B_MHZ": _format_float(section.B_MHz),
        "C_MHZ": _format_float(section.C_MHz),
        "CONSTANT_KIND": section.constant_kind or None,
        "CONSTANT_SOURCE": section.constant_source or None,
        "Q_ROT": _format_float(section.q_rot),
    }
    if section.dipole_debye is not None:
        a, b, c = section.dipole_debye
        values.update(
            {
                "DIPOLE_A_D": _format_float(a),
                "DIPOLE_B_D": _format_float(b),
                "DIPOLE_C_D": _format_float(c),
            }
        )
    if section.equilibrium_MHz is not None:
        a, b, c = section.equilibrium_MHz
        values.update(
            {
                "AE_MHZ": _format_float(a),
                "BE_MHZ": _format_float(b),
                "CE_MHZ": _format_float(c),
            }
        )
    if section.ground_state_MHz is not None:
        a, b, c = section.ground_state_MHz
        values.update(
            {
                "A0_MHZ": _format_float(a),
                "B0_MHZ": _format_float(b),
                "C0_MHZ": _format_float(c),
            }
        )
    if section.delta_vib_MHz is not None:
        a, b, c = section.delta_vib_MHz
        values.update(
            {
                "DVIBA_MHZ": _format_float(a),
                "DVIBB_MHZ": _format_float(b),
                "DVIBC_MHZ": _format_float(c),
            }
        )
    reduction = "A" if section.watson_reduction.strip().upper().startswith("A") else "S"
    quartic_names = (
        ("DELTA_J_MHZ", "DELTA_JK_MHZ", "DELTA_K_MHZ", "DELTA_SMALL_J_MHZ", "DELTA_SMALL_K_MHZ")
        if reduction == "A" else ("DJ_MHZ", "DJK_MHZ", "DK_MHZ", "D1_MHZ", "D2_MHZ")
    )
    sextic_names = (
        ("PHI_J_MHZ", "PHI_JK_MHZ", "PHI_KJ_MHZ", "PHI_K_MHZ", "PHI_SMALL_J_MHZ", "PHI_SMALL_JK_MHZ", "PHI_SMALL_K_MHZ")
        if reduction == "A" else ("H_J_MHZ", "H_JK_MHZ", "H_KJ_MHZ", "H_K_MHZ", "H1_MHZ", "H2_MHZ", "H3_MHZ")
    )
    octic_names = (
        (
            "L_J_MHZ", "L_JJK_MHZ", "L_JK_MHZ", "L_KKJ_MHZ", "L_K_MHZ",
            "L_SMALL_J_MHZ", "L_SMALL_JK_MHZ", "L_SMALL_KJ_MHZ", "L_SMALL_K_MHZ",
        )
        if reduction == "A" else (
            "L_J_MHZ", "L_JJK_MHZ", "L_JK_MHZ", "L_KKJ_MHZ", "L_K_MHZ",
            "L1_MHZ", "L2_MHZ", "L3_MHZ", "L4_MHZ",
        )
    )
    values["DISTORTION_SOURCE"] = section.distortion_source or None
    values.update({name: _format_float(value) for name, value in zip(quartic_names, section.quartic_distortion_MHz)})
    values.update({name: _format_float(value) for name, value in zip(sextic_names, section.sextic_distortion_MHz)})
    values.update({name: _format_float(value) for name, value in zip(octic_names, section.octic_distortion_MHz)})
    return key_value_section_lines(
        ORACLE_XYZ_ROTATIONAL_SCHEMA,
        values,
        key_order=(
            "ROTOR_TYPE",
            "REPRESENTATION",
            "POINT_GROUP",
            "WATSON_REDUCTION",
            "SYMM_NUMBER",
            "T_K",
            "P_ATM",
            "A_MHZ",
            "B_MHZ",
            "C_MHZ",
            "AE_MHZ",
            "BE_MHZ",
            "CE_MHZ",
            "A0_MHZ",
            "B0_MHZ",
            "C0_MHZ",
            "CONSTANT_KIND",
            "CONSTANT_SOURCE",
            "DIPOLE_A_D",
            "DIPOLE_B_D",
            "DIPOLE_C_D",
            "DVIBA_MHZ",
            "DVIBB_MHZ",
            "DVIBC_MHZ",
            "Q_ROT",
            "DISTORTION_SOURCE",
            *quartic_names,
            *sextic_names,
            *octic_names,
        ),
    )


def read_rotational_section(path: Path) -> RotationalSection:
    return parse_rotational_section(section_content(read_sectioned_lines(Path(path)), "ROTATIONAL"))


def write_rotational_section(path: Path, section: RotationalSection) -> None:
    replace_section(Path(path), "ROTATIONAL", rotational_section_lines(section))


def resolved_rotational_constants_MHz(
    section: RotationalSection,
    *,
    state: str = "active",
    deltabvib: DeltaBVibSection | None = None,
) -> tuple[float | None, float | None, float | None]:
    """Resolve equilibrium or ground-state constants without double applying DeltaBVib.

    Legacy sections containing only A/B/C retain their historical interpretation:
    they are treated as equilibrium constants when a DeltaBVib payload is present,
    otherwise they are the active constants as written.
    """
    target = state.strip().lower().replace("-", "_")
    legacy = (section.A_MHz, section.B_MHz, section.C_MHz)
    kind = section.constant_kind.strip().lower().replace("-", "_")
    equilibrium = section.equilibrium_MHz
    ground = section.ground_state_MHz
    if target in {"equilibrium", "e", "be"}:
        if equilibrium is not None:
            return equilibrium
        if kind in {"equilibrium", "e", "be"} or not kind:
            return legacy
        return (None, None, None)
    if target not in {"active", "ground", "ground_state", "zero", "b0", "effective"}:
        raise ValueError(f"unsupported rotational constant state: {state}")
    if target == "active" and kind in {"effective", "ground", "ground_state", "zero", "b0"}:
        return ground if ground is not None else legacy
    if ground is not None:
        return ground
    if kind in {"effective", "ground", "ground_state", "zero", "b0"}:
        return legacy
    base = equilibrium if equilibrium is not None else legacy
    delta = section.delta_vib_MHz
    if delta is None and deltabvib is not None and deltabvib.available:
        delta = (deltabvib.delta_A_MHz, deltabvib.delta_B_MHz, deltabvib.delta_C_MHz)
    if delta is None:
        return base
    return tuple(
        None if value is None else float(value) + (0.0 if correction is None else float(correction))
        for value, correction in zip(base, delta)
    )


def parse_vibrational_section(lines: Iterable[str]) -> VibrationalSection:
    raw_lines = list(lines)
    values = parse_key_value_section(raw_lines)
    return VibrationalSection(
        linear=_optional_bool(values.get("LINEAR")),
        nvib=_optional_int(values.get("NVIB")),
        n_imag_like=_optional_int(values.get("N_IMAG_LIKE")),
        symmetry_group=values.get("SYMMETRY_GROUP", ""),
        frequencies_cm1=tuple(_number_list(values.get("FREQ_CM1") or values.get("FREQUENCIES"))),
        anharmonic_frequencies_cm1=tuple(
            _number_list(values.get("ANHARMONIC_FREQ_CM1") or values.get("ANH_FREQ_CM1"))
        ),
        ir_intensities_km_mol=tuple(
            _number_list(values.get("IR_INTEN_KM_MOL") or values.get("IR_INTEN"))
        ),
        raman_activities_A4_amu=tuple(
            _number_list(values.get("RAMAN_ACT_A4_AMU") or values.get("RAMAN_ACT"))
        ),
        vcd_rot_strengths=tuple(
            _number_list(values.get("VCD_ROT_STRENGTH") or values.get("VCD_ROT"))
        ),
        roa_intensities=tuple(_number_list(values.get("ROA_INTEN") or values.get("ROA"))),
        anharmonic_ir_intensities_km_mol=tuple(
            _number_list(values.get("ANHARMONIC_IR_INTEN_KM_MOL") or values.get("ANH_IR_INTEN"))
        ),
        anharmonic_raman_activities_A4_amu=tuple(
            _number_list(values.get("ANHARMONIC_RAMAN_ACT_A4_AMU") or values.get("ANH_RAMAN_ACT"))
        ),
        anharmonic_vcd_rot_strengths=tuple(
            _number_list(values.get("ANHARMONIC_VCD_ROT_STRENGTH") or values.get("ANH_VCD_ROT"))
        ),
        anharmonic_roa_intensities=tuple(
            _number_list(values.get("ANHARMONIC_ROA_INTEN") or values.get("ANH_ROA"))
        ),
        chi_cm1=tuple(_parse_chi_block(raw_lines)),
        schema=values.get("SCHEMA", ORACLE_XYZ_VIBRATIONAL_SCHEMA),
    )


def vibrational_section_lines(section: VibrationalSection) -> list[str]:
    values: dict[str, object] = {
        "LINEAR": None if section.linear is None else int(bool(section.linear)),
        "NVIB": section.nvib,
        "N_IMAG_LIKE": section.n_imag_like,
        "SYMMETRY_GROUP": section.symmetry_group or None,
        "FREQ_CM1": _format_float_list(section.frequencies_cm1),
        "ANHARMONIC_FREQ_CM1": _format_float_list(section.anharmonic_frequencies_cm1),
        "IR_INTEN_KM_MOL": _format_float_list(section.ir_intensities_km_mol),
        "RAMAN_ACT_A4_AMU": _format_float_list(section.raman_activities_A4_amu),
        "VCD_ROT_STRENGTH": _format_float_list(section.vcd_rot_strengths),
        "ROA_INTEN": _format_float_list(section.roa_intensities),
        "ANHARMONIC_IR_INTEN_KM_MOL": _format_float_list(section.anharmonic_ir_intensities_km_mol),
        "ANHARMONIC_RAMAN_ACT_A4_AMU": _format_float_list(
            section.anharmonic_raman_activities_A4_amu
        ),
        "ANHARMONIC_VCD_ROT_STRENGTH": _format_float_list(section.anharmonic_vcd_rot_strengths),
        "ANHARMONIC_ROA_INTEN": _format_float_list(section.anharmonic_roa_intensities),
    }
    lines = key_value_section_lines(
        ORACLE_XYZ_VIBRATIONAL_SCHEMA,
        values,
        key_order=(
            "LINEAR",
            "NVIB",
            "N_IMAG_LIKE",
            "SYMMETRY_GROUP",
            "FREQ_CM1",
            "ANHARMONIC_FREQ_CM1",
            "IR_INTEN_KM_MOL",
            "RAMAN_ACT_A4_AMU",
            "VCD_ROT_STRENGTH",
            "ROA_INTEN",
            "ANHARMONIC_IR_INTEN_KM_MOL",
            "ANHARMONIC_RAMAN_ACT_A4_AMU",
            "ANHARMONIC_VCD_ROT_STRENGTH",
            "ANHARMONIC_ROA_INTEN",
        ),
    )
    if section.chi_cm1:
        lines.append("CHI_CM1 = [")
        lines.extend(f"{i:d} {j:d} {value:.8f}" for i, j, value in section.chi_cm1)
        lines.append("]")
    return lines


def read_vibrational_section(path: Path) -> VibrationalSection:
    content = section_content(read_sectioned_lines(Path(path)), "VIBRATIONAL")
    return parse_vibrational_section(content)


def write_vibrational_section(path: Path, section: VibrationalSection) -> None:
    replace_section(Path(path), "VIBRATIONAL", vibrational_section_lines(section))


def parse_deltabvib_section(lines: Iterable[str]) -> DeltaBVibSection:
    raw_lines = list(lines)
    values = parse_key_value_section(raw_lines)
    return DeltaBVibSection(
        delta_A_MHz=_optional_float(values.get("DVIBA_MHZ") or values.get("DELTA_A_MHZ")),
        delta_B_MHz=_optional_float(values.get("DVIBB_MHZ") or values.get("DELTA_B_MHZ")),
        delta_C_MHz=_optional_float(values.get("DVIBC_MHZ") or values.get("DELTA_C_MHZ")),
        available=(
            _optional_bool(values.get("AVAILABLE")) if values.get("AVAILABLE") is not None else True
        ),
        source=values.get("SOURCE", ""),
        reason=values.get("REASON", ""),
        alpha_rows_MHz=tuple(_parse_alpha_rows(raw_lines)),
        excluded_modes=tuple(int(value) for value in _number_list(values.get("EXCLUDED_MODES"))),
        invert_imaginary_modes=(
            _optional_bool(values.get("INVERT_IMAGINARY_MODES"))
            if values.get("INVERT_IMAGINARY_MODES") is not None
            else True
        ),
        schema=values.get("SCHEMA", ORACLE_XYZ_DELTABVIB_SCHEMA),
    )


def deltabvib_section_lines(section: DeltaBVibSection) -> list[str]:
    values: dict[str, object] = {
        "AVAILABLE": int(bool(section.available)),
        "SOURCE": section.source or None,
        "REASON": section.reason or None,
        "DVIBA_MHZ": _format_float(section.delta_A_MHz),
        "DVIBB_MHZ": _format_float(section.delta_B_MHz),
        "DVIBC_MHZ": _format_float(section.delta_C_MHz),
        "INVERT_IMAGINARY_MODES": int(bool(section.invert_imaginary_modes)),
        "EXCLUDED_MODES": (
            " ".join(str(mode) for mode in section.excluded_modes)
            if section.excluded_modes
            else None
        ),
    }
    lines = key_value_section_lines(
        ORACLE_XYZ_DELTABVIB_SCHEMA,
        values,
        key_order=(
            "AVAILABLE",
            "SOURCE",
            "REASON",
            "DVIBA_MHZ",
            "DVIBB_MHZ",
            "DVIBC_MHZ",
            "INVERT_IMAGINARY_MODES",
            "EXCLUDED_MODES",
        ),
    )
    if section.alpha_rows_MHz:
        lines.append("ALPHA_MHZ = [")
        lines.extend(
            f"{row.mode:d} {row.a_MHz:.8f} {row.b_MHz:.8f} {row.c_MHz:.8f}"
            for row in section.alpha_rows_MHz
        )
        lines.append("]")
    return lines


def read_deltabvib_section(path: Path) -> DeltaBVibSection:
    content = section_content(read_sectioned_lines(Path(path)), "DELTABVIB")
    return parse_deltabvib_section(content)


def write_deltabvib_section(path: Path, section: DeltaBVibSection) -> None:
    replace_section(Path(path), "DELTABVIB", deltabvib_section_lines(section))


def _parse_delta_vib(
    values: dict[str, str],
) -> tuple[float | None, float | None, float | None] | None:
    a = _optional_float(values.get("DVIBA_MHZ") or values.get("DVIB_A_MHZ"))
    b = _optional_float(values.get("DVIBB_MHZ") or values.get("DVIB_B_MHZ"))
    c = _optional_float(values.get("DVIBC_MHZ") or values.get("DVIB_C_MHZ"))
    if a is None and b is None and c is None:
        return None
    return a, b, c


def _parse_watson_constants(values: dict[str, str], *, order: int) -> tuple[float, ...]:
    reduction = "A" if values.get("WATSON_REDUCTION", values.get("REDUCTION", "S")).strip().upper().startswith("A") else "S"
    if order == 4 and reduction == "A":
        aliases = (
            ("DELTA_J_MHZ", "DELJ_MHZ"), ("DELTA_JK_MHZ", "DELJK_MHZ"),
            ("DELTA_K_MHZ", "DELK_MHZ"), ("DELTA_SMALL_J_MHZ", "DELTAJ_MHZ"),
            ("DELTA_SMALL_K_MHZ", "DELTAK_MHZ"),
        )
    elif order == 4:
        aliases = (("DJ_MHZ",), ("DJK_MHZ",), ("DK_MHZ",), ("D1_MHZ",), ("D2_MHZ",))
    elif order == 6 and reduction == "A":
        aliases = (
            ("PHI_J_MHZ", "PHI_N_MHZ"), ("PHI_JK_MHZ", "PHI_NK_MHZ"),
            ("PHI_KJ_MHZ", "PHI_KN_MHZ"), ("PHI_K_MHZ",),
            ("PHI_SMALL_J_MHZ", "PHI_N_SMALL_MHZ"),
            ("PHI_SMALL_JK_MHZ", "PHI_NK_SMALL_MHZ"),
            ("PHI_SMALL_K_MHZ",),
        )
    elif order == 6:
        aliases = (
            ("H_J_MHZ", "H_N_MHZ"), ("H_JK_MHZ", "H_NK_MHZ"),
            ("H_KJ_MHZ", "H_KN_MHZ"), ("H_K_MHZ",),
            ("H1_MHZ",), ("H2_MHZ",), ("H3_MHZ",),
        )
    elif reduction == "A":
        aliases = (
            ("L_J_MHZ",), ("L_JJK_MHZ",), ("L_JK_MHZ",),
            ("L_KKJ_MHZ",), ("L_K_MHZ",), ("L_SMALL_J_MHZ",),
            ("L_SMALL_JK_MHZ",), ("L_SMALL_KJ_MHZ",), ("L_SMALL_K_MHZ",),
        )
    else:
        aliases = (
            ("L_J_MHZ",), ("L_JJK_MHZ",), ("L_JK_MHZ",),
            ("L_KKJ_MHZ",), ("L_K_MHZ",), ("L1_MHZ",), ("L2_MHZ",),
            ("L3_MHZ",), ("L4_MHZ",),
        )
    parsed = []
    present = False
    for names in aliases:
        value = next((_optional_float(values.get(name)) for name in names if values.get(name) is not None), None)
        present = present or value is not None
        parsed.append(0.0 if value is None else float(value))
    return tuple(parsed) if present else ()


def _parse_axis_triplet(
    values: dict[str, str], keys: tuple[str, str, str]
) -> tuple[float | None, float | None, float | None] | None:
    result = tuple(_optional_float(values.get(key)) for key in keys)
    if all(value is None for value in result):
        return None
    return result


def _parse_dipole(
    values: dict[str, str],
) -> tuple[float | None, float | None, float | None] | None:
    a = _optional_float(
        values.get("DIPOLE_A_D")
        or values.get("DIPOLE_A_DEBYE")
        or values.get("MU_A_D")
        or values.get("MU_A")
    )
    b = _optional_float(
        values.get("DIPOLE_B_D")
        or values.get("DIPOLE_B_DEBYE")
        or values.get("MU_B_D")
        or values.get("MU_B")
    )
    c = _optional_float(
        values.get("DIPOLE_C_D")
        or values.get("DIPOLE_C_DEBYE")
        or values.get("MU_C_D")
        or values.get("MU_C")
    )
    if a is None and b is None and c is None:
        return None
    return a, b, c


def _parse_chi_block(lines: list[str]) -> list[tuple[int, int, float]]:
    out: list[tuple[int, int, float]] = []
    in_chi = False
    for raw in lines:
        line = raw.strip()
        upper = line.upper()
        if upper.startswith("CHI_CM1"):
            in_chi = True
            continue
        if in_chi and "]" in line:
            break
        if not in_chi or not line:
            continue
        values = _number_list(line)
        if len(values) >= 3:
            out.append((int(values[0]), int(values[1]), float(values[2])))
    return out


def _parse_alpha_rows(lines: list[str]) -> list[DeltaBVibAlphaRow]:
    out: list[DeltaBVibAlphaRow] = []
    in_alpha = False
    for raw in lines:
        line = raw.strip()
        upper = line.upper()
        if upper.startswith("ALPHA_MHZ"):
            in_alpha = True
            continue
        if in_alpha and "]" in line:
            break
        if not in_alpha or not line:
            continue
        values = _number_list(line)
        if len(values) >= 4:
            out.append(
                DeltaBVibAlphaRow(
                    mode=int(values[0]),
                    a_MHz=float(values[1]),
                    b_MHz=float(values[2]),
                    c_MHz=float(values[3]),
                )
            )
    return out


def _number_list(text: str | None) -> list[float]:
    if not text:
        return []
    return [
        float(item.replace("D", "E").replace("d", "e"))
        for item in re.findall(r"[-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?", text)
    ]


def _optional_float(text: str | None) -> float | None:
    values = _number_list(text)
    return values[0] if values else None


def _optional_int(text: str | None) -> int | None:
    value = _optional_float(text)
    return None if value is None else int(value)


def _optional_bool(text: str | None) -> bool | None:
    if text is None:
        return None
    key = normalize_key(text)
    if key in {"1", "TRUE", "YES", "Y"}:
        return True
    if key in {"0", "FALSE", "NO", "N"}:
        return False
    value = _optional_int(text)
    return None if value is None else bool(value)


def _format_float(value: float | None) -> str | None:
    return None if value is None else f"{float(value):.8g}"


def _format_float_list(values: tuple[float, ...]) -> str | None:
    if not values:
        return None
    return " ".join(f"{float(value):.8f}" for value in values)
