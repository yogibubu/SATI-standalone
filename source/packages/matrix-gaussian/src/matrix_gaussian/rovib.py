from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from matrix_rovib import (
    DeltaBVibAlphaRow,
    DeltaBVibSection,
    RotationalSection,
    VibrationalSection,
    read_rotational_section,
    write_deltabvib_section,
    write_rotational_section,
    write_vibrational_section,
)

from .semidiagonal import (
    GaussianSemiDiagonalCubicRovibData,
    SemiDiagonalDeltaBVibResult,
    _parse_semidiagonal_cubic_rovib,
    compute_deltabvib_from_semidiagonal_cubic_data,
    read_gaussian_semidiagonal_cubic_rovib,
    read_gaussian_vibrot_diagnostics,
    semidiagonal_cubic_cm,
    semidiagonal_data_from_cartesian_cubic,
    semidiagonal_data_from_parent_normal_cubic,
)

__all__ = [
    "GaussianRovibData",
    "GaussianRovibPromotion",
    "GaussianSemiDiagonalCubicRovibData",
    "SemiDiagonalDeltaBVibResult",
    "compute_deltabvib_from_semidiagonal_cubic_data",
    "compute_deltavib_from_alpha",
    "parse_gaussian_rovib_log",
    "promote_gaussian_rovib_to_xyzin",
    "read_gaussian_semidiagonal_cubic_rovib",
    "read_gaussian_vibrot_diagnostics",
    "semidiagonal_cubic_cm",
    "semidiagonal_data_from_cartesian_cubic",
    "semidiagonal_data_from_parent_normal_cubic",
]


@dataclass(frozen=True)
class GaussianRovibData:
    log_path: Path
    vibrational: VibrationalSection | None
    rotational: RotationalSection | None
    deltabvib: DeltaBVibSection | None
    semidiagonal_cubic: GaussianSemiDiagonalCubicRovibData | None = None


@dataclass(frozen=True)
class GaussianRovibPromotion:
    xyzin: Path
    log_path: Path
    wrote_vibrational: bool
    wrote_rotational: bool
    wrote_deltabvib: bool
    wrote_semidiagonal_cubic: bool = False


def parse_gaussian_rovib_log(
    path: Path | str,
    *,
    invert_imaginary_modes: bool = True,
    exclude_modes: tuple[int, ...] = (),
) -> GaussianRovibData:
    target = Path(path)
    lines = target.read_text(encoding="utf-8", errors="ignore").splitlines()
    vibrational = _vibrational_section_from_log(lines)
    alpha_rows = tuple(_parse_vibro_rot_alpha_matrix(lines))
    semidiagonal = _parse_semidiagonal_cubic_rovib(lines, strict=False)
    deltabvib = _deltabvib_section_from_log(
        lines,
        alpha_rows=alpha_rows,
        semidiagonal=semidiagonal,
        frequencies_cm1=() if vibrational is None else vibrational.frequencies_cm1,
        invert_imaginary_modes=invert_imaginary_modes,
        exclude_modes=exclude_modes,
    )
    rotational = _rotational_section_from_log(lines, deltabvib=deltabvib)
    return GaussianRovibData(
        log_path=target,
        vibrational=vibrational,
        rotational=rotational,
        deltabvib=deltabvib,
        semidiagonal_cubic=semidiagonal,
    )


def promote_gaussian_rovib_to_xyzin(
    log_path: Path | str,
    xyzin: Path | str,
    *,
    write_vibrational: bool = True,
    write_rotational: bool = True,
    write_deltabvib: bool = True,
    write_semidiagonal_cubic: bool = True,
    invert_imaginary_modes: bool = True,
    exclude_modes: tuple[int, ...] = (),
) -> GaussianRovibPromotion:
    from matrix_qm import semidiagonal_cubic_section_from_gaussian_data
    from matrix_qm import write_semidiagonal_cubic_section

    target = Path(xyzin)
    data = parse_gaussian_rovib_log(
        log_path,
        invert_imaginary_modes=invert_imaginary_modes,
        exclude_modes=exclude_modes,
    )
    wrote_vib = False
    wrote_rot = False
    wrote_delta = False
    wrote_semidiagonal = False
    if write_vibrational and data.vibrational is not None:
        write_vibrational_section(target, data.vibrational)
        wrote_vib = True
    if write_deltabvib and data.deltabvib is not None:
        write_deltabvib_section(target, data.deltabvib)
        wrote_delta = True
    if write_semidiagonal_cubic and data.semidiagonal_cubic is not None:
        write_semidiagonal_cubic_section(
            target,
            semidiagonal_cubic_section_from_gaussian_data(
                data.semidiagonal_cubic,
                source="gaussian-semidiagonal-cubic-vibrot",
            ),
        )
        wrote_semidiagonal = True
    if write_rotational and data.rotational is not None:
        existing = read_rotational_section(target)
        write_rotational_section(target, _merge_rotational_sections(existing, data.rotational))
        wrote_rot = True
    return GaussianRovibPromotion(
        xyzin=target,
        log_path=data.log_path,
        wrote_vibrational=wrote_vib,
        wrote_rotational=wrote_rot,
        wrote_deltabvib=wrote_delta,
        wrote_semidiagonal_cubic=wrote_semidiagonal,
    )


def compute_deltavib_from_alpha(
    alpha_rows: tuple[DeltaBVibAlphaRow, ...],
    frequencies_cm1: tuple[float, ...] = (),
    *,
    invert_imaginary_modes: bool = True,
    exclude_modes: tuple[int, ...] = (),
) -> tuple[float, float, float] | None:
    if not alpha_rows:
        return None
    excluded = set(exclude_modes)
    sum_a = sum_b = sum_c = 0.0
    for row in alpha_rows:
        if row.mode in excluded:
            continue
        a, b, c = row.a_MHz, row.b_MHz, row.c_MHz
        if invert_imaginary_modes and row.mode - 1 < len(frequencies_cm1):
            if frequencies_cm1[row.mode - 1] < 0.0:
                a, b, c = -a, -b, -c
        sum_a += a
        sum_b += b
        sum_c += c
    return sum_a / 2.0, sum_b / 2.0, sum_c / 2.0


def _vibrational_section_from_log(lines: list[str]) -> VibrationalSection | None:
    freq, ir = _parse_harmonic_freq_ir_from_log(lines)
    if not freq:
        return None
    chi = _parse_anharmonic_x_matrix(lines)
    chi_rows: list[tuple[int, int, float]] = []
    if chi is not None:
        for i, row in enumerate(chi, start=1):
            for j, value in enumerate(row[:i], start=1):
                if value is not None:
                    chi_rows.append((i, j, float(value)))
    return VibrationalSection(
        nvib=len(freq),
        n_imag_like=sum(1 for value in freq if value < 0.0),
        frequencies_cm1=tuple(freq),
        ir_intensities_km_mol=tuple(ir) if ir and len(ir) == len(freq) else (),
        chi_cm1=tuple(chi_rows),
    )


def _deltabvib_section_from_log(
    lines: list[str],
    *,
    alpha_rows: tuple[DeltaBVibAlphaRow, ...],
    semidiagonal: GaussianSemiDiagonalCubicRovibData | None,
    frequencies_cm1: tuple[float, ...],
    invert_imaginary_modes: bool,
    exclude_modes: tuple[int, ...],
) -> DeltaBVibSection | None:
    source = "gaussian"
    delta = None
    reason = ""
    if semidiagonal is not None and not exclude_modes:
        try:
            delta = compute_deltabvib_from_semidiagonal_cubic_data(semidiagonal).total_MHz
            source = "gaussian-semidiagonal-cubic-vibrot"
        except ValueError as exc:
            reason = str(exc)
    if delta is None:
        rot_delta = _parse_deltabvib_from_rot_constants(lines)
        if rot_delta is not None:
            delta = rot_delta
            source = "gaussian-rotational-constants"
            reason = ""
    if delta is None:
        delta = compute_deltavib_from_alpha(
            alpha_rows,
            frequencies_cm1,
            invert_imaginary_modes=invert_imaginary_modes,
            exclude_modes=exclude_modes,
        )
        source = "gaussian-alpha" if delta is not None else "gaussian"
        reason = "" if delta is not None else "DeltaBvib not found in Gaussian log"
    if delta is None and not alpha_rows and semidiagonal is None:
        return None
    return DeltaBVibSection(
        delta_A_MHz=None if delta is None else delta[0],
        delta_B_MHz=None if delta is None else delta[1],
        delta_C_MHz=None if delta is None else delta[2],
        available=delta is not None,
        source=source,
        reason=reason,
        alpha_rows_MHz=alpha_rows,
        excluded_modes=exclude_modes,
        invert_imaginary_modes=invert_imaginary_modes,
    )


def _rotational_section_from_log(
    lines: list[str],
    *,
    deltabvib: DeltaBVibSection | None,
) -> RotationalSection | None:
    rot = _parse_rot_constants_mhz(lines)
    temperature = _parse_temperature_from_log(lines)
    point_group = _parse_point_group_from_log(lines) or ""
    dipole = _parse_dipole_debye_from_log(lines)

    def pick(label: str, suffixes: tuple[str, ...]) -> float | None:
        for suffix in suffixes:
            key = f"{label}{suffix}"
            if key in rot:
                return rot[key]
        return None

    equilibrium = tuple(pick(label, ("e",)) for label in "ABC")
    ground = tuple(pick(label, ("0", "00")) for label in "ABC")
    if all(value is None for value in equilibrium):
        equilibrium = None
    if all(value is None for value in ground):
        ground = None
    active = ground if ground is not None else equilibrium
    A, B, C = active if active is not None else (None, None, None)
    delta = None
    if deltabvib is not None and deltabvib.available:
        delta = (deltabvib.delta_A_MHz, deltabvib.delta_B_MHz, deltabvib.delta_C_MHz)
    if (
        A is None
        and B is None
        and C is None
        and temperature is None
        and not point_group
        and delta is None
        and dipole is None
    ):
        return None
    return RotationalSection(
        point_group=point_group,
        temperature_K=temperature,
        A_MHz=A,
        B_MHz=B,
        C_MHz=C,
        equilibrium_MHz=equilibrium,
        ground_state_MHz=ground,
        constant_kind="ground_state" if ground is not None else "equilibrium",
        constant_source="gaussian-rovib-log",
        dipole_debye=dipole,
        delta_vib_MHz=delta,
    )


def _parse_harmonic_freq_ir_from_log(lines: list[str]) -> tuple[list[float], list[float] | None]:
    header_idx = None
    for idx, line in enumerate(lines):
        if "Harmonic frequencies (cm**-1)" in line:
            header_idx = idx
    search = lines if header_idx is None else lines[header_idx + 1 :]
    freq: list[float] = []
    ir: list[float] = []
    for raw in search:
        text = raw.strip()
        if header_idx is not None:
            if text.startswith("----") and freq:
                break
            if text.startswith("----"):
                continue
            if text.startswith(("Thermochemistry", "Temperature")):
                break
        if "Frequencies --" in text:
            freq.extend(_numbers_after_marker(text, "--"))
            continue
        if text.startswith("IR Inten"):
            ir.extend(_numbers_after_marker(text, "--") or _number_list(text))
    return freq, (ir if ir else None)


def _parse_anharmonic_x_matrix(lines: list[str]) -> list[list[float | None]] | None:
    header_idx = None
    for idx, line in enumerate(lines):
        if line.strip() == "Total Anharmonic X Matrix (in cm^-1)":
            header_idx = idx
    if header_idx is None:
        return None
    columns: list[int] = []
    entries: dict[tuple[int, int], float] = {}
    max_index = 0
    parsed_any = False
    for raw in lines[header_idx + 1 :]:
        text = raw.strip()
        if not text:
            continue
        if text.startswith("="):
            if parsed_any:
                break
            continue
        if text.startswith("-"):
            continue
        parts = text.split()
        if all(_is_int_token(part) for part in parts):
            columns = [int(part) for part in parts]
            if columns:
                max_index = max(max_index, max(columns))
            continue
        if not columns or not parts or not _is_int_token(parts[0]):
            if parsed_any:
                break
            continue
        row = int(parts[0])
        max_index = max(max_index, row)
        row_parsed = False
        for col, token in zip(columns, parts[1:]):
            value = _gaussian_float(token)
            if value is None:
                continue
            entries[(row, col)] = value
            entries[(col, row)] = value
            max_index = max(max_index, row, col)
            row_parsed = True
        parsed_any = parsed_any or row_parsed
    if not entries or max_index <= 0:
        return None
    chi: list[list[float | None]] = [[None for _ in range(max_index)] for _ in range(max_index)]
    for (row, col), value in entries.items():
        chi[row - 1][col - 1] = value
    for idx in range(max_index):
        if chi[idx][idx] is None:
            chi[idx][idx] = 0.0
    return chi


def _parse_vibro_rot_alpha_matrix(lines: list[str]) -> list[DeltaBVibAlphaRow]:
    header_idx = None
    for idx, line in enumerate(lines):
        if line.strip() == "Vibro-Rot alpha Matrix (in MHz)":
            header_idx = idx
    if header_idx is None:
        return []
    rows: list[DeltaBVibAlphaRow] = []
    mode_pattern = re.compile(r"Q\(\s*(\d+)\)")
    for raw in lines[header_idx + 1 :]:
        text = raw.strip()
        if not text:
            if rows:
                break
            continue
        if text.startswith("-") or text.startswith("A(") or text.startswith("B("):
            continue
        match = mode_pattern.search(text)
        if match is None:
            if rows:
                break
            continue
        nums = _number_list(text)
        if len(nums) < 4:
            continue
        rows.append(
            DeltaBVibAlphaRow(
                mode=int(match.group(1)),
                a_MHz=float(nums[1]),
                b_MHz=float(nums[2]),
                c_MHz=float(nums[3]),
            )
        )
    return rows


def _parse_deltabvib_from_rot_constants(lines: list[str]) -> tuple[float, float, float] | None:
    rot = _parse_rot_constants_mhz(lines)
    try:
        return rot["A00"] - rot["Ae"], rot["B00"] - rot["Be"], rot["C00"] - rot["Ce"]
    except KeyError:
        return None


def _parse_rot_constants_mhz(lines: list[str]) -> dict[str, float]:
    idx = None
    for line_idx, line in enumerate(lines):
        if line.strip() == "Rotational Constants (in MHz)":
            idx = line_idx
            break
    if idx is None:
        return {}
    out: dict[str, float] = {}
    pattern = re.compile(r"([A-Z][A-Za-z0-9]*)=\s*([+-]?\d*\.?\d+(?:[DEde][+-]?\d+)?)")
    for raw in lines[idx : min(idx + 30, len(lines))]:
        for match in pattern.finditer(raw):
            value = _gaussian_float(match.group(2))
            if value is not None:
                out[match.group(1)] = value
    return out


def _parse_temperature_from_log(lines: list[str]) -> float | None:
    values: list[float] = []
    pattern = re.compile(r"^\s*T\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*K;", re.I)
    for line in lines:
        match = pattern.match(line)
        if match is not None:
            values.append(float(match.group(1)))
    if not values:
        return None
    for value in values:
        if abs(value - 298.15) < 1.0e-2:
            return value
    return values[0]


def _parse_point_group_from_log(lines: list[str]) -> str | None:
    point_group = None
    for line in lines:
        if "Full point group" not in line:
            continue
        parts = line.split()
        if "group" in parts:
            try:
                point_group = parts[parts.index("group") + 1]
            except IndexError:
                pass
    return point_group


def _parse_dipole_debye_from_log(
    lines: list[str],
) -> tuple[float | None, float | None, float | None] | None:
    dipole = None
    for idx, line in enumerate(lines[:-1]):
        text = line.strip()
        if text.startswith("Dipole moment (field-independent basis, Debye):"):
            match = re.search(
                r"X=\s*([-+]?\d*\.?\d+(?:[DEde][-+]?\d+)?)\s+"
                r"Y=\s*([-+]?\d*\.?\d+(?:[DEde][-+]?\d+)?)\s+"
                r"Z=\s*([-+]?\d*\.?\d+(?:[DEde][-+]?\d+)?)",
                lines[idx + 1],
            )
            if match:
                dipole = tuple(_gaussian_float(token) for token in match.groups())
        elif text.startswith("Dipole moment (Debye):"):
            values = _number_list(lines[idx + 1])
            if len(values) >= 3:
                dipole = (values[0], values[1], values[2])
    return dipole


def _merge_rotational_sections(
    existing: RotationalSection, incoming: RotationalSection
) -> RotationalSection:
    return RotationalSection(
        rotor_type=incoming.rotor_type or existing.rotor_type,
        representation=incoming.representation or existing.representation,
        point_group=incoming.point_group or existing.point_group,
        watson_reduction=incoming.watson_reduction or existing.watson_reduction,
        symmetry_number=incoming.symmetry_number
        if incoming.symmetry_number is not None
        else existing.symmetry_number,
        temperature_K=incoming.temperature_K
        if incoming.temperature_K is not None
        else existing.temperature_K,
        pressure_atm=incoming.pressure_atm
        if incoming.pressure_atm is not None
        else existing.pressure_atm,
        A_MHz=incoming.A_MHz if incoming.A_MHz is not None else existing.A_MHz,
        B_MHz=incoming.B_MHz if incoming.B_MHz is not None else existing.B_MHz,
        C_MHz=incoming.C_MHz if incoming.C_MHz is not None else existing.C_MHz,
        equilibrium_MHz=(
            incoming.equilibrium_MHz
            if incoming.equilibrium_MHz is not None
            else existing.equilibrium_MHz
        ),
        ground_state_MHz=(
            incoming.ground_state_MHz
            if incoming.ground_state_MHz is not None
            else existing.ground_state_MHz
        ),
        constant_kind=incoming.constant_kind or existing.constant_kind,
        constant_source=incoming.constant_source or existing.constant_source,
        dipole_debye=(
            incoming.dipole_debye if incoming.dipole_debye is not None else existing.dipole_debye
        ),
        q_rot=incoming.q_rot if incoming.q_rot is not None else existing.q_rot,
        delta_vib_MHz=(
            incoming.delta_vib_MHz if incoming.delta_vib_MHz is not None else existing.delta_vib_MHz
        ),
    )


def _numbers_after_marker(text: str, marker: str) -> list[float]:
    if marker not in text:
        return []
    return _number_list(text.split(marker, 1)[1])


def _number_list(text: str) -> list[float]:
    values: list[float] = []
    for token in re.findall(r"[-+]?\d*\.?\d+(?:[DEde][+-]?\d+)?", text):
        value = _gaussian_float(token)
        if value is not None:
            values.append(value)
    return values


def _gaussian_float(token: str) -> float | None:
    try:
        return float(token.replace("D", "E").replace("d", "e"))
    except Exception:
        return None


def _is_int_token(token: str) -> bool:
    return bool(re.fullmatch(r"\d+", token))
