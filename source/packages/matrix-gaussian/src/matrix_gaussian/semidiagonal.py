from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt
from pathlib import Path
import re

import numpy as np

from matrix_chem.physical_constants import Phy, get_physical_constants

_CONSTANTS = get_physical_constants()
AMU_KG = _CONSTANTS[Phy.TO_KG]
ANG_PER_BOHR = _CONSTANTS[Phy.TO_ANG]
M_PER_BOHR = _CONSTANTS[Phy.M_PER_B]
CLIGHT_CM_S = _CONSTANTS[Phy.C_LIGHT]
PLANCK_J_S = _CONSTANTS[Phy.PLANCK]
PLANCK_AU = PLANCK_J_S / (AMU_KG * M_PER_BOHR**2)
J_PER_HARTREE = _CONSTANTS[Phy.HARTREE]


@dataclass(frozen=True)
class GaussianSemiDiagonalCubicRovibData:
    """Gaussian vibrot data needed for semidiagonal cubic DeltaBVib corrections.

    The `cubic_f3ijj_mw` matrix stores Gaussian's raw semidiagonal cubic constants
    in mass-weighted normal coordinates before conversion to MATRIX `cubic_cm`
    terms.  Indices are zero-based in the Python object.
    """

    natoms: int
    nvib: int
    linear: bool
    frequencies_cm1: tuple[float, ...]
    frequency_order: tuple[int, ...]
    inertia_tensor_au: tuple[tuple[float, float, float], ...]
    beq_MHz: tuple[float, float, float]
    inertia_derivatives_amu_sqrt_ang: tuple[tuple[tuple[float, ...], ...], ...]
    coriolis: dict[str, tuple[tuple[float, ...], ...]]
    cubic_f3ijj_mw: tuple[tuple[float, ...], ...]

    def to_anharmonic_input(self):
        from matrix_vpt2_vci import AnharmonicInput

        frequencies = np.asarray(self.frequencies_cm1, dtype=float)
        cubic = semidiagonal_cubic_cm(self)
        data = AnharmonicInput(
            harmonic_frequencies_cm=frequencies,
            anharmonic_frequencies_cm=np.array((), dtype=float),
            cubic_cm=cubic,
            quartic_cm={},
            source="gaussian-semidiagonal-cubic-vibrot",
        )
        data.validate()
        return data


@dataclass(frozen=True)
class SemiDiagonalDeltaBVibResult:
    harmonic_MHz: tuple[float, float, float]
    coriolis_MHz: tuple[float, float, float]
    anharmonic_MHz: tuple[float, float, float]
    total_MHz: tuple[float, float, float]
    beq_MHz: tuple[float, float, float]


def read_gaussian_semidiagonal_cubic_rovib(
    path: Path | str,
) -> GaussianSemiDiagonalCubicRovibData:
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    return _parse_semidiagonal_cubic_rovib(lines, strict=True)


def read_gaussian_vibrot_diagnostics(
    path: Path | str,
) -> GaussianSemiDiagonalCubicRovibData:
    """Read Gaussian vibrot diagnostics without requiring numerical ``F3ijj``.

    A zero semidiagonal cubic matrix is used when the log is a harmonic
    ``freq=vibrot`` calculation.  This keeps the one-mode inertia diagnostic
    usable independently of the optional anharmonic potential terms.
    """
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    return _parse_semidiagonal_cubic_rovib(lines, strict=True, require_cubic=False)


def compute_deltabvib_from_semidiagonal_cubic_data(
    data: GaussianSemiDiagonalCubicRovibData,
) -> SemiDiagonalDeltaBVibResult:
    freq = np.asarray(data.frequencies_cm1, dtype=float)
    if np.any(freq <= 0.0):
        raise ValueError("semidiagonal DeltaBVib requires positive harmonic frequencies")
    atens = np.asarray(data.inertia_derivatives_amu_sqrt_ang, dtype=float) / ANG_PER_BOHR
    imat = np.asarray(data.inertia_tensor_au, dtype=float)
    beq_MHz = np.asarray(data.beq_MHz, dtype=float)
    beq_cm = beq_MHz / (CLIGHT_CM_S * 1.0e-6)
    fijj = np.asarray(data.cubic_f3ijj_mw, dtype=float)
    zeta = {
        axis: np.asarray(data.coriolis[axis], dtype=float)
        for axis in ("x", "y", "z")
    }

    axes = ("x", "y", "z")
    dbvib = np.zeros((3, 4), dtype=float)
    for tau in range(3):
        if not np.isfinite(beq_cm[tau]) or abs(beq_cm[tau]) < 1.0e-14:
            continue
        for i, wi in enumerate(freq):
            for eta in range(3):
                inertia = imat[eta, eta]
                if abs(inertia) < 1.0e-14:
                    continue
                dbvib[tau, 0] -= 3.0 * atens[tau, eta, i] ** 2 / (4.0 * wi * inertia)
            for j, wj in enumerate(freq):
                z2ij = zeta[axes[tau]][i, j] ** 2
                dbvib[tau, 1] += z2ij * (wi - wj) ** 2 / (2.0 * wi * wj * (wi + wj))
                f_red = _fijkl_from_mass_weighted_Q_to_dimensionless_q(fijj[j, i], freq, (i, i, j))
                dbvib[tau, 2] -= (
                    pi
                    * sqrt(CLIGHT_CM_S / PLANCK_AU)
                    * atens[tau, tau, j]
                    * f_red
                    / (wj * sqrt(wj))
                )
        dbvib[tau, 3] = dbvib[tau, 0] + dbvib[tau, 1] + dbvib[tau, 2]
        dbvib[tau, :] *= -(beq_cm[tau] ** 2) * CLIGHT_CM_S * 1.0e-6

    return SemiDiagonalDeltaBVibResult(
        harmonic_MHz=tuple(float(value) for value in dbvib[:, 0]),
        coriolis_MHz=tuple(float(value) for value in dbvib[:, 1]),
        anharmonic_MHz=tuple(float(value) for value in dbvib[:, 2]),
        total_MHz=tuple(float(value) for value in dbvib[:, 3]),
        beq_MHz=tuple(float(value) for value in beq_MHz),
    )


def semidiagonal_data_from_cartesian_cubic(
    symbols,
    coordinates_angstrom,
    hessian_hartree_per_bohr2,
    cubic_hartree_per_bohr3=None,
    *,
    substitutions: dict[int, int] | None = None,
    cubic_unique_indices=None,
    cubic_unique_values_hartree_per_bohr3=None,
) -> GaussianSemiDiagonalCubicRovibData:
    """Re-express one Cartesian F2/F3 for a requested isotopologue."""

    from matrix_rovib import transform_cartesian_cubic_for_isotopologue

    transformed = transform_cartesian_cubic_for_isotopologue(
        symbols,
        coordinates_angstrom,
        hessian_hartree_per_bohr2,
        cubic_hartree_per_bohr3,
        substitutions=substitutions,
        cubic_unique_indices=cubic_unique_indices,
        cubic_unique_values_hartree_per_bohr3=cubic_unique_values_hartree_per_bohr3,
    )
    nvib = transformed.frequencies_cm1.size
    inertia = np.asarray(transformed.inertia_principal_amu_bohr2, dtype=float)
    didq = np.asarray(transformed.inertia_derivatives_amu_sqrt_ang, dtype=float)
    derivatives = np.zeros((3, 3, nvib), dtype=float)
    derivatives[0, 0] = didq[0]
    derivatives[1, 1] = didq[1]
    derivatives[2, 2] = didq[2]
    derivatives[0, 1] = derivatives[1, 0] = didq[3]
    derivatives[0, 2] = derivatives[2, 0] = didq[4]
    derivatives[1, 2] = derivatives[2, 1] = didq[5]
    return GaussianSemiDiagonalCubicRovibData(
        natoms=len(symbols),
        nvib=int(nvib),
        linear=bool(nvib == 3 * len(symbols) - 5),
        frequencies_cm1=tuple(float(value) for value in transformed.frequencies_cm1),
        frequency_order=tuple(range(1, nvib + 1)),
        inertia_tensor_au=_tensor2_to_tuple(inertia),
        beq_MHz=tuple(_rotational_constant_mhz(float(inertia[i, i])) for i in range(3)),
        inertia_derivatives_amu_sqrt_ang=_tensor3_to_tuple(derivatives),
        coriolis={axis: _tensor2_to_tuple(values) for axis, values in transformed.coriolis.items()},
        cubic_f3ijj_mw=_tensor2_to_tuple(transformed.semidiagonal_f3ijj_mw),
    )


def semidiagonal_data_from_parent_normal_cubic(
    symbols,
    coordinates_angstrom,
    hessian_hartree_per_bohr2,
    cubic_parent_qmw,
    parent_modes_cartesian,
    parent_reduced_masses_amu,
    parent_masses_amu,
    *,
    substitutions: dict[int, int] | None = None,
) -> GaussianSemiDiagonalCubicRovibData:
    """Transform a parent Freq=Anharm F3 directly into one isotope basis."""

    from matrix_rovib import transform_normal_cubic_for_isotopologue

    transformed = transform_normal_cubic_for_isotopologue(
        symbols,
        coordinates_angstrom,
        hessian_hartree_per_bohr2,
        cubic_parent_qmw,
        parent_modes_cartesian,
        parent_reduced_masses_amu,
        parent_masses_amu,
        substitutions=substitutions,
    )
    nvib = transformed.frequencies_cm1.size
    inertia = np.asarray(transformed.inertia_principal_amu_bohr2, dtype=float)
    didq = np.asarray(transformed.inertia_derivatives_amu_sqrt_ang, dtype=float)
    derivatives = np.zeros((3, 3, nvib), dtype=float)
    derivatives[0, 0], derivatives[1, 1], derivatives[2, 2] = didq[:3]
    derivatives[0, 1] = derivatives[1, 0] = didq[3]
    derivatives[0, 2] = derivatives[2, 0] = didq[4]
    derivatives[1, 2] = derivatives[2, 1] = didq[5]
    return GaussianSemiDiagonalCubicRovibData(
        natoms=len(symbols),
        nvib=int(nvib),
        linear=bool(nvib == 3 * len(symbols) - 5),
        frequencies_cm1=tuple(float(value) for value in transformed.frequencies_cm1),
        frequency_order=tuple(range(1, nvib + 1)),
        inertia_tensor_au=_tensor2_to_tuple(inertia),
        beq_MHz=tuple(_rotational_constant_mhz(float(inertia[i, i])) for i in range(3)),
        inertia_derivatives_amu_sqrt_ang=_tensor3_to_tuple(derivatives),
        coriolis={axis: _tensor2_to_tuple(values) for axis, values in transformed.coriolis.items()},
        cubic_f3ijj_mw=_tensor2_to_tuple(transformed.semidiagonal_f3ijj_mw),
    )


def semidiagonal_cubic_cm(
    data: GaussianSemiDiagonalCubicRovibData,
) -> dict[tuple[int, int, int], float]:
    frequencies = np.asarray(data.frequencies_cm1, dtype=float)
    if np.any(frequencies <= 0.0):
        raise ValueError("semidiagonal cubic conversion requires positive harmonic frequencies")
    fijj = np.asarray(data.cubic_f3ijj_mw, dtype=float)
    cubic: dict[tuple[int, int, int], float] = {}
    for i in range(data.nvib):
        for j in range(data.nvib):
            value = _fijkl_from_mass_weighted_Q_to_dimensionless_q(fijj[i, j], frequencies, (i, j, j))
            if value != 0.0:
                cubic[tuple(sorted((i, j, j)))] = float(value)
    return cubic


def _parse_semidiagonal_cubic_rovib(
    lines: list[str],
    *,
    strict: bool,
    require_cubic: bool = True,
) -> GaussianSemiDiagonalCubicRovibData | None:
    natoms = _parse_natoms(lines)
    if natoms is None:
        if strict:
            raise ValueError("NAtoms marker not found in Gaussian log")
        return None
    linear = _is_linear_molecule(lines)
    nvib = 3 * natoms - (5 if linear else 6)
    if nvib <= 0:
        if strict:
            raise ValueError("semidiagonal DeltaBVib currently requires a nonlinear molecule")
        return None

    frequencies = _parse_first_n_frequencies(lines, nvib)
    inertia = _parse_principal_axis_inertia_tensor(lines)
    didq = _parse_inertia_derivatives(lines, nvib)
    coriolis = _parse_coriolis_couplings(lines, nvib)
    cubic = _parse_diagonal_normal_mode_f3ijj(lines, nvib)
    frequency_order = _parse_frequency_order(lines, nvib)
    if not frequency_order:
        frequency_order = tuple(range(1, nvib + 1))

    missing: list[str] = []
    if len(frequencies) != nvib:
        missing.append("harmonic frequencies")
    if inertia is None:
        missing.append("principal-axis inertia tensor")
    if didq is None:
        missing.append("inertia derivatives")
    if any(axis not in coriolis for axis in ("x", "y", "z")):
        missing.append("Coriolis couplings")
    if cubic is None and require_cubic:
        missing.append("semidiagonal cubic F3ijj")
    if missing:
        if strict:
            raise ValueError(
                "Gaussian semidiagonal rovib data incomplete: "
                + ", ".join(missing)
                + ". Expected a harmonic step with freq=vibrot and a second step with "
                + "freq=(numer,readharm,vibrot); F3ijj is printed by the numerical-gradient "
                + "step, while Coriolis and inertia derivatives come from vibrot output."
            )
        return None

    if cubic is None:
        cubic = np.zeros((nvib, nvib), dtype=float)
    if frequency_order != tuple(range(1, nvib + 1)):
        frequencies, cubic = _reorder_semidiagonal_cubic(frequencies, cubic, frequency_order)

    beq_MHz = tuple(
        _rotational_constant_mhz(float(inertia[idx][idx]))
        for idx in range(3)
    )
    return GaussianSemiDiagonalCubicRovibData(
        natoms=natoms,
        nvib=nvib,
        linear=linear,
        frequencies_cm1=tuple(float(value) for value in frequencies),
        frequency_order=frequency_order,
        inertia_tensor_au=tuple(tuple(float(value) for value in row) for row in inertia),
        beq_MHz=beq_MHz,
        inertia_derivatives_amu_sqrt_ang=_tensor3_to_tuple(didq),
        coriolis={
            axis: tuple(tuple(float(value) for value in row) for row in coriolis[axis])
            for axis in ("x", "y", "z")
        },
        cubic_f3ijj_mw=tuple(tuple(float(value) for value in row) for row in cubic),
    )


def _is_linear_molecule(lines: list[str]) -> bool:
    for line in lines:
        text = line.lower()
        if "linear molecule" in text or "linear rotor" in text:
            return True
        if "full point group" in text and ("d*h" in text or "c*v" in text):
            return True
    return False


def _rotational_constant_mhz(moment_au: float) -> float:
    if abs(moment_au) < 1.0e-14:
        return 0.0
    return float(PLANCK_AU / (8.0 * pi**2 * CLIGHT_CM_S * moment_au) * CLIGHT_CM_S * 1e-6)


def _parse_natoms(lines: list[str]) -> int | None:
    pattern = re.compile(r"\bNAtoms=\s*(\d+)")
    for line in lines:
        match = pattern.search(line)
        if match is not None:
            return int(match.group(1))
    compact = re.sub(r"\s+", "", "".join(lines))
    starts = [match.start() for match in re.finditer(r"1\\1\\GINC", compact)]
    for start in reversed(starts):
        end = compact.find(r"\@", start)
        if end < 0:
            continue
        entry = compact[start : end + 2]
        geometry = re.search(
            r"\\\\-?\d+,\d+\\(?P<body>.*?)(?=\\\\Version=)", entry
        )
        if geometry is None:
            continue
        count = 0
        for token in geometry.group("body").split("\\"):
            parts = token.split(",")
            if len(parts) >= 4 and re.fullmatch(r"[A-Za-z]{1,3}", parts[0]):
                count += 1
        if count:
            return count
    return None


def _parse_first_n_frequencies(lines: list[str], count: int) -> tuple[float, ...]:
    values: list[float] = []
    for line in lines:
        if "Frequencies --" not in line:
            continue
        values.extend(_numbers_after_marker(line, "--"))
        if len(values) >= count:
            break
    return tuple(values[:count])


def _parse_principal_axis_inertia_tensor(lines: list[str]) -> np.ndarray | None:
    idx = _last_index_containing(lines, "Principal axes and moments of inertia in atomic units:")
    if idx is None:
        return None
    eigenvalues: list[float] | None = None
    axes: list[list[float]] = []
    for raw in lines[idx + 1 : min(idx + 12, len(lines))]:
        text = raw.strip()
        if text.startswith("Eigenvalues --"):
            values = _number_list(text)
            if len(values) >= 3:
                eigenvalues = values[:3]
            continue
        parts = text.split()
        if parts and parts[0] in {"X", "Y", "Z"}:
            values = [_gaussian_float(token) for token in parts[1:4]]
            if len(values) == 3 and all(value is not None for value in values):
                axes.append([float(value) for value in values if value is not None])
    if eigenvalues is None or len(axes) != 3:
        return None
    imat = np.diag(np.asarray(eigenvalues, dtype=float))
    axis_matrix = np.asarray(axes, dtype=float)
    return axis_matrix @ imat @ axis_matrix.T


def _parse_inertia_derivatives(lines: list[str], nvib: int) -> np.ndarray | None:
    idx = _last_index_containing(lines, "Inertia Moments Derivatives w.r.t. Normal Modes")
    if idx is None:
        return None
    tensor = np.zeros((3, 3, nvib), dtype=float)
    rows = 0
    for raw in lines[idx + 1 :]:
        text = raw.strip()
        if not text:
            continue
        if rows and (text.startswith("=") or "Vibro-rotational" in text):
            break
        values = _number_list(text)
        if len(values) < 7:
            continue
        data = values[-6:]
        tensor[0, 0, rows] = data[0]
        tensor[0, 1, rows] = tensor[1, 0, rows] = data[1]
        tensor[1, 1, rows] = data[2]
        tensor[0, 2, rows] = tensor[2, 0, rows] = data[3]
        tensor[1, 2, rows] = tensor[2, 1, rows] = data[4]
        tensor[2, 2, rows] = data[5]
        rows += 1
        if rows == nvib:
            return tensor
    return None


def _parse_coriolis_couplings(lines: list[str], nvib: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for idx, raw in enumerate(lines):
        text = raw.strip()
        if not text.startswith("Coriolis Couplings along the"):
            continue
        axis_match = re.search(r"along the\s+([XYZ])\s+axis", text, re.I)
        if axis_match is None:
            continue
        axis = axis_match.group(1).lower()
        matrix, _ = _parse_gaussian_block_matrix(lines, idx + 1, nvib, antisymmetric=True)
        if matrix is not None:
            out[axis] = matrix
    return out


def _parse_diagonal_normal_mode_f3ijj(lines: list[str], nvib: int) -> np.ndarray | None:
    idx = _last_index_containing(lines, "Diagonal normal mode F3ijj")
    if idx is None:
        return None
    matrix, _ = _parse_gaussian_block_matrix(lines, idx + 1, nvib, antisymmetric=False)
    return matrix


def _parse_gaussian_block_matrix(
    lines: list[str],
    start: int,
    size: int,
    *,
    antisymmetric: bool,
) -> tuple[np.ndarray | None, int]:
    matrix = np.zeros((size, size), dtype=float)
    columns: list[int] = []
    parsed_any = False
    idx = start
    while idx < len(lines):
        text = lines[idx].strip()
        if not text or text.startswith("-"):
            idx += 1
            continue
        parts = text.split()
        if parts and all(_is_int_token(part) for part in parts):
            columns = [int(part) - 1 for part in parts if 1 <= int(part) <= size]
            idx += 1
            continue
        if not columns or not parts or not _is_int_token(parts[0]):
            if parsed_any:
                break
            idx += 1
            continue
        row = int(parts[0]) - 1
        if row < 0 or row >= size:
            if parsed_any:
                break
            idx += 1
            continue
        row_parsed = False
        for col, token in zip(columns, parts[1:]):
            value = _gaussian_float(token)
            if value is None:
                continue
            matrix[row, col] = value
            if antisymmetric:
                matrix[col, row] = -value
            row_parsed = True
        parsed_any = parsed_any or row_parsed
        idx += 1
    if not parsed_any:
        return None, idx
    return matrix, idx


def _parse_frequency_order(lines: list[str], nvib: int) -> tuple[int, ...]:
    idx = _last_index_containing(lines, "Input/Output information")
    if idx is None:
        return ()
    order: list[int] = []
    for raw in lines[idx + 1 : min(idx + 80, len(lines))]:
        text = raw.replace("|", " ").strip()
        if text.startswith("(A)"):
            values = _unique_mode_labels(text, nvib)
            order.extend(values)
            if len(order) >= nvib:
                candidate = tuple(order[:nvib])
                return candidate if _is_complete_mode_permutation(candidate, nvib) else ()
        elif order and text.startswith("Normal modes will be READ"):
            break
    candidate = tuple(order[:nvib]) if len(order) == nvib else ()
    return candidate if _is_complete_mode_permutation(candidate, nvib) else ()


def _unique_mode_labels(text: str, nvib: int) -> list[int]:
    raw_labels = re.findall(r"\d+[A-Za-z]?", text)
    parsed: list[tuple[int, str]] = []
    for label in raw_labels:
        match = re.fullmatch(r"(\d+)([A-Za-z]?)", label)
        if match is not None:
            parsed.append((int(match.group(1)), match.group(2).lower()))
    duplicate_bases = {
        base
        for base in {item[0] for item in parsed}
        if sum(1 for item in parsed if item[0] == base) > 1
    }
    explicit_singletons = {
        base
        for base, suffix in parsed
        if not suffix and base not in duplicate_bases and 1 <= base <= nvib
    }
    used: set[int] = set()
    out: list[int] = []
    for index, (base, suffix) in enumerate(parsed):
        if suffix or base in duplicate_bases:
            occurrence = sum(1 for prev_base, _ in parsed[: index + 1] if prev_base == base) - 1
            value = _degenerate_partner_index(base, occurrence, nvib, used | explicit_singletons)
        else:
            value = base
        used.add(value)
        out.append(value)
    return out


def _degenerate_partner_index(
    base: int,
    occurrence: int,
    nvib: int,
    reserved: set[int],
) -> int:
    preferred = base + occurrence
    if 1 <= preferred <= nvib and (preferred == base or preferred not in reserved):
        return preferred
    for candidate in range(base, nvib + 1):
        if candidate not in reserved:
            return candidate
    for candidate in range(1, base):
        if candidate not in reserved:
            return candidate
    return preferred


def _is_complete_mode_permutation(order: tuple[int, ...], nvib: int) -> bool:
    return len(order) == nvib and set(order) == set(range(1, nvib + 1))


def _reorder_semidiagonal_cubic(
    frequencies: tuple[float, ...],
    cubic: np.ndarray,
    order: tuple[int, ...],
) -> tuple[tuple[float, ...], np.ndarray]:
    freq_sort = np.zeros(len(frequencies), dtype=float)
    cubic_sort = np.zeros_like(cubic)
    for harmonic_idx, sorted_mode in enumerate(order):
        freq_sort[sorted_mode - 1] = frequencies[harmonic_idx]
    for i, sorted_i in enumerate(order):
        for j, sorted_j in enumerate(order):
            cubic_sort[sorted_i - 1, sorted_j - 1] = cubic[i, j]
    return tuple(float(value) for value in freq_sort), cubic_sort


def _tensor3_to_tuple(tensor: np.ndarray) -> tuple[tuple[tuple[float, ...], ...], ...]:
    return tuple(
        tuple(tuple(float(value) for value in tensor[i, j, :]) for j in range(tensor.shape[1]))
        for i in range(tensor.shape[0])
    )


def _tensor2_to_tuple(tensor: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in np.asarray(tensor, dtype=float))


def _fijkl_from_mass_weighted_Q_to_dimensionless_q(
    f_mw: float,
    frequencies_cm1: np.ndarray,
    modes: tuple[int, int, int],
) -> float:
    hbar_au = PLANCK_AU / (2.0 * pi)
    f_red = float(f_mw)
    for mode in modes:
        f_red *= sqrt(hbar_au / (2.0 * pi * CLIGHT_CM_S * float(frequencies_cm1[mode])))
    f_red *= J_PER_HARTREE
    f_red /= PLANCK_J_S * CLIGHT_CM_S
    return f_red


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


def _last_index_containing(lines: list[str], marker: str) -> int | None:
    found = None
    for idx, line in enumerate(lines):
        if marker in line:
            found = idx
    return found
