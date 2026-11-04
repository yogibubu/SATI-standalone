"""Gaussian reduced normal-mode cubic and quartic force constants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Mapping


_ROW3 = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+([-+]?\d+(?:\.\d*)?(?:[DEde][-+]?\d+)?)"
)
_ROW4 = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([-+]?\d+(?:\.\d*)?(?:[DEde][-+]?\d+)?)"
)
_ROW2 = re.compile(
    r"^\s*(\d+)\s+\1\s+([-+]?\d+(?:\.\d*)?(?:[DEde][-+]?\d+)?)"
)


@dataclass(frozen=True)
class GaussianReducedNormalQFF:
    """Reduced constants in cm-1, indexed in ascending-frequency order."""

    harmonic_frequencies_cm: tuple[float, ...]
    cubic_cm: dict[tuple[int, int, int], float]
    quartic_cm: dict[tuple[int, int, int, int], float]
    source: str


CM_PER_HARTREE = 219474.6313705


def write_gaussian_qred_anharmonic_data(
    path: Path | str,
    cubic_cm: Mapping[tuple[int, int, int], float],
    quartic_cm: Mapping[tuple[int, int, int, int], float],
    *,
    mode_signs: tuple[float, ...] | list[float] | None = None,
    threshold_cm: float = 0.0,
) -> Path:
    """Write the PESAnh part accepted as Gaussian ``InQRedX`` data.

    Values are reduced derivatives with respect to dimensionless normal
    coordinates.  MATRIX stores ascending tuple indices; Gaussian's external
    block prints the same mode order but writes every symmetric tuple from the
    largest index to the smallest.  ``mode_signs`` maps the MATRIX eigenvector
    gauge onto the Gaussian reference gauge.
    """

    target = Path(path)
    all_indices = (*cubic_cm.keys(), *quartic_cm.keys())
    mode_count = 1 + max((max(key) for key in all_indices), default=-1)
    signs = [1.0] * mode_count if mode_signs is None else [float(x) for x in mode_signs]
    if len(signs) < mode_count or any(abs(value) != 1.0 for value in signs):
        raise ValueError("Gaussian QRed mode signs must contain one +/-1 per mode")

    rows = [" ENERGY_D3N"]
    for indices, value in sorted(cubic_cm.items()):
        gauged = float(value)
        for index in indices:
            gauged *= signs[index]
        if abs(gauged) >= threshold_cm:
            labels = "".join(f"{index + 1:8d}" for index in reversed(indices))
            rows.append(f"{labels}  {gauged / CM_PER_HARTREE: .10E}".replace("E", "D"))
    rows.append(" ENERGY_D4N")
    for indices, value in sorted(quartic_cm.items()):
        gauged = float(value)
        for index in indices:
            gauged *= signs[index]
        if abs(gauged) >= threshold_cm:
            labels = "".join(f"{index + 1:8d}" for index in reversed(indices))
            rows.append(f"{labels}  {gauged / CM_PER_HARTREE: .10E}".replace("E", "D"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return target


def write_gaussian_qmw_anharmonic_constants(
    path: Path | str,
    cubic_cm: Mapping[tuple[int, int, int], float],
    quartic_cm: Mapping[tuple[int, int, int, int], float],
    *,
    harmonic_frequencies_cm: tuple[float, ...] | list[float],
    harmonic_qmw_force_constants: tuple[float, ...] | list[float],
    mode_signs: tuple[float, ...] | list[float] | None = None,
    threshold_cm: float = 0.0,
) -> Path:
    """Write sparse Gaussian ``InDerAU`` D3/D4 constants.

    ZAFF/TRINITY reduced constants are derivatives in dimensionless normal
    coordinates.  Gaussian's compact atomic-unit reader expects derivatives
    in mass-weighted normal coordinates.  For mode i the conversion scale is
    ``sqrt((omega_i / Eh_to_cm) / K_ii)``.
    """

    import numpy as np

    frequencies = np.asarray(harmonic_frequencies_cm, dtype=float)
    quadratic = np.asarray(harmonic_qmw_force_constants, dtype=float)
    if frequencies.shape != quadratic.shape or np.any(frequencies <= 0.0) or np.any(quadratic <= 0.0):
        raise ValueError("Gaussian QMW conversion needs matching positive frequencies and Kii")
    scale = np.sqrt((frequencies / CM_PER_HARTREE) / quadratic)
    signs = (
        np.ones(frequencies.size, dtype=float)
        if mode_signs is None
        else np.asarray(mode_signs, dtype=float)
    )
    if signs.shape != frequencies.shape or np.any(np.abs(signs) != 1.0):
        raise ValueError("Gaussian QMW mode signs must contain one +/-1 per mode")
    rows: list[str] = []
    for tensor in (cubic_cm, quartic_cm):
        for indices, value in sorted(tensor.items()):
            gauged = float(value) * float(np.prod(signs[list(indices)]))
            if abs(gauged) < threshold_cm:
                continue
            qmw = (gauged / CM_PER_HARTREE) / float(np.prod(scale[list(indices)]))
            labels = "".join(f"{index + 1:8d}" for index in reversed(indices))
            rows.append(f"{labels}  {qmw: .10E}".replace("E", "D"))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return target


def merge_gaussian_qred_anharmonic_data(
    template: Path | str,
    replacement: Path | str,
    output: Path | str,
) -> Path:
    """Insert QRed D3/D4 blocks into a Gaussian InDataNM metadata envelope."""

    template_lines = Path(template).read_text(encoding="utf-8").splitlines()
    replacement_lines = Path(replacement).read_text(encoding="utf-8").splitlines()
    start = _section_index(template_lines, "ENERGY_D3N")
    end = _section_index(template_lines, "CORIOLIS")
    if not any(line.strip() == "ENERGY_D4N" for line in replacement_lines):
        raise ValueError("replacement data needs ENERGY_D3N and ENERGY_D4N blocks")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "\n".join((*template_lines[:start], *replacement_lines, *template_lines[end:])) + "\n",
        encoding="utf-8",
    )
    return target


def read_gaussian_reduced_normal_qff(path: Path | str) -> GaussianReducedNormalQFF:
    """Read Gaussian's printed ``Freq=Anharm`` normal-mode force constants.

    The printed H/A correspondence maps symmetry-grouped anharmonic indices
    to harmonic order. Legacy files without that table retain their descending
    frequency convention.  Terms omitted by Gaussian's print threshold are
    intentionally absent from the returned sparse dictionaries.
    """

    source = Path(path)
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    quadratic = _heading(lines, "QUADRATIC FORCE CONSTANTS IN NORMAL MODES")
    cubic_start = _heading(lines, "CUBIC FORCE CONSTANTS IN NORMAL MODES")
    quartic_start = _heading(lines, "QUARTIC FORCE CONSTANTS IN NORMAL MODES")

    descending: dict[int, float] = {}
    for line in lines[quadratic:cubic_start]:
        match = _ROW2.match(line)
        if match:
            descending[int(match.group(1))] = _float(match.group(2))
    if not descending:
        raise ValueError(f"Gaussian quadratic normal-mode table not found in {source}")
    mode_count = max(descending)
    if set(descending) != set(range(1, mode_count + 1)):
        raise ValueError("Gaussian quadratic normal-mode table is incomplete")
    if any("Input/Output information" in line for line in lines):
        from .anharmonic_cubic import _harmonic_to_anharmonic_order

        h_to_a = _harmonic_to_anharmonic_order(lines, mode_count)
    else:
        h_to_a = tuple(range(mode_count, 0, -1))
    a_to_h = {a: h for h, a in enumerate(h_to_a)}
    frequencies = tuple(descending[a] for a in h_to_a)

    cubic_end = _find_after(lines, cubic_start, "Num. of 3rd derivatives")
    quartic_end = _find_after(lines, quartic_start, "Num. of 4th derivatives")
    cubic: dict[tuple[int, int, int], float] = {}
    for line in lines[cubic_start:cubic_end]:
        match = _ROW3.match(line)
        if match:
            indices = tuple(sorted(a_to_h[int(value)] for value in match.groups()[:3]))
            cubic[indices] = _float(match.group(4))
    quartic: dict[tuple[int, int, int, int], float] = {}
    for line in lines[quartic_start:quartic_end]:
        match = _ROW4.match(line)
        if match:
            indices = tuple(sorted(a_to_h[int(value)] for value in match.groups()[:4]))
            quartic[indices] = _float(match.group(5))
    if not cubic or not quartic:
        raise ValueError(f"Gaussian cubic/quartic normal-mode tables not found in {source}")
    return GaussianReducedNormalQFF(frequencies, cubic, quartic, str(source))


def _heading(lines: list[str], text: str) -> int:
    matches = [index for index, line in enumerate(lines) if text in line]
    if not matches:
        raise ValueError(f"Gaussian normal-mode heading not found: {text}")
    return matches[-1]


def _section_index(lines: list[str], name: str) -> int:
    for index, line in enumerate(lines):
        if line.strip() == name:
            return index
    raise ValueError(f"Gaussian external-data section not found: {name}")


def _find_after(lines: list[str], start: int, text: str) -> int:
    for index in range(start + 1, len(lines)):
        if text in lines[index]:
            return index
    raise ValueError(f"Gaussian normal-mode table terminator not found: {text}")


def _ascending_indices(values: tuple[str, ...], mode_count: int):
    return tuple(sorted(mode_count - int(value) for value in values))


def _float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))
