from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
import re

import numpy as np

from matrix_chem import MolecularGeometry
from matrix_chem.topology.elements import atomic_symbol
from matrix_rovib import cartesian_cubic_from_normal_modes
from matrix_qm import HESSIAN_EIGENVALUE_TO_CM

from .cartesian_cubic import GaussianCartesianCubicForceField
from .fchk import BOHR_TO_ANGSTROM, read_gaussian_fchk
from .normal_qff import CM_PER_HARTREE


_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][-+]?\d+)?"
_CUBIC_ROW = re.compile(
    rf"^\s*(\d+)\s+(\d+)\s+(\d+)\s+({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s*$"
)


@dataclass(frozen=True)
class GaussianAnharmonicNormalCubic:
    """Full Gaussian ``Freq=Anharm`` cubic field in harmonic-mode order."""

    path: Path
    harmonic_frequencies_cm1: np.ndarray
    cubic_qmw_hartree_amu32_bohr3: np.ndarray
    anharmonic_to_harmonic: tuple[int, ...]
    coefficient_source: str = "printed_atomic_units"


def read_gaussian_anharmonic_normal_cubic(
    path: Path | str,
    *,
    coefficient_source: str = "auto",
) -> GaussianAnharmonicNormalCubic:
    """Read the complete normal-coordinate cubic table from ``Freq=Anharm``.

    Gaussian prints the anharmonic tables grouped by irreducible
    representation.  The explicit (H)/(A) equivalence table is therefore
    applied before the tensor is returned.  Missing symmetry-forbidden rows
    and below-threshold rows are represented by the explicit printed model's
    zeros. At a minimum, ``auto`` converts the documented reduced FI column
    instead of reading the much more rounded atomic-unit K column. Logs without
    a documented FI column retain the legacy atomic-unit convention. Saddles
    retain their real K convention; their complex FI normalization is owned by
    the separate saddle-QFF adapter. ``reduced`` requires a stable minimum.
    """

    target = Path(path)
    if coefficient_source not in {"auto", "reduced", "atomic_units"}:
        raise ValueError("coefficient_source must be auto, reduced or atomic_units")
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    heading = _last_heading(lines, "CUBIC FORCE CONSTANTS IN NORMAL MODES")
    end = _first_after(lines, heading, "Num. of 3rd derivatives")
    quadratic = _last_heading_before(lines, "QUADRATIC FORCE CONSTANTS IN NORMAL MODES", heading)
    frequencies_a: dict[int, float] = {}
    for line in lines[quadratic:heading]:
        parts = line.split()
        if len(parts) >= 5 and parts[0].isdigit() and parts[1] == parts[0]:
            frequencies_a[int(parts[0])] = _number(parts[2])
    if not frequencies_a:
        raise ValueError(f"Gaussian Freq=Anharm quadratic table not found in {target}")
    nvib = max(frequencies_a)
    h_to_a = _harmonic_to_anharmonic_order(lines, nvib)
    a_to_h = {a: h for h, a in enumerate(h_to_a, start=1)}
    if set(a_to_h) != set(range(1, nvib + 1)):
        raise ValueError("Gaussian anharmonic mode equivalence table is incomplete")
    stable_minimum = all(value > 0.0 for value in frequencies_a.values())
    documented_reduced_column = any(
        "FI(I,J,K)" in line or ("FI =" in line and "Reduced" in line)
        for line in lines[heading:end]
    )
    use_reduced = coefficient_source == "reduced" or (
        coefficient_source == "auto" and stable_minimum and documented_reduced_column
    )
    if use_reduced and not stable_minimum:
        raise ValueError("reduced cubic conversion requires a minimum; use the saddle-QFF adapter")
    gamma = {
        index: np.sqrt(value * CM_PER_HARTREE) / HESSIAN_EIGENVALUE_TO_CM
        for index, value in frequencies_a.items()
    } if use_reduced else {}

    cubic = np.zeros((nvib, nvib, nvib), dtype=float)
    parsed = 0
    for line in lines[heading:end]:
        match = _CUBIC_ROW.match(line)
        if match is None:
            continue
        a_indices = tuple(int(match.group(i)) for i in (1, 2, 3))
        h_indices = tuple(a_to_h[index] - 1 for index in a_indices)
        if use_reduced:
            value = (
                _number(match.group(4))
                * float(np.prod([gamma[index] for index in a_indices]))
                / CM_PER_HARTREE
            )
        else:
            value = _number(match.group(6))
        for permuted in set(permutations(h_indices)):
            cubic[permuted] = value
        parsed += 1
    if not parsed:
        raise ValueError(f"Gaussian Freq=Anharm cubic table not found in {target}")
    frequencies_h = np.zeros(nvib, dtype=float)
    for h, a in enumerate(h_to_a):
        frequencies_h[h] = frequencies_a[a]
    return GaussianAnharmonicNormalCubic(
        path=target,
        harmonic_frequencies_cm1=frequencies_h,
        cubic_qmw_hartree_amu32_bohr3=cubic,
        anharmonic_to_harmonic=tuple(a_to_h[index] for index in range(1, nvib + 1)),
        coefficient_source=("printed_FI_converted_to_QMW" if use_reduced else "printed_atomic_units"),
    )


def cartesian_cubic_from_gaussian_anharmonic(
    log_path: Path | str,
    fchk_path: Path | str,
    *,
    zero_tolerance: float = 0.0,
) -> GaussianCartesianCubicForceField:
    """Recover the BO Cartesian F3 from a Gaussian ``Freq=Anharm`` result.

    The fchk supplies the harmonic eigenvectors in Gaussian's own orientation
    and gauge.  The published 3N linear transformation then restores a
    Cartesian tensor before any isotopic masses or Eckart frame are selected.
    """

    normal = read_gaussian_anharmonic_normal_cubic(log_path)
    fchk = read_gaussian_fchk(Path(fchk_path))
    natoms = int(fchk.atomic_numbers.size)
    dimension = 3 * natoms
    nvib = normal.harmonic_frequencies_cm1.size
    if fchk.normal_modes.size != nvib * dimension:
        raise ValueError("Gaussian fchk normal-mode dimension disagrees with Freq=Anharm")
    if fchk.reduced_masses_amu.size != nvib:
        raise ValueError("Gaussian fchk does not contain one reduced mass per vibration")
    if fchk.harmonic_frequencies_cm.size == nvib and not np.allclose(
        fchk.harmonic_frequencies_cm,
        normal.harmonic_frequencies_cm1,
        rtol=0.0,
        atol=0.05,
    ):
        raise ValueError("Gaussian log/fchk harmonic-mode orders are inconsistent")
    modes = np.asarray(fchk.normal_modes, dtype=float).reshape((nvib, dimension))
    indices, values = cartesian_cubic_from_normal_modes(
        normal.cubic_qmw_hartree_amu32_bohr3,
        modes,
        fchk.masses_amu,
        fchk.reduced_masses_amu,
        zero_tolerance=zero_tolerance,
    )
    geometry = MolecularGeometry(
        atoms=tuple(atomic_symbol(int(number)) for number in fchk.atomic_numbers),
        coordinates_angstrom=np.asarray(fchk.cartesian_coordinates_bohr, dtype=float)
        * BOHR_TO_ANGSTROM,
        comment=Path(log_path).stem,
        source_format="gaussian_freq_anharm",
        source_path=Path(log_path),
    )
    return GaussianCartesianCubicForceField(
        path=Path(log_path),
        geometry=geometry,
        gradient_hartree_per_bohr=np.zeros(dimension, dtype=float),
        hessian_hartree_per_bohr2=_lower_to_symmetric(
            np.asarray(fchk.cartesian_hessian_lower, dtype=float), dimension
        ),
        cubic_unique_indices=indices,
        cubic_unique_values_hartree_per_bohr3=values,
    )


def _harmonic_to_anharmonic_order(lines: list[str], nvib: int) -> tuple[int, ...]:
    starts = [i for i, line in enumerate(lines) if "Input/Output information" in line]
    if not starts:
        return tuple(range(1, nvib + 1))
    mapping: dict[int, int] = {}
    expected = set(range(1, nvib + 1))
    for idx in range(starts[-1] + 1, len(lines)):
        if "FORCE CONSTANTS IN NORMAL MODES" in lines[idx]:
            break
        if not re.match(r"^\s*\(H\)\s*\|", lines[idx]):
            continue
        if idx + 1 >= len(lines) or not re.match(
            r"^\s*\(A\)\s*\|", lines[idx + 1]
        ):
            raise ValueError("Gaussian harmonic/anharmonic mode table has an unpaired row")
        harmonic = [int(value) for value in re.findall(r"\d+", lines[idx])]
        anharmonic = [int(value) for value in re.findall(r"\d+", lines[idx + 1])]
        if not harmonic or len(harmonic) != len(anharmonic):
            raise ValueError("Gaussian harmonic/anharmonic mode table has unequal rows")
        for h, a in zip(harmonic, anharmonic, strict=True):
            if h not in expected or a not in expected or h in mapping or a in mapping.values():
                raise ValueError("Gaussian harmonic/anharmonic mode table is not a permutation")
            mapping[h] = a
    if set(mapping) != expected or set(mapping.values()) != expected:
        raise ValueError("Gaussian harmonic/anharmonic mode equivalence table is incomplete")
    return tuple(mapping[h] for h in range(1, nvib + 1))


def _last_heading(lines: list[str], text: str) -> int:
    matches = [index for index, line in enumerate(lines) if text in line]
    if not matches:
        raise ValueError(f"Gaussian Freq=Anharm section not found: {text}")
    return matches[-1]


def _last_heading_before(lines: list[str], text: str, before: int) -> int:
    matches = [index for index, line in enumerate(lines[:before]) if text in line]
    if not matches:
        raise ValueError(f"Gaussian Freq=Anharm section not found: {text}")
    return matches[-1]


def _first_after(lines: list[str], start: int, text: str) -> int:
    for index in range(start + 1, len(lines)):
        if text in lines[index]:
            return index
    raise ValueError(f"Gaussian Freq=Anharm section terminator not found: {text}")


def _number(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def _lower_to_symmetric(values: np.ndarray, dimension: int) -> np.ndarray:
    expected = dimension * (dimension + 1) // 2
    if values.size != expected:
        raise ValueError("Gaussian fchk Cartesian Hessian has the wrong size")
    result = np.zeros((dimension, dimension), dtype=float)
    cursor = 0
    for i in range(dimension):
        for j in range(i + 1):
            result[i, j] = result[j, i] = values[cursor]
            cursor += 1
    return result


__all__ = [
    "GaussianAnharmonicNormalCubic",
    "cartesian_cubic_from_gaussian_anharmonic",
    "read_gaussian_anharmonic_normal_cubic",
]
