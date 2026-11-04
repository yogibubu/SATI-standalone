from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

import numpy as np

from .modes import GAUSSIAN_MODE_CONVERSION_PROTOCOL, gaussian_cartesian_modes_to_mass_weighted

BOHR_TO_ANGSTROM = 0.529177210903

_FLOAT_TOKEN_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][-+]?\d+)?"
_HEADER_RE = re.compile(
    rf"^(?P<label>.+?)\s+(?P<kind>[IRC])(?:\s+N=\s*(?P<count>\d+)|\s+(?P<scalar>{_FLOAT_TOKEN_RE}))\s*$"
)


@dataclass(frozen=True)
class FCHKData:
    """Gaussian FCHK adapter output for harmonic and anharmonic consumers."""

    atomic_numbers: np.ndarray
    cartesian_coordinates_bohr: np.ndarray
    masses_amu: np.ndarray
    cartesian_hessian_lower: np.ndarray
    harmonic_frequencies_cm: np.ndarray
    anharmonic_frequencies_cm: np.ndarray
    anharmonic_e2: np.ndarray
    normal_modes: np.ndarray
    mulliken_charges: np.ndarray
    reduced_masses_amu: np.ndarray
    dipole_moment_au: np.ndarray
    dipole_derivatives_au_per_bohr: np.ndarray
    ir_intensities_km_mol: np.ndarray
    cartesian_gradient_hartree_per_bohr: np.ndarray = field(
        default_factory=lambda: np.array((), dtype=float)
    )
    total_energy_hartree: float | None = None
    multiplicity: int | None = None
    alpha_orbital_energies_hartree: tuple[float, ...] = ()
    beta_orbital_energies_hartree: tuple[float, ...] = ()
    has_total_scf_density: bool = False

    def to_hessian_input(self):
        from matrix_qm import HessianInput

        hessian = lower_to_symmetric(self.cartesian_hessian_lower)
        data = HessianInput(
            atomic_numbers=self.atomic_numbers,
            cartesian_coordinates_bohr=self.cartesian_coordinates_bohr,
            masses_amu=self.masses_amu,
            cartesian_hessian=hessian,
            harmonic_frequencies_cm=self.harmonic_frequencies_cm,
            source="gaussian-fchk",
        )
        data.validate()
        return data

    def to_anharmonic_input(self):
        from matrix_vpt2_vci.models import AnharmonicInput

        data = AnharmonicInput(
            harmonic_frequencies_cm=self.harmonic_frequencies_cm,
            anharmonic_frequencies_cm=self.anharmonic_frequencies_cm,
            cubic_cm={},
            quartic_cm={},
            source="gaussian-fchk",
        )
        data.validate()
        return data


@dataclass(frozen=True)
class GaussianFCHKPointResult:
    """Geometry, energy, and analytic gradient from a formatted checkpoint.

    Unlike :class:`FCHKData`, this deliberately has no vibrational-data
    requirements and is therefore valid for an ordinary ``Force`` job.
    """

    atomic_numbers: np.ndarray
    cartesian_coordinates_bohr: np.ndarray
    cartesian_gradient_hartree_per_bohr: np.ndarray
    total_energy_hartree: float | None


@dataclass(frozen=True)
class GaussianFCHKPromotion:
    xyzin: Path
    fchk_path: Path
    wrote_cartesian_hessian: bool
    wrote_normal_modes: bool
    wrote_qff: bool
    wrote_electronic: bool = False
    wrote_orbitals: bool = False


def read_gaussian_fchk(path: Path) -> FCHKData:
    """Read harmonic Hessian and Gaussian anharmonic arrays from an FCHK file."""
    blocks = _read_fchk_blocks(Path(path))
    atomic_numbers = _first_array(blocks, "Atomic numbers").astype(int)
    coords = _first_array(blocks, "Current cartesian coordinates").reshape((-1, 3))
    masses = _first_array(
        blocks,
        "Real atomic weights",
        "Atomic masses",
        "Vib-AtMass",
        "Anharmonic Vib-AtMass",
    )
    hessian = _first_array(blocks, "Cartesian Force Constants")
    vib_e2 = _first_array(blocks, "Vib-E2") if "Vib-E2" in blocks else np.array((), dtype=float)
    anh_e2 = (
        _first_array(blocks, "Anharmonic Vib-E2")
        if "Anharmonic Vib-E2" in blocks
        else np.array((), dtype=float)
    )
    # ``Vib-NDim`` is Gaussian's internal atomic dimension, not 3N-6.
    # The ordinary Vib-Modes block is the harmonic mode matrix; the much larger
    # Anharmonic Vib-Modes block has a different meaning and must not replace it.
    modes = _optional_array(blocks, "Vib-Modes")
    alpha_orbital_energies = _optional_array(blocks, "Alpha Orbital Energies")
    beta_orbital_energies = _optional_array(blocks, "Beta Orbital Energies")
    mulliken_charges = _optional_array(blocks, "Mulliken Charges")
    dipole_moment = _optional_array(blocks, "Dipole Moment")
    dipole_derivatives = _optional_array(blocks, "Dipole Derivatives")
    cartesian_gradient = _optional_array(blocks, "Cartesian Gradient")

    coordinate_count = 3 * len(atomic_numbers)
    mode_count = (
        int(modes.size // coordinate_count)
        if modes.size and modes.size % coordinate_count == 0
        else max(0, coordinate_count - 6)
    )
    harmonic = vib_e2[:mode_count] if vib_e2.size else np.array((), dtype=float)
    reduced_masses = (
        vib_e2[mode_count : 2 * mode_count]
        if vib_e2.size >= 2 * mode_count
        else np.array((), dtype=float)
    )
    ir_intensities = (
        vib_e2[3 * mode_count : 4 * mode_count]
        if vib_e2.size >= 4 * mode_count
        else np.array((), dtype=float)
    )
    n_anh = mode_count
    anharmonic = anh_e2[:n_anh] if anh_e2.size else np.array((), dtype=float)
    return FCHKData(
        atomic_numbers=atomic_numbers,
        cartesian_coordinates_bohr=coords,
        masses_amu=masses,
        cartesian_hessian_lower=hessian,
        harmonic_frequencies_cm=harmonic,
        anharmonic_frequencies_cm=anharmonic,
        anharmonic_e2=anh_e2,
        normal_modes=modes,
        mulliken_charges=mulliken_charges,
        reduced_masses_amu=reduced_masses,
        dipole_moment_au=dipole_moment,
        dipole_derivatives_au_per_bohr=dipole_derivatives,
        ir_intensities_km_mol=ir_intensities,
        cartesian_gradient_hartree_per_bohr=cartesian_gradient,
        total_energy_hartree=_optional_scalar(blocks, "Total Energy", "SCF Energy"),
        multiplicity=_optional_int_scalar(blocks, "Multiplicity"),
        alpha_orbital_energies_hartree=tuple(float(value) for value in alpha_orbital_energies),
        beta_orbital_energies_hartree=tuple(float(value) for value in beta_orbital_energies),
        has_total_scf_density="Total SCF Density" in blocks,
    )


def read_gaussian_fchk_point_result(path: Path) -> GaussianFCHKPointResult:
    """Read the non-rounded point derivatives written by ``formchk``."""

    blocks = _read_fchk_blocks(Path(path))
    atomic_numbers = _first_array(blocks, "Atomic numbers").astype(int)
    coordinates = _first_array(blocks, "Current cartesian coordinates").reshape((-1, 3))
    gradient = _first_array(blocks, "Cartesian Gradient").reshape(-1)
    if coordinates.shape != (len(atomic_numbers), 3):
        raise ValueError("Gaussian FCHK Cartesian coordinates have the wrong dimension")
    if gradient.shape != (3 * len(atomic_numbers),):
        raise ValueError("Gaussian FCHK Cartesian gradient has the wrong dimension")
    return GaussianFCHKPointResult(
        atomic_numbers=atomic_numbers,
        cartesian_coordinates_bohr=coordinates,
        cartesian_gradient_hartree_per_bohr=gradient,
        total_energy_hartree=_optional_scalar(blocks, "Total Energy", "SCF Energy"),
    )


def read_gaussian_fchk_geometry(path: Path):
    """Read geometry, charge and multiplicity from a Gaussian FCHK/FCH file."""
    from matrix_chem import MolecularGeometry
    from matrix_chem.topology.elements import atomic_symbol

    target = Path(path)
    blocks = _read_fchk_blocks(target)
    atomic_numbers = _first_array(blocks, "Atomic numbers").astype(int)
    coords_bohr = _first_array(blocks, "Current cartesian coordinates").reshape((-1, 3))
    charge = _optional_int_scalar(blocks, "Charge")
    multiplicity = _optional_int_scalar(blocks, "Multiplicity")
    return MolecularGeometry(
        atoms=tuple(atomic_symbol(int(number)) for number in atomic_numbers),
        coordinates_angstrom=np.asarray(coords_bohr, dtype=float) * BOHR_TO_ANGSTROM,
        comment=target.stem,
        source_format="gaussian_fchk",
        source_path=target,
        charge=charge,
        multiplicity=multiplicity,
    )


def hessian_input_from_gaussian_fchk(path: Path):
    """Gaussian FCHK adapter: return the canonical ORACLE Hessian input."""
    return read_gaussian_fchk(path).to_hessian_input()


def read_gaussian_fchk_qff(path: Path) -> FCHKData:
    """Compatibility alias for the Gaussian FCHK/QFF adapter."""
    return read_gaussian_fchk(path)


def anharmonic_input_from_gaussian_fchk(path: Path):
    """Gaussian FCHK adapter: return the canonical ORACLE anharmonic input."""
    return read_gaussian_fchk(path).to_anharmonic_input()


def promote_gaussian_fchk_to_xyzin(
    fchk_path: Path | str,
    xyzin: Path | str,
    *,
    write_cartesian_hessian: bool = True,
    write_normal_modes: bool = True,
    write_qff: bool = True,
    write_electronic: bool = True,
    write_orbitals: bool = True,
) -> GaussianFCHKPromotion:
    """Promote Gaussian FCHK harmonic/QFF payloads into shared MATRIX xyzin sections."""
    from matrix_qm import (
        ElectronicSection,
        ElectronicStateRecord,
        cartesian_hessian_section_from_hessian_input,
        merge_orbitals_section,
        normal_modes_section_from_arrays,
        orbital_file_record_from_path,
        qff_section_from_anharmonic_input,
        write_cartesian_hessian_section,
        write_electronic_section,
        write_normal_modes_section,
        write_qff_section,
    )

    source = Path(fchk_path)
    target = Path(xyzin)
    data = read_gaussian_fchk(source)
    wrote_hessian = False
    wrote_modes = False
    wrote_force_field = False
    wrote_electronic_section = False
    wrote_orbitals_section = False
    if write_cartesian_hessian:
        write_cartesian_hessian_section(
            target,
            cartesian_hessian_section_from_hessian_input(
                data.to_hessian_input(),
                source="gaussian-fchk",
                atomic_charges=data.mulliken_charges,
                charge_model="Mulliken" if data.mulliken_charges.size else "",
            ),
        )
        wrote_hessian = True
    if write_normal_modes and data.normal_modes.size:
        coordinate_count = 3 * len(data.atomic_numbers)
        mass_weighted_modes = gaussian_cartesian_modes_to_mass_weighted(
            data.normal_modes,
            data.masses_amu,
            coordinate_count=coordinate_count,
        )
        frequencies = (
            data.anharmonic_frequencies_cm
            if data.anharmonic_frequencies_cm.size
            else data.harmonic_frequencies_cm
        )
        write_normal_modes_section(
            target,
            normal_modes_section_from_arrays(
                frequencies,
                mass_weighted_modes,
                source=f"gaussian-fchk; {GAUSSIAN_MODE_CONVERSION_PROTOCOL}",
                coordinate_count=coordinate_count,
            ),
        )
        wrote_modes = True
    if write_qff:
        write_qff_section(
            target,
            qff_section_from_anharmonic_input(data.to_anharmonic_input(), source="gaussian-fchk"),
        )
        wrote_force_field = True
    if write_electronic and data.total_energy_hartree is not None:
        write_electronic_section(
            target,
            ElectronicSection(
                (
                    ElectronicStateRecord(
                        label="S0",
                        energy_hartree=data.total_energy_hartree,
                        energy_ev=0.0,
                        multiplicity="" if data.multiplicity is None else str(data.multiplicity),
                        source="gaussian-fchk",
                    ),
                )
            ),
        )
        wrote_electronic_section = True
    if write_orbitals:
        records = [
            orbital_file_record_from_path(source, role="orbitals", source="gaussian-fchk"),
        ]
        if data.has_total_scf_density:
            records.append(
                orbital_file_record_from_path(source, role="density", source="gaussian-fchk")
            )
        merge_orbitals_section(target, tuple(records))
        wrote_orbitals_section = True
    return GaussianFCHKPromotion(
        xyzin=target,
        fchk_path=source,
        wrote_cartesian_hessian=wrote_hessian,
        wrote_normal_modes=wrote_modes,
        wrote_qff=wrote_force_field,
        wrote_electronic=wrote_electronic_section,
        wrote_orbitals=wrote_orbitals_section,
    )


def read_indexed_qff_text(path: Path, frequencies_cm: np.ndarray | None = None):
    """Read an indexed cubic/quartic normal-coordinate force field.

    Accepted records are intentionally simple and solver-independent:

    - `FREQ i value_cm`
    - `CUBIC i j k value_cm`
    - `QUARTIC i j k l value_cm`

    Mode indices are one-based in the file and converted to zero-based tuples.
    Header lines from Gaussian-like sections are ignored, so extracted blocks can
    be pasted directly into a normalized `.qff` file.
    """
    from matrix_vpt2_vci.vci import QuarticForceField

    freqs: dict[int, float] = {}
    cubic: dict[tuple[int, int, int], float] = {}
    quartic: dict[tuple[int, int, int, int], float] = {}
    for raw in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        parts = line.replace(",", " ").split()
        tag = parts[0].upper()
        try:
            if tag in {"FREQ", "FREQUENCY"} and len(parts) >= 3:
                freqs[int(parts[1]) - 1] = float(parts[2].replace("D", "E"))
            elif tag in {"CUBIC", "C3"} and len(parts) >= 5:
                key = tuple(sorted(int(value) - 1 for value in parts[1:4]))
                cubic[key] = float(parts[4].replace("D", "E"))
            elif tag in {"QUARTIC", "C4"} and len(parts) >= 6:
                key = tuple(sorted(int(value) - 1 for value in parts[1:5]))
                quartic[key] = float(parts[5].replace("D", "E"))
            elif len(parts) in {4, 5} and all(token.lstrip("+-").isdigit() for token in parts[:-1]):
                key = tuple(sorted(int(value) - 1 for value in parts[:-1]))
                value = float(parts[-1].replace("D", "E"))
                if len(key) == 3:
                    cubic[key] = value
                else:
                    quartic[key] = value
        except ValueError:
            continue

    if frequencies_cm is None:
        if not freqs:
            raise ValueError("No frequencies found in indexed QFF text")
        n_modes = max(freqs) + 1
        frequencies = np.array([freqs[idx] for idx in range(n_modes)], dtype=float)
    else:
        frequencies = np.asarray(frequencies_cm, dtype=float)
        if freqs:
            frequencies = frequencies.copy()
            for index, value in freqs.items():
                if index >= len(frequencies):
                    raise ValueError("QFF frequency index exceeds supplied frequency array")
                frequencies[index] = value
    return QuarticForceField(frequencies, cubic, quartic)


def lower_to_symmetric(lower: np.ndarray) -> np.ndarray:
    """Convert Gaussian lower-triangular packed storage to a full matrix."""
    n_float = (np.sqrt(8 * len(lower) + 1) - 1) / 2
    n = int(round(n_float))
    if n * (n + 1) // 2 != len(lower):
        raise ValueError("Packed lower-triangular array has an invalid length")
    mat = np.zeros((n, n), dtype=float)
    idx = 0
    for i in range(n):
        for j in range(i + 1):
            mat[i, j] = lower[idx]
            mat[j, i] = lower[idx]
            idx += 1
    return mat


def _read_fchk_blocks(path: Path) -> dict[str, np.ndarray | int | float | str]:
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    blocks: dict[str, np.ndarray | int | float | str] = {}
    i = 0
    while i < len(lines):
        match = _HEADER_RE.match(lines[i])
        if match is None:
            i += 1
            continue
        label = match.group("label").strip()
        kind = match.group("kind")
        count = match.group("count")
        scalar = match.group("scalar")
        if count is None:
            if scalar is not None:
                blocks[label] = int(scalar) if kind == "I" else float(scalar)
            i += 1
            continue

        nvalues = int(count)
        raw: list[str] = []
        i += 1
        while len(raw) < nvalues and i < len(lines):
            raw.extend(lines[i].split())
            i += 1
        if kind == "I":
            blocks[label] = np.array([int(x) for x in raw[:nvalues]], dtype=int)
        elif kind == "R":
            blocks[label] = np.array(
                [float(x.replace("D", "E")) for x in raw[:nvalues]], dtype=float
            )
        else:
            blocks[label] = " ".join(raw[:nvalues])
    return blocks


def _first_array(blocks: dict[str, object], *labels: str) -> np.ndarray:
    for label in labels:
        value = blocks.get(label)
        if isinstance(value, np.ndarray):
            return value.astype(float, copy=False)
    raise ValueError(f"FCHK block not found: {' / '.join(labels)}")


def _optional_array(blocks: dict[str, object], *labels: str) -> np.ndarray:
    for label in labels:
        value = blocks.get(label)
        if isinstance(value, np.ndarray):
            return value.astype(float, copy=False)
    return np.array((), dtype=float)


def _optional_scalar(blocks: dict[str, object], *labels: str) -> float | None:
    for label in labels:
        value = blocks.get(label)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _optional_int_scalar(blocks: dict[str, object], *labels: str) -> int | None:
    value = _optional_scalar(blocks, *labels)
    return None if value is None else int(value)
