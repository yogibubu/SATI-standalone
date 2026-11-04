from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess
from typing import Mapping

import numpy as np

from matrix_chem import (
    MolecularGeometry,
    atomic_mass,
    expected_vibrational_mode_count,
    is_linear_geometry,
)
from matrix_chem.geometry_io import GeometryParseError, normalize_atom_symbol
from matrix_chem.topology.elements import atomic_number


ORCA_OUTPUT_FORMAT = "orca_output"
ANGSTROM_TO_BOHR = 1.0 / 0.52917721092
BOHR_TO_ANGSTROM = 0.52917721092
FINAL_ENERGY_RE = re.compile(
    r"FINAL\s+SINGLE\s+POINT\s+ENERGY\s+([-+]?(?:\d+\.\d*|\.\d+)(?:[DEde][-+]?\d+)?)",
    re.I,
)
ORCA_STATE_RE = re.compile(
    r"^\s*STATE\s+(\d+)\s*:\s*E=\s*"
    r"([-+]?(?:\d+\.\d*|\.\d+)(?:[DEde][-+]?\d+)?)\s+au",
    re.I,
)
ORCA_EXCITATION_RE = re.compile(
    r"^\s*(\d+)a\s*->\s*(\d+)a\s*:\s*"
    r"[-+]?(?:\d+\.\d*|\.\d+)(?:[DEde][-+]?\d+)?\s*"
    r"\(c=\s*([-+]?(?:\d+\.\d*|\.\d+)(?:[DEde][-+]?\d+)?)\)",
    re.I,
)
ORCA_BASIS_COUNT_RE = re.compile(
    r"^\s*Number\s+of\s+basis\s+functions\s+\.\.\.\s*(\d+)\s*$",
    re.I,
)
ORCA_ELECTRIC_DIPOLE_RE = re.compile(
    r"^\s*0-\d+\S*\s*->\s*(\d+)-\d+\S*\s+"
    r"[-+]?\d+(?:\.\d*)?\s+[-+]?\d+(?:\.\d*)?\s+"
    r"[-+]?\d+(?:\.\d*)?\s+[-+]?\d+(?:\.\d*)?\s+"
    r"[-+]?\d+(?:\.\d*)?\s+"
    r"([-+]?\d+(?:\.\d*)?(?:[DEde][-+]?\d+)?)\s+"
    r"([-+]?\d+(?:\.\d*)?(?:[DEde][-+]?\d+)?)\s+"
    r"([-+]?\d+(?:\.\d*)?(?:[DEde][-+]?\d+)?)\s*$"
)
FREQUENCY_RE = re.compile(
    r"^\s*\d+\s*:\s*([-+]?(?:\d+\.\d*|\.\d+)(?:[DEde][-+]?\d+)?)\s*cm\*\*-1",
    re.I,
)
CHARGE_RE = re.compile(r"\bTotal\s+Charge\b.*?([-+]?\d+)\s*$", re.I)
MULT_RE = re.compile(r"\bMultiplicity\b.*?(\d+)\s*$", re.I)
XYZ_INPUT_RE = re.compile(r"^\s*\*\s+xyz(?:file)?\s+([-+]?\d+)\s+(\d+)\b", re.I)
OPTIMIZATION_CYCLE_RE = re.compile(
    r"GEOMETRY\s+OPTIMIZATION\s+CYCLE\s+\d+",
    re.I,
)


@dataclass(frozen=True)
class OrcaOutputSummary:
    path: Path
    geometry: MolecularGeometry
    charge: int
    multiplicity: int
    final_energy_hartree: float | None
    frequencies_cm: tuple[float, ...]
    cartesian_hessian: np.ndarray | None
    cartesian_coordinate_blocks: int
    optimization_steps: int
    normal_termination: bool


@dataclass(frozen=True)
class OrcaOutputPromotion:
    xyzin: Path
    output_path: Path
    wrote_geometry: bool
    wrote_cartesian_hessian: bool


@dataclass(frozen=True)
class OrcaMoldenPromotion:
    xyzin: Path
    gbw_path: Path
    molden_path: Path
    command: tuple[str, ...]
    wrote_orbitals: bool


@dataclass(frozen=True)
class OrcaPointResult:
    """Single-point energy and Cartesian gradient parsed from ORCA outputs."""

    path: Path
    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray
    normal_termination: bool


def orca_state_fingerprint_data(
    path: Path | str,
    *,
    molden: Path | str | None = None,
):
    """Parse ORCA TDA roots into an APOC state-fingerprint manifold.

    When a Molden orbital export is supplied, the printed occupied--virtual
    amplitudes are transformed to AO transition densities.  This is the
    transferable representation required for comparisons between geometries:
    raw MO amplitudes cannot be compared directly because the molecular
    orbitals may rotate between SCF calculations.  The raw amplitude form is
    retained only for parsing legacy outputs that have no orbital export.
    """
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    roots: dict[int, float] = {}
    transitions: dict[int, list[tuple[int, int, float]]] = {}
    current: int | None = None
    state_header_index: int | None = None
    for line_index, line in enumerate(lines):
        match = ORCA_STATE_RE.match(line)
        if match:
            if state_header_index is None:
                state_header_index = line_index
            current = int(match.group(1))
            roots[current] = float(match.group(2).replace("D", "E").replace("d", "e"))
            transitions[current] = []
            continue
        match = ORCA_EXCITATION_RE.match(line)
        if match and current is not None:
            transitions[current].append(
                (
                    int(match.group(1)),
                    int(match.group(2)),
                    float(match.group(3).replace("D", "E").replace("d", "e")),
                )
            )
    if not roots:
        return None
    max_occ = max((item[0] for values in transitions.values() for item in values), default=0)
    max_virt = max((item[1] for values in transitions.values() for item in values), default=0)
    nmo = None
    if state_header_index is not None:
        for line in reversed(lines[:state_header_index]):
            match = ORCA_BASIS_COUNT_RE.match(line)
            if match:
                nmo = int(match.group(1))
                break
    # ORCA can print several basis counts for RI and auxiliary spaces.  The
    # nearest count before the TDA block is the orbital-space count.  A
    # conservative fallback keeps small synthetic fixtures usable.
    orbital_dimension = max(1, int(nmo or max(max_occ, max_virt) + 1))
    state_ids = np.asarray(sorted(roots), dtype=int)
    excitation_energies = np.asarray([roots[idx] for idx in state_ids], dtype=float)
    if molden is not None:
        from matrix_pyscf import load_molden_wavefunction

        _mol, _energies, coefficients, _occupations, _labels, _spins = (
            load_molden_wavefunction(molden)
        )
        coefficient = coefficients[0] if isinstance(coefficients, tuple) else coefficients
        coefficient = np.asarray(coefficient, dtype=float)
        if coefficient.ndim != 2:
            raise ValueError("ORCA Molden orbital coefficients must be a matrix")
        nao, nmo_molden = coefficient.shape
        vectors = np.zeros((state_ids.size, nao * nao), dtype=float)
        for row, state_id in enumerate(state_ids):
            transition_density = np.zeros((nao, nao), dtype=float)
            for occupied, virtual, amplitude in transitions.get(int(state_id), ()):
                if occupied >= nmo_molden or virtual >= nmo_molden:
                    raise ValueError(
                        "ORCA TDA amplitude references an orbital absent from the Molden export"
                    )
                transition_density += amplitude * np.outer(
                    coefficient[:, occupied], coefficient[:, virtual]
                )
            vectors[row] = transition_density.reshape(-1)
        return (
            state_ids,
            excitation_energies,
            vectors,
            "orca-tda-transition-density-ao",
        )
    width = orbital_dimension * orbital_dimension
    vectors = np.zeros((state_ids.size, width), dtype=float)
    for row, state_id in enumerate(state_ids):
        for occupied, virtual, coefficient in transitions.get(int(state_id), ()):
            index = occupied * orbital_dimension + virtual
            if index < width:
                vectors[row, index] = coefficient
    return state_ids, excitation_energies, vectors, "orca-tda-amplitudes"


def orca_transition_electric_dipoles(path: Path | str) -> dict[int, tuple[float, float, float]]:
    """Return ground-to-excited-state electric transition dipoles in atomic units."""

    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    in_electric_spectrum = False
    dipoles: dict[int, tuple[float, float, float]] = {}
    for line in lines:
        if "ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS" in line:
            in_electric_spectrum = True
            continue
        if in_electric_spectrum and "ABSORPTION SPECTRUM VIA TRANSITION VELOCITY" in line:
            break
        if not in_electric_spectrum:
            continue
        match = ORCA_ELECTRIC_DIPOLE_RE.match(line)
        if match is None:
            continue
        dipoles[int(match.group(1))] = tuple(
            float(match.group(index).replace("D", "E").replace("d", "e"))
            for index in (2, 3, 4)
        )
    return dipoles


def write_orca_state_fingerprint_archive(
    path: Path | str,
    output: Path | str,
    *,
    molden: Path | str | None = None,
) -> Path | None:
    data = orca_state_fingerprint_data(path, molden=molden)
    if data is None:
        return None
    state_ids, excitation_energies, vectors, representation = data
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_ids": state_ids,
        "excitation_energies_hartree": excitation_energies,
        "vectors": vectors,
        "representation": np.asarray([representation]),
    }
    if molden is not None:
        payload["state_source_file"] = np.asarray([str(Path(molden).resolve())])
    np.savez_compressed(target, **payload)
    return target


def write_orca_point_input(
    output: Path | str,
    atoms: tuple[str, ...] | list[str],
    coordinates_angstrom,
    *,
    route: str = "HF STO-3G EnGrad",
    charge: int = 0,
    multiplicity: int = 1,
    electronic_state: int = 0,
    excited_states: int | None = None,
    processors: int = 1,
    memory_gb: int | None = None,
    restart_orbitals: Path | str | None = None,
    restart_projection: str | None = None,
    scf_convergence: Mapping[str, float] | None = None,
    state_fingerprint_print_threshold: float | None = None,
) -> Path:
    """Write an ORCA Cartesian input suitable for LINK point scans."""

    coords = np.asarray(coordinates_angstrom, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise GeometryParseError("ORCA point coordinates must have shape natoms x 3")
    if len(atoms) != coords.shape[0]:
        raise GeometryParseError("ORCA atom count does not match coordinate array")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    route_lines = [line.rstrip() for line in route.strip().splitlines() if line.strip()]
    route_text = route_lines[0] if route_lines else "HF STO-3G EnGrad"
    if route_text.startswith("!"):
        header = route_text
    else:
        header = f"! {route_text}"
    lines = [header]
    lines.extend(route_lines[1:])
    nprocs = max(1, int(processors))
    if nprocs > 1:
        lines.append(f"%pal nprocs {nprocs} end")
    if memory_gb is not None:
        total_mb = int(memory_gb) * 1024
        if total_mb <= 0:
            raise ValueError("ORCA memory must be a positive number of GB")
        lines.append(f"%maxcore {max(1, total_mb // nprocs)}")
    scf_lines: list[str] = []
    if scf_convergence is not None:
        required_scf = {
            "energy_tolerance": "TolE",
            "orbital_gradient_tolerance": "TolG",
            "diis_error_tolerance": "TolErr",
        }
        if set(scf_convergence) != set(required_scf):
            raise ValueError("ORCA SCF convergence contract is incomplete")
        for key, keyword in required_scf.items():
            value = float(scf_convergence[key])
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError("ORCA SCF tolerances must be finite and positive")
            scf_lines.append(f"  {keyword} {value:.12g}")
    if restart_orbitals is not None:
        projection = str(restart_projection or "FMatrix").strip()
        if projection.casefold() not in {"fmatrix", "cmatrix"}:
            raise ValueError("ORCA restart projection must be FMatrix or CMatrix")
        scf_lines.extend(
            [
                "  Guess MORead",
                f'  MOInp "{Path(restart_orbitals)}"',
                f"  GuessMode {projection}",
            ]
        )
    if scf_lines:
        lines.extend(["%scf", *scf_lines, "end"])
    if int(electronic_state) > 0:
        nroots = max(int(excited_states or electronic_state), int(electronic_state))
        tprint = []
        if state_fingerprint_print_threshold is not None:
            threshold = float(state_fingerprint_print_threshold)
            if not 0.0 < threshold < 1.0:
                raise ValueError("state fingerprint print threshold must lie in (0, 1)")
            tprint.append(f"  TPrint {threshold:.12g}")
        lines.extend(
            [
                "%tddft",
                f"  NRoots {nroots}",
                f"  IRoot {int(electronic_state)}",
                "  TDA true",
                *tprint,
                "end",
            ]
        )
    lines.append(f"* xyz {int(charge)} {int(multiplicity)}")
    for atom, xyz in zip(atoms, coords, strict=True):
        lines.append(
            f"{atom:<3s} {float(xyz[0]): .16f} {float(xyz[1]): .16f} {float(xyz[2]): .16f}"
        )
    lines.append("*")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def read_orca_point_result(
    output: Path | str,
    *,
    engrad: Path | str | None = None,
    require_gradient: bool = True,
) -> OrcaPointResult:
    """Read the final energy and Cartesian gradient from an ORCA point job."""

    target = Path(output)
    text = target.read_text(encoding="utf-8", errors="replace")
    gradient_path = Path(engrad) if engrad is not None else target.with_suffix(".engrad")
    if gradient_path.is_file():
        energy, gradient = _parse_orca_engrad(gradient_path)
    else:
        energy = _parse_final_energy(text)
        gradient = _parse_orca_output_gradient(text.splitlines())
    if energy is None:
        raise GeometryParseError(f"ORCA output contains no parseable point energy: {target}")
    if gradient is None and require_gradient:
        raise GeometryParseError(f"ORCA output contains no Cartesian gradient: {target}")
    return OrcaPointResult(
        path=target,
        energy_hartree=float(energy),
        gradient_hartree_per_bohr=(
            None if gradient is None else np.asarray(gradient, dtype=float).reshape(-1)
        ),
        normal_termination="ORCA TERMINATED NORMALLY" in text.upper(),
    )


def summarize_orca_output(path: Path | str) -> OrcaOutputSummary:
    target = Path(path)
    text = target.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    geometry = _parse_last_cartesian_coordinates(lines, source_path=target)
    charge, multiplicity = _parse_charge_multiplicity(lines)
    frequencies = tuple(_parse_frequencies(lines))
    hessian = _parse_cartesian_hessian(lines, size=3 * geometry.natoms)
    if hessian is not None and not np.allclose(hessian, hessian.T):
        hessian = 0.5 * (hessian + hessian.T)
    return OrcaOutputSummary(
        path=target,
        geometry=geometry,
        charge=charge,
        multiplicity=multiplicity,
        final_energy_hartree=_parse_final_energy(text),
        frequencies_cm=frequencies,
        cartesian_hessian=hessian,
        cartesian_coordinate_blocks=_coordinate_block_count(lines),
        optimization_steps=len(OPTIMIZATION_CYCLE_RE.findall(text)),
        normal_termination="ORCA TERMINATED NORMALLY" in text.upper(),
    )


def read_orca_output_geometry(path: Path | str) -> MolecularGeometry:
    return summarize_orca_output(path).geometry


def hessian_input_from_orca_output(path: Path | str):
    from matrix_qm import HessianInput

    target = Path(path)
    summary = summarize_orca_output(target)
    if summary.cartesian_hessian is None:
        hess = target.with_suffix(".hess")
        if hess.is_file():
            return hessian_input_from_orca_hessian_file(hess)
        raise GeometryParseError("ORCA output contains no readable Cartesian Hessian")
    geometry = summary.geometry
    numbers = np.asarray(_atomic_numbers_from_geometry(geometry), dtype=int)
    masses = np.asarray([atomic_mass(int(number)) for number in numbers], dtype=float)
    frequencies, frequency_provenance = _canonical_orca_vibrational_frequencies(
        summary.frequencies_cm,
        geometry.coordinates_angstrom,
    )
    data = HessianInput(
        atomic_numbers=numbers,
        cartesian_coordinates_bohr=np.asarray(geometry.coordinates_angstrom, dtype=float)
        * ANGSTROM_TO_BOHR,
        masses_amu=masses,
        cartesian_hessian=np.asarray(summary.cartesian_hessian, dtype=float),
        harmonic_frequencies_cm=frequencies,
        source="orca-output",
        provenance=frequency_provenance,
    )
    data.validate()
    return data


def hessian_input_from_orca_hessian_file(path: Path | str):
    from matrix_qm import HessianInput

    target = Path(path)
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    atoms, masses, coordinates_bohr = _parse_orca_hess_atoms(lines)
    size = 3 * len(atoms)
    hessian = _parse_orca_hess_matrix(lines, size=size)
    frequencies = _parse_orca_hess_frequencies(lines)
    numbers = np.asarray([atomic_number(atom) for atom in atoms], dtype=int)
    mass_values = np.asarray(masses, dtype=float)
    if mass_values.shape != numbers.shape or np.any(mass_values <= 0.0):
        mass_values = np.asarray([atomic_mass(int(number)) for number in numbers], dtype=float)
    coordinates_array = np.asarray(coordinates_bohr, dtype=float)
    frequencies_array, frequency_provenance = _canonical_orca_vibrational_frequencies(
        frequencies,
        coordinates_array * BOHR_TO_ANGSTROM,
    )
    data = HessianInput(
        atomic_numbers=numbers,
        cartesian_coordinates_bohr=coordinates_array,
        masses_amu=mass_values,
        cartesian_hessian=np.asarray(hessian, dtype=float),
        harmonic_frequencies_cm=frequencies_array,
        source="orca-hess",
        provenance=frequency_provenance,
    )
    data.validate()
    return data


def promote_orca_output_to_xyzin(
    source: Path | str,
    xyzin: Path | str,
    *,
    symmetry_distance: float = 1.0e-3,
    symmetry_inertia: float = 1.0e-3,
    max_rotation_order: int = 6,
) -> OrcaOutputPromotion:
    from matrix_chem import SymmetryThresholds, preprocess_to_enriched_xyz
    from matrix_qm import cartesian_hessian_section_from_hessian_input, write_cartesian_hessian_section

    output = Path(source)
    target = Path(xyzin)
    preprocess_to_enriched_xyz(
        output,
        target,
        source_kind="orca",
        symmetry_thresholds=SymmetryThresholds(
            distance_angstrom=symmetry_distance,
            inertia_relative=symmetry_inertia,
            max_rotation_order=max_rotation_order,
        ),
    )
    wrote_hessian = False
    try:
        hessian_input = hessian_input_from_orca_output(output)
    except GeometryParseError:
        pass
    else:
        write_cartesian_hessian_section(
            target,
            cartesian_hessian_section_from_hessian_input(hessian_input, source="orca-output"),
        )
        wrote_hessian = True
    return OrcaOutputPromotion(
        xyzin=target,
        output_path=output,
        wrote_geometry=True,
        wrote_cartesian_hessian=wrote_hessian,
    )


def convert_orca_gbw_to_molden(
    gbw: Path | str,
    *,
    output: Path | str | None = None,
    executable: str = "orca_2mkl",
    timeout: float | None = None,
) -> OrcaMoldenPromotion:
    """Run ORCA's orca_2mkl converter and return the generated Molden path.

    The returned object has an empty xyzin path because this helper only creates
    the external file. Use :func:`promote_orca_molden_to_xyzin` to register it.
    """
    source = Path(gbw)
    if source.suffix.lower() != ".gbw":
        raise ValueError("ORCA Molden conversion requires a .gbw file")
    if not source.is_file():
        raise FileNotFoundError(source)
    basename = source.with_suffix("")
    generated = basename.with_suffix(".molden.input")
    command = (executable, str(basename), "-molden")
    subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if not generated.is_file():
        raise FileNotFoundError(f"orca_2mkl did not create expected Molden file: {generated}")
    target = Path(output) if output is not None else generated
    if target != generated:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(generated, target)
    return OrcaMoldenPromotion(
        xyzin=Path(),
        gbw_path=source,
        molden_path=target,
        command=command,
        wrote_orbitals=False,
    )


def promote_orca_molden_to_xyzin(
    gbw: Path | str,
    xyzin: Path | str,
    *,
    output: Path | str | None = None,
    executable: str = "orca_2mkl",
    timeout: float | None = None,
) -> OrcaMoldenPromotion:
    """Convert an ORCA GBW file to Molden and register it in #ORBITALS."""
    from matrix_qm import merge_orbitals_section, orbital_file_record_from_path

    target = Path(xyzin)
    converted = convert_orca_gbw_to_molden(
        gbw,
        output=output,
        executable=executable,
        timeout=timeout,
    )
    merge_orbitals_section(
        target,
        (
            orbital_file_record_from_path(
                converted.molden_path,
                role="orbitals",
                label=converted.molden_path.stem,
                source="orca_2mkl",
            ),
        ),
    )
    return OrcaMoldenPromotion(
        xyzin=target,
        gbw_path=converted.gbw_path,
        molden_path=converted.molden_path,
        command=converted.command,
        wrote_orbitals=True,
    )


def _parse_last_cartesian_coordinates(
    lines: list[str],
    *,
    source_path: Path,
) -> MolecularGeometry:
    block = _last_coordinate_block(lines, unit="angstrom")
    unit = "angstrom"
    if block is None:
        block = _last_coordinate_block(lines, unit="bohr")
        unit = "bohr"
    if block is None:
        raise GeometryParseError("ORCA output contains no Cartesian coordinate block")
    atoms, coords = block
    coordinates = np.asarray(coords, dtype=float)
    if unit == "bohr":
        coordinates = coordinates * BOHR_TO_ANGSTROM
    charge, multiplicity = _parse_charge_multiplicity(lines)
    return MolecularGeometry(
        atoms=tuple(atoms),
        coordinates_angstrom=coordinates,
        comment=source_path.name,
        source_format=ORCA_OUTPUT_FORMAT,
        source_path=source_path,
        charge=charge,
        multiplicity=multiplicity,
        metadata={"coordinate_block": f"CARTESIAN COORDINATES ({unit.upper()})"},
    )


def _last_coordinate_block(
    lines: list[str],
    *,
    unit: str,
) -> tuple[list[str], list[list[float]]] | None:
    needle = (
        "CARTESIAN COORDINATES (ANGSTROEM)"
        if unit == "angstrom"
        else "CARTESIAN COORDINATES (A.U.)"
    )
    starts = [idx for idx, line in enumerate(lines) if needle in line.upper()]
    for start in reversed(starts):
        atoms: list[str] = []
        coords: list[list[float]] = []
        for raw in lines[start + 1 :]:
            text = raw.strip()
            if not text or set(text) <= {"-"}:
                if atoms:
                    break
                continue
            parsed = _parse_coordinate_line(raw)
            if parsed is None:
                if atoms:
                    break
                continue
            atom, xyz = parsed
            atoms.append(atom)
            coords.append(xyz)
        if atoms:
            return atoms, coords
    return None


def _parse_coordinate_line(line: str) -> tuple[str, list[float]] | None:
    parts = line.split()
    if len(parts) < 4:
        return None
    try:
        atom = normalize_atom_symbol(parts[0])
    except GeometryParseError:
        return None
    numeric = [_float_or_none(token) for token in parts[1:]]
    values = [value for value in numeric if value is not None]
    if len(values) < 3:
        return None
    return atom, [float(value) for value in values[-3:]]


def _parse_charge_multiplicity(lines: list[str]) -> tuple[int, int]:
    charge = 0
    multiplicity = 1
    for line in lines:
        charge_match = CHARGE_RE.search(line)
        if charge_match is not None:
            charge = int(charge_match.group(1))
        mult_match = MULT_RE.search(line)
        if mult_match is not None:
            multiplicity = int(mult_match.group(1))
        xyz_match = XYZ_INPUT_RE.search(line)
        if xyz_match is not None:
            charge = int(xyz_match.group(1))
            multiplicity = int(xyz_match.group(2))
    return charge, multiplicity


def _parse_final_energy(text: str) -> float | None:
    matches = tuple(FINAL_ENERGY_RE.finditer(text))
    if not matches:
        return None
    return float(matches[-1].group(1).replace("D", "E").replace("d", "e"))


def _parse_orca_engrad(path: Path) -> tuple[float | None, np.ndarray | None]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    energy: float | None = None
    gradient: list[float] = []
    in_gradient = False
    expect_energy = False
    for raw in lines:
        text = raw.strip()
        lower = text.lower()
        if not text:
            continue
        if text.startswith("#") or text.startswith("!"):
            if "current total energy" in lower:
                in_gradient = False
                expect_energy = True
            elif "current gradient" in lower:
                in_gradient = True
                expect_energy = False
            elif in_gradient and ("coordinates" in lower or "atomic numbers" in lower):
                in_gradient = False
            continue
        value = _float_or_none(text.split()[0])
        if value is None:
            continue
        if in_gradient:
            gradient.append(value)
        elif expect_energy:
            energy = value
            expect_energy = False
    if energy is None or len(gradient) < 3:
        return None, None
    return energy, np.asarray(gradient, dtype=float)


def _parse_orca_output_gradient(lines: list[str]) -> np.ndarray | None:
    starts = [idx for idx, line in enumerate(lines) if "CARTESIAN GRADIENT" in line.upper()]
    for start in reversed(starts):
        values: list[float] = []
        idx = start + 1
        while idx < len(lines):
            parts = lines[idx].replace(":", " ").split()
            floats = [_float_or_none(token) for token in parts]
            row_values = [value for value in floats if value is not None]
            if len(row_values) >= 3:
                values.extend(row_values[-3:])
            elif values and lines[idx].strip() and not lines[idx].strip().startswith("-"):
                break
            idx += 1
        if values:
            return np.asarray(values, dtype=float)
    return None


def _parse_frequencies(lines: list[str]) -> list[float]:
    values: list[float] = []
    in_block = False
    for line in lines:
        upper = line.upper()
        if "VIBRATIONAL FREQUENCIES" in upper:
            values = []
            in_block = True
            continue
        if not in_block:
            continue
        match = FREQUENCY_RE.match(line)
        if match is not None:
            values.append(float(match.group(1).replace("D", "E").replace("d", "e")))
            continue
        if values and upper.strip() and upper == upper.upper() and ":" not in line:
            break
    return values


def _parse_cartesian_hessian(lines: list[str], *, size: int) -> np.ndarray | None:
    starts = [
        idx
        for idx, line in enumerate(lines)
        if "CARTESIAN FORCE CONSTANT MATRIX" in line.upper()
        or "CARTESIAN HESSIAN" in line.upper()
    ]
    for start in reversed(starts):
        matrix = _parse_block_matrix(lines[start + 1 :], size=size)
        if matrix is not None:
            return matrix
    return None


def _parse_orca_hess_matrix(lines: list[str], *, size: int) -> np.ndarray:
    for idx, line in enumerate(lines):
        if line.strip().lower() != "$hessian":
            continue
        matrix = _parse_block_matrix(lines[idx + 1 :], size=size)
        if matrix is not None:
            return 0.5 * (matrix + matrix.T)
    raise GeometryParseError("ORCA Hessian file contains no $hessian block")


def _parse_orca_hess_atoms(lines: list[str]) -> tuple[list[str], list[float], list[list[float]]]:
    for idx, line in enumerate(lines):
        if line.strip().lower() != "$atoms":
            continue
        if idx + 1 >= len(lines):
            break
        try:
            natoms = int(lines[idx + 1].strip())
        except ValueError as exc:
            raise GeometryParseError("Invalid ORCA $atoms count") from exc
        atoms: list[str] = []
        masses: list[float] = []
        coords: list[list[float]] = []
        for raw in lines[idx + 2 : idx + 2 + natoms]:
            parts = raw.split()
            if len(parts) < 5:
                raise GeometryParseError("Invalid ORCA $atoms row")
            atoms.append(normalize_atom_symbol(parts[0]))
            try:
                masses.append(float(parts[1]))
                coords.append([float(parts[2]), float(parts[3]), float(parts[4])])
            except ValueError as exc:
                raise GeometryParseError("Invalid numeric value in ORCA $atoms row") from exc
        return atoms, masses, coords
    raise GeometryParseError("ORCA Hessian file contains no $atoms block")


def _parse_orca_hess_frequencies(lines: list[str]) -> list[float]:
    for idx, line in enumerate(lines):
        if line.strip().lower() != "$vibrational_frequencies":
            continue
        if idx + 1 >= len(lines):
            break
        try:
            count = int(lines[idx + 1].strip())
        except ValueError as exc:
            raise GeometryParseError("Invalid ORCA $vibrational_frequencies count") from exc
        values: list[float] = []
        for raw in lines[idx + 2 : idx + 2 + count]:
            parts = raw.split()
            if len(parts) < 2:
                continue
            try:
                values.append(float(parts[1]))
            except ValueError as exc:
                raise GeometryParseError(
                    "Invalid numeric value in ORCA $vibrational_frequencies row"
                ) from exc
        return values
    return []


def _canonical_orca_vibrational_frequencies(
    frequencies,
    coordinates_angstrom,
) -> tuple[np.ndarray, dict[str, str]]:
    """Remove ORCA's leading translational/rotational frequency records."""

    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    raw = np.asarray(frequencies, dtype=float).reshape(-1)
    expected = expected_vibrational_mode_count(coordinates)
    full_count = 3 * coordinates.shape[0]
    if not np.all(np.isfinite(raw)):
        raise GeometryParseError("ORCA vibrational frequencies contain non-finite values")
    if raw.size == expected:
        vibrational = raw
    elif raw.size == full_count:
        vibrational = raw[-expected:] if expected else raw[:0]
    elif raw.size == 0:
        vibrational = raw
    else:
        raise GeometryParseError(
            "ORCA vibrational-frequency count is incomplete: "
            f"got {raw.size}, expected {expected} projected or {full_count} full Cartesian modes"
        )
    return vibrational, {
        "PROGRAM": "ORCA",
        "RAW_FREQUENCY_COUNT": str(raw.size),
        "EXPECTED_MODE_COUNT": str(expected),
        "LINEAR": str(is_linear_geometry(coordinates)).lower(),
    }


def _parse_block_matrix(lines: list[str], *, size: int) -> np.ndarray | None:
    matrix = np.zeros((size, size), dtype=float)
    filled: set[tuple[int, int]] = set()
    columns: list[int] = []
    for raw in lines:
        text = raw.strip()
        if not text or set(text) <= {"-"}:
            continue
        parts = text.split()
        int_tokens = [_int_or_none(token) for token in parts]
        if int_tokens and all(value is not None for value in int_tokens):
            columns = [int(value) for value in int_tokens if value is not None]
            continue
        row = _int_or_none(parts[0]) if parts else None
        if row is None:
            if filled and text.upper() == text and not _contains_float(text):
                break
            continue
        values = [_float_or_none(token) for token in parts[1:]]
        numeric = [value for value in values if value is not None]
        if not numeric:
            continue
        if columns and len(numeric) <= len(columns):
            selected_columns = columns[: len(numeric)]
        elif len(numeric) == size:
            selected_columns = list(range(size))
        else:
            continue
        if not 0 <= row < size:
            continue
        for column, value in zip(selected_columns, numeric):
            if 0 <= column < size:
                matrix[row, column] = float(value)
                filled.add((row, column))
        if len(filled) >= size * size:
            return matrix
    if not filled:
        return None
    if _looks_lower_triangular(filled, size):
        matrix = matrix + matrix.T - np.diag(np.diag(matrix))
    return matrix


def _looks_lower_triangular(filled: set[tuple[int, int]], size: int) -> bool:
    expected = {(row, col) for row in range(size) for col in range(row + 1)}
    return expected.issubset(filled) and not any(col > row for row, col in filled)


def _coordinate_block_count(lines: list[str]) -> int:
    return sum(
        1
        for line in lines
        if "CARTESIAN COORDINATES (ANGSTROEM)" in line.upper()
        or "CARTESIAN COORDINATES (A.U.)" in line.upper()
    )


def _atomic_numbers_from_geometry(geometry: MolecularGeometry) -> tuple[int, ...]:
    numbers: list[int] = []
    for atom in geometry.atoms:
        number = atomic_number(atom)
        if number is None or number <= 0:
            raise GeometryParseError(f"unknown element symbol in ORCA output: {atom}")
        numbers.append(int(number))
    return tuple(numbers)


def _float_or_none(token: str) -> float | None:
    try:
        return float(token.replace("D", "E").replace("d", "e"))
    except ValueError:
        return None


def _int_or_none(token: str) -> int | None:
    try:
        return int(token)
    except ValueError:
        return None


def _contains_float(text: str) -> bool:
    return any(_float_or_none(token) is not None for token in text.split())
