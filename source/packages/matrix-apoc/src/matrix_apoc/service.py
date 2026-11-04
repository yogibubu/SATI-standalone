"""Backend adapters and persistence for the APOC v1 contract."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from matrix_chem import BOHR_TO_ANGSTROM, MolecularGeometry, read_enriched_xyz
from matrix_chem.topology.elements import atomic_number, atomic_symbol
from matrix_core import write_sectioned_lines
from matrix_chem import kabsch_align
from matrix_qm import (
    QMPopulationObservables,
    population_observables_from_dict,
    read_qm_population_section,
    write_qm_population_section,
)


APOC_ANALYSIS_SCHEMA = "matrix.apoc.analysis.v1"


@dataclass(frozen=True)
class ApocAnalysis:
    source: Path
    backend: str
    geometry: MolecularGeometry
    observables: QMPopulationObservables

    def __post_init__(self) -> None:
        if self.geometry.natoms != self.observables.natoms:
            raise ValueError("APOC geometry and population atom counts differ")

    def to_dict(self) -> dict[str, object]:
        payload = self.observables.to_dict(source=str(self.source))
        payload.update(
            {
                "backend": self.backend,
                "atoms": list(self.geometry.atoms),
                "coordinates_angstrom": np.asarray(
                    self.geometry.coordinates_angstrom, dtype=float
                ).tolist(),
            }
        )
        return payload

    def report_lines(self) -> list[str]:
        cm5 = self.observables.cm5_charges
        mayer = self.observables.mayer_bond_orders
        strongest = sorted(
            (
                (float(mayer[i, j]), i + 1, j + 1)
                for i in range(self.observables.natoms)
                for j in range(i + 1, self.observables.natoms)
            ),
            reverse=True,
        )[: min(12, self.observables.natoms * (self.observables.natoms - 1) // 2)]
        lines = [
            "APOC electronic analysis",
            "========================",
            f"Source: {self.source}",
            f"Backend: {self.backend}",
            f"Atoms: {self.observables.natoms}",
            f"Electrons: {self.observables.electron_count:.8f}",
            f"Molecular charge: {self.observables.charge:.8f}",
            "Charge model: CM5 (from Hirshfeld)",
            "Bond-order model: Mayer",
            "",
            "Atomic charges",
            "--------------",
            "atom symbol hirshfeld cm5",
        ]
        lines.extend(
            f"{index + 1:4d} {self.geometry.atoms[index]:>3s} "
            f"{self.observables.hirshfeld_charges[index]: .8f} {cm5[index]: .8f}"
            for index in range(self.observables.natoms)
        )
        lines.extend(["", "Largest Mayer bond orders", "-------------------------"])
        lines.extend(
            f"{left:4d} {right:4d} {value: .8f}" for value, left, right in strongest
        )
        return lines


def analyze_gaussian(path: Path | str) -> ApocAnalysis:
    from matrix_gaussian import read_gaussian_log_geometry, read_gaussian_population

    source = Path(path)
    geometry = read_gaussian_log_geometry(source)
    data = read_gaussian_population(source)
    natoms = geometry.natoms
    required = set(range(1, natoms + 1))
    if set(data.hirshfeld_charges) != required or set(data.cm5_charges) != required:
        raise ValueError(
            "Gaussian output lacks a complete Hirshfeld/CM5 table; request Pop=Hirshfeld"
        )
    mayer = np.zeros((natoms, natoms), dtype=float)
    for (left, right), value in data.mayer_bond_orders.items():
        if 1 <= left <= natoms and 1 <= right <= natoms and left != right:
            mayer[left - 1, right - 1] = mayer[right - 1, left - 1] = float(value)
    if len(data.mayer_bond_orders) < natoms * (natoms - 1) // 2:
        raise ValueError(
            "Gaussian output lacks the complete Mayer matrix; request IOp(6/80=1)"
        )
    numbers = np.asarray([_atomic_number(atom) for atom in geometry.atoms], dtype=int)
    molecular_charge = float(round(sum(data.cm5_charges.values())))
    observables = QMPopulationObservables(
        hirshfeld_charges=np.asarray(
            [data.hirshfeld_charges[index] for index in range(1, natoms + 1)]
        ),
        cm5_charges=np.asarray(
            [data.cm5_charges[index] for index in range(1, natoms + 1)]
        ),
        mayer_bond_orders=mayer,
        electron_count=float(np.sum(numbers) - molecular_charge),
        charge=molecular_charge,
    )
    return ApocAnalysis(source, "Gaussian/GDV", geometry, observables)


def analyze_gaussian_fchk(
    path: Path | str,
    *,
    grid_level: int = 4,
    charge: int | None = None,
) -> ApocAnalysis:
    """Compute APOC observables directly from a Gaussian FCHK wavefunction."""

    from .population import population_observables_from_electronic_state
    from .state import electronic_state_from_gaussian_fchk

    source = Path(path)
    state = electronic_state_from_gaussian_fchk(source)
    if charge is not None and int(charge) != int(state.charge):
        raise ValueError(
            f"requested charge {charge} differs from FCHK charge {state.charge}"
        )
    observables = population_observables_from_electronic_state(
        state,
        grid_level=grid_level,
    )
    geometry = MolecularGeometry(
        atoms=tuple(atomic_symbol(int(number)) for number in state.atomic_numbers),
        coordinates_angstrom=np.asarray(state.coordinates_bohr, dtype=float)
        * BOHR_TO_ANGSTROM,
        comment=f"APOC from {source.name}",
        source_format="gaussian_fchk",
        source_path=source,
        charge=int(state.charge),
        multiplicity=int(state.multiplicity),
    )
    return ApocAnalysis(source, "Gaussian FCHK/APOC", geometry, observables)


def analyze_molden(
    path: Path | str,
    *,
    grid_level: int = 4,
    charge: int | None = None,
) -> ApocAnalysis:
    from matrix_pyscf import population_observables_from_molden

    try:
        from pyscf.tools import molden
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("APOC Molden analysis requires PySCF") from exc
    source = Path(path)
    mol, _energies, _coefficients, _occupations, _labels, _spins = molden.load(str(source))
    observables = population_observables_from_molden(
        source,
        grid_level=grid_level,
        charge=charge,
    )
    geometry = MolecularGeometry(
        atoms=tuple(mol.atom_pure_symbol(index) for index in range(mol.natm)),
        coordinates_angstrom=np.asarray(mol.atom_coords(unit="Angstrom"), dtype=float),
        comment=f"APOC from {source.name}",
        source_format="molden",
        source_path=source,
        charge=int(round(observables.charge)),
        multiplicity=int(mol.spin) + 1,
    )
    return ApocAnalysis(source, "Molden/PySCF", geometry, observables)


def analyze_orca(
    path: Path | str,
    *,
    molden_output: Path | str | None = None,
    converter_executable: str = "orca_2mkl",
    timeout: float = 120.0,
    grid_level: int = 4,
    charge: int | None = None,
) -> ApocAnalysis:
    """Convert an ORCA GBW to Molden and apply the common APOC definitions."""

    from matrix_orca import convert_orca_gbw_to_molden

    source = Path(path)
    gbw = source if source.suffix.lower() == ".gbw" else source.with_suffix(".gbw")
    if not gbw.is_file():
        raise FileNotFoundError(
            f"APOC ORCA analysis requires the matching GBW file: {gbw}"
        )
    converted = convert_orca_gbw_to_molden(
        gbw,
        output=molden_output,
        executable=converter_executable,
        timeout=timeout,
    )
    analysis = analyze_molden(
        converted.molden_path,
        grid_level=grid_level,
        charge=charge,
    )
    return ApocAnalysis(source, "ORCA/Molden/PySCF", analysis.geometry, analysis.observables)


def analyze_pyscf_output(path: Path | str) -> ApocAnalysis:
    from matrix_pyscf import (
        population_observables_from_pyscf_output,
        read_pyscf_output_geometry,
    )

    source = Path(path)
    return ApocAnalysis(
        source,
        "PySCF",
        read_pyscf_output_geometry(source),
        population_observables_from_pyscf_output(source),
    )


def analyze_source(
    path: Path | str,
    *,
    source_format: str = "auto",
    grid_level: int = 4,
    charge: int | None = None,
    orca_converter: str = "orca_2mkl",
    orca_timeout: float = 120.0,
) -> ApocAnalysis:
    """Dispatch one QM source into the fixed APOC analysis contract."""

    source = Path(path)
    selected = str(source_format).strip().lower().replace("_", "-")
    if selected == "auto":
        if source.suffix.lower() in {".fchk", ".fch"}:
            selected = "gaussian-fchk"
        elif source.suffix.lower() == ".gbw":
            selected = "orca"
        else:
            head = source.read_text(encoding="utf-8", errors="replace")[:200_000]
            if "MATRIX_PYSCF_RESULT " in head:
                selected = "pyscf-output"
            elif "[Molden Format]" in head or source.name.lower().endswith(
                (".molden", ".molden.input")
            ):
                selected = "molden"
            elif "O   R   C   A" in head or "Program Version" in head and "ORCA" in head:
                selected = "orca"
            elif "Gaussian" in head or "Entering Gaussian System" in head:
                selected = "gaussian"
            else:
                raise ValueError(
                    "cannot infer APOC source format; choose gaussian, gaussian-fchk, "
                    "orca, molden or pyscf-output"
                )
    if selected in {"gaussian", "gdv", "g16"}:
        return analyze_gaussian(source)
    if selected in {"gaussian-fchk", "fchk", "fch"}:
        return analyze_gaussian_fchk(
            source,
            grid_level=grid_level,
            charge=charge,
        )
    if selected == "molden":
        return analyze_molden(source, grid_level=grid_level, charge=charge)
    if selected in {"pyscf", "pyscf-output"}:
        return analyze_pyscf_output(source)
    if selected == "orca":
        return analyze_orca(
            source,
            converter_executable=orca_converter,
            timeout=orca_timeout,
            grid_level=grid_level,
            charge=charge,
        )
    raise ValueError(f"unsupported APOC source format: {source_format}")


def save_apoc_analysis(path: Path | str, analysis: ApocAnalysis) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(analysis.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def load_apoc_analysis(path: Path | str) -> ApocAnalysis:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != APOC_ANALYSIS_SCHEMA:
        raise ValueError("unsupported APOC analysis JSON schema")
    atoms = tuple(str(atom) for atom in payload.get("atoms", []))
    coordinates = np.asarray(payload.get("coordinates_angstrom", []), dtype=float)
    geometry = MolecularGeometry(
        atoms=atoms,
        coordinates_angstrom=coordinates,
        comment=f"APOC from {source.name}",
        source_format="apoc_json",
        source_path=source,
        charge=int(round(float(payload.get("molecular_charge", 0.0)))),
        multiplicity=1,
    )
    return ApocAnalysis(
        Path(str(payload.get("source", source))),
        str(payload.get("backend", "unknown")),
        geometry,
        population_observables_from_dict(payload),
    )


def attach_apoc_analysis(path: Path | str, analysis: ApocAnalysis) -> Path:
    target = Path(path)
    if not target.exists():
        write_sectioned_lines(target, analysis.geometry.xyz_lines())
    else:
        geometry = read_enriched_xyz(target)
        if geometry.atoms != analysis.geometry.atoms:
            raise ValueError("APOC and target xyzin atom sequences differ")
        aligned = kabsch_align(
            analysis.geometry.coordinates_angstrom,
            geometry.coordinates_angstrom,
        )
        if not np.allclose(
            geometry.coordinates_angstrom,
            aligned,
            atol=2.0e-6,
            rtol=0.0,
        ):
            raise ValueError("APOC and target xyzin geometries differ")
    return write_qm_population_section(
        target,
        analysis.observables,
        source=f"APOC {analysis.backend} {analysis.source}",
    )


def analysis_from_xyzin(path: Path | str) -> tuple[QMPopulationObservables, str]:
    return read_qm_population_section(path)


def _atomic_number(symbol: str) -> int:
    value = atomic_number(symbol)
    if value is None:
        raise ValueError(f"unknown atom symbol: {symbol}")
    return int(value)
