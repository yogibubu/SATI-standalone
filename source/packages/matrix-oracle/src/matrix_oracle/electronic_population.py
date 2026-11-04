"""ORACLE-owned electronic population and H-bond calibration workflows.

ORACLE defines the calculation, caching, fragmentation and synthesis
contracts.  Electronic-structure packages are replaceable executors beneath
that contract; importing ORACLE never requires a particular executor.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from matrix_chem.topology.elements import atomic_number
from matrix_core import (
    ElectronicObservableCompleteness,
    ElectronicObservableKind,
    ElectronicObservableOwner,
    ElectronicObservableProvenance,
    atomic_json_write,
)
from matrix_qm import QMPopulationObservables, population_observables_from_dict

from .electrostatics import standard_cm5_mayer_request
from .hbond_charge_response import HydrogenBondResponseCalibration
from .hbond_training import (
    HydrogenBondTrainingGeometry,
    fit_training_geometry_cm5,
    standard_hbond_training_complexes,
    build_hbond_training_geometry,
)


ORACLE_POPULATION_CALCULATION_SCHEMA = "matrix.oracle.population_calculation.v1"
ORACLE_POPULATION_CACHE_VERSION = "oracle-population-cache-v2"
ORACLE_HBOND_TRAINING_SCHEMA = "matrix.oracle.hbond_training.v1"
ORACLE_ELECTRONEGATIVITY_PROVENANCE = ElectronicObservableProvenance(
    owner=ElectronicObservableOwner.ORACLE,
    observable=ElectronicObservableKind.ELECTRONEGATIVITY_ESTIMATE,
    completeness=ElectronicObservableCompleteness.FALLBACK_COMPLETE,
    source="ORACLE electronegativity estimate",
)
ORACLE_PAULING_PROVENANCE = ElectronicObservableProvenance(
    owner=ElectronicObservableOwner.ORACLE,
    observable=ElectronicObservableKind.PAULING_ESTIMATE,
    completeness=ElectronicObservableCompleteness.FALLBACK_COMPLETE,
    source="ORACLE Pauling estimate",
)


class OraclePopulationBackend(Protocol):
    """Executor interface controlled by the ORACLE population workflow."""

    name: str

    def calculate(
        self,
        atoms: Sequence[str],
        coordinates_angstrom: np.ndarray,
        *,
        charge: int,
        multiplicity: int,
        request: dict[str, object],
    ) -> tuple[QMPopulationObservables, float, bool]: ...


@dataclass(frozen=True)
class OraclePopulationCalculation:
    """One complete paired CM5/Mayer calculation with provenance."""

    identifier: str
    atoms: tuple[str, ...]
    coordinates_angstrom: np.ndarray
    charge: int
    multiplicity: int
    request: dict[str, object]
    observables: QMPopulationObservables
    energy_hartree: float
    backend: str
    cache_key: str
    converged: bool
    schema: str = ORACLE_POPULATION_CALCULATION_SCHEMA


@dataclass(frozen=True)
class OracleHydrogenBondTrainingResult:
    """Monomer/dimer calculations and fitted oriented contact rules."""

    monomers: tuple[OraclePopulationCalculation, ...]
    dimers: tuple[OraclePopulationCalculation, ...]
    geometries: tuple[HydrogenBondTrainingGeometry, ...]
    calibrations: tuple[HydrogenBondResponseCalibration, ...]
    population_level: str
    intramolecular_geometry_contract: str = "FIXED_RIGID_MONOMERS"
    schema: str = ORACLE_HBOND_TRAINING_SCHEMA


class PySCFOraclePopulationBackend:
    """Local PySCF executor for the ORACLE L0/L1 population contract."""

    name = "PySCF"

    def __init__(
        self,
        *,
        scf_grid_level: int = 4,
        population_grid_level: int = 4,
        threads: int | None = None,
    ) -> None:
        self.scf_grid_level = int(scf_grid_level)
        self.population_grid_level = int(population_grid_level)
        self.threads = None if threads is None else int(threads)

    def calculate(
        self,
        atoms: Sequence[str],
        coordinates_angstrom: np.ndarray,
        *,
        charge: int,
        multiplicity: int,
        request: dict[str, object],
    ) -> tuple[QMPopulationObservables, float, bool]:
        try:
            from pyscf import dft, gto, lib, mp, scf
            from matrix_pyscf import population_observables_from_pyscf
        except ImportError as exc:  # pragma: no cover - optional executor
            raise RuntimeError(
                "the local ORACLE population executor requires matrix-pyscf[runtime]"
            ) from exc
        if self.threads is not None:
            if self.threads < 1:
                raise ValueError("the PySCF thread count must be positive")
            lib.num_threads(self.threads)
        xyz = np.asarray(coordinates_angstrom, dtype=float)
        labels = tuple(str(atom) for atom in atoms)
        if xyz.shape != (len(labels), 3):
            raise ValueError("population coordinates must have shape (natoms, 3)")
        numbers = tuple(int(atomic_number(atom) or 0) for atom in labels)
        if not numbers or min(numbers) <= 0:
            raise ValueError("population calculation contains an unknown element")
        basis = str(request["basis"])
        molecule_kwargs: dict[str, object] = {
            "atom": [
                (atom, tuple(float(value) for value in coordinate))
                for atom, coordinate in zip(labels, xyz, strict=True)
            ],
            "basis": basis,
            "charge": int(charge),
            "spin": int(multiplicity) - 1,
            "unit": "Angstrom",
            "verbose": 0,
        }
        if str(request.get("pseudopotential", "")).startswith("def2-ECP"):
            # PySCF interprets a scalar ECP name as a request for every
            # element and consequently warns or fails for light atoms that
            # have no def2 pseudopotential.  Restrict the mapping to the
            # heavy centers covered by the ORACLE Rb--Rn contract.
            molecule_kwargs["ecp"] = _def2_ecp_mapping(labels, numbers, basis)
        mol = gto.M(**molecule_kwargs)
        method = str(request["method"]).upper()
        if method == "PBE0":
            mean_field = dft.RKS(mol) if multiplicity == 1 else dft.UKS(mol)
            mean_field.xc = "PBE0"
            mean_field.grids.level = self.scf_grid_level
            mean_field.conv_tol = 1.0e-10
            mean_field.max_cycle = 100
            energy = float(mean_field.kernel())
            density = mean_field.make_rdm1()
            converged = bool(mean_field.converged)
        elif method == "MP2":
            reference = scf.RHF(mol) if multiplicity == 1 else scf.UHF(mol)
            reference.conv_tol = 1.0e-10
            reference.max_cycle = 100
            reference.kernel()
            correlated = mp.MP2(reference)
            correlation_energy, _amplitudes = correlated.kernel()
            energy = float(reference.e_tot + correlation_energy)
            density = correlated.make_rdm1(ao_repr=True)
            converged = bool(reference.converged and correlated.converged)
        else:
            raise ValueError(f"unsupported ORACLE population method for PySCF: {method}")
        if not converged:
            raise RuntimeError("the ORACLE electronic population calculation did not converge")
        observables = population_observables_from_pyscf(
            mol,
            density,
            grid_level=self.population_grid_level,
        )
        return observables, energy, converged


class OracleElectronicPopulationWorkflow:
    """Execute and cache the electronic work selected by ORACLE."""

    def __init__(
        self,
        backend: OraclePopulationBackend,
        *,
        cache_directory: Path | str | None = None,
    ) -> None:
        self.backend = backend
        self.cache_directory = (
            None if cache_directory is None else Path(cache_directory)
        )
        self._memory_cache: dict[str, dict[str, object]] = {}

    def calculate(
        self,
        identifier: str,
        atoms: Sequence[str],
        coordinates_angstrom: np.ndarray,
        *,
        charge: int = 0,
        multiplicity: int = 1,
        accuracy_level: str = "L0",
        basis: str | None = None,
    ) -> OraclePopulationCalculation:
        labels = tuple(str(atom) for atom in atoms)
        xyz = np.asarray(coordinates_angstrom, dtype=float)
        numbers = tuple(int(atomic_number(atom) or 0) for atom in labels)
        request = standard_cm5_mayer_request(
            accuracy_level=str(accuracy_level).upper(),
            atomic_numbers=numbers,
            basis=basis,
        )
        key = _population_cache_key(
            labels,
            xyz,
            charge=int(charge),
            multiplicity=int(multiplicity),
            request=request,
            backend=self.backend.name,
        )
        cached = self._read_cache(key)
        if cached is not None:
            return OraclePopulationCalculation(
                identifier=str(identifier),
                atoms=labels,
                coordinates_angstrom=xyz.copy(),
                charge=int(charge),
                multiplicity=int(multiplicity),
                request=request,
                observables=population_observables_from_dict(cached["observables"]),
                energy_hartree=float(cached["energy_hartree"]),
                backend=str(cached["backend"]),
                cache_key=key,
                converged=bool(cached["converged"]),
            )
        observables, energy, converged = self.backend.calculate(
            labels,
            xyz,
            charge=int(charge),
            multiplicity=int(multiplicity),
            request=request,
        )
        calculation = OraclePopulationCalculation(
            identifier=str(identifier),
            atoms=labels,
            coordinates_angstrom=xyz.copy(),
            charge=int(charge),
            multiplicity=int(multiplicity),
            request=request,
            observables=observables,
            energy_hartree=float(energy),
            backend=self.backend.name,
            cache_key=key,
            converged=bool(converged),
        )
        self._write_cache(calculation)
        return calculation
    def _cache_path(self, key: str) -> Path | None:
        if self.cache_directory is None:
            return None
        return self.cache_directory / f"{key}.json"

    def _read_cache(self, key: str) -> dict[str, object] | None:
        cached = self._memory_cache.get(key)
        if cached is not None:
            return cached
        path = self._cache_path(key)
        if path is None or not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("schema") != ORACLE_POPULATION_CALCULATION_SCHEMA
            or payload.get("cache_version") != ORACLE_POPULATION_CACHE_VERSION
            or payload.get("cache_key") != key
        ):
            raise ValueError(f"invalid ORACLE population cache record: {path}")
        self._memory_cache[key] = payload
        return payload

    def _write_cache(self, calculation: OraclePopulationCalculation) -> None:
        path = self._cache_path(calculation.cache_key)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": calculation.schema,
            "cache_version": ORACLE_POPULATION_CACHE_VERSION,
            "identifier": calculation.identifier,
            "cache_key": calculation.cache_key,
            "backend": calculation.backend,
            "energy_hartree": calculation.energy_hartree,
            "converged": calculation.converged,
            "request": calculation.request,
            "observables": calculation.observables.to_dict(
                source=f"ORACLE/{calculation.backend}"
            ),
        }
        atomic_json_write(path, payload)
        self._memory_cache[calculation.cache_key] = payload


def create_portable_l0_population_workflow(
    *,
    cache_directory: Path | str | None = None,
    threads: int | None = None,
) -> OracleElectronicPopulationWorkflow:
    """Create ORACLE's architecture-neutral local L0 population executor.

    PySCF is used as a CPU backend on every supported MATRIX architecture.
    Hardware identity does not change the scientific request or the 60-heavy-
    atom whole-molecule policy; ``threads`` is only an execution resource.
    """

    return OracleElectronicPopulationWorkflow(
        PySCFOraclePopulationBackend(threads=threads),
        cache_directory=cache_directory,
    )


def run_standard_hbond_training(
    workflow: OracleElectronicPopulationWorkflow,
    *,
    accuracy_level: str = "L0",
    basis: str | None = None,
) -> OracleHydrogenBondTrainingResult:
    """Calculate five monomers and twenty rigid reference dimers in ORACLE."""

    from .hbond_training import standard_hbond_training_molecules

    catalogue = standard_hbond_training_molecules()
    monomer_by_name: dict[str, OraclePopulationCalculation] = {}
    for name, molecule in catalogue.items():
        monomer_by_name[name] = workflow.calculate(
            f"monomer__{name}",
            molecule.atoms,
            molecule.coordinates_angstrom,
            accuracy_level=accuracy_level,
            basis=basis,
        )
    dimers: list[OraclePopulationCalculation] = []
    geometries: list[HydrogenBondTrainingGeometry] = []
    calibrations: list[HydrogenBondResponseCalibration] = []
    for specification in standard_hbond_training_complexes():
        geometry = build_hbond_training_geometry(specification)
        from .hbond_training import audit_hbond_training_geometry

        audit = audit_hbond_training_geometry(geometry)
        if not audit.passed:
            raise ValueError(
                f"H-bond training geometry failed chemical preflight: {audit}"
            )
        calculation = workflow.calculate(
            f"dimer__{specification.identifier}",
            geometry.atoms,
            geometry.coordinates_angstrom,
            accuracy_level=accuracy_level,
            basis=basis,
        )
        donor = monomer_by_name[specification.donor_molecule]
        acceptor = monomer_by_name[specification.acceptor_molecule]
        calibration = fit_training_geometry_cm5(
            geometry,
            calculation.observables.cm5_charges,
            donor.observables.cm5_charges,
            acceptor.observables.cm5_charges,
            reference_mayer_bond_order=float(
                calculation.observables.mayer_bond_orders[
                    geometry.hydrogen, geometry.acceptor
                ]
            ),
            population_level=(
                f"{calculation.request['accuracy_level']}:"
                f"{calculation.request['method']}/{calculation.request['basis']}"
            ),
        )
        dimers.append(calculation)
        geometries.append(geometry)
        calibrations.append(calibration)
    first = next(iter(monomer_by_name.values()))
    return OracleHydrogenBondTrainingResult(
        monomers=tuple(monomer_by_name.values()),
        dimers=tuple(dimers),
        geometries=tuple(geometries),
        calibrations=tuple(calibrations),
        population_level=(
            f"{first.request['accuracy_level']}:"
            f"{first.request['method']}/{first.request['basis']}"
        ),
    )


def write_hbond_training_result(
    result: OracleHydrogenBondTrainingResult,
    path: Path | str,
) -> Path:
    """Write the compact fitted library; raw populations stay in the cache."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": result.schema,
        "population_level": result.population_level,
        "intramolecular_geometry_contract": result.intramolecular_geometry_contract,
        "monomers": [
            {
                "identifier": item.identifier,
                "cache_key": item.cache_key,
                "energy_hartree": item.energy_hartree,
            }
            for item in result.monomers
        ],
        "dimers": [
            {
                "identifier": calculation.identifier,
                "cache_key": calculation.cache_key,
                "energy_hartree": calculation.energy_hartree,
                "hydrogen_acceptor_distance_angstrom": geometry.hydrogen_acceptor_distance_angstrom,
                "dha_angle_degrees": geometry.dha_angle_degrees,
            }
            for calculation, geometry in zip(
                result.dimers, result.geometries, strict=True
            )
        ],
        "calibrations": [
            {
                **calibration.to_record(),
                "donor_molecule": geometry.specification.donor_molecule,
                "acceptor_molecule": geometry.specification.acceptor_molecule,
            }
            for calibration, geometry in zip(
                result.calibrations, result.geometries, strict=True
            )
        ],
    }
    atomic_json_write(target, payload)
    return target


def _population_cache_key(
    atoms: tuple[str, ...],
    coordinates_angstrom: np.ndarray,
    *,
    charge: int,
    multiplicity: int,
    request: dict[str, object],
    backend: str,
) -> str:
    payload = {
        "cache_version": ORACLE_POPULATION_CACHE_VERSION,
        "atoms": atoms,
        "coordinates_angstrom": np.asarray(
            coordinates_angstrom, dtype=float
        ).round(12).tolist(),
        "charge": int(charge),
        "multiplicity": int(multiplicity),
        "request": request,
        "backend": str(backend),
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _def2_ecp_mapping(
    labels: Sequence[str],
    atomic_numbers: Sequence[int],
    basis: str,
) -> dict[str, str]:
    """Return the element-specific PySCF ECP map for the Rb--Rn contract."""

    if len(labels) != len(atomic_numbers):
        raise ValueError("ECP labels and atomic numbers must have the same length")
    return {
        str(label): str(basis)
        for label, number in zip(labels, atomic_numbers, strict=True)
        if 37 <= int(number) <= 86
    }


__all__ = [
    "ORACLE_HBOND_TRAINING_SCHEMA",
    "ORACLE_POPULATION_CALCULATION_SCHEMA",
    "ORACLE_POPULATION_CACHE_VERSION",
    "OracleElectronicPopulationWorkflow",
    "OracleHydrogenBondTrainingResult",
    "OraclePopulationBackend",
    "OraclePopulationCalculation",
    "PySCFOraclePopulationBackend",
    "create_portable_l0_population_workflow",
    "run_standard_hbond_training",
    "write_hbond_training_result",
]
