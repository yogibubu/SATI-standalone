"""Resident MATRIX-GFN2 library interface.

The surface follows the stateful TBLite contract: construct once, update
positions in bohr, and execute repeated single points while the calculator
retains its electronic restart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence
from pathlib import Path

import numpy as np

from .calculator import ANGSTROM_TO_BOHR, GFN2Calculator, GFN2Result
from .parameters import ELEMENT_SYMBOLS


LIBRARY_API_VERSION = (1, 2)


@dataclass
class Result:
    """Library result with property access compatible with TBLite clients."""

    _result: GFN2Result

    def get(self, name: str) -> Any:
        key = str(name).strip().casefold().replace("_", "-")
        values = {
            "energy": self._result.energy_hartree,
            "gradient": self._result.gradient_hartree_per_bohr,
            "charges": self._result.charges_electron,
            "bond-orders": self._result.bond_orders,
            "iterations": self._result.iterations,
            "orbital-state": self._result.orbital_state,
        }
        try:
            value = values[key]
        except KeyError as error:
            raise KeyError(f"unsupported MATRIX-GFN2 result property: {name}") from error
        return value.copy() if isinstance(value, np.ndarray) else value

    def write_wfn_xtb(self, path: str | Path = "wfn.xtb"):
        from .wfn_xtb import write_wfn_xtb

        if self._result.orbital_state is None:
            raise RuntimeError(
                "wfn.xtb export requires calculator option retain-orbitals=true"
            )
        return write_wfn_xtb(path, self._result.orbital_state)


class Calculator:
    """Persistent GFN2-xTB session accepting atomic numbers and bohr."""

    def __init__(
        self,
        method: str,
        numbers: Sequence[int],
        positions: np.ndarray,
        *,
        charge: int = 0,
        uhf: int = 0,
        density_solver: str = "auto",
        temperature_kelvin: float = 300.0,
        device: str | None = None,
    ) -> None:
        normalized = str(method).strip().casefold().replace("_", "-")
        if normalized not in {"gfn2-xtb", "gfn2", "matrix-gfn2"}:
            raise ValueError("MATRIX-GFN2 library supports only GFN2-xTB")
        atomic_numbers = np.asarray(numbers, dtype=np.int64)
        if atomic_numbers.ndim != 1 or not len(atomic_numbers):
            raise ValueError("atomic numbers must be a nonempty vector")
        if np.any(atomic_numbers < 1) or np.any(atomic_numbers > len(ELEMENT_SYMBOLS)):
            raise ValueError("atomic number is outside the MATRIX-GFN2 table")
        self._numbers = atomic_numbers.copy()
        self._atoms = tuple(ELEMENT_SYMBOLS[number - 1] for number in atomic_numbers)
        self._calculator = GFN2Calculator(
            self._atoms,
            charge=int(charge),
            multiplicity=int(uhf) + 1,
            scf_convergence_mode=("tblite" if int(uhf) == 0 else "density"),
            density_solver=density_solver,
            temperature_kelvin=float(temperature_kelvin),
            device=device,
        )
        self._positions = np.empty((len(self._atoms), 3), dtype=float)
        self._verbosity = 0
        self._accuracy = 1.0
        self._scf_convergence = "tblite" if int(uhf) == 0 else "density"
        self._scf_energy_tolerance_override: float | None = None
        self._scf_density_tolerance_override: float | None = None
        self._scf_mixer_tolerance_override: float | None = None
        self._scf_max_iterations_override: int | None = None
        self._compute_bond_orders = True
        self._retain_orbitals = False
        self.update(positions=positions)

    @property
    def api_version(self) -> tuple[int, int]:
        return LIBRARY_API_VERSION

    def set(self, name: str, value: Any) -> None:
        key = str(name).strip().casefold().replace("_", "-")
        if key == "verbosity":
            self._verbosity = int(value)
            return
        if key == "accuracy":
            accuracy = float(value)
            if not np.isfinite(accuracy) or accuracy <= 0.0:
                raise ValueError("accuracy must be positive and finite")
            self._accuracy = accuracy
            self._apply_scf_settings()
            return
        if key == "scf-convergence":
            mode = str(value).strip().lower()
            if mode not in {"density", "tblite"}:
                raise ValueError("scf-convergence must be 'density' or 'tblite'")
            if mode == "tblite" and self._calculator.multiplicity != 1:
                raise ValueError("TBLite-like mixer convergence requires a closed shell")
            if mode == "density" and self._calculator.multiplicity == 1:
                raise ValueError("closed-shell GFN2 always uses the TBLite SCF criterion")
            self._scf_convergence = mode
            self._apply_scf_settings()
            return
        if key in {"scf-energy-tolerance", "scf-density-tolerance", "scf-mixer-tolerance"}:
            tolerance = float(value)
            if not np.isfinite(tolerance) or tolerance <= 0.0:
                raise ValueError(f"{key} must be positive and finite")
            attribute = {
                "scf-energy-tolerance": "_scf_energy_tolerance_override",
                "scf-density-tolerance": "_scf_density_tolerance_override",
                "scf-mixer-tolerance": "_scf_mixer_tolerance_override",
            }[key]
            setattr(self, attribute, tolerance)
            self._apply_scf_settings()
            return
        if key in {"max-iter", "scf-max-iterations"}:
            iterations = int(value)
            if iterations < 1 or iterations != value:
                raise ValueError("maximum SCF iterations must be a positive integer")
            self._scf_max_iterations_override = iterations
            self._apply_scf_settings()
            return
        if key == "bond-orders":
            self._compute_bond_orders = bool(value)
            return
        if key == "retain-orbitals":
            self._retain_orbitals = bool(value)
            return
        raise KeyError(f"unsupported MATRIX-GFN2 calculator option: {name}")

    def _apply_scf_settings(self) -> None:
        tblite_mode = self._scf_convergence == "tblite"
        self._calculator.scf_convergence_mode = self._scf_convergence
        self._calculator.scf_energy_tolerance_hartree = (
            self._scf_energy_tolerance_override
            if self._scf_energy_tolerance_override is not None
            else (1.0e-6 if tblite_mode else 1.0e-8) * self._accuracy
        )
        self._calculator.scf_wavefunction_tolerance_electron = (
            self._scf_density_tolerance_override
            if self._scf_density_tolerance_override is not None
            else 1.0e-6 * self._accuracy
        )
        self._calculator.scf_mixer_tolerance = (
            self._scf_mixer_tolerance_override
            if self._scf_mixer_tolerance_override is not None
            else 2.0e-5 * self._accuracy
        )
        self._calculator.maximum_scf_iterations = (
            self._scf_max_iterations_override
            if self._scf_max_iterations_override is not None
            else 250 if tblite_mode else 500
        )
        self._calculator._last_result_key = None

    def update(self, *, positions: np.ndarray) -> None:
        coordinates = np.asarray(positions, dtype=float)
        if coordinates.shape != self._positions.shape or not np.all(np.isfinite(coordinates)):
            raise ValueError("positions must be a finite natoms x 3 array in bohr")
        np.copyto(self._positions, coordinates)

    def singlepoint(self, result: Result | None = None) -> Result:
        if result is not None and not isinstance(result, Result):
            raise TypeError("restart result must originate from MATRIX-GFN2")
        if len(self._numbers) < 21:
            # This stateful library surface already stores bohr coordinates.
            # For the compiled small-system regime enter the resident dense
            # evaluator directly, avoiding a bohr->angstrom->bohr round trip
            # and the general sparse/symmetry router.  Calculators created by
            # this interface have neither embedding nor symmetry contracts,
            # so the selected physical path is exactly the same dense GFN2
            # path that the router would choose.
            evaluated = self._calculator._single_point(
                self._positions,
                compute_cm5=False,
                compute_bond_orders=self._compute_bond_orders,
                retain_orbitals=self._retain_orbitals,
                collect_telemetry=False,
            )
        else:
            evaluated = self._calculator.evaluate(
                self._positions / ANGSTROM_TO_BOHR,
                compute_cm5=False,
                compute_bond_orders=self._compute_bond_orders,
                retain_orbitals=self._retain_orbitals,
                collect_telemetry=False,
            )
        if result is not None:
            result._result = evaluated
            return result
        return Result(evaluated)


__all__ = ["Calculator", "LIBRARY_API_VERSION", "Result"]
