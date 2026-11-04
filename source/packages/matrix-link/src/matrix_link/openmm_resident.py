"""Resident OpenMM calculator used by LINK Cartesian preoptimization."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Sequence

import numpy as np


KJ_MOL_PER_HARTREE = 2625.4996394799
BOHR_IN_NANOMETER = 0.0529177210903


def openmm_ml_available() -> bool:
    """Return whether OpenMM, OpenMM-ML, and AIMNet2 are importable."""

    return all(find_spec(name) is not None for name in ("openmm", "openmmml", "aimnet"))


def openmm_available() -> bool:
    """Return whether the resident OpenMM runtime is importable."""

    return find_spec("openmm") is not None


@dataclass(frozen=True)
class OpenMMEvaluation:
    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray
    potential: str


class OpenMMResidentSession:
    """Own one OpenMM ``Context`` for all LINK geometry evaluations.

    SWITCH supplies atom order and, when available, the trusted molecular
    graph.  OpenMM-ML supplies the potential; neither package is allowed to
    reinterpret a SMILES string or reorder atoms.
    """

    def __init__(
        self,
        atoms: Sequence[str],
        coordinates_angstrom: np.ndarray,
        *,
        potential: str = "aimnet2",
        charge: int = 0,
        multiplicity: int = 1,
        bonds: Sequence[tuple[int, int]] = (),
        system_xml: Path | str | None = None,
        platform: str | None = None,
    ) -> None:
        try:
            import openmm
            from openmm import app, unit
        except ImportError as exc:
            raise RuntimeError("OpenMM preoptimization requires the openmm package") from exc

        self._openmm = openmm
        self._unit = unit
        self.atoms = tuple(str(atom) for atom in atoms)
        self.charge = int(charge)
        self.multiplicity = int(multiplicity)
        if self.multiplicity < 1:
            raise ValueError("OpenMM multiplicity must be positive")
        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        if coordinates.shape != (len(self.atoms), 3) or np.any(~np.isfinite(coordinates)):
            raise ValueError("OpenMM coordinates must have shape natoms x 3")

        topology = app.Topology()
        chain = topology.addChain("A")
        residue = topology.addResidue("MOL", chain)
        topology_atoms = []
        for index, symbol in enumerate(self.atoms):
            try:
                element = app.element.Element.getBySymbol(symbol)
            except Exception as exc:
                raise ValueError(f"OpenMM does not recognize element {symbol!r}") from exc
            topology_atoms.append(topology.addAtom(f"{symbol}{index + 1}", element, residue))
        canonical_bonds = tuple(
            sorted({tuple(sorted((int(left), int(right)))) for left, right in bonds})
        )
        for left, right in canonical_bonds:
            if left < 0 or right >= len(self.atoms) or left == right:
                raise ValueError(f"invalid SWITCH topology bond {(left, right)}")
            topology.addBond(topology_atoms[left], topology_atoms[right])

        if system_xml is not None:
            source = Path(system_xml).expanduser().resolve()
            if not source.is_file():
                raise FileNotFoundError(f"OpenMM System XML not found: {source}")
            system = openmm.XmlSerializer.deserialize(source.read_text(encoding="utf-8"))
            resolved_potential = f"system:{source}"
        else:
            try:
                from openmmml import MLPotential
            except ImportError as exc:
                raise RuntimeError(
                    "OpenMM ML preoptimization requires openmmml and aimnet"
                ) from exc
            requested = str(potential).strip().casefold().replace("_", "-") or "aimnet2"
            resolved_potential = (
                "aimnet2-nse"
                if requested == "aimnet2" and self.multiplicity > 1
                else requested
            )
            system = MLPotential(resolved_potential).createSystem(
                topology,
                charge=self.charge,
                multiplicity=self.multiplicity,
            )
        if system.getNumParticles() != len(self.atoms):
            raise ValueError("OpenMM System particle count does not match SWITCH atom order")

        integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
        # The Reference platform avoids loading a second OpenMP runtime into
        # Python processes that already host PySCF/PyFock/Numba on macOS.
        # Callers may still request another OpenMM platform explicitly.
        selected_platform = openmm.Platform.getPlatformByName(str(platform or "Reference"))
        context = openmm.Context(system, integrator, selected_platform)
        self.topology = topology
        self.system = system
        self.integrator = integrator
        self.context = context
        self.platform = selected_platform.getName()
        self.potential = resolved_potential
        self.evaluate(coordinates)

    def evaluate(self, coordinates_angstrom: np.ndarray) -> OpenMMEvaluation:
        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        if coordinates.shape != (len(self.atoms), 3) or np.any(~np.isfinite(coordinates)):
            raise ValueError("OpenMM coordinates must have shape natoms x 3")
        self.context.setPositions(coordinates * self._unit.angstrom)
        state = self.context.getState(getEnergy=True, getForces=True)
        energy_kj_mol = float(
            state.getPotentialEnergy().value_in_unit(self._unit.kilojoule_per_mole)
        )
        forces_kj_mol_nm = np.asarray(
            state.getForces(asNumpy=True).value_in_unit(
                self._unit.kilojoule_per_mole / self._unit.nanometer
            ),
            dtype=float,
        )
        gradient = -forces_kj_mol_nm * BOHR_IN_NANOMETER / KJ_MOL_PER_HARTREE
        if np.any(~np.isfinite(gradient)) or not np.isfinite(energy_kj_mol):
            raise RuntimeError("OpenMM returned non-finite energy or forces")
        return OpenMMEvaluation(
            energy_hartree=energy_kj_mol / KJ_MOL_PER_HARTREE,
            gradient_hartree_per_bohr=gradient,
            potential=self.potential,
        )

    def close(self) -> None:
        """Release the OpenMM context owned by this session."""

        self.context = None
        self.integrator = None


__all__ = [
    "OpenMMEvaluation",
    "OpenMMResidentSession",
    "openmm_available",
    "openmm_ml_available",
]
