"""SATI consumption of EPOC state-specific nonequilibrium free energies.

Providers return an identified state density/source at each geometry. This
module never substitutes a transition density or relaxes an electronic state.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ElectronicSolvationSample:
    """Electronic energy baseline and matching EPOC total-density source.

    ``vacuum_expectation`` is <H_vac> for the supplied density, including
    nuclear repulsion, even if that density was relaxed in a reaction field.
    ``equilibrium_free_energy`` already includes exactly EPOC's static
    electrostatic free energy. An embedded SCF eigenvalue sum/total energy
    with unremoved reaction-field work is neither convention.
    """

    source: object
    energy_hartree: float
    energy_convention: str
    density_regime: str

    def __post_init__(self):
        if isinstance(self.energy_hartree, bool) or not math.isfinite(self.energy_hartree):
            raise ValueError("a finite electronic energy in hartree is required")
        if self.energy_convention not in {"vacuum_expectation", "equilibrium_free_energy"}:
            raise ValueError("explicit vacuum_expectation or equilibrium_free_energy convention required")
        if self.density_regime not in {"provided_density", "state_specific_scf"}:
            raise ValueError("explicit provided_density or state_specific_scf regime required")


def _solvated_energy(model, frozen, sample):
    if not isinstance(sample, ElectronicSolvationSample):
        raise TypeError("an ElectronicSolvationSample is required")
    solvent = model.evaluate(sample.source, frozen)
    vacuum = sample.energy_hartree
    if sample.energy_convention == "equilibrium_free_energy":
        vacuum -= solvent.equilibrium_energy_hartree
    return vacuum + solvent.energy_hartree, vacuum, solvent


@dataclass(frozen=True)
class VerticalSolvationTransition:
    process: str
    photon_energy_hartree: float
    electronic_gap_hartree: float
    solvent_shift_hartree: float
    final_reorganization_energy_hartree: float
    initial_state_id: str
    final_state_id: str
    frozen_reference_sha256: str
    initial_density_regime: str
    final_density_regime: str


def vertical_solvation_transition(model, frozen, initial, final, *, process):
    """Absorption/emission at one geometry and one prepared solvent field.

    Absorption prepares the slow field on the ground state; emission prepares
    it on the emitting state. To obtain a Stokes shift the caller supplies two
    separate preparations at their respective geometries. A relaxed 0-0 gap
    uses equilibrium surfaces, not either frozen-field vertical gap.
    """
    if process not in {"absorption", "emission"}:
        raise ValueError("process must be absorption or emission")
    if not isinstance(initial, ElectronicSolvationSample) or not isinstance(final, ElectronicSolvationSample):
        raise TypeError("identified initial and final samples are required")
    if initial.source != frozen.reference:
        raise ValueError("prepare the slow field on the supplied initial electronic source")
    if initial.source.geometry_id != final.source.geometry_id:
        raise ValueError("a vertical transition requires the same nuclear geometry")
    if initial.source.state_id == final.source.state_id:
        raise ValueError("a transition requires distinct electronic states")
    ei, vi, si = _solvated_energy(model, frozen, initial)
    ef, vf, sf = _solvated_energy(model, frozen, final)
    sign = 1.0 if process == "absorption" else -1.0
    photon = sign * (ef - ei)
    if not math.isfinite(photon) or photon <= 0.0:
        raise ValueError("the supplied states/energies do not yield a positive photon energy")
    return VerticalSolvationTransition(
        process, photon, sign * (vf - vi), sign * (sf.energy_hartree - si.energy_hartree),
        sf.reorganization_energy_hartree, initial.source.state_id, final.source.state_id,
        frozen.sha256, initial.density_regime, final.density_regime,
    )


@dataclass(frozen=True)
class NonequilibriumCartesianEnergy:
    """Angstrom -> hartree callback for SATI NativeCartesianModalPotential.

    ``sample_provider(geometry_angstrom)`` returns an ElectronicSolvationSample
    in the declared state. The EPOC cavity and prepared slow field stay fixed
    across all samples, including displaced vibronic grid points.
    """

    model: object
    frozen: object
    sample_provider: object
    state_id: str

    def __post_init__(self):
        if not callable(self.sample_provider) or not isinstance(self.state_id, str) or not self.state_id.strip():
            raise ValueError("an identified state and callable sample provider are required")
        self.model.validate_frozen(self.frozen)

    def __call__(self, geometry_angstrom):
        if np.iscomplexobj(geometry_angstrom):
            raise ValueError("real Cartesian geometry required")
        xyz = np.asarray(geometry_angstrom, dtype=float)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.all(np.isfinite(xyz)):
            raise ValueError("finite (atom,3) Cartesian geometry in angstrom required")
        sample = self.sample_provider(xyz.copy())
        if not isinstance(sample, ElectronicSolvationSample) or sample.source.state_id != self.state_id:
            raise ValueError("the provider returned a different or unidentified electronic state")
        # Enforce the callback's exact angstrom/bohr conversion boundary.
        from matrix_chem import BOHR_TO_ANGSTROM
        supplied_bohr = xyz / BOHR_TO_ANGSTROM
        expected = self.model.geometry_identifier(supplied_bohr)
        if sample.source.geometry_id != expected:
            raise ValueError("source geometry differs from the requested Cartesian sample")
        return _solvated_energy(self.model, self.frozen, sample)[0]


__all__ = ["ElectronicSolvationSample", "VerticalSolvationTransition",
           "vertical_solvation_transition", "NonequilibriumCartesianEnergy"]
