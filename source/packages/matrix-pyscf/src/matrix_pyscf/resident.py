"""Persistent in-process PySCF energy/gradient calculator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .parsers import _resolve_pyscf_reference


@dataclass(frozen=True)
class PySCFResidentEvaluation:
    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray | None
    scf_cycles: int
    density: np.ndarray
    fock: np.ndarray
    overlap: np.ndarray


class PySCFResidentCalculator:
    """Keep a PySCF gradient scanner alive across geometry iterations."""

    def __init__(
        self,
        atoms: Sequence[str],
        coordinates_angstrom: np.ndarray,
        *,
        method: str = "HF",
        reference: str | None = None,
        basis: str = "sto-3g",
        charge: int = 0,
        multiplicity: int = 1,
        density_fitting: bool = True,
        processors: int = 1,
    ) -> None:
        from pyscf import dft, gto, lib, scf

        lib.num_threads(max(1, int(processors)))
        self.atoms = tuple(atoms)
        self.method = str(method).strip() or "HF"
        self.basis = str(basis).strip() or "sto-3g"
        selected = _resolve_pyscf_reference(
            method=self.method,
            multiplicity=int(multiplicity),
            reference=reference,
        )
        molecule = gto.M(
            atom=self._geometry_payload(coordinates_angstrom),
            basis=self.basis,
            charge=int(charge),
            spin=int(multiplicity) - 1,
            unit="Angstrom",
            symmetry=False,
            verbose=0,
        )
        constructors = {
            "RHF": scf.RHF,
            "ROHF": scf.ROHF,
            "UHF": scf.UHF,
            "RKS": dft.RKS,
            "ROKS": dft.ROKS,
            "UKS": dft.UKS,
        }
        mean_field = constructors[selected](molecule)
        mean_field.verbose = 0
        if selected.endswith("KS"):
            mean_field.xc = self.method
        if density_fitting:
            mean_field = mean_field.density_fit()
            mean_field.verbose = 0
        self._molecule = molecule
        self._scanner = mean_field.nuc_grad_method().as_scanner()

    def _geometry_payload(self, coordinates_angstrom: np.ndarray):
        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        if coordinates.shape != (len(self.atoms), 3):
            raise ValueError("PySCF resident coordinates must have shape natoms x 3")
        return [
            [atom, tuple(float(value) for value in xyz)]
            for atom, xyz in zip(self.atoms, coordinates, strict=True)
        ]

    def evaluate(
        self,
        coordinates_angstrom: np.ndarray,
        *,
        density_guess: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None,
        include_gradient: bool = True,
    ) -> PySCFResidentEvaluation:
        molecule = self._molecule.set_geom_(
            self._geometry_payload(coordinates_angstrom),
            unit="Angstrom",
            inplace=False,
        )
        self._scanner.reset(molecule)
        mean_field = self._scanner.base
        if density_guess is None:
            energy = mean_field(molecule)
        else:
            energy = mean_field(molecule, dm0=density_guess)
        gradient = self._scanner.kernel() if include_gradient else None
        density = np.asarray(mean_field.make_rdm1(), dtype=float)
        fock = np.asarray(mean_field.get_fock(dm=density), dtype=float)
        overlap = np.asarray(mean_field.get_ovlp(), dtype=float)
        return PySCFResidentEvaluation(
            energy_hartree=float(energy),
            gradient_hartree_per_bohr=(
                None
                if gradient is None
                else np.asarray(gradient, dtype=float).reshape(-1)
            ),
            scf_cycles=int(getattr(mean_field, "cycles", 0)),
            density=density,
            fock=fock,
            overlap=overlap,
        )


__all__ = ["PySCFResidentCalculator", "PySCFResidentEvaluation"]
