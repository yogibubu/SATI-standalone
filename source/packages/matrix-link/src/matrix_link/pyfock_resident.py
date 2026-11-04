"""Persistent PyFock configuration and basis cache for LINK."""

from __future__ import annotations

from typing import Sequence

import numpy as np


class PyFockResidentSession:
    """Cache PyFock imports and atom/basis data across geometry evaluations.

    PyFock 0.1.7 does not expose a geometry-reset method for a DFT object, so
    geometry-dependent integral containers are rebuilt.  The resident session
    still avoids repeated basis-file parsing; APOC owns the accepted density
    and Fock state used between evaluations.
    """

    def __init__(
        self,
        atoms: Sequence[str],
        coordinates_angstrom: np.ndarray,
        *,
        basis_name: str,
        method: str,
        charge: int,
        processors: int,
    ) -> None:
        try:
            from pyfock import Basis, DFT, DFT_Grad, DFT_NumGrad, Mol
            from pyfock.Integrals import overlap_mat_symm
        except ImportError as exc:
            raise RuntimeError(
                "PyFock backend requested but pyfock is not installed in the MATRIX environment"
            ) from exc
        self.Basis = Basis
        self.DFT = DFT
        self.DFT_Grad = DFT_Grad
        self.DFT_NumGrad = DFT_NumGrad
        self.Mol = Mol
        self.overlap_mat_symm = overlap_mat_symm
        self.atoms = tuple(atoms)
        self.basis_name = str(basis_name)
        self.method = str(method)
        self.charge = int(charge)
        self.processors = max(1, int(processors))
        molecule = self.molecule(coordinates_angstrom)
        self._basis_payloads = {}
        for name in (self.basis_name, "sto-3g", "def2-universal-jfit"):
            key = name.casefold()
            if key not in self._basis_payloads:
                self._basis_payloads[key] = {
                    "all": Basis.load(mol=molecule, basis_name=name)
                }

    def molecule(self, coordinates_angstrom: np.ndarray):
        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        if coordinates.shape != (len(self.atoms), 3):
            raise ValueError("PyFock resident coordinates must have shape natoms x 3")
        payload = [
            [symbol, *map(float, xyz)] for symbol, xyz in zip(self.atoms, coordinates, strict=True)
        ]
        return self.Mol(atoms=payload, charge=self.charge)

    def basis(self, molecule, basis_name: str | None = None):
        name = str(basis_name or self.basis_name)
        key = name.casefold()
        payload = self._basis_payloads.get(key)
        if payload is None:
            payload = {"all": self.Basis.load(mol=molecule, basis_name=name)}
            self._basis_payloads[key] = payload
        return self.Basis(molecule, payload)

    def calculation(self, molecule, basis, *, method: str | None = None):
        auxiliary = self.basis(molecule, "def2-universal-jfit")
        return self.DFT(
            molecule,
            basis,
            auxbasis=auxiliary,
            xc=str(method or self.method),
            ncores=self.processors,
        )


__all__ = ["PyFockResidentSession"]
