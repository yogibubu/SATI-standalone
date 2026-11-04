"""Sparse resident GFN-FF hydrogen/halogen bonds and bonded ATM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .native_kernels import gfnff_nci_energy_gradient, gfnff_nci_hessian_vector


@dataclass(frozen=True)
class GFNFFNCIResult:
    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray
    hydrogen_bond_count: int
    halogen_bond_count: int
    bonded_atm_count: int
    backend: str = "CPP_DIRECTIONAL_JET_GFNFF_NCI"


@dataclass(frozen=True)
class GFNFFNCIModel:
    """Prepared nonperiodic GFN-FF NCI terms.

    Hydrogen-bond rows have seven atom slots ``A,B,H,N1..N4`` and sixteen
    resolved parameters. Modes 1--4 reproduce the terminal,
    acceptor-neighbor-oriented, aromatic lone-pair, and carbonyl-specific xTB
    forms. Unused atom slots are -1.
    Halogen bonds and bonded ATM use fixed three-atom rows.
    """

    hydrogen_bond_atoms: np.ndarray
    hydrogen_bond_parameters: np.ndarray
    halogen_bond_atoms: np.ndarray
    halogen_bond_parameters: np.ndarray
    bonded_atm_atoms: np.ndarray
    bonded_atm_amplitudes_hartree_bohr9: np.ndarray

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "GFNFFNCIModel":
        if record.get("kind") != "GFNFF_EXACT_NCI":
            raise ValueError("unsupported GFN-FF NCI record")
        return cls(
            hydrogen_bond_atoms=np.asarray(record.get("hydrogen_bond_atoms", ()), dtype=np.intp),
            hydrogen_bond_parameters=np.asarray(
                record.get("hydrogen_bond_parameters", ()), dtype=float
            ),
            halogen_bond_atoms=np.asarray(record.get("halogen_bond_atoms", ()), dtype=np.intp),
            halogen_bond_parameters=np.asarray(
                record.get("halogen_bond_parameters", ()), dtype=float
            ),
            bonded_atm_atoms=np.asarray(record.get("bonded_atm_atoms", ()), dtype=np.intp),
            bonded_atm_amplitudes_hartree_bohr9=np.asarray(
                record.get("bonded_atm_amplitudes_hartree_bohr9", ()),
                dtype=float,
            ),
        )

    def __post_init__(self) -> None:
        arrays = {
            "hydrogen_bond_atoms": np.asarray(self.hydrogen_bond_atoms, dtype=np.intp).reshape(
                -1, 7
            ),
            "hydrogen_bond_parameters": np.asarray(
                self.hydrogen_bond_parameters, dtype=float
            ).reshape(-1, 22),
            "halogen_bond_atoms": np.asarray(self.halogen_bond_atoms, dtype=np.intp).reshape(-1, 3),
            "halogen_bond_parameters": np.asarray(
                self.halogen_bond_parameters, dtype=float
            ).reshape(-1, 7),
            "bonded_atm_atoms": np.asarray(self.bonded_atm_atoms, dtype=np.intp).reshape(-1, 3),
            "bonded_atm_amplitudes_hartree_bohr9": np.asarray(
                self.bonded_atm_amplitudes_hartree_bohr9, dtype=float
            ).reshape(-1),
        }
        if len(arrays["hydrogen_bond_atoms"]) != len(arrays["hydrogen_bond_parameters"]):
            raise ValueError("GFN-FF hydrogen-bond arrays disagree")
        if len(arrays["halogen_bond_atoms"]) != len(arrays["halogen_bond_parameters"]):
            raise ValueError("GFN-FF halogen-bond arrays disagree")
        if len(arrays["bonded_atm_atoms"]) != len(arrays["bonded_atm_amplitudes_hartree_bohr9"]):
            raise ValueError("GFN-FF bonded-ATM arrays disagree")
        if any(np.any(~np.isfinite(array)) for array in arrays.values()):
            raise ValueError("GFN-FF NCI parameters must be finite")
        for name, array in arrays.items():
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @property
    def term_count(self) -> int:
        return (
            len(self.hydrogen_bond_atoms)
            + len(self.halogen_bond_atoms)
            + len(self.bonded_atm_atoms)
        )

    def evaluate(self, coordinates_bohr: np.ndarray) -> GFNFFNCIResult:
        xyz = self._coordinates(coordinates_bohr)
        energy, gradient, count = gfnff_nci_energy_gradient(
            xyz,
            self.hydrogen_bond_atoms,
            self.hydrogen_bond_parameters,
            self.halogen_bond_atoms,
            self.halogen_bond_parameters,
            self.bonded_atm_atoms,
            self.bonded_atm_amplitudes_hartree_bohr9,
        )
        if count != self.term_count:
            raise RuntimeError("native GFN-FF NCI term count changed")
        return GFNFFNCIResult(
            energy_hartree=energy,
            gradient_hartree_per_bohr=gradient.reshape(-1),
            hydrogen_bond_count=len(self.hydrogen_bond_atoms),
            halogen_bond_count=len(self.halogen_bond_atoms),
            bonded_atm_count=len(self.bonded_atm_atoms),
        )

    def hessian_vector_product(
        self, coordinates_bohr: np.ndarray, direction_bohr: np.ndarray
    ) -> np.ndarray:
        xyz = self._coordinates(coordinates_bohr)
        direction = np.asarray(direction_bohr, dtype=float)
        if direction.shape != xyz.shape or np.any(~np.isfinite(direction)):
            raise ValueError("GFN-FF NCI direction has wrong dimensions")
        product, count = gfnff_nci_hessian_vector(
            xyz,
            self.hydrogen_bond_atoms,
            self.hydrogen_bond_parameters,
            self.halogen_bond_atoms,
            self.halogen_bond_parameters,
            self.bonded_atm_atoms,
            self.bonded_atm_amplitudes_hartree_bohr9,
            direction,
        )
        if count != self.term_count:
            raise RuntimeError("native GFN-FF NCI term count changed")
        return product.reshape(-1)

    def _coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        xyz = np.asarray(coordinates, dtype=float)
        active = []
        for atoms in (
            self.hydrogen_bond_atoms,
            self.halogen_bond_atoms,
            self.bonded_atm_atoms,
        ):
            active.extend(int(value) for value in atoms.reshape(-1) if value >= 0)
        if (
            xyz.ndim != 2
            or xyz.shape[1] != 3
            or (active and xyz.shape[0] <= max(active))
            or np.any(~np.isfinite(xyz))
        ):
            raise ValueError("GFN-FF NCI coordinates have wrong dimensions")
        return xyz


__all__ = ["GFNFFNCIModel", "GFNFFNCIResult"]
