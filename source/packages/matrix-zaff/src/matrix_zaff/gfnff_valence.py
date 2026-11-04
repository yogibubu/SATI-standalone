"""Resident exact local GFN-FF angle and torsion families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .native_kernels import (
    gfnff_valence_energy_gradient,
    gfnff_valence_hessian_vector,
)


@dataclass(frozen=True)
class GFNFFValenceResult:
    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray
    angle_count: int
    torsion_count: int
    special_torsion_count: int
    backend: str = "CPP_DIRECTIONAL_JET_GFNFF_VALENCE"


@dataclass(frozen=True)
class GFNFFValenceModel:
    """Prepared nonperiodic valence terms in xTB atomic units.

    Angle parameter columns are ``theta0, k, rcut1^2, rcut2^2``. Torsion
    columns are ``mode, periodicity, phi0, k, rcut1^2, rcut2^2, rcut3^2``;
    mode is positive for proper torsions, zero for single-minimum impropers,
    and -1 for double-minimum impropers.
    """

    angle_atoms: np.ndarray
    angle_parameters: np.ndarray
    torsion_atoms: np.ndarray
    torsion_parameters: np.ndarray
    special_torsion_atoms: np.ndarray
    special_torsion_amplitudes_hartree: np.ndarray

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "GFNFFValenceModel":
        if record.get("kind") != "GFNFF_EXACT_LOCAL_VALENCE":
            raise ValueError("unsupported GFN-FF valence record")
        return cls(
            angle_atoms=np.asarray(record.get("angle_atoms", ()), dtype=np.intp),
            angle_parameters=np.asarray(record.get("angle_parameters", ()), dtype=float),
            torsion_atoms=np.asarray(record.get("torsion_atoms", ()), dtype=np.intp),
            torsion_parameters=np.asarray(record.get("torsion_parameters", ()), dtype=float),
            special_torsion_atoms=np.asarray(
                record.get("special_torsion_atoms", ()), dtype=np.intp
            ),
            special_torsion_amplitudes_hartree=np.asarray(
                record.get("special_torsion_amplitudes_hartree", ()), dtype=float
            ),
        )

    def __post_init__(self) -> None:
        arrays = {
            "angle_atoms": np.asarray(self.angle_atoms, dtype=np.intp).reshape(-1, 3),
            "angle_parameters": np.asarray(self.angle_parameters, dtype=float).reshape(-1, 4),
            "torsion_atoms": np.asarray(self.torsion_atoms, dtype=np.intp).reshape(-1, 4),
            "torsion_parameters": np.asarray(self.torsion_parameters, dtype=float).reshape(-1, 7),
            "special_torsion_atoms": np.asarray(self.special_torsion_atoms, dtype=np.intp).reshape(
                -1, 4
            ),
            "special_torsion_amplitudes_hartree": np.asarray(
                self.special_torsion_amplitudes_hartree, dtype=float
            ).reshape(-1),
        }
        if len(arrays["angle_atoms"]) != len(arrays["angle_parameters"]):
            raise ValueError("GFN-FF angle arrays have inconsistent dimensions")
        if len(arrays["torsion_atoms"]) != len(arrays["torsion_parameters"]):
            raise ValueError("GFN-FF torsion arrays have inconsistent dimensions")
        if len(arrays["special_torsion_atoms"]) != len(
            arrays["special_torsion_amplitudes_hartree"]
        ):
            raise ValueError("GFN-FF special torsion arrays have inconsistent dimensions")
        if any(np.any(~np.isfinite(array)) for array in arrays.values()):
            raise ValueError("GFN-FF valence parameters must be finite")
        for name, array in arrays.items():
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    def evaluate(self, coordinates_bohr: np.ndarray) -> GFNFFValenceResult:
        xyz = self._coordinates(coordinates_bohr)
        energy, gradient, count = gfnff_valence_energy_gradient(
            xyz,
            self.angle_atoms,
            self.angle_parameters,
            self.torsion_atoms,
            self.torsion_parameters,
            self.special_torsion_atoms,
            self.special_torsion_amplitudes_hartree,
        )
        expected = self.term_count
        if count != expected:
            raise RuntimeError("native GFN-FF valence term count changed")
        return GFNFFValenceResult(
            energy_hartree=energy,
            gradient_hartree_per_bohr=gradient.reshape(-1),
            angle_count=len(self.angle_atoms),
            torsion_count=len(self.torsion_atoms),
            special_torsion_count=len(self.special_torsion_atoms),
        )

    def hessian_vector_product(
        self, coordinates_bohr: np.ndarray, direction_bohr: np.ndarray
    ) -> np.ndarray:
        xyz = self._coordinates(coordinates_bohr)
        direction = np.asarray(direction_bohr, dtype=float)
        if direction.shape != xyz.shape or np.any(~np.isfinite(direction)):
            raise ValueError("GFN-FF valence direction has wrong dimensions")
        product, count = gfnff_valence_hessian_vector(
            xyz,
            self.angle_atoms,
            self.angle_parameters,
            self.torsion_atoms,
            self.torsion_parameters,
            self.special_torsion_atoms,
            self.special_torsion_amplitudes_hartree,
            direction,
        )
        if count != self.term_count:
            raise RuntimeError("native GFN-FF valence term count changed")
        return product.reshape(-1)

    @property
    def term_count(self) -> int:
        return len(self.angle_atoms) + len(self.torsion_atoms) + len(self.special_torsion_atoms)

    def _coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        xyz = np.asarray(coordinates, dtype=float)
        atom_limit = max(
            (
                int(np.max(array))
                for array in (
                    self.angle_atoms,
                    self.torsion_atoms,
                    self.special_torsion_atoms,
                )
                if len(array)
            ),
            default=-1,
        )
        if (
            xyz.ndim != 2
            or xyz.shape[1] != 3
            or xyz.shape[0] <= atom_limit
            or np.any(~np.isfinite(xyz))
        ):
            raise ValueError("GFN-FF valence coordinates have wrong dimensions")
        return xyz


__all__ = ["GFNFFValenceModel", "GFNFFValenceResult"]
