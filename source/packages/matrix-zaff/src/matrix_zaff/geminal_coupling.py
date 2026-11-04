"""Dissociating geminal X-H cross term for the general ZAFF field."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class ZaffGeminalXHCoupling:
    """K Δr1 Δr2 exp[-(α1 Δr1)^4] exp[-(α2 Δr2)^4].

    Distances and alpha use bohr. The coupled Hessian at the reference is
    exactly K times the symmetric outer product of the two bond B rows.
    """

    center: int
    hydrogen1: int
    hydrogen2: int
    reference1_bohr: float
    reference2_bohr: float
    alpha1_per_bohr: float
    alpha2_per_bohr: float
    force_constant_hartree_per_bohr2: float

    def __post_init__(self) -> None:
        if min(self.center, self.hydrogen1, self.hydrogen2) < 0 or len({
            self.center, self.hydrogen1, self.hydrogen2,
        }) != 3:
            raise ValueError("geminal coupling needs distinct center and hydrogen atoms")
        values = (
            self.reference1_bohr, self.reference2_bohr,
            self.alpha1_per_bohr, self.alpha2_per_bohr,
            self.force_constant_hartree_per_bohr2,
        )
        if not all(isfinite(float(value)) for value in values) or min(values[:4]) <= 0:
            raise ValueError("geminal coupling parameters must be finite and radial scales positive")

    @staticmethod
    def _radial(distance: float, reference: float, alpha: float) -> tuple[float, float, float]:
        displacement = distance - reference
        scaled = alpha * displacement
        if abs(scaled) >= 10.0:
            return 0.0, 0.0, 0.0
        damping = exp(-(scaled**4))
        return (
            displacement * damping,
            damping * (1.0 - 4.0 * scaled**4),
            damping * alpha * (-20.0 * scaled**3 + 16.0 * scaled**7),
        )

    @staticmethod
    def _bond_derivatives(xyz: np.ndarray, center: int, hydrogen: int) -> tuple[float, np.ndarray, np.ndarray]:
        delta = xyz[hydrogen] - xyz[center]
        distance = float(np.linalg.norm(delta))
        if distance <= 0 or not isfinite(distance):
            raise ValueError("geminal X-H distance must be positive and finite")
        unit = delta / distance
        b = np.zeros(xyz.size)
        b[3 * center:3 * center + 3] = -unit
        b[3 * hydrogen:3 * hydrogen + 3] = unit
        radial = (np.eye(3) - np.outer(unit, unit)) / distance
        hessian = np.zeros((xyz.size, xyz.size))
        c = slice(3 * center, 3 * center + 3)
        h = slice(3 * hydrogen, 3 * hydrogen + 3)
        hessian[c, c] += radial
        hessian[h, h] += radial
        hessian[c, h] -= radial
        hessian[h, c] -= radial
        return distance, b, hessian

    def reference_hessian(self, reference_xyz_bohr: np.ndarray) -> np.ndarray:
        xyz = np.asarray(reference_xyz_bohr, dtype=float)
        r1, b1, _ = self._bond_derivatives(xyz, self.center, self.hydrogen1)
        r2, b2, _ = self._bond_derivatives(xyz, self.center, self.hydrogen2)
        if abs(r1 - self.reference1_bohr) > 1.0e-6 or abs(r2 - self.reference2_bohr) > 1.0e-6:
            raise ValueError("geminal coupling reference geometry differs from the field")
        return self.force_constant_hartree_per_bohr2 * (
            np.outer(b1, b2) + np.outer(b2, b1)
        )

    def correction(
        self, xyz_bohr: np.ndarray, reference_xyz_bohr: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Return the nonlinear term minus its Cartesian reference quadratic."""

        xyz = np.asarray(xyz_bohr, dtype=float)
        reference = np.asarray(reference_xyz_bohr, dtype=float)
        if xyz.shape != reference.shape or max(self.center, self.hydrogen1, self.hydrogen2) >= len(xyz):
            raise ValueError("geminal coupling atom indices differ from the field")
        r1, b1, hr1 = self._bond_derivatives(xyz, self.center, self.hydrogen1)
        r2, b2, hr2 = self._bond_derivatives(xyz, self.center, self.hydrogen2)
        f1, d1, dd1 = self._radial(r1, self.reference1_bohr, self.alpha1_per_bohr)
        f2, d2, dd2 = self._radial(r2, self.reference2_bohr, self.alpha2_per_bohr)
        k = self.force_constant_hartree_per_bohr2
        energy = k * f1 * f2
        gradient = k * (d1 * f2 * b1 + f1 * d2 * b2)
        hessian = k * (
            dd1 * f2 * np.outer(b1, b1) + d1 * f2 * hr1
            + f1 * dd2 * np.outer(b2, b2) + f1 * d2 * hr2
            + d1 * d2 * (np.outer(b1, b2) + np.outer(b2, b1))
        )
        reference_hessian = self.reference_hessian(reference)
        displacement = (xyz - reference).reshape(-1)
        energy -= 0.5 * float(displacement @ reference_hessian @ displacement)
        gradient -= reference_hessian @ displacement
        hessian -= reference_hessian
        return float(energy), gradient, hessian

    def to_dict(self) -> dict[str, Any]:
        return {
            "functional_form": "DISPLACEMENT_QUARTIC_DAMPED_PRODUCT",
            "center_atom_one_based": self.center + 1,
            "hydrogens_one_based": [self.hydrogen1 + 1, self.hydrogen2 + 1],
            "reference_distances_bohr": [self.reference1_bohr, self.reference2_bohr],
            "alpha_per_bohr": [self.alpha1_per_bohr, self.alpha2_per_bohr],
            "force_constant_hartree_per_bohr2": self.force_constant_hartree_per_bohr2,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ZaffGeminalXHCoupling":
        if payload.get("functional_form") != "DISPLACEMENT_QUARTIC_DAMPED_PRODUCT":
            raise ValueError("unsupported geminal ZAFF coupling form")
        hydrogens = payload["hydrogens_one_based"]
        references = payload["reference_distances_bohr"]
        alphas = payload["alpha_per_bohr"]
        return cls(
            center=int(payload["center_atom_one_based"]) - 1,
            hydrogen1=int(hydrogens[0]) - 1,
            hydrogen2=int(hydrogens[1]) - 1,
            reference1_bohr=float(references[0]),
            reference2_bohr=float(references[1]),
            alpha1_per_bohr=float(alphas[0]),
            alpha2_per_bohr=float(alphas[1]),
            force_constant_hartree_per_bohr2=float(payload["force_constant_hartree_per_bohr2"]),
        )
