"""Persistent ASE-compatible potential boundary for LINK path calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EV_PER_HARTREE = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903


@dataclass(frozen=True)
class ASEPotentialEvaluation:
    """Energy and Cartesian gradient returned in LINK atomic units."""

    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray


class ASEPotentialSession:
    """Adapt a persistent ASE-style atoms/calculator pair to LINK.

    The calculator need only implement ``get_potential_energy(atoms)`` and
    ``get_forces(atoms)``.  This includes ordinary ASE calculators and neural
    potentials that expose the same two methods without subclassing ASE's
    calculator base class.
    """

    def __init__(self, atoms: object, calculator: object) -> None:
        self.atoms = atoms
        self.calculator = calculator
        self.electronic_calls = 0
        self.cache_hits = 0
        self._cache: dict[bytes, ASEPotentialEvaluation] = {}

    def evaluate(self, coordinates_angstrom: np.ndarray) -> ASEPotentialEvaluation:
        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        atom_count = len(self.atoms)
        if coordinates.shape != (atom_count, 3) or not np.all(np.isfinite(coordinates)):
            raise ValueError("ASE potential coordinates must have shape natoms x 3")
        key = np.asarray(coordinates, dtype="<f8").tobytes(order="C")
        if key in self._cache:
            self.cache_hits += 1
            cached = self._cache[key]
            return ASEPotentialEvaluation(
                cached.energy_hartree,
                cached.gradient_hartree_per_bohr.copy(),
            )
        self.atoms.set_positions(coordinates)
        energy_ev = float(
            np.asarray(self.calculator.get_potential_energy(self.atoms)).reshape(-1)[0]
        )
        forces_ev_per_angstrom = np.asarray(
            self.calculator.get_forces(self.atoms), dtype=float,
        )
        if forces_ev_per_angstrom.shape != coordinates.shape:
            raise ValueError("ASE potential returned forces with the wrong shape")
        result = ASEPotentialEvaluation(
            energy_ev / EV_PER_HARTREE,
            (
                -forces_ev_per_angstrom
                * BOHR_TO_ANGSTROM
                / EV_PER_HARTREE
            ).reshape(-1),
        )
        if (
            not np.isfinite(result.energy_hartree)
            or not np.all(np.isfinite(result.gradient_hartree_per_bohr))
        ):
            raise ValueError("ASE potential returned a non-finite energy or force")
        self._cache[key] = result
        self.electronic_calls += 1
        return ASEPotentialEvaluation(
            result.energy_hartree,
            result.gradient_hartree_per_bohr.copy(),
        )

    def numerical_hessian(
        self,
        coordinates_angstrom: np.ndarray,
        *,
        step_bohr: float = 2.0e-3,
    ) -> np.ndarray:
        """Differentiate analytic gradients to a Cartesian Hessian.

        Displacements are specified in bohr, and the returned symmetric matrix
        is in hartree/bohr squared.
        """

        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        if step_bohr <= 0.0 or not np.isfinite(step_bohr):
            raise ValueError("numerical Hessian step must be positive and finite")
        flat = coordinates.reshape(-1)
        hessian = np.empty((flat.size, flat.size))
        step_angstrom = step_bohr * BOHR_TO_ANGSTROM
        for column in range(flat.size):
            positive = flat.copy()
            negative = flat.copy()
            positive[column] += step_angstrom
            negative[column] -= step_angstrom
            positive_gradient = self.evaluate(
                positive.reshape(coordinates.shape)
            ).gradient_hartree_per_bohr
            negative_gradient = self.evaluate(
                negative.reshape(coordinates.shape)
            ).gradient_hartree_per_bohr
            hessian[:, column] = (
                positive_gradient - negative_gradient
            ) / (2.0 * step_bohr)
        return 0.5 * (hessian + hessian.T)


__all__ = ["ASEPotentialEvaluation", "ASEPotentialSession"]
