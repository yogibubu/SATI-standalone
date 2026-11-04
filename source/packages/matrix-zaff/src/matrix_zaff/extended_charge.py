"""Constrained SCF and extended-Lagrangian propagation of fitted charges."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


KB_HARTREE_PER_KELVIN = 3.166811563e-6


def _constraint_projector(matrix: np.ndarray) -> np.ndarray:
    constraints = np.asarray(matrix, dtype=float)
    if constraints.ndim != 2 or constraints.shape[0] < 1:
        raise ValueError("charge constraint matrix must have shape ncharge x nconstraint")
    gram = constraints.T @ constraints
    if np.linalg.matrix_rank(gram) != gram.shape[0]:
        raise ValueError("charge constraints must be linearly independent")
    return constraints @ np.linalg.solve(gram, constraints.T)


def project_constrained_charges(
    charges_e: np.ndarray,
    constraint_matrix: np.ndarray,
    totals_e: np.ndarray,
) -> np.ndarray:
    """Project charges onto ``C.T @ q = totals`` with minimum Euclidean change."""

    charges = np.asarray(charges_e, dtype=float).reshape(-1)
    constraints = np.asarray(constraint_matrix, dtype=float)
    totals = np.asarray(totals_e, dtype=float).reshape(-1)
    if constraints.shape[0] != charges.size or constraints.shape[1] != totals.size:
        raise ValueError("charge constraints, totals, and charge vector disagree")
    correction = constraints @ np.linalg.solve(
        constraints.T @ constraints,
        totals - constraints.T @ charges,
    )
    return charges + correction


@dataclass(frozen=True)
class ExtendedLagrangianChargeState:
    charges_e: np.ndarray
    velocities_e_per_atomic_time: np.ndarray
    target_charges_e: np.ndarray
    step: int = 0
    rng_state: dict[str, object] | None = None

    def __post_init__(self) -> None:
        charges = np.asarray(self.charges_e, dtype=float).reshape(-1)
        velocities = np.asarray(self.velocities_e_per_atomic_time, dtype=float).reshape(-1)
        target = np.asarray(self.target_charges_e, dtype=float).reshape(-1)
        if charges.size == 0 or velocities.shape != charges.shape or target.shape != charges.shape:
            raise ValueError("extended-charge state arrays are inconsistent")
        if np.any(~np.isfinite(charges)) or np.any(~np.isfinite(velocities)) or np.any(~np.isfinite(target)):
            raise ValueError("extended-charge state must be finite")
        object.__setattr__(self, "charges_e", charges.copy())
        object.__setattr__(self, "velocities_e_per_atomic_time", velocities.copy())
        object.__setattr__(self, "target_charges_e", target.copy())


@dataclass(frozen=True)
class ExtendedLagrangianChargePropagator:
    """BAOAB propagation around the instantaneous density-fitted charge target."""

    time_step_atomic: float
    frequency_per_atomic_time: float
    friction_per_atomic_time: float
    fictitious_mass_hartree_atomic_time2_per_e2: float
    temperature_kelvin: float
    constraint_matrix: np.ndarray
    totals_e: np.ndarray
    initial_seed: int = 0

    def __post_init__(self) -> None:
        finite = np.asarray(
            [
                self.time_step_atomic,
                self.frequency_per_atomic_time,
                self.friction_per_atomic_time,
                self.fictitious_mass_hartree_atomic_time2_per_e2,
                self.temperature_kelvin,
            ],
            dtype=float,
        )
        if (
            np.any(~np.isfinite(finite))
            or self.time_step_atomic <= 0.0
            or self.frequency_per_atomic_time <= 0.0
            or self.friction_per_atomic_time < 0.0
            or self.fictitious_mass_hartree_atomic_time2_per_e2 <= 0.0
            or self.temperature_kelvin < 0.0
        ):
            raise ValueError("extended-charge propagation controls are invalid")
        constraints = np.asarray(self.constraint_matrix, dtype=float)
        totals = np.asarray(self.totals_e, dtype=float).reshape(-1)
        _constraint_projector(constraints)
        if constraints.shape[1] != totals.size:
            raise ValueError("extended-charge totals disagree with constraints")
        object.__setattr__(self, "constraint_matrix", constraints.copy())
        object.__setattr__(self, "totals_e", totals.copy())

    def initialize(self, target_charges_e: np.ndarray) -> ExtendedLagrangianChargeState:
        target = project_constrained_charges(
            target_charges_e, self.constraint_matrix, self.totals_e
        )
        return ExtendedLagrangianChargeState(target, np.zeros_like(target), target)

    def step(
        self,
        state: ExtendedLagrangianChargeState,
        target_charges_e: np.ndarray,
    ) -> ExtendedLagrangianChargeState:
        target = project_constrained_charges(
            target_charges_e, self.constraint_matrix, self.totals_e
        )
        if target.shape != state.charges_e.shape:
            raise ValueError("new fitted charges disagree with the extended-charge state")
        dt = float(self.time_step_atomic)
        omega2 = float(self.frequency_per_atomic_time) ** 2
        charges = state.charges_e.copy()
        velocities = state.velocities_e_per_atomic_time.copy()
        velocities -= 0.5 * dt * omega2 * (charges - target)
        charges += 0.5 * dt * velocities
        charges = project_constrained_charges(
            charges, self.constraint_matrix, self.totals_e
        )

        rng = np.random.default_rng(
            np.random.SeedSequence([int(self.initial_seed), int(state.step)])
        )
        if state.rng_state is not None:
            rng.bit_generator.state = state.rng_state
        damping = np.exp(-float(self.friction_per_atomic_time) * dt)
        variance = (
            KB_HARTREE_PER_KELVIN
            * float(self.temperature_kelvin)
            / float(self.fictitious_mass_hartree_atomic_time2_per_e2)
            * (1.0 - damping * damping)
        )
        velocities = damping * velocities + np.sqrt(max(0.0, variance)) * rng.normal(
            size=velocities.size
        )
        tangent = np.eye(velocities.size) - _constraint_projector(self.constraint_matrix)
        velocities = tangent @ velocities

        charges += 0.5 * dt * velocities
        charges = project_constrained_charges(
            charges, self.constraint_matrix, self.totals_e
        )
        velocities -= 0.5 * dt * omega2 * (charges - target)
        velocities = tangent @ velocities
        return ExtendedLagrangianChargeState(
            charges,
            velocities,
            target,
            step=int(state.step) + 1,
            rng_state=rng.bit_generator.state,
        )

    def harmonic_energy_hartree(self, state: ExtendedLagrangianChargeState) -> float:
        displacement = state.charges_e - state.target_charges_e
        coefficient = (
            self.fictitious_mass_hartree_atomic_time2_per_e2
            * self.frequency_per_atomic_time**2
        )
        return 0.5 * float(coefficient * np.dot(displacement, displacement))


@dataclass(frozen=True)
class SelfConsistentChargeResult:
    charges_e: np.ndarray
    iterations: int
    residual_e: float


def solve_self_consistent_charges(
    initial_charges_e: np.ndarray,
    target: Callable[[np.ndarray], np.ndarray],
    constraint_matrix: np.ndarray,
    totals_e: np.ndarray,
    *,
    tolerance_e: float = 1.0e-7,
    maximum_iterations: int = 100,
    mixing: float = 0.5,
) -> SelfConsistentChargeResult:
    """Converge a fitted-charge/embedding macroiteration with exact constraints."""

    if tolerance_e <= 0.0 or maximum_iterations < 1 or not 0.0 < mixing <= 1.0:
        raise ValueError("SCF charge controls are invalid")
    charges = project_constrained_charges(
        initial_charges_e, constraint_matrix, totals_e
    )
    for iteration in range(1, int(maximum_iterations) + 1):
        fitted = project_constrained_charges(
            target(charges.copy()), constraint_matrix, totals_e
        )
        residual = float(np.max(np.abs(fitted - charges)))
        charges = project_constrained_charges(
            charges + float(mixing) * (fitted - charges),
            constraint_matrix,
            totals_e,
        )
        if residual <= tolerance_e:
            return SelfConsistentChargeResult(charges, iteration, residual)
    raise RuntimeError(
        "fitted-charge/embedding macroiteration did not converge within the limit"
    )


__all__ = [
    "ExtendedLagrangianChargePropagator",
    "ExtendedLagrangianChargeState",
    "SelfConsistentChargeResult",
    "project_constrained_charges",
    "solve_self_consistent_charges",
]
