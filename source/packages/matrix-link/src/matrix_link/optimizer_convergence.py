"""Convergence tests and gradient selection for the LINK optimizer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import numpy as np

from .optimizer_metrics import rms
from .scan import ANGSTROM_TO_BOHR


# Frozen role profile matching the xTB 6.7.1 ``normal`` ANC termination
# contract. These constants are deliberately independent of backend name:
# the caller selects the geometry-seed role explicitly.
GEOMETRY_SEED_ENERGY_TOLERANCE_HARTREE = 5.0e-6
GEOMETRY_SEED_GRADIENT_NORM_TOLERANCE_HARTREE_PER_BOHR = 1.0e-3
GEOMETRY_SEED_FINAL_ENERGY_RISE_TOLERANCE_HARTREE = 1.0e-10
# ASE's ``fmax=0.02`` criterion used by PyFock's native optimizer, converted
# from eV/Angstrom to Hartree/Bohr.
PYFOCK_ASE_FMAX_EV_PER_ANGSTROM = 2.0e-2
PYFOCK_ASE_FMAX_HARTREE_PER_BOHR = (
    PYFOCK_ASE_FMAX_EV_PER_ANGSTROM
    * 0.03674932217565499
    * 1.8897261246257702
)


class OptimizerConvergenceSettings(Protocol):
    convergence_profile: str
    stationary_point: str
    energy_tolerance: float
    max_force_tolerance: float
    rms_force_tolerance: float
    max_displacement_tolerance: float
    rms_displacement_tolerance: float
    freeze_inactive_sonic: bool


class OptimizerEvaluationLike(Protocol):
    gradient_hartree_per_bohr: np.ndarray | None
    coordinates_angstrom: np.ndarray


class CoordinateModelLike(Protocol):
    kind: str


class GeometryEvaluationServiceLike(Protocol):
    coordinate_model: CoordinateModelLike

    def coordinate_directions(self, coordinates_angstrom: np.ndarray) -> np.ndarray: ...


def gaussian_like_convergence(
    settings: OptimizerConvergenceSettings,
    energy_change: float,
    gradient: np.ndarray,
    step: np.ndarray,
) -> dict[str, bool]:
    """Evaluate the active force/displacement convergence profile."""

    grad = np.asarray(gradient, dtype=float).reshape(-1)
    disp = np.asarray(step, dtype=float).reshape(-1)
    max_force = float(np.max(np.abs(grad))) if grad.size else 0.0
    max_disp = float(np.max(np.abs(disp))) if disp.size else 0.0
    if settings.convergence_profile == "geometry_seed":
        return {
            "energy": abs(float(energy_change)) <= GEOMETRY_SEED_ENERGY_TOLERANCE_HARTREE,
            "gradient_norm": float(np.linalg.norm(grad))
            <= GEOMETRY_SEED_GRADIENT_NORM_TOLERANCE_HARTREE_PER_BOHR,
            "nonincreasing_energy": float(energy_change)
            <= GEOMETRY_SEED_FINAL_ENERGY_RISE_TOLERANCE_HARTREE,
        }
    if settings.convergence_profile == "pyfock_ase":
        # Match PyFockCalculator + ASE exactly: ASE stops on maximum force
        # only. Energy and displacement are diagnostics, not extra gates.
        force_satisfied = max_force <= PYFOCK_ASE_FMAX_HARTREE_PER_BOHR
        return {
            "energy": True,
            "max_force": force_satisfied,
            "rms_force": force_satisfied,
            "max_displacement": True,
            "rms_displacement": True,
        }
    return {
        # GDV GrdOpt/CONVEF applies the four force/displacement tests for a
        # transition state; |dE| is printed elsewhere but is not a stopping
        # condition. Preserve the energy criterion for LINK minimum searches.
        "energy": (
            True
            if settings.stationary_point == "transition_state"
            else abs(float(energy_change)) <= settings.energy_tolerance
        ),
        "max_force": max_force <= settings.max_force_tolerance,
        "rms_force": rms(grad) <= settings.rms_force_tolerance,
        "max_displacement": max_disp <= settings.max_displacement_tolerance,
        "rms_displacement": rms(disp) <= settings.rms_displacement_tolerance,
    }


def convergence_force_satisfied(convergence: Mapping[str, bool]) -> bool:
    """Return the active profile's stationary-force condition."""

    if "gradient_norm" in convergence:
        return bool(convergence["gradient_norm"])
    return bool(convergence.get("max_force") and convergence.get("rms_force"))


def gdv_transition_state_prospective_convergence(
    settings: OptimizerConvergenceSettings,
    internal_gradient: np.ndarray,
    proposed_cartesian_step_bohr: np.ndarray,
) -> dict[str, bool] | None:
    """Replicate the GrdOpt/CONVEF pre-RedCar TS stopping test."""

    if settings.stationary_point != "transition_state":
        return None
    return gaussian_like_convergence(
        settings,
        0.0,
        np.asarray(internal_gradient, dtype=float),
        np.asarray(proposed_cartesian_step_bohr, dtype=float),
    )


def gaussian_minimum_prospective_convergence(
    settings: OptimizerConvergenceSettings,
    internal_gradient: np.ndarray,
    proposed_cartesian_step_bohr: np.ndarray,
) -> dict[str, bool] | None:
    """Berny minimum stopping check before evaluating the proposed point."""
    if settings.stationary_point != "minimum":
        return None
    if settings.convergence_profile == "geometry_seed":
        # The xTB-normal-equivalent contract contains an evaluated energy
        # change and a non-increasing-energy gate.  Neither quantity exists
        # for an unevaluated Berny trial, so this role must always evaluate
        # the proposed point before deciding convergence.
        return None
    gradient = np.asarray(internal_gradient, dtype=float).reshape(-1)
    step = np.asarray(proposed_cartesian_step_bohr, dtype=float).reshape(-1)
    if not (gradient.size and step.size and np.all(np.isfinite(gradient)) and np.all(np.isfinite(step))):
        return None
    return {
        "energy": True,
        "max_force": float(np.max(np.abs(gradient))) <= settings.max_force_tolerance,
        "rms_force": rms(gradient) <= settings.rms_force_tolerance,
        "max_displacement": float(np.max(np.abs(step))) <= settings.max_displacement_tolerance,
        "rms_displacement": rms(step) <= settings.rms_displacement_tolerance,
    }


def convergence_gradient(
    evaluation: OptimizerEvaluationLike,
    internal_gradient: np.ndarray,
    settings: OptimizerConvergenceSettings,
    service: GeometryEvaluationServiceLike,
) -> np.ndarray:
    """Select the force covector used by the active convergence profile."""

    if settings.stationary_point == "transition_state" and service.coordinate_model.kind == "sonic":
        # GDV GrdOpt/CONVEF tests the force vector F in the active internal
        # variables. Re-projecting that covector into Cartesian space changes
        # the max-force test near its threshold.
        return np.asarray(internal_gradient, dtype=float).reshape(-1)
    if service.coordinate_model.kind in {
        "sonic",
        "typed_onic",
        "sycart",
        "simple_redundant",
        "simple_delocalized",
        "hessian_modes",
    } and settings.freeze_inactive_sonic:
        cartesian = evaluation.gradient_hartree_per_bohr
        if cartesian is None:
            # Coordinate finite differences return dE/dq. Map it to the
            # unique minimum-norm Cartesian tangent covector before applying
            # a criterion expressed in Eh/bohr.
            mapper = getattr(service, "cartesian_covector_from_optimizer", None)
            if callable(mapper):
                return np.asarray(
                    mapper(internal_gradient, evaluation.coordinates_angstrom),
                    dtype=float,
                )
            directions_bohr = (
                np.asarray(
                    service.coordinate_directions(evaluation.coordinates_angstrom),
                    dtype=float,
                )
                * ANGSTROM_TO_BOHR
            )
            return np.linalg.pinv(directions_bohr, rcond=1.0e-8) @ np.asarray(
                internal_gradient, dtype=float
            ).reshape(-1)
        projector = getattr(service, "project_cartesian_to_active_tangent", None)
        if callable(projector):
            return np.asarray(
                projector(cartesian, evaluation.coordinates_angstrom),
                dtype=float,
            )
        directions = service.coordinate_directions(evaluation.coordinates_angstrom).T
        tangent_projector = directions @ np.linalg.pinv(directions, rcond=1.0e-8)
        return tangent_projector @ np.asarray(cartesian, dtype=float).reshape(-1)
    cartesian = evaluation.gradient_hartree_per_bohr
    if cartesian is not None:
        return np.asarray(cartesian, dtype=float).reshape(-1)
    return np.asarray(internal_gradient, dtype=float).reshape(-1)
