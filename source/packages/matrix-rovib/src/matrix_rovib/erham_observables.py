"""Intensities, predictive uncertainties and QM predicates for ERHAM models."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping, Sequence

import numpy as np

from .erham import ERHAMModel, solve_erham_block
from .erham_fit import (
    ERHAMMotionVariable,
    ERHAMObservation,
    erham_model_coefficients,
    erham_motion_jacobian,
    evaluate_erham_observations,
)
from .native_rotor import MHZ_TO_KELVIN, dipole_transition_matrix


@dataclass(frozen=True)
class ERHAMIntensityModel:
    temperature_K: float
    dipole_debye: tuple[float, float, float]
    species_weights: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if self.temperature_K <= 0.0 or not math.isfinite(self.temperature_K):
            raise ValueError("ERHAM intensity temperature must be positive and finite")
        if len(self.dipole_debye) != 3 or not all(
            math.isfinite(value) for value in self.dipole_debye
        ):
            raise ValueError("ERHAM intensities require three finite dipole components")
        if self.species_weights is not None and any(
            value <= 0.0 or not math.isfinite(value)
            for value in self.species_weights.values()
        ):
            raise ValueError("ERHAM species weights must be positive and finite")


def _erham_line_strength_data(
    model: ERHAMModel,
    observations: tuple[ERHAMObservation, ...],
    intensity: ERHAMIntensityModel,
) -> tuple[np.ndarray, dict, dict]:
    keys = {(level[0], row.species) for row in observations for level in (row.upper, row.lower)}
    blocks = {key: solve_erham_block(model, *key) for key in keys}
    lookup = {
        key: {(Ka, Kc): index for index, (Ka, Kc, _parity) in enumerate(block.assignments)}
        for key, block in blocks.items()
    }
    result = np.empty(len(observations), dtype=float)
    for row_index, row in enumerate(observations):
        Ju, Kau, Kcu = row.upper
        Jl, Kal, Kcl = row.lower
        upper_key, lower_key = (Ju, row.species), (Jl, row.species)
        upper, lower = blocks[upper_key], blocks[lower_key]
        iu = lookup[upper_key][(Kau, Kcu)]
        il = lookup[lower_key][(Kal, Kcl)]
        line_strength = 0.0
        for axis, dipole in zip(("a", "b", "c"), intensity.dipole_debye, strict=True):
            if dipole:
                raw = dipole_transition_matrix(Jl, Ju, axis, model.representation)
                amplitude = np.vdot(lower.vectors[:, il], raw @ upper.vectors[:, iu])
                line_strength += dipole * dipole * abs(amplitude) ** 2
        species_weight = 1.0
        if intensity.species_weights is not None:
            species_weight = float(intensity.species_weights.get(row.species, 1.0))
        result[row_index] = max(
            np.finfo(float).tiny,
            species_weight * line_strength,
        )
    return result, blocks, lookup


def erham_transition_line_strengths(
    model: ERHAMModel,
    observations: Sequence[ERHAMObservation],
    intensity: ERHAMIntensityModel,
) -> np.ndarray:
    """Return temperature-independent assigned line strengths in D squared."""
    strengths, _blocks, _lookup = _erham_line_strength_data(
        model, tuple(observations), intensity
    )
    return strengths


def erham_transition_intensities(
    model: ERHAMModel,
    observations: Sequence[ERHAMObservation],
    intensity: ERHAMIntensityModel,
) -> np.ndarray:
    """Return population, stimulated-emission and frequency weighted strengths."""
    observations = tuple(observations)
    strengths, blocks, lookup = _erham_line_strength_data(
        model, observations, intensity
    )
    beta = MHZ_TO_KELVIN / intensity.temperature_K
    ground = min(float(np.min(block.energies_MHz)) for block in blocks.values())
    result = np.empty(len(observations), dtype=float)
    for row_index, (row, strength) in enumerate(zip(observations, strengths, strict=True)):
        Ju, Kau, Kcu = row.upper
        Jl, Kal, Kcl = row.lower
        upper, lower = blocks[(Ju, row.species)], blocks[(Jl, row.species)]
        iu = lookup[(Ju, row.species)][(Kau, Kcu)]
        il = lookup[(Jl, row.species)][(Kal, Kcl)]
        frequency = abs(float(upper.energies_MHz[iu] - lower.energies_MHz[il]))
        population = math.exp(-(float(lower.energies_MHz[il]) - ground) * beta)
        stimulated = -math.expm1(-frequency * beta)
        result[row_index] = max(
            np.finfo(float).tiny,
            population * stimulated * frequency * strength,
        )
    return result


def with_erham_intensity_weights(
    model: ERHAMModel,
    observations: Sequence[ERHAMObservation],
    intensity: ERHAMIntensityModel,
) -> tuple[ERHAMObservation, ...]:
    strengths = erham_transition_intensities(model, observations, intensity)
    return tuple(
        replace(row, blend_weight=float(weight))
        for row, weight in zip(observations, strengths, strict=True)
    )


def erham_predictive_uncertainties(
    model: ERHAMModel,
    observations: Sequence[ERHAMObservation],
    parameter_names: Sequence[str],
    covariance: np.ndarray,
    *,
    motion_variables: Mapping[str, ERHAMMotionVariable] | None = None,
    workers: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate a fitted covariance to assigned frequencies."""
    names = tuple(parameter_names)
    matrix = np.asarray(covariance, dtype=float)
    if matrix.shape != (len(names), len(names)) or not np.allclose(matrix, matrix.T):
        raise ValueError("ERHAM covariance must be symmetric and match the parameter list")
    available = erham_model_coefficients(model)
    motions = {} if motion_variables is None else dict(motion_variables)
    unknown = set(names) - set(available) - set(motions)
    if unknown:
        raise ValueError(f"unknown ERHAM uncertainty parameters: {sorted(unknown)}")
    linear_names = tuple(name for name in names if name in available)
    motion_names = tuple(name for name in names if name in motions)
    calculated, linear = evaluate_erham_observations(
        model, observations, parameter_names=linear_names, workers=workers
    )
    nonlinear = erham_motion_jacobian(
        model, observations, tuple(motions[name] for name in motion_names), workers=workers
    )
    columns = {
        **{name: linear[:, index] for index, name in enumerate(linear_names)},
        **{name: nonlinear[:, index] for index, name in enumerate(motion_names)},
    }
    jacobian = np.column_stack([columns[name] for name in names])
    variance = np.einsum("ij,jk,ik->i", jacobian, matrix, jacobian)
    return calculated, np.sqrt(np.maximum(variance, 0.0))


def erham_qm_predicates(
    seed: Mapping[str, object], available_parameters: Sequence[str]
) -> dict[str, tuple[float, float]]:
    """Map generic QM-seed paths onto ERHAM coefficient and motion names."""
    aliases = {
        "rotational.A_MHz": "A", "rotational.B_MHz": "B", "rotational.C_MHz": "C",
        "torsion.rho": "rho", "torsion.beta_rad": "beta_rad", "torsion.alpha_rad": "alpha_rad",
    }
    available = set(available_parameters)
    result: dict[str, tuple[float, float]] = {}
    for item in seed.get("parameters", ()):  # type: ignore[union-attr]
        if not isinstance(item, Mapping) or not bool(item.get("use_as_prior", True)):
            continue
        path = str(item.get("path", ""))
        name = aliases.get(path, path.rsplit(".", 1)[-1])
        if name in available:
            result[name] = (float(item["value"]), float(item["sigma"]))
    # Backward-compatible ingestion of nano-MATRIX QM seeds predating the
    # canonical path list.  The mapping remains schema based and molecule free.
    rotational = seed.get("rotational_MHz", {})
    priors = seed.get("priors", {})
    if isinstance(rotational, Mapping):
        relative = float(priors.get("rotational_scale_sigma", 0.02)) if isinstance(priors, Mapping) else 0.02
        for name in ("A", "B", "C"):
            if name in available and name in rotational:
                value = float(rotational[name])
                result.setdefault(name, (value, abs(value) * relative))
    torsion = seed.get("torsion", {})
    if isinstance(torsion, Mapping):
        relative = float(priors.get("rho_scale_sigma", 0.15)) if isinstance(priors, Mapping) else 0.15
        if "rho" in available and "rho" in torsion:
            value = float(torsion["rho"])
            result.setdefault("rho", (value, max(abs(value) * relative, 1.0e-8)))
        if "beta_rad" in available:
            if "beta_rad" in torsion:
                value = float(torsion["beta_rad"])
                result.setdefault("beta_rad", (value, math.radians(5.0)))
            elif "axis_angle_degree" in torsion:
                value = math.radians(float(torsion["axis_angle_degree"]))
                result.setdefault("beta_rad", (value, math.radians(5.0)))
    return result


__all__ = [
    "ERHAMIntensityModel", "erham_predictive_uncertainties", "erham_qm_predicates",
    "erham_transition_intensities", "erham_transition_line_strengths",
    "with_erham_intensity_weights",
]
