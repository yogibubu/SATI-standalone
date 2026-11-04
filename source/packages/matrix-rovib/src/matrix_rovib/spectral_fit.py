"""Weighted nonlinear least-squares fitting of assigned LOCK spectra.

The fitter deliberately works with assigned energy levels rather than nearest
frequency matching.  This makes the objective continuous across iterations and
supports allowed, forbidden, and blended lines with the same representation.
Experimental standard uncertainties define the weights.  Optional Gaussian
priors can carry theoretical information without hiding the rank supplied by
the measured spectrum itself.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from matrix_numerics.least_squares import svd_covariance

from .model_selection import linearized_nested_model_selection
from .qm_parameter_graph import build_qm_parameter_graph
from .uncertainty import correlated_linearized_uncertainty

from .native_rotor import (
    NativeRotorParameters,
    angular_momentum_operators,
    centrifugal_distortion_operators,
    diagonalize_rotor,
)
from .spin_hyperfine import (
    NuclearSpin,
    NuclearSpinSpinCoupling,
    SpinHyperfineLevel,
    SpinHyperfineModel,
    _assemble_f_block_hamiltonian,
    _basis_group_slices,
    hellmann_feynman_diagonal,
    solve_spin_hyperfine_rotor,
)


SPECTRAL_FIT_SCHEMA = "matrix.lock.spectral_fit.v1"
SPECTRAL_FIT_RESULT_SCHEMA = "matrix.lock.spectral_fit_result.v1"
_TENSOR_COMPONENTS = {
    name: index for index, name in enumerate(("aa", "bb", "cc", "ab", "ac", "bc"))
}


@dataclass(frozen=True)
class LevelSelector:
    N: int
    two_J: int
    two_F: int
    Ka: int
    Kc: int
    nuclear_path: tuple[int, ...] | None = None
    lam_parent: int | None = None
    torsional_state: int | None = None
    lam_symmetry: str | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LevelSelector":
        required = ("N", "two_J", "two_F", "Ka", "Kc")
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(f"level selector is missing {', '.join(missing)}")
        path = payload.get("nuclear_path")
        return cls(
            **{name: int(payload[name]) for name in required},
            nuclear_path=None if path is None else tuple(int(value) for value in path),
            lam_parent=_optional_int(payload.get("lam_parent")),
            torsional_state=_optional_int(payload.get("torsional_state")),
            lam_symmetry=None
            if payload.get("lam_symmetry") is None
            else str(payload["lam_symmetry"]),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "N": self.N,
            "two_J": self.two_J,
            "two_F": self.two_F,
            "Ka": self.Ka,
            "Kc": self.Kc,
        }
        for name in ("nuclear_path", "lam_parent", "torsional_state", "lam_symmetry"):
            value = getattr(self, name)
            if value is not None:
                result[name] = list(value) if name == "nuclear_path" else value
        return result

    def matches(self, level: SpinHyperfineLevel) -> bool:
        for name in ("N", "two_J", "two_F", "Ka", "Kc"):
            if getattr(level, name) != getattr(self, name):
                return False
        for name in ("nuclear_path", "lam_parent", "torsional_state", "lam_symmetry"):
            selected = getattr(self, name)
            if selected is not None and getattr(level, name) != selected:
                return False
        return True


@dataclass(frozen=True)
class TransitionComponent:
    lower: LevelSelector
    upper: LevelSelector
    weight: float = 1.0

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TransitionComponent":
        weight = float(payload.get("weight", 1.0))
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("blend component weights must be positive and finite")
        return cls(
            LevelSelector.from_dict(_mapping(payload.get("lower"), "lower")),
            LevelSelector.from_dict(_mapping(payload.get("upper"), "upper")),
            weight,
        )


@dataclass(frozen=True)
class SpectralObservation:
    frequency_MHz: float
    uncertainty_MHz: float
    components: tuple[TransitionComponent, ...]
    label: str = ""
    weight: float = 1.0
    state: LevelSelector | None = None
    dataset: str = "default"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], index: int) -> "SpectralObservation":
        is_term_value = "energy_MHz" in payload or "state" in payload
        if is_term_value:
            if "energy_MHz" not in payload or "state" not in payload:
                raise ValueError("a term value requires energy_MHz and state")
            components: tuple[TransitionComponent, ...] = ()
            state = LevelSelector.from_dict(_mapping(payload["state"], "state"))
        else:
            state = None
        if ("components" in payload) == ("lower" in payload or "upper" in payload):
            if not is_term_value:
                raise ValueError("an observation requires either lower/upper or components")
        if is_term_value:
            pass
        elif "components" in payload:
            raw = payload["components"]
            if not isinstance(raw, list) or not raw:
                raise ValueError("components must be a nonempty array")
            components = tuple(
                TransitionComponent.from_dict(_mapping(item, "component")) for item in raw
            )
        else:
            components = (TransitionComponent.from_dict(payload),)
        frequency = float(payload["energy_MHz"] if is_term_value else payload["frequency_MHz"])
        uncertainty = float(payload["uncertainty_MHz"])
        weight = float(payload.get("weight", 1.0))
        if not math.isfinite(frequency):
            raise ValueError("observed frequencies must be finite")
        if not math.isfinite(uncertainty) or uncertainty <= 0.0:
            raise ValueError("experimental uncertainties must be positive and finite")
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("observation weights must be positive and finite")
        return cls(
            frequency,
            uncertainty,
            components,
            str(payload.get("label", f"line-{index + 1}")),
            weight,
            state,
            str(payload.get("dataset", "default")),
        )


@dataclass(frozen=True)
class ParameterTarget:
    path: str
    factor: float = 1.0
    offset: float = 0.0


@dataclass(frozen=True)
class FitVariable:
    name: str
    targets: tuple[ParameterTarget, ...]
    initial: float
    scale: float
    lower: float = -math.inf
    upper: float = math.inf
    prior_value: float | None = None
    prior_sigma: float | None = None

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        parameters: NativeRotorParameters,
        model: SpinHyperfineModel,
        index: int,
    ) -> "FitVariable":
        if ("targets" in payload) == ("path" in payload):
            raise ValueError("a variable requires either path or targets")
        if "targets" in payload:
            raw_targets = payload["targets"]
            if not isinstance(raw_targets, list) or not raw_targets:
                raise ValueError("variable targets must be a nonempty array")
            targets = tuple(
                ParameterTarget(
                    path=str(_mapping(item, "target")["path"]),
                    factor=float(_mapping(item, "target").get("factor", 1.0)),
                    offset=float(_mapping(item, "target").get("offset", 0.0)),
                )
                for item in raw_targets
            )
        else:
            targets = (ParameterTarget(str(payload["path"])),)
        for target in targets:
            _get_parameter(parameters, model, target.path)
            if (
                not all(math.isfinite(value) for value in (target.factor, target.offset))
                or target.factor == 0.0
            ):
                raise ValueError("target factors must be nonzero and finite")
        if "initial" in payload:
            initial = float(payload["initial"])
        elif len(targets) == 1:
            target = targets[0]
            initial = (
                _get_parameter(parameters, model, target.path) - target.offset
            ) / target.factor
        else:
            raise ValueError("a tied variable requires an explicit initial value")
        scale = float(payload.get("scale", max(abs(initial), 1.0)))
        lower, upper = float(payload.get("lower", -math.inf)), float(payload.get("upper", math.inf))
        for target in targets:
            if target.path in {"rotor.A_MHz", "rotor.B_MHz", "rotor.C_MHz"}:
                zero = -target.offset / target.factor
                if target.factor > 0.0:
                    lower = max(lower, math.nextafter(zero, math.inf))
                else:
                    upper = min(upper, math.nextafter(zero, -math.inf))
        prior = payload.get("prior")
        prior_value = prior_sigma = None
        if prior is not None:
            prior_map = _mapping(prior, "prior")
            prior_value, prior_sigma = float(prior_map["value"]), float(prior_map["sigma"])
            if (
                not math.isfinite(prior_value)
                or not math.isfinite(prior_sigma)
                or prior_sigma <= 0.0
            ):
                raise ValueError("a Gaussian prior requires a finite value and positive sigma")
        if (
            not math.isfinite(initial)
            or not math.isfinite(scale)
            or scale <= 0.0
            or math.isnan(lower)
            or math.isnan(upper)
            or lower >= upper
        ):
            raise ValueError("invalid variable initial value, scale, or bounds")
        if not lower <= initial <= upper:
            raise ValueError("variable initial value lies outside its bounds")
        name = str(payload.get("name", f"p{index + 1}")).strip()
        if not name:
            raise ValueError("fit variable names must not be empty")
        return cls(
            name,
            targets,
            initial,
            scale,
            lower,
            upper,
            prior_value,
            prior_sigma,
        )


@dataclass(frozen=True)
class SpectralFitResult:
    payload: Mapping[str, Any]
    parameters: NativeRotorParameters
    model: SpinHyperfineModel

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


def _selection_group(variable: FitVariable) -> str:
    """Map fit paths to molecule-independent Hamiltonian families."""

    families = []
    for target in variable.targets:
        parts = target.path.split(".")
        if parts[0] == "rotor" and len(parts) >= 2:
            name = parts[1]
            family = {
                "A_MHz": "rigid_rotation",
                "B_MHz": "rigid_rotation",
                "C_MHz": "rigid_rotation",
                "quartic_MHz": "quartic_distortion",
                "sextic_MHz": "sextic_distortion",
                "octic_MHz": "octic_distortion",
            }.get(name, f"rotor:{name}")
        elif parts[0] == "electron" and len(parts) >= 2:
            family = f"electron:{parts[1]}"
        elif parts[0].startswith("nucleus:") and len(parts) >= 2:
            family = f"nuclear:{parts[1]}"
        elif parts[0].startswith("pair:") and len(parts) >= 2:
            family = f"nuclear_pair:{parts[1]}"
        else:
            family = parts[0]
        families.append(family)
    return "+".join(dict.fromkeys(families))


def fit_assigned_spectrum(
    parameters: NativeRotorParameters,
    model: SpinHyperfineModel,
    request: Mapping[str, Any],
) -> SpectralFitResult:
    """Fit a rigid-rotor fine/hyperfine model to assigned line frequencies."""
    if request.get("schema") != SPECTRAL_FIT_SCHEMA:
        raise ValueError(f"unsupported spectral-fit schema: {request.get('schema')!r}")
    raw_n_values = tuple(request.get("n_values", ()))
    if any(isinstance(value, bool) or int(value) != value or value < 0 for value in raw_n_values):
        raise ValueError("n_values must contain nonnegative integers")
    n_values = tuple(sorted({int(value) for value in raw_n_values}))
    if not n_values:
        raise ValueError("n_values must contain nonnegative integers")
    raw_variables = request.get("variables")
    raw_observations = request.get("observations")
    if not isinstance(raw_variables, list) or not raw_variables:
        raise ValueError("variables must be a nonempty array")
    if not isinstance(raw_observations, list) or not raw_observations:
        raise ValueError("observations must be a nonempty array")
    variables = tuple(
        FitVariable.from_dict(_mapping(item, "variable"), parameters, model, index)
        for index, item in enumerate(raw_variables)
    )
    observations = tuple(
        SpectralObservation.from_dict(_mapping(item, "observation"), index)
        for index, item in enumerate(raw_observations)
    )
    names = tuple(variable.name for variable in variables)
    if len(names) != len(set(names)):
        raise ValueError("fit variable names must be unique")
    paths = [target.path for variable in variables for target in variable.targets]
    if len(paths) != len(set(paths)):
        raise ValueError("each parameter path may be controlled by only one fit variable")

    options = _mapping(request.get("options", {}), "options")
    loss = str(options.get("loss", "linear"))
    if loss not in {"linear", "soft_l1", "huber"}:
        raise ValueError("loss must be linear, soft_l1, or huber")
    f_scale = float(options.get("f_scale", 1.0))
    rank_rcond = float(options.get("rank_rcond", 1.0e-10))
    max_nfev = int(options.get("max_nfev", 500))
    outlier_threshold = float(options.get("outlier_threshold", 4.0))
    if not math.isfinite(f_scale) or f_scale <= 0.0:
        raise ValueError("f_scale must be positive and finite")
    if not math.isfinite(rank_rcond) or rank_rcond <= 0.0:
        raise ValueError("rank_rcond must be positive and finite")
    if max_nfev <= 0:
        raise ValueError("max_nfev must be positive")
    if not math.isfinite(outlier_threshold) or outlier_threshold <= 0.0:
        raise ValueError("outlier_threshold must be positive and finite")
    assembly_backend = options.get("assembly_backend", "compiled")
    diagonalizer_backend = options.get("diagonalizer_backend")
    requested_jacobian = str(options.get("jacobian", "auto"))
    if requested_jacobian not in {"auto", "hellmann_feynman", "three_point"}:
        raise ValueError("jacobian must be auto, hellmann_feynman, or three_point")
    tracking = str(options.get("tracking", "overlap"))
    if tracking not in {"labels", "overlap"}:
        raise ValueError("tracking must be labels or overlap")

    x0 = np.asarray([item.initial for item in variables], dtype=float)
    scales = np.asarray([item.scale for item in variables], dtype=float)
    lower = np.asarray([item.lower for item in variables], dtype=float)
    upper = np.asarray([item.upper for item in variables], dtype=float)
    prior_rows = tuple(
        index for index, item in enumerate(variables) if item.prior_sigma is not None
    )
    cache: dict[
        bytes,
        tuple[
            np.ndarray,
            NativeRotorParameters,
            SpinHyperfineModel,
            Any,
            Mapping[LevelSelector, tuple[int, int]],
        ],
    ] = {}
    reference_parameters, reference_model = _apply_variables(parameters, model, variables, x0)
    reference_solution = solve_spin_hyperfine_rotor(
        reference_parameters,
        reference_model,
        n_values=n_values,
        assembly_backend=assembly_backend,
        diagonalizer_backend=diagonalizer_backend,
    )
    reference_locations = _selector_locations(observations, reference_solution)
    closed_shell_hf = _supports_closed_shell_hf(model, variables) and tracking == "labels"
    use_hf = requested_jacobian != "three_point"

    def predict(x: np.ndarray):
        key = np.asarray(x, dtype=np.float64).tobytes()
        cached = cache.get(key)
        if cached is not None:
            return cached
        fitted_parameters, fitted_model = _apply_variables(parameters, model, variables, x)
        solution = solve_spin_hyperfine_rotor(
            fitted_parameters,
            fitted_model,
            n_values=n_values,
            assembly_backend=assembly_backend,
            diagonalizer_backend=diagonalizer_backend,
        )
        if tracking == "overlap":
            mappings = _spin_overlap_mappings(reference_solution, solution)
            locations = {
                selector: (two_F, mappings[two_F][index])
                for selector, (two_F, index) in reference_locations.items()
            }
            energies = np.asarray(
                [
                    _observation_value_at_locations(observation, solution, locations)
                    for observation in observations
                ]
            )
        else:
            levels = tuple(level for block in solution.blocks for level in block.levels)
            locations = _selector_locations(observations, solution)
            energies = np.asarray(
                [_observation_value(observation, levels) for observation in observations]
            )
        cached = (energies, fitted_parameters, fitted_model, solution, locations)
        cache[key] = cached
        return cached

    sigma = np.asarray(
        [
            observation.uncertainty_MHz / math.sqrt(observation.weight)
            for observation in observations
        ]
    )
    observed = np.asarray([observation.frequency_MHz for observation in observations])

    def data_residual(x: np.ndarray) -> np.ndarray:
        return (predict(x)[0] - observed) / sigma

    def residual(x: np.ndarray) -> np.ndarray:
        data = data_residual(x)
        if not prior_rows:
            return data
        priors = np.asarray(
            [
                (x[index] - float(variables[index].prior_value))
                / float(variables[index].prior_sigma)
                for index in prior_rows
            ]
        )
        return np.concatenate((data, priors))

    def analytic_jacobian(x: np.ndarray) -> np.ndarray:
        data = (
            _closed_shell_hf_jacobian(parameters, model, variables, x, observations, sigma)
            if closed_shell_hf
            else _spin_hyperfine_hf_jacobian(
                parameters,
                model,
                variables,
                x,
                observations,
                sigma,
                n_values,
                assembly_backend,
                diagonalizer_backend,
                solution=predict(x)[3],
                locations=predict(x)[4],
            )
        )
        if not prior_rows:
            return data
        prior_jacobian = np.zeros((len(prior_rows), len(variables)))
        for row, column in enumerate(prior_rows):
            prior_jacobian[row, column] = 1.0 / float(variables[column].prior_sigma)
        return np.vstack((data, prior_jacobian))

    from scipy.optimize import least_squares

    optimized = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        method="trf",
        jac=analytic_jacobian if use_hf else "3-point",
        x_scale=scales,
        loss=loss,
        f_scale=f_scale,
        max_nfev=max_nfev,
    )
    fitted_values = np.asarray(optimized.x, dtype=float)
    calculated, fitted_parameters, fitted_model, _solution, _locations = predict(fitted_values)
    raw_residual_MHz = calculated - observed
    weighted = raw_residual_MHz / sigma
    if use_hf:
        augmented_jacobian = analytic_jacobian(fitted_values)
        data_jacobian = augmented_jacobian[: len(observations)]
    else:
        data_jacobian = _finite_difference_jacobian(
            data_residual, fitted_values, scales, lower, upper
        )
        augmented_jacobian = _finite_difference_jacobian(
            residual, fitted_values, scales, lower, upper
        )
    data_diag = _svd_diagnostics(data_jacobian, scales, rank_rcond)
    augmented_diag = _svd_diagnostics(augmented_jacobian, scales, rank_rcond)
    covariance, _covariance_rank, _covariance_singular = svd_covariance(augmented_jacobian, scales=scales, rcond=rank_rcond)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    correlation = _correlation(covariance, standard_errors)
    prediction_gradients = data_jacobian * sigma[:, None]
    prediction_variances = np.einsum(
        "ij,jk,ik->i", prediction_gradients, covariance, prediction_gradients
    )
    prediction_uncertainties = np.sqrt(np.maximum(prediction_variances, 0.0))
    chi_square = float(weighted @ weighted)
    dof = len(observations) - data_diag["rank"]
    reduced_chi_square = None if dof <= 0 else chi_square / dof
    identifiable_data = data_diag["rank"] == len(variables)
    identifiable_augmented = augmented_diag["rank"] == len(variables)
    status = (
        "converged"
        if optimized.success and identifiable_augmented
        else "rank_deficient"
        if optimized.success
        else "failed"
    )
    active_mask = np.asarray(optimized.active_mask, dtype=int)

    result_variables = []
    for index, variable in enumerate(variables):
        entry: dict[str, Any] = {
            "name": variable.name,
            "value": float(fitted_values[index]),
            "standard_error": (float(standard_errors[index]) if identifiable_augmented else None),
            "initial": variable.initial,
            "scale": variable.scale,
            "lower": variable.lower if math.isfinite(variable.lower) else None,
            "upper": variable.upper if math.isfinite(variable.upper) else None,
            "active_bound": {0: None, -1: "lower", 1: "upper"}[int(active_mask[index])],
            "targets": [target.__dict__ for target in variable.targets],
        }
        if variable.prior_sigma is not None:
            entry["prior"] = {"value": variable.prior_value, "sigma": variable.prior_sigma}
        result_variables.append(entry)
    comparisons = [
        {
            "label": observation.label,
            "dataset": observation.dataset,
            "kind": "term_value" if observation.state is not None else "transition",
            "observed_MHz": observation.frequency_MHz,
            "uncertainty_MHz": observation.uncertainty_MHz,
            "weight": observation.weight,
            "calculated_MHz": float(calculated[index]),
            "calculated_standard_uncertainty_MHz": (
                float(prediction_uncertainties[index]) if identifiable_augmented else None
            ),
            "calculated_minus_observed_MHz": float(raw_residual_MHz[index]),
            "standardized_residual": float(weighted[index]),
            "outlier": bool(abs(weighted[index]) > outlier_threshold),
            "blend_components": len(observation.components),
        }
        for index, observation in enumerate(observations)
    ]
    high_correlations = [
        {"left": names[i], "right": names[j], "correlation": float(correlation[i, j])}
        for i in range(len(names))
        for j in range(i + 1, len(names))
        if abs(correlation[i, j]) >= 0.95
    ]
    data_uncertainty = correlated_linearized_uncertainty(
        data_jacobian, rcond=rank_rcond,
    )
    automatic_selection = linearized_nested_model_selection(
        names,
        tuple(_selection_group(variable) for variable in variables),
        x0,
        fitted_values,
        weighted,
        data_jacobian,
        validation_fraction=float(options.get("selection_validation_fraction", 0.2)),
        rank_rcond=rank_rcond,
        correlation_limit=float(options.get("selection_correlation_limit", 0.98)),
    )
    qm_provenance = options.get("qm_parameter_provenance", {})
    qm_parameter_graph = None
    if prior_rows:
        qm_parameters = []
        for index in prior_rows:
            variable = variables[index]
            provenance = (
                qm_provenance.get(variable.name, {})
                if isinstance(qm_provenance, Mapping) else {}
            )
            if not isinstance(provenance, Mapping):
                provenance = {}
            qm_parameters.append({
                "path": variable.name,
                "value": float(variable.prior_value),
                "sigma": float(variable.prior_sigma),
                "source_quantity": provenance.get(
                    "source_quantity", "declared quantum-chemical predicate"
                ),
                "source_artifact": provenance.get("source_artifact"),
                "transformation": provenance.get("transformation", "declared mapping"),
                "operator_family": provenance.get(
                    "operator_family", _selection_group(variable)
                ),
                "unit": provenance.get("unit", "MHz"),
                "interpretation": provenance.get("interpretation", ""),
            })
        qm_parameter_graph = build_qm_parameter_graph(
            qm_parameters,
            [
                {
                    "path": variable.name,
                    "value": float(fitted_values[index]),
                    "standard_error": (
                        float(standard_errors[index]) if identifiable_augmented else None
                    ),
                    "status": "refined",
                }
                for index, variable in enumerate(variables)
            ],
            experimental_jacobian=data_jacobian,
            augmented_jacobian=augmented_jacobian,
            rank_rcond=rank_rcond,
        )
    payload: dict[str, Any] = {
        "schema": SPECTRAL_FIT_RESULT_SCHEMA,
        "status": status,
        "success": bool(optimized.success and identifiable_augmented),
        "optimizer": {
            "method": "scipy.optimize.least_squares/trf",
            "jacobian": "compiled Hellmann-Feynman"
            if use_hf
            else "three-point eigenvalue differences",
            "loss": loss,
            "message": str(optimized.message),
            "function_evaluations": int(optimized.nfev),
            "jacobian_evaluations": None if optimized.njev is None else int(optimized.njev),
            "hamiltonian_evaluations": len(cache),
            "state_tracking": tracking,
        },
        "weighting": "(calculated-observed)/experimental_sigma times sqrt(line_weight)",
        "fixed_parameter_policy": "parameters absent from variables retain the supplied model or theoretical value",
        "prior_policy": "Gaussian priors are appended as pseudo-observations; data and augmented ranks are reported separately",
        "variables": result_variables,
        "comparisons": comparisons,
        "metrics": {
            "observation_count": len(observations),
            "free_parameter_count": len(variables),
            "prior_count": len(prior_rows),
            "chi_square": chi_square,
            "degrees_of_freedom": dof,
            "reduced_chi_square": reduced_chi_square,
            "weighted_rms": float(np.sqrt(np.mean(weighted * weighted))),
            "rms_residual_MHz": float(np.sqrt(np.mean(raw_residual_MHz * raw_residual_MHz))),
            "maximum_absolute_residual_MHz": float(np.max(np.abs(raw_residual_MHz))),
            "outlier_threshold": outlier_threshold,
            "outlier_count": int(np.count_nonzero(np.abs(weighted) > outlier_threshold)),
        },
        "identifiability": {
            "identifiable_from_data": identifiable_data,
            "identifiable_with_priors": identifiable_augmented,
            "data": data_diag,
            "augmented": augmented_diag,
            "high_correlations": high_correlations,
        },
        "automatic_model_selection": automatic_selection,
        "qm_parameter_graph": qm_parameter_graph,
        "uncertainty_analysis": {
            "method": "generalized least squares in the uncertainty-whitened Jacobian",
            "experimental_only": {
                "rank": data_uncertainty.rank,
                "condition_number": data_uncertainty.condition_number,
                "standard_errors": list(data_uncertainty.standard_errors),
                "correlation": data_uncertainty.correlation.tolist(),
            },
            "with_predicates": {
                "rank": int(augmented_diag["rank"]),
                "condition_number": augmented_diag.get("condition_number_scaled"),
                "standard_errors": standard_errors.tolist(),
                "correlation": correlation.tolist(),
            },
            "interpretation": (
                "experimental-only and predicate-augmented uncertainties are reported "
                "separately; the automatic model hierarchy is advisory"
            ),
        },
        "covariance": covariance.tolist(),
        "correlation": correlation.tolist(),
        "covariance_note": (
            "local linearized covariance for absolute experimental sigmas; robust-loss covariance is approximate"
            if identifiable_augmented
            else "Moore-Penrose diagnostic only; standard errors are omitted because the augmented Jacobian is rank deficient"
        ),
        "fitted_rotor": _rotor_to_dict(fitted_parameters),
        "fitted_spin_hyperfine": fitted_model.to_dict(),
    }
    return SpectralFitResult(payload, fitted_parameters, fitted_model)


def read_spectral_fit_request(path: Path | str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("a spectral-fit request must be a JSON object")
    return payload


def write_spectral_fit_result(path: Path | str, result: SpectralFitResult) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return target


def write_spectral_fit_catalog_csv(path: Path | str, result: SpectralFitResult) -> Path:
    """Write observed/calculated lines with propagated one-sigma uncertainty."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "label",
        "dataset",
        "kind",
        "observed_MHz",
        "uncertainty_MHz",
        "calculated_MHz",
        "calculated_standard_uncertainty_MHz",
        "calculated_minus_observed_MHz",
        "standardized_residual",
        "outlier",
    )
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result.payload["comparisons"])
    return target


def write_spectral_fit_diagnostics_plot(path: Path | str, result: SpectralFitResult) -> Path:
    """Write a publication-ready observed-minus-calculated residual plot."""
    import matplotlib.pyplot as plt

    comparisons = result.payload["comparisons"]
    observed = np.asarray([item["observed_MHz"] for item in comparisons])
    residual = np.asarray([item["calculated_minus_observed_MHz"] for item in comparisons])
    uncertainty = np.asarray([item["uncertainty_MHz"] for item in comparisons])
    datasets = [str(item.get("dataset", "default")) for item in comparisons]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    for dataset in dict.fromkeys(datasets):
        selected = np.asarray([value == dataset for value in datasets])
        axis.errorbar(
            observed[selected],
            residual[selected],
            yerr=uncertainty[selected],
            marker="o",
            linestyle="none",
            capsize=2,
            label=dataset,
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Observed frequency / MHz")
    axis.set_ylabel("Calculated − observed / MHz")
    if len(set(datasets)) > 1:
        axis.legend(frameon=False)
    figure.savefig(target, dpi=300)
    plt.close(figure)
    return target


def _apply_variables(parameters, model, variables, values):
    current_parameters, current_model = parameters, model
    for variable, value in zip(variables, values, strict=True):
        for target in variable.targets:
            current_parameters, current_model = _set_parameter(
                current_parameters,
                current_model,
                target.path,
                target.factor * float(value) + target.offset,
            )
    return current_parameters, current_model


def _get_parameter(
    parameters: NativeRotorParameters, model: SpinHyperfineModel, path: str
) -> float:
    parts = path.split(".")
    if parts[0] == "rotor":
        if len(parts) == 2 and parts[1] in {"A_MHz", "B_MHz", "C_MHz"}:
            return float(getattr(parameters, parts[1]))
        if len(parts) == 3 and parts[1] in {"quartic_MHz", "sextic_MHz", "octic_MHz"}:
            return float(getattr(parameters, parts[1])[int(parts[2])])
    if parts[0] == "electron":
        if len(parts) == 2 and parts[1] == "isotropic_spin_rotation_MHz":
            return model.electron.isotropic_spin_rotation_MHz
        if len(parts) == 3 and parts[1] in {
            "anisotropic_spin_rotation_tensor_MHz",
            "zero_field_splitting_tensor_MHz",
        }:
            return float(getattr(model.electron, parts[1])[_tensor_index(parts[2])])
    if parts[0].startswith("nucleus:"):
        nucleus = _find_nucleus(model, parts[0][8:])
        if len(parts) == 2 and parts[1] in {
            "fermi_contact_MHz",
            "isotropic_nuclear_spin_rotation_MHz",
        }:
            return float(getattr(nucleus, parts[1]))
        if len(parts) == 3 and parts[1] in {
            "quadrupole_tensor_MHz",
            "electron_nuclear_dipolar_tensor_MHz",
        }:
            return float(getattr(nucleus, parts[1])[_tensor_index(parts[2])])
    if parts[0].startswith("pair:"):
        pair = _find_pair(model, parts[0][5:])
        if len(parts) == 2 and parts[1] == "isotropic_MHz":
            return pair.isotropic_MHz
        if len(parts) == 3 and parts[1] == "dipolar_tensor_MHz":
            return float(pair.dipolar_tensor_MHz[_tensor_index(parts[2])])
    raise ValueError(f"unsupported LOCK fit parameter path: {path}")


def _set_parameter(parameters, model, path, value):
    parts = path.split(".")
    if parts[0] == "rotor":
        if len(parts) == 2 and parts[1] in {"A_MHz", "B_MHz", "C_MHz"}:
            return replace(parameters, **{parts[1]: value}), model
        if len(parts) == 3 and parts[1] in {"quartic_MHz", "sextic_MHz", "octic_MHz"}:
            values = list(getattr(parameters, parts[1]))
            values[int(parts[2])] = value
            return replace(parameters, **{parts[1]: tuple(values)}), model
    if parts[0] == "electron":
        electron = model.electron
        if len(parts) == 2 and parts[1] == "isotropic_spin_rotation_MHz":
            return parameters, replace(model, electron=replace(electron, **{parts[1]: value}))
        if len(parts) == 3 and parts[1] in {
            "anisotropic_spin_rotation_tensor_MHz",
            "zero_field_splitting_tensor_MHz",
        }:
            values = list(getattr(electron, parts[1]))
            values[_tensor_index(parts[2])] = value
            return parameters, replace(
                model, electron=replace(electron, **{parts[1]: tuple(values)})
            )
    if parts[0].startswith("nucleus:"):
        index = _find_nucleus_index(model, parts[0][8:])
        nucleus = model.nuclei[index]
        if len(parts) == 2 and parts[1] in {
            "fermi_contact_MHz",
            "isotropic_nuclear_spin_rotation_MHz",
        }:
            replacement = replace(nucleus, **{parts[1]: value})
        elif len(parts) == 3 and parts[1] in {
            "quadrupole_tensor_MHz",
            "electron_nuclear_dipolar_tensor_MHz",
        }:
            values = list(getattr(nucleus, parts[1]))
            values[_tensor_index(parts[2])] = value
            replacement = replace(nucleus, **{parts[1]: tuple(values)})
        else:
            raise ValueError(f"unsupported LOCK fit parameter path: {path}")
        nuclei = list(model.nuclei)
        nuclei[index] = replacement
        return parameters, replace(model, nuclei=tuple(nuclei))
    if parts[0].startswith("pair:"):
        index = _find_pair_index(model, parts[0][5:])
        pair = model.nuclear_spin_spin[index]
        if len(parts) == 2 and parts[1] == "isotropic_MHz":
            replacement = replace(pair, isotropic_MHz=value)
        elif len(parts) == 3 and parts[1] == "dipolar_tensor_MHz":
            values = list(pair.dipolar_tensor_MHz)
            values[_tensor_index(parts[2])] = value
            replacement = replace(pair, dipolar_tensor_MHz=tuple(values))
        else:
            raise ValueError(f"unsupported LOCK fit parameter path: {path}")
        pairs = list(model.nuclear_spin_spin)
        pairs[index] = replacement
        return parameters, replace(model, nuclear_spin_spin=tuple(pairs))
    raise ValueError(f"unsupported LOCK fit parameter path: {path}")


def _observation_value(observation, levels):
    if observation.state is not None:
        return _unique_level(observation.state, levels).energy_MHz
    values, weights = [], []
    for component in observation.components:
        lower = _unique_level(component.lower, levels)
        upper = _unique_level(component.upper, levels)
        values.append(upper.energy_MHz - lower.energy_MHz)
        weights.append(component.weight)
    return float(np.average(values, weights=weights))


def _selector_locations(observations, solution):
    selectors = {
        selector for observation in observations for selector in _observation_states(observation)
    }
    result = {}
    for selector in selectors:
        matches = [
            (block.two_F, index)
            for block in solution.blocks
            for index, level in enumerate(block.levels)
            if selector.matches(level)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"assigned selector {selector.to_dict()} matched {len(matches)} levels; "
                "refine the assignment"
            )
        result[selector] = matches[0]
    return result


def _spin_overlap_mappings(reference, current):
    from scipy.optimize import linear_sum_assignment

    current_by_f = {block.two_F: block for block in current.blocks}
    mappings = {}
    for left in reference.blocks:
        right = current_by_f.get(left.two_F)
        if right is None or left.basis != right.basis or left.vectors.shape != right.vectors.shape:
            raise ValueError("spin/hyperfine overlap tracking requires identical complete F blocks")
        rows, columns = linear_sum_assignment(-(np.abs(left.vectors.conj().T @ right.vectors) ** 2))
        mappings[left.two_F] = {
            int(row): int(column) for row, column in zip(rows, columns, strict=True)
        }
    return mappings


def _observation_value_at_locations(observation, solution, locations):
    blocks = {block.two_F: block for block in solution.blocks}

    def energy(selector):
        two_F, index = locations[selector]
        return float(blocks[two_F].energies_MHz[index])

    if observation.state is not None:
        return energy(observation.state)
    values = [
        energy(component.upper) - energy(component.lower) for component in observation.components
    ]
    weights = [component.weight for component in observation.components]
    return float(np.average(values, weights=weights))


def _unique_level(selector, levels):
    matches = [level for level in levels if selector.matches(level)]
    if len(matches) != 1:
        raise ValueError(
            f"assigned selector {selector.to_dict()} matched {len(matches)} levels; refine the assignment"
        )
    return matches[0]


def _supports_closed_shell_hf(model, variables):
    return (
        model.electron.two_S == 0
        and not model.nuclei
        and not model.nuclear_spin_spin
        and all(
            target.path.startswith("rotor.")
            for variable in variables
            for target in variable.targets
        )
    )


def _closed_shell_hf_jacobian(parameters, model, variables, x, observations, sigma):
    fitted, _model = _apply_variables(parameters, model, variables, x)
    by_n: dict[int, tuple[Any, np.ndarray]] = {}
    for N in sorted(
        {state.N for observation in observations for state in _observation_states(observation)}
    ):
        block = diagonalize_rotor(fitted, N)
        operators = []
        for variable in variables:
            operator = np.zeros((2 * N + 1, 2 * N + 1), dtype=complex)
            for target in variable.targets:
                operator += target.factor * _rotor_parameter_operator(fitted, N, target.path)
            operators.append(operator)
        derivatives = hellmann_feynman_diagonal(block.vectors, operators)
        by_n[N] = (block, derivatives)
    jacobian = np.empty((len(observations), len(variables)))
    for row, observation in enumerate(observations):
        if observation.state is not None:
            jacobian[row] = _selector_rotor_derivative(observation.state, by_n)
        else:
            total = np.zeros(len(variables))
            weight = 0.0
            for component in observation.components:
                total += component.weight * (
                    _selector_rotor_derivative(component.upper, by_n)
                    - _selector_rotor_derivative(component.lower, by_n)
                )
                weight += component.weight
            jacobian[row] = total / weight
        jacobian[row] /= sigma[row]
    return jacobian


def _spin_hyperfine_hf_jacobian(
    parameters,
    model,
    variables,
    x,
    observations,
    sigma,
    n_values,
    assembly_backend,
    diagonalizer_backend,
    *,
    solution=None,
    locations=None,
):
    """Differentiate the assembled Hamiltonian before taking expectations.

    Every supported rigid-rotor and spin/hyperfine parameter enters the
    Hamiltonian linearly. A symmetric unit displacement therefore constructs
    its exact operator in the common coupled basis without differencing
    eigenvalues or assignments.
    """
    fitted_parameters, fitted_model = _apply_variables(parameters, model, variables, x)
    if solution is None:
        solution = solve_spin_hyperfine_rotor(
            fitted_parameters,
            fitted_model,
            n_values=n_values,
            assembly_backend=assembly_backend,
            diagonalizer_backend=diagonalizer_backend,
        )
    derivatives_by_f = spin_hyperfine_level_derivatives(
        fitted_parameters,
        fitted_model,
        variables,
        solution,
        assembly_backend=assembly_backend,
    )
    levels_by_f = {block.two_F: block.levels for block in solution.blocks}

    def derivative(selector):
        if locations is not None:
            two_F, index = locations[selector]
            return derivatives_by_f[two_F][:, index]
        matches = [
            (two_F, index)
            for two_F, levels in levels_by_f.items()
            for index, level in enumerate(levels)
            if selector.matches(level)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"assigned selector {selector.to_dict()} matched {len(matches)} levels; "
                "refine the assignment"
            )
        two_F, index = matches[0]
        return derivatives_by_f[two_F][:, index]

    jacobian = np.empty((len(observations), len(variables)))
    for row, observation in enumerate(observations):
        if observation.state is not None:
            jacobian[row] = derivative(observation.state)
        else:
            total = np.zeros(len(variables))
            weight = 0.0
            for component in observation.components:
                total += component.weight * (
                    derivative(component.upper) - derivative(component.lower)
                )
                weight += component.weight
            jacobian[row] = total / weight
        jacobian[row] /= sigma[row]
    return jacobian


def spin_hyperfine_level_derivatives(
    parameters: NativeRotorParameters,
    model: SpinHyperfineModel,
    variables: tuple[FitVariable, ...],
    solution: Any,
    *,
    assembly_backend: str | None = None,
) -> dict[int, np.ndarray]:
    """Return exact local dE/dp arrays for every resolved F block.

    Rows follow ``variables`` and columns follow the eigenlevels in each block.
    Unit Hamiltonian displacements are exact because every supported LOCK
    spectroscopic constant enters the Hamiltonian linearly.
    """
    two_spins = tuple(nucleus.two_I for nucleus in model.nuclei)
    derivatives_by_f: dict[int, np.ndarray] = {}
    for block in solution.blocks:
        groups = _basis_group_slices(block.basis)
        operators = []
        for variable in variables:
            operator = np.zeros((len(block.basis), len(block.basis)), dtype=complex)
            for target in variable.targets:
                current = _get_parameter(parameters, model, target.path)
                plus_parameters, plus_model = _set_parameter(
                    parameters, model, target.path, current + 0.5
                )
                minus_parameters, minus_model = _set_parameter(
                    parameters, model, target.path, current - 0.5
                )
                plus, _backend = _assemble_f_block_hamiltonian(
                    plus_parameters,
                    plus_model,
                    two_spins,
                    block.two_F,
                    block.basis,
                    groups,
                    assembly_backend,
                )
                minus, _backend = _assemble_f_block_hamiltonian(
                    minus_parameters,
                    minus_model,
                    two_spins,
                    block.two_F,
                    block.basis,
                    groups,
                    assembly_backend,
                )
                operator += target.factor * (plus - minus)
            operators.append(operator)
        derivatives_by_f[block.two_F] = hellmann_feynman_diagonal(
            block.vectors, operators
        )
    return derivatives_by_f


def _observation_states(observation):
    if observation.state is not None:
        return (observation.state,)
    return tuple(
        state
        for component in observation.components
        for state in (component.lower, component.upper)
    )


def _selector_rotor_derivative(selector, by_n):
    block, derivatives = by_n[selector.N]
    matches = [
        index
        for index, assignment in enumerate(block.assignments)
        if assignment[0] == selector.Ka and assignment[1] == selector.Kc
    ]
    if len(matches) != 1:
        raise ValueError(
            f"closed-shell selector {selector.to_dict()} matched {len(matches)} rotor states"
        )
    return derivatives[:, matches[0]]


def _rotor_parameter_operator(parameters, N, path):
    parts = path.split(".")
    if len(parts) == 2 and parts[0] == "rotor" and parts[1] in {"A_MHz", "B_MHz", "C_MHz"}:
        axis = {"A_MHz": 0, "B_MHz": 1, "C_MHz": 2}[parts[1]]
        angular = angular_momentum_operators(N, parameters.representation)[axis]
        return angular @ angular
    if (len(parts) == 3 and parts[0] == "rotor"
            and parts[1] in {"quartic_MHz", "sextic_MHz", "octic_MHz"}):
        offset = {"quartic_MHz": 0, "sextic_MHz": 5, "octic_MHz": 12}[parts[1]]
        return centrifugal_distortion_operators(N, parameters.reduction, parameters.representation)[
            offset + int(parts[2])
        ]
    raise ValueError(f"Hellmann-Feynman operator is unavailable for {path}")


def _finite_difference_jacobian(function, x, scales, lower, upper):
    base = np.asarray(function(x), dtype=float)
    jacobian = np.empty((len(base), len(x)))
    for index in range(len(x)):
        step = np.cbrt(np.finfo(float).eps) * max(abs(x[index]), scales[index], 1.0)
        plus, minus = x.copy(), x.copy()
        plus[index] = min(x[index] + step, upper[index])
        minus[index] = max(x[index] - step, lower[index])
        if plus[index] == minus[index]:
            jacobian[:, index] = 0.0
        else:
            jacobian[:, index] = (function(plus) - function(minus)) / (plus[index] - minus[index])
    return jacobian


def _svd_diagnostics(jacobian, scales, rcond):
    singular = np.linalg.svd(jacobian * scales[np.newaxis, :], compute_uv=False)
    threshold = 0.0 if not len(singular) else rcond * singular[0]
    rank = int(np.count_nonzero(singular > threshold))
    condition = None if rank == 0 or rank < jacobian.shape[1] else float(singular[0] / singular[-1])
    return {
        "rank": rank,
        "column_count": jacobian.shape[1],
        "singular_values_scaled": singular.tolist(),
        "rank_rcond": rcond,
        "condition_number_scaled": condition,
    }


def _correlation(covariance, errors):
    denominator = np.outer(errors, errors)
    result = np.zeros_like(covariance)
    np.divide(covariance, denominator, out=result, where=denominator > 0.0)
    np.fill_diagonal(result, 1.0)
    return result


def _find_nucleus_index(model, label):
    indices = [
        index
        for index, item in enumerate(model.nuclei)
        if item.label.casefold() == label.casefold()
    ]
    if len(indices) != 1:
        raise ValueError(f"unknown or ambiguous nucleus label: {label}")
    return indices[0]


def _find_nucleus(model, label) -> NuclearSpin:
    return model.nuclei[_find_nucleus_index(model, label)]


def _find_pair_index(model, token):
    labels = token.split("|")
    if len(labels) != 2:
        raise ValueError("pair paths use pair:left|right")
    wanted = {label.casefold() for label in labels}
    indices = [
        index
        for index, item in enumerate(model.nuclear_spin_spin)
        if {item.left.casefold(), item.right.casefold()} == wanted
    ]
    if len(indices) != 1:
        raise ValueError(f"unknown or ambiguous nuclear pair: {token}")
    return indices[0]


def _find_pair(model, token) -> NuclearSpinSpinCoupling:
    return model.nuclear_spin_spin[_find_pair_index(model, token)]


def _tensor_index(component):
    try:
        return _TENSOR_COMPONENTS[component]
    except KeyError as exc:
        raise ValueError(
            f"tensor component must be one of {', '.join(_TENSOR_COMPONENTS)}"
        ) from exc


def _mapping(value, name):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _optional_int(value):
    return None if value is None else int(value)


def _rotor_to_dict(parameters):
    return {
        "A_MHz": parameters.A_MHz,
        "B_MHz": parameters.B_MHz,
        "C_MHz": parameters.C_MHz,
        "quartic_MHz": list(parameters.quartic_MHz),
        "sextic_MHz": list(parameters.sextic_MHz),
        "octic_MHz": list(parameters.octic_MHz),
        "reduction": parameters.reduction,
        "representation": parameters.representation,
    }


__all__ = [
    "FitVariable",
    "LevelSelector",
    "ParameterTarget",
    "SPECTRAL_FIT_RESULT_SCHEMA",
    "SPECTRAL_FIT_SCHEMA",
    "SpectralFitResult",
    "SpectralObservation",
    "TransitionComponent",
    "fit_assigned_spectrum",
    "read_spectral_fit_request",
    "write_spectral_fit_result",
    "write_spectral_fit_catalog_csv",
    "write_spectral_fit_diagnostics_plot",
    "spin_hyperfine_level_derivatives",
]
