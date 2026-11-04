"""Global assigned-spectrum fits across states, isotopologues, and datasets."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares, minimize, OptimizeResult
from matrix_numerics.least_squares import svd_covariance

from .native_rotor import NativeRotorParameters
from .spectral_fit import SpectralObservation, _get_parameter, _observation_value, _set_parameter
from .spin_hyperfine import SpinHyperfineModel, solve_spin_hyperfine_rotor


GLOBAL_SPECTRAL_FIT_SCHEMA = "matrix.lock.global_spectral_fit.v1"
GLOBAL_SPECTRAL_FIT_RESULT_SCHEMA = "matrix.lock.global_spectral_fit_result.v1"


def fit_global_assigned_spectra(
    systems: Mapping[str, tuple[NativeRotorParameters, SpinHyperfineModel]],
    request: Mapping[str, Any],
    *,
    parameterization: Any = None,
) -> dict[str, Any]:
    """Fit shared parameters with priors, correlations, and validation data.

    A ``system`` can identify any isotopologue, vibrational state, or electronic
    state. Variables share values through target factors and offsets. A target
    can additionally specify Dunham reduced-mass scaling.

    ``parameterization`` optionally binds physical variables nonlinearly to
    system parameters. It supplies ``parameter_names``, ``controlled_targets``,
    ``apply(values, systems)``, ``residuals(values)`` and ``describe(values)``.
    Its controlled targets cannot also be independent affine variables. Optional
    ``propagate_covariance(values, names, covariance)`` reports derived errors
    using the absolute observation covariance, without residual scaling.
    Optional ``inequality_constraints(values)`` returns dimensionless slacks
    that must be nonnegative. Nonempty slacks select SLSQP (linear loss only);
    errors are suppressed at active inequalities as at variable bounds.
    """
    if request.get("schema") != GLOBAL_SPECTRAL_FIT_SCHEMA:
        raise ValueError(f"unsupported global fit schema: {request.get('schema')!r}")
    datasets = request.get("datasets")
    variables_raw = request.get("variables")
    if not isinstance(datasets, list) or not datasets or not isinstance(variables_raw, list) or not variables_raw:
        raise ValueError("global fit requires nonempty datasets and variables")

    observations: list[tuple[str, SpectralObservation]] = []
    roles: list[str] = []
    dataset_names: list[str] = []
    covariance_blocks: list[tuple[slice, np.ndarray]] = []
    n_values: dict[str, set[int]] = {name: set() for name in systems}
    for dataset_index, dataset in enumerate(datasets):
        if not isinstance(dataset, Mapping) or str(dataset.get("system")) not in systems:
            raise ValueError("each dataset must name a supplied system")
        system = str(dataset["system"])
        role = str(dataset.get("role", "training")).lower()
        if role not in {"training", "validation"}:
            raise ValueError("dataset role must be training or validation")
        raw = dataset.get("observations")
        if not isinstance(raw, list) or not raw:
            raise ValueError("each global dataset needs observations")
        start = len(observations)
        dataset_name = str(dataset.get("name", f"dataset-{dataset_index + 1}"))
        for item in raw:
            observation = SpectralObservation.from_dict(item, len(observations))
            observations.append((system, observation))
            roles.append(role)
            dataset_names.append(dataset_name)
            states = (
                [observation.state]
                if observation.state is not None
                else [
                    state
                    for component in observation.components
                    for state in (component.lower, component.upper)
                ]
            )
            n_values[system].update(state.N for state in states)
        if "covariance_MHz2" in dataset:
            covariance = np.asarray(dataset["covariance_MHz2"], dtype=float)
            if covariance.shape != (len(raw), len(raw)) or not np.allclose(covariance, covariance.T):
                raise ValueError("dataset covariance_MHz2 must be a symmetric square matrix")
            if np.min(np.linalg.eigvalsh(covariance)) <= 0.0:
                raise ValueError("dataset covariance_MHz2 must be positive definite")
            covariance_blocks.append((slice(start, len(observations)), covariance))

    training = np.asarray([role == "training" for role in roles], dtype=bool)
    if not np.any(training):
        raise ValueError("a global fit needs at least one training observation")
    observed = np.asarray([item.frequency_MHz for _system, item in observations])
    sigma = np.asarray([
        item.uncertainty_MHz / math.sqrt(item.weight) for _system, item in observations
    ])
    observation_covariance = np.diag(sigma * sigma)
    for location, block in covariance_blocks:
        observation_covariance[location, location] = block
    training_indices = np.flatnonzero(training)
    validation_indices = np.flatnonzero(~training)
    training_whitener = _inverse_sqrt(observation_covariance[np.ix_(training_indices, training_indices)])
    validation_whitener = (
        _inverse_sqrt(observation_covariance[np.ix_(validation_indices, validation_indices)])
        if len(validation_indices) else np.empty((0, 0))
    )

    controlled_paths: set[tuple[str, str]] = set()
    physical_names = set()
    if parameterization is not None:
        physical_names = set(parameterization.parameter_names)
        for system, path in parameterization.controlled_targets:
            if system not in systems:
                raise ValueError("physical parameterization names an unknown system")
            _get_parameter(*systems[system], path)
            controlled_paths.add((system, path))
    fixed = []
    base_systems = {name: value for name, value in systems.items()}
    for item in request.get("fixed_parameters", ()):
        if not isinstance(item, Mapping):
            raise ValueError("fixed parameters must be objects")
        value = float(item["value"])
        if not math.isfinite(value):
            raise ValueError("a fixed theoretical value must be finite")
        targets = _read_targets(item.get("targets"), base_systems, controlled_paths)
        for system, path, factor, offset in targets:
            base_systems[system] = _set_parameter(*base_systems[system], path, factor * value + offset)
        fixed.append({"name": str(item.get("name", f"fixed-{len(fixed) + 1}")), "value": value, "targets": targets})

    variables = []
    for index, item in enumerate(variables_raw):
        if not isinstance(item, Mapping):
            raise ValueError("global variables must be objects")
        if str(item.get("name", "")) in physical_names:
            if item.get("targets"):
                raise ValueError("a physical variable cannot also have affine targets")
            targets = []
        else:
            targets = _read_targets(item.get("targets"), base_systems, controlled_paths)
        if "initial" in item:
            initial = float(item["initial"])
        elif len(targets) == 1:
            system, path, factor, offset = targets[0]
            initial = (_get_parameter(*base_systems[system], path) - offset) / factor
        else:
            raise ValueError("shared global variables require an initial value")
        scale = float(item.get("scale", max(abs(initial), 1.0)))
        lower, upper = float(item.get("lower", -math.inf)), float(item.get("upper", math.inf))
        prior = item.get("prior", item.get("theoretical_prior"))
        if not math.isfinite(initial) or not math.isfinite(scale) or scale <= 0.0 or math.isnan(lower) or math.isnan(upper) or lower >= upper or not lower <= initial <= upper:
            raise ValueError("invalid global variable initial value, scale, or bounds")
        if prior is not None:
            if not isinstance(prior, Mapping):
                raise ValueError("global variable prior must be an object")
            prior_value, prior_sigma = float(prior["value"]), float(prior["sigma"])
            if not math.isfinite(prior_value) or not math.isfinite(prior_sigma) or prior_sigma <= 0.0:
                raise ValueError("global priors require a finite value and positive sigma")
        variables.append({
            "name": str(item.get("name", f"p{index + 1}")), "targets": targets,
            "initial": initial, "scale": scale, "lower": lower, "upper": upper, "prior": prior,
        })
    names = [item["name"] for item in variables]
    if any(not name.strip() for name in names) or len(names) != len(set(names)):
        raise ValueError("global variable names must be nonempty and unique")
    if not physical_names <= set(names):
        raise ValueError("global request omits physical parameterization variables")

    x0 = np.asarray([item["initial"] for item in variables])
    scales = np.asarray([item["scale"] for item in variables])
    lower = np.asarray([item["lower"] for item in variables])
    upper = np.asarray([item["upper"] for item in variables])
    cache: dict[bytes, tuple[np.ndarray, Mapping[str, tuple[NativeRotorParameters, SpinHyperfineModel]]]] = {}

    def apply(values: Sequence[float]):
        current = {name: value for name, value in base_systems.items()}
        for variable, value in zip(variables, values, strict=True):
            for system, path, factor, offset in variable["targets"]:
                current[system] = _set_parameter(*current[system], path, factor * float(value) + offset)
        if parameterization is not None:
            current = parameterization.apply(dict(zip(names, values, strict=True)), current)
        return current

    def predict(values: Sequence[float]):
        key = np.asarray(values, dtype=np.float64).tobytes()
        if key in cache:
            return cache[key]
        fitted = apply(values)
        levels = {}
        for name, (parameters, model) in fitted.items():
            if n_values[name]:
                # Rank-2 spin and hyperfine operators couple N to N +/- 2.
                # Retain two unobserved guard manifolds so a dataset split
                # cannot introduce a basis-truncation shift in the fit.
                guarded_n_values = range(0, max(n_values[name]) + 3)
                solved = solve_spin_hyperfine_rotor(
                    parameters, model, n_values=guarded_n_values, assembly_backend="compiled"
                )
                levels[name] = tuple(level for block in solved.blocks for level in block.levels)
        predictions = np.asarray([
            _observation_value(observation, levels[system]) for system, observation in observations
        ])
        cache[key] = predictions, fitted
        return cache[key]

    prior_indices = [index for index, item in enumerate(variables) if item["prior"] is not None]

    def training_residual(values: Sequence[float]) -> np.ndarray:
        raw = predict(values)[0][training_indices] - observed[training_indices]
        return training_whitener @ raw

    def residual(values: Sequence[float]) -> np.ndarray:
        data = training_residual(values)
        priors = [
            (values[index] - float(variables[index]["prior"]["value"]))
            / float(variables[index]["prior"]["sigma"])
            for index in prior_indices
        ]
        physical = (
            np.asarray(parameterization.residuals(dict(zip(names, values, strict=True))), float)
            if parameterization is not None else np.empty(0)
        )
        if physical.ndim != 1 or not np.all(np.isfinite(physical)):
            raise ValueError("physical parameterization returned invalid residuals")
        return np.concatenate((data, priors, physical))

    options = request.get("options", {})
    if not isinstance(options, Mapping):
        raise ValueError("global fit options must be an object")
    loss = str(options.get("loss", "linear"))
    if loss not in {"linear", "soft_l1", "huber"}:
        raise ValueError("global fit loss must be linear, soft_l1, or huber")
    # A tiny but nonzero physical starting vector can collapse TRF's initial
    # trust radius. Start at unit scaled coordinates for every parameter.
    # Differentiate in physical units before scaling, retaining the same
    # finite-difference resolution as covariance and rank diagnostics.
    def physical_values(scaled_values):
        return x0 + (np.asarray(scaled_values) - 1.0) * scales

    def scaled_jacobian(scaled_values):
        values = physical_values(scaled_values)
        return _finite_jacobian(residual, values, scales, lower, upper) * scales[None, :]

    constraint_function = getattr(parameterization, "inequality_constraints", None)
    def constraint_slacks(scaled_values):
        slacks = np.asarray(constraint_function(dict(zip(names, physical_values(scaled_values), strict=True))), float)
        if slacks.ndim != 1 or not np.all(np.isfinite(slacks)):
            raise ValueError("physical parameterization returned invalid inequality slacks")
        return slacks

    constrained = constraint_function is not None and len(constraint_slacks(np.ones_like(x0))) > 0
    scaled_lower = 1.0 + (lower - x0) / scales
    scaled_upper = 1.0 + (upper - x0) / scales
    if constrained:
        if loss != "linear":
            raise ValueError("nonlinear physical constraints currently require linear loss")
        # Normalize the entire objective by one constant for SLSQP's numerical
        # conditioning; relative data/prior/predicate weights stay unchanged.
        normalization = max(float(residual(x0) @ residual(x0)), 1.0)
        def objective(z):
            r = residual(physical_values(z))
            return float(r @ r) / normalization
        def gradient(z):
            return 2.0 * scaled_jacobian(z).T @ residual(physical_values(z)) / normalization
        def constraint_jacobian(z):
            values = physical_values(z)
            def physical_slacks(v):
                return np.asarray(constraint_function(dict(zip(names, v, strict=True))), float)
            return _finite_jacobian(physical_slacks, values, scales, lower, upper) * scales[None, :]
        solved = minimize(
            objective, np.ones_like(x0), jac=gradient, method="SLSQP",
            bounds=list(zip(scaled_lower, scaled_upper)),
            constraints=[dict(type="ineq", fun=constraint_slacks, jac=constraint_jacobian)],
            options=dict(maxiter=int(options.get("max_nfev", 500)),
                         ftol=float(options.get("constrained_ftol", 1e-12))),
        )
        minimum_slack = float(np.min(constraint_slacks(solved.x)))
        optimized = OptimizeResult(
            x=solved.x, success=bool(solved.success and minimum_slack >= -1e-7),
            message=solved.message, nfev=solved.nfev,
            optimality=float(np.linalg.norm(gradient(solved.x), ord=np.inf)),
            active_mask=np.where(np.isclose(solved.x, scaled_lower, atol=1e-7, rtol=0), -1,
                        np.where(np.isclose(solved.x, scaled_upper, atol=1e-7, rtol=0), 1, 0)),
        )
    else:
        optimized = least_squares(
            lambda values: residual(physical_values(values)), np.ones_like(x0),
            bounds=(scaled_lower, scaled_upper), x_scale=1.0, jac=scaled_jacobian,
            loss=loss, max_nfev=int(options.get("max_nfev", 500)),
        )
    optimized.x = physical_values(optimized.x)
    calculated, fitted = predict(optimized.x)
    raw = calculated - observed
    prediction_jacobian = _finite_jacobian(lambda values: predict(values)[0], optimized.x, scales, lower, upper)
    augmented_jacobian = _finite_jacobian(residual, optimized.x, scales, lower, upper)
    scaled_singular = np.linalg.svd(augmented_jacobian * scales[None, :], compute_uv=False)
    rcond = float(options.get("rank_rcond", 1.0e-10))
    rank = int(np.count_nonzero(scaled_singular > (rcond * scaled_singular[0] if len(scaled_singular) else 0.0)))
    data_jacobian = training_whitener @ prediction_jacobian[training_indices]
    data_singular = np.linalg.svd(data_jacobian * scales[None, :], compute_uv=False)
    data_rank = int(np.count_nonzero(data_singular > (rcond * data_singular[0] if len(data_singular) else 0.0)))
    covariance, _covariance_rank, _ = svd_covariance(augmented_jacobian, scales=scales, rcond=rcond)
    identifiable = rank == len(variables)
    final_slacks = constraint_slacks(1.0 + (optimized.x - x0) / scales) if constrained else np.empty(0)
    interior = not np.any(optimized.active_mask) and not np.any(final_slacks <= 1e-6)
    gaussian_uncertainties = identifiable and interior
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    prediction_uncertainties = np.sqrt(np.maximum(
        np.einsum("ij,jk,ik->i", prediction_jacobian, covariance, prediction_jacobian), 0.0
    ))
    train_whitened = training_whitener @ raw[training_indices]
    validation_whitened = validation_whitener @ raw[validation_indices] if len(validation_indices) else np.asarray([])
    prediction_total_sigma = np.sqrt(np.maximum(np.diag(observation_covariance) + prediction_uncertainties**2, 0.0))
    predictive_z = raw / prediction_total_sigma

    return {
        "schema": GLOBAL_SPECTRAL_FIT_RESULT_SCHEMA,
        "status": "converged" if optimized.success and identifiable else "rank_deficient" if optimized.success else "failed",
        "success": bool(optimized.success and identifiable),
        "systems": {
            name: {
                "rotor": {"A_MHz": pair[0].A_MHz, "B_MHz": pair[0].B_MHz, "C_MHz": pair[0].C_MHz},
                "spin_hyperfine": pair[1].to_dict(),
            }
            for name, pair in fitted.items()
        },
        "fixed_parameters": fixed,
        "variables": [
            {
                "name": item["name"], "value": float(optimized.x[index]),
                "standard_error": float(standard_errors[index]) if gaussian_uncertainties else None,
                "targets": item["targets"], "prior": item["prior"],
            }
            for index, item in enumerate(variables)
        ],
        "comparisons": [
            {
                "system": system, "dataset": dataset_names[index], "role": roles[index],
                "label": observation.label, "observed_MHz": observation.frequency_MHz,
                "calculated_MHz": float(calculated[index]),
                "calculated_standard_uncertainty_MHz": float(prediction_uncertainties[index]) if gaussian_uncertainties else None,
                "calculated_minus_observed_MHz": float(raw[index]),
                "standardized_residual": float(raw[index] / sigma[index]),
                "predictive_standardized_residual": float(predictive_z[index]) if gaussian_uncertainties else None,
            }
            for index, (system, observation) in enumerate(observations)
        ],
        "metrics": {
            "system_count": len(systems), "dataset_count": len(datasets),
            "observation_count": len(observations),
            "training_observation_count": int(np.count_nonzero(training)),
            "validation_observation_count": int(np.count_nonzero(~training)),
            "free_parameter_count": len(variables), "rank": rank,
            "data_only_rank": data_rank,
            "training_rms_residual_MHz": _rms(raw[training_indices]),
            "training_chi_square": float(np.dot(train_whitened, train_whitened)),
            "validation_rms_residual_MHz": _rms(raw[validation_indices]) if len(validation_indices) else None,
            "validation_chi_square": float(np.dot(validation_whitened, validation_whitened)) if len(validation_indices) else None,
            "validation_predictive_coverage": _coverage(predictive_z[validation_indices]) if gaussian_uncertainties else None,
            "rms_residual_MHz": _rms(raw),
            "chi_square": float(np.dot(train_whitened, train_whitened)),
        },
        "singular_values_scaled": scaled_singular.tolist(),
        "data_only_singular_values_scaled": data_singular.tolist(),
        "optimization": {
            "message": str(optimized.message), "nfev": int(optimized.nfev),
            "optimality": float(optimized.optimality),
            "active_bounds": optimized.active_mask.tolist(),
            "optimizer": "SLSQP" if constrained else "TRF",
            "optimality_convention": (
                "raw normalized objective gradient infinity norm; not a KKT residual"
                if constrained else "TRF scaled gradient norm"
            ),
            "inequality_constraint_slacks": final_slacks.tolist(),
            "constraint_chi_square": float(np.dot(residual(optimized.x)[len(training_indices):], residual(optimized.x)[len(training_indices):])),
        },
        "covariance": covariance.tolist(),
        "correlation": _covariance_to_correlation(covariance).tolist(),
        "physical_parameterization": (
            None if parameterization is None else parameterization.describe(
                dict(zip(names, optimized.x, strict=True))
            )
        ),
        "physical_uncertainties": (
            parameterization.propagate_covariance(
                dict(zip(names, optimized.x, strict=True)), names, covariance
            )
            if gaussian_uncertainties and parameterization is not None
            and hasattr(parameterization, "propagate_covariance") else None
        ),
        "covariance_convention": "absolute observation covariance and supplied priors; no residual scaling",
        "uncertainty_status": (
            "rank_deficient" if not identifiable else "bound_limited_local_curvature_only"
            if not interior else "interior_gaussian_conditional_on_model"
        ),
    }


def _read_targets(raw_targets, systems, controlled_paths):
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError("global parameters require targets")
    targets = []
    for target in raw_targets:
        system, path = str(target["system"]), str(target["path"])
        if system not in systems:
            raise ValueError(f"unknown global-fit system: {system}")
        _get_parameter(*systems[system], path)
        factor, offset = float(target.get("factor", 1.0)), float(target.get("offset", 0.0))
        scaling = target.get("dunham_scaling")
        if scaling is not None:
            if not isinstance(scaling, Mapping):
                raise ValueError("dunham_scaling must be an object")
            reference, reduced = float(scaling["reference_reduced_mass_u"]), float(scaling["reduced_mass_u"])
            k, ell = int(scaling["k"]), int(scaling["l"])
            if reference <= 0.0 or reduced <= 0.0 or k < 0 or ell < 0:
                raise ValueError("Dunham masses must be positive and k,l nonnegative")
            factor *= (reference / reduced) ** (0.5 * k + ell)
        if not math.isfinite(factor) or factor == 0.0 or not math.isfinite(offset):
            raise ValueError("global target factors must be finite and nonzero")
        if (system, path) in controlled_paths:
            raise ValueError(f"global parameter path is controlled twice: {system}:{path}")
        controlled_paths.add((system, path))
        targets.append((system, path, factor, offset))
    return targets


def _inverse_sqrt(covariance: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(np.asarray(covariance, dtype=float))
    if np.min(values) <= 0.0:
        raise ValueError("observation covariance must be positive definite")
    return (vectors * values**-0.5) @ vectors.T


def _finite_jacobian(function, x, scales, lower, upper):
    result = np.empty((len(function(x)), len(x)))
    for column in range(len(x)):
        step = np.cbrt(np.finfo(float).eps) * max(abs(x[column]), scales[column], 1.0)
        plus, minus = x.copy(), x.copy()
        plus[column], minus[column] = min(upper[column], x[column] + step), max(lower[column], x[column] - step)
        result[:, column] = (function(plus) - function(minus)) / (plus[column] - minus[column])
    return result


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values) ** 2)))


def _coverage(z_scores: np.ndarray) -> dict[str, float] | None:
    if not len(z_scores):
        return None
    absolute = np.abs(z_scores)
    return {f"within_{count}_sigma_fraction": float(np.mean(absolute <= count)) for count in (1, 2, 3)}


def _covariance_to_correlation(covariance: np.ndarray) -> np.ndarray:
    sigma = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    denominator = np.outer(sigma, sigma)
    return np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > 0.0)


__all__ = [
    "GLOBAL_SPECTRAL_FIT_RESULT_SCHEMA",
    "GLOBAL_SPECTRAL_FIT_SCHEMA",
    "fit_global_assigned_spectra",
]
