"""Native weighted least-squares fitting for ERHAM effective Hamiltonians."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import math
from typing import Mapping, Sequence

import numpy as np

from .erham import (
    ERHAMBlock,
    ERHAMFourierTerm as ERHAMFourierTerm,
    ERHAMModel,
    erham_hamiltonian_motion_derivative,
    solve_erham_block,
)


ERHAM_FIT_RESULT_SCHEMA = "matrix.lock.erham-effective-fit-result.v1"


@dataclass(frozen=True)
class ERHAMObservation:
    species: str
    upper: tuple[int, int, int]
    lower: tuple[int, int, int]
    frequency_MHz: float
    uncertainty_MHz: float
    feature_id: str = ""
    role: str = "training"
    blend_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.uncertainty_MHz <= 0.0 or not math.isfinite(self.uncertainty_MHz):
            raise ValueError("ERHAM observations require positive finite uncertainties")
        if self.blend_weight <= 0.0 or not math.isfinite(self.blend_weight):
            raise ValueError("ERHAM blend weights must be positive and finite")


@dataclass(frozen=True)
class ERHAMFitIteration:
    iteration: int
    weighted_rms: float
    unweighted_rms_MHz: float
    maximum_absolute_residual_MHz: float
    step_norm: float


@dataclass(frozen=True)
class ERHAMFitResult:
    model: ERHAMModel
    calculated_MHz: np.ndarray
    residuals_MHz: np.ndarray
    parameter_names: tuple[str, ...]
    covariance: np.ndarray
    history: tuple[ERHAMFitIteration, ...]


@dataclass(frozen=True)
class ERHAMMotionVariable:
    """One linked nonlinear direction in rho-axis parameter space."""

    name: str
    rho_direction: tuple[float, ...]
    beta_direction: tuple[float, ...]
    finite_step: float
    alpha_direction: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.finite_step <= 0.0 or not math.isfinite(self.finite_step):
            raise ValueError("a positive derivative-qualification step is required")
        if len(self.rho_direction) != len(self.beta_direction):
            raise ValueError("rho and beta directions must have equal length")
        if self.alpha_direction and len(self.alpha_direction) != len(self.rho_direction):
            raise ValueError("alpha directions must match the rho direction length")


_BASE_NAMES = (
    "A", "B", "C", "DeltaJ", "DeltaJK", "DeltaK", "deltaJ", "deltaK",
    "PhiJ", "PhiJK", "PhiKJ", "PhiK", "phiJ", "phiJK", "phiK",
    "LJ", "LJJK", "LJK", "LJKK", "LK", "lJ", "lJJK", "lJKK", "lK",
)


def erham_model_coefficients(model: ERHAMModel) -> dict[str, float]:
    values = model.rotational_MHz + model.quartic_MHz + model.sextic_MHz + model.octic_MHz
    result = dict(zip(_BASE_NAMES, values, strict=True))
    result.update((term.name, term.coefficient_MHz) for term in model.fourier_terms)
    result.update((term.name, term.coefficient_MHz) for term in model.kpr_terms)
    return result


def replace_erham_coefficients(model: ERHAMModel, values: Mapping[str, float]) -> ERHAMModel:
    """Return a model with selected linear Hamiltonian coefficients replaced."""
    current = erham_model_coefficients(model)
    unknown = set(values) - set(current)
    if unknown:
        raise ValueError(f"unknown ERHAM coefficients: {sorted(unknown)}")
    current.update((name, float(value)) for name, value in values.items())
    terms = tuple(
        replace(term, coefficient_MHz=current[term.name]) for term in model.fourier_terms
    )
    kpr_terms = tuple(
        replace(term, coefficient_MHz=current[term.name]) for term in model.kpr_terms
    )
    return replace(
        model,
        rotational_MHz=tuple(current[name] for name in _BASE_NAMES[:3]),
        quartic_MHz=tuple(current[name] for name in _BASE_NAMES[3:8]),
        sextic_MHz=tuple(current[name] for name in _BASE_NAMES[8:15]),
        octic_MHz=tuple(current[name] for name in _BASE_NAMES[15:24]),
        fourier_terms=terms,
        kpr_terms=kpr_terms,
    )


def shift_erham_motions(
    model: ERHAMModel, variable: ERHAMMotionVariable, amount: float
) -> ERHAMModel:
    """Move a model along a linked, molecule-independent rho-axis direction."""
    if len(variable.rho_direction) != len(model.motions):
        raise ValueError("the motion-variable dimension does not match the model")
    alpha = variable.alpha_direction or (0.0,) * len(model.motions)
    motions = tuple(
        replace(
            motion,
            rho=motion.rho + amount * drho,
            beta_rad=motion.beta_rad + amount * dbeta,
            alpha_rad=motion.alpha_rad + amount * dalpha,
        )
        for motion, drho, dbeta, dalpha in zip(
            model.motions, variable.rho_direction, variable.beta_direction, alpha, strict=True
        )
    )
    return replace(model, motions=motions)


def erham_motion_jacobian(
    model: ERHAMModel,
    observations: Sequence[ERHAMObservation],
    variables: Sequence[ERHAMMotionVariable],
    *,
    workers: int = 1,
    _blocks=None,
) -> np.ndarray:
    """Analytic Hellmann-Feynman derivatives for linked rho-axis variables."""
    observations = tuple(observations)
    variables = tuple(variables)
    if not variables:
        return np.empty((len(observations), 0))
    keys = sorted(
        {(level[0], row.species) for row in observations for level in (row.upper, row.lower)}
    )

    def solve(key: tuple[int, str]) -> tuple[tuple[int, str], ERHAMBlock]:
        J, species = key
        return key, solve_erham_block(model, J, species, parameter_names=())

    if _blocks is not None:
        blocks = _blocks
        if set(blocks) != set(keys):
            raise ValueError("point-local ERHAM blocks do not match the observations")
    elif workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            blocks = dict(pool.map(solve, keys))
    else:
        blocks = dict(solve(key) for key in keys)
    indices = {
        key: {(Ka, Kc): index for index, (Ka, Kc, _parity) in enumerate(block.assignments)}
        for key, block in blocks.items()
    }
    level_derivatives: dict[tuple[int, str], np.ndarray] = {}
    for key, block in blocks.items():
        J, species = key
        block_derivative = np.empty((len(block.energies_MHz), len(variables)))
        for column, variable in enumerate(variables):
            alpha = variable.alpha_direction or (0.0,) * len(model.motions)
            derivative = np.zeros((2 * J + 1, 2 * J + 1), dtype=complex)
            for motion_index, (drho, dbeta, dalpha) in enumerate(
                zip(variable.rho_direction, variable.beta_direction, alpha, strict=True)
            ):
                for direction, field in (
                    (drho, "rho"), (dbeta, "beta_rad"), (dalpha, "alpha_rad")
                ):
                    if direction:
                        derivative += direction * erham_hamiltonian_motion_derivative(
                            model, J, species, motion_index, field
                        )
            block_derivative[:, column] = np.real(
                np.einsum("ij,ij->j", block.vectors.conj(), derivative @ block.vectors)
            )
        level_derivatives[key] = block_derivative
    jacobian = np.empty((len(observations), len(variables)))
    for row_index, row in enumerate(observations):
        upper_key = (row.upper[0], row.species)
        lower_key = (row.lower[0], row.species)
        upper_index = indices[upper_key][row.upper[1:]]
        lower_index = indices[lower_key][row.lower[1:]]
        jacobian[row_index] = (
            level_derivatives[upper_key][upper_index]
            - level_derivatives[lower_key][lower_index]
        )
    return jacobian


def erham_motion_coordinate(model: ERHAMModel, variable: ERHAMMotionVariable) -> float:
    """Project a physical rho-axis field onto one linked fit direction."""
    alpha = variable.alpha_direction or (0.0,) * len(model.motions)
    fields = (
        (variable.rho_direction, tuple(item.rho for item in model.motions)),
        (variable.beta_direction, tuple(item.beta_rad for item in model.motions)),
        (alpha, tuple(item.alpha_rad for item in model.motions)),
    )
    active = [(direction, values) for direction, values in fields if any(direction)]
    if len(active) != 1:
        raise ValueError("an absolute motion prior requires a direction in exactly one field")
    direction, values = active[0]
    norm = sum(value * value for value in direction)
    return sum(weight * value for weight, value in zip(direction, values, strict=True)) / norm


def evaluate_erham_observations(
    model: ERHAMModel,
    observations: Sequence[ERHAMObservation],
    *,
    parameter_names: Sequence[str] = (),
    workers: int = 1,
    reference_blocks: Mapping[tuple[int, str], ERHAMBlock] | None = None,
    _return_blocks: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate transitions and an analytic Hellmann-Feynman Jacobian."""
    keys = sorted({(level[0], row.species) for row in observations for level in (row.upper, row.lower)})

    def solve(key: tuple[int, str]) -> tuple[tuple[int, str], ERHAMBlock]:
        J, species = key
        return key, solve_erham_block(model, J, species, parameter_names=parameter_names)

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            blocks = dict(pool.map(solve, keys))
    else:
        blocks = dict(solve(key) for key in keys)
    indices = {}
    permutations = {}
    for key, block in blocks.items():
        reference = None if reference_blocks is None else reference_blocks[key]
        if reference is None:
            assignments = block.assignments
            permutation = tuple(range(len(block.energies_MHz)))
        else:
            assignments = reference.assignments
            from scipy.optimize import linear_sum_assignment

            rows, columns = linear_sum_assignment(
                -(np.abs(reference.vectors.conj().T @ block.vectors) ** 2)
            )
            permutation_array = np.empty(len(rows), dtype=int)
            permutation_array[rows] = columns
            permutation = tuple(int(value) for value in permutation_array)
        indices[key] = {
            (Ka, Kc): index for index, (Ka, Kc, _parity) in enumerate(assignments)
        }
        permutations[key] = permutation
    calculated = np.empty(len(observations), dtype=float)
    jacobian = np.zeros((len(observations), len(parameter_names)), dtype=float)
    for row_index, row in enumerate(observations):
        Ju, Kau, Kcu = row.upper
        Jl, Kal, Kcl = row.lower
        upper_key, lower_key = (Ju, row.species), (Jl, row.species)
        upper_reference = indices[upper_key][(Kau, Kcu)]
        lower_reference = indices[lower_key][(Kal, Kcl)]
        upper_index = permutations[upper_key][upper_reference]
        lower_index = permutations[lower_key][lower_reference]
        upper, lower = blocks[upper_key], blocks[lower_key]
        calculated[row_index] = upper.energies_MHz[upper_index] - lower.energies_MHz[lower_index]
        for column, name in enumerate(parameter_names):
            jacobian[row_index, column] = (
                upper.derivatives[name][upper_index] - lower.derivatives[name][lower_index]
            )
    return (calculated, jacobian, blocks) if _return_blocks else (calculated, jacobian)


def collapse_erham_blends(
    observations: Sequence[ERHAMObservation],
    calculated_MHz: np.ndarray,
    jacobian: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, tuple[str, ...]]:
    """Collapse assigned components to measured features with normalized weights."""
    observations = tuple(observations)
    calculated = np.asarray(calculated_MHz, dtype=float)
    derivatives = None if jacobian is None else np.asarray(jacobian, dtype=float)
    if calculated.shape != (len(observations),):
        raise ValueError("one calculated frequency is required per ERHAM component")
    if derivatives is not None and derivatives.shape[0] != len(observations):
        raise ValueError("the ERHAM Jacobian must have one row per component")
    keys = tuple(row.feature_id or f"component-{index}" for index, row in enumerate(observations))
    order = tuple(dict.fromkeys(keys))
    grouped = {key: [] for key in order}
    for index, key in enumerate(keys):
        grouped[key].append(index)
    observed, sigma, predicted = [], [], []
    collapsed_jacobian = []
    for key in order:
        indices = grouped[key]
        frequencies = np.asarray([observations[index].frequency_MHz for index in indices])
        uncertainties = np.asarray([observations[index].uncertainty_MHz for index in indices])
        if np.max(np.abs(frequencies - frequencies[0])) > 1.0e-9:
            raise ValueError(f"blend feature {key!r} has inconsistent observed frequencies")
        if np.max(np.abs(uncertainties - uncertainties[0])) > 1.0e-12:
            raise ValueError(f"blend feature {key!r} has inconsistent uncertainties")
        weights = np.asarray(
            [observations[index].blend_weight for index in indices], dtype=float
        )
        weights /= float(np.sum(weights))
        observed.append(float(frequencies[0]))
        sigma.append(float(uncertainties[0]))
        predicted.append(float(weights @ calculated[indices]))
        if derivatives is not None:
            collapsed_jacobian.append(weights @ derivatives[indices])
    matrix = None if derivatives is None else np.asarray(collapsed_jacobian, dtype=float)
    return np.asarray(observed), np.asarray(sigma), np.asarray(predicted), matrix, order


def _fit_arrays(
    observations: tuple[ERHAMObservation, ...],
    calculated: np.ndarray,
    jacobian: np.ndarray,
    collapse_blends: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if collapse_blends:
        observed, sigma, predicted, collapsed, _ids = collapse_erham_blends(
            observations, calculated, jacobian
        )
        assert collapsed is not None
        return observed, sigma, predicted, collapsed
    return (
        np.asarray([row.frequency_MHz for row in observations]),
        np.asarray([row.uncertainty_MHz for row in observations]),
        calculated,
        jacobian,
    )


def fit_erham_linear(
    model: ERHAMModel,
    observations: Sequence[ERHAMObservation],
    parameter_names: Sequence[str],
    *,
    workers: int = 1,
    maximum_iterations: int = 8,
    relative_step_tolerance: float = 1.0e-10,
    svd_rcond: float = 1.0e-12,
    collapse_blends: bool = False,
    priors: Mapping[str, tuple[float, float]] | None = None,
) -> ERHAMFitResult:
    """Fit any linear ERHAM coefficients with analytic derivatives.

    Eigenvectors are recomputed after every Gauss-Newton update, so the method
    remains valid through asymmetric-top mixing.  Singular directions are
    removed by the requested SVD threshold rather than hidden by parameter
    clipping or molecule-specific constraints.
    """
    observations = tuple(observations)
    names = tuple(parameter_names)
    available = erham_model_coefficients(model)
    if not observations or not names or len(set(names)) != len(names):
        raise ValueError("provide observations and unique fitted parameter names")
    missing = set(names) - set(available)
    if missing:
        raise ValueError(f"unknown fitted ERHAM coefficients: {sorted(missing)}")
    priors = {} if priors is None else dict(priors)
    if set(priors) - set(names):
        raise ValueError("ERHAM priors must refer to fitted linear coefficients")
    if any(sigma <= 0.0 or not math.isfinite(sigma) for _value, sigma in priors.values()):
        raise ValueError("ERHAM prior uncertainties must be positive and finite")
    history: list[ERHAMFitIteration] = []
    fitted = model
    reference_keys = sorted(
        {(level[0], row.species) for row in observations for level in (row.upper, row.lower)}
    )
    reference_blocks = {
        key: solve_erham_block(model, *key) for key in reference_keys
    }
    best_model = fitted
    best_weighted_sum = math.inf
    covariance = np.full((len(names), len(names)), np.nan)
    for iteration in range(maximum_iterations):
        calculated, jacobian = evaluate_erham_observations(
            fitted,
            observations,
            parameter_names=names,
            workers=workers,
            reference_blocks=reference_blocks,
        )
        observed, sigma, fitted_calculated, fitted_jacobian = _fit_arrays(
            observations, calculated, jacobian, collapse_blends
        )
        residual = observed - fitted_calculated
        weighted_jacobian = fitted_jacobian / sigma[:, None]
        weighted_residual = residual / sigma
        if priors:
            coefficients = erham_model_coefficients(fitted)
            prior_jacobian = np.zeros((len(priors), len(names)))
            prior_residual = np.empty(len(priors))
            for row, (name, (center, prior_sigma)) in enumerate(priors.items()):
                prior_jacobian[row, names.index(name)] = 1.0 / prior_sigma
                prior_residual[row] = (center - coefficients[name]) / prior_sigma
            weighted_jacobian = np.vstack((weighted_jacobian, prior_jacobian))
            weighted_residual = np.concatenate((weighted_residual, prior_residual))
        weighted_sum = float(np.dot(weighted_residual, weighted_residual))
        if weighted_sum < best_weighted_sum:
            best_weighted_sum = weighted_sum
            best_model = fitted
        scales = np.linalg.norm(weighted_jacobian, axis=0)
        scales[scales == 0.0] = 1.0
        scaled = weighted_jacobian / scales[None, :]
        step_scaled, _residuals, rank, singular = np.linalg.lstsq(
            scaled, weighted_residual, rcond=svd_rcond
        )
        step = step_scaled / scales
        values = erham_model_coefficients(fitted)
        accepted = None
        accepted_factor = 0.0
        # A full Gauss-Newton step can cross an avoided crossing or make a
        # rotational constant nonphysical.  Backtracking keeps the iteration
        # on the same assigned branch and is independent of the molecule and
        # of the chosen operator set.
        for backtrack in range(30):
            factor = 0.5**backtrack
            try:
                candidate = replace_erham_coefficients(
                    fitted,
                    {
                        name: values[name] + factor * step[index]
                        for index, name in enumerate(names)
                    },
                )
                candidate_calculated, _ = evaluate_erham_observations(
                    candidate,
                    observations,
                    workers=workers,
                    reference_blocks=reference_blocks,
                )
                candidate_observed, candidate_sigma, candidate_prediction, _ = _fit_arrays(
                    observations,
                    candidate_calculated,
                    np.empty((len(observations), 0)),
                    collapse_blends,
                )
                candidate_sum = float(
                    np.dot(
                        (candidate_observed - candidate_prediction) / candidate_sigma,
                        (candidate_observed - candidate_prediction) / candidate_sigma,
                    )
                )
                if priors:
                    candidate_coefficients = erham_model_coefficients(candidate)
                    candidate_sum += sum(
                        ((candidate_coefficients[name] - center) / prior_sigma) ** 2
                        for name, (center, prior_sigma) in priors.items()
                    )
            except (ValueError, np.linalg.LinAlgError):
                continue
            if candidate_sum < weighted_sum:
                accepted, accepted_factor = candidate, factor
                break
        if accepted is None:
            history.append(
                ERHAMFitIteration(
                    iteration=iteration,
                    weighted_rms=float(np.sqrt(np.mean(weighted_residual**2))),
                    unweighted_rms_MHz=float(np.sqrt(np.mean(residual**2))),
                    maximum_absolute_residual_MHz=float(np.max(np.abs(residual))),
                    step_norm=0.0,
                )
            )
            break
        fitted = accepted
        history.append(
            ERHAMFitIteration(
                iteration=iteration,
                weighted_rms=float(np.sqrt(np.mean(weighted_residual**2))),
                unweighted_rms_MHz=float(np.sqrt(np.mean(residual**2))),
                maximum_absolute_residual_MHz=float(np.max(np.abs(residual))),
                step_norm=float(accepted_factor * np.linalg.norm(step_scaled)),
            )
        )
        if rank:
            inverse = np.linalg.pinv(scaled.T @ scaled, rcond=svd_rcond)
            covariance = inverse / scales[:, None] / scales[None, :]
        if accepted_factor * np.linalg.norm(step_scaled) <= relative_step_tolerance:
            break
    calculated, _jacobian = evaluate_erham_observations(
        fitted,
        observations,
        parameter_names=(),
        workers=workers,
        reference_blocks=reference_blocks,
    )
    final_jacobian = np.empty((len(observations), 0))
    objective_observed, objective_sigma, objective_calculated, _ = _fit_arrays(
        observations, calculated, final_jacobian, collapse_blends
    )
    residual = np.asarray([row.frequency_MHz for row in observations]) - calculated
    objective_residual = objective_observed - objective_calculated
    final_weighted_sum = float(
        np.dot(objective_residual / objective_sigma, objective_residual / objective_sigma)
    )
    if priors:
        coefficients = erham_model_coefficients(fitted)
        final_weighted_sum += sum(
            ((coefficients[name] - center) / prior_sigma) ** 2
            for name, (center, prior_sigma) in priors.items()
        )
    if final_weighted_sum < best_weighted_sum:
        best_model = fitted
    if best_model is not fitted:
        calculated, _jacobian = evaluate_erham_observations(
            best_model,
            observations,
            parameter_names=(),
            workers=workers,
            reference_blocks=reference_blocks,
        )
        residual = np.asarray([row.frequency_MHz for row in observations]) - calculated
    return ERHAMFitResult(
        best_model, calculated, residual, names, covariance, tuple(history)
    )


def fit_erham_mixed(
    model: ERHAMModel,
    observations: Sequence[ERHAMObservation],
    linear_parameter_names: Sequence[str],
    motion_variables: Sequence[ERHAMMotionVariable],
    *,
    workers: int = 1,
    maximum_iterations: int = 2,
    damping: float = 0.5,
    svd_rcond: float = 1.0e-12,
    collapse_blends: bool = False,
    priors: Mapping[str, tuple[float, float]] | None = None,
) -> ERHAMFitResult:
    """Jointly refine linear coefficients and linked rho-axis parameters."""
    observations = tuple(observations)
    linear_names = tuple(linear_parameter_names)
    motion_variables = tuple(motion_variables)
    if not 0.0 < damping <= 1.0:
        raise ValueError("mixed-fit damping must lie in (0,1]")
    all_names = linear_names + tuple(item.name for item in motion_variables)
    priors = {} if priors is None else dict(priors)
    if set(priors) - set(all_names):
        raise ValueError("ERHAM mixed priors must refer to fitted parameters")
    if any(sigma <= 0.0 or not math.isfinite(sigma) for _value, sigma in priors.values()):
        raise ValueError("ERHAM prior uncertainties must be positive and finite")
    fitted = model
    best_model = fitted
    best_weighted_sum = math.inf
    history: list[ERHAMFitIteration] = []
    covariance = np.full(
        (len(linear_names) + len(motion_variables),) * 2, np.nan
    )
    for iteration in range(maximum_iterations):
        calculated, linear_jacobian, blocks = evaluate_erham_observations(
            fitted, observations, parameter_names=linear_names, workers=workers, _return_blocks=True
        )
        motion_jacobian = erham_motion_jacobian(
            fitted, observations, motion_variables, workers=workers, _blocks=blocks
        )
        jacobian = np.column_stack((linear_jacobian, motion_jacobian))
        observed, sigma, fitted_calculated, fitted_jacobian = _fit_arrays(
            observations, calculated, jacobian, collapse_blends
        )
        residual = observed - fitted_calculated
        weighted_jacobian = fitted_jacobian / sigma[:, None]
        weighted_residual = residual / sigma
        if priors:
            coefficients = erham_model_coefficients(fitted)
            prior_jacobian = np.zeros((len(priors), len(all_names)))
            prior_residual = np.empty(len(priors))
            variable_lookup = {item.name: item for item in motion_variables}
            for row, (name, (center, prior_sigma)) in enumerate(priors.items()):
                prior_jacobian[row, all_names.index(name)] = 1.0 / prior_sigma
                current = (
                    coefficients[name] if name in coefficients
                    else erham_motion_coordinate(fitted, variable_lookup[name])
                )
                prior_residual[row] = (center - current) / prior_sigma
            weighted_jacobian = np.vstack((weighted_jacobian, prior_jacobian))
            weighted_residual = np.concatenate((weighted_residual, prior_residual))
        weighted_sum = float(np.dot(weighted_residual, weighted_residual))
        if weighted_sum < best_weighted_sum:
            best_weighted_sum = weighted_sum
            best_model = fitted
        scales = np.linalg.norm(weighted_jacobian, axis=0)
        scales[scales == 0.0] = 1.0
        scaled = weighted_jacobian / scales[None, :]
        step_scaled, _residuals, rank, _singular = np.linalg.lstsq(
            scaled, weighted_residual, rcond=svd_rcond
        )
        step = damping * step_scaled / scales
        values = erham_model_coefficients(fitted)
        fitted = replace_erham_coefficients(
            fitted,
            {
                name: values[name] + step[index]
                for index, name in enumerate(linear_names)
            },
        )
        for offset, variable in enumerate(motion_variables, start=len(linear_names)):
            # A finite local step keeps the Fourier derivative on its tracked
            # branch; larger changes are accumulated over later iterations.
            limit = 100.0 * variable.finite_step
            amount = float(np.clip(step[offset], -limit, limit))
            fitted = shift_erham_motions(fitted, variable, amount)
        history.append(
            ERHAMFitIteration(
                iteration=iteration,
                weighted_rms=float(np.sqrt(np.mean(weighted_residual**2))),
                unweighted_rms_MHz=float(np.sqrt(np.mean(residual**2))),
                maximum_absolute_residual_MHz=float(np.max(np.abs(residual))),
                step_norm=float(np.linalg.norm(step_scaled)),
            )
        )
        if rank:
            inverse = np.linalg.pinv(scaled.T @ scaled, rcond=svd_rcond)
            covariance = inverse / scales[:, None] / scales[None, :]
    calculated, _ = evaluate_erham_observations(fitted, observations, workers=workers)
    final_jacobian = np.empty((len(observations), 0))
    objective_observed, objective_sigma, objective_calculated, _ = _fit_arrays(
        observations, calculated, final_jacobian, collapse_blends
    )
    residual = np.asarray([row.frequency_MHz for row in observations]) - calculated
    objective_residual = objective_observed - objective_calculated
    final_weighted_sum = float(
        np.dot(objective_residual / objective_sigma, objective_residual / objective_sigma)
    )
    if priors:
        coefficients = erham_model_coefficients(fitted)
        variable_lookup = {item.name: item for item in motion_variables}
        for name, (center, prior_sigma) in priors.items():
            current = (
                coefficients[name] if name in coefficients
                else erham_motion_coordinate(fitted, variable_lookup[name])
            )
            final_weighted_sum += ((current - center) / prior_sigma) ** 2
    if final_weighted_sum < best_weighted_sum:
        best_model = fitted
    if best_model is not fitted:
        calculated, _ = evaluate_erham_observations(best_model, observations, workers=workers)
        residual = np.asarray([row.frequency_MHz for row in observations]) - calculated
    return ERHAMFitResult(
        best_model,
        calculated,
        residual,
        all_names,
        covariance,
        tuple(history),
    )


__all__ = [
    "ERHAM_FIT_RESULT_SCHEMA", "ERHAMFitIteration", "ERHAMFitResult",
    "ERHAMMotionVariable", "ERHAMObservation", "erham_model_coefficients",
    "collapse_erham_blends", "erham_motion_coordinate", "erham_motion_jacobian", "evaluate_erham_observations", "fit_erham_linear",
    "fit_erham_mixed", "replace_erham_coefficients", "shift_erham_motions",
]
