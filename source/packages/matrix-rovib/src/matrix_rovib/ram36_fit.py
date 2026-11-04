"""Robust nonlinear least squares for table-driven RAM36 models."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from scipy.linalg import qr


RAM36_FIT_RESULT_SCHEMA = "matrix.lock.ram36_fit_result.v1"
RAM36_FIT_CHECKPOINT_SCHEMA = "matrix.lock.ram36_fit_checkpoint.v1"


def collapse_ram36_blends(
    component_values: np.ndarray,
    frequency_ids: Sequence[str],
    *,
    component_weights: Sequence[float] | None = None,
    component_jacobian: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, tuple[str, ...]]:
    """Collapse transition components into intensity-weighted line frequencies."""

    values = np.asarray(component_values, dtype=float)
    if values.ndim != 1 or len(values) != len(frequency_ids):
        raise ValueError("RAM36 blend values and frequency IDs must share one dimension")
    weights = (
        np.ones(len(values), dtype=float)
        if component_weights is None
        else np.asarray(component_weights, dtype=float)
    )
    if weights.shape != values.shape or not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("RAM36 blend weights must be finite and positive")
    jacobian = None if component_jacobian is None else np.asarray(component_jacobian, dtype=float)
    if jacobian is not None and (jacobian.ndim != 2 or jacobian.shape[0] != len(values)):
        raise ValueError("RAM36 component Jacobian must have one row per component")

    order = tuple(dict.fromkeys(str(value) for value in frequency_ids))
    indices = {value: index for index, value in enumerate(order)}
    totals = np.zeros(len(order))
    normalizers = np.zeros(len(order))
    collapsed_jacobian = np.zeros((len(order), jacobian.shape[1])) if jacobian is not None else None
    for row, frequency_id in enumerate(frequency_ids):
        target = indices[str(frequency_id)]
        totals[target] += weights[row] * values[row]
        normalizers[target] += weights[row]
        if collapsed_jacobian is not None:
            collapsed_jacobian[target] += weights[row] * jacobian[row]
    totals /= normalizers
    if collapsed_jacobian is not None:
        collapsed_jacobian /= normalizers[:, None]
    return totals, collapsed_jacobian, order


def _robust_weights(residual: np.ndarray, loss: str) -> np.ndarray:
    absolute = np.abs(residual)
    if loss == "linear":
        return np.ones_like(residual)
    if loss == "huber":
        return np.where(absolute <= 1.0, 1.0, np.sqrt(1.0 / np.maximum(absolute, 1.0e-300)))
    if loss == "soft_l1":
        return (1.0 + residual * residual) ** -0.25
    raise ValueError(f"unsupported RAM36 fit loss: {loss}")


def _objective(residual: np.ndarray, loss: str) -> float:
    squared = residual * residual
    if loss == "linear":
        return float(0.5 * np.sum(squared))
    if loss == "huber":
        absolute = np.abs(residual)
        return float(np.sum(np.where(absolute <= 1.0, 0.5 * squared, absolute - 0.5)))
    if loss == "soft_l1":
        return float(np.sum(np.sqrt(1.0 + squared) - 1.0))
    raise ValueError(f"unsupported RAM36 fit loss: {loss}")


def _independent_columns(jacobian: np.ndarray, rcond: float) -> tuple[np.ndarray, np.ndarray, int]:
    if jacobian.ndim != 2 or jacobian.shape[1] == 0:
        raise ValueError("RAM36 fit Jacobian must be a nonempty matrix")
    _q, triangular, pivots = qr(jacobian, mode="economic", pivoting=True, check_finite=False)
    diagonal = np.abs(np.diag(triangular))
    rank = int(np.count_nonzero(diagonal > (diagonal[0] * rcond if len(diagonal) else 0.0)))
    selected = np.sort(np.asarray(pivots[:rank], dtype=int))
    singular = np.linalg.svd(jacobian[:, selected], compute_uv=False) if rank else np.empty(0)
    return selected, singular, rank


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            Path(temporary).unlink()
        except FileNotFoundError:
            pass
        raise


def fit_ram36_nonlinear(
    initial_values: Mapping[str, float],
    observed_MHz: Sequence[float],
    uncertainty_MHz: Sequence[float],
    evaluator: Callable[[Mapping[str, float], bool], tuple[np.ndarray, np.ndarray | None]],
    *,
    parameter_scales: Mapping[str, float] | None = None,
    parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
    parameter_priors: Mapping[str, tuple[float, float]] | None = None,
    prior_covariance: Sequence[Sequence[float]] | None = None,
    training_mask: Sequence[bool] | None = None,
    loss: str = "linear",
    maximum_iterations: int = 12,
    rank_rcond: float = 1.0e-10,
    trust_radius: float = 1.0,
    jacobian_refresh_interval: int = 3,
    checkpoint: Path | None = None,
    resume: bool = False,
    accepted_callback: Callable[[Mapping[str, float]], None] | None = None,
) -> dict[str, Any]:
    """Fit any RAM36 predictor with analytic physical-parameter derivatives.

    ``evaluator(parameters, derivatives)`` returns line predictions and, when
    requested, a Jacobian whose columns follow ``initial_values`` insertion
    order. Rank-deficient columns are removed by pivoted QR at every refresh.
    Accepted steps update the Jacobian with a good-Broyden rank-one correction.
    ``accepted_callback`` commits external state-tracking data only after the
    line search has accepted the corresponding parameter point.
    """

    names = tuple(initial_values)
    if not names or len(names) != len(set(names)):
        raise ValueError("RAM36 fit parameter names must be nonempty and unique")
    initial = np.asarray([float(initial_values[name]) for name in names])
    scales = np.asarray(
        [
            float(parameter_scales[name])
            if parameter_scales and name in parameter_scales
            else max(abs(initial[index]), 1.0)
            for index, name in enumerate(names)
        ]
    )
    if not np.all(np.isfinite(initial)) or not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("RAM36 fit values and scales must be finite and scales positive")
    bounds = parameter_bounds or {}
    lower = np.asarray([float(bounds.get(name, (-math.inf, math.inf))[0]) for name in names])
    upper = np.asarray([float(bounds.get(name, (-math.inf, math.inf))[1]) for name in names])
    if np.any(lower >= upper) or np.any(initial < lower) or np.any(initial > upper):
        raise ValueError("RAM36 fit bounds must contain the initial values")
    lower_coordinates = (lower - initial) / scales
    upper_coordinates = (upper - initial) / scales
    observed = np.asarray(observed_MHz, dtype=float)
    sigma = np.asarray(uncertainty_MHz, dtype=float)
    if observed.ndim != 1 or sigma.shape != observed.shape or np.any(sigma <= 0.0):
        raise ValueError("RAM36 observations require one positive uncertainty each")
    training = (
        np.ones(len(observed), dtype=bool)
        if training_mask is None
        else np.asarray(training_mask, dtype=bool)
    )
    if training.shape != observed.shape or not np.any(training):
        raise ValueError("RAM36 fit needs a valid nonempty training mask")
    if maximum_iterations < 1 or trust_radius <= 0.0 or jacobian_refresh_interval < 1:
        raise ValueError("RAM36 fit iteration and trust controls must be positive")

    priors = parameter_priors or {}
    unknown_priors = set(priors) - set(names)
    if unknown_priors:
        raise ValueError(f"RAM36 priors reference unknown parameters: {sorted(unknown_priors)}")
    prior_indices = np.asarray(
        [index for index, name in enumerate(names) if name in priors], dtype=int
    )
    prior_values = np.asarray([float(priors[names[index]][0]) for index in prior_indices])
    prior_sigmas = np.asarray([float(priors[names[index]][1]) for index in prior_indices])
    if (
        np.any(~np.isfinite(prior_values))
        or np.any(~np.isfinite(prior_sigmas))
        or np.any(prior_sigmas <= 0.0)
    ):
        raise ValueError("RAM36 priors require finite values and positive sigmas")
    if len(prior_indices):
        prior_covariance_array = (
            np.diag(prior_sigmas * prior_sigmas)
            if prior_covariance is None
            else np.asarray(prior_covariance, dtype=float)
        )
        if (
            prior_covariance_array.shape != (len(prior_indices), len(prior_indices))
            or not np.all(np.isfinite(prior_covariance_array))
            or not np.allclose(prior_covariance_array, prior_covariance_array.T)
        ):
            raise ValueError(
                "RAM36 prior covariance must be finite, symmetric, and match the priors"
            )
        prior_eigenvalues, prior_eigenvectors = np.linalg.eigh(prior_covariance_array)
        if np.any(prior_eigenvalues <= 0.0):
            raise ValueError("RAM36 prior covariance must be positive definite")
        prior_whitener = (prior_eigenvectors * prior_eigenvalues**-0.5) @ prior_eigenvectors.T
        prior_jacobian = np.zeros((len(prior_indices), len(names)))
        prior_jacobian[:, prior_indices] = prior_whitener @ np.diag(scales[prior_indices])
    else:
        if prior_covariance is not None:
            raise ValueError("RAM36 prior covariance requires parameter_priors")
        prior_covariance_array = np.empty((0, 0))
        prior_whitener = np.empty((0, 0))
        prior_jacobian = np.empty((0, len(names)))
    fingerprint_payload = {
        "parameter_names": names,
        "initial": initial.tolist(),
        "scales": scales.tolist(),
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "observed": observed.tolist(),
        "sigma": sigma.tolist(),
        "training": training.tolist(),
        "loss": loss,
        "prior_indices": prior_indices.tolist(),
        "prior_values": prior_values.tolist(),
        "prior_covariance": prior_covariance_array.tolist(),
    }
    fit_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    coordinates = np.zeros(len(names))
    prior_history: list[dict[str, Any]] = []
    completed_before = 0
    saved_predictions: np.ndarray | None = None
    saved_jacobian: np.ndarray | None = None
    saved_accepted_since_refresh = 0
    if resume:
        if checkpoint is None or not checkpoint.exists():
            raise ValueError("RAM36 fit resume requires an existing checkpoint")
        saved = json.loads(checkpoint.read_text())
        if saved.get("schema") != RAM36_FIT_CHECKPOINT_SCHEMA or saved.get(
            "parameter_names"
        ) != list(names):
            raise ValueError("RAM36 fit checkpoint is incompatible with this model")
        if saved.get("fit_fingerprint", fit_fingerprint) != fit_fingerprint:
            raise ValueError(
                "RAM36 fit checkpoint belongs to different data, bounds, loss, or priors"
            )
        if "initial_values" in saved and not np.array_equal(
            np.asarray(saved["initial_values"], dtype=float), initial
        ):
            raise ValueError("RAM36 fit checkpoint uses different initial parameter values")
        if "parameter_scales" in saved and not np.array_equal(
            np.asarray(saved["parameter_scales"], dtype=float), scales
        ):
            raise ValueError("RAM36 fit checkpoint uses different parameter scales")
        coordinates = np.asarray(saved["scaled_coordinates"], dtype=float)
        prior_history = list(saved.get("history", []))
        completed_before = int(saved.get("completed_iterations", 0))
        trust_radius = float(
            saved.get(
                "trust_radius",
                prior_history[-1].get("trust_radius", trust_radius)
                if prior_history
                else trust_radius,
            )
        )
        if "predictions_MHz" in saved and "jacobian_scaled" in saved:
            saved_predictions = np.asarray(saved["predictions_MHz"], dtype=float)
            saved_jacobian = np.asarray(saved["jacobian_scaled"], dtype=float)
            saved_accepted_since_refresh = int(saved.get("accepted_since_refresh", 0))

    def parameters(at: np.ndarray) -> dict[str, float]:
        values = initial + scales * at
        return dict(zip(names, values, strict=True))

    def prior_residual(at: np.ndarray) -> np.ndarray:
        if not len(prior_indices):
            return np.empty(0)
        physical = initial + scales * at
        return prior_whitener @ (physical[prior_indices] - prior_values)

    if saved_predictions is not None and saved_jacobian is not None:
        predictions = saved_predictions
        jacobian = saved_jacobian
    else:
        predictions, jacobian_physical = evaluator(parameters(coordinates), True)
        predictions = np.asarray(predictions, dtype=float)
        if predictions.shape != observed.shape or jacobian_physical is None:
            raise ValueError("RAM36 evaluator returned an invalid prediction or missing Jacobian")
        jacobian = np.asarray(jacobian_physical, dtype=float) * scales[None, :]
    if predictions.shape != observed.shape:
        raise ValueError("RAM36 checkpoint predictions do not match the observation set")
    if jacobian.shape != (len(observed), len(names)):
        raise ValueError("RAM36 evaluator Jacobian has an invalid shape")

    history = []
    accepted_since_refresh = saved_accepted_since_refresh
    selected = np.arange(len(names))
    singular = np.empty(0)
    for iteration in range(maximum_iterations):
        residual = (predictions - observed) / sigma
        weights = _robust_weights(residual[training], loss)
        data_jacobian = weights[:, None] * jacobian[training] / sigma[training, None]
        weighted_jacobian = (
            np.vstack((data_jacobian, prior_jacobian)) if len(prior_indices) else data_jacobian
        )
        selected, singular, rank = _independent_columns(weighted_jacobian, rank_rcond)
        if rank == 0:
            raise RuntimeError("RAM36 fit Jacobian has zero numerical rank")
        active_jacobian = weighted_jacobian[:, selected]
        active_residual = np.concatenate(
            (weights * residual[training], prior_residual(coordinates))
        )
        step_active = np.linalg.lstsq(active_jacobian, -active_residual, rcond=rank_rcond)[0]
        step = np.zeros(len(names))
        step[selected] = step_active
        maximum_step = float(np.max(np.abs(step), initial=0.0))
        if maximum_step > trust_radius:
            step *= trust_radius / maximum_step
        before = _objective(residual[training], loss) + 0.5 * float(
            prior_residual(coordinates) @ prior_residual(coordinates)
        )
        previous_predictions = predictions.copy()
        accepted = False
        accepted_damping = None
        for damping in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
            candidate_coordinates = np.clip(
                coordinates + damping * step, lower_coordinates, upper_coordinates
            )
            candidate_predictions, _unused = evaluator(parameters(candidate_coordinates), False)
            candidate_predictions = np.asarray(candidate_predictions, dtype=float)
            candidate_residual = (candidate_predictions - observed) / sigma
            candidate_prior = prior_residual(candidate_coordinates)
            candidate_objective = _objective(candidate_residual[training], loss) + 0.5 * float(
                candidate_prior @ candidate_prior
            )
            if candidate_objective < before:
                accepted = True
                accepted_damping = damping
                break
        if not accepted:
            history.append(
                {
                    "iteration": completed_before + iteration,
                    "accepted": False,
                    "objective": before,
                    "rank": rank,
                    "trust_radius": trust_radius,
                }
            )
            break
        accepted_step = candidate_coordinates - coordinates
        coordinates = candidate_coordinates
        predictions = candidate_predictions
        if accepted_callback is not None:
            accepted_callback(parameters(coordinates))
        predicted = residual + jacobian @ accepted_step / sigma
        predicted_prior = prior_residual(coordinates)
        predicted_reduction = before - (
            _objective(predicted[training], loss) + 0.5 * float(predicted_prior @ predicted_prior)
        )
        actual_reduction = before - candidate_objective
        ratio = actual_reduction / predicted_reduction if predicted_reduction > 0.0 else None
        denominator = float(accepted_step @ accepted_step)
        if denominator > 1.0e-20:
            jacobian += (
                np.outer(
                    predictions - previous_predictions - jacobian @ accepted_step,
                    accepted_step,
                )
                / denominator
            )
        accepted_since_refresh += 1
        refresh = (
            accepted_since_refresh >= jacobian_refresh_interval
            or ratio is None
            or ratio < 0.25
            or ratio > 2.0
        )
        if refresh:
            refreshed_predictions, refreshed_jacobian = evaluator(parameters(coordinates), True)
            predictions = np.asarray(refreshed_predictions, dtype=float)
            if refreshed_jacobian is None:
                raise ValueError("RAM36 evaluator omitted a requested refreshed Jacobian")
            jacobian = np.asarray(refreshed_jacobian, dtype=float) * scales[None, :]
            accepted_since_refresh = 0
        trust_radius = (
            min(5.0, trust_radius * 1.5)
            if ratio is not None and ratio >= 0.75 and accepted_damping >= 0.5
            else max(1.0e-4, trust_radius * 0.5 if ratio is None or ratio < 0.25 else trust_radius)
        )
        history.append(
            {
                "iteration": completed_before + iteration,
                "accepted": True,
                "accepted_damping": accepted_damping,
                "objective_before": before,
                "objective_after": candidate_objective,
                "actual_to_predicted_reduction_ratio": ratio,
                "rank": rank,
                "refreshed_jacobian": refresh,
                "trust_radius": trust_radius,
            }
        )
        if checkpoint is not None:
            _atomic_json(
                checkpoint,
                {
                    "schema": RAM36_FIT_CHECKPOINT_SCHEMA,
                    "fit_fingerprint": fit_fingerprint,
                    "parameter_names": list(names),
                    "initial_values": initial.tolist(),
                    "parameter_scales": scales.tolist(),
                    "scaled_coordinates": coordinates.tolist(),
                    "completed_iterations": completed_before + iteration + 1,
                    "trust_radius": trust_radius,
                    "accepted_since_refresh": accepted_since_refresh,
                    "predictions_MHz": predictions.tolist(),
                    "jacobian_scaled": jacobian.tolist(),
                    "prior_parameter_names": [names[index] for index in prior_indices],
                    "history": prior_history + history,
                },
            )
        if np.max(np.abs(accepted_step), initial=0.0) < 1.0e-8:
            break

    final_residual = predictions - observed
    final_standardized = final_residual[training] / sigma[training]
    final_weights = _robust_weights(final_standardized, loss)
    data_final = final_weights[:, None] * jacobian[training] / sigma[training, None]
    augmented_final = np.vstack((data_final, prior_jacobian)) if len(prior_indices) else data_final
    _data_selected, _data_singular, data_only_rank = _independent_columns(data_final, rank_rcond)
    selected, singular, rank = _independent_columns(augmented_final, rank_rcond)
    weighted_final = augmented_final[:, selected]
    covariance_scaled = np.linalg.pinv(weighted_final.T @ weighted_final, rcond=rank_rcond)
    covariance = np.zeros((len(names), len(names)))
    covariance[np.ix_(selected, selected)] = (
        scales[selected, None] * covariance_scaled * scales[None, selected]
    )
    fitted = parameters(coordinates)
    rejected = [name for index, name in enumerate(names) if index not in set(selected.tolist())]
    validation = ~training
    return {
        "schema": RAM36_FIT_RESULT_SCHEMA,
        "fit_fingerprint": fit_fingerprint,
        "parameter_names": list(names),
        "parameters": fitted,
        "standard_errors": dict(
            zip(names, np.sqrt(np.maximum(np.diag(covariance), 0.0)), strict=True)
        ),
        "covariance": covariance.tolist(),
        "rank": rank,
        "data_only_rank": data_only_rank,
        "selected_parameters": [names[index] for index in selected],
        "rejected_parameters": rejected,
        "singular_values": singular.tolist(),
        "training_rms_MHz": float(np.sqrt(np.mean(final_residual[training] ** 2))),
        "validation_rms_MHz": (
            float(np.sqrt(np.mean(final_residual[validation] ** 2))) if np.any(validation) else None
        ),
        "prior_chi_square": float(prior_residual(coordinates) @ prior_residual(coordinates)),
        "prior_parameter_names": [names[index] for index in prior_indices],
        "history": prior_history + history,
        "predictions_MHz": predictions.tolist(),
        "residuals_MHz": final_residual.tolist(),
    }


__all__ = [
    "RAM36_FIT_CHECKPOINT_SCHEMA",
    "RAM36_FIT_RESULT_SCHEMA",
    "collapse_ram36_blends",
    "fit_ram36_nonlinear",
]
