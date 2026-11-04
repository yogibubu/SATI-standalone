"""Prediction budgets with separate statistical errors and numerical bounds."""

from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar


def _covariance(value, size, label):
    matrix = np.asarray(value, float)
    if (
        matrix.shape != (size, size)
        or not np.all(np.isfinite(matrix))
        or not np.allclose(matrix, matrix.T, rtol=1.0e-10, atol=0.0)
    ):
        raise ValueError(f"invalid {label} covariance")
    eigen = np.linalg.eigvalsh(matrix)
    if np.min(eigen, initial=0.0) < -128 * np.finfo(float).eps * max(size, 1) * max(
        np.finfo(float).tiny, np.max(abs(eigen), initial=0.0)
    ):
        raise ValueError(f"{label} covariance is not positive semidefinite")
    return (matrix + matrix.T) / 2


def prediction_uncertainty_budget(
    predictions,
    jacobian,
    parameter_covariance,
    *,
    numerical_checks=None,
    validation_residuals=None,
    validation_residual_covariance=None,
    null_space=None,
    minimum_validation_count=3,
):
    """Propagate fitted covariance; infer discrepancy only from held-out data.

    Numerical differences are conservative additive envelopes, not random
    standard deviations. The discrepancy model is explicitly iid Gaussian;
    its estimated variance is conditional on held-out residual covariance.
    Its calibration uncertainty is not folded into the displayed intervals.
    """
    if minimum_validation_count < 3:
        raise ValueError("at least three validation observations are required")
    predictions = np.asarray(predictions, float)
    j = np.asarray(jacobian, float)
    if (
        predictions.ndim != 1
        or j.ndim != 2
        or len(j) != len(predictions)
        or not np.all(np.isfinite(predictions))
        or not np.all(np.isfinite(j))
    ):
        raise ValueError("invalid prediction sensitivity matrix")
    c = _covariance(parameter_covariance, j.shape[1], "parameter")
    variance = np.maximum(np.einsum("ij,jk,ik->i", j, c, j, optimize=True), 0.0)
    identified = np.ones(len(predictions), bool)
    if null_space is not None:
        null = np.asarray(null_space, float)
        if null.ndim != 2 or null.shape[0] != j.shape[1]:
            raise ValueError("invalid parameter null space")
        sensitivity = np.linalg.norm(j @ null, axis=1)
        identified = sensitivity <= 1.0e-10 * np.maximum(1.0, np.linalg.norm(j, axis=1))
    bounds = None
    checks = {}
    if numerical_checks is not None:
        if not numerical_checks:
            raise ValueError("at least one numerical comparison is required")
        bounds = np.zeros(len(predictions))
        for axis, delta in numerical_checks.items():
            delta = np.asarray(delta, float)
            if delta.shape != predictions.shape or not np.all(np.isfinite(delta)):
                raise ValueError("invalid numerical error comparison")
            envelope = abs(delta)
            bounds += envelope
            checks[axis] = envelope.tolist()
    model = dict(
        status="unavailable",
        reason="independent validation residuals are required",
        assumption="iid Gaussian model discrepancy",
    )
    scatter = None
    if validation_residuals is not None:
        residuals = np.asarray(validation_residuals, float)
        if residuals.ndim != 1 or not np.all(np.isfinite(residuals)):
            raise ValueError("invalid validation residuals")
        if len(residuals) < minimum_validation_count:
            model.update(
                reason="insufficient independent validation observations",
                validation_count=len(residuals),
            )
        else:
            known = _covariance(
                validation_residual_covariance, len(residuals), "validation residual"
            )
            eigen, q = np.linalg.eigh(known)
            z = q.T @ residuals

            def objective(variance):
                diagonal = eigen + variance
                if np.any(diagonal <= 0):
                    return np.inf
                return float(np.sum(np.log(diagonal) + z * z / diagonal))

            scale = max(float(np.mean(residuals**2)), float(np.max(eigen)), np.finfo(float).tiny)
            # Optimize a dimensionless variance so small MHz scales stay resolved.
            optimized = minimize_scalar(
                lambda a: objective(a * scale),
                bounds=(0.0, max(10.0, 10 * np.mean(residuals**2) / scale)),
                method="bounded",
                options=dict(xatol=1.0e-12),
            )
            candidate = float(optimized.x * scale)
            scatter = 0.0 if objective(0.0) <= objective(candidate) else candidate
            precision = np.linalg.pinv(known + scatter * np.eye(len(residuals)))
            one = np.ones(len(residuals))
            bias = float(one @ precision @ residuals / (one @ precision @ one))
            model = dict(
                status="estimated_from_validation",
                validation_count=len(residuals),
                variance_MHz2=scatter,
                standard_uncertainty_MHz=float(np.sqrt(scatter)),
                weighted_bias_MHz=bias,
                bias_corrected=False,
                assumption="iid zero-mean Gaussian model discrepancy; no training-residual calibration",
                interval_scope="conditional on estimated discrepancy variance; calibration uncertainty not included",
            )
    parameter_sigma = np.sqrt(variance)
    total_sigma = None if scatter is None else np.sqrt(variance + scatter)
    intervals = None
    if total_sigma is not None and bounds is not None:
        width = 1.959963984540054 * total_sigma + bounds
        intervals = [
            [float(p - d), float(p + d)] if good else None
            for p, d, good in zip(predictions, width, identified, strict=True)
        ]
    return dict(
        schema="matrix.lock.prediction_uncertainty_budget.v1",
        central_predictions_MHz=predictions.tolist(),
        parameter_standard_uncertainty_MHz=[
            float(s) if good else None for s, good in zip(parameter_sigma, identified, strict=True)
        ],
        identifiable_predictions=identified.tolist(),
        numerical_error_bound_MHz=None if bounds is None else bounds.tolist(),
        numerical_comparisons=checks,
        numerical_interpretation="sum of absolute refinement differences; not a probabilistic error bar or rigorous bound",
        model_discrepancy=model,
        total_statistical_standard_uncertainty_MHz=None
        if total_sigma is None
        else [float(s) if good else None for s, good in zip(total_sigma, identified, strict=True)],
        conditional_95_percent_intervals_MHz=intervals,
        complete=bool(bounds is not None and scatter is not None and np.all(identified)),
    )


def lam_prediction_budget(request, fit, physical_jacobian, *, training_whitener, training_jacobian):
    """Build the held-out residual covariance, preserving train/validation correlations."""
    observations = request["observations"]
    j = np.asarray(physical_jacobian, float)
    covariance = np.asarray(fit["covariance"], float)
    options = request.get("uncertainty_budget", {})
    training = np.array(
        [i for i, o in enumerate(observations) if o.get("role", "training") == "training"], int
    )
    validation = np.array(
        [i for i, o in enumerate(observations) if o.get("role", "training") == "validation"], int
    )
    sigma = np.array([o["uncertainty_MHz"] / np.sqrt(o.get("weight", 1.0)) for o in observations])
    raw = request.get("covariance_MHz2")
    experimental = None
    if raw is not None:
        w = np.sqrt([o.get("weight", 1.0) for o in observations])
        experimental = np.asarray(raw, float) / w[:, None] / w[None, :]
    residuals = None
    residual_covariance = None
    if len(validation):
        residuals = np.array([o["calculated_minus_observed_MHz"] for o in fit["comparisons"]])[
            validation
        ]
        vv = (
            np.diag(sigma[validation] ** 2)
            if experimental is None
            else experimental[np.ix_(validation, validation)]
        )
        residual_covariance = vv + j[validation] @ covariance @ j[validation].T
        if experimental is not None:
            cross = training_whitener.apply(experimental[np.ix_(training, validation)])
            shared = j[validation] @ covariance @ np.asarray(training_jacobian).T @ cross
            residual_covariance -= shared + shared.T
    null = None
    if not fit.get("success", False):
        # A stopped/deficient fit has no qualified local covariance budget.
        return dict(
            schema="matrix.lock.prediction_uncertainty_budget.v1",
            complete=False,
            reason="fit is not converged",
        )
    return prediction_uncertainty_budget(
        np.array([o["calculated_MHz"] for o in fit["comparisons"]]),
        j,
        covariance,
        numerical_checks=options.get("numerical_checks"),
        validation_residuals=residuals,
        validation_residual_covariance=residual_covariance,
        null_space=null,
        minimum_validation_count=int(options.get("minimum_validation_count", 3)),
    )


def attach_numerical_budget(budget, checks):
    """Attach the final qualification without reevaluating physical predictions."""
    from copy import deepcopy

    result = deepcopy(budget)
    if not checks or "parameter_standard_uncertainty_MHz" not in result:
        return result
    count = len(result["parameter_standard_uncertainty_MHz"])
    envelope = np.zeros(count)
    for delta in checks.values():
        a = np.asarray(delta, float)
        if a.shape != (count,) or not np.all(np.isfinite(a)):
            raise ValueError("invalid final-basis comparison")
        envelope += abs(a)
    result["numerical_error_bound_MHz"] = envelope.tolist()
    result["numerical_comparisons"] = checks
    sigma = result["total_statistical_standard_uncertainty_MHz"]
    central = result.get("central_predictions_MHz")
    if sigma is not None and central is not None:
        result["conditional_95_percent_intervals_MHz"] = [
            None
            if s is None
            else [float(p - 1.959963984540054 * s - d), float(p + 1.959963984540054 * s + d)]
            for p, s, d in zip(central, sigma, envelope, strict=True)
        ]
        result["complete"] = bool(all(result["identifiable_predictions"]))
    return result
