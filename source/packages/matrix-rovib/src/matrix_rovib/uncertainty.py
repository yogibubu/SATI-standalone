"""Correlated and nonlinear uncertainty tools for general LOCK fits."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math

import numpy as np
from matrix_numerics.least_squares import svd_covariance


HARTREE_TO_MHZ = 6.5796839204999e9


@dataclass(frozen=True)
class CorrelatedUncertainty:
    covariance: np.ndarray
    standard_errors: tuple[float, ...]
    correlation: np.ndarray
    rank: int
    condition_number: float | None


def correlated_linearized_uncertainty(
    jacobian: np.ndarray,
    *,
    observation_covariance: np.ndarray | None = None,
    rcond: float = 1.0e-10,
) -> CorrelatedUncertainty:
    """Compute the generalized least-squares covariance without diagonal assumptions."""

    design = np.asarray(jacobian, dtype=float)
    if design.ndim != 2 or not np.all(np.isfinite(design)) or rcond <= 0.0:
        raise ValueError("uncertainty analysis requires a finite matrix and positive rcond")
    if observation_covariance is None:
        whitened = design
    else:
        covariance_y = np.asarray(observation_covariance, dtype=float)
        if covariance_y.shape != (design.shape[0], design.shape[0]):
            raise ValueError("observation covariance shape does not match the Jacobian")
        if not np.allclose(covariance_y, covariance_y.T, atol=1.0e-12):
            raise ValueError("observation covariance must be symmetric")
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_y)
        if eigenvalues[0] <= 0.0:
            raise ValueError("observation covariance must be positive definite")
        whitener = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
        whitened = whitener @ design
    covariance, rank, singular = svd_covariance(whitened, rcond=rcond)
    errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    denominator = np.outer(errors, errors)
    correlation = np.zeros_like(covariance)
    np.divide(covariance, denominator, out=correlation, where=denominator > 0.0)
    condition = (
        float(singular[0] / singular[-1])
        if singular.size and rank == design.shape[1] and singular[-1] > 0.0
        else None
    )
    return CorrelatedUncertainty(
        covariance=covariance,
        standard_errors=tuple(float(value) for value in errors),
        correlation=correlation,
        rank=rank,
        condition_number=condition,
    )


def sample_correlated_parameters(
    optimum: Sequence[float],
    covariance: np.ndarray,
    *,
    sample_count: int = 1000,
    seed: int = 1729,
) -> np.ndarray:
    """Draw reproducible samples while tolerating semidefinite fit covariances."""

    center = np.asarray(optimum, dtype=float)
    matrix = np.asarray(covariance, dtype=float)
    if matrix.shape != (center.size, center.size) or sample_count < 1:
        raise ValueError("parameter covariance shape or sample count is invalid")
    if not np.allclose(matrix, matrix.T, atol=1.0e-12):
        raise ValueError("parameter covariance must be symmetric")
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    tolerance = 1.0e-12 * max(float(np.max(np.abs(eigenvalues), initial=0.0)), 1.0)
    if eigenvalues[0] < -tolerance:
        raise ValueError("parameter covariance is not positive semidefinite")
    transform = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))
    normal = np.random.default_rng(int(seed)).standard_normal((sample_count, center.size))
    return center + normal @ transform.T


def propagate_surface_model_uncertainty(
    surface_parameters: Sequence[float],
    surface_covariance: np.ndarray,
    predict_observables: Callable[[np.ndarray], Sequence[float]],
    *,
    observable_labels: Sequence[str] = (),
    sample_count: int = 500,
    seed: int = 1729,
) -> dict[str, object]:
    """Propagate a correlated QM surface model through the complete LOCK solve.

    ``surface_parameters`` may contain potential, dipole-surface and fitted
    Hamiltonian coefficients in any caller-defined order. The callback must
    run the same production solver used for the central prediction. Sampling
    is reproducible and retains observable correlations rather than reporting
    independent pointwise error bars.
    """

    center = np.asarray(surface_parameters, dtype=float).reshape(-1)
    if center.size == 0 or not np.all(np.isfinite(center)):
        raise ValueError("surface uncertainty requires finite model parameters")
    samples = sample_correlated_parameters(
        center,
        surface_covariance,
        sample_count=sample_count,
        seed=seed,
    )
    central = np.asarray(tuple(predict_observables(center.copy())), dtype=float).reshape(-1)
    if central.size == 0 or not np.all(np.isfinite(central)):
        raise ValueError("LOCK uncertainty callback returned an invalid central prediction")
    predicted = np.empty((sample_count, central.size), dtype=float)
    for index, sample in enumerate(samples):
        values = np.asarray(tuple(predict_observables(sample)), dtype=float).reshape(-1)
        if values.shape != central.shape or not np.all(np.isfinite(values)):
            raise ValueError(f"LOCK uncertainty callback failed for deterministic sample {index}")
        predicted[index] = values
    labels = tuple(str(label) for label in observable_labels)
    if labels and len(labels) != central.size:
        raise ValueError("observable labels do not match the LOCK prediction")
    covariance = np.atleast_2d(np.cov(predicted, rowvar=False, ddof=1))
    return {
        "schema": "matrix.lock.surface_uncertainty.v1",
        "sample_count": int(sample_count),
        "seed": int(seed),
        "observable_labels": list(labels),
        "central": central.tolist(),
        "mean": np.mean(predicted, axis=0).tolist(),
        "standard_deviation": np.std(predicted, axis=0, ddof=1).tolist(),
        "percentile_2_5": np.percentile(predicted, 2.5, axis=0).tolist(),
        "percentile_50": np.percentile(predicted, 50.0, axis=0).tolist(),
        "percentile_97_5": np.percentile(predicted, 97.5, axis=0).tolist(),
        "observable_covariance": covariance.tolist(),
    }


def qim_surface_parameter_distribution(
    result: object,
    *,
    include_dipole: bool = True,
) -> dict[str, object]:
    """Build the correlated LOCK parameter distribution from a LINK Qim path."""

    points = tuple(getattr(result, "points"))
    if not points:
        raise ValueError("Qim uncertainty requires at least one converged point")
    energies = np.asarray([point.energy_hartree for point in points], dtype=float)
    reference = int(np.argmin(energies))
    parameters = ((energies - energies[reference]) * HARTREE_TO_MHZ).tolist()
    labels = [f"V_MHz[{point.q_im:g}]" for point in points]
    covariance = np.zeros((len(points), len(points)), dtype=float)
    energy_sigma = [point.energy_standard_error_hartree for point in points]
    if all(value is not None for value in energy_sigma):
        variance = (HARTREE_TO_MHZ * np.asarray(energy_sigma, dtype=float)) ** 2
        covariance += np.diag(variance)
        covariance += variance[reference]
        covariance[:, reference] -= variance[reference]
        covariance[reference, :] -= variance[reference]
    elif any(value is not None for value in energy_sigma):
        raise ValueError("Qim energy uncertainty is missing at one or more points")
    if include_dipole:
        dipoles = [point.dipole_au for point in points]
        dipole_covariances = [point.dipole_covariance_au2 for point in points]
        if any(value is not None for value in dipoles):
            if not all(value is not None for value in dipoles):
                raise ValueError("Qim dipole surface is incomplete")
            if not all(value is not None for value in dipole_covariances):
                raise ValueError("Qim dipole covariance is incomplete")
            old_size = covariance.shape[0]
            dipole_values = np.asarray(dipoles, dtype=float).reshape(-1)
            parameters.extend(dipole_values.tolist())
            labels.extend(f"mu_{axis}_au[{point.q_im:g}]" for point in points for axis in "xyz")
            expanded = np.zeros((old_size + dipole_values.size,) * 2, dtype=float)
            expanded[:old_size, :old_size] = covariance
            for index, block in enumerate(dipole_covariances):
                slot = slice(old_size + 3 * index, old_size + 3 * index + 3)
                expanded[slot, slot] = np.asarray(block, dtype=float)
            covariance = expanded
    if not np.any(covariance):
        raise ValueError("LINK Qim result contains no propagated QM uncertainty")
    return {
        "parameters": parameters,
        "covariance": covariance,
        "labels": labels,
        "reference_grid_index": reference,
    }


def parametric_bootstrap(
    predicted_observations: Sequence[float],
    observation_covariance: np.ndarray,
    refit: Callable[[np.ndarray], Sequence[float]],
    *,
    sample_count: int = 200,
    seed: int = 1729,
) -> dict[str, object]:
    """Refit reproducible correlated synthetic spectra and report percentiles."""

    center = np.asarray(predicted_observations, dtype=float)
    covariance = np.asarray(observation_covariance, dtype=float)
    if covariance.shape != (center.size, center.size) or sample_count < 2:
        raise ValueError("bootstrap covariance shape or sample count is invalid")
    if not np.allclose(covariance, covariance.T, atol=1.0e-12):
        raise ValueError("bootstrap observation covariance must be symmetric")
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues[0] <= 0.0:
        raise ValueError("bootstrap observation covariance must be positive definite")
    generator = np.random.default_rng(int(seed))
    synthetic = generator.multivariate_normal(center, covariance, size=sample_count)
    fitted = np.asarray([tuple(refit(sample)) for sample in synthetic], dtype=float)
    if fitted.ndim != 2 or fitted.shape[0] != sample_count or not np.all(np.isfinite(fitted)):
        raise ValueError("bootstrap refit must return equal finite parameter vectors")
    return {
        "sample_count": sample_count,
        "seed": int(seed),
        "mean": np.mean(fitted, axis=0).tolist(),
        "standard_deviation": np.std(fitted, axis=0, ddof=1).tolist(),
        "percentile_2_5": np.percentile(fitted, 2.5, axis=0).tolist(),
        "percentile_50": np.percentile(fitted, 50.0, axis=0).tolist(),
        "percentile_97_5": np.percentile(fitted, 97.5, axis=0).tolist(),
    }


def profile_likelihood_intervals(
    objective: Callable[[np.ndarray], float],
    optimum: Sequence[float],
    covariance: np.ndarray,
    *,
    sigma_extent: float = 3.0,
    grid_points: int = 9,
) -> tuple[dict[str, object], ...]:
    """Profile each parameter while reoptimizing every nuisance parameter.

    The objective must be chi-square-like. Returned intervals use likelihood
    ratio thresholds of 1.0 and 3.841 for one profiled degree of freedom.
    """

    from scipy.optimize import minimize

    center = np.asarray(optimum, dtype=float)
    matrix = np.asarray(covariance, dtype=float)
    if matrix.shape != (center.size, center.size):
        raise ValueError("profile covariance shape does not match the optimum")
    if not math.isfinite(sigma_extent) or sigma_extent <= 0.0:
        raise ValueError("profile sigma extent must be positive and finite")
    if grid_points < 5 or grid_points % 2 == 0:
        raise ValueError("profile grid_points must be an odd integer of at least five")
    base = float(objective(center.copy()))
    if not math.isfinite(base):
        raise ValueError("profile objective is nonfinite at the optimum")
    rows = []
    for index, variance in enumerate(np.diag(matrix)):
        sigma = math.sqrt(max(float(variance), 0.0))
        if sigma == 0.0:
            rows.append(
                {
                    "parameter_index": index,
                    "grid": [float(center[index])],
                    "delta_chi_square": [0.0],
                    "interval_68": [float(center[index]), float(center[index])],
                    "interval_95": [float(center[index]), float(center[index])],
                }
            )
            continue
        grid = center[index] + sigma * np.linspace(-sigma_extent, sigma_extent, grid_points)
        nuisance = np.delete(center, index)
        profile = []

        def assemble(values: np.ndarray, fixed: float) -> np.ndarray:
            trial = np.empty_like(center)
            trial[index] = fixed
            trial[np.arange(center.size) != index] = values
            return trial

        for fixed in grid:
            if nuisance.size:
                optimized = minimize(
                    lambda values: float(objective(assemble(values, float(fixed)))),
                    nuisance,
                    method="BFGS",
                )
                if not optimized.success and not math.isfinite(float(optimized.fun)):
                    raise RuntimeError(f"profile optimization failed for parameter {index}")
                nuisance = np.asarray(optimized.x, dtype=float)
                value = float(optimized.fun)
            else:
                value = float(objective(np.asarray([fixed], dtype=float)))
            profile.append(max(value - base, 0.0))
        rows.append(
            {
                "parameter_index": index,
                "grid": [float(value) for value in grid],
                "delta_chi_square": profile,
                "interval_68": _profile_interval(grid, profile, 1.0, center[index]),
                "interval_95": _profile_interval(grid, profile, 3.841458820694124, center[index]),
            }
        )
    return tuple(rows)


def _profile_interval(
    grid: np.ndarray, profile: Sequence[float], threshold: float, center: float
) -> list[float | None]:
    values = np.asarray(profile, dtype=float)
    middle = int(np.argmin(np.abs(grid - center)))

    def crossing(indices: Sequence[int]) -> float | None:
        for left, right in zip(indices[:-1], indices[1:], strict=True):
            y0, y1 = values[left] - threshold, values[right] - threshold
            if y0 == 0.0:
                return float(grid[left])
            if y0 * y1 <= 0.0 and y1 != y0:
                fraction = -y0 / (y1 - y0)
                return float(grid[left] + fraction * (grid[right] - grid[left]))
        return None

    lower = crossing(tuple(range(middle, -1, -1)))
    upper = crossing(tuple(range(middle, len(grid))))
    return [lower, upper]


__all__ = [
    "CorrelatedUncertainty",
    "correlated_linearized_uncertainty",
    "parametric_bootstrap",
    "profile_likelihood_intervals",
    "propagate_surface_model_uncertainty",
    "qim_surface_parameter_distribution",
    "sample_correlated_parameters",
]
