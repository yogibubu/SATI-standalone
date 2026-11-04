"""Numerical checks for coordinate-dependent SONIC G(q) metrics."""

from __future__ import annotations

import numpy as np


def validate_metric_grid(metrics: np.ndarray, *, tolerance: float = 1.0e-10) -> dict[str, object]:
    values = np.asarray(metrics, dtype=float)
    if values.ndim < 2 or values.shape[-1] != values.shape[-2]:
        raise ValueError("metrics must have trailing square matrix dimensions")
    errors: list[str] = []
    symmetry_error = 0.0
    minimum_eigenvalue = float("inf")
    for matrix in values.reshape((-1, values.shape[-1], values.shape[-1])):
        symmetry_error = max(symmetry_error, float(np.max(np.abs(matrix - matrix.T))))
        if not np.all(np.isfinite(matrix)):
            errors.append("non-finite metric entry")
            continue
        minimum_eigenvalue = min(minimum_eigenvalue, float(np.min(np.linalg.eigvalsh((matrix + matrix.T) / 2))))
    if symmetry_error > tolerance:
        errors.append("metric is not symmetric within tolerance")
    if minimum_eigenvalue <= 0.0:
        errors.append("metric is not positive definite")
    return {"valid": not errors, "symmetry_error": symmetry_error, "minimum_eigenvalue": minimum_eigenvalue, "errors": errors}


def compare_metric_representations(sonic: np.ndarray, cartesian: np.ndarray, *, tolerance: float = 1.0e-6) -> dict[str, float | bool]:
    left, right = np.asarray(sonic, dtype=float), np.asarray(cartesian, dtype=float)
    if left.shape != right.shape:
        raise ValueError("metric representations must have equal shapes")
    difference = np.abs(left - right)
    maximum = float(np.max(difference, initial=0.0))
    return {"max_abs_difference": maximum, "rms_difference": float(np.sqrt(np.mean(difference * difference))), "within_tolerance": maximum <= tolerance}


__all__ = ["compare_metric_representations", "validate_metric_grid"]
