"""Minimal two-state EVB evaluator with analytic first derivatives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .surface_representation import SurfaceRepresentationRequest, validate_basin_transition


@dataclass(frozen=True)
class EVBEvaluation:
    energy: float
    gradient: np.ndarray
    state_weights: np.ndarray
    gap: float


def evaluate_two_state_evb(
    energies: tuple[float, float] | np.ndarray,
    gradients: tuple[np.ndarray, np.ndarray] | np.ndarray,
    coupling: float,
    coupling_gradient: np.ndarray | None = None,
) -> EVBEvaluation:
    """Evaluate the lower eigenvalue of a two-state diabatic Hamiltonian."""

    diagonal = np.asarray(energies, dtype=float).reshape(-1)
    if diagonal.shape != (2,) or not np.all(np.isfinite(diagonal)):
        raise ValueError("EVB requires two finite diabatic energies")
    grad = np.asarray(gradients, dtype=float)
    if grad.ndim != 2 or grad.shape[0] != 2 or not np.all(np.isfinite(grad)):
        raise ValueError("EVB gradients must have shape (2, n) and be finite")
    coupling = float(coupling)
    if not np.isfinite(coupling):
        raise ValueError("EVB coupling must be finite")
    matrix = np.asarray(((diagonal[0], coupling), (coupling, diagonal[1])), dtype=float)
    values, vectors = np.linalg.eigh(matrix)
    coefficients = vectors[:, 0]
    weights = coefficients * coefficients
    gap = float(values[1] - values[0])
    gradient = weights[0] * grad[0] + weights[1] * grad[1]
    if coupling_gradient is not None:
        derivative = np.asarray(coupling_gradient, dtype=float).reshape(-1)
        if derivative.shape != (grad.shape[1],) or not np.all(np.isfinite(derivative)):
            raise ValueError("EVB coupling gradient has invalid shape or values")
        gradient = gradient + 2.0 * coefficients[0] * coefficients[1] * derivative
    return EVBEvaluation(float(values[0]), gradient, weights, gap)


def evaluate_two_state_evb_request(
    request: SurfaceRepresentationRequest,
    energies: tuple[float, float] | np.ndarray,
    gradients: tuple[np.ndarray, np.ndarray] | np.ndarray,
    coupling: float,
    coupling_gradient: np.ndarray | None = None,
) -> EVBEvaluation:
    """Evaluate EVB only when the explicit representation/gate contract is present."""

    if request.provider != "EVB":
        raise ValueError("EVB evaluator requires provider='EVB'")
    validate_basin_transition(request)
    return evaluate_two_state_evb(energies, gradients, coupling, coupling_gradient)
