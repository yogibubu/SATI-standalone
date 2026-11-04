"""Bowman Q_im transition-vector path with orthogonal-mode relaxation.

The implementation follows Wang--Bowman (JCP 129, 121103, 2008) and
Wang--Bowman (JCP 139, 154303, 2013): a mass-scaled imaginary-frequency
transition vector defines Q_im, while the remaining saddle modes are relaxed
at fixed Q_im.  The resulting relaxed V(Q_im) is suitable for a 1D DVR.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class QimPathPoint:
    q_im: float
    coordinates: np.ndarray
    energy: float
    orthogonal_coordinates: np.ndarray
    converged: bool
    iterations: int
    gradient_norm: float


@dataclass(frozen=True)
class QimPathResult:
    transition_vector: np.ndarray
    masses: np.ndarray
    q_values: np.ndarray
    points: tuple[QimPathPoint, ...]
    orthogonal_projections: dict[str, float]
    turning_points: tuple[float, ...]


def mass_scaled_transition_vector(hessian: np.ndarray, masses: Sequence[float], *, imaginary_index: int | None = None) -> tuple[np.ndarray, float]:
    """Return the normalized imaginary-frequency eigenvector and frequency."""
    matrix = np.asarray(hessian, dtype=float)
    mass = np.asarray(masses, dtype=float).reshape(-1)
    if matrix.shape != (mass.size, mass.size):
        raise ValueError("Hessian shape must match the flattened mass vector")
    if np.any(mass <= 0.0):
        raise ValueError("masses must be positive")
    mass_inv_sqrt = np.diag(1.0 / np.sqrt(mass))
    mass_weighted = 0.5 * (mass_inv_sqrt @ matrix @ mass_inv_sqrt + mass_inv_sqrt @ matrix.T @ mass_inv_sqrt)
    values, vectors = np.linalg.eigh(mass_weighted)
    negative = np.flatnonzero(values < 0.0)
    if imaginary_index is None:
        if negative.size == 0:
            raise ValueError("Hessian has no imaginary-frequency mode")
        index = int(negative[np.argmin(values[negative])])
    else:
        index = int(imaginary_index)
        if values[index] >= 0.0:
            raise ValueError("selected transition-vector eigenvalue is not imaginary")
    vector = mass_inv_sqrt @ vectors[:, index]
    vector /= np.linalg.norm(np.sqrt(mass) * vector)
    return vector, float(np.sqrt(abs(values[index])))


def orthogonal_mode_basis(transition_vector: np.ndarray, *, count: int | None = None) -> np.ndarray:
    """Construct an orthonormal Euclidean basis perpendicular to Q_im."""
    vector = np.asarray(transition_vector, dtype=float).reshape(-1)
    vector = vector / np.linalg.norm(vector)
    basis = []
    for index in range(vector.size):
        trial = np.zeros_like(vector); trial[index] = 1.0
        for previous in [vector, *basis]:
            trial -= np.dot(trial, previous) * previous
        norm = np.linalg.norm(trial)
        if norm > 1.0e-10:
            basis.append(trial / norm)
        if count is not None and len(basis) >= count:
            break
    return np.asarray(basis)


def optimize_qim_path(
    transition_vector: np.ndarray,
    reference_coordinates: np.ndarray,
    q_values: Sequence[float],
    evaluator: Callable[[np.ndarray], tuple[float, np.ndarray]],
    *,
    perpendicular_basis: np.ndarray | None = None,
    initial_orthogonal: np.ndarray | None = None,
    max_iterations: int = 200,
    gradient_tolerance: float = 1.0e-6,
) -> QimPathResult:
    """Relax all perpendicular coordinates at fixed Q_im using an energy/gradient evaluator."""
    reference = np.asarray(reference_coordinates, dtype=float).reshape(-1)
    direction = np.asarray(transition_vector, dtype=float).reshape(-1)
    direction /= np.linalg.norm(direction)
    basis = orthogonal_mode_basis(direction) if perpendicular_basis is None else np.asarray(perpendicular_basis, dtype=float)
    basis = np.asarray([row / np.linalg.norm(row) for row in basis if np.linalg.norm(row) > 1.0e-12])
    previous = np.zeros(basis.shape[0]) if initial_orthogonal is None else np.asarray(initial_orthogonal, dtype=float).copy()
    points = []
    for q in np.asarray(q_values, dtype=float):
        orthogonal = previous.copy()
        converged = False
        gradient_norm = float("inf")
        for iteration in range(1, max_iterations + 1):
            coordinates = reference + q * direction + orthogonal @ basis
            energy, gradient = evaluator(coordinates)
            projected = basis @ np.asarray(gradient, dtype=float).reshape(-1)
            gradient_norm = float(np.linalg.norm(projected))
            if gradient_norm <= gradient_tolerance:
                converged = True
                break
            step = 0.15 / max(1.0, gradient_norm)
            orthogonal -= step * projected
        energy, _ = evaluator(reference + q * direction + orthogonal @ basis)
        final_coordinates = reference + q * direction + orthogonal @ basis
        points.append(QimPathPoint(float(q), final_coordinates.copy(), float(energy), orthogonal.copy(), converged, iteration, gradient_norm))
        previous = orthogonal
    return QimPathResult(direction, np.ones(direction.size), np.asarray(q_values, dtype=float), tuple(points), {}, _turning_points(np.asarray(q_values), np.asarray([point.energy for point in points])))


def mode_projections(minimum_modes: np.ndarray, transition_vector: np.ndarray) -> dict[str, float]:
    modes = np.asarray(minimum_modes, dtype=float)
    vector = np.asarray(transition_vector, dtype=float).reshape(-1); vector /= np.linalg.norm(vector)
    projections = modes.reshape(modes.shape[0], -1) @ vector
    return {f"mode_{index + 1}": float(value) for index, value in enumerate(projections)}


def _turning_points(q_values: np.ndarray, energies: np.ndarray) -> tuple[float, ...]:
    if energies.size < 3:
        return ()
    turning = []
    for index in range(1, len(energies) - 1):
        if (energies[index] - energies[index - 1]) * (energies[index + 1] - energies[index]) <= 0.0:
            turning.append(float(q_values[index]))
    return tuple(turning)


def write_qim_path(path: Path | str, result: QimPathResult) -> Path:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["q_im", "energy", "converged", "iterations", "gradient_norm"])
        writer.writerows((point.q_im, point.energy, point.converged, point.iterations, point.gradient_norm) for point in result.points)
    target.write_text(json.dumps({"schema": "matrix.tunneling.qim_path.v1", "q_values": result.q_values.tolist(), "turning_points": list(result.turning_points), "orthogonal_projections": result.orthogonal_projections, "points": [{"q_im": point.q_im, "energy": point.energy, "converged": point.converged, "iterations": point.iterations, "gradient_norm": point.gradient_norm} for point in result.points]}, indent=2) + "\n", encoding="utf-8")
    return target


__all__ = ["QimPathPoint", "QimPathResult", "mass_scaled_transition_vector", "mode_projections", "optimize_qim_path", "orthogonal_mode_basis", "write_qim_path"]
