"""Adaptive, serializable representations of contracted effective surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np

from .reduced_anharmonic import total_degree_powers


ADAPTIVE_EFFECTIVE_SURFACE_SCHEMA = "matrix.gf.adaptive_effective_surface.v1"


@dataclass(frozen=True)
class AdaptiveEffectiveSurface:
    """A polynomial or tabulated effective surface with qualification data."""

    representation: str
    dimension: int
    points: np.ndarray
    values: np.ndarray
    powers: tuple[tuple[int, ...], ...] = ()
    coefficients: np.ndarray | None = None
    axes: tuple[np.ndarray, ...] = ()
    degree: int | None = None
    maximum_training_error: float = 0.0
    cross_validation_rms: float = 0.0
    tolerance: float = 0.0
    qualified: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": ADAPTIVE_EFFECTIVE_SURFACE_SCHEMA,
            "representation": self.representation,
            "dimension": self.dimension,
            "degree": self.degree,
            "powers": [list(item) for item in self.powers],
            "coefficients": (
                None if self.coefficients is None else self.coefficients.tolist()
            ),
            "points": self.points.tolist(),
            "values": self.values.tolist(),
            "axes": [axis.tolist() for axis in self.axes],
            "maximum_training_error": self.maximum_training_error,
            "cross_validation_rms": self.cross_validation_rms,
            "tolerance": self.tolerance,
            "qualified": self.qualified,
            "reason": self.reason,
        }

    def evaluate(self, coordinates: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.ndim == 1:
            coordinates = coordinates[None, :]
        if coordinates.ndim != 2 or coordinates.shape[1] != self.dimension:
            raise ValueError("effective-surface coordinates have an incompatible shape")
        if self.representation == "polynomial":
            design = _polynomial_design(coordinates, self.powers)
            return design @ np.asarray(self.coefficients, dtype=float).T
        if self.representation == "tabulated-linear":
            from scipy.interpolate import RegularGridInterpolator

            shape = tuple(len(axis) for axis in self.axes)
            bounded = coordinates.copy()
            for index, axis in enumerate(self.axes):
                bounded[:, index] = np.clip(bounded[:, index], axis[0], axis[-1])
            channels = self.values.shape[0]
            result = []
            for channel in range(channels):
                grid_values = _values_on_cartesian_grid(
                    self.points, self.values[channel], self.axes
                ).reshape(shape)
                interpolator = RegularGridInterpolator(
                    self.axes, grid_values, method="linear", bounds_error=False, fill_value=None
                )
                result.append(interpolator(bounded))
            return np.asarray(result).T
        raise ValueError(f"unsupported adaptive surface representation {self.representation!r}")


def fit_adaptive_effective_surface(
    points: Sequence[Sequence[float]] | np.ndarray,
    values: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
    *,
    minimum_degree: int = 2,
    maximum_degree: int = 8,
    tolerance: float = 1.0,
    cross_validation_fraction: float = 0.2,
) -> AdaptiveEffectiveSurface:
    """Select the smallest qualified polynomial, then retain a tabulated surface.

    The deterministic holdout guards against accepting a high-degree interpolation
    merely because it reproduces its training grid.  If no full-rank polynomial
    reaches the requested error, a tensor-product tabulation is returned instead
    of silently accepting a poor projection or aborting the scientific workflow.
    """

    sample_points = np.asarray(points, dtype=float)
    sample_values = np.asarray(values, dtype=float)
    if sample_values.ndim == 1:
        sample_values = sample_values[None, :]
    if (
        sample_points.ndim != 2
        or sample_values.ndim != 2
        or sample_values.shape[1] != len(sample_points)
        or len(sample_points) < 3
        or not np.all(np.isfinite(sample_points))
        or not np.all(np.isfinite(sample_values))
        or minimum_degree < 1
        or maximum_degree < minimum_degree
        or tolerance <= 0.0
        or not 0.0 < cross_validation_fraction < 0.5
    ):
        raise ValueError("adaptive effective-surface inputs are incompatible")

    count, dimension = sample_points.shape
    stride = max(int(round(1.0 / cross_validation_fraction)), 2)
    holdout = np.zeros(count, dtype=bool)
    holdout[::stride] = True
    if np.count_nonzero(~holdout) < 2 or np.count_nonzero(holdout) < 1:
        holdout[:] = False
        holdout[-1] = True
    best: AdaptiveEffectiveSurface | None = None
    for degree in range(minimum_degree, maximum_degree + 1):
        powers = total_degree_powers(dimension, degree)
        design = _polynomial_design(sample_points, powers)
        training = design[~holdout]
        if np.linalg.matrix_rank(training) != len(powers):
            continue
        fitted = np.linalg.lstsq(training, sample_values[:, ~holdout].T, rcond=None)[0].T
        predicted = design @ fitted.T
        training_error = float(np.max(np.abs(
            predicted[~holdout] - sample_values[:, ~holdout].T
        )))
        validation_residual = predicted[holdout] - sample_values[:, holdout].T
        validation_rms = float(np.sqrt(np.mean(validation_residual**2)))
        candidate = AdaptiveEffectiveSurface(
            representation="polynomial",
            dimension=dimension,
            points=sample_points.copy(),
            values=sample_values.copy(),
            powers=powers,
            coefficients=fitted,
            degree=degree,
            maximum_training_error=training_error,
            cross_validation_rms=validation_rms,
            tolerance=tolerance,
            qualified=max(training_error, validation_rms) <= tolerance,
            reason="smallest full-rank polynomial satisfying training and holdout errors",
        )
        if best is None or (
            max(candidate.maximum_training_error, candidate.cross_validation_rms), degree
        ) < (
            max(best.maximum_training_error, best.cross_validation_rms), best.degree or 0
        ):
            best = candidate
        if candidate.qualified:
            return candidate

    axes = tuple(np.unique(sample_points[:, index]) for index in range(dimension))
    complete_grid = int(np.prod([len(axis) for axis in axes])) == count
    if complete_grid and _cartesian_grid_is_complete(sample_points, axes):
        return AdaptiveEffectiveSurface(
            representation="tabulated-linear",
            dimension=dimension,
            points=sample_points.copy(),
            values=sample_values.copy(),
            axes=axes,
            maximum_training_error=0.0,
            cross_validation_rms=(
                0.0 if best is None else best.cross_validation_rms
            ),
            tolerance=tolerance,
            qualified=True,
            reason=(
                "polynomial projection did not meet tolerance; retained the complete "
                "tensor grid for direct DVR interpolation"
            ),
        )
    if best is None:
        raise ValueError(
            "sampling design supports neither a full-rank polynomial nor a complete tensor grid"
        )
    return AdaptiveEffectiveSurface(
        **{**best.__dict__, "qualified": False, "reason": (
            "best available polynomial exceeds tolerance and the sampling is not a "
            "complete tensor grid; acquire additional points"
        )}
    )


def adaptive_surface_oscillator_matrices(
    surface: AdaptiveEffectiveSurface,
    occupations: Sequence[Sequence[int]] | np.ndarray,
    *,
    quadrature_order: int | None = None,
) -> np.ndarray:
    """Project a polynomial or tabulated surface into a product oscillator basis."""

    from scipy.special import eval_hermite, gammaln

    states = np.asarray(occupations, dtype=int)
    if (
        states.ndim != 2
        or states.shape[1] != surface.dimension
        or np.any(states < 0)
    ):
        raise ValueError("oscillator occupations do not match the effective surface")
    maximum = int(np.max(states, initial=0))
    order = quadrature_order or max(2 * maximum + 5, 24)
    if order <= 2 * maximum:
        raise ValueError("quadrature order is too small for the oscillator basis")
    nodes_1d, weights_1d = np.polynomial.hermite.hermgauss(order)
    nodes = np.asarray(list(product(nodes_1d, repeat=surface.dimension)))
    weights = np.asarray([
        float(np.prod(item)) for item in product(weights_1d, repeat=surface.dimension)
    ])
    amplitudes = np.ones((len(nodes), len(states)))
    for coordinate in range(surface.dimension):
        quantum = states[:, coordinate]
        normalization = np.exp(
            -0.5 * (quantum * np.log(2.0) + gammaln(quantum + 1.0) + 0.5 * np.log(np.pi))
        )
        amplitudes *= np.asarray([
            eval_hermite(int(value), nodes[:, coordinate])
            for value in quantum
        ]).T * normalization[None, :]
    sampled = surface.evaluate(nodes).T
    matrices = np.einsum(
        "gi,pg,g,gj->pij", amplitudes, sampled, weights, amplitudes, optimize=True
    )
    return 0.5 * (matrices + np.swapaxes(matrices, 1, 2))


def _polynomial_design(
    points: np.ndarray, powers: Sequence[Sequence[int]]
) -> np.ndarray:
    return np.asarray([
        [float(np.prod(point ** np.asarray(power, dtype=int))) for power in powers]
        for point in points
    ])


def _cartesian_grid_is_complete(
    points: np.ndarray, axes: Sequence[np.ndarray]
) -> bool:
    expected = {tuple(float(value) for value in item) for item in product(*axes)}
    observed = {tuple(float(value) for value in item) for item in points}
    return observed == expected


def _values_on_cartesian_grid(
    points: np.ndarray, values: np.ndarray, axes: Sequence[np.ndarray]
) -> np.ndarray:
    lookup = {tuple(float(x) for x in point): float(value)
              for point, value in zip(points, values, strict=True)}
    return np.asarray([lookup[tuple(float(x) for x in point)] for point in product(*axes)])


__all__ = [
    "ADAPTIVE_EFFECTIVE_SURFACE_SCHEMA",
    "AdaptiveEffectiveSurface",
    "fit_adaptive_effective_surface",
    "adaptive_surface_oscillator_matrices",
]
