"""Local internal-coordinate maps with exactly quadratic diagonal models.

The LINK optimizer stores gradients and Hessians in its ordinary SONIC chart.
This module supplies a local, separable change of displacement variables for
the step calculation only. Each map has unit Jacobian at the current point,
so the existing metric, RFO, trust-region and secant infrastructure remains
authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


MORSE = "morse"
GAUSSIAN = "gaussian"
PERIODIC = "periodic"
LINEAR = "linear"
_KINDS = frozenset({MORSE, GAUSSIAN, PERIODIC, LINEAR})
_DOMAIN_MARGIN = 1.0 - 1.0e-10


@dataclass(frozen=True)
class QuadraticizedCoordinateMap:
    """Separable local map from physical displacements ``s`` to quadratic ``u``."""

    kinds: tuple[str, ...]
    scales: np.ndarray
    periods: np.ndarray
    # Dimensionless local Morse/Gaussian exponents.  ``scales`` remains the
    # public displacement scale for backwards compatibility; alpha is stored
    # explicitly so the map can be audited against the SH force constants.
    alphas: np.ndarray | None = None

    def __post_init__(self) -> None:
        kinds = tuple(str(item).strip().lower() for item in self.kinds)
        scales = np.asarray(self.scales, dtype=float).reshape(-1)
        periods = np.asarray(self.periods, dtype=float).reshape(-1)
        alphas = (
            1.0 / scales
            if self.alphas is None
            else np.asarray(self.alphas, dtype=float).reshape(-1)
        )
        if len(kinds) == 0 or scales.shape != (len(kinds),) or periods.shape != (len(kinds),):
            raise ValueError("quadraticized coordinate metadata has inconsistent dimensions")
        if any(kind not in _KINDS for kind in kinds):
            raise ValueError(
                "quadraticized coordinate kind must be Morse, Gaussian, periodic or linear"
            )
        if np.any(~np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError("quadraticized coordinate scales must be finite and positive")
        if alphas.shape != scales.shape or np.any(~np.isfinite(alphas)) or np.any(alphas <= 0.0):
            raise ValueError("quadraticized coordinate alphas must be finite and positive")
        periodic = np.asarray([kind == PERIODIC for kind in kinds], dtype=bool)
        if np.any(~np.isfinite(periods[periodic])) or np.any(periods[periodic] <= 0.0):
            raise ValueError("periodic quadraticized coordinates require a positive period")
        object.__setattr__(self, "kinds", kinds)
        object.__setattr__(self, "scales", scales)
        object.__setattr__(self, "periods", periods)
        object.__setattr__(self, "alphas", alphas)

    @property
    def size(self) -> int:
        return len(self.kinds)

    def forward(self, displacement: Sequence[float] | np.ndarray) -> np.ndarray:
        """Map a physical internal displacement to its quadratic variable."""

        step = self._vector(displacement, "physical displacement")
        transformed = np.empty_like(step)
        for index, (kind, value) in enumerate(zip(self.kinds, step, strict=True)):
            scale = float(self.scales[index])
            if kind == MORSE:
                transformed[index] = -scale * math.expm1(-float(value) / scale)
            elif kind == GAUSSIAN:
                ratio = float(value) / scale
                magnitude = scale * math.sqrt(
                    max(0.0, -2.0 * math.expm1(-0.5 * ratio * ratio))
                )
                transformed[index] = math.copysign(magnitude, float(value))
            elif kind == PERIODIC:
                period = float(self.periods[index])
                frequency = 2.0 * math.pi / period
                local = (float(value) + 0.5 * period) % period - 0.5 * period
                transformed[index] = 2.0 * math.sin(0.5 * frequency * local) / frequency
            else:
                transformed[index] = float(value)
        return transformed

    def inverse(self, transformed_step: Sequence[float] | np.ndarray) -> tuple[np.ndarray, float]:
        """Return the physical displacement and any common domain scaling."""

        requested = self._vector(transformed_step, "quadraticized displacement")
        factor = 1.0
        for index, (kind, value) in enumerate(zip(self.kinds, requested, strict=True)):
            if kind == MORSE and value > 0.0:
                limit = float(self.scales[index]) * _DOMAIN_MARGIN
                factor = min(factor, limit / float(value))
            elif kind == GAUSSIAN and value != 0.0:
                limit = math.sqrt(2.0) * float(self.scales[index]) * _DOMAIN_MARGIN
                factor = min(factor, limit / abs(float(value)))
            elif kind == PERIODIC and value != 0.0:
                frequency = 2.0 * math.pi / float(self.periods[index])
                limit = 2.0 / frequency * _DOMAIN_MARGIN
                factor = min(factor, limit / abs(float(value)))
        factor = float(np.clip(factor, 0.0, 1.0))
        values = factor * requested
        displacement = np.empty_like(values)
        for index, (kind, value) in enumerate(zip(self.kinds, values, strict=True)):
            scale = float(self.scales[index])
            if kind == MORSE:
                displacement[index] = -scale * math.log1p(-float(value) / scale)
            elif kind == GAUSSIAN:
                ratio2 = 0.5 * (float(value) / scale) ** 2
                magnitude = scale * math.sqrt(max(0.0, -2.0 * math.log1p(-ratio2)))
                displacement[index] = math.copysign(magnitude, float(value))
            elif kind == PERIODIC:
                frequency = 2.0 * math.pi / float(self.periods[index])
                argument = float(np.clip(0.5 * frequency * float(value), -1.0, 1.0))
                displacement[index] = 2.0 * math.asin(argument) / frequency
            else:
                displacement[index] = float(value)
        if not np.all(np.isfinite(displacement)):
            raise FloatingPointError("quadraticized coordinate inverse produced a non-finite step")
        return displacement, factor

    def hessian_at_origin(
        self,
        gradient: Sequence[float] | np.ndarray,
        physical_hessian: np.ndarray,
    ) -> np.ndarray:
        """Transform a physical Hessian to the local quadratic coordinates."""

        covector = self._vector(gradient, "gradient")
        matrix = np.asarray(physical_hessian, dtype=float)
        if matrix.shape != (self.size, self.size) or not np.all(np.isfinite(matrix)):
            raise ValueError("physical Hessian does not match the quadraticized coordinates")
        transformed = 0.5 * (matrix + matrix.T)
        transformed = transformed.copy()
        diagonal = np.diag(transformed).copy()
        for index, kind in enumerate(self.kinds):
            if kind == MORSE:
                diagonal[index] += covector[index] / self.scales[index]
        np.fill_diagonal(transformed, diagonal)
        return transformed

    def covector_to_transformed(
        self,
        displacement: Sequence[float] | np.ndarray,
        physical_covector: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Transform a covector at a displaced historical point to ``u``."""

        step = self._vector(displacement, "historical displacement")
        covector = self._vector(physical_covector, "historical gradient")
        inverse_derivative = np.empty(self.size, dtype=float)
        for index, (kind, value) in enumerate(zip(self.kinds, step, strict=True)):
            scale = float(self.scales[index])
            if kind == MORSE:
                inverse_derivative[index] = math.exp(float(value) / scale)
            elif kind == GAUSSIAN:
                ratio = abs(float(value) / scale)
                if ratio <= 1.0e-7:
                    inverse_derivative[index] = 1.0 + 0.375 * ratio * ratio
                else:
                    exponential = math.exp(-0.5 * ratio * ratio)
                    image = math.sqrt(max(0.0, 2.0 * (1.0 - exponential)))
                    inverse_derivative[index] = image / (ratio * exponential)
            elif kind == PERIODIC:
                period = float(self.periods[index])
                frequency = 2.0 * math.pi / period
                local = (float(value) + 0.5 * period) % period - 0.5 * period
                cosine = math.cos(0.5 * frequency * local)
                inverse_derivative[index] = 1.0 / max(cosine, 1.0e-10)
            else:
                inverse_derivative[index] = 1.0
        transformed = inverse_derivative * covector
        if not np.all(np.isfinite(transformed)):
            raise FloatingPointError("quadraticized historical covector is non-finite")
        return transformed

    def predicted_reduction(
        self,
        physical_step: Sequence[float] | np.ndarray,
        gradient: Sequence[float] | np.ndarray,
        transformed_hessian: np.ndarray,
    ) -> float:
        """Evaluate the exact quadratic model after mapping a realized step."""

        covector = self._vector(gradient, "gradient")
        matrix = np.asarray(transformed_hessian, dtype=float)
        if matrix.shape != (self.size, self.size):
            raise ValueError("transformed Hessian has the wrong shape")
        step = self.forward(physical_step)
        return float(-(covector @ step + 0.5 * step @ (0.5 * (matrix + matrix.T)) @ step))

    def _vector(self, values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
        vector = np.asarray(values, dtype=float).reshape(-1)
        if vector.shape != (self.size,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} does not match the quadraticized coordinates")
        return vector


@dataclass(frozen=True)
class ProjectedQuadraticizedCoordinateMap:
    """Quadraticize SONIC components within a full-rank projected task space."""

    components: QuadraticizedCoordinateMap
    sonic_from_coordinates: np.ndarray

    def __post_init__(self) -> None:
        transform = np.asarray(self.sonic_from_coordinates, dtype=float)
        if transform.ndim != 2 or transform.shape[0] != self.components.size:
            raise ValueError("projected quadraticization has incompatible SONIC dimensions")
        if not np.all(np.isfinite(transform)) or np.linalg.matrix_rank(transform) != transform.shape[1]:
            raise ValueError("projected quadraticization requires full column rank")
        left_inverse = np.linalg.pinv(transform, rcond=1.0e-12)
        if not np.allclose(left_inverse @ transform, np.eye(transform.shape[1]), atol=1.0e-9):
            raise ValueError("projected SONIC map has an unstable left inverse")
        object.__setattr__(self, "sonic_from_coordinates", transform)
        object.__setattr__(self, "_left_inverse", left_inverse)

    @property
    def size(self) -> int:
        return int(self.sonic_from_coordinates.shape[1])

    def _vector(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        vector = np.asarray(values, dtype=float).reshape(-1)
        if vector.shape != (self.size,) or not np.all(np.isfinite(vector)):
            raise ValueError("projected displacement has invalid dimension or values")
        return vector

    def forward(self, displacement: Sequence[float] | np.ndarray) -> np.ndarray:
        active = self._vector(displacement)
        return self._left_inverse @ self.components.forward(self.sonic_from_coordinates @ active)

    def _jacobian(self, active: np.ndarray) -> np.ndarray:
        component_step = self.sonic_from_coordinates @ active
        inverse_derivative = self.components.covector_to_transformed(
            component_step, np.ones(self.components.size, dtype=float)
        )
        return self._left_inverse @ (
            self.sonic_from_coordinates / inverse_derivative[:, None]
        )

    def inverse(self, transformed_step: Sequence[float] | np.ndarray) -> tuple[np.ndarray, float]:
        requested = self._vector(transformed_step)
        factor = 1.0
        for _attempt in range(12):
            target = factor * requested
            active = target.copy()
            for _iteration in range(30):
                try:
                    residual = self.forward(active) - target
                    error = float(np.linalg.norm(residual))
                    if error <= 1.0e-10 * max(1.0, float(np.linalg.norm(target))):
                        return active, factor
                    correction = np.linalg.solve(self._jacobian(active), residual)
                    for line_search in range(12):
                        candidate = active - (0.5**line_search) * correction
                        if float(np.linalg.norm(self.forward(candidate) - target)) < error:
                            active = candidate
                            break
                    else:
                        break
                except (FloatingPointError, OverflowError, ValueError, np.linalg.LinAlgError):
                    break
            factor *= 0.5
        raise ValueError("projected quadraticized displacement has no realizable local inverse")

    def hessian_at_origin(
        self, gradient: Sequence[float] | np.ndarray, physical_hessian: np.ndarray
    ) -> np.ndarray:
        covector = self._vector(gradient)
        matrix = np.asarray(physical_hessian, dtype=float)
        if matrix.shape != (self.size, self.size) or not np.all(np.isfinite(matrix)):
            raise ValueError("projected Hessian has invalid dimension or values")
        component_covector = self._left_inverse.T @ covector
        curvature = np.asarray(
            [
                component_covector[index] / self.components.scales[index]
                if kind == MORSE else 0.0
                for index, kind in enumerate(self.components.kinds)
            ],
            dtype=float,
        )
        transformed = matrix + self.sonic_from_coordinates.T @ (
            curvature[:, None] * self.sonic_from_coordinates
        )
        return 0.5 * (transformed + transformed.T)

    def covector_to_transformed(
        self, displacement: Sequence[float] | np.ndarray,
        physical_covector: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        active = self._vector(displacement)
        covector = self._vector(physical_covector)
        return np.linalg.solve(self._jacobian(active).T, covector)

    def predicted_reduction(
        self, physical_step: Sequence[float] | np.ndarray,
        gradient: Sequence[float] | np.ndarray,
        transformed_hessian: np.ndarray,
    ) -> float:
        step = self.forward(physical_step)
        covector = self._vector(gradient)
        matrix = np.asarray(transformed_hessian, dtype=float)
        if matrix.shape != (self.size, self.size):
            raise ValueError("projected transformed Hessian has the wrong shape")
        return float(-(covector @ step + 0.5 * step @ matrix @ step))
