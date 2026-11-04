"""Local natural-variable charts; chemical family policy remains in ORACLE.

These maps act on *local* SONIC components before symmetry mixing. They
reparameterize an existing potential, not its force constants or energy.
In particular, inverse distance must never be applied to a stretch SALC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


class SonicVariable(Protocol):
    """Extensible scalar chart, with exact derivatives of its inverse."""

    def forward(self, value: float) -> float: ...

    def inverse(self, variable: float) -> tuple[float, float, float]:
        """Return q(u), dq/du, d²q/du² in the potential's original units."""
        ...


@dataclass(frozen=True)
class NaturalSonicVariable:
    """Centered distance, angle, periodic or regular-component chart.

    ``scale`` divides the centered natural variable; its default is one.
    Linear bends require nonsingular transverse components, not cos(theta).
    Flaps, mixed-family components and quaternion tangent components may
    use ``linear``; a raw four-component quaternion is not an unconstrained
    coordinate chart. Angles and torsions are in radians, distances in the
    same units as the bound physical potential (normally Angstrom).
    """

    kind: str
    reference: float
    scale: float = 1.0
    singularity_margin: float = 1.0e-8

    def __post_init__(self):
        if self.kind not in {"linear", "inverse_distance", "cosine_angle", "periodic"}:
            raise ValueError(f"unsupported natural SONIC chart: {self.kind}")
        if not np.isfinite(self.reference) or not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("reference and positive chart scale must be finite")
        if not 0 < self.singularity_margin < 0.1:
            raise ValueError("chart singularity margin must lie between zero and 0.1")
        self.forward(self.reference)

    def forward(self, value: float) -> float:
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("SONIC coordinate must be finite")
        if self.kind == "inverse_distance":
            if min(value, self.reference) <= 0:
                raise ValueError("inverse-distance chart requires positive distances")
            return (1.0 / value - 1.0 / self.reference) / self.scale
        if self.kind == "cosine_angle":
            if not all(self.singularity_margin < q < np.pi - self.singularity_margin
                       for q in (value, self.reference)):
                raise ValueError("cosine-angle chart is singular; use transverse linear bends")
            return (np.cos(value) - np.cos(self.reference)) / self.scale
        difference = value - self.reference
        if self.kind == "periodic":
            difference = (difference + np.pi) % (2 * np.pi) - np.pi
            if abs(difference) >= np.pi - self.singularity_margin:
                raise ValueError("periodic SONIC chart needs recentering at its branch cut")
        return difference / self.scale

    def inverse(self, variable: float) -> tuple[float, float, float]:
        value = float(variable) * self.scale
        if not np.isfinite(value):
            raise ValueError("natural SONIC variable must be finite")
        if self.kind == "inverse_distance":
            reciprocal = value + 1.0 / self.reference
            if reciprocal <= 0:
                raise ValueError("inverse-distance step crosses the dissociation boundary")
            distance = 1.0 / reciprocal
            result = (distance, -self.scale * distance**2,
                      2 * self.scale**2 * distance**3)
        elif self.kind == "cosine_angle":
            cosine = value + np.cos(self.reference)
            if not -1.0 < cosine < 1.0:
                raise ValueError("cosine-angle step crosses the chart boundary")
            angle = float(np.arccos(cosine))
            self.forward(angle)
            sine = np.sqrt((1.0 - cosine) * (1.0 + cosine))
            result = (angle, -self.scale / sine, -self.scale**2 * cosine / sine**3)
        else:
            if self.kind == "periodic" and abs(value) >= np.pi - self.singularity_margin:
                raise ValueError("periodic SONIC chart needs recentering at its branch cut")
            result = (self.reference + value, self.scale, 0.0)
        if not np.all(np.isfinite(result)):
            raise ValueError("natural SONIC inverse is numerically singular")
        return tuple(float(item) for item in result)


class SonicVariableMap:
    """O(n) component maps, with no B matrix or global Jacobian/Hessian."""

    def __init__(self, variables: Sequence[SonicVariable]):
        self.variables = tuple(variables)

    def _vector(self, values):
        result = np.asarray(values, dtype=float)
        if result.shape != (len(self.variables),) or not np.all(np.isfinite(result)):
            raise ValueError("SONIC vector does not match the variable map")
        return result

    def forward(self, physical_values) -> np.ndarray:
        values = self._vector(physical_values)
        return np.asarray([chart.forward(q) for chart, q in zip(self.variables, values, strict=True)])

    def inverse(self, natural_values) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = self._vector(natural_values)
        triples = np.asarray([chart.inverse(u) for chart, u in
                              zip(self.variables, values, strict=True)], dtype=float).reshape(-1, 3)
        return triples[:, 0], triples[:, 1], triples[:, 2]

    def pullback(self, natural_values, physical_gradient, physical_hvp, direction):
        """Exact g_u and H_u v, including the nonstationary chart-curvature term.

        ``physical_hvp`` is a callable H_q @ vector; no dense H_q is needed.
        These are scalar variable maps, not Cartesian-to-SONIC transforms.
        """
        _q, first, second = self.inverse(natural_values)
        gradient = self._vector(physical_gradient)
        vector = self._vector(direction)
        product = self._vector(physical_hvp(first * vector))
        return first * gradient, first * product + second * gradient * vector


__all__ = ["NaturalSonicVariable", "SonicVariable", "SonicVariableMap"]
