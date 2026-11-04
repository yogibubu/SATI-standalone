"""Local derivative matching with finite dissociation and periodic references."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.polynomial import Polynomial


def _match_polynomial(variable_jet, derivatives):
    variable = Polynomial(variable_jet)
    matrix = np.zeros((5, 5))
    for k in range(5):
        c = (variable**k).coef[:5]
        matrix[: len(c), k] = c
    return np.linalg.solve(matrix, np.asarray(derivatives) / np.array([1, 1, 2, 6, 24]))


@dataclass(frozen=True)
class NativeReferencePotential:
    """A fourth-order local constraint represented by a nonpolynomial function.

    Distance is in angstrom, angles in radians, energy in hartree. Radial
    energies use the existing library component's zero: its dissociation
    limit is De, even when the supplied reference is not its local minimum.
    The radial switch uses the library alpha and starts at sixth order, so
    matching through the fourth derivative is not altered by dissociation.
    Bond switches for angular and mixed terms are applied by the ZAFF field.
    """

    kind: str
    reference: float
    coefficients: tuple[float, ...]
    asymptotic_energy: float | None = None
    alpha: float | None = None
    periodicity: int | None = None

    def value_gradient_hessian(self, coordinate: float) -> tuple[float, float, float]:
        q = float(coordinate)
        if not math.isfinite(q):
            raise ValueError("nonfinite native coordinate")
        delta = q - self.reference
        if self.kind == "stretch":
            if q <= 0:
                raise ValueError("bond length must be positive")
            u = self.reference**2 * (1 / self.reference - 1 / q)
            du, ddu = self.reference**2 / q**2, -2 * self.reference**2 / q**3
        elif self.kind == "angle":
            if not 0 <= q <= math.pi:
                raise ValueError("valence angle must be in [0, pi]")
            u, du, ddu = math.cos(q) - math.cos(self.reference), -math.sin(q), -math.cos(q)
        elif self.kind == "torsion":
            p = self.periodicity
            s, c = math.sin(p * delta), math.cos(p * delta)
            a, b, d, e, f = self.coefficients
            value = a + b * s + d * (1 - c) + e * s * (1 - c) + f * (1 - c) ** 2
            first = p * (b * c + d * s + e * (c * (1 - c) + s * s) + 2 * f * (1 - c) * s)
            second = p * p * (-b * s + d * c + e * s * (4 * c - 1) + 2 * f * (s * s + (1 - c) * c))
            return value, first, second
        else:
            raise ValueError("unsupported native reference kind")
        polynomial = Polynomial(self.coefficients)
        value = float(polynomial(u))
        first = float(polynomial.deriv()(u)) * du
        second = float(polynomial.deriv(2)(u)) * du * du + float(polynomial.deriv()(u)) * ddu
        if self.kind == "angle":
            return value, first, second
        t = self.alpha * delta
        if abs(t) > 4.0:  # exp(-4**6) is below floating-point range
            return self.asymptotic_energy, 0.0, 0.0
        switch = math.exp(-(t**6))
        log_first, log_second = -6 * self.alpha * t**5, -30 * self.alpha**2 * t**4
        difference = value - self.asymptotic_energy
        return (
            self.asymptotic_energy + switch * difference,
            switch * (first + log_first * difference),
            switch * (second + 2 * log_first * first + (log_second + log_first**2) * difference),
        )


def native_reference_from_derivatives(
    kind: str,
    reference: float,
    derivatives,
    *,
    dissociation_energy: float | None = None,
    library_alpha: float | None = None,
    periodicity: int | None = None,
) -> NativeReferencePotential:
    """Match E and derivatives 1..4 without inferring a LAM barrier from them.

    Missing higher derivatives must not be passed as measured zeros. The
    Hessian-only hierarchy keeps its existing library/Morse reference.
    Periodicity and dissociation parameters are supplied by chemical typing.
    """
    d = np.asarray(derivatives, dtype=float)
    if d.shape != (5,) or not np.all(np.isfinite(d)) or not np.isfinite(reference):
        raise ValueError("five finite local derivatives (orders 0..4) are required")
    if kind == "stretch":
        if (
            reference <= 0
            or dissociation_energy is None
            or library_alpha is None
            or not np.isfinite(dissociation_energy)
            or dissociation_energy <= 0
            or not np.isfinite(library_alpha)
            or library_alpha <= 0
        ):
            raise ValueError("stretch requires positive reference, library De and alpha")
        coefficients = _match_polynomial(
            [0.0, 1.0, -1 / reference, 1 / reference**2, -1 / reference**3], d
        )
        return NativeReferencePotential(
            kind,
            reference,
            tuple(coefficients),
            float(dissociation_energy),
            float(library_alpha),
        )
    if kind == "angle":
        if not 0 < reference < math.pi or abs(math.sin(reference)) < 1e-6:
            raise ValueError("linear angles require the existing linear-bend coordinates")
        s, c = math.sin(reference), math.cos(reference)
        coefficients = _match_polynomial([0.0, -s, -c / 2, s / 6, c / 24], d)
        return NativeReferencePotential(kind, reference, tuple(coefficients))
    if kind == "torsion":
        if (
            not isinstance(periodicity, (int, np.integer))
            or isinstance(periodicity, bool)
            or periodicity < 1
        ):
            raise ValueError("torsion requires its chemically assigned positive periodicity")
        p = int(periodicity)
        c1, c2 = d[1] / p, d[2] / p**2
        coefficients = (d[0], c1, c2, (d[3] / p**3 + c1) / 3, (d[4] / p**4 + c2) / 6)
        return NativeReferencePotential(kind, reference, coefficients, periodicity=p)
    raise ValueError("unsupported native reference kind")
