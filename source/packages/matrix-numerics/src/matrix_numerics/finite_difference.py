"""Backend-neutral local polynomial derivative fitting."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence

import numpy as np


# Shared production contract. Central differences are diagnostic/analysis
# choices and are never selected implicitly by a MATRIX tool.
DEFAULT_FINITE_DIFFERENCE_STENCIL = "forward"
ALLOWED_FINITE_DIFFERENCE_STENCILS = frozenset({"forward", "central"})


@dataclass(frozen=True)
class ScalarFiniteDifferenceResult:
    """Derivatives of a scalar surface at one fixed Cartesian frame.

    Scalar energies carry no orientation information.  The caller supplies
    displacements in one declared coordinate (SONIC or a normal mode); no
    Kabsch alignment, symmetry projection, or reordering is applied here.
    ``plus_step`` and ``minus_step`` are the realized signed-coordinate
    distances, so unequal steps cannot silently contaminate a curvature.
    """

    gradient: float
    curvature: float | None
    stencil: str
    plus_step: float
    minus_step: float | None


def scalar_finite_difference(
    center_energy: float,
    plus_energy: float,
    plus_step: float,
    *,
    minus_energy: float | None = None,
    minus_step: float | None = None,
) -> ScalarFiniteDifferenceResult:
    """Evaluate a validated one- or two-sided scalar energy difference.

    The central point is never moved or replaced by a lower-energy displaced
    point.  A two-sided curvature is returned only when both realized steps
    are supplied; this prevents the old failure mode in which a nominal step
    was used after geometry generation changed the actual displacement.
    """

    values = (float(center_energy), float(plus_energy), float(plus_step))
    if not all(math.isfinite(value) for value in values) or plus_step <= 0.0:
        raise ValueError("scalar finite-difference energies and plus_step must be finite")
    if (minus_energy is None) != (minus_step is None):
        raise ValueError("minus_energy and minus_step must be supplied together")
    if minus_energy is None:
        return ScalarFiniteDifferenceResult(
            gradient=(plus_energy - center_energy) / plus_step,
            curvature=None,
            stencil="forward",
            plus_step=plus_step,
            minus_step=None,
        )
    minus_energy = float(minus_energy)
    minus_step = float(minus_step)
    if not math.isfinite(minus_energy) or not math.isfinite(minus_step) or minus_step <= 0.0:
        raise ValueError("scalar finite-difference minus values must be finite")
    gradient = (
        -plus_step / (minus_step * (minus_step + plus_step)) * minus_energy
        + (plus_step - minus_step) / (minus_step * plus_step) * center_energy
        + minus_step / (plus_step * (minus_step + plus_step)) * plus_energy
    )
    curvature = 2.0 * (
        minus_energy / minus_step
        - center_energy * (1.0 / minus_step + 1.0 / plus_step)
        + plus_energy / plus_step
    ) / (minus_step + plus_step)
    return ScalarFiniteDifferenceResult(
        gradient=gradient,
        curvature=curvature,
        stencil="central",
        plus_step=plus_step,
        minus_step=minus_step,
    )


def normalize_finite_difference_stencil(stencil: str) -> str:
    """Validate and normalize the shared finite-difference stencil policy."""

    value = str(stencil).strip().casefold().replace("_", "-")
    if value not in ALLOWED_FINITE_DIFFERENCE_STENCILS:
        allowed = ", ".join(sorted(ALLOWED_FINITE_DIFFERENCE_STENCILS))
        raise ValueError(f"finite-difference stencil must be one of: {allowed}")
    return value


@dataclass(frozen=True)
class TaylorDerivativeFit:
    """Derivatives obtained from a factorial-scaled local Taylor fit."""

    derivatives: tuple[np.ndarray, ...]
    rank: int
    residual_norm: float


def fit_taylor_derivatives(
    displacements: Sequence[float] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    *,
    max_order: int = 4,
) -> TaylorDerivativeFit:
    """Fit derivatives at zero for scalar or array-valued observations."""

    x = np.asarray(displacements, dtype=float).reshape(-1)
    if x.size < 2:
        raise ValueError("at least two points are needed for finite differences")
    if not np.all(np.isfinite(x)):
        raise ValueError("finite-difference displacements must be finite")
    if np.unique(x).size != x.size:
        raise ValueError("finite-difference displacements must be unique")

    observed = np.asarray(values, dtype=float)
    if observed.ndim == 0 or observed.shape[0] != x.size:
        raise ValueError("values must have one leading entry per displacement")
    if not np.all(np.isfinite(observed)):
        raise ValueError("finite-difference values must be finite")

    order = max(1, min(int(max_order), x.size - 1))
    design = np.column_stack(
        [
            np.power(x, power) / float(math.factorial(power))
            for power in range(order + 1)
        ]
    )
    value_shape = observed.shape[1:]
    flattened = observed.reshape(x.size, -1)
    coefficients, _residuals, rank, _singular = np.linalg.lstsq(
        design, flattened, rcond=None
    )
    fitted = design @ coefficients
    residual_norm = float(np.linalg.norm(flattened - fitted))
    derivatives = tuple(
        np.asarray(row, dtype=float).reshape(value_shape)
        for row in coefficients[1:]
    )
    return TaylorDerivativeFit(
        derivatives=derivatives,
        rank=int(rank),
        residual_norm=residual_norm,
    )
