"""LINK state rules for an adaptive SMITH coordinate-basis epoch."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


LINK_ADAPTIVE_CHART_SCHEMA = "matrix.link.adaptive_chart.v1"


@dataclass(frozen=True)
class AdaptiveChartEpochTransition:
    gradient: np.ndarray
    hessian: np.ndarray
    reset_gdiis: bool
    reset_quasi_newton_memory: bool
    epoch_increment: int
    schema: str = LINK_ADAPTIVE_CHART_SCHEMA


def change_adaptive_chart_epoch(
    cartesian_gradient: np.ndarray,
    cartesian_hessian: np.ndarray,
    new_cartesian_to_coordinate: np.ndarray,
) -> AdaptiveChartEpochTransition:
    """Transport current tensors and start a clean optimizer-history epoch.

    The map is supplied by SMITH. LINK infers neither chemistry nor masses.
    GDIIS and quasi-Newton secants are not transported because their finite
    differences belong to the old nonlinear chart.
    """

    gradient_x = np.asarray(cartesian_gradient, dtype=float).reshape(-1)
    hessian_x = np.asarray(cartesian_hessian, dtype=float)
    transform = np.asarray(new_cartesian_to_coordinate, dtype=float)
    if hessian_x.shape != (gradient_x.size, gradient_x.size):
        raise ValueError("Cartesian Hessian and gradient dimensions differ")
    if transform.ndim != 2 or transform.shape[1] != gradient_x.size:
        raise ValueError("new chart tangent map has the wrong Cartesian dimension")
    if not all(np.all(np.isfinite(value)) for value in (gradient_x, hessian_x, transform)):
        raise ValueError("adaptive chart transport contains non-finite data")
    return AdaptiveChartEpochTransition(
        gradient=transform @ gradient_x,
        hessian=transform @ hessian_x @ transform.T,
        reset_gdiis=True,
        reset_quasi_newton_memory=True,
        epoch_increment=1,
    )


__all__ = [
    "LINK_ADAPTIVE_CHART_SCHEMA",
    "AdaptiveChartEpochTransition",
    "change_adaptive_chart_epoch",
]
