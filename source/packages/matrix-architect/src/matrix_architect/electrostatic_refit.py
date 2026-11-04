"""Offline residual fits; no mutation of force-field libraries or runtimes.

The caller selects an existing reference, or GFN-FF only for missing cases,
and supplies complete electrostatic derivatives including charge response.
Parameter estimation belongs to ARCHITECT, not to the force evaluator.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class ResidualScan:
    """One coordinate scan: columns are energy and coordinate derivative.

    Energies are in hartree and derivatives in hartree/coordinate-unit.
    ``derivative_scale`` has coordinate units and balances energy and slope
    residuals. Each scan has its own energy anchor, not an adjustable offset.
    """

    coordinate: np.ndarray
    reference: np.ndarray
    electrostatics: np.ndarray
    fixed: np.ndarray
    anchor: int
    derivative_scale: float
    provenance: str

    def __post_init__(self):
        x = np.array(self.coordinate, dtype=float, copy=True)
        if x.ndim != 1 or len(x) < 3 or not np.all(np.isfinite(x)):
            raise ValueError("a finite one-dimensional scan needs at least three points")
        if not 0 <= self.anchor < len(x) or not self.provenance:
            raise ValueError("scan needs a valid energy anchor and provenance")
        if not np.isfinite(self.derivative_scale) or self.derivative_scale <= 0:
            raise ValueError("derivative scale must be finite and positive")
        for name in ("reference", "electrostatics", "fixed"):
            array = np.array(getattr(self, name), dtype=float, copy=True)
            if array.shape != (len(x), 2) or not np.all(np.isfinite(array)):
                raise ValueError("scan energy/derivative arrays have inconsistent dimensions")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        x.setflags(write=False)
        object.__setattr__(self, "coordinate", x)

    @property
    def target(self):
        result = self.reference - self.electrostatics - self.fixed
        result[:, 0] -= result[self.anchor, 0]
        return result


def evaluate_scalar_scan(potential, coordinate):
    rows = []
    for value in coordinate:
        derivative = potential.derivatives(float(value))
        rows.append((derivative.energy, derivative.first))
    return np.asarray(rows)


def fit_residual_scans(scans, potential_factory, initial_parameters, bounds):
    """Fit one shared scalar potential to independent scans of its family.

    Numerical differences, if used by scipy, differentiate *parameters*, not
    molecular coordinates. The supplied energy/coordinate gradients and the
    fitted runtime potential derivatives are analytic.
    """
    scans = tuple(scans)
    if not scans:
        raise ValueError("at least one residual scan is required")
    targets = tuple(scan.target for scan in scans)

    def residual(parameters):
        potential = potential_factory(parameters)
        rows = []
        for scan, target in zip(scans, targets, strict=True):
            prediction = evaluate_scalar_scan(potential, scan.coordinate)
            prediction[:, 0] -= prediction[scan.anchor, 0]
            error = prediction - target
            error[:, 1] *= scan.derivative_scale
            rows.append(error.ravel())
        return np.concatenate(rows)

    result = least_squares(residual, initial_parameters, bounds=bounds,
                           x_scale="jac", ftol=1e-12, xtol=1e-12, gtol=1e-12,
                           max_nfev=2000)
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"residual fit failed: {result.message}")
    potential = potential_factory(result.x)
    diagnostics = []
    for scan, target in zip(scans, targets, strict=True):
        prediction = evaluate_scalar_scan(potential, scan.coordinate)
        prediction[:, 0] -= prediction[scan.anchor, 0]
        error = prediction - target
        diagnostics.append({"provenance": scan.provenance,
                            "energy_rmse_hartree": float(np.sqrt(np.mean(error[:, 0]**2))),
                            "gradient_rmse": float(np.sqrt(np.mean(error[:, 1]**2)))})
    singular = np.linalg.svd(result.jac, compute_uv=False)
    return result.x, {
        "schema": "matrix.architect.electrostatic_residual_fit.v1",
        "parameters": result.x.tolist(), "success": bool(result.success),
        "weighted_cost": float(result.cost), "optimality": float(result.optimality),
        "parameter_evaluations": int(result.nfev),
        "active_bounds": result.active_mask.tolist(),
        "jacobian_singular_values": singular.tolist(),
        "scans": diagnostics, "source_parameters_modified": False,
    }
