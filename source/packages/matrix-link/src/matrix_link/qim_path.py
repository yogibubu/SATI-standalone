"""LINK coordinate models for relaxed Qim paths in a frozen SONIC contract."""

from __future__ import annotations

import numpy as np

from .optimizer import OptimizerCoordinateModel


def qim_displaced_sonic_values(
    reference_values: np.ndarray,
    sonic_direction: np.ndarray,
    displacement: float,
) -> np.ndarray:
    """Return the absolute SONIC target for one rectilinear Qim predictor."""

    reference = np.asarray(reference_values, dtype=float).reshape(-1)
    direction = np.asarray(sonic_direction, dtype=float).reshape(-1)
    if reference.shape != direction.shape or not np.all(np.isfinite(direction)):
        raise ValueError("Qim direction must match the frozen SONIC contract")
    if not np.isfinite(displacement):
        raise ValueError("Qim displacement must be finite")
    return reference + float(displacement) * direction


def qim_transverse_coordinate_model(
    sonic_model: OptimizerCoordinateModel,
    transverse_sonic_basis: np.ndarray,
    *,
    labels: tuple[str, ...] = (),
) -> OptimizerCoordinateModel:
    """Build a LINK optimization model spanning only path-transverse SONIC modes.

    Columns of ``transverse_sonic_basis`` express the transverse variables in
    the underlying frozen SONIC coordinates.  Away from a stationary point
    they must be obtained from a curvature-corrected internal Hessian (including
    the gradient-dependent second derivative of the coordinate transform).
    Merely rediagonalizing a Cartesian Hessian or updating a basis does not make
    the modes curvilinear.  The projected variables are explicitly
    dimensionless; LINK's Cartesian metric retains their physical scaling.
    """

    if sonic_model.kind != "sonic":
        raise ValueError("Qim transverse optimization requires a SONIC coordinate model")
    if sonic_model.sonic_from_coordinates is not None:
        raise ValueError("Qim model must be built from the unprojected SONIC contract")
    basis = np.asarray(transverse_sonic_basis, dtype=float)
    coordinate_count = len(sonic_model.labels)
    if basis.ndim != 2 or basis.shape[0] != coordinate_count:
        raise ValueError("transverse basis must have one row per SONIC coordinate")
    if basis.shape[1] != coordinate_count - 1:
        raise ValueError("a one-dimensional Qim path requires nSONIC-1 transverse modes")
    if not np.all(np.isfinite(basis)) or np.linalg.matrix_rank(basis) != basis.shape[1]:
        raise ValueError("transverse SONIC basis must be finite and full column rank")
    resolved_labels = labels or tuple(f"QIM_T{index + 1:03d}" for index in range(basis.shape[1]))
    if len(resolved_labels) != basis.shape[1]:
        raise ValueError("transverse labels must match the transverse basis")
    directions = basis.T @ np.asarray(sonic_model.directions_angstrom, dtype=float)
    return OptimizerCoordinateModel(
        kind="sonic",
        labels=tuple(str(item) for item in resolved_labels),
        directions_angstrom=directions,
        metric_diagonal=np.ones(basis.shape[1], dtype=float),
        sonic_labels=tuple(sonic_model.labels),
        sonic_from_coordinates=basis,
        projected_unit_scales=np.ones(basis.shape[1], dtype=float),
        exact_projected_targets=True,
        sonic_definition=sonic_model.sonic_definition,
        pes_exploration=sonic_model.pes_exploration,
        retained_group=sonic_model.retained_group,
    )
