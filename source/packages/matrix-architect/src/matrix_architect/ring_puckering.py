"""Construction facade and compatibility exports for ZAFF ring puckering."""

from typing import Any, Sequence

import numpy as np

from matrix_zaff.ring_puckering import *  # noqa: F401,F403
from matrix_zaff.ring_puckering import (
    ZaffRingPuckeringChart,
    ZaffRingPuckeringGeometryPoint,
    ZaffRingPuckeringScanPoint,
    build_ring_puckering_scan_plan,
    native_components_from_ring_puckering_point,
)

def build_ring_puckering_geometry_scan(
    definition: Any,
    reference_coordinates_angstrom: np.ndarray,
    chart: ZaffRingPuckeringChart,
    *,
    points: Sequence[ZaffRingPuckeringScanPoint] | None = None,
    tolerance: float = 1.0e-9,
) -> tuple[ZaffRingPuckeringGeometryPoint, ...]:
    """Realize a vertical RPck scan while keeping the other frozen SONICs fixed."""

    from matrix_link import nonlinear_internal_coordinate_step
    from matrix_smith import build_gic_b_matrix, evaluate_gic_values

    reference = np.asarray(reference_coordinates_angstrom, dtype=float)
    reference_values = evaluate_gic_values(definition, coordinates_angstrom=reference)
    selected = (
        tuple(points)
        if points is not None
        else build_ring_puckering_scan_plan(
            chart,
            reference_values[np.asarray(chart.source_coordinate_indices, dtype=int)],
        )
    )

    def evaluate(coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (
            evaluate_gic_values(definition, coordinates_angstrom=coordinates),
            np.asarray(
                build_gic_b_matrix(definition, coordinates_angstrom=coordinates).rows,
                dtype=float,
            ),
        )

    output = []
    indices = np.asarray(chart.source_coordinate_indices, dtype=int)
    for point in selected:
        native = native_components_from_ring_puckering_point(chart, point)
        target = np.array(reference_values, copy=True)
        target[indices] = native
        # The ordinary hybrid ring predictor intentionally realizes phase-only
        # moves at nearly fixed radial amplitude. A complete Q/theta/phi
        # surface instead requires all native RPck components to reach their
        # explicit targets, so use the full frozen-chart solver here.
        result = nonlinear_internal_coordinate_step(
            reference,
            target,
            evaluate,
            tolerance=tolerance,
            max_iterations=400,
            max_cartesian_step_angstrom=0.08,
        )
        if not result.converged:
            raise ValueError(
                f"ring-puckering back-transform failed at {point.identifier}; "
                f"residual={float(np.linalg.norm(result.residual)):.3e}"
            )
        actual_values, _actual_b = evaluate(result.coordinates_angstrom)
        actual_residual = target - actual_values
        output.append(
            ZaffRingPuckeringGeometryPoint(
                scan_point=point,
                target_native_components=tuple(float(value) for value in native),
                achieved_native_components=tuple(float(value) for value in actual_values[indices]),
                coordinates_angstrom=np.asarray(result.coordinates_angstrom, dtype=float),
                residual_norm=float(np.linalg.norm(actual_residual)),
            )
        )
    return tuple(output)

