"""Dense reference algebra, isolated from production coordinate paths."""

from __future__ import annotations

import numpy as np

from matrix_numerics import numerical_matrix_rank, normalized_matrix_condition, singular_spectrum


DENSE_AUDIT_SCHEMA = "matrix.smith.dense_audit.v1"


def normalized_dense_rank(rows: np.ndarray, *, absolute_tolerance: float,
                          zero_row_tolerance: float | None = None) -> int:
    matrix = np.asarray(rows, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("dense audit rows must be two-dimensional")
    if not matrix.size:
        return 0
    norms = np.linalg.norm(matrix, axis=1)
    limit = float(absolute_tolerance if zero_row_tolerance is None else zero_row_tolerance)
    if not np.all(np.isfinite(norms)) or np.any(norms <= limit):
        return 0
    return numerical_matrix_rank(matrix / norms[:, None], absolute_tolerance=float(absolute_tolerance))


def normalized_dense_condition_and_rank(rows: np.ndarray, *, rank_tolerance: float) -> tuple[float, int]:
    matrix = np.asarray(rows, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("dense audit rows must be two-dimensional")
    if not len(matrix):
        return 1.0, 0
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1.0e-12):
        return float("inf"), 0
    singular = np.linalg.svd(matrix / norms[:, None], compute_uv=False)
    rank = int(np.count_nonzero(singular > float(rank_tolerance) * singular[0]))
    return (float(singular[0] / singular[-1]) if singular[-1] else float("inf")), rank


def dense_gic_tangent_diagnostics(definition, coordinates_angstrom: np.ndarray) -> dict[str, float | int]:
    from ..evaluation import build_gic_b_matrix
    rows = np.asarray(build_gic_b_matrix(
        definition, coordinates_angstrom=np.asarray(coordinates_angstrom, dtype=float)
    ).rows, dtype=float)
    spectrum = singular_spectrum(rows, absolute_tolerance=1.0e-10)
    singular = spectrum.singular_values
    return {"wilson_tangent_rank": spectrum.rank,
            "wilson_tangent_singular_min": float(singular[-1]) if singular.size else 0.0,
            "wilson_tangent_singular_max": float(singular[0]) if singular.size else 0.0}


def dense_gic_chart_condition(definition, *, rank_tolerance: float) -> float:
    from ..evaluation import build_gic_b_matrix
    rows = np.asarray(build_gic_b_matrix(definition).rows, dtype=float)
    if not rows.size:
        return 1.0
    return normalized_matrix_condition(rows, absolute_tolerance=float(rank_tolerance),
                                       zero_row_tolerance=float(rank_tolerance),
                                       required_rank=len(definition.gics))


def dense_primitive_chart_condition(primitives, coordinates_angstrom: np.ndarray, *, rank_tolerance: float) -> float:
    from ..numerics import _analytic_b_row
    selected = tuple(primitives)
    if not selected:
        return 1.0
    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    rows = np.vstack([_analytic_b_row(primitive, coordinates) for primitive in selected])
    return normalized_matrix_condition(rows, absolute_tolerance=float(rank_tolerance),
                                       zero_row_tolerance=float(rank_tolerance),
                                       required_rank=len(selected))


def normalize_dense_rows(rows: np.ndarray, *, tolerance: float) -> np.ndarray:
    values = np.asarray(rows, dtype=float)
    if values.ndim != 2:
        raise ValueError("dense audit rows must be two-dimensional")
    if not len(values):
        return values.copy()
    norms = np.linalg.norm(values, axis=1)
    return np.vstack([row / norm if norm > float(tolerance) else row
                      for row, norm in zip(values, norms, strict=True)])
