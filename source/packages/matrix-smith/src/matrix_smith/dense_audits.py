"""Backward-compatible import path for explicit dense SMITH audits."""

from .audits.dense import (
    DENSE_AUDIT_SCHEMA,
    dense_gic_chart_condition,
    dense_gic_tangent_diagnostics,
    dense_primitive_chart_condition,
    normalize_dense_rows,
    normalized_dense_condition_and_rank,
    normalized_dense_rank,
)

__all__ = [
    "DENSE_AUDIT_SCHEMA", "dense_gic_chart_condition",
    "dense_gic_tangent_diagnostics", "dense_primitive_chart_condition",
    "normalize_dense_rows", "normalized_dense_condition_and_rank",
    "normalized_dense_rank",
]
