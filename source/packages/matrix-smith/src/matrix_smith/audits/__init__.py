"""Explicit, opt-in numerical audit implementations.

Production ORACLE--SMITH construction and LINK runtime code must not import
from this namespace.  It is reserved for small-system reference checks,
regression certification, and export validation.
"""

from .dense import (
    DENSE_AUDIT_SCHEMA,
    dense_gic_chart_condition,
    dense_gic_tangent_diagnostics,
    dense_primitive_chart_condition,
    normalize_dense_rows,
    normalized_dense_condition_and_rank,
    normalized_dense_rank,
)

__all__ = [
    "DENSE_AUDIT_SCHEMA",
    "dense_gic_chart_condition",
    "dense_gic_tangent_diagnostics",
    "dense_primitive_chart_condition",
    "normalize_dense_rows",
    "normalized_dense_condition_and_rank",
    "normalized_dense_rank",
]
