"""Compatibility facade for generic constrained fitting now owned by matrix-qm."""

from matrix_qm.surface_fit import (
    ARCHITECT_LINEAR_FIT_PROBLEM_SCHEMA,
    ARCHITECT_LINEAR_FIT_RESULT_SCHEMA,
    ArchitectLinearConstraint,
    ArchitectLinearFitDiagnostics,
    ArchitectLinearFitProblem,
    ArchitectLinearFitResult,
    fit_architect_linear_surface,
)

__all__ = [
    "ARCHITECT_LINEAR_FIT_PROBLEM_SCHEMA",
    "ARCHITECT_LINEAR_FIT_RESULT_SCHEMA",
    "ArchitectLinearConstraint",
    "ArchitectLinearFitDiagnostics",
    "ArchitectLinearFitProblem",
    "ArchitectLinearFitResult",
    "fit_architect_linear_surface",
]
