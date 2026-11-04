"""Compatibility facade for generic property surfaces now owned by matrix-qm."""

from matrix_qm.property_surface import (
    ARCHITECT_PROPERTY_SURFACE_PROBLEM_SCHEMA,
    ARCHITECT_PROPERTY_SURFACE_RESULT_SCHEMA,
    ArchitectPropertyComponent,
    ArchitectPropertyEvaluation,
    ArchitectPropertySurfaceProblem,
    ArchitectPropertySurfaceResult,
    evaluate_architect_property_surface,
    fit_architect_property_surface,
)

__all__ = [
    "ARCHITECT_PROPERTY_SURFACE_PROBLEM_SCHEMA",
    "ARCHITECT_PROPERTY_SURFACE_RESULT_SCHEMA",
    "ArchitectPropertyComponent",
    "ArchitectPropertyEvaluation",
    "ArchitectPropertySurfaceProblem",
    "ArchitectPropertySurfaceResult",
    "evaluate_architect_property_surface",
    "fit_architect_property_surface",
]
