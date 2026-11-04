"""Compatibility facade for ZAFF seed compilation and resident execution.

New ARCHITECT implementation code imports :mod:`matrix_architect.seed_compiler`;
runtime consumers import :mod:`matrix_zaff.seed_runtime` directly.
"""

from matrix_zaff.seed_runtime import (
    ZAFF_FAST_RIGID_TORSIONAL_ZOOM,
    ZAFF_SEED_MODEL_SCHEMA,
    ZaffSeedCoverageError,
    ZaffSeedDynamicsRuntime,
    ZaffSecondOrderAnchoredSeedRuntime,
    ZaffSeedEvaluation,
    evaluate_zaff_fast_rigid_torsional_batch_energies,
    evaluate_zaff_seed_model,
)

from .seed_compiler import (
    ZAFF_SEED_PRIOR_SCHEMA,
    assemble_fragment_cm5_charges,
    build_zaff_seed_force_field,
    compile_zaff_seed_model,
    zaff_seed_full_fit_priors,
)

__all__ = [
    "ZAFF_FAST_RIGID_TORSIONAL_ZOOM",
    "ZAFF_SEED_MODEL_SCHEMA",
    "ZAFF_SEED_PRIOR_SCHEMA",
    "ZaffSeedCoverageError",
    "ZaffSeedDynamicsRuntime",
    "ZaffSecondOrderAnchoredSeedRuntime",
    "ZaffSeedEvaluation",
    "assemble_fragment_cm5_charges",
    "build_zaff_seed_force_field",
    "compile_zaff_seed_model",
    "evaluate_zaff_fast_rigid_torsional_batch_energies",
    "evaluate_zaff_seed_model",
    "zaff_seed_full_fit_priors",
]
