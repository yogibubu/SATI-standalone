"""Shared numerical kernels and compute-backend selection for MATRIX."""

from .compute import (
    COMPUTE_BACKEND_SCHEMA,
    DEFAULT_GPU_MIN_BATCH,
    ComputeBackend,
    available_compute_backends,
    compute_backend_report,
    coreml_compute_unit,
    gpu_validation_enabled,
    neural_engine_validation_enabled,
    resolve_compute_backend,
    resolve_neural_inference_backend,
)
from .diagonalizer import (
    DiagonalizerBackend,
    EighResult,
    available_diagonalizer_backends,
    best_diagonalizer_backend,
    diagonalize_hermitian,
    eigh_arrays,
    eigvalsh_array,
)
from .finite_difference import (
    ALLOWED_FINITE_DIFFERENCE_STENCILS,
    DEFAULT_FINITE_DIFFERENCE_STENCIL,
    TaylorDerivativeFit,
    ScalarFiniteDifferenceResult,
    fit_taylor_derivatives,
    normalize_finite_difference_stencil,
    scalar_finite_difference,
)
from .iterative_eigensolver import (
    IterativeEigenResult,
    diaglib_available,
    iterative_eigensolver_backends,
    solve_symmetric_matrix_free,
)
from .native import (
    DEFAULT_NATIVE_MIN_WORK,
    NATIVE_BACKEND_SCHEMA,
    NativeBackend,
    resolve_native_backend,
)
from .rank_revealing import (
    RankRevealingRowSelection,
    select_rank_revealing_rows,
)
from .linear_algebra import (
    AdaptiveRankCertificate,
    SingularSpectrum,
    adaptive_singular_spectrum,
    normalized_matrix_condition,
    numerical_matrix_rank,
    singular_spectrum,
    spectrum_rank,
)

from .trust_region import TrustRegionLMStep, svd_trust_region_lm_step

__all__ = [
    "TrustRegionLMStep",
    "svd_trust_region_lm_step",
    "COMPUTE_BACKEND_SCHEMA",
    "DEFAULT_GPU_MIN_BATCH",
    "DEFAULT_NATIVE_MIN_WORK",
    "ComputeBackend",
    "DiagonalizerBackend",
    "EighResult",
    "IterativeEigenResult",
    "NATIVE_BACKEND_SCHEMA",
    "NativeBackend",
    "RankRevealingRowSelection",
    "AdaptiveRankCertificate",
    "SingularSpectrum",
    "TaylorDerivativeFit",
    "ScalarFiniteDifferenceResult",
    "ALLOWED_FINITE_DIFFERENCE_STENCILS",
    "DEFAULT_FINITE_DIFFERENCE_STENCIL",
    "available_compute_backends",
    "adaptive_singular_spectrum",
    "available_diagonalizer_backends",
    "best_diagonalizer_backend",
    "compute_backend_report",
    "coreml_compute_unit",
    "diagonalize_hermitian",
    "diaglib_available",
    "eigh_arrays",
    "eigvalsh_array",
    "iterative_eigensolver_backends",
    "gpu_validation_enabled",
    "fit_taylor_derivatives",
    "normalize_finite_difference_stencil",
    "scalar_finite_difference",
    "neural_engine_validation_enabled",
    "resolve_compute_backend",
    "resolve_neural_inference_backend",
    "resolve_native_backend",
    "normalized_matrix_condition",
    "numerical_matrix_rank",
    "select_rank_revealing_rows",
    "singular_spectrum",
    "solve_symmetric_matrix_free",
    "spectrum_rank",
]
