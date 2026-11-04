"""Explicit derivative and Wilson-metric contracts for SONIC coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import GICBMatrix


@dataclass(frozen=True)
class SonicMetricContract:
    """Mass-weighted Wilson metric and its numerical audit metadata."""

    metric: np.ndarray
    inverse_metric: np.ndarray
    masses_amu: tuple[float, ...]
    rank: int
    condition_number: float
    eigenvalues: tuple[float, ...]
    symmetry_residual: float
    positive_semidefinite: bool
    rank_tolerance: float


@dataclass(frozen=True)
class SonicDerivativeContract:
    """Declared derivative capabilities of one frozen SONIC B matrix."""

    b_matrix: np.ndarray
    metric: SonicMetricContract
    b_mode: str = "ANALYTIC"
    b_prime_mode: str = "AUDIT_ONLY"
    hessian_mode: str = "CURVILINEAR_CHAIN_RULE_REQUIRED"


def build_sonic_metric_contract(
    b_matrix: GICBMatrix | np.ndarray,
    masses_amu: tuple[float, ...] | list[float] | np.ndarray,
    *,
    rank_tolerance: float = 1.0e-10,
) -> SonicMetricContract:
    """Build and audit ``G = B M^-1 B.T`` for a frozen coordinate set."""

    b = _as_b_matrix(b_matrix)
    masses = np.asarray(tuple(masses_amu), dtype=float)
    if masses.ndim != 1 or masses.size * 3 != b.shape[1]:
        raise ValueError("one positive mass is required for each atom in the B matrix")
    if not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("SONIC masses must be finite and strictly positive")
    tolerance = float(rank_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("SONIC metric rank_tolerance must be positive and finite")

    inverse_mass = np.repeat(1.0 / masses, 3)
    metric = np.asarray((b * inverse_mass) @ b.T, dtype=float)
    symmetric = 0.5 * (metric + metric.T)
    symmetry_residual = float(np.max(np.abs(metric - metric.T), initial=0.0))
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues), initial=0.0)), 1.0)
    cutoff = tolerance * scale
    active = eigenvalues > cutoff
    rank = int(np.count_nonzero(active))
    if rank:
        inverse_metric = (eigenvectors[:, active] / eigenvalues[active]) @ eigenvectors[:, active].T
        condition_number = float(eigenvalues[active][-1] / eigenvalues[active][0])
    else:
        inverse_metric = np.zeros_like(symmetric)
        condition_number = float("inf")
    positive_semidefinite = bool(float(np.min(eigenvalues, initial=0.0)) >= -cutoff)
    if not positive_semidefinite:
        raise ValueError("SONIC Wilson metric is not positive semidefinite")
    return SonicMetricContract(
        metric=symmetric,
        inverse_metric=inverse_metric,
        masses_amu=tuple(float(value) for value in masses),
        rank=rank,
        condition_number=condition_number,
        eigenvalues=tuple(float(value) for value in eigenvalues),
        symmetry_residual=symmetry_residual,
        positive_semidefinite=positive_semidefinite,
        rank_tolerance=tolerance,
    )


def build_sonic_derivative_contract(
    b_matrix: GICBMatrix | np.ndarray,
    masses_amu: tuple[float, ...] | list[float] | np.ndarray,
    *,
    rank_tolerance: float = 1.0e-10,
    b_prime_mode: str = "AUDIT_ONLY",
    hessian_mode: str = "CURVILINEAR_CHAIN_RULE_REQUIRED",
) -> SonicDerivativeContract:
    """Return the explicit derivative contract for one SONIC definition."""

    b = _as_b_matrix(b_matrix)
    prime_mode = str(b_prime_mode).strip().upper()
    hess_mode = str(hessian_mode).strip().upper()
    if prime_mode not in {"AUDIT_ONLY", "ANALYTIC"}:
        raise ValueError("unsupported SONIC B-prime mode")
    if not hess_mode or hess_mode == "UNSPECIFIED":
        raise ValueError("SONIC Hessian mode must be explicitly declared")
    return SonicDerivativeContract(
        b_matrix=b,
        metric=build_sonic_metric_contract(
            b_matrix,
            masses_amu,
            rank_tolerance=rank_tolerance,
        ),
        b_prime_mode=prime_mode,
        hessian_mode=hess_mode,
    )


def _as_b_matrix(b_matrix: GICBMatrix | np.ndarray) -> np.ndarray:
    if isinstance(b_matrix, GICBMatrix):
        array = np.asarray(b_matrix.rows, dtype=float)
    else:
        array = np.asarray(b_matrix, dtype=float)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError("SONIC B matrix must be a finite two-dimensional array")
    return array
