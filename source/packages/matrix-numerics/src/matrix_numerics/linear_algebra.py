"""Shared, scale-aware rank and conditioning policy for MATRIX."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SingularSpectrum:
    """One immutable SVD rank/condition certificate."""

    singular_values: np.ndarray
    rank: int
    maximum_rank: int
    cutoff: float
    condition_number: float
    minimum_active: float
    maximum: float


@dataclass(frozen=True)
class AdaptiveRankCertificate:
    """Rank decision including precision, spectral-gap and backward-error evidence."""

    spectrum: SingularSpectrum
    platform_cutoff: float
    gap_cutoff: float
    boundary_gap_ratio: float
    relative_backward_error: float
    precision: str

    @property
    def rank(self) -> int:
        return self.spectrum.rank

    @property
    def condition_number(self) -> float:
        return self.spectrum.condition_number


def adaptive_singular_spectrum(
    matrix: np.ndarray,
    *,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
    normalize_rows: bool = False,
    zero_row_tolerance: float = 0.0,
    precision_safety_factor: float = 32.0,
    minimum_gap_ratio: float = 1.0e3,
) -> AdaptiveRankCertificate:
    """Choose rank from hard limits plus precision-scaled gap evidence.

    A spectral gap may raise, but can never lower, the caller's absolute or
    relative cutoff.  Gap adaptation is allowed only inside the conventional
    square-root-epsilon uncertainty band, preventing physical soft modes from
    being discarded merely because they are separated from stiff modes.
    """

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("adaptive rank input must be a finite two-dimensional matrix")
    factor = float(precision_safety_factor)
    gap_required = float(minimum_gap_ratio)
    if factor <= 0.0 or gap_required <= 1.0:
        raise ValueError("adaptive rank safety factor/gap ratio are invalid")
    analyzed = values
    if normalize_rows and values.shape[0]:
        norms = np.linalg.norm(values, axis=1)
        zero = _nonnegative_finite(zero_row_tolerance, "zero-row tolerance")
        if np.any(norms <= zero):
            analyzed = np.zeros((0, values.shape[1]), dtype=values.dtype)
        else:
            analyzed = values / norms[:, None]
    singular = np.linalg.svd(analyzed, compute_uv=False) if analyzed.size else np.zeros(0)
    maximum = float(singular[0]) if singular.size else 0.0
    epsilon = float(np.finfo(values.dtype).eps)
    platform_cutoff = factor * epsilon * max(values.shape, default=0) * maximum
    hard_cutoff = max(
        _nonnegative_finite(absolute_tolerance, "absolute tolerance"),
        _nonnegative_finite(relative_tolerance, "relative tolerance") * maximum,
        platform_cutoff,
    )
    gap_cutoff = 0.0
    if singular.size > 1 and maximum > 0.0:
        ratios = singular[:-1] / np.maximum(singular[1:], np.finfo(float).tiny)
        boundary = int(np.argmax(ratios))
        if (
            float(ratios[boundary]) >= gap_required
            and singular[boundary + 1] <= np.sqrt(epsilon) * maximum
        ):
            gap_cutoff = float(np.sqrt(singular[boundary] * singular[boundary + 1]))
    cutoff = max(hard_cutoff, gap_cutoff)
    spectrum = _singular_spectrum_from_values(
        singular,
        maximum_rank=min(values.shape),
        absolute_tolerance=cutoff,
        relative_tolerance=0.0,
    )
    rank = spectrum.rank
    inactive = float(singular[rank]) if rank < singular.size else 0.0
    gap_ratio = (
        float(singular[rank - 1] / max(inactive, np.finfo(float).tiny))
        if 0 < rank < singular.size
        else float("inf")
    )
    backward = inactive / maximum if maximum > 0.0 else 0.0
    return AdaptiveRankCertificate(
        spectrum=spectrum,
        platform_cutoff=platform_cutoff,
        gap_cutoff=gap_cutoff,
        boundary_gap_ratio=gap_ratio,
        relative_backward_error=backward,
        precision=str(values.dtype),
    )


def singular_spectrum(
    matrix: np.ndarray,
    *,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
    normalize_rows: bool = False,
    zero_row_tolerance: float = 0.0,
) -> SingularSpectrum:
    """Return the canonical SVD certificate used by scientific packages.

    The active cutoff is ``max(abs_tol, rel_tol * s_max)``.  Row
    normalization is explicit because it changes the metric and is therefore
    never inferred from matrix shape or units.
    """

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("singular-spectrum input must be a two-dimensional matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError("singular-spectrum input contains non-finite values")
    absolute = _nonnegative_finite(absolute_tolerance, "absolute tolerance")
    relative = _nonnegative_finite(relative_tolerance, "relative tolerance")
    zero_row = _nonnegative_finite(zero_row_tolerance, "zero-row tolerance")
    analyzed = values
    if normalize_rows and values.shape[0]:
        norms = np.linalg.norm(values, axis=1)
        if np.any(norms <= zero_row):
            return _singular_spectrum_from_values(
                np.zeros(0, dtype=float),
                maximum_rank=min(values.shape),
                absolute_tolerance=absolute,
                relative_tolerance=relative,
            )
        analyzed = values / norms[:, None]
    singular = np.linalg.svd(analyzed, compute_uv=False) if analyzed.size else np.zeros(0)
    return _singular_spectrum_from_values(
        singular,
        maximum_rank=min(values.shape),
        absolute_tolerance=absolute,
        relative_tolerance=relative,
    )


def spectrum_rank(
    singular_values: np.ndarray,
    *,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
) -> int:
    """Apply the common cutoff policy to an already computed spectrum."""

    values = np.asarray(singular_values, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("singular values must be one-dimensional and finite")
    return _singular_spectrum_from_values(
        values,
        maximum_rank=len(values),
        absolute_tolerance=_nonnegative_finite(
            absolute_tolerance,
            "absolute tolerance",
        ),
        relative_tolerance=_nonnegative_finite(
            relative_tolerance,
            "relative tolerance",
        ),
    ).rank


def numerical_matrix_rank(
    matrix: np.ndarray,
    *,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
) -> int:
    return singular_spectrum(
        matrix,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    ).rank


def normalized_matrix_condition(
    matrix: np.ndarray,
    *,
    absolute_tolerance: float,
    relative_tolerance: float = 0.0,
    zero_row_tolerance: float | None = None,
    required_rank: int | None = None,
) -> float:
    """Return row-normalized condition, or infinity without required rank."""

    certificate = singular_spectrum(
        matrix,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        normalize_rows=True,
        zero_row_tolerance=(
            absolute_tolerance if zero_row_tolerance is None else zero_row_tolerance
        ),
    )
    expected = certificate.maximum_rank if required_rank is None else int(required_rank)
    return (
        certificate.condition_number
        if certificate.rank == expected
        else float("inf")
    )


def _singular_spectrum_from_values(
    singular_values: np.ndarray,
    *,
    maximum_rank: int,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> SingularSpectrum:
    singular = np.asarray(singular_values, dtype=float)
    maximum = float(singular[0]) if singular.size else 0.0
    cutoff = max(absolute_tolerance, relative_tolerance * maximum)
    rank = int(np.count_nonzero(singular > cutoff))
    minimum_active = float(singular[rank - 1]) if rank else 0.0
    condition = (
        float(maximum / minimum_active)
        if maximum > 0.0 and minimum_active > 0.0
        else float("inf")
    )
    return SingularSpectrum(
        singular_values=singular,
        rank=rank,
        maximum_rank=int(maximum_rank),
        cutoff=float(cutoff),
        condition_number=condition,
        minimum_active=minimum_active,
        maximum=maximum,
    )


def _nonnegative_finite(value: float, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return parsed


__all__ = [
    "AdaptiveRankCertificate",
    "SingularSpectrum",
    "adaptive_singular_spectrum",
    "normalized_matrix_condition",
    "numerical_matrix_rank",
    "singular_spectrum",
    "spectrum_rank",
]
