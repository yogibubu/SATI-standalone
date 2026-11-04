from __future__ import annotations

"""Provider-independent routing between variational and perturbative modes."""

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from matrix_core import ScientificValidationError


@dataclass(frozen=True)
class PathModeSelectionSettings:
    """Numerical acceptance criteria for a large-amplitude normal-mode block."""

    minimum_principal_overlap: float = 0.90
    minimum_projection_gap: float = 0.10
    maximum_active_projection: float = 0.20
    minimum_tangent_singular_value: float = 1.0e-6
    maximum_mode_orthogonality_error: float = 1.0e-5

    def validate(self) -> None:
        for name in (
            "minimum_principal_overlap",
            "minimum_projection_gap",
            "maximum_active_projection",
        ):
            value = float(getattr(self, name))
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.minimum_tangent_singular_value <= 0.0:
            raise ValueError("minimum_tangent_singular_value must be positive")
        if self.maximum_mode_orthogonality_error <= 0.0:
            raise ValueError("maximum_mode_orthogonality_error must be positive")


@dataclass(frozen=True)
class PathModeSelection:
    """Auditable division of harmonic modes into path and perturbative blocks."""

    path_mode_indices: tuple[int, ...]
    active_mode_indices: tuple[int, ...]
    projection_scores: np.ndarray
    principal_overlaps: np.ndarray
    projection_gap: float
    maximum_active_projection: float
    mode_orthogonality_error: float
    tangent_rank: int
    settings: PathModeSelectionSettings


@dataclass(frozen=True)
class HybridModeBand:
    mode_index: int
    harmonic_cm: float
    fundamental_cm: float | None
    treatment: str
    path_projection: float


@dataclass(frozen=True)
class VariationalBand:
    label: str
    transition_cm: float
    lower_state: int = 0
    upper_state: int = 1


@dataclass(frozen=True)
class HybridVibrationalSpectrum:
    ordinary_modes: tuple[HybridModeBand, ...]
    variational_bands: tuple[VariationalBand, ...]
    path_mode_indices: tuple[int, ...]


def mass_weighted_align(
    coordinates: np.ndarray,
    reference: np.ndarray,
    masses_amu: np.ndarray,
) -> np.ndarray:
    """Rigidly align one Cartesian geometry to a reference using atomic masses."""
    moving = _coordinates(coordinates, "coordinates")
    target = _coordinates(reference, "reference")
    masses = _masses(masses_amu, len(target))
    if moving.shape != target.shape:
        raise ValueError("coordinates and reference must have the same shape")
    weights = masses / float(np.sum(masses))
    moving_center = np.sum(weights[:, None] * moving, axis=0)
    target_center = np.sum(weights[:, None] * target, axis=0)
    moving_zero = moving - moving_center
    target_zero = target - target_center
    covariance = (moving_zero * weights[:, None]).T @ target_zero
    left, _, right_t = np.linalg.svd(covariance)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
    return moving_zero @ rotation + target_center


def centered_path_tangent(
    lower_coordinates: np.ndarray,
    upper_coordinates: np.ndarray,
    reference_coordinates: np.ndarray,
    masses_amu: np.ndarray,
    *,
    lower_parameter: float | None = None,
    upper_parameter: float | None = None,
) -> np.ndarray:
    """Return an aligned central path tangent in Cartesian displacement units."""
    lower = mass_weighted_align(lower_coordinates, reference_coordinates, masses_amu)
    upper = mass_weighted_align(upper_coordinates, reference_coordinates, masses_amu)
    scale = 1.0
    if lower_parameter is not None or upper_parameter is not None:
        if lower_parameter is None or upper_parameter is None:
            raise ValueError("both path parameters are required when either is supplied")
        scale = float(upper_parameter) - float(lower_parameter)
        if scale <= 0.0:
            raise ValueError("upper_parameter must exceed lower_parameter")
    tangent = (upper - lower).reshape(-1) / scale
    if not np.all(np.isfinite(tangent)) or np.linalg.norm(tangent) == 0.0:
        raise ScientificValidationError("aligned path tangent is zero or non-finite")
    return tangent


def select_path_mode_block(
    normal_modes: np.ndarray,
    path_tangents: np.ndarray,
    masses_amu: np.ndarray,
    *,
    settings: PathModeSelectionSettings | None = None,
) -> PathModeSelection:
    """Select the normal-mode subspace representing one large-amplitude block.

    The selection maximizes the projection onto the mass-weighted tangent
    subspace.  Principal overlaps validate the whole subspace, so rotations
    within a near-degenerate path block do not create an artificial failure.
    """
    options = settings or PathModeSelectionSettings()
    options.validate()
    modes = np.asarray(normal_modes, dtype=float)
    tangents = np.asarray(path_tangents, dtype=float)
    if modes.ndim != 2:
        raise ValueError("normal_modes must be a two-dimensional array")
    if tangents.ndim == 1:
        tangents = tangents.reshape((1, -1))
    if tangents.ndim != 2 or tangents.shape[1] != modes.shape[1]:
        raise ValueError("path_tangents must have the same Cartesian width as normal_modes")
    if not np.all(np.isfinite(modes)) or not np.all(np.isfinite(tangents)):
        raise ValueError("normal modes and path tangents must be finite")
    atom_count, remainder = divmod(modes.shape[1], 3)
    if remainder:
        raise ValueError("normal-mode Cartesian width must be divisible by three")
    masses = _masses(masses_amu, atom_count)
    if tangents.shape[0] > modes.shape[0]:
        raise ScientificValidationError("path tangent rank cannot exceed the mode count")

    mass_scale = np.sqrt(np.repeat(masses, 3))
    weighted_modes = _normalized_rows(modes * mass_scale[None, :], "normal modes")
    orthogonality_error = float(
        np.max(np.abs(weighted_modes @ weighted_modes.T - np.eye(len(weighted_modes))))
    )
    if orthogonality_error > options.maximum_mode_orthogonality_error:
        raise ScientificValidationError(
            "normal modes are not orthogonal under the supplied mass metric: "
            f"maximum error {orthogonality_error:.3e}, allowed "
            f"{options.maximum_mode_orthogonality_error:.3e}"
        )
    weighted_tangents = tangents * mass_scale[None, :]
    _, tangent_singular, tangent_right = np.linalg.svd(weighted_tangents, full_matrices=False)
    if tangent_singular.size == 0:
        raise ScientificValidationError("no path tangent was supplied")
    relative_singular = tangent_singular / tangent_singular[0]
    tangent_rank = int(np.sum(relative_singular >= options.minimum_tangent_singular_value))
    if tangent_rank != tangents.shape[0]:
        raise ScientificValidationError(
            "path tangents are linearly dependent under the mass metric: "
            f"rank {tangent_rank} for {tangents.shape[0]} vectors"
        )
    tangent_basis = tangent_right[:tangent_rank]
    projections = weighted_modes @ tangent_basis.T
    scores = np.sum(projections * projections, axis=1)
    # For an orthonormal normal-mode basis, the sum of squared principal
    # overlaps of a selected coordinate subspace is exactly the sum of these
    # per-mode projection scores.  The top-k choice is therefore the global
    # optimum; no combinatorial subset search is required.
    selected = tuple(
        sorted(np.argsort(-scores, kind="stable")[:tangent_rank].tolist())
    )
    selected_basis = _row_space(weighted_modes[np.asarray(selected, dtype=int)])
    principal = np.linalg.svd(tangent_basis @ selected_basis.T, compute_uv=False)
    if principal.size != tangent_rank or float(np.min(principal)) < options.minimum_principal_overlap:
        raise ScientificValidationError(
            "large-amplitude normal-mode subspace fails the principal-overlap criterion: "
            f"minimum {float(np.min(principal)):.6f}, required "
            f"{options.minimum_principal_overlap:.6f}"
        )
    selected_scores = scores[np.asarray(selected, dtype=int)]
    remaining = tuple(index for index in range(len(modes)) if index not in selected)
    best_rejected = max((float(scores[index]) for index in remaining), default=0.0)
    gap = float(np.min(selected_scores)) - best_rejected
    if gap < options.minimum_projection_gap:
        raise ScientificValidationError(
            "large-amplitude mode selection is ambiguous: projection-score gap "
            f"{gap:.6f}, required {options.minimum_projection_gap:.6f}"
        )
    maximum_active = max((float(scores[index]) for index in remaining), default=0.0)
    if maximum_active > options.maximum_active_projection:
        raise ScientificValidationError(
            "an active perturbative mode retains excessive path character: "
            f"projection {maximum_active:.6f}, allowed {options.maximum_active_projection:.6f}"
        )
    return PathModeSelection(
        path_mode_indices=tuple(index + 1 for index in selected),
        active_mode_indices=tuple(index + 1 for index in remaining),
        projection_scores=scores,
        principal_overlaps=principal,
        projection_gap=gap,
        maximum_active_projection=maximum_active,
        mode_orthogonality_error=orthogonality_error,
        tangent_rank=tangent_rank,
        settings=options,
    )


def assemble_hybrid_spectrum(
    harmonic_frequencies_cm: Sequence[float],
    perturbative_fundamentals_cm: Mapping[int, float],
    selection: PathModeSelection,
    variational_bands: Sequence[VariationalBand],
) -> HybridVibrationalSpectrum:
    """Combine ordinary perturbative bands with a separate variational block."""
    harmonic = np.asarray(harmonic_frequencies_cm, dtype=float)
    if harmonic.ndim != 1 or len(harmonic) != len(selection.projection_scores):
        raise ValueError("harmonic frequencies must match the selected normal-mode count")
    active = set(selection.active_mode_indices)
    unexpected = set(int(index) for index in perturbative_fundamentals_cm) - active
    missing = active - set(int(index) for index in perturbative_fundamentals_cm)
    if unexpected:
        raise ScientificValidationError(
            f"perturbative fundamentals contain excluded path modes: {sorted(unexpected)}"
        )
    if missing:
        raise ScientificValidationError(
            f"missing perturbative fundamentals for active modes: {sorted(missing)}"
        )
    ordinary = tuple(
        HybridModeBand(
            mode_index=index,
            harmonic_cm=float(harmonic[index - 1]),
            fundamental_cm=float(perturbative_fundamentals_cm[index]),
            treatment="perturbative",
            path_projection=float(selection.projection_scores[index - 1]),
        )
        for index in selection.active_mode_indices
    )
    bands = tuple(variational_bands)
    for band in bands:
        if band.upper_state <= band.lower_state:
            raise ValueError("variational band upper_state must exceed lower_state")
        if not np.isfinite(band.transition_cm) or band.transition_cm <= 0.0:
            raise ValueError("variational transition must be finite and positive")
    return HybridVibrationalSpectrum(ordinary, bands, selection.path_mode_indices)


def _row_space(rows: np.ndarray) -> np.ndarray:
    _, singular, right = np.linalg.svd(np.asarray(rows, dtype=float), full_matrices=False)
    if singular.size == 0 or singular[-1] <= 1.0e-12 * singular[0]:
        raise ScientificValidationError("selected normal modes are linearly dependent")
    return right


def _normalized_rows(rows: np.ndarray, label: str) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1)
    if np.any(norms <= 0.0):
        raise ScientificValidationError(f"{label} contain a zero vector")
    return rows / norms[:, None]


def _coordinates(value: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be a finite (N, 3) array")
    return array


def _masses(value: np.ndarray, atom_count: int) -> np.ndarray:
    masses = np.asarray(value, dtype=float)
    if masses.shape != (atom_count,) or not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("masses_amu must contain one finite positive mass per atom")
    return masses
