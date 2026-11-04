from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np
from matrix_numerics import fit_taylor_derivatives

from .harmonic import GFResult, HESSIAN_EIGENVALUE_TO_CM, solve_wilson_gf
from .internal import BOHR_TO_ANGSTROM


@dataclass(frozen=True)
class GFDiagonalScalingResult:
    """Coordinate-diagonal multilevel scaling data."""

    factors: np.ndarray
    effective_force_constants: np.ndarray
    low_level_diagonal: np.ndarray
    high_level_diagonal: np.ndarray
    off_diagonal_scaling: str = "geometric"
    symmetry_zeroed_pairs: int = 0


@dataclass(frozen=True)
class GFSelectiveDiagonalScalingResult:
    """Primitive-component weighted diagonal multilevel scaling data."""

    selected_primitive_indices: tuple[int, ...]
    selected_primitive_labels: tuple[str, ...]
    source_coordinate_indices: tuple[int, ...]
    component_weights: np.ndarray
    high_level_diagonal: np.ndarray
    scaling: GFDiagonalScalingResult


@dataclass(frozen=True)
class GFDelocalizedInternalBasis:
    """Non-redundant delocalized internal-coordinate basis from primitive B rows."""

    b_matrix: np.ndarray
    u_matrix: np.ndarray
    eigenvalues: np.ndarray
    rank: int


@dataclass(frozen=True)
class GFEmpiricalClassScalingFactor:
    """One fitted diagonal scaling factor for a coordinate class."""

    name: str
    factor: float
    support: int
    mean_abs_log_residual: float


@dataclass(frozen=True)
class GFEmpiricalClassScalingFit:
    """Empirical coordinate-class scaling model fitted from matched force fields."""

    factors: tuple[GFEmpiricalClassScalingFactor, ...]
    class_labels: tuple[str, ...]
    assigned_factors: np.ndarray
    default_factor: float


@dataclass(frozen=True)
class GFEmpiricalClassScalingResult:
    """Application of an empirical class-scaling model to one force field."""

    fit: GFEmpiricalClassScalingFit
    scaling: GFDiagonalScalingResult


@dataclass(frozen=True)
class GFFrequencyErrorMetrics:
    """Summary error metrics for two harmonic frequency sets."""

    rms_delta_cm: float
    max_abs_delta_cm: float
    mean_abs_delta_cm: float
    deltas_cm: np.ndarray


@dataclass(frozen=True)
class GFModeOverlapResult:
    """Absolute mode-overlap diagnostics after deterministic greedy matching."""

    overlap_matrix: np.ndarray
    assignment: tuple[tuple[int, int, float], ...]
    mean_assigned_overlap: float
    min_assigned_overlap: float


@dataclass(frozen=True)
class GFAdaptiveCoordinateMetadata:
    """Metadata for one SONIC coordinate used by adaptive GF refinement.

    Indices in ``neighbors`` are zero-based.  Empty labels mean that the
    corresponding constraint or descriptor is unavailable, not that two
    coordinates are equivalent.
    """

    identifier: str
    family: str = ""
    irrep: str = ""
    fragment_id: str = ""
    ring_id: str = ""
    synthon_id: str = ""
    neighbors: tuple[int, ...] = ()
    protected: bool = False


@dataclass(frozen=True)
class GFAdaptiveRankingWeights:
    """Configurable terms used to rank admissible off-diagonal acquisitions."""

    diagonal_correction: float = 1.0
    normal_mode_proximity: float = 1.0
    coupling_magnitude: float = 1.0
    same_family: float = 0.25
    same_fragment: float = 0.25
    same_ring: float = 0.25
    same_synthon: float = 0.25
    topological_distance: float = 0.25
    b_row_similarity: float = 0.25
    protected: float = 0.10


@dataclass(frozen=True)
class GFAdaptiveCostModel:
    """Relative acquisition costs for adaptive high-level information."""

    diagonal: float = 1.0
    off_diagonal: float = 4.0
    same_family_discount: float = 0.8
    same_fragment_discount: float = 0.9


@dataclass(frozen=True)
class GFAdaptiveStoppingCriteria:
    """Stopping criteria for adaptive multilevel GF refinement."""

    rms_frequency_cm: float | None = None
    max_frequency_cm: float | None = None
    min_mode_overlap: float | None = None
    max_cost: float | None = None
    min_benefit_cost: float = 0.0
    max_cycles: int = 1
    batch_size: int = 1


@dataclass(frozen=True)
class GFAdaptiveObjectiveWeights:
    """Weights for the monotonic refinement objective."""

    frequency: float = 1.0
    mode_overlap: float = 1.0
    cost: float = 0.0
    hessian: float = 0.0


@dataclass(frozen=True)
class GFAdaptiveCandidate:
    """One admissible off-diagonal high-level acquisition candidate."""

    i: int
    j: int
    benefit: float
    cost: float
    benefit_cost: float
    score: float
    reason: str


@dataclass(frozen=True)
class GFAdaptiveCycle:
    """Audit record for one adaptive refinement cycle."""

    cycle: int
    accepted_pairs: tuple[tuple[int, int], ...]
    rejected_pairs: tuple[tuple[int, int], ...]
    objective: float
    total_cost: float
    stop_reason: str = ""


@dataclass(frozen=True)
class GFAdaptiveValidation:
    """Validation diagnostics against a complete high-level Hessian."""

    hessian_rms_error: float
    frequency_metrics: GFFrequencyErrorMetrics
    mode_overlap: GFModeOverlapResult
    mac_matrix: np.ndarray
    harmonic_energy_delta_rms: float


@dataclass(frozen=True)
class GFGeometricOffDiagonalDiagnostics:
    """Accuracy of geometric-mean off-diagonal scaling against a full reference."""

    off_diagonal_rms_error: float
    off_diagonal_relative_rms_error: float
    off_diagonal_correlation: float
    diagonal_rms_error: float


@dataclass(frozen=True)
class GFAdaptiveConvergencePoint:
    """One point of a selected-element convergence scan."""

    acquired_pairs: int
    acquired_fraction: float
    frequency_rms_cm: float
    frequency_max_cm: float
    mean_mode_overlap: float
    min_mode_overlap: float
    hessian_rms_error: float
    harmonic_energy_delta_rms: float


@dataclass(frozen=True)
class GFAdaptiveConvergenceScan:
    """Prefix scan of L0 plus selected L1 Hessian elements toward full L1."""

    points: tuple[GFAdaptiveConvergencePoint, ...]
    candidates: tuple[GFAdaptiveCandidate, ...]
    geometric_scaling: GFGeometricOffDiagonalDiagnostics


@dataclass(frozen=True)
class GFIterativeDiagonalStage:
    """One stage of two-basis diagonal L1 reconstruction."""

    label: str
    basis: str
    frequency_rms_cm: float
    frequency_max_cm: float
    mean_mode_overlap: float
    min_mode_overlap: float
    hessian_rms_error: float


@dataclass(frozen=True)
class GFIterativeDiagonalRescalingResult:
    """Validation of repeated diagonal-only L1 acquisition in updated bases."""

    stages: tuple[GFIterativeDiagonalStage, ...]
    first_pass_force_constants: np.ndarray
    second_pass_force_constants: np.ndarray
    updated_mode_basis: np.ndarray


@dataclass(frozen=True)
class GFAdaptiveModeDiagonalRescalingResult:
    """Validation of mode-change-triggered second diagonal L1 acquisition."""

    stages: tuple[GFIterativeDiagonalStage, ...]
    first_pass_force_constants: np.ndarray
    adaptive_second_pass_force_constants: np.ndarray
    updated_mode_basis: np.ndarray
    selected_mode_indices: tuple[int, ...]
    mode_overlap_threshold: float
    first_pass_mode_overlaps: np.ndarray


@dataclass(frozen=True)
class GFAdaptiveSecondPassCycle:
    """One batch of high-level curvatures along first-pass normal modes."""

    cycle: int
    acquired_mode_indices: tuple[int, ...]
    total_acquired: int
    observed_rms_change_cm: float
    observed_max_change_cm: float
    predicted_remaining_rms_cm: float
    predicted_remaining_max_cm: float
    reference_residual_rms_cm: float
    reference_residual_max_cm: float
    order_risk_mode_indices: tuple[int, ...] = ()
    stop_reason: str = ""


@dataclass(frozen=True)
class GFAdaptiveSecondPassResult:
    """Adaptive high-level acquisition in the modes defined by a full SONIC first pass."""

    stages: tuple[GFIterativeDiagonalStage, ...]
    first_pass_force_constants: np.ndarray
    adaptive_second_pass_force_constants: np.ndarray
    updated_mode_basis: np.ndarray
    selected_mode_indices: tuple[int, ...]
    first_pass_mode_overlaps: np.ndarray
    off_diagonal_model_spread_cm: np.ndarray
    ranking_scores: np.ndarray
    ranking_strategy: str
    cycles: tuple[GFAdaptiveSecondPassCycle, ...]
    stop_reason: str
    rms_change_threshold_cm: float
    max_change_threshold_cm: float
    batch_size: int
    patience: int
    same_irrep_curvature_order_inversions: int


@dataclass(frozen=True)
class GFConcordantModeProjectionResult:
    """Validation of diagonal high-level projection in a low-level mode basis."""

    frequency_metrics: GFFrequencyErrorMetrics
    mode_overlap: GFModeOverlapResult
    low_level_frequencies_cm: np.ndarray
    projected_frequencies_cm: np.ndarray
    reference_frequencies_cm: np.ndarray
    projected_force_constants: np.ndarray
    projected_diagonal: np.ndarray
    l0_mode_basis: np.ndarray
    mode_start: int


@dataclass(frozen=True)
class GFModeFirstSonicSelectionResult:
    """Mode-first L1 diagonal transformed to SONIC for sparse-pair selection."""

    high_level_mode_diagonal: np.ndarray
    low_level_mode_curvatures: np.ndarray
    l0_mode_basis: np.ndarray
    mode_projected_force_constants: np.ndarray
    symmetry_projected_l0_force_constants: np.ndarray
    symmetry_projected_g_matrix: np.ndarray
    scaled_l0_force_constants: np.ndarray
    candidates: tuple[GFAdaptiveCandidate, ...]
    curvature_order_inversions: tuple[tuple[int, int], ...]
    ranking_strategy: str
    off_diagonal_scaling: str


@dataclass(frozen=True)
class GFAdaptiveRefinementResult:
    """Result of adaptive multilevel GF refinement."""

    initial_force_constants: np.ndarray
    final_force_constants: np.ndarray
    initial_frequencies_cm: np.ndarray
    final_frequencies_cm: np.ndarray
    initial_modes: np.ndarray
    final_modes: np.ndarray
    candidates: tuple[GFAdaptiveCandidate, ...]
    cycles: tuple[GFAdaptiveCycle, ...]
    acquired_pairs: tuple[tuple[int, int], ...]
    total_cost: float
    objective: float
    validation: GFAdaptiveValidation | None
    stop_reason: str


@dataclass(frozen=True)
class GFScanAcquisitionDecision:
    """Acquisition method selected for one coordinate."""

    index: int
    label: str
    method: str
    max_relative_coupling: float
    coupled_partners: tuple[int, ...]
    estimated_cost: float
    reason: str


@dataclass(frozen=True)
class GFScanAcquisitionPlan:
    """Energy/gradient acquisition plan for L1 scans in a fixed coordinate basis."""

    decisions: tuple[GFScanAcquisitionDecision, ...]
    energy_indices: tuple[int, ...]
    gradient_indices: tuple[int, ...]
    selected_pairs: tuple[tuple[int, int], ...]
    total_estimated_cost: float
    relative_coupling_matrix: np.ndarray
    gradient_threshold: float
    basis: str = "sonic"


@dataclass(frozen=True)
class GFAcquiredForceConstants:
    """Multilevel force field assembled from external scan derivatives."""

    force_constants: np.ndarray
    scaling: GFDiagonalScalingResult
    acquired_diagonal: np.ndarray
    acquired_off_diagonal_pairs: tuple[tuple[int, int], ...]
    basis: str = "sonic"


@dataclass(frozen=True)
class GFSemidiagonalCubicDerivatives:
    """Semidiagonal cubic constants from gradient finite differences.

    ``cubic_fjii[j, i]`` stores the projection of ``d2 grad_cart / dq_i2`` on
    coordinate direction ``j``.  Indices are zero-based.
    """

    cubic_fjii: np.ndarray
    scanned_indices: tuple[int, ...]
    basis: str = "sonic"


@dataclass(frozen=True)
class GFAcquiredGFResult:
    """GF/PED result obtained after replacing L0 by acquired L1 force constants."""

    force_field: GFAcquiredForceConstants
    gf_result: object
    acquisition_plan: GFScanAcquisitionPlan | None = None


@dataclass(frozen=True)
class GFFiniteDifferenceStepEstimate:
    """Derivative estimate obtained with one finite-difference step."""

    step: float
    value: float
    residual_norm: float = 0.0
    point_count: int = 0


@dataclass(frozen=True)
class GFFiniteDifferenceStepSelection:
    """Per-coordinate finite-difference step selection diagnostic."""

    coordinate_label: str
    selected_step: float
    selected_value: float
    estimates: tuple[GFFiniteDifferenceStepEstimate, ...]
    relative_spread: float


def _zero_forbidden_irrep_couplings(
    matrix: np.ndarray,
    gic_irreps: tuple[str, ...],
) -> tuple[np.ndarray, int]:
    projected = np.asarray(matrix, dtype=float).copy()
    irreps = tuple(str(value).strip() for value in gic_irreps)
    if irreps and len(irreps) != projected.shape[0]:
        raise ValueError("GIC irrep count must match force-constant dimensions")
    zeroed = 0
    if irreps:
        for i in range(projected.shape[0]):
            for j in range(i + 1, projected.shape[0]):
                if irreps[i] and irreps[j] and irreps[i] != irreps[j]:
                    projected[i, j] = projected[j, i] = 0.0
                    zeroed += 1
    return projected, zeroed


def _symmetry_preserving_eigh(
    matrix: np.ndarray,
    gic_irreps: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize independent irrep blocks without cross-irrep degenerate mixing."""
    symmetric = _symmetric_part(matrix)
    irreps = tuple(str(value).strip() for value in gic_irreps)
    if not irreps:
        return np.linalg.eigh(symmetric)
    if len(irreps) != symmetric.shape[0]:
        raise ValueError("GIC irrep count must match matrix dimensions")
    eigenpairs: list[tuple[float, int, np.ndarray]] = []
    for irrep in dict.fromkeys(irreps):
        indices = np.asarray([idx for idx, value in enumerate(irreps) if value == irrep])
        block_values, block_vectors = np.linalg.eigh(symmetric[np.ix_(indices, indices)])
        for local_index, value in enumerate(block_values):
            vector = np.zeros(symmetric.shape[0], dtype=float)
            vector[indices] = block_vectors[:, local_index]
            eigenpairs.append((float(value), len(eigenpairs), vector))
    eigenpairs.sort(key=lambda item: (item[0], item[1]))
    return (
        np.asarray([item[0] for item in eigenpairs], dtype=float),
        np.column_stack([item[2] for item in eigenpairs]),
    )


def diagonal_high_level_scaling(
    low_level_force_constants: np.ndarray,
    high_level_diagonal: np.ndarray,
    *,
    off_diagonal_scaling: str = "geometric",
    gic_irreps: tuple[str, ...] = (),
    min_abs_diagonal: float = 1.0e-14,
) -> GFDiagonalScalingResult:
    """Build the diagonal high-level effective force field in SONIC coordinates.

    The high-level diagonal is exact by construction.  Residual L0
    off-diagonals may be left unchanged or multiplied by the harmonic,
    geometric, arithmetic, or RMS mean of the two diagonal factors.  When
    irreps are supplied, cross-irrep elements are set exactly to zero before
    any residual scaling.
    """
    f_ll = np.asarray(low_level_force_constants, dtype=float)
    if f_ll.ndim != 2 or f_ll.shape[0] != f_ll.shape[1]:
        raise ValueError("Low-level force constants must be a square matrix")
    if not np.allclose(f_ll, f_ll.T):
        raise ValueError("Low-level force constants must be symmetric")
    hl_diag = np.asarray(high_level_diagonal, dtype=float).reshape(-1)
    if hl_diag.shape != (f_ll.shape[0],):
        raise ValueError(f"High-level diagonal must have length {f_ll.shape[0]}")
    ll_diag = np.diag(f_ll).copy()
    if np.any(np.abs(ll_diag) < float(min_abs_diagonal)):
        raise ValueError("Low-level diagonal contains near-zero force constants")
    factors = hl_diag / ll_diag
    if np.any(~np.isfinite(factors)):
        raise ValueError("Non-finite diagonal scaling factor")
    if np.any(factors < 0.0):
        raise ValueError("Diagonal high-level scaling requires non-negative factors")
    strategy = str(off_diagonal_scaling).strip().lower()
    if strategy not in {"none", "harmonic", "geometric", "arithmetic", "rms"}:
        raise ValueError(
            "off_diagonal_scaling must be none, harmonic, geometric, arithmetic, or rms"
        )
    irreps = tuple(str(value).strip() for value in gic_irreps)
    effective, symmetry_zeroed_pairs = _zero_forbidden_irrep_couplings(f_ll, irreps)
    np.fill_diagonal(effective, hl_diag)
    for i in range(f_ll.shape[0]):
        for j in range(i + 1, f_ll.shape[0]):
            if irreps and irreps[i] and irreps[j] and irreps[i] != irreps[j]:
                multiplier = 0.0
            elif strategy == "none":
                multiplier = 1.0
            elif strategy == "harmonic":
                denominator = factors[i] + factors[j]
                multiplier = 0.0 if denominator == 0.0 else 2.0 * factors[i] * factors[j] / denominator
            elif strategy == "geometric":
                multiplier = float(np.sqrt(factors[i] * factors[j]))
            elif strategy == "arithmetic":
                multiplier = 0.5 * (factors[i] + factors[j])
            else:
                multiplier = float(np.sqrt(0.5 * (factors[i] ** 2 + factors[j] ** 2)))
            effective[i, j] = effective[j, i] = f_ll[i, j] * multiplier
    return GFDiagonalScalingResult(
        factors=factors,
        effective_force_constants=effective,
        low_level_diagonal=ll_diag,
        high_level_diagonal=hl_diag,
        off_diagonal_scaling=strategy,
        symmetry_zeroed_pairs=symmetry_zeroed_pairs,
    )


def plan_gf_l1_scan_acquisition(
    low_level_force_constants: np.ndarray,
    *,
    coordinate_labels: tuple[str, ...] = (),
    coordinate_metadata: tuple[GFAdaptiveCoordinateMetadata, ...] = (),
    energy_cost: float = 1.0,
    gradient_cost: float = 8.0,
    base_coupling_threshold: float = 0.12,
    max_gradient_fraction: float | None = None,
    basis: str = "sonic",
    min_abs_diagonal: float = 1.0e-14,
) -> GFScanAcquisitionPlan:
    """Decide which coordinates need energy scans and which need gradient scans.

    The decision is made from the L0 force-constant matrix in the target basis.
    Coordinates with strong normalized off-diagonal couplings are assigned to
    gradient scans, because one gradient scan supplies a full force-constant
    column.  The gradient threshold is raised when gradients are expensive.
    """

    f_ll = _validate_square_symmetric(low_level_force_constants, "Low-level force constants")
    n_coord = f_ll.shape[0]
    metadata = _complete_metadata(coordinate_metadata, n_coord)
    if coordinate_labels and len(coordinate_labels) != n_coord:
        raise ValueError("Coordinate label count must match force-constant dimensions")
    labels = coordinate_labels or tuple(item.identifier for item in metadata)
    e_cost = float(energy_cost)
    g_cost = float(gradient_cost)
    if e_cost <= 0.0 or g_cost <= 0.0:
        raise ValueError("Energy and gradient costs must be positive")
    diag = np.diag(f_ll)
    if np.any(np.abs(diag) < float(min_abs_diagonal)):
        raise ValueError("Low-level diagonal contains near-zero force constants")
    scale = np.sqrt(np.abs(np.outer(diag, diag)))
    relative = np.zeros_like(f_ll, dtype=float)
    mask = scale > 0.0
    relative[mask] = np.abs(f_ll[mask]) / scale[mask]
    np.fill_diagonal(relative, 0.0)
    threshold = min(0.85, float(base_coupling_threshold) * np.sqrt(g_cost / e_cost))
    pair_set: set[tuple[int, int]] = set()
    for i in range(n_coord):
        for j in range(i + 1, n_coord):
            if relative[i, j] >= threshold and _is_admissible_pair(metadata[i], metadata[j], i, j):
                pair_set.add((i, j))
    uncovered = set(pair_set)
    gradient_list: list[int] = []
    while uncovered:
        scores: list[tuple[float, int, int]] = []
        for index in range(n_coord):
            if index in gradient_list:
                continue
            covered = [pair for pair in uncovered if index in pair]
            if not covered:
                continue
            benefit = float(sum(relative[left, right] for left, right in covered))
            scores.append((benefit / g_cost, len(covered), index))
        if not scores:
            break
        _score, _count, selected = max(scores, key=lambda item: (item[0], item[1], -item[2]))
        gradient_list.append(selected)
        uncovered = {pair for pair in uncovered if selected not in pair}
    gradient_scores = [
        (float(np.sum(relative[index])), index)
        for index in gradient_list
    ]
    if max_gradient_fraction is not None:
        fraction = float(max_gradient_fraction)
        if fraction <= 0.0 or fraction > 1.0:
            raise ValueError("max_gradient_fraction must be in (0, 1]")
        limit = max(1, int(np.ceil(fraction * n_coord)))
        gradient_scores = gradient_scores[:limit]
    gradient_indices = tuple(index for _score, index in gradient_scores)
    gradient_set = set(gradient_indices)
    decisions: list[GFScanAcquisitionDecision] = []
    selected_pairs = {
        pair
        for pair in pair_set
        if pair[0] in gradient_set or pair[1] in gradient_set
    }
    for index in range(n_coord):
        partners = tuple(
            partner
            for partner in range(n_coord)
            if (min(index, partner), max(index, partner)) in selected_pairs and partner != index
        )
        max_coupling = float(np.max(relative[index])) if n_coord > 1 else 0.0
        if index in gradient_set:
            method = "gradient"
            cost = g_cost
            reason = (
                f"max_relative_coupling={max_coupling:.3g} >= threshold={threshold:.3g}; "
                f"partners={','.join(str(item + 1) for item in partners)}"
            )
        else:
            method = "energy"
            cost = e_cost
            reason = (
                f"max_relative_coupling={max_coupling:.3g} < threshold={threshold:.3g} "
                "or gradient not cost-effective"
            )
        decisions.append(
            GFScanAcquisitionDecision(
                index=index,
                label=labels[index],
                method=method,
                max_relative_coupling=max_coupling,
                coupled_partners=partners,
                estimated_cost=cost,
                reason=reason,
            )
        )
    energy_indices = tuple(item.index for item in decisions if item.method == "energy")
    return GFScanAcquisitionPlan(
        decisions=tuple(decisions),
        energy_indices=energy_indices,
        gradient_indices=gradient_indices,
        selected_pairs=tuple(sorted(selected_pairs)),
        total_estimated_cost=float(sum(item.estimated_cost for item in decisions)),
        relative_coupling_matrix=relative,
        gradient_threshold=float(threshold),
        basis=str(basis),
    )


def reconstruct_force_constants_with_acquired_offdiagonals(
    low_level_force_constants: np.ndarray,
    high_level_diagonal: np.ndarray,
    acquired_off_diagonal: np.ndarray | dict[tuple[int, int], float] | None = None,
    *,
    basis: str = "sonic",
    min_abs_diagonal: float = 1.0e-14,
) -> GFAcquiredForceConstants:
    """Scale L0 diagonally and overwrite available L1 off-diagonal elements."""

    f_ll = _validate_square_symmetric(low_level_force_constants, "Low-level force constants")
    scaling = diagonal_high_level_scaling(
        f_ll,
        np.asarray(high_level_diagonal, dtype=float),
        min_abs_diagonal=min_abs_diagonal,
    )
    force_constants = scaling.effective_force_constants.copy()
    acquired_pairs: list[tuple[int, int]] = []
    if acquired_off_diagonal is not None:
        if isinstance(acquired_off_diagonal, dict):
            items = acquired_off_diagonal.items()
        else:
            acquired = np.asarray(acquired_off_diagonal, dtype=float)
            if acquired.shape != f_ll.shape:
                raise ValueError("Acquired off-diagonal matrix must match force-constant shape")
            items = (
                ((i, j), float(acquired[i, j]))
                for i in range(acquired.shape[0])
                for j in range(i + 1, acquired.shape[1])
                if np.isfinite(acquired[i, j])
            )
        for (i_raw, j_raw), value_raw in items:
            i = _checked_index(int(i_raw), f_ll.shape[0], "off-diagonal row")
            j = _checked_index(int(j_raw), f_ll.shape[0], "off-diagonal column")
            if i == j:
                continue
            left, right = (i, j) if i < j else (j, i)
            value = float(value_raw)
            if not np.isfinite(value):
                continue
            force_constants[left, right] = value
            force_constants[right, left] = value
            acquired_pairs.append((left, right))
    return GFAcquiredForceConstants(
        force_constants=_symmetric_part(force_constants),
        scaling=scaling,
        acquired_diagonal=scaling.high_level_diagonal,
        acquired_off_diagonal_pairs=tuple(dict.fromkeys(acquired_pairs)),
        basis=str(basis),
    )


def force_constants_from_external_derivatives(
    low_level_force_constants: np.ndarray,
    coordinate_directions_angstrom: np.ndarray,
    *,
    energy_second_derivatives: np.ndarray | None = None,
    gradient_first_derivatives: dict[int, np.ndarray] | None = None,
    basis: str = "sonic",
    min_abs_diagonal: float = 1.0e-14,
) -> GFAcquiredForceConstants:
    """Assemble an L1/L0 GF force field from energy or gradient scan derivatives.

    ``energy_second_derivatives`` supplies diagonal L1 curvatures.  Each entry in
    ``gradient_first_derivatives`` is the Cartesian gradient derivative with
    respect to one scanned coordinate; projection on all coordinate directions
    gives the corresponding L1 force-constant column, including couplings.
    """

    f_ll = _validate_square_symmetric(low_level_force_constants, "Low-level force constants")
    directions = np.asarray(coordinate_directions_angstrom, dtype=float)
    if directions.ndim != 2 or directions.shape[0] != f_ll.shape[0]:
        raise ValueError("Coordinate directions must have shape ncoord x ncart")
    if not np.all(np.isfinite(directions)):
        raise ValueError("Coordinate directions contain non-finite values")
    n_coord = f_ll.shape[0]
    high_diag = np.diag(f_ll).copy()
    energy_finite = np.zeros(n_coord, dtype=bool)
    if energy_second_derivatives is not None:
        energy_diag = np.asarray(energy_second_derivatives, dtype=float).reshape(-1)
        if energy_diag.shape != (n_coord,):
            raise ValueError(f"Energy second derivatives must have length {n_coord}")
        energy_finite = np.isfinite(energy_diag)
        high_diag[energy_finite] = energy_diag[energy_finite]

    acquired = np.full_like(f_ll, np.nan, dtype=float)
    if gradient_first_derivatives:
        directions_bohr = directions / BOHR_TO_ANGSTROM
        for scanned_index, gradient_derivative in gradient_first_derivatives.items():
            j = _checked_index(int(scanned_index), n_coord, "gradient-scan coordinate")
            grad = np.asarray(gradient_derivative, dtype=float).reshape(-1)
            if grad.shape != (directions.shape[1],):
                raise ValueError(
                    f"Gradient derivative for coordinate {j} has {grad.size} components, "
                    f"expected {directions.shape[1]}"
                )
            column = directions_bohr @ grad
            if not np.all(np.isfinite(column)):
                raise ValueError(f"Gradient derivative for coordinate {j} projects to non-finite values")
            for i in range(n_coord):
                if i == j:
                    if energy_second_derivatives is None or not energy_finite[i]:
                        high_diag[i] = float(column[i])
                    continue
                left, right = (i, j) if i < j else (j, i)
                existing = acquired[left, right]
                acquired[left, right] = (
                    float(column[i])
                    if not np.isfinite(existing)
                    else 0.5 * (float(existing) + float(column[i]))
                )
                acquired[right, left] = acquired[left, right]
    if np.any(~np.isfinite(high_diag)):
        raise ValueError("High-level diagonal contains non-finite values")
    return reconstruct_force_constants_with_acquired_offdiagonals(
        f_ll,
        high_diag,
        acquired,
        basis=basis,
        min_abs_diagonal=min_abs_diagonal,
    )


def force_constants_from_scan_acquisition_plan(
    plan: GFScanAcquisitionPlan,
    low_level_force_constants: np.ndarray,
    coordinate_directions_angstrom: np.ndarray,
    *,
    energy_second_derivatives: dict[int, float] | np.ndarray,
    gradient_first_derivatives: dict[int, np.ndarray] | None = None,
    require_all_planned: bool = True,
    min_abs_diagonal: float = 1.0e-14,
) -> GFAcquiredForceConstants:
    """Build the acquired force field from a GF scan acquisition plan."""

    f_ll = _validate_square_symmetric(low_level_force_constants, "Low-level force constants")
    n_coord = f_ll.shape[0]
    if isinstance(energy_second_derivatives, dict):
        energy_diag = np.full(n_coord, np.nan, dtype=float)
        for index_raw, value_raw in energy_second_derivatives.items():
            index = _checked_index(int(index_raw), n_coord, "energy-scan coordinate")
            energy_diag[index] = float(value_raw)
    else:
        energy_diag = np.asarray(energy_second_derivatives, dtype=float).reshape(-1)
        if energy_diag.shape != (n_coord,):
            raise ValueError(f"Energy second derivatives must have length {n_coord}")
    if require_all_planned:
        missing_energy = [
            index
            for index in plan.energy_indices
            if not np.isfinite(energy_diag[index])
            and index not in (gradient_first_derivatives or {})
        ]
        missing_gradient = [
            index
            for index in plan.gradient_indices
            if index not in (gradient_first_derivatives or {})
        ]
        if missing_energy or missing_gradient:
            raise ValueError(
                "Missing planned L1 scan derivatives: "
                f"energy={tuple(index + 1 for index in missing_energy)} "
                f"gradient={tuple(index + 1 for index in missing_gradient)}"
            )
    return force_constants_from_external_derivatives(
        f_ll,
        coordinate_directions_angstrom,
        energy_second_derivatives=energy_diag,
        gradient_first_derivatives=gradient_first_derivatives,
        basis=plan.basis,
        min_abs_diagonal=min_abs_diagonal,
    )


def semidiagonal_cubic_from_gradient_second_derivatives(
    coordinate_directions_angstrom: np.ndarray,
    gradient_second_derivatives: dict[int, np.ndarray],
    *,
    basis: str = "sonic",
) -> GFSemidiagonalCubicDerivatives:
    """Project gradient second derivatives onto semidiagonal cubic constants."""

    directions = np.asarray(coordinate_directions_angstrom, dtype=float)
    if directions.ndim != 2:
        raise ValueError("Coordinate directions must have shape ncoord x ncart")
    if not np.all(np.isfinite(directions)):
        raise ValueError("Coordinate directions contain non-finite values")
    n_coord, n_cart = directions.shape
    cubic = np.full((n_coord, n_coord), np.nan, dtype=float)
    directions_bohr = directions / BOHR_TO_ANGSTROM
    scanned: list[int] = []
    for scanned_index, gradient_second in gradient_second_derivatives.items():
        i = _checked_index(int(scanned_index), n_coord, "gradient-scan coordinate")
        grad2 = np.asarray(gradient_second, dtype=float).reshape(-1)
        if grad2.shape != (n_cart,):
            raise ValueError(
                f"Gradient second derivative for coordinate {i} has {grad2.size} components, "
                f"expected {n_cart}"
            )
        column = directions_bohr @ grad2
        if not np.all(np.isfinite(column)):
            raise ValueError(f"Gradient second derivative for coordinate {i} projects to non-finite values")
        cubic[:, i] = column
        scanned.append(i)
    return GFSemidiagonalCubicDerivatives(
        cubic_fjii=cubic,
        scanned_indices=tuple(dict.fromkeys(scanned)),
        basis=str(basis),
    )


def semidiagonal_cubic_to_json(cubic: GFSemidiagonalCubicDerivatives) -> dict[str, object]:
    return {
        "schema": "matrix.gf.semidiagonal_cubic_derivatives.v1",
        "basis": cubic.basis,
        "scanned_indices": list(cubic.scanned_indices),
        "cubic_fjii": np.asarray(cubic.cubic_fjii, dtype=float).tolist(),
    }


def write_semidiagonal_cubic_derivatives(
    path: Path | str,
    cubic: GFSemidiagonalCubicDerivatives,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(semidiagonal_cubic_to_json(cubic), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def select_finite_difference_step(
    estimates: tuple[GFFiniteDifferenceStepEstimate, ...] | list[GFFiniteDifferenceStepEstimate],
    *,
    coordinate_label: str = "",
) -> GFFiniteDifferenceStepSelection:
    """Select the most stable finite-difference step for one coordinate."""

    ordered = tuple(sorted(estimates, key=lambda item: float(item.step)))
    if len(ordered) < 2:
        raise ValueError("At least two step estimates are required")
    for estimate in ordered:
        if estimate.step <= 0.0 or not np.isfinite(estimate.step):
            raise ValueError("Finite-difference steps must be positive and finite")
        if not np.isfinite(estimate.value):
            raise ValueError("Finite-difference estimates must be finite")
    values = np.asarray([estimate.value for estimate in ordered], dtype=float)
    scale = max(float(np.max(np.abs(values))), 1.0e-14)
    scores: list[float] = []
    for index, estimate in enumerate(ordered):
        neighbors = []
        if index > 0:
            neighbors.append(abs(estimate.value - ordered[index - 1].value) / scale)
        if index + 1 < len(ordered):
            neighbors.append(abs(estimate.value - ordered[index + 1].value) / scale)
        local_spread = float(np.mean(neighbors)) if neighbors else 0.0
        scores.append(local_spread + abs(float(estimate.residual_norm)) / scale)
    best = int(np.argmin(np.asarray(scores, dtype=float)))
    spread = float(max(scores[best], 0.0))
    return GFFiniteDifferenceStepSelection(
        coordinate_label=coordinate_label,
        selected_step=float(ordered[best].step),
        selected_value=float(ordered[best].value),
        estimates=ordered,
        relative_spread=spread,
    )


def select_energy_second_derivative_step(
    step_result_sets: dict[float, object] | tuple[tuple[float, object], ...],
    *,
    coordinate_label: str = "",
    max_order: int = 4,
) -> GFFiniteDifferenceStepSelection:
    """Choose a coordinate step from energy-based second-derivative estimates."""

    items = step_result_sets.items() if isinstance(step_result_sets, dict) else step_result_sets
    estimates: list[GFFiniteDifferenceStepEstimate] = []
    for step, results in items:
        points = tuple(results)
        completed = tuple(item for item in points if item.status == "completed")
        energies = tuple(item.energy_hartree for item in completed)
        if len(completed) < 3 or any(value is None for value in energies):
            continue
        derivatives = fit_taylor_derivatives(
            tuple(item.displacement for item in completed),
            np.asarray(energies, dtype=float),
            max_order=max_order,
        )
        if len(derivatives.derivatives) < 2:
            continue
        estimates.append(
            GFFiniteDifferenceStepEstimate(
                step=float(step),
                value=float(derivatives.derivatives[1]),
                residual_norm=derivatives.residual_norm,
                point_count=len(points),
            )
        )
    return select_finite_difference_step(tuple(estimates), coordinate_label=coordinate_label)


def primitive_component_presence_weights(
    u_matrix: np.ndarray,
    *,
    primitive_indices: tuple[int, ...] = (),
    primitive_labels: tuple[str, ...] = (),
    requested_primitive_labels: tuple[str, ...] = (),
    source_coordinate_indices: tuple[int, ...] = (),
    source_coordinate_names: tuple[str, ...] = (),
    gic_names: tuple[str, ...] = (),
    min_presence: float = 1.0e-12,
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    """Return per-coordinate percentages for selected primitive components.

    The input ``u_matrix`` is the primitive-to-SONIC coefficient matrix, with
    primitives on rows and final non-redundant coordinates on columns.  A
    component can be selected directly by primitive index/label or indirectly by
    naming a SONIC coordinate; in the latter case the primitive with the largest
    absolute coefficient in that coordinate is selected.  For each SONIC
    coordinate, the returned weight is the fraction of its squared norm carried
    by the selected primitive component(s), clipped to the interval [0, 1].
    """
    u = np.asarray(u_matrix, dtype=float)
    if u.ndim != 2:
        raise ValueError("U matrix must have shape (n_primitives, n_coordinates)")
    n_primitive, n_coord = u.shape
    if n_primitive == 0 or n_coord == 0:
        raise ValueError("U matrix cannot be empty")
    if primitive_labels and len(primitive_labels) != n_primitive:
        raise ValueError("Primitive label count does not match U matrix rows")
    if gic_names and len(gic_names) != n_coord:
        raise ValueError("GIC name count does not match U matrix columns")

    selected: list[int] = []
    source_indices: list[int] = []
    for index in primitive_indices:
        selected.append(_checked_index(index, n_primitive, "primitive"))
    for label in requested_primitive_labels:
        selected.append(_label_index(label, primitive_labels, "primitive"))
    for index in source_coordinate_indices:
        source = _checked_index(index, n_coord, "source coordinate")
        source_indices.append(source)
        selected.append(_dominant_primitive_index(u[:, source], source))
    for name in source_coordinate_names:
        source = _label_index(name, gic_names, "source coordinate")
        source_indices.append(source)
        selected.append(_dominant_primitive_index(u[:, source], source))
    selected_tuple = tuple(dict.fromkeys(selected))
    if not selected_tuple:
        raise ValueError("At least one primitive component or source coordinate is required")

    norms = np.sum(u * u, axis=0)
    if np.any(norms <= 0.0):
        raise ValueError("Every SONIC coordinate must have a non-zero primitive norm")
    presence = np.sum(u[np.asarray(selected_tuple, dtype=int), :] ** 2, axis=0) / norms
    presence = np.clip(presence, 0.0, 1.0)
    threshold = float(min_presence)
    if threshold < 0.0:
        raise ValueError("Minimum component presence must be non-negative")
    presence = np.where(presence >= threshold, presence, 0.0)
    return presence, selected_tuple, tuple(dict.fromkeys(source_indices))


def selective_primitive_component_scaling(
    low_level_force_constants: np.ndarray,
    high_level_diagonal: np.ndarray,
    u_matrix: np.ndarray,
    *,
    primitive_indices: tuple[int, ...] = (),
    primitive_labels: tuple[str, ...] = (),
    requested_primitive_labels: tuple[str, ...] = (),
    source_coordinate_indices: tuple[int, ...] = (),
    source_coordinate_names: tuple[str, ...] = (),
    gic_names: tuple[str, ...] = (),
    min_presence: float = 1.0e-12,
    min_abs_diagonal: float = 1.0e-14,
) -> GFSelectiveDiagonalScalingResult:
    """Apply diagonal L1 information by primitive-component participation.

    A selected primitive component contributes to every SONIC coordinate that
    contains it.  If the component accounts for 30% of a coordinate norm, only
    30% of that coordinate's L0-to-L1 diagonal correction is applied.  The
    resulting diagonal is then passed through the standard geometric-mean
    off-diagonal reconstruction.
    """
    f_ll = _validate_square_symmetric(low_level_force_constants, "Low-level force constants")
    hl_diag = np.asarray(high_level_diagonal, dtype=float).reshape(-1)
    if hl_diag.shape != (f_ll.shape[0],):
        raise ValueError(f"High-level diagonal must have length {f_ll.shape[0]}")
    weights, selected, sources = primitive_component_presence_weights(
        u_matrix,
        primitive_indices=primitive_indices,
        primitive_labels=primitive_labels,
        requested_primitive_labels=requested_primitive_labels,
        source_coordinate_indices=source_coordinate_indices,
        source_coordinate_names=source_coordinate_names,
        gic_names=gic_names,
        min_presence=min_presence,
    )
    if weights.shape != (f_ll.shape[0],):
        raise ValueError("U matrix coordinate count must match force-constant dimensions")
    ll_diag = np.diag(f_ll).copy()
    selective_diagonal = ll_diag + weights * (hl_diag - ll_diag)
    scaling = diagonal_high_level_scaling(
        f_ll,
        selective_diagonal,
        min_abs_diagonal=min_abs_diagonal,
    )
    selected_labels = tuple(
        primitive_labels[index] if primitive_labels else f"primitive:{index}"
        for index in selected
    )
    return GFSelectiveDiagonalScalingResult(
        selected_primitive_indices=selected,
        selected_primitive_labels=selected_labels,
        source_coordinate_indices=sources,
        component_weights=weights,
        high_level_diagonal=selective_diagonal,
        scaling=scaling,
    )


def delocalized_internal_basis_from_b_matrix(
    b_matrix: np.ndarray,
    *,
    rcond: float = 1.0e-10,
) -> GFDelocalizedInternalBasis:
    """Construct a non-redundant delocalized internal basis from primitive rows.

    This is the standard eigenvector construction from the row Gram matrix
    ``B B^T``.  It is deliberately independent of SMITH/SONIC metadata: callers
    can use it as a lightweight fallback when a chemically localized coordinate
    contract is unavailable.
    """
    b_primitive = np.asarray(b_matrix, dtype=float)
    if b_primitive.ndim != 2:
        raise ValueError("B matrix must be two-dimensional")
    if b_primitive.shape[0] == 0 or b_primitive.shape[1] == 0:
        raise ValueError("B matrix cannot be empty")
    threshold = float(rcond)
    if threshold < 0.0:
        raise ValueError("rcond must be non-negative")
    gram = _symmetric_part(b_primitive @ b_primitive.T)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    max_eval = float(eigenvalues[0]) if eigenvalues.size else 0.0
    keep = eigenvalues > max(max_eval * threshold, 0.0)
    kept_values = eigenvalues[keep]
    kept_vectors = eigenvectors[:, keep]
    b_delocalized = kept_vectors.T @ b_primitive
    return GFDelocalizedInternalBasis(
        b_matrix=b_delocalized,
        u_matrix=kept_vectors,
        eigenvalues=kept_values,
        rank=int(kept_values.size),
    )


def coordinate_primitive_class_labels(
    u_matrix: np.ndarray,
    *,
    primitive_labels: tuple[str, ...],
    atomic_numbers: tuple[int, ...] = (),
    gic_names: tuple[str, ...] = (),
    fallback_families: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Assign each internal coordinate to a dominant primitive scaling class."""
    u = np.asarray(u_matrix, dtype=float)
    if u.ndim != 2:
        raise ValueError("U matrix must have shape (n_primitives, n_coordinates)")
    if len(primitive_labels) != u.shape[0]:
        raise ValueError("Primitive label count does not match U matrix rows")
    if gic_names and len(gic_names) != u.shape[1]:
        raise ValueError("GIC name count does not match U matrix columns")
    if fallback_families and len(fallback_families) != u.shape[1]:
        raise ValueError("Fallback family count does not match U matrix columns")
    classes: list[str] = []
    for coord_index in range(u.shape[1]):
        primitive_index = _dominant_primitive_index(u[:, coord_index], coord_index)
        primitive = primitive_labels[primitive_index]
        family = fallback_families[coord_index] if fallback_families else ""
        name = gic_names[coord_index] if gic_names else ""
        classes.append(_primitive_scaling_class(primitive, atomic_numbers, family, name))
    return tuple(classes)


def fit_empirical_diagonal_scaling_factors(
    low_level_force_constants: np.ndarray,
    high_level_force_constants: np.ndarray,
    class_labels: tuple[str, ...],
    *,
    default_factor: float = 1.0,
    aggregation: str = "geometric_mean",
    min_abs_diagonal: float = 1.0e-14,
) -> GFEmpiricalClassScalingFit:
    """Fit diagonal scaling factors by coordinate class from paired force fields."""
    f_ll = _validate_square_symmetric(low_level_force_constants, "Low-level force constants")
    f_hl = _validate_square_symmetric(high_level_force_constants, "High-level force constants")
    if f_ll.shape != f_hl.shape:
        raise ValueError("Low-level and high-level force constants must have the same shape")
    if len(class_labels) != f_ll.shape[0]:
        raise ValueError("Class label count must match force-constant dimensions")
    ll_diag = np.diag(f_ll)
    hl_diag = np.diag(f_hl)
    if np.any(np.abs(ll_diag) < float(min_abs_diagonal)):
        raise ValueError("Low-level diagonal contains near-zero force constants")
    ratios = hl_diag / ll_diag
    if np.any(~np.isfinite(ratios)) or np.any(ratios <= 0.0):
        raise ValueError("Empirical scaling requires positive finite diagonal ratios")
    default = float(default_factor)
    labels = tuple(str(label) if str(label) else "unclassified" for label in class_labels)
    factors: list[GFEmpiricalClassScalingFactor] = []
    assigned = np.full(f_ll.shape[0], default, dtype=float)
    for label in sorted(set(labels)):
        indices = np.asarray([idx for idx, item in enumerate(labels) if item == label], dtype=int)
        values = ratios[indices]
        if aggregation == "geometric_mean":
            factor = float(np.exp(np.mean(np.log(values))))
        elif aggregation == "median":
            factor = float(np.median(values))
        elif aggregation == "mean":
            factor = float(np.mean(values))
        else:
            raise ValueError("aggregation must be geometric_mean, median or mean")
        assigned[indices] = factor
        residual = np.abs(np.log(values / factor))
        factors.append(
            GFEmpiricalClassScalingFactor(
                name=label,
                factor=factor,
                support=int(indices.size),
                mean_abs_log_residual=float(np.mean(residual)) if residual.size else 0.0,
            )
        )
    return GFEmpiricalClassScalingFit(
        factors=tuple(factors),
        class_labels=labels,
        assigned_factors=assigned,
        default_factor=default,
    )


def apply_empirical_diagonal_scaling(
    low_level_force_constants: np.ndarray,
    fit: GFEmpiricalClassScalingFit,
    *,
    min_abs_diagonal: float = 1.0e-14,
) -> GFEmpiricalClassScalingResult:
    """Apply fitted class factors as an L0-only diagonal scaling model."""
    f_ll = _validate_square_symmetric(low_level_force_constants, "Low-level force constants")
    factors = np.asarray(fit.assigned_factors, dtype=float).reshape(-1)
    if factors.shape != (f_ll.shape[0],):
        raise ValueError("Fitted factor count must match force-constant dimensions")
    high_diagonal = np.diag(f_ll) * factors
    return GFEmpiricalClassScalingResult(
        fit=fit,
        scaling=diagonal_high_level_scaling(
            f_ll,
            high_diagonal,
            min_abs_diagonal=min_abs_diagonal,
        ),
    )


def rank_adaptive_couplings(
    low_level_force_constants: np.ndarray,
    high_level_diagonal: np.ndarray,
    *,
    coordinate_metadata: tuple[GFAdaptiveCoordinateMetadata, ...] = (),
    b_matrix: np.ndarray | None = None,
    l0_frequencies_cm: np.ndarray | None = None,
    l0_modes: np.ndarray | None = None,
    ranking_strategy: str = "composite",
    off_diagonal_scaling: str = "geometric",
    ranking_weights: GFAdaptiveRankingWeights | None = None,
    cost_model: GFAdaptiveCostModel | None = None,
    pt2_denominator_floor: float = 1.0e-8,
    min_abs_diagonal: float = 1.0e-14,
) -> tuple[GFAdaptiveCandidate, ...]:
    """Rank admissible off-diagonal elements for adaptive high-level acquisition.

    Symmetry is enforced as a hard constraint before scoring: non-empty,
    different irreducible representations never enter the candidate list.
    Disconnected non-empty fragments are also excluded by default.  The
    ``pt2`` strategy ranks the scaled L0 coupling by the magnitude of its
    second-order eigenvalue shift,
    ``abs(F_ij)**2 / abs(F_ii - F_jj)``, using the available L1 diagonal
    separation and a relative near-degeneracy floor.  A short high-score list
    can be acquired instead of performing a second diagonal pass in the
    updated normal-mode basis; the returned scores define an ordering, not a
    universal selection threshold.
    """
    f_ll = _validate_square_symmetric(low_level_force_constants, "Low-level force constants")
    n_coord = f_ll.shape[0]
    metadata = _complete_metadata(coordinate_metadata, n_coord)
    weights = ranking_weights or GFAdaptiveRankingWeights()
    costs = cost_model or GFAdaptiveCostModel()
    scaling = diagonal_high_level_scaling(
        f_ll,
        high_level_diagonal,
        off_diagonal_scaling=off_diagonal_scaling,
        gic_irreps=tuple(item.irrep for item in metadata),
        min_abs_diagonal=min_abs_diagonal,
    )
    strategy = str(ranking_strategy).strip().lower()
    if strategy not in {"composite", "pt2"}:
        raise ValueError("ranking_strategy must be composite or pt2")
    relative_floor = float(pt2_denominator_floor)
    if not np.isfinite(relative_floor) or relative_floor <= 0.0:
        raise ValueError("pt2_denominator_floor must be a positive finite number")
    high_diag = np.asarray(high_level_diagonal, dtype=float).reshape(-1)
    diagonal_scale = max(float(np.max(np.abs(high_diag))), np.finfo(float).eps)
    absolute_pt2_floor = max(relative_floor * diagonal_scale, np.finfo(float).eps)
    sqrt_scale = np.sqrt(scaling.factors)
    b_rows = _optional_b_matrix(b_matrix, n_coord)
    l0_freq = _optional_vector(l0_frequencies_cm, n_coord, "L0 frequencies")
    mode_rows = _optional_mode_matrix(l0_modes, n_coord)
    max_coupling = float(np.max(np.abs(f_ll - np.diag(np.diag(f_ll)))))
    freq_scale = float(np.ptp(l0_freq)) if l0_freq is not None and l0_freq.size > 1 else 1.0
    if max_coupling <= 0.0:
        max_coupling = 1.0
    if freq_scale <= 0.0:
        freq_scale = 1.0

    candidates: list[GFAdaptiveCandidate] = []
    for i in range(n_coord):
        for j in range(i + 1, n_coord):
            if not _is_admissible_pair(metadata[i], metadata[j], i, j):
                continue
            cost = _candidate_cost(costs, metadata[i], metadata[j])
            if strategy == "pt2":
                coupling = abs(float(scaling.effective_force_constants[i, j]))
                diagonal_separation = abs(float(high_diag[i] - high_diag[j]))
                denominator = max(diagonal_separation, absolute_pt2_floor)
                score = coupling * coupling / denominator
                reason = (
                    f"pt2={score:.6g}; scaled_l0_coupling={coupling:.6g}; "
                    f"l1_diagonal_separation={diagonal_separation:.6g}; "
                    f"denominator={denominator:.6g}"
                )
                candidates.append(
                    GFAdaptiveCandidate(
                        i=i,
                        j=j,
                        benefit=float(score),
                        cost=float(cost),
                        benefit_cost=float(score),
                        score=float(score),
                        reason=reason,
                    )
                )
                continue
            diagonal_term = abs(float(sqrt_scale[i] * sqrt_scale[j] - 1.0))
            coupling_term = abs(float(f_ll[i, j])) / max_coupling
            proximity_term = 0.0
            if l0_freq is not None:
                proximity_term = 1.0 / (1.0 + abs(float(l0_freq[i] - l0_freq[j])) / freq_scale)
            if mode_rows is not None:
                mode_term = _mode_participation_similarity(mode_rows, i, j)
                proximity_term = 0.5 * (proximity_term + mode_term) if l0_freq is not None else mode_term
            topo_term = _topological_proximity(metadata, i, j)
            b_term = _b_row_similarity(b_rows, i, j)
            family_term = _same_nonempty(metadata[i].family, metadata[j].family)
            fragment_term = _same_nonempty(metadata[i].fragment_id, metadata[j].fragment_id)
            ring_term = _same_nonempty(metadata[i].ring_id, metadata[j].ring_id)
            synthon_term = _same_nonempty(metadata[i].synthon_id, metadata[j].synthon_id)
            protected_term = 1.0 if metadata[i].protected or metadata[j].protected else 0.0
            score = (
                weights.diagonal_correction * diagonal_term
                + weights.normal_mode_proximity * proximity_term
                + weights.coupling_magnitude * coupling_term
                + weights.same_family * family_term
                + weights.same_fragment * fragment_term
                + weights.same_ring * ring_term
                + weights.same_synthon * synthon_term
                + weights.topological_distance * topo_term
                + weights.b_row_similarity * b_term
                + weights.protected * protected_term
            )
            ratio = score / cost if cost > 0.0 else float("inf")
            reason = (
                f"diag={diagonal_term:.3g}; prox={proximity_term:.3g}; "
                f"coupling={coupling_term:.3g}; topo={topo_term:.3g}; brow={b_term:.3g}"
            )
            candidates.append(
                GFAdaptiveCandidate(
                    i=i,
                    j=j,
                    benefit=float(score),
                    cost=float(cost),
                    benefit_cost=float(ratio),
                    score=float(score),
                    reason=reason,
                )
            )
    candidates.sort(key=lambda item: (-item.benefit_cost, -item.benefit, item.i, item.j))
    return tuple(candidates)


def adaptive_multilevel_refinement(
    low_level_force_constants: np.ndarray,
    g_matrix: np.ndarray,
    high_level_diagonal: np.ndarray,
    *,
    high_level_force_constants: np.ndarray | None = None,
    coordinate_metadata: tuple[GFAdaptiveCoordinateMetadata, ...] = (),
    b_matrix: np.ndarray | None = None,
    l0_frequencies_cm: np.ndarray | None = None,
    l0_modes: np.ndarray | None = None,
    ranking_strategy: str = "composite",
    off_diagonal_scaling: str = "geometric",
    ranking_weights: GFAdaptiveRankingWeights | None = None,
    cost_model: GFAdaptiveCostModel | None = None,
    pt2_denominator_floor: float = 1.0e-8,
    stopping: GFAdaptiveStoppingCriteria | None = None,
    objective_weights: GFAdaptiveObjectiveWeights | None = None,
    condition_threshold: float = 1.0e-10,
    scale_to_cm: bool = True,
    min_abs_diagonal: float = 1.0e-14,
) -> GFAdaptiveRefinementResult:
    """Run adaptive multilevel GF refinement in a fixed SONIC coordinate basis.

    The complete high-level Hessian, when supplied, is used as a validation and
    selected-element acquisition oracle.  Production workflows can replace that
    oracle by electronic-structure calls returning individual Hessian elements.
    """
    f_ll = _validate_square_symmetric(low_level_force_constants, "Low-level force constants")
    g_mat = _validate_square_symmetric(g_matrix, "G matrix")
    if f_ll.shape != g_mat.shape:
        raise ValueError("Low-level force constants and G matrix must have the same shape")
    _assert_coordinate_conditioning(g_mat, condition_threshold)
    n_coord = f_ll.shape[0]
    metadata = _complete_metadata(coordinate_metadata, n_coord)
    stop = stopping or GFAdaptiveStoppingCriteria()
    obj_weights = objective_weights or GFAdaptiveObjectiveWeights()
    costs = cost_model or GFAdaptiveCostModel()

    scaling = diagonal_high_level_scaling(
        f_ll,
        high_level_diagonal,
        off_diagonal_scaling=off_diagonal_scaling,
        gic_irreps=tuple(item.irrep for item in metadata),
        min_abs_diagonal=min_abs_diagonal,
    )
    initial_f = scaling.effective_force_constants.copy()
    current_f = initial_f.copy()
    initial_result = solve_wilson_gf(current_f, g_mat, scale_to_cm=scale_to_cm)
    current_result = initial_result
    total_cost = float(costs.diagonal * n_coord)

    reference_f = None
    reference_result = None
    if high_level_force_constants is not None:
        reference_f = _validate_square_symmetric(high_level_force_constants, "High-level force constants")
        if reference_f.shape != f_ll.shape:
            raise ValueError("High-level force constants must match low-level shape")
        reference_result = solve_wilson_gf(reference_f, g_mat, scale_to_cm=scale_to_cm)

    candidates = rank_adaptive_couplings(
        f_ll,
        high_level_diagonal,
        coordinate_metadata=metadata,
        b_matrix=b_matrix,
        l0_frequencies_cm=(
            np.asarray(l0_frequencies_cm, dtype=float)
            if l0_frequencies_cm is not None
            else solve_wilson_gf(f_ll, g_mat, scale_to_cm=scale_to_cm).frequencies_cm
        ),
        l0_modes=(
            np.asarray(l0_modes, dtype=float)
            if l0_modes is not None
            else solve_wilson_gf(f_ll, g_mat, scale_to_cm=scale_to_cm).normal_modes
        ),
        ranking_strategy=ranking_strategy,
        off_diagonal_scaling=off_diagonal_scaling,
        ranking_weights=ranking_weights,
        cost_model=costs,
        pt2_denominator_floor=pt2_denominator_floor,
        min_abs_diagonal=min_abs_diagonal,
    )

    current_objective = _adaptive_objective(
        current_f,
        current_result,
        reference_f,
        reference_result,
        total_cost,
        obj_weights,
    )
    if reference_result is not None and _meets_stopping(current_result, reference_result, stop, total_cost):
        validation = validate_adaptive_refinement(current_f, g_mat, reference_f, scale_to_cm=scale_to_cm)
        return GFAdaptiveRefinementResult(
            initial_force_constants=initial_f,
            final_force_constants=current_f,
            initial_frequencies_cm=initial_result.frequencies_cm,
            final_frequencies_cm=current_result.frequencies_cm,
            initial_modes=initial_result.normal_modes,
            final_modes=current_result.normal_modes,
            candidates=candidates,
            cycles=(),
            acquired_pairs=(),
            total_cost=total_cost,
            objective=current_objective,
            validation=validation,
            stop_reason="TARGET_REACHED",
        )
    if reference_f is None:
        return GFAdaptiveRefinementResult(
            initial_force_constants=initial_f,
            final_force_constants=current_f,
            initial_frequencies_cm=initial_result.frequencies_cm,
            final_frequencies_cm=current_result.frequencies_cm,
            initial_modes=initial_result.normal_modes,
            final_modes=current_result.normal_modes,
            candidates=candidates,
            cycles=(),
            acquired_pairs=(),
            total_cost=total_cost,
            objective=current_objective,
            validation=None,
            stop_reason="NO_HIGH_LEVEL_MATRIX",
        )

    acquired: set[tuple[int, int]] = set()
    rejected: set[tuple[int, int]] = set()
    cycles: list[GFAdaptiveCycle] = []
    stop_reason = "MAX_CYCLES"
    for cycle_index in range(1, max(0, int(stop.max_cycles)) + 1):
        batch = [
            candidate
            for candidate in candidates
            if (candidate.i, candidate.j) not in acquired
            and (candidate.i, candidate.j) not in rejected
            and candidate.benefit_cost >= stop.min_benefit_cost
        ][: max(1, int(stop.batch_size))]
        if not batch:
            stop_reason = "NO_CANDIDATE"
            break
        incremental_cost = float(sum(candidate.cost for candidate in batch))
        if stop.max_cost is not None and total_cost + incremental_cost > stop.max_cost:
            stop_reason = "BUDGET_REACHED"
            break

        trial_f = current_f.copy()
        for candidate in batch:
            trial_f[candidate.i, candidate.j] = reference_f[candidate.i, candidate.j]
            trial_f[candidate.j, candidate.i] = reference_f[candidate.i, candidate.j]
        trial_result = solve_wilson_gf(trial_f, g_mat, scale_to_cm=scale_to_cm)
        trial_cost = total_cost + incremental_cost
        trial_objective = _adaptive_objective(
            trial_f,
            trial_result,
            reference_f,
            reference_result,
            trial_cost,
            obj_weights,
        )
        pairs = tuple((candidate.i, candidate.j) for candidate in batch)
        if trial_objective <= current_objective + 1.0e-12:
            current_f = trial_f
            current_result = trial_result
            total_cost = trial_cost
            current_objective = trial_objective
            acquired.update(pairs)
            cycle_stop = ""
            if _meets_stopping(current_result, reference_result, stop, total_cost):
                stop_reason = "TARGET_REACHED"
                cycle_stop = stop_reason
            cycles.append(
                GFAdaptiveCycle(
                    cycle=cycle_index,
                    accepted_pairs=pairs,
                    rejected_pairs=(),
                    objective=current_objective,
                    total_cost=total_cost,
                    stop_reason=cycle_stop,
                )
            )
            if cycle_stop:
                break
        else:
            rejected.update(pairs)
            cycles.append(
                GFAdaptiveCycle(
                    cycle=cycle_index,
                    accepted_pairs=(),
                    rejected_pairs=pairs,
                    objective=current_objective,
                    total_cost=total_cost,
                    stop_reason="ROLLBACK",
                )
            )
    validation = validate_adaptive_refinement(current_f, g_mat, reference_f, scale_to_cm=scale_to_cm)
    return GFAdaptiveRefinementResult(
        initial_force_constants=initial_f,
        final_force_constants=current_f,
        initial_frequencies_cm=initial_result.frequencies_cm,
        final_frequencies_cm=current_result.frequencies_cm,
        initial_modes=initial_result.normal_modes,
        final_modes=current_result.normal_modes,
        candidates=candidates,
        cycles=tuple(cycles),
        acquired_pairs=tuple(sorted(acquired)),
        total_cost=total_cost,
        objective=current_objective,
        validation=validation,
        stop_reason=stop_reason,
    )


def validate_adaptive_refinement(
    force_constants: np.ndarray,
    g_matrix: np.ndarray,
    reference_force_constants: np.ndarray,
    *,
    scale_to_cm: bool = True,
) -> GFAdaptiveValidation:
    """Validate an adaptive force field against a complete high-level reference."""
    candidate_f = _validate_square_symmetric(force_constants, "Candidate force constants")
    reference_f = _validate_square_symmetric(reference_force_constants, "Reference force constants")
    if candidate_f.shape != reference_f.shape:
        raise ValueError("Candidate and reference force constants must have the same shape")
    g_mat = _validate_square_symmetric(g_matrix, "G matrix")
    candidate = solve_wilson_gf(candidate_f, g_mat, scale_to_cm=scale_to_cm)
    reference = solve_wilson_gf(reference_f, g_mat, scale_to_cm=scale_to_cm)
    freq = frequency_error_metrics(candidate.frequencies_cm, reference.frequencies_cm)
    overlap = mode_overlap_diagnostics(candidate.normal_modes, reference.normal_modes)
    eig_delta = candidate.eigenvalues - reference.eigenvalues
    return GFAdaptiveValidation(
        hessian_rms_error=float(np.sqrt(np.mean((candidate_f - reference_f) ** 2))),
        frequency_metrics=freq,
        mode_overlap=overlap,
        mac_matrix=overlap.overlap_matrix**2,
        harmonic_energy_delta_rms=float(np.sqrt(np.mean(eig_delta * eig_delta))),
    )


def geometric_off_diagonal_diagnostics(
    low_level_force_constants: np.ndarray,
    high_level_force_constants: np.ndarray,
    high_level_diagonal: np.ndarray | None = None,
    *,
    min_abs_diagonal: float = 1.0e-14,
) -> GFGeometricOffDiagonalDiagnostics:
    """Compare geometric-mean off-diagonal scaling with full high-level couplings."""
    f_ll = _validate_square_symmetric(low_level_force_constants, "Low-level force constants")
    f_hl = _validate_square_symmetric(high_level_force_constants, "High-level force constants")
    if f_ll.shape != f_hl.shape:
        raise ValueError("Low-level and high-level force constants must have the same shape")
    hl_diag = np.diag(f_hl) if high_level_diagonal is None else np.asarray(high_level_diagonal, dtype=float)
    scaled = diagonal_high_level_scaling(f_ll, hl_diag, min_abs_diagonal=min_abs_diagonal)
    n_coord = f_ll.shape[0]
    mask = ~np.eye(n_coord, dtype=bool)
    delta = scaled.effective_force_constants[mask] - f_hl[mask]
    ref = f_hl[mask]
    ref_rms = float(np.sqrt(np.mean(ref * ref))) if ref.size else 0.0
    corr = float("nan")
    if ref.size > 1:
        left = scaled.effective_force_constants[mask]
        left_norm = float(np.linalg.norm(left))
        reference_norm = float(np.linalg.norm(ref))
        # Pearson correlation is undefined for the exact zero off-diagonal
        # prediction obtained in an L0 normal-mode basis.  Do not turn floating
        # diagonalization noise into a seemingly meaningful near-zero number.
        matrix_scale = float(np.linalg.norm(scaled.effective_force_constants))
        correlation_floor = 1.0e-12 * max(
            matrix_scale, reference_norm, float(np.finfo(float).tiny)
        )
        if (
            left_norm > correlation_floor
            and np.std(left) > correlation_floor / max(np.sqrt(left.size), 1.0)
            and np.std(ref) > 0.0
        ):
            corr = float(np.corrcoef(left, ref)[0, 1])
    diag_delta = np.diag(scaled.effective_force_constants) - np.diag(f_hl)
    return GFGeometricOffDiagonalDiagnostics(
        off_diagonal_rms_error=float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0,
        off_diagonal_relative_rms_error=(
            float(np.sqrt(np.mean(delta * delta)) / ref_rms) if ref_rms > 0.0 else 0.0
        ),
        off_diagonal_correlation=corr,
        diagonal_rms_error=float(np.sqrt(np.mean(diag_delta * diag_delta))),
    )


def adaptive_convergence_scan(
    low_level_force_constants: np.ndarray,
    g_matrix: np.ndarray,
    high_level_force_constants: np.ndarray,
    *,
    coordinate_metadata: tuple[GFAdaptiveCoordinateMetadata, ...] = (),
    b_matrix: np.ndarray | None = None,
    l0_frequencies_cm: np.ndarray | None = None,
    l0_modes: np.ndarray | None = None,
    ranking_strategy: str = "composite",
    off_diagonal_scaling: str = "geometric",
    ranking_weights: GFAdaptiveRankingWeights | None = None,
    cost_model: GFAdaptiveCostModel | None = None,
    pt2_denominator_floor: float = 1.0e-8,
    acquired_pair_counts: tuple[int, ...] | None = None,
    scale_to_cm: bool = True,
    min_abs_diagonal: float = 1.0e-14,
) -> GFAdaptiveConvergenceScan:
    """Scan convergence from diagonal scaling to full L1 by selected off-diagonals."""
    f_ll = _validate_square_symmetric(low_level_force_constants, "Low-level force constants")
    f_hl = _validate_square_symmetric(high_level_force_constants, "High-level force constants")
    g_mat = _validate_square_symmetric(g_matrix, "G matrix")
    if f_ll.shape != f_hl.shape or f_ll.shape != g_mat.shape:
        raise ValueError("Low-level, high-level and G matrices must have the same shape")
    metadata = _complete_metadata(coordinate_metadata, f_ll.shape[0])
    irreps = tuple(item.irrep for item in metadata)
    f_ll, _ = _zero_forbidden_irrep_couplings(f_ll, irreps)
    g_mat, _ = _zero_forbidden_irrep_couplings(g_mat, irreps)
    f_hl, _ = _zero_forbidden_irrep_couplings(f_hl, irreps)
    high_diag = np.diag(f_hl)
    scaled = diagonal_high_level_scaling(
        f_ll,
        high_diag,
        off_diagonal_scaling=off_diagonal_scaling,
        gic_irreps=irreps,
        min_abs_diagonal=min_abs_diagonal,
    )
    candidates = rank_adaptive_couplings(
        f_ll,
        high_diag,
        coordinate_metadata=metadata,
        b_matrix=b_matrix,
        l0_frequencies_cm=l0_frequencies_cm,
        l0_modes=l0_modes,
        ranking_strategy=ranking_strategy,
        off_diagonal_scaling=off_diagonal_scaling,
        ranking_weights=ranking_weights,
        cost_model=cost_model,
        pt2_denominator_floor=pt2_denominator_floor,
        min_abs_diagonal=min_abs_diagonal,
    )
    max_pairs = len(candidates)
    if acquired_pair_counts is None:
        default = {0, max_pairs}
        for fraction in (0.05, 0.10, 0.20, 0.40, 0.60, 0.80):
            default.add(int(round(max_pairs * fraction)))
        counts = tuple(sorted(count for count in default if 0 <= count <= max_pairs))
    else:
        counts = tuple(sorted(set(max(0, min(max_pairs, int(count))) for count in acquired_pair_counts)))

    points: list[GFAdaptiveConvergencePoint] = []
    for count in counts:
        trial = scaled.effective_force_constants.copy()
        for candidate in candidates[:count]:
            trial[candidate.i, candidate.j] = f_hl[candidate.i, candidate.j]
            trial[candidate.j, candidate.i] = f_hl[candidate.i, candidate.j]
        validation = validate_adaptive_refinement(trial, g_mat, f_hl, scale_to_cm=scale_to_cm)
        points.append(
            GFAdaptiveConvergencePoint(
                acquired_pairs=count,
                acquired_fraction=float(count / max_pairs) if max_pairs else 1.0,
                frequency_rms_cm=validation.frequency_metrics.rms_delta_cm,
                frequency_max_cm=validation.frequency_metrics.max_abs_delta_cm,
                mean_mode_overlap=validation.mode_overlap.mean_assigned_overlap,
                min_mode_overlap=validation.mode_overlap.min_assigned_overlap,
                hessian_rms_error=validation.hessian_rms_error,
                harmonic_energy_delta_rms=validation.harmonic_energy_delta_rms,
            )
        )
    return GFAdaptiveConvergenceScan(
        points=tuple(points),
        candidates=candidates,
        geometric_scaling=geometric_off_diagonal_diagnostics(
            f_ll,
            f_hl,
            high_diag,
            min_abs_diagonal=min_abs_diagonal,
        ),
    )


def iterative_diagonal_rescaling_validation(
    low_level_force_constants: np.ndarray,
    g_matrix: np.ndarray,
    high_level_force_constants: np.ndarray,
    *,
    off_diagonal_scaling: str = "geometric",
    gic_irreps: tuple[str, ...] = (),
    scale_to_cm: bool = True,
    min_abs_diagonal: float = 1.0e-14,
) -> GFIterativeDiagonalRescalingResult:
    """Validate a two-pass diagonal L1 reconstruction against a full L1 Hessian.

    The first pass is the SONIC-coordinate diagonal replacement with the
    requested residual off-diagonal rule.  The first-pass Hessian is then
    diagonalized, the complete L1 Hessian is projected only for its diagonal in
    that updated eigenvector basis, and a second diagonal-only force field is
    formed.  The full L1 matrix is used here only as a diagnostic reference for
    the diagonal projections and convergence metrics.
    """
    f_ll = _validate_square_symmetric(low_level_force_constants, "Low-level force constants")
    f_hl = _validate_square_symmetric(high_level_force_constants, "High-level force constants")
    g_mat = _validate_square_symmetric(g_matrix, "G matrix")
    if f_ll.shape != f_hl.shape or f_ll.shape != g_mat.shape:
        raise ValueError("Low-level, high-level and G matrices must have the same shape")
    f_ll, _ = _zero_forbidden_irrep_couplings(f_ll, gic_irreps)
    f_hl, _ = _zero_forbidden_irrep_couplings(f_hl, gic_irreps)
    g_mat, _ = _zero_forbidden_irrep_couplings(g_mat, gic_irreps)

    first = diagonal_high_level_scaling(
        f_ll,
        np.diag(f_hl),
        off_diagonal_scaling=off_diagonal_scaling,
        gic_irreps=gic_irreps,
        min_abs_diagonal=min_abs_diagonal,
    ).effective_force_constants
    stage_l0 = _iterative_stage("L0", "SONIC coordinates", f_ll, g_mat, f_hl, scale_to_cm=scale_to_cm)
    stage_first = _iterative_stage(
        "L1 diagonal pass 1",
        "SONIC coordinates",
        first,
        g_mat,
        f_hl,
        scale_to_cm=scale_to_cm,
    )

    g_half, g_half_inv = _g_metric_half_and_inverse(g_mat)
    first_sym = g_half @ first @ g_half
    high_sym = g_half @ f_hl @ g_half
    _eval, updated_basis = np.linalg.eigh(0.5 * (first_sym + first_sym.T))
    high_in_updated_basis = updated_basis.T @ high_sym @ updated_basis
    second_mode = np.diag(np.diag(high_in_updated_basis))
    second_sym = updated_basis @ second_mode @ updated_basis.T
    second = g_half_inv @ second_sym @ g_half_inv
    second = 0.5 * (second + second.T)
    stage_second = _iterative_stage(
        "L1 diagonal pass 2",
        "first-pass eigenvectors",
        second_mode,
        np.eye(second_mode.shape[0]),
        high_in_updated_basis,
        scale_to_cm=scale_to_cm,
    )

    return GFIterativeDiagonalRescalingResult(
        stages=(stage_l0, stage_first, stage_second),
        first_pass_force_constants=first,
        second_pass_force_constants=second,
        updated_mode_basis=updated_basis,
    )


def adaptive_mode_diagonal_rescaling_validation(
    low_level_force_constants: np.ndarray,
    g_matrix: np.ndarray,
    high_level_force_constants: np.ndarray,
    *,
    mode_overlap_threshold: float = 0.999,
    off_diagonal_scaling: str = "geometric",
    gic_irreps: tuple[str, ...] = (),
    scale_to_cm: bool = True,
    min_abs_diagonal: float = 1.0e-14,
) -> GFAdaptiveModeDiagonalRescalingResult:
    """Validate an overlap-triggered second diagonal L1 pass.

    The first pass is the usual SONIC-coordinate diagonal replacement.  The
    first-pass Hessian is then diagonalized and each updated mode is compared
    with its closest L0 mode in the same metric-orthonormal representation.  A
    second L1 diagonal curvature is requested only for updated modes whose best
    overlap with the L0 basis is below ``mode_overlap_threshold``; all other
    modes retain their first-pass curvatures.
    """
    f_ll = _validate_square_symmetric(low_level_force_constants, "Low-level force constants")
    f_hl = _validate_square_symmetric(high_level_force_constants, "High-level force constants")
    g_mat = _validate_square_symmetric(g_matrix, "G matrix")
    if f_ll.shape != f_hl.shape or f_ll.shape != g_mat.shape:
        raise ValueError("Low-level, high-level and G matrices must have the same shape")
    f_ll, _ = _zero_forbidden_irrep_couplings(f_ll, gic_irreps)
    f_hl, _ = _zero_forbidden_irrep_couplings(f_hl, gic_irreps)
    g_mat, _ = _zero_forbidden_irrep_couplings(g_mat, gic_irreps)
    threshold = float(mode_overlap_threshold)
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("Mode-overlap threshold must be between 0 and 1")

    first = diagonal_high_level_scaling(
        f_ll,
        np.diag(f_hl),
        off_diagonal_scaling=off_diagonal_scaling,
        gic_irreps=gic_irreps,
        min_abs_diagonal=min_abs_diagonal,
    ).effective_force_constants
    stage_l0 = _iterative_stage("L0", "SONIC coordinates", f_ll, g_mat, f_hl, scale_to_cm=scale_to_cm)
    stage_first = _iterative_stage(
        "L1 diagonal pass 1",
        "SONIC coordinates",
        first,
        g_mat,
        f_hl,
        scale_to_cm=scale_to_cm,
    )

    g_half, g_half_inv = _g_metric_half_and_inverse(g_mat)
    low_sym = _symmetric_part(g_half @ f_ll @ g_half)
    first_sym = _symmetric_part(g_half @ first @ g_half)
    high_sym = _symmetric_part(g_half @ f_hl @ g_half)
    _low_eval, low_basis = np.linalg.eigh(low_sym)
    _first_eval, updated_basis = np.linalg.eigh(first_sym)
    best_overlaps = np.max(np.abs(low_basis.T @ updated_basis), axis=0)
    selected = tuple(int(index) for index in np.flatnonzero(best_overlaps < threshold))

    first_in_updated_basis = updated_basis.T @ first_sym @ updated_basis
    high_in_updated_basis = updated_basis.T @ high_sym @ updated_basis
    adaptive_diagonal = np.diag(first_in_updated_basis).copy()
    if selected:
        selected_array = np.asarray(selected, dtype=int)
        adaptive_diagonal[selected_array] = np.diag(high_in_updated_basis)[selected_array]
    adaptive_second_mode = np.diag(adaptive_diagonal)
    adaptive_second = g_half_inv @ updated_basis @ adaptive_second_mode @ updated_basis.T @ g_half_inv
    adaptive_second = _symmetric_part(adaptive_second)

    stage_adaptive = _iterative_stage(
        "adaptive L1 diagonal pass 2",
        f"first-pass eigenvectors, overlap < {threshold:.3f}",
        adaptive_second_mode,
        np.eye(adaptive_second_mode.shape[0]),
        high_in_updated_basis,
        scale_to_cm=scale_to_cm,
    )
    return GFAdaptiveModeDiagonalRescalingResult(
        stages=(stage_l0, stage_first, stage_adaptive),
        first_pass_force_constants=first,
        adaptive_second_pass_force_constants=adaptive_second,
        updated_mode_basis=updated_basis,
        selected_mode_indices=selected,
        mode_overlap_threshold=threshold,
        first_pass_mode_overlaps=best_overlaps,
    )


def adaptive_second_pass_validation(
    low_level_force_constants: np.ndarray,
    g_matrix: np.ndarray,
    high_level_force_constants: np.ndarray,
    *,
    ranking_strategy: str = "hybrid",
    batch_size: int = 3,
    patience: int = 2,
    rms_change_threshold_cm: float = 0.5,
    max_change_threshold_cm: float = 2.0,
    off_diagonal_scaling: str = "geometric",
    gic_irreps: tuple[str, ...] = (),
    scale_to_cm: bool = True,
    min_abs_diagonal: float = 1.0e-14,
) -> GFAdaptiveSecondPassResult:
    """Validate adaptive acquisition only after a complete SONIC first pass.

    The first pass acquires every high-level SONIC diagonal and reconstructs a
    coupled field.  Its eigenvectors define the candidate directions for the
    second pass.  Candidates can be ranked by their rotation from the low-level
    modes, by the spread of their predicted frequencies across admissible
    residual-coupling rules, or by the normalized maximum of both diagnostics.
    No unseen second-pass high-level curvature enters that ranking.

    High-level directional curvatures are then revealed in batches.  Stopping
    requires ``patience`` stable batches and a sub-threshold predicted response
    from all remaining modes.  A newly observed curvature-order crossing
    promotes every implicated unacquired same-symmetry mode into the next
    batch.  The complete high-level Hessian is an acquisition oracle and
    retrospective reference only.
    """

    f_ll = _validate_square_symmetric(low_level_force_constants, "Low-level force constants")
    f_hl = _validate_square_symmetric(high_level_force_constants, "High-level force constants")
    g_mat = _validate_square_symmetric(g_matrix, "G matrix")
    if f_ll.shape != f_hl.shape or f_ll.shape != g_mat.shape:
        raise ValueError("Low-level, high-level and G matrices must have the same shape")
    irreps = tuple(str(value).strip() for value in gic_irreps)
    f_ll, _ = _zero_forbidden_irrep_couplings(f_ll, irreps)
    f_hl, _ = _zero_forbidden_irrep_couplings(f_hl, irreps)
    g_mat, _ = _zero_forbidden_irrep_couplings(g_mat, irreps)

    strategy = str(ranking_strategy).strip().lower()
    if strategy not in {"mode_rotation", "off_diagonal_model_spread", "hybrid"}:
        raise ValueError(
            "ranking_strategy must be mode_rotation, off_diagonal_model_spread, or hybrid"
        )
    batch = int(batch_size)
    required_stable = int(patience)
    rms_threshold = float(rms_change_threshold_cm)
    max_threshold = float(max_change_threshold_cm)
    if batch < 1:
        raise ValueError("batch_size must be positive")
    if required_stable < 1:
        raise ValueError("patience must be positive")
    if not np.isfinite(rms_threshold) or rms_threshold < 0.0:
        raise ValueError("rms_change_threshold_cm must be non-negative and finite")
    if not np.isfinite(max_threshold) or max_threshold < 0.0:
        raise ValueError("max_change_threshold_cm must be non-negative and finite")

    first = diagonal_high_level_scaling(
        f_ll,
        np.diag(f_hl),
        off_diagonal_scaling=off_diagonal_scaling,
        gic_irreps=irreps,
        min_abs_diagonal=min_abs_diagonal,
    ).effective_force_constants
    stage_l0 = _iterative_stage("L0", "SONIC coordinates", f_ll, g_mat, f_hl, scale_to_cm=scale_to_cm)
    stage_first = _iterative_stage(
        "L1 diagonal pass 1",
        "SONIC coordinates",
        first,
        g_mat,
        f_hl,
        scale_to_cm=scale_to_cm,
    )

    g_half, g_half_inv = _g_metric_half_and_inverse(g_mat)
    low_sym = _symmetric_part(g_half @ f_ll @ g_half)
    first_sym = _symmetric_part(g_half @ first @ g_half)
    high_sym = _symmetric_part(g_half @ f_hl @ g_half)
    _low_values, low_basis = _symmetry_preserving_eigh(low_sym, irreps)
    first_values, updated_basis = _symmetry_preserving_eigh(first_sym, irreps)
    best_overlaps = np.max(np.abs(low_basis.T @ updated_basis), axis=0)
    rotation_scores = 1.0 - best_overlaps

    residual_rules = ("none", "harmonic", "geometric", "arithmetic", "rms")
    ensemble_frequencies: list[np.ndarray] = []
    for residual_rule in residual_rules:
        variant = diagonal_high_level_scaling(
            f_ll,
            np.diag(f_hl),
            off_diagonal_scaling=residual_rule,
            gic_irreps=irreps,
            min_abs_diagonal=min_abs_diagonal,
        ).effective_force_constants
        variant_sym = _symmetric_part(g_half @ variant @ g_half)
        variant_diagonal = np.diag(updated_basis.T @ variant_sym @ updated_basis)
        ensemble_frequencies.append(
            np.sign(variant_diagonal)
            * np.sqrt(np.abs(variant_diagonal))
            * (HESSIAN_EIGENVALUE_TO_CM if scale_to_cm else 1.0)
        )
    ensemble = np.vstack(ensemble_frequencies)
    reference_ensemble = ensemble[
        residual_rules.index(str(off_diagonal_scaling).strip().lower())
    ]
    model_spread = np.max(np.abs(ensemble - reference_ensemble[None, :]), axis=0)
    rotation_scale = max(float(np.max(rotation_scores)), np.finfo(float).eps)
    spread_scale = max(float(np.max(model_spread)), np.finfo(float).eps)
    if strategy == "mode_rotation":
        ranking_scores = rotation_scores
    elif strategy == "off_diagonal_model_spread":
        ranking_scores = model_spread
    else:
        ranking_scores = np.maximum(
            rotation_scores / rotation_scale,
            model_spread / spread_scale,
        )
    initial_order = [int(index) for index in np.argsort(-ranking_scores, kind="stable")]

    high_in_updated_basis = updated_basis.T @ high_sym @ updated_basis
    high_diagonal = np.diag(high_in_updated_basis).copy()
    current_diagonal = first_values.copy()
    current_frequencies = (
        np.sign(current_diagonal)
        * np.sqrt(np.abs(current_diagonal))
        * (HESSIAN_EIGENVALUE_TO_CM if scale_to_cm else 1.0)
    )
    complete_frequencies = (
        np.sign(high_diagonal)
        * np.sqrt(np.abs(high_diagonal))
        * (HESSIAN_EIGENVALUE_TO_CM if scale_to_cm else 1.0)
    )
    mode_irreps = tuple(
        irreps[int(np.argmax(np.abs(updated_basis[:, mode])))] if irreps else ""
        for mode in range(updated_basis.shape[1])
    )
    inversion_count = sum(
        1
        for i in range(len(first_values))
        for j in range(i + 1, len(first_values))
        if mode_irreps[i] == mode_irreps[j]
        and (first_values[i] - first_values[j]) * (high_diagonal[i] - high_diagonal[j]) < 0.0
    )

    queue = list(initial_order)
    selected_all: list[int] = []
    cycles: list[GFAdaptiveSecondPassCycle] = []
    stable_cycles = 0
    stop_reason = "ALL_ACQUIRED"
    while queue:
        selected = tuple(queue.pop(0) for _ in range(min(batch, len(queue))))
        previous_frequencies = current_frequencies.copy()
        for index in selected:
            current_diagonal[index] = high_diagonal[index]
            selected_all.append(index)
        current_frequencies = (
            np.sign(current_diagonal)
            * np.sqrt(np.abs(current_diagonal))
            * (HESSIAN_EIGENVALUE_TO_CM if scale_to_cm else 1.0)
        )
        observed_delta = current_frequencies - previous_frequencies
        observed_rms = float(np.sqrt(np.mean(observed_delta * observed_delta)))
        observed_max = float(np.max(np.abs(observed_delta)))

        order_risk: set[int] = set()
        for acquired_index in selected:
            for remaining_index in queue:
                if mode_irreps[acquired_index] != mode_irreps[remaining_index]:
                    continue
                if (
                    (first_values[acquired_index] - first_values[remaining_index])
                    * (current_diagonal[acquired_index] - current_diagonal[remaining_index])
                    < 0.0
                ):
                    order_risk.add(remaining_index)
        if order_risk:
            queue = (
                sorted(order_risk, key=lambda index: (-ranking_scores[index], index))
                + [index for index in queue if index not in order_risk]
            )
            stable_cycles = 0

        remaining_array = np.asarray(queue, dtype=int)
        if len(remaining_array):
            predicted_rms = float(
                np.sqrt(np.sum(model_spread[remaining_array] ** 2) / len(first_values))
            )
            predicted_max = float(np.max(model_spread[remaining_array]))
        else:
            predicted_rms = predicted_max = 0.0
        residual = current_frequencies - complete_frequencies
        residual_rms = float(np.sqrt(np.mean(residual * residual)))
        residual_max = float(np.max(np.abs(residual)))
        observed_stable = observed_rms <= rms_threshold and observed_max <= max_threshold
        stable_cycles = stable_cycles + 1 if observed_stable and not order_risk else 0
        cycle_stop = ""
        if (
            queue
            and stable_cycles >= required_stable
            and predicted_rms <= rms_threshold
            and predicted_max <= max_threshold
        ):
            stop_reason = "OBSERVED_AND_PREDICTED_CONVERGENCE"
            cycle_stop = stop_reason
        cycles.append(
            GFAdaptiveSecondPassCycle(
                cycle=len(cycles) + 1,
                acquired_mode_indices=selected,
                total_acquired=len(selected_all),
                observed_rms_change_cm=observed_rms,
                observed_max_change_cm=observed_max,
                predicted_remaining_rms_cm=predicted_rms,
                predicted_remaining_max_cm=predicted_max,
                reference_residual_rms_cm=residual_rms,
                reference_residual_max_cm=residual_max,
                order_risk_mode_indices=tuple(sorted(order_risk)),
                stop_reason=cycle_stop,
            )
        )
        if cycle_stop:
            break

    adaptive_mode = np.diag(current_diagonal)
    adaptive_second = (
        g_half_inv @ updated_basis @ adaptive_mode @ updated_basis.T @ g_half_inv
    )
    adaptive_second = _symmetric_part(adaptive_second)
    stage_adaptive = _iterative_stage(
        "adaptive L1 diagonal pass 2",
        f"first-pass eigenvectors, {strategy}",
        adaptive_mode,
        np.eye(adaptive_mode.shape[0]),
        high_in_updated_basis,
        scale_to_cm=scale_to_cm,
    )
    return GFAdaptiveSecondPassResult(
        stages=(stage_l0, stage_first, stage_adaptive),
        first_pass_force_constants=first,
        adaptive_second_pass_force_constants=adaptive_second,
        updated_mode_basis=updated_basis,
        selected_mode_indices=tuple(selected_all),
        first_pass_mode_overlaps=best_overlaps,
        off_diagonal_model_spread_cm=model_spread,
        ranking_scores=ranking_scores,
        ranking_strategy=strategy,
        cycles=tuple(cycles),
        stop_reason=stop_reason,
        rms_change_threshold_cm=rms_threshold,
        max_change_threshold_cm=max_threshold,
        batch_size=batch,
        patience=required_stable,
        same_irrep_curvature_order_inversions=inversion_count,
    )


def mode_first_sonic_coupling_selection(
    low_level_force_constants: np.ndarray,
    g_matrix: np.ndarray,
    high_level_mode_diagonal: np.ndarray,
    *,
    coordinate_metadata: tuple[GFAdaptiveCoordinateMetadata, ...] = (),
    off_diagonal_scaling: str = "geometric",
    ranking_strategy: str = "discrepancy",
    pt2_denominator_floor: float = 1.0e-8,
    curvature_order_tolerance: float = 1.0e-10,
    allow_curvature_order_inversions: bool = False,
    min_abs_diagonal: float = 1.0e-14,
) -> GFModeFirstSonicSelectionResult:
    """Transform an L1 diagonal in the L0 mode basis back to SONIC and rank pairs.

    The transformed mode-diagonal field supplies implied SONIC diagonals and an
    off-diagonal diagnostic.  Its symmetry-allowed off-diagonals are compared
    directly with the unscaled L0 couplings.  The separately returned scaled
    L0 field implements the no-acquisition inverse reconstruction.
    By default candidates are ordered by the absolute discrepancy from the
    unscaled L0 coupling.  ``ranking_strategy="pt2"`` instead divides its
    square by the implied diagonal separation.  Exact L1 pairs must still be
    acquired externally; this routine only supplies the auditable ordering.
    L1 curvatures remain
    attached to the L0 eigenvectors on which they were acquired: they are
    never sorted by value.  A reversal of curvature order relative to L0 is
    rejected by default because it can indicate a mode-assignment inversion;
    a verified physical crossing requires explicit opt-in.
    """
    f_ll = _validate_square_symmetric(low_level_force_constants, "Low-level force constants")
    g_mat = _validate_square_symmetric(g_matrix, "G matrix")
    if f_ll.shape != g_mat.shape:
        raise ValueError("Low-level force constants and G matrix must have the same shape")
    n_coord = f_ll.shape[0]
    metadata = _complete_metadata(coordinate_metadata, n_coord)
    irreps = tuple(item.irrep for item in metadata)
    f_ll, _ = _zero_forbidden_irrep_couplings(f_ll, irreps)
    g_mat, _ = _zero_forbidden_irrep_couplings(g_mat, irreps)
    mode_diagonal = np.asarray(high_level_mode_diagonal, dtype=float).reshape(-1)
    if mode_diagonal.shape != (n_coord,):
        raise ValueError(f"High-level mode diagonal must have length {n_coord}")
    if np.any(~np.isfinite(mode_diagonal)):
        raise ValueError("High-level mode diagonal must be finite")

    order_tolerance = float(curvature_order_tolerance)
    if not np.isfinite(order_tolerance) or order_tolerance < 0.0:
        raise ValueError("curvature_order_tolerance must be a non-negative finite number")
    strategy = str(ranking_strategy).strip().lower()
    if strategy not in {"discrepancy", "pt2"}:
        raise ValueError("ranking_strategy must be discrepancy or pt2")

    g_half, g_half_inv = _g_metric_half_and_inverse(g_mat)
    low_sym = _symmetric_part(g_half @ f_ll @ g_half)
    low_eigenvalues, low_basis = _symmetry_preserving_eigh(low_sym, irreps)

    comparison_scale = max(
        float(np.max(np.abs(low_eigenvalues))),
        float(np.max(np.abs(mode_diagonal))),
        np.finfo(float).eps,
    )
    absolute_order_tolerance = order_tolerance * comparison_scale
    inversions: list[tuple[int, int]] = []
    for i in range(n_coord):
        for j in range(i + 1, n_coord):
            low_difference = float(low_eigenvalues[j] - low_eigenvalues[i])
            high_difference = float(mode_diagonal[j] - mode_diagonal[i])
            if (
                low_difference > absolute_order_tolerance
                and high_difference < -absolute_order_tolerance
            ):
                inversions.append((i, j))
    if inversions and not allow_curvature_order_inversions:
        preview = ", ".join(f"({i}, {j})" for i, j in inversions[:8])
        suffix = " ..." if len(inversions) > 8 else ""
        raise ValueError(
            "L1 mode curvatures reverse the L0 curvature order for mode pairs "
            f"{preview}{suffix}; preserve mode identity and set "
            "allow_curvature_order_inversions=True only for verified physical crossings"
        )
    projected = g_half_inv @ low_basis @ np.diag(mode_diagonal) @ low_basis.T @ g_half_inv
    projected = _symmetric_part(projected)
    projected, _ = _zero_forbidden_irrep_couplings(projected, irreps)
    implied_diagonal = np.diag(projected)
    scaled = diagonal_high_level_scaling(
        f_ll,
        implied_diagonal,
        off_diagonal_scaling=off_diagonal_scaling,
        gic_irreps=irreps,
        min_abs_diagonal=min_abs_diagonal,
    )
    relative_floor = float(pt2_denominator_floor)
    if not np.isfinite(relative_floor) or relative_floor <= 0.0:
        raise ValueError("pt2_denominator_floor must be a positive finite number")
    diagonal_scale = max(float(np.max(np.abs(implied_diagonal))), np.finfo(float).eps)
    denominator_floor = max(relative_floor * diagonal_scale, np.finfo(float).eps)
    candidates: list[GFAdaptiveCandidate] = []
    for i in range(n_coord):
        for j in range(i + 1, n_coord):
            if not _is_admissible_pair(metadata[i], metadata[j], i, j):
                continue
            discrepancy = abs(float(projected[i, j] - f_ll[i, j]))
            separation = abs(float(implied_diagonal[i] - implied_diagonal[j]))
            denominator = max(separation, denominator_floor)
            score = (
                discrepancy
                if strategy == "discrepancy"
                else discrepancy * discrepancy / denominator
            )
            candidates.append(
                GFAdaptiveCandidate(
                    i=i,
                    j=j,
                    benefit=float(score),
                    cost=1.0,
                    benefit_cost=float(score),
                    score=float(score),
                    reason=(
                        f"ranking_strategy={strategy}; "
                        f"mode_sonic_discrepancy={discrepancy:.6g}; "
                        f"implied_diagonal_separation={separation:.6g}; "
                        f"denominator={denominator:.6g}"
                    ),
                )
            )
    candidates.sort(key=lambda item: (-item.score, item.i, item.j))
    return GFModeFirstSonicSelectionResult(
        high_level_mode_diagonal=mode_diagonal.copy(),
        low_level_mode_curvatures=low_eigenvalues,
        l0_mode_basis=low_basis,
        mode_projected_force_constants=projected,
        symmetry_projected_l0_force_constants=f_ll,
        symmetry_projected_g_matrix=g_mat,
        scaled_l0_force_constants=scaled.effective_force_constants,
        candidates=tuple(candidates),
        curvature_order_inversions=tuple(inversions),
        ranking_strategy=strategy,
        off_diagonal_scaling=scaled.off_diagonal_scaling,
    )


def concordant_mode_projection_validation(
    low_level_force_constants: np.ndarray,
    g_matrix: np.ndarray,
    high_level_force_constants: np.ndarray,
    *,
    mode_start: int = 0,
    gic_irreps: tuple[str, ...] = (),
    scale_to_cm: bool = True,
) -> GFConcordantModeProjectionResult:
    """Validate diagonal L1 curvatures projected on L0 normal modes.

    This is the generic concordant-mode analogue of the SONIC diagonal
    replacement.  The low-level GF problem defines the mode basis; the
    high-level force field is projected into that basis, only the projected
    diagonal is retained, and the resulting frequencies are compared with the
    complete high-level GF reference.  ``mode_start`` can be used to discard
    translational/rotational zero modes in Cartesian mass-weighted applications.
    """
    f_ll = _validate_square_symmetric(low_level_force_constants, "Low-level force constants")
    f_hl = _validate_square_symmetric(high_level_force_constants, "High-level force constants")
    g_mat = _validate_square_symmetric(g_matrix, "G matrix")
    if f_ll.shape != f_hl.shape or f_ll.shape != g_mat.shape:
        raise ValueError("Low-level, high-level and G matrices must have the same shape")
    f_ll, _ = _zero_forbidden_irrep_couplings(f_ll, gic_irreps)
    f_hl, _ = _zero_forbidden_irrep_couplings(f_hl, gic_irreps)
    g_mat, _ = _zero_forbidden_irrep_couplings(g_mat, gic_irreps)
    start = int(mode_start)
    n_coord = f_ll.shape[0]
    if start < 0 or start >= n_coord:
        raise ValueError(f"Mode start must be between 0 and {n_coord - 1}")

    g_half, g_half_inv = _g_metric_half_and_inverse(g_mat)
    low_sym = _symmetric_part(g_half @ f_ll @ g_half)
    high_sym = _symmetric_part(g_half @ f_hl @ g_half)
    low_modes = solve_wilson_gf(f_ll, g_mat, scale_to_cm=scale_to_cm)
    high_modes = solve_wilson_gf(f_hl, g_mat, scale_to_cm=scale_to_cm)
    _low_eval, low_basis = _symmetry_preserving_eigh(low_sym, gic_irreps)
    active_low_basis = low_basis[:, start:]
    high_in_low_basis = active_low_basis.T @ high_sym @ active_low_basis
    projected_mode_force_constants = np.diag(np.diag(high_in_low_basis))
    projected_force_constants = (
        g_half_inv @ active_low_basis @ projected_mode_force_constants @ active_low_basis.T @ g_half_inv
    )
    projected_force_constants = _symmetric_part(projected_force_constants)
    projected_modes = solve_wilson_gf(
        projected_mode_force_constants,
        np.eye(projected_mode_force_constants.shape[0]),
        scale_to_cm=scale_to_cm,
    )

    candidate_freq = projected_modes.frequencies_cm
    reference_freq = high_modes.frequencies_cm[start:]
    frequency_metrics = frequency_error_metrics(candidate_freq, reference_freq)
    overlap = mode_overlap_diagnostics(
        low_modes.normal_modes[:, start:],
        high_modes.normal_modes[:, start:],
    )
    return GFConcordantModeProjectionResult(
        frequency_metrics=frequency_metrics,
        mode_overlap=overlap,
        low_level_frequencies_cm=low_modes.frequencies_cm[start:],
        projected_frequencies_cm=candidate_freq,
        reference_frequencies_cm=reference_freq,
        projected_force_constants=projected_force_constants,
        projected_diagonal=np.diag(high_in_low_basis).copy(),
        l0_mode_basis=low_basis,
        mode_start=start,
    )


def frequency_error_metrics(
    candidate_cm: np.ndarray,
    reference_cm: np.ndarray,
) -> GFFrequencyErrorMetrics:
    """Return RMS, maximum and mean absolute frequency errors in cm-1."""
    candidate = np.asarray(candidate_cm, dtype=float).reshape(-1)
    reference = np.asarray(reference_cm, dtype=float).reshape(-1)
    if candidate.shape != reference.shape:
        raise ValueError("Frequency arrays must have the same length")
    if candidate.size == 0:
        raise ValueError("Frequency arrays cannot be empty")
    deltas = candidate - reference
    return GFFrequencyErrorMetrics(
        rms_delta_cm=float(np.sqrt(np.mean(deltas * deltas))),
        max_abs_delta_cm=float(np.max(np.abs(deltas))),
        mean_abs_delta_cm=float(np.mean(np.abs(deltas))),
        deltas_cm=deltas,
    )


def mode_overlap_diagnostics(
    candidate_modes: np.ndarray,
    reference_modes: np.ndarray,
) -> GFModeOverlapResult:
    """Compute absolute Duschinsky-like overlaps and a deterministic matching.

    The function assumes both mode matrices are expressed in the same metric or
    have already been normalized by the caller.  It deliberately does not rotate
    modes inside degenerate subspaces; the returned overlap matrix is the raw
    diagnostic used to decide whether a normal-mode transfer assumption is safe.
    """
    candidate = np.asarray(candidate_modes, dtype=float)
    reference = np.asarray(reference_modes, dtype=float)
    if candidate.ndim != 2 or reference.ndim != 2 or candidate.shape != reference.shape:
        raise ValueError("Mode matrices must be two-dimensional arrays with the same shape")
    overlap = np.abs(candidate.T @ reference)
    assignment = _greedy_overlap_assignment(overlap)
    assigned = [value for _left, _right, value in assignment]
    return GFModeOverlapResult(
        overlap_matrix=overlap,
        assignment=tuple(assignment),
        mean_assigned_overlap=float(np.mean(assigned)) if assigned else 0.0,
        min_assigned_overlap=float(np.min(assigned)) if assigned else 0.0,
    )


def _validate_square_symmetric(matrix: np.ndarray, label: str) -> np.ndarray:
    value = np.asarray(matrix, dtype=float)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError(f"{label} must be a square matrix")
    if not np.allclose(value, value.T):
        raise ValueError(f"{label} must be symmetric")
    return value


def _symmetric_part(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def _iterative_stage(
    label: str,
    basis: str,
    force_constants: np.ndarray,
    g_matrix: np.ndarray,
    reference_force_constants: np.ndarray,
    *,
    scale_to_cm: bool,
) -> GFIterativeDiagonalStage:
    validation = validate_adaptive_refinement(
        force_constants,
        g_matrix,
        reference_force_constants,
        scale_to_cm=scale_to_cm,
    )
    return GFIterativeDiagonalStage(
        label=label,
        basis=basis,
        frequency_rms_cm=validation.frequency_metrics.rms_delta_cm,
        frequency_max_cm=validation.frequency_metrics.max_abs_delta_cm,
        mean_mode_overlap=validation.mode_overlap.mean_assigned_overlap,
        min_mode_overlap=validation.mode_overlap.min_assigned_overlap,
        hessian_rms_error=validation.hessian_rms_error,
    )


def _g_metric_half_and_inverse(g_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g_eval, g_vec = np.linalg.eigh(0.5 * (g_matrix + g_matrix.T))
    if np.any(g_eval <= 0.0):
        raise ValueError("G matrix must be positive definite")
    g_half = (g_vec * np.sqrt(g_eval)) @ g_vec.T
    g_half_inv = (g_vec * (1.0 / np.sqrt(g_eval))) @ g_vec.T
    return g_half, g_half_inv


def _optional_vector(vector: np.ndarray | None, n_coord: int, label: str) -> np.ndarray | None:
    if vector is None:
        return None
    value = np.asarray(vector, dtype=float).reshape(-1)
    if value.shape != (n_coord,):
        raise ValueError(f"{label} must have length {n_coord}")
    return value


def _optional_b_matrix(matrix: np.ndarray | None, n_coord: int) -> np.ndarray | None:
    if matrix is None:
        return None
    value = np.asarray(matrix, dtype=float)
    if value.ndim != 2 or value.shape[0] != n_coord:
        raise ValueError(f"B matrix must have {n_coord} rows")
    return value


def _optional_mode_matrix(matrix: np.ndarray | None, n_coord: int) -> np.ndarray | None:
    if matrix is None:
        return None
    value = np.asarray(matrix, dtype=float)
    if value.ndim != 2 or value.shape[0] != n_coord:
        raise ValueError(f"L0 normal modes must have {n_coord} coordinate rows")
    return value


def _primitive_scaling_class(
    primitive_label: str,
    atomic_numbers: tuple[int, ...],
    fallback_family: str,
    gic_name: str,
) -> str:
    atoms = _primitive_atom_indices(primitive_label)
    upper = primitive_label.strip().upper()
    text = f"{gic_name} {fallback_family}".lower()
    if upper.startswith("R(") and len(atoms) >= 2:
        elements = [_atomic_number(atomic_numbers, atom) for atom in atoms[:2]]
        if 1 in elements:
            heavy = elements[0] if elements[1] == 1 else elements[1]
            if heavy == 6:
                return "CH_stretch"
            if heavy == 7:
                return "NH_stretch"
            if heavy == 8:
                return "OH_stretch"
            return "XH_stretch"
        left, right = sorted(_element_symbol(element) for element in elements)
        if left != "X" and right != "X":
            return f"{left}{right}_stretch"
        return "XY_stretch"
    if upper.startswith("A("):
        elements = [_atomic_number(atomic_numbers, atom) for atom in atoms[:3]]
        if elements and 1 in elements:
            return "XH_bend"
        return "heavy_bend"
    if upper.startswith("D("):
        return "torsion"
    if upper.startswith("U(") or "oop" in text:
        return "out_of_plane"
    if any(token in text for token in ("rpck", "qpck", "phip", "pck")):
        return "ring_puckering"
    if "tors" in text or "dih" in text:
        return "torsion"
    if "bend" in text or "angle" in text or "rock" in text or "rdef" in text:
        return "bend"
    if "stretch" in text or "stre" in text:
        return "XY_stretch"
    return fallback_family or "unclassified"


_ELEMENT_SYMBOLS = {
    1: "H",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    35: "Br",
    53: "I",
}


def _element_symbol(atomic_number: int) -> str:
    return _ELEMENT_SYMBOLS.get(int(atomic_number), "X")


def _primitive_atom_indices(label: str) -> tuple[int, ...]:
    match = re.search(r"\(([^)]*)\)", label)
    if not match:
        return ()
    atoms: list[int] = []
    for item in re.split(r"[,;\s]+", match.group(1).strip()):
        if not item:
            continue
        try:
            atoms.append(int(item))
        except ValueError:
            continue
    return tuple(atoms)


def _atomic_number(atomic_numbers: tuple[int, ...], atom_index: int) -> int:
    if not atomic_numbers or atom_index < 1 or atom_index > len(atomic_numbers):
        return 0
    return int(atomic_numbers[atom_index - 1])


def _complete_metadata(
    metadata: tuple[GFAdaptiveCoordinateMetadata, ...],
    n_coord: int,
) -> tuple[GFAdaptiveCoordinateMetadata, ...]:
    if not metadata:
        return tuple(GFAdaptiveCoordinateMetadata(identifier=f"q{i + 1}") for i in range(n_coord))
    if len(metadata) != n_coord:
        raise ValueError(f"Coordinate metadata must contain {n_coord} entries")
    return tuple(metadata)


def _assert_coordinate_conditioning(g_matrix: np.ndarray, threshold: float) -> None:
    eig = np.linalg.eigvalsh((g_matrix + g_matrix.T) * 0.5)
    if np.any(eig <= 0.0):
        raise ValueError("SONIC coordinate G matrix is not positive definite")
    condition_ratio = float(np.min(eig) / np.max(eig))
    if condition_ratio < float(threshold):
        raise ValueError(
            "SONIC coordinate conditioning is below threshold "
            f"({condition_ratio:.3e} < {float(threshold):.3e})"
        )


def _is_admissible_pair(
    left: GFAdaptiveCoordinateMetadata,
    right: GFAdaptiveCoordinateMetadata,
    i: int,
    j: int,
) -> bool:
    if left.irrep and right.irrep and left.irrep != right.irrep:
        return False
    if left.fragment_id and right.fragment_id and left.fragment_id != right.fragment_id:
        return False
    left_family = left.family.lower()
    right_family = right.family.lower()
    if "ring" in left_family or "rpck" in left_family or "rdef" in left_family:
        return _same_nonempty(left.ring_id, right.ring_id) > 0.0 or j in left.neighbors or i in right.neighbors
    if "ring" in right_family or "rpck" in right_family or "rdef" in right_family:
        return _same_nonempty(left.ring_id, right.ring_id) > 0.0 or j in left.neighbors or i in right.neighbors
    return True


def _same_nonempty(left: str, right: str) -> float:
    return 1.0 if left and right and left == right else 0.0


def _candidate_cost(
    model: GFAdaptiveCostModel,
    left: GFAdaptiveCoordinateMetadata,
    right: GFAdaptiveCoordinateMetadata,
) -> float:
    cost = float(model.off_diagonal)
    if left.family and right.family and left.family == right.family:
        cost *= float(model.same_family_discount)
    if left.fragment_id and right.fragment_id and left.fragment_id == right.fragment_id:
        cost *= float(model.same_fragment_discount)
    return max(cost, 1.0e-12)


def _topological_proximity(
    metadata: tuple[GFAdaptiveCoordinateMetadata, ...],
    start: int,
    end: int,
) -> float:
    if end in metadata[start].neighbors or start in metadata[end].neighbors:
        return 1.0
    visited = {start}
    frontier = [(start, 0)]
    while frontier:
        current, distance = frontier.pop(0)
        for neighbor in metadata[current].neighbors:
            if neighbor < 0 or neighbor >= len(metadata) or neighbor in visited:
                continue
            if neighbor == end:
                return 1.0 / float(distance + 2)
            visited.add(neighbor)
            frontier.append((neighbor, distance + 1))
    return 0.0


def _b_row_similarity(b_matrix: np.ndarray | None, i: int, j: int) -> float:
    if b_matrix is None:
        return 0.0
    left = b_matrix[i]
    right = b_matrix[j]
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 0.0:
        return 0.0
    return abs(float(np.dot(left, right) / denom))


def _checked_index(index: int, size: int, label: str) -> int:
    value = int(index)
    if value < 0 or value >= size:
        raise ValueError(f"{label.capitalize()} index {value} out of range 0..{size - 1}")
    return value


def _label_index(label: str, labels: tuple[str, ...], kind: str) -> int:
    if not labels:
        raise ValueError(f"Cannot resolve {kind} label without labels")
    matches = [index for index, item in enumerate(labels) if item == label]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"Unknown {kind} label {label!r}")
    raise ValueError(f"Ambiguous {kind} label {label!r}")


def _dominant_primitive_index(column: np.ndarray, source_coordinate: int) -> int:
    values = np.asarray(column, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("Cannot select a primitive from an empty U column")
    index = int(np.argmax(np.abs(values)))
    if abs(float(values[index])) <= 0.0:
        raise ValueError(
            f"Source coordinate {source_coordinate} has no non-zero primitive component"
        )
    return index


def _mode_participation_similarity(modes: np.ndarray, i: int, j: int) -> float:
    left = np.abs(modes[i])
    right = np.abs(modes[j])
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 0.0:
        return 0.0
    return abs(float(np.dot(left, right) / denom))


def _adaptive_objective(
    force_constants: np.ndarray,
    result: GFResult,
    reference_force_constants: np.ndarray | None,
    reference_result: GFResult | None,
    total_cost: float,
    weights: GFAdaptiveObjectiveWeights,
) -> float:
    objective = float(weights.cost) * float(total_cost)
    if reference_result is not None:
        freq = frequency_error_metrics(result.frequencies_cm, reference_result.frequencies_cm)
        overlap = mode_overlap_diagnostics(result.normal_modes, reference_result.normal_modes)
        objective += float(weights.frequency) * freq.rms_delta_cm
        objective += float(weights.mode_overlap) * (1.0 - overlap.mean_assigned_overlap)
    if reference_force_constants is not None and weights.hessian:
        objective += float(weights.hessian) * float(
            np.sqrt(np.mean((force_constants - reference_force_constants) ** 2))
        )
    return float(objective)


def _meets_stopping(
    result: GFResult,
    reference: GFResult,
    criteria: GFAdaptiveStoppingCriteria,
    total_cost: float,
) -> bool:
    freq = frequency_error_metrics(result.frequencies_cm, reference.frequencies_cm)
    overlap = mode_overlap_diagnostics(result.normal_modes, reference.normal_modes)
    if criteria.rms_frequency_cm is not None and freq.rms_delta_cm > criteria.rms_frequency_cm:
        return False
    if criteria.max_frequency_cm is not None and freq.max_abs_delta_cm > criteria.max_frequency_cm:
        return False
    if criteria.min_mode_overlap is not None and overlap.min_assigned_overlap < criteria.min_mode_overlap:
        return False
    return any(
        value is not None
        for value in (
            criteria.rms_frequency_cm,
            criteria.max_frequency_cm,
            criteria.min_mode_overlap,
        )
    )


def _greedy_overlap_assignment(overlap: np.ndarray) -> list[tuple[int, int, float]]:
    n_modes = overlap.shape[0]
    candidates: list[tuple[float, int, int]] = []
    for left in range(n_modes):
        for right in range(n_modes):
            candidates.append((float(overlap[left, right]), left, right))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_left: set[int] = set()
    used_right: set[int] = set()
    assignment: list[tuple[int, int, float]] = []
    for value, left, right in candidates:
        if left in used_left or right in used_right:
            continue
        used_left.add(left)
        used_right.add(right)
        assignment.append((left + 1, right + 1, value))
        if len(assignment) == n_modes:
            break
    assignment.sort(key=lambda item: item[0])
    return assignment
