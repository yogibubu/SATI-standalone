"""Sparse inverse-distance substrate blocks for typed ONIC contracts.

The production construction works through the Cartesian Gram matrix
``B_D.T @ B_D``.  It never materializes the source-space projector
``B_D @ pinv(B_D)``; an explicitly bounded audit helper provides that matrix
only for small scientific fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import re
from typing import Sequence

import numpy as np
from scipy.sparse import csr_matrix

from matrix_chem import MolecularSymmetry
from matrix_numerics import numerical_matrix_rank

from .bmatrix import SparseBMatrix
from .cartesian_blocks import (
    CARTESIAN_BLOCK_COVARIANCE_TOLERANCE,
    SymmetryAdaptedCartesianBasis,
    onic_degeneracy_groups_from_cartesian_basis,
    symmetry_adapted_cartesian_basis,
)
from .onic_blocks import (
    OnicBlockDiagnostics,
    OnicCoordinateBlock,
    OnicMatrixRecord,
    onic_reference_fingerprint,
)
from .symmetry_labels import irrep_name_prefix


INVERSE_DISTANCE_BLOCK_RANK_METHOD = "SPARSE_B_CARTESIAN_GRAM_EIGH"
INVERSE_DISTANCE_BLOCK_GAUGE = "SITE_FRAME_CARTESIAN_DUAL"
INVERSE_DISTANCE_BLOCK_ABSOLUTE_RANK_TOLERANCE = 1.0e-10
INVERSE_DISTANCE_BLOCK_RELATIVE_RANK_TOLERANCE = 1.0e-8
INVERSE_DISTANCE_SINGULARITY_THRESHOLD_ANGSTROM = 1.0e-8
INVERSE_DISTANCE_EXPLICIT_AUDIT_MAX_SOURCES = 512
_PAIR_IDENTIFIER_PATTERN = re.compile(r"\.Pair([0-9]+)_([0-9]+)$")


@dataclass(frozen=True)
class InverseDistanceSourceState:
    pairs_one_based: tuple[tuple[int, int], ...]
    values_inverse_angstrom: np.ndarray
    b_matrix: csr_matrix
    minimum_distance_angstrom: float


@dataclass(frozen=True)
class InverseDistanceBlockEvaluation:
    coordinate_values_angstrom: np.ndarray
    source_values_inverse_angstrom: np.ndarray
    b_matrix: SparseBMatrix
    minimum_distance_angstrom: float
    orientation_guard_status: str


@dataclass(frozen=True)
class InverseDistanceProjectorAudit:
    source_count: int
    rank: int
    eigenvalues: tuple[float, ...]
    symmetry_residual: float
    idempotency_residual: float
    row_space_residual: float | None


def inverse_distance_source_state(
    coordinates_angstrom: np.ndarray | Sequence[Sequence[float]],
    pairs_one_based: Sequence[tuple[int, int]],
    *,
    singularity_threshold_angstrom: float = INVERSE_DISTANCE_SINGULARITY_THRESHOLD_ANGSTROM,
) -> InverseDistanceSourceState:
    """Evaluate all frozen 1/r sources and their six-entry sparse B rows."""

    coordinates, pairs, values, delta, distances, minimum_distance = (
        _inverse_distance_source_arrays(
            coordinates_angstrom,
            pairs_one_based,
            singularity_threshold_angstrom=singularity_threshold_angstrom,
        )
    )
    if not pairs:
        return InverseDistanceSourceState(
            pairs_one_based=(),
            values_inverse_angstrom=np.zeros(0, dtype=float),
            b_matrix=csr_matrix((0, 3 * len(coordinates)), dtype=float),
            minimum_distance_angstrom=float("inf"),
        )
    pair_array = np.asarray(pairs, dtype=int) - 1
    left = pair_array[:, 0]
    right = pair_array[:, 1]
    left_gradients = -delta / distances[:, None] ** 3
    pair_count = len(pairs)
    local_axes = np.arange(3, dtype=int)
    left_columns = 3 * left[:, None] + local_axes[None, :]
    right_columns = 3 * right[:, None] + local_axes[None, :]
    columns = np.hstack((left_columns, right_columns)).reshape(-1)
    data = np.hstack((left_gradients, -left_gradients)).reshape(-1)
    rows = np.repeat(np.arange(pair_count, dtype=int), 6)
    b_matrix = csr_matrix(
        (data, (rows, columns)),
        shape=(pair_count, 3 * len(coordinates)),
        dtype=float,
    )
    b_matrix.sum_duplicates()
    b_matrix.eliminate_zeros()
    b_matrix.sort_indices()
    return InverseDistanceSourceState(
        pairs_one_based=pairs,
        values_inverse_angstrom=values,
        b_matrix=b_matrix,
        minimum_distance_angstrom=minimum_distance,
    )


def _inverse_distance_source_arrays(
    coordinates_angstrom: np.ndarray | Sequence[Sequence[float]],
    pairs_one_based: Sequence[tuple[int, int]],
    *,
    singularity_threshold_angstrom: float,
) -> tuple[
    np.ndarray,
    tuple[tuple[int, int], ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    """Evaluate shared 1/r source arrays without constructing derivative rows."""

    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or not np.all(np.isfinite(coordinates)):
        raise ValueError("inverse-distance coordinates must be a finite natoms-by-3 array")
    pairs = _normalized_pairs(
        pairs_one_based,
        atom_indices=tuple(range(1, len(coordinates) + 1)),
    )
    threshold = _positive_finite(
        singularity_threshold_angstrom,
        "inverse-distance singularity threshold",
    )
    if not pairs:
        return (
            coordinates,
            (),
            np.zeros(0, dtype=float),
            np.empty((0, 3), dtype=float),
            np.zeros(0, dtype=float),
            float("inf"),
        )
    pair_array = np.asarray(pairs, dtype=int) - 1
    delta = coordinates[pair_array[:, 0]] - coordinates[pair_array[:, 1]]
    distances = np.linalg.norm(delta, axis=1)
    minimum_distance = float(np.min(distances))
    if not np.all(np.isfinite(distances)) or minimum_distance <= threshold:
        raise FloatingPointError(
            "inverse-distance source contains a coincident or near-coincident atom pair"
        )
    return coordinates, pairs, 1.0 / distances, delta, distances, minimum_distance


def build_inverse_distance_projector_block(
    atoms: Sequence[str],
    reference_coordinates_angstrom: np.ndarray | Sequence[Sequence[float]],
    *,
    symmetry: MolecularSymmetry,
    atom_indices_one_based: Sequence[int],
    site_anchor_atom_indices_one_based: Sequence[int],
    pairs_one_based: Sequence[tuple[int, int]] | None = None,
    block_identifier: str = "SUB1",
    local_symmetry_provenance: str = "ORACLE_DECLARED_SITE_SUBGROUP",
    protected: bool = True,
    active: bool = True,
    observable: bool = False,
    rank_absolute_tolerance: float = INVERSE_DISTANCE_BLOCK_ABSOLUTE_RANK_TOLERANCE,
    rank_relative_tolerance: float = INVERSE_DISTANCE_BLOCK_RELATIVE_RANK_TOLERANCE,
    singularity_threshold_angstrom: float = INVERSE_DISTANCE_SINGULARITY_THRESHOLD_ANGSTROM,
) -> OnicCoordinateBlock:
    """Build a frozen 1/r source-space block without a dense source projector."""

    symbols = tuple(str(atom) for atom in atoms)
    full_reference = np.asarray(reference_coordinates_angstrom, dtype=float)
    if full_reference.shape != (len(symbols), 3) or not np.all(np.isfinite(full_reference)):
        raise ValueError("inverse-distance reference must be a finite natoms-by-3 array")
    subset = _normalized_atom_subset(atom_indices_one_based, natoms=len(symbols))
    anchors = tuple(int(atom) for atom in site_anchor_atom_indices_one_based)
    if len(anchors) < 3 or len(set(anchors)) != len(anchors) or not set(anchors).issubset(subset):
        raise ValueError("inverse-distance substrate blocks require three unique subset anchors")
    absolute_tolerance = _positive_finite(rank_absolute_tolerance, "absolute rank tolerance")
    relative_tolerance = _positive_finite(rank_relative_tolerance, "relative rank tolerance")
    singularity_threshold = _positive_finite(
        singularity_threshold_angstrom,
        "inverse-distance singularity threshold",
    )
    requested_pairs = (
        tuple(combinations(sorted(subset), 2))
        if pairs_one_based is None
        else tuple(pairs_one_based)
    )
    pairs = _normalized_pairs(requested_pairs, atom_indices=subset)
    if not pairs:
        raise ValueError("inverse-distance substrate block needs at least one frozen pair")

    subset_lookup = {atom: index for index, atom in enumerate(subset)}
    subset_reference = full_reference[np.asarray([atom - 1 for atom in subset], dtype=int)]
    local_pairs = tuple(
        (subset_lookup[left] + 1, subset_lookup[right] + 1) for left, right in pairs
    )
    source_state = inverse_distance_source_state(
        subset_reference,
        local_pairs,
        singularity_threshold_angstrom=singularity_threshold,
    )
    cartesian_basis = symmetry_adapted_cartesian_basis(
        symbols,
        full_reference,
        symmetry=symmetry,
        atom_indices_one_based=subset,
        site_anchor_atom_indices_one_based=anchors,
        require_site_anchor=True,
        rank_absolute_tolerance=absolute_tolerance,
        rank_relative_tolerance=relative_tolerance,
    )
    coefficients, metric_eigenvalues, singular_values, residuals = (
        _cartesian_dual_source_coefficients(
            source_state.b_matrix,
            cartesian_basis,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
    )
    target_rank = cartesian_basis.target_rank
    if coefficients.shape != (target_rank, len(pairs)):
        raise ValueError("inverse-distance coefficient operator has an inconsistent shape")
    coordinate_ids = _inverse_distance_coordinate_identifiers(
        block_identifier,
        cartesian_basis.irreps,
    )
    source_order = tuple(
        f"{block_identifier}.Pair{left:04d}_{right:04d}" for left, right in pairs
    )
    degeneracy_groups = onic_degeneracy_groups_from_cartesian_basis(
        cartesian_basis,
        coordinate_ids,
        source_count=len(source_order),
        block_identifier=block_identifier,
        component_gauge=INVERSE_DISTANCE_BLOCK_GAUGE,
    )
    reference = tuple(tuple(float(value) for value in row) for row in subset_reference)
    validity_radius = max(
        (source_state.minimum_distance_angstrom - singularity_threshold) / np.sqrt(2.0),
        0.0,
    )
    condition_number = float(singular_values[0] / singular_values[-1])
    return OnicCoordinateBlock(
        identifier=block_identifier,
        kind="SUBSTRATE",
        representation="INVERSE_DISTANCE_PROJECTOR",
        atom_indices_one_based=subset,
        atom_indices_zero_based=tuple(atom - 1 for atom in subset),
        reference_coordinates_angstrom=reference,
        reference_fingerprint_sha256=onic_reference_fingerprint(subset, reference),
        source_family_identifiers=("INVERSE_DISTANCE_PAIR",),
        source_order=source_order,
        coordinate_identifiers=coordinate_ids,
        target_rank=target_rank,
        source_count=len(source_order),
        nullity=len(source_order) - target_rank,
        linearity=cartesian_basis.linearity,
        rank_method=INVERSE_DISTANCE_BLOCK_RANK_METHOD,
        rank_absolute_tolerance=absolute_tolerance,
        rank_relative_tolerance=relative_tolerance,
        coefficient_operator=OnicMatrixRecord(
            rows=target_rank,
            columns=len(source_order),
            storage="DENSE",
            dense_rows=tuple(tuple(float(value) for value in row) for row in coefficients),
        ),
        local_symmetry_provenance=local_symmetry_provenance,
        exact_retained_group=cartesian_basis.point_group,
        irrep_labels=cartesian_basis.irreps,
        degeneracy_groups=degeneracy_groups,
        component_gauge=INVERSE_DISTANCE_BLOCK_GAUGE,
        unit="ANGSTROM",
        scaling_policy="FROZEN_CARTESIAN_DUAL_OF_REFERENCE_INVERSE_DISTANCES",
        scale_factors=(1.0,) * target_rank,
        protected=protected,
        active=active,
        observable=observable,
        analytic_derivative_status="ANALYTIC_FIRST_ORDER",
        second_derivative_status="GENERAL_SPARSE_B_PRIME",
        diagnostics=OnicBlockDiagnostics(
            spectrum=tuple(float(value) for value in metric_eigenvalues),
            condition_number=condition_number,
            projector_symmetry_residual=0.0,
            projector_idempotency_residual=residuals["source_orthonormality"],
            row_space_residual=residuals["row_space"],
            covariance_residual=max(
                cartesian_basis.covariance_residual,
                residuals["cartesian_projector"],
            ),
            validity_radius=float(validity_radius),
            singularity_threshold=singularity_threshold,
            chirality_policy="REFERENCE_SITE_ORIENTATION_STRATUM",
            messages=(
                f"PAIR_SOURCE_COUNT={len(pairs)}",
                f"PHYSICAL_POINT_GROUP={cartesian_basis.physical_point_group}",
                f"COMPUTATIONAL_POINT_GROUP={cartesian_basis.point_group}",
                "SOURCE_PROJECTOR=IMPLICIT_NEVER_MATERIALIZED_IN_PRODUCTION",
                "VALUES=FROZEN_REFERENCE_CENTERED_INVERSE_DISTANCES",
                "COEFFICIENT_UNITS=ANGSTROM^2",
            ),
        ),
        site_frame=cartesian_basis.site_frame,
        provenance=(
            "SMITH_OWNED_INVERSE_DISTANCE_KERNEL",
            "REDUCTION=CARTESIAN_GRAM",
            f"POINT_GROUP={cartesian_basis.point_group}",
            f"PHYSICAL_POINT_GROUP={cartesian_basis.physical_point_group}",
        ),
    )


def evaluate_inverse_distance_projector_block(
    block: OnicCoordinateBlock,
    coordinates_angstrom: np.ndarray | Sequence[Sequence[float]],
) -> InverseDistanceBlockEvaluation:
    """Evaluate frozen inverse-distance values and analytic reduced B rows."""

    if block.representation != "INVERSE_DISTANCE_PROJECTOR":
        raise ValueError("inverse-distance evaluator received a different block representation")
    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or not np.all(np.isfinite(coordinates)):
        raise ValueError("inverse-distance evaluation needs a finite natoms-by-3 geometry")
    if max(block.atom_indices_one_based) > len(coordinates):
        raise ValueError("inverse-distance block references atoms outside the current geometry")
    pairs = _pairs_from_source_order(block)
    threshold = float(block.diagnostics.singularity_threshold or 0.0)
    current_state = inverse_distance_source_state(
        coordinates,
        pairs,
        singularity_threshold_angstrom=threshold,
    )
    subset_lookup = {
        atom: index for index, atom in enumerate(block.atom_indices_one_based)
    }
    local_pairs = tuple(
        (subset_lookup[left] + 1, subset_lookup[right] + 1) for left, right in pairs
    )
    reference_state = inverse_distance_source_state(
        np.asarray(block.reference_coordinates_angstrom, dtype=float),
        local_pairs,
        singularity_threshold_angstrom=threshold,
    )
    coefficients = np.asarray(block.coefficient_operator.to_dense(), dtype=float)
    source_delta = (
        current_state.values_inverse_angstrom - reference_state.values_inverse_angstrom
    )
    coordinate_values = coefficients @ source_delta
    reduced_dense_b = np.asarray(
        (current_state.b_matrix.T @ coefficients.T).T,
        dtype=float,
    )
    orientation_status = _orientation_guard_status(block, coordinates)
    if orientation_status == "REFLECTION_DETECTED":
        raise ValueError("inverse-distance chart crossed its frozen orientation stratum")
    return InverseDistanceBlockEvaluation(
        coordinate_values_angstrom=coordinate_values,
        source_values_inverse_angstrom=current_state.values_inverse_angstrom,
        b_matrix=SparseBMatrix.from_dense(
            reduced_dense_b,
            row_labels=block.coordinate_identifiers,
            backend="smith-inverse-distance-analytic.v1",
        ),
        minimum_distance_angstrom=current_state.minimum_distance_angstrom,
        orientation_guard_status=orientation_status,
    )


def evaluate_inverse_distance_projector_block_values(
    block: OnicCoordinateBlock,
    coordinates_angstrom: np.ndarray | Sequence[Sequence[float]],
) -> np.ndarray:
    """Evaluate only frozen reduced 1/r values, without constructing ``B``."""

    if block.representation != "INVERSE_DISTANCE_PROJECTOR":
        raise ValueError("inverse-distance evaluator received a different block representation")
    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or not np.all(np.isfinite(coordinates)):
        raise ValueError("inverse-distance evaluation needs a finite natoms-by-3 geometry")
    if max(block.atom_indices_one_based) > len(coordinates):
        raise ValueError("inverse-distance block references atoms outside the current geometry")
    pairs = _pairs_from_source_order(block)
    threshold = float(block.diagnostics.singularity_threshold or 0.0)
    _coords, _pairs, current_values, _delta, _distances, _minimum = (
        _inverse_distance_source_arrays(
            coordinates,
            pairs,
            singularity_threshold_angstrom=threshold,
        )
    )
    subset_lookup = {
        atom: index for index, atom in enumerate(block.atom_indices_one_based)
    }
    local_pairs = tuple(
        (subset_lookup[left] + 1, subset_lookup[right] + 1) for left, right in pairs
    )
    _reference, _local_pairs, reference_values, _delta, _distances, _minimum = (
        _inverse_distance_source_arrays(
            np.asarray(block.reference_coordinates_angstrom, dtype=float),
            local_pairs,
            singularity_threshold_angstrom=threshold,
        )
    )
    orientation_status = _orientation_guard_status(block, coordinates)
    if orientation_status == "REFLECTION_DETECTED":
        raise ValueError("inverse-distance chart crossed its frozen orientation stratum")
    coefficients = np.asarray(block.coefficient_operator.to_dense(), dtype=float)
    return np.asarray(coefficients @ (current_values - reference_values), dtype=float)


def explicit_inverse_distance_projector_audit(
    coordinates_angstrom: np.ndarray | Sequence[Sequence[float]],
    pairs_one_based: Sequence[tuple[int, int]],
    *,
    expected_cartesian_projector: np.ndarray | None = None,
    maximum_source_count: int = INVERSE_DISTANCE_EXPLICIT_AUDIT_MAX_SOURCES,
    singularity_threshold_angstrom: float = INVERSE_DISTANCE_SINGULARITY_THRESHOLD_ANGSTROM,
) -> InverseDistanceProjectorAudit:
    """Materialize ``B B+`` only for an explicitly bounded small audit."""

    state = inverse_distance_source_state(
        coordinates_angstrom,
        pairs_one_based,
        singularity_threshold_angstrom=singularity_threshold_angstrom,
    )
    source_count = state.b_matrix.shape[0]
    if source_count > int(maximum_source_count):
        raise ValueError(
            f"explicit source-projector audit is limited to {int(maximum_source_count)} pairs"
        )
    dense_b = state.b_matrix.toarray()
    projector = dense_b @ np.linalg.pinv(dense_b, rcond=1.0e-12)
    projector = 0.5 * (projector + projector.T)
    eigenvalues = np.linalg.eigvalsh(projector)
    symmetry_residual = float(np.linalg.norm(projector - projector.T, ord=2))
    idempotency_residual = float(np.linalg.norm(projector @ projector - projector, ord=2))
    row_space_residual = None
    if expected_cartesian_projector is not None:
        expected = np.asarray(expected_cartesian_projector, dtype=float)
        realized = dense_b.T @ np.linalg.pinv(dense_b @ dense_b.T, rcond=1.0e-12) @ dense_b
        if expected.shape != realized.shape:
            raise ValueError("expected Cartesian projector has an incompatible shape")
        row_space_residual = float(np.linalg.norm(realized - expected, ord=2))
    return InverseDistanceProjectorAudit(
        source_count=source_count,
        rank=numerical_matrix_rank(dense_b, absolute_tolerance=1.0e-10),
        eigenvalues=tuple(float(value) for value in eigenvalues),
        symmetry_residual=symmetry_residual,
        idempotency_residual=idempotency_residual,
        row_space_residual=row_space_residual,
    )


def _cartesian_dual_source_coefficients(
    source_b: csr_matrix,
    cartesian_basis: SymmetryAdaptedCartesianBasis,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    gram = np.asarray((source_b.T @ source_b).toarray(), dtype=float)
    gram = 0.5 * (gram + gram.T)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    maximum_eigenvalue = float(eigenvalues[0]) if len(eigenvalues) else 0.0
    # B.T @ B is the required production reduction, but taking square roots of
    # its round-off-level null eigenvalues turns O(eps) metric noise into
    # O(sqrt(eps)) apparent singular values.  Rank is therefore determined in
    # the native Gram spectrum, with the same absolute/relative contract used
    # by the other metric reductions in SMITH.
    eigenvalue_cutoff = max(
        absolute_tolerance,
        relative_tolerance * maximum_eigenvalue,
    )
    rank = int(np.count_nonzero(eigenvalues > eigenvalue_cutoff))
    if rank != cartesian_basis.target_rank:
        raise ValueError(
            "inverse-distance source space is not a complete internal chart: "
            f"rank={rank}, required={cartesian_basis.target_rank}; choose a nonlinear subset "
            "and/or add frozen pair sources"
        )
    retained_eigenvalues = eigenvalues[:rank]
    retained_vectors = eigenvectors[:, :rank]
    singular_values = np.sqrt(retained_eigenvalues)
    cartesian_columns = np.asarray(cartesian_basis.cartesian_from_q, dtype=float)
    dual_action = retained_vectors @ (
        (retained_vectors.T @ cartesian_columns) / retained_eigenvalues[:, None]
    )
    coefficients = np.asarray(source_b @ dual_action, dtype=float).T
    realized_b = np.asarray((source_b.T @ coefficients.T).T, dtype=float)
    row_space_residual = float(np.linalg.norm(realized_b - cartesian_columns.T, ord=2))
    cartesian_projector_residual = float(
        np.linalg.norm(
            realized_b.T @ realized_b - cartesian_columns @ cartesian_columns.T,
            ord=2,
        )
    )
    source_orthonormal = np.asarray(source_b @ retained_vectors, dtype=float)
    source_orthonormal /= singular_values[None, :]
    source_orthonormality_residual = float(
        np.linalg.norm(source_orthonormal.T @ source_orthonormal - np.eye(rank), ord=2)
    )
    maximum_residual = max(
        row_space_residual,
        cartesian_projector_residual,
        source_orthonormality_residual,
    )
    if maximum_residual > CARTESIAN_BLOCK_COVARIANCE_TOLERANCE:
        raise ValueError(
            "inverse-distance Cartesian-dual construction failed its row-space audit "
            f"(residual={maximum_residual:.3e})"
        )
    return (
        coefficients,
        retained_eigenvalues,
        singular_values,
        {
            "row_space": row_space_residual,
            "cartesian_projector": cartesian_projector_residual,
            "source_orthonormality": source_orthonormality_residual,
        },
    )


def _normalized_atom_subset(indices: Sequence[int], *, natoms: int) -> tuple[int, ...]:
    subset = tuple(int(atom) for atom in indices)
    if not subset or any(atom < 1 or atom > natoms for atom in subset):
        raise ValueError("inverse-distance subset contains an invalid one-based atom index")
    if len(set(subset)) != len(subset):
        raise ValueError("inverse-distance subset contains duplicate atoms")
    return subset


def _normalized_pairs(
    pairs: Sequence[tuple[int, int]],
    *,
    atom_indices: Sequence[int],
) -> tuple[tuple[int, int], ...]:
    allowed = set(int(atom) for atom in atom_indices)
    normalized: list[tuple[int, int]] = []
    for raw_pair in pairs:
        if len(raw_pair) != 2:
            raise ValueError("inverse-distance pair records must contain two atoms")
        left, right = sorted((int(raw_pair[0]), int(raw_pair[1])))
        if left == right or left not in allowed or right not in allowed:
            raise ValueError("inverse-distance pair must contain two distinct subset atoms")
        normalized.append((left, right))
    if len(set(normalized)) != len(normalized):
        raise ValueError("inverse-distance pair source contains duplicate unordered pairs")
    return tuple(normalized)


def _inverse_distance_coordinate_identifiers(
    block_identifier: str,
    irreps: Sequence[str],
) -> tuple[str, ...]:
    counters: dict[str, int] = {}
    identifiers: list[str] = []
    for irrep in irreps:
        counters[irrep] = counters.get(irrep, 0) + 1
        identifiers.append(
            f"{block_identifier}.{irrep_name_prefix(irrep)}Inv{counters[irrep]:04d}"
        )
    return tuple(identifiers)


def _pairs_from_source_order(block: OnicCoordinateBlock) -> tuple[tuple[int, int], ...]:
    pairs: list[tuple[int, int]] = []
    for source in block.source_order:
        match = _PAIR_IDENTIFIER_PATTERN.search(source)
        if match is None:
            raise ValueError(f"invalid inverse-distance source identifier: {source}")
        pairs.append((int(match.group(1)), int(match.group(2))))
    return _normalized_pairs(pairs, atom_indices=block.atom_indices_one_based)


def _orientation_signature(
    coordinates: np.ndarray,
    subset_atoms_one_based: Sequence[int],
    anchor_atoms_one_based: Sequence[int],
) -> float | None:
    subset = coordinates[np.asarray([atom - 1 for atom in subset_atoms_one_based], dtype=int)]
    anchors = coordinates[np.asarray([atom - 1 for atom in anchor_atoms_one_based], dtype=int)]
    anchor_center = np.mean(anchors, axis=0)
    radial = anchor_center - np.mean(subset, axis=0)
    first = anchors[0] - anchor_center
    for anchor in anchors[1:]:
        normal = np.cross(first, anchor - anchor_center)
        signature = float(normal @ radial)
        if abs(signature) > 1.0e-12:
            return signature
    return None


def _orientation_guard_status(
    block: OnicCoordinateBlock,
    current_coordinates: np.ndarray,
) -> str:
    if block.site_frame is None or len(block.site_frame.anchor_atom_indices_one_based) < 3:
        return "LEGACY_COEFFICIENT_GAUGE_ONLY"
    reference_full = np.zeros_like(current_coordinates)
    for atom, row in zip(
        block.atom_indices_one_based,
        block.reference_coordinates_angstrom,
        strict=True,
    ):
        reference_full[atom - 1] = np.asarray(row, dtype=float)
    anchors = block.site_frame.anchor_atom_indices_one_based
    reference_signature = _orientation_signature(
        reference_full,
        block.atom_indices_one_based,
        anchors,
    )
    current_signature = _orientation_signature(
        current_coordinates,
        block.atom_indices_one_based,
        anchors,
    )
    if reference_signature is None or current_signature is None:
        return "ORIENTATION_UNRESOLVED_CONTINUATION_REQUIRED"
    return (
        "REFERENCE_ORIENTATION_RETAINED"
        if reference_signature * current_signature > 0.0
        else "REFLECTION_DETECTED"
    )


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


__all__ = [
    "INVERSE_DISTANCE_BLOCK_ABSOLUTE_RANK_TOLERANCE",
    "INVERSE_DISTANCE_BLOCK_GAUGE",
    "INVERSE_DISTANCE_BLOCK_RANK_METHOD",
    "INVERSE_DISTANCE_BLOCK_RELATIVE_RANK_TOLERANCE",
    "INVERSE_DISTANCE_EXPLICIT_AUDIT_MAX_SOURCES",
    "INVERSE_DISTANCE_SINGULARITY_THRESHOLD_ANGSTROM",
    "InverseDistanceBlockEvaluation",
    "InverseDistanceProjectorAudit",
    "InverseDistanceSourceState",
    "build_inverse_distance_projector_block",
    "evaluate_inverse_distance_projector_block",
    "evaluate_inverse_distance_projector_block_values",
    "explicit_inverse_distance_projector_audit",
    "inverse_distance_source_state",
]
