"""Closure-aware contact--pose selection from a frozen ORACLE contract."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np

from matrix_chem import read_molecular_symmetry
from matrix_numerics import (
    normalized_matrix_condition,
    numerical_matrix_rank,
    select_rank_revealing_rows,
)

from .contracts import GICForgeContractError
from .bmatrix import SparseBRow
from .audits.dense import normalize_dense_rows
from .oracle_contract_consumer import load_validated_oracle_candidate_pool
from .policy import RANK_TOLERANCE
from .symmetrization import _cartesian_operation_matrix
from .symmetry_labels import irrep_characters_for_operations


CLOSED_CONTACT_SELECTION_SCHEMA = "matrix.smith.closed_contact_selection.v1"
_COVARIANCE_TOLERANCE = 1.0e-7
_MAX_EXACT_POSE_SUBSETS = 4096


@dataclass(frozen=True)
class SymmetryAdaptedContactCoordinate:
    irrep: str
    coefficients: tuple[tuple[str, float], ...]
    wilson_row: tuple[float, ...]


@dataclass(frozen=True)
class ClosedContactSelection:
    policy: str
    intrafragment_candidate_ids: tuple[str, ...]
    contact_coordinates: tuple[SymmetryAdaptedContactCoordinate, ...]
    pose_candidate_ids: tuple[str, ...]
    observable_contact_ids: tuple[str, ...]
    intrafragment_rank: int
    contact_rank: int
    relative_rank: int
    normalized_relative_condition_number: float
    fallback_reason: str = ""
    provenance: str = CLOSED_CONTACT_SELECTION_SCHEMA


def select_closed_contact_pose(
    path: Path,
    *,
    rank_tolerance: float = RANK_TOLERANCE,
    contact_realization_available: bool = True,
) -> ClosedContactSelection:
    """Apply the frozen contact--pose construction or its pure-pose fallback.

    Chemical membership and contact status come entirely from ORACLE.  SMITH
    performs only Wilson-space projection, symmetry adaptation, finite-function
    subset selection, and numerical audits.
    """

    pool = load_validated_oracle_candidate_pool(Path(path))
    contract = pool.contract
    candidates = contract.primitive_candidates
    rows = tuple(
        row.to_dense() if isinstance(row, SparseBRow) else np.asarray(row, dtype=float)
        for row in pool.rows
    )
    index_by_id = {candidate.candidate_id: index for index, candidate in enumerate(candidates)}
    closed_components = _closed_fragment_components(contract.auxiliary_contacts)
    if not closed_components:
        raise GICForgeContractError("ORACLE contract contains no closed auxiliary-contact set")

    selected_contact_ids = {
        contact.contact_id
        for contact in contract.auxiliary_contacts
        if any(set(contact.fragment_ids).issubset(component) for component in closed_components)
    }
    contact_candidate_ids = tuple(
        candidate_id
        for contact in contract.auxiliary_contacts
        if contact.contact_id in selected_contact_ids
        for candidate_id in contact.primitive_candidate_ids
        if candidates[index_by_id[candidate_id]].family != "CONTACT_SUPPORT"
    )
    if not contact_candidate_ids:
        contact_realization_available = False

    involved_fragments = set().union(*closed_components)
    pose_indices = tuple(
        index
        for index, candidate in enumerate(candidates)
        if candidate.domain_id == "FRAGMENT_POSE"
        and _pose_owner_fragments(candidate.owner_id).issubset(involved_fragments)
    )
    if not pose_indices:
        raise GICForgeContractError(
            "closed auxiliary contacts require ORACLE-supplied finite fragment-pose functions"
        )

    excluded = {index_by_id[candidate_id] for candidate_id in contact_candidate_ids}
    excluded.update(index for index, candidate in enumerate(candidates)
                    if candidate.family == "CONTACT_SUPPORT")
    excluded.update(pose_indices)
    b0_indices = tuple(index for index in range(len(candidates)) if index not in excluded)
    b0_indices = _independent_row_indices(rows, b0_indices, rank_tolerance=rank_tolerance)
    b0 = _rows(rows, b0_indices, columns=3 * contract.primary_topology.natoms)
    identity = np.eye(3 * contract.primary_topology.natoms, dtype=float)
    p0 = np.linalg.pinv(b0, rcond=rank_tolerance) @ b0 if len(b0) else np.zeros_like(identity)
    q0 = identity - p0
    bp = _rows(rows, pose_indices, columns=identity.shape[0])
    relative_rank = _rank(bp @ q0, rank_tolerance)
    if relative_rank <= 0:
        raise GICForgeContractError("ORACLE fragment-pose block has no relative Wilson rank")

    symmetry = read_molecular_symmetry(Path(path))
    operations = tuple(symmetry.operations)
    if not contact_realization_available:
        return _pure_pose_fallback(
            candidates,
            rows,
            b0_indices=b0_indices,
            pose_indices=pose_indices,
            projector=q0,
            relative_rank=relative_rank,
            operations=operations,
            observable_contact_ids=tuple(sorted(selected_contact_ids)),
            rank_tolerance=rank_tolerance,
            reason="ORACLE contact realization is unavailable",
        )

    contact_indices = tuple(index_by_id[candidate_id] for candidate_id in contact_candidate_ids)
    cc = _rows(rows, contact_indices, columns=identity.shape[0])
    cc_perp = cc @ q0
    contact_rank = _rank(cc_perp, rank_tolerance)
    if contact_rank <= 0 or contact_rank > relative_rank:
        return _pure_pose_fallback(
            candidates,
            rows,
            b0_indices=b0_indices,
            pose_indices=pose_indices,
            projector=q0,
            relative_rank=relative_rank,
            operations=operations,
            observable_contact_ids=tuple(sorted(selected_contact_ids)),
            rank_tolerance=rank_tolerance,
            reason="contact distances have invalid independent relative rank",
        )
    try:
        contact_coordinates, cind = _symmetry_adapted_contact_rows(
            candidates,
            cc,
            cc_perp,
            contact_indices=contact_indices,
            operations=operations,
            point_group=symmetry.point_group,
            target_rank=contact_rank,
            rank_tolerance=rank_tolerance,
        )
    except GICForgeContractError as exc:
        return _pure_pose_fallback(
            candidates,
            rows,
            b0_indices=b0_indices,
            pose_indices=pose_indices,
            projector=q0,
            relative_rank=relative_rank,
            operations=operations,
            observable_contact_ids=tuple(sorted(selected_contact_ids)),
            rank_tolerance=rank_tolerance,
            reason=str(exc),
        )

    pc = np.linalg.pinv(cind, rcond=rank_tolerance) @ cind
    pose_projector = identity - p0 - pc
    pose_count = relative_rank - contact_rank
    chosen = _best_finite_pose_subset(
        rows,
        pose_indices,
        projector=pose_projector,
        count=pose_count,
        operations=operations,
        fixed_rows=cind,
        rank_tolerance=rank_tolerance,
    )
    if chosen is None:
        return _pure_pose_fallback(
            candidates,
            rows,
            b0_indices=b0_indices,
            pose_indices=pose_indices,
            projector=q0,
            relative_rank=relative_rank,
            operations=operations,
            observable_contact_ids=tuple(sorted(selected_contact_ids)),
            rank_tolerance=rank_tolerance,
            reason="no symmetry-closed finite pose subset completes the contact span",
        )
    chosen_indices, condition = chosen
    raw_contacts = np.vstack(
        [np.asarray(item.wilson_row, dtype=float) for item in contact_coordinates]
    )
    final_rows = np.vstack(
        (b0, raw_contacts, _rows(rows, chosen_indices, columns=identity.shape[0]))
    )
    expected_rank = len(b0_indices) + contact_rank + pose_count
    if _rank(final_rows, rank_tolerance) != expected_rank:
        raise GICForgeContractError("contact-pose Wilson rank audit failed")
    return ClosedContactSelection(
        policy="CONTACT_POSE",
        intrafragment_candidate_ids=tuple(candidates[index].candidate_id for index in b0_indices),
        contact_coordinates=contact_coordinates,
        pose_candidate_ids=tuple(candidates[index].candidate_id for index in chosen_indices),
        observable_contact_ids=(),
        intrafragment_rank=len(b0_indices),
        contact_rank=contact_rank,
        relative_rank=relative_rank,
        normalized_relative_condition_number=condition,
    )


def _closed_fragment_components(contacts) -> tuple[frozenset[str], ...]:
    edges = [(set(contact.fragment_ids), contact.open_or_closing == "CLOSING") for contact in contacts]
    components: list[set[str]] = []
    closed_flags: list[bool] = []
    for edge, is_closing in edges:
        merged = set(edge)
        merged_closed = bool(is_closing)
        retained_components: list[set[str]] = []
        retained_flags: list[bool] = []
        for component, was_closed in zip(components, closed_flags, strict=True):
            if component.intersection(edge):
                merged.update(component)
                merged_closed = merged_closed or was_closed
            else:
                retained_components.append(component)
                retained_flags.append(was_closed)
        components = [*retained_components, merged]
        closed_flags = [*retained_flags, merged_closed]
    return tuple(
        frozenset(component)
        for component, is_closed in zip(components, closed_flags, strict=True)
        if is_closed
    )


def _pose_owner_fragments(owner_id: str) -> frozenset[str]:
    if not owner_id.startswith("FRAGMENT_PAIR:"):
        return frozenset()
    return frozenset(value for value in owner_id.split(":", 1)[1].split("|") if value)


def _independent_row_indices(
    rows: tuple[np.ndarray, ...],
    indices: tuple[int, ...],
    *,
    rank_tolerance: float,
) -> tuple[int, ...]:
    if not indices:
        return ()
    matrix = _rows(rows, indices, columns=len(rows[0]))
    normalized = _normalize_rows(matrix, rank_tolerance)
    selected = select_rank_revealing_rows(
        normalized,
        tolerance=rank_tolerance,
        tie_tolerance=1.0e-12,
    )
    return tuple(indices[index] for index in selected.indices)


def _symmetry_adapted_contact_rows(
    candidates,
    cc: np.ndarray,
    cc_perp: np.ndarray,
    *,
    contact_indices: tuple[int, ...],
    operations,
    point_group: str,
    target_rank: int,
    rank_tolerance: float,
) -> tuple[tuple[SymmetryAdaptedContactCoordinate, ...], np.ndarray]:
    if not operations:
        raise GICForgeContractError("closed contact selection needs frozen symmetry operations")
    transforms = tuple(
        _row_representation_transform(cc, operation, rank_tolerance=rank_tolerance)
        for operation in operations
    )
    labels = tuple(operation.label for operation in operations)
    matrices = tuple(operation.rotation for operation in operations)
    projected: list[tuple[str, np.ndarray, np.ndarray]] = []
    for irrep, characters in irrep_characters_for_operations(
        labels,
        point_group,
        operation_matrices=matrices,
    ):
        for source in range(len(contact_indices)):
            vector = np.zeros(len(contact_indices), dtype=float)
            vector[source] = 1.0
            coefficients = sum(
                float(character) * (transform @ vector)
                for character, transform in zip(characters, transforms, strict=True)
            ) / float(len(transforms))
            norm = float(np.linalg.norm(coefficients))
            if norm <= rank_tolerance:
                continue
            coefficients /= norm
            row = coefficients @ cc_perp
            row_norm = float(np.linalg.norm(row))
            if row_norm <= rank_tolerance:
                continue
            projected.append((irrep, coefficients, row / row_norm))
    if not projected:
        raise GICForgeContractError("contact symmetry projection produced no finite rows")
    selection = select_rank_revealing_rows(
        np.vstack([record[2] for record in projected]),
        target_rank=target_rank,
        tolerance=rank_tolerance,
        tie_tolerance=1.0e-12,
    )
    if selection.rank != target_rank:
        raise GICForgeContractError("contact symmetry projection lost Wilson rank")
    coordinates: list[SymmetryAdaptedContactCoordinate] = []
    cind_rows: list[np.ndarray] = []
    for selected_index in selection.indices:
        irrep, coefficients, _normalized_row = projected[selected_index]
        coefficients = _canonical_vector_sign(coefficients)
        row = coefficients @ cc
        cind_rows.append(coefficients @ cc_perp)
        coordinates.append(
            SymmetryAdaptedContactCoordinate(
                irrep=irrep,
                coefficients=tuple(
                    (candidates[index].candidate_id, float(value))
                    for index, value in zip(contact_indices, coefficients, strict=True)
                    if abs(float(value)) > 1.0e-12
                ),
                wilson_row=tuple(float(value) for value in row),
            )
        )
    cind = np.vstack(cind_rows)
    if _rank(cind, rank_tolerance) != target_rank:
        raise GICForgeContractError("contact symmetry-adapted rows are rank deficient")
    return tuple(coordinates), cind


def _row_representation_transform(matrix: np.ndarray, operation, *, rank_tolerance: float) -> np.ndarray:
    cartesian = _cartesian_operation_matrix(operation, natoms=matrix.shape[1] // 3)
    transform = np.zeros((matrix.shape[0], matrix.shape[0]), dtype=float)
    for source, row in enumerate(matrix):
        transformed = row @ cartesian
        coefficients, *_ = np.linalg.lstsq(matrix.T, transformed.T, rcond=rank_tolerance)
        residual = float(np.linalg.norm(coefficients @ matrix - transformed))
        if residual > _COVARIANCE_TOLERANCE:
            raise GICForgeContractError(
                "ORACLE closing-contact orbit is not Wilson-covariant under molecular symmetry"
            )
        transform[:, source] = coefficients
    return transform


def _best_finite_pose_subset(
    rows: tuple[np.ndarray, ...],
    pose_indices: tuple[int, ...],
    *,
    projector: np.ndarray,
    count: int,
    operations,
    fixed_rows: np.ndarray,
    rank_tolerance: float,
) -> tuple[tuple[int, ...], float] | None:
    if count == 0:
        return (), _normalized_condition(fixed_rows, rank_tolerance)
    if count < 0 or count > len(pose_indices):
        return None
    if comb(len(pose_indices), count) > _MAX_EXACT_POSE_SUBSETS:
        residual = _rows(rows, pose_indices, columns=projector.shape[0]) @ projector
        selected = select_rank_revealing_rows(
            _normalize_rows(residual, rank_tolerance),
            target_rank=count,
            tolerance=rank_tolerance,
            tie_tolerance=1.0e-12,
        )
        subsets = (tuple(pose_indices[index] for index in selected.indices),)
    else:
        subsets = combinations(pose_indices, count)
    best: tuple[tuple[int, ...], float] | None = None
    for subset in subsets:
        subset = tuple(subset)
        raw = _rows(rows, subset, columns=projector.shape[0])
        residual = raw @ projector
        if _rank(residual, rank_tolerance) != count:
            continue
        if not _subspace_is_covariant(
            raw,
            residual,
            projector=projector,
            operations=operations,
            rank_tolerance=rank_tolerance,
        ):
            continue
        condition = _normalized_condition(np.vstack((fixed_rows, residual)), rank_tolerance)
        if not np.isfinite(condition):
            continue
        if best is None or condition < best[1] - 1.0e-12:
            best = (subset, condition)
    return best


def _subspace_is_covariant(
    raw: np.ndarray,
    residual: np.ndarray,
    *,
    projector: np.ndarray,
    operations,
    rank_tolerance: float,
) -> bool:
    for operation in operations:
        cartesian = _cartesian_operation_matrix(operation, natoms=raw.shape[1] // 3)
        transformed = (raw @ cartesian) @ projector
        coefficients, *_ = np.linalg.lstsq(residual.T, transformed.T, rcond=rank_tolerance)
        if float(np.linalg.norm(coefficients.T @ residual - transformed)) > _COVARIANCE_TOLERANCE:
            return False
    return True


def _pure_pose_fallback(
    candidates,
    rows: tuple[np.ndarray, ...],
    *,
    b0_indices: tuple[int, ...],
    pose_indices: tuple[int, ...],
    projector: np.ndarray,
    relative_rank: int,
    operations,
    observable_contact_ids: tuple[str, ...],
    rank_tolerance: float,
    reason: str,
) -> ClosedContactSelection:
    chosen = _best_finite_pose_subset(
        rows,
        pose_indices,
        projector=projector,
        count=relative_rank,
        operations=operations,
        fixed_rows=np.zeros((0, projector.shape[0])),
        rank_tolerance=rank_tolerance,
    )
    if chosen is None:
        raise GICForgeContractError(
            f"contact-pose failed ({reason}); pure-pose fallback also failed"
        )
    indices, condition = chosen
    return ClosedContactSelection(
        policy="PURE_POSE_FALLBACK",
        intrafragment_candidate_ids=tuple(candidates[index].candidate_id for index in b0_indices),
        contact_coordinates=(),
        pose_candidate_ids=tuple(candidates[index].candidate_id for index in indices),
        observable_contact_ids=observable_contact_ids,
        intrafragment_rank=len(b0_indices),
        contact_rank=0,
        relative_rank=relative_rank,
        normalized_relative_condition_number=condition,
        fallback_reason=reason,
    )


def _rows(rows: tuple[np.ndarray, ...], indices: tuple[int, ...], *, columns: int) -> np.ndarray:
    return np.vstack([rows[index] for index in indices]) if indices else np.zeros((0, columns))


def _rank(matrix: np.ndarray, tolerance: float) -> int:
    return numerical_matrix_rank(
        np.asarray(matrix, dtype=float),
        absolute_tolerance=tolerance,
    )


def _normalize_rows(matrix: np.ndarray, tolerance: float) -> np.ndarray:
    return normalize_dense_rows(matrix, tolerance=tolerance)


def _normalized_condition(matrix: np.ndarray, tolerance: float) -> float:
    values = np.asarray(matrix, dtype=float)
    if not len(values):
        return 1.0
    return normalized_matrix_condition(
        values,
        absolute_tolerance=tolerance,
        zero_row_tolerance=tolerance,
        required_rank=len(values),
    )


def _canonical_vector_sign(vector: np.ndarray) -> np.ndarray:
    result = np.asarray(vector, dtype=float).copy()
    nonzero = np.flatnonzero(np.abs(result) > 1.0e-12)
    if len(nonzero) and result[int(nonzero[0])] < 0.0:
        result *= -1.0
    return result


__all__ = [
    "CLOSED_CONTACT_SELECTION_SCHEMA",
    "ClosedContactSelection",
    "SymmetryAdaptedContactCoordinate",
    "select_closed_contact_pose",
]
