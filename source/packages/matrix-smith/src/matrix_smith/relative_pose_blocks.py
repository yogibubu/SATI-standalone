"""Typed ONIC adapter for frozen relative-fragment pose coordinates.

The scientific coordinate kernels are the existing SMITH ``FTRANS`` and
``FROT`` primitives.  This module only selects one complete, homogeneous
translation/rotation sextet from a frozen GIC payload, audits ownership and
rank, and serializes its block identity.  It never rebuilds fragment frames or
defines a second quaternion/exponential-map implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Sequence

import numpy as np

from .bmatrix import SparseBMatrix
from .block_payload import (
    compact_sparse_b_matrix,
    embed_local_sparse_rows,
    frozen_payload_reference_coordinates,
    payload_owned_atom_frame,
    positive_finite,
)
from .coordinate_diagnostics import sonic_condition_diagnostics
from .definition import sonic_definition_identity_sha256
from .evaluation import build_sparse_gic_b_matrix, evaluate_gic_values_subset
from .models import GICDefinition, GICPrimitive
from .onic_blocks import (
    OnicBlockDiagnostics,
    OnicCoordinateBlock,
    OnicDegeneracyGroup,
    OnicMatrixRecord,
    onic_reference_fingerprint,
)
from .policy import RANK_TOLERANCE
from .symmetry_labels import irrep_dimension, irrep_name_prefix


RELATIVE_POSE_BLOCK_RANK_METHOD = "FROZEN_FTRANS_FROT_SPARSE_B_SVD_AUDIT"
RELATIVE_POSE_BLOCK_GAUGE = "FROZEN_REFERENCE_BODY_FRAME_EXPONENTIAL_MAP"
RELATIVE_POSE_BLOCK_ABSOLUTE_RANK_TOLERANCE = 1.0e-10
RELATIVE_POSE_BLOCK_RELATIVE_RANK_TOLERANCE = RANK_TOLERANCE
RELATIVE_POSE_BLOCK_SUPPORT_TOLERANCE = 1.0e-12
RELATIVE_POSE_BLOCK_RIGID_INVARIANCE_TOLERANCE = 5.0e-8
RELATIVE_POSE_BLOCK_CHART_LIMIT_RADIAN = float(np.pi)

_TRANSLATION = "FRAG_TRANSLATION"
_ROTATION = "FRAG_ORIENTATION"
_POSE_FAMILIES = (_TRANSLATION, _ROTATION)
_FUNCTION_BY_FAMILY = {_TRANSLATION: "FTRANS", _ROTATION: "FROT"}


@dataclass(frozen=True)
class RelativePoseBlockEvaluation:
    coordinate_values: np.ndarray
    b_matrix: SparseBMatrix
    payload_coordinate_indices: tuple[int, ...]
    translation_coordinate_indices: tuple[int, ...]
    rotation_coordinate_indices: tuple[int, ...]
    payload_identity_sha256: str


@dataclass(frozen=True)
class _RelativePoseEvaluationContext:
    current_coordinates: np.ndarray
    dependency_atoms: tuple[int, ...]
    payload_frame: str
    payload_coordinates: np.ndarray
    rotation_reference_coordinates: np.ndarray
    selected_indices: tuple[int, ...]
    family_by_index: dict[int, str]
    payload_identity_sha256: str


def build_exponential_map_relative_pose_block(
    definition: GICDefinition,
    *,
    reference_block: OnicCoordinateBlock,
    moving_block: OnicCoordinateBlock,
    block_identifier: str = "POSE1",
    local_symmetry_provenance: str = "FROZEN_GIC_RELATIVE_FRAGMENT_POSE",
    protected: bool = True,
    active: bool = True,
    observable: bool = False,
    rank_absolute_tolerance: float = RELATIVE_POSE_BLOCK_ABSOLUTE_RANK_TOLERANCE,
    rank_relative_tolerance: float = RELATIVE_POSE_BLOCK_RELATIVE_RANK_TOLERANCE,
    support_tolerance: float = RELATIVE_POSE_BLOCK_SUPPORT_TOLERANCE,
    rigid_invariance_tolerance: float = RELATIVE_POSE_BLOCK_RIGID_INVARIANCE_TOLERANCE,
) -> OnicCoordinateBlock:
    """Wrap one frozen nonlinear-fragment ``FTRANS/FROT`` pose sextet.

    The two dependency blocks own disjoint atoms.  ``definition`` may be a
    block-local payload ordered as reference atoms followed by moving atoms,
    or the complete frozen system GIC payload.
    """

    dependency_atoms = _validate_dependency_blocks(reference_block, moving_block)
    reference_full = frozen_payload_reference_coordinates(
        definition,
        payload_name="relative-pose frozen GIC",
    )
    payload_atoms, payload_frame = payload_owned_atom_frame(
        dependency_atoms,
        payload_natoms=len(reference_full),
        payload_name="relative-pose",
        explicit_local_order=True,
    )
    reference_payload_atoms, moving_payload_atoms = _payload_dependency_atoms(
        reference_block,
        moving_block,
        payload_frame=payload_frame,
    )
    reference_block_coordinates = _dependency_reference_coordinates(
        reference_block,
        moving_block,
    )
    payload_block_reference = _payload_block_reference(
        reference_full,
        dependency_atoms,
        payload_frame=payload_frame,
    )
    if not np.allclose(
        payload_block_reference,
        reference_block_coordinates,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("relative-pose frozen GIC reference does not match its dependency blocks")

    absolute_tolerance = positive_finite(
        rank_absolute_tolerance,
        "relative-pose absolute rank tolerance",
    )
    relative_tolerance = positive_finite(
        rank_relative_tolerance,
        "relative-pose relative rank tolerance",
    )
    support_limit = positive_finite(
        support_tolerance,
        "relative-pose support tolerance",
    )
    rigid_limit = positive_finite(
        rigid_invariance_tolerance,
        "relative-pose rigid-invariance tolerance",
    )
    selected_indices, family_by_index = _pose_coordinate_indices(
        definition,
        reference_payload_atoms=reference_payload_atoms,
        moving_payload_atoms=moving_payload_atoms,
    )
    _validate_pose_primitives(
        definition,
        selected_indices,
        family_by_index,
        reference_payload_atoms=reference_payload_atoms,
        moving_payload_atoms=moving_payload_atoms,
    )
    if (
        definition.point_group.strip().upper() not in {"C1", "UNKNOWN"}
        and not definition.symmetrize
    ):
        raise ValueError(
            "relative-pose nontrivial irreps require a symmetry-adapted frozen GIC payload"
        )

    sparse_b = build_sparse_gic_b_matrix(
        definition,
        coordinates_angstrom=definition.reference_coordinates_angstrom,
        coordinate_indices=selected_indices,
    )
    compact_b, outside_support = compact_sparse_b_matrix(
        sparse_b,
        payload_atoms=payload_atoms,
    )
    if outside_support > support_limit:
        raise ValueError(
            "relative-pose frozen GIC has Cartesian support outside its dependency union "
            f"(maximum={outside_support:.3e}, tolerance={support_limit:.3e})"
        )
    rank_diagnostics = sonic_condition_diagnostics(
        compact_b,
        tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    if int(rank_diagnostics["rank"]) != 6:
        raise ValueError(
            "relative-pose frozen GIC Jacobian is incomplete: "
            f"rank={rank_diagnostics['rank']}, required=6, "
            f"status={rank_diagnostics['status']}"
        )
    _validate_family_ranks(
        compact_b,
        selected_indices,
        family_by_index,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    rigid_residual = _rigid_invariance_residual(compact_b, reference_block_coordinates)
    if rigid_residual > rigid_limit:
        raise ValueError(
            "relative-pose Wilson rows do not annihilate whole-system rigid motion "
            f"(residual={rigid_residual:.3e}, tolerance={rigid_limit:.3e})"
        )

    selected_gics = tuple(definition.gics[index] for index in selected_indices)
    source_order = tuple(f"{block_identifier}.{gic.identifier}" for gic in selected_gics)
    coordinate_ids = tuple(f"{block_identifier}.{gic.name}" for gic in selected_gics)
    if len(set(coordinate_ids)) != 6:
        raise ValueError("relative-pose payload contains duplicate frozen GIC names")
    irreps = tuple(str(gic.irrep) for gic in selected_gics)
    ordered_families = tuple(family_by_index[index] for index in selected_indices)
    _validate_irrep_components(irreps, ordered_families)
    frame_metadata = _pose_frame_metadata(
        definition,
        selected_indices,
        family_by_index,
    )
    payload_identity = relative_pose_payload_identity_sha256(definition)
    reference = tuple(tuple(float(value) for value in row) for row in reference_block_coordinates)
    singular_values = tuple(float(value) for value in rank_diagnostics["singular_values"][:6])
    return OnicCoordinateBlock(
        identifier=block_identifier,
        kind="RELATIVE_POSE",
        representation="EXPONENTIAL_MAP",
        atom_indices_one_based=dependency_atoms,
        atom_indices_zero_based=tuple(atom - 1 for atom in dependency_atoms),
        reference_coordinates_angstrom=reference,
        reference_fingerprint_sha256=onic_reference_fingerprint(
            dependency_atoms,
            reference,
        ),
        source_family_identifiers=tuple(
            f"{block_identifier}.{family}" for family in _POSE_FAMILIES
        ),
        source_order=source_order,
        coordinate_identifiers=coordinate_ids,
        target_rank=6,
        source_count=6,
        nullity=0,
        linearity="NOT_APPLICABLE",
        rank_method=RELATIVE_POSE_BLOCK_RANK_METHOD,
        rank_absolute_tolerance=absolute_tolerance,
        rank_relative_tolerance=relative_tolerance,
        coefficient_operator=OnicMatrixRecord(rows=6, columns=6, storage="IDENTITY"),
        local_symmetry_provenance=local_symmetry_provenance,
        exact_retained_group=definition.point_group,
        irrep_labels=irreps,
        degeneracy_groups=_pose_degeneracy_groups(
            block_identifier,
            coordinate_ids,
            irreps,
            ordered_families,
        ),
        component_gauge=RELATIVE_POSE_BLOCK_GAUGE,
        unit="MIXED_ANGSTROM_RADIAN",
        scaling_policy="FROZEN_GIC_NATIVE_TRANSLATION_ROTATION_UNITS",
        scale_factors=(1.0,) * 6,
        protected=protected,
        active=active,
        observable=observable,
        analytic_derivative_status="ANALYTIC_FIRST_ORDER",
        second_derivative_status="GENERAL_SPARSE_B_PRIME",
        diagnostics=OnicBlockDiagnostics(
            spectrum=singular_values,
            condition_number=float(rank_diagnostics["condition_number"]),
            projector_symmetry_residual=0.0,
            projector_idempotency_residual=0.0,
            row_space_residual=outside_support,
            validity_radius=RELATIVE_POSE_BLOCK_CHART_LIMIT_RADIAN,
            chirality_policy="FROZEN_RIGHT_HANDED_FRAGMENT_BODY_FRAMES",
            messages=(
                f"PAYLOAD_ATOM_FRAME={payload_frame}",
                "TRANSLATION=REFERENCE_BODY_FIXED_FTRANS",
                "ROTATION=QUATERNION_EXPONENTIAL_MAP_FROT",
                "ROTATION_CHART_REBASE=FRAGMENT_ROTATION_ATLAS",
                f"MOVING_FRAME_ATOMS={frame_metadata[0]}",
                f"REFERENCE_FRAME_ATOMS={frame_metadata[1]}",
                f"RIGID_INVARIANCE_RESIDUAL={rigid_residual:.12g}",
                "COORDINATE_EVALUATION=DELEGATED_TO_FROZEN_GIC",
                "GLOBAL_REDUCTION=FORBIDDEN",
            ),
        ),
        payload_schema=definition.contract_schema_version,
        payload_identity_sha256=payload_identity,
        reference_block_id=reference_block.identifier,
        moving_block_id=moving_block.identifier,
        provenance=(
            "SMITH_EXISTING_RELATIVE_FRAGMENT_FTRANS_FROT",
            f"PAYLOAD_IDENTITY_SHA256={payload_identity}",
            f"PAYLOAD_ATOM_FRAME={payload_frame}",
            "POLAR_TRANSLATION_AND_AXIAL_ROTATION_FAMILIES_KEPT_SEPARATE",
        ),
    )


def evaluate_exponential_map_relative_pose_block(
    block: OnicCoordinateBlock,
    definition: GICDefinition,
    coordinates_angstrom: np.ndarray | Sequence[Sequence[float]],
    *,
    reference_block: OnicCoordinateBlock,
    moving_block: OnicCoordinateBlock,
    rotation_reference_coordinates: np.ndarray | Sequence[Sequence[float]] | None = None,
    parallel_workers: int = 1,
) -> RelativePoseBlockEvaluation:
    """Evaluate a typed pose block through the canonical frozen-GIC APIs."""

    context = _relative_pose_evaluation_context(
        block,
        definition,
        coordinates_angstrom,
        reference_block=reference_block,
        moving_block=moving_block,
        rotation_reference_coordinates=rotation_reference_coordinates,
    )
    values = evaluate_gic_values_subset(
        definition,
        context.selected_indices,
        coordinates_angstrom=context.payload_coordinates,
        rotation_reference_coordinates=context.rotation_reference_coordinates,
    )
    payload_b = build_sparse_gic_b_matrix(
        definition,
        coordinates_angstrom=context.payload_coordinates,
        rotation_reference_coordinates=context.rotation_reference_coordinates,
        coordinate_indices=context.selected_indices,
        parallel_workers=parallel_workers,
    )
    rows = (
        embed_local_sparse_rows(
            payload_b.rows,
            context.dependency_atoms,
            full_natoms=len(context.current_coordinates),
            payload_name="relative-pose",
        )
        if context.payload_frame == "LOCAL"
        else payload_b.rows
    )
    translation = tuple(
        position
        for position, payload_index in enumerate(context.selected_indices)
        if context.family_by_index[payload_index] == _TRANSLATION
    )
    rotation = tuple(
        position
        for position, payload_index in enumerate(context.selected_indices)
        if context.family_by_index[payload_index] == _ROTATION
    )
    return RelativePoseBlockEvaluation(
        coordinate_values=np.asarray(values, dtype=float),
        b_matrix=SparseBMatrix(
            rows=rows,
            column_count=context.current_coordinates.size,
            row_labels=block.coordinate_identifiers,
            backend="smith-relative-pose-frozen-gic.v1",
        ),
        payload_coordinate_indices=context.selected_indices,
        translation_coordinate_indices=translation,
        rotation_coordinate_indices=rotation,
        payload_identity_sha256=context.payload_identity_sha256,
    )


def evaluate_exponential_map_relative_pose_block_values(
    block: OnicCoordinateBlock,
    definition: GICDefinition,
    coordinates_angstrom: np.ndarray | Sequence[Sequence[float]],
    *,
    reference_block: OnicCoordinateBlock,
    moving_block: OnicCoordinateBlock,
    rotation_reference_coordinates: np.ndarray | Sequence[Sequence[float]] | None = None,
) -> np.ndarray:
    """Evaluate only the frozen relative-pose sextet, without constructing ``B``."""

    context = _relative_pose_evaluation_context(
        block,
        definition,
        coordinates_angstrom,
        reference_block=reference_block,
        moving_block=moving_block,
        rotation_reference_coordinates=rotation_reference_coordinates,
    )
    return np.asarray(
        evaluate_gic_values_subset(
            definition,
            context.selected_indices,
            coordinates_angstrom=context.payload_coordinates,
            rotation_reference_coordinates=context.rotation_reference_coordinates,
        ),
        dtype=float,
    )


def _relative_pose_evaluation_context(
    block: OnicCoordinateBlock,
    definition: GICDefinition,
    coordinates_angstrom: np.ndarray | Sequence[Sequence[float]],
    *,
    reference_block: OnicCoordinateBlock,
    moving_block: OnicCoordinateBlock,
    rotation_reference_coordinates: np.ndarray | Sequence[Sequence[float]] | None,
) -> _RelativePoseEvaluationContext:
    """Validate and resolve a pose payload for value-only or full evaluation."""

    if block.kind != "RELATIVE_POSE" or block.representation != "EXPONENTIAL_MAP":
        raise ValueError("exponential-map pose evaluator received another block type")
    dependency_atoms = _validate_dependency_blocks(reference_block, moving_block)
    if block.reference_block_id != reference_block.identifier:
        raise ValueError("relative-pose reference dependency does not match its typed block")
    if block.moving_block_id != moving_block.identifier:
        raise ValueError("relative-pose moving dependency does not match its typed block")
    if block.atom_indices_one_based != dependency_atoms:
        raise ValueError("relative-pose dependency atom union does not match its typed block")
    payload_identity = relative_pose_payload_identity_sha256(definition)
    if block.payload_schema != definition.contract_schema_version:
        raise ValueError("relative-pose payload schema does not match its typed block")
    if block.payload_identity_sha256 != payload_identity:
        raise ValueError("relative-pose payload checksum does not match its typed block")

    current = np.asarray(coordinates_angstrom, dtype=float)
    if current.ndim != 2 or current.shape[1] != 3 or not np.all(np.isfinite(current)):
        raise ValueError("relative-pose evaluation requires a finite natoms-by-3 geometry")
    if max(dependency_atoms) > len(current):
        raise ValueError("relative-pose block references atoms outside the current geometry")
    reference_full = frozen_payload_reference_coordinates(
        definition,
        payload_name="relative-pose frozen GIC",
    )
    _payload_atoms, payload_frame = payload_owned_atom_frame(
        dependency_atoms,
        payload_natoms=len(reference_full),
        payload_name="relative-pose",
        explicit_local_order=True,
    )
    payload_block_reference = _payload_block_reference(
        reference_full,
        dependency_atoms,
        payload_frame=payload_frame,
    )
    if not np.allclose(
        payload_block_reference,
        np.asarray(block.reference_coordinates_angstrom, dtype=float),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("relative-pose payload reference does not match its typed block")
    payload_coordinates = _payload_coordinates(
        current,
        dependency_atoms,
        payload_frame=payload_frame,
    )
    if payload_coordinates.shape != reference_full.shape:
        raise ValueError("current geometry does not match the frozen relative-pose payload")
    rotation_reference = _rotation_reference_payload(
        rotation_reference_coordinates,
        current=current,
        dependency_atoms=dependency_atoms,
        payload_frame=payload_frame,
        default=reference_full,
    )
    selected_indices = _payload_coordinate_indices(block, definition)
    reference_payload_atoms, moving_payload_atoms = _payload_dependency_atoms(
        reference_block,
        moving_block,
        payload_frame=payload_frame,
    )
    verified_indices, family_by_index = _pose_coordinate_indices(
        definition,
        reference_payload_atoms=reference_payload_atoms,
        moving_payload_atoms=moving_payload_atoms,
    )
    if selected_indices != verified_indices:
        raise ValueError("relative-pose source order does not match its frozen pose sextet")
    return _RelativePoseEvaluationContext(
        current_coordinates=current,
        dependency_atoms=dependency_atoms,
        payload_frame=payload_frame,
        payload_coordinates=payload_coordinates,
        rotation_reference_coordinates=rotation_reference,
        selected_indices=selected_indices,
        family_by_index=family_by_index,
        payload_identity_sha256=payload_identity,
    )


def _validate_dependency_blocks(
    reference_block: OnicCoordinateBlock,
    moving_block: OnicCoordinateBlock,
) -> tuple[int, ...]:
    if reference_block.kind == "RELATIVE_POSE" or moving_block.kind == "RELATIVE_POSE":
        raise ValueError("relative-pose dependencies must be owned atom blocks")
    if reference_block.identifier == moving_block.identifier:
        raise ValueError("relative-pose dependencies must be distinct blocks")
    overlap = set(reference_block.atom_indices_one_based).intersection(
        moving_block.atom_indices_one_based
    )
    if overlap:
        raise ValueError(
            "relative-pose dependency blocks overlap on atoms "
            + ",".join(str(atom) for atom in sorted(overlap))
        )
    return (
        *reference_block.atom_indices_one_based,
        *moving_block.atom_indices_one_based,
    )


def relative_pose_payload_identity_sha256(definition: GICDefinition) -> str:
    """Return a frame- and reference-sensitive identity for a pose payload."""

    reference = frozen_payload_reference_coordinates(
        definition,
        payload_name="relative-pose frozen GIC",
    )
    payload = {
        "frozen_sonic_identity_sha256": sonic_definition_identity_sha256(definition),
        "contract_schema_version": definition.contract_schema_version,
        "point_group": definition.point_group,
        "symmetrize": bool(definition.symmetrize),
        "reference_fingerprint_sha256": onic_reference_fingerprint(
            tuple(range(1, len(reference) + 1)),
            reference,
        ),
        "primitives": [
            {
                "identifier": primitive.identifier,
                "name": primitive.name,
                "family": primitive.family,
                "function": primitive.function,
                "atoms": list(primitive.atoms),
                "mode": int(primitive.mode),
                "ref_atoms": list(primitive.ref_atoms),
                "refs": list(primitive.refs),
                "frame_atoms": list(primitive.frame_atoms),
                "ref_frame_atoms": list(primitive.ref_frame_atoms),
            }
            for primitive in definition.primitives
        ],
        "gics": [
            {
                "identifier": gic.identifier,
                "name": gic.name,
                "family": gic.family,
                "irrep": gic.irrep,
                "primitive_id": gic.primitive_id,
                "coefficients": [
                    [primitive_id, float(coefficient).hex()]
                    for primitive_id, coefficient in (
                        gic.coefficients or ((gic.primitive_id, 1.0),)
                    )
                ],
            }
            for gic in definition.gics
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _dependency_reference_coordinates(
    reference_block: OnicCoordinateBlock,
    moving_block: OnicCoordinateBlock,
) -> np.ndarray:
    return np.vstack(
        (
            np.asarray(reference_block.reference_coordinates_angstrom, dtype=float),
            np.asarray(moving_block.reference_coordinates_angstrom, dtype=float),
        )
    )


def _payload_dependency_atoms(
    reference_block: OnicCoordinateBlock,
    moving_block: OnicCoordinateBlock,
    *,
    payload_frame: str,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if payload_frame == "LOCAL":
        split = len(reference_block.atom_indices_one_based)
        return tuple(range(1, split + 1)), tuple(
            range(split + 1, split + 1 + len(moving_block.atom_indices_one_based))
        )
    return reference_block.atom_indices_one_based, moving_block.atom_indices_one_based


def _payload_block_reference(
    reference_full: np.ndarray,
    dependency_atoms: tuple[int, ...],
    *,
    payload_frame: str,
) -> np.ndarray:
    if payload_frame == "LOCAL":
        return reference_full
    return reference_full[np.asarray([atom - 1 for atom in dependency_atoms], dtype=int)]


def _payload_coordinates(
    coordinates: np.ndarray,
    dependency_atoms: tuple[int, ...],
    *,
    payload_frame: str,
) -> np.ndarray:
    if payload_frame == "LOCAL":
        return coordinates[np.asarray([atom - 1 for atom in dependency_atoms], dtype=int)]
    return coordinates


def _rotation_reference_payload(
    coordinates: np.ndarray | Sequence[Sequence[float]] | None,
    *,
    current: np.ndarray,
    dependency_atoms: tuple[int, ...],
    payload_frame: str,
    default: np.ndarray,
) -> np.ndarray:
    if coordinates is None:
        return default
    candidate = np.asarray(coordinates, dtype=float)
    if candidate.ndim != 2 or candidate.shape[1] != 3 or not np.all(np.isfinite(candidate)):
        raise ValueError("relative-pose rotation reference must be a finite natoms-by-3 geometry")
    if candidate.shape == current.shape:
        return _payload_coordinates(candidate, dependency_atoms, payload_frame=payload_frame)
    if candidate.shape == default.shape:
        return candidate
    raise ValueError("relative-pose rotation reference has an incompatible atom count")


def _pose_coordinate_indices(
    definition: GICDefinition,
    *,
    reference_payload_atoms: tuple[int, ...],
    moving_payload_atoms: tuple[int, ...],
) -> tuple[tuple[int, ...], dict[int, str]]:
    primitive_by_id = {primitive.identifier: primitive for primitive in definition.primitives}
    selected: list[int] = []
    family_by_index: dict[int, str] = {}
    for index, gic in enumerate(definition.gics):
        matching: list[str] = []
        nonmatching = False
        coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
        for primitive_id, coefficient in coefficients:
            number = float(coefficient)
            if not np.isfinite(number):
                raise ValueError(f"frozen GIC {gic.identifier} has a non-finite coefficient")
            if abs(number) <= 1.0e-15:
                continue
            primitive = primitive_by_id.get(primitive_id)
            if primitive is None:
                raise ValueError(
                    f"frozen GIC {gic.identifier} references missing primitive {primitive_id}"
                )
            family = _matching_pose_family(
                primitive,
                reference_payload_atoms=reference_payload_atoms,
                moving_payload_atoms=moving_payload_atoms,
            )
            if family is None:
                nonmatching = True
            else:
                matching.append(family)
        if not matching:
            continue
        if nonmatching or len(set(matching)) != 1:
            raise ValueError(
                f"frozen GIC mixes relative-pose families or ownership: {gic.identifier}"
            )
        family = matching[0]
        if str(gic.family).strip().upper() != family:
            raise ValueError(f"frozen pose GIC {gic.identifier} family contradicts its primitives")
        selected.append(index)
        family_by_index[index] = family
    if len(selected) != 6:
        counts = {
            family: sum(family_by_index[index] == family for index in selected)
            for family in _POSE_FAMILIES
        }
        raise ValueError(
            "relative-pose payload must contain one complete translation/rotation sextet: "
            f"translations={counts[_TRANSLATION]}, rotations={counts[_ROTATION]}"
        )
    return tuple(selected), family_by_index


def _matching_pose_family(
    primitive: GICPrimitive,
    *,
    reference_payload_atoms: tuple[int, ...],
    moving_payload_atoms: tuple[int, ...],
) -> str | None:
    if set(primitive.atoms) != set(moving_payload_atoms):
        return None
    if set(primitive.ref_atoms) != set(reference_payload_atoms):
        return None
    for family, function in _FUNCTION_BY_FAMILY.items():
        if primitive.function == function:
            return family
    return None


def _validate_pose_primitives(
    definition: GICDefinition,
    selected_indices: tuple[int, ...],
    family_by_index: dict[int, str],
    *,
    reference_payload_atoms: tuple[int, ...],
    moving_payload_atoms: tuple[int, ...],
) -> None:
    primitive_by_id = {primitive.identifier: primitive for primitive in definition.primitives}
    primitives_by_family: dict[str, list[GICPrimitive]] = {family: [] for family in _POSE_FAMILIES}
    for index in selected_indices:
        gic = definition.gics[index]
        for primitive_id, coefficient in gic.coefficients or ((gic.primitive_id, 1.0),):
            if abs(float(coefficient)) <= 1.0e-15:
                continue
            primitive = primitive_by_id[primitive_id]
            family = family_by_index[index]
            if (
                _matching_pose_family(
                    primitive,
                    reference_payload_atoms=reference_payload_atoms,
                    moving_payload_atoms=moving_payload_atoms,
                )
                != family
            ):
                raise ValueError("relative-pose frozen GIC ownership changed during validation")
            primitives_by_family[family].append(primitive)
    for family in _POSE_FAMILIES:
        modes = {int(primitive.mode) for primitive in primitives_by_family[family]}
        if modes != {0, 1, 2}:
            raise ValueError(f"relative-pose {family} primitive modes must be exactly 0,1,2")
    translation_reference_frames = {
        tuple(primitive.ref_frame_atoms) for primitive in primitives_by_family[_TRANSLATION]
    }
    rotation_frames = {
        tuple(primitive.frame_atoms) for primitive in primitives_by_family[_ROTATION]
    }
    rotation_reference_frames = {
        tuple(primitive.ref_frame_atoms) for primitive in primitives_by_family[_ROTATION]
    }
    if (
        len(translation_reference_frames) != 1
        or len(rotation_frames) != 1
        or len(rotation_reference_frames) != 1
    ):
        raise ValueError("relative-pose primitives do not share one frozen body-frame gauge")
    translation_reference = next(iter(translation_reference_frames))
    moving_frame = next(iter(rotation_frames))
    rotation_reference = next(iter(rotation_reference_frames))
    if not moving_frame or not rotation_reference or translation_reference != rotation_reference:
        raise ValueError(
            "relative-pose exponential map requires one explicit moving frame and one "
            "shared body-fixed reference frame"
        )
    if not set(moving_frame).issubset(moving_payload_atoms):
        raise ValueError("relative-pose moving-frame anchors lie outside the moving block")
    if not set(rotation_reference).issubset(reference_payload_atoms):
        raise ValueError("relative-pose reference-frame anchors lie outside the reference block")


def _validate_family_ranks(
    compact_b: np.ndarray,
    selected_indices: tuple[int, ...],
    family_by_index: dict[int, str],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> None:
    for family in _POSE_FAMILIES:
        positions = tuple(
            position
            for position, index in enumerate(selected_indices)
            if family_by_index[index] == family
        )
        if len(positions) != 3:
            raise ValueError(f"relative-pose {family} block must contain exactly three rows")
        diagnostics = sonic_condition_diagnostics(
            compact_b[np.asarray(positions, dtype=int)],
            tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
        if int(diagnostics["rank"]) != 3:
            raise ValueError(
                f"relative-pose {family} block has rank {diagnostics['rank']}, required 3"
            )


def _rigid_invariance_residual(b_matrix: np.ndarray, coordinates: np.ndarray) -> float:
    centered = coordinates - np.mean(coordinates, axis=0)
    modes: list[np.ndarray] = []
    for axis in range(3):
        translation = np.zeros_like(coordinates)
        translation[:, axis] = 1.0
        modes.append(translation.reshape(-1))
    for axis in np.eye(3, dtype=float):
        modes.append(np.cross(centered, axis).reshape(-1))
    return float(np.max(np.abs(b_matrix @ np.asarray(modes, dtype=float).T), initial=0.0))


def _validate_irrep_components(
    irreps: tuple[str, ...],
    families: tuple[str, ...],
) -> None:
    for family in _POSE_FAMILIES:
        family_irreps = tuple(
            irrep
            for irrep, coordinate_family in zip(irreps, families, strict=True)
            if coordinate_family == family
        )
        for irrep in dict.fromkeys(family_irreps):
            count = family_irreps.count(irrep)
            dimension = irrep_dimension(irrep)
            if count % dimension:
                raise ValueError(
                    "relative-pose frozen GIC splits a multidimensional irrep inside "
                    f"{family}: {irrep} has {count} components, not a multiple of {dimension}"
                )


def _pose_frame_metadata(
    definition: GICDefinition,
    selected_indices: tuple[int, ...],
    family_by_index: dict[int, str],
) -> tuple[str, str]:
    primitive_by_id = {primitive.identifier: primitive for primitive in definition.primitives}
    moving_frames: set[tuple[int, ...]] = set()
    reference_frames: set[tuple[int, ...]] = set()
    for index in selected_indices:
        gic = definition.gics[index]
        for primitive_id, coefficient in gic.coefficients or ((gic.primitive_id, 1.0),):
            if abs(float(coefficient)) <= 1.0e-15:
                continue
            primitive = primitive_by_id[primitive_id]
            if family_by_index[index] == _ROTATION:
                moving_frames.add(tuple(primitive.frame_atoms))
            reference_frames.add(tuple(primitive.ref_frame_atoms))
    moving = next(iter(moving_frames))
    reference = next(iter(reference_frames))
    return ",".join(str(atom) for atom in moving), ",".join(str(atom) for atom in reference)


def _pose_degeneracy_groups(
    block_identifier: str,
    coordinate_ids: tuple[str, ...],
    irreps: tuple[str, ...],
    families: tuple[str, ...],
) -> tuple[OnicDegeneracyGroup, ...]:
    keys = tuple(dict.fromkeys(zip(families, irreps, strict=True)))
    groups: list[OnicDegeneracyGroup] = []
    for group_index, (family, irrep) in enumerate(keys, start=1):
        group_coordinates = tuple(
            coordinate
            for coordinate, coordinate_irrep, coordinate_family in zip(
                coordinate_ids,
                irreps,
                families,
                strict=True,
            )
            if coordinate_irrep == irrep and coordinate_family == family
        )
        group_prefix = f"{block_identifier}.{family}.{irrep_name_prefix(irrep)}Iso{group_index:03d}"
        groups.append(
            OnicDegeneracyGroup(
                identifier=group_prefix,
                irrep=irrep,
                coordinate_identifiers=group_coordinates,
                component_gauge=RELATIVE_POSE_BLOCK_GAUGE,
                projector=OnicMatrixRecord(
                    rows=6,
                    columns=6,
                    storage="IMPLICIT_FROM_COEFFICIENTS",
                    reference=f"{group_prefix}.frozen-gic-isotypic-subspace",
                ),
            )
        )
    return tuple(groups)


def _payload_coordinate_indices(
    block: OnicCoordinateBlock,
    definition: GICDefinition,
) -> tuple[int, ...]:
    index_by_identifier = {gic.identifier: index for index, gic in enumerate(definition.gics)}
    prefix = f"{block.identifier}."
    indices: list[int] = []
    for source in block.source_order:
        if not source.startswith(prefix):
            raise ValueError(f"invalid relative-pose source identifier: {source}")
        identifier = source[len(prefix) :]
        if identifier not in index_by_identifier:
            raise ValueError(f"relative-pose source {identifier} is absent from its frozen payload")
        indices.append(index_by_identifier[identifier])
    return tuple(indices)


__all__ = [
    "RELATIVE_POSE_BLOCK_ABSOLUTE_RANK_TOLERANCE",
    "RELATIVE_POSE_BLOCK_CHART_LIMIT_RADIAN",
    "RELATIVE_POSE_BLOCK_GAUGE",
    "RELATIVE_POSE_BLOCK_RANK_METHOD",
    "RELATIVE_POSE_BLOCK_RELATIVE_RANK_TOLERANCE",
    "RELATIVE_POSE_BLOCK_RIGID_INVARIANCE_TOLERANCE",
    "RELATIVE_POSE_BLOCK_SUPPORT_TOLERANCE",
    "RelativePoseBlockEvaluation",
    "build_exponential_map_relative_pose_block",
    "evaluate_exponential_map_relative_pose_block",
    "evaluate_exponential_map_relative_pose_block_values",
    "relative_pose_payload_identity_sha256",
]
