"""Typed ONIC adapter for existing frozen natural/domain-first SONIC charts.

This module does not generate, symmetrize or reduce molecular coordinates.
Those scientific operations remain owned by the canonical ``GICDefinition``
builder.  It only selects an atom-owned, already non-redundant GIC subchart,
audits its Cartesian support and serializes the resulting typed block.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

import numpy as np

from matrix_chem import expected_vibrational_mode_count, is_linear_geometry

from .bmatrix import SparseBMatrix
from .block_payload import (
    compact_sparse_b_matrix,
    compact_sparse_b_matrix_sparse,
    embed_local_sparse_rows,
    frozen_payload_reference_coordinates,
    normalized_owned_atoms,
    payload_owned_atom_frame,
    positive_finite,
)
from .coordinate_diagnostics import (
    sonic_condition_diagnostics,
    sparse_full_row_rank_diagnostics,
)
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


NATURAL_INTERNAL_BLOCK_RANK_METHOD = "FROZEN_GIC_SPARSE_B_SVD_AUDIT"
NATURAL_INTERNAL_BLOCK_GAUGE = "FROZEN_DOMAIN_FIRST_GIC_GAUGE"
NATURAL_INTERNAL_BLOCK_ABSOLUTE_RANK_TOLERANCE = 1.0e-10
NATURAL_INTERNAL_BLOCK_RELATIVE_RANK_TOLERANCE = RANK_TOLERANCE
NATURAL_INTERNAL_BLOCK_SUPPORT_TOLERANCE = 1.0e-12
_SAFE_TOKEN_PATTERN = re.compile(r"[^A-Za-z0-9_.:-]+")


@dataclass(frozen=True)
class NaturalInternalBlockEvaluation:
    coordinate_values: np.ndarray
    b_matrix: SparseBMatrix
    payload_coordinate_indices: tuple[int, ...]
    payload_identity_sha256: str


@dataclass(frozen=True)
class _NaturalInternalEvaluationContext:
    current_coordinates: np.ndarray
    payload_coordinates: np.ndarray
    payload_frame: str
    selected_indices: tuple[int, ...]
    payload_identity_sha256: str
    payload_to_current_rotation: np.ndarray | None = None


def build_natural_internal_block(
    definition: GICDefinition,
    *,
    atom_indices_one_based: Sequence[int],
    block_identifier: str = "MOL1",
    block_kind: str = "MOLECULE_INTERNAL",
    local_symmetry_provenance: str = "FROZEN_GIC_DOMAIN_FIRST",
    protected: bool = True,
    active: bool = True,
    observable: bool = False,
    rank_absolute_tolerance: float = NATURAL_INTERNAL_BLOCK_ABSOLUTE_RANK_TOLERANCE,
    rank_relative_tolerance: float = NATURAL_INTERNAL_BLOCK_RELATIVE_RANK_TOLERANCE,
    support_tolerance: float = NATURAL_INTERNAL_BLOCK_SUPPORT_TOLERANCE,
) -> OnicCoordinateBlock:
    """Wrap one owned frozen GIC subchart without rebuilding its coordinates.

    ``definition`` may describe only the owned molecule or a larger system.  A
    molecule-only payload is mapped in its native atom order onto the declared
    block atoms; a larger payload is filtered by exact atom ownership.
    """

    kind = str(block_kind).strip().upper().replace("-", "_")
    if kind not in {"SUBSTRATE", "MOLECULE_INTERNAL"}:
        raise ValueError("natural-internal block kind must be SUBSTRATE or MOLECULE_INTERNAL")
    reference_full = frozen_payload_reference_coordinates(
        definition,
        payload_name="frozen GIC",
    )
    block_atoms = normalized_owned_atoms(
        atom_indices_one_based,
        block_name="natural-internal block",
    )
    payload_atoms, payload_frame = payload_owned_atom_frame(
        block_atoms,
        payload_natoms=len(reference_full),
        payload_name="natural-internal",
    )
    absolute_tolerance = positive_finite(
        rank_absolute_tolerance,
        "natural-internal absolute rank tolerance",
    )
    relative_tolerance = positive_finite(
        rank_relative_tolerance,
        "natural-internal relative rank tolerance",
    )
    support_limit = positive_finite(
        support_tolerance,
        "natural-internal support tolerance",
    )
    selected_indices = _owned_gic_indices(definition, payload_atoms)
    if not selected_indices:
        raise ValueError("natural-internal block contains no atom-owned frozen GICs")
    selected_gics = tuple(definition.gics[index] for index in selected_indices)
    if len({gic.identifier for gic in selected_gics}) != len(selected_gics):
        raise ValueError("natural-internal payload contains duplicate GIC identifiers")
    if (
        definition.point_group.strip().upper() not in {"C1", "UNKNOWN"}
        and not definition.symmetrize
    ):
        raise ValueError(
            "natural-internal nontrivial irreps require a symmetry-adapted frozen GIC payload"
        )

    reference_block = (
        reference_full
        if payload_frame == "LOCAL"
        else reference_full[np.asarray([atom - 1 for atom in block_atoms], dtype=int)]
    )
    expected_rank = expected_vibrational_mode_count(reference_block)
    linearity = _linearity(reference_block)
    if len(selected_gics) != expected_rank:
        raise ValueError(
            "natural-internal frozen GIC count does not match the owned molecular rank: "
            f"coordinates={len(selected_gics)}, required={expected_rank} ({linearity})"
        )

    sparse_b = build_sparse_gic_b_matrix(
        definition,
        coordinates_angstrom=definition.reference_coordinates_angstrom,
        coordinate_indices=selected_indices,
    )
    large_sparse_audit = expected_rank >= 512
    if large_sparse_audit:
        compact_b, outside_support = compact_sparse_b_matrix_sparse(
            sparse_b,
            payload_atoms=payload_atoms,
        )
    else:
        compact_b, outside_support = compact_sparse_b_matrix(
            sparse_b,
            payload_atoms=payload_atoms,
        )
    if outside_support > support_limit:
        raise ValueError(
            "natural-internal frozen GIC has Cartesian support outside its owned block "
            f"(maximum={outside_support:.3e}, tolerance={support_limit:.3e})"
        )
    rank_diagnostics = (
        sparse_full_row_rank_diagnostics(
            compact_b,
            reference_block,
            tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
        if large_sparse_audit
        else sonic_condition_diagnostics(
            compact_b,
            tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
    )
    evaluated_rank = int(rank_diagnostics["rank"])
    if evaluated_rank != expected_rank:
        raise ValueError(
            "natural-internal frozen GIC Jacobian is incomplete: "
            f"rank={evaluated_rank}, required={expected_rank}, "
            f"status={rank_diagnostics['status']}"
        )

    source_order = tuple(f"{block_identifier}.{gic.identifier}" for gic in selected_gics)
    coordinate_ids = tuple(f"{block_identifier}.{gic.name}" for gic in selected_gics)
    if len(set(coordinate_ids)) != len(coordinate_ids):
        raise ValueError("natural-internal payload contains duplicate frozen GIC names")
    irreps = tuple(str(gic.irrep) for gic in selected_gics)
    for irrep in dict.fromkeys(irreps):
        count = irreps.count(irrep)
        dimension = irrep_dimension(irrep)
        if count % dimension:
            raise ValueError(
                "natural-internal frozen GIC splits a multidimensional irrep: "
                f"{irrep} has {count} components, not a multiple of {dimension}"
            )
    payload_identity = sonic_definition_identity_sha256(definition)
    reference = tuple(tuple(float(value) for value in row) for row in reference_block)
    singular_values = (
        ()
        if large_sparse_audit
        else tuple(float(value) for value in rank_diagnostics["singular_values"])
    )
    source_families = tuple(
        dict.fromkeys(
            f"{block_identifier}.Family.{_safe_token(gic.family)}" for gic in selected_gics
        )
    )
    return OnicCoordinateBlock(
        identifier=block_identifier,
        kind=kind,
        representation="NATURAL_INTERNAL",
        atom_indices_one_based=block_atoms,
        atom_indices_zero_based=tuple(atom - 1 for atom in block_atoms),
        reference_coordinates_angstrom=reference,
        reference_fingerprint_sha256=onic_reference_fingerprint(block_atoms, reference),
        source_family_identifiers=source_families,
        source_order=source_order,
        coordinate_identifiers=coordinate_ids,
        target_rank=expected_rank,
        source_count=expected_rank,
        nullity=0,
        linearity=linearity,
        rank_method=(
            str(rank_diagnostics.get("rank_method", NATURAL_INTERNAL_BLOCK_RANK_METHOD))
            if large_sparse_audit
            else NATURAL_INTERNAL_BLOCK_RANK_METHOD
        ),
        rank_absolute_tolerance=absolute_tolerance,
        rank_relative_tolerance=relative_tolerance,
        coefficient_operator=OnicMatrixRecord(
            rows=expected_rank,
            columns=expected_rank,
            storage="IDENTITY",
        ),
        local_symmetry_provenance=local_symmetry_provenance,
        exact_retained_group=definition.point_group,
        irrep_labels=irreps,
        degeneracy_groups=_natural_degeneracy_groups(
            block_identifier,
            coordinate_ids,
            irreps,
            source_count=expected_rank,
        ),
        component_gauge=NATURAL_INTERNAL_BLOCK_GAUGE,
        unit="MIXED",
        scaling_policy="FROZEN_GIC_NATIVE_UNITS",
        scale_factors=(1.0,) * expected_rank,
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
            messages=(
                f"PAYLOAD_ATOM_FRAME={payload_frame}",
                f"PAYLOAD_COORDINATE_COUNT={len(selected_indices)}",
                "COORDINATE_GENERATION=DELEGATED_TO_FROZEN_GIC",
                "GLOBAL_REDUCTION=FORBIDDEN",
            ),
        ),
        payload_schema=definition.contract_schema_version,
        payload_identity_sha256=payload_identity,
        provenance=(
            "SMITH_EXISTING_NATURAL_DOMAIN_FIRST_SONIC",
            f"PAYLOAD_IDENTITY_SHA256={payload_identity}",
            f"PAYLOAD_ATOM_FRAME={payload_frame}",
        ),
    )


def evaluate_natural_internal_block(
    block: OnicCoordinateBlock,
    definition: GICDefinition,
    coordinates_angstrom: np.ndarray | Sequence[Sequence[float]],
    *,
    parallel_workers: int = 1,
) -> NaturalInternalBlockEvaluation:
    """Evaluate a typed natural block through the canonical frozen-GIC APIs."""

    context = _natural_internal_evaluation_context(
        block,
        definition,
        coordinates_angstrom,
    )
    values = evaluate_gic_values_subset(
        definition,
        context.selected_indices,
        coordinates_angstrom=context.payload_coordinates,
    )
    payload_b = build_sparse_gic_b_matrix(
        definition,
        coordinates_angstrom=context.payload_coordinates,
        coordinate_indices=context.selected_indices,
        parallel_workers=parallel_workers,
    )
    payload_rows = payload_b.rows
    if context.payload_to_current_rotation is not None:
        from .bmatrix import SparseBRow

        rotation = context.payload_to_current_rotation
        payload_rows = tuple(
            SparseBRow.from_dense(
                (row.to_dense().reshape(-1, 3) @ rotation.T).reshape(-1)
            )
            for row in payload_rows
        )
    rows = (
        embed_local_sparse_rows(
            payload_rows,
            block.atom_indices_one_based,
            full_natoms=len(context.current_coordinates),
            payload_name="natural-internal",
        )
        if context.payload_frame == "LOCAL"
        else payload_rows
    )
    return NaturalInternalBlockEvaluation(
        coordinate_values=np.asarray(values, dtype=float),
        b_matrix=SparseBMatrix(
            rows=rows,
            column_count=context.current_coordinates.size,
            row_labels=block.coordinate_identifiers,
            backend="smith-natural-internal-frozen-gic.v1",
        ),
        payload_coordinate_indices=context.selected_indices,
        payload_identity_sha256=context.payload_identity_sha256,
    )


def evaluate_natural_internal_block_values(
    block: OnicCoordinateBlock,
    definition: GICDefinition,
    coordinates_angstrom: np.ndarray | Sequence[Sequence[float]],
) -> np.ndarray:
    """Evaluate only frozen natural-coordinate values, without building ``B``."""

    context = _natural_internal_evaluation_context(
        block,
        definition,
        coordinates_angstrom,
    )
    return np.asarray(
        evaluate_gic_values_subset(
            definition,
            context.selected_indices,
            coordinates_angstrom=context.payload_coordinates,
        ),
        dtype=float,
    )


def _natural_internal_evaluation_context(
    block: OnicCoordinateBlock,
    definition: GICDefinition,
    coordinates_angstrom: np.ndarray | Sequence[Sequence[float]],
) -> _NaturalInternalEvaluationContext:
    """Validate a frozen natural payload once for value or derivative evaluation."""

    if block.representation != "NATURAL_INTERNAL":
        raise ValueError("natural-internal evaluator received another representation")
    payload_identity = sonic_definition_identity_sha256(definition)
    if block.payload_schema != definition.contract_schema_version:
        raise ValueError("natural-internal payload schema does not match its typed block")
    if block.payload_identity_sha256 != payload_identity:
        raise ValueError("natural-internal payload checksum does not match its typed block")
    current = np.asarray(coordinates_angstrom, dtype=float)
    if current.ndim != 2 or current.shape[1] != 3 or not np.all(np.isfinite(current)):
        raise ValueError("natural-internal evaluation requires a finite natoms-by-3 geometry")
    if max(block.atom_indices_one_based) > len(current):
        raise ValueError("natural-internal block references atoms outside the current geometry")

    reference_full = frozen_payload_reference_coordinates(
        definition,
        payload_name="frozen GIC",
    )
    payload_atoms, payload_frame = payload_owned_atom_frame(
        block.atom_indices_one_based,
        payload_natoms=len(reference_full),
        payload_name="natural-internal",
    )
    payload_reference_block = (
        reference_full
        if payload_frame == "LOCAL"
        else reference_full[
            np.asarray([atom - 1 for atom in block.atom_indices_one_based], dtype=int)
        ]
    )
    if not np.allclose(
        payload_reference_block,
        np.asarray(block.reference_coordinates_angstrom, dtype=float),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("natural-internal payload reference does not match its typed block")
    payload_coordinates = (
        current[np.asarray([atom - 1 for atom in block.atom_indices_one_based], dtype=int)]
        if payload_frame == "LOCAL"
        else current
    )
    if payload_coordinates.shape != reference_full.shape:
        raise ValueError("current geometry does not match the frozen natural-internal payload")
    payload_to_current_rotation = None
    if payload_frame == "LOCAL" and any(
        primitive.function == "SOOP" for primitive in definition.primitives
    ):
        current_center = payload_coordinates.mean(axis=0)
        reference_center = reference_full.mean(axis=0)
        covariance = (
            (payload_coordinates - current_center).T
            @ (reference_full - reference_center)
        )
        left, _values, right = np.linalg.svd(covariance)
        rotation = left @ right
        if np.linalg.det(rotation) < 0.0:
            left[:, -1] *= -1.0
            rotation = left @ right
        payload_coordinates = (
            (payload_coordinates - current_center) @ rotation + reference_center
        )
        payload_to_current_rotation = rotation
    return _NaturalInternalEvaluationContext(
        current_coordinates=current,
        payload_coordinates=payload_coordinates,
        payload_frame=payload_frame,
        selected_indices=_payload_coordinate_indices(block, definition),
        payload_identity_sha256=payload_identity,
        payload_to_current_rotation=payload_to_current_rotation,
    )


def _owned_gic_indices(
    definition: GICDefinition,
    payload_atoms: tuple[int, ...],
) -> tuple[int, ...]:
    owned = set(payload_atoms)
    primitive_by_id = {primitive.identifier: primitive for primitive in definition.primitives}
    selected: list[int] = []
    for index, gic in enumerate(definition.gics):
        primitive_ids = tuple(
            primitive_id
            for primitive_id, _coefficient in (gic.coefficients or ((gic.primitive_id, 1.0),))
        )
        relations: list[str] = []
        for primitive_id in primitive_ids:
            primitive = primitive_by_id.get(primitive_id)
            if primitive is None:
                raise ValueError(
                    f"frozen GIC {gic.identifier} references missing primitive {primitive_id}"
                )
            dependencies = _primitive_dependency_atoms(primitive)
            if not dependencies:
                raise ValueError(
                    f"frozen GIC primitive {primitive.identifier} has no atom ownership"
                )
            if dependencies.issubset(owned):
                relations.append("OWNED")
            elif dependencies.isdisjoint(owned):
                relations.append("OUTSIDE")
            else:
                relations.append("BOUNDARY")
        if relations and all(relation == "OWNED" for relation in relations):
            selected.append(index)
        elif "OWNED" in relations:
            raise ValueError(
                "frozen GIC mixes molecular-internal and external ownership: "
                f"{gic.identifier} ({','.join(relations)})"
            )
    return tuple(selected)


def _primitive_dependency_atoms(primitive: GICPrimitive) -> set[int]:
    return {
        int(atom)
        for atoms in (
            primitive.atoms,
            primitive.ref_atoms,
            primitive.frame_atoms,
            primitive.ref_frame_atoms,
        )
        for atom in atoms
    }


def _natural_degeneracy_groups(
    block_identifier: str,
    coordinate_ids: tuple[str, ...],
    irreps: tuple[str, ...],
    *,
    source_count: int,
) -> tuple[OnicDegeneracyGroup, ...]:
    irrep_order = tuple(dict.fromkeys(irreps))
    groups: list[OnicDegeneracyGroup] = []
    for group_index, irrep in enumerate(irrep_order, start=1):
        group_coordinates = tuple(
            coordinate
            for coordinate, coordinate_irrep in zip(coordinate_ids, irreps, strict=True)
            if coordinate_irrep == irrep
        )
        group_prefix = f"{block_identifier}.{irrep_name_prefix(irrep)}Iso{group_index:03d}"
        groups.append(
            OnicDegeneracyGroup(
                identifier=group_prefix,
                irrep=irrep,
                coordinate_identifiers=group_coordinates,
                component_gauge=NATURAL_INTERNAL_BLOCK_GAUGE,
                projector=OnicMatrixRecord(
                    rows=source_count,
                    columns=source_count,
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
            raise ValueError(f"invalid natural-internal source identifier: {source}")
        identifier = source[len(prefix) :]
        if identifier not in index_by_identifier:
            raise ValueError(
                f"natural-internal source {identifier} is absent from its frozen payload"
            )
        indices.append(index_by_identifier[identifier])
    return tuple(indices)


def _linearity(coordinates: np.ndarray) -> str:
    if len(coordinates) == 1:
        return "MONATOMIC"
    return "LINEAR" if is_linear_geometry(coordinates) else "NONLINEAR"


def _safe_token(value: str) -> str:
    token = _SAFE_TOKEN_PATTERN.sub("_", str(value).strip())
    if not token or not token[0].isalpha():
        token = f"Family_{token}"
    return token


__all__ = [
    "NATURAL_INTERNAL_BLOCK_ABSOLUTE_RANK_TOLERANCE",
    "NATURAL_INTERNAL_BLOCK_GAUGE",
    "NATURAL_INTERNAL_BLOCK_RANK_METHOD",
    "NATURAL_INTERNAL_BLOCK_RELATIVE_RANK_TOLERANCE",
    "NATURAL_INTERNAL_BLOCK_SUPPORT_TOLERANCE",
    "NaturalInternalBlockEvaluation",
    "build_natural_internal_block",
    "evaluate_natural_internal_block",
    "evaluate_natural_internal_block_values",
]
