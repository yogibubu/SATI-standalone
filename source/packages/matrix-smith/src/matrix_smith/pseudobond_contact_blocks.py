"""Typed pseudo-bond contact blocks backed by frozen natural internals."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np

from .block_payload import (
    compact_sparse_b_matrix,
    frozen_payload_reference_coordinates,
    normalized_owned_atoms,
    payload_owned_atom_frame,
)
from .coordinate_diagnostics import sonic_condition_diagnostics
from .definition import sonic_definition_identity_sha256
from .evaluation import build_sparse_gic_b_matrix
from .models import GICDefinition
from .natural_internal_blocks import (
    NaturalInternalBlockEvaluation,
    _natural_degeneracy_groups,
    evaluate_natural_internal_block,
    evaluate_natural_internal_block_values,
)
from .onic_blocks import (
    OnicBlockDiagnostics,
    OnicCoordinateBlock,
    OnicMatrixRecord,
    onic_reference_fingerprint,
)
from .policy import RANK_TOLERANCE


def build_pseudobond_contact_block(
    definition: GICDefinition,
    *,
    reference_block: OnicCoordinateBlock,
    moving_block: OnicCoordinateBlock,
    block_identifier: str = "CONTACT1",
    protected: bool = True,
    active: bool = True,
    observable: bool = False,
    rank_absolute_tolerance: float = 1.0e-10,
    rank_relative_tolerance: float = RANK_TOLERANCE,
    support_tolerance: float = 1.0e-12,
) -> OnicCoordinateBlock:
    """Wrap six frozen pseudo-bond/pseudo-cycle rows as a relative pose.

    The payload owns the union of both fragments but selects only six
    interfragment rows.  It is therefore audited as a relative chart and is
    never forced through a fictitious molecular ``3N-6`` contract.
    """

    if reference_block.kind == "RELATIVE_POSE" or moving_block.kind == "RELATIVE_POSE":
        raise ValueError("pseudo-bond contact dependencies must own atoms")
    if set(reference_block.atom_indices_one_based).intersection(
        moving_block.atom_indices_one_based
    ):
        raise ValueError("pseudo-bond contact dependencies must be disjoint")
    dependency_atoms = normalized_owned_atoms(
        (*reference_block.atom_indices_one_based, *moving_block.atom_indices_one_based),
        block_name="pseudo-bond contact",
    )
    reference_full = frozen_payload_reference_coordinates(
        definition,
        payload_name="pseudo-bond contact frozen GIC",
    )
    payload_atoms, payload_frame = payload_owned_atom_frame(
        dependency_atoms,
        payload_natoms=len(reference_full),
        payload_name="pseudo-bond contact",
        explicit_local_order=True,
    )
    if len(definition.gics) != 6:
        raise ValueError(
            "pseudo-bond contact payload must contain exactly six independent relative rows"
        )
    sparse_b = build_sparse_gic_b_matrix(
        definition,
        coordinates_angstrom=definition.reference_coordinates_angstrom,
    )
    compact_b, outside_support = compact_sparse_b_matrix(
        sparse_b,
        payload_atoms=payload_atoms,
    )
    if outside_support > float(support_tolerance):
        raise ValueError("pseudo-bond contact payload has support outside its dependency union")
    rank_diagnostics = sonic_condition_diagnostics(
        compact_b,
        tolerance=float(rank_relative_tolerance),
        absolute_tolerance=float(rank_absolute_tolerance),
    )
    if int(rank_diagnostics["rank"]) != 6:
        raise ValueError(
            "pseudo-bond contact payload is not a complete relative chart: "
            f"rank={rank_diagnostics['rank']}, required=6"
        )
    reference = (
        reference_full
        if payload_frame == "LOCAL"
        else reference_full[np.asarray([atom - 1 for atom in dependency_atoms], dtype=int)]
    )
    coordinate_ids = tuple(f"{block_identifier}.{gic.name}" for gic in definition.gics)
    source_order = tuple(f"{block_identifier}.{gic.identifier}" for gic in definition.gics)
    irreps = tuple(str(gic.irrep) for gic in definition.gics)
    payload_identity = sonic_definition_identity_sha256(definition)
    reference_tuple = tuple(tuple(float(value) for value in row) for row in reference)
    families = tuple(dict.fromkeys(gic.family for gic in definition.gics))
    return OnicCoordinateBlock(
        identifier=block_identifier,
        kind="RELATIVE_POSE",
        representation="PSEUDO_BOND_CONTACT",
        atom_indices_one_based=dependency_atoms,
        atom_indices_zero_based=tuple(atom - 1 for atom in dependency_atoms),
        reference_coordinates_angstrom=reference_tuple,
        reference_fingerprint_sha256=onic_reference_fingerprint(
            dependency_atoms,
            reference_tuple,
        ),
        source_family_identifiers=tuple(
            f"{block_identifier}.Family{index}" for index, _family in enumerate(families, start=1)
        ),
        source_order=source_order,
        coordinate_identifiers=coordinate_ids,
        target_rank=6,
        source_count=6,
        nullity=0,
        linearity="NOT_APPLICABLE",
        rank_method="FROZEN_PSEUDOBOND_GIC_SPARSE_B_SVD_AUDIT",
        rank_absolute_tolerance=float(rank_absolute_tolerance),
        rank_relative_tolerance=float(rank_relative_tolerance),
        coefficient_operator=OnicMatrixRecord(rows=6, columns=6, storage="IDENTITY"),
        local_symmetry_provenance="FROZEN_ORACLE_SMITH_PSEUDOBOND_CONTACT",
        exact_retained_group=definition.point_group,
        irrep_labels=irreps,
        degeneracy_groups=_natural_degeneracy_groups(
            block_identifier,
            coordinate_ids,
            irreps,
            source_count=6,
        ),
        component_gauge="FROZEN_PSEUDOBOND_NATURAL_INTERNAL_GAUGE",
        unit="MIXED",
        scaling_policy="FROZEN_GIC_NATIVE_UNITS",
        scale_factors=(1.0,) * 6,
        protected=protected,
        active=active,
        observable=observable,
        analytic_derivative_status="ANALYTIC_FIRST_ORDER",
        second_derivative_status="GENERAL_SPARSE_B_PRIME",
        diagnostics=OnicBlockDiagnostics(
            spectrum=tuple(float(value) for value in rank_diagnostics["singular_values"][:6]),
            condition_number=float(rank_diagnostics["condition_number"]),
            row_space_residual=outside_support,
            messages=(
                f"PAYLOAD_ATOM_FRAME={payload_frame}",
                "COORDINATE_EVALUATION=DELEGATED_TO_FROZEN_GIC",
                "CONTACT_TOPOLOGY=OWNED_BY_ORACLE_SMITH",
                "GLOBAL_REDUCTION=FORBIDDEN",
            ),
        ),
        payload_schema=definition.contract_schema_version,
        payload_identity_sha256=payload_identity,
        reference_block_id=reference_block.identifier,
        moving_block_id=moving_block.identifier,
        provenance=(
            "SMITH_FROZEN_PSEUDOBOND_CONTACT_CHART",
            f"PAYLOAD_IDENTITY_SHA256={payload_identity}",
            "NO_MOLECULE_SPECIFIC_CONTACT_INFERENCE",
        ),
    )


def evaluate_pseudobond_contact_block(
    block: OnicCoordinateBlock,
    definition: GICDefinition,
    coordinates_angstrom: Sequence[Sequence[float]],
    *,
    reference_block: OnicCoordinateBlock,
    moving_block: OnicCoordinateBlock,
    parallel_workers: int = 1,
) -> NaturalInternalBlockEvaluation:
    """Evaluate a frozen pseudo-bond contact through the natural GIC kernel."""

    natural_view = _pseudobond_natural_view(
        block,
        reference_block=reference_block,
        moving_block=moving_block,
    )
    return evaluate_natural_internal_block(
        natural_view,
        definition,
        coordinates_angstrom,
        parallel_workers=parallel_workers,
    )


def evaluate_pseudobond_contact_block_values(
    block: OnicCoordinateBlock,
    definition: GICDefinition,
    coordinates_angstrom: Sequence[Sequence[float]],
    *,
    reference_block: OnicCoordinateBlock,
    moving_block: OnicCoordinateBlock,
) -> np.ndarray:
    """Evaluate only pseudo-bond contact values through the natural kernel."""

    natural_view = _pseudobond_natural_view(
        block,
        reference_block=reference_block,
        moving_block=moving_block,
    )
    return evaluate_natural_internal_block_values(
        natural_view,
        definition,
        coordinates_angstrom,
    )


def _pseudobond_natural_view(
    block: OnicCoordinateBlock,
    *,
    reference_block: OnicCoordinateBlock,
    moving_block: OnicCoordinateBlock,
) -> OnicCoordinateBlock:
    """Validate dependencies and expose the canonical natural-kernel view."""

    if block.kind != "RELATIVE_POSE" or block.representation != "PSEUDO_BOND_CONTACT":
        raise ValueError("pseudo-bond contact evaluator received another block type")
    expected_atoms = tuple(
        dict.fromkeys(
            (*reference_block.atom_indices_one_based, *moving_block.atom_indices_one_based)
        )
    )
    if block.reference_block_id != reference_block.identifier:
        raise ValueError("pseudo-bond contact reference dependency does not match")
    if block.moving_block_id != moving_block.identifier:
        raise ValueError("pseudo-bond contact moving dependency does not match")
    if block.atom_indices_one_based != expected_atoms:
        raise ValueError("pseudo-bond contact atom union does not match its dependencies")
    return replace(
        block,
        kind="MOLECULE_INTERNAL",
        representation="NATURAL_INTERNAL",
        reference_block_id="",
        moving_block_id="",
    )


__all__ = [
    "build_pseudobond_contact_block",
    "evaluate_pseudobond_contact_block",
    "evaluate_pseudobond_contact_block_values",
]
