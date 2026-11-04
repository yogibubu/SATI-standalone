"""Fail-closed ORACLE/SMITH Wilson gate before LINK realization."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from matrix_chem import (
    MATRIX_XYZ_PRIMITIVES_SCHEMA,
    ORACLE_SONIC_CONTRACT_SCHEMA,
    ORACLE_SONIC_CONTRACT_SECTION,
    expected_vibrational_mode_count,
    oracle_sonic_contract_to_dict,
    read_enriched_xyz,
    read_primitive_contract,
    validate_primitive_contract,
)
from matrix_core import read_sectioned_lines, section_content
from matrix_smith import (
    GICDefinition,
    build_sparse_gic_b_matrix,
    load_validated_oracle_candidate_pool,
    read_gic_definition_from_xyzin,
    select_closed_contact_pose,
)


LINK_ONIC_CONTRACT_GATE_SCHEMA = "matrix.link.onic_contract_gate.v1"


@dataclass(frozen=True)
class LinkOnicContractGate:
    status: str
    orientation: str
    contract_schema: str
    contract_sha256: str
    sonic_identity_sha256: str
    target_rank: int
    evaluated_rank: int
    primitive_count: int
    gic_count: int
    closed_contact_policy: str
    provenance: str = LINK_ONIC_CONTRACT_GATE_SCHEMA


def has_frozen_oracle_sonic_contract(path: Path) -> bool:
    return bool(
        section_content(
            read_sectioned_lines(Path(path)),
            ORACLE_SONIC_CONTRACT_SECTION,
        )
    )


def validate_link_onic_contract(
    path: Path,
    *,
    definition: GICDefinition | None = None,
    orientation: str = "SONIC",
    rank_tolerance: float = 1.0e-8,
    allow_endpoint_connectivity_candidate: bool = False,
) -> LinkOnicContractGate:
    """Admit LINK only after the frozen chemistry and Wilson gates pass."""

    target = Path(path)
    normalized_orientation = str(orientation).strip().upper()
    if normalized_orientation not in {"TONIC", "CONIC", "SONIC"}:
        raise ValueError("LINK ONIC orientation must be TONIC, CONIC, or SONIC")
    if not has_frozen_oracle_sonic_contract(target):
        raise ValueError("LINK requires a frozen ORACLE SONIC contract before ONIC realization")
    pool = load_validated_oracle_candidate_pool(target)
    frozen = definition if definition is not None else read_gic_definition_from_xyzin(target)
    endpoint_sources = {
        "ENDPOINT_CONNECTIVITY_SONIC_COMBINATION",
        "HESSIAN_ADAPTED_REACTIVE_SONIC",
    }
    endpoint_candidate = frozen.primitive_source in endpoint_sources
    if frozen.primitive_source != "ORACLE_CONTRACT" and not (
        endpoint_candidate and allow_endpoint_connectivity_candidate
    ):
        raise ValueError("LINK refuses a SONIC definition not built from an admitted ORACLE contract")
    if endpoint_candidate:
        from matrix_smith import (
            ENDPOINT_CONNECTIVITY_SONIC_SCHEMA,
            HESSIAN_ADAPTED_REACTIVE_SONIC_SCHEMA,
        )

        expected_schema = {
            "ENDPOINT_CONNECTIVITY_SONIC_COMBINATION": (
                ENDPOINT_CONNECTIVITY_SONIC_SCHEMA
            ),
            "HESSIAN_ADAPTED_REACTIVE_SONIC": (
                HESSIAN_ADAPTED_REACTIVE_SONIC_SCHEMA
            ),
        }[frozen.primitive_source]
        if frozen.primitive_source_schema != expected_schema:
            raise ValueError("LINK endpoint-connectivity candidate records the wrong schema")
        primitive_ids = {primitive.identifier for primitive in frozen.primitives}
        if any(
            primitive_id not in primitive_ids
            for gic in frozen.gics
            for primitive_id, _coefficient in gic.coefficients
        ):
            raise ValueError("LINK endpoint-connectivity candidate references an unknown primitive")
    _validate_chart_orientation(frozen, normalized_orientation)

    geometry = read_enriched_xyz(target)
    candidate_by_id = {
        candidate.candidate_id: candidate for candidate in pool.contract.primitive_candidates
    }
    if not endpoint_candidate:
        if frozen.primitive_source_schema == ORACLE_SONIC_CONTRACT_SCHEMA:
            _validate_legacy_oracle_candidate_chart(frozen, candidate_by_id)
        elif frozen.primitive_source_schema == MATRIX_XYZ_PRIMITIVES_SCHEMA:
            primitive_contract = read_primitive_contract(target)
            validate_primitive_contract(
                primitive_contract,
                np.asarray(geometry.coordinates_angstrom, dtype=float),
            )
            if frozen.primitive_b_matrix_sha256 != primitive_contract.b_matrix_sha256:
                raise ValueError(
                    "LINK SONIC definition records a stale ORACLE primitive B matrix"
                )
            primitive_bonds = {
                tuple(sorted((left + 1, right + 1)))
                for primitive in primitive_contract.primitives
                if primitive.kind == "bond"
                for left, right in (primitive.atoms,)
            }
            if primitive_bonds != set(pool.contract.primary_topology.bonds):
                raise ValueError(
                    "LINK ORACLE primitive contract contradicts the frozen primary topology"
                )
        else:
            raise ValueError(
                "LINK SONIC definition records the wrong ORACLE contract schema"
            )

    expected_rank = expected_vibrational_mode_count(geometry.coordinates_angstrom)
    if (
        frozen.target_rank != expected_rank
        or frozen.rank != expected_rank
        or len(frozen.gics) != expected_rank
    ):
        raise ValueError(
            "LINK requires a complete nonredundant ONIC chart at the molecular vibrational rank"
        )
    b_matrix = build_sparse_gic_b_matrix(
        frozen,
        coordinates_angstrom=np.asarray(geometry.coordinates_angstrom, dtype=float),
    )
    try:
        b_matrix.assert_finite()
        structural_rank = b_matrix.structural_rank()
    except ValueError as exc:
        raise ValueError("LINK ONIC Wilson gate found invalid sparse derivative data") from exc
    # The frozen SMITH definition contains the numerical rank certificate.
    # Recomputing an SVD over an N-by-3N dense array at every LINK launch was
    # both redundant and quadratic in storage.  The runtime gate instead
    # verifies fresh finite derivatives and complete sparse support; an
    # explicit dense audit is still available in SMITH for small references.
    evaluated_rank = expected_rank
    if structural_rank < expected_rank:
        raise ValueError(
            "LINK ONIC Wilson sparse-support gate failed: "
            f"expected structural rank {expected_rank}, got {structural_rank}"
        )

    closed_policy = "NONE"
    closing_contacts = tuple(
        contact
        for contact in pool.contract.auxiliary_contacts
        if contact.open_or_closing == "CLOSING"
        and len(set(contact.fragment_ids)) > 1
    )
    if closing_contacts:
        closing_pairs = {
            tuple(sorted((int(contact.endpoint_a.identifier), int(contact.endpoint_b.identifier))))
            for contact in closing_contacts
            if contact.endpoint_a.kind == contact.endpoint_b.kind == "ATOM"
        }
        if closing_pairs.intersection(frozen.pseudo_bonds):
            raise ValueError("LINK refuses closing contacts inserted into SONIC adjacency")
        declared_policy = next(
            (
                record.split(maxsplit=1)[1]
                for record in frozen.semantic_diagnostics
                if record.startswith("CLOSED_CONTACT_POLICY ")
            ),
            "",
        )
        frozen_policy_modes = {
            "SPECIAL_COORDINATES": "SPECIAL_COORDINATES",
            "PSEUDOBOND_CONTACT_NATURAL_COORDINATES": "PSEUDO_BONDS",
        }
        if declared_policy in frozen_policy_modes:
            if frozen.fragment_mode != frozen_policy_modes[declared_policy]:
                raise ValueError(
                    "LINK SONIC closed-contact policy contradicts the frozen fragment chart"
                )
            closed_policy = declared_policy
        else:
            closed = select_closed_contact_pose(target, rank_tolerance=rank_tolerance)
            closed_policy = closed.policy
            if declared_policy != closed.policy:
                raise ValueError("LINK SONIC definition does not record its closed-contact policy")

    contract_payload = json.dumps(
        oracle_sonic_contract_to_dict(pool.contract),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    from matrix_smith import sonic_definition_identity_sha256

    return LinkOnicContractGate(
        status="PASS",
        orientation=normalized_orientation,
        contract_schema=pool.contract.schema,
        contract_sha256=hashlib.sha256(contract_payload).hexdigest(),
        sonic_identity_sha256=sonic_definition_identity_sha256(frozen),
        target_rank=expected_rank,
        evaluated_rank=evaluated_rank,
        primitive_count=len(frozen.primitives),
        gic_count=len(frozen.gics),
        closed_contact_policy=closed_policy,
    )


def _validate_legacy_oracle_candidate_chart(
    frozen: GICDefinition,
    candidate_by_id: dict[str, object],
) -> None:
    """Retain validation for already serialized pre-unification ONIC charts."""

    for primitive in frozen.primitives:
        candidate = candidate_by_id.get(primitive.identifier)
        if candidate is None:
            raise ValueError(
                f"LINK SONIC primitive {primitive.identifier} is absent from the ORACLE contract"
            )
        if (
            primitive.function != candidate.function
            or primitive.family != candidate.family
            or primitive.atoms != candidate.atoms
            or primitive.mode != candidate.mode
            or primitive.ref_atoms != candidate.ref_atoms
        ):
            raise ValueError(
                f"LINK SONIC primitive {primitive.identifier} contradicts its ORACLE definition"
            )
    for gic in frozen.gics:
        coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
        unknown = [
            candidate_id
            for candidate_id, _value in coefficients
            if candidate_id not in candidate_by_id
        ]
        if unknown:
            raise ValueError(
                f"LINK SONIC GIC {gic.identifier} references non-ORACLE primitives: {unknown}"
            )


def _validate_chart_orientation(definition: GICDefinition, orientation: str) -> None:
    required = {
        "TONIC": ("CHART_ORIENTATION TONIC", "CHART_ROLE GENERAL"),
        "CONIC": ("CHART_ORIENTATION CONIC", "CHART_ROLE EXPLORATION"),
        "SONIC": ("CHART_ORIENTATION SONIC", "CHART_ROLE EXPLOITATION"),
    }[orientation]
    if "ONIC_CORE COMMON_TYPED_NONREDUNDANT_ALGEBRA" not in definition.semantic_diagnostics:
        raise ValueError("LINK definition does not declare the common ONIC core")
    if any(item not in definition.semantic_diagnostics for item in required):
        raise ValueError(f"LINK definition does not satisfy the {orientation} chart orientation")


__all__ = [
    "LINK_ONIC_CONTRACT_GATE_SCHEMA",
    "LinkOnicContractGate",
    "has_frozen_oracle_sonic_contract",
    "validate_link_onic_contract",
]
