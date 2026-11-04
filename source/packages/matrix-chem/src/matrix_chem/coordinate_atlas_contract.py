"""Frozen ORACLE scientific-atlas prescription consumed by SMITH.

The transport schema lives in :mod:`matrix_chem` so ORACLE and SMITH can
share it without reversing package dependencies.  Scientific ownership
remains exclusively with ORACLE.  The contract deliberately keeps chemical
semantics, graph effects, coordinate roles, and numerical compatibility in
separate fields; validators reject any attempt to infer one from another.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from matrix_core import read_sectioned_lines, replace_section, section_content

from .coordinate_library import COORDINATE_LIBRARY_SCHEMA, coordinate_library_manifest
from .salc_library import (
    COMPLETION_EXACT_RANK,
    SALC_LIBRARY_SCHEMA,
    SALC_NONE,
    salc_algorithm,
    salc_library_manifest,
)


ORACLE_COORDINATE_ATLAS_SCHEMA = "matrix.oracle.coordinate_atlas.v3"
ORACLE_COORDINATE_ATLAS_SECTION = "ORACLE_COORDINATE_ATLAS"
ORACLE_COORDINATE_ATLAS_OWNER = "ORACLE"

ATLAS_TASK_MINIMUM = "MINIMUM"
ATLAS_TASK_EXPLORATION = "EXPLORATION"
ATLAS_TASK_TRANSITION_STATE = "TRANSITION_STATE"
ATLAS_TASK_REGIMES = frozenset(
    {ATLAS_TASK_MINIMUM, ATLAS_TASK_EXPLORATION, ATLAS_TASK_TRANSITION_STATE}
)

GRAPH_ROLE_PRIMARY = "PRIMARY"
GRAPH_ROLE_OPEN = "OPEN"
GRAPH_ROLE_CLOSING = "CLOSING"
GRAPH_ROLE_REACTIVE_SUPPORT = "REACTIVE_SUPPORT"
GRAPH_ROLE_NONE = "NONE"
GRAPH_ROLES = frozenset(
    {
        GRAPH_ROLE_PRIMARY,
        GRAPH_ROLE_OPEN,
        GRAPH_ROLE_CLOSING,
        GRAPH_ROLE_REACTIVE_SUPPORT,
        GRAPH_ROLE_NONE,
    }
)

PSEUDOBOND_FORBIDDEN = "FORBIDDEN"
PSEUDOBOND_ALLOWED = "ALLOWED"
PSEUDOBOND_REQUIRED = "REQUIRED"
PSEUDOBOND_POLICIES = frozenset(
    {PSEUDOBOND_FORBIDDEN, PSEUDOBOND_ALLOWED, PSEUDOBOND_REQUIRED}
)

COORDINATE_INTERNAL_VALENCE = "INTERNAL_VALENCE"
COORDINATE_PHYSICAL_CONTACT_DISTANCE = "PHYSICAL_CONTACT_DISTANCE"
COORDINATE_REACTION_DISTANCE = "REACTION_DISTANCE"
COORDINATE_REACTION_DISTANCE_ONLY = "REACTION_DISTANCE_ONLY"
COORDINATE_FRAGMENT_TRANSLATION = "FRAGMENT_TRANSLATION"
COORDINATE_FRAGMENT_ORIENTATION = "FRAGMENT_ORIENTATION"
COORDINATE_FRAGMENT_POSE_SUPPORT = "FRAGMENT_POSE_SUPPORT"
COORDINATE_TS_SUPPORT = "TS_SUPPORT"
COORDINATE_OBSERVABLE_ONLY = "OBSERVABLE_ONLY"
COORDINATE_ROLES = frozenset(
    {
        COORDINATE_INTERNAL_VALENCE,
        COORDINATE_PHYSICAL_CONTACT_DISTANCE,
        COORDINATE_REACTION_DISTANCE,
        COORDINATE_REACTION_DISTANCE_ONLY,
        COORDINATE_FRAGMENT_TRANSLATION,
        COORDINATE_FRAGMENT_ORIENTATION,
        COORDINATE_FRAGMENT_POSE_SUPPORT,
        COORDINATE_TS_SUPPORT,
        COORDINATE_OBSERVABLE_ONLY,
    }
)

CANDIDATE_REQUIRED = "REQUIRED"
CANDIDATE_OPTIONAL = "OPTIONAL"
CANDIDATE_FORBIDDEN = "FORBIDDEN"
CANDIDATE_REQUIREMENTS = frozenset(
    {CANDIDATE_REQUIRED, CANDIDATE_OPTIONAL, CANDIDATE_FORBIDDEN}
)

BODY_POINT = "POINT"
BODY_LINEAR = "LINEAR"
BODY_NONLINEAR = "NONLINEAR"
BODY_DIMENSIONS = frozenset({BODY_POINT, BODY_LINEAR, BODY_NONLINEAR})
BODY_DOF = {
    BODY_POINT: (3, 0),
    BODY_LINEAR: (3, 2),
    BODY_NONLINEAR: (3, 3),
}

MINIMUM_PSEUDOBOND_CONTACT_KINDS = frozenset(
    {
        "HYDROGEN_BOND",
        "DATIVE_CONTACT",
        "STRUCTURAL_LIGAND_CONTACT",
        "TETREL_BOND",
        "PNICTOGEN_BOND",
        "CHALCOGEN_BOND",
        "HALOGEN_BOND",
    }
)

ATLAS_FALLBACK_ORDER = (
    "ORACLE_CANDIDATES",
    "ANALYTIC_SALC",
    "EXACT_RANK_SELECTION",
    "DECLARED_MINIMAL_COMPLETION",
    "DECLARED_CHART_TRANSITION",
    "FAIL_UNSUPPORTED_ATLAS_CELL",
)


class OracleCoordinateAtlasError(ValueError):
    """Raised when a frozen scientific-atlas prescription is inconsistent."""


@dataclass(frozen=True)
class AtlasInteractionDecision:
    decision_id: str
    interaction_id: str
    semantic_kind: str
    endpoint_a: tuple[str, str]
    endpoint_b: tuple[str, str]
    fragment_ids: tuple[str, ...]
    graph_role: str
    delta_beta1: int
    coordinate_role: str
    pseudobond_policy: str
    primitive_candidate_ids: tuple[str, ...]
    rule_id: str
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class AtlasBodyPrescription:
    body_id: str
    atoms: tuple[int, ...]
    dimension: str
    translation_dofs: int
    orientation_dofs: int
    pose_chart: str
    pose_role: str
    rule_id: str


@dataclass(frozen=True)
class AtlasCandidatePrescription:
    candidate_id: str
    scientific_role: str
    requirement: str
    mixing_block: str
    completion_block: str
    source_decision_id: str
    rule_id: str


@dataclass(frozen=True)
class AtlasFamilyCompatibility:
    block_id: str
    task_regimes: tuple[str, ...]
    families: tuple[str, ...]
    salc_algorithm_id: str
    completion_algorithm_id: str
    substitutions: bool
    require_same_domain: bool
    require_same_irrep: bool
    condition_trigger: float | None
    minimum_relative_gain: float | None
    rule_id: str


@dataclass(frozen=True)
class AtlasLocalDomainPrescription:
    """ORACLE-owned local geometry domain for one stable MINIMUM contact."""

    domain_id: str
    interaction_id: str
    reference_fragment_id: str
    moving_fragment_id: str
    reference_endpoint_atom: int
    moving_endpoint_atom: int
    moving_atoms: tuple[int, ...]
    radial_step_angstrom: float
    axial_half_width_degrees: float
    axial_step_degrees: float
    tilt_degrees: float
    sampling_policy: str
    topology_guard: str
    contact_guard: str
    rule_id: str


@dataclass(frozen=True)
class AtlasReactiveZonePrescription:
    """ORACLE-owned primitive exclusions for one local TS reaction zone."""

    zone_id: str
    atoms: tuple[int, ...]
    source_interaction_ids: tuple[str, ...]
    excluded_primitive_functions: tuple[str, ...]
    rule_id: str


@dataclass(frozen=True)
class OracleCoordinateAtlasContract:
    schema: str
    owner: str
    policy_id: str
    policy_version: str
    policy_sha256: str
    coordinate_library_schema: str
    coordinate_library_sha256: str
    salc_library_schema: str
    salc_library_sha256: str
    task_regime: str
    topology_hash: str
    interactions: tuple[AtlasInteractionDecision, ...]
    bodies: tuple[AtlasBodyPrescription, ...]
    candidates: tuple[AtlasCandidatePrescription, ...]
    family_compatibility: tuple[AtlasFamilyCompatibility, ...]
    local_domains: tuple[AtlasLocalDomainPrescription, ...]
    reactive_zones: tuple[AtlasReactiveZonePrescription, ...]
    fallback_order: tuple[str, ...]
    provenance: str
    # MINIMUM default for pairs not covered by explicit interactions: pose
    # only, observable-only, no pseudobond. Empty means legacy explicit mode.
    implicit_unclassified_fragments: tuple[str, ...] = ()


def validate_oracle_coordinate_atlas_contract(
    contract: OracleCoordinateAtlasContract,
) -> None:
    """Validate the scientific prescription without performing perception."""

    if contract.schema != ORACLE_COORDINATE_ATLAS_SCHEMA:
        raise OracleCoordinateAtlasError(
            f"unsupported ORACLE coordinate-atlas schema: {contract.schema}"
        )
    if contract.owner != ORACLE_COORDINATE_ATLAS_OWNER:
        raise OracleCoordinateAtlasError("ORACLE must be the sole atlas owner")
    if contract.task_regime not in ATLAS_TASK_REGIMES:
        raise OracleCoordinateAtlasError(
            f"unsupported atlas task regime: {contract.task_regime}"
        )
    if not all(
        (
            contract.policy_id,
            contract.policy_version,
            contract.policy_sha256,
            contract.coordinate_library_schema,
            contract.coordinate_library_sha256,
            contract.salc_library_schema,
            contract.salc_library_sha256,
            contract.topology_hash,
            contract.provenance,
        )
    ):
        raise OracleCoordinateAtlasError("atlas identity and provenance must be complete")
    coordinate_manifest = coordinate_library_manifest()
    salc_manifest = salc_library_manifest()
    if (
        contract.coordinate_library_schema != COORDINATE_LIBRARY_SCHEMA
        or contract.coordinate_library_sha256 != coordinate_manifest["sha256"]
    ):
        raise OracleCoordinateAtlasError("atlas coordinate-library identity is stale")
    if (
        contract.salc_library_schema != SALC_LIBRARY_SCHEMA
        or contract.salc_library_sha256 != salc_manifest["sha256"]
    ):
        raise OracleCoordinateAtlasError("atlas SALC-library identity is stale")
    if len(contract.policy_sha256) != 64:
        raise OracleCoordinateAtlasError("atlas policy SHA256 must contain 64 hexadecimal digits")
    try:
        int(contract.policy_sha256, 16)
    except ValueError as exc:
        raise OracleCoordinateAtlasError("atlas policy SHA256 is not hexadecimal") from exc
    if contract.fallback_order != ATLAS_FALLBACK_ORDER:
        raise OracleCoordinateAtlasError(
            "atlas fallback order must be the canonical fail-closed sequence"
        )

    _validate_unique((item.decision_id for item in contract.interactions), "decision id")
    if not isinstance(contract.implicit_unclassified_fragments, tuple) or any(
        not isinstance(item, str) or not item for item in contract.implicit_unclassified_fragments
    ):
        raise OracleCoordinateAtlasError("implicit fragment ids must be a tuple of nonempty strings")
    _validate_unique(contract.implicit_unclassified_fragments, "implicit fragment id")
    if contract.implicit_unclassified_fragments:
        if contract.task_regime != ATLAS_TASK_MINIMUM:
            raise OracleCoordinateAtlasError("implicit unclassified pairs require MINIMUM")
        members = set(contract.implicit_unclassified_fragments)
        if any(not set(item.fragment_ids).issubset(members) for item in contract.interactions):
            raise OracleCoordinateAtlasError("interaction outside implicit fragment domain")
        if any(item.body_id not in members for item in contract.bodies):
            raise OracleCoordinateAtlasError("body outside implicit fragment domain")
    _validate_unique((item.interaction_id for item in contract.interactions), "interaction id")
    _validate_unique((item.body_id for item in contract.bodies), "body id")
    _validate_unique((item.candidate_id for item in contract.candidates), "candidate id")
    _validate_unique((item.block_id for item in contract.family_compatibility), "family block")
    _validate_unique((item.domain_id for item in contract.local_domains), "local domain id")
    _validate_unique((item.zone_id for item in contract.reactive_zones), "reactive zone id")

    decision_ids = {item.decision_id for item in contract.interactions}
    decisions_by_id = {
        item.decision_id: item for item in contract.interactions
    }
    candidate_ids = {item.candidate_id for item in contract.candidates}
    block_ids = {item.block_id for item in contract.family_compatibility}
    for item in contract.interactions:
        _validate_interaction(item, contract.task_regime)
        missing = set(item.primitive_candidate_ids) - candidate_ids
        if missing:
            raise OracleCoordinateAtlasError(
                f"interaction {item.interaction_id} references unknown candidates: "
                + ",".join(sorted(missing))
            )
    for item in contract.bodies:
        _validate_body(item)
    if contract.bodies:
        references = tuple(item for item in contract.bodies if item.pose_role == "REFERENCE")
        if len(references) != 1:
            raise OracleCoordinateAtlasError(
                "an atlas body set must contain exactly one REFERENCE body"
            )
    for item in contract.candidates:
        if item.scientific_role not in COORDINATE_ROLES:
            raise OracleCoordinateAtlasError(
                f"unsupported scientific coordinate role: {item.scientific_role}"
            )
        if item.requirement not in CANDIDATE_REQUIREMENTS:
            raise OracleCoordinateAtlasError(
                f"unsupported candidate requirement: {item.requirement}"
            )
        if item.mixing_block not in block_ids or item.completion_block not in block_ids:
            raise OracleCoordinateAtlasError(
                f"candidate {item.candidate_id} references an unknown compatibility block"
            )
        if item.source_decision_id and item.source_decision_id not in decision_ids:
            raise OracleCoordinateAtlasError(
                f"candidate {item.candidate_id} references an unknown atlas decision"
            )
        if (
            contract.task_regime == ATLAS_TASK_MINIMUM
            and item.source_decision_id
        ):
            source = decisions_by_id[item.source_decision_id]
            if source.pseudobond_policy == PSEUDOBOND_REQUIRED:
                if (
                    item.requirement != CANDIDATE_REQUIRED
                    or item.scientific_role
                    != COORDINATE_PHYSICAL_CONTACT_DISTANCE
                ):
                    raise OracleCoordinateAtlasError(
                        "a stable OPEN MINIMUM contact must be a required physical "
                        "pseudobond candidate"
                    )
            elif (
                item.requirement != CANDIDATE_FORBIDDEN
                or item.scientific_role != COORDINATE_OBSERVABLE_ONLY
            ):
                raise OracleCoordinateAtlasError(
                    "a MINIMUM contact outside the pseudobond gate must remain a "
                    "forbidden observable candidate"
                )
        if (
            contract.task_regime == ATLAS_TASK_EXPLORATION
            and item.source_decision_id
            and (
                item.requirement != CANDIDATE_FORBIDDEN
                or item.scientific_role != COORDINATE_OBSERVABLE_ONLY
            )
        ):
            raise OracleCoordinateAtlasError(
                "an EXPLORATION contact must remain a forbidden observable candidate"
            )
    minimum_has_required_pseudobond = (
        contract.task_regime == ATLAS_TASK_MINIMUM
        and any(
            item.pseudobond_policy == PSEUDOBOND_REQUIRED
            for item in contract.interactions
        )
    )
    if minimum_has_required_pseudobond:
        # Pseudobonds constrain the contact graph, while fragment-pose
        # coordinates complete the rigid translations/rotations.  They are
        # complementary for multi-fragment minima and must be allowed
        # together; forbidding pose support caused artificial rank loss in
        # solvent clusters.
        pass
    elif contract.task_regime == ATLAS_TASK_MINIMUM and len(contract.bodies) > 1:
        if any(
            item.scientific_role == COORDINATE_FRAGMENT_POSE_SUPPORT
            and item.requirement == CANDIDATE_FORBIDDEN
            for item in contract.candidates
        ):
            raise OracleCoordinateAtlasError(
                "a disconnected MINIMUM outside the pseudobond gate requires its "
                "dimension-aware fragment-pose candidates"
            )
    elif contract.task_regime == ATLAS_TASK_EXPLORATION and len(contract.bodies) > 1:
        if not any(
            item.scientific_role == COORDINATE_FRAGMENT_POSE_SUPPORT
            and item.requirement != CANDIDATE_FORBIDDEN
            for item in contract.candidates
        ):
            raise OracleCoordinateAtlasError(
                "a disconnected EXPLORATION atlas requires dimension-aware "
                "fragment-pose candidates"
            )
    for item in contract.family_compatibility:
        _validate_family_block(item)
        if contract.task_regime not in item.task_regimes:
            continue
    _validate_local_domains(contract)
    _validate_reactive_zones(contract)


def _validate_reactive_zones(contract: OracleCoordinateAtlasContract) -> None:
    if contract.task_regime != ATLAS_TASK_TRANSITION_STATE and contract.reactive_zones:
        raise OracleCoordinateAtlasError(
            f"{contract.task_regime} atlases cannot declare TS reactive zones"
        )
    interaction_ids = {item.interaction_id for item in contract.interactions}
    for zone in contract.reactive_zones:
        if not zone.atoms or len(zone.atoms) != len(set(zone.atoms)) or min(zone.atoms) < 1:
            raise OracleCoordinateAtlasError(
                f"reactive zone {zone.zone_id} must contain unique positive atom indices"
            )
        if (
            not zone.source_interaction_ids
            or not set(zone.source_interaction_ids).issubset(interaction_ids)
        ):
            raise OracleCoordinateAtlasError(
                f"reactive zone {zone.zone_id} references unknown interactions"
            )
        if (
            not zone.excluded_primitive_functions
            or any(not item.strip() for item in zone.excluded_primitive_functions)
            or len(zone.excluded_primitive_functions)
            != len(set(zone.excluded_primitive_functions))
        ):
            raise OracleCoordinateAtlasError(
                f"reactive zone {zone.zone_id} needs unique primitive exclusions"
            )
        if not zone.rule_id:
            raise OracleCoordinateAtlasError(
                f"reactive zone {zone.zone_id} lacks atlas rule provenance"
            )


def _validate_local_domains(contract: OracleCoordinateAtlasContract) -> None:
    interaction_by_id = {
        item.interaction_id: item for item in contract.interactions
    }
    required_minimum_interactions = {
        item.interaction_id
        for item in contract.interactions
        if contract.task_regime == ATLAS_TASK_MINIMUM
        and item.pseudobond_policy == PSEUDOBOND_REQUIRED
    }
    domain_interactions = {item.interaction_id for item in contract.local_domains}
    if domain_interactions != required_minimum_interactions:
        raise OracleCoordinateAtlasError(
            "local chart domains must correspond exactly to required MINIMUM pseudobonds"
        )
    for item in contract.local_domains:
        interaction = interaction_by_id.get(item.interaction_id)
        if interaction is None:
            raise OracleCoordinateAtlasError(
                f"local domain {item.domain_id} references an unknown interaction"
            )
        if interaction.pseudobond_policy != PSEUDOBOND_REQUIRED:
            raise OracleCoordinateAtlasError(
                "local chart domains are allowed only for required pseudobonds"
            )
        if (
            not item.reference_fragment_id
            or not item.moving_fragment_id
            or item.reference_fragment_id == item.moving_fragment_id
            or set((item.reference_fragment_id, item.moving_fragment_id))
            != set(interaction.fragment_ids)
        ):
            raise OracleCoordinateAtlasError(
                f"local domain {item.domain_id} has inconsistent fragment ownership"
            )
        if (
            item.reference_endpoint_atom <= 0
            or item.moving_endpoint_atom <= 0
            or item.reference_endpoint_atom == item.moving_endpoint_atom
            or not item.moving_atoms
            or len(item.moving_atoms) != len(set(item.moving_atoms))
            or any(atom <= 0 for atom in item.moving_atoms)
            or item.moving_endpoint_atom not in item.moving_atoms
            or item.reference_endpoint_atom in item.moving_atoms
        ):
            raise OracleCoordinateAtlasError(
                f"local domain {item.domain_id} has invalid atom ownership"
            )
        if (
            not np.isfinite(item.radial_step_angstrom)
            or item.radial_step_angstrom <= 0.0
            or not np.isfinite(item.axial_half_width_degrees)
            or item.axial_half_width_degrees <= 0.0
            or not np.isfinite(item.axial_step_degrees)
            or item.axial_step_degrees <= 0.0
            or item.axial_step_degrees > item.axial_half_width_degrees
            or not np.isfinite(item.tilt_degrees)
            or item.tilt_degrees <= 0.0
        ):
            raise OracleCoordinateAtlasError(
                f"local domain {item.domain_id} has invalid finite sampling bounds"
            )
        ratio = item.axial_half_width_degrees / item.axial_step_degrees
        if abs(ratio - round(ratio)) > 1.0e-12:
            raise OracleCoordinateAtlasError(
                "local axial half width must be an integral number of steps"
            )
        if not all(
            (item.sampling_policy, item.topology_guard, item.contact_guard, item.rule_id)
        ):
            raise OracleCoordinateAtlasError(
                f"local domain {item.domain_id} has incomplete provenance"
            )


def _validate_interaction(item: AtlasInteractionDecision, task_regime: str) -> None:
    if not all((item.decision_id, item.interaction_id, item.semantic_kind, item.rule_id)):
        raise OracleCoordinateAtlasError("atlas interaction identity must be complete")
    if item.graph_role not in GRAPH_ROLES:
        raise OracleCoordinateAtlasError(
            f"unsupported graph role for {item.interaction_id}: {item.graph_role}"
        )
    if item.coordinate_role not in COORDINATE_ROLES:
        raise OracleCoordinateAtlasError(
            f"unsupported coordinate role for {item.interaction_id}: {item.coordinate_role}"
        )
    if item.pseudobond_policy not in PSEUDOBOND_POLICIES:
        raise OracleCoordinateAtlasError(
            f"unsupported pseudobond policy for {item.interaction_id}: "
            f"{item.pseudobond_policy}"
        )
    if item.delta_beta1 < 0:
        raise OracleCoordinateAtlasError("delta_beta1 cannot be negative")
    if item.graph_role == GRAPH_ROLE_OPEN and item.delta_beta1 != 0:
        raise OracleCoordinateAtlasError("OPEN interactions must have delta_beta1=0")
    if item.graph_role == GRAPH_ROLE_CLOSING and item.delta_beta1 <= 0:
        raise OracleCoordinateAtlasError("CLOSING interactions must increase beta1")
    if item.graph_role == GRAPH_ROLE_REACTIVE_SUPPORT:
        if task_regime != ATLAS_TASK_TRANSITION_STATE:
            raise OracleCoordinateAtlasError(
                "REACTIVE_SUPPORT graph edges are permitted only for transition states"
            )
        if item.delta_beta1 != 0:
            raise OracleCoordinateAtlasError(
                "REACTIVE_SUPPORT graph edges must not alter chemical cycle rank"
            )
        if item.coordinate_role not in {
            COORDINATE_REACTION_DISTANCE,
            COORDINATE_REACTION_DISTANCE_ONLY,
            COORDINATE_TS_SUPPORT,
        }:
            raise OracleCoordinateAtlasError(
                "REACTIVE_SUPPORT graph edges must belong to the operative TS kernel"
            )
    if len(item.fragment_ids) != len(set(item.fragment_ids)):
        raise OracleCoordinateAtlasError("interaction fragment identifiers must be unique")
    for endpoint in (item.endpoint_a, item.endpoint_b):
        if len(endpoint) != 2 or not all(endpoint):
            raise OracleCoordinateAtlasError("atlas interaction endpoint is incomplete")

    admits_pseudobond = item.pseudobond_policy != PSEUDOBOND_FORBIDDEN
    if task_regime == ATLAS_TASK_MINIMUM:
        if item.pseudobond_policy == PSEUDOBOND_ALLOWED:
            raise OracleCoordinateAtlasError(
                "MINIMUM atlas decisions are binary: REQUIRED pseudobond or "
                "FORBIDDEN quaternion-pose fallback"
            )
        if item.pseudobond_policy == PSEUDOBOND_FORBIDDEN:
            if item.coordinate_role != COORDINATE_OBSERVABLE_ONLY:
                raise OracleCoordinateAtlasError(
                    "MINIMUM contacts outside the pseudobond gate must be "
                    "observable-only"
                )
    if task_regime == ATLAS_TASK_EXPLORATION:
        if item.pseudobond_policy != PSEUDOBOND_FORBIDDEN:
            raise OracleCoordinateAtlasError(
                "EXPLORATION interactions cannot freeze a minimum-specific pseudobond"
            )
        if item.coordinate_role != COORDINATE_OBSERVABLE_ONLY:
            raise OracleCoordinateAtlasError(
                "EXPLORATION interactions must remain observable-only"
            )
    if task_regime == ATLAS_TASK_MINIMUM and admits_pseudobond:
        if item.semantic_kind not in MINIMUM_PSEUDOBOND_CONTACT_KINDS:
            raise OracleCoordinateAtlasError(
                "MINIMUM pseudobonds require an explicitly classified contact"
            )
        if item.graph_role != GRAPH_ROLE_OPEN:
            raise OracleCoordinateAtlasError(
                "MINIMUM pseudobonds are permitted only for OPEN contact graphs"
            )
        if item.coordinate_role != COORDINATE_PHYSICAL_CONTACT_DISTANCE:
            raise OracleCoordinateAtlasError(
                "MINIMUM pseudobonds must retain the physical-contact coordinate role"
            )
        if item.pseudobond_policy != PSEUDOBOND_REQUIRED or (
            item.rule_id != "MINIMUM.CLASSIFIED_STABLE_OPEN_CONTACT"
        ):
            raise OracleCoordinateAtlasError(
                "MINIMUM pseudobonds require the classified, stable, OPEN atlas rule"
            )
    if task_regime == ATLAS_TASK_TRANSITION_STATE and admits_pseudobond:
        if item.coordinate_role not in {COORDINATE_REACTION_DISTANCE, COORDINATE_TS_SUPPORT}:
            raise OracleCoordinateAtlasError(
                "TS pseudobonds must belong to the reaction kernel or an explicit TS support"
            )


def _validate_body(item: AtlasBodyPrescription) -> None:
    if not item.body_id or not item.atoms or len(item.atoms) != len(set(item.atoms)):
        raise OracleCoordinateAtlasError("atlas body membership must be non-empty and unique")
    if any(atom <= 0 for atom in item.atoms):
        raise OracleCoordinateAtlasError("atlas body atom indices must be one based")
    if item.dimension not in BODY_DIMENSIONS:
        raise OracleCoordinateAtlasError(f"unsupported body dimension: {item.dimension}")
    expected_translation, expected_orientation = BODY_DOF[item.dimension]
    if (item.translation_dofs, item.orientation_dofs) != (
        expected_translation,
        expected_orientation,
    ):
        raise OracleCoordinateAtlasError(
            f"body {item.body_id} has inconsistent rigid-body dimensions"
        )
    allowed_charts = {
        BODY_POINT: {"NONE", "AXIAL_JACOBI"},
        BODY_LINEAR: {"QUATERNION", "AXIS_AXIS_STEREOGRAPHIC"},
        BODY_NONLINEAR: {"QUATERNION"},
    }[item.dimension]
    if item.pose_chart not in allowed_charts:
        raise OracleCoordinateAtlasError(
            f"body {item.body_id} has unsupported {item.dimension} pose chart "
            f"{item.pose_chart}"
        )
    if item.pose_role not in {"REFERENCE", "MOVING"}:
        raise OracleCoordinateAtlasError("atlas body pose role must be REFERENCE or MOVING")
    if not item.rule_id:
        raise OracleCoordinateAtlasError("atlas body prescription requires a rule id")


def _validate_family_block(item: AtlasFamilyCompatibility) -> None:
    if not item.block_id or not item.rule_id or not item.families:
        raise OracleCoordinateAtlasError("family compatibility block is incomplete")
    if len(item.families) != len(set(item.families)):
        raise OracleCoordinateAtlasError("family compatibility block repeats a family")
    if not item.task_regimes or not set(item.task_regimes).issubset(ATLAS_TASK_REGIMES):
        raise OracleCoordinateAtlasError("family block has invalid task regimes")
    try:
        salc_algorithm(item.salc_algorithm_id)
        completion = salc_algorithm(item.completion_algorithm_id)
    except KeyError as exc:
        raise OracleCoordinateAtlasError(str(exc)) from exc
    if completion.algorithm_id != COMPLETION_EXACT_RANK:
        raise OracleCoordinateAtlasError(
            "atlas completion blocks must use the registered exact-rank algorithm"
        )
    if item.substitutions and item.salc_algorithm_id == SALC_NONE:
        raise OracleCoordinateAtlasError(
            "a substitution block requires an explicit analytic SALC algorithm"
        )
    if item.condition_trigger is not None and item.condition_trigger <= 0.0:
        raise OracleCoordinateAtlasError("condition trigger must be positive")
    if item.minimum_relative_gain is not None and not 0.0 <= item.minimum_relative_gain <= 1.0:
        raise OracleCoordinateAtlasError("minimum relative gain must lie in [0, 1]")
    if (item.condition_trigger is None) != (item.minimum_relative_gain is None):
        raise OracleCoordinateAtlasError(
            "condition trigger and relative gain must be declared together"
        )


def oracle_coordinate_atlas_contract_to_dict(
    contract: OracleCoordinateAtlasContract,
) -> dict[str, Any]:
    validate_oracle_coordinate_atlas_contract(contract)
    payload = asdict(contract)
    if not contract.implicit_unclassified_fragments:
        # Preserve serialized identity/hashes of previously frozen contracts.
        payload.pop("implicit_unclassified_fragments")
    return payload


def oracle_coordinate_atlas_contract_from_dict(
    payload: dict[str, Any],
) -> OracleCoordinateAtlasContract:
    try:
        if not isinstance(payload.get("implicit_unclassified_fragments", ()), (list, tuple)):
            raise OracleCoordinateAtlasError("implicit fragment domain must be a sequence")
        contract = OracleCoordinateAtlasContract(
            schema=str(payload["schema"]),
            owner=str(payload["owner"]),
            policy_id=str(payload["policy_id"]),
            policy_version=str(payload["policy_version"]),
            policy_sha256=str(payload["policy_sha256"]),
            coordinate_library_schema=str(payload["coordinate_library_schema"]),
            coordinate_library_sha256=str(payload["coordinate_library_sha256"]),
            salc_library_schema=str(payload["salc_library_schema"]),
            salc_library_sha256=str(payload["salc_library_sha256"]),
            task_regime=str(payload["task_regime"]),
            topology_hash=str(payload["topology_hash"]),
            interactions=tuple(
                AtlasInteractionDecision(
                    decision_id=str(item["decision_id"]),
                    interaction_id=str(item["interaction_id"]),
                    semantic_kind=str(item["semantic_kind"]),
                    endpoint_a=tuple(str(value) for value in item["endpoint_a"]),
                    endpoint_b=tuple(str(value) for value in item["endpoint_b"]),
                    fragment_ids=tuple(str(value) for value in item["fragment_ids"]),
                    graph_role=str(item["graph_role"]),
                    delta_beta1=int(item["delta_beta1"]),
                    coordinate_role=str(item["coordinate_role"]),
                    pseudobond_policy=str(item["pseudobond_policy"]),
                    primitive_candidate_ids=tuple(
                        str(value) for value in item["primitive_candidate_ids"]
                    ),
                    rule_id=str(item["rule_id"]),
                    evidence=tuple(str(value) for value in item["evidence"]),
                )
                for item in payload["interactions"]
            ),
            bodies=tuple(
                AtlasBodyPrescription(
                    body_id=str(item["body_id"]),
                    atoms=tuple(int(value) for value in item["atoms"]),
                    dimension=str(item["dimension"]),
                    translation_dofs=int(item["translation_dofs"]),
                    orientation_dofs=int(item["orientation_dofs"]),
                    pose_chart=str(item["pose_chart"]),
                    pose_role=str(item["pose_role"]),
                    rule_id=str(item["rule_id"]),
                )
                for item in payload["bodies"]
            ),
            candidates=tuple(
                AtlasCandidatePrescription(
                    candidate_id=str(item["candidate_id"]),
                    scientific_role=str(item["scientific_role"]),
                    requirement=str(item["requirement"]),
                    mixing_block=str(item["mixing_block"]),
                    completion_block=str(item["completion_block"]),
                    source_decision_id=str(item["source_decision_id"]),
                    rule_id=str(item["rule_id"]),
                )
                for item in payload["candidates"]
            ),
            family_compatibility=tuple(
                AtlasFamilyCompatibility(
                    block_id=str(item["block_id"]),
                    task_regimes=tuple(str(value) for value in item["task_regimes"]),
                    families=tuple(str(value) for value in item["families"]),
                    salc_algorithm_id=str(item["salc_algorithm_id"]),
                    completion_algorithm_id=str(item["completion_algorithm_id"]),
                    substitutions=bool(item["substitutions"]),
                    require_same_domain=bool(item["require_same_domain"]),
                    require_same_irrep=bool(item["require_same_irrep"]),
                    condition_trigger=(
                        None
                        if item["condition_trigger"] is None
                        else float(item["condition_trigger"])
                    ),
                    minimum_relative_gain=(
                        None
                        if item["minimum_relative_gain"] is None
                        else float(item["minimum_relative_gain"])
                    ),
                    rule_id=str(item["rule_id"]),
                )
                for item in payload["family_compatibility"]
            ),
            local_domains=tuple(
                AtlasLocalDomainPrescription(
                    domain_id=str(item["domain_id"]),
                    interaction_id=str(item["interaction_id"]),
                    reference_fragment_id=str(item["reference_fragment_id"]),
                    moving_fragment_id=str(item["moving_fragment_id"]),
                    reference_endpoint_atom=int(item["reference_endpoint_atom"]),
                    moving_endpoint_atom=int(item["moving_endpoint_atom"]),
                    moving_atoms=tuple(int(value) for value in item["moving_atoms"]),
                    radial_step_angstrom=float(item["radial_step_angstrom"]),
                    axial_half_width_degrees=float(item["axial_half_width_degrees"]),
                    axial_step_degrees=float(item["axial_step_degrees"]),
                    tilt_degrees=float(item["tilt_degrees"]),
                    sampling_policy=str(item["sampling_policy"]),
                    topology_guard=str(item["topology_guard"]),
                    contact_guard=str(item["contact_guard"]),
                    rule_id=str(item["rule_id"]),
                )
                for item in payload["local_domains"]
            ),
            reactive_zones=tuple(
                AtlasReactiveZonePrescription(
                    zone_id=str(item["zone_id"]),
                    atoms=tuple(int(value) for value in item["atoms"]),
                    source_interaction_ids=tuple(
                        str(value) for value in item["source_interaction_ids"]
                    ),
                    excluded_primitive_functions=tuple(
                        str(value) for value in item["excluded_primitive_functions"]
                    ),
                    rule_id=str(item["rule_id"]),
                )
                for item in payload["reactive_zones"]
            ),
            fallback_order=tuple(str(value) for value in payload["fallback_order"]),
            provenance=str(payload["provenance"]),
            implicit_unclassified_fragments=tuple(payload.get("implicit_unclassified_fragments", ())),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise OracleCoordinateAtlasError("invalid ORACLE coordinate-atlas payload") from exc
    validate_oracle_coordinate_atlas_contract(contract)
    return contract


class AtlasFragmentPairLookup:
    """Index explicit decisions once; resolve one unclassified pair on demand.

    Never enumerate the complement graph. Explicit contacts take precedence,
    including multiple contacts for the same pair. Legacy explicit contracts
    remain explicit and do not acquire new default interactions.
    """

    def __init__(self, contract: OracleCoordinateAtlasContract):
        validate_oracle_coordinate_atlas_contract(contract)
        self.implicit = frozenset(contract.implicit_unclassified_fragments)
        self.members = set(self.implicit) | {body.body_id for body in contract.bodies}
        self.explicit = {}
        for item in contract.interactions:
            self.members.update(item.fragment_ids)
            if len(set(item.fragment_ids)) == 2:
                self.explicit.setdefault(tuple(sorted(set(item.fragment_ids))), []).append(item)

    def for_pair(self, left: str, right: str) -> tuple[AtlasInteractionDecision, ...]:
        if left == right or left not in self.members or right not in self.members:
            raise OracleCoordinateAtlasError("fragment pair must contain two distinct known fragments")
        pair = tuple(sorted((left, right)))
        if pair in self.explicit:
            return tuple(self.explicit[pair])
        if not self.implicit:
            return ()
        return (AtlasInteractionDecision(
            decision_id=f"IMPLICIT:{pair[0]}|{pair[1]}",
            interaction_id=f"UNCLASSIFIED:{pair[0]}|{pair[1]}",
            semantic_kind="UNCLASSIFIED_DISPERSIVE",
            endpoint_a=("FRAGMENT", pair[0]), endpoint_b=("FRAGMENT", pair[1]),
            fragment_ids=pair, graph_role=GRAPH_ROLE_NONE, delta_beta1=0,
            coordinate_role=COORDINATE_OBSERVABLE_ONLY,
            pseudobond_policy=PSEUDOBOND_FORBIDDEN, primitive_candidate_ids=(),
            rule_id="MINIMUM.UNCLASSIFIED_POSE_ONLY",
            evidence=("NO_REGISTERED_ORACLE_AUXILIARY_CONTACT",),
        ),)


def coordinate_atlas_payload_sha256(contract: OracleCoordinateAtlasContract) -> str:
    payload = oracle_coordinate_atlas_contract_to_dict(contract)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_oracle_coordinate_atlas_contract(
    path: Path,
    contract: OracleCoordinateAtlasContract,
) -> None:
    payload = oracle_coordinate_atlas_contract_to_dict(contract)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    replace_section(
        Path(path),
        ORACLE_COORDINATE_ATLAS_SECTION,
        (
            f"SCHEMA {contract.schema}",
            f"OWNER {contract.owner}",
            f"PAYLOAD_SHA256 {coordinate_atlas_payload_sha256(contract)}",
            f"JSON {encoded}",
        ),
    )


def read_oracle_coordinate_atlas_contract(path: Path) -> OracleCoordinateAtlasContract:
    content = section_content(
        read_sectioned_lines(Path(path)), ORACLE_COORDINATE_ATLAS_SECTION
    )
    if not content:
        raise OracleCoordinateAtlasError(
            f"missing #{ORACLE_COORDINATE_ATLAS_SECTION} section"
        )
    metadata: dict[str, str] = {}
    for line in content:
        key, separator, value = line.partition(" ")
        if separator:
            metadata[key.strip().upper()] = value.strip()
    if metadata.get("SCHEMA") != ORACLE_COORDINATE_ATLAS_SCHEMA:
        raise OracleCoordinateAtlasError("invalid coordinate-atlas section schema")
    if metadata.get("OWNER") != ORACLE_COORDINATE_ATLAS_OWNER:
        raise OracleCoordinateAtlasError("invalid coordinate-atlas section owner")
    try:
        payload = json.loads(metadata["JSON"])
    except (KeyError, json.JSONDecodeError) as exc:
        raise OracleCoordinateAtlasError("invalid coordinate-atlas section JSON") from exc
    contract = oracle_coordinate_atlas_contract_from_dict(payload)
    expected = coordinate_atlas_payload_sha256(contract)
    if metadata.get("PAYLOAD_SHA256") != expected:
        raise OracleCoordinateAtlasError("coordinate-atlas payload SHA256 mismatch")
    return contract


def _validate_unique(values, label: str) -> None:
    records = tuple(values)
    if any(not value for value in records) or len(records) != len(set(records)):
        raise OracleCoordinateAtlasError(f"atlas {label}s must be non-empty and unique")


__all__ = [
    name
    for name in globals()
    if name.startswith("ATLAS_")
    or name.startswith("BODY_")
    or name.startswith("CANDIDATE_")
    or name.startswith("COORDINATE_")
    or name.startswith("GRAPH_ROLE_")
    or name.startswith("MINIMUM_")
    or name.startswith("ORACLE_COORDINATE_")
    or name.startswith("PSEUDOBOND_")
    or name.startswith("Atlas")
    or name.startswith("OracleCoordinate")
    or name
    in {
        "coordinate_atlas_payload_sha256",
        "oracle_coordinate_atlas_contract_from_dict",
        "oracle_coordinate_atlas_contract_to_dict",
        "read_oracle_coordinate_atlas_contract",
        "validate_oracle_coordinate_atlas_contract",
        "write_oracle_coordinate_atlas_contract",
    }
]
