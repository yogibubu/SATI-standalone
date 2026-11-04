"""ORACLE-owned scientific atlas and per-system prescriptions.

This module is the only policy layer that maps frozen chemical evidence to
coordinate capabilities.  It does not perform numerical rank selection; that
is SMITH's responsibility after this complete prescription has been frozen.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

from matrix_chem import (
    COMPLETION_EXACT_RANK,
    COORDINATE_LIBRARY_SCHEMA,
    SALC_B_ORTHOGONAL,
    SALC_LIBRARY_SCHEMA,
    SALC_NONE,
    SALC_POINT_GROUP_PROJECTOR,
    SALC_PROJECTOR_THEN_B_ORTHOGONAL,
    coordinate_library_manifest,
    is_linear_geometry,
    read_enriched_xyz,
    salc_library_manifest,
    topology_snapshot_from_xyzin,
)
from matrix_chem.coordinate_atlas_contract import (
    ATLAS_FALLBACK_ORDER,
    ATLAS_TASK_EXPLORATION,
    ATLAS_TASK_MINIMUM,
    ATLAS_TASK_TRANSITION_STATE,
    AtlasBodyPrescription,
    AtlasCandidatePrescription,
    AtlasFamilyCompatibility,
    AtlasInteractionDecision,
    AtlasLocalDomainPrescription,
    AtlasReactiveZonePrescription,
    CANDIDATE_OPTIONAL,
    CANDIDATE_FORBIDDEN,
    CANDIDATE_REQUIRED,
    COORDINATE_FRAGMENT_POSE_SUPPORT,
    COORDINATE_INTERNAL_VALENCE,
    COORDINATE_OBSERVABLE_ONLY,
    COORDINATE_PHYSICAL_CONTACT_DISTANCE,
    COORDINATE_REACTION_DISTANCE,
    COORDINATE_REACTION_DISTANCE_ONLY,
    COORDINATE_TS_SUPPORT,
    GRAPH_ROLE_NONE,
    GRAPH_ROLE_REACTIVE_SUPPORT,
    MINIMUM_PSEUDOBOND_CONTACT_KINDS,
    ORACLE_COORDINATE_ATLAS_OWNER,
    ORACLE_COORDINATE_ATLAS_SCHEMA,
    OracleCoordinateAtlasContract,
    PSEUDOBOND_FORBIDDEN,
    PSEUDOBOND_REQUIRED,
    validate_oracle_coordinate_atlas_contract,
    write_oracle_coordinate_atlas_contract,
)
from matrix_chem.transition_state_contract import (
    OracleTransitionStateGeometryContract,
    transition_state_descriptor,
    validate_oracle_transition_state_geometry_contract,
)
from matrix_chem.oracle_sonic_contract import (
    AuxiliaryContact,
    FragmentMembership,
    OracleSonicContract,
    read_oracle_sonic_contract,
    validate_oracle_sonic_contract,
)

from .auxiliary_contacts import AuxiliaryContactProviderSettings
from .perception_policy import (
    ORACLE_CHEMICAL_PERCEPTION_POLICIES,
    chemical_perception_policy_manifest,
)


ORACLE_COORDINATE_ATLAS_POLICY_ID = "ORACLE_COORDINATE_SCIENTIFIC_ATLAS"
ORACLE_COORDINATE_ATLAS_POLICY_VERSION = "12"
ORACLE_COORDINATE_ATLAS_BUILDER = "ORACLE_COORDINATE_ATLAS_BUILDER@1"
BODY_LINEAR_RELATIVE_SINGULAR_THRESHOLD = 1.0e-6
MINIMUM_LOCAL_DOMAIN_RADIAL_CAP_ANGSTROM = 0.10
MINIMUM_LOCAL_DOMAIN_AXIAL_HALF_WIDTH_DEGREES = 60.0
MINIMUM_LOCAL_DOMAIN_AXIAL_STEP_DEGREES = 15.0
MINIMUM_LOCAL_DOMAIN_TILT_DEGREES = 10.0
MINIMUM_LOCAL_DOMAIN_CONDITION_TRIGGER = 10.0
MINIMUM_LOCAL_DOMAIN_RELATIVE_GAIN = 0.05
MINIMUM_LOCAL_DOMAIN_SAMPLING_POLICY = "CONTACT_LOCAL_RIGID_STENCIL_V1"
TS_RADIAL_CONDITION_TRIGGER = 10.0
TS_RADIAL_MINIMUM_RELATIVE_GAIN = 0.05
TS_RADIAL_SELECTION_POLICY = "LOCALITY_UNLESS_MATERIAL_CONDITION_GAIN_V1"
MINIMUM_FRAGMENT_REPRESENTATION_AUTOMATIC = "automatic"
MINIMUM_FRAGMENT_REPRESENTATION_PSEUDO_BONDS = "pseudo-bonds"
MINIMUM_FRAGMENT_REPRESENTATION_QUATERNION = "quaternion"
MINIMUM_FRAGMENT_REPRESENTATIONS = frozenset(
    {
        MINIMUM_FRAGMENT_REPRESENTATION_AUTOMATIC,
        MINIMUM_FRAGMENT_REPRESENTATION_PSEUDO_BONDS,
        MINIMUM_FRAGMENT_REPRESENTATION_QUATERNION,
    }
)
_MINIMUM_CONTACT_PROVIDER_SETTINGS = AuxiliaryContactProviderSettings()
_MINIMUM_CONTACT_POLICIES = {
    item.family: item
    for item in ORACLE_CHEMICAL_PERCEPTION_POLICIES
    if item.family in MINIMUM_PSEUDOBOND_CONTACT_KINDS
}


@lru_cache(maxsize=1)
def _coordinate_atlas_policy_manifest_json() -> str:
    payload = {
        "schema": "matrix.oracle.coordinate_atlas_policy.v1",
        "owner": ORACLE_COORDINATE_ATLAS_OWNER,
        "policy_id": ORACLE_COORDINATE_ATLAS_POLICY_ID,
        "policy_version": ORACLE_COORDINATE_ATLAS_POLICY_VERSION,
        "coordinate_library_sha256": coordinate_library_manifest()["sha256"],
        "salc_library_sha256": salc_library_manifest()["sha256"],
        "chemical_perception_policy_sha256": (
            chemical_perception_policy_manifest()["sha256"]
        ),
        "body_linear_relative_singular_threshold": (
            BODY_LINEAR_RELATIVE_SINGULAR_THRESHOLD
        ),
        "minimum_local_domain": {
            "radial_cap_angstrom": MINIMUM_LOCAL_DOMAIN_RADIAL_CAP_ANGSTROM,
            "axial_half_width_degrees": (
                MINIMUM_LOCAL_DOMAIN_AXIAL_HALF_WIDTH_DEGREES
            ),
            "axial_step_degrees": MINIMUM_LOCAL_DOMAIN_AXIAL_STEP_DEGREES,
            "tilt_degrees": MINIMUM_LOCAL_DOMAIN_TILT_DEGREES,
            "condition_trigger": MINIMUM_LOCAL_DOMAIN_CONDITION_TRIGGER,
            "minimum_relative_gain": MINIMUM_LOCAL_DOMAIN_RELATIVE_GAIN,
            "sampling_policy": MINIMUM_LOCAL_DOMAIN_SAMPLING_POLICY,
        },
        "transition_state_radial_conditioning": {
            "condition_trigger": TS_RADIAL_CONDITION_TRIGGER,
            "minimum_relative_gain": TS_RADIAL_MINIMUM_RELATIVE_GAIN,
            "selection_policy": TS_RADIAL_SELECTION_POLICY,
            "strict_trigger": "DIRECT_NORMALIZED_CONDITION_GREATER_THAN_TRIGGER",
        },
        "transition_state_reactive_zone": {
            "atom_scope": "OPERATIVE_REACTION_KERNEL_INCIDENT_ATOMS",
            "excluded_primitive_functions": ["L"],
            "completion_graph": "PRIMARY_PLUS_OPERATIVE_REACTIVE_SUPPORT",
            "completion_graph_trigger": "PRIMARY_CHART_BELOW_EXACT_RANK",
            "completion_graph_cycle_semantics": "NONCHEMICAL_SUPPORT_ONLY",
            "completion_block": "TS_REACTIVE_COMPLETION",
            "completion_families": [
                "FRAG_DISTANCE",
                "OUT_OF_PLANE",
                "PSEUDO_BOND_BEND",
                "PSEUDO_BOND_TORSION",
            ],
            "completion_policy": "LOCAL_EXACT_RANK_MINIMAL",
            "rule_id": "TS.REACTIVE_ZONE_NO_LINEAR_BEND",
        },
        "task_regimes": [
            ATLAS_TASK_MINIMUM,
            ATLAS_TASK_EXPLORATION,
            ATLAS_TASK_TRANSITION_STATE,
        ],
        "minimum_pseudobond_contact_kinds": sorted(
            MINIMUM_PSEUDOBOND_CONTACT_KINDS
        ),
        "minimum_contact_stability_gate": [
            {
                "family": family,
                "metric": policy.metric,
                "entry_threshold": policy.entry_threshold,
                "active_direction": policy.active_direction,
                "minimum_confidence": (
                    _MINIMUM_CONTACT_PROVIDER_SETTINGS.minimum_confidence_for(
                        family
                    )
                ),
            }
            for family, policy in sorted(_MINIMUM_CONTACT_POLICIES.items())
        ],
        "minimum_rules": [
            "CLASSIFIED_STABLE_OPEN_CONTACT_REQUIRED_PSEUDOBOND",
            "ALL_OTHER_DISCONNECTED_MINIMUMS_DIMENSION_AWARE_QUATERNION_POSE",
            "PSEUDOBOND_CONTACT_AND_RIGID_POSE_COMPLEMENTARY",
            "NO_CLOSEST_PAIR_SCIENTIFIC_INFERENCE",
        ],
        "exploration_rules": [
            "CONTACTS_REMAIN_OBSERVABLE_ONLY",
            "NO_MINIMUM_SPECIFIC_PSEUDOBONDS",
            "DISCONNECTED_FRAGMENTS_USE_DIMENSION_AWARE_POSE",
        ],
        "family_rules": [
            "REPRESENTATION_IS_NOT_SCIENTIFIC_ROLE",
            "SAME_DOMAIN_AND_IRREP_BY_DEFAULT",
            "NO_CROSS_FAMILY_SUBSTITUTION_UNLESS_DECLARED",
            "POSE_NEVER_MIXES_WITH_INTERNAL_VALENCE",
        ],
        "fallback_order": list(ATLAS_FALLBACK_ORDER),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return json.dumps(
        {**payload, "sha256": hashlib.sha256(encoded).hexdigest()},
        sort_keys=True,
        separators=(",", ":"),
    )


def coordinate_atlas_policy_manifest() -> dict:
    """Return a fresh view of the cached immutable atlas policy."""

    return json.loads(_coordinate_atlas_policy_manifest_json())


def build_minimum_coordinate_atlas_contract(
    sonic_contract: OracleSonicContract,
    *,
    coordinates_angstrom: np.ndarray | None = None,
    fragment_representation: str = MINIMUM_FRAGMENT_REPRESENTATION_AUTOMATIC,
) -> OracleCoordinateAtlasContract:
    """Map a frozen ORACLE SONIC contract to a MINIMUM prescription."""

    validate_oracle_sonic_contract(sonic_contract)
    representation = _minimum_fragment_representation(fragment_representation)
    interactions, decision_by_contact = _minimum_interaction_decisions(sonic_contract)
    if representation == MINIMUM_FRAGMENT_REPRESENTATION_QUATERNION:
        interactions = tuple(
            replace(
                item,
                coordinate_role=COORDINATE_OBSERVABLE_ONLY,
                pseudobond_policy=PSEUDOBOND_FORBIDDEN,
                rule_id="MINIMUM.EXPLICIT_QUATERNION_POSE_COMPARISON",
                evidence=(
                    *item.evidence,
                    "EXPLICIT_FRAGMENT_REPRESENTATION=QUATERNION",
                ),
            )
            for item in interactions
        )
        decision_by_contact = {
            item.interaction_id: item
            for item in interactions
            if not item.interaction_id.startswith("UNCLASSIFIED:")
        }
    has_required_pseudobond = any(
        item.pseudobond_policy == PSEUDOBOND_REQUIRED for item in interactions
    )
    if (
        representation == MINIMUM_FRAGMENT_REPRESENTATION_PSEUDO_BONDS
        and not has_required_pseudobond
        and len(sonic_contract.primary_topology.fragments) > 1
    ):
        raise ValueError(
            "the explicit pseudo-bonds representation requires at least one "
            "ORACLE-classified stable open interfragment contact"
        )
    # Stable inter-fragment contacts are constraints, not a replacement for
    # the rigid pose coordinates of disconnected bodies.  Keep the pose
    # block available as completion support; otherwise a cluster with many
    # contacts can still miss the three rotational DOFs of each fragment.
    candidates, block_families = _minimum_candidate_prescriptions(
        sonic_contract,
        decision_by_contact=decision_by_contact,
        allow_fragment_pose=True,
    )
    contract = _coordinate_atlas_contract(
        task_regime=ATLAS_TASK_MINIMUM,
        topology_hash=sonic_contract.primary_topology.topology_hash,
        interactions=interactions,
        bodies=_body_prescriptions(
            sonic_contract.primary_topology.fragments,
            coordinates_angstrom,
        ),
        candidates=candidates,
        family_compatibility=_minimum_family_compatibility(block_families),
        local_domains=_minimum_local_domains(
            sonic_contract,
            interactions=interactions,
        ),
        reactive_zones=(),
    )
    contract = replace(contract, implicit_unclassified_fragments=tuple(
        fragment.fragment_id for fragment in sonic_contract.primary_topology.fragments
    ))
    validate_oracle_coordinate_atlas_contract(contract)
    return contract


def _minimum_fragment_representation(value: str) -> str:
    normalized = str(value).strip().casefold().replace("_", "-")
    aliases = {
        "auto": MINIMUM_FRAGMENT_REPRESENTATION_AUTOMATIC,
        "automatic": MINIMUM_FRAGMENT_REPRESENTATION_AUTOMATIC,
        "pseudo-bond": MINIMUM_FRAGMENT_REPRESENTATION_PSEUDO_BONDS,
        "pseudo-bonds": MINIMUM_FRAGMENT_REPRESENTATION_PSEUDO_BONDS,
        "pseudobond": MINIMUM_FRAGMENT_REPRESENTATION_PSEUDO_BONDS,
        "pseudobonds": MINIMUM_FRAGMENT_REPRESENTATION_PSEUDO_BONDS,
        "quaternion": MINIMUM_FRAGMENT_REPRESENTATION_QUATERNION,
        "quaternions": MINIMUM_FRAGMENT_REPRESENTATION_QUATERNION,
        "special-coordinates": MINIMUM_FRAGMENT_REPRESENTATION_QUATERNION,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        choices = ", ".join(sorted(MINIMUM_FRAGMENT_REPRESENTATIONS))
        raise ValueError(
            f"minimum fragment representation must be one of: {choices}"
        ) from exc


def build_exploration_coordinate_atlas_contract(
    sonic_contract: OracleSonicContract,
    *,
    coordinates_angstrom: np.ndarray | None = None,
) -> OracleCoordinateAtlasContract:
    """Build the dynamic chart used only to search between endpoint minima."""

    validate_oracle_sonic_contract(sonic_contract)
    minimum_interactions, minimum_decisions = _minimum_interaction_decisions(
        sonic_contract
    )
    interactions = tuple(
        replace(
            item,
            coordinate_role=COORDINATE_OBSERVABLE_ONLY,
            pseudobond_policy=PSEUDOBOND_FORBIDDEN,
            rule_id="EXPLORATION.OBSERVABLE_CONTACT_WITH_POSE_CHART",
            evidence=(*item.evidence, "EXPLORATION_CONTACT_IS_NOT_A_FROZEN_EDGE"),
        )
        for item in minimum_interactions
    )
    exploration_by_interaction = {
        item.interaction_id: item for item in interactions
    }
    decision_by_contact = {
        contact_id: exploration_by_interaction[item.interaction_id]
        for contact_id, item in minimum_decisions.items()
    }
    candidates, block_families = _exploration_candidate_prescriptions(
        sonic_contract,
        decision_by_contact=decision_by_contact,
    )
    contract = _coordinate_atlas_contract(
        task_regime=ATLAS_TASK_EXPLORATION,
        topology_hash=sonic_contract.primary_topology.topology_hash,
        interactions=interactions,
        bodies=_body_prescriptions(
            sonic_contract.primary_topology.fragments,
            coordinates_angstrom,
        ),
        candidates=candidates,
        family_compatibility=_exploration_family_compatibility(block_families),
        local_domains=(),
        reactive_zones=(),
    )
    validate_oracle_coordinate_atlas_contract(contract)
    return contract


def _minimum_interaction_decisions(
    sonic_contract: OracleSonicContract,
) -> tuple[
    tuple[AtlasInteractionDecision, ...],
    dict[str, AtlasInteractionDecision],
]:
    interactions: list[AtlasInteractionDecision] = []
    decision_by_contact: dict[str, AtlasInteractionDecision] = {}
    for index, contact in enumerate(sonic_contract.auxiliary_contacts, start=1):
        if contact.kind not in MINIMUM_PSEUDOBOND_CONTACT_KINDS:
            raise ValueError(
                f"auxiliary contact {contact.contact_id} has no registered MINIMUM atlas class: "
                f"{contact.kind}"
            )
        fragment_ids = tuple(dict.fromkeys(contact.fragment_ids))
        is_open_interfragment = (
            contact.open_or_closing == "OPEN" and len(fragment_ids) == 2
        )
        is_stable, qualification_evidence = _minimum_contact_is_stable(contact)
        requires_pseudobond = is_open_interfragment and is_stable
        decision_id = f"D{index:04d}"
        rule_id = (
            "MINIMUM.CLASSIFIED_STABLE_OPEN_CONTACT"
            if requires_pseudobond
            else "MINIMUM.QUATERNION_POSE_FALLBACK"
        )
        decision = AtlasInteractionDecision(
            decision_id=decision_id,
            interaction_id=contact.contact_id,
            semantic_kind=contact.kind,
            endpoint_a=(contact.endpoint_a.kind, contact.endpoint_a.identifier),
            endpoint_b=(contact.endpoint_b.kind, contact.endpoint_b.identifier),
            fragment_ids=fragment_ids,
            graph_role=contact.open_or_closing,
            delta_beta1=int(contact.delta_beta1_if_added),
            coordinate_role=(
                COORDINATE_PHYSICAL_CONTACT_DISTANCE
                if requires_pseudobond
                else COORDINATE_OBSERVABLE_ONLY
            ),
            pseudobond_policy=(
                PSEUDOBOND_REQUIRED
                if requires_pseudobond
                else PSEUDOBOND_FORBIDDEN
            ),
            primitive_candidate_ids=contact.primitive_candidate_ids,
            rule_id=rule_id,
            evidence=(
                f"{contact.provider}@{contact.provider_version}",
                *qualification_evidence,
                f"OPEN_INTERFRAGMENT={str(is_open_interfragment).upper()}",
                contact.provenance,
            ),
        )
        interactions.append(decision)
        decision_by_contact[contact.contact_id] = decision

    return tuple(interactions), decision_by_contact


def _minimum_contact_is_stable(
    contact: AuxiliaryContact,
) -> tuple[bool, tuple[str, ...]]:
    """Apply ORACLE's registered entry policy to one classified contact.

    The contact provider has already established the semantic class.  The
    atlas reuses the same family confidence threshold and the chemical-policy
    entry threshold; it does not invent a second geometric cutoff.  Contacts
    in the entry/exit hysteresis band remain observables and therefore select
    the dimension-aware fragment-pose chart.
    """

    policy = _MINIMUM_CONTACT_POLICIES.get(contact.kind)
    if policy is None:
        raise ValueError(
            f"auxiliary contact {contact.contact_id} has no registered MINIMUM "
            f"stability policy: {contact.kind}"
        )
    if policy.metric != "RHO_VDW_AND_DIRECTIONAL_CONFIDENCE":
        raise ValueError(
            f"MINIMUM contact policy {contact.kind} has an unsupported stability metric"
        )
    confidence_threshold = (
        _MINIMUM_CONTACT_PROVIDER_SETTINGS.minimum_confidence_for(contact.kind)
    )
    radial_entry = (
        contact.rho_vdw <= policy.entry_threshold
        if policy.active_direction == "LOWER"
        else contact.rho_vdw >= policy.entry_threshold
    )
    confidence_entry = contact.confidence >= confidence_threshold
    stable = radial_entry and confidence_entry
    return stable, (
        "CLASSIFIED=TRUE",
        f"STABLE={str(stable).upper()}",
        f"RHO_VDW={contact.rho_vdw:.12g}",
        f"RHO_ENTRY_THRESHOLD={policy.entry_threshold:.12g}",
        f"CONFIDENCE={contact.confidence:.12g}",
        f"CONFIDENCE_THRESHOLD={confidence_threshold:.12g}",
        f"PERSISTENCE={contact.persistence:.12g}",
        f"STABILITY_POLICY={policy.provider}@{policy.provider_version}",
    )


def _minimum_local_domains(
    sonic_contract: OracleSonicContract,
    *,
    interactions: tuple[AtlasInteractionDecision, ...],
) -> tuple[AtlasLocalDomainPrescription, ...]:
    """Freeze one chemistry-preserving local stencil per required contact."""

    contact_by_id = {
        contact.contact_id: contact for contact in sonic_contract.auxiliary_contacts
    }
    fragment_by_id = {
        fragment.fragment_id: fragment
        for fragment in sonic_contract.primary_topology.fragments
    }
    domains: list[AtlasLocalDomainPrescription] = []
    for interaction in interactions:
        if interaction.pseudobond_policy != PSEUDOBOND_REQUIRED:
            continue
        contact = contact_by_id[interaction.interaction_id]
        endpoint_atoms = (
            _atom_contact_endpoint(contact.endpoint_a, contact.contact_id),
            _atom_contact_endpoint(contact.endpoint_b, contact.contact_id),
        )
        endpoint_fragment_ids = tuple(
            _fragment_containing_atom(
                fragment_by_id,
                atom,
                contact_id=contact.contact_id,
            )
            for atom in endpoint_atoms
        )
        if endpoint_fragment_ids[0] == endpoint_fragment_ids[1]:
            raise ValueError(
                f"required MINIMUM contact {contact.contact_id} is not interfragment"
            )
        reference_fragment_id = min(
            endpoint_fragment_ids,
            key=lambda fragment_id: (
                -len(fragment_by_id[fragment_id].atoms),
                fragment_id,
            ),
        )
        moving_fragment_id = next(
            fragment_id
            for fragment_id in endpoint_fragment_ids
            if fragment_id != reference_fragment_id
        )
        endpoint_by_fragment = dict(zip(endpoint_fragment_ids, endpoint_atoms))
        radial_step = _minimum_contact_radial_step(contact)
        domains.append(
            AtlasLocalDomainPrescription(
                domain_id=f"LD{len(domains) + 1:04d}",
                interaction_id=interaction.interaction_id,
                reference_fragment_id=reference_fragment_id,
                moving_fragment_id=moving_fragment_id,
                reference_endpoint_atom=endpoint_by_fragment[reference_fragment_id],
                moving_endpoint_atom=endpoint_by_fragment[moving_fragment_id],
                moving_atoms=tuple(fragment_by_id[moving_fragment_id].atoms),
                radial_step_angstrom=radial_step,
                axial_half_width_degrees=(
                    MINIMUM_LOCAL_DOMAIN_AXIAL_HALF_WIDTH_DEGREES
                ),
                axial_step_degrees=MINIMUM_LOCAL_DOMAIN_AXIAL_STEP_DEGREES,
                tilt_degrees=MINIMUM_LOCAL_DOMAIN_TILT_DEGREES,
                sampling_policy=MINIMUM_LOCAL_DOMAIN_SAMPLING_POLICY,
                topology_guard="PRIMARY_TOPOLOGY_INVARIANT",
                contact_guard="CLASSIFIED_STABLE_OPEN_CONTACT_INVARIANT",
                rule_id="MINIMUM.LOCAL_PSEUDOBOND_CHART_DOMAIN",
            )
        )
    return tuple(domains)


def _atom_contact_endpoint(endpoint, contact_id: str) -> int:
    if endpoint.kind != "ATOM":
        raise ValueError(
            f"required MINIMUM contact {contact_id} needs atom endpoints for its local domain"
        )
    try:
        atom = int(endpoint.identifier)
    except ValueError as exc:
        raise ValueError(
            f"required MINIMUM contact {contact_id} has a non-integer atom endpoint"
        ) from exc
    if atom <= 0:
        raise ValueError(
            f"required MINIMUM contact {contact_id} has an invalid atom endpoint"
        )
    return atom


def _fragment_containing_atom(
    fragments: dict[str, FragmentMembership],
    atom: int,
    *,
    contact_id: str,
) -> str:
    owners = tuple(
        fragment_id
        for fragment_id, fragment in fragments.items()
        if atom in fragment.atoms
    )
    if len(owners) != 1:
        raise ValueError(
            f"required MINIMUM contact {contact_id} endpoint {atom} has ambiguous fragment ownership"
        )
    return owners[0]


def _minimum_contact_radial_step(contact: AuxiliaryContact) -> float:
    policy = _MINIMUM_CONTACT_POLICIES[contact.kind]
    if policy.active_direction != "LOWER" or contact.rho_vdw <= 0.0:
        raise ValueError(
            f"MINIMUM local domain requires a lower-active radial policy for {contact.kind}"
        )
    boundary_distance = (
        contact.distance_angstrom * policy.entry_threshold / contact.rho_vdw
    )
    margin = boundary_distance - contact.distance_angstrom
    step = min(MINIMUM_LOCAL_DOMAIN_RADIAL_CAP_ANGSTROM, 0.5 * margin)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError(
            f"stable MINIMUM contact {contact.contact_id} has no positive radial domain"
        )
    return float(step)


def _minimum_candidate_prescriptions(
    sonic_contract: OracleSonicContract,
    *,
    decision_by_contact: dict[str, AtlasInteractionDecision],
    allow_fragment_pose: bool,
) -> tuple[tuple[AtlasCandidatePrescription, ...], dict[str, set[str]]]:
    block_families: dict[str, set[str]] = {}
    candidate_records: list[AtlasCandidatePrescription] = []
    contacts = {
        contact.contact_id: contact for contact in sonic_contract.auxiliary_contacts
    }
    for candidate in sonic_contract.primitive_candidates:
        source_decision = decision_by_contact.get(candidate.owner_id)
        source_decision_id = (
            "" if source_decision is None else source_decision.decision_id
        )
        if candidate.domain_id == "AUXILIARY_CONTACTS":
            contact = contacts[candidate.owner_id]
            if source_decision is None:
                raise ValueError(
                    f"missing MINIMUM atlas decision for contact {contact.contact_id}"
                )
            requires_pseudobond = (
                source_decision.pseudobond_policy == PSEUDOBOND_REQUIRED
            )
            block_id = (
                "MINIMUM_CONTACT::PSEUDOBOND"
                if requires_pseudobond
                else "MINIMUM_CONTACT::OBSERVABLE"
            )
            role = source_decision.coordinate_role
            requirement = (
                CANDIDATE_REQUIRED
                if requires_pseudobond
                else CANDIDATE_FORBIDDEN
            )
            rule_id = source_decision.rule_id
        elif candidate.domain_id == "FRAGMENT_POSE":
            block_id = "MINIMUM_FRAGMENT_POSE"
            role = COORDINATE_FRAGMENT_POSE_SUPPORT
            requirement = (
                CANDIDATE_OPTIONAL if allow_fragment_pose else CANDIDATE_FORBIDDEN
            )
            rule_id = (
                "MINIMUM.POSE_SUPPORT"
                if allow_fragment_pose
                else "MINIMUM.PSEUDOBOND_CONTACT_EXCLUDES_POSE"
            )
        elif candidate.domain_id == "STRUCTURAL_SITES":
            block_id = "MINIMUM_STRUCTURAL_SITE"
            role = COORDINATE_PHYSICAL_CONTACT_DISTANCE
            requirement = CANDIDATE_OPTIONAL
            rule_id = "MINIMUM.STRUCTURAL_SITE_DISTANCE"
        else:
            block_id = f"INTERNAL::{candidate.family}"
            role = COORDINATE_INTERNAL_VALENCE
            requirement = CANDIDATE_OPTIONAL
            rule_id = "COMMON.INTERNAL_VALENCE"
        block_families.setdefault(block_id, set()).add(candidate.family)
        candidate_records.append(
            AtlasCandidatePrescription(
                candidate_id=candidate.candidate_id,
                scientific_role=role,
                requirement=requirement,
                mixing_block=block_id,
                completion_block=block_id,
                source_decision_id=source_decision_id,
                rule_id=rule_id,
            )
        )
    pseudobond_block = block_families.get("MINIMUM_CONTACT::PSEUDOBOND")
    if pseudobond_block is not None:
        pseudobond_block.update(
            {
                "FRAG_DISTANCE",
                "PSEUDO_BOND_BEND",
                "PSEUDO_BOND_TORSION",
            }
        )
    return tuple(candidate_records), block_families


def _exploration_candidate_prescriptions(
    sonic_contract: OracleSonicContract,
    *,
    decision_by_contact: dict[str, AtlasInteractionDecision],
) -> tuple[tuple[AtlasCandidatePrescription, ...], dict[str, set[str]]]:
    """Keep endpoint chemistry observable while exposing the full pose chart."""

    block_families: dict[str, set[str]] = {}
    records: list[AtlasCandidatePrescription] = []
    for candidate in sonic_contract.primitive_candidates:
        source_decision = decision_by_contact.get(candidate.owner_id)
        source_decision_id = (
            "" if source_decision is None else source_decision.decision_id
        )
        if candidate.domain_id == "AUXILIARY_CONTACTS":
            if source_decision is None:
                raise ValueError(
                    f"missing EXPLORATION atlas decision for contact {candidate.owner_id}"
                )
            block_id = "EXPLORATION_CONTACT::OBSERVABLE"
            role = COORDINATE_OBSERVABLE_ONLY
            requirement = CANDIDATE_FORBIDDEN
            rule_id = "EXPLORATION.CONTACT_OBSERVABLE_ONLY"
        elif candidate.domain_id == "FRAGMENT_POSE":
            block_id = "EXPLORATION_FRAGMENT_POSE"
            role = COORDINATE_FRAGMENT_POSE_SUPPORT
            requirement = CANDIDATE_OPTIONAL
            rule_id = "EXPLORATION.DIMENSION_AWARE_POSE_SUPPORT"
        elif candidate.domain_id == "STRUCTURAL_SITES":
            block_id = "EXPLORATION_STRUCTURAL_SITE"
            role = COORDINATE_PHYSICAL_CONTACT_DISTANCE
            requirement = CANDIDATE_OPTIONAL
            rule_id = "EXPLORATION.STRUCTURAL_SITE_DISTANCE"
        else:
            block_id = f"INTERNAL::{candidate.family}"
            role = COORDINATE_INTERNAL_VALENCE
            requirement = CANDIDATE_OPTIONAL
            rule_id = "COMMON.INTERNAL_VALENCE"
        block_families.setdefault(block_id, set()).add(candidate.family)
        records.append(
            AtlasCandidatePrescription(
                candidate_id=candidate.candidate_id,
                scientific_role=role,
                requirement=requirement,
                mixing_block=block_id,
                completion_block=block_id,
                source_decision_id=source_decision_id,
                rule_id=rule_id,
            )
        )
    return tuple(records), block_families


def _minimum_family_compatibility(
    block_families: dict[str, set[str]],
) -> tuple[AtlasFamilyCompatibility, ...]:
    return tuple(
        AtlasFamilyCompatibility(
            block_id=block_id,
            task_regimes=(ATLAS_TASK_MINIMUM,),
            families=tuple(sorted(families)),
            salc_algorithm_id=(
                SALC_POINT_GROUP_PROJECTOR
                if block_id.startswith("INTERNAL::")
                or block_id == "MINIMUM_CONTACT::PSEUDOBOND"
                else SALC_NONE
            ),
            completion_algorithm_id=COMPLETION_EXACT_RANK,
            substitutions=(
                block_id.startswith("INTERNAL::")
                or block_id == "MINIMUM_CONTACT::PSEUDOBOND"
            ),
            require_same_domain=True,
            require_same_irrep=True,
            condition_trigger=(
                MINIMUM_LOCAL_DOMAIN_CONDITION_TRIGGER
                if block_id.startswith("INTERNAL::")
                or block_id == "MINIMUM_CONTACT::PSEUDOBOND"
                else None
            ),
            minimum_relative_gain=(
                MINIMUM_LOCAL_DOMAIN_RELATIVE_GAIN
                if block_id.startswith("INTERNAL::")
                or block_id == "MINIMUM_CONTACT::PSEUDOBOND"
                else None
            ),
            rule_id=(
                "FAMILY.MINIMUM_CONTACT"
                if block_id == "MINIMUM_CONTACT::PSEUDOBOND"
                else "FAMILY.MINIMUM_OBSERVABLE"
                if block_id == "MINIMUM_CONTACT::OBSERVABLE"
                else "FAMILY.MINIMUM_POSE"
                if block_id == "MINIMUM_FRAGMENT_POSE"
                else "FAMILY.INTERNAL_EXACT"
            ),
        )
        for block_id, families in sorted(block_families.items())
    )


def _exploration_family_compatibility(
    block_families: dict[str, set[str]],
) -> tuple[AtlasFamilyCompatibility, ...]:
    return tuple(
        AtlasFamilyCompatibility(
            block_id=block_id,
            task_regimes=(ATLAS_TASK_EXPLORATION,),
            families=tuple(sorted(families)),
            salc_algorithm_id=(
                SALC_POINT_GROUP_PROJECTOR
                if block_id.startswith("INTERNAL::")
                else SALC_NONE
            ),
            completion_algorithm_id=COMPLETION_EXACT_RANK,
            substitutions=block_id.startswith("INTERNAL::"),
            require_same_domain=True,
            require_same_irrep=True,
            condition_trigger=(
                MINIMUM_LOCAL_DOMAIN_CONDITION_TRIGGER
                if block_id.startswith("INTERNAL::")
                else None
            ),
            minimum_relative_gain=(
                MINIMUM_LOCAL_DOMAIN_RELATIVE_GAIN
                if block_id.startswith("INTERNAL::")
                else None
            ),
            rule_id=(
                "FAMILY.EXPLORATION_OBSERVABLE"
                if block_id == "EXPLORATION_CONTACT::OBSERVABLE"
                else "FAMILY.EXPLORATION_POSE"
                if block_id == "EXPLORATION_FRAGMENT_POSE"
                else "FAMILY.EXPLORATION_STRUCTURAL_SITE"
                if block_id == "EXPLORATION_STRUCTURAL_SITE"
                else "FAMILY.INTERNAL_EXACT"
            ),
        )
        for block_id, families in sorted(block_families.items())
    )


def _coordinate_atlas_contract(
    *,
    task_regime: str,
    topology_hash: str,
    interactions: tuple[AtlasInteractionDecision, ...],
    bodies: tuple[AtlasBodyPrescription, ...],
    candidates: tuple[AtlasCandidatePrescription, ...],
    family_compatibility: tuple[AtlasFamilyCompatibility, ...],
    local_domains: tuple[AtlasLocalDomainPrescription, ...],
    reactive_zones: tuple[AtlasReactiveZonePrescription, ...],
) -> OracleCoordinateAtlasContract:
    manifest = coordinate_atlas_policy_manifest()
    return OracleCoordinateAtlasContract(
        schema=ORACLE_COORDINATE_ATLAS_SCHEMA,
        owner=ORACLE_COORDINATE_ATLAS_OWNER,
        policy_id=ORACLE_COORDINATE_ATLAS_POLICY_ID,
        policy_version=ORACLE_COORDINATE_ATLAS_POLICY_VERSION,
        policy_sha256=str(manifest["sha256"]),
        coordinate_library_schema=COORDINATE_LIBRARY_SCHEMA,
        coordinate_library_sha256=str(coordinate_library_manifest()["sha256"]),
        salc_library_schema=SALC_LIBRARY_SCHEMA,
        salc_library_sha256=str(salc_library_manifest()["sha256"]),
        task_regime=task_regime,
        topology_hash=topology_hash,
        interactions=interactions,
        bodies=bodies,
        candidates=candidates,
        family_compatibility=family_compatibility,
        local_domains=local_domains,
        reactive_zones=reactive_zones,
        fallback_order=ATLAS_FALLBACK_ORDER,
        provenance=ORACLE_COORDINATE_ATLAS_BUILDER,
    )


def write_minimum_coordinate_atlas_contract(
    path: Path,
    sonic_contract: OracleSonicContract | None = None,
    *,
    fragment_representation: str = MINIMUM_FRAGMENT_REPRESENTATION_AUTOMATIC,
) -> OracleCoordinateAtlasContract:
    """Build and freeze the MINIMUM atlas beside the ORACLE SONIC contract."""

    source = sonic_contract or read_oracle_sonic_contract(Path(path))
    geometry = read_enriched_xyz(Path(path))
    atlas = build_minimum_coordinate_atlas_contract(
        source,
        coordinates_angstrom=np.asarray(geometry.coordinates_angstrom, dtype=float),
        fragment_representation=fragment_representation,
    )
    from .sonic_contract_builder import extend_contact_primitive_pool
    from matrix_chem import write_oracle_sonic_contract
    source = extend_contact_primitive_pool(source, atlas, np.asarray(geometry.coordinates_angstrom, dtype=float))
    write_oracle_sonic_contract(Path(path), source)
    atlas = build_minimum_coordinate_atlas_contract(source,
        coordinates_angstrom=np.asarray(geometry.coordinates_angstrom, dtype=float),
        fragment_representation=fragment_representation)
    write_oracle_coordinate_atlas_contract(Path(path), atlas)
    return atlas


def write_exploration_coordinate_atlas_contract(
    path: Path,
    sonic_contract: OracleSonicContract | None = None,
) -> OracleCoordinateAtlasContract:
    """Freeze an EXPLORATION atlas without changing endpoint perception."""

    source = sonic_contract or read_oracle_sonic_contract(Path(path))
    geometry = read_enriched_xyz(Path(path))
    atlas = build_exploration_coordinate_atlas_contract(
        source,
        coordinates_angstrom=np.asarray(geometry.coordinates_angstrom, dtype=float),
    )
    write_oracle_coordinate_atlas_contract(Path(path), atlas)
    return atlas


def _validate_ts_atlas_inputs(
    transition_state_contract: OracleTransitionStateGeometryContract,
    sonic_contract: OracleSonicContract | None,
) -> None:
    validate_oracle_transition_state_geometry_contract(transition_state_contract)
    if sonic_contract is None:
        return
    validate_oracle_sonic_contract(sonic_contract)
    if (
        sonic_contract.primary_topology.topology_hash
        != transition_state_contract.topology_hash
    ):
        raise ValueError("ORACLE SONIC and TS contracts have different topology hashes")


def _ts_pair_maps(
    transition_state_contract: OracleTransitionStateGeometryContract,
    sonic_contract: OracleSonicContract | None,
):
    prescribed_by_pair = {
        tuple(sorted(record.atoms)): record
        for record in transition_state_contract.prescribed_pseudobonds
    }
    distance_only_pairs = _descriptor_pairs(
        transition_state_descriptor(
            transition_state_contract,
            "DISTANCE_ONLY_KERNEL_EDGES",
            "NONE",
        )
    )
    candidate_ids_by_pair: dict[tuple[int, int], tuple[str, ...]] = {}
    for candidate in (() if sonic_contract is None else sonic_contract.primitive_candidates):
        if candidate.function != "R" or len(candidate.atoms) != 2:
            continue
        pair = tuple(sorted((int(candidate.atoms[0]), int(candidate.atoms[1]))))
        candidate_ids_by_pair[pair] = (
            *candidate_ids_by_pair.get(pair, ()),
            candidate.candidate_id,
        )
    return prescribed_by_pair, distance_only_pairs, candidate_ids_by_pair


def _ts_kernel_decision(
    edge,
    *,
    decision_id: str,
    prescribed,
    is_distance_only: bool,
    candidate_ids: tuple[str, ...],
) -> AtlasInteractionDecision:
    pair = tuple(sorted(edge.atoms))
    is_operative = edge.role == "BREAKING" or prescribed is not None or is_distance_only
    coordinate_role = (
        COORDINATE_REACTION_DISTANCE_ONLY
        if is_distance_only
        else COORDINATE_OBSERVABLE_ONLY
        if not is_operative
        else COORDINATE_REACTION_DISTANCE
    )
    return AtlasInteractionDecision(
        decision_id=decision_id,
        interaction_id=f"TS_KERNEL:{pair[0]}-{pair[1]}:{edge.role}",
        semantic_kind=(
            prescribed.kind if prescribed is not None else f"TS_{edge.role}_{edge.kind}"
        ),
        endpoint_a=("ATOM", str(pair[0])),
        endpoint_b=("ATOM", str(pair[1])),
        fragment_ids=(),
        graph_role=(
            GRAPH_ROLE_REACTIVE_SUPPORT if is_operative else GRAPH_ROLE_NONE
        ),
        delta_beta1=0,
        coordinate_role=coordinate_role,
        pseudobond_policy=(
            PSEUDOBOND_REQUIRED
            if prescribed is not None and not is_distance_only
            else PSEUDOBOND_FORBIDDEN
        ),
        primitive_candidate_ids=candidate_ids,
        rule_id=(
            "TS.LOCAL_KERNEL_PSEUDOBOND"
            if prescribed is not None and not is_distance_only
            else "TS.KERNEL_ALTERNATIVE_OBSERVABLE"
            if not is_operative
            else "TS.LOCAL_KERNEL_DISTANCE_ONLY"
            if is_distance_only
            else "TS.LOCAL_KERNEL_BREAKING_DISTANCE"
        ),
        evidence=(edge.provenance, f"PRIORITY={edge.priority}"),
    )


def _ts_interaction_decisions(
    transition_state_contract: OracleTransitionStateGeometryContract,
    *,
    prescribed_by_pair,
    distance_only_pairs: frozenset[tuple[int, int]],
    candidate_ids_by_pair: dict[tuple[int, int], tuple[str, ...]],
) -> tuple[tuple[AtlasInteractionDecision, ...], dict[tuple[int, int], str]]:
    interactions: list[AtlasInteractionDecision] = []
    decision_by_pair: dict[tuple[int, int], str] = {}
    for edge in transition_state_contract.reaction_kernel:
        pair = tuple(sorted(edge.atoms))
        decision_id = f"D{len(interactions) + 1:04d}"
        decision_by_pair[pair] = decision_id
        interactions.append(
            _ts_kernel_decision(
                edge,
                decision_id=decision_id,
                prescribed=prescribed_by_pair.get(pair),
                is_distance_only=pair in distance_only_pairs,
                candidate_ids=candidate_ids_by_pair.get(pair, ()),
            )
        )
    for pair, record in sorted(prescribed_by_pair.items()):
        if pair in decision_by_pair:
            continue
        decision_id = f"D{len(interactions) + 1:04d}"
        decision_by_pair[pair] = decision_id
        interactions.append(
            AtlasInteractionDecision(
                decision_id=decision_id,
                interaction_id=f"TS_SUPPORT:{pair[0]}-{pair[1]}",
                semantic_kind=record.kind,
                endpoint_a=("ATOM", str(pair[0])),
                endpoint_b=("ATOM", str(pair[1])),
                fragment_ids=(),
                graph_role=GRAPH_ROLE_REACTIVE_SUPPORT,
                delta_beta1=0,
                coordinate_role=COORDINATE_TS_SUPPORT,
                pseudobond_policy=PSEUDOBOND_REQUIRED,
                primitive_candidate_ids=candidate_ids_by_pair.get(pair, ()),
                rule_id="TS.EXPLICIT_SUPPORT",
                evidence=(record.provenance, f"PRIORITY={record.priority}"),
            )
        )
    if not transition_state_contract.reaction_kernel:
        interactions.append(_ts_unresolved_kernel_decision(transition_state_contract))
    return tuple(interactions), decision_by_pair


def _ts_reactive_zones(
    interactions: tuple[AtlasInteractionDecision, ...],
) -> tuple[AtlasReactiveZonePrescription, ...]:
    """Freeze the atoms whose local topology changes in the TS kernel."""

    operative = tuple(
        item
        for item in interactions
        if (
            item.coordinate_role
            in {COORDINATE_REACTION_DISTANCE, COORDINATE_REACTION_DISTANCE_ONLY}
            or any(
                evidence.startswith("ORACLE_TS_LIFECYCLE_KERNEL_ANCHOR@")
                for evidence in item.evidence
            )
        )
        and item.endpoint_a[0] == item.endpoint_b[0] == "ATOM"
    )
    if not operative:
        return ()
    atoms = tuple(
        sorted(
            {
                int(endpoint[1])
                for item in operative
                for endpoint in (item.endpoint_a, item.endpoint_b)
            }
        )
    )
    return (
        AtlasReactiveZonePrescription(
            zone_id="TS_REACTIVE_ZONE",
            atoms=atoms,
            source_interaction_ids=tuple(item.interaction_id for item in operative),
            excluded_primitive_functions=("L",),
            rule_id="TS.REACTIVE_ZONE_NO_LINEAR_BEND",
        ),
    )


def _ts_unresolved_kernel_decision(
    transition_state_contract: OracleTransitionStateGeometryContract,
) -> AtlasInteractionDecision:
    return AtlasInteractionDecision(
        decision_id="D0001",
        interaction_id="TS_KERNEL_STATUS",
        semantic_kind="NO_RESOLVED_LOCAL_KERNEL",
        endpoint_a=("SYSTEM", "SYSTEM"),
        endpoint_b=("SYSTEM", "SYSTEM"),
        fragment_ids=(),
        graph_role=GRAPH_ROLE_NONE,
        delta_beta1=0,
        coordinate_role=COORDINATE_OBSERVABLE_ONLY,
        pseudobond_policy=PSEUDOBOND_FORBIDDEN,
        primitive_candidate_ids=(),
        rule_id="TS.NO_RESOLVED_LOCAL_KERNEL",
        evidence=(
            f"CATEGORY={transition_state_contract.category_id}",
            "TASK_REMAINS_TRANSITION_STATE",
        ),
    )


def _ts_candidate_prescriptions(
    sonic_contract: OracleSonicContract | None,
    *,
    interactions: tuple[AtlasInteractionDecision, ...],
    decision_by_pair: dict[tuple[int, int], str],
    atlas_bodies: tuple[AtlasBodyPrescription, ...],
) -> tuple[tuple[AtlasCandidatePrescription, ...], dict[str, set[str]]]:
    block_families: dict[str, set[str]] = {
        "TS_RADIAL": {"STRETCH", "TS_REACTION_DISTANCE"},
        "TS_REACTIVE_COMPLETION": {
            "FRAG_DISTANCE",
            "OUT_OF_PLANE",
            "PSEUDO_BOND_BEND",
            "PSEUDO_BOND_TORSION",
        },
    }
    records: list[AtlasCandidatePrescription] = []
    decisions = {item.decision_id: item for item in interactions}
    for candidate in (() if sonic_contract is None else sonic_contract.primitive_candidates):
        pair = (
            tuple(sorted((int(candidate.atoms[0]), int(candidate.atoms[1]))))
            if candidate.function == "R" and len(candidate.atoms) == 2
            else ()
        )
        if pair in decision_by_pair:
            decision = decisions[decision_by_pair[pair]]
            role = decision.coordinate_role
            block_id = (
                "TS_SUPPORT"
                if role == COORDINATE_TS_SUPPORT
                else "TS_ALTERNATIVE_OBSERVABLE"
                if role == COORDINATE_OBSERVABLE_ONLY
                else "TS_RADIAL"
            )
            requirement = (
                CANDIDATE_FORBIDDEN
                if role == COORDINATE_OBSERVABLE_ONLY
                else CANDIDATE_REQUIRED
            )
            source_decision_id, rule_id = decision.decision_id, decision.rule_id
        elif candidate.domain_id == "FRAGMENT_POSE":
            if not atlas_bodies:
                continue
            role, block_id = COORDINATE_FRAGMENT_POSE_SUPPORT, "TS_FRAGMENT_POSE"
            requirement, source_decision_id, rule_id = (
                CANDIDATE_OPTIONAL,
                "",
                "TS.POSE_SUPPORT",
            )
        else:
            role = COORDINATE_INTERNAL_VALENCE
            block_id = (
                "TS_RADIAL"
                if candidate.family == "STRETCH"
                else f"INTERNAL::{candidate.family}"
            )
            requirement, source_decision_id, rule_id = (
                CANDIDATE_OPTIONAL,
                "",
                "COMMON.INTERNAL_VALENCE",
            )
        block_families.setdefault(block_id, set()).add(candidate.family)
        records.append(
            AtlasCandidatePrescription(
                candidate_id=candidate.candidate_id,
                scientific_role=role,
                requirement=requirement,
                mixing_block=block_id,
                completion_block=block_id,
                source_decision_id=source_decision_id,
                rule_id=rule_id,
            )
        )
    return tuple(records), block_families


def _ts_family_compatibility(
    block_families: dict[str, set[str]],
) -> tuple[AtlasFamilyCompatibility, ...]:
    return tuple(
        AtlasFamilyCompatibility(
            block_id=block_id,
            task_regimes=(ATLAS_TASK_TRANSITION_STATE,),
            families=tuple(sorted(families)),
            salc_algorithm_id=(
                SALC_PROJECTOR_THEN_B_ORTHOGONAL
                if block_id == "TS_REACTIVE_COMPLETION"
                or block_id.startswith("INTERNAL::")
                else SALC_B_ORTHOGONAL
                if block_id == "TS_RADIAL"
                else SALC_NONE
            ),
            completion_algorithm_id=COMPLETION_EXACT_RANK,
            substitutions=(
                block_id == "TS_RADIAL"
                or block_id == "TS_REACTIVE_COMPLETION"
                or block_id.startswith("INTERNAL::")
            ),
            require_same_domain=(block_id != "TS_RADIAL"),
            require_same_irrep=True,
            condition_trigger=(
                TS_RADIAL_CONDITION_TRIGGER
                if block_id == "TS_RADIAL"
                or block_id == "TS_REACTIVE_COMPLETION"
                or block_id.startswith("INTERNAL::")
                else None
            ),
            minimum_relative_gain=(
                TS_RADIAL_MINIMUM_RELATIVE_GAIN
                if block_id == "TS_RADIAL"
                or block_id == "TS_REACTIVE_COMPLETION"
                or block_id.startswith("INTERNAL::")
                else None
            ),
            rule_id=(
                "FAMILY.TS_RADIAL_LOCALITY"
                if block_id == "TS_RADIAL"
                else "FAMILY.TS_REACTIVE_LOCAL_COMPLETION"
                if block_id == "TS_REACTIVE_COMPLETION"
                else "FAMILY.TS_SUPPORT"
                if block_id == "TS_SUPPORT"
                else "FAMILY.TS_POSE"
                if block_id == "TS_FRAGMENT_POSE"
                else "FAMILY.INTERNAL_EXACT"
            ),
        )
        for block_id, families in sorted(block_families.items())
    )


def build_transition_state_coordinate_atlas_contract(
    transition_state_contract: OracleTransitionStateGeometryContract,
    sonic_contract: OracleSonicContract | None = None,
    *,
    coordinates_angstrom: np.ndarray | None = None,
    fragments: tuple[FragmentMembership, ...] = (),
) -> OracleCoordinateAtlasContract:
    """Build a TS-only atlas from ORACLE's frozen local reaction kernel."""

    _validate_ts_atlas_inputs(transition_state_contract, sonic_contract)
    prescribed_by_pair, distance_only_pairs, candidate_ids_by_pair = _ts_pair_maps(
        transition_state_contract,
        sonic_contract,
    )
    interactions, decision_by_pair = _ts_interaction_decisions(
        transition_state_contract,
        prescribed_by_pair=prescribed_by_pair,
        distance_only_pairs=distance_only_pairs,
        candidate_ids_by_pair=candidate_ids_by_pair,
    )
    has_reactive_pseudobond = any(
        item.pseudobond_policy == PSEUDOBOND_REQUIRED for item in interactions
    )
    atlas_bodies = (
        ()
        if has_reactive_pseudobond
        else _body_prescriptions(
            sonic_contract.primary_topology.fragments
            if sonic_contract is not None
            else fragments,
            coordinates_angstrom,
        )
    )
    candidates, block_families = _ts_candidate_prescriptions(
        sonic_contract,
        interactions=interactions,
        decision_by_pair=decision_by_pair,
        atlas_bodies=atlas_bodies,
    )
    contract = _coordinate_atlas_contract(
        task_regime=ATLAS_TASK_TRANSITION_STATE,
        topology_hash=transition_state_contract.topology_hash,
        interactions=interactions,
        bodies=atlas_bodies,
        candidates=candidates,
        family_compatibility=_ts_family_compatibility(block_families),
        local_domains=(),
        reactive_zones=_ts_reactive_zones(interactions),
    )
    validate_oracle_coordinate_atlas_contract(contract)
    return contract


def write_transition_state_coordinate_atlas_contract(
    path: Path,
    transition_state_contract: OracleTransitionStateGeometryContract,
    sonic_contract: OracleSonicContract | None = None,
) -> OracleCoordinateAtlasContract:
    geometry = read_enriched_xyz(Path(path))
    snapshot = topology_snapshot_from_xyzin(Path(path))
    fragments = tuple(
        FragmentMembership(f"F{index:03d}", tuple(int(atom) for atom in atoms))
        for index, atoms in enumerate(snapshot["fragments"], start=1)
    )
    atlas = build_transition_state_coordinate_atlas_contract(
        transition_state_contract,
        sonic_contract,
        coordinates_angstrom=np.asarray(geometry.coordinates_angstrom, dtype=float),
        fragments=fragments,
    )
    if sonic_contract is not None:
        from .sonic_contract_builder import extend_reactive_primitive_pool
        from matrix_chem import write_oracle_sonic_contract
        sonic_contract = extend_reactive_primitive_pool(
            sonic_contract, atlas, np.asarray(geometry.coordinates_angstrom, dtype=float))
        write_oracle_sonic_contract(Path(path), sonic_contract)
        atlas = build_transition_state_coordinate_atlas_contract(
            transition_state_contract, sonic_contract,
            coordinates_angstrom=np.asarray(geometry.coordinates_angstrom, dtype=float),
            fragments=fragments)
    write_oracle_coordinate_atlas_contract(Path(path), atlas)
    return atlas


def _descriptor_pairs(text: str) -> frozenset[tuple[int, int]]:
    value = str(text).strip().upper()
    if not value or value == "NONE":
        return frozenset()
    pairs = set()
    for item in value.split(","):
        try:
            left, right = (int(atom) for atom in item.split("-", 1))
        except ValueError as exc:
            raise ValueError(f"invalid TS atlas pair descriptor: {item}") from exc
        pairs.add(tuple(sorted((left, right))))
    return frozenset(pairs)


def _body_prescriptions(
    fragments: tuple[FragmentMembership, ...],
    coordinates_angstrom: np.ndarray | None,
) -> tuple[AtlasBodyPrescription, ...]:
    """Freeze body dimensions and the quaternion pose chart in ORACLE."""

    if coordinates_angstrom is None:
        return ()
    coords = np.asarray(coordinates_angstrom, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3 or np.any(~np.isfinite(coords)):
        raise ValueError("atlas body coordinates must be finite (natoms, 3)")
    records = tuple(fragments) or (
        FragmentMembership("F001", tuple(range(1, len(coords) + 1))),
    )
    membership = tuple(atom for record in records for atom in record.atoms)
    if set(membership) != set(range(1, len(coords) + 1)) or len(membership) != len(
        set(membership)
    ):
        raise ValueError("atlas body partition must cover every atom exactly once")

    dimensions: dict[str, str] = {}
    dimension_priority = {"POINT": 0, "LINEAR": 1, "NONLINEAR": 2}
    for record in records:
        local = coords[[atom - 1 for atom in record.atoms]]
        dimension = (
            "POINT"
            if len(record.atoms) == 1
            else "LINEAR"
            if is_linear_geometry(
                local,
                tolerance=BODY_LINEAR_RELATIVE_SINGULAR_THRESHOLD,
            )
            else "NONLINEAR"
        )
        dimensions[record.fragment_id] = dimension
    reference = min(
        records,
        key=lambda record: (
            -dimension_priority[dimensions[record.fragment_id]],
            -len(record.atoms),
            record.fragment_id,
        ),
    )
    reference_dimension = dimensions[reference.fragment_id]

    def pose_chart(record: FragmentMembership) -> str:
        dimension = dimensions[record.fragment_id]
        if record.fragment_id == reference.fragment_id:
            return "NONE" if dimension == "POINT" else "QUATERNION"
        if reference_dimension == "LINEAR":
            return (
                "AXIAL_JACOBI"
                if dimension == "POINT"
                else "AXIS_AXIS_STEREOGRAPHIC"
            )
        return "NONE" if dimension == "POINT" else "QUATERNION"

    return tuple(
        AtlasBodyPrescription(
            body_id=record.fragment_id,
            atoms=record.atoms,
            dimension=dimensions[record.fragment_id],
            translation_dofs=3,
            orientation_dofs={"POINT": 0, "LINEAR": 2, "NONLINEAR": 3}[
                dimensions[record.fragment_id]
            ],
            pose_chart=pose_chart(record),
            pose_role=(
                "REFERENCE" if record.fragment_id == reference.fragment_id else "MOVING"
            ),
            rule_id=f"BODY.{dimensions[record.fragment_id]}_DIMENSION_AWARE_POSE",
        )
        for record in sorted(records, key=lambda item: item.fragment_id)
    )


__all__ = [
    "MINIMUM_FRAGMENT_REPRESENTATION_AUTOMATIC",
    "MINIMUM_FRAGMENT_REPRESENTATION_PSEUDO_BONDS",
    "MINIMUM_FRAGMENT_REPRESENTATION_QUATERNION",
    "MINIMUM_FRAGMENT_REPRESENTATIONS",
    "ORACLE_COORDINATE_ATLAS_BUILDER",
    "ORACLE_COORDINATE_ATLAS_POLICY_ID",
    "ORACLE_COORDINATE_ATLAS_POLICY_VERSION",
    "build_exploration_coordinate_atlas_contract",
    "build_minimum_coordinate_atlas_contract",
    "build_transition_state_coordinate_atlas_contract",
    "coordinate_atlas_policy_manifest",
    "write_exploration_coordinate_atlas_contract",
    "write_minimum_coordinate_atlas_contract",
    "write_transition_state_coordinate_atlas_contract",
]
