"""Versioned authority for the new canonical MATRIX QM hierarchy.

The packaged JSON resource is intentionally separate from all historical L0/L1
and double-hybrid namespaces.  Callers may inspect or resolve it, but they may
not provide a runtime replacement for the approved scientific manifest.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from importlib import resources
import json
from pathlib import Path
from typing import Any, Mapping


CANONICAL_QM_LADDER_SCHEMA = "matrix.qm.canonical_ladder.v1"
CANONICAL_QM_LADDER_ID = "matrix-canonical-qm-ladder-v1"
CANONICAL_QM_LADDER_VERSION = "1.0.11"

_LEVEL_IDS = ("PRE0", "L0", "L1", "L2", "L3", "L4")
_PROFILE_IDS = ("DL2", "FL4", "PL4")


@dataclass(frozen=True)
class CanonicalQMLadder:
    """Validated immutable handle to the packaged scientific ladder."""

    payload: dict[str, Any]
    sha256: str
    source: str

    @property
    def protocol_id(self) -> str:
        return str(self.payload["protocol_id"])

    @property
    def manifest_version(self) -> str:
        return str(self.payload["manifest_version"])

    @property
    def production_ready(self) -> bool:
        return bool(self.payload["production_ready"])

    def level(self, level_id: str) -> dict[str, Any]:
        requested = str(level_id).strip().upper()
        for record in self.payload["levels"]:
            if record["id"] == requested:
                return deepcopy(record)
        raise ValueError(f"unknown canonical MATRIX QM level: {level_id}")

    def profile(self, profile_id: str) -> dict[str, Any]:
        requested = str(profile_id).strip().upper()
        for record in self.payload["profiles"]:
            if record["id"] == requested:
                return deepcopy(record)
        raise ValueError(f"unknown explicit MATRIX QM profile: {profile_id}")

    def variant(
        self,
        level_id: str,
        *,
        electronic_state: str,
        spin: str,
    ) -> dict[str, Any]:
        state = str(electronic_state).strip().casefold()
        spin_key = str(spin).strip().casefold()
        if state not in {"ground", "excited"}:
            raise ValueError("electronic_state must be ground or excited")
        if spin_key not in {"closed_shell", "open_shell"}:
            raise ValueError("spin must be closed_shell or open_shell")
        level = self.level(level_id)
        candidates = [
            item
            for item in level["variants"]
            if item["electronic_state"] == state and item["spin"] in {spin_key, "any"}
        ]
        if not candidates:
            raise RuntimeError(
                f"canonical level {level['id']} has no declared {state}/{spin_key} variant"
            )
        candidates.sort(key=lambda item: item["spin"] == "any")
        return deepcopy(candidates[0])

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(self.payload)


def load_canonical_qm_ladder() -> CanonicalQMLadder:
    """Load the sole packaged canonical ladder and validate it fail-closed."""

    resource = resources.files("matrix_qm").joinpath("data/canonical_qm_ladder.json")
    raw = resource.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("canonical MATRIX QM ladder is unreadable") from exc
    validate_canonical_qm_ladder(payload)
    return CanonicalQMLadder(
        payload=deepcopy(payload),
        sha256=hashlib.sha256(raw).hexdigest(),
        source="matrix_qm:data/canonical_qm_ladder.json",
    )


def canonical_qm_ladder_schema_path() -> Path:
    return Path(
        str(resources.files("matrix_qm").joinpath("schemas/canonical-qm-ladder-v1.schema.json"))
    )


def validate_canonical_qm_ladder(payload: Mapping[str, Any]) -> None:
    """Enforce scientific invariants independently of optional jsonschema."""

    if payload.get("schema") != CANONICAL_QM_LADDER_SCHEMA:
        raise RuntimeError("unsupported canonical MATRIX QM ladder schema")
    if (
        payload.get("protocol_id") != CANONICAL_QM_LADDER_ID
        or payload.get("manifest_version") != CANONICAL_QM_LADDER_VERSION
        or payload.get("status") != "approved"
        or payload.get("production_ready") is not False
    ):
        raise RuntimeError("canonical MATRIX QM ladder identity or status changed")

    expected_owners = {
        "scientific_resolution": "MATRIX_QM",
        "geometry_quality_and_symmetry": "ORACLE",
        "coordinate_definition": "SMITH",
        "optimization_and_hessians": "LINK",
        "electronic_state_identity": "APOC",
        "structural_preparation": "ARCHITECT",
        "human_orchestration": "KEYMAKER",
    }
    if payload.get("owners") != expected_owners:
        raise RuntimeError("canonical MATRIX QM owner boundaries changed")

    expected_governance = {
        "legacy_manifests_unchanged": True,
        "silent_substitution": False,
        "molecule_specific_patches": False,
        "backend_specific_optimizer_patches": False,
        "method_specific_optimizer_patches": False,
        "external_qm_source_modified": False,
        "change_policy": "new_manifest_version_and_explicit_approval",
    }
    if payload.get("governance") != expected_governance:
        raise RuntimeError("canonical MATRIX QM governance changed")

    if payload.get("optimizer_contract") != {
        "protocol_id": "link-sonic-optimizer-v2",
        "manifest_version": "2.15.9",
        "settings_duplicated": False,
    }:
        raise RuntimeError("canonical ladder must reference, not duplicate, LINK")

    applicability = payload.get("applicability", {})
    if applicability.get("outside_domain_policy") != (
        "fail_closed_without_method_or_basis_substitution"
    ):
        raise RuntimeError("canonical applicability must fail closed")
    if applicability.get("l2_open_shell_derivative_policy") != {
        "status": "validation_only_energy_only",
        "domain": "high_spin_single_reference_ground_state",
        "reference": "semicanonical_ROHF",
        "gradient": "LINK_one_sided_numerical_energy",
        "formal_stability_analysis": "unavailable_in_ORCA",
        "unrestricted_substitution": False,
        "activation_policy": "exact_energy_and_reference_continuity_certification",
    }:
        raise RuntimeError("canonical L2 open-shell derivative policy changed")

    preparation = payload.get("preparation", {})
    if preparation.get("all_initial_geometries_symmetrized") is not True:
        raise RuntimeError("ORACLE must symmetrize every initial geometry")
    if preparation.get("credible_geometry_preoptimization") is not False:
        raise RuntimeError("credible geometry must bypass structural preoptimization")
    if preparation.get("ordered_levels", ())[:2] != ["Lindh_1995_ANC_plus_Swart_2006_special", "Almloef"]:
        raise RuntimeError("canonical preparation must audit Lindh-1995 ANC plus Swart-special before Almlöf")
    if preparation.get("l0_without_xtb_seed") != ["Lindh_1995_ANC_plus_Swart_2006_special", "Almloef", "UFF"]:
        raise RuntimeError("L0 without xTB must not use L0 as its own lower-level seed")

    levels = payload.get("levels")
    if not isinstance(levels, list) or tuple(item.get("id") for item in levels) != _LEVEL_IDS:
        raise RuntimeError("canonical MATRIX QM level order changed")
    for level in levels:
        variants = level.get("variants")
        if not isinstance(variants, list) or not variants:
            raise RuntimeError(f"canonical level {level.get('id')} has no variants")
        seen: set[tuple[str, str]] = set()
        for variant in variants:
            key = (str(variant.get("electronic_state")), str(variant.get("spin")))
            if key in seen:
                raise RuntimeError(f"duplicate canonical variant {level['id']} {key}")
            seen.add(key)
            status = variant.get("status")
            method = variant.get("method")
            reference = variant.get("reference")
            if status == "defined" and (not method or not reference):
                raise RuntimeError(f"defined canonical variant {level['id']} lacks semantics")
            if status == "unsupported" and (method is not None or reference is not None):
                raise RuntimeError(f"unsupported canonical variant {level['id']} declares a method")

    _require_variant(levels, "L2", "ground", "open_shell", "MP2", "semicanonical_ROHF")
    _require_variant(levels, "L3", "ground", "open_shell", "CCSD", "semicanonical_ROHF")
    _require_variant(
        levels,
        "L4",
        "ground",
        "open_shell",
        "CCSD(T)",
        "stable_UHF_from_L3_ROHF",
    )

    profiles = payload.get("profiles")
    if not isinstance(profiles, list) or tuple(item.get("id") for item in profiles) != _PROFILE_IDS:
        raise RuntimeError("explicit reduced-cost profile set changed")
    if any(item.get("canonical_replacement") is not False for item in profiles):
        raise RuntimeError("an explicit profile cannot replace canonical L2/L4 automatically")

    corrections = payload.get("corrections", {})
    if corrections.get("element_policy_id") != "matrix-qm-element-policy-v1":
        raise RuntimeError("canonical ladder element-policy reference changed")
    cv = corrections.get("core_valence", {})
    if cv.get("any_ecp_policy") != "unsupported_for_complete_system":
        raise RuntimeError("CV must be disabled for the complete system when any ECP is used")
    if cv.get("partial_correction") is not False:
        raise RuntimeError("partial core-valence corrections are forbidden")
    bsse = corrections.get("bsse", {})
    if (
        bsse.get("scope") != "ground_state_interaction_energy_only"
        or bsse.get("componentwise_assembly") is not True
        or bsse.get("mixed_branch_components") is not False
        or bsse.get("excited_state") != "unsupported"
    ):
        raise RuntimeError("canonical BSSE branch contract changed")

    validation = payload.get("validation_state", {})
    if validation != {
        "production_enabled": False,
        "certification_required_per_full_capability_key": True,
        "uncertified_combination": "unsupported",
    }:
        raise RuntimeError("uncertified canonical combinations must remain unsupported")


def _require_variant(
    levels: list[Mapping[str, Any]],
    level_id: str,
    electronic_state: str,
    spin: str,
    method: str,
    reference: str,
) -> None:
    level = next(item for item in levels if item.get("id") == level_id)
    variant = next(
        (
            item
            for item in level["variants"]
            if item.get("electronic_state") == electronic_state and item.get("spin") == spin
        ),
        None,
    )
    if variant is None or variant.get("method") != method or variant.get("reference") != reference:
        raise RuntimeError(f"canonical {level_id} {electronic_state}/{spin} semantics changed")




__all__ = [
    "CANONICAL_QM_LADDER_ID",
    "CANONICAL_QM_LADDER_SCHEMA",
    "CANONICAL_QM_LADDER_VERSION",
    "CanonicalQMLadder",
    "canonical_qm_ladder_schema_path",
    "load_canonical_qm_ladder",
    "validate_canonical_qm_ladder",
]
