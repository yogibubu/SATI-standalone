"""Frozen multilevel initialization protocol shared by CLI and Keymaker."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import resources
import json
from typing import Any, Mapping


INITIALIZATION_PROTOCOL_SCHEMA = "matrix.link.initialization_protocol_manifest.v1"
INITIALIZATION_PROTOCOL_ID = "matrix-multilevel-initialization-v1"
INITIALIZATION_PROTOCOL_VERSION = "1.2.0"


@dataclass(frozen=True)
class InitializationProtocol:
    payload: dict[str, Any]
    sha256: str
    source: str

    def route(
        self,
        *,
        geometry_status: str,
        method_class: str,
        xtb_available: bool,
    ) -> tuple[dict[str, str], ...]:
        # ``xtb_available`` is retained in the request for provenance and
        # backwards-compatible deserialization only.  GFN2 initialization is
        # always supplied by the resident MATRIX-GFN2 calculator; an external
        # xTB executable must never change the scientific route or provide a
        # fallback.  Keep the manifest key for schema compatibility while
        # selecting the autonomous route unconditionally.
        availability = "xtb_available"
        try:
            stages = self.payload["routes"][availability][geometry_status][method_class]
        except KeyError as exc:
            raise ValueError(
                "initialization route requires geometry_status GOOD_UNCHANGED or "
                "PREOPTIMIZE and method_class low_cost or higher_level"
            ) from exc
        return tuple(dict(stage) for stage in stages)


def load_initialization_protocol() -> InitializationProtocol:
    resource = resources.files("matrix_link").joinpath(
        "data/initialization_protocol_manifest.json"
    )
    raw = resource.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("MATRIX initialization protocol manifest is unreadable") from exc
    validate_initialization_protocol(payload)
    return InitializationProtocol(
        payload=dict(payload),
        sha256=hashlib.sha256(raw).hexdigest(),
        source="matrix_link:data/initialization_protocol_manifest.json",
    )


def validate_initialization_protocol(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != INITIALIZATION_PROTOCOL_SCHEMA:
        raise RuntimeError("unsupported MATRIX initialization protocol schema")
    if (
        payload.get("protocol_id") != INITIALIZATION_PROTOCOL_ID
        or payload.get("manifest_version") != INITIALIZATION_PROTOCOL_VERSION
        or payload.get("status") != "frozen"
    ):
        raise RuntimeError("MATRIX initialization protocol must be frozen version 1.2.0")
    if payload.get("owners") != {
        "initial_geometry_routing": "KEYMAKER",
        "connectivity_seed": "SWITCH",
        "geometry_quality_and_symmetry": "ORACLE",
        "coordinate_definition": "SMITH",
        "optimization_and_hessian_projection": "LINK",
        "electronic_state_and_scf_reuse": "APOC",
        "uff_and_future_zaff": "ARCHITECT",
        "backend_capability_resolution": "MATRIX_QM",
        "human_orchestration": "KEYMAKER",
    }:
        raise RuntimeError("MATRIX initialization owner boundaries changed")
    if payload.get("algorithm_governance") != {
        "inspect_existing_implementation_before_any_change": True,
        "reuse_existing_algorithms_and_shared_components": "mandatory",
        "parallel_reimplementation_of_existing_capability": "forbidden",
        "molecule_specific_patches": False,
        "backend_specific_workflow_patches": False,
        "method_specific_workflow_patches": False,
        "single_shared_cli_and_keymaker_entry_point": True,
        "change_policy": "new_manifest_version_and_explicit_user_confirmation",
    }:
        raise RuntimeError("MATRIX initialization algorithm governance changed")
    quality = payload.get("geometry_quality", {})
    if quality.get("schema") != "matrix.oracle.initial_geometry_quality.v1":
        raise RuntimeError("MATRIX initialization geometry-quality contract changed")
    if quality.get("source_or_molecule_specific_thresholds") is not False:
        raise RuntimeError("source- or molecule-specific geometry thresholds are forbidden")
    if quality.get("preoptimization_handoff") != (
        "KEYMAKER_routes_noncredible_geometry_to_LINK_MATRIX_GFN2_before_ORACLE_and_SMITH"
    ):
        raise RuntimeError("Cartesian preoptimization must precede ORACLE and SMITH")
    classification = payload.get("method_classification", {})
    if classification != {
        "low_cost": "explicit_verified_method_capability_class_including_GFN_xTB",
        "higher_level": "explicit_verified_ab_initio_DFT_or_wavefunction_method_class",
        "backend_identity_does_not_determine_method_class": True,
        "unknown_class": "fail_closed_and_require_explicit_selection",
    }:
        raise RuntimeError("MATRIX initialization method classification changed")
    expected_routes = {
        "xtb_available": {
            "GOOD_UNCHANGED": {
                "low_cost": (("GFN-xTB", "hessian_at_unchanged_geometry"), ("FINAL", "optimize_with_previous_hessian")),
                "higher_level": (("GFN-xTB", "hessian_at_unchanged_geometry"), ("FINAL", "optimize_with_previous_hessian")),
            },
            "PREOPTIMIZE": {
                "low_cost": (("GFN-xTB", "optimize_and_hessian"), ("FINAL", "optimize_with_previous_hessian")),
                "higher_level": (("GFN-xTB", "optimize_and_hessian"), ("FINAL", "optimize_with_previous_hessian")),
            },
        },
        "xtb_unavailable": {
            "GOOD_UNCHANGED": {
                "low_cost": (("GFN-xTB", "hessian_at_unchanged_geometry"), ("FINAL", "optimize_with_previous_hessian")),
                "higher_level": (("GFN-xTB", "hessian_at_unchanged_geometry"), ("FINAL", "optimize_with_previous_hessian")),
            },
            "PREOPTIMIZE": {
                "low_cost": (("GFN-xTB", "optimize_and_hessian"), ("FINAL", "optimize_with_previous_hessian")),
                "higher_level": (("GFN-xTB", "optimize_and_hessian"), ("FINAL", "optimize_with_previous_hessian")),
            },
        },
    }
    observed_routes = _normalized_routes(payload.get("routes", {}))
    if observed_routes != expected_routes:
        raise RuntimeError("MATRIX initialization routes changed")
    providers = payload.get("providers", {})
    if set(providers) != {"GFN-FF", "GFN-xTB", "UFF", "HF/STO-3G", "FINAL"}:
        raise RuntimeError("MATRIX initialization provider set changed")
    if providers["UFF"].get("owner") != "ARCHITECT":
        raise RuntimeError("UFF must remain owned by ARCHITECT")
    if providers["GFN-FF"].get("owner") != "ARCHITECT_xTB_adapter":
        raise RuntimeError("GFN-FF initial refinement must pass through ARCHITECT")
    if providers["GFN-xTB"].get("owner") != "MATRIX_GFN2_resident_calculator":
        raise RuntimeError("GFN-xTB initialization must use autonomous MATRIX-GFN2")
    if providers["HF/STO-3G"].get("resolution") != "MATRIX_QM_verified_capability_registry":
        raise RuntimeError("HF/STO-3G must use the verified MATRIX_QM registry")
    stage = payload.get("stage_contract", {})
    required_stage_keys = {
        "symmetry",
        "coordinates",
        "optimizer",
        "hessian_timing",
        "initial_guess_transform",
        "final_method",
        "provenance",
        "generated_geometry_handoff",
    }
    if set(stage) != required_stage_keys:
        raise RuntimeError("MATRIX initialization stage contract changed")
    if stage.get("initial_guess_transform") != "linear_congruence_to_SONIC_without_B_prime":
        raise RuntimeError("initial Hessian transformation must omit B-prime")
    if stage.get("generated_geometry_handoff") != (
        "SWITCH_or_LCB26_seed_to_KEYMAKER_LINK_MATRIX_GFN2_Cartesian_"
        "preoptimization_to_ORACLE_to_SMITH_to_LINK"
    ):
        raise RuntimeError("generated geometries must pass through the canonical owner chain")
    future = payload.get("future_replacement", {})
    if future != {
        "ZAFF": "replace_the_GFN-FF_or_UFF_low_layer_without_changing_route_semantics",
        "workflow_change_required": False,
    }:
        raise RuntimeError("ZAFF replacement contract changed")


def _normalized_routes(routes: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for availability, geometry_routes in routes.items():
        normalized[availability] = {}
        for geometry_status, method_routes in geometry_routes.items():
            normalized[availability][geometry_status] = {}
            for method_class, stages in method_routes.items():
                normalized[availability][geometry_status][method_class] = tuple(
                    (str(stage.get("provider")), str(stage.get("action")))
                    for stage in stages
                )
    return normalized


__all__ = [
    "INITIALIZATION_PROTOCOL_ID",
    "INITIALIZATION_PROTOCOL_SCHEMA",
    "INITIALIZATION_PROTOCOL_VERSION",
    "InitializationProtocol",
    "load_initialization_protocol",
    "validate_initialization_protocol",
]
