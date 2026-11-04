"""Fail-closed realization checks for ORACLE coordinate-atlas prescriptions."""

from __future__ import annotations

from matrix_chem.coordinate_atlas_contract import (
    ATLAS_TASK_MINIMUM,
    ATLAS_TASK_TRANSITION_STATE,
    PSEUDOBOND_REQUIRED,
    OracleCoordinateAtlasContract,
)

from .contracts import GICForgeContractError
from .models import GICDefinition, GICPrimitive
from .policy import (
    FRAGMENT_MODE_PSEUDO_BONDS,
    FRAGMENT_MODE_SPECIAL_COORDINATES,
)


def apply_reactive_zone_exclusions(
    candidates: tuple[GICPrimitive, ...],
    contract: OracleCoordinateAtlasContract,
) -> tuple[tuple[GICPrimitive, ...], tuple[str, ...]]:
    """Apply ORACLE's frozen TS primitive exclusions without reinterpretation."""

    excluded_by_zone = {zone.zone_id: [] for zone in contract.reactive_zones}
    retained: list[GICPrimitive] = []
    for primitive in candidates:
        matching_zones = tuple(
            zone
            for zone in contract.reactive_zones
            if primitive.function in zone.excluded_primitive_functions
            and bool(set(primitive.atoms).intersection(zone.atoms))
        )
        if matching_zones:
            for zone in matching_zones:
                excluded_by_zone[zone.zone_id].append(primitive.identifier)
        else:
            retained.append(primitive)
    diagnostics = tuple(
        "ATLAS_REACTIVE_ZONE_EXCLUSION "
        f"ZONE={zone.zone_id} ATOMS={','.join(str(atom) for atom in zone.atoms)} "
        f"FUNCTIONS={','.join(zone.excluded_primitive_functions)} "
        f"REMOVED={','.join(excluded_by_zone[zone.zone_id]) or 'NONE'}"
        for zone in contract.reactive_zones
    )
    return tuple(retained), diagnostics


def validate_atlas_chart_realization(
    definition: GICDefinition,
    contract: OracleCoordinateAtlasContract,
) -> None:
    """Reject any SMITH chart that contradicts its ORACLE atlas gates."""

    _validate_reactive_zone_exclusions(definition, contract)
    _validate_reactive_completion_families(definition, contract)
    if contract.task_regime != ATLAS_TASK_MINIMUM:
        return
    required = tuple(
        sorted(
            tuple(sorted((int(item.endpoint_a[1]), int(item.endpoint_b[1]))))
            for item in contract.interactions
            if item.pseudobond_policy == PSEUDOBOND_REQUIRED
            and item.endpoint_a[0] == item.endpoint_b[0] == "ATOM"
        )
    )
    realized = tuple(sorted(tuple(sorted(pair)) for pair in definition.pseudo_bonds))
    pose_families = {"FRAG_TRANSLATION", "FRAG_ORIENTATION"}
    has_pose = any(gic.family in pose_families for gic in definition.gics)
    has_pseudobond = any(
        gic.family.startswith("PSEUDO_BOND") for gic in definition.gics
    )
    if required:
        _validate_pseudobond_realization(
            definition,
            required=required,
            realized=realized,
        )
        return
    if realized or has_pseudobond:
        raise GICForgeContractError(
            "SMITH introduced a MINIMUM pseudobond outside ORACLE's stable OPEN gate"
        )
    if len(contract.bodies) > 1:
        if definition.fragment_mode != FRAGMENT_MODE_SPECIAL_COORDINATES or not has_pose:
            raise GICForgeContractError(
                "SMITH did not realize ORACLE's dimension-aware quaternion pose fallback"
            )
    elif definition.fragment_mode == FRAGMENT_MODE_PSEUDO_BONDS:
        raise GICForgeContractError(
            "SMITH selected a pseudobond mode without an ORACLE prescription"
        )


def _validate_reactive_zone_exclusions(
    definition: GICDefinition,
    contract: OracleCoordinateAtlasContract,
) -> None:
    primitive_by_id = {item.identifier: item for item in definition.primitives}
    for gic in definition.gics:
        coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
        for primitive_id, coefficient in coefficients:
            if abs(float(coefficient)) <= 1.0e-14:
                continue
            primitive = primitive_by_id.get(primitive_id)
            if primitive is None:
                continue
            for zone in contract.reactive_zones:
                if (
                    primitive.function in zone.excluded_primitive_functions
                    and bool(set(primitive.atoms).intersection(zone.atoms))
                ):
                    raise GICForgeContractError(
                        f"SMITH selected {primitive.function} primitive {primitive.identifier} "
                        f"inside ORACLE reactive zone {zone.zone_id}"
                    )


def _validate_reactive_completion_families(
    definition: GICDefinition,
    contract: OracleCoordinateAtlasContract,
) -> None:
    if contract.task_regime != ATLAS_TASK_TRANSITION_STATE:
        return
    support = tuple(
        primitive
        for primitive in definition.primitives
        if primitive.refs and primitive.refs[0] == "PSEUDOBOND_CONTACT_SUPPORT"
        and primitive.family != "TS_REACTION_DISTANCE"
    )
    if not support:
        return
    allowed = {
        family
        for block in contract.family_compatibility
        if block.block_id == "TS_REACTIVE_COMPLETION"
        for family in block.families
    }
    unsupported = tuple(
        primitive for primitive in support if primitive.family not in allowed
    )
    if unsupported:
        detail = ",".join(
            f"{primitive.identifier}:{primitive.family}" for primitive in unsupported
        )
        raise GICForgeContractError(
            "SMITH generated pseudobond contact support outside ORACLE's "
            f"TS_REACTIVE_COMPLETION block: {detail}"
        )


def _validate_pseudobond_realization(
    definition: GICDefinition,
    *,
    required: tuple[tuple[int, int], ...],
    realized: tuple[tuple[int, int], ...],
) -> None:
    if definition.fragment_mode != FRAGMENT_MODE_PSEUDO_BONDS:
        raise GICForgeContractError(
            "SMITH did not realize ORACLE's stable OPEN MINIMUM pseudobond chart"
        )
    if realized != required:
        raise GICForgeContractError(
            "SMITH pseudobonds differ from ORACLE's exact MINIMUM prescription"
        )
    # Local pseudobond coordinates constrain classified contacts; relative
    # fragment-pose coordinates span the remaining rigid-body directions.
    # ORACLE may prescribe both and SMITH must preserve that direct-sum chart.


__all__ = ["apply_reactive_zone_exclusions", "validate_atlas_chart_realization"]
