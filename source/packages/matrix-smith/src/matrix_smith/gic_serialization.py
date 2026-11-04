"""Canonical JSON serialization for frozen :class:`GICDefinition` payloads.

The enriched-XYZ ``#GIC`` section remains the public human-readable contract.
Typed ONIC artifacts need to carry more than one frozen GIC payload, so this
module provides the lossless JSON form used inside that container.  It does
not rebuild coordinates or infer scientific metadata.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any, Mapping

from .models import (
    FrozenGIC,
    GICDefinition,
    GICPrimitive,
    GICReductionDiagnostics,
    GICSymmetrizationDiagnostics,
    GICSymmetrizedGroup,
)
from .fallback_ledger import fallback_event_from_dict, fallback_event_to_dict
from .periodic_estimates import PeriodicCoordinateEstimate


GIC_DEFINITION_JSON_SCHEMA = "matrix.smith.gic_definition_payload.v1"


def gic_definition_to_dict(definition: GICDefinition) -> dict[str, Any]:
    """Return a deterministic JSON-compatible frozen-GIC record."""

    if not isinstance(definition, GICDefinition):
        raise TypeError("GIC payload serialization requires a GICDefinition")
    record = json.loads(json.dumps(asdict(definition), allow_nan=False))
    record["fallback_events"] = [
        fallback_event_to_dict(event) for event in definition.fallback_events
    ]
    symmetry_record = record.get("symmetry_diagnostics")
    if isinstance(symmetry_record, dict) and definition.symmetry_diagnostics is not None:
        symmetry_record["fallback_events"] = [
            fallback_event_to_dict(event)
            for event in definition.symmetry_diagnostics.fallback_events
        ]
    return {
        "schema": GIC_DEFINITION_JSON_SCHEMA,
        "definition": record,
    }


def gic_definition_from_dict(payload: Mapping[str, Any]) -> GICDefinition:
    """Reconstruct a frozen GIC payload without scientific regeneration."""

    if not isinstance(payload, Mapping) or payload.get("schema") != GIC_DEFINITION_JSON_SCHEMA:
        raise ValueError("unsupported frozen GIC payload schema")
    try:
        record = payload["definition"]
        if not isinstance(record, Mapping):
            raise TypeError("frozen GIC definition record must be a mapping")
        primitive_source = str(record["primitive_source"]).strip()
        if not primitive_source or primitive_source == "LEGACY_RECONSTRUCTED":
            raise ValueError("frozen GIC payload has no explicit primitive source")
        primitive_source_schema = str(record["primitive_source_schema"]).strip()
        primitive_b_matrix_sha256 = str(record["primitive_b_matrix_sha256"]).strip()
        if primitive_source == "ORACLE_CONTRACT" and (
            not primitive_source_schema or not primitive_b_matrix_sha256
        ):
            raise ValueError("ORACLE primitive source metadata is incomplete")
        reduction_record = record.get("reduction_diagnostics")
        symmetry_record = record.get("symmetry_diagnostics")
        reduction = (
            None
            if reduction_record is None
            else GICReductionDiagnostics(
                rank_method=str(reduction_record["rank_method"]),
                reduction_policy=str(reduction_record["reduction_policy"]),
                selected=tuple(str(item) for item in reduction_record.get("selected", ())),
                skipped_singular=tuple(
                    str(item) for item in reduction_record.get("skipped_singular", ())
                ),
                skipped_dependent=tuple(
                    str(item) for item in reduction_record.get("skipped_dependent", ())
                ),
                selected_by_family=tuple(
                    str(item) for item in reduction_record.get("selected_by_family", ())
                ),
                skipped_singular_details=tuple(
                    str(item)
                    for item in reduction_record.get("skipped_singular_details", ())
                ),
                skipped_dependent_details=tuple(
                    str(item)
                    for item in reduction_record.get("skipped_dependent_details", ())
                ),
                conditioning_decisions=tuple(
                    str(item)
                    for item in reduction_record.get("conditioning_decisions", ())
                ),
            )
        )
        symmetry = (
            None
            if symmetry_record is None
            else GICSymmetrizationDiagnostics(
                method=str(symmetry_record["method"]),
                policy=str(symmetry_record["policy"]),
                status=str(symmetry_record["status"]),
                point_group=str(symmetry_record["point_group"]),
                symmetry_group=str(symmetry_record["symmetry_group"]),
                total_symmetric_irrep=str(symmetry_record["total_symmetric_irrep"]),
                total_symmetric_gics=tuple(
                    str(item) for item in symmetry_record.get("total_symmetric_gics", ())
                ),
                groups=tuple(
                    GICSymmetrizedGroup(
                        block=str(item["block"]),
                        family=str(item["family"]),
                        signature=str(item["signature"]),
                        source_gics=tuple(str(value) for value in item.get("source_gics", ())),
                        output_gics=tuple(str(value) for value in item.get("output_gics", ())),
                    )
                    for item in symmetry_record.get("groups", ())
                ),
                sign_gauge_policy=str(
                    symmetry_record.get("sign_gauge_policy", "largest_abs_coefficient_pivot")
                ),
                path_gauge_policy=str(
                    symmetry_record.get("path_gauge_policy", "subspace_overlap_procrustes")
                ),
                path_overlap_warning_threshold=float(
                    symmetry_record.get("path_overlap_warning_threshold", 0.98)
                ),
                operation_tolerance_angstrom=float(
                    symmetry_record.get("operation_tolerance_angstrom", 1.0e-3)
                ),
                max_operation_residual_angstrom=float(
                    symmetry_record.get("max_operation_residual_angstrom", 0.0)
                ),
                min_operation_margin_angstrom=float(
                    symmetry_record.get("min_operation_margin_angstrom", 0.0)
                ),
                near_threshold_operations=tuple(
                    str(item) for item in symmetry_record.get("near_threshold_operations", ())
                ),
                fallback_events=tuple(
                    fallback_event_from_dict(item)
                    for item in symmetry_record.get("fallback_events", ())
                ),
            )
        )
        return GICDefinition(
            backend=str(record["backend"]),
            point_group=str(record["point_group"]),
            symmetrize=bool(record["symmetrize"]),
            target_rank=int(record["target_rank"]),
            rank=int(record["rank"]),
            candidate_count=int(record["candidate_count"]),
            reference_coordinates_angstrom=tuple(
                tuple(float(value) for value in row)
                for row in record["reference_coordinates_angstrom"]
            ),
            primitives=tuple(
                GICPrimitive(
                    identifier=str(item["identifier"]),
                    name=str(item["name"]),
                    family=str(item["family"]),
                    function=str(item["function"]),
                    atoms=tuple(int(value) for value in item["atoms"]),
                    mode=int(item.get("mode", 0)),
                    ref_atoms=tuple(int(value) for value in item.get("ref_atoms", ())),
                    refs=tuple(str(value) for value in item.get("refs", ())),
                    frame_atoms=tuple(int(value) for value in item.get("frame_atoms", ())),
                    ref_frame_atoms=tuple(
                        int(value) for value in item.get("ref_frame_atoms", ())
                    ),
                    provenance=str(item.get("provenance", "AUTO")),
                    semantic_id=str(item.get("semantic_id", "")),
                    semantic_type=str(item.get("semantic_type", "")),
                    chart=str(item.get("chart", "PRINCIPAL")),
                    chart_reference_radian=(
                        None
                        if item.get("chart_reference_radian") is None
                        else float(item["chart_reference_radian"])
                    ),
                )
                for item in record["primitives"]
            ),
            gics=tuple(
                FrozenGIC(
                    identifier=str(item["identifier"]),
                    name=str(item["name"]),
                    family=str(item["family"]),
                    irrep=str(item["irrep"]),
                    primitive_id=str(item["primitive_id"]),
                    gaussian_expression=str(item["gaussian_expression"]),
                    coefficients=tuple(
                        (str(pair[0]), float(pair[1]))
                        for pair in item.get("coefficients", ())
                    ),
                )
                for item in record["gics"]
            ),
            reduction_diagnostics=reduction,
            symmetry_diagnostics=symmetry,
            fragment_mode=str(record.get("fragment_mode", "NONE")),
            pseudo_bonds=tuple(
                tuple(int(value) for value in pair) for pair in record.get("pseudo_bonds", ())
            ),
            pseudo_bond_kinds=tuple(
                str(item) for item in record.get("pseudo_bond_kinds", ())
            ),
            aromatic_atoms=tuple(int(item) for item in record.get("aromatic_atoms", ())),
            xh_stretch_policy=str(record.get("xh_stretch_policy", "SYMMETRIZE")),
            local_xh_bonds=tuple(
                tuple(int(value) for value in pair)
                for pair in record.get("local_xh_bonds", ())
            ),
            local_xh_classes=tuple(
                str(item) for item in record.get("local_xh_classes", ())
            ),
            ring_puckering_diagnostics=tuple(
                str(item) for item in record.get("ring_puckering_diagnostics", ())
            ),
            periodic_coordinate_estimates=tuple(
                _periodic_estimate_from_dict(item)
                for item in record.get("periodic_coordinate_estimates", ())
            ),
            contract_schema_version=str(record.get("contract_schema_version", "")),
            semantic_grammar_version=str(record.get("semantic_grammar_version", "")),
            semantic_diagnostics=tuple(
                str(item) for item in record.get("semantic_diagnostics", ())
            ),
            fallback_diagnostics=tuple(
                str(item) for item in record.get("fallback_diagnostics", ())
            ),
            fallback_events=tuple(
                fallback_event_from_dict(item)
                for item in record.get("fallback_events", ())
            ),
            primitive_source=primitive_source,
            primitive_source_schema=primitive_source_schema,
            primitive_b_matrix_sha256=primitive_b_matrix_sha256,
            wilson_tangent_rank=int(record.get("wilson_tangent_rank", 0)),
            wilson_tangent_singular_min=float(
                record.get("wilson_tangent_singular_min", 0.0)
            ),
            wilson_tangent_singular_max=float(
                record.get("wilson_tangent_singular_max", 0.0)
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("frozen GIC payload is incomplete or malformed") from exc


def gic_definition_json(definition: GICDefinition) -> str:
    return json.dumps(
        gic_definition_to_dict(definition),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _periodic_estimate_from_dict(record: Mapping[str, Any]) -> PeriodicCoordinateEstimate:
    values = dict(record)
    for field in ("central_bonds",):
        values[field] = tuple(
            tuple(int(value) for value in pair) for pair in values.get(field, ())
        )
    for field in ("ring_atoms",):
        values[field] = tuple(int(value) for value in values.get(field, ()))
    for field in ("source_coordinates",):
        values[field] = tuple(str(value) for value in values.get(field, ()))
    return PeriodicCoordinateEstimate(**values)


__all__ = [
    "GIC_DEFINITION_JSON_SCHEMA",
    "gic_definition_from_dict",
    "gic_definition_json",
    "gic_definition_to_dict",
]
