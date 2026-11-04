"""Build auditable LCB25/LCB26 synthon transfer-table records."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .synthon_table import ZAFF_SYNTHON_TABLE_SCHEMA


def build_synthon_parameter_table(
    enrichment_records: Sequence[Mapping[str, Any]],
    *,
    library_mapping: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Convert enriched raw records into a derived transfer-table manifest.

    Parameter vectors are accepted only when supplied by an ARCHITECT seed
    build under ``zaff_parameters``.  Electronic records alone remain pending.
    """

    rows: list[dict[str, Any]] = []
    pending: list[str] = []
    for record in enrichment_records:
        identifier = str(record.get("identifier", ""))
        descriptors = record.get("synthon_descriptors", ())
        if not identifier or not descriptors:
            raise ValueError("each enrichment record needs an identifier and synthon_descriptors")
        library = str(record.get("source_library", ""))
        if library not in {"LCB25", "LCB26"}:
            raise ValueError("synthon table accepts only LCB25 or LCB26 records")
        parameters = record.get("zaff_parameters")
        for descriptor in descriptors:
            atom = int(descriptor["atom"])
            row = {
                "schema": ZAFF_SYNTHON_TABLE_SCHEMA,
                "library": library,
                "molecule": identifier,
                "atom": atom,
                "atomic_number": int(descriptor["atomic_number"]),
                "cm5_charge_e": float(descriptor["cm5_charge_e"]),
                "zeff_normalized": float(descriptor["zeff_normalized"]),
                "central_bond_order": float(descriptor.get("central_bond_order", 0.0)),
                "parameters": None if parameters is None else dict(parameters),
                "status": "PARAMETERS_PENDING" if parameters is None else "READY",
                "raw_record_hash": str(record.get("geometry_sha256", "")),
            }
            rows.append(row)
            if parameters is None:
                pending.append(f"{identifier}:{atom}")
    return {
        "schema": ZAFF_SYNTHON_TABLE_SCHEMA,
        "table_kind": "LCB25_LCB26_DERIVED_SYNTHON_PARAMETER_TABLE",
        "raw_records_immutable": True,
        "cross_library_mapping": dict(library_mapping or {}),
        "rows": rows,
        "pending_rows": pending,
    }


__all__ = ["build_synthon_parameter_table"]
