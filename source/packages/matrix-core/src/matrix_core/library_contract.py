"""Structural validation for catalogues used by KEYMAKER."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def validate_fragment_library_records(
    records: Sequence[Mapping[str, Any]], *, expected_variants: Sequence[str] = ()
) -> dict[str, Any]:
    errors: list[str] = []
    ids: set[str] = set()
    for index, record in enumerate(records):
        identifier = str(record.get("id", "")).strip()
        if not identifier:
            errors.append(f"record {index}: missing id")
        elif identifier in ids:
            errors.append(f"duplicate id: {identifier}")
        ids.add(identifier)
        status = str(record.get("status", "available")).strip()
        if status not in {"available", "PENDING_QM", "unavailable"}:
            errors.append(f"{identifier}: invalid status {status}")
        variants = dict(record.get("variants", {}))
        for variant in expected_variants:
            if variant not in variants:
                errors.append(f"{identifier}: missing variant {variant}")
        sequences = [tuple(item) for item in record.get("atom_sequences", [])]
        if sequences and any(sequence != sequences[0] for sequence in sequences[1:]):
            errors.append(f"{identifier}: inconsistent atom ordering")
    return {"valid": not errors, "records": len(records), "errors": errors}
