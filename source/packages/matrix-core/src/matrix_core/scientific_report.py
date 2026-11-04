"""Canonical scientific result report for one KEYMAKER molecule."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCIENTIFIC_REPORT_SCHEMA = "matrix.keymaker.molecule-scientific-report.v1"


def build_molecule_scientific_report(
    *,
    molecule_id: str,
    records: Sequence[Mapping[str, Any]],
    protocol_stages: Sequence[str],
    artifacts: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Create one stable report without interpreting or altering QM values."""

    if not str(molecule_id).strip():
        raise ValueError("molecule_id must not be empty")
    normalized = []
    required = ("id", "basin", "medoid", "energy_xtb", "energy_l0", "h_bonds", "classification")
    for raw in records:
        missing = [key for key in required if key not in raw]
        if missing:
            raise ValueError(f"scientific record is missing: {', '.join(missing)}")
        normalized.append({
            "id": str(raw["id"]),
            "basin": str(raw["basin"]),
            "medoid": bool(raw["medoid"]),
            "energy_xtb": raw["energy_xtb"],
            "energy_l0": raw["energy_l0"],
            "frequencies_cm1": list(raw.get("frequencies_cm1", [])),
            "h_bonds": list(raw["h_bonds"]),
            "classification": str(raw["classification"]),
            "artifacts": list(raw.get("artifacts", [])),
        })
    return {
        "schema": SCIENTIFIC_REPORT_SCHEMA,
        "molecule_id": str(molecule_id),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_stages": list(protocol_stages),
        "records": normalized,
        "artifacts": [dict(item) for item in artifacts],
        "counts": {
            "records": len(normalized),
            "minima": sum(item["classification"] == "minimum" for item in normalized),
            "transition_states": sum(item["classification"] == "transition_state_candidate" for item in normalized),
        },
    }


def write_molecule_scientific_report(path: str | Path, report: Mapping[str, Any]) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
