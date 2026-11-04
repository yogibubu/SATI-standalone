"""Reference and experimental validation for tunneling workflow artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping

import numpy as np


def validate_reference(
    result_json: Path | str,
    reference_json: Path | str,
    *,
    tolerances: Mapping[str, float] | None = None,
) -> dict[str, object]:
    result = json.loads(Path(result_json).read_text(encoding="utf-8"))
    reference = json.loads(Path(reference_json).read_text(encoding="utf-8"))
    limits = {str(key): float(value) for key, value in (tolerances or {}).items()}
    comparisons: list[dict[str, object]] = []
    for key, expected in reference.get("values", {}).items():
        if key not in result:
            comparisons.append({"key": key, "status": "missing"})
            continue
        actual = np.asarray(result[key], dtype=float)
        target = np.asarray(expected, dtype=float)
        if actual.shape != target.shape:
            comparisons.append({"key": key, "status": "shape_mismatch", "actual_shape": actual.shape, "reference_shape": target.shape})
            continue
        error = float(np.max(np.abs(actual - target), initial=0.0))
        tolerance = limits.get(key, float(reference.get("tolerances", {}).get(key, 1.0e-6)))
        comparisons.append({"key": key, "status": "pass" if error <= tolerance else "fail", "max_abs_error": error, "tolerance": tolerance})
    passed = all(item["status"] == "pass" for item in comparisons) if comparisons else False
    return {"schema": "matrix.tunneling.reference_validation.v1", "valid": passed, "comparisons": comparisons}


def validate_surface_csv(surface: Path | str, reference: Path | str, *, columns: tuple[str, ...] = ("energy_hartree", "A_MHz", "B_MHz", "C_MHz")) -> dict[str, object]:
    with Path(surface).open(newline="", encoding="utf-8") as handle:
        actual_rows = list(csv.DictReader(handle))
    with Path(reference).open(newline="", encoding="utf-8") as handle:
        reference_rows = list(csv.DictReader(handle))
    if len(actual_rows) != len(reference_rows):
        return {"schema": "matrix.tunneling.surface_validation.v1", "valid": False, "reason": "row_count_mismatch"}
    errors = {}
    for column in columns:
        if not actual_rows or column not in actual_rows[0] or column not in reference_rows[0]:
            continue
        errors[column] = float(max(abs(float(a[column]) - float(b[column])) for a, b in zip(actual_rows, reference_rows)))
    return {"schema": "matrix.tunneling.surface_validation.v1", "valid": True, "max_abs_error": errors, "rows": len(actual_rows)}


def write_validation_report(path: Path | str, payload: dict[str, object]) -> Path:
    target = Path(path)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


__all__ = ["validate_reference", "validate_surface_csv", "write_validation_report"]
