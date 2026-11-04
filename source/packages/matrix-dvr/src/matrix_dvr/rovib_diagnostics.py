"""Diagnostics and interchange artifacts for the ESALC rovibrational workflow."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
import numpy as np


WMS_ROT_SCHEMA = "matrix.wms_rot.rovib_surface.v1"
LOCK_ROVIB_SCHEMA = "matrix.lock.rovib_handoff.v1"


def validate_scan_artifacts(
    *,
    source: Path,
    dimension: int,
    coordinates: tuple[str, ...],
    geometry_grid: Path | None = None,
) -> dict[str, object]:
    """Run inexpensive, deterministic physical/integrity checks."""
    errors: list[str] = []
    warnings: list[str] = []
    if not source.is_file():
        errors.append(f"missing scan source: {source}")
    if len(coordinates) != dimension:
        errors.append("coordinate count does not match scan dimension")
    rows: list[dict[str, str]] = []
    if source.suffix.casefold() == ".csv" and source.is_file():
        with source.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            errors.append("scan CSV is empty")
        required = {"energy_hartree"} if dimension == 2 else set()
        missing = sorted(required - set(rows[0] if rows else ()))
        if missing:
            errors.append(f"scan CSV lacks columns: {', '.join(missing)}")
        for key in ("energy_hartree", "A_MHz", "B_MHz", "C_MHz"):
            if rows and key in rows[0]:
                values = np.asarray([float(row[key]) for row in rows], dtype=float)
                if not np.all(np.isfinite(values)):
                    errors.append(f"non-finite values in {key}")
                if key.endswith("_MHz") and np.any(values <= 0.0):
                    warnings.append(f"non-positive rotational constant in {key}")
        if dimension == 2:
            expected = len({row[coordinates[0]] for row in rows}) * len(
                {row[coordinates[1]] for row in rows}
            )
            if expected != len(rows):
                errors.append("2D scan is not a complete rectangular grid")
    if dimension == 2 and geometry_grid is not None and not geometry_grid.is_file():
        errors.append(f"missing geometry grid: {geometry_grid}")
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "dimension": dimension,
        "coordinates": list(coordinates),
        "point_count": len(rows),
    }


def write_lock_export(
    outdir: Path,
    *,
    source: Path,
    diagnostics: dict[str, object],
    dvr_result: Path | None = None,
    coriolis_file: Path | None = None,
    expectations_file: Path | None = None,
    lock_surface: Path | None = None,
) -> Path:
    """Write the native LOCK hand-off."""
    outdir.mkdir(parents=True, exist_ok=True)
    native_input = _write_lock_rotor_input(outdir, source)
    surface_columns: list[str] = []
    if source.is_file() and source.suffix.casefold() == ".csv":
        with source.open(newline="", encoding="utf-8") as handle:
            surface_columns = list(csv.DictReader(handle).fieldnames or [])
    payload = {
        "schema": LOCK_ROVIB_SCHEMA,
        "producer": "nano-MATRIX",
        "source": str(source),
        "surface_file": str(source),
        "rotational_derivatives_file": str(source.parent / "rotational_derivatives.csv")
        if (source.parent / "rotational_derivatives.csv").is_file()
        else None,
        "dvr_result": None if dvr_result is None else str(dvr_result),
        "coriolis_file": None if coriolis_file is None else str(coriolis_file),
        "expectations_file": None if expectations_file is None else str(expectations_file),
        "lock_surface": None if lock_surface is None else str(lock_surface),
        "lock_rotor_input": None if native_input is None else str(native_input),
        "diagnostics": diagnostics,
        "parameter_columns": surface_columns or ["A_MHz", "B_MHz", "C_MHz"],
        "solver_owner": "LOCK",
        "metric_owner": "TRINITY",
        "fitting_owner": "LOCK",
        "units": {"energy": "hartree", "rotational_constants": "MHz"},
    }
    path = outdir / "lock_input.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def write_wms_rot_export(
    outdir: Path,
    **kwargs,
) -> Path:
    """Compatibility wrapper returning the legacy WMS-Rot JSON path."""
    lock_path = write_lock_export(outdir, **kwargs)
    compatibility_path = outdir / "wms_rot_input.json"
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    payload.update(schema=WMS_ROT_SCHEMA, role="reference_compatibility_only")
    compatibility_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return compatibility_path


def _write_lock_rotor_input(outdir: Path, source: Path) -> Path | None:
    """Convert the first scan surface row into the LOCK rotor text input."""
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    if source.suffix.casefold() == ".csv" and source.is_file():
        with source.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    if not rows or not {"A_MHz", "B_MHz", "C_MHz"}.issubset(rows[0]):
        return None
    from matrix_rovib.contracts import RotationalSection
    from matrix_rovib.wmsrot import LOCKRotorInputOptions, wmsrot_input_lines

    rotational = RotationalSection(
        rotor_type="asymmetric",
        representation="Ir",
        point_group="C1",
        watson_reduction="S",
        A_MHz=float(rows[0]["A_MHz"]),
        B_MHz=float(rows[0]["B_MHz"]),
        C_MHz=float(rows[0]["C_MHz"]),
        constant_kind="scan_reference",
        constant_source=str(source),
    )
    target = outdir / "lock_rotor_input.inp"
    target.write_text(
        "\n".join(wmsrot_input_lines(rotational, options=LOCKRotorInputOptions())) + "\n",
        encoding="utf-8",
    )
    return target


def write_human_report(
    outdir: Path,
    *,
    result: dict[str, object],
    diagnostics: dict[str, object],
) -> Path:
    """Create a self-contained, readable HTML summary for a calculation."""
    errors = "".join(f"<li>{html.escape(str(item))}</li>" for item in diagnostics["errors"])
    warnings = "".join(f"<li>{html.escape(str(item))}</li>" for item in diagnostics["warnings"])
    body = html.escape(json.dumps(result, indent=2, ensure_ascii=False))
    document = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><title>MATRIX ESALC tunneling report</title>
<style>body{{font:15px system-ui;max-width:1000px;margin:2em auto}} pre{{background:#f4f4f4;padding:1em;overflow:auto}} .bad{{color:#a00}} .warn{{color:#986000}}</style>
<h1>MATRIX ESALC tunneling report</h1>
<p><b>Validation:</b> {'PASS' if diagnostics['valid'] else '<span class="bad">FAIL</span>'}</p>
<h2>Errors</h2><ul>{errors or '<li>none</li>'}</ul>
<h2>Warnings</h2><ul class="warn">{warnings or '<li>none</li>'}</ul>
<h2>Machine-readable result</h2><pre>{body}</pre>
</html>
"""
    path = outdir / "esalc_scan_report.html"
    path.write_text(document, encoding="utf-8")
    return path


__all__ = [
    "LOCK_ROVIB_SCHEMA",
    "WMS_ROT_SCHEMA",
    "validate_scan_artifacts",
    "write_human_report",
    "write_lock_export",
    "write_wms_rot_export",
]
