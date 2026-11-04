"""Fail-closed KeyMaker validation of frozen TRINITY chart evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile
from typing import Any


PROTOCOL_ID = "trinity-chart-evidence-validation.v1"
STENCIL_SCHEMA = "matrix.trinity.formic-electronic-stencil.v1"
DVR_POLICY = "matrix-trinity.large-amplitude-dvr.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_stencil(archive: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != STENCIL_SCHEMA:
        raise ValueError("unsupported electronic stencil schema")
    if _sha256(archive) != manifest.get("archive_sha256"):
        raise ValueError("electronic stencil archive checksum mismatch")
    expected = manifest.get("files")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("electronic stencil manifest has no files")
    with tarfile.open(archive, "r:gz") as packed:
        members = packed.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or set(names) != set(expected) | {"manifest.json"}:
            raise ValueError("electronic stencil member list mismatch")
        if not all(member.isfile() for member in members):
            raise ValueError("electronic stencil contains a non-file member")
        embedded = json.load(packed.extractfile("manifest.json"))
        if embedded != {key: value for key, value in manifest.items()
                        if key != "archive_sha256"}:
            raise ValueError("embedded electronic stencil manifest mismatch")
        for name, record in expected.items():
            member = packed.getmember(name)
            if member.size != record.get("bytes"):
                raise ValueError(f"electronic stencil size mismatch: {name}")
            digest = hashlib.sha256()
            with packed.extractfile(member) as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != record.get("sha256"):
                raise ValueError(f"electronic stencil checksum mismatch: {name}")
    return {
        "archive": str(archive.resolve()),
        "manifest": str(manifest_path.resolve()),
        "archive_sha256": manifest["archive_sha256"],
        "manifest_sha256": _sha256(manifest_path),
        "molecules": manifest.get("molecules"),
        "files_verified": len(expected),
    }


def _verify_dvr(report_path: Path, mode: int) -> dict[str, Any]:
    from .coordinate_gauge import assess_large_amplitude_dvr

    report = json.loads(report_path.read_text())
    if report.get("policy_id") != DVR_POLICY or report.get("mode") != mode:
        raise ValueError(f"DVR diagnostic does not cover mode {mode + 1}")
    if report.get("passed") is not True:
        raise ValueError(f"DVR diagnostic failed for mode {mode + 1}")
    computed = assess_large_amplitude_dvr(
        mode=mode,
        harmonic_cm=report["harmonic_cm"],
        potential_intermode_norm_native_cm=report["potential_intermode_norm_native_cm"],
        potential_intermode_norm_reference_cm=report["potential_intermode_norm_reference_cm"],
        dvr_levels_relative_cm=report["dvr_levels_relative_cm"],
        path_mode_overlap=report["path_mode_overlap"],
        grid_gap_difference_cm=report["grid_gap_difference_cm"],
        grid_ir_difference_km_mol=report["grid_ir_difference_km_mol"],
        grid_gap_tolerance_cm=report["grid_gap_tolerance_cm"],
        grid_ir_tolerance_km_mol=report["grid_ir_tolerance_km_mol"],
        minimum_path_mode_overlap=report["minimum_path_mode_overlap"],
    )
    if any(report.get(key) != value for key, value in computed.items()):
        raise ValueError(f"DVR diagnostic is inconsistent for mode {mode + 1}")
    source = report.get("source")
    if not isinstance(source, dict) or not source:
        raise ValueError("DVR diagnostic lacks source provenance")
    for label, item in source.items():
        if not isinstance(item, dict) or not item.get("file") or not item.get("sha256"):
            raise ValueError(f"incomplete DVR source: {label}")
        source_file = report_path.parent / item["file"]
        if _sha256(source_file) != item["sha256"]:
            raise ValueError(f"DVR source checksum mismatch: {label}")
    return {
        "mode_one_based": mode + 1,
        "molecule": report.get("molecule"),
        "report": str(report_path.resolve()),
        "report_sha256": _sha256(report_path),
        "first_dvr_gap_cm": report.get("first_dvr_gap_cm"),
    }


def validate_chart_evidence(
    *,
    archive: Path | None = None,
    manifest: Path | None = None,
    large_amplitude_modes: tuple[int, ...] = (),
    dvr_reports: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Verify saved electronic points and one DVR report per declared 0-based mode.

    Electronic and DVR evidence can be checked in separate KeyMaker calls because
    they may belong to different molecules.  An empty call is never accepted.
    """
    if (archive is None) != (manifest is None):
        raise ValueError("electronic stencil archive and manifest must be paired")
    if not large_amplitude_modes and archive is None:
        raise ValueError("no electronic stencil or large-amplitude modes supplied")
    if len(large_amplitude_modes) != len(set(large_amplitude_modes)):
        raise ValueError("duplicate large-amplitude mode")
    if any(mode < 0 for mode in large_amplitude_modes):
        raise ValueError("large-amplitude modes must be nonnegative")
    if len(large_amplitude_modes) != len(dvr_reports):
        raise ValueError("one DVR diagnostic is required per large-amplitude mode")
    stencil = _verify_stencil(Path(archive), Path(manifest)) if archive is not None else None
    dvr = [_verify_dvr(Path(path), mode)
           for mode, path in zip(large_amplitude_modes, dvr_reports)]
    return {
        "schema": "matrix.trinity.chart_evidence_validation.v1",
        "protocol_id": PROTOCOL_ID,
        "passed": True,
        "electronic_stencil": stencil,
        "large_amplitude_dvr": dvr,
    }
