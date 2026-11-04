"""Content-bound KEYMAKER handoffs from SWITCH through MORPHEUS.

Scientific execution stays in the registered owner packages.  This module
freezes the route and verifies the files exchanged between those owners.
"""

from __future__ import annotations

import hashlib
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .protocol_library import (
    ACTIVE_GFN2_ZAFF_FIT_PROTOCOL_ID,
    GFN2_ZAFF_FIT_PROTOCOL_IDS,
    TWO_MORSE_GFN2_ZAFF_FIT_PROTOCOL_IDS,
    KNOWN_KEYMAKER_PROTOCOL_LIBRARY_SHA256S,
    KeymakerProtocolManifest,
    load_keymaker_protocol_library,
)


TOOLCHAIN_SCHEMA = "matrix.keymaker.canonical_toolchain.v1"
TOOLCHAIN_STAGES = (
    "switch-source-normalization.v1",
    "oracle-perception.v1",
    "sonic-soop-default.v1",
    "link-minimum-optimization.v1",
    "apoc-electronic-analysis.v1",
    "zaff-geminal-coupled-harmonic-fit.v1",
    "trinity-harmonic-field.v1",
    "trinity-selective-anharmonic-field.v1",
    "morpheus-semiexperimental-refinement.v1",
)


def _digest(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def toolchain_artifact(path: Path | str, *, schema: str, owner: str) -> dict[str, str]:
    """Bind an existing artifact to its schema, owner and exact bytes."""

    target = Path(path).expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(target)
    if not schema.strip() or not owner.strip():
        raise ValueError("toolchain artifact requires a schema and owner")
    return {
        "path": str(target),
        "schema": schema,
        "owner": owner.upper(),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }


def _verify_artifact(record: Mapping[str, Any]) -> None:
    if set(record) != {"path", "schema", "owner", "sha256"}:
        raise ValueError("toolchain artifact reference has incomplete fields")
    current = toolchain_artifact(
        str(record["path"]), schema=str(record["schema"]), owner=str(record["owner"])
    )
    if dict(record) != current:
        raise ValueError(f"toolchain artifact identity changed: {record['path']}")


def _stages(include_architect: bool | str, *, cached_msr: bool = False) -> tuple[str, ...]:
    if include_architect is True or include_architect == "geminal_coupled":
        stages = TOOLCHAIN_STAGES
    elif include_architect == "diagonal":
        stages = tuple(
            "zaff-diagonal-harmonic-fit.v1"
            if item == "zaff-geminal-coupled-harmonic-fit.v1" else item
            for item in TOOLCHAIN_STAGES
        )
    elif include_architect == "stretch_coupled":
        stages = tuple(
            "zaff-stretch-coupled-harmonic-fit.v1"
            if item == "zaff-geminal-coupled-harmonic-fit.v1" else item
            for item in TOOLCHAIN_STAGES
        )
    elif include_architect in {"gfn2_residual", "gfn2_residual_v1"}:
        stages = tuple(
            (ACTIVE_GFN2_ZAFF_FIT_PROTOCOL_ID if include_architect == "gfn2_residual" else "zaff-gfn2-residual-fit.v1")
            if item == "zaff-geminal-coupled-harmonic-fit.v1" else item
            for item in TOOLCHAIN_STAGES
            if item != "apoc-electronic-analysis.v1"
        )
    elif include_architect is False:
        stages = tuple(item for item in TOOLCHAIN_STAGES if item != "zaff-geminal-coupled-harmonic-fit.v1")
    else:
        raise ValueError(f"unknown ARCHITECT model: {include_architect}")
    if cached_msr:
        return (*stages[:-2], "morpheus-msr-cached-refinement.v1")
    return stages


def build_toolchain_plan(
    source_artifacts: Sequence[Mapping[str, Any]], *, include_architect: bool | str = False,
    cached_msr: bool = False,
) -> dict[str, Any]:
    """Freeze approved installed protocols and external input identities."""

    library = load_keymaker_protocol_library()
    artifacts = [dict(item) for item in source_artifacts]
    for artifact in artifacts:
        _verify_artifact(artifact)
    if not any(item["schema"] == "matrix.keymaker.source.v1" for item in artifacts):
        raise ValueError("toolchain requires a KEYMAKER molecular source")
    if cached_msr and not any(item["schema"] == "matrix.morpheus.msr_input.v1" for item in artifacts):
        raise ValueError("cached MORPHEUS route requires a frozen MSR input")
    if any(item["schema"] == "matrix.trinity.computed_cubic_job.v1" for item in artifacts):
        if cached_msr:
            raise ValueError("computed TRINITY F2/F3 cannot use the cached MSR branch")
        required = {
            "matrix.qm.hessian.v1",
            "matrix.morpheus.microwave_observations.v1",
            "matrix.morpheus.structural_reference.v1",
        }
        missing = required - {item["schema"] for item in artifacts}
        if missing:
            raise ValueError(f"computed TRINITY route lacks frozen inputs: {sorted(missing)}")
    stages = _stages(include_architect, cached_msr=cached_msr)
    bindings: list[dict[str, str]] = []
    for identifier in stages:
        manifest = library.resolve(identifier)
        if manifest.status != "approved":
            raise ValueError(f"toolchain protocol is not approved: {identifier}")
        missing_dependencies = set(manifest.dependencies) - set(stages[: len(bindings)])
        if missing_dependencies:
            raise ValueError(f"{identifier} has missing earlier dependencies: {sorted(missing_dependencies)}")
        bindings.append({
            "protocol_id": manifest.protocol_id,
            "manifest_version": manifest.manifest_version,
            "owner_operation": manifest.owner_operation,
            "sha256": manifest.sha256,
        })
    payload: dict[str, Any] = {
        "schema": TOOLCHAIN_SCHEMA,
        "issuer": "KEYMAKER",
        "library_sha256": library.sha256,
        "source_artifacts": artifacts,
        "stages": bindings,
        "records": [],
    }
    payload["plan_sha256"] = _digest({k: v for k, v in payload.items() if k not in {"records", "plan_sha256"}})
    return payload


def validate_toolchain_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Reject changed protocols, files, reordered stages and altered records."""

    payload = json.loads(json.dumps(dict(plan)))
    if set(payload) != {
        "schema", "issuer", "library_sha256", "source_artifacts",
        "stages", "records", "plan_sha256",
    } or payload["schema"] != TOOLCHAIN_SCHEMA or payload["issuer"] != "KEYMAKER":
        raise ValueError("invalid KEYMAKER toolchain plan")
    expected = _digest({k: v for k, v in payload.items() if k not in {"records", "plan_sha256"}})
    if payload["plan_sha256"] != expected:
        raise ValueError("toolchain plan identity changed")
    library = load_keymaker_protocol_library()
    if payload["library_sha256"] not in KNOWN_KEYMAKER_PROTOCOL_LIBRARY_SHA256S:
        raise ValueError("installed KEYMAKER protocol library changed")
    identifiers = tuple(item.get("protocol_id") for item in payload["stages"])
    if identifiers not in {
        _stages(architect, cached_msr=cached)
        for architect in (False, True, "diagonal", "stretch_coupled", "gfn2_residual", "gfn2_residual_v1")
        for cached in (False, True)
    }:
        raise ValueError("toolchain stage sequence changed")
    for binding in payload["stages"]:
        manifest = library.resolve(binding["protocol_id"])
        if binding != {
            "protocol_id": manifest.protocol_id,
            "manifest_version": manifest.manifest_version,
            "owner_operation": manifest.owner_operation,
            "sha256": manifest.sha256,
        } or manifest.status != "approved":
            raise ValueError(f"toolchain protocol identity changed: {manifest.protocol_id}")
        if set(manifest.dependencies) - set(identifiers[: identifiers.index(manifest.protocol_id)]):
            raise ValueError(f"toolchain dependencies changed: {manifest.protocol_id}")
    for artifact in payload["source_artifacts"]:
        _verify_artifact(artifact)
    if not any(item["schema"] == "matrix.keymaker.source.v1" for item in payload["source_artifacts"]):
        raise ValueError("toolchain molecular source is missing")
    if not isinstance(payload["records"], list) or len(payload["records"]) > len(identifiers):
        raise ValueError("invalid toolchain stage records")
    available = list(payload["source_artifacts"])
    for index, record in enumerate(payload["records"]):
        manifest = library.resolve(identifiers[index])
        _verify_stage_record(record, manifest, available)
        available.extend(record["outputs"])
    return payload


def _verify_stage_record(
    record: Mapping[str, Any], manifest: KeymakerProtocolManifest,
    available: Sequence[Mapping[str, Any]],
) -> None:
    if set(record) != {"protocol_id", "protocol_sha256", "outputs", "record_sha256"}:
        raise ValueError("incomplete toolchain stage record")
    if record["protocol_id"] != manifest.protocol_id or record["protocol_sha256"] != manifest.sha256:
        raise ValueError("toolchain stage protocol identity changed")
    if record["record_sha256"] != _digest({k: v for k, v in record.items() if k != "record_sha256"}):
        raise ValueError("toolchain stage record identity changed")
    available_schemas = {item["schema"] for item in available}
    missing = {item.schema for item in manifest.inputs if item.required} - available_schemas
    if missing:
        raise ValueError(f"{manifest.protocol_id} lacks required inputs: {sorted(missing)}")
    outputs = record["outputs"]
    if not isinstance(outputs, list):
        raise ValueError("toolchain stage outputs must be a list")
    declared = {item.schema for item in manifest.outputs}
    observed = [item["schema"] for item in outputs]
    if len(observed) != len(set(observed)) or not set(observed) <= declared:
        raise ValueError("toolchain stage has duplicate or undeclared output schemas")
    if {item.schema for item in manifest.outputs if item.required} - set(observed):
        raise ValueError("toolchain stage lacks required outputs")
    for artifact in outputs:
        _verify_artifact(artifact)
        if artifact["owner"] != manifest.owner_operation.upper():
            raise ValueError("toolchain output owner differs from protocol owner")
        if manifest.protocol_id == "zaff-diagonal-harmonic-fit.v1":
            _verify_architect_fit(Path(artifact["path"]), manifest.sha256)
        elif manifest.protocol_id == "zaff-stretch-coupled-harmonic-fit.v1":
            _verify_stretch_coupled_fit(Path(artifact["path"]), manifest.sha256)
        elif manifest.protocol_id == "zaff-geminal-coupled-harmonic-fit.v1" and artifact["schema"] == "matrix.architect.zaff_geminal_coupled_harmonic_fit.v1":
            _verify_geminal_coupled_fit(Path(artifact["path"]), manifest.sha256)
        elif manifest.protocol_id == "trinity-harmonic-field.v1":
            _verify_trinity_harmonic_field(Path(artifact["path"]), manifest.sha256, available)
        elif manifest.protocol_id == "trinity-selective-anharmonic-field.v1":
            _verify_trinity_computed_field(Path(artifact["path"]), manifest.sha256, available)
    if manifest.protocol_id == "morpheus-msr-cached-refinement.v1":
        _verify_morpheus_cached_outputs(outputs, manifest.sha256, available)
    if manifest.protocol_id == "morpheus-semiexperimental-refinement.v1":
        _verify_morpheus_computed_outputs(outputs, manifest.sha256, available)
    if manifest.protocol_id == "zaff-geminal-coupled-harmonic-fit.v1":
        _verify_geminal_field_outputs(outputs)
    if manifest.protocol_id in GFN2_ZAFF_FIT_PROTOCOL_IDS:
        _verify_gfn2_residual_outputs(outputs, manifest.sha256, manifest.protocol_id)


def _verify_morpheus_cached_outputs(
    outputs: Sequence[Mapping[str, Any]], protocol_sha256: str,
    available: Sequence[Mapping[str, Any]],
) -> None:
    by_schema = {item["schema"]: item for item in outputs}
    reliability = json.loads(Path(by_schema["matrix.morpheus.reliability.v1"]["path"]).read_text())
    sources = {item["schema"]: item["sha256"] for item in available}
    safety = reliability.get("safety", {})
    if (
        reliability.get("schema") != "matrix.morpheus.reliability.v1"
        or reliability.get("protocol_sha256") != protocol_sha256
        or reliability.get("msr_sha256") != sources.get("matrix.morpheus.msr_input.v1")
        or reliability.get("hessian_sha256") != sources.get("matrix.qm.hessian.v1")
        or reliability.get("harmonic_sha256") != sources.get("matrix.trinity.harmonic_field.v1")
        or reliability.get("geometry_sha256") != by_schema["matrix.morpheus.refined_structure.v1"]["sha256"]
        or reliability.get("rank") != reliability.get("n_optimized_parameters")
        or not isinstance(reliability.get("rank"), int)
        or reliability["rank"] < 1
        or not isinstance(reliability.get("rotational_rms_MHz"), (int, float))
        or not math.isfinite(reliability["rotational_rms_MHz"])
        or safety.get("stationary_point") != "minimum"
        or safety.get("full_rank") is not True
        or not isinstance(safety.get("condition_number"), (int, float))
        or not math.isfinite(safety["condition_number"])
        or safety["condition_number"] > safety.get("condition_limit", 0.0)
    ):
        raise ValueError("MORPHEUS cached refinement has invalid fit evidence or provenance")
    if reliability.get("status") == "CERTIFIED":
        if safety.get("reliable") is not True or safety.get("accepted") is not True:
            raise ValueError("MORPHEUS certified status lacks a passing geometry gate")
    elif reliability.get("status") == "REVIEW_REQUIRED":
        if safety.get("reliable") is not False or safety.get("policy") != "parameter_review_required":
            raise ValueError("MORPHEUS review status lacks parameter-specific policy")
    else:
        raise ValueError("MORPHEUS cached refinement has an unknown scientific status")
    for name in ("fit_comparison", "fit_manifest"):
        ref = reliability.get(name)
        if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
            raise ValueError(f"MORPHEUS {name} provenance is incomplete")
        path = Path(ref["path"])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != ref["sha256"]:
            raise ValueError(f"MORPHEUS {name} provenance changed")
    comparison = json.loads(Path(reliability["fit_comparison"]["path"]).read_text())
    fit = comparison.get("constrained_fit", {})
    if (
        comparison.get("schema") != "matrix.morpheus.fit_comparison.v1"
        or fit.get("rank") != reliability["rank"]
        or fit.get("n_optimized_parameters") != reliability["n_optimized_parameters"]
        or fit.get("rotational_rms_MHz") != reliability["rotational_rms_MHz"]
        or fit.get("max_atom_displacement_A") != safety.get("max_atom_displacement_A")
    ):
        raise ValueError("MORPHEUS cached fit comparison contradicts reliability evidence")


def _verify_morpheus_computed_outputs(
    outputs: Sequence[Mapping[str, Any]], protocol_sha256: str,
    available: Sequence[Mapping[str, Any]],
) -> None:
    by_schema = {item["schema"]: item for item in outputs}
    reliability = json.loads(Path(by_schema["matrix.morpheus.reliability.v1"]["path"]).read_text())
    sources = {item["schema"]: item["sha256"] for item in available}
    safety = reliability.get("safety", {})
    if (
        reliability.get("schema") != "matrix.morpheus.reliability.v1"
        or reliability.get("protocol_sha256") != protocol_sha256
        or reliability.get("msr_sha256") != sources.get("matrix.morpheus.microwave_observations.v1")
        or reliability.get("hessian_sha256") != sources.get("matrix.qm.hessian.v1")
        or reliability.get("computed_field_sha256") != sources.get("matrix.trinity.selective_anharmonic_field.v1")
        or reliability.get("geometry_sha256") != by_schema["matrix.morpheus.refined_structure.v1"]["sha256"]
        or reliability.get("rank") != reliability.get("n_optimized_parameters")
        or not isinstance(reliability.get("rank"), int)
        or reliability["rank"] < 1
        or not isinstance(reliability.get("rotational_rms_MHz"), (int, float))
        or not math.isfinite(reliability["rotational_rms_MHz"])
        or safety.get("stationary_point") != "minimum"
        or safety.get("full_rank") is not True
        or not isinstance(safety.get("condition_number"), (int, float))
        or not math.isfinite(safety["condition_number"])
        or safety["condition_number"] > safety.get("condition_limit", 0.0)
    ):
        raise ValueError("MORPHEUS computed refinement has invalid fit evidence or provenance")
    if reliability.get("status") == "CERTIFIED":
        if safety.get("reliable") is not True or safety.get("accepted") is not True:
            raise ValueError("MORPHEUS computed certification lacks a passing geometry gate")
    elif reliability.get("status") == "REVIEW_REQUIRED":
        if safety.get("reliable") is not False or safety.get("policy") != "parameter_review_required":
            raise ValueError("MORPHEUS computed review status lacks a parameter review policy")
    else:
        raise ValueError("MORPHEUS computed refinement has an unknown scientific status")
    for name in ("corrected_observations", "correction_report", "fit_comparison", "fit_manifest"):
        ref = reliability.get(name)
        if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
            raise ValueError(f"MORPHEUS computed {name} provenance is incomplete")
        path = Path(ref["path"])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != ref["sha256"]:
            raise ValueError(f"MORPHEUS computed {name} provenance changed")
    field = json.loads(Path(next(
        item["path"] for item in available
        if item["schema"] == "matrix.trinity.selective_anharmonic_field.v1"
    )).read_text())
    corrections = {
        item["label"]: item["delta_MHz"]
        for item in field["isotopologue_corrections"]
    }
    with Path(reliability["corrected_observations"]["path"]).open(
        newline="", encoding="utf-8"
    ) as handle:
        observed = list(csv.DictReader(handle))
    if len(observed) != len(corrections) or {row.get("label") for row in observed} != set(corrections):
        raise ValueError("MORPHEUS computed observations omit a TRINITY isotopologue")
    for row in observed:
        delta = corrections[row["label"]]
        try:
            written = tuple(float(row[name]) for name in (
                "delta_A_MHz", "delta_B_MHz", "delta_C_MHz"
            ))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("MORPHEUS computed observations lack DeltaBvib values") from exc
        if (
            any(abs(a - b) > 1e-8 for a, b in zip(written, delta, strict=True))
            or row.get("correction_convention") != "subtract"
            or "TRINITY" not in row.get("correction_source", "")
        ):
            raise ValueError("MORPHEUS observations differ from computed TRINITY corrections")
    report = json.loads(Path(reliability["correction_report"]["path"]).read_text())
    if (
        report.get("schema") != "matrix.morpheus.isotopic-deltavib.v2"
        or report.get("authoritative_channel") != "curvilinear-sonic"
        or {item.get("label") for item in report.get("curvilinear", [])} != set(corrections)
    ):
        raise ValueError("MORPHEUS computed correction report differs from TRINITY")
    comparison = json.loads(Path(reliability["fit_comparison"]["path"]).read_text())
    fit = comparison.get("constrained_fit", {})
    if (
        comparison.get("schema") != "matrix.morpheus.fit_comparison.v1"
        or fit.get("rank") != reliability["rank"]
        or fit.get("n_optimized_parameters") != reliability["n_optimized_parameters"]
        or fit.get("rotational_rms_MHz") != reliability["rotational_rms_MHz"]
        or fit.get("max_atom_displacement_A") != safety.get("max_atom_displacement_A")
    ):
        raise ValueError("MORPHEUS computed fit comparison contradicts reliability evidence")


def _verify_trinity_harmonic_field(
    path: Path, protocol_sha256: str, available: Sequence[Mapping[str, Any]]
) -> None:
    """Check the actual TRINITY transform and its frozen input identities."""

    field = json.loads(path.read_text(encoding="utf-8"))
    sources = {item["schema"]: item["sha256"] for item in available}
    labels = field.get("primitive_labels")
    force_constants = field.get("force_constants_hartree_per_internal_unit_squared")
    rank = field.get("primitive_rank")
    residual = field.get("roundtrip_max_abs_hartree_per_bohr2")
    if (
        field.get("schema") != "matrix.trinity.harmonic_field.v1"
        or field.get("representation") != "frozen-smith-sonic"
        or field.get("protocol_sha256") != protocol_sha256
        or field.get("hessian_sha256") != sources.get("matrix.qm.hessian.v1")
        or field.get("smith_sha256") != sources.get("matrix.smith.definition.v1")
        or not isinstance(labels, list)
        or not labels
        or len(labels) != len(set(labels))
        or not isinstance(force_constants, list)
        or len(force_constants) != len(labels)
        or any(not isinstance(row, list) or len(row) != len(labels) for row in force_constants)
        or not isinstance(rank, int)
        or rank != field.get("vibrational_rank")
        or rank < 1
        or not isinstance(residual, (int, float))
        or not math.isfinite(residual)
        or residual > 1.0e-6
    ):
        raise ValueError("TRINITY harmonic field has an invalid transform or input provenance")
    for i, row in enumerate(force_constants):
        for j, value in enumerate(row):
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("TRINITY harmonic field contains non-finite force constants")
            if abs(value - force_constants[j][i]) > 1.0e-8:
                raise ValueError("TRINITY harmonic field is not symmetric")


def _verify_trinity_computed_field(
    path: Path, protocol_sha256: str, available: Sequence[Mapping[str, Any]]
) -> None:
    field = json.loads(path.read_text(encoding="utf-8"))
    sources = {item["schema"]: item["sha256"] for item in available}
    paths = {item["schema"]: item["path"] for item in available}
    rows = field.get("isotopologue_corrections")
    if (
        field.get("schema") != "matrix.trinity.selective_anharmonic_field.v1"
        or field.get("protocol_sha256") != protocol_sha256
        or field.get("computed_job_sha256") != sources.get("matrix.trinity.computed_cubic_job.v1")
        or field.get("computed_job_path") != paths.get("matrix.trinity.computed_cubic_job.v1")
        or field.get("hessian_sha256") != sources.get("matrix.qm.hessian.v1")
        or field.get("harmonic_sha256") != sources.get("matrix.trinity.harmonic_field.v1")
        or field.get("method_artifact_sha256") not in {
            item["sha256"] for item in available
            if item["schema"] in {"matrix.qm.hessian.v1", "matrix.zaff.force_field.v1"}
        }
        or not isinstance(field.get("method"), str)
        or not field["method"].strip()
        or not isinstance(field.get("coordinate_count"), int)
        or field["coordinate_count"] < 1
        or not isinstance(field.get("harmonic_relative_error"), (int, float))
        or not 0.0 <= field["harmonic_relative_error"] <= 1e-3
        or not isinstance(rows, list)
        or not rows
    ):
        raise ValueError("TRINITY computed F2/F3 field lacks same-method evidence")
    labels = [row.get("label") for row in rows if isinstance(row, dict)]
    if len(labels) != len(rows) or len(set(labels)) != len(rows):
        raise ValueError("TRINITY computed F2/F3 isotope rows are incomplete")
    for row in rows:
        delta = row.get("delta_MHz")
        if (
            row.get("schema") != "matrix.trinity.curvilinear-deltabvib.v1"
            or not isinstance(delta, list)
            or len(delta) != 3
            or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in delta)
        ):
            raise ValueError("TRINITY computed F2/F3 isotope correction is invalid")


def _verify_architect_fit(path: Path, protocol_sha256: str) -> None:
    """Keep a failed diagonal fit out of the canonical scientific chain."""

    fit = json.loads(path.read_text(encoding="utf-8"))
    if (
        fit.get("schema") != "matrix.architect.zaff_diagonal_harmonic_fit.v1"
        or fit.get("status") != "COMPLETED"
        or fit.get("protocol", {}).get("sha256") != protocol_sha256
    ):
        raise ValueError("ARCHITECT fit has an invalid protocol or incomplete result")
    if fit.get("excluded_large_amplitude_modes") != []:
        raise ValueError("ARCHITECT canonical admission requires an all-mode fit")
    frequency = fit.get("ordinary_mode_metrics", {})
    overlap = fit.get("mode_overlap", {})
    rms = float(frequency.get("rms_error_cm-1", float("nan")))
    mean_overlap = float(overlap.get("mean_assigned_overlap", float("nan")))
    if not (
        math.isfinite(rms)
        and math.isfinite(mean_overlap)
        and rms <= 10.0
        and frequency.get("accepted") is True
    ):
        raise ValueError("ARCHITECT diagonal fit is not eligible: all-mode frequency gate must pass")


def _verify_stretch_coupled_fit(path: Path, protocol_sha256: str) -> None:
    fit = json.loads(path.read_text(encoding="utf-8"))
    rms = float(fit.get("frequency_rms_error_cm-1", float("nan")))
    overlap = float(fit.get("mean_assigned_mode_overlap", float("nan")))
    if not (
        fit.get("schema") == "matrix.architect.zaff_stretch_coupled_harmonic_fit.v1"
        and fit.get("protocol", {}).get("sha256") == protocol_sha256
        and fit.get("status") == "ACCEPTED"
        and fit.get("accepted") is True
        and fit.get("excluded_modes") == []
        and fit.get("primitive_rank") == 3
        and fit.get("solver_success") is True
        and math.isfinite(rms) and rms <= 10.0
        and math.isfinite(overlap) and overlap >= 0.80
    ):
        raise ValueError("ARCHITECT stretch-coupled fit is not eligible")


def _verify_geminal_coupled_fit(path: Path, protocol_sha256: str) -> None:
    fit = json.loads(path.read_text(encoding="utf-8"))
    rms = float(fit.get("frequency_rms_error_cm-1", float("nan")))
    overlap = float(fit.get("mean_assigned_mode_overlap", float("nan")))
    pairs = fit.get("geminal_xh_couplings")
    if not (
        fit.get("schema") == "matrix.architect.zaff_geminal_coupled_harmonic_fit.v1"
        and fit.get("protocol", {}).get("sha256") == protocol_sha256
        and fit.get("status") == "ACCEPTED"
        and fit.get("accepted") is True
        and fit.get("all_modes_included") is True
        and fit.get("design_rank") == fit.get("parameter_count")
        and isinstance(pairs, list)
        and all(item.get("functional_form") == "DISPLACEMENT_QUARTIC_DAMPED_PRODUCT" for item in pairs)
        and math.isfinite(rms) and rms <= 10.0
        and math.isfinite(overlap) and overlap >= 0.80
    ):
        raise ValueError("ARCHITECT geminal-coupled fit is not eligible")


def _verify_geminal_field_outputs(outputs: Sequence[Mapping[str, Any]]) -> None:
    by_schema = {item["schema"]: Path(item["path"]) for item in outputs}
    fit = json.loads(by_schema["matrix.architect.zaff_geminal_coupled_harmonic_fit.v1"].read_text())
    field = json.loads(by_schema["matrix.zaff.force_field.v1"].read_text())
    if (
        field.get("schema") != "matrix.zaff.force_field.v1"
        or fit.get("fitted_field") != str(by_schema["matrix.zaff.force_field.v1"])
        or field.get("geminal_xh_couplings") != [
            {key: value for key, value in item.items() if key != "primitive_indices_zero_based"}
            for item in fit["geminal_xh_couplings"]
        ]
        or field.get("hessian_hartree_per_bohr2") != fit.get("model_cartesian_hessian_hartree_per_bohr2")
    ):
        raise ValueError("ZAFF field differs from the accepted geminal fit")


def _verify_gfn2_residual_outputs(
    outputs: Sequence[Mapping[str, Any]], protocol_sha256: str, protocol_id: str,
) -> None:
    by_schema = {item["schema"]: Path(item["path"]) for item in outputs}
    fit_path = by_schema["matrix.architect.zaff_gfn2_residual_fit.v1"]
    baseline_path = by_schema["matrix.gfn2.residual_baseline.v1"]
    field_path = by_schema["matrix.zaff.force_field.v1"]
    fit = json.loads(fit_path.read_text())
    baseline = json.loads(baseline_path.read_text())
    field = json.loads(field_path.read_text())
    gradient_error = float(fit.get("gradient_max_error_hartree_per_bohr", float("nan")))
    hessian_error = float(fit.get("hessian_relative_frobenius_error", float("nan")))
    if not (
        fit.get("schema") == "matrix.architect.zaff_gfn2_residual_fit.v1"
        and fit.get("protocol", {}).get("sha256") == protocol_sha256
        and fit.get("accepted") is True
        and fit.get("status") == "ACCEPTED"
        and fit.get("gfn2_baseline") == str(baseline_path)
        and fit.get("fitted_field") == str(field_path)
        and baseline.get("schema") == "matrix.gfn2.residual_baseline.v1"
        and baseline.get("converged") is True
        and field.get("schema") == "matrix.zaff.force_field.v1"
        and field.get("metadata", {}).get("baseline") == "GFN2-XTB"
        and (
            protocol_id == "zaff-gfn2-residual-fit.v1"
            or field.get("metadata", {}).get("primitive_residual_model", {}).get("schema")
            == ("matrix.zaff.primitive_residual.v2" if protocol_id in TWO_MORSE_GFN2_ZAFF_FIT_PROTOCOL_IDS
                else "matrix.zaff.primitive_residual.v1")
        )
        and field.get("metadata", {}).get("nonbonded_terms") == "NONE"
        and fit.get("subtraction") == "QM_MINUS_COMPLETE_RESIDENT_GFN2_ONLY"
        and fit.get("zaff_atomic_charges_e") == [0.0] * len(baseline.get("atoms", []))
        and field.get("metadata", {}).get("zaff_atomic_charges_e") == fit.get("zaff_atomic_charges_e")
        and field.get("metadata", {}).get("zaff_vdw_terms") == []
        and fit.get("zaff_reference_energy_hartree") == 0.0
        and field.get("energy_reference_hartree") == 0.0
        and field.get("gradient_reference_hartree_per_bohr") == fit.get("fitted_gradient_hartree_per_bohr")
        and field.get("hessian_hartree_per_bohr2") == fit.get("fitted_hessian_hartree_per_bohr2")
        and math.isfinite(gradient_error) and gradient_error <= 1.0e-6
        and math.isfinite(hessian_error) and hessian_error <= 0.05
    ):
        raise ValueError("GFN2 residual field is not eligible for canonical ARCHITECT")


def toolchain_readiness(plan: Mapping[str, Any]) -> dict[str, Any]:
    payload = validate_toolchain_plan(plan)
    index = len(payload["records"])
    if index == len(payload["stages"]):
        status = {"complete": True, "ready": False, "next_protocol": None, "missing": []}
        if payload["stages"][-1]["protocol_id"] == "morpheus-msr-cached-refinement.v1":
            reliability = next(
                item for item in payload["records"][-1]["outputs"]
                if item["schema"] == "matrix.morpheus.reliability.v1"
            )
            status["scientific_status"] = json.loads(
                Path(reliability["path"]).read_text(encoding="utf-8")
            )["status"]
        return status
    manifest = load_keymaker_protocol_library().resolve(payload["stages"][index]["protocol_id"])
    available = payload["source_artifacts"] + [
        artifact for record in payload["records"] for artifact in record["outputs"]
    ]
    schemas = {item["schema"] for item in available}
    missing = sorted({item.schema for item in manifest.inputs if item.required} - schemas)
    return {
        "complete": False,
        "ready": not missing,
        "next_protocol": manifest.protocol_id,
        "missing": missing,
    }


def record_toolchain_stage(
    plan: Mapping[str, Any], outputs: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    payload = validate_toolchain_plan(plan)
    index = len(payload["records"])
    if index >= len(payload["stages"]):
        raise ValueError("toolchain is already complete")
    manifest = load_keymaker_protocol_library().resolve(payload["stages"][index]["protocol_id"])
    record: dict[str, Any] = {
        "protocol_id": manifest.protocol_id,
        "protocol_sha256": manifest.sha256,
        "outputs": [dict(item) for item in outputs],
    }
    record["record_sha256"] = _digest(record)
    payload["records"].append(record)
    return validate_toolchain_plan(payload)
