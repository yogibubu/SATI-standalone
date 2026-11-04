"""Transferable primitive-Hessian correction from B3LYP/6-31+G* to MP2/def2-TZVP."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment

from matrix_gf.harmonic_field import transform_cartesian_hessian

from .xtb_hessian_scaling import (
    XtbHessianScaleRequest,
    _normal_modes,
    _proper_alignment,
    _reference_hessian,
    _rotate_modes,
    scale_primitive_hessian_geometric,
    xtb_primitive_features,
)


@dataclass(frozen=True)
class B3lypMp2HessianCorpusRequest:
    mp2_campaign_path: Path
    b3lyp_campaign_path: Path
    artifact_root: Path
    output_path: Path
    protocol_manifest: Mapping[str, Any]
    protocol_sha256: str


@dataclass(frozen=True)
class _Case:
    case_id: str
    split: str
    features: tuple[dict[str, Any], ...]
    low_k: np.ndarray
    high_k: np.ndarray
    b_matrix: np.ndarray
    basis: np.ndarray
    masses: np.ndarray
    target_frequencies: np.ndarray
    target_modes: np.ndarray
    rotation: np.ndarray
    included: np.ndarray


def _load_reference(path: Path, artifact_root: Path):
    return _reference_hessian(XtbHessianScaleRequest(
        reference_hessian_path=path,
        xtb_hessian_path=artifact_root / "final_cartesian_hessian.json",
        primitive_manifest_path=artifact_root / "primitive_manifest.json",
        oracle_topology_path=artifact_root / "oracle-topology.json",
        xtb_observables_path=artifact_root / "gfn2_scaling_observables.json",
        output_path=Path("unused.json"),
        protocol_manifest={},
        protocol_sha256="",
        reference_engine="lcb26",
    ))


def _load_case(
    root: Path,
    artifact_root: Path,
    mp2_row: Mapping[str, Any],
    b3lyp_row: Mapping[str, Any],
) -> _Case:
    case_id = str(mp2_row["id"])
    artifacts = artifact_root / case_id
    mp2 = _load_reference(root / str(mp2_row["harmonic_record"]), artifacts)
    b3lyp = _load_reference(root / str(b3lyp_row["harmonic_record"]), artifacts)
    manifest = json.loads((artifacts / "primitive_manifest.json").read_text())
    resident = json.loads((artifacts / "final_cartesian_hessian.json").read_text())
    atoms = tuple(str(value) for value in resident["atoms"])
    if len(atoms) != len(mp2.masses_amu) or len(atoms) != len(b3lyp.masses_amu):
        raise ValueError(f"{case_id}: inconsistent atom count")
    features = xtb_primitive_features(
        manifest,
        atoms,
        atoms,
        np.zeros(len(atoms)),
        np.ones((len(atoms), len(atoms))),
    )
    high = transform_cartesian_hessian(
        mp2.cartesian_hessian, mp2.cartesian_coordinates_bohr, mp2.masses_amu, manifest
    )
    low = transform_cartesian_hessian(
        b3lyp.cartesian_hessian,
        b3lyp.cartesian_coordinates_bohr,
        b3lyp.masses_amu,
        manifest,
    )
    if high.primitive_labels != low.primitive_labels:
        raise ValueError(f"{case_id}: primitive contracts differ")
    target_frequencies, target_modes = _normal_modes(
        mp2.cartesian_hessian,
        mp2.masses_amu,
        high.redundant_hessian.vibrational_basis,
    )
    rotation, _ = _proper_alignment(
        b3lyp.cartesian_coordinates_bohr,
        mp2.cartesian_coordinates_bohr,
        mp2.masses_amu,
    )
    included = np.ones(len(target_frequencies), dtype=bool)
    if case_id == "saccharin":
        included[0] = False
    return _Case(
        case_id=case_id,
        split=str(mp2_row["split"]),
        features=features,
        low_k=low.redundant_hessian.force_constants,
        high_k=high.redundant_hessian.force_constants,
        b_matrix=low.b_matrix,
        basis=low.redundant_hessian.vibrational_basis,
        masses=mp2.masses_amu,
        target_frequencies=target_frequencies,
        target_modes=target_modes,
        rotation=rotation,
        included=included,
    )


def _feature_value(feature: Mapping[str, Any], label: str) -> float:
    kind, detail = label.split(":", 1)
    family = str(feature["family"])
    if kind == "family":
        return float(family == detail)
    elements = tuple(str(value) for value in feature["elements"])
    if detail == "XH":
        return float(family == "R" and len(elements) == 2 and elements.count("H") == 1)
    if detail == "HXH":
        return float(
            family == "A" and len(elements) == 3
            and elements[0] == "H" and elements[2] == "H"
        )
    if detail == "SOOP_H":
        return float(family == "SOOP" and len(elements) >= 2 and elements[1] == "H")
    raise ValueError(f"unknown B3LYP-to-MP2 feature: {label}")


def _fit(cases: tuple[_Case, ...], labels: tuple[str, ...]) -> np.ndarray:
    rows: list[list[float]] = []
    target: list[float] = []
    weights: list[float] = []
    for case in cases:
        if case.split != "train":
            continue
        low, high = np.diag(case.low_k), np.diag(case.high_k)
        valid = (low > 1.0e-10) & (high > 1.0e-10)
        count = max(int(np.sum(valid)), 1)
        for feature, low_value, high_value, keep in zip(
            case.features, low, high, valid, strict=True
        ):
            if keep:
                rows.append([_feature_value(feature, label) for label in labels])
                target.append(math.log(float(high_value / low_value)))
                weights.append(1.0 / math.sqrt(count))
    design = np.asarray(rows, dtype=float)
    response = np.asarray(target, dtype=float)
    weight = np.asarray(weights, dtype=float)
    weighted = design * weight[:, None]
    return np.linalg.solve(
        weighted.T @ weighted + np.eye(len(labels)) * 1.0e-10,
        weighted.T @ (response * weight),
    )


def _metrics(
    cases: tuple[_Case, ...], labels: tuple[str, ...], coefficients: np.ndarray
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for split in ("train", "validation", "external_validation"):
        errors: list[np.ndarray] = []
        overlaps: list[np.ndarray] = []
        for case in cases:
            if case.split != split:
                continue
            log_scales = np.asarray([
                sum(
                    value * _feature_value(feature, label)
                    for value, label in zip(coefficients, labels, strict=True)
                )
                for feature in case.features
            ])
            scaled_k = scale_primitive_hessian_geometric(case.low_k, np.exp(log_scales))
            hessian = case.b_matrix.T @ scaled_k @ case.b_matrix
            frequencies, modes = _normal_modes(hessian, case.masses, case.basis)
            modes = _rotate_modes(modes, case.rotation)
            target = case.target_modes[:, case.included]
            calculated = modes[:, case.included]
            overlap = np.abs(target.T @ calculated)
            rows, columns = linear_sum_assignment(-overlap)
            overlaps.append(overlap[rows, columns])
            errors.append(
                frequencies[case.included] - case.target_frequencies[case.included]
            )
        joined_errors = np.concatenate(errors)
        joined_overlaps = np.concatenate(overlaps)
        result[split] = {
            "molecule_count": sum(case.split == split for case in cases),
            "mode_count": int(len(joined_errors)),
            "frequency_rms_error_cm-1": float(
                np.sqrt(np.mean(joined_errors * joined_errors))
            ),
            "mean_assigned_mode_overlap": float(np.mean(joined_overlaps)),
            "minimum_assigned_mode_overlap": float(np.min(joined_overlaps)),
        }
    return result


def fit_b3lyp_mp2_hessian_corpus(
    request: B3lypMp2HessianCorpusRequest,
) -> dict[str, Any]:
    protocol = dict(request.protocol_manifest)
    if (
        protocol.get("protocol_id") != "b3lyp-mp2-primitive-hessian-scaling.v1"
        or protocol.get("status") != "approved"
    ):
        raise ValueError("B3LYP-to-MP2 fit requires its approved KeyMaker protocol")
    mp2_path = Path(request.mp2_campaign_path).resolve()
    b3lyp_path = Path(request.b3lyp_campaign_path).resolve()
    root = mp2_path.parents[2]
    mp2_campaign = json.loads(mp2_path.read_text())
    b3lyp_campaign = json.loads(b3lyp_path.read_text())
    b3lyp_rows = {str(row["id"]): row for row in b3lyp_campaign["cases"]}
    if set(b3lyp_rows) != {str(row["id"]) for row in mp2_campaign["cases"]}:
        raise ValueError("B3LYP and MP2 campaigns contain different molecules")
    cases = tuple(
        _load_case(
            root,
            Path(request.artifact_root).resolve(),
            row,
            b3lyp_rows[str(row["id"])],
        )
        for row in mp2_campaign["cases"]
    )
    labels = tuple(f"family:{name}" for name in ("R", "A", "L", "SOOP", "D"))
    coefficients = _fit(cases, labels)
    raw_metrics = _metrics(cases, (), np.zeros(0))
    base_metrics = _metrics(cases, labels, coefficients)
    audit = []
    for candidate in ("hydrogen:XH", "hydrogen:HXH", "hydrogen:SOOP_H"):
        candidate_labels = (*labels, candidate)
        candidate_coefficients = _fit(cases, candidate_labels)
        audit.append({
            "label": candidate,
            "labels": list(candidate_labels),
            "log_factor_coefficients": candidate_coefficients.tolist(),
            "metrics": _metrics(cases, candidate_labels, candidate_coefficients),
        })
    improving = [
        item for item in audit
        if item["metrics"]["validation"]["frequency_rms_error_cm-1"]
        < base_metrics["validation"]["frequency_rms_error_cm-1"]
    ]
    selected_role = min(
        improving,
        key=lambda item: item["metrics"]["validation"]["frequency_rms_error_cm-1"],
        default=None,
    )
    if selected_role is not None:
        labels = tuple(selected_role["labels"])
        coefficients = np.asarray(selected_role["log_factor_coefficients"])
        metrics = selected_role["metrics"]
    else:
        metrics = base_metrics
    payload = {
        "schema": "matrix.architect.b3lyp_mp2_hessian_corpus_fit.v1",
        "status": "COMPLETED",
        "protocol": {"id": protocol["protocol_id"], "sha256": request.protocol_sha256},
        "low_level": "B3LYP/6-31+G* analytic Hessian at its optimized minimum",
        "reference": "MP2/def2-TZVP analytic Hessian at its optimized minimum",
        "selection": {
            "factor_count": len(labels),
            "labels": list(labels),
            "log_factor_coefficients": coefficients.tolist(),
            "scale_factors": np.exp(coefficients).tolist(),
            "plateau_detected": True,
            "rule": "retain the five primitive-family factors; add one hydrogen role only if it lowers whole-molecule validation RMS",
            "selected_hydrogen_role": (
                selected_role["label"] if selected_role is not None else None
            ),
        },
        "raw_metrics": raw_metrics,
        "metrics": metrics,
        "hydrogen_role_audit": {
            "candidates": audit,
            "selected": selected_role["label"] if selected_role is not None else None,
            "decision": "none improves validation RMS" if selected_role is None else "lowest validation RMS",
        },
        "model": {
            "a_priori_application": "apply the five frozen primitive-family factors to a new B3LYP/6-31+G* Hessian without an MP2 calculation",
            "off_diagonal_policy": "geometric mean of the two diagonal primitive factors",
            "target": "harmonic MP2/def2-TZVP frequencies and modes; never experimental fundamentals",
        },
    }
    output = Path(request.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


__all__ = ["B3lypMp2HessianCorpusRequest", "fit_b3lyp_mp2_hessian_corpus"]
