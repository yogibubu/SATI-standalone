"""Whole-molecule fit of transferable LINK-XTB primitive-Hessian corrections."""

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
    _oracle_perceived_bond_orders,
    _proper_alignment,
    _reference_hessian,
    _rotate_modes,
    _xtb_scaling_observables,
    scale_primitive_hessian_geometric,
    xtb_primitive_features,
)


@dataclass(frozen=True)
class XtbHessianCorpusRequest:
    campaign_path: Path
    artifact_root: Path
    output_path: Path
    protocol_manifest: Mapping[str, Any]
    protocol_sha256: str
    maximum_factor_count: int = 24


@dataclass(frozen=True)
class _Case:
    case_id: str
    split: str
    features: tuple[dict[str, Any], ...]
    raw_k: np.ndarray
    reference_k: np.ndarray
    b_matrix: np.ndarray
    basis: np.ndarray
    masses: np.ndarray
    target_frequencies: np.ndarray
    target_modes: np.ndarray
    rotation: np.ndarray
    included: np.ndarray


def _load_case(root: Path, artifact_root: Path, row: Mapping[str, Any], protocol: Mapping[str, Any], digest: str) -> _Case:
    case_id = str(row["id"])
    art = artifact_root / case_id
    request = XtbHessianScaleRequest(
        reference_hessian_path=root / str(row["harmonic_record"]),
        xtb_hessian_path=art / "final_cartesian_hessian.json",
        primitive_manifest_path=art / "primitive_manifest.json",
        oracle_topology_path=art / "oracle-topology.json",
        xtb_observables_path=art / "gfn2_scaling_observables.json",
        output_path=Path("unused.json"), protocol_manifest=protocol,
        protocol_sha256=digest, reference_engine="lcb26",
    )
    reference = _reference_hessian(request)
    xtb = json.loads(request.xtb_hessian_path.read_text())
    manifest = json.loads(request.primitive_manifest_path.read_text())
    observables = json.loads(request.xtb_observables_path.read_text())
    atoms = tuple(str(atom) for atom in xtb["atoms"])
    if tuple(manifest["atoms"]) != atoms or tuple(observables["atoms"]) != atoms:
        raise ValueError(f"{case_id}: inconsistent atom order across resident artifacts")
    charges, mayer = _xtb_scaling_observables(observables, atoms)
    # The transferable corpus model deliberately has no atom-type lookup.
    # ``xtb_primitive_features`` also serves the legacy typed fit; element
    # symbols are harmless placeholders here because v3 consumes only the
    # primitive family and continuous same-state Mayer/SCC descriptors.
    topology = json.loads(request.oracle_topology_path.read_text())
    features = xtb_primitive_features(manifest, atoms, atoms, charges, mayer,
        perceived_bond_orders=_oracle_perceived_bond_orders(topology))
    xtb_coordinates = np.asarray(xtb["coordinates_angstrom"], dtype=float) / 0.52917721092
    raw_hessian = np.asarray(xtb["hessian"], dtype=float)
    ref_transform = transform_cartesian_hessian(
        reference.cartesian_hessian, reference.cartesian_coordinates_bohr,
        reference.masses_amu, manifest,
    )
    raw_transform = transform_cartesian_hessian(
        raw_hessian, xtb_coordinates, reference.masses_amu, manifest,
    )
    if ref_transform.primitive_labels != raw_transform.primitive_labels:
        raise ValueError(f"{case_id}: primitive contracts differ")
    frequencies, modes = _normal_modes(
        reference.cartesian_hessian, reference.masses_amu,
        ref_transform.redundant_hessian.vibrational_basis,
    )
    included = np.ones(len(frequencies), dtype=bool)
    if case_id == "saccharin":
        included[0] = False
    rotation, _rmsd = _proper_alignment(
        xtb_coordinates, reference.cartesian_coordinates_bohr, reference.masses_amu
    )
    return _Case(
        case_id=case_id, split=str(row["split"]), features=features,
        raw_k=raw_transform.redundant_hessian.force_constants,
        reference_k=ref_transform.redundant_hessian.force_constants,
        b_matrix=raw_transform.b_matrix,
        basis=raw_transform.redundant_hessian.vibrational_basis,
        masses=reference.masses_amu, target_frequencies=frequencies,
        target_modes=modes, rotation=rotation, included=included,
    )


def _feature_value(feature: Mapping[str, Any], label: str) -> float:
    family = str(feature["family"])
    kind, _, detail = label.partition(":")
    if kind == "bond_type" and detail == "SO_gt2":
        if family != "R" or not feature.get("terminal_sulfur_oxygen_stretch", False):
            return 0.0
        order = feature.get("perceived_bond_order")
        if order is None:
            raise ValueError("SO_gt2 scaling requires the ORACLE perception bond order, not Mayer")
        return float(float(order) > 2.0)
    if kind == "bond_type" and detail == "SO":
        return float(family == "R" and feature.get("terminal_sulfur_oxygen_stretch", False))
    if kind == "bond_order" and detail == "SO":
        return float(family == "R" and feature.get("terminal_sulfur_oxygen_stretch", False)) * math.log(
            max(float(feature["geometric_mean_mayer_order"]), 1.0e-6)
        )
    if kind == "family":
        return float(family == detail)
    if kind == "bond_order":
        return float(family == detail) * math.log(
            max(float(feature["geometric_mean_mayer_order"]), 1.0e-6)
        )
    if kind == "scc":
        return float(family == detail) * float(feature["mean_abs_scc_charge"])
    if kind == "hydrogen":
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
    raise ValueError(f"unknown LINK-XTB correction feature: {label}")


def _candidate_labels(cases: tuple[_Case, ...]) -> tuple[str, ...]:
    labels = [f"bond_order:{family}" for family in ("R", "A", "L", "SOOP", "D")]
    labels += [f"scc:{family}" for family in ("R", "A", "L", "SOOP", "D")]
    return tuple(labels)


def _fit(cases: tuple[_Case, ...], labels: tuple[str, ...]) -> np.ndarray:
    rows: list[list[float]] = []
    target: list[float] = []
    weights: list[float] = []
    for case in cases:
        if case.split != "train":
            continue
        raw = np.diag(case.raw_k)
        ref = np.diag(case.reference_k)
        valid = (raw > 1.0e-10) & (ref > 1.0e-10)
        count = max(int(np.sum(valid)), 1)
        for feature, raw_value, ref_value, keep in zip(case.features, raw, ref, valid, strict=True):
            if not keep:
                continue
            rows.append([_feature_value(feature, label) for label in labels])
            target.append(math.log(float(ref_value / raw_value)))
            weights.append(1.0 / math.sqrt(count))
    design = np.asarray(rows, dtype=float)
    response = np.asarray(target, dtype=float)
    weight = np.asarray(weights, dtype=float)
    weighted_design = design * weight[:, None]
    weighted_response = response * weight
    ridge = np.diag([1.0e-10 if label.startswith("family:") else 2.5e-3 for label in labels])
    return np.linalg.solve(weighted_design.T @ weighted_design + ridge, weighted_design.T @ weighted_response)


def _diagonal_sse(cases: tuple[_Case, ...], labels: tuple[str, ...], coefficients: np.ndarray) -> float:
    error = 0.0
    for case in cases:
        if case.split != "train":
            continue
        raw, ref = np.diag(case.raw_k), np.diag(case.reference_k)
        valid = (raw > 1.0e-10) & (ref > 1.0e-10)
        prediction = np.asarray([
            sum(value * _feature_value(feature, label) for value, label in zip(coefficients, labels, strict=True))
            for feature in case.features
        ])
        residual = prediction[valid] - np.log(ref[valid] / raw[valid])
        error += float(np.mean(residual * residual))
    return error


def _metrics(cases: tuple[_Case, ...], labels: tuple[str, ...], coefficients: np.ndarray) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for split in ("train", "validation", "external_validation"):
        frequency_errors: list[np.ndarray] = []
        overlaps: list[np.ndarray] = []
        for case in cases:
            if case.split != split:
                continue
            log_scales = np.asarray([
                sum(value * _feature_value(feature, label) for value, label in zip(coefficients, labels, strict=True))
                for feature in case.features
            ])
            scaled_k = scale_primitive_hessian_geometric(case.raw_k, np.exp(log_scales))
            hessian = case.b_matrix.T @ scaled_k @ case.b_matrix
            frequencies, modes = _normal_modes(hessian, case.masses, case.basis)
            rotated = _rotate_modes(modes, case.rotation)
            target = case.target_modes[:, case.included]
            calculated = rotated[:, case.included]
            overlap = np.abs(target.T @ calculated)
            rows, columns = linear_sum_assignment(-overlap)
            overlaps.append(overlap[rows, columns])
            frequency_errors.append(frequencies[case.included] - case.target_frequencies[case.included])
        if frequency_errors:
            errors = np.concatenate(frequency_errors)
            assigned = np.concatenate(overlaps)
            result[split] = {
                "molecule_count": sum(case.split == split for case in cases),
                "mode_count": int(len(errors)),
                "frequency_rms_error_cm-1": float(np.sqrt(np.mean(errors * errors))),
                "mean_assigned_mode_overlap": float(np.mean(assigned)),
                "minimum_assigned_mode_overlap": float(np.min(assigned)),
            }
    return result


def fit_xtb_hessian_corpus(request: XtbHessianCorpusRequest) -> dict[str, Any]:
    """Greedily fit shared factors and stop on the validation-set plateau."""
    protocol = dict(request.protocol_manifest)
    protocol_id = str(protocol.get("protocol_id", ""))
    if protocol_id not in {
        "xtb-primitive-hessian-scaling.v3",
        "xtb-primitive-hessian-scaling.v4",
    } or protocol.get("status") != "approved":
        raise ValueError("the corpus fit requires an approved atom-type-free protocol")
    campaign_path = Path(request.campaign_path).resolve()
    root = campaign_path.parents[2]
    campaign = json.loads(campaign_path.read_text())
    cases = tuple(
        _load_case(root, Path(request.artifact_root).resolve(), row, protocol, request.protocol_sha256)
        for row in campaign["cases"]
    )
    raw_metrics = _metrics(cases, (), np.zeros(0))
    present_families = tuple(
        family for family in ("R", "A", "L", "SOOP", "D")
        if any(feature["family"] == family for case in cases if case.split == "train" for feature in case.features)
    )
    family = tuple(f"family:{name}" for name in present_families)
    available = list(_candidate_labels(cases))
    labels = family
    coefficients = _fit(cases, labels)
    path: list[dict[str, Any]] = []

    def append_point() -> None:
        path.append({
            "factor_count": len(labels), "labels": list(labels),
            "coefficients": coefficients.tolist(), "metrics": _metrics(cases, labels, coefficients),
        })

    append_point()
    plateau = False
    plateau_gains: list[float] = []
    while available and len(labels) < int(request.maximum_factor_count):
        if not any(label.startswith("bond_order:") for label in labels):
            eligible = [label for label in available if label.startswith("bond_order:")]
        else:
            eligible = list(available)
        trials = []
        for candidate in eligible:
            trial_labels = (*labels, candidate)
            trial_coefficients = _fit(cases, trial_labels)
            trials.append((_diagonal_sse(cases, trial_labels, trial_coefficients), candidate, trial_coefficients))
        _sse, selected, coefficients = min(trials, key=lambda item: (item[0], item[1]))
        labels = (*labels, selected)
        available.remove(selected)
        append_point()
        if len(path) >= 3 and len(labels) >= 5:
            validation = [point["metrics"]["validation"] for point in path[-3:]]
            gains = []
            for left, right in zip(validation[:-1], validation[1:], strict=True):
                previous = max(float(left["frequency_rms_error_cm-1"]), 1.0e-12)
                gains.append(max(0.0, (previous - float(right["frequency_rms_error_cm-1"])) / previous))
            if (
                all(gain <= 0.06 for gain in gains)
                and validation[-1]["mean_assigned_mode_overlap"] >= 0.85
            ):
                path[-1]["plateau_relative_validation_rms_gains"] = gains
                plateau_gains = gains
                plateau = True
                break
    hydrogen_role_audit: dict[str, Any] | None = None
    if protocol_id == "xtb-primitive-hessian-scaling.v4":
        trials: list[tuple[float, str, np.ndarray, dict[str, dict[str, float]]]] = []
        for candidate in ("hydrogen:XH", "hydrogen:HXH", "hydrogen:SOOP_H"):
            trial_labels = (*labels, candidate)
            trial_coefficients = _fit(cases, trial_labels)
            trial_metrics = _metrics(cases, trial_labels, trial_coefficients)
            trials.append((
                float(trial_metrics["validation"]["frequency_rms_error_cm-1"]),
                candidate, trial_coefficients, trial_metrics,
            ))
        _rms, selected_hydrogen, coefficients, _trial_metrics = min(
            trials, key=lambda item: (item[0], item[1])
        )
        labels = (*labels, selected_hydrogen)
        append_point()
        hydrogen_role_audit = {
            "candidates": [
                {
                    "label": candidate,
                    "labels": [*labels[:-1], candidate],
                    "log_factor_coefficients": coefficient.tolist(),
                    "metrics": metrics,
                }
                for _value, candidate, coefficient, metrics in trials
            ],
            "selected": selected_hydrogen,
            "selection": "lowest whole-molecule validation frequency RMS; external saccharin remains confirmatory",
            "rejected": [candidate for candidate in ("hydrogen:XH", "hydrogen:HXH", "hydrogen:SOOP_H") if candidate != selected_hydrogen],
        }
    selected = path[-1]
    reference_target = str(
        campaign.get("reference_target", "harmonic reference Hessian")
    ).strip()
    selected_elemental_role = (
        str(hydrogen_role_audit["selected"])
        if hydrogen_role_audit is not None
        else "none"
    )
    payload = {
        "schema": "matrix.architect.xtb_hessian_corpus_fit.v3" if protocol_id.endswith(".v4") else "matrix.architect.xtb_hessian_corpus_fit.v2",
        "status": "COMPLETED",
        "protocol": {"id": protocol["protocol_id"], "sha256": request.protocol_sha256},
        "campaign": str(campaign_path),
        "split_molecules": {
            split: [case.case_id for case in cases if case.split == split]
            for split in ("train", "validation", "external_validation")
        },
        "selection": {
            "factor_count": selected["factor_count"], "labels": selected["labels"],
            "log_factor_coefficients": selected["coefficients"],
            "plateau_detected": plateau,
            "rule": (
                "v3 plateau followed by one registered XH/HXH/SOOP-H role audit"
                if protocol_id.endswith(".v4") else
                "first validation plateau from five factors: two successive relative RMS gains <= 6% and mean overlap >= 0.85"
            ),
            "base_plateau_relative_validation_rms_gains": plateau_gains,
        },
        "metrics": selected["metrics"],
        "raw_metrics": raw_metrics,
        "path": path,
        "model": {
            "scale": "exp(family baseline + Mayer-order slope * log(geometric-mean Mayer order) + SCC slope * mean absolute SCC charge + selected elemental-role response)",
            "atom_type_policy": "forbidden; no categorical atom or environment lookup enters the transferable model",
            "elemental_role_policy": (
                "v4 audits X-H stretch, H-X-H bend, and moving-H SOOP indicators "
                f"and retains at most one by whole-molecule validation; selected={selected_elemental_role}"
            ),
            "a_priori_application": "apply the frozen coefficients to any new LINK-XTB Hessian using observables from that same converged state; no molecule-specific reference or refit",
            "off_diagonal_policy": "geometric mean of diagonal primitive factors",
            "descriptor_source": "same-state LINK-XTB SCC charges and Mayer bond orders only",
            "reference_target": f"{reference_target}; never experimental fundamentals",
        },
    }
    if hydrogen_role_audit is not None:
        payload["hydrogen_role_audit"] = hydrogen_role_audit
    output = Path(request.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


__all__ = ["XtbHessianCorpusRequest", "fit_xtb_hessian_corpus"]
