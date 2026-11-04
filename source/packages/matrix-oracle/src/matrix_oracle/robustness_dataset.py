"""Versioned manuscript dataset for ORACLE perception robustness."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from matrix_chem import LocalPerceptionSettings, PerceptionNoiseModel, PerceptionNoiseSettings

from .perception_history import HysteresisPolicy, PerceptionTracker, TemporalDecisionEvidence
from .perception_robustness import (
    PerceptionAuditPolicy,
    audit_perception_robustness,
    deterministic_cartesian_perturbations,
    perceive_oracle_state,
)


ORACLE_ROBUSTNESS_DATASET_SCHEMA = "matrix.oracle.perception_robustness_dataset.v1"
ORACLE_ROBUSTNESS_DATASET_VERSION = "1.0.0"
ORACLE_ROBUSTNESS_MANUSCRIPT_CLAIM = (
    "ORACLE filtra e certifica la percezione in presenza di rumore e congela uno "
    "stato auditable; SONIC preserva rango e simmetria per quello stato."
)


@dataclass(frozen=True)
class RobustnessDatasetCase:
    case_id: str
    label: str
    atomic_numbers: tuple[int, ...]
    coordinates_angstrom: tuple[tuple[float, float, float], ...]
    expected_strict_groups: tuple[str, ...]
    expected_proposed_group: str
    polarity: str
    forbidden_proposed_group: str = ""
    required_local_marker: str = ""
    local_distance_tolerance_angstrom: float = 1.0e-3

    def __post_init__(self) -> None:
        if self.polarity not in {"POSITIVE", "NEGATIVE"}:
            raise ValueError("dataset case polarity must be POSITIVE or NEGATIVE")
        if len(self.atomic_numbers) != len(self.coordinates_angstrom):
            raise ValueError("dataset case geometry and element counts differ")
        if not self.expected_strict_groups or not self.expected_proposed_group:
            raise ValueError("dataset case requires explicit expected symmetry")


def default_robustness_corpus() -> tuple[RobustnessDatasetCase, ...]:
    tetrahedron = np.asarray(
        ((1.0, 1.0, 1.0), (1.0, -1.0, -1.0), (-1.0, 1.0, -1.0), (-1.0, -1.0, 1.0))
    ) / math.sqrt(3.0)
    methane = np.vstack((np.zeros((1, 3)), 1.09 * tetrahedron))
    sf6 = np.asarray(
        (
            (0.000, 0.000, 0.000),
            (1.575, 0.012, -0.008),
            (-1.552, -0.006, 0.004),
            (0.009, 1.568, 0.010),
            (-0.011, -1.557, -0.005),
            (0.006, -0.010, 1.582),
            (-0.004, 0.008, -1.548),
        )
    )
    zrf8 = np.asarray(
        (
            (0.000000, 0.000000, 0.000000),
            (1.875436, 0.000000, 0.945000),
            (0.000000, 1.875436, 0.945000),
            (-1.875436, 0.000000, 0.945000),
            (0.000000, -1.875436, 0.945000),
            (1.326134, 1.326134, -0.945000),
            (-1.326134, 1.326134, -0.945000),
            (-1.326134, -1.326134, -0.945000),
            (1.326134, -1.326134, -0.945000),
        )
    )
    return (
        RobustnessDatasetCase(
            "water_exact",
            "exact C2v water positive control",
            (8, 1, 1),
            ((0.0, 0.0, 0.0), (0.7586, 0.0, 0.5043), (-0.7586, 0.0, 0.5043)),
            ("C2v",),
            "C2v",
            "POSITIVE",
        ),
        RobustnessDatasetCase(
            "water_quasi",
            "quasi-symmetric water requiring an explicit decision",
            (8, 1, 1),
            ((0.0, 0.0, 0.0), (0.7586, 0.0, 0.5043), (-0.7580, 0.0102, 0.5080)),
            ("Cs", "C1"),
            "C2v",
            "POSITIVE",
        ),
        RobustnessDatasetCase(
            "water_asymmetric",
            "asymmetric water negative promotion control",
            (8, 1, 1),
            ((0.0, 0.0, 0.0), (0.9572, 0.0, 0.0), (-0.10, 1.05, 0.08)),
            ("Cs", "C1"),
            "Cs",
            "NEGATIVE",
            forbidden_proposed_group="C2v",
        ),
        RobustnessDatasetCase(
            "methane_td",
            "tetrahedral methane local-equivalence positive control",
            (6, 1, 1, 1, 1),
            tuple(tuple(float(value) for value in row) for row in methane),
            ("Td",),
            "Td",
            "POSITIVE",
            required_local_marker="GROUP=Td",
        ),
        RobustnessDatasetCase(
            "sf6_distorted",
            "distorted octahedral SF6 local-template positive control",
            (16, 9, 9, 9, 9, 9, 9),
            tuple(tuple(float(value) for value in row) for row in sf6),
            ("C1", "Ci"),
            "Ci",
            "POSITIVE",
            required_local_marker="TEMPLATE=OCTAHEDRAL",
            local_distance_tolerance_angstrom=5.0e-2,
        ),
        RobustnessDatasetCase(
            "zrf8_square_antiprismatic",
            "square-antiprismatic ZrF8 local-template positive control",
            (40, 9, 9, 9, 9, 9, 9, 9, 9),
            tuple(tuple(float(value) for value in row) for row in zrf8),
            ("D4d",),
            "D4d",
            "POSITIVE",
            required_local_marker="TEMPLATE=SQUARE_ANTIPRISMATIC",
        ),
    )


def generate_robustness_dataset(
    *,
    amplitudes_angstrom: Iterable[float] = (1.0e-5, 1.0e-4, 5.0e-4),
    perturbation_count: int = 8,
    parallel_workers: int = 1,
) -> dict:
    """Generate the canonical scientific dataset document in memory."""

    amplitudes = tuple(float(value) for value in amplitudes_angstrom)
    if not amplitudes or any(not math.isfinite(value) or value <= 0.0 for value in amplitudes):
        raise ValueError("dataset noise amplitudes must be positive finite angstrom values")
    rows = []
    for case in default_robustness_corpus():
        xyz = np.asarray(case.coordinates_angstrom, dtype=float)
        local_settings = LocalPerceptionSettings(
            distance_tolerance_angstrom=case.local_distance_tolerance_angstrom,
            zeff_tolerance=(1.0e-1 if case.case_id == "sf6_distorted" else 5.0e-4),
        )
        policy = PerceptionAuditPolicy(local_perception_settings=local_settings)
        reference = perceive_oracle_state(case.atomic_numbers, xyz, policy=policy)
        if reference.strict_group not in case.expected_strict_groups:
            raise RuntimeError(
                f"dataset reference {case.case_id} has unexpected strict group "
                f"{reference.strict_group}"
            )
        if not _sample_is_correct(case, reference):
            raise RuntimeError(f"dataset reference {case.case_id} violates its expected state")
        for amplitude in amplitudes:
            settings = PerceptionNoiseSettings(
                PerceptionNoiseModel(
                    category="NUMERICAL",
                    representation="BOUND",
                    natoms=len(case.atomic_numbers),
                    amplitude_angstrom=amplitude,
                ),
                perturbation_count=int(perturbation_count),
            )
            audit = audit_perception_robustness(
                case.atomic_numbers,
                xyz,
                settings,
                policy=policy,
                symmetry_decision="RETAIN",
                parallel_workers=parallel_workers,
            )
            displacements = deterministic_cartesian_perturbations(
                xyz, case.atomic_numbers, settings
            )
            samples = tuple(
                perceive_oracle_state(
                    case.atomic_numbers, xyz + displacement, policy=policy
                )
                for displacement in displacements
            )
            rows.append(_dataset_row(case, amplitude, reference, samples, audit))
    semantic_payload = {
        "schema": ORACLE_ROBUSTNESS_DATASET_SCHEMA,
        "dataset_version": ORACLE_ROBUSTNESS_DATASET_VERSION,
        "owner": "ORACLE",
        "noise_semantics": (
            "GEOMETRIC_NUMERICAL_NOISE_IN_ANGSTROM; NO_TEMPERATURE_LABEL_OR_THERMAL_CLAIM"
        ),
        "perturbation_scheme": "SYMMETRIC_SIGMA_POINTS",
        "perturbation_count": int(perturbation_count),
        "manuscript_claim": ORACLE_ROBUSTNESS_MANUSCRIPT_CLAIM,
        "rows": rows,
    }
    fingerprint_payload = _without_runtime(semantic_payload)
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {**semantic_payload, "semantic_sha256_excluding_runtime": fingerprint}


def write_robustness_dataset(
    json_path: Path,
    csv_path: Path,
    **generate_options,
) -> tuple[Path, Path]:
    """Generate and write the paired versioned JSON and flat CSV artifacts."""

    document = generate_robustness_dataset(**generate_options)
    json_target = Path(json_path)
    csv_target = Path(csv_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = document["rows"]
    with csv_target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=tuple(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return json_target, csv_target


def _dataset_row(case, amplitude, reference, samples, audit) -> dict:
    correct = tuple(_sample_is_correct(case, state) for state in samples)
    sampled_hashes = tuple(state.state_hash for state in samples)
    # Sigma-point perturbations are independent noise impulses, not an MD
    # ordering.  Embed each impulse in a return-to-reference trajectory before
    # measuring temporal chatter; otherwise array order invents persistence.
    temporal_hashes = tuple(
        item
        for sampled in sampled_hashes
        for item in (sampled, reference.state_hash)
    )
    state_hashes = (reference.state_hash, *temporal_hashes)
    transitions = sum(left != right for left, right in zip(state_hashes, state_hashes[1:]))
    hysteresis_events = _hysteresis_events(reference.state_hash, temporal_hashes)
    local_reference = reference.local_signatures
    false_splits = sum(
        case.polarity == "POSITIVE" and state.local_signatures != local_reference
        for state in samples
    )
    false_promotions = sum(
        bool(case.forbidden_proposed_group)
        and state.proposed_group == case.forbidden_proposed_group
        for state in samples
    )
    margins = tuple(
        item.signed_margin for item in audit.decisions if item.signed_margin is not None
    )
    count = len(samples)
    temporal_denominator = max(1, len(temporal_hashes))
    return {
        "case_id": case.case_id,
        "label": case.label,
        "polarity": case.polarity,
        "noise_category": "NUMERICAL_GEOMETRIC",
        "noise_representation": "BOUND",
        "noise_amplitude_angstrom": amplitude,
        "temperature_kelvin": "",
        "correct_state_fraction": sum(correct) / count,
        "false_promotion_rate": false_promotions / count,
        "false_splitting_rate": false_splits / count,
        "flip_rate_without_hysteresis": transitions / temporal_denominator,
        "flip_rate_with_hysteresis": len(hysteresis_events) / temporal_denominator,
        "handoff_delay_frames": _handoff_delay(correct, window=3),
        "minimum_signed_margin": min(margins) if margins else "",
        "strict_group": reference.strict_group,
        "proposed_group": reference.proposed_group,
        "topology_hash": reference.topology_hash,
        "state_hash": reference.state_hash,
        "audit_status": audit.status,
        "ordinary_runtime_seconds": audit.ordinary_runtime_seconds,
        "ensemble_runtime_seconds": audit.ensemble_runtime_seconds,
        "perturbation_count": count,
        "temporal_frame_count": len(temporal_hashes),
        "local_distance_tolerance_angstrom": case.local_distance_tolerance_angstrom,
    }


def _sample_is_correct(case: RobustnessDatasetCase, state) -> bool:
    local_ok = not case.required_local_marker or any(
        case.required_local_marker in signature for signature in state.local_signatures
    )
    return (
        state.proposed_group == case.expected_proposed_group
        and local_ok
    )


def _hysteresis_events(reference_hash: str, hashes: tuple[str, ...]) -> tuple:
    family = "STATE_STABILITY"
    identifier = f"{family}:CANONICAL_STATE"
    tracker = PerceptionTracker(
        (HysteresisPolicy(family, 0.75, 0.25, 3, 3),),
        initial_states={identifier: "PRESENT"},
    )
    emitted = []
    for state_hash in hashes:
        score = 1.0 if state_hash == reference_hash else 0.0
        evidence = TemporalDecisionEvidence(
            decision_id=identifier,
            family=family,
            score=score,
            components=(("CANONICAL_STATE_MATCH", score),),
            provider="ORACLE_ROBUSTNESS_DATASET",
            provider_version=ORACLE_ROBUSTNESS_DATASET_VERSION,
        )
        emitted.extend(tracker.update((evidence,)))
    return tuple(emitted)


def _handoff_delay(correct: tuple[bool, ...], *, window: int) -> int | str:
    run = 0
    for frame, value in enumerate(correct, start=1):
        run = run + 1 if value else 0
        if run >= window:
            return frame
    return ""


def _without_runtime(value):
    if isinstance(value, dict):
        return {
            key: _without_runtime(item)
            for key, item in value.items()
            if key not in {"ordinary_runtime_seconds", "ensemble_runtime_seconds"}
        }
    if isinstance(value, list):
        return [_without_runtime(item) for item in value]
    return value


__all__ = [
    "ORACLE_ROBUSTNESS_DATASET_SCHEMA",
    "ORACLE_ROBUSTNESS_DATASET_VERSION",
    "ORACLE_ROBUSTNESS_MANUSCRIPT_CLAIM",
    "RobustnessDatasetCase",
    "default_robustness_corpus",
    "generate_robustness_dataset",
    "write_robustness_dataset",
]
