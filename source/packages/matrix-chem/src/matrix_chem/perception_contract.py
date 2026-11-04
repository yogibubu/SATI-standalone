"""Immutable transport records for auditable ORACLE perception.

The records live in the shared chemistry package so that downstream tools can
validate and serialize them without depending on :mod:`matrix_oracle`.
Chemical decisions and every state transition remain owned by ORACLE.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np


PERCEPTION_NOISE_MODEL_SCHEMA = "matrix.oracle.perception_noise_model.v1"
PERCEPTION_NOISE_AUDIT_SCHEMA = "matrix.oracle.perception_noise_audit.v1"
PERCEPTION_HISTORY_SCHEMA = "matrix.oracle.perception_history.v1"
PERCEPTION_HANDOFF_SCHEMA = "matrix.oracle.perception_handoff.v1"

NOISE_CATEGORIES = frozenset({"NUMERICAL", "THERMAL"})
NOISE_REPRESENTATIONS = frozenset({"BOUND", "SIGMA", "COVARIANCE"})
AUDIT_STATUSES = frozenset(
    {"ROBUST", "AMBIGUOUS", "REQUIRES_DECISION", "TOPOLOGY_TRANSITION", "FAILED"}
)
HANDOFF_STATUSES = frozenset(
    {
        "EXPLORATION",
        "PROPOSED",
        "REQUIRES_DECISION",
        "FROZEN_EXPLOITATION",
        "STOP_ON_TOPOLOGY_CHANGE",
    }
)


@dataclass(frozen=True)
class PerceptionNoiseModel:
    """Declared numerical or physically derived thermal uncertainty.

    ``covariance_angstrom2`` is a Cartesian ``3N x 3N`` matrix.  A covariance
    is called thermal only when its physical origin, temperature, electronic
    model, basis, and external-mode treatment are all documented.
    """

    category: str
    representation: str
    natoms: int
    amplitude_angstrom: float | None = None
    covariance_angstrom2: tuple[tuple[float, ...], ...] = ()
    covariance_origin: str = ""
    temperature_kelvin: float | None = None
    electronic_method: str = ""
    basis: str = ""
    external_mode_treatment: str = ""
    frozen_atoms: tuple[int, ...] = ()
    constraint_projector: tuple[tuple[float, ...], ...] = ()
    schema: str = PERCEPTION_NOISE_MODEL_SCHEMA

    def __post_init__(self) -> None:
        category = self.category.strip().upper()
        representation = self.representation.strip().upper()
        if self.schema != PERCEPTION_NOISE_MODEL_SCHEMA:
            raise ValueError(f"unsupported perception-noise schema: {self.schema}")
        if category not in NOISE_CATEGORIES:
            raise ValueError(f"unsupported uncertainty category: {self.category}")
        if representation not in NOISE_REPRESENTATIONS:
            raise ValueError(f"unsupported uncertainty representation: {self.representation}")
        if int(self.natoms) < 1:
            raise ValueError("perception-noise model requires at least one atom")
        if len(set(self.frozen_atoms)) != len(self.frozen_atoms) or any(
            atom < 1 or atom > int(self.natoms) for atom in self.frozen_atoms
        ):
            raise ValueError("frozen atoms must be unique one-based indices")
        if representation in {"BOUND", "SIGMA"}:
            if self.amplitude_angstrom is None or not math.isfinite(
                self.amplitude_angstrom
            ) or self.amplitude_angstrom <= 0.0:
                raise ValueError("bound/sigma noise requires a positive amplitude in angstrom")
            if self.covariance_angstrom2:
                raise ValueError("bound/sigma noise must not also provide a covariance")
        else:
            _validate_covariance(self.covariance_angstrom2, int(self.natoms))
            if self.amplitude_angstrom is not None:
                raise ValueError("covariance noise must not also provide a scalar amplitude")
            if not self.covariance_origin.strip():
                raise ValueError("Cartesian covariance requires a documented origin")
        if category == "THERMAL":
            if representation != "COVARIANCE":
                raise ValueError("thermal uncertainty must be represented by a covariance")
            if (
                self.temperature_kelvin is None
                or not math.isfinite(self.temperature_kelvin)
                or self.temperature_kelvin <= 0.0
                or not self.electronic_method.strip()
                or not self.basis.strip()
                or not self.external_mode_treatment.strip()
            ):
                raise ValueError(
                    "thermal covariance requires temperature, method, basis, and external-mode treatment"
                )
        elif self.temperature_kelvin is not None:
            raise ValueError("numerical/geometric noise must not carry a temperature label")
        if self.constraint_projector:
            _validate_projector(self.constraint_projector, int(self.natoms))


@dataclass(frozen=True)
class PerceptionNoiseSettings:
    """Deterministic ensemble construction settings."""

    model: PerceptionNoiseModel
    perturbation_count: int = 16
    scheme: str = "SYMMETRIC_SIGMA_POINTS"
    seed: int | None = None
    remove_translations: bool = True
    remove_rotations: bool = True
    deterministic: bool = True

    def __post_init__(self) -> None:
        scheme = self.scheme.strip().upper()
        if int(self.perturbation_count) < 2:
            raise ValueError("noise audit requires at least two perturbations")
        if scheme not in {
            "SYMMETRIC_SIGMA_POINTS",
            "FIXED_LOW_DISCREPANCY",
            "SEEDED_PSEUDORANDOM",
        }:
            raise ValueError(f"unsupported perturbation scheme: {self.scheme}")
        if scheme == "SEEDED_PSEUDORANDOM" and self.seed is None:
            raise ValueError("pseudorandom perturbations require a serialized seed")
        if self.deterministic and scheme == "SEEDED_PSEUDORANDOM" and self.seed is None:
            raise ValueError("deterministic pseudorandom audit requires a seed")


@dataclass(frozen=True)
class DecisionThreshold:
    name: str
    value: float
    unit: str
    entry_value: float | None = None
    exit_value: float | None = None
    provider: str = ""
    provider_version: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.unit.strip() or not math.isfinite(self.value):
            raise ValueError("decision threshold requires a finite value, name, and unit")
        for item in (self.entry_value, self.exit_value):
            if item is not None and not math.isfinite(item):
                raise ValueError("entry/exit thresholds must be finite")
        if (self.entry_value is None) != (self.exit_value is None):
            raise ValueError("entry and exit thresholds must be provided together")
        if self.entry_value is not None and self.entry_value == self.exit_value:
            raise ValueError("entry and exit thresholds must differ for hysteresis")


@dataclass(frozen=True)
class DecisionRobustness:
    decision_id: str
    family: str
    accepted_class: str
    competing_class: str | None
    raw_score: float | None
    normalized_score: float | None
    signed_margin: float | None
    stability_fraction: float
    worst_case_perturbation: int
    fallback: str | None
    decision_reason: str
    thresholds: tuple[DecisionThreshold, ...]
    provider: str
    provider_version: str

    def __post_init__(self) -> None:
        if not all(
            (
                self.decision_id.strip(),
                self.family.strip(),
                self.accepted_class.strip(),
                self.decision_reason.strip(),
                self.provider.strip(),
                self.provider_version.strip(),
            )
        ):
            raise ValueError("decision robustness record is incomplete")
        if not 0.0 <= self.stability_fraction <= 1.0:
            raise ValueError("stability fraction must lie in [0, 1]")
        if int(self.worst_case_perturbation) < 0:
            raise ValueError("worst-case perturbation index must be non-negative")
        for value in (self.raw_score, self.normalized_score, self.signed_margin):
            if value is not None and not math.isfinite(value):
                raise ValueError("decision scores and margins must be finite")


@dataclass(frozen=True)
class PerceptionTransitionEvent:
    frame_index: int
    time_value: float | None
    decision_id: str
    family: str
    previous_state: str
    new_state: str
    reason: str
    evidence_score: float
    threshold_name: str
    threshold_value: float
    persistence_count: int
    provider: str
    provider_version: str

    def __post_init__(self) -> None:
        if self.frame_index < 0 or self.persistence_count < 1:
            raise ValueError("transition event has an invalid frame/count")
        if self.time_value is not None and not math.isfinite(self.time_value):
            raise ValueError("transition time must be finite")
        if not math.isfinite(self.evidence_score) or not math.isfinite(self.threshold_value):
            raise ValueError("transition evidence and threshold must be finite")
        if not all(
            (
                self.decision_id,
                self.family,
                self.previous_state,
                self.new_state,
                self.reason,
                self.threshold_name,
                self.provider,
                self.provider_version,
            )
        ):
            raise ValueError("transition event is incomplete")


@dataclass(frozen=True)
class PerceptionHistory:
    frame_count: int
    active_states: tuple[tuple[str, str], ...]
    entry_counters: tuple[tuple[str, int], ...]
    exit_counters: tuple[tuple[str, int], ...]
    events: tuple[PerceptionTransitionEvent, ...]
    fingerprint: str
    schema: str = PERCEPTION_HISTORY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PERCEPTION_HISTORY_SCHEMA or self.frame_count < 0:
            raise ValueError("invalid perception history")
        if len({key for key, _value in self.active_states}) != len(self.active_states):
            raise ValueError("perception history contains duplicate active-state keys")
        if any(value < 0 for _key, value in (*self.entry_counters, *self.exit_counters)):
            raise ValueError("perception history counters must be non-negative")
        if self.fingerprint != perception_history_fingerprint(
            self.frame_count,
            self.active_states,
            self.entry_counters,
            self.exit_counters,
            self.events,
        ):
            raise ValueError("perception history fingerprint is inconsistent")


@dataclass(frozen=True)
class PerceptionNoiseAudit:
    status: str
    noise_settings: PerceptionNoiseSettings
    reference_state_hash: str
    sampled_state_hashes: tuple[str, ...]
    decisions: tuple[DecisionRobustness, ...]
    strict_group: str
    proposed_group: str
    symmetry_decision: str
    handoff_status: str
    worst_case_perturbation: int
    ordinary_runtime_seconds: float
    ensemble_runtime_seconds: float
    transition_events: tuple[PerceptionTransitionEvent, ...] = ()
    history_fingerprint: str = ""
    diagnostics: tuple[str, ...] = ()
    schema: str = PERCEPTION_NOISE_AUDIT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PERCEPTION_NOISE_AUDIT_SCHEMA or self.status not in AUDIT_STATUSES:
            raise ValueError("invalid perception audit schema/status")
        if self.handoff_status not in HANDOFF_STATUSES:
            raise ValueError("invalid perception handoff status")
        if not self.reference_state_hash or len(self.sampled_state_hashes) != int(
            self.noise_settings.perturbation_count
        ):
            raise ValueError("perception audit state hashes are incomplete")
        if not self.strict_group or not self.proposed_group:
            raise ValueError("perception audit requires strict and proposed groups")
        if self.symmetry_decision not in {"PROJECT", "RETAIN", "REQUIRES_DECISION"}:
            raise ValueError("invalid explicit symmetry decision")
        if self.worst_case_perturbation < 0:
            raise ValueError("invalid worst-case perturbation")
        if (
            not math.isfinite(self.ordinary_runtime_seconds)
            or not math.isfinite(self.ensemble_runtime_seconds)
            or min(self.ordinary_runtime_seconds, self.ensemble_runtime_seconds) < 0.0
        ):
            raise ValueError("perception audit runtimes must be finite and non-negative")


@dataclass(frozen=True)
class FrozenPerceptionHandoff:
    state_hash: str
    topology_hash: str
    accepted_group: str
    symmetry_decision: str
    audit_hash: str
    contract_hash: str
    stable_frame_count: int
    provenance: str
    schema: str = PERCEPTION_HANDOFF_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PERCEPTION_HANDOFF_SCHEMA:
            raise ValueError("unsupported perception handoff schema")
        if self.symmetry_decision not in {"PROJECT", "RETAIN"}:
            raise ValueError("frozen handoff requires an explicit symmetry decision")
        if self.stable_frame_count < 1 or not all(
            (
                self.state_hash,
                self.topology_hash,
                self.accepted_group,
                self.audit_hash,
                self.contract_hash,
                self.provenance,
            )
        ):
            raise ValueError("frozen perception handoff is incomplete")


def perception_history_fingerprint(
    frame_count: int,
    active_states: tuple[tuple[str, str], ...],
    entry_counters: tuple[tuple[str, int], ...],
    exit_counters: tuple[tuple[str, int], ...],
    events: tuple[PerceptionTransitionEvent, ...],
) -> str:
    payload = {
        "frame_count": int(frame_count),
        "active_states": sorted((str(key), str(value)) for key, value in active_states),
        "entry_counters": sorted((str(key), int(value)) for key, value in entry_counters),
        "exit_counters": sorted((str(key), int(value)) for key, value in exit_counters),
        "events": [asdict(item) for item in events],
    }
    return _stable_sha256(payload)


def perception_contract_to_dict(value: Any) -> dict[str, Any]:
    """Return a canonical JSON-compatible dictionary for a transport record."""

    if not hasattr(value, "__dataclass_fields__"):
        raise TypeError("perception contract serialization requires a dataclass record")
    return asdict(value)


def perception_contract_sha256(value: Any) -> str:
    return _stable_sha256(perception_contract_to_dict(value))


def perception_noise_model_from_dict(data: dict[str, Any]) -> PerceptionNoiseModel:
    return PerceptionNoiseModel(
        category=str(data["category"]),
        representation=str(data["representation"]),
        natoms=int(data["natoms"]),
        amplitude_angstrom=(
            None if data.get("amplitude_angstrom") is None else float(data["amplitude_angstrom"])
        ),
        covariance_angstrom2=tuple(
            tuple(float(value) for value in row)
            for row in data.get("covariance_angstrom2", ())
        ),
        covariance_origin=str(data.get("covariance_origin", "")),
        temperature_kelvin=(
            None if data.get("temperature_kelvin") is None else float(data["temperature_kelvin"])
        ),
        electronic_method=str(data.get("electronic_method", "")),
        basis=str(data.get("basis", "")),
        external_mode_treatment=str(data.get("external_mode_treatment", "")),
        frozen_atoms=tuple(int(value) for value in data.get("frozen_atoms", ())),
        constraint_projector=tuple(
            tuple(float(value) for value in row)
            for row in data.get("constraint_projector", ())
        ),
        schema=str(data.get("schema", PERCEPTION_NOISE_MODEL_SCHEMA)),
    )


def perception_noise_settings_from_dict(data: dict[str, Any]) -> PerceptionNoiseSettings:
    return PerceptionNoiseSettings(
        model=perception_noise_model_from_dict(data["model"]),
        perturbation_count=int(data["perturbation_count"]),
        scheme=str(data["scheme"]),
        seed=None if data.get("seed") is None else int(data["seed"]),
        remove_translations=bool(data.get("remove_translations", True)),
        remove_rotations=bool(data.get("remove_rotations", True)),
        deterministic=bool(data.get("deterministic", True)),
    )


def decision_threshold_from_dict(data: dict[str, Any]) -> DecisionThreshold:
    return DecisionThreshold(
        name=str(data["name"]),
        value=float(data["value"]),
        unit=str(data["unit"]),
        entry_value=None if data.get("entry_value") is None else float(data["entry_value"]),
        exit_value=None if data.get("exit_value") is None else float(data["exit_value"]),
        provider=str(data.get("provider", "")),
        provider_version=str(data.get("provider_version", "")),
    )


def decision_robustness_from_dict(data: dict[str, Any]) -> DecisionRobustness:
    optional = lambda name: None if data.get(name) is None else float(data[name])
    return DecisionRobustness(
        decision_id=str(data["decision_id"]),
        family=str(data["family"]),
        accepted_class=str(data["accepted_class"]),
        competing_class=(
            None if data.get("competing_class") is None else str(data["competing_class"])
        ),
        raw_score=optional("raw_score"),
        normalized_score=optional("normalized_score"),
        signed_margin=optional("signed_margin"),
        stability_fraction=float(data["stability_fraction"]),
        worst_case_perturbation=int(data["worst_case_perturbation"]),
        fallback=None if data.get("fallback") is None else str(data["fallback"]),
        decision_reason=str(data["decision_reason"]),
        thresholds=tuple(
            decision_threshold_from_dict(item) for item in data.get("thresholds", ())
        ),
        provider=str(data["provider"]),
        provider_version=str(data["provider_version"]),
    )


def perception_transition_event_from_dict(data: dict[str, Any]) -> PerceptionTransitionEvent:
    return PerceptionTransitionEvent(
        frame_index=int(data["frame_index"]),
        time_value=None if data.get("time_value") is None else float(data["time_value"]),
        decision_id=str(data["decision_id"]),
        family=str(data["family"]),
        previous_state=str(data["previous_state"]),
        new_state=str(data["new_state"]),
        reason=str(data["reason"]),
        evidence_score=float(data["evidence_score"]),
        threshold_name=str(data["threshold_name"]),
        threshold_value=float(data["threshold_value"]),
        persistence_count=int(data["persistence_count"]),
        provider=str(data["provider"]),
        provider_version=str(data["provider_version"]),
    )


def perception_history_from_dict(data: dict[str, Any]) -> PerceptionHistory:
    return PerceptionHistory(
        frame_count=int(data["frame_count"]),
        active_states=tuple((str(key), str(value)) for key, value in data["active_states"]),
        entry_counters=tuple((str(key), int(value)) for key, value in data["entry_counters"]),
        exit_counters=tuple((str(key), int(value)) for key, value in data["exit_counters"]),
        events=tuple(
            perception_transition_event_from_dict(item) for item in data.get("events", ())
        ),
        fingerprint=str(data["fingerprint"]),
        schema=str(data.get("schema", PERCEPTION_HISTORY_SCHEMA)),
    )


def perception_noise_audit_from_dict(data: dict[str, Any]) -> PerceptionNoiseAudit:
    return PerceptionNoiseAudit(
        status=str(data["status"]),
        noise_settings=perception_noise_settings_from_dict(data["noise_settings"]),
        reference_state_hash=str(data["reference_state_hash"]),
        sampled_state_hashes=tuple(str(value) for value in data["sampled_state_hashes"]),
        decisions=tuple(
            decision_robustness_from_dict(item) for item in data.get("decisions", ())
        ),
        strict_group=str(data["strict_group"]),
        proposed_group=str(data["proposed_group"]),
        symmetry_decision=str(data["symmetry_decision"]),
        handoff_status=str(data["handoff_status"]),
        worst_case_perturbation=int(data["worst_case_perturbation"]),
        ordinary_runtime_seconds=float(data["ordinary_runtime_seconds"]),
        ensemble_runtime_seconds=float(data["ensemble_runtime_seconds"]),
        transition_events=tuple(
            perception_transition_event_from_dict(item)
            for item in data.get("transition_events", ())
        ),
        history_fingerprint=str(data.get("history_fingerprint", "")),
        diagnostics=tuple(str(value) for value in data.get("diagnostics", ())),
        schema=str(data.get("schema", PERCEPTION_NOISE_AUDIT_SCHEMA)),
    )


def frozen_perception_handoff_from_dict(data: dict[str, Any]) -> FrozenPerceptionHandoff:
    return FrozenPerceptionHandoff(
        state_hash=str(data["state_hash"]),
        topology_hash=str(data["topology_hash"]),
        accepted_group=str(data["accepted_group"]),
        symmetry_decision=str(data["symmetry_decision"]),
        audit_hash=str(data["audit_hash"]),
        contract_hash=str(data["contract_hash"]),
        stable_frame_count=int(data["stable_frame_count"]),
        provenance=str(data["provenance"]),
        schema=str(data.get("schema", PERCEPTION_HANDOFF_SCHEMA)),
    )


def _validate_covariance(values: tuple[tuple[float, ...], ...], natoms: int) -> None:
    expected = 3 * int(natoms)
    covariance = np.asarray(values, dtype=float)
    if covariance.shape != (expected, expected) or np.any(~np.isfinite(covariance)):
        raise ValueError("Cartesian covariance must be a finite 3N x 3N matrix")
    if not np.allclose(covariance, covariance.T, atol=1.0e-12, rtol=1.0e-10):
        raise ValueError("Cartesian covariance must be symmetric")
    scale = max(1.0, float(np.linalg.norm(covariance, ord=2)))
    if float(np.min(np.linalg.eigvalsh(covariance))) < -1.0e-10 * scale:
        raise ValueError("Cartesian covariance must be positive semidefinite")


def _validate_projector(values: tuple[tuple[float, ...], ...], natoms: int) -> None:
    expected = 3 * int(natoms)
    projector = np.asarray(values, dtype=float)
    if projector.shape != (expected, expected) or np.any(~np.isfinite(projector)):
        raise ValueError("constraint projector must be a finite 3N x 3N matrix")
    if not np.allclose(projector, projector.T, atol=1.0e-10, rtol=1.0e-9):
        raise ValueError("constraint projector must be symmetric")
    if not np.allclose(projector @ projector, projector, atol=1.0e-9, rtol=1.0e-8):
        raise ValueError("constraint projector must be idempotent")


def _stable_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "AUDIT_STATUSES",
    "DecisionRobustness",
    "DecisionThreshold",
    "FrozenPerceptionHandoff",
    "HANDOFF_STATUSES",
    "NOISE_CATEGORIES",
    "NOISE_REPRESENTATIONS",
    "PERCEPTION_HISTORY_SCHEMA",
    "PERCEPTION_HANDOFF_SCHEMA",
    "PERCEPTION_NOISE_AUDIT_SCHEMA",
    "PERCEPTION_NOISE_MODEL_SCHEMA",
    "PerceptionHistory",
    "PerceptionNoiseAudit",
    "PerceptionNoiseModel",
    "PerceptionNoiseSettings",
    "PerceptionTransitionEvent",
    "perception_contract_sha256",
    "perception_contract_to_dict",
    "decision_robustness_from_dict",
    "decision_threshold_from_dict",
    "frozen_perception_handoff_from_dict",
    "perception_history_fingerprint",
    "perception_history_from_dict",
    "perception_noise_audit_from_dict",
    "perception_noise_model_from_dict",
    "perception_noise_settings_from_dict",
    "perception_transition_event_from_dict",
]
