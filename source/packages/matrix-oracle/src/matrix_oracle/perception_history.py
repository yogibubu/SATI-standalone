"""General hysteresis and temporal persistence for ORACLE decisions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping

from matrix_chem import (
    PerceptionHistory,
    PerceptionTransitionEvent,
    perception_history_fingerprint,
)


ORACLE_PERCEPTION_TRACKER_PROVIDER = "ORACLE_PERCEPTION_HYSTERESIS_TRACKER"
ORACLE_PERCEPTION_TRACKER_PROVIDER_VERSION = "1"


@dataclass(frozen=True)
class HysteresisPolicy:
    """Entry/exit policy for a normalized evidence score.

    For ``HIGHER_IS_ACTIVE``, entry must be above exit.  For
    ``LOWER_IS_ACTIVE`` (for example a normalized distance), entry must be
    below exit.  This makes a single threshold impossible by construction.
    """

    family: str
    entry_threshold: float
    exit_threshold: float
    entry_window: int
    exit_window: int
    direction: str = "HIGHER_IS_ACTIVE"
    threshold_name: str = "NORMALIZED_EVIDENCE"
    unit: str = "DIMENSIONLESS"
    provider: str = ORACLE_PERCEPTION_TRACKER_PROVIDER
    provider_version: str = ORACLE_PERCEPTION_TRACKER_PROVIDER_VERSION

    def __post_init__(self) -> None:
        direction = self.direction.strip().upper()
        if direction not in {"HIGHER_IS_ACTIVE", "LOWER_IS_ACTIVE"}:
            raise ValueError("hysteresis direction is unsupported")
        if not math.isfinite(self.entry_threshold) or not math.isfinite(self.exit_threshold):
            raise ValueError("hysteresis thresholds must be finite")
        if direction == "HIGHER_IS_ACTIVE" and self.entry_threshold <= self.exit_threshold:
            raise ValueError("higher-is-active entry threshold must exceed exit threshold")
        if direction == "LOWER_IS_ACTIVE" and self.entry_threshold >= self.exit_threshold:
            raise ValueError("lower-is-active entry threshold must be below exit threshold")
        if self.entry_window < 1 or self.exit_window < 1:
            raise ValueError("hysteresis persistence windows must be positive")
        if not all(
            (self.family, self.threshold_name, self.unit, self.provider, self.provider_version)
        ):
            raise ValueError("hysteresis policy provenance is incomplete")

    def entry_satisfied(self, score: float) -> bool:
        return (
            score >= self.entry_threshold
            if self.direction.strip().upper() == "HIGHER_IS_ACTIVE"
            else score <= self.entry_threshold
        )

    def exit_satisfied(self, score: float) -> bool:
        return (
            score <= self.exit_threshold
            if self.direction.strip().upper() == "HIGHER_IS_ACTIVE"
            else score >= self.exit_threshold
        )


@dataclass(frozen=True)
class TemporalDecisionEvidence:
    decision_id: str
    family: str
    score: float
    components: tuple[tuple[str, float], ...]
    provider: str
    provider_version: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.score) or any(
            not name or not math.isfinite(value) for name, value in self.components
        ):
            raise ValueError("temporal evidence must be finite and named")
        if not all((self.decision_id, self.family, self.provider, self.provider_version)):
            raise ValueError("temporal evidence provenance is incomplete")


@dataclass(frozen=True)
class HysteresisCalibrationTrajectory:
    scores: tuple[float, ...]
    expected_states: tuple[str, ...]
    initial_state: str

    def __post_init__(self) -> None:
        if not self.scores or len(self.scores) != len(self.expected_states):
            raise ValueError("hysteresis calibration trajectory has inconsistent frames")
        if any(not math.isfinite(float(score)) for score in self.scores):
            raise ValueError("hysteresis calibration scores must be finite")
        _normalized_binary_state(self.initial_state)
        for state in self.expected_states:
            _normalized_binary_state(state)


@dataclass(frozen=True)
class HysteresisCalibration:
    policy: HysteresisPolicy
    state_error_rate: float
    chatter_rate: float
    transition_delay_frames: float
    objective: float
    trajectory_count: int
    schema: str = "matrix.oracle.hysteresis_calibration.v1"


def calibrate_hysteresis_policy(
    family: str,
    trajectories: Iterable[HysteresisCalibrationTrajectory],
    candidate_policies: Iterable[HysteresisPolicy],
    *,
    chatter_weight: float = 2.0,
    delay_weight: float = 0.1,
) -> HysteresisCalibration:
    """Select a policy on ordered trajectories, never on shuffled snapshots."""

    records = tuple(trajectories)
    candidates = tuple(candidate_policies)
    normalized_family = str(family).strip().upper()
    if not records or not candidates:
        raise ValueError("hysteresis calibration requires trajectories and candidates")
    if any(item.family.strip().upper() != normalized_family for item in candidates):
        raise ValueError("hysteresis calibration candidates must share one family")
    if chatter_weight < 0.0 or delay_weight < 0.0:
        raise ValueError("hysteresis calibration weights must be non-negative")

    scored: list[HysteresisCalibration] = []
    identifier = f"{normalized_family}:CALIBRATION"
    for policy in candidates:
        mismatches = transitions = delay_total = frames = expected_transitions = 0
        for trajectory in records:
            tracker = PerceptionTracker(
                (policy,), initial_states={identifier: trajectory.initial_state}
            )
            prior_expected = _normalized_binary_state(trajectory.initial_state)
            last_expected_transition = None
            for frame, (score, expected) in enumerate(
                zip(trajectory.scores, trajectory.expected_states, strict=True)
            ):
                expected_state = _normalized_binary_state(expected)
                if expected_state != prior_expected:
                    expected_transitions += 1
                    last_expected_transition = frame
                    prior_expected = expected_state
                evidence = TemporalDecisionEvidence(
                    decision_id=identifier,
                    family=normalized_family,
                    score=float(score),
                    components=(("CALIBRATION_SCORE", float(score)),),
                    provider="ORACLE_HYSTERESIS_CALIBRATION",
                    provider_version="1",
                )
                emitted = tracker.update((evidence,))
                transitions += len(emitted)
                observed = dict(tracker.snapshot().active_states)[identifier]
                mismatches += observed != expected_state
                frames += 1
                if emitted and last_expected_transition is not None:
                    delay_total += max(0, frame - last_expected_transition)
                    last_expected_transition = None
        state_error = mismatches / max(frames, 1)
        chatter = max(0, transitions - expected_transitions) / max(frames, 1)
        delay = delay_total / max(expected_transitions, 1)
        objective = state_error + chatter_weight * chatter + delay_weight * delay / max(
            max(len(item.scores) for item in records), 1
        )
        scored.append(HysteresisCalibration(
            policy=policy,
            state_error_rate=state_error,
            chatter_rate=chatter,
            transition_delay_frames=delay,
            objective=objective,
            trajectory_count=len(records),
        ))
    return min(
        scored,
        key=lambda item: (
            item.objective,
            item.chatter_rate,
            item.transition_delay_frames,
            item.policy.entry_window + item.policy.exit_window,
            item.policy.entry_threshold,
            item.policy.exit_threshold,
        ),
    )


DEFAULT_HYSTERESIS_POLICIES = (
    HysteresisPolicy("AUXILIARY_CONTACT", 0.75, 0.45, 3, 3),
    HysteresisPolicy("COVALENT_BOND", 0.80, 0.55, 3, 3),
    HysteresisPolicy("COORDINATION", 0.80, 0.55, 3, 3),
    HysteresisPolicy("LOCAL_EQUIVALENCE", 0.85, 0.60, 3, 3),
    HysteresisPolicy("LOCAL_TEMPLATE", 0.85, 0.60, 3, 3),
    HysteresisPolicy("STRUCTURAL_SITE", 0.80, 0.55, 3, 3),
    HysteresisPolicy("QUASI_SYMMETRY", 0.90, 0.65, 4, 4),
)


class PerceptionTracker:
    """Stateful evidence integrator with immutable snapshots and event log."""

    def __init__(
        self,
        policies: Iterable[HysteresisPolicy] = DEFAULT_HYSTERESIS_POLICIES,
        *,
        initial_states: Mapping[str, str],
        new_decision_initial_state: str | None = None,
    ) -> None:
        policy_records = tuple(policies)
        self._policies = {item.family.upper(): item for item in policy_records}
        if len(self._policies) != len(policy_records):
            raise ValueError("hysteresis policies must have unique families")
        self._states = {
            str(key): _normalized_binary_state(value) for key, value in initial_states.items()
        }
        self._new_state = (
            None
            if new_decision_initial_state is None
            else _normalized_binary_state(new_decision_initial_state)
        )
        self._entry = {key: 0 for key in self._states}
        self._exit = {key: 0 for key in self._states}
        self._events: list[PerceptionTransitionEvent] = []
        self._frame_count = 0

    def update(
        self,
        evidence: Iterable[TemporalDecisionEvidence],
        *,
        frame_index: int | None = None,
        time_value: float | None = None,
    ) -> tuple[PerceptionTransitionEvent, ...]:
        """Integrate one frame and return only transitions emitted by it."""

        frame = self._frame_count if frame_index is None else int(frame_index)
        if frame != self._frame_count:
            raise ValueError("perception tracker frames must be consecutive and zero based")
        records = tuple(sorted(evidence, key=lambda item: item.decision_id))
        if len({item.decision_id for item in records}) != len(records):
            raise ValueError("a frame contains duplicate temporal decision evidence")
        by_id = {item.decision_id: item for item in records}
        for identifier in by_id:
            if identifier not in self._states:
                if self._new_state is None:
                    raise ValueError(
                        f"decision {identifier} has no explicit initial temporal state"
                    )
                self._states[identifier] = self._new_state
                self._entry[identifier] = 0
                self._exit[identifier] = 0

        emitted: list[PerceptionTransitionEvent] = []
        for identifier in sorted(self._states):
            item = by_id.get(identifier)
            family = item.family.upper() if item is not None else _family_from_identifier(identifier)
            policy = self._policies.get(family) or self._policies.get("AUXILIARY_CONTACT")
            if policy is None:
                raise ValueError(f"no hysteresis policy is defined for family {family}")
            state = self._states[identifier]
            score = item.score if item is not None else _missing_evidence_score(policy)
            if state == "ABSENT":
                self._exit[identifier] = 0
                self._entry[identifier] = (
                    self._entry[identifier] + 1 if policy.entry_satisfied(score) else 0
                )
                if self._entry[identifier] >= policy.entry_window:
                    event = _transition_event(
                        frame,
                        time_value,
                        identifier,
                        family,
                        "ABSENT",
                        "PRESENT",
                        "ENTRY_PERSISTED",
                        score,
                        policy,
                        self._entry[identifier],
                    )
                    self._states[identifier] = "PRESENT"
                    self._entry[identifier] = 0
                    emitted.append(event)
            else:
                self._entry[identifier] = 0
                self._exit[identifier] = (
                    self._exit[identifier] + 1 if policy.exit_satisfied(score) else 0
                )
                if self._exit[identifier] >= policy.exit_window:
                    event = _transition_event(
                        frame,
                        time_value,
                        identifier,
                        family,
                        "PRESENT",
                        "ABSENT",
                        "EXIT_PERSISTED",
                        score,
                        policy,
                        self._exit[identifier],
                    )
                    self._states[identifier] = "ABSENT"
                    self._exit[identifier] = 0
                    emitted.append(event)
        self._events.extend(emitted)
        self._frame_count += 1
        return tuple(emitted)

    def snapshot(self) -> PerceptionHistory:
        active = tuple(sorted(self._states.items()))
        entry = tuple(sorted(self._entry.items()))
        exit_values = tuple(sorted(self._exit.items()))
        events = tuple(self._events)
        fingerprint = perception_history_fingerprint(
            self._frame_count, active, entry, exit_values, events
        )
        return PerceptionHistory(
            frame_count=self._frame_count,
            active_states=active,
            entry_counters=entry,
            exit_counters=exit_values,
            events=events,
            fingerprint=fingerprint,
        )


def contact_temporal_evidence(
    contact,
    *,
    maximum_rho_vdw: float,
    minimum_confidence: float,
) -> TemporalDecisionEvidence:
    """Build temporal evidence while retaining independent contact metrics."""

    radial = max(0.0, min(1.0, 1.0 - float(contact.rho_vdw) / maximum_rho_vdw + 0.5))
    confidence = float(contact.confidence)
    directional = min(
        (
            float(value)
            for name, value in contact.directional_descriptors
            if "SCORE" in name or "STRENGTH" in name
        ),
        default=confidence,
    )
    score = min(confidence, radial, directional)
    endpoints = tuple(
        sorted(
            (
                f"{contact.endpoint_a.kind}:{contact.endpoint_a.identifier}",
                f"{contact.endpoint_b.kind}:{contact.endpoint_b.identifier}",
            )
        )
    )
    return TemporalDecisionEvidence(
        decision_id=f"AUXILIARY_CONTACT:{contact.kind}:{'|'.join(endpoints)}",
        family="AUXILIARY_CONTACT",
        score=score,
        components=(
            ("CONFIDENCE", confidence),
            ("RHO_VDW", float(contact.rho_vdw)),
            ("DIRECTIONAL_SCORE", directional),
            ("SINGLE_GEOMETRY_PERSISTENCE", float(contact.persistence)),
            ("MINIMUM_CONFIDENCE", float(minimum_confidence)),
        ),
        provider=contact.provider,
        provider_version=contact.provider_version,
    )


def _transition_event(
    frame,
    time_value,
    identifier,
    family,
    previous,
    new,
    reason,
    score,
    policy,
    count,
) -> PerceptionTransitionEvent:
    threshold = policy.entry_threshold if new == "PRESENT" else policy.exit_threshold
    return PerceptionTransitionEvent(
        frame_index=frame,
        time_value=time_value,
        decision_id=identifier,
        family=family,
        previous_state=previous,
        new_state=new,
        reason=reason,
        evidence_score=float(score),
        threshold_name=(
            f"{policy.threshold_name}_{'ENTRY' if new == 'PRESENT' else 'EXIT'}"
        ),
        threshold_value=float(threshold),
        persistence_count=int(count),
        provider=policy.provider,
        provider_version=policy.provider_version,
    )


def _normalized_binary_state(value: str) -> str:
    normalized = str(value).strip().upper()
    if normalized not in {"PRESENT", "ABSENT"}:
        raise ValueError("temporal initial state must be PRESENT or ABSENT")
    return normalized


def _family_from_identifier(identifier: str) -> str:
    return str(identifier).split(":", 1)[0].upper()


def _missing_evidence_score(policy: HysteresisPolicy) -> float:
    gap = max(1.0, abs(policy.entry_threshold - policy.exit_threshold))
    return (
        policy.exit_threshold - gap
        if policy.direction.upper() == "HIGHER_IS_ACTIVE"
        else policy.exit_threshold + gap
    )


__all__ = [
    "DEFAULT_HYSTERESIS_POLICIES",
    "HysteresisCalibration",
    "HysteresisCalibrationTrajectory",
    "HysteresisPolicy",
    "ORACLE_PERCEPTION_TRACKER_PROVIDER",
    "ORACLE_PERCEPTION_TRACKER_PROVIDER_VERSION",
    "PerceptionTracker",
    "TemporalDecisionEvidence",
    "calibrate_hysteresis_policy",
    "contact_temporal_evidence",
]
