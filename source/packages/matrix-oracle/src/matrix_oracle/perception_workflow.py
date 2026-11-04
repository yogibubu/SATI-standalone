"""ORACLE state machine for the explicit exploration/exploitation handoff."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

from matrix_chem import FrozenPerceptionHandoff, PerceptionNoiseAudit

from .perception_robustness import OraclePerceptionState
from .selective_topology import SelectiveTopologyTracker


ORACLE_PERCEPTION_WORKFLOW_SCHEMA = "matrix.oracle.perception_workflow.v1"
ORACLE_PERCEPTION_HANDOFF_SCHEMA = "matrix.oracle.perception_handoff.v1"


@dataclass(frozen=True)
class PerceptionBasinPolicy:
    handoff_window: int = 5
    persistent_change_window: int = 3
    exploration_group: str = "C1"
    topology_displacement_threshold_angstrom: float = 0.05
    topology_neighbor_hops: int = 2

    def __post_init__(self) -> None:
        if self.handoff_window < 2 or self.persistent_change_window < 2:
            raise ValueError("handoff/change persistence windows must be at least two frames")
        if not self.exploration_group.strip():
            raise ValueError("exploration requires C1 or an explicitly conserved group")
        if self.topology_displacement_threshold_angstrom <= 0 or self.topology_neighbor_hops < 0:
            raise ValueError("invalid selective topology-update policy")


@dataclass(frozen=True)
class PerceptionWorkflowEvent:
    frame_index: int
    previous_status: str
    new_status: str
    reason: str
    reference_state_hash: str
    observed_state_hash: str


@dataclass(frozen=True)
class PerceptionWorkflowSnapshot:
    status: str
    frame_count: int
    exploration_group: str
    instantaneous_proposed_group: str
    stable_basin_frames: int
    persistent_change_frames: int
    candidate_state_hash: str
    frozen_handoff: FrozenPerceptionHandoff | None
    events: tuple[PerceptionWorkflowEvent, ...]
    topology_update_atoms: tuple[int, ...] = ()
    schema: str = ORACLE_PERCEPTION_WORKFLOW_SCHEMA


class PerceptionWorkflow:
    """Explicit state machine; it never mutates a frozen exploitation state."""

    def __init__(self, policy: PerceptionBasinPolicy = PerceptionBasinPolicy()) -> None:
        self.policy = policy
        self._status = "EXPLORATION"
        self._frame_count = 0
        self._instantaneous_group = "C1"
        self._candidate: OraclePerceptionState | None = None
        self._candidate_identity: tuple | None = None
        self._stable_frames = 0
        self._change_frames = 0
        self._handoff: FrozenPerceptionHandoff | None = None
        self._events: list[PerceptionWorkflowEvent] = []
        self._last_audit: PerceptionNoiseAudit | None = None
        self._topology_tracker = SelectiveTopologyTracker(
            policy.topology_displacement_threshold_angstrom,
            policy.topology_neighbor_hops,
        )
        self._topology_update_atoms: tuple[int, ...] = ()

    def topology_update_domain(self, coordinates_angstrom, bonds) -> tuple[int, ...]:
        """Atoms ORACLE must reperceive before the next workflow observation."""

        return self._topology_tracker.affected_atoms(coordinates_angstrom, bonds)

    def observe(
        self,
        state: OraclePerceptionState,
        *,
        audit: PerceptionNoiseAudit | None = None,
        coordinates_angstrom=None,
        bonds=(),
    ) -> PerceptionWorkflowSnapshot:
        if coordinates_angstrom is not None:
            self._topology_update_atoms = self.topology_update_domain(
                coordinates_angstrom, bonds
            )
        if self._status in {"EXPLORATION", "PROPOSED", "REQUIRES_DECISION"}:
            self._observe_exploration(state, audit)
        elif self._status == "FROZEN_EXPLOITATION":
            self._observe_exploitation(state)
        elif self._status == "STOP_ON_TOPOLOGY_CHANGE":
            # Explicit return_to_exploration is required; no silent recovery.
            self._instantaneous_group = state.proposed_group
        else:
            raise RuntimeError(f"unsupported perception workflow status: {self._status}")
        self._frame_count += 1
        return self.snapshot()

    def accept_handoff(
        self,
        *,
        symmetry_decision: str,
        contract_hash: str,
    ) -> FrozenPerceptionHandoff:
        """Freeze a proposed basin exactly once after an explicit decision."""

        if self._status not in {"PROPOSED", "REQUIRES_DECISION"} or self._candidate is None:
            raise ValueError("no stable exploitation basin is awaiting acceptance")
        decision = str(symmetry_decision).strip().upper()
        if decision not in {"PROJECT", "RETAIN"}:
            raise ValueError("handoff symmetry decision must be PROJECT or RETAIN")
        if self._last_audit is None or self._last_audit.status not in {
            "ROBUST",
            "REQUIRES_DECISION",
        }:
            raise ValueError("handoff requires a robust audit of the accepted state")
        accepted_group = (
            self._candidate.proposed_group if decision == "PROJECT" else self._candidate.strict_group
        )
        audit_hash = _audit_hash(self._last_audit)
        handoff = FrozenPerceptionHandoff(
            state_hash=self._candidate.state_hash,
            topology_hash=self._candidate.topology_hash,
            accepted_group=accepted_group,
            symmetry_decision=decision,
            audit_hash=audit_hash,
            contract_hash=str(contract_hash),
            stable_frame_count=self._stable_frames,
            provenance="ORACLE_EXPLICIT_EXPLORATION_TO_EXPLOITATION_HANDOFF@1",
        )
        previous = self._status
        self._status = "FROZEN_EXPLOITATION"
        self._handoff = handoff
        self._change_frames = 0
        self._events.append(
            PerceptionWorkflowEvent(
                frame_index=self._frame_count,
                previous_status=previous,
                new_status=self._status,
                reason="EXPLICIT_HANDOFF_ACCEPTED_AND_FROZEN",
                reference_state_hash=handoff.state_hash,
                observed_state_hash=handoff.state_hash,
            )
        )
        return handoff

    def return_to_exploration(self) -> PerceptionWorkflowSnapshot:
        if self._status != "STOP_ON_TOPOLOGY_CHANGE":
            raise ValueError("return to exploration is allowed only after a persistent stop")
        reference = self._handoff.state_hash if self._handoff is not None else "NONE"
        self._events.append(
            PerceptionWorkflowEvent(
                frame_index=self._frame_count,
                previous_status=self._status,
                new_status="EXPLORATION",
                reason="EXPLICIT_CONTROL_RETURN_TO_EXPLORATION",
                reference_state_hash=reference,
                observed_state_hash=(self._candidate.state_hash if self._candidate else "NONE"),
            )
        )
        self._status = "EXPLORATION"
        self._candidate = None
        self._candidate_identity = None
        self._stable_frames = 0
        self._change_frames = 0
        self._handoff = None
        self._last_audit = None
        return self.snapshot()

    def snapshot(self) -> PerceptionWorkflowSnapshot:
        return PerceptionWorkflowSnapshot(
            status=self._status,
            frame_count=self._frame_count,
            exploration_group=self.policy.exploration_group,
            instantaneous_proposed_group=self._instantaneous_group,
            stable_basin_frames=self._stable_frames,
            persistent_change_frames=self._change_frames,
            candidate_state_hash=(self._candidate.state_hash if self._candidate else ""),
            frozen_handoff=self._handoff,
            events=tuple(self._events),
            topology_update_atoms=self._topology_update_atoms,
        )

    def _observe_exploration(
        self,
        state: OraclePerceptionState,
        audit: PerceptionNoiseAudit | None,
    ) -> None:
        self._instantaneous_group = state.proposed_group
        identity = _basin_identity(state)
        eligible_audit = audit is not None and audit.reference_state_hash == state.state_hash
        audit_stable = eligible_audit and audit.status in {"ROBUST", "REQUIRES_DECISION"}
        if identity == self._candidate_identity and audit_stable:
            self._stable_frames += 1
        else:
            self._candidate = state
            self._candidate_identity = identity
            self._stable_frames = 1 if audit_stable else 0
        if eligible_audit:
            self._last_audit = audit
        if self._stable_frames >= self.policy.handoff_window:
            next_status = (
                "REQUIRES_DECISION"
                if self._last_audit is not None
                and self._last_audit.symmetry_decision == "REQUIRES_DECISION"
                else "PROPOSED"
            )
            if self._status != next_status:
                self._events.append(
                    PerceptionWorkflowEvent(
                        frame_index=self._frame_count,
                        previous_status=self._status,
                        new_status=next_status,
                        reason="BASIN_STATE_PERSISTED_FOR_HANDOFF_WINDOW",
                        reference_state_hash=state.state_hash,
                        observed_state_hash=state.state_hash,
                    )
                )
            self._status = next_status
        else:
            self._status = "EXPLORATION"

    def _observe_exploitation(self, state: OraclePerceptionState) -> None:
        if self._handoff is None or self._candidate is None:
            raise RuntimeError("exploitation has no frozen ORACLE handoff")
        self._instantaneous_group = state.proposed_group
        changed = _critical_structural_identity(state) != _critical_structural_identity(
            self._candidate
        )
        self._change_frames = self._change_frames + 1 if changed else 0
        if self._change_frames < self.policy.persistent_change_window:
            return
        self._events.append(
            PerceptionWorkflowEvent(
                frame_index=self._frame_count,
                previous_status=self._status,
                new_status="STOP_ON_TOPOLOGY_CHANGE",
                reason="PERSISTENT_BOND_COORDINATION_FRAGMENT_OR_TOPOLOGY_CHANGE",
                reference_state_hash=self._handoff.state_hash,
                observed_state_hash=state.state_hash,
            )
        )
        self._status = "STOP_ON_TOPOLOGY_CHANGE"


def _basin_identity(state: OraclePerceptionState) -> tuple:
    return (
        _critical_structural_identity(state),
        state.structural_site_signatures,
        state.contact_signatures,
        state.local_signatures,
        state.strict_group,
        state.proposed_group,
    )


def _critical_structural_identity(state: OraclePerceptionState) -> tuple:
    return (
        state.topology_hash,
        state.fragment_signatures,
        state.ring_signatures,
        state.multicenter_signatures,
        state.atom_class_signatures,
        state.primary_cycle_rank,
    )


def _audit_hash(audit: PerceptionNoiseAudit) -> str:
    payload = (
        audit.schema,
        audit.status,
        audit.reference_state_hash,
        audit.sampled_state_hashes,
        audit.strict_group,
        audit.proposed_group,
        audit.symmetry_decision,
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


__all__ = [
    "FrozenPerceptionHandoff",
    "ORACLE_PERCEPTION_HANDOFF_SCHEMA",
    "ORACLE_PERCEPTION_WORKFLOW_SCHEMA",
    "PerceptionBasinPolicy",
    "PerceptionWorkflow",
    "PerceptionWorkflowEvent",
    "PerceptionWorkflowSnapshot",
]
