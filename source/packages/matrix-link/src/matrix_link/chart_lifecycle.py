"""Accepted-state lifecycle for native LINK optimization charts.

ORACLE owns every chemical decision and SMITH owns every coordinate build.
This module only validates and applies their typed results at accepted LINK
states.  It deliberately contains no atom-, molecule-, contact- or
stationary-point classification rules.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np

from matrix_chem.coordinate_atlas_contract import (
    ATLAS_TASK_MINIMUM as CHART_TASK_MINIMUM,
    ATLAS_TASK_TRANSITION_STATE as CHART_TASK_TRANSITION_STATE,
)
from matrix_numerics import adaptive_singular_spectrum

from .coordinate_domain import near_linear_ordinary_angle_ids


LINK_CHART_LIFECYCLE_SCHEMA = "matrix.link.chart_lifecycle.v1"
CHART_ACTION_KEEP = "KEEP"
CHART_ACTION_DEFER = "DEFER"
CHART_ACTION_REBUILD = "REBUILD"
CHART_ACTION_INVALID = "INVALID"


class ChartLifecycleError(RuntimeError):
    """A typed ORACLE/SMITH chart transition failed a LINK invariant."""


class ChartCandidateUnavailable(ChartLifecycleError):
    """A proposed chart cannot satisfy the frozen rank/conditioning gates."""


@dataclass(frozen=True)
class ChartIdentity:
    task_regime: str
    atom_order_sha256: str
    oracle_contract_sha256: str
    oracle_topology_sha256: str
    smith_definition_sha256: str
    target_rank: int

    def __post_init__(self) -> None:
        regime = str(self.task_regime).strip().upper()
        if regime not in {CHART_TASK_MINIMUM, CHART_TASK_TRANSITION_STATE}:
            raise ValueError("chart task regime must be MINIMUM or TRANSITION_STATE")
        object.__setattr__(self, "task_regime", regime)
        for name in (
            "atom_order_sha256",
            "oracle_contract_sha256",
            "oracle_topology_sha256",
            "smith_definition_sha256",
        ):
            value = str(getattr(self, name)).strip().lower()
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
            object.__setattr__(self, name, value)
        if int(self.target_rank) < 0:
            raise ValueError("chart target rank must be non-negative")
        object.__setattr__(self, "target_rank", int(self.target_rank))


@dataclass(frozen=True)
class OracleChartDecision:
    """Typed ORACLE disposition consumed mechanically by LINK."""

    action: str
    task_regime: str
    reason: str
    oracle_contract_sha256: str
    oracle_topology_sha256: str
    persistence_certificate: str
    topology_update_atoms: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        action = str(self.action).strip().upper()
        if action not in {
            CHART_ACTION_KEEP,
            CHART_ACTION_DEFER,
            CHART_ACTION_REBUILD,
            CHART_ACTION_INVALID,
        }:
            raise ValueError("unsupported ORACLE chart lifecycle action")
        regime = str(self.task_regime).strip().upper()
        if regime not in {CHART_TASK_MINIMUM, CHART_TASK_TRANSITION_STATE}:
            raise ValueError("ORACLE chart decision has an unsupported task regime")
        if not str(self.reason).strip():
            raise ValueError("ORACLE chart decision requires an auditable reason")
        if action == CHART_ACTION_REBUILD and not str(self.persistence_certificate).strip():
            raise ValueError("ORACLE rebuild decisions require a persistence certificate")
        for name in ("oracle_contract_sha256", "oracle_topology_sha256"):
            value = str(getattr(self, name)).strip().lower()
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "task_regime", regime)
        update_atoms = tuple(sorted({int(atom) for atom in self.topology_update_atoms}))
        if any(atom < 0 for atom in update_atoms):
            raise ValueError("topology update atoms must be zero-based non-negative indices")
        object.__setattr__(self, "topology_update_atoms", update_atoms)


@dataclass(frozen=True)
class ChartCandidate:
    """One SMITH-built chart at the accepted Cartesian geometry."""

    identity: ChartIdentity
    coordinate_model: object
    reference_coordinates_angstrom: np.ndarray
    b_matrix: np.ndarray
    source_xyzin_path: Path | None = None
    sparse_b_matrix: object | None = None

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.reference_coordinates_angstrom, dtype=float)
        b_matrix = np.asarray(self.b_matrix, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1:] != (3,):
            raise ValueError("chart reference coordinates must have shape natoms x 3")
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("chart reference coordinates contain non-finite values")
        if b_matrix.shape != (self.identity.target_rank, coordinates.size):
            raise ValueError("chart Wilson matrix shape contradicts its target rank")
        if not np.all(np.isfinite(b_matrix)):
            raise ValueError("chart Wilson matrix contains non-finite values")
        object.__setattr__(self, "reference_coordinates_angstrom", coordinates.copy())
        object.__setattr__(self, "b_matrix", b_matrix.copy())
        if self.sparse_b_matrix is not None:
            from matrix_smith import SparseBMatrix

            if not isinstance(self.sparse_b_matrix, SparseBMatrix):
                raise TypeError("chart sparse Wilson matrix must be a SparseBMatrix")
            if self.sparse_b_matrix.row_count != self.identity.target_rank or (
                self.sparse_b_matrix.column_count != coordinates.size
            ):
                raise ValueError("chart sparse Wilson matrix has the wrong shape")
        if self.source_xyzin_path is not None:
            object.__setattr__(self, "source_xyzin_path", Path(self.source_xyzin_path))


@dataclass(frozen=True)
class ChartSnapshot:
    epoch: int
    candidate: ChartCandidate

    def __post_init__(self) -> None:
        if int(self.epoch) < 0:
            raise ValueError("chart epoch must be non-negative")
        object.__setattr__(self, "epoch", int(self.epoch))


@dataclass(frozen=True)
class ChartLifecycleResult:
    status: str
    reason: str
    decision: OracleChartDecision
    previous: ChartSnapshot
    current: ChartSnapshot
    previous_chart_valid_for_hessian_transport: bool = True

    @property
    def changed(self) -> bool:
        return self.current.epoch != self.previous.epoch

    @property
    def coordinate_changed(self) -> bool:
        if self.status == "SMITH_REFRESH":
            return True
        return self.current.candidate.identity.smith_definition_sha256 != (
            self.previous.candidate.identity.smith_definition_sha256
        )

    @property
    def requires_commit(self) -> bool:
        return self.status in {"REBUILT", "REVALIDATED", "REFRAMED", "SMITH_REFRESH"}


class OracleChartAssessor(Protocol):
    def assess_accepted_geometry(
        self,
        snapshot: ChartSnapshot,
        atoms: tuple[str, ...],
        coordinates_angstrom: np.ndarray,
    ) -> OracleChartDecision: ...

    def commit_rebuild(self, decision: OracleChartDecision) -> None: ...


class SmithChartRebuilder(Protocol):
    def rebuild_chart(
        self,
        decision: OracleChartDecision,
        snapshot: ChartSnapshot,
        atoms: tuple[str, ...],
        coordinates_angstrom: np.ndarray,
    ) -> ChartCandidate: ...


class ChartLifecycleController:
    """Apply typed ORACLE/SMITH results at accepted geometries only."""

    def __init__(
        self,
        snapshot: ChartSnapshot,
        *,
        oracle: OracleChartAssessor,
        smith: SmithChartRebuilder,
        rank_absolute_tolerance: float = 1.0e-8,
        maximum_condition_number: float = 1.0e8,
        geometry_tolerance_angstrom: float = 1.0e-12,
        smith_refresh_interval: int = 0,
        soop_validity_evaluator: Callable[[ChartSnapshot, np.ndarray], object] | None = None,
        oracle_condition_threshold: float | None = None,
        topology_irregularity_evaluator: Callable[[ChartSnapshot, np.ndarray], bool] | None = None,
    ) -> None:
        if rank_absolute_tolerance <= 0.0:
            raise ValueError("chart rank tolerance must be positive")
        if maximum_condition_number <= 1.0:
            raise ValueError("chart maximum condition number must exceed one")
        if geometry_tolerance_angstrom < 0.0:
            raise ValueError("chart geometry tolerance must be non-negative")
        if int(smith_refresh_interval) < 0:
            raise ValueError("SMITH refresh interval must be non-negative")
        if soop_validity_evaluator is not None and not callable(soop_validity_evaluator):
            raise TypeError("SOOP validity evaluator must be callable")
        if oracle_condition_threshold is not None and oracle_condition_threshold <= 1.0:
            raise ValueError("ORACLE condition threshold must exceed one")
        self.snapshot = snapshot
        self.oracle = oracle
        self.smith = smith
        self.rank_absolute_tolerance = float(rank_absolute_tolerance)
        self.maximum_condition_number = float(maximum_condition_number)
        self.geometry_tolerance_angstrom = float(geometry_tolerance_angstrom)
        self.smith_refresh_interval = int(smith_refresh_interval)
        self.soop_validity_evaluator = soop_validity_evaluator
        self.oracle_condition_threshold = (
            None if oracle_condition_threshold is None else float(oracle_condition_threshold)
        )
        self.topology_irregularity_evaluator = topology_irregularity_evaluator

    def validate_initial_chart(
        self,
        atoms: Sequence[str],
        coordinates_angstrom: np.ndarray,
        coordinate_model: object,
        *,
        task_regime: str,
    ) -> None:
        candidate = self.snapshot.candidate
        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        self._validate_accepted_frame(self.snapshot, tuple(atoms), coordinates)
        if self.snapshot.epoch != 0:
            raise ChartLifecycleError("dynamic chart resume requires a persisted epoch manifest")
        candidate_definition = getattr(candidate.coordinate_model, "sonic_definition", None)
        optimizer_definition = getattr(coordinate_model, "sonic_definition", None)
        if candidate.coordinate_model is not coordinate_model:
            from matrix_smith import sonic_definition_identity_sha256

            if candidate_definition is None or optimizer_definition is None or (
                sonic_definition_identity_sha256(candidate_definition)
                != sonic_definition_identity_sha256(optimizer_definition)
            ):
                raise ChartLifecycleError(
                    "initial lifecycle chart and optimizer model have different SONIC definitions"
                )
            complete_labels = frozenset(
                gic.name or gic.identifier for gic in candidate_definition.gics
            )
            if not set(getattr(coordinate_model, "labels", ())).issubset(complete_labels):
                raise ChartLifecycleError(
                    "initial optimizer rows are not a task subspace of the lifecycle chart"
                )
        if candidate.identity.target_rank != len(
            getattr(candidate.coordinate_model, "labels", ())
        ):
            raise ChartLifecycleError("initial lifecycle identity contradicts complete chart rank")
        if candidate.identity.task_regime != str(task_regime).strip().upper():
            raise ChartLifecycleError("initial lifecycle regime contradicts optimizer task")
        if not np.allclose(
            candidate.reference_coordinates_angstrom,
            coordinates,
            rtol=0.0,
            atol=self.geometry_tolerance_angstrom,
        ):
            raise ChartLifecycleError("initial lifecycle geometry is not the optimizer reference")

    def validate_proposed_geometry(
        self,
        atoms: Sequence[str],
        coordinates_angstrom: np.ndarray,
    ) -> tuple[bool, str]:
        """Check a prospective TS geometry before LINK accepts its step.

        This is deliberately a fixed-chart numerical check only.  It does not
        ask ORACLE for a lifecycle decision and therefore has no chemical
        policy or side effects.  A prospective point that leaves the current
        chart domain is rejected by LINK before it becomes an accepted state;
        the normal trust-region recovery can then reduce the Cartesian step.
        """

        previous = self.snapshot
        definition = getattr(previous.candidate.coordinate_model, "sonic_definition", None)
        if definition is not None and definition.gics and all(
            gic.family == "SYCART" for gic in definition.gics
        ):
            return True, "WHOLE_MOLECULE_SYCART_FIXED"
        validity = self._soop_geometric_validity(previous, coordinates_angstrom)
        if validity is not None and str(validity.status).upper() == "INVALID":
            return (
                False,
                "SOOP_GEOMETRIC_VALIDITY_INVALID "
                f"MAX_NORMAL_ROTATION_DEGREE={validity.maximum_normal_rotation_degree:.6g}",
            )
        identity = previous.candidate.identity
        decision = OracleChartDecision(
            action=CHART_ACTION_KEEP,
            task_regime=identity.task_regime,
            reason="LINK_PRE_ACCEPT_FIXED_CHART_VALIDATION",
            oracle_contract_sha256=identity.oracle_contract_sha256,
            oracle_topology_sha256=identity.oracle_topology_sha256,
            persistence_certificate="",
        )
        _candidate, reason, valid = self._revalidate_fixed_ts_chart(
            previous,
            decision,
            np.asarray(coordinates_angstrom, dtype=float),
        )
        return bool(valid), reason

    def evaluate_accepted_geometry(
        self,
        atoms: Sequence[str],
        coordinates_angstrom: np.ndarray,
        *,
        accepted_step_index: int | None = None,
        search_irregularity: bool = False,
        fallback_irregularity: bool = False,
    ) -> ChartLifecycleResult:
        atom_order = tuple(str(atom) for atom in atoms)
        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        previous = self.snapshot
        self._validate_accepted_frame(previous, atom_order, coordinates)
        definition = getattr(previous.candidate.coordinate_model, "sonic_definition", None)
        if definition is not None and definition.gics and all(
            gic.family == "SYCART" for gic in definition.gics
        ):
            identity = previous.candidate.identity
            decision = OracleChartDecision(
                action=CHART_ACTION_KEEP,
                task_regime=identity.task_regime,
                reason="WHOLE_MOLECULE_SYCART_FIXED",
                oracle_contract_sha256=identity.oracle_contract_sha256,
                oracle_topology_sha256=identity.oracle_topology_sha256,
                persistence_certificate="",
            )
            return ChartLifecycleResult(
                status=CHART_ACTION_KEEP,
                reason=decision.reason,
                decision=decision,
                previous=previous,
                current=previous,
            )
        has_soop = definition is not None and any(
            primitive.function == "SOOP" for primitive in definition.primitives
        )
        topology_irregularity = (
            self.topology_irregularity_evaluator(previous, coordinates)
            if self.topology_irregularity_evaluator is not None
            else False
        )
        condition = (
            self._fixed_coordinate_condition(previous, coordinates)
            if self.oracle_condition_threshold is not None else None
        )
        if (
            condition is not None
            and not search_irregularity
            and not topology_irregularity
            and not fallback_irregularity
        ):
            identity = previous.candidate.identity
            provisional = OracleChartDecision(
                action=CHART_ACTION_KEEP,
                task_regime=identity.task_regime,
                reason="WELL_CONDITIONED_FIXED_COORDINATES",
                oracle_contract_sha256=identity.oracle_contract_sha256,
                oracle_topology_sha256=identity.oracle_topology_sha256,
                persistence_certificate="",
            )
            soop_candidate = None
            if has_soop and getattr(self.smith, "reference_definition", None) is not None:
                try:
                    soop_candidate = self.smith.refresh_frozen_soop_chart(
                        previous, coordinates.copy()
                    )
                except ChartCandidateUnavailable:
                    soop_candidate = None
            assessment_snapshot = (
                previous if soop_candidate is None else ChartSnapshot(previous.epoch, soop_candidate)
            )
            condition = self._fixed_coordinate_condition(assessment_snapshot, coordinates)
            if condition <= self.oracle_condition_threshold:
                if soop_candidate is not None:
                    self._validate_rebuild(
                        previous, provisional, soop_candidate, atom_order, coordinates
                    )
                    return ChartLifecycleResult(
                        status="REFRAMED",
                        reason=f"FROZEN_TOPOLOGY_SOOP_REFRESH CONDITION={condition:.12g}",
                        decision=provisional,
                        previous=previous,
                        current=ChartSnapshot(previous.epoch + 1, soop_candidate),
                    )
                if not has_soop:
                    return ChartLifecycleResult(
                        status=CHART_ACTION_KEEP,
                        reason=f"WELL_CONDITIONED_FIXED_COORDINATES CONDITION={condition:.12g}",
                        decision=provisional,
                        previous=previous,
                        current=previous,
                    )
        decision = self.oracle.assess_accepted_geometry(
            previous,
            atom_order,
            coordinates.copy(),
        )
        if decision.task_regime != previous.candidate.identity.task_regime:
            raise ChartLifecycleError("ORACLE changed task regime within one optimization")
        if decision.action == CHART_ACTION_INVALID:
            raise ChartLifecycleError(f"ORACLE rejected accepted geometry: {decision.reason}")
        if (
            self.oracle_condition_threshold is not None
            and (condition is not None and condition > self.oracle_condition_threshold
                 or topology_irregularity or fallback_irregularity)
            and decision.action in {CHART_ACTION_KEEP, CHART_ACTION_DEFER}
        ):
            escalation = getattr(self.oracle, "require_rebuild_for_invalid_chart", None)
            if not callable(escalation):
                raise ChartLifecycleError("SONIC fallback requires an ORACLE rebuild authorization")
            decision = escalation(
                decision,
                "FULL_SYCART_SONIC_FALLBACK "
                f"CONDITION={condition if condition is not None else 'NOT_EVALUATED'} "
                f"TOPOLOGY_IRREGULARITY={topology_irregularity} "
                f"SEARCH_INSTABILITY={fallback_irregularity}",
            )

        # A changed chemical topology is an ORACLE-owned rebuild.  Resolve
        # this before entering the KEEP/DEFER branch so that the canonical
        # rebuild path below validates and commits the new primitive chart.
        if (
            decision.action == CHART_ACTION_KEEP
            and decision.oracle_topology_sha256
            != previous.candidate.identity.oracle_topology_sha256
        ):
            escalation = getattr(
                self.oracle,
                "require_rebuild_for_invalid_chart",
                None,
            )
            if not callable(escalation):
                raise ChartLifecycleError(
                    "ORACLE KEEP decision changed frozen topology identity"
                )
            decision = escalation(
                decision,
                "ORACLE KEEP reported a changed topology identity",
            )

        if decision.action in {CHART_ACTION_KEEP, CHART_ACTION_DEFER}:
            soop_validity = self._soop_geometric_validity(previous, coordinates)
            (
                _fixed_candidate,
                fixed_reason,
                previous_chart_valid_for_hessian_transport,
            ) = self._revalidate_fixed_ts_chart(previous, decision, coordinates)
            if not previous_chart_valid_for_hessian_transport:
                escalation = getattr(
                    self.oracle,
                    "require_rebuild_for_invalid_chart",
                    None,
                )
                if not callable(escalation):
                    raise ChartLifecycleError(
                        "invalid fixed chart cannot be deferred without an ORACLE "
                        "rebuild authorization"
                    )
                decision = escalation(decision, fixed_reason)
            else:
                if decision.action == CHART_ACTION_KEEP:
                    if decision.oracle_contract_sha256 != previous.candidate.identity.oracle_contract_sha256:
                        # The contract hash includes geometry-dependent numerical
                        # descriptors.  With the chemical topology unchanged,
                        # this is a legitimate accepted-state update owned by
                        # SMITH; retain the lifecycle contract identity while
                        # rebuilding the non-redundant chart.
                        decision = replace(
                            decision,
                            oracle_contract_sha256=(
                                previous.candidate.identity.oracle_contract_sha256
                            ),
                        )
                definition = getattr(
                    previous.candidate.coordinate_model, "sonic_definition", None
                )
                soop_validity_status = (
                    "VALID"
                    if soop_validity is None
                    else str(soop_validity.status).strip().upper()
                )
                soop_geometry_refresh = soop_validity_status in {"WARN", "INVALID"}
                if decision.action == CHART_ACTION_KEEP and definition is not None and not any(
                    gic.family == "SYCART" for gic in definition.gics
                ) and any(
                    primitive.function == "SOOP" for primitive in definition.primitives
                ) and (
                    soop_geometry_refresh
                    or (
                        int(getattr(self, "smith_refresh_interval", 0)) > 0
                        and accepted_step_index is not None
                        and int(accepted_step_index) > 0
                        and int(accepted_step_index)
                        % int(getattr(self, "smith_refresh_interval", 0))
                        == 0
                    )
                ):
                    # Chemical ownership is unchanged. Refresh only the local
                    # accepted-state SOOP frame, then let LINK transport its
                    # optimizer tensors through the new tangent map.
                    try:
                        candidate = self.smith.rebuild_chart(
                            decision, previous, atom_order, coordinates.copy()
                        )
                    except ChartCandidateUnavailable as exc:
                        if getattr(self.smith, "reference_definition", None) is not None:
                            raise ChartLifecycleError(
                                "fixed-family SOOP cannot be refreshed; "
                                "the reactive region requires SYCART"
                            ) from exc
                        if soop_validity_status == "INVALID":
                            raise ChartLifecycleError(
                                "invalid SOOP geometric domain cannot retain the frozen chart"
                            ) from exc
                        return ChartLifecycleResult(
                            status=CHART_ACTION_KEEP,
                            reason=f"{decision.reason}; SOOP_ACCEPTED_REFERENCE_REFRESH_DEFERRED: {exc}",
                            decision=decision,
                            previous=previous,
                            current=previous,
                            previous_chart_valid_for_hessian_transport=True,
                        )
                    self._validate_rebuild(
                        previous, decision, candidate, atom_order, coordinates
                    )
                    return ChartLifecycleResult(
                        status="REFRAMED",
                        reason=(
                            f"{decision.reason}; SOOP_ACCEPTED_REFERENCE_REFRESH "
                            f"GEOMETRIC_STATUS={soop_validity_status}"
                        ),
                        decision=decision,
                        previous=previous,
                        current=ChartSnapshot(
                            epoch=previous.epoch + 1, candidate=candidate
                        ),
                        previous_chart_valid_for_hessian_transport=(
                            previous_chart_valid_for_hessian_transport
                            and soop_validity_status != "INVALID"
                        ),
                    )
                # A stable ORACLE topology can still acquire a different
                # numerically useful non-redundant SMITH chart as the
                # geometry moves.  Periodic refresh is deliberately owned by
                # SMITH: the ORACLE contract/topology hashes remain frozen.
                refresh_interval = int(
                    getattr(self, "smith_refresh_interval", 0)
                )
                if (
                    decision.action == CHART_ACTION_KEEP
                    and
                    refresh_interval > 0
                    and not (
                        definition is not None
                        and any(primitive.function == "SYCART" for primitive in definition.primitives)
                    )
                    and accepted_step_index is not None
                    and int(accepted_step_index) > 0
                    and int(accepted_step_index) % refresh_interval == 0
                ):
                    try:
                        refreshed = self.smith.rebuild_chart(
                            OracleChartDecision(
                                action=CHART_ACTION_KEEP,
                                task_regime=decision.task_regime,
                                reason=(
                                    "SMITH_PERIODIC_REFRESH_WITH_FROZEN_ORACLE_TOPOLOGY"
                                ),
                                oracle_contract_sha256=previous.candidate.identity.oracle_contract_sha256,
                                oracle_topology_sha256=previous.candidate.identity.oracle_topology_sha256,
                                persistence_certificate="",
                                topology_update_atoms=(),
                            ),
                            previous,
                            atom_order,
                            coordinates.copy(),
                        )
                    except ChartCandidateUnavailable as exc:
                        # A periodic SMITH proposal is opportunistic.  If its
                        # condition gate rejects the candidate, retain the
                        # validated chart and continue; an Oracle-authorized
                        # topology rebuild remains fail-closed below.
                        return ChartLifecycleResult(
                            status=CHART_ACTION_KEEP,
                            reason=(
                                f"{decision.reason}; SMITH_PERIODIC_REFRESH_DEFERRED: {exc}"
                            ),
                            decision=decision,
                            previous=previous,
                            current=previous,
                            previous_chart_valid_for_hessian_transport=True,
                        )
                    self._validate_rebuild(
                        previous,
                        decision,
                        refreshed,
                        atom_order,
                        coordinates,
                    )
                    return ChartLifecycleResult(
                        status="SMITH_REFRESH",
                        reason=(
                            f"{decision.reason}; "
                            "SMITH_PERIODIC_REFRESH_WITH_FROZEN_ORACLE_TOPOLOGY"
                        ),
                        decision=decision,
                        previous=previous,
                        current=ChartSnapshot(
                            epoch=previous.epoch + 1,
                            candidate=refreshed,
                        ),
                        previous_chart_valid_for_hessian_transport=True,
                    )
                return ChartLifecycleResult(
                    status=decision.action,
                    reason=decision.reason,
                    decision=decision,
                    previous=previous,
                    current=previous,
                )
        (
            fixed_candidate,
            fixed_reason,
            previous_chart_valid_for_hessian_transport,
        ) = self._revalidate_fixed_ts_chart(
            previous,
            decision,
            coordinates,
        )
        if fixed_candidate is not None and not (
            decision.action == CHART_ACTION_REBUILD
            and getattr(self.smith, "reference_definition", None) is not None
        ):
            return ChartLifecycleResult(
                status="REVALIDATED",
                reason=f"{decision.reason}; {fixed_reason}",
                decision=decision,
                previous=previous,
                current=ChartSnapshot(epoch=previous.epoch, candidate=fixed_candidate),
                previous_chart_valid_for_hessian_transport=True,
            )
        try:
            candidate = self.smith.rebuild_chart(
                decision,
                previous,
                atom_order,
                coordinates.copy(),
            )
            self._validate_rebuild(previous, decision, candidate, atom_order, coordinates)
        except ChartCandidateUnavailable as exc:
            if not previous_chart_valid_for_hessian_transport:
                raise ChartLifecycleError(
                    "SMITH cannot replace a numerically invalid fixed chart"
                ) from exc
            return ChartLifecycleResult(
                status="REBUILD_DEFERRED",
                reason=f"{decision.reason}; {fixed_reason}; {exc}",
                decision=decision,
                previous=previous,
                current=previous,
                previous_chart_valid_for_hessian_transport=(
                    previous_chart_valid_for_hessian_transport
                ),
            )
        coordinate_changed = candidate.identity.smith_definition_sha256 != (
            previous.candidate.identity.smith_definition_sha256
        )
        if not coordinate_changed:
            return ChartLifecycleResult(
                status="REVALIDATED",
                reason=f"{decision.reason}; {fixed_reason}",
                decision=decision,
                previous=previous,
                current=ChartSnapshot(epoch=previous.epoch, candidate=candidate),
                previous_chart_valid_for_hessian_transport=(
                    previous_chart_valid_for_hessian_transport
                ),
            )
        current = ChartSnapshot(epoch=previous.epoch + 1, candidate=candidate)
        return ChartLifecycleResult(
            status="REBUILT",
            reason=f"{decision.reason}; {fixed_reason}",
            decision=decision,
            previous=previous,
            current=current,
            previous_chart_valid_for_hessian_transport=(
                previous_chart_valid_for_hessian_transport
            ),
        )

    def _soop_geometric_validity(
        self,
        snapshot: ChartSnapshot,
        coordinates_angstrom: np.ndarray,
    ):
        definition = getattr(snapshot.candidate.coordinate_model, "sonic_definition", None)
        if definition is None or not any(
            primitive.function == "SOOP" for primitive in definition.primitives
        ):
            return None
        evaluator = self.soop_validity_evaluator
        if evaluator is None:
            return None
        report = evaluator(snapshot, np.asarray(coordinates_angstrom, dtype=float))
        status = str(getattr(report, "status", "")).strip().upper()
        if status not in {"VALID", "WARN", "INVALID"}:
            raise ChartLifecycleError("SOOP validity evaluator returned an invalid status")
        return report

    def _fixed_coordinate_condition(
        self,
        snapshot: ChartSnapshot,
        coordinates_angstrom: np.ndarray,
    ) -> float:
        definition = getattr(snapshot.candidate.coordinate_model, "sonic_definition", None)
        if definition is None or near_linear_ordinary_angle_ids(definition, coordinates_angstrom):
            return float("inf")
        from matrix_smith import build_gic_b_matrix

        try:
            rows = np.asarray(
                build_gic_b_matrix(
                    definition, coordinates_angstrom=coordinates_angstrom
                ).rows,
                dtype=float,
            )
            if rows.shape != (snapshot.candidate.identity.target_rank, coordinates_angstrom.size):
                return float("inf")
            norms = np.linalg.norm(rows, axis=1)
            if np.any(~np.isfinite(norms)) or np.any(norms <= self.rank_absolute_tolerance):
                return float("inf")
            spectrum = adaptive_singular_spectrum(
                rows / norms[:, None],
                absolute_tolerance=self.rank_absolute_tolerance,
            )
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            return float("inf")
        if spectrum.rank != snapshot.candidate.identity.target_rank:
            return float("inf")
        return float(spectrum.condition_number)

    def _revalidate_fixed_ts_chart(
        self,
        previous: ChartSnapshot,
        decision: OracleChartDecision,
        coordinates: np.ndarray,
    ) -> tuple[ChartCandidate | None, str, bool]:
        """Keep a TS chart unless its current Wilson matrix is invalid."""

        if decision.task_regime != CHART_TASK_TRANSITION_STATE:
            return None, "NON_TS_DYNAMIC_CHART_POLICY", True
        model = previous.candidate.coordinate_model
        definition = getattr(model, "sonic_definition", None)
        if str(getattr(model, "kind", "")).strip().casefold() != "sonic" or definition is None:
            return None, "TS_FIXED_CHART_UNAVAILABLE_NON_SONIC_MODEL", False
        invalid_angles = near_linear_ordinary_angle_ids(definition, coordinates)
        if invalid_angles:
            return (
                None,
                "TS_FIXED_CHART_ORDINARY_ANGLE_DOMAIN_INVALID="
                + ",".join(invalid_angles),
                False,
            )
        from matrix_smith import build_gic_b_matrix

        try:
            b_matrix = np.asarray(
                build_gic_b_matrix(
                    definition,
                    coordinates_angstrom=coordinates,
                ).rows,
                dtype=float,
            )
        except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
            return None, f"TS_FIXED_CHART_DOMAIN_INVALID={type(exc).__name__}", False
        target_rank = previous.candidate.identity.target_rank
        expected_shape = (target_rank, coordinates.size)
        if b_matrix.shape != expected_shape:
            return (
                None,
                "TS_FIXED_CHART_SHAPE_INVALID="
                f"{b_matrix.shape!r}!={expected_shape!r}",
                False,
            )
        row_norms = np.linalg.norm(b_matrix, axis=1)
        if np.any(~np.isfinite(row_norms)) or np.any(
            row_norms <= self.rank_absolute_tolerance
        ):
            return None, "TS_FIXED_CHART_ZERO_OR_NONFINITE_ROW", False
        normalized = b_matrix / row_norms[:, None]
        spectrum = adaptive_singular_spectrum(
            normalized,
            absolute_tolerance=self.rank_absolute_tolerance,
        )
        if spectrum.rank != target_rank:
            return (
                None,
                f"TS_FIXED_CHART_RANK_INVALID={spectrum.rank}/{target_rank}",
                False,
            )
        if not np.isfinite(spectrum.condition_number) or (
            spectrum.condition_number > self.maximum_condition_number
        ):
            return (
                None,
                "TS_FIXED_CHART_CONDITION_INVALID="
                f"{spectrum.condition_number:.12g}>{self.maximum_condition_number:.12g}",
                False,
            )
        identity = ChartIdentity(
            task_regime=decision.task_regime,
            atom_order_sha256=previous.candidate.identity.atom_order_sha256,
            oracle_contract_sha256=decision.oracle_contract_sha256,
            oracle_topology_sha256=decision.oracle_topology_sha256,
            smith_definition_sha256=(
                previous.candidate.identity.smith_definition_sha256
            ),
            target_rank=target_rank,
        )
        candidate = ChartCandidate(
            identity=identity,
            coordinate_model=model,
            reference_coordinates_angstrom=coordinates,
            b_matrix=b_matrix,
            source_xyzin_path=previous.candidate.source_xyzin_path,
        )
        return (
            candidate,
            "TS_FIXED_CHART_NUMERICALLY_VALID "
            f"RANK={spectrum.rank}/{target_rank} "
            f"NORMALIZED_CONDITION={spectrum.condition_number:.12g}",
            True,
        )

    def commit_transition(self, result: ChartLifecycleResult) -> None:
        """Commit ORACLE and LINK state after optimizer transport succeeds."""

        if not result.requires_commit:
            raise ChartLifecycleError("a non-rebuild chart result cannot be committed")
        if result.previous is not self.snapshot:
            raise ChartLifecycleError("chart transition result is stale")
        if result.status not in {"REFRAMED", "SMITH_REFRESH"}:
            self.oracle.commit_rebuild(result.decision)
        self.snapshot = result.current

    def _validate_accepted_frame(
        self,
        snapshot: ChartSnapshot,
        atoms: tuple[str, ...],
        coordinates: np.ndarray,
    ) -> None:
        if coordinates.shape != snapshot.candidate.reference_coordinates_angstrom.shape:
            raise ChartLifecycleError("accepted geometry changed the chart atom frame")
        if not np.all(np.isfinite(coordinates)):
            raise ChartLifecycleError("accepted geometry contains non-finite values")
        if atom_order_sha256(atoms) != snapshot.candidate.identity.atom_order_sha256:
            raise ChartLifecycleError("accepted geometry changed atom order")

    def _validate_rebuild(
        self,
        previous: ChartSnapshot,
        decision: OracleChartDecision,
        candidate: ChartCandidate,
        atoms: tuple[str, ...],
        coordinates: np.ndarray,
    ) -> None:
        identity = candidate.identity
        if identity.task_regime != decision.task_regime:
            raise ChartLifecycleError("SMITH chart changed the ORACLE task regime")
        if identity.atom_order_sha256 != atom_order_sha256(atoms):
            raise ChartLifecycleError("SMITH chart changed atom order")
        if identity.oracle_contract_sha256 != decision.oracle_contract_sha256:
            raise ChartLifecycleError("SMITH chart does not consume the assessed ORACLE contract")
        if identity.oracle_topology_sha256 != decision.oracle_topology_sha256:
            raise ChartLifecycleError("SMITH chart does not consume the assessed ORACLE topology")
        if identity.target_rank != previous.candidate.identity.target_rank:
            raise ChartLifecycleError("SMITH chart changed the vibrational target rank")
        if (
            identity.task_regime == CHART_TASK_TRANSITION_STATE
            and getattr(self.smith, "reference_definition", None) is not None
        ):
            old_definition = getattr(previous.candidate.coordinate_model, "sonic_definition", None)
            new_definition = getattr(candidate.coordinate_model, "sonic_definition", None)
            if old_definition is not None:
                if new_definition is None:
                    raise ChartLifecycleError(
                        "transition-state optimization cannot change its coordinate family"
                    )
                old_signature = tuple(
                    (gic.identifier, gic.name, gic.family, gic.irrep)
                    for gic in old_definition.gics
                )
                new_signature = tuple(
                    (gic.identifier, gic.name, gic.family, gic.irrep)
                    for gic in new_definition.gics
                )
                local_sycart_transition = (
                    decision.action == CHART_ACTION_REBUILD
                    and str(new_definition.backend) == "matrix-smith-local-sycart-transition.v1"
                    and any(gic.family == "SYCART" for gic in new_definition.gics)
                )
                if new_signature != old_signature and not local_sycart_transition:
                    raise ChartLifecycleError(
                        "transition-state optimization cannot reselect coordinates"
                    )
        if not np.allclose(
            candidate.reference_coordinates_angstrom,
            coordinates,
            rtol=0.0,
            atol=self.geometry_tolerance_angstrom,
        ):
            raise ChartLifecycleError("SMITH changed the accepted post-ORACLE geometry")
        spectrum = adaptive_singular_spectrum(
            candidate.b_matrix,
            absolute_tolerance=self.rank_absolute_tolerance,
        )
        if spectrum.rank != identity.target_rank:
            raise ChartLifecycleError("SMITH rebuilt a rank-deficient chart")
        if not np.isfinite(spectrum.condition_number) or (
            spectrum.condition_number > self.maximum_condition_number
        ):
            raise ChartLifecycleError("SMITH rebuilt an ill-conditioned chart")


def atom_order_sha256(atoms: Sequence[str]) -> str:
    payload = "\0".join(str(atom).strip() for atom in atoms).encode("utf-8")
    return sha256(payload).hexdigest()


def chart_lifecycle_result_to_json(result: ChartLifecycleResult) -> dict[str, object]:
    """Serialize one transition decision without backend or chemistry inference."""

    from matrix_core import ScientificValidationRecord

    spectrum = adaptive_singular_spectrum(
        result.current.candidate.b_matrix,
        absolute_tolerance=1.0e-8,
    )
    finite_condition = (
        float(spectrum.condition_number)
        if np.isfinite(spectrum.condition_number)
        else float(np.finfo(float).max)
    )
    validation = ScientificValidationRecord(
        owner="LINK",
        subject="CHART_LIFECYCLE_TRANSITION",
        passed=(
            spectrum.rank == result.current.candidate.identity.target_rank
            and np.isfinite(spectrum.condition_number)
        ),
        metrics={
            "current_rank": float(spectrum.rank),
            "target_rank": float(result.current.candidate.identity.target_rank),
            "condition_number": finite_condition,
            "relative_backward_error": float(spectrum.relative_backward_error),
        },
        thresholds={"absolute_rank_tolerance": 1.0e-8},
        context={
            "status": result.status,
            "previous_chart_valid_for_hessian_transport": (
                result.previous_chart_valid_for_hessian_transport
            ),
        },
    )
    return {
        "schema": LINK_CHART_LIFECYCLE_SCHEMA,
        "status": result.status,
        "reason": result.reason,
        "action": result.decision.action,
        "task_regime": result.decision.task_regime,
        "previous_epoch": result.previous.epoch,
        "current_epoch": result.current.epoch,
        "oracle_contract_sha256": result.decision.oracle_contract_sha256,
        "oracle_topology_sha256": result.decision.oracle_topology_sha256,
        "persistence_certificate": result.decision.persistence_certificate,
        "topology_update_atoms": list(result.decision.topology_update_atoms),
        "previous_smith_definition_sha256": (
            result.previous.candidate.identity.smith_definition_sha256
        ),
        "current_smith_definition_sha256": (
            result.current.candidate.identity.smith_definition_sha256
        ),
        "target_rank": result.current.candidate.identity.target_rank,
        "changed": result.changed,
        "coordinate_changed": result.coordinate_changed,
        "previous_chart_valid_for_hessian_transport": (
            result.previous_chart_valid_for_hessian_transport
        ),
        "scientific_validation": validation.to_dict(),
    }


class OraclePerceptionChartAdapter:
    """Translate ORACLE's native assessment without adding LINK policy."""

    def __init__(self, assessor: object) -> None:
        from matrix_oracle import OptimizationChartAssessor

        if not isinstance(assessor, OptimizationChartAssessor):
            raise TypeError("assessor must be an ORACLE OptimizationChartAssessor")
        self.assessor = assessor
        self._pending: dict[str, object] = {}
        self._latest_assessment: object | None = None

    def assess_accepted_geometry(
        self,
        snapshot: ChartSnapshot,
        atoms: tuple[str, ...],
        coordinates_angstrom: np.ndarray,
    ) -> OracleChartDecision:
        from matrix_chem.topology.elements import atomic_number

        numbers = tuple(int(atomic_number(atom) or 0) for atom in atoms)
        if numbers != self.assessor.atomic_numbers:
            raise ChartLifecycleError("accepted atom identities contradict the ORACLE assessor")
        assessment = self.assessor.assess_accepted_geometry(coordinates_angstrom)
        self._latest_assessment = assessment
        decision = OracleChartDecision(
            action=assessment.action,
            task_regime=assessment.task_regime,
            reason=assessment.reason,
            oracle_contract_sha256=assessment.contract_sha256,
            oracle_topology_sha256=assessment.topology_sha256,
            persistence_certificate=assessment.persistence_certificate,
            topology_update_atoms=assessment.topology_update_atoms,
        )
        self._pending.clear()
        if decision.action == CHART_ACTION_REBUILD:
            self._pending[decision.persistence_certificate] = assessment
        return decision

    def commit_rebuild(self, decision: OracleChartDecision) -> None:
        assessment = self._pending.pop(decision.persistence_certificate, None)
        if assessment is None:
            raise ChartLifecycleError("ORACLE rebuild decision has no pending native assessment")
        self.assessor.commit_rebuild(assessment)

    def require_rebuild_for_invalid_chart(
        self,
        decision: OracleChartDecision,
        chart_domain_reason: str,
    ) -> OracleChartDecision:
        """Ask ORACLE to sign the current state for immediate chart recovery."""

        assessment = self._latest_assessment
        if assessment is None or (
            assessment.task_regime != decision.task_regime
            or assessment.contract_sha256 != decision.oracle_contract_sha256
            or assessment.topology_sha256 != decision.oracle_topology_sha256
        ):
            raise ChartLifecycleError("invalid-chart escalation has no matching ORACLE assessment")
        assessment = self.assessor.require_rebuild_for_invalid_chart(
            assessment,
            chart_domain_reason=chart_domain_reason,
        )
        rebuilt = OracleChartDecision(
            action=assessment.action,
            task_regime=assessment.task_regime,
            reason=assessment.reason,
            oracle_contract_sha256=assessment.contract_sha256,
            oracle_topology_sha256=assessment.topology_sha256,
            persistence_certificate=assessment.persistence_certificate,
            topology_update_atoms=assessment.topology_update_atoms,
        )
        self._pending.clear()
        self._pending[rebuilt.persistence_certificate] = assessment
        return rebuilt


class SmithXyzinChartRebuilder:
    """Build a new chart from a fresh ORACLE artifact using canonical SMITH."""

    def __init__(
        self,
        artifact_provider: Callable[
            [OracleChartDecision, tuple[str, ...], np.ndarray], Path | str
        ],
        *,
        symmetrize: bool = False,
        reference_definition: object | None = None,
    ) -> None:
        if not callable(artifact_provider):
            raise TypeError("artifact_provider must be callable")
        self.artifact_provider = artifact_provider
        self.symmetrize = bool(symmetrize)
        self.reference_definition = reference_definition

    def refresh_frozen_soop_chart(
        self,
        snapshot: ChartSnapshot,
        coordinates_angstrom: np.ndarray,
    ) -> ChartCandidate:
        """Update SOOP planes with a frozen topology and coordinate family."""
        from matrix_chem import read_molecular_symmetry, read_oracle_sonic_contract
        from matrix_smith import (
            SOOPMaterializationError,
            build_gic_b_matrix,
            sonic_definition_identity_sha256,
        )
        from matrix_smith.soop_definition import refresh_frozen_soop_definition
        from .internal_coordinates import cartesian_from_internal_jacobian
        from .optimizer import OptimizerCoordinateModel
        from .scan import ANGSTROM_TO_BOHR

        if self.reference_definition is None:
            raise ChartCandidateUnavailable("frozen SOOP reference is unavailable")
        source = snapshot.candidate.source_xyzin_path
        if source is None:
            raise ChartCandidateUnavailable("frozen ORACLE source is unavailable")
        xyz = np.asarray(coordinates_angstrom, dtype=float)
        try:
            definition = refresh_frozen_soop_definition(
                self.reference_definition,
                read_oracle_sonic_contract(source),
                xyz,
                symmetry_operations=read_molecular_symmetry(source).operations,
            )
            b_matrix = np.asarray(
                build_gic_b_matrix(definition, coordinates_angstrom=xyz).rows,
                dtype=float,
            )
        except (SOOPMaterializationError, FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
            raise ChartCandidateUnavailable(f"frozen SOOP refresh failed: {exc}") from exc
        labels = tuple(gic.name or gic.identifier for gic in definition.gics)
        directions = cartesian_from_internal_jacobian(b_matrix, rcond=1.0e-8).T
        model = OptimizerCoordinateModel(
            kind="sonic",
            labels=labels,
            directions_angstrom=directions,
            metric_diagonal=np.maximum(
                np.sum((directions * ANGSTROM_TO_BOHR) ** 2, axis=1), 1.0e-12
            ),
            sonic_labels=labels,
            sonic_definition=definition,
        )
        identity = snapshot.candidate.identity
        return ChartCandidate(
            identity=replace(
                identity,
                smith_definition_sha256=sonic_definition_identity_sha256(definition),
            ),
            coordinate_model=model,
            reference_coordinates_angstrom=xyz,
            b_matrix=b_matrix,
            source_xyzin_path=source,
        )

    def rebuild_chart(
        self,
        decision: OracleChartDecision,
        snapshot: ChartSnapshot,
        atoms: tuple[str, ...],
        coordinates_angstrom: np.ndarray,
    ) -> ChartCandidate:
        from matrix_core.sectioned_xyz import read_sectioned_lines, section_content
        from matrix_smith import (
            GICForgeContractError,
            GICForgeRankDeficiencyError,
            SOOPMaterializationError,
            build_sparse_gic_b_matrix,
            sycart_definition_section_lines,
            sonic_definition_identity_sha256,
            update_sparse_gic_b_matrix,
            write_gicforge_build_sections,
            write_sycart_build_section,
        )

        artifact = Path(self.artifact_provider(decision, atoms, coordinates_angstrom.copy()))
        if section_content(read_sectioned_lines(artifact), "GIC"):
            raise ChartLifecycleError(
                "ORACLE chart artifact contains a stale SMITH GIC section"
            )
        try:
            old_definition = getattr(
                getattr(getattr(snapshot, "candidate", None), "coordinate_model", None),
                "sonic_definition",
                None,
            )
            persistent_sycart = bool(
                old_definition is not None
                and any(gic.family == "SYCART" for gic in old_definition.gics)
            )
            if decision.action == CHART_ACTION_REBUILD and getattr(snapshot, "candidate", None) is not None and (
                decision.task_regime == CHART_TASK_TRANSITION_STATE
                or self.reference_definition is not None or persistent_sycart
            ):
                from matrix_chem import read_oracle_sonic_contract
                from matrix_smith.local_sycart_transition import build_local_sycart_transition_definition
                from matrix_smith import build_gic_b_matrix, sonic_definition_identity_sha256
                from .optimizer import OptimizerCoordinateModel
                from .internal_coordinates import cartesian_from_internal_jacobian
                from .scan import ANGSTROM_TO_BOHR

                previous_contract = read_oracle_sonic_contract(
                    snapshot.candidate.source_xyzin_path
                )
                current_contract = read_oracle_sonic_contract(artifact)
                old_bonds = {
                    frozenset(pair) for pair in previous_contract.primary_topology.bonds
                }
                new_bonds = {
                    frozenset(pair) for pair in current_contract.primary_topology.bonds
                }
                if old_bonds != new_bonds or decision.task_regime == CHART_TASK_TRANSITION_STATE:
                    from matrix_chem import read_molecular_symmetry
                    from matrix_smith.definition import build_sycart_definition_from_xyzin

                    definition, _zone = build_local_sycart_transition_definition(
                        old_definition,
                        previous_contract,
                        current_contract,
                        coordinates_angstrom,
                        sycart_definition=(
                            build_sycart_definition_from_xyzin(artifact)
                            if old_definition.point_group.strip().upper() != "C1"
                            else None
                        ),
                        symmetry_operations=read_molecular_symmetry(artifact).operations,
                        whole_molecule=(decision.task_regime == CHART_TASK_TRANSITION_STATE),
                    )
                else:
                    # A semantic ORACLE rebuild does not authorize changing
                    # coordinates when the bond topology remains the same.
                    definition = old_definition
                b_matrix = np.asarray(
                    build_gic_b_matrix(definition, coordinates_angstrom=coordinates_angstrom).rows
                )
                directions = cartesian_from_internal_jacobian(b_matrix, rcond=1.0e-8).T
                labels = tuple(gic.name or gic.identifier for gic in definition.gics)
                full_sycart = bool(definition.gics) and all(
                    gic.family == "SYCART" for gic in definition.gics
                )
                model = OptimizerCoordinateModel(
                    kind="sycart" if full_sycart else "sonic",
                    labels=labels,
                    directions_angstrom=directions,
                    metric_diagonal=np.maximum(
                        np.sum((directions * ANGSTROM_TO_BOHR) ** 2, axis=1), 1.0e-12
                    ),
                    sonic_labels=labels,
                    sonic_definition=definition,
                )
                return ChartCandidate(
                    identity=ChartIdentity(
                        task_regime=decision.task_regime,
                        atom_order_sha256=atom_order_sha256(atoms),
                        oracle_contract_sha256=decision.oracle_contract_sha256,
                        oracle_topology_sha256=decision.oracle_topology_sha256,
                        smith_definition_sha256=sonic_definition_identity_sha256(definition),
                        target_rank=len(labels),
                    ),
                    coordinate_model=model,
                    reference_coordinates_angstrom=coordinates_angstrom,
                    b_matrix=b_matrix,
                    source_xyzin_path=artifact,
                )
            if self.reference_definition is not None and decision.action == CHART_ACTION_KEEP:
                from collections import Counter
                from matrix_core import replace_section
                from matrix_chem import read_molecular_symmetry
                from matrix_smith import gic_definition_section_lines, load_validated_oracle_candidate_pool
                from matrix_smith.soop_definition import materialize_reference_soop_definition

                template = self.reference_definition
                current_definition = write_gicforge_build_sections(
                    artifact,
                    symmetrize=self.symmetrize,
                    fragment_context=(
                        "minimum" if decision.task_regime == CHART_TASK_MINIMUM
                        else "transition_state"
                    ),
                    out_of_plane_model="soop",
                )
                definition = materialize_reference_soop_definition(
                    template,
                    current_definition,
                    load_validated_oracle_candidate_pool(artifact),
                    symmetry_operations=read_molecular_symmetry(artifact).operations,
                )
                expected = Counter(gic.family for gic in template.gics)
                actual = Counter(gic.family for gic in definition.gics)
                if actual != expected:
                    raise ChartCandidateUnavailable(
                        f"SOOP refresh changed coordinate-family dimensions: {expected} -> {actual}"
                    )
                if tuple(
                    (gic.identifier, gic.name, gic.family, gic.irrep)
                    for gic in definition.gics
                ) != tuple(
                    (gic.identifier, gic.name, gic.family, gic.irrep)
                    for gic in template.gics
                ):
                    raise ChartCandidateUnavailable(
                        "SOOP refresh changed the frozen coordinate identities"
                    )
                replace_section(artifact, "GIC", gic_definition_section_lines(definition))
            else:
                definition = write_gicforge_build_sections(
                    artifact,
                    symmetrize=self.symmetrize,
                    fragment_context=(
                        "minimum"
                        if decision.task_regime == CHART_TASK_MINIMUM
                        else "transition_state"
                    ),
                    out_of_plane_model="soop",
                )
        except GICForgeRankDeficiencyError as exc:
            raise ChartCandidateUnavailable(
                "SMITH exact-rank candidate unavailable "
                f"(target={exc.target_rank}, rank={exc.selected_rank}, "
                f"candidates={exc.candidate_count})"
            ) from exc
        except SOOPMaterializationError as exc:
            # A failed SOOP chart is never repaired with legacy U/IMPD rows.
            # The complete affected fragment is routed through the chemically
            # neutral, symmetry-adapted Cartesian chart.  This rebuilder owns a
            # single molecular fragment, so a reported failure domain covering
            # its atoms becomes the exact global SYCART recovery chart.
            affected = tuple(exc.atom_indices_zero_based)
            if set(affected) != set(range(len(atoms))):
                raise ChartCandidateUnavailable(
                    "SMITH requires a local mixed SONIC/SYCART recovery domain "
                    f"for atoms {affected}"
                ) from exc
            sycart = write_sycart_build_section(artifact)
            from .optimizer import coordinate_model_from_xyzin

            model = coordinate_model_from_xyzin(
                artifact,
                kind="sycart",
                task_regime=decision.task_regime,
            )
            b_matrix = np.asarray(model.directions_angstrom, dtype=float)
            serialized = "\n".join(sycart_definition_section_lines(sycart)).encode("utf-8")
            identity = ChartIdentity(
                task_regime=decision.task_regime,
                atom_order_sha256=atom_order_sha256(atoms),
                oracle_contract_sha256=decision.oracle_contract_sha256,
                oracle_topology_sha256=decision.oracle_topology_sha256,
                smith_definition_sha256=sha256(serialized).hexdigest(),
                target_rank=len(model.labels),
            )
            return ChartCandidate(
                identity=identity,
                coordinate_model=model,
                reference_coordinates_angstrom=coordinates_angstrom,
                b_matrix=b_matrix,
                source_xyzin_path=artifact,
            )
        except GICForgeContractError as exc:
            raise ChartCandidateUnavailable(
                f"SMITH rejected the candidate chart ({exc})"
            ) from exc
        previous_definition = getattr(
            snapshot.candidate.coordinate_model, "sonic_definition", None
        )
        previous_sparse = snapshot.candidate.sparse_b_matrix
        same_chart = (
            previous_definition is not None
            and tuple(previous_definition.primitives) == tuple(definition.primitives)
            and tuple(previous_definition.gics) == tuple(definition.gics)
        )
        if same_chart and previous_sparse is not None and decision.topology_update_atoms:
            sparse_b_matrix, _updated_rows = update_sparse_gic_b_matrix(
                definition,
                previous_sparse,
                coordinates_angstrom=coordinates_angstrom,
                affected_atoms=decision.topology_update_atoms,
            )
        else:
            sparse_b_matrix = build_sparse_gic_b_matrix(
                definition,
                coordinates_angstrom=coordinates_angstrom,
            )
        b_matrix = sparse_b_matrix.to_dense()
        from .internal_coordinates import cartesian_from_internal_jacobian
        from .optimizer import OptimizerCoordinateModel
        from .scan import ANGSTROM_TO_BOHR

        directions = cartesian_from_internal_jacobian(b_matrix, rcond=1.0e-8).T
        labels = tuple(gic.name or gic.identifier for gic in definition.gics)
        model = OptimizerCoordinateModel(
            kind="sonic",
            labels=labels,
            directions_angstrom=directions,
            metric_diagonal=np.maximum(
                np.sum((directions * ANGSTROM_TO_BOHR) ** 2, axis=1),
                1.0e-12,
            ),
            sonic_labels=labels,
            sonic_definition=definition,
        )
        identity = ChartIdentity(
            task_regime=decision.task_regime,
            atom_order_sha256=atom_order_sha256(atoms),
            oracle_contract_sha256=decision.oracle_contract_sha256,
            oracle_topology_sha256=decision.oracle_topology_sha256,
            smith_definition_sha256=sonic_definition_identity_sha256(definition),
            target_rank=len(labels),
        )
        return ChartCandidate(
            identity=identity,
            coordinate_model=model,
            reference_coordinates_angstrom=coordinates_angstrom,
            b_matrix=b_matrix,
            source_xyzin_path=artifact,
            sparse_b_matrix=sparse_b_matrix,
        )


def chart_lifecycle_controller_from_xyzin(
    source_xyzin: Path | str,
    *,
    run_dir: Path | str,
    coordinate_model: object,
    stationary_point: str,
    smith_refresh_interval: int = 0,
    soop_reference_definition_path: Path | str | None = None,
) -> ChartLifecycleController:
    """Compose the canonical ORACLE--SMITH lifecycle for one LINK SONIC run."""

    from matrix_chem import read_enriched_xyz, read_molecular_symmetry, read_oracle_sonic_contract
    from matrix_chem.topology.elements import atomic_number
    from matrix_oracle import (
        OptimizationChartAssessor,
        materialize_optimization_chart_artifact,
    )
    from matrix_smith import (
        audit_soop_geometric_validity,
        build_sparse_gic_b_matrix,
        sonic_definition_identity_sha256,
    )

    source = Path(source_xyzin).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"chart lifecycle source does not exist: {source}")
    kind = str(getattr(coordinate_model, "kind", "")).strip().casefold()
    definition = getattr(coordinate_model, "sonic_definition", None)
    if kind != "sonic" or definition is None:
        raise ChartLifecycleError("chart lifecycle requires the canonical SONIC model")
    labels = tuple(str(value) for value in getattr(coordinate_model, "labels", ()))
    full_labels = tuple(gic.name or gic.identifier for gic in definition.gics)
    if getattr(coordinate_model, "rank_reduced_labels", ()):
        raise ChartLifecycleError("SONIC lifecycle input contains numerically dropped rows")
    if labels != full_labels:
        # A symmetry-preserving minimum exposes only the totally symmetric
        # optimization block.  Lifecycle ownership nevertheless belongs to
        # the complete molecular chart, from which each accepted-state task
        # block is selected again after reframing.
        from .optimizer import coordinate_model_from_xyzin

        coordinate_model = coordinate_model_from_xyzin(
            source,
            kind="sonic",
            coordinates=full_labels,
            sonic_definition=definition,
        )
        labels = tuple(str(value) for value in coordinate_model.labels)
    if labels != full_labels or int(definition.target_rank) != len(labels):
        raise ChartLifecycleError("SONIC definition does not have its exact target rank")
    normalized_stationary_point = str(stationary_point).strip().casefold().replace("-", "_")
    if normalized_stationary_point == "minimum":
        regime = CHART_TASK_MINIMUM
    elif normalized_stationary_point == "transition_state":
        regime = CHART_TASK_TRANSITION_STATE
    else:
        raise ChartLifecycleError(
            "chart lifecycle requires an explicit minimum or transition-state task"
        )

    geometry = read_enriched_xyz(source)
    atoms = tuple(geometry.atoms)
    coordinates = np.asarray(geometry.coordinates_angstrom, dtype=float)
    numbers = tuple(int(atomic_number(atom) or 0) for atom in atoms)
    if any(number <= 0 for number in numbers):
        raise ChartLifecycleError("chart lifecycle source has an unsupported atomic label")
    assessor = OptimizationChartAssessor(
        numbers,
        coordinates,
        task_regime=regime,
    )
    reference = assessor.reference_identity
    sparse_b_matrix = build_sparse_gic_b_matrix(
        definition,
        coordinates_angstrom=coordinates,
    )
    b_matrix = sparse_b_matrix.to_dense()
    spectrum = adaptive_singular_spectrum(b_matrix, absolute_tolerance=1.0e-8)
    if spectrum.rank != len(labels):
        raise ChartLifecycleError("initial SONIC lifecycle chart is rank deficient")
    if not np.isfinite(spectrum.condition_number) or spectrum.condition_number > 1.0e8:
        raise ChartLifecycleError("initial SONIC lifecycle chart is ill-conditioned")
    initial = ChartCandidate(
        identity=ChartIdentity(
            task_regime=regime,
            atom_order_sha256=atom_order_sha256(atoms),
            oracle_contract_sha256=reference.contract_sha256,
            oracle_topology_sha256=reference.topology_sha256,
            smith_definition_sha256=sonic_definition_identity_sha256(definition),
            target_rank=len(labels),
        ),
        coordinate_model=coordinate_model,
        reference_coordinates_angstrom=coordinates,
        b_matrix=b_matrix,
        source_xyzin_path=source,
        sparse_b_matrix=sparse_b_matrix,
    )
    root = Path(run_dir).expanduser().resolve() / "chart_epochs"
    root.mkdir(parents=True, exist_ok=True)
    reference_definition = None
    if soop_reference_definition_path is None and any(
        primitive.function == "SOOP" for primitive in definition.primitives
    ) and not any(gic.family == "SYCART" for gic in definition.gics):
        from matrix_smith.oracle_contract_consumer import build_gic_definition_from_oracle_contract
        from matrix_smith.soop_definition import materialize_soop_definition

        provisional = build_gic_definition_from_oracle_contract(
            source,
            symmetrize=bool(definition.symmetrize),
            scientific_path=(
                "TRANSITION_STATE" if regime == CHART_TASK_TRANSITION_STATE else "MINIMUM"
            ),
            validate_sonic_basis=False,
            materialize_soop=False,
        )
        reconstructed = materialize_soop_definition(
            provisional,
            read_oracle_sonic_contract(source),
            validate_basis=False,
            symmetry_operations=read_molecular_symmetry(source).operations,
        )
        if tuple(reconstructed.gics) != tuple(definition.gics) or tuple(
            reconstructed.primitives
        ) != tuple(definition.primitives):
            raise ChartLifecycleError(
                "frozen SOOP reference cannot reproduce the initial Keymaker coordinates"
            )
        reference_definition = provisional
    if soop_reference_definition_path is not None:
        from collections import Counter
        from matrix_smith import load_validated_oracle_candidate_pool, read_gic_definition_from_xyzin
        from matrix_smith.soop_definition import materialize_reference_soop_definition

        reference_definition = read_gic_definition_from_xyzin(
            Path(soop_reference_definition_path).expanduser().resolve(),
            allow_legacy_out_of_plane=True,
        )
        if not any(
            primitive.function in {"U", "IMPD"}
            for primitive in reference_definition.primitives
        ):
            raise ChartLifecycleError("SOOP reference must contain the pre-SOOP out-of-plane basis")
        if Counter(gic.family for gic in reference_definition.gics) != Counter(
            gic.family for gic in definition.gics
        ):
            raise ChartLifecycleError("initial SOOP coordinates differ from reference family dimensions")
        if not np.allclose(
            reference_definition.reference_coordinates_angstrom,
            coordinates,
            rtol=0.0,
            atol=1.0e-8,
        ):
            raise ChartLifecycleError("SOOP reference and optimizer start at different geometries")
        expected_definition = materialize_reference_soop_definition(
            reference_definition,
            definition,
            load_validated_oracle_candidate_pool(source),
            symmetry_operations=read_molecular_symmetry(source).operations,
        )
        if tuple(expected_definition.gics) != tuple(definition.gics) or tuple(
            expected_definition.primitives
        ) != tuple(definition.primitives):
            raise ChartLifecycleError(
                "initial SOOP definition does not match the supplied reference selection"
            )
    sequence = 0

    def artifact_provider(
        decision: OracleChartDecision,
        accepted_atoms: tuple[str, ...],
        accepted_coordinates: np.ndarray,
    ) -> Path:
        nonlocal sequence
        sequence += 1
        target = root / f"oracle_candidate_{sequence:06d}.xyzin"
        return materialize_optimization_chart_artifact(
            source,
            target,
            accepted_atoms,
            accepted_coordinates,
            task_regime=decision.task_regime,
            # A KEEP decision authorizes a SMITH-only re-chart at the
            # accepted geometry.  ORACLE's numerical state hash is allowed
            # to vary with that geometry even though the frozen topology and
            # contract identity remain unchanged; requiring the old hash
            # here incorrectly turns a valid SMITH refresh into a rebuild
            # failure.  ORACLE-authorized REBUILD decisions remain checked.
            expected_contract_sha256=(
                ""
                if decision.action == CHART_ACTION_KEEP
                else decision.oracle_contract_sha256
            ),
        )

    def soop_validity_evaluator(
        snapshot: ChartSnapshot,
        accepted_coordinates: np.ndarray,
    ):
        contract = read_oracle_sonic_contract(snapshot.candidate.source_xyzin_path)
        return audit_soop_geometric_validity(contract, accepted_coordinates)

    topology_screen_cache = {}

    def topology_irregularity_evaluator(
        snapshot: ChartSnapshot,
        accepted_coordinates: np.ndarray,
    ) -> bool:
        # This is only a distance-change screen against the frozen ORACLE
        # topology.  ORACLE alone decides whether the topology actually changes.
        current = np.asarray(accepted_coordinates, dtype=float)
        source_path = snapshot.candidate.source_xyzin_path
        if source_path is None:
            return True
        if source_path not in topology_screen_cache:
            frozen = read_oracle_sonic_contract(source_path)
            topology_screen_cache[source_path] = (
                np.asarray(read_enriched_xyz(source_path).coordinates_angstrom, dtype=float),
                {
                    tuple(sorted((int(left) - 1, int(right) - 1)))
                    for left, right in frozen.primary_topology.bonds
                },
            )
        reference, bonds = topology_screen_cache[source_path]
        for left in range(len(current)):
            for right in range(left + 1, len(current)):
                old_distance = float(np.linalg.norm(reference[left] - reference[right]))
                new_distance = float(np.linalg.norm(current[left] - current[right]))
                if old_distance <= 1.0e-12:
                    continue
                if (left, right) in bonds:
                    if new_distance > 1.25 * old_distance:
                        return True
                elif new_distance < 2.0 and new_distance < 0.75 * old_distance:
                    return True
        return False

    return ChartLifecycleController(
        ChartSnapshot(epoch=0, candidate=initial),
        oracle=OraclePerceptionChartAdapter(assessor),
        smith=SmithXyzinChartRebuilder(
            artifact_provider,
            symmetrize=bool(definition.symmetrize),
            reference_definition=reference_definition,
        ),
        smith_refresh_interval=int(smith_refresh_interval),
        soop_validity_evaluator=soop_validity_evaluator,
        oracle_condition_threshold=1.0e3,
        topology_irregularity_evaluator=topology_irregularity_evaluator,
    )


__all__ = [
    "CHART_ACTION_DEFER",
    "CHART_ACTION_INVALID",
    "CHART_ACTION_KEEP",
    "CHART_ACTION_REBUILD",
    "CHART_TASK_MINIMUM",
    "CHART_TASK_TRANSITION_STATE",
    "LINK_CHART_LIFECYCLE_SCHEMA",
    "ChartCandidate",
    "ChartIdentity",
    "ChartLifecycleController",
    "ChartLifecycleError",
    "ChartLifecycleResult",
    "ChartSnapshot",
    "OracleChartAssessor",
    "OracleChartDecision",
    "OraclePerceptionChartAdapter",
    "SmithXyzinChartRebuilder",
    "SmithChartRebuilder",
    "atom_order_sha256",
    "chart_lifecycle_controller_from_xyzin",
    "chart_lifecycle_result_to_json",
]
