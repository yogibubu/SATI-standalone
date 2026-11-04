"""ORACLE-owned lifecycle decisions for optimizer coordinate charts.

The state machine compares semantic perception states.  It does not build
coordinates and it does not inspect optimizer or backend data.  MINIMUM and
TRANSITION_STATE use distinct, explicit identities; a persistent identity
change is required before a chart rebuild is authorized.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .perception_robustness import OraclePerceptionState, perceive_oracle_state
from .perception_workflow import PerceptionBasinPolicy
from .selective_topology import SelectiveTopologyTracker


ORACLE_OPTIMIZATION_CHART_SCHEMA = "matrix.oracle.optimization_chart.v2"
OPTIMIZATION_CHART_MINIMUM = "MINIMUM"
OPTIMIZATION_CHART_TRANSITION_STATE = "TRANSITION_STATE"


@dataclass(frozen=True)
class OptimizationChartIdentity:
    task_regime: str
    contract_sha256: str
    topology_sha256: str
    state_sha256: str


@dataclass(frozen=True)
class OptimizationChartAssessment:
    action: str
    task_regime: str
    reason: str
    contract_sha256: str
    topology_sha256: str
    persistence_certificate: str = ""
    topology_update_atoms: tuple[int, ...] = ()


class OptimizationChartAssessor:
    """Apply ORACLE persistence policy to accepted Cartesian geometries."""

    def __init__(
        self,
        atomic_numbers: Sequence[int],
        reference_coordinates_angstrom: np.ndarray,
        *,
        task_regime: str,
        policy: PerceptionBasinPolicy = PerceptionBasinPolicy(),
    ) -> None:
        self.atomic_numbers = tuple(int(value) for value in atomic_numbers)
        self.task_regime = _task_regime(task_regime)
        self.policy = policy
        reference = perceive_oracle_state(
            self.atomic_numbers,
            np.asarray(reference_coordinates_angstrom, dtype=float),
        )
        self._reference = optimization_chart_identity(reference, self.task_regime)
        self._candidate: OptimizationChartIdentity | None = None
        self._candidate_frames = 0
        self._observed = self._reference
        self._topology_tracker = SelectiveTopologyTracker(
            displacement_threshold_angstrom=(
                policy.topology_displacement_threshold_angstrom
            ),
            neighbor_hops=policy.topology_neighbor_hops,
        )
        from matrix_chem import build_topology_objects

        _continuous, discrete, _rings, _synthons, _aromaticity = (
            build_topology_objects(reference_coordinates_angstrom, self.atomic_numbers)
        )
        from matrix_chem.spatial_regions import PersistentNeighborList

        self._neighbor_list = PersistentNeighborList(
            cutoff=float(_continuous.local_cutoff_angstrom),
        )
        self._neighbor_list.candidate_pairs(
            np.asarray(reference_coordinates_angstrom, dtype=float)
        )
        self._reference_bonds = tuple(
            sorted(tuple(sorted(map(int, pair))) for pair in discrete.bonds)
        )
        self._topology_tracker.affected_atoms(
            np.asarray(reference_coordinates_angstrom, dtype=float),
            self._reference_bonds,
        )
        self._last_topology_update_atoms = tuple(range(len(self.atomic_numbers)))


    @property
    def reference_identity(self) -> OptimizationChartIdentity:
        return self._reference

    @property
    def last_topology_update_atoms(self) -> tuple[int, ...]:
        """Zero-based ORACLE invalidation domain used at the latest LINK state."""

        return self._last_topology_update_atoms

    def assess_accepted_geometry(
        self,
        coordinates_angstrom: np.ndarray,
    ) -> OptimizationChartAssessment:
        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        affected = self._topology_tracker.affected_atoms(
            coordinates,
            self._reference_bonds,
        )
        self._last_topology_update_atoms = affected
        if not affected:
            return _assessment(
                "KEEP",
                self._observed,
                "ORACLE_SELECTIVE_TOPOLOGY_DOMAIN_EMPTY",
                topology_update_atoms=(),
            )
        state = perceive_oracle_state(
            self.atomic_numbers,
            coordinates,
            candidate_pairs=self._neighbor_list.candidate_pairs(coordinates),
        )
        observed = optimization_chart_identity(state, self.task_regime)
        self._observed = observed
        if observed.contract_sha256 == self._reference.contract_sha256:
            self._candidate = None
            self._candidate_frames = 0
            return _assessment(
                "KEEP",
                observed,
                "ORACLE_SEMANTIC_IDENTITY_UNCHANGED",
                topology_update_atoms=affected,
            )
        if (
            self._candidate is not None
            and self._candidate.contract_sha256 == observed.contract_sha256
        ):
            self._candidate_frames += 1
            self._candidate = observed
        else:
            self._candidate = observed
            self._candidate_frames = 1
        if self._candidate_frames < self.policy.persistent_change_window:
            return _assessment(
                "DEFER",
                observed,
                "ORACLE_CHANGE_AWAITS_PERSISTENCE_WINDOW",
                topology_update_atoms=affected,
            )
        certificate = _sha256(
            {
                "schema": ORACLE_OPTIMIZATION_CHART_SCHEMA,
                "task_regime": self.task_regime,
                "reference": self._reference.contract_sha256,
                "candidate": observed.contract_sha256,
                "frames": self._candidate_frames,
                "required_frames": self.policy.persistent_change_window,
            }
        )
        return _assessment(
            "REBUILD",
            observed,
            "PERSISTENT_ORACLE_SEMANTIC_IDENTITY_CHANGE",
            persistence_certificate=certificate,
            topology_update_atoms=affected,
        )

    def require_rebuild_for_invalid_chart(
        self,
        assessment: OptimizationChartAssessment,
        *,
        chart_domain_reason: str,
    ) -> OptimizationChartAssessment:
        """Authorize the current ORACLE state when the fixed chart is unusable.

        Persistence protects a numerically valid chart from transient chemical
        perception changes.  It cannot defer recovery from a chart whose
        mathematical domain or Wilson rank is already invalid.  LINK supplies
        only that numerical fact; ORACLE still owns and signs the complete
        scientific identity used by the replacement chart.
        """

        reason = str(chart_domain_reason).strip()
        if not reason:
            raise ValueError("invalid-chart rebuild requires an auditable domain reason")
        observed = self._observed
        if (
            assessment.task_regime != self.task_regime
            or assessment.contract_sha256 != observed.contract_sha256
            or assessment.topology_sha256 != observed.topology_sha256
        ):
            raise ValueError("invalid-chart rebuild assessment is stale")
        self._candidate = observed
        self._candidate_frames = max(1, self._candidate_frames)
        certificate = _sha256(
            {
                "schema": ORACLE_OPTIMIZATION_CHART_SCHEMA,
                "task_regime": self.task_regime,
                "reference": self._reference.contract_sha256,
                "candidate": observed.contract_sha256,
                "trigger": "INVALID_FIXED_CHART",
                "chart_domain_reason": reason,
            }
        )
        return _assessment(
            "REBUILD",
            observed,
            "ORACLE_REBUILD_REQUIRED_BY_INVALID_FIXED_CHART",
            persistence_certificate=certificate,
            topology_update_atoms=self._last_topology_update_atoms,
        )

    def commit_rebuild(self, assessment: OptimizationChartAssessment) -> None:
        """Commit a candidate only after LINK validates the SMITH rebuild."""

        if assessment.action != "REBUILD" or self._candidate is None:
            raise ValueError("only a pending ORACLE rebuild can be committed")
        if assessment.contract_sha256 != self._candidate.contract_sha256:
            raise ValueError("ORACLE rebuild assessment is stale")
        self._reference = self._candidate
        self._candidate = None
        self._candidate_frames = 0


_REBUILT_ARTIFACT_SECTIONS = (
    "BASIC",
    "SYMMETRY",
    "PERMUTATION_INVERSION",
    "ORACLE_GEOMETRY_IDENTITY",
    "TOPOLOGY",
    "AROMATICITY",
    "SYNTHONS",
    "PRIMITIVES",
    "ORACLE_SONIC_CONTRACT",
    "ORACLE_COORDINATE_ATLAS",
    "ORACLE_TRANSITION_STATE_GEOMETRY",
    "FRAGMENTS",
    "INTERACTION_CENTERS",
    "GIC",
    "SYCART",
    "ONIC_BLOCKS",
    "TYPED_ONIC",
    "VALIDATION",
)


def materialize_optimization_chart_artifact(
    source_xyzin: Path | str,
    target_xyzin: Path | str,
    atoms: Sequence[str],
    coordinates_angstrom: np.ndarray,
    *,
    task_regime: str,
    expected_contract_sha256: str = "",
) -> Path:
    """Rebuild ORACLE evidence at an accepted canonical Cartesian frame.

    The source contributes only persistent input evidence such as electronic
    annotations. Every geometry-derived ORACLE section and every downstream
    SMITH section is regenerated. Coordinates are never transformed here.
    """

    from matrix_chem import (
        MolecularGeometry,
        analyze_molecular_symmetry,
        build_geometry_identity_certificate,
        read_enriched_xyz,
        read_oracle_transition_state_geometry_contract,
        write_geometry_identity_certificate,
        write_validation_section,
    )
    from matrix_chem.link import (
        read_symmetry_thresholds,
        write_basic_section_from_geometry,
        write_primitive_coordinate_section,
        write_symmetry_section,
        write_topology_and_synthons_sections,
    )
    from matrix_chem.topology.elements import atomic_number
    from matrix_core import (
        read_sectioned_lines,
        remove_section_from_lines,
        replace_xyz_block_in_lines,
        write_sectioned_lines,
    )
    from matrix_fragments import write_fragment_build_section

    source = Path(source_xyzin).expanduser().resolve()
    target = Path(target_xyzin).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"optimization chart source does not exist: {source}")
    labels = tuple(str(atom) for atom in atoms)
    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    if coordinates.shape != (len(labels), 3) or not np.all(np.isfinite(coordinates)):
        raise ValueError("accepted optimization geometry must be finite natoms x 3")
    source_geometry = read_enriched_xyz(source)
    if tuple(source_geometry.atoms) != labels:
        raise ValueError("accepted optimization atom order differs from the source artifact")
    regime = _task_regime(task_regime)
    lifecycle_anchor = (
        read_oracle_transition_state_geometry_contract(source)
        if regime == OPTIMIZATION_CHART_TRANSITION_STATE
        else None
    )

    lines = read_sectioned_lines(source)
    for section_name in _REBUILT_ARTIFACT_SECTIONS:
        lines = remove_section_from_lines(lines, section_name)
    geometry = MolecularGeometry(
        atoms=labels,
        coordinates_angstrom=coordinates,
        comment="MATRIX accepted post-ORACLE optimization chart geometry",
        source_format="link_accepted_chart",
        source_path=target,
        charge=source_geometry.charge,
        multiplicity=source_geometry.multiplicity,
        metadata=dict(source_geometry.metadata),
    )
    xyz_lines = [
        str(len(labels)),
        geometry.comment,
        *(
            f"{atom} {xyz[0]:.16g} {xyz[1]:.16g} {xyz[2]:.16g}"
            for atom, xyz in zip(labels, coordinates)
        ),
    ]
    write_sectioned_lines(target, replace_xyz_block_in_lines(lines, xyz_lines))
    serialized = read_enriched_xyz(target)
    identity = build_geometry_identity_certificate(
        labels,
        coordinates,
        serialized.atoms,
        serialized.coordinates_angstrom,
    )
    write_geometry_identity_certificate(target, identity)

    thresholds = read_symmetry_thresholds(source)
    symmetry = analyze_molecular_symmetry(
        serialized,
        distance_tolerance=thresholds.distance_angstrom,
        inertia_tolerance=thresholds.inertia_relative,
        max_rotation_order=thresholds.max_rotation_order,
    )
    write_basic_section_from_geometry(
        target,
        geometry=serialized,
        point_group=symmetry.point_group,
    )
    write_symmetry_section(target, symmetry=symmetry, thresholds=thresholds)
    write_topology_and_synthons_sections(target, serialized)
    write_fragment_build_section(target)
    # Materialize ORACLE's special interaction-centre primitives before
    # constructing the SONIC contract.  In particular, metal--ring eta_n
    # contacts must cross the ORACLE -> SMITH boundary as explicit
    # CENTER_ATOM_DIST candidates; a fragment section alone only records the
    # ordinary fragment centre and silently drops haptic coordinates.
    from matrix_fragments import write_interaction_center_section

    write_interaction_center_section(target)
    write_primitive_coordinate_section(target, serialized)
    from .sonic_contract_builder import write_oracle_sonic_contract_from_xyzin

    write_oracle_sonic_contract_from_xyzin(target)
    if regime == OPTIMIZATION_CHART_TRANSITION_STATE:
        from .transition_state_geometry import (
            write_oracle_transition_state_geometry_contract_from_xyzin,
        )

        write_oracle_transition_state_geometry_contract_from_xyzin(
            target,
            lifecycle_anchor=lifecycle_anchor,
        )
    write_validation_section(target)

    numbers = tuple(int(atomic_number(atom) or 0) for atom in labels)
    if any(number <= 0 for number in numbers):
        raise ValueError("optimization chart contains an unsupported atomic label")
    observed = optimization_chart_identity(
        perceive_oracle_state(numbers, serialized.coordinates_angstrom),
        regime,
    )
    expected = str(expected_contract_sha256).strip().lower()
    if expected and observed.contract_sha256 != expected:
        raise RuntimeError("materialized ORACLE chart differs from the assessed identity")
    return target


def optimization_chart_identity(
    state: OraclePerceptionState,
    task_regime: str,
) -> OptimizationChartIdentity:
    """Return the regime-specific semantic identity of one ORACLE state."""

    regime = _task_regime(task_regime)
    # Lifecycle identity contains only properties whose persistent change can
    # alter the scientific basin or the coordinate atlas.  Atom equivalence
    # classes and local point-group labels remain valuable perception evidence,
    # but they may split and merge under harmless finite motion while the
    # topology, fragments, contacts, sites, and global symmetry are unchanged.
    # Including that numerical equivalence jitter here causes spurious chart
    # rebuilds and is therefore deliberately forbidden for both task regimes.
    basin = {
        "topology": state.topology_hash,
        "fragments": state.fragment_signatures,
        "rings": state.ring_signatures,
        "structural_sites": state.structural_site_signatures,
        "multicenter_domains": state.multicenter_signatures,
        "contacts": state.contact_signatures,
        "primitive_candidates": state.primitive_signatures,
        "cycle_ranks": (state.primary_cycle_rank, state.auxiliary_cycle_rank),
        "strict_symmetry": (state.strict_group, state.strict_operation_signatures),
    }
    scientific = (
        {"minimum_basin": basin}
        if regime == OPTIMIZATION_CHART_MINIMUM
        else {
            "transition_state_basin": basin,
            "transition_signatures": state.transition_signatures,
        }
    )
    return OptimizationChartIdentity(
        task_regime=regime,
        contract_sha256=_sha256(
            {
                "schema": ORACLE_OPTIMIZATION_CHART_SCHEMA,
                "task_regime": regime,
                "scientific_identity": scientific,
            }
        ),
        topology_sha256=str(state.topology_hash),
        state_sha256=str(state.state_hash),
    )


def _assessment(
    action: str,
    identity: OptimizationChartIdentity,
    reason: str,
    *,
    persistence_certificate: str = "",
    topology_update_atoms: tuple[int, ...] = (),
) -> OptimizationChartAssessment:
    return OptimizationChartAssessment(
        action=action,
        task_regime=identity.task_regime,
        reason=reason,
        contract_sha256=identity.contract_sha256,
        topology_sha256=identity.topology_sha256,
        persistence_certificate=persistence_certificate,
        topology_update_atoms=tuple(int(atom) for atom in topology_update_atoms),
    )


def _task_regime(value: str) -> str:
    normalized = str(value).strip().upper()
    if normalized not in {
        OPTIMIZATION_CHART_MINIMUM,
        OPTIMIZATION_CHART_TRANSITION_STATE,
    }:
        raise ValueError("optimization chart regime must be MINIMUM or TRANSITION_STATE")
    return normalized


def _sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "OPTIMIZATION_CHART_MINIMUM",
    "OPTIMIZATION_CHART_TRANSITION_STATE",
    "ORACLE_OPTIMIZATION_CHART_SCHEMA",
    "OptimizationChartAssessment",
    "OptimizationChartAssessor",
    "OptimizationChartIdentity",
    "materialize_optimization_chart_artifact",
    "optimization_chart_identity",
]
