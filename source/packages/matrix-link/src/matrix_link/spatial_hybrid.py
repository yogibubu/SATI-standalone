"""Adaptive spatial SONIC/SYCART optimization controlled by ORACLE domains."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .optimizer import (
    ANGSTROM_TO_BOHR,
    GeometryEvaluationService,
    OptimizerCoordinateModel,
    OptimizerResult,
    OptimizerSettings,
    coordinate_model_from_xyzin,
    optimize_geometry,
)


LINK_SPATIAL_HYBRID_SCHEMA = "matrix.link.spatial_hybrid.v1"


@dataclass(frozen=True)
class SpatialHybridStage:
    cycle: int
    coordinate_kind: str
    active_atoms: tuple[int, ...]
    fixed_atoms: tuple[int, ...]
    source_xyzin: Path
    result: OptimizerResult


@dataclass(frozen=True)
class SpatialHybridOptimizationResult:
    stages: tuple[SpatialHybridStage, ...]
    terminal_result: OptimizerResult
    final_sonic: OptimizerResult | None
    final_coordinate_kind: str
    final_xyzin: Path
    audit_path: Path

    @property
    def converged(self) -> bool:
        return bool(self.terminal_result.converged)

    @property
    def total_steps(self) -> int:
        return sum(len(stage.result.iterations) for stage in self.stages) + len(
            self.terminal_result.iterations
        )

    @property
    def optimizer_wall_seconds(self) -> float:
        return sum(stage.result.optimizer_wall_seconds for stage in self.stages) + float(
            self.terminal_result.optimizer_wall_seconds
        )

    @property
    def calculator_wall_seconds(self) -> float:
        return sum(stage.result.calculator_wall_seconds for stage in self.stages) + float(
            self.terminal_result.calculator_wall_seconds
        )


def optimize_geometry_spatial_hybrid(
    xyzin_path: Path | str,
    *,
    run_dir: Path | str,
    backend=None,
    settings: OptimizerSettings | None = None,
    task_regime: str = "MINIMUM",
    reactive_atom_indices: Sequence[int] = (),
    additional_uncertain_atom_indices: Sequence[int] = (),
    sycart_steps_per_cycle: int = 5,
    stable_sonic_steps_per_cycle: int = 5,
    maximum_mixed_cycles: int = 36,
    gradient_stationary_threshold: float = 0.50,
    complete_sonic_handoff_gradient: float = 0.05,
    force_domain_gradient_threshold: float = 0.001,
    engine_command: str = "",
    timeout: float | None = None,
) -> SpatialHybridOptimizationResult:
    """Optimize local uncertain regions in SYCART without delaying SONIC.

    Each cycle moves only the ORACLE-uncertain atoms in a Cartesian chart,
    then moves the certified complement with SONIC rows that have no support
    on those atoms.  ORACLE is rerun at the accepted geometry.  As soon as the
    topology is well defined, SMITH rebuilds a complete chart and LINK finishes
    in SONIC.  A TS supplies its reactive atoms through the same partition;
    its final stage remains a complete SONIC transition-state optimization.
    """

    regime = str(task_regime).strip().upper().replace("-", "_")
    if regime not in {"MINIMUM", "TRANSITION_STATE"}:
        raise ValueError("spatial hybrid task_regime must be MINIMUM or TRANSITION_STATE")
    if min(int(sycart_steps_per_cycle), int(stable_sonic_steps_per_cycle)) <= 0:
        raise ValueError("spatial hybrid block lengths must be positive")
    if int(maximum_mixed_cycles) <= 0:
        raise ValueError("spatial hybrid maximum_mixed_cycles must be positive")
    if not np.isfinite(complete_sonic_handoff_gradient) or complete_sonic_handoff_gradient <= 0.0:
        raise ValueError("complete SONIC handoff gradient threshold must be positive")
    if not np.isfinite(force_domain_gradient_threshold) or force_domain_gradient_threshold <= 0.0:
        raise ValueError("force-domain gradient threshold must be positive")

    from matrix_chem import read_xyzin_geometry
    from matrix_chem.topology.covalent_radii import covalent_radius
    from matrix_chem.topology.elements import atomic_number
    from matrix_oracle.topology_domains import assess_topology_domains
    from matrix_smith.definition import read_gic_definition_from_xyzin
    from matrix_smith.mixed_domains import (
        build_global_sycart_domain_plan,
        build_mixed_domain_plan,
    )

    root = Path(run_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = Path(xyzin_path).expanduser().resolve()
    geometry = read_xyzin_geometry(source)
    atoms = tuple(geometry.atoms)
    numbers = tuple(int(atomic_number(atom) or 0) for atom in atoms)
    if any(number <= 0 for number in numbers):
        raise ValueError("spatial hybrid input contains an unsupported atom")
    coordinates = np.asarray(geometry.coordinates_angstrom, dtype=float)
    base = settings or OptimizerSettings(
        stationary_point=("transition_state" if regime == "TRANSITION_STATE" else "minimum")
    )
    try:
        from matrix_chem import topology_snapshot_from_xyzin

        if len(topology_snapshot_from_xyzin(source).get("fragments", ())) > 1:
            base = replace(base, fragment_hessian_blocks=True)
    except (ImportError, OSError, RuntimeError, ValueError, TypeError):
        # Chart-free inputs are handled by the Cartesian domain route; the
        # block seed is enabled automatically once ORACLE exposes fragments.
        pass
    final_sonic_entry_threshold = max(float(base.gradient_tolerance), 5.0e-4)
    expected_stationary = "transition_state" if regime == "TRANSITION_STATE" else "minimum"
    if str(base.stationary_point).strip().lower() != expected_stationary:
        base = replace(base, stationary_point=expected_stationary)

    current = root / "chart_000.xyzin"
    try:
        initial_frozen_definition = read_gic_definition_from_xyzin(source)
    except (RuntimeError, ValueError):
        initial_frozen_definition = None
    chart_certified, chart_available, soop_fallback_atoms = _materialize_chart(
        source,
        current,
        atoms,
        coordinates,
        regime,
        fallback_definition=initial_frozen_definition,
    )
    soop_recovery_successes = 0

    # The mandatory first Cartesian gradient classifies both distance from a
    # stationary region and the initial ORACLE topology.  It is evaluated in
    # the resident calculator session and recorded separately from optimizer
    # iterations.
    classification_model = coordinate_model_from_xyzin(
        current,
        kind="sycart",
        task_regime="MINIMUM",
        matrix_free=True,
    )
    classification_settings = replace(
        base,
        stationary_point="minimum",
        step_model="lbfgs",
        compute_final_hessian=False,
        require_exact_final_hessian=False,
        verify_irc=False,
    )
    classification_service = GeometryEvaluationService(
        xyzin_path=current,
        run_dir=root / "classification",
        coordinate_model=classification_model,
        engine_command=engine_command,
        backend=backend,
        timeout=timeout,
        settings=classification_settings,
    )
    classification = classification_service.evaluate(
        np.zeros(len(classification_model.labels), dtype=float),
        tag="spatial-hybrid-initial-cartesian-gradient",
        requested_properties=("energy", "gradient"),
    )
    gradient = classification.gradient_hartree_per_bohr
    if gradient is None:
        raise RuntimeError("spatial hybrid classification requires a Cartesian gradient")
    initial_gradient_inf = float(np.max(np.abs(gradient), initial=0.0))

    def reassess_cartesian_gradient(path: Path, directory: Path, tag: str) -> np.ndarray:
        """Evaluate the full Cartesian gradient used by the domain router."""

        reassessment_model = coordinate_model_from_xyzin(
            path,
            kind="sycart",
            task_regime="MINIMUM",
            matrix_free=True,
        )
        reassessment_service = GeometryEvaluationService(
            xyzin_path=path,
            run_dir=directory,
            coordinate_model=reassessment_model,
            engine_command=engine_command,
            backend=backend,
            timeout=timeout,
            settings=classification_settings,
        )
        evaluation = reassessment_service.evaluate(
            np.zeros(len(reassessment_model.labels), dtype=float),
            tag=tag,
            requested_properties=("energy", "gradient"),
        )
        if evaluation.gradient_hartree_per_bohr is None:
            raise RuntimeError("spatial hybrid reassessment requires a Cartesian gradient")
        return np.asarray(evaluation.gradient_hartree_per_bohr, dtype=float)

    stages: list[SpatialHybridStage] = []
    audits: list[dict[str, object]] = []
    force_one_global_sycart = False
    carry_uncertain: tuple[int, ...] = ()
    sonic_cooldown = 0
    for cycle in range(1, int(maximum_mixed_cycles) + 1):
        assessment = assess_topology_domains(
            numbers,
            coordinates,
            task_regime=regime,
            gradient_hartree_per_bohr=gradient,
            reactive_atom_indices=reactive_atom_indices,
            additional_uncertain_atom_indices=tuple(sorted(set(
                additional_uncertain_atom_indices
            ).union(soop_fallback_atoms))),
            gradient_stationary_threshold=gradient_stationary_threshold,
        )
        uncertain = assessment.uncertain_atoms
        gradient_inf = float(np.max(np.abs(gradient), initial=0.0))
        cartesian_gradient = np.asarray(gradient, dtype=float).reshape(len(atoms), 3)
        active_force_domain_threshold = (
            float(force_domain_gradient_threshold)
            if not stages
            else min(
                float(force_domain_gradient_threshold),
                0.5 * final_sonic_entry_threshold,
            )
        )
        high_force_atoms = tuple(
            int(atom)
            for atom in np.flatnonzero(
                np.max(np.abs(cartesian_gradient), axis=1)
                > active_force_domain_threshold
            )
        )
        if high_force_atoms:
            uncertain = tuple(sorted(set(uncertain).union(high_force_atoms)))
            force_halo = set(high_force_atoms)
            radii = tuple(float(covalent_radius(number) or 0.75) for number in numbers)
            for seed in high_force_atoms:
                distances = np.linalg.norm(coordinates - coordinates[seed], axis=1)
                for atom, distance in enumerate(distances):
                    if atom == seed:
                        continue
                    if float(distance) <= 1.35 * (radii[seed] + radii[atom]):
                        force_halo.add(atom)
            uncertain = tuple(sorted(set(uncertain).union(force_halo)))
        if uncertain:
            # The first stable shell is a Schwarz halo, not a new claim of
            # topological uncertainty.  Moving a reactive/ambiguous core
            # against a frozen covalent boundary creates artificial forces;
            # including this bounded shell lets the local Cartesian block
            # relax its coupling while preserving linear storage.
            uncertain = tuple(sorted(set(uncertain).union(assessment.boundary_atoms)))
        if len(uncertain) > max(32, int(np.ceil(0.25 * len(atoms)))):
            # A distributed high-force region is no longer local.  Switch to
            # the O(N)-storage global SYCART projector instead of materializing
            # an O(mN) subset direction matrix with m proportional to N.
            uncertain = tuple(range(len(atoms)))
        if not chart_certified and carry_uncertain:
            # ORACLE may already see a discrete topology while SMITH still
            # rejects the complete SONIC chart numerically.  Continue only the
            # preceding local recovery domain until both gates pass.
            uncertain = carry_uncertain
        elif (
            not uncertain
            and carry_uncertain
            and gradient_inf > final_sonic_entry_threshold
        ):
            # A discrete bond graph can become certifiable before the local
            # high-gradient block is ready for a coupled nonlinear SONIC
            # realization.  Preserve that block for another alternating
            # macrocycle; this is a numerical handoff gate, not a demand that
            # SYCART converge the whole molecule.
            uncertain = carry_uncertain
        uncertain = tuple(sorted(set(uncertain).union(soop_fallback_atoms)))
        if not uncertain and assessment.geometry_near_stationary is False and not stages:
            # A chemically unambiguous but very poor starting geometry needs
            # only one topology-independent macroblock, never SYCART convergence.
            uncertain = tuple(range(len(atoms)))
            force_one_global_sycart = True
        if (
            not uncertain
            and chart_certified
            and gradient_inf
            <= (
                float(complete_sonic_handoff_gradient)
                if not stages
                else final_sonic_entry_threshold
            )
        ):
            audits.append({"cycle": cycle, "oracle": assessment.to_dict(), "action": "FINAL_SONIC"})
            break

        chart_free_global_sycart = not chart_available
        definition = (
            None
            if chart_free_global_sycart
            else read_gic_definition_from_xyzin(current)
        )
        carry_uncertain = tuple(uncertain)
        plan_boundary = tuple(
            atom for atom in assessment.boundary_atoms if atom not in set(uncertain)
        )
        plan = (
            build_global_sycart_domain_plan(len(atoms), task_regime=regime)
            if chart_free_global_sycart
            else build_mixed_domain_plan(
                definition,
                coordinates,
                uncertain_atom_indices=uncertain,
                boundary_atom_indices=plan_boundary,
                task_regime=regime,
            )
        )
        cycle_audit: dict[str, object] = {
            "cycle": cycle,
            "oracle": assessment.to_dict(),
            "smith": plan.to_dict(),
            "active_force_domain_threshold_hartree_per_bohr": (
                active_force_domain_threshold
            ),
            "stages": [],
        }
        if soop_fallback_atoms:
            cycle_audit["soop_component_fallback"] = {
                "policy": "WHOLE_COUPLED_RING_COMPONENT_LOCAL_SYCART",
                "atom_indices_zero_based": list(soop_fallback_atoms),
                "legacy_out_of_plane_allowed": False,
            }

        # A certified chart with no uncertain atoms is already ready for the
        # SONIC side of the fallback.  Do not manufacture an empty Cartesian
        # domain (which is undefined); use the complete chart directly.
        direct_sonic = not uncertain and chart_certified
        local_model = (
            coordinate_model_from_xyzin(
                current,
                kind="sonic",
                task_regime=regime,
                matrix_free=(regime == "MINIMUM"),
            )
            if direct_sonic
            else _local_sycart_model(current, coordinates, plan, regime)
        )
        local_block_steps = _adaptive_sycart_block_length(
            int(sycart_steps_per_cycle), gradient_inf
        )
        local_settings = replace(
            base,
            max_steps=local_block_steps,
            fixed_atoms=() if direct_sonic else plan.stable_atoms,
            stationary_point=expected_stationary,
            # The block is bounded by the local uncertainty domain.  LINK may
            # therefore select its small dense RFO kernel without changing
            # the end-to-end linear storage of the molecular calculation.
            step_model=(
                "lbfgs"
                if direct_sonic
                else ("dense_rfo" if regime == "TRANSITION_STATE" else "auto")
            ),
            compute_final_hessian=False,
            require_exact_final_hessian=False,
            verify_irc=False,
        )
        local_result = optimize_geometry(
            current,
            run_dir=root / f"cycle_{cycle:03d}_local_sycart",
            coordinate_model=local_model,
            backend=backend,
            engine_command=engine_command,
            timeout=timeout,
            settings=local_settings,
        )
        stages.append(
            SpatialHybridStage(
                cycle,
                "direct_sonic" if direct_sonic else "local_sycart",
                plan.uncertain_atoms,
                plan.stable_atoms,
                current,
                local_result,
            )
        )
        coordinates = np.asarray(local_result.final_coordinates_angstrom, dtype=float)
        cycle_audit["stages"].append(_stage_audit(stages[-1]))
        previous_definition = definition
        current = root / f"chart_{cycle:03d}_after_sycart.xyzin"
        chart_certified, chart_available, current_soop_failure_atoms = _materialize_chart(
            source,
            current,
            atoms,
            coordinates,
            regime,
            fallback_definition=previous_definition,
        )
        soop_fallback_atoms, soop_recovery_successes = (
            _advance_soop_fallback_hysteresis(
                soop_fallback_atoms,
                soop_recovery_successes,
                current_soop_failure_atoms,
            )
        )
        if current_soop_failure_atoms:
            cycle_audit["soop_component_fallback_after_sycart"] = {
                "policy": "WHOLE_COUPLED_RING_COMPONENT_LOCAL_SYCART",
                "atom_indices_zero_based": list(current_soop_failure_atoms),
                "legacy_out_of_plane_allowed": False,
            }
        elif soop_fallback_atoms:
            cycle_audit["soop_fragment_recovery_hysteresis"] = {
                "policy": "TWO_CONSECUTIVE_CERTIFIED_REBUILDS",
                "successful_rebuilds": soop_recovery_successes,
                "atom_indices_zero_based": list(soop_fallback_atoms),
            }
        gradient = reassess_cartesian_gradient(
            current,
            root / f"cycle_{cycle:03d}_after_sycart_reassessment",
            f"spatial-hybrid-cycle-{cycle}-after-sycart",
        )
        cycle_audit["after_sycart_cartesian_gradient_inf"] = float(
            np.max(np.abs(gradient), initial=0.0)
        )
        cycle_audit["sycart_reduction_factor"] = (
            cycle_audit["after_sycart_cartesian_gradient_inf"] / gradient_inf
            if gradient_inf > 0.0
            else 0.0
        )

        # Rebuild before selecting stable rows: the local Cartesian move may
        # already have resolved part of the uncertain topology.
        after_local = assess_topology_domains(
            numbers,
            coordinates,
            task_regime=regime,
            gradient_hartree_per_bohr=gradient,
            reactive_atom_indices=reactive_atom_indices,
            additional_uncertain_atom_indices=tuple(sorted(set(
                additional_uncertain_atom_indices
            ).union(soop_fallback_atoms))),
            gradient_stationary_threshold=gradient_stationary_threshold,
        )
        after_local_gradient_inf = float(np.max(np.abs(gradient), initial=0.0))
        if not chart_certified and not (chart_available and soop_fallback_atoms):
            cycle_audit["stable_sonic_deferred"] = {
                "reason": "NO_CERTIFIED_SMITH_CHART",
                "next_action": "CONTINUE_MATRIX_FREE_SYCART_AND_REASSESS_ORACLE",
            }
            audits.append(cycle_audit)
            continue
        definition = read_gic_definition_from_xyzin(current)
        complete_chart_ready_after_local = (
            regime == "MINIMUM"
            and chart_certified
            and after_local_gradient_inf <= float(complete_sonic_handoff_gradient)
        )
        if complete_chart_ready_after_local:
            # The next SONIC macroblock may use the complete certified chart.
            # It remains a short block: a fresh Cartesian gradient decides
            # whether another local SYCART correction is needed.
            cycle_audit["action"] = "COMPLETE_SONIC_MACROBLOCK_AFTER_LOCAL"
            cycle_audit["handoff_evidence"] = (
                "ORACLE_TOPOLOGY_AND_SMITH_CHART"
                if after_local.topology_well_defined
                else "CONSERVATIVE_ORACLE_FLAG_OVERRIDDEN_BY_CERTIFIED_SMITH_CHART"
            )
        sonic_uncertain = tuple(sorted(
            set(after_local.uncertain_atoms).union(soop_fallback_atoms)
        ))
        if complete_chart_ready_after_local:
            sonic_uncertain = ()
        if (
            not sonic_uncertain
            and carry_uncertain
            and after_local_gradient_inf > float(complete_sonic_handoff_gradient)
        ):
            sonic_uncertain = carry_uncertain
        if regime == "TRANSITION_STATE" and reactive_atom_indices:
            sonic_uncertain = tuple(sorted(set(sonic_uncertain).union(reactive_atom_indices)))
        stable_plan = build_mixed_domain_plan(
            definition,
            coordinates,
            uncertain_atom_indices=sonic_uncertain,
            boundary_atom_indices=tuple(
                atom
                for atom in after_local.boundary_atoms
                if atom not in set(sonic_uncertain)
            ),
            task_regime=regime,
        )
        sonic_entry_allowed = sonic_cooldown == 0
        if (
            stable_plan.stable_sonic_indices
            and after_local_gradient_inf <= float(complete_sonic_handoff_gradient)
            and sonic_entry_allowed
        ):
            stable_model = _stable_sonic_model(definition, stable_plan)
            stable_settings = replace(
                base,
                max_steps=int(stable_sonic_steps_per_cycle),
                # Domain separation is already exact in row support.  A
                # second Cartesian freeze conflicts with the global removal
                # of rotations/translations and can make even an infinitesimal
                # partial-chart step unrealizable.
                fixed_atoms=(),
                stationary_point="minimum",
                step_model="lbfgs",
                compute_final_hessian=False,
                require_exact_final_hessian=False,
                verify_irc=False,
                # A partial chart with a fixed boundary may need more Wilson
                # corrections than a complete-chart handoff.  It is bounded
                # to five macrosteps, so retain the certified 60-step local
                # corrector here while the final handoff uses the fast guard.
                trust_radius=(
                    min(base.trust_radius, 0.01)
                    if complete_chart_ready_after_local
                    else base.trust_radius
                ),
                backtransform_max_substeps=(
                    min(base.backtransform_max_substeps, 4)
                    if complete_chart_ready_after_local
                    else max(base.backtransform_max_substeps, 60)
                ),
            )
            try:
                stable_result = optimize_geometry(
                    current,
                    run_dir=root / f"cycle_{cycle:03d}_stable_sonic",
                    coordinate_model=stable_model,
                    backend=backend,
                    engine_command=engine_command,
                    timeout=timeout,
                    settings=stable_settings,
                )
            except RuntimeError as exc:
                if "unable to realize a nonzero optimizer step" not in str(exc):
                    raise
                stable_result = None
                cycle_audit["stable_sonic_deferred"] = {
                    "reason": "NONZERO_STEP_NOT_REALIZABLE_ON_CURRENT_CHART",
                    "next_action": "REASSESS_AND_EXPAND_LOCAL_SYCART_DOMAIN",
                }
            if stable_result is not None:
                stages.append(
                    SpatialHybridStage(
                        cycle,
                        "stable_sonic",
                        stable_plan.stable_atoms,
                        (),
                        current,
                        stable_result,
                    )
                )
                coordinates = np.asarray(stable_result.final_coordinates_angstrom, dtype=float)
                cycle_audit["stages"].append(_stage_audit(stages[-1]))
                previous_definition = definition
                current = root / f"chart_{cycle:03d}_after_sonic.xyzin"
                chart_certified, chart_available, current_soop_failure_atoms = _materialize_chart(
                    source,
                    current,
                    atoms,
                    coordinates,
                    regime,
                    fallback_definition=previous_definition,
                )
                soop_fallback_atoms, soop_recovery_successes = (
                    _advance_soop_fallback_hysteresis(
                        soop_fallback_atoms,
                        soop_recovery_successes,
                        current_soop_failure_atoms,
                    )
                )
                if (
                    not chart_certified
                    and not stable_plan.uncertain_atoms
                    and stable_result.iterations
                ):
                    # Accepted complete SONIC steps certify their sparse
                    # operator at the current geometry.
                    chart_certified = True
                    cycle_audit["runtime_chart_recertification"] = (
                        "ACCEPTED_COMPLETE_SONIC_SPARSE_OPERATOR"
                    )
                gradient = reassess_cartesian_gradient(
                    current,
                    root / f"cycle_{cycle:03d}_after_sonic_reassessment",
                    f"spatial-hybrid-cycle-{cycle}-after-sonic",
                )
                cycle_audit["after_sonic_cartesian_gradient_inf"] = float(
                    np.max(np.abs(gradient), initial=0.0)
                )
                sonic_reduction = (
                    cycle_audit["after_sonic_cartesian_gradient_inf"]
                    / after_local_gradient_inf
                    if after_local_gradient_inf > 0.0
                    else 0.0
                )
                cycle_audit["sonic_reduction_factor"] = sonic_reduction
                # A SONIC block that reduces the full Cartesian gradient by
                # less than two percent is locally sound but not currently
                # competitive.  Retain the chart and give SYCART two complete
                # macroblocks before trying SONIC again.  This decision uses
                # measured progress, not a molecule-specific iteration count.
                sycart_reduction = float(cycle_audit["sycart_reduction_factor"])
                sonic_gain = max(0.0, 1.0 - sonic_reduction)
                sycart_gain = max(0.0, 1.0 - sycart_reduction)
                if _sonic_block_needs_cooldown(sycart_reduction, sonic_reduction):
                    sonic_cooldown = 2
                    cycle_audit["next_sonic_cooldown_cycles"] = sonic_cooldown
                    cycle_audit["sonic_competitiveness"] = {
                        "sonic_fractional_gain": sonic_gain,
                        "sycart_fractional_gain": sycart_gain,
                        "minimum_relative_gain": 0.5,
                    }
        elif stable_plan.stable_sonic_indices:
            if sonic_cooldown:
                cycle_audit["stable_sonic_deferred"] = {
                    "reason": "RECENT_SONIC_BLOCK_NOT_COMPETITIVE",
                    "cooldown_cycles_remaining": sonic_cooldown,
                    "next_action": "CONTINUE_SYCART_LOCAL_RELAXATION",
                }
                sonic_cooldown -= 1
            else:
                cycle_audit["stable_sonic_deferred"] = {
                    "reason": "FULL_CARTESIAN_GRADIENT_ABOVE_SONIC_ENTRY_GATE",
                    "gradient_inf_hartree_per_bohr": after_local_gradient_inf,
                    "threshold_hartree_per_bohr": float(complete_sonic_handoff_gradient),
                }
        audits.append(cycle_audit)

        if force_one_global_sycart:
            force_one_global_sycart = False
            break
        assess_topology_domains(
            numbers,
            coordinates,
            task_regime=regime,
            gradient_hartree_per_bohr=gradient,
            reactive_atom_indices=(),
            additional_uncertain_atom_indices=tuple(sorted(set(
                additional_uncertain_atom_indices
            ).union(soop_fallback_atoms))),
            gradient_stationary_threshold=gradient_stationary_threshold,
        )
        # TS reactive seeds select the mixed block but do not prevent the
        # explicitly built reactive SONIC chart from taking over afterwards.
        if (
            chart_certified
            and float(np.max(np.abs(gradient), initial=0.0))
            <= final_sonic_entry_threshold
        ):
            break

    terminal_assessment = assess_topology_domains(
        numbers,
        coordinates,
        task_regime=regime,
        gradient_hartree_per_bohr=gradient,
        reactive_atom_indices=reactive_atom_indices,
        additional_uncertain_atom_indices=tuple(sorted(set(
            additional_uncertain_atom_indices
        ).union(soop_fallback_atoms))),
        gradient_stationary_threshold=gradient_stationary_threshold,
    )
    sonic_terminal = bool(
        chart_certified and len(terminal_assessment.uncertain_atoms) < len(atoms)
    )
    final_hessian_source = "spatial-hybrid-sonic-sh"
    if sonic_terminal:
        final_model = coordinate_model_from_xyzin(
            current,
            kind="sonic",
            task_regime=regime,
            matrix_free=(regime == "MINIMUM"),
        )
        final_coordinate_kind = "sonic"
        final_result = optimize_geometry(
            current,
            run_dir=root / "final_sonic",
            coordinate_model=final_model,
            backend=backend,
            engine_command=engine_command,
            timeout=timeout,
            settings=replace(
                base,
                fixed_atoms=(),
                trust_radius=min(base.trust_radius, 0.01),
                backtransform_max_substeps=min(base.backtransform_max_substeps, 4),
            ),
            initial_hessian_source=final_hessian_source,
        )
    else:
        # Topology-free aggregates and persistently ambiguous global domains
        # are legitimate terminal cases.  Finish with the O(N)-storage SYCART
        # projector instead of manufacturing a chemically arbitrary chart.
        final_model = coordinate_model_from_xyzin(
            current,
            kind="sycart",
            task_regime=regime,
            matrix_free=(regime == "MINIMUM"),
        )
        final_coordinate_kind = "sycart"
        final_hessian_source = "spatial-hybrid-topology-free-sycart"
        final_result = optimize_geometry(
            current,
            run_dir=root / "final_sycart",
            coordinate_model=final_model,
            backend=backend,
            engine_command=engine_command,
            timeout=timeout,
            settings=replace(base, fixed_atoms=()),
            initial_hessian_source=final_hessian_source,
        )
    audit_path = root / "spatial_hybrid_audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "schema": LINK_SPATIAL_HYBRID_SCHEMA,
                "task_regime": regime,
                "classification_energy_hartree": classification.energy_hartree,
                "classification_gradient_inf_hartree_per_bohr": initial_gradient_inf,
                "cycles": audits,
                "final_coordinate_kind": final_coordinate_kind,
                "final_initial_hessian_source": final_hessian_source,
                "final_converged": final_result.converged,
                "final_sonic_steps": (
                    len(final_result.iterations) if sonic_terminal else 0
                ),
                "final_sycart_steps": (
                    0 if sonic_terminal else len(final_result.iterations)
                ),
                "total_steps": sum(len(stage.result.iterations) for stage in stages)
                + len(final_result.iterations),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return SpatialHybridOptimizationResult(
        stages=tuple(stages),
        terminal_result=final_result,
        final_sonic=final_result if sonic_terminal else None,
        final_coordinate_kind=final_coordinate_kind,
        final_xyzin=current,
        audit_path=audit_path,
    )


def _materialize_chart(
    source,
    target,
    atoms,
    coordinates,
    regime,
    *,
    fallback_definition=None,
) -> tuple[bool, bool, tuple[int, ...]]:
    from matrix_oracle import materialize_optimization_chart_artifact
    from matrix_smith import (
        recenter_sonic_definition_from_xyzin,
        write_gicforge_build_sections,
        write_sycart_build_section,
    )

    materialize_optimization_chart_artifact(
        source,
        target,
        atoms,
        np.asarray(coordinates, dtype=float),
        task_regime=regime,
    )
    rebuilt = True
    failure = None
    try:
        write_gicforge_build_sections(
            target,
            symmetrize=True,
            out_of_plane_model="soop",
            fragment_context=(
                "transition_state" if regime == "TRANSITION_STATE" else "minimum"
            ),
        )
    except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        failure = exc
        # A locally symmetric projection can lose one numerical direction at
        # a slightly distorted C1 geometry although the unsymmetrized SONIC
        # primitive space has the required 3N-6 rank.  Rebuild that equivalent
        # C1 chart before falling back to an older definition.
        try:
            write_gicforge_build_sections(
                target,
                symmetrize=False,
                out_of_plane_model="soop",
                fragment_context=(
                    "transition_state" if regime == "TRANSITION_STATE" else "minimum"
                ),
            )
        except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            # Preserve a component-scoped SOOP failure from the symmetric
            # attempt if the generic C1 retry fails without a more precise
            # domain.  Otherwise LINK would silently expand a local recovery
            # into a chart-free/global SYCART route.
            previous_atoms = tuple(
                int(atom)
                for atom in getattr(failure, "atom_indices_zero_based", ())
            )
            current_atoms = tuple(
                int(atom)
                for atom in getattr(exc, "atom_indices_zero_based", ())
            )
            if current_atoms or not previous_atoms:
                failure = exc
            rebuilt = False
            if fallback_definition is not None:
                # Keep the previous chemical chart only as a temporary mixed-domain
                # identity.  ORACLE evidence and the reference geometry are new; the
                # next cycle retries a complete SMITH rebuild.
                recenter_sonic_definition_from_xyzin(
                    target,
                    fallback_definition,
                    preserve_primitive_source=False,
                )
    write_sycart_build_section(target)
    fallback_atoms = tuple(
        int(atom)
        for atom in getattr(failure, "atom_indices_zero_based", ())
    )
    return rebuilt, bool(rebuilt or fallback_definition is not None), fallback_atoms


def _local_sycart_model(path, coordinates, plan, regime) -> OptimizerCoordinateModel:
    from matrix_smith.mixed_domains import local_sycart_axis_labels

    if not plan.uncertain_atoms:
        raise ValueError("local SYCART needs a nonempty uncertain domain")
    if len(plan.uncertain_atoms) == plan.atom_count:
        return coordinate_model_from_xyzin(
            path,
            kind="sycart",
            task_regime=regime,
            matrix_free=(regime == "MINIMUM"),
        )
    directions = np.zeros((3 * len(plan.uncertain_atoms), 3 * plan.atom_count), dtype=float)
    for row, atom in enumerate(plan.uncertain_atoms):
        directions[3 * row + 0, 3 * atom + 0] = 1.0
        directions[3 * row + 1, 3 * atom + 1] = 1.0
        directions[3 * row + 2, 3 * atom + 2] = 1.0
    return OptimizerCoordinateModel(
        kind="sycart",
        labels=local_sycart_axis_labels(plan.uncertain_atoms),
        directions_angstrom=directions,
        metric_diagonal=np.full(directions.shape[0], ANGSTROM_TO_BOHR**2, dtype=float),
        retained_group="LOCAL_C1",
    )


def _stable_sonic_model(definition, plan) -> OptimizerCoordinateModel:
    count = len(plan.stable_sonic_identifiers)
    return OptimizerCoordinateModel(
        kind="sonic",
        labels=plan.stable_sonic_identifiers,
        directions_angstrom=np.empty((count, 0), dtype=float),
        metric_diagonal=np.ones(count, dtype=float),
        sonic_labels=plan.stable_sonic_identifiers,
        sonic_definition=definition,
        retained_group=str(definition.point_group),
    )


def _stage_audit(stage: SpatialHybridStage) -> dict[str, object]:
    return {
        "coordinate_kind": stage.coordinate_kind,
        "active_atoms_zero_based": list(stage.active_atoms),
        "fixed_atoms_zero_based": list(stage.fixed_atoms),
        "steps": len(stage.result.iterations),
        "converged_subproblem": stage.result.converged,
        "optimizer_wall_seconds": stage.result.optimizer_wall_seconds,
        "calculator_wall_seconds": stage.result.calculator_wall_seconds,
        "coordinate_wall_seconds": stage.result.coordinate_realization_wall_seconds,
    }


def _adaptive_sycart_block_length(base_steps: int, gradient_inf: float) -> int:
    """Use longer Cartesian blocks only after the gross distortion is removed."""

    base = max(1, int(base_steps))
    if float(gradient_inf) <= 0.05:
        return max(base, 10)
    return base


def _sonic_block_needs_cooldown(
    sycart_reduction_factor: float,
    sonic_reduction_factor: float,
) -> bool:
    """Return whether SONIC delivered less than half the preceding SYCART gain."""

    sonic_gain = max(0.0, 1.0 - float(sonic_reduction_factor))
    sycart_gain = max(0.0, 1.0 - float(sycart_reduction_factor))
    return float(sonic_reduction_factor) >= 0.98 or sonic_gain < 0.5 * sycart_gain


def _advance_soop_fallback_hysteresis(
    active_atoms: tuple[int, ...],
    successful_rebuilds: int,
    current_failure_atoms: tuple[int, ...],
    *,
    required_successes: int = 2,
) -> tuple[tuple[int, ...], int]:
    """Keep a failed SOOP fragment Cartesian until rebuilds are stably valid.

    A single successful rebuild can lie immediately across a numerical chart
    boundary. Requiring two consecutive certifications prevents a fragment
    from alternating SOOP/SYCART on adjacent accepted macroblocks.
    """

    required = int(required_successes)
    if required < 1:
        raise ValueError("SOOP fallback hysteresis needs at least one success")
    active = tuple(sorted({int(atom) for atom in active_atoms}))
    failed = tuple(sorted({int(atom) for atom in current_failure_atoms}))
    if failed:
        return tuple(sorted(set(active).union(failed))), 0
    if not active:
        return (), 0
    streak = int(successful_rebuilds) + 1
    if streak >= required:
        return (), streak
    return active, streak


__all__ = [
    "LINK_SPATIAL_HYBRID_SCHEMA",
    "SpatialHybridOptimizationResult",
    "SpatialHybridStage",
    "optimize_geometry_spatial_hybrid",
]
