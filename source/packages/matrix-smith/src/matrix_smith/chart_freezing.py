from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re

import numpy as np

from matrix_chem import read_enriched_xyz
from matrix_chem.topology.elements import atomic_number
from matrix_core import read_sectioned_lines

from .chart_atlas import SonicChartState, build_sonic_chart_atlas, classify_sonic_chart
from .contracts import GICForgeContractError
from .evaluation import evaluate_gic_values
from .fallback_ledger import build_fallback_ledger
from .gaussian_export import (
    _condensed_ring_pucker_keys,
    _ring_pucker_source_ring_keys_for_gic,
)
from .models import FrozenGIC, GICDefinition
from .periodic_estimates import RingPhaseCoordinate, build_periodic_coordinate_estimates
from .topology_io import _topology_rings
from .audits.transition_state import _normalized_gic_chart_condition
from .policy import PHASE_ZERO_TOLERANCE as _PHASE_ZERO_TOLERANCE

def _with_periodic_coordinate_estimates(
    definition: GICDefinition,
    path: Path,
    *,
    native_definition: GICDefinition,
) -> GICDefinition:
    lines = read_sectioned_lines(Path(path))
    geometry = read_enriched_xyz(Path(path))
    coordinate_values = _selected_gic_values(
        definition,
        families={"TORSION", "PSEUDO_CYCLE_TORSION", "IMPROPER_DIHEDRAL"},
    )
    estimates = build_periodic_coordinate_estimates(
        definition,
        lines,
        tuple(geometry.atoms),
        ring_phase_coordinates=_ring_puckering_phase_coordinates(
            native_definition,
            atom_symbols=tuple(geometry.atoms),
            topology_rings=tuple(
                ring for _index, ring in _topology_rings(lines, natoms=geometry.natoms)
            ),
        ),
        coordinate_values_radian=coordinate_values,
    )
    return replace(definition, periodic_coordinate_estimates=estimates)


def _ring_puckering_phase_coordinates(
    native_definition: GICDefinition,
    *,
    atom_symbols: tuple[str, ...],
    topology_rings: tuple[tuple[int, ...], ...] = (),
) -> tuple[RingPhaseCoordinate, ...]:
    primitive_by_id = {
        primitive.identifier: primitive for primitive in native_definition.primitives
    }
    values = _selected_gic_values(
        native_definition,
        families={"RING_PUCKER_COMPONENT"},
    )
    groups: dict[tuple[int, ...], list[FrozenGIC]] = {}
    for gic in native_definition.gics:
        if gic.family != "RING_PUCKER_COMPONENT":
            continue
        coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
        # Amplitude/phase reporting is defined for cyclic Fourier projections
        # of either local U sources or CHARM H sources. Independent triangular
        # flap angles are already periodic exponential coordinates and must
        # not be paired according to an arbitrary tree ordering.
        source_functions = {
            primitive_by_id[primitive_id].function
            for primitive_id, _coefficient in coefficients
            if primitive_id in primitive_by_id
        }
        if any(function not in {"U", "H"} for function in source_functions):
            continue
        ring_keys = _ring_pucker_source_ring_keys_for_gic(gic, primitive_by_id)
        if not ring_keys and source_functions == {"H"}:
            source_atoms = {
                atom
                for primitive_id, _coefficient in coefficients
                if primitive_id in primitive_by_id
                for atom in primitive_by_id[primitive_id].atoms
            }
            ring_keys = tuple(ring for ring in topology_rings if set(ring) == source_atoms)
        if ring_keys:
            groups.setdefault(ring_keys[0], []).append(gic)
    phases: list[RingPhaseCoordinate] = []
    phase_index = 0
    for ring_atoms, gics in groups.items():
        if len(gics) < 2:
            continue
        priority_atom = max(
            ring_atoms,
            key=lambda atom: (int(atomic_number(atom_symbols[atom - 1]) or 0), -atom),
        )
        first, second = gics[:2]
        first_value, second_value = values[first.identifier], values[second.identifier]
        radial = float(np.hypot(first_value, second_value))
        if radial <= _PHASE_ZERO_TOLERANCE:
            azimuth = 0.0
            azimuth_status = "GAUGE_FIXED_ZERO_RADIAL_AMPLITUDE"
        else:
            azimuth = float(np.arctan2(second_value, first_value))
            if abs(azimuth) <= _PHASE_ZERO_TOLERANCE:
                azimuth = 0.0
                azimuth_status = "DEFINED_CLAMPED_TO_ZERO"
            else:
                azimuth_status = "DEFINED"
        phase_index += 1
        phases.append(
            RingPhaseCoordinate(
                identifier=f"PhiP{phase_index:04d}",
                source_coordinates=(second.name, first.name),
                ring_atoms=ring_atoms,
                reference_value_radian=azimuth,
                reference_value_status=azimuth_status,
                coordinate_domain="PERIODIC_2PI",
                periodicity=max(1, len(ring_atoms) // np.gcd(len(ring_atoms), 2)),
                coordinate_definition=f"ATAN2({second.name},{first.name})",
                priority_atom=priority_atom,
            )
        )
        if len(gics) >= 3:
            third = gics[2]
            third_value = values[third.identifier]
            priority_position = ring_atoms.index(priority_atom)
            chair_sign = 1.0 if priority_position % 2 == 0 else -1.0
            oriented_third_value = chair_sign * third_value
            total = float(np.hypot(radial, third_value))
            if total <= _PHASE_ZERO_TOLERANCE:
                polar = 0.0
                polar_status = "GAUGE_FIXED_ZERO_TOTAL_AMPLITUDE"
            else:
                polar = float(np.arctan2(radial, oriented_third_value))
                if abs(polar) <= _PHASE_ZERO_TOLERANCE:
                    polar = 0.0
                    polar_status = "DEFINED_CLAMPED_TO_ZERO"
                else:
                    polar_status = "DEFINED"
            if polar > 0.5 * np.pi + _PHASE_ZERO_TOLERANCE:
                # A mirrored but otherwise equivalent embedded ring changes
                # hyperspherical gauge as (azimuth + pi, pi - polar).  Select
                # the priority-atom hemisphere so SMILES embeddings serialize
                # the same phase on x86_64 and ARM64.
                polar = float(np.pi - polar)
                chair_sign = -chair_sign
                # The signed observable definitions must follow the same
                # frozen inversion as their reported reference values.
                if radial <= _PHASE_ZERO_TOLERANCE:
                    azimuth = 0.0
                    azimuth_status = "GAUGE_FIXED_ZERO_RADIAL_AMPLITUDE"
                else:
                    azimuth = float(np.arctan2(-second_value, -first_value))
                    azimuth_status = "DEFINED_CANONICAL_HEMISPHERE"
                    if abs(azimuth) <= _PHASE_ZERO_TOLERANCE:
                        azimuth = 0.0
                phases[-1] = replace(
                    phases[-1],
                    reference_value_radian=azimuth,
                    reference_value_status=azimuth_status,
                    coordinate_definition=f"ATAN2(-{second.name},-{first.name})",
                )
                if abs(polar) <= _PHASE_ZERO_TOLERANCE:
                    polar = 0.0
                    polar_status = "DEFINED_CLAMPED_TO_ZERO"
                else:
                    polar_status = "DEFINED_CANONICAL_HEMISPHERE"
            phase_index += 1
            phases.append(
                RingPhaseCoordinate(
                    identifier=f"PhiP{phase_index:04d}",
                    source_coordinates=(first.name, second.name, third.name),
                    ring_atoms=ring_atoms,
                    reference_value_radian=polar,
                    reference_value_status=polar_status,
                    coordinate_domain="BOUNDED_0_PI",
                    periodicity=2,
                    coordinate_definition=(
                        f"ATAN2(SQRT({first.name}^2+{second.name}^2),"
                        f"{'-' if chair_sign < 0.0 else ''}{third.name})"
                    ),
                    priority_atom=priority_atom,
                )
            )
    return tuple(phases)


def _selected_gic_values(
    definition: GICDefinition,
    *,
    families: set[str],
) -> dict[str, float]:
    """Evaluate only requested GICs, avoiding unrelated singular coordinates."""

    values: dict[str, float] = {}
    for gic in definition.gics:
        if gic.family not in families:
            continue
        try:
            value = float(evaluate_gic_values(replace(definition, gics=(gic,)))[0])
        except FloatingPointError:
            continue
        values[gic.identifier] = value
        values[gic.name] = value
    return values


def _attach_chart_atlas(definition: GICDefinition) -> GICDefinition:
    """Attach deterministic chart-boundary diagnostics to a final definition."""

    controlled_prefixes = (
        "ONIC_CORE ",
        "CHART_ORIENTATION ",
        "CHART_ROLE ",
        "CHART_ATLAS STATUS=",
        "RING_PHASE_CHART ",
    )
    definition = replace(
        definition,
        semantic_diagnostics=(
            *(
                record
                for record in definition.semantic_diagnostics
                if not record.startswith(controlled_prefixes)
            ),
            "ONIC_CORE COMMON_TYPED_NONREDUNDANT_ALGEBRA",
            "CHART_ORIENTATION SONIC",
            "CHART_ROLE EXPLOITATION",
        ),
    )
    definition = _attach_ring_phase_charts(definition)
    definition = _apply_minimum_periodic_continuations(definition)
    _validate_periodic_continuations(definition)

    try:
        evaluate_gic_values(definition)
        states = _operative_source_chart_states(definition)
    except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
        raise GICForgeContractError(
            f"SONIC chart atlas cannot evaluate the frozen chart: {type(exc).__name__}"
        ) from exc
    atlas = build_sonic_chart_atlas(states)
    transition_text = ",".join(atlas.transitions_required) or "NONE"
    if not atlas.safe:
        requested = ",".join(
            f"{state.identifier}->{state.recommended_chart}"
            for state in atlas.states
            if state.identifier in atlas.transitions_required
        )
        raise GICForgeContractError(
            "unresolved operative SONIC chart transition: " + requested
        )
    diagnostics = (
        *definition.semantic_diagnostics,
        f"CHART_ATLAS STATUS={'SAFE' if atlas.safe else 'BOUNDARY'} "
        f"TRANSITIONS={transition_text}",
    )
    charted = replace(
        definition,
        semantic_diagnostics=diagnostics,
    )
    fallback_events = build_fallback_ledger(charted)
    return replace(
        charted,
        fallback_events=fallback_events,
        fallback_diagnostics=tuple(event.source for event in fallback_events),
    )


def _selected_primitive_ids(definition: GICDefinition) -> set[str]:
    return {
        primitive_id
        for gic in definition.gics
        for primitive_id, _coefficient in (
            gic.coefficients or ((gic.primitive_id, 1.0),)
        )
    }


def _operative_source_chart_states(
    definition: GICDefinition,
) -> tuple[SonicChartState, ...]:
    """Classify chart-bearing source primitives, never arbitrary SALC values."""

    from .numerics import _primitive_value

    selected = _selected_primitive_ids(definition)
    scientific_path = _definition_scientific_path(definition)
    coords = np.asarray(definition.reference_coordinates_angstrom, dtype=float)
    states: list[SonicChartState] = []
    for primitive in definition.primitives:
        if primitive.identifier not in selected:
            continue
        if primitive.function == "A":
            value = float(_primitive_value(primitive, coords, reference_coords=coords))
            states.append(classify_sonic_chart(primitive.identifier, "BEND", value))
        elif primitive.function == "D":
            value = float(_primitive_value(primitive, coords, reference_coords=coords))
            state = classify_sonic_chart(
                primitive.identifier,
                "TORSION",
                value,
                periodicity=1.0,
            )
            if primitive.chart == "PERIODIC_CONTINUATION":
                state = replace(
                    state,
                    chart="PERIODIC_CONTINUATION",
                    status="REGULAR",
                    distance_to_boundary=float("inf"),
                    recommended_chart="PERIODIC_CONTINUATION",
                    reason="REFERENCE_CENTERED_CONTINUATION_REALIZED",
                )
            elif scientific_path in {"TRANSITION_STATE", "EXPLORATION"}:
                chart = (
                    "TS_FROZEN_PRINCIPAL"
                    if scientific_path == "TRANSITION_STATE"
                    else "EXPLORATION_FROZEN_PRINCIPAL"
                )
                state = replace(
                    state,
                    chart=chart,
                    status="REGULAR",
                    distance_to_boundary=float("inf"),
                    recommended_chart=chart,
                    reason=f"{scientific_path}_TASK_POLICY_PRESERVES_FROZEN_TORSION",
                )
            states.append(state)
    return tuple(states)


def _attach_ring_phase_charts(definition: GICDefinition) -> GICDefinition:
    """Freeze every operative ring chart before Gaussian serialization."""

    ring_gics = tuple(
        gic for gic in definition.gics if gic.family == "RING_PUCKER_COMPONENT"
    )
    if not ring_gics:
        return definition

    primitive_by_id = {primitive.identifier: primitive for primitive in definition.primitives}
    condensed_ring_keys = _condensed_ring_pucker_keys(definition, primitive_by_id)
    groups: dict[tuple[tuple[int, ...], str], list[FrozenGIC]] = {}
    for gic in ring_gics:
        ring_keys = _ring_pucker_source_ring_keys_for_gic(gic, primitive_by_id)
        if ring_keys:
            irrep = gic.irrep if definition.symmetrize else "UNSYMMETRIZED"
            groups.setdefault((ring_keys[0], irrep), []).append(gic)
    if not groups:
        return definition
    values = _selected_gic_values(definition, families={"RING_PUCKER_COMPONENT"})
    records: list[str] = []
    aromatic_ring_keys: set[tuple[int, ...]] = set()
    for diagnostic in definition.ring_puckering_diagnostics:
        if " MODEL=AROMATIC_LOCAL_OUT_OF_PLANE " not in diagnostic:
            continue
        match = re.search(r"\bATOMS=([^ ]+)", diagnostic)
        if match is not None:
            aromatic_ring_keys.add(
                tuple(sorted(int(atom) for atom in match.group(1).split(",") if atom))
            )
    ring_amplitudes = {
        ring_atoms: float(
            np.linalg.norm(
                [
                    values[gic.identifier]
                    for (candidate_ring, _irrep), candidate_gics in groups.items()
                    if candidate_ring == ring_atoms
                    for gic in candidate_gics
                ]
            )
        )
        for ring_atoms, _irrep in groups
    }
    for (ring_atoms, irrep), gics in groups.items():
        source_functions = {
            primitive_by_id[primitive_id].function
            for gic in gics
            for primitive_id, _coefficient in (
                gic.coefficients or ((gic.primitive_id, 1.0),)
            )
            if primitive_id in primitive_by_id
        }
        component_names = tuple(gic.name for gic in gics)
        ring = ",".join(str(atom) for atom in ring_atoms)
        common = f"RING={ring} IRREP={irrep}"
        if source_functions == {"U"} and tuple(sorted(ring_atoms)) in aromatic_ring_keys:
            records.append(
                f"RING_PHASE_CHART {common} COMPONENTS={','.join(component_names)} "
                "CHART=RING_CARTESIAN_COMPONENTS REASON=AROMATIC_LOCAL_OUT_OF_PLANE_U"
            )
            continue
        if ring_atoms in condensed_ring_keys:
            records.append(
                f"RING_PHASE_CHART {common} COMPONENTS={','.join(component_names)} "
                "CHART=RING_CARTESIAN_COMPONENTS REASON=CONDENSED_RING"
            )
            continue
        if ring_amplitudes[ring_atoms] <= _PHASE_ZERO_TOLERANCE:
            records.append(
                f"RING_PHASE_CHART {common} COMPONENTS={','.join(component_names)} "
                "CHART=RING_CARTESIAN_COMPONENTS REASON=ZERO_RING_AMPLITUDE "
                f"AMPLITUDE={ring_amplitudes[ring_atoms]:.17g} "
                f"TOLERANCE={_PHASE_ZERO_TOLERANCE:.17g}"
            )
            continue
        if len(component_names) == 1 and len(ring_atoms) == 4:
            records.append(
                f"RING_PHASE_CHART {common} COMPONENTS={component_names[0]} "
                "CHART=RING_POLAR MODE=SINGLE_COMPONENT"
            )
            continue
        for index in range(0, len(gics) - 1, 2):
            left, right = gics[index : index + 2]
            pair_amplitude = float(
                np.hypot(values[left.identifier], values[right.identifier])
            )
            mode = "ZERO_GAUGE" if pair_amplitude <= _PHASE_ZERO_TOLERANCE else "ATAN2"
            records.append(
                f"RING_PHASE_CHART {common} COMPONENTS={left.name},{right.name} "
                f"CHART=RING_POLAR MODE={mode} AMPLITUDE={pair_amplitude:.17g} "
                f"TOLERANCE={_PHASE_ZERO_TOLERANCE:.17g}"
            )
    if not records:
        return definition
    return replace(
        definition,
        semantic_diagnostics=(*definition.semantic_diagnostics, *records),
    )


def _source_periodic_branch_states(definition: GICDefinition) -> tuple[SonicChartState, ...]:
    """Inspect selected source torsions, before any SALC can hide a branch seam."""

    scientific_path = _definition_scientific_path(definition)
    if scientific_path != "MINIMUM":
        return ()
    selected_source_ids = _selected_primitive_ids(definition)
    coords = np.asarray(definition.reference_coordinates_angstrom, dtype=float)
    from .numerics import _primitive_value

    states: list[SonicChartState] = []
    for primitive in definition.primitives:
        if (
            primitive.identifier not in selected_source_ids
            or primitive.function != "D"
            or primitive.chart == "PERIODIC_CONTINUATION"
        ):
            continue
        value = float(_primitive_value(primitive, coords, reference_coords=coords))
        state = classify_sonic_chart(
            primitive.identifier,
            "TORSION",
            value,
            periodicity=1.0,
        )
        if state.recommended_chart == "PERIODIC_CONTINUATION":
            states.append(state)
    return tuple(states)


def _apply_minimum_periodic_continuations(definition: GICDefinition) -> GICDefinition:
    """Freeze reference-centered local torsion charts selected by SMITH."""

    states = _source_periodic_branch_states(definition)
    if not states:
        return definition
    reference_by_id = {state.identifier: float(state.value) for state in states}
    primitives = tuple(
        replace(
            primitive,
            chart="PERIODIC_CONTINUATION",
            chart_reference_radian=reference_by_id[primitive.identifier],
        )
        if primitive.identifier in reference_by_id
        else primitive
        for primitive in definition.primitives
    )
    records = tuple(
        f"SOURCE_CHART {state.identifier} PERIODIC_CONTINUATION "
        f"REFERENCE_RADIAN={state.value:.17g} "
        f"BRANCH_DISTANCE={state.distance_to_boundary:.17g}"
        for state in states
    )
    transformed = replace(
        definition,
        primitives=primitives,
        semantic_diagnostics=(*definition.semantic_diagnostics, *records),
    )
    from .evaluation import build_gic_b_matrix

    before = np.asarray(build_gic_b_matrix(definition).rows, dtype=float)
    after = np.asarray(build_gic_b_matrix(transformed).rows, dtype=float)
    if before.shape != after.shape or not np.array_equal(before, after):
        raise GICForgeContractError(
            "periodic continuation changed the frozen Wilson tangent matrix"
        )
    condition = _normalized_gic_chart_condition(transformed)
    if not np.isfinite(condition):
        raise GICForgeContractError(
            "periodic continuation does not preserve a finite exact-rank SONIC chart"
        )
    return replace(
        transformed,
        semantic_diagnostics=(
            *transformed.semantic_diagnostics,
            f"SOURCE_CHART_GATE RANK={transformed.rank}/{transformed.target_rank} "
            f"NORMALIZED_CONDITION={condition:.17g} B_MATRIX=IDENTICAL",
        ),
    )


def _definition_scientific_path(definition: GICDefinition) -> str:
    return next(
        (
            record.split(maxsplit=1)[1].strip().upper()
            for record in definition.semantic_diagnostics
            if record.upper().startswith("SCIENTIFIC_PATH ")
        ),
        "",
    )


def _validate_periodic_continuations(definition: GICDefinition) -> None:
    continuations = tuple(
        primitive
        for primitive in definition.primitives
        if primitive.chart == "PERIODIC_CONTINUATION"
    )
    if not continuations:
        return
    if _definition_scientific_path(definition) != "MINIMUM":
        raise GICForgeContractError(
            "periodic source continuation is permitted only on the MINIMUM path"
        )
    if definition.rank != definition.target_rank:
        raise GICForgeContractError(
            "periodic source continuation requires an exact-rank frozen chart"
        )
    selected_source_ids = _selected_primitive_ids(definition)
    coords = np.asarray(definition.reference_coordinates_angstrom, dtype=float)
    from .numerics import _primitive_value

    for primitive in continuations:
        reference = primitive.chart_reference_radian
        if (
            primitive.function != "D"
            or primitive.identifier not in selected_source_ids
            or reference is None
            or not np.isfinite(float(reference))
        ):
            raise GICForgeContractError(
                f"invalid periodic continuation contract for {primitive.identifier}"
            )
        source_value = float(_primitive_value(primitive, coords, reference_coords=coords))
        delta = (source_value - float(reference) + np.pi) % (2.0 * np.pi) - np.pi
        state = classify_sonic_chart(
            primitive.identifier,
            "TORSION",
            source_value,
            periodicity=1.0,
        )
        if abs(delta) > 1.0e-12 or state.recommended_chart != "PERIODIC_CONTINUATION":
            raise GICForgeContractError(
                f"stale periodic continuation reference for {primitive.identifier}"
            )
    condition = _normalized_gic_chart_condition(definition)
    if not np.isfinite(condition):
        raise GICForgeContractError(
            "periodic continuation failed the exact-rank normalized-condition gate"
        )
