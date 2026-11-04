from __future__ import annotations

from dataclasses import replace
import re

import numpy as np
from scipy.linalg import qr

from matrix_chem.coordinate_atlas_contract import (
    OracleCoordinateAtlasContract,
    validate_oracle_coordinate_atlas_contract,
)

from .contracts import GICForgeContractError
from .audits.dense import (
    dense_gic_chart_condition,
    dense_gic_tangent_diagnostics,
    dense_primitive_chart_condition,
)
from .evaluation import build_gic_b_matrix
from .models import (
    FrozenGIC,
    GICDefinition,
    GICPointGroupOperation,
    GICPrimitive,
    GICReductionDiagnostics,
)
from .numerics import _analytic_b_row
from .policy import RANK_TOLERANCE
from .symmetrization import _select_ranked_primitives_with_diagnostics
from .policy import FRAGMENT_CONTEXT_TRANSITION_STATE as _FRAGMENT_CONTEXT_TRANSITION_STATE

def _wilson_tangent_diagnostics(
    definition: GICDefinition, coords: np.ndarray
) -> dict[str, float | int]:
    """Record the reusable GIC Wilson tangent diagnostics for LINK."""
    return dense_gic_tangent_diagnostics(definition, coords)


def _normalized_primitive_chart_condition(
    primitives: tuple[GICPrimitive, ...],
    coords: np.ndarray,
    *,
    rank_tolerance: float,
) -> float:
    """Return the row-normalized condition number of a selected primitive chart."""

    return dense_primitive_chart_condition(
        primitives, coords, rank_tolerance=rank_tolerance
    )


def _normalized_gic_chart_condition(definition: GICDefinition) -> float:
    """Return the row-normalized condition of one final SONIC chart."""

    return dense_gic_chart_condition(definition, rank_tolerance=RANK_TOLERANCE)


def _ts_radial_salc_is_materially_better(
    direct_condition: float,
    salc_condition: float,
    *,
    prefer_locality: bool,
    minimum_relative_gain: float,
) -> tuple[bool, float, str]:
    """Prefer a local reaction kernel unless radial mixing is materially better.

    A delocalized radial SALC is accepted only when the direct chart exceeds
    the atlas condition trigger and the normalized global condition improves
    by at least the atlas relative-gain threshold.  Otherwise the direct
    ORACLE reaction-distance coordinates retain the local TS meaning.
    """

    direct = float(direct_condition)
    salc = float(salc_condition)
    if not np.isfinite(salc) or salc + 1.0e-12 >= direct:
        return False, 0.0, "LOCALITY"
    relative_gain = (
        float((direct - salc) / direct)
        if np.isfinite(direct) and direct > 0.0
        else float("inf")
    )
    if not prefer_locality:
        return False, relative_gain, "CONDITION_BELOW_TRIGGER"
    if relative_gain + 1.0e-12 >= float(minimum_relative_gain):
        return True, relative_gain, "MATERIAL_RELATIVE_GAIN"
    return False, relative_gain, "LOCALITY"


def _ts_radial_locality_preference_is_active(
    direct_condition: float,
    *,
    condition_trigger: float,
) -> bool:
    """Apply the TS radial locality tie-break only above the condition trigger."""

    return float(direct_condition) > float(condition_trigger)


def _gic_source_families(
    gic: FrozenGIC,
    *,
    primitive_by_id: dict[str, GICPrimitive],
) -> frozenset[str] | None:
    families: set[str] = set()
    for primitive_id, _coefficient in (
        gic.coefficients or ((gic.primitive_id, 1.0),)
    ):
        primitive = primitive_by_id.get(primitive_id)
        if primitive is None:
            return None
        families.add(primitive.family)
    return frozenset(families)


def _b_orthogonal_salc_block_candidate(
    definition: GICDefinition,
    *,
    block_families: tuple[str, ...],
) -> GICDefinition | None:
    """B-orthogonalize one declared macrofamily, separately in every irrep."""

    primitive_by_id = {
        primitive.identifier: primitive for primitive in definition.primitives
    }
    family_set = frozenset(block_families)
    family_order = {family: index for index, family in enumerate(block_families)}
    groups: dict[str, list[int]] = {}
    for index, gic in enumerate(definition.gics):
        families = _gic_source_families(gic, primitive_by_id=primitive_by_id)
        if families and families.issubset(family_set):
            groups.setdefault(gic.irrep, []).append(index)
    if not any(len(indices) > 1 for indices in groups.values()):
        return None

    source_rows = np.asarray(build_gic_b_matrix(definition).rows, dtype=float)
    trial_gics = list(definition.gics)
    changed = False
    for _irrep, source_indices in sorted(groups.items()):
        indices = sorted(
            source_indices,
            key=lambda index: (
                family_order.get(definition.gics[index].family, len(family_order)),
                index,
            ),
        )
        if len(indices) <= 1:
            continue
        primitive_ids = tuple(
            dict.fromkeys(
                primitive_id
                for index in indices
                for primitive_id, _coefficient in (
                    definition.gics[index].coefficients
                    or ((definition.gics[index].primitive_id, 1.0),)
                )
            )
        )
        primitive_index = {
            primitive_id: index for index, primitive_id in enumerate(primitive_ids)
        }
        source_coefficients = np.zeros((len(indices), len(primitive_ids)), dtype=float)
        for row, index in enumerate(indices):
            for primitive_id, coefficient in (
                definition.gics[index].coefficients
                or ((definition.gics[index].primitive_id, 1.0),)
            ):
                source_coefficients[row, primitive_index[primitive_id]] += float(
                    coefficient
                )

        b_basis: list[np.ndarray] = []
        transform_basis: list[np.ndarray] = []
        for row, index in enumerate(indices):
            b_row = np.array(source_rows[index], dtype=float, copy=True)
            transform = np.eye(len(indices), dtype=float)[row]
            for _pass in range(2):
                for prior_b, prior_transform in zip(
                    b_basis, transform_basis, strict=True
                ):
                    projection = float(np.dot(b_row, prior_b))
                    b_row -= projection * prior_b
                    transform -= projection * prior_transform
            norm = float(np.linalg.norm(b_row))
            if not np.isfinite(norm) or norm <= RANK_TOLERANCE:
                return None
            b_basis.append(b_row / norm)
            transform_basis.append(transform / norm)

            coefficients = (transform / norm) @ source_coefficients
            coefficient_norm = float(np.linalg.norm(coefficients))
            if not np.isfinite(coefficient_norm) or coefficient_norm <= RANK_TOLERANCE:
                return None
            coefficients /= coefficient_norm
            significant = np.flatnonzero(np.abs(coefficients) > 1.0e-12)
            if significant.size and coefficients[int(significant[0])] < 0.0:
                coefficients = -coefficients
            frozen_coefficients = tuple(
                (primitive_id, float(coefficients[primitive_index[primitive_id]]))
                for primitive_id in primitive_ids
                if abs(float(coefficients[primitive_index[primitive_id]])) > 1.0e-12
            )
            trial_gics[index] = replace(
                definition.gics[index],
                primitive_id=frozen_coefficients[0][0],
                gaussian_expression="LINEAR_COMBINATION",
                coefficients=frozen_coefficients,
            )
        changed = True

    if not changed:
        return None
    candidate = replace(definition, gics=tuple(trial_gics))
    if not np.isfinite(_normalized_gic_chart_condition(candidate)):
        return None
    return candidate


def _condition_transition_state_salc_blocks(
    baseline: GICDefinition,
    *,
    coordinate_atlas_contract: OracleCoordinateAtlasContract,
    excluded_families: frozenset[str] = frozenset(),
) -> GICDefinition:
    """Apply deterministic material-gain refinement within atlas blocks."""

    blocks = tuple(
        block
        for block in coordinate_atlas_contract.family_compatibility
        if block.substitutions
        and block.condition_trigger is not None
        and block.minimum_relative_gain is not None
        and not excluded_families.intersection(block.families)
    )
    if not blocks:
        return baseline
    policies = {
        (float(block.condition_trigger), float(block.minimum_relative_gain))
        for block in blocks
    }
    if len(policies) != 1:
        raise GICForgeContractError(
            "TS conditioning blocks must share one condition/locality policy"
        )
    condition_trigger, minimum_relative_gain = next(iter(policies))
    initial_condition = _normalized_gic_chart_condition(baseline)
    current = baseline
    current_condition = initial_condition
    remaining = {block.block_id: block for block in blocks}
    diagnostics: list[str] = []

    while remaining and current_condition > condition_trigger:
        alternatives: list[tuple[float, str, float, GICDefinition]] = []
        for block_id, block in sorted(remaining.items()):
            candidate = _b_orthogonal_salc_block_candidate(
                current,
                block_families=block.families,
            )
            if candidate is None:
                continue
            candidate_condition = _normalized_gic_chart_condition(candidate)
            relative_gain = (current_condition - candidate_condition) / current_condition
            if (
                candidate_condition < current_condition
                and relative_gain + 1.0e-12 >= minimum_relative_gain
            ):
                alternatives.append(
                    (candidate_condition, block_id, relative_gain, candidate)
                )
        if not alternatives:
            break
        candidate_condition, block_id, relative_gain, candidate = min(
            alternatives,
            key=lambda item: (item[0], item[1]),
        )
        diagnostics.append(
            "TS_SALC_BLOCK_CONDITIONING "
            f"BLOCK={block_id} INITIAL={current_condition:.12g} "
            f"FINAL={candidate_condition:.12g} RELATIVE_GAIN={relative_gain:.12g} "
            "SELECTED=CONDITIONED REASON=MATERIAL_RELATIVE_GAIN"
        )
        current = candidate
        current_condition = candidate_condition
        remaining.pop(block_id)

    diagnostics.append(
        "TS_CHART_CONDITIONING "
        f"INITIAL={initial_condition:.12g} FINAL={current_condition:.12g} "
        f"CONDITION_TRIGGER={condition_trigger:.12g} "
        f"MINIMUM_RELATIVE_GAIN={minimum_relative_gain:.12g} "
        f"SELECTED={'CONDITIONED' if current is not baseline else 'LOCAL'}"
    )
    return replace(
        current,
        semantic_diagnostics=(*current.semantic_diagnostics, *diagnostics),
    )


def _condition_transition_state_salc_alternatives(
    definition: GICDefinition,
    *,
    context: str,
    atom_symbols: tuple[str, ...],
    symmetry_operations: tuple[GICPointGroupOperation, ...],
    coordinate_atlas_contract: OracleCoordinateAtlasContract,
    separate_exocyclic_torsions: bool = False,
) -> GICDefinition:
    """Build analytic SALCs and apply the atlas-declared TS conditioning pass."""

    # Resolve the high-level builder only when conditioning is requested;
    # importing the independent audit/phase services must not load it cyclically.
    from .definition import symmetrize_gic_definition

    baseline = symmetrize_gic_definition(
        definition,
        atom_symbols=atom_symbols,
        symmetry_operations=symmetry_operations,
    )
    separate_native_torsions = bool(
        separate_exocyclic_torsions
        and (
            context != _FRAGMENT_CONTEXT_TRANSITION_STATE
            or baseline.point_group.strip().upper() in {"C1", "UNKNOWN"}
        )
    )
    if separate_native_torsions:
        from .definition import _retain_individual_exocyclic_torsions

        baseline = _retain_individual_exocyclic_torsions(baseline, definition)
    elif separate_exocyclic_torsions:
        # A native one-dihedral coordinate is generally exchanged with other
        # torsions by a non-trivial molecular symmetry operation.  Restoring
        # those rows after projection would therefore introduce UNASSIGNED
        # coordinates and make the retained-group TS chart inconsistent.
        # Keep the symmetry-adapted torsional SALCs here; individual scan
        # torsions remain the correct contract for exploration and C1 TSs.
        baseline = replace(
            baseline,
            semantic_diagnostics=(
                *baseline.semantic_diagnostics,
                "TS_EXOCYCLIC_TORSIONS SYMMETRY_ADAPTED "
                f"POINT_GROUP={baseline.point_group}",
            ),
        )
    if context != _FRAGMENT_CONTEXT_TRANSITION_STATE:
        return baseline
    validate_oracle_coordinate_atlas_contract(coordinate_atlas_contract)
    return _condition_transition_state_salc_blocks(
        baseline,
        coordinate_atlas_contract=coordinate_atlas_contract,
        excluded_families=(
            frozenset({"TORSION"}) if separate_native_torsions else frozenset()
        ),
    )


def _materially_better_conditioned_exact_chart(
    candidates: tuple[GICPrimitive, ...],
    selected: tuple[GICPrimitive, ...],
    diagnostics: GICReductionDiagnostics,
    coords: np.ndarray,
    *,
    target_rank: int,
    rank_tolerance: float,
    condition_trigger: float,
    minimum_relative_gain: float,
) -> tuple[tuple[GICPrimitive, ...], GICReductionDiagnostics]:
    """Refine an exact TS chart only for a material condition-number gain."""

    current_condition = _normalized_primitive_chart_condition(
        selected,
        coords,
        rank_tolerance=rank_tolerance,
    )
    if current_condition <= condition_trigger:
        decision = (
            "TS_EXACT_CHART_CONDITIONING:SELECTED=LOCAL:"
            f"K_INITIAL={current_condition:.12g}:K_FINAL={current_condition:.12g}:"
            "RELATIVE_GAIN=0:REASON=CONDITION_BELOW_TRIGGER"
        )
        return selected, replace(
            diagnostics,
            conditioning_decisions=(*diagnostics.conditioning_decisions, decision),
        )

    greedy, greedy_rank, greedy_diagnostics = (
        _select_ranked_primitives_with_diagnostics(
            candidates,
            coords,
            target_rank=target_rank,
            rank_tolerance=rank_tolerance,
            condition_ordinary=True,
        )
    )
    alternatives: list[
        tuple[float, str, tuple[GICPrimitive, ...], GICReductionDiagnostics]
    ] = []
    if greedy_rank == target_rank:
        alternatives.append(
            (
                _normalized_primitive_chart_condition(
                    greedy,
                    coords,
                    rank_tolerance=rank_tolerance,
                ),
                "GREEDY_CONDITIONED",
                greedy,
                greedy_diagnostics,
            )
        )
    pivoted = _pivoted_exact_rank_primitive_chart(
        candidates,
        coords,
        target_rank=target_rank,
        rank_tolerance=rank_tolerance,
    )
    if pivoted is not None:
        alternatives.append(
            (
                _normalized_primitive_chart_condition(
                    pivoted,
                    coords,
                    rank_tolerance=rank_tolerance,
                ),
                "PIVOTED_QR",
                pivoted,
                diagnostics,
            )
        )
    if not alternatives:
        decision = (
            "TS_EXACT_CHART_CONDITIONING:SELECTED=LOCAL:"
            f"K_INITIAL={current_condition:.12g}:K_FINAL={current_condition:.12g}:"
            "RELATIVE_GAIN=0:REASON=ALTERNATIVE_NOT_EXACT_RANK"
        )
        return selected, replace(
            diagnostics,
            conditioning_decisions=(*diagnostics.conditioning_decisions, decision),
        )
    (
        alternative_condition,
        alternative_algorithm,
        alternative,
        alternative_diagnostics,
    ) = min(
        alternatives,
        key=lambda item: (item[0], item[1]),
    )
    relative_gain = (
        (current_condition - alternative_condition) / current_condition
        if np.isfinite(current_condition) and current_condition > 0.0
        else float("inf")
    )
    accept = (
        alternative_condition < current_condition
        and relative_gain + 1.0e-12 >= minimum_relative_gain
    )
    selected_label = "CONDITIONED" if accept else "LOCAL"
    reason = "MATERIAL_RELATIVE_GAIN" if accept else "LOCALITY"
    final_condition = alternative_condition if accept else current_condition
    decision = (
        f"TS_EXACT_CHART_CONDITIONING:SELECTED={selected_label}:"
        f"K_INITIAL={current_condition:.12g}:K_FINAL={final_condition:.12g}:"
        f"RELATIVE_GAIN={relative_gain:.12g}:REASON={reason}:"
        f"ALGORITHM={alternative_algorithm}"
    )
    if not accept:
        return selected, replace(
            diagnostics,
            conditioning_decisions=(*diagnostics.conditioning_decisions, decision),
        )
    return alternative, replace(
        alternative_diagnostics,
        conditioning_decisions=(
            *alternative_diagnostics.conditioning_decisions,
            decision,
        ),
    )


def _pivoted_exact_rank_primitive_chart(
    candidates: tuple[GICPrimitive, ...],
    coords: np.ndarray,
    *,
    target_rank: int,
    rank_tolerance: float,
) -> tuple[GICPrimitive, ...] | None:
    """Select a deterministic, conditioned exact-rank TS primitive subset.

    Atlas-declared reaction distances are retained first. Column-pivoted QR
    then selects the remaining Wilson rows by independent volume, avoiding a
    locally sensible but globally near-dependent chart. The operation is
    activated only downstream of the atlas condition trigger.
    """

    if target_rank < 1 or len(candidates) < target_rank:
        return None
    rows = np.vstack([_analytic_b_row(item, coords) for item in candidates])
    norms = np.linalg.norm(rows, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= rank_tolerance):
        return None
    normalized = rows / norms[:, None]
    required = tuple(
        index
        for index, primitive in enumerate(candidates)
        if primitive.family == "TS_REACTION_DISTANCE"
        or primitive.semantic_type == "REACTION_DISTANCE"
    )
    if len(required) > target_rank:
        return None
    selected = list(required)
    if required:
        required_rows = normalized[list(required)]
        _left, singular, right = np.linalg.svd(required_rows, full_matrices=False)
        required_rank = int(np.count_nonzero(singular > rank_tolerance))
        if required_rank != len(required):
            return None
        required_basis = right[:required_rank].T
    else:
        required_basis = np.zeros((normalized.shape[1], 0), dtype=float)
    remaining = tuple(index for index in range(len(candidates)) if index not in required)
    residual = normalized[list(remaining)]
    if required_basis.shape[1]:
        residual = residual - (residual @ required_basis) @ required_basis.T
    _q, diagonal, pivots = qr(residual.T, mode="economic", pivoting=True)
    diagonal_values = np.abs(np.diag(diagonal))
    needed = target_rank - len(selected)
    if diagonal_values.size < needed or np.any(
        diagonal_values[:needed] <= rank_tolerance
    ):
        return None
    selected.extend(remaining[int(position)] for position in pivots[:needed])
    selected.sort()
    trial = normalized[selected]
    singular = np.linalg.svd(trial, compute_uv=False)
    if singular.size < target_rank or singular[-1] <= rank_tolerance:
        return None
    return tuple(candidates[index] for index in selected)


def _select_conserved_reactive_exact_chart(
    candidates: tuple[GICPrimitive, ...],
    common_count: int,
    coords: np.ndarray,
    *,
    target_rank: int,
    rank_tolerance: float,
    diagnostics: GICReductionDiagnostics,
    coordinate_atlas_contract: OracleCoordinateAtlasContract,
) -> tuple[tuple[GICPrimitive, ...], GICReductionDiagnostics]:
    """Choose a local or pivoted complement while retaining conserved SONICs."""

    count = int(common_count)
    if count < 0 or count > len(candidates):
        raise ValueError("reactive connectivity common count is invalid")
    rows = np.vstack([_analytic_b_row(item, coords) for item in candidates])
    norms = np.linalg.norm(rows, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= rank_tolerance):
        raise GICForgeContractError(
            "reactive connectivity pool contains a singular primitive"
        )
    normalized = rows / norms[:, None]
    common_rank = int(
        np.linalg.matrix_rank(normalized[:count], tol=rank_tolerance)
    )
    conserved, retained_rank, _common_diagnostics = (
        _select_ranked_primitives_with_diagnostics(
            candidates[:count],
            coords,
            target_rank=common_rank,
            rank_tolerance=rank_tolerance,
        )
    )
    if retained_rank != common_rank:
        raise GICForgeContractError(
            "conserved connectivity SONIC block cannot retain its numerical rank"
        )
    index_by_id = {
        primitive.identifier: index for index, primitive in enumerate(candidates)
    }
    conserved_indices = tuple(index_by_id[item.identifier] for item in conserved)
    fixed = normalized[np.asarray(conserved_indices, dtype=int)]
    if fixed.size:
        _left, singular, right = np.linalg.svd(fixed, full_matrices=False)
        fixed_rank = int(np.count_nonzero(singular > rank_tolerance))
        basis = right[:fixed_rank].T
    else:
        basis = np.zeros((normalized.shape[1], 0), dtype=float)
    reactive_indices = tuple(
        index for index in range(count, len(candidates))
    )
    residual = normalized[np.asarray(reactive_indices, dtype=int)]
    if basis.shape[1]:
        residual = residual - (residual @ basis) @ basis.T
    needed = target_rank - len(conserved)
    if needed < 0 or len(reactive_indices) < needed:
        raise GICForgeContractError(
            "reactive connectivity complement cannot match the target rank"
        )
    _q_matrix, triangular, pivots = qr(
        residual.T, mode="economic", pivoting=True
    )
    diagonal = np.abs(np.diag(triangular))
    if diagonal.size < needed or np.any(diagonal[:needed] <= rank_tolerance):
        raise GICForgeContractError(
            "reactive connectivity complement is rank deficient"
        )
    reactive = tuple(
        candidates[reactive_indices[int(position)]] for position in pivots[:needed]
    )
    pivoted = conserved + reactive
    final = normalized[
        np.asarray([index_by_id[item.identifier] for item in pivoted], dtype=int)
    ]
    if int(np.linalg.matrix_rank(final, tol=rank_tolerance)) != target_rank:
        raise GICForgeContractError(
            "conserved/reactive pivoted chart is not exact rank"
        )
    local = tuple(
        candidates[index_by_id[identifier]] for identifier in diagnostics.selected
    )
    local_condition = _normalized_primitive_chart_condition(
        local,
        coords,
        rank_tolerance=rank_tolerance,
    )
    pivoted_condition = _normalized_primitive_chart_condition(
        pivoted,
        coords,
        rank_tolerance=rank_tolerance,
    )
    condition_trigger, minimum_relative_gain = _transition_state_chart_conditioning_policy(
        coordinate_atlas_contract
    )
    relative_gain = (
        (local_condition - pivoted_condition) / local_condition
        if np.isfinite(local_condition) and local_condition > 0.0
        else float("inf")
    )
    use_pivoted = bool(
        local_condition > condition_trigger
        and pivoted_condition < local_condition
        and relative_gain + 1.0e-12 >= minimum_relative_gain
    )
    selected = pivoted if use_pivoted else local
    decision = (
        "TS_REACTIVE_CONNECTIVITY_REDUCTION:"
        f"SELECTED={'CONSERVED_PLUS_PIVOTED_REACTIVE' if use_pivoted else 'LOCAL'}:"
        f"COMMON_POOL={count}:COMMON_RANK={len(conserved)}:"
        f"REACTIVE_POOL={len(reactive_indices)}:REACTIVE_RANK={needed}:"
        f"K_LOCAL={local_condition:.12g}:K_PIVOTED={pivoted_condition:.12g}:"
        f"RELATIVE_GAIN={relative_gain:.12g}"
    )
    return selected, replace(
        diagnostics,
        selected=tuple(item.identifier for item in selected),
        conditioning_decisions=(*diagnostics.conditioning_decisions, decision),
    )


def _transition_state_chart_conditioning_policy(
    coordinate_atlas_contract: OracleCoordinateAtlasContract,
) -> tuple[float, float]:
    """Read the common TS condition/locality gate from the ORACLE atlas."""

    validate_oracle_coordinate_atlas_contract(coordinate_atlas_contract)
    policies = {
        (float(block.condition_trigger), float(block.minimum_relative_gain))
        for block in coordinate_atlas_contract.family_compatibility
        if block.condition_trigger is not None
        and block.minimum_relative_gain is not None
    }
    if len(policies) != 1:
        raise GICForgeContractError(
            "the TS atlas must declare one common condition/locality policy"
        )
    return next(iter(policies))


def _select_rank_aware_chart(
    candidates: tuple[GICPrimitive, ...],
    coords: np.ndarray,
    *,
    target_rank: int,
    rank_tolerance: float,
    condition_ordinary: bool,
    transition_state: bool,
    coordinate_atlas_contract: OracleCoordinateAtlasContract,
    condition_pseudobond_support: bool = False,
    condition_transition_state_exact_chart: bool = True,
) -> tuple[tuple[GICPrimitive, ...], int, GICReductionDiagnostics]:
    """Select, complete, and condition one chart through a single rank gate.

    The exact-rank gate is shared by minimum and TS charts.  Conditioning is
    strictly downstream of that gate and is enabled only for an ORACLE
    reactive-TS prescription; minimum-like charts therefore retain their
    canonical selection policy.
    """

    selected, rank, diagnostics = _select_ranked_primitives_with_diagnostics(
        candidates,
        coords,
        target_rank=target_rank,
        rank_tolerance=rank_tolerance,
        condition_ordinary=condition_ordinary,
        condition_pseudobond_support=condition_pseudobond_support,
    )
    if transition_state and rank == target_rank and condition_transition_state_exact_chart:
        condition_trigger, minimum_relative_gain = (
            _transition_state_chart_conditioning_policy(coordinate_atlas_contract)
        )
        selected, diagnostics = _materially_better_conditioned_exact_chart(
            candidates,
            selected,
            diagnostics,
            coords,
            target_rank=target_rank,
            rank_tolerance=rank_tolerance,
            condition_trigger=condition_trigger,
            minimum_relative_gain=minimum_relative_gain,
        )
    return selected, rank, diagnostics




def _with_prescribed_distance_only_primitives(
    candidates: tuple[GICPrimitive, ...],
    pairs: tuple[tuple[int, int], ...],
) -> tuple[GICPrimitive, ...]:
    existing = {
        tuple(sorted(primitive.atoms))
        for primitive in candidates
        if primitive.function == "R" and len(primitive.atoms) == 2
    }
    missing = tuple(pair for pair in pairs if pair not in existing)
    if not missing:
        return candidates
    serials = [
        int(match.group(1))
        for primitive in candidates
        if (match := re.fullmatch(r"P([0-9]+)", primitive.identifier)) is not None
    ]
    serial = max(serials, default=len(candidates))
    stretch_count = sum(primitive.family == "STRETCH" for primitive in candidates)
    additions: list[GICPrimitive] = []
    for pair in missing:
        serial += 1
        stretch_count += 1
        additions.append(
            GICPrimitive(
                identifier=f"P{serial:03d}",
                name=f"Str{stretch_count:04d}",
                family="STRETCH",
                function="R",
                atoms=pair,
                provenance="ORACLE_TS_CONTRACT",
                semantic_id=f"TS_KERNEL_DISTANCE_{pair[0]}_{pair[1]}",
                semantic_type="DISTANCE_ONLY",
            )
        )
    insertion = next(
        (index for index, primitive in enumerate(candidates) if primitive.family != "STRETCH"),
        len(candidates),
    )
    return candidates[:insertion] + tuple(additions) + candidates[insertion:]


def _with_transition_state_reaction_distance_family(
    candidates: tuple[GICPrimitive, ...],
    pairs: tuple[tuple[int, int], ...],
) -> tuple[GICPrimitive, ...]:
    """Keep ORACLE's reaction kernel separate from ordinary valence stretches."""

    reaction_pairs = {tuple(sorted(pair)) for pair in pairs}
    if not reaction_pairs:
        return candidates
    count = 0
    output: list[GICPrimitive] = []
    materialized: set[tuple[int, int]] = set()
    for primitive in candidates:
        pair = (
            tuple(sorted(primitive.atoms))
            if primitive.function == "R" and len(primitive.atoms) == 2
            else ()
        )
        if pair not in reaction_pairs:
            output.append(primitive)
            continue
        count += 1
        materialized.add(pair)
        output.append(
            replace(
                primitive,
                name=f"TSRe{count:04d}",
                family="TS_REACTION_DISTANCE",
                provenance="ORACLE_TS_CONTRACT",
                semantic_id=f"TS_REACTION_DISTANCE_{pair[0]}_{pair[1]}",
                semantic_type="REACTION_DISTANCE",
            )
        )
    if materialized != reaction_pairs:
        missing = ",".join(
            f"{left}-{right}" for left, right in sorted(reaction_pairs - materialized)
        )
        raise GICForgeContractError(
            f"ORACLE transition-state reaction distances were not materialized: {missing}"
        )
    return tuple(output)
