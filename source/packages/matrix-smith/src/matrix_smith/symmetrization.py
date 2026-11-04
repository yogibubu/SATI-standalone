"""Rank reduction and symmetry adaptation for frozen SMITH coordinates."""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from itertools import combinations, permutations, product

import numpy as np

from matrix_chem import (
    CoordinateComponent,
    cartesian_operation_matrix,
    coordinate_selection_units,
)
from matrix_numerics import (
    normalized_matrix_condition,
    numerical_matrix_rank,
    select_rank_revealing_rows,
    singular_spectrum,
)

from .cartesian_blocks import _linear_computational_subgroup
from .contracts import GICForgeContractError
from .audits.dense import normalized_dense_rank
from .fallback_ledger import make_fallback_event
from .coordinate_registry import FRAGMENT_BODY_PAIR_FUNCTIONS
from .models import (
    FrozenGIC,
    GICDefinition,
    GICPointGroupOperation,
    GICPrimitive,
    GICReductionDiagnostics,
    GICSymmetrizationDiagnostics,
    GICSymmetrizedGroup,
)
from .numerics import (
    _analytic_b_row,
    _angle_component_terms_from_refs,
    _ring_pucker_terms_from_refs,
    _sparse_analytic_b_row,
)
from .policy import (
    LOCAL_SYMMETRIZATION_METHOD,
    POINT_GROUP_PROJECTOR_METHOD,
    PSEUDOBOND_TORSION_METRIC_SYMMETRIZATION_POLICY,
    PRIMITIVE_FAMILY_ORDER,
    PROJECTOR_SYMMETRIZATION_POLICY,
    RANK_METHOD,
    RANK_TOLERANCE,
    REDUCTION_POLICY,
    SPECIAL_REDUCTION_CLASS,
    SYMMETRIZATION_POLICY,
    SYMMETRY_OPERATION_NEAR_THRESHOLD_FRACTION,
    SYMMETRY_OPERATION_TOLERANCE_ANGSTROM,
    TS_REACTIVE_COORDINATE_CLASS,
    primitive_prefix,
    primitive_reduction_class,
    primitive_symmetry_block,
)
from .semantic import USER_PROVENANCE
from .symmetry_labels import (
    irrep_characters_for_operations,
    irrep_dimension,
    irrep_name_prefix,
    is_total_symmetric_irrep,
    non_total_irrep_sequence,
    total_symmetric_irrep,
)


_CARTESIAN_INTERFRAGMENT_PROJECTOR_TOLERANCE = 1.0e-7


def _partition_quota_rank_revealing_selection(
    candidates: list[tuple[object, ...]],
    *,
    quotas: dict[object, int],
) -> tuple[tuple[object, ...], ...] | None:
    """Select a full-rank Cartesian basis subject to exact partition quotas.

    Candidate Wilson B rows define a linear matroid, while capacities on the
    value in field zero define a partition matroid.  The partition may encode
    coordinate families or irreducible representations.  The augmenting-path
    intersection below therefore removes ordering dependence.  A same-part
    exchange pass subsequently improves numerical conditioning without
    changing the prescribed composition.
    """

    eligible = tuple(
        index
        for index, candidate in sorted(enumerate(candidates), key=lambda item: int(item[1][5]))
        if candidate[0] in quotas and quotas[candidate[0]] > 0
    )
    target_count = int(sum(quotas.values()))
    if target_count == 0:
        return ()
    if any(
        sum(candidates[index][0] == key for index in eligible) < quota
        for key, quota in quotas.items()
    ):
        return None

    def linear_independent(indices: set[int]) -> bool:
        if not indices:
            return True
        rows = np.vstack(
            [np.asarray(candidates[index][4], dtype=float) for index in sorted(indices)]
        )
        return numerical_matrix_rank(
            rows,
            absolute_tolerance=_CARTESIAN_INTERFRAGMENT_PROJECTOR_TOLERANCE,
        ) == len(indices)

    def partition_independent(indices: set[int]) -> bool:
        counts = {key: 0 for key in quotas}
        for index in indices:
            key = candidates[index][0]
            if key not in counts:
                return False
            counts[key] += 1
            if counts[key] > quotas[key]:
                return False
        return True

    selected: set[int] = set()
    while len(selected) < target_count:
        outside = tuple(index for index in eligible if index not in selected)
        sources = tuple(index for index in outside if linear_independent(selected | {index}))
        sinks = {index for index in outside if partition_independent(selected | {index})}
        predecessors: dict[int, int | None] = {index: None for index in sources}
        queue = deque(sources)
        terminal: int | None = None
        while queue and terminal is None:
            node = queue.popleft()
            if node not in selected:
                if node in sinks:
                    terminal = node
                    break
                for incumbent in sorted(selected):
                    if incumbent in predecessors:
                        continue
                    if partition_independent((selected - {incumbent}) | {node}):
                        predecessors[incumbent] = node
                        queue.append(incumbent)
            else:
                for challenger in outside:
                    if challenger in predecessors:
                        continue
                    if linear_independent((selected - {node}) | {challenger}):
                        predecessors[challenger] = node
                        queue.append(challenger)
        if terminal is None:
            break
        augmenting_path: list[int] = []
        cursor: int | None = terminal
        while cursor is not None:
            augmenting_path.append(cursor)
            cursor = predecessors[cursor]
        selected.symmetric_difference_update(augmenting_path)

    if len(selected) != target_count or not linear_independent(selected):
        return None
    if any(
        sum(candidates[index][0] == key for index in selected) != quota
        for key, quota in quotas.items()
    ):
        return None

    def conditioning_score(indices: set[int]) -> tuple[float, float]:
        rows = np.vstack(
            [np.asarray(candidates[index][4], dtype=float) for index in sorted(indices)]
        )
        spectrum = singular_spectrum(
            rows,
            absolute_tolerance=_CARTESIAN_INTERFRAGMENT_PROJECTOR_TOLERANCE,
        )
        singular_values = spectrum.singular_values
        if (
            len(singular_values) != len(indices)
            or spectrum.rank != len(indices)
        ):
            return (-np.inf, -np.inf)
        return (
            float(singular_values[-1]),
            float(np.sum(np.log(singular_values))),
        )

    current_score = conditioning_score(selected)
    while True:
        best_exchange: tuple[int, int] | None = None
        best_score = current_score
        for incumbent in sorted(selected):
            incumbent_key = candidates[incumbent][0]
            for challenger in eligible:
                if challenger in selected or candidates[challenger][0] != incumbent_key:
                    continue
                trial = (selected - {incumbent}) | {challenger}
                score = conditioning_score(trial)
                better_smallest = score[0] > best_score[0] + 1.0e-12
                tied_smallest = abs(score[0] - best_score[0]) <= 1.0e-12
                better_volume = score[1] > best_score[1] + 1.0e-12
                if better_smallest or (tied_smallest and better_volume):
                    best_exchange = (incumbent, challenger)
                    best_score = score
        if best_exchange is None:
            break
        selected.remove(best_exchange[0])
        selected.add(best_exchange[1])
        current_score = best_score

    return tuple(candidates[index] for index in sorted(selected))


def _rank_revealing_residual_primitive_order(
    candidates: tuple[GICPrimitive, ...],
    *,
    coords: np.ndarray,
    basis: list[np.ndarray],
    count: int,
    rank_tolerance: float,
) -> tuple[GICPrimitive, ...]:
    """Put the best-conditioned residual primitive basis first.

    The global Cartesian projector already uses the partition-matroid
    selector above to remove ordering dependence from redundant fragment-pose
    pools. Native (unsymmetrized) SONIC charts need the same operation when a
    chemically meaningful distance has already consumed part of that pose
    span. The residual rows retain their natural scale so the selector avoids
    directions that are nearly contained in the protected basis.
    """

    if count <= 0 or not candidates:
        return candidates
    records: list[tuple[GICPrimitive, np.ndarray]] = []
    for sequence, primitive in enumerate(candidates):
        normalized = _normalized_b_row_or_none(
            primitive,
            coords,
            rank_tolerance=rank_tolerance,
        )
        if normalized is None:
            continue
        residual = _b_row_residual_against_basis(normalized, basis)
        if float(np.linalg.norm(residual)) <= rank_tolerance:
            continue
        records.append((primitive, residual))
    if len(records) < count:
        return candidates
    chosen = select_rank_revealing_rows(
        np.vstack([residual for _primitive, residual in records]),
        target_rank=count,
        tolerance=rank_tolerance,
        tie_tolerance=1.0e-12,
    )
    if chosen.rank != count:
        return candidates
    selected = tuple(records[index][0] for index in chosen.indices)
    selected_ids = {primitive.identifier for primitive in selected}
    return selected + tuple(
        primitive for primitive in candidates if primitive.identifier not in selected_ids
    )


def _exact_residual_completion(
    order: tuple[GICPrimitive, ...],
    *,
    coords: np.ndarray,
    basis: list[np.ndarray],
    count: int,
    rank_tolerance: float,
) -> tuple[GICPrimitive, ...]:
    """Take the first exact-rank residual completion in a frozen order."""

    trial_basis = list(basis)
    chosen: list[GICPrimitive] = []
    for primitive in order:
        normalized = _normalized_b_row_or_none(
            primitive,
            coords,
            rank_tolerance=rank_tolerance,
        )
        if normalized is None:
            continue
        orthonormal = _orthonormal_residual_or_none(
            trial_basis,
            normalized,
            rank_tolerance=rank_tolerance,
        )
        if orthonormal is None:
            continue
        trial_basis.append(orthonormal)
        chosen.append(primitive)
        if len(chosen) == count:
            break
    return tuple(chosen)


def _normalized_primitive_chart_condition(
    primitives: tuple[GICPrimitive, ...],
    *,
    coords: np.ndarray,
    rank_tolerance: float,
) -> float:
    """Return the row-normalized condition, or infinity if rank is lost."""

    rows = []
    for primitive in primitives:
        normalized = _normalized_b_row_or_none(
            primitive,
            coords,
            rank_tolerance=rank_tolerance,
        )
        if normalized is None:
            return float("inf")
        rows.append(normalized)
    if not rows:
        return float("inf")
    return normalized_matrix_condition(
        np.vstack(rows),
        absolute_tolerance=rank_tolerance,
        required_rank=len(rows),
    )


def _condition_improving_exchange_completion(
    seed: tuple[GICPrimitive, ...],
    candidates: tuple[GICPrimitive, ...],
    *,
    coords: np.ndarray,
    selected_prefix: tuple[GICPrimitive, ...],
    rank_tolerance: float,
) -> tuple[GICPrimitive, ...]:
    """Apply condition-improving exchanges toward source-local support."""

    selected = list(seed)
    selected_ids = {primitive.identifier for primitive in selected}
    candidate_position = {
        primitive.identifier: position for position, primitive in enumerate(candidates)
    }
    current_condition = _normalized_primitive_chart_condition(
        (*selected_prefix, *selected),
        coords=coords,
        rank_tolerance=rank_tolerance,
    )
    while True:
        best_trial: list[GICPrimitive] | None = None
        best_condition = current_condition
        for position in range(len(selected)):
            incumbent = selected[position]
            for challenger in candidates:
                if challenger.identifier in selected_ids:
                    continue
                if (
                    candidate_position[challenger.identifier]
                    >= candidate_position[incumbent.identifier]
                    or challenger.family != incumbent.family
                    or challenger.reduction_class != incumbent.reduction_class
                ):
                    continue
                trial = list(selected)
                trial[position] = challenger
                condition = _normalized_primitive_chart_condition(
                    (*selected_prefix, *trial),
                    coords=coords,
                    rank_tolerance=rank_tolerance,
                )
                if condition < best_condition and not np.isclose(
                    condition,
                    best_condition,
                    rtol=1.0e-12,
                    atol=0.0,
                ):
                    best_trial = trial
                    best_condition = condition
        if best_trial is None:
            return tuple(selected)
        selected = best_trial
        selected_ids = {primitive.identifier for primitive in selected}
        current_condition = best_condition


def _conditioned_residual_primitive_order(
    candidates: tuple[GICPrimitive, ...],
    *,
    coords: np.ndarray,
    selected_prefix: tuple[GICPrimitive, ...],
    basis: list[np.ndarray],
    count: int,
    rank_tolerance: float,
) -> tuple[GICPrimitive, ...]:
    """Choose the best exact support completion, preserving local ties.

    The two alternatives span the same declared pseudobond-support block.  One
    follows its deterministic source-local order; the other is the common
    rank-revealing completion.  Same-block exchanges improve the full chart,
    not an isolated residual submatrix.  This is a numerical realization
    choice: no contact chemistry or task-regime threshold is introduced.
    """

    if count <= 0 or not candidates:
        return candidates

    local = _exact_residual_completion(
        candidates,
        coords=coords,
        basis=basis,
        count=count,
        rank_tolerance=rank_tolerance,
    )
    if len(local) != count:
        return candidates
    rank_revealing_order = _rank_revealing_residual_primitive_order(
        candidates,
        coords=coords,
        basis=basis,
        count=count,
        rank_tolerance=rank_tolerance,
    )
    rank_revealing = _exact_residual_completion(
        rank_revealing_order,
        coords=coords,
        basis=basis,
        count=count,
        rank_tolerance=rank_tolerance,
    )
    if len(rank_revealing) != count:
        return candidates
    local = _condition_improving_exchange_completion(
        local,
        candidates,
        coords=coords,
        selected_prefix=selected_prefix,
        rank_tolerance=rank_tolerance,
    )
    rank_revealing = _condition_improving_exchange_completion(
        rank_revealing,
        candidates,
        coords=coords,
        selected_prefix=selected_prefix,
        rank_tolerance=rank_tolerance,
    )
    local_condition = _normalized_primitive_chart_condition(
        (*selected_prefix, *local),
        coords=coords,
        rank_tolerance=rank_tolerance,
    )
    rank_revealing_condition = _normalized_primitive_chart_condition(
        (*selected_prefix, *rank_revealing),
        coords=coords,
        rank_tolerance=rank_tolerance,
    )
    choose_rank_revealing = (
        rank_revealing_condition < local_condition
        and not np.isclose(
            rank_revealing_condition,
            local_condition,
            rtol=1.0e-12,
            atol=0.0,
        )
    )
    chosen = rank_revealing if choose_rank_revealing else local
    chosen_ids = {primitive.identifier for primitive in chosen}
    return chosen + tuple(
        primitive for primitive in candidates if primitive.identifier not in chosen_ids
    )


def _family_conditioned_residual_primitive_order(
    candidates: tuple[GICPrimitive, ...],
    *,
    coords: np.ndarray,
    basis: list[np.ndarray],
    count: int,
    rank_tolerance: float,
) -> tuple[GICPrimitive, ...]:
    """Condition an exact ordinary completion without changing family quotas.

    First reproduce the established semantic ordering to determine how many
    rows each primitive family contributes.  Then reuse the partition-matroid
    selector to choose the best-conditioned independent representatives with
    exactly those quotas in the residual space left by protected analytic
    coordinates.
    """

    if count <= 0 or not candidates:
        return candidates
    provisional_basis = list(basis)
    quotas: dict[tuple[str, str], int] = {}
    provisional_count = 0
    for primitive in candidates:
        normalized = _normalized_b_row_or_none(
            primitive,
            coords,
            rank_tolerance=rank_tolerance,
        )
        if normalized is None:
            continue
        orthonormal = _orthonormal_residual_or_none(
            provisional_basis,
            normalized,
            rank_tolerance=rank_tolerance,
        )
        if orthonormal is None:
            continue
        provisional_basis.append(orthonormal)
        key = (primitive.family, primitive.reduction_class)
        quotas[key] = quotas.get(key, 0) + 1
        provisional_count += 1
        if provisional_count == count:
            break
    if provisional_count != count:
        return candidates

    records: list[tuple[object, ...]] = []
    for sequence, primitive in enumerate(candidates):
        normalized = _normalized_b_row_or_none(
            primitive,
            coords,
            rank_tolerance=rank_tolerance,
        )
        if normalized is None:
            continue
        residual = _b_row_residual_against_basis(normalized, basis)
        if float(np.linalg.norm(residual)) <= rank_tolerance:
            continue
        key = (primitive.family, primitive.reduction_class)
        if key not in quotas:
            continue
        records.append((key, "", (), np.zeros(0), residual, sequence, primitive))
    chosen = _partition_quota_rank_revealing_selection(records, quotas=quotas)
    if chosen is None:
        return candidates
    selected = tuple(record[6] for record in chosen)
    selected_ids = {primitive.identifier for primitive in selected}
    return selected + tuple(
        primitive for primitive in candidates if primitive.identifier not in selected_ids
    )


def _csv_or_none(values: tuple[str, ...]) -> str:
    return ",".join(values) if values else "NONE"


def _select_ranked_primitives(
    candidates: tuple[GICPrimitive, ...],
    coords: np.ndarray,
    *,
    target_rank: int,
    rank_tolerance: float,
) -> tuple[tuple[GICPrimitive, ...], int]:
    selected, rank, _ = _select_ranked_primitives_with_diagnostics(
        candidates,
        coords,
        target_rank=target_rank,
        rank_tolerance=rank_tolerance,
    )
    return selected, rank


def _primitive_candidate_partitions(
    candidates: tuple[GICPrimitive, ...],
    *,
    coords: np.ndarray,
    rank_tolerance: float,
    condition_ordinary: bool,
) -> tuple[
    list[GICPrimitive],
    list[GICPrimitive],
    list[GICPrimitive],
    list[GICPrimitive],
    list[GICPrimitive],
]:
    user = [primitive for primitive in candidates if _is_user_protected_primitive(primitive)]
    pseudo_cycle = [
        primitive
        for primitive in candidates
        if not _is_user_protected_primitive(primitive)
        and _is_pseudo_cycle_primitive(primitive)
    ]
    analytic_salc = [
        primitive
        for primitive in candidates
        if not _is_user_protected_primitive(primitive)
        and not _is_pseudo_cycle_primitive(primitive)
        and _is_analytic_salc_candidate(primitive)
    ]
    special = [
        primitive
        for primitive in candidates
        if _is_special_primitive(primitive)
        and not _is_user_protected_primitive(primitive)
        and not _is_pseudo_cycle_primitive(primitive)
        and not _is_analytic_salc_candidate(primitive)
    ]
    ordinary = [
        primitive
        for primitive in candidates
        if not _is_user_protected_primitive(primitive)
        and not _is_special_primitive(primitive)
        and not _is_pseudo_cycle_primitive(primitive)
        and not _is_analytic_salc_candidate(primitive)
    ]
    ordinary = list(
        _coordinated_angular_candidate_order(
            tuple(ordinary),
            coords=coords,
            rank_tolerance=rank_tolerance,
        )
    )
    return user, pseudo_cycle, analytic_salc, special, ordinary


def _repair_selected_primitive_completion(
    selected: list[GICPrimitive],
    candidates: tuple[GICPrimitive, ...],
    *,
    rank: int,
    target_rank: int,
    coords: np.ndarray,
    rank_tolerance: float,
    skipped_singular: list[str],
    skipped_dependent: list[str],
    skipped_singular_details: list[str],
    skipped_dependent_details: list[str],
) -> tuple[list[GICPrimitive], int]:
    if rank >= target_rank:
        return selected, rank
    repaired = _repair_indivisible_completion(
        tuple(selected),
        candidates,
        coords=coords,
        target_rank=target_rank,
        rank_tolerance=rank_tolerance,
    )
    if repaired is None:
        return selected, rank
    selected = list(repaired)
    selected_ids = {primitive.identifier for primitive in selected}
    skipped_singular[:] = [
        identifier for identifier in skipped_singular if identifier not in selected_ids
    ]
    skipped_singular_details[:] = [
        item
        for item in skipped_singular_details
        if item.split(":", 1)[0] not in selected_ids
    ]
    skipped_dependent[:] = [
        primitive.identifier
        for primitive in candidates
        if primitive.identifier not in selected_ids
        and primitive.identifier not in skipped_singular
    ]
    skipped_dependent_details[:] = [
        _primitive_diagnostic_token(primitive)
        for primitive in candidates
        if primitive.identifier in skipped_dependent
    ]
    return selected, target_rank


def _select_large_sparse_covalent_chart(
    candidates: tuple[GICPrimitive, ...],
    coords: np.ndarray,
    *,
    target_rank: int,
    rank_tolerance: float,
) -> tuple[tuple[GICPrimitive, ...], int, GICReductionDiagnostics] | None:
    """Select a large local chart with linear-storage sparse rank algebra."""

    from scipy.linalg import qr
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching
    from scipy.sparse.linalg import splu

    coordinates = np.asarray(coords, dtype=float)
    ncart = coordinates.size
    gauge_count = ncart - int(target_rank)
    if gauge_count < 0 or gauge_count > 6:
        return None
    centered = coordinates - np.mean(coordinates, axis=0)
    rigid = np.zeros((6, ncart), dtype=float)
    for atom, (x, y, z) in enumerate(centered):
        offset = 3 * atom
        rigid[0, offset] = 1.0
        rigid[1, offset + 1] = 1.0
        rigid[2, offset + 2] = 1.0
        rigid[3, offset : offset + 3] = (0.0, -z, y)
        rigid[4, offset : offset + 3] = (z, 0.0, -x)
        rigid[5, offset : offset + 3] = (-y, x, 0.0)
    _q, _r, pivots = qr(rigid, mode="economic", pivoting=True)
    fixed = set(int(value) for value in pivots[:gauge_count])
    free = tuple(column for column in range(ncart) if column not in fixed)
    if len(free) != target_rank:
        return None
    free_position = {column: index for index, column in enumerate(free)}

    data: list[float] = []
    indices: list[int] = []
    indptr = [0]
    nonsingular_candidates: list[GICPrimitive] = []
    singular: list[GICPrimitive] = []
    for primitive in candidates:
        try:
            row = _sparse_analytic_b_row(
                primitive,
                coordinates,
                reference_coords=coordinates if primitive.function == "FROT" else None,
            )
        except (ArithmeticError, FloatingPointError, ValueError):
            singular.append(primitive)
            continue
        norm = row.norm()
        if not np.isfinite(norm) or norm <= rank_tolerance:
            singular.append(primitive)
            continue
        for column, value in row.entries:
            position = free_position.get(int(column))
            if position is not None and abs(float(value)) > 1.0e-15:
                indices.append(position)
                data.append(float(value) / norm)
        indptr.append(len(data))
        nonsingular_candidates.append(primitive)
    matrix = csr_matrix(
        (
            np.asarray(data, dtype=float),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=(len(nonsingular_candidates), target_rank),
    )
    matching = maximum_bipartite_matching(matrix, perm_type="column")
    selected_indices = np.flatnonzero(matching >= 0)
    if selected_indices.size != target_rank:
        return None
    selected_matrix = matrix[selected_indices, :].tocsc()
    try:
        factor = splu(selected_matrix, permc_spec="COLAMD")
    except RuntimeError:
        return None
    pivots_u = np.abs(factor.U.diagonal())
    if (
        pivots_u.size != target_rank
        or float(np.min(pivots_u, initial=np.inf))
        <= float(rank_tolerance) * max(float(np.max(pivots_u, initial=0.0)), 1.0)
    ):
        return None
    selected = [nonsingular_candidates[int(index)] for index in selected_indices]
    selected_ids = {primitive.identifier for primitive in selected}
    skipped_dependent = [
        primitive for primitive in nonsingular_candidates if primitive.identifier not in selected_ids
    ]
    diagnostics = GICReductionDiagnostics(
        rank_method="SPARSE_GAUGE_MATCHING_SUPERLU_CERTIFIED",
        reduction_policy="LOCAL_CSR_FIXED_RIGID_GAUGE_EXACT_RANK",
        selected=tuple(primitive.identifier for primitive in selected),
        skipped_singular=tuple(primitive.identifier for primitive in singular),
        skipped_dependent=tuple(primitive.identifier for primitive in skipped_dependent),
        selected_by_family=_family_count_tokens(selected),
        skipped_singular_details=tuple(_primitive_diagnostic_token(item) for item in singular),
        skipped_dependent_details=tuple(
            _primitive_diagnostic_token(item) for item in skipped_dependent
        ),
        conditioning_decisions=(
            f"CSR_NNZ={matrix.nnz}",
            f"RIGID_GAUGE_COLUMNS={','.join(str(value) for value in sorted(fixed))}",
            f"MIN_SUPERLU_PIVOT={float(np.min(pivots_u)):.12g}",
        ),
    )
    return tuple(selected), target_rank, diagnostics


def _select_ranked_primitives_with_diagnostics(
    candidates: tuple[GICPrimitive, ...],
    coords: np.ndarray,
    *,
    target_rank: int,
    rank_tolerance: float,
    condition_ordinary: bool = False,
    condition_pseudobond_support: bool = False,
) -> tuple[tuple[GICPrimitive, ...], int, GICReductionDiagnostics]:
    if (
        target_rank >= 512
        and not condition_ordinary
        and not condition_pseudobond_support
        and all(
            not _is_special_primitive(primitive)
            and not _is_pseudo_cycle_primitive(primitive)
            and not _is_analytic_salc_candidate(primitive)
            and primitive.provenance != USER_PROVENANCE
            for primitive in candidates
        )
    ):
        sparse_selection = _select_large_sparse_covalent_chart(
            candidates,
            coords,
            target_rank=target_rank,
            rank_tolerance=rank_tolerance,
        )
        if sparse_selection is not None:
            return sparse_selection
    skipped_singular: list[str] = []
    skipped_dependent: list[str] = []
    skipped_singular_details: list[str] = []
    skipped_dependent_details: list[str] = []
    if target_rank == 0:
        return (
            (),
            0,
            GICReductionDiagnostics(
                rank_method=RANK_METHOD,
                reduction_policy=REDUCTION_POLICY,
            ),
        )
    selected: list[GICPrimitive] = []
    basis: list[np.ndarray] = []
    rank = 0

    (
        user_candidates,
        pseudo_cycle_candidates,
        analytic_salc_candidates,
        special_candidates,
        ordinary_candidates,
    ) = _primitive_candidate_partitions(
        candidates,
        coords=coords,
        rank_tolerance=rank_tolerance,
        condition_ordinary=condition_ordinary,
    )

    def select_candidates(
        group: tuple[GICPrimitive, ...],
        *,
        protected: bool,
    ) -> tuple[bool, int]:
        nonlocal rank
        component_units = coordinate_selection_units(
            tuple(
                CoordinateComponent(
                    operator=primitive.function,
                    atoms=primitive.atoms,
                    mode=primitive.mode,
                    ref_atoms=primitive.ref_atoms,
                    context=(primitive.family, *primitive.refs),
                )
                for primitive in group
            )
        )
        for unit_index, indices in enumerate(component_units):
            if rank == target_rank:
                if protected:
                    remaining = tuple(
                        group[index]
                        for unit in component_units[unit_index:]
                        for index in unit
                    )
                    singular, dependent, singular_details, dependent_details = (
                        _raise_if_remaining_special_independent(
                            remaining,
                            coords,
                            basis,
                            rank,
                            rank_tolerance=rank_tolerance,
                        )
                    )
                    skipped_singular.extend(singular)
                    skipped_dependent.extend(dependent)
                    skipped_singular_details.extend(singular_details)
                    skipped_dependent_details.extend(dependent_details)
                    return True, rank
                return False, rank
            primitives = tuple(group[index] for index in indices)
            if rank + len(primitives) > target_rank:
                for primitive in primitives:
                    _record_skip(
                        primitive,
                        "dependent",
                        skipped_singular,
                        skipped_dependent,
                        skipped_singular_details,
                        skipped_dependent_details,
                    )
                continue
            original_rank = rank
            original_selected = len(selected)
            original_basis = len(basis)
            statuses: list[str] = []
            for primitive in primitives:
                rank, status = _try_select_ranked_primitive(
                    primitive,
                    coords,
                    selected,
                    basis,
                    rank,
                    rank_tolerance=rank_tolerance,
                )
                statuses.append(status)
            if rank - original_rank != len(primitives):
                del selected[original_selected:]
                del basis[original_basis:]
                rank = original_rank
                statuses = [
                    "singular" if "singular" in statuses else "dependent"
                ] * len(primitives)
            for primitive, status in zip(primitives, statuses, strict=True):
                _record_skip(
                    primitive,
                    status,
                    skipped_singular,
                    skipped_dependent,
                    skipped_singular_details,
                    skipped_dependent_details,
                )
        return False, rank

    should_return, rank = select_candidates(tuple(user_candidates), protected=True)
    if should_return:
        return (
            tuple(selected),
            rank,
            _make_reduction_diagnostics(
                selected,
                skipped_singular=skipped_singular,
                skipped_dependent=skipped_dependent,
                skipped_singular_details=skipped_singular_details,
                skipped_dependent_details=skipped_dependent_details,
            ),
        )

    should_return, rank = select_candidates(tuple(pseudo_cycle_candidates), protected=True)
    if should_return:
        return (
            tuple(selected),
            rank,
            _make_reduction_diagnostics(
                selected,
                skipped_singular=skipped_singular,
                skipped_dependent=skipped_dependent,
                skipped_singular_details=skipped_singular_details,
                skipped_dependent_details=skipped_dependent_details,
            ),
        )

    should_return, rank = select_candidates(tuple(analytic_salc_candidates), protected=True)
    if should_return:
        return (
            tuple(selected),
            rank,
            _make_reduction_diagnostics(
                selected,
                skipped_singular=skipped_singular,
                skipped_dependent=skipped_dependent,
                skipped_singular_details=skipped_singular_details,
                skipped_dependent_details=skipped_dependent_details,
            ),
        )

    ts_contact_support = tuple(
        primitive
        for primitive in candidates
        if primitive.refs and primitive.refs[0] == "PSEUDOBOND_CONTACT_SUPPORT"
    )
    ts_reaction_distances = tuple(
        primitive
        for primitive in special_candidates
        if primitive.family == "TS_REACTION_DISTANCE"
    )
    has_ts_reactive_contact_completion = bool(
        ts_reaction_distances and ts_contact_support
    )
    has_tric_fragment_block = not pseudo_cycle_candidates and any(
        primitive.family in {"FRAG_TRANSLATION", "FRAG_ORIENTATION", "FRAG_DISTANCE"}
        for primitive in special_candidates
    )
    if has_ts_reactive_contact_completion:
        # A TS reaction distance is an ORACLE-required coordinate, while the
        # local contact pool is only a mathematical completion after reactive
        # L pairs have been excluded.  Preserve ordinary valence coordinates,
        # then add exactly the missing local in-plane/out-of-plane directions.
        support_ids = {primitive.identifier for primitive in ts_contact_support}
        select_candidates(ts_reaction_distances, protected=True)
        select_candidates(
            tuple(
                primitive
                for primitive in ordinary_candidates
                if primitive.identifier not in support_ids
            ),
            protected=False,
        )
        support = _conditioned_residual_primitive_order(
            ts_contact_support,
            coords=coords,
            selected_prefix=tuple(selected),
            basis=basis,
            count=target_rank - rank,
            rank_tolerance=rank_tolerance,
        )
        select_candidates(support, protected=False)
        select_candidates(
            tuple(
                primitive
                for primitive in special_candidates
                if primitive not in ts_reaction_distances
                and primitive.identifier not in support_ids
            ),
            protected=False,
        )
    elif has_tric_fragment_block:
        # A fragmented coordinate chart is the direct sum of complete
        # intrafragment SONIC spaces and the relative rigid-body TRIC block.
        # Complete the rigid inter-fragment block first.  Selecting all
        # intrafragment rows first can consume the available rank and leave
        # the last rotational pose directions unavailable.
        tric_candidates = tuple(
            primitive
            for primitive in special_candidates
            if primitive.family
            in {"FRAG_TRANSLATION", "FRAG_ORIENTATION", "FRAG_DISTANCE"}
        )
        center_candidates = tuple(
            primitive
            for primitive in special_candidates
            if primitive.family in {"CENTER_ATOM_DISTANCE", "FRAG_CENTER_ATOM_DISTANCE"}
        )
        prescribed_distance_candidates = tuple(
            primitive
            for primitive in special_candidates
            if primitive.function == "R"
            and primitive.family in {"FRAG_DISTANCE", "PSEUDO_BOND_DISTANCE"}
        )
        semantic_interfragment_candidates = tuple(
            primitive
            for primitive in special_candidates
            if primitive not in tric_candidates
            and primitive not in center_candidates
            and primitive not in prescribed_distance_candidates
        )
        # Declared haptic/virtual centers are scientific coordinates rather
        # than optional atom--atom contact observations.  Preserve their
        # independent radial rows before completing the relative pose block.
        select_candidates(center_candidates, protected=True)
        # SPECIAL_COORDINATES is the atlas-prescribed pose chart and is
        # mutually exclusive with a pseudobond chart.  Realize its complete
        # dimension-aware translation/orientation block first.  Classified
        # contacts remain observable ORACLE facts, but OPTIONAL contact rows
        # cannot displace quaternion, axial or singleton pose directions.
        tric_candidates = _rank_revealing_residual_primitive_order(
            tric_candidates,
            coords=coords,
            basis=basis,
            count=target_rank - rank,
            rank_tolerance=rank_tolerance,
        )
        # Pose candidates form one connected selection unit, but that unit is
        # deliberately redundant.  Selecting it atomically rolls back every
        # independent pose row as soon as one later row is dependent.  Consume
        # the rank-revealing order row by row instead.
        for primitive in tric_candidates:
            select_candidates((primitive,), protected=False)
        select_candidates(tuple(ordinary_candidates), protected=False)
        if rank < target_rank:
            # A dimensionally incomplete pose block may use only the declared
            # atlas completion pool.  This is a mathematical completion, not
            # a downstream reinterpretation of contact chemistry.
            completion_candidates = tuple(
                dict.fromkeys(
                    (
                        *center_candidates,
                        *prescribed_distance_candidates,
                        *semantic_interfragment_candidates,
                    )
                )
            )
            completion_candidates = _rank_revealing_residual_primitive_order(
                completion_candidates,
                coords=coords,
                basis=basis,
                count=target_rank - rank,
                rank_tolerance=rank_tolerance,
            )
            select_candidates(completion_candidates, protected=False)
        if rank == target_rank:
            selected_ids = {primitive.identifier for primitive in selected}
            unselected_observables = tuple(
                primitive
                for primitive in (
                    *center_candidates,
                    *prescribed_distance_candidates,
                    *semantic_interfragment_candidates,
                )
                if primitive.identifier not in selected_ids
            )
            singular, dependent, singular_details, dependent_details = (
                _raise_if_remaining_special_independent(
                    unselected_observables,
                    coords,
                    basis,
                    rank,
                    rank_tolerance=rank_tolerance,
                )
            )
            skipped_singular.extend(singular)
            skipped_dependent.extend(dependent)
            skipped_singular_details.extend(singular_details)
            skipped_dependent_details.extend(dependent_details)
    elif any(
        primitive.family == "PSEUDO_BOND_DISTANCE"
        for primitive in special_candidates
    ):
        # A classified pseudobond selects the natural contact chart instead
        # of the mutually exclusive rigid-pose chart.  Preserve complete
        # intrafragment SONIC spaces first, then choose the exact-rank local
        # distance complement around the prescribed contact.
        select_candidates(tuple(ordinary_candidates), protected=False)
        contact_candidates = tuple(
            primitive
            for primitive in special_candidates
            if primitive.family
            in {
                "PSEUDO_BOND_DISTANCE",
                "PSEUDO_BOND_BEND",
                "PSEUDO_BOND_TORSION",
                "FRAG_DISTANCE",
            }
        )
        prescribed = tuple(
            primitive
            for primitive in contact_candidates
            if primitive.family == "PSEUDO_BOND_DISTANCE"
        )
        support = tuple(
            primitive
            for primitive in contact_candidates
            if primitive.family != "PSEUDO_BOND_DISTANCE"
        )
        select_candidates(prescribed, protected=True)
        if condition_pseudobond_support:
            support = _conditioned_residual_primitive_order(
                support,
                coords=coords,
                selected_prefix=tuple(selected),
                basis=basis,
                count=target_rank - rank,
                rank_tolerance=rank_tolerance,
            )
        else:
            support = _rank_revealing_residual_primitive_order(
                support,
                coords=coords,
                basis=basis,
                count=target_rank - rank,
                rank_tolerance=rank_tolerance,
            )
        select_candidates(support, protected=False)
        select_candidates(
            tuple(
                primitive
                for primitive in special_candidates
                if primitive not in contact_candidates
            ),
            protected=False,
        )
    elif pseudo_cycle_candidates:
        select_candidates(tuple(ordinary_candidates), protected=False)
        select_candidates(tuple(special_candidates), protected=False)
    else:
        should_return, rank = select_candidates(tuple(special_candidates), protected=True)
        if should_return:
            return (
                tuple(selected),
                rank,
                _make_reduction_diagnostics(
                    selected,
                    skipped_singular=skipped_singular,
                    skipped_dependent=skipped_dependent,
                    skipped_singular_details=skipped_singular_details,
                    skipped_dependent_details=skipped_dependent_details,
                ),
            )
        if condition_ordinary:
            ordinary_candidates = list(
                _family_conditioned_residual_primitive_order(
                    tuple(ordinary_candidates),
                    coords=coords,
                    basis=basis,
                    count=target_rank - rank,
                    rank_tolerance=rank_tolerance,
                )
            )
        select_candidates(tuple(ordinary_candidates), protected=False)
    selected, rank = _repair_selected_primitive_completion(
        selected,
        candidates,
        rank=rank,
        target_rank=target_rank,
        coords=coords,
        rank_tolerance=rank_tolerance,
        skipped_singular=skipped_singular,
        skipped_dependent=skipped_dependent,
        skipped_singular_details=skipped_singular_details,
        skipped_dependent_details=skipped_dependent_details,
    )
    return (
        tuple(selected),
        rank,
        _make_reduction_diagnostics(
            selected,
            skipped_singular=skipped_singular,
            skipped_dependent=skipped_dependent,
            skipped_singular_details=skipped_singular_details,
            skipped_dependent_details=skipped_dependent_details,
        ),
    )


def _repair_indivisible_completion(
    selected: tuple[GICPrimitive, ...],
    candidates: tuple[GICPrimitive, ...],
    *,
    coords: np.ndarray,
    target_rank: int,
    rank_tolerance: float,
) -> tuple[GICPrimitive, ...] | None:
    """Close a one-step rank gap by exchanging scalars for one complete unit."""

    gap = target_rank - len(selected)
    if gap <= 0:
        return None
    components = tuple(
        CoordinateComponent(
            operator=primitive.function,
            atoms=primitive.atoms,
            mode=primitive.mode,
            ref_atoms=primitive.ref_atoms,
            context=(primitive.family, *primitive.refs),
        )
        for primitive in candidates
    )
    units = coordinate_selection_units(components)
    selected_ids = {primitive.identifier for primitive in selected}
    for unit in units:
        if len(unit) <= 1 or len(unit) < gap:
            continue
        additions = tuple(candidates[index] for index in unit)
        if any(primitive.identifier in selected_ids for primitive in additions):
            continue
        family = additions[0].family
        refs = additions[0].refs
        removable = tuple(
            primitive
            for primitive in selected
            if primitive.function != "L"
            and primitive.family == family
            and primitive.refs == refs
        )
        remove_count = len(unit) - gap
        for removed in combinations(removable, remove_count):
            removed_ids = {primitive.identifier for primitive in removed}
            trial = tuple(
                primitive
                for primitive in selected
                if primitive.identifier not in removed_ids
            ) + additions
            rows = tuple(
                _normalized_b_row_or_none(
                    primitive,
                    coords,
                    rank_tolerance=rank_tolerance,
                )
                for primitive in trial
            )
            if any(row is None for row in rows):
                continue
            matrix = np.vstack(rows)
            if numerical_matrix_rank(
                matrix,
                absolute_tolerance=rank_tolerance,
            ) == target_rank:
                return trial
    return None


def _coordinated_angular_candidate_order(
    candidates: tuple[GICPrimitive, ...],
    *,
    coords: np.ndarray,
    rank_tolerance: float,
) -> tuple[GICPrimitive, ...]:
    """Order complete CN=3/4 angular spaces by their joint numerical rank.

    Pair angles and out-of-plane primitives are alternative generators of one
    local angular tangent space.  Its exact dimension is ``2*CN-3``: three at
    a tricoordinate center and five at a tetracoordinate center.  Selecting
    them jointly avoids both the planar three-angle dependence and arbitrary
    five-of-six angle choices at tetrahedral/square-planar centers.
    """

    bends_by_center: dict[int, list[GICPrimitive]] = {}
    for primitive in candidates:
        if primitive.function == "A" and len(primitive.atoms) == 3:
            bends_by_center.setdefault(primitive.atoms[1], []).append(primitive)
    out_of_plane_by_center: dict[int, list[GICPrimitive]] = {}
    for primitive in candidates:
        if (
            primitive.family in {"OUT_OF_PLANE", "IMPROPER_DIHEDRAL"}
            and len(primitive.atoms) == 4
        ):
            out_of_plane_by_center.setdefault(primitive.atoms[0], []).append(primitive)

    chosen_by_center: dict[int, tuple[GICPrimitive, ...]] = {}
    local_ids_by_center: dict[int, set[str]] = {}
    for center, bends in bends_by_center.items():
        neighbors = {
            atom
            for primitive in bends
            for atom in (primitive.atoms[0], primitive.atoms[2])
        }
        coordination = len(neighbors)
        if coordination not in {3, 4}:
            continue
        expected_bends = coordination * (coordination - 1) // 2
        if len(bends) != expected_bends:
            continue
        local = tuple((*bends, *out_of_plane_by_center.get(center, ())))
        rows = []
        regular = []
        for primitive in local:
            normalized = _normalized_b_row_or_none(
                primitive,
                coords,
                rank_tolerance=rank_tolerance,
            )
            if normalized is not None:
                regular.append(primitive)
                rows.append(normalized)
        target = 2 * coordination - 3
        if len(regular) < target:
            continue
        selection = select_rank_revealing_rows(
            np.vstack(rows),
            target_rank=target,
            tolerance=rank_tolerance,
            tie_tolerance=1.0e-12,
        )
        if selection.rank != target:
            continue
        chosen_by_center[center] = tuple(regular[index] for index in selection.indices)
        local_ids_by_center[center] = {primitive.identifier for primitive in local}

    emitted: set[int] = set()
    deferred: list[GICPrimitive] = []
    ordered: list[GICPrimitive] = []
    for primitive in candidates:
        owners = tuple(
            center
            for center, identifiers in local_ids_by_center.items()
            if primitive.identifier in identifiers
        )
        if not owners:
            ordered.append(primitive)
            continue
        if len(owners) != 1:
            raise ValueError("angular primitive belongs to multiple local centers")
        center = owners[0]
        if center not in emitted:
            ordered.extend(chosen_by_center[center])
            emitted.add(center)
        if primitive.identifier not in {
            item.identifier for item in chosen_by_center[center]
        }:
            deferred.append(primitive)
    ordered.extend(deferred)
    return tuple(ordered)


def _is_special_primitive(primitive: GICPrimitive) -> bool:
    return primitive.reduction_class == SPECIAL_REDUCTION_CLASS or _is_user_protected_primitive(
        primitive
    )


def _is_user_protected_primitive(primitive: GICPrimitive) -> bool:
    return primitive.provenance.upper() == USER_PROVENANCE


def _is_pseudo_cycle_primitive(primitive: GICPrimitive) -> bool:
    return primitive.family in {"PSEUDO_CYCLE_BEND", "PSEUDO_CYCLE_TORSION"}


def _is_analytic_salc_candidate(primitive: GICPrimitive) -> bool:
    """Identify precomposed analytic coordinates that own their local subspace."""

    return primitive.function == "RPU"


def _protected_gic_count(primitives: tuple[GICPrimitive, ...]) -> int:
    return sum(
        1
        for primitive in primitives
        if _is_special_primitive(primitive) or _is_pseudo_cycle_primitive(primitive)
    )


def _skipped_singular_count(definition: GICDefinition) -> int:
    if definition.reduction_diagnostics is None:
        return 0
    return len(definition.reduction_diagnostics.skipped_singular)


def _skipped_dependent_count(definition: GICDefinition) -> int:
    if definition.reduction_diagnostics is None:
        return 0
    return len(definition.reduction_diagnostics.skipped_dependent)


def _reduction_diagnostics_lines(definition: GICDefinition) -> list[str]:
    diagnostics = definition.reduction_diagnostics or GICReductionDiagnostics(
        rank_method=RANK_METHOD,
        reduction_policy=REDUCTION_POLICY,
        selected=tuple(primitive.identifier for primitive in definition.primitives),
    )
    return [
        f"RANK_METHOD {diagnostics.rank_method}",
        f"REDUCTION_POLICY {diagnostics.reduction_policy}",
        f"SELECTED {_csv_or_none(diagnostics.selected)}",
        f"SELECTED_BY_FAMILY {_csv_or_none(diagnostics.selected_by_family)}",
        f"SKIPPED_SINGULAR {_csv_or_none(diagnostics.skipped_singular)}",
        f"SKIPPED_DEPENDENT {_csv_or_none(diagnostics.skipped_dependent)}",
        f"SKIPPED_SINGULAR_DETAILS {_csv_or_none(diagnostics.skipped_singular_details)}",
        f"SKIPPED_DEPENDENT_DETAILS {_csv_or_none(diagnostics.skipped_dependent_details)}",
        f"CONDITIONING_DECISIONS {_csv_or_none(diagnostics.conditioning_decisions)}",
    ]


def _symmetry_diagnostics_lines(definition: GICDefinition) -> list[str]:
    diagnostics = definition.symmetry_diagnostics or _empty_symmetry_diagnostics(
        definition.point_group,
        requested=definition.symmetrize,
    )
    lines = [
        f"METHOD {diagnostics.method}",
        f"POLICY {diagnostics.policy}",
        f"STATUS {diagnostics.status}",
        f"POINT_GROUP {diagnostics.point_group}",
        f"SYMMETRY_GROUP {diagnostics.symmetry_group}",
        f"TOTAL_SYMMETRIC_IRREP {diagnostics.total_symmetric_irrep}",
        f"TOTAL_SYMMETRIC_GICS {_csv_or_none(diagnostics.total_symmetric_gics)}",
        f"SIGN_GAUGE_POLICY {diagnostics.sign_gauge_policy}",
        f"PATH_GAUGE_POLICY {diagnostics.path_gauge_policy}",
        f"PATH_OVERLAP_WARNING_THRESHOLD {diagnostics.path_overlap_warning_threshold:.12g}",
        f"OPERATION_TOLERANCE_ANGSTROM {diagnostics.operation_tolerance_angstrom:.12g}",
        f"MAX_OPERATION_RESIDUAL_ANGSTROM {diagnostics.max_operation_residual_angstrom:.12g}",
        f"MIN_OPERATION_MARGIN_ANGSTROM {diagnostics.min_operation_margin_angstrom:.12g}",
        f"NEAR_THRESHOLD_OPERATIONS {_csv_or_none(diagnostics.near_threshold_operations)}",
        f"GROUP_COUNT {len(diagnostics.groups)}",
        f"SYMMETRIZED_GIC_COUNT {_symmetrized_gic_count(diagnostics)}",
    ]
    for index, group in enumerate(diagnostics.groups, start=1):
        lines.append(
            f"GROUP {index} BLOCK={group.block} FAMILY={group.family} "
            f"SIGNATURE={group.signature} "
            f"SOURCES={_csv_or_none(group.source_gics)} "
            f"OUTPUTS={_csv_or_none(group.output_gics)}"
        )
    return lines


def _empty_symmetry_diagnostics(
    point_group: str,
    *,
    requested: bool,
) -> GICSymmetrizationDiagnostics:
    return GICSymmetrizationDiagnostics(
        method="NONE",
        policy=SYMMETRIZATION_POLICY,
        status="NOT_REQUESTED" if not requested else "NO_ELIGIBLE_GROUPS",
        point_group=point_group,
        symmetry_group=point_group,
        total_symmetric_irrep=total_symmetric_irrep(point_group),
        total_symmetric_gics=(),
    )


def _symmetrized_gic_count(diagnostics: GICSymmetrizationDiagnostics) -> int:
    return sum(len(group.output_gics) for group in diagnostics.groups)


def _make_reduction_diagnostics(
    selected: list[GICPrimitive],
    *,
    skipped_singular: list[str],
    skipped_dependent: list[str],
    skipped_singular_details: list[str] | None = None,
    skipped_dependent_details: list[str] | None = None,
) -> GICReductionDiagnostics:
    return GICReductionDiagnostics(
        rank_method=RANK_METHOD,
        reduction_policy=REDUCTION_POLICY,
        selected=tuple(primitive.identifier for primitive in selected),
        skipped_singular=tuple(skipped_singular),
        skipped_dependent=tuple(skipped_dependent),
        selected_by_family=_family_count_tokens(selected),
        skipped_singular_details=tuple(skipped_singular_details or ()),
        skipped_dependent_details=tuple(skipped_dependent_details or ()),
    )


def _record_skip(
    primitive: GICPrimitive,
    status: str,
    skipped_singular: list[str],
    skipped_dependent: list[str],
    skipped_singular_details: list[str],
    skipped_dependent_details: list[str],
) -> None:
    if status == "singular":
        skipped_singular.append(primitive.identifier)
        skipped_singular_details.append(_primitive_diagnostic_token(primitive))
    elif status == "dependent":
        skipped_dependent.append(primitive.identifier)
        skipped_dependent_details.append(_primitive_diagnostic_token(primitive))


def _family_count_tokens(primitives: list[GICPrimitive]) -> tuple[str, ...]:
    counts: dict[str, int] = {}
    for primitive in primitives:
        counts[primitive.family] = counts.get(primitive.family, 0) + 1
    return tuple(f"{family}:{count}" for family, count in sorted(counts.items()))


def _primitive_diagnostic_token(primitive: GICPrimitive) -> str:
    return f"{primitive.identifier}:{primitive.family}:{primitive.name}"


def _apply_local_symmetrization(
    gics: tuple[FrozenGIC, ...],
    primitives: tuple[GICPrimitive, ...],
    *,
    atom_symbols: tuple[str, ...],
    point_group: str,
    requested: bool,
    symmetry_operations: tuple[GICPointGroupOperation, ...] = (),
    reference_coordinates_angstrom: tuple[tuple[float, float, float], ...] | None = None,
    intermolecular: bool = False,
    pseudobond_mode: bool = False,
) -> tuple[tuple[FrozenGIC, ...], GICSymmetrizationDiagnostics]:
    projector_operations, projector_point_group = _computational_projector_contract(
        point_group,
        symmetry_operations,
    )

    def with_operation_margins(
        result: tuple[tuple[FrozenGIC, ...], GICSymmetrizationDiagnostics],
    ) -> tuple[tuple[FrozenGIC, ...], GICSymmetrizationDiagnostics]:
        if reference_coordinates_angstrom is not None:
            expected_irrep_counts = _vibrational_irrep_coordinate_counts(
                point_group=point_group,
                operations=_valid_projector_operations(symmetry_operations),
                reference_coordinates_angstrom=np.asarray(
                    reference_coordinates_angstrom,
                    dtype=float,
                ),
            )
            has_adaptive_fullerene = any(
                str(gic.family).strip().upper() == "FULLERENE_FACE_DIAGONAL"
                for gic in result[0]
            )
            if (
                not has_adaptive_fullerene
                and
                expected_irrep_counts is not None
                and sum(expected_irrep_counts.values()) == len(result[0])
            ):
                actual_irrep_counts = {
                    irrep: sum(gic.irrep == irrep for gic in result[0])
                    for irrep in expected_irrep_counts
                }
                if actual_irrep_counts != expected_irrep_counts:
                    raise GICForgeContractError(
                        "symmetrized SONIC basis has incorrect molecular vibrational "
                        f"irrep counts for {point_group}: expected "
                        f"{expected_irrep_counts}, obtained {actual_irrep_counts}"
                    )
        projected_rank = _frozen_gic_b_matrix_rank(
            result[0],
            primitive_by_id={primitive.identifier: primitive for primitive in primitives},
            reference_coordinates_angstrom=reference_coordinates_angstrom,
        )
        has_native_ring_puckering = any(
            primitive.family == "RING_PUCKER_COMPONENT" and primitive.function in {"U", "D"}
            for primitive in primitives
        )
        if (
            has_native_ring_puckering
            and projected_rank is not None
            and projected_rank < len(result[0])
        ):
            # In C1 there is no molecular symmetry to impose.  Local SALCs
            # are an optional change of basis and must never destroy a valid
            # exact-rank chart, in particular when independent fragment
            # contact and rigid-pose blocks coexist.  Preserve the already
            # certified SMITH coordinates if a local regrouping loses rank.
            if projector_point_group.strip().upper() in {"C1", "UNKNOWN"}:
                fallback = _prefix_symmetrized_singletons(
                    gics,
                    point_group=projector_point_group,
                )
                fallback_rank = _frozen_gic_b_matrix_rank(
                    fallback,
                    primitive_by_id={
                        primitive.identifier: primitive for primitive in primitives
                    },
                    reference_coordinates_angstrom=reference_coordinates_angstrom,
                )
                if fallback_rank == len(fallback):
                    result = (
                        fallback,
                        replace(
                            result[1],
                            status="C1_RANK_PRESERVING_SINGLETON_FALLBACK",
                            groups=(),
                            total_symmetric_gics=tuple(gic.name for gic in fallback),
                        ),
                    )
                    projected_rank = fallback_rank
            if projected_rank < len(result[0]):
                raise GICForgeContractError(
                    "symmetry projection produced a rank-deficient global SONIC B matrix: "
                    f"rank {projected_rank} for {len(result[0])} coordinates"
                )
        diagnostics = result[1]
        if projector_point_group != point_group:
            diagnostics = replace(
                diagnostics,
                point_group=point_group,
                symmetry_group=projector_point_group,
            )
        return (
            result[0],
            _diagnostics_with_operation_margins(
                diagnostics,
                operations=symmetry_operations,
                reference_coordinates_angstrom=reference_coordinates_angstrom,
            ),
        )

    if not requested:
        return gics, _empty_symmetry_diagnostics(point_group, requested=False)

    # C1 has no non-trivial symmetry operation: every vibrational coordinate
    # is already totally symmetric.  Remember ordinary minima here so the
    # local fallback below can leave their proper torsions attached to their
    # own central bonds.  Other established local SALCs (notably XY3 angular
    # coordinates), pseudobond completions, and reactive charts remain intact.
    is_c1 = projector_point_group.strip().upper() in {"C1", "UNKNOWN"}
    is_reactive = any(
        gic.family in {"TS_REACTION_DISTANCE", TS_REACTIVE_COORDINATE_CLASS}
        for gic in gics
    )
    protect_c1_proper_torsions = is_c1 and not pseudobond_mode and not is_reactive

    projected = _apply_point_group_projector(
        gics,
        primitives,
        point_group=projector_point_group,
        symmetry_operations=projector_operations,
        reference_coordinates_angstrom=reference_coordinates_angstrom,
        intermolecular=intermolecular,
        prefer_cartesian_projection=projector_point_group != point_group,
    )
    if projected is not None:
        return with_operation_margins(projected)

    torsion_metric_result = _pseudobond_torsion_metric_eigenbasis(
        gics,
        primitives,
        point_group=projector_point_group,
        reference_coordinates_angstrom=reference_coordinates_angstrom,
        enabled=pseudobond_mode,
    )
    metric_torsion_gics: tuple[FrozenGIC, ...] = ()
    metric_torsion_source_ids: frozenset[str] = frozenset()
    metric_torsion_group: GICSymmetrizedGroup | None = None
    if torsion_metric_result is not None:
        metric_torsion_gics, metric_torsion_source_ids, metric_torsion_group = torsion_metric_result

    local_group_inputs = tuple(
        gic for gic in gics if gic.identifier not in metric_torsion_source_ids
    )
    source_groups = _local_symmetry_groups(
        local_group_inputs,
        primitives,
        atom_symbols=atom_symbols,
        intermolecular=intermolecular,
        symmetry_operations=projector_operations,
    )
    if protect_c1_proper_torsions:
        source_groups = {
            key: group for key, group in source_groups.items() if key[1] != "TORSION"
        }
    if not source_groups and metric_torsion_group is None:
        prefixed_gics = _prefix_symmetrized_singletons(
            gics,
            point_group=projector_point_group,
        )
        return with_operation_margins(
            (
                prefixed_gics,
                GICSymmetrizationDiagnostics(
                    method=LOCAL_SYMMETRIZATION_METHOD,
                    policy=SYMMETRIZATION_POLICY,
                    status="NO_ELIGIBLE_GROUPS",
                    point_group=point_group,
                    symmetry_group=projector_point_group,
                    total_symmetric_irrep=total_symmetric_irrep(projector_point_group),
                    total_symmetric_gics=tuple(
                        gic.name
                        for gic in prefixed_gics
                        if is_total_symmetric_irrep(projector_point_group, gic.irrep)
                    ),
                ),
            )
        )

    groups_by_first = {group[0].identifier: (key, group) for key, group in source_groups.items()}
    grouped_ids = {gic.identifier for group in source_groups.values() for gic in group}
    name_counters: dict[tuple[str, str, str], int] = {}
    output: list[FrozenGIC] = []
    diagnostics: list[GICSymmetrizedGroup] = []

    for gic in gics:
        if gic.identifier in metric_torsion_source_ids:
            if metric_torsion_gics:
                renumbered = tuple(
                    _renumber_frozen_gic(metric_gic, len(output) + offset)
                    for offset, metric_gic in enumerate(metric_torsion_gics, start=1)
                )
                output.extend(renumbered)
                assert metric_torsion_group is not None
                diagnostics.append(
                    replace(
                        metric_torsion_group,
                        output_gics=tuple(item.name for item in renumbered),
                    )
                )
                metric_torsion_gics = ()
            continue
        if gic.identifier in groups_by_first:
            key, group = groups_by_first[gic.identifier]
            new_gics = _symmetrized_group_gics(
                key,
                group,
                first_index=len(output) + 1,
                name_counters=name_counters,
                point_group=projector_point_group,
            )
            output.extend(new_gics)
            diagnostics.append(
                GICSymmetrizedGroup(
                    block=key[0],
                    family=key[1],
                    signature=key[2],
                    source_gics=tuple(source.name for source in group),
                    output_gics=tuple(new_gic.name for new_gic in new_gics),
                )
            )
            continue
        if gic.identifier in grouped_ids:
            continue
        output.append(
            _renumber_frozen_gic(
                _prefix_symmetrized_gic(gic, point_group=projector_point_group),
                len(output) + 1,
            )
        )

    output_tuple = tuple(output)
    return with_operation_margins(
        (
            output_tuple,
            GICSymmetrizationDiagnostics(
                method=LOCAL_SYMMETRIZATION_METHOD,
                policy=(
                    PSEUDOBOND_TORSION_METRIC_SYMMETRIZATION_POLICY
                    if metric_torsion_group is not None
                    else SYMMETRIZATION_POLICY
                ),
                status="APPLIED",
                point_group=point_group,
                symmetry_group=projector_point_group,
                total_symmetric_irrep=total_symmetric_irrep(projector_point_group),
                total_symmetric_gics=tuple(
                    gic.name
                    for gic in output_tuple
                    if is_total_symmetric_irrep(projector_point_group, gic.irrep)
                ),
                groups=tuple(diagnostics),
            ),
        )
    )


def _pseudobond_torsion_metric_eigenbasis(
    gics: tuple[FrozenGIC, ...],
    primitives: tuple[GICPrimitive, ...],
    *,
    point_group: str,
    reference_coordinates_angstrom: tuple[tuple[float, float, float], ...] | None,
    enabled: bool,
) -> tuple[tuple[FrozenGIC, ...], frozenset[str], GICSymmetrizedGroup] | None:
    """Build the non-null torsion complement in its Cartesian B metric.

    Pseudobond cycles can generate several formally distinct valence torsions
    whose Wilson rows span the same directions after stretches and bends have
    been retained.  A Euclidean sum/difference SALC exposes arbitrary,
    poorly-conditioned combinations of that span.  For C1 pseudobond charts,
    project every torsion candidate against the already selected non-torsion
    space, diagonalize the residual ``B B^T`` metric, and retain exactly the
    non-null complement required by the frozen chart.  Coefficients remain an
    ordinary orthonormal rotation within the TORSION family; coordinate types
    and physical periodic units are therefore not mixed or rescaled.
    """

    if (
        not enabled
        or point_group.strip().upper() not in {"C1", "UNKNOWN"}
        or reference_coordinates_angstrom is None
    ):
        return None
    torsion_source_gics = tuple(gic for gic in gics if gic.family == "TORSION")
    if len(torsion_source_gics) < 2:
        return None
    non_torsion_gics = tuple(gic for gic in gics if gic.family != "TORSION")
    torsion_candidates = tuple(
        primitive for primitive in primitives if primitive.family == "TORSION"
    )
    if len(torsion_candidates) < len(torsion_source_gics):
        return None

    coords = np.asarray(reference_coordinates_angstrom, dtype=float)
    primitive_by_id = {primitive.identifier: primitive for primitive in primitives}
    try:
        fixed_rows = np.vstack(
            [
                _frozen_gic_reference_b_row(
                    gic,
                    primitive_by_id=primitive_by_id,
                    coords=coords,
                )
                for gic in non_torsion_gics
            ]
        )
        candidate_rows = np.vstack(
            [_reference_tangent_b_row(primitive, coords) for primitive in torsion_candidates]
        )
    except (FloatingPointError, GICForgeContractError, ValueError):
        return None

    if fixed_rows.size:
        _u, singular_values, vh = np.linalg.svd(fixed_rows, full_matrices=False)
        fixed_rank = int(np.sum(singular_values > RANK_TOLERANCE))
        fixed_basis = vh[:fixed_rank]
        residual_rows = candidate_rows - (candidate_rows @ fixed_basis.T) @ fixed_basis
    else:
        residual_rows = candidate_rows

    residual_metric = residual_rows @ residual_rows.T
    residual_metric = 0.5 * (residual_metric + residual_metric.T)
    eigenvalues, eigenvectors = np.linalg.eigh(residual_metric)
    maximum = float(np.max(eigenvalues, initial=0.0))
    tolerance = max(RANK_TOLERANCE**2, maximum * 1.0e-10)
    non_null = tuple(
        index for index in np.argsort(eigenvalues)[::-1] if float(eigenvalues[index]) > tolerance
    )
    required = len(torsion_source_gics)
    if len(non_null) != required:
        raise GICForgeContractError(
            "pseudobond torsion B-metric complement does not match the frozen chart: "
            f"need {required}, obtained {len(non_null)} non-null modes from "
            f"{len(torsion_candidates)} candidates"
        )

    output: list[FrozenGIC] = []
    for mode, eigen_index in enumerate(non_null, start=1):
        vector = _canonical_vector_sign(np.asarray(eigenvectors[:, eigen_index], dtype=float))
        coefficients = tuple(
            (primitive.identifier, float(coefficient))
            for primitive, coefficient in zip(torsion_candidates, vector)
            if abs(float(coefficient)) > 1.0e-12
        )
        output.append(
            FrozenGIC(
                identifier=f"GIC{mode:03d}",
                name=f"{irrep_name_prefix(total_symmetric_irrep(point_group))}TorsM{mode:03d}",
                family="TORSION",
                irrep=total_symmetric_irrep(point_group),
                primitive_id=coefficients[0][0],
                gaussian_expression="LINEAR_COMBINATION",
                coefficients=coefficients,
            )
        )

    source_ids = frozenset(gic.identifier for gic in torsion_source_gics)
    group = GICSymmetrizedGroup(
        block="TORSION",
        family="TORSION",
        signature=(
            "RESIDUAL_B_METRIC_EIGENBASIS:"
            f"CANDIDATES={len(torsion_candidates)}:"
            f"RANK={required}:NULL={len(torsion_candidates) - required}"
        ),
        source_gics=tuple(gic.name for gic in torsion_source_gics),
        output_gics=tuple(gic.name for gic in output),
    )
    return tuple(output), source_ids, group


def _frozen_gic_reference_b_row(
    gic: FrozenGIC,
    *,
    primitive_by_id: dict[str, GICPrimitive],
    coords: np.ndarray,
) -> np.ndarray:
    row = np.zeros(coords.size, dtype=float)
    for primitive_id, coefficient in gic.coefficients or ((gic.primitive_id, 1.0),):
        primitive = primitive_by_id.get(primitive_id)
        if primitive is None:
            raise GICForgeContractError(
                f"missing primitive {primitive_id!r} in frozen GIC {gic.name!r}"
            )
        row += float(coefficient) * _reference_tangent_b_row(primitive, coords)
    return row


def _frozen_gic_b_matrix_rank(
    gics: tuple[FrozenGIC, ...],
    *,
    primitive_by_id: dict[str, GICPrimitive],
    reference_coordinates_angstrom: tuple[tuple[float, float, float], ...] | None,
) -> int | None:
    """Return the Cartesian rank of a complete frozen SONIC set when evaluable."""
    if reference_coordinates_angstrom is None:
        return None
    coords = np.asarray(reference_coordinates_angstrom, dtype=float)
    primitive_rows: dict[str, np.ndarray] = {}
    rows: list[np.ndarray] = []
    for gic in gics:
        row = np.zeros(coords.size, dtype=float)
        for primitive_id, coefficient in gic.coefficients or ((gic.primitive_id, 1.0),):
            primitive = primitive_by_id.get(primitive_id)
            if primitive is None:
                return None
            try:
                primitive_row = primitive_rows.get(primitive_id)
                if primitive_row is None:
                    primitive_row = _reference_tangent_b_row(primitive, coords)
                    primitive_rows[primitive_id] = primitive_row
                row += float(coefficient) * primitive_row
            except (FloatingPointError, GICForgeContractError, ValueError):
                return None
        rows.append(row)
    if not rows:
        return 0
    matrix = np.vstack(rows)
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1.0e-14):
        return 0
    # This is an explicit small-system reference audit. Production large
    # charts use the sparse direct-sum/rank certificate instead.
    return normalized_dense_rank(
        matrix,
        absolute_tolerance=RANK_TOLERANCE,
        zero_row_tolerance=1.0e-14,
    )


def _apply_point_group_projector(
    gics: tuple[FrozenGIC, ...],
    primitives: tuple[GICPrimitive, ...],
    *,
    point_group: str,
    symmetry_operations: tuple[GICPointGroupOperation, ...],
    reference_coordinates_angstrom: tuple[tuple[float, float, float], ...] | None,
    intermolecular: bool = False,
    prefer_cartesian_projection: bool = False,
) -> tuple[tuple[FrozenGIC, ...], GICSymmetrizationDiagnostics] | None:
    operations = _valid_projector_operations(symmetry_operations)
    if len(operations) <= 1 or point_group.upper() in {"C1", "UNKNOWN"}:
        return None
    group_key = point_group.strip().upper()
    coords_for_global = (
        np.asarray(reference_coordinates_angstrom, dtype=float)
        if reference_coordinates_angstrom is not None
        else None
    )
    if coords_for_global is not None and any(
        primitive.family == "LINEAR_BEND" for primitive in primitives
    ):
        coupled_angular = _apply_cartesian_interfragment_projector(
            gics,
            primitives,
            point_group=point_group,
            operations=operations,
            reference_coordinates_angstrom=coords_for_global,
            couple_linear_angular=True,
        )
        if coupled_angular is not None:
            return coupled_angular
    has_local_xh = any(gic.family == "LOCAL_XH_STRETCH" for gic in gics)
    has_native_ring_puckering = any(
        primitive.family == "RING_PUCKER_COMPONENT" and primitive.function in {"U", "D"}
        for primitive in primitives
    )
    if prefer_cartesian_projection and coords_for_global is not None:
        projected = _apply_cartesian_interfragment_projector(
            gics,
            primitives,
            point_group=point_group,
            operations=operations,
            reference_coordinates_angstrom=coords_for_global,
        )
        if projected is not None:
            return projected
    if (
        intermolecular
        and group_key not in {"C1", "UNKNOWN"}
        and coords_for_global is not None
        and not has_local_xh
    ):
        projected = _apply_cartesian_interfragment_projector(
            gics,
            primitives,
            point_group=point_group,
            operations=operations,
            reference_coordinates_angstrom=coords_for_global,
        )
        if projected is not None:
            return projected
    use_global_b_selection = has_native_ring_puckering or group_key in {
        "T",
        "TD",
        "O",
        "OH",
        "I",
        "IH",
    }
    coords = (
        np.asarray(reference_coordinates_angstrom, dtype=float)
        if use_global_b_selection and reference_coordinates_angstrom is not None
        else None
    )

    primitive_by_id = {primitive.identifier: primitive for primitive in primitives}
    blocks: dict[tuple[str, str], list[FrozenGIC]] = {}
    for gic in gics:
        key = _symmetry_pool_key(gic.family)
        blocks.setdefault(key, []).append(gic)

    output: list[FrozenGIC] = []
    diagnostics: list[GICSymmetrizedGroup] = []
    name_counters: dict[tuple[str, str], int] = {}
    global_b_basis: tuple[np.ndarray, ...] = ()
    reference_b_cache: dict[str, np.ndarray] = {}
    for key, block_gics in sorted(
        blocks.items(),
        key=lambda item: _projector_block_sort_key(
            item,
            protect_special=use_global_b_selection,
        ),
    ):
        if key[1] == "LOCAL_XH_STRETCH":
            block_output = tuple(
                _renumber_frozen_gic(gic, len(output) + offset)
                for offset, gic in enumerate(block_gics, start=1)
            )
            output.extend(block_output)
            diagnostics.append(
                GICSymmetrizedGroup(
                    block=key[0],
                    family=key[1],
                    signature="UNSYMMETRIZED_LOCAL_XH",
                    source_gics=tuple(gic.name for gic in block_gics),
                    output_gics=tuple(gic.name for gic in block_output),
                )
            )
            continue
        ring_pucker_coords = (
            np.asarray(reference_coordinates_angstrom, dtype=float)
            if key[1] == "RING_PUCKER_COMPONENT" and reference_coordinates_angstrom is not None
            else None
        )
        projected = _project_gic_block(
            key,
            tuple(block_gics),
            primitive_by_id=primitive_by_id,
            operations=operations,
            point_group=point_group,
            first_index=len(output) + 1,
            name_counters=name_counters,
            reference_coordinates_angstrom=coords,
            ring_pucker_coordinates_angstrom=ring_pucker_coords,
            global_b_basis=global_b_basis,
            reference_b_cache=reference_b_cache,
        )
        if projected is None:
            if coords_for_global is not None:
                global_projected = _apply_cartesian_interfragment_projector(
                    gics,
                    primitives,
                    point_group=point_group,
                    operations=operations,
                    reference_coordinates_angstrom=coords_for_global,
                )
                if global_projected is not None:
                    return global_projected
                # A local family can fail closure/covariance although the
                # complete symmetry-closed primitive pool spans the required
                # molecular irreps. Use the same audited global selector as
                # the multiplicity/covariance gates below before returning to
                # local SALC heuristics. Otherwise this early return bypasses
                # that general construction and can assign incorrect labels.
                global_projected = _apply_rank_revealing_global_projector(
                    gics,
                    primitives,
                    point_group=point_group,
                    operations=operations,
                    reference_coordinates_angstrom=coords_for_global,
                )
                if global_projected is not None:
                    return global_projected
            return None
        block_output, block_diagnostics, global_b_basis = projected
        output.extend(block_output)
        diagnostics.append(block_diagnostics)

    output_tuple = tuple(output)
    if coords_for_global is not None:
        expected_irrep_counts = _vibrational_irrep_coordinate_counts(
            point_group=point_group,
            operations=operations,
            reference_coordinates_angstrom=coords_for_global,
        )
        actual_irrep_counts = {
            irrep: sum(gic.irrep == irrep for gic in output_tuple)
            for irrep in (expected_irrep_counts or {})
        }
        if (
            expected_irrep_counts is not None
            and sum(expected_irrep_counts.values()) == len(output_tuple)
            and actual_irrep_counts != expected_irrep_counts
        ):
            # A family-by-family projector may close every local block while
            # still selecting an incomplete component of a multidimensional
            # molecular irrep.  Rebuild globally under the exact 3N-6 irrep
            # multiplicities instead of accepting locally plausible labels.
            # First project the Wilson rows themselves.  This is essential
            # for pseudoscalar coordinates whose symbolic atom permutation
            # can exchange conventional B1/B2 labels under an improper
            # operation even though their Cartesian transformation is
            # unambiguous.  If decimal geometry noise prevents that exact
            # family-preserving construction, use the symbolic global pool.
            global_projected = _apply_cartesian_interfragment_projector(
                gics,
                primitives,
                point_group=point_group,
                operations=operations,
                reference_coordinates_angstrom=coords_for_global,
            )
            if global_projected is None and any(
                primitive.family == "LINEAR_BEND" for primitive in primitives
            ):
                # At an exactly linear center, bends, linear bends, torsions,
                # and out-of-plane primitives can be individually singular or
                # redundant even though their combined angular tangent space
                # is closed under the molecular point group.  Project that
                # angular space as one dimensionless block while keeping bond
                # stretches separate.  This preserves the full molecular
                # symmetry without mixing length and angular units.
                global_projected = _apply_cartesian_interfragment_projector(
                    gics,
                    primitives,
                    point_group=point_group,
                    operations=operations,
                    reference_coordinates_angstrom=coords_for_global,
                    couple_linear_angular=True,
                )
            if global_projected is None:
                global_projected = _apply_rank_revealing_global_projector(
                gics,
                primitives,
                point_group=point_group,
                operations=operations,
                reference_coordinates_angstrom=coords_for_global,
                )
            if global_projected is not None:
                return global_projected
            raise GICForgeContractError(
                "symmetry projection cannot realize the molecular vibrational "
                f"irrep counts for {point_group}: expected {expected_irrep_counts}, "
                f"obtained {actual_irrep_counts}"
            )
        operation_margins = _symmetry_operation_margins(
            operations,
            reference_coordinates_angstrom=tuple(
                tuple(float(value) for value in row) for row in coords_for_global
            ),
        )
        quasi_operations = bool(
            operation_margins is not None
            and operation_margins[0]
            > _CARTESIAN_INTERFRAGMENT_PROJECTOR_TOLERANCE
        )
        try:
            covariance_failures = _cartesian_irrep_b_covariance_failures(
                output_tuple,
                primitive_by_id=primitive_by_id,
                point_group=point_group,
                reference_coordinates_angstrom=coords_for_global,
                operations=operations,
            )
            family_repair_allowed = (
                covariance_failures
                and not any(gic.family == "RING_PUCKER_COMPONENT" for gic in output_tuple)
                and _point_group_has_only_one_dimensional_irreps(
                    point_group,
                    operations,
                )
            )
            if family_repair_allowed:
                repaired = _apply_cartesian_interfragment_projector(
                    output_tuple,
                    primitives,
                    point_group=point_group,
                    operations=operations,
                    reference_coordinates_angstrom=coords_for_global,
                    repair_keys=frozenset(
                        (
                            primitive_symmetry_block(failure[0].family),
                            failure[0].family,
                        )
                        for failure in covariance_failures
                    ),
                )
                if repaired is not None:
                    return repaired
            elif covariance_failures and _is_even_dnd_group(point_group):
                # A reduced family can contain fewer rows than are required
                # to close a multidimensional irrep.  Exact family quotas are
                # then mathematically impossible; retain the established
                # global Cartesian selector rather than splitting a multiplet.
                global_projected = _apply_cartesian_interfragment_projector(
                    gics,
                    primitives,
                    point_group=point_group,
                    operations=operations,
                    reference_coordinates_angstrom=coords_for_global,
                )
                if global_projected is not None:
                    return global_projected
            if covariance_failures:
                if quasi_operations:
                    # Decimal endpoint geometries can retain a rigorously
                    # identified point group while missing exact Cartesian
                    # covariance at the fixed numerical audit threshold.  A
                    # global rank-revealing projection is stable under that
                    # rounding and is still constrained by the molecular
                    # irrep multiplicities.
                    global_projected = _apply_rank_revealing_global_projector(
                        gics,
                        primitives,
                        point_group=point_group,
                        operations=operations,
                        reference_coordinates_angstrom=coords_for_global,
                    )
                    if global_projected is not None:
                        return global_projected
                    return None
                _validate_cartesian_irrep_b_covariance(
                    output_tuple,
                    primitive_by_id=primitive_by_id,
                    point_group=point_group,
                    reference_coordinates_angstrom=coords_for_global,
                    operations=operations,
                )
        except (FloatingPointError, ValueError):
            # Synthetic definitions used without a physical reference
            # geometry cannot support a Cartesian covariance audit.
            pass
        projected_rank = _frozen_gic_b_matrix_rank(
            output_tuple,
            primitive_by_id=primitive_by_id,
            reference_coordinates_angstrom=reference_coordinates_angstrom,
        )
        if projected_rank is not None and projected_rank < len(output_tuple):
            if quasi_operations:
                return None
            global_projected = _apply_rank_revealing_global_projector(
                gics,
                primitives,
                point_group=point_group,
                operations=operations,
                reference_coordinates_angstrom=coords_for_global,
            )
            if global_projected is not None:
                return global_projected
            return None
    return (
        output_tuple,
        GICSymmetrizationDiagnostics(
            method=POINT_GROUP_PROJECTOR_METHOD,
            policy=PROJECTOR_SYMMETRIZATION_POLICY,
            status="APPLIED",
            point_group=point_group,
            symmetry_group=point_group,
            total_symmetric_irrep=total_symmetric_irrep(point_group),
            total_symmetric_gics=tuple(
                gic.name for gic in output_tuple if is_total_symmetric_irrep(point_group, gic.irrep)
            ),
            groups=tuple(diagnostics),
        ),
    )


def _apply_cartesian_interfragment_projector(
    gics: tuple[FrozenGIC, ...],
    primitives: tuple[GICPrimitive, ...],
    *,
    point_group: str,
    operations: tuple[GICPointGroupOperation, ...],
    reference_coordinates_angstrom: np.ndarray,
    repair_keys: frozenset[tuple[str, str]] | None = None,
    couple_linear_angular: bool = False,
) -> tuple[tuple[FrozenGIC, ...], GICSymmetrizationDiagnostics] | None:
    """Build pure-family SALCs for an intermolecular coordinate model.

    Linear bends and fragment-frame coordinates do not, in general, transform
    by a literal permutation of their symbolic definitions.  Their Wilson B
    rows do transform exactly in Cartesian space.  This projector therefore
    closes each primitive family under the molecular operations, projects its
    B-row span, and performs one global rank-revealing selection without ever
    mixing coordinate families.
    """

    if point_group.strip().upper() in {"C1", "UNKNOWN"} or len(operations) <= 1:
        return None
    coords = np.asarray(reference_coordinates_angstrom, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        return None
    projector_representation_tolerance = _cartesian_covariance_tolerance(
        operations,
        reference_coordinates_angstrom=coords,
    )

    primitive_by_id = {primitive.identifier: primitive for primitive in primitives}
    separate_domains = repair_keys is None
    source_blocks: dict[tuple[str, str], list[FrozenGIC]] = {}
    for gic in gics:
        key = _projector_gic_pool_key(
            gic,
            primitive_by_id=primitive_by_id,
            separate_domains=separate_domains,
            couple_linear_angular=couple_linear_angular,
        )
        source_blocks.setdefault(key, []).append(gic)
    if not source_blocks:
        return None
    if repair_keys is not None and (not repair_keys or not repair_keys.issubset(source_blocks)):
        return None

    operation_labels = tuple(operation.label for operation in operations)
    operation_matrices = tuple(operation.rotation for operation in operations)
    irreps = irrep_characters_for_operations(
        operation_labels,
        point_group,
        operation_matrices=operation_matrices,
    )
    expected_irrep_counts = (
        _vibrational_irrep_coordinate_counts(
            point_group=point_group,
            operations=operations,
            reference_coordinates_angstrom=coords,
        )
        if repair_keys is None
        else None
    )
    cartesian_operations = tuple(
        _cartesian_operation_matrix(operation, natoms=len(coords)) for operation in operations
    )

    # candidate = (block key, irrep, primitive tuple, coefficient vector,
    #              normalized Cartesian B row, deterministic sequence)
    candidates: list[
        tuple[
            tuple[str, str],
            str,
            tuple[GICPrimitive, ...],
            np.ndarray,
            np.ndarray,
            int,
        ]
    ] = []
    sequence = 0
    for key, _block_gics in sorted(
        source_blocks.items(),
        key=lambda item: _projector_block_sort_key(item),
    ):
        if repair_keys is not None and key not in repair_keys:
            continue
        block_primitives = tuple(
            primitive
            for primitive in primitives
            if _projector_primitive_pool_key(
                primitive,
                separate_domains=separate_domains,
                couple_linear_angular=couple_linear_angular,
            )
            == key
        )
        if not block_primitives:
            return None
        try:
            primitive_b_rows = np.vstack(
                [_reference_tangent_b_row(primitive, coords) for primitive in block_primitives]
            )
        except (FloatingPointError, GICForgeContractError, ValueError):
            return None

        if key[1] == "RING_PUCKER_COMPONENT":
            primitive_index = _projector_primitive_index_with_aliases(
                block_primitives,
                primitive_by_id=primitive_by_id,
                key=key,
            )
            source_vectors = tuple(
                _gic_coefficient_vector(
                    gic,
                    primitive_index=primitive_index,
                    vector_size=len(block_primitives),
                )
                for gic in _block_gics
            )
            if any(vector is None for vector in source_vectors):
                return None
            physical_source_vectors = tuple(
                np.asarray(vector, dtype=float)
                for vector in source_vectors
                if vector is not None
            )
            if _ring_source_domains_are_disjoint(
                block_primitives,
                physical_source_vectors,
            ) and _point_group_has_only_one_dimensional_irreps(
                point_group,
                operations,
            ):
                physical_ring_candidates = _physical_ring_projector_candidates(
                    block_primitives,
                    physical_source_vectors,
                    block_b_matrix=primitive_b_rows,
                    operations=operations,
                    point_group=point_group,
                )
                if physical_ring_candidates is None:
                    return None
                for irrep, coefficients, normalized_b_row in physical_ring_candidates:
                    candidates.append(
                        (
                            key,
                            irrep,
                            block_primitives,
                            coefficients,
                            normalized_b_row,
                            sequence,
                        )
                    )
                    sequence += 1
                continue

        irrep_bases: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        for irrep, characters in irreps:
            if len(characters) != len(operations):
                return None
            if all(abs(float(character)) <= 1.0e-14 for character in characters):
                continue
            basis = irrep_bases.setdefault(irrep, [])
            for source_b_row in primitive_b_rows:
                projected_b_row = sum(
                    float(character) * (source_b_row @ cartesian)
                    for character, cartesian in zip(characters, cartesian_operations)
                ) / float(len(operations))
                source_norm = float(np.linalg.norm(source_b_row))
                if float(np.linalg.norm(projected_b_row)) <= (
                    _CARTESIAN_INTERFRAGMENT_PROJECTOR_TOLERANCE * max(1.0, source_norm)
                ):
                    continue
                coefficients, *_ = np.linalg.lstsq(
                    primitive_b_rows.T,
                    projected_b_row.T,
                    rcond=1.0e-10,
                )
                represented_b_row = coefficients @ primitive_b_rows
                representation_error = float(np.linalg.norm(represented_b_row - projected_b_row))
                # Candidate rows are normalized below.  Therefore the
                # representability audit must be relative to the projected
                # row itself: an absolute unit-scale floor would admit the
                # same Cartesian contamination for a weak projected
                # component and then amplify it during normalization.  The
                # preceding null-row guard already excludes projections too
                # small to support a stable relative test.
                projected_norm = float(np.linalg.norm(projected_b_row))
                if representation_error > (
                    projector_representation_tolerance * projected_norm
                ):
                    continue

                residual_b_row = np.asarray(represented_b_row, dtype=float)
                residual_coefficients = np.asarray(coefficients, dtype=float)
                for _pass in range(2):
                    for basis_b_row, basis_coefficients in basis:
                        overlap = float(np.dot(residual_b_row, basis_b_row))
                        residual_b_row -= overlap * basis_b_row
                        residual_coefficients -= overlap * basis_coefficients
                norm = float(np.linalg.norm(residual_b_row))
                if not np.isfinite(norm) or norm <= _CARTESIAN_INTERFRAGMENT_PROJECTOR_TOLERANCE:
                    continue
                residual_b_row /= norm
                residual_coefficients /= norm
                significant = np.flatnonzero(np.abs(residual_coefficients) > 1.0e-12)
                if significant.size and residual_coefficients[int(significant[0])] < 0.0:
                    residual_b_row = -residual_b_row
                    residual_coefficients = -residual_coefficients
                basis.append((residual_b_row, residual_coefficients))
                candidates.append(
                    (
                        key,
                        irrep,
                        block_primitives,
                        residual_coefficients,
                        residual_b_row,
                        sequence,
                    )
                )
                sequence += 1

    if not candidates:
        return None

    selected: list[
        tuple[
            tuple[str, str],
            str,
            tuple[GICPrimitive, ...],
            np.ndarray,
            np.ndarray,
            int,
        ]
    ] = []
    global_basis: list[np.ndarray] = []
    if repair_keys is not None:
        fixed_gics = tuple(
            gic
            for gic in gics
            if _symmetry_pool_key(gic.family) not in repair_keys
        )
        try:
            fixed_rows = (
                np.vstack(
                    [
                        _frozen_gic_reference_b_row(
                            gic,
                            primitive_by_id=primitive_by_id,
                            coords=coords,
                        )
                        for gic in fixed_gics
                    ]
                )
                if fixed_gics
                else np.empty((0, coords.size), dtype=float)
            )
        except (FloatingPointError, GICForgeContractError, ValueError):
            return None
        if len(fixed_rows):
            _u, fixed_singular_values, fixed_vh = np.linalg.svd(
                fixed_rows,
                full_matrices=False,
            )
            fixed_rank = int(
                np.sum(fixed_singular_values > _CARTESIAN_INTERFRAGMENT_PROJECTOR_TOLERANCE)
            )
            if fixed_rank != len(fixed_gics):
                return None
            global_basis.extend(np.asarray(row, dtype=float) for row in fixed_vh[:fixed_rank])

    def select_from_pool(
        pool: list[
            tuple[
                tuple[str, str],
                str,
                tuple[GICPrimitive, ...],
                np.ndarray,
                np.ndarray,
                int,
            ]
        ],
        count: int,
        irrep_remaining: dict[str, int] | None = None,
    ) -> bool:
        for _index in range(count):
            best_index: int | None = None
            best_residual: np.ndarray | None = None
            best_score = -1.0
            best_sequence = 0
            for candidate_index, candidate in enumerate(pool):
                if (
                    irrep_remaining is not None
                    and irrep_remaining.get(str(candidate[1]), 0) <= 0
                ):
                    continue
                residual = _b_row_residual_against_basis(candidate[4], global_basis)
                score = float(np.linalg.norm(residual))
                if score > best_score + 1.0e-12 or (
                    abs(score - best_score) <= 1.0e-12 and candidate[5] < best_sequence
                ):
                    best_index = candidate_index
                    best_residual = residual
                    best_score = score
                    best_sequence = candidate[5]
            if (
                best_index is None
                or best_residual is None
                or best_score <= _CARTESIAN_INTERFRAGMENT_PROJECTOR_TOLERANCE
            ):
                return False
            chosen = pool.pop(best_index)
            global_basis.append(best_residual / best_score)
            selected.append(chosen)
            if irrep_remaining is not None:
                irrep = str(chosen[1])
                irrep_remaining[irrep] -= 1
        return True

    has_fragment_pose = any(
        key[1] in {"FRAG_TRANSLATION", "FRAG_ORIENTATION"} for key in source_blocks
    )
    used_partition_quota_fallback = False
    if repair_keys is not None:
        residual_candidates = [
            (
                key,
                irrep,
                block_primitives,
                coefficients,
                _b_row_residual_against_basis(b_row, global_basis),
                sequence,
            )
            for key, irrep, block_primitives, coefficients, b_row, sequence in candidates
        ]
        quota_selection = _partition_quota_rank_revealing_selection(
            residual_candidates,
            quotas={key: len(source_blocks[key]) for key in repair_keys},
        )
        if quota_selection is None:
            return None
        candidate_by_residual_id = {
            id(residual): candidate for residual, candidate in zip(residual_candidates, candidates)
        }
        selected = [candidate_by_residual_id[id(candidate)] for candidate in quota_selection]
        used_partition_quota_fallback = True
    elif has_fragment_pose:
        # The relative rigid pose has exactly three translations and three
        # rotations.  Preserve each reduced family's dimension while letting
        # the global B-rank choose its stable, symmetry-pure representatives.
        # Select the two pose families jointly: a sequential greedy choice can
        # consume a direction belonging to both local-frame spans and leave
        # the second family one row short even though a full-rank 3+3 basis
        # exists.
        ordered_keys = sorted(
            source_blocks,
            key=lambda key: (
                primitive_reduction_class(key[1]) == SPECIAL_REDUCTION_CLASS,
                _projector_block_sort_key((key, source_blocks[key])),
            ),
        )
        pose_keys = tuple(
            key for key in ordered_keys if key[1] in {"FRAG_TRANSLATION", "FRAG_ORIENTATION"}
        )
        selection_failed = False
        for key in (
            candidate_key for candidate_key in ordered_keys if candidate_key not in pose_keys
        ):
            pool = [candidate for candidate in candidates if candidate[0] == key]
            if not select_from_pool(pool, len(source_blocks[key])):
                selection_failed = True
                break
        if not selection_failed:
            pose_pools = tuple(
                tuple(candidate for candidate in candidates if candidate[0] == key)
                for key in pose_keys
            )
            pose_quotas = tuple(len(source_blocks[key]) for key in pose_keys)
            if any(len(pool) < quota for pool, quota in zip(pose_pools, pose_quotas)):
                selection_failed = True
            else:
                best_pose_choice: (
                    tuple[
                        tuple[
                            tuple[
                                tuple[str, str],
                                str,
                                tuple[GICPrimitive, ...],
                                np.ndarray,
                                np.ndarray,
                                int,
                            ],
                            ...,
                        ],
                        ...,
                    ]
                    | None
                ) = None
                best_pose_score: tuple[float, float] | None = None
                best_pose_sequence: tuple[int, ...] | None = None
                # ``global_basis`` is immutable while the Cartesian pose
                # combinations are scored.  A candidate can occur in many
                # combinations, so compute its residual once instead of
                # repeating the same two-pass orthogonalization for every
                # combination containing it.
                pose_residuals = {
                    candidate[5]: _b_row_residual_against_basis(candidate[4], global_basis)
                    for pool in pose_pools
                    for candidate in pool
                }
                for pose_choice in product(
                    *(
                        tuple(combinations(pool, quota))
                        for pool, quota in zip(pose_pools, pose_quotas)
                    )
                ):
                    chosen_candidates = tuple(
                        candidate for group in pose_choice for candidate in group
                    )
                    residual_rows = np.vstack(
                        [pose_residuals[candidate[5]] for candidate in chosen_candidates]
                    )
                    singular_values = np.linalg.svd(residual_rows, compute_uv=False)
                    if (
                        len(singular_values) != len(chosen_candidates)
                        or float(singular_values[-1])
                        <= _CARTESIAN_INTERFRAGMENT_PROJECTOR_TOLERANCE
                    ):
                        continue
                    score = (
                        float(singular_values[-1]),
                        float(np.prod(singular_values)),
                    )
                    sequence_key = tuple(candidate[5] for candidate in chosen_candidates)
                    if (
                        best_pose_score is None
                        or score > best_pose_score
                        or (score == best_pose_score and sequence_key < (best_pose_sequence or ()))
                    ):
                        best_pose_choice = pose_choice
                        best_pose_score = score
                        best_pose_sequence = sequence_key
                if best_pose_choice is None:
                    selection_failed = True
                else:
                    for chosen_group in best_pose_choice:
                        for chosen in chosen_group:
                            residual = _b_row_residual_against_basis(chosen[4], global_basis)
                            residual_norm = float(np.linalg.norm(residual))
                            if residual_norm <= _CARTESIAN_INTERFRAGMENT_PROJECTOR_TOLERANCE:
                                selection_failed = True
                                break
                            global_basis.append(residual / residual_norm)
                            selected.append(chosen)
                        if selection_failed:
                            break
        if selection_failed:
            quota_selection = _partition_quota_rank_revealing_selection(
                candidates,
                quotas={key: len(block_gics) for key, block_gics in source_blocks.items()},
            )
            if quota_selection is None:
                return None
            selected = list(quota_selection)
            used_partition_quota_fallback = True
    elif any(
        gic.family in {"TS_REACTION_DISTANCE", "PSEUDO_BOND_DISTANCE"}
        for gic in gics
    ):
        # A pseudo-bond augments graph connectivity, but it does not erase the
        # physical identity of the reduced SONIC families.  Project and
        # condition the complete vibrational space subject to the exact
        # reduced family dimensions; otherwise a global selector can replace
        # bends and reaction distances by a large torsional subspace.
        quota_selection = _partition_quota_rank_revealing_selection(
            candidates,
            quotas={key: len(block_gics) for key, block_gics in source_blocks.items()},
        )
        if quota_selection is None:
            return None
        selected = list(quota_selection)
        used_partition_quota_fallback = True
    else:
        # Coordinate-family dimensions are part of the ORACLE/SMITH semantic
        # contract for every chart, not only for reaction pseudobonds.  The
        # molecular irrep audit may require a global exchange, but it must not
        # silently replace puckering or butterfly coordinates with ordinary
        # stretches and bends.
        ordered_keys = tuple(sorted(
            source_blocks,
            key=lambda item: _projector_block_sort_key((item, source_blocks[item])),
        ))
        key_orders = (
            tuple(permutations(ordered_keys))
            if len(ordered_keys) <= 6
            else tuple(
                dict.fromkeys(
                    ordered_keys[index:] + ordered_keys[:index]
                    for index in range(len(ordered_keys))
                )
            )
            + (tuple(reversed(ordered_keys)),)
        )
        selection_failed = True
        for key_order in key_orders:
            selected = []
            global_basis = []
            irrep_remaining = (
                None
                if expected_irrep_counts is None
                else dict(expected_irrep_counts)
            )
            if all(
                select_from_pool(
                    [candidate for candidate in candidates if candidate[0] == key],
                    len(source_blocks[key]),
                    irrep_remaining,
                )
                for key in key_order
            ) and (
                irrep_remaining is None
                or all(remaining == 0 for remaining in irrep_remaining.values())
            ):
                selection_failed = False
                break
        if selection_failed:
            quota_selection = _partition_quota_rank_revealing_selection(
                candidates,
                quotas={key: len(block_gics) for key, block_gics in source_blocks.items()},
            )
            if quota_selection is None:
                return None
            selected = list(quota_selection)
            used_partition_quota_fallback = True

    target_count = (
        sum(len(source_blocks[key]) for key in repair_keys)
        if repair_keys is not None
        else len(gics)
    )
    if len(selected) != target_count:
        return None
    if expected_irrep_counts is not None:
        actual_irrep_counts = {
            irrep: sum(str(candidate[1]) == irrep for candidate in selected)
            for irrep in expected_irrep_counts
        }
        if actual_irrep_counts != expected_irrep_counts:
            return None

    output: list[FrozenGIC] = []
    name_counters: dict[tuple[str, str], int] = {}
    diagnostics_by_key: dict[tuple[str, str], list[str]] = {}
    if repair_keys is not None:
        for source in gics:
            key = (primitive_symmetry_block(source.family), source.family)
            if key not in repair_keys:
                counter_key = (source.family, source.irrep)
                name_counters[counter_key] = name_counters.get(counter_key, 0) + 1
    for key, irrep, block_primitives, vector, _b_row, _sequence in selected:
        normalized_vector = _normalized_coefficient_vector_or_none(vector)
        if normalized_vector is None:
            return None
        coefficients = _coefficients_from_vector(
            block_primitives,
            _canonical_vector_sign(normalized_vector),
        )
        if not coefficients:
            return None
        family = key[1]
        gic = FrozenGIC(
            identifier=f"GIC{len(output) + 1:03d}",
            name=_next_projected_name(family, irrep, name_counters),
            family=family,
            irrep=irrep,
            primitive_id=coefficients[0][0],
            gaussian_expression="LINEAR_COMBINATION",
            coefficients=coefficients,
        )
        output.append(gic)
        diagnostics_by_key.setdefault(key, []).append(gic.name)

    if repair_keys is not None:
        replacements = {
            key: [
                gic for gic in output if (primitive_symmetry_block(gic.family), gic.family) == key
            ]
            for key in repair_keys
        }
        offsets = {key: 0 for key in repair_keys}
        merged: list[FrozenGIC] = []
        for source in gics:
            key = (primitive_symmetry_block(source.family), source.family)
            if key not in repair_keys:
                merged.append(_renumber_frozen_gic(source, len(merged) + 1))
                continue
            replacement = replacements[key][offsets[key]]
            offsets[key] += 1
            merged.append(_renumber_frozen_gic(replacement, len(merged) + 1))
        output = merged
        diagnostics_by_key = {
            key: [
                gic.name
                for gic in output
                if (primitive_symmetry_block(gic.family), gic.family) == key
            ]
            for key in source_blocks
        }

    output_tuple = tuple(output)
    try:
        _validate_cartesian_irrep_b_covariance(
            output_tuple,
            primitive_by_id={primitive.identifier: primitive for primitive in primitives},
            point_group=point_group,
            reference_coordinates_angstrom=coords,
            operations=operations,
            tolerance=projector_representation_tolerance,
        )
    except GICForgeContractError:
        # This Cartesian-family realization is only one exact construction
        # route. Let the caller use the symbolic global projector when the
        # selected rows fail the final normalized rank/covariance gate.
        return None
    diagnostics = tuple(
        GICSymmetrizedGroup(
            block=key[0],
            family=key[1],
            signature=(
                "CARTESIAN_BROWS_OPS="
                if repair_keys is None
                else ("CARTESIAN_B_FAMILY_REPAIR_OPS=" if key in repair_keys else "UNCHANGED_OPS=")
            )
            + ",".join(operation_labels),
            source_gics=tuple(gic.name for gic in source_blocks[key]),
            output_gics=tuple(output_names),
        )
        for key, output_names in diagnostics_by_key.items()
    )
    return (
        output_tuple,
        GICSymmetrizationDiagnostics(
            method=POINT_GROUP_PROJECTOR_METHOD,
            policy=PROJECTOR_SYMMETRIZATION_POLICY,
            status=(
                "APPLIED_PARTITION_QUOTA_RANK_REVEALING"
                if used_partition_quota_fallback and repair_keys is None
                else "APPLIED"
            ),
            point_group=point_group,
            symmetry_group=point_group,
            total_symmetric_irrep=total_symmetric_irrep(point_group),
            total_symmetric_gics=tuple(
                gic.name for gic in output_tuple if is_total_symmetric_irrep(point_group, gic.irrep)
            ),
            groups=diagnostics,
            fallback_events=(
                (
                    make_fallback_event(
                        stage="SMITH_SYMMETRIZATION",
                        algorithm_id="PARTITION_QUOTA_RANK_REVEALING_PROJECTOR",
                        trigger="SEQUENTIAL_FAMILY_PROJECTOR_COULD_NOT_SATISFY_EXACT_QUOTAS",
                        source="PARTITION_QUOTA_RANK_REVEALING_PROJECTOR",
                    ),
                )
                if used_partition_quota_fallback and repair_keys is None
                else ()
            ),
        ),
    )


def _cartesian_irrep_character_projectors(
    *,
    point_group: str,
    operations: tuple[GICPointGroupOperation, ...],
    natoms: int,
) -> tuple[tuple[str, np.ndarray], ...]:
    """Build the central Cartesian projector for every real irrep."""

    operation_labels = tuple(operation.label for operation in operations)
    operation_matrices = tuple(operation.rotation for operation in operations)
    cartesian_operations = tuple(
        _cartesian_operation_matrix(operation, natoms=natoms) for operation in operations
    )
    projectors: list[tuple[str, np.ndarray]] = []
    for irrep, characters in irrep_characters_for_operations(
        operation_labels,
        point_group,
        operation_matrices=operation_matrices,
    ):
        if len(characters) != len(cartesian_operations):
            raise GICForgeContractError(
                f"Cartesian character projector cannot resolve irrep {irrep} in {point_group}"
            )
        character_norm = sum(float(character) ** 2 for character in characters) / float(
            len(characters)
        )
        if character_norm <= 1.0e-12:
            raise GICForgeContractError(
                f"Cartesian character projector found a null irrep {irrep} in {point_group}"
            )
        projector = sum(
            float(character) * cartesian
            for character, cartesian in zip(
                characters,
                cartesian_operations,
                strict=True,
            )
        )
        projector *= float(irrep_dimension(irrep)) / (
            character_norm * float(len(cartesian_operations))
        )
        projectors.append((irrep, projector))
    return tuple(projectors)


def _computational_projector_contract(
    physical_point_group: str,
    operations: tuple[GICPointGroupOperation, ...],
) -> tuple[tuple[GICPointGroupOperation, ...], str]:
    """Return the closed operation set used for character projectors.

    Linear molecules retain their physical ``Cinfv`` or ``Dinfh`` identity in
    the frozen molecular contract.  Their finite ORACLE operation sample is
    not itself a group, so all SMITH character operations consistently use the
    same highest closed subgroup selected for symmetry-adapted Cartesian
    blocks.
    """

    validated = _valid_projector_operations(operations)
    if physical_point_group.strip().upper() not in {"CINFV", "DINFH"} or not validated:
        return validated, physical_point_group
    operation_records, computational_point_group = _linear_computational_subgroup(
        tuple(
            (
                operation.label,
                np.asarray(operation.rotation, dtype=float),
                operation.permutation,
            )
            for operation in validated
        ),
        physical_point_group=physical_point_group,
    )
    return (
        tuple(
            GICPointGroupOperation(
                label=label,
                rotation=tuple(tuple(float(value) for value in row) for row in rotation),
                permutation=tuple(int(atom) for atom in permutation),
            )
            for label, rotation, permutation in operation_records
        ),
        computational_point_group,
    )


def _vibrational_irrep_coordinate_counts(
    *,
    point_group: str,
    operations: tuple[GICPointGroupOperation, ...],
    reference_coordinates_angstrom: np.ndarray,
) -> dict[str, int] | None:
    """Return the exact number of vibrational rows required for each irrep.

    The character of the Cartesian displacement representation is the number
    of atoms fixed by an operation times the trace of its spatial matrix.
    Translations transform as polar vectors and rotations as axial vectors.
    For a linear geometry only the two rotations perpendicular to the
    molecular axis are removed.  Decomposition of the remaining character
    gives a molecule-wide invariant that applies equally to minima and
    transition states and does not depend on the chosen internal-coordinate
    families.
    """

    validated, computational_point_group = _computational_projector_contract(
        point_group,
        operations,
    )
    coords = np.asarray(reference_coordinates_angstrom, dtype=float)
    if (
        len(validated) <= 1
        or point_group.strip().upper() in {"C1", "UNKNOWN"}
        or coords.ndim != 2
        or coords.shape[1] != 3
        or coords.shape[0] != len(validated[0].permutation)
    ):
        return None

    centered = coords - np.mean(coords, axis=0)
    _u, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    geometry_scale = max(1.0, float(singular_values[0]) if len(singular_values) else 0.0)
    geometry_rank = int(np.sum(singular_values > 1.0e-10 * geometry_scale))
    linear_axis = np.asarray(vh[0], dtype=float) if geometry_rank == 1 else None

    vibrational_characters: list[float] = []
    for operation in validated:
        rotation = np.asarray(operation.rotation, dtype=float)
        trace = float(np.trace(rotation))
        determinant = float(np.linalg.det(rotation))
        fixed_atoms = sum(
            atom_index == target_atom
            for atom_index, target_atom in enumerate(operation.permutation, start=1)
        )
        translation_character = trace
        if geometry_rank == 0:
            rotation_character = 0.0
        elif linear_axis is not None:
            axial_trace = determinant * trace
            axial_parallel = determinant * float(linear_axis @ rotation @ linear_axis)
            rotation_character = axial_trace - axial_parallel
        else:
            rotation_character = determinant * trace
        vibrational_characters.append(
            float(fixed_atoms) * trace
            - translation_character
            - rotation_character
        )

    operation_labels = tuple(operation.label for operation in validated)
    operation_matrices = tuple(operation.rotation for operation in validated)
    result: dict[str, int] = {}
    for irrep, characters in irrep_characters_for_operations(
        operation_labels,
        computational_point_group,
        operation_matrices=operation_matrices,
    ):
        if len(characters) != len(validated):
            return None
        character_norm = sum(float(value) ** 2 for value in characters) / float(
            len(validated)
        )
        if character_norm <= 1.0e-12:
            return None
        multiplicity = sum(
            vibrational * float(character)
            for vibrational, character in zip(
                vibrational_characters,
                characters,
                strict=True,
            )
        ) / (float(len(validated)) * character_norm)
        rounded = int(round(multiplicity))
        if rounded < 0 or abs(multiplicity - rounded) > 1.0e-6:
            return None
        result[irrep] = rounded * irrep_dimension(irrep)

    rotational_dimension = 0 if geometry_rank == 0 else 2 if geometry_rank == 1 else 3
    expected_rank = 3 * len(coords) - 3 - rotational_dimension
    if sum(result.values()) != expected_rank:
        return None
    return result


def _cartesian_covariance_tolerance(
    operations: tuple[GICPointGroupOperation, ...],
    *,
    reference_coordinates_angstrom: np.ndarray,
) -> float:
    """Scale the B-row covariance audit by frozen-geometry rounding only."""

    margins = _symmetry_operation_margins(
        operations,
        reference_coordinates_angstrom=tuple(
            tuple(float(value) for value in row)
            for row in np.asarray(reference_coordinates_angstrom, dtype=float)
        ),
    )
    operation_residual = 0.0 if margins is None else float(margins[0])
    return max(
        _CARTESIAN_INTERFRAGMENT_PROJECTOR_TOLERANCE,
        5.0 * operation_residual,
    )


def validate_molecular_vibrational_irrep_counts(
    definition: GICDefinition,
    *,
    symmetry_operations: tuple[GICPointGroupOperation, ...],
) -> dict[str, int]:
    """Validate the complete molecular irrep composition of a SONIC basis.

    This is a consumer-side contract gate, not merely a construction check.
    Every tool reading a symmetry-adapted SONIC artifact can therefore reject
    stale or incorrectly labelled coordinates before using active-variable
    subsets, Hessians, optimizers, or backend exports.
    """

    if not definition.symmetrize or definition.point_group.strip().upper() in {
        "C1",
        "UNKNOWN",
    }:
        return {}
    # Fullerene cage charts use graph-orbit/local SALCs.  Under small
    # distortions the detected molecular group (often Ci/Cs) need not retain
    # the ideal-cage irrep multiplicities, while the sparse cage basis remains
    # complete and rank-certified.  Defer irrep-count enforcement to the
    # graph-orbit construction for this adaptive family.
    if any(
        str(gic.family).strip().upper() == "FULLERENE_FACE_DIAGONAL"
        for gic in definition.gics
    ):
        return {}
    expected = _vibrational_irrep_coordinate_counts(
        point_group=definition.point_group,
        operations=_valid_projector_operations(symmetry_operations),
        reference_coordinates_angstrom=np.asarray(
            definition.reference_coordinates_angstrom,
            dtype=float,
        ),
    )
    if expected is None:
        if not _valid_projector_operations(symmetry_operations):
            # Standalone unit/block constructions can request local naming
            # without carrying a molecular operation contract. Persisted
            # molecular artifacts always carry the ORACLE operations and are
            # checked on read below.
            return {}
        raise GICForgeContractError(
            "cannot derive molecular vibrational irrep counts from the frozen "
            f"{definition.point_group} symmetry operations"
        )
    if sum(expected.values()) != definition.target_rank:
        # A deliberately partial coordinate block is not a complete molecular
        # vibrational basis. Its enclosing molecular artifact is validated at
        # the consumer boundary once all blocks have been assembled.
        return {}
    if definition.target_rank != len(definition.gics):
        raise GICForgeContractError(
            "symmetry-adapted SONIC coordinate count does not span the complete "
            f"vibrational representation: expected {sum(expected.values())}, "
            f"obtained {len(definition.gics)}"
        )
    actual = {
        irrep: sum(gic.irrep == irrep for gic in definition.gics)
        for irrep in expected
    }
    unexpected = sorted(
        {
            gic.irrep
            for gic in definition.gics
            if gic.irrep not in expected
        }
    )
    if actual != expected or unexpected:
        suffix = f"; unexpected labels {unexpected}" if unexpected else ""
        raise GICForgeContractError(
            "symmetry-adapted SONIC basis has incorrect molecular vibrational "
            f"irrep counts for {definition.point_group}: expected {expected}, "
            f"obtained {actual}{suffix}"
        )
    return expected


def _cartesian_irrep_b_covariance_failures(
    gics: tuple[FrozenGIC, ...],
    *,
    primitive_by_id: dict[str, GICPrimitive],
    point_group: str,
    reference_coordinates_angstrom: np.ndarray,
    operations: tuple[GICPointGroupOperation, ...],
    tolerance: float = 1.0e-7,
) -> tuple[tuple[FrozenGIC, str, float], ...]:
    """Return Cartesian rows outside their declared irrep isotypic component.

    Invariance of all rows carrying one label is insufficient: a direct sum of
    two different one-dimensional irreps is itself invariant.  The character
    projector below verifies the declared irrep before the existing covariance
    audit checks closure of repeated copies under every group operation.
    """

    rows_by_irrep: dict[str, list[tuple[FrozenGIC, np.ndarray]]] = {}
    for gic in gics:
        row = np.zeros(reference_coordinates_angstrom.size, dtype=float)
        for primitive_id, coefficient in gic.coefficients or ((gic.primitive_id, 1.0),):
            primitive = primitive_by_id.get(primitive_id)
            if primitive is None:
                raise GICForgeContractError(
                    f"Cartesian B covariance audit cannot resolve primitive {primitive_id}"
                )
            row += float(coefficient) * _reference_tangent_b_row(
                primitive,
                reference_coordinates_angstrom,
            )
        norm = float(np.linalg.norm(row))
        if not np.isfinite(norm) or norm <= 1.0e-12:
            raise GICForgeContractError(
                f"Cartesian B covariance audit found a singular SONIC row {gic.name}"
            )
        rows_by_irrep.setdefault(gic.irrep, []).append((gic, row / norm))

    all_rows = np.vstack([row for named_rows in rows_by_irrep.values() for _gic, row in named_rows])
    if numerical_matrix_rank(
        all_rows,
        absolute_tolerance=_CARTESIAN_INTERFRAGMENT_PROJECTOR_TOLERANCE,
    ) != len(gics):
        raise GICForgeContractError(
            "Cartesian symmetry projector produced a rank-deficient intermolecular SONIC basis"
        )

    natoms = len(reference_coordinates_angstrom)
    cartesian_operations = tuple(
        _cartesian_operation_matrix(operation, natoms=natoms) for operation in operations
    )
    projectors_by_irrep = dict(
        _cartesian_irrep_character_projectors(
            point_group=point_group,
            operations=operations,
            natoms=natoms,
        )
    )
    failures: list[tuple[FrozenGIC, str, float]] = []
    for irrep, named_rows in rows_by_irrep.items():
        character_projector = projectors_by_irrep.get(irrep)
        if character_projector is None:
            raise GICForgeContractError(
                f"Cartesian B covariance audit cannot resolve irrep {irrep} in {point_group}"
            )
        for gic, row in named_rows:
            projected = row @ character_projector
            residual = float(np.linalg.norm(projected - row))
            if residual > tolerance:
                failures.append((gic, "CHARACTER_PROJECTOR", residual))

        basis = np.vstack([row for _gic, row in named_rows])
        _u, singular_values, vh = np.linalg.svd(basis, full_matrices=False)
        rank = int(np.sum(singular_values > _CARTESIAN_INTERFRAGMENT_PROJECTOR_TOLERANCE))
        irrep_span = vh[:rank]
        for operation, cartesian in zip(operations, cartesian_operations, strict=True):
            for gic, row in named_rows:
                transformed = row @ cartesian
                represented = (transformed @ irrep_span.T) @ irrep_span
                residual = float(np.linalg.norm(represented - transformed))
                if residual > tolerance:
                    failures.append((gic, operation.label, residual))
    return tuple(failures)


def _validate_cartesian_irrep_b_covariance(
    gics: tuple[FrozenGIC, ...],
    *,
    primitive_by_id: dict[str, GICPrimitive],
    point_group: str,
    reference_coordinates_angstrom: np.ndarray,
    operations: tuple[GICPointGroupOperation, ...],
    tolerance: float = 1.0e-7,
) -> None:
    """Require every projected Cartesian B row to remain in its irrep span."""

    failures = _cartesian_irrep_b_covariance_failures(
        gics,
        primitive_by_id=primitive_by_id,
        point_group=point_group,
        reference_coordinates_angstrom=reference_coordinates_angstrom,
        operations=operations,
        tolerance=tolerance,
    )
    if not failures:
        return
    gic, operation_label, residual = max(failures, key=lambda item: item[2])
    raise GICForgeContractError(
        "Cartesian Wilson B subspace is not covariant with "
        f"irrep {gic.irrep}: residual {residual:.6g} for {gic.name} under "
        f"{operation_label} (tolerance {tolerance:.3g})"
    )


def _point_group_has_only_one_dimensional_irreps(
    point_group: str,
    operations: tuple[GICPointGroupOperation, ...],
) -> bool:
    irreps = irrep_characters_for_operations(
        tuple(operation.label for operation in operations),
        point_group,
        operation_matrices=tuple(operation.rotation for operation in operations),
    )
    return bool(irreps) and all(
        characters and int(round(abs(float(characters[0])))) == 1 for _irrep, characters in irreps
    )


def _is_even_dnd_group(point_group: str) -> bool:
    group_key = point_group.strip().upper()
    order = group_key[1:-1] if group_key.startswith("D") and group_key.endswith("D") else ""
    return order.isdigit() and int(order) % 2 == 0


def _apply_rank_revealing_global_projector(
    gics: tuple[FrozenGIC, ...],
    primitives: tuple[GICPrimitive, ...],
    *,
    point_group: str,
    operations: tuple[GICPointGroupOperation, ...],
    reference_coordinates_angstrom: np.ndarray,
) -> tuple[tuple[FrozenGIC, ...], GICSymmetrizationDiagnostics] | None:
    primitive_by_id = {primitive.identifier: primitive for primitive in primitives}
    blocks: dict[tuple[str, str], list[FrozenGIC]] = {}
    for gic in gics:
        key = _symmetry_pool_key(gic.family)
        blocks.setdefault(key, []).append(gic)

    candidates: list[
        tuple[
            bool,
            bool,
            tuple[str, str],
            str,
            np.ndarray,
            tuple[GICPrimitive, ...],
            np.ndarray,
            int,
        ]
    ] = []
    operation_labels = tuple(operation.label for operation in operations)
    operation_matrices = tuple(operation.rotation for operation in operations)
    cartesian_projectors_by_irrep = dict(
        _cartesian_irrep_character_projectors(
            point_group=point_group,
            operations=operations,
            natoms=len(reference_coordinates_angstrom),
        )
    )
    cartesian_covariance_tolerance = _cartesian_covariance_tolerance(
        operations,
        reference_coordinates_angstrom=reference_coordinates_angstrom,
    )
    reference_b_cache: dict[str, np.ndarray] = {}
    sequence = 0
    for key, block_gics in sorted(
        blocks.items(),
        key=lambda item: _projector_block_sort_key(item, protect_special=True),
    ):
        block_primitives = _block_primitives_for_gics(
            tuple(block_gics),
            all_primitives=tuple(primitive_by_id.values()),
            primitive_by_id=primitive_by_id,
            key=key,
        )
        if block_primitives is None:
            continue
        primitive_index = _projector_primitive_index_with_aliases(
            block_primitives,
            primitive_by_id=primitive_by_id,
            key=key,
        )
        source_vectors = tuple(
            _gic_coefficient_vector(
                gic,
                primitive_index=primitive_index,
                vector_size=len(block_primitives),
            )
            for gic in block_gics
        )
        if any(vector is None for vector in source_vectors):
            continue
        source_vectors = tuple(
            np.asarray(vector, dtype=float) for vector in source_vectors if vector is not None
        )
        primitive_vectors = tuple(np.eye(len(block_primitives), dtype=float))
        primitive_key_index = _primitive_projector_key_index(block_primitives)
        if primitive_key_index is None:
            continue
        transforms = tuple(
            _operation_primitive_transform(
                block_primitives,
                operation=operation,
                primitive_key_index=primitive_key_index,
            )
            for operation in operations
        )
        if any(transform is None for transform in transforms):
            continue
        transform_stack = np.stack(
            [np.asarray(transform, dtype=float) for transform in transforms],
            axis=0,
        )
        try:
            block_b_rows: list[np.ndarray] = []
            for primitive in block_primitives:
                cached_b_row = reference_b_cache.get(primitive.identifier)
                if cached_b_row is None:
                    cached_b_row = _reference_tangent_b_row(
                        primitive,
                        reference_coordinates_angstrom,
                    )
                    reference_b_cache[primitive.identifier] = cached_b_row
                block_b_rows.append(cached_b_row)
            block_b_matrix = np.vstack(block_b_rows)
        except (FloatingPointError, GICForgeContractError, ValueError):
            block_b_matrix = None
        seen_vectors: set[tuple[float, ...]] = set()
        for irrep, characters in irrep_characters_for_operations(
            operation_labels,
            point_group,
            operation_matrices=operation_matrices,
        ):
            if len(characters) != len(operations):
                return None
            if all(abs(character) <= 1.0e-14 for character in characters):
                continue
            projector = np.tensordot(
                np.asarray(characters, dtype=float),
                transform_stack,
                axes=(0, 0),
            ) / float(len(transforms))
            for completion, vectors in ((False, source_vectors), (True, primitive_vectors)):
                projected_vectors = np.vstack(vectors) @ projector.T
                irrep_candidates = [
                    normalized
                    for projected in projected_vectors
                    if (
                        normalized := _normalized_coefficient_vector_or_none(projected)
                    )
                    is not None
                ]
                projection_candidates = (
                    _canonical_basis_for_span(irrep_candidates)
                    if irrep_dimension(irrep) > 1
                    else tuple(irrep_candidates)
                )
                for normalized in projection_candidates:
                    vector_key = tuple(round(float(value), 10) for value in normalized)
                    opposite_key = tuple(round(float(-value), 10) for value in normalized)
                    if vector_key in seen_vectors or opposite_key in seen_vectors:
                        continue
                    seen_vectors.add(vector_key)
                    b_row = (
                        normalized @ block_b_matrix
                        if block_b_matrix is not None
                        else _projected_vector_b_row(
                            block_primitives,
                            normalized,
                            coords=reference_coordinates_angstrom,
                        )
                    )
                    normalized_b = _normalized_cartesian_b_row_or_none(b_row)
                    if normalized_b is None:
                        continue
                    cartesian_projector = cartesian_projectors_by_irrep.get(irrep)
                    if cartesian_projector is None or float(
                        np.linalg.norm(normalized_b @ cartesian_projector - normalized_b)
                    ) > cartesian_covariance_tolerance:
                        # Symbolic primitive permutations can mislabel
                        # pseudoscalar rows.  Admit a candidate only when its
                        # actual Cartesian Wilson row transforms as the
                        # proposed molecular irrep.
                        continue
                    candidates.append(
                        (
                            completion,
                            primitive_reduction_class(key[1]) == SPECIAL_REDUCTION_CLASS,
                            key,
                            irrep,
                            normalized,
                            block_primitives,
                            normalized_b,
                            sequence,
                        )
                    )
                    sequence += 1
    if not candidates:
        return None

    selected: list[
        tuple[
            tuple[str, str],
            str,
            np.ndarray,
            tuple[GICPrimitive, ...],
            np.ndarray,
        ]
    ] = []
    selected_b_rows: list[np.ndarray] = []
    analytic_first = any(gic.family == "TS_REACTION_DISTANCE" for gic in gics)
    completion_count = 0
    irrep_quota_reselection = False
    if analytic_first:
        source_candidates = [candidate for candidate in candidates if not candidate[0]]
        completion_candidates = [candidate for candidate in candidates if candidate[0]]
        if not source_candidates:
            return None

        source_selection = select_rank_revealing_rows(
            np.vstack([candidate[6] for candidate in source_candidates]),
            target_rank=len(gics),
            tolerance=1.0e-8,
            priorities=tuple(0 if candidate[1] else 1 for candidate in source_candidates),
            tie_tolerance=1.0e-12,
        )
        for index in source_selection.indices:
            (
                _completion,
                _is_special,
                key,
                irrep,
                vector,
                block_primitives,
                b_row,
                _sequence,
            ) = source_candidates[index]
            selected.append((key, irrep, vector, block_primitives, b_row))
            selected_b_rows.append(b_row)

        missing = len(gics) - source_selection.rank
        completion_count = missing
        if missing > 0:
            if not completion_candidates or not selected_b_rows:
                return None
            _u, source_singular, source_vh = np.linalg.svd(
                np.vstack(selected_b_rows),
                full_matrices=False,
            )
            source_rank = int(
                np.sum(source_singular > _CARTESIAN_INTERFRAGMENT_PROJECTOR_TOLERANCE)
            )
            source_basis = list(source_vh[:source_rank])
            residual_rows = np.vstack(
                [
                    _b_row_residual_against_basis(candidate[6], source_basis)
                    for candidate in completion_candidates
                ]
            )
            completion_selection = select_rank_revealing_rows(
                residual_rows,
                target_rank=missing,
                tolerance=1.0e-8,
                priorities=tuple(
                    0 if candidate[1] else 1 for candidate in completion_candidates
                ),
                tie_tolerance=1.0e-12,
            )
            if completion_selection.rank != missing:
                return None
            for index in completion_selection.indices:
                (
                    _completion,
                    _is_special,
                    key,
                    irrep,
                    vector,
                    block_primitives,
                    b_row,
                    _sequence,
                ) = completion_candidates[index]
                selected.append((key, irrep, vector, block_primitives, b_row))
                selected_b_rows.append(b_row)
    else:
        rank_selection = select_rank_revealing_rows(
            np.vstack([candidate[6] for candidate in candidates]),
            target_rank=len(gics),
            tolerance=1.0e-8,
            priorities=tuple(0 if candidate[1] else 1 for candidate in candidates),
            tie_tolerance=1.0e-12,
        )
        if rank_selection.rank != len(gics):
            return None
        for index in rank_selection.indices:
            (
                _completion,
                _is_special,
                key,
                irrep,
                vector,
                block_primitives,
                b_row,
                _sequence,
            ) = candidates[index]
            selected.append((key, irrep, vector, block_primitives, b_row))
            selected_b_rows.append(b_row)

    expected_irrep_counts = _vibrational_irrep_coordinate_counts(
        point_group=point_group,
        operations=operations,
        reference_coordinates_angstrom=reference_coordinates_angstrom,
    )
    if (
        expected_irrep_counts is not None
        and sum(expected_irrep_counts.values()) == len(gics)
    ):
        selected_irrep_counts = {
            irrep: sum(candidate[1] == irrep for candidate in selected)
            for irrep in expected_irrep_counts
        }
        if selected_irrep_counts != expected_irrep_counts:
            quota_candidates = [
                (
                    candidate[3],
                    candidate[0],
                    candidate[1],
                    candidate,
                    candidate[6],
                    candidate[7],
                )
                for candidate in candidates
            ]
            quota_selection = _partition_quota_rank_revealing_selection(
                quota_candidates,
                quotas=dict(expected_irrep_counts),
            )
            if quota_selection is None and analytic_first:
                return None
            if quota_selection is not None:
                selected_candidates = [item[3] for item in quota_selection]
                selected = [
                    (
                        candidate[2],
                        candidate[3],
                        candidate[4],
                        candidate[5],
                        candidate[6],
                    )
                    for candidate in selected_candidates
                ]
                selected_b_rows = [candidate[6] for candidate in selected_candidates]
                completion_count = sum(bool(candidate[0]) for candidate in selected_candidates)
                irrep_quota_reselection = True

    if len(selected) != len(gics):
        return None
    if expected_irrep_counts is not None and sum(expected_irrep_counts.values()) == len(gics):
        final_irrep_counts = {
            irrep: sum(candidate[1] == irrep for candidate in selected)
            for irrep in expected_irrep_counts
        }
        if final_irrep_counts != expected_irrep_counts and analytic_first:
            raise GICForgeContractError(
                "global symmetry projector violated the molecular vibrational "
                f"irrep counts for {point_group}: expected {expected_irrep_counts}, "
                f"obtained {final_irrep_counts}"
            )

    output: list[FrozenGIC] = []
    name_counters: dict[tuple[str, str], int] = {}
    diagnostics_by_key: dict[tuple[str, str], list[str]] = {}
    for key, irrep, vector, block_primitives, _b_row in selected:
        _block, family = key
        coefficients = _coefficients_from_vector(block_primitives, vector)
        if not coefficients:
            return None
        coefficient_ids = {primitive_id for primitive_id, _ in coefficients}
        semantic_family = (
            "TS_REACTION_DISTANCE"
            if any(
                primitive.identifier in coefficient_ids
                and primitive.family == "TS_REACTION_DISTANCE"
                for primitive in block_primitives
            )
            else family
        )
        gic = FrozenGIC(
            identifier=f"GIC{len(output) + 1:03d}",
            name=_next_projected_name(semantic_family, irrep, name_counters),
            family=semantic_family,
            irrep=irrep,
            primitive_id=coefficients[0][0],
            gaussian_expression="LINEAR_COMBINATION",
            coefficients=coefficients,
        )
        output.append(gic)
        diagnostics_by_key.setdefault(key, []).append(gic.name)

    diagnostics = tuple(
        GICSymmetrizedGroup(
            block=key[0],
            family=key[1],
            signature=(
                (
                    f"ANALYTIC_FIRST_COMPLETION={completion_count};OPS="
                    if analytic_first
                    else "OPS="
                )
                + ",".join(operation_labels)
            ),
            source_gics=tuple(gic.name for gic in blocks[key]),
            output_gics=tuple(output_names),
        )
        for key, output_names in diagnostics_by_key.items()
    )
    output_tuple = tuple(output)
    output_rows = np.vstack(selected_b_rows)
    output_rank = numerical_matrix_rank(
        output_rows,
        absolute_tolerance=RANK_TOLERANCE,
    )
    if output_rank != len(output_tuple):
        raise GICForgeContractError(
            "global symmetry projector selected a rank-deficient Cartesian basis: "
            f"rank {output_rank} for {len(output_tuple)} coordinates"
        )
    covariance_failures = _cartesian_irrep_b_covariance_failures(
        output_tuple,
        primitive_by_id=primitive_by_id,
        point_group=point_group,
        reference_coordinates_angstrom=reference_coordinates_angstrom,
        operations=operations,
        tolerance=_cartesian_covariance_tolerance(
            operations,
            reference_coordinates_angstrom=reference_coordinates_angstrom,
        ),
    )
    # The symbolic primitive permutation can assign a conventional irrep
    # label that disagrees with the Cartesian Wilson-row transformation,
    # notably for pseudoscalar ring-puckering rows in groups with
    # multidimensional irreps (for example C3v).  The Cartesian projector is
    # the authoritative representation here and audits its repaired result,
    # so allow the same family-local repair for these blocks as well.
    family_repair_allowed = bool(covariance_failures)
    if family_repair_allowed:
        repaired = _apply_cartesian_interfragment_projector(
            output_tuple,
            primitives,
            point_group=point_group,
            operations=operations,
            reference_coordinates_angstrom=reference_coordinates_angstrom,
            repair_keys=frozenset(
                (
                    primitive_symmetry_block(failure[0].family),
                    failure[0].family,
                )
                for failure in covariance_failures
            ),
        )
        if repaired is None:
            _validate_cartesian_irrep_b_covariance(
                output_tuple,
                primitive_by_id=primitive_by_id,
                point_group=point_group,
                reference_coordinates_angstrom=reference_coordinates_angstrom,
                operations=operations,
                tolerance=_cartesian_covariance_tolerance(
                    operations,
                    reference_coordinates_angstrom=reference_coordinates_angstrom,
                ),
            )
            raise AssertionError("unreachable Cartesian covariance audit")
        return repaired
    if covariance_failures:
        _validate_cartesian_irrep_b_covariance(
            output_tuple,
            primitive_by_id=primitive_by_id,
            point_group=point_group,
            reference_coordinates_angstrom=reference_coordinates_angstrom,
            operations=operations,
            tolerance=_cartesian_covariance_tolerance(
                operations,
                reference_coordinates_angstrom=reference_coordinates_angstrom,
            ),
        )
    return (
        output_tuple,
        GICSymmetrizationDiagnostics(
            method=POINT_GROUP_PROJECTOR_METHOD,
            policy=PROJECTOR_SYMMETRIZATION_POLICY,
            status=(
                "APPLIED_VIBRATIONAL_IRREP_QUOTAS"
                if irrep_quota_reselection
                else (
                    "APPLIED_ANALYTIC_FIRST"
                    if completion_count == 0
                    else "APPLIED_ANALYTIC_FIRST_MINIMAL_COMPLETION"
                )
                if analytic_first
                else "APPLIED"
            ),
            point_group=point_group,
            symmetry_group=point_group,
            total_symmetric_irrep=total_symmetric_irrep(point_group),
            total_symmetric_gics=tuple(
                gic.name for gic in output_tuple if is_total_symmetric_irrep(point_group, gic.irrep)
            ),
            groups=diagnostics,
            fallback_events=(
                (
                    make_fallback_event(
                        stage="SMITH_SYMMETRIZATION",
                        algorithm_id="GLOBAL_RANK_REVEALING_PROJECTOR",
                        trigger=(
                            "ANALYTIC_PROJECTOR_REQUIRED_GLOBAL_EXACT_RANK_COMPLETION"
                            if completion_count
                            else "ANALYTIC_PROJECTOR_SELECTED_GLOBAL_EXACT_RANK_BASIS"
                        ),
                        rank_before=len(output_tuple) - completion_count,
                        rank_after=len(output_tuple),
                        source=(
                            "GLOBAL_RANK_REVEALING_PROJECTOR "
                            f"COMPLETION={completion_count}"
                        ),
                    ),
                )
                if analytic_first
                else ()
            ),
        ),
    )


def _b_row_residual_against_basis(
    row: np.ndarray,
    q_basis: list[np.ndarray],
) -> np.ndarray:
    residual = np.array(row, dtype=float, copy=True)
    for _pass in range(2):
        for basis_row in q_basis:
            residual -= float(np.dot(residual, basis_row)) * basis_row
    return residual


def _projector_block_sort_key(
    item: tuple[tuple[str, str], list[FrozenGIC]],
    *,
    protect_special: bool = False,
) -> tuple[int, int]:
    _block, family = item[0]
    protected_family = (
        primitive_reduction_class(family) == SPECIAL_REDUCTION_CLASS
        or family in {"RING_PUCKER_COMPONENT", "CYCLIC_BEND", "BUTTERFLY", "SPIRO_BEND"}
    )
    if protect_special and protected_family:
        try:
            order = PRIMITIVE_FAMILY_ORDER.index(family)
        except ValueError:
            order = len(PRIMITIVE_FAMILY_ORDER)
        return (-1, order)
    priority = {
        "STRETCH": 0,
        "BEND": 1,
        "BUTTERFLY": 2,
        "CYCLIC_BEND": 3,
    }
    if family in priority:
        return (priority[family], len(item[1]))
    try:
        order = PRIMITIVE_FAMILY_ORDER.index(family)
    except ValueError:
        order = len(PRIMITIVE_FAMILY_ORDER)
    return (10 + order, len(item[1]))


def _valid_projector_operations(
    operations: tuple[GICPointGroupOperation, ...],
) -> tuple[GICPointGroupOperation, ...]:
    if not operations:
        return ()
    natoms = len(operations[0].permutation)
    expected = tuple(range(1, natoms + 1))
    if natoms == 0:
        return ()
    validated: list[GICPointGroupOperation] = []
    seen: set[tuple[tuple[int, ...], tuple[float, ...]]] = set()
    for operation in operations:
        if len(operation.permutation) != natoms:
            return ()
        if tuple(sorted(operation.permutation)) != expected:
            return ()
        unique_key = (
            operation.permutation,
            tuple(round(float(value), 10) for row in operation.rotation for value in row),
        )
        if unique_key in seen:
            continue
        seen.add(unique_key)
        validated.append(operation)
    identity_index = next(
        (
            idx
            for idx, operation in enumerate(validated)
            if operation.label == "E" and operation.permutation == expected
        ),
        None,
    )
    if identity_index is None:
        return ()
    if identity_index:
        identity = validated.pop(identity_index)
        validated.insert(0, identity)
    return tuple(validated)


def _diagnostics_with_operation_margins(
    diagnostics: GICSymmetrizationDiagnostics,
    *,
    operations: tuple[GICPointGroupOperation, ...],
    reference_coordinates_angstrom: tuple[tuple[float, float, float], ...] | None,
) -> GICSymmetrizationDiagnostics:
    operation_margins = _symmetry_operation_margins(
        operations,
        reference_coordinates_angstrom=reference_coordinates_angstrom,
    )
    if operation_margins is None:
        return diagnostics
    max_residual, min_margin, near_threshold = operation_margins
    return replace(
        diagnostics,
        operation_tolerance_angstrom=SYMMETRY_OPERATION_TOLERANCE_ANGSTROM,
        max_operation_residual_angstrom=max_residual,
        min_operation_margin_angstrom=min_margin,
        near_threshold_operations=near_threshold,
    )


def _symmetry_operation_margins(
    operations: tuple[GICPointGroupOperation, ...],
    *,
    reference_coordinates_angstrom: tuple[tuple[float, float, float], ...] | None,
) -> tuple[float, float, tuple[str, ...]] | None:
    validated = _valid_projector_operations(operations)
    if not validated or reference_coordinates_angstrom is None:
        return None
    coords = np.asarray(reference_coordinates_angstrom, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] != len(validated[0].permutation):
        return None
    residuals: list[tuple[str, float]] = []
    for operation in validated:
        rotation = np.asarray(operation.rotation, dtype=float)
        target = coords[np.asarray(operation.permutation, dtype=int) - 1]
        rotated_right = coords @ rotation.T
        rotated_left = coords @ rotation
        residual = min(
            float(np.max(np.linalg.norm(rotated_right - target, axis=1))),
            float(np.max(np.linalg.norm(rotated_left - target, axis=1))),
        )
        residuals.append((operation.label, residual))
    if not residuals:
        return None
    tolerance = float(SYMMETRY_OPERATION_TOLERANCE_ANGSTROM)
    max_residual = max(residual for _label, residual in residuals)
    margins = tuple(tolerance - residual for _label, residual in residuals)
    min_margin = min(margins)
    near_width = tolerance * float(SYMMETRY_OPERATION_NEAR_THRESHOLD_FRACTION)
    near_threshold = tuple(
        f"{label}:{residual:.6g}"
        for label, residual in residuals
        if tolerance - residual <= near_width
    )
    return max_residual, min_margin, near_threshold


def _symmetry_closed_projector_primitives(
    primitives: tuple[GICPrimitive, ...],
    *,
    symmetry_operations: tuple[GICPointGroupOperation, ...],
    include_cartesian_interfragment_orbits: bool = False,
) -> tuple[GICPrimitive, ...]:
    operations = _valid_projector_operations(symmetry_operations)
    if len(operations) <= 1:
        return primitives

    fragment_ids_by_membership = _projector_fragment_ids_by_membership(primitives)
    output = list(primitives)
    used_identifiers = {primitive.identifier for primitive in output}
    next_index = (
        max(
            (
                int(primitive.identifier[1:])
                for primitive in output
                if primitive.identifier.startswith("P") and primitive.identifier[1:].isdigit()
            ),
            default=0,
        )
        + 1
    )
    keys = {
        key
        for primitive in output
        if (
            key := _projector_orbit_key(
                primitive,
                include_cartesian_interfragment_orbits=include_cartesian_interfragment_orbits,
            )
        )
        is not None
    }
    cursor = 0
    while cursor < len(output):
        primitive = output[cursor]
        cursor += 1
        for operation in operations:
            mapped = _mapped_projector_primitive(
                primitive,
                operation,
                include_cartesian_interfragment_orbits=include_cartesian_interfragment_orbits,
                fragment_ids_by_membership=fragment_ids_by_membership,
            )
            if mapped is None:
                continue
            key = _projector_orbit_key(
                mapped,
                include_cartesian_interfragment_orbits=include_cartesian_interfragment_orbits,
            )
            if key is None or key in keys:
                continue
            keys.add(key)
            while f"P{next_index:03d}" in used_identifiers:
                next_index += 1
            identifier = f"P{next_index:03d}"
            used_identifiers.add(identifier)
            next_index += 1
            output.append(
                replace(
                    mapped,
                    identifier=identifier,
                    name=identifier,
                )
            )
    return tuple(output)


def _mapped_projector_primitive(
    primitive: GICPrimitive,
    operation: GICPointGroupOperation,
    *,
    include_cartesian_interfragment_orbits: bool = False,
    fragment_ids_by_membership: dict[frozenset[int], str] | None = None,
) -> GICPrimitive | None:
    if primitive.family == "LINEAR_BEND" and not include_cartesian_interfragment_orbits:
        return None
    if (
        primitive.family in {"FRAG_TRANSLATION", "FRAG_ORIENTATION"}
        and not include_cartesian_interfragment_orbits
    ):
        return None

    mapped_atoms = tuple(_mapped_atom(operation, atom) for atom in primitive.atoms)
    mapped_refs = tuple(_mapped_atom(operation, atom) for atom in primitive.ref_atoms)
    mapped_frame = tuple(_mapped_atom(operation, atom) for atom in primitive.frame_atoms)
    mapped_ref_frame = tuple(_mapped_atom(operation, atom) for atom in primitive.ref_frame_atoms)
    if any(atom < 1 for atom in mapped_atoms + mapped_refs + mapped_frame + mapped_ref_frame):
        return None
    mapped_symbolic_refs = primitive.refs
    if primitive.function in FRAGMENT_BODY_PAIR_FUNCTIONS:
        if len(primitive.refs) < 2 or fragment_ids_by_membership is None:
            return None
        moving_id = fragment_ids_by_membership.get(frozenset(mapped_atoms))
        reference_id = fragment_ids_by_membership.get(frozenset(mapped_refs))
        if moving_id is None or reference_id is None:
            return None
        mapped_symbolic_refs = (
            moving_id,
            reference_id,
            *primitive.refs[2:],
        )
    return replace(
        primitive,
        atoms=mapped_atoms,
        ref_atoms=mapped_refs,
        frame_atoms=mapped_frame,
        ref_frame_atoms=mapped_ref_frame,
        refs=mapped_symbolic_refs,
    )


def _projector_fragment_ids_by_membership(
    primitives: tuple[GICPrimitive, ...],
) -> dict[frozenset[int], str]:
    """Index frozen body IDs so symmetry operations also transform typed refs."""

    result: dict[frozenset[int], str] = {}
    for primitive in primitives:
        if (
            primitive.function not in FRAGMENT_BODY_PAIR_FUNCTIONS
            or len(primitive.refs) < 2
        ):
            continue
        for fragment_id, atoms in (
            (primitive.refs[0], primitive.atoms),
            (primitive.refs[1], primitive.ref_atoms),
        ):
            membership = frozenset(int(atom) for atom in atoms)
            if not fragment_id or not membership:
                raise GICForgeContractError(
                    f"invalid typed fragment reference {fragment_id!r}"
                )
            previous = result.setdefault(membership, fragment_id)
            if previous != fragment_id:
                raise GICForgeContractError(
                    "distinct fragment IDs have identical atom membership"
                )
    return result


def _projector_orbit_key(
    primitive: GICPrimitive,
    *,
    include_cartesian_interfragment_orbits: bool,
) -> tuple[object, ...] | None:
    """Return the symbolic-orbit key, preserving fragment-frame gauge when required."""

    key = _primitive_projector_key(primitive)
    if key is None:
        return None
    if include_cartesian_interfragment_orbits and primitive.family in {
        "FRAG_TRANSLATION",
        "FRAG_ORIENTATION",
    }:
        return (*key, primitive.frame_atoms, primitive.ref_frame_atoms)
    return key


def _project_gic_block(
    key: tuple[str, str],
    gics: tuple[FrozenGIC, ...],
    *,
    primitive_by_id: dict[str, GICPrimitive],
    operations: tuple[GICPointGroupOperation, ...],
    point_group: str,
    first_index: int,
    name_counters: dict[tuple[str, str], int],
    reference_coordinates_angstrom: np.ndarray | None = None,
    ring_pucker_coordinates_angstrom: np.ndarray | None = None,
    global_b_basis: tuple[np.ndarray, ...] = (),
    reference_b_cache: dict[str, np.ndarray] | None = None,
) -> tuple[tuple[FrozenGIC, ...], GICSymmetrizedGroup, tuple[np.ndarray, ...]] | None:
    block, family = key
    block_primitives = _block_primitives_for_gics(
        gics,
        all_primitives=tuple(primitive_by_id.values()),
        primitive_by_id=primitive_by_id,
        key=key,
    )
    if block_primitives is None:
        return None

    primitive_index = _projector_primitive_index_with_aliases(
        block_primitives,
        primitive_by_id=primitive_by_id,
        key=key,
    )
    source_vectors = tuple(
        _gic_coefficient_vector(
            gic,
            primitive_index=primitive_index,
            vector_size=len(block_primitives),
        )
        for gic in gics
    )
    if any(vector is None for vector in source_vectors):
        return None

    source_matrix = np.vstack(
        [np.asarray(vector, dtype=float) for vector in source_vectors if vector is not None]
    )
    block_b_matrix: np.ndarray | None = None
    if reference_coordinates_angstrom is not None:
        cache = reference_b_cache if reference_b_cache is not None else {}
        try:
            block_b_rows: list[np.ndarray] = []
            for primitive in block_primitives:
                cached_b_row = cache.get(primitive.identifier)
                if cached_b_row is None:
                    cached_b_row = _reference_tangent_b_row(
                        primitive,
                        reference_coordinates_angstrom,
                    )
                    cache[primitive.identifier] = cached_b_row
                block_b_rows.append(cached_b_row)
            block_b_matrix = np.vstack(block_b_rows)
        except (FloatingPointError, GICForgeContractError, ValueError):
            # Non-standard primitives retain the scalar evaluator below.
            block_b_matrix = None

    ring_brow_projected: (
        tuple[tuple[FrozenGIC, ...], GICSymmetrizedGroup, tuple[np.ndarray, ...]] | None
    ) = None
    ring_brow_counters: dict[tuple[str, str], int] | None = None
    ring_brow_attempted = False

    def ring_brow_result() -> (
        tuple[tuple[FrozenGIC, ...], GICSymmetrizedGroup, tuple[np.ndarray, ...]] | None
    ):
        nonlocal ring_brow_attempted, ring_brow_projected, ring_brow_counters
        if ring_brow_attempted:
            return ring_brow_projected
        ring_brow_attempted = True
        if (
            family != "RING_PUCKER_COMPONENT"
            or ring_pucker_coordinates_angstrom is None
            or len(source_vectors) != len(gics)
        ):
            return None
        candidate_counters = dict(name_counters)
        ring_brow_projected = _project_ring_pucker_source_block(
            key,
            gics,
            block_primitives,
            tuple(
                np.asarray(vector, dtype=float) for vector in source_vectors if vector is not None
            ),
            operations=operations,
            point_group=point_group,
            first_index=first_index,
            name_counters=candidate_counters,
            reference_coordinates_angstrom=ring_pucker_coordinates_angstrom,
            global_b_basis=global_b_basis,
            block_b_matrix=block_b_matrix,
        )
        if ring_brow_projected is not None:
            ring_brow_counters = candidate_counters
        return ring_brow_projected

    def use_result(
        result: tuple[tuple[FrozenGIC, ...], GICSymmetrizedGroup, tuple[np.ndarray, ...]],
        counters: dict[tuple[str, str], int] | None,
    ) -> tuple[tuple[FrozenGIC, ...], GICSymmetrizedGroup, tuple[np.ndarray, ...]]:
        if counters is not None:
            name_counters.clear()
            name_counters.update(counters)
        return result

    primitive_key_index = _primitive_projector_key_index(block_primitives)
    if primitive_key_index is None:
        projected_ring = ring_brow_result()
        return use_result(projected_ring, ring_brow_counters) if projected_ring else None
    transforms = tuple(
        _operation_primitive_transform(
            block_primitives,
            operation=operation,
            primitive_key_index=primitive_key_index,
        )
        for operation in operations
    )
    if any(transform is None for transform in transforms):
        projected_ring = ring_brow_result()
        return use_result(projected_ring, ring_brow_counters) if projected_ring else None
    transform_stack = np.stack(
        [np.asarray(transform, dtype=float) for transform in transforms],
        axis=0,
    )

    projected_vectors: list[tuple[str, np.ndarray]] = []
    basis: list[np.ndarray] = []
    b_basis = list(global_b_basis)
    operation_labels = tuple(operation.label for operation in operations)
    operation_matrices = tuple(operation.rotation for operation in operations)
    for irrep, characters in irrep_characters_for_operations(
        operation_labels,
        point_group,
        operation_matrices=operation_matrices,
    ):
        if len(characters) != len(operations):
            return None
        if all(abs(character) <= 1.0e-14 for character in characters):
            continue
        # For polyhedral groups, canonicalize the projected source span before
        # selecting rows by Cartesian rank.  Incoming GICs can carry an
        # arbitrary basis inside a degenerate SALC subspace (for example from
        # a degenerate inertia eigensystem); the span is physical, that basis
        # is not.
        canonical_projected_basis = irrep_dimension(irrep) > 1 or point_group.strip().upper() in {
            "T",
            "TD",
            "TH",
            "O",
            "OH",
            "I",
            "IH",
        }
        projector = np.tensordot(
            np.asarray(characters, dtype=float),
            transform_stack,
            axes=(0, 0),
        ) / float(len(transforms))
        projected_source_matrix = source_matrix @ projector.T
        irrep_candidates: list[np.ndarray] = []
        for projected in projected_source_matrix:
            normalized = _normalized_coefficient_vector_or_none(projected)
            if normalized is None:
                continue
            irrep_candidates.append(normalized)
        projection_candidates = (
            _canonical_basis_for_span(irrep_candidates)
            if canonical_projected_basis
            else tuple(irrep_candidates)
        )
        for normalized in projection_candidates:
            independent = _orthonormal_coefficient_residual_or_none(basis, normalized)
            if independent is None:
                continue
            independent = _canonical_vector_sign(independent)
            b_independent = None
            if reference_coordinates_angstrom is not None:
                b_row = (
                    independent @ block_b_matrix
                    if block_b_matrix is not None
                    else _projected_vector_b_row(
                        block_primitives,
                        independent,
                        coords=reference_coordinates_angstrom,
                    )
                )
                normalized_b = _normalized_coefficient_vector_or_none(b_row)
                if normalized_b is None:
                    continue
                b_independent = _orthonormal_coefficient_residual_or_none(
                    b_basis,
                    normalized_b,
                )
                if b_independent is None:
                    continue
            basis.append(independent)
            if b_independent is not None:
                b_basis.append(b_independent)
            projected_vectors.append((irrep, independent))
            if len(projected_vectors) == len(gics):
                break
        if len(projected_vectors) == len(gics):
            break
    if len(projected_vectors) != len(gics):
        projected_ring = ring_brow_result()
        return use_result(projected_ring, ring_brow_counters) if projected_ring else None

    primitive_counters = dict(name_counters)
    output: list[FrozenGIC] = []
    for offset, (irrep, vector) in enumerate(projected_vectors):
        coefficients = _coefficients_from_vector(block_primitives, vector)
        if not coefficients:
            return None
        semantic_family = _semantic_family_for_projected_coefficients(
            family, coefficients, block_primitives
        )
        output.append(
            FrozenGIC(
                identifier=f"GIC{first_index + offset:03d}",
                name=_next_projected_name(semantic_family, irrep, primitive_counters),
                family=semantic_family,
                irrep=irrep,
                primitive_id=coefficients[0][0],
                gaussian_expression="LINEAR_COMBINATION",
                coefficients=coefficients,
            )
        )

    primitive_result = (
        tuple(output),
        GICSymmetrizedGroup(
            block=block,
            family=family,
            signature="OPS=" + ",".join(operation_labels),
            source_gics=tuple(gic.name for gic in gics),
            output_gics=tuple(gic.name for gic in output),
        ),
        tuple(b_basis),
    )
    projected_ring = ring_brow_result()
    # A minimal triangular-flap chart is closed under the point group through
    # linear combinations of its B rows, not through a literal permutation of
    # one privileged triangulation.  The B-row representation is therefore
    # the physically relevant projector whenever it is available.  Requiring
    # agreement with the overcomplete symbolic primitive orbit would reinsert
    # triangulation-dependent, spurious irreps.
    uses_intrinsic_ring_sources = family == "RING_PUCKER_COMPONENT" and any(
        primitive.function in {"D", "U"} for primitive in block_primitives
    )
    if projected_ring is not None and (
        uses_intrinsic_ring_sources
        or _same_projected_irrep_sequence(projected_ring, primitive_result)
    ):
        return use_result(projected_ring, ring_brow_counters)
    return use_result(primitive_result, primitive_counters)


def _semantic_family_for_projected_coefficients(
    pool_family: str,
    coefficients: tuple[tuple[str, float], ...],
    block_primitives: tuple[GICPrimitive, ...],
) -> str:
    """Keep ORACLE reaction-distance ownership inside a shared radial pool."""

    coefficient_ids = {primitive_id for primitive_id, _ in coefficients}
    if any(
        primitive.identifier in coefficient_ids
        and primitive.family == "TS_REACTION_DISTANCE"
        for primitive in block_primitives
    ):
        return "TS_REACTION_DISTANCE"
    return pool_family


def _same_projected_irrep_sequence(
    left: tuple[tuple[FrozenGIC, ...], GICSymmetrizedGroup, tuple[np.ndarray, ...]],
    right: tuple[tuple[FrozenGIC, ...], GICSymmetrizedGroup, tuple[np.ndarray, ...]],
) -> bool:
    return tuple(gic.irrep for gic in left[0]) == tuple(gic.irrep for gic in right[0])


def _project_ring_pucker_source_block(
    key: tuple[str, str],
    block_gics: tuple[FrozenGIC, ...],
    block_primitives: tuple[GICPrimitive, ...],
    source_vectors: tuple[np.ndarray, ...],
    *,
    operations: tuple[GICPointGroupOperation, ...],
    point_group: str,
    first_index: int,
    name_counters: dict[tuple[str, str], int],
    reference_coordinates_angstrom: np.ndarray,
    global_b_basis: tuple[np.ndarray, ...],
    block_b_matrix: np.ndarray | None = None,
) -> tuple[tuple[FrozenGIC, ...], GICSymmetrizedGroup, tuple[np.ndarray, ...]] | None:
    if not source_vectors:
        return None
    try:
        source_b_rows = (
            np.vstack(source_vectors) @ block_b_matrix
            if block_b_matrix is not None
            else np.vstack(
                [
                    _projected_vector_b_row(
                        block_primitives,
                        source_vector,
                        coords=reference_coordinates_angstrom,
                    )
                    for source_vector in source_vectors
                ]
            )
        )
    except (FloatingPointError, ValueError):
        return None
    if numerical_matrix_rank(
        source_b_rows,
        absolute_tolerance=1.0e-8,
    ) < len(source_vectors):
        return None
    transforms = tuple(
        _source_b_row_transform(
            source_b_rows,
            operation=operation,
            natoms=len(reference_coordinates_angstrom),
        )
        for operation in operations
    )
    if any(transform is None for transform in transforms):
        return None

    projected_vectors: list[tuple[str, np.ndarray]] = []
    source_basis: list[np.ndarray] = []
    b_basis = list(global_b_basis)
    operation_labels = tuple(operation.label for operation in operations)
    operation_matrices = tuple(operation.rotation for operation in operations)
    source_units = tuple(np.eye(len(source_vectors), dtype=float))
    for irrep, characters in irrep_characters_for_operations(
        operation_labels,
        point_group,
        operation_matrices=operation_matrices,
    ):
        if len(characters) != len(operations):
            return None
        if all(abs(character) <= 1.0e-14 for character in characters):
            continue
        for source_unit in source_units:
            projected_source = _project_vector_for_irrep(
                source_unit,
                characters=characters,
                transforms=transforms,
            )
            normalized_source = _normalized_coefficient_vector_or_none(projected_source)
            if normalized_source is None:
                continue
            independent_source = _orthonormal_coefficient_residual_or_none(
                source_basis,
                normalized_source,
            )
            if independent_source is None:
                continue
            primitive_vector = np.zeros_like(source_vectors[0], dtype=float)
            for source_coefficient, source_vector in zip(independent_source, source_vectors):
                primitive_vector += float(source_coefficient) * source_vector
            try:
                b_row = _projected_vector_b_row(
                    block_primitives,
                    primitive_vector,
                    coords=reference_coordinates_angstrom,
                )
            except (FloatingPointError, ValueError):
                return None
            normalized_b = _normalized_coefficient_vector_or_none(b_row)
            if normalized_b is None:
                continue
            b_independent = _orthonormal_coefficient_residual_or_none(b_basis, normalized_b)
            if b_independent is None:
                continue
            source_basis.append(independent_source)
            b_basis.append(b_independent)
            projected_vectors.append((irrep, primitive_vector))
            if len(projected_vectors) == len(block_gics):
                break
        if len(projected_vectors) == len(block_gics):
            break
    if len(projected_vectors) != len(block_gics):
        return None

    _block, family = key
    output: list[FrozenGIC] = []
    for offset, (irrep, primitive_vector) in enumerate(projected_vectors):
        normalized = _normalized_coefficient_vector_or_none(primitive_vector)
        if normalized is None:
            return None
        coefficients = _coefficients_from_vector(block_primitives, normalized)
        if not coefficients:
            return None
        semantic_family = _semantic_family_for_projected_coefficients(
            family, coefficients, block_primitives
        )
        output.append(
            FrozenGIC(
                identifier=f"GIC{first_index + offset:03d}",
                name=_next_projected_name(semantic_family, irrep, name_counters),
                family=semantic_family,
                irrep=irrep,
                primitive_id=coefficients[0][0],
                gaussian_expression="LINEAR_COMBINATION",
                coefficients=coefficients,
            )
        )
    return (
        tuple(output),
        GICSymmetrizedGroup(
            block=key[0],
            family=family,
            signature="BROWS_OPS=" + ",".join(operation_labels),
            source_gics=tuple(gic.name for gic in block_gics),
            output_gics=tuple(gic.name for gic in output),
        ),
        tuple(b_basis),
    )


def _physical_ring_projector_candidates(
    block_primitives: tuple[GICPrimitive, ...],
    source_vectors: tuple[np.ndarray, ...],
    *,
    block_b_matrix: np.ndarray,
    operations: tuple[GICPointGroupOperation, ...],
    point_group: str,
) -> tuple[tuple[str, np.ndarray, np.ndarray], ...] | None:
    """Project an intrinsic ring chart by its physical Cartesian tangents.

    Symbolic primitive orbits may contain formal projected components whose
    Cartesian tangent is nearly null.  Their inclusion can satisfy a global
    irrep quota while replacing a real ring mode.  The source B rows carry
    the actual finite-dimensional representation and therefore determine the
    admissible multiplicity of each irrep.
    """

    if not source_vectors:
        return None
    source_b_rows = np.vstack(source_vectors) @ block_b_matrix
    if numerical_matrix_rank(source_b_rows, absolute_tolerance=1.0e-8) < len(
        source_vectors
    ):
        return None
    transforms = tuple(
        _source_b_row_transform(
            source_b_rows,
            operation=operation,
            natoms=source_b_rows.shape[1] // 3,
        )
        for operation in operations
    )
    if any(transform is None for transform in transforms):
        return None
    physical_transforms = tuple(
        np.asarray(transform, dtype=float)
        for transform in transforms
        if transform is not None
    )
    source_basis: list[np.ndarray] = []
    candidates: list[tuple[str, np.ndarray, np.ndarray]] = []
    source_units = tuple(np.eye(len(source_vectors), dtype=float))
    operation_labels = tuple(operation.label for operation in operations)
    operation_matrices = tuple(operation.rotation for operation in operations)
    for irrep, characters in irrep_characters_for_operations(
        operation_labels,
        point_group,
        operation_matrices=operation_matrices,
    ):
        if len(characters) != len(operations):
            return None
        if all(abs(character) <= 1.0e-14 for character in characters):
            continue
        for source_unit in source_units:
            projected_source = _project_vector_for_irrep(
                source_unit,
                characters=characters,
                transforms=physical_transforms,
            )
            normalized_source = _normalized_coefficient_vector_or_none(projected_source)
            if normalized_source is None:
                continue
            independent_source = _orthonormal_coefficient_residual_or_none(
                source_basis,
                normalized_source,
            )
            if independent_source is None:
                continue
            primitive_vector = np.zeros_like(source_vectors[0], dtype=float)
            for coefficient, source_vector in zip(independent_source, source_vectors):
                primitive_vector += float(coefficient) * source_vector
            normalized_vector = _normalized_coefficient_vector_or_none(primitive_vector)
            if normalized_vector is None:
                continue
            b_row = normalized_vector @ block_b_matrix
            normalized_b = _normalized_cartesian_b_row_or_none(b_row)
            if normalized_b is None:
                continue
            source_basis.append(independent_source)
            candidates.append((irrep, _canonical_vector_sign(normalized_vector), normalized_b))
    if len(candidates) != len(source_vectors):
        return None
    return tuple(candidates)


def _ring_source_domains_are_disjoint(
    block_primitives: tuple[GICPrimitive, ...],
    source_vectors: tuple[np.ndarray, ...],
) -> bool:
    """Return whether intrinsic puckering sources belong to disjoint rings.

    Fused rings share atoms and their special coordinates must remain in the
    existing joint endocyclic selection.  A single ring and multiple separate
    rings instead carry independent physical B-row representations.
    """

    domains: list[frozenset[int]] = []
    for vector in source_vectors:
        atoms = frozenset(
            atom
            for primitive, coefficient in zip(block_primitives, vector)
            if abs(float(coefficient)) > 1.0e-12
            for atom in primitive.atoms
        )
        if not atoms:
            return False
        if atoms not in domains:
            domains.append(atoms)
    return len(domains) > 1 and all(
        not left.intersection(right)
        for index, left in enumerate(domains)
        for right in domains[index + 1 :]
    )


def _source_b_row_transform(
    source_b_rows: np.ndarray,
    *,
    operation: GICPointGroupOperation,
    natoms: int,
) -> np.ndarray | None:
    cartesian = _cartesian_operation_matrix(operation, natoms=natoms)
    matrix = np.zeros((source_b_rows.shape[0], source_b_rows.shape[0]), dtype=float)
    for source_index, row in enumerate(source_b_rows):
        transformed = row @ cartesian
        values, *_ = np.linalg.lstsq(source_b_rows.T, transformed.T, rcond=1.0e-10)
        residual = float(np.linalg.norm(values @ source_b_rows - transformed))
        if residual > 1.0e-8:
            return None
        matrix[:, source_index] = values
    return matrix


def _cartesian_operation_matrix(operation: GICPointGroupOperation, *, natoms: int) -> np.ndarray:
    return cartesian_operation_matrix(
        operation.rotation,
        tuple(int(target_atom) - 1 for target_atom in operation.permutation),
        natoms=natoms,
    )


def _projected_vector_b_row(
    primitives: tuple[GICPrimitive, ...],
    vector: np.ndarray,
    *,
    coords: np.ndarray,
) -> np.ndarray:
    row = np.zeros(coords.size, dtype=float)
    for primitive, coefficient in zip(primitives, vector):
        if abs(float(coefficient)) <= 1.0e-12:
            continue
        row += float(coefficient) * _reference_tangent_b_row(primitive, coords)
    return row


def _reference_tangent_b_row(
    primitive: GICPrimitive,
    coords: np.ndarray,
) -> np.ndarray:
    """Evaluate a primitive tangent at the frozen reference geometry."""

    return _analytic_b_row(
        primitive,
        coords,
        reference_coords=coords if primitive.function == "FROT" else None,
    )




def _block_primitives_for_gics(
    gics: tuple[FrozenGIC, ...],
    *,
    all_primitives: tuple[GICPrimitive, ...],
    primitive_by_id: dict[str, GICPrimitive],
    key: tuple[str, str],
) -> tuple[GICPrimitive, ...] | None:
    block, family = key
    ordered: list[GICPrimitive] = []
    index_by_key: dict[tuple[object, ...], int] = {}
    included_ids: set[str] = set()
    required_ids = {
        primitive_id
        for gic in gics
        for primitive_id, _coefficient in (gic.coefficients or ((gic.primitive_id, 1.0),))
    }

    def add_primitive(primitive: GICPrimitive) -> bool:
        primitive_key = _symmetry_pool_key(primitive.family)
        if primitive_key != (block, family):
            return True
        projector_key = _primitive_projector_key(primitive)
        if projector_key is None:
            return False
        duplicate_index = index_by_key.get(projector_key)
        if duplicate_index is None:
            index_by_key[projector_key] = len(ordered)
            included_ids.add(primitive.identifier)
            ordered.append(primitive)
            return True
        existing = ordered[duplicate_index]
        if existing.identifier == primitive.identifier:
            return True
        existing_required = existing.identifier in required_ids
        current_required = primitive.identifier in required_ids
        if current_required and existing_required:
            return True
        if current_required and not existing_required:
            included_ids.discard(existing.identifier)
            included_ids.add(primitive.identifier)
            ordered[duplicate_index] = primitive
        return True

    for gic in gics:
        coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
        for primitive_id, _coefficient in coefficients:
            primitive = primitive_by_id.get(primitive_id)
            if primitive is None:
                return None
            if not add_primitive(primitive):
                return None

    for primitive in all_primitives:
        if primitive.identifier in included_ids:
            continue
        if not add_primitive(primitive):
            return None
    return tuple(ordered)


def _projector_primitive_index_with_aliases(
    block_primitives: tuple[GICPrimitive, ...],
    *,
    primitive_by_id: dict[str, GICPrimitive],
    key: tuple[str, str],
) -> dict[str, int]:
    block, family = key
    key_to_index = {
        _primitive_projector_key(primitive): index
        for index, primitive in enumerate(block_primitives)
    }
    index = {
        primitive.identifier: primitive_index
        for primitive_index, primitive in enumerate(block_primitives)
    }
    for primitive in primitive_by_id.values():
        primitive_key = _symmetry_pool_key(primitive.family)
        if primitive_key != (block, family):
            continue
        projector_key = _primitive_projector_key(primitive)
        primitive_index = key_to_index.get(projector_key)
        if primitive_index is None:
            continue
        index.setdefault(primitive.identifier, primitive_index)
    return index


def _gic_coefficient_vector(
    gic: FrozenGIC,
    *,
    primitive_index: dict[str, int],
    vector_size: int | None = None,
) -> np.ndarray | None:
    vector = np.zeros(vector_size or len(primitive_index), dtype=float)
    coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
    for primitive_id, coefficient in coefficients:
        idx = primitive_index.get(primitive_id)
        if idx is None:
            return None
        vector[idx] += float(coefficient)
    return vector


def _primitive_projector_key_index(
    primitives: tuple[GICPrimitive, ...],
) -> dict[tuple[object, ...], int] | None:
    out: dict[tuple[object, ...], int] = {}
    for idx, primitive in enumerate(primitives):
        key = _primitive_projector_key(primitive)
        if key is None or key in out:
            return None
        out[key] = idx
    return out


def _primitive_projector_orientation_sign(primitive: GICPrimitive) -> float:
    """Return the phase relating an oriented primitive to its canonical key."""
    if (
        primitive.family == "RING_PUCKER_COMPONENT"
        and primitive.function == "U"
        and len(primitive.atoms) == 4
    ):
        _center, plane1, plane2, _out = primitive.atoms
        return _permutation_parity_sign((plane1, plane2), tuple(sorted((plane1, plane2))))
    if (
        primitive.family
        in {
            "TORSION",
            "CYCLIC_TORSION",
            "CONDENSED_RING_TORSION",
            "BUTTERFLY",
            "RING_PUCKER_COMPONENT",
        }
        and primitive.function == "D"
        and len(primitive.atoms) == 4
    ):
        _canonical, sign = _canonical_torsion_key_and_sign(primitive.atoms)
        return sign
    if primitive.family in {"OUT_OF_PLANE", "IMPROPER_DIHEDRAL"} and len(primitive.atoms) == 4:
        _center, plane1, plane2, _out = primitive.atoms
        return _permutation_parity_sign((plane1, plane2), tuple(sorted((plane1, plane2))))
    return 1.0


def _operation_primitive_transform(
    primitives: tuple[GICPrimitive, ...],
    *,
    operation: GICPointGroupOperation,
    primitive_key_index: dict[tuple[object, ...], int],
) -> np.ndarray | None:
    if all(
        primitive.family == "RING_PUCKER_COMPONENT" and primitive.function == "RPCK"
        for primitive in primitives
    ):
        return _operation_ring_pucker_transform(primitives, operation=operation)
    if all(
        primitive.family == "PSEUDO_CYCLE_BEND" and primitive.function == "RPCB"
        for primitive in primitives
    ):
        return _operation_angle_component_transform(primitives, operation=operation)
    if all(
        primitive.family == "PSEUDO_CYCLE_TORSION" and primitive.function == "RPCK"
        for primitive in primitives
    ):
        return _operation_ring_pucker_transform(primitives, operation=operation)

    matrix = np.zeros((len(primitives), len(primitives)), dtype=float)
    for source_index, primitive in enumerate(primitives):
        terms = _mapped_primitive_projector_terms(primitive, operation)
        if terms is None:
            return None
        for target_key, coefficient in terms:
            if abs(float(coefficient)) <= 1.0e-12:
                continue
            target_index = primitive_key_index.get(target_key)
            if target_index is None:
                return None
            target_orientation = _primitive_projector_orientation_sign(primitives[target_index])
            matrix[target_index, source_index] += float(coefficient) * target_orientation
    return matrix


def _operation_ring_pucker_transform(
    primitives: tuple[GICPrimitive, ...],
    *,
    operation: GICPointGroupOperation,
) -> np.ndarray | None:
    source_coeffs = [
        _ring_pucker_canonical_torsion_coefficients(_ring_pucker_terms_from_refs(primitive))
        for primitive in primitives
    ]
    pseudoscalar_sign = _operation_pseudoscalar_sign(operation)
    mapped_coeffs = []
    for primitive in primitives:
        mapped_terms = []
        for coefficient, atoms in _ring_pucker_terms_from_refs(primitive):
            mapped_atoms = tuple(_mapped_atom(operation, atom) for atom in atoms)
            if any(atom < 1 for atom in mapped_atoms):
                return None
            mapped_terms.append((coefficient * pseudoscalar_sign, mapped_atoms))
        mapped_coeffs.append(_ring_pucker_canonical_torsion_coefficients(tuple(mapped_terms)))

    torsion_keys = sorted({key for coeffs in (*source_coeffs, *mapped_coeffs) for key in coeffs})
    if not torsion_keys:
        return None
    basis = np.array(
        [[coeffs.get(key, 0.0) for coeffs in source_coeffs] for key in torsion_keys],
        dtype=float,
    )
    if numerical_matrix_rank(basis, absolute_tolerance=1.0e-10) == 0:
        return None
    matrix = np.zeros((len(primitives), len(primitives)), dtype=float)
    for source_index, coeffs in enumerate(mapped_coeffs):
        target = np.array([coeffs.get(key, 0.0) for key in torsion_keys], dtype=float)
        values, *_ = np.linalg.lstsq(basis, target, rcond=1.0e-10)
        residual = float(np.linalg.norm(basis @ values - target))
        if residual > 1.0e-8:
            return None
        matrix[:, source_index] = values
    return matrix


def _operation_angle_component_transform(
    primitives: tuple[GICPrimitive, ...],
    *,
    operation: GICPointGroupOperation,
) -> np.ndarray | None:
    source_coeffs = [
        _angle_component_canonical_coefficients(_angle_component_terms_from_refs(primitive))
        for primitive in primitives
    ]
    mapped_coeffs = []
    for primitive in primitives:
        mapped_terms = []
        for coefficient, atoms in _angle_component_terms_from_refs(primitive):
            mapped_atoms = tuple(_mapped_atom(operation, atom) for atom in atoms)
            if any(atom < 1 for atom in mapped_atoms):
                return None
            mapped_terms.append((coefficient, mapped_atoms))
        mapped_coeffs.append(_angle_component_canonical_coefficients(tuple(mapped_terms)))

    angle_keys = sorted({key for coeffs in (*source_coeffs, *mapped_coeffs) for key in coeffs})
    if not angle_keys:
        return None
    basis = np.array(
        [[coeffs.get(key, 0.0) for coeffs in source_coeffs] for key in angle_keys],
        dtype=float,
    )
    if numerical_matrix_rank(basis, absolute_tolerance=1.0e-10) == 0:
        return None
    matrix = np.zeros((len(primitives), len(primitives)), dtype=float)
    for source_index, coeffs in enumerate(mapped_coeffs):
        target = np.array([coeffs.get(key, 0.0) for key in angle_keys], dtype=float)
        values, *_ = np.linalg.lstsq(basis, target, rcond=1.0e-10)
        residual = float(np.linalg.norm(basis @ values - target))
        if residual > 1.0e-8:
            return None
        matrix[:, source_index] = values
    return matrix


def _project_vector_for_irrep(
    vector: np.ndarray,
    *,
    characters: tuple[float, ...],
    transforms: tuple[np.ndarray | None, ...],
) -> np.ndarray:
    projected = np.zeros_like(vector, dtype=float)
    for character, transform in zip(characters, transforms):
        assert transform is not None
        projected += float(character) * (transform @ vector)
    return projected / float(len(transforms))


def _normalized_coefficient_vector_or_none(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1.0e-10:
        return None
    return vector / norm


def _normalized_cartesian_b_row_or_none(vector: np.ndarray) -> np.ndarray | None:
    """Normalize a Cartesian tangent while rejecting numerical null rows."""

    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= _CARTESIAN_INTERFRAGMENT_PROJECTOR_TOLERANCE:
        return None
    return vector / norm


def _canonical_vector_sign(vector: np.ndarray) -> np.ndarray:
    """Choose the sign whose first significant coefficient is positive."""

    significant = np.flatnonzero(np.abs(vector) > 1.0e-12)
    if significant.size and float(vector[int(significant[0])]) < 0.0:
        return -vector
    return vector


def _canonical_basis_for_span(vectors: list[np.ndarray]) -> tuple[np.ndarray, ...]:
    """Return a basis fixed by coordinate order, not by the input basis."""

    if not vectors:
        return ()
    rows = np.vstack(vectors)
    gram = rows @ rows.T
    projector = rows.T @ np.linalg.pinv(gram, rcond=1.0e-12) @ rows
    projector = 0.5 * (projector + projector.T)
    rank = numerical_matrix_rank(projector, absolute_tolerance=1.0e-10)
    basis: list[np.ndarray] = []
    for axis in range(projector.shape[0]):
        normalized = _normalized_coefficient_vector_or_none(projector[:, axis])
        if normalized is None:
            continue
        independent = _orthonormal_coefficient_residual_or_none(basis, normalized)
        if independent is None:
            continue
        basis.append(_canonical_vector_sign(independent))
        if len(basis) == rank:
            break
    return tuple(basis)


def _orthonormal_coefficient_residual_or_none(
    basis: list[np.ndarray],
    normalized: np.ndarray,
) -> np.ndarray | None:
    residual = np.array(normalized, dtype=float, copy=True)
    for _pass in range(2):
        for vector in basis:
            residual -= float(np.dot(residual, vector)) * vector
    norm = float(np.linalg.norm(residual))
    if not np.isfinite(norm) or norm <= 1.0e-10:
        return None
    return residual / norm

def _coefficients_from_vector(
    primitives: tuple[GICPrimitive, ...],
    vector: np.ndarray,
) -> tuple[tuple[str, float], ...]:
    return tuple(
        (primitive.identifier, float(value))
        for primitive, value in zip(primitives, vector)
        if abs(float(value)) > 1.0e-12
    )


def _next_projected_name(
    family: str,
    irrep: str,
    counters: dict[tuple[str, str], int],
) -> str:
    key = (family, irrep)
    counters[key] = counters.get(key, 0) + 1
    return f"{irrep_name_prefix(irrep)}{primitive_prefix(family)}{counters[key]:03d}"


def _primitive_projector_key(primitive: GICPrimitive) -> tuple[object, ...] | None:
    from matrix_chem.face_primitives import FACE_OPERATORS
    if primitive.function in FACE_OPERATORS.values():
        return (primitive.family, primitive.function, primitive.atoms, primitive.ref_atoms,
                primitive.frame_atoms, primitive.ref_frame_atoms, primitive.mode)
    if (
        primitive.family
        in {
            "STRETCH",
            "HBOND_DISTANCE",
            "TS_REACTION_DISTANCE",
        }
        and len(primitive.atoms) == 2
    ):
        return (primitive.family, tuple(sorted(primitive.atoms)))
    if primitive.family in {"BEND", "CYCLIC_BEND", "SPIRO_BEND"} and len(primitive.atoms) == 3:
        return (
            primitive.family,
            primitive.atoms[1],
            tuple(sorted((primitive.atoms[0], primitive.atoms[2]))),
        )
    if primitive.family == "LINEAR_BEND" and len(primitive.atoms) == 3:
        return (
            "LINEAR_BEND",
            primitive.atoms[1],
            tuple(sorted((primitive.atoms[0], primitive.atoms[2]))),
            primitive.mode,
        )
    if (
        primitive.family == "RING_PUCKER_COMPONENT"
        and primitive.function == "U"
        and len(primitive.atoms) == 4
    ):
        center, plane1, plane2, out = primitive.atoms
        return (
            "RING_PUCKER_COMPONENT",
            center,
            out,
            tuple(sorted((plane1, plane2))),
        )
    if (
        primitive.family
        in {
            "TORSION",
            "CYCLIC_TORSION",
            "CONDENSED_RING_TORSION",
            "BUTTERFLY",
            "RING_PUCKER_COMPONENT",
        }
        and len(primitive.atoms) == 4
    ):
        canonical, _sign = _canonical_torsion_key_and_sign(primitive.atoms)
        return (primitive.family, canonical)
    if primitive.family == "RING_PUCKER_COMPONENT" and primitive.function == "RPCK":
        signature = _ring_pucker_projector_signature(primitive)
        if signature is None:
            return None
        key, _sign = signature
        return ("RING_PUCKER_COMPONENT", key)
    if primitive.family == "PSEUDO_CYCLE_BEND" and primitive.function == "RPCB":
        key = _angle_component_projector_key(primitive)
        return ("PSEUDO_CYCLE_BEND", key) if key is not None else None
    if primitive.family == "PSEUDO_CYCLE_TORSION" and primitive.function == "RPCK":
        signature = _ring_pucker_projector_signature(primitive)
        if signature is None:
            return None
        key, _sign = signature
        return ("PSEUDO_CYCLE_TORSION", key)
    if primitive.family in {"OUT_OF_PLANE", "IMPROPER_DIHEDRAL"} and len(primitive.atoms) == 4:
        center, plane1, plane2, out = primitive.atoms
        return (
            primitive.family,
            center,
            out,
            tuple(sorted((plane1, plane2))),
        )
    if primitive.family == "FRAG_DISTANCE":
        pair = tuple(sorted((_atom_set_key(primitive.atoms), _atom_set_key(primitive.ref_atoms))))
        return ("FRAG_DISTANCE", pair)
    if primitive.family == "FRAG_CENTER_ATOM_DISTANCE":
        return (
            "FRAG_CENTER_ATOM_DISTANCE",
            _atom_set_key(primitive.atoms),
            _atom_set_key(primitive.ref_atoms),
        )
    if primitive.family == "CENTER_ATOM_DISTANCE":
        return (
            "CENTER_ATOM_DISTANCE",
            _atom_set_key(primitive.atoms),
            _atom_set_key(primitive.ref_atoms),
        )
    if primitive.family == "FRAG_TRANSLATION":
        return (
            "FRAG_TRANSLATION",
            primitive.mode,
            _atom_set_key(primitive.atoms),
            _atom_set_key(primitive.ref_atoms),
        )
    if primitive.family == "FRAG_ORIENTATION":
        return (
            "FRAG_ORIENTATION",
            primitive.mode,
            _atom_set_key(primitive.atoms),
            _atom_set_key(primitive.ref_atoms),
            _atom_set_key(primitive.frame_atoms),
            _atom_set_key(primitive.ref_frame_atoms),
        )
    return None


def _symmetry_pool_key(family: str) -> tuple[str, str]:
    """Return the SALC pool key without erasing semantic family ownership."""
    return (primitive_symmetry_block(family), family)


def _projector_primitive_pool_key(
    primitive: GICPrimitive,
    *,
    separate_domains: bool,
    couple_linear_angular: bool = False,
) -> tuple[str, str]:
    if couple_linear_angular and primitive.family in {
        "BEND",
        "LINEAR_BEND",
        "TORSION",
        "OUT_OF_PLANE",
        "IMPROPER_DIHEDRAL",
    }:
        return ("COUPLED_ANGULAR", "COUPLED_ANGULAR")
    base = _symmetry_pool_key(primitive.family)
    if not separate_domains or not (
        primitive.refs and primitive.refs[0] == "PSEUDOBOND_CONTACT_SUPPORT"
    ):
        return base
    fragments = tuple(sorted(ref for ref in primitive.refs[1:] if ref.startswith("F")))
    if len(fragments) != 2:
        return base
    return (f"{base[0]}::{'::'.join(fragments)}", base[1])


def _projector_gic_pool_key(
    gic: FrozenGIC,
    *,
    primitive_by_id: dict[str, GICPrimitive],
    separate_domains: bool,
    couple_linear_angular: bool = False,
) -> tuple[str, str]:
    if couple_linear_angular and gic.family in {
        "BEND",
        "LINEAR_BEND",
        "TORSION",
        "OUT_OF_PLANE",
        "IMPROPER_DIHEDRAL",
    }:
        return ("COUPLED_ANGULAR", "COUPLED_ANGULAR")
    base = _symmetry_pool_key(gic.family)
    if not separate_domains:
        return base
    coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
    keys = {
        _projector_primitive_pool_key(
            primitive_by_id[primitive_id],
            separate_domains=True,
        )
        for primitive_id, _coefficient in coefficients
    }
    return next(iter(keys)) if len(keys) == 1 else base


def _mapped_primitive_projector_terms(
    primitive: GICPrimitive,
    operation: GICPointGroupOperation,
) -> tuple[tuple[tuple[object, ...], float], ...] | None:
    mapped_atoms = tuple(_mapped_atom(operation, atom) for atom in primitive.atoms)
    mapped_refs = tuple(_mapped_atom(operation, atom) for atom in primitive.ref_atoms)
    mapped_frame = tuple(_mapped_atom(operation, atom) for atom in primitive.frame_atoms)
    mapped_ref_frame = tuple(_mapped_atom(operation, atom) for atom in primitive.ref_frame_atoms)
    if any(atom < 1 for atom in mapped_atoms + mapped_refs + mapped_frame + mapped_ref_frame):
        return None

    from matrix_chem.face_primitives import FACE_OPERATORS
    if primitive.function in FACE_OPERATORS.values():
        mapped = replace(primitive, atoms=mapped_atoms, ref_atoms=mapped_refs,
                         frame_atoms=mapped_frame, ref_frame_atoms=mapped_ref_frame)
        sign = (_operation_pseudoscalar_sign(operation)
                if primitive.function in {"FACE_FLAP", "FACE_HINGE", "FACE_DIRECTION_ANGLE"} else 1.)
        return ((_primitive_projector_key(mapped), sign),)

    if (
        primitive.family
        in {
            "STRETCH",
            "HBOND_DISTANCE",
            "TS_REACTION_DISTANCE",
        }
        and len(mapped_atoms) == 2
    ):
        return (((primitive.family, tuple(sorted(mapped_atoms))), 1.0),)
    if primitive.family in {"BEND", "CYCLIC_BEND", "SPIRO_BEND"} and len(mapped_atoms) == 3:
        return (
            (
                (
                    primitive.family,
                    mapped_atoms[1],
                    tuple(sorted((mapped_atoms[0], mapped_atoms[2]))),
                ),
                1.0,
            ),
        )
    if primitive.family == "LINEAR_BEND":
        if not _is_identity_operation(operation):
            return None
        key = _primitive_projector_key(primitive)
        return ((key, 1.0),) if key is not None else None
    if (
        primitive.family == "RING_PUCKER_COMPONENT"
        and primitive.function == "U"
        and len(mapped_atoms) == 4
    ):
        center, plane1, plane2, out = mapped_atoms
        sorted_plane = tuple(sorted((plane1, plane2)))
        return (
            (
                ("RING_PUCKER_COMPONENT", center, out, sorted_plane),
                _permutation_parity_sign((plane1, plane2), sorted_plane)
                * _operation_pseudoscalar_sign(operation),
            ),
        )
    if (
        primitive.family
        in {
            "TORSION",
            "CYCLIC_TORSION",
            "CONDENSED_RING_TORSION",
            "BUTTERFLY",
            "RING_PUCKER_COMPONENT",
        }
        and len(mapped_atoms) == 4
    ):
        canonical, sign = _canonical_torsion_key_and_sign(mapped_atoms)
        return (
            (
                (primitive.family, canonical),
                sign * _operation_pseudoscalar_sign(operation),
            ),
        )
    if primitive.family == "RING_PUCKER_COMPONENT" and primitive.function == "RPCK":
        pseudoscalar_sign = _operation_pseudoscalar_sign(operation)
        mapped_terms = []
        for coefficient, atoms in _ring_pucker_terms_from_refs(primitive):
            mapped_term_atoms = tuple(_mapped_atom(operation, atom) for atom in atoms)
            if any(atom < 1 for atom in mapped_term_atoms):
                return None
            mapped_terms.append((coefficient * pseudoscalar_sign, mapped_term_atoms))
        signature = _ring_pucker_projector_signature_from_terms(tuple(mapped_terms))
        if signature is None:
            return None
        key, sign = signature
        return ((("RING_PUCKER_COMPONENT", key), sign),)
    if primitive.family in {"OUT_OF_PLANE", "IMPROPER_DIHEDRAL"} and len(mapped_atoms) == 4:
        center, plane1, plane2, out = mapped_atoms
        sorted_plane = tuple(sorted((plane1, plane2)))
        return (
            (
                (
                    primitive.family,
                    center,
                    out,
                    sorted_plane,
                ),
                _permutation_parity_sign((plane1, plane2), sorted_plane)
                * _operation_pseudoscalar_sign(operation),
            ),
        )
    if primitive.family == "FRAG_DISTANCE":
        pair = tuple(sorted((_atom_set_key(mapped_atoms), _atom_set_key(mapped_refs))))
        return ((("FRAG_DISTANCE", pair), 1.0),)
    if primitive.family == "FRAG_CENTER_ATOM_DISTANCE":
        return (
            (
                (
                    "FRAG_CENTER_ATOM_DISTANCE",
                    _atom_set_key(mapped_atoms),
                    _atom_set_key(mapped_refs),
                ),
                1.0,
            ),
        )
    if primitive.family == "CENTER_ATOM_DISTANCE":
        return (
            (
                (
                    "CENTER_ATOM_DISTANCE",
                    _atom_set_key(mapped_atoms),
                    _atom_set_key(mapped_refs),
                ),
                1.0,
            ),
        )
    if primitive.family == "FRAG_TRANSLATION":
        return tuple(
            (
                (
                    "FRAG_TRANSLATION",
                    target_mode,
                    _atom_set_key(mapped_atoms),
                    _atom_set_key(mapped_refs),
                ),
                coefficient,
            )
            for target_mode, coefficient in _vector_component_terms(
                operation,
                source_mode=primitive.mode,
                axial=False,
            )
        )
    if primitive.family == "FRAG_ORIENTATION":
        return tuple(
            (
                (
                    "FRAG_ORIENTATION",
                    target_mode,
                    _atom_set_key(mapped_atoms),
                    _atom_set_key(mapped_refs),
                    _atom_set_key(mapped_frame),
                    _atom_set_key(mapped_ref_frame),
                ),
                coefficient,
            )
            for target_mode, coefficient in _vector_component_terms(
                operation,
                source_mode=primitive.mode,
                axial=True,
            )
        )
    return None


def _mapped_atom(operation: GICPointGroupOperation, atom: int) -> int:
    if 1 <= atom <= len(operation.permutation):
        return operation.permutation[atom - 1]
    return -1


def _is_identity_operation(operation: GICPointGroupOperation) -> bool:
    return operation.permutation == tuple(range(1, len(operation.permutation) + 1))


def _vector_component_terms(
    operation: GICPointGroupOperation,
    *,
    source_mode: int,
    axial: bool,
) -> tuple[tuple[int, float], ...]:
    if source_mode not in {0, 1, 2}:
        return ()
    rotation = np.asarray(operation.rotation, dtype=float)
    if rotation.shape != (3, 3):
        return ()
    if axial:
        rotation = float(np.linalg.det(rotation)) * rotation
    return tuple(
        (target_mode, float(rotation[source_mode, target_mode]))
        for target_mode in range(3)
        if abs(float(rotation[source_mode, target_mode])) > 1.0e-10
    )


def _operation_pseudoscalar_sign(operation: GICPointGroupOperation) -> float:
    rotation = np.asarray(operation.rotation, dtype=float)
    if rotation.shape != (3, 3):
        return 1.0
    return -1.0 if float(np.linalg.det(rotation)) < 0.0 else 1.0


def _atom_set_key(atoms: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(sorted(int(atom) for atom in atoms))


def _canonical_torsion_key_and_sign(atoms: tuple[int, ...]) -> tuple[tuple[int, ...], float]:
    forward = tuple(atoms)
    backward = tuple(reversed(atoms))
    if backward < forward:
        # Reversing all four labels leaves a signed dihedral unchanged:
        # both plane normals are reversed/exchanged together. A spatial
        # improper operation supplies its separate pseudoscalar sign.
        return backward, 1.0
    return forward, 1.0


def _ring_pucker_projector_signature(
    primitive: GICPrimitive,
) -> tuple[tuple[tuple[tuple[int, ...], float], ...], float] | None:
    return _ring_pucker_projector_signature_from_terms(_ring_pucker_terms_from_refs(primitive))


def _ring_pucker_projector_signature_from_terms(
    terms: tuple[tuple[float, tuple[int, ...]], ...],
) -> tuple[tuple[tuple[tuple[int, ...], float], ...], float] | None:
    by_torsion = _ring_pucker_canonical_torsion_coefficients(terms)
    compact = {
        atoms: coefficient
        for atoms, coefficient in by_torsion.items()
        if abs(float(coefficient)) > 1.0e-12
    }
    if not compact:
        return None
    _first_atoms, first_coefficient = next(iter(sorted(compact.items())))
    overall_sign = -1.0 if first_coefficient < 0.0 else 1.0
    key = tuple(
        (atoms, round(float(coefficient) * overall_sign, 12))
        for atoms, coefficient in sorted(compact.items())
    )
    return key, overall_sign


def _ring_pucker_canonical_torsion_coefficients(
    terms: tuple[tuple[float, tuple[int, ...]], ...],
) -> dict[tuple[int, ...], float]:
    by_torsion: dict[tuple[int, ...], float] = {}
    for coefficient, atoms in terms:
        if len(atoms) != 4:
            continue
        canonical, sign = _canonical_torsion_key_and_sign(atoms)
        by_torsion[canonical] = by_torsion.get(canonical, 0.0) + float(coefficient) * sign
    return {
        atoms: coefficient
        for atoms, coefficient in by_torsion.items()
        if abs(float(coefficient)) > 1.0e-12
    }


def _angle_component_projector_key(
    primitive: GICPrimitive,
) -> tuple[tuple[tuple[object, ...], float], ...] | None:
    coefficients = _angle_component_canonical_coefficients(
        _angle_component_terms_from_refs(primitive)
    )
    compact = {
        key: coefficient
        for key, coefficient in coefficients.items()
        if abs(float(coefficient)) > 1.0e-12
    }
    if not compact:
        return None
    _first_key, first_coefficient = next(iter(sorted(compact.items())))
    overall_sign = -1.0 if first_coefficient < 0.0 else 1.0
    return tuple(
        (key, round(float(coefficient) * overall_sign, 12))
        for key, coefficient in sorted(compact.items())
    )


def _angle_component_canonical_coefficients(
    terms: tuple[tuple[float, tuple[int, ...]], ...],
) -> dict[tuple[object, ...], float]:
    by_angle: dict[tuple[object, ...], float] = {}
    for coefficient, atoms in terms:
        if len(atoms) != 3:
            continue
        key = (atoms[1], tuple(sorted((atoms[0], atoms[2]))))
        by_angle[key] = by_angle.get(key, 0.0) + float(coefficient)
    return {
        key: coefficient
        for key, coefficient in by_angle.items()
        if abs(float(coefficient)) > 1.0e-12
    }


def _permutation_parity_sign(
    order: tuple[int, ...],
    target_order: tuple[int, ...],
) -> float:
    positions = {atom: idx for idx, atom in enumerate(target_order)}
    try:
        indexes = [positions[atom] for atom in order]
    except KeyError:
        return 1.0
    inversions = 0
    for left in range(len(indexes)):
        for right in range(left + 1, len(indexes)):
            if indexes[left] > indexes[right]:
                inversions += 1
    return -1.0 if inversions % 2 else 1.0


def _local_symmetry_groups(
    gics: tuple[FrozenGIC, ...],
    primitives: tuple[GICPrimitive, ...],
    *,
    atom_symbols: tuple[str, ...],
    intermolecular: bool = False,
    symmetry_operations: tuple[GICPointGroupOperation, ...] = (),
) -> dict[tuple[str, str, str], tuple[FrozenGIC, ...]]:
    primitive_by_id = {primitive.identifier: primitive for primitive in primitives}
    component_orbits = (
        _ordinary_component_orbits(
            primitives,
            natoms=len(atom_symbols),
            symmetry_operations=symmetry_operations,
        )
        if intermolecular
        else {}
    )
    grouped: dict[tuple[str, str, str], list[FrozenGIC]] = {}
    for gic in gics:
        primitive = _single_source_primitive(gic, primitive_by_id)
        if primitive is None:
            continue
        signature = _local_symmetry_signature(primitive, atom_symbols=atom_symbols)
        if not signature:
            continue
        component = _intracomponent_orbit(primitive, component_orbits)
        if component is not None:
            signature = f"COMPONENT_ORBIT={component}|{signature}"
        key = (primitive_symmetry_block(primitive.family), primitive.family, signature)
        grouped.setdefault(key, []).append(gic)
    return {key: tuple(group) for key, group in grouped.items() if len(group) > 1}


def _ordinary_component_orbits(
    primitives: tuple[GICPrimitive, ...],
    *,
    natoms: int,
    symmetry_operations: tuple[GICPointGroupOperation, ...],
) -> dict[int, int]:
    """Map atoms to covalent-component orbits under actual symmetry operations."""

    if natoms <= 0:
        return {}
    parent = list(range(natoms + 1))

    def find(atom: int) -> int:
        while parent[atom] != atom:
            parent[atom] = parent[parent[atom]]
            atom = parent[atom]
        return atom

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for primitive in primitives:
        if (
            primitive.family == "STRETCH"
            and primitive.function == "R"
            and len(primitive.atoms) == 2
            and not _is_special_primitive(primitive)
        ):
            left, right = (int(atom) for atom in primitive.atoms)
            if 1 <= left <= natoms and 1 <= right <= natoms:
                union(left, right)

    components: dict[int, set[int]] = {}
    for atom in range(1, natoms + 1):
        components.setdefault(find(atom), set()).add(atom)
    component_by_atom = {
        atom: root for root, members in components.items() for atom in members
    }
    orbit_parent = {root: root for root in components}

    def orbit_find(root: int) -> int:
        while orbit_parent[root] != root:
            orbit_parent[root] = orbit_parent[orbit_parent[root]]
            root = orbit_parent[root]
        return root

    def orbit_union(left: int, right: int) -> None:
        left_root = orbit_find(left)
        right_root = orbit_find(right)
        if left_root != right_root:
            orbit_parent[max(left_root, right_root)] = min(left_root, right_root)

    for operation in _valid_projector_operations(symmetry_operations):
        for source_root, source_atoms in components.items():
            mapped_atoms = {
                int(operation.permutation[atom - 1]) for atom in source_atoms
            }
            mapped_roots = {component_by_atom.get(atom) for atom in mapped_atoms}
            if len(mapped_roots) != 1 or None in mapped_roots:
                continue
            target_root = next(iter(mapped_roots))
            if mapped_atoms == components[target_root]:
                orbit_union(source_root, target_root)

    orbit_label = {
        root: min(
            min(components[candidate])
            for candidate in components
            if orbit_find(candidate) == orbit_find(root)
        )
        for root in components
    }
    return {
        atom: orbit_label[component_by_atom[atom]] for atom in range(1, natoms + 1)
    }


def _intracomponent_orbit(
    primitive: GICPrimitive,
    component_orbits: dict[int, int],
) -> int | None:
    if not component_orbits:
        return None
    atoms = tuple(int(atom) for atom in primitive.atoms)
    if not atoms:
        return None
    orbits = {component_orbits.get(atom) for atom in atoms}
    if len(orbits) != 1 or None in orbits:
        return None
    return next(iter(orbits))


def _single_source_primitive(
    gic: FrozenGIC,
    primitive_by_id: dict[str, GICPrimitive],
) -> GICPrimitive | None:
    coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
    if len(coefficients) != 1:
        return None
    primitive_id, coefficient = coefficients[0]
    if abs(float(coefficient) - 1.0) > 1.0e-12:
        return None
    return primitive_by_id.get(primitive_id)


def _local_symmetry_signature(
    primitive: GICPrimitive,
    *,
    atom_symbols: tuple[str, ...],
) -> str | None:
    if primitive.family in {"STRETCH", "TS_REACTION_DISTANCE"} and len(primitive.atoms) == 2:
        radial_family = "STRETCH"
        return (
            radial_family + ":R:" + "-".join(_sorted_atom_symbols(primitive.atoms, atom_symbols))
        )
    if primitive.family in {"BEND", "CYCLIC_BEND", "SPIRO_BEND"} and len(primitive.atoms) == 3:
        end_symbols = _sorted_atom_symbols(
            (primitive.atoms[0], primitive.atoms[2]),
            atom_symbols,
        )
        return (
            f"{primitive.family}:A:"
            f"CENTER={primitive.atoms[1]}:"
            f"{_atom_symbol(primitive.atoms[1], atom_symbols)}:"
            f"{'-'.join(end_symbols)}"
        )
    if primitive.family == "LINEAR_BEND" and len(primitive.atoms) == 3:
        end_symbols = _sorted_atom_symbols(
            (primitive.atoms[0], primitive.atoms[2]),
            atom_symbols,
        )
        return (
            f"L:{primitive.mode}:{_atom_symbol(primitive.atoms[1], atom_symbols)}:"
            f"{'-'.join(end_symbols)}"
        )
    if (
        primitive.family
        in {
            "TORSION",
            "CYCLIC_TORSION",
            "CONDENSED_RING_TORSION",
            "BUTTERFLY",
        }
        and len(primitive.atoms) == 4
    ):
        return f"{primitive.family}:D:" + "-".join(
            _atom_symbol(atom, atom_symbols) for atom in primitive.atoms
        )
    if primitive.family in {"OUT_OF_PLANE", "IMPROPER_DIHEDRAL"} and len(primitive.atoms) == 4:
        substituents = _sorted_atom_symbols(primitive.atoms[1:], atom_symbols)
        return (
            f"{primitive.function}:{_atom_symbol(primitive.atoms[0], atom_symbols)}:"
            f"{'-'.join(substituents)}"
        )
    if primitive.family == "FRAG_DISTANCE":
        left = _atom_multiset_signature(primitive.atoms, atom_symbols)
        right = _atom_multiset_signature(primitive.ref_atoms, atom_symbols)
        pair = tuple(sorted((left, right)))
        return f"FC_DIST:{pair[0]}:{pair[1]}"
    if primitive.family == "FRAG_CENTER_ATOM_DISTANCE":
        return (
            "FCA_DIST:"
            f"{_atom_multiset_signature(primitive.atoms, atom_symbols)}:"
            f"{_atom_multiset_signature(primitive.ref_atoms, atom_symbols)}"
        )
    if primitive.family == "FRAG_TRANSLATION":
        return (
            f"FTRANS:{primitive.mode}:"
            f"{_atom_multiset_signature(primitive.atoms, atom_symbols)}:"
            f"{_atom_multiset_signature(primitive.ref_atoms, atom_symbols)}"
        )
    if primitive.family == "FRAG_ORIENTATION":
        return (
            f"FROT:{primitive.mode}:"
            f"{_atom_multiset_signature(primitive.atoms, atom_symbols)}:"
            f"{_atom_multiset_signature(primitive.ref_atoms, atom_symbols)}:"
            f"{_atom_multiset_signature(primitive.frame_atoms, atom_symbols)}:"
            f"{_atom_multiset_signature(primitive.ref_frame_atoms, atom_symbols)}"
        )
    if primitive.family == "CENTER_ATOM_DISTANCE":
        return (
            "CENTER_ATOM_DIST:"
            f"{_atom_multiset_signature(primitive.atoms, atom_symbols)}:"
            f"{_atom_multiset_signature(primitive.ref_atoms, atom_symbols)}"
        )
    return None


def _symmetrized_group_gics(
    key: tuple[str, str, str],
    group: tuple[FrozenGIC, ...],
    *,
    first_index: int,
    name_counters: dict[tuple[str, str, str], int],
    point_group: str,
) -> tuple[FrozenGIC, ...]:
    _block, family, _signature = key
    size = len(group)
    output: list[FrozenGIC] = []
    symmetric_weight = 1.0 / np.sqrt(float(size))
    symmetric_irrep = _local_symmetry_irrep(point_group=point_group, kind="S", index=0)
    symmetric_name = (
        _next_local_angle_salc_name("SymD", name_counters, irrep=symmetric_irrep)
        if family in {"BEND", "CYCLIC_BEND", "SPIRO_BEND"}
        else _next_symmetrized_name(family, "S", name_counters, irrep=symmetric_irrep)
    )
    output.append(
        FrozenGIC(
            identifier=f"GIC{first_index:03d}",
            name=symmetric_name,
            family=family,
            irrep=symmetric_irrep,
            primitive_id=group[0].primitive_id,
            gaussian_expression="LINEAR_COMBINATION",
            coefficients=_combine_gic_coefficients(
                group,
                tuple(symmetric_weight for _idx in range(size)),
            ),
        )
    )
    for group_index in range(1, size):
        weights = [0.0 for _idx in group]
        weights[group_index - 1] = 1.0 / np.sqrt(2.0)
        weights[group_index] = -1.0 / np.sqrt(2.0)
        difference_irrep = _local_symmetry_irrep(
            point_group=point_group,
            kind="D",
            index=group_index - 1,
        )
        difference_name = (
            _next_local_angle_salc_name("Rock", name_counters, irrep=difference_irrep)
            if family in {"BEND", "CYCLIC_BEND", "SPIRO_BEND"}
            else _next_symmetrized_name(
                family,
                "D",
                name_counters,
                irrep=difference_irrep,
            )
        )
        output.append(
            FrozenGIC(
                identifier=f"GIC{first_index + group_index:03d}",
                name=difference_name,
                family=family,
                irrep=difference_irrep,
                primitive_id=group[group_index - 1].primitive_id,
                gaussian_expression="LINEAR_COMBINATION",
                coefficients=_combine_gic_coefficients(group, tuple(weights)),
            )
        )
    return tuple(output)


def _next_local_angle_salc_name(
    stem: str,
    counters: dict[tuple[str, str, str], int],
    *,
    irrep: str,
) -> str:
    """Name one center-local analytic bend SALC independently of global symmetry."""

    key = ("LOCAL_ANGLE", stem, irrep)
    counters[key] = counters.get(key, 0) + 1
    return f"{irrep_name_prefix(irrep)}{stem}{counters[key]:03d}"


def _combine_gic_coefficients(
    gics: tuple[FrozenGIC, ...],
    weights: tuple[float, ...],
) -> tuple[tuple[str, float], ...]:
    totals: dict[str, float] = {}
    order: list[str] = []
    for gic, weight in zip(gics, weights):
        if abs(weight) <= 1.0e-14:
            continue
        coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
        for primitive_id, coefficient in coefficients:
            if primitive_id not in totals:
                order.append(primitive_id)
                totals[primitive_id] = 0.0
            totals[primitive_id] += float(weight) * float(coefficient)
    return tuple(
        (primitive_id, totals[primitive_id])
        for primitive_id in order
        if abs(totals[primitive_id]) > 1.0e-14
    )


def _renumber_frozen_gic(gic: FrozenGIC, index: int) -> FrozenGIC:
    return FrozenGIC(
        identifier=f"GIC{index:03d}",
        name=gic.name,
        family=gic.family,
        irrep=gic.irrep,
        primitive_id=gic.primitive_id,
        gaussian_expression=gic.gaussian_expression,
        coefficients=gic.coefficients,
    )


def _prefix_symmetrized_singletons(
    gics: tuple[FrozenGIC, ...],
    *,
    point_group: str,
) -> tuple[FrozenGIC, ...]:
    return tuple(_prefix_symmetrized_gic(gic, point_group=point_group) for gic in gics)


def _prefix_symmetrized_gic(gic: FrozenGIC, *, point_group: str) -> FrozenGIC:
    if gic.family == "LOCAL_XH_STRETCH":
        return gic
    irrep = (
        gic.irrep if gic.irrep and gic.irrep != "UNASSIGNED" else total_symmetric_irrep(point_group)
    )
    prefix = irrep_name_prefix(irrep)
    name = gic.name if gic.name.startswith(prefix) else f"{prefix}{gic.name}"
    return FrozenGIC(
        identifier=gic.identifier,
        name=name,
        family=gic.family,
        irrep=irrep,
        primitive_id=gic.primitive_id,
        gaussian_expression=gic.gaussian_expression,
        coefficients=gic.coefficients,
    )


def _next_symmetrized_name(
    family: str,
    kind: str,
    counters: dict[tuple[str, str, str], int],
    *,
    irrep: str,
) -> str:
    key = (family, kind, irrep)
    counters[key] = counters.get(key, 0) + 1
    return f"{irrep_name_prefix(irrep)}{primitive_prefix(family)}{kind}{counters[key]:03d}"


def _local_symmetry_irrep(
    *,
    point_group: str,
    kind: str,
    index: int,
) -> str:
    if kind == "S":
        return total_symmetric_irrep(point_group)
    non_total = non_total_irrep_sequence(point_group)
    if not non_total:
        return total_symmetric_irrep(point_group)
    return non_total[index % len(non_total)]


def _sorted_atom_symbols(
    atoms: tuple[int, ...],
    atom_symbols: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(sorted(_atom_symbol(atom, atom_symbols) for atom in atoms))


def _atom_multiset_signature(
    atoms: tuple[int, ...],
    atom_symbols: tuple[str, ...],
) -> str:
    if not atoms:
        return "NONE"
    return ".".join(_sorted_atom_symbols(atoms, atom_symbols))


def _atom_symbol(atom: int, atom_symbols: tuple[str, ...]) -> str:
    if 1 <= atom <= len(atom_symbols):
        return atom_symbols[atom - 1].upper()
    return f"A{atom}"


def _try_select_ranked_primitive(
    primitive: GICPrimitive,
    coords: np.ndarray,
    selected: list[GICPrimitive],
    basis: list[np.ndarray],
    rank: int,
    *,
    rank_tolerance: float,
) -> tuple[int, str]:
    normalized = _normalized_b_row_or_none(
        primitive,
        coords,
        rank_tolerance=rank_tolerance,
    )
    if normalized is None:
        return rank, "singular"
    orthonormal = _orthonormal_residual_or_none(
        basis,
        normalized,
        rank_tolerance=rank_tolerance,
    )
    if orthonormal is None:
        return rank, "dependent"
    selected.append(primitive)
    basis.append(orthonormal)
    return rank + 1, "selected"


def _raise_if_remaining_special_independent(
    primitives: tuple[GICPrimitive, ...],
    coords: np.ndarray,
    basis: list[np.ndarray],
    rank: int,
    *,
    rank_tolerance: float,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    skipped_singular: list[str] = []
    skipped_dependent: list[str] = []
    skipped_singular_details: list[str] = []
    skipped_dependent_details: list[str] = []
    for primitive in primitives:
        normalized = _normalized_b_row_or_none(
            primitive,
            coords,
            rank_tolerance=rank_tolerance,
        )
        if normalized is None:
            skipped_singular.append(primitive.identifier)
            skipped_singular_details.append(_primitive_diagnostic_token(primitive))
            continue
        if (
            _orthonormal_residual_or_none(
                basis,
                normalized,
                rank_tolerance=rank_tolerance,
            )
            is not None
        ):
            raise GICForgeContractError(
                "protected special primitive set exceeds the vibrational rank: "
                f"{primitive.identifier} {primitive.name} would add an independent "
                "row after the target rank was reached"
            )
        skipped_dependent.append(primitive.identifier)
        skipped_dependent_details.append(_primitive_diagnostic_token(primitive))
    return (
        tuple(skipped_singular),
        tuple(skipped_dependent),
        tuple(skipped_singular_details),
        tuple(skipped_dependent_details),
    )


def _normalized_b_row_or_none(
    primitive: GICPrimitive,
    coords: np.ndarray,
    *,
    rank_tolerance: float,
) -> np.ndarray | None:
    try:
        # FROT is a frozen, reference-relative exponential-map increment.
        # Rank selection must therefore inspect its tangent at the frozen
        # reference, not the absolute logarithm between the two canonical
        # fragment frames.  The latter is singular when the frames differ by
        # 180 degrees even though the local rotational chart is regular.
        reference_coords = coords if primitive.function == "FROT" else None
        row = _analytic_b_row(
            primitive,
            coords,
            reference_coords=reference_coords,
        )
    except FloatingPointError:
        return None
    norm = float(np.linalg.norm(row))
    if not np.isfinite(norm) or norm <= rank_tolerance:
        return None
    return row / norm


def _orthonormal_residual_or_none(
    basis: list[np.ndarray],
    normalized: np.ndarray,
    *,
    rank_tolerance: float,
) -> np.ndarray | None:
    residual = np.array(normalized, dtype=float, copy=True)
    for vector in basis:
        residual -= float(np.dot(residual, vector)) * vector
    norm = float(np.linalg.norm(residual))
    if not np.isfinite(norm) or norm <= rank_tolerance:
        return None
    return residual / norm
