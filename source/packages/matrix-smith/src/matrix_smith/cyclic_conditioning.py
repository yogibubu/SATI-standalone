"""Minimum delocalization of ill-conditioned cyclic SONIC blocks.

Stretching and the other already selected coordinates remain explicit rows.
Only cyclic bend/puckering rows are replaced by their local complement against
overlapping ordinary intrafragment rows.  The resulting transformation is
block triangular, so it preserves the exact molecular row space while curing
the frequent bend/out-of-plane near dependence of distorted rings.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .coordinate_diagnostics import normalized_sonic_condition_diagnostics
from .evaluation import build_gic_b_matrix, build_primitive_b_matrix
from .models import FrozenGIC, GICDefinition, GICReductionDiagnostics
from .policy import MAX_UNPARTITIONED_SONIC_CONDITION


_CYCLIC_FAMILIES = frozenset(
    {"CYCLIC_BEND", "RING_PUCKER_COMPONENT", "BUTTERFLY"}
)
_LOCAL_COMPLEMENT_FAMILIES = frozenset(
    {"STRETCH", "BEND", "LINEAR_BEND", "OUT_OF_PLANE"}
)
_MAX_LOCAL_ROWS = 256
_MAX_CYCLIC_LOCAL_CONDITION = 5.0e2
# Pool recovery may select only true ORACLE primitives.  Puckering and
# butterfly coordinates are SMITH-derived combinations and must never be
# rebound as if they were primitive measurements.
_CYCLIC_POOL_FAMILIES = frozenset({"CYCLIC_BEND", "OUT_OF_PLANE"})


def _components(gic: FrozenGIC) -> tuple[tuple[str, float], ...]:
    return gic.coefficients or ((gic.primitive_id, 1.0),)


def _support(gic: FrozenGIC, primitive_by_id: dict[str, object]) -> frozenset[int]:
    atoms: set[int] = set()
    for primitive_id, _coefficient in _components(gic):
        primitive = primitive_by_id[primitive_id]
        atoms.update(int(atom) for atom in getattr(primitive, "atoms", ()))
        atoms.update(int(atom) for atom in getattr(primitive, "ref_atoms", ()))
        atoms.update(int(atom) for atom in getattr(primitive, "frame_atoms", ()))
        atoms.update(int(atom) for atom in getattr(primitive, "ref_frame_atoms", ()))
    return frozenset(atoms)


def _overlap_blocks(
    indices: tuple[int, ...], supports: tuple[frozenset[int], ...]
) -> tuple[tuple[int, ...], ...]:
    remaining = set(indices)
    blocks: list[tuple[int, ...]] = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        block = {first}
        atoms = set(supports[first])
        changed = True
        while changed:
            changed = False
            for index in sorted(remaining):
                if atoms.intersection(supports[index]):
                    remaining.remove(index)
                    block.add(index)
                    atoms.update(supports[index])
                    changed = True
        blocks.append(tuple(sorted(block)))
    return tuple(blocks)


def _expanded_coefficients(
    gics: tuple[FrozenGIC, ...], weights: np.ndarray
) -> tuple[tuple[str, float], ...]:
    combined: dict[str, float] = {}
    order: list[str] = []
    for weight, gic in zip(weights, gics, strict=True):
        if abs(float(weight)) <= 1.0e-14:
            continue
        for primitive_id, coefficient in _components(gic):
            if primitive_id not in combined:
                order.append(primitive_id)
                combined[primitive_id] = 0.0
            combined[primitive_id] += float(weight) * float(coefficient)
    vector = np.asarray([combined[item] for item in order], dtype=float)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("cyclic SONIC complement produced a null coordinate")
    vector /= norm
    significant = np.flatnonzero(np.abs(vector) > 1.0e-12)
    if significant.size and vector[int(significant[0])] < 0.0:
        vector = -vector
    return tuple(
        (primitive_id, float(vector[index]))
        for index, primitive_id in enumerate(order)
        if abs(float(vector[index])) > 1.0e-12
    )


def _cyclic_local_complement(
    definition: GICDefinition,
    *,
    matrix: np.ndarray,
    cyclic: tuple[int, ...],
    supports: tuple[frozenset[int], ...],
    shells: int,
) -> GICDefinition | None:
    """Residualize cyclic rows against a bounded local standard-coordinate shell."""

    transformation = np.eye(len(definition.gics), dtype=float)
    blocks = _overlap_blocks(cyclic, supports)
    for block in blocks:
        block_atoms = set().union(*(supports[index] for index in block))
        complement_set: set[int] = set()
        for _shell in range(max(int(shells), 1)):
            added = {
                index
                for index, gic in enumerate(definition.gics)
                if index not in cyclic
                and index not in complement_set
                and gic.family in _LOCAL_COMPLEMENT_FAMILIES
                and bool(supports[index].intersection(block_atoms))
            }
            if not added:
                break
            complement_set.update(added)
            for index in added:
                block_atoms.update(supports[index])
        complement = tuple(sorted(complement_set))
        if not complement or len(block) + len(complement) > _MAX_LOCAL_ROWS:
            return None
        cyclic_rows = matrix[np.asarray(block, dtype=int)]
        fixed_rows = matrix[np.asarray(complement, dtype=int)]
        projection = cyclic_rows @ np.linalg.pinv(fixed_rows, rcond=1.0e-10)
        residual = cyclic_rows - projection @ fixed_rows
        left, singular, _right = np.linalg.svd(residual, full_matrices=False)
        if (
            singular.size != len(block)
            or float(singular[-1])
            <= max(float(singular[0]) * 1.0e-10, 1.0e-12)
        ):
            return None
        block_transform = left.T
        transformation[np.ix_(block, block)] = block_transform
        transformation[np.ix_(block, complement)] = -block_transform @ projection

    trial_gics = list(definition.gics)
    for serial, index in enumerate(cyclic, 1):
        coefficients = _expanded_coefficients(definition.gics, transformation[index])
        trial_gics[index] = replace(
            definition.gics[index],
            name=f"CyDl{serial:04d}",
            family="CYCLIC_BEND",
            primitive_id=coefficients[0][0],
            gaussian_expression="LINEAR_COMBINATION",
            coefficients=coefficients,
        )
    candidate = replace(definition, gics=tuple(trial_gics))
    candidate_matrix = np.asarray(
        build_gic_b_matrix(
            candidate,
            coordinates_angstrom=candidate.reference_coordinates_angstrom,
        ).rows,
        dtype=float,
    )
    # Coefficient-space normalization is not a physical metric when lengths
    # and angular primitives are combined.  Normalize the newly constructed
    # rows in the Wilson metric at the chart origin.  Without this step a
    # perfectly independent cyclic row may have a norm near 1e-6, producing
    # enormous optimizer covectors and an unusably small nonlinear chart.
    for index in cyclic:
        row_norm = float(np.linalg.norm(candidate_matrix[index]))
        if not np.isfinite(row_norm) or row_norm <= 1.0e-12:
            return None
        gic = trial_gics[index]
        trial_gics[index] = replace(
            gic,
            coefficients=tuple(
                (primitive_id, float(coefficient) / row_norm)
                for primitive_id, coefficient in gic.coefficients
            ),
        )
    return replace(definition, gics=tuple(trial_gics))


def condition_cyclic_minimum_chart(
    definition: GICDefinition,
    *,
    rank_tolerance: float,
) -> GICDefinition:
    """Apply a local cyclic complement only when the complete chart needs it."""

    if definition.point_group.strip().upper() not in {"C1", "UNKNOWN", ""}:
        return definition

    matrix = np.asarray(
        build_gic_b_matrix(
            definition,
            coordinates_angstrom=definition.reference_coordinates_angstrom,
        ).rows,
        dtype=float,
    )
    before = normalized_sonic_condition_diagnostics(
        matrix,
        tolerance=rank_tolerance,
        maximum_condition_number=MAX_UNPARTITIONED_SONIC_CONDITION,
    )
    if (
        int(before["rank"]) == definition.target_rank
        and np.isfinite(float(before["condition_number"]))
        and float(before["condition_number"]) <= _MAX_CYCLIC_LOCAL_CONDITION
    ):
        return definition

    primitive_by_id = {primitive.identifier: primitive for primitive in definition.primitives}
    supports = tuple(_support(gic, primitive_by_id) for gic in definition.gics)
    cyclic = tuple(
        index
        for index, gic in enumerate(definition.gics)
        if gic.family in _CYCLIC_FAMILIES
    )
    if not cyclic:
        return definition

    blocks = _overlap_blocks(cyclic, supports)
    pool_recovery = False
    base = definition
    base_matrix = matrix
    if int(before["rank"]) != definition.target_rank:
        recovered = _cyclic_pool_recovery(
            definition,
            matrix=matrix,
            cyclic=cyclic,
            supports=supports,
            rank_tolerance=rank_tolerance,
        )
        if recovered is None:
            return definition
        pool_recovery = True
        base = recovered
        base_matrix = np.asarray(
            build_gic_b_matrix(
                base,
                coordinates_angstrom=base.reference_coordinates_angstrom,
            ).rows,
            dtype=float,
        )
        base_diagnostics = normalized_sonic_condition_diagnostics(
            base_matrix,
            tolerance=rank_tolerance,
            maximum_condition_number=MAX_UNPARTITIONED_SONIC_CONDITION,
        )
        if (
            int(base_diagnostics["rank"]) != definition.target_rank
            or not np.isfinite(float(base_diagnostics["condition_number"]))
        ):
            return definition

    base_primitive_by_id = {
        primitive.identifier: primitive for primitive in base.primitives
    }
    base_supports = tuple(
        _support(gic, base_primitive_by_id) for gic in base.gics
    )
    candidate = base
    after = normalized_sonic_condition_diagnostics(
        base_matrix,
        tolerance=rank_tolerance,
        maximum_condition_number=MAX_UNPARTITIONED_SONIC_CONDITION,
    )
    selected_shells = 0
    for shells in (1, 2):
        trial = _cyclic_local_complement(
            base,
            matrix=base_matrix,
            cyclic=cyclic,
            supports=base_supports,
            shells=shells,
        )
        if trial is None:
            continue
        trial_matrix = np.asarray(
            build_gic_b_matrix(
                trial,
                coordinates_angstrom=trial.reference_coordinates_angstrom,
            ).rows,
            dtype=float,
        )
        trial_diagnostics = normalized_sonic_condition_diagnostics(
            trial_matrix,
            tolerance=rank_tolerance,
            maximum_condition_number=MAX_UNPARTITIONED_SONIC_CONDITION,
        )
        if (
            int(trial_diagnostics["rank"]) == definition.target_rank
            and np.isfinite(float(trial_diagnostics["condition_number"]))
            and float(trial_diagnostics["condition_number"])
            < float(after["condition_number"])
        ):
            candidate = trial
            after = trial_diagnostics
            selected_shells = shells
        if float(after["condition_number"]) <= _MAX_CYCLIC_LOCAL_CONDITION:
            break

    if (
        int(after["rank"]) != definition.target_rank
        or not np.isfinite(float(after["condition_number"]))
        or float(after["condition_number"]) > MAX_UNPARTITIONED_SONIC_CONDITION
        or (
            not pool_recovery
            and float(after["condition_number"]) >= float(before["condition_number"])
        )
    ):
        return definition

    previous = definition.reduction_diagnostics
    decisions = tuple(previous.conditioning_decisions if previous is not None else ()) + (
        "CYCLIC_BEND_OUT_OF_PLANE_LOCAL_COMPLEMENT",
        f"CYCLIC_CONDITION_BEFORE={float(before['condition_number']):.12g}",
        f"CYCLIC_CONDITION_AFTER={float(after['condition_number']):.12g}",
        f"CYCLIC_LOCAL_BLOCKS={len(blocks)}",
        f"CYCLIC_LOCAL_SHELLS={selected_shells}",
        *(
            ("CYCLIC_POOL_JOINT_BEND_OUT_OF_PLANE_SELECTION",)
            if pool_recovery
            else ()
        ),
    )
    diagnostics = GICReductionDiagnostics(
        rank_method=(previous.rank_method if previous is not None else "EVALUATED_WILSON"),
        reduction_policy="MINIMUM_CYCLIC_DELOCALIZATION",
        selected=(previous.selected if previous is not None else tuple(g.identifier for g in candidate.gics)),
        skipped_singular=(previous.skipped_singular if previous is not None else ()),
        skipped_dependent=(previous.skipped_dependent if previous is not None else ()),
        selected_by_family=(previous.selected_by_family if previous is not None else ()),
        skipped_singular_details=(previous.skipped_singular_details if previous is not None else ()),
        skipped_dependent_details=(previous.skipped_dependent_details if previous is not None else ()),
        conditioning_decisions=decisions,
    )
    return replace(candidate, reduction_diagnostics=diagnostics)


def _cyclic_pool_recovery(
    definition: GICDefinition,
    *,
    matrix: np.ndarray,
    cyclic: tuple[int, ...],
    supports: tuple[frozenset[int], ...],
    rank_tolerance: float,
) -> GICDefinition | None:
    """Keep fixed rows and delocalize only cyclic bend/OOP primitive pools."""

    fixed = tuple(index for index in range(len(definition.gics)) if index not in cyclic)
    fixed_rows = matrix[np.asarray(fixed, dtype=int)]
    _left, singular, _right = np.linalg.svd(fixed_rows, full_matrices=False)
    fixed_rank = int(
        np.count_nonzero(
            singular > max(float(singular[0]) * rank_tolerance, 1.0e-12)
        )
    )
    if fixed_rank != len(fixed):
        return None
    primitive_matrix = np.asarray(
        build_primitive_b_matrix(
            definition,
            coordinates_angstrom=definition.reference_coordinates_angstrom,
        ).rows,
        dtype=float,
    )
    fixed_space = _right[:fixed_rank]
    cyclic_atoms = set().union(*(supports[index] for index in cyclic))
    eligible = tuple(
        index
        for index, primitive in enumerate(definition.primitives)
        if primitive.family in _CYCLIC_POOL_FAMILIES
        and len(
            cyclic_atoms.intersection(
                (
                    *primitive.atoms,
                    *primitive.ref_atoms,
                    *primitive.frame_atoms,
                    *primitive.ref_frame_atoms,
                )
            )
        )
        >= 2
    )
    if len(eligible) < len(cyclic):
        return None
    candidate_rows = primitive_matrix[np.asarray(eligible, dtype=int)]
    residual = candidate_rows - (candidate_rows @ fixed_space.T) @ fixed_space
    left, singular, _right_residual = np.linalg.svd(residual, full_matrices=False)
    threshold = max(float(singular[0]) * rank_tolerance, 1.0e-12)
    if int(np.count_nonzero(singular > threshold)) < len(cyclic):
        return None
    # Whiten only the cyclic/OOP residual subspace.  Do not subtract fixed GIC
    # rows from the coordinate definitions: that would mix intrafragment
    # cyclic coordinates with interfragment translations or rotations.  The
    # left singular combinations contain only ORACLE cyclic/OOP primitives,
    # while their residuals against the complete fixed chart have unit norm.
    pool_weights = left[:, : len(cyclic)].T / singular[: len(cyclic), None]
    gics = list(definition.gics)
    for serial, (slot, weights) in enumerate(
        zip(cyclic, pool_weights, strict=True), 1
    ):
        coefficients = np.asarray(weights, dtype=float)
        significant = np.flatnonzero(np.abs(coefficients) > 1.0e-12)
        if significant.size and coefficients[int(significant[0])] < 0.0:
            coefficients = -coefficients
        frozen_coefficients = tuple(
            (definition.primitives[primitive_index].identifier, float(coefficient))
            for primitive_index, coefficient in zip(eligible, coefficients, strict=True)
            if abs(float(coefficient)) > 1.0e-12
        )
        if not frozen_coefficients:
            return None
        gics[slot] = replace(
            gics[slot],
            name=f"CyDl{serial:04d}",
            family="CYCLIC_BEND",
            primitive_id=frozen_coefficients[0][0],
            gaussian_expression="LINEAR_COMBINATION",
            coefficients=frozen_coefficients,
        )
    return replace(definition, gics=tuple(gics))


__all__ = ["condition_cyclic_minimum_chart"]
