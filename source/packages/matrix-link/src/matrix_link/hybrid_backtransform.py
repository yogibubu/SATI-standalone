"""LINK finite-predictor/constrained-corrector realization of SONIC changes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

import numpy as np

from matrix_chem import rotation_matrix_from_vector

from .coordinate_domain import near_linear_ordinary_angle_ids
from .fragment_backtransform import direct_fragment_rigid_prediction
from .internal_coordinates import (
    InternalCoordinateBackTransform,
    cartesian_from_internal_jacobian,
    constrained_internal_coordinate_step,
)

if TYPE_CHECKING:
    from .rigid_pose import RigidComplexModel
    from matrix_smith.models import GICDefinition


_SOFT_FUNCTIONS = {"D", "IMPD", "U", "RPCK", "FROT", "FTRANS"}
_SOFT_FAMILY_TOKENS = (
    "TORSION",
    "DIHEDRAL",
    "RING_PUCKERING",
    "PUCKER",
    "BUTTERFLY",
    "OUT_OF_PLANE",
    "INVERSION",
    "FRAG_ROTATION",
    "FRAG_TRANSLATION",
    "HBOND",
)


@dataclass(frozen=True)
class AcyclicTorsionSpec:
    coordinate_index: int
    primitive_id: str
    atoms: tuple[int, int, int, int]
    coefficient: float
    moving_atoms: tuple[int, ...]
    counter_moving_atoms: tuple[int, ...] = ()
    period_radian: float = 2.0 * np.pi


@dataclass(frozen=True)
class DirectAcyclicTorsionBackTransform:
    """Exact rigid-subgraph realization of pure acyclic torsion targets."""

    coordinates_angstrom: np.ndarray
    values: np.ndarray
    residual: np.ndarray
    torsion_indices: tuple[int, ...]
    method: str = "DIRECT_RIGID_ACYCLIC_TORSION"


@dataclass(frozen=True)
class DirectRigidSoftBackTransform:
    """B-free realization of acyclic torsions and fragment poses together."""

    coordinates_angstrom: np.ndarray
    values: np.ndarray
    residual: np.ndarray
    torsion_indices: tuple[int, ...]
    fragment_indices: tuple[int, ...]
    ring_indices: tuple[int, ...] = ()
    method: str = "DIRECT_RIGID_SOFT_COORDINATES"


@dataclass(frozen=True)
class RingPhaseSpec:
    """A cosine/sine pair describing one polar ring-puckering phase."""

    coordinate_indices: tuple[int, int]
    primitive_ids: tuple[str, str]
    atoms: tuple[int, ...]


@dataclass(frozen=True)
class RingPuckeringBlockSpec:
    """RPck rows and Cartesian atoms forming one LINK-local solve block."""

    coordinate_indices: tuple[int, ...]
    primitive_ids: tuple[str, ...]
    atoms: tuple[int, ...]


@dataclass(frozen=True)
class HybridInternalCoordinateBackTransform:
    coordinates_angstrom: np.ndarray
    values: np.ndarray
    residual: np.ndarray
    iterations: int
    converged: bool
    cartesian_from_q: np.ndarray
    finite_fragment_indices: tuple[int, ...]
    finite_torsion_indices: tuple[int, ...]
    continuation_indices: tuple[int, ...]
    hard_indices: tuple[int, ...]
    substeps: int
    corrector_iterations: int
    finite_ring_phase_indices: tuple[int, ...] = ()
    finite_ring_indices: tuple[int, ...] = ()
    method: str = "FINITE_SOFT_PREDICTOR_CONSTRAINED_HARD_CORRECTOR"


def hybrid_internal_coordinate_step(
    definition: "GICDefinition",
    coordinates_angstrom: np.ndarray,
    target_values: np.ndarray,
    evaluate: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    *,
    evaluate_subset: Callable[[np.ndarray, tuple[int, ...]], tuple[np.ndarray, np.ndarray]]
    | None = None,
    evaluate_values: Callable[[np.ndarray], np.ndarray] | None = None,
    evaluate_values_subset: Callable[[np.ndarray, tuple[int, ...]], np.ndarray] | None = None,
    rigid_model: "RigidComplexModel | None" = None,
    rigid_target_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    fixed_atom_indices: tuple[int, ...] = (),
    project_coordinates: Callable[[np.ndarray], np.ndarray] | None = None,
    tolerance: float = 1.0e-9,
    max_continuation_increment: float = 0.12,
    max_substeps: int = 32,
) -> HybridInternalCoordinateBackTransform:
    """Realize large SONIC moves with finite soft predictors and a hard corrector.

    Complete fragment translations/rotations are imposed on their rigid groups.
    A one-primitive dihedral whose central bond is a graph bridge is imposed by
    an exact finite rotation of one separated subgraph.  Ring, inversion and
    mixed soft coordinates follow a continuation path.  At every path point,
    the remaining coordinates are corrected in the null space of the already
    imposed finite soft coordinates.
    """

    coords = np.asarray(coordinates_angstrom, dtype=float).copy()
    target = np.asarray(target_values, dtype=float).reshape(-1)
    start_values, start_b = _validated_evaluation(evaluate, coords, target.size)
    ring_blocks = ring_puckering_block_specs(definition)
    ring_phases = _ring_phase_specs(definition, blocks=ring_blocks)
    ring_indices = tuple(index for block in ring_blocks for index in block.coordinate_indices)
    ring_phase_indices = tuple(index for spec in ring_phases for index in spec.coordinate_indices)
    target = ring_phase_only_targets(definition, start_values, target)
    initial_residual = target - start_values
    initial_norm = float(np.linalg.norm(initial_residual))
    if initial_norm <= tolerance:
        projector = np.linalg.pinv(start_b, rcond=1.0e-8)
        return HybridInternalCoordinateBackTransform(
            coordinates_angstrom=coords,
            values=start_values,
            residual=initial_residual,
            iterations=0,
            converged=True,
            cartesian_from_q=projector,
            finite_fragment_indices=(),
            finite_torsion_indices=(),
            continuation_indices=(),
            hard_indices=tuple(range(target.size)),
            substeps=0,
            corrector_iterations=0,
            finite_ring_phase_indices=ring_phase_indices,
            finite_ring_indices=ring_indices,
        )

    torsions = _acyclic_torsion_specs(
        definition,
        natoms=coords.shape[0],
        fixed_atom_indices=fixed_atom_indices,
    )
    torsion_indices = tuple(item.coordinate_index for item in torsions)
    if rigid_model is None:
        fragment_probe = direct_fragment_rigid_prediction(
            definition,
            coords,
            target,
            lambda trial: _validated_evaluation(evaluate, trial, target.size)[0],
            evaluate_subset=evaluate_subset,
            evaluate_values_subset=evaluate_values_subset,
        )
        fragment_indices = fragment_probe.handled_indices
    else:
        fragment_indices = rigid_model.coordinate_indices
    finite_indices = tuple(sorted(set((*fragment_indices, *torsion_indices, *ring_indices))))
    soft_indices = _soft_coordinate_indices(definition)
    continuation_indices = tuple(sorted(set(soft_indices) - set(finite_indices)))
    hard_indices = tuple(sorted(set(range(target.size)) - set(soft_indices)))
    solve_indices = tuple(sorted((*hard_indices, *continuation_indices)))

    continuation_delta = (
        initial_residual[list(continuation_indices)] if continuation_indices else ()
    )
    max_delta = (
        float(np.max(np.abs(np.asarray(continuation_delta, dtype=float))))
        if len(continuation_delta)
        else 0.0
    )
    substeps = max(1, int(np.ceil(max_delta / max(max_continuation_increment, 1.0e-6))))
    if near_linear_ordinary_angle_ids(definition, coords):
        soft_delta = initial_residual[list(soft_indices)] if soft_indices else initial_residual
        near_linear_delta = float(np.max(np.abs(soft_delta))) if soft_delta.size else 0.0
        substeps = max(
            substeps,
            2,
            int(np.ceil(near_linear_delta / max(0.5 * max_continuation_increment, 1.0e-6))),
        )
    substeps = min(substeps, max(int(max_substeps), 1))
    corrector_iterations = 0
    total_iterations = 0
    last_result: InternalCoordinateBackTransform | None = None
    attempted_substeps = 0

    for substep in range(1, substeps + 1):
        attempted_substeps = substep
        fraction = substep / float(substeps)
        subtarget = start_values + fraction * initial_residual
        subtarget = _interpolate_ring_phase_targets(
            start_values, target, subtarget, fraction, ring_phases
        )
        # Reapply the finite predictors after each nonlinear hard correction:
        # this is the outer predictor--corrector loop that prevents soft drift.
        # Alternating finite fragment poses and constrained intrafragment
        # corrections is a nonlinear block solve.  Four sweeps are enough
        # near the frozen reference, but reactive or strongly ionic
        # rearrangements can require additional contraction after a fragment
        # frame has changed substantially.  Keep the bound finite while
        # allowing the same predictor/corrector to reach its declared
        # tolerance instead of returning a nearly untouched geometry.
        substep_converged = False
        previous_residual_norm = float("inf")
        stagnant_sweeps = 0
        for _outer in range(8):
            if rigid_model is None:
                fragment = direct_fragment_rigid_prediction(
                    definition,
                    coords,
                    subtarget,
                    lambda trial: _validated_evaluation(evaluate, trial, target.size)[0],
                    evaluate_subset=evaluate_subset,
                    evaluate_values_subset=evaluate_values_subset,
                )
                coords = fragment.coordinates_angstrom
                handled_fragment_indices = fragment.handled_indices
            else:
                rigid_target = (
                    subtarget
                    if rigid_target_transform is None
                    else np.asarray(rigid_target_transform(subtarget), dtype=float)
                )
                coords = rigid_model.realize_sonic_from_base(rigid_target, coords)
                handled_fragment_indices = fragment_indices
            coords = direct_acyclic_torsion_prediction(
                coords,
                subtarget,
                evaluate,
                torsions,
                evaluate_values=evaluate_values,
                evaluate_values_subset=evaluate_values_subset,
            )
            coords = direct_ring_puckering_prediction(
                coords,
                subtarget,
                evaluate,
                ring_blocks,
                evaluate_subset=evaluate_subset,
                evaluate_values_subset=evaluate_values_subset,
                fixed_atom_indices=fixed_atom_indices,
            )
            if project_coordinates is not None:
                coords = np.asarray(project_coordinates(coords), dtype=float)
            protected = tuple(
                sorted(set((*handled_fragment_indices, *torsion_indices, *ring_indices)))
            )
            # The set handled by a finite fragment predictor can grow after a
            # contact closes during continuation.  Such a coordinate has
            # already been imposed by the predictor and must move from the
            # hard-corrector set to its protected null space for this sweep.
            # Keeping the original frozen partition here would ask the same
            # row to be solved and protected simultaneously.
            current_solve_indices = tuple(
                index for index in solve_indices if index not in set(protected)
            )
            if current_solve_indices:
                last_result = constrained_internal_coordinate_step(
                    coords,
                    subtarget,
                    evaluate,
                    evaluate_values=evaluate_values,
                    solve_indices=current_solve_indices,
                    protected_indices=protected,
                    max_iterations=24,
                    # The outer hybrid solve performs its own complete
                    # target certification at ``tolerance``.  Driving every
                    # nested hard corrector ten times below that threshold
                    # only adds Wilson rebuilds and cannot improve the
                    # certified Cartesian result.
                    tolerance=max(tolerance, 1.0e-11),
                    fixed_atom_indices=fixed_atom_indices,
                    project_coordinates=project_coordinates,
                )
                coords = last_result.coordinates_angstrom
                corrector_iterations += last_result.iterations
                total_iterations += last_result.iterations
            if evaluate_values is None:
                values, _b = _validated_evaluation(evaluate, coords, target.size)
            else:
                values = np.asarray(evaluate_values(coords), dtype=float).reshape(-1)
                if values.shape != (target.size,) or not np.all(np.isfinite(values)):
                    raise ValueError(
                        "internal-coordinate value evaluator returned incompatible values"
                    )
            sub_residual = subtarget - values
            total_iterations += 1
            residual_norm = float(np.linalg.norm(sub_residual))
            if residual_norm <= tolerance * max(
                1.0, float(np.linalg.norm(subtarget - start_values))
            ):
                substep_converged = True
                break
            # Avoid spending sixteen expensive corrector sweeps on a stalled
            # ordinary step; nonlinear cases retain the caller's fallback.
            if residual_norm >= previous_residual_norm * (1.0 - 1.0e-3):
                stagnant_sweeps += 1
            else:
                stagnant_sweeps = 0
            previous_residual_norm = residual_norm
            if stagnant_sweeps >= 2:
                break
        if not substep_converged:
            # Continuation waypoints are ordered constraints.  Advancing to a
            # later target after the current waypoint failed only compounds
            # the residual and repeats expensive corrector work at geometries
            # that are already outside the realizable path.
            break

    values, b_matrix = _validated_evaluation(evaluate, coords, target.size)
    residual = target - values
    converged = float(np.linalg.norm(residual)) <= tolerance * max(1.0, initial_norm)
    projector = cartesian_from_internal_jacobian(b_matrix, rcond=1.0e-8)
    return HybridInternalCoordinateBackTransform(
        coordinates_angstrom=coords,
        values=values,
        residual=residual,
        iterations=total_iterations,
        converged=converged,
        cartesian_from_q=projector,
        finite_fragment_indices=fragment_indices,
        finite_torsion_indices=torsion_indices,
        finite_ring_phase_indices=ring_phase_indices,
        finite_ring_indices=ring_indices,
        continuation_indices=continuation_indices,
        hard_indices=hard_indices,
        substeps=attempted_substeps,
        corrector_iterations=corrector_iterations,
    )


def ring_phase_only_targets(
    definition: "GICDefinition",
    current_values: np.ndarray,
    requested_targets: np.ndarray,
    *,
    minimum_amplitude: float = 1.0e-10,
) -> np.ndarray:
    """Keep ring-puckering amplitudes fixed and retain only requested phases.

    SMITH stores a polar puckering mode as adjacent cosine/sine ``RPCK``
    components.  LINK optimizes the phase on the circle through the current
    point; the small radial amplitude is deliberately not stepped.
    """

    current = np.asarray(current_values, dtype=float).reshape(-1)
    target = np.asarray(requested_targets, dtype=float).reshape(-1).copy()
    if current.shape != target.shape:
        raise ValueError("current and target internal-coordinate vectors must match")
    for spec in _ring_phase_specs(definition):
        left, right = spec.coordinate_indices
        amplitude = float(np.hypot(current[left], current[right]))
        if amplitude <= float(minimum_amplitude):
            # The phase is undefined at q=0.  Freezing both components is the
            # only deterministic phase-only operation in this limit.
            target[[left, right]] = current[[left, right]]
            continue
        requested_phase = float(np.arctan2(target[right], target[left]))
        target[left] = amplitude * np.cos(requested_phase)
        target[right] = amplitude * np.sin(requested_phase)
    return target


def direct_ring_phase_prediction(
    coordinates_angstrom: np.ndarray,
    target_values: np.ndarray,
    evaluate: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    phases: tuple[RingPhaseSpec, ...],
    *,
    fixed_atom_indices: tuple[int, ...] = (),
    max_iterations: int = 6,
    max_cartesian_step_angstrom: float = 0.12,
) -> np.ndarray:
    """Realize explicit phase pairs with local 2x2 solves."""

    blocks = tuple(
        RingPuckeringBlockSpec(
            coordinate_indices=spec.coordinate_indices,
            primitive_ids=spec.primitive_ids,
            atoms=spec.atoms,
        )
        for spec in phases
    )
    return direct_ring_puckering_prediction(
        coordinates_angstrom,
        target_values,
        evaluate,
        blocks,
        fixed_atom_indices=fixed_atom_indices,
        max_iterations=max_iterations,
        max_cartesian_step_angstrom=max_cartesian_step_angstrom,
    )


def direct_ring_puckering_prediction(
    coordinates_angstrom: np.ndarray,
    target_values: np.ndarray,
    evaluate: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    blocks: tuple[RingPuckeringBlockSpec, ...],
    *,
    evaluate_subset: Callable[[np.ndarray, tuple[int, ...]], tuple[np.ndarray, np.ndarray]]
    | None = None,
    evaluate_values_subset: Callable[
        [np.ndarray, tuple[int, ...]], np.ndarray
    ]
    | None = None,
    fixed_atom_indices: tuple[int, ...] = (),
    max_iterations: int = 6,
    max_cartesian_step_angstrom: float = 0.12,
) -> np.ndarray:
    """Realize every RPck block without a pseudoinverse of the complete B.

    Each solve uses only the RPck rows in the block and Cartesian columns of
    its ring atoms.  The dense system has size ``n_block x n_block`` regardless
    of the number of molecular coordinates.
    """

    coords = np.asarray(coordinates_angstrom, dtype=float).copy()
    target = np.asarray(target_values, dtype=float).reshape(-1)
    fixed = {int(index) for index in fixed_atom_indices}
    for block in blocks:
        indices = np.asarray(block.coordinate_indices, dtype=int)
        movable_atoms = tuple(atom - 1 for atom in block.atoms if atom - 1 not in fixed)
        if not movable_atoms:
            continue
        columns = np.asarray(
            [3 * atom + axis for atom in movable_atoms for axis in range(3)], dtype=int
        )
        for _iteration in range(max(int(max_iterations), 1)):
            if evaluate_subset is None:
                values, b_matrix = _validated_evaluation(evaluate, coords, target.size)
                local_values = values[indices]
                local_b = b_matrix[np.ix_(indices, columns)]
            else:
                local_values, subset_b = evaluate_subset(coords, block.coordinate_indices)
                local_values = np.asarray(local_values, dtype=float).reshape(-1)
                subset_b = np.asarray(subset_b, dtype=float)
                if local_values.shape != (indices.size,) or subset_b.shape != (
                    indices.size,
                    coords.size,
                ):
                    raise ValueError("RPck subset evaluator returned incompatible dimensions")
                local_b = subset_b[:, columns]
            residual = target[indices] - local_values
            current_error = float(np.linalg.norm(residual))
            if current_error <= 1.0e-10:
                break
            gram = local_b @ local_b.T
            scale = max(float(np.trace(gram)), 1.0)
            try:
                multipliers = np.linalg.solve(
                    gram + (1.0e-12 * scale) * np.eye(indices.size), residual
                )
            except np.linalg.LinAlgError:
                break
            local_step = local_b.T @ multipliers
            step_norm = float(np.linalg.norm(local_step))
            if not np.isfinite(step_norm) or step_norm <= 1.0e-14:
                break
            if max_cartesian_step_angstrom > 0.0 and step_norm > max_cartesian_step_angstrom:
                local_step *= max_cartesian_step_angstrom / step_norm
            accepted = False
            for fraction in (1.0, 0.5, 0.25, 0.125):
                trial = coords.copy().reshape(-1)
                trial[columns] += fraction * local_step
                trial = trial.reshape(coords.shape)
                trial_local_values = (
                    np.asarray(
                        evaluate_values_subset(
                            trial, block.coordinate_indices
                        ),
                        dtype=float,
                    ).reshape(-1)
                    if evaluate_values_subset is not None
                    else _validated_evaluation(
                        evaluate, trial, target.size
                    )[0][indices]
                )
                trial_error = float(
                    np.linalg.norm(target[indices] - trial_local_values)
                )
                if trial_error + 1.0e-13 < current_error:
                    coords = trial
                    accepted = True
                    break
            if not accepted:
                break
    return coords


def ring_puckering_block_specs(
    definition: "GICDefinition",
) -> tuple[RingPuckeringBlockSpec, ...]:
    """Group native RPCK and U/D combinations by their ring atoms."""

    primitive_by_id = {item.identifier: item for item in definition.primitives}
    primitive_order = {
        item.identifier: position for position, item in enumerate(definition.primitives)
    }
    grouped: dict[tuple[int, ...], list[tuple[int, tuple[str, ...]]]] = {}
    for index, gic in enumerate(definition.gics):
        if str(gic.family).upper() != "RING_PUCKER_COMPONENT":
            continue
        coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
        primitives = tuple(
            primitive_by_id.get(primitive_id) for primitive_id, _coefficient in coefficients
        )
        if not primitives or any(primitive is None for primitive in primitives):
            continue
        typed_primitives = tuple(primitive for primitive in primitives if primitive is not None)
        if any(
            primitive.function not in {"RPCK", "U", "D"}
            or primitive.family != "RING_PUCKER_COMPONENT"
            for primitive in typed_primitives
        ):
            continue
        atoms = tuple(
            sorted({int(atom) for primitive in typed_primitives for atom in primitive.atoms})
        )
        if len(atoms) < 4:
            continue
        aromatic_atoms = frozenset(int(atom) for atom in getattr(definition, "aromatic_atoms", ()))
        if aromatic_atoms and set(atoms).issubset(aromatic_atoms):
            continue
        primitive_ids = tuple(
            sorted(
                (primitive.identifier for primitive in typed_primitives),
                key=primitive_order.__getitem__,
            )
        )
        grouped.setdefault(atoms, []).append((index, primitive_ids))

    return tuple(
        RingPuckeringBlockSpec(
            coordinate_indices=tuple(item[0] for item in components),
            primitive_ids=tuple(
                dict.fromkeys(
                    primitive_id
                    for _index, primitive_ids in components
                    for primitive_id in primitive_ids
                )
            ),
            atoms=atoms,
        )
        for atoms, components in grouped.items()
    )


# Compatibility spelling for callers predating the public compiled plan.
_ring_puckering_block_specs = ring_puckering_block_specs


def _ring_phase_specs(
    definition: "GICDefinition",
    *,
    blocks: tuple[RingPuckeringBlockSpec, ...] | None = None,
) -> tuple[RingPhaseSpec, ...]:
    """Identify isolated RPck cosine/sine pairs eligible for phase-only moves."""

    ring_blocks = ring_puckering_block_specs(definition) if blocks is None else blocks
    condensed = {
        index
        for index, left in enumerate(ring_blocks)
        if any(
            index != other_index and len(set(left.atoms).intersection(right.atoms)) >= 2
            for other_index, right in enumerate(ring_blocks)
        )
    }
    specs: list[RingPhaseSpec] = []
    for block_index, block in enumerate(ring_blocks):
        if block_index in condensed or len(block.atoms) < 5:
            continue
        by_irrep: dict[str, list[int]] = {}
        for coordinate_index in block.coordinate_indices:
            irrep = str(definition.gics[coordinate_index].irrep) if definition.symmetrize else ""
            by_irrep.setdefault(irrep, []).append(coordinate_index)
        for indices in by_irrep.values():
            for offset in range(0, len(indices) - 1, 2):
                pair = (indices[offset], indices[offset + 1])
                pair_primitives = tuple(
                    dict.fromkeys(
                        primitive_id
                        for coordinate_index in pair
                        for primitive_id, _coefficient in (
                            definition.gics[coordinate_index].coefficients
                            or ((definition.gics[coordinate_index].primitive_id, 1.0),)
                        )
                    )
                )
                specs.append(
                    RingPhaseSpec(
                        coordinate_indices=pair,
                        primitive_ids=pair_primitives,
                        atoms=block.atoms,
                    )
                )
    return tuple(specs)


def _interpolate_ring_phase_targets(
    start: np.ndarray,
    target: np.ndarray,
    interpolated: np.ndarray,
    fraction: float,
    phases: tuple[RingPhaseSpec, ...],
) -> np.ndarray:
    result = np.asarray(interpolated, dtype=float).copy()
    for spec in phases:
        left, right = spec.coordinate_indices
        amplitude = float(np.hypot(start[left], start[right]))
        if amplitude <= 1.0e-10:
            result[[left, right]] = start[[left, right]]
            continue
        start_phase = float(np.arctan2(start[right], start[left]))
        target_phase = float(np.arctan2(target[right], target[left]))
        delta = _periodic_difference(target_phase, start_phase)
        phase = start_phase + float(fraction) * delta
        result[left] = amplitude * np.cos(phase)
        result[right] = amplitude * np.sin(phase)
    return result


def direct_acyclic_torsion_prediction(
    coordinates_angstrom: np.ndarray,
    target_values: np.ndarray,
    evaluate: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] | None,
    torsions: tuple[AcyclicTorsionSpec, ...],
    *,
    evaluate_values: Callable[[np.ndarray], np.ndarray] | None = None,
    evaluate_values_subset: Callable[
        [np.ndarray, tuple[int, ...]], np.ndarray
    ]
    | None = None,
) -> np.ndarray:
    """Apply exact finite rotations for independent acyclic dihedrals."""

    coords = np.asarray(coordinates_angstrom, dtype=float).copy()
    target = np.asarray(target_values, dtype=float).reshape(-1)
    if not torsions:
        return coords

    def torsion_value_at(trial: np.ndarray, coordinate_index: int) -> float:
        if evaluate_values_subset is not None:
            values = np.asarray(
                evaluate_values_subset(trial, (coordinate_index,)), dtype=float
            ).reshape(-1)
            if values.shape != (1,) or not np.all(np.isfinite(values)):
                raise ValueError("torsion subset evaluator returned incompatible values")
            return float(values[0])
        if evaluate_values is not None:
            return float(
                _validated_values(evaluate_values, trial, target.size)[coordinate_index]
            )
        if evaluate is None:
            raise ValueError("a torsion value evaluator is required")
        values, _b = _validated_evaluation(evaluate, trial, target.size)
        return float(values[coordinate_index])

    # Nested bridge torsions share downstream atoms.  Apply the largest
    # moving subgraph first so subsequent inner rotations do not overwrite an
    # already imposed outer dihedral; the repeated passes then remove the
    # residual coupling deterministically for any number of torsions.
    ordered_torsions = tuple(
        sorted(
            torsions,
            key=lambda spec: (-len(spec.moving_atoms), spec.coordinate_index),
        )
    )
    for _pass in range(3):
        improved = False
        for spec in ordered_torsions:
            current_error = _periodic_difference(
                target[spec.coordinate_index],
                torsion_value_at(coords, spec.coordinate_index),
                period=spec.period_radian,
            )
            if abs(current_error) <= 1.0e-11:
                continue
            trial = _rotate_acyclic_torsion(coords, spec, current_error)
            if trial is not None:
                coords = trial
                improved = True
        if not improved:
            break
    return coords


def direct_acyclic_torsion_step(
    definition: "GICDefinition",
    coordinates_angstrom: np.ndarray,
    target_values: np.ndarray,
    evaluate_values: Callable[[np.ndarray], np.ndarray],
    *,
    fixed_atom_indices: tuple[int, ...] = (),
    tolerance: float = 1.0e-9,
    torsions: tuple[AcyclicTorsionSpec, ...] | None = None,
) -> DirectAcyclicTorsionBackTransform | None:
    """Realize a pure rigid acyclic-torsion move without constructing a B matrix.

    The fast path is deliberately conservative.  Every requested displacement
    must belong to a one-primitive dihedral whose central bond separates the
    molecular graph.  After the finite subgraph rotation, every coordinate in
    the frozen SONIC contract must match its target.  Mixed coordinates, ring
    torsions, coupled rotations, or any unintended drift return ``None`` and
    therefore retain the complete hybrid SONIC fallback.
    """

    coords = np.asarray(coordinates_angstrom, dtype=float)
    target = np.asarray(target_values, dtype=float).reshape(-1)
    if target.shape != (len(definition.gics),):
        return None
    values = _validated_values(evaluate_values, coords, target.size)
    torsions = (
        acyclic_torsion_specs(
            definition,
            natoms=coords.shape[0],
            fixed_atom_indices=fixed_atom_indices,
        )
        if torsions is None
        else torsions
    )
    if not torsions:
        return None
    torsion_by_index = {item.coordinate_index: item for item in torsions}
    residual = _coordinate_residual(definition, target, values)
    changed = tuple(int(index) for index in np.flatnonzero(np.abs(residual) > tolerance))
    if not changed or any(index not in torsion_by_index for index in changed):
        return None
    selected = tuple(
        sorted(
            (torsion_by_index[index] for index in changed),
            key=lambda spec: (-len(spec.moving_atoms), spec.coordinate_index),
        )
    )
    realized = coords.copy()
    for spec in selected:
        current_error = _periodic_difference(
            target[spec.coordinate_index],
            values[spec.coordinate_index],
            period=spec.period_radian,
        )
        trial = _rotate_acyclic_torsion(realized, spec, current_error)
        if trial is None:
            return None
        realized = trial
    final_values = _validated_values(evaluate_values, realized, target.size)
    final_residual = _coordinate_residual(definition, target, final_values)
    if float(np.max(np.abs(final_residual), initial=0.0)) > tolerance:
        return None
    return DirectAcyclicTorsionBackTransform(
        coordinates_angstrom=realized,
        values=final_values,
        residual=final_residual,
        torsion_indices=changed,
    )


def direct_rigid_soft_step(
    definition: "GICDefinition",
    coordinates_angstrom: np.ndarray,
    target_values: np.ndarray,
    evaluate_values: Callable[[np.ndarray], np.ndarray],
    *,
    evaluate_subset: Callable[
        [np.ndarray, tuple[int, ...]], tuple[np.ndarray, np.ndarray]
    ]
    | None = None,
    evaluate_values_subset: Callable[
        [np.ndarray, tuple[int, ...]], np.ndarray
    ]
    | None = None,
    rigid_model: "RigidComplexModel | None" = None,
    rigid_target_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    torsions: tuple[AcyclicTorsionSpec, ...] | None = None,
    ring_blocks: tuple[RingPuckeringBlockSpec, ...] = (),
    current_values: np.ndarray | None = None,
    fixed_atom_indices: tuple[int, ...] = (),
    tolerance: float = 1.0e-9,
) -> DirectRigidSoftBackTransform | None:
    """Realize rigid soft coordinates with only ring-local Wilson rows.

    Bridge torsions and fragment poses are reconstructed analytically.  Native
    ring-puckering blocks use their local Wilson rows, while line-search trials
    evaluate values only.  A final full value check certifies that inactive
    coordinates stayed fixed; otherwise the caller retains the complete SONIC
    fallback.
    """

    coords = np.asarray(coordinates_angstrom, dtype=float)
    target = np.asarray(target_values, dtype=float).reshape(-1)
    if target.shape != (len(definition.gics),):
        return None
    compiled_torsions = (
        acyclic_torsion_specs(
            definition,
            natoms=coords.shape[0],
            fixed_atom_indices=fixed_atom_indices,
        )
        if torsions is None
        else torsions
    )
    torsion_by_index = {
        item.coordinate_index: item for item in compiled_torsions
    }
    fragment_indices = (
        () if rigid_model is None else rigid_model.coordinate_indices
    )
    ring_indices = tuple(
        index for block in ring_blocks for index in block.coordinate_indices
    )
    supported = set((*torsion_by_index, *fragment_indices, *ring_indices))
    values = (
        _validated_values(evaluate_values, coords, target.size)
        if current_values is None
        else _validated_values_array(current_values, target.size)
    )
    effective_target = (
        ring_phase_only_targets(definition, values, target)
        if ring_blocks
        else target
    )
    residual = _coordinate_residual(definition, effective_target, values)
    changed = tuple(
        int(index) for index in np.flatnonzero(np.abs(residual) > tolerance)
    )
    if not changed or any(index not in supported for index in changed):
        return None

    selected_torsions = tuple(
        torsion_by_index[index] for index in changed if index in torsion_by_index
    )
    ring_index_set = set(ring_indices)
    changed_ring_indices = tuple(
        index for index in changed if index in ring_index_set
    )
    realized = coords.copy()
    if changed_ring_indices:
        if evaluate_subset is None:
            return None
        realized = direct_ring_puckering_prediction(
            realized,
            effective_target,
            lambda trial: (
                _validated_values(evaluate_values, trial, target.size),
                np.zeros((target.size, trial.size), dtype=float),
            ),
            ring_blocks,
            evaluate_subset=evaluate_subset,
            evaluate_values_subset=evaluate_values_subset,
            fixed_atom_indices=fixed_atom_indices,
        )
    torsion_values = values
    if changed_ring_indices and selected_torsions:
        selected_indices = tuple(
            item.coordinate_index for item in selected_torsions
        )
        selected_values = (
            np.asarray(
                evaluate_values_subset(realized, selected_indices),
                dtype=float,
            ).reshape(-1)
            if evaluate_values_subset is not None
            else _validated_values(
                evaluate_values, realized, target.size
            )[list(selected_indices)]
        )
        torsion_values = values.copy()
        torsion_values[list(selected_indices)] = selected_values
    # Rotations around graph bridges preserve every other internal coordinate,
    # so all finite angles can be applied from the single initial value pass.
    for spec in selected_torsions:
        trial = _rotate_acyclic_torsion(
            realized,
            spec,
            _periodic_difference(
                effective_target[spec.coordinate_index],
                torsion_values[spec.coordinate_index],
                period=spec.period_radian,
            ),
        )
        if trial is None:
            return None
        realized = trial
    if rigid_model is not None:
        rigid_target = (
            effective_target
            if rigid_target_transform is None
            else np.asarray(rigid_target_transform(effective_target), dtype=float)
        )
        realized = rigid_model.realize_sonic_from_base(
            rigid_target, realized
        )

    final_values = _validated_values(evaluate_values, realized, target.size)
    final_residual = _coordinate_residual(
        definition, effective_target, final_values
    )
    if float(np.max(np.abs(final_residual), initial=0.0)) > tolerance:
        return None
    fragment_index_set = set(fragment_indices)
    changed_fragments = tuple(index for index in changed if index in fragment_index_set)
    if changed_ring_indices:
        method = (
            "DIRECT_RIGID_RING_PUCKERING"
            if not selected_torsions and not changed_fragments
            else "DIRECT_RIGID_RING_SOFT_COORDINATES"
        )
    elif selected_torsions and not changed_fragments:
        method = "DIRECT_RIGID_ACYCLIC_TORSION"
    elif changed_fragments and not selected_torsions:
        method = "DIRECT_RIGID_FRAGMENT_POSE"
    else:
        method = "DIRECT_RIGID_SOFT_COORDINATES"
    return DirectRigidSoftBackTransform(
        coordinates_angstrom=realized,
        values=final_values,
        residual=final_residual,
        torsion_indices=tuple(
            item.coordinate_index for item in selected_torsions
        ),
        fragment_indices=changed_fragments,
        ring_indices=changed_ring_indices,
        method=method,
    )


def _rotate_acyclic_torsion(
    coordinates_angstrom: np.ndarray,
    spec: AcyclicTorsionSpec,
    coordinate_error: float,
) -> np.ndarray | None:
    """Apply the unique finite rotation that changes ``spec`` by ``error``."""

    coords = np.asarray(coordinates_angstrom, dtype=float)
    j = spec.atoms[1] - 1
    k = spec.atoms[2] - 1
    axis = coords[k] - coords[j]
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1.0e-12:
        return None
    axis /= axis_norm
    moving = np.asarray(spec.moving_atoms, dtype=int)
    counter = np.asarray(spec.counter_moving_atoms, dtype=int)
    # A bridge torsion can be realized by rotating either graph component.
    # Rotating only the side containing atom l can move almost the entire
    # molecule when the opposite side is a methyl group.  Split the relative
    # rotation between both components in inverse proportion to their axial
    # moments.  This is the minimum-displacement rigid realization and is the
    # finite counterpart of the Wilson pseudoinverse tangent.  A fixed atom
    # leaves ``counter_moving_atoms`` empty and retains the one-sided path.
    moving_fraction, counter_fraction = _torsion_rotation_fractions(
        coords,
        spec,
        axis=axis,
        origin=coords[j],
    )
    orientation = 1.0 if spec.atoms[3] - 1 in spec.moving_atoms else -1.0
    physical_delta = orientation * float(coordinate_error) / spec.coefficient
    trial = coords.copy()
    centered = trial[moving] - coords[j]
    trial[moving] = (
        centered
        @ rotation_matrix_from_vector(moving_fraction * physical_delta * axis)
        + coords[j]
    )
    if counter.size:
        centered = trial[counter] - coords[j]
        trial[counter] = (
            centered
            @ rotation_matrix_from_vector(-counter_fraction * physical_delta * axis)
            + coords[j]
        )
    return trial


def acyclic_torsion_cartesian_tangent(
    coordinates_angstrom: np.ndarray,
    spec: AcyclicTorsionSpec,
) -> np.ndarray | None:
    """Return the exact local derivative used by the finite torsion predictor.

    The Cartesian gauge (which side of the bridge moves) is deliberately the
    same as :func:`_rotate_acyclic_torsion`; using a Wilson pseudoinverse here
    would represent the same internal displacement with a different Cartesian
    null-space component and break derivative/backtransform coherence.
    """

    coords = np.asarray(coordinates_angstrom, dtype=float)
    j = spec.atoms[1] - 1
    k = spec.atoms[2] - 1
    axis = coords[k] - coords[j]
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1.0e-12:
        return None
    axis /= axis_norm
    orientation = 1.0 if spec.atoms[3] - 1 in spec.moving_atoms else -1.0
    moving_fraction, counter_fraction = _torsion_rotation_fractions(
        coords,
        spec,
        axis=axis,
        origin=coords[j],
    )
    scale = orientation * moving_fraction / spec.coefficient
    skew = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=float,
    )
    displacement = np.zeros_like(coords)
    moving = np.asarray(spec.moving_atoms, dtype=int)
    displacement[moving] = (coords[moving] - coords[j]) @ (-skew) * scale
    counter = np.asarray(spec.counter_moving_atoms, dtype=int)
    if counter.size:
        counter_scale = -orientation * counter_fraction / spec.coefficient
        displacement[counter] = (
            (coords[counter] - coords[j]) @ (-skew) * counter_scale
        )
    return displacement.reshape(-1)


def _torsion_rotation_fractions(
    coordinates_angstrom: np.ndarray,
    spec: AcyclicTorsionSpec,
    *,
    axis: np.ndarray,
    origin: np.ndarray,
) -> tuple[float, float]:
    """Return the minimum-displacement split of one relative rotation."""

    if not spec.counter_moving_atoms:
        return 1.0, 0.0

    coords = np.asarray(coordinates_angstrom, dtype=float)

    def axial_moment(indices: tuple[int, ...]) -> float:
        offsets = coords[np.asarray(indices, dtype=int)] - origin
        perpendicular = offsets - np.outer(offsets @ axis, axis)
        return float(np.sum(perpendicular * perpendicular))

    moving_moment = axial_moment(spec.moving_atoms)
    counter_moment = axial_moment(spec.counter_moving_atoms)
    total = moving_moment + counter_moment
    if not np.isfinite(total) or total <= 1.0e-20:
        return 0.5, 0.5
    return counter_moment / total, moving_moment / total


def acyclic_torsion_specs(
    definition: "GICDefinition",
    *,
    natoms: int,
    fixed_atom_indices: tuple[int, ...],
) -> tuple[AcyclicTorsionSpec, ...]:
    primitive_by_id = {item.identifier: item for item in definition.primitives}
    adjacency = [set() for _ in range(natoms)]
    for primitive in definition.primitives:
        if primitive.function == "R" and len(primitive.atoms) == 2:
            left, right = (atom - 1 for atom in primitive.atoms)
            if 0 <= left < natoms and 0 <= right < natoms and left != right:
                adjacency[left].add(right)
                adjacency[right].add(left)
    fixed = set(int(item) for item in fixed_atom_indices)
    periodicity_by_identifier = {
        str(estimate.coordinate_identifier): max(int(estimate.periodicity), 1)
        for estimate in definition.periodic_coordinate_estimates
        if str(estimate.coordinate_domain).strip().upper() == "PERIODIC_2PI"
    }
    specs: list[AcyclicTorsionSpec] = []
    for index, gic in enumerate(definition.gics):
        coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
        if len(coefficients) != 1:
            continue
        primitive = primitive_by_id.get(coefficients[0][0])
        coefficient = float(coefficients[0][1])
        if (
            primitive is None
            or primitive.function != "D"
            or len(primitive.atoms) != 4
            or abs(coefficient) <= 1.0e-12
        ):
            continue
        j, k = primitive.atoms[1] - 1, primitive.atoms[2] - 1
        if k not in adjacency[j]:
            adjacency[j].add(k)
            adjacency[k].add(j)
        component_k = _component_without_edge(adjacency, k, (j, k))
        if j in component_k:
            continue
        component_j = _component_without_edge(adjacency, j, (j, k))
        fourth = primitive.atoms[3] - 1
        moving = component_k if fourth in component_k else component_j
        alternate = component_j if moving is component_k else component_k
        counter_moving: set[int] = set(alternate)
        if fixed.intersection(moving):
            if fixed.intersection(alternate):
                continue
            moving = alternate
            counter_moving = set()
        elif fixed.intersection(alternate):
            counter_moving = set()
        specs.append(
            AcyclicTorsionSpec(
                coordinate_index=index,
                primitive_id=primitive.identifier,
                atoms=tuple(int(atom) for atom in primitive.atoms),  # type: ignore[arg-type]
                coefficient=coefficient,
                moving_atoms=tuple(sorted(moving)),
                counter_moving_atoms=tuple(sorted(counter_moving)),
                period_radian=(
                    2.0
                    * np.pi
                    / periodicity_by_identifier.get(str(gic.identifier), 1)
                ),
            )
        )
    return tuple(specs)


# Compatibility spelling for internal callers predating the public compiler.
_acyclic_torsion_specs = acyclic_torsion_specs


def soft_coordinate_indices(definition: "GICDefinition") -> tuple[int, ...]:
    """Return SONIC rows requiring finite/continuation rather than hard correction.

    This is the single LINK hard/soft classification used both by the hybrid
    back-transform and by adaptive finite-difference step selection.
    """

    primitive_by_id = {item.identifier: item for item in definition.primitives}
    soft: list[int] = []
    for index, gic in enumerate(definition.gics):
        family = str(gic.family).upper()
        coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
        functions = {
            primitive.function
            for primitive_id, _coefficient in coefficients
            if (primitive := primitive_by_id.get(primitive_id)) is not None
        }
        if functions.intersection(_SOFT_FUNCTIONS) or any(
            token in family for token in _SOFT_FAMILY_TOKENS
        ):
            soft.append(index)
    return tuple(soft)


# Internal spelling retained while callers migrate to the public contract.
_soft_coordinate_indices = soft_coordinate_indices


def _component_without_edge(
    adjacency: list[set[int]],
    start: int,
    removed_edge: tuple[int, int],
) -> set[int]:
    removed = {tuple(removed_edge), tuple(reversed(removed_edge))}
    seen = {start}
    stack = [start]
    while stack:
        atom = stack.pop()
        for neighbor in adjacency[atom]:
            if (atom, neighbor) in removed or neighbor in seen:
                continue
            seen.add(neighbor)
            stack.append(neighbor)
    return seen


def _periodic_difference(
    target: float,
    value: float,
    period: float = 2.0 * np.pi,
) -> float:
    from .periodic import gdv_match_dihedral_phase

    return gdv_match_dihedral_phase(target, value, period) - float(value)


def _coordinate_residual(
    definition: "GICDefinition",
    target: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    residual = np.asarray(target, dtype=float) - np.asarray(values, dtype=float)
    primitive_by_id = {item.identifier: item for item in definition.primitives}
    periodicity_by_identifier = {
        str(estimate.coordinate_identifier): max(int(estimate.periodicity), 1)
        for estimate in definition.periodic_coordinate_estimates
        if str(estimate.coordinate_domain).strip().upper() == "PERIODIC_2PI"
    }
    for index, gic in enumerate(definition.gics):
        coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
        if len(coefficients) != 1:
            continue
        primitive = primitive_by_id.get(coefficients[0][0])
        if primitive is not None and primitive.function == "D":
            period = 2.0 * np.pi / periodicity_by_identifier.get(
                str(gic.identifier), 1
            )
            residual[index] = _periodic_difference(
                target[index], values[index], period=period
            )
    return residual


def _validated_values(
    evaluate_values: Callable[[np.ndarray], np.ndarray],
    coordinates: np.ndarray,
    coordinate_count: int,
) -> np.ndarray:
    values = evaluate_values(np.asarray(coordinates, dtype=float))
    return _validated_values_array(values, coordinate_count)


def _validated_values_array(
    values: np.ndarray,
    coordinate_count: int,
) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.shape != (coordinate_count,):
        raise ValueError("internal-coordinate value evaluator returned incompatible dimensions")
    if not np.all(np.isfinite(values)):
        raise ValueError("internal-coordinate value evaluator returned non-finite values")
    return values


def _validated_evaluation(
    evaluate: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    coordinates: np.ndarray,
    coordinate_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    values, b_matrix = evaluate(np.asarray(coordinates, dtype=float))
    values = np.asarray(values, dtype=float).reshape(-1)
    b_matrix = np.asarray(b_matrix, dtype=float)
    if values.shape != (coordinate_count,) or b_matrix.shape != (
        coordinate_count,
        coordinates.size,
    ):
        raise ValueError("internal-coordinate evaluator returned incompatible dimensions")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(b_matrix)):
        raise ValueError("internal-coordinate evaluator returned non-finite values")
    return values, b_matrix
