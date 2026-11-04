"""LINK direct rigid-fragment predictors for SONIC realization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

import numpy as np

from matrix_chem import rotation_matrix_from_vector

if TYPE_CHECKING:
    from matrix_smith.models import GICDefinition


@dataclass(frozen=True)
class FragmentRigidPrediction:
    coordinates_angstrom: np.ndarray
    handled_indices: tuple[int, ...]


@dataclass(frozen=True)
class FragmentRigidTangent:
    """Cartesian tangents consistent with the finite fragment predictor."""

    cartesian_from_q: np.ndarray
    handled_indices: tuple[int, ...]


def direct_fragment_rigid_tangent(
    definition: "GICDefinition",
    coordinates_angstrom: np.ndarray,
    b_matrix: np.ndarray,
    *,
    fixed_atom_indices: tuple[int, ...] = (),
) -> FragmentRigidTangent:
    """Return ``dx/dq`` columns for complete FTRANS/FROT triplets.

    The columns are the differential of the same rigid translations and
    exponential-map rotations used by :func:`direct_fragment_rigid_prediction`.
    They therefore provide the correct pullback of a Cartesian gradient for
    special fragment coordinates; a global Moore--Penrose inverse need not
    choose the same tangent in this subspace.

    The physical generators are analytic.  Their small dense Jacobian in the
    stored SONIC chart is obtained by contracting them with the analytic B
    matrix, so no finite-difference derivative is introduced here.
    """

    coords = np.asarray(coordinates_angstrom, dtype=float)
    matrix = np.asarray(b_matrix, dtype=float)
    nq = len(definition.gics)
    ncart = coords.size
    if matrix.shape != (nq, ncart):
        raise ValueError(f"fragment tangent needs a B matrix of shape {(nq, ncart)}")
    fixed = {int(index) for index in fixed_atom_indices}
    physical_columns: list[np.ndarray] = []
    handled: list[int] = []

    for (function, atoms, ref_atoms), modes in _unit_fragment_groups(definition).items():
        if set(modes) != {0, 1, 2}:
            continue
        atom_indices = np.asarray([atom - 1 for atom in atoms], dtype=int)
        ref_indices = np.asarray([atom - 1 for atom in ref_atoms], dtype=int)
        if fixed.intersection(atom_indices.tolist()) or fixed.intersection(ref_indices.tolist()):
            continue
        moving_share = len(ref_atoms) / float(len(atoms) + len(ref_atoms))
        reference_share = len(atoms) / float(len(atoms) + len(ref_atoms))
        generators = np.zeros((ncart, 3), dtype=float)
        if function == "FTRANS":
            axes = _fragment_translation_axes(coords, ref_atoms, modes[0][1])
            for axis in range(3):
                displacement = np.zeros_like(coords)
                displacement[atom_indices, :] = moving_share * axes[:, axis]
                displacement[ref_indices, :] = -reference_share * axes[:, axis]
                generators[:, axis] = displacement.reshape(-1)
        else:
            center = np.mean(coords[atom_indices, :], axis=0)
            centered = coords[atom_indices, :] - center
            ref_center = np.mean(coords[ref_indices, :], axis=0)
            ref_centered = coords[ref_indices, :] - ref_center
            for axis in range(3):
                unit = np.zeros(3, dtype=float)
                unit[axis] = 1.0
                skew = np.asarray(
                    [
                        [0.0, -unit[2], unit[1]],
                        [unit[2], 0.0, -unit[0]],
                        [-unit[1], unit[0], 0.0],
                    ],
                    dtype=float,
                )
                displacement = np.zeros_like(coords)
                # MATRIX uses row-vector rotations R(v)=I-[v]_x+O(v^2).
                displacement[atom_indices, :] = moving_share * (centered @ (-skew))
                displacement[ref_indices, :] = reference_share * (ref_centered @ skew)
                generators[:, axis] = displacement.reshape(-1)
        physical_columns.extend(generators[:, axis] for axis in range(3))
        handled.extend(modes[mode][0] for mode in range(3))

    # The minimum chart uses a radial center distance plus two transverse
    # FTRANS components.  It is the same three-dimensional relative
    # translation space as an FTRANS triplet and therefore admits the same
    # direct analytic tangent without a global pseudoinverse.
    for (atoms, ref_atoms), members in _radial_transverse_groups(definition).items():
        atom_indices = np.asarray([atom - 1 for atom in atoms], dtype=int)
        ref_indices = np.asarray([atom - 1 for atom in ref_atoms], dtype=int)
        if fixed.intersection(atom_indices.tolist()) or fixed.intersection(ref_indices.tolist()):
            continue
        moving_share = len(ref_atoms) / float(len(atoms) + len(ref_atoms))
        reference_share = len(atoms) / float(len(atoms) + len(ref_atoms))
        generators = np.zeros((ncart, 3), dtype=float)
        for axis in range(3):
            displacement = np.zeros_like(coords)
            displacement[atom_indices, axis] = moving_share
            displacement[ref_indices, axis] = -reference_share
            generators[:, axis] = displacement.reshape(-1)
        indices = [index for index, _primitive in members]
        local_jacobian = matrix[np.asarray(indices, dtype=int), :] @ generators
        if np.linalg.matrix_rank(local_jacobian, tol=1.0e-10) != 3:
            continue
        physical_columns.extend(generators[:, axis] for axis in range(3))
        handled.extend(indices)

    empty = np.zeros((ncart, nq), dtype=float)
    if not handled:
        return FragmentRigidTangent(empty, ())

    physical = np.column_stack(physical_columns)
    handled_array = np.asarray(handled, dtype=int)
    chart_jacobian = matrix[handled_array, :] @ physical
    if np.linalg.matrix_rank(chart_jacobian, tol=1.0e-10) != len(handled):
        return FragmentRigidTangent(empty, ())
    mapped = physical @ np.linalg.solve(chart_jacobian, np.eye(len(handled)))

    selector = np.zeros((nq, len(handled)), dtype=float)
    selector[handled_array, np.arange(len(handled))] = 1.0
    mapped = _constrained_fragment_tangent(
        matrix,
        mapped,
        handled_array,
        selector,
    )
    if mapped is None:
        return FragmentRigidTangent(empty, ())
    result = empty
    result[:, handled_array] = mapped
    return FragmentRigidTangent(result, tuple(int(index) for index in handled))


def _constrained_fragment_tangent(
    b_matrix: np.ndarray,
    physical_tangent: np.ndarray,
    handled_indices: np.ndarray,
    selector: np.ndarray,
) -> np.ndarray | None:
    """Apply the linear hard-coordinate corrector used by the finite predictor."""

    matrix = np.asarray(b_matrix, dtype=float)
    mapped = np.asarray(physical_tangent, dtype=float)
    residual = np.asarray(selector, dtype=float) - matrix @ mapped
    if float(np.max(np.abs(residual))) <= 1.0e-10:
        return mapped

    protected = matrix[handled_indices, :]
    _u, singular, vh = np.linalg.svd(protected, full_matrices=True)
    cutoff = 1.0e-10 * max(float(singular[0]) if singular.size else 0.0, 1.0)
    rank = int(np.count_nonzero(singular > cutoff))
    null_basis = vh[rank:, :].T
    if null_basis.shape[1] == 0:
        return None

    handled = {int(index) for index in handled_indices}
    solve_indices = np.asarray(
        [index for index in range(matrix.shape[0]) if index not in handled],
        dtype=int,
    )
    reduced = matrix[solve_indices, :] @ null_basis
    correction = null_basis @ np.linalg.pinv(reduced, rcond=1.0e-10) @ residual[solve_indices, :]
    candidate = mapped + correction
    if float(np.max(np.abs(matrix @ candidate - selector))) > 1.0e-7:
        return None
    return candidate


def direct_fragment_rigid_prediction(
    definition: "GICDefinition",
    coordinates_angstrom: np.ndarray,
    target_values: np.ndarray,
    evaluate_values: Callable[[np.ndarray], np.ndarray],
    *,
    evaluate_subset: Callable[[np.ndarray, tuple[int, ...]], tuple[np.ndarray, np.ndarray]]
    | None = None,
    evaluate_values_subset: Callable[[np.ndarray, tuple[int, ...]], np.ndarray] | None = None,
) -> FragmentRigidPrediction:
    """Impose complete FTRANS/FROT triplets by fragment rigid motions.

    Only unit, one-primitive GIC rows are eligible.  Incomplete or symmetry
    mixed blocks are left to the general nonlinear back-transformer.
    """

    coords = np.asarray(coordinates_angstrom, dtype=float).copy()
    target = np.asarray(target_values, dtype=float).reshape(-1)
    handled: list[int] = []
    groups = _unit_fragment_groups(definition)
    # A body-fixed translation is measured in the final reference-fragment
    # frame.  Realize orientation first and translation second; the reverse
    # order lets the subsequent reference-frame rotation change an already
    # satisfied FTRANS target.  The ordering is representation-level and
    # applies uniformly to every rigid complex.
    ordered_groups = sorted(
        groups.items(),
        key=lambda item: 0 if item[0][0] == "FROT" else 1,
    )
    for (function, atoms, ref_atoms), modes in ordered_groups:
        if set(modes) != {0, 1, 2}:
            continue
        indices = [modes[mode][0] for mode in range(3)]
        index_tuple = tuple(indices)
        atom_indices = np.asarray([atom - 1 for atom in atoms], dtype=int)
        ref_indices = np.asarray([atom - 1 for atom in ref_atoms], dtype=int)
        moving_share = len(ref_atoms) / float(len(atoms) + len(ref_atoms))
        reference_share = len(atoms) / float(len(atoms) + len(ref_atoms))
        if evaluate_values_subset is not None:
            current = np.asarray(
                evaluate_values_subset(coords, index_tuple), dtype=float
            ).reshape(3)
            local_b = None
        elif evaluate_subset is None:
            current = np.asarray(evaluate_values(coords), dtype=float)[indices]
            local_b = None
        else:
            current, local_b = evaluate_subset(coords, index_tuple)
            current = np.asarray(current, dtype=float).reshape(3)
            local_b = np.asarray(local_b, dtype=float)
        delta = target[indices] - current
        if function == "FTRANS":
            axes = _fragment_translation_axes(coords, ref_atoms, modes[0][1])
            laboratory_delta = axes @ delta
            coords[atom_indices, :] += moving_share * laboratory_delta
            coords[ref_indices, :] -= reference_share * laboratory_delta
        else:
            for _iteration in range(4):
                if evaluate_values_subset is not None:
                    current = np.asarray(
                        evaluate_values_subset(coords, index_tuple), dtype=float
                    ).reshape(3)
                    local_b = None
                elif evaluate_subset is None:
                    full_current = np.asarray(evaluate_values(coords), dtype=float)
                    current = full_current[indices]
                    local_b = None
                else:
                    current, local_b = evaluate_subset(coords, index_tuple)
                    current = np.asarray(current, dtype=float).reshape(3)
                    local_b = np.asarray(local_b, dtype=float)
                delta = target[indices] - current
                if float(np.linalg.norm(delta)) <= 1.0e-10:
                    break
                center = np.mean(coords[atom_indices, :], axis=0)
                centered = coords[atom_indices, :] - center
                ref_center = np.mean(coords[ref_indices, :], axis=0)
                ref_centered = coords[ref_indices, :] - ref_center

                def rotated(increment: np.ndarray) -> np.ndarray:
                    trial = coords.copy()
                    trial[atom_indices, :] = (
                        centered @ rotation_matrix_from_vector(moving_share * increment) + center
                    )
                    trial[ref_indices, :] = (
                        ref_centered @ rotation_matrix_from_vector(-reference_share * increment)
                        + ref_center
                    )
                    return trial

                if local_b is None:
                    jacobian = np.zeros((3, 3), dtype=float)
                    epsilon = 1.0e-5
                    for axis in range(3):
                        increment = np.zeros(3, dtype=float)
                        increment[axis] = epsilon
                        trial = rotated(increment)
                        trial_values = (
                            np.asarray(
                                evaluate_values_subset(trial, index_tuple),
                                dtype=float,
                            ).reshape(3)
                            if evaluate_values_subset is not None
                            else np.asarray(evaluate_values(trial), dtype=float)[
                                indices
                            ]
                        )
                        jacobian[:, axis] = (
                            trial_values - current
                        ) / epsilon
                else:
                    generators = np.zeros((coords.size, 3), dtype=float)
                    for axis in range(3):
                        unit = np.zeros(3, dtype=float)
                        unit[axis] = 1.0
                        skew = np.asarray(
                            [
                                [0.0, -unit[2], unit[1]],
                                [unit[2], 0.0, -unit[0]],
                                [-unit[1], unit[0], 0.0],
                            ],
                            dtype=float,
                        )
                        displacement = np.zeros_like(coords)
                        displacement[atom_indices, :] = moving_share * (centered @ (-skew))
                        displacement[ref_indices, :] = reference_share * (ref_centered @ skew)
                        generators[:, axis] = displacement.reshape(-1)
                    jacobian = local_b @ generators
                try:
                    physical_increment = np.linalg.solve(jacobian, delta)
                except np.linalg.LinAlgError:
                    physical_increment = np.linalg.pinv(jacobian, rcond=1.0e-10) @ delta
                candidates = []
                for scale in (1.0, 0.5, 0.25):
                    trial = rotated(scale * physical_increment)
                    if evaluate_values_subset is not None:
                        trial_values = np.asarray(
                            evaluate_values_subset(trial, index_tuple), dtype=float
                        ).reshape(3)
                    elif evaluate_subset is not None:
                        trial_values = np.asarray(
                            evaluate_subset(trial, index_tuple)[0], dtype=float
                        ).reshape(3)
                    else:
                        trial_values = np.asarray(evaluate_values(trial), dtype=float)[indices]
                    residual = target[indices] - trial_values
                    candidates.append((float(np.linalg.norm(residual)), trial))
                best_norm, best_trial = min(candidates, key=lambda item: item[0])
                if best_norm >= float(np.linalg.norm(delta)):
                    break
                coords = best_trial
        # A fragment triplet is protected from the nonlinear corrector only
        # when its finite rigid predictor actually reached the requested
        # values.  During reactive rearrangements a fragment's canonical
        # frame can become ill-conditioned; marking a failed FROT predictor as
        # handled then removes those residuals from the only solver capable of
        # resolving them and creates a false optimizer plateau.
        if evaluate_values_subset is not None:
            realized = np.asarray(
                evaluate_values_subset(coords, index_tuple), dtype=float
            ).reshape(3)
        elif evaluate_subset is not None:
            realized = np.asarray(evaluate_subset(coords, index_tuple)[0], dtype=float).reshape(3)
        else:
            realized = np.asarray(evaluate_values(coords), dtype=float)[indices]
        residual_norm = float(np.linalg.norm(target[indices] - realized))
        target_change = float(np.linalg.norm(target[indices] - current))
        if residual_norm <= max(1.0e-9, 1.0e-7 * max(1.0, target_change)):
            handled.extend(indices)

    for (atoms, ref_atoms), members in _radial_transverse_groups(definition).items():
        indices = [index for index, _primitive in members]
        index_tuple = tuple(indices)
        atom_indices = np.asarray([atom - 1 for atom in atoms], dtype=int)
        ref_indices = np.asarray([atom - 1 for atom in ref_atoms], dtype=int)
        moving_share = len(ref_atoms) / float(len(atoms) + len(ref_atoms))
        reference_share = len(atoms) / float(len(atoms) + len(ref_atoms))

        def current_and_b(trial: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
            if evaluate_values_subset is not None:
                return (
                    np.asarray(evaluate_values_subset(trial, index_tuple), dtype=float).reshape(3),
                    None,
                )
            if evaluate_subset is not None:
                values, local = evaluate_subset(trial, index_tuple)
                return np.asarray(values, dtype=float).reshape(3), np.asarray(local, dtype=float)
            return np.asarray(evaluate_values(trial), dtype=float)[indices], None

        initial, _initial_b = current_and_b(coords)
        for _iteration in range(5):
            current, local_b = current_and_b(coords)
            delta = target[indices] - current
            if float(np.linalg.norm(delta)) <= 1.0e-10:
                break
            generators = np.zeros((coords.size, 3), dtype=float)
            for axis in range(3):
                displacement = np.zeros_like(coords)
                displacement[atom_indices, axis] = moving_share
                displacement[ref_indices, axis] = -reference_share
                generators[:, axis] = displacement.reshape(-1)
            if local_b is None:
                jacobian = np.zeros((3, 3), dtype=float)
                epsilon = 1.0e-6
                for axis in range(3):
                    trial = coords.copy()
                    trial.reshape(-1)[:] += epsilon * generators[:, axis]
                    trial_values, _trial_b = current_and_b(trial)
                    jacobian[:, axis] = (trial_values - current) / epsilon
            else:
                jacobian = local_b @ generators
            try:
                laboratory_delta = np.linalg.solve(jacobian, delta)
            except np.linalg.LinAlgError:
                laboratory_delta = np.linalg.pinv(jacobian, rcond=1.0e-10) @ delta
            candidates = []
            for scale in (1.0, 0.5, 0.25):
                trial = coords.copy().reshape(-1)
                trial += generators @ (scale * laboratory_delta)
                trial = trial.reshape(coords.shape)
                trial_values, _trial_b = current_and_b(trial)
                candidates.append((float(np.linalg.norm(target[indices] - trial_values)), trial))
            best_norm, best_trial = min(candidates, key=lambda item: item[0])
            if best_norm >= float(np.linalg.norm(delta)):
                break
            coords = best_trial
        realized, _realized_b = current_and_b(coords)
        residual_norm = float(np.linalg.norm(target[indices] - realized))
        target_change = float(np.linalg.norm(target[indices] - initial))
        if residual_norm <= max(1.0e-9, 1.0e-7 * max(1.0, target_change)):
            handled.extend(indices)
    return FragmentRigidPrediction(coords, tuple(sorted(set(handled))))


def _fragment_translation_axes(
    coordinates_angstrom: np.ndarray,
    ref_atoms: tuple[int, ...],
    primitive: object,
) -> np.ndarray:
    ref_frame_atoms = tuple(getattr(primitive, "ref_frame_atoms", ()))
    if not ref_frame_atoms:
        return np.eye(3, dtype=float)
    from matrix_smith.numerics import _fragment_frame

    return np.asarray(
        _fragment_frame(
            np.asarray(coordinates_angstrom, dtype=float),
            ref_atoms,
            frame_atoms=ref_frame_atoms,
        ),
        dtype=float,
    )


def _unit_fragment_groups(
    definition: "GICDefinition",
) -> dict[tuple[str, tuple[int, ...], tuple[int, ...]], dict[int, tuple[int, object]]]:
    """Collect unit, one-primitive fragment-coordinate triplets."""

    primitive_by_id = {item.identifier: item for item in definition.primitives}
    groups: dict[tuple[str, tuple[int, ...], tuple[int, ...]], dict[int, tuple[int, object]]] = {}
    for index, gic in enumerate(definition.gics):
        coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
        if len(coefficients) != 1 or not np.isclose(float(coefficients[0][1]), 1.0):
            continue
        primitive = primitive_by_id.get(coefficients[0][0])
        if primitive is None or primitive.function not in {"FTRANS", "FROT"}:
            continue
        key = (primitive.function, tuple(primitive.atoms), tuple(primitive.ref_atoms))
        groups.setdefault(key, {})[int(primitive.mode)] = (index, primitive)
    return groups


def _radial_transverse_groups(
    definition: "GICDefinition",
) -> dict[tuple[tuple[int, ...], tuple[int, ...]], tuple[tuple[int, object], ...]]:
    """Collect exact-rank ``FC_DIST + 2 FTRANS`` relative-translation charts."""

    primitive_by_id = {item.identifier: item for item in definition.primitives}
    candidates: dict[
        tuple[tuple[int, ...], tuple[int, ...]], list[tuple[int, object]]
    ] = {}
    for index, gic in enumerate(definition.gics):
        coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
        if len(coefficients) != 1 or not np.isclose(float(coefficients[0][1]), 1.0):
            continue
        primitive = primitive_by_id.get(coefficients[0][0])
        if primitive is None or primitive.function not in {"FC_DIST", "FTRANS"}:
            continue
        key = (tuple(primitive.atoms), tuple(primitive.ref_atoms))
        candidates.setdefault(key, []).append((index, primitive))
    result = {}
    for key, members in candidates.items():
        functions = [getattr(primitive, "function") for _index, primitive in members]
        translation_modes = {
            int(getattr(primitive, "mode"))
            for _index, primitive in members
            if getattr(primitive, "function") == "FTRANS"
        }
        if functions.count("FC_DIST") == 1 and len(translation_modes) == 2 and len(members) == 3:
            result[key] = tuple(
                sorted(
                    members,
                    key=lambda item: (
                        0 if getattr(item[1], "function") == "FC_DIST" else 1,
                        int(getattr(item[1], "mode")),
                    ),
                )
            )
    return result
