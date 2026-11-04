"""Exact rigid-equivalence checks and AO rotations for fragment reuse."""

from __future__ import annotations

import numpy as np

from .symmetry import _angular_representation


def rigid_equivalence_rotation(source: object, target: object, *, tolerance: float = 1.0e-6) -> np.ndarray | None:
    """Return source-to-target proper rotation only for equivalent QM+MM frames."""
    if (
        source.atomic_numbers != target.atomic_numbers
        or len(source.embedding_atoms) != len(target.embedding_atoms)
        or len(source.link_bonds) != len(target.link_bonds)
    ):
        return None
    start = np.asarray(source.coordinates_angstrom, dtype=float)
    goal = np.asarray(target.coordinates_angstrom, dtype=float)
    if start.shape != goal.shape or len(start) < 3:
        return None
    centered_start = start - start.mean(axis=0)
    centered_goal = goal - goal.mean(axis=0)
    left, _values, right = np.linalg.svd(centered_start.T @ centered_goal)
    handedness = np.linalg.det(left @ right)
    rotation = left @ np.diag((1.0, 1.0, 1.0 if handedness > 0.0 else -1.0)) @ right
    translation = goal.mean(axis=0) - start.mean(axis=0) @ rotation
    if np.max(np.linalg.norm(start @ rotation + translation - goal, axis=1)) > tolerance:
        return None
    source_points = np.asarray(source.embedding_coordinates_angstrom, dtype=float)
    target_points = np.asarray(target.embedding_coordinates_angstrom, dtype=float)
    source_charges = np.asarray(source.embedding_charges_electron, dtype=float)
    target_charges = np.asarray(target.embedding_charges_electron, dtype=float)
    unmatched = set(range(len(target_points)))
    for point, charge in zip(source_points @ rotation + translation, source_charges, strict=True):
        options = [
            index for index in unmatched
            if abs(charge - target_charges[index]) <= 1.0e-8
            and np.linalg.norm(point - target_points[index]) <= tolerance
        ]
        if not options:
            return None
        unmatched.remove(options[0])
    return rotation


def rotate_fragment_aos(rotation: np.ndarray, orbital_atom: np.ndarray, orbital_shell: tuple[str, ...]) -> np.ndarray:
    """Build the real s/p/d AO coefficient rotation for a rigid fragment."""
    atom = np.asarray(orbital_atom, dtype=np.intp)
    if len(atom) != len(orbital_shell):
        raise ValueError("fragment AO labels are inconsistent")
    result = np.zeros((len(atom), len(atom)))
    groups: dict[tuple[int, str], list[int]] = {}
    for index, key in enumerate(zip(atom, orbital_shell, strict=True)):
        groups.setdefault((int(key[0]), key[1]), []).append(index)
    for indices in groups.values():
        angular = {1: 0, 3: 1, 5: 2}.get(len(indices))
        if angular is None:
            raise ValueError("unsupported fragment AO angular block")
        result[np.ix_(indices, indices)] = _angular_representation(rotation, angular).T
    return result


def fragment_shape_key(frame: object) -> tuple[object, ...]:
    """Bounded cheap bucket key; exact rigid and MM checks remain mandatory."""
    positions = np.asarray(frame.coordinates_angstrom, dtype=float)
    squared = np.sum((positions[:, None, :] - positions[None, :, :]) ** 2, axis=2)
    upper = squared[np.triu_indices(len(positions), 1)]
    return (
        frame.atomic_numbers,
        tuple(np.rint(upper * 1.0e6).astype(np.int64)),
        tuple(sorted(np.rint(np.asarray(frame.embedding_charges_electron) * 1.0e8).astype(np.int64))),
    )
