"""Numerical domain checks shared by LINK chart lifecycle and realization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matrix_smith.models import GICDefinition


ORDINARY_ANGLE_MINIMUM_SINE = 0.12


def near_linear_ordinary_angle_ids(
    definition: "GICDefinition",
    coordinates_angstrom: np.ndarray,
    *,
    minimum_sine: float = ORDINARY_ANGLE_MINIMUM_SINE,
) -> tuple[str, ...]:
    """Return ordinary-angle primitive IDs outside their stable domain."""

    threshold = float(minimum_sine)
    if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("ordinary-angle minimum sine must lie in (0, 1)")
    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1:] != (3,):
        raise ValueError("ordinary-angle domain coordinates must have shape natoms x 3")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("ordinary-angle domain coordinates contain non-finite values")
    invalid: list[str] = []
    for primitive in definition.primitives:
        if primitive.function != "A" or len(primitive.atoms) != 3:
            continue
        first, center, second = (atom - 1 for atom in primitive.atoms)
        left = coordinates[first] - coordinates[center]
        right = coordinates[second] - coordinates[center]
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator <= 1.0e-14:
            invalid.append(primitive.identifier)
            continue
        cosine = float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
        sine = float(np.sqrt(max(1.0 - cosine * cosine, 0.0)))
        if sine < threshold:
            invalid.append(primitive.identifier)
    return tuple(invalid)
