from __future__ import annotations

import numpy as np


def is_linear_geometry(coordinates_angstrom, *, tolerance: float = 1.0e-7) -> bool:
    """Return whether a multi-atom Cartesian geometry is linear."""

    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates must have shape natoms x 3")
    natoms = coordinates.shape[0]
    if natoms < 2:
        return False
    centered = coordinates - np.mean(coordinates, axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    scale = max(float(singular_values[0]), 1.0)
    return bool(float(singular_values[1]) <= float(tolerance) * scale)


def expected_vibrational_mode_count(coordinates_angstrom) -> int:
    """Return 3N-5 for a linear molecule, 3N-6 otherwise, and zero for an atom."""

    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates must have shape natoms x 3")
    natoms = int(coordinates.shape[0])
    if natoms <= 1:
        return 0
    return 3 * natoms - (5 if is_linear_geometry(coordinates) else 6)
