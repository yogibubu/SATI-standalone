"""Symmetry labels and avoided-crossing diagnostics for DVR levels."""

from __future__ import annotations

import numpy as np


def detect_avoided_crossings(coordinates: object, levels: object, *, gap_threshold_cm: float = 1.0) -> list[dict[str, float | int]]:
    q = np.asarray(coordinates, dtype=float)
    values = np.asarray(levels, dtype=float)
    if values.ndim != 2 or values.shape[0] != q.size:
        raise ValueError("levels must have shape (coordinate_points, states)")
    crossings: list[dict[str, float | int]] = []
    for state in range(values.shape[1] - 1):
        gap = np.abs(values[:, state + 1] - values[:, state])
        index = int(np.argmin(gap))
        if float(gap[index]) <= gap_threshold_cm and 0 < index < len(q) - 1:
            crossings.append({"state_lower": state, "state_upper": state + 1, "coordinate": float(q[index]), "gap_cm": float(gap[index])})
    return crossings


def symmetry_parity(labels: object, vectors: object, *, tolerance: float = 0.1) -> list[str]:
    """Assign even/odd parity from overlap with a reversed grid."""
    values = np.asarray(vectors, dtype=float)
    if values.ndim != 2 or values.shape[0] != len(labels):
        raise ValueError("vectors and labels have incompatible shapes")
    output = []
    for column in values.T:
        overlap = float(np.dot(column, column[::-1]))
        output.append("even" if overlap > tolerance else "odd" if overlap < -tolerance else "mixed")
    return output


__all__ = ["detect_avoided_crossings", "symmetry_parity"]
