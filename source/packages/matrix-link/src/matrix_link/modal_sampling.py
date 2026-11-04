from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModalDisplacementBatch:
    """Geometry-only +/-Q batch realized by LINK for an external evaluator."""

    reference_bohr: np.ndarray
    plus_bohr: np.ndarray
    minus_bohr: np.ndarray
    steps_sqrt_amu_bohr: np.ndarray


def prepare_modal_gradient_batch(
    coordinates_bohr,
    modes_mw,
    masses_amu,
    steps_sqrt_amu_bohr,
) -> ModalDisplacementBatch:
    """Realize independent +/- normal-mode points without owning the stencil.

    TRINITY supplies modes and steps. LINK only converts those requested points
    into Cartesian geometries suitable for its existing parallel/restartable
    external-evaluation service.
    """

    coordinates = np.asarray(coordinates_bohr, dtype=float)
    modes = np.asarray(modes_mw, dtype=float)
    masses = np.asarray(masses_amu, dtype=float).reshape(-1)
    steps = np.asarray(steps_sqrt_amu_bohr, dtype=float).reshape(-1)
    if coordinates.shape != (masses.size, 3):
        raise ValueError("coordinates must have shape natoms x 3")
    if modes.shape != (steps.size, masses.size, 3):
        raise ValueError("modes and step vector have incompatible dimensions")
    if np.any(steps <= 0.0) or not np.all(np.isfinite(steps)):
        raise ValueError("normal-mode finite-difference steps must be positive and finite")
    displacement = modes / np.sqrt(masses)[None, :, None]
    plus = coordinates[None, :, :] + steps[:, None, None] * displacement
    minus = coordinates[None, :, :] - steps[:, None, None] * displacement
    return ModalDisplacementBatch(coordinates, plus, minus, steps)


__all__ = ["ModalDisplacementBatch", "prepare_modal_gradient_batch"]
