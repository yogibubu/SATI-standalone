"""Frame-consistent QM data for the approved stationary ZAFF reference.

Lengths and their derivatives must use one unit throughout (normally Angstrom
in ARCHITECT). Energies are complete QM totals, including backend dispersion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from matrix_chem.geometry_alignment import kabsch_rotation, rotate_cartesian_derivatives


@dataclass(frozen=True)
class StationaryPoint:
    coordinates: np.ndarray
    energy: float
    gradient: np.ndarray
    hessian: np.ndarray | None
    rotation: np.ndarray


def stationary_qm_point(
    *,
    coordinates: np.ndarray,
    energy: float,
    gradient: np.ndarray,
    hessian: np.ndarray | None,
    target_coordinates: np.ndarray,
    reference_coordinates: np.ndarray,
    reference_energy: float,
    reference_gradient: np.ndarray,
    minimum_energy: float = 0.0,
    geometry_tolerance: float = 2.0e-6,
) -> StationaryPoint:
    """Rotate a QM point to its requested frame, then apply the v2 convention.

    ``target_coordinates`` are the *same displaced geometry*, expressed in the
    frame of the reference. They are not the equilibrium geometry. Atom order
    must already be canonical. A mismatched geometry is rejected rather than
    assigning another point's derivatives to it. Return the aligned, full
    precision source geometry, not the potentially rounded target coordinates.

    With ``minimum_energy=0`` the energy zero is arbitrary and declared by the
    caller. No Hessian correction is made apart from rotation and symmetry.
    """
    xyz = np.asarray(coordinates, dtype=float)
    target = np.asarray(target_coordinates, dtype=float)
    reference = np.asarray(reference_coordinates, dtype=float)
    g = np.asarray(gradient, dtype=float).reshape(-1)
    gref = np.asarray(reference_gradient, dtype=float).reshape(-1)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0
            or target.shape != xyz.shape or reference.shape != xyz.shape
            or g.shape != (xyz.size,) or gref.shape != g.shape):
        raise ValueError("stationary QM coordinates/derivatives have incompatible dimensions")
    if not all(np.all(np.isfinite(a)) for a in (xyz, target, reference, g, gref)):
        raise ValueError("stationary QM data must be finite")
    if (not all(np.isfinite(v) for v in (energy, reference_energy, minimum_energy,
                                        geometry_tolerance)) or geometry_tolerance <= 0):
        raise ValueError("stationary QM energies and geometry tolerance are invalid")
    hh = None if hessian is None else np.asarray(hessian, dtype=float)
    if hh is not None:
        if hh.shape != (xyz.size, xyz.size) or not np.all(np.isfinite(hh)):
            raise ValueError("stationary QM Hessian is invalid")
        hh = (hh + hh.T) * 0.5
    rotation = kabsch_rotation(xyz, target)
    aligned = (xyz - xyz.mean(axis=0)) @ rotation + target.mean(axis=0)
    if np.max(np.abs(aligned - target)) > geometry_tolerance:
        raise ValueError("QM derivatives and requested geometry refer to different points")
    rotated_g, rotated_h = rotate_cartesian_derivatives(g, rotation, hh)
    corrected_energy = (float(energy) - float(reference_energy) + float(minimum_energy)
                        - float(gref @ (aligned - reference).reshape(-1)))
    return StationaryPoint(
        aligned, corrected_energy, rotated_g - gref, rotated_h, rotation,
    )
