"""Deterministic sphere-to-ellipsoid initialization around a fixed solute."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Sequence

import numpy as np

from matrix_chem.topology.elements import atomic_number
from matrix_chem.topology.vdw_radii import descriptor_vdw_radius, uff_vdw_radius


SOLVATED_ELLIPSOID_SCHEMA = "matrix.architect.solvated_ellipsoid.v1"


@dataclass(frozen=True)
class SolvatedEllipsoidAssembly:
    solute_elements: tuple[str, ...]
    solute_coordinates_angstrom: np.ndarray
    solvent_elements: tuple[str, ...]
    solvent_coordinates_angstrom: np.ndarray
    semiaxes_angstrom: np.ndarray
    parent_sphere_radius_angstrom: float
    solvent_vdw_diameter_angstrom: float
    shell_count: int
    end_clearance_angstrom: float
    parent_candidate_count: int
    retained_solvent_count: int
    canonical_rotation: np.ndarray
    canonical_origin_angstrom: np.ndarray
    sha256: str
    schema: str = SOLVATED_ELLIPSOID_SCHEMA


def _radius(symbol: str) -> float:
    number = int(atomic_number(str(symbol)) or 0)
    radius = descriptor_vdw_radius(number)
    if radius is None:
        radius = uff_vdw_radius(number)
    if radius is None or not np.isfinite(radius) or radius <= 0.0:
        raise ValueError(f"no van der Waals radius is available for {symbol}")
    return float(radius)


def _principal_frame(coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    origin = np.mean(coordinates, axis=0)
    centered = coordinates - origin
    values, axes = np.linalg.eigh(centered.T @ centered)
    order = np.argsort(values)[::-1]
    values = values[order]
    axes = axes[:, order]
    scale = max(float(np.max(np.abs(values))), 1.0)
    if np.any(np.abs(np.diff(values)) <= 1.0e-12 * scale):
        axes = np.eye(3)
    local = centered @ axes
    for column in range(3):
        pivot = int(np.argmax(np.abs(local[:, column])))
        if local[pivot, column] < 0.0:
            axes[:, column] *= -1.0
            local[:, column] *= -1.0
    if np.linalg.det(axes) < 0.0:
        axes[:, -1] *= -1.0
        local[:, -1] *= -1.0
    return local, axes, origin


def _haar_rotation(rng: np.random.Generator) -> np.ndarray:
    quaternion = rng.normal(size=4)
    quaternion /= np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    return np.asarray(
        (
            (1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)),
            (2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)),
            (2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)),
        )
    )


def build_solvated_ellipsoid(
    solute_elements: Sequence[str],
    solute_coordinates_angstrom: np.ndarray,
    solvent_elements: Sequence[str],
    solvent_coordinates_angstrom: np.ndarray,
    *,
    solvent_count: int,
    shell_count: int,
    end_clearance_angstrom: float = 1.0,
    boundary_scale: float = 0.999,
    exclusion_scale: float = 0.92,
    seed: int = 61453,
) -> SolvatedEllipsoidAssembly:
    """Build a spherical bath, carve its centre, deform it and clip it.

    The solute geometric centre is the origin and its principal axes are the
    laboratory axes.  Only solvent molecular centres undergo the affine
    sphere-to-ellipsoid deformation; each solvent remains a rigid molecule.
    """

    solute_symbols = tuple(str(value) for value in solute_elements)
    solvent_symbols = tuple(str(value) for value in solvent_elements)
    solute = np.asarray(solute_coordinates_angstrom, dtype=float)
    solvent = np.asarray(solvent_coordinates_angstrom, dtype=float)
    count = int(solvent_count)
    shells = int(shell_count)
    clearance = float(end_clearance_angstrom)
    if (
        not solute_symbols
        or solute.shape != (len(solute_symbols), 3)
        or not solvent_symbols
        or solvent.shape != (len(solvent_symbols), 3)
        or np.any(~np.isfinite(solute))
        or np.any(~np.isfinite(solvent))
        or count < 1
        or shells < 1
        or not np.isfinite(clearance)
        or clearance <= 0.0
        or not 0.0 < float(boundary_scale) < 1.0
        or not 0.0 < float(exclusion_scale) <= 1.0
    ):
        raise ValueError("solute, solvent or solvated-ellipsoid controls are invalid")

    canonical, rotation, origin = _principal_frame(solute)
    solute_radii = np.asarray([_radius(symbol) for symbol in solute_symbols])
    lower = np.min(canonical - solute_radii[:, None], axis=0)
    upper = np.max(canonical + solute_radii[:, None], axis=0)

    solvent_centered = solvent - np.mean(solvent, axis=0)
    solvent_radii = np.asarray([_radius(symbol) for symbol in solvent_symbols])
    solvent_envelope = float(
        np.max(np.linalg.norm(solvent_centered, axis=1) + solvent_radii)
    )
    solvent_diameter = 2.0 * solvent_envelope
    margin = shells * solvent_diameter + clearance
    semiaxes = np.maximum(np.abs(lower), np.abs(upper)) + margin
    sphere_radius = float(np.max(semiaxes))

    spacing = solvent_diameter + 0.25
    extent = int(math.ceil(sphere_radius / spacing)) + 1
    parent_centers = []
    for i in range(-extent, extent + 1):
        for j in range(-extent, extent + 1):
            for k in range(-extent, extent + 1):
                center = spacing * np.asarray((i, j, k), dtype=float)
                if np.linalg.norm(center) + solvent_envelope < sphere_radius:
                    parent_centers.append(center)
    parent_centers.sort(key=lambda value: (float(np.dot(value, value)), *value.tolist()))
    rng = np.random.default_rng(int(seed))
    retained: list[np.ndarray] = []
    for parent_center in parent_centers:
        orientation = _haar_rotation(rng)
        parent_molecule = solvent_centered @ orientation.T + parent_center
        parent_distance = np.linalg.norm(
            parent_molecule[:, None, :] - canonical[None, :, :], axis=2
        )
        exclusion = exclusion_scale * (
            solvent_radii[:, None] + solute_radii[None, :]
        )
        if np.any(parent_distance < exclusion):
            continue
        center = parent_center * (semiaxes / sphere_radius)
        molecule = solvent_centered @ orientation.T + center
        if np.max(np.linalg.norm(molecule / semiaxes, axis=1)) >= boundary_scale:
            continue
        distance = np.linalg.norm(
            molecule[:, None, :] - canonical[None, :, :], axis=2
        )
        # Compression along a short ellipsoid axis can reduce a separation
        # that was safe in the parent sphere, so retain a final safety gate.
        if np.any(distance < exclusion):
            continue
        retained.append(molecule)
        if len(retained) == count:
            break
    if len(retained) != count:
        raise ValueError(
            "requested solvent count does not fit the sphere-to-ellipsoid shell: "
            f"requested={count} retained={len(retained)}"
        )
    solvent_instances = np.asarray(retained, dtype=float)
    payload = {
        "schema": SOLVATED_ELLIPSOID_SCHEMA,
        "solute": canonical.tolist(),
        "solvent": solvent_instances.tolist(),
        "semiaxes": semiaxes.tolist(),
        "shell_count": shells,
        "solvent_vdw_diameter_angstrom": solvent_diameter,
        "end_clearance_angstrom": clearance,
        "seed": int(seed),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return SolvatedEllipsoidAssembly(
        solute_elements=solute_symbols,
        solute_coordinates_angstrom=canonical,
        solvent_elements=solvent_symbols,
        solvent_coordinates_angstrom=solvent_instances,
        semiaxes_angstrom=semiaxes,
        parent_sphere_radius_angstrom=sphere_radius,
        solvent_vdw_diameter_angstrom=solvent_diameter,
        shell_count=shells,
        end_clearance_angstrom=clearance,
        parent_candidate_count=len(parent_centers),
        retained_solvent_count=len(retained),
        canonical_rotation=rotation,
        canonical_origin_angstrom=origin,
        sha256=digest,
    )


__all__ = [
    "SOLVATED_ELLIPSOID_SCHEMA",
    "SolvatedEllipsoidAssembly",
    "build_solvated_ellipsoid",
]
