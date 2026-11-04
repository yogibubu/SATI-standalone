"""Rigid realization of single- and multi-contact docking proposals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from matrix_chem import kabsch_rotation, rotation_vector_from_matrix


@dataclass(frozen=True)
class DockingContact:
    moving_atom: int
    target_position_angstrom: tuple[float, float, float]
    complementarity: float = 1.0
    fixed_accessibility: float = 1.0
    moving_accessibility: float = 1.0


@dataclass(frozen=True)
class RigidDockingPose:
    coordinates_angstrom: np.ndarray
    translation_angstrom: np.ndarray
    rotation_vector: np.ndarray
    contact_rmsd_angstrom: float
    guidance_score: float
    contact_count: int


def two_contact_closure_directions(
    fixed_acceptor: Sequence[float],
    moving_acceptor_target: Sequence[float],
    *,
    hydrogen_bond_length_angstrom: float,
    moving_site_separation_angstrom: float,
    accessible_directions: Sequence[Sequence[float]] = (),
    azimuth_samples: int = 24,
) -> np.ndarray:
    """Sample the accessible sphere-intersection circle for two contacts.

    The second moving site must lie both on the contact sphere around the
    fixed acceptor and on the rigid intrafragment sphere around the first
    moving-site target. Empty output means that the two requested contacts
    cannot be closed by the rigid fragment.
    """

    origin = np.asarray(fixed_acceptor, dtype=float)
    target = np.asarray(moving_acceptor_target, dtype=float)
    directions = np.asarray(accessible_directions, dtype=float)
    if origin.shape != (3,) or target.shape != (3,):
        raise ValueError("two-contact closure centers must be 3-vectors")
    radius = float(hydrogen_bond_length_angstrom)
    separation = float(moving_site_separation_angstrom)
    displacement = target - origin
    distance = float(np.linalg.norm(displacement))
    if (
        radius <= 0.0
        or separation <= 0.0
        or distance <= 1.0e-12
        or distance > radius + separation
        or distance < abs(radius - separation)
    ):
        return np.empty((0, 3), dtype=float)
    axis = displacement / distance
    cosine = float(
        np.clip(
            (radius * radius + distance * distance - separation * separation)
            / (2.0 * radius * distance),
            -1.0,
            1.0,
        )
    )
    sine = float(np.sqrt(max(0.0, 1.0 - cosine * cosine)))
    trial = np.asarray((1.0, 0.0, 0.0))
    if abs(float(axis[0])) > 0.8:
        trial = np.asarray((0.0, 1.0, 0.0))
    first = np.cross(axis, trial)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    angles = np.linspace(0.0, 2.0 * np.pi, max(int(azimuth_samples), 1), endpoint=False)
    circle = np.asarray(
        [
            cosine * axis
            + sine * (np.cos(angle) * first + np.sin(angle) * second)
            for angle in angles
        ]
    )
    if directions.size == 0:
        return circle
    if directions.ndim != 2 or directions.shape[1:] != (3,):
        raise ValueError("accessible directions must have shape (n, 3)")
    norms = np.linalg.norm(directions, axis=1)
    if np.any(norms <= 1.0e-12):
        raise ValueError("accessible directions must be nonzero")
    unit_directions = directions / norms[:, None]
    accessible = np.max(circle @ unit_directions.T, axis=1) >= np.cos(
        np.deg2rad(35.0)
    )
    return circle[accessible]


def fit_rigid_docking_contacts(
    moving_coordinates_angstrom: Sequence[Sequence[float]],
    contacts: Sequence[DockingContact],
) -> RigidDockingPose:
    """Fit one rigid partner to one or more complementary contact targets."""

    moving = np.asarray(moving_coordinates_angstrom, dtype=float)
    if moving.ndim != 2 or moving.shape[1:] != (3,) or not np.all(np.isfinite(moving)):
        raise ValueError("docking needs finite moving Cartesian coordinates")
    if not contacts:
        raise ValueError("docking needs at least one contact")
    atoms = np.asarray([int(contact.moving_atom) for contact in contacts], dtype=int)
    if np.any(atoms < 0) or np.any(atoms >= len(moving)):
        raise ValueError("docking contact moving atom is outside the fragment")
    targets = np.asarray(
        [contact.target_position_angstrom for contact in contacts], dtype=float
    )
    if targets.shape != (len(contacts), 3) or not np.all(np.isfinite(targets)):
        raise ValueError("docking contact targets must be finite 3-vectors")
    components = np.asarray(
        [
            (
                contact.complementarity,
                contact.fixed_accessibility,
                contact.moving_accessibility,
            )
            for contact in contacts
        ],
        dtype=float,
    )
    if np.any(~np.isfinite(components)) or np.any(components < 0.0) or np.any(
        components > 1.0
    ):
        raise ValueError("docking complementarity/accessibility must lie in [0, 1]")
    source = moving[atoms]
    rotation = kabsch_rotation(source, targets)
    translation = np.mean(targets, axis=0) - np.mean(source, axis=0) @ rotation
    realized = moving @ rotation + translation
    residual = realized[atoms] - targets
    rmsd = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    weakest = float(np.min(np.prod(components, axis=1)))
    # Geometry quality is a prior for initialization only.  It is deliberately
    # not an energy and never enters the subsequent Pareto objectives.
    guidance = weakest * float(np.exp(-0.5 * (rmsd / 0.35) ** 2))
    return RigidDockingPose(
        coordinates_angstrom=realized,
        translation_angstrom=translation,
        rotation_vector=rotation_vector_from_matrix(rotation),
        contact_rmsd_angstrom=rmsd,
        guidance_score=guidance,
        contact_count=len(contacts),
    )
