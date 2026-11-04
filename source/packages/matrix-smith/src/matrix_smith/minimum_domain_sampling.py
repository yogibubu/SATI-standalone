"""Geometry stencils prescribed by ORACLE for stable MINIMUM contacts."""

from __future__ import annotations

import numpy as np

from matrix_chem.coordinate_atlas_contract import AtlasLocalDomainPrescription

from .contracts import GICForgeContractError
from .policy import RANK_TOLERANCE


def local_domain_samples(
    reference_coordinates: object,
    domains: tuple[AtlasLocalDomainPrescription, ...],
) -> tuple[tuple[str, np.ndarray], ...]:
    """Materialize the immutable local domains owned by ORACLE."""

    reference = np.asarray(reference_coordinates, dtype=float)
    samples: list[tuple[str, np.ndarray]] = [("REFERENCE", reference)]
    for domain in domains:
        samples.extend(_domain_samples(reference, domain))
    return tuple(samples)


def _domain_samples(
    reference: np.ndarray,
    domain: AtlasLocalDomainPrescription,
) -> tuple[tuple[str, np.ndarray], ...]:
    moving = tuple(atom - 1 for atom in domain.moving_atoms)
    if max(moving, default=-1) >= len(reference):
        raise GICForgeContractError(
            f"ORACLE local domain {domain.domain_id} exceeds the atom count"
        )
    anchor = reference[domain.moving_endpoint_atom - 1]
    axis = anchor - reference[domain.reference_endpoint_atom - 1]
    axis_norm = float(np.linalg.norm(axis))
    if not np.isfinite(axis_norm) or axis_norm <= RANK_TOLERANCE:
        raise GICForgeContractError(
            f"ORACLE local domain {domain.domain_id} has a singular contact axis"
        )
    axis /= axis_norm
    transverse = _transverse_axes(axis)
    samples = _radial_samples(reference, domain, moving=moving, axis=axis)
    step_count = int(
        round(domain.axial_half_width_degrees / domain.axial_step_degrees)
    )
    for step in range(-step_count, step_count + 1):
        angle = float(step) * domain.axial_step_degrees
        axial = _rotate_fragment(
            reference,
            moving,
            origin=anchor,
            axis=axis,
            degrees=angle,
        )
        if step:
            samples.append((f"{domain.domain_id}:AXIAL:{angle:+.12g}", axial))
        samples.extend(
            _tilt_samples(
                axial,
                domain,
                moving=moving,
                origin=anchor,
                axes=transverse,
                angle=angle,
            )
        )
    return tuple(samples)


def _radial_samples(
    reference: np.ndarray,
    domain: AtlasLocalDomainPrescription,
    *,
    moving: tuple[int, ...],
    axis: np.ndarray,
) -> list[tuple[str, np.ndarray]]:
    samples: list[tuple[str, np.ndarray]] = []
    for sign in (-1.0, 1.0):
        displaced = np.array(reference, copy=True)
        displaced[np.asarray(moving, dtype=int)] += (
            sign * domain.radial_step_angstrom * axis
        )
        samples.append((f"{domain.domain_id}:RADIAL:{sign:+.0f}", displaced))
    return samples


def _tilt_samples(
    axial: np.ndarray,
    domain: AtlasLocalDomainPrescription,
    *,
    moving: tuple[int, ...],
    origin: np.ndarray,
    axes: tuple[np.ndarray, np.ndarray],
    angle: float,
) -> tuple[tuple[str, np.ndarray], ...]:
    samples: list[tuple[str, np.ndarray]] = []
    for axis_index, axis in enumerate(axes, start=1):
        for tilt in (-domain.tilt_degrees, domain.tilt_degrees):
            geometry = _rotate_fragment(
                axial,
                moving,
                origin=origin,
                axis=axis,
                degrees=tilt,
            )
            label = (
                f"{domain.domain_id}:AXIAL:{angle:+.12g}:"
                f"TILT{axis_index}:{tilt:+.12g}"
            )
            samples.append((label, geometry))
    return tuple(samples)


def _transverse_axes(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cartesian = np.eye(3, dtype=float)
    seed = cartesian[int(np.argmin(np.abs(cartesian @ axis)))]
    first = np.cross(axis, seed)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    second /= np.linalg.norm(second)
    return first, second


def _rotate_fragment(
    coordinates: np.ndarray,
    atoms: tuple[int, ...],
    *,
    origin: np.ndarray,
    axis: np.ndarray,
    degrees: float,
) -> np.ndarray:
    angle = float(np.deg2rad(degrees))
    cosine, sine = float(np.cos(angle)), float(np.sin(angle))
    output = np.array(coordinates, copy=True)
    for atom in atoms:
        vector = output[atom] - origin
        rotated = (
            vector * cosine
            + np.cross(axis, vector) * sine
            + axis * np.dot(axis, vector) * (1.0 - cosine)
        )
        output[atom] = origin + rotated
    return output


__all__ = ["local_domain_samples"]
