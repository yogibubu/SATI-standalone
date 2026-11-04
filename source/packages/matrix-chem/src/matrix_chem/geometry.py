from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def rigid_geometry_max_error(
    reference_coordinates_angstrom: np.ndarray,
    candidate_coordinates_angstrom: np.ndarray,
) -> float:
    """Return the largest intramolecular distance error for rigid copies.

    ``candidate_coordinates_angstrom`` may contain one molecule with shape
    ``(natoms, 3)`` or any leading batch dimensions ending in
    ``(natoms, 3)``.  Comparing complete distance matrices makes the contract
    independent of translation, rotation, atom types, and solvent identity.
    """
    reference = np.asarray(reference_coordinates_angstrom, dtype=float)
    candidates = np.asarray(candidate_coordinates_angstrom, dtype=float)
    if reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("reference coordinates must have shape (natoms, 3)")
    if candidates.shape[-2:] != reference.shape:
        raise ValueError(
            "candidate coordinates must end with the reference shape "
            f"{reference.shape}, got {candidates.shape}"
        )
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidates)):
        raise ValueError("rigid-geometry coordinates must be finite")
    reference_distances = np.linalg.norm(
        reference[:, None, :] - reference[None, :, :],
        axis=-1,
    )
    candidate_distances = np.linalg.norm(
        candidates[..., :, None, :] - candidates[..., None, :, :],
        axis=-1,
    )
    return float(np.max(np.abs(candidate_distances - reference_distances), initial=0.0))


def validate_rigid_geometry(
    reference_coordinates_angstrom: np.ndarray,
    candidate_coordinates_angstrom: np.ndarray,
    *,
    tolerance_angstrom: float = 1.0e-6,
) -> float:
    """Validate rigid molecular copies and return their maximum distance error."""
    if not np.isfinite(tolerance_angstrom) or tolerance_angstrom < 0.0:
        raise ValueError("rigid-geometry tolerance must be finite and non-negative")
    error = rigid_geometry_max_error(
        reference_coordinates_angstrom,
        candidate_coordinates_angstrom,
    )
    if error > tolerance_angstrom:
        raise ValueError(
            "intramolecular rigid-geometry contract violated: "
            f"maximum distance error {error:.6g} angstrom exceeds "
            f"{tolerance_angstrom:.6g} angstrom"
        )
    return error


def isotropic_cartesian_limits(
    coordinates_angstrom: np.ndarray,
    *,
    padding_fraction: float = 0.10,
    minimum_span_angstrom: float = 2.0,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Return equal-span Cartesian limits for distortion-free 3-D rendering."""
    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or not len(coordinates):
        raise ValueError("coordinates must have non-empty shape (natoms, 3)")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("coordinates must be finite")
    if not np.isfinite(padding_fraction) or padding_fraction < 0.0:
        raise ValueError("padding fraction must be finite and non-negative")
    if not np.isfinite(minimum_span_angstrom) or minimum_span_angstrom <= 0.0:
        raise ValueError("minimum span must be finite and positive")
    center = 0.5 * (coordinates.min(axis=0) + coordinates.max(axis=0))
    span = max(
        float(np.ptp(coordinates, axis=0).max()) * (1.0 + padding_fraction),
        float(minimum_span_angstrom),
    )
    half_span = 0.5 * span
    return tuple(
        (float(value - half_span), float(value + half_span)) for value in center
    )


@dataclass(frozen=True)
class MolecularGeometry:
    """Canonical ORACLE Cartesian molecular geometry.

    Coordinates are stored in Angstrom. Atom labels are normalized element
    symbols. Program-specific parser details belong in `metadata`.
    """

    atoms: tuple[str, ...]
    coordinates_angstrom: np.ndarray
    comment: str = ""
    source_format: str = "unknown"
    source_path: Path | None = None
    charge: int | None = None
    multiplicity: int | None = None
    fixed_parameters: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        coords = np.asarray(self.coordinates_angstrom, dtype=float)
        if coords.shape != (len(self.atoms), 3):
            raise ValueError(
                f"coordinates shape must be ({len(self.atoms)}, 3), got {coords.shape}"
            )
        if not np.all(np.isfinite(coords)):
            raise ValueError("coordinates must be finite")
        object.__setattr__(self, "coordinates_angstrom", coords)
        object.__setattr__(self, "atoms", tuple(self.atoms))
        object.__setattr__(self, "fixed_parameters", tuple(self.fixed_parameters))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def natoms(self) -> int:
        return len(self.atoms)

    def xyz_lines(self) -> list[str]:
        lines = [str(self.natoms), self.comment]
        for atom, (x, y, z) in zip(self.atoms, self.coordinates_angstrom):
            lines.append(f"{atom:2s} {x:19.12f} {y:19.12f} {z:19.12f}")
        return lines
