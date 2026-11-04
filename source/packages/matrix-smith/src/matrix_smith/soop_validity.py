"""Geometric validity certificate for frozen-reference SOOP charts."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


SOOP_GEOMETRIC_VALIDITY_SCHEMA = "matrix.smith.soop_geometric_validity.v1"


@dataclass(frozen=True)
class SOOPGeometricValidity:
    status: str
    maximum_normal_rotation_degree: float
    minimum_plane_sine: float
    warning_angle_degree: float
    failure_angle_degree: float
    minimum_allowed_plane_sine: float
    warning_primitive_ids: tuple[str, ...]
    invalid_primitive_ids: tuple[str, ...]
    fragment_ids: tuple[str, ...]
    atom_indices_zero_based: tuple[int, ...]
    schema: str = SOOP_GEOMETRIC_VALIDITY_SCHEMA

    @property
    def valid(self) -> bool:
        return self.status != "INVALID"

    def scientific_validation_record(self):
        from matrix_core import ScientificValidationRecord

        return ScientificValidationRecord(
            owner="SMITH",
            subject="SOOP_GEOMETRIC_VALIDITY",
            passed=self.valid,
            metrics={
                "maximum_normal_rotation_degree": self.maximum_normal_rotation_degree,
                "minimum_plane_sine": self.minimum_plane_sine,
            },
            thresholds={
                "failure_angle_degree": self.failure_angle_degree,
                "minimum_allowed_plane_sine": self.minimum_allowed_plane_sine,
            },
            context={
                "status": self.status,
                "warning_angle_degree": self.warning_angle_degree,
                "affected_fragment_count": len(self.fragment_ids),
                "affected_atom_count": len(self.atom_indices_zero_based),
            },
        )


def audit_soop_geometric_validity(
    contract,
    coordinates_angstrom: np.ndarray,
    *,
    warning_angle_degree: float = 15.0,
    failure_angle_degree: float = 30.0,
    minimum_plane_sine: float = 0.05,
) -> SOOPGeometricValidity:
    """Compare current local face normals with ORACLE's frozen SOOP frames.

    This is a semantic chart-radius test, not a derivative test: a frozen SOOP
    remains exactly affine at any geometry, but ceases to represent the same
    local out-of-plane motion after the supporting face rotates too far.
    """

    warning = float(warning_angle_degree)
    failure = float(failure_angle_degree)
    sine_limit = float(minimum_plane_sine)
    if not 0.0 < warning < failure < 180.0:
        raise ValueError("SOOP validity angles must satisfy 0 < warning < failure < 180")
    if not 0.0 < sine_limit < 1.0:
        raise ValueError("SOOP validity minimum plane sine must lie in (0, 1)")
    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != 3
        or not np.all(np.isfinite(coordinates))
    ):
        raise ValueError("SOOP validity coordinates must be finite with shape (natoms, 3)")

    records = tuple(contract.selective_out_of_plane_primitives)
    warning_ids: list[str] = []
    invalid_ids: list[str] = []
    affected_atoms: set[int] = set()
    rotations: list[float] = []
    plane_sines: list[float] = []
    for record in records:
        center, first, second = (int(atom) - 1 for atom in record.plane_atoms)
        if min(center, first, second) < 0 or max(center, first, second) >= len(coordinates):
            raise ValueError(f"SOOP primitive {record.identifier} lies outside the geometry")
        left = coordinates[first] - coordinates[center]
        right = coordinates[second] - coordinates[center]
        left_norm = float(np.linalg.norm(left))
        right_norm = float(np.linalg.norm(right))
        cross = np.cross(left, right)
        cross_norm = float(np.linalg.norm(cross))
        plane_sine = (
            cross_norm / (left_norm * right_norm)
            if left_norm > 1.0e-14 and right_norm > 1.0e-14
            else 0.0
        )
        plane_sines.append(plane_sine)
        if cross_norm > 1.0e-14:
            current_normal = cross / cross_norm
            reference_normal = np.asarray(record.normal, dtype=float)
            cosine = float(np.clip(np.dot(current_normal, reference_normal), -1.0, 1.0))
            angle = math.degrees(math.acos(cosine))
        else:
            angle = 180.0
        rotations.append(angle)
        invalid = (
            str(record.status) != "REGULAR"
            or plane_sine < sine_limit
            or angle >= failure
        )
        warned = invalid or angle >= warning
        if warned:
            warning_ids.append(str(record.identifier))
            affected_atoms.update(
                int(atom)
                for atom in (
                    record.center_atom,
                    record.moving_atom,
                    *record.plane_atoms,
                )
            )
        if invalid:
            invalid_ids.append(str(record.identifier))

    fragments = tuple(
        fragment
        for fragment in contract.primary_topology.fragments
        if affected_atoms.intersection(int(atom) for atom in fragment.atoms)
    )
    fragment_ids = tuple(str(fragment.fragment_id) for fragment in fragments)
    fragment_atoms = tuple(
        sorted({int(atom) - 1 for fragment in fragments for atom in fragment.atoms})
    )
    status = "INVALID" if invalid_ids else ("WARN" if warning_ids else "VALID")
    return SOOPGeometricValidity(
        status=status,
        maximum_normal_rotation_degree=max(rotations, default=0.0),
        minimum_plane_sine=min(plane_sines, default=1.0),
        warning_angle_degree=warning,
        failure_angle_degree=failure,
        minimum_allowed_plane_sine=sine_limit,
        warning_primitive_ids=tuple(warning_ids),
        invalid_primitive_ids=tuple(invalid_ids),
        fragment_ids=fragment_ids,
        atom_indices_zero_based=fragment_atoms,
    )


__all__ = [
    "SOOP_GEOMETRIC_VALIDITY_SCHEMA",
    "SOOPGeometricValidity",
    "audit_soop_geometric_validity",
]
