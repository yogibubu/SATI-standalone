"""SMITH construction of production coordinates for large-amplitude tunneling.

The construction is molecule independent.  ORACLE supplies periodicity and
Cartesian symmetry actions; a certified transition-state Hessian supplies the
index and direction of the saddle.  A nonperiodic index-one saddle selects the
rectilinear, mass-scaled ``Q_im`` coordinate.  Curvilinear path length is not
silently substituted for its kinetic coordinate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


SMITH_TUNNELING_COORDINATE_SCHEMA = "matrix.smith.tunneling_coordinate.v1"
QIM_KINETIC_COEFFICIENT_MHZ = 505379.0090099937


@dataclass(frozen=True)
class TunnelingCoordinateContract:
    """Frozen SMITH coordinate consumed by LINK and LOCK."""

    label: str
    points: tuple[float, ...]
    cartesian_direction: tuple[float, ...]
    cartesian_masses: tuple[float, ...]
    imaginary_curvature: float
    imaginary_frequency_cm1: float | None = None
    oracle_point_group: str = "unknown"
    symmetry_characters: tuple[tuple[str, int], ...] = ()
    schema: str = SMITH_TUNNELING_COORDINATE_SCHEMA
    coordinate_kind: str = "rectilinear_qim"
    periodic: bool = False
    kinetic_model: str = "constant_mass_rectilinear"
    vam_treatment: str = "omitted"
    coordinate_unit: str = "angstrom_sqrt_amu"
    cartesian_direction_unit: str = "amu^-1/2"
    kinetic_coefficient_MHz: float = QIM_KINETIC_COEFFICIENT_MHZ

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=float)
        direction = np.asarray(self.cartesian_direction, dtype=float)
        masses = np.asarray(self.cartesian_masses, dtype=float)
        if self.schema != SMITH_TUNNELING_COORDINATE_SCHEMA:
            raise ValueError("unsupported SMITH tunneling-coordinate schema")
        if not self.label.strip() or points.size < 3 or np.any(np.diff(points) <= 0.0):
            raise ValueError("Qim requires a label and at least three increasing points")
        if not np.all(np.isfinite(points)):
            raise ValueError("Qim points must be finite")
        if direction.ndim != 1 or masses.shape != direction.shape:
            raise ValueError("Qim direction and Cartesian masses must have equal dimensions")
        if np.any(masses <= 0.0) or not np.all(np.isfinite(direction)):
            raise ValueError("Qim masses must be positive and its direction finite")
        norm = float(np.dot(direction * masses, direction))
        if not np.isclose(norm, 1.0, rtol=1.0e-8, atol=1.0e-10):
            raise ValueError("Qim direction must have unit mass-weighted norm")
        if not np.isfinite(self.imaginary_curvature) or self.imaginary_curvature >= 0.0:
            raise ValueError("Qim requires one negative mass-weighted curvature")
        if self.imaginary_frequency_cm1 is not None and (
            not np.isfinite(self.imaginary_frequency_cm1) or self.imaginary_frequency_cm1 == 0.0
        ):
            raise ValueError("imaginary frequency must be finite and nonzero when supplied")
        if self.periodic or self.coordinate_kind != "rectilinear_qim":
            raise ValueError("Qim is a nonperiodic rectilinear coordinate")
        if self.kinetic_model != "constant_mass_rectilinear" or self.vam_treatment != "omitted":
            raise ValueError("the production Qim contract requires constant mass and no VAM")
        if (
            self.coordinate_unit != "angstrom_sqrt_amu"
            or self.cartesian_direction_unit != "amu^-1/2"
            or not np.isclose(
                self.kinetic_coefficient_MHz,
                QIM_KINETIC_COEFFICIENT_MHZ,
                rtol=1.0e-12,
                atol=0.0,
            )
        ):
            raise ValueError("Qim units or the mass-normalized kinetic coefficient are invalid")
        if any(character not in (-1, 1) for _, character in self.symmetry_characters):
            raise ValueError("an isolated Qim mode must carry one-dimensional characters")
        labels = tuple(label for label, _ in self.symmetry_characters)
        if len(set(labels)) != len(labels):
            raise ValueError("ORACLE symmetry-operation labels must be unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "label": self.label,
            "points": list(self.points),
            "coordinate_kind": self.coordinate_kind,
            "periodic": self.periodic,
            "cartesian_direction": list(self.cartesian_direction),
            "cartesian_masses": list(self.cartesian_masses),
            "imaginary_curvature": self.imaginary_curvature,
            "imaginary_frequency_cm-1": self.imaginary_frequency_cm1,
            "kinetic_model": self.kinetic_model,
            "vam_treatment": self.vam_treatment,
            "coordinate_unit": self.coordinate_unit,
            "cartesian_direction_unit": self.cartesian_direction_unit,
            "kinetic_coefficient_MHz": self.kinetic_coefficient_MHz,
            "oracle_point_group": self.oracle_point_group,
            "symmetry_characters": dict(self.symmetry_characters),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TunnelingCoordinateContract":
        characters = payload.get("symmetry_characters", {})
        if not isinstance(characters, Mapping):
            raise ValueError("symmetry_characters must be a mapping")
        frequency = payload.get("imaginary_frequency_cm-1")
        return cls(
            label=str(payload["label"]),
            points=tuple(float(value) for value in payload["points"]),  # type: ignore[arg-type]
            cartesian_direction=tuple(
                float(value)
                for value in payload["cartesian_direction"]  # type: ignore[arg-type]
            ),
            cartesian_masses=tuple(
                float(value)
                for value in payload["cartesian_masses"]  # type: ignore[arg-type]
            ),
            imaginary_curvature=float(payload["imaginary_curvature"]),
            imaginary_frequency_cm1=None if frequency is None else float(frequency),
            oracle_point_group=str(payload.get("oracle_point_group", "unknown")),
            symmetry_characters=tuple(
                (str(label), int(value)) for label, value in sorted(characters.items())
            ),
            schema=str(payload.get("schema", "")),
            coordinate_kind=str(payload.get("coordinate_kind", "")),
            periodic=bool(payload.get("periodic", False)),
            kinetic_model=str(payload.get("kinetic_model", "")),
            vam_treatment=str(payload.get("vam_treatment", "")),
            coordinate_unit=str(payload.get("coordinate_unit", "angstrom_sqrt_amu")),
            cartesian_direction_unit=str(payload.get("cartesian_direction_unit", "amu^-1/2")),
            kinetic_coefficient_MHz=float(
                payload.get("kinetic_coefficient_MHz", QIM_KINETIC_COEFFICIENT_MHZ)
            ),
        )


def _cartesian_masses(masses: Sequence[float], dimension: int) -> np.ndarray:
    values = np.asarray(masses, dtype=float).reshape(-1)
    if values.size * 3 == dimension:
        values = np.repeat(values, 3)
    if values.shape != (dimension,) or np.any(values <= 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("provide one positive mass per atom or Cartesian component")
    return values


def _mode_characters(
    mode: np.ndarray,
    masses: np.ndarray,
    actions: Mapping[str, np.ndarray] | None,
    tolerance: float,
) -> tuple[tuple[str, int], ...]:
    if not actions:
        return ()
    sqrt_mass = np.sqrt(masses)
    mass_mode = sqrt_mass * mode
    result: list[tuple[str, int]] = []
    for label, raw_action in sorted(actions.items()):
        action = np.asarray(raw_action, dtype=float)
        if action.shape != (mode.size, mode.size) or not np.all(np.isfinite(action)):
            raise ValueError(f"ORACLE symmetry action {label!r} has the wrong shape")
        transformed = sqrt_mass * (action @ mode)
        character_value = float(np.dot(mass_mode, transformed))
        character = 1 if character_value >= 0.0 else -1
        if np.linalg.norm(transformed - character * mass_mode) > tolerance:
            raise ValueError(f"isolated Qim mode is not invariant under ORACLE operation {label!r}")
        result.append((str(label), character))
    return tuple(result)


def build_tunneling_coordinate_contract(
    hessian: np.ndarray,
    masses: Sequence[float],
    points: Sequence[float],
    *,
    periodic: bool,
    reference_coordinates_angstrom: np.ndarray | None = None,
    label: str = "Qim",
    imaginary_frequency_cm1: float | None = None,
    oracle_point_group: str = "unknown",
    oracle_symmetry_actions: Mapping[str, np.ndarray] | None = None,
    relative_curvature_tolerance: float = 1.0e-9,
    symmetry_tolerance: float = 1.0e-7,
) -> TunnelingCoordinateContract:
    """Select and construct the production tunneling coordinate.

    Periodic coordinates belong to SMITH's periodic rotor construction and are
    rejected here.  For a nonperiodic coordinate, exactly one significant
    negative eigenvalue is required.  This fail-closed rule prevents an MEP
    arc length or an index-two saddle from being promoted as a production
    one-dimensional tunneling model.
    """

    if periodic:
        raise ValueError("periodic tunneling coordinates require the Fourier-rotor contract")
    if relative_curvature_tolerance <= 0.0 or symmetry_tolerance <= 0.0:
        raise ValueError("Qim curvature and symmetry tolerances must be positive")
    matrix = np.asarray(hessian, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.all(np.isfinite(matrix)):
        raise ValueError("transition-state Hessian must be a finite square matrix")
    cartesian_masses = _cartesian_masses(masses, matrix.shape[0])
    inverse_sqrt = 1.0 / np.sqrt(cartesian_masses)
    weighted = matrix * inverse_sqrt[:, None] * inverse_sqrt[None, :]
    weighted = 0.5 * (weighted + weighted.T)
    if reference_coordinates_angstrom is None:
        vibrational_basis = np.eye(matrix.shape[0])
    else:
        geometry = np.asarray(reference_coordinates_angstrom, dtype=float)
        if geometry.shape != (matrix.shape[0] // 3, 3) or not np.all(np.isfinite(geometry)):
            raise ValueError("reference Cartesian geometry must match the transition-state Hessian")
        atom_masses = cartesian_masses.reshape((-1, 3))[:, 0]
        if not np.allclose(
            cartesian_masses.reshape((-1, 3)), atom_masses[:, None], atol=1e-12, rtol=0.0
        ):
            raise ValueError("rigid-mode projection requires one common mass per atom")
        centered = geometry - np.sum(atom_masses[:, None] * geometry, axis=0) / np.sum(
            atom_masses
        )
        rigid = []
        for axis in np.eye(3):
            rigid.append((np.sqrt(atom_masses)[:, None] * axis).reshape(-1))
        for axis in np.eye(3):
            rotation = np.cross(np.broadcast_to(axis, centered.shape), centered)
            rigid.append((np.sqrt(atom_masses)[:, None] * rotation).reshape(-1))
        _left, singular, right = np.linalg.svd(np.asarray(rigid), full_matrices=True)
        threshold = 1e-10 * max(float(singular[0]), 1.0)
        rigid_rank = int(np.count_nonzero(singular > threshold))
        vibrational_basis = right[rigid_rank:].T
    reduced = vibrational_basis.T @ weighted @ vibrational_basis
    eigenvalues, reduced_eigenvectors = np.linalg.eigh(0.5 * (reduced + reduced.T))
    eigenvectors = vibrational_basis @ reduced_eigenvectors
    scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
    negative = np.flatnonzero(eigenvalues < -abs(float(relative_curvature_tolerance)) * scale)
    if negative.size != 1:
        raise ValueError(
            "automatic Qim construction requires a certified index-one saddle; "
            f"found {negative.size} significant negative modes"
        )
    index = int(negative[0])
    direction = inverse_sqrt * eigenvectors[:, index]
    direction /= np.sqrt(np.dot(direction * cartesian_masses, direction))
    # Fix the otherwise arbitrary eigenvector sign deterministically.
    pivot = int(np.argmax(np.abs(direction)))
    if direction[pivot] < 0.0:
        direction *= -1.0
    characters = _mode_characters(
        direction,
        cartesian_masses,
        oracle_symmetry_actions,
        symmetry_tolerance,
    )
    return TunnelingCoordinateContract(
        label=str(label),
        points=tuple(float(value) for value in points),
        cartesian_direction=tuple(float(value) for value in direction),
        cartesian_masses=tuple(float(value) for value in cartesian_masses),
        imaginary_curvature=float(eigenvalues[index]),
        imaginary_frequency_cm1=(
            None if imaginary_frequency_cm1 is None else float(imaginary_frequency_cm1)
        ),
        oracle_point_group=str(oracle_point_group),
        symmetry_characters=characters,
    )


__all__ = [
    "QIM_KINETIC_COEFFICIENT_MHZ",
    "SMITH_TUNNELING_COORDINATE_SCHEMA",
    "TunnelingCoordinateContract",
    "build_tunneling_coordinate_contract",
]
