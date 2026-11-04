"""Curvilinear SMITH paths with an explicit Cartesian kinetic metric."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np


SMITH_CURVILINEAR_PATH_SCHEMA = "matrix.smith.curvilinear_tunneling_path.v1"


def _atom_masses(masses: Sequence[float], atom_count: int) -> np.ndarray:
    values = np.asarray(masses, dtype=float).reshape(-1)
    if values.size == 3 * atom_count:
        triples = values.reshape((-1, 3))
        if not np.allclose(triples, triples[:, :1], rtol=0.0, atol=1.0e-12):
            raise ValueError("Cartesian x, y, and z masses must agree for every atom")
        values = triples[:, 0]
    if values.shape != (atom_count,) or np.any(values <= 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("provide one positive finite mass per atom or Cartesian component")
    return values


def mass_eckart_align(
    coordinates_angstrom: np.ndarray,
    reference_angstrom: np.ndarray,
    masses_amu: Sequence[float],
    *, return_rotation: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Remove translation and rotation by a mass-weighted Kabsch fit."""

    moving = np.asarray(coordinates_angstrom, dtype=float)
    reference = np.asarray(reference_angstrom, dtype=float)
    if moving.ndim != 2 or moving.shape[1] != 3 or reference.shape != moving.shape:
        raise ValueError("Eckart alignment requires matching natoms by 3 geometries")
    masses = _atom_masses(masses_amu, len(moving))
    total = float(np.sum(masses))
    moving_center = np.sum(masses[:, None] * moving, axis=0) / total
    reference_center = np.sum(masses[:, None] * reference, axis=0) / total
    x = moving - moving_center
    y = reference - reference_center
    left, _singular, right = np.linalg.svd((x * masses[:, None]).T @ y)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(left @ right))
    rotation = left @ correction @ right
    aligned = x @ rotation + reference_center
    return (aligned, rotation) if return_rotation else aligned


@dataclass(frozen=True)
class CurvilinearTunnelingPath:
    """A relaxed nonperiodic path and its retained mass metric."""

    label: str
    coordinates: tuple[float, ...]
    geometries_angstrom: np.ndarray
    energies_hartree: tuple[float, ...]
    atom_masses_amu: tuple[float, ...]
    kinetic_metric: tuple[float, ...]
    curvature: tuple[float, ...]
    center_index: int
    metadata: Mapping[str, object] | None = None
    schema: str = SMITH_CURVILINEAR_PATH_SCHEMA
    coordinate_kind: str = "curvilinear_mass_path"
    periodic: bool = False
    kinetic_model: str = "cartesian_mass_metric"
    vam_treatment: str = "explicit_metric"
    coordinate_unit: str = "angstrom_sqrt_amu"

    def __post_init__(self) -> None:
        grid = np.asarray(self.coordinates, dtype=float)
        geometries = np.asarray(self.geometries_angstrom, dtype=float)
        energies = np.asarray(self.energies_hartree, dtype=float)
        metric = np.asarray(self.kinetic_metric, dtype=float)
        curvature = np.asarray(self.curvature, dtype=float)
        if self.schema != SMITH_CURVILINEAR_PATH_SCHEMA:
            raise ValueError("unsupported SMITH curvilinear-path schema")
        if not self.label.strip() or grid.ndim != 1 or grid.size < 5 or np.any(np.diff(grid) <= 0.0):
            raise ValueError("a curvilinear tunneling path needs five increasing images")
        if geometries.ndim != 3 or geometries.shape != (grid.size, len(self.atom_masses_amu), 3):
            raise ValueError("curvilinear path geometries must have shape nimages x natoms x 3")
        if energies.shape != grid.shape or metric.shape != grid.shape or curvature.shape != grid.shape:
            raise ValueError("path energies, metric, and curvature must match the image grid")
        if not np.all(np.isfinite(geometries)) or not np.all(np.isfinite(energies)):
            raise ValueError("curvilinear path values must be finite")
        if np.any(metric <= 1.0e-12) or not np.all(np.isfinite(metric)) or np.any(curvature < 0.0):
            raise ValueError("path metric must be positive and curvature nonnegative")
        if not 0 <= self.center_index < grid.size or abs(grid[self.center_index]) > 1.0e-10:
            raise ValueError("the curvilinear saddle image must have coordinate zero")
        if self.periodic or self.kinetic_model != "cartesian_mass_metric":
            raise ValueError("this contract describes a nonperiodic Cartesian-metric path")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "label": self.label,
            "coordinates": list(self.coordinates),
            "coordinate_kind": self.coordinate_kind,
            "coordinate_unit": self.coordinate_unit,
            "periodic": self.periodic,
            "kinetic_model": self.kinetic_model,
            "vam_treatment": self.vam_treatment,
            "geometries_angstrom": np.asarray(self.geometries_angstrom).tolist(),
            "energies_hartree": list(self.energies_hartree),
            "atom_masses_amu": list(self.atom_masses_amu),
            "kinetic_metric": list(self.kinetic_metric),
            "curvature": list(self.curvature),
            "center_index": self.center_index,
            "metadata": dict(self.metadata or {}),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "CurvilinearTunnelingPath":
        return cls(
            label=str(payload["label"]),
            coordinates=tuple(float(x) for x in payload["coordinates"]),  # type: ignore[arg-type]
            geometries_angstrom=np.asarray(payload["geometries_angstrom"], dtype=float),
            energies_hartree=tuple(float(x) for x in payload["energies_hartree"]),  # type: ignore[arg-type]
            atom_masses_amu=tuple(float(x) for x in payload["atom_masses_amu"]),  # type: ignore[arg-type]
            kinetic_metric=tuple(float(x) for x in payload["kinetic_metric"]),  # type: ignore[arg-type]
            curvature=tuple(float(x) for x in payload["curvature"]),  # type: ignore[arg-type]
            center_index=int(payload["center_index"]),
            metadata=payload.get("metadata"),  # type: ignore[arg-type]
            schema=str(payload.get("schema", "")),
        )


def build_curvilinear_tunneling_path(
    geometries_angstrom: Sequence[np.ndarray],
    energies_hartree: Sequence[float],
    masses_amu: Sequence[float],
    *,
    center_index: int,
    label: str = "s",
    metadata: Mapping[str, object] | None = None,
) -> CurvilinearTunnelingPath:
    """Build an Eckart-aligned, mass-arc-length path without discarding its metric."""

    raw = np.asarray(geometries_angstrom, dtype=float)
    if raw.ndim != 3 or raw.shape[2] != 3 or not 0 <= center_index < len(raw):
        raise ValueError("path images and center_index are inconsistent")
    masses = _atom_masses(masses_amu, raw.shape[1])
    aligned = np.empty_like(raw)
    aligned[center_index] = raw[center_index]
    for index in range(center_index + 1, len(raw)):
        aligned[index] = mass_eckart_align(raw[index], aligned[index - 1], masses)
    for index in range(center_index - 1, -1, -1):
        aligned[index] = mass_eckart_align(raw[index], aligned[index + 1], masses)
    segment = np.sqrt(
        np.einsum("i,qia,qia->q", masses, np.diff(aligned, axis=0), np.diff(aligned, axis=0))
    )
    if np.any(segment <= 1.0e-10):
        raise ValueError("successive curvilinear images must be distinct")
    coordinate = np.concatenate(([0.0], np.cumsum(segment)))
    coordinate -= coordinate[center_index]
    edge_order = 2 if len(raw) > 2 else 1
    derivative = np.gradient(aligned, coordinate, axis=0, edge_order=edge_order)
    metric = np.einsum("i,qia,qia->q", masses, derivative, derivative)
    tangent = derivative / np.sqrt(metric)[:, None, None]
    tangent_derivative = np.gradient(tangent, coordinate, axis=0, edge_order=edge_order)
    curvature = np.sqrt(np.einsum("i,qia,qia->q", masses, tangent_derivative, tangent_derivative))
    return CurvilinearTunnelingPath(
        label=label,
        coordinates=tuple(float(x) for x in coordinate),
        geometries_angstrom=aligned,
        energies_hartree=tuple(float(x) for x in energies_hartree),
        atom_masses_amu=tuple(float(x) for x in masses),
        kinetic_metric=tuple(float(x) for x in metric),
        curvature=tuple(float(x) for x in curvature),
        center_index=center_index,
        metadata=metadata,
    )


def reconstruct_curvilinear_isotopologue(
    path: CurvilinearTunnelingPath,
    masses_amu: Sequence[float],
    *,
    label: str,
    substitutions: Mapping[int, int] | None = None,
) -> CurvilinearTunnelingPath:
    """Rebuild alignment, arc length, and kinetic metric for new nuclear masses.

    The Born--Oppenheimer Cartesian geometries and energies are reused, while
    every mass-dependent path quantity is recomputed.  Substitution indices in
    the provenance mapping are one based.
    """

    return build_curvilinear_tunneling_path(
        np.asarray(path.geometries_angstrom, dtype=float),
        path.energies_hartree,
        masses_amu,
        center_index=path.center_index,
        label=path.label,
        metadata={
            **dict(path.metadata or {}),
            "isotopologue": str(label),
            "isotope_substitutions": {
                str(index): int(mass_number)
                for index, mass_number in sorted((substitutions or {}).items())
            },
            "isotopic_reconstruction": (
                "Born-Oppenheimer Cartesian path retained; Eckart alignment, "
                "mass arc length, and kinetic metric rebuilt"
            ),
        },
    )


def replace_curvilinear_path_energies(
    path: CurvilinearTunnelingPath,
    energies_hartree: Sequence[float],
    *,
    electronic_model: str,
    metadata: Mapping[str, object] | None = None,
) -> CurvilinearTunnelingPath:
    """Attach higher-level single-point energies without changing path kinetics."""

    energies = np.asarray(energies_hartree, dtype=float)
    if energies.shape != np.asarray(path.coordinates).shape or not np.all(np.isfinite(energies)):
        raise ValueError("replacement energies must be finite and match every path image")
    if not electronic_model.strip():
        raise ValueError("replacement path energies require an electronic-model identity")
    return replace(
        path,
        energies_hartree=tuple(float(value) for value in energies),
        metadata={
            **dict(path.metadata or {}),
            **dict(metadata or {}),
            "electronic_model": str(electronic_model),
            "energy_replacement": "single-point energies on unchanged SMITH geometries",
        },
    )


__all__ = [
    "CurvilinearTunnelingPath",
    "SMITH_CURVILINEAR_PATH_SCHEMA",
    "build_curvilinear_tunneling_path",
    "mass_eckart_align",
    "reconstruct_curvilinear_isotopologue",
    "replace_curvilinear_path_energies",
]
