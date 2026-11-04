"""Multidimensional curvilinear tunneling manifolds with retained kinetics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .curvilinear_tunneling import _atom_masses, mass_eckart_align


SMITH_TUNNELING_MANIFOLD_SCHEMA = "matrix.smith.curvilinear_tunneling_manifold.v1"


@dataclass(frozen=True)
class TunnelingSubspaceSelection:
    """Primary reaction direction and automatically retained transverse modes."""

    vectors: np.ndarray
    eigenvalues: tuple[float, ...]
    transverse_indices: tuple[int, ...]
    selection_scores: tuple[float, ...]


def select_tunneling_subspace(
    mass_weighted_hessian: np.ndarray,
    primary_direction: Sequence[float],
    *,
    path_curvature_direction: Sequence[float] | None = None,
    maximum_transverse_modes: int = 3,
    minimum_eigenvalue: float = 1.0e-10,
) -> TunnelingSubspaceSelection:
    """Select soft transverse modes after projecting out the reaction direction.

    If a path-curvature direction is supplied, modes are ranked by its squared
    overlap divided by their positive Hessian eigenvalue.  Otherwise the
    softest positive modes are selected.  This criterion is coordinate and
    molecule independent and records every score used in the decision.
    """

    hessian = np.asarray(mass_weighted_hessian, dtype=float)
    primary = np.asarray(primary_direction, dtype=float).reshape(-1)
    if hessian.shape != (primary.size, primary.size) or not np.all(np.isfinite(hessian)):
        raise ValueError("mass-weighted Hessian and primary direction are incompatible")
    if not np.allclose(hessian, hessian.T, atol=1.0e-10):
        raise ValueError("mass-weighted Hessian must be symmetric")
    norm = float(np.linalg.norm(primary))
    if norm <= 0.0 or maximum_transverse_modes < 0 or minimum_eigenvalue <= 0.0:
        raise ValueError("invalid tunneling-subspace selection controls")
    primary = primary / norm
    projector = np.eye(primary.size) - np.outer(primary, primary)
    projected = projector @ hessian @ projector
    values, vectors = np.linalg.eigh(0.5 * (projected + projected.T))
    candidates = np.flatnonzero(values > minimum_eigenvalue)
    scores = np.zeros_like(values)
    if path_curvature_direction is None:
        np.divide(1.0, values, out=scores, where=values > minimum_eigenvalue)
    else:
        curvature = projector @ np.asarray(path_curvature_direction, dtype=float).reshape(-1)
        if curvature.shape != primary.shape or not np.all(np.isfinite(curvature)):
            raise ValueError("path curvature direction is incompatible")
        overlaps = np.abs(vectors.T @ curvature) ** 2
        np.divide(overlaps, values, out=scores, where=values > minimum_eigenvalue)
    ordered = sorted(candidates, key=lambda index: (-scores[index], values[index], index))
    selected = tuple(int(index) for index in ordered[:maximum_transverse_modes])
    basis = np.column_stack((primary, *(vectors[:, index] for index in selected)))
    return TunnelingSubspaceSelection(
        vectors=basis,
        eigenvalues=tuple(float(values[index]) for index in selected),
        transverse_indices=selected,
        selection_scores=tuple(float(scores[index]) for index in selected),
    )


@dataclass(frozen=True)
class CurvilinearTunnelingManifold:
    """Tensor-grid reaction manifold with a full coordinate-dependent metric."""

    labels: tuple[str, ...]
    coordinates: tuple[tuple[float, ...], ...]
    geometries_angstrom: np.ndarray
    energies_hartree: np.ndarray
    atom_masses_amu: tuple[float, ...]
    kinetic_metric: np.ndarray
    center_indices: tuple[int, ...]
    metadata: Mapping[str, object] | None = None
    schema: str = SMITH_TUNNELING_MANIFOLD_SCHEMA
    coordinate_kind: str = "curvilinear_mass_manifold"
    periodic: bool = False

    def __post_init__(self) -> None:
        axes = tuple(np.asarray(axis, dtype=float) for axis in self.coordinates)
        shape = tuple(len(axis) for axis in axes)
        geometry = np.asarray(self.geometries_angstrom, dtype=float)
        energy = np.asarray(self.energies_hartree, dtype=float)
        metric = np.asarray(self.kinetic_metric, dtype=float)
        dimension = len(axes)
        if self.schema != SMITH_TUNNELING_MANIFOLD_SCHEMA or not 1 < dimension <= 4:
            raise ValueError("a curvilinear tunneling manifold needs two to four coordinates")
        if len(self.labels) != dimension or len(set(self.labels)) != dimension:
            raise ValueError("provide one unique label per manifold coordinate")
        if any(len(axis) < 3 or np.any(np.diff(axis) <= 0.0) for axis in axes):
            raise ValueError("manifold axes need three increasing points")
        natoms = len(self.atom_masses_amu)
        if geometry.shape != shape + (natoms, 3) or energy.shape != shape:
            raise ValueError("manifold geometry or energy tensor has the wrong shape")
        if metric.shape != shape + (dimension, dimension):
            raise ValueError("manifold kinetic metric has the wrong shape")
        if len(self.center_indices) != dimension or any(
            index < 0 or index >= shape[axis]
            for axis, index in enumerate(self.center_indices)
        ):
            raise ValueError("manifold center indices are invalid")
        if not np.all(np.isfinite(geometry)) or not np.all(np.isfinite(energy)):
            raise ValueError("manifold values must be finite")
        if not np.allclose(metric, np.swapaxes(metric, -1, -2), atol=1.0e-10):
            raise ValueError("manifold kinetic metric must be symmetric")
        if np.min(np.linalg.eigvalsh(metric)) <= 1.0e-12:
            raise ValueError("manifold kinetic metric must be positive definite")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "labels": list(self.labels),
            "coordinates": [list(axis) for axis in self.coordinates],
            "coordinate_kind": self.coordinate_kind,
            "periodic": self.periodic,
            "geometries_angstrom": np.asarray(self.geometries_angstrom).tolist(),
            "energies_hartree": np.asarray(self.energies_hartree).tolist(),
            "atom_masses_amu": list(self.atom_masses_amu),
            "kinetic_metric": np.asarray(self.kinetic_metric).tolist(),
            "center_indices": list(self.center_indices),
            "metadata": dict(self.metadata or {}),
        }


def build_curvilinear_tunneling_manifold(
    coordinate_axes: Sequence[Sequence[float]],
    geometries_angstrom: np.ndarray,
    energies_hartree: np.ndarray,
    masses_amu: Sequence[float],
    *,
    center_indices: Sequence[int],
    labels: Sequence[str] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> CurvilinearTunnelingManifold:
    """Eckart-align a tensor grid and retain its full curvilinear metric."""

    axes = tuple(np.asarray(axis, dtype=float) for axis in coordinate_axes)
    shape = tuple(len(axis) for axis in axes)
    raw = np.asarray(geometries_angstrom, dtype=float)
    if raw.ndim != len(axes) + 2 or raw.shape[:-2] != shape or raw.shape[-1] != 3:
        raise ValueError("geometry tensor does not match the coordinate axes")
    masses = _atom_masses(masses_amu, raw.shape[-2])
    centers = tuple(int(value) for value in center_indices)
    if len(centers) != len(axes):
        raise ValueError("provide one center index per coordinate")
    reference = raw[centers]
    aligned = np.empty_like(raw)
    for index in np.ndindex(shape):
        aligned[index] = mass_eckart_align(raw[index], reference, masses)
    derivatives = [
        np.gradient(aligned, axis, axis=coordinate, edge_order=2)
        for coordinate, axis in enumerate(axes)
    ]
    metric = np.empty(shape + (len(axes), len(axes)))
    for left, derivative_left in enumerate(derivatives):
        for right, derivative_right in enumerate(derivatives):
            metric[..., left, right] = np.einsum(
                "i,...ia,...ia->...", masses, derivative_left, derivative_right
            )
    return CurvilinearTunnelingManifold(
        labels=tuple(labels or (f"q{index + 1}" for index in range(len(axes)))),
        coordinates=tuple(tuple(float(value) for value in axis) for axis in axes),
        geometries_angstrom=aligned,
        energies_hartree=np.asarray(energies_hartree, dtype=float),
        atom_masses_amu=tuple(float(value) for value in masses),
        kinetic_metric=metric,
        center_indices=centers,
        metadata=metadata,
    )


__all__ = [
    "CurvilinearTunnelingManifold",
    "SMITH_TUNNELING_MANIFOLD_SCHEMA",
    "TunnelingSubspaceSelection",
    "build_curvilinear_tunneling_manifold",
    "select_tunneling_subspace",
]
