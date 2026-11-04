"""Small tensor-product DVR solver for multimode SONIC surfaces."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import csv


@dataclass(frozen=True)
class TensorDVRRequest:
    grids: tuple[np.ndarray, ...]
    potential_cm: np.ndarray
    kinetic_coefficients: tuple[float, ...] = ()
    levels: int = 12
    periodic: bool = True

    def __post_init__(self) -> None:
        if not 1 <= len(self.grids) <= 6:
            raise ValueError("tensor DVR supports one to six coordinates")
        shape = tuple(len(np.asarray(grid)) for grid in self.grids)
        potential = np.asarray(self.potential_cm, dtype=float)
        if potential.shape != shape or not np.all(np.isfinite(potential)):
            raise ValueError("multimode potential shape does not match coordinate grids")
        if any(len(grid) < 3 for grid in self.grids):
            raise ValueError("each multimode grid needs at least three points")
        if self.kinetic_coefficients and len(self.kinetic_coefficients) != len(self.grids):
            raise ValueError("kinetic coefficient count must match coordinate count")


@dataclass(frozen=True)
class TensorDVRResult:
    levels_cm: np.ndarray
    vectors: np.ndarray
    grid_shape: tuple[int, ...]
    solver: str = "tensor-product-finite-difference"


def solve_tensor_dvr(request: TensorDVRRequest) -> TensorDVRResult:
    shape = tuple(len(np.asarray(grid)) for grid in request.grids)
    coefficients = request.kinetic_coefficients or tuple(1.0 for _ in shape)
    dimension = int(np.prod(shape))
    hamiltonian = np.diag(np.asarray(request.potential_cm, dtype=float).reshape(-1))
    for axis, (grid, coefficient) in enumerate(zip(request.grids, coefficients, strict=True)):
        values = np.asarray(grid, dtype=float)
        spacing = float(values[1] - values[0])
        if spacing <= 0.0 or not np.allclose(np.diff(values), spacing):
            raise ValueError("multimode grids must be uniformly spaced")
        laplacian = _finite_difference_laplacian(len(values), spacing, request.periodic)
        left = int(np.prod(shape[:axis]))
        right = int(np.prod(shape[axis + 1:]))
        operator = np.kron(np.eye(left), np.kron(laplacian * float(coefficient), np.eye(right)))
        hamiltonian += operator
    values, vectors = np.linalg.eigh(0.5 * (hamiltonian + hamiltonian.T))
    count = min(max(1, int(request.levels)), dimension)
    return TensorDVRResult(values[:count], vectors[:, :count], shape)


def solve_tensor_dvr_from_surface(path: Path | str, coordinates: tuple[str, ...], *, levels: int = 12) -> TensorDVRResult:
    """Build and solve a tensor DVR directly from a rectangular SONIC CSV."""
    source = Path(path)
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not all(name in rows[0] for name in coordinates):
        raise ValueError("surface lacks the requested multimode coordinates")
    energy_key = "energy_cm-1" if "energy_cm-1" in rows[0] else "energy_hartree"
    grids = tuple(np.asarray(sorted({float(row[name]) for row in rows}), dtype=float) for name in coordinates)
    shape = tuple(len(grid) for grid in grids)
    values = np.full(shape, np.nan)
    for row in rows:
        index = tuple(int(np.searchsorted(grid, float(row[name]))) for grid, name in zip(grids, coordinates, strict=True))
        values[index] = float(row[energy_key])
    if not np.all(np.isfinite(values)):
        raise ValueError("multimode surface is not a complete rectangular tensor")
    if energy_key == "energy_hartree":
        values = (values - np.min(values)) * 219474.6313705
    return solve_tensor_dvr(TensorDVRRequest(grids=grids, potential_cm=values, levels=levels))


def _finite_difference_laplacian(size: int, spacing: float, periodic: bool) -> np.ndarray:
    matrix = np.diag(np.full(size, 2.0 / spacing**2))
    matrix += np.diag(np.full(size - 1, -1.0 / spacing**2), 1)
    matrix += np.diag(np.full(size - 1, -1.0 / spacing**2), -1)
    if periodic:
        matrix[0, -1] = matrix[-1, 0] = -1.0 / spacing**2
    else:
        matrix[0, 0] = matrix[-1, -1] = 1.0 / spacing**2
    return matrix


def write_tensor_dvr_result(path: Path | str, result: TensorDVRResult) -> Path:
    target = Path(path)
    target.write_text(json.dumps({
        "schema": "matrix.mctdh.tensor_dvr_result.v1",
        "solver": result.solver,
        "grid_shape": list(result.grid_shape),
        "levels_cm": [float(value) for value in result.levels_cm],
    }, indent=2) + "\n", encoding="utf-8")
    return target


__all__ = ["TensorDVRRequest", "TensorDVRResult", "solve_tensor_dvr", "solve_tensor_dvr_from_surface", "write_tensor_dvr_result"]
