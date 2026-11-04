"""Vectorized pair kernels shared by CPU and accelerator batches."""

from __future__ import annotations

from typing import Any
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ._array_cache import triangular_indices

try:
    from . import _gfn2_native
except ImportError:  # pragma: no cover - portable source-only installation
    _gfn2_native = None


@dataclass(frozen=True)
class SparsePairGeometry:
    left: np.ndarray
    right: np.ndarray
    displacement: np.ndarray
    distance: np.ndarray
    omitted_pairs: int
    cutoff_bohr: float


def sparse_pair_geometry(
    coordinates_bohr: np.ndarray, *, cutoff_bohr: float
) -> SparsePairGeometry:
    """Build unique atom pairs with the native cell-list implementation."""

    coordinates = np.asarray(coordinates_bohr, dtype=float)
    return _sparse_pair_geometry_cached(
        tuple(coordinates.shape), coordinates.tobytes(), float(cutoff_bohr)
    )


@lru_cache(maxsize=64)
def _sparse_pair_geometry_cached(
    shape: tuple[int, ...], coordinate_bytes: bytes, cutoff_bohr: float
) -> SparsePairGeometry:
    """Reuse one atom-pair geometry across all terms of an evaluation."""

    coordinates = np.frombuffer(coordinate_bytes, dtype=float).reshape(shape)
    natoms = len(coordinates)
    total = natoms * (natoms - 1) // 2
    if total <= 750_000:
        # In the dense molecular regime, vectorized triangular distances are
        # cheaper than building a cell-list graph and use bounded scratch.
        all_left, all_right = triangular_indices(natoms, diagonal=False)
        all_displacement = coordinates[all_left] - coordinates[all_right]
        distance2 = np.einsum("ij,ij->i", all_displacement, all_displacement)
        keep = distance2 <= cutoff_bohr * cutoff_bohr
        left, right = all_left[keep], all_right[keep]
        displacement = all_displacement[keep]
    elif _gfn2_native is not None and hasattr(_gfn2_native, "localized_domains"):
        _, _, left, right = _gfn2_native.localized_domains(
            coordinates, np.arange(natoms, dtype=np.int64), float(cutoff_bohr), 0
        )
        keep = left != right
        left = np.asarray(left[keep], dtype=np.intp)
        right = np.asarray(right[keep], dtype=np.intp)
        displacement = coordinates[left] - coordinates[right]
    else:
        from .locality import LocalityPolicy, build_localized_domains

        domains = build_localized_domains(
            coordinates, tuple(range(natoms)),
            LocalityPolicy(
                interaction_cutoff_bohr=float(cutoff_bohr),
                buffer_hops=0,
                truncation_threshold=1.0e-14,
            ),
        )
        left, right = domains.allowed_pair_arrays()
        keep = left != right
        left = np.asarray(left[keep], dtype=np.intp)
        right = np.asarray(right[keep], dtype=np.intp)
        displacement = coordinates[left] - coordinates[right]
    distance = np.linalg.norm(displacement, axis=1)
    return SparsePairGeometry(
        left, right, displacement, distance, total - len(left), cutoff_bohr
    )


def gfn2_coordination_sparse(
    coordinates_bohr: np.ndarray,
    covalent_radii_bohr: np.ndarray,
    *,
    cutoff_bohr: float = 24.0,
) -> tuple[np.ndarray, float]:
    geometry = sparse_pair_geometry(coordinates_bohr, cutoff_bohr=cutoff_bohr)
    radii = np.asarray(covalent_radii_bohr, dtype=float)
    radius_sum = radii[geometry.left] + radii[geometry.right]
    first = 1.0 / (1.0 + np.exp(-10.0 * (4.0 * radius_sum / (3.0 * geometry.distance) - 1.0)))
    second = 1.0 / (
        1.0 + np.exp(-20.0 * ((4.0 * radius_sum / 3.0 + 2.0) / geometry.distance - 1.0))
    )
    contribution = first * second
    result = np.zeros(len(radii), dtype=float)
    np.add.at(result, geometry.left, contribution)
    np.add.at(result, geometry.right, contribution)
    maximum_radius = float(np.max(radii, initial=0.0))
    radius_sum_bound = 2.0 * maximum_radius
    tail_pair = (
        1.0 / (1.0 + np.exp(-10.0 * (4.0 * radius_sum_bound / (3.0 * cutoff_bohr) - 1.0)))
        / (1.0 + np.exp(-20.0 * ((4.0 * radius_sum_bound / 3.0 + 2.0) / cutoff_bohr - 1.0)))
    )
    return result, float(geometry.omitted_pairs * tail_pair)


def d4_coordination_sparse(
    coordinates_bohr: np.ndarray,
    covalent_radii_bohr: np.ndarray,
    electronegativities: np.ndarray,
    *,
    cutoff_bohr: float = 24.0,
) -> tuple[np.ndarray, float]:
    from math import erf

    geometry = sparse_pair_geometry(coordinates_bohr, cutoff_bohr=cutoff_bohr)
    radii = np.asarray(covalent_radii_bohr, dtype=float)
    en = np.asarray(electronegativities, dtype=float)
    critical = (4.0 / 3.0) * (radii[geometry.left] + radii[geometry.right])
    argument = -7.5 * (geometry.distance - critical) / critical
    count = 0.5 * (1.0 + np.fromiter((erf(value) for value in argument), float))
    factor = 4.10451 * np.exp(
        -(np.abs(en[geometry.left] - en[geometry.right]) + 19.08857) ** 2
        / (2.0 * 11.28174**2)
    )
    contribution = count * factor
    result = np.zeros(len(radii), dtype=float)
    np.add.at(result, geometry.left, contribution)
    np.add.at(result, geometry.right, contribution)
    maximum_critical = (8.0 / 3.0) * float(np.max(radii, initial=0.0))
    tail_argument = -7.5 * (cutoff_bohr - maximum_critical) / maximum_critical
    tail_pair = 0.5 * (1.0 + erf(tail_argument)) * 4.10451
    return result, float(geometry.omitted_pairs * tail_pair)


def _namespace(backend: str) -> tuple[Any, str]:
    normalized = str(backend).strip().lower()
    if normalized in {"numpy", "cpu"}:
        return np, "numpy"
    if normalized == "cupy":
        import cupy as cp

        return cp, "cupy"
    if normalized in {"torch-cuda", "cuda", "torch-mps", "mps"}:
        import torch

        return torch, "torch"
    raise ValueError(f"unsupported pair-kernel backend: {backend}")


def batched_pair_geometry(
    coordinates: np.ndarray,
    *,
    backend: str = "numpy",
):
    """Return displacement and distance arrays for one geometry or a batch."""

    values = np.asarray(coordinates, dtype=float)
    single = values.ndim == 2
    if single:
        values = values[None, ...]
    if values.ndim != 3 or values.shape[2] != 3 or not np.all(np.isfinite(values)):
        raise ValueError("pair geometry requires (..., natoms, 3) finite coordinates")
    namespace, kind = _namespace(backend)
    if kind == "torch":
        import torch

        device = "mps" if str(backend).lower() in {"mps", "torch-mps"} else "cuda"
        dtype = torch.float32 if device == "mps" else torch.float64
        array = torch.as_tensor(values, dtype=dtype, device=device)
        displacement = array[:, :, None, :] - array[:, None, :, :]
        distances = torch.linalg.vector_norm(displacement, dim=-1)
    else:
        array = namespace.asarray(values)
        displacement = array[:, :, None, :] - array[:, None, :, :]
        distances = namespace.linalg.norm(displacement, axis=-1)
    if single:
        return displacement[0], distances[0]
    return displacement, distances


def batched_gfn2_coordination(
    coordinates_bohr: np.ndarray,
    covalent_radii_bohr: np.ndarray,
    *,
    backend: str = "numpy",
):
    """Vectorized GFN2 Eq. 18 coordination numbers for batched structures."""

    if backend == "torch":
        import torch

        values = torch.as_tensor(coordinates_bohr, dtype=torch.float64)
        single = values.ndim == 2
        if single:
            values = values.unsqueeze(0)
        radii = torch.as_tensor(covalent_radii_bohr, dtype=values.dtype, device=values.device)
        if radii.shape != (values.shape[1],):
            raise ValueError("coordination radii must contain one value per atom")
        eye = torch.eye(len(radii), dtype=torch.bool, device=values.device)[None]
        squared = ((values[:, :, None] - values[:, None])**2).sum(-1)
        safe = torch.sqrt(torch.where(eye, torch.ones_like(squared), squared))
        critical = 4.0 / 3.0 * (radii[:, None] + radii[None, :])
        first = torch.sigmoid(10.0 * (critical / safe - 1.0))
        second = torch.sigmoid(20.0 * ((critical + 2.0) / safe - 1.0))
        result = torch.where(eye, 0.0, first * second).sum(-1)
        return result[0] if single else result

    values = np.asarray(coordinates_bohr, dtype=float)
    single = values.ndim == 2
    if single:
        values = values[None, ...]
    radii = np.asarray(covalent_radii_bohr, dtype=float)
    if radii.shape != (values.shape[1],):
        raise ValueError("coordination radii must contain one value per atom")
    if single and _gfn2_native is not None and hasattr(
        _gfn2_native, "gfn2_coordination"
    ):
        return np.asarray(_gfn2_native.gfn2_coordination(values[0], radii))
    _, distances = batched_pair_geometry(values, backend=backend)
    namespace, kind = _namespace(backend)
    if kind == "torch":
        import torch

        device = distances.device
        radius = torch.as_tensor(radii, dtype=distances.dtype, device=device)
        eye = torch.eye(len(radii), dtype=torch.bool, device=device)[None, :, :]
        safe = torch.where(eye, torch.ones_like(distances), distances)
        radius_sum = radius[:, None] + radius[None, :]
        first = torch.sigmoid(10.0 * (4.0 * radius_sum / (3.0 * safe) - 1.0))
        second = torch.sigmoid(
            20.0 * ((4.0 * radius_sum / 3.0 + 2.0) / safe - 1.0)
        )
        result = torch.sum(torch.where(eye, 0.0, first * second), dim=2)
    else:
        eye = namespace.eye(len(radii), dtype=bool)[None, :, :]
        safe = namespace.where(eye, 1.0, distances)
        radius = namespace.asarray(radii)
        radius_sum = radius[:, None] + radius[None, :]
        first = 1.0 / (1.0 + namespace.exp(-10.0 * (4.0 * radius_sum / (3.0 * safe) - 1.0)))
        second = 1.0 / (
            1.0
            + namespace.exp(
                -20.0 * ((4.0 * radius_sum / 3.0 + 2.0) / safe - 1.0)
            )
        )
        result = namespace.sum(namespace.where(eye, 0.0, first * second), axis=2)
    return result[0] if single else result


def batched_d4_coordination(
    coordinates_bohr: np.ndarray,
    covalent_radii_bohr: np.ndarray,
    electronegativities: np.ndarray,
) -> np.ndarray:
    """Vectorized D4 covalent coordination including EN weighting."""

    values = np.asarray(coordinates_bohr, dtype=float)
    single = values.ndim == 2
    if single:
        values = values[None, ...]
    radii = np.asarray(covalent_radii_bohr, dtype=float)
    en = np.asarray(electronegativities, dtype=float)
    if radii.shape != (values.shape[1],) or en.shape != radii.shape:
        raise ValueError("D4 pair parameters must contain one value per atom")
    if single and _gfn2_native is not None and hasattr(
        _gfn2_native, "d4_coordination"
    ):
        return np.asarray(_gfn2_native.d4_coordination(values[0], radii, en))
    _, distances = batched_pair_geometry(values)
    eye = np.eye(len(radii), dtype=bool)[None, :, :]
    critical = (4.0 / 3.0) * (radii[:, None] + radii[None, :])
    argument = -7.5 * (distances - critical) / critical
    count = 0.5 * (1.0 + np.vectorize(__import__("math").erf)(argument))
    factor = 4.10451 * np.exp(
        -(np.abs(en[:, None] - en[None, :]) + 19.08857) ** 2
        / (2.0 * 11.28174**2)
    )
    return np.sum(np.where(eye, 0.0, count * factor), axis=2)[0 if single else slice(None)]


__all__ = [
    "batched_d4_coordination",
    "batched_gfn2_coordination",
    "batched_pair_geometry",
    "SparsePairGeometry",
    "d4_coordination_sparse",
    "gfn2_coordination_sparse",
    "sparse_pair_geometry",
]
