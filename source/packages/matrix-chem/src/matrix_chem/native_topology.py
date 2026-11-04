"""Optional compiled and GPU topology kernels with Python reference fallback."""

from __future__ import annotations

from typing import Any

import numpy as np

from matrix_numerics import NativeBackend, resolve_native_backend


try:
    from . import _chem_native
except ImportError:
    _chem_native = None


NATIVE_TOPOLOGY_EXTENSION = "matrix_chem._chem_native"


def native_topology_backend(
    workload_size: int,
    requested: str | None = None,
) -> NativeBackend:
    return resolve_native_backend(
        extension_available=_chem_native is not None,
        extension_name=NATIVE_TOPOLOGY_EXTENSION,
        workload_size=int(workload_size),
        requested=requested,
    )


def native_topology_build_info() -> dict[str, Any]:
    if _chem_native is None:
        return {
            "implementation": "python",
            "extension": NATIVE_TOPOLOGY_EXTENSION,
            "available": False,
        }
    return {
        **dict(_chem_native.build_info()),
        "extension": NATIVE_TOPOLOGY_EXTENSION,
        "available": True,
    }


def compiled_elementary_cycle_basis(
    natoms: int,
    edges: tuple[tuple[int, int], ...],
    allowed_atoms: set[int],
    ring_max: int | None,
) -> tuple[tuple[tuple[int, ...], ...], int, int]:
    if _chem_native is None:
        raise RuntimeError(f"{NATIVE_TOPOLOGY_EXTENSION} is unavailable")
    edge_array = np.asarray(edges, dtype=np.intp)
    if edge_array.size == 0:
        edge_array = np.empty((0, 2), dtype=np.intp)
    else:
        edge_array = edge_array.reshape(-1, 2)
    allowed = np.asarray(sorted(allowed_atoms), dtype=np.intp)
    cycles, candidate_count, cycle_rank = _chem_native.elementary_cycle_basis(
        int(natoms),
        edge_array,
        allowed,
        ring_max,
    )
    return (
        tuple(tuple(int(atom) for atom in cycle) for cycle in cycles),
        int(candidate_count),
        int(cycle_rank),
    )


def compiled_continuous_graph(
    coordinates: np.ndarray,
    atomic_numbers: np.ndarray,
    standard_radii: np.ndarray,
    pyykko_radii: np.ndarray,
    *,
    cutoff: float,
    cna_alpha: float,
    distance_scale: float,
    switch_alpha: float,
    lambda_strong: float,
    lambda_weak: float,
) -> tuple[Any, ...]:
    if _chem_native is None:
        raise RuntimeError(f"{NATIVE_TOPOLOGY_EXTENSION} is unavailable")
    values = _chem_native.perceive_continuous_graph(
            coordinates,
            np.asarray(atomic_numbers, dtype=np.intp),
            standard_radii,
            pyykko_radii,
            float(cutoff),
            float(cna_alpha),
            float(distance_scale),
            float(switch_alpha),
            float(lambda_strong),
            float(lambda_weak),
        )
    arrays = tuple(np.asarray(value) for value in values[:10])
    cycles = tuple(tuple(int(atom) for atom in cycle) for cycle in values[10])
    return (
        *arrays,
        cycles,
        int(values[11]),
        int(values[12]),
    )


__all__ = [
    "NATIVE_TOPOLOGY_EXTENSION",
    "compiled_elementary_cycle_basis",
    "compiled_continuous_graph",
    "native_topology_backend",
    "native_topology_build_info",
]
