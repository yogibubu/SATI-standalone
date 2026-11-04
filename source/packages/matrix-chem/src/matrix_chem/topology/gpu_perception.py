"""GPU screening for topology pairs with mandatory CPU float64 certification."""

from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np

from matrix_numerics import resolve_compute_backend


DEFAULT_TOPOLOGY_GPU_MIN_ATOMS = 1024
DEFAULT_TOPOLOGY_GPU_TILE = 1024


@dataclass(frozen=True)
class GPUCandidatePairs:
    left: np.ndarray
    right: np.ndarray
    distances: np.ndarray
    backend: str
    device: str
    screening_precision: str
    certified_precision: str = "float64"


def gpu_screen_candidate_pairs(
    coordinates: np.ndarray,
    *,
    cutoff: float,
) -> GPUCandidatePairs | None:
    """Screen all-pair distances on GPU and certify survivors on the CPU.

    MPS screening uses a conservative float32 envelope. CUDA uses float64.
    Returned membership and distances are always recomputed in CPU float64,
    so downstream graph construction is independent of accelerator precision.
    """

    xyz = np.asarray(coordinates, dtype=float)
    count = len(xyz)
    requested_device = os.environ.get(
        "MATRIX_DEVICE",
        os.environ.get("MATRIX_COMPUTE_DEVICE", "auto"),
    ).strip().casefold()
    if not _truthy_env("MATRIX_TOPOLOGY_GPU_ENABLED", False) and requested_device in {
        "",
        "auto",
        "cpu",
        "numpy",
    }:
        return None
    minimum = _positive_env_int(
        "MATRIX_TOPOLOGY_GPU_MIN_ATOMS",
        DEFAULT_TOPOLOGY_GPU_MIN_ATOMS,
    )
    if count < minimum:
        return None
    selected = resolve_compute_backend(
        workload_size=count,
        allow_mixed_precision=True,
        require_float64=False,
        gpu_min_batch=minimum,
    )
    if selected.device not in {"mps", "cuda"} or not selected.available:
        return None
    try:
        return _torch_screen(xyz, float(cutoff), selected.device)
    except Exception:
        if _truthy_env("MATRIX_DEVICE_STRICT", False):
            raise
        return None


def _torch_screen(
    coordinates: np.ndarray,
    cutoff: float,
    device: str,
) -> GPUCandidatePairs:
    import torch

    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("topology GPU cutoff must be finite and positive")
    tile = _positive_env_int(
        "MATRIX_TOPOLOGY_GPU_TILE",
        DEFAULT_TOPOLOGY_GPU_TILE,
    )
    centered = coordinates - np.mean(coordinates, axis=0)
    dtype = torch.float32 if device == "mps" else torch.float64
    precision = "float32" if device == "mps" else "float64"
    epsilon = np.finfo(np.float32).eps if device == "mps" else np.finfo(float).eps
    extent = float(np.max(np.abs(centered), initial=0.0))
    # Subtraction, squaring and summation each contribute rounding error.
    # The deliberately loose envelope may add false positives, all removed by
    # the mandatory float64 certification below, but cannot alter graph edges.
    distance_envelope = 64.0 * epsilon * max(1.0, extent, cutoff)
    screening_cutoff2 = (cutoff + distance_envelope) ** 2
    tensor = torch.as_tensor(centered, dtype=dtype, device=device)
    left_parts: list[np.ndarray] = []
    right_parts: list[np.ndarray] = []
    for left_start in range(0, len(centered), tile):
        left_stop = min(left_start + tile, len(centered))
        left_block = tensor[left_start:left_stop]
        for right_start in range(left_start, len(centered), tile):
            right_stop = min(right_start + tile, len(centered))
            right_block = tensor[right_start:right_stop]
            squared = torch.sum(
                (left_block[:, None, :] - right_block[None, :, :]) ** 2,
                dim=2,
            )
            mask = squared <= screening_cutoff2
            if right_start == left_start:
                mask = torch.triu(mask, diagonal=1)
            indices = torch.nonzero(mask, as_tuple=False).detach().cpu().numpy()
            if len(indices):
                left_parts.append(indices[:, 0].astype(np.intp) + left_start)
                right_parts.append(indices[:, 1].astype(np.intp) + right_start)
    if not left_parts:
        empty = np.empty(0, dtype=np.intp)
        return GPUCandidatePairs(
            empty,
            empty.copy(),
            np.empty(0, dtype=float),
            backend="torch",
            device=device,
            screening_precision=precision,
        )
    left = np.concatenate(left_parts)
    right = np.concatenate(right_parts)
    order = np.lexsort((right, left))
    left = left[order]
    right = right[order]
    deltas = coordinates[left] - coordinates[right]
    squared = np.einsum("ij,ij->i", deltas, deltas)
    certified = squared <= cutoff**2
    return GPUCandidatePairs(
        left=left[certified],
        right=right[certified],
        distances=np.sqrt(squared[certified]),
        backend="torch",
        device=device,
        screening_precision=precision,
    )


def _positive_env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        return int(default)
    return value if value > 0 else int(default)


def _truthy_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().casefold() not in {"", "0", "false", "no", "off"}


__all__ = [
    "DEFAULT_TOPOLOGY_GPU_MIN_ATOMS",
    "DEFAULT_TOPOLOGY_GPU_TILE",
    "GPUCandidatePairs",
    "gpu_screen_candidate_pairs",
]
