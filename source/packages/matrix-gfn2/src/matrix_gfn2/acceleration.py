"""Hardware-keyed crossover calibration for MATRIX-GFN2 algebra."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np

from .locality import LocalizedOrbitalDomains
from .localized_solver import SparseSymmetricMatrix


CALIBRATION_SCHEMA = "matrix.gfn2.acceleration-calibration.v1"


@dataclass(frozen=True)
class HardwareFingerprint:
    digest: str
    system: str
    machine: str
    processor: str
    cpu_count: int
    python: str
    numpy: str
    accelerator: str


@dataclass(frozen=True)
class CrossoverCalibration:
    schema: str
    fingerprint: HardwareFingerprint
    gpu_backend: str
    gpu_min_orbitals: int
    sparse_min_orbitals: int
    sample_sizes: tuple[int, ...]
    dense_seconds: tuple[float, ...]
    accelerator_seconds: tuple[float, ...]
    sparse_seconds: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["sample_sizes"] = list(self.sample_sizes)
        result["dense_seconds"] = list(self.dense_seconds)
        result["accelerator_seconds"] = list(self.accelerator_seconds)
        result["sparse_seconds"] = list(self.sparse_seconds)
        return result


def _accelerator_name() -> str:
    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() > 0:
            return "cupy"
    except (ImportError, RuntimeError):
        pass
    try:
        import torch

        if torch.cuda.is_available():
            return "torch-cuda"
        if torch.backends.mps.is_built() and torch.backends.mps.is_available():
            return "torch-mps"
    except (ImportError, RuntimeError):
        pass
    return "numpy"


def hardware_fingerprint() -> HardwareFingerprint:
    fields = {
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": int(os.cpu_count() or 1),
        "python": ".".join(map(str, sys.version_info[:3])),
        "numpy": np.__version__,
        "accelerator": _accelerator_name(),
    }
    digest = sha256(
        json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]
    return HardwareFingerprint(digest=digest, **fields)


def default_calibration_path() -> Path:
    override = os.environ.get("MATRIX_GFN2_CALIBRATION", "").strip()
    if override:
        return Path(override).expanduser()
    cache_root = Path(
        os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    ).expanduser()
    return cache_root / "matrix" / "gfn2-acceleration.json"


def _minimum_time(operation, repeats: int) -> float:
    operation()
    samples = []
    for _ in range(max(1, repeats)):
        start = perf_counter()
        operation()
        samples.append(perf_counter() - start)
    return float(min(samples))


def _accelerator_square_time(matrix: np.ndarray, backend: str, repeats: int) -> float:
    values = np.asarray(matrix, dtype=np.float32)
    if backend == "cupy":
        import cupy as cp

        device = cp.asarray(values)

        def operation() -> None:
            _ = device @ device
            cp.cuda.get_current_stream().synchronize()

    elif backend in {"torch-cuda", "torch-mps"}:
        import torch

        name = "cuda" if backend == "torch-cuda" else "mps"
        device = torch.as_tensor(values, device=name)

        def operation() -> None:
            _ = device @ device
            if name == "cuda":
                torch.cuda.synchronize()
            else:
                torch.mps.synchronize()

    else:
        return float("inf")
    return _minimum_time(operation, repeats)


def _banded_sparse_time(matrix: np.ndarray, repeats: int) -> float:
    size = len(matrix)
    width = min(8, max(1, size - 1))
    pairs = tuple(
        (left, right)
        for left in range(size)
        for right in range(max(0, left - width), left + 1)
    )
    domains = LocalizedOrbitalDomains(
        atom_neighbors=tuple((index,) for index in range(size)),
        orbital_atom=tuple(range(size)),
        allowed_pairs=pairs,
    )
    sparse = SparseSymmetricMatrix.from_dense(matrix, domains, threshold=0.0)
    return _minimum_time(lambda: sparse.square(domains, threshold=0.0), repeats)


def calibrate_crossovers(
    *,
    sample_sizes: tuple[int, ...] = (48, 96, 192),
    repeats: int = 2,
) -> CrossoverCalibration:
    """Measure kernels and extrapolate conservative crossover dimensions."""

    if not sample_sizes or any(size < 8 for size in sample_sizes):
        raise ValueError("calibration sample sizes must be at least eight")
    rng = np.random.default_rng(20260923)
    dense_times = []
    accelerator_times = []
    sparse_times = []
    fingerprint = hardware_fingerprint()
    for size in sample_sizes:
        raw = rng.normal(size=(size, size))
        matrix = np.asarray(0.5 * (raw + raw.T), dtype=np.float64)
        dense_times.append(_minimum_time(lambda: matrix @ matrix, repeats))
        accelerator_times.append(
            _accelerator_square_time(matrix, fingerprint.accelerator, repeats)
        )
        sparse_times.append(_banded_sparse_time(matrix, repeats))
    finite_accelerator = [
        size
        for size, dense, accelerated in zip(
            sample_sizes, dense_times, accelerator_times, strict=True
        )
        if np.isfinite(accelerated) and accelerated < dense
    ]
    gpu_min = min(finite_accelerator) if finite_accelerator else sys.maxsize
    dense_coefficient = float(
        np.median(
            [time / size**3 for size, time in zip(sample_sizes, dense_times, strict=True)]
        )
    )
    sparse_coefficient = float(
        np.median(
            [time / size for size, time in zip(sample_sizes, sparse_times, strict=True)]
        )
    )
    sparse_min = int(
        np.clip(np.sqrt(sparse_coefficient / max(dense_coefficient, 1.0e-30)), 256, 16384)
    )
    return CrossoverCalibration(
        CALIBRATION_SCHEMA,
        fingerprint,
        fingerprint.accelerator,
        gpu_min,
        sparse_min,
        tuple(map(int, sample_sizes)),
        tuple(map(float, dense_times)),
        tuple(map(float, accelerator_times)),
        tuple(map(float, sparse_times)),
    )


def load_calibration(path: Path | None = None) -> CrossoverCalibration | None:
    source = default_calibration_path() if path is None else Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        fingerprint = HardwareFingerprint(**payload["fingerprint"])
        if payload["schema"] != CALIBRATION_SCHEMA:
            return None
        if fingerprint.digest != hardware_fingerprint().digest:
            return None
        return CrossoverCalibration(
            schema=payload["schema"],
            fingerprint=fingerprint,
            gpu_backend=payload["gpu_backend"],
            gpu_min_orbitals=int(payload["gpu_min_orbitals"]),
            sparse_min_orbitals=int(payload["sparse_min_orbitals"]),
            sample_sizes=tuple(payload["sample_sizes"]),
            dense_seconds=tuple(payload["dense_seconds"]),
            accelerator_seconds=tuple(payload["accelerator_seconds"]),
            sparse_seconds=tuple(payload["sparse_seconds"]),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def load_or_calibrate(path: Path | None = None) -> CrossoverCalibration:
    target = default_calibration_path() if path is None else Path(path)
    cached = load_calibration(target)
    if cached is not None:
        return cached
    calibrated = calibrate_crossovers()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(calibrated.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return calibrated


__all__ = [
    "CALIBRATION_SCHEMA",
    "CrossoverCalibration",
    "HardwareFingerprint",
    "calibrate_crossovers",
    "default_calibration_path",
    "hardware_fingerprint",
    "load_calibration",
    "load_or_calibrate",
]
