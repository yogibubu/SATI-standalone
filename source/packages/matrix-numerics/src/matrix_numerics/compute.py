"""Runtime selection of optional MATRIX compute accelerators.

The CPU/NumPy implementation is always the scientific reference.  Optional
accelerators are selected at runtime and never change the serialized model or
the public scientific API.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from typing import Any

from .native import DEFAULT_NATIVE_MIN_WORK


DEFAULT_GPU_MIN_BATCH = 512
COMPUTE_BACKEND_SCHEMA = "matrix.compute.backend.v1"


@dataclass(frozen=True)
class ComputeBackend:
    """One available or requested numerical execution backend."""

    name: str
    device: str
    precision: str
    available: bool
    accelerated: bool
    supports_float64: bool
    core_count: int = 0
    requested: str = "auto"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": COMPUTE_BACKEND_SCHEMA,
            "requested": self.requested,
            "backend": self.name,
            "device": self.device,
            "precision": self.precision,
            "available": self.available,
            "accelerated": self.accelerated,
            "supports_float64": self.supports_float64,
            "core_count": self.core_count,
            "reason": self.reason,
        }


@lru_cache(maxsize=1)
def available_compute_backends() -> tuple[ComputeBackend, ...]:
    """Return optional numerical and neural accelerators plus the CPU reference."""

    return (
        _torch_cuda_backend(),
        _torch_mps_backend(),
        _coreml_neural_engine_backend(),
        _cpu_backend(),
    )


def resolve_compute_backend(
    requested: str | None = None,
    *,
    workload_size: int = 0,
    allow_mixed_precision: bool = False,
    require_float64: bool = False,
    gpu_min_batch: int | None = None,
) -> ComputeBackend:
    """Select a backend while preserving a deterministic CPU fallback.

    ``auto`` only selects a GPU for a sufficiently large workload.  Apple MPS
    is selected only when the caller explicitly permits mixed precision,
    because its current Torch backend does not implement ``float64``.
    """

    choice = (
        requested
        if requested is not None
        else os.environ.get("MATRIX_DEVICE", os.environ.get("MATRIX_COMPUTE_DEVICE", "auto"))
    )
    normalized = str(choice).strip().lower() or "auto"
    aliases = {
        "gpu": "auto-gpu",
        "torch-mps": "mps",
        "torch-cuda": "cuda",
        "numpy": "cpu",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"auto", "auto-gpu", "cpu", "mps", "cuda"}:
        raise ValueError("MATRIX device must be auto, cpu, mps or cuda")
    minimum = (
        _positive_env_int("MATRIX_GPU_MIN_BATCH", DEFAULT_GPU_MIN_BATCH)
        if gpu_min_batch is None
        else int(gpu_min_batch)
    )
    if minimum < 1:
        raise ValueError("MATRIX GPU minimum batch size must be positive")

    if normalized == "cpu":
        return _with_request(_cpu_backend(), normalized)

    candidates = {
        "cuda": _torch_cuda_backend(),
        "mps": _torch_mps_backend(),
    }
    if normalized in candidates:
        selected = candidates[normalized]
        rejection = _backend_rejection(
            selected,
            allow_mixed_precision=allow_mixed_precision,
            require_float64=require_float64,
        )
        if not rejection:
            return _with_request(selected, normalized)
        if _truthy_env("MATRIX_DEVICE_STRICT", False):
            raise RuntimeError(f"requested MATRIX backend {normalized} is unavailable: {rejection}")
        return _with_request(_cpu_backend(reason=f"{normalized} fallback: {rejection}"), normalized)

    if normalized == "auto" and int(workload_size) < minimum:
        return _with_request(
            _cpu_backend(reason=f"GPU threshold not reached ({workload_size} < {minimum})"),
            normalized,
        )
    for name in ("cuda", "mps"):
        selected = candidates[name]
        rejection = _backend_rejection(
            selected,
            allow_mixed_precision=allow_mixed_precision,
            require_float64=require_float64,
        )
        if not rejection:
            return _with_request(selected, normalized)
    return _with_request(_cpu_backend(reason="no suitable GPU backend"), normalized)


def compute_backend_report() -> dict[str, Any]:
    """Describe the runtime policy and all detected backends."""

    return {
        "schema": "matrix.compute.report.v1",
        "policy": {
            "device": os.environ.get(
                "MATRIX_DEVICE", os.environ.get("MATRIX_COMPUTE_DEVICE", "auto")
            ),
            "gpu_min_batch": _positive_env_int(
                "MATRIX_GPU_MIN_BATCH", DEFAULT_GPU_MIN_BATCH
            ),
            "gpu_validation": _truthy_env("MATRIX_GPU_VALIDATE", True),
            "neural_engine": os.environ.get("MATRIX_NEURAL_ENGINE", "auto"),
            "neural_engine_validation": _truthy_env(
                "MATRIX_NEURAL_ENGINE_VALIDATE", True
            ),
            "cpu_reference_precision": "float64",
            "numerical_backend": os.environ.get(
                "MATRIX_NUMERICAL_BACKEND", "auto"
            ),
            "native_minimum_work": _positive_env_int(
                "MATRIX_NATIVE_MIN_WORK", DEFAULT_NATIVE_MIN_WORK
            ),
        },
        "backends": [backend.to_dict() for backend in available_compute_backends()],
    }


def gpu_validation_enabled() -> bool:
    return _truthy_env("MATRIX_GPU_VALIDATE", True)


def neural_engine_validation_enabled() -> bool:
    return _truthy_env("MATRIX_NEURAL_ENGINE_VALIDATE", True)


def resolve_neural_inference_backend(requested: str | None = None) -> ComputeBackend:
    """Select Core ML/ANE for neural inference, with an explicit CPU fallback.

    This selector is deliberately separate from :func:`resolve_compute_backend`:
    the Neural Engine is appropriate for converted neural networks, not for the
    float64 linear algebra used by Hessians, GF calculations or fitting.
    """

    choice = requested if requested is not None else os.environ.get("MATRIX_NEURAL_ENGINE", "auto")
    normalized = str(choice).strip().casefold() or "auto"
    aliases = {"on": "auto", "yes": "auto", "1": "auto", "no": "off", "0": "off"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"auto", "off", "required"}:
        raise ValueError("MATRIX neural-engine policy must be auto, off or required")
    if normalized == "off":
        return _with_request(_cpu_backend(reason="Neural Engine disabled by policy"), normalized)
    backend = _coreml_neural_engine_backend()
    if backend.available:
        return _with_request(backend, normalized)
    if normalized == "required":
        raise RuntimeError(f"Apple Neural Engine is unavailable: {backend.reason}")
    return _with_request(
        _cpu_backend(reason=f"Neural Engine fallback: {backend.reason}"), normalized
    )


def coreml_compute_unit(requested: str | None = None):
    """Return the Core ML compute-unit policy for validated neural inference."""

    backend = resolve_neural_inference_backend(requested)
    try:
        import coremltools as ct
    except Exception as exc:
        if backend.device == "ane":
            raise RuntimeError(f"Core ML import failed after ANE detection: {exc}") from exc
        return None
    return ct.ComputeUnit.ALL if backend.device == "ane" else ct.ComputeUnit.CPU_ONLY


def _backend_rejection(
    backend: ComputeBackend,
    *,
    allow_mixed_precision: bool,
    require_float64: bool,
) -> str:
    if not backend.available:
        return backend.reason or "backend is unavailable"
    if require_float64 and not backend.supports_float64:
        return "float64 is required"
    if not backend.supports_float64 and not allow_mixed_precision:
        return "mixed precision was not enabled"
    return ""


def _with_request(backend: ComputeBackend, requested: str) -> ComputeBackend:
    return ComputeBackend(
        name=backend.name,
        device=backend.device,
        precision=backend.precision,
        available=backend.available,
        accelerated=backend.accelerated,
        supports_float64=backend.supports_float64,
        core_count=backend.core_count,
        requested=requested,
        reason=backend.reason,
    )


def _cpu_backend(*, reason: str = "") -> ComputeBackend:
    return ComputeBackend(
        name="numpy",
        device="cpu",
        precision="float64",
        available=True,
        accelerated=False,
        supports_float64=True,
        reason=reason,
    )


@lru_cache(maxsize=1)
def _torch_cuda_backend() -> ComputeBackend:
    try:
        import torch

        available = bool(torch.cuda.is_available())
        return ComputeBackend(
            name="torch",
            device="cuda",
            precision="float64",
            available=available,
            accelerated=True,
            supports_float64=True,
            reason="" if available else "Torch reports no CUDA device",
        )
    except Exception as exc:
        return ComputeBackend(
            name="torch",
            device="cuda",
            precision="float64",
            available=False,
            accelerated=True,
            supports_float64=True,
            reason=f"Torch CUDA probe failed: {exc}",
        )


@lru_cache(maxsize=1)
def _torch_mps_backend() -> ComputeBackend:
    try:
        import torch

        available = bool(torch.backends.mps.is_built() and torch.backends.mps.is_available())
        return ComputeBackend(
            name="torch",
            device="mps",
            precision="float32",
            available=available,
            accelerated=True,
            supports_float64=False,
            reason="" if available else "Torch reports no usable MPS device",
        )
    except Exception as exc:
        return ComputeBackend(
            name="torch",
            device="mps",
            precision="float32",
            available=False,
            accelerated=True,
            supports_float64=False,
            reason=f"Torch MPS probe failed: {exc}",
        )


@lru_cache(maxsize=1)
def _coreml_neural_engine_backend() -> ComputeBackend:
    try:
        import coremltools as ct
        from coremltools.models.compute_device import MLNeuralEngineComputeDevice

        devices = ct.models.MLModel.get_available_compute_devices()
        neural_devices = tuple(
            device
            for device in devices
            if isinstance(device, MLNeuralEngineComputeDevice)
        )
        core_count = sum(
            int(
                getattr(
                    device,
                    "total_core_count",
                    getattr(device, "core_count", 0),
                )
                or 0
            )
            for device in neural_devices
        )
        available = bool(neural_devices)
        return ComputeBackend(
            name="coreml",
            device="ane",
            precision="float16",
            available=available,
            accelerated=True,
            supports_float64=False,
            core_count=core_count,
            reason="" if available else "Core ML reports no Apple Neural Engine",
        )
    except Exception as exc:
        return ComputeBackend(
            name="coreml",
            device="ane",
            precision="float16",
            available=False,
            accelerated=True,
            supports_float64=False,
            reason=f"Core ML Neural Engine probe failed: {exc}",
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
    return value.strip().lower() not in {"0", "false", "no", "off", ""}
