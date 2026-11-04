from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from typing import Any

import numpy as np


DEFAULT_GPU_MIN_SIZE = 128
DEFAULT_NUMPY_MAX_SIZE = 96
DEFAULT_CPU_EVD_MAX_SIZE = 1024
GPU_BACKENDS = ("cupy", "torch-cuda", "torch-mps")
CPU_BACKENDS = ("scipy", "numpy")


@dataclass(frozen=True)
class DiagonalizerBackend:
    name: str
    available: bool
    accelerated: bool = False
    device: str = "cpu"
    reason: str = ""


@dataclass(frozen=True)
class EighResult:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray | None
    backend: str
    device: str
    accelerated: bool


@lru_cache(maxsize=1)
def available_diagonalizer_backends() -> tuple[DiagonalizerBackend, ...]:
    return tuple(_backend_by_name(name) for name in (*GPU_BACKENDS, *CPU_BACKENDS))


def best_diagonalizer_backend(matrix_shape: tuple[int, ...] | None = None) -> DiagonalizerBackend:
    requested = _env_value("MATRIX_DIAGONALIZER_BACKEND", "ORACLE_DIAGONALIZER_BACKEND", "auto")
    requested = requested.strip().lower()
    min_gpu_size = _env_int(
        "MATRIX_DIAGONALIZER_GPU_MIN_SIZE",
        DEFAULT_GPU_MIN_SIZE,
        fallback_name="ORACLE_DIAGONALIZER_GPU_MIN_SIZE",
    )
    if requested and requested != "auto":
        return _requested_backend(requested)
    n = int(matrix_shape[0]) if matrix_shape and matrix_shape else 0
    if n >= min_gpu_size:
        for name in GPU_BACKENDS:
            backend = _backend_by_name(name)
            if backend.available:
                return backend
    numpy_max_size = _env_int(
        "MATRIX_DIAGONALIZER_NUMPY_MAX_SIZE",
        DEFAULT_NUMPY_MAX_SIZE,
        fallback_name="ORACLE_DIAGONALIZER_NUMPY_MAX_SIZE",
    )
    if 0 < n <= numpy_max_size:
        # NumPy and SciPy call the same platform LAPACK at this scale.  NumPy
        # avoids importing the much larger SciPy stack on the first molecular
        # point, with indistinguishable steady-state kernel timings.
        return _backend_by_name("numpy")
    for name in CPU_BACKENDS:
        backend = _backend_by_name(name)
        if backend.available:
            return backend
    return DiagonalizerBackend("numpy", True, accelerated=False, device="cpu")


def diagonalize_hermitian(
    matrix: Any,
    *,
    backend: str | None = None,
    eigvals_only: bool = False,
    subset_by_index: tuple[int, int] | None = None,
    lower: bool = True,
    check_finite: bool = False,
) -> EighResult:
    array = np.asarray(matrix)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"eigh requires a square matrix; got shape {array.shape}")
    selected = _requested_backend(backend) if backend else best_diagonalizer_backend(array.shape)
    if selected.accelerated:
        return _diagonalize_gpu(
            array,
            backend=selected,
            eigvals_only=eigvals_only,
            subset_by_index=subset_by_index,
            lower=lower,
        )
    if selected.name == "scipy" and selected.available:
        return _diagonalize_scipy(
            array,
            eigvals_only=eigvals_only,
            subset_by_index=subset_by_index,
            lower=lower,
            check_finite=check_finite,
        )
    if selected.name == "numpy" and selected.available:
        return _diagonalize_numpy(
            array,
            eigvals_only=eigvals_only,
            subset_by_index=subset_by_index,
        )
    raise RuntimeError(f"requested diagonalizer backend is not available: {selected.name}")


def eigh_arrays(
    matrix: Any,
    *,
    backend: str | None = None,
    subset_by_index: tuple[int, int] | None = None,
    lower: bool = True,
    check_finite: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    result = diagonalize_hermitian(
        matrix,
        backend=backend,
        eigvals_only=False,
        subset_by_index=subset_by_index,
        lower=lower,
        check_finite=check_finite,
    )
    if result.eigenvectors is None:
        raise RuntimeError("internal diagonalizer error: eigenvectors were not returned")
    return result.eigenvalues, result.eigenvectors


def eigvalsh_array(
    matrix: Any,
    *,
    backend: str | None = None,
    subset_by_index: tuple[int, int] | None = None,
    lower: bool = True,
    check_finite: bool = False,
) -> np.ndarray:
    result = diagonalize_hermitian(
        matrix,
        backend=backend,
        eigvals_only=True,
        subset_by_index=subset_by_index,
        lower=lower,
        check_finite=check_finite,
    )
    return result.eigenvalues


def _diagonalize_scipy(
    array: np.ndarray,
    *,
    eigvals_only: bool,
    subset_by_index: tuple[int, int] | None,
    lower: bool,
    check_finite: bool,
) -> EighResult:
    from scipy import linalg as scipy_linalg

    kwargs: dict[str, Any] = {
        "lower": lower,
        "eigvals_only": eigvals_only,
        "check_finite": check_finite,
    }
    if subset_by_index is not None:
        kwargs["subset_by_index"] = subset_by_index
        kwargs["driver"] = "evr"
    elif array.shape[0] <= _env_int(
        "MATRIX_DIAGONALIZER_CPU_EVD_MAX_SIZE",
        DEFAULT_CPU_EVD_MAX_SIZE,
        fallback_name="ORACLE_DIAGONALIZER_CPU_EVD_MAX_SIZE",
    ):
        # The complete eigenbasis is required by finite-temperature SCC.  On
        # CPU LAPACK, divide-and-conquer has materially lower overhead than
        # the MRRR default for the small/medium matrices used here.
        kwargs["driver"] = "evd"
    out = scipy_linalg.eigh(array, **kwargs)
    if eigvals_only:
        return EighResult(np.asarray(out), None, "scipy", "cpu", False)
    values, vectors = out
    return EighResult(np.asarray(values), np.asarray(vectors), "scipy", "cpu", False)


def _diagonalize_numpy(
    array: np.ndarray,
    *,
    eigvals_only: bool,
    subset_by_index: tuple[int, int] | None,
) -> EighResult:
    if eigvals_only:
        values = np.linalg.eigvalsh(array)
        vectors = None
    else:
        values, vectors = np.linalg.eigh(array)
    if subset_by_index is not None:
        start, stop = subset_by_index
        values = values[start : stop + 1]
        if vectors is not None:
            vectors = vectors[:, start : stop + 1]
    return EighResult(
        np.asarray(values),
        None if vectors is None else np.asarray(vectors),
        "numpy",
        "cpu",
        False,
    )


def _diagonalize_gpu(
    array: np.ndarray,
    *,
    backend: DiagonalizerBackend,
    eigvals_only: bool,
    subset_by_index: tuple[int, int] | None,
    lower: bool,
) -> EighResult:
    try:
        if backend.name == "cupy":
            return _diagonalize_cupy(
                array,
                eigvals_only=eigvals_only,
                subset_by_index=subset_by_index,
            )
        if backend.name in {"torch-cuda", "torch-mps"}:
            return _diagonalize_torch(
                array,
                backend=backend,
                eigvals_only=eigvals_only,
                subset_by_index=subset_by_index,
                lower=lower,
            )
    except Exception:
        if (
            _env_value("MATRIX_DIAGONALIZER_STRICT_GPU", "ORACLE_DIAGONALIZER_STRICT_GPU", "0")
            == "1"
        ):
            raise
    return _diagonalize_scipy(
        array,
        eigvals_only=eigvals_only,
        subset_by_index=subset_by_index,
        lower=lower,
        check_finite=False,
    )


def _diagonalize_cupy(
    array: np.ndarray,
    *,
    eigvals_only: bool,
    subset_by_index: tuple[int, int] | None,
) -> EighResult:
    import cupy as cp

    gpu_array = cp.asarray(array)
    if eigvals_only:
        values_gpu = cp.linalg.eigvalsh(gpu_array)
        vectors_gpu = None
    else:
        values_gpu, vectors_gpu = cp.linalg.eigh(gpu_array)
    values = cp.asnumpy(values_gpu)
    vectors = None if vectors_gpu is None else cp.asnumpy(vectors_gpu)
    if subset_by_index is not None:
        start, stop = subset_by_index
        values = values[start : stop + 1]
        if vectors is not None:
            vectors = vectors[:, start : stop + 1]
    return EighResult(values, vectors, "cupy", "cuda", True)


def _diagonalize_torch(
    array: np.ndarray,
    *,
    backend: DiagonalizerBackend,
    eigvals_only: bool,
    subset_by_index: tuple[int, int] | None,
    lower: bool,
) -> EighResult:
    import torch

    tensor = torch.as_tensor(array, device=backend.device)
    uplo = "L" if lower else "U"
    if eigvals_only:
        values_tensor = torch.linalg.eigvalsh(tensor, UPLO=uplo)
        vectors_tensor = None
    else:
        values_tensor, vectors_tensor = torch.linalg.eigh(tensor, UPLO=uplo)
    values = values_tensor.detach().cpu().numpy()
    vectors = None if vectors_tensor is None else vectors_tensor.detach().cpu().numpy()
    if subset_by_index is not None:
        start, stop = subset_by_index
        values = values[start : stop + 1]
        if vectors is not None:
            vectors = vectors[:, start : stop + 1]
    return EighResult(values, vectors, backend.name, backend.device, True)


def _requested_backend(name: str | None) -> DiagonalizerBackend:
    if name is None:
        return best_diagonalizer_backend(None)
    normalized = name.strip().lower()
    if not normalized or normalized == "auto":
        return best_diagonalizer_backend(None)
    aliases = {"gpu": GPU_BACKENDS, "cpu": CPU_BACKENDS}
    if normalized in aliases:
        for candidate in aliases[normalized]:
            backend = _backend_by_name(candidate)
            if backend.available:
                return backend
        return DiagonalizerBackend(normalized, False, reason="no matching backend available")
    return _backend_by_name(normalized)


@lru_cache(maxsize=None)
def _backend_by_name(name: str) -> DiagonalizerBackend:
    if name == "cupy":
        return _cupy_backend()
    if name == "torch-cuda":
        return _torch_cuda_backend()
    if name == "torch-mps":
        return _torch_mps_backend()
    if name == "scipy":
        return _scipy_backend()
    if name == "numpy":
        return DiagonalizerBackend("numpy", True, accelerated=False, device="cpu")
    return DiagonalizerBackend(name, False, reason="unknown backend")


def _cupy_backend() -> DiagonalizerBackend:
    try:
        import cupy as cp

        count = int(cp.cuda.runtime.getDeviceCount())
        if count > 0:
            return DiagonalizerBackend("cupy", True, accelerated=True, device="cuda")
        return DiagonalizerBackend(
            "cupy",
            False,
            accelerated=True,
            device="cuda",
            reason="no CUDA device",
        )
    except Exception as exc:
        return DiagonalizerBackend("cupy", False, accelerated=True, device="cuda", reason=str(exc))


def _torch_cuda_backend() -> DiagonalizerBackend:
    try:
        import torch

        if bool(torch.cuda.is_available()):
            return DiagonalizerBackend("torch-cuda", True, accelerated=True, device="cuda")
        return DiagonalizerBackend(
            "torch-cuda",
            False,
            accelerated=True,
            device="cuda",
            reason="no CUDA device",
        )
    except Exception as exc:
        return DiagonalizerBackend(
            "torch-cuda",
            False,
            accelerated=True,
            device="cuda",
            reason=str(exc),
        )


def _torch_mps_backend() -> DiagonalizerBackend:
    try:
        import torch

        mps = getattr(torch.backends, "mps", None)
        if mps is not None and bool(mps.is_available()):
            return DiagonalizerBackend(
                "torch-mps",
                False,
                accelerated=True,
                device="mps",
                reason="MPS lacks float64 required by the scientific diagonalizer",
            )
        return DiagonalizerBackend(
            "torch-mps",
            False,
            accelerated=True,
            device="mps",
            reason="no MPS device",
        )
    except Exception as exc:
        return DiagonalizerBackend(
            "torch-mps",
            False,
            accelerated=True,
            device="mps",
            reason=str(exc),
        )


def _scipy_backend() -> DiagonalizerBackend:
    try:
        import scipy.linalg  # noqa: F401

        return DiagonalizerBackend("scipy", True, accelerated=False, device="cpu")
    except Exception as exc:
        return DiagonalizerBackend("scipy", False, accelerated=False, device="cpu", reason=str(exc))


def _env_value(name: str, fallback_name: str, default: str) -> str:
    return os.environ.get(name) or os.environ.get(fallback_name, default)


def _env_int(name: str, default: int, *, fallback_name: str | None = None) -> int:
    try:
        raw = os.environ.get(name)
        if raw is None and fallback_name is not None:
            raw = os.environ.get(fallback_name)
        return int(str(default) if raw is None else raw)
    except ValueError:
        return default
