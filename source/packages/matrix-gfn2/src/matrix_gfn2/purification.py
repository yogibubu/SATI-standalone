"""Trace-correcting SP2 density-matrix purification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from . import _gfn2_native
except ImportError:  # pragma: no cover
    _gfn2_native = None

from .locality import LocalizedOrbitalDomains, truncate_symmetric_matrix
from .localized_solver import SparseSymmetricMatrix


@dataclass(frozen=True)
class PurificationResult:
    projector: np.ndarray
    iterations: int
    converged: bool
    used_dense_fallback: bool
    accumulated_truncation_bound: float
    idempotency_error: float
    product_backend: str = "numpy"
    mixed_precision_iterations: int = 0
    fp64_iterations: int = 0
    device_transfers: int = 0
    peak_device_bytes: int = 0


@dataclass(frozen=True)
class SparsePurificationResult:
    projector: SparseSymmetricMatrix
    iterations: int
    converged: bool
    accumulated_truncation_bound: float
    idempotency_error: float
    history: np.ndarray | None = None
    branches: np.ndarray | None = None
    stationarity_error: float = 0.0


@dataclass(frozen=True)
class SparseProjectorState:
    """Certified SP2 path reusable by density-matrix perturbation theory."""

    lower_bound_hartree: float
    upper_bound_hartree: float
    occupied_orbitals: int
    history: np.ndarray
    branches: np.ndarray


def _assemble_mixed_square(
    diagonal: np.ndarray,
    off_diagonal: np.ndarray,
    off_square: np.ndarray,
    *,
    block_rows: int = 256,
) -> np.ndarray:
    """Add diagonal cross terms in bounded FP64 workspace."""

    result = np.asarray(off_square, dtype=np.float64)
    size = len(diagonal)
    if block_rows < 1:
        raise ValueError("mixed-square block size must be positive")
    for start in range(0, size, block_rows):
        stop = min(start + block_rows, size)
        cross = off_diagonal[start:stop].astype(np.float64)
        cross *= diagonal[start:stop, None] + diagonal[None, :]
        result[start:stop] += cross
    np.fill_diagonal(result, diagonal * diagonal + np.diag(off_square))
    return 0.5 * (result + result.T)


def _mixed_precision_square_numpy(matrix: np.ndarray) -> np.ndarray:
    """Square a symmetric matrix while retaining its dominant diagonal in FP64."""

    diagonal = np.diag(matrix).astype(np.float64, copy=True)
    off_diagonal = np.asarray(matrix, dtype=np.float32).copy()
    np.fill_diagonal(off_diagonal, 0.0)
    off_square = np.asarray(off_diagonal @ off_diagonal, dtype=np.float64)
    return _assemble_mixed_square(diagonal, off_diagonal, off_square)


def _mixed_precision_square_cupy(matrix: np.ndarray) -> np.ndarray:
    import cupy as cp

    values = cp.asarray(matrix, dtype=cp.float64)
    diagonal = cp.diag(values)
    off_diagonal = values.astype(cp.float32)
    cp.fill_diagonal(off_diagonal, 0.0)
    off_square = (off_diagonal @ off_diagonal).astype(cp.float64)
    result = off_square + (diagonal[:, None] + diagonal[None, :]) * off_diagonal
    cp.fill_diagonal(result, diagonal * diagonal + cp.diag(off_square))
    return cp.asnumpy(0.5 * (result + result.T))


def _mixed_precision_square_torch_device(
    matrix: np.ndarray,
    device_name: str,
) -> np.ndarray:
    import torch

    device = torch.device(device_name)
    diagonal = np.diag(matrix).astype(np.float64, copy=True)
    off_diagonal_cpu = np.asarray(matrix, dtype=np.float32).copy()
    np.fill_diagonal(off_diagonal_cpu, 0.0)
    off_diagonal = torch.as_tensor(off_diagonal_cpu, device=device)
    off_square = (off_diagonal @ off_diagonal).cpu().numpy().astype(np.float64)
    return _assemble_mixed_square(diagonal, off_diagonal_cpu, off_square)


def _mixed_precision_square_torch_cuda(matrix: np.ndarray) -> np.ndarray:
    return _mixed_precision_square_torch_device(matrix, "cuda")


def _mixed_precision_square_torch_mps(matrix: np.ndarray) -> np.ndarray:
    return _mixed_precision_square_torch_device(matrix, "mps")


def mixed_precision_square(
    matrix: np.ndarray,
    *,
    backend: str = "numpy",
) -> np.ndarray:
    """Mixed FP64-diagonal/FP32-off-diagonal symmetric product.

    The layout follows the stable consumer-GPU strategy described by
    Steinbach and Bannwarth: the numerically dominant diagonal and the final
    accumulation remain double precision, while the cubic off-diagonal GEMM
    is evaluated in single precision.
    """

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("mixed-precision square requires a square matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError("mixed-precision square requires finite values")
    if not np.allclose(values, values.T, rtol=0.0, atol=1.0e-13):
        raise ValueError("mixed-precision square requires a symmetric matrix")
    normalized = str(backend).strip().lower()
    implementations: dict[str, Any] = {
        "numpy": _mixed_precision_square_numpy,
        "cpu": _mixed_precision_square_numpy,
        "cupy": _mixed_precision_square_cupy,
        "torch-cuda": _mixed_precision_square_torch_cuda,
        "cuda": _mixed_precision_square_torch_cuda,
        "torch-mps": _mixed_precision_square_torch_mps,
        "mps": _mixed_precision_square_torch_mps,
    }
    if normalized not in implementations:
        raise ValueError(f"unsupported SP2 product backend: {backend}")
    return implementations[normalized](values)


def _spectral_bounds_gershgorin(matrix: np.ndarray) -> tuple[float, float]:
    diagonal = np.diag(matrix)
    radius = np.sum(np.abs(matrix), axis=1) - np.abs(diagonal)
    return float(np.min(diagonal - radius)), float(np.max(diagonal + radius))


def _exact_projector(fock: np.ndarray, occupied_orbitals: int) -> np.ndarray:
    _, coefficients = np.linalg.eigh(fock)
    occupied = coefficients[:, :occupied_orbitals]
    return occupied @ occupied.T


def _resident_device_sp2(
    projector: np.ndarray,
    occupied_orbitals: int,
    *,
    tolerance: float,
    max_iterations: int,
    backend: str,
    verification_interval: int,
) -> PurificationResult:
    """Run the mixed phase without round-tripping the projector each step."""

    normalized = str(backend).strip().lower()
    size = len(projector)
    transfers = 1
    peak_bytes = 2 * size * size * np.dtype(np.float32).itemsize
    verification_trigger = max(
        100.0 * tolerance,
        16.0 * float(np.finfo(np.float32).eps) * size,
    )
    if normalized == "cupy":
        import cupy as cp

        device_projector = cp.asarray(projector, dtype=cp.float32)

        def square_and_metrics():
            square = device_projector @ device_projector
            trace = float(cp.trace(device_projector).item())
            error = float(cp.linalg.norm(square - device_projector).item())
            return square, trace, error

        def update(square, use_square: bool) -> None:
            nonlocal device_projector
            candidate = square if use_square else 2.0 * device_projector - square
            device_projector = 0.5 * (candidate + candidate.T)

        def snapshot() -> np.ndarray:
            return cp.asnumpy(device_projector).astype(np.float64)

    elif normalized in {"torch-cuda", "cuda", "torch-mps", "mps"}:
        import torch

        device_name = "mps" if normalized in {"torch-mps", "mps"} else "cuda"
        device_projector = torch.as_tensor(
            projector, dtype=torch.float32, device=torch.device(device_name)
        )

        def square_and_metrics():
            square = device_projector @ device_projector
            trace = float(torch.trace(device_projector).item())
            error = float(torch.linalg.vector_norm(square - device_projector).item())
            return square, trace, error

        def update(square, use_square: bool) -> None:
            nonlocal device_projector
            candidate = square if use_square else 2.0 * device_projector - square
            device_projector = 0.5 * (candidate + candidate.T)

        def snapshot() -> np.ndarray:
            return device_projector.cpu().numpy().astype(np.float64)

    else:
        raise ValueError(f"unsupported resident SP2 backend: {backend}")

    mixed_iterations = 0
    host_projector = np.asarray(projector, dtype=np.float64)
    for iteration in range(1, max_iterations + 1):
        square, trace, approximate_error = square_and_metrics()
        mixed_iterations = iteration
        verify = (
            approximate_error <= verification_trigger
            or iteration % verification_interval == 0
        )
        if verify:
            host_projector = snapshot()
            transfers += 1
            exact_square = host_projector @ host_projector
            exact_error = float(np.linalg.norm(exact_square - host_projector))
            trace_error = abs(float(np.trace(host_projector)) - occupied_orbitals)
            if exact_error <= tolerance and trace_error <= tolerance:
                return PurificationResult(
                    host_projector,
                    iteration - 1,
                    True,
                    False,
                    0.0,
                    exact_error,
                    normalized,
                    mixed_iterations,
                    0,
                    transfers,
                    peak_bytes,
                )
            if exact_error <= verification_trigger:
                break
        update(square, trace > occupied_orbitals)
    else:
        host_projector = snapshot()
        transfers += 1

    fp64_iterations = 0
    for iteration in range(mixed_iterations + 1, max_iterations + 1):
        square = host_projector @ host_projector
        error = float(np.linalg.norm(square - host_projector))
        trace_error = abs(float(np.trace(host_projector)) - occupied_orbitals)
        if error <= tolerance and trace_error <= tolerance:
            return PurificationResult(
                host_projector,
                iteration - 1,
                True,
                False,
                0.0,
                error,
                normalized,
                mixed_iterations,
                fp64_iterations,
                transfers,
                peak_bytes,
            )
        host_projector = (
            square
            if float(np.trace(host_projector)) > occupied_orbitals
            else 2.0 * host_projector - square
        )
        host_projector = 0.5 * (host_projector + host_projector.T)
        fp64_iterations += 1
    square = host_projector @ host_projector
    error = float(np.linalg.norm(square - host_projector))
    trace_error = abs(float(np.trace(host_projector)) - occupied_orbitals)
    return PurificationResult(
        host_projector,
        mixed_iterations + fp64_iterations,
        error <= tolerance and trace_error <= tolerance,
        False,
        0.0,
        error,
        normalized,
        mixed_iterations,
        fp64_iterations,
        transfers,
        peak_bytes,
    )


def sp2_purify(
    fock_orthogonal: np.ndarray,
    occupied_orbitals: int,
    *,
    tolerance: float = 1.0e-9,
    max_iterations: int = 100,
    mixed_precision: bool = True,
    estimated_gap_hartree: float | None = None,
    minimum_gap_hartree: float = 1.0e-3,
    domains: LocalizedOrbitalDomains | None = None,
    truncation_threshold: float = 1.0e-10,
    product_backend: str = "numpy",
    verification_interval: int = 4,
) -> PurificationResult:
    """Build an occupied projector without diagonalization in the gapped regime.

    Products may be evaluated in FP32, but accumulation, trace decisions, error
    measures, and the returned projector remain FP64.  Small-gap systems are
    routed to the robust dense diagonalization fallback instead of silently
    applying locality assumptions outside their validity domain.
    """

    fock = np.asarray(fock_orthogonal, dtype=float)
    if fock.ndim != 2 or fock.shape[0] != fock.shape[1]:
        raise ValueError("orthogonal Fock matrix must be square")
    if not np.all(np.isfinite(fock)):
        raise ValueError("orthogonal Fock matrix must be finite")
    if not np.allclose(fock, fock.T, rtol=0.0, atol=1.0e-13):
        raise ValueError("SP2 requires a symmetric orthogonal Fock matrix")
    size = len(fock)
    if not 0 <= occupied_orbitals <= size:
        raise ValueError("occupied_orbitals lies outside the matrix dimension")
    if tolerance <= 0.0 or not np.isfinite(tolerance):
        raise ValueError("SP2 tolerance must be positive and finite")
    if max_iterations < 1:
        raise ValueError("SP2 maximum iterations must be positive")
    if truncation_threshold < 0.0 or not np.isfinite(truncation_threshold):
        raise ValueError("SP2 truncation threshold must be nonnegative and finite")
    if verification_interval < 1:
        raise ValueError("SP2 verification interval must be positive")
    if estimated_gap_hartree is not None and not np.isfinite(estimated_gap_hartree):
        raise ValueError("estimated gap must be finite")
    if minimum_gap_hartree < 0.0 or not np.isfinite(minimum_gap_hartree):
        raise ValueError("minimum gap must be nonnegative and finite")
    if estimated_gap_hartree is not None and estimated_gap_hartree < minimum_gap_hartree:
        projector = _exact_projector(fock, occupied_orbitals)
        error = float(np.linalg.norm(projector @ projector - projector))
        return PurificationResult(projector, 0, True, True, 0.0, error)
    if occupied_orbitals in (0, size):
        projector = np.eye(size) if occupied_orbitals == size else np.zeros_like(fock)
        return PurificationResult(projector, 0, True, False, 0.0, 0.0)

    lower, upper = _spectral_bounds_gershgorin(fock)
    if upper - lower <= np.finfo(float).eps:
        raise ValueError("SP2 cannot select an occupied subspace from a degenerate spectrum")
    identity = np.eye(size)
    projector = (upper * identity - fock) / (upper - lower)
    normalized_backend = str(product_backend).strip().lower()
    if (
        mixed_precision
        and domains is None
        and normalized_backend
        in {"cupy", "torch-cuda", "cuda", "torch-mps", "mps"}
    ):
        return _resident_device_sp2(
            projector,
            occupied_orbitals,
            tolerance=tolerance,
            max_iterations=max_iterations,
            backend=normalized_backend,
            verification_interval=verification_interval,
        )
    truncation_bound = 0.0
    error = float("inf")
    updates = 0
    mixed_active = mixed_precision
    for iteration in range(1, max_iterations + 1):
        square = (
            mixed_precision_square(projector, backend=product_backend)
            if mixed_active
            else projector @ projector
        )
        trace_error = abs(float(np.trace(projector)) - occupied_orbitals)
        approximate_error = float(np.linalg.norm(square - projector))
        verification_trigger = max(
            100.0 * tolerance,
            16.0 * float(np.finfo(np.float32).eps) * size,
        )
        verify = not mixed_active or (
            approximate_error <= verification_trigger
            or (updates > 0 and updates % verification_interval == 0)
        )
        if verify:
            exact_square = projector @ projector if mixed_active else square
            error = float(np.linalg.norm(exact_square - projector))
            if error <= tolerance and trace_error <= tolerance:
                return PurificationResult(
                    projector, updates, True, False, truncation_bound, error
                )
            if mixed_active and error <= verification_trigger:
                square = exact_square
                mixed_active = False
        if float(np.trace(projector)) > occupied_orbitals:
            candidate = square
        else:
            candidate = 2.0 * projector - square
        candidate = 0.5 * (candidate + candidate.T)
        if domains is not None:
            candidate, discarded = truncate_symmetric_matrix(
                candidate, domains, threshold=truncation_threshold
            )
            truncation_bound += discarded
        if not np.all(np.isfinite(candidate)):
            return PurificationResult(
                projector, updates, False, False, truncation_bound, float("inf")
            )
        projector = candidate
        updates = iteration
    exact_square = projector @ projector
    error = float(np.linalg.norm(exact_square - projector))
    trace_error = abs(float(np.trace(projector)) - occupied_orbitals)
    return PurificationResult(
        projector,
        updates,
        error <= tolerance and trace_error <= tolerance,
        False,
        truncation_bound,
        error,
    )


def sp2_purify_sparse(
    fock_orthogonal: SparseSymmetricMatrix,
    occupied_orbitals: int,
    domains: LocalizedOrbitalDomains,
    *,
    lower_bound_hartree: float,
    upper_bound_hartree: float,
    tolerance: float = 1.0e-9,
    max_iterations: int = 100,
    truncation_threshold: float = 1.0e-12,
) -> SparsePurificationResult:
    """Trace-correcting SP2 using only domain-restricted sparse products."""

    size = fock_orthogonal.size
    if size != domains.norbitals:
        raise ValueError("sparse Fock matrix and domains differ in size")
    if not 0 <= occupied_orbitals <= size:
        raise ValueError("occupied_orbitals lies outside the matrix dimension")
    if tolerance <= 0.0 or not np.isfinite(tolerance):
        raise ValueError("sparse SP2 tolerance must be positive and finite")
    if max_iterations < 1:
        raise ValueError("sparse SP2 maximum iterations must be positive")
    if truncation_threshold < 0.0 or not np.isfinite(truncation_threshold):
        raise ValueError(
            "sparse SP2 truncation threshold must be nonnegative and finite"
        )
    width = float(upper_bound_hartree) - float(lower_bound_hartree)
    if not np.isfinite(width) or width <= np.finfo(float).eps:
        raise ValueError("invalid spectral bounds for sparse SP2")
    if _gfn2_native is not None:
        allowed_left, allowed_right = domains.allowed_pair_arrays()
        topology = domains.native_sp2_topology()
        topology_args = (None,) * 6 if topology is None else tuple(topology)
        if len(topology_args) == 3:
            topology_args += (None, None, None)
        rows, columns, values, iterations, converged, bound, error, history, branches = (
            _gfn2_native.sparse_sp2(
                int(size),
                np.asarray(fock_orthogonal.rows, dtype=np.int32),
                np.asarray(fock_orthogonal.columns, dtype=np.int32),
                np.asarray(fock_orthogonal.values, dtype=float),
                int(occupied_orbitals),
                allowed_left,
                allowed_right,
                float(lower_bound_hartree),
                float(upper_bound_hartree),
                float(tolerance),
                int(max_iterations),
                float(truncation_threshold),
                *topology_args,
                int(domains.native_threads),
            )
        )
        history_array = np.asarray(history, dtype=float)
        error_value = float(error)
        if not np.isfinite(error_value):
            finite_rows = np.all(np.isfinite(history_array), axis=1)
            first_bad = int(np.flatnonzero(~finite_rows)[0]) if np.any(~finite_rows) else None
            finite_history = history_array[finite_rows]
            largest_value = (
                float(np.max(np.abs(finite_history), initial=0.0))
                if finite_history.size else float("nan")
            )
            diagonal_mask = allowed_left == allowed_right
            trace_at_end = float(np.sum(history_array[-1, diagonal_mask]))
            recent = []
            end = (
                min(len(history_array), first_bad + 1)
                if first_bad is not None else len(history_array)
            )
            start = max(0, end - 8)
            for step in range(start, end):
                step_values = history_array[step]
                step_trace = float(np.sum(step_values[diagonal_mask]))
                step_maximum = float(
                    np.max(np.abs(step_values), initial=0.0)
                )
                branch = (
                    int(np.asarray(branches, dtype=np.int8)[step - 1])
                    if 0 < step <= len(branches) else 0
                )
                recent.append(
                    f"{step}:tr={step_trace:.4g},max={step_maximum:.4g},"
                    f"branch={branch}"
                )
            detail = (
                f"first non-finite projector at update {first_bad}"
                if first_bad is not None
                else "projector history is finite but the residual overflowed"
            )
            raise FloatingPointError(
                "sparse SP2 became non-finite: "
                f"{detail}; updates={int(iterations)}, "
                f"trace={trace_at_end:.8g}, max_finite_projector="
                f"{largest_value:.8g}, spectral_bounds=({lower_bound_hartree:.8g}, "
                f"{upper_bound_hartree:.8g}), recent_steps=["
                + "; ".join(recent) + "]"
            )
        return SparsePurificationResult(
            SparseSymmetricMatrix(
                size,
                np.asarray(rows, dtype=np.intp),
                np.asarray(columns, dtype=np.intp),
                np.asarray(values, dtype=float),
            ),
            int(iterations),
            bool(converged),
            float(bound),
            error_value,
            history_array,
            np.asarray(branches, dtype=np.int8),
        )
    identity = SparseSymmetricMatrix(
        size,
        np.arange(size, dtype=np.intp),
        np.arange(size, dtype=np.intp),
        np.ones(size),
    )
    projector, discarded = identity.scaled_add(
        fock_orthogonal,
        self_scale=float(upper_bound_hartree) / width,
        other_scale=-1.0 / width,
        threshold=truncation_threshold,
    )
    bound = discarded
    error = float("inf")
    extensive_tolerance = tolerance * (1.0 if size < 128 else np.sqrt(size))
    trace_tolerance = tolerance * (1.0 if size < 128 else size)
    updates = 0
    for iteration in range(1, max_iterations + 1):
        square, discarded = projector.square(
            domains, threshold=truncation_threshold
        )
        bound += discarded
        difference, _ = square.scaled_add(projector, other_scale=-1.0)
        error = float(np.linalg.norm(difference.values)) + discarded
        trace_error = abs(projector.trace() - occupied_orbitals)
        if error <= extensive_tolerance and trace_error <= trace_tolerance:
            return SparsePurificationResult(
                projector, updates, True, bound, error
            )
        if projector.trace() > occupied_orbitals:
            candidate = square
        else:
            candidate, discarded = projector.scaled_add(
                square,
                self_scale=2.0,
                other_scale=-1.0,
                threshold=truncation_threshold,
            )
            bound += discarded
        projector = candidate
        updates = iteration
    square, _ = projector.square(domains, threshold=0.0)
    difference, _ = square.scaled_add(projector, other_scale=-1.0)
    error = float(np.linalg.norm(difference.values))
    trace_error = abs(projector.trace() - occupied_orbitals)
    return SparsePurificationResult(
        projector,
        updates,
        error <= extensive_tolerance and trace_error <= trace_tolerance,
        bound,
        error,
    )


def sp2_purify_sparse_incremental(
    fock_orthogonal: SparseSymmetricMatrix,
    occupied_orbitals: int,
    domains: LocalizedOrbitalDomains,
    state: SparseProjectorState,
    *,
    tolerance: float = 1.0e-9,
    truncation_threshold: float = 1.0e-12,
) -> SparsePurificationResult:
    """Propagate an SP2 projector along a previously certified polynomial path.

    This is the exact density-matrix perturbation recurrence of Niklasson and
    Challacombe for a fixed SP2 branch sequence.  Canonical trace and
    idempotency certification are deliberately fail-closed: callers rebuild a
    complete path whenever the chemical potential crosses a branch boundary.
    """

    if _gfn2_native is None or not hasattr(_gfn2_native, "sparse_sp2_incremental"):
        return SparsePurificationResult(
            state_to_projector(state, domains), 0, False, 0.0, float("inf")
        )
    if state.occupied_orbitals != int(occupied_orbitals):
        return SparsePurificationResult(
            state_to_projector(state, domains), 0, False, 0.0, float("inf")
        )
    allowed_left, allowed_right = domains.allowed_pair_arrays()
    rows, columns, values, converged, bound, error, stationarity, history = (
        _gfn2_native.sparse_sp2_incremental(
            int(fock_orthogonal.size),
            np.asarray(fock_orthogonal.rows, dtype=np.int32),
            np.asarray(fock_orthogonal.columns, dtype=np.int32),
            np.asarray(fock_orthogonal.values, dtype=float),
            int(occupied_orbitals),
            allowed_left,
            allowed_right,
            float(state.lower_bound_hartree),
            float(state.upper_bound_hartree),
            float(tolerance),
            float(truncation_threshold),
            np.asarray(state.history, dtype=float),
            np.asarray(state.branches, dtype=np.int8),
        )
    )
    return SparsePurificationResult(
        SparseSymmetricMatrix(
            fock_orthogonal.size,
            np.asarray(rows, dtype=np.intp),
            np.asarray(columns, dtype=np.intp),
            np.asarray(values, dtype=float),
        ),
        int(len(state.branches)),
        bool(converged),
        float(bound),
        float(error),
        np.asarray(history, dtype=float),
        np.asarray(state.branches, dtype=np.int8).copy(),
        float(stationarity),
    )


def state_to_projector(
    state: SparseProjectorState, domains: LocalizedOrbitalDomains
) -> SparseSymmetricMatrix:
    """Materialize the final projector from a fixed-pattern SP2 history."""

    left, right = domains.allowed_pair_arrays()
    values = np.asarray(state.history[-1], dtype=float)
    keep = (left == right) | (values != 0.0)
    return SparseSymmetricMatrix(
        domains.norbitals, left[keep].copy(), right[keep].copy(), values[keep].copy()
    )


__all__ = [
    "PurificationResult",
    "SparsePurificationResult",
    "SparseProjectorState",
    "mixed_precision_square",
    "sp2_purify",
    "sp2_purify_sparse",
    "sp2_purify_sparse_incremental",
]
