"""Hybrid density builders for the autonomous MATRIX-GFN2 ISCF."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from typing import Any
import warnings

import numpy as np
from matrix_numerics import best_diagonalizer_backend

try:
    from . import _gfn2_native
except ImportError:  # pragma: no cover
    _gfn2_native = None

from .locality import LocalizedOrbitalDomains
from .localized_solver import SparseSymmetricMatrix
from .purification import (
    SparseProjectorState,
    sp2_purify,
    sp2_purify_sparse,
    sp2_purify_sparse_incremental,
)
from .scc import (
    fermi_occupations,
    solve_generalized_symmetric,
    solve_generalized_symmetric_direct,
    symmetric_orthogonalizer,
)
from .symmetry import SymmetryBlockedMatrix


@lru_cache(maxsize=1)
def _hardware_calibration():
    if os.environ.get("MATRIX_DISABLE_AUTOCALIBRATION", "").strip() == "1":
        return None
    try:
        from .acceleration import load_or_calibrate

        return load_or_calibrate()
    except (ImportError, OSError, RuntimeError, ValueError):
        return None


@dataclass(frozen=True)
class SCFSolverPolicy:
    """Fail-closed selection policy for diagonalization and purification."""

    mode: str = "auto"
    dense_sp2_min_orbitals: int = 256
    # The one-thread recurrent energy+gradient crossover on the supported
    # Accelerate/LAPACK host lies between 2800 and 3000 orbitals.  Sparse SP2
    # is therefore automatic only from 3000 orbitals and still requires low
    # fill; explicit ``sparse-sp2`` requests remain available below it.
    sparse_sp2_min_orbitals: int = 3000
    sparse_sp2_maximum_fill_fraction: float = 0.25
    minimum_gap_hartree: float = 1.0e-3
    fractional_occupation_tolerance: float = 1.0e-8
    purification_tolerance: float = 2.0e-7
    maximum_purification_iterations: int = 80
    mixed_precision: bool = True
    incremental_transforms: bool = True
    transform_reset_interval: int = 16
    sparse_truncation_threshold: float = 1.0e-12

    def __post_init__(self) -> None:
        mode = self.normalized_mode
        if mode not in {"auto", "diagonalization", "sp2", "dense-sp2", "sparse-sp2"}:
            raise ValueError("SCF density solver must be auto, diagonalization, or SP2")
        if self.dense_sp2_min_orbitals < 1 or self.sparse_sp2_min_orbitals < 1:
            raise ValueError("SP2 size thresholds must be positive")
        if not 0.0 < self.sparse_sp2_maximum_fill_fraction <= 1.0:
            raise ValueError("sparse SP2 maximum fill fraction must lie in (0, 1]")
        if self.minimum_gap_hartree < 0.0 or not np.isfinite(
            self.minimum_gap_hartree
        ):
            raise ValueError("SP2 minimum gap must be nonnegative")
        if (
            self.fractional_occupation_tolerance < 0.0
            or not np.isfinite(self.fractional_occupation_tolerance)
        ):
            raise ValueError("fractional-occupation tolerance must be nonnegative")
        if self.purification_tolerance <= 0.0 or not np.isfinite(
            self.purification_tolerance
        ):
            raise ValueError("SP2 tolerance must be positive")
        if self.maximum_purification_iterations < 1:
            raise ValueError("SP2 maximum iterations must be positive")
        if self.transform_reset_interval < 1:
            raise ValueError("incremental transform reset interval must be positive")
        if self.sparse_truncation_threshold < 0.0 or not np.isfinite(
            self.sparse_truncation_threshold
        ):
            raise ValueError("sparse truncation threshold must be nonnegative")

    @property
    def normalized_mode(self) -> str:
        return str(self.mode).strip().lower().replace("_", "-")


@dataclass(frozen=True)
class DensityBuildResult:
    density: np.ndarray
    solver: str
    purification_iterations: int
    used_fallback: bool
    idempotency_error: float
    product_backend: str = "numpy"
    mixed_precision_iterations: int = 0
    fp64_iterations: int = 0
    device_transfers: int = 0
    peak_device_bytes: int = 0
    fallback_reason: str | None = None
    fallback_category: str | None = None
    orbital_energies: np.ndarray | None = None
    orbital_coefficients: np.ndarray | None = None


@dataclass(frozen=True)
class SparseDensityBuildResult:
    density: SparseSymmetricMatrix
    orthogonal_projector: SparseSymmetricMatrix
    orthogonal_fock: SparseSymmetricMatrix
    iterations: int
    idempotency_error: float
    accumulated_truncation_bound: float
    converged: bool
    projector_state: SparseProjectorState | None = None
    used_incremental_projector: bool = False
    incremental_fallback: bool = False


def sparse_closed_shell_density(
    fock: SparseSymmetricMatrix,
    orthogonalizer: SparseSymmetricMatrix,
    domains: LocalizedOrbitalDomains,
    occupied_orbitals: int,
    *,
    tolerance: float = 2.0e-7,
    maximum_iterations: int = 80,
    truncation_threshold: float = 1.0e-12,
    projector_state: SparseProjectorState | None = None,
    build_ao_density: bool = True,
) -> SparseDensityBuildResult:
    """Build a closed-shell AO density without allocating a dense matrix."""

    orthogonal_fock, transform_discarded = orthogonalizer.congruence(
        fock, domains, threshold=truncation_threshold
    )
    diagonal = np.zeros(fock.size, dtype=float)
    radius = np.zeros(fock.size, dtype=float)
    is_diagonal = orthogonal_fock.rows == orthogonal_fock.columns
    diagonal[orthogonal_fock.rows[is_diagonal]] = orthogonal_fock.values[is_diagonal]
    off = ~is_diagonal
    absolute = np.abs(orthogonal_fock.values[off])
    np.add.at(radius, orthogonal_fock.rows[off], absolute)
    np.add.at(radius, orthogonal_fock.columns[off], absolute)
    lower = float(np.min(diagonal - radius))
    upper = float(np.max(diagonal + radius))
    if upper <= lower:
        upper = lower + max(1.0, abs(lower)) * np.finfo(float).eps * 16.0
    used_incremental = False
    incremental_fallback = False
    interval_slack = 64.0 * np.finfo(float).eps * max(
        1.0, abs(lower), abs(upper)
    )
    reusable = (
        projector_state is not None
        and fock.size >= 128
        and os.environ.get("MATRIX_DISABLE_INCREMENTAL_SP2", "").strip() != "1"
        and projector_state.occupied_orbitals == int(occupied_orbitals)
        and lower >= projector_state.lower_bound_hartree - interval_slack
        and upper <= projector_state.upper_bound_hartree + interval_slack
    )
    incremental_threshold = max(truncation_threshold, 1.0e-10)
    if reusable:
        allowed_left, allowed_right = domains.allowed_pair_arrays()
        previous_x0 = np.asarray(projector_state.history[0], dtype=float)
        width = (
            projector_state.upper_bound_hartree
            - projector_state.lower_bound_hartree
        )
        current_x0 = -orthogonal_fock.values_on(
            allowed_left, allowed_right
        ) / width
        current_x0[allowed_left == allowed_right] += (
            projector_state.upper_bound_hartree / width
        )
        active_delta = np.count_nonzero(
            np.abs(current_x0 - previous_x0) >= incremental_threshold
        )
        # Exact perturbative SP2 uses two sparse products per stage instead of
        # one.  It is profitable only while the correction remains genuinely
        # local; otherwise a fresh canonical purification is both faster and
        # safer.  The 5% crossover is deliberately conservative.
        reusable = active_delta <= max(8, int(0.05 * len(current_x0)))
    if reusable:
        purified = sp2_purify_sparse_incremental(
            orthogonal_fock,
            int(occupied_orbitals),
            domains,
            projector_state,
            tolerance=tolerance,
            truncation_threshold=incremental_threshold,
        )
        used_incremental = purified.converged
        incremental_fallback = not purified.converged
    if not reusable or not purified.converged:
        purified = sp2_purify_sparse(
            orthogonal_fock,
            int(occupied_orbitals),
            domains,
            lower_bound_hartree=lower,
            upper_bound_hartree=upper,
            tolerance=tolerance,
            max_iterations=maximum_iterations,
            truncation_threshold=truncation_threshold,
        )
    next_state = None
    if purified.converged and purified.history is not None and purified.branches is not None:
        next_state = SparseProjectorState(
            float(lower if not used_incremental else projector_state.lower_bound_hartree),
            float(upper if not used_incremental else projector_state.upper_bound_hartree),
            int(occupied_orbitals),
            purified.history,
            purified.branches,
        )
    if build_ao_density:
        doubled_projector = SparseSymmetricMatrix(
            purified.projector.size,
            purified.projector.rows.copy(),
            purified.projector.columns.copy(),
            2.0 * purified.projector.values,
        )
        density, density_discarded = orthogonalizer.congruence(
            doubled_projector, domains, threshold=truncation_threshold
        )
    else:
        density = SparseSymmetricMatrix(
            purified.projector.size,
            np.empty(0, dtype=np.intp),
            np.empty(0, dtype=np.intp),
            np.empty(0, dtype=float),
        )
        density_discarded = 0.0
    return SparseDensityBuildResult(
        density,
        purified.projector,
        orthogonal_fock,
        purified.iterations,
        purified.idempotency_error,
        transform_discarded
        + purified.accumulated_truncation_bound
        + density_discarded,
        purified.converged,
        next_state,
        used_incremental,
        incremental_fallback,
    )


class IncrementalCongruenceTransform:
    """Accumulate ``X A X`` from small matrix increments with periodic resets."""

    def __init__(
        self,
        transformation: np.ndarray,
        *,
        mixed_precision: bool,
        reset_interval: int,
        product_backend: str = "numpy",
    ) -> None:
        self.transformation = np.asarray(transformation, dtype=np.float64)
        if (
            self.transformation.ndim != 2
            or self.transformation.shape[0] != self.transformation.shape[1]
            or not np.all(np.isfinite(self.transformation))
        ):
            raise ValueError("congruence transformation must be finite and square")
        self.mixed_precision = bool(mixed_precision)
        self.reset_interval = int(reset_interval)
        self.product_backend = str(product_backend)
        self._reference_input: np.ndarray | None = None
        self._reference_output: np.ndarray | None = None
        self._increments = 0
        self._accelerator_transformation: Any | None = None

    def reset(self) -> None:
        self._reference_input = None
        self._reference_output = None
        self._increments = 0

    def apply(self, matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix, dtype=np.float64)
        if values.shape != self.transformation.shape or not np.all(
            np.isfinite(values)
        ):
            raise ValueError("congruence input must be finite and dimensionally compatible")
        full = (
            self._reference_input is None
            or self._increments >= self.reset_interval
            or not self.mixed_precision
        )
        x = self.transformation
        if full:
            transformed = x @ values @ x.T
            self._increments = 0
        else:
            difference = values - self._reference_input
            delta, self._accelerator_transformation = _mixed_congruence(
                x,
                difference,
                backend=self.product_backend,
                prepared_transformation=self._accelerator_transformation,
            )
            transformed = self._reference_output + delta
            self._increments += 1
        transformed = 0.5 * (transformed + transformed.T)
        self._reference_input = values.copy()
        self._reference_output = transformed.copy()
        return transformed


def _product_backend(
    diagonalizer_backend: str | None,
    matrix_size: int,
) -> str:
    requested = os.environ.get("MATRIX_SP2_PRODUCT_BACKEND", "").strip().lower()
    if requested:
        return requested
    calibration = _hardware_calibration()
    if (
        calibration is not None
        and calibration.gpu_backend != "numpy"
        and matrix_size >= calibration.gpu_min_orbitals
    ):
        return calibration.gpu_backend
    if diagonalizer_backend is None:
        selected = best_diagonalizer_backend((matrix_size, matrix_size))
        if selected.available and selected.accelerated:
            return selected.name
        minimum_mps_size = int(
            os.environ.get("MATRIX_MPS_MIN_ORBITALS", "2048")
        )
        if matrix_size >= minimum_mps_size:
            try:
                import torch

                if torch.backends.mps.is_built() and torch.backends.mps.is_available():
                    return "torch-mps"
            except (ImportError, RuntimeError, ValueError):
                pass
    if diagonalizer_backend in {"cupy", "torch-cuda", "cuda", "torch-mps", "mps"}:
        return str(diagonalizer_backend)
    return "numpy"


def _mixed_congruence(
    transformation: np.ndarray,
    matrix: np.ndarray,
    *,
    backend: str,
    prepared_transformation: Any | None = None,
) -> tuple[np.ndarray, Any | None]:
    """Evaluate the cubic FP32 congruence on the selected accelerator."""

    x = np.asarray(transformation, dtype=np.float32)
    values = np.asarray(matrix, dtype=np.float32)
    normalized = str(backend).strip().lower()
    if normalized == "cupy":
        import cupy as cp

        if prepared_transformation is None:
            x_device = cp.asarray(x)
            values_device = cp.empty_like(x_device)
            prepared_transformation = (x_device, values_device)
        else:
            x_device, values_device = prepared_transformation
        cp.copyto(values_device, cp.asarray(values))
        return (
            cp.asnumpy(x_device @ values_device @ x_device.T).astype(np.float64),
            prepared_transformation,
        )
    if normalized in {"torch-cuda", "cuda", "torch-mps", "mps"}:
        import torch

        device = "mps" if normalized in {"torch-mps", "mps"} else "cuda"
        if prepared_transformation is None:
            x_device = torch.as_tensor(x, device=device)
            values_device = torch.empty_like(x_device)
            prepared_transformation = (x_device, values_device)
        else:
            x_device, values_device = prepared_transformation
        values_device.copy_(torch.as_tensor(values, device=device))
        return (
            (x_device @ values_device @ x_device.T)
            .cpu()
            .numpy()
            .astype(np.float64),
            prepared_transformation,
        )
    return np.asarray(x @ values @ x.T, dtype=np.float64), None


class HybridDensitySolver:
    """Stateful per-geometry density builder with conservative fallbacks."""

    def __init__(
        self,
        *,
        overlap: np.ndarray,
        orthogonalizer: np.ndarray,
        electrons: int,
        multiplicity: int,
        temperature_kelvin: float,
        initial_energies: np.ndarray,
        initial_occupations: np.ndarray,
        policy: SCFSolverPolicy,
        diagonalizer_backend: str | None,
        domains: LocalizedOrbitalDomains | None = None,
        symmetry_adapter: Any | None = None,
        direct_generalized: bool = False,
        native_context: Any | None = None,
    ) -> None:
        self.overlap = np.asarray(overlap, dtype=np.float64)
        self.orthogonalizer = np.asarray(orthogonalizer, dtype=np.float64)
        self.electrons = int(electrons)
        self.multiplicity = int(multiplicity)
        self.temperature_kelvin = float(temperature_kelvin)
        self.policy = policy
        self.diagonalizer_backend = diagonalizer_backend
        self.domains = domains
        self.symmetry_adapter = symmetry_adapter
        self.direct_generalized = bool(direct_generalized)
        self.native_context = native_context
        self.disabled = False
        self._fallback_category: str | None = None
        self.last_solver = "diagonalization"
        self.iterative_refinement = False
        self.canonical_refinement = False
        self.refinement_steps = 0
        self.canonical_solves = 0
        self._canonical_reference = None
        self._reference_metric = None
        self._reference_gaps = None
        self.transported_reference = False
        self.reuse_refinement_blocks = True
        self._gap = self._frontier_gap(initial_energies)
        occupations = np.asarray(initial_occupations, dtype=float)
        self._fractionality = float(
            np.max(np.minimum(np.abs(occupations), np.abs(occupations - 2.0)))
        )
        # Small auto-routed systems will diagonalize and never execute SP2.
        # Avoid hardware calibration (and accelerator imports) on this path;
        # it was larger than the complete SCC cost for typical molecules.
        if (
            policy.normalized_mode == "diagonalization"
            or (
                policy.normalized_mode == "auto"
                and len(self.overlap) < policy.dense_sp2_min_orbitals
            )
        ):
            self.product_backend = "numpy"
        else:
            self.product_backend = _product_backend(
                diagonalizer_backend, len(self.overlap)
            )
        self._symmetry_block_occupancies: tuple[int, ...] | None = None
        # The overlap blocks are geometry-invariant throughout one SCC
        # solve.  Cache their inverse square roots just like the full dense
        # route caches its congruence transform.
        self._symmetry_block_orthogonalizers: tuple[np.ndarray, ...] | None = None
        self._symmetry_sp2_cycles = 0
        mixed = policy.mixed_precision and self.product_backend != "numpy"
        # The resident generalized solver never forms an orthogonal-space
        # SP2 projector for small/medium auto-routed systems.  Avoid creating
        # two incremental congruence workspaces that cannot be reached on
        # that path; a fallback diagonalization also uses the native context
        # directly.
        if (
            self.direct_generalized
            and self.native_context is not None
            and not self.eligible_for_sp2
        ):
            self._fock_transform = None
            self._density_transform = None
        else:
            self._fock_transform = IncrementalCongruenceTransform(
                self.orthogonalizer,
                mixed_precision=mixed and policy.incremental_transforms,
                reset_interval=policy.transform_reset_interval,
                product_backend=self.product_backend,
            )
            self._density_transform = IncrementalCongruenceTransform(
                self.orthogonalizer,
                mixed_precision=mixed and policy.incremental_transforms,
                reset_interval=policy.transform_reset_interval,
                product_backend=self.product_backend,
            )

    def _disable_product_acceleration(self, error: BaseException) -> None:
        warnings.warn(
            f"{self.product_backend} SP2 acceleration failed; retrying in CPU FP64: "
            f"{error}",
            RuntimeWarning,
            stacklevel=2,
        )
        self.product_backend = "numpy"
        if isinstance(error, ImportError):
            self._fallback_category = "backend-unavailable"
        elif isinstance(error, MemoryError):
            self._fallback_category = "device-memory"
        else:
            self._fallback_category = "device-runtime"
        for transform in (self._fock_transform, self._density_transform):
            transform.product_backend = "numpy"
            transform.mixed_precision = False
            transform._accelerator_transformation = None
            transform.reset()

    def _frontier_gap(self, energies: np.ndarray) -> float:
        values = np.asarray(energies, dtype=float)
        unpaired = self.multiplicity - 1
        alpha = (self.electrons + unpaired) // 2
        beta = self.electrons - alpha
        gaps = [
            float(values[count] - values[count - 1])
            for count in (alpha, beta)
            if 0 < count < len(values)
        ]
        return min(gaps, default=float("inf"))

    @property
    def frontier_gap_hartree(self) -> float:
        return float(self._gap)

    def _density_to_ao(self, projector: np.ndarray) -> np.ndarray:
        try:
            return self._density_transform.apply(2.0 * projector)
        except (ImportError, RuntimeError, MemoryError) as error:
            self._disable_product_acceleration(error)
            return self._density_transform.apply(2.0 * projector)

    @property
    def eligible_for_sp2(self) -> bool:
        mode = self.policy.normalized_mode
        if mode == "diagonalization" or self.disabled:
            return False
        if self.multiplicity != 1 or self.electrons % 2:
            return False
        if self._gap < self.policy.minimum_gap_hartree:
            return False
        if self._fractionality > self.policy.fractional_occupation_tolerance:
            return False
        if mode == "auto" and len(self.overlap) < self.policy.dense_sp2_min_orbitals:
            return False
        return True

    def _solve_symmetry_blocks(self, fock: np.ndarray) -> DensityBuildResult:
        """Purify independent symmetry blocks and reconstruct the AO density."""

        assert self.symmetry_adapter is not None
        fock_blocks = self.symmetry_adapter.to_blocks(fock).values
        refresh_occupancies = (
            self._symmetry_block_occupancies is None
            or self._symmetry_sp2_cycles % 8 == 0
        )
        if refresh_occupancies:
            block_energies = [
                solve_generalized_symmetric(
                    block,
                    overlap,
                    diagonalizer_backend=self.diagonalizer_backend,
                )[0]
                for block, overlap in zip(
                    fock_blocks, self.symmetry_adapter.overlap_blocks, strict=True
                )
            ]
            all_energies = np.concatenate(block_energies)
            self._gap = self._frontier_gap(np.sort(all_energies, kind="stable"))
            order = np.argsort(all_energies, kind="stable")
            sorted_occupations = fermi_occupations(
                all_energies[order],
                self.electrons,
                multiplicity=self.multiplicity,
                temperature_kelvin=self.temperature_kelvin,
            )
            all_occupations = np.empty_like(sorted_occupations)
            all_occupations[order] = sorted_occupations
            occupation_blocks: list[np.ndarray] = []
            offset = 0
            for energies in block_energies:
                occupation_blocks.append(all_occupations[offset : offset + len(energies)])
                offset += len(energies)
            if any(
                np.max(np.minimum(np.abs(values), np.abs(values - 2.0)))
                > self.policy.fractional_occupation_tolerance
                for values in occupation_blocks
            ):
                raise RuntimeError("fractional occupations are not eligible for block SP2")
            occupancies = tuple(
                int(round(float(np.sum(values)) / 2.0))
                for values in occupation_blocks
            )
            if (
                self._symmetry_block_occupancies is not None
                and occupancies != self._symmetry_block_occupancies
            ):
                raise RuntimeError("symmetry block occupations changed during SP2")
            self._symmetry_block_occupancies = occupancies
        else:
            occupancies = self._symmetry_block_occupancies
            block_energies = [None] * len(fock_blocks)

        density_blocks: list[np.ndarray] = []
        iterations = 0
        idempotency_error = 0.0
        mixed_precision_iterations = 0
        fp64_iterations = 0
        device_transfers = 0
        peak_device_bytes = 0
        for block_index, (fock_block, overlap_block, energies) in enumerate(zip(
            fock_blocks,
            self.symmetry_adapter.overlap_blocks,
            block_energies,
            strict=True,
        )):
            occupied = occupancies[block_index]
            if self._symmetry_block_orthogonalizers is None:
                self._symmetry_block_orthogonalizers = tuple(
                    symmetric_orthogonalizer(
                        block,
                        diagonalizer_backend=self.diagonalizer_backend,
                    )
                    for block in self.symmetry_adapter.overlap_blocks
                )
            orthogonalizer = self._symmetry_block_orthogonalizers[block_index]
            orthogonal_fock = orthogonalizer @ fock_block @ orthogonalizer
            if energies is not None and 0 < occupied < len(energies):
                gap = float(energies[occupied] - energies[occupied - 1])
            else:
                gap = None
            try:
                purified = sp2_purify(
                    orthogonal_fock,
                    occupied,
                    tolerance=self.policy.purification_tolerance,
                    max_iterations=self.policy.maximum_purification_iterations,
                    mixed_precision=self.policy.mixed_precision,
                    estimated_gap_hartree=gap,
                    minimum_gap_hartree=self.policy.minimum_gap_hartree,
                    product_backend=self.product_backend,
                )
            except (ImportError, RuntimeError, MemoryError) as error:
                # A single problematic irrep must not discard all other
                # symmetry blocks.  Retry that block in deterministic FP64
                # before escalating to a full-matrix diagonalization.
                self._disable_product_acceleration(error)
                purified = sp2_purify(
                    orthogonal_fock,
                    occupied,
                    tolerance=self.policy.purification_tolerance,
                    max_iterations=self.policy.maximum_purification_iterations,
                    mixed_precision=False,
                    estimated_gap_hartree=gap,
                    minimum_gap_hartree=self.policy.minimum_gap_hartree,
                    product_backend="numpy",
                )
            if not purified.converged or purified.used_dense_fallback:
                # Certification failures are numerical, rather than a
                # symmetry-model failure.  One conservative FP64 retry keeps
                # the symmetry route alive on marginally conditioned blocks.
                if self.product_backend != "numpy":
                    self._disable_product_acceleration(
                        RuntimeError("symmetry block SP2 certification failed")
                    )
                    purified = sp2_purify(
                        orthogonal_fock,
                        occupied,
                        tolerance=self.policy.purification_tolerance,
                        max_iterations=self.policy.maximum_purification_iterations,
                        mixed_precision=False,
                        estimated_gap_hartree=gap,
                        minimum_gap_hartree=self.policy.minimum_gap_hartree,
                        product_backend="numpy",
                    )
            if not purified.converged or purified.used_dense_fallback:
                raise RuntimeError("symmetry block SP2 certification failed")
            density_blocks.append(
                orthogonalizer @ (2.0 * purified.projector) @ orthogonalizer
            )
            iterations = max(iterations, purified.iterations)
            idempotency_error = max(idempotency_error, purified.idempotency_error)
            mixed_precision_iterations += purified.mixed_precision_iterations
            fp64_iterations += purified.fp64_iterations
            device_transfers += purified.device_transfers
            peak_device_bytes = max(peak_device_bytes, purified.peak_device_bytes)
        density = self.symmetry_adapter.from_blocks(
            SymmetryBlockedMatrix(
                labels=self.symmetry_adapter.block_labels,
                indices=self.symmetry_adapter.blocks,
                values=tuple(density_blocks),
            )
        )
        self.last_solver = "symmetry-block-sp2"
        self._symmetry_sp2_cycles += 1
        return DensityBuildResult(
            density,
            self.last_solver,
            iterations,
            False,
            idempotency_error,
            self.product_backend,
            mixed_precision_iterations,
            fp64_iterations,
            device_transfers,
            peak_device_bytes,
        )

    def solve(self, fock: np.ndarray) -> DensityBuildResult:
        if self.canonical_refinement:
            return self._diagonalize(fock, used_fallback=False)
        if self.iterative_refinement and self._canonical_reference is not None:
            refined = self._refine_occupied(fock)
            if refined is not None:
                return refined
        if not self.eligible_for_sp2:
            return self._diagonalize(fock, used_fallback=False)
        try:
            if self.symmetry_adapter is not None:
                return self._solve_symmetry_blocks(fock)
            orthogonal_fock = self._fock_transform.apply(fock)
        except (ImportError, RuntimeError, MemoryError) as error:
            if self.symmetry_adapter is not None:
                self.disabled = True
                return self._diagonalize(
                    fock,
                    used_fallback=True,
                    fallback_reason=str(error),
                    fallback_category="symmetry-block-fallback",
                )
            self._disable_product_acceleration(error)
            orthogonal_fock = self._fock_transform.apply(fock)
        occupied = self.electrons // 2
        mode = self.policy.normalized_mode
        wants_sparse = mode == "sparse-sp2" or (
            mode == "auto"
            and len(fock)
            >= (
                _hardware_calibration().sparse_min_orbitals
                if _hardware_calibration() is not None
                else self.policy.sparse_sp2_min_orbitals
            )
        )
        if wants_sparse and self.domains is not None:
            diagonal = np.diag(orthogonal_fock)
            radii = np.sum(np.abs(orthogonal_fock), axis=1) - np.abs(diagonal)
            lower = float(np.min(diagonal - radii))
            upper = float(np.max(diagonal + radii))
            sparse_fock = SparseSymmetricMatrix.from_dense(
                orthogonal_fock,
                self.domains,
                threshold=self.policy.sparse_truncation_threshold,
            )
            sparse = sp2_purify_sparse(
                sparse_fock,
                occupied,
                self.domains,
                lower_bound_hartree=lower,
                upper_bound_hartree=upper,
                tolerance=self.policy.purification_tolerance,
                max_iterations=self.policy.maximum_purification_iterations,
                truncation_threshold=self.policy.sparse_truncation_threshold,
            )
            if sparse.converged:
                projector = sparse.projector.to_dense()
                density = self._density_to_ao(projector)
                self.last_solver = "sparse-sp2"
                return DensityBuildResult(
                    density,
                    self.last_solver,
                    sparse.iterations,
                    False,
                    sparse.idempotency_error,
                    "sparse-native-cpu" if _gfn2_native is not None else "sparse-cpu",
                )
        try:
            dense = sp2_purify(
                orthogonal_fock,
                occupied,
                tolerance=self.policy.purification_tolerance,
                max_iterations=self.policy.maximum_purification_iterations,
                mixed_precision=(
                    self.policy.mixed_precision
                    and self.product_backend != "numpy"
                ),
                estimated_gap_hartree=self._gap,
                minimum_gap_hartree=self.policy.minimum_gap_hartree,
                product_backend=self.product_backend,
            )
        except (ImportError, RuntimeError, MemoryError) as error:
            self._disable_product_acceleration(error)
            orthogonal_fock = self._fock_transform.apply(fock)
            dense = sp2_purify(
                orthogonal_fock,
                occupied,
                tolerance=self.policy.purification_tolerance,
                max_iterations=self.policy.maximum_purification_iterations,
                mixed_precision=False,
                estimated_gap_hartree=self._gap,
                minimum_gap_hartree=self.policy.minimum_gap_hartree,
            )
        if dense.converged and not dense.used_dense_fallback:
            density = self._density_to_ao(dense.projector)
            self.last_solver = "dense-sp2"
            return DensityBuildResult(
                density,
                self.last_solver,
                dense.iterations,
                False,
                dense.idempotency_error,
                dense.product_backend,
                dense.mixed_precision_iterations,
                dense.fp64_iterations,
                dense.device_transfers,
                dense.peak_device_bytes,
            )
        self.disabled = True
        self._fock_transform.reset()
        self._density_transform.reset()
        return self._diagonalize(
            fock,
            used_fallback=True,
            fallback_reason="SP2 certification failed",
            fallback_category="sp2-certification",
        )

    def seed_orbital_reference(self, coefficients, energies, occupations) -> bool:
        """Transport a certified frame into the new metric, without an eigensolve."""
        from scipy.linalg import cholesky, solve_triangular
        c = np.asarray(coefficients, dtype=float)
        e = np.asarray(energies, dtype=float)
        occ = np.asarray(occupations, dtype=float)
        if (c.shape != self.overlap.shape or e.shape != (len(c),)
                or occ.shape != e.shape or self.multiplicity != 1
                or not np.all(np.isfinite(c)) or not np.all(np.isfinite(e))
                or not np.all(np.isfinite(occ)) or np.any(occ < 0) or np.any(occ > 2)
                or abs(float(np.sum(occ))-self.electrons) > 1e-7):
            return False
        metric_c = self.overlap @ c
        gram = c.T @ metric_c
        if np.max(np.sum(abs(gram-np.eye(len(c))), axis=1)) > .05:
            return False
        try:
            factor = cholesky(0.5*(gram+gram.T), lower=True, check_finite=True)
            c = solve_triangular(factor, c.T, lower=True, check_finite=False).T
            metric_c = solve_triangular(factor, metric_c.T, lower=True, check_finite=False).T
        except (ValueError, np.linalg.LinAlgError):
            return False
        if np.max(abs(c.T@metric_c-np.eye(len(c)))) > 2e-10:
            return False
        self._canonical_reference = (c, e.copy(), occ.copy())
        self._reference_metric = metric_c
        self._reference_gaps = None
        self.transported_reference = True
        return True

    def _refine_occupied(self, fock: np.ndarray) -> DensityBuildResult | None:
        """Refine a nearby occupied projector using existing Sylvester/Cholesky solvers.

        This is an intermediate density construction. Canonical eigenvectors
        are still independently computed before the final state is accepted.
        Only numerically integer Fermi occupations are eligible.
        """
        from scipy.linalg import cholesky, solve_triangular
        from .sf_tda_gradient import _matrix_free_gmres
        from .scc import BOLTZMANN_HARTREE_PER_KELVIN

        coefficients, energies, occupations = self._canonical_reference
        # A transported frame is independent of its old occupation weights.
        # Eligibility must use the CURRENT SCF temperature, including the
        # calculator's existing temperature schedule, not the previous one.
        occupations = fermi_occupations(
            energies, self.electrons, multiplicity=self.multiplicity,
            temperature_kelvin=self.temperature_kelvin,
        )
        occupied = self.electrons // 2
        if (self.multiplicity != 1 or not 0 < occupied < len(fock)
                or np.max(np.abs(occupations-np.rint(occupations))) > 1e-12):
            return None
        co, cv = coefficients[:, :occupied], coefficients[:, occupied:]
        # Three independent blocks suffice: F_ov is exactly F_vo.T. Keep
        # F times the columns for rotation/residual checks below.
        if self.reuse_refinement_blocks:
            ao, av = fock@co, fock@cv
            foo, fvv, fvo = co.T@ao, cv.T@av, cv.T@ao
            perturbation_bound = float(max(
                np.max(np.sum(abs(foo-np.diag(energies[:occupied])), axis=1)
                       + np.sum(abs(fvo), axis=0)),
                np.max(np.sum(abs(fvv-np.diag(energies[occupied:])), axis=1)
                       + np.sum(abs(fvo), axis=1)),
            ))
        else:
            mo = coefficients.T @ fock @ coefficients
            perturbation_bound = float(np.max(np.sum(np.abs(mo-np.diag(energies)), axis=1)))
        gap = energies[occupied]-energies[occupied-1]
        thermal_bound = 2*BOLTZMANN_HARTREE_PER_KELVIN*self.temperature_kelvin*np.log(
            4*len(fock)/np.finfo(float).eps)
        if gap-2*perturbation_bound <= max(thermal_bound, 1e-3):
            return None
        if perturbation_bound > 0.02*gap:
            return None
        if self._reference_metric is None:
            self._reference_metric = self.overlap@coefficients
        so, sv = self._reference_metric[:, :occupied], self._reference_metric[:, occupied:]
        if self._reference_gaps is None:
            self._reference_gaps = energies[occupied:, None]-energies[None, :occupied]
        for _ in range(2):
            if _ > 0:
                ao = fock @ co
                av = fock @ cv
                foo, fvv, fvo = co.T@ao, cv.T@av, cv.T@ao
            elif not self.reuse_refinement_blocks:
                foo, fvv, fvo = mo[:occupied, :occupied], mo[occupied:, occupied:], mo[occupied:, :occupied]
            try:
                gaps = (self._reference_gaps if self.reuse_refinement_blocks
                        else np.diag(fvv)[:, None]-np.diag(foo)[None, :])
                if np.min(gaps) <= 1e-3:
                    return None

                def sylvester_action(vector):
                    trial = vector.reshape(fvo.shape)
                    return ((fvv@trial-trial@foo)/gaps).ravel()

                solution, response_residual, _steps = _matrix_free_gmres(
                    sylvester_action, (-fvo/gaps).ravel(), tolerance=1e-12,
                    maximum_iterations=12, single_pass=True,
                )
                if response_residual > 1e-11:
                    return None
                rotation = solution.reshape(fvo.shape)
                if np.max(np.abs(rotation)) > 0.02:
                    return None
                trial_o = co+cv@rotation
                lo = cholesky(np.eye(occupied)+rotation.T@rotation, lower=True, check_finite=False)
                new_co = solve_triangular(lo, trial_o.T, lower=True, check_finite=False).T
                if self.reuse_refinement_blocks:
                    fc = solve_triangular(lo, (ao+av@rotation).T, lower=True, check_finite=False).T
                    metric_co = solve_triangular(lo, (so+sv@rotation).T, lower=True, check_finite=False).T
                else:
                    fc, metric_co = fock@new_co, self.overlap@new_co
            except np.linalg.LinAlgError:
                return None
            self.refinement_steps += 1
            stationarity = fc-metric_co@(new_co.T@fc)
            if np.max(np.abs(stationarity)) <= 1e-11:
                if np.max(np.abs(new_co.T@metric_co-np.eye(occupied))) > 2e-10:
                    return None
                density = 2*new_co@new_co.T
                self.last_solver = "occupied-refinement"
                return DensityBuildResult(
                    density, self.last_solver, 0, False,
                    float(np.max(np.abs(new_co.T@metric_co-np.eye(occupied)))),
                    "cpu-fp64-sylvester",
                )
            if _ == 0:
                # Virtual columns are unnecessary if the occupied projector
                # already passed. Construct them only for a second update.
                lv = cholesky(np.eye(len(fock)-occupied)+rotation@rotation.T,
                              lower=True, check_finite=False)
                cv = solve_triangular(lv, (cv-co@rotation.T).T, lower=True, check_finite=False).T
                co = new_co
                so, sv = self.overlap@co, self.overlap@cv
        return None

    def _diagonalize(
        self,
        fock: np.ndarray,
        *,
        used_fallback: bool,
        fallback_reason: str | None = None,
        fallback_category: str | None = None,
    ) -> DensityBuildResult:
        actual_backend = "symmetry-blocks" if self.symmetry_adapter is not None else "numpy"
        if self.symmetry_adapter is None:
            if self.direct_generalized:
                actual_backend = "native-lapack" if self.native_context is not None else "scipy"
                if self.native_context is not None:
                    energies, coefficients = self.native_context.generalized_eigh(fock)
                else:
                    energies, coefficients = solve_generalized_symmetric_direct(
                        fock, self.overlap
                    )
            else:
                energies, coefficients = solve_generalized_symmetric(
                    fock,
                    self.overlap,
                    orthogonalizer=self.orthogonalizer,
                    diagonalizer_backend=self.diagonalizer_backend,
                )
                # The general diagonalizer explicitly falls back to host
                # LAPACK when MPS cannot solve an FP64 eigenproblem.
                requested = str(self.diagonalizer_backend or "auto")
                actual_backend = (
                    "scipy" if requested in {"mps", "torch-mps"}
                    else best_diagonalizer_backend(fock.shape).name
                    if requested == "auto" else requested
                )
        else:
            energies, coefficients = self.symmetry_adapter.solve(
                fock,
                diagonalizer_backend=self.diagonalizer_backend,
            )
        occupations = fermi_occupations(
            energies,
            self.electrons,
            multiplicity=self.multiplicity,
            temperature_kelvin=self.temperature_kelvin,
        )
        self._gap = self._frontier_gap(energies)
        self.canonical_solves += 1
        if self.iterative_refinement:
            self._canonical_reference = (coefficients, energies, occupations)
            self._reference_metric = None
            self._reference_gaps = None
        density = (coefficients * occupations) @ coefficients.T
        self.last_solver = "diagonalization"
        return DensityBuildResult(
            density,
            self.last_solver,
            0,
            used_fallback,
            0.0,
            actual_backend,
            fallback_reason=fallback_reason,
            fallback_category=fallback_category or self._fallback_category,
            orbital_energies=energies,
            orbital_coefficients=coefficients,
        )


__all__ = [
    "DensityBuildResult",
    "HybridDensitySolver",
    "IncrementalCongruenceTransform",
    "SCFSolverPolicy",
    "SparseDensityBuildResult",
    "sparse_closed_shell_density",
]
