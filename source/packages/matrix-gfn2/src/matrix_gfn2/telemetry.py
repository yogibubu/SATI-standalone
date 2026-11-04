"""Structured, serializable execution telemetry for autonomous calculators."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


TELEMETRY_SCHEMA = "matrix.gfn2.telemetry.v1.1"


@dataclass(frozen=True)
class SolverTelemetry:
    requested_solver: str
    used_solver: str
    requested_backend: str
    used_backend: str
    fallback_reason: str | None
    purification_iterations: int
    mixed_precision_iterations: int
    fp64_iterations: int
    device_transfers: int
    peak_device_bytes: int
    idempotency_error: float
    fallback_category: str | None = None


@dataclass(frozen=True)
class GFN2Telemetry:
    schema: str
    solver: SolverTelemetry
    scc_iterations: int
    final_density_residual: float
    final_energy_residual_hartree: float
    maximum_density_residual: float
    maximum_energy_residual_hartree: float
    converged: bool
    accelerator_execution: Mapping[str, object] | None = None
    total_wall_seconds: float = 0.0
    peak_rss_bytes: int = 0
    model_wall_seconds: float = 0.0
    initial_solver_wall_seconds: float = 0.0
    iscf_wall_seconds: float = 0.0
    final_solver_wall_seconds: float = 0.0
    post_scf_wall_seconds: float = 0.0
    gradient_wall_seconds: float = 0.0
    bond_order_wall_seconds: float = 0.0
    analysis_wall_seconds: float = 0.0
    scc_energy_monotonic: bool = True
    minimum_damping: float = 1.0
    symmetry_point_group: str = "C1"
    symmetry_operation_count: int = 1
    symmetry_atom_orbit_count: int = 0
    symmetry_shell_orbit_count: int = 0
    symmetry_block_dimensions: tuple[int, ...] = ()
    symmetry_block_labels: tuple[str, ...] = ()
    symmetry_block_batches: tuple[tuple[int, tuple[int, ...]], ...] = ()
    symmetry_block_storage_fraction: float = 1.0
    symmetry_overlap_invariance_error: float = 0.0
    symmetry_geometry_error_angstrom: float = 0.0
    symmetry_fock_off_block_error: float = 0.0
    symmetry_density_off_block_error: float = 0.0
    symmetry_fock_commutator_error: float = 0.0
    symmetry_density_commutator_error: float = 0.0
    symmetry_irrep_degeneracy_error_hartree: float = 0.0
    symmetry_embedding_mode: str = "none"
    symmetry_solver_mode: str = "full-matrix"
    symmetry_embedding_orbits: tuple[tuple[int, ...], ...] = ()
    symmetry_embedding_residual: float = 0.0
    symmetry_full_matrix_reason: str | None = None
    symmetry_cache_hits: int = 0
    symmetry_cache_misses: int = 0
    symmetry_cache_evictions: int = 0

    def validate(self) -> None:
        """Validate the public telemetry contract before crossing process boundaries."""
        if self.schema != TELEMETRY_SCHEMA:
            raise ValueError(f"unsupported GFN2 telemetry schema: {self.schema}")
        if self.total_wall_seconds < 0.0 or self.peak_rss_bytes < 0:
            raise ValueError("telemetry resource counters must be non-negative")
        if min(self.symmetry_cache_hits, self.symmetry_cache_misses, self.symmetry_cache_evictions) < 0:
            raise ValueError("symmetry cache counters must be non-negative")
        if self.symmetry_block_storage_fraction <= 0.0:
            raise ValueError("symmetry block storage fraction must be positive")
        if self.symmetry_solver_mode not in {
            "full-matrix",
            "full-matrix-fallback",
            "symmetry-blocked",
        }:
            raise ValueError("unknown symmetry solver mode")

    def to_dict(self) -> Mapping[str, object]:
        return asdict(self)


__all__ = ["GFN2Telemetry", "SolverTelemetry", "TELEMETRY_SCHEMA"]
