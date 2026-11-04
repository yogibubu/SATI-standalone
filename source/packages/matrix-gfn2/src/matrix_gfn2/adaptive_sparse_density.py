"""Adaptive sparse-density SCF with untruncated sparse-projector algebra.

The orthogonal-basis audit covers fill outside the retained pair domain, but
does not certify AO integrals never built by the sparse model. Energy/gradient
parity against an independent dense reference is optional and explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy.sparse import csr_matrix

from .calculator import ANGSTROM_TO_BOHR, GFN2Calculator
from .locality import (
    LocalityPolicy, LocalizedOrbitalDomains,
    localized_domains_from_atom_neighbors,
)
from .sparse_projector_audit import audit_sparse_projector


@dataclass(frozen=True)
class SparseDomainAttempt:
    radius_bohr: float
    buffer_hops: int
    pair_fill_fraction: float
    seconds: float
    iterations: int | None
    maximum_orthogonal_residual: float | None
    maximum_idempotency_error: float | None
    electron_count_error: float | None
    failure: str | None
    residual_growth_pairs: int = 0
    reference_energy_error_hartree: float | None = None
    reference_gradient_error_hartree_per_bohr: float | None = None
    truncation_bound: float | None = None


@dataclass(frozen=True)
class AdaptiveSparseDensityResult:
    result: object
    attempts: tuple[SparseDomainAttempt, ...]


def _grow_from_external_residual(
    domains: LocalizedOrbitalDomains,
    commutator: csr_matrix,
    *,
    tolerance: float,
    maximum_atom_pairs: int,
) -> tuple[LocalizedOrbitalDomains, int] | None:
    """Add atom pairs from nonzero commutator entries outside the domain."""
    entries = commutator.tocoo()
    ranking = np.argsort(abs(entries.data))[::-1]
    neighbors = [set(row) for row in domains.atom_neighbors]
    added = 0
    for index in ranking:
        if abs(entries.data[index]) <= tolerance:
            break
        left, right = int(entries.row[index]), int(entries.col[index])
        left_atom = domains.orbital_atom[left]
        right_atom = domains.orbital_atom[right]
        if right_atom in neighbors[left_atom]:
            continue
        neighbors[left_atom].add(right_atom)
        neighbors[right_atom].add(left_atom)
        added += 1
        if added >= maximum_atom_pairs:
            break
    if not added:
        return None
    return localized_domains_from_atom_neighbors(
        neighbors, domains.orbital_atom,
    ), added


def adaptive_sparse_density_scf(
    symbols: tuple[str, ...],
    coordinates_angstrom: np.ndarray,
    *,
    domain_schedule: tuple[tuple[float, int], ...] = (
        (20.0, 1), (24.0, 1), (32.0, 1),
    ),
    orthogonal_residual_tolerance: float = 1.0e-6,
    idempotency_tolerance: float = 1.0e-6,
    electron_count_tolerance: float = 1.0e-5,
    maximum_scf_iterations: int = 250,
    maximum_residual_expansions: int = 2,
    atom_pairs_per_expansion: int = 8,
    reference_result: object | None = None,
    reference_energy_tolerance_hartree: float = 1.0e-7,
    reference_gradient_tolerance_hartree_per_bohr: float = 1.0e-6,
) -> AdaptiveSparseDensityResult:
    """Expand domains until the retained sparse model passes all local gates.

    This is not a full-AO certificate: omitted AO integrals require a separate
    bound. The optional dense result independently gates energy and gradient
    parity and is never used as a seed or as part of the sparse audit.
    """
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    if (len(symbols) < 21
            or xyz.shape != (len(symbols), 3)
            or not np.all(np.isfinite(xyz))
            or not domain_schedule or orthogonal_residual_tolerance <= 0.0
            or idempotency_tolerance <= 0.0
            or electron_count_tolerance <= 0.0
            or maximum_residual_expansions < 0
            or atom_pairs_per_expansion < 1
            or not np.isfinite(reference_energy_tolerance_hartree)
            or not np.isfinite(reference_gradient_tolerance_hartree_per_bohr)
            or reference_energy_tolerance_hartree < 0.0
            or reference_gradient_tolerance_hartree_per_bohr < 0.0):
        raise ValueError("invalid adaptive sparse-density audit")
    if reference_result is not None:
        reference_energy = float(reference_result.energy_hartree)
        reference_gradient = np.asarray(
            reference_result.gradient_hartree_per_bohr, dtype=float,
        )
        if (not np.isfinite(reference_energy)
                or reference_gradient.shape != xyz.shape
                or not np.all(np.isfinite(reference_gradient))):
            raise ValueError("invalid independent dense reference")
    coordinates_bohr = xyz * ANGSTROM_TO_BOHR
    attempts = []
    for radius, hops in domain_schedule:
        policy = LocalityPolicy(
            interaction_cutoff_bohr=float(radius), buffer_hops=int(hops),
        )
        calculator = GFN2Calculator(
            symbols, density_solver="sparse-sp2",
            maximum_scf_iterations=maximum_scf_iterations,
        )
        domains = calculator._localized_domains(
            coordinates_bohr, calculator._basis_template.spherical_atom,
            policy,
        )
        growth_pairs = 0
        for refinement in range(maximum_residual_expansions + 1):
            started = perf_counter()
            try:
                result = calculator._single_point_sparse(
                    coordinates_bohr, domains=domains,
                    locality_policy=policy, compute_bond_orders=False,
                    retain_projectors=True,
                )
            except (RuntimeError, FloatingPointError) as exc:
                if not any(word in str(exc) for word in (
                    "did not certify", "did not converge",
                    "sparse SP2 became non-finite",
                )):
                    raise
                attempts.append(SparseDomainAttempt(
                    radius, hops, domains.fill_fraction,
                    perf_counter() - started, None, None, None, None,
                    str(exc), growth_pairs,
                ))
                break
            if result.sparse_projector_state is None:
                raise RuntimeError("sparse SCF did not retain its projector")
            audit = audit_sparse_projector(result.sparse_projector_state)
            residual = audit.maximum_commutator
            projector_error = audit.maximum_idempotency_error
            charge_error = audit.electron_count_error
            energy_error = (
                abs(float(result.energy_hartree) - reference_energy)
                if reference_result is not None else None
            )
            gradient_error = (
                float(np.max(abs(
                    result.gradient_hartree_per_bohr - reference_gradient
                ))) if reference_result is not None else None
            )
            accepted = (
                np.isfinite(residual)
                and np.isfinite(projector_error)
                and np.isfinite(charge_error)
                and residual <= orthogonal_residual_tolerance
                and projector_error <= idempotency_tolerance
                and charge_error <= electron_count_tolerance
                and audit.truncation_bound <= idempotency_tolerance
                and (energy_error is None or (
                    np.isfinite(energy_error)
                    and energy_error <= reference_energy_tolerance_hartree
                ))
                and (gradient_error is None or (
                    np.isfinite(gradient_error)
                    and gradient_error <= reference_gradient_tolerance_hartree_per_bohr
                ))
            )
            attempts.append(SparseDomainAttempt(
                radius, hops, domains.fill_fraction, perf_counter() - started,
                result.iterations, residual, projector_error, charge_error,
                None if accepted else "sparse projector/parity audit failed",
                growth_pairs, energy_error, gradient_error,
                audit.truncation_bound,
            ))
            if accepted:
                return AdaptiveSparseDensityResult(result, tuple(attempts))
            if refinement == maximum_residual_expansions:
                break
            expanded = _grow_from_external_residual(
                domains, audit.commutator,
                tolerance=orthogonal_residual_tolerance,
                maximum_atom_pairs=atom_pairs_per_expansion,
            )
            if expanded is None:
                break
            domains, growth_pairs = expanded
    raise RuntimeError(
        "adaptive sparse density exhausted its domain schedule without "
        f"a sparse projector certificate; last attempt: {attempts[-1]}"
    )


__all__ = [
    "SparseDomainAttempt", "AdaptiveSparseDensityResult",
    "adaptive_sparse_density_scf",
]
