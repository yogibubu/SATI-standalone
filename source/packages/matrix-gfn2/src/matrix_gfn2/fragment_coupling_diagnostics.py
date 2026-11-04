"""Measure whether fragment-local orbital rotations concentrate couplings.

Orthogonal rotations within fragments cannot reduce an interfragment block's
Frobenius norm. They may concentrate that norm in fewer matrix elements;
this module quantifies that opportunity before any truncation is attempted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh

from .hybrid_basis import HybridAtomicBasis
from .hybrid_domains import PersistentHybridDomains
from .bond_localized_basis import BondLocalizedBasis


@dataclass(frozen=True)
class FragmentCouplingMeasurement:
    fragment_pair: tuple[int, int]
    frobenius_norm: float
    original_entries_for_target: int
    local_mo_entries_for_target: int
    total_entries: int


@dataclass(frozen=True)
class BondCouplingMeasurement:
    fragment_pair: tuple[int, int]
    frobenius_norm: float
    hybrid_entries_for_target: int
    bond_entries_for_target: int
    total_entries: int


def _entries_for_fraction(block: np.ndarray, fraction: float) -> int:
    weights = np.sort(np.square(np.abs(block.ravel())))[::-1]
    total = float(np.sum(weights))
    if total <= 0.0:
        return 0
    return int(np.searchsorted(np.cumsum(weights), fraction * total, side="left") + 1)


def measure_local_fragment_couplings(
    operator: np.ndarray,
    overlap: np.ndarray,
    hybrid: HybridAtomicBasis,
    domains: PersistentHybridDomains,
    *,
    retained_norm_fraction: float = 0.999,
) -> tuple[FragmentCouplingMeasurement, ...]:
    """Compare block concentration before/after local H/S diagonalization."""
    h = hybrid.transform_operator(operator)
    s = hybrid.transform_operator(overlap)
    if (
        not 0.0 < retained_norm_fraction <= 1.0
        or h.shape != s.shape or h.shape != (hybrid.size, hybrid.size)
    ):
        raise ValueError("invalid fragment coupling diagnostic model")
    orthogonalizers = []
    local_rotations = []
    for indices in domains.orbitals:
        local_s = s[np.ix_(indices, indices)]
        eigenvalues, eigenvectors = eigh(local_s)
        if np.min(eigenvalues) <= 1.0e-10:
            raise ValueError("fragment overlap is not positive definite")
        x = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
        local_h = x.T @ h[np.ix_(indices, indices)] @ x
        _, rotation = eigh(local_h)
        orthogonalizers.append(x)
        local_rotations.append(rotation)
    measurements = []
    for left, left_indices in enumerate(domains.orbitals):
        for right in range(left):
            right_indices = domains.orbitals[right]
            block = (
                orthogonalizers[left].T
                @ h[np.ix_(left_indices, right_indices)]
                @ orthogonalizers[right]
            )
            rotated = local_rotations[left].T @ block @ local_rotations[right]
            norm = float(np.linalg.norm(block))
            if not np.isclose(np.linalg.norm(rotated), norm, atol=1.0e-10):
                raise RuntimeError("local rotation changed interfragment block norm")
            measurements.append(FragmentCouplingMeasurement(
                (right, left), norm,
                _entries_for_fraction(block, retained_norm_fraction),
                _entries_for_fraction(rotated, retained_norm_fraction),
                block.size,
            ))
    return tuple(measurements)


def measure_bond_localized_couplings(
    operator: np.ndarray,
    hybrid: HybridAtomicBasis,
    bond_basis: BondLocalizedBasis,
    domains: PersistentHybridDomains,
    *,
    retained_norm_fraction: float = 0.999,
) -> tuple[BondCouplingMeasurement, ...]:
    """Compare atom hybrids with exact internal-bond rotations on each block."""
    if not 0.0 < retained_norm_fraction <= 1.0:
        raise ValueError("retained coupling fraction must be positive")
    hybrid_operator = hybrid.transform_operator(operator)
    bond_operator = bond_basis.transform_operator(hybrid_operator)
    result = []
    for left, left_indices in enumerate(domains.orbitals):
        for right in range(left):
            right_indices = domains.orbitals[right]
            before = hybrid_operator[np.ix_(left_indices, right_indices)]
            after = bond_operator[np.ix_(left_indices, right_indices)]
            norm = float(np.linalg.norm(before))
            if not np.isclose(norm, np.linalg.norm(after), atol=1.0e-10):
                raise RuntimeError("bond rotation altered interfragment coupling norm")
            result.append(BondCouplingMeasurement(
                (right, left), norm,
                _entries_for_fraction(before, retained_norm_fraction),
                _entries_for_fraction(after, retained_norm_fraction),
                before.size,
            ))
    return tuple(result)
