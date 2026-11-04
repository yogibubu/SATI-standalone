"""Pair-natural orbital compression on persistent local hybrid domains.

The caller must supply a correlated pair density. ORACLE charges, a neutral
atom density, or an SCF density are not valid substitutes for that object.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh


@dataclass(frozen=True)
class LocalPairNaturalOrbitals:
    domain_orbitals: np.ndarray
    coefficients: np.ndarray
    occupations: np.ndarray
    discarded_occupation: float
    total_occupation: float

    @property
    def retained_count(self) -> int:
        return self.coefficients.shape[1]


def diagonalize_pair_density(
    pair_density: np.ndarray,
    overlap: np.ndarray,
    domain_orbitals: np.ndarray,
    *,
    occupation_cutoff: float = 1.0e-7,
    maximum_discarded_fraction: float | None = None,
) -> LocalPairNaturalOrbitals:
    """Diagonalize a contravariant pair density in its local overlap metric.

    The density has trace Tr(P S); the generalized eigenvectors satisfy
    C.T S C = I. The returned coefficient rows use the persistent domain's
    orbital indices and are zero outside that domain by construction.
    """
    density = np.asarray(pair_density, dtype=float)
    metric = np.asarray(overlap, dtype=float)
    indices = np.asarray(domain_orbitals, dtype=np.intp)
    if (
        density.ndim != 2 or density.shape[0] != density.shape[1]
        or metric.shape != density.shape or indices.ndim != 1
        or not len(indices) or len(np.unique(indices)) != len(indices)
        or np.any(indices < 0) or np.any(indices >= len(density))
        or occupation_cutoff < 0.0
        or maximum_discarded_fraction is not None
        and not 0.0 <= maximum_discarded_fraction <= 1.0
    ):
        raise ValueError("invalid correlated pair density/domain")
    local_density = density[np.ix_(indices, indices)]
    local_metric = metric[np.ix_(indices, indices)]
    outside = np.ones(len(density), dtype=bool)
    outside[indices] = False
    if (
        not np.allclose(local_density, local_density.T, atol=1.0e-12)
        or not np.allclose(local_metric, local_metric.T, atol=1.0e-12)
        or np.max(np.abs(density[outside, :]), initial=0.0) > 1.0e-10
        or np.max(np.abs(density[:, outside]), initial=0.0) > 1.0e-10
    ):
        raise ValueError("pair density must be symmetric and supported on its domain")
    values, vectors = eigh(
        local_metric @ local_density @ local_metric, local_metric,
    )
    if np.min(values) < -1.0e-9:
        raise ValueError("correlated pair density is not positive semidefinite")
    values = np.maximum(values, 0.0)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    total = float(np.sum(values))
    keep = values >= occupation_cutoff
    if maximum_discarded_fraction is not None and total > 0:
        discarded = float(np.sum(values[~keep]))
        for index in range(len(values)):
            if keep[index]:
                continue
            if discarded / total <= maximum_discarded_fraction:
                break
            keep[index] = True
            discarded -= values[index]
    return LocalPairNaturalOrbitals(
        indices.copy(), vectors[:, keep], values[keep],
        float(np.sum(values[~keep])), total,
    )
