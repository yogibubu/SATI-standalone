"""Higher order rotation couplings for any number of internal coordinates.

The three operators are {sum_mu (p_mu-rho_mu.J)^2, B}, with B equal to
J^2, J_a^2, or J_b^2-J_c^2. The caller supplies projected DVR matrices;
there is no molecule, torsional periodicity, or symmetry label in this module.
"""

from __future__ import annotations

import numpy as np


def project_covariant_momentum_matrices(
    vectors: np.ndarray,
    rho: np.ndarray,
    momentum_action,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project p², {p,rho}, and rho-rho for arbitrary LAM coordinates.

    ``rho`` has shape (nprimitive, 3, ncoord); ``momentum_action`` applies
    the DVR momentum for one coordinate to a matrix of primitive vectors.
    """
    u = np.asarray(vectors, dtype=complex)
    field = np.asarray(rho)
    if u.ndim != 2 or field.ndim != 3 or field.shape[:2] != (u.shape[0], 3):
        raise ValueError("vectors and rho have incompatible primitive dimensions")
    ncoord = field.shape[2]
    nstate = u.shape[1]
    p2 = np.empty((ncoord, nstate, nstate), dtype=complex)
    prho = np.empty((ncoord, 3, nstate, nstate), dtype=complex)
    rho2 = np.empty((ncoord, 3, 3, nstate, nstate), dtype=complex)
    for mu in range(ncoord):
        pu = momentum_action(mu, u)
        p2[mu] = u.conj().T @ momentum_action(mu, pu)
        for a in range(3):
            value = field[:, a, mu]
            prho[mu, a] = u.conj().T @ (
                momentum_action(mu, value[:, None] * u) + value[:, None] * pu
            )
            for b in range(3):
                rho2[mu, a, b] = u.conj().T @ (
                    (value * field[:, b, mu])[:, None] * u
                )
    return p2, prho, rho2


def pi2_rotation_factors(
    p2: np.ndarray,
    p_rho_anticommutator: np.ndarray,
    rho_products: np.ndarray,
    angular_momenta: tuple[np.ndarray, np.ndarray, np.ndarray],
    rho_scales: np.ndarray,
    *, _operator_algebra=False,
) -> tuple[tuple[tuple, ...], tuple[tuple[tuple, ...], ...]]:
    """Return factors and derivatives with respect to three rho scales.

    The arrays are projected <p_mu^2>, <{p_mu,rho_a,mu}> and
    <rho_a,mu rho_b,mu>, with leading coordinate and Cartesian axes.
    Each factor is (coefficient, LAM matrix, rotational matrix).
    """
    p2 = np.asarray(p2)
    prho = np.asarray(p_rho_anticommutator)
    rho2 = np.asarray(rho_products)
    scales = np.asarray(rho_scales)
    if p2.ndim != 3 or p2.shape[1] != p2.shape[2]:
        raise ValueError("p2 must have shape (ncoord,nstate,nstate)")
    ncoord, nstate, _ = p2.shape
    if prho.shape != (ncoord, 3, nstate, nstate):
        raise ValueError("p-rho anticommutators have incompatible shape")
    if rho2.shape != (ncoord, 3, 3, nstate, nstate):
        raise ValueError("rho products have incompatible shape")
    if scales.shape != (3,) or len(angular_momenta) != 3:
        raise ValueError("three angular momenta and rho scales are required")
    rot = tuple(angular_momenta) if _operator_algebra else tuple(np.asarray(value) for value in angular_momenta)
    if not _operator_algebra and (
        any(value.shape != rot[0].shape for value in rot)
        or rot[0].ndim != 2
        or rot[0].shape[0] != rot[0].shape[1]
    ):
        raise ValueError("angular momenta must be equally sized square matrices")
    r2 = [value @ value for value in rot]
    bases = [sum(r2), r2[0], r2[1] - r2[2]]
    families = [[], [], []]
    derivatives = [[[], [], []] for _ in range(3)]
    p2_total = np.sum(p2, axis=0)
    prho_total = np.sum(prho, axis=0)
    rho2_total = np.sum(rho2, axis=0)
    for k, base in enumerate(bases):
        families[k].append((2.0, p2_total, base))
        for a in range(3):
            anticommutator = rot[a] @ base + base @ rot[a]
            families[k].append((-float(scales[a]), prho_total[a], anticommutator))
            derivatives[a][k].append((-1.0, prho_total[a], anticommutator))
            for b in range(3):
                product = rot[a] @ rot[b]
                coupled = product @ base + base @ product
                field = rho2_total[a, b]
                families[k].append((float(scales[a] * scales[b]), field, coupled))
                for c in range(3):
                    coefficient = (scales[b] if a == c else 0.0) + (
                        scales[a] if b == c else 0.0
                    )
                    if coefficient:
                        derivatives[c][k].append((float(coefficient), field, coupled))
    return (
        tuple(tuple(family) for family in families),
        tuple(tuple(tuple(family) for family in derivative) for derivative in derivatives),
    )


def assemble_contracted_factors(
    nstate: int,
    nrot: int,
    factors: tuple[tuple, ...],
) -> np.ndarray:
    """Assemble in LOCK's LAM-major product basis, using C++ when available."""
    from .lam_rotation import _lock_native

    if _lock_native is not None and hasattr(_lock_native, "assemble_ram_factorized"):
        identity = np.broadcast_to(np.eye(nstate, dtype=complex), (nrot, nstate, nstate))
        result = _lock_native.assemble_ram_factorized(identity, factors)
        return result.reshape(nrot, nstate, nrot, nstate).transpose(1, 0, 3, 2).reshape(
            nstate * nrot, nstate * nrot
        )
    result = np.zeros((nstate * nrot, nstate * nrot), dtype=complex)
    for coefficient, lam, rot in factors:
        result += coefficient * np.kron(lam, rot)
    return result
