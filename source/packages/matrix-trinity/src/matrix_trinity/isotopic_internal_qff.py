from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NonredundantInternalCubicField:
    """Mass-independent harmonic/cubic field in nonredundant internals."""

    harmonic_internal: np.ndarray
    cubic_internal: np.ndarray
    parent_d_condition: float
    coordinate_count: int
    representation: str = "nonredundant-internal-b-bprime"


def internal_cubic_from_parent_normal_field(
    harmonic_parent_qmw,
    cubic_parent_qmw,
    b_matrix,
    b_prime,
    cartesian_per_parent_q,
    *,
    rank_tolerance: float = 1.0e-10,
) -> NonredundantInternalCubicField:
    """Recover a unique internal F2/F3 from one parent normal QFF.

    Only a square, full-rank nonredundant coordinate basis is accepted.  No
    pseudoinverse or redundant-coordinate gauge is introduced.
    """

    phi2 = np.asarray(harmonic_parent_qmw, dtype=float)
    phi3 = np.asarray(cubic_parent_qmw, dtype=float)
    b = np.asarray(b_matrix, dtype=float)
    bp = np.asarray(b_prime, dtype=float)
    c = np.asarray(cartesian_per_parent_q, dtype=float)
    ncoord, ncart = b.shape
    if c.shape != (ncart, ncoord):
        raise ValueError("nonredundant transform requires 3N x (3N-6) parent modes")
    if phi2.shape != (ncoord, ncoord) or phi3.shape != (ncoord, ncoord, ncoord):
        raise ValueError("parent normal force field and nonredundant basis disagree")
    if bp.shape != (ncoord, ncart, ncart):
        raise ValueError("B-prime must have shape ncoord x 3N x 3N")
    d = b @ c
    singular = np.linalg.svd(d, compute_uv=False)
    threshold = float(rank_tolerance) * max(float(singular[0]), 1.0)
    rank = int(np.count_nonzero(singular > threshold))
    if rank != ncoord:
        raise ValueError(
            f"SONIC normal-coordinate Jacobian has rank {rank}, expected {ncoord}; "
            "redundant/pseudoinverse transformation is forbidden"
        )
    inverse_d = np.linalg.solve(d, np.eye(ncoord))
    force2 = inverse_d.T @ phi2 @ inverse_d
    e = np.einsum("iab,ap,bq->ipq", bp, c, c, optimize=True)
    curvature = _normal_cubic_curvature(force2, d, e)
    rectilinear = phi3 - curvature
    force3 = np.einsum(
        "abc,ai,bj,ck->ijk",
        rectilinear,
        inverse_d,
        inverse_d,
        inverse_d,
        optimize=True,
    )
    force3 = _symmetrize_rank3(force3)
    return NonredundantInternalCubicField(
        harmonic_internal=0.5 * (force2 + force2.T),
        cubic_internal=force3,
        parent_d_condition=float(singular[0] / singular[-1]),
        coordinate_count=ncoord,
    )


def normal_cubic_from_internal_field(
    field: NonredundantInternalCubicField,
    b_matrix,
    b_prime,
    cartesian_per_target_q,
) -> tuple[np.ndarray, np.ndarray]:
    """Express a mass-independent internal F2/F3 in target-isotope modes."""

    b = np.asarray(b_matrix, dtype=float)
    bp = np.asarray(b_prime, dtype=float)
    c = np.asarray(cartesian_per_target_q, dtype=float)
    ncoord, ncart = b.shape
    if ncoord != field.coordinate_count or c.shape != (ncart, ncoord):
        raise ValueError("target modes do not match the nonredundant internal field")
    if bp.shape != (ncoord, ncart, ncart):
        raise ValueError("B-prime must have shape ncoord x 3N x 3N")
    d = b @ c
    if np.linalg.matrix_rank(d) != ncoord:
        raise ValueError("target-isotope SONIC Jacobian is rank deficient")
    e = np.einsum("iab,ap,bq->ipq", bp, c, c, optimize=True)
    harmonic = d.T @ field.harmonic_internal @ d
    cubic = np.einsum(
        "ijk,ia,jb,kc->abc",
        field.cubic_internal,
        d,
        d,
        d,
        optimize=True,
    )
    cubic += _normal_cubic_curvature(field.harmonic_internal, d, e)
    return 0.5 * (harmonic + harmonic.T), _symmetrize_rank3(cubic)


def _normal_cubic_curvature(force2: np.ndarray, d: np.ndarray, e: np.ndarray) -> np.ndarray:
    term = np.einsum("ij,iab,jc->abc", force2, e, d, optimize=True)
    return term + term.transpose(0, 2, 1) + term.transpose(2, 1, 0)


def _symmetrize_rank3(tensor: np.ndarray) -> np.ndarray:
    return (
        tensor
        + tensor.transpose(0, 2, 1)
        + tensor.transpose(1, 0, 2)
        + tensor.transpose(1, 2, 0)
        + tensor.transpose(2, 0, 1)
        + tensor.transpose(2, 1, 0)
    ) / 6.0


__all__ = [
    "NonredundantInternalCubicField",
    "internal_cubic_from_parent_normal_field",
    "normal_cubic_from_internal_field",
]
