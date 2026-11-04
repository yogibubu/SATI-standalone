"""Untruncated sparse algebra audit of an orthogonal-basis SCF projector.

This checks every entry created by sparse products, including entries outside
the retained pair domain. It does not bound AO integrals never constructed by
the sparse model; a physical full-AO certificate needs a separate tail bound.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


@dataclass(frozen=True)
class SparseProjectorAudit:
    maximum_commutator: float
    maximum_idempotency_error: float
    electron_count_error: float
    truncation_bound: float
    commutator: csr_matrix


def _symmetric_csr(size, rows, columns, values) -> csr_matrix:
    lower = np.asarray(rows, dtype=np.int64)
    upper = np.asarray(columns, dtype=np.int64)
    data = np.asarray(values, dtype=float)
    if (size < 1 or lower.ndim != 1 or upper.shape != lower.shape
            or data.shape != lower.shape or np.any(lower < upper)
            or np.any(lower >= size) or np.any(upper < 0)
            or not np.all(np.isfinite(data))):
        raise ValueError("invalid pair-packed sparse matrix")
    off = lower != upper
    return coo_matrix((
        np.concatenate((data, data[off])),
        (np.concatenate((lower, upper[off])),
         np.concatenate((upper, lower[off]))),
    ), shape=(size, size)).tocsr()


def audit_sparse_projector(state) -> SparseProjectorAudit:
    """Audit all product fill without materializing an AO square.

    The tested equations are [F', P'] = 0, P'^2 = P', and 2 tr(P') = N.
    The stored F' and P' are those actually used by the sparse SCF.
    """
    size = int(state.size)
    projector = _symmetric_csr(
        size, state.projector_rows, state.projector_columns,
        state.projector_values,
    )
    fock = _symmetric_csr(
        size, state.fock_rows, state.fock_columns, state.fock_values,
    )
    commutator = fock @ projector - projector @ fock
    idempotency = projector @ projector - projector
    commutator.eliminate_zeros()
    idempotency.eliminate_zeros()
    bound = float(state.truncation_bound)
    if not np.isfinite(bound) or bound < 0.0:
        raise ValueError("invalid sparse projector truncation bound")
    return SparseProjectorAudit(
        float(np.max(abs(commutator.data), initial=0.0)),
        float(np.max(abs(idempotency.data), initial=0.0)),
        abs(2.0 * float(projector.diagonal().sum())
            - 2.0 * int(state.occupied_orbitals)),
        bound, commutator,
    )


__all__ = ["SparseProjectorAudit", "audit_sparse_projector"]
