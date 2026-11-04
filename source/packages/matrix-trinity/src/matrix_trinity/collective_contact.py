"""Symmetric collective contact closure on a fixed oriented P/Q partition.

Additional approximation for property functions, not conventional scalar
HDCPT2 and not an invariant-graph solver. No physical-Hamiltonian eigensolve
or post-variational refinement. See COLLECTIVE_CONTACT_CLOSURE.md for proofs,
branch orientation, lost finite-gap DC exactness, and validation status.
"""
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .collective_gram import CollectiveGramResult, _hermitian, _positive_function


@dataclass(frozen=True)
class CollectiveContactResult(CollectiveGramResult):
    """Reuse graph normalization, general properties and subspace interfaces."""


def collective_contact_graph(a, d, b, *, maximum_workspace_bytes=536870912):
    """Construct X = -[ {L,K^-1}/2 + 8 K^-3/4 M K^-3/4 ] B.

    L(Y)=DY-YA, M(Y)=(BB†Y+YB†B)/2, K=L²+4M on coherence space.
    All inputs use one energy unit; B has shape (q,p). P/Q orientation and
    ambient representation are fixed input data. Arbitrary signs and unequal
    gaps are allowed. K>0 is required; a null kernel indicates dark physical
    degeneracy in exact arithmetic, or unresolved conditioning numerically.

    The symmetric even term is the explicit new interpolation assumption. Its
    coefficient follows from requiring X=-polar(B) at a common zero gap.
    There is no fitted width. Exact DC functions at finite gap are not retained.
    Dense reference prototype: O((p*q)**2) storage, cubic auxiliary root work.
    """
    a, d = _hermitian(a, "a"), _hermitian(d, "d")
    b = np.asarray(b, dtype=complex)
    p, q = a.shape[0], d.shape[0]
    if b.shape != (q, p) or not np.all(np.isfinite(b)):
        raise ValueError("b must be finite with shape (q,p)")
    if (isinstance(maximum_workspace_bytes, (bool, np.bool_))
            or not isinstance(maximum_workspace_bytes, (int, np.integer))
            or maximum_workspace_bytes <= 0):
        raise ValueError("maximum_workspace_bytes must be a positive integer")
    k = p * q
    estimated_bytes = 32 * 16 * (k * k + (p + q) ** 2)
    if estimated_bytes > maximum_workspace_bytes:
        raise MemoryError(f"collective workspace estimate {estimated_bytes} exceeds budget")
    origin = float(np.max(np.linalg.eigvalsh(a)))
    ip, iq = np.eye(p), np.eye(q)
    aa, dd = a - origin * ip, d - origin * iq
    scale = max(float(np.linalg.norm(aa, 2)), float(np.linalg.norm(dd, 2)),
                float(np.linalg.norm(b, 2)))
    if not np.isfinite(scale) or scale == 0:
        raise FloatingPointError("zero or unresolved collective energy scale; degenerate block data")
    # Work in dimensionless units; the graph is invariant under a positive
    # rescaling and under shifts of the common energy origin.
    ascaled, dscaled, bs = aa / scale, dd / scale, b / scale
    ell = np.kron(ip, dscaled) - np.kron(ascaled.T, iq)
    gram = (np.kron(ip, bs @ bs.conj().T)
            + np.kron((bs.conj().T @ bs).T, iq)) / 2
    kernel = ell @ ell + 4 * gram
    kernel = (kernel + kernel.conj().T) / 2
    eigenvalues, vectors = np.linalg.eigh(kernel)
    roundoff = 64 * np.finfo(float).eps * k * float(np.max(np.abs(eigenvalues)))
    if np.min(eigenvalues) <= roundoff:
        raise FloatingPointError(
            "singular or numerically unresolved collective contact kernel; "
            "do not substitute a range inverse for degenerate-subspace data")

    def action(power, value):
        return vectors @ ((eigenvalues ** power) * (vectors.conj().T @ value))

    source = bs.reshape(k, order="F")
    first = (ell @ action(-1, source) + action(-1, ell @ source)) / 2
    resonant = 8 * action(-.75, gram @ action(-.75, source))
    x = -(first + resonant).reshape((q, p), order="F")
    c = _positive_function(ip + x.conj().T @ x, inverse=True)
    f = _positive_function(iq + x @ x.conj().T, inverse=True)
    unitary = np.block([[c, -x.conj().T @ f], [x @ c, f]])
    residual = b + dd @ x - x @ aa - x @ b.conj().T @ x
    diagnostics = MappingProxyType({
        "approximation": "symmetric_collective_contact_v1",
        "reference": "fixed_oriented_pq_partition",
        "spectral_branch_selection": False,
        "conventional_scalar_hdc_equivalence": False,
        "finite_gap_dc_exactness": False,
        "p_dimension": p, "q_dimension": q, "coherence_dimension": k,
        "energy_origin": origin, "energy_scale": scale,
        "scaled_kernel_minimum": float(np.min(eigenvalues)),
        "kernel_condition_estimate": float(np.max(eigenvalues) / np.min(eigenvalues)),
        "kernel_roundoff_scale": roundoff,
        "workspace_estimate_bytes": estimated_bytes,
        "riccati_residual_frobenius": float(np.linalg.norm(residual, "fro")),
        "hamiltonian_defect_spectral": float(np.linalg.norm(f @ residual @ c, 2)),
        "unitarity_defect_frobenius": float(np.linalg.norm(
            unitary.conj().T @ unitary - np.eye(p + q), "fro")),
        "physical_validation": "not_performed",
    })
    x.setflags(write=False)
    unitary.setflags(write=False)
    return CollectiveContactResult(x, unitary, diagnostics)
