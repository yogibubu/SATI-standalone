"""Collective Gram approximation including interleaved block spectra.

Research prototype: the operator prescription is additional to scalar HDCPT2.
It constructs no physical-Hamiltonian eigenstates and performs no variational
refinement. Dense auxiliary matrix functions limit its usable block size.
See docs/science/COLLECTIVE_GRAM_APPROXIMATION.md for domain and derivation.
"""
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np


def _hermitian(value, name):
    array = np.asarray(value, dtype=complex)
    if array.ndim != 2 or array.shape[0] != array.shape[1] or not array.size:
        raise ValueError(f"{name} must be a nonempty square matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    error = np.linalg.norm(array - array.conj().T, ord="fro")
    tolerance = 64 * np.finfo(float).eps * max(1, array.shape[0]) * np.linalg.norm(array, ord="fro")
    if error > tolerance:
        raise ValueError(f"{name} must be Hermitian")
    return (array + array.conj().T) / 2


def _positive_function(matrix, inverse=False):
    values, vectors = np.linalg.eigh((matrix + matrix.conj().T) / 2)
    if np.any(values <= 0):
        raise FloatingPointError("normalization Gram matrix lost positive definiteness")
    powers = 1 / np.sqrt(values) if inverse else np.sqrt(values)
    return (vectors * powers) @ vectors.conj().T


@dataclass(frozen=True)
class CollectiveGramResult:
    """One common approximate transformation; energies are not prescribed here."""
    graph: np.ndarray
    unitary: np.ndarray
    diagnostics: object

    def subspace(self, *, block, coordinate_id):
        """Expose a declared cluster to chart/transport operations without a new solve."""
        from .collective_subspaces import CollectiveSubspaceFrame
        return CollectiveSubspaceFrame.from_collective_result(
            self, block=block, coordinate_id=coordinate_id)

    def transform_property(self, operator):
        """Transform scalar or stacked property components, preserving interference.

        Last two axes have the full P+Q dimension. Non-Hermitian transition
        operators are allowed. No photon factor, population or energy is added.
        """
        value = np.asarray(operator, dtype=complex)
        size = self.unitary.shape[0]
        if value.ndim < 2 or value.shape[-2:] != (size, size):
            raise ValueError("property must have full P+Q matrix dimensions")
        if not np.all(np.isfinite(value)):
            raise ValueError("property must be finite")
        return self.unitary.conj().T @ value @ self.unitary


def collective_gram_graph(a, d, b, *, maximum_workspace_bytes=536870912,
                          require_regular_chart=True):
    """Approximate B + D X - X A - X B† X = 0, including signed gaps.

    a: (p,p), d: (q,q) Hermitian energies in one common unit.
    b: (q,p) couplings in that unit. No block energy ordering is required.
    On the open domain T=L+sqrt(L²+4M)>0 the transformation is globally
    single-valued and continuous, including gap sign changes. Full rectangular
    rank of B is sufficient, not necessary, for T>0.

    The common-gap branch follows the lower bright subspace, not the original
    bare-state label on both sides of an avoided crossing. No physical resonance
    threshold is used. Numerical kernel decisions use backward-error scales.
    Singular graph charts are rejected by default: their range inverse is not a
    globally continuous extension. require_regular_chart=False explicitly allows
    that pointwise range choice, with a diagnostic denying continuity coverage.

    The dense coherence-space square root uses O((p*q)**2) storage and cubic
    work. The budget is a conservative array estimate, not a process RSS limit.
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
    if not isinstance(require_regular_chart, (bool, np.bool_)):
        raise ValueError("require_regular_chart must be boolean")
    k = p * q
    estimated_bytes = 32 * 16 * (k * k + (p + q) ** 2)
    if estimated_bytes > maximum_workspace_bytes:
        raise MemoryError(f"collective workspace estimate {estimated_bytes} exceeds budget")
    a_values, d_values = np.linalg.eigvalsh(a), np.linalg.eigvalsh(d)
    separation = float(d_values[0] - a_values[-1])
    # Shift the common energy origin before forming a Sylvester difference.
    origin = float(a_values[-1])
    aa, dd = a - origin * np.eye(p), d - origin * np.eye(q)
    energy_scale = max(float(np.linalg.norm(aa, 2)), float(np.linalg.norm(dd, 2)),
                       float(np.linalg.norm(b, 2)))
    eps = np.finfo(float).eps
    identity_p, identity_q = np.eye(p), np.eye(q)
    sylvester = np.kron(identity_p, dd) - np.kron(aa.T, identity_q)
    gram = (np.kron(identity_p, b @ b.conj().T)
            + np.kron((b.conj().T @ b).T, identity_q)) / 2
    square = sylvester @ sylvester + 4 * gram
    square = (square + square.conj().T) / 2
    values, vectors = np.linalg.eigh(square)
    square_tolerance = 64 * eps * k * float(np.linalg.norm(square, 2))
    if np.min(values) < -square_tolerance:
        raise FloatingPointError("collective squared kernel lost positive semidefiniteness")
    root = (vectors * np.sqrt(np.maximum(values, 0))) @ vectors.conj().T
    denominator = (sylvester + root)
    denominator = (denominator + denominator.conj().T) / 2
    values, vectors = np.linalg.eigh(denominator)
    solve_tolerance = 64 * eps * k * max(float(np.max(np.abs(values))), energy_scale)
    if np.min(values) < -solve_tolerance:
        raise FloatingPointError("collective denominator lost positive semidefiniteness")
    active = values > solve_tolerance
    regular = bool(np.all(active))
    if require_regular_chart and not regular:
        raise FloatingPointError(
            "singular or numerically unresolved collective graph chart; "
            "a continuous basis requires subspace/branch data, not a range-inverse reset")
    source = b.reshape(k, order="F")
    projected = vectors.conj().T @ source
    discarded = float(np.linalg.norm(projected[~active]))
    source_tolerance = 256 * eps * max(1, k) * float(np.linalg.norm(source))
    if discarded > source_tolerance:
        raise FloatingPointError("unresolved source in numerical kernel; rescale or increase precision")
    solution = np.zeros(k, dtype=complex)
    solution[active] = -2 * projected[active] / values[active]
    x = (vectors @ solution).reshape((q, p), order="F")
    c = _positive_function(identity_p + x.conj().T @ x, inverse=True)
    f = _positive_function(identity_q + x @ x.conj().T, inverse=True)
    unitary = np.block([[c, -x.conj().T @ f], [x @ c, f]])
    riccati = b + dd @ x - x @ aa - x @ b.conj().T @ x
    off_diagonal = f @ riccati @ c
    defect = float(np.linalg.norm(off_diagonal, 2))
    diagnostics = MappingProxyType({
        "approximation": "collective_gram_square_root_v2",
        "branch": "common_gap_lower_bright_continuation",
        "regular_chart": regular,
        "continuity_domain": "positive_collective_denominator" if regular else "pointwise_range_choice",
        "minimum_denominator_eigenvalue": float(np.min(values)),
        "denominator_condition_estimate": float(np.max(values) / np.min(values)) if regular else float("inf"),
        "p_dimension": p, "q_dimension": q, "coherence_dimension": k,
        "ordered_gap": separation, "energy_origin": origin,
        "workspace_estimate_bytes": estimated_bytes,
        "numerical_kernel_dimension": int(k - np.count_nonzero(active)),
        "solve_roundoff_scale": solve_tolerance,
        "discarded_source_norm": discarded,
        "riccati_residual_frobenius": float(np.linalg.norm(riccati, "fro")),
        "hamiltonian_defect_spectral": defect,
        "unitarity_defect_frobenius": float(np.linalg.norm(
            unitary.conj().T @ unitary - np.eye(p + q), "fro")),
        "physical_validation": "not_performed",
    })
    x.setflags(write=False)
    unitary.setflags(write=False)
    return CollectiveGramResult(x, unitary, diagnostics)
