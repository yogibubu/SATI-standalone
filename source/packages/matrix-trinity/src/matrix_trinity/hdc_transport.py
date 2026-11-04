"""Overlap transport of HDC branches and collective spectral subspaces.

Rotating a nondegenerate group also rotates its energy matrix. Its diagonal
must never be substituted for its eigenvalues or for the prescribed X model.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import svd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


@dataclass(frozen=True)
class HDCBranchFrame:
    vectors: np.ndarray
    energy_matrix_cm: np.ndarray
    rotation: np.ndarray
    assignment: np.ndarray
    blocks: tuple
    diagnostics: dict


def _frame(energies, vectors):
    e = np.asarray(energies, float)
    u = np.asarray(vectors, complex)
    if (e.ndim != 1 or not len(e) or u.ndim != 2 or u.shape[1] != len(e)
            or not np.all(np.isfinite(e)) or not np.all(np.isfinite(u))
            or not np.allclose(u.conj().T@u, np.eye(len(e)), rtol=0, atol=1e-9)):
        raise ValueError("finite energies and orthonormal frame columns required")
    return e, u


def initialize_hdc_branches(energies_cm, vectors):
    e, u = _frame(energies_cm, vectors)
    k = len(e)
    return HDCBranchFrame(u.copy(), np.diag(e), np.eye(k, dtype=complex),
        np.arange(k), tuple((i,) for i in range(k)),
        dict(initialized=True, individual_energy_diagonals_are_eigenvalues=True))


def transport_hdc_branches(previous, energies_cm, vectors, *, basis_overlap=None,
                           minimum_subspace_overlap=.9, mixing_threshold=.1,
                           energy_degeneracy_tolerance_cm=1e-8):
    """Match branches, merge ambiguous groups, then polar-transport each block.

    Thresholds are declared numerical controls, independent of the molecule.
    Existing blocks persist along a path. Poor principal overlaps reject the
    step (refine the path or enlarge the retained root pool); no interpolation
    of energies or empirical intensity correction is performed.
    """
    if not isinstance(previous, HDCBranchFrame):
        raise TypeError("HDCBranchFrame required")
    if (not 0 < minimum_subspace_overlap <= 1 or not 0 < mixing_threshold < 1
            or not np.isfinite(energy_degeneracy_tolerance_cm)
            or energy_degeneracy_tolerance_cm < 0):
        raise ValueError("invalid branch transport controls")
    e, u = _frame(energies_cm, vectors)
    old = previous.vectors
    k = old.shape[1]
    if len(e) < k:
        raise ValueError("current root pool is smaller than the retained branches")
    if basis_overlap is None:
        if old.shape[0] != u.shape[0]:
            raise ValueError("different coordinate bases require their physical overlap")
        overlap = old.conj().T@u
    else:
        s = np.asarray(basis_overlap, complex)
        if s.shape != (old.shape[0], u.shape[0]) or not np.all(np.isfinite(s)):
            raise ValueError("finite physical basis-overlap matrix required")
        overlap = old.conj().T@s@u
    if np.linalg.norm(overlap, 2) > 1+1e-8:
        raise ValueError("basis overlap is not a contraction between orthonormal spaces")
    _, assignment = linear_sum_assignment(-abs(overlap)**2)
    selected = overlap[:, assignment]
    energies = e[assignment]
    graph = np.eye(k, dtype=bool)
    for block in previous.blocks:
        graph[np.ix_(block, block)] = True
    weights = abs(selected)**2
    for i in range(k):
        for j in range(i):
            mix = max(weights[i,j], weights[j,i])
            ambiguous = mix >= mixing_threshold*max(weights[i,i], weights[j,j], 1e-30)
            degenerate = abs(energies[i]-energies[j]) <= energy_degeneracy_tolerance_cm
            if ambiguous or degenerate:
                graph[i,j] = graph[j,i] = True
    count, labels = connected_components(csr_matrix(graph), directed=False)
    blocks = tuple(tuple(np.flatnonzero(labels == i)) for i in range(count))
    align = np.zeros((k,k), complex)
    minimum = 1.
    for block in blocks:
        ids = np.array(block)
        left, values, right = svd(selected[np.ix_(ids,ids)], full_matrices=False)
        minimum = min(minimum, float(np.min(values)**2))
        align[np.ix_(ids,ids)] = right.conj().T@left.conj().T
    if minimum < minimum_subspace_overlap:
        raise ValueError(f"branch step unresolved: squared principal overlap {minimum:.6g}; refine path or enlarge root pool")
    rotation = np.zeros((len(e),k), complex)
    rotation[assignment] = align
    h = rotation.conj().T@(e[:,None]*rotation)
    return HDCBranchFrame(u@rotation, h, rotation, assignment, blocks, dict(
        schema="matrix.trinity.hdc_branch_transport.v1",
        minimum_squared_principal_overlap=minimum,
        eigenvalue_preservation_error_cm=float(np.max(abs(np.linalg.eigvalsh(h)-np.sort(energies)))),
        offdiagonal_energy_norm_cm=float(np.linalg.norm(h-np.diag(np.diag(h)))),
        individual_energy_diagonals_are_eigenvalues=all(len(b)==1 for b in blocks),
        energy_coefficients_modified=False, previous_blocks_retained=True))


def transport_hdc_operator(operator, final, initial=None):
    """Transform an eigenbasis property, retaining arbitrary component axes."""
    initial = final if initial is None else initial
    if not isinstance(final,HDCBranchFrame) or not isinstance(initial,HDCBranchFrame):
        raise TypeError("HDCBranchFrame endpoints required")
    a = np.asarray(operator)
    if a.ndim < 2 or a.shape[:2] != (final.rotation.shape[0],initial.rotation.shape[0]) or not np.all(np.isfinite(a)):
        raise ValueError("finite operator with matching eigenbasis axes required")
    return np.einsum("am,ab...,bn->mn...", final.rotation.conj(),a,initial.rotation,optimize=True)
