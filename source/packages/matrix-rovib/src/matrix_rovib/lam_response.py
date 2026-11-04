"""Factorized operator actions and analytic variational-subspace response."""

from __future__ import annotations
import numpy as np
from .lam_errors import LAMTruncationError
from scipy.linalg import qr, solve_triangular


def apply_factors(factors, columns, nlam, nrot):
    values = np.asarray(columns)
    vector = values.ndim == 1
    tensor = values.reshape(nlam, nrot, -1)
    out = np.zeros_like(tensor, dtype=complex)
    for left, right in factors:
        if not np.any(left) or not np.any(right):
            continue
        acted = (left @ tensor.reshape(nlam, -1)).reshape(tensor.shape)
        out += np.einsum("rs,asl->arl", right, acted, optimize=True)
    result = out.reshape(nlam * nrot, -1)
    return result[:, 0] if vector else result


def _closed_span_response(raw, response, action):
    """Differentiate a constant-rank finite-group closure with bounded memory."""
    q, dq = raw, response
    front, dfront = q, dq
    generators = action.generator_indices or range(len(action.operations))
    while front.shape[1] and q.shape[1] < len(q):
        count = q.shape[1]
        for index in generators:
            a = action.operations[index]
            candidate = a @ front
            residual = candidate - q @ (q.conj().T @ candidate)
            _, r, pivot = qr(residual, mode="economic", pivoting=True)
            rank = int(np.sum(abs(np.diag(r)) > 1.0e-10))
            if not rank:
                continue
            selected = pivot[:rank]
            b = np.column_stack((q, candidate[:, selected]))
            db = np.column_stack((dq, (a @ dfront)[:, selected]))
            q, r = np.linalg.qr(b, mode="reduced")
            z = solve_triangular(r.T, db.T, lower=True).T
            c = q.conj().T @ z
            lower = np.tril(c, -1)
            omega = lower - lower.conj().T
            dq = z - q @ c + q @ omega
        front, dfront = q[:, count:], dq[:, count:]
    return q, dq


def k_contraction_response(contraction, factors, derivative_factors, vectors):
    """Return the normal derivative of the selected K subspace on Ritz states.

    Only retained/discarded gaps enter. Degeneracy within a retained manifold
    requires no phase choice; a degenerate cut is rejected. Group closure is
    differentiated using the full supplied finite-group action.
    """
    nrot = 2 * contraction.J + 1
    response = np.zeros_like(vectors, dtype=complex)
    for sub in contraction.subspaces:
        indices = sub.lam_indices
        size = len(indices)
        local = [(a[np.ix_(indices, indices)], b) for a, b in factors]
        local_d = [(a[np.ix_(indices, indices)], b) for a, b in derivative_factors]
        raw = np.zeros((size * nrot, sum(u.shape[1] for u in sub.vectors_by_k)), complex)
        draw = np.zeros_like(raw)
        offset = 0
        for k, u in enumerate(sub.vectors_by_k):
            keep = u.shape[1]
            columns = slice(offset, offset + keep)
            raw[k::nrot, columns] = u
            offset += keep
            if keep == size:
                continue
            d = sum((b[k, k] * a for a, b in local), start=np.zeros((size, size), complex))
            d = (d + d.conj().T) / 2
            dd = sum((b[k, k] * a for a, b in local_d), start=np.zeros((size, size), complex))
            dd = (dd + dd.conj().T) / 2
            energies, all_vectors = np.linalg.eigh(d - np.trace(d).real / size * np.eye(size))
            discarded = all_vectors[:, keep:]
            gaps = energies[:keep][None, :] - energies[keep:, None]
            if np.min(abs(gaps)) < max(1.0e-8, 256 * np.spacing(max(abs(energies), default=0.0))):
                raise LAMTruncationError("k_contraction",
                    "moving K contraction cuts a degenerate manifold; increase retained states"
                )
            # Use retained vectors as stored by the production diagonalizer.
            retained_energy = np.real(np.diag(u.conj().T @ d @ u))
            discarded_energy = np.real(np.diag(discarded.conj().T @ d @ discarded))
            tangent = discarded @ (
                (discarded.conj().T @ dd @ u)
                / (retained_energy[None, :] - discarded_energy[:, None])
            )
            draw[k::nrot, columns] = tangent
        if sub.symmetry_action is not None:
            if contraction.state_action is None:
                raise RuntimeError("missing full molecular action for contraction response")
            raw, draw = _closed_span_response(raw, draw, contraction.state_action)
        rows = (indices[:, None] * nrot + np.arange(nrot)).reshape(-1)
        normal = draw - raw @ (raw.conj().T @ draw)
        response[rows] = normal @ (raw.conj().T @ vectors[rows])
    return response


def moving_ritz_derivatives(contraction, factors, derivatives, energies, vectors):
    nrot = 2 * contraction.J + 1
    nlam = len(vectors) // nrot
    acted = apply_factors(factors, vectors, nlam, nrot)
    residual = acted - vectors * np.asarray(energies)[None, :]
    dpsi = k_contraction_response(contraction, factors, derivatives, vectors)
    direct = apply_factors(derivatives, vectors, nlam, nrot)
    return np.real(
        np.sum(vectors.conj() * direct, axis=0) + 2 * np.sum(dpsi.conj() * residual, axis=0)
    )
