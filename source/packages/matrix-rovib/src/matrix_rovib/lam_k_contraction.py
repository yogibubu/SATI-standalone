"""General variational J,K contraction of a factorized LAM--rotation model.

All diagonal-K terms of the declared Hamiltonian select the local functions.
All off-diagonal-K terms are then projected without approximation beyond that
explicit basis truncation. A frozen contraction gives a parameter-independent
variational subspace and exact Hellmann--Feynman derivatives during a fit.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np
from matrix_numerics import diagonalize_hermitian


def _basis_digest(basis):
    digest = hashlib.sha256()
    for values in (basis.vectors, basis.energies_MHz, np.asarray(basis.symmetry_sectors, dtype=int)):
        array = np.ascontiguousarray(values)
        digest.update(str((array.shape, array.dtype)).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def _action_digest(action):
    if action is None:
        return None
    digest = hashlib.sha256(repr((action.group.operations, action.group.multiplication_table)).encode())
    for operation in action.operations:
        digest.update(np.ascontiguousarray(operation).tobytes())
    return digest.hexdigest()


def _group_factors(factors):
    grouped = {}
    for left, right in factors:
        left, right = np.asarray(left), np.asarray(right)
        if not np.any(left) or not np.any(right):
            continue
        key = (right.shape, right.dtype.str, right.tobytes())
        if key in grouped:
            grouped[key][0][:] += left
        else:
            grouped[key] = (np.array(left, dtype=complex, copy=True), right)
    return tuple(grouped.values())


def _validate_sector_factors(basis, factors):
    labels = np.asarray(basis.symmetry_sectors, dtype=int)
    forbidden = labels[:, None] != labels[None, :]
    for left, _right in factors:
        scale = max(1., float(np.max(np.abs(left), initial=0.)))
        if np.max(np.abs(left[forbidden]), initial=0.) > 1.e-9*scale:
            raise ValueError("J,K contraction factors mix distinct translation sectors; use the full molecular symmetry action instead")


def _group_closed_span(embedding, action):
    """Close the span with bounded-memory, rank-revealing QR updates."""
    from scipy.linalg import qr

    complete, frontier = embedding, embedding
    generators = action.generator_indices or range(len(action.operations))
    while frontier.shape[1] and complete.shape[1] < len(complete):
        added = []
        for index in generators:
            candidate = action.operations[index] @ frontier
            # Reorthogonalization is essential when an orbit is almost already
            # in the retained span. The input columns have unit scale.
            for _ in range(2):
                candidate -= complete @ (complete.conj().T @ candidate)
            q, r, _ = qr(candidate, mode="economic", pivoting=True, check_finite=False)
            rank = int(np.count_nonzero(np.abs(np.diag(r)) > 1.e-10))
            if rank:
                new = q[:, :rank]
                complete = np.column_stack((complete, new))
                added.append(new)
        frontier = np.column_stack(added) if added else np.empty((len(complete), 0), complex)
    return complete


@dataclass(frozen=True)
class _KSubspace:
    sector: int
    lam_indices: np.ndarray
    vectors_by_k: tuple[np.ndarray, ...]
    embedding: np.ndarray
    symmetry_action: object | None = None

    def project(self, factors):
        """Project sparse rotational words without forming the product matrix."""
        nlam = len(self.lam_indices)
        nrot = len(self.vectors_by_k)
        reduced = self.embedding.shape[1]
        result = np.zeros((reduced, reduced), complex)
        if self.symmetry_action is not None:
            # Group closure can mix K. Apply each factor to all columns using
            # the same tensor order as the production coupled Hamiltonian.
            tensor = self.embedding.reshape(nlam, nrot, reduced)
            for left, right in factors:
                acted = (left @ tensor.reshape(nlam, -1)).reshape(tensor.shape)
                acted = np.einsum("kl,alr->akr", right, acted, optimize=True)
                result += self.embedding.conj().T @ acted.reshape(nlam*nrot, reduced)
            return result
        offsets = np.cumsum([0] + [v.shape[1] for v in self.vectors_by_k])
        packed = np.column_stack(self.vectors_by_k)
        for left, right in factors:
            acted = left @ packed
            rows, columns = np.nonzero(right)
            for row, column in zip(rows, columns, strict=True):
                r = slice(offsets[row], offsets[row+1]); c = slice(offsets[column], offsets[column+1])
                result[r, c] += right[row, column] * (self.vectors_by_k[row].conj().T @ acted[:, c])
        return result


@dataclass(frozen=True)
class LAMKContraction:
    """Reusable contraction at a specified reference Hamiltonian and J."""

    J: int
    states_per_k: int
    basis_digest: str
    subspaces: tuple[_KSubspace, ...]
    action_digest: str | None = None
    state_action: object | None = None

    @property
    def dimension(self):
        return sum(item.embedding.shape[1] for item in self.subspaces)

    @property
    def diagnostics(self):
        return {"J": self.J, "states_per_k_per_sector": self.states_per_k,
            "reduced_dimension": self.dimension,
            "sector_dimensions": [item.embedding.shape[1] for item in self.subspaces],
            "states_by_sector_and_k": [[v.shape[1] for v in item.vectors_by_k] for item in self.subspaces],
            "group_closed": any(item.symmetry_action is not None for item in self.subspaces)}


def prepare_k_contraction(basis, J, factors, states_per_k, *, state_action=None, backend=None):
    if isinstance(states_per_k, (bool, np.bool_)) or not isinstance(states_per_k, (int, np.integer)) or states_per_k < 1:
        raise ValueError("k_contraction_states must be a positive integer")
    factors = _group_factors(factors)
    nrot = 2*J+1
    labels = np.asarray(basis.symmetry_sectors, dtype=int)
    _validate_sector_factors(basis, factors)
    subspaces = []
    for sector in np.unique(labels):
        indices = np.flatnonzero(labels == sector)
        local = [(left[np.ix_(indices, indices)], right) for left, right in factors]
        functions = []
        for k in range(nrot):
            diagonal = sum((right[k, k] * left for left, right in local), start=np.zeros((len(indices), len(indices)), complex))
            diagonal = .5 * (diagonal + diagonal.conj().T)
            diagonal -= np.trace(diagonal).real/len(indices)*np.eye(len(indices))
            # A small local full solve avoids splitting a degenerate boundary.
            solved = diagonalize_hermitian(diagonal, backend=None if backend == "arpack" else backend)
            keep = min(states_per_k, len(indices))
            tolerance = max(1.e-6, 256 * float(np.spacing(float(np.max(np.abs(solved.eigenvalues), initial=0.)))))
            while keep < len(indices) and abs(solved.eigenvalues[keep] - solved.eigenvalues[keep-1]) <= tolerance:
                keep += 1
            functions.append(solved.eigenvectors[:, :keep])
        dimension = sum(v.shape[1] for v in functions)
        embedding = np.zeros((len(indices)*nrot, dimension), complex)
        offset = 0
        for k, vectors in enumerate(functions):
            embedding[k::nrot, offset:offset+vectors.shape[1]] = vectors
            offset += vectors.shape[1]
        projected_action = None
        if state_action is not None:
            if len(indices) != len(labels):
                raise ValueError("a full molecular action must be supplied without independent translation-sector labels")
            # Close the selected span under the complete supplied finite group;
            # no commutativity assumption or special group name is involved.
            embedding = _group_closed_span(embedding, state_action)
            from .rovibrational_symmetry import RovibrationalSymmetryAction
            projected_action = RovibrationalSymmetryAction(state_action.group,
                tuple(embedding.conj().T @ operation @ embedding for operation in state_action.operations),
                state_action.generator_indices)
        for array in (indices, embedding, *functions):
            array.setflags(write=False)
        subspaces.append(_KSubspace(int(sector), indices, tuple(functions), embedding, projected_action))
    return LAMKContraction(J, int(states_per_k), _basis_digest(basis), tuple(subspaces), _action_digest(state_action), state_action)


def solve_k_contracted(basis, J, factors, *, states_per_k, level_count=None, contraction=None, state_action=None, backend=None):
    """Project the complete factor expansion and expand resulting eigenvectors."""
    if contraction is None:
        contraction = prepare_k_contraction(basis, J, factors, states_per_k, state_action=state_action, backend=backend)
    elif (contraction.J != J or contraction.states_per_k != states_per_k
          or contraction.basis_digest != _basis_digest(basis)
          or contraction.action_digest != _action_digest(state_action)):
        raise ValueError("frozen J,K contraction does not match the LAM basis or requested truncation")
    factors = _group_factors(factors)
    _validate_sector_factors(basis, factors)
    records = []; errors = []; backends = set()
    target = contraction.dimension if level_count is None else min(level_count, contraction.dimension)
    for subspace in contraction.subspaces:
        local = [(left[np.ix_(subspace.lam_indices, subspace.lam_indices)], right) for left, right in factors]
        matrix = subspace.project(local)
        errors.append(float(np.max(np.abs(matrix-matrix.conj().T), initial=0.)))
        matrix = .5*(matrix+matrix.conj().T)
        if subspace.symmetry_action is not None:
            blocks = subspace.symmetry_action.diagonalize(matrix)
            entries = sorted((float(energy), b, i) for b, block in enumerate(blocks) for i, energy in enumerate(block.energies))
            energies = np.array([entry[0] for entry in entries])
            vectors = np.column_stack([blocks[b].eigenvectors[:, i] for _, b, i in entries])
            irreps = [blocks[b].irrep for _, b, _ in entries]
            records.append([subspace, matrix, energies, vectors, irreps]); backends.add("symmetry-blocked")
        else:
            records.append([subspace, matrix, None, None, None])
    def diagonalize(record, roots):
        matrix = record[1]
        solved = diagonalize_hermitian(matrix, backend=None if backend == "arpack" else backend,
            subset_by_index=(0, roots-1) if roots < len(matrix) else None)
        record[2:4] = [solved.eigenvalues, solved.eigenvectors]; backends.add(solved.backend)
    for record in records:
        if record[2] is None:
            diagonalize(record, min(len(record[1]), max(2, math.ceil(target/len(records))+2)))
    while True:
        energies = np.sort(np.concatenate([record[2] for record in records]))
        cutoff = energies[min(target, len(energies))-1]
        tolerance = max(1.e-6, 256*float(np.spacing(abs(float(cutoff)))))
        grow = [record for record in records if len(record[2]) < len(record[1])
                and (len(energies) < target or record[2][-1] <= cutoff+tolerance)]
        if not grow: break
        for record in grow:
            diagonalize(record, min(len(record[1]), max(len(record[2])+1, 2*len(record[2]))))
    entries = sorted((float(energy), r, i) for r, record in enumerate(records) for i, energy in enumerate(record[2]) if energy <= cutoff+tolerance)
    expanded = np.zeros((len(basis.energies_MHz)*(2*J+1), len(entries)), complex)
    sectors, irreps = [], []
    for column, (_, r, i) in enumerate(entries):
        subspace, _, _, vectors, labels = records[r]
        rows = (subspace.lam_indices[:, None]*(2*J+1)+np.arange(2*J+1)).reshape(-1)
        expanded[rows, column] = subspace.embedding @ vectors[:, i]
        sectors.append(subspace.sector); irreps.append(None if labels is None else labels[i])
    return (np.array([entry[0] for entry in entries]), expanded, tuple(sectors), tuple(irreps),
            "+".join(sorted(backends)), max(errors, default=0.), contraction)


__all__ = ["LAMKContraction"]
