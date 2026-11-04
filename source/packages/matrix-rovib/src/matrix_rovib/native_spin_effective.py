"""Automatic effective spin operators from the production coupled-F assembler."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .ordered_downfold import downfold_ordered_operators


def native_spin_n_values(model, two_F):
    """All integer N allowed by F, S and every coupled nuclear-spin path."""
    from .spin_hyperfine import coupled_nuclear_spin_paths
    if isinstance(two_F, bool) or not isinstance(two_F, (int, np.integer)) or two_F < 0:
        raise ValueError('two_F must be a nonnegative doubled integer')
    two_S = model.electron.two_S
    two_max_I = sum(n.two_I for n in model.nuclei)
    lower = max(0, (two_F-two_S-two_max_I+1)//2)
    upper = (two_F+two_S+two_max_I)//2
    paths = coupled_nuclear_spin_paths(tuple(n.two_I for n in model.nuclei))
    allowed = []
    for N in range(lower, upper+1):
        if any(abs(j-(path[-1] if path else 0)) <= two_F <= j+(path[-1] if path else 0)
               and (j+(path[-1] if path else 0)-two_F) % 2 == 0
               for j in range(abs(2*N-two_S), 2*N+two_S+1, 2) for path in paths):
            allowed.append(N)
    if not allowed:
        raise ValueError('F is incompatible with the electron and nuclear spins')
    return tuple(allowed)


@dataclass(frozen=True)
class NativeSpinOperatorBlock:
    two_F: int
    basis: tuple
    hamiltonian_MHz: np.ndarray
    model: object


def build_native_lam_spin_operator(surface, lam_result, model, *, two_F,
                                   memory_budget_bytes=512*1024**2,_include_spin_free=True,
                                   _spin_free_blocks=None,_contractions=None):
    """Contract all supported constant/surface spin interactions automatically.

    Uses exactly the same coupled basis, Wigner coefficients, phases and
    spatial contractions as solve_spin_hyperfine_lam_rotation. Both diagonal
    and N-changing spin operators are retained; no Cartesian substitution or
    molecule-specific operator registry is introduced.
    """
    from .spin_hyperfine import (
        _validate_lam_spin_surface_keys, _lam_spin_basis_registry,
        _lam_f_block_operator_terms, _assemble_spin_hyperfine_block,
    )
    if isinstance(two_F, bool) or not isinstance(two_F, (int, np.integer)) or two_F < 0:
        raise ValueError('two_F must be a nonnegative doubled integer')
    _validate_lam_spin_surface_keys(surface, model)
    spins, registry, groups = _lam_spin_basis_registry(lam_result, model, only_two_F=two_F)
    if two_F not in registry:
        raise ValueError('requested F has no states in the supplied native N range')
    basis = tuple(registry[two_F])
    if 16*len(basis)**2*8 > memory_budget_bytes:
        raise MemoryError('native spin operator exceeds its dense workspace budget')
    diagonal, coupling = _lam_f_block_operator_terms(surface, lam_result, model, spins,
        two_F, basis, tuple(groups[two_F]),include_spin_free=_include_spin_free,spin_free_blocks=_spin_free_blocks,contractions=_contractions)
    h, _ = _assemble_spin_hyperfine_block(len(basis), diagonal, coupling, backend=None)
    scale = max(1., float(np.max(abs(h), initial=0.)))
    if not np.all(np.isfinite(h)) or np.max(abs(h-h.conj().T), initial=0.) > 2e-10*scale:
        raise ValueError('native coupled-F spin operator is not finite Hermitian')
    h = .5*(h+h.conj().T)
    h.setflags(write=False)
    return NativeSpinOperatorBlock(int(two_F), basis, h, model)


@dataclass(frozen=True)
class NativeSpinEffectiveHamiltonian:
    block: NativeSpinOperatorBlock
    reduction: object
    reference_vectors: np.ndarray
    requested_basis_indices: tuple

    def hamiltonian(self, *, coupling_scale=1., order=None):
        """Evaluate the intermanifold expansion at fixed F, not a J-power series."""
        degree = self.reduction.maximum_order if order is None else order
        if (not np.isfinite(coupling_scale) or isinstance(degree, bool)
                or int(degree) != degree or not 0 <= degree <= self.reduction.maximum_order):
            raise ValueError('finite coupling scale and available integer order required')
        out = np.zeros((len(self.reduction.active_states),)*2, complex)
        for (word, _sources), matrix in self.reduction.coefficients.items():
            if len(word) <= degree:
                out += coupling_scale**len(word)*matrix
        return out

    def embedding(self, *, coupling_scale=1., order=None):
        """Isometric approximate wave operator in the original coupled-F basis."""
        if not np.isfinite(coupling_scale):
            raise ValueError('finite coupling scale required')
        degree = self.reduction.maximum_order if order is None else order
        if isinstance(degree,bool) or int(degree)!=degree or not 0<=degree<=self.reduction.maximum_order:
            raise ValueError('available integer embedding order required')
        p, q = self.reduction.active_states, self.reduction.external_states
        chi = np.zeros((len(q), len(p)), complex)
        for (word, _sources), matrix in self.reduction.wave_operator.items():
            if len(word)<=degree:
                chi += coupling_scale**len(word)*matrix
        metric = np.eye(len(p))+chi.conj().T@chi
        e, u = np.linalg.eigh(metric)
        return (self.reference_vectors[:, p]+self.reference_vectors[:, q]@chi)@((u/np.sqrt(e))@u.conj().T)

    def to_dict(self):
        report = self.reduction.to_dict()
        report['schema'] = 'matrix.lock.native_spin_effective.v1'
        report['two_F'] = self.block.two_F
        report['spin_model'] = self.block.model.to_dict()
        report['maximum_coupling_order'] = report.pop('maximum_rotation_order')
        for group in ('coefficients', 'wave_operator', 'second_order_external_contributions'):
            report[group] = [dict(record) for record in report[group]]
            for record in report[group]:
                record['coupling_order'] = len(record.pop('rotation_word'))
        report.update(requested_basis_indices=list(self.requested_basis_indices),
            basis=[dict(N=b.N, native_level=b.k_index, two_J=b.two_J,
                nuclear_path=list(b.nuclear_path)) for b in self.block.basis],
            reference_vectors=dict(real=self.reference_vectors.real.tolist(),
                                   imag=self.reference_vectors.imag.tolist()),
            convention='native coupled-F Wigner Hamiltonian; canonical Hermitian intermanifold reduction',
            scope='finite-order coupling expansion at fixed F and supplied LAM/N basis; not a J polynomial')
        return report


def native_spin_effective_hamiltonian(surface, lam_result, model, *, two_F, active_levels,
        maximum_order=4, include_gap_MHz=0., maximum_terms=100000,
        memory_budget_bytes=512*1024**2):
    """Build and reduce native spin couplings without an external angular builder.

    ``active_levels`` maps N to native LAM-rotation level indices. Every
    electron/nuclear coupling path of each selected level is retained.
    H0 contains the complete PHP and QHQ, including spin, exactly. Only
    P-Q coupling is expanded. Near-degenerate reference states are closed
    using include_gap_MHz before division. N-changing rank-1/rank-2 terms are
    generated by the production Wigner producer, not fitted in J.
    """
    block = build_native_lam_spin_operator(surface, lam_result, model, two_F=two_F,
                                          memory_budget_bytes=memory_budget_bytes)
    requested = set()
    for N, levels in active_levels.items():
        if isinstance(N, bool) or not isinstance(N, (int, np.integer)) or N not in lam_result.blocks:
            raise ValueError('active levels require native integer N keys')
        for level in levels:
            if (isinstance(level, bool) or not isinstance(level, (int, np.integer))
                    or not 0 <= level < len(lam_result.blocks[N].energies_MHz)):
                raise ValueError('active level index outside the native LAM-rotation basis')
            requested.add((N, level))
    p = tuple(i for i, b in enumerate(block.basis) if (b.N, b.k_index) in requested)
    if not p:
        raise ValueError('active levels do not intersect the requested F block')
    return reduce_native_spin_operator(block,p,maximum_order=maximum_order,
        include_gap_MHz=include_gap_MHz,maximum_terms=maximum_terms,
        memory_budget_bytes=memory_budget_bytes)


def reduce_native_spin_operator(block, basis_indices, *, maximum_order=4,
        include_gap_MHz=0., maximum_terms=100000, memory_budget_bytes=512*1024**2):
    """Reduce an already assembled physical F block without rebuilding it."""
    p = tuple(basis_indices)
    if (not p or len(set(p))!=len(p) or any(isinstance(i,bool) or
            not isinstance(i,(int,np.integer)) or not 0<=i<len(block.basis) for i in p)):
        raise ValueError('unique valid native basis indices required')
    selected = set(p)
    q = tuple(i for i in range(len(block.basis)) if i not in selected)
    h = block.hamiltonian_MHz
    ep, up = np.linalg.eigh(h[np.ix_(p, p)])
    eq, uq = np.linalg.eigh(h[np.ix_(q, q)])
    rotation = np.zeros_like(h)
    rotation[np.ix_(p, range(len(p)))] = up
    rotation[np.ix_(q, range(len(p), len(h)))] = uq
    # Diagonalize P and Q separately so an accidental degeneracy cannot
    # rotate the requested model space before the quasi-degenerate closure.
    energy = np.r_[ep, eq]
    permutation = np.argsort(energy, kind='stable')
    vectors = rotation[:, permutation]
    active = np.flatnonzero(permutation < len(p))
    transformed = vectors.conj().T@h@vectors
    perturbation = transformed-np.diag(energy[permutation])
    same_partition = (permutation[:, None] < len(p)) == (permutation[None, :] < len(p))
    perturbation[same_partition] = 0.
    perturbation = .5*(perturbation+perturbation.conj().T)
    # One formal generator represents coupling scale lambda, not J_a.
    reduced = downfold_ordered_operators(np.diag(energy[permutation]),
        [dict(rotation_word=(0,), matrix_MHz=perturbation, source='native:P-Q-spin')], active,
        maximum_order=maximum_order, include_gap_MHz=include_gap_MHz,
        maximum_terms=maximum_terms, memory_budget_bytes=memory_budget_bytes)
    vectors.setflags(write=False)
    return NativeSpinEffectiveHamiltonian(block, reduced, vectors, p)
