"""Effective transition operators in the same native coupled-F model space."""
from __future__ import annotations

from types import SimpleNamespace
import numpy as np


def native_spin_dipole_operators(surface, lam_result, blocks, *, dipole_au=None,
                                memory_budget_bytes=512*1024**2):
    """Return raw native Wigner E1 operators, with no fictitious eigenvectors."""
    from .spin_hyperfine import spin_hyperfine_lam_reduced_dipole_matrices
    blocks = tuple(sorted(blocks, key=lambda b:b.two_F))
    if not blocks or len({b.two_F for b in blocks}) != len(blocks):
        raise ValueError('nonempty unique native F blocks required')
    model = blocks[0].model
    if any(b.model != model for b in blocks):
        raise ValueError('dipole blocks must share one physical spin model')
    required = sum(16*len(l.basis)*len(r.basis) for i,l in enumerate(blocks)
                   for r in blocks[i:] if abs(l.two_F-r.two_F)<=2 and l.two_F+r.two_F>0)
    if required*4 > memory_budget_bytes:
        raise MemoryError('native spin dipole operators exceed workspace budget')
    view = SimpleNamespace(model=model, blocks=blocks,
        n_values=tuple(sorted({b.N for block in blocks for b in block.basis})),
        representation=lam_result.representation)
    return spin_hyperfine_lam_reduced_dipole_matrices(surface,lam_result,view,
        dipole_au=dipole_au,_raw_operators=True)


def native_spin_effective_dipoles(surface, lam_result, models, *, order=None, dipole_au=None,
                                  memory_budget_bytes=512*1024**2):
    """W_F^dagger mu_FF' W_F', using order-matched, isometric wave operators.

    Normalization resums (I+chi_m^dagger chi_m)^(-1/2); this is not a claim
    that every higher-order term of the transition polynomial is retained.
    """
    models = dict(models)
    if not models or any(f != m.block.two_F for f,m in models.items()):
        raise ValueError("model keys must match nonempty native F blocks")
    raw = native_spin_dipole_operators(surface,lam_result,[m.block for m in models.values()],
        dipole_au=dipole_au,memory_budget_bytes=memory_budget_bytes)
    embeddings = {f:m.embedding(order=order) for f,m in models.items()}
    return {key:embeddings[key[0]].conj().T@mu@embeddings[key[1]] for key,mu in raw.items()}


def native_spin_effective_spectrum(surface, lam_result, models, *, order=None, dipole_au=None,
                                   temperature_K=None, intensity_cut=0.):
    """Production hyperfine line strengths and populations from reduced states."""
    from .spin_hyperfine import (
        SpinHyperfineResult, SpinHyperfineBlock, _assign_lam_spin_levels,
        _basis_group_slices, spin_hyperfine_transitions,
    )
    models = dict(models)
    dipoles = native_spin_effective_dipoles(surface,lam_result,models,order=order,dipole_au=dipole_au)
    blocks = []
    eigenvectors = {}
    for f,m in sorted(models.items()):
        e,u = np.linalg.eigh(m.hamiltonian(order=order))
        w = m.embedding(order=order)@u
        eigenvectors[f] = u
        residual = m.block.hamiltonian_MHz@w-w*e
        levels = _assign_lam_spin_levels(lam_result,f,e,w,_basis_group_slices(m.block.basis))
        blocks.append(SpinHyperfineBlock(f,e,w,m.block.basis,levels,0.,
            float(np.max(np.linalg.norm(residual,axis=0)/np.maximum(1.,np.linalg.norm(m.block.hamiltonian_MHz,ord=np.inf)+abs(e)),initial=0.)),
            float(np.max(abs(w.conj().T@w-np.eye(len(e))),initial=0.)),
            'native-effective','numpy'))
    result = SpinHyperfineResult(next(iter(models.values())).block.model,
        tuple(sorted({b.N for block in blocks for b in block.basis})),tuple(blocks),lam_result.representation)
    transformed = {key:eigenvectors[key[0]].conj().T@mu@eigenvectors[key[1]]
                   for key,mu in dipoles.items()}
    return result, spin_hyperfine_transitions(result,(0.,0.,0.),temperature_K=temperature_K,
        intensity_cut=intensity_cut,_reduced_matrices=transformed)
