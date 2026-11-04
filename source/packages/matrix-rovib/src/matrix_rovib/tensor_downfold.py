"""Ordered effective operators on explicit vibration/electron/nuclear spaces.

Spin factors are supplied by the native angular tensor producer. This layer
does not substitute body-fixed rotor components for space-fixed spin tensors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .operator_engine import CompiledOperatorModel, FactorizedOperatorTerm, _term_matrix
from .ordered_downfold import downfold_ordered_operators


@dataclass(frozen=True)
class TensorEffectiveHamiltonian:
    reduction: object
    spaces: tuple
    reference_vectors: np.ndarray

    def hamiltonian(self, J, **options):
        return self.reduction.hamiltonian(J, **options)

    def to_dict(self):
        report = self.reduction.to_dict()
        report['schema'] = 'matrix.lock.tensor_effective_hamiltonian.v1'
        report['internal_spaces'] = [dict(name=s.name, dimension=s.dimension) for s in self.spaces]
        report['reference_vectors'] = dict(real=self.reference_vectors.real.tolist(),
                                           imag=self.reference_vectors.imag.tolist())
        report['basis'] = 'eigenvectors of the complete rotation-independent tensor Hamiltonian'
        report['spin_scope'] = 'explicit supplied finite tensor spaces; no spin trace or angular averaging'
        return report


def downfold_tensor_operator_model(model: CompiledOperatorModel, rotation_words: Mapping,
        active_states, *, parameters=None, maximum_order=4, include_gap_MHz=0.,
        maximum_terms=100000, memory_budget_bytes=512*1024**2, external_labels=None):
    """Downfold a native factorized registry without tracing over its spins.

    ``model.spaces`` contains LAM and every retained electron/nuclear space;
    rotational factors are supplied as ordered words keyed by unique term
    label. An unmapped term belongs to H0. Spin-only interactions (ZFS,
    contact, dipolar or quadrupolar terms) therefore enter the denominators
    exactly, rather than being divided by bare vibrational energy gaps.

    Factors and parameters use the production operator engine convention.
    For a symmetrized term both the internal factors AND the rotation word
    are adjointed. Active indices refer to the full H0 spectrum, not bare
    LAM indices. Exact/declared near-degenerate spin partners are retained by
    the existing quasi-degenerate closure. Source labels survive every order.
    """
    labels = [t.label for t in model.terms]
    if len(set(labels)) != len(labels) or set(rotation_words)-set(labels):
        raise ValueError('unique term labels and known rotation-word sources required')
    # Keep reference eigenvectors as well as the coefficient workspace bounded.
    if 16 * model.dimension**2 * 8 > memory_budget_bytes:
        raise MemoryError('explicit spin tensor reference exceeds memory budget')
    values = dict(model.parameter_defaults)
    if parameters is not None:
        if set(parameters)-set(values):
            raise ValueError('unknown tensor operator parameter')
        values.update(parameters)
    if not all(np.isfinite(float(v)) for v in values.values()):
        raise ValueError('finite tensor operator parameters required')
    h0 = np.zeros((model.dimension, model.dimension), complex)
    terms = []
    for term in model.terms:
        word = tuple(rotation_words.get(term.label, ()))
        if any(isinstance(a, bool) or not isinstance(a, (int, np.integer)) or a not in (0, 1, 2) for a in word):
            raise ValueError('rotation words require Cartesian axes 0,1,2')
        raw = FactorizedOperatorTerm(term.label, term.factors, coefficient=term.coefficient,
            parameter=term.parameter, adjoint_symmetrized=False, symmetry=term.symmetry)
        coefficient = term.coefficient * (float(values[term.parameter]) if term.parameter else 1.)
        matrix = coefficient * _term_matrix(model, raw)
        pieces = [(word, matrix)] if not term.adjoint_symmetrized else [
            (word, .5*matrix), (tuple(reversed(word)), .5*matrix.conj().T)]
        for w, m in pieces:
            if w:
                terms.append(dict(rotation_word=w, matrix_MHz=m, source=term.label))
            else:
                h0 += m
    result = downfold_ordered_operators(h0, terms, active_states,
        maximum_order=maximum_order, include_gap_MHz=include_gap_MHz,
        maximum_terms=maximum_terms, memory_budget_bytes=memory_budget_bytes,
        external_labels=external_labels)
    _, vectors = np.linalg.eigh(h0)
    vectors.setflags(write=False)
    return TensorEffectiveHamiltonian(result, model.spaces, vectors)
