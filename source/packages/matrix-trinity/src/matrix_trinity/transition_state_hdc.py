"""HDC unitary completion of transverse saddle channels at fixed real action.

The hyperbolic coordinate is eliminated canonically, never quantized as a
bound oscillator. Only retained transverse couplings enter the HDC map;
already eliminated virtual couplings enter once through the saddle Z4.
"""
from dataclasses import dataclass
from itertools import product
import numpy as np

from .transition_state import (
    TransitionStateNormalForm, _real, _basis, _action_polynomial_matrices,
)
from .hdc_hamiltonian import complete_hdc_normal_form, _memory_guard, _controls
from .vibronic_kernels import worker_count


@dataclass(frozen=True)
class TransitionStateHDCResult:
    normal_form: TransitionStateNormalForm
    reactive_action: float
    basis: np.ndarray
    energies_cm: np.ndarray
    eigenvectors: np.ndarray
    diagnostics: dict


def _components(polynomials, stable):
    parent = list(range(len(stable)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for poly in polynomials:
        for key, value in poly.items():
            if value == 0:
                continue
            support = [j for j,i in enumerate(stable) if key[2*i] or key[2*i+1]]
            for j in support[1:]:
                parent[root(j)] = root(support[0])
    groups = {}
    for j in range(len(stable)):
        groups.setdefault(root(j), []).append(j)
    return list(groups.values())


def _evaluate(normal_form, polynomial, action, states):
    a,b,c = _action_polynomial_matrices(normal_form, tuple(map(tuple,states)), polynomial)
    matrix = a+action*b+action**2*c
    if not np.all(np.isfinite(matrix)):
        raise FloatingPointError('nonfinite saddle HDC channel matrix')
    return matrix


def solve_transition_state_hdc(normal_form, reactive_action, bath_states, *,
                               alpha=1., beta_cm2=5.e5, mixing_weight=None,
                               maximum_states=4096, maximum_dense_bytes=512*1024**2,
                               workers=0):
    """Fiberwise Hermitian HDC normal form, preserving independent bath factors.

    This is the HDC *unitary extension*, not scalar pair-additive HDCPT2.
    The source normal form must retain separate Z3/Z4 tensors. A connected
    bath permits arbitrary finite occupation sets. For independent mode
    components the supplied basis must be their Cartesian product; otherwise
    it is rejected rather than changing the nonlinear completion's size
    consistency. Increasing the space tests this finite model, not a global
    quartic PES or a reactive bound-state spectrum.
    """
    if not isinstance(normal_form, TransitionStateNormalForm):
        raise TypeError('TransitionStateNormalForm required')
    if normal_form.normal_cubic is None or normal_form.normal_quartic is None:
        raise ValueError('separate normal-form Z3/Z4 required; do not infer perturbation orders from the combined channel matrix')
    _controls(alpha,beta_cm2,mixing_weight)
    workers = worker_count(workers)
    action = _real(reactive_action, 'reactive_action')
    stable = [i for i in range(len(normal_form.field.signed_frequencies_cm))
              if i != normal_form.field.reaction_mode]
    states = np.asarray(_basis(bath_states,len(stable),maximum_states,maximum_dense_bytes),int)
    _memory_guard(len(states),maximum_states,maximum_dense_bytes,copies=16)
    polynomials = (normal_form.normal_cubic,normal_form.normal_quartic)
    groups = _components(polynomials,stable)
    local_bases = [sorted({tuple(state[g]) for state in states}) for g in groups]
    expected = int(np.prod([len(b) for b in local_bases],dtype=object))
    if expected != len(states):
        raise ValueError('independent transverse components require a Cartesian-product bath basis')
    # Reactive-only and scalar quantum terms are counted once, outside every
    # independent bath completion. No precomputed effective X is added.
    scalar = (normal_form.field.barrier_energy_cm
              + abs(normal_form.field.signed_frequencies_cm[normal_form.field.reaction_mode])*action)
    for poly in polynomials:
        independent = {key:value for key,value in poly.items()
                       if not any(key[2*i] or key[2*i+1] for i in stable)}
        scalar += _evaluate(normal_form,independent,action,np.zeros((1,len(stable)),int))[0,0]
    results = []
    for group, local_basis in zip(groups,local_bases):
        local_basis = np.asarray(local_basis,int)
        embedded = np.zeros((len(local_basis),len(stable)),int);embedded[:,group]=local_basis
        matrices = []
        modes = {stable[j] for j in group}
        for poly in polynomials:
            selected = {}
            for key,value in poly.items():
                support = {i for i in stable if key[2*i] or key[2*i+1]}
                if support and support <= modes:
                    selected[key]=value
            matrices.append(_evaluate(normal_form,selected,action,embedded))
        frequencies = normal_form.field.signed_frequencies_cm[[stable[j] for j in group]]
        harmonic = (local_basis+.5)@frequencies
        results.append(complete_hdc_normal_form(harmonic,*matrices,
                       alpha=alpha,beta_cm2=beta_cm2,mixing_weight=mixing_weight,
                       maximum_states=maximum_states,maximum_dense_bytes=maximum_dense_bytes,
                       workers=workers,reconstruct_hamiltonian=False))
    if results:
        combinations = list(product(*(range(len(r.energies_cm)) for r in results)))
        energies = np.array([scalar+sum(r.energies_cm[k] for r,k in zip(results,idx)) for idx in combinations])
        vectors = np.ones((1,1),complex)
        for result in results:
            vectors = np.kron(vectors,result.eigenvectors)
        # Kronecker rows use component-major lexicographic bare ordering.
        row_lookup = [{state:i for i,state in enumerate(b)} for b in local_bases]
        row_indices = []
        for state in states:
            index = 0
            for group,lookup,b in zip(groups,row_lookup,local_bases):
                index = index*len(b)+lookup[tuple(state[group])]
            row_indices.append(index)
        vectors = vectors[row_indices]
    else:
        energies = np.array([scalar]);vectors = np.ones((1,1),complex)
    if np.max(abs(energies.imag))>1e-9*max(1.,float(np.max(abs(energies)))):
        raise ValueError('nonreal saddle HDC energies')
    energies = energies.real
    order = np.argsort(energies,kind='stable');energies=energies[order];vectors=vectors[:,order]
    if not np.all(np.isfinite(energies)) or not np.all(np.isfinite(vectors)):
        raise FloatingPointError('nonfinite saddle HDC result')
    orth = float(np.linalg.norm(vectors.conj().T@vectors-np.eye(len(energies))))
    return TransitionStateHDCResult(normal_form,action,states,energies,vectors,dict(
        schema='matrix.trinity.transition_state_hdc.v1',
        formulation='hyperbolic_saddle_normal_form_then_transverse_hdc_unitary_extension',
        conventional_pair_additive_hdcpt2_equivalence=False,
        reactive_coordinate_is_bound=False,reactive_action=action,
        hdc_parameters=dict(alpha=alpha,beta_cm2=beta_cm2,mixing_weight=mixing_weight),workers=workers,
        bath_components=groups,component_diagnostics=[r.diagnostics for r in results],
        independent_components_completed_separately=True,
        retained_resonance_window_cm=normal_form.diagnostics['resonance_window_cm'],
        eliminated_cubic_self_energy_counted_once=True,
        quartic_globally_diagonalized=False,orthogonality_error=orth,
        model_basis_size=len(states),infinite_basis_convergence_certified=False,
        scattering_transmission_certified=False,
    ))


def transition_state_hdc_operator(result, channel_operator):
    """Transform a supplied operator in this same fixed-action bath model.

    The input must already be in the saddle normal-form channel convention.
    This function applies the transverse HDC rotation only. Reactive-action
    derivatives would also require derivatives of the action-dependent basis.
    """
    if not isinstance(result,TransitionStateHDCResult):
        raise TypeError('TransitionStateHDCResult required')
    operator = np.asarray(channel_operator)
    if operator.ndim < 2 or operator.shape[:2] != (len(result.basis),)*2 or not np.all(np.isfinite(operator)):
        raise ValueError('finite channel operator with leading bath axes required')
    shape=operator.shape[2:]
    components=operator.reshape(len(result.basis),len(result.basis),-1)
    transformed=np.einsum('fm,fic,in->mnc',result.eigenvectors.conj(),components,
                          result.eigenvectors,optimize=True)
    return transformed.reshape((len(result.basis),)*2+shape)
