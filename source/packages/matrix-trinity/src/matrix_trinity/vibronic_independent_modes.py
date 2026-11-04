"""Independent native modal states and an explicit energy-only bath correction.

One-dimensional wavefunctions are retained for subsequent overlap corrections.
The energy-only adapter deliberately leaves the coupled Gaussian source intact.
No multidimensional product Hamiltonian or product eigenbasis is constructed.
"""
from collections import OrderedDict
import hashlib

import numpy as np
from scipy.linalg import eigh

from .native_pes_hamiltonian import build_native_modal_hamiltonian
from .vibronic_variational import FiniteVibrationalStates


def solve_independent_native_modes(frequencies_cm, potential_evaluator, *,
        potential_identifier, modes=None, maximum_quanta=64, quadrature_points=144,
        maximum_workspace_bytes=256*1024**2):
    """Solve native 1D slices V(q_i,0_other)-V(0) with the existing HO projector.

    The callback receives dimensionless coordinates in the FULL supplied modal
    frame. Modes are returned in the explicit requested order. Every result
    retains absolute 1D energies (including its ZPE) and the HO eigenvectors.
    The additive model omits all intermode potential terms; this is not a new
    fit or an approximation applied to the frozen potential itself.
    Basis and quadrature convergence must be checked separately.
    """
    frequencies=np.asarray(frequencies_cm,float)
    if (frequencies.ndim!=1 or not len(frequencies) or np.any(frequencies<=0)
            or not np.all(np.isfinite(frequencies)) or not callable(potential_evaluator)):
        raise ValueError('positive modal frequencies and complete potential callback required')
    selected=np.arange(len(frequencies)) if modes is None else np.asarray(modes)
    if (selected.ndim!=1 or selected.dtype.kind not in 'iu' or not len(selected)
            or len(np.unique(selected))!=len(selected) or np.any(selected<0)
            or np.any(selected>=len(frequencies))):
        raise ValueError('unique valid modal indices required')
    if (isinstance(maximum_quanta,bool) or not isinstance(maximum_quanta,int)
            or maximum_quanta<1 or isinstance(quadrature_points,bool)
            or not isinstance(quadrature_points,int) or quadrature_points<1
            or not isinstance(potential_identifier,str) or not potential_identifier.strip()
            or isinstance(maximum_workspace_bytes,bool)
            or not isinstance(maximum_workspace_bytes,int) or maximum_workspace_bytes<1):
        raise ValueError('positive integer basis and workspace budgets required')
    retained=16*len(selected)*(maximum_quanta+1)**2+16*len(frequencies)*quadrature_points
    if retained+64*(maximum_quanta+1)**2 > maximum_workspace_bytes:
        raise MemoryError('full-frame 1D coordinates exceed the workspace budget')
    zero=np.asarray(potential_evaluator(np.zeros((1,len(frequencies)))))
    if zero.shape!=(1,) or np.iscomplexobj(zero) or not np.isfinite(zero[0]):
        raise ValueError('potential origin must be one finite real energy')
    states=[]
    for index in selected:
        def potential(q,mode=int(index)):
            full=np.zeros((len(q),len(frequencies)));full[:,mode]=q[:,0]
            return np.asarray(potential_evaluator(full))-zero[0]
        h=build_native_modal_hamiltonian(frequencies[index:index+1],potential,
            potential_identifier=potential_identifier,maximum_total_quanta=maximum_quanta,
            maximum_harmonic_energy_cm=float(frequencies[index]*maximum_quanta),
            quadrature_points=quadrature_points,maximum_states=maximum_quanta+1,
            maximum_grid_points=quadrature_points,maximum_workspace_bytes=maximum_workspace_bytes)
        energies,vectors=eigh(h.matrix_cm.toarray(),check_finite=False)
        diagnostics=dict(h.diagnostics,mode_index=int(index),potential_origin_removed_cm=float(zero[0]),
            independent_modal_slice=True,intermode_couplings_included=False,
            one_dimensional_wavefunctions_retained=True)
        states.append(FiniteVibrationalStates(h.basis,energies,vectors,h.frequencies_cm,diagnostics))
    return tuple(states)


def harmonic_mode_states(frequency_cm, *, maximum_quanta=64):
    """Exact harmonic ladder for the harmonic tier of the same bath interface."""
    if (not np.isfinite(frequency_cm) or frequency_cm<=0
            or isinstance(maximum_quanta,bool) or not isinstance(maximum_quanta,int)
            or maximum_quanta<1):
        raise ValueError('positive frequency and integer harmonic occupation budget required')
    basis=np.arange(maximum_quanta+1)[:,None]
    return FiniteVibrationalStates(basis,frequency_cm*(basis[:,0]+.5),
        np.eye(len(basis)),np.array([frequency_cm]),dict(exact_harmonic_ladder=True))


def independent_mode_energies(states, occupations):
    """Sum supplied 1D absolute eigenenergies; never extrapolate or clip a ladder."""
    states=tuple(states);rows=np.asarray(occupations)
    if (not states or any(not isinstance(s,FiniteVibrationalStates)
            or len(s.frequencies_cm)!=1 for s in states)
            or rows.ndim!=2 or rows.shape[1]!=len(states) or rows.dtype.kind not in 'iu'
            or np.any(rows<0)):
        raise ValueError('one 1D state set per nonnegative integer occupation column required')
    result=np.zeros(len(rows))
    for column,state in enumerate(states):
        if np.any(rows[:,column]>=len(state.energies_cm)):
            raise ValueError('occupation exceeds the supplied independent eigenlevel ladder')
        result+=state.energies_cm[rows[:,column]]
    return result


def qualify_independent_ladders(reference,expanded,maximum_occupations,*,
                                energy_tolerance_cm,minimum_fidelity):
    """Compare corresponding 1D eigenlevels and return their qualified prefixes.

    States are compared in identical HO reference coordinates. Eigenvectors
    may have arbitrary complex phases and different HO row ordering. This is
    a two-representation diagnostic, not an infinite-basis error certificate.
    """
    old,new=tuple(reference),tuple(expanded);maxima=np.asarray(maximum_occupations)
    if (not old or len(new)!=len(old) or maxima.shape!=(len(old),)
            or maxima.dtype.kind not in 'iu' or np.any(maxima<0)
            or not np.isfinite(energy_tolerance_cm) or energy_tolerance_cm<=0
            or not np.isfinite(minimum_fidelity) or not 0<minimum_fidelity<=1):
        raise ValueError('matching 1D ladders, integer occupation limits and finite tolerances required')
    rows=[]
    for left,right,maximum in zip(old,new,maxima):
        if (not isinstance(left,FiniteVibrationalStates) or not isinstance(right,FiniteVibrationalStates)
                or len(left.frequencies_cm)!=1 or not np.array_equal(left.frequencies_cm,right.frequencies_cm)
                or maximum>=min(len(left.energies_cm),len(right.energies_cm))):
            raise ValueError('matching 1D references covering every requested eigenlevel required')
        lookup={tuple(row):i for i,row in enumerate(right.basis)}
        try:embedding=np.array([lookup[tuple(row)] for row in left.basis])
        except KeyError as error:raise ValueError('expanded HO reference must contain the original rows') from error
        count=int(maximum)+1
        overlap=np.sum(left.eigenvectors[:,:count].conj()*right.eigenvectors[embedding,:count],axis=0)
        fidelity=abs(overlap)**2
        differences=abs(right.energies_cm[:count]-left.energies_cm[:count])
        failed=np.flatnonzero((differences>energy_tolerance_cm)|(fidelity<minimum_fidelity))
        prefix=int(failed[0])-1 if len(failed) else int(maximum)
        rows.append(dict(maximum_requested_occupation=int(maximum),maximum_qualified_occupation=prefix,
            absolute_energy_changes_cm=differences,fidelities=fidelity,
            compared_levels=count,infinite_basis_convergence_certified=False))
    return dict(maximum_qualified_occupations=np.array([r['maximum_qualified_occupation'] for r in rows]),
        energy_tolerance_cm=float(energy_tolerance_cm),minimum_fidelity=float(minimum_fidelity),modes=rows)


class EnergyCorrectedSpectatorObservable:
    """Energy-only diagonal bath model over an existing coupled TI source.

    The label n uses its 1D variational eigenenergy but its original harmonic
    overlap coefficient. This is an explicit energy-only approximation, NOT
    a transformation of the overlap into anharmonic 1D wavefunctions.
    The caller supplies the initial-state energy correction explicitly, so
    spectator ZPE changes are neither inferred nor counted a second time.
    """
    def __init__(self,reference,mode_states,*,initial_energy_shift_cm,
                 maximum_cached_states=4096):
        from .vibronic_sparse import SparseSpectatorObservable
        from .vibronic_contracted_rates import ContractedSpectatorObservable
        if not isinstance(reference,(SparseSpectatorObservable,ContractedSpectatorObservable)) or reference.prepared_spectrum is None:
            raise TypeError('existing sparse spectator observable with prepared spectrum required')
        states=tuple(mode_states)
        independent_mode_energies(states,np.zeros((1,len(states)),int))
        if (len(states)!=len(reference.spectator_modes) or not np.array_equal(
                [s.frequencies_cm[0] for s in states],reference.frequencies_cm)
                or not np.isfinite(initial_energy_shift_cm)
                or isinstance(maximum_cached_states,bool) or not isinstance(maximum_cached_states,int)
                or maximum_cached_states<1):
            raise ValueError('ordered matching bath modes, finite initial shift and positive cache budget required')
        self.reference,self.mode_states=reference,states
        self.initial_energy_shift_cm=float(initial_energy_shift_cm)
        self.maximum_cached_states=maximum_cached_states;self._cache=OrderedDict()
        digest=hashlib.sha256(reference.persistent_model_key.encode())
        for s in states:
            digest.update(np.int64(len(s.energies_cm)).tobytes())
            digest.update(s.energies_cm.tobytes())
        digest.update(np.float64(initial_energy_shift_cm).tobytes())
        self.model_key=digest.hexdigest()
        self.diagnostics=dict(model='independent 1D eigenenergy correction with harmonic overlap labels',
            source_wavefunctions_changed=False,one_dimensional_overlap_corrections_included=False,
            initial_energy_shift_cm=self.initial_energy_shift_cm,model_key=self.model_key,
            active_hamiltonian_sha256=(reference.final.fingerprint if isinstance(reference,SparseSpectatorObservable)
                else reference.active_hamiltonian_sha256),
            spectator_zero_point_energy_cm=float(independent_mode_energies(states,np.zeros((1,len(states)),int))[0]),
            spectator_interactions_included=False)

    def _pair(self,occupations):
        rows=np.asarray(occupations)
        independent_mode_energies(self.mode_states,rows)
        keys=[tuple(row) for row in rows];pending=list(dict.fromkeys(k for k in keys if k not in self._cache))
        values={k:self._cache[k] for k in keys if k in self._cache}
        for first in range(0,len(pending),self.reference.maximum_batch_size):
            block=pending[first:first+self.reference.maximum_batch_size];states=np.asarray(block,int)
            doorway=self.reference.conditional_doorways(states)
            targets=(doorway['initial_energy_cm']+self.initial_energy_shift_cm
                     -independent_mode_energies(self.mode_states,states))
            reference_targets=doorway['initial_energy_cm']-doorway['spectator_energies_cm']
            results=[self.reference.prepared_spectrum.evaluate(doorway['source_hartree'],energy,
                sigma_cm=self.reference.sigma,tolerance_per_strength=self.reference.uniform_density_tolerance,
                workers=self.reference.workers) for energy in (reference_targets,targets)]
            if not all(np.all(result['accepted']) for result in results):
                raise ValueError('prepared spectrum does not enclose the corrected targets at the requested tolerance')
            for key,left,right in zip(block,results[0]['densities_per_cm'],results[1]['densities_per_cm']):
                values[key]=(float(left),float(right));self._cache[key]=values[key]
                if len(self._cache)>self.maximum_cached_states:self._cache.popitem(last=False)
        return np.asarray([values[k] for k in keys]).reshape(len(rows),2)

    def anharmonic(self,occupations):
        return self._pair(occupations)[:,1]

    def reference_anharmonic(self,occupations):
        """Harmonic-bath reference on the SAME conditional source evaluations."""
        return self._pair(occupations)[:,0]
