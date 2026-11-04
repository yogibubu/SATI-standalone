"""Finite low-GVPT2 / explicit high-HDC transverse partition functions.

At each fixed reactive action, supply a GVPT2 result in the SAME transverse
basis as the HDC energy model. Low polyads replace complete basis blocks.
WL walks the full explicit high-model occupation domain to retain connectivity,
but its Boltzmann moments omit the replaced states. The low spectral sum is
then added exactly, with no subtraction of noisy partition sums.
"""
from dataclasses import dataclass
from math import comb
import numpy as np
from scipy.special import logsumexp
HC_OVER_KB_CM_K = 1.438776877  # preserved historical cm/K conversion; no numerical change


def occupation_shells(n, maximum):
    if n == 0:
        yield ()
    else:
        for k in range(maximum+1):
            for rest in occupation_shells(n-1, maximum-k):
                yield (k,)+rest


@dataclass(frozen=True)
class MixedTransverseModel:
    hdc_model: object
    low_basis_states: np.ndarray
    low_excitations_cm: np.ndarray
    max_total_quanta: int
    mode_levels_cm: np.ndarray | None = None

    def __post_init__(self):
        from matrix_trinity.hdc_polynomial import HDCPolynomialEnergy
        from matrix_sentinel.quantum_wang_landau import _certified_polynomial_cutoff
        model=self.hdc_model
        if not isinstance(model,HDCPolynomialEnergy):raise TypeError('HDCPolynomialEnergy required')
        n=len(model.fundamentals_cm);states=np.asarray(self.low_basis_states);low=np.asarray(self.low_excitations_cm,float)
        maximum=self.max_total_quanta
        if isinstance(maximum,bool) or not isinstance(maximum,int) or maximum<1:
            raise ValueError('positive integer occupation domain required')
        if (states.ndim!=2 or states.shape[1]!=n or states.dtype.kind not in 'iu' or not len(states)
                or np.any(states<0) or np.any(states.sum(axis=1)>maximum)
                or len(np.unique(states,axis=0))!=len(states)
                or not np.any(np.all(states==0,axis=1))
                or low.shape!=(len(states),) or not np.all(np.isfinite(low))
                or np.any(low<0) or np.min(low)!=0):
            raise ValueError('complete low basis, equal eigenvalue count and ground-anchored excitations required')
        levels=None if self.mode_levels_cm is None else np.asarray(self.mode_levels_cm,float)
        if levels is not None and (levels.shape!=(n,maximum+1) or not np.all(np.isfinite(levels))
                or not np.allclose(levels[:,0],0.,atol=1e-12,rtol=0)
                or not np.allclose(levels[:,1],model.fundamentals_cm,atol=1e-8,rtol=0)):
            raise ValueError('regularized mode levels must preserve the declared ground/fundamental anchors')
        cutoff,_=_certified_polynomial_cutoff(model.fundamentals_cm,model.x_matrix_cm,
                                              model.y_tensor_cm,maximum,levels)
        if cutoff!=maximum:
            raise ValueError(f'requested occupation domain is not monotone; certified maximum is {cutoff}')
        for name,value in [('low_basis_states',states.astype(np.int64,copy=True)),
                           ('low_excitations_cm',low.copy()),('mode_levels_cm',None if levels is None else levels.copy())]:
            if value is not None:value.setflags(write=False)
            object.__setattr__(self,name,value)

    def high_excitations(self, states):
        if self.mode_levels_cm is None:
            return self.hdc_model.energies(states)-self.hdc_model.zero_point_cm
        states=np.asarray(states)
        model=self.hdc_model
        values=self.mode_levels_cm[np.arange(states.shape[1])[None,:],states].sum(axis=1)
        off=model.x_matrix_cm-np.diag(np.diag(model.x_matrix_cm))
        values+=.5*np.einsum('bi,ij,bj->b',states,off,states)
        if model.y_tensor_cm is not None:
            # Keep the same declared falling-factorial Y contribution.
            base=model.energies(states)-model.zero_point_cm
            quadratic=states@model.fundamentals_cm+(states*(states-1))@np.diag(model.x_matrix_cm)
            quadratic+=.5*np.einsum('bi,ij,bj->b',states,off,states)
            values+=base-quadratic
        return values


def mixed_from_gvpt2(hdc_model, gvpt2_result, *, low_max_total_quanta,
                     max_total_quanta, mode_levels_cm=None):
    """Close low selection over every returned GVPT2 block before replacement.

    An included polyad touching the supplied GVPT2 basis boundary is rejected:
    acquire a larger basis and compare it. Closure is relative to the supplied
    resonance policy; no assertion of all physically possible resonances is made.
    """
    if (isinstance(low_max_total_quanta,bool) or not isinstance(low_max_total_quanta,int)
            or not 0<=low_max_total_quanta<max_total_quanta):
        raise ValueError('low cutoff must be a nonnegative integer below the total domain')
    states=np.asarray(gvpt2_result.states,dtype=np.int64)
    if states.ndim!=2 or states.shape[1]!=len(hdc_model.fundamentals_cm):
        raise ValueError('GVPT2 and HDC must share the transverse occupation basis')
    chosen=set(np.flatnonzero(states.sum(axis=1)<=low_max_total_quanta).tolist())
    changed=True
    while changed:
        changed=False
        for block in gvpt2_result.blocks:
            members=set(block['indices'])
            if members&chosen and not members<=chosen:
                chosen|=members;changed=True
    boundary=int(max(states.sum(axis=1)))
    for block in gvpt2_result.blocks:
        if chosen.intersection(block['indices']) and any(states[i].sum()>=boundary for i in block['indices']):
            raise ValueError('selected polyad touches GVPT2 basis boundary; enlarge the low spectral calculation')
    ids=sorted(chosen)
    # energies_cm is relative to the GVPT2 ground; the absolute PT vacuum is
    # independently supplied by hdc_model.zero_point_cm and is preserved.
    return MixedTransverseModel(hdc_model,states[ids],np.asarray(gvpt2_result.energies_cm)[ids],
                                max_total_quanta,mode_levels_cm)


def mixed_transverse_partition(model, temperatures_K, *, method='enumeration',
                                maximum_enumerated_states=100000, replicas=3,
                                seed=61453, sampling_controls=None):
    """Exact low sum plus an exact or WL high-complement moment estimate."""
    if not isinstance(model,MixedTransverseModel):raise TypeError('MixedTransverseModel required')
    t=np.asarray(temperatures_K,float)
    if t.ndim!=1 or not len(t) or not np.all(np.isfinite(t)) or np.any(t<=0):
        raise ValueError('positive finite temperature vector required')
    beta=HC_OVER_KB_CM_K/t;n=len(model.hdc_model.fundamentals_cm)
    count=comb(n+model.max_total_quanta,model.max_total_quanta)
    low_log=logsumexp(-beta[:,None]*model.low_excitations_cm[None,:],axis=1)
    excluded={tuple(s) for s in model.low_basis_states.tolist()}
    samples=[];diagnostics=[]
    if method=='enumeration':
        if count>maximum_enumerated_states:raise MemoryError('enumeration exceeds the declared state budget')
        high=np.array([s for s in occupation_shells(n,model.max_total_quanta) if s not in excluded],int).reshape(-1,n)
        high_e=model.high_excitations(high)
        high_log=logsumexp(-beta[:,None]*high_e[None,:],axis=1) if len(high) else np.full(len(t),-np.inf)
        samples.append(np.logaddexp(low_log,high_log))
    elif method=='wang_landau':
        from matrix_sentinel.quantum_wang_landau import sample_integer_quadratic_wang_landau
        if isinstance(replicas,bool) or not isinstance(replicas,int) or replicas<2:
            raise ValueError('at least two independent WL replicas required')
        controls=dict(sampling_controls or {})
        allowed={'energy_bins','steps','flatness_fraction','minimum_visits_per_bin','check_interval','walkers'}
        if set(controls)-allowed:raise ValueError('unknown CPU mixed-sector sampling controls')
        for replica in range(replicas):
            dos=sample_integer_quadratic_wang_landau(model.hdc_model.fundamentals_cm,model.hdc_model.x_matrix_cm,
                max_total_quanta=model.max_total_quanta,inverse_temperatures=beta,
                mode_levels_cm=model.mode_levels_cm,y_tensor_cm=model.hdc_model.y_tensor_cm,
                excluded_moment_states=model.low_basis_states,device='cpu',precision='float64',
                seed=seed+104729*replica,**controls)
            with np.errstate(divide='ignore'):high_log=np.log(dos.partition_sums)
            samples.append(np.logaddexp(low_log,high_log))
            diagnostics.append(dict(final_log_modification_factor=dos.final_log_modification_factor,
                refinement_count=dos.refinement_count,one_over_t_active=dos.one_over_t_active,
                state_count=dos.state_count,acceptance_rate=dos.acceptance_rate))
    else:raise ValueError('method must be enumeration or wang_landau')
    samples=np.asarray(samples)-beta[None,:]*model.hdc_model.zero_point_cm
    mean_log=logsumexp(samples,axis=0)-np.log(len(samples))
    return dict(schema='matrix.kinetics.mixed_transverse.v1',temperatures_K=t,
        log_partition=mean_log,replica_log_partition=samples,
        replica_log_partition_std=np.std(samples,axis=0,ddof=1) if len(samples)>1 else np.zeros(len(t)),
        full_occupation_state_count=count,low_state_count=len(excluded),high_state_count=count-len(excluded),
        max_total_quanta=model.max_total_quanta,high_model='regularized_mode_levels_HDC_cross' if model.mode_levels_cm is not None else 'HDC_anchor_polynomial',
        perturbative_ground_energy_cm=model.hdc_model.zero_point_cm,
        low_sector_diagonalized_before_sampling=True,diagonalization_per_WL_move=False,
        high_moments_exclude_low_states=True,walk_domain='full certified HDC domain for connectivity',
        sampling_diagnostics=diagnostics,molecular_rate_qualified=False)
