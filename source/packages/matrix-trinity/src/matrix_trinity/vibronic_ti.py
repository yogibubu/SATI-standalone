"""Class-limited harmonic FC/HT sum over states with exact Gaussian recurrences."""
from __future__ import annotations
from math import comb
from .vibronic_kernels import (worker_count, parallel_rows, fill_occupations,
    recurrence_values, transition_amplitudes)
import numpy as np
from .vibronic_preparation import preparation_session
from .vibronic_harmonic import (orient_harmonic_model,harmonic_gaussian_generator,
    harmonic_strength_sum_rule,harmonic_thermal_factors)


def _occupation_array(modes,maximum_quanta,maximum_class,maximum_states,workers,native_library=None):
    for value in (modes,maximum_quanta,maximum_class,maximum_states):
        if isinstance(value,bool) or not isinstance(value,int) or value<0:raise ValueError('integer occupation budgets required')
    if modes<1 or maximum_states<1:raise ValueError('positive mode count and state budget required')
    limit=min(modes,maximum_quanta,maximum_class)
    count=1+sum(comb(modes,k)*comb(maximum_quanta,k) for k in range(1,limit+1))
    if count>maximum_states:raise MemoryError('TI state budget exceeded; narrow classes/quanta or use TD')
    states=np.zeros((count,modes),dtype=np.int64)
    if count==1:return states
    # Saturated binomial table supports large mode counts without integer overflow.
    table=np.zeros((max(modes,maximum_quanta)+1,limit+1),dtype=np.int64)
    table[:,0]=1
    for row in range(1,len(table)):
        for k in range(1,min(row,limit)+1):
            table[row,k]=min(int(table[row-1,k-1])+int(table[row-1,k]),np.iinfo(np.int64).max)
    totals=[];classes=[];offsets=[];offset=1
    for total in range(1,maximum_quanta+1):
        for k in range(1,min(limit,total)+1):
            totals.append(total);classes.append(k);offsets.append(offset)
            offset+=comb(modes,k)*comb(total-1,k-1)
    if native_library is None:
        parallel_rows(fill_occupations,count,workers,states,np.array(totals),np.array(classes),np.array(offsets),table)
    else:native_library.occupations(states,totals,classes,offsets,table)
    return states


def harmonic_vibronic_occupations(modes,maximum_quanta,maximum_class,*,maximum_states=100000,workers=0,
        backend='python',native_executable=None):
    """Exactly counted, compiled combinatorial unranking with parallel row ownership."""
    if backend not in ('python','cpp'):raise ValueError('backend must be python or cpp')
    library=None
    if backend=='cpp':
        if native_executable is None:raise ValueError('Explicit native_executable required for C++')
        from .vibronic_native_kernels import NativeKernels
        library=NativeKernels(str(native_executable)+'.kernels.so')
    return [tuple(row) for row in _occupation_array(modes,maximum_quanta,maximum_class,
        maximum_states,worker_count(workers),library).tolist()]


def _integral_dag(states,matrix,r,htlinear,maximum_integrals,ground):
    """One shared topological cache; omit only algebraically zero dependency edges."""
    dimension=states.shape[1];zero=(0,)*dimension
    known={zero:0};offsets=[0,0];parents=[];factors=[]
    def dependencies(state):
        j=next(k for k,v in enumerate(state) if v)
        previous=list(state);previous[j]-=1;denominator=np.sqrt(state[j])
        edges=[]
        if r[j]!=0:edges.append((tuple(previous),r[j]/denominator))
        for k,v in enumerate(previous):
            if v and matrix[j,k]!=0:
                previous[k]-=1;edges.append((tuple(previous),matrix[j,k]*np.sqrt(v)/denominator));previous[k]+=1
        return edges
    def ensure(state):
        stack=[(state,None)]
        while stack:
            current,edges=stack.pop()
            if current in known:continue
            if edges is None:
                edges=dependencies(current);stack.append((current,edges))
                for parent,_ in reversed(edges):
                    if parent not in known:stack.append((parent,None))
            else:
                if len(known)>=maximum_integrals:raise MemoryError('TI recurrence cache budget exceeded; reduce the requested state space')
                known[current]=len(known)
                for parent,factor in edges:parents.append(known[parent]);factors.append(factor)
                offsets.append(len(parents))
        return known[state]
    ids=np.full((len(states),dimension+1),-1,dtype=np.int64)
    active_ht=np.any(htlinear!=0,axis=0)
    for row,values in enumerate(states.tolist()):
        ids[row,0]=ensure(tuple(values))
        for mode,value in enumerate(values):
            if value and active_ht[mode]:
                values[mode]-=1;ids[row,mode+1]=ensure(tuple(values));values[mode]+=1
    values=recurrence_values(np.array(offsets,dtype=np.int64),np.array(parents,dtype=np.int64),np.array(factors),ground)
    return ids,values


@preparation_session
def harmonic_vibronic_ti(model,*,temperature_K=0.,kind='absorption',operator='FCHT',
        maximum_initial_quanta=None,maximum_final_quanta=None,maximum_class=None,
        maximum_states=100000,maximum_transitions=100000,maximum_integrals=500000,
        minimum_strength_au2=0.,target_fraction=.999,workers=0,selection='enumerate',
        selection_batch_size=None,selection_maximum_per_mode_quanta=128,line_format="records",
        checkpoint_path=None,resume=False,checkpoint_interval_batches=10, checkpoint_interval_seconds=60.,
        selection_proposal=None,backend='python',native_executable=None,line_storage_path=None,preparation_cache=None,
        recurrence_mode='direct',recurrence_workers=1,checkpoint_async=True,
        selection_windows_cm=None,selection_reference_controls=None,selection_profile_points=65,
        selection_energy_bias_per_cm=None,selection_profile_peak_tolerance=.02,selection_profile_L1_tolerance=.01,
        selection_profile_absolute_tolerance=0.,selection_window_rankings=True,selection_assignment_limit=50,
        selection_assignment_report_path=None,selection_rate_at_cm=None,selection_energy_range_cm=None):
    """Compute a finite TI progression without renormalizing it.

    C++ recurrence_mode='levels' contracts dependencies by total occupation;
    recurrence_workers bounds its independent-node executor (one by default).
    selection='targeted' uses a same-model TD reference and converges the
    requested fraction in every selection_windows_cm interval. Optional
    selection_rate_at_cm instead stops on the positive density at one energy,
    using a same-model TD reference; this does not certify the full window.
    selection_reference_controls set its time grid and line width. Global
    strength completeness remains a separate diagnostic. Targeted stopping
    requires every observable profile to meet its peak/L1 tolerances too.
    Per-window rankings share a deduplicated transition set and integral cache.
    A bounded assignment table includes physical FC/HT/interference strengths
    and complete inside-center/external-tail totals.

    """
    spectral_target=None
    if selection=='targeted':
        if selection_windows_cm is None:raise ValueError('targeted TI requires selection_windows_cm')
        from .vibronic_td import harmonic_vibronic_td
        from .vibronic_targeted import SpectralSelectionTarget
        controls=dict(time_step_fs=.25,time_points=4097,gaussian_sigma_cm=50.,lorentz_hwhm_cm=0.)
        supplied={} if selection_reference_controls is None else dict(selection_reference_controls)
        allowed={'time_step_fs','time_points','gaussian_sigma_cm','lorentz_hwhm_cm','tail_tolerance',
            'chunk_size','memory_budget_bytes','fft_workers'}
        if set(supplied)-allowed:raise ValueError('unknown targeted TD reference controls')
        controls.update(supplied)
        reference=harmonic_vibronic_td(model,temperature_K=temperature_K,kind=kind,operator=operator,
            backend=backend,native_executable=native_executable,workers=workers,**controls)
        density=reference['density_au2_per_cm']
        spectral_target=SpectralSelectionTarget(selection_windows_cm,reference['wavenumbers_cm'],
            density[None],density,gaussian_sigma_cm=controls['gaussian_sigma_cm'],
            lorentz_hwhm_cm=controls['lorentz_hwhm_cm'],profile_points=selection_profile_points,
            profile_peak_tolerance=selection_profile_peak_tolerance,profile_L1_tolerance=selection_profile_L1_tolerance,
            profile_absolute_tolerance=selection_profile_absolute_tolerance,window_rankings=selection_window_rankings,
            assignment_limit=selection_assignment_limit,rate_at_cm=selection_rate_at_cm)
        selection='adaptive'
        if selection_proposal is None:selection_proposal='operator_c2'
    elif selection_windows_cm is not None or selection_reference_controls is not None or selection_assignment_report_path is not None or selection_rate_at_cm is not None or selection_energy_range_cm is not None:
        raise ValueError('spectral windows and TD reference controls require selection=targeted')
    if selection_proposal is None:selection_proposal='gaussian_c2'
    if selection_energy_bias_per_cm is None and spectral_target is None:selection_energy_bias_per_cm=0.
    if backend!='cpp' and (recurrence_mode!='direct' or recurrence_workers!=1):
        raise ValueError('Recurrence workspace controls require the C++ backend')
    if backend not in ('python','cpp'):raise ValueError('backend must be python or cpp')
    if preparation_cache is not None and backend!='cpp':raise ValueError('Preparation reuse requires the C++ backend')
    native_library=None
    if backend=='cpp':
        if native_executable is None:raise ValueError('Explicit native_executable required for C++')
        from .vibronic_native_kernels import NativeKernels
        native_library=NativeKernels(str(native_executable)+'.kernels.so',
            recurrence_mode=recurrence_mode,recurrence_workers=recurrence_workers)
    if selection=='adaptive':
        from .vibronic_ti_adaptive import harmonic_vibronic_ti_adaptive
        return harmonic_vibronic_ti_adaptive(model,temperature_K=temperature_K,kind=kind,operator=operator,
            maximum_initial_quanta=128 if maximum_initial_quanta is None else maximum_initial_quanta,
            maximum_final_quanta=128 if maximum_final_quanta is None else maximum_final_quanta,
            maximum_class=maximum_class,maximum_states=maximum_states,maximum_transitions=maximum_transitions,
            maximum_integrals=maximum_integrals,minimum_strength_au2=minimum_strength_au2,target_fraction=target_fraction,
            workers=workers,selection_batch_size=selection_batch_size,
            selection_maximum_per_mode_quanta=selection_maximum_per_mode_quanta,line_format=line_format,
            checkpoint_path=checkpoint_path,resume=resume,checkpoint_async=checkpoint_async,checkpoint_interval_batches=checkpoint_interval_batches, checkpoint_interval_seconds=checkpoint_interval_seconds,
            selection_proposal=selection_proposal,native_library=native_library,line_storage_path=line_storage_path,preparation_cache=preparation_cache,
            spectral_target=spectral_target,selection_energy_bias_per_cm=selection_energy_bias_per_cm,
            assignment_report_path=selection_assignment_report_path,selection_energy_range_cm=selection_energy_range_cm)
    if selection_energy_bias_per_cm!=0:raise ValueError('energy bias requires adaptive or targeted TI')
    if line_storage_path is not None:raise ValueError('Disk-backed line storage requires adaptive selection')
    if selection_proposal!='gaussian_c2':raise ValueError('selection_proposal requires adaptive selection')
    if checkpoint_path is not None or resume:raise ValueError('checkpoints require adaptive selection')
    if line_format not in ('records','columns'):raise ValueError('line_format must be records or columns')
    if selection!='enumerate':raise ValueError('selection must be enumerate or adaptive')
    maximum_initial_quanta=2 if maximum_initial_quanta is None else maximum_initial_quanta
    maximum_final_quanta=8 if maximum_final_quanta is None else maximum_final_quanta
    maximum_class=2 if maximum_class is None else maximum_class
    workers=worker_count(workers)
    source=model;model,direction=orient_harmonic_model(model,kind)
    rule=harmonic_strength_sum_rule(source,temperature_K,kind=kind,operator=operator)
    if not np.isfinite(minimum_strength_au2) or minimum_strength_au2<0 or not 0<target_fraction<=1:
        raise ValueError('nonnegative strength floor and target fraction in (0,1] required')
    for budget in (maximum_transitions,maximum_integrals):
        if isinstance(budget,bool) or not isinstance(budget,int) or budget<1:raise ValueError('positive TI transition/integral budgets required')
    for limit in (maximum_initial_quanta,maximum_final_quanta,maximum_class):
        if isinstance(limit,bool) or not isinstance(limit,int) or limit<0:raise ValueError('nonnegative integer quanta/class limits required')
    wi=model.initial_frequencies_cm;wf=model.final_frequencies_cm;n=len(wi)
    q=harmonic_thermal_factors(wi,temperature_K)
    initial=_occupation_array(n,0 if temperature_K==0 else maximum_initial_quanta,maximum_class,maximum_states,workers,native_library)
    final=_occupation_array(n,maximum_final_quanta,maximum_class,maximum_states,workers,native_library)
    initial_energy=initial@wi;final_energy=final@wf
    order=np.argsort(initial_energy,kind='stable');initial=initial[order];initial_energy=initial_energy[order]
    order=np.argsort(final_energy,kind='stable');final=final[order];final_energy=final_energy[order]
    populations=np.prod(1-q)*np.prod(q[None,:]**initial,axis=1)
    live=np.flatnonzero(populations!=0)
    possible=len(live)*len(final);evaluated=min(possible,maximum_transitions)
    pair=np.arange(evaluated);ii=live[pair//len(final)];ff=pair%len(final)
    states=np.column_stack([initial[ii],final[ff]])
    gaussian=(harmonic_gaussian_generator(model) if native_library is None else
        native_library.prepare_dipole(model,temperature_K) if preparation_cache is None else
        preparation_cache.gaussian(native_library,model,temperature_K))
    if gaussian['log_ground']<np.log(np.finfo(float).tiny)+4:
        raise FloatingPointError('TI ground overlap underflows; use the analytic TD formulation')
    integral_backend='Python dependency graph with compiled contraction'
    if native_library is not None or (n<=6 and evaluated>=256):
        # The energy-ordered class/quanta enumeration is downward closed:
        # lowering any occupied index reaches an earlier retained product.
        # Reuse the adaptive owner's compiled exact-key recurrence, bounded
        # by this finite product count rather than a large generic budget.
        from .vibronic_selection import PackedHermiteCache
        budget=min(maximum_integrals,evaluated);cap=max(maximum_initial_quanta,maximum_final_quanta)
        if preparation_cache is None:
            cache=PackedHermiteCache(gaussian,gaussian['ht_linear'],budget,cap,native_library=native_library)
            ids,exhausted=cache.evaluate(states)
            if exhausted:raise MemoryError('TI recurrence cache budget exceeded; reduce the requested state space')
            values=cache.values[:cache.count]
        else:
            ids,values=preparation_cache.finite_integrals(native_library,model,temperature_K,
                gaussian,gaussian['ht_linear'],states,budget,cap)
        integral_backend='compiled exact-key recurrence reused from adaptive TI'
    else:
        ids,values=_integral_dag(states,gaussian['matrix'],gaussian['linear'],gaussian['ht_linear'],
            maximum_integrals,np.exp(gaussian['log_ground']))
    if native_library is None:
        amplitudes=np.empty((evaluated,5))
        used=parallel_rows(transition_amplitudes,evaluated,workers,states,ids,values,model.transition_dipole_au,
            gaussian['ht_constant'],gaussian['ht_linear'],amplitudes)
    else:
        amplitudes=native_library.amplitudes(states,ids,values,model.transition_dipole_au,gaussian['ht_constant'],gaussian['ht_linear'])
        used=1;integral_backend='C++ collision-checked packed recurrence'
    component_strengths=amplitudes[:,1:4]*populations[ii,None]
    totals=dict(zip(('FC','HT','interference'),map(float,component_strengths.sum(axis=0)),strict=True))
    strengths=amplitudes[:,1 if operator=='FC' else 2 if operator=='HT' else 4]
    weighted=strengths*populations[ii];computed_strength=float(weighted.sum())
    retained=weighted>minimum_strength_au2;discarded=float(weighted[~retained].sum())
    energies=direction*(model.origin_cm+final_energy[ff]-initial_energy[ii])
    rows=[];column_output={}
    if line_format=='columns':
        selected=np.flatnonzero(retained)
        column_output=dict(line_format='columns',line_columns=dict(initial_quanta=initial[ii[selected]],
            final_quanta=final[ff[selected]],wavenumber_cm=energies[selected],population=populations[ii[selected]],
            FC_overlap=amplitudes[selected,0],FC_strength_au2=amplitudes[selected,1],HT_strength_au2=amplitudes[selected,2],
            interference_au2=amplitudes[selected,3],strength_au2=strengths[selected],weighted_strength_au2=weighted[selected]))
    else:
        rows=[]
        for row in np.flatnonzero(retained):
            rows.append(dict(initial_quanta=initial[ii[row]].tolist(),final_quanta=final[ff[row]].tolist(),
                wavenumber_cm=float(energies[row]),population=float(populations[ii[row]]),FC_overlap=float(amplitudes[row,0]),
                FC_strength_au2=float(amplitudes[row,1]),HT_strength_au2=float(amplitudes[row,2]),
                interference_au2=float(amplitudes[row,3]),strength_au2=float(strengths[row]),weighted_strength_au2=float(weighted[row])))
    budget_exhausted=possible>evaluated
    strength=computed_strength
    if rule['total']==0:
        scale=float(np.sum(abs(model.transition_dipole_au)**2)+np.sum(abs(model.transition_gradient_dimensionless_au)**2)+np.sum(abs(model.transition_momentum_dimensionless_au)**2))
        if strength>64*np.finfo(float).eps**2*max(scale,np.finfo(float).tiny):raise RuntimeError('TI progression violates a zero total-strength sum rule')
        fraction=1.
    else: fraction=strength/rule['total']
    if fraction>1+1.e-7:raise RuntimeError('TI progression violates the exact harmonic total-strength sum rule')
    return dict(**column_output,schema='matrix.trinity.harmonic_vibronic_ti.v1',kind=kind,operator=operator,
        origin_cm=float(direction*model.origin_cm),temperature_K=float(temperature_K),lines=rows,
        diagnostics=dict(preparation_cache=None if preparation_cache is None else preparation_cache.snapshot(),initial_population_retained=float(sum(populations)),strength_sum_rule=rule,
            computed_strength_au2=float(strength),reported_strength_au2=float(strength-discarded),
            component_totals_au2=totals,captured_fraction=float(fraction),target_fraction=target_fraction,
            converged=bool(fraction>=target_fraction and not budget_exhausted),
            transition_budget_exhausted=budget_exhausted,evaluated_transitions=evaluated,
            cached_integrals=len(values),workers=used,requested_worker_limit=workers,
            integral_backend=integral_backend,
            numerical_backend=backend,selection_backend=('C++ combinatorial unranking' if native_library is not None else 'compiled combinatorial unranking'),amplitude_backend=('C++ shared recurrence' if native_library is not None else 'numba nogil shared DAG'),initial_states=len(initial),final_states=len(final),
            maximum_class=maximum_class,maximum_initial_quanta=maximum_initial_quanta,maximum_final_quanta=maximum_final_quanta),
        provenance=model.provenance,scope='harmonic two-state dipole strengths, not absolute cross sections; retained thermal populations and progression strengths are never renormalized')


__all__=['harmonic_vibronic_ti','harmonic_vibronic_occupations']
