"""Harmonic KHD resonance Raman: coherent TI and analytic-intermediate TD.

Native models map the same lower electronic surface to one or more excited
surfaces. Output tensors have Cartesian order (scattered, incident). Gamma is
an amplitude-denominator HWHM in cm^-1, distinct from Raman line broadening.
"""
from __future__ import annotations
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
import numpy as np
from numba import njit
from scipy.constants import c, epsilon_0, physical_constants
from threadpoolctl import threadpool_limits
from matrix_gf.harmonic import CM_PER_HARTREE
from .vibronic_harmonic import harmonic_gaussian_generator, harmonic_thermal_factors, PHASE_PER_CM_FS
from .vibronic_ti import _occupation_array, _integral_dag
from .vibronic_kernels import worker_count, parallel_rows


@njit(cache=True,nogil=True)
def _dipole_rows(start,stop,states,ids,values,mu,ht0,htlinear,output):
    for row in range(start,stop):
        overlap=values[ids[row,0]]
        for axis in range(3):
            value=(mu[axis]+ht0[axis])*overlap
            for mode in range(states.shape[1]):
                if ids[row,mode+1]>=0:
                    value+=htlinear[axis,mode]*np.sqrt(states[row,mode])*values[ids[row,mode+1]]
            output[row,axis]=value


def _dipoles(model,ground,excited,maximum_integrals,workers):
    count=len(ground)*len(excited)
    states=np.column_stack([np.repeat(ground,len(excited),axis=0),np.tile(excited,(len(ground),1))])
    g=harmonic_gaussian_generator(model)
    if g['log_ground']<np.log(np.finfo(float).tiny)+4:raise FloatingPointError('Raman TI ground overlap underflows; use TD')
    ids,values=_integral_dag(states,g['matrix'],g['linear'],g['ht_linear'],maximum_integrals,np.exp(g['log_ground']))
    output=np.empty((count,3),complex)
    parallel_rows(_dipole_rows,count,workers,states,ids,values,model.transition_dipole_au,g['ht_constant'],g['ht_linear'],output)
    return output.reshape(len(ground),len(excited),3),len(values)


def _state_strength(model,states):
    return float(np.vdot(model.transition_dipole_au,model.transition_dipole_au).real)+states@(np.sum(abs(model.transition_gradient_dimensionless_au)**2,axis=0))+.5*np.sum(abs(model.transition_gradient_dimensionless_au)**2)


def _models(models,operator):
    if hasattr(models,'initial_frequencies_cm'):models=[models]
    else:models=list(models)
    if not models:raise ValueError('at least one excited-state model required')
    if operator not in ('FC','HT','FCHT'):raise ValueError('operator must be FC, HT or FCHT')
    reference=models[0]
    identity=reference.provenance.get('initial_normal_basis_sha256')
    for model in models:
        if model.initial_frequencies_cm.shape!=reference.initial_frequencies_cm.shape or not np.allclose(model.initial_frequencies_cm,reference.initial_frequencies_cm,rtol=1e-12,atol=1e-10):raise ValueError('Raman electronic surfaces require the same lower-state frequencies and normal basis')
        if len(models)>1 and (not identity or model.provenance.get('initial_normal_basis_sha256')!=identity):raise ValueError('multiple electronic surfaces require a shared initial_normal_basis_sha256; use the native ESALC producer')
        if np.any(model.transition_momentum_dimensionless_au!=0):raise ValueError('Raman electric-dipole models require position-dependent transition moments, not nonradiative momentum couplings')
    if operator=='FC':models=[replace(m,transition_gradient_dimensionless_au=np.zeros_like(m.transition_gradient_dimensionless_au)) for m in models]
    elif operator=='HT':models=[replace(m,transition_dipole_au=np.zeros(3)) for m in models]
    return models


def _channels(model,temperature,initial_quanta,final_quanta,maximum_class,maximum_states,maximum_transitions,workers,include_rayleigh):
    n=len(model.initial_frequencies_cm);q=harmonic_thermal_factors(model.initial_frequencies_cm,temperature)
    initial=_occupation_array(n,0 if temperature==0 else initial_quanta,maximum_class,maximum_states,workers)
    final=_occupation_array(n,final_quanta,maximum_class,maximum_states,workers)
    if len(initial)*len(final)>maximum_transitions:raise MemoryError('Raman ground-state transition budget exceeded; reduce external-state classes/quanta')
    initial=initial[np.argsort(initial@model.initial_frequencies_cm,kind='stable')]
    final=final[np.argsort(final@model.initial_frequencies_cm,kind='stable')]
    population=np.prod(1-q)*np.prod(q[None,:]**initial,axis=1)
    ii=np.repeat(np.arange(len(initial)),len(final));ff=np.tile(np.arange(len(final)),len(initial))
    mask=population[ii]>0
    if not include_rayleigh:mask&=np.any(initial[ii]!=final[ff],axis=1)
    ii=ii[mask];ff=ff[mask]
    if not len(ii):raise ValueError('no requested Raman channels remain; include Rayleigh or increase final quanta')
    return initial,final,ii,ff,population


def _results(laser,ground,final,ii,ff,populations,tensor,diagnostics,models,method):
    wi=models[0].initial_frequencies_cm;shifts=(final[ff]-ground[ii])@wi
    scattered=laser[:,None]-shifts[None,:]
    trace=np.trace(tensor,axis1=-2,axis2=-1)
    symmetric=.5*(tensor+tensor.swapaxes(-1,-2));antisymmetric=.5*(tensor-tensor.swapaxes(-1,-2))
    deviator=symmetric-trace[:,:,None,None]*np.eye(3)[None,None,:,:]/3
    snorm=np.sum(abs(deviator)**2,axis=(-2,-1));anorm=np.sum(abs(antisymmetric)**2,axis=(-2,-1))
    parallel=abs(trace)**2/9+2*snorm/15
    perpendicular=snorm/10+anorm/6
    au_polarizability=physical_constants['atomic unit of electric polarizability'][0]
    # Photon-count differential cross section (power convention adds nu_s/nu_L).
    prefactor=au_polarizability**2/(16*np.pi**2*epsilon_0**2*c**4)
    photons=prefactor*(2*np.pi*c*100)**4*laser[:,None]*np.maximum(scattered,0.)**3*populations[ii][None,:]
    cross_parallel=photons*parallel;cross_perpendicular=photons*perpendicular
    return dict(schema='matrix.sati.resonance_raman_result.v1',method=method,laser_wavenumbers_cm=laser,
        initial_quanta=ground[ii],final_quanta=final[ff],raman_shifts_cm=shifts,scattered_wavenumbers_cm=scattered,
        thermal_populations=populations[ii],polarizability_au=tensor,
        isotropic_90deg_parallel_m2_sr=cross_parallel,isotropic_90deg_perpendicular_m2_sr=cross_perpendicular,
        isotropic_90deg_total_m2_sr=cross_parallel+cross_perpendicular,
        diagnostics={**diagnostics,'initial_population_retained':float(populations.sum()),'incoherent_electronic_state_sum':False,
            'nonpositive_scattered_photon_channels':int(np.count_nonzero(scattered<=0)),
            'geometry':'isotropic molecular orientations, 90 degree detection, linearly polarized incident light',
            'cross_section_convention':'photon-count differential cross sections in vacuum; multiply by nu_scattered/nu_laser for scattered-power convention',
            'scope':'harmonic electric-dipole KHD; no bath dynamics, Raman optical activity or environmental local-field correction'},
        provenance=[m.provenance for m in models])


from .gaussian_hermite import (hermite_recurrence_plan as _td_plan,
    hermite_polynomial_coefficients as _td_coefficients)


def _td_correlation(model,tau,plan,g=None):
    """Infinite excited-oscillator contraction, external number states explicit."""
    if g is None:g=harmonic_gaussian_generator(model)
    n=len(model.initial_frequencies_cm)
    z=np.exp(-1j*model.final_frequencies_cm*tau)
    from .gaussian_partial_trace import gaussian_partial_number_trace
    traced=gaussian_partial_number_trace(g['matrix'],g['linear'],g['log_ground'],np.arange(n,2*n),z)
    covariance,mean,response=(traced[k] for k in ('covariance','mean','response'))
    matrix,r,constant=(traced[k] for k in ('matrix','linear','constant'))
    ha=g['ht_linear'][:,:n];hb=g['ht_linear'][:,n:];mu=model.transition_dipole_au+g['ht_constant']
    left=np.empty((3,2*n+1),complex);right=np.empty_like(left)
    left[:,0]=mu.conj()+hb.conj()@mean[:n];right[:,0]=mu+hb@mean[n:]
    left[:,1:]=hb.conj()@covariance[:n,:]@response;left[:,1:n+1]+=ha.conj()
    right[:,1:]=hb@covariance[n:,:]@response;right[:,n+1:]+=ha
    insertion_covariance=hb.conj()@covariance[:n,n:]@hb.T
    return _td_coefficients(matrix,r,constant,left,right,insertion_covariance,*plan)


def _exponential_weights(s,step,count):
    """Integrate cubic Lagrange correlation interpolants against exact carriers.

    Exponential moments are evaluated by a convergent series near zero and a
    recurrence elsewhere. One-sided endpoint stencils need no extra samples.
    """
    z=s*step;moments=np.empty((len(s),4),complex);small=abs(z)<1
    v=z[small];term=np.ones_like(v);series=np.zeros((len(v),4),complex)
    for order in range(40):
        series+=term[:,None]/(order+np.arange(4)+1)[None,:]
        term*=v/(order+1)
    moments[small]=series
    v=z[~small];moments[~small,0]=np.expm1(v)/v
    for order in range(1,4):moments[~small,order]=np.exp(v)/v-order*moments[~small,order-1]/v
    weights=np.zeros((len(s),count),complex)
    if count==3:groups=[(np.array([0]),np.arange(3)),(np.array([1]),np.arange(-1,2))]
    else:groups=[(np.array([0]),np.arange(4)),(np.arange(1,count-2),np.arange(-1,3)),(np.array([count-2]),np.arange(-2,2))]
    for intervals,nodes in groups:
        if not len(intervals):continue
        for slot,node in enumerate(nodes):
            others=np.delete(nodes,slot)
            polynomial=np.polynomial.polynomial.polyfromroots(others)/np.prod(node-others)
            coefficient=moments[:,:len(polynomial)]@polynomial
            weights[:,intervals+node]+=step*np.exp(s[:,None]*(intervals*step)[None,:])*coefficient[:,None]
    return weights


def harmonic_resonance_raman(models,laser_wavenumbers_cm,*,method='TI',operator='FCHT',
        gamma_cm=100.,temperature_K=0.,maximum_initial_quanta=2,maximum_final_quanta=2,
        maximum_intermediate_quanta=12,maximum_class=2,maximum_states=10000,
        maximum_transitions=10000,maximum_dipole_pairs=500000,maximum_integrals=500000,
        include_rayleigh=False,include_antiresonant=True,workers=0,
        time_step_fs=.25,time_points=4097,chunk_size=8,memory_budget_bytes=256*1024**2,
        backend='python',native_executable=None):
    """Coherent multi-surface KHD Raman tensor and vacuum differential cross sections.

    TI truncates excited intermediate states and reports a Cauchy missing-tensor
    bound. TD sums those states analytically, retains explicit external states,
    and reports a lifetime-tail bound. It does NOT Fourier transform a linear
    absorption autocorrelation. Time-grid refinement remains an independent test.
    """
    workers=worker_count(workers);models=_models(models,operator)
    laser=np.atleast_1d(np.asarray(laser_wavenumbers_cm,float))
    if laser.ndim!=1 or not len(laser) or not np.all(np.isfinite(laser)) or np.any(laser<=0):raise ValueError('positive finite incident laser wavenumbers required')
    widths=np.broadcast_to(np.asarray(gamma_cm,float),(len(models),))
    if not np.all(np.isfinite(widths)) or np.any(widths<=0):raise ValueError('strictly positive amplitude HWHM gamma_cm required')
    for value in (maximum_states,maximum_transitions,maximum_dipole_pairs,maximum_integrals,chunk_size,memory_budget_bytes):
        if isinstance(value,bool) or not isinstance(value,int) or value<1:raise ValueError('positive integer Raman budgets required')
    for value in (maximum_initial_quanta,maximum_final_quanta,maximum_intermediate_quanta,maximum_class):
        if isinstance(value,bool) or not isinstance(value,int) or value<0:raise ValueError('nonnegative integer Raman quanta/class limits required')
    if not isinstance(include_rayleigh,bool) or not isinstance(include_antiresonant,bool):raise ValueError('boolean Rayleigh/antiresonant controls required')
    if method not in ('TI','TD'):raise ValueError('Raman method must be TI or TD')
    initial,final,ii,ff,population=_channels(models[0],temperature_K,maximum_initial_quanta,maximum_final_quanta,maximum_class,maximum_states,maximum_transitions,workers,include_rayleigh)
    wi=models[0].initial_frequencies_cm;ei=initial[ii]@wi;ef=final[ff]@wi
    output_bytes=len(laser)*len(ii)*9*16
    if output_bytes*6>memory_budget_bytes:raise MemoryError('Raman output tensors exceed the declared memory budget')
    tensor=np.zeros((len(laser),len(ii),3,3),complex);diagnostics=dict(operator=operator,workers=workers,
        gamma_cm=widths.tolist(),include_antiresonant=include_antiresonant,include_rayleigh=include_rayleigh,
        external_initial_states=len(initial),external_final_states=len(final),raman_transitions=len(ii),
        external_state_convergence_requires_larger_progression=True,intermediate_states_enumerated=method=='TI')
    if backend not in ('python','cpp'):raise ValueError('backend must be python or cpp')
    if backend=='cpp':
        import os
        from .vibronic_native import native_raman_from_channels
        executable=native_executable or os.environ.get('MATRIX_VIBRONIC_EXECUTABLE')
        if not executable:raise ValueError('Explicitly built C++ executable required')
        return native_raman_from_channels(models,laser,widths,initial,final,ii,ff,population,diagnostics,
            executable=executable,method=method,operator=operator,temperature_K=temperature_K,
            maximum_intermediate_quanta=maximum_intermediate_quanta,maximum_class=maximum_class,
            maximum_states=maximum_states,maximum_transitions=maximum_transitions,
            maximum_dipole_pairs=maximum_dipole_pairs,maximum_integrals=maximum_integrals,
            include_antiresonant=include_antiresonant,time_step_fs=time_step_fs,time_points=time_points,
            memory_budget_bytes=memory_budget_bytes)
    if method=='TI':
        combined=np.unique(np.vstack([initial,final]),axis=0)
        lookup={tuple(s):i for i,s in enumerate(combined.tolist())}
        gi=np.array([lookup[tuple(row)] for row in initial[ii].tolist()]);gf=np.array([lookup[tuple(row)] for row in final[ff].tolist()])
        errors=np.zeros(len(ii));captures=[];counts=[]
        for model,gamma in zip(models,widths,strict=True):
            excited=_occupation_array(len(wi),maximum_intermediate_quanta,maximum_class,maximum_states,workers)
            if len(combined)*len(excited)>maximum_dipole_pairs:raise MemoryError('Raman TI dipole-pair budget exceeded; narrow the intermediate space or use TD')
            pair_workspace=len(combined)*len(excited)*(2*len(wi)*16+128)
            available=memory_budget_bytes-pair_workspace-6*output_bytes
            per_worker_minimum=max(1,len(excited))*16*13
            if available<per_worker_minimum:raise MemoryError('Raman TI pair workspace exceeds memory budget; reduce states or use TD')
            model_workers=min(workers,len(ii),max(1,available//per_worker_minimum))
            laser_block=max(1,min(len(laser),(available//model_workers//(16*max(1,len(excited)))-9)//4))
            dipoles,cache=_dipoles(model,combined,excited,maximum_integrals,workers)
            energies=model.origin_cm+excited@model.final_frequencies_cm
            exact=_state_strength(model,combined);captured=np.sum(abs(dipoles)**2,axis=(1,2));missing=np.maximum(exact-captured,0)
            captures.append(float(np.min(captured/np.maximum(exact,np.finfo(float).tiny))))
            counts.append(dict(intermediate_states=len(excited),cached_integrals=cache,workers=int(model_workers),laser_block_size=int(laser_block)))
            errors+=CM_PER_HARTREE*np.sqrt(missing[gi]*missing[gf])/gamma*(2 if include_antiresonant else 1)
            # Independent channels are parallel; complex summation over intermediates remains coherent.
            def evaluate(row):
                product=np.einsum('ma,mb->mab',dipoles[gf[row]].conj(),dipoles[gi[row]])
                for start in range(0,len(laser),laser_block):
                    stop=min(start+laser_block,len(laser));frequencies=laser[start:stop,None]
                    resonant=1/(energies[None,:]-ei[row]-frequencies-1j*gamma)
                    result=np.einsum('lm,mab->lab',resonant,product,optimize=True)
                    if include_antiresonant:
                        anti=1/(energies[None,:]+frequencies-ef[row]+1j*gamma)
                        result+=np.einsum('lm,mba->lab',anti,product,optimize=True)
                    tensor[start:stop,row]+=CM_PER_HARTREE*result
            with threadpool_limits(limits=1,user_api='blas'):
                if model_workers==1:
                    for row in range(len(ii)):evaluate(row)
                else:
                    with ThreadPoolExecutor(max_workers=model_workers) as pool:
                        iterator=iter(range(len(ii)))
                        while batch:=list(islice(iterator,model_workers)):
                            for job in [pool.submit(evaluate,row) for row in batch]:job.result()
        diagnostics.update(intermediate_capture_minimum=captures,intermediate_spaces=counts,
            omitted_polarizability_frobenius_bound_au=errors,missing_bound_scope='all omitted intermediate states in the supplied electronic surfaces; no bound for omitted electronic surfaces')
    else:
        if not np.isfinite(time_step_fs) or time_step_fs<=0 or isinstance(time_points,bool) or not isinstance(time_points,int) or not 3<=time_points<=131073:raise ValueError('positive TD Raman step and 3..131073 points required')
        states=np.column_stack([final[ff],initial[ii]])
        degree=operator!='FC';plan=_td_plan(states,degree,maximum_integrals)
        step=PHASE_PER_CM_FS*time_step_fs;times=np.arange(time_points)*step
        static_bytes=output_bytes*6+2*len(laser)*time_points*16+sum(a.nbytes for a in plan)
        per_point=16*(48*len(wi)**2+len(ii)*36+len(plan[0])*2)
        capacity=(memory_budget_bytes-static_bytes)//per_point
        if capacity<1:raise MemoryError('Raman TD workspace exceeds declared memory budget')
        workers=min(workers,int(capacity));chunk_size=min(chunk_size,max(1,int(capacity)//workers))
        errors=np.zeros(len(ii));tails=[]
        # Warm the compiled recurrence before parallel calls.
        _td_correlation(models[0],0.,plan)
        for model,gamma in zip(models,widths,strict=True):
            gaussian=harmonic_gaussian_generator(model)
            weights=_exponential_weights(-gamma+1j*(laser-model.origin_cm),step,time_points)
            anti_weights=_exponential_weights(-gamma+1j*(laser+model.origin_cm),step,time_points) if include_antiresonant else None
            tail=np.exp(-gamma*times[-1]);tails.append(float(tail))
            errors+=CM_PER_HARTREE*np.sqrt(_state_strength(model,initial[ii])*_state_strength(model,final[ff]))*tail/gamma*(2 if include_antiresonant else 1)
            def evaluate(start):
                stop=min(start+chunk_size,time_points)
                correlation=np.asarray([_td_correlation(model,t,plan,gaussian) for t in times[start:stop]])
                correlation*=np.exp(1j*times[start:stop,None]*ei[None,:])[:,:,None,None]
                negative=None
                if include_antiresonant:
                    negative=np.asarray([_td_correlation(model,-t,plan,gaussian) for t in times[start:stop]])
                    negative=negative.swapaxes(-1,-2)*np.exp(-1j*times[start:stop,None]*ef[None,:])[:,:,None,None]
                return start,stop,correlation,negative
            def accumulate(result):
                start,stop,correlation,negative=result
                tensor[:]+=1j*CM_PER_HARTREE*np.einsum('lt,tfab->lfab',weights[:,start:stop],correlation,optimize=True)
                if negative is not None:tensor[:]+=-1j*CM_PER_HARTREE*np.einsum('lt,tfab->lfab',anti_weights[:,start:stop],negative,optimize=True)
            with threadpool_limits(limits=1,user_api='blas'):
                if workers==1:
                    for start in range(0,time_points,chunk_size):accumulate(evaluate(start))
                else:
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        iterator=iter(range(0,time_points,chunk_size))
                        while batch:=list(islice(iterator,workers)):
                            for job in [pool.submit(evaluate,start) for start in batch]:accumulate(job.result())
        diagnostics.update(workers=workers,time_step_fs=time_step_fs,time_points=time_points,chunk_size=chunk_size,
            lifetime_window_tail_factors=tails,lifetime_tail_polarizability_frobenius_bound_au=errors,
            estimated_parallel_workspace_bytes=int(static_bytes+workers*chunk_size*per_point),
            intermediate_sum='analytic infinite harmonic Gaussian contraction',external_cached_integrals=len(plan[0]),
            time_integration='exact carrier integration with cubic Lagrange correlation; refine step independently',
            time_grid_converged=None)
    return _results(laser,initial,final,ii,ff,population,tensor,diagnostics,models,method)


__all__=['harmonic_resonance_raman']


def run_esalc_resonance_raman_workflow(configuration):
    """JSON-compatible SATI Raman request using the native ESALC/GF producer."""
    import hashlib
    import json
    from .vibronic_harmonic import ESALCHarmonicState,harmonic_vibronic_from_esalc
    from .vibronic_workflow import _complex_input,vibronic_json_value
    if configuration.get('schema')!='matrix.sati.resonance_raman_request.v1':raise ValueError('unsupported SATI Raman request schema')
    if set(configuration)-{'schema','initial','surfaces','laser_wavenumbers_cm','raman'}:raise ValueError('unknown SATI Raman request fields')
    initial=ESALCHarmonicState(**configuration['initial']);models=[]
    surfaces=configuration['surfaces']
    if not isinstance(surfaces,list) or not surfaces:raise ValueError('nonempty Raman electronic surfaces required')
    for surface in surfaces:
        allowed={'final','electronic_gap_cm','transition_dipole_au','transition_gradient_esalc_au','transition_reference_esalc','pes_model','reference_state'}
        if not isinstance(surface,dict) or set(surface)-allowed:raise ValueError('unknown Raman surface fields')
        final=dict(surface['final']);final.setdefault('force_constants_hartree',None);final.setdefault('metric_inverse_amu',None)
        models.append(harmonic_vibronic_from_esalc(initial,ESALCHarmonicState(**final),
            electronic_gap_cm=surface['electronic_gap_cm'],transition_dipole_au=_complex_input(surface['transition_dipole_au']),
            transition_gradient_esalc_au=_complex_input(surface.get('transition_gradient_esalc_au')),
            transition_reference_esalc=surface.get('transition_reference_esalc'),pes_model=surface.get('pes_model','AH'),reference_state=surface.get('reference_state','initial')))
    controls=configuration.get('raman',{})
    import inspect
    allowed=set(inspect.signature(harmonic_resonance_raman).parameters)-{'models','laser_wavenumbers_cm'}
    if not isinstance(controls,dict) or set(controls)-allowed:raise ValueError('unknown Raman solver controls')
    result=harmonic_resonance_raman(models,configuration['laser_wavenumbers_cm'],**controls)
    result['configuration_sha256']=hashlib.sha256(json.dumps(vibronic_json_value(configuration),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    result['status']='evaluated_with_explicit_convergence_controls'
    return vibronic_json_value(result)


__all__.append('run_esalc_resonance_raman_workflow')
