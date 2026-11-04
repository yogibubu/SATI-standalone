"""Analytic harmonic FC/HT thermal correlations, independent of TI truncation.

Gaussian contraction is evaluated in Bargmann space. log(det(I-B)) is the
sum log(1-lambda(B)) on the analytic unit-disk branch, avoiding a signed
square root of a wrapped determinant and real-time Mehler caustics.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from scipy import fft
from threadpoolctl import threadpool_limits
import numpy as np
from .vibronic_preparation import preparation_session
from .vibronic_parallel import worker_count
from .vibronic_harmonic import (orient_harmonic_model,harmonic_gaussian_generator,
    harmonic_thermal_factors,harmonic_strength_sum_rule,PHASE_PER_CM_FS)


@preparation_session
def harmonic_vibronic_correlation(model,times_fs,*,temperature_K=0.,kind='absorption',
        operator='FCHT',include_origin=True,chunk_size=None,memory_budget_bytes=256*1024**2,workers=0,backend='python',native_executable=None,preparation_cache=None):
    if backend not in ('python','cpp'):raise ValueError('backend must be python or cpp')
    if preparation_cache is not None and backend!='cpp':raise ValueError('Preparation reuse requires the C++ backend')
    workers=worker_count(workers)
    source=model;model,direction=orient_harmonic_model(model,kind)
    rule=harmonic_strength_sum_rule(source,temperature_K,kind=kind,operator=operator)
    times=np.asarray(times_fs,float);n=len(model.initial_frequencies_cm);d=2*n
    if times.ndim!=1 or not len(times) or not np.all(np.isfinite(times)) or np.any(np.diff(times)<0):
        raise ValueError('finite increasing time samples required')
    automatic_chunk=chunk_size is None
    if automatic_chunk:chunk_size=256 if n<=6 else 8
    for value in (chunk_size,memory_budget_bytes):
        if isinstance(value,bool) or not isinstance(value,int) or value<1:raise ValueError('positive integer TD chunk/memory budgets required')
    q=harmonic_thermal_factors(model.initial_frequencies_cm,temperature_K)
    prefactors=np.r_[q,np.ones(n)]
    active=np.flatnonzero(prefactors!=0)
    phase_frequencies=np.r_[-model.initial_frequencies_cm,model.final_frequencies_cm][active]
    prefactors=prefactors[active];d=len(active)
    # Exact vacuum elimination: no thermal cutoff or approximate mode truncation.
    # Temporary complex matrices and factorization workspace, conservatively bounded.
    per_point=16*(12*d*d+12*d)
    capacity=memory_budget_bytes//per_point
    if capacity<1:raise MemoryError('TD matrix workspace exceeds the declared memory budget')
    workers=min(workers,max(1,(len(times)+chunk_size-1)//chunk_size),capacity)
    chunk_size=min(chunk_size,capacity//workers)
    if backend=='cpp':
        if native_executable is None:raise ValueError('Explicit native_executable required for C++')
        from .vibronic_native_kernels import NativeKernels
        library=NativeKernels(str(native_executable)+'.kernels.so')
        with threadpool_limits(limits=1,user_api='blas'):
            values,details=(library.correlation(model,times,temperature_K,direction) if preparation_cache is None else
                preparation_cache.correlation(library,model,times,temperature_K,direction))
        if include_origin:values*=np.exp(-1j*direction*model.origin_cm*PHASE_PER_CM_FS*times)
        output=dict(zip(('FC','HT','interference'),values,strict=True))
        total=values[0] if operator=='FC' else values[1] if operator=='HT' else values.sum(axis=0)
        zero=np.flatnonzero(times==0.)
        error=None if not len(zero) else float(abs(total[zero[0]]-rule['total']))
        if error is not None and error>1e-7*max(1.,rule['total']):raise RuntimeError('Native TD violates zero-time sum rule')
        return dict(schema='matrix.trinity.harmonic_vibronic_correlation.v1',times_fs=times,
            correlation=total,components=output,origin_cm=float(direction*model.origin_cm),
            diagnostics=dict(**details,preparation_cache=None if preparation_cache is None else preparation_cache.snapshot(),numerical_backend='cpp',zero_time_sum_rule=rule,zero_time_error=error,
                active_gaussian_dimension=d,workers=1,blas_threads=1,finite_state_sum_used=False),provenance=model.provenance)
    g=harmonic_gaussian_generator(model);r=g['linear'][active];matrix=g['matrix'][np.ix_(active,active)];conjugate=matrix.conj()
    if np.any(q>=1):raise FloatingPointError('thermal partition function is unresolved at this precision')
    log_partition=np.log1p(-q).sum()
    output={name:np.empty(len(times),complex) for name in ('FC','HT','interference')}
    mu=model.transition_dipole_au;ht0=g['ht_constant'];htlinear=g['ht_linear'][:,active]
    def evaluate_chunk(start):
        stop=min(start+chunk_size,len(times));tau=direction*PHASE_PER_CM_FS*times[start:stop]
        z=prefactors[None,:]*np.exp(-1j*tau[:,None]*phase_frequencies[None,:])
        product=(z[:,:,None]*conjugate[None,:,:]*z[:,None,:])@matrix
        eigen=np.linalg.eigvals(product);distance=np.min(abs(1-eigen))
        if np.any(abs(eigen)>1+1e-7) or distance<1e-12:
            raise FloatingPointError('ill-conditioned harmonic thermal contraction; refine numerical precision/model')
        logdet=np.log1p(-eigen).sum(axis=1)
        a=np.eye(d)[None,:,:]-product
        rhs=z*r.conj()[None,:]+z*np.einsum('ij,tj->ti',conjugate,z*r[None,:])
        # Contract HT into the right-hand sides first: four columns instead of 2N+1.
        right=np.concatenate([rhs[:,:,None],z[:,:,None]*htlinear.conj().T[None,:,:]],axis=2)
        solved=np.linalg.solve(a,right);m1=solved[:,:,0];u12=solved[:,:,1:]
        m2=z*r[None,:]+z*np.einsum('ij,tj->ti',matrix,m1)
        exponent=(2*g['log_ground']+log_partition-.5*logdet
            +.5*np.einsum('i,ti->t',r,m1)+.5*np.einsum('i,ti->t',r.conj(),m2))
        if include_origin:exponent-=1j*direction*model.origin_cm*PHASE_PER_CM_FS*times[start:stop]
        base=np.exp(exponent)
        left=ht0[None,:]+np.einsum('ad,td->ta',htlinear,m1)
        right_ht=ht0.conj()[None,:]+np.einsum('ad,td->ta',htlinear.conj(),m2)
        output['FC'][start:stop]=base*np.vdot(mu,mu).real
        output['HT'][start:stop]=base*(np.sum(left*right_ht,axis=1)+np.einsum('ad,tda->t',htlinear,u12))
        output['interference'][start:stop]=base*np.sum(mu[None,:]*right_ht+left*mu.conj()[None,:],axis=1)
        return float(distance)
    starts=range(0,len(times),chunk_size)
    # Only W chunks may exist at once; BLAS is not multiplied by outer workers.
    with threadpool_limits(limits=1,user_api='blas'):
        if workers==1:
            minimum_distance=min(map(evaluate_chunk,starts))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                minimum_distance=1.
                iterator=iter(starts)
                while batch:=list(islice(iterator,workers)):
                    jobs=[pool.submit(evaluate_chunk,start) for start in batch]
                    minimum_distance=min(minimum_distance,*(job.result() for job in jobs))
    total=output['FC'] if operator=='FC' else output['HT'] if operator=='HT' else sum(output.values())
    zero=np.flatnonzero(times==0.)
    error=None if not len(zero) else float(abs(total[zero[0]]-rule['total']))
    if error is not None and error>1e-7*max(1.,rule['total']):
        raise RuntimeError('analytic TD correlation violates its zero-time thermal sum rule')
    return dict(schema='matrix.trinity.harmonic_vibronic_correlation.v1',times_fs=times,
        correlation=total,components=output,origin_cm=float(direction*model.origin_cm),
        diagnostics=dict(zero_time_sum_rule=rule,zero_time_error=error,
            minimum_contraction_distance=minimum_distance,chunk_size=chunk_size,
            automatic_chunk_size=automatic_chunk,
            estimated_chunk_workspace_bytes=chunk_size*per_point,
            estimated_parallel_workspace_bytes=workers*chunk_size*per_point,workers=workers,
            linear_solve_rhs=4,active_gaussian_dimension=d,blas_threads=1,finite_state_sum_used=False),provenance=model.provenance)


@preparation_session
def harmonic_vibronic_td(model,*,time_step_fs=.25,time_points=2049,temperature_K=0.,kind='absorption',
        operator='FCHT',gaussian_sigma_cm=50.,lorentz_hwhm_cm=0.,tail_tolerance=1e-6,fft_workers=0,**controls):
    if (not np.isfinite(time_step_fs) or time_step_fs<=0 or isinstance(time_points,bool)
            or not isinstance(time_points,int) or time_points<3 or time_points>131073
            or not np.isfinite(gaussian_sigma_cm+lorentz_hwhm_cm) or gaussian_sigma_cm<0 or lorentz_hwhm_cm<0
            or gaussian_sigma_cm+lorentz_hwhm_cm<=0 or not 0<tail_tolerance<1):
        raise ValueError('finite positive time grid, broadening and valid tail tolerance required')
    times=np.arange(time_points)*time_step_fs
    result=harmonic_vibronic_correlation(model,times,temperature_K=temperature_K,kind=kind,
        operator=operator,include_origin=False,**controls)
    tau=PHASE_PER_CM_FS*times
    damping=np.exp(-lorentz_hwhm_cm*tau-.5*(gaussian_sigma_cm*tau)**2)
    signal=result['correlation']*damping
    # A real Hermitian transform stores only positive times. Batch components
    # so PocketFFT has independent transforms to distribute across its workers.
    names=('FC','HT','interference')
    batch=np.asarray([result['components'][name]*damping for name in names])
    fft_workers=min(worker_count(fft_workers),len(names))
    length=2*(time_points-1)
    native=controls.get('backend')=='cpp'
    spacing=PHASE_PER_CM_FS*time_step_fs
    if native:
        from .vibronic_native_kernels import NativeKernels
        densities=NativeKernels(str(controls['native_executable'])+'.kernels.so').fft(batch,spacing)
    else:densities=fft.fftshift(fft.hfft(batch.conj(),n=length,axis=-1,workers=fft_workers),axes=-1)
    spacing=PHASE_PER_CM_FS*time_step_fs
    axis=fft.fftshift(2*np.pi*fft.fftfreq(length,d=spacing))+result['origin_cm']
    if not native:densities*=spacing/(2*np.pi)
    component_densities=dict(zip(names,densities,strict=True))
    density=densities[0] if operator=='FC' else densities[1] if operator=='HT' else densities.sum(axis=0)
    reference=max(float(abs(signal[0])),np.finfo(float).tiny)
    tail=float(np.max(abs(signal[-min(16,len(signal)):]))/reference)
    frequency_step=float(axis[1]-axis[0]);negative=float(-np.minimum(density,0).sum()*frequency_step)
    carrier=np.exp(-1j*result['origin_cm']*tau)
    return dict(schema='matrix.trinity.harmonic_vibronic_td.v1',kind=kind,operator=operator,
        wavenumbers_cm=axis,density_au2_per_cm=density,component_densities_au2_per_cm=component_densities,times_fs=times,correlation=result['correlation']*carrier,
        components={k:v*carrier for k,v in result['components'].items()},correlation_includes_origin=True,origin_cm=result['origin_cm'],
        diagnostics={**result['diagnostics'],'time_window_tail_ratio':tail,'tail_tolerance':tail_tolerance,
            'time_window_converged':bool(tail<=tail_tolerance),'frequency_step_cm':frequency_step,
            'relative_nyquist_cm':float(np.pi/spacing),'integrated_strength_au2':float(density.sum()*frequency_step),
            'negative_density_area_au2':negative,'clipped_negative_density':False,
            'fft_carrier_demodulated':True,'fft_backend':'C++ PocketFFT' if native else 'scipy pocketfft batched hfft',
            'fft_workers':fft_workers,'fft_batch_size':len(names),'scope':'time-step/aliasing convergence requires a separate finer-grid comparison'},
        provenance=result['provenance'],scope='analytic harmonic two-state thermal FC/HT dipole-strength density; not absolute cross sections or nonadiabatic dynamics')


__all__=['harmonic_vibronic_correlation','harmonic_vibronic_td']
