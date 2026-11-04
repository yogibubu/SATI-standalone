"""Prepared-input standalone execution for harmonic FC/HT models.

The executable reconstructs the Gaussian from physical frequencies/J/K and
the operator; Gaussian preparation remains inside its measured execution.
The Python adapter prepares validated inputs and a provenance manifest only.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import uuid
import numpy as np
from scipy import constants
from matrix_gf.harmonic import CM_PER_HARTREE
from .vibronic_harmonic import HCK_K_CM,PHASE_PER_CM_FS


def native_harmonic_build_command(output,*,compiler='c++',lapack_flags=('-lopenblas',)):
    source=Path(__file__).parent/'native/harmonic_runner.cpp'
    return [compiler,'-O3','-std=c++17',str(source),str(source.parent/'vendor/Faddeeva.cc'),*lapack_flags,'-pthread','-o',str(output)]


def native_shared_build_command(output,*,compiler='c++',lapack_flags=('-lopenblas',)):
    source=Path(__file__).parent/'native/shared_runner.cpp'
    return [compiler,'-O3','-std=c++17','-shared','-fPIC',str(source),str(source.parent/'vendor/Faddeeva.cc'),
        *lapack_flags,'-pthread','-o',str(output)]


def build_native_harmonic_runner(output,*,compiler='c++',lapack_flags=('-lopenblas',)):
    """Explicitly compile the packaged executor; retain a build provenance file."""
    output=Path(output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_name(output.name+'.build-'+uuid.uuid4().hex[:10])
    library=Path(str(output)+'.kernels.so');temporary_library=Path(str(temporary)+'.kernels.so')
    command=native_harmonic_build_command(temporary,compiler=compiler,lapack_flags=lapack_flags)
    try:
        result=subprocess.run(command,capture_output=True,text=True,check=True)
        shared=subprocess.run(native_shared_build_command(temporary_library,compiler=compiler,lapack_flags=lapack_flags),
            capture_output=True,text=True,check=True)
        temporary.replace(output)
        temporary_library.replace(library)
    finally:
        if temporary.exists():temporary.unlink()
        if temporary_library.exists():temporary_library.unlink()
    source=Path(__file__).parent/'native/harmonic_runner.cpp'
    metadata=dict(schema='matrix.trinity.native_harmonic_build.v1',compiler=compiler,
        compiler_version=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0],
        lapack_flags=list(lapack_flags),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        bundled_source_sha256={str(p.relative_to(source.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(source.parent.rglob('*')) if p.is_file()},
        executable_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),build_stderr=result.stderr)
    metadata.update(shared_library=str(library),shared_library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        shared_build_stderr=shared.stderr)
    output.with_suffix(output.suffix+'.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata


def native_adaptive_operator_spectrum(model,constants,forms,*,executable,gradients=None,kind='absorption',
        temperature_K=0.,**controls):
    """Adaptive native kernels with the established proposal/checkpoint owners."""
    from .vibronic_observables import prepare_operator_bank
    from .vibronic_native_kernels import NativeKernels
    from .vibronic_operator_adaptive import adaptive_operator_ti
    oriented,direction,c,h,w=prepare_operator_bank(model,constants,gradients,forms,kind)
    library=NativeKernels(str(executable)+'.kernels.so')
    g,constant,linear,rules,norm,q=library.prepare_bank(oriented,c,h,w,temperature_K)
    common=dict(schema='matrix.sati.operator_spectrum.v1',method='TI',kind=kind,
        origin_cm=direction*oriented.origin_cm,sum_rules=rules,bank_norm_sum_rule=norm)
    return adaptive_operator_ti(oriented,direction,c,h,w,g,constant,linear,common,
        temperature_K=temperature_K,native_library=library,**controls)


def write_native_operator_input(model,path,constants,forms,*,gradients=None,momenta=None,
        kind='absorption',method='TD',gaussian_sigma_cm=50.,lorentz_hwhm_cm=0.,export_profile=True,**controls):
    """Prepare arbitrary Hermitian FC/HT observable forms for native TI or TD.

    Electronic tensor/orientation forms are supplied by their existing optical
    owners. Signed observables retain their interference and share one Gaussian
    and recurrence. Constants/gradients are in the input initial-state basis.
    """
    from dataclasses import replace
    from .vibronic_observables import prepare_operator_bank
    oriented,direction,c,h,w=prepare_operator_bank(model,constants,gradients,forms,kind)
    p=np.zeros_like(h) if momenta is None else np.asarray(momenta,complex)
    if p.shape!=h.shape or not np.all(np.isfinite(p)):
        raise ValueError('Finite momentum coefficients matching the operator bank required')
    if kind=='emission':
        reverse_p=np.sqrt(model.final_frequencies_cm)[:,None]*model.mode_map/np.sqrt(model.initial_frequencies_cm)[None,:]
        p=p.conj()@reverse_p.T
    if (not np.isfinite(gaussian_sigma_cm+lorentz_hwhm_cm)
            or min(gaussian_sigma_cm,lorentz_hwhm_cm)<0 or gaussian_sigma_cm+lorentz_hwhm_cm<=0):
        raise ValueError('Nonnegative nonzero Gaussian/Lorentz widths required')
    dummy=replace(oriented,transition_dipole_au=np.zeros(3),
        transition_gradient_dimensionless_au=np.zeros((3,len(oriented.initial_frequencies_cm))),
        transition_momentum_dimensionless_au=np.zeros((3,len(oriented.initial_frequencies_cm))))
    controls.setdefault('lower_cm',direction*oriented.origin_cm-500.)
    controls.setdefault('upper_cm',direction*oriented.origin_cm+12000.)
    metadata=write_native_harmonic_input(dummy,path,method=method,operator='FCHT',
        gaussian_sigma_cm=max(gaussian_sigma_cm,np.finfo(float).tiny),**controls)
    path=Path(path);rows=path.read_text().splitlines()
    header=rows[2].split();header[3]=format(gaussian_sigma_cm,'.17g');rows[2]=' '.join(header)
    rows.extend(['MATRIX_OPERATOR_BANK_V1',f'{len(c)} {len(w)} {direction:.17g} {lorentz_hwhm_cm:.17g}'])
    for value in (c,h,p,w):rows.append(' '.join(f'{v.real:.17g} {v.imag:.17g}' for v in value.ravel()))
    if not export_profile:
        if method!='TI':raise ValueError('Sticks-only output requires TI')
        rows.append('STICKS_ONLY')
    path.write_text('\n'.join(rows)+'\n')
    metadata.update(schema='matrix.trinity.prepared_native_operator.v1',channels=len(c),forms=len(w),
        kind=kind,Gaussian_sigma_cm=gaussian_sigma_cm,lorentz_hwhm_cm=lorentz_hwhm_cm,
        input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),scope='general Hermitian complex operator bank, TI/TD')
    path.with_suffix(path.suffix+'.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata


def execute_prepared_native_harmonic(input_path,output_prefix,*,executable,blas_threads=1):
    """Run the standalone program and return its typed result and artifact paths.

    Existing Python TI/TD APIs remain available. This explicit prepared-input
    interface avoids importing Python inside the measured native process.
    """
    if isinstance(blas_threads,bool) or not isinstance(blas_threads,int) or blas_threads<1:
        raise ValueError('Positive BLAS thread budget required')
    input_path=Path(input_path).resolve();prefix=Path(output_prefix).resolve();prefix.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run([str(Path(executable).resolve()),str(input_path),str(prefix)],check=True,
        capture_output=True,text=True,env={**os.environ,'OPENBLAS_NUM_THREADS':str(blas_threads),
            'OMP_NUM_THREADS':str(blas_threads)})
    result=json.loads(prefix.with_name(prefix.name+'-result.json').read_text())
    result['input_sha256']=hashlib.sha256(input_path.read_bytes()).hexdigest()
    result['artifacts']={name:str(prefix.with_name(prefix.name+'-'+name+'.tsv'))
        for name in ('spectrum','lines' if result['method']=='TI' else 'correlation','raman')
        if prefix.with_name(prefix.name+'-'+name+'.tsv').exists()}
    return result


def native_harmonic_operator_spectrum(model,constants,forms,*,executable,gradients=None,
        method='TI',kind='absorption',temperature_K=0.,maximum_initial_quanta=2,
        maximum_final_quanta=10,maximum_class=2,maximum_states=100000,
        maximum_transitions=100000,maximum_integrals=500000,time_step_fs=.25,
        time_points=2049,gaussian_sigma_cm=50.,lorentz_hwhm_cm=0.,workers=1):
    """Return the shared operator-spectrum schema from the native numerical core."""
    import csv
    import tempfile
    from .vibronic_harmonic import orient_harmonic_model
    oriented,direction=orient_harmonic_model(model,kind)
    if maximum_class is None:maximum_class=len(oriented.initial_frequencies_cm)
    controls=dict(temperature_K=temperature_K,maximum_initial_quanta=maximum_initial_quanta,
        maximum_final_quanta=maximum_final_quanta,maximum_class=maximum_class,
        maximum_states=maximum_states,maximum_transitions=maximum_transitions,
        maximum_integrals=maximum_integrals,time_step_fs=time_step_fs,time_points=time_points,
        fft_padding_factor=1)
    if method=='TD':
        length=2*(time_points-1);step=2*np.pi/(length*PHASE_PER_CM_FS*time_step_fs)
        controls.update(lower_cm=direction*oriented.origin_cm-length//2*step,
            upper_cm=direction*oriented.origin_cm+(length//2-1)*step,grain_cm=step)
    with tempfile.TemporaryDirectory(prefix='matrix-native-operator-') as folder:
        folder=Path(folder);path=folder/'model.dat'
        write_native_operator_input(model,path,constants,forms,gradients=gradients,
            method=method,kind=kind,gaussian_sigma_cm=gaussian_sigma_cm,
            lorentz_hwhm_cm=lorentz_hwhm_cm,export_profile=method!='TI',**controls)
        result=execute_prepared_native_harmonic(path,folder/'result',executable=executable,blas_threads=1)
        rules=np.asarray(result['sum_rules']);norm=result['bank_norm_sum_rule']
        common=dict(schema='matrix.sati.operator_spectrum.v1',method=method,kind=kind,
            origin_cm=direction*oriented.origin_cm,sum_rules=rules,bank_norm_sum_rule=norm)
        diagnostics=dict(native_backend='C++',native_execution=result,
            cached_integrals=result['cached_integrals'],evaluated_transitions=result['transitions'],workers=1)
        if method=='TI':
            with Path(result['artifacts']['lines']).open() as stream:rows=list(csv.DictReader(stream,delimiter='\t'))
            initial=np.array([[int(x) for x in r['initial_quanta'].split(',') if x] for r in rows],dtype=np.int64)
            final=np.array([[int(x) for x in r['final_quanta'].split(',') if x] for r in rows],dtype=np.int64)
            strengths=np.array([[float(r[f'strength_{f}']) for r in rows] for f in range(len(rules))])
            missing=max(0.,norm-result['captured_bank_norm'])
            q=np.zeros(len(oriented.initial_frequencies_cm)) if temperature_K==0 else np.exp(-HCK_K_CM*oriented.initial_frequencies_cm/temperature_K)
            population=float(np.sum(np.prod(1-q)*np.prod(q[None,:]**np.unique(initial,axis=0),axis=1)))
            diagnostics.update(finite_state_sum_used=True,missing_bank_norm=missing,
                observable_omission_bounds=np.max(abs(np.linalg.eigvalsh(forms)),axis=1)*missing,
                initial_population_retained=population)
            return dict(**common,wavenumbers_cm=np.array([float(r['wavenumber_cm']) for r in rows]),
                strengths=strengths,initial_quanta=initial,final_quanta=final,diagnostics=diagnostics)
        spectrum=np.loadtxt(result['artifacts']['spectrum'],skiprows=1)
        trace=np.loadtxt(result['artifacts']['correlation'],skiprows=1);times=trace[:,0]
        correlation=(trace[:,1::2]+1j*trace[:,2::2]).T
        correlation*=np.exp(-1j*direction*oriented.origin_cm*PHASE_PER_CM_FS*times)
        diagnostics.update(finite_state_sum_used=False,clipped_signed_density=False,
            time_window_tail_relative_bank_norm=np.asarray(result['time_window_tail_relative_bank_norm']))
        return dict(**common,wavenumbers_cm=spectrum[:,0],densities_per_cm=spectrum[:,1:].T,
            times_fs=times,correlations=correlation,diagnostics=diagnostics)


def native_raman_from_channels(models,laser,widths,initial,final,ii,ff,population,diagnostics,*,
        executable,method,operator,temperature_K,maximum_intermediate_quanta,maximum_class,
        maximum_states,maximum_transitions,maximum_dipole_pairs,maximum_integrals,
        include_antiresonant,time_step_fs,time_points,memory_budget_bytes):
    """Execute coherent multi-surface KHD TI/TD in the native numerical owner."""
    import tempfile
    from scipy.constants import c,epsilon_0,physical_constants
    n=len(models[0].initial_frequencies_cm);nr=len(ii)
    polarizability=physical_constants['atomic unit of electric polarizability'][0]
    photon_prefactor=polarizability**2/(16*np.pi**2*epsilon_0**2*c**4)*(2*np.pi*c*100)**4
    with tempfile.TemporaryDirectory(prefix='matrix-native-raman-') as folder:
        folder=Path(folder);inputs=[]
        for k,model in enumerate(models):
            path=folder/f'surface-{k}.dat'
            write_native_harmonic_input(model,path,method=method,operator=operator,temperature_K=temperature_K,
                maximum_initial_quanta=0,maximum_final_quanta=maximum_intermediate_quanta,
                maximum_class=maximum_class,maximum_states=maximum_states,
                maximum_transitions=maximum_transitions,maximum_integrals=maximum_integrals,
                time_step_fs=time_step_fs,time_points=time_points)
            inputs.append(path.read_text())
        rows=[inputs[0],'MATRIX_RAMAN_V1',str(len(models)),*inputs[1:],
            f'{len(laser)} {len(initial)} {len(final)} {nr} {int(include_antiresonant)} {maximum_dipole_pairs} {memory_budget_bytes} {CM_PER_HARTREE:.17g} {photon_prefactor:.17g}']
        for array in (laser,widths,initial,final,population,np.column_stack([ii,ff])):
            rows.append(' '.join(format(v,'.17g') for v in np.asarray(array).ravel()))
        path=folder/'raman.dat';path.write_text('\n'.join(rows)+'\n')
        metadata=execute_prepared_native_harmonic(path,folder/'result',executable=executable)
        data=np.loadtxt(metadata['artifacts']['raman'],skiprows=1).reshape(len(laser),nr,-1)
        tensor=(data[:,:,7::2]+1j*data[:,:,8::2]).reshape(len(laser),nr,3,3)
        out_diagnostics={**diagnostics,'native_backend':'C++','native_execution':metadata,
            'workers':1,'incoherent_electronic_state_sum':False,
            'initial_population_retained':float(population.sum()),
            'nonpositive_scattered_photon_channels':int(np.count_nonzero(data[:,:,3]<=0)),
            'geometry':'isotropic molecular orientations, 90 degree detection, linearly polarized incident light',
            'cross_section_convention':'photon-count differential cross sections in vacuum; multiply by nu_scattered/nu_laser for scattered-power convention',
            'scope':'harmonic electric-dipole KHD; no bath dynamics, Raman optical activity or environmental local-field correction'}
        if method=='TI':
            out_diagnostics.update(intermediate_capture_minimum=metadata['intermediate_capture_minimum'],
                omitted_polarizability_frobenius_bound_au=np.array(metadata['polarizability_error_bound_au']),
                missing_bound_scope='all omitted intermediate states in the supplied electronic surfaces; no bound for omitted electronic surfaces')
        else:
            out_diagnostics.update(time_step_fs=time_step_fs,time_points=time_points,
                lifetime_window_tail_factors=metadata['lifetime_window_tail_factors'],
                lifetime_tail_polarizability_frobenius_bound_au=np.array(metadata['polarizability_error_bound_au']),
                intermediate_sum='analytic infinite harmonic Gaussian contraction',
                external_cached_integrals=metadata['cached_integrals'],
                time_integration='exact carrier integration with cubic Lagrange correlation; refine step independently',
                time_grid_converged=None)
        return dict(schema='matrix.sati.resonance_raman_result.v1',method=method,
            laser_wavenumbers_cm=np.asarray(laser),initial_quanta=initial[ii],final_quanta=final[ff],
            raman_shifts_cm=data[0,:,2],scattered_wavenumbers_cm=data[:,:,3],thermal_populations=data[0,:,4],
            polarizability_au=tensor,isotropic_90deg_parallel_m2_sr=data[:,:,5],
            isotropic_90deg_perpendicular_m2_sr=data[:,:,6],isotropic_90deg_total_m2_sr=data[:,:,5]+data[:,:,6],
            diagnostics=out_diagnostics,provenance=[m.provenance for m in models])


def write_native_harmonic_input(model,path,*,method,operator='FCHT',temperature_K=0.,
        gaussian_sigma_cm=50.,lower_cm=None,upper_cm=None,grain_cm=1.,
        maximum_initial_quanta=4,maximum_final_quanta=16,maximum_class=None,
        maximum_states=100000,maximum_transitions=100000,maximum_integrals=2000000,
        minimum_strength_au2=1e-18,time_step_fs=.25,time_points=4097,fft_padding_factor=8):
    """Export an absorption model, with no precomputed Gaussian or intensities.

    Mode dimensions and exact recurrence keys are dynamic. The TD transform
    can pad the damped correlation with zeros; it does not replace
    the independent time-step/window controls with a larger sampled interval.
    """
    n=len(model.initial_frequencies_cm)
    maximum_class=n if maximum_class is None else maximum_class
    if n<1 or method not in ('TI','TD') or operator not in ('FC','HT','FCHT'):
        raise ValueError('Native harmonic absorption requires positive mode dimension and TI/TD, FC/HT/FCHT')
    integers=[maximum_initial_quanta,maximum_final_quanta,maximum_class,maximum_states,
        maximum_transitions,maximum_integrals,time_points,fft_padding_factor]
    if any(isinstance(v,bool) or not isinstance(v,int) or v<0 for v in integers):
        raise ValueError('Nonnegative integer occupation, cache and grid controls required')
    if maximum_class>n or min(maximum_states,maximum_transitions,maximum_integrals)<1:
        raise ValueError('Positive finite budgets and a valid class limit required')
    if max(maximum_initial_quanta,maximum_final_quanta)>np.iinfo(np.int32).max or not 3<=time_points<=131073 or not 1<=fft_padding_factor<=64:
        raise ValueError('Controls outside the declared native executor bounds')
    lower_cm=model.origin_cm-500 if lower_cm is None else lower_cm
    upper_cm=model.origin_cm+12000 if upper_cm is None else upper_cm
    if (not np.all(np.isfinite([temperature_K,gaussian_sigma_cm,lower_cm,upper_cm,grain_cm,
            minimum_strength_au2,time_step_fs])) or temperature_K<0 or gaussian_sigma_cm<=0
            or upper_cm<=lower_cm or grain_cm<=0 or minimum_strength_au2<0 or time_step_fs<=0):
        raise ValueError('Finite positive spectral/grid controls required')
    prefactor=(constants.Avogadro*constants.e**2/
        (40*constants.epsilon_0*constants.m_e*constants.c**2*np.log(10)))*2/3/CM_PER_HARTREE
    rows=['MATRIX_HARMONIC_EXECUTOR_V1',method+' '+operator,
        ' '.join(format(v,'.17g') for v in [n,temperature_K,model.electronic_gap_cm,
            gaussian_sigma_cm,lower_cm,upper_cm,grain_cm,prefactor]),
        ' '.join(str(v) for v in [maximum_initial_quanta,maximum_final_quanta,maximum_class,
            maximum_states,maximum_transitions,maximum_integrals])+f' {minimum_strength_au2:.17g}',
        f'{time_step_fs:.17g} {time_points} {fft_padding_factor} {HCK_K_CM:.17g} {PHASE_PER_CM_FS:.17g}']
    for value in (model.initial_frequencies_cm,model.final_frequencies_cm,model.mode_map,
            model.final_shift_dimensionless):
        rows.append(' '.join(format(v,'.17g') for v in value.ravel()))
    for value in (model.transition_dipole_au,model.transition_gradient_dimensionless_au,
            model.transition_momentum_dimensionless_au):
        rows.append(' '.join(f'{v.real:.17g} {v.imag:.17g}' for v in value.ravel()))
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text('\n'.join(rows)+'\n')
    source=Path(__file__).parent/'native/harmonic_runner.cpp'
    metadata=dict(schema='matrix.trinity.prepared_native_harmonic.v1',method=method,operator=operator,
        scope='harmonic absorption, dynamic mode dimension, complex linear position/momentum operator',
        input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        bundled_source_sha256={str(p.relative_to(source.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(source.parent.rglob('*')) if p.is_file()},
        Gaussian_preparation_inside_executor=True,prepared_intensities=False,
        temperature_K=temperature_K,origin_cm=model.origin_cm,
        time_step_fs=time_step_fs,time_points=time_points,fft_padding_factor=fft_padding_factor,
        Gaussian_sigma_cm=gaussian_sigma_cm,model_provenance=model.provenance)
    path.with_suffix(path.suffix+'.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata
