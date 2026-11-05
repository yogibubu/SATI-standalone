"""Converge zero-temperature block TI from a finite existing harmonic model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
from time import perf_counter


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',type=Path,required=True)
    request=json.loads(parser.parse_args().request.read_text())
    if socket.getfqdn().casefold()!=request['compute_host'].casefold():raise RuntimeError('Explicit remote compute host required')
    import numpy as np
    from dataclasses import replace
    from matrix_trinity.vibronic_harmonic import HarmonicVibronicModel,harmonic_strength_sum_rule,PHASE_PER_CM_FS
    from matrix_trinity.vibronic_gaussian_blocks import zero_temperature_gaussian_blocks,convolve_gaussian_block_lines
    from matrix_trinity.vibronic_ti import harmonic_vibronic_ti
    from matrix_trinity.vibronic_td import harmonic_vibronic_correlation
    from matrix_trinity.vibronic_product_trace import broadened_trace
    from matrix_trinity.vibronic_targeted import SpectralSelectionTarget
    from matrix_trinity.vibronic_td_rates import nonradiative_rate_from_correlation
    from matrix_trinity.vibronic_rates import GOLDEN_RULE_FACTOR
    from matrix_trinity.vibronic_workflow import vibronic_json_value
    out=Path(request['output']);out.mkdir(parents=True,exist_ok=True)
    def save(name,data):(out/name).write_text(json.dumps(vibronic_json_value(data),indent=2)+'\n')
    input_identity=dict(model_sha256=hashlib.sha256(Path(request['model']).read_bytes()).hexdigest(),
        **{key:request[key] for key in ('kind','matrix_tolerance','operator_relative_tolerance','channel_weights','maximum_quanta')})
    identity_path=out/'input-identity.json'
    if identity_path.exists():
        if json.loads(identity_path.read_text())!=input_identity:raise ValueError('Cached block tables belong to another model or partition')
    elif any(out.glob('FC*-*.npz')):
        raise ValueError('Existing block tables lack verified source identity')
    else:save(identity_path.name,input_identity)
    x=np.load(request['model']);m=HarmonicVibronicModel(x['wi'],x['wf'],x['J'],x['shift'],float(x['gap']),x['mu'],x['gradient'])
    prepared=zero_temperature_gaussian_blocks(m,kind=request['kind'],matrix_tolerance=request['matrix_tolerance'],
        operator_relative_tolerance=request['operator_relative_tolerance'],channel_weights=request['channel_weights'])
    save('preparation.json',dict(diagnostics=prepared['diagnostics'],groups=prepared['final_mode_groups'],channel_blocks=prepared['channel_blocks']))
    native=dict(backend='cpp',native_executable=request['native_executable'],workers=1)
    times=np.arange(request['time_points'])*request['time_step_fs']
    reference=replace(m,transition_dipole_au=m.transition_dipole_au*np.sqrt(request['channel_weights']),
        transition_gradient_dimensionless_au=m.transition_gradient_dimensionless_au*np.sqrt(request['channel_weights'])[:,None])
    full=harmonic_vibronic_correlation(reference,times,kind=request['kind'],include_origin=False,**native)['correlation']
    fc_traces=[harmonic_vibronic_correlation(b,times,include_origin=False,**native)['correlation'] for b in prepared['fc_models']]
    block_trace=np.zeros(len(times),complex)
    for channel,active in enumerate(prepared['channel_blocks']):
        trace=harmonic_vibronic_correlation(prepared['operator_models'][channel],times,include_origin=False,**native)['correlation']
        for j,fc in enumerate(fc_traces):
            if j!=active:trace=trace*fc
        block_trace+=trace
    if prepared['direction']<0:block_trace=block_trace.conj()
    axis,td=broadened_trace(full,times,prepared['origin_cm'],sigma_cm=request['sigma_cm'],fft_padding_factor=4)
    _,blocked_td=broadened_trace(block_trace,times,prepared['origin_cm'],sigma_cm=request['sigma_cm'],fft_padding_factor=4)
    profile_axis=np.linspace(*request['window_cm'],request.get('profile_points',65))
    ref_target=SpectralSelectionTarget([request['window_cm']],axis,td[None],td,
        gaussian_sigma_cm=request['sigma_cm'],profile_points=len(profile_axis))
    block_target=SpectralSelectionTarget([request['window_cm']],axis,blocked_td[None],blocked_td,
        gaussian_sigma_cm=request['sigma_cm'],profile_points=len(profile_axis))
    ref=ref_target.reference_profiles[0,0];projected=block_target.reference_profiles[0,0]
    def direct_rate(values):
        correlation=dict(times_fs=times,values=(values*np.exp(-1j*prepared['origin_cm']*PHASE_PER_CM_FS*times))[:,None],
            correlation_units='hartree_squared',diagnostics=dict(relative_propagator_error_bound=0.))
        return nonradiative_rate_from_correlation(correlation,gaussian_sigma_cm=request['sigma_cm'])
    full_rate=direct_rate(full);blocked_rate=direct_rate(block_trace)
    save('TD-final-rate.json',full_rate)
    comparison=dict(full_TD_rate_s_1=full_rate['rate_s_1'],
        blocked_TD_rate_s_1=blocked_rate['rate_s_1'],zero_energy_sampling='exact shared Fourier zero bin',
        correlation_relative_error=float(np.max(abs(block_trace-full))/abs(full[0])),
        window_relative_peak_error=float(np.max(abs(projected-ref))/np.max(abs(ref))))
    save('factorization-TD-check.json',comparison)
    if (comparison['window_relative_peak_error']>request.get('factorization_profile_tolerance',.001)
            or abs(comparison['blocked_TD_rate_s_1']/comparison['full_TD_rate_s_1']-1)>request.get('factorization_profile_tolerance',.001)):
        raise RuntimeError('Block factorization fails original-model TD comparison')
    np.savez(out/'TD-reference.npz',axis=profile_axis,density=ref,projected_density=projected)
    specs=[('FC'+str(i),b) for i,b in enumerate(prepared['fc_models'])]+[('operator'+str(i),b) for i,b in enumerate(prepared['operator_models'])]
    for budget in request['budgets']:
        spectra={};details={}
        for label,block in specs:
            final=out/f'{label}-{budget}.json';table=out/f'{label}-{budget}.npz';checkpoint=out/(label+'-checkpoint.npz')
            if table.exists() and final.exists():
                spectra[label]=dict(np.load(table));details[label]=json.loads(final.read_text());continue
            if harmonic_strength_sum_rule(block,0.)['total'] == 0.:
                spectra[label]=dict(energy=np.empty(0),strength=np.empty(0),
                    states=np.empty((0,len(block.final_frequencies_cm)),dtype=np.int16))
                details[label]=dict(seconds=0.,diagnostics=dict(zero_source=True,
                    captured_fraction=1.,evaluated_transitions=0))
                np.savez(table,**spectra[label]);save(final.name,details[label]);continue
            start=perf_counter()
            result=harmonic_vibronic_ti(block,selection='adaptive',selection_proposal='operator_c2',
                target_fraction=request['block_target_fraction'],maximum_states=budget,maximum_transitions=budget,
                maximum_integrals=request['maximum_integrals'],maximum_final_quanta=request['maximum_quanta'],
                maximum_class=len(block.final_frequencies_cm),line_format='columns',
                checkpoint_path=checkpoint,resume=checkpoint.exists(),checkpoint_async=False,
                line_storage_path=out/(label+'-lines'),**native)
            cols=result['line_columns'];spectra[label]=dict(energy=cols['wavenumber_cm'],strength=cols['weighted_strength_au2'],states=cols['final_quanta'])
            np.savez(table,**spectra[label]);details[label]=dict(seconds=perf_counter()-start,diagnostics=result['diagnostics'])
            save(final.name,details[label]);print(label,budget,result['diagnostics']['captured_fraction'],result['diagnostics']['evaluated_transitions'],flush=True)
        fc=[(spectra['FC'+str(i)]['energy'],spectra['FC'+str(i)]['strength']) for i in range(len(prepared['fc_models']))]
        ops=[(spectra['operator'+str(i)]['energy'],spectra['operator'+str(i)]['strength']) for i in range(3)]
        profiles=[]
        for delta in request['bin_widths_cm']:
            result=convolve_gaussian_block_lines(prepared,fc,ops,gaussian_sigma_cm=request['sigma_cm'],
                bin_width_cm=delta,profile_axis_cm=profile_axis,excitation_cutoff_cm=request['excitation_cutoff_cm'])
            result['rate_recovery']=result['zero_energy_rate_s_1']/comparison['full_TD_rate_s_1']
            result['relative_peak_error']=float(np.max(abs(result['density']-ref))/np.max(abs(ref)))
            result['relative_L1_error']=float(np.trapezoid(abs(result['density']-ref),profile_axis)/np.trapezoid(abs(ref),profile_axis))
            result['window_recovery']=float(np.trapezoid(result['density'],profile_axis)/np.trapezoid(ref,profile_axis))
            profiles.append(result)
        result=dict(host=socket.getfqdn(),model_sha256=hashlib.sha256(Path(request['model']).read_bytes()).hexdigest(),
            budget=budget,preparation=prepared['diagnostics'],TD_comparison=comparison,blocks=details,profiles=profiles,
            final_mode_groups=prepared['final_mode_groups'],channel_blocks=prepared['channel_blocks'],new_electronic_jobs=0)
        save(f'convergence-{budget}.json',result)
        last=profiles[-1];print('CONVERGENCE',budget,last['rate_recovery'],last['window_recovery'],last['relative_peak_error'],flush=True)
        if (abs(last['rate_recovery']-1)<=.01 and abs(last['window_recovery']-1)<=.01 and last['relative_peak_error']<=.01
                and last['relative_L1_error']<=.01):
            save('converged.json',result);break


if __name__=='__main__':main()
