"""Serializable ESALC/GF harmonic vibronic workflow shared by TRINITY and LOCK."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import numpy as np
from .vibronic_harmonic import ESALCHarmonicState,harmonic_vibronic_from_esalc,PHASE_PER_CM_FS


def vibronic_json_value(value):
    if isinstance(value,np.ndarray):
        if np.iscomplexobj(value):return dict(real=value.real.tolist(),imag=value.imag.tolist())
        return value.tolist()
    if isinstance(value,np.generic):return vibronic_json_value(value.item())
    if isinstance(value,complex):return dict(real=value.real,imag=value.imag)
    if isinstance(value,dict):return {str(k):vibronic_json_value(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [vibronic_json_value(v) for v in value]
    return value


def _complex_input(value):
    if isinstance(value,dict) and set(value)=={'real','imag'}:return np.asarray(value['real'])+1j*np.asarray(value['imag'])
    return value


def run_esalc_vibronic_workflow(configuration):
    """Prepare one native common-chart model, compute TI and/or analytic TD."""
    original=deepcopy(configuration);config=deepcopy(configuration)
    if config.pop('schema',None)!='matrix.trinity.esalc_vibronic_request.v1':raise ValueError('unsupported ESALC vibronic request schema')
    allowed={'initial','final','electronic_gap_cm','transition_dipole_au','transition_gradient_esalc_au',
        'transition_reference_esalc','pes_model','reference_state','temperature_K','kind','operator','ti','td'}
    if set(config)-allowed:raise ValueError(f'unknown vibronic request fields: {sorted(set(config)-allowed)}')
    initial_data=dict(config['initial'])
    initial_data.setdefault('force_constants_hartree',None);initial_data.setdefault('metric_inverse_amu',None)
    initial=ESALCHarmonicState(**initial_data)
    final_data=dict(config['final'])
    final_data.setdefault('force_constants_hartree',None);final_data.setdefault('metric_inverse_amu',None)
    final=ESALCHarmonicState(**final_data)
    model=harmonic_vibronic_from_esalc(initial,final,electronic_gap_cm=config['electronic_gap_cm'],
        transition_dipole_au=_complex_input(config['transition_dipole_au']),
        transition_gradient_esalc_au=_complex_input(config.get('transition_gradient_esalc_au')),
        transition_reference_esalc=config.get('transition_reference_esalc'),pes_model=config.get('pes_model','local'),reference_state=config.get('reference_state','initial'))
    controls={
        'ti':{'maximum_initial_quanta','maximum_final_quanta','maximum_class','maximum_states',
              'maximum_transitions','maximum_integrals','minimum_strength_au2','target_fraction','workers',
              'selection','selection_proposal','selection_windows_cm','selection_reference_controls','selection_profile_points',
              'selection_energy_bias_per_cm','selection_profile_peak_tolerance','selection_profile_L1_tolerance',
              'selection_profile_absolute_tolerance','selection_window_rankings','selection_assignment_limit',
              'selection_assignment_report_path','selection_batch_size','selection_maximum_per_mode_quanta','line_format',
              'checkpoint_path','resume','checkpoint_interval_batches','checkpoint_interval_seconds'},
        'td':{'time_step_fs','time_points','gaussian_sigma_cm','lorentz_hwhm_cm','tail_tolerance',
              'chunk_size','memory_budget_bytes','workers','fft_workers'}}
    for method,keys in controls.items():
        if config.get(method) is not None and (not isinstance(config[method],dict) or set(config[method])-keys):
            raise ValueError(f'unknown {method.upper()} controls; temperature, kind and operator are shared at request level')
    common={k:config[k] for k in ('temperature_K','kind','operator') if k in config}
    ti=td=None
    if config.get('ti') is not None:
        from .vibronic_ti import harmonic_vibronic_ti
        ti=harmonic_vibronic_ti(model,**common,**config['ti'])
    if config.get('td') is not None:
        from .vibronic_td import harmonic_vibronic_td
        td=harmonic_vibronic_td(model,**common,**config['td'])
    if ti is None and td is None:raise ValueError('request at least one TI or TD calculation')
    comparison=None
    if ti is not None and td is not None:
        samples=np.unique(np.linspace(0,len(td['times_fs'])-1,min(65,len(td['times_fs'])),dtype=int))
        times=td['times_fs'][samples];correlation=np.zeros(len(samples),complex)
        for line in ti['lines']:
            correlation+=line['weighted_strength_au2']*np.exp(-1j*line['wavenumber_cm']*PHASE_PER_CM_FS*times)
        exact=td['correlation'][samples];error=float(np.max(abs(correlation-exact)))
        rule=ti['diagnostics']['strength_sum_rule']['total']
        missing=max(0.,rule-ti['diagnostics']['reported_strength_au2'])
        comparison=dict(sample_times_fs=times,maximum_correlation_error=error,
            unreported_strength_bound=missing,error_within_strength_bound=bool(error<=missing+1e-8*max(rule,1.)),
            scope='independent TI/TD correlation check; not a time-step convergence or experimental accuracy claim')
    identity=hashlib.sha256(json.dumps(vibronic_json_value(original),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    checks=dict(TI_strength=None if ti is None else ti['diagnostics']['converged'],
        TD_time_window=None if td is None else td['diagnostics']['time_window_converged'],
        TI_TD_consistency=None if comparison is None else comparison['error_within_strength_bound'])
    passed=all(v for v in checks.values() if v is not None)
    from matrix_core.exchange_contracts import validate_exchange_tree
    return validate_exchange_tree(vibronic_json_value(dict(schema='matrix.trinity.esalc_vibronic_result.v1',configuration_sha256=identity,
        status='internal_checks_passed' if passed else 'incomplete_numerical_controls',internal_checks=checks,
        model=dict(initial_frequencies_cm=model.initial_frequencies_cm,final_frequencies_cm=model.final_frequencies_cm,
            mode_map=model.mode_map,final_shift_dimensionless=model.final_shift_dimensionless,
            electronic_gap_cm=model.electronic_gap_cm,origin_cm=model.origin_cm,
            transition_dipole_au=model.transition_dipole_au,
            transition_gradient_dimensionless_au=model.transition_gradient_dimensionless_au,provenance=model.provenance),
        time_independent=ti,time_dependent=td,comparison=comparison)))


__all__=['run_esalc_vibronic_workflow','vibronic_json_value']
