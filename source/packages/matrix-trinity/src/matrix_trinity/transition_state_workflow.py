"""Strict JSON entry point for the local saddle normal form and channel scattering."""
from copy import deepcopy
import hashlib
import json
import numpy as np
from .curvilinear_vpt2 import SonicVPT2Input
from .transition_state import (
    TransitionStateField, solve_transition_state_normal_form,
    transition_state_channels, transition_state_transmission,
)
from .vibronic_workflow import vibronic_json_value


def run_transition_state_workflow(configuration):
    request = deepcopy(configuration)
    if request.pop('schema', None) != 'matrix.trinity.transition_state_request.v1':
        raise ValueError('unsupported transition-state request schema')
    if set(request) - {'field', 'normal_form', 'bath_states', 'reactive_action',
                       'transmission', 'hdc', 'maximum_states', 'maximum_dense_bytes'}:
        raise ValueError('unknown transition-state request fields')
    if 'hdc' in request and 'transmission' in request:
        raise ValueError('HDC channel transmission is not certified by the polynomial saddle certificate')
    raw = dict(request['field']); representation = raw.pop('representation', 'modal')
    metric_model = raw.pop('metric_model', None)
    if representation == 'modal':
        allowed = {'signed_frequencies_cm', 'cubic_cm', 'quartic_cm', 'metric_first_cm',
                   'metric_second_cm', 'constant_shift_cm', 'quartic_coverage',
                   'barrier_energy_cm', 'basis_id'}
        first, second = 'metric_first_cm', 'metric_second_cm'
    elif representation == 'esalc':
        allowed = {'force_constants_hartree', 'cubic_hartree', 'quartic_hartree',
                   'metric_inverse_amu', 'metric_first_inverse_amu',
                   'metric_second_inverse_amu', 'constant_shift_cm',
                   'quartic_coverage', 'barrier_energy_cm', 'basis_id'}
        first, second = 'metric_first_inverse_amu', 'metric_second_inverse_amu'
    else:
        raise ValueError('field representation must be modal or esalc')
    if set(raw) - allowed:
        raise ValueError('unknown transition-state field entries')
    if metric_model not in {'constant', 'curvilinear'}:
        raise ValueError('explicit constant or curvilinear metric_model required')
    if metric_model == 'curvilinear' and (first not in raw or second not in raw):
        raise ValueError('curvilinear saddle requires both metric derivatives')
    if metric_model == 'constant' and any(np.any(np.asarray(raw.get(k, 0)) != 0) for k in (first, second)):
        raise ValueError('constant metric_model conflicts with nonzero metric derivatives')
    if representation == 'modal':
        field = TransitionStateField(**raw)
    else:
        n = len(raw['force_constants_hartree'])
        source = SonicVPT2Input(raw['force_constants_hartree'], raw['cubic_hartree'],
                               raw['quartic_hartree'], raw['metric_inverse_amu'],
                               raw.get(first, np.zeros((n,)*3)),
                               raw.get(second, np.zeros((n,)*4)),
                               raw.get('constant_shift_cm', 0.), raw.get('quartic_coverage', 'complete'))
        field = TransitionStateField.from_trinity(source,
                    barrier_energy_cm=raw.get('barrier_energy_cm', 0.), basis_id=raw.get('basis_id', ''))
    options = dict(request.get('normal_form', {}))
    if set(options) - {'resonance_window_cm', 'maximum_terms', 'maximum_bracket_pairs'}:
        raise ValueError('unknown saddle normal-form controls')
    result = solve_transition_state_normal_form(field, **options)
    controls = {k: request[k] for k in ('maximum_states', 'maximum_dense_bytes') if k in request}
    channels = transition_state_channels(result, request.get('reactive_action', 0.),
                                         request['bath_states'], **controls)
    output = dict(diagnostics=result.diagnostics, channels=channels,
                  polynomial=[dict(powers=key, coefficient_real=float(np.real(v)),
                                   coefficient_imag=float(np.imag(v)))
                              for key, v in sorted(result.polynomial.items())])
    if 'hdc' in request:
        from .transition_state_hdc import solve_transition_state_hdc
        settings = dict(request['hdc'])
        if set(settings) - {'alpha', 'beta_cm2', 'mixing_weight', 'workers'}:
            raise ValueError('unknown saddle HDC controls')
        hdc = solve_transition_state_hdc(result, request.get('reactive_action', 0.),
                    request['bath_states'], **settings, **controls)
        output['hdc_channels'] = dict(energies_cm=hdc.energies_cm,
                    eigenvectors=hdc.eigenvectors, diagnostics=hdc.diagnostics)
    if 'transmission' in request:
        settings = dict(request['transmission'])
        if set(settings) - {'energies_cm', 'channels', 'action_interval'}:
            raise ValueError('unknown local transmission options')
        energies = np.asarray(settings.pop('energies_cm'), float)
        if energies.ndim != 1 or not len(energies) or not np.all(np.isfinite(energies)):
            raise ValueError('nonempty finite transmission energy vector required')
        output['transmission'] = [dict(energy_cm=float(e), **transition_state_transmission(
            result, e, request['bath_states'], **settings, **controls)) for e in energies]
    identity = hashlib.sha256(json.dumps(configuration, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return vibronic_json_value(dict(schema='matrix.trinity.transition_state_result.v1',
                                   configuration_sha256=identity, result=output))
