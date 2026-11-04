"""Native ESALC/GF two-surface nonresonant SAM input for SATI."""
from copy import deepcopy
import hashlib
from inspect import signature
import json
import numpy as np
from .curvilinear_vpt2 import SonicVPT2Input
from .vibronic_harmonic import ESALCHarmonicState, harmonic_vibronic_from_esalc
from .vibronic_vpt2 import SAMVPT2Field, vpt2_vibronic_ti
from .vibronic_workflow import _complex_input, vibronic_json_value


def _prepare_esalc_sam_state(data):
    data = dict(data); anharmonic = dict(data.pop('anharmonic'))
    state = ESALCHarmonicState(**data)
    allowed = {'cubic_hartree', 'quartic_hartree', 'metric_first_inverse_amu',
               'metric_second_inverse_amu', 'constant_shift_cm', 'quartic_coverage', 'metric_model'}
    if set(anharmonic)-allowed: raise ValueError('unknown ESALC anharmonic field fields')
    n = len(state.reference_esalc)
    supplied = [key in anharmonic for key in ('metric_first_inverse_amu', 'metric_second_inverse_amu')]
    metric_model = anharmonic.get('metric_model')
    if metric_model not in (None, 'constant', 'curvilinear'):
        raise ValueError('metric_model must be constant or curvilinear')
    if any(supplied) != all(supplied) or (not all(supplied) and metric_model != 'constant'):
        raise ValueError('supply both metric derivatives or explicitly declare metric_model=constant')
    if metric_model == 'constant' and any(np.any(np.asarray(anharmonic[key]) != 0)
            for key in ('metric_first_inverse_amu', 'metric_second_inverse_amu') if key in anharmonic):
        raise ValueError('constant metric contradicts nonzero metric derivatives')
    native = SonicVPT2Input(state.force_constants_hartree, anharmonic['cubic_hartree'],
        anharmonic['quartic_hartree'], state.metric_inverse_amu,
        anharmonic.get('metric_first_inverse_amu', np.zeros((n,)*3)),
        anharmonic.get('metric_second_inverse_amu', np.zeros((n,)*4)),
        anharmonic.get('constant_shift_cm', 0.), anharmonic.get('quartic_coverage', 'complete'))
    return state, SAMVPT2Field.from_trinity(native, state)


def run_esalc_vpt2_vibronic_workflow(configuration):
    request = deepcopy(configuration)
    if request.pop('schema', None) != 'matrix.sati.esalc_vpt2_request.v1':
        raise ValueError('unsupported SATI VPT2 request schema')
    allowed = {'initial', 'final', 'electronic_gap_cm', 'transition_dipole_au',
        'transition_gradient_esalc_au', 'transition_reference_esalc', 'controls'}
    if set(request)-allowed:
        raise ValueError(f'unknown SATI VPT2 request fields: {sorted(set(request)-allowed)}')
    initial, initial_field = _prepare_esalc_sam_state(request['initial'])
    final, final_field = _prepare_esalc_sam_state(request['final'])
    model = harmonic_vibronic_from_esalc(initial, final,
        electronic_gap_cm=request['electronic_gap_cm'], transition_dipole_au=_complex_input(request['transition_dipole_au']),
        transition_gradient_esalc_au=_complex_input(request.get('transition_gradient_esalc_au')),
        transition_reference_esalc=request.get('transition_reference_esalc'), pes_model='AH')
    controls = dict(request.get('controls', {}))
    if set(controls)-(set(signature(vpt2_vibronic_ti).parameters)-{'model', 'initial_field', 'final_field'}):
        raise ValueError('unknown SATI VPT2 controls')
    result = vpt2_vibronic_ti(model, initial_field, final_field, **controls)
    identity = hashlib.sha256(json.dumps(configuration, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return vibronic_json_value(dict(schema='matrix.sati.esalc_vpt2_result.v1', configuration_sha256=identity,
        status='explicit_nonresonant_targets_evaluated', result=result))


__all__ = ['run_esalc_vpt2_vibronic_workflow']
