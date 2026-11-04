"""Native ESALC/GF preparation and reproducible optical-property requests."""
from copy import deepcopy
import hashlib
import json
import numpy as np
from .vibronic_harmonic import (ESALCHarmonicState, harmonic_vibronic_from_esalc,
                               _prepare_pes_model)
from .vibronic_optical import (harmonic_circular_spectrum, harmonic_two_photon_spectrum,
                              harmonic_mcd_spectrum, harmonic_mcd_manifold_spectrum)
from .vibronic_workflow import _complex_input, vibronic_json_value


def run_esalc_optical_workflow(configuration):
    """Transport every tensor/derivative through the SAME native normal basis.

    Property records have value, optional gradient_esalc, and optional
    reference_esalc (defaults to the initial force-field expansion point).
    """
    request = deepcopy(configuration)
    if request.pop('schema', None) != 'matrix.sati.optical_request.v1':
        raise ValueError('unsupported optical request schema')
    allowed = {'spectroscopy', 'initial', 'final', 'electronic_gap_cm', 'pes_model',
               'reference_state', 'properties', 'controls', 'response'}
    if set(request)-allowed:
        raise ValueError(f'unknown optical request fields: {sorted(set(request)-allowed)}')
    initial = ESALCHarmonicState(**request['initial'])
    final = ESALCHarmonicState(**request['final'])
    pes = request.get('pes_model', 'local'); reference_state = request.get('reference_state', 'initial')
    model = harmonic_vibronic_from_esalc(initial, final, electronic_gap_cm=request['electronic_gap_cm'],
                                        transition_dipole_au=np.zeros(3), pes_model=pes, reference_state=reference_state)
    if reference_state == 'initial':
        prepared, _ = _prepare_pes_model(initial, final, pes)
    else:
        _, prepared = _prepare_pes_model(final, initial, pes)
    frequencies, transform, minimum, _ = prepared.modes()
    properties = request['properties']
    def prop(name, shape, required=True):
        if name not in properties:
            if required: raise ValueError(f'missing optical property {name}')
            return None, None
        data = properties[name]
        if set(data)-{'value', 'gradient_esalc', 'reference_esalc'}:
            raise ValueError(f'unknown fields in optical property {name}')
        value = np.asarray(_complex_input(data['value']), complex)
        gradient = np.asarray(_complex_input(data.get('gradient_esalc', np.zeros(shape+(len(frequencies),)))), complex)
        reference = np.asarray(data.get('reference_esalc', initial.reference_esalc), float)
        if (value.shape != shape or gradient.shape != shape+(len(frequencies),)
                or reference.shape != frequencies.shape or not all(np.all(np.isfinite(x)) for x in (value, gradient, reference))):
            raise ValueError(f'invalid ESALC shape or nonfinite optical property {name}')
        return value+gradient @ (minimum-reference), (gradient @ transform)/np.sqrt(frequencies)
    spectroscopy = request['spectroscopy']
    controls = dict(request.get('controls', {})); response = dict(request.get('response', {}))
    from inspect import signature
    from .vibronic_observables import harmonic_operator_spectrum
    allowed_controls = set(signature(harmonic_operator_spectrum).parameters)-{'model', 'constants', 'forms', 'gradients', 'kind'}
    if set(controls)-allowed_controls:
        raise ValueError(f'unknown optical controls: {sorted(set(controls)-allowed_controls)}')
    response_fields = {'photon_wavenumber_cm', 'tensor_origin_bohr', 'gauge'} if spectroscopy == 'TPCD' else ({
        'transition_zeeman_cm_per_T', 'lower_zeeman_cm_per_T', 'electronic_population',
        'lower_zeeman_matrices_cm_per_T', 'upper_zeeman_matrices_cm_per_T', 'lower_ensemble_mean_zeeman_cm_per_T'} if spectroscopy == 'MCD' else set())
    if set(response)-response_fields:
        raise ValueError(f'unknown response controls for {spectroscopy}: {sorted(set(response)-response_fields)}')
    for key in ('lower_zeeman_matrices_cm_per_T', 'upper_zeeman_matrices_cm_per_T'):
        if key in response: response[key] = _complex_input(response[key])
    if spectroscopy in ('ECD', 'CPL'):
        expected = {'electric_dipole_au', 'magnetic_dipole_over_c_au'}
        electric, de = prop('electric_dipole_au', (3,)); magnetic, dm = prop('magnetic_dipole_over_c_au', (3,))
        if response: raise ValueError('ECD/CPL do not accept electronic-response controls')
        result = harmonic_circular_spectrum(model, electric, magnetic, electric_gradient=de, magnetic_gradient=dm,
                                             spectroscopy=spectroscopy, **controls)
    elif spectroscopy in ('TPA', 'TPCD'):
        expected = {'electric_tensor'} if spectroscopy == 'TPA' else {'electric_tensor', 'magnetic_tensor_over_c', 'quadrupole_tensor'}
        electric, de = prop('electric_tensor', (3, 3))
        magnetic, dm = prop('magnetic_tensor_over_c', (3, 3), spectroscopy == 'TPCD')
        quadrupole, dq = prop('quadrupole_tensor', (3, 3, 3), spectroscopy == 'TPCD')
        result = harmonic_two_photon_spectrum(model, electric, electric_gradient=de,
                    magnetic_tensor_over_c=magnetic, magnetic_gradient=dm, quadrupole_tensor=quadrupole,
                    quadrupole_gradient=dq, spectroscopy=spectroscopy, **response, **controls)
    elif spectroscopy == 'MCD':
        expected = {'electric_dipole_au', 'electric_field_response_per_T'}
        shape = np.shape(_complex_input(properties['electric_dipole_au']['value']))
        if shape != (3,) and (len(shape) != 3 or shape[-1] != 3):
            raise ValueError('MCD dipoles require shape (3,) or (n_lower,n_upper,3)')
        electric, de = prop('electric_dipole_au', shape)
        derivative, dd = prop('electric_field_response_per_T', shape+(3,))
        solver = harmonic_mcd_spectrum if shape == (3,) else harmonic_mcd_manifold_spectrum
        result = solver(model, electric, derivative, electric_gradient=de,
                                       response_gradient=dd, **response, **controls)
    else:
        raise ValueError('spectroscopy must be ECD, CPL, MCD, TPA or TPCD')
    if set(properties)-expected:
        raise ValueError(f'unknown properties for {spectroscopy}: {sorted(set(properties)-expected)}')
    identity = hashlib.sha256(json.dumps(configuration, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return vibronic_json_value(dict(schema='matrix.sati.optical_result.v1', configuration_sha256=identity,
                                    provenance=model.provenance, result=result))


__all__ = ['run_esalc_optical_workflow']
