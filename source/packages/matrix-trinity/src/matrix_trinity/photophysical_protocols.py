"""KEYMAKER preflight for radiative, radiationless and dissociative rates.

Only evidence-backed capabilities for the requested method, basis and state
model are eligible. This planner never submits electronic jobs, adds keywords,
or substitutes a missing transition operator. Existing rate owners execute
subsequent, separately authorized stages.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math

PROCESSES = {
    'fluorescence': {'operator': ('electric_dipole',),
        'derivative': 'electric_dipole_gradient',
        'preferred': ('gaussian', 'orca', 'turbomole'),
        'owner': 'matrix_trinity.radiative_td.harmonic_radiative_emission_rate_td',
        'ti_owner': 'matrix_trinity.vibronic_rates.harmonic_radiative_emission_rate'},
    'phosphorescence': {'operator': ('soc_matrix', 'electric_dipole', 'electronic_state_tracking'),
        'derivative': 'soc_mixed_dipole_gradient',
        'preferred': ('orca', 'turbomole', 'gaussian'),
        'owner': 'matrix_trinity.radiative_td.harmonic_radiative_emission_rate_td',
        'ti_owner': 'matrix_trinity.vibronic_rates.harmonic_radiative_emission_rate'},
    'internal_conversion': {'operator': ('complete_nonadiabatic_operator', 'electronic_state_tracking'),
        'derivative': None, 'preferred': ('turbomole', 'orca', 'gaussian'),
        'owner': 'matrix_trinity.vibronic_rates.harmonic_nonradiative_rate',
        'ti_owner': 'matrix_trinity.vibronic_rates.harmonic_nonradiative_rate'},
    'intersystem_crossing': {'operator': ('soc_matrix', 'electronic_state_tracking'),
        'derivative': 'soc_gradient', 'preferred': ('orca', 'turbomole', 'gaussian'),
        'owner': 'matrix_trinity.vibronic_rates.harmonic_nonradiative_rate',
        'ti_owner': 'matrix_trinity.vibronic_rates.harmonic_nonradiative_rate'},
    'photodissociation': {'operator': ('electric_dipole', 'electronic_state_tracking', 'dissociation_asymptote'),
        'derivative': 'electric_dipole_gradient',
        'preferred': ('gaussian', 'orca', 'turbomole'), 'owner': None, 'ti_owner': None},
}


class MissingPhotophysicalCapability(RuntimeError):
    """A required datum/solver is unavailable; the workflow must stop."""
    def __init__(self, report):
        self.report = report
        super().__init__('Photophysical workflow stopped: '+report['message'])


def _certified(profile, feature, method, basis, state_model):
    return any(item.get('feature') == feature and item.get('certified') is True
               and item.get('method') == method and item.get('basis') == basis
               and item.get('state_model') == state_model and item.get('evidence')
               for item in profile.get('capability_records', ()))


def prepare_photophysical_rate_request(request: Mapping) -> dict:
    """Select QM providers by complete capability coverage, then cost preference.

    backend_profiles describe the actual installed versions, not a program-wide
    promise. Different certified providers may supply nuclear and operator data;
    their method/basis, state identities, frames and units must be reconciled.
    An unavailable feature raises before any acquisition or rate evaluation.
    """
    if request.get('schema') != 'matrix.trinity.photophysical_rate_request.v1':
        raise ValueError('Explicit photophysical-rate request schema required')
    process = request.get('process')
    if process not in PROCESSES:
        raise ValueError('Unsupported photophysical process')
    specification = PROCESSES[process]
    for key in ('method', 'basis', 'state_model'):
        if not isinstance(request.get(key), str) or not request[key].strip():
            raise ValueError('Explicit '+key+' required')
    profiles = list(request.get('backend_profiles', ()))
    required = {'nuclear': ['energy', 'geometry', 'analytic_hessian'],
                'operator': list(specification['operator'])}
    order = request.get('operator_order', 'constant')
    if order not in ('constant', 'linear'):
        raise ValueError('Operator order must be constant or linear')
    if request.get('constant_operator_forbidden_by_symmetry') and order != 'linear':
        raise ValueError('A symmetry-forbidden constant operator requires its linear coupling')
    if order == 'linear' and specification['derivative']:
        required['operator'].append(specification['derivative'])
    if process == 'photodissociation':
        required['nuclear'] = ['energy', 'geometry', 'dissociation_potential']
    nuclear_model = request.get('nuclear_model', 'harmonic')
    if nuclear_model not in ('harmonic', 'dvr_bath', 'vpt2', 'vci'):
        raise ValueError('Explicit supported nuclear model required')
    if nuclear_model == 'vpt2':
        required['nuclear'].append('cubic_and_semidiagonal_quartic_field')
    if nuclear_model in ('dvr_bath', 'vci'):
        required['nuclear'].append('frozen_qualified_analytic_potential')
    selected = {}
    failures = {}
    theory_by_role = request.get('theory_by_role', {})
    for role, features in required.items():
        theory = theory_by_role.get(role, request)
        if any(not theory.get(key) for key in ('method', 'basis', 'state_model')):
            raise ValueError('Each explicitly different role theory needs method, basis and state_model')
        preference = ('gaussian', 'orca', 'turbomole') if role == 'nuclear' else specification['preferred']
        eligible = []
        rejected = []
        for profile in profiles:
            name = str(profile.get('name', '')).casefold()
            missing = [feature for feature in features if not _certified(
                profile, feature, theory['method'], theory['basis'], theory['state_model'])]
            if profile.get('available') is not True or not profile.get('version') or not profile.get('executable'):
                missing = ['installed_version_and_executable', *missing]
            if missing:
                rejected.append(dict(backend=name, missing=missing))
            else:
                # Explicit measured estimates take priority over the documented default order.
                cost = profile.get('estimated_cost_seconds', {}).get(role)
                if cost is not None and (not isinstance(cost, (float, int)) or not 0 <= cost < float('inf')):
                    raise ValueError('Backend cost estimate must be finite and nonnegative')
                eligible.append(((cost is None, float(cost or 0), preference.index(name) if name in preference else len(preference)), profile))
        if not eligible:
            failures[role] = rejected or [{'backend': None, 'missing': features}]
        else:
            selected[role] = deepcopy(min(eligible, key=lambda x: x[0])[1])
    if failures:
        raise MissingPhotophysicalCapability(dict(status='blocked_missing_qm_data', process=process,
            message='No installed certified QM backend covers the required data for the selected method, basis and state model.',
            missing_by_role=failures, required=required, submitted_jobs=0))
    methods = request.get('rate_methods', ['TI', 'TD'])
    if not methods or any(m not in ('TI', 'TD') for m in methods):
        raise ValueError('TI and/or TD rate methods required')
    owners = {'TI': specification['ti_owner'], 'TD': specification['owner']}
    radiative = process in ('fluorescence', 'phosphorescence')
    if nuclear_model == 'vci':
        owner = 'matrix_trinity.common_mode_rates.common_vci_'
        owners = ({'TI': owner+'radiative_emission_rate', 'TD': None} if radiative else
                  {'TI': owner+'nonradiative_rate', 'TD': owner+'nonradiative_rate'})
    elif nuclear_model == 'vpt2':
        owners = {'TI': None if radiative else 'matrix_trinity.vibronic_rates.sam_vpt2_nonradiative_rate', 'TD': None}
    elif nuclear_model == 'dvr_bath':
        owners = ({'TI': None, 'TD': ['matrix_trinity.vibronic_product_trace.product_dipole_correlation',
                   'matrix_trinity.radiative_td.radiative_rate_from_td_density']} if radiative else
                  {'TI': 'matrix_trinity.vibronic_spectator_rates.variational_spectator_nonradiative_rate',
                   'TD': 'matrix_trinity.vibronic_rates.analytic_bath_nonradiative_rate'})
    if process != 'photodissociation' and any(owners[method] is None for method in methods):
        raise MissingPhotophysicalCapability(dict(status='blocked_missing_nuclear_rate_adapter', process=process,
            message='The requested nuclear model and TI/TD rate adapter are not both implemented; choose an explicitly supported route.',
            nuclear_model=nuclear_model, missing_methods=[m for m in methods if owners[m] is None], submitted_jobs=0))
    if process in ('internal_conversion', 'intersystem_crossing') and not request.get('energy_conservation_model'):
        raise ValueError('Explicit energy-conservation linewidth/bath model required')
    if process == 'phosphorescence':
        populations = request.get('spin_populations', ())
        if (not populations or any(not math.isfinite(float(p)) or p < 0 for p in populations)
                or not math.isclose(sum(populations), 1., rel_tol=0., abs_tol=1e-12)):
            raise ValueError('Explicit normalized nonnegative emitting spin populations required')
    if process == 'photodissociation':
        solver = request.get('outgoing_channel_solver', {})
        if solver.get('certified') is not True or not solver.get('owner_api') or not solver.get('evidence'):
            raise MissingPhotophysicalCapability(dict(status='blocked_missing_outgoing_solver',
                process=process, message='A certified outgoing-channel solver and flux normalization are required.', submitted_jobs=0))
        if not request.get('photon_flux'):
            raise ValueError('A photon flux is required to turn a dissociation cross section into a rate')
    return dict(schema='matrix.trinity.photophysical_rate_plan.v1', status='preflight_passed_no_jobs_submitted',
        process=process, selected_backends=selected, required=required, nuclear_model=nuclear_model,
        rate_methods=list(methods), operator_order=order,
        execution_owners=owners, theory_by_role=deepcopy(theory_by_role),
        stages=['reuse_verified_artifacts', 'identify_missing_inputs', 'finite_acquisition_plan',
                'explicit_keyword_and_resource_authorization', 'acquire_on_verified_compute_host',
                'track_states_and_phases', 'transport_all_operators_and_fields',
                'evaluate_TI_TD_with_existing_owners', 'check_rate_convergence', 'archive'],
        mandatory_gates=['No silent backend, method, operator or derivative substitution',
            'Complex phases and channel populations explicit; energy and operator units checked',
            'Finite DVR points and frozen fitted potential; no electronic grid refinement',
            'TI and TD use identical fields, gap, operator, temperature and linewidth',
            'Nuclear, electronic-manifold and statistical errors reported separately',
            'Resource ceiling explicit; retain and resume checkpoints'],
        submitted_jobs=0)


def _prepare(process, request):
    if request.get('process', process) != process:
        raise ValueError('Protocol and requested process differ')
    return prepare_photophysical_rate_request({**request, 'process': process})


def prepare_fluorescence(request):
    return _prepare('fluorescence', request)


def prepare_phosphorescence(request):
    return _prepare('phosphorescence', request)


def prepare_internal_conversion(request):
    return _prepare('internal_conversion', request)


def prepare_intersystem_crossing(request):
    return _prepare('intersystem_crossing', request)


def prepare_photodissociation(request):
    return _prepare('photodissociation', request)
