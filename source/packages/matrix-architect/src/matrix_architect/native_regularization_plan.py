"""Deterministic, provenance-bound plans for native anharmonic qualification.

This owner prepares an experimental plan. It neither fits a field nor promotes
one: numerical minima, spectral and domain evidence must subsequently be run.
The policy is a development draft, not a validated molecular protocol.
"""
from pathlib import Path
import hashlib
import json


POLICY = {
    'id': 'zaff4-robust-regularization-plan.v1',
    'status': 'draft',
    'base_scales': 'bond=1/parent_Morse_alpha; angular=1 radian',
    'bond_scale_multipliers': [1., 2., 4.],
    'angular_scale_multiplier': 1.,
    'radius_catalog': [8., 4., 2., 1., .75, .625, .5, .375, .25,
                       .125, .0625, .03125, .015625],
    'neighbour_radius_ratios': [.8, 1., 1.2],
    'screen_seed': 20261023,
    'qualification_seed': 20261024,
    'minimum_starts_per_field': 71,
    'maximum_domain_expansions': 4,
    'domain_expansion_factor': 1.25,
    'low_levels_per_subspace': 4,
    'tolerances': {
        'grid_energy_cm': .01,
        'box_energy_cm': .01,
        'basis_energy_cm': .1,
        'regularization_level_spread_cm': 2.,
        'cumulative_pruning_level_change_cm': .5,
        'probability_outside_search_domain': 1e-7,
        'outer_box_probability': 1e-7,
        'projected_gradient': 1e-6,
        'minimum_projected_curvature': 1e-7,
    },
    'selection': [
        'Reject any candidate without a passing frozen-local-fit report.',
        'Screen fixed candidate families; accumulate counterexamples only in the screening set.',
        'Test spectra only for screened candidates; both radius neighbours must also pass minima checks.',
        'Require numerical grid, box, basis and all-mode/active-space controls before comparing spectra.',
        'Require a 2 cm-1 maximum spread of corresponding low levels over the screened radius neighbourhood.',
        'Choose the passing candidate with the smallest maximum level spread, then the smallest bond multiplier, then the largest radius.',
        'Confirm the selected candidate and neighbours with the reserved independent starts and occupied-domain check.',
        'A reserved-set failure rejects qualification; changing the plan requires a new recorded revision.',
    ],
    'scope': 'low vibrational levels; high-energy accepting states and rates require separate qualification',
    'development_system': '4-pyrone S0/T1; a second molecule is required before general validation',
}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def build_native_regularization_plan(request_path):
    """Bind an immutable draft policy to exact inputs, code and resources.

    Repeating the same request and inputs produces the same plan bytes. A
    changed source never silently reuses an existing plan. No jobs are started.
    """
    request_path = Path(request_path).resolve()
    raw = request_path.read_bytes()
    request = json.loads(raw)
    required = {'schema', 'sources', 'code_revision', 'code_snapshot_sha256',
                'required_host', 'memory_gib', 'workers', 'output'}
    if set(request) != required or request['schema'] != 'matrix.architect.native_regularization_plan_request.v1':
        raise ValueError('complete native regularization plan request required')
    names = {'parent_field', 'local_fitted_field', 'source_derivatives',
             'modal_frame', 'minima_search', 'local_fit_report', 'code_manifest', 'runtime_manifest'}
    if not isinstance(request['sources'], dict) or set(request['sources']) != names:
        raise ValueError('parent, fitted field, derivative gauge, modal frame, search and fit evidence required')
    if (request['memory_gib'] != 128 or not isinstance(request['workers'], int)
            or isinstance(request['workers'], bool) or not 1 <= request['workers'] <= 128
            or not isinstance(request['required_host'], str) or not request['required_host'].strip()
            or not isinstance(request['code_revision'], str) or not request['code_revision'].strip()):
        raise ValueError('explicit code revision, remote host, workers and 128 GiB required')
    patch = request['code_snapshot_sha256']
    if not isinstance(patch, str) or len(patch) != 64 or any(c not in '0123456789abcdef' for c in patch):
        raise ValueError('SHA256 of the actual code manifest required, including untracked source files')
    sources = {}
    for name, relative in sorted(request['sources'].items()):
        path = (request_path.parent / relative).resolve()
        sources[name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if sources['code_manifest']['sha256'] != patch:
        raise ValueError('declared code snapshot differs from supplied code manifest')
    code = json.loads(Path(sources['code_manifest']['path']).read_text())
    if set(code) != {'root', 'files'} or not isinstance(code['files'], dict) or not code['files']:
        raise ValueError('nonempty exact code inventory required')
    for relative, expected in code['files'].items():
        if hashlib.sha256((Path(code['root'])/relative).read_bytes()).hexdigest() != expected:
            raise ValueError('code changed since the recorded snapshot: '+relative)
    policy_hash = hashlib.sha256(_canonical(POLICY)).hexdigest()
    plan = dict(schema='matrix.architect.native_regularization_plan.v1',
        policy=POLICY, policy_sha256=policy_hash,
        request_sha256=hashlib.sha256(raw).hexdigest(), sources=sources,
        code_revision=request['code_revision'], code_snapshot_sha256=patch,
        required_host=request['required_host'], memory_gib=128, workers=request['workers'],
        candidate_count=len(POLICY['bond_scale_multipliers'])*len(POLICY['radius_catalog']),
        execution_started=False, qualified=False,
        candidate_budget_note='Neighbours and independent qualification are additional; no automatic expansion beyond this declared policy.')
    output = (request_path.parent / request['output']).resolve()
    encoded = _canonical(plan) + b'\n'
    if output.exists():
        if output.read_bytes() != encoded:
            raise FileExistsError('saved plan differs; use a new revision and output path')
    else:
        with output.open('xb') as stream:
            stream.write(encoded)
    return plan
