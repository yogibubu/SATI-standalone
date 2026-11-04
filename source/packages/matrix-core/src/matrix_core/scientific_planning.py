"""KEYMAKER scientific proposals from the installed protocol library.

Planning is deterministic and read-only: no scientific owner, backend, artifact
payload, network service or job scheduler is invoked by this module.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from math import isfinite
from pathlib import Path
import re

from .protocol_library import load_keymaker_protocol_library

SCIENTIFIC_PLAN_SCHEMA = 'matrix.keymaker.scientific_plan.v1'
SCIENTIFIC_REQUEST_SCHEMA = 'matrix.keymaker.scientific_plan_request.v1'

# Language expansion supports retrieval; it does not infer a method or grant
# semantic equivalence between protocols. Ambiguous candidates remain explicit.
_EXPANSIONS = {
    'rotazionale': 'rotational', 'rotazionali': 'rotational',
    'rovibrazionale': 'rovibrational', 'rovibrazionali': 'rovibrational',
    'vibronico': 'vibronic', 'vibronici': 'vibronic',
    'vibrazionale': 'vibrational', 'vibrazionali': 'vibrational',
    'armonico': 'harmonic', 'armonica': 'harmonic',
    'anarmonico': 'anharmonic', 'anarmonica': 'anharmonic',
    'spettro': 'spectrum', 'spettri': 'spectrum',
    'spettroscopia': 'spectroscopy', 'ottimizzazione': 'optimization',
    'geometria': 'geometry', 'hessiano': 'hessian', 'hessiana': 'hessian',
    'gradiente': 'gradient', 'gradienti': 'gradient',
    'eccitati': 'excited', 'eccitato': 'excited',
    'accoppiamenti': 'coupling', 'tunnellamento': 'tunneling',
}
_STOP = frozenset('the a an to for and of with from in on di da del della delle un una con per e il lo la i gli le vorrei calcola calcolare voglio ottenere please compute calculate native matrix v1 v2 v3 v4 v5 v6'.split())


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a nonempty string')
    return value.strip()


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return float(value)


def _fields(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= set(value) or set(value)-set(required)-set(optional):
        raise ValueError(f'expected fields {sorted(required)}, optional {sorted(optional)}')


def _words(text):
    words = re.findall(r'[a-z0-9]+', text.casefold())
    return {_EXPANSIONS.get(word, word) for word in words if word not in _STOP and not word.isdecimal()}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _validate_request(request):
    _fields(request, ('schema', 'objective'), ('protocol_id', 'available_artifacts',
            'cost_estimates', 'maximum_serial_seconds', 'acceptance_targets'))
    if request['schema'] != SCIENTIFIC_REQUEST_SCHEMA:
        raise ValueError('unsupported scientific planning request schema')
    _text(request['objective'], 'objective')
    if 'protocol_id' in request:
        _text(request['protocol_id'], 'protocol_id')
    artifacts = request.get('available_artifacts', [])
    if not isinstance(artifacts, list):
        raise ValueError('available_artifacts must be a list')
    identities = set()
    for artifact in artifacts:
        _fields(artifact, ('name', 'schema', 'reference'), ('protocol_id',))
        for name, value in artifact.items():
            _text(value, f'artifact {name}')
        identity = (artifact.get('protocol_id'), artifact['name'], artifact['schema'])
        if identity in identities:
            raise ValueError('duplicate artifact declaration')
        identities.add(identity)
    estimates = request.get('cost_estimates', {})
    if not isinstance(estimates, dict):
        raise ValueError('cost_estimates must be a protocol-keyed object')
    for identifier, estimate in estimates.items():
        _text(identifier, 'cost protocol_id')
        _fields(estimate, ('work_units', 'unit', 'seconds_per_unit', 'calibration_reference'))
        count = estimate['work_units']
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError('work_units must be a positive integer')
        _positive(estimate['seconds_per_unit'], 'seconds_per_unit')
        _text(estimate['unit'], 'work unit')
        _text(estimate['calibration_reference'], 'calibration_reference')
    if 'maximum_serial_seconds' in request:
        _positive(request['maximum_serial_seconds'], 'maximum_serial_seconds')
    targets = request.get('acceptance_targets', [])
    if not isinstance(targets, list):
        raise ValueError('acceptance_targets must be a list')
    for target in targets:
        _fields(target, ('protocol_id', 'gate', 'metric', 'operator', 'threshold', 'unit'))
        for field in ('protocol_id', 'gate', 'metric', 'unit'):
            _text(target[field], field)
        if not isinstance(target['operator'], str) or target['operator'] not in {'<=', '>=', '=='}:
            raise ValueError('acceptance operator must be <=, >= or ==')
        number = target['threshold']
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not isfinite(number):
            raise ValueError('acceptance thresholds must be finite real numbers')


def _candidates(objective, library):
    query = _words(objective)
    ranked = []
    for manifest in library.list_protocols(status='approved'):
        searchable = ' '.join((manifest.protocol_id, manifest.title,
                               manifest.calculation_protocol, *manifest.aliases))
        matched = sorted(query & _words(searchable))
        if matched:
            ranked.append(dict(protocol_id=manifest.protocol_id, title=manifest.title,
                               score=len(matched), matched_terms=matched))
    return sorted(ranked, key=lambda item: (-item['score'], item['protocol_id']))


def build_scientific_plan(request, *, library=None):
    """Propose an auditable plan without authorizing or executing its steps.

    Artifacts are declarations, never certified as valid from their paths.
    Cost models are caller-supplied per-protocol calibrations, not guessed from
    molecular size. Unknown steps keep the total estimate explicitly partial.
    """
    _validate_request(request)
    request = deepcopy(request)
    library = library if library is not None else load_keymaker_protocol_library()
    candidates = _candidates(request['objective'], library)
    selected = None
    reason = 'No approved protocol matches the objective; specify a registered protocol or clarify the objective.'
    if request.get('protocol_id'):
        selected = library.resolve(request['protocol_id'])
        if selected.status != 'approved':
            raise ValueError('scientific planning requires an approved protocol')
        reason = 'Explicit protocol selection; compatibility with the scientific objective requires review.'
    elif candidates and candidates[0]['score'] >= 2 and (
        len(candidates) == 1 or candidates[0]['score'] >= candidates[1]['score']+2
    ):
        selected = library.resolve(candidates[0]['protocol_id'])
        reason = 'Unique lexical proposal with an explicit score margin; scientific review remains required.'
    elif candidates:
        reason = 'The objective does not identify a unique protocol; review the candidates and specify protocol_id.'

    plan = dict(schema=SCIENTIFIC_PLAN_SCHEMA, objective=request['objective'],
                library_sha256=library.sha256, selection_reason=reason,
                candidates=candidates[:10], selected_protocol_id=None,
                status='needs_clarification', execution_authorized=False,
                physical_accuracy_certified=False, steps=[], missing_data=[],
                acceptance_criteria=[], cost=None,
                warnings=['Artifact references are unvalidated declarations.',
                          'Protocol registration does not certify physical accuracy.',
                          'No scientific calculation or job submission was performed.'])
    if selected is None:
        if request.get('cost_estimates') or request.get('acceptance_targets'):
            plan['warnings'].append('Supplied costs and targets remain unapplied until a protocol is selected.')
        return _seal(plan, request)

    chain = library.dependency_order(selected.protocol_id)
    ids = {manifest.protocol_id for manifest in chain}
    for manifest in chain:
        if manifest.status != 'approved':
            raise ValueError(f'dependency is not approved: {manifest.protocol_id}')
    estimates = request.get('cost_estimates', {})
    if set(estimates)-ids:
        raise ValueError('cost estimates must use exact protocol IDs in the selected dependency chain')
    artifacts = request.get('available_artifacts', [])
    for artifact in artifacts:
        if artifact.get('protocol_id') and artifact['protocol_id'] not in ids:
            raise ValueError('scoped artifacts must refer to the selected dependency chain')
    targets = request.get('acceptance_targets', [])
    known_gates = {(manifest.protocol_id, gate) for manifest in chain for gate in manifest.gates}
    if any((target['protocol_id'], target['gate']) not in known_gates for target in targets):
        raise ValueError('acceptance targets must refer to an exact registered protocol gate')
    producers = []
    known_seconds = 0.
    unknown_costs = []
    for manifest in chain:
        step_inputs = []
        for item in manifest.inputs:
            matches = [a for a in artifacts if a['name'] == item.name and a['schema'] == item.schema
                       and a.get('protocol_id', manifest.protocol_id) == manifest.protocol_id]
            if len(matches) > 1:
                raise ValueError(f'ambiguous artifact declaration: {manifest.protocol_id}/{item.name}')
            upstream = [p for p in producers if p['schema'] == item.schema]
            # Schema equality alone cannot establish semantic identity of two
            # scientific inputs. Offer upstream sources for explicit binding.
            state = 'declared_unvalidated' if matches else 'missing' if item.required else 'optional_absent'
            entry = dict(name=item.name, schema=item.schema, required=item.required,
                         status=state, reference=matches[0]['reference'] if matches else None,
                         upstream_candidates=deepcopy(upstream))
            step_inputs.append(entry)
            if item.required and not matches:
                plan['missing_data'].append(dict(protocol_id=manifest.protocol_id, **entry))
        estimate = estimates.get(manifest.protocol_id)
        if estimate is None:
            cost = dict(status='unknown', serial_seconds=None,
                        reason='Workload and a per-unit calibration for this protocol are required.')
            unknown_costs.append(manifest.protocol_id)
        else:
            seconds = estimate['work_units']*estimate['seconds_per_unit']
            if not isfinite(seconds):
                raise ValueError('cost estimate exceeds finite numeric range')
            known_seconds += seconds
            cost = dict(status='calibrated_estimate', serial_seconds=seconds, **estimate)
        criteria = []
        for index, gate in enumerate(manifest.gates):
            criterion = dict(criterion_id=f'{manifest.protocol_id}:gate:{index+1}',
                             protocol_id=manifest.protocol_id, requirement=gate,
                             status='not_evaluated', evidence_required=True,
                             numerical_targets=[deepcopy(t) for t in targets
                                                if t['protocol_id'] == manifest.protocol_id and t['gate'] == gate])
            criteria.append(criterion)
        plan['acceptance_criteria'].extend(criteria)
        plan['steps'].append(dict(protocol_id=manifest.protocol_id, protocol_sha256=manifest.sha256,
                                  owner_operation=manifest.owner_operation, owner_api=manifest.owner_api,
                                  action=manifest.action, dependencies=list(manifest.dependencies),
                                  routes=[dict(menu=r.menu, action=r.action, options=dict(r.option_equals))
                                          for r in manifest.routes],
                                  inputs=step_inputs, outputs=[dict(name=o.name, schema=o.schema, required=o.required)
                                                               for o in manifest.outputs],
                                  coordinate_policy=manifest.coordinate_policy,
                                  backend_policy=manifest.backend_policy, cost=cost))
        producers.extend(dict(protocol_id=manifest.protocol_id, name=o.name, schema=o.schema,
                              status='planned_not_produced') for o in manifest.outputs)
    if not isfinite(known_seconds):
        raise ValueError('total cost exceeds finite numeric range')
    budget = request.get('maximum_serial_seconds')
    budget_status = ('not_specified' if budget is None else 'exceeded'
                     if known_seconds > budget else 'unknown' if unknown_costs else 'within_estimate')
    plan.update(selected_protocol_id=selected.protocol_id,
                status='needs_inputs' if plan['missing_data'] else 'proposed_for_review',
                cost=dict(status='partial' if unknown_costs else 'calibrated_estimate',
                          known_serial_seconds=known_seconds,
                          total_serial_seconds=None if unknown_costs else known_seconds,
                          unknown_protocols=unknown_costs, maximum_serial_seconds=budget,
                          budget_status=budget_status,
                          wall_seconds=None, memory_bytes=None,
                          assumptions='Sequential sum of supplied workload calibrations; no scheduling or parallel scaling model.'))
    if budget_status == 'exceeded':
        plan['status'] = 'over_budget'
    if unknown_costs:
        plan['warnings'].append('The total cost and budget feasibility remain unknown for uncalibrated steps.')
    if any(entry['upstream_candidates'] for entry in plan['missing_data']):
        plan['warnings'].append('Upstream artifacts require explicit semantic binding and validation; schema equality is insufficient.')
    return _seal(plan, request)


def _seal(plan, request):
    plan['request'] = request
    plan['plan_sha256'] = hashlib.sha256(_canonical(plan).encode()).hexdigest()
    return plan


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate request field: {key}')
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f'nonfinite JSON constant: {value}')


def main(argv=None):
    parser = argparse.ArgumentParser(description='KEYMAKER scientific planning; emits a proposal without execution')
    parser.add_argument('request', type=Path, help='Scientific planning request JSON')
    args = parser.parse_args(argv)
    try:
        with args.request.open('rb') as stream:
            raw = stream.read(1024*1024+1)
        if len(raw) > 1024*1024:
            raise ValueError('planning request exceeds 1 MiB')
        request = json.loads(raw, object_pairs_hook=_object, parse_constant=_constant)
        plan = build_scientific_plan(request)
    except (OSError, ValueError, KeyError, RuntimeError, OverflowError) as exc:
        parser.exit(2, f'KEYMAKER planning: {exc}\n')
    print(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
