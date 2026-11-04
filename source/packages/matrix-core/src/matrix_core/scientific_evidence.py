"""Evaluate frozen KEYMAKER gates using registered, tool-owned evidence."""
from __future__ import annotations

from copy import deepcopy

from ._scientific_records import fields, number, read_json, scientific_identity, seal, text, verify_plan
from .scientific_validation import SCIENTIFIC_VALIDATION_SCHEMA, ScientificValidationRecord

OWNER_EVIDENCE_SCHEMA = 'matrix.keymaker.owner_evidence.v1'
EVIDENCE_REVIEW_SCHEMA = 'matrix.keymaker.evidence_review.v1'


def _owner_names(manifest):
    names = {manifest.owner_operation.upper(), manifest.owner_api.split('.')[0].removeprefix('matrix_').upper()}
    if manifest.owner_operation == 'rovib':
        names.add('LOCK')
    if manifest.owner_operation == 'trinity':
        names.add('SATI')
    return names


def owner_validation_record(payload):
    """Read an existing tool-owned record with strict scalar/type checks."""
    fields(payload, ('schema', 'owner', 'subject', 'status', 'metrics', 'thresholds', 'context'))
    if payload['schema'] != SCIENTIFIC_VALIDATION_SCHEMA or payload['status'] not in ('PASS', 'FAIL'):
        raise ValueError('unsupported owner validation schema or status')
    for name in ('metrics', 'thresholds', 'context'):
        if not isinstance(payload[name], dict):
            raise ValueError(f'{name} must be an object')
    for key, value in payload['metrics'].items():
        text(key, 'metric'); number(value, key, minimum=float('-inf'))
    for key, value in payload['thresholds'].items():
        text(key, 'threshold'); number(value, key)
    return ScientificValidationRecord(owner=text(payload['owner'], 'owner'),
                                      subject=text(payload['subject'], 'subject'),
                                      passed=payload['status'] == 'PASS', metrics=payload['metrics'],
                                      thresholds=payload['thresholds'], context=payload['context'])


def _check_sources(registry, sources, *, manifest, run_id, identity):
    if not isinstance(sources, (list, tuple)) or not sources or len(set(sources)) != len(sources):
        raise ValueError('distinct registered output artifacts are required')
    schemas = {output.schema for output in manifest.outputs}
    for artifact_id in sources:
        record = registry.get(artifact_id, expected_identity=identity)
        if (record['protocol_id'] != manifest.protocol_id or record['run_id'] != run_id
                or record['protocol_sha256'] != manifest.sha256
                or record['role'] != 'output' or record['artifact_schema'] not in schemas):
            raise ValueError('evidence must refer to declared outputs of the same protocol and run')


def register_owner_evidence(registry, *, plan, criterion_id, validation, metric_units,
                            source_artifact_ids, identity, run_id):
    """Adapt the existing ScientificValidationRecord without interpreting physics.

    The tool must identify the exact criterion in ``subject``. Source artifacts
    must already be registered. Caller-supplied producer identity is not an
    authentication signature and cannot replace independent scientific review.
    """
    verify_plan(plan, registry.library)
    criteria = {c['criterion_id']: c for c in plan['acceptance_criteria']}
    if criterion_id not in criteria:
        raise ValueError('unknown criterion in the frozen plan')
    criterion = criteria[criterion_id]
    manifest = registry.library.resolve(criterion['protocol_id'])
    if not isinstance(validation, ScientificValidationRecord) or not isinstance(validation.passed, bool):
        raise ValueError('an explicit tool-owned ScientificValidationRecord is required')
    if validation.owner not in _owner_names(manifest) or validation.subject != criterion_id:
        raise ValueError('validation owner or criterion does not match the plan')
    if not isinstance(metric_units, dict) or set(metric_units) != set(validation.metrics):
        raise ValueError('every validation metric requires an explicit unit')
    for unit in metric_units.values():
        text(unit, 'metric unit')
    scientific_identity(identity)
    _check_sources(registry, source_artifact_ids, manifest=manifest, run_id=run_id, identity=identity)
    payload = dict(schema=OWNER_EVIDENCE_SCHEMA, plan_sha256=plan['plan_sha256'],
                   criterion_id=criterion_id, protocol_id=manifest.protocol_id,
                   protocol_sha256=manifest.sha256, run_id=run_id,
                   source_artifact_ids=list(source_artifact_ids), metric_units=deepcopy(metric_units),
                   validation=validation.to_dict())
    return registry.register_json(payload, artifact_schema=OWNER_EVIDENCE_SCHEMA, identity=identity,
                                  protocol_id=manifest.protocol_id, run_id=run_id, role='owner-evidence')


def evaluate_scientific_plan(plan, registry, *, evidence_ids, expected_identities):
    """Return passed/failed/unverified criteria; never execute an owner or fit.

    Full, consumer-specified scientific identity is required per protocol.
    More than one distinct evidence record for a gate is an explicit conflict.
    """
    verify_plan(plan, registry.library)
    protocols = {s['protocol_id'] for s in plan['steps']}
    if not isinstance(expected_identities, dict) or set(expected_identities) != protocols:
        raise ValueError('expected scientific identity is required for every planned protocol')
    for identity in expected_identities.values():
        scientific_identity(identity)
    if not isinstance(evidence_ids, (list, tuple)) or len(set(evidence_ids)) != len(evidence_ids):
        raise ValueError('evidence IDs must be a distinct list')
    grouped = {}
    known = {c['criterion_id'] for c in plan['acceptance_criteria']}
    for identifier in evidence_ids:
        record = registry.get(identifier, artifact_schema=OWNER_EVIDENCE_SCHEMA)
        if record['role'] != 'owner-evidence':
            raise ValueError('a registered owner-evidence artifact is required')
        protocol = record['protocol_id']
        if protocol not in expected_identities:
            raise ValueError('evidence protocol is outside the plan')
        registry.get(identifier, expected_identity=expected_identities[protocol])
        payload = read_json(registry.path(identifier))
        fields(payload, ('schema', 'plan_sha256', 'criterion_id', 'protocol_id', 'protocol_sha256',
                         'run_id', 'source_artifact_ids', 'metric_units', 'validation'))
        if (payload['schema'] != OWNER_EVIDENCE_SCHEMA or payload['plan_sha256'] != plan['plan_sha256']
                or payload['protocol_id'] != protocol or payload['run_id'] != record['run_id']
                or payload['protocol_sha256'] != registry.library.resolve(protocol).sha256
                or payload['protocol_sha256'] != record['protocol_sha256']
                or payload['criterion_id'] not in known):
            raise ValueError('evidence belongs to a different plan, protocol or run')
        validation = payload['validation']
        owner_validation_record(validation)
        if (validation['schema'] != SCIENTIFIC_VALIDATION_SCHEMA
                or validation['owner'] not in _owner_names(registry.library.resolve(protocol))
                or validation['subject'] != payload['criterion_id']
                or validation['status'] not in ('PASS', 'FAIL')):
            raise ValueError('invalid tool-owned validation identity or status')
        criterion = next(c for c in plan['acceptance_criteria'] if c['criterion_id'] == payload['criterion_id'])
        if criterion['protocol_id'] != protocol:
            raise ValueError('evidence criterion has a different scientific owner')
        if not isinstance(validation['metrics'], dict) or not validation['metrics']:
            raise ValueError('owner evidence requires numerical metrics')
        if not isinstance(payload['metric_units'], dict) or set(payload['metric_units']) != set(validation['metrics']):
            raise ValueError('missing metric units')
        for key, value in validation['metrics'].items():
            text(key, 'metric'); number(value, key, minimum=float('-inf'))
            text(payload['metric_units'][key], 'metric unit')
        _check_sources(registry, payload['source_artifact_ids'], manifest=registry.library.resolve(protocol),
                       run_id=record['run_id'], identity=expected_identities[protocol])
        grouped.setdefault(payload['criterion_id'], []).append((identifier, payload))
    results = []
    for criterion in plan['acceptance_criteria']:
        entries = grouped.get(criterion['criterion_id'], [])
        result = dict(criterion_id=criterion['criterion_id'], protocol_id=criterion['protocol_id'],
                      requirement=criterion['requirement'], evidence_ids=[i for i, _ in entries],
                      status='unverified', numerical_checks=[], reason='No owner evidence supplied')
        if len(entries) > 1:
            result.update(status='failed', reason='Conflicting evidence: choose the intended run explicitly')
        elif entries:
            _, payload = entries[0]
            validation = payload['validation']
            checks = []
            for target in criterion['numerical_targets']:
                value = validation['metrics'].get(target['metric'])
                unit_matches = payload['metric_units'].get(target['metric']) == target['unit']
                passed = value is not None and unit_matches and {
                    '<=': lambda: value <= target['threshold'],
                    '>=': lambda: value >= target['threshold'],
                    '==': lambda: value == target['threshold'],
                }[target['operator']]()
                checks.append(dict(metric=target['metric'], value=value, unit=target['unit'],
                                   actual_unit=payload['metric_units'].get(target['metric']),
                                   operator=target['operator'], threshold=target['threshold'], passed=passed,
                                   status='unverified' if value is None else 'passed' if passed else 'failed'))
            status = ('failed' if validation['status'] == 'FAIL' or any(c['status'] == 'failed' for c in checks)
                      else 'unverified' if any(c['status'] == 'unverified' for c in checks) else 'passed')
            result.update(status=status, numerical_checks=checks,
                          reason='Owner decision and explicit numerical targets evaluated',
                          run_id=payload['run_id'], source_artifact_ids=payload['source_artifact_ids'])
        results.append(result)
    state = ('failed' if any(c['status'] == 'failed' for c in results) else
             'passed' if results and all(c['status'] == 'passed' for c in results) else 'unverified')
    return seal(dict(schema=EVIDENCE_REVIEW_SCHEMA, plan_sha256=plan['plan_sha256'],
                     library_sha256=registry.library.sha256, status=state, criteria=results,
                     expected_identities=deepcopy(expected_identities), evidence_ids=list(evidence_ids),
                     execution_authorized=False, physical_accuracy_certified=False), 'review_sha256')
