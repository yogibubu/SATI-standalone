"""Opt-in collection of real owner diagnostics at scientific API boundaries."""
from __future__ import annotations

from contextvars import ContextVar
from copy import deepcopy

from ._scientific_records import verify_plan
from .scientific_evidence import _owner_names, register_owner_evidence
from .scientific_validation import ScientificValidationRecord

_ACTIVE = ContextVar('matrix_owner_evidence_capture', default=None)
OWNER_OBSERVATION_SCHEMA = 'matrix.keymaker.owner_observation.v1'


def emit_owner_validation(owner, subject, *, passed, metrics, units, thresholds=None, context=None, gates=()):
    capture = _ACTIVE.get()
    if capture is None:
        return
    validation = ScientificValidationRecord(owner=owner, subject=subject, passed=passed,
        metrics=metrics, thresholds=thresholds or {}, context=context or {})
    capture.observe(validation, units=units, gates=gates)


class OwnerEvidenceCapture:
    """Capture actual owner emissions, snapshot outputs and bind only exact owner gates.

    Unmapped numerical observations are retained without promoting protocol
    acceptance. Each owner supplies its gate text; callers cannot silently map
    a diagnostic to an unrelated gate. A failed execution never flushes success.
    """
    def __init__(self, registry, *, plan, protocol_id, run_id, identity, output_paths, observation_only=False):
        verify_plan(plan, registry.library)
        self.registry, self.plan, self.protocol_id = registry, deepcopy(plan), protocol_id
        self.run_id, self.identity, self.output_paths = run_id, deepcopy(identity), dict(output_paths)
        self.manifest = registry.library.resolve(protocol_id)
        self.observation_only = observation_only
        self.observations, self.outputs, self.evidence_ids, self.observation_ids = [], {}, [], []

    def __enter__(self):
        self.token = _ACTIVE.set(self)
        return self

    def observe(self, validation, *, units, gates):
        if validation.owner not in _owner_names(self.manifest):
            return  # Nested numerical helpers retain their separate scientific owner.
        if set(units) != set(validation.metrics):
            raise ValueError('owner diagnostics require explicit units for every metric')
        matched = tuple(gate for gate in gates if gate in self.manifest.gates)
        self.observations.append((validation, deepcopy(units), () if self.observation_only else matched))

    def __exit__(self, exception_type, *_):
        _ACTIVE.reset(self.token)
        if exception_type is not None:
            return False
        declared = {output.name: output for output in self.manifest.outputs}
        if set(self.output_paths)-declared.keys() or (not self.observation_only and
                any(o.required and o.name not in self.output_paths for o in self.manifest.outputs)):
            raise ValueError('capture requires every declared required scientific output')
        for name, path in self.output_paths.items():
            artifact = self.registry.register(path, artifact_schema=declared[name].schema,
                identity=self.identity, protocol_id=self.protocol_id, run_id=self.run_id, role='output')
            self.outputs[name] = artifact['artifact_id']
        for validation, units, gates in self.observations:
            record = self.registry.register_json(dict(schema=OWNER_OBSERVATION_SCHEMA,
                plan_sha256=self.plan['plan_sha256'], validation=validation.to_dict(), metric_units=units,
                source_artifact_ids=list(self.outputs.values())), artifact_schema=OWNER_OBSERVATION_SCHEMA,
                identity=self.identity, protocol_id=self.protocol_id, run_id=self.run_id, role='owner-observation')
            self.observation_ids.append(record['artifact_id'])
            for gate in gates:
                criterion = next(c for c in self.plan['acceptance_criteria']
                                 if c['protocol_id'] == self.protocol_id and c['requirement'] == gate)
                bound = ScientificValidationRecord(owner=validation.owner, subject=criterion['criterion_id'],
                    passed=validation.passed, metrics=validation.metrics, thresholds=validation.thresholds,
                    context={**validation.context, 'native_subject': validation.subject,
                             'observation_artifact_id': record['artifact_id']})
                artifact = register_owner_evidence(self.registry, plan=self.plan,
                    criterion_id=criterion['criterion_id'], validation=bound, metric_units=units,
                    source_artifact_ids=list(self.outputs.values()), identity=self.identity, run_id=self.run_id)
                self.evidence_ids.append(artifact['artifact_id'])
        return False
