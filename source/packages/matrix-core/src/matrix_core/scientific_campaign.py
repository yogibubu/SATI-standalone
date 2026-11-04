"""Persistent campaign checkpoints and conservative reuse decisions."""
from __future__ import annotations

from copy import deepcopy
from functools import wraps
import fcntl
from importlib import metadata
from pathlib import Path
import platform
import socket
import sys
import uuid

from ._scientific_records import (fingerprint, scientific_identity, seal,
                                 text, verify, verify_plan)
from .artifact_integrity import artifact_digest
from .scientific_evidence import evaluate_scientific_plan

RUNTIME_SNAPSHOT_SCHEMA = 'matrix.keymaker.runtime_snapshot.v1'
CAMPAIGN_RUN_SCHEMA = 'matrix.keymaker.campaign_run.v1'


def _spec_digest(spec):
    # Updated estimates or wording do not invalidate unchanged scientific work.
    # Gates, targets, protocol bytes, inputs, parameters and runtime still do.
    return fingerprint({key: value for key, value in spec.items() if key != 'plan_sha256'})


def _mutating(method):
    @wraps(method)
    def locked(self, *args, **kwargs):
        with (self.registry.ledger.workspace.state/'scientific-campaign.lock').open('a') as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                return method(self, *args, **kwargs)
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return locked


def capture_runtime_snapshot(*, code_files, backend_files, backend_name, backend_version,
                             dependencies, configuration):
    """Capture this node's actual bytes and installed distribution versions.

    ``code_files`` must cover the scientific implementation used by the caller;
    KEYMAKER does not infer completeness or hash credentials/environment secrets.
    Backend/configuration files are supplied explicitly. Paths are informational;
    logical names and content digests define the portable runtime fingerprint.
    """
    for value, name in ((backend_name, 'backend_name'), (backend_version, 'backend_version')):
        text(value, name)
    if not isinstance(code_files, dict) or not code_files or not isinstance(backend_files, dict):
        raise ValueError('explicit code files and backend-file mapping are required')
    if not isinstance(configuration, dict) or not isinstance(dependencies, (list, tuple)):
        raise ValueError('explicit configuration and dependency names are required')
    def hashes(files):
        result = {}
        for name, path in files.items():
            text(name, 'logical file name')
            result[name] = artifact_digest(Path(path))
        return result
    versions = {}
    for distribution in dependencies:
        text(distribution, 'distribution name')
        versions[distribution] = metadata.version(distribution)
    return seal(dict(schema=RUNTIME_SNAPSHOT_SCHEMA, host=socket.gethostname(),
                     system=platform.system(), architecture=platform.machine(),
                     python=platform.python_version(), python_abi=sys.implementation.cache_tag,
                     interpreter_sha256=artifact_digest(Path(sys.executable).resolve()),
                     code_files=hashes(code_files), backend_files=hashes(backend_files),
                     backend_name=backend_name, backend_version=backend_version,
                     dependencies=versions, configuration=deepcopy(configuration)), 'runtime_sha256')


class ScientificCampaign:
    """Checkpoint state through the existing locked, hash-chained ledger.

    This class never submits a calculation. Existing executors supply run IDs,
    remote handles, output bindings and tool-owned evidence at their boundaries.
    """
    def __init__(self, registry):
        self.registry = registry

    def runs(self):
        self.registry.records()  # Verify the shared ledger before replaying it.
        runs = {}
        for event in self.registry.ledger.events():
            run = event.payload.get('scientific_run')
            if run is not None:
                verify(run, 'record_sha256', CAMPAIGN_RUN_SCHEMA)
                runs[run['run_id']] = run
        return deepcopy(list(runs.values()))

    def get(self, run_id):
        matches = [r for r in self.runs() if r['run_id'] == run_id]
        if len(matches) != 1:
            raise KeyError(f'unknown campaign run: {run_id}')
        return matches[0]

    def _spec(self, *, plan, protocol_id, parameters, input_bindings, input_identities,
              output_identity, runtime):
        verify_plan(plan, self.registry.library)
        manifest = self.registry.library.resolve(protocol_id)
        if protocol_id not in {s['protocol_id'] for s in plan['steps']}:
            raise ValueError('protocol must use its exact planned identifier')
        verify(runtime, 'runtime_sha256', RUNTIME_SNAPSHOT_SCHEMA)
        if not isinstance(parameters, dict) or not isinstance(input_bindings, dict) or not isinstance(input_identities, dict):
            raise ValueError('parameters, input bindings and input identities must be objects')
        declared = {i.name: i for i in manifest.inputs}
        if set(input_bindings)-declared.keys() or set(input_identities) != set(input_bindings):
            raise ValueError('input bindings and consumer identities must match declared input names')
        if any(i.required and i.name not in input_bindings for i in manifest.inputs):
            raise ValueError('all required input artifacts must be bound before preparing a run')
        bound = {}
        for name, identifier in input_bindings.items():
            record = self.registry.get(identifier, expected_identity=input_identities[name],
                                       artifact_schema=declared[name].schema)
            bound[name] = dict(artifact_id=identifier, content_sha256=record['content_sha256'],
                               identity=record['identity'])
        return dict(plan_sha256=plan['plan_sha256'], protocol_id=protocol_id,
                    protocol_sha256=manifest.sha256, parameters=deepcopy(parameters),
                    protocol_chain=[dict(protocol_id=s['protocol_id'], protocol_sha256=s['protocol_sha256'])
                                    for s in plan['steps']],
                    acceptance_criteria=[deepcopy(c) for c in plan['acceptance_criteria']
                                         if c['protocol_id'] == protocol_id],
                    inputs=bound, output_identity=scientific_identity(output_identity),
                    runtime=deepcopy(runtime))

    def _append(self, run, event_type):
        run = deepcopy(run)
        run.pop('record_sha256', None)
        run = seal(run, 'record_sha256')
        self.registry.ledger.append(event_type, actor='keymaker', subject_id=run['run_id'],
                                    payload={'scientific_run': run})
        return run

    @_mutating
    def prepare(self, **specification):
        spec = self._spec(**specification)
        run = dict(schema=CAMPAIGN_RUN_SCHEMA, run_id='run-'+uuid.uuid4().hex,
                   spec=spec, spec_sha256=_spec_digest(spec), status='prepared',
                   plan=deepcopy(specification['plan']), remote_handle=None,
                   outputs={}, evidence_ids=[], review=None, measurements=None)
        return self._append(run, 'job.queued')

    @_mutating
    def start(self, run_id, *, remote_handle=None):
        run = self.get(run_id)
        if run['status'] != 'prepared':
            raise ValueError('only a prepared run can start; use a new attempt for retries')
        if remote_handle is not None:
            verify(remote_handle, 'handle_sha256', 'matrix.keymaker.remote_resume_handle.v1')
            if remote_handle['runtime'] != run['spec']['runtime']:
                raise ValueError('remote handle does not match the prepared runtime')
            if (run['spec']['parameters'].get('execution_request_sha256') != remote_handle['request_sha256']
                    or run['spec']['parameters'].get('execution_command') != remote_handle['command']):
                raise ValueError('remote handle does not match the prepared request and command')
        run.update(status='running', remote_handle=deepcopy(remote_handle))
        return self._append(run, 'job.running')

    @_mutating
    def interrupt(self, run_id, *, reason, unknown=False):
        run = self.get(run_id)
        if run['status'] not in {'running', 'interrupted', 'unknown'}:
            raise ValueError('only an active run can be interrupted')
        text(reason, 'interruption reason')
        run.update(status='unknown' if unknown else 'interrupted', interruption_reason=reason)
        return self._append(run, 'job.failed')

    @_mutating
    def fail(self, run_id, *, reason):
        run = self.get(run_id)
        if run['status'] not in {'prepared', 'running', 'interrupted', 'unknown'}:
            raise ValueError('cannot fail a terminal run')
        text(reason, 'failure reason')
        run.update(status='failed', failure_reason=reason)
        return self._append(run, 'job.failed')

    @_mutating
    def complete(self, run_id, *, plan, outputs, evidence_ids, expected_identities, measurements):
        from .scientific_costs import validate_measurements

        run = self.get(run_id)
        if run['status'] not in {'running', 'interrupted', 'unknown'}:
            raise ValueError('only an existing active run can complete')
        if plan['plan_sha256'] != run['spec']['plan_sha256']:
            raise ValueError('completion plan does not match the prepared run')
        manifest = self.registry.library.resolve(run['spec']['protocol_id'])
        declared = {o.name: o for o in manifest.outputs}
        if not isinstance(outputs, dict) or set(outputs)-declared.keys() or any(o.required and o.name not in outputs for o in manifest.outputs):
            raise ValueError('outputs must bind all required output names and no unknown names')
        if len(set(outputs.values())) != len(outputs):
            raise ValueError('each named output requires its own artifact identity')
        for name, identifier in outputs.items():
            record = self.registry.get(identifier, expected_identity=run['spec']['output_identity'],
                                       artifact_schema=declared[name].schema)
            if record['run_id'] != run_id or record['protocol_id'] != manifest.protocol_id:
                raise ValueError('output belongs to another run or protocol')
        review = evaluate_scientific_plan(plan, self.registry, evidence_ids=evidence_ids,
                                           expected_identities=expected_identities)
        own = [c for c in review['criteria'] if c['protocol_id'] == manifest.protocol_id]
        if not own or any(c['status'] != 'passed' or c.get('run_id') != run_id for c in own):
            raise ValueError('all owner criteria must pass for this exact run before completion')
        if any(set(c['source_artifact_ids'])-set(outputs.values()) for c in own):
            raise ValueError('gate evidence does not refer exclusively to the completed outputs')
        run.update(status='completed', outputs=deepcopy(outputs), evidence_ids=list(evidence_ids),
                   review=review, measurements=validate_measurements(measurements))
        return self._append(run, 'job.completed')

    def reuse_decision(self, run_id, **specification):
        run = self.get(run_id)
        spec = self._spec(**specification)
        decision = dict(run_id=run_id, action='new_attempt_requires_authorization', reusable=False,
                        reason='The run is not complete', execution_authorized=False)
        if run['spec_sha256'] != _spec_digest(spec):
            decision['reason'] = 'Plan, parameters, input identity/bytes, code or runtime changed'
            return decision
        if run['status'] in {'running', 'interrupted', 'unknown'}:
            decision.update(action='reattach_existing_job' if run['remote_handle'] else 'inspect_checkpoint',
                            reason='Existing job state must be reconciled; do not resubmit implicitly',
                            remote_handle=deepcopy(run['remote_handle']))
        elif run['status'] == 'completed':
            try:
                for identifier in run['outputs'].values():
                    self.registry.get(identifier, expected_identity=spec['output_identity'])
                review = evaluate_scientific_plan(run['plan'], self.registry,
                                                  evidence_ids=run['evidence_ids'],
                                                  expected_identities=run['review']['expected_identities'])
                own = [c for c in review['criteria'] if c['protocol_id'] == spec['protocol_id']]
                if not own or any(c['status'] != 'passed' for c in own):
                    raise ValueError('owner evidence no longer passes')
            except (OSError, KeyError, ValueError) as exc:
                decision['reason'] = 'Cached result is invalid: '+str(exc)
            else:
                decision.update(action='reuse_verified_outputs', reusable=True,
                                reason='Exact inputs/runtime and current output/evidence checks match',
                                evidence_plan_sha256=run['plan']['plan_sha256'],
                                requested_plan_sha256=specification['plan']['plan_sha256'],
                                outputs=deepcopy(run['outputs']))
        elif run['status'] == 'prepared':
            decision.update(action='await_execution_authorization', reason='No execution has started')
        return decision
