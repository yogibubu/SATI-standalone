"""Content-addressed scientific artifacts linked to the existing provenance ledger."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil
import tempfile

from ._scientific_records import canonical, digest, fingerprint, scientific_identity, text
from .artifact_integrity import artifact_digest
from .atomic_io import atomic_output_path
from .provenance import ProvenanceLedger
from .protocol_library import load_keymaker_protocol_library

ARTIFACT_RECORD_SCHEMA = 'matrix.keymaker.scientific_artifact.v1'


class ScientificArtifactRegistry:
    """Snapshot bytes, record scientific identity, and recheck before reuse.

    Metadata is explicitly supplied by its producer. Hash verification detects
    byte changes; it does not certify that the declared physical identity is true.
    """
    def __init__(self, workspace, *, library=None):
        self.ledger = ProvenanceLedger(workspace)
        self.library = library if library is not None else load_keymaker_protocol_library()
        self.objects = self.ledger.workspace.state/'scientific-artifacts'

    def records(self):
        if not self.ledger.path.exists():
            return []
        result = self.ledger.verify()
        if not result.valid:
            raise ValueError('invalid provenance ledger: '+ '; '.join(result.errors))
        records = {}
        for event in self.ledger.events():
            record = event.payload.get('scientific_artifact')
            if record is None:
                continue
            if record.get('schema') != ARTIFACT_RECORD_SCHEMA:
                raise ValueError('unsupported scientific artifact record')
            identity = scientific_identity(record['identity'])
            binding = dict(content_sha256=record['content_sha256'], artifact_schema=record['artifact_schema'],
                           identity=identity, protocol_id=record['protocol_id'], protocol_sha256=record['protocol_sha256'],
                           run_id=record['run_id'], role=record['role'], metadata=record['metadata'])
            if record['artifact_id'] != fingerprint(binding):
                raise ValueError('artifact identity digest mismatch')
            records[record['artifact_id']] = record
        return deepcopy(list(records.values()))

    def register(self, source, *, artifact_schema, identity, protocol_id, run_id, role, metadata=None):
        identity = scientific_identity(identity)
        for value, name in ((artifact_schema, 'artifact_schema'), (run_id, 'run_id'), (role, 'role')):
            text(value, name)
        manifest = self.library.resolve(protocol_id)
        if manifest.status != 'approved':
            raise ValueError('artifact producer protocol must be approved')
        metadata = {} if metadata is None else deepcopy(metadata)
        if not isinstance(metadata, dict):
            raise ValueError('artifact metadata must be an object')
        canonical(metadata)
        self.records()  # Fail before writing if existing provenance is corrupt.
        self.objects.mkdir(parents=True, exist_ok=True)
        # Stage a private snapshot first: hashing and copying a mutable source
        # in separate passes would permit mismatched content under concurrent writes.
        with tempfile.TemporaryDirectory(dir=self.objects) as directory:
            snapshot = Path(directory)/'snapshot'
            shutil.copyfile(Path(source), snapshot)
            content_hash = artifact_digest(snapshot)
            target = self.objects/content_hash
            if target.exists():
                if artifact_digest(target) != content_hash:
                    raise ValueError('stored artifact content is corrupt')
            else:
                with atomic_output_path(target) as temporary:
                    shutil.copyfile(snapshot, temporary)
        binding = dict(content_sha256=content_hash, artifact_schema=artifact_schema,
                       identity=identity, protocol_id=manifest.protocol_id, protocol_sha256=manifest.sha256,
                       run_id=run_id, role=role, metadata=metadata)
        record = dict(schema=ARTIFACT_RECORD_SCHEMA, artifact_id=fingerprint(binding), **binding)
        self.ledger.append('artifact.registered', actor='keymaker', subject_id=record['artifact_id'],
                           payload={'scientific_artifact': record})
        return deepcopy(record)

    def register_json(self, payload, **metadata):
        self.objects.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.objects) as directory:
            path = Path(directory)/'record.json'
            path.write_text(canonical(payload)+'\n')
            return self.register(path, **metadata)

    def get(self, artifact_id, *, expected_identity=None, artifact_schema=None):
        digest(artifact_id, 'artifact_id')
        matches = [r for r in self.records() if r['artifact_id'] == artifact_id]
        if len(matches) != 1:
            raise KeyError(f'unknown scientific artifact: {artifact_id}')
        record = matches[0]
        path = self.objects/digest(record['content_sha256'])
        if not path.is_file() or artifact_digest(path) != record['content_sha256']:
            raise ValueError('scientific artifact bytes are missing or altered')
        if expected_identity is not None and record['identity'] != scientific_identity(expected_identity):
            raise ValueError('scientific artifact identity, frame or units are incompatible')
        if artifact_schema is not None and record['artifact_schema'] != artifact_schema:
            raise ValueError('scientific artifact schema is incompatible')
        return record

    def path(self, artifact_id, **requirements):
        record = self.get(artifact_id, **requirements)
        return self.objects/record['content_sha256']

    def planning_input(self, artifact_id, *, name, protocol_id, expected_identity, artifact_schema):
        """Resolve a checked registry artifact into the planner's existing input contract."""
        text(name, 'input name')
        return dict(name=name, schema=artifact_schema, protocol_id=protocol_id,
                    reference=str(self.path(artifact_id, expected_identity=expected_identity,
                                            artifact_schema=artifact_schema)))
