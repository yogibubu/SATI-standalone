"""Single-attempt remote job supervisor with identity, heartbeat and sealed completion."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

from ._scientific_records import digest, fingerprint, read_json, seal, text, verify
from .artifact_integrity import artifact_digest
from .atomic_io import atomic_json_write

JOB_IDENTITY_SCHEMA = 'matrix.keymaker.persistent_job_identity.v1'
JOB_COMPLETION_SCHEMA = 'matrix.keymaker.persistent_job_completion.v1'


def process_start(pid):
    result = subprocess.run(['ps', '-o', 'lstart=', '-p', str(pid)], capture_output=True, text=True, timeout=5,
                            env={'PATH': os.defpath, 'LC_ALL': 'C'})
    return result.stdout.strip() if result.returncode == 0 else ''


def supervise(directory, *, token, request_sha256, command_identity_sha256, command, heartbeat_seconds=1.):
    """Execute exactly once under a process-held lock; caller owns authorization."""
    directory = Path(directory).resolve()
    text(token, 'job token'); digest(request_sha256); digest(command_identity_sha256)
    if not command or not all(isinstance(p, str) and p for p in command):
        raise ValueError('an explicit argument vector is required')
    if heartbeat_seconds <= 0:
        raise ValueError('heartbeat interval must be positive')
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory/'identity.json').exists():
            raise FileExistsError('this attempt already has an identity; never relaunch it')
        if artifact_digest(directory/'request.json') != request_sha256:
            raise ValueError('staged request bytes do not match the pinned request')
        identity = seal(dict(schema=JOB_IDENTITY_SCHEMA, token=token, pid=os.getpid(),
                             process_start=process_start(os.getpid()),
                             request_sha256=request_sha256, command_identity_sha256=command_identity_sha256,
                             effective_command=list(command), effective_command_sha256=fingerprint(list(command))),
                        'identity_sha256')
        if not identity['process_start']:
            raise RuntimeError('cannot identify supervisor process start time')
        atomic_json_write(directory/'identity.json', identity, allow_nan=False)
        started = time.monotonic()
        stop = threading.Event()
        with (directory/'job.log').open('wb') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            def heartbeat():
                while not stop.is_set():
                    atomic_json_write(directory/'heartbeat.json', seal(dict(
                        schema='matrix.keymaker.job_heartbeat.v1', identity_sha256=identity['identity_sha256'],
                        token=token, epoch=time.time(), child_pid=child.pid), 'heartbeat_sha256'), allow_nan=False)
                    stop.wait(heartbeat_seconds)
            thread = threading.Thread(target=heartbeat, daemon=True)
            thread.start()
            def terminate(signum, _frame):
                # Propagate supervisor termination to its own child group only.
                if child.poll() is None:
                    os.killpg(child.pid, signum)
            old = {sig: signal.signal(sig, terminate) for sig in (signal.SIGTERM, signal.SIGINT)}
            try:
                code = child.wait()
                if code < 0:
                    code = 128-code  # Conventional positive status for termination by signal.
            finally:
                stop.set(); thread.join(timeout=5)
                for sig, handler in old.items():
                    signal.signal(sig, handler)
        result_file = directory/'result.json'
        completion = seal(dict(schema=JOB_COMPLETION_SCHEMA, token=token,
                               identity_sha256=identity['identity_sha256'], exit_code=code,
                               elapsed_seconds=time.monotonic()-started,
                               result_sha256=artifact_digest(result_file) if result_file.is_file() else None),
                          'completion_sha256')
        atomic_json_write(directory/'completion.json', completion, allow_nan=False)
        # Keep the old marker for older read-only monitors, never as authority
        # for the new identity-aware recovery path.
        (directory/'exitcode').write_text(str(code)+'\n')
        return code


def inspect_job(directory, *, token, request_sha256, command_identity_sha256, stale_seconds=15.):
    directory = Path(directory)
    digest(request_sha256); digest(command_identity_sha256); text(token, 'job token')
    identity = verify(read_json(directory/'identity.json'), 'identity_sha256', JOB_IDENTITY_SCHEMA)
    if (identity['token'] != token or identity['request_sha256'] != request_sha256
            or identity['command_identity_sha256'] != command_identity_sha256
            or fingerprint(identity['effective_command']) != identity['effective_command_sha256']):
        raise ValueError('persistent job identity mismatch')
    result = dict(schema='matrix.keymaker.persistent_job_status.v1', identity=identity,
                  status='unknown', heartbeat_age_seconds=None, completion=None)
    if (directory/'completion.json').is_file():
        complete = verify(read_json(directory/'completion.json'), 'completion_sha256', JOB_COMPLETION_SCHEMA)
        if complete['token'] != token or complete['identity_sha256'] != identity['identity_sha256']:
            raise ValueError('completion belongs to another attempt')
        if complete['exit_code'] == 0:
            if not (directory/'result.json').is_file() or artifact_digest(directory/'result.json') != complete['result_sha256']:
                raise ValueError('successful job result is absent or changed')
        result.update(status='finished' if complete['exit_code'] == 0 else 'failed', completion=complete)
        return result
    if process_start(identity['pid']) != identity['process_start']:
        return result  # Missing process or reused PID cannot represent this attempt.
    with (directory/'supervisor.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            return result  # An unrelated live PID without this job's lock is insufficient.
    if (directory/'heartbeat.json').is_file():
        heartbeat = verify(read_json(directory/'heartbeat.json'), 'heartbeat_sha256', 'matrix.keymaker.job_heartbeat.v1')
        if heartbeat['token'] != token or heartbeat['identity_sha256'] != identity['identity_sha256']:
            raise ValueError('heartbeat belongs to another attempt')
        age = time.time()-heartbeat['epoch']
        result.update(status='running' if 0 <= age <= stale_seconds else 'stale', heartbeat_age_seconds=age)
    return result


def main(argv=None):
    import json
    import sys
    arguments = list(sys.argv[1:] if argv is None else argv)
    command = []
    if '--' in arguments:
        separator = arguments.index('--')
        arguments, command = arguments[:separator], arguments[separator+1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run', 'inspect'))
    parser.add_argument('--directory', required=True)
    parser.add_argument('--token', required=True)
    parser.add_argument('--request-sha256', required=True)
    parser.add_argument('--command-identity-sha256', required=True)
    args = parser.parse_args(arguments)
    parameters = dict(token=args.token, request_sha256=args.request_sha256,
                      command_identity_sha256=args.command_identity_sha256)
    if args.action == 'inspect':
        print(json.dumps(inspect_job(args.directory, **parameters), sort_keys=True))
        return 0
    return supervise(args.directory, command=command, **parameters)


if __name__ == '__main__':
    raise SystemExit(main())
