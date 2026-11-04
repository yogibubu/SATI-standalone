"""Batch helpers for resumable, deterministic ORACLE manifests."""
from __future__ import annotations
from pathlib import Path
from typing import Iterable
from .api import OracleAnalysisRequest
import signal

from matrix_core import atomic_json_write

def run_batch_safe(requests: Iterable[OracleAnalysisRequest], runner, *, retries: int = 0) -> dict[str, object]:
    """Run requests with structured per-item failures and bounded retries."""
    records = []
    for request in requests:
        error = None
        result = None
        for _ in range(max(0, retries) + 1):
            try:
                result = runner(request)
                error = None
                break
            except Exception as exc:  # report boundary intentionally catches job failures
                error = {"type": type(exc).__name__, "message": str(exc)}
        records.append({"output": str(request.output), "result": result.to_dict() if result else None,
                        "error": error, "status": "FAIL" if error else "PASS"})
    return {"schema": "matrix.oracle.batch_result.v1", "count": len(records), "results": records,
            "failed": sum(record["status"] == "FAIL" for record in records)}

def pending_requests(requests: Iterable[OracleAnalysisRequest], *, resume: bool = True) -> tuple[OracleAnalysisRequest, ...]:
    jobs = tuple(requests)
    outputs = [Path(job.output).expanduser().resolve() for job in jobs]
    if len(outputs) != len(set(outputs)):
        raise ValueError("parallel ORACLE requests must have distinct output files")
    if not resume: return jobs
    def complete(job: OracleAnalysisRequest) -> bool:
        output = Path(job.output).expanduser()
        source = Path(job.source).expanduser()
        try:
            return output.is_file() and output.stat().st_size > 0 and output.stat().st_mtime >= source.stat().st_mtime
        except OSError:
            return False
    return tuple(job for job in jobs if not complete(job))

def write_batch_manifest(payload: dict[str, object], path: str | Path) -> Path:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(target, payload)
    return target

def checkpoint_batch(completed: Iterable[str], path: str | Path) -> Path:
    return write_batch_manifest({"schema": "matrix.oracle.batch_checkpoint.v1", "completed": list(completed)}, path)

class BatchInterrupt:
    def __init__(self): self.interrupted = False
    def __enter__(self):
        self._old = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        for sig in self._old: signal.signal(sig, lambda *_: setattr(self, "interrupted", True))
        return self
    def __exit__(self, *_):
        for sig, handler in self._old.items(): signal.signal(sig, handler)
