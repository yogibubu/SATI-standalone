"""One deterministic CPU/memory budget for heterogeneous LOCK tasks."""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from threading import RLock, local
import os
import numpy as np
from threadpoolctl import threadpool_limits
from .resource_planner import _digest


@dataclass(frozen=True)
class LOCKTask:
    key: str
    workspace_bytes: int
    retained_bytes: int = 0
    relative_cost: float = 1.0

    def __post_init__(self):
        if (not self.key or any(isinstance(v,bool) or not isinstance(v,(int,np.integer)) or v < 0
                for v in (self.workspace_bytes,self.retained_bytes))
            or not np.isfinite(self.relative_cost) or self.relative_cost <= 0):
            raise ValueError("invalid LOCK task resource requirement")


class LOCKResourceExecutor:
    """Thread executor with conservative declared memory and bounded reuse.

    One outer executor owns the numerical-library thread limit. Functions
    submitted here must use one inner worker. Task memory is an estimate,
    not an operating-system allocation limit. map() returns input order;
    execution waves use largest-cost-first and deterministic tie breaking.
    Independent fit functions must own their own mutable optimizer/producer.
    """
    def __init__(self, *, workers=1, memory_budget_bytes=2*1024**3,
                 shared_resident_bytes=0, cache_budget_bytes=0):
        for v in (workers,memory_budget_bytes,shared_resident_bytes,cache_budget_bytes):
            if isinstance(v,bool) or not isinstance(v,(int,np.integer)):
                raise ValueError("integer resource budgets required")
        if (workers < 1 or memory_budget_bytes < 1 or min(shared_resident_bytes,cache_budget_bytes) < 0
            or shared_resident_bytes+cache_budget_bytes >= memory_budget_bytes):
            raise ValueError("resource budgets exceed available memory")
        self.workers=min(int(workers), max(1,os.cpu_count() or 1))
        self.memory_budget_bytes=int(memory_budget_bytes)
        self.shared_resident_bytes=int(shared_resident_bytes)
        self.cache_budget_bytes=int(cache_budget_bytes)
        self._cache=OrderedDict(); self._cache_bytes=0; self._lock=RLock()
        self._phase_lock=RLock();self._thread_state=local()
        self.history=[]; self._pool=None; self._limits=None

    def __enter__(self):
        if self._pool is not None:
            raise RuntimeError("executor already active")
        self._limits=threadpool_limits(limits=1)
        self._pool=ThreadPoolExecutor(max_workers=self.workers)
        return self

    def __exit__(self, *exc):
        try:
            self._pool.shutdown(wait=True,cancel_futures=True)
        finally:
            self._pool=None
            self._limits.restore_original_limits()

    def map(self, function, tasks):
        if getattr(self._thread_state,"running",False):
            raise RuntimeError("nested task execution must use one inner worker")
        with self._phase_lock:
            return self._map(function,tasks)

    def _map(self,function,tasks):
        if self._pool is None:
            raise RuntimeError("use LOCKResourceExecutor as a context manager")
        tasks=list(tasks)
        if len({t.key for t in tasks}) != len(tasks):
            raise ValueError("task keys must be unique within an execution phase")
        retained=sum(t.retained_bytes for t in tasks)
        available=self.memory_budget_bytes-self.shared_resident_bytes-self.cache_budget_bytes-retained
        if any(t.workspace_bytes > available for t in tasks):
            raise MemoryError("task workspace plus all retained outputs exceeds the declared budget")
        pending=sorted(tasks,key=lambda t:(-t.relative_cost,t.key)); results={}
        while pending:
            wave=[]; used=0
            for task in pending:
                if len(wave)<self.workers and used+task.workspace_bytes<=available:
                    wave.append(task);used+=task.workspace_bytes
            if not wave:
                raise MemoryError("no admissible execution wave")
            pending=[t for t in pending if t not in wave]
            self.history.append(dict(keys=[t.key for t in wave],workspace_bytes=used,
                retained_output_bytes=retained,blas_threads=1,inner_workers=1))
            def run(key):
                self._thread_state.running=True
                try:return function(key)
                finally:self._thread_state.running=False
            futures=[(t,self._pool.submit(run,t.key)) for t in wave]
            try:
                for task,future in futures:
                    results[task.key]=future.result()
            except BaseException:
                for _,future in futures:
                    future.cancel()
                wait([future for _,future in futures])
                raise
        return [results[t.key] for t in tasks]

    def cached_arrays(self, key, contract, factory):
        """Reuse only immutable ndarray/tuple trees under an exact contract."""
        identity=(str(key),_digest(contract))
        def size_and_freeze(value):
            if isinstance(value,np.ndarray):
                if not np.all(np.isfinite(value)):
                    raise ValueError("nonfinite reusable numerical array")
                value.setflags(write=False)
                return value.nbytes
            if isinstance(value,tuple):
                return sum(size_and_freeze(x) for x in value)
            if value is None or isinstance(value,(str,bool,int,float,complex,np.generic)):
                if isinstance(value,(float,complex,np.number)) and not np.isfinite(value):
                    raise ValueError("nonfinite scalar in reusable factorization")
                return 64 + (len(value) if isinstance(value,str) else 0)
            raise TypeError("reusable factors must be arrays, immutable scalars or tuples")
        with self._lock:
            if identity in self._cache:
                value,size=self._cache.pop(identity);self._cache[identity]=(value,size)
                return value
            value=factory();size=size_and_freeze(value)
            if size > self.cache_budget_bytes:
                return value
            while self._cache and self._cache_bytes+size>self.cache_budget_bytes:
                _,(_,oldsize)=self._cache.popitem(last=False);self._cache_bytes-=oldsize
            self._cache[identity]=(value,size);self._cache_bytes+=size
            return value

    def report(self):
        return dict(schema="matrix.lock.execution_budget.v1",workers=self.workers,
            memory_budget_bytes=self.memory_budget_bytes,shared_resident_bytes=self.shared_resident_bytes,
            cache_budget_bytes=self.cache_budget_bytes,cache_resident_bytes=self._cache_bytes,
            waves=list(self.history),memory_policy="declared conservative estimates, not an OS limit")


def map_lock_refits(requests, *, workspace_bytes, retained_bytes=0, workers=1,
                    memory_budget_bytes=2*1024**3):
    """Parallel independent serialized structural fits (isotopes or resamples).

    Each request constructs its own producer, caches and accepted state. Nested
    LOCK parallelism is disabled. Output order equals request order.
    """
    from copy import deepcopy
    from .structure_spectrum import fit_structure_pes_spectra
    payloads=[deepcopy(r) for r in requests]
    tasks=[LOCKTask(str(i),int(workspace_bytes),int(retained_bytes)) for i in range(len(payloads))]
    def run(key):
        request=payloads[int(key)]
        request.setdefault("options",{}).pop("execution_budget",None)
        for dataset in request["datasets"]:
            dataset["request"].setdefault("options",{}).update(workers=1,block_workers=1,jacobian_workers=1)
        return fit_structure_pes_spectra(request)
    with LOCKResourceExecutor(workers=workers,memory_budget_bytes=memory_budget_bytes) as executor:
        results=executor.map(run,tasks)
        return dict(results=results,execution_budget=executor.report())
