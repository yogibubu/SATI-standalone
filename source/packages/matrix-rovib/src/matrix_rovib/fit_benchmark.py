"""End-to-end native LAM fit timing, convergence and restart qualification."""
from __future__ import annotations

from .lam_checkpoint import copy_workspace_request
from pathlib import Path
from tempfile import TemporaryDirectory
import platform
import resource
import statistics
import time
import tracemalloc
import numpy as np
from threadpoolctl import threadpool_limits
from .lam_fit import fit_lam_assigned_spectrum


def benchmark_lam_fit(surface, request, *, worker_counts=(1, 2), repeats=3,
                      frequency_tolerance_MHz=1e-7, _fit_function=None):
    """Measure full fits from identical starts and accepted-workspace replays.

    Timing includes Python allocation tracing and checkpoint I/O. Traced peak
    memory excludes uninstrumented native allocations. RSS is the process
    lifetime high-water mark, not a per-fit peak. No refit is called a speedup
    unless both runs are statistically converged and spectrally equivalent.
    Existing user checkpoint/output paths are never written by this benchmark.
    """
    if (isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1
        or not worker_counts or len(set(worker_counts)) != len(worker_counts)
        or any(isinstance(n,bool) or not isinstance(n,int) or n < 1 for n in worker_counts)
        or not np.isfinite(frequency_tolerance_MHz) or frequency_tolerance_MHz <= 0):
        raise ValueError('invalid fit benchmark budgets')
    if tracemalloc.is_tracing():
        raise RuntimeError('benchmark requires ownership of the allocation tracer')
    fit_function=fit_lam_assigned_spectrum if _fit_function is None else _fit_function
    rows=[];reference=None
    def measure(local):
        tracemalloc.start()
        start=time.perf_counter()
        try:
            result=fit_function(surface,local).to_dict()
            seconds=time.perf_counter()-start
            _,peak=tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss=int(rss if platform.system()=='Darwin' else 1024*rss)
        return result,seconds,peak,rss
    with TemporaryDirectory(prefix='lock-fit-benchmark-') as folder, threadpool_limits(limits=1):
        for workers in worker_counts:
            for repeat in range(repeats):
                local=copy_workspace_request(request)
                options=local.setdefault('options',{})
                for key in ('resume_workspace','resume_checkpoint','optimizer_state','_workspace_resume_state',
                            'checkpoint_path','workspace_checkpoint_path','linearization_path','evaluation_log_path',
                            'iteration_trace_path'):
                    options.pop(key,None)
                # One outer level owns the CPU budget; inner BLAS uses one thread.
                options.update(workers=workers,jacobian_workers=workers,block_workers=1)
                checkpoint=Path(folder)/f'{workers}-{repeat}.npz'
                options['workspace_checkpoint_path']=str(checkpoint)
                for phase in ('fit','accepted-restart'):
                    if phase=='accepted-restart':
                        options['resume_workspace']=str(checkpoint)
                    payload,seconds,peak,rss=measure(local)
                    frequencies=np.array([r['calculated_MHz'] for r in payload['comparisons']])
                    if reference is None:reference=frequencies.copy()
                    maximum=float(np.max(np.abs(frequencies-reference),initial=0.))
                    converged=bool(payload['success'] and payload['convergence']['converged'])
                    equivalent=maximum <= frequency_tolerance_MHz
                    rows.append(dict(workers=workers,repeat=repeat,phase=phase,seconds=seconds,
                        traced_peak_bytes=peak,process_lifetime_peak_rss_bytes=rss,
                        converged=converged,spectrally_equivalent=equivalent,
                        maximum_frequency_difference_MHz=maximum,
                        hamiltonian_solves=payload['metrics']['hamiltonian_solves'],
                        hamiltonian_evaluations=payload['metrics']['hamiltonian_evaluations'],
                        jacobian_evaluations=payload['metrics']['jacobian_evaluations'],
                        rms_residual_MHz=payload['metrics']['rms_residual_MHz'],
                        chi_square=payload['metrics']['chi_square'],
                        workspace_reused=payload['workspace_restart']['reused']))
    summaries=[]
    baseline=statistics.median(r['seconds'] for r in rows if r['workers']==worker_counts[0] and r['phase']=='fit')
    baseline_qualified=all(r['converged'] and r['spectrally_equivalent'] for r in rows if r['workers']==worker_counts[0] and r['phase']=='fit')
    for workers in worker_counts:
        for phase in ('fit','accepted-restart'):
            selected=[r for r in rows if r['workers']==workers and r['phase']==phase]
            eligible=all(r['converged'] and r['spectrally_equivalent'] for r in selected)
            elapsed=statistics.median(r['seconds'] for r in selected)
            summaries.append(dict(workers=workers,phase=phase,qualified=eligible,
                median_seconds=elapsed,maximum_traced_peak_bytes=max(r['traced_peak_bytes'] for r in selected),
                speedup_vs_first_complete_fit=baseline/elapsed if eligible and baseline_qualified and phase=='fit' else None))
    return dict(schema='matrix.lock.full_fit_benchmark.v1',runs=rows,summaries=summaries,
        reference_frequencies_MHz=reference.tolist(),
        qualified=all(r['converged'] and r['spectrally_equivalent'] for r in rows),
        frequency_tolerance_MHz=frequency_tolerance_MHz,blas_threads=1,
        timing_scope='complete fit including allocation tracing and checkpoint I/O; warm process caches',
        memory_scope='traced allocations plus process-lifetime RSS; no per-fit native-memory bound',
        restart_scope='accepted-workspace replay, reported separately from fitting',
        environment=dict(python=platform.python_version(),platform=platform.platform()))
