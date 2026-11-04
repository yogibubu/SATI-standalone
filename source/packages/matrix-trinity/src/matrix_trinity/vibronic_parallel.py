"""Lightweight worker policy shared by compiled TI and NumPy TD kernels."""
from concurrent.futures import ThreadPoolExecutor
import os
import numpy as np


def worker_count(workers):
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise ValueError('workers must be a nonnegative integer (0 selects at most four CPUs)')
    return max(1, min(workers or 4, os.cpu_count() or 1))


def parallel_rows(function, count, workers, *arguments, minimum_rows=256):
    """Bounded task count, no nested pools, same row order at every worker count."""
    count_workers = min(workers, max(1, count // minimum_rows))
    function(0, min(1, count), *arguments)
    bounds = np.linspace(min(1, count), count, count_workers + 1, dtype=int)
    if count_workers == 1:
        function(int(bounds[0]), count, *arguments)
    else:
        with ThreadPoolExecutor(max_workers=count_workers) as pool:
            jobs = [pool.submit(function, int(a), int(b), *arguments)
                    for a, b in zip(bounds[:-1], bounds[1:], strict=True) if b > a]
            for job in jobs:job.result()
    return count_workers
