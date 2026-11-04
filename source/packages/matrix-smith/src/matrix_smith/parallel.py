"""Shared bounded parallelism policy for SMITH local kernels."""

from __future__ import annotations

import os


MAX_LOCAL_WORKERS = 8


def resolve_local_workers(requested: int, task_count: int) -> int:
    """Resolve a bounded worker count without nested oversubscription."""

    workers = int(requested)
    tasks = int(task_count)
    if workers <= 0:
        raise ValueError("parallel_workers must be positive")
    if tasks <= 0:
        return 1
    environment_limit = os.environ.get("MATRIX_SMITH_MAX_WORKERS", "")
    limit = MAX_LOCAL_WORKERS if not environment_limit else int(environment_limit)
    if limit <= 0:
        raise ValueError("MATRIX_SMITH_MAX_WORKERS must be positive")
    return min(workers, tasks, limit)
