"""Deterministic performance and parity audits for LOCK operator kernels."""

from __future__ import annotations

import math
from statistics import median
from time import perf_counter_ns
from typing import Any

import numpy as np

from .operator_engine import CompiledOperatorModel, apply_operator, assemble_operator


LOCK_PROFILE_SCHEMA = "matrix.lock.performance_profile.v1"


def _timed(action, repeats: int) -> tuple[float, Any]:
    samples = []
    result = None
    for _ in range(repeats):
        start = perf_counter_ns()
        result = action()
        samples.append((perf_counter_ns() - start) * 1.0e-9)
    return float(median(samples)), result


def profile_operator_model(
    model: CompiledOperatorModel,
    *,
    repeats: int = 5,
    vector_count: int = 4,
    seed: int = 1729,
    dense_max_dimension: int = 2048,
) -> dict[str, Any]:
    """Measure real LOCK kernels and certify every accelerated result."""

    if repeats < 1 or vector_count < 1 or dense_max_dimension < 1:
        raise ValueError("profile sizes must be positive")
    rng = np.random.default_rng(int(seed))
    vectors = rng.normal(size=(model.dimension, vector_count)) + 1j * rng.normal(
        size=(model.dimension, vector_count)
    )
    vectors /= np.linalg.norm(vectors, axis=0, keepdims=True)
    python_seconds, python_result = _timed(
        lambda: apply_operator(model, vectors, backend="python"), repeats
    )
    kernels: dict[str, Any] = {
        "python_factorized": {"median_seconds": python_seconds, "available": True}
    }
    try:
        compiled_seconds, compiled_result = _timed(
            lambda: apply_operator(model, vectors, backend="compiled"), repeats
        )
    except RuntimeError:
        compiled_result = None
        kernels["compiled_factorized"] = {"available": False}
    else:
        parity = float(np.max(np.abs(compiled_result - python_result), initial=0.0))
        scale = max(float(np.max(np.abs(python_result), initial=0.0)), 1.0)
        kernels["compiled_factorized"] = {
            "available": True,
            "median_seconds": compiled_seconds,
            "speedup": python_seconds / compiled_seconds if compiled_seconds > 0.0 else None,
            "maximum_absolute_error": parity,
            "passed": parity <= 5.0e-13 * scale,
        }
    dense_bytes = model.dimension * model.dimension * np.dtype(complex).itemsize
    if model.dimension <= dense_max_dimension:
        dense_seconds, dense = _timed(lambda: assemble_operator(model), repeats)
        dense_result = dense @ vectors
        error = float(np.max(np.abs(dense_result - python_result), initial=0.0))
        scale = max(float(np.max(np.abs(dense_result), initial=0.0)), 1.0)
        kernels["dense_assembly_and_matvec"] = {
            "available": True,
            "median_seconds": dense_seconds,
            "maximum_absolute_error": error,
            "passed": error <= 5.0e-13 * scale,
        }
    else:
        kernels["dense_assembly_and_matvec"] = {
            "available": False,
            "reason": "dimension exceeds dense profiling limit",
        }
    compiled = kernels["compiled_factorized"]
    recommendation = "python-factorized"
    if compiled.get("available") and compiled.get("passed"):
        recommendation = "compiled-factorized"
    if not math.isfinite(python_seconds):
        raise RuntimeError("LOCK profiler produced a nonfinite timing")
    return {
        "schema": LOCK_PROFILE_SCHEMA,
        "dimension": model.dimension,
        "term_count": len(model.terms),
        "vector_count": vector_count,
        "repeats": repeats,
        "seed": int(seed),
        "dense_bytes": dense_bytes,
        "kernels": kernels,
        "recommended_backend": recommendation,
        "certified": all(
            row.get("passed", True) for row in kernels.values() if row.get("available")
        ),
    }


__all__ = ["LOCK_PROFILE_SCHEMA", "profile_operator_model"]
