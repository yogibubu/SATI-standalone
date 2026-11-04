"""Frozen cross-program numerical benchmarks for LOCK."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


LOCK_BENCHMARK_SCHEMA = "matrix.lock.benchmark_suite.v1"


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    reference_program: str
    labels: tuple[str, ...]
    reference_MHz: tuple[float, ...]
    calculated_MHz: tuple[float, ...]
    rms_tolerance_MHz: float
    maximum_tolerance_MHz: float
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        size = len(self.labels)
        if not self.name.strip() or not self.reference_program.strip() or size < 1:
            raise ValueError("benchmark cases require names, programs, and labelled values")
        if len(set(self.labels)) != size or len(self.reference_MHz) != size or len(self.calculated_MHz) != size:
            raise ValueError("benchmark labels and numerical arrays must have equal unique lengths")
        values = (*self.reference_MHz, *self.calculated_MHz)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("benchmark frequencies must be finite")
        if self.rms_tolerance_MHz < 0.0 or self.maximum_tolerance_MHz < 0.0:
            raise ValueError("benchmark tolerances must be nonnegative")


def run_benchmark_suite(cases: Sequence[BenchmarkCase]) -> dict[str, object]:
    """Evaluate frozen SPFIT/XIAM/ERHAM/BELGI or experimental comparisons."""

    ordered = tuple(cases)
    if not ordered or len({case.name for case in ordered}) != len(ordered):
        raise ValueError("benchmark suite requires uniquely named cases")
    rows = []
    for case in ordered:
        residual = np.asarray(case.calculated_MHz) - np.asarray(case.reference_MHz)
        rms = float(np.sqrt(np.mean(residual * residual)))
        maximum = float(np.max(np.abs(residual), initial=0.0))
        rows.append(
            {
                "name": case.name,
                "reference_program": case.reference_program,
                "line_count": len(case.labels),
                "rms_MHz": rms,
                "maximum_absolute_MHz": maximum,
                "rms_tolerance_MHz": case.rms_tolerance_MHz,
                "maximum_tolerance_MHz": case.maximum_tolerance_MHz,
                "passed": rms <= case.rms_tolerance_MHz
                and maximum <= case.maximum_tolerance_MHz,
                "worst_line": case.labels[int(np.argmax(np.abs(residual)))],
                "metadata": dict(case.metadata or {}),
            }
        )
    identity = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return {
        "schema": LOCK_BENCHMARK_SCHEMA,
        "passed": all(row["passed"] for row in rows),
        "case_count": len(rows),
        "suite_sha256": identity,
        "cases": rows,
    }


def write_benchmark_suite(path: Path | str, result: Mapping[str, object]) -> Path:
    if result.get("schema") != LOCK_BENCHMARK_SCHEMA:
        raise ValueError("unsupported LOCK benchmark result")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(dict(result), indent=2, sort_keys=True) + "\n")
    temporary.replace(target)
    return target


__all__ = [
    "BenchmarkCase",
    "LOCK_BENCHMARK_SCHEMA",
    "run_benchmark_suite",
    "write_benchmark_suite",
]
