"""Conservative workflow resource estimates for Keymaker previews."""
from __future__ import annotations
from typing import Mapping, Sequence

def estimate_workflow_resources(steps: Sequence[Mapping[str, object]]) -> dict[str, object]:
    processors = max((int(step.get("processors", 1)) for step in steps), default=1)
    memory = sum(float(step.get("memory_gb", 0.0)) for step in steps)
    walltime = sum(int(step.get("walltime_minutes", 0)) for step in steps)
    return {"schema": "matrix.keymaker.resource_estimate.v1", "processors_peak": processors, "memory_gb_total": memory, "walltime_minutes_total": walltime, "conservative": True}
