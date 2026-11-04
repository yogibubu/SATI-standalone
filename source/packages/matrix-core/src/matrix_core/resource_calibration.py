"""Empirical resource calibration from completed workflow timings."""
from __future__ import annotations
from statistics import median
from typing import Mapping, Sequence

def calibrate_walltime(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    values = [float(item["walltime_minutes"]) for item in records if float(item.get("walltime_minutes", 0)) > 0]
    return {"schema": "matrix.keymaker.resource_calibration.v1", "sample_count": len(values), "median_walltime_minutes": median(values) if values else None, "p90_walltime_minutes": sorted(values)[max(0, int(.9 * len(values)) - 1)] if values else None}
