"""Execution-state diagnostics for Gaussian-compatible SONIC calculations."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from .parsers import check_gaussian_readallgic_log, summarize_gaussian_log


_FATAL_RE = re.compile(
    r"Error termination|Some coordinates are missing|Error in internal coordinate system",
    flags=re.IGNORECASE,
)


def diagnose_gdv_sonic_log(
    path: Path,
    *,
    expected_rank: int | None = None,
    expected_point_group: str | None = None,
    require_frequency: bool = False,
    require_no_imaginary: bool = False,
) -> dict[str, Any]:
    """Return a JSON-serializable PASS/RUNNING/FAILED diagnostic."""

    target = Path(path).expanduser().resolve()
    if not target.is_file():
        return {
            "schema": "matrix.gdv.sonic-log-check.v1",
            "path": str(target),
            "status": "MISSING",
        }
    text = target.read_text(encoding="utf-8", errors="replace")
    check = check_gaussian_readallgic_log(
        target,
        expected_point_group=expected_point_group,
        expected_rank=expected_rank,
        require_frequency=require_frequency,
        require_no_imaginary=require_no_imaginary,
    )
    summary = summarize_gaussian_log(target)
    fatal_markers = tuple(
        dict.fromkeys(match.group(0) for match in _FATAL_RE.finditer(text))
    )
    if check.normal_termination_count:
        status = "PASS" if check.ok and not fatal_markers else "FAILED"
    else:
        status = "FAILED" if fatal_markers else "RUNNING"
    frequencies = tuple(float(value) for value in summary.frequencies_cm)
    return {
        "schema": "matrix.gdv.sonic-log-check.v1",
        "path": str(target),
        "status": status,
        "normal_termination_count": check.normal_termination_count,
        "route_has_readallgic": check.route_has_readallgic,
        "point_groups": list(check.point_groups),
        "ranks": [list(pair) for pair in check.ranks],
        "last_rank": list(check.ranks[-1]) if check.ranks else None,
        "optimization_completed": check.optimization_completed,
        "stationary_point": check.stationary_point,
        "frequency_count": check.frequency_count,
        "frequency_range_cm-1": (
            [min(frequencies), max(frequencies)] if frequencies else None
        ),
        "imaginary_frequency_count": check.imaginary_frequency_count,
        "n_imag": check.n_imag,
        "fatal_markers": list(fatal_markers),
        "errors": list(check.errors),
    }
