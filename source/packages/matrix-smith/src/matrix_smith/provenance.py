"""Reproducibility and coordinate-quality metadata for SMITH artifacts."""
from __future__ import annotations
import hashlib
import platform
import subprocess
import sys
from pathlib import Path

def artifact_provenance(path: str | Path | None = None) -> dict[str, object]:
    commit = branch = None
    try:
        commit = subprocess.check_output(("git", "rev-parse", "HEAD"), text=True, stderr=subprocess.DEVNULL).strip()
        branch = subprocess.check_output(("git", "branch", "--show-current"), text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError): pass
    payload={"schema":"matrix.smith.provenance.v1", "python":sys.version.split()[0], "platform":platform.platform(), "git_commit":commit, "git_branch":branch}
    if path is not None:
        digest=hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).is_file() else None
        payload["input_sha256"]=digest
    return payload

def attach_provenance(report: dict[str, object], path: str | Path | None = None) -> dict[str, object]:
    result=dict(report); result.setdefault("provenance", artifact_provenance(path)); return result


def coordinate_provenance(
    *,
    coordinate_ids: tuple[str, ...] | list[str],
    chart_states: tuple[object, ...] | list[object] = (),
    conditioning: dict[str, object] | None = None,
    selection: object | None = None,
    derivative_modes: dict[str, str] | None = None,
) -> dict[str, object]:
    """Build explicit provenance and uncertainty indicators for SONIC rows."""

    states = []
    for state in chart_states:
        states.append(
            {
                "identifier": str(getattr(state, "identifier", "")),
                "chart": str(getattr(state, "chart", "UNKNOWN")),
                "status": str(getattr(state, "status", "UNKNOWN")),
                "distance_to_boundary": float(
                    getattr(state, "distance_to_boundary", float("nan"))
                ),
                "recommended_chart": str(getattr(state, "recommended_chart", "UNKNOWN")),
                "reason": str(getattr(state, "reason", "")),
            }
        )
    payload: dict[str, object] = {
        "schema": "matrix.smith.coordinate_provenance.v1",
        "coordinates": tuple(str(identifier) for identifier in coordinate_ids),
        "chart_states": tuple(states),
        "conditioning": dict(conditioning or {}),
        "derivative_modes": dict(derivative_modes or {}),
        "uncertainty_policy": "FAIL_CLOSED_NEAR_SINGULAR_OR_OUT_OF_CHART",
    }
    if selection is not None:
        payload["selection_policy"] = str(getattr(selection, "policy", "UNKNOWN"))
        payload["selection_role"] = str(getattr(selection, "role", "UNKNOWN"))
        payload["selected_coordinates"] = tuple(getattr(selection, "selected", ()))
    return payload
