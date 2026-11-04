"""Scientific dashboard contract consumed by LOCK Studio and command-line reports."""

from __future__ import annotations

from typing import Any, Mapping

from .staged_project import LOCK_STAGED_PROJECT_SCHEMA, promotion_assessment


LOCK_STUDIO_DASHBOARD_SCHEMA = "matrix.lock.studio_dashboard.v1"


def build_lock_studio_dashboard(
    project: Mapping[str, Any],
    *,
    qm_parameter_graph: Mapping[str, Any] | None = None,
    model_selection: Mapping[str, Any] | None = None,
    resource_plan: Mapping[str, Any] | None = None,
    surface_qualification: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize scientific decisions without duplicating solver logic in the GUI."""

    if project.get("schema") != LOCK_STAGED_PROJECT_SCHEMA:
        raise ValueError("LOCK Studio requires a staged LOCK project")
    stage = str(project["current_stage"])
    target = {"explore": "refine", "refine": "validate", "validate": None}[stage]
    promotion = None if target is None else promotion_assessment(project, target_stage=target)
    graph = dict(qm_parameter_graph or {})
    parameters = [dict(item) for item in graph.get("parameters", [])]
    large_shifts = [item for item in parameters if item.get("large_qm_shift")]
    selection = dict(model_selection or {})
    resources = dict(resource_plan or {})
    surface = dict(surface_qualification or {})
    attention = []
    if promotion and not promotion["ready"]:
        attention.extend(
            {"kind": "promotion_gate", "message": name}
            for name in promotion["blockers"]
        )
    attention.extend({
        "kind": "large_qm_shift",
        "message": f"{item['path']}: {item['shift_over_qm_sigma']:.2f} sigma",
    } for item in large_shifts)
    if selection and selection.get("selected") is None:
        attention.append({
            "kind": "model_selection",
            "message": str(selection.get("reason", "no identifiable physical model")),
        })
    if resources and not resources.get("ready", True):
        attention.append({
            "kind": "resources",
            "message": "insufficient memory for " + ", ".join(resources.get("blocked_blocks", [])),
        })
    if surface and not surface.get("qualified", True):
        attention.append({
            "kind": "surface",
            "message": str(surface.get("reason", "effective surface is not qualified")),
        })
    return {
        "schema": LOCK_STUDIO_DASHBOARD_SCHEMA,
        "project": project.get("name", ""),
        "stage": stage,
        "stage_description": {
            "explore": "Identify active coordinates, symmetry, states and a defensible model.",
            "refine": "Converge the basis and refine identifiable parameters against observations.",
            "validate": "Test held-out lines, higher-J predictions and transferable observables.",
        }[stage],
        "next_stage": target,
        "promotion": promotion,
        "surface": surface,
        "model": {
            "selected": selection.get("selected"),
            "reason": selection.get("reason"),
        },
        "parameters": parameters,
        "identifiability": {
            "experimental": graph.get("experimental_identifiability"),
            "augmented": graph.get("augmented_identifiability"),
            "rank_gain_from_qm": graph.get("rank_gain_from_qm"),
        },
        "resources": {
            "workers": resources.get("selected_workers"),
            "schedule": resources.get("deterministic_schedule", []),
            "blocks": resources.get("blocks", []),
        },
        "attention": attention,
        "ready_for_next_stage": bool(promotion and promotion["ready"]),
    }


__all__ = ["LOCK_STUDIO_DASHBOARD_SCHEMA", "build_lock_studio_dashboard"]
