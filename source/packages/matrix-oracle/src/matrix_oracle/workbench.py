from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .guidance import missing_sections_guidance
from .project import load_oracle_project_state
from .workflows import WindowSpec, window_spec


@dataclass(frozen=True)
class WorkbenchTable:
    title: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class WorkbenchGuiState:
    key: str
    xyzin: Path
    exists: bool
    title: str
    status: str
    message: str
    sections: WorkbenchTable
    actions: WorkbenchTable
    capabilities: WorkbenchTable
    exports: WorkbenchTable


def load_workbench_gui_state(path: Path | str, key: str) -> WorkbenchGuiState:
    target = Path(path)
    project = load_oracle_project_state(target)
    spec = window_spec(key)
    workflow_status = ""
    workflow_message = ""
    try:
        workflow = project.workflow(key)
        workflow_status = workflow.status.value
        workflow_message = workflow.message
    except KeyError:
        workflow_status = "unknown"
        workflow_message = "workflow is not registered in project state"
    present = set(project.section_names)
    return WorkbenchGuiState(
        key=key,
        xyzin=target,
        exists=project.exists,
        title=spec.title,
        status=workflow_status,
        message=workflow_message,
        sections=_section_table(spec, present),
        actions=_action_table(spec, present),
        capabilities=_single_column_table("Capabilities", spec.capabilities),
        exports=_exports_table(spec),
    )


def workbench_gui_state_lines(state: WorkbenchGuiState) -> list[str]:
    return [
        f"xyzin: {state.xyzin}",
        f"exists: {int(state.exists)}",
        f"window: {state.title}",
        f"status: {state.status}",
        f"message: {state.message}",
    ]


def _section_table(spec: WindowSpec, present: set[str]) -> WorkbenchTable:
    rows: list[tuple[str, str, str]] = []
    for section in spec.required_sections:
        rows.append(("required", section, "yes" if section in present else "no"))
    for section in spec.produced_sections:
        rows.append(("produced", section, "yes" if section in present else "no"))
    return WorkbenchTable("Sections", ("Role", "Section", "Present"), tuple(rows))


def _action_table(spec: WindowSpec, present: set[str]) -> WorkbenchTable:
    rows: list[tuple[str, str, str, str, str, str]] = []
    for action in spec.actions:
        missing = tuple(section for section in action.required_sections if section not in present)
        rows.append(
            (
                action.key,
                action.label,
                action.command,
                "yes" if not missing else "no",
                ", ".join(missing) if missing else "none",
                " | ".join(missing_sections_guidance(missing)) if missing else "none",
            )
        )
    return WorkbenchTable(
        "Actions",
        ("Key", "Action", "Command", "Ready", "Missing", "Suggestion"),
        tuple(rows),
    )


def _exports_table(spec: WindowSpec) -> WorkbenchTable:
    rows: list[tuple[str, str]] = []
    rows.extend(("publication", item) for item in spec.publication_exports)
    rows.extend(("viewer", item) for item in spec.external_viewers)
    return WorkbenchTable("Exports / Viewers", ("Kind", "Target"), tuple(rows))


def _single_column_table(title: str, values: tuple[str, ...]) -> WorkbenchTable:
    return WorkbenchTable(title, (title.rstrip("s"),), tuple((value,) for value in values))
