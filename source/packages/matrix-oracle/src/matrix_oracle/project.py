from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from matrix_core import (
    is_section_header_line,
    read_basic_section,
    read_sectioned_lines,
    section_content,
    xyz_tail_start,
)
from matrix_chem import read_enriched_xyz, validate_enriched_molecule

from .guidance import missing_sections_message
from .workflows import ORACLE_GUI_WINDOWS, WorkflowStatus


@dataclass(frozen=True)
class SectionState:
    name: str
    present: bool
    line_count: int = 0
    schema: str | None = None


@dataclass(frozen=True)
class WorkflowState:
    key: str
    title: str
    status: WorkflowStatus
    message: str
    required_sections: tuple[str, ...] = ()
    produced_sections: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return self.status in {
            WorkflowStatus.READY,
            WorkflowStatus.WARNING,
            WorkflowStatus.COMPLETE,
        }


@dataclass(frozen=True)
class OracleProjectState:
    xyzin: Path
    exists: bool
    atom_count: int
    comment: str
    point_group: str
    sections: tuple[SectionState, ...]
    validation_status: str
    validation_messages: tuple[str, ...]
    workflows: tuple[WorkflowState, ...]

    @property
    def section_names(self) -> tuple[str, ...]:
        return tuple(section.name for section in self.sections if section.present)

    def section(self, name: str) -> SectionState:
        target = name.strip().upper().lstrip("#")
        for section in self.sections:
            if section.name == target:
                return section
        return SectionState(target, False)

    def workflow(self, key: str) -> WorkflowState:
        target = "link" if key == "babel" else key
        for workflow in self.workflows:
            if workflow.key == target:
                return workflow
        raise KeyError(f"unknown ORACLE GUI workflow: {key}")


def load_oracle_project_state(
    path: Path | str, *, require_fragments: bool = False
) -> OracleProjectState:
    target = Path(path)
    if not target.exists():
        return OracleProjectState(
            xyzin=target,
            exists=False,
            atom_count=0,
            comment="",
            point_group="",
            sections=(),
            validation_status="MISSING",
            validation_messages=(f"Missing file: {target}",),
            workflows=_workflow_states(
                present_sections=(),
                exists=False,
                validation_status="MISSING",
            ),
        )

    lines = read_sectioned_lines(target)
    sections = _section_states(lines)
    section_names = tuple(section.name for section in sections if section.present)
    geometry_message = None
    try:
        geometry = read_enriched_xyz(target)
        atom_count = geometry.natoms
        comment = geometry.comment
    except Exception as exc:  # GUI state must report broken files instead of crashing.
        atom_count = 0
        comment = ""
        geometry_message = f"ERROR INVALID_XYZ {exc}"

    try:
        point_group = read_basic_section(target).point_group
    except Exception:
        point_group = ""

    try:
        validation = validate_enriched_molecule(target, require_fragments=require_fragments)
        validation_status = validation.status
        validation_messages = tuple(
            f"{message.level} {message.code} {message.text}" for message in validation.messages
        )
    except Exception as exc:
        validation_status = "FAIL"
        validation_messages = (f"ERROR VALIDATION_EXCEPTION {exc}",)

    if geometry_message is not None:
        validation_status = "FAIL"
        validation_messages = (geometry_message, *validation_messages)

    return OracleProjectState(
        xyzin=target,
        exists=True,
        atom_count=atom_count,
        comment=comment,
        point_group=point_group,
        sections=sections,
        validation_status=validation_status,
        validation_messages=validation_messages,
        workflows=_workflow_states(
            present_sections=section_names,
            exists=True,
            validation_status=validation_status,
        ),
    )


def project_state_lines(state: OracleProjectState) -> list[str]:
    lines = [
        f"xyzin: {state.xyzin}",
        f"exists: {int(state.exists)}",
        f"atoms: {state.atom_count}",
        f"point_group: {state.point_group or 'unknown'}",
        f"validation: {state.validation_status}",
        "sections: " + (", ".join(state.section_names) if state.section_names else "none"),
    ]
    lines.extend(
        f"workflow {workflow.key}: {workflow.status.value} - {workflow.message}"
        for workflow in state.workflows
    )
    return lines


def _section_states(lines: list[str]) -> tuple[SectionState, ...]:
    names = []
    for raw in lines[xyz_tail_start(lines) :]:
        if is_section_header_line(raw):
            names.append(raw.strip()[1:].strip().upper())
    return tuple(_section_state(lines, name) for name in names)


def _section_state(lines: list[str], name: str) -> SectionState:
    content = section_content(lines, name)
    schema = None
    for raw in content:
        text = raw.strip()
        if text:
            if text.upper().startswith("SCHEMA "):
                schema = text.split(None, 1)[1].strip()
            break
    return SectionState(name=name, present=True, line_count=len(content), schema=schema)


def _workflow_states(
    *,
    present_sections: tuple[str, ...],
    exists: bool,
    validation_status: str,
) -> tuple[WorkflowState, ...]:
    present = {section.upper() for section in present_sections}
    return tuple(
        WorkflowState(
            key=spec.key,
            title=spec.title,
            status=_workflow_status(
                spec.key,
                present=present,
                exists=exists,
                validation_status=validation_status,
                required=spec.required_sections,
                produced=spec.produced_sections,
            ),
            message=_workflow_message(
                spec.key,
                present=present,
                exists=exists,
                validation_status=validation_status,
                required=spec.required_sections,
                produced=spec.produced_sections,
            ),
            required_sections=spec.required_sections,
            produced_sections=spec.produced_sections,
        )
        for spec in ORACLE_GUI_WINDOWS
    )


def _workflow_status(
    key: str,
    *,
    present: set[str],
    exists: bool,
    validation_status: str,
    required: tuple[str, ...],
    produced: tuple[str, ...],
) -> WorkflowStatus:
    if key == "dashboard":
        return WorkflowStatus.READY if exists else WorkflowStatus.MISSING
    if key in {"link", "babel"}:
        if {"SYMMETRY", "TOPOLOGY", "SYNTHONS"}.issubset(present):
            return WorkflowStatus.COMPLETE
        return WorkflowStatus.READY if exists else WorkflowStatus.MISSING
    if key == "avogadro":
        return WorkflowStatus.READY if exists else WorkflowStatus.MISSING
    if produced and set(produced).intersection(present):
        return WorkflowStatus.COMPLETE
    if required and set(required).issubset(present):
        if validation_status == "FAIL" and key in {"gicforge", "gf", "sefit"}:
            return WorkflowStatus.WARNING
        return WorkflowStatus.READY
    if not required and exists:
        return WorkflowStatus.READY
    return WorkflowStatus.MISSING


def _workflow_message(
    key: str,
    *,
    present: set[str],
    exists: bool,
    validation_status: str,
    required: tuple[str, ...],
    produced: tuple[str, ...],
) -> str:
    if not exists:
        return "open or create a MATRIX xyzin project"
    if key in {"link", "babel"}:
        missing = sorted({"SYMMETRY", "TOPOLOGY", "SYNTHONS"} - present)
        return "preprocessing complete" if not missing else missing_sections_message(missing)
    if produced and set(produced).intersection(present):
        produced_now = ", ".join(section for section in produced if section in present)
        return f"available sections: {produced_now}"
    missing = [section for section in required if section not in present]
    if missing:
        return missing_sections_message(missing)
    if validation_status == "FAIL" and key in {"gicforge", "gf", "sefit"}:
        return "inputs present, but validation has failures"
    return "ready"
