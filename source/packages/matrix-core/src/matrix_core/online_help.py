from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .tool_contracts import (
    ToolContract,
    ToolReadiness,
    tool_contract,
    tool_contract_readiness,
    tool_contracts,
)


@dataclass(frozen=True)
class SectionCompletionHint:
    section: str
    window: str
    action: str
    command: str
    note: str = ""

    def line(self) -> str:
        text = f"#{self.section}: use {self.window} -> {self.action} ({self.command})"
        if self.note:
            text += f"; {self.note}"
        return text

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ToolHelp:
    contract: ToolContract
    manual_paths: tuple[str, ...] = ()
    quickstart: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    see_also: tuple[str, ...] = ()
    readiness: ToolReadiness | None = None

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "contract": self.contract.to_dict(),
            "manual_paths": self.manual_paths,
            "quickstart": self.quickstart,
            "notes": self.notes,
            "see_also": self.see_also,
        }
        if self.readiness is not None:
            data["readiness"] = self.readiness.to_dict()
            data["completion_hints"] = tuple(
                hint.to_dict()
                for section in self.readiness.missing_required_sections
                if (hint := section_completion_hint(section)) is not None
            )
        return data


SECTION_COMPLETION_HINTS: dict[str, SectionCompletionHint] = {
    "SOURCE": SectionCompletionHint(
        "SOURCE",
        "Structure / ORACLE Import",
        "Preprocess",
        "matrix link preprocess",
    ),
    "BASIC": SectionCompletionHint(
        "BASIC",
        "Structure / ORACLE Import",
        "Preprocess",
        "matrix link preprocess",
    ),
    "VALIDATION": SectionCompletionHint(
        "VALIDATION",
        "Structure / ORACLE Import",
        "Validate",
        "matrix validate",
        "validation is the gate before SMITH/SONIC",
    ),
    "SYMMETRY": SectionCompletionHint(
        "SYMMETRY",
        "Structure / ORACLE Import",
        "Preprocess",
        "matrix link preprocess",
    ),
    "TOPOLOGY": SectionCompletionHint(
        "TOPOLOGY",
        "Structure / ORACLE Import",
        "Preprocess",
        "matrix link preprocess",
        "topology is computed once from the normalized geometry",
    ),
    "SYNTHONS": SectionCompletionHint(
        "SYNTHONS",
        "Structure / ORACLE Import",
        "Preprocess",
        "matrix link preprocess",
        "synthons are derived from the shared topology/charge model",
    ),
    "PRIMITIVES": SectionCompletionHint(
        "PRIMITIVES",
        "Structure / ORACLE Import",
        "Preprocess",
        "matrix link preprocess",
        "ORACLE freezes the redundant primitive/Wilson-B source consumed by SMITH",
    ),
    "FRAGMENTS": SectionCompletionHint(
        "FRAGMENTS",
        "Structure",
        "Build Fragments",
        "matrix fragments build",
    ),
    "GIC": SectionCompletionHint(
        "GIC",
        "SMITH / SONIC",
        "Build GICs",
        "matrix smith build; compatibility alias: matrix gicforge build",
    ),
    "SYCART": SectionCompletionHint(
        "SYCART",
        "SMITH / SONIC",
        "Build GICs with SYCART",
        "matrix smith build --sycart; compatibility alias: matrix gicforge build --sycart",
    ),
    "CARTESIAN_HESSIAN": SectionCompletionHint(
        "CARTESIAN_HESSIAN",
        "QM Jobs",
        "Promote Gaussian FCHK/log or ORCA output",
        "matrix gaussian promote-fchk; matrix gaussian promote-log-hessian; matrix orca promote",
        "TRINITY can also accept an FCHK file directly",
    ),
    "NORMAL_MODES": SectionCompletionHint(
        "NORMAL_MODES",
        "QM Jobs",
        "Promote Gaussian FCHK/log",
        "matrix gaussian promote-fchk; matrix gaussian promote-log-hessian",
    ),
    "QFF": SectionCompletionHint(
        "QFF",
        "QM Jobs",
        "Promote Gaussian FCHK with QFF data",
        "matrix gaussian promote-fchk",
    ),
    "ROTATIONAL": SectionCompletionHint(
        "ROTATIONAL",
        "QM Jobs",
        "Promote rovibrational Gaussian log",
        "matrix gaussian promote-rovib",
    ),
    "VIBRATIONAL": SectionCompletionHint(
        "VIBRATIONAL",
        "QM Jobs",
        "Promote rovibrational Gaussian log",
        "matrix gaussian promote-rovib",
    ),
    "DELTABVIB": SectionCompletionHint(
        "DELTABVIB",
        "QM Jobs",
        "Promote rovibrational Gaussian log",
        "matrix gaussian promote-rovib",
    ),
    "ELECTRONIC": SectionCompletionHint(
        "ELECTRONIC",
        "QM Jobs / Electronic",
        "Promote electronic states",
        "matrix gaussian promote-electronic",
    ),
    "TRANSITIONS": SectionCompletionHint(
        "TRANSITIONS",
        "QM Jobs / Electronic",
        "Promote electronic transitions",
        "matrix gaussian promote-electronic",
    ),
    "ORBITALS": SectionCompletionHint(
        "ORBITALS",
        "QM Jobs / Electronic",
        "Register orbital or density files",
        "matrix gaussian promote-electronic --orbital-file FILE",
        "view with Molden, Avogadro or MOrbVis-browser where available",
    ),
    "PROPERTIES": SectionCompletionHint(
        "PROPERTIES",
        "QM Jobs",
        "Promote normalized QM properties",
        "matrix properties summary molecule.xyzin; adapter-specific promote commands",
        "stores program, method, level, unit and conversion metadata for QM properties",
    ),
    "GF_PED": SectionCompletionHint(
        "GF_PED",
        "TRINITY",
        "Run TRINITY harmonic analysis",
        "matrix gf",
    ),
    "ISOTOPOLOGUES": SectionCompletionHint(
        "ISOTOPOLOGUES",
        "SEFit / Rotational Spectroscopy",
        "Import or define isotopologues",
        "matrix semiexp --job ... --xyzin ...",
        "the #ISOTOPOLOGUES section is the standalone SEFit input contract",
    ),
    "MORPHEUS": SectionCompletionHint(
        "MORPHEUS",
        "SEFit",
        "Run SEFit",
        "matrix semiexp",
    ),
    "VPT2_VCI": SectionCompletionHint(
        "VPT2_VCI",
        "Anharmonic",
        "Run or collect VPT2/VCI",
        "matrix vpt2-vci",
    ),
    "DVR": SectionCompletionHint(
        "DVR",
        "Anharmonic",
        "Prepare, run or collect DVR",
        "matrix dvr prepare/run/collect",
    ),
    "THERMO": SectionCompletionHint(
        "THERMO",
        "Thermo/Kinetics",
        "Run Thermo",
        "matrix thermo",
    ),
    "KINETICS": SectionCompletionHint(
        "KINETICS",
        "Thermo/Kinetics",
        "Run single-reaction TST/RRKM",
        "matrix kinetics single",
        "uses normalized reactant/transition-state DOS and a network-ready manifest",
    ),
    "LINK": SectionCompletionHint(
        "LINK",
        "LINK",
        "Prepare or run LINK",
        "matrix link prepare; matrix link optimize",
    ),
}


MANUAL_PATHS: dict[str, tuple[str, ...]] = {
    "link": (
        "README.md",
        "docs/architecture/MATRIX_XYZIN_CONTAINER.md",
        "docs/architecture/ADR-0003-UNIFIED-GEOMETRY-AND-QM-PARSERS.md",
    ),
    "qm_adapters": (
        "docs/manuals/matrix_first_release_manual.pdf",
        "docs/manuals/oracle_qm_remote_manual.pdf",
        "docs/architecture/ADR-0003-UNIFIED-GEOMETRY-AND-QM-PARSERS.md",
        "docs/architecture/MATRIX_XYZIN_CONTAINER.md",
    ),
    "fragments": (
        "docs/architecture/ADR-0005-FRAGMENT-LEGO-TOPOLOGY-CLIENT.md",
        "docs/architecture/LCB25_IMPORT.md",
    ),
    "gicforge": (
        "docs/manuals/gicforge_manual.pdf",
        "docs/manuals/gicforge_manual.tex",
        "docs/architecture/SMITH_GIC_METHOD.md",
    ),
    "gf": (
        "docs/manuals/gf_manual.pdf",
        "docs/manuals/gf_manual.tex",
        "docs/architecture/PACKAGE_ARCHITECTURE.md",
    ),
    "morpheus": (
        "docs/manuals/morpheus_manual.pdf",
        "docs/manuals/morpheus_manual.tex",
        "docs/manuals/multistructure_manual.pdf",
        "docs/manuals/multistructure_manual.tex",
    ),
    "trinity": (
        "docs/manuals/gf_manual.pdf",
        "docs/manuals/gf_manual.tex",
        "docs/architecture/ADR-0008-STANDALONE-XYZIN-WORKFLOWS.md",
    ),
    "rovib": (
        "docs/manuals/matrix_first_release_manual.pdf",
        "docs/architecture/ADR-0010-ROTATIONAL-SPECTROSCOPY-WMSROT.md",
        "docs/architecture/MATRIX_DIAGONALIZATION.md",
    ),
    "thermo": (
        "docs/manuals/matrix_first_release_manual.pdf",
        "docs/manuals/matrix_first_release_manual.tex",
    ),
    "vpt2_vci": (
        "docs/manuals/matrix_first_release_manual.pdf",
        "docs/architecture/PACKAGE_ARCHITECTURE.md",
        "docs/architecture/MATRIX_DIAGONALIZATION.md",
    ),
    "dvr": (
        "docs/manuals/matrix_first_release_manual.pdf",
        "docs/architecture/MATRIX_DIAGONALIZATION.md",
    ),
    "gui": (
        "docs/architecture/ORACLE_GUI_ARCHITECTURE.md",
        "docs/manuals/matrix_first_release_manual.pdf",
        "docs/INSTALL_MATRIX.md",
    ),
}


QUICKSTART: dict[str, tuple[str, ...]] = {
    "link": (
        "matrix link preprocess SOURCE molecule.xyzin",
        "matrix validate molecule.xyzin",
    ),
    "qm_adapters": (
        "matrix qm remote-submit calc.gjf --engine gdv32 --machine ORACLE",
        "matrix qm remote-status --machine ORACLE",
        "matrix qm remote-fetch JOB --machine ORACLE --dest runs",
        "matrix gaussian promote-fchk calc.fchk molecule.xyzin",
        "matrix gaussian promote-rovib calc.log molecule.xyzin",
        "matrix gaussian promote-electronic calc.log molecule.xyzin",
        "matrix gaussian promote-quadrupole calc.log molecule.xyzin",
        "matrix molpro promote-quadrupole calc.out molecule.xyzin --atom 1 --isotope 14N",
        "matrix orca promote-quadrupole calc.out molecule.xyzin",
        "matrix orca promote calc.out molecule.xyzin",
        "matrix properties summary molecule.xyzin",
    ),
    "fragments": ("matrix fragments build molecule.xyzin",),
    "gicforge": (
        "matrix smith standalone molecule.smith.xyz molecule.xyzin",
        "matrix smith build molecule.xyzin",
        "matrix smith report molecule.xyzin --output sonic_report.txt",
        "matrix smith motions molecule.xyzin",
        "matrix smith view molecule.xyzin",
    ),
    "gf": (
        "matrix gaussian promote-fchk calc.fchk molecule.xyzin",
        "matrix gf --xyzin molecule.xyzin",
        'matrix gf --xyzin molecule.xyzin --scale-class "CH:0.970:R(1,6)|R(2,7)"',
        'matrix gf --xyzin molecule.xyzin --scale-preview --scale-class "CH:0.970:R("',
    ),
    "morpheus": ("matrix semiexp --xyzin molecule.xyzin --job job.toml --outdir run",),
    "trinity": (
        "matrix gf --xyzin molecule.xyzin",
        "matrix trinity ir-from-fchk calc.fchk --output ir.json",
    ),
    "rovib": (
        "matrix lock summarize molecule.xyzin",
        "matrix lock rotor-input molecule.xyzin --out wmsrot.inp",
        "matrix lock rotor-spectrum molecule.xyzin --out rotational.csv",
    ),
    "thermo": ("matrix thermo molecule.xyzin",),
    "vpt2_vci": ("matrix vpt2-vci --xyzin molecule.xyzin --run-dir run",),
    "dvr": (
        "matrix dvr prepare --xyzin molecule.xyzin --log scan.log --outdir dvr",
        "matrix dvr run --xyzin molecule.xyzin",
        "matrix dvr collect --xyzin molecule.xyzin",
    ),
    "gui": ("matrix gui molecule.xyzin", "matrix gui --project-root ./project"),
}


SEE_ALSO: dict[str, tuple[str, ...]] = {
    "link": ("qm_adapters", "gicforge", "fragments"),
    "qm_adapters": ("link", "gf", "rovib", "vpt2_vci", "dvr", "gui"),
    "fragments": ("link", "gicforge"),
    "gicforge": ("link", "gf", "morpheus", "trinity"),
    "gf": ("gicforge", "rovib", "vpt2_vci"),
    "morpheus": ("gicforge", "rovib"),
    "trinity": ("gicforge", "gf"),
    "rovib": ("gf", "thermo"),
    "thermo": ("rovib", "dvr"),
    "vpt2_vci": ("gf", "dvr"),
    "dvr": ("vpt2_vci", "thermo"),
    "gui": ("link", "gicforge", "gf", "morpheus", "vpt2_vci", "dvr"),
}


def section_completion_hint(section: str) -> SectionCompletionHint | None:
    return SECTION_COMPLETION_HINTS.get(_normalize_section(section))


def missing_sections_guidance(sections: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    lines: list[str] = []
    for section in sections:
        hint = section_completion_hint(section)
        if hint is None:
            lines.append(
                f"#{_normalize_section(section)}: add this section with the owning adapter/tool"
            )
        else:
            lines.append(hint.line())
    return tuple(lines)


def missing_sections_message(sections: tuple[str, ...] | list[str]) -> str:
    normalized = tuple(_normalize_section(section) for section in sections)
    if not normalized:
        return "project is not available"
    lines = ["missing " + ", ".join(normalized), "Suggested completion:"]
    lines.extend(f"- {line}" for line in missing_sections_guidance(normalized))
    return "\n".join(lines)


def tool_help(tool: ToolContract | str, *, xyzin: Path | str | None = None) -> ToolHelp:
    contract = tool_contract(tool) if isinstance(tool, str) else tool
    readiness = tool_contract_readiness(xyzin, contract) if xyzin is not None else None
    return ToolHelp(
        contract=contract,
        manual_paths=MANUAL_PATHS.get(contract.key, ()),
        quickstart=QUICKSTART.get(contract.key, (contract.standalone_command,)),
        notes=_tool_help_notes(contract),
        see_also=SEE_ALSO.get(contract.key, ()),
        readiness=readiness,
    )


def online_help_records(
    tool: str | None = None,
    *,
    xyzin: Path | str | None = None,
    include_gui: bool = True,
) -> tuple[ToolHelp, ...]:
    if tool:
        return (tool_help(tool, xyzin=xyzin),)
    return tuple(
        tool_help(contract, xyzin=xyzin) for contract in tool_contracts(include_gui=include_gui)
    )


def online_help_lines(
    tool: str | None = None,
    *,
    xyzin: Path | str | None = None,
    include_gui: bool = True,
) -> list[str]:
    records = online_help_records(tool, xyzin=xyzin, include_gui=include_gui)
    lines: list[str] = []
    for index, record in enumerate(records):
        if index:
            lines.append("")
        lines.extend(tool_help_lines(record))
    return lines


def tool_help_lines(record: ToolHelp) -> list[str]:
    contract = record.contract
    display = contract.display_name
    if contract.planned_name and contract.planned_name not in display:
        display = f"{contract.planned_name} / {display}"
    lines = [f"{contract.key}: {display}"]
    if contract.expanded_name:
        lines.append(f"  acronym: {contract.expanded_name}")
    lines.extend(
        [
            f"  package: {contract.current_package}",
            f"  command: {contract.standalone_command}",
            f"  required xyzin: {_join_sections(contract.required_sections)}",
            f"  optional xyzin: {_join_sections(contract.optional_sections)}",
            f"  produced xyzin: {_join_sections(contract.produced_sections)}",
            f"  status: {contract.status}",
        ]
    )
    if record.manual_paths:
        lines.append("  manuals:")
        lines.extend(f"    - {path}" for path in record.manual_paths)
    if record.quickstart:
        lines.append("  quickstart:")
        lines.extend(f"    $ {command}" for command in record.quickstart)
    for note in record.notes:
        lines.append(f"  note: {note}")
    if record.readiness is not None:
        readiness = record.readiness
        lines.append(
            "  current xyzin: "
            f"{readiness.xyzin_path} ready={int(readiness.ready)} "
            f"missing={_join_sections(readiness.missing_required_sections)}"
        )
        if readiness.missing_required_sections:
            lines.append("  completion:")
            lines.extend(
                f"    - {line}"
                for line in missing_sections_guidance(readiness.missing_required_sections)
            )
    if record.see_also:
        lines.append(f"  see also: {', '.join(record.see_also)}")
    return lines


def online_help_markdown(
    tool: str | None = None,
    *,
    xyzin: Path | str | None = None,
    include_gui: bool = True,
) -> str:
    records = online_help_records(tool, xyzin=xyzin, include_gui=include_gui)
    lines = ["# MATRIX Online Help", ""]
    for record in records:
        contract = record.contract
        title = contract.display_name
        if contract.planned_name and contract.planned_name not in title:
            title = f"{contract.planned_name} / {title}"
        lines.extend(
            [
                f"## {title}",
                "",
                f"- Key: `{contract.key}`",
                f"- Package: `{contract.current_package}`",
                f"- Command: `{contract.standalone_command}`",
                f"- Required xyzin: {_join_sections(contract.required_sections)}",
                f"- Optional xyzin: {_join_sections(contract.optional_sections)}",
                f"- Produced xyzin: {_join_sections(contract.produced_sections)}",
                f"- Status: `{contract.status}`",
            ]
        )
        if contract.expanded_name:
            lines.append(f"- Acronym: {contract.expanded_name}")
        if record.manual_paths:
            lines.append("- Manuals: " + ", ".join(f"`{path}`" for path in record.manual_paths))
        if record.quickstart:
            lines.append("")
            lines.append("Quickstart:")
            lines.append("")
            lines.extend(f"```bash\n{command}\n```" for command in record.quickstart)
        if record.notes:
            lines.append("")
            lines.extend(f"- {note}" for note in record.notes)
        if record.readiness is not None:
            readiness = record.readiness
            readiness_status = (
                "ready"
                if readiness.ready
                else "missing " + _join_sections(readiness.missing_required_sections)
            )
            lines.append("")
            lines.append(f"Readiness for `{readiness.xyzin_path}`: {readiness_status}")
            if readiness.missing_required_sections:
                lines.extend(
                    f"- {line}"
                    for line in missing_sections_guidance(readiness.missing_required_sections)
                )
        if record.see_also:
            lines.append("")
            lines.append("See also: " + ", ".join(f"`{item}`" for item in record.see_also))
        lines.append("")
    return "\n".join(lines).rstrip()


def online_help_json(
    tool: str | None = None,
    *,
    xyzin: Path | str | None = None,
    include_gui: bool = True,
) -> str:
    records = online_help_records(tool, xyzin=xyzin, include_gui=include_gui)
    return json.dumps([record.to_dict() for record in records], indent=2, sort_keys=True)


def online_help_text(
    tool: str | None = None,
    *,
    xyzin: Path | str | None = None,
    include_gui: bool = True,
) -> str:
    return "\n".join(online_help_lines(tool, xyzin=xyzin, include_gui=include_gui))


def _tool_help_notes(contract: ToolContract) -> tuple[str, ...]:
    notes: list[str] = []
    if contract.notes:
        notes.append(contract.notes)
    if contract.required_sections:
        notes.append("Can run standalone when the listed required xyzin sections are present.")
    if contract.owned_sections:
        notes.append("This tool owns the listed output sections; downstream tools consume them.")
    return tuple(notes)


def _normalize_section(section: str) -> str:
    return section.strip().upper().lstrip("#")


def _join_sections(sections: tuple[str, ...]) -> str:
    return ", ".join(sections) if sections else "none"
