from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matrix_gf import GFPEDSection

from .commands import OracleGuiCommand, gf_command, gf_scaling_preview_command


@dataclass(frozen=True)
class GFTable:
    title: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class GFSummary:
    status: str = ""
    source_kind: str = ""
    source_path: Path | None = None
    hessian_source: str = ""
    coordinate_source: str = ""
    report_path: Path | None = None
    csv_dir: Path | None = None
    point_group: str = ""
    symmetrized_gics: bool = False
    matrix_model: str = ""
    hessian_correction: str = ""
    force_threshold: float | None = None
    mode_count: int = 0
    gic_count: int = 0


@dataclass(frozen=True)
class GFGuiState:
    xyzin: Path
    exists: bool
    ready: bool
    summary: GFSummary
    frequencies: GFTable
    gics: GFTable
    ped: GFTable
    diagnostics: GFTable
    messages: tuple[str, ...] = ()


class OracleGFController:
    def __init__(self, xyzin: Path | str | None = None) -> None:
        self.xyzin = None if xyzin is None else Path(xyzin)

    def set_xyzin(self, xyzin: Path | str | None) -> GFGuiState | None:
        self.xyzin = None if xyzin is None else Path(xyzin)
        if self.xyzin is None:
            return None
        return self.state()

    def state(self) -> GFGuiState:
        if self.xyzin is None:
            raise ValueError("no MATRIX xyzin project is loaded")
        return load_gf_gui_state(self.xyzin)

    def run_command(
        self,
        *,
        fchk: Path | str | None = None,
        out: Path | str | None = None,
        csv_dir: Path | str | None = None,
        scale_file: Path | str | None = None,
        scale_records: tuple[str, ...] = (),
        scale_class_records: tuple[str, ...] = (),
        local: bool = False,
        symmetry_blocks: bool = True,
        force_threshold: float | None = None,
        subtract_electrostatic: bool = False,
        subtract_uff_vdw: bool = False,
        nonbonded_14_scale: float = 0.5,
        write_section: bool = True,
    ) -> OracleGuiCommand:
        if self.xyzin is None:
            raise ValueError("no MATRIX xyzin project is loaded")
        return gf_command(
            self.xyzin,
            fchk=fchk,
            out=out,
            csv_dir=csv_dir,
            scale_file=scale_file,
            scale_records=scale_records,
            scale_class_records=scale_class_records,
            local=local,
            symmetry_blocks=symmetry_blocks,
            force_threshold=force_threshold,
            subtract_electrostatic=subtract_electrostatic,
            subtract_uff_vdw=subtract_uff_vdw,
            nonbonded_14_scale=nonbonded_14_scale,
            write_section=write_section,
        )

    def scaling_preview_command(
        self,
        *,
        scale_file: Path | str | None = None,
        scale_records: tuple[str, ...] = (),
        scale_class_records: tuple[str, ...] = (),
    ) -> OracleGuiCommand:
        if self.xyzin is None:
            raise ValueError("no MATRIX xyzin project is loaded")
        return gf_scaling_preview_command(
            self.xyzin,
            scale_file=scale_file,
            scale_records=scale_records,
            scale_class_records=scale_class_records,
        )


def load_gf_gui_state(path: Path | str) -> GFGuiState:
    from matrix_gf import read_gf_ped_section

    target = Path(path)
    empty = _empty_state(target, exists=target.exists())
    if not target.exists():
        return _replace_messages(empty, (f"Missing file: {target}",))
    try:
        section = read_gf_ped_section(target)
    except (OSError, ValueError) as exc:
        return _replace_messages(empty, (str(exc),))
    return GFGuiState(
        xyzin=target,
        exists=True,
        ready=True,
        summary=_summary(section),
        frequencies=_frequency_table(section),
        gics=_gic_table(section),
        ped=_ped_table(section),
        diagnostics=_diagnostics_table(section),
        messages=(),
    )


def gf_gui_state_lines(state: GFGuiState) -> list[str]:
    if not state.ready:
        return [
            f"xyzin: {state.xyzin}",
            f"ready: {int(state.ready)}",
            *state.messages,
        ]
    summary = state.summary
    lines = [
        f"xyzin: {state.xyzin}",
        f"ready: {int(state.ready)}",
        f"status: {summary.status}",
        f"source kind: {summary.source_kind}",
        f"hessian source: {summary.hessian_source}",
        f"coordinate source: {summary.coordinate_source}",
        f"point group: {summary.point_group}",
        f"symmetrized GICs: {summary.symmetrized_gics}",
        f"matrix model: {summary.matrix_model}",
        f"hessian correction: {summary.hessian_correction}",
        f"mode count: {summary.mode_count}",
        f"GIC count: {summary.gic_count}",
    ]
    if summary.force_threshold is not None:
        lines.append(f"force threshold: {summary.force_threshold:g}")
    if summary.report_path is not None:
        lines.append(f"report: {summary.report_path}")
    if summary.csv_dir is not None:
        lines.append(f"csv dir: {summary.csv_dir}")
    return lines


def default_gf_report_output(xyzin: Path | str) -> Path:
    target = Path(xyzin)
    return target.with_name(f"{target.stem}.gf_ped_report.txt")


def default_gf_csv_dir(xyzin: Path | str) -> Path:
    target = Path(xyzin)
    return target.with_name(f"{target.stem}.gf_csv")


def _summary(section: GFPEDSection) -> GFSummary:
    return GFSummary(
        status=section.status,
        source_kind=section.source_kind,
        source_path=section.source_path,
        hessian_source=section.hessian_source,
        coordinate_source=section.coordinate_source,
        report_path=section.report_path,
        csv_dir=section.csv_dir,
        point_group=section.point_group,
        symmetrized_gics=section.symmetrized_gics,
        matrix_model=section.matrix_model,
        hessian_correction=section.hessian_correction,
        force_threshold=section.force_threshold,
        mode_count=len(section.modes),
        gic_count=len(section.gics),
    )


def _frequency_table(section: GFPEDSection) -> GFTable:
    rows = tuple(
        (
            str(mode.index),
            _format_float(mode.frequency_cm),
            mode.block,
        )
        for mode in section.modes
    )
    return GFTable("Frequencies", ("Mode", "Frequency cm-1", "Block"), rows)


def _gic_table(section: GFPEDSection) -> GFTable:
    rows = []
    for gic in section.gics:
        mode_index, contribution = _dominant_ped(gic.ped)
        rows.append(
            (
                gic.identifier,
                gic.name,
                gic.irrep,
                "" if gic.scaling_factor is None else _format_float(gic.scaling_factor),
                str(mode_index) if mode_index else "",
                "" if contribution is None else _format_float(contribution),
                gic.label,
            )
        )
    return GFTable(
        "GICs",
        ("ID", "Name", "Irrep", "Scale", "Max PED mode", "Max PED %", "Label"),
        tuple(rows),
    )


def _ped_table(section: GFPEDSection) -> GFTable:
    columns = ("GIC", "Name", "Irrep", *(f"M{mode.index:02d}" for mode in section.modes))
    rows = tuple(
        (
            gic.identifier,
            gic.name,
            gic.irrep,
            *(_format_float(value) for value in gic.ped),
        )
        for gic in section.gics
    )
    return GFTable("PED", columns, rows)


def _diagnostics_table(section: GFPEDSection) -> GFTable:
    rows = [
        ("Status", section.status),
        ("Source kind", section.source_kind),
        ("Source path", "" if section.source_path is None else str(section.source_path)),
        ("Hessian source", section.hessian_source),
        ("Coordinate source", section.coordinate_source),
        ("Report", "" if section.report_path is None else str(section.report_path)),
        ("CSV dir", "" if section.csv_dir is None else str(section.csv_dir)),
        ("Point group", section.point_group),
        ("Symmetrized GICs", str(section.symmetrized_gics)),
        ("Matrix model", section.matrix_model),
        ("Hessian correction", section.hessian_correction),
        (
            "Force threshold",
            "" if section.force_threshold is None else _format_float(section.force_threshold),
        ),
        ("Mode count", str(len(section.modes))),
        ("GIC count", str(len(section.gics))),
    ]
    return GFTable("Diagnostics", ("Field", "Value"), tuple(rows))


def _empty_state(target: Path, *, exists: bool) -> GFGuiState:
    return GFGuiState(
        xyzin=target,
        exists=exists,
        ready=False,
        summary=GFSummary(),
        frequencies=GFTable("Frequencies", ("Mode", "Frequency cm-1", "Block"), ()),
        gics=GFTable(
            "GICs",
            ("ID", "Name", "Irrep", "Scale", "Max PED mode", "Max PED %", "Label"),
            (),
        ),
        ped=GFTable("PED", ("GIC", "Name", "Irrep"), ()),
        diagnostics=GFTable("Diagnostics", ("Field", "Value"), ()),
        messages=(),
    )


def _replace_messages(state: GFGuiState, messages: tuple[str, ...]) -> GFGuiState:
    return GFGuiState(
        xyzin=state.xyzin,
        exists=state.exists,
        ready=state.ready,
        summary=state.summary,
        frequencies=state.frequencies,
        gics=state.gics,
        ped=state.ped,
        diagnostics=state.diagnostics,
        messages=messages,
    )


def _dominant_ped(values: tuple[float, ...]) -> tuple[int, float | None]:
    if not values:
        return 0, None
    index, value = max(enumerate(values, start=1), key=lambda item: abs(item[1]))
    return index, float(value)


def _format_float(value: float) -> str:
    return f"{float(value):.8g}"
