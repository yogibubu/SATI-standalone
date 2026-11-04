from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from matrix_chem import (
    format_substitutions,
    read_xyzin_isotopologue_records,
)
if TYPE_CHECKING:
    from matrix_morpheus import MorpheusSection

from .commands import OracleGuiCommand, semiexp_command


@dataclass(frozen=True)
class SEFitTable:
    title: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class SEFitSummary:
    status: str = ""
    run_dir: Path | None = None
    manifest_path: Path | None = None
    text_report_path: Path | None = None
    html_report_path: Path | None = None
    latex_tables_path: Path | None = None
    backend: str = ""
    coordinate_model: str = ""
    observable: str = ""
    components: tuple[str, ...] = ()
    rms_MHz: float = 0.0
    rotational_rms_MHz: float = 0.0
    iterations: int = 0
    stationary_point: str = ""
    convergence: str = ""
    rank: int = 0
    condition_number: float = 0.0
    isotopologue_count: int = 0
    parameter_count: int = 0
    active_parameter_count: int = 0
    warning_count: int = 0


@dataclass(frozen=True)
class SEFitGuiState:
    xyzin: Path
    exists: bool
    ready: bool
    summary: SEFitSummary
    isotopologues: SEFitTable
    outputs: SEFitTable
    diagnostics: SEFitTable
    messages: tuple[str, ...] = ()


class OracleSEFitController:
    def __init__(self, xyzin: Path | str | None = None) -> None:
        self.xyzin = None if xyzin is None else Path(xyzin)

    def set_xyzin(self, xyzin: Path | str | None) -> SEFitGuiState | None:
        self.xyzin = None if xyzin is None else Path(xyzin)
        if self.xyzin is None:
            return None
        return self.state()

    def state(self) -> SEFitGuiState:
        if self.xyzin is None:
            raise ValueError("no MATRIX xyzin project is loaded")
        return load_sefit_gui_state(self.xyzin)

    def run_command(
        self,
        *,
        job: Path | str,
        outdir: Path | str,
        backend: str = "python",
        write_section: bool = True,
        extra_args: tuple[str, ...] = (),
    ) -> OracleGuiCommand:
        if self.xyzin is None:
            raise ValueError("no MATRIX xyzin project is loaded")
        return semiexp_command(
            job,
            outdir,
            xyzin=self.xyzin,
            backend=backend,
            write_section=write_section,
            extra_args=extra_args,
        )


def load_sefit_gui_state(path: Path | str) -> SEFitGuiState:
    from matrix_morpheus import read_morpheus_section

    target = Path(path)
    empty = _empty_state(target, exists=target.exists())
    if not target.exists():
        return _replace_messages(empty, (f"Missing file: {target}",))
    messages: list[str] = []
    isotopologues = _isotopologue_table(target, messages)
    try:
        section = read_morpheus_section(target)
    except (OSError, ValueError) as exc:
        return SEFitGuiState(
            xyzin=target,
            exists=True,
            ready=False,
            summary=empty.summary,
            isotopologues=isotopologues,
            outputs=empty.outputs,
            diagnostics=empty.diagnostics,
            messages=tuple([*messages, str(exc)]),
        )
    return SEFitGuiState(
        xyzin=target,
        exists=True,
        ready=True,
        summary=_summary(section),
        isotopologues=isotopologues,
        outputs=_outputs_table(section),
        diagnostics=_diagnostics_table(section),
        messages=tuple(messages),
    )


def sefit_gui_state_lines(state: SEFitGuiState) -> list[str]:
    if not state.ready:
        return [
            f"xyzin: {state.xyzin}",
            f"ready: {int(state.ready)}",
            *state.messages,
        ]
    summary = state.summary
    return [
        f"xyzin: {state.xyzin}",
        f"ready: {int(state.ready)}",
        f"status: {summary.status}",
        f"backend: {summary.backend}",
        f"coordinate model: {summary.coordinate_model}",
        f"observable: {summary.observable}",
        f"components: {','.join(summary.components)}",
        f"rms MHz: {summary.rms_MHz:.8g}",
        f"rotational rms MHz: {summary.rotational_rms_MHz:.8g}",
        f"iterations: {summary.iterations}",
        f"stationary point: {summary.stationary_point}",
        f"convergence: {summary.convergence}",
        f"rank: {summary.rank}",
        f"condition number: {summary.condition_number:.8g}",
        f"isotopologues: {summary.isotopologue_count}",
        f"parameters: {summary.parameter_count}",
        f"active parameters: {summary.active_parameter_count}",
        f"warnings: {summary.warning_count}",
    ]


def default_sefit_outdir(xyzin: Path | str) -> Path:
    target = Path(xyzin)
    return target.with_name(f"{target.stem}.semiexp")


def _summary(section: MorpheusSection) -> SEFitSummary:
    return SEFitSummary(
        status=section.status,
        run_dir=section.run_dir,
        manifest_path=section.manifest_path,
        text_report_path=section.text_report_path,
        html_report_path=section.html_report_path,
        latex_tables_path=section.latex_tables_path,
        backend=section.backend,
        coordinate_model=section.coordinate_model,
        observable=section.observable,
        components=section.components,
        rms_MHz=section.rms_MHz,
        rotational_rms_MHz=section.rotational_rms_MHz,
        iterations=section.iterations,
        stationary_point=section.stationary_point,
        convergence=section.convergence,
        rank=section.rank,
        condition_number=section.condition_number,
        isotopologue_count=section.isotopologue_count,
        parameter_count=section.parameter_count,
        active_parameter_count=section.active_parameter_count,
        warning_count=section.warning_count,
    )


def _isotopologue_table(target: Path, messages: list[str]) -> SEFitTable:
    try:
        records = read_xyzin_isotopologue_records(target)
    except (OSError, ValueError) as exc:
        messages.append(str(exc))
        records = ()
    rows = []
    for record in records:
        rows.append(
            (
                record.label,
                format_substitutions(record.substitutions) or "parent",
                _triple(record.rotational_MHz),
                _triple(record.deltavib_MHz),
                _triple(record.deltael_MHz),
                _triple(record.sigma_MHz),
            )
        )
    return SEFitTable(
        "Isotopologues",
        ("Label", "Substitutions", "B0 MHz", "DeltaVib MHz", "DeltaEl MHz", "Sigma MHz"),
        tuple(rows),
    )


def _outputs_table(section: MorpheusSection) -> SEFitTable:
    rows = (
        ("Run dir", _path_text(section.run_dir)),
        ("Manifest", _path_text(section.manifest_path)),
        ("Text report", _path_text(section.text_report_path)),
        ("HTML report", _path_text(section.html_report_path)),
        ("LaTeX tables", _path_text(section.latex_tables_path)),
        ("Geometry", _path_text(section.geometry_path)),
        ("Parameters", _path_text(section.parameters_path)),
        ("Residuals", _path_text(section.residuals_path)),
        ("Rotational constants", _path_text(section.rotational_constants_path)),
        ("Diagnostics", _path_text(section.diagnostics_path)),
    )
    return SEFitTable("Outputs", ("Output", "Path"), rows)


def _diagnostics_table(section: MorpheusSection) -> SEFitTable:
    rows = (
        ("Status", section.status),
        ("Backend", section.backend),
        ("Coordinate model", section.coordinate_model),
        ("Observable", section.observable),
        ("Components", ",".join(section.components)),
        ("RMS MHz", _format_float(section.rms_MHz)),
        ("Rotational RMS MHz", _format_float(section.rotational_rms_MHz)),
        ("Rotational MSE MHz2", _format_float(section.rotational_mean_square_MHz2)),
        ("Iterations", str(section.iterations)),
        ("Stationary point", section.stationary_point),
        ("Convergence", section.convergence),
        ("Rank", str(section.rank)),
        ("Condition number", _format_float(section.condition_number)),
        ("Isotopologues", str(section.isotopologue_count)),
        ("Parameters", str(section.parameter_count)),
        ("Active parameters", str(section.active_parameter_count)),
        ("Warnings", str(section.warning_count)),
    )
    return SEFitTable("Diagnostics", ("Field", "Value"), rows)


def _empty_state(target: Path, *, exists: bool) -> SEFitGuiState:
    return SEFitGuiState(
        xyzin=target,
        exists=exists,
        ready=False,
        summary=SEFitSummary(),
        isotopologues=SEFitTable(
            "Isotopologues",
            ("Label", "Substitutions", "B0 MHz", "DeltaVib MHz", "DeltaEl MHz", "Sigma MHz"),
            (),
        ),
        outputs=SEFitTable("Outputs", ("Output", "Path"), ()),
        diagnostics=SEFitTable("Diagnostics", ("Field", "Value"), ()),
        messages=(),
    )


def _replace_messages(state: SEFitGuiState, messages: tuple[str, ...]) -> SEFitGuiState:
    return SEFitGuiState(
        xyzin=state.xyzin,
        exists=state.exists,
        ready=state.ready,
        summary=state.summary,
        isotopologues=state.isotopologues,
        outputs=state.outputs,
        diagnostics=state.diagnostics,
        messages=messages,
    )


def _triple(values: tuple[float, float, float] | None) -> str:
    if values is None:
        return ""
    return ",".join(_format_float(value) for value in values)


def _path_text(value: Path | None) -> str:
    return "" if value is None else str(value)


def _format_float(value: float) -> str:
    return f"{float(value):.8g}"
