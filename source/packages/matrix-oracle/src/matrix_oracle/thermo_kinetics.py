from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from matrix_core import has_section
from matrix_rovib import read_dos_binned
from matrix_thermo import read_thermo_section
from matrix_thermo.models import THERMO_KEYS, THERMO_LABELS, ThermoSection
from matrix_kinetics import read_kinetics_section

from .commands import (
    OracleGuiCommand,
    rovib_density_command,
    rovib_summary_command,
    rovib_vibrational_dos_command,
    single_reaction_kinetics_command,
    thermo_command,
)
from .publication import PublicationExportResult, export_thermo_table


@dataclass(frozen=True)
class ThermoKineticsTable:
    title: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class ThermoKineticsSummary:
    thermo_ready: bool = False
    kinetics_ready: bool = False
    vibrational_dos_path: Path | None = None
    rovibrational_dos_path: Path | None = None
    q_path: Path | None = None


@dataclass(frozen=True)
class ThermoKineticsGuiState:
    xyzin: Path
    exists: bool
    ready: bool
    summary: ThermoKineticsSummary
    thermo: ThermoKineticsTable
    dos: ThermoKineticsTable
    kinetics: ThermoKineticsTable
    messages: tuple[str, ...] = ()


class OracleThermoKineticsController:
    def __init__(self, xyzin: Path | str | None = None) -> None:
        self.xyzin = None if xyzin is None else Path(xyzin)

    def set_xyzin(self, xyzin: Path | str | None) -> ThermoKineticsGuiState | None:
        self.xyzin = None if xyzin is None else Path(xyzin)
        if self.xyzin is None:
            return None
        return self.state()

    def state(self) -> ThermoKineticsGuiState:
        if self.xyzin is None:
            raise ValueError("no MATRIX xyzin project is loaded")
        return load_thermo_kinetics_gui_state(self.xyzin)

    def rovib_summary_command(self) -> OracleGuiCommand:
        if self.xyzin is None:
            raise ValueError("no MATRIX xyzin project is loaded")
        return rovib_summary_command(self.xyzin)

    def thermo_command(
        self,
        *,
        out: Path | str | None = None,
        report: bool = True,
        write_section: bool = True,
        cutoff_cm1: float = 10.0,
        keep_low_positive: bool = False,
    ) -> OracleGuiCommand:
        if self.xyzin is None:
            raise ValueError("no MATRIX xyzin project is loaded")
        return thermo_command(
            self.xyzin,
            out=out,
            report=report,
            write_section=write_section,
            cutoff_cm1=cutoff_cm1,
            keep_low_positive=keep_low_positive,
        )

    def vibrational_dos_command(
        self,
        *,
        out: Path | str | None = None,
        vmax: int = 6,
        emax_cm1: float = 8000.0,
        bin_cm1: float = 50.0,
        ncap: float = 10.0,
    ) -> OracleGuiCommand:
        if self.xyzin is None:
            raise ValueError("no MATRIX xyzin project is loaded")
        return rovib_vibrational_dos_command(
            self.xyzin,
            out=out,
            vmax=vmax,
            emax_cm1=emax_cm1,
            bin_cm1=bin_cm1,
            ncap=ncap,
        )

    def rovibrational_dos_command(
        self,
        *,
        vib_dos: Path | str | None = None,
        out: Path | str | None = None,
        rot_out: Path | str | None = None,
        q_out: Path | str | None = None,
        emax_rot: float | None = None,
        jmax: int | None = None,
    ) -> OracleGuiCommand:
        if self.xyzin is None:
            raise ValueError("no MATRIX xyzin project is loaded")
        return rovib_density_command(
            self.xyzin,
            vib_dos=vib_dos,
            out=out,
            rot_out=rot_out,
            q_out=q_out,
            emax_rot=emax_rot,
            jmax=jmax,
        )

    def single_reaction_command(
        self,
        transition_state_xyzin: Path | str,
        *,
        reactant_dos: Path | str,
        transition_state_dos: Path | str,
        barrier_cm1: float,
        rrkm_out: Path | str | None = None,
        manifest: Path | str | None = None,
        report: Path | str | None = None,
        path_degeneracy: float = 1.0,
        tunneling: str = "none",
        imaginary_frequency_cm1: float | None = None,
    ) -> OracleGuiCommand:
        if self.xyzin is None:
            raise ValueError("no MATRIX xyzin project is loaded")
        return single_reaction_kinetics_command(
            self.xyzin,
            transition_state_xyzin,
            reactant_dos=reactant_dos,
            transition_state_dos=transition_state_dos,
            barrier_cm1=barrier_cm1,
            rrkm_out=rrkm_out,
            manifest=manifest,
            report=report,
            path_degeneracy=path_degeneracy,
            tunneling=tunneling,
            imaginary_frequency_cm1=imaginary_frequency_cm1,
        )

    def export_thermo_publication(
        self,
        outdir: Path | str,
        *,
        formats: tuple[str, ...],
    ) -> PublicationExportResult:
        if self.xyzin is None:
            raise ValueError("no MATRIX xyzin project is loaded")
        return export_thermo_table(self.xyzin, outdir, formats=formats)


def default_thermo_report_output(xyzin: Path | str) -> Path:
    target = Path(xyzin)
    return target.with_name(f"{target.stem}.thermo_report.txt")


def default_vibrational_dos_output(xyzin: Path | str) -> Path:
    target = Path(xyzin)
    return target.with_name(f"{target.stem}.dos_vib.dat")


def default_rotational_dos_output(xyzin: Path | str) -> Path:
    target = Path(xyzin)
    return target.with_name(f"{target.stem}.dos_rot.dat")


def default_rovibrational_dos_output(xyzin: Path | str) -> Path:
    target = Path(xyzin)
    return target.with_name(f"{target.stem}.dos_rovib.dat")


def default_rovib_q_output(xyzin: Path | str) -> Path:
    target = Path(xyzin)
    return target.with_name(f"{target.stem}.rovib_qt.dat")


def default_thermo_export_dir(xyzin: Path | str) -> Path:
    target = Path(xyzin)
    return target.with_name(f"{target.stem}.publication")


def load_thermo_kinetics_gui_state(path: Path | str) -> ThermoKineticsGuiState:
    target = Path(path)
    empty = _empty_state(target, exists=target.exists())
    if not target.exists():
        return _replace_messages(empty, (f"Missing file: {target}",))
    messages: list[str] = []
    thermo_section = None
    if has_section(target, "THERMO"):
        try:
            thermo_section = read_thermo_section(target)
        except (OSError, ValueError) as exc:
            messages.append(str(exc))
    else:
        messages.append("missing #THERMO section")
    kinetics_rows = _kinetics_rows(target)
    summary = ThermoKineticsSummary(
        thermo_ready=thermo_section is not None,
        kinetics_ready=has_section(target, "KINETICS"),
        vibrational_dos_path=_existing_path(default_vibrational_dos_output(target)),
        rovibrational_dos_path=_existing_path(default_rovibrational_dos_output(target)),
        q_path=_existing_path(default_rovib_q_output(target)),
    )
    return ThermoKineticsGuiState(
        xyzin=target,
        exists=True,
        ready=thermo_section is not None,
        summary=summary,
        thermo=_thermo_table(thermo_section),
        dos=_dos_table(target),
        kinetics=ThermoKineticsTable(
            "Kinetics",
            ("Item", "Status", "Value"),
            tuple(kinetics_rows),
        ),
        messages=tuple(messages),
    )


def thermo_kinetics_gui_state_lines(state: ThermoKineticsGuiState) -> list[str]:
    lines = [
        f"xyzin: {state.xyzin}",
        f"thermo ready: {int(state.summary.thermo_ready)}",
        f"kinetics ready: {int(state.summary.kinetics_ready)}",
    ]
    if state.summary.vibrational_dos_path is not None:
        lines.append(f"vibrational DOS: {state.summary.vibrational_dos_path}")
    if state.summary.rovibrational_dos_path is not None:
        lines.append(f"rovibrational DOS: {state.summary.rovibrational_dos_path}")
    if state.summary.q_path is not None:
        lines.append(f"rovib Q(T): {state.summary.q_path}")
    lines.extend(state.messages)
    return lines


def _empty_state(target: Path, *, exists: bool) -> ThermoKineticsGuiState:
    return ThermoKineticsGuiState(
        xyzin=target,
        exists=exists,
        ready=False,
        summary=ThermoKineticsSummary(),
        thermo=ThermoKineticsTable("Thermo", ("Component", *THERMO_KEYS), ()),
        dos=ThermoKineticsTable("DOS", ("File", "Exists", "Bins", "Emin", "Bin"), ()),
        kinetics=ThermoKineticsTable("Kinetics", ("Item", "Status", "Value"), ()),
        messages=(),
    )


def _replace_messages(
    state: ThermoKineticsGuiState,
    messages: tuple[str, ...],
) -> ThermoKineticsGuiState:
    return ThermoKineticsGuiState(
        xyzin=state.xyzin,
        exists=state.exists,
        ready=state.ready,
        summary=state.summary,
        thermo=state.thermo,
        dos=state.dos,
        kinetics=state.kinetics,
        messages=messages,
    )


def _thermo_table(section: ThermoSection | None) -> ThermoKineticsTable:
    rows: list[tuple[str, ...]] = []
    if section is not None:
        for label in THERMO_LABELS:
            contribution = section.contribution(label)
            if contribution is None:
                continue
            rows.append(
                (
                    label,
                    *(_format_value(getattr(contribution, key)) for key in THERMO_KEYS),
                )
            )
    return ThermoKineticsTable("Thermo", ("Component", *THERMO_KEYS), tuple(rows))


def _dos_table(target: Path) -> ThermoKineticsTable:
    files = (
        ("vibrational", default_vibrational_dos_output(target)),
        ("rotational", default_rotational_dos_output(target)),
        ("rovibrational", default_rovibrational_dos_output(target)),
    )
    rows = []
    for label, path in files:
        bins, emin, bin_cm1 = read_dos_binned(path)
        rows.append(
            (
                label,
                str(path),
                "yes" if path.exists() else "no",
                str(len(bins)),
                _format_value(emin),
                _format_value(bin_cm1),
            )
        )
    q_path = default_rovib_q_output(target)
    rows.append(("Q(T)", str(q_path), "yes" if q_path.exists() else "no", "", "", ""))
    return ThermoKineticsTable(
        "DOS",
        ("Kind", "File", "Exists", "Bins", "Emin cm-1", "Bin cm-1"),
        tuple(rows),
    )


def _kinetics_rows(target: Path) -> list[tuple[str, str, str]]:
    if not has_section(target, "KINETICS"):
        return [("KINETICS", "missing", "run matrix kinetics single")]
    section = read_kinetics_section(target)
    rows = [
        ("network", "ready", section.network_id),
        ("reaction", section.model or "present", section.active_reaction_id),
        ("channel", "ready", f"{section.reactant_id} -> {section.product_id}"),
        (
            "barrier E0 / cm-1",
            section.barrier_reference or "zero_point",
            _format_value(section.barrier_forward_cm1),
        ),
        ("k_TST / s-1", "computed", _format_value(section.k_tst_s1)),
    ]
    if section.rrkm_csv is not None:
        rows.append(("RRKM", "computed", str(section.rrkm_csv)))
    return rows


def _existing_path(path: Path) -> Path | None:
    return path if path.exists() else None


def _format_value(value: float | None) -> str:
    return "" if value is None else f"{float(value):.12g}"
