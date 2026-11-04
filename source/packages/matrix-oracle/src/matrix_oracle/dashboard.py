from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from matrix_core.cli import find_repo_root

from .commands import (
    OracleGuiCommand,
    avogadro_command,
    dvr_collect_command,
    fragments_command,
    gf_command,
    gicforge_bmatrix_command,
    gicforge_build_command,
    gicforge_report_command,
    rovib_summary_command,
    thermo_command,
    link_status_command,
    validate_command,
    vpt2_vci_collect_command,
)
from .guidance import missing_sections_message
from .project import OracleProjectState, load_oracle_project_state


CommandFactory = Callable[[Path], OracleGuiCommand]


@dataclass(frozen=True)
class DashboardActionTemplate:
    key: str
    window_key: str
    label: str
    factory: CommandFactory
    required_sections: tuple[str, ...] = ()


@dataclass(frozen=True)
class DashboardAction:
    key: str
    window_key: str
    label: str
    command: OracleGuiCommand
    enabled: bool
    reason: str
    required_sections: tuple[str, ...] = ()
    produced_sections: tuple[str, ...] = ()

    @property
    def shell_line(self) -> str:
        return self.command.shell_line()


@dataclass(frozen=True)
class DashboardRunResult:
    action: DashboardAction
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def default_dashboard_action_templates() -> tuple[DashboardActionTemplate, ...]:
    return (
        DashboardActionTemplate("validate", "dashboard", "Validate molecule", validate_command),
        DashboardActionTemplate("avogadro", "avogadro", "Open in Avogadro", avogadro_command),
        DashboardActionTemplate(
            "fragments_build",
            "topology",
            "Build fragments",
            lambda xyzin: fragments_command(xyzin, "build"),
            required_sections=("TOPOLOGY", "SYNTHONS"),
        ),
        DashboardActionTemplate(
            "gicforge_build",
            "gicforge",
            "Build GICs",
            gicforge_build_command,
            required_sections=("SYMMETRY", "TOPOLOGY", "SYNTHONS"),
        ),
        DashboardActionTemplate(
            "gicforge_report",
            "gicforge",
            "Write GIC report",
            lambda xyzin: gicforge_report_command(
                xyzin,
                xyzin.with_name(f"{xyzin.stem}.gicforge_report.txt"),
            ),
            required_sections=("GIC",),
        ),
        DashboardActionTemplate(
            "gicforge_bmatrix",
            "gicforge",
            "Evaluate B matrix",
            lambda xyzin: gicforge_bmatrix_command(
                xyzin,
                xyzin.with_name(f"{xyzin.stem}.gic_bmatrix.txt"),
            ),
            required_sections=("GIC",),
        ),
        DashboardActionTemplate(
            "gf_run",
            "gf",
            "Run TRINITY harmonic",
            lambda xyzin: gf_command(
                xyzin,
                out=xyzin.with_name(f"{xyzin.stem}.gf_ped_report.txt"),
                csv_dir=xyzin.with_name(f"{xyzin.stem}.gf_csv"),
            ),
            required_sections=("GIC", "CARTESIAN_HESSIAN"),
        ),
        DashboardActionTemplate(
            "rovib_summary",
            "rotational_spectroscopy",
            "Summarize rovib sections",
            rovib_summary_command,
            required_sections=("ROTATIONAL",),
        ),
        DashboardActionTemplate(
            "thermo",
            "thermochemistry_kinetics",
            "Run Thermo",
            lambda xyzin: thermo_command(xyzin, out=xyzin.with_name("thermo.report")),
            required_sections=("BASIC", "ROTATIONAL"),
        ),
        DashboardActionTemplate(
            "vpt2_vci_collect",
            "vibrational_spectroscopy",
            "Collect VPT2/VCI",
            vpt2_vci_collect_command,
            required_sections=("VPT2_VCI",),
        ),
        DashboardActionTemplate(
            "dvr_collect",
            "vibrational_spectroscopy",
            "Collect DVR",
            dvr_collect_command,
            required_sections=("DVR",),
        ),
        DashboardActionTemplate(
            "link_status",
            "link",
            "Summarize LINK",
            link_status_command,
            required_sections=("LINK",),
        ),
    )


class OracleDashboardController:
    def __init__(
        self,
        xyzin: Path | str | None = None,
        *,
        actions: Sequence[DashboardActionTemplate] | None = None,
        repo_root: Path | None = None,
    ) -> None:
        self.xyzin = None if xyzin is None else Path(xyzin)
        self.action_templates = tuple(actions or default_dashboard_action_templates())
        self.repo_root = Path(repo_root) if repo_root is not None else find_repo_root()
        self.log_lines: list[str] = []

    def set_xyzin(self, xyzin: Path | str | None) -> OracleProjectState | None:
        self.xyzin = None if xyzin is None else Path(xyzin)
        if self.xyzin is None:
            return None
        return self.state()

    def state(self) -> OracleProjectState:
        if self.xyzin is None:
            raise ValueError("no MATRIX xyzin project is loaded")
        return load_oracle_project_state(self.xyzin)

    def actions(self, state: OracleProjectState | None = None) -> tuple[DashboardAction, ...]:
        if self.xyzin is None:
            return ()
        project_state = self.state() if state is None else state
        present = set(project_state.section_names)
        result: list[DashboardAction] = []
        for template in self.action_templates:
            command = self._with_default_cwd(template.factory(project_state.xyzin))
            required = template.required_sections or command.required_sections
            missing = tuple(section for section in required if section not in present)
            enabled = project_state.exists and not missing
            reason = "ready" if enabled else _missing_reason(missing)
            result.append(
                DashboardAction(
                    key=template.key,
                    window_key=template.window_key,
                    label=template.label,
                    command=command,
                    enabled=enabled,
                    reason=reason,
                    required_sections=required,
                    produced_sections=command.produced_sections,
                )
            )
        return tuple(result)

    def action(self, key: str, state: OracleProjectState | None = None) -> DashboardAction:
        for action in self.actions(state):
            if action.key == key:
                return action
        raise KeyError(f"unknown dashboard action: {key}")

    def run_action(self, key: str, *, timeout: float | None = None) -> DashboardRunResult:
        action = self.action(key)
        if not action.enabled:
            raise ValueError(f"dashboard action {key!r} is not ready: {action.reason}")
        self.log(f"$ {action.shell_line}")
        completed = action.command.run(timeout=timeout)
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        if stdout:
            self.log(stdout.rstrip())
        if stderr:
            self.log(stderr.rstrip())
        self.log(f"[exit {completed.returncode}] {action.label}")
        return DashboardRunResult(
            action=action,
            returncode=int(completed.returncode),
            stdout=stdout,
            stderr=stderr,
        )

    def log(self, text: str) -> None:
        if not text:
            return
        self.log_lines.extend(str(text).splitlines())

    def clear_log(self) -> None:
        self.log_lines.clear()

    def prepare_command(self, command: OracleGuiCommand) -> OracleGuiCommand:
        return self._with_default_cwd(command)

    def _with_default_cwd(self, command: OracleGuiCommand) -> OracleGuiCommand:
        if command.cwd is not None:
            return command
        if len(command.argv) >= 3 and command.argv[1:3] in {("-m", "matrix"), ("-m", "oracle")}:
            return replace(command, cwd=self.repo_root)
        return command


def _missing_reason(missing_sections: tuple[str, ...]) -> str:
    return missing_sections_message(missing_sections)
