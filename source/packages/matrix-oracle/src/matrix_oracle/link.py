from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matrix_link import LinkSection

from .commands import OracleGuiCommand, link_prepare_command


@dataclass(frozen=True)
class LinkTable:
    title: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class LinkSummary:
    status: str = ""
    run_dir: Path | None = None
    manifest_path: Path | None = None
    engine_command: str = ""
    external_protocol: str = ""
    coordinate_model: str = ""
    active_space: str = ""
    energy_unit: str = ""
    gradient_unit: str = ""
    max_steps: int = 0
    trust_radius: float = 0.0
    gradient_tolerance: float = 0.0
    step_tolerance: float = 0.0
    energy_tolerance: float = 0.0


@dataclass(frozen=True)
class LinkGuiState:
    xyzin: Path
    exists: bool
    ready: bool
    summary: LinkSummary
    settings: LinkTable
    outputs: LinkTable
    messages: tuple[str, ...] = ()


class OracleLinkController:
    def __init__(self, xyzin: Path | str | None = None) -> None:
        self.xyzin = None if xyzin is None else Path(xyzin)

    def set_xyzin(self, xyzin: Path | str | None) -> LinkGuiState | None:
        self.xyzin = None if xyzin is None else Path(xyzin)
        if self.xyzin is None:
            return None
        return self.state()

    def state(self) -> LinkGuiState:
        if self.xyzin is None:
            raise ValueError("no MATRIX xyzin project is loaded")
        return load_link_gui_state(self.xyzin)

    def prepare_command(
        self,
        *,
        run_dir: Path | str,
        engine_command: str,
        coordinate_model: str = "gic",
        active_space: str = "total_symmetric",
        max_steps: int = 50,
        trust_radius: float = 0.2,
        gradient_tolerance: float = 1.0e-5,
        step_tolerance: float = 1.0e-5,
        energy_tolerance: float = 1.0e-8,
    ) -> OracleGuiCommand:
        if self.xyzin is None:
            raise ValueError("no MATRIX xyzin project is loaded")
        return link_prepare_command(
            self.xyzin,
            run_dir=run_dir,
            engine_command=engine_command,
            coordinate_model=coordinate_model,
            active_space=active_space,
            max_steps=max_steps,
            trust_radius=trust_radius,
            gradient_tolerance=gradient_tolerance,
            step_tolerance=step_tolerance,
            energy_tolerance=energy_tolerance,
        )


def load_link_gui_state(path: Path | str) -> LinkGuiState:
    from matrix_link import read_link_section

    target = Path(path)
    empty = _empty_state(target, exists=target.exists())
    if not target.exists():
        return _replace_messages(empty, (f"Missing file: {target}",))
    try:
        section = read_link_section(target)
    except (OSError, ValueError) as exc:
        return _replace_messages(empty, (str(exc),))
    return LinkGuiState(
        xyzin=target,
        exists=True,
        ready=True,
        summary=_summary(section),
        settings=_settings_table(section),
        outputs=_outputs_table(section),
        messages=(),
    )


def link_gui_state_lines(state: LinkGuiState) -> list[str]:
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
        f"engine command: {summary.engine_command}",
        f"external protocol: {summary.external_protocol}",
        f"coordinate model: {summary.coordinate_model}",
        f"active space: {summary.active_space}",
        f"run dir: {_path_text(summary.run_dir)}",
        f"manifest: {_path_text(summary.manifest_path)}",
        f"max steps: {summary.max_steps}",
        f"trust radius: {_format_float(summary.trust_radius)}",
        f"gradient tolerance: {_format_float(summary.gradient_tolerance)}",
        f"step tolerance: {_format_float(summary.step_tolerance)}",
        f"energy tolerance: {_format_float(summary.energy_tolerance)}",
    ]


def default_link_run_dir(xyzin: Path | str) -> Path:
    target = Path(xyzin)
    return target.with_name(f"{target.stem}.link")


def _summary(section: LinkSection) -> LinkSummary:
    return LinkSummary(
        status=section.status,
        run_dir=section.run_dir,
        manifest_path=section.manifest_path,
        engine_command=section.engine_command,
        external_protocol=section.external_protocol,
        coordinate_model=section.coordinate_model,
        active_space=section.active_space,
        energy_unit=section.energy_unit,
        gradient_unit=section.gradient_unit,
        max_steps=section.max_steps,
        trust_radius=section.trust_radius,
        gradient_tolerance=section.gradient_tolerance,
        step_tolerance=section.step_tolerance,
        energy_tolerance=section.energy_tolerance,
    )


def _settings_table(section: LinkSection) -> LinkTable:
    rows = (
        ("Status", section.status),
        ("Engine command", section.engine_command),
        ("External protocol", section.external_protocol),
        ("Coordinate model", section.coordinate_model),
        ("Active space", section.active_space),
        ("Energy unit", section.energy_unit),
        ("Gradient unit", section.gradient_unit),
        ("Max steps", str(section.max_steps)),
        ("Trust radius", _format_float(section.trust_radius)),
        ("Gradient tolerance", _format_float(section.gradient_tolerance)),
        ("Step tolerance", _format_float(section.step_tolerance)),
        ("Energy tolerance", _format_float(section.energy_tolerance)),
    )
    return LinkTable("Settings", ("Field", "Value"), rows)


def _outputs_table(section: LinkSection) -> LinkTable:
    rows = (
        ("Run dir", _path_text(section.run_dir)),
        ("Manifest", _path_text(section.manifest_path)),
        ("Trajectory", _path_text(section.trajectory_path)),
        ("Final geometry", _path_text(section.final_geometry_path)),
        ("Energy/gradient log", _path_text(section.energy_gradient_log_path)),
    )
    extra = tuple((f"Output {name}", str(path)) for name, path in sorted(section.outputs.items()))
    return LinkTable("Outputs", ("Output", "Path"), (*rows, *extra))


def _empty_state(target: Path, *, exists: bool) -> LinkGuiState:
    return LinkGuiState(
        xyzin=target,
        exists=exists,
        ready=False,
        summary=LinkSummary(),
        settings=LinkTable("Settings", ("Field", "Value"), ()),
        outputs=LinkTable("Outputs", ("Output", "Path"), ()),
        messages=(),
    )


def _replace_messages(state: LinkGuiState, messages: tuple[str, ...]) -> LinkGuiState:
    return LinkGuiState(
        xyzin=state.xyzin,
        exists=state.exists,
        ready=state.ready,
        summary=state.summary,
        settings=state.settings,
        outputs=state.outputs,
        messages=messages,
    )


def _format_float(value: float) -> str:
    return f"{float(value):.8g}"


def _path_text(path: Path | None) -> str:
    return "" if path is None else str(path)
