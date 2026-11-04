from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from matrix_dvr import DVROutputSnapshot, collect_dvr_outputs_from_xyzin, refresh_dvr_section


@dataclass(frozen=True)
class DVRGuiState:
    xyzin: Path
    snapshot: DVROutputSnapshot

    @property
    def ready(self) -> bool:
        return bool(self.snapshot.levels)

    @property
    def status(self) -> str:
        return self.snapshot.status

    @property
    def level_count(self) -> int:
        return len(self.snapshot.levels)

    @property
    def grid_point_count(self) -> int:
        return len(self.snapshot.grid)


def load_dvr_gui_state(path: Path | str, *, refresh: bool = False) -> DVRGuiState:
    target = Path(path)
    snapshot = refresh_dvr_section(target) if refresh else collect_dvr_outputs_from_xyzin(target)
    return DVRGuiState(xyzin=target, snapshot=snapshot)


def dvr_gui_state_lines(state: DVRGuiState) -> list[str]:
    lines = [
        f"xyzin: {state.xyzin}",
        f"status: {state.status}",
        f"ready: {int(state.ready)}",
        f"levels: {state.level_count}",
        f"grid_points: {state.grid_point_count}",
        f"outputs: {len(state.snapshot.outputs)}",
    ]
    ground = state.snapshot.ground_cm
    if ground is not None:
        lines.append(f"ground_cm-1: {ground:.10g}")
    if state.snapshot.missing_outputs:
        lines.append("missing_primary: " + ", ".join(state.snapshot.missing_outputs))
    return lines
