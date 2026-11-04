from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matrix_vpt2_vci import VPT2VCIOutputSnapshot


@dataclass(frozen=True)
class VPT2VCIGuiState:
    xyzin: Path
    snapshot: VPT2VCIOutputSnapshot

    @property
    def ready(self) -> bool:
        return bool(self.snapshot.comparison)

    @property
    def status(self) -> str:
        return self.snapshot.status

    @property
    def root_count(self) -> int:
        return len(self.snapshot.comparison)

    @property
    def mode_count(self) -> int:
        return len(self.snapshot.frequencies)


def load_vpt2_vci_gui_state(path: Path | str, *, refresh: bool = False) -> VPT2VCIGuiState:
    from matrix_vpt2_vci import (
        collect_vpt2_vci_outputs_from_xyzin,
        refresh_vpt2_vci_section,
    )

    target = Path(path)
    snapshot = (
        refresh_vpt2_vci_section(target) if refresh else collect_vpt2_vci_outputs_from_xyzin(target)
    )
    return VPT2VCIGuiState(xyzin=target, snapshot=snapshot)


def vpt2_vci_gui_state_lines(state: VPT2VCIGuiState) -> list[str]:
    lines = [
        f"xyzin: {state.xyzin}",
        f"status: {state.status}",
        f"ready: {int(state.ready)}",
        f"roots: {state.root_count}",
        f"modes: {state.mode_count}",
        f"outputs: {len(state.snapshot.outputs)}",
    ]
    if state.snapshot.missing_outputs:
        lines.append("missing_primary: " + ", ".join(state.snapshot.missing_outputs))
    return lines
