from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from matrix_core import ToolReadiness, tool_contract_readinesses, tool_contracts


@dataclass(frozen=True)
class ToolContractTable:
    title: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class ToolContractGuiState:
    xyzin: Path
    exists: bool
    ready_count: int
    total_count: int
    table: ToolContractTable
    messages: tuple[str, ...] = ()


def load_tool_contract_gui_state(
    path: Path | str,
    *,
    include_gui: bool = False,
) -> ToolContractGuiState:
    target = Path(path)
    contracts = tool_contracts(include_gui=include_gui)
    readinesses = tool_contract_readinesses(target, contracts)
    ready_count = sum(1 for readiness in readinesses if readiness.ready)
    messages = () if target.exists() else (f"Missing file: {target}",)
    return ToolContractGuiState(
        xyzin=target,
        exists=target.exists(),
        ready_count=ready_count,
        total_count=len(readinesses),
        table=_readiness_table(readinesses),
        messages=messages,
    )


def tool_contract_gui_state_lines(state: ToolContractGuiState) -> list[str]:
    return [
        f"xyzin: {state.xyzin}",
        f"exists: {int(state.exists)}",
        f"ready tools: {state.ready_count}/{state.total_count}",
        *state.messages,
    ]


def _readiness_table(readinesses: tuple[ToolReadiness, ...]) -> ToolContractTable:
    rows = []
    for readiness in readinesses:
        contract = readiness.contract
        rows.append(
            (
                contract.key,
                contract.display_name,
                contract.planned_name,
                "yes" if readiness.ready else "no",
                _join(contract.required_sections),
                _join(readiness.missing_required_sections),
                _join(contract.produced_sections),
                contract.standalone_command,
            )
        )
    return ToolContractTable(
        "Tool Contracts",
        (
            "Key",
            "Current",
            "Future",
            "Ready",
            "Required",
            "Missing",
            "Produced",
            "Command",
        ),
        tuple(rows),
    )


def _join(values: tuple[str, ...]) -> str:
    return ", ".join(values) if values else "none"
