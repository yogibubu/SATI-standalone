from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from matrix_core import key_value_section_lines, replace_section


ORACLE_XYZ_ROTATIONAL_SPECTRUM_SCHEMA = "oracle.xyz.rotational_spectrum.v1"


@dataclass(frozen=True)
class RotationalTransition:
    frequency_MHz: float
    intensity: float
    relative_intensity: float
    lower: str = ""
    upper: str = ""
    branch: str = ""


@dataclass(frozen=True)
class RotationalSpectrumResult:
    transitions: tuple[RotationalTransition, ...]
    csv_path: Path
    plot_path: Path | None
    table_path: Path | None
    j_min: int
    j_max: int
    temperature_K: float | None
    constant_kind: str
    constant_source: str
    backend: str = "matrix-rovib.lock.native_rotor"


def transitions_from_wmsrot(result: Any) -> tuple[RotationalTransition, ...]:
    if hasattr(result, "to_dict"):
        rows = result.to_dict(orient="records")
    else:
        rows = list(result)
    transitions = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        transitions.append(
            RotationalTransition(
                frequency_MHz=float(row.get("Frequency (MHz)", row.get("frequency_MHz", 0.0))),
                intensity=float(row.get("Intensity", row.get("intensity", 0.0))),
                relative_intensity=float(
                    row.get("Relative intensity", row.get("relative_intensity", 0.0))
                ),
                lower=_assignment(row, "l"),
                upper=_assignment(row, "u"),
                branch=str(row.get("Branch", row.get("branch", ""))),
            )
        )
    return tuple(sorted(transitions, key=lambda item: item.frequency_MHz))


def write_rotational_spectrum_plot(
    path: Path | str,
    transitions: Iterable[RotationalTransition],
    *,
    fwhm_MHz: float = 0.0,
    step_MHz: float | None = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = tuple(transitions)
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    if lines and fwhm_MHz > 0.0:
        step = float(step_MHz or max(fwhm_MHz / 12.0, 1.0e-6))
        xmin = min(item.frequency_MHz for item in lines) - 4.0 * fwhm_MHz
        xmax = max(item.frequency_MHz for item in lines) + 4.0 * fwhm_MHz
        x = np.arange(xmin, xmax + step, step)
        sigma = fwhm_MHz / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        y = np.zeros_like(x)
        for item in lines:
            y += item.relative_intensity * np.exp(-0.5 * ((x - item.frequency_MHz) / sigma) ** 2)
        ax.plot(x, y, color="#1f4e79", linewidth=1.2)
    else:
        for item in lines:
            ax.vlines(item.frequency_MHz, 0.0, item.relative_intensity, color="#1f4e79", linewidth=0.8)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Relative intensity")
    ax.set_title("Rotational spectrum")
    ax.set_ylim(bottom=0.0)
    fig.tight_layout()
    fig.savefig(target)
    plt.close(fig)
    return target


def write_rotational_spectrum_latex(
    path: Path | str, transitions: Iterable[RotationalTransition]
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        r"\begin{tabular}{rrrrl}",
        r"\hline",
        r"Frequency / MHz & Intensity & Relative & Branch & Assignment \\",
        r"\hline",
    ]
    for item in transitions:
        assignment = f"{item.lower} $\\rightarrow$ {item.upper}".strip()
        rows.append(
            f"{item.frequency_MHz:.6f} & {item.intensity:.6e} & "
            f"{item.relative_intensity:.6f} & {item.branch} & {assignment} \\\\"
        )
    rows.extend((r"\hline", r"\end{tabular}"))
    target.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return target


def write_rotational_spectrum_section(path: Path | str, result: RotationalSpectrumResult) -> None:
    lines = key_value_section_lines(
        ORACLE_XYZ_ROTATIONAL_SPECTRUM_SCHEMA,
        {
            "BACKEND": result.backend,
            "CSV_PATH": str(result.csv_path),
            "PLOT_PATH": None if result.plot_path is None else str(result.plot_path),
            "TABLE_PATH": None if result.table_path is None else str(result.table_path),
            "N_TRANSITIONS": len(result.transitions),
            "J_MIN": result.j_min,
            "J_MAX": result.j_max,
            "T_K": result.temperature_K,
            "CONSTANT_KIND": result.constant_kind or None,
            "CONSTANT_SOURCE": result.constant_source or None,
        },
        key_order=(
            "BACKEND",
            "CSV_PATH",
            "PLOT_PATH",
            "TABLE_PATH",
            "N_TRANSITIONS",
            "J_MIN",
            "J_MAX",
            "T_K",
            "CONSTANT_KIND",
            "CONSTANT_SOURCE",
        ),
    )
    replace_section(Path(path), "ROTATIONAL_SPECTRUM", lines)


def _assignment(row: dict[str, Any], suffix: str) -> str:
    fields = []
    for label in ("J", "Ka", "Kc", "F", "I12", "sp"):
        key = f"{label}{suffix}" if label == "J" else f"{label}_{suffix}"
        if key in row and row[key] not in {None, ""}:
            fields.append(f"{label}={row[key]}")
    return " ".join(fields)
