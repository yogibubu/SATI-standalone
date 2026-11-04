from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from matrix_core import BasicSection, has_section, read_basic_section

from .contracts import (
    RotationalSection,
    VibrationalSection,
    read_rotational_section,
    read_vibrational_section,
)


@dataclass(frozen=True)
class RovibSummary:
    path: Path
    basic: BasicSection
    rotational: RotationalSection | None
    vibrational: VibrationalSection | None


def summarize_xyzin(path: Path) -> RovibSummary:
    target = Path(path)
    return RovibSummary(
        path=target,
        basic=read_basic_section(target),
        rotational=read_rotational_section(target) if has_section(target, "ROTATIONAL") else None,
        vibrational=(
            read_vibrational_section(target) if has_section(target, "VIBRATIONAL") else None
        ),
    )


def rovib_summary_lines(summary: RovibSummary) -> list[str]:
    lines = [
        f"xyzin: {summary.path}",
        f"charge: {summary.basic.charge}",
        f"multiplicity: {summary.basic.multiplicity}",
        f"point_group: {summary.basic.point_group}",
    ]
    if summary.rotational is None:
        lines.append("rotational: missing")
    else:
        rot = summary.rotational
        constants = [
            item
            for item in (
                _constant("A", rot.A_MHz),
                _constant("B", rot.B_MHz),
                _constant("C", rot.C_MHz),
            )
            if item
        ]
        lines.append(f"rotational: {' '.join(constants) if constants else 'present'}")
    if summary.vibrational is None:
        lines.append("vibrational: missing")
    else:
        lines.append(f"vibrational: {len(summary.vibrational.frequencies_cm1)} frequencies")
    return lines


def _constant(label: str, value: float | None) -> str:
    return "" if value is None else f"{label}={value:.8g}MHz"
