from __future__ import annotations

"""Gaussian adapter for reduced-dimensional anharmonic mode selections."""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

import numpy as np

from matrix_core import ParseError, ScientificValidationError


@dataclass(frozen=True)
class GaussianFundamental:
    internal_mode_index: int
    printed_mode_index: int
    status: str
    harmonic_cm: float
    fundamental_cm: float


def gaussian_mode_selection_card(mode_indices: Sequence[int], mode_count: int) -> str:
    """Return a deterministic ``Modes=`` card using Gaussian internal order."""
    count = int(mode_count)
    selected = sorted(set(int(value) for value in mode_indices))
    if count < 1:
        raise ValueError("mode_count must be positive")
    if not selected:
        raise ValueError("at least one Gaussian anharmonic mode must be active")
    if selected[0] < 1 or selected[-1] > count:
        raise ValueError("Gaussian mode index is outside 1..mode_count")
    ranges: list[str] = []
    start = previous = selected[0]
    for value in selected[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return "Modes=" + ",".join(ranges)


def write_gaussian_select_anharmonic_input(
    path: Path | str,
    *,
    checkpoint: str,
    route: str,
    active_mode_indices: Sequence[int],
    mode_count: int,
    processors: int | None = None,
    memory: str | None = None,
) -> Path:
    """Write a checkpoint-based selective anharmonic Gaussian input.

    ``route`` supplies the electronic model and state request.  Geometry and
    orbitals are read from ``checkpoint``; the adapter owns the anharmonic
    selection keywords and refuses a conflicting route.
    """
    route_text = " ".join(str(route).strip().split())
    lowered = route_text.lower()
    if not route_text:
        raise ValueError("route cannot be empty")
    conflicts = tuple(
        keyword
        for keyword in ("freq", "geom", "guess", "selectanharmonicmodes")
        if re.search(rf"\b{keyword}\b", lowered)
    )
    if conflicts:
        raise ValueError(
            "route must not contain adapter-owned Freq, Geom, Guess or "
            f"SelectAnharmonicModes keywords (found {', '.join(conflicts)})"
        )
    if not route_text.startswith("#"):
        route_text = "#p " + route_text
    card = gaussian_mode_selection_card(active_mode_indices, mode_count)
    lines = [f"%chk={checkpoint}"]
    if processors is not None:
        if processors < 1:
            raise ValueError("processors must be positive")
        lines.append(f"%NProcShared={int(processors)}")
    if memory is not None:
        lines.append(f"%Mem={memory}")
    lines.extend(
        [
            route_text
            + " Geom=AllCheck Guess=TCheck Freq=(Anharmonic,SelectAnharmonicModes)",
            "",
            card,
            "",
        ]
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def read_gaussian_fundamentals(
    path: Path | str,
    harmonic_frequencies_cm: Sequence[float],
    *,
    harmonic_tolerance_cm: float = 0.05,
) -> tuple[GaussianFundamental, ...]:
    """Read the first Gaussian anharmonic fundamental table and validate order."""
    harmonic = np.asarray(harmonic_frequencies_cm, dtype=float)
    if harmonic.ndim != 1 or harmonic.size < 1:
        raise ValueError("harmonic_frequencies_cm must be a non-empty vector")
    if harmonic_tolerance_cm <= 0.0:
        raise ValueError("harmonic_tolerance_cm must be positive")
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    starts = [
        index
        for index, line in enumerate(lines)
        if line.strip() == "Fundamental Bands"
        and index + 2 < len(lines)
        and "E(harm)" in lines[index + 2]
    ]
    if not starts:
        raise ParseError("Gaussian anharmonic fundamental table was not found")
    pattern = re.compile(
        r"^\s*(\d+)\(1\)\s+(\w+)\s+([-+\d.]+)\s+([-+\d.]+)"
    )
    rows: list[GaussianFundamental] = []
    for line in lines[starts[0] + 3 :]:
        if line.strip() == "Overtones":
            break
        match = pattern.match(line)
        if match is None:
            continue
        printed = int(match.group(1))
        internal = len(harmonic) + 1 - printed
        if internal < 1 or internal > len(harmonic):
            raise ParseError(f"Gaussian printed mode {printed} is outside the harmonic mode count")
        row = GaussianFundamental(
            internal_mode_index=internal,
            printed_mode_index=printed,
            status=match.group(2).lower(),
            harmonic_cm=float(match.group(3)),
            fundamental_cm=float(match.group(4)),
        )
        expected = float(harmonic[internal - 1])
        if abs(row.harmonic_cm - expected) > harmonic_tolerance_cm:
            raise ScientificValidationError(
                "Gaussian harmonic mode ordering is inconsistent with the reference: "
                f"internal mode {internal}, table {row.harmonic_cm:.6f}, "
                f"reference {expected:.6f} cm-1"
            )
        rows.append(row)
    if len(rows) != len(harmonic):
        raise ParseError(
            f"Gaussian fundamental table contains {len(rows)} modes; expected {len(harmonic)}"
        )
    if len({row.internal_mode_index for row in rows}) != len(rows):
        raise ParseError("Gaussian fundamental table contains duplicate mode indices")
    return tuple(sorted(rows, key=lambda row: row.internal_mode_index))
