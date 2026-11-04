from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable
import re


BOND_ORDER_WRITE_THRESHOLD = 0.2


@dataclass(frozen=True)
class GaussianTopologyData:
    """Gaussian-derived topology annotations imported before ORACLE topology."""

    cm5_charges: dict[int, float]
    mayer_bond_orders: dict[tuple[int, int], float]

    @property
    def has_data(self) -> bool:
        return bool(self.cm5_charges or self.mayer_bond_orders)


@dataclass(frozen=True)
class GaussianPopulationData:
    """Complete Gaussian Hirshfeld/CM5/Mayer observables for APOC."""

    hirshfeld_charges: dict[int, float]
    cm5_charges: dict[int, float]
    mayer_bond_orders: dict[tuple[int, int], float]


def read_gaussian_topology(path: Path) -> GaussianTopologyData:
    """Parse CM5 charges and Mayer bond orders from a Gaussian log/out file."""
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    return parse_gaussian_topology(lines)


def parse_gaussian_topology(lines: Iterable[str]) -> GaussianTopologyData:
    """Parse the Gaussian data ORACLE accepts as topology overrides.

    Atom indices are one-based, matching Gaussian and the enriched XYZ section.
    """
    source_lines = list(lines)
    population = parse_gaussian_population(source_lines)
    mayer = {
        pair: value
        for pair, value in population.mayer_bond_orders.items()
        if value >= BOND_ORDER_WRITE_THRESHOLD
    }
    return GaussianTopologyData(
        cm5_charges=population.cm5_charges,
        mayer_bond_orders=mayer,
    )


def read_gaussian_population(path: Path) -> GaussianPopulationData:
    """Read the complete Gaussian observables consumed by APOC."""

    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    return parse_gaussian_population(lines)


def parse_gaussian_population(lines: Iterable[str]) -> GaussianPopulationData:
    source_lines = list(lines)
    hirshfeld, cm5 = _parse_hirshfeld_and_cm5_charges(source_lines)
    return GaussianPopulationData(
        hirshfeld_charges=hirshfeld,
        cm5_charges=cm5,
        mayer_bond_orders=_parse_mayer_bond_orders(source_lines),
    )


def gaussian_topology_section_lines(path: Path) -> list[str] | None:
    """Return #GAUSSIAN_TOPOLOGY section lines for a Gaussian log/out file."""
    data = read_gaussian_topology(path)
    if not data.has_data:
        return None
    lines = [
        "SCHEMA matrix.xyz.gaussian_topology.v1",
        "INDEXING ATOMS=ONE_BASED",
    ]
    if data.cm5_charges:
        lines.append(f"CM5_COUNT = {len(data.cm5_charges)}")
        for idx in sorted(data.cm5_charges):
            lines.append(f"CM5 {idx} {data.cm5_charges[idx]: .10f}")
    if data.mayer_bond_orders:
        lines.append("BO_SOURCE = Mayer")
        lines.append(f"BO_COUNT = {len(data.mayer_bond_orders)}")
        for (i, j), value in sorted(data.mayer_bond_orders.items()):
            lines.append(f"BO {i} {j} {value: .10f}")
    return lines


def _parse_hirshfeld_and_cm5_charges(
    lines: list[str],
) -> tuple[dict[int, float], dict[int, float]]:
    """Return one-based Hirshfeld and CM5 charges from a Gaussian log."""
    header_indices = [
        idx
        for idx, line in enumerate(lines)
        if "hirshfeld charges" in line.lower() and "cm5 charges" in line.lower()
    ]
    if not header_indices:
        return {}, {}

    row_pat = re.compile(r"^\s*(\d+)\s+([A-Za-z]{1,2})\b")
    num_pat = re.compile(r"[+-]?\d*\.?\d+(?:[DEde][+-]?\d+)?")
    last_complete: tuple[dict[int, float], dict[int, float]] = ({}, {})
    for header_idx in header_indices:
        hirshfeld: dict[int, float] = {}
        cm5: dict[int, float] = {}
        started = False
        for raw in lines[header_idx + 1 : min(header_idx + 180, len(lines))]:
            text = raw.strip()
            if not text:
                if started:
                    break
                continue
            if text.startswith(("----", "Tot", "Sum")):
                if started:
                    break
                continue
            match = row_pat.match(text)
            if match is None:
                if started and text.lower().startswith(("hirshfeld", "mulliken", "natural")):
                    break
                continue
            nums = num_pat.findall(text[match.end() :])
            if not nums:
                continue
            q_h = _gaussian_float(nums[0])
            q_cm5 = _gaussian_float(nums[-1])
            if q_h is None or q_cm5 is None:
                continue
            hirshfeld[int(match.group(1))] = q_h
            cm5[int(match.group(1))] = q_cm5
            started = True
        if cm5:
            last_complete = hirshfeld, cm5
    return last_complete


def _parse_mayer_bond_orders(lines: list[str]) -> dict[tuple[int, int], float]:
    """Return one-based Mayer bond orders from pair or matrix Gaussian blocks."""
    header_idx = None
    for idx, line in enumerate(lines):
        lower = line.lower()
        if "mayer bond orders" in lower or "mayer atomic bond orders" in lower:
            header_idx = idx
    if header_idx is None:
        return {}

    out: dict[tuple[int, int], float] = {}
    num_pat = re.compile(r"[+-]?\d*\.?\d+(?:[DEde][+-]?\d+)?")
    cols_pat = re.compile(r"^\s*(\d+(?:\s+\d+)*)\s*$")
    row_pat = re.compile(r"^\s*(\d+)\s+[A-Za-z]{1,3}\s+(.+?)\s*$")
    current_cols: list[int] | None = None
    parsed_matrix = False
    for raw in lines[header_idx + 1 : min(header_idx + 260, len(lines))]:
        if not raw.strip():
            continue
        lower = raw.lower().strip()
        if "total bond order" in lower and out:
            break
        if lower.startswith(("wiberg", "natural", "mulliken", "hirshfeld")) and out:
            break

        for i, j, value in _parse_bond_order_pairs_from_line(raw):
            key = (i, j) if i < j else (j, i)
            out[key] = value

        match_cols = cols_pat.match(raw)
        if match_cols is not None:
            current_cols = [int(token) for token in match_cols.group(1).split()]
            parsed_matrix = True
            continue

        match_row = row_pat.match(raw)
        if match_row is not None and current_cols:
            i = int(match_row.group(1))
            nums = num_pat.findall(match_row.group(2))
            for j, token in zip(current_cols, nums):
                if i == j:
                    continue
                value = _gaussian_float(token)
                if value is None:
                    continue
                key = (i, j) if i < j else (j, i)
                out[key] = value
            continue

        if parsed_matrix and out and not lower.startswith("----"):
            break
    return out


def _parse_bond_order_pairs_from_line(line: str) -> list[tuple[int, int, float]]:
    patterns = [
        re.compile(
            r"B\(\s*(\d+)\s*-[A-Za-z]{1,2}\s*,\s*(\d+)\s*-[A-Za-z]{1,2}\s*\)\s*=\s*([+-]?\d*\.?\d+(?:[DEde][+-]?\d+)?)"
        ),
        re.compile(r"B\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*=\s*([+-]?\d*\.?\d+(?:[DEde][+-]?\d+)?)"),
    ]
    out: list[tuple[int, int, float]] = []
    for pattern in patterns:
        for match in pattern.finditer(line):
            value = _gaussian_float(match.group(3))
            if value is not None:
                out.append((int(match.group(1)), int(match.group(2)), value))
    return out


def _gaussian_float(token: str) -> float | None:
    try:
        return float(token.replace("D", "E").replace("d", "e"))
    except ValueError:
        return None
