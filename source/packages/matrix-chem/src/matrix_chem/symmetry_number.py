"""Rotational symmetry numbers used by spectroscopy and thermochemistry."""

from __future__ import annotations

import re


def rotational_symmetry_number(point_group: str) -> int:
    pg = (point_group or "").strip().upper().replace("∞", "INF")
    if not pg:
        raise ValueError("point group is required to determine the rotational symmetry number")
    if pg in {"CINFV", "C*V"}:
        return 1
    if pg in {"DINFH", "D*H"}:
        return 2
    if pg in {"C1", "CS", "CI"}:
        return 1
    if pg in {"T", "TD", "TH"}:
        return 12
    if pg in {"O", "OH"}:
        return 24
    if pg in {"I", "IH"}:
        return 60
    match = re.match(r"D(\d+)", pg)
    if match:
        return 2 * int(match.group(1))
    match = re.match(r"C(\d+)", pg)
    if match:
        return int(match.group(1))
    match = re.match(r"S(\d+)", pg)
    if match:
        order = int(match.group(1))
        if order % 2:
            raise ValueError(f"invalid improper-rotation point group: {point_group}")
        return order // 2
    raise ValueError(f"unsupported point group for rotational symmetry number: {point_group}")
