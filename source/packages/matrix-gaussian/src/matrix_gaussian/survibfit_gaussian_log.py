from __future__ import annotations

from pathlib import Path
import re
import numpy as np


def _parse_float(token: str):
    try:
        return float(token.replace("D", "E").replace("d", "e"))
    except Exception:
        return None


def read_standard_orientation(log_path: Path):
    """Parse the last Standard/Input orientation block from a Gaussian log."""
    lines = Path(log_path).read_text(errors="ignore").splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if "Standard orientation:" in line or "Input orientation:" in line:
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("----"):
                i += 1
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("----"):
                i += 1
            i += 1

            geom = []
            while i < len(lines):
                l = lines[i].strip()
                if not l or l.startswith("----"):
                    break
                fields = l.split()
                if len(fields) < 6:
                    break
                atomic_number = int(fields[1])
                x, y, z = map(float, fields[3:6])
                geom.append((atomic_number, x, y, z))
                i += 1
            if geom:
                blocks.append(geom)
        i += 1

    if not blocks:
        raise ValueError("No orientation block found in Gaussian log")
    geom = blocks[-1]
    Z = np.array([g[0] for g in geom], dtype=int)
    coords = np.array([[g[1], g[2], g[3]] for g in geom], dtype=float)
    return Z, coords


def read_cartesian_force_constants(log_path: Path):
    """Parse Cartesian force constants (full symmetric Hessian) from Gaussian log."""
    lines = Path(log_path).read_text(errors="ignore").splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if "Force constants in Cartesian coordinates" in line:
            header_idx = i
    if header_idx is None:
        raise ValueError("Cartesian force constants block not found")

    i = header_idx + 1
    # Skip blank lines
    while i < len(lines) and not lines[i].strip():
        i += 1

    cols = []
    entries = []
    max_idx = 0
    stop_markers = (
        "Force constants in internal coordinates",
        "FormGI",
        "Leave Link",
    )

    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        if any(m in s for m in stop_markers):
            break

        # header line with column indices
        if not re.match(r"^\d", s):
            cols = [int(x) for x in re.findall(r"\d+", s)]
            if cols:
                max_idx = max(max_idx, max(cols))
            i += 1
            continue

        parts = s.split()
        if not parts[0].isdigit():
            break
        row = int(parts[0])
        max_idx = max(max_idx, row)
        vals = parts[1:]
        for k, tok in enumerate(vals):
            if k >= len(cols):
                break
            col = cols[k]
            val = _parse_float(tok)
            if val is None:
                continue
            entries.append((row - 1, col - 1, val))
        i += 1

    if max_idx == 0:
        raise ValueError("Failed to parse force-constants block")

    H = np.zeros((max_idx, max_idx), dtype=float)
    for i, j, val in entries:
        H[i, j] = val
        H[j, i] = val
    return H
