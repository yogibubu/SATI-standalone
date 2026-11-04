from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from matrix_core.sectioned_xyz import read_sectioned_lines, write_sectioned_lines, xyz_tail_start


@dataclass(frozen=True)
class XyzinGeometry:
    atoms: tuple[str, ...]
    coordinates_angstrom: np.ndarray
    comment: str = ""


def read_xyzin_geometry(path: Path) -> XyzinGeometry:
    target = Path(path)
    lines = read_sectioned_lines(target)
    if len(lines) < 2:
        raise ValueError(f"xyzin has no XYZ header: {target}")
    try:
        natoms = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"xyzin first line must be the atom count: {target}") from exc
    if len(lines) < natoms + 2:
        raise ValueError(f"xyzin XYZ block is incomplete: {target}")
    atoms: list[str] = []
    coords: list[tuple[float, float, float]] = []
    for raw in lines[2 : 2 + natoms]:
        parts = raw.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid xyzin coordinate line: {raw}")
        atoms.append(parts[0])
        coords.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return XyzinGeometry(tuple(atoms), np.asarray(coords, dtype=float), lines[1])


def replace_xyzin_geometry(path: Path, atoms, coords_angstrom, *, comment: str = "") -> Path:
    target = Path(path)
    lines = read_sectioned_lines(target)
    tail = lines[xyz_tail_start(lines) :]
    xyz_lines = _xyz_lines(tuple(atoms), np.asarray(coords_angstrom, dtype=float), comment=comment)
    write_sectioned_lines(target, [*xyz_lines, *tail])
    return target


def _xyz_lines(atoms: tuple[str, ...], coords: np.ndarray, *, comment: str) -> list[str]:
    if coords.shape != (len(atoms), 3):
        raise ValueError("xyzin geometry coordinates must have shape natoms x 3")
    lines = [str(len(atoms)), comment]
    for atom, xyz in zip(atoms, coords):
        lines.append(f"{atom:2s} {xyz[0]:19.12f} {xyz[1]:19.12f} {xyz[2]:19.12f}")
    return lines
