"""Compatibility adapter from legacy primitive manifests to SMITH rows.

The manifest is only a serialization boundary. SMITH remains the owner of
the primitive definitions and every Wilson derivative used below.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .models import GICPrimitive
from .numerics import _analytic_b_row


def primitive_b_matrix_from_manifest(
    coordinates: np.ndarray, manifest: Mapping[str, Any]
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Materialize a serialized R/A/L/SOOP/D primitive pool through SMITH."""
    xyz = np.asarray(coordinates, dtype=float)
    records = manifest.get("coordinates")
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not isinstance(records, Mapping):
        raise ValueError("invalid geometry or primitive manifest")
    primitives: list[GICPrimitive] = []
    labels: list[str] = []

    def append(function: str, atoms: tuple[int, ...], label: str, **kwargs: Any) -> None:
        primitives.append(GICPrimitive(
            identifier=f"MANIFEST_{len(primitives) + 1:06d}",
            name=label, family=function, function=function, atoms=atoms, **kwargs,
        ))
        labels.append(label)

    for raw_i, raw_j in records.get("R", ()):
        atoms = (int(raw_i), int(raw_j))
        append("R", atoms, f"R({atoms[0]},{atoms[1]})")
    for raw_i, raw_j, raw_k in records.get("A", ()):
        atoms = (int(raw_i), int(raw_j), int(raw_k))
        append("A", atoms, f"A({atoms[0]},{atoms[1]},{atoms[2]})")
    for record in records.get("L", ()):
        raw_atoms = record["atoms"] if isinstance(record, Mapping) else record
        atoms = tuple(int(value) for value in raw_atoms)
        for mode, axis in zip((-1, 1), ("x", "y"), strict=True):
            append("L", atoms, f"L{axis}({','.join(map(str, atoms))})", mode=mode)
    for record in records.get("SOOP", ()):
        center, moving = int(record["center"]), int(record["moving"])
        plane = tuple(int(value) for value in record["plane"])
        others = tuple(atom for atom in plane if atom != center)
        if len(others) != 2:
            raise ValueError("SOOP reference plane must contain center and two supports")
        normal = np.cross(
            xyz[others[0] - 1] - xyz[center - 1],
            xyz[others[1] - 1] - xyz[center - 1],
        )
        magnitude = float(np.linalg.norm(normal))
        if magnitude <= 0.0:
            raise ValueError("singular SOOP reference plane")
        normal /= magnitude
        row = np.zeros(xyz.size)
        start = 3 * (moving - 1)
        row[start:start + 3] = normal
        refs = (f"OFFSET:{float(row @ xyz.reshape(-1)):.17g}",) + tuple(
            f"B:{column}:{value:.17g}"
            for column, value in enumerate(row)
            if value != 0.0
        )
        label = f"SOOP({center};{moving}|{others[0]},{others[1]})"
        append(
            "SOOP", (center, moving, *others), label,
            refs=refs, chart="FROZEN_REFERENCE_CARTESIAN",
        )
    for raw_atoms in records.get("D", ()):
        atoms = tuple(int(value) for value in raw_atoms)
        append("D", atoms, f"D({','.join(map(str, atoms))})")
    if not primitives:
        raise ValueError("empty primitive manifest")
    rows = np.asarray([_analytic_b_row(primitive, xyz) for primitive in primitives])
    if not np.all(np.isfinite(rows)):
        raise ValueError("primitive manifest produced non-finite SMITH rows")
    return rows, tuple(labels)


__all__ = ["primitive_b_matrix_from_manifest"]
