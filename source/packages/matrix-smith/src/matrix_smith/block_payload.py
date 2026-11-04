"""Shared payload helpers for typed ONIC block adapters.

The helpers in this module only reconcile atom numbering and sparse Cartesian
support between a frozen SMITH payload and a typed block.  They deliberately
contain no coordinate-generation or rank-selection policy.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .bmatrix import SparseBMatrix, SparseBRow
from .models import GICDefinition


def frozen_payload_reference_coordinates(
    definition: GICDefinition,
    *,
    payload_name: str,
) -> np.ndarray:
    coordinates = np.asarray(definition.reference_coordinates_angstrom, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or not np.all(np.isfinite(coordinates)):
        raise ValueError(f"{payload_name} payload has an invalid reference geometry")
    return coordinates


def normalized_owned_atoms(
    atoms: Sequence[int],
    *,
    block_name: str,
) -> tuple[int, ...]:
    normalized = tuple(int(atom) for atom in atoms)
    if not normalized or any(atom < 1 for atom in normalized):
        raise ValueError(f"{block_name} atoms must be positive and one based")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{block_name} contains duplicate atoms")
    return normalized


def payload_owned_atom_frame(
    block_atoms_one_based: Sequence[int],
    *,
    payload_natoms: int,
    payload_name: str,
    explicit_local_order: bool = False,
) -> tuple[tuple[int, ...], str]:
    """Return payload atom labels and whether the payload is local or global."""

    block_atoms = tuple(int(atom) for atom in block_atoms_one_based)
    if payload_natoms == len(block_atoms):
        local_labels = tuple(range(1, payload_natoms + 1))
        if (
            not explicit_local_order
            and set(block_atoms) == set(local_labels)
            and block_atoms != local_labels
        ):
            raise ValueError(
                f"{payload_name} all-atom payload uses an ambiguous permuted atom order"
            )
        return local_labels, "LOCAL"
    if not block_atoms or max(block_atoms) > payload_natoms:
        raise ValueError(
            f"{payload_name} payload is neither block-local nor a matching full-system chart"
        )
    return block_atoms, "FULL_SYSTEM"


def compact_sparse_b_matrix(
    matrix: SparseBMatrix,
    *,
    payload_atoms: Sequence[int],
) -> tuple[np.ndarray, float]:
    """Return owned Cartesian columns and maximum support outside ownership."""

    payload_atoms = tuple(int(atom) for atom in payload_atoms)
    owned_columns = tuple(3 * (atom - 1) + axis for atom in payload_atoms for axis in range(3))
    owned_set = set(owned_columns)
    lookup = {column: index for index, column in enumerate(owned_columns)}
    compact = np.zeros((matrix.row_count, len(owned_columns)), dtype=float)
    outside = 0.0
    for row_index, row in enumerate(matrix.rows):
        for column, value in row.entries:
            if column in owned_set:
                compact[row_index, lookup[column]] = value
            else:
                outside = max(outside, abs(float(value)))
    return compact, outside


def compact_sparse_b_matrix_sparse(
    matrix: SparseBMatrix,
    *,
    payload_atoms: Sequence[int],
) -> tuple[SparseBMatrix, float]:
    """Return owned Cartesian columns without dense block materialization."""

    payload_atoms = tuple(int(atom) for atom in payload_atoms)
    owned_columns = tuple(3 * (atom - 1) + axis for atom in payload_atoms for axis in range(3))
    owned_set = set(owned_columns)
    lookup = {column: index for index, column in enumerate(owned_columns)}
    rows = []
    outside = 0.0
    for row in matrix.rows:
        entries = []
        for column, value in row.entries:
            if column in owned_set:
                entries.append((lookup[column], value))
            else:
                outside = max(outside, abs(float(value)))
        rows.append(SparseBRow(size=len(owned_columns), entries=tuple(entries)))
    return (
        SparseBMatrix(
            rows=tuple(rows),
            column_count=len(owned_columns),
            row_labels=matrix.row_labels,
            backend=matrix.backend + ":compact-owned-columns",
        ),
        outside,
    )


def embed_local_sparse_rows(
    rows: tuple[SparseBRow, ...],
    block_atoms_one_based: Sequence[int],
    *,
    full_natoms: int,
    payload_name: str,
) -> tuple[SparseBRow, ...]:
    """Embed block-local sparse Wilson rows in the complete Cartesian space."""

    block_atoms = tuple(int(atom) for atom in block_atoms_one_based)
    output: list[SparseBRow] = []
    for row in rows:
        if row.size != 3 * len(block_atoms):
            raise ValueError(f"local {payload_name} B row has an inconsistent atom count")
        entries = tuple(
            (
                3 * (block_atoms[column // 3] - 1) + (column % 3),
                value,
            )
            for column, value in row.entries
        )
        output.append(SparseBRow(size=3 * full_natoms, entries=entries))
    return tuple(output)


def positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


__all__ = [
    "compact_sparse_b_matrix",
    "compact_sparse_b_matrix_sparse",
    "embed_local_sparse_rows",
    "frozen_payload_reference_coordinates",
    "normalized_owned_atoms",
    "payload_owned_atom_frame",
    "positive_finite",
]
