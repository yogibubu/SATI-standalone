"""Continuous SO(3) charts for frozen SONIC fragment rotations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from matrix_chem import RotationChart

if TYPE_CHECKING:
    from .models import GICDefinition
    from .sparse_b_matrix import SparseBMatrix


@dataclass(frozen=True)
class FragmentRotationGroup:
    key: tuple[str, ...]
    coordinate_indices: tuple[int, int, int]


class FragmentRotationAtlas:
    """Transport FROT values and B rows across locally rebased SO(3) charts."""

    def __init__(self, definition: "GICDefinition") -> None:
        primitive_by_id = {item.identifier: item for item in definition.primitives}
        grouped: dict[tuple[str, ...], dict[int, int]] = {}
        for index, gic in enumerate(definition.gics):
            coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
            if len(coefficients) != 1 or float(coefficients[0][1]) != 1.0:
                continue
            primitive = primitive_by_id.get(coefficients[0][0])
            if primitive is None or primitive.function != "FROT":
                continue
            key = (
                *primitive.refs,
                *(str(item) for item in primitive.frame_atoms),
                "|",
                *(str(item) for item in primitive.ref_frame_atoms),
            )
            grouped.setdefault(key, {})[int(primitive.mode)] = index
        self.groups = tuple(
            FragmentRotationGroup(key, (modes[0], modes[1], modes[2]))
            for key, modes in grouped.items()
            if set(modes) == {0, 1, 2}
        )
        self.reference_coordinates = np.asarray(
            definition.reference_coordinates_angstrom, dtype=float
        ).copy()
        self._charts = {group.key: RotationChart.identity() for group in self.groups}

    @property
    def active(self) -> bool:
        return bool(self.groups)

    def transform(
        self,
        local_values: np.ndarray,
        local_rows: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        values = np.asarray(local_values, dtype=float).copy()
        rows = None if local_rows is None else np.asarray(local_rows, dtype=float).copy()
        for group in self.groups:
            indices = list(group.coordinate_indices)
            chart = self._charts[group.key]
            values[indices] = chart.value(values[indices])
            if rows is not None:
                rows[indices, :] = chart.rows(rows[indices, :])
        return values, rows

    def to_local_values(self, chart_values: np.ndarray) -> np.ndarray:
        """Return values in the current local SO(3) charts.

        Non-rotational coordinates are unchanged.  This is the inverse of
        :meth:`transform` for coordinate values and is the canonical bridge
        between LINK's continuous optimizer coordinates and a rigid-pose
        realization based at ``reference_coordinates``.
        """

        values = np.asarray(chart_values, dtype=float).copy()
        for group in self.groups:
            indices = list(group.coordinate_indices)
            chart = self._charts[group.key]
            values[indices] = np.linalg.solve(
                chart.tangent,
                values[indices] - chart.offset,
            )
        return values

    def cartesian_columns_from_local(self, local_columns: np.ndarray) -> np.ndarray:
        """Convert ``dx/d(local FROT)`` columns to ``dx/d(chart FROT)``.

        A rebased rigid-pose model differentiates with respect to its local
        exponential map.  LINK stores gradients and Hessians in the continuous
        chart coordinates, so the inverse chart tangent must be applied to the
        corresponding Cartesian-Jacobian columns.
        """

        columns = np.asarray(local_columns, dtype=float).copy()
        if columns.ndim != 2:
            raise ValueError("local Cartesian columns must be a matrix")
        for group in self.groups:
            indices = list(group.coordinate_indices)
            if max(indices) >= columns.shape[1]:
                raise ValueError("rotation coordinate index exceeds Cartesian-column matrix")
            chart = self._charts[group.key]
            columns[:, indices] = columns[:, indices] @ np.linalg.inv(chart.tangent)
        return columns

    def max_local_norm(self, local_values: np.ndarray) -> float:
        values = np.asarray(local_values, dtype=float)
        return max(
            (
                float(np.linalg.norm(values[list(group.coordinate_indices)]))
                for group in self.groups
            ),
            default=0.0,
        )

    def transform_subset(
        self,
        coordinate_indices: tuple[int, ...],
        local_values: np.ndarray,
        local_rows: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Transform complete FROT triplets contained in a coordinate subset."""

        indices = tuple(int(index) for index in coordinate_indices)
        positions = {index: position for position, index in enumerate(indices)}
        values = np.asarray(local_values, dtype=float).reshape(len(indices)).copy()
        rows = None if local_rows is None else np.asarray(local_rows, dtype=float).copy()
        for group in self.groups:
            if not all(index in positions for index in group.coordinate_indices):
                continue
            local_positions = [positions[index] for index in group.coordinate_indices]
            chart = self._charts[group.key]
            values[local_positions] = chart.value(values[local_positions])
            if rows is not None:
                rows[local_positions, :] = chart.rows(rows[local_positions, :])
        return values, rows

    def transform_sparse_subset(
        self,
        coordinate_indices: tuple[int, ...],
        local_values: np.ndarray,
        local_rows: "SparseBMatrix",
    ) -> tuple[np.ndarray, "SparseBMatrix"]:
        """Apply complete local FROT chart blocks without densifying ``B``.

        Each chart tangent only mixes the three rows belonging to one
        fragment rotation.  Sparse row combinations therefore preserve the
        union of their local supports and the operation remains linear in the
        stored nonzeros.
        """

        from .bmatrix import SparseBMatrix, SparseBRow

        indices = tuple(int(index) for index in coordinate_indices)
        if len(indices) != local_rows.row_count:
            raise ValueError("coordinate subset and sparse row count differ")
        positions = {index: position for position, index in enumerate(indices)}
        values = np.asarray(local_values, dtype=float).reshape(len(indices)).copy()
        rows = list(local_rows.rows)
        for group in self.groups:
            if not all(index in positions for index in group.coordinate_indices):
                continue
            local_positions = [positions[index] for index in group.coordinate_indices]
            chart = self._charts[group.key]
            values[local_positions] = chart.value(values[local_positions])
            source_rows = tuple(rows[position] for position in local_positions)
            transformed_rows = tuple(
                SparseBRow.combine(
                    *tuple(
                        (float(chart.tangent[row_index, column_index]), source_rows[column_index])
                        for column_index in range(3)
                    )
                )
                for row_index in range(3)
            )
            for position, row in zip(local_positions, transformed_rows, strict=True):
                rows[position] = row
        return values, SparseBMatrix(
            rows=tuple(rows),
            column_count=local_rows.column_count,
            row_labels=local_rows.row_labels,
            backend=f"{local_rows.backend}+frot-atlas",
        )

    def rebase(self, local_values: np.ndarray, coordinates_angstrom: np.ndarray) -> None:
        values = np.asarray(local_values, dtype=float)
        for group in self.groups:
            indices = list(group.coordinate_indices)
            self._charts[group.key] = self._charts[group.key].rebase(values[indices])
        self.reference_coordinates = np.asarray(coordinates_angstrom, dtype=float).copy()
