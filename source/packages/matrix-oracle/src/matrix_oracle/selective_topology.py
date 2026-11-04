"""Local invalidation domain for iterative ORACLE topology perception."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class SelectiveTopologyTracker:
    displacement_threshold_angstrom: float = 0.05
    neighbor_hops: int = 2
    _coordinates: np.ndarray | None = None

    def affected_atoms(self, coordinates_angstrom, bonds) -> tuple[int, ...]:
        """Return moved atoms and their frozen-graph closure; indices are zero based."""

        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3 or not np.isfinite(coordinates).all():
            raise ValueError("topology-update coordinates must have shape (natoms, 3)")
        if self.displacement_threshold_angstrom <= 0 or self.neighbor_hops < 0:
            raise ValueError("invalid selective-topology policy")
        if self._coordinates is None or self._coordinates.shape != coordinates.shape:
            moved = set(range(len(coordinates)))
            affected = set(moved)
            self._coordinates = coordinates.copy()
        else:
            displacement = np.linalg.norm(coordinates - self._coordinates, axis=1)
            moved = set(
                np.flatnonzero(
                    displacement >= self.displacement_threshold_angstrom
                ).tolist()
            )
            affected = set(moved)
        adjacency = [set() for _ in range(len(coordinates))]
        for left, right in bonds:
            left = int(left); right = int(right)
            adjacency[left].add(right); adjacency[right].add(left)
        frontier = set(affected)
        for _ in range(self.neighbor_hops):
            frontier = {neighbor for atom in frontier for neighbor in adjacency[atom]} - affected
            affected.update(frontier)
        # Keep an atom's last-perceived position as its baseline.  Updating the
        # complete frame here would let a sequence of individually small steps
        # evade the threshold forever.
        if self._coordinates is not None and moved:
            moved_indexes = np.fromiter(sorted(moved), dtype=int)
            self._coordinates[moved_indexes] = coordinates[moved_indexes]
        return tuple(sorted(affected))
