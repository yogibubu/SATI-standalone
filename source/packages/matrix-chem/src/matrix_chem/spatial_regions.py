"""Deterministic spatial regions for local molecular work.

The cell list is shared by ORACLE-owned perception primitives and by
downstream neighbor-list builders.  Construction is linear in atom count and
candidate enumeration is linear at bounded density for a fixed cutoff.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
import math
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class SpatialRegions:
    coordinates: np.ndarray
    cell_size: float
    cells: dict[tuple[int, int, int], tuple[int, ...]]

    @classmethod
    def build(cls, coordinates: np.ndarray, *, cell_size: float) -> "SpatialRegions":
        xyz = np.asarray(coordinates, dtype=float)
        width = float(cell_size)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("spatial-region coordinates must have shape (natoms, 3)")
        if not np.isfinite(width) or width <= 0.0:
            raise ValueError("spatial-region cell size must be finite and positive")
        if np.any(~np.isfinite(xyz)):
            raise ValueError("spatial-region coordinates must be finite")
        buckets: dict[tuple[int, int, int], list[int]] = {}
        indices = np.floor(xyz / width).astype(np.int64)
        for atom, raw_key in enumerate(indices):
            key = tuple(int(value) for value in raw_key)
            buckets.setdefault(key, []).append(atom)
        return cls(
            coordinates=xyz,
            cell_size=width,
            cells={key: tuple(values) for key, values in buckets.items()},
        )

    def candidate_pairs(self, cutoff: float) -> Iterator[tuple[int, int]]:
        """Yield every pair within ``cutoff`` once, in deterministic order."""

        radius = float(cutoff)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("spatial cutoff must be finite and positive")
        shell = int(math.ceil(radius / self.cell_size))
        offsets = tuple(product(range(-shell, shell + 1), repeat=3))
        cutoff2 = radius * radius
        for key in sorted(self.cells):
            atoms = self.cells[key]
            for offset in offsets:
                other_key = tuple(key[axis] + offset[axis] for axis in range(3))
                if other_key not in self.cells or other_key < key:
                    continue
                others = self.cells[other_key]
                for left in atoms:
                    for right in others:
                        if other_key == key and right <= left:
                            continue
                        delta = self.coordinates[left] - self.coordinates[right]
                        if float(np.dot(delta, delta)) <= cutoff2:
                            yield (left, right) if left < right else (right, left)


@dataclass
class PersistentNeighborList:
    """Deterministic Verlet list reused until half of its skin is consumed."""

    cutoff: float
    skin: float = 0.6
    _reference_coordinates: np.ndarray | None = field(default=None, init=False, repr=False)
    _pairs: tuple[tuple[int, int], ...] = field(default=(), init=False, repr=False)
    rebuild_count: int = field(default=0, init=False)
    reuse_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.cutoff = float(self.cutoff)
        self.skin = float(self.skin)
        if not np.isfinite(self.cutoff) or self.cutoff <= 0.0:
            raise ValueError("neighbor-list cutoff must be finite and positive")
        if not np.isfinite(self.skin) or self.skin <= 0.0:
            raise ValueError("neighbor-list skin must be finite and positive")

    def candidate_pairs(self, coordinates: np.ndarray) -> tuple[tuple[int, int], ...]:
        xyz = np.asarray(coordinates, dtype=float)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or np.any(~np.isfinite(xyz)):
            raise ValueError("neighbor-list coordinates must have shape (natoms, 3)")
        rebuild = self._reference_coordinates is None
        if not rebuild:
            if self._reference_coordinates.shape != xyz.shape:
                rebuild = True
            else:
                maximum_displacement = float(
                    np.max(np.linalg.norm(xyz - self._reference_coordinates, axis=1), initial=0.0)
                )
                rebuild = maximum_displacement >= 0.5 * self.skin
        if rebuild:
            regions = SpatialRegions.build(
                xyz,
                cell_size=max(2.0, self.cutoff + self.skin),
            )
            self._pairs = tuple(
                sorted(regions.candidate_pairs(self.cutoff + self.skin))
            )
            self._reference_coordinates = xyz.copy()
            self.rebuild_count += 1
        else:
            self.reuse_count += 1
        return self._pairs


def bounded_topological_distances(
    adjacency: list[set[int]] | tuple[set[int], ...] | dict[int, set[int]],
    *,
    maximum_distance: int = 3,
) -> dict[tuple[int, int], int]:
    """Return only graph distances that alter standard nonbonded scaling."""

    from collections import deque

    limit = int(maximum_distance)
    if limit < 1:
        return {}
    atoms = sorted(adjacency) if isinstance(adjacency, dict) else list(range(len(adjacency)))
    distances: dict[tuple[int, int], int] = {}
    for start in atoms:
        seen = {start: 0}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            depth = seen[current]
            if depth >= limit:
                continue
            neighbors = adjacency.get(current, set()) if isinstance(adjacency, dict) else adjacency[current]
            for neighbor in sorted(neighbors):
                if neighbor in seen:
                    continue
                seen[neighbor] = depth + 1
                queue.append(neighbor)
        for end, distance in seen.items():
            if start < end and distance <= limit:
                pair = (start, end)
                previous = distances.get(pair)
                if previous is None or distance < previous:
                    distances[pair] = distance
    return distances
