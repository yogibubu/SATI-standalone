"""Energy-free rigid-fragment repair of hard intermolecular overlaps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from matrix_chem.topology.elements import atomic_number
from matrix_chem.topology.vdw_radii import descriptor_vdw_radius


@dataclass(frozen=True)
class RigidOverlapRepair:
    coordinates_angstrom: np.ndarray
    converged: bool
    iterations: int
    initial_deficit_angstrom: float
    final_deficit_angstrom: float
    fragment_displacements_angstrom: tuple[float, ...]
    reason: str


def repair_rigid_fragment_overlaps(
    elements: Sequence[str],
    coordinates_angstrom: Sequence[Sequence[float]],
    rigid_fragments: Sequence[Sequence[int]],
    *,
    index_base: int = 0,
    clearance_scale: float = 0.55,
    tolerance_angstrom: float = 1.0e-3,
    maximum_iterations: int = 100,
) -> RigidOverlapRepair:
    """Translate whole fragments until hard vdW overlaps are removed.

    Intrafragment distances are exactly preserved.  The operation has no
    energetic meaning and is intended only as a feasibility repair for seeds.
    """

    xyz = np.asarray(coordinates_angstrom, dtype=float).copy()
    if xyz.shape != (len(elements), 3) or not np.all(np.isfinite(xyz)):
        raise ValueError("overlap repair needs finite Cartesian coordinates")
    if not 0.0 < clearance_scale <= 1.0 or tolerance_angstrom < 0.0:
        raise ValueError("invalid overlap-repair clearance or tolerance")
    if maximum_iterations < 1:
        raise ValueError("overlap-repair maximum_iterations must be positive")
    fragments = tuple(
        tuple(int(atom) - index_base for atom in fragment) for fragment in rigid_fragments
    )
    if not fragments or any(not fragment for fragment in fragments):
        raise ValueError("overlap repair needs non-empty rigid fragments")
    flat = [atom for fragment in fragments for atom in fragment]
    if len(set(flat)) != len(flat) or min(flat) < 0 or max(flat) >= len(xyz):
        raise ValueError("rigid fragments must be disjoint valid atom sets")
    atom_fragment = {atom: index for index, fragment in enumerate(fragments) for atom in fragment}
    radii = np.asarray(
        [
            descriptor_vdw_radius(atomic_number(symbol) or 0) or 1.5
            for symbol in elements
        ],
        dtype=float,
    )

    def contacts() -> tuple[float, list[tuple[int, int, float, np.ndarray]]]:
        maximum = 0.0
        result = []
        for left in range(len(xyz)):
            for right in range(left):
                lf = atom_fragment.get(left)
                rf = atom_fragment.get(right)
                if lf is None or rf is None or lf == rf:
                    continue
                vector = xyz[left] - xyz[right]
                distance = float(np.linalg.norm(vector))
                target = clearance_scale * (radii[left] + radii[right])
                deficit = target - distance
                if deficit > tolerance_angstrom:
                    direction = (
                        vector / distance
                        if distance > 1.0e-12
                        else np.asarray((1.0, 0.0, 0.0))
                    )
                    result.append((lf, rf, deficit, direction))
                    maximum = max(maximum, deficit)
        return maximum, result

    initial, active = contacts()
    displacement = np.zeros((len(fragments), 3), dtype=float)
    iteration = 0
    while active and iteration < maximum_iterations:
        shifts = np.zeros_like(displacement)
        counts = np.zeros(len(fragments), dtype=float)
        for left, right, deficit, direction in active:
            shift = 0.55 * deficit * direction
            shifts[left] += shift
            shifts[right] -= shift
            counts[left] += 1.0
            counts[right] += 1.0
        for index, fragment in enumerate(fragments):
            if counts[index]:
                step = shifts[index] / counts[index]
                xyz[list(fragment)] += step
                displacement[index] += step
        iteration += 1
        final, active = contacts()
    final, _ = contacts()
    converged = final <= tolerance_angstrom
    return RigidOverlapRepair(
        coordinates_angstrom=xyz,
        converged=converged,
        iterations=iteration,
        initial_deficit_angstrom=initial,
        final_deficit_angstrom=final,
        fragment_displacements_angstrom=tuple(
            float(np.linalg.norm(vector)) for vector in displacement
        ),
        reason="hard overlaps removed" if converged else "iteration limit reached",
    )
