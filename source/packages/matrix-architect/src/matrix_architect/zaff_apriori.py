"""Chemical a-priori catalog for native ZAFF anharmonic couplings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ZaffAprioriCouplingPolicy:
    """Transferable locality rules applied before any anharmonic fit."""

    maximum_primitive_gap: int = 0
    maximum_synthon_diameter: int | None = None
    maximum_local_scale_ratio: float = 8.0
    preserve_local_torsion_couplings: bool = True
    preserve_geminal_xh_couplings: bool = True

    def __post_init__(self) -> None:
        if self.maximum_primitive_gap < 0:
            raise ValueError("ZAFF primitive gap cannot be negative")
        if self.maximum_synthon_diameter is not None and self.maximum_synthon_diameter < 1:
            raise ValueError("ZAFF synthon diameter must be positive when specified")
        if not np.isfinite(self.maximum_local_scale_ratio) or self.maximum_local_scale_ratio < 1:
            raise ValueError("ZAFF local-scale ratio must be finite and at least one")


@dataclass(frozen=True)
class ZaffAprioriSelection:
    """Selected catalog and auditable rejection counts."""

    retained: np.ndarray
    diagonal_terms: int
    local_synthon_terms: int
    geminal_xh_exceptions: int
    torsion_exceptions: int
    rejected_disconnected: int
    rejected_synthon_diameter: int
    rejected_scale_separation: int


def _factor_index(factor: object) -> int:
    if hasattr(factor, "primitive_index"):
        return int(getattr(factor, "primitive_index"))
    return int(factor[0])


def _topological_distances(atom_count: int, bonds: Sequence[Sequence[int]]) -> np.ndarray:
    adjacency = [set() for _ in range(atom_count)]
    for edge in bonds:
        if len(edge) != 2:
            raise ValueError("ZAFF topology edges must contain two atoms")
        left, right = map(int, edge)
        if left < 0 or right < 0 or left >= atom_count or right >= atom_count:
            raise ValueError("ZAFF topology edge is outside the atom range")
        adjacency[left].add(right)
        adjacency[right].add(left)
    distances = np.full((atom_count, atom_count), atom_count + 1, dtype=int)
    for source in range(atom_count):
        distances[source, source] = 0
        queue = [source]
        for atom in queue:
            for neighbor in adjacency[atom]:
                if distances[source, neighbor] > distances[source, atom] + 1:
                    distances[source, neighbor] = distances[source, atom] + 1
                    queue.append(neighbor)
    return distances


def _support_is_connected(
    indices: Sequence[int],
    atom_supports: Sequence[frozenset[int]],
    distances: np.ndarray,
    maximum_gap: int,
) -> bool:
    reached = {int(indices[0])}
    while True:
        neighbors = {
            index
            for index in indices
            if index not in reached
            and any(
                min(
                    distances[left, right]
                    for left in atom_supports[index]
                    for right in atom_supports[known]
                )
                <= maximum_gap
                for known in reached
            )
        }
        if not neighbors:
            break
        reached.update(neighbors)
    return len(reached) == len(indices)


def _is_xh_bond(primitive: object, atomic_numbers: np.ndarray) -> bool:
    atoms = tuple(map(int, getattr(primitive, "atoms")))
    return (
        str(getattr(primitive, "kind")) == "bond"
        and len(atoms) == 2
        and sum(atomic_numbers[list(atoms)] == 1) == 1
    )


def _geminal_xh(indices: Sequence[int], primitives: Sequence[object], numbers: np.ndarray) -> bool:
    if len(indices) != 2 or not all(_is_xh_bond(primitives[index], numbers) for index in indices):
        return False
    heavy = []
    for index in indices:
        heavy.append(next(atom for atom in primitives[index].atoms if numbers[int(atom)] != 1))
    return heavy[0] == heavy[1]


def select_zaff_apriori_couplings(
    primitives: Sequence[object],
    terms: Sequence[Sequence[object]],
    topology_bonds: Sequence[Sequence[int]],
    atomic_numbers: Sequence[int],
    local_frequency_scales: Sequence[float] | None = None,
    *,
    policy: ZaffAprioriCouplingPolicy = ZaffAprioriCouplingPolicy(),
) -> ZaffAprioriSelection:
    """Select terms from chemistry and harmonic data, before anharmonic fitting.

    A mixed term must form one shared-atom primitive network inside a compact
    covalent synthon. When harmonic scales are available, coordinates on
    strongly separated scales are rejected. Their omission keeps the same
    topology/type catalog usable at ZAFF0. Geminal XH stretches and local
    torsional couplings are physical exceptions because their coupling is part
    of the defining ZAFF model.
    Energies, cubic/quartic derivatives, VPT2 constants and frequencies are
    deliberately absent from this interface.
    """

    primitive_count = len(primitives)
    numbers = np.asarray(atomic_numbers, dtype=int)
    scales = (
        None if local_frequency_scales is None else np.asarray(local_frequency_scales, dtype=float)
    )
    if primitive_count == 0 or numbers.ndim != 1 or np.any(numbers <= 0):
        raise ValueError("invalid ZAFF a-priori chemical data")
    if scales is not None and (
        scales.shape != (primitive_count,) or not np.all(np.isfinite(scales)) or np.any(scales <= 0)
    ):
        raise ValueError("invalid ZAFF a-priori local harmonic scales")
    supports = tuple(frozenset(map(int, getattr(item, "atoms"))) for item in primitives)
    if any(not support or max(support) >= len(numbers) or min(support) < 0 for support in supports):
        raise ValueError("invalid ZAFF primitive atom support")
    distances = _topological_distances(len(numbers), topology_bonds)
    retained = np.zeros(len(terms), dtype=bool)
    counts = {
        "diagonal_terms": 0,
        "local_synthon_terms": 0,
        "geminal_xh_exceptions": 0,
        "torsion_exceptions": 0,
        "rejected_disconnected": 0,
        "rejected_synthon_diameter": 0,
        "rejected_scale_separation": 0,
    }
    for term_index, factors in enumerate(terms):
        indices = sorted({_factor_index(factor) for factor in factors})
        if not indices or indices[-1] >= primitive_count or indices[0] < 0:
            raise ValueError("ZAFF term references an unavailable primitive")
        if len(indices) == 1:
            retained[term_index] = True
            counts["diagonal_terms"] += 1
            continue
        if not _support_is_connected(
            indices,
            supports,
            distances,
            policy.maximum_primitive_gap,
        ):
            counts["rejected_disconnected"] += 1
            continue
        atoms = sorted(set().union(*(supports[index] for index in indices)))
        diameter = max(distances[left, right] for left in atoms for right in atoms)
        if (
            policy.maximum_synthon_diameter is not None
            and diameter > policy.maximum_synthon_diameter
        ):
            counts["rejected_synthon_diameter"] += 1
            continue
        geminal = policy.preserve_geminal_xh_couplings and _geminal_xh(indices, primitives, numbers)
        torsional = policy.preserve_local_torsion_couplings and any(
            str(getattr(primitives[index], "kind")) == "dihedral" for index in indices
        )
        ratio = 1.0 if scales is None else max(scales[indices]) / min(scales[indices])
        if (
            scales is not None
            and ratio > policy.maximum_local_scale_ratio
            and not (geminal or torsional)
        ):
            counts["rejected_scale_separation"] += 1
            continue
        retained[term_index] = True
        if geminal:
            counts["geminal_xh_exceptions"] += 1
        elif torsional:
            counts["torsion_exceptions"] += 1
        else:
            counts["local_synthon_terms"] += 1
    return ZaffAprioriSelection(retained=retained, **counts)


__all__ = [
    "ZaffAprioriCouplingPolicy",
    "ZaffAprioriSelection",
    "select_zaff_apriori_couplings",
]
