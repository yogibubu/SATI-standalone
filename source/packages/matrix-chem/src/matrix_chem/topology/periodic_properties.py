"""Finite periodic-table descriptors used by transferable MATRIX models.

Primary tabulations retain their original APIs and missing-value semantics.
This module is the explicit, provenance-bearing completion layer used when a
model requires a finite descriptor for every real element, Z=1--118.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

from .covalent_radii import covalent_radius
from .electronegativity import PAULING
from .polarizability import POLARIZABILITY
from .vdw_radii import UFF, UFF_WELL_DEPTH_KCAL


PERIODIC_PROPERTIES_SCHEMA = "matrix.chem.periodic_properties.v1"

_GROUP_ROWS = {
    1: (1, 3, 11, 19, 37, 55, 87),
    2: (4, 12, 20, 38, 56, 88),
    3: (21, 39, 57, 89),
    4: (22, 40, 72, 104),
    5: (23, 41, 73, 105),
    6: (24, 42, 74, 106),
    7: (25, 43, 75, 107),
    8: (26, 44, 76, 108),
    9: (27, 45, 77, 109),
    10: (28, 46, 78, 110),
    11: (29, 47, 79, 111),
    12: (30, 48, 80, 112),
    13: (5, 13, 31, 49, 81, 113),
    14: (6, 14, 32, 50, 82, 114),
    15: (7, 15, 33, 51, 83, 115),
    16: (8, 16, 34, 52, 84, 116),
    17: (9, 17, 35, 53, 85, 117),
    18: (2, 10, 18, 36, 54, 86, 118),
}
_GROUP = {z: group for group, row in _GROUP_ROWS.items() for z in row}
_GROUP.update({z: 3 for z in range(57, 72)})
_GROUP.update({z: 3 for z in range(89, 104)})


@dataclass(frozen=True)
class PeriodicAtomicProperties:
    atomic_number: int
    period: int
    group: int
    block: str
    covalent_radius_angstrom: float
    vdw_radius_angstrom: float
    vdw_well_depth_kcal_per_mol: float
    electronegativity: float
    polarizability_angstrom3: float
    sources: tuple[str, ...]
    schema: str = PERIODIC_PROPERTIES_SCHEMA


@lru_cache(maxsize=118)
def periodic_atomic_properties(atomic_number: int) -> PeriodicAtomicProperties:
    """Resolve finite transferable descriptors for every element Z=1--118."""

    z = int(atomic_number)
    if not 1 <= z <= 118:
        raise ValueError("periodic atomic properties require Z in [1, 118]")
    period = next(
        index
        for index, end in enumerate((2, 10, 18, 36, 54, 86, 118), start=1)
        if z <= end
    )
    group = _GROUP[z]
    block = "f" if 57 <= z <= 71 or 89 <= z <= 103 else (
        "s" if group <= 2 else "d" if group <= 12 else "p"
    )
    rcov = float(covalent_radius(z))
    sources: list[str] = ["MANTINA_TRUHLAR_COVALENT_RADIUS"]

    raw_vdw = UFF.get(z)
    if raw_vdw is None:
        source_z = _nearest_group_member(z, lambda item: UFF.get(item) is not None)
        source_radius = float(UFF[source_z])
        source_rcov = float(covalent_radius(source_z))
        vdw = source_radius * rcov / source_rcov
        sources.append(f"PERIODIC_GROUP_UFF_RADIUS_TRANSFER_Z{source_z}")
    else:
        vdw = float(raw_vdw)
        sources.append("UFF_ELEMENT_RADIUS")

    raw_epsilon = UFF_WELL_DEPTH_KCAL.get(z)
    if raw_epsilon is None:
        source_z = _nearest_group_member(
            z, lambda item: UFF_WELL_DEPTH_KCAL.get(item) is not None
        )
        source_epsilon = float(UFF_WELL_DEPTH_KCAL[source_z])
        source_vdw = float(UFF.get(source_z))
        epsilon = source_epsilon * max(vdw / source_vdw, 0.25) ** 2
        sources.append(f"PERIODIC_GROUP_UFF_WELL_TRANSFER_Z{source_z}")
    else:
        epsilon = float(raw_epsilon)
        sources.append("UFF_ELEMENT_WELL_DEPTH")

    raw_chi = PAULING.get(z)
    if raw_chi is None:
        available = [item for item in _GROUP_ROWS[group] if PAULING.get(item) is not None]
        if available:
            chi = _inverse_period_average(z, available, PAULING)
            sources.append("PERIODIC_GROUP_ELECTRONEGATIVITY_INTERPOLATION")
        else:
            chi = 0.0
            sources.append("NOBLE_OR_UNRESOLVED_ZERO_CHARGE_TRANSFER_REFERENCE")
    else:
        chi = float(raw_chi)
        sources.append("PAULING_ELECTRONEGATIVITY")

    raw_alpha = POLARIZABILITY.get(z)
    if raw_alpha is None:
        # UFF volume supplies a finite, smooth lower-level prior. The factor
        # reproduces the order of magnitude of neutral main-group atoms and is
        # explicitly not presented as an experimental polarizability.
        alpha = 0.25 * vdw**3
        sources.append("UFF_VOLUME_POLARIZABILITY_PRIOR")
    else:
        alpha = float(raw_alpha)
        sources.append("TABULATED_STATIC_POLARIZABILITY")

    values = (rcov, vdw, epsilon, chi, alpha)
    if any(not math.isfinite(value) for value in values) or any(
        value <= 0.0 for value in (rcov, vdw, epsilon, alpha)
    ):
        raise RuntimeError(f"periodic completion failed for Z={z}")
    return PeriodicAtomicProperties(
        z, period, group, block, rcov, vdw, epsilon, chi, alpha,
        tuple(sources),
    )


def _nearest_group_member(z: int, predicate) -> int:
    candidates = [item for item in _GROUP_ROWS[_GROUP[z]] if predicate(item)]
    if not candidates:
        candidates = [item for item in range(1, 119) if predicate(item)]
    if not candidates:
        raise RuntimeError(f"no periodic contributor is available for Z={z}")
    target_period = _period(z)
    return min(candidates, key=lambda item: (abs(_period(item) - target_period), abs(item - z)))


def _inverse_period_average(z: int, candidates, table) -> float:
    target = _period(z)
    weights = [1.0 / (1.0 + abs(_period(item) - target)) for item in candidates]
    return float(
        sum(weight * float(table[item]) for weight, item in zip(weights, candidates, strict=True))
        / sum(weights)
    )


def _period(z: int) -> int:
    return next(
        index
        for index, end in enumerate((2, 10, 18, 36, 54, 86, 118), start=1)
        if int(z) <= end
    )


__all__ = [
    "PERIODIC_PROPERTIES_SCHEMA",
    "PeriodicAtomicProperties",
    "periodic_atomic_properties",
]
