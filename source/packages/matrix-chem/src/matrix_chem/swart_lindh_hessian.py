"""Generalized Swart--Lindh primitive screening and Hessian seed.

The base equations are Eqs. (6)--(10) and (12) of Swart and Bickelhaupt,
Int. J. Quantum Chem. 106, 2536--2544 (2006), DOI 10.1002/qua.21049.
They generalize the Lindh initial-Hessian idea to a single description of
strong and weak coordinates.  MATRIX extends only the edge screening input:
special, explicitly declared interactions may carry an effective bond order.
This permits hydrogen bonds and haptic metal--center coordinates without
assigning fictitious atomic radii to geometric centers.

Distances and covalent-radius sums must use the same unit.  Bond force
constants returned by this module are in hartree per bohr squared; angular
force constants are in hartree per radian squared.
"""

from __future__ import annotations

import math
from typing import Sequence


SWART_LINDH_SCHEMA = "matrix.chem.swart_lindh_hessian.v1"
SWART_LINDH_REFERENCE = "Swart_Bickelhaupt_IJQC_2006_DOI_10.1002/qua.21049"
SWART_LINDH_PARENT_REFERENCE = (
    "Lindh_et_al_CPL_1995_DOI_10.1016/0009-2614(95)00646-L"
)

SWART_LINDH_STRONG_SCREENING_THRESHOLD = 0.7
SWART_LINDH_PRIMITIVE_WEIGHT_THRESHOLD = 0.3
SWART_LINDH_LINEAR_DAMPING = 0.12
SWART_LINDH_BOND_FORCE = 0.35
SWART_LINDH_ANGLE_FORCE = 0.15
SWART_LINDH_DIHEDRAL_FORCE = 0.005
SWART_LINDH_IMPROPER_FORCE = 0.005
DEFAULT_SPECIAL_EDGE_EFFECTIVE_ORDER = 0.25


def swart_lindh_screening(
    distance: float,
    covalent_radius_sum: float,
    *,
    effective_order: float = 1.0,
) -> float:
    """Return generalized pair screening ``order*exp(-(r/C - 1))``.

    ``effective_order=1`` is exactly Eq. (6).  A different value is allowed
    only as an explicit descriptor for a special edge; callers own and record
    the chemical policy that supplied it.
    """

    r = _positive_finite(distance, "distance")
    covalent_sum = _positive_finite(covalent_radius_sum, "covalent_radius_sum")
    order = _positive_finite(effective_order, "effective_order")
    return float(order * math.exp(-(r / covalent_sum - 1.0)))


def swart_lindh_center_screening(*, effective_order: float) -> float:
    """Return screening for an edge ending at a declared geometric center.

    A geometric center has no atomic covalent radius.  Its explicit effective
    order is therefore the complete dimensionless screening descriptor.
    """

    return _positive_finite(effective_order, "effective_order")


def swart_lindh_primitive_weight(
    family: str,
    edge_screenings: Sequence[float],
    *,
    angles_radian: Sequence[float] = (),
    damping: float = SWART_LINDH_LINEAR_DAMPING,
) -> float:
    """Evaluate Eqs. (7)--(10) for bond, angle and torsional primitives."""

    normalized = _normalize_family(family)
    rho = _validated_screenings(edge_screenings)
    damp = float(damping)
    if not math.isfinite(damp) or not 0.0 <= damp <= 1.0:
        raise ValueError("damping must be finite and lie in [0, 1]")
    expected_edges, expected_angles = {
        "bond": (1, 0),
        "angle": (2, 1),
        "dihedral": (3, 2),
        "improper": (3, 2),
    }[normalized]
    if len(rho) != expected_edges:
        raise ValueError(
            f"{normalized} requires {expected_edges} edge screening values"
        )
    angles = tuple(float(value) for value in angles_radian)
    if len(angles) != expected_angles or not all(math.isfinite(value) for value in angles):
        raise ValueError(f"{normalized} requires {expected_angles} finite angles")
    geometric_mean = math.prod(rho) ** (1.0 / len(rho))
    angular_factor = math.prod(
        damp + (1.0 - damp) * abs(math.sin(value)) for value in angles
    )
    return float(geometric_mean * angular_factor)


def swart_lindh_force_constant(
    family: str,
    edge_screenings: Sequence[float],
) -> float:
    """Evaluate the primitive diagonal initial-Hessian model of Eq. (12)."""

    normalized = _normalize_family(family)
    rho = _validated_screenings(edge_screenings)
    expected_edges = {
        "bond": 1,
        "angle": 2,
        "dihedral": 3,
        "improper": 3,
    }[normalized]
    if len(rho) != expected_edges:
        raise ValueError(
            f"{normalized} requires {expected_edges} edge screening values"
        )
    prefactor = {
        "bond": SWART_LINDH_BOND_FORCE,
        "angle": SWART_LINDH_ANGLE_FORCE,
        "dihedral": SWART_LINDH_DIHEDRAL_FORCE,
        "improper": SWART_LINDH_IMPROPER_FORCE,
    }[normalized]
    return float(prefactor * math.prod(rho))


def _normalize_family(family: str) -> str:
    normalized = str(family).strip().lower().replace("-", "_")
    aliases = {
        "r": "bond",
        "stretch": "bond",
        "bond": "bond",
        "a": "angle",
        "l": "angle",
        "bend": "angle",
        "linear_bend": "angle",
        "angle": "angle",
        "d": "dihedral",
        "torsion": "dihedral",
        "dihedral": "dihedral",
        "u": "improper",
        "impd": "improper",
        "out_of_plane": "improper",
        "improper_dihedral": "improper",
        "improper": "improper",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported Swart--Lindh primitive family {family!r}") from exc


def _validated_screenings(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or not all(math.isfinite(value) and value > 0.0 for value in result):
        raise ValueError("edge screening values must be finite and positive")
    return result


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


__all__ = [
    "DEFAULT_SPECIAL_EDGE_EFFECTIVE_ORDER",
    "SWART_LINDH_ANGLE_FORCE",
    "SWART_LINDH_BOND_FORCE",
    "SWART_LINDH_DIHEDRAL_FORCE",
    "SWART_LINDH_IMPROPER_FORCE",
    "SWART_LINDH_LINEAR_DAMPING",
    "SWART_LINDH_PARENT_REFERENCE",
    "SWART_LINDH_PRIMITIVE_WEIGHT_THRESHOLD",
    "SWART_LINDH_REFERENCE",
    "SWART_LINDH_SCHEMA",
    "SWART_LINDH_STRONG_SCREENING_THRESHOLD",
    "swart_lindh_center_screening",
    "swart_lindh_force_constant",
    "swart_lindh_primitive_weight",
    "swart_lindh_screening",
]
