from __future__ import annotations

import math


GDV_PERIODIC_TOLERANCE = 1.0e-6


def _validated_period(period: float) -> float:
    value = float(period)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("periodic-coordinate period must be finite and positive")
    return value


def gdv_principal_dihedral(value: float, period: float = 2.0 * math.pi) -> float:
    """Return the canonical value in a GDV-compatible periodic domain.

    Ordinary labelled dihedrals use ``2*pi``.  A torsion with physical
    periodicity ``n`` uses ``2*pi/n``; the boundary convention remains the
    one used by GDV's ``FixCrd(IOp=2)``.
    """

    full_period = _validated_period(period)
    half_period = 0.5 * full_period
    result = math.fmod(float(value), full_period)
    if abs(result) > half_period:
        result = -math.copysign(full_period - abs(result), result)
    if abs(result + half_period) < GDV_PERIODIC_TOLERANCE or abs(
        result - half_period
    ) < GDV_PERIODIC_TOLERANCE:
        result = half_period
    if abs(result) < GDV_PERIODIC_TOLERANCE:
        result = 0.0
    return result


def gdv_match_dihedral_phase(
    value: float,
    reference: float,
    period: float = 2.0 * math.pi,
) -> float:
    """Return the physically equivalent branch nearest ``reference``."""

    full_period = _validated_period(period)
    half_period = 0.5 * full_period
    delta = math.fmod(float(value) - float(reference), full_period)
    if abs(delta - full_period) < GDV_PERIODIC_TOLERANCE or abs(
        delta + full_period
    ) < GDV_PERIODIC_TOLERANCE:
        delta = 0.0
    if abs(delta - half_period) < GDV_PERIODIC_TOLERANCE:
        delta = half_period
    if abs(delta + half_period) < GDV_PERIODIC_TOLERANCE:
        delta = -half_period

    result = float(reference) + delta
    if delta > half_period:
        result -= full_period
    elif delta < -half_period:
        result += full_period
    if abs(result) < GDV_PERIODIC_TOLERANCE:
        result = 0.0
    if abs(result - half_period) < GDV_PERIODIC_TOLERANCE:
        result = half_period
    if abs(result + half_period) < GDV_PERIODIC_TOLERANCE:
        result = -half_period
    return result


__all__ = [
    "GDV_PERIODIC_TOLERANCE",
    "gdv_match_dihedral_phase",
    "gdv_principal_dihedral",
]
