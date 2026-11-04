"""Analytic surface bases retained from the useful SURVIBTM design."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, sin, sqrt
from typing import Any, Mapping


ZAFF_SPECIAL_BASIS_SCHEMA = "matrix.zaff.special_basis.v1"


@dataclass(frozen=True)
class ZaffSpecialBasis:
    """One serializable one-coordinate basis used in mixed PES products.

    ``order`` follows the historic SURVIBTM convention for Fourier terms
    (positive sine, negative cosine) and its triangular enumeration of
    associated Legendre functions.  The implementation is independent and
    uses no legacy global state.
    """

    kind: str
    reference: float = 0.0
    schema: str = ZAFF_SPECIAL_BASIS_SCHEMA

    def __post_init__(self) -> None:
        allowed = {
            "TAYLOR",
            "SIMONS_PARR_FINLAN",
            "OGILVIE",
            "FOURIER",
            "HALF_ANGLE_FOURIER",
            "ASSOCIATED_LEGENDRE",
            "ASSOCIATED_LEGENDRE_COSINE",
        }
        kind = str(self.kind).upper()
        if kind not in allowed:
            raise ValueError(f"unsupported ZAFF special basis {self.kind!r}")
        if not isfinite(float(self.reference)):
            raise ValueError("special-basis reference must be finite")
        if kind in {"SIMONS_PARR_FINLAN", "OGILVIE"} and self.reference <= 0.0:
            raise ValueError("radial compactification requires a positive reference")
        object.__setattr__(self, "kind", kind)

    def evaluate(self, coordinate: float, order: int) -> float:
        value = float(coordinate)
        power = int(order)
        if not isfinite(value):
            raise ValueError("special-basis coordinate must be finite")
        if self.kind == "TAYLOR":
            return (value - self.reference) ** power
        if self.kind == "SIMONS_PARR_FINLAN":
            return ((value - self.reference) / value) ** power
        if self.kind == "OGILVIE":
            return ((value - self.reference) / (value + self.reference)) ** power
        if self.kind in {"FOURIER", "HALF_ANGLE_FOURIER"}:
            if power == 0:
                return 1.0
            argument = abs(power) * value
            if self.kind == "HALF_ANGLE_FOURIER":
                argument *= 0.5
            return sin(argument) if power > 0 else cos(argument)
        if power <= 0:
            return 1.0 if power == 0 else _raise_legendre_order()
        degree, magnetic = _survibtm_legendre_indices(power)
        argument = value
        if self.kind == "ASSOCIATED_LEGENDRE_COSINE":
            argument = cos(value)
        return _associated_legendre_no_condon_shortley(degree, magnetic, argument)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "kind": self.kind, "reference": self.reference}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ZaffSpecialBasis":
        if payload.get("schema") != ZAFF_SPECIAL_BASIS_SCHEMA:
            raise ValueError("unsupported ZAFF special-basis schema")
        return cls(kind=str(payload["kind"]), reference=float(payload.get("reference", 0.0)))


def _raise_legendre_order() -> float:
    raise ValueError("associated Legendre enumeration must be nonnegative")


def _survibtm_legendre_indices(index: int) -> tuple[int, int]:
    degree = 1
    preceding = 0
    while index > preceding + degree + 1:
        preceding += degree + 1
        degree += 1
    return degree, index - preceding - 1


def _associated_legendre_no_condon_shortley(
    degree: int,
    magnetic: int,
    argument: float,
) -> float:
    if not -1.0 <= argument <= 1.0:
        raise ValueError("associated Legendre argument must lie in [-1, 1]")
    if not 0 <= magnetic <= degree:
        raise ValueError("associated Legendre indices require 0 <= m <= l")
    p_mm = 1.0
    root = sqrt(max(0.0, 1.0 - argument * argument))
    for factor in range(1, magnetic + 1):
        p_mm *= (2 * factor - 1) * root
    if degree == magnetic:
        return p_mm
    p_m1m = (2 * magnetic + 1) * argument * p_mm
    if degree == magnetic + 1:
        return p_m1m
    previous, current = p_mm, p_m1m
    for ell in range(magnetic + 2, degree + 1):
        following = (
            (2 * ell - 1) * argument * current
            - (ell + magnetic - 1) * previous
        ) / (ell - magnetic)
        previous, current = current, following
    return current
