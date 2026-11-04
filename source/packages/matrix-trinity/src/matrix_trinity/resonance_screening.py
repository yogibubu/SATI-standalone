"""Occupation-aware energy/coefficient tests inspired by Yang--Bloino (2022).

The published two-test principle is extended with exact two-level energy
scores and directed second-order coefficients. No band intensity is divided
by another intensity. Historical Gaussian-comparison profiles are separate.
"""

from dataclasses import asdict, dataclass
from typing import Callable, Mapping

import numpy as np


PROFILE_ID = "vp2td-intensity-v1"
State = tuple[int, ...]


@dataclass(frozen=True)
class IntensityResonanceCriteria:
    energy_tolerance_cm: float = 1.0
    first_order_limit: float = 0.1
    second_order_limit: float = 0.3

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("resonance criteria must be positive and finite")

    @classmethod
    def from_options(cls, options: Mapping[str, float] | None):
        try:
            return cls(**({} if options is None else dict(options)))
        except TypeError as exc:
            raise ValueError("unknown intensity resonance criterion") from exc


def pair_decision(
    *, kind: str, gap_cm: float, coupling_cm: float,
    numerators_cm: tuple[complex, complex] | None,
    criteria: IntensityResonanceCriteria,
    denominator_tolerance_cm: float, numerical_zero_cm: float,
) -> dict[str, object]:
    """Explicit SI energy and directed-wavefunction criteria for one pair."""
    gap, v = abs(float(gap_cm)), abs(float(coupling_cm))
    shift = 0.0 if v == 0 else v * (2 * v / (np.hypot(gap, 2 * v) + gap))
    pole = gap <= denominator_tolerance_cm
    if kind == "1-2":
        numerators = (complex(coupling_cm), complex(coupling_cm))
        limit = criteria.first_order_limit
        # s^2/delta is exactly |W|^2/delta - s, without cancellation.
        score = (np.inf if v > numerical_zero_cm else 0.0) if pole else shift * (shift / gap)
    else:
        if numerators_cm is None:
            raise ValueError("Darling--Dennison screening requires directed numerators")
        numerators = numerators_cm
        limit = criteria.second_order_limit
        score = shift
    numerator_max = max(abs(value) for value in numerators)
    coefficient_pole = pole and numerator_max > numerical_zero_cm
    coefficient = (np.inf if coefficient_pole else 0.0) if pole else numerator_max / gap
    active = max(v, numerator_max) > numerical_zero_cm
    energy_pass = bool(active and score >= criteria.energy_tolerance_cm)
    coefficient_pass = bool(active and coefficient >= limit)
    return {
        "screening_profile": PROFILE_ID,
        "pair_shift_cm": float(shift),
        "energy_score_cm": None if not np.isfinite(score) else float(score),
        "energy_score_pole": bool(not np.isfinite(score)),
        "coefficient_abs_forward": None if coefficient_pole else (0.0 if pole else float(abs(numerators[0]) / gap)),
        "coefficient_abs_reverse": None if coefficient_pole else (0.0 if pole else float(abs(numerators[1]) / gap)),
        "coefficient_max": None if coefficient_pole else float(coefficient),
        "coefficient_pole": coefficient_pole,
        "coefficient_threshold": limit,
        "energy_threshold_cm": criteria.energy_tolerance_cm,
        "energy_selected": energy_pass,
        "coefficient_selected": coefficient_pass,
        "decision": "accepted" if energy_pass or coefficient_pass else "energy and coefficient tests",
    }


def directed_cubic_numerators(
    source: State, target: State, frequencies: np.ndarray,
    source_connections: Mapping[State, complex],
    target_connections: Mapping[State, complex],
    is_fermi: Callable[[State, State, complex], bool],
    *, denominator_tolerance_cm: float,
) -> tuple[complex, complex]:
    """Cubic part of both directed C2 numerators, including -A H1_PP.

    Use sparse operator connections already built by the solver. Shared
    component membership never removes a direct channel. Both accepted
    Fermi channels are omitted; with only one accepted channel, the folded
    term uses the other, nonresonant denominator.
    """
    forward = reverse = 0j
    ea, eb = (float(np.dot(frequencies, state)) for state in (source, target))
    for intermediate in sorted(source_connections.keys() & target_connections.keys()):
        product = np.conjugate(target_connections[intermediate]) * source_connections[intermediate]
        if product == 0:
            continue
        ra = is_fermi(source, intermediate, source_connections[intermediate])
        rb = is_fermi(target, intermediate, target_connections[intermediate])
        if ra and rb:
            continue
        et = float(np.dot(frequencies, intermediate))
        da, db = ea - et, eb - et
        forward_den = db if ra else da
        reverse_den = da if rb else db
        if min(abs(forward_den), abs(reverse_den)) <= denominator_tolerance_cm:
            raise ValueError("unresolved external denominator in second-order resonance coefficient")
        forward += product / forward_den
        reverse += np.conjugate(product) / reverse_den
    return forward, reverse
