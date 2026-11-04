"""Uncertainty-bearing periodic-table priors for missing ZAFF parameters.

The implementation borrows the *structure* of the transferable GFN-FF
strategy, not its fitted parameter tables.  All coefficients are determined
from validated ZAFF observations supplied by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


ZAFF_PERIODIC_PRIOR_SCHEMA = "matrix.zaff.periodic_parameter_prior.v1"


@dataclass(frozen=True)
class PeriodicPosition:
    atomic_number: int
    period: int
    group: int
    block: str
    series_coordinate: int
    electronegativity: float | None
    polarizability_angstrom3: float | None


@dataclass(frozen=True)
class ElementParameterObservation:
    atomic_number: int
    value: float
    standard_uncertainty: float = 0.0

    def __post_init__(self) -> None:
        if not 1 <= int(self.atomic_number) <= 118:
            raise ValueError("atomic number must be in [1, 118]")
        if not math.isfinite(float(self.value)):
            raise ValueError("element-parameter value must be finite")
        if not math.isfinite(float(self.standard_uncertainty)) or float(
            self.standard_uncertainty
        ) < 0.0:
            raise ValueError("standard uncertainty must be finite and nonnegative")


@dataclass(frozen=True)
class PeriodicTrendEstimate:
    atomic_number: int
    value: float
    standard_uncertainty: float
    relative_uncertainty: float
    trusted: bool
    selection: str
    contributors: tuple[int, ...]
    nearest_descriptor_distance: float
    validation_rmse: float
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": ZAFF_PERIODIC_PRIOR_SCHEMA,
            "atomic_number": self.atomic_number,
            "value": self.value,
            "standard_uncertainty": self.standard_uncertainty,
            "relative_uncertainty": self.relative_uncertainty,
            "trusted": self.trusted,
            "selection": self.selection,
            "contributors": list(self.contributors),
            "nearest_descriptor_distance": self.nearest_descriptor_distance,
            "validation_rmse": self.validation_rmse,
            "message": self.message,
        }


def periodic_position(atomic_number: int) -> PeriodicPosition:
    """Return periodic coordinates plus canonical ORACLE atomic descriptors."""

    z = int(atomic_number)
    if not 1 <= z <= 118:
        raise ValueError("atomic number must be in [1, 118]")
    period = next(index for index, end in enumerate((2, 10, 18, 36, 54, 86, 118), 1) if z <= end)
    from matrix_chem import electronegativity, polarizability

    chi = electronegativity(z)
    alpha = polarizability(z)
    descriptor_tail = (
        None if chi is None else float(chi),
        None if alpha is None else float(alpha),
    )
    if z == 1:
        return PeriodicPosition(z, 1, 1, "s", 1, *descriptor_tail)
    if z == 2:
        return PeriodicPosition(z, 1, 18, "s", 2, *descriptor_tail)
    if 57 <= z <= 71:
        return PeriodicPosition(z, 6, 3, "f", z - 56, *descriptor_tail)
    if 89 <= z <= 103:
        return PeriodicPosition(z, 7, 3, "f", z - 88, *descriptor_tail)
    group = _GROUP_BY_ATOMIC_NUMBER[z]
    block = "s" if group <= 2 else ("d" if group <= 12 else "p")
    coordinate = group if block != "d" else group - 2
    return PeriodicPosition(z, period, group, block, coordinate, *descriptor_tail)


def estimate_periodic_parameter(
    atomic_number: int,
    observations: Sequence[ElementParameterObservation],
    *,
    positive: bool = True,
    maximum_relative_uncertainty: float = 0.35,
) -> PeriodicTrendEstimate:
    """Estimate one missing elemental parameter by a local group/period fit.

    A target is trusted only when it is bracketed along a chemically coherent
    axis (group, period, or f-series coordinate), the local weighted fit has
    sufficient support, and its propagated uncertainty is below the requested
    ceiling.  Callers must retain their lower-level fallback for untrusted
    estimates.
    """

    target = periodic_position(atomic_number)
    unique: dict[int, ElementParameterObservation] = {}
    for item in observations:
        if positive and float(item.value) <= 0.0:
            raise ValueError("positive periodic parameters require positive observations")
        if int(item.atomic_number) in unique:
            raise ValueError(f"duplicate observation for Z={item.atomic_number}")
        unique[int(item.atomic_number)] = item
    if target.atomic_number in unique:
        item = unique[target.atomic_number]
        sigma = float(item.standard_uncertainty)
        relative = sigma / max(abs(float(item.value)), np.finfo(float).tiny)
        return PeriodicTrendEstimate(
            atomic_number=target.atomic_number,
            value=float(item.value),
            standard_uncertainty=sigma,
            relative_uncertainty=relative,
            trusted=True,
            selection="ZAFF_VALIDATED_ELEMENT",
            contributors=(target.atomic_number,),
            nearest_descriptor_distance=0.0,
            validation_rmse=0.0,
            message="used the validated ZAFF elemental parameter",
        )
    candidates = [
        item
        for item in unique.values()
        if periodic_position(item.atomic_number).block == target.block
    ]
    if len(candidates) < 3:
        return _unsupported_estimate(target, candidates, "fewer than three same-block observations")
    ranked = sorted(
        ((_descriptor_distance(target, periodic_position(item.atomic_number)), item) for item in candidates),
        key=lambda pair: (pair[0], pair[1].atomic_number),
    )
    local = [item for _distance, item in ranked[: min(12, len(ranked))]]
    value, sigma, rmse = _local_weighted_fit(target, local, positive=positive)
    nearest = float(ranked[0][0])
    relative = sigma / max(abs(value), np.finfo(float).tiny)
    bracketed = _is_bracketed(target, local)
    trusted = (
        bracketed
        and len(local) >= 3
        and nearest <= 2.5
        and relative <= float(maximum_relative_uncertainty)
    )
    reason = (
        "validated local interpolation"
        if trusted
        else "periodic estimate retained only as an untrusted prior; use the configured fallback"
    )
    return PeriodicTrendEstimate(
        atomic_number=target.atomic_number,
        value=value,
        standard_uncertainty=sigma,
        relative_uncertainty=relative,
        trusted=trusted,
        selection="ZAFF_PERIODIC_INTERPOLATION" if trusted else "PERIODIC_PRIOR_UNTRUSTED",
        contributors=tuple(sorted(item.atomic_number for item in local)),
        nearest_descriptor_distance=nearest,
        validation_rmse=rmse,
        message=reason,
    )


def _local_weighted_fit(
    target: PeriodicPosition,
    observations: Sequence[ElementParameterObservation],
    *,
    positive: bool,
) -> tuple[float, float, float]:
    coefficients, in_sample_rmse = _weighted_periodic_coefficients(
        target,
        observations,
        positive=positive,
    )
    positions = [periodic_position(item.atomic_number) for item in observations]
    leave_one_out_residuals: list[float] = []
    if len(observations) >= 4:
        for omitted, item in enumerate(observations):
            training = tuple(
                candidate
                for index, candidate in enumerate(observations)
                if index != omitted
            )
            prediction, _training_rmse = _weighted_periodic_coefficients(
                periodic_position(item.atomic_number),
                training,
                positive=positive,
            )
            observed = math.log(float(item.value)) if positive else float(item.value)
            leave_one_out_residuals.append(observed - float(prediction[0]))
    validation_rmse_transformed = (
        float(np.sqrt(np.mean(np.square(leave_one_out_residuals))))
        if leave_one_out_residuals
        else in_sample_rmse
    )
    rmse = max(in_sample_rmse, validation_rmse_transformed)
    nearest = min(_descriptor_distance(target, position) for position in positions)
    sigma_transformed = math.sqrt(rmse * rmse + (0.04 * nearest) ** 2)
    if positive:
        value = math.exp(float(coefficients[0]))
        sigma = value * math.expm1(min(sigma_transformed, 2.0))
        validation_rmse = math.expm1(min(rmse, 2.0))
    else:
        value = float(coefficients[0])
        sigma = sigma_transformed
        validation_rmse = rmse
    return value, sigma, validation_rmse


def _weighted_periodic_coefficients(
    target: PeriodicPosition,
    observations: Sequence[ElementParameterObservation],
    *,
    positive: bool,
) -> tuple[np.ndarray, float]:
    rows = []
    weights = []
    values = []
    for item in observations:
        position = periodic_position(item.atomic_number)
        dg, dp, ds, dchi, dalpha = _descriptor_deltas(target, position)
        rows.append((1.0, dg, dp, ds, dchi, dalpha))
        distance = _descriptor_distance(target, position)
        relative_sigma = float(item.standard_uncertainty) / max(
            abs(float(item.value)), 1.0e-12
        )
        weights.append(
            math.exp(-0.5 * distance * distance)
            / max(relative_sigma**2, 0.01**2)
        )
        values.append(math.log(float(item.value)) if positive else float(item.value))
    x = np.asarray(rows, dtype=float)
    y = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    root_w = np.sqrt(w / max(float(np.max(w)), np.finfo(float).tiny))
    ridge = np.diag((1.0e-10, 0.08, 0.08, 0.08, 0.12, 0.12))
    normal = (x * root_w[:, None]).T @ (x * root_w[:, None]) + ridge
    rhs = (x * root_w[:, None]).T @ (y * root_w)
    coefficients = np.linalg.solve(normal, rhs)
    residual = y - x @ coefficients
    rmse = float(np.sqrt(np.average(residual * residual, weights=w)))
    return coefficients, rmse


def _descriptor_deltas(
    target: PeriodicPosition, source: PeriodicPosition
) -> tuple[float, float, float, float, float]:
    group_delta = (float(source.group) - float(target.group)) / 2.0
    period_delta = float(source.period) - float(target.period)
    series_delta = (
        (float(source.series_coordinate) - float(target.series_coordinate)) / 3.0
        if target.block == "f"
        else 0.0
    )
    electronegativity_delta = (
        float(source.electronegativity) - float(target.electronegativity)
        if source.electronegativity is not None and target.electronegativity is not None
        else 0.0
    )
    polarizability_delta = (
        math.log(float(source.polarizability_angstrom3))
        - math.log(float(target.polarizability_angstrom3))
        if source.polarizability_angstrom3 is not None
        and target.polarizability_angstrom3 is not None
        else 0.0
    )
    return (
        group_delta,
        period_delta,
        series_delta,
        electronegativity_delta,
        polarizability_delta,
    )


def _descriptor_distance(target: PeriodicPosition, source: PeriodicPosition) -> float:
    dg, dp, ds, dchi, dalpha = _descriptor_deltas(target, source)
    return math.sqrt(dg * dg + dp * dp + ds * ds + dchi * dchi + dalpha * dalpha)


def _is_bracketed(
    target: PeriodicPosition, observations: Sequence[ElementParameterObservation]
) -> bool:
    positions = [periodic_position(item.atomic_number) for item in observations]
    if target.block == "f":
        coordinates = [
            item.series_coordinate for item in positions if item.period == target.period
        ]
        return bool(coordinates) and min(coordinates) < target.series_coordinate < max(coordinates)
    groups = [item.group for item in positions if item.period == target.period]
    periods = [item.period for item in positions if item.group == target.group]
    return (
        bool(groups) and min(groups) < target.group < max(groups)
    ) or (
        bool(periods) and min(periods) < target.period < max(periods)
    )


def _unsupported_estimate(
    target: PeriodicPosition,
    observations: Sequence[ElementParameterObservation],
    message: str,
) -> PeriodicTrendEstimate:
    contributors = tuple(sorted(item.atomic_number for item in observations))
    return PeriodicTrendEstimate(
        atomic_number=target.atomic_number,
        value=float("nan"),
        standard_uncertainty=float("inf"),
        relative_uncertainty=float("inf"),
        trusted=False,
        selection="PERIODIC_PRIOR_UNAVAILABLE",
        contributors=contributors,
        nearest_descriptor_distance=float("inf"),
        validation_rmse=float("inf"),
        message=message,
    )


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
_GROUP_BY_ATOMIC_NUMBER = {
    atomic_number: group for group, row in _GROUP_ROWS.items() for atomic_number in row
}


__all__ = [
    "ZAFF_PERIODIC_PRIOR_SCHEMA",
    "ElementParameterObservation",
    "PeriodicPosition",
    "PeriodicTrendEstimate",
    "estimate_periodic_parameter",
    "periodic_position",
]
