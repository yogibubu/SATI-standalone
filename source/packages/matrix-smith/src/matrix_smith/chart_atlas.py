"""Local validity and chart-state atlas for SONIC coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CHART_ATLAS_SCHEMA = "matrix.smith.sonic_chart_atlas.v1"
DEFAULT_PERIODIC_BRANCH_MARGIN_RADIAN = 1.0e-6


@dataclass(frozen=True)
class SonicChartState:
    identifier: str
    family: str
    chart: str
    status: str
    value: float
    distance_to_boundary: float
    recommended_chart: str
    reason: str


@dataclass(frozen=True)
class SonicChartAtlas:
    schema: str
    states: tuple[SonicChartState, ...]
    safe: bool
    transitions_required: tuple[str, ...]


def classify_sonic_chart(
    identifier: str,
    family: str,
    value: float,
    *,
    periodicity: float | None = None,
    amplitude: float | None = None,
    linear_angle_threshold_radian: float = float(np.deg2rad(165.0)),
    singular_amplitude_tolerance: float = 1.0e-10,
    branch_margin_radian: float = DEFAULT_PERIODIC_BRANCH_MARGIN_RADIAN,
) -> SonicChartState:
    """Classify one coordinate without changing its value or chart."""

    coordinate = float(value)
    if not np.isfinite(coordinate):
        raise ValueError("SONIC chart value must be finite")
    normalized_family = str(family).strip().upper()
    chart = "CARTESIAN_LOCAL"
    status = "REGULAR"
    distance = float("inf")
    recommended = chart
    reason = "LOCAL_CHART_VALID"

    if normalized_family in {"TORSION", "IMPROPER_DIHEDRAL", "CYCLIC_TORSION"}:
        chart = "PERIODIC_PRINCIPAL"
        period = 2.0 * np.pi / max(float(periodicity or 1.0), 1.0e-12)
        wrapped = (coordinate + 0.5 * period) % period - 0.5 * period
        distance = abs(0.5 * period - abs(wrapped))
        if distance <= float(branch_margin_radian):
            status = "NEAR_BRANCH"
            recommended = "PERIODIC_CONTINUATION"
            reason = "PRINCIPAL_BRANCH_BOUNDARY"
    elif normalized_family in {"RING_PUCKER_COMPONENT", "RING_PHASE"}:
        chart = "RING_POLAR"
        magnitude = abs(float(amplitude if amplitude is not None else coordinate))
        distance = magnitude
        if magnitude <= float(singular_amplitude_tolerance):
            status = "NEAR_SINGULAR"
            recommended = "RING_CARTESIAN_COMPONENTS"
            reason = "POLAR_ANGLE_UNDEFINED_AT_ZERO_AMPLITUDE"
    elif normalized_family in {"BEND", "CYCLIC_BEND", "LINEAR_BEND"}:
        chart = "ANGULAR"
        magnitude = abs(coordinate)
        collinear_distance = min(magnitude, abs(np.pi - magnitude))
        collinear_margin = np.pi - float(linear_angle_threshold_radian)
        distance = abs(collinear_distance - collinear_margin)
        if collinear_distance <= collinear_margin:
            status = "NEAR_SINGULAR"
            recommended = "LINEAR_BEND"
            reason = "ANGLE_NEAR_COLLINEAR_LIMIT"

    return SonicChartState(
        identifier=str(identifier),
        family=normalized_family,
        chart=chart,
        status=status,
        value=coordinate,
        distance_to_boundary=float(distance),
        recommended_chart=recommended,
        reason=reason,
    )


def build_sonic_chart_atlas(states: tuple[SonicChartState, ...] | list[SonicChartState]) -> SonicChartAtlas:
    """Build a deterministic atlas and list coordinates needing transitions."""

    records = tuple(states)
    transitions = tuple(
        state.identifier
        for state in records
        if state.status in {"NEAR_BRANCH", "NEAR_SINGULAR", "OUT_OF_CHART"}
    )
    return SonicChartAtlas(
        schema=CHART_ATLAS_SCHEMA,
        states=records,
        safe=not transitions,
        transitions_required=transitions,
    )
