"""Reproducible paths through a frozen SONIC ring-coordinate chart.

The helpers in this module generate *targets* for constrained optimizations.
They do not interpolate Cartesian structures and they do not define a
molecule-specific reaction coordinate.  Each segment follows the shortest
continuous displacement in the selected SONIC chart; LINK is responsible for
realizing every target and relaxing the coordinates outside the frozen ring
block.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Callable, Sequence, TypeVar

import numpy as np


@dataclass(frozen=True)
class SonicRingPathLandmark:
    """Stationary or reference structure expressed in one SONIC ring chart."""

    label: str
    values: tuple[float, ...]
    stationary_order: int | None = None

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("ring-path landmark label cannot be empty")
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("ring-path landmark values must be a nonempty vector")
        if not np.all(np.isfinite(values)):
            raise ValueError("ring-path landmark values must be finite")
        if self.stationary_order is not None and self.stationary_order < 0:
            raise ValueError("stationary_order cannot be negative")


@dataclass(frozen=True)
class SonicRingPathImage:
    """One constrained target on a piecewise SONIC ring path."""

    index: int
    segment: int
    fraction: float
    arc_fraction: float
    values: tuple[float, ...]
    left_label: str
    right_label: str
    landmark_label: str | None = None


@dataclass(frozen=True)
class SonicPathVariantSelection:
    """Globally minimum continuous choice from successive symmetry orbits."""

    variant_indices: tuple[int, ...]
    total_cost: float


_Variant = TypeVar("_Variant")


def cyclic_ring_automorphisms(
    ring_size: int, *, include_reversals: bool = True
) -> tuple[tuple[int, ...], ...]:
    """Return the topological automorphisms of an unlabelled simple cycle.

    Callers must filter these permutations using atom, bond and synthon labels
    before applying them to a substituted or fused chemical ring.  More
    general ring graphs pass the automorphisms supplied by the topology layer
    directly to :func:`select_minimum_cost_path_variants`.
    """

    size = int(ring_size)
    if size < 3:
        raise ValueError("a cyclic ring automorphism needs at least three vertices")
    permutations = [
        tuple((index + shift) % size for index in range(size))
        for shift in range(size)
    ]
    if include_reversals:
        permutations.extend(
            tuple((shift - index) % size for index in range(size))
            for shift in range(size)
        )
    return tuple(dict.fromkeys(permutations))


def labelled_graph_automorphisms(
    adjacency: Sequence[Sequence[bool | int]] | np.ndarray,
    *,
    vertex_labels: Sequence[object] | None = None,
    edge_labels: dict[tuple[int, int], object] | None = None,
    max_automorphisms: int = 4096,
) -> tuple[tuple[int, ...], ...]:
    """Compatibility facade; ORACLE owns labelled graph automorphisms."""

    from matrix_chem import enumerate_labelled_graph_automorphisms

    return enumerate_labelled_graph_automorphisms(
        adjacency,
        vertex_labels=vertex_labels,
        edge_labels=edge_labels,
        max_automorphisms=max_automorphisms,
    )


def select_minimum_cost_path_variants(
    candidate_groups: Sequence[Sequence[_Variant]],
    distance: Callable[[_Variant, _Variant], float],
) -> SonicPathVariantSelection:
    """Choose one candidate per landmark by global dynamic programming.

    The first and final structures, a desired endpoint sector, or any other
    path constraint are represented by passing a singleton candidate group.
    Intermediate groups may contain any chemically admissible graph-
    automorphism orbit.  This keeps molecule-specific symmetry labels out of
    the path algorithm.
    """

    groups = tuple(tuple(group) for group in candidate_groups)
    if len(groups) < 2 or any(not group for group in groups):
        raise ValueError("path-variant selection needs at least two nonempty groups")

    costs = np.zeros(len(groups[0]), dtype=float)
    back_pointers: list[tuple[int, ...]] = []
    for left_group, right_group in zip(groups[:-1], groups[1:], strict=True):
        next_costs = np.empty(len(right_group), dtype=float)
        pointers: list[int] = []
        for right_index, right in enumerate(right_group):
            alternatives = []
            for left_index, left in enumerate(left_group):
                step_cost = float(distance(left, right))
                if not np.isfinite(step_cost) or step_cost < 0.0:
                    raise ValueError("path-variant distance must be finite and non-negative")
                alternatives.append(float(costs[left_index]) + step_cost)
            selected = int(np.argmin(np.asarray(alternatives, dtype=float)))
            next_costs[right_index] = alternatives[selected]
            pointers.append(selected)
        costs = next_costs
        back_pointers.append(tuple(pointers))

    selected = int(np.argmin(costs))
    indices = [selected]
    for pointers in reversed(back_pointers):
        selected = pointers[selected]
        indices.append(selected)
    indices.reverse()
    return SonicPathVariantSelection(
        variant_indices=tuple(indices),
        total_cost=float(np.min(costs)),
    )


def shortest_periodic_delta(target: float, source: float, period: float = 2.0 * pi) -> float:
    """Return the signed shortest displacement from ``source`` to ``target``."""

    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("period must be positive and finite")
    delta = (float(target) - float(source) + 0.5 * period) % period - 0.5 * period
    # Resolve the exactly antipodal case reproducibly from the unwrapped input.
    if np.isclose(abs(delta), 0.5 * period, rtol=0.0, atol=1.0e-14):
        delta = np.copysign(0.5 * period, float(target) - float(source))
    return float(delta)


def build_sonic_pseudorotation_cycle(
    reference_values: Sequence[float],
    *,
    images: int = 9,
    metric: np.ndarray | None = None,
    include_closure: bool = True,
) -> tuple[SonicRingPathImage, ...]:
    """Generate a constant-amplitude cycle in a two-dimensional ring chart.

    Five-membered rings have two independent puckering coordinates.  The
    cycle is constructed after whitening the supplied positive-definite
    metric, so its amplitude is chart-metric invariant rather than tied to
    the numerical scaling of either coordinate.  No molecular symmetry is
    assumed; substituted and heteroatomic rings therefore traverse the full
    pseudorotation cycle.
    """

    reference = np.asarray(reference_values, dtype=float)
    if reference.shape != (2,) or not np.all(np.isfinite(reference)):
        raise ValueError("pseudorotation requires two finite ring coordinates")
    count = int(images)
    minimum = 3 if include_closure else 2
    if count < minimum:
        raise ValueError(f"pseudorotation needs at least {minimum} images")
    if metric is None:
        metric_array = np.eye(2, dtype=float)
    else:
        metric_array = np.asarray(metric, dtype=float)
        if metric_array.shape != (2, 2):
            raise ValueError("pseudorotation metric must be 2 by 2")
        if not np.allclose(metric_array, metric_array.T, rtol=0.0, atol=1.0e-12):
            raise ValueError("pseudorotation metric must be symmetric")
        if np.min(np.linalg.eigvalsh(metric_array)) <= 0.0:
            raise ValueError("pseudorotation metric must be positive definite")
    factor = np.linalg.cholesky(metric_array)
    whitened = factor.T @ reference
    amplitude = float(np.linalg.norm(whitened))
    if amplitude <= 1.0e-14:
        raise ValueError("reference puckering amplitude is zero")
    phase_zero = float(np.arctan2(whitened[1], whitened[0]))
    denominator = count - 1 if include_closure else count
    output = []
    for index in range(count):
        fraction = float(index) / float(denominator)
        phase = phase_zero + 2.0 * pi * fraction
        target_whitened = amplitude * np.asarray((np.cos(phase), np.sin(phase)))
        values = np.linalg.solve(factor.T, target_whitened)
        output.append(
            SonicRingPathImage(
                index=index,
                segment=0,
                fraction=fraction,
                arc_fraction=fraction,
                values=tuple(float(value) for value in values),
                left_label="pseudorotation_0",
                right_label="pseudorotation_2pi",
                landmark_label=(
                    "pseudorotation_0"
                    if index == 0
                    else "pseudorotation_2pi"
                    if include_closure and index == count - 1
                    else None
                ),
            )
        )
    return tuple(output)


def build_sonic_ring_path(
    landmarks: Sequence[SonicRingPathLandmark],
    *,
    images_per_segment: int | Sequence[int] = 7,
    periodic_indices: Sequence[int] = (),
    periods: Sequence[float] | None = None,
    metric: np.ndarray | None = None,
) -> tuple[SonicRingPathImage, ...]:
    """Interpolate a landmark string in a fixed SONIC ring-coordinate chart.

    End points are included and shared landmarks occur only once.  Angular
    components follow their shortest periodic displacement.  ``arc_fraction``
    is computed from the supplied positive-definite coordinate metric (the
    identity by default) and is intended only as a reproducible scan label;
    the electronic energy is never fitted as a polynomial in that label.
    """

    points = tuple(landmarks)
    if len(points) < 2:
        raise ValueError("a ring path requires at least two landmarks")
    dimension = len(points[0].values)
    if any(len(point.values) != dimension for point in points):
        raise ValueError("all ring-path landmarks must use the same SONIC dimension")

    nsegments = len(points) - 1
    if isinstance(images_per_segment, int):
        counts = (images_per_segment,) * nsegments
    else:
        counts = tuple(int(value) for value in images_per_segment)
        if len(counts) != nsegments:
            raise ValueError("images_per_segment must contain one value per segment")
    if any(value < 2 for value in counts):
        raise ValueError("every ring-path segment needs at least two images")

    periodic = tuple(int(index) for index in periodic_indices)
    if len(set(periodic)) != len(periodic) or any(index < 0 or index >= dimension for index in periodic):
        raise ValueError("periodic_indices must be unique valid coordinate indices")
    if periods is None:
        period_values = (2.0 * pi,) * len(periodic)
    else:
        period_values = tuple(float(value) for value in periods)
        if len(period_values) != len(periodic):
            raise ValueError("periods must match periodic_indices")
        if any(not np.isfinite(value) or value <= 0.0 for value in period_values):
            raise ValueError("all periods must be positive and finite")
    periodic_map = dict(zip(periodic, period_values, strict=True))

    if metric is None:
        metric_array = np.eye(dimension, dtype=float)
    else:
        metric_array = np.asarray(metric, dtype=float)
        if metric_array.shape != (dimension, dimension):
            raise ValueError("ring-path metric has the wrong shape")
        if not np.allclose(metric_array, metric_array.T, rtol=0.0, atol=1.0e-12):
            raise ValueError("ring-path metric must be symmetric")
        if np.min(np.linalg.eigvalsh(metric_array)) <= 0.0:
            raise ValueError("ring-path metric must be positive definite")

    segment_deltas: list[np.ndarray] = []
    segment_lengths: list[float] = []
    for left, right in zip(points[:-1], points[1:], strict=True):
        source = np.asarray(left.values, dtype=float)
        target = np.asarray(right.values, dtype=float)
        delta = target - source
        for coordinate_index, period in periodic_map.items():
            delta[coordinate_index] = shortest_periodic_delta(
                target[coordinate_index], source[coordinate_index], period
            )
        segment_deltas.append(delta)
        segment_lengths.append(float(np.sqrt(delta @ metric_array @ delta)))

    total_length = float(sum(segment_lengths))
    if total_length <= 0.0:
        raise ValueError("ring-path landmarks do not define a finite displacement")

    images: list[SonicRingPathImage] = []
    traversed = 0.0
    for segment, (left, right, delta, length, count) in enumerate(
        zip(points[:-1], points[1:], segment_deltas, segment_lengths, counts, strict=True)
    ):
        source = np.asarray(left.values, dtype=float)
        first_local_index = 0 if segment == 0 else 1
        for local_index in range(first_local_index, count):
            fraction = float(local_index) / float(count - 1)
            values = source + fraction * delta
            for coordinate_index, period in periodic_map.items():
                values[coordinate_index] = (
                    values[coordinate_index] + 0.5 * period
                ) % period - 0.5 * period
            landmark_label = None
            if local_index == 0:
                landmark_label = left.label
            elif local_index == count - 1:
                landmark_label = right.label
            images.append(
                SonicRingPathImage(
                    index=len(images),
                    segment=segment,
                    fraction=fraction,
                    arc_fraction=(traversed + fraction * length) / total_length,
                    values=tuple(float(value) for value in values),
                    left_label=left.label,
                    right_label=right.label,
                    landmark_label=landmark_label,
                )
            )
        traversed += length
    return tuple(images)
