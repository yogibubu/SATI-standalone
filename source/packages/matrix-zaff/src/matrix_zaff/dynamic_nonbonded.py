"""Persistent hierarchical electrostatics and cell pair lists for local moves."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from collections.abc import Callable

from numba import njit
import numpy as np


@dataclass(frozen=True)
class AccelerationAudit:
    """Fail-closed comparison of an accelerated evaluator with its reference."""

    sample_count: int
    maximum_absolute_error: float
    rms_error: float
    accelerated_seconds: float
    reference_seconds: float
    measured_speedup: float
    unit: str

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "sample_count": self.sample_count,
            "maximum_absolute_error": self.maximum_absolute_error,
            "rms_error": self.rms_error,
            "accelerated_seconds": self.accelerated_seconds,
            "reference_seconds": self.reference_seconds,
            "measured_speedup": self.measured_speedup,
            "unit": self.unit,
        }


@dataclass(frozen=True)
class ConsistencyAudit:
    """Comparison of incrementally maintained and exactly rebuilt state."""

    value_count: int
    maximum_absolute_error: float
    maximum_relative_error: float
    resynchronized: bool
    unit: str

    def as_dict(self) -> dict[str, float | int | str | bool]:
        return {
            "value_count": self.value_count,
            "maximum_absolute_error": self.maximum_absolute_error,
            "maximum_relative_error": self.maximum_relative_error,
            "resynchronized": self.resynchronized,
            "unit": self.unit,
        }


def audit_accelerated_reference(
    accelerated: Callable[[], np.ndarray],
    reference: Callable[[], np.ndarray],
    *,
    maximum_error: float,
    minimum_speedup: float = 0.0,
    repeats: int = 3,
    unit: str = "",
) -> AccelerationAudit:
    """Warm, time, and validate two batch evaluators under one public contract.

    The callbacks must be side-effect free and return identically shaped finite
    arrays.  A failed accuracy or speed gate raises immediately, so production
    cannot silently continue on an invalid accelerated backend.
    """

    if not math.isfinite(maximum_error) or maximum_error < 0.0:
        raise ValueError("acceleration-audit maximum error must be finite and nonnegative")
    if not math.isfinite(minimum_speedup) or minimum_speedup < 0.0:
        raise ValueError("acceleration-audit minimum speedup must be finite and nonnegative")
    if int(repeats) != repeats or int(repeats) < 1:
        raise ValueError("acceleration-audit repeat count must be a positive integer")
    accelerated()
    reference()
    accelerated_timings: list[float] = []
    reference_timings: list[float] = []
    accelerated_values = np.empty(0)
    reference_values = np.empty(0)
    for _ in range(int(repeats)):
        start = time.perf_counter()
        accelerated_values = np.asarray(accelerated(), dtype=float)
        accelerated_timings.append(time.perf_counter() - start)
        start = time.perf_counter()
        reference_values = np.asarray(reference(), dtype=float)
        reference_timings.append(time.perf_counter() - start)
    if accelerated_values.shape != reference_values.shape:
        raise RuntimeError("accelerated and reference evaluators returned different shapes")
    if np.any(~np.isfinite(accelerated_values)) or np.any(~np.isfinite(reference_values)):
        raise RuntimeError("acceleration audit received non-finite evaluator output")
    error = accelerated_values - reference_values
    maximum_absolute_error = float(np.max(np.abs(error), initial=0.0))
    rms_error = float(np.sqrt(np.mean(error**2))) if error.size else 0.0
    accelerated_seconds = float(np.median(accelerated_timings))
    reference_seconds = float(np.median(reference_timings))
    measured_speedup = reference_seconds / max(accelerated_seconds, 1.0e-15)
    if maximum_absolute_error > maximum_error:
        raise RuntimeError(
            "accelerated evaluator failed its direct-reference audit: "
            f"{maximum_absolute_error:.6g} > {maximum_error:.6g} {unit}".rstrip()
        )
    if minimum_speedup and measured_speedup < minimum_speedup:
        raise RuntimeError(
            "accelerated evaluator failed its speed gate: "
            f"{measured_speedup:.3f} < {minimum_speedup:.3f}"
        )
    return AccelerationAudit(
        sample_count=int(accelerated_values.size),
        maximum_absolute_error=maximum_absolute_error,
        rms_error=rms_error,
        accelerated_seconds=accelerated_seconds,
        reference_seconds=reference_seconds,
        measured_speedup=measured_speedup,
        unit=str(unit),
    )


def audit_incremental_state(
    incremental: np.ndarray,
    exact: np.ndarray,
    *,
    maximum_absolute_error: float = math.inf,
    maximum_relative_error: float = math.inf,
    resynchronize: Callable[[np.ndarray], None] | None = None,
    unit: str = "",
) -> ConsistencyAudit:
    """Validate incremental state against an exact rebuild and optionally reset it."""

    approximate = np.asarray(incremental, dtype=float)
    rebuilt = np.asarray(exact, dtype=float)
    if approximate.shape != rebuilt.shape:
        raise ValueError("incremental and exact states must have identical shapes")
    if np.any(~np.isfinite(approximate)) or np.any(~np.isfinite(rebuilt)):
        raise RuntimeError("state-consistency audit received non-finite values")
    if maximum_absolute_error < 0.0 or maximum_relative_error < 0.0:
        raise ValueError("state-consistency tolerances must be nonnegative")
    difference = approximate - rebuilt
    absolute_error = float(np.max(np.abs(difference), initial=0.0))
    relative_error = float(
        np.max(
            np.abs(difference) / np.maximum(1.0, np.abs(rebuilt)),
            initial=0.0,
        )
    )
    if absolute_error > maximum_absolute_error:
        raise RuntimeError(
            "incremental state failed its absolute consistency gate: "
            f"{absolute_error:.6g} > {maximum_absolute_error:.6g} {unit}".rstrip()
        )
    if relative_error > maximum_relative_error:
        raise RuntimeError(
            "incremental state failed its relative consistency gate: "
            f"{relative_error:.6g} > {maximum_relative_error:.6g}"
        )
    if resynchronize is not None:
        resynchronize(rebuilt.copy())
    return ConsistencyAudit(
        value_count=int(rebuilt.size),
        maximum_absolute_error=absolute_error,
        maximum_relative_error=relative_error,
        resynchronized=resynchronize is not None,
        unit=str(unit),
    )


@njit(cache=True)
def _fmm_potential_kernel(
    targets: np.ndarray,
    excluded_group: int,
    coordinates: np.ndarray,
    charges: np.ndarray,
    node_centers: np.ndarray,
    node_source_radius: np.ndarray,
    node_children: np.ndarray,
    leaf_group_offsets: np.ndarray,
    leaf_groups: np.ndarray,
    excluded_ancestors: np.ndarray,
    node_q: np.ndarray,
    node_p: np.ndarray,
    node_m2: np.ndarray,
    node_m3: np.ndarray,
    node_s3: np.ndarray,
    node_m4: np.ndarray,
    node_v4: np.ndarray,
    node_w4: np.ndarray,
    expansion_order: int,
    opening_angle: float,
) -> np.ndarray:
    values = np.zeros(len(targets))
    stack = np.empty(len(node_centers), dtype=np.int64)
    for target_index in range(len(targets)):
        stack_size = 1
        stack[0] = 0
        potential = 0.0
        while stack_size:
            stack_size -= 1
            node = stack[stack_size]
            dx = targets[target_index, 0] - node_centers[node, 0]
            dy = targets[target_index, 1] - node_centers[node, 1]
            dz = targets[target_index, 2] - node_centers[node, 2]
            radius2 = dx * dx + dy * dy + dz * dz
            radius = math.sqrt(radius2)
            admissible = (
                not excluded_ancestors[node]
                and radius > 1.0e-14
                and node_source_radius[node] <= opening_angle * radius
            )
            if admissible:
                inverse = 1.0 / radius
                inverse2 = inverse * inverse
                potential += node_q[node] * inverse
                potential += (
                    node_p[node, 0] * dx
                    + node_p[node, 1] * dy
                    + node_p[node, 2] * dz
                ) * inverse * inverse2
                if expansion_order >= 2:
                    contraction2 = 0.0
                    trace2 = 0.0
                    displacement = (dx, dy, dz)
                    for left in range(3):
                        trace2 += node_m2[node, left, left]
                        for right in range(3):
                            contraction2 += (
                                displacement[left]
                                * node_m2[node, left, right]
                                * displacement[right]
                            )
                    potential += 0.5 * (
                        3.0 * contraction2 * inverse**5
                        - trace2 * inverse**3
                    )
                if expansion_order >= 3:
                    contraction3 = 0.0
                    displacement = (dx, dy, dz)
                    for a in range(3):
                        for b in range(3):
                            for c in range(3):
                                contraction3 += (
                                    displacement[a]
                                    * displacement[b]
                                    * displacement[c]
                                    * node_m3[node, a, b, c]
                                )
                    trace3 = (
                        node_s3[node, 0] * dx
                        + node_s3[node, 1] * dy
                        + node_s3[node, 2] * dz
                    )
                    potential += (
                        2.5 * contraction3 * inverse**7
                        - 1.5 * trace3 * inverse**5
                    )
                if expansion_order >= 4:
                    contraction4 = 0.0
                    trace4 = 0.0
                    displacement = (dx, dy, dz)
                    for a in range(3):
                        for b in range(3):
                            trace4 += (
                                displacement[a]
                                * node_v4[node, a, b]
                                * displacement[b]
                            )
                            for c in range(3):
                                for d in range(3):
                                    contraction4 += (
                                        displacement[a]
                                        * displacement[b]
                                        * displacement[c]
                                        * displacement[d]
                                        * node_m4[node, a, b, c, d]
                                    )
                    potential += 0.125 * (
                        35.0 * contraction4 * inverse**9
                        - 30.0 * trace4 * inverse**7
                        + 3.0 * node_w4[node] * inverse**5
                    )
                continue
            has_children = node_children[node, 0] >= 0
            if has_children:
                for child_slot in range(8):
                    child = node_children[node, child_slot]
                    if child < 0:
                        break
                    stack[stack_size] = child
                    stack_size += 1
                continue
            for flat_index in range(
                leaf_group_offsets[node], leaf_group_offsets[node + 1]
            ):
                group = leaf_groups[flat_index]
                if group == excluded_group:
                    continue
                for site in range(coordinates.shape[1]):
                    sx = targets[target_index, 0] - coordinates[group, site, 0]
                    sy = targets[target_index, 1] - coordinates[group, site, 1]
                    sz = targets[target_index, 2] - coordinates[group, site, 2]
                    distance = math.sqrt(sx * sx + sy * sy + sz * sz)
                    potential += charges[site] / distance
        values[target_index] = potential
    return values


@njit(cache=True)
def _fmm_potential_gradient_kernel(
    targets: np.ndarray,
    excluded_group: int,
    coordinates: np.ndarray,
    charges: np.ndarray,
    node_centers: np.ndarray,
    node_source_radius: np.ndarray,
    node_children: np.ndarray,
    leaf_group_offsets: np.ndarray,
    leaf_groups: np.ndarray,
    excluded_ancestors: np.ndarray,
    node_q: np.ndarray,
    node_p: np.ndarray,
    node_m2: np.ndarray,
    node_m3: np.ndarray,
    node_s3: np.ndarray,
    node_m4: np.ndarray,
    node_v4: np.ndarray,
    node_w4: np.ndarray,
    expansion_order: int,
    opening_angle: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate potential and its analytic target gradient."""

    values = np.zeros(len(targets))
    gradients = np.zeros((len(targets), 3))
    stack = np.empty(len(node_centers), dtype=np.int64)
    for target_index in range(len(targets)):
        stack_size = 1
        stack[0] = 0
        potential = 0.0
        gx = 0.0
        gy = 0.0
        gz = 0.0
        while stack_size:
            stack_size -= 1
            node = stack[stack_size]
            dx = targets[target_index, 0] - node_centers[node, 0]
            dy = targets[target_index, 1] - node_centers[node, 1]
            dz = targets[target_index, 2] - node_centers[node, 2]
            radius2 = dx * dx + dy * dy + dz * dz
            radius = math.sqrt(radius2)
            admissible = (
                not excluded_ancestors[node]
                and radius > 1.0e-14
                and node_source_radius[node] <= opening_angle * radius
            )
            if admissible:
                displacement = (dx, dy, dz)
                inverse = 1.0 / radius
                inverse2 = inverse * inverse
                inverse3 = inverse * inverse2
                potential += node_q[node] * inverse
                for axis in range(3):
                    gradients[target_index, axis] -= (
                        node_q[node] * displacement[axis] * inverse3
                    )
                contraction1 = (
                    node_p[node, 0] * dx
                    + node_p[node, 1] * dy
                    + node_p[node, 2] * dz
                )
                potential += contraction1 * inverse3
                inverse5 = inverse3 * inverse2
                for axis in range(3):
                    gradients[target_index, axis] += (
                        node_p[node, axis] * inverse3
                        - 3.0 * contraction1 * displacement[axis] * inverse5
                    )
                if expansion_order >= 2:
                    contraction2 = 0.0
                    trace2 = 0.0
                    m2d = np.zeros(3)
                    for left in range(3):
                        trace2 += node_m2[node, left, left]
                        for right in range(3):
                            m2d[left] += (
                                node_m2[node, left, right] * displacement[right]
                            )
                            contraction2 += (
                                displacement[left]
                                * node_m2[node, left, right]
                                * displacement[right]
                            )
                    potential += 0.5 * (
                        3.0 * contraction2 * inverse5 - trace2 * inverse3
                    )
                    inverse7 = inverse5 * inverse2
                    for axis in range(3):
                        gradients[target_index, axis] += (
                            3.0 * m2d[axis] * inverse5
                            - 7.5 * contraction2 * displacement[axis] * inverse7
                            + 1.5 * trace2 * displacement[axis] * inverse5
                        )
                if expansion_order >= 3:
                    contraction3 = 0.0
                    m3dd = np.zeros(3)
                    for a in range(3):
                        for b in range(3):
                            for c in range(3):
                                value = node_m3[node, a, b, c]
                                contraction3 += (
                                    displacement[a]
                                    * displacement[b]
                                    * displacement[c]
                                    * value
                                )
                                m3dd[a] += (
                                    value * displacement[b] * displacement[c]
                                )
                    trace3 = (
                        node_s3[node, 0] * dx
                        + node_s3[node, 1] * dy
                        + node_s3[node, 2] * dz
                    )
                    inverse7 = inverse5 * inverse2
                    inverse9 = inverse7 * inverse2
                    potential += (
                        2.5 * contraction3 * inverse7
                        - 1.5 * trace3 * inverse5
                    )
                    for axis in range(3):
                        gradients[target_index, axis] += (
                            7.5 * m3dd[axis] * inverse7
                            - 17.5 * contraction3 * displacement[axis] * inverse9
                            - 1.5 * node_s3[node, axis] * inverse5
                            + 7.5 * trace3 * displacement[axis] * inverse7
                        )
                if expansion_order >= 4:
                    contraction4 = 0.0
                    trace4 = 0.0
                    m4ddd = np.zeros(3)
                    v4d = np.zeros(3)
                    for a in range(3):
                        for b in range(3):
                            v4d[a] += node_v4[node, a, b] * displacement[b]
                            trace4 += (
                                displacement[a]
                                * node_v4[node, a, b]
                                * displacement[b]
                            )
                            for c in range(3):
                                for d in range(3):
                                    value = node_m4[node, a, b, c, d]
                                    contraction4 += (
                                        displacement[a]
                                        * displacement[b]
                                        * displacement[c]
                                        * displacement[d]
                                        * value
                                    )
                                    m4ddd[a] += (
                                        value
                                        * displacement[b]
                                        * displacement[c]
                                        * displacement[d]
                                    )
                    inverse7 = inverse5 * inverse2
                    inverse9 = inverse7 * inverse2
                    inverse11 = inverse9 * inverse2
                    potential += 0.125 * (
                        35.0 * contraction4 * inverse9
                        - 30.0 * trace4 * inverse7
                        + 3.0 * node_w4[node] * inverse5
                    )
                    for axis in range(3):
                        gradients[target_index, axis] += 0.125 * (
                            140.0 * m4ddd[axis] * inverse9
                            - 315.0
                            * contraction4
                            * displacement[axis]
                            * inverse11
                            - 60.0 * v4d[axis] * inverse7
                            + 210.0
                            * trace4
                            * displacement[axis]
                            * inverse9
                            - 15.0
                            * node_w4[node]
                            * displacement[axis]
                            * inverse7
                        )
                continue
            has_children = node_children[node, 0] >= 0
            if has_children:
                for child_slot in range(8):
                    child = node_children[node, child_slot]
                    if child < 0:
                        break
                    stack[stack_size] = child
                    stack_size += 1
                continue
            for flat_index in range(
                leaf_group_offsets[node], leaf_group_offsets[node + 1]
            ):
                group = leaf_groups[flat_index]
                if group == excluded_group:
                    continue
                for site in range(coordinates.shape[1]):
                    sx = targets[target_index, 0] - coordinates[group, site, 0]
                    sy = targets[target_index, 1] - coordinates[group, site, 1]
                    sz = targets[target_index, 2] - coordinates[group, site, 2]
                    distance2 = sx * sx + sy * sy + sz * sz
                    inverse = 1.0 / math.sqrt(distance2)
                    potential += charges[site] * inverse
                    scale = -charges[site] * inverse / distance2
                    gx += scale * sx
                    gy += scale * sy
                    gz += scale * sz
        values[target_index] = potential
        gradients[target_index, 0] += gx
        gradients[target_index, 1] += gy
        gradients[target_index, 2] += gz
    return values, gradients


@dataclass
class _MultipoleNode:
    center: np.ndarray
    half_width: float
    group_indices: np.ndarray
    children: tuple[int, ...]
    parent: int | None
    moments: dict[str, np.ndarray | float]
    source_radius: float


class DynamicMolecularFMM:
    """Dynamic P2M/M2P Laplace hierarchy for fixed-charge molecular groups.

    The source tree persists across local Monte Carlo moves.  An accepted
    molecule updates only the multipoles on one root-to-leaf path; crossing a
    leaf boundary triggers a deterministic rebuild.  Near cells are evaluated
    directly, while admissible far cells use spherical multipoles.  This is
    the local-target specialization of an FMM and avoids rebuilding a global
    black-box FMM for the three targets of one rigid-water proposal.
    """

    def __init__(
        self,
        coordinates: np.ndarray,
        charges: np.ndarray,
        *,
        expansion_order: int = 4,
        opening_angle: float = 0.42,
        leaf_size: int = 16,
    ) -> None:
        xyz = np.asarray(coordinates, dtype=float)
        q = np.asarray(charges, dtype=float).reshape(-1)
        if (
            xyz.ndim != 3
            or xyz.shape[2] != 3
            or xyz.shape[1] != len(q)
            or np.any(~np.isfinite(xyz))
            or np.any(~np.isfinite(q))
        ):
            raise ValueError("dynamic FMM needs finite (groups, sites, 3) coordinates")
        if (
            int(expansion_order) < 1
            or int(expansion_order) > 4
            or int(expansion_order) != expansion_order
        ):
            raise ValueError("dynamic FMM expansion order must be an integer from 1 to 4")
        if not 0.0 < float(opening_angle) < 1.0:
            raise ValueError("dynamic FMM opening angle must lie between zero and one")
        if int(leaf_size) < 2:
            raise ValueError("dynamic FMM leaf size must be at least two")
        self.coordinates = xyz.copy()
        self.charges = q.copy()
        self.expansion_order = int(expansion_order)
        self.opening_angle = float(opening_angle)
        self.leaf_size = int(leaf_size)
        self.nodes: list[_MultipoleNode] = []
        self.group_leaf = np.empty(len(xyz), dtype=int)
        self.group_ancestors: list[frozenset[int]] = []
        self.levels: tuple[np.ndarray, ...] = ()
        self.rebuild_count = 0
        self.local_update_count = 0
        self._build()

    @property
    def backend(self) -> str:
        return (
            f"DYNAMIC_FMM_P{self.expansion_order}_THETA"
            f"{self.opening_angle:.3f}_LEAF{self.leaf_size}"
        )

    def _build(self) -> None:
        centers = self.coordinates[:, 0]
        lower = np.min(centers, axis=0)
        upper = np.max(centers, axis=0)
        root_center = 0.5 * (lower + upper)
        half_width = max(0.5 * float(np.max(upper - lower)) * 1.000001, 1.0e-6)
        self.nodes = []

        def recurse(indices: np.ndarray, center: np.ndarray, width: float, parent: int | None) -> int:
            node_index = len(self.nodes)
            self.nodes.append(
                _MultipoleNode(
                    center=center.copy(),
                    half_width=float(width),
                    group_indices=indices.copy(),
                    children=(),
                    parent=parent,
                    moments={},
                    source_radius=0.0,
                )
            )
            children: list[int] = []
            if len(indices) > self.leaf_size and width > 1.0e-8:
                codes = np.sum(
                    (centers[indices] >= center).astype(int)
                    * np.asarray((1, 2, 4))[None, :],
                    axis=1,
                )
                child_width = 0.5 * width
                for code in range(8):
                    selected = indices[codes == code]
                    if not len(selected):
                        continue
                    sign = np.asarray(
                        (
                            1.0 if code & 1 else -1.0,
                            1.0 if code & 2 else -1.0,
                            1.0 if code & 4 else -1.0,
                        )
                    )
                    children.append(
                        recurse(
                            selected,
                            center + sign * child_width,
                            child_width,
                            node_index,
                        )
                    )
            node = self.nodes[node_index]
            node.children = tuple(children)
            node.moments, node.source_radius = self._moments(indices, node.center)
            if not children:
                self.group_leaf[indices] = node_index
            return node_index

        recurse(np.arange(len(self.coordinates), dtype=int), root_center, half_width, None)
        self.group_ancestors = []
        for group in range(len(self.coordinates)):
            ancestors = []
            node_index: int | None = int(self.group_leaf[group])
            while node_index is not None:
                ancestors.append(node_index)
                node_index = self.nodes[node_index].parent
            self.group_ancestors.append(frozenset(ancestors))
        depth = np.zeros(len(self.nodes), dtype=int)
        for index, node in enumerate(self.nodes):
            if node.parent is not None:
                depth[index] = depth[node.parent] + 1
        self.levels = tuple(
            np.flatnonzero(depth == level)
            for level in range(int(np.max(depth, initial=0)) + 1)
        )
        self._pack_moments()
        self.rebuild_count += 1

    def _pack_moments(self) -> None:
        count = len(self.nodes)
        self.node_centers = np.asarray([node.center for node in self.nodes])
        self.node_source_radius = np.asarray(
            [node.source_radius for node in self.nodes]
        )
        self.node_q = np.asarray([node.moments["q"] for node in self.nodes])
        self.node_p = np.asarray([node.moments["p"] for node in self.nodes])
        self.node_m2 = np.zeros((count, 3, 3))
        self.node_m3 = np.zeros((count, 3, 3, 3))
        self.node_s3 = np.zeros((count, 3))
        self.node_m4 = np.zeros((count, 3, 3, 3, 3))
        self.node_v4 = np.zeros((count, 3, 3))
        self.node_w4 = np.zeros(count)
        if self.expansion_order >= 2:
            self.node_m2 = np.asarray([node.moments["m2"] for node in self.nodes])
        if self.expansion_order >= 3:
            self.node_m3 = np.asarray([node.moments["m3"] for node in self.nodes])
            self.node_s3 = np.asarray([node.moments["s3"] for node in self.nodes])
        if self.expansion_order >= 4:
            self.node_m4 = np.asarray([node.moments["m4"] for node in self.nodes])
            self.node_v4 = np.asarray([node.moments["v4"] for node in self.nodes])
            self.node_w4 = np.asarray([node.moments["w4"] for node in self.nodes])
        self.node_children = np.full((count, 8), -1, dtype=np.int64)
        flat_groups: list[int] = []
        self.leaf_group_offsets = np.zeros(count + 1, dtype=np.int64)
        for index, node in enumerate(self.nodes):
            if node.children:
                self.node_children[index, : len(node.children)] = node.children
            else:
                flat_groups.extend(int(value) for value in node.group_indices)
            self.leaf_group_offsets[index + 1] = len(flat_groups)
        self.leaf_groups = np.asarray(flat_groups, dtype=np.int64)

    def _moments(
        self, indices: np.ndarray, center: np.ndarray
    ) -> tuple[dict[str, np.ndarray | float], float]:
        sites = self.coordinates[indices].reshape(-1, 3)
        charges = np.tile(self.charges, len(indices))
        relative = sites - center
        radius = np.linalg.norm(relative, axis=1)
        return self._cartesian_moments(relative, charges), float(
            np.max(radius, initial=0.0)
        )

    def _cartesian_moments(
        self,
        relative: np.ndarray,
        charges: np.ndarray,
    ) -> dict[str, np.ndarray | float]:
        moments: dict[str, np.ndarray | float] = {
            "q": float(np.sum(charges)),
            "p": np.einsum("n,na->a", charges, relative),
        }
        if self.expansion_order >= 2:
            moments["m2"] = np.einsum(
                "n,na,nb->ab", charges, relative, relative
            )
        if self.expansion_order >= 3:
            radius2 = np.einsum("na,na->n", relative, relative)
            moments["m3"] = np.einsum(
                "n,na,nb,nc->abc", charges, relative, relative, relative
            )
            moments["s3"] = np.einsum("n,na,n->a", charges, relative, radius2)
        if self.expansion_order >= 4:
            radius2 = np.einsum("na,na->n", relative, relative)
            moments["m4"] = np.einsum(
                "n,na,nb,nc,nd->abcd",
                charges,
                relative,
                relative,
                relative,
                relative,
            )
            moments["v4"] = np.einsum(
                "n,na,nb,n->ab", charges, relative, relative, radius2
            )
            moments["w4"] = float(np.sum(charges * radius2**2))
        return moments

    def potential(
        self,
        targets: np.ndarray,
        *,
        excluded_group: int | None = None,
    ) -> np.ndarray:
        """Return the point-charge potential at targets in inverse-length units."""

        target_array = np.asarray(targets, dtype=float).reshape(-1, 3)
        if np.any(~np.isfinite(target_array)):
            raise ValueError("dynamic FMM targets must be finite")
        group = -1 if excluded_group is None else int(excluded_group)
        if group >= len(self.coordinates) or group < -1:
            raise IndexError("excluded FMM group is out of range")
        excluded_ancestors = np.zeros(len(self.nodes), dtype=np.bool_)
        if group >= 0:
            excluded_ancestors[
                np.fromiter(self.group_ancestors[group], dtype=np.int64)
            ] = True
        return _fmm_potential_kernel(
            target_array,
            group,
            self.coordinates,
            self.charges,
            self.node_centers,
            self.node_source_radius,
            self.node_children,
            self.leaf_group_offsets,
            self.leaf_groups,
            excluded_ancestors,
            self.node_q,
            self.node_p,
            self.node_m2,
            self.node_m3,
            self.node_s3,
            self.node_m4,
            self.node_v4,
            self.node_w4,
            self.expansion_order,
            self.opening_angle,
        )

    def potential_gradient(
        self,
        targets: np.ndarray,
        *,
        excluded_group: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return potential and analytic target gradient in inverse-length units."""

        target_array = np.asarray(targets, dtype=float).reshape(-1, 3)
        if np.any(~np.isfinite(target_array)):
            raise ValueError("dynamic FMM targets must be finite")
        group = -1 if excluded_group is None else int(excluded_group)
        if group >= len(self.coordinates) or group < -1:
            raise IndexError("excluded FMM group is out of range")
        excluded_ancestors = np.zeros(len(self.nodes), dtype=np.bool_)
        if group >= 0:
            excluded_ancestors[
                np.fromiter(self.group_ancestors[group], dtype=np.int64)
            ] = True
        return _fmm_potential_gradient_kernel(
            target_array,
            group,
            self.coordinates,
            self.charges,
            self.node_centers,
            self.node_source_radius,
            self.node_children,
            self.leaf_group_offsets,
            self.leaf_groups,
            excluded_ancestors,
            self.node_q,
            self.node_p,
            self.node_m2,
            self.node_m3,
            self.node_s3,
            self.node_m4,
            self.node_v4,
            self.node_w4,
            self.expansion_order,
            self.opening_angle,
        )

    def interaction_energy(
        self,
        molecule: np.ndarray,
        group_index: int,
    ) -> float:
        """Return one molecule's electrostatic interaction with all other groups."""

        geometry = np.asarray(molecule, dtype=float).reshape(len(self.charges), 3)
        return float(
            np.dot(
                self.charges,
                self.potential(geometry, excluded_group=int(group_index)),
            )
        )

    def interaction_energy_difference(
        self,
        candidate: np.ndarray,
        reference: np.ndarray,
        group_index: int,
    ) -> float:
        """Return a local move difference in one hierarchical traversal call."""

        candidate_geometry = np.asarray(candidate, dtype=float).reshape(
            len(self.charges),
            3,
        )
        reference_geometry = np.asarray(reference, dtype=float).reshape(
            len(self.charges),
            3,
        )
        potentials = self.potential(
            np.concatenate((candidate_geometry, reference_geometry), axis=0),
            excluded_group=int(group_index),
        ).reshape(2, len(self.charges))
        return float(np.dot(self.charges, potentials[0] - potentials[1]))

    def interaction_energy_gradient(
        self,
        molecule: np.ndarray,
        group_index: int,
    ) -> tuple[float, np.ndarray]:
        """Return molecular interaction energy and analytic site gradient."""

        geometry = np.asarray(molecule, dtype=float).reshape(len(self.charges), 3)
        potential, potential_gradient = self.potential_gradient(
            geometry,
            excluded_group=int(group_index),
        )
        return (
            float(np.dot(self.charges, potential)),
            self.charges[:, None] * potential_gradient,
        )

    def update_group(self, group_index: int, coordinates: np.ndarray) -> None:
        """Commit one accepted molecular move with an O(log N) moment update."""

        group = int(group_index)
        new_coordinates = np.asarray(coordinates, dtype=float).reshape(
            len(self.charges), 3
        )
        old_coordinates = self.coordinates[group].copy()
        old_leaf = int(self.group_leaf[group])
        leaf = self.nodes[old_leaf]
        center = new_coordinates[0]
        if np.any(np.abs(center - leaf.center) > leaf.half_width):
            self.coordinates[group] = new_coordinates
            self._build()
            return
        self.coordinates[group] = new_coordinates
        for node_index in self.group_ancestors[group]:
            node = self.nodes[node_index]
            for sign, geometry in ((-1.0, old_coordinates), (1.0, new_coordinates)):
                relative = geometry - node.center
                delta = self._cartesian_moments(relative, self.charges)
                for name, value in delta.items():
                    node.moments[name] += sign * value
            node.source_radius = max(
                node.source_radius,
                float(np.max(np.linalg.norm(new_coordinates - node.center, axis=1))),
            )
            self.node_q[node_index] = node.moments["q"]
            self.node_p[node_index] = node.moments["p"]
            if self.expansion_order >= 2:
                self.node_m2[node_index] = node.moments["m2"]
            if self.expansion_order >= 3:
                self.node_m3[node_index] = node.moments["m3"]
                self.node_s3[node_index] = node.moments["s3"]
            if self.expansion_order >= 4:
                self.node_m4[node_index] = node.moments["m4"]
                self.node_v4[node_index] = node.moments["v4"]
                self.node_w4[node_index] = node.moments["w4"]
            self.node_source_radius[node_index] = node.source_radius
        self.local_update_count += 1

    def reset_coordinates(self, coordinates: np.ndarray) -> None:
        """Replace every group coordinate and deterministically rebuild the tree."""

        xyz = np.asarray(coordinates, dtype=float)
        if (
            xyz.shape != self.coordinates.shape
            or np.any(~np.isfinite(xyz))
        ):
            raise ValueError("replacement FMM coordinates are inconsistent")
        self.coordinates = xyz.copy()
        self._build()


class DynamicCenterPairList:
    """Persistent cell pair list for switched short-range center interactions."""

    def __init__(
        self,
        centers: np.ndarray,
        *,
        cutoff: float,
    ) -> None:
        xyz = np.asarray(centers, dtype=float).reshape(-1, 3)
        if not math.isfinite(float(cutoff)) or float(cutoff) <= 0.0:
            raise ValueError("pair-list cutoff must be finite and positive")
        self.centers = xyz.copy()
        self.cutoff = float(cutoff)
        self.cells: dict[tuple[int, int, int], set[int]] = {}
        self.group_cell: list[tuple[int, int, int]] = []
        for index, center in enumerate(self.centers):
            cell = self._cell(center)
            self.group_cell.append(cell)
            self.cells.setdefault(cell, set()).add(index)

    def _cell(self, center: np.ndarray) -> tuple[int, int, int]:
        return tuple(np.floor(np.asarray(center) / self.cutoff).astype(int))

    def candidates(self, center: np.ndarray, *, excluded_group: int) -> np.ndarray:
        home = self._cell(center)
        candidates: set[int] = set()
        for x in (-1, 0, 1):
            for y in (-1, 0, 1):
                for z in (-1, 0, 1):
                    candidates.update(
                        self.cells.get(
                            (home[0] + x, home[1] + y, home[2] + z),
                            (),
                        )
                    )
        candidates.discard(int(excluded_group))
        if not candidates:
            return np.empty(0, dtype=int)
        return np.asarray(sorted(candidates), dtype=int)

    def update_group(self, group_index: int, center: np.ndarray) -> None:
        group = int(group_index)
        new_center = np.asarray(center, dtype=float).reshape(3)
        old_cell = self.group_cell[group]
        new_cell = self._cell(new_center)
        if new_cell != old_cell:
            self.cells[old_cell].remove(group)
            if not self.cells[old_cell]:
                del self.cells[old_cell]
            self.cells.setdefault(new_cell, set()).add(group)
            self.group_cell[group] = new_cell
        self.centers[group] = new_center

    def reset_centers(self, centers: np.ndarray) -> None:
        """Replace all centers and rebuild the cell membership."""

        xyz = np.asarray(centers, dtype=float).reshape(-1, 3)
        if xyz.shape != self.centers.shape or np.any(~np.isfinite(xyz)):
            raise ValueError("replacement pair-list centers are inconsistent")
        self.centers = xyz.copy()
        self.cells = {}
        self.group_cell = []
        for index, center in enumerate(self.centers):
            cell = self._cell(center)
            self.group_cell.append(cell)
            self.cells.setdefault(cell, set()).add(index)


__all__ = [
    "AccelerationAudit",
    "ConsistencyAudit",
    "DynamicCenterPairList",
    "DynamicMolecularFMM",
    "audit_accelerated_reference",
    "audit_incremental_state",
]
