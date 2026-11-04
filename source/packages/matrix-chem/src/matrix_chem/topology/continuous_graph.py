"""
continuous_graph.py
===================

Continuous geometry-based molecular descriptors.

Provides:
- Continuous Coordination Number (CNA)
- Coordination-dependent covalent radii (Pyykkö)
- a continuous connectivity weight used only to perceive the graph
- one chemical bond order: Mayer when supplied, otherwise Pauling

All quantities are continuous and suitable for differentiation.
"""

from dataclasses import dataclass
from functools import lru_cache
from bisect import bisect_right
import math
import numpy as np

from ..spatial_regions import SpatialRegions
from ..native_topology import compiled_continuous_graph, native_topology_backend
from .gpu_perception import gpu_screen_candidate_pairs
from .descriptor_parameters import (
    CNA_ALPHA,
    BO_LAMBDA_STRONG,
    BO_LAMBDA_WEAK,
    ALPHA_LAMBDA,
    BO_COMPONENT_RAMP_WIDTH,
    BO_PAULING_DECAY_ANGSTROM,
)
from .pykko_radii import PYYKKO
from .covalent_radii import covalent_radius as standard_rcov
from .vdw_radii import uff_vdw_radius

DISCRETE_DISTANCE_SCALE = 1.25
_MAX_ATOMIC_NUMBER = 118
_STANDARD_RADII = np.asarray(
    [
        np.nan
        if (radius := standard_rcov(atomic_number)) is None
        else float(radius)
        for atomic_number in range(_MAX_ATOMIC_NUMBER + 1)
    ],
    dtype=float,
)
_PYYKKO_TABLES = tuple(
    (
        tuple(sorted(PYYKKO.get(atomic_number, {}))),
        tuple(
            float(PYYKKO[atomic_number][coordination])
            for coordination in sorted(PYYKKO.get(atomic_number, {}))
        ),
    )
    for atomic_number in range(_MAX_ATOMIC_NUMBER + 1)
)
_PYYKKO_MAX_COORDINATION = max(
    (
        coordination
        for table in PYYKKO.values()
        for coordination in table
    ),
    default=0,
)
_PYYKKO_DENSE = np.full(
    (_MAX_ATOMIC_NUMBER + 1, _PYYKKO_MAX_COORDINATION + 1),
    np.nan,
    dtype=float,
)
for _atomic_number, _table in PYYKKO.items():
    for _coordination, _radius in _table.items():
        _PYYKKO_DENSE[int(_atomic_number), int(_coordination)] = float(_radius)


@lru_cache(maxsize=64)
def _upper_triangle_indices(natoms: int) -> tuple[np.ndarray, np.ndarray]:
    """Cache immutable all-pairs indices for repeated small-molecule builds."""

    left, right = np.triu_indices(int(natoms), k=1)
    left.setflags(write=False)
    right.setflags(write=False)
    return left, right


class ContinuousGraph:
    """
    Minimal continuous graph used by the pipeline.

    ``CONNECTIVITY`` is a geometric graph-perception weight, not a second bond
    order. ``BO`` is the sole chemical bond-order observable exposed to
    synthons and downstream tools.
    """

    def __init__(
        self,
        coords,
        Z,
        *,
        bond_order_overrides=None,
        connectivity_overrides=None,
        candidate_pairs_override=None,
    ):
        self.coords = np.array(coords, dtype=float, copy=True)
        self.Z = np.array(Z, dtype=int, copy=True)
        self.natoms = len(self.Z)
        self.bond_order_overrides = bond_order_overrides or {}
        self.connectivity_overrides = connectivity_overrides or {}

        standard_radii = _standard_radii_for_atomic_numbers(self.Z)
        finite_radii = standard_radii[np.isfinite(standard_radii)]
        maximum_radius = float(np.max(finite_radii)) if finite_radii.size else 2.5
        # At this margin the omitted CNA term is below double-precision
        # descriptor significance: 0.5*(1+erf(-6)) < 1.1e-17.
        self.local_cutoff_angstrom = 2.0 * maximum_radius + 0.75
        self.pair_screening_backend = "numpy"
        native_graph = (
            native_topology_backend(self.natoms).accelerated
            and self.natoms <= 256
            and not self.bond_order_overrides
            and not self.connectivity_overrides
            and candidate_pairs_override is None
        )
        native_result = None
        if native_graph:
            native_result = compiled_continuous_graph(
                self.coords,
                self.Z,
                _STANDARD_RADII,
                _PYYKKO_DENSE,
                cutoff=self.local_cutoff_angstrom,
                cna_alpha=CNA_ALPHA,
                distance_scale=DISCRETE_DISTANCE_SCALE,
                switch_alpha=ALPHA_LAMBDA,
                lambda_strong=BO_LAMBDA_STRONG,
                lambda_weak=BO_LAMBDA_WEAK,
            )
            (
                pair_left,
                pair_right,
                pair_distance,
                native_coordination,
                native_effective_radii,
                native_discrete_left,
                native_discrete_right,
                native_discrete_connectivity,
                native_accepted_left,
                native_accepted_right,
                native_cycles,
                native_cycle_candidate_count,
                native_cycle_rank,
            ) = native_result
            pair_left = pair_left.astype(np.intp, copy=False)
            pair_right = pair_right.astype(np.intp, copy=False)
            self.pair_screening_backend = "cpp-float64"
            self.native_accepted_left = native_accepted_left.astype(
                np.intp, copy=False
            )
            self.native_accepted_right = native_accepted_right.astype(
                np.intp, copy=False
            )
            self.native_cycle_basis = native_cycles
            self.native_cycle_candidate_count = native_cycle_candidate_count
            self.native_cycle_rank = native_cycle_rank
        elif candidate_pairs_override is not None:
            ordered_pairs = tuple(
                sorted(
                    {
                        tuple(sorted((int(left), int(right))))
                        for left, right in candidate_pairs_override
                        if int(left) != int(right)
                    }
                )
            )
            if any(left < 0 or right >= self.natoms for left, right in ordered_pairs):
                raise ValueError("candidate-pair override contains an invalid atom index")
            candidate_left = np.fromiter(
                (pair[0] for pair in ordered_pairs), dtype=np.intp, count=len(ordered_pairs)
            )
            candidate_right = np.fromiter(
                (pair[1] for pair in ordered_pairs), dtype=np.intp, count=len(ordered_pairs)
            )
            deltas = self.coords[candidate_left] - self.coords[candidate_right]
            squared = np.einsum("ij,ij->i", deltas, deltas)
            selected = squared <= self.local_cutoff_angstrom**2
            pair_left = candidate_left[selected]
            pair_right = candidate_right[selected]
            pair_distance = np.sqrt(squared[selected])
            self.pair_screening_backend = "persistent-verlet-list"
        elif self.natoms <= 256:
            left, right = _upper_triangle_indices(self.natoms)
            deltas = self.coords[left] - self.coords[right]
            squared = np.einsum("ij,ij->i", deltas, deltas)
            selected = squared <= self.local_cutoff_angstrom**2
            pair_left = left[selected]
            pair_right = right[selected]
            pair_distance = np.sqrt(squared[selected])
        else:
            gpu_pairs = gpu_screen_candidate_pairs(
                self.coords,
                cutoff=self.local_cutoff_angstrom,
            )
            if gpu_pairs is not None:
                pair_left = gpu_pairs.left
                pair_right = gpu_pairs.right
                pair_distance = gpu_pairs.distances
                self.pair_screening_backend = (
                    f"{gpu_pairs.backend}-{gpu_pairs.device}-"
                    f"{gpu_pairs.screening_precision}-screen/"
                    f"{gpu_pairs.certified_precision}-certified"
                )
            else:
                regions = SpatialRegions.build(
                    self.coords,
                    cell_size=max(2.0, self.local_cutoff_angstrom),
                )
                local_pairs = tuple(
                    sorted(regions.candidate_pairs(self.local_cutoff_angstrom))
                )
                pair_left = np.fromiter(
                    (pair[0] for pair in local_pairs),
                    dtype=np.intp,
                    count=len(local_pairs),
                )
                pair_right = np.fromiter(
                    (pair[1] for pair in local_pairs),
                    dtype=np.intp,
                    count=len(local_pairs),
                )
                deltas = self.coords[pair_left] - self.coords[pair_right]
                pair_distance = np.sqrt(np.einsum("ij,ij->i", deltas, deltas))
                self.pair_screening_backend = "spatial-regions-cpu"
        if self.bond_order_overrides or self.connectivity_overrides:
            local_pairs = set(
                zip(
                    (int(value) for value in pair_left),
                    (int(value) for value in pair_right),
                    strict=True,
                )
            )
            local_pairs.update(
                tuple(sorted((int(left), int(right))))
                for left, right in self.bond_order_overrides
                if int(left) != int(right)
            )
            local_pairs.update(
                tuple(sorted((int(left), int(right))))
                for left, right in self.connectivity_overrides
                if int(left) != int(right)
            )
            ordered_pairs = tuple(sorted(local_pairs))
            pair_left = np.fromiter(
                (pair[0] for pair in ordered_pairs),
                dtype=np.intp,
                count=len(ordered_pairs),
            )
            pair_right = np.fromiter(
                (pair[1] for pair in ordered_pairs),
                dtype=np.intp,
                count=len(ordered_pairs),
            )
            deltas = self.coords[pair_left] - self.coords[pair_right]
            pair_distance = np.sqrt(np.einsum("ij,ij->i", deltas, deltas))
        self._pair_left = pair_left
        self._pair_right = pair_right
        self._pair_distance = pair_distance
        self._candidate_pairs: tuple[tuple[int, int], ...] | None = None
        self._pair_distances: dict[tuple[int, int], float] | None = None
        self.coordination_numbers = np.zeros(self.natoms, dtype=float)
        if native_result is not None:
            self.coordination_numbers = native_coordination
        elif len(pair_left):
            radius_sums = standard_radii[pair_left] + standard_radii[pair_right]
            finite = np.isfinite(radius_sums)
            contributions = np.zeros(len(pair_left), dtype=float)
            contributions[finite] = np.fromiter(
                (
                    0.5
                    * (
                        1.0
                        + math.erf(
                            CNA_ALPHA * (radius_sum - distance)
                        )
                    )
                    for radius_sum, distance in zip(
                        radius_sums[finite],
                        pair_distance[finite],
                        strict=True,
                    )
                ),
                dtype=float,
                count=int(np.count_nonzero(finite)),
            )
            self.coordination_numbers = (
                np.bincount(
                    pair_left,
                    weights=contributions,
                    minlength=self.natoms,
                )
                + np.bincount(
                    pair_right,
                    weights=contributions,
                    minlength=self.natoms,
                )
            )

        if native_result is not None:
            self._effective_radii = native_effective_radii
            self.discrete_candidate_left = native_discrete_left.astype(
                np.intp, copy=False
            )
            self.discrete_candidate_right = native_discrete_right.astype(
                np.intp, copy=False
            )
            self.discrete_candidate_connectivity = native_discrete_connectivity
        else:
            self._effective_radii = np.fromiter(
                (
                    connectivity_effective_covalent_radius(
                        int(number),
                        coordination,
                    )
                    for number, coordination in zip(
                        self.Z,
                        self.coordination_numbers,
                        strict=True,
                    )
                ),
                dtype=float,
                count=self.natoms,
            )
            geometric_graph_mask = (
                np.isfinite(standard_radii[pair_left])
                & np.isfinite(standard_radii[pair_right])
                & (
                    pair_distance
                    <= DISCRETE_DISTANCE_SCALE
                    * (standard_radii[pair_left] + standard_radii[pair_right])
                )
            )
            override_values = np.fromiter(
                (
                    float(self.connectivity_overrides.get((int(left), int(right)), np.nan))
                    for left, right in zip(pair_left, pair_right, strict=True)
                ),
                dtype=float,
                count=len(pair_left),
            )
            overridden = np.isfinite(override_values)
            graph_mask = np.where(
                overridden,
                override_values >= 0.2,
                geometric_graph_mask,
            )
            self.discrete_candidate_left = pair_left[graph_mask]
            self.discrete_candidate_right = pair_right[graph_mask]
            graph_distances = pair_distance[graph_mask]
            graph_references = (
                self._effective_radii[self.discrete_candidate_left]
                + self._effective_radii[self.discrete_candidate_right]
            )
            geometric_connectivity = _bond_order_switched_values(
                graph_distances,
                graph_references,
            )
            self.discrete_candidate_connectivity = np.where(
                overridden[graph_mask],
                np.maximum(override_values[graph_mask], 0.0),
                geometric_connectivity,
            )
        self._discrete_candidate_pairs: tuple[tuple[int, int], ...] | None = None
        self._CONNECTIVITY: np.ndarray | None = None
        self._BO: np.ndarray | None = None
        self._BO_SIGMA: np.ndarray | None = None
        self._BO_PI: np.ndarray | None = None
        self._BO_PI_PI: np.ndarray | None = None
        self._standard_radii = standard_radii

    @property
    def candidate_pairs(self) -> tuple[tuple[int, int], ...]:
        if self._candidate_pairs is None:
            self._candidate_pairs = tuple(
                zip(
                    (int(value) for value in self._pair_left),
                    (int(value) for value in self._pair_right),
                    strict=True,
                )
            )
        return self._candidate_pairs

    @property
    def discrete_candidate_pairs(self) -> tuple[tuple[int, int], ...]:
        if self._discrete_candidate_pairs is None:
            self._discrete_candidate_pairs = tuple(
                zip(
                    (int(value) for value in self.discrete_candidate_left),
                    (int(value) for value in self.discrete_candidate_right),
                    strict=True,
                )
            )
        return self._discrete_candidate_pairs

    @property
    def pair_distances(self) -> dict[tuple[int, int], float]:
        if self._pair_distances is None:
            self._pair_distances = dict(
                zip(
                    self.candidate_pairs,
                    (float(value) for value in self._pair_distance),
                    strict=True,
                )
            )
        return self._pair_distances

    @property
    def CONNECTIVITY(self) -> np.ndarray:
        self._materialize_continuous_matrices()
        assert self._CONNECTIVITY is not None
        return self._CONNECTIVITY

    @property
    def BO(self) -> np.ndarray:
        self._materialize_continuous_matrices()
        assert self._BO is not None
        return self._BO

    @property
    def BO_SIGMA(self) -> np.ndarray:
        self._materialize_continuous_matrices()
        assert self._BO_SIGMA is not None
        return self._BO_SIGMA

    @property
    def BO_PI(self) -> np.ndarray:
        self._materialize_continuous_matrices()
        assert self._BO_PI is not None
        return self._BO_PI

    @property
    def BO_PI_PI(self) -> np.ndarray:
        self._materialize_continuous_matrices()
        assert self._BO_PI_PI is not None
        return self._BO_PI_PI

    def _materialize_continuous_matrices(self) -> None:
        if self._CONNECTIVITY is not None:
            return
        shape = (self.natoms, self.natoms)
        connectivity_matrix = np.zeros(shape, dtype=float)
        order_matrix = np.zeros(shape, dtype=float)
        sigma_matrix = np.zeros(shape, dtype=float)
        pi_matrix = np.zeros(shape, dtype=float)
        pi_pi_matrix = np.zeros(shape, dtype=float)
        if len(self._pair_left):
            connectivity = np.fromiter(
                (
                    _bond_order_switched(distance, reference)
                    for distance, reference in zip(
                        self._pair_distance,
                        self._effective_radii[self._pair_left]
                        + self._effective_radii[self._pair_right],
                        strict=True,
                    )
                ),
                dtype=float,
                count=len(self._pair_left),
            )
            orders = np.fromiter(
                (
                    self.bond_order_overrides.get(
                        (int(left), int(right)),
                        math.exp(
                            (
                                radius_left
                                + radius_right
                                - distance
                            )
                            / BO_PAULING_DECAY_ANGSTROM
                        )
                        if np.isfinite(radius_left)
                        and np.isfinite(radius_right)
                        else 0.0,
                    )
                    for left, right, radius_left, radius_right, distance in zip(
                        self._pair_left,
                        self._pair_right,
                        self._standard_radii[self._pair_left],
                        self._standard_radii[self._pair_right],
                        self._pair_distance,
                        strict=True,
                    )
                ),
                dtype=float,
                count=len(self._pair_left),
            )
            if len(self._pair_left) <= 6:
                components = tuple(
                    bond_order_components(order) for order in orders
                )
                sigma = np.fromiter(
                    (component.sigma for component in components),
                    dtype=float,
                    count=len(components),
                )
                pi = np.fromiter(
                    (component.pi for component in components),
                    dtype=float,
                    count=len(components),
                )
                pi_pi = np.fromiter(
                    (component.pi_pi for component in components),
                    dtype=float,
                    count=len(components),
                )
            else:
                sigma, pi, pi_pi = _bond_order_component_arrays(orders)
            for matrix, values in (
                (connectivity_matrix, connectivity),
                (order_matrix, orders),
                (sigma_matrix, sigma),
                (pi_matrix, pi),
                (pi_pi_matrix, pi_pi),
            ):
                matrix[self._pair_left, self._pair_right] = values
                matrix[self._pair_right, self._pair_left] = values
        self._CONNECTIVITY = connectivity_matrix
        self._BO = order_matrix
        self._BO_SIGMA = sigma_matrix
        self._BO_PI = pi_matrix
        self._BO_PI_PI = pi_pi_matrix


@dataclass(frozen=True)
class BondOrderComponents:
    """Ordinal sigma/first-pi/second-pi resolution of a multiplicity order."""

    sigma: float
    pi: float
    pi_pi: float

    @property
    def total(self) -> float:
        return float(self.sigma + self.pi + self.pi_pi)

    @property
    def total_pi(self) -> float:
        return float(self.pi + self.pi_pi)


def _c2_positive_excess(value: float, width: float) -> float:
    """Return a one-sided C2 approximation to ``max(value, 0)``.

    The polynomial joins zero at the integer bond-order boundary to the
    exact positive excess at ``width``.  It has matching first and second
    derivatives at both ends and introduces no pre-boundary pi leakage.
    """

    x = float(value)
    w = float(width)
    if not math.isfinite(x):
        raise ValueError("bond order must be finite")
    if not math.isfinite(w) or w <= 0.0:
        raise ValueError("bond-order component ramp width must be positive")
    if x <= 0.0:
        return 0.0
    if x >= w:
        return x
    t = x / w
    return w * (6.0 * t**3 - 8.0 * t**4 + 3.0 * t**5)


def bond_order_components(
    total_bond_order: float,
    *,
    ramp_width: float = BO_COMPONENT_RAMP_WIDTH,
) -> BondOrderComponents:
    """Resolve a non-negative multiplicity order into sigma, pi and pi-pi indices.

    ``pi`` is the first pi-bond occupancy and ``pi_pi`` is the second.  The
    three returned components are non-negative and sum to the input total to
    floating-point precision.
    """

    total = float(total_bond_order)
    if not math.isfinite(total):
        raise ValueError("bond order must be finite")
    total = max(total, 0.0)
    excess_one = _c2_positive_excess(total - 1.0, ramp_width)
    excess_two = _c2_positive_excess(total - 2.0, ramp_width)
    sigma = total - excess_one
    pi = excess_one - excess_two
    pi_pi = excess_two
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, total)
    values = [0.0 if abs(value) <= tolerance else float(value) for value in (sigma, pi, pi_pi)]
    if any(value < 0.0 for value in values):
        raise ArithmeticError("invalid negative bond-order component")
    return BondOrderComponents(*values)


def _bond_order_component_arrays(
    total_bond_order: np.ndarray,
    *,
    ramp_width: float = BO_COMPONENT_RAMP_WIDTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = np.maximum(np.asarray(total_bond_order, dtype=float), 0.0)

    def excess(boundary: float) -> np.ndarray:
        value = total - boundary
        output = np.zeros_like(value)
        high = value >= ramp_width
        middle = (value > 0.0) & ~high
        output[high] = value[high]
        reduced = value[middle] / ramp_width
        output[middle] = ramp_width * (6.0 * reduced**3 - 8.0 * reduced**4 + 3.0 * reduced**5)
        return output

    excess_one = excess(1.0)
    excess_two = excess(2.0)
    return total - excess_one, excess_one - excess_two, excess_two


def pauling_bond_order(
    i,
    j,
    Z,
    coords,
    *,
    decay_length: float = BO_PAULING_DECAY_ANGSTROM,
    distance: float | None = None,
):
    """Return the radial Pauling bond-multiplicity index used by ORACLE.

    The fixed covalent radii retain radial multiple-bond character. External
    Mayer orders supersede this geometric estimate at the pipeline level.
    """

    decay = float(decay_length)
    if not math.isfinite(decay) or decay <= 0.0:
        raise ValueError("Pauling bond-order decay length must be positive")
    rcov_i = standard_rcov(int(Z[i]))
    rcov_j = standard_rcov(int(Z[j]))
    if rcov_i is None or rcov_j is None:
        return 0.0
    separation = (
        float(distance)
        if distance is not None
        else float(np.linalg.norm(np.asarray(coords[i]) - np.asarray(coords[j])))
    )
    return math.exp((float(rcov_i) + float(rcov_j) - separation) / decay)


def noncovalent_pauling_bond_order(
    i,
    j,
    Z,
    coords,
    *,
    distance: float | None = None,
):
    """Return a parameter-free vdW-normalized Pauling interaction order.

    The van der Waals radius sum replaces both the covalent reference length
    and the empirical decay length of the covalent Pauling expression:

    ``B_vdW(r) = exp[-r / (R_vdW,i + R_vdW,j)]``.

    Ratios to a reference contact therefore require no fitted constant and
    retain analytic Cartesian derivatives.
    """

    radius_i = uff_vdw_radius(int(Z[i]))
    radius_j = uff_vdw_radius(int(Z[j]))
    if radius_i is None or radius_j is None:
        return 0.0
    radius_sum = float(radius_i) + float(radius_j)
    if radius_sum <= 0.0:
        return 0.0
    separation = (
        float(distance)
        if distance is not None
        else float(np.linalg.norm(np.asarray(coords[i]) - np.asarray(coords[j])))
    )
    return math.exp(-separation / radius_sum)


# ============================================================
# Principal quantum number
# ============================================================


def principal_quantum_number(Z):
    if Z <= 2:
        return 1
    elif Z <= 10:
        return 2
    elif Z <= 18:
        return 3
    elif Z <= 36:
        return 4
    elif Z <= 54:
        return 5
    elif Z <= 86:
        return 6
    else:
        return 7


# ============================================================
# Continuous Coordination Number (CNA)
# ============================================================


def continuous_coordination_number(i, Z, coords, neighbors, *, distances=None):
    Zi = Z[i]
    Ri = coords[i]
    cna = 0.0

    for j in neighbors[i]:
        Zj = Z[j]
        Rj = coords[j]
        key = (i, j) if i < j else (j, i)
        Rij = (
            distances[key]
            if distances is not None and key in distances
            else np.linalg.norm(Ri - Rj)
        )

        rcov_i = standard_rcov(Zi)
        rcov_j = standard_rcov(Zj)
        if rcov_i is None or rcov_j is None:
            continue

        R0 = rcov_i + rcov_j
        x = CNA_ALPHA * (R0 - Rij)
        cna += 0.5 * (1.0 + math.erf(x))

    return cna


# ============================================================
# Effective covalent radius (Pyykkö, C¹ interpolation)
# ============================================================


def hermite_c1(t, y0, y1, m0, m1):
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    return h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1


def hermite_slope(table, keys, key):
    """Return the canonical finite-difference slope for a tabulated descriptor."""
    if key <= keys[0]:
        return table[keys[1]] - table[keys[0]]
    if key >= keys[-1]:
        return table[keys[-1]] - table[keys[-2]]
    lower = max(item for item in keys if item < key and item in table)
    upper = min(item for item in keys if item > key and item in table)
    if lower == keys[0]:
        left = lower
    else:
        left = max(item for item in keys if item < lower and item in table)
    if upper == keys[-1]:
        right = upper
    else:
        right = min(item for item in keys if item > upper and item in table)
    return (
        0.5 * (table[upper] - table[lower]) if right == left else 0.5 * (table[right] - table[left])
    )


def connectivity_effective_covalent_radius(Zi, cna):
    """Radius used by the graph-perception weight, not a synthon observable."""
    atomic_number = int(Zi)
    if atomic_number < 0 or atomic_number >= len(_PYYKKO_TABLES):
        return standard_rcov(atomic_number)
    keys, values = _PYYKKO_TABLES[atomic_number]
    if not keys:
        return standard_rcov(Zi)
    if len(keys) == 1 or cna <= keys[0]:
        return values[0]
    if cna >= keys[-1]:
        return values[-1]
    lower_index = bisect_right(keys, cna) - 1
    upper_index = lower_index + 1
    coordination_lower = keys[lower_index]
    coordination_upper = keys[upper_index]
    radius_lower = values[lower_index]
    radius_upper = values[upper_index]
    t = (cna - coordination_lower) / (
        coordination_upper - coordination_lower
    )
    interval_slope = radius_upper - radius_lower
    return hermite_c1(
        t,
        radius_lower,
        radius_upper,
        interval_slope,
        interval_slope,
    )


# ============================================================
# Bond order
# ============================================================


def _bond_order_switched(Rij, R0):
    """
    Smoothly blend strong- and weak-decay exponentials.
    Short bonds use the strong decay, long distances use the weak decay.
    """
    if R0 <= 1.0e-12:
        return 0.0

    x = (Rij - R0) / R0
    w_strong = 0.5 * (1.0 - math.tanh(ALPHA_LAMBDA * x))
    bo_strong = math.exp((R0 - Rij) / BO_LAMBDA_STRONG)
    bo_weak = math.exp((R0 - Rij) / BO_LAMBDA_WEAK)
    return w_strong * bo_strong + (1.0 - w_strong) * bo_weak


def _bond_order_switched_array(
    distances: np.ndarray,
    reference_distances: np.ndarray,
) -> np.ndarray:
    distances = np.asarray(distances, dtype=float)
    references = np.asarray(reference_distances, dtype=float)
    output = np.zeros_like(distances)
    valid = references > 1.0e-12
    if not np.any(valid):
        return output
    reduced = (distances[valid] - references[valid]) / references[valid]
    strong_weight = 0.5 * (1.0 - np.tanh(ALPHA_LAMBDA * reduced))
    strong = np.exp(
        (references[valid] - distances[valid]) / BO_LAMBDA_STRONG
    )
    weak = np.exp(
        (references[valid] - distances[valid]) / BO_LAMBDA_WEAK
    )
    output[valid] = strong_weight * strong + (1.0 - strong_weight) * weak
    output[~np.isfinite(references)] = np.nan
    return output


def _bond_order_switched_values(
    distances: np.ndarray,
    reference_distances: np.ndarray,
) -> np.ndarray:
    if len(distances) <= 30:
        return np.fromiter(
            (
                _bond_order_switched(distance, reference)
                for distance, reference in zip(
                    distances,
                    reference_distances,
                    strict=True,
                )
            ),
            dtype=float,
            count=len(distances),
        )
    return _bond_order_switched_array(distances, reference_distances)


def _standard_radii_for_atomic_numbers(atomic_numbers: np.ndarray) -> np.ndarray:
    numbers = np.asarray(atomic_numbers, dtype=int)
    if numbers.size and np.min(numbers) >= 0 and np.max(numbers) <= _MAX_ATOMIC_NUMBER:
        return _STANDARD_RADII[numbers]
    result = np.full(numbers.shape, np.nan, dtype=float)
    valid = (numbers >= 0) & (numbers <= _MAX_ATOMIC_NUMBER)
    result[valid] = _STANDARD_RADII[numbers[valid]]
    return result


def connectivity_weight(i, j, Z, coords, neighbors, cache=None):
    """Return the smooth geometric weight used only for graph perception."""
    if cache is not None:
        key = (i, j) if i < j else (j, i)
        if key in cache:
            return cache[key]

    Ri = coords[i]
    Rj = coords[j]
    Rij = np.linalg.norm(Ri - Rj)

    cna_i = continuous_coordination_number(i, Z, coords, neighbors)
    cna_j = continuous_coordination_number(j, Z, coords, neighbors)

    rcov_i = connectivity_effective_covalent_radius(Z[i], cna_i)
    rcov_j = connectivity_effective_covalent_radius(Z[j], cna_j)

    if rcov_i is None or rcov_j is None:
        bo = 0.0
    else:
        bo = _bond_order_switched(Rij, rcov_i + rcov_j)

    if cache is not None:
        cache[key] = bo
    return bo


def connectivity_weight_from_coordination(
    i,
    j,
    Z,
    coords,
    coordination_numbers,
    *,
    distance: float | None = None,
):
    """Evaluate one connectivity weight from cached local coordination."""

    separation = (
        float(distance)
        if distance is not None
        else float(np.linalg.norm(np.asarray(coords[i]) - np.asarray(coords[j])))
    )
    radius_i = connectivity_effective_covalent_radius(Z[i], coordination_numbers[i])
    radius_j = connectivity_effective_covalent_radius(Z[j], coordination_numbers[j])
    if radius_i is None or radius_j is None:
        return 0.0
    return _bond_order_switched(separation, float(radius_i) + float(radius_j))
