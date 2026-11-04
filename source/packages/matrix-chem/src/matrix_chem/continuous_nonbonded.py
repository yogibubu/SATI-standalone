"""Analytic continuous-topology scaling for non-bonded pair potentials."""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, pi, sqrt
from typing import Callable, Protocol

import numpy as np

from .spatial_regions import SpatialRegions, bounded_topological_distances
from .topology.covalent_radii import covalent_radius
from .topology.descriptor_parameters import CNA_SCALE


BOHR_TO_ANGSTROM = 0.52917721092


class RadialDerivatives(Protocol):
    energy: float
    first: float
    second: float


@dataclass(frozen=True)
class _Scalar:
    value: float
    gradient: dict[int, float]
    hessian: dict[tuple[int, int], float]


def continuous_topology_scaled_pair_derivatives(
    coordinates_bohr: np.ndarray,
    atomic_numbers: np.ndarray,
    radial_factory: Callable[[int, int, float], RadialDerivatives | None],
    *,
    one_four_scale: float,
    switch_alpha_per_angstrom: float = 8.0,
    pair_components: dict[
        tuple[int, int], tuple[float, np.ndarray, np.ndarray]
    ] | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate all pairs with smooth 1-2/1-3/1-4 path attenuation.

    Edge membership is the same error-function contribution used in ORACLE's
    continuous coordination number.  Soft unions of one-, two- and three-edge
    paths replace discrete topological exclusions.  A sparse second-order
    algebra carries the complete analytic chain rule through the path weights.
    """

    coords = np.asarray(coordinates_bohr, dtype=float)
    numbers = np.asarray(atomic_numbers, dtype=int).reshape(-1)
    natoms = len(numbers)
    pairs = tuple((i, j) for i in range(natoms) for j in range(i + 1, natoms))
    pair_index = {pair: index for index, pair in enumerate(pairs)}
    distances: dict[tuple[int, int], float] = {}
    variables: dict[tuple[int, int], _Scalar] = {}
    alpha_bohr = float(switch_alpha_per_angstrom) * BOHR_TO_ANGSTROM
    if not np.isfinite(alpha_bohr) or alpha_bohr <= 0.0:
        raise ValueError("continuous topology switch alpha must be finite and positive")
    for pair, index in pair_index.items():
        i, j = pair
        distance = float(np.linalg.norm(coords[i] - coords[j]))
        if distance <= 1.0e-12:
            raise ValueError(f"non-bonded atoms {i + 1} and {j + 1} are coincident")
        distances[pair] = distance
        variables[pair] = _variable(index, distance, 1.0, 0.0)

    edge_pairs = _continuous_edge_pairs(coords, numbers, alpha_bohr)
    edges: dict[tuple[int, int], _Scalar] = {}
    edge_neighbors = [set() for _ in range(natoms)]
    for pair in edge_pairs:
        i, j = pair
        distance = distances[pair]
        ri = covalent_radius(int(numbers[i]))
        rj = covalent_radius(int(numbers[j]))
        if ri is None or rj is None:
            continue
        radius_sum_bohr = CNA_SCALE * (float(ri) + float(rj)) / BOHR_TO_ANGSTROM
        argument = alpha_bohr * (radius_sum_bohr - distance)
        gaussian = exp(-(argument**2)) / sqrt(pi)
        value = 0.5 * (1.0 + erf(argument))
        first = -alpha_bohr * gaussian
        second = -2.0 * alpha_bohr**2 * argument * gaussian
        edges[pair] = _variable(pair_index[pair], value, first, second)
        edge_neighbors[i].add(j)
        edge_neighbors[j].add(i)

    total = _constant(0.0)
    one = _constant(1.0)
    for i, j in pairs:
        radial = radial_factory(i, j, distances[(i, j)])
        if radial is None:
            continue
        one_edge = _edge(edges, i, j)
        two_paths, three_paths = _local_path_products(
            i,
            j,
            edges,
            edge_neighbors,
            include_three_paths=float(one_four_scale) != 1.0,
        )
        two_edge = _soft_union(two_paths)
        three_edge = _soft_union(three_paths)
        scale = _multiply(
            _multiply(_subtract(one, one_edge), _subtract(one, two_edge)),
            _subtract(one, _scale(three_edge, 1.0 - float(one_four_scale))),
        )
        radial_scalar = _variable(
            pair_index[(i, j)],
            float(radial.energy),
            float(radial.first),
            float(radial.second),
        )
        contribution = _multiply(scale, radial_scalar)
        if pair_components is not None:
            pair_components[(i, j)] = _to_cartesian(
                contribution,
                coords,
                pairs,
                distances,
            )
        total = _add(total, contribution)
    return _to_cartesian(total, coords, pairs, distances)


def continuous_topology_pair_corrections(
    coordinates_bohr: np.ndarray,
    atomic_numbers: np.ndarray,
    radial_factory: Callable[[int, int, float], RadialDerivatives | None],
    *,
    one_four_scale: float,
    switch_alpha_per_angstrom: float = 8.0,
    vector: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray | None, int]:
    """Return sparse corrections from full pair weight one to continuous paths.

    Only pairs connected by one-, two- or three-edge local paths can differ
    from unit weight. Distance derivatives are retained in sparse dictionaries;
    an optional Hessian action is accumulated directly without materializing a
    Cartesian Hessian.
    """

    coords = np.asarray(coordinates_bohr, dtype=float)
    numbers = np.asarray(atomic_numbers, dtype=int).reshape(-1)
    natoms = len(numbers)
    if coords.shape != (natoms, 3):
        raise ValueError("continuous-topology coordinates and elements are inconsistent")
    alpha_bohr = float(switch_alpha_per_angstrom) * BOHR_TO_ANGSTROM
    if not np.isfinite(alpha_bohr) or alpha_bohr <= 0.0:
        raise ValueError("continuous topology switch alpha must be finite and positive")

    edge_pairs = _continuous_edge_pairs(coords, numbers, alpha_bohr)
    edge_adjacency = [set() for _ in range(natoms)]
    for i, j in edge_pairs:
        edge_adjacency[i].add(j)
        edge_adjacency[j].add(i)
    affected_pairs = tuple(
        sorted(bounded_topological_distances(edge_adjacency, maximum_distance=3))
    )
    variable_pairs = tuple(sorted(set(edge_pairs) | set(affected_pairs)))
    pair_index = {pair: index for index, pair in enumerate(variable_pairs)}
    distances = {
        pair: float(np.linalg.norm(coords[pair[0]] - coords[pair[1]]))
        for pair in variable_pairs
    }
    if any(distance <= 1.0e-12 for distance in distances.values()):
        raise ValueError("coincident atoms in continuous-topology correction")
    edges: dict[tuple[int, int], _Scalar] = {}
    for pair in edge_pairs:
        i, j = pair
        ri = covalent_radius(int(numbers[i]))
        rj = covalent_radius(int(numbers[j]))
        if ri is None or rj is None:
            continue
        argument = alpha_bohr * (
            CNA_SCALE * (float(ri) + float(rj)) / BOHR_TO_ANGSTROM
            - distances[pair]
        )
        gaussian = exp(-(argument**2)) / sqrt(pi)
        edges[pair] = _variable(
            pair_index[pair],
            0.5 * (1.0 + erf(argument)),
            -alpha_bohr * gaussian,
            -2.0 * alpha_bohr**2 * argument * gaussian,
        )

    total = _constant(0.0)
    one = _constant(1.0)
    used_pairs = 0
    for i, j in affected_pairs:
        radial = radial_factory(i, j, distances[(i, j)])
        if radial is None:
            continue
        two_paths, three_paths = _local_path_products(
            i,
            j,
            edges,
            edge_adjacency,
            include_three_paths=float(one_four_scale) != 1.0,
        )
        scale = _multiply(
            _multiply(
                _subtract(one, _edge(edges, i, j)),
                _subtract(one, _soft_union(two_paths)),
            ),
            _subtract(
                one,
                _scale(_soft_union(three_paths), 1.0 - float(one_four_scale)),
            ),
        )
        radial_scalar = _variable(
            pair_index[(i, j)],
            float(radial.energy),
            float(radial.first),
            float(radial.second),
        )
        total = _add(total, _multiply(_subtract(scale, one), radial_scalar))
        used_pairs += 1
    energy, gradient, product = _to_cartesian_sparse_action(
        total,
        coords,
        variable_pairs,
        distances,
        vector=vector,
    )
    return energy, gradient, product, used_pairs


def continuous_topology_pair_scales(
    coordinates_bohr: np.ndarray,
    atomic_numbers: np.ndarray,
    *,
    one_four_scale: float = 0.5,
    switch_alpha_per_angstrom: float = 8.0,
) -> dict[tuple[int, int], float]:
    """Return one-based continuous pair weights for diagnostics."""
    coords = np.asarray(coordinates_bohr, dtype=float)
    numbers = np.asarray(atomic_numbers, dtype=int)
    if coords.shape != (len(numbers), 3):
        raise ValueError("continuous-topology coordinates and elements are inconsistent")
    alpha_bohr = float(switch_alpha_per_angstrom) * BOHR_TO_ANGSTROM
    if not np.isfinite(alpha_bohr) or alpha_bohr <= 0.0:
        raise ValueError("continuous topology switch alpha must be finite and positive")

    edge_pairs = _continuous_edge_pairs(coords, numbers, alpha_bohr)
    edges: dict[tuple[int, int], _Scalar] = {}
    edge_neighbors = [set() for _ in range(len(numbers))]
    for pair in edge_pairs:
        left, right = pair
        radius_left = covalent_radius(int(numbers[left]))
        radius_right = covalent_radius(int(numbers[right]))
        if radius_left is None or radius_right is None:
            continue
        distance = float(np.linalg.norm(coords[left] - coords[right]))
        argument = alpha_bohr * (
            CNA_SCALE * (float(radius_left) + float(radius_right)) / BOHR_TO_ANGSTROM
            - distance
        )
        edges[pair] = _constant(0.5 * (1.0 + erf(argument)))
        edge_neighbors[left].add(right)
        edge_neighbors[right].add(left)

    one = _constant(1.0)
    values: dict[tuple[int, int], float] = {}
    for left in range(len(numbers)):
        for right in range(left + 1, len(numbers)):
            two_paths, three_paths = _local_path_products(
                left,
                right,
                edges,
                edge_neighbors,
                include_three_paths=float(one_four_scale) != 1.0,
            )
            scale = _multiply(
                _multiply(
                    _subtract(one, _edge(edges, left, right)),
                    _subtract(one, _soft_union(two_paths)),
                ),
                _subtract(
                    one,
                    _scale(
                        _soft_union(three_paths),
                        1.0 - float(one_four_scale),
                    ),
                ),
            )
            values[(left + 1, right + 1)] = float(scale.value)
    return values


def _continuous_edge_pairs(
    coordinates_bohr: np.ndarray,
    atomic_numbers: np.ndarray,
    alpha_bohr: float,
) -> tuple[tuple[int, int], ...]:
    radii = [
        float(radius)
        for radius in (covalent_radius(int(value)) for value in atomic_numbers)
        if radius is not None
    ]
    if not radii:
        return ()
    # Beyond eight error-function widths the omitted edge and its first two
    # derivatives are below the numerical precision requested of this model.
    cutoff = (
        2.0 * CNA_SCALE * max(radii) / BOHR_TO_ANGSTROM
        + 8.0 / float(alpha_bohr)
    )
    regions = SpatialRegions.build(
        coordinates_bohr,
        cell_size=max(cutoff, 1.0),
    )
    return tuple(regions.candidate_pairs(cutoff))


def _local_path_products(
    i: int,
    j: int,
    edges: dict[tuple[int, int], _Scalar],
    neighbors: list[set[int]],
    *,
    include_three_paths: bool = True,
) -> tuple[list[_Scalar], list[_Scalar]]:
    two_paths = [
        _multiply(_edge(edges, i, middle), _edge(edges, middle, j))
        for middle in sorted(neighbors[i] & neighbors[j])
        if middle not in {i, j}
    ]
    three_paths: list[_Scalar] = []
    # At unit 1-4 scale the entire three-path term is multiplied by exact zero.
    # Avoid building its large second-order jet; no tolerance or path is pruned.
    if not include_three_paths:
        return two_paths, three_paths
    for left_middle in sorted(neighbors[i] - {j}):
        for right_middle in sorted(neighbors[left_middle] & neighbors[j]):
            if right_middle in {i, j, left_middle}:
                continue
            three_paths.append(
                _multiply(
                    _multiply(
                        _edge(edges, i, left_middle),
                        _edge(edges, left_middle, right_middle),
                    ),
                    _edge(edges, right_middle, j),
                )
            )
    return two_paths, three_paths


def _edge(edges: dict[tuple[int, int], _Scalar], i: int, j: int) -> _Scalar:
    return edges.get(tuple(sorted((i, j))), _constant(0.0))


def _constant(value: float) -> _Scalar:
    return _Scalar(float(value), {}, {})


def _variable(index: int, value: float, first: float, second: float) -> _Scalar:
    gradient = {} if first == 0.0 else {index: float(first)}
    hessian = {} if second == 0.0 else {(index, index): float(second)}
    return _Scalar(float(value), gradient, hessian)


def _add(left: _Scalar, right: _Scalar) -> _Scalar:
    gradient = dict(left.gradient)
    hessian = dict(left.hessian)
    for key, value in right.gradient.items():
        gradient[key] = gradient.get(key, 0.0) + value
    for key, value in right.hessian.items():
        hessian[key] = hessian.get(key, 0.0) + value
    return _Scalar(left.value + right.value, gradient, hessian)


def _scale(item: _Scalar, factor: float) -> _Scalar:
    return _Scalar(
        factor * item.value,
        {key: factor * value for key, value in item.gradient.items()},
        {key: factor * value for key, value in item.hessian.items()},
    )


def _subtract(left: _Scalar, right: _Scalar) -> _Scalar:
    return _add(left, _scale(right, -1.0))


def _multiply(left: _Scalar, right: _Scalar) -> _Scalar:
    gradient: dict[int, float] = {}
    hessian: dict[tuple[int, int], float] = {}
    for key in left.gradient.keys() | right.gradient.keys():
        gradient[key] = left.gradient.get(key, 0.0) * right.value + left.value * right.gradient.get(
            key, 0.0
        )
    for key, value in left.hessian.items():
        hessian[key] = hessian.get(key, 0.0) + value * right.value
    for key, value in right.hessian.items():
        hessian[key] = hessian.get(key, 0.0) + value * left.value
    for i, value_i in left.gradient.items():
        for j, value_j in right.gradient.items():
            key = tuple(sorted((i, j)))
            factor = 2.0 if i == j else 1.0
            hessian[key] = hessian.get(key, 0.0) + factor * value_i * value_j
    return _Scalar(left.value * right.value, gradient, hessian)


def _soft_union(items: list[_Scalar]) -> _Scalar:
    complement = _constant(1.0)
    one = _constant(1.0)
    for item in items:
        complement = _multiply(complement, _subtract(one, item))
    return _subtract(one, complement)


def _to_cartesian(
    value: _Scalar,
    coordinates_bohr: np.ndarray,
    pairs: tuple[tuple[int, int], ...],
    distances: dict[tuple[int, int], float],
) -> tuple[float, np.ndarray, np.ndarray]:
    size = 3 * len(coordinates_bohr)
    distance_gradients: list[np.ndarray] = []
    gradient = np.zeros(size, dtype=float)
    hessian = np.zeros((size, size), dtype=float)
    for index, (i, j) in enumerate(pairs):
        rvec = coordinates_bohr[i] - coordinates_bohr[j]
        distance = distances[(i, j)]
        unit = rvec / distance
        dq = np.zeros(size, dtype=float)
        dq[3 * i : 3 * i + 3] = unit
        dq[3 * j : 3 * j + 3] = -unit
        distance_gradients.append(dq)
        coefficient = value.gradient.get(index, 0.0)
        gradient += coefficient * dq
        geometric = coefficient / distance * (np.eye(3) - np.outer(unit, unit))
        si = slice(3 * i, 3 * i + 3)
        sj = slice(3 * j, 3 * j + 3)
        hessian[si, si] += geometric
        hessian[sj, sj] += geometric
        hessian[si, sj] -= geometric
        hessian[sj, si] -= geometric
    for (left, right), coefficient in value.hessian.items():
        outer = np.outer(distance_gradients[left], distance_gradients[right])
        hessian += coefficient * (outer if left == right else outer + outer.T)
    return float(value.value), gradient, 0.5 * (hessian + hessian.T)


def _to_cartesian_sparse_action(
    value: _Scalar,
    coordinates_bohr: np.ndarray,
    pairs: tuple[tuple[int, int], ...],
    distances: dict[tuple[int, int], float],
    *,
    vector: np.ndarray | None,
) -> tuple[float, np.ndarray, np.ndarray | None]:
    natoms = len(coordinates_bohr)
    gradient = np.zeros((natoms, 3), dtype=float)
    direction = None
    product = None
    if vector is not None:
        direction = np.asarray(vector, dtype=float)
        if direction.size == 3 * natoms:
            direction = direction.reshape(natoms, 3)
        if direction.shape != (natoms, 3):
            raise ValueError("continuous-topology Hessian-vector dimensions are inconsistent")
        product = np.zeros((natoms, 3), dtype=float)

    units: list[np.ndarray] = []
    directional_distance_derivatives: list[float] = []
    for index, (i, j) in enumerate(pairs):
        delta = coordinates_bohr[i] - coordinates_bohr[j]
        distance = distances[(i, j)]
        unit = delta / distance
        units.append(unit)
        coefficient = value.gradient.get(index, 0.0)
        gradient[i] += coefficient * unit
        gradient[j] -= coefficient * unit
        if direction is None or product is None:
            directional_distance_derivatives.append(0.0)
            continue
        relative_direction = direction[i] - direction[j]
        directional_distance_derivatives.append(float(unit @ relative_direction))
        geometric = coefficient / distance * (
            relative_direction - unit * float(unit @ relative_direction)
        )
        product[i] += geometric
        product[j] -= geometric

    if product is not None:
        for (left, right), coefficient in value.hessian.items():
            left_pair = pairs[left]
            left_vector = coefficient * units[left] * directional_distance_derivatives[right]
            product[left_pair[0]] += left_vector
            product[left_pair[1]] -= left_vector
            if left != right:
                right_pair = pairs[right]
                right_vector = (
                    coefficient
                    * units[right]
                    * directional_distance_derivatives[left]
                )
                product[right_pair[0]] += right_vector
                product[right_pair[1]] -= right_vector
    return (
        float(value.value),
        gradient.reshape(-1),
        None if product is None else product.reshape(-1),
    )
