"""Topology-free collective variables shared by MATRIX samplers.

The definitions in this module are deliberately independent of MC, MD, and
genetic algorithms.  A variable can therefore be used without reinterpretation
as a Monte Carlo bias, a metadynamics or umbrella coordinate, or a Pareto
objective.  Cartesian derivatives are analytic; unsupported non-smooth
definitions fail explicitly when a derivative is requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Mapping, Sequence

import numpy as np


BOHR_TO_ANGSTROM = 0.529177210903


@dataclass(frozen=True)
class CollectiveVariableEvaluation:
    """Value and optional analytic Cartesian derivative of one variable."""

    name: str
    value: float
    gradient_per_angstrom: np.ndarray | None

    def __post_init__(self) -> None:
        if not self.name or not np.isfinite(self.value):
            raise ValueError("collective-variable names and values must be valid")
        gradient = self.gradient_per_angstrom
        if gradient is not None:
            array = np.asarray(gradient, dtype=float)
            if array.ndim != 2 or array.shape[1] != 3 or np.any(~np.isfinite(array)):
                raise ValueError("collective-variable gradient must have shape (atoms, 3)")
            object.__setattr__(self, "gradient_per_angstrom", array.copy())


def evaluate_collective_variable(
    definition: Mapping[str, object],
    coordinates: np.ndarray,
    *,
    coordinate_unit: str = "angstrom",
    require_gradient: bool = False,
) -> CollectiveVariableEvaluation:
    """Evaluate one declared collective variable.

    Supported smooth variables are distances, continuous contact/coordination
    numbers, and atom-centred hydrogen-bond scores.  Ring-centred hydrogen-bond
    scores remain available for energy-only exploration; their best-fit-plane
    derivative is intentionally rejected instead of approximated numerically.
    """

    raw = np.asarray(coordinates, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 3 or len(raw) == 0 or np.any(~np.isfinite(raw)):
        raise ValueError("collective variables need finite Cartesian coordinates")
    unit = str(coordinate_unit).strip().lower()
    if unit in {"angstrom", "ang", "a"}:
        xyz = raw
    elif unit in {"bohr", "atomic"}:
        xyz = raw * BOHR_TO_ANGSTROM
    else:
        raise ValueError("coordinate_unit must be angstrom or bohr")
    name = str(definition.get("name", "")).strip()
    if not name:
        raise ValueError("each collective variable needs a stable name")
    kind = str(definition.get("kind", "")).strip().lower().replace("_", "-")
    index_base = int(definition.get("index_base", 1))
    if kind in {"distance", "atom-distance"}:
        atoms = definition.get("atoms")
        if not isinstance(atoms, Sequence) or isinstance(atoms, (str, bytes)) or len(atoms) != 2:
            raise ValueError("a distance variable needs two atom indices")
        left, right = _indices(atoms, len(xyz), index_base=index_base)
        displacement = xyz[right] - xyz[left]
        distance = float(np.linalg.norm(displacement))
        if distance <= 1.0e-14:
            raise ValueError("a distance variable is singular at zero separation")
        gradient = np.zeros_like(xyz)
        gradient[right] = displacement / distance
        gradient[left] = -gradient[right]
        return CollectiveVariableEvaluation(name, distance, gradient)
    if kind in {"coordination", "contact-number", "coordination-number"}:
        pairs = definition.get("pairs")
        if not isinstance(pairs, list) or not pairs:
            raise ValueError("a coordination variable needs a non-empty pair list")
        reference = float(definition.get("reference_distance_angstrom", 3.0))
        exponent = float(definition.get("exponent", 6.0))
        if reference <= 0.0 or exponent <= 0.0:
            raise ValueError("coordination reference distance and exponent must be positive")
        value = 0.0
        gradient = np.zeros_like(xyz)
        for pair in pairs:
            if not isinstance(pair, Sequence) or isinstance(pair, (str, bytes)) or len(pair) != 2:
                raise ValueError("each coordination pair needs two atom indices")
            left, right = _indices(pair, len(xyz), index_base=index_base)
            displacement = xyz[right] - xyz[left]
            distance = float(np.linalg.norm(displacement))
            if distance <= 1.0e-14:
                raise ValueError("a coordination pair is singular at zero separation")
            ratio = distance / reference
            power = ratio**exponent
            score = 1.0 / (1.0 + power)
            derivative = -exponent * power / (distance * (1.0 + power) ** 2)
            direction = displacement / distance
            gradient[right] += derivative * direction
            gradient[left] -= derivative * direction
            value += score
        return CollectiveVariableEvaluation(name, float(value), gradient)
    if kind in {"radial-moment", "radial-extent"}:
        atoms = definition.get("atoms")
        if not isinstance(atoms, Sequence) or isinstance(atoms, (str, bytes)) or not atoms:
            raise ValueError("a radial-moment variable needs a non-empty atom list")
        indices = _indices(atoms, len(xyz), index_base=index_base)
        center_raw = definition.get("center", (0.0, 0.0, 0.0))
        if not isinstance(center_raw, Sequence) or len(center_raw) != 3:
            raise ValueError("a radial-moment variable needs a three-dimensional center")
        center = np.asarray(tuple(float(value) for value in center_raw), dtype=float)
        power = float(definition.get("power", 2.0))
        if power <= 0.0 or not np.isfinite(power):
            raise ValueError("radial-moment power must be positive and finite")
        displacement = xyz[list(indices)] - center[None, :]
        radii = np.linalg.norm(displacement, axis=1)
        if np.any(radii <= 1.0e-14):
            raise ValueError("a radial-moment variable is singular at the center")
        value = float(np.mean(radii**power))
        gradient = np.zeros_like(xyz)
        gradient[list(indices)] = (
            power * radii[:, None] ** (power - 2.0) * displacement / len(indices)
        )
        return CollectiveVariableEvaluation(name, value, gradient)
    if kind in {"hydrogen-bond", "hbond"}:
        if "acceptor_ring" in definition:
            value = _ring_hbond_value(definition, xyz, index_base=index_base)
            if require_gradient:
                raise ValueError(
                    "ring-centred hydrogen-bond CVs have no analytic plane derivative; "
                    "use atom-centred acceptors for MD biases"
                )
            return CollectiveVariableEvaluation(name, value, None)
        value, gradient = _atom_hbond_value_gradient(
            definition,
            xyz,
            index_base=index_base,
        )
        return CollectiveVariableEvaluation(name, value, gradient)
    raise ValueError(f"unknown collective-variable kind: {kind}")


def evaluate_collective_variables(
    definitions: Sequence[Mapping[str, object]],
    coordinates: np.ndarray,
    *,
    coordinate_unit: str = "angstrom",
    require_gradient: bool = False,
) -> tuple[CollectiveVariableEvaluation, ...]:
    """Evaluate a uniquely named collection in declared order."""

    results = tuple(
        evaluate_collective_variable(
            definition,
            coordinates,
            coordinate_unit=coordinate_unit,
            require_gradient=require_gradient,
        )
        for definition in definitions
    )
    names = tuple(item.name for item in results)
    if len(set(names)) != len(names):
        raise ValueError("collective-variable names must be unique")
    return results


def _atom_hbond_value_gradient(
    definition: Mapping[str, object],
    xyz: np.ndarray,
    *,
    index_base: int,
) -> tuple[float, np.ndarray]:
    donor, hydrogen, acceptor = _indices(
        (
            definition["donor"],
            definition["hydrogen"],
            definition["acceptor"],
        ),
        len(xyz),
        index_base=index_base,
    )
    if len({donor, hydrogen, acceptor}) != 3:
        raise ValueError("hydrogen-bond donor, hydrogen, and acceptor must be distinct")
    u = xyz[donor] - xyz[hydrogen]
    v = xyz[acceptor] - xyz[hydrogen]
    norm_u = float(np.linalg.norm(u))
    distance = float(np.linalg.norm(v))
    if min(norm_u, distance) <= 1.0e-14:
        raise ValueError("hydrogen-bond geometry is singular")
    cosine = float(np.clip(np.dot(u, v) / (norm_u * distance), -1.0, 1.0))
    angle = float(np.arccos(cosine))
    cutoff = float(definition.get("distance_cutoff_angstrom", 2.5))
    distance_width = float(definition.get("distance_width_angstrom", 0.2))
    angle_center_degrees = float(definition.get("angle_center_degrees", 180.0))
    angle_width_degrees = float(definition.get("angle_width_degrees", 25.0))
    if distance_width <= 0.0 or angle_width_degrees <= 0.0:
        raise ValueError("hydrogen-bond switching widths must be positive")
    exponent = float(np.clip((distance - cutoff) / distance_width, -700.0, 700.0))
    distance_score = float(1.0 / (1.0 + np.exp(exponent)))
    angle_degrees = angle * 180.0 / pi
    delta_degrees = angle_center_degrees - angle_degrees
    angle_score = float(np.exp(-0.5 * (delta_degrees / angle_width_degrees) ** 2))
    score = distance_score * angle_score

    distance_derivative = (
        -distance_score * (1.0 - distance_score) / distance_width
    )
    gradient_distance_v = distance_derivative * v / distance
    gradient = np.zeros_like(xyz)
    gradient[acceptor] += angle_score * gradient_distance_v
    gradient[hydrogen] -= angle_score * gradient_distance_v

    sine = float(np.sqrt(max(1.0e-24, 1.0 - cosine * cosine)))
    dc_du = v / (norm_u * distance) - cosine * u / (norm_u * norm_u)
    dc_dv = u / (norm_u * distance) - cosine * v / (distance * distance)
    dscore_dangle = (
        distance_score
        * angle_score
        * delta_degrees
        / (angle_width_degrees * angle_width_degrees)
        * 180.0
        / pi
    )
    dscore_dc = -dscore_dangle / sine
    gradient[donor] += dscore_dc * dc_du
    gradient[acceptor] += dscore_dc * dc_dv
    gradient[hydrogen] -= dscore_dc * (dc_du + dc_dv)
    return float(score), gradient


def _ring_hbond_value(
    definition: Mapping[str, object],
    xyz: np.ndarray,
    *,
    index_base: int,
) -> float:
    donor, hydrogen = _indices(
        (definition["donor"], definition["hydrogen"]),
        len(xyz),
        index_base=index_base,
    )
    raw_ring = definition["acceptor_ring"]
    if not isinstance(raw_ring, Sequence) or isinstance(raw_ring, (str, bytes)) or len(raw_ring) < 3:
        raise ValueError("acceptor_ring must contain at least three atoms")
    ring = _indices(raw_ring, len(xyz), index_base=index_base)
    ring_xyz = xyz[list(ring)]
    center = np.mean(ring_xyz, axis=0)
    _u, singular_values, vh = np.linalg.svd(ring_xyz - center, full_matrices=False)
    if len(singular_values) < 2 or singular_values[1] <= 1.0e-10:
        raise ValueError("acceptor_ring atoms do not define a plane")
    atom_definition = dict(definition)
    atom_definition.pop("acceptor_ring", None)
    temporary = np.vstack((xyz, center))
    atom_definition["acceptor"] = len(temporary) - 1 + index_base
    value, _gradient = _atom_hbond_value_gradient(
        atom_definition,
        temporary,
        index_base=index_base,
    )
    approach = center - xyz[hydrogen]
    approach_norm = float(np.linalg.norm(approach))
    normal_angle = (
        0.5 * pi
        if approach_norm <= 1.0e-14
        else float(np.arccos(np.clip(abs(np.dot(approach, vh[-1])) / approach_norm, 0.0, 1.0)))
    )
    width = float(definition.get("ring_normal_width_degrees", 30.0)) * pi / 180.0
    if width <= 0.0:
        raise ValueError("ring_normal_width_degrees must be positive")
    return float(value * np.exp(-0.5 * (normal_angle / width) ** 2))


def _indices(
    raw: Sequence[object],
    atom_count: int,
    *,
    index_base: int,
) -> tuple[int, ...]:
    indices = tuple(int(value) - index_base for value in raw)
    if any(index < 0 or index >= atom_count for index in indices):
        raise ValueError("collective-variable atom index is outside the geometry")
    return indices
