"""Deterministic valence completion and initial hydrogen placement."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from math import pi, sqrt
from typing import Mapping, Sequence

import numpy as np

from .geometry import MolecularGeometry
from .topology.elements import atomic_number
from .topology.pykko_radii import covalent_radius


HYDROGEN_COMPLETION_SCHEMA = "matrix.oracle.hydrogen_completion.v1"
_DEFAULT_VALENCE = {5: 3, 6: 4, 7: 3, 8: 2, 9: 1, 14: 4, 17: 1, 35: 1, 53: 1}


@dataclass(frozen=True)
class AddedHydrogen:
    atom: int
    parent: int
    bond_length_angstrom: float


@dataclass(frozen=True)
class HydrogenCompletion:
    geometry: MolecularGeometry
    bonds: tuple[tuple[int, int], ...]
    additions: tuple[AddedHydrogen, ...]
    schema: str = HYDROGEN_COMPLETION_SCHEMA


def complete_valence_hydrogens(
    geometry: MolecularGeometry,
    bonds: Sequence[tuple[int, int]],
    *,
    bond_orders: Mapping[tuple[int, int], float] | None = None,
    requested_counts: Sequence[int | None] | None = None,
) -> HydrogenCompletion:
    """Complete ordinary valences and place hydrogens deterministically.

    ``requested_counts`` carries explicit SMILES hydrogen semantics, including
    aromatic ``[nH]``.  Without it, ORACLE infers only unambiguous ordinary
    main-group valences and leaves P, S, metals and already hypercoordinate
    atoms untouched.
    """

    atoms = tuple(geometry.atoms)
    xyz = np.asarray(geometry.coordinates_angstrom, dtype=float)
    natoms = len(atoms)
    canonical_bonds = tuple(
        sorted({tuple(sorted((int(left), int(right)))) for left, right in bonds})
    )
    if any(left < 0 or right >= natoms or left == right for left, right in canonical_bonds):
        raise ValueError("hydrogen completion received an invalid bond")
    adjacency: list[list[int]] = [[] for _ in atoms]
    for left, right in canonical_bonds:
        adjacency[left].append(right)
        adjacency[right].append(left)
    numbers = tuple(_number(symbol) for symbol in atoms)
    orders = {
        tuple(sorted((int(left), int(right)))): float(value)
        for (left, right), value in (bond_orders or {}).items()
    }
    if requested_counts is not None and len(requested_counts) != natoms:
        raise ValueError("requested hydrogen counts must match the input atom count")

    output_atoms = list(atoms)
    output_xyz = [np.asarray(point, dtype=float) for point in xyz]
    output_bonds = list(canonical_bonds)
    additions: list[AddedHydrogen] = []
    planar_hydrogens: list[int] = []
    placement_rules: dict[int, str] = {}
    sphere = _fibonacci_sphere(512)

    for parent, number in enumerate(numbers):
        requested = None if requested_counts is None else requested_counts[parent]
        count = (
            _inferred_hydrogen_count(parent, number, adjacency, orders)
            if requested is None
            else int(requested)
        )
        if count < 0 or count > 4:
            raise ValueError(f"invalid hydrogen count {count} for atom {parent + 1}")
        if count == 0:
            continue
        occupied = []
        for neighbor in adjacency[parent]:
            vector = xyz[neighbor] - xyz[parent]
            norm = float(np.linalg.norm(vector))
            if norm > 1.0e-12:
                occupied.append(vector / norm)
        selected = _valence_hydrogen_directions(
            number,
            occupied,
            count,
            candidates=sphere,
        )
        coordination = len(occupied) + count
        if _is_planar_amide_nitrogen(
            parent,
            adjacency,
            orders,
            numbers,
            count,
        ):
            direction = -np.sum(np.asarray(occupied, dtype=float), axis=0)
            norm = float(np.linalg.norm(direction))
            if norm <= 1.0e-12:
                raise ValueError("cannot construct a planar amide N-H direction")
            selected = [direction / norm]
            placement_rules[parent] = "AMIDE_TRIGONAL_PLANAR"
        elif coordination == 2:
            placement_rules[parent] = "BENT_OR_LINEAR_COORDINATION_2"
        elif coordination == 3:
            placement_rules[parent] = "TRIGONAL_COORDINATION_3"
        elif coordination == 4:
            placement_rules[parent] = "TETRAHEDRAL_COORDINATION_4"
        else:
            placement_rules[parent] = f"COORDINATION_{coordination}"
        length = _hydrogen_bond_length(number, len(adjacency[parent]) + count)
        for direction in selected:
            atom_index = len(output_atoms)
            output_atoms.append("H")
            output_xyz.append(xyz[parent] + length * direction)
            output_bonds.append((parent, atom_index))
            additions.append(
                AddedHydrogen(
                    atom=atom_index,
                    parent=parent,
                    bond_length_angstrom=length,
                )
            )
            if _is_planar_amide_nitrogen(parent, adjacency, orders, numbers, count):
                planar_hydrogens.append(atom_index)

    completed = MolecularGeometry(
        atoms=tuple(output_atoms),
        coordinates_angstrom=np.asarray(output_xyz, dtype=float),
        comment=geometry.comment,
        source_format=geometry.source_format,
        source_path=geometry.source_path,
        charge=geometry.charge,
        multiplicity=geometry.multiplicity,
        metadata={
            **dict(geometry.metadata),
            "hydrogen_completion": HYDROGEN_COMPLETION_SCHEMA,
            "added_hydrogen_count": len(additions),
            "planar_amide_hydrogens": tuple(planar_hydrogens),
            "hydrogen_placement_rules": {
                str(parent + 1): rule for parent, rule in sorted(placement_rules.items())
            },
        },
    )
    _validate_hydrogen_completion(completed, additions, adjacency, numbers)
    return HydrogenCompletion(
        geometry=completed,
        bonds=tuple(sorted(output_bonds)),
        additions=tuple(additions),
    )


def release_completed_hydrogen_clashes(
    completion: HydrogenCompletion,
    *,
    minimum_separation_angstrom: float = 1.35,
    passes: int = 8,
    atom_indices: Sequence[int] | None = None,
) -> HydrogenCompletion:
    """Reorient only newly added X--H bonds away from Cartesian clashes.

    The heavy-atom geometry, X--H distances, constitution, and atom ordering
    remain unchanged.  This is intended for a completed geometry whose heavy
    skeleton has subsequently been assembled or back-transformed.
    """

    minimum = float(minimum_separation_angstrom)
    if not np.isfinite(minimum) or minimum <= 0.0:
        raise ValueError("minimum hydrogen separation must be positive")
    coordinates = np.asarray(completion.geometry.coordinates_angstrom, dtype=float).copy()
    selected_indices = None if atom_indices is None else {int(index) for index in atom_indices}
    selected = (
        completion.additions
        if selected_indices is None
        else tuple(
            addition for addition in completion.additions if int(addition.atom) in selected_indices
        )
    )
    if selected_indices is not None and len(selected) != len(selected_indices):
        raise ValueError("clash-release atom indices must identify newly added hydrogens")
    candidates = _fibonacci_sphere(512)
    for _ in range(max(1, int(passes))):
        for addition in selected:
            parent = int(addition.parent)
            hydrogen = int(addition.atom)
            origin = coordinates[parent]
            others = np.asarray(
                [
                    coordinates[index]
                    for index in range(len(coordinates))
                    if index not in {parent, hydrogen}
                ],
                dtype=float,
            )
            if not len(others):
                continue
            current = float(np.min(np.linalg.norm(others - coordinates[hydrogen], axis=1)))
            if current >= minimum:
                continue
            trial = origin + float(addition.bond_length_angstrom) * candidates
            clearances = np.min(
                np.linalg.norm(trial[:, np.newaxis, :] - others[np.newaxis, :, :], axis=2),
                axis=1,
            )
            coordinates[hydrogen] = trial[int(np.argmax(clearances))]
    geometry = MolecularGeometry(
        atoms=completion.geometry.atoms,
        coordinates_angstrom=coordinates,
        comment=completion.geometry.comment,
        source_format=completion.geometry.source_format,
        source_path=completion.geometry.source_path,
        charge=completion.geometry.charge,
        multiplicity=completion.geometry.multiplicity,
        fixed_parameters=completion.geometry.fixed_parameters,
        metadata={
            **dict(completion.geometry.metadata),
            "hydrogen_clash_release": "NEW_HYDROGENS_ONLY",
            "hydrogen_minimum_separation_angstrom": minimum,
            "hydrogen_clash_release_count": len(selected),
        },
    )
    return HydrogenCompletion(
        geometry=geometry,
        bonds=completion.bonds,
        additions=completion.additions,
        schema=completion.schema,
    )


def _inferred_hydrogen_count(
    atom: int,
    number: int,
    adjacency: Sequence[Sequence[int]],
    orders: Mapping[tuple[int, int], float],
) -> int:
    target = 2 if number == 16 else _DEFAULT_VALENCE.get(number)
    if target is None:
        return 0
    valence = sum(orders.get(tuple(sorted((atom, neighbor))), 1.0) for neighbor in adjacency[atom])
    missing = float(target) - valence
    if missing <= 0.35:
        return 0
    rounded = int(round(missing))
    if abs(missing - rounded) > 0.35:
        raise ValueError(
            f"ambiguous valence completion for atom {atom + 1}: missing valence {missing:.3f}"
        )
    return max(0, rounded)


def _is_planar_amide_nitrogen(
    parent: int,
    adjacency: Sequence[Sequence[int]],
    orders: Mapping[tuple[int, int], float],
    numbers: Sequence[int],
    count: int,
) -> bool:
    """Recognize an N-H amide from graph bond orders, not atom position."""

    if numbers[parent] != 7 or count != 1 or len(adjacency[parent]) != 2:
        return False
    for carbon in adjacency[parent]:
        if numbers[carbon] != 6:
            continue
        carbonyl_bond = orders.get(tuple(sorted((parent, carbon))), 1.0)
        if carbonyl_bond > 1.5:
            continue
        for oxygen in adjacency[carbon]:
            if numbers[oxygen] != 8:
                continue
            if orders.get(tuple(sorted((carbon, oxygen))), 1.0) > 1.5:
                return True
    return False


def _validate_hydrogen_completion(
    geometry: MolecularGeometry,
    additions: Sequence[AddedHydrogen],
    adjacency: Sequence[Sequence[int]],
    numbers: Sequence[int],
) -> None:
    coordinates = np.asarray(geometry.coordinates_angstrom, dtype=float)
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("hydrogen completion produced non-finite coordinates")
    for addition in additions:
        parent = int(addition.parent)
        hydrogen = int(addition.atom)
        distance = float(np.linalg.norm(coordinates[hydrogen] - coordinates[parent]))
        if not np.isfinite(distance) or distance <= 0.5 or distance >= 1.5:
            raise ValueError(
                f"invalid X-H distance for atom {hydrogen + 1}: {distance:.6f} Angstrom"
            )
        vectors = []
        for neighbour in adjacency[parent]:
            vector = coordinates[neighbour] - coordinates[parent]
            norm = float(np.linalg.norm(vector))
            if norm > 1.0e-12:
                vectors.append(vector / norm)
        vectors.append((coordinates[hydrogen] - coordinates[parent]) / distance)
        if len(vectors) < 2:
            continue
        hydrogen_direction = vectors[-1]
        if any(float(np.dot(reference, hydrogen_direction)) > 0.999 for reference in vectors[:-1]):
            raise ValueError(f"degenerate hydrogen coordination at atom {parent + 1}")
    del numbers  # retained in the signature for explicit validation context


def _least_crowded_direction(
    candidates: np.ndarray,
    occupied: Sequence[np.ndarray],
) -> np.ndarray:
    if not occupied:
        return np.asarray((1.0, 0.0, 0.0))
    references = np.asarray(occupied, dtype=float)
    # Minimize the largest cosine: maximize the smallest angle to every
    # existing bond.  Stable candidate order resolves exact ties.
    scores = np.max(candidates @ references.T, axis=1)
    return candidates[int(np.argmin(scores))]


def _valence_hydrogen_directions(
    atomic_number_value: int,
    occupied: Sequence[np.ndarray],
    count: int,
    *,
    candidates: np.ndarray,
) -> list[np.ndarray]:
    coordination = len(occupied) + count
    template = _valence_template(atomic_number_value, coordination)
    if template is None or len(occupied) >= coordination:
        selected: list[np.ndarray] = []
        for _ in range(count):
            direction = _least_crowded_direction(candidates, [*occupied, *selected])
            selected.append(direction)
        return selected
    if not occupied:
        return [np.asarray(vector, dtype=float) for vector in template[:count]]

    references = np.asarray(occupied, dtype=float)
    references /= np.linalg.norm(references, axis=1)[:, None]
    best_score = float("inf")
    best_rotation = np.eye(3)
    best_assignment: tuple[int, ...] = ()
    for assignment in permutations(range(coordination), len(occupied)):
        source = template[np.asarray(assignment, dtype=int)]
        rotation = _vector_alignment(source, references)
        residual = source @ rotation.T - references
        score = float(np.sum(residual * residual))
        if score < best_score - 1.0e-14:
            best_score = score
            best_rotation = rotation
            best_assignment = tuple(assignment)
    remaining = [index for index in range(coordination) if index not in best_assignment]
    return [
        np.asarray(template[index] @ best_rotation.T, dtype=float) for index in remaining[:count]
    ]


def _valence_template(
    atomic_number_value: int,
    coordination: int,
) -> np.ndarray | None:
    if coordination == 1:
        return np.asarray(((1.0, 0.0, 0.0),), dtype=float)
    if coordination == 2:
        angle = np.deg2rad(104.5 if atomic_number_value in {8, 16, 34, 52} else 180.0)
        half = 0.5 * angle
        return np.asarray(
            (
                (np.cos(half), np.sin(half), 0.0),
                (np.cos(half), -np.sin(half), 0.0),
            ),
            dtype=float,
        )
    if coordination == 3:
        if atomic_number_value in {7, 15, 33, 51}:
            pair_cosine = float(np.cos(np.deg2rad(107.0)))
            z = sqrt(max(0.0, (pair_cosine + 0.5) / 1.5))
            radial = sqrt(max(0.0, 1.0 - z * z))
            return np.asarray(
                tuple(
                    (
                        radial * np.cos(2.0 * pi * index / 3.0),
                        radial * np.sin(2.0 * pi * index / 3.0),
                        z,
                    )
                    for index in range(3)
                ),
                dtype=float,
            )
        return np.asarray(
            (
                (1.0, 0.0, 0.0),
                (-0.5, sqrt(3.0) / 2.0, 0.0),
                (-0.5, -sqrt(3.0) / 2.0, 0.0),
            ),
            dtype=float,
        )
    if coordination == 4:
        return np.asarray(
            (
                (1.0, 1.0, 1.0),
                (1.0, -1.0, -1.0),
                (-1.0, 1.0, -1.0),
                (-1.0, -1.0, 1.0),
            ),
            dtype=float,
        ) / sqrt(3.0)
    return None


def _vector_alignment(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if len(source) == 1:
        left = source[0] / np.linalg.norm(source[0])
        right = target[0] / np.linalg.norm(target[0])
        cross = np.cross(left, right)
        sine = float(np.linalg.norm(cross))
        cosine = float(np.dot(left, right))
        if sine < 1.0e-14:
            if cosine > 0.0:
                return np.eye(3)
            axis = np.asarray((1.0, 0.0, 0.0))
            if abs(float(np.dot(axis, left))) > 0.9:
                axis = np.asarray((0.0, 1.0, 0.0))
            axis -= float(np.dot(axis, left)) * left
            axis /= np.linalg.norm(axis)
            return 2.0 * np.outer(axis, axis) - np.eye(3)
        skew = np.asarray(
            (
                (0.0, -cross[2], cross[1]),
                (cross[2], 0.0, -cross[0]),
                (-cross[1], cross[0], 0.0),
            )
        )
        return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / (sine * sine))
    covariance = source.T @ target
    left_vectors, _, right_vectors_t = np.linalg.svd(covariance)
    rotation = right_vectors_t.T @ left_vectors.T
    if np.linalg.det(rotation) < 0.0:
        right_vectors_t[-1] *= -1.0
        rotation = right_vectors_t.T @ left_vectors.T
    return rotation


def _fibonacci_sphere(count: int) -> np.ndarray:
    golden = pi * (3.0 - sqrt(5.0))
    points = []
    for index in range(count):
        y = 1.0 - 2.0 * (index + 0.5) / count
        radius = sqrt(max(0.0, 1.0 - y * y))
        angle = golden * index
        points.append((radius * np.cos(angle), y, radius * np.sin(angle)))
    return np.asarray(points, dtype=float)


def _hydrogen_bond_length(parent_number: int, coordination: int) -> float:
    parent = covalent_radius(parent_number, coordination)
    hydrogen = covalent_radius(1, 1)
    if parent is None or hydrogen is None:
        raise ValueError(f"no covalent radius for hydrogen parent Z={parent_number}")
    return float(parent + hydrogen)


def _number(symbol: str) -> int:
    value = atomic_number(symbol)
    if value is None or value < 1:
        raise ValueError(f"invalid atom symbol for hydrogen completion: {symbol!r}")
    return int(value)


__all__ = [
    "HYDROGEN_COMPLETION_SCHEMA",
    "AddedHydrogen",
    "HydrogenCompletion",
    "complete_valence_hydrogens",
    "release_completed_hydrogen_clashes",
]
