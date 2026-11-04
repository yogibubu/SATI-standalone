"""Shared numerical kernels for ORACLE-owned local chemical perception.

The functions in this module are deliberately free of workflow ownership.
ORACLE calls them to make and serialize semantic decisions; SMITH may reuse
the same numerical kernels for legacy compatibility, but the frozen
ORACLE-to-SMITH contract remains the authoritative source of those decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


LOCAL_ZEFF_TOLERANCE = 5.0e-4
LOCAL_DISTANCE_TOLERANCE_ANGSTROM = 1.0e-3
LOCAL_TEMPLATE_RMS_THRESHOLD = 1.2e-1
LOCAL_TEMPLATE_MIN_MARGIN = 2.0e-2
LOCAL_ANGLE_CLASS_TOLERANCE = 2.0e-2
LOCAL_THRESHOLD_SENSITIVITY_FRACTION = 1.0e-1


@dataclass(frozen=True)
class LocalPerceptionSettings:
    """Validated thresholds for local equivalence and template recognition."""

    zeff_tolerance: float = LOCAL_ZEFF_TOLERANCE
    distance_tolerance_angstrom: float = LOCAL_DISTANCE_TOLERANCE_ANGSTROM
    template_rms_threshold: float = LOCAL_TEMPLATE_RMS_THRESHOLD
    template_min_margin: float = LOCAL_TEMPLATE_MIN_MARGIN
    angle_class_tolerance: float = LOCAL_ANGLE_CLASS_TOLERANCE

    def __post_init__(self) -> None:
        for name, value in (
            ("zeff_tolerance", self.zeff_tolerance),
            ("distance_tolerance_angstrom", self.distance_tolerance_angstrom),
            ("template_rms_threshold", self.template_rms_threshold),
            ("angle_class_tolerance", self.angle_class_tolerance),
        ):
            if not np.isfinite(value) or float(value) <= 0.0:
                raise ValueError(f"{name} must be a positive finite number")
        if not np.isfinite(self.template_min_margin) or self.template_min_margin < 0.0:
            raise ValueError("template_min_margin must be a non-negative finite number")


@dataclass(frozen=True)
class LocalCoordinationTemplate:
    name: str
    directions: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class LocalCoordinationMatch:
    """Rotation/permutation-invariant local-template decision."""

    template: LocalCoordinationTemplate | None
    best_template: LocalCoordinationTemplate | None
    score: float
    margin: float
    status: str
    rms_headroom: float = float("inf")
    margin_headroom: float = float("inf")
    sensitivity: str = "NOT_APPLICABLE"
    competing_template: LocalCoordinationTemplate | None = None
    competing_score: float = float("inf")
    nearest_template: LocalCoordinationTemplate | None = None


def local_ligand_equivalence_classes(
    center: int,
    neighbors: tuple[int, ...] | list[int],
    *,
    effective_atomic_numbers: tuple[float, ...],
    coordinates_angstrom: np.ndarray,
    zeff_tolerance: float = LOCAL_ZEFF_TOLERANCE,
    distance_tolerance_angstrom: float = LOCAL_DISTANCE_TOLERANCE_ANGSTROM,
) -> tuple[tuple[int, ...], ...]:
    """Return deterministic local ligand classes in zero-based indexing."""

    xyz = _validated_coordinates(coordinates_angstrom, len(effective_atomic_numbers))
    groups: list[list[int]] = []
    keys: list[tuple[float, float]] = []
    for atom in sorted(int(value) for value in neighbors):
        if atom < 0 or atom >= len(effective_atomic_numbers) or atom == int(center):
            raise ValueError("local ligand index is invalid")
        distance = float(np.linalg.norm(xyz[atom] - xyz[int(center)]))
        key = (float(effective_atomic_numbers[atom]), distance)
        match = next(
            (
                index
                for index, other in enumerate(keys)
                if abs(key[0] - other[0]) <= float(zeff_tolerance)
                and abs(key[1] - other[1]) <= float(distance_tolerance_angstrom)
            ),
            None,
        )
        if match is None:
            keys.append(key)
            groups.append([atom])
        else:
            groups[match].append(atom)
    return tuple(tuple(group) for _key, group in sorted(zip(keys, groups)))


def local_coordination_match(
    center: int,
    neighbors: tuple[int, ...] | list[int],
    *,
    coordinates_angstrom: np.ndarray,
    max_rms_cosine_error: float = LOCAL_TEMPLATE_RMS_THRESHOLD,
    min_score_margin: float = LOCAL_TEMPLATE_MIN_MARGIN,
) -> LocalCoordinationMatch:
    """Match a coordination sphere without depending on orientation or ordering."""

    xyz = np.asarray(coordinates_angstrom, dtype=float)
    actual = sorted_pair_cosines(local_ligand_unit_vectors(center, neighbors, xyz))
    scores: list[tuple[float, str, LocalCoordinationTemplate]] = []
    for template in local_coordination_templates(len(neighbors)):
        ideal = _template_pair_cosines(template)
        if len(ideal) == len(actual):
            score = float(np.sqrt(np.mean((actual - ideal) ** 2)))
            scores.append((score, template.name, template))
    if not scores:
        return LocalCoordinationMatch(None, None, float("inf"), float("inf"), "GENERIC")
    scores.sort(key=lambda item: (item[0], item[1]))
    best_score, _name, best_template = scores[0]
    if len(scores) > 1:
        second_score, _second_name, second_template = scores[1]
    else:
        second_score, second_template = float("inf"), None
    margin = float(second_score - best_score)
    rms_headroom = float(max_rms_cosine_error - best_score)
    margin_headroom = float(margin - min_score_margin)
    sensitivity = template_threshold_sensitivity(
        rms_headroom,
        margin_headroom,
        rms_threshold=max_rms_cosine_error,
        margin_threshold=min_score_margin,
    )
    common = dict(
        best_template=best_template,
        nearest_template=best_template,
        score=best_score,
        margin=margin,
        rms_headroom=rms_headroom,
        margin_headroom=margin_headroom,
        sensitivity=sensitivity,
        competing_template=second_template,
        competing_score=float(second_score),
    )
    effective_rms_threshold = (
        min(float(max_rms_cosine_error), 8.0e-2) if len(neighbors) >= 10 else float(max_rms_cosine_error)
    )
    if best_score > effective_rms_threshold:
        return LocalCoordinationMatch(template=None, status="GENERIC", **common)
    # Dense high-coordination templates can have similar unordered pair-
    # cosine spectra (notably the two CN11 references).  Preserve the
    # ambiguity guard unless the best match is itself decisively smaller than
    # their measured separation; an exact catalog geometry must remain
    # recognizable.
    decisively_closer = best_score <= max(1.0e-12, 0.25 * margin)
    if len(neighbors) < 10 and margin < float(min_score_margin) and not decisively_closer:
        return LocalCoordinationMatch(template=None, status="AMBIGUOUS", **common)
    return LocalCoordinationMatch(template=best_template, status="FROZEN", **common)


def infer_local_pseudogroup(
    center: int,
    neighbors: tuple[int, ...] | list[int],
    classes: tuple[tuple[int, ...], ...],
    *,
    coordinates_angstrom: np.ndarray,
    settings: LocalPerceptionSettings | None = None,
    match: LocalCoordinationMatch | None = None,
) -> tuple[str, str]:
    """Return the existing conservative local group and confidence policy."""

    policy = settings or LocalPerceptionSettings()
    local_match = match or local_coordination_match(
        center,
        neighbors,
        coordinates_angstrom=coordinates_angstrom,
        max_rms_cosine_error=policy.template_rms_threshold,
        min_score_margin=policy.template_min_margin,
    )
    multiplicities = sorted((len(group) for group in classes), reverse=True)
    template, score = local_match.template, local_match.score
    if template is not None:
        group_by_template = {
            "TETRAHEDRAL": "Td",
            "SQUARE_PLANAR": "D4h",
            "TRIGONAL_BIPYRAMIDAL": "D3h",
            "SQUARE_PYRAMIDAL": "C4v",
            "OCTAHEDRAL": "Oh",
            "TRIGONAL_PRISMATIC": "D3h",
            "PENTAGONAL_BIPYRAMIDAL": "D5h",
            "CAPPED_OCTAHEDRAL": "C3v",
            "SQUARE_ANTIPRISMATIC": "D4d",
            "DODECAHEDRAL_LIKE": "D2d",
            "TRICAPPED_TRIGONAL_PRISMATIC": "D3h",
            "CAPPED_SQUARE_ANTIPRISMATIC": "C4v",
            "BICAPPED_SQUARE_ANTIPRISMATIC": "D4d",
            "BICAPPED_DODECAHEDRAL": "D2d",
            "CAPPED_ICOSAHEDRAL": "C5v",
            "OCTADECAHEDRAL": "C2v",
            "ICOSAHEDRAL": "Ih",
            "CUBOCTAHEDRAL": "Oh",
            "HEXAGONAL_BIPYRAMIDAL": "D6h",
            "ANTICUBOCTAHEDRAL": "D3h",
            "CAPPED_CUBOCTAHEDRAL": "C3v",
            "BICAPPED_HEXAGONAL_ANTIPRISMATIC": "D6d",
        }
        base = group_by_template.get(template.name, "C1")
        if len(classes) == 1:
            return base, "HIGH" if score <= 0.06 else "MEDIUM"
    if len(neighbors) == 2 and multiplicities == [2]:
        return "C2v", "HIGH"
    if len(neighbors) == 3 and multiplicities == [3]:
        vectors = local_ligand_unit_vectors(center, neighbors, coordinates_angstrom)
        planarity = abs(float(np.linalg.det(vectors)))
        return ("D3h" if planarity <= 0.08 else "C3v"), "HIGH"
    if len(neighbors) == 4 and multiplicities == [4]:
        cosines = sorted_pair_cosines(
            local_ligand_unit_vectors(center, neighbors, coordinates_angstrom)
        )
        tetra_error = float(np.sqrt(np.mean((cosines + 1.0 / 3.0) ** 2)))
        if tetra_error <= 0.12:
            return "Td", "HIGH" if tetra_error <= 0.06 else "MEDIUM"
        return "D4h", "MEDIUM"
    if multiplicities and multiplicities[0] >= 3:
        return "C3v", "MEDIUM"
    if multiplicities[:2] == [2, 2]:
        return "C2v", "MEDIUM"
    if multiplicities and multiplicities[0] == 2:
        return "Cs", "MEDIUM"
    return "C1", "HIGH"


def ring_local_pseudogroup(
    ring: tuple[int, ...],
    *,
    effective_atomic_numbers: tuple[float, ...],
    coordinates_angstrom: np.ndarray,
    settings: LocalPerceptionSettings | None = None,
) -> tuple[str, str, int]:
    """Return color-preserving local ring operations using the shared policy."""

    policy = settings or LocalPerceptionSettings()
    xyz = _validated_coordinates(coordinates_angstrom, len(effective_atomic_numbers))
    size = len(ring)
    colors = tuple(
        (
            round(float(effective_atomic_numbers[atom]) / policy.zeff_tolerance),
            round(
                float(np.linalg.norm(xyz[atom] - xyz[ring[(index + 1) % size]]))
                / policy.distance_tolerance_angstrom
            ),
        )
        for index, atom in enumerate(ring)
    )
    rotations = sum(
        all(colors[index] == colors[(index + shift) % size] for index in range(size))
        for shift in range(size)
    )
    reflections = sum(
        all(colors[index] == colors[(shift - index) % size] for index in range(size))
        for shift in range(size)
    )
    if rotations > 1 and reflections > 0:
        group = f"D{rotations}"
    elif rotations > 1:
        group = f"C{rotations}"
    elif reflections > 0:
        group = "Cs"
    else:
        group = "C1"
    return group, ("HIGH" if rotations + reflections > 1 else "MEDIUM"), rotations + reflections


def template_threshold_sensitivity(
    rms_headroom: float,
    margin_headroom: float,
    *,
    rms_threshold: float,
    margin_threshold: float,
) -> str:
    near_rms = abs(float(rms_headroom)) <= max(
        1.0e-6,
        LOCAL_THRESHOLD_SENSITIVITY_FRACTION * abs(float(rms_threshold)),
    )
    near_margin = np.isfinite(margin_headroom) and abs(float(margin_headroom)) <= max(
        1.0e-6,
        LOCAL_THRESHOLD_SENSITIVITY_FRACTION
        * max(abs(float(margin_threshold)), LOCAL_TEMPLATE_MIN_MARGIN),
    )
    if near_rms and near_margin:
        return "NEAR_BOTH"
    if near_rms:
        return "NEAR_RMS"
    if near_margin:
        return "NEAR_MARGIN"
    return "STABLE"


@lru_cache(maxsize=None)
def template_pair_cosine_classes(
    template: LocalCoordinationTemplate,
    tolerance: float = LOCAL_ANGLE_CLASS_TOLERANCE,
) -> tuple[float, ...]:
    cosines = _template_pair_cosines(template)
    classes: list[float] = []
    for value in cosines:
        if not classes or abs(float(value) - classes[-1]) > float(tolerance):
            classes.append(float(value))
        else:
            classes[-1] = 0.5 * (classes[-1] + float(value))
    return tuple(classes)


def nearest_cosine_class(value: float, classes: tuple[float, ...]) -> int:
    if not classes:
        return 0
    return min(range(len(classes)), key=lambda index: abs(float(value) - classes[index]))


def ligand_pair_cosine(
    center: int,
    first: int,
    second: int,
    coordinates_angstrom: np.ndarray,
) -> float:
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    first_vector = xyz[int(first)] - xyz[int(center)]
    second_vector = xyz[int(second)] - xyz[int(center)]
    first_norm = float(np.linalg.norm(first_vector))
    second_norm = float(np.linalg.norm(second_vector))
    if first_norm == 0.0 or second_norm == 0.0:
        return 1.0
    return float(np.dot(first_vector, second_vector) / (first_norm * second_norm))


def local_ligand_unit_vectors(
    center: int,
    neighbors: tuple[int, ...] | list[int],
    coordinates_angstrom: np.ndarray,
) -> np.ndarray:
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    vectors = xyz[np.asarray(neighbors, dtype=int)] - xyz[int(center)]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms != 0.0)


def sorted_pair_cosines(vectors: np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    normalized = np.divide(array, norms, out=np.zeros_like(array), where=norms != 0.0)
    cosine_matrix = normalized @ normalized.T
    upper = cosine_matrix[np.triu_indices(len(normalized), k=1)]
    return np.sort(np.asarray(upper, dtype=float))


@lru_cache(maxsize=None)
def _template_pair_cosines(template: LocalCoordinationTemplate) -> np.ndarray:
    return sorted_pair_cosines(np.asarray(template.directions, dtype=float))


def local_coordination_templates(coordination: int) -> tuple[LocalCoordinationTemplate, ...]:
    return LOCAL_COORDINATION_TEMPLATES.get(int(coordination), ())


def _regular_polygon_directions(count: int, *, z: float = 0.0, phase: float = 0.0):
    radius = float(np.sqrt(max(0.0, 1.0 - z * z)))
    return tuple(
        (
            radius * np.cos(phase + 2.0 * np.pi * index / count),
            radius * np.sin(phase + 2.0 * np.pi * index / count),
            z,
        )
        for index in range(count)
    )


def _normalized_directions(*directions: tuple[float, float, float]):
    normalized = []
    for direction in directions:
        vector = np.asarray(direction, dtype=float)
        norm = float(np.linalg.norm(vector))
        normalized.append(tuple((vector / norm).tolist()) if norm else tuple(vector.tolist()))
    return tuple(normalized)


LOCAL_COORDINATION_TEMPLATES: dict[int, tuple[LocalCoordinationTemplate, ...]] = {
    4: (
        LocalCoordinationTemplate(
            "TETRAHEDRAL",
            _normalized_directions(
                (1.0, 1.0, 1.0),
                (1.0, -1.0, -1.0),
                (-1.0, 1.0, -1.0),
                (-1.0, -1.0, 1.0),
            ),
        ),
        LocalCoordinationTemplate(
            "SQUARE_PLANAR",
            ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
        ),
    ),
    5: (
        LocalCoordinationTemplate(
            "TRIGONAL_BIPYRAMIDAL",
            _regular_polygon_directions(3) + ((0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
        ),
        LocalCoordinationTemplate(
            "SQUARE_PYRAMIDAL",
            _regular_polygon_directions(4, z=-0.35, phase=np.pi / 4.0)
            + ((0.0, 0.0, 1.0),),
        ),
    ),
    6: (
        LocalCoordinationTemplate(
            "OCTAHEDRAL",
            (
                (1.0, 0.0, 0.0),
                (-1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, -1.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, -1.0),
            ),
        ),
        LocalCoordinationTemplate(
            "TRIGONAL_PRISMATIC",
            _regular_polygon_directions(3, z=0.55)
            + _regular_polygon_directions(3, z=-0.55),
        ),
    ),
    7: (
        LocalCoordinationTemplate(
            "PENTAGONAL_BIPYRAMIDAL",
            _regular_polygon_directions(5) + ((0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
        ),
        LocalCoordinationTemplate(
            "CAPPED_OCTAHEDRAL",
            (
                (1.0, 0.0, 0.0),
                (-1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, -1.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, -1.0),
                (1.0, 1.0, 1.0),
            ),
        ),
    ),
    8: (
        LocalCoordinationTemplate(
            "SQUARE_ANTIPRISMATIC",
            _regular_polygon_directions(4, z=0.45)
            + _regular_polygon_directions(4, z=-0.45, phase=np.pi / 4.0),
        ),
        LocalCoordinationTemplate(
            "DODECAHEDRAL_LIKE",
            _normalized_directions(
                (1.0, 1.0, 1.0),
                (1.0, 1.0, -1.0),
                (1.0, -1.0, 1.0),
                (1.0, -1.0, -1.0),
                (-1.0, 1.0, 1.0),
                (-1.0, 1.0, -1.0),
                (-1.0, -1.0, 1.0),
                (-1.0, -1.0, -1.0),
            ),
        ),
    ),
    9: (
        LocalCoordinationTemplate(
            "TRICAPPED_TRIGONAL_PRISMATIC",
            _regular_polygon_directions(3, z=0.58)
            + _regular_polygon_directions(3, z=-0.58)
            + _regular_polygon_directions(3, z=0.0, phase=np.pi / 3.0),
        ),
        LocalCoordinationTemplate(
            "CAPPED_SQUARE_ANTIPRISMATIC",
            _regular_polygon_directions(4, z=0.42)
            + _regular_polygon_directions(4, z=-0.42, phase=np.pi / 4.0)
            + ((0.0, 0.0, 1.0),),
        ),
    ),
    10: (
        LocalCoordinationTemplate(
            "BICAPPED_SQUARE_ANTIPRISMATIC",
            _regular_polygon_directions(4, z=0.45)
            + _regular_polygon_directions(4, z=-0.45, phase=np.pi / 4.0)
            + ((0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
        ),
        LocalCoordinationTemplate(
            "BICAPPED_DODECAHEDRAL",
            _normalized_directions(
                (0.0, 1.0, 1.61803398875), (0.0, -1.0, 1.61803398875),
                (0.0, 1.0, -1.61803398875), (0.0, -1.0, -1.61803398875),
                (1.0, 1.61803398875, 0.0), (-1.0, 1.61803398875, 0.0),
                (1.0, -1.61803398875, 0.0), (-1.0, -1.61803398875, 0.0),
                (0.0, 0.0, 1.0), (0.0, 0.0, -1.0),
            ),
        ),
    ),
    11: (
        LocalCoordinationTemplate(
            "CAPPED_ICOSAHEDRAL",
            _normalized_directions(
                (0.0, 1.0, 1.61803398875), (0.0, -1.0, 1.61803398875),
                (0.0, 1.0, -1.61803398875), (0.0, -1.0, -1.61803398875),
                (1.0, 1.61803398875, 0.0), (-1.0, 1.61803398875, 0.0),
                (1.0, -1.61803398875, 0.0), (-1.0, -1.61803398875, 0.0),
                (1.61803398875, 0.0, 1.0), (1.61803398875, 0.0, -1.0),
                (-1.61803398875, 0.0, 1.0),
            ),
        ),
        LocalCoordinationTemplate(
            "OCTADECAHEDRAL",
            _regular_polygon_directions(5, z=0.45)
            + _regular_polygon_directions(5, z=-0.45, phase=np.pi / 5.0)
            + ((0.0, 0.0, 1.0),),
        ),
    ),
    12: (
        LocalCoordinationTemplate(
            "ICOSAHEDRAL",
            _normalized_directions(
                (0.0, 1.0, 1.61803398875), (0.0, -1.0, 1.61803398875),
                (0.0, 1.0, -1.61803398875), (0.0, -1.0, -1.61803398875),
                (1.0, 1.61803398875, 0.0), (-1.0, 1.61803398875, 0.0),
                (1.0, -1.61803398875, 0.0), (-1.0, -1.61803398875, 0.0),
                (1.61803398875, 0.0, 1.0), (1.61803398875, 0.0, -1.0),
                (-1.61803398875, 0.0, 1.0), (-1.61803398875, 0.0, -1.0),
            ),
        ),
        LocalCoordinationTemplate(
            "CUBOCTAHEDRAL",
            _normalized_directions(
                (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
                (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
                (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
            ),
        ),
        LocalCoordinationTemplate(
            "HEXAGONAL_BIPYRAMIDAL",
            _regular_polygon_directions(6, z=0.5)
            + _regular_polygon_directions(6, z=-0.5),
        ),
        LocalCoordinationTemplate(
            "ANTICUBOCTAHEDRAL",
            _regular_polygon_directions(6, z=0.0)
            + _regular_polygon_directions(3, z=0.7, phase=np.pi / 6.0)
            + _regular_polygon_directions(3, z=-0.7, phase=np.pi / 6.0),
        ),
    ),
    13: (
        LocalCoordinationTemplate(
            "CAPPED_CUBOCTAHEDRAL",
            _normalized_directions(
                (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
                (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
                (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
                (0, 0, 1),
            ),
        ),
    ),
    14: (
        LocalCoordinationTemplate(
            "BICAPPED_HEXAGONAL_ANTIPRISMATIC",
            _regular_polygon_directions(6, z=0.45)
            + _regular_polygon_directions(6, z=-0.45, phase=np.pi / 6.0)
            + ((0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
        ),
    ),
}


def _validated_coordinates(coordinates: np.ndarray, natoms: int) -> np.ndarray:
    xyz = np.asarray(coordinates, dtype=float)
    if xyz.shape != (int(natoms), 3) or np.any(~np.isfinite(xyz)):
        raise ValueError("local perception coordinates must be finite (natoms, 3)")
    return xyz


__all__ = [
    "LOCAL_ANGLE_CLASS_TOLERANCE",
    "LOCAL_COORDINATION_TEMPLATES",
    "LOCAL_DISTANCE_TOLERANCE_ANGSTROM",
    "LOCAL_TEMPLATE_MIN_MARGIN",
    "LOCAL_TEMPLATE_RMS_THRESHOLD",
    "LOCAL_THRESHOLD_SENSITIVITY_FRACTION",
    "LOCAL_ZEFF_TOLERANCE",
    "LocalCoordinationMatch",
    "LocalCoordinationTemplate",
    "LocalPerceptionSettings",
    "infer_local_pseudogroup",
    "ligand_pair_cosine",
    "local_coordination_match",
    "local_coordination_templates",
    "local_ligand_equivalence_classes",
    "local_ligand_unit_vectors",
    "nearest_cosine_class",
    "ring_local_pseudogroup",
    "sorted_pair_cosines",
    "template_pair_cosine_classes",
    "template_threshold_sensitivity",
]
