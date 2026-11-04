"""Shared directional-contact contract for ORACLE, SENTINEL and ZAFF."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Iterable, Mapping, Sequence

import numpy as np


DIRECTIONAL_CONTACT_SCHEMA = "matrix.directional_contacts.v2"
HALOGEN_ATOMIC_NUMBERS = frozenset({9, 17, 35, 53, 85})
HALOGEN_SYMBOLS = frozenset({"F", "CL", "BR", "I", "AT"})
XB_ACCEPTOR_ATOMIC_NUMBERS = frozenset({7, 8, 16})
XB_ACCEPTOR_SYMBOLS = frozenset({"N", "O", "S"})
XB_DEFAULTS = {
    "distance_cutoff_angstrom": 3.5,
    "distance_width_angstrom": 0.25,
    "angle_center_degrees": 180.0,
    "angle_width_degrees": 20.0,
}

DIRECTIONAL_CENTER_ATOMIC_NUMBERS = {
    "tetrel-bond": frozenset({6, 14, 32, 50, 82}),
    "pnictogen-bond": frozenset({7, 15, 33, 51, 83}),
    "chalcogen-bond": frozenset({8, 16, 34, 52, 84}),
    "halogen-bond": HALOGEN_ATOMIC_NUMBERS,
}
# First-row C/N/O centers are valid in explicitly assigned models but are too
# ubiquitous for conservative topology-only discovery.
AUTOMATIC_DIRECTIONAL_CENTER_ATOMIC_NUMBERS = {
    "tetrel-bond": frozenset({14, 32, 50, 82}),
    "pnictogen-bond": frozenset({15, 33, 51, 83}),
    "chalcogen-bond": frozenset({16, 34, 52, 84}),
    "halogen-bond": HALOGEN_ATOMIC_NUMBERS,
}
DIRECTIONAL_ACCEPTOR_ATOMIC_NUMBERS = frozenset({7, 8, 9, 15, 16, 17, 34, 35, 53})
DIRECTIONAL_SCREENING_DEFAULTS = {
    "distance_cutoff_angstrom": 4.0,
    "distance_width_angstrom": 0.30,
    "angle_cutoff_degrees": 140.0,
    "angle_width_degrees": 20.0,
    "strength_cutoff": 0.20,
}


@dataclass(frozen=True)
class DirectionalContact:
    """One conservative axis--center···acceptor directional contact."""

    kind: str
    anchor: int
    center: int
    acceptor: int
    distance_angstrom: float
    angle_degrees: float
    strength: float
    schema: str = DIRECTIONAL_CONTACT_SCHEMA


def perceive_directional_contacts(
    atomic_numbers: Sequence[int],
    coordinates_angstrom: np.ndarray,
    bonded_pairs: Iterable[tuple[int, int]],
    *,
    kinds: Sequence[str] = (
        "halogen-bond",
        "chalcogen-bond",
        "pnictogen-bond",
        "tetrel-bond",
    ),
    screening: Mapping[str, float] | None = None,
) -> tuple[DirectionalContact, ...]:
    """Discover conservative sigma/pi-hole contacts without parameterizing them.

    The returned ``strength`` is a smooth screening score only.  It is not a
    force-field parameter and does not authorize a CM5 response amplitude.
    Explicitly assigned first-row centers remain supported by downstream
    contracts but are excluded from this automatic topology-only detector.
    """

    numbers = tuple(int(value) for value in atomic_numbers)
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    if xyz.shape != (len(numbers), 3) or np.any(~np.isfinite(xyz)):
        raise ValueError("coordinates must be a finite (natoms, 3) array")
    normalized_kinds = tuple(_normalize_directional_kind(kind) for kind in kinds)
    unsupported = set(normalized_kinds) - set(DIRECTIONAL_CENTER_ATOMIC_NUMBERS)
    if unsupported:
        raise ValueError(
            "unsupported automatic directional contact kinds: "
            + ", ".join(sorted(unsupported))
        )
    settings = {**DIRECTIONAL_SCREENING_DEFAULTS, **dict(screening or {})}
    distance_cutoff = float(settings["distance_cutoff_angstrom"])
    distance_width = float(settings["distance_width_angstrom"])
    angle_cutoff = float(settings["angle_cutoff_degrees"])
    angle_width = float(settings["angle_width_degrees"])
    strength_cutoff = float(settings["strength_cutoff"])
    if distance_cutoff <= 0.0 or distance_width <= 0.0:
        raise ValueError("directional distance screening parameters must be positive")
    if not 0.0 < angle_cutoff < 180.0 or angle_width <= 0.0:
        raise ValueError("directional angle screening parameters are invalid")
    if not 0.0 <= strength_cutoff <= 1.0:
        raise ValueError("directional strength cutoff must lie in [0, 1]")

    bonds = {tuple(sorted((int(left), int(right)))) for left, right in bonded_pairs}
    adjacency = [set() for _ in numbers]
    for left, right in bonds:
        if left < 0 or right >= len(numbers) or left == right:
            raise ValueError("bonded pairs contain an invalid atom index")
        adjacency[left].add(right)
        adjacency[right].add(left)
    contacts = []
    for kind in normalized_kinds:
        centers = AUTOMATIC_DIRECTIONAL_CENTER_ATOMIC_NUMBERS[kind]
        for center, atomic_number in enumerate(numbers):
            if atomic_number not in centers:
                continue
            for anchor in sorted(adjacency[center]):
                axis = xyz[anchor] - xyz[center]
                axis_norm = float(np.linalg.norm(axis))
                if axis_norm <= 1.0e-12:
                    continue
                for acceptor, acceptor_number in enumerate(numbers):
                    if (
                        acceptor_number not in DIRECTIONAL_ACCEPTOR_ATOMIC_NUMBERS
                        or acceptor in {anchor, center}
                        or acceptor in adjacency[center]
                    ):
                        continue
                    radial = xyz[acceptor] - xyz[center]
                    distance = float(np.linalg.norm(radial))
                    if distance <= 1.0e-12 or distance > distance_cutoff:
                        continue
                    cosine = float(
                        np.clip(np.dot(axis, radial) / (axis_norm * distance), -1.0, 1.0)
                    )
                    angle = float(np.degrees(np.arccos(cosine)))
                    if angle < angle_cutoff:
                        continue
                    radial_score = 1.0 / (
                        1.0 + exp((distance - distance_cutoff) / distance_width)
                    )
                    angle_score = 1.0 / (
                        1.0 + exp((angle_cutoff - angle) / angle_width)
                    )
                    strength = float(radial_score * angle_score)
                    if strength >= strength_cutoff:
                        contacts.append(
                            DirectionalContact(
                                kind=kind,
                                anchor=anchor,
                                center=center,
                                acceptor=acceptor,
                                distance_angstrom=distance,
                                angle_degrees=angle,
                                strength=strength,
                            )
                        )
    return tuple(
        sorted(
            contacts,
            key=lambda item: (item.kind, item.center, item.anchor, item.acceptor),
        )
    )


def _normalize_directional_kind(kind: str) -> str:
    value = str(kind).strip().lower().replace("_", "-")
    return {"xb": "halogen-bond", "xbond": "halogen-bond"}.get(value, value)


def directional_axis_angle_degrees(
    coordinates_angstrom: np.ndarray,
    anchor: int,
    halogen: int,
    acceptor: int,
) -> float:
    """Return the anchor--X...acceptor angle in degrees (180° is ideal)."""

    xyz = np.asarray(coordinates_angstrom, dtype=float)
    axis = xyz[int(anchor)] - xyz[int(halogen)]
    target = xyz[int(acceptor)] - xyz[int(halogen)]
    denominator = float(np.linalg.norm(axis) * np.linalg.norm(target))
    if denominator <= 1.0e-14:
        return 0.0
    return float(
        np.degrees(
            np.arccos(np.clip(float(np.dot(axis, target) / denominator), -1.0, 1.0))
        )
    )


def normalize_element_symbols(elements: Sequence[str]) -> tuple[str, ...]:
    """Normalize symbols used by topology-only automatic detectors."""

    return tuple(str(value).strip().upper() for value in elements)


__all__ = [
    "AUTOMATIC_DIRECTIONAL_CENTER_ATOMIC_NUMBERS",
    "DIRECTIONAL_CONTACT_SCHEMA",
    "DIRECTIONAL_ACCEPTOR_ATOMIC_NUMBERS",
    "DIRECTIONAL_CENTER_ATOMIC_NUMBERS",
    "DIRECTIONAL_SCREENING_DEFAULTS",
    "DirectionalContact",
    "HALOGEN_ATOMIC_NUMBERS",
    "HALOGEN_SYMBOLS",
    "XB_ACCEPTOR_ATOMIC_NUMBERS",
    "XB_ACCEPTOR_SYMBOLS",
    "XB_DEFAULTS",
    "directional_axis_angle_degrees",
    "normalize_element_symbols",
    "perceive_directional_contacts",
]
