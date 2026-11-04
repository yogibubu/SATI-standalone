"""ORACLE-owned local equivalence and pseudosymmetry provider."""

from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from matrix_core import read_sectioned_lines, section_content
from matrix_chem import (
    LocalEquivalenceClass,
    LocalPerceptionDomain,
    LocalPerceptionSettings,
    LocalTemplateDecision,
    infer_local_pseudogroup,
    local_coordination_match,
    local_ligand_equivalence_classes,
    ring_local_pseudogroup,
)


ORACLE_LOCAL_PERCEPTION_PROVIDER = "ORACLE_LOCAL_EQUIVALENCE_AND_PSEUDOSYMMETRY"
ORACLE_LOCAL_PERCEPTION_PROVIDER_VERSION = "1"


def read_frozen_effective_atomic_numbers(
    path: Path,
    *,
    natoms: int,
) -> tuple[float, ...]:
    """Read the complete frozen ``ZEFF`` vector from ``#SYNTHONS``.

    Local equivalence must be based on the ORACLE state already serialized in
    the enriched XYZ, not on a second topology/perception pass performed while
    building the downstream contract.
    """

    content = section_content(read_sectioned_lines(Path(path)), "SYNTHONS")
    columns: tuple[str, ...] = ()
    for raw in content:
        fields = raw.split()
        if fields and fields[0].upper() == "COLUMNS":
            columns = tuple(value.upper() for value in fields[1:])
            break
    if not columns or "ATOM" not in columns or "ZEFF" not in columns:
        raise ValueError("#SYNTHONS must declare complete ATOM and ZEFF columns")
    atom_column = columns.index("ATOM")
    zeff_column = columns.index("ZEFF")
    values: dict[int, float] = {}
    for raw in content:
        fields = raw.split()
        if not fields or not fields[0].isdigit() or len(fields) < len(columns):
            continue
        atom = int(fields[atom_column])
        zeff = float(fields[zeff_column])
        if atom in values or atom < 1 or atom > int(natoms) or not math.isfinite(zeff):
            raise ValueError("#SYNTHONS contains an invalid or duplicate ZEFF record")
        values[atom] = zeff
    expected = set(range(1, int(natoms) + 1))
    if set(values) != expected:
        raise ValueError("#SYNTHONS does not contain one ZEFF value for every atom")
    return tuple(values[atom] for atom in sorted(values))


def perceive_local_perception_domains(
    atomic_numbers: Sequence[int],
    coordinates_angstrom: np.ndarray,
    bonded_pairs: Iterable[tuple[int, int]],
    rings: Iterable[tuple[int, ...]],
    *,
    effective_atomic_numbers: Sequence[float],
    settings: LocalPerceptionSettings | None = None,
) -> tuple[LocalPerceptionDomain, ...]:
    """Return immutable local decisions in one-based contract indexing.

    Inputs use zero-based atom indices.  The function is deterministic and
    invariant to rigid rotation/translation; no SMITH code is called.
    """

    numbers = tuple(int(value) for value in atomic_numbers)
    effective = tuple(float(value) for value in effective_atomic_numbers)
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    policy = settings or LocalPerceptionSettings()
    if xyz.shape != (len(numbers), 3) or np.any(~np.isfinite(xyz)):
        raise ValueError("local perception coordinates must be finite (natoms, 3)")
    if len(effective) != len(numbers) or any(not math.isfinite(value) for value in effective):
        raise ValueError("effective atomic numbers must be finite and complete")
    bonds = {tuple(sorted((int(left), int(right)))) for left, right in bonded_pairs}
    if any(left < 0 or right >= len(numbers) or left == right for left, right in bonds):
        raise ValueError("local perception contains an invalid primary bond")
    adjacency = [set() for _ in numbers]
    for left, right in bonds:
        adjacency[left].add(right)
        adjacency[right].add(left)

    thresholds = (
        ("ZEFF_EQUIVALENCE", policy.zeff_tolerance, "DIMENSIONLESS"),
        ("RADIAL_EQUIVALENCE", policy.distance_tolerance_angstrom, "ANGSTROM"),
        ("TEMPLATE_RMS", policy.template_rms_threshold, "COSINE_RMS"),
        ("TEMPLATE_MARGIN", policy.template_min_margin, "COSINE_RMS"),
        ("ANGLE_CLASS", policy.angle_class_tolerance, "COSINE"),
    )
    output: list[LocalPerceptionDomain] = []
    for center, neighbor_set in enumerate(adjacency):
        neighbors = tuple(sorted(neighbor_set))
        if len(neighbors) < 2:
            continue
        classes = local_ligand_equivalence_classes(
            center,
            neighbors,
            effective_atomic_numbers=effective,
            coordinates_angstrom=xyz,
            zeff_tolerance=policy.zeff_tolerance,
            distance_tolerance_angstrom=policy.distance_tolerance_angstrom,
        )
        match = local_coordination_match(
            center,
            neighbors,
            coordinates_angstrom=xyz,
            max_rms_cosine_error=policy.template_rms_threshold,
            min_score_margin=policy.template_min_margin,
        )
        group, confidence = infer_local_pseudogroup(
            center,
            neighbors,
            classes,
            coordinates_angstrom=xyz,
            settings=policy,
            match=match,
        )
        output.append(
            LocalPerceptionDomain(
                domain_id=f"LC{center + 1:05d}",
                kind="ATOM_CENTER",
                center_atom=center + 1,
                members=tuple(atom + 1 for atom in neighbors),
                equivalence_classes=_center_equivalence_records(
                    center, classes, effective=effective, coordinates=xyz
                ),
                proposed_group=group,
                confidence=confidence,
                operation_count=_finite_local_group_order(group),
                template_decision=(
                    _template_decision(match) if len(neighbors) >= 4 else None
                ),
                thresholds=thresholds,
                provider=ORACLE_LOCAL_PERCEPTION_PROVIDER,
                provider_version=ORACLE_LOCAL_PERCEPTION_PROVIDER_VERSION,
                provenance=(
                    f"{ORACLE_LOCAL_PERCEPTION_PROVIDER}@"
                    f"{ORACLE_LOCAL_PERCEPTION_PROVIDER_VERSION}:ATOM_CENTER"
                ),
            )
        )

    for ring_index, raw_ring in enumerate(rings, start=1):
        ring = tuple(int(atom) for atom in raw_ring)
        if len(ring) < 3 or len(set(ring)) != len(ring):
            raise ValueError("local ring domain must contain at least three unique atoms")
        if min(ring) < 0 or max(ring) >= len(numbers):
            raise ValueError("local ring domain lies outside the molecule")
        group, confidence, operations = ring_local_pseudogroup(
            ring,
            effective_atomic_numbers=effective,
            coordinates_angstrom=xyz,
            settings=policy,
        )
        output.append(
            LocalPerceptionDomain(
                domain_id=f"LR{ring_index:05d}",
                kind="RING",
                center_atom=None,
                members=tuple(atom + 1 for atom in ring),
                equivalence_classes=_ring_equivalence_records(
                    ring, effective=effective, coordinates=xyz, settings=policy
                ),
                proposed_group=group,
                confidence=confidence,
                operation_count=operations,
                template_decision=None,
                thresholds=thresholds,
                provider=ORACLE_LOCAL_PERCEPTION_PROVIDER,
                provider_version=ORACLE_LOCAL_PERCEPTION_PROVIDER_VERSION,
                provenance=(
                    f"{ORACLE_LOCAL_PERCEPTION_PROVIDER}@"
                    f"{ORACLE_LOCAL_PERCEPTION_PROVIDER_VERSION}:RING"
                ),
            )
        )
    return tuple(output)


def local_perception_settings_dict(settings: LocalPerceptionSettings) -> dict[str, float]:
    """Expose the exact provider settings without introducing a second schema."""

    return {name: float(value) for name, value in asdict(settings).items()}


def _center_equivalence_records(
    center: int,
    classes: tuple[tuple[int, ...], ...],
    *,
    effective: tuple[float, ...],
    coordinates: np.ndarray,
) -> tuple[LocalEquivalenceClass, ...]:
    output = []
    for index, members in enumerate(classes, start=1):
        zeff = np.asarray([effective[atom] for atom in members], dtype=float)
        radial = np.asarray(
            [np.linalg.norm(coordinates[atom] - coordinates[center]) for atom in members],
            dtype=float,
        )
        output.append(
            LocalEquivalenceClass(
                class_id=f"C{index:03d}",
                members=tuple(atom + 1 for atom in members),
                centroid_effective_atomic_number=float(np.mean(zeff)),
                centroid_distance_angstrom=float(np.mean(radial)),
                maximum_zeff_spread=float(np.ptp(zeff)),
                maximum_distance_spread_angstrom=float(np.ptp(radial)),
            )
        )
    return tuple(output)


def _ring_equivalence_records(
    ring: tuple[int, ...],
    *,
    effective: tuple[float, ...],
    coordinates: np.ndarray,
    settings: LocalPerceptionSettings,
) -> tuple[LocalEquivalenceClass, ...]:
    groups: list[list[int]] = []
    keys: list[tuple[float, float]] = []
    size = len(ring)
    edge_length = {
        atom: float(np.linalg.norm(coordinates[atom] - coordinates[ring[(index + 1) % size]]))
        for index, atom in enumerate(ring)
    }
    for atom in ring:
        key = (effective[atom], edge_length[atom])
        selected = next(
            (
                index
                for index, other in enumerate(keys)
                if abs(key[0] - other[0]) <= settings.zeff_tolerance
                and abs(key[1] - other[1]) <= settings.distance_tolerance_angstrom
            ),
            None,
        )
        if selected is None:
            keys.append(key)
            groups.append([atom])
        else:
            groups[selected].append(atom)
    records = []
    for index, members in enumerate(groups, start=1):
        zeff = np.asarray([effective[atom] for atom in members], dtype=float)
        radial = np.asarray([edge_length[atom] for atom in members], dtype=float)
        records.append(
            LocalEquivalenceClass(
                class_id=f"R{index:03d}",
                members=tuple(sorted(atom + 1 for atom in members)),
                centroid_effective_atomic_number=float(np.mean(zeff)),
                centroid_distance_angstrom=float(np.mean(radial)),
                maximum_zeff_spread=float(np.ptp(zeff)),
                maximum_distance_spread_angstrom=float(np.ptp(radial)),
            )
        )
    return tuple(records)


def _template_decision(match) -> LocalTemplateDecision:
    return LocalTemplateDecision(
        selected_template=(match.template.name if match.template is not None else None),
        best_template=(match.best_template.name if match.best_template is not None else None),
        competing_template=(
            match.competing_template.name if match.competing_template is not None else None
        ),
        score=_finite_or_none(match.score),
        competing_score=_finite_or_none(match.competing_score),
        margin=_finite_or_none(match.margin),
        status=match.status,
        rms_headroom=_finite_or_none(match.rms_headroom),
        margin_headroom=_finite_or_none(match.margin_headroom),
        threshold_sensitivity=match.sensitivity,
    )


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _finite_local_group_order(group: str) -> int:
    label = str(group).strip()
    fixed = {"C1": 1, "Cs": 2, "Ci": 2, "Td": 24, "Oh": 48, "Ih": 120}
    if label in fixed:
        return fixed[label]
    import re

    match = re.fullmatch(r"([CDS])(\d+)([vhd]?)", label, flags=re.IGNORECASE)
    if match is None:
        return 1
    family, order, suffix = match.groups()
    n = int(order)
    if family.upper() == "C":
        return n * (2 if suffix else 1)
    if family.upper() == "D":
        return 2 * n * (2 if suffix else 1)
    return 2 * n


__all__ = [
    "ORACLE_LOCAL_PERCEPTION_PROVIDER",
    "ORACLE_LOCAL_PERCEPTION_PROVIDER_VERSION",
    "local_perception_settings_dict",
    "perceive_local_perception_domains",
    "read_frozen_effective_atomic_numbers",
]
