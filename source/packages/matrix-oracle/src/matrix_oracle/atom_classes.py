"""Deterministic atom classes from ORACLE continuous synthon descriptors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


SYNTHON_ATOM_CLASS_SCHEMA = "matrix.oracle.synthon_atom_classes.v2"
_DESCRIPTORS = (
    "z_eff",
    "charge",
    "covalency",
    "delocalization",
    "strain",
    "pi_index",
    "pi_pi_index",
)


@dataclass(frozen=True)
class SynthonAtomClassThresholds:
    """Maximum within-class ranges for the continuous synthon descriptors."""

    z_eff: float = 0.08
    charge: float = 0.08
    covalency: float = 0.04
    delocalization: float = 0.04
    strain: float = 0.05
    pi_index: float = 0.10
    pi_pi_index: float = 0.10

    def __post_init__(self) -> None:
        for name in _DESCRIPTORS:
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} threshold must be positive")


@dataclass(frozen=True)
class SynthonAtomClass:
    """One element-preserving class with one-based atom identifiers."""

    identifier: str
    element: str
    atoms: tuple[int, ...]
    centroid: Mapping[str, float]
    maximum_spread: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "element": self.element,
            "atoms": list(self.atoms),
            "centroid": dict(self.centroid),
            "maximum_spread": dict(self.maximum_spread),
        }


@dataclass(frozen=True)
class SynthonAtomClassResult:
    """Complete, auditable partition returned by synthon classification."""

    thresholds: SynthonAtomClassThresholds
    classes: tuple[SynthonAtomClass, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SYNTHON_ATOM_CLASS_SCHEMA,
            "thresholds": asdict(self.thresholds),
            "classes": [item.to_dict() for item in self.classes],
        }


def classify_synthon_atoms(
    atoms: Sequence[Mapping[str, Any]],
    thresholds: SynthonAtomClassThresholds | None = None,
) -> SynthonAtomClassResult:
    """Partition atoms by element and bounded continuous-descriptor spread.

    Complete-link agglomeration prevents threshold chaining: every descriptor
    in every returned class has a range no larger than its declared threshold.
    Ties are resolved by one-based atom identifiers, so the partition is stable
    across Python versions and independent of the input record order.
    """

    policy = thresholds or SynthonAtomClassThresholds()
    records = [_validated_record(atom) for atom in atoms]
    identifiers = [int(atom["atom"]) for atom in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("synthon atom identifiers must be unique")
    records.sort(key=lambda atom: (str(atom["element"]), int(atom["atom"])))

    clusters: list[list[dict[str, Any]]] = [[atom] for atom in records]
    while True:
        candidates: list[tuple[float, tuple[int, ...], int, int]] = []
        for left in range(len(clusters)):
            for right in range(left + 1, len(clusters)):
                combined = clusters[left] + clusters[right]
                if combined[0]["element"] != combined[-1]["element"]:
                    continue
                score = _admissible_score(combined, policy)
                if score is None:
                    continue
                atom_ids = tuple(sorted(int(atom["atom"]) for atom in combined))
                candidates.append((score, atom_ids, left, right))
        if not candidates:
            break
        _, _, left, right = min(candidates)
        clusters[left] = sorted(
            clusters[left] + clusters[right], key=lambda atom: int(atom["atom"])
        )
        del clusters[right]

    clusters.sort(key=lambda group: min(int(atom["atom"]) for atom in group))
    element_counts: dict[str, int] = {}
    classes: list[SynthonAtomClass] = []
    for group in clusters:
        element = str(group[0]["element"])
        element_counts[element] = element_counts.get(element, 0) + 1
        values = {
            name: [float(atom[name]) for atom in group]
            for name in _DESCRIPTORS
        }
        classes.append(
            SynthonAtomClass(
                identifier=f"{element}{element_counts[element]}",
                element=element,
                atoms=tuple(sorted(int(atom["atom"]) for atom in group)),
                centroid={name: sum(series) / len(series) for name, series in values.items()},
                maximum_spread={name: max(series) - min(series) for name, series in values.items()},
            )
        )
    return SynthonAtomClassResult(policy, tuple(classes))


def _validated_record(atom: Mapping[str, Any]) -> dict[str, Any]:
    required = {"atom", "element", "z_eff", "charge", "covalency", "delocalization", "strain"}
    missing = sorted(required - set(atom))
    if missing:
        raise ValueError(f"synthon atom record is missing: {', '.join(missing)}")
    record = dict(atom)
    record["atom"] = int(record["atom"])
    record["element"] = str(record["element"])
    record.setdefault("pi_index", 0.0)
    record.setdefault("pi_pi_index", 0.0)
    for name in _DESCRIPTORS:
        record[name] = float(record[name])
    return record


def _admissible_score(
    atoms: Sequence[Mapping[str, Any]],
    thresholds: SynthonAtomClassThresholds,
) -> float | None:
    normalized_ranges = []
    for name in _DESCRIPTORS:
        values = [float(atom[name]) for atom in atoms]
        normalized = (max(values) - min(values)) / float(getattr(thresholds, name))
        if normalized > 1.0 + 1.0e-12:
            return None
        normalized_ranges.append(normalized)
    return max(normalized_ranges, default=0.0)
