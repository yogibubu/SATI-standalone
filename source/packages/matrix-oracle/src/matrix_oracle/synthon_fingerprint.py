"""Continuous, topology-light fingerprints derived from ORACLE synthons."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


ORACLE_SYNTHON_FINGERPRINT_SCHEMA = "matrix.oracle.synthon_fingerprint.v1"
_DEFAULT_WIDTHS = {
    "z_eff": 0.08,
    "charge": 0.08,
    "covalency": 0.04,
    "delocalization": 0.04,
    "strain": 0.05,
    "pi_index": 0.10,
    "pi_pi_index": 0.10,
}


@dataclass(frozen=True)
class SynthonFingerprint:
    """Sparse, auditable feature vector with linear soft binning."""

    features: tuple[tuple[str, float], ...]
    atom_count: int
    schema: str = ORACLE_SYNTHON_FINGERPRINT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "atom_count": self.atom_count,
            "features": dict(self.features),
        }

    def as_mapping(self) -> dict[str, float]:
        return dict(self.features)


def build_synthon_fingerprint(
    atoms: Sequence[Mapping[str, Any]],
    *,
    bonds: Sequence[
        Mapping[str, Any] | tuple[int, int] | tuple[int, int, float]
    ] = (),
    widths: Mapping[str, float] | None = None,
) -> SynthonFingerprint:
    """Build an atom-and-bond fingerprint that varies continuously.

    Every continuous descriptor contributes to its two adjacent bins with
    linear weights. Small geometric or electronic changes therefore produce a
    small fingerprint change instead of a discontinuous bit flip.
    """

    policy = dict(_DEFAULT_WIDTHS)
    if widths:
        unknown = sorted(set(widths) - set(policy))
        if unknown:
            raise ValueError(f"unknown synthon fingerprint widths: {unknown}")
        policy.update({name: float(value) for name, value in widths.items()})
    if any(value <= 0.0 for value in policy.values()):
        raise ValueError("synthon fingerprint widths must be positive")

    records: dict[int, dict[str, Any]] = {}
    features: defaultdict[str, float] = defaultdict(float)
    for raw in atoms:
        record = _validated_atom(raw)
        atom = record["atom"]
        if atom in records:
            raise ValueError(f"duplicate synthon atom identifier: {atom}")
        records[atom] = record
        element = record["element"]
        features[f"element:{element}"] += 1.0
        for descriptor, width in policy.items():
            for index, weight in _soft_bins(record[descriptor], width):
                features[f"atom:{element}:{descriptor}:{index}"] += weight

    for raw_bond in bonds:
        left, right, order = _coerce_bond(raw_bond)
        if left not in records or right not in records:
            raise ValueError(
                f"fingerprint bond {(left, right)} references an absent atom"
            )
        pair = "-".join(
            sorted((records[left]["element"], records[right]["element"]))
        )
        for index, weight in _soft_bins(order, 0.10):
            features[f"bond:{pair}:order:{index}"] += weight
        for descriptor in ("z_eff", "charge", "covalency", "delocalization"):
            mean = 0.5 * (
                records[left][descriptor] + records[right][descriptor]
            )
            for index, weight in _soft_bins(mean, policy[descriptor]):
                features[f"bond:{pair}:{descriptor}:{index}"] += weight

    return SynthonFingerprint(
        features=tuple(
            (key, float(value))
            for key, value in sorted(features.items())
            if abs(value) > 1.0e-15
        ),
        atom_count=len(records),
    )


def synthon_fingerprint_similarity(
    left: SynthonFingerprint,
    right: SynthonFingerprint,
    *,
    metric: str = "tanimoto",
) -> float:
    """Compare sparse fingerprints by generalized Tanimoto or cosine."""

    a = left.as_mapping()
    b = right.as_mapping()
    keys = set(a) | set(b)
    dot = sum(a.get(key, 0.0) * b.get(key, 0.0) for key in keys)
    aa = sum(value * value for value in a.values())
    bb = sum(value * value for value in b.values())
    normalized = metric.strip().lower()
    if normalized == "cosine":
        denominator = math.sqrt(aa * bb)
    elif normalized == "tanimoto":
        denominator = aa + bb - dot
    else:
        raise ValueError("synthon fingerprint metric must be tanimoto or cosine")
    if denominator <= 1.0e-30:
        return 1.0 if not a and not b else 0.0
    return max(0.0, min(1.0, dot / denominator))


def _validated_atom(raw: Mapping[str, Any]) -> dict[str, Any]:
    required = {"atom", "element", *_DEFAULT_WIDTHS}
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(
            f"synthon fingerprint atom is missing: {', '.join(missing)}"
        )
    record: dict[str, Any] = {
        "atom": int(raw["atom"]),
        "element": str(raw["element"]),
    }
    for descriptor in _DEFAULT_WIDTHS:
        value = float(raw[descriptor])
        if not math.isfinite(value):
            raise ValueError(f"non-finite synthon descriptor: {descriptor}")
        record[descriptor] = value
    return record


def _coerce_bond(
    raw: Mapping[str, Any] | tuple[int, int] | tuple[int, int, float],
) -> tuple[int, int, float]:
    if isinstance(raw, Mapping):
        left = raw.get("left", raw.get("atom1"))
        right = raw.get("right", raw.get("atom2"))
        order = raw.get("order", raw.get("bond_order", 1.0))
    else:
        if len(raw) not in {2, 3}:
            raise ValueError("fingerprint bonds need two atoms and optional order")
        left, right = raw[:2]
        order = 1.0 if len(raw) == 2 else raw[2]
    return int(left), int(right), float(order)


def _soft_bins(value: float, width: float) -> tuple[tuple[int, float], ...]:
    position = float(value) / float(width)
    lower = math.floor(position)
    fraction = position - lower
    if fraction <= 1.0e-15:
        return ((lower, 1.0),)
    return ((lower, 1.0 - fraction), (lower + 1, fraction))


__all__ = [
    "ORACLE_SYNTHON_FINGERPRINT_SCHEMA",
    "SynthonFingerprint",
    "build_synthon_fingerprint",
    "synthon_fingerprint_similarity",
]
