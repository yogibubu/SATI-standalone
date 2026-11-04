"""Local synthon tables for auditable ZAFF-fast/ZAFF0 transfer."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import sqrt
from typing import Any, Mapping, Sequence

import numpy as np


ZAFF_SYNTHON_TABLE_SCHEMA = "matrix.zaff.synthon_parameter_table.v1"
ROBUST_SYNTHON_TUNING = 2.5
ROBUST_SYNTHON_ITERATIONS = 3


@lru_cache(maxsize=512)
def normalized_zeff(zeff: float, atomic_number: int) -> float:
    """Map Zeff to the fixed ``[Z-0.5, Z+5]`` synthon scale."""

    lower = float(atomic_number) - 0.5
    return (float(zeff) - lower) / 5.5


@dataclass(frozen=True)
class ZaffSynthonTableRow:
    """One LCB25/LCB26 local parameter donor."""

    library: str
    molecule: str
    atom_type: str
    atomic_number: int
    cm5_charge: float
    zeff_normalized: float
    central_bond_order: float
    parameters: Mapping[str, float]
    source_level: str = "ZAFF-fast"

    def descriptor(self) -> np.ndarray:
        return np.asarray(
            [self.cm5_charge, self.zeff_normalized, self.central_bond_order],
            dtype=float,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ZAFF_SYNTHON_TABLE_SCHEMA,
            "library": self.library,
            "molecule": self.molecule,
            "atom_type": self.atom_type,
            "atomic_number": self.atomic_number,
            "cm5_charge_e": self.cm5_charge,
            "zeff_normalized": self.zeff_normalized,
            "central_bond_order": self.central_bond_order,
            "parameters": dict(self.parameters),
            "source_level": self.source_level,
        }


def interpolate_synthon_parameters(
    target: ZaffSynthonTableRow,
    donors: Sequence[ZaffSynthonTableRow],
    *,
    bandwidth: float = 1.0,
    minimum_donors: int = 2,
    library_mapping: Mapping[tuple[str, str], str] | None = None,
    robust_tuning: float = ROBUST_SYNTHON_TUNING,
    robust_iterations: int = ROBUST_SYNTHON_ITERATIONS,
) -> dict[str, Any]:
    """Transfer parameters from nearest compatible rows with audit metadata."""

    if bandwidth <= 0.0:
        raise ValueError("synthon-table bandwidth must be positive")
    if robust_tuning <= 0.0:
        raise ValueError("robust synthon tuning must be positive")
    if robust_iterations < 0:
        raise ValueError("robust synthon iterations cannot be negative")
    candidates = [
        row
        for row in donors
        if row.atomic_number == target.atomic_number
        and row.atom_type == target.atom_type
        and row.source_level == target.source_level
    ]
    if not candidates:
        raise ValueError("no compatible LCB25/LCB26 synthon-table donor")
    libraries = {row.library for row in candidates}
    if len(libraries) > 1:
        mapping_key = (target.library, target.atom_type)
        if library_mapping is None or mapping_key not in library_mapping:
            raise ValueError(
                "cross-library synthon donors require an explicit LCB25/LCB26 mapping"
            )
        mapped_type = str(library_mapping[mapping_key])
        if mapped_type != target.atom_type:
            raise ValueError("cross-library mapping does not preserve the atom type")
    descriptor_matrix = np.asarray([row.descriptor() for row in candidates], dtype=float)
    distances = np.linalg.norm(descriptor_matrix - target.descriptor(), axis=1)
    weights = np.exp(-0.5 * (distances / float(bandwidth)) ** 2)
    keep = weights > 0.0
    if not np.any(keep):
        raise ValueError("all synthon-table donors have zero weight")
    candidates = [row for row, ok in zip(candidates, keep) if ok]
    distances = distances[keep]
    weights = weights[keep]
    weights /= float(weights.sum())
    base_weights = weights.copy()
    robust_outlier_count = 0
    completed_iterations = 0
    keys = set(candidates[0].parameters)
    if any(set(row.parameters) != keys for row in candidates):
        raise ValueError("synthon-table donors do not share a parameter schema")
    values: dict[str, float] = {}
    parameter_weights: dict[str, np.ndarray] = {}
    for key in sorted(keys):
        parameter_values = np.asarray(
            [float(row.parameters[key]) for row in candidates], dtype=float
        )
        weights, outlier_count, iterations_done = _robust_reweight(
            parameter_values,
            base_weights,
            tuning=float(robust_tuning),
            iterations=int(robust_iterations),
        )
        robust_outlier_count = max(robust_outlier_count, outlier_count)
        completed_iterations = max(completed_iterations, iterations_done)
        parameter_weights[key] = weights
        values[key] = float(weights @ parameter_values)
    uncertainty = {
        key: sqrt(
            max(
                0.0,
                float(
                    sum(
                        weight * (float(row.parameters[key]) - values[key]) ** 2
                        for weight, row in zip(parameter_weights[key], candidates)
                    )
                ),
            )
        )
        for key in sorted(keys)
    }
    return {
        "schema": ZAFF_SYNTHON_TABLE_SCHEMA,
        "parameters": values,
        "uncertainty": uncertainty,
        "donor_count": len(candidates),
        "effective_donor_count": float(1.0 / np.sum(weights**2)),
        "status": "SUPPORTED" if len(candidates) >= minimum_donors else "SINGLE_DONOR_PROVISIONAL",
        "robust_outlier_count": robust_outlier_count,
        "robust_iterations": completed_iterations,
        "target": target.to_dict(),
        "donors": [
            {"molecule": row.molecule, "library": row.library, "distance": float(distance), "weight": float(weight)}
            for row, distance, weight in zip(candidates, distances, weights)
        ],
    }


def _robust_reweight(values, weights, *, tuning: float, iterations: int):
    """Suppress isolated parameter outliers after Gaussian descriptor weighting."""

    current = np.asarray(weights, dtype=float).reshape(-1)
    observations = np.asarray(values, dtype=float).reshape(-1)
    if current.size < 3 or not np.all(np.isfinite(observations)):
        return current, 0, 0
    outliers = np.zeros(current.size, dtype=bool)
    completed = 0
    for step in range(iterations):
        center = float(np.median(observations))
        residual = np.abs(observations - center)
        median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median)))
        scale = max(1.4826 * mad, median, 1.0e-12)
        standardized = residual / scale
        outliers = standardized > tuning
        if not np.any(outliers):
            break
        factors = np.ones_like(current)
        factors[outliers] = tuning / np.maximum(standardized[outliers], 1.0e-12)
        updated = current * factors
        updated /= max(float(np.sum(updated)), 1.0e-15)
        if np.allclose(updated, current, rtol=0.0, atol=1.0e-12):
            break
        current = updated
        completed = step + 1
    return current, int(np.count_nonzero(outliers)), completed


def rank_synthon_donors(
    target: ZaffSynthonTableRow,
    donors: Sequence[ZaffSynthonTableRow],
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Return nearest compatible donors for inspection before interpolation."""

    if limit < 1:
        raise ValueError("synthon donor limit must be positive")
    compatible = [
        row for row in donors
        if row.atomic_number == target.atomic_number
        and row.atom_type == target.atom_type
        and row.source_level == target.source_level
    ]
    ranked = sorted(
        ((float(np.linalg.norm(row.descriptor() - target.descriptor())), row) for row in compatible),
        key=lambda item: (item[0], item[1].library, item[1].molecule),
    )[:limit]
    return [
        {"distance": distance, "library": row.library, "molecule": row.molecule, "atom_type": row.atom_type}
        for distance, row in ranked
    ]


__all__ = [
    "ZAFF_SYNTHON_TABLE_SCHEMA",
    "ZaffSynthonTableRow",
    "interpolate_synthon_parameters",
    "rank_synthon_donors",
    "normalized_zeff",
]
