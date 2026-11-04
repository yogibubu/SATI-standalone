"""Auditable similarity-weighted transfer of geometric and ZAFF parameters."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, sqrt
from typing import Any, Mapping, Sequence

import numpy as np


ZAFF_PARAMETER_TRANSFER_SCHEMA = "matrix.zaff.parameter_transfer.v1"
ROBUST_TRANSFER_TUNING = 2.5
ROBUST_TRANSFER_ITERATIONS = 3


@dataclass(frozen=True)
class ZaffParameterDonor:
    molecule: str
    parameter_key: str
    parameter_kind: str
    value: float
    unit: str
    molecular_similarity: float
    local_synthon_distance: float
    source_level: str
    source_atoms: tuple[int, ...] = ()

    @property
    def weight(self) -> float:
        return max(0.0, float(self.molecular_similarity)) * exp(
            -0.5 * float(self.local_synthon_distance) ** 2
        )


@dataclass(frozen=True)
class ZaffTransferredParameter:
    parameter_key: str
    parameter_kind: str
    value: float
    uncertainty: float
    unit: str
    donor_count: int
    effective_donor_count: float
    donors: tuple[ZaffParameterDonor, ...]
    status: str
    robust_outlier_count: int = 0
    robust_iterations: int = 0
    schema: str = ZAFF_PARAMETER_TRANSFER_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "parameter_key": self.parameter_key,
            "parameter_kind": self.parameter_kind,
            "value": self.value,
            "uncertainty": self.uncertainty,
            "unit": self.unit,
            "donor_count": self.donor_count,
            "effective_donor_count": self.effective_donor_count,
            "status": self.status,
            "robust_outlier_count": self.robust_outlier_count,
            "robust_iterations": self.robust_iterations,
            "donors": [
                {
                    "molecule": donor.molecule,
                    "parameter_key": donor.parameter_key,
                    "parameter_kind": donor.parameter_kind,
                    "value": donor.value,
                    "unit": donor.unit,
                    "molecular_similarity": donor.molecular_similarity,
                    "local_synthon_distance": donor.local_synthon_distance,
                    "source_level": donor.source_level,
                    "source_atoms": list(donor.source_atoms),
                    "weight": donor.weight,
                }
                for donor in self.donors
            ],
        }


def transfer_zaff_parameter(
    donors: Sequence[ZaffParameterDonor],
    *,
    minimum_similarity: float = 0.10,
    minimum_donors: int = 2,
    robust_tuning: float = ROBUST_TRANSFER_TUNING,
    robust_iterations: int = ROBUST_TRANSFER_ITERATIONS,
) -> ZaffTransferredParameter:
    """Transfer one parameter and retain uncertainty instead of silently typing it."""

    if not isfinite(float(robust_tuning)) or float(robust_tuning) <= 0.0:
        raise ValueError("robust transfer tuning must be positive")
    if int(robust_iterations) < 0:
        raise ValueError("robust transfer iterations cannot be negative")
    # Repeated symmetry-equivalent rows must not acquire extra influence.
    unique: dict[tuple[Any, ...], ZaffParameterDonor] = {}
    for donor in donors:
        if donor.molecular_similarity < minimum_similarity or donor.weight <= 0.0:
            continue
        key = (
            donor.molecule,
            donor.parameter_key,
            donor.parameter_kind,
            donor.unit,
            tuple(donor.source_atoms),
            round(float(donor.value), 12),
        )
        unique.setdefault(key, donor)
    selected = tuple(unique.values())
    if not selected:
        raise ValueError("no supported donor reaches the transfer threshold")
    keys = {(item.parameter_key, item.parameter_kind, item.unit) for item in selected}
    if len(keys) != 1:
        raise ValueError("parameter donors do not describe one common typed parameter")
    if any(not isfinite(float(item.value)) for item in selected):
        raise ValueError("parameter donor values must be finite")
    weights = np.asarray([item.weight for item in selected], dtype=float)
    values = np.asarray([item.value for item in selected], dtype=float)
    weights /= float(np.sum(weights))
    weights, robust_outlier_count, robust_iteration_count = _robust_reweight(
        values, weights, tuning=float(robust_tuning), iterations=int(robust_iterations)
    )
    value = float(weights @ values)
    variance = float(weights @ (values - value) ** 2)
    effective = float(1.0 / np.sum(weights**2))
    key, kind, unit = next(iter(keys))
    return ZaffTransferredParameter(
        parameter_key=key,
        parameter_kind=kind,
        value=value,
        uncertainty=sqrt(max(0.0, variance)),
        unit=unit,
        donor_count=len(selected),
        effective_donor_count=effective,
        donors=selected,
        status=(
            "SUPPORTED_TRANSFER_REQUIRES_TARGET_HESSIAN_VALIDATION"
            if len(selected) >= minimum_donors
            else "SINGLE_DONOR_PROVISIONAL"
        ),
        robust_outlier_count=robust_outlier_count,
        robust_iterations=robust_iteration_count,
    )


def _robust_reweight(values, weights, *, tuning: float, iterations: int):
    """Bounded Huber reweighting after the existing molecular/local Gaussian kernel."""

    observations = np.asarray(values, dtype=float).reshape(-1, 1)
    current = np.asarray(weights, dtype=float).reshape(-1)
    if current.size < 3 or not np.all(np.isfinite(observations)):
        return current, 0, 0
    outliers = np.zeros(current.size, dtype=bool)
    completed = 0
    for step in range(iterations):
        center = float(np.median(observations[:, 0]))
        residual = np.abs(observations[:, 0] - center)
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


def leave_one_out_parameter_transfer(
    records: Sequence[Mapping[str, Any]],
    *,
    held_out_molecule: str,
    minimum_similarity: float = 0.10,
) -> Mapping[str, Any]:
    """Validate transfer without allowing the target molecule to donate to itself."""

    target_rows = [row for row in records if str(row["molecule"]) == held_out_molecule]
    donor_rows = [row for row in records if str(row["molecule"]) != held_out_molecule]
    results = []
    for target in target_rows:
        key = str(target["parameter_key"])
        kind = str(target["parameter_kind"])
        candidates = [
            ZaffParameterDonor(
                molecule=str(row["molecule"]),
                parameter_key=key,
                parameter_kind=kind,
                value=float(row["value"]),
                unit=str(row["unit"]),
                molecular_similarity=float(row["similarity_to_target"]),
                local_synthon_distance=float(row["local_synthon_distance"]),
                source_level=str(row.get("source_level", "UNKNOWN")),
                source_atoms=tuple(int(value) for value in row.get("source_atoms", ())),
            )
            for row in donor_rows
            if str(row["parameter_key"]) == key and str(row["parameter_kind"]) == kind
        ]
        try:
            transferred = transfer_zaff_parameter(
                candidates,
                minimum_similarity=minimum_similarity,
            )
        except ValueError as exc:
            results.append(
                {
                    "parameter_key": key,
                    "parameter_kind": kind,
                    "status": "UNSUPPORTED",
                    "reason": str(exc),
                }
            )
            continue
        error = transferred.value - float(target["value"])
        results.append(
            {
                **transferred.to_dict(),
                "held_out_value": float(target["value"]),
                "error": error,
                "absolute_error": abs(error),
            }
        )
    supported = [row for row in results if row["status"] != "UNSUPPORTED"]
    return {
        "schema": "matrix.zaff.parameter_transfer.leave_one_out.v1",
        "held_out_molecule": held_out_molecule,
        "self_donation_forbidden": True,
        "results": results,
        "supported_count": len(supported),
        "unsupported_count": len(results) - len(supported),
        "rms_error": (
            None
            if not supported
            else sqrt(float(np.mean([float(row["error"]) ** 2 for row in supported])))
        ),
    }
