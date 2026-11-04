"""ARCHITECT-owned derivation of intrinsic CM5 charge baselines."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Sequence

import numpy as np


CM5_BASELINE_SCHEMA = "matrix.architect.cm5_baseline.v1"


@dataclass(frozen=True)
class Cm5Baseline:
    """Raw CM5, derived intrinsic baseline and reversible response."""

    raw_charges_e: tuple[float, ...]
    response_delta_e: tuple[float, ...]
    intrinsic_charges_e: tuple[float, ...]
    molecular_charge: float
    source: str
    schema: str = CM5_BASELINE_SCHEMA

    def __post_init__(self) -> None:
        raw = np.asarray(self.raw_charges_e, dtype=float)
        response = np.asarray(self.response_delta_e, dtype=float)
        intrinsic = np.asarray(self.intrinsic_charges_e, dtype=float)
        if raw.ndim != 1 or response.shape != raw.shape or intrinsic.shape != raw.shape:
            raise ValueError("CM5 baseline vectors must have identical one-dimensional shapes")
        if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(response)) or not np.all(np.isfinite(intrinsic)):
            raise ValueError("CM5 baseline vectors must be finite")
        if not np.allclose(intrinsic, raw - response, atol=1.0e-12, rtol=0.0):
            raise ValueError("intrinsic CM5 must equal raw CM5 minus the declared response")
        if abs(float(np.sum(response))) > 1.0e-10:
            raise ValueError("CM5 response must conserve molecular charge")
        if abs(float(np.sum(intrinsic)) - float(self.molecular_charge)) > 1.0e-8:
            raise ValueError("intrinsic CM5 charges do not conserve molecular charge")

    @property
    def raw_hash(self) -> str:
        payload = json.dumps(self.raw_charges_e, separators=(",", ":"))
        return sha256(payload.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "raw_cm5_charges_e": list(self.raw_charges_e),
            "response_delta_e": list(self.response_delta_e),
            "intrinsic_cm5_charges_e": list(self.intrinsic_charges_e),
            "molecular_charge": float(self.molecular_charge),
            "source": self.source,
            "raw_cm5_sha256": self.raw_hash,
            "derivation": "INTRINSIC = RAW_CM5 - DECLARED_RESPONSE",
        }


def derive_cm5_intrinsic_baseline(
    raw_cm5_charges_e: Sequence[float],
    *,
    molecular_charge: float,
    response_delta_e: Sequence[float] | None = None,
    source: str = "ARCHITECT_FIELD_BUILD",
) -> Cm5Baseline:
    """Derive a field-build baseline without mutating the library raw record."""

    raw = np.asarray(raw_cm5_charges_e, dtype=float).reshape(-1)
    response = np.zeros_like(raw) if response_delta_e is None else np.asarray(response_delta_e, dtype=float).reshape(-1)
    if raw.size == 0 or response.shape != raw.shape:
        raise ValueError("raw CM5 and response vectors must have the same nonzero length")
    intrinsic = raw - response
    return Cm5Baseline(
        raw_charges_e=tuple(float(value) for value in raw),
        response_delta_e=tuple(float(value) for value in response),
        intrinsic_charges_e=tuple(float(value) for value in intrinsic),
        molecular_charge=float(molecular_charge),
        source=str(source),
    )


__all__ = ["CM5_BASELINE_SCHEMA", "Cm5Baseline", "derive_cm5_intrinsic_baseline"]
