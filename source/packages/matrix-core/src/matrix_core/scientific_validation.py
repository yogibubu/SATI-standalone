"""Uniform machine-readable records for numerical scientific gates."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping


SCIENTIFIC_VALIDATION_SCHEMA = "matrix.core.scientific_validation.v1"


@dataclass(frozen=True)
class ScientificValidationRecord:
    """One tool-owned pass/fail decision with metrics and declared limits."""

    owner: str
    subject: str
    passed: bool
    metrics: Mapping[str, float]
    thresholds: Mapping[str, float] = field(default_factory=dict)
    context: Mapping[str, object] = field(default_factory=dict)
    schema: str = SCIENTIFIC_VALIDATION_SCHEMA

    def __post_init__(self) -> None:
        owner = str(self.owner).strip().upper()
        subject = str(self.subject).strip()
        metrics = {str(key): float(value) for key, value in self.metrics.items()}
        thresholds = {
            str(key): float(value) for key, value in self.thresholds.items()
        }
        if self.schema != SCIENTIFIC_VALIDATION_SCHEMA:
            raise ValueError("unsupported scientific validation schema")
        if not owner or not subject or not metrics:
            raise ValueError("scientific validation owner, subject, and metrics are required")
        if any(not key or not math.isfinite(value) for key, value in metrics.items()):
            raise ValueError("scientific validation metrics must be named and finite")
        if any(
            not key or not math.isfinite(value) or value < 0.0
            for key, value in thresholds.items()
        ):
            raise ValueError("scientific validation thresholds must be named and nonnegative")
        object.__setattr__(self, "owner", owner)
        object.__setattr__(self, "subject", subject)
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "thresholds", thresholds)
        object.__setattr__(self, "context", dict(self.context))

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "owner": self.owner,
            "subject": self.subject,
            "status": self.status,
            "metrics": dict(sorted(self.metrics.items())),
            "thresholds": dict(sorted(self.thresholds.items())),
            "context": dict(sorted(self.context.items())),
        }


__all__ = ["SCIENTIFIC_VALIDATION_SCHEMA", "ScientificValidationRecord"]
