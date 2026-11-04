"""Independent validation and active learning for fitted ZAFF fields."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


ZAFF_VALIDATION_SCHEMA = "matrix.architect.zaff_validation.v1"
ZAFF_ACTIVE_LEARNING_SCHEMA = "matrix.architect.zaff_active_learning.v1"


@dataclass(frozen=True)
class ZaffValidationPoint:
    point_id: str
    partition: str
    reference_energy_kj_mol: float
    predicted_energy_kj_mol: float
    reference_gradient: np.ndarray | None = None
    predicted_gradient: np.ndarray | None = None
    reference_hessian: np.ndarray | None = None
    predicted_hessian: np.ndarray | None = None
    observables: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    provenance: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ZaffValidationReport:
    training_count: int
    validation_count: int
    metrics: Mapping[str, float]
    residuals: tuple[Mapping[str, object], ...]
    transferable: bool
    schema: str = ZAFF_VALIDATION_SCHEMA


@dataclass(frozen=True)
class ZaffAcquisitionCandidate:
    point_id: str
    predictive_std_kj_mol: float
    descriptor_distance: float
    physical_risk: float = 0.0
    metadata: Mapping[str, object] = field(default_factory=dict)


def validate_zaff_field(
    points: list[ZaffValidationPoint] | tuple[ZaffValidationPoint, ...],
    *,
    maximum_validation_energy_rmse_kj_mol: float = 2.0,
    maximum_validation_gradient_rmse: float = 0.02,
) -> ZaffValidationReport:
    items = tuple(points)
    training = [item for item in items if item.partition == "train"]
    validation = [item for item in items if item.partition == "validation"]
    if not training or not validation:
        raise ValueError("ZAFF validation requires non-empty train and validation partitions")
    residuals = []
    for item in items:
        energy_error = item.predicted_energy_kj_mol - item.reference_energy_kj_mol
        gradient_rmse = _paired_rmse(item.predicted_gradient, item.reference_gradient)
        hessian_rmse = _paired_rmse(item.predicted_hessian, item.reference_hessian)
        residuals.append(
            {
                "point_id": item.point_id,
                "partition": item.partition,
                "energy_error_kj_mol": float(energy_error),
                "gradient_rmse": gradient_rmse,
                "hessian_rmse": hessian_rmse,
                "observable_errors": {
                    name: float(predicted - reference)
                    for name, (reference, predicted) in item.observables.items()
                },
                "provenance": dict(item.provenance),
            }
        )
    metrics = {}
    for partition in ("train", "validation"):
        selected = [item for item in residuals if item["partition"] == partition]
        metrics[f"{partition}_energy_rmse_kj_mol"] = _rmse(
            [float(item["energy_error_kj_mol"]) for item in selected]
        )
        gradients = [
            float(item["gradient_rmse"])
            for item in selected
            if item["gradient_rmse"] is not None
        ]
        if gradients:
            metrics[f"{partition}_gradient_rmse"] = _rmse(gradients)
    transferable = (
        metrics["validation_energy_rmse_kj_mol"]
        <= maximum_validation_energy_rmse_kj_mol
        and metrics.get("validation_gradient_rmse", 0.0)
        <= maximum_validation_gradient_rmse
    )
    return ZaffValidationReport(
        training_count=len(training),
        validation_count=len(validation),
        metrics=metrics,
        residuals=tuple(residuals),
        transferable=transferable,
    )


def rank_zaff_active_learning_candidates(
    candidates: list[ZaffAcquisitionCandidate]
    | tuple[ZaffAcquisitionCandidate, ...],
    *,
    maximum_points: int,
    uncertainty_weight: float = 1.0,
    extrapolation_weight: float = 1.0,
    physical_risk_weight: float = 1.0,
) -> tuple[Mapping[str, object], ...]:
    if maximum_points < 1:
        raise ValueError("maximum_points must be positive")
    ranked = []
    for item in candidates:
        score = (
            uncertainty_weight * item.predictive_std_kj_mol
            + extrapolation_weight * item.descriptor_distance
            + physical_risk_weight * item.physical_risk
        )
        ranked.append(
            {
                "schema": ZAFF_ACTIVE_LEARNING_SCHEMA,
                "point_id": item.point_id,
                "score": float(score),
                "predictive_std_kj_mol": float(item.predictive_std_kj_mol),
                "descriptor_distance": float(item.descriptor_distance),
                "physical_risk": float(item.physical_risk),
                "metadata": dict(item.metadata),
            }
        )
    ranked.sort(key=lambda item: (-float(item["score"]), str(item["point_id"])))
    return tuple(ranked[:maximum_points])


def _paired_rmse(predicted, reference) -> float | None:
    if predicted is None and reference is None:
        return None
    if predicted is None or reference is None:
        raise ValueError("reference and predicted derivatives must be provided together")
    first = np.asarray(predicted, dtype=float)
    second = np.asarray(reference, dtype=float)
    if first.shape != second.shape:
        raise ValueError("reference and predicted derivative shapes differ")
    return float(np.sqrt(np.mean((first - second) ** 2)))


def _rmse(values) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(array * array)))


__all__ = [
    "ZAFF_ACTIVE_LEARNING_SCHEMA",
    "ZAFF_VALIDATION_SCHEMA",
    "ZaffAcquisitionCandidate",
    "ZaffValidationPoint",
    "ZaffValidationReport",
    "rank_zaff_active_learning_candidates",
    "validate_zaff_field",
]
