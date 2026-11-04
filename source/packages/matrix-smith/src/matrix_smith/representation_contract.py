"""Shared scalar, periodic-embedding and rigid-pose contracts for SONIC."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


REPRESENTATION_CONTRACT_SCHEMA = "matrix.smith.representation_contract.v1"


@dataclass(frozen=True)
class ScalarChartContract:
    identifier: str
    family: str
    unit: str
    periodicity: float | None = None
    chart: str = "LOCAL_SCALAR"


@dataclass(frozen=True)
class PeriodicEmbeddingContract:
    identifier: str
    period: float
    phase: float = 0.0
    embedding: str = "COS_SIN"
    constraint: str = "UNIT_CIRCLE"

    def __post_init__(self) -> None:
        if not np.isfinite(self.period) or self.period <= 0.0:
            raise ValueError("periodic embedding period must be positive and finite")
        if not np.isfinite(self.phase):
            raise ValueError("periodic embedding phase must be finite")


@dataclass(frozen=True)
class QuaternionPoseContract:
    identifier: str
    axis: tuple[float, float, float]
    gauge: str = "REFERENCE_DOT_NONNEGATIVE"
    constraint: str = "UNIT_QUATERNION"

    def __post_init__(self) -> None:
        axis = np.asarray(self.axis, dtype=float)
        if axis.shape != (3,) or not np.all(np.isfinite(axis)):
            raise ValueError("quaternion pose axis must be a finite three-vector")
        norm = float(np.linalg.norm(axis))
        if norm <= 1.0e-14:
            raise ValueError("quaternion pose axis must be nonzero")


def periodic_embed(value: float, contract: PeriodicEmbeddingContract) -> np.ndarray:
    """Map a scalar periodic coordinate to a unit-circle embedding."""

    angle = 2.0 * np.pi * (float(value) - contract.phase) / contract.period
    result = np.asarray((np.cos(angle), np.sin(angle)), dtype=float)
    return result / np.linalg.norm(result)


def periodic_embedding_derivative(
    value: float,
    contract: PeriodicEmbeddingContract,
) -> np.ndarray:
    """Return the analytic two-component derivative with respect to value."""

    scale = 2.0 * np.pi / contract.period
    embedded = periodic_embed(value, contract)
    return scale * np.asarray((-embedded[1], embedded[0]), dtype=float)


def periodic_embedding_second_derivative(
    value: float,
    contract: PeriodicEmbeddingContract,
) -> np.ndarray:
    """Return the analytic second derivative of the periodic embedding."""

    scale = 2.0 * np.pi / contract.period
    return -(scale**2) * periodic_embed(value, contract)


def periodic_constraint_residual(embedded: tuple[float, float] | np.ndarray) -> float:
    """Return ``c²+s²-1`` for a periodic embedding."""

    vector = np.asarray(embedded, dtype=float)
    if vector.shape != (2,) or not np.all(np.isfinite(vector)):
        raise ValueError("periodic embedding must be a finite two-vector")
    return float(np.dot(vector, vector) - 1.0)


def periodic_decode(
    embedded: tuple[float, float] | np.ndarray,
    contract: PeriodicEmbeddingContract,
    *,
    reference: float | None = None,
    norm_tolerance: float = 1.0e-8,
) -> float:
    """Decode an embedding, optionally unwrapping it near a reference value."""

    vector = np.asarray(embedded, dtype=float)
    if vector.shape != (2,) or not np.all(np.isfinite(vector)):
        raise ValueError("periodic embedding must be a finite two-vector")
    norm = float(np.linalg.norm(vector))
    if abs(norm - 1.0) > float(norm_tolerance) or norm <= 1.0e-14:
        raise ValueError("periodic embedding violates the unit-circle constraint")
    vector = vector / norm
    angle = float(np.arctan2(vector[1], vector[0]))
    value = contract.phase + contract.period * angle / (2.0 * np.pi)
    if reference is not None:
        if not np.isfinite(reference):
            raise ValueError("periodic decode reference must be finite")
        value += contract.period * round((float(reference) - value) / contract.period)
    return float(value)


def quaternion_from_exponential(
    angle: float,
    contract: QuaternionPoseContract,
) -> np.ndarray:
    """Construct a unit quaternion from an exponential-map rotation."""

    axis = np.asarray(contract.axis, dtype=float)
    axis /= np.linalg.norm(axis)
    half = 0.5 * float(angle)
    quaternion = np.concatenate(([np.cos(half)], axis * np.sin(half)))
    return quaternion / np.linalg.norm(quaternion)


def quaternion_exponential_derivative(
    angle: float,
    contract: QuaternionPoseContract,
) -> np.ndarray:
    """Return the analytic quaternion derivative with respect to angle."""

    axis = np.asarray(contract.axis, dtype=float)
    axis /= np.linalg.norm(axis)
    half = 0.5 * float(angle)
    return np.concatenate(([-0.5 * np.sin(half)], 0.5 * axis * np.cos(half)))


def quaternion_exponential_second_derivative(
    angle: float,
    contract: QuaternionPoseContract,
) -> np.ndarray:
    """Return the analytic second quaternion derivative with respect to angle."""

    axis = np.asarray(contract.axis, dtype=float)
    axis /= np.linalg.norm(axis)
    half = 0.5 * float(angle)
    return np.concatenate(([-0.25 * np.cos(half)], -0.25 * axis * np.sin(half)))


def quaternion_constraint_residual(
    quaternion: tuple[float, float, float, float] | np.ndarray,
) -> float:
    """Return ``q·q-1`` for a quaternion pose."""

    vector = np.asarray(quaternion, dtype=float)
    if vector.shape != (4,) or not np.all(np.isfinite(vector)):
        raise ValueError("quaternion must be a finite four-vector")
    return float(np.dot(vector, vector) - 1.0)


def align_quaternion_gauge(
    quaternion: tuple[float, float, float, float] | np.ndarray,
    reference: tuple[float, float, float, float] | np.ndarray,
) -> np.ndarray:
    """Choose the sign of a quaternion continuously relative to a reference."""

    current = np.asarray(quaternion, dtype=float)
    prior = np.asarray(reference, dtype=float)
    if current.shape != (4,) or prior.shape != (4,):
        raise ValueError("quaternions must be four-vectors")
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(prior)):
        raise ValueError("quaternions must be finite")
    current_norm = float(np.linalg.norm(current))
    prior_norm = float(np.linalg.norm(prior))
    if current_norm <= 1.0e-14 or prior_norm <= 1.0e-14:
        raise ValueError("quaternions must be nonzero")
    current /= current_norm
    prior /= prior_norm
    if float(np.dot(current, prior)) < 0.0:
        current = -current
    return current
