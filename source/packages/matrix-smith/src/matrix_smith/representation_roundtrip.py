"""Round-trip helpers for declared SONIC representations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .representation_contract import (
    PeriodicEmbeddingContract,
    periodic_constraint_residual,
    periodic_decode,
    periodic_embed,
)


@dataclass(frozen=True)
class PeriodicRoundTrip:
    identifiers: tuple[str, ...]
    scalar_values: tuple[float, ...]
    embedded_values: tuple[tuple[float, float], ...]
    decoded_values: tuple[float, ...]
    periodic_errors: tuple[float, ...]
    constraint_residuals: tuple[float, ...]
    passed: bool


def periodic_round_trip(
    values: tuple[float, ...] | list[float] | np.ndarray,
    contracts: tuple[PeriodicEmbeddingContract, ...] | list[PeriodicEmbeddingContract],
    *,
    tolerance: float = 1.0e-10,
) -> PeriodicRoundTrip:
    """Embed and decode periodic SONIC values with local branch tracking."""

    scalar = tuple(float(value) for value in values)
    definitions = tuple(contracts)
    if len(scalar) != len(definitions):
        raise ValueError("periodic values and contracts must have equal length")
    threshold = float(tolerance)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("round-trip tolerance must be positive and finite")
    embedded = tuple(
        tuple(float(value) for value in periodic_embed(value, contract))
        for value, contract in zip(scalar, definitions, strict=True)
    )
    decoded = tuple(
        periodic_decode(vector, contract, reference=value)
        for vector, contract, value in zip(embedded, definitions, scalar, strict=True)
    )
    errors = tuple(
        float(decoded_value - value)
        for decoded_value, value in zip(decoded, scalar, strict=True)
    )
    residuals = tuple(periodic_constraint_residual(vector) for vector in embedded)
    passed = all(abs(error) <= threshold for error in errors) and all(
        abs(residual) <= threshold for residual in residuals
    )
    return PeriodicRoundTrip(
        identifiers=tuple(contract.identifier for contract in definitions),
        scalar_values=scalar,
        embedded_values=embedded,
        decoded_values=decoded,
        periodic_errors=errors,
        constraint_residuals=residuals,
        passed=passed,
    )
