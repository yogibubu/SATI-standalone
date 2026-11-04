"""LINK adapter for SMITH representation requests.

The adapter deliberately contains only representation bookkeeping.  Cartesian
realization remains in :class:`GeometryEvaluationService` and therefore uses
the established LINK back-transformers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from matrix_smith import (
    PeriodicEmbeddingContract,
    RepresentationRequest,
    periodic_decode,
)


@dataclass(frozen=True)
class RepresentationRealization:
    """Decoded scalar coordinates plus the requested representation metadata."""

    mode: str
    scalar_values: np.ndarray
    input_size: int
    unwrapped: bool


@dataclass(frozen=True)
class ResolvedRepresentation:
    request: RepresentationRequest
    fallback_used: bool = False
    fallback_reason: str = ""


def validate_link_representation(request: RepresentationRequest) -> RepresentationRequest:
    """Validate and return the canonical request consumed by LINK."""

    if not isinstance(request, RepresentationRequest):
        raise TypeError("LINK requires a matrix_smith.RepresentationRequest")
    return request


def resolve_link_representation(
    request: RepresentationRequest,
    *,
    supported_modes=frozenset({"SCALAR", "PERIODIC_EMBEDDING", "CARTESIAN"}),
    allow_scalar_fallback: bool = False,
) -> ResolvedRepresentation:
    """Resolve support explicitly; never silently downgrade a global request."""

    validated = validate_link_representation(request)
    modes = frozenset(str(mode).strip().upper() for mode in supported_modes)
    if validated.mode in modes:
        return ResolvedRepresentation(validated)
    if not allow_scalar_fallback:
        raise ValueError(
            f"LINK representation mode {validated.mode} is unavailable; "
            "enable scalar fallback explicitly if acceptable"
        )
    fallback = RepresentationRequest(mode="SCALAR", purpose="LOCAL_OPTIMIZATION")
    return ResolvedRepresentation(
        fallback,
        fallback_used=True,
        fallback_reason=f"requested {validated.mode} is unsupported by this LINK path",
    )


def decode_periodic_embedding(
    values: np.ndarray,
    contracts: tuple[PeriodicEmbeddingContract, ...],
    *,
    reference: np.ndarray | None = None,
) -> RepresentationRealization:
    """Decode a ``2*n`` unit-circle vector for the existing LINK realization.

    Decoding is explicit and contract-driven: no coordinate ordering or
    periodicity is guessed.  The unwrapped branch is selected near the supplied
    scalar reference, after which ``GeometryEvaluationService.coordinates_from_q``
    performs the canonical Cartesian realization.
    """

    vector = np.asarray(values, dtype=float).reshape(-1)
    if len(contracts) == 0:
        raise ValueError("periodic embedding requires at least one coordinate contract")
    if vector.shape != (2 * len(contracts),):
        raise ValueError("periodic embedding length must be twice the contract count")
    prior = None if reference is None else np.asarray(reference, dtype=float).reshape(-1)
    if prior is not None and prior.shape != (len(contracts),):
        raise ValueError("periodic decode reference must match the contract count")
    decoded = np.asarray(
        [
            periodic_decode(
                vector[2 * index : 2 * index + 2],
                contract,
                reference=None if prior is None else float(prior[index]),
            )
            for index, contract in enumerate(contracts)
        ],
        dtype=float,
    )
    return RepresentationRealization(
        mode="PERIODIC_EMBEDDING",
        scalar_values=decoded,
        input_size=int(vector.size),
        unwrapped=prior is not None,
    )
