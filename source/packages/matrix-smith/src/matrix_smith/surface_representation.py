"""Representation contracts for ZAFF, EVB and residual surface providers."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

from .representation_request import RepresentationRequest


@dataclass(frozen=True)
class BasinTransitionContract:
    """Provider-neutral contract for continuous basin-to-basin handoff."""

    basin_ids: tuple[str, ...]
    coupling_model: str = "EXPLICIT_PROVIDER"
    switching_coordinate: str = "EMBEDDED_DISTANCE"
    width: float = 1.0

    def __post_init__(self) -> None:
        basins = tuple(str(item).strip() for item in self.basin_ids)
        if len(basins) < 2 or any(not item for item in basins):
            raise ValueError("a basin transition requires at least two non-empty basin ids")
        if float(self.width) <= 0.0:
            raise ValueError("basin transition width must be positive")
        object.__setattr__(self, "basin_ids", basins)


@dataclass(frozen=True)
class SurfaceRepresentationRequest:
    provider: str
    representation: RepresentationRequest
    scalar_potential: bool = True
    asymptotic_gate: str = "REQUIRED"
    provenance: str = "EXPLICIT"
    transition: BasinTransitionContract | None = None

    def __post_init__(self) -> None:
        provider = str(self.provider).strip().upper()
        if provider not in {"ZAFF", "EVB", "DELTAML"}:
            raise ValueError(f"unsupported surface provider: {self.provider}")
        if not isinstance(self.representation, RepresentationRequest):
            raise TypeError("surface representation must use RepresentationRequest")
        if provider in {"EVB", "DELTAML"} and self.representation.mode != "PERIODIC_EMBEDDING":
            raise ValueError(f"{provider} global surfaces require PERIODIC_EMBEDDING")
        if self.asymptotic_gate != "REQUIRED":
            raise ValueError("surface providers require an explicit asymptotic gate")
        object.__setattr__(self, "provider", provider)


def validate_basin_transition(
    request: SurfaceRepresentationRequest,
) -> BasinTransitionContract:
    """Require an explicit transition contract for multi-basin providers."""

    if request.provider not in {"EVB", "DELTAML"}:
        raise ValueError("only EVB and DeltaML surface requests have basin transitions")
    if request.transition is None:
        raise ValueError(f"{request.provider} requires an explicit basin transition contract")
    return request.transition


def evaluate_surface_request(
    request: SurfaceRepresentationRequest,
    evaluator: Callable[[Any, SurfaceRepresentationRequest], Any],
    values: Any,
) -> Any:
    """Dispatch a validated surface request without duplicating provider logic."""

    if not isinstance(request, SurfaceRepresentationRequest):
        raise TypeError("surface evaluation requires a SurfaceRepresentationRequest")
    if not callable(evaluator):
        raise TypeError("surface evaluator must be callable")
    return evaluator(values, request)
