"""Runtime-only view of a compiled ``matrix.zaff.anharmonic.v1`` record."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .ring_puckering import ZaffRingPuckeringSurface


ZAFF_ANHARMONIC_SCHEMA = "matrix.zaff.anharmonic.v1"


@dataclass(frozen=True)
class ZaffRuntimeCoordinate:
    index: int
    identifier: str
    name: str
    family: str
    reference_value: float

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ZaffRuntimeCoordinate":
        return cls(
            int(payload["index"]),
            str(payload["identifier"]),
            str(payload["name"]),
            str(payload["family"]),
            float(payload["reference_value"]),
        )


@dataclass(frozen=True)
class ZaffRuntimePeriodicCoordinate:
    identifier: str
    name: str
    coordinate_domain: str

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ZaffRuntimePeriodicCoordinate":
        return cls(
            str(payload["identifier"]),
            str(payload.get("name", payload["identifier"])),
            str(payload.get("coordinate_domain", "PERIODIC_2PI")),
        )


@dataclass(frozen=True)
class ZaffRuntimeDiagonalTerm:
    coordinate_index: int
    functional_form: str
    quadratic: float
    cubic: float
    quartic: float
    parameters: Mapping[str, Any]
    status: str = "COMPILED"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ZaffRuntimeDiagonalTerm":
        derivatives = dict(payload["derivatives"])
        return cls(
            int(payload["coordinate_index"]),
            str(payload["functional_form"]),
            float(derivatives["quadratic"]),
            float(derivatives["cubic"]),
            float(derivatives["quartic"]),
            dict(payload.get("parameters", {})),
            str(payload.get("status", "COMPILED")),
        )


@dataclass(frozen=True)
class ZaffRuntimeSonicModel:
    coordinates: tuple[ZaffRuntimeCoordinate, ...]
    periodic_coordinates: tuple[ZaffRuntimePeriodicCoordinate, ...]
    terms: tuple[ZaffRuntimeDiagonalTerm, ...]
    linear_gradient_internal: np.ndarray | None
    quadratic_matrix: np.ndarray | None
    semidiagonal_cubic_i_j_j: np.ndarray | None
    coupled_terms: tuple[Mapping[str, Any], ...]
    ring_puckering_surfaces: tuple[ZaffRingPuckeringSurface, ...]
    truncation_contract: Mapping[str, Any]
    redundant_field_contract: Mapping[str, Any]
    status: str
    schema: str = ZAFF_ANHARMONIC_SCHEMA

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ZaffRuntimeSonicModel":
        if payload.get("schema") != ZAFF_ANHARMONIC_SCHEMA:
            raise ValueError("unsupported ZAFF anharmonic schema")
        model = cls(
            coordinates=tuple(
                ZaffRuntimeCoordinate.from_dict(item)
                for item in payload.get("coordinates", ())
            ),
            periodic_coordinates=tuple(
                ZaffRuntimePeriodicCoordinate.from_dict(item)
                for item in payload.get("periodic_coordinates", ())
            ),
            terms=tuple(
                ZaffRuntimeDiagonalTerm.from_dict(item)
                for item in payload.get("terms", ())
            ),
            linear_gradient_internal=_optional_array(
                payload.get("linear_gradient_internal")
            ),
            quadratic_matrix=_optional_array(payload.get("quadratic_matrix")),
            semidiagonal_cubic_i_j_j=_optional_array(
                payload.get("semidiagonal_cubic_i_j_j")
            ),
            coupled_terms=tuple(
                dict(item) for item in payload.get("coupled_terms", ())
            ),
            ring_puckering_surfaces=tuple(
                ZaffRingPuckeringSurface.from_dict(item)
                for item in payload.get("ring_puckering_surfaces", ())
            ),
            truncation_contract=dict(payload.get("truncation_contract", {})),
            redundant_field_contract=dict(payload.get("redundant_field_contract", {})),
            status=str(payload.get("status", "PLANNED")),
        )
        if tuple(item.index for item in model.coordinates) != tuple(
            range(len(model.coordinates))
        ):
            raise ValueError("runtime SONIC coordinates are not contiguous")
        return model


def _optional_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    result = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(result)):
        raise ValueError("runtime SONIC derivative tensor contains non-finite values")
    return result


__all__ = [
    "ZAFF_ANHARMONIC_SCHEMA",
    "ZaffRuntimeCoordinate",
    "ZaffRuntimeDiagonalTerm",
    "ZaffRuntimePeriodicCoordinate",
    "ZaffRuntimeSonicModel",
]
