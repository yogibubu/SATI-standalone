"""Shared, incrementable catalogue of mathematical coordinate capabilities.

The catalogue contains no molecular perception and no task-specific policy.
ORACLE atlases select capabilities from it; numerical tools implement those
capabilities without changing their scientific role.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
from importlib.resources import files
import json
from typing import Any, Mapping


COORDINATE_LIBRARY_SCHEMA = "matrix.coordinate_library.v2"
COMPONENT_SELECTION_SCALAR = "SCALAR"
COMPONENT_SELECTION_INDISSOLUBLE = "INDISSOLUBLE_COMPLETE_SET"
LINEAR_BEND_COMPONENT_MODES = (-1, -2)


@dataclass(frozen=True)
class CoordinateCapability:
    capability_id: str
    layer: str
    operator: str
    arity: str
    components: str
    value_domain: str
    periodic: bool
    analytic_derivatives: bool
    gaussian_native: bool
    implementation: str
    component_selection: str = COMPONENT_SELECTION_SCALAR


@lru_cache(maxsize=1)
def coordinate_capabilities() -> tuple[CoordinateCapability, ...]:
    """Return the deterministic shared catalogue."""

    payload = _library_payload("coordinate_library_v2.json", COORDINATE_LIBRARY_SCHEMA)
    records = payload.get("capabilities")
    if not isinstance(records, list):
        raise RuntimeError("coordinate library capabilities must be an array")
    capabilities = tuple(
        CoordinateCapability(**_typed_record(record, CoordinateCapability))
        for record in records
    )
    identifiers = tuple(item.capability_id for item in capabilities)
    operators = tuple((item.layer, item.operator) for item in capabilities)
    if len(identifiers) != len(set(identifiers)) or len(operators) != len(set(operators)):
        raise RuntimeError("coordinate capability identifiers and operators must be unique")
    return capabilities


def _library_payload(resource: str, schema: str) -> Mapping[str, Any]:
    payload = json.loads(
        files("matrix_chem").joinpath("data", resource).read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        raise RuntimeError(f"invalid declarative library schema: {resource}")
    return payload


def _typed_record(record: object, model: type) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise RuntimeError("declarative library entries must be objects")
    expected = set(model.__dataclass_fields__)
    if set(record) != expected:
        raise RuntimeError("declarative library entry fields do not match the typed model")
    return dict(record)


def coordinate_capability(operator: str, *, layer: str) -> CoordinateCapability:
    key = (str(layer).strip().upper(), str(operator).strip().upper())
    for item in coordinate_capabilities():
        if (item.layer, item.operator) == key:
            return item
    raise KeyError(f"unregistered coordinate capability: {key[0]}:{key[1]}")


@dataclass(frozen=True)
class CoordinateComponent:
    """Tool-neutral description of one component in a coordinate pool."""

    operator: str
    atoms: tuple[int, ...]
    mode: int = 0
    ref_atoms: tuple[int, ...] = ()
    context: tuple[str, ...] = ()


def coordinate_selection_units(
    components: tuple[CoordinateComponent, ...],
) -> tuple[tuple[int, ...], ...]:
    """Return indivisible selection units, rejecting incomplete component sets."""

    grouped: dict[tuple[object, ...], list[tuple[int, int]]] = {}
    scalar: list[tuple[int, tuple[int, ...]]] = []
    for index, component in enumerate(components):
        operator = component.operator.strip().upper()
        if operator != "L":
            scalar.append((index, (index,)))
            continue
        key = _linear_bend_group_key(component)
        grouped.setdefault(key, []).append((component.mode, index))

    units = list(scalar)
    for key, records in grouped.items():
        by_mode = {mode: index for mode, index in records}
        if (
            len(by_mode) != len(records)
            or set(by_mode) != set(LINEAR_BEND_COMPONENT_MODES)
        ):
            raise ValueError(
                "linear-bend components are an indivisible L(-1)/L(-2) pair: "
                f"incomplete or duplicate group {key}"
            )
        indices = tuple(by_mode[mode] for mode in LINEAR_BEND_COMPONENT_MODES)
        units.append((min(indices), indices))
    return tuple(unit for _first, unit in sorted(units, key=lambda item: item[0]))


def validate_coordinate_component_transform(
    components: tuple[CoordinateComponent, ...],
    transform: object,
    *,
    tolerance: float = 1.0e-12,
) -> None:
    """Reject a selected coordinate span containing only one component of an L pair."""

    import numpy as np

    matrix = np.asarray(transform, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != len(components):
        raise ValueError("coordinate component transform has an inconsistent shape")
    for unit in coordinate_selection_units(components):
        if len(unit) == 1:
            continue
        block = matrix[np.asarray(unit, dtype=int), :]
        if float(np.linalg.norm(block)) <= tolerance:
            continue
        if int(np.linalg.matrix_rank(block, tol=tolerance)) != len(unit):
            raise ValueError(
                "linear-bend components are indivisible: the selected span must "
                "contain both L(-1) and L(-2) components"
            )


def _linear_bend_group_key(component: CoordinateComponent) -> tuple[object, ...]:
    if len(component.atoms) != 3:
        raise ValueError("a linear bend requires exactly three atoms")
    first, center, third = component.atoms
    endpoints = tuple(sorted((int(first), int(third))))
    return (
        "L",
        int(center),
        endpoints,
        tuple(int(atom) for atom in component.ref_atoms),
        tuple(str(item) for item in component.context),
    )


def coordinate_library_manifest() -> dict[str, object]:
    payload = {
        "schema": COORDINATE_LIBRARY_SCHEMA,
        "capabilities": [asdict(item) for item in coordinate_capabilities()],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**payload, "sha256": hashlib.sha256(encoded).hexdigest()}


__all__ = [
    "COMPONENT_SELECTION_INDISSOLUBLE",
    "COMPONENT_SELECTION_SCALAR",
    "COORDINATE_LIBRARY_SCHEMA",
    "CoordinateComponent",
    "CoordinateCapability",
    "LINEAR_BEND_COMPONENT_MODES",
    "coordinate_capabilities",
    "coordinate_capability",
    "coordinate_library_manifest",
    "coordinate_selection_units",
    "validate_coordinate_component_transform",
]
