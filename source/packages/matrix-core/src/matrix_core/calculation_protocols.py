"""Versioned, owner-neutral calculation protocol atlas for Keymaker."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from importlib import resources
import json
from typing import Any, Mapping

from .operation_registry import operation_contract


CALCULATION_PROTOCOL_ATLAS_SCHEMA = "matrix.calculation_protocol_atlas.v1"
CALCULATION_PROTOCOL_ATLAS_ID = "nano-matrix-calculation-protocols-v29"
CALCULATION_PROTOCOL_ATLAS_VERSION = "29.0.0"
EXPECTED_CALCULATION_PROTOCOL_ATLAS_SHA256 = (
    "4244b2e289887c3bee60ce71462f9f267e8c6565c3f27feeb931a32dba33eb14"
)
CALCULATION_LIFECYCLE = (
    "request",
    "owner_prepare",
    "owner_preflight",
    "exact_launch_plan",
    "explicit_authorization",
    "execute",
    "monitor",
    "owner_validate",
    "promote",
    "provenance",
)


@dataclass(frozen=True)
class CalculationProtocol:
    """One auditable calculation class; scientific choices stay with its owner."""

    protocol_id: str
    aliases: tuple[str, ...]
    category: str
    owner_operation: str
    stationary_point: str
    topology_policy: str
    coordinate_policy: str
    backend_policy: str
    result_gate: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "CalculationProtocol":
        return cls(
            protocol_id=str(payload["id"]),
            aliases=tuple(str(item) for item in payload.get("aliases", ())),
            category=str(payload["category"]),
            owner_operation=str(payload["owner_operation"]),
            stationary_point=str(payload["stationary_point"]),
            topology_policy=str(payload["topology_policy"]),
            coordinate_policy=str(payload["coordinate_policy"]),
            backend_policy=str(payload["backend_policy"]),
            result_gate=str(payload["result_gate"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.protocol_id,
            "aliases": list(self.aliases),
            "category": self.category,
            "owner_operation": self.owner_operation,
            "stationary_point": self.stationary_point,
            "topology_policy": self.topology_policy,
            "coordinate_policy": self.coordinate_policy,
            "backend_policy": self.backend_policy,
            "result_gate": self.result_gate,
        }


@dataclass(frozen=True)
class KeymakerProtocolRoute:
    """Exact declarative mapping from one Keymaker action to one protocol."""

    menu: str
    action: str
    option_equals: Mapping[str, str]
    protocol_id: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "KeymakerProtocolRoute":
        raw_options = payload.get("option_equals", {})
        if not isinstance(raw_options, Mapping):
            raise RuntimeError("Keymaker protocol route options must be an object")
        return cls(
            menu=str(payload["menu"]),
            action=str(payload["action"]),
            option_equals={str(key): str(value) for key, value in raw_options.items()},
            protocol_id=str(payload["protocol"]),
        )


@dataclass(frozen=True)
class CalculationExecutionDirective:
    """Owner-declared backend directive with explicit applicability conditions."""

    directive_id: str
    owner_operation: str
    applies_to: tuple[str, ...]
    keyword: str
    allowed_conditions: tuple[str, ...]
    forbidden_keywords: tuple[str, ...]
    writer_policy: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "CalculationExecutionDirective":
        return cls(
            directive_id=str(payload["id"]),
            owner_operation=str(payload["owner_operation"]),
            applies_to=tuple(str(item) for item in payload["applies_to"]),
            keyword=str(payload["keyword"]),
            allowed_conditions=tuple(str(item) for item in payload["allowed_conditions"]),
            forbidden_keywords=tuple(str(item) for item in payload["forbidden_keywords"]),
            writer_policy=str(payload["writer_policy"]),
        )


@dataclass(frozen=True)
class CalculationProtocolAtlas:
    """Validated immutable view of the packaged protocol authority."""

    protocols: tuple[CalculationProtocol, ...]
    keymaker_routes: tuple[KeymakerProtocolRoute, ...]
    execution_directives: tuple[CalculationExecutionDirective, ...]
    sha256: str
    payload: Mapping[str, Any]

    def resolve(self, identifier: str) -> CalculationProtocol:
        normalized = str(identifier).strip().casefold().replace("-", "_")
        matches = tuple(
            protocol
            for protocol in self.protocols
            if normalized
            in {
                protocol.protocol_id.casefold().replace("-", "_"),
                *(alias.casefold().replace("-", "_") for alias in protocol.aliases),
            }
        )
        if len(matches) != 1:
            raise KeyError(f"unknown or ambiguous calculation protocol: {identifier}")
        return matches[0]

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(dict(self.payload))

    def resolve_keymaker_action(
        self,
        menu: str,
        action: str,
        *,
        options: Mapping[str, object] | None = None,
    ) -> CalculationProtocol:
        """Resolve only exact atlas routes; no action-name inference is allowed."""

        normalized_options = {
            str(key): _normalized_token(value) for key, value in dict(options or {}).items()
        }
        matches = tuple(
            route
            for route in self.keymaker_routes
            if route.menu == str(menu)
            and route.action == str(action)
            and all(
                normalized_options.get(key) == _normalized_token(value)
                for key, value in route.option_equals.items()
            )
        )
        if len(matches) != 1:
            raise KeyError(
                f"unknown or ambiguous Keymaker calculation route: {menu}/{action}"
            )
        return self.resolve(matches[0].protocol_id)

    def execution_directive(self, identifier: str) -> CalculationExecutionDirective:
        normalized = _normalized_token(identifier)
        matches = tuple(
            item
            for item in self.execution_directives
            if _normalized_token(item.directive_id) == normalized
        )
        if len(matches) != 1:
            raise KeyError(f"unknown or ambiguous calculation execution directive: {identifier}")
        return matches[0]


def load_calculation_protocol_atlas() -> CalculationProtocolAtlas:
    resource = resources.files("matrix_core").joinpath("data/calculation_protocol_atlas_v29.json")
    raw = resource.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_CALCULATION_PROTOCOL_ATLAS_SHA256:
        raise RuntimeError("installed calculation protocol atlas differs from its frozen fingerprint")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("calculation protocol atlas is unreadable") from exc
    validate_calculation_protocol_atlas(payload)
    return CalculationProtocolAtlas(
        protocols=tuple(CalculationProtocol.from_dict(item) for item in payload["protocols"]),
        keymaker_routes=tuple(
            KeymakerProtocolRoute.from_dict(item) for item in payload["keymaker_routes"]
        ),
        execution_directives=tuple(
            CalculationExecutionDirective.from_dict(item)
            for item in payload["execution_directives"]
        ),
        sha256=digest,
        payload=deepcopy(payload),
    )


def validate_calculation_protocol_atlas(payload: Mapping[str, Any]) -> None:
    """Fail closed on ownership, lifecycle, or gateway-policy drift."""

    if payload.get("schema") != CALCULATION_PROTOCOL_ATLAS_SCHEMA:
        raise RuntimeError("unsupported calculation protocol atlas schema")
    if payload.get("atlas_id") != CALCULATION_PROTOCOL_ATLAS_ID:
        raise RuntimeError("unexpected calculation protocol atlas")
    if payload.get("manifest_version") != CALCULATION_PROTOCOL_ATLAS_VERSION:
        raise RuntimeError("unsupported calculation protocol atlas version")
    if payload.get("status") != "approved":
        raise RuntimeError("calculation protocol atlas is not approved")
    if payload.get("entrypoint_policy") != (
        "keymaker_is_the_only_user_facing_calculation_launcher"
    ):
        raise RuntimeError("Keymaker calculation gateway policy was weakened")
    if payload.get("scientific_decision_policy") != "scientific_owner_only":
        raise RuntimeError("scientific decisions escaped their owner packages")
    if payload.get("change_policy") != ("new_manifest_version_and_explicit_approval"):
        raise RuntimeError("calculation protocol change policy was weakened")
    if tuple(payload.get("lifecycle", ())) != CALCULATION_LIFECYCLE:
        raise RuntimeError("calculation lifecycle differs from the approved protocol")
    records = payload.get("protocols")
    if not isinstance(records, list) or not records:
        raise RuntimeError("calculation protocol atlas has no records")
    required = {
        "id",
        "aliases",
        "category",
        "owner_operation",
        "stationary_point",
        "topology_policy",
        "coordinate_policy",
        "backend_policy",
        "result_gate",
    }
    identifiers: set[str] = set()
    aliases: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping) or set(record) != required:
            raise RuntimeError("calculation protocol record has incomplete fields")
        protocol = CalculationProtocol.from_dict(record)
        normalized_id = protocol.protocol_id.casefold().replace("-", "_")
        if not normalized_id or normalized_id in identifiers or normalized_id in aliases:
            raise RuntimeError("calculation protocol identifiers must be unique")
        identifiers.add(normalized_id)
        operation_contract(protocol.owner_operation)
        values = (
            protocol.category,
            protocol.stationary_point,
            protocol.topology_policy,
            protocol.coordinate_policy,
            protocol.backend_policy,
            protocol.result_gate,
        )
        if any(not value.strip() for value in values):
            raise RuntimeError(f"incomplete calculation protocol: {protocol.protocol_id}")
        for alias in protocol.aliases:
            normalized = alias.casefold().replace("-", "_")
            if not normalized or normalized in aliases or normalized in identifiers:
                raise RuntimeError("calculation protocol aliases must be unique")
            aliases.add(normalized)
    routes = payload.get("keymaker_routes")
    if not isinstance(routes, list) or not routes:
        raise RuntimeError("calculation protocol atlas has no Keymaker routes")
    route_keys: set[tuple[str, str, tuple[tuple[str, str], ...]]] = set()
    for record in routes:
        if not isinstance(record, Mapping) or set(record) != {
            "menu",
            "action",
            "option_equals",
            "protocol",
        }:
            raise RuntimeError("Keymaker protocol route has incomplete fields")
        route = KeymakerProtocolRoute.from_dict(record)
        if not route.menu or not route.action:
            raise RuntimeError("Keymaker protocol route is incomplete")
        normalized_protocol = route.protocol_id.casefold().replace("-", "_")
        if normalized_protocol not in identifiers:
            raise RuntimeError("Keymaker protocol route references an unknown protocol")
        key = (
            route.menu,
            route.action,
            tuple(sorted((name, _normalized_token(value)) for name, value in route.option_equals.items())),
        )
        if key in route_keys:
            raise RuntimeError("Keymaker protocol routes must be unique")
        route_keys.add(key)
    directives = payload.get("execution_directives")
    if not isinstance(directives, list) or not directives:
        raise RuntimeError("calculation protocol atlas has no execution directives")
    directive_ids: set[str] = set()
    directive_fields = {
        "id",
        "owner_operation",
        "applies_to",
        "keyword",
        "allowed_conditions",
        "forbidden_keywords",
        "writer_policy",
    }
    for record in directives:
        if not isinstance(record, Mapping) or set(record) != directive_fields:
            raise RuntimeError("calculation execution directive has incomplete fields")
        directive = CalculationExecutionDirective.from_dict(record)
        normalized = _normalized_token(directive.directive_id)
        if not normalized or normalized in directive_ids:
            raise RuntimeError("calculation execution directive identifiers must be unique")
        directive_ids.add(normalized)
        operation_contract(directive.owner_operation)
        if not directive.keyword or not directive.allowed_conditions or not directive.writer_policy:
            raise RuntimeError("calculation execution directive is incomplete")
        if any(_normalized_token(item) not in identifiers for item in directive.applies_to):
            raise RuntimeError("calculation execution directive references an unknown protocol")


def calculation_protocol(identifier: str) -> CalculationProtocol:
    return load_calculation_protocol_atlas().resolve(identifier)


def calculation_execution_directive(identifier: str) -> CalculationExecutionDirective:
    return load_calculation_protocol_atlas().execution_directive(identifier)


def _normalized_token(value: object) -> str:
    return str(value).strip().casefold().replace("-", "_")


__all__ = [
    "CALCULATION_LIFECYCLE",
    "CALCULATION_PROTOCOL_ATLAS_ID",
    "CALCULATION_PROTOCOL_ATLAS_SCHEMA",
    "CALCULATION_PROTOCOL_ATLAS_VERSION",
    "CalculationProtocol",
    "CalculationProtocolAtlas",
    "CalculationExecutionDirective",
    "KeymakerProtocolRoute",
    "calculation_protocol",
    "calculation_execution_directive",
    "load_calculation_protocol_atlas",
    "validate_calculation_protocol_atlas",
]
