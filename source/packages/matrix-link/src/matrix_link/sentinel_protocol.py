from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "1.0"
REQUEST_SCHEMA = "matrix.link.sentinel.request.v1"
RESPONSE_SCHEMA = "matrix.link.sentinel.response.v1"
ACTIVE_VARIABLES_SCHEMA = "matrix.link.active_variables.v1"
CHECKPOINT_SCHEMA = "matrix.link.sentinel.checkpoint.v1"
ERROR_SCHEMA = "matrix.link.sentinel.error.v1"
_PROPERTIES = frozenset({"energy", "gradient", "hessian"})
_POINT_FORMS = (
    "variable_values",
    "variable_displacements",
    "sonic_values",
    "sonic_displacements",
)


class ProtocolValidationError(ValueError):
    """A LINK-SENTINEL v1 message violates the frozen wire contract."""


def contract_digest(coordinate_contract: Mapping[str, Any]) -> str:
    """Return the canonical SHA-256 identity of an immutable coordinate contract."""

    encoded = json.dumps(
        coordinate_contract, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_request(payload: object) -> dict[str, Any]:
    message = _object(payload, "request")
    _const(message, "schema", REQUEST_SCHEMA)
    _const(message, "sender", "LINK")
    _const(message, "receiver", "SENTINEL")
    _nonnegative_integer(message.get("cycle"), "cycle")
    _optional_envelope(message)
    contract = _object(message.get("coordinate_contract"), "coordinate_contract")
    exploration = _object(
        contract.get("pes_exploration"), "coordinate_contract.pes_exploration"
    )
    _const(exploration, "mode", "PES_EXPLORATION")
    retained_group = _identifier(exploration.get("retained_group"), "retained_group")
    pointwise_oracle = exploration.get("pointwise_oracle_symmetry")
    pointwise_projection = exploration.get("pointwise_cartesian_symmetrization")
    if not isinstance(pointwise_oracle, bool) or not isinstance(pointwise_projection, bool):
        raise ProtocolValidationError("pointwise PES policies must be Boolean")
    if pointwise_oracle is not pointwise_projection:
        raise ProtocolValidationError(
            "pointwise Cartesian projection requires pointwise ORACLE symmetry"
        )
    if pointwise_oracle is False and retained_group != "C1":
        raise ProtocolValidationError(
            "disabling pointwise symmetry is allowed only for independent C1 candidates"
        )
    if exploration.get("separate_exocyclic_torsions") is not True:
        raise ProtocolValidationError("PES exploration requires separate exocyclic torsions")
    active = _object(contract.get("active_variables"), "coordinate_contract.active_variables")
    _const(active, "schema", ACTIVE_VARIABLES_SCHEMA)
    variable_labels = _string_list(active.get("variable_labels"), "variable_labels", nonempty=True)
    variable_reference = _number_list(
        active.get("reference_values"), "active-variable reference_values"
    )
    if len(variable_labels) != len(variable_reference):
        raise ProtocolValidationError("active-variable labels/reference dimensions differ")
    sonic = _object(contract.get("sonic"), "coordinate_contract.sonic")
    sonic_labels = _string_list(sonic.get("labels"), "SONIC labels", nonempty=True)
    sonic_reference = _number_list(sonic.get("reference_values"), "SONIC reference_values")
    if len(sonic_labels) != len(sonic_reference):
        raise ProtocolValidationError("SONIC labels/reference dimensions differ")
    projection = active.get("sonic_from_variable_displacements")
    if not isinstance(projection, list) or len(projection) != len(sonic_labels):
        raise ProtocolValidationError("active-variable projection has the wrong row count")
    for row in projection:
        if len(_number_list(row, "active-variable projection row")) != len(variable_labels):
            raise ProtocolValidationError("active-variable projection has the wrong column count")
    digest = message.get("coordinate_contract_sha256")
    if digest is not None and digest != contract_digest(contract):
        raise ProtocolValidationError("coordinate_contract_sha256 does not match the contract")
    points = message.get("points")
    if not isinstance(points, list) or not points:
        raise ProtocolValidationError("points must be a non-empty array")
    seen: set[str] = set()
    for raw in points:
        point = _object(raw, "point")
        point_id = _identifier(point.get("point_id"), "point_id")
        if point_id in seen:
            raise ProtocolValidationError(f"duplicate point_id: {point_id}")
        seen.add(point_id)
        if point.get("evaluation_owner") not in {"link", "driver"}:
            raise ProtocolValidationError(f"point {point_id} has an invalid evaluation_owner")
        _properties(point.get("requested_properties"), point_id)
        active_point = _object(point.get("active_variables"), f"point {point_id} active_variables")
        sonic_point = _object(point.get("sonic"), f"point {point_id} SONIC")
        _coordinate_vector(active_point, variable_labels, f"point {point_id} active_variables")
        _coordinate_vector(sonic_point, sonic_labels, f"point {point_id} SONIC")
        cartesian = _object(point.get("cartesian"), f"point {point_id} Cartesian")
        atoms = _string_list(cartesian.get("atoms"), "Cartesian atoms", nonempty=True)
        coordinates = cartesian.get("coordinates_angstrom")
        if not isinstance(coordinates, list) or len(coordinates) != len(atoms):
            raise ProtocolValidationError(f"point {point_id} has the wrong atom/coordinate count")
        for xyz in coordinates:
            if len(_number_list(xyz, "Cartesian coordinate")) != 3:
                raise ProtocolValidationError("each Cartesian coordinate must have three values")
        point_symmetry = _object(
            point.get("point_symmetry"), f"point {point_id} point_symmetry"
        )
        _identifier(point_symmetry.get("point_group"), f"point {point_id} point_group")
        _nonnegative_integer(
            point_symmetry.get("operation_count"),
            f"point {point_id} symmetry operation_count",
        )
        _object(point.get("properties"), f"point {point_id} properties")
    _finite_tree(message)
    return message


def validate_response(
    payload: object, *, request: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    message = _object(payload, "response")
    _const(message, "schema", RESPONSE_SCHEMA)
    status = message.get("status")
    if status not in {"continue", "complete", "error"}:
        raise ProtocolValidationError(f"unsupported response status: {status!r}")
    _optional_envelope(message)
    if request is not None:
        for name in ("run_id", "transaction_id", "cycle"):
            if name in message and message[name] != request.get(name):
                raise ProtocolValidationError(f"response {name} does not match the request")
    evaluations = message.get("evaluations", [])
    if not isinstance(evaluations, list):
        raise ProtocolValidationError("evaluations must be an array")
    evaluation_ids: set[str] = set()
    for raw in evaluations:
        evaluation = _object(raw, "evaluation")
        point_id = _identifier(evaluation.get("point_id"), "evaluation point_id")
        if point_id in evaluation_ids:
            raise ProtocolValidationError(f"duplicate evaluation point_id: {point_id}")
        evaluation_ids.add(point_id)
    next_points = message.get("next_points", [])
    if not isinstance(next_points, list):
        raise ProtocolValidationError("next_points must be an array")
    next_ids: set[str] = set()
    request_active_labels: tuple[str, ...] = ()
    request_sonic_labels: tuple[str, ...] = ()
    allowed_calculators: set[str] = set()
    if request is not None:
        coordinate_contract = _object(
            request.get("coordinate_contract"), "request coordinate_contract"
        )
        active_contract = _object(
            coordinate_contract.get("active_variables"), "request active_variables"
        )
        sonic_contract = _object(coordinate_contract.get("sonic"), "request SONIC")
        request_active_labels = _string_list(
            active_contract.get("variable_labels"), "request variable_labels"
        )
        request_sonic_labels = _string_list(
            sonic_contract.get("labels"), "request SONIC labels"
        )
        capabilities = request.get("capabilities", {})
        if isinstance(capabilities, dict):
            calculators = capabilities.get("link_calculators", [])
            if isinstance(calculators, list):
                allowed_calculators = {str(item) for item in calculators}
    for raw in next_points:
        point = _object(raw, "next point")
        point_id = _identifier(point.get("point_id"), "next point_id")
        if point_id in next_ids:
            raise ProtocolValidationError(f"duplicate next point_id: {point_id}")
        next_ids.add(point_id)
        forms = [name for name in _POINT_FORMS if name in point]
        nested_forms = []
        for name in ("active_variables", "sonic"):
            nested = point.get(name)
            if isinstance(nested, dict):
                nested_forms.extend(
                    f"{name}.{form}"
                    for form in ("values", "displacements")
                    if form in nested
                )
        if len(forms) + len(nested_forms) != 1:
            raise ProtocolValidationError(
                f"point {point_id} must supply exactly one variable or SONIC vector"
            )
        if point.get("evaluation_owner", "link") not in {"link", "driver"}:
            raise ProtocolValidationError(f"point {point_id} has an invalid evaluation_owner")
        owner = str(point.get("evaluation_owner", "link"))
        calculator_id = str(
            point.get("calculator_id", "link-default" if owner == "link" else "sentinel")
        )
        if owner == "link" and allowed_calculators and calculator_id not in allowed_calculators:
            raise ProtocolValidationError(
                f"point {point_id} requests unadvertised calculator_id {calculator_id!r}"
            )
        if request is not None:
            _validate_next_point_vector(
                point,
                point_id=point_id,
                active_labels=request_active_labels,
                sonic_labels=request_sonic_labels,
            )
        _properties(point.get("requested_properties", ["energy"]), point_id)
    errors = message.get("errors", [])
    if not isinstance(errors, list):
        raise ProtocolValidationError("errors must be an array")
    if status == "error" and not errors:
        raise ProtocolValidationError("an error response must contain at least one error")
    if status == "continue" and not next_points:
        raise ProtocolValidationError("a continuing response must contain next_points")
    if status == "complete" and next_points:
        raise ProtocolValidationError("a complete response cannot contain next_points")
    _finite_tree(message)
    return message


def load_and_validate(path: Path | str, *, kind: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if kind == "request":
        return validate_request(payload)
    if kind == "response":
        return validate_response(payload)
    raise ValueError("kind must be 'request' or 'response'")


def _optional_envelope(message: Mapping[str, Any]) -> None:
    if "protocol_version" in message and message["protocol_version"] != PROTOCOL_VERSION:
        raise ProtocolValidationError("unsupported protocol_version")
    for name in ("run_id", "transaction_id"):
        if name in message:
            _identifier(message[name], name)


def _coordinate_vector(payload: Mapping[str, Any], labels: Sequence[str], context: str) -> None:
    if tuple(_string_list(payload.get("labels"), f"{context} labels")) != tuple(labels):
        raise ProtocolValidationError(f"{context} labels do not match the contract")
    for name in ("values", "displacements"):
        if len(_number_list(payload.get(name), f"{context} {name}")) != len(labels):
            raise ProtocolValidationError(f"{context} {name} has the wrong length")


def _properties(value: object, point_id: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ProtocolValidationError(f"point {point_id} needs requested_properties")
    normalized = tuple(str(item) for item in value)
    if len(set(normalized)) != len(normalized) or set(normalized) - _PROPERTIES:
        raise ProtocolValidationError(f"point {point_id} has invalid requested_properties")
    return normalized


def _validate_next_point_vector(
    point: Mapping[str, Any],
    *,
    point_id: str,
    active_labels: Sequence[str],
    sonic_labels: Sequence[str],
) -> None:
    if "labels" in point:
        supplied = _string_list(point["labels"], f"point {point_id} labels")
        if tuple(supplied) != tuple(active_labels):
            raise ProtocolValidationError(
                f"point {point_id} variable labels do not match the request"
            )
    for name in ("variable_values", "variable_displacements"):
        if name in point and len(_number_list(point[name], f"point {point_id} {name}")) != len(
            active_labels
        ):
            raise ProtocolValidationError(f"point {point_id} {name} has the wrong length")
    for name in ("sonic_values", "sonic_displacements"):
        if name in point and len(_number_list(point[name], f"point {point_id} {name}")) != len(
            sonic_labels
        ):
            raise ProtocolValidationError(f"point {point_id} {name} has the wrong length")
    for group_name, labels in (("active_variables", active_labels), ("sonic", sonic_labels)):
        group = point.get(group_name)
        if not isinstance(group, dict):
            continue
        if "labels" in group and tuple(
            _string_list(group["labels"], f"point {point_id} {group_name} labels")
        ) != tuple(labels):
            raise ProtocolValidationError(f"point {point_id} {group_name} labels do not match")
        for name in ("values", "displacements"):
            if name in group and len(
                _number_list(group[name], f"point {point_id} {group_name} {name}")
            ) != len(labels):
                raise ProtocolValidationError(
                    f"point {point_id} {group_name} {name} has the wrong length"
                )


def _object(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolValidationError(f"{context} must be a JSON object")
    return value


def _const(payload: Mapping[str, Any], name: str, expected: object) -> None:
    if payload.get(name) != expected:
        raise ProtocolValidationError(f"{name} must be {expected!r}")


def _identifier(value: object, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ProtocolValidationError(f"{name} must be a non-empty string")
    return text


def _string_list(value: object, name: str, *, nonempty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or (nonempty and not value):
        qualifier = " non-empty" if nonempty else ""
        raise ProtocolValidationError(f"{name} must be a{qualifier} array")
    if any(not isinstance(item, str) or not item for item in value):
        raise ProtocolValidationError(f"{name} must contain non-empty strings")
    result = tuple(value)
    return result


def _number_list(value: object, name: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ProtocolValidationError(f"{name} must be an array")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ProtocolValidationError(f"{name} must contain JSON numbers")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ProtocolValidationError(f"{name} contains a non-finite number")
    return result


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProtocolValidationError(f"{name} must be a non-negative integer")
    return value


def _finite_tree(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ProtocolValidationError("messages cannot contain NaN or infinity")
    if isinstance(value, Mapping):
        for child in value.values():
            _finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            _finite_tree(child)
