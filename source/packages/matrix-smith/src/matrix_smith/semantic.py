from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Iterable

from matrix_core import section_content

from .contracts import GICForgeContractError
from .coordinate_functions import (
    CoordinateFunctionDefinition,
    cosine,
    difference,
    exponential_distance,
    elliptic_radius,
    inverse_distance,
    linear_combination,
    polar_radius,
    proton_transfer,
    sine,
    sum_coordinates,
)
from .coordinate_registry import CoordinateSignature, function_signature, primitive_signature


SEMANTIC_GRAMMAR_VERSION = "sonic.semantic.v1"
LEGACY_SEMANTIC_GRAMMAR_VERSION = "smith.semantic.v1"
SUPPORTED_SEMANTIC_GRAMMAR_VERSIONS = frozenset(
    {SEMANTIC_GRAMMAR_VERSION, LEGACY_SEMANTIC_GRAMMAR_VERSION}
)
SEMANTIC_SECTION_NAMES = (
    "SONIC_SEMANTIC",
    "GSNIC_SEMANTIC",
    "SMITH_SEMANTIC",
    "SEMANTIC_COORDINATES",
)
USER_PROVENANCE = "USER"
AUTO_PROVENANCE = "AUTO"
DERIVED_PROVENANCE = "DERIVED"

PROTECT_LAYER = "PROTECT"
OBSERVABLE_LAYER = "OBSERVABLE"

_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_ASSIGNMENT_RE = re.compile(
    r"^(?P<identifier>[A-Za-z][A-Za-z0-9_]*)\s*=\s*"
    r"(?P<operator>[A-Za-z][A-Za-z0-9_]*)\s*\((?P<arguments>.*)\)\s*$"
)
_RESERVED_ID_PREFIXES = ("AUTO_", "SONIC_", "GSNIC_", "SMITH_", "GIC")

_PRIMITIVE_PROTECT_OPERATORS = {
    "DISTANCE": ("STRETCH", "R", 2),
    "ANGLE": ("BEND", "A", 3),
    "TORSION": ("TORSION", "D", 4),
    "OUT_OF_PLANE": ("OUT_OF_PLANE", "U", 4),
}

_SEMANTIC_PROTECT_OPERATORS = {
    "BUTTERFLY",
    "CENTER_ANGLE",
    "CENTER_DISTANCE",
    "CENTER_TORSION",
    "FRAGMENT_ROTATION",
    "FRAGMENT_TRANSLATION",
    "INTERACTION_DISTANCE",
    "RING_DEFORMATION",
    "RING_PUCKERING",
}

_OBSERVABLE_OPERATORS = {
    "CREMER_POPLE",
    "COS",
    "DIFFERENCE",
    "ELLIPTIC_RADIUS",
    "EXP_DISTANCE",
    "EXPONENTIAL_DISTANCE",
    "INVERSE_DISTANCE",
    "LINEAR_COMBINATION",
    "POLAR_RADIUS",
    "PROTON_TRANSFER",
    "SIN",
    "SUM",
}


@dataclass(frozen=True)
class SemanticCoordinate:
    """User-facing semantic request attached to a frozen coordinate contract."""

    identifier: str
    layer: str
    semantic_type: str
    arguments: tuple[str, ...]
    provenance: str
    input_order: int
    signature: CoordinateSignature
    generated_primitives: tuple[str, ...] = ()
    generated_gics: tuple[str, ...] = ()
    observable_function: CoordinateFunctionDefinition | None = None
    diagnostics: tuple[str, ...] = ()

    @property
    def dedup_key(self) -> str:
        return self.signature.text()


@dataclass(frozen=True)
class SemanticContract:
    grammar_version: str = SEMANTIC_GRAMMAR_VERSION
    coordinates: tuple[SemanticCoordinate, ...] = ()
    diagnostics: tuple[str, ...] = ()

    @property
    def protect_coordinates(self) -> tuple[SemanticCoordinate, ...]:
        return tuple(
            coordinate for coordinate in self.coordinates if coordinate.layer == PROTECT_LAYER
        )

    @property
    def observable_coordinates(self) -> tuple[SemanticCoordinate, ...]:
        return tuple(
            coordinate for coordinate in self.coordinates if coordinate.layer == OBSERVABLE_LAYER
        )

    def coordinate_by_id(self) -> dict[str, SemanticCoordinate]:
        return {coordinate.identifier: coordinate for coordinate in self.coordinates}

    def with_coordinate_generation(
        self,
        generated: dict[str, tuple[tuple[str, ...], tuple[str, ...]]],
    ) -> "SemanticContract":
        coordinates = []
        for coordinate in self.coordinates:
            primitives, gics = generated.get(coordinate.identifier, ((), ()))
            coordinates.append(
                replace(
                    coordinate,
                    generated_primitives=tuple(primitives),
                    generated_gics=tuple(gics),
                )
            )
        return replace(self, coordinates=tuple(coordinates))


def semantic_contract_from_sectioned_lines(lines: Iterable[str]) -> SemanticContract:
    """Read the optional semantic-coordinate section from an enriched XYZ container."""

    all_lines = list(lines)
    material: list[str] = []
    for section_name in SEMANTIC_SECTION_NAMES:
        material = section_content(all_lines, section_name)
        if material:
            break
    if not material:
        return SemanticContract()
    return parse_semantic_coordinate_lines(material)


def parse_semantic_coordinate_lines(lines: Iterable[str]) -> SemanticContract:
    grammar_version = SEMANTIC_GRAMMAR_VERSION
    layer: str | None = None
    coordinates: list[SemanticCoordinate] = []
    by_id: dict[str, SemanticCoordinate] = {}
    protect_signatures: dict[CoordinateSignature, str] = {}
    diagnostics: list[str] = []

    for raw_line in lines:
        text = _strip_semantic_comment(raw_line).strip()
        if not text or text.upper() == "NONE":
            continue
        upper = text.upper()
        if upper.startswith("SCHEMA "):
            grammar_version = text.split(maxsplit=1)[1].strip()
            if grammar_version not in SUPPORTED_SEMANTIC_GRAMMAR_VERSIONS:
                raise GICForgeContractError(
                    f"unsupported SONIC semantic grammar {grammar_version!r}; "
                    f"expected one of {sorted(SUPPORTED_SEMANTIC_GRAMMAR_VERSIONS)!r}"
                )
            continue
        if upper in {"PROTECT", "PROTECT {"} or upper == "PROTECT{":
            layer = PROTECT_LAYER
            continue
        if upper in {"OBSERVABLE", "OBSERVABLE {"} or upper == "OBSERVABLE{":
            layer = OBSERVABLE_LAYER
            continue
        if upper == "}":
            layer = None
            continue
        if upper.startswith("PROTECT "):
            layer, text = PROTECT_LAYER, text[len("PROTECT ") :].strip()
            if text == "{":
                continue
        elif upper.startswith("OBSERVABLE "):
            layer, text = OBSERVABLE_LAYER, text[len("OBSERVABLE ") :].strip()
            if text == "{":
                continue
        if layer not in {PROTECT_LAYER, OBSERVABLE_LAYER}:
            raise GICForgeContractError(
                f"semantic coordinate assignment outside PROTECT/OBSERVABLE block: {raw_line}"
            )

        coordinate = _parse_semantic_assignment(
            text,
            layer=layer,
            input_order=len(coordinates) + 1,
            by_id=by_id,
        )
        if coordinate.identifier in by_id:
            raise GICForgeContractError(
                f"duplicate SONIC semantic coordinate id: {coordinate.identifier}"
            )
        if coordinate.layer == PROTECT_LAYER:
            duplicate = protect_signatures.get(coordinate.signature)
            if duplicate is not None:
                raise GICForgeContractError(
                    "duplicate SONIC protected semantic coordinate: "
                    f"{coordinate.identifier} is row-equivalent to {duplicate}"
                )
            protect_signatures[coordinate.signature] = coordinate.identifier
        by_id[coordinate.identifier] = coordinate
        coordinates.append(coordinate)

    if coordinates:
        diagnostics.append(f"SEMANTIC_GRAMMAR={grammar_version}")
        diagnostics.append(f"USER_COORDINATE_COUNT={len(coordinates)}")
    return SemanticContract(
        grammar_version=grammar_version,
        coordinates=tuple(coordinates),
        diagnostics=tuple(diagnostics),
    )


def semantic_preview_lines(contract: SemanticContract) -> list[str]:
    lines = [
        f"SCHEMA {contract.grammar_version}",
        f"PROTECT_COUNT {len(contract.protect_coordinates)}",
        f"OBSERVABLE_COUNT {len(contract.observable_coordinates)}",
    ]
    if not contract.coordinates:
        lines.append("NONE")
        return lines
    for coordinate in contract.coordinates:
        lines.append(
            f"{coordinate.layer} {coordinate.identifier} "
            f"TYPE={coordinate.semantic_type} "
            f"PROVENANCE={coordinate.provenance} "
            f"SIGNATURE={coordinate.signature.text()}"
        )
    if contract.diagnostics:
        lines.append("DIAGNOSTICS " + ";".join(contract.diagnostics))
    return lines


def semantic_observable_function_definitions(
    contract: SemanticContract,
) -> tuple[CoordinateFunctionDefinition, ...]:
    by_id = contract.coordinate_by_id()
    definitions: list[CoordinateFunctionDefinition] = []
    for coordinate in contract.observable_coordinates:
        definitions.append(_observable_function_definition(coordinate, by_id))
    return tuple(definitions)


def primitive_semantic_signature(
    family: str,
    function: str,
    atoms: Iterable[int],
    *,
    mode: int = 0,
    ref_atoms: Iterable[int] = (),
) -> CoordinateSignature:
    del family
    return primitive_signature(function, atoms, mode=mode, ref_atoms=ref_atoms)


def semantic_signature_for_primitive_like(
    function: str,
    atoms: Iterable[int],
    *,
    mode: int = 0,
    ref_atoms: Iterable[int] = (),
) -> CoordinateSignature:
    return primitive_signature(function, atoms, mode=mode, ref_atoms=ref_atoms)


def _parse_semantic_assignment(
    text: str,
    *,
    layer: str,
    input_order: int,
    by_id: dict[str, SemanticCoordinate],
) -> SemanticCoordinate:
    match = _ASSIGNMENT_RE.match(text)
    if match is None:
        raise GICForgeContractError(f"invalid SONIC semantic coordinate line: {text}")
    identifier = match.group("identifier")
    _validate_user_identifier(identifier)
    semantic_type = match.group("operator").upper()
    raw_arguments = _split_argument_list(match.group("arguments"))
    if layer == PROTECT_LAYER:
        signature = _protect_signature(semantic_type, raw_arguments)
        arguments = tuple(argument.strip() for argument in raw_arguments)
        diagnostics = _protect_diagnostics(semantic_type)
        return SemanticCoordinate(
            identifier=identifier,
            layer=layer,
            semantic_type=semantic_type,
            arguments=arguments,
            provenance=USER_PROVENANCE,
            input_order=input_order,
            signature=signature,
            diagnostics=diagnostics,
        )
    if semantic_type not in _OBSERVABLE_OPERATORS:
        raise GICForgeContractError(f"unsupported SONIC OBSERVABLE operator: {semantic_type}")
    arguments, parameters = _observable_arguments_and_parameters(
        semantic_type,
        raw_arguments,
        by_id,
    )
    signature = function_signature(
        _observable_signature_operator(semantic_type),
        tuple(by_id[argument].signature for argument in arguments),
        parameters=parameters,
    )
    diagnostics = (
        ("PROTON_TRANSFER sugar for DIFFERENCE of two protected distances",)
        if semantic_type == "PROTON_TRANSFER"
        else ()
    )
    return SemanticCoordinate(
        identifier=identifier,
        layer=layer,
        semantic_type=semantic_type,
        arguments=arguments,
        provenance=DERIVED_PROVENANCE,
        input_order=input_order,
        signature=signature,
        diagnostics=diagnostics,
    )


def _protect_signature(operator: str, raw_arguments: tuple[str, ...]) -> CoordinateSignature:
    if operator in _PRIMITIVE_PROTECT_OPERATORS:
        _family, primitive_function, arity = _PRIMITIVE_PROTECT_OPERATORS[operator]
        atoms = _atom_arguments(operator, raw_arguments, arity=arity)
        return primitive_signature(primitive_function, atoms)
    if operator == "RING_PUCKERING":
        atoms = _variable_atom_arguments(operator, raw_arguments, minimum=4)
        return CoordinateSignature(
            kind="PROTECT",
            operator=operator,
            atoms=tuple(atoms),
        )
    if operator in _SEMANTIC_PROTECT_OPERATORS:
        return CoordinateSignature(
            kind="PROTECT",
            operator=operator,
            parameters=tuple((f"arg{idx}", argument) for idx, argument in enumerate(raw_arguments, 1)),
        )
    raise GICForgeContractError(f"unsupported SONIC PROTECT operator: {operator}")


def _protect_diagnostics(operator: str) -> tuple[str, ...]:
    if operator == "RING_PUCKERING":
        return (
            "RING_PUCKERING request follows the ring model selected by SONIC; "
            "rigid-bond contraction remains active when applicable",
        )
    if operator in _PRIMITIVE_PROTECT_OPERATORS:
        return ()
    return (
        f"{operator} semantic request is stored in the contract and preview layer; "
        "primitive generation is delegated to the corresponding SONIC family",
    )


def _observable_arguments_and_parameters(
    operator: str,
    raw_arguments: tuple[str, ...],
    by_id: dict[str, SemanticCoordinate],
) -> tuple[tuple[str, ...], tuple[tuple[str, object], ...]]:
    arguments: list[str] = []
    parameters: list[tuple[str, object]] = []

    if operator in {"INVERSE_DISTANCE", "SIN", "COS"}:
        _require_arity(operator, raw_arguments, 1)
        arguments = [_observable_id(raw_arguments[0], by_id)]
    elif operator in {"DIFFERENCE", "POLAR_RADIUS", "ELLIPTIC_RADIUS", "PROTON_TRANSFER"}:
        minimum = 2
        _require_minimum_arity(operator, raw_arguments, minimum)
        value_args, keyword_args = _split_keyword_arguments(raw_arguments)
        _require_arity(operator, value_args, 2)
        arguments = [_observable_id(argument, by_id) for argument in value_args]
        if operator == "PROTON_TRANSFER":
            _require_distance_arguments(operator, arguments, by_id)
        parameters.extend(keyword_args)
    elif operator in {"EXP_DISTANCE", "EXPONENTIAL_DISTANCE"}:
        _require_minimum_arity(operator, raw_arguments, 1)
        value_args, keyword_args = _split_keyword_arguments(raw_arguments)
        if len(value_args) > 2:
            raise GICForgeContractError(f"{operator} accepts one coordinate and optional alpha")
        arguments = [_observable_id(value_args[0], by_id)]
        if len(value_args) == 2:
            parameters.append(("alpha", _float_token(operator, value_args[1])))
        parameters.extend(keyword_args)
        if not any(key == "alpha" for key, _value in parameters):
            parameters.append(("alpha", 1.0))
    elif operator in {"SUM", "CREMER_POPLE"}:
        _require_minimum_arity(operator, raw_arguments, 1)
        arguments = [_observable_id(argument, by_id) for argument in raw_arguments]
    elif operator == "LINEAR_COMBINATION":
        _require_minimum_arity(operator, raw_arguments, 1)
        coefficients: list[float] = []
        for raw_argument in raw_arguments:
            identifier, coefficient = _linear_combination_item(operator, raw_argument)
            arguments.append(_observable_id(identifier, by_id))
            coefficients.append(coefficient)
        parameters.append(("coefficients", tuple(coefficients)))
    else:
        raise GICForgeContractError(f"unsupported SONIC OBSERVABLE operator: {operator}")
    return tuple(arguments), tuple(parameters)


def _observable_function_definition(
    coordinate: SemanticCoordinate,
    by_id: dict[str, SemanticCoordinate],
) -> CoordinateFunctionDefinition:
    args = tuple(by_id[identifier].signature for identifier in coordinate.arguments)
    operator = coordinate.semantic_type
    if operator == "INVERSE_DISTANCE":
        return inverse_distance(args[0])
    if operator in {"EXP_DISTANCE", "EXPONENTIAL_DISTANCE"}:
        alpha = float(dict(coordinate.signature.parameters).get("alpha", "1.0"))
        return exponential_distance(args[0], alpha=alpha)
    if operator == "SIN":
        return sine(args[0])
    if operator == "COS":
        return cosine(args[0])
    if operator in {"DIFFERENCE", "PROTON_TRANSFER"}:
        if operator == "PROTON_TRANSFER":
            return proton_transfer(args[0], args[1])
        return difference(args[0], args[1])
    if operator == "SUM":
        return sum_coordinates(*args)
    if operator == "LINEAR_COMBINATION":
        coefficients = _coefficients_from_signature(coordinate.signature)
        return linear_combination(args, coefficients=coefficients)
    if operator == "POLAR_RADIUS":
        return polar_radius(args[0], args[1])
    if operator == "ELLIPTIC_RADIUS":
        parameters = dict(coordinate.signature.parameters)
        return elliptic_radius(
            args[0],
            args[1],
            a=float(parameters.get("a", "1.0")),
            b=float(parameters.get("b", "1.0")),
        )
    return CoordinateFunctionDefinition(
        signature=coordinate.signature,
        operator=operator,
        arguments=args,
        parameters=coordinate.signature.parameters,
    )


def _observable_signature_operator(operator: str) -> str:
    if operator == "PROTON_TRANSFER":
        return "DIFFERENCE"
    if operator == "EXP_DISTANCE":
        return "EXPONENTIAL_DISTANCE"
    return operator


def _require_distance_arguments(
    operator: str,
    arguments: tuple[str, ...] | list[str],
    by_id: dict[str, SemanticCoordinate],
) -> None:
    for identifier in arguments:
        signature = by_id[identifier].signature
        if signature.kind.upper() != "PRIMITIVE" or signature.operator.upper() != "R":
            raise GICForgeContractError(
                f"{operator} expects two protected distance coordinate ids"
            )


def _validate_user_identifier(identifier: str) -> None:
    if _ID_RE.match(identifier) is None:
        raise GICForgeContractError(f"invalid SONIC semantic coordinate id: {identifier!r}")
    upper = identifier.upper()
    if any(upper.startswith(prefix) for prefix in _RESERVED_ID_PREFIXES) or re.fullmatch(
        r"P\d+", upper
    ):
        raise GICForgeContractError(
            f"reserved SONIC semantic coordinate id prefix in {identifier!r}"
        )


def _strip_semantic_comment(line: str) -> str:
    text = line.strip()
    for marker in ("//", "!"):
        if marker in text:
            text = text.split(marker, 1)[0]
    return text


def _split_argument_list(text: str) -> tuple[str, ...]:
    arguments: list[str] = []
    current: list[str] = []
    depth = 0
    for character in text:
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        if character == "," and depth == 0:
            argument = "".join(current).strip()
            if argument:
                arguments.append(argument)
            current = []
            continue
        current.append(character)
    argument = "".join(current).strip()
    if argument:
        arguments.append(argument)
    return tuple(arguments)


def _atom_arguments(operator: str, raw_arguments: tuple[str, ...], *, arity: int) -> tuple[int, ...]:
    _require_arity(operator, raw_arguments, arity)
    return tuple(_atom_token(operator, argument) for argument in raw_arguments)


def _variable_atom_arguments(
    operator: str,
    raw_arguments: tuple[str, ...],
    *,
    minimum: int,
) -> tuple[int, ...]:
    _require_minimum_arity(operator, raw_arguments, minimum)
    return tuple(_atom_token(operator, argument) for argument in raw_arguments)


def _atom_token(operator: str, token: str) -> int:
    try:
        value = int(token)
    except ValueError as exc:
        raise GICForgeContractError(
            f"{operator} PROTECT expects one-based atom indices; found {token!r}"
        ) from exc
    if value <= 0:
        raise GICForgeContractError(f"{operator} PROTECT atom index must be positive")
    return value


def _observable_id(token: str, by_id: dict[str, SemanticCoordinate]) -> str:
    text = token.strip()
    if re.fullmatch(r"[-+]?\d+", text):
        raise GICForgeContractError(
            "SONIC OBSERVABLE arguments must be semantic coordinate ids, not raw atom indices"
        )
    if text not in by_id:
        raise GICForgeContractError(f"unknown SONIC semantic coordinate id: {text}")
    return text


def _split_keyword_arguments(
    raw_arguments: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[tuple[str, object], ...]]:
    value_args: list[str] = []
    keyword_args: list[tuple[str, object]] = []
    for argument in raw_arguments:
        if "=" in argument:
            key, value = argument.split("=", 1)
            keyword_args.append((key.strip().lower(), _float_token(key.strip(), value.strip())))
        else:
            value_args.append(argument)
    return tuple(value_args), tuple(keyword_args)


def _linear_combination_item(operator: str, raw_argument: str) -> tuple[str, float]:
    text = raw_argument.strip()
    if "*" in text:
        coefficient_text, identifier = text.split("*", 1)
        return identifier.strip(), _float_token(operator, coefficient_text.strip())
    if ":" in text:
        identifier, coefficient_text = text.split(":", 1)
        return identifier.strip(), _float_token(operator, coefficient_text.strip())
    return text, 1.0


def _float_token(operator: str, token: str) -> float:
    try:
        return float(token)
    except ValueError as exc:
        raise GICForgeContractError(
            f"{operator} expects a numeric parameter; found {token!r}"
        ) from exc


def _require_arity(operator: str, arguments: tuple[str, ...], arity: int) -> None:
    if len(arguments) != arity:
        raise GICForgeContractError(f"{operator} expects {arity} arguments; found {len(arguments)}")


def _require_minimum_arity(operator: str, arguments: tuple[str, ...], minimum: int) -> None:
    if len(arguments) < minimum:
        raise GICForgeContractError(
            f"{operator} expects at least {minimum} arguments; found {len(arguments)}"
        )


def _coefficients_from_signature(signature: CoordinateSignature) -> tuple[float, ...]:
    parameters = dict(signature.parameters)
    text = parameters.get("coefficients", "")
    if text.startswith("[") and text.endswith("]"):
        return tuple(float(value) for value in text[1:-1].split(",") if value)
    return ()
