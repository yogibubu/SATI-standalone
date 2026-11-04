from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from .bmatrix import SparseBRow
from .coordinate_registry import CoordinateSignature, function_signature


@dataclass(frozen=True)
class CoordinateFunctionDefinition:
    """Derived coordinate function evaluated from a frozen base coordinate set."""

    signature: CoordinateSignature
    operator: str
    arguments: tuple[CoordinateSignature, ...]
    parameters: tuple[tuple[str, str], ...] = ()
    coefficients: tuple[float, ...] = ()


@dataclass(frozen=True)
class CoordinateFunctionEvaluation:
    value: float
    b_row: SparseBRow


def inverse_distance(argument: CoordinateSignature) -> CoordinateFunctionDefinition:
    return _definition("INVERSE_DISTANCE", (argument,))


def exponential_distance(
    argument: CoordinateSignature,
    *,
    alpha: float,
) -> CoordinateFunctionDefinition:
    return _definition("EXPONENTIAL_DISTANCE", (argument,), parameters={"alpha": float(alpha)})


def sine(argument: CoordinateSignature) -> CoordinateFunctionDefinition:
    return _definition("SIN", (argument,))


def cosine(argument: CoordinateSignature) -> CoordinateFunctionDefinition:
    return _definition("COS", (argument,))


def difference(
    left: CoordinateSignature,
    right: CoordinateSignature,
) -> CoordinateFunctionDefinition:
    return linear_combination((left, right), coefficients=(1.0, -1.0), operator="DIFFERENCE")


def proton_transfer(
    donor_distance: CoordinateSignature,
    acceptor_distance: CoordinateSignature,
) -> CoordinateFunctionDefinition:
    """Return the standard donor-minus-acceptor proton-transfer observable."""

    return difference(donor_distance, acceptor_distance)


def sum_coordinates(
    *arguments: CoordinateSignature,
) -> CoordinateFunctionDefinition:
    if not arguments:
        raise ValueError("sum_coordinates needs at least one argument")
    return linear_combination(
        arguments,
        coefficients=tuple(1.0 for _argument in arguments),
        operator="SUM",
    )


def linear_combination(
    arguments: Iterable[CoordinateSignature],
    *,
    coefficients: Iterable[float],
    operator: str = "LINEAR_COMBINATION",
) -> CoordinateFunctionDefinition:
    args = tuple(arguments)
    coeffs = tuple(float(coefficient) for coefficient in coefficients)
    if len(args) != len(coeffs):
        raise ValueError("linear_combination needs one coefficient per argument")
    return _definition(
        operator,
        args,
        parameters={"coefficients": coeffs},
        coefficients=coeffs,
    )


def polar_radius(
    x_coordinate: CoordinateSignature,
    y_coordinate: CoordinateSignature,
) -> CoordinateFunctionDefinition:
    return _definition("POLAR_RADIUS", (x_coordinate, y_coordinate))


def elliptic_quadratic(
    x_coordinate: CoordinateSignature,
    y_coordinate: CoordinateSignature,
    *,
    a: float,
    b: float,
) -> CoordinateFunctionDefinition:
    return _definition(
        "ELLIPTIC_QUADRATIC",
        (x_coordinate, y_coordinate),
        parameters={"a": float(a), "b": float(b)},
    )


def elliptic_radius(
    x_coordinate: CoordinateSignature,
    y_coordinate: CoordinateSignature,
    *,
    a: float,
    b: float,
) -> CoordinateFunctionDefinition:
    return _definition(
        "ELLIPTIC_RADIUS",
        (x_coordinate, y_coordinate),
        parameters={"a": float(a), "b": float(b)},
    )


def evaluate_coordinate_function(
    definition: CoordinateFunctionDefinition,
    values: Mapping[CoordinateSignature, float],
    b_rows: Mapping[CoordinateSignature, SparseBRow],
) -> CoordinateFunctionEvaluation:
    args = definition.arguments
    arg_values = tuple(float(values[arg]) for arg in args)
    arg_rows = tuple(b_rows[arg] for arg in args)
    if not arg_rows:
        raise ValueError("coordinate function has no arguments")
    operator = definition.operator.upper()

    if operator == "INVERSE_DISTANCE":
        r = arg_values[0]
        if abs(r) <= 1.0e-14:
            raise FloatingPointError("inverse distance is singular")
        value = 1.0 / r
        derivatives = (-1.0 / (r * r),)
    elif operator == "EXPONENTIAL_DISTANCE":
        alpha = _float_parameter(definition, "alpha")
        value = float(np.exp(-alpha * arg_values[0]))
        derivatives = (-alpha * value,)
    elif operator == "SIN":
        value = float(np.sin(arg_values[0]))
        derivatives = (float(np.cos(arg_values[0])),)
    elif operator == "COS":
        value = float(np.cos(arg_values[0]))
        derivatives = (-float(np.sin(arg_values[0])),)
    elif operator in {"LINEAR_COMBINATION", "DIFFERENCE", "SUM"}:
        coefficients = definition.coefficients or tuple(1.0 for _arg in args)
        value = float(sum(coefficient * arg for coefficient, arg in zip(coefficients, arg_values)))
        derivatives = coefficients
    elif operator == "POLAR_RADIUS":
        x_value, y_value = arg_values
        value = float(np.hypot(x_value, y_value))
        if value <= 1.0e-14:
            raise FloatingPointError("polar radius is singular at the origin")
        derivatives = (x_value / value, y_value / value)
    elif operator == "ELLIPTIC_QUADRATIC":
        a = _float_parameter(definition, "a")
        b = _float_parameter(definition, "b")
        if abs(a) <= 1.0e-14 or abs(b) <= 1.0e-14:
            raise FloatingPointError("elliptic quadratic semi-axis is singular")
        x_value, y_value = arg_values
        value = float((x_value / a) ** 2 + (y_value / b) ** 2)
        derivatives = (2.0 * x_value / (a * a), 2.0 * y_value / (b * b))
    elif operator == "ELLIPTIC_RADIUS":
        a = _float_parameter(definition, "a")
        b = _float_parameter(definition, "b")
        if abs(a) <= 1.0e-14 or abs(b) <= 1.0e-14:
            raise FloatingPointError("elliptic radius semi-axis is singular")
        x_value, y_value = arg_values
        quadratic = float((x_value / a) ** 2 + (y_value / b) ** 2)
        value = float(np.sqrt(quadratic))
        if value <= 1.0e-14:
            raise FloatingPointError("elliptic radius is singular at the origin")
        derivatives = (x_value / (a * a * value), y_value / (b * b * value))
    else:
        raise ValueError(f"unsupported coordinate function operator: {definition.operator}")

    return CoordinateFunctionEvaluation(
        value=value,
        b_row=SparseBRow.combine(*tuple(zip(derivatives, arg_rows))),
    )


def evaluate_coordinate_functions(
    definitions: Iterable[CoordinateFunctionDefinition],
    values: Mapping[CoordinateSignature, float],
    b_rows: Mapping[CoordinateSignature, SparseBRow],
) -> tuple[
    dict[CoordinateSignature, float],
    dict[CoordinateSignature, SparseBRow],
    dict[CoordinateSignature, CoordinateFunctionEvaluation],
]:
    """Evaluate a deterministic stack of derived coordinate functions."""

    all_values = dict(values)
    all_rows = dict(b_rows)
    evaluations: dict[CoordinateSignature, CoordinateFunctionEvaluation] = {}
    for definition in definitions:
        evaluation = evaluate_coordinate_function(definition, all_values, all_rows)
        all_values[definition.signature] = evaluation.value
        all_rows[definition.signature] = evaluation.b_row
        evaluations[definition.signature] = evaluation
    return all_values, all_rows, evaluations


def _definition(
    operator: str,
    arguments: tuple[CoordinateSignature, ...],
    *,
    parameters: Mapping[str, object] = (),
    coefficients: tuple[float, ...] = (),
) -> CoordinateFunctionDefinition:
    signature = function_signature(operator, arguments, parameters=parameters)
    return CoordinateFunctionDefinition(
        signature=signature,
        operator=operator.upper(),
        arguments=arguments,
        parameters=signature.parameters,
        coefficients=coefficients,
    )


def _float_parameter(definition: CoordinateFunctionDefinition, name: str) -> float:
    values = dict(definition.parameters)
    try:
        return float(values[name])
    except KeyError as exc:
        raise ValueError(f"missing coordinate-function parameter {name!r}") from exc
