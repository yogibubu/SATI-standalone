"""Portable declarative operator compiler for LOCK and WMS-Rot.

The manifest contains only data and a deliberately small expression grammar.
Backends provide the primitive matrices for each Hilbert-space factor.  The
compiler then produces the factorized model used by the dense, matrix-free,
and compiled LOCK executors.  The same JSON contract can be consumed by the
Pyodide WMS-Rot application without embedding Python expressions in input.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from .operator_engine import (
    CompiledOperatorModel,
    FactorizedOperatorTerm,
    OperatorSpace,
)


LOCK_OPERATOR_MANIFEST_SCHEMA = "matrix.lock.operator_manifest.v1"
LOCK_OPERATOR_AUDIT_SCHEMA = "matrix.lock.operator_manifest_audit.v1"


def rotational_primitive_registry(
    J: int, representation: str = "Ir", reduction: str = "S"
) -> dict[str, dict[str, np.ndarray]]:
    """Build the common rigid-rotor primitive registry for one ``J`` block."""

    from .native_rotor import angular_momentum_operators, centrifugal_distortion_operators

    Ja, Jb, Jc = angular_momentum_operators(J, representation)
    registry = {"Ja": Ja, "Jb": Jb, "Jc": Jc}
    names = (
        "watson_DJ", "watson_DJK", "watson_DK", "watson_d1", "watson_d2",
        "watson_HJ", "watson_HJK", "watson_HKJ", "watson_HK",
        "watson_h1", "watson_h2", "watson_h3",
        "watson_LJ", "watson_LJJK", "watson_LJK", "watson_LKKJ", "watson_LK",
        "watson_l1", "watson_l2", "watson_l3", "watson_l4",
    )
    registry.update(zip(names, centrifugal_distortion_operators(J, reduction, representation), strict=True))
    return {"rotation": registry}


def rigid_rotor_operator_manifest(
    A_MHz: float,
    B_MHz: float,
    C_MHz: float,
    *,
    profile: str = "portable",
    source: str = "",
) -> dict[str, object]:
    """Return the shared WMS-Rot/LOCK rigid-rotor Hamiltonian manifest."""

    parameters = []
    for axis, value in zip("ABC", (A_MHz, B_MHz, C_MHz), strict=True):
        parameter: dict[str, object] = {
            "name": axis,
            "value": _finite_real(value, field=axis),
            "unit": "MHz",
        }
        if source.strip():
            parameter["source"] = source.strip()
        parameters.append(parameter)
    terms = []
    for axis in "abc":
        terms.append({
            "label": f"J{axis}^2",
            "parameter": axis.upper(),
            "coefficient": 1.0,
            "hermitian": "already_hermitian",
            "symmetry": "totally-symmetric",
            "factors": [{
                "space": "rotation",
                "expression": {
                    "power": {"base": {"primitive": f"J{axis}"}, "exponent": 2}
                },
            }],
        })
    return normalize_operator_manifest({
        "schema": LOCK_OPERATOR_MANIFEST_SCHEMA,
        "name": "rigid-asymmetric-rotor",
        "description": "A Ja^2 + B Jb^2 + C Jc^2 shared by WMS-Rot and LOCK",
        "profile": profile,
        "spaces": [{"name": "rotation", "basis": "symmetric-top |J,K>"}],
        "parameters": parameters,
        "terms": terms,
    })


def watson_rotor_operator_manifest(
    rotational: object,
    *,
    profile: str | None = None,
) -> dict[str, object]:
    """Build a rigid plus available Watson-distortion manifest from one section."""

    from .native_rotor import native_rotor_parameters_from_section

    parameters = native_rotor_parameters_from_section(rotational)
    constant_source = str(getattr(rotational, "constant_source", "")).strip()
    distortion_source = str(getattr(rotational, "distortion_source", "")).strip()
    manifest = rigid_rotor_operator_manifest(
        parameters.A_MHz,
        parameters.B_MHz,
        parameters.C_MHz,
        profile="portable",
        source=constant_source,
    )
    raw_quartic = tuple(getattr(rotational, "quartic_distortion_MHz", ()))
    raw_sextic = tuple(getattr(rotational, "sextic_distortion_MHz", ()))
    raw_octic = tuple(getattr(rotational, "octic_distortion_MHz", ()))
    families = (
        (("DJ", "DJK", "DK", "d1", "d2"), raw_quartic, "quartic"),
        (("HJ", "HJK", "HKJ", "HK", "h1", "h2", "h3"), raw_sextic, "sextic"),
        (("LJ", "LJJK", "LJK", "LKKJ", "LK", "l1", "l2", "l3", "l4"), raw_octic, "octic"),
    )
    manifest_parameters = list(manifest["parameters"])
    terms = list(manifest["terms"])
    maximum_order = 2
    for names, values, family in families:
        if not values:
            continue
        if len(values) != len(names):
            raise ValueError(f"{family} Watson constants have an incompatible length")
        maximum_order = {"quartic": 4, "sextic": 6, "octic": 8}[family]
        for name, value in zip(names, values, strict=True):
            parameter: dict[str, object] = {
                "name": name,
                "value": _finite_real(value, field=name),
                "unit": "MHz",
            }
            if distortion_source:
                parameter["source"] = distortion_source
            manifest_parameters.append(parameter)
            terms.append({
                "label": f"Watson {family} {name}",
                "parameter": name,
                "coefficient": 1.0,
                "hermitian": "already_hermitian",
                "symmetry": "totally-symmetric",
                "factors": [{
                    "space": "rotation",
                    "expression": {"primitive": f"watson_{name}"},
                }],
            })
    selected_profile = profile or ("lock-full" if maximum_order >= 8 else "portable")
    return normalize_operator_manifest({
        **manifest,
        "name": f"watson-{parameters.reduction.lower()}-rotor-order-{maximum_order}",
        "description": (
            f"Rigid rotor plus Watson {parameters.reduction.upper()} terms through order "
            f"{maximum_order} in representation {parameters.representation}"
        ),
        "profile": selected_profile,
        "parameters": manifest_parameters,
        "terms": terms,
    })


def operator_manifest_from_xyzin(path: Path | str) -> dict[str, object]:
    """Derive the active rotational operator manifest from a MATRIX project."""

    from .contracts import read_rotational_section

    return watson_rotor_operator_manifest(read_rotational_section(path))


def _finite_real(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite real number")
    return result


def _complex_scalar(value: object, *, field: str) -> complex:
    if isinstance(value, Mapping):
        unknown = set(value) - {"real", "imag"}
        if unknown or "real" not in value:
            raise ValueError(f"{field} complex scalars require real and optional imag")
        result = complex(
            _finite_real(value["real"], field=f"{field}.real"),
            _finite_real(value.get("imag", 0.0), field=f"{field}.imag"),
        )
    else:
        result = complex(_finite_real(value, field=field))
    return result


def _canonical_json(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def normalize_operator_manifest(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate and normalize the representation-independent manifest data."""

    raw = dict(payload)
    if raw.get("schema") != LOCK_OPERATOR_MANIFEST_SCHEMA:
        raise ValueError(f"unsupported LOCK operator manifest schema: {raw.get('schema')!r}")
    allowed_top = {"schema", "name", "description", "profile", "spaces", "parameters", "terms"}
    unknown_top = set(raw) - allowed_top
    if unknown_top:
        raise ValueError(f"unknown operator manifest fields: {sorted(unknown_top)}")
    name = str(raw.get("name", "")).strip()
    if not name:
        raise ValueError("operator manifests require a name")
    profile = str(raw.get("profile", "lock-full")).strip()
    if profile not in {"portable", "wms-rot", "lock-full"}:
        raise ValueError("operator manifest profile must be portable, wms-rot, or lock-full")

    raw_spaces = raw.get("spaces")
    if not isinstance(raw_spaces, list) or not raw_spaces:
        raise ValueError("operator manifests require a nonempty spaces list")
    spaces: list[dict[str, str]] = []
    for index, item in enumerate(raw_spaces):
        if not isinstance(item, Mapping) or set(item) - {"name", "basis"}:
            raise ValueError(f"space {index} must contain only name and basis")
        space_name = str(item.get("name", "")).strip()
        basis = str(item.get("basis", "")).strip()
        if not space_name or not basis:
            raise ValueError(f"space {index} requires name and basis")
        spaces.append({"name": space_name, "basis": basis})
    space_names = [item["name"] for item in spaces]
    if len(space_names) != len(set(space_names)):
        raise ValueError("operator manifest space names must be unique")

    raw_parameters = raw.get("parameters", [])
    if not isinstance(raw_parameters, list):
        raise ValueError("operator manifest parameters must be a list")
    parameters: list[dict[str, object]] = []
    for index, item in enumerate(raw_parameters):
        if not isinstance(item, Mapping) or set(item) - {"name", "value", "unit", "source"}:
            raise ValueError(f"parameter {index} has unsupported fields")
        parameter_name = str(item.get("name", "")).strip()
        unit = str(item.get("unit", "")).strip()
        if not parameter_name or not unit:
            raise ValueError(f"parameter {index} requires name and unit")
        entry: dict[str, object] = {
            "name": parameter_name,
            "value": _finite_real(item.get("value"), field=f"parameter {parameter_name}"),
            "unit": unit,
        }
        if str(item.get("source", "")).strip():
            entry["source"] = str(item["source"]).strip()
        parameters.append(entry)
    parameter_names = [str(item["name"]) for item in parameters]
    if len(parameter_names) != len(set(parameter_names)):
        raise ValueError("operator manifest parameter names must be unique")

    raw_terms = raw.get("terms")
    if not isinstance(raw_terms, list) or not raw_terms:
        raise ValueError("operator manifests require a nonempty terms list")
    terms: list[dict[str, object]] = []
    used_parameters: set[str] = set()
    for index, item in enumerate(raw_terms):
        if not isinstance(item, Mapping):
            raise ValueError(f"operator term {index} must be an object")
        unknown = set(item) - {
            "label", "parameter", "coefficient", "hermitian", "symmetry", "factors"
        }
        if unknown:
            raise ValueError(f"operator term {index} has unsupported fields: {sorted(unknown)}")
        label = str(item.get("label", "")).strip()
        parameter = str(item.get("parameter", "")).strip()
        hermitian = str(item.get("hermitian", "half_sum")).strip()
        symmetry = str(item.get("symmetry", "totally-symmetric")).strip()
        if not label or hermitian not in {"half_sum", "already_hermitian"} or not symmetry:
            raise ValueError(f"operator term {index} has invalid metadata")
        if parameter and parameter not in parameter_names:
            raise ValueError(f"operator term {label!r} uses unknown parameter {parameter!r}")
        if parameter:
            used_parameters.add(parameter)
        coefficient = _complex_scalar(item.get("coefficient", 1.0), field=f"term {label}.coefficient")
        if hermitian == "half_sum" and coefficient.imag:
            raise ValueError(f"half-sum term {label!r} requires a real coefficient")
        raw_factors = item.get("factors")
        if not isinstance(raw_factors, list) or not raw_factors:
            raise ValueError(f"operator term {label!r} requires factors")
        factors: list[dict[str, object]] = []
        factor_spaces: set[str] = set()
        for factor in raw_factors:
            if not isinstance(factor, Mapping) or set(factor) != {"space", "expression"}:
                raise ValueError(f"term {label!r} factors require space and expression")
            factor_space = str(factor["space"]).strip()
            if factor_space not in space_names or factor_space in factor_spaces:
                raise ValueError(f"term {label!r} has an unknown or repeated factor space")
            factor_spaces.add(factor_space)
            _validate_expression(factor["expression"], context=f"term {label}/{factor_space}")
            factors.append({"space": factor_space, "expression": factor["expression"]})
        scalar: object = coefficient.real
        if coefficient.imag:
            scalar = {"real": coefficient.real, "imag": coefficient.imag}
        terms.append({
            "label": label,
            "parameter": parameter,
            "coefficient": scalar,
            "hermitian": hermitian,
            "symmetry": symmetry,
            "factors": factors,
        })
    unused = set(parameter_names) - used_parameters
    if unused:
        raise ValueError(f"operator manifest has unused parameters: {sorted(unused)}")
    normalized: dict[str, object] = {
        "schema": LOCK_OPERATOR_MANIFEST_SCHEMA,
        "name": name,
        "profile": profile,
        "spaces": spaces,
        "parameters": parameters,
        "terms": terms,
    }
    if str(raw.get("description", "")).strip():
        normalized["description"] = str(raw["description"]).strip()
    return normalized


def _validate_expression(expression: object, *, context: str) -> None:
    if not isinstance(expression, Mapping) or len(expression) != 1:
        raise ValueError(f"{context} expression must have exactly one operation")
    operation, argument = next(iter(expression.items()))
    if operation == "primitive":
        if not str(argument).strip():
            raise ValueError(f"{context} primitive name is empty")
        return
    if operation == "identity":
        if argument is not True:
            raise ValueError(f"{context} identity must be true")
        return
    if operation in {"adjoint", "negative"}:
        _validate_expression(argument, context=context)
        return
    if operation == "scale":
        if not isinstance(argument, Mapping) or set(argument) != {"coefficient", "expression"}:
            raise ValueError(f"{context} scale requires coefficient and expression")
        _complex_scalar(argument["coefficient"], field=f"{context}.scale")
        _validate_expression(argument["expression"], context=context)
        return
    if operation == "power":
        if not isinstance(argument, Mapping) or set(argument) != {"base", "exponent"}:
            raise ValueError(f"{context} power requires base and exponent")
        exponent = argument["exponent"]
        if isinstance(exponent, bool) or int(exponent) != exponent or int(exponent) < 0:
            raise ValueError(f"{context} power exponent must be a nonnegative integer")
        _validate_expression(argument["base"], context=context)
        return
    if operation in {"product", "sum", "symmetrized_product", "commutator"}:
        minimum = 2 if operation in {"symmetrized_product", "commutator"} else 1
        if not isinstance(argument, list) or len(argument) < minimum:
            raise ValueError(f"{context} {operation} requires at least {minimum} operands")
        if operation in {"symmetrized_product", "commutator"} and len(argument) != 2:
            raise ValueError(f"{context} {operation} requires exactly two operands")
        for child in argument:
            _validate_expression(child, context=context)
        return
    raise ValueError(f"{context} uses unsupported expression operation {operation!r}")


def _evaluate_expression(
    expression: Mapping[str, object], primitives: Mapping[str, np.ndarray], dimension: int
) -> np.ndarray:
    operation, argument = next(iter(expression.items()))
    if operation == "primitive":
        name = str(argument)
        if name not in primitives:
            raise ValueError(f"operator manifest requires unavailable primitive {name!r}")
        return primitives[name]
    if operation == "identity":
        return np.eye(dimension, dtype=complex)
    if operation == "adjoint":
        return _evaluate_expression(argument, primitives, dimension).conj().T
    if operation == "negative":
        return -_evaluate_expression(argument, primitives, dimension)
    if operation == "scale":
        return _complex_scalar(argument["coefficient"], field="expression scale") * _evaluate_expression(
            argument["expression"], primitives, dimension
        )
    if operation == "power":
        return np.linalg.matrix_power(
            _evaluate_expression(argument["base"], primitives, dimension),
            int(argument["exponent"]),
        )
    values = [_evaluate_expression(child, primitives, dimension) for child in argument]
    if operation == "product":
        result = np.eye(dimension, dtype=complex)
        for value in values:
            result = result @ value
        return result
    if operation == "sum":
        return np.add.reduce(values)
    if operation == "symmetrized_product":
        return 0.5 * (values[0] @ values[1] + values[1] @ values[0])
    if operation == "commutator":
        return values[0] @ values[1] - values[1] @ values[0]
    raise AssertionError(operation)


def compile_operator_manifest(
    payload: Mapping[str, object],
    primitive_registry: Mapping[str, Mapping[str, np.ndarray]],
) -> CompiledOperatorModel:
    """Compile a portable manifest against backend-supplied primitive matrices."""

    manifest = normalize_operator_manifest(payload)
    spaces: list[OperatorSpace] = []
    registries: dict[str, dict[str, np.ndarray]] = {}
    for entry in manifest["spaces"]:
        name = str(entry["name"])
        raw_registry = primitive_registry.get(name)
        if not raw_registry:
            raise ValueError(f"operator manifest space {name!r} has no primitive registry")
        registry: dict[str, np.ndarray] = {}
        dimension: int | None = None
        for primitive_name, raw_matrix in raw_registry.items():
            matrix = np.asarray(raw_matrix, dtype=complex)
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.all(np.isfinite(matrix)):
                raise ValueError(f"primitive {name}/{primitive_name} must be a finite square matrix")
            if dimension is not None and matrix.shape[0] != dimension:
                raise ValueError(f"primitive registry {name!r} mixes matrix dimensions")
            dimension = matrix.shape[0]
            registry[str(primitive_name)] = matrix
        assert dimension is not None
        spaces.append(OperatorSpace(name, dimension))
        registries[name] = registry

    terms: list[FactorizedOperatorTerm] = []
    for entry in manifest["terms"]:
        factors = []
        for factor in entry["factors"]:
            space = str(factor["space"])
            dimension = next(item.dimension for item in spaces if item.name == space)
            factors.append((space, _evaluate_expression(
                factor["expression"], registries[space], dimension
            )))
        terms.append(FactorizedOperatorTerm(
            label=str(entry["label"]),
            factors=tuple(factors),
            coefficient=_complex_scalar(entry["coefficient"], field="term coefficient"),
            parameter=str(entry["parameter"]),
            adjoint_symmetrized=entry["hermitian"] == "half_sum",
            symmetry=str(entry["symmetry"]),
        ))
    defaults = {str(item["name"]): float(item["value"]) for item in manifest["parameters"]}
    model = CompiledOperatorModel(tuple(spaces), tuple(terms), defaults)
    operator_manifest_audit(manifest, primitive_registry, compiled=model)
    return model


def operator_manifest_audit(
    payload: Mapping[str, object],
    primitive_registry: Mapping[str, Mapping[str, np.ndarray]],
    *,
    compiled: CompiledOperatorModel | None = None,
) -> dict[str, object]:
    """Return a reproducible structural and numerical audit of a manifest."""

    manifest = normalize_operator_manifest(payload)
    model = compiled or compile_operator_manifest(manifest, primitive_registry)
    term_audits = []
    maximum = 0.0
    for term in model.terms:
        matrices = dict(term.factors)
        value = np.ones((1, 1), dtype=complex)
        for space in model.spaces:
            value = np.kron(value, matrices.get(space.name, np.eye(space.dimension)))
        if term.adjoint_symmetrized:
            value = 0.5 * (value + value.conj().T)
        value *= term.coefficient
        defect = float(np.max(np.abs(value - value.conj().T), initial=0.0))
        scale = max(float(np.max(np.abs(value), initial=0.0)), 1.0)
        relative = defect / scale
        maximum = max(maximum, relative)
        term_audits.append({
            "label": term.label,
            "parameter": term.parameter or None,
            "symmetry": term.symmetry,
            "relative_hermiticity_defect": relative,
            "nonzero_elements": int(np.count_nonzero(np.abs(value) > 1.0e-14 * scale)),
        })
    if maximum > 2.0e-12:
        raise ValueError(f"operator manifest contains a non-Hermitian term: {maximum:.3e}")
    canonical = _canonical_json(manifest)
    return {
        "schema": LOCK_OPERATOR_AUDIT_SCHEMA,
        "manifest_schema": LOCK_OPERATOR_MANIFEST_SCHEMA,
        "manifest_name": manifest["name"],
        "manifest_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "profile": manifest["profile"],
        "dimension": model.dimension,
        "spaces": [{"name": item.name, "dimension": item.dimension} for item in model.spaces],
        "parameters": list(model.parameter_names),
        "term_count": len(model.terms),
        "maximum_relative_hermiticity_defect": maximum,
        "terms": term_audits,
        "passed": True,
    }


__all__ = [
    "LOCK_OPERATOR_AUDIT_SCHEMA",
    "LOCK_OPERATOR_MANIFEST_SCHEMA",
    "compile_operator_manifest",
    "normalize_operator_manifest",
    "operator_manifest_audit",
    "operator_manifest_from_xyzin",
    "rigid_rotor_operator_manifest",
    "rotational_primitive_registry",
    "watson_rotor_operator_manifest",
]
