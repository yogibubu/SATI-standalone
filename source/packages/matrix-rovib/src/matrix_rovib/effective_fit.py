"""Weighted fits of contracted rovibronic and field-dressed LOCK Hamiltonians."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment

from .external_fields import (
    ExternalFieldModel,
    external_field_hamiltonian,
    read_external_field_model,
)
from .rovibronic import RovibronicModel, read_rovibronic_model, rovibronic_hamiltonian
from .spin_hyperfine import hellmann_feynman_diagonal


EFFECTIVE_FIT_SCHEMA = "matrix.lock.effective_fit.v1"
EFFECTIVE_FIT_RESULT_SCHEMA = "matrix.lock.effective_fit_result.v1"


@dataclass(frozen=True)
class _Variable:
    path: str
    initial: float
    lower: float
    upper: float
    scale: float


def fit_effective_hamiltonian(
    model: RovibronicModel | ExternalFieldModel,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit assigned transition frequencies with overlap tracking and HF derivatives."""
    if request.get("schema") != EFFECTIVE_FIT_SCHEMA:
        raise ValueError(f"unsupported effective-fit schema: {request.get('schema')!r}")
    observations = tuple(request.get("observations", ()))
    if not observations:
        raise ValueError("an effective-Hamiltonian fit requires observations")
    variables = _read_variables(model, request.get("variables", ()))
    if not variables:
        raise ValueError("an effective-Hamiltonian fit requires variables")
    fields = (
        float(request.get("electric_field_V_cm", 0.0)),
        float(request.get("magnetic_field_T", 0.0)),
    )
    reference_model = _apply_values(model, variables, [item.initial for item in variables])
    reference = _blocks(reference_model, observations, fields)
    cache: dict[tuple[float, ...], tuple[Any, np.ndarray, np.ndarray, list[dict[str, Any]]]] = {}

    def evaluate(values: Sequence[float]) -> tuple[Any, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        key = tuple(float(value) for value in values)
        if key in cache:
            return cache[key]
        fitted = _apply_values(model, variables, values)
        blocks = _tracked_blocks(fitted, observations, fields, reference)
        operators = _derivative_operators(fitted, variables, observations, fields)
        derivatives = {
            key: hellmann_feynman_diagonal(
                vectors, [parameter_ops[key] for parameter_ops in operators]
            )
            for key, (_energies, vectors) in blocks.items()
        }
        calculated, jacobian, rows = [], [], []
        for observation in observations:
            lower_key, lower_level = _selector(observation["lower"], fitted)
            upper_key, upper_level = _selector(observation["upper"], fitted)
            lower_e, _lower_v = blocks[lower_key]
            upper_e, _upper_v = blocks[upper_key]
            calc = float(upper_e[upper_level] - lower_e[lower_level])
            sigma = float(observation.get("uncertainty_MHz", 1.0))
            if sigma <= 0.0:
                raise ValueError("observation uncertainties must be positive")
            gradient = []
            for parameter_index in range(len(operators)):
                dlow = float(derivatives[lower_key][parameter_index, lower_level])
                dup = float(derivatives[upper_key][parameter_index, upper_level])
                gradient.append((dup - dlow) / sigma)
            observed = float(observation["frequency_MHz"])
            calculated.append((calc - observed) / sigma)
            jacobian.append(gradient)
            rows.append({
                "id": str(observation.get("id", len(rows))),
                "observed_MHz": observed,
                "calculated_MHz": calc,
                "residual_MHz": observed - calc,
                "uncertainty_MHz": sigma,
            })
        answer = fitted, np.asarray(calculated), np.asarray(jacobian), rows
        cache[key] = answer
        return answer

    x0 = np.asarray([item.initial for item in variables])
    lower = np.asarray([item.lower for item in variables])
    upper = np.asarray([item.upper for item in variables])
    scales = np.asarray([item.scale for item in variables])
    optimized = least_squares(
        lambda values: evaluate(values)[1],
        x0,
        jac=lambda values: evaluate(values)[2],
        bounds=(lower, upper),
        x_scale=scales,
        method="trf",
        loss=str(request.get("loss", "linear")),
        max_nfev=int(request.get("max_nfev", 400)),
    )
    fitted, weighted, jacobian, rows = evaluate(optimized.x)
    residuals = np.asarray([row["residual_MHz"] for row in rows])
    rank = int(np.linalg.matrix_rank(jacobian))
    dof = max(0, len(rows) - len(variables))
    chi_square = float(np.dot(weighted, weighted))
    reduced_chi_square = chi_square / dof if dof else None
    covariance = np.linalg.pinv(jacobian.T @ jacobian, rcond=1.0e-12)
    if reduced_chi_square is not None:
        covariance *= reduced_chi_square
    uncertainties = np.sqrt(np.maximum(0.0, np.diag(covariance)))
    return {
        "schema": EFFECTIVE_FIT_RESULT_SCHEMA,
        "success": bool(optimized.success),
        "status": str(optimized.message),
        "model_kind": "rovibronic" if isinstance(fitted, RovibronicModel) else "external-field",
        "optimizer": {
            "method": "scipy.optimize.least_squares/trf",
            "jacobian": "analytic Hamiltonian operators plus Hellmann-Feynman",
            "state_tracking": "maximum overlap against the initial eigenvectors",
            "function_evaluations": int(optimized.nfev),
        },
        "parameters": [
            {
                "path": item.path,
                "value": float(value),
                "uncertainty": float(uncertainty),
                "lower": item.lower if math.isfinite(item.lower) else None,
                "upper": item.upper if math.isfinite(item.upper) else None,
            }
            for item, value, uncertainty in zip(variables, optimized.x, uncertainties, strict=True)
        ],
        "metrics": {
            "observation_count": len(rows),
            "free_parameter_count": len(variables),
            "rms_residual_MHz": float(np.sqrt(np.mean(residuals * residuals))),
            "max_abs_residual_MHz": float(np.max(np.abs(residuals))),
            "chi_square": chi_square,
            "reduced_chi_square": reduced_chi_square,
        },
        "identifiability": {
            "rank": rank,
            "full_rank": rank == len(variables),
            "singular_values": np.linalg.svd(jacobian, compute_uv=False).tolist(),
        },
        "covariance": covariance.tolist(),
        "lines": rows,
        "fitted_model": fitted.to_dict(),
    }


def _read_variables(model: Any, raw: Sequence[Mapping[str, Any]]) -> tuple[_Variable, ...]:
    answer = []
    for item in raw:
        path = str(item["path"])
        value = _parameter_value(model, path)
        initial = float(item.get("initial", value))
        scale = float(item.get("scale", max(1.0, abs(initial))))
        if scale <= 0.0 or not math.isfinite(scale):
            raise ValueError(f"variable scale must be positive for {path}")
        answer.append(_Variable(
            path, initial, float(item.get("lower", -np.inf)),
            float(item.get("upper", np.inf)), scale,
        ))
    if len({item.path for item in answer}) != len(answer):
        raise ValueError("effective-fit variable paths must be unique")
    return tuple(answer)


def _parameter_value(model: Any, path: str) -> float:
    parts = path.split(".")
    if parts[0] == "states" and len(parts) == 3:
        value = getattr(model.states[int(parts[1])], parts[2])
    elif parts[0] in {"interactions", "couplings"} and len(parts) in {3, 4}:
        value = getattr(getattr(model, parts[0])[int(parts[1])], parts[2])
        if len(parts) == 4:
            value = getattr(complex(value), parts[3])
    else:
        raise ValueError(f"unsupported effective-fit parameter path: {path}")
    if isinstance(value, complex):
        if abs(value.imag) > 1.0e-15:
            raise ValueError(f"complex parameter requires a .real or .imag suffix: {path}")
        value = value.real
    if not isinstance(value, (int, float)):
        raise ValueError(f"parameter is not numeric: {path}")
    return float(value)


def _apply_values(model: Any, variables: Sequence[_Variable], values: Sequence[float]) -> Any:
    fitted = model
    for variable, value in zip(variables, values, strict=True):
        fitted = _replace_parameter(fitted, variable.path, float(value))
    return fitted


def _replace_parameter(model: Any, path: str, value: float) -> Any:
    parts = path.split(".")
    collection_name, index, field = parts[0], int(parts[1]), parts[2]
    collection = list(getattr(model, collection_name))
    old = getattr(collection[index], field)
    if len(parts) == 4:
        old_complex = complex(old)
        old = complex(value, old_complex.imag) if parts[3] == "real" else complex(old_complex.real, value)
    else:
        old = value
    collection[index] = replace(collection[index], **{field: old})
    return replace(model, **{collection_name: tuple(collection)})


def _keys(observations: Sequence[Mapping[str, Any]], model: Any) -> tuple[Any, ...]:
    values = set()
    for observation in observations:
        values.add(_selector(observation["lower"], model)[0])
        values.add(_selector(observation["upper"], model)[0])
    return tuple(sorted(values))


def _selector(raw: Mapping[str, Any], model: Any) -> tuple[Any, int]:
    if isinstance(model, RovibronicModel):
        return (int(raw["two_J"]), int(raw["parity"])), int(raw["level"])
    return int(raw["two_M"]), int(raw["level"])


def _matrix(model: Any, key: Any, fields: tuple[float, float]) -> np.ndarray:
    if isinstance(model, RovibronicModel):
        return rovibronic_hamiltonian(model, key[0], key[1])[1]
    return external_field_hamiltonian(model, key, fields[0], fields[1])[1]


def _blocks(model: Any, observations: Sequence[Mapping[str, Any]], fields: tuple[float, float]):
    answer = {}
    for key in _keys(observations, model):
        energies, vectors = np.linalg.eigh(_matrix(model, key, fields))
        answer[key] = energies, vectors
    return answer


def _tracked_blocks(model: Any, observations, fields, reference):
    answer = {}
    for key in _keys(observations, model):
        energies, vectors = np.linalg.eigh(_matrix(model, key, fields))
        overlap = np.abs(reference[key][1].conj().T @ vectors) ** 2
        rows, columns = linear_sum_assignment(-overlap)
        permutation = columns[np.argsort(rows)]
        answer[key] = energies[permutation], vectors[:, permutation]
    return answer


def _derivative_operators(model: Any, variables, observations, fields):
    base = {key: _matrix(model, key, fields) for key in _keys(observations, model)}
    answer = []
    for variable in variables:
        shifted = _replace_parameter(model, variable.path, _parameter_value(model, variable.path) + 1.0)
        answer.append({key: _matrix(shifted, key, fields) - matrix for key, matrix in base.items()})
    return answer


def read_effective_fit_request(path: Path | str) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("an effective-fit request must be a JSON object")
    return payload


def read_effective_model(path: Path | str) -> RovibronicModel | ExternalFieldModel:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = payload.get("schema") if isinstance(payload, Mapping) else None
    if schema == "matrix.lock.rovibronic_model.v1":
        return read_rovibronic_model(path)
    if schema == "matrix.lock.external_field_model.v1":
        return read_external_field_model(path)
    raise ValueError(f"unsupported LOCK effective model schema: {schema!r}")


def write_effective_fit_result(path: Path | str, result: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return target


__all__ = [
    "EFFECTIVE_FIT_SCHEMA", "EFFECTIVE_FIT_RESULT_SCHEMA", "fit_effective_hamiltonian",
    "read_effective_fit_request", "read_effective_model", "write_effective_fit_result",
]
