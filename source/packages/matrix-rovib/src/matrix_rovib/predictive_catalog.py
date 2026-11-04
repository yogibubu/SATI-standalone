"""Predictive LOCK catalogs with fit covariance and partition diagnostics."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .native_rotor import MHZ_TO_KELVIN, NativeRotorParameters
from .spectral_fit import (
    FitVariable,
    ParameterTarget,
    SPECTRAL_FIT_RESULT_SCHEMA,
    spin_hyperfine_level_derivatives,
)
from .spin_hyperfine import (
    SpinHyperfineModel,
    SpinHyperfineResult,
    solve_spin_hyperfine_rotor,
    spin_hyperfine_transitions,
)


PREDICTIVE_CATALOG_SCHEMA = "matrix.lock.predictive_catalog.v1"


@dataclass(frozen=True)
class PredictiveCatalogResult:
    payload: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


def _rotor_from_dict(payload: Mapping[str, Any]) -> NativeRotorParameters:
    return NativeRotorParameters(
        A_MHz=float(payload["A_MHz"]),
        B_MHz=float(payload["B_MHz"]),
        C_MHz=float(payload["C_MHz"]),
        quartic_MHz=tuple(float(value) for value in payload.get("quartic_MHz", (0.0,) * 5)),
        sextic_MHz=tuple(float(value) for value in payload.get("sextic_MHz", (0.0,) * 7)),
        octic_MHz=tuple(float(value) for value in payload.get("octic_MHz", (0.0,) * 9)),
        reduction=str(payload.get("reduction", "S")),
        representation=str(payload.get("representation", "Ir")),
    )


def _variables_from_fit(payload: Mapping[str, Any]) -> tuple[FitVariable, ...]:
    variables: list[FitVariable] = []
    for index, item in enumerate(payload["variables"]):
        if not isinstance(item, Mapping):
            raise ValueError("fit variables must be objects")
        targets = tuple(
            ParameterTarget(
                path=str(target["path"]),
                factor=float(target.get("factor", 1.0)),
                offset=float(target.get("offset", 0.0)),
            )
            for target in item.get("targets", ())
        )
        if not targets:
            raise ValueError("each fitted variable must retain at least one target")
        value = float(item["value"])
        variables.append(
            FitVariable(
                name=str(item.get("name", f"p{index + 1}")),
                targets=targets,
                initial=value,
                scale=float(item.get("scale", max(abs(value), 1.0))),
            )
        )
    return tuple(variables)


def _partition(
    solution: SpinHyperfineResult,
    *,
    maximum_N: int,
    temperature_K: float,
    ground_MHz: float,
) -> float:
    beta = MHZ_TO_KELVIN / temperature_K
    return math.fsum(
        (block.two_F + 1.0)
        * level.spin_weight
        * math.exp(-(float(energy) - ground_MHz) * beta)
        for block in solution.blocks
        for energy, level in zip(block.energies_MHz, block.levels, strict=True)
        if level.N <= maximum_N
    )


def predictive_catalog_from_fit(
    fit_payload: Mapping[str, Any],
    *,
    dipole_debye: Sequence[float],
    maximum_N: int,
    temperature_K: float,
    completeness_maximum_N: int | None = None,
    intensity_cut: float = 0.0,
    assembly_backend: str | None = None,
    diagonalizer_backend: str | None = None,
    workers: int = 1,
) -> PredictiveCatalogResult:
    """Build a hyperfine-resolved catalog and propagate the full fit covariance."""
    if fit_payload.get("schema") != SPECTRAL_FIT_RESULT_SCHEMA:
        raise ValueError("predictive catalogs require a LOCK spectral-fit result")
    if maximum_N < 0:
        raise ValueError("maximum_N must be nonnegative")
    if temperature_K <= 0.0 or not math.isfinite(temperature_K):
        raise ValueError("temperature_K must be positive and finite")
    reference_N = maximum_N + 5 if completeness_maximum_N is None else completeness_maximum_N
    if reference_N < maximum_N:
        raise ValueError("completeness_maximum_N must not be smaller than maximum_N")

    parameters = _rotor_from_dict(fit_payload["fitted_rotor"])
    model = SpinHyperfineModel.from_dict(fit_payload["fitted_spin_hyperfine"])
    variables = _variables_from_fit(fit_payload)
    covariance = np.asarray(fit_payload["covariance"], dtype=float)
    expected = (len(variables), len(variables))
    if covariance.shape != expected or not np.all(np.isfinite(covariance)):
        raise ValueError(f"fit covariance must be a finite {expected[0]} by {expected[1]} matrix")
    covariance = 0.5 * (covariance + covariance.T)

    # Two guard manifolds reduce the top-N boundary error for rank-two terms.
    solved_maximum_N = reference_N + 2
    solution = solve_spin_hyperfine_rotor(
        parameters,
        model,
        n_values=range(solved_maximum_N + 1),
        assembly_backend=assembly_backend,
        diagonalizer_backend=diagonalizer_backend,
        workers=workers,
    )
    derivatives = spin_hyperfine_level_derivatives(
        parameters,
        model,
        variables,
        solution,
        assembly_backend=assembly_backend,
    )
    transitions = tuple(
        line
        for line in spin_hyperfine_transitions(
            solution,
            dipole_debye,
            temperature_K=temperature_K,
            intensity_cut=intensity_cut,
        )
        if line.lower_N <= maximum_N and line.upper_N <= maximum_N
    )
    maximum_intensity = max((line.intensity for line in transitions), default=1.0)
    rows = []
    for index, line in enumerate(transitions, start=1):
        gradient = (
            derivatives[line.upper_two_F][:, line.upper_level]
            - derivatives[line.lower_two_F][:, line.lower_level]
        )
        variance = float(gradient @ covariance @ gradient)
        tolerance = 1.0e-12 * max(1.0, float(np.max(np.abs(covariance))))
        if variance < -tolerance:
            raise ValueError("fit covariance produced a negative predicted variance")
        uncertainty = math.sqrt(max(0.0, variance))
        rows.append(
            {
                "line": index,
                "frequency_MHz": line.frequency_MHz,
                "standard_uncertainty_MHz": uncertainty,
                "line_strength_D2": line.line_strength_D2,
                "intensity": line.intensity,
                "relative_intensity": 100.0 * line.intensity / maximum_intensity,
                "lower": {
                    "N": line.lower_N,
                    "two_J": line.lower_two_J,
                    "two_F": line.lower_two_F,
                    "Ka": line.lower_Ka,
                    "Kc": line.lower_Kc,
                    "nuclear_path": list(line.lower_nuclear_path),
                },
                "upper": {
                    "N": line.upper_N,
                    "two_J": line.upper_two_J,
                    "two_F": line.upper_two_F,
                    "Ka": line.upper_Ka,
                    "Kc": line.upper_Kc,
                    "nuclear_path": list(line.upper_nuclear_path),
                },
                "frequency_gradient": {
                    variable.name: float(value)
                    for variable, value in zip(variables, gradient, strict=True)
                },
            }
        )

    ground = min(
        float(energy)
        for block in solution.blocks
        for energy in block.energies_MHz
    )
    q_catalog = _partition(
        solution,
        maximum_N=maximum_N,
        temperature_K=temperature_K,
        ground_MHz=ground,
    )
    q_reference = _partition(
        solution,
        maximum_N=reference_N,
        temperature_K=temperature_K,
        ground_MHz=ground,
    )
    completeness = 1.0 if q_reference == 0.0 else q_catalog / q_reference
    payload = {
        "schema": PREDICTIVE_CATALOG_SCHEMA,
        "source_fit_schema": SPECTRAL_FIT_RESULT_SCHEMA,
        "model_schema": model.schema,
        "temperature_K": float(temperature_K),
        "dipole_debye": [float(value) for value in dipole_debye],
        "maximum_N": int(maximum_N),
        "solved_maximum_N": int(solved_maximum_N),
        "uncertainty": {
            "method": "analytic Hellmann-Feynman covariance propagation",
            "coverage": "one standard uncertainty",
            "variables": [variable.name for variable in variables],
        },
        "partition_function": {
            "catalog": q_catalog,
            "reference": q_reference,
            "reference_maximum_N": int(reference_N),
            "completeness_fraction": completeness,
            "missing_fraction_upper_estimate": max(0.0, 1.0 - completeness),
        },
        "line_count": len(rows),
        "lines": rows,
    }
    return PredictiveCatalogResult(payload)


def read_spectral_fit_result(path: Path | str) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("a spectral-fit result must be a JSON object")
    return payload


def write_predictive_catalog(path: Path | str, result: PredictiveCatalogResult) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    return target


def write_predictive_catalog_csv(path: Path | str, result: PredictiveCatalogResult) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "line",
        "frequency_MHz",
        "standard_uncertainty_MHz",
        "line_strength_D2",
        "intensity",
        "relative_intensity",
        "lower_N",
        "lower_two_J",
        "lower_two_F",
        "lower_Ka",
        "lower_Kc",
        "upper_N",
        "upper_two_J",
        "upper_two_F",
        "upper_Ka",
        "upper_Kc",
    )
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for line in result.payload["lines"]:
            row = {name: line[name] for name in fields[:6]}
            row.update({f"lower_{name}": value for name, value in line["lower"].items() if name != "nuclear_path"})
            row.update({f"upper_{name}": value for name, value in line["upper"].items() if name != "nuclear_path"})
            writer.writerow(row)
    return target


__all__ = [
    "PREDICTIVE_CATALOG_SCHEMA",
    "PredictiveCatalogResult",
    "predictive_catalog_from_fit",
    "read_spectral_fit_result",
    "write_predictive_catalog",
    "write_predictive_catalog_csv",
]
