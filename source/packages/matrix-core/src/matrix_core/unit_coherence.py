"""Shared unit normalization and handoff validation for MATRIX artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


UNIT_COHERENCE_SCHEMA = "matrix.core.unit_coherence.v1"

_UNITS: dict[str, tuple[str, str, float]] = {
    "angstrom": ("length", "angstrom", 1.0),
    "ang": ("length", "angstrom", 1.0),
    "å": ("length", "angstrom", 1.0),
    "bohr": ("length", "bohr", 0.529177210903),
    "hartree": ("energy", "hartree", 1.0),
    "eh": ("energy", "hartree", 1.0),
    "kcal/mol": ("energy", "kcal/mol", 1.0 / 627.509474),
    "kj/mol": ("energy", "kJ/mol", 1.0 / 2625.499639),
    "cm-1": ("wavenumber", "cm-1", 1.0),
    "cm^-1": ("wavenumber", "cm-1", 1.0),
    "ghz": ("frequency", "GHz", 1.0),
    "mhz": ("frequency", "MHz", 0.001),
    "k": ("temperature", "K", 1.0),
    "kelvin": ("temperature", "K", 1.0),
    "debye": ("dipole", "debye", 1.0),
    "amu": ("mass", "amu", 1.0),
    "si": ("system", "SI", 1.0),
    "dimensionless": ("dimensionless", "dimensionless", 1.0),
}


@dataclass(frozen=True)
class UnitDescriptor:
    dimension: str
    canonical: str
    to_base: float


def describe_unit(unit: str) -> UnitDescriptor:
    normalized = str(unit).strip().casefold().replace(" ", "")
    record = _UNITS.get(normalized)
    if record is None:
        raise ValueError(f"unsupported MATRIX unit: {unit}")
    return UnitDescriptor(*record)


def convert_value(value: float, source_unit: str, target_unit: str) -> float:
    source = describe_unit(source_unit)
    target = describe_unit(target_unit)
    if source.dimension != target.dimension:
        raise ValueError(
            f"incompatible MATRIX units: {source_unit} and {target_unit}"
        )
    return float(value) * source.to_base / target.to_base


def validate_unit_handoff(
    artifacts: Sequence[Mapping[str, Any]],
    *,
    expected_dimensions: Mapping[str, str] | None = None,
    expected_conventions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate declared units without guessing missing scientific conventions."""

    expected = dict(expected_dimensions or {})
    conventions = dict(expected_conventions or {})
    records: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []
    for artifact in artifacts:
        role = str(artifact.get("role", "artifact"))
        raw_unit = str(artifact.get("units", "")).strip()
        required_dimension = str(expected.get(role, "")).strip()
        if not raw_unit:
            status = "missing" if required_dimension else "not_declared"
            records.append({"role": role, "status": status, "units": ""})
            if required_dimension:
                issues.append(
                    {
                        "role": role,
                        "reason": "units are required for this handoff",
                    }
                )
            continue
        try:
            descriptor = describe_unit(raw_unit)
        except ValueError as exc:
            records.append({"role": role, "status": "unsupported", "units": raw_unit})
            issues.append({"role": role, "reason": str(exc)})
            continue
        status = "valid"
        if required_dimension and descriptor.dimension != required_dimension:
            status = "incompatible"
            issues.append(
                {
                    "role": role,
                    "reason": (
                        f"expected {required_dimension}, found {descriptor.dimension}"
                    ),
                }
            )
        records.append(
            {
                "role": role,
                "status": status,
                "units": raw_unit,
                "canonical_units": descriptor.canonical,
                "dimension": descriptor.dimension,
            }
        )
        for key in ("atom_order_hash", "axis_convention", "normal_mode_order"):
            if key not in conventions:
                continue
            actual = artifact.get(key)
            if actual != conventions[key]:
                issues.append(
                    {
                        "role": role,
                        "reason": (
                            f"{key} mismatch: expected {conventions[key]!r}, "
                            f"found {actual!r}"
                        ),
                    }
                )
                records[-1]["status"] = "incompatible"
    return {
        "schema": UNIT_COHERENCE_SCHEMA,
        "valid": not issues,
        "records": records,
        "issues": issues,
    }


__all__ = [
    "UNIT_COHERENCE_SCHEMA",
    "UnitDescriptor",
    "convert_value",
    "describe_unit",
    "validate_unit_handoff",
]
