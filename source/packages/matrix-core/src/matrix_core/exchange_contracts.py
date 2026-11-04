"""Portable result validation at scientific owner exchange boundaries.

This validates structure and declared qualification scope. It does not replace
the owner's numerical convergence, covariance or physical acceptance gates.
Only the standard library is imported, including in the KEYMAKER runtime.
"""
from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from importlib import resources
import json
import math
from pathlib import Path


@lru_cache(maxsize=1)
def exchange_contracts():
    payload = json.loads(resources.files("matrix_core").joinpath("data/exchange_contracts_v1.json").read_text())
    rows = payload["contracts"]
    result = {row["identifier"]: row for row in rows}
    if len(result) != len(rows) or any(len(row["required_fields"]) < 2 for row in rows):
        raise RuntimeError("duplicate or empty exchange contract")
    return result


_BOOLEAN_FIELDS = {
    "complete", "converged", "qualified", "success", "local_state_numerical_controls_passed",
    "global_bound_state_claimed", "full_molecular_rate_qualified", "potential_modified",
    "polynomial_reexpansion_used", "physical_field_qualified", "infinite_basis_convergence_certified",
    "quartic_globally_diagonalized", "reactive_coordinate_is_bound", "global_scattering_claimed",
    "local_derivatives_passed", "local_and_nonlocal_gates_passed", "reduction_accepted",
    "reference_jet_preserved",
}
_COUNT_FIELDS = {
    "rank", "parameter_count", "training_count", "penalty_count", "workers", "cpu_budget",
    "memory_budget_bytes", "basis_size", "nonzeros", "evaluation_count", "maximum_evaluations",
}
_OBJECT_FIELDS = {
    "matrix.lock.assignment_workflow.v1": ("exploration",),
    "matrix.lock.spectral_profile_fit.v1": ("fit_request", "profile"),
    "matrix.lock.assignment_workflow_result.v1": ("assignments",),
    "matrix.lock.assignment_review.v1": ("decisions",),
    "matrix.trinity.esalc_vibronic_result.v1": ("internal_checks", "model"),
    "matrix.trinity.native_pes_variational_result.v1": ("independent_variations", "tolerances", "protocol"),
}


def portable_value(value):
    """Normalize NumPy scalars/arrays without importing a numerical kernel."""
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("portable records require string mapping keys")
        return {key: portable_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [portable_value(item) for item in value]
    if type(value).__module__.split(".")[0] == "numpy":
        return portable_value(value.tolist())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("nonfinite value at a portable scientific exchange boundary")
        return value
    raise TypeError(f"unsupported portable scientific value: {type(value).__name__}")


def validate_exchange_tree(value, *, expected_schema=None):
    """Validate all recognized nested records; return a portable independent copy."""
    normalized = portable_value(value)
    if expected_schema is not None:
        if expected_schema not in exchange_contracts():
            raise ValueError(f"unknown exchange contract: {expected_schema}")
        if not isinstance(normalized, dict) or normalized.get("schema") != expected_schema:
            raise ValueError(f"expected exchange schema {expected_schema}")

    def check(item, path):
        if isinstance(item, list):
            for index, child in enumerate(item):
                check(child, f"{path}[{index}]")
        elif isinstance(item, dict):
            identifier = item.get("schema")
            contract = exchange_contracts().get(identifier) if isinstance(identifier, str) else None
            if contract is not None:
                missing = set(contract["required_fields"]) - set(item)
                if missing:
                    raise ValueError(f"{path}: {identifier} missing fields {sorted(missing)}")
                for key in _OBJECT_FIELDS.get(identifier, ()):
                    if not isinstance(item[key], dict):
                        raise ValueError(f"{path}.{key}: object required")
                for key in contract["required_fields"]:
                    if key.endswith("_sha256"):
                        digest = item[key]
                        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                            raise ValueError(f"{path}.{key}: SHA-256 identity required")
                for key in _BOOLEAN_FIELDS & item.keys():
                    if not isinstance(item[key], bool):
                        raise ValueError(f"{path}.{key}: explicit boolean required")
                for key in _COUNT_FIELDS & item.keys():
                    number = item[key]
                    if isinstance(number, bool) or not isinstance(number, int) or number < 0:
                        raise ValueError(f"{path}.{key}: nonnegative integer required")
                if identifier == "matrix.trinity.esalc_vibronic_result.v1":
                    if item["status"] == "internal_checks_passed" and any(
                        result is False for result in item["internal_checks"].values()
                    ):
                        raise ValueError("vibronic status contradicts its numerical checks")
                if identifier == "matrix.trinity.native_pes_variational_result.v1":
                    for key in ("global_bound_state_claimed", "full_molecular_rate_qualified",
                                "potential_modified", "polynomial_reexpansion_used"):
                        if item[key] is not False:
                            raise ValueError(f"local native-PES result cannot claim {key}")
                    if item["local_state_numerical_controls_passed"] and (
                        not item["accepted_local_state_files"] or len(item["cases"]) < 4
                        or not all(item["independent_variations"].values())
                    ):
                        raise ValueError("native-PES acceptance lacks independent controls")
                if identifier.startswith("matrix.architect.") and "physical_field_qualified" in item:
                    if item["physical_field_qualified"] is not False:
                        raise ValueError("a native fit candidate does not establish physical qualification")
            for key, child in item.items():
                check(child, f"{path}.{key}")

    check(normalized, "$")
    return normalized


def validate_request_budget(value):
    """Check an explicit native protocol host/resource budget before side effects."""
    if not isinstance(value, dict):
        raise ValueError("native protocol request must be an object")
    host = value.get("required_host")
    if not isinstance(host, str) or not host.strip():
        raise ValueError("explicit nonempty required_host required")
    memory = value.get("memory_gib")
    if isinstance(memory, bool) or not isinstance(memory, int) or memory != 128:
        raise ValueError("native protocol requires an integer 128 GiB budget")
    workers = value.get("workers", 1)
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 128:
        raise ValueError("workers must be an integer between 1 and 128")
    return value


def write_exchange_json(path, value, *, expected_schema=None):
    """Validate before any write and atomically replace an exchange artifact."""
    value = validate_exchange_tree(value, expected_schema=expected_schema)
    encoded = json.dumps(value, indent=2, allow_nan=False)+"\n"
    from .atomic_io import atomic_write_text
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(target, encoded)
    return target
