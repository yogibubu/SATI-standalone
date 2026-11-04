"""Fail-closed validation for the frozen Baker TS regression contract."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


BAKER_TS_REGRESSION_CONTRACT_SCHEMA = "matrix.link.baker_ts_regression_contract.v1"
BAKER_TS_EXACT_FREQUENCY_REFERENCE_SCHEMA = (
    "matrix.link.baker_ts_exact_frequency_reference.v1"
)


class BakerTsRegressionContractError(RuntimeError):
    """The stored Baker TS acceptance contract is incomplete or inconsistent."""


@dataclass(frozen=True)
class BakerTsRegressionContract:
    """Validated immutable reference for the 25-case Baker TS qualification."""

    path: Path
    payload: Mapping[str, Any]

    @property
    def cases(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.payload["cases"])

    @property
    def acceptance_policy(self) -> Mapping[str, Any]:
        return self.payload["acceptance_policy"]


def load_baker_ts_regression_contract(
    path: Path | str,
    *,
    repository_root: Path | str | None = None,
) -> BakerTsRegressionContract:
    """Load and fully validate a frozen 25-case Baker TS contract."""

    target = Path(path).expanduser().resolve()
    if not target.is_file():
        raise BakerTsRegressionContractError(f"Baker TS contract does not exist: {target}")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BakerTsRegressionContractError("Baker TS contract is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise BakerTsRegressionContractError("Baker TS contract root must be an object")
    _validate_contract(payload, target=target, repository_root=repository_root)
    return BakerTsRegressionContract(path=target, payload=payload)


def _validate_contract(
    payload: Mapping[str, Any],
    *,
    target: Path,
    repository_root: Path | str | None,
) -> None:
    _equal(payload.get("schema"), BAKER_TS_REGRESSION_CONTRACT_SCHEMA, "schema")
    _equal(payload.get("task_regime"), "TRANSITION_STATE", "task_regime")
    _equal(payload.get("case_count"), 25, "case_count")
    policy = _mapping(payload.get("acceptance_policy"), "acceptance_policy")
    _equal(policy.get("required_converged_cases"), 25, "required_converged_cases")
    required_index = _positive_integer(
        policy.get("required_approximate_hessian_index"),
        "required_approximate_hessian_index",
    )
    _equal(policy.get("exact_frequency_validation_required"), True, "frequency requirement")
    _equal(
        policy.get("required_totally_symmetric_imaginary_frequency_count"),
        1,
        "required totally symmetric imaginary-frequency count",
    )
    _equal(policy.get("full_space_first_order_required"), False, "full-space policy")
    _equal(policy.get("symmetry_preservation_required"), True, "symmetry policy")
    frequency_reference_path = (
        target.parent / str(policy.get("exact_frequency_reference_path", ""))
    ).resolve()
    if frequency_reference_path.parent != target.parent or not frequency_reference_path.is_file():
        raise BakerTsRegressionContractError("exact-frequency reference escapes or is absent")
    _equal(
        _file_sha256(frequency_reference_path),
        _digest(
            policy.get("exact_frequency_reference_sha256"),
            "exact_frequency_reference_sha256",
        ),
        "exact-frequency reference SHA-256",
    )
    frequency_cases = _load_exact_frequency_reference(frequency_reference_path)

    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != 25:
        raise BakerTsRegressionContractError("Baker TS contract must contain exactly 25 cases")
    identifiers = [_positive_integer(case.get("case_id"), "case_id") for case in cases]
    if identifiers != list(range(1, 26)):
        raise BakerTsRegressionContractError("Baker TS case identifiers must be exactly 1..25")
    names = [str(case.get("name", "")) for case in cases]
    if any(not name for name in names) or len(set(names)) != 25:
        raise BakerTsRegressionContractError("Baker TS case names must be non-empty and unique")

    totals = {key: 0 for key in ("link_gdv", "sonic_readallgic", "g16_native")}
    converged = totals.copy()
    geometry_root = (target.parent / "geometries").resolve()
    for case in cases:
        _validate_digest_fields(_mapping(case.get("input_reference"), "input_reference"))
        link = _mapping(case.get("link_gdv"), "link_gdv")
        _equal(link.get("status"), "converged_transition_state", "LINK status")
        _equal(link.get("approximate_hessian_index"), required_index, "LINK Hessian index")
        _finite(link.get("final_energy_hartree"), "final_energy_hartree")
        for key in (
            "final_gradient_inf_norm",
            "final_gradient_rms_norm",
            "final_displacement_inf_norm",
            "final_displacement_rms_norm",
        ):
            if _finite(link.get(key), key) < 0.0:
                raise BakerTsRegressionContractError(f"{key} must be non-negative")
        frequency = _mapping(link.get("exact_frequency_validation"), "frequency validation")
        _equal(frequency.get("performed"), True, "frequency validation performed")
        frequency_case = frequency_cases[int(case["case_id"])]
        for key in (
            "final_point_group",
            "totally_symmetric_irrep",
            "imaginary_mode_irreps",
            "totally_symmetric_imaginary_frequency_count",
            "symmetry_breaking_imaginary_frequency_count",
            "full_space_hessian_index",
            "classification",
        ):
            _equal(frequency.get(key), frequency_case.get(key), f"frequency validation {key}")
        _equal(
            frequency.get("imaginary_frequency_count"),
            frequency_case.get("full_space_hessian_index"),
            "frequency validation imaginary_frequency_count",
        )
        _equal(
            frequency.get("imaginary_frequencies_cm_1"),
            frequency_case.get("imaginary_frequencies_cm-1"),
            "imaginary frequencies",
        )
        geometry = (target.parent / str(link.get("final_geometry_path", ""))).resolve()
        if geometry.parent != geometry_root or not geometry.is_file():
            raise BakerTsRegressionContractError("final geometry escapes or is absent from corpus")
        _equal(
            _file_sha256(geometry),
            _digest(link.get("final_geometry_sha256"), "final_geometry_sha256"),
            "final geometry SHA-256",
        )
        _equal(
            frequency_case.get("source_geometry_sha256"),
            link.get("final_geometry_sha256"),
            "frequency source geometry SHA-256",
        )
        source = _mapping(link.get("source"), "LINK source")
        _digest(source.get("repository_commit"), "repository_commit", length=40)
        for key in (
            "optimizer_protocol_sha256",
            "optimizer_summary_sha256",
            "runtime_method_manifest_sha256",
        ):
            _digest(source.get(key), key)
        if source.get("host") not in {"oracle", "mac-studio"}:
            raise BakerTsRegressionContractError("LINK source host must be oracle or mac-studio")

        for key in totals:
            method = _mapping(case.get(key), key)
            totals[key] += _positive_integer(method.get("optimization_steps"), f"{key} steps")
            converged[key] += int(method.get("converged") is True)

    summary = _mapping(payload.get("summary"), "summary")
    for key in totals:
        method_summary = _mapping(summary.get(key), f"summary.{key}")
        _equal(method_summary.get("total_iterations"), totals[key], f"{key} total")
        _equal(method_summary.get("converged_cases"), converged[key], f"{key} converged cases")
        mean = _finite(method_summary.get("mean_iterations"), f"{key} mean")
        if not math.isclose(mean, totals[key] / 25.0, rel_tol=0.0, abs_tol=5.0e-13):
            raise BakerTsRegressionContractError(f"{key} mean does not match its total")

    provenance = _mapping(payload.get("provenance"), "provenance")
    _digest(provenance.get("comparison_table_sha256"), "comparison_table_sha256")
    frozen_digest = _digest(
        provenance.get("frozen_chart_manifest_sha256"),
        "frozen_chart_manifest_sha256",
    )
    _equal(
        provenance.get("exact_frequency_reference_sha256"),
        policy.get("exact_frequency_reference_sha256"),
        "frequency provenance SHA-256",
    )
    if repository_root is not None:
        root = Path(repository_root).expanduser().resolve()
        frozen = (root / str(provenance.get("frozen_chart_manifest", ""))).resolve()
        if not frozen.is_relative_to(root) or not frozen.is_file():
            raise BakerTsRegressionContractError("frozen chart manifest is outside repository")
        _equal(_file_sha256(frozen), frozen_digest, "frozen chart manifest SHA-256")
        from .frozen_chart_replay import load_frozen_chart_reference

        for case in cases:
            reference = load_frozen_chart_reference(frozen, case["case_id"])
            stored = case["input_reference"]
            _equal(stored["xyzin_sha256"], reference.xyzin_sha256, "input XYZIN SHA-256")
            _equal(
                stored["sonic_definition_sha256"],
                reference.sonic_definition_sha256,
                "SONIC definition SHA-256",
            )
            _equal(
                stored["initial_geometry_sha256"],
                reference.geometry_sha256,
                "initial geometry SHA-256",
            )
            _equal(stored["target_rank"], reference.target_rank, "target SONIC rank")


def _load_exact_frequency_reference(path: Path) -> dict[int, Mapping[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BakerTsRegressionContractError(
            "exact-frequency reference is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise BakerTsRegressionContractError("exact-frequency reference root must be an object")
    _equal(payload.get("schema"), BAKER_TS_EXACT_FREQUENCY_REFERENCE_SCHEMA, "frequency schema")
    _digest(payload.get("calculation_repository_commit"), "frequency calculation commit", length=40)
    _digest(payload.get("collector_repository_commit"), "frequency collector commit", length=40)
    _digest(payload.get("optimization_contract_sha256"), "optimization contract SHA-256")
    _equal(payload.get("method"), "PBE0/def2-SVP", "frequency method")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != 25:
        raise BakerTsRegressionContractError(
            "exact-frequency reference must contain exactly 25 cases"
        )
    identifiers = [_positive_integer(case.get("case_id"), "frequency case_id") for case in cases]
    if sorted(identifiers) != list(range(1, 26)) or len(set(identifiers)) != 25:
        raise BakerTsRegressionContractError("exact-frequency identifiers must be exactly 1..25")
    full_space_first_order = 0
    symmetry_constrained_index_two = 0
    for case in cases:
        for key in (
            "source_geometry_sha256",
            "input_sha256",
            "launch_plan_sha256",
            "authorization_sha256",
            "log_sha256",
        ):
            _digest(case.get(key), f"frequency {key}")
        if case.get("host") not in {"oracle", "mac-studio"}:
            raise BakerTsRegressionContractError("frequency source host is unsupported")
        frequencies = case.get("frequencies_cm-1")
        irreps = case.get("mode_irreps")
        count = _positive_integer(case.get("frequency_count"), "frequency_count")
        _equal(case.get("expected_frequency_count"), count, "expected frequency count")
        if not isinstance(frequencies, list) or len(frequencies) != count:
            raise BakerTsRegressionContractError("frequency spectrum has the wrong length")
        if not isinstance(irreps, list) or len(irreps) != count or any(
            not isinstance(label, str) or not label for label in irreps
        ):
            raise BakerTsRegressionContractError("frequency irrep list is incomplete")
        spectrum = [_finite(value, "frequency") for value in frequencies]
        imaginary = [value for value in spectrum if value < -1.0e-6]
        imaginary_irreps = [
            irrep for value, irrep in zip(spectrum, irreps, strict=True) if value < -1.0e-6
        ]
        _equal(case.get("imaginary_frequencies_cm-1"), imaginary, "imaginary spectrum")
        _equal(case.get("imaginary_mode_irreps"), imaginary_irreps, "imaginary irreps")
        index = _positive_integer(case.get("full_space_hessian_index"), "Hessian index")
        _equal(index, len(imaginary), "full-space Hessian index")
        total_irrep = str(case.get("totally_symmetric_irrep", ""))
        total_count = sum(label == total_irrep for label in imaginary_irreps)
        _equal(
            case.get("totally_symmetric_imaginary_frequency_count"),
            total_count,
            "totally symmetric imaginary-frequency count",
        )
        _equal(total_count, 1, "totally symmetric Hessian index")
        symmetry_breaking_count = len(imaginary) - total_count
        _equal(
            case.get("symmetry_breaking_imaginary_frequency_count"),
            symmetry_breaking_count,
            "symmetry-breaking imaginary-frequency count",
        )
        classification = case.get("classification")
        if index == 1:
            _equal(classification, "full_space_first_order_saddle", "frequency classification")
            full_space_first_order += 1
        elif index == 2 and symmetry_breaking_count == 1:
            _equal(
                classification,
                "symmetry_constrained_first_order_saddle",
                "frequency classification",
            )
            symmetry_constrained_index_two += 1
        else:
            raise BakerTsRegressionContractError("unsupported exact-frequency Hessian index")
        rmsd = _finite(case.get("geometry_aligned_rmsd_angstrom"), "frequency geometry RMSD")
        if rmsd < 0.0 or rmsd > 1.0e-6:
            raise BakerTsRegressionContractError("frequency calculation changed the geometry")
    summary = _mapping(payload.get("summary"), "frequency summary")
    _equal(summary.get("case_count"), 25, "frequency case count")
    _equal(summary.get("accepted_cases"), 25, "frequency accepted cases")
    _equal(
        summary.get("full_space_first_order_cases"),
        full_space_first_order,
        "full-space first-order count",
    )
    _equal(
        summary.get("symmetry_constrained_index_two_cases"),
        symmetry_constrained_index_two,
        "symmetry-constrained index-two count",
    )
    return {int(case["case_id"]): case for case in cases}


def _validate_digest_fields(reference: Mapping[str, Any]) -> None:
    for key in ("xyzin_sha256", "sonic_definition_sha256", "initial_geometry_sha256"):
        _digest(reference.get(key), key)
    _positive_integer(reference.get("target_rank"), "target_rank")


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise BakerTsRegressionContractError(f"{field} must be an object")
    return value


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BakerTsRegressionContractError(f"{field} must be a positive integer")
    return value


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise BakerTsRegressionContractError(f"{field} must be finite")
    return float(value)


def _digest(value: Any, field: str, *, length: int = 64) -> str:
    if not isinstance(value, str):
        raise BakerTsRegressionContractError(f"{field} must be a hexadecimal digest")
    normalized = value.strip().lower()
    if len(normalized) != length or any(item not in "0123456789abcdef" for item in normalized):
        raise BakerTsRegressionContractError(f"{field} must be a {length}-character digest")
    return normalized


def _equal(actual: Any, expected: Any, field: str) -> None:
    if actual != expected:
        raise BakerTsRegressionContractError(f"{field} differs from the frozen contract")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "BAKER_TS_EXACT_FREQUENCY_REFERENCE_SCHEMA",
    "BAKER_TS_REGRESSION_CONTRACT_SCHEMA",
    "BakerTsRegressionContract",
    "BakerTsRegressionContractError",
    "load_baker_ts_regression_contract",
]
