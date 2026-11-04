"""Fail-closed loader for the frozen covalent/noncovalent minimum corpus."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


MINIMUM_REGRESSION_CONTRACT_SCHEMA = "matrix.link.minimum_regression_contract.v1"


class MinimumRegressionContractError(RuntimeError):
    """The minimum reference corpus is incomplete or inconsistent."""


@dataclass(frozen=True)
class MinimumRegressionContract:
    path: Path
    payload: Mapping[str, Any]
    cases: tuple[Mapping[str, Any], ...]


def load_minimum_regression_contract(path: Path | str) -> MinimumRegressionContract:
    target = Path(path).expanduser().resolve()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MinimumRegressionContractError("minimum contract is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise MinimumRegressionContractError("minimum contract root must be an object")
    _equal(payload.get("schema"), MINIMUM_REGRESSION_CONTRACT_SCHEMA, "schema")
    _equal(payload.get("task_regime"), "MINIMUM", "task regime")
    _equal(payload.get("method"), "GFN2-xTB 6.7.1", "method")
    _equal(payload.get("case_count"), 18, "case count")
    policy = _mapping(payload.get("acceptance_policy"), "acceptance policy")
    _equal(policy.get("required_covalent_cases"), 6, "covalent count")
    _equal(policy.get("required_noncovalent_cases"), 12, "noncovalent count")
    _equal(policy.get("required_converged_cases"), 18, "converged count")
    _equal(policy.get("stationary_point"), "minimum", "stationary point")
    _equal(
        policy.get("transition_state_fallbacks_forbidden"),
        True,
        "transition-state fallback policy",
    )
    protocol = _mapping(payload.get("protocol"), "protocol")
    _digest(protocol.get("repository_commit"), "repository commit", length=40)
    protocol_digest = _digest(
        protocol.get("optimizer_protocol_sha256"), "optimizer protocol SHA-256"
    )
    campaigns = payload.get("campaigns")
    if not isinstance(campaigns, list) or len(campaigns) != 2:
        raise MinimumRegressionContractError("minimum contract needs two campaigns")
    cases: list[Mapping[str, Any]] = []
    for campaign in campaigns:
        record = _mapping(campaign, "campaign")
        domain = str(record.get("domain", ""))
        if domain not in {"covalent", "noncovalent"}:
            raise MinimumRegressionContractError("minimum domain is unsupported")
        table = _local_file(target.parent, record.get("table"), "minimum table")
        source_manifest = _local_file(
            target.parent, record.get("source_manifest"), "minimum source manifest"
        )
        _equal(
            _file_sha256(table),
            _digest(record.get("table_sha256"), "minimum table SHA-256"),
            "minimum table SHA-256",
        )
        _equal(
            _file_sha256(source_manifest),
            _digest(record.get("source_manifest_sha256"), "source manifest SHA-256"),
            "source manifest SHA-256",
        )
        source = json.loads(source_manifest.read_text(encoding="utf-8"))
        source_protocol = (
            source.get("optimizer_protocol_sha256")
            if domain == "covalent"
            else _mapping(source.get("frozen_protocol"), "frozen protocol").get("sha256")
        )
        _equal(source_protocol, protocol_digest, "source optimizer protocol SHA-256")
        rows = tuple(csv.DictReader(table.read_text(encoding="utf-8").splitlines()))
        selected = record.get("selected_systems")
        if selected != "all":
            if not isinstance(selected, list) or len(selected) != len(set(selected)):
                raise MinimumRegressionContractError("selected minimum systems are invalid")
            rows = tuple(row for row in rows if row.get("system") in selected)
            _equal([row["system"] for row in rows], selected, "selected minimum systems")
        cases.extend(_normalize_case(domain, row) for row in rows)
    domains = [case["domain"] for case in cases]
    _equal(domains.count("covalent"), 6, "loaded covalent count")
    _equal(domains.count("noncovalent"), 12, "loaded noncovalent count")
    _equal(len(cases), 18, "loaded case count")
    names = [str(case["system"]) for case in cases]
    if len(names) != len(set(names)):
        raise MinimumRegressionContractError("minimum case names must be unique")
    return MinimumRegressionContract(path=target, payload=payload, cases=tuple(cases))


def _normalize_case(domain: str, row: Mapping[str, str]) -> Mapping[str, Any]:
    system = str(row.get("system", "")).strip()
    if not system:
        raise MinimumRegressionContractError("minimum system name is empty")
    if domain == "covalent":
        return {
            "domain": domain,
            "system": system,
            "native_steps": _positive_int(row.get("native_steps"), "native steps"),
            "link_analytic_steps": _positive_int(
                row.get("link_analytic_steps"), "analytic steps"
            ),
            "link_analytic_evaluations": _positive_int(
                row.get("link_analytic_energy_evaluations"), "analytic evaluations"
            ),
            "link_analytic_final_energy_hartree": _finite(
                row.get("link_analytic_final_energy_hartree"), "analytic energy"
            ),
            "link_numerical_steps": _positive_int(
                row.get("link_numerical_steps"), "numerical steps"
            ),
            "link_numerical_evaluations": _positive_int(
                row.get("link_numerical_energy_evaluations"), "numerical evaluations"
            ),
            "link_numerical_final_energy_hartree": _finite(
                row.get("link_numerical_final_energy_hartree"), "numerical energy"
            ),
        }
    _equal(row.get("backend"), "xTB 6.7.1 / GFN2-xTB", "noncovalent backend")
    _digest(row.get("initial_geometry_sha256"), "initial geometry SHA-256")
    return {
        "domain": domain,
        "system": system,
        "initial_geometry_sha256": row["initial_geometry_sha256"],
        "native_steps": _positive_int(row.get("native_steps"), "native steps"),
        "link_analytic_steps": _positive_int(row.get("link_analytic_steps"), "analytic steps"),
        "link_analytic_evaluations": _positive_int(
            row.get("link_analytic_energy_evaluations"), "analytic evaluations"
        ),
        "link_analytic_final_energy_hartree": _finite(
            row.get("link_analytic_final_energy_hartree"), "analytic energy"
        ),
        "link_numerical_steps": _positive_int(
            row.get("link_numerical_steps"), "numerical steps"
        ),
        "link_numerical_evaluations": _positive_int(
            row.get("link_numerical_energy_evaluations"), "numerical evaluations"
        ),
        "link_numerical_final_energy_hartree": _finite(
            row.get("link_numerical_final_energy_hartree"), "numerical energy"
        ),
    }


def _local_file(root: Path, value: Any, field: str) -> Path:
    path = (root / str(value or "")).resolve()
    if path.parent != root or not path.is_file():
        raise MinimumRegressionContractError(f"{field} escapes or is absent")
    return path


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise MinimumRegressionContractError(f"{field} must be an object")
    return value


def _positive_int(value: Any, field: str) -> int:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as exc:
        raise MinimumRegressionContractError(f"{field} must be an integer") from exc
    if parsed <= 0:
        raise MinimumRegressionContractError(f"{field} must be positive")
    return parsed


def _finite(value: Any, field: str) -> float:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError) as exc:
        raise MinimumRegressionContractError(f"{field} must be finite") from exc
    if not math.isfinite(parsed):
        raise MinimumRegressionContractError(f"{field} must be finite")
    return parsed


def _digest(value: Any, field: str, *, length: int = 64) -> str:
    text = str(value or "").strip().lower()
    if len(text) != length or any(character not in "0123456789abcdef" for character in text):
        raise MinimumRegressionContractError(f"{field} must be a {length}-character digest")
    return text


def _equal(actual: Any, expected: Any, field: str) -> None:
    if actual != expected:
        raise MinimumRegressionContractError(f"{field} differs from the frozen contract")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "MINIMUM_REGRESSION_CONTRACT_SCHEMA",
    "MinimumRegressionContract",
    "MinimumRegressionContractError",
    "load_minimum_regression_contract",
]
