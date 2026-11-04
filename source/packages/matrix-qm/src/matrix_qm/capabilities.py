"""Exact, versioned scientific capability records for canonical QM resolution.

This v2 registry deliberately coexists with the historical broad capability
table in :mod:`matrix_qm.resources`.  It never expands a method family or
silently substitutes a basis, reference, derivative path, or backend.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from importlib import resources
import json
from pathlib import Path
from typing import Any, Mapping


QM_CAPABILITY_REGISTRY_V2_SCHEMA = "matrix.qm.capability_registry.v2"
QM_CAPABILITY_REGISTRY_V2_ID = "matrix-qm-capabilities-v2"
QM_CAPABILITY_REGISTRY_V2_VERSION = "2.1.3"

_ALLOWED_BACKENDS = ("xtb", "pyscf", "pyfock", "et", "orca", "psi4")
_DERIVATIVE_KINDS = ("energy", "gradient", "hessian")
_LICENSE_MODELS = {"open-source", "free-academic"}
_CERTIFICATION_STATES = {"certified", "documented_candidate"}


@dataclass(frozen=True)
class ScientificCapabilityKey:
    """Complete scientific and execution key used for exact capability lookup."""

    backend: str
    method: str
    basis: str
    electronic_state: str
    spin: str
    reference: str
    state_model: str
    charge: int
    multiplicity: int
    frozen_core_policy: str
    ecp_policy: str
    derivative: str
    derivative_mode: str
    accelerator: str
    platform: str
    require_production: bool = True

    def __post_init__(self) -> None:
        if self.derivative not in _DERIVATIVE_KINDS:
            raise ValueError(f"unknown derivative kind: {self.derivative}")
        if not isinstance(self.charge, int) or isinstance(self.charge, bool):
            raise TypeError("charge must be an integer")
        if (
            not isinstance(self.multiplicity, int)
            or isinstance(self.multiplicity, bool)
            or self.multiplicity < 1
        ):
            raise ValueError("multiplicity must be a positive integer")


@dataclass(frozen=True)
class QMScientificCapability:
    """One immutable exact capability record."""

    payload: dict[str, Any]

    @property
    def capability_id(self) -> str:
        return str(self.payload["id"])

    @property
    def backend(self) -> str:
        return str(self.payload["backend"])

    @property
    def production_enabled(self) -> bool:
        return bool(self.payload["production_enabled"])

    def derivative_mode(self, derivative: str) -> str:
        if derivative not in _DERIVATIVE_KINDS:
            raise ValueError(f"unknown derivative kind: {derivative}")
        return str(self.payload["derivatives"][derivative])

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(self.payload)

    def matches(self, key: ScientificCapabilityKey) -> bool:
        record = self.payload
        exact_fields = (
            "backend",
            "method",
            "basis",
            "electronic_state",
            "reference",
            "state_model",
            "frozen_core_policy",
            "ecp_policy",
        )
        if any(record[field] != getattr(key, field) for field in exact_fields):
            return False
        if record["spin"] not in {key.spin, "any"}:
            return False
        if not _domain_contains(record["charge_domain"], key.charge, quantity="charge"):
            return False
        if not _domain_contains(
            record["multiplicity_domain"], key.multiplicity, quantity="multiplicity"
        ):
            return False
        if record["derivatives"][key.derivative] != key.derivative_mode:
            return False
        if key.accelerator not in record["accelerators"]:
            return False
        if key.platform not in record["platforms"]:
            return False
        if key.require_production and not record["production_enabled"]:
            return False
        return True


@dataclass(frozen=True)
class QMCapabilityRegistryV2:
    """Validated handle to the packaged exact capability authority."""

    payload: dict[str, Any]
    sha256: str
    source: str

    @property
    def records(self) -> tuple[QMScientificCapability, ...]:
        return tuple(QMScientificCapability(deepcopy(item)) for item in self.payload["records"])

    def by_id(self, capability_id: str) -> QMScientificCapability:
        matches = [item for item in self.records if item.capability_id == capability_id]
        if len(matches) != 1:
            raise ValueError(f"unknown exact QM capability: {capability_id}")
        return matches[0]

    def matching(self, key: ScientificCapabilityKey) -> tuple[QMScientificCapability, ...]:
        return tuple(item for item in self.records if item.matches(key))

    def resolve_exact(self, key: ScientificCapabilityKey) -> QMScientificCapability:
        matches = self.matching(key)
        if not matches:
            mode = "production" if key.require_production else "validation"
            raise RuntimeError(
                "unsupported exact QM capability in "
                f"{mode} mode: {key.backend}/{key.method}/{key.basis}/"
                f"{key.electronic_state}/{key.spin}/{key.reference}/"
                f"{key.state_model}/{key.derivative}={key.derivative_mode}"
            )
        if len(matches) != 1:
            raise RuntimeError("ambiguous exact QM capability registry match")
        return matches[0]

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(self.payload)


def load_qm_capability_registry_v2() -> QMCapabilityRegistryV2:
    resource = resources.files("matrix_qm").joinpath("data/qm_capabilities_v2.json")
    raw = resource.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("MATRIX exact QM capability registry is unreadable") from exc
    validate_qm_capability_registry_v2(payload)
    return QMCapabilityRegistryV2(
        payload=deepcopy(payload),
        sha256=hashlib.sha256(raw).hexdigest(),
        source="matrix_qm:data/qm_capabilities_v2.json",
    )


def qm_capability_registry_v2_schema_path() -> Path:
    return Path(
        str(resources.files("matrix_qm").joinpath("schemas/qm-capability-registry-v2.schema.json"))
    )


def validate_qm_capability_registry_v2(payload: Mapping[str, Any]) -> None:
    """Reject scientific, licensing, or governance drift without heuristics."""

    if payload.get("schema") != QM_CAPABILITY_REGISTRY_V2_SCHEMA:
        raise RuntimeError("unsupported exact QM capability registry schema")
    if (
        payload.get("registry_id") != QM_CAPABILITY_REGISTRY_V2_ID
        or payload.get("manifest_version") != QM_CAPABILITY_REGISTRY_V2_VERSION
        or payload.get("status") != "approved"
    ):
        raise RuntimeError("exact QM capability registry identity or status changed")
    if payload.get("selection_policy") != "exact_full_key_fail_closed":
        raise RuntimeError("exact QM capability registry must fail closed")
    if tuple(payload.get("allowed_backends", ())) != _ALLOWED_BACKENDS:
        raise RuntimeError("approved open/free QM backend set changed")
    if payload.get("change_policy") != "new_manifest_version_and_explicit_approval":
        raise RuntimeError("exact QM capability change policy changed")

    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise RuntimeError("exact QM capability registry has no records")
    required = {
        "id",
        "backend",
        "method",
        "basis",
        "electronic_state",
        "spin",
        "reference",
        "state_model",
        "charge_domain",
        "multiplicity_domain",
        "frozen_core_policy",
        "ecp_policy",
        "derivatives",
        "state_vectors",
        "accelerators",
        "platforms",
        "license_model",
        "certification",
        "production_enabled",
        "note",
    }
    ids: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping) or set(record) != required:
            raise RuntimeError("exact QM capability record has incomplete or unknown fields")
        capability_id = str(record["id"])
        if not capability_id or capability_id in ids:
            raise RuntimeError("exact QM capability identifiers must be unique and nonempty")
        ids.add(capability_id)
        if record["backend"] not in _ALLOWED_BACKENDS:
            raise RuntimeError("unapproved QM backend in exact capability registry")
        if record["license_model"] not in _LICENSE_MODELS:
            raise RuntimeError("non-open/non-free QM backend in exact capability registry")
        if record["certification"] not in _CERTIFICATION_STATES:
            raise RuntimeError("unknown exact QM capability certification state")
        if record["production_enabled"] and record["certification"] != "certified":
            raise RuntimeError("uncertified exact QM capability enabled for production")
        if record["production_enabled"] and record["backend"] != "xtb":
            raise RuntimeError("uncertified canonical backend family enabled for production")
        if record["charge_domain"] not in {"integer", "any_supported_by_backend"}:
            raise RuntimeError("unknown charge domain in exact QM capability")
        if record["multiplicity_domain"] not in {
            "1",
            ">=2_high_spin",
            "any_supported_by_backend",
        }:
            raise RuntimeError("unknown multiplicity domain in exact QM capability")
        derivatives = record["derivatives"]
        if not isinstance(derivatives, Mapping) or set(derivatives) != set(_DERIVATIVE_KINDS):
            raise RuntimeError("exact QM capability has an incomplete derivative contract")
        if any(
            not isinstance(record[field], str) or not record[field]
            for field in required
            - {
                "derivatives",
                "accelerators",
                "platforms",
                "production_enabled",
            }
        ):
            raise RuntimeError("exact QM capability contains an empty scientific key field")
        if not record["accelerators"] or not record["platforms"]:
            raise RuntimeError("exact QM capability lacks accelerator or platform scope")

    _require_derivative_contract(
        records,
        "et.eomccsd.def2tzvp.rhf.fc.v1",
        gradient="link_numerical_energy",
        hessian="unavailable",
    )
    _require_derivative_contract(
        records,
        "pyscf.fno_ccsdt.def2tzvpp.rhf.v1",
        gradient="link_numerical_energy",
        hessian="unavailable",
    )
    _require_derivative_contract(
        records,
        "orca.dlpno_ccsdt1.def2tzvpp.rhf.v1",
        gradient="link_numerical_energy",
        hessian="unavailable",
    )
    _require_derivative_contract(
        records,
        "orca.rimp2.def2tzvp.rhf.fc.v1",
        gradient="analytic",
        hessian="modal_finite_difference_analytic_gradient",
    )
    _require_derivative_contract(
        records,
        "orca.rimp2.def2tzvp.rhf.fc.energy-only.v1",
        gradient="link_numerical_energy",
        hessian="unavailable",
    )
    _require_derivative_contract(
        records,
        "orca.rimp2.def2tzvp.rohf.fc.v1",
        gradient="link_numerical_energy",
        hessian="unavailable",
    )
    if not any(item["backend"] == "psi4" and item["spin"] == "open_shell" for item in records):
        raise RuntimeError("restricted-open Psi4 validation candidates are missing")
    if not any(
        item["backend"] == "xtb"
        and item["certification"] == "certified"
        and item["production_enabled"]
        for item in records
    ):
        raise RuntimeError("certified production xTB baseline is missing")


def _domain_contains(domain: str, value: int, *, quantity: str) -> bool:
    if domain == "any_supported_by_backend":
        return True
    if quantity == "charge" and domain == "integer":
        return True
    if quantity == "multiplicity" and domain == ">=2_high_spin":
        return value >= 2
    try:
        return value == int(domain)
    except (TypeError, ValueError):
        raise RuntimeError(
            f"unsupported {quantity} domain in exact QM capability: {domain}"
        ) from None


def _require_derivative_contract(
    records: list[Mapping[str, Any]],
    capability_id: str,
    *,
    gradient: str,
    hessian: str,
) -> None:
    matches = [item for item in records if item["id"] == capability_id]
    if len(matches) != 1:
        raise RuntimeError(f"required exact QM capability is missing: {capability_id}")
    if matches[0]["derivatives"]["gradient"] != gradient:
        raise RuntimeError(f"approved gradient contract changed: {capability_id}")
    if matches[0]["derivatives"]["hessian"] != hessian:
        raise RuntimeError(f"approved Hessian contract changed: {capability_id}")
