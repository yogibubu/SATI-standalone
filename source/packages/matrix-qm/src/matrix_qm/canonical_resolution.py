"""Fail-closed resolution of the approved canonical MATRIX QM hierarchy."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, fields
import hashlib
from importlib import resources
import json
from pathlib import Path
from typing import Any, Mapping

from .capabilities import (
    QMScientificCapability,
    ScientificCapabilityKey,
    load_qm_capability_registry_v2,
)
from .element_policy import resolve_element_policy
from .ladder import load_canonical_qm_ladder
from .resources import BasisSetConfirmationRequired


CANONICAL_QM_RUN_SCHEMA = "matrix.qm.canonical_run.v1"
CANONICAL_QM_REQUEST_SCHEMA = "matrix.qm.canonical_request.v1"


class BasisAvailabilityUnverified(RuntimeError):
    """The backend/library basis inventory was not checked."""


class CanonicalBasisSetConfirmationRequired(BasisSetConfirmationRequired):
    """A missing canonical basis may only be acquired after confirmation."""

    def __init__(self, *, basis: str, backend: str) -> None:
        self.basis = basis
        self.backend = backend
        super().__init__(
            f"{basis} is unavailable for {backend}; explicit confirmation is required "
            "before automatic Basis Set Exchange acquisition"
        )


@dataclass(frozen=True)
class CanonicalQMRequest:
    """User-visible inputs that are not already fixed by the ladder."""

    level_or_profile: str
    backend: str
    electronic_state: str
    spin: str
    state_root: int | None
    target_state_spin: str | None
    charge: int
    multiplicity: int
    atomic_numbers: tuple[int, ...]
    ecp_mask: tuple[bool, ...]
    derivative: str
    derivative_mode: str
    accelerator: str
    platform: str
    application_domain: str
    single_reference_confirmed: bool
    basis_available: bool | None
    basis_source: str | None
    core_valence_requested: bool = False
    validation_mode: bool = False

    def __post_init__(self) -> None:
        text_fields = (
            "level_or_profile",
            "backend",
            "electronic_state",
            "spin",
            "derivative",
            "derivative_mode",
            "accelerator",
            "platform",
            "application_domain",
        )
        if any(
            not isinstance(getattr(self, name), str) or not getattr(self, name).strip()
            for name in text_fields
        ):
            raise TypeError("canonical QM text fields must be nonempty strings")
        for name in (
            "single_reference_confirmed",
            "core_valence_requested",
            "validation_mode",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")
        for name in ("charge", "multiplicity"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
        if self.state_root is not None and (
            not isinstance(self.state_root, int) or isinstance(self.state_root, bool)
        ):
            raise TypeError("state_root must be an integer or null")
        if self.target_state_spin is not None and not isinstance(self.target_state_spin, str):
            raise TypeError("target_state_spin must be a string or null")
        if self.basis_available is not None and not isinstance(self.basis_available, bool):
            raise TypeError("basis_available must be boolean or null")
        if self.basis_source is not None and not isinstance(self.basis_source, str):
            raise TypeError("basis_source must be a string or null")
        if not isinstance(self.atomic_numbers, tuple) or any(
            not isinstance(value, int) or isinstance(value, bool) for value in self.atomic_numbers
        ):
            raise TypeError("atomic_numbers must be a tuple of integers")
        if not isinstance(self.ecp_mask, tuple) or any(
            not isinstance(value, bool) for value in self.ecp_mask
        ):
            raise TypeError("ecp_mask must be a tuple of booleans")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema"] = CANONICAL_QM_REQUEST_SCHEMA
        payload["atomic_numbers"] = list(self.atomic_numbers)
        payload["ecp_mask"] = list(self.ecp_mask)
        return payload


@dataclass(frozen=True)
class ResolvedCanonicalQMRun:
    payload: dict[str, Any]

    @property
    def manifest_sha256(self) -> str:
        return str(self.payload["manifest_sha256"])

    @property
    def capability_id(self) -> str:
        return str(self.payload["capability"]["record"]["id"])

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(self.payload)


def resolve_canonical_qm_run(request: CanonicalQMRequest) -> ResolvedCanonicalQMRun:
    """Resolve one exact run without scientific or backend substitution."""

    ladder = load_canonical_qm_ladder()
    _validate_request_domain(request, ladder.payload["applicability"])
    model = _scientific_model(request, ladder)
    if not request.validation_mode and not ladder.production_ready:
        raise RuntimeError(
            "canonical MATRIX QM ladder is not production-certified; "
            "an explicit validation run is required"
        )

    basis = _resolve_basis(request, basis=str(model["basis"]))
    frozen_core_required = model["frozen_core"] == "mandatory"
    element_policy = resolve_element_policy(
        request.atomic_numbers,
        ecp_mask=request.ecp_mask,
        frozen_core_required=frozen_core_required,
        core_valence_requested=request.core_valence_requested,
    )
    if request.core_valence_requested and not element_policy.core_valence_supported:
        raise RuntimeError(
            "canonical complete-system core--valence correction is unsupported: "
            f"{element_policy.core_valence_status}"
        )

    registry = load_qm_capability_registry_v2()
    capability = _resolve_capability(
        request,
        model=model,
        registry_records=registry.records,
        frozen_core_required=frozen_core_required,
        any_ecp=element_policy.any_ecp,
    )
    status = (
        "production"
        if ladder.production_ready and capability.production_enabled
        else "validation_only"
    )
    if status != "production" and not request.validation_mode:
        raise RuntimeError("exact canonical QM capability is not production-certified")

    request_record = {
        "level_or_profile": request.level_or_profile.upper(),
        "backend": request.backend,
        "electronic_state": request.electronic_state,
        "spin": request.spin,
        "state_root": request.state_root,
        "target_state_spin": request.target_state_spin,
        "charge": request.charge,
        "multiplicity": request.multiplicity,
        "atomic_numbers": list(request.atomic_numbers),
        "ecp_mask": list(request.ecp_mask),
        "derivative": request.derivative,
        "derivative_mode": request.derivative_mode,
        "accelerator": request.accelerator,
        "platform": request.platform,
        "application_domain": request.application_domain,
        "single_reference_confirmed": request.single_reference_confirmed,
        "core_valence_requested": request.core_valence_requested,
        "validation_mode": request.validation_mode,
    }
    payload: dict[str, Any] = {
        "schema": CANONICAL_QM_RUN_SCHEMA,
        "manifest_id": (
            f"{ladder.protocol_id}:{request.level_or_profile.upper()}:{capability.capability_id}"
        ),
        "resolution_status": status,
        "request": request_record,
        "scientific_model": model,
        "capability": {
            "record": capability.to_dict(),
            "registry_id": registry.payload["registry_id"],
            "manifest_version": registry.payload["manifest_version"],
            "sha256": registry.sha256,
        },
        "element_policy": element_policy.to_dict(),
        "basis": basis,
        "authorities": {
            "ladder": {
                "protocol_id": ladder.protocol_id,
                "manifest_version": ladder.manifest_version,
                "sha256": ladder.sha256,
            },
            "capabilities": {
                "registry_id": registry.payload["registry_id"],
                "manifest_version": registry.payload["manifest_version"],
                "sha256": registry.sha256,
            },
            "element_policy": {
                "policy_id": element_policy.policy_id,
                "manifest_version": element_policy.policy_version,
                "sha256": element_policy.policy_sha256,
            },
            "optimizer": deepcopy(ladder.payload["optimizer_contract"]),
        },
        "governance": {
            "silent_substitution": False,
            "molecule_specific_patches": False,
            "backend_specific_optimizer_patches": False,
            "change_policy": "new_manifest_version_and_explicit_approval",
        },
    }
    payload["manifest_sha256"] = _sha256(payload)
    return ResolvedCanonicalQMRun(payload=payload)


def canonical_qm_request_from_dict(payload: Mapping[str, Any]) -> CanonicalQMRequest:
    """Load the one shared CLI/KEYMAKER canonical request contract."""

    data = dict(payload)
    if data.pop("schema", None) != CANONICAL_QM_REQUEST_SCHEMA:
        raise ValueError("unsupported canonical QM request schema")
    allowed = {item.name for item in fields(CanonicalQMRequest)}
    unknown = sorted(set(data) - allowed)
    missing = sorted(allowed - set(data))
    if unknown or missing:
        detail = []
        if missing:
            detail.append("missing=" + ",".join(missing))
        if unknown:
            detail.append("unknown=" + ",".join(unknown))
        raise ValueError("invalid canonical QM request fields: " + "; ".join(detail))
    if not isinstance(data["atomic_numbers"], list) or any(
        not isinstance(value, int) or isinstance(value, bool) for value in data["atomic_numbers"]
    ):
        raise TypeError("canonical atomic_numbers must be an array of integers")
    if not isinstance(data["ecp_mask"], list) or any(
        not isinstance(value, bool) for value in data["ecp_mask"]
    ):
        raise TypeError("canonical ecp_mask must be an array of booleans")
    data["atomic_numbers"] = tuple(data["atomic_numbers"])
    data["ecp_mask"] = tuple(data["ecp_mask"])
    return CanonicalQMRequest(**data)


def resolve_canonical_qm_run_from_dict(
    payload: Mapping[str, Any],
) -> ResolvedCanonicalQMRun:
    return resolve_canonical_qm_run(canonical_qm_request_from_dict(payload))


def canonical_qm_run_schema_path() -> Path:
    return Path(
        str(resources.files("matrix_qm").joinpath("schemas/canonical-qm-run-v1.schema.json"))
    )


def canonical_qm_request_schema_path() -> Path:
    return Path(
        str(resources.files("matrix_qm").joinpath("schemas/canonical-qm-request-v1.schema.json"))
    )


def validate_resolved_canonical_qm_run(payload: dict[str, Any]) -> None:
    """Verify a serialized resolved run against all packaged authorities."""

    if payload.get("schema") != CANONICAL_QM_RUN_SCHEMA:
        raise RuntimeError("unsupported resolved canonical QM run schema")
    recorded_hash = str(payload.get("manifest_sha256", ""))
    unhashed = deepcopy(payload)
    unhashed.pop("manifest_sha256", None)
    if recorded_hash != _sha256(unhashed):
        raise RuntimeError("resolved canonical QM run hash mismatch")
    if payload.get("governance") != {
        "silent_substitution": False,
        "molecule_specific_patches": False,
        "backend_specific_optimizer_patches": False,
        "change_policy": "new_manifest_version_and_explicit_approval",
    }:
        raise RuntimeError("resolved canonical QM governance changed")

    ladder = load_canonical_qm_ladder()
    registry = load_qm_capability_registry_v2()
    authorities = payload.get("authorities", {})
    if authorities.get("ladder") != {
        "protocol_id": ladder.protocol_id,
        "manifest_version": ladder.manifest_version,
        "sha256": ladder.sha256,
    }:
        raise RuntimeError("resolved run does not match the packaged canonical ladder")
    if authorities.get("capabilities") != {
        "registry_id": registry.payload["registry_id"],
        "manifest_version": registry.payload["manifest_version"],
        "sha256": registry.sha256,
    }:
        raise RuntimeError("resolved run does not match the packaged capability registry")
    capability_record = payload.get("capability", {}).get("record", {})
    capability = registry.by_id(str(capability_record.get("id", "")))
    if capability_record != capability.to_dict():
        raise RuntimeError("resolved run capability record was modified")
    request = payload.get("request", {})
    model = payload.get("scientific_model", {})
    if (
        request.get("backend") != capability.backend
        or model.get("method") != capability.payload["method"]
        or model.get("basis") != capability.payload["basis"]
        or model.get("reference") != capability.payload["reference"]
        or model.get("state_model") != capability.payload["state_model"]
    ):
        raise RuntimeError("resolved run scientific key and capability record differ")


def _scientific_model(request: CanonicalQMRequest, ladder: Any) -> dict[str, Any]:
    key = str(request.level_or_profile).strip().upper()
    level_ids = {item["id"] for item in ladder.payload["levels"]}
    profile_ids = {item["id"] for item in ladder.payload["profiles"]}
    if key in level_ids:
        record = ladder.level(key)
        variant = ladder.variant(
            key,
            electronic_state=request.electronic_state,
            spin=request.spin,
        )
        if variant["status"] != "defined":
            raise RuntimeError(
                f"canonical level {key} is unsupported for "
                f"{request.electronic_state}/{request.spin}"
            )
        return {
            "level_or_profile": key,
            "kind": "level",
            "method": variant["method"],
            "basis": record["basis"],
            "electronic_state": request.electronic_state,
            "spin": request.spin,
            "reference": variant["reference"],
            "state_model": variant.get("state_model", "ground"),
            "frozen_core": record["frozen_core"],
        }
    if key in profile_ids:
        profile = ladder.profile(key)
        if request.electronic_state != "ground" or request.spin != "closed_shell":
            raise RuntimeError(f"canonical profile {key} is closed-shell ground-state only")
        return {
            "level_or_profile": key,
            "kind": "profile",
            "method": profile["method"],
            "basis": profile["basis"],
            "electronic_state": "ground",
            "spin": "closed_shell",
            "reference": profile["reference"],
            "state_model": "ground",
            "frozen_core": profile["frozen_core"],
        }
    raise ValueError(f"unknown canonical level or profile: {request.level_or_profile}")


def _resolve_capability(
    request: CanonicalQMRequest,
    *,
    model: dict[str, Any],
    registry_records: tuple[QMScientificCapability, ...],
    frozen_core_required: bool,
    any_ecp: bool,
) -> QMScientificCapability:
    matches: list[QMScientificCapability] = []
    for capability in registry_records:
        record = capability.payload
        if frozen_core_required:
            if not str(record["frozen_core_policy"]).startswith("mandatory_"):
                continue
        elif record["frozen_core_policy"] != "not_applicable":
            continue
        if any_ecp and record["ecp_policy"] != "native":
            continue
        key = ScientificCapabilityKey(
            backend=request.backend,
            method=str(model["method"]),
            basis=str(model["basis"]),
            electronic_state=request.electronic_state,
            spin=request.spin,
            reference=str(model["reference"]),
            state_model=str(model["state_model"]),
            charge=request.charge,
            multiplicity=request.multiplicity,
            frozen_core_policy=str(record["frozen_core_policy"]),
            ecp_policy=str(record["ecp_policy"]),
            derivative=request.derivative,
            derivative_mode=request.derivative_mode,
            accelerator=request.accelerator,
            platform=request.platform,
            require_production=not request.validation_mode,
        )
        if capability.matches(key):
            matches.append(capability)
    if not matches:
        raise RuntimeError(
            "unsupported exact canonical QM capability; no method, basis, reference, "
            "derivative, accelerator, platform, ECP, or backend substitution was attempted"
        )
    if len(matches) != 1:
        raise RuntimeError("ambiguous exact canonical QM capability")
    return matches[0]


def _resolve_basis(request: CanonicalQMRequest, *, basis: str) -> dict[str, Any]:
    if basis == "native":
        return {
            "name": basis,
            "availability": "native",
            "source": "backend_native_model",
            "bse_confirmation_required": False,
        }
    if request.basis_available is None:
        raise BasisAvailabilityUnverified(
            f"availability of {basis} for {request.backend} was not verified"
        )
    if request.basis_available is False:
        raise CanonicalBasisSetConfirmationRequired(basis=basis, backend=request.backend)
    if not request.basis_source or not request.basis_source.strip():
        raise ValueError("an available non-native basis requires explicit provenance")
    return {
        "name": basis,
        "availability": "available",
        "source": request.basis_source.strip(),
        "bse_confirmation_required": False,
    }


def _validate_request_domain(request: CanonicalQMRequest, applicability: dict[str, Any]) -> None:
    if request.application_domain not in applicability["supported_initial_domain"]:
        raise RuntimeError(
            "canonical QM hierarchy is outside its approved applicability domain; "
            "a separate explicitly approved profile is required"
        )
    if not request.single_reference_confirmed:
        raise RuntimeError("canonical QM hierarchy requires confirmed single-reference character")
    if request.electronic_state not in {"ground", "excited"}:
        raise ValueError("electronic_state must be ground or excited")
    if request.spin not in {"closed_shell", "open_shell"}:
        raise ValueError("spin must be closed_shell or open_shell")
    if request.spin == "closed_shell" and request.multiplicity != 1:
        raise ValueError("closed-shell requests require multiplicity 1")
    if request.spin == "open_shell" and request.multiplicity < 2:
        raise ValueError("open-shell requests require multiplicity at least 2")
    if request.electronic_state == "ground":
        if request.state_root not in {None, 0} or request.target_state_spin is not None:
            raise ValueError("ground-state requests cannot select an excited root or spin")
    else:
        if request.state_root is None or int(request.state_root) < 1:
            raise ValueError("excited-state requests require a positive one-based state_root")
        if request.target_state_spin not in {"singlet", "triplet"}:
            raise ValueError("excited-state requests require target_state_spin singlet or triplet")
    if len(request.atomic_numbers) != len(request.ecp_mask):
        raise ValueError("ecp_mask must have one entry per atom")
    electron_count = sum(request.atomic_numbers) - request.charge
    if electron_count < 1:
        raise ValueError("charge leaves no electrons")
    if electron_count % 2 == request.multiplicity % 2:
        raise ValueError("electron-count and multiplicity parity are inconsistent")


def _sha256(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
