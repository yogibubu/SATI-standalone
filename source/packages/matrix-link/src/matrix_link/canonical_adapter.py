"""Exact canonical-QM mappings onto the existing LINK point boundary."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from importlib import resources
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from matrix_qm.canonical_resolution import validate_resolved_canonical_qm_run
from matrix_qm.ladder import load_canonical_qm_ladder

from .modal_hessian import acquire_modal_hessian
from .protocol_manifest import LINK_PROTOCOL_ID, LINK_PROTOCOL_VERSION
from .scan import QMScanBackend


CANONICAL_BACKEND_ADAPTER_SCHEMA = "matrix.link.canonical_backend_adapters.v1"
CANONICAL_BACKEND_ADAPTER_ID = "matrix-canonical-backend-adapters-v1"
CANONICAL_BACKEND_ADAPTER_VERSION = "1.1.12"


@dataclass(frozen=True)
class CanonicalBackendAdapterRegistry:
    payload: dict[str, Any]
    sha256: str
    source: str

    def by_capability_id(self, capability_id: str) -> dict[str, Any]:
        matches = [
            record for record in self.payload["records"] if record["capability_id"] == capability_id
        ]
        if len(matches) != 1:
            raise RuntimeError(f"no unique canonical backend mapping for {capability_id}")
        return deepcopy(matches[0])


@dataclass(frozen=True)
class CanonicalAdapterPlan:
    capability_id: str
    backend: str
    engine_method: str
    engine_reference: str
    engine_route: str
    point_adapter: str
    adapter_status: str
    blockers: tuple[str, ...]
    derivative: str
    derivative_mode: str
    hessian_driver: str
    state_following: str
    executable: bool
    validation_execution: bool
    resolved_run_sha256: str
    adapter_registry_sha256: str
    backend_spec: QMScanBackend | None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "capability_id": self.capability_id,
            "backend": self.backend,
            "engine_method": self.engine_method,
            "engine_reference": self.engine_reference,
            "engine_route": self.engine_route,
            "point_adapter": self.point_adapter,
            "adapter_status": self.adapter_status,
            "blockers": list(self.blockers),
            "derivative": self.derivative,
            "derivative_mode": self.derivative_mode,
            "hessian_driver": self.hessian_driver,
            "state_following": self.state_following,
            "executable": self.executable,
            "validation_execution": self.validation_execution,
            "resolved_run_sha256": self.resolved_run_sha256,
            "adapter_registry_sha256": self.adapter_registry_sha256,
        }
        if self.backend_spec is not None:
            payload["backend_spec"] = {
                "name": self.backend_spec.name,
                "route": self.backend_spec.route,
                "method": self.backend_spec.method,
                "reference": self.backend_spec.reference,
                "basis": self.backend_spec.basis,
                "dispersion_contract": self.backend_spec.dispersion_contract,
                "charge": self.backend_spec.charge,
                "multiplicity": self.backend_spec.multiplicity,
                "gradient_mode": self.backend_spec.gradient_mode,
                "electronic_state": self.backend_spec.electronic_state,
                "state_spin": self.backend_spec.state_spin,
                "freeze_core": self.backend_spec.freeze_core,
                "properties": list(self.backend_spec.properties),
                "scf_convergence": (
                    None
                    if self.backend_spec.scf_convergence is None
                    else dict(self.backend_spec.scf_convergence)
                ),
                "restart_reuse_for_displacements": (
                    self.backend_spec.restart_reuse_for_displacements
                ),
            }
        return payload


def load_canonical_backend_adapter_registry() -> CanonicalBackendAdapterRegistry:
    resource = resources.files("matrix_link").joinpath("data/canonical_backend_adapters_v1.json")
    raw = resource.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("canonical backend adapter registry is unreadable") from exc
    validate_canonical_backend_adapter_registry(payload)
    return CanonicalBackendAdapterRegistry(
        payload=deepcopy(payload),
        sha256=hashlib.sha256(raw).hexdigest(),
        source="matrix_link:data/canonical_backend_adapters_v1.json",
    )


def canonical_backend_adapter_schema_path() -> Path:
    return Path(
        str(
            resources.files("matrix_link").joinpath(
                "data/canonical_backend_adapters_v1.schema.json"
            )
        )
    )


def validate_canonical_backend_adapter_registry(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != CANONICAL_BACKEND_ADAPTER_SCHEMA:
        raise RuntimeError("unsupported canonical backend adapter schema")
    if (
        payload.get("registry_id") != CANONICAL_BACKEND_ADAPTER_ID
        or payload.get("manifest_version") != CANONICAL_BACKEND_ADAPTER_VERSION
        or payload.get("status") != "approved_validation"
    ):
        raise RuntimeError("canonical backend adapter registry identity changed")
    if payload.get("execution_contract") != {
        "point_result": "oracle.link.point_result.v1",
        "optimizer": f"{LINK_PROTOCOL_ID}@{LINK_PROTOCOL_VERSION}",
        "modal_hessian": "matrix.link.modal_hessian_result.v2",
        "state_identity_owner": "APOC",
        "frozen_core_owner": "MATRIX_QM",
        "external_qm_source_modified": False,
    }:
        raise RuntimeError("canonical backend execution boundary changed")
    if payload.get("governance") != {
        "mapping_key": "exact_capability_id",
        "uncertified_execution": "explicit_validation_only",
        "missing_mapping": "fail_closed",
        "molecule_specific_patches": False,
        "backend_specific_optimizer_patches": False,
        "change_policy": "new_manifest_version_and_explicit_approval",
    }:
        raise RuntimeError("canonical backend adapter governance changed")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise RuntimeError("canonical backend adapter registry has no records")
    ids: set[str] = set()
    required = {
        "capability_id",
        "backend",
        "engine_method",
        "engine_reference",
        "point_adapter",
        "status",
        "blockers",
    }
    for record in records:
        if not isinstance(record, Mapping) or not set(record).issubset(
            required | {"engine_route", "scf_convergence", "restart_policy"}
        ) or not required.issubset(record):
            raise RuntimeError("canonical backend mapping has incomplete or unknown fields")
        if "engine_route" in record and not str(record["engine_route"]).strip():
            raise RuntimeError("canonical backend mapping has an empty engine route")
        if "scf_convergence" in record:
            if record["backend"] != "orca":
                raise RuntimeError("explicit SCF convergence is currently an ORCA contract")
            scf = record["scf_convergence"]
            if not isinstance(scf, Mapping) or set(scf) != {
                "energy_tolerance",
                "orbital_gradient_tolerance",
                "diis_error_tolerance",
            }:
                raise RuntimeError("canonical backend mapping has an invalid SCF contract")
            if any(float(value) <= 0.0 for value in scf.values()):
                raise RuntimeError("canonical SCF tolerances must be positive")
        if record.get("restart_policy", "central_point_orbitals") not in {
            "central_point_orbitals",
            "independent_points",
        }:
            raise RuntimeError("canonical backend mapping has an invalid restart policy")
        identifier = str(record["capability_id"])
        if not identifier or identifier in ids:
            raise RuntimeError("canonical backend mapping identifiers must be unique")
        ids.add(identifier)
        if record["status"] not in {
            "implemented_certified",
            "implemented_pending_certification",
            "mapping_only",
        }:
            raise RuntimeError("unknown canonical backend adapter status")
        if record["status"] == "implemented_certified" and record["blockers"]:
            raise RuntimeError("a certified canonical adapter cannot retain blockers")

    from matrix_qm.capabilities import load_qm_capability_registry_v2

    capabilities = load_qm_capability_registry_v2()
    capability_ids = {item.capability_id for item in capabilities.records}
    if ids != capability_ids:
        raise RuntimeError("canonical backend mappings and exact capabilities differ")
    for record in records:
        capability = capabilities.by_id(str(record["capability_id"]))
        if record["backend"] != capability.backend:
            raise RuntimeError("canonical backend mapping changes the selected backend")
        if record["engine_reference"] != capability.payload[
            "reference"
        ] and capability.capability_id not in {
            "et.eomccsd.def2tzvp.rhf.fc.v1",
            "et.cc3.def2tzvpp.rhf.v1",
        }:
            raise RuntimeError("canonical adapter reference mapping changed")


def plan_canonical_backend_adapter(
    resolved_run: Mapping[str, Any],
    *,
    allow_uncertified_validation: bool = False,
    executable: str | None = None,
    basis_file: str | Path | None = None,
    timeout: float | None = None,
    env: Mapping[str, str] | None = None,
    extra_args: Sequence[str] = (),
    processors: int = 1,
    memory_gb: int | None = None,
) -> CanonicalAdapterPlan:
    run = deepcopy(dict(resolved_run))
    validate_resolved_canonical_qm_run(run)
    if tuple(extra_args):
        raise ValueError(
            "canonical QM adapters reject free-form backend arguments; add any "
            "scientific setting to a new approved manifest version"
        )
    registry = load_canonical_backend_adapter_registry()
    capability = run["capability"]["record"]
    mapping = registry.by_capability_id(str(capability["id"]))
    status = str(mapping["status"])
    validation_run = bool(run["request"]["validation_mode"])
    executable_mapping = status == "implemented_certified" or (
        status == "implemented_pending_certification"
        and allow_uncertified_validation
        and validation_run
    )
    derivative = str(run["request"]["derivative"])
    derivative_mode = str(run["request"]["derivative_mode"])
    if derivative_mode in {"analytic", "composed_analytic"}:
        gradient_mode = "analytic"
    elif derivative_mode == "link_numerical_energy":
        gradient_mode = "numerical"
    elif derivative_mode == "unavailable":
        gradient_mode = "energy"
    else:
        gradient_mode = "analytic"
    hessian_driver = (
        "LINK_modal_hessian"
        if derivative == "hessian"
        and derivative_mode == "modal_finite_difference_analytic_gradient"
        else ("backend" if derivative == "hessian" else "none")
    )
    state_following = (
        "strict_APOC_root_expansion_then_displacement_halving"
        if run["request"]["electronic_state"] == "excited"
        else "not_applicable"
    )
    backend_spec = None
    if executable_mapping:
        electronic_state = int(run["request"].get("state_root") or 0)
        state_spin = str(run["request"].get("target_state_spin") or "singlet")
        if derivative == "hessian" and derivative_mode == "analytic":
            properties = ("energy", "gradient", "hessian")
        elif derivative in {"gradient", "hessian"} and derivative_mode in {
            "analytic",
            "composed_analytic",
            "modal_finite_difference_analytic_gradient",
        }:
            properties = ("energy", "gradient")
        else:
            properties = ("energy",)
        route = str(
            mapping.get(
                "engine_route",
                "--gfn 2" if mapping["backend"] == "xtb" else "",
            )
        )
        resolved_basis_file = (
            None if basis_file is None else Path(basis_file).expanduser().resolve()
        )
        if resolved_basis_file is not None and not resolved_basis_file.is_file():
            raise FileNotFoundError(
                f"canonical basis artifact does not exist: {resolved_basis_file}"
            )
        adapter_extra_args: tuple[str, ...] = ()
        if mapping["backend"] == "et" and resolved_basis_file is not None:
            adapter_extra_args = ("-basis", str(resolved_basis_file.parent))
        level_or_profile = str(run["request"]["level_or_profile"])
        dispersion_contract = None
        if str(run["scientific_model"]["kind"]) == "level":
            dispersion_contract = load_canonical_qm_ladder().level(level_or_profile).get(
                "dispersion_contract"
            )
        backend_spec = QMScanBackend(
            name=str(mapping["backend"]),
            route=route,
            method=str(mapping["engine_method"]),
            reference=str(mapping["engine_reference"]),
            basis=str(run["scientific_model"]["basis"]),
            dispersion_contract=(
                None if dispersion_contract is None else str(dispersion_contract)
            ),
            basis_file=resolved_basis_file,
            charge=int(run["request"]["charge"]),
            multiplicity=int(run["request"]["multiplicity"]),
            executable=executable,
            timeout=timeout,
            env=None if env is None else dict(env),
            extra_args=adapter_extra_args,
            processors=max(1, int(processors)),
            memory_gb=memory_gb,
            resolution={
                "schema": run["schema"],
                "manifest_sha256": run["manifest_sha256"],
                "capability_id": capability["id"],
                "adapter_registry_sha256": registry.sha256,
            },
            properties=properties,
            gradient_mode=gradient_mode,
            electronic_state=electronic_state,
            excited_states=(max(6, electronic_state + 3) if electronic_state > 0 else None),
            state_spin=state_spin,
            freeze_core=str(run["capability"]["record"]["frozen_core_policy"]).startswith(
                "mandatory_"
            ),
            state_tracking="apoc",
            scf_convergence=(
                None
                if mapping.get("scf_convergence") is None
                else {
                    str(key): float(value)
                    for key, value in mapping["scf_convergence"].items()
                }
            ),
            restart_reuse_for_displacements=(
                mapping.get("restart_policy", "central_point_orbitals")
                == "central_point_orbitals"
            ),
        )
    return CanonicalAdapterPlan(
        capability_id=str(capability["id"]),
        backend=str(mapping["backend"]),
        engine_method=str(mapping["engine_method"]),
        engine_reference=str(mapping["engine_reference"]),
        engine_route=str(mapping.get("engine_route", "")),
        point_adapter=str(mapping["point_adapter"]),
        adapter_status=status,
        blockers=tuple(str(item) for item in mapping["blockers"]),
        derivative=derivative,
        derivative_mode=derivative_mode,
        hessian_driver=hessian_driver,
        state_following=state_following,
        executable=bool(executable_mapping),
        validation_execution=bool(executable_mapping and status != "implemented_certified"),
        resolved_run_sha256=str(run["manifest_sha256"]),
        adapter_registry_sha256=registry.sha256,
        backend_spec=backend_spec,
    )


def acquire_canonical_modal_hessian(
    xyzin_path,
    *,
    run_dir,
    resolved_run: Mapping[str, Any],
    curvatures_hartree_per_q2,
    curvature_floor_hartree_per_q2: float,
    allow_uncertified_validation: bool = False,
    workers: int = 1,
    backend_processors: int | None = None,
    **adapter_runtime: Any,
) -> dict[str, object]:
    """Use the one existing LINK modal-Hessian engine for a canonical run."""

    plan = plan_canonical_backend_adapter(
        resolved_run,
        allow_uncertified_validation=allow_uncertified_validation,
        processors=(workers if backend_processors is None else backend_processors),
        **adapter_runtime,
    )
    if not plan.executable or plan.backend_spec is None:
        raise RuntimeError(
            f"canonical adapter {plan.capability_id} is not executable: " + ", ".join(plan.blockers)
        )
    if plan.hessian_driver != "LINK_modal_hessian":
        raise RuntimeError("resolved canonical run does not request a LINK modal Hessian")
    return acquire_modal_hessian(
        xyzin_path,
        run_dir=run_dir,
        backend=plan.backend_spec,
        workers=workers,
        property_source="cartesian-gradient",
        curvatures_hartree_per_q2=curvatures_hartree_per_q2,
        curvature_floor_hartree_per_q2=curvature_floor_hartree_per_q2,
        curvature_source="resolved_canonical_lower_level_hessian",
        verify_step_convergence=True,
        totally_symmetric_only=True,
    )


__all__ = [
    "CANONICAL_BACKEND_ADAPTER_ID",
    "CANONICAL_BACKEND_ADAPTER_SCHEMA",
    "CANONICAL_BACKEND_ADAPTER_VERSION",
    "CanonicalAdapterPlan",
    "CanonicalBackendAdapterRegistry",
    "acquire_canonical_modal_hessian",
    "canonical_backend_adapter_schema_path",
    "load_canonical_backend_adapter_registry",
    "plan_canonical_backend_adapter",
    "validate_canonical_backend_adapter_registry",
]
