"""Backend selection and Basis Set Exchange acquisition for Keymaker.

The module is deliberately independent of the GUI.  It turns an explicit or
automatic backend request into an auditable recommendation and materializes a
named orbital basis in the serializer understood by that backend.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib.resources import files
import json
from pathlib import Path
import re
from typing import Callable, Iterable
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from matrix_core.availability import discover_execution_combinations
from matrix_core.atomic_io import atomic_write_bytes
from matrix_core.environment import MatrixEnvironment


QM_SELECTION_SCHEMA = "matrix.keymaker.qm_selection.v1"
BASIS_ARTIFACT_SCHEMA = "matrix.keymaker.basis_set.v1"
QM_CAPABILITY_REGISTRY_SCHEMA = "matrix.keymaker.qm_capability_registry.v1"
BSE_BASE_URL = "https://www.basissetexchange.org"
DERIVATIVE_POLICIES = ("analytic", "prefer-analytic", "allow-numerical")
PACKAGED_BASIS_REGISTRY_SCHEMA = "matrix.basis_set_registry.v1"

# Canonical Gaussian-compatible L1 contract shared by Keymaker/LINK and the
# ORACLE PL1 campaign.  The packaged 3F12- Gen/GenECP file is complete, so
# every consumer must keep Gen and Pseudo=Read together.
GDV_L1_METHOD = "revDSDPBEP86D3"
GDV_L1_BASIS = "3F12-"
GDV_L1_ROUTE = "#p revDSDPBEP86D3 Gen Pseudo=Read Opt output=pickett"
GDV_L1_CHECKPOINT_ROUTE = "#p revDSDPBEP86D3 Gen Pseudo=Read Opt=ReadFC Guess=Read Geom=Checkpoint"


@dataclass(frozen=True)
class QMBackendCapability:
    key: str
    label: str
    method_families: tuple[str, ...]
    analytic_gradient_families: tuple[str, ...]
    bse_format: str | None
    efficiency_rank: int
    note: str = ""
    energy_methods: tuple[str, ...] = ()
    analytic_gradient_methods: tuple[str, ...] = ()
    analytic_hessian_families: tuple[str, ...] = ()
    analytic_hessian_methods: tuple[str, ...] = ()
    numerical_gradients: bool = True
    numerical_hessians: bool = True
    basis_contract: str = "bse"
    ecp_contract: str = "unverified"
    license_model: str = "external"
    platforms: tuple[str, ...] = ("linux", "macos")
    accelerators: tuple[str, ...] = ("cpu",)
    suggestible: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


BACKEND_CAPABILITIES: tuple[QMBackendCapability, ...] = (
    QMBackendCapability(
        "turbomole", "TURBOMOLE", ("hf", "dft"), ("hf", "dft"),
        "turbomole", 33, energy_methods=("CC2", "ADC(2)"),
        analytic_gradient_methods=("CC2", "ADC(2)"),
        basis_contract="native-name", ecp_contract="unverified", license_model="commercial",
        suggestible=False,
        note="LINK control-template adapter; external numerical qualification required before automatic selection.",
    ),
    QMBackendCapability(
        "gdv", "Gaussian development version", ("hf", "dft", "mp2"),
        ("hf", "dft", "mp2"), "gaussian94", 10,
        analytic_hessian_families=("hf", "dft", "mp2"), ecp_contract="native",
        license_model="institutional",
    ),
    QMBackendCapability(
        "g16", "Gaussian 16", ("hf", "dft", "mp2"),
        ("hf", "dft", "mp2"), "gaussian94", 20,
        analytic_hessian_families=("hf", "dft", "mp2"), ecp_contract="native",
        license_model="commercial",
    ),
    QMBackendCapability(
        "gaussian", "Gaussian-compatible alias", ("hf", "dft", "mp2"),
        ("hf", "dft", "mp2"), "gaussian94", 21,
        analytic_hessian_families=("hf", "dft", "mp2"), ecp_contract="native",
        license_model="external", suggestible=False,
        note="Compatibility alias resolved to GDV or Gaussian 16; never ranked separately.",
    ),
    QMBackendCapability(
        "orca", "ORCA", ("hf", "dft", "mp2"), ("hf", "dft", "mp2"),
        "orca", 30, energy_methods=("CCSD", "CCSD(T)", "DLPNO-CCSD", "DLPNO-CCSD(T)"),
        analytic_hessian_families=("hf", "dft"), ecp_contract="native",
        license_model="academic",
    ),
    QMBackendCapability(
        "nwchem", "NWChem", ("hf", "dft", "mp2"), ("hf", "dft", "mp2"),
        "nwchem", 31, analytic_hessian_families=(), ecp_contract="native",
        license_model="open-source",
        note="Analytic energy/gradient point calculator; geometry optimization is owned by LINK.",
    ),
    QMBackendCapability(
        "pyscf", "PySCF", ("hf", "dft", "mp2"), ("hf", "dft"),
        "psi4", 35, "BSE text is acquired through its Psi4-compatible serializer.",
        energy_methods=("CCSD", "CCSD(T)"), analytic_hessian_families=("hf", "dft"),
        ecp_contract="native", license_model="open-source",
    ),
    QMBackendCapability(
        "pyfock", "PyFock", ("hf", "dft"), ("dft",), "psi4", 36,
        "In-process PyFock DFT provider; analytic LDA/GGA/meta-GGA gradients are supplied by PyFock.",
        energy_methods=("HF", "PBE", "R2SCAN"),
        analytic_gradient_methods=("PBE", "R2SCAN"),
        numerical_gradients=True, numerical_hessians=True,
        basis_contract="native-name", ecp_contract="unverified", license_model="open-source",
        suggestible=False,
    ),
    QMBackendCapability(
        "psi4", "Psi4", ("hf", "dft", "mp2"), ("hf", "dft", "mp2"),
        "psi4", 34, energy_methods=("CCSD", "CCSD(T)"),
        analytic_hessian_families=("hf", "dft"), ecp_contract="native",
        license_model="open-source", suggestible=False,
        note="Detected by matrix-build, but the executable MATRIX point adapter is not yet validated.",
    ),
    QMBackendCapability(
        "et", "eT", (), (), "gaussian94", 5,
        "The verified MATRIX/eT contract has analytical gradients only for HF and CCSD; "
        "its external Libint basis path has no ECP contract.",
        energy_methods=("HF", "CC2", "CCSD", "CC3", "CCSD(T)", "CCSDT"),
        analytic_gradient_methods=("HF", "CCSD"), basis_contract="bse-normalized",
        ecp_contract="none", license_model="open-source",
    ),
    QMBackendCapability(
        "molpro", "Molpro", ("hf", "dft", "mp2", "multireference"),
        ("hf", "dft", "mp2", "multireference"), "molpro", 40,
        energy_methods=("CC2", "CCSD", "CCSD(T)", "CC3"),
        analytic_gradient_methods=("CC2", "CCSD", "CCSD(T)", "CC3"),
        ecp_contract="native", license_model="commercial",
    ),
    QMBackendCapability(
        "mrcc", "MRCC", ("hf", "dft", "mp2"), ("hf", "dft"), None, 45,
        "BSE does not advertise an MRCC serializer; MATRIX preserves its native named-basis contract.",
        energy_methods=("CC2", "CCSD", "CCSD(T)", "CC3", "CCSDT", "CCSDT(Q)"),
        basis_contract="native-name", ecp_contract="native", license_model="academic",
    ),
    QMBackendCapability(
        "cfour", "CFOUR", ("hf", "mp2"), ("hf", "mp2"), "cfour", 42,
        energy_methods=("CC2", "CCSD", "CCSD(T)", "CC3", "CCSDT"),
        analytic_gradient_methods=("CC2", "CCSD", "CCSD(T)", "CC3"),
        ecp_contract="unverified", license_model="academic",
    ),
    QMBackendCapability(
        "xtb", "xTB", ("xtb",), ("xtb",), None, 1,
        "xTB does not use a user-selected Gaussian orbital basis.",
        analytic_hessian_families=("xtb",), basis_contract="fixed",
        ecp_contract="not-applicable", license_model="open-source",
    ),
)


@dataclass(frozen=True)
class QMMethodProfile:
    key: str
    family: str
    accuracy_rank: int
    label: str


METHOD_PROFILES: tuple[QMMethodProfile, ...] = (
    QMMethodProfile("GFN2-XTB", "xtb", 5, "GFN2-xTB"),
    QMMethodProfile("HF", "hf", 10, "Hartree--Fock"),
    QMMethodProfile("B3LYP", "dft", 20, "B3LYP"),
    QMMethodProfile("PBE0", "dft", 22, "PBE0"),
    QMMethodProfile("MP2", "mp2", 30, "MP2"),
    QMMethodProfile("CC2", "coupled-cluster", 34, "CC2"),
    QMMethodProfile("CCSD", "coupled-cluster", 40, "CCSD"),
    QMMethodProfile("CCSD(T)", "coupled-cluster", 50, "CCSD(T)"),
    QMMethodProfile("CC3", "coupled-cluster", 55, "CC3"),
    QMMethodProfile("CCSDT", "coupled-cluster", 60, "CCSDT"),
    QMMethodProfile("CASSCF", "multireference", 45, "CASSCF"),
)


@dataclass(frozen=True)
class BackendCandidate:
    backend: str
    available: bool
    compatible: bool
    analytic_gradient: bool
    efficiency_rank: int
    reason: str
    analytic_hessian: bool = False
    derivative_support: str = "unavailable"
    execution_locations: tuple[str, ...] = ()
    basis_contract: str = ""
    ecp_contract: str = ""
    license_model: str = ""
    platforms: tuple[str, ...] = ()
    accelerators: tuple[str, ...] = ()
    suggestible: bool = True


@dataclass(frozen=True)
class MethodAlternative:
    method: str
    backend: str
    derivative_support: str
    accuracy_distance: int
    reason: str


@dataclass(frozen=True)
class BackendRecommendation:
    backend: str
    method: str
    method_family: str
    derivative: str
    explicit: bool
    reason: str
    candidates: tuple[BackendCandidate, ...]
    status: str = "exact"
    requested_method: str = ""
    derivative_policy: str = "prefer-analytic"
    ecp_required: bool = False
    alternatives: tuple[MethodAlternative, ...] = ()
    requires_confirmation: bool = False
    schema: str = QM_SELECTION_SCHEMA

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class InitializationHFProvider:
    """Deterministic open-source provider for the HF/STO-3G fallback."""

    backend: str
    executable: str
    method: str = "HF"
    basis: str = "STO-3G"
    derivative: str = "analytic_hessian"
    license_model: str = "open-source"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BasisSetRequest:
    name: str
    backend: str
    elements: tuple[str, ...] = ()
    version: str | None = None
    header: bool = True


@dataclass(frozen=True)
class BasisSetArtifact:
    name: str
    backend: str
    bse_format: str
    elements: tuple[str, ...]
    version: str | None
    source_url: str
    content_path: str
    manifest_path: str
    sha256: str
    retrieved_utc: str
    reused: bool = False
    normalization: str = "none"
    schema: str = BASIS_ARTIFACT_SCHEMA

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class BasisSetConfirmationRequired(PermissionError):
    """Raised when a missing basis requires explicit user confirmation."""


def method_family(method: str) -> str:
    normalized = re.sub(r"[\s_-]+", "", str(method).strip().casefold())
    if normalized.startswith(("gfn", "xtb")):
        return "xtb"
    if normalized in {"hf", "rhf", "uhf", "rohf"}:
        return "hf"
    if normalized.startswith(("cc", "dlpnocc", "eomcc")):
        return "coupled-cluster"
    if normalized.startswith(("casscf", "caspt", "nevpt", "mrci")):
        return "multireference"
    if normalized.startswith("mp") or normalized.startswith("scsmp"):
        return "mp2"
    return "dft"


def backend_capability(key: str) -> QMBackendCapability:
    normalized = str(key).strip().casefold()
    for capability in BACKEND_CAPABILITIES:
        if capability.key == normalized:
            return capability
    raise ValueError(f"unknown MATRIX QM backend: {key}")


def backend_capability_records() -> tuple[dict[str, object], ...]:
    """Return the single, serializable capability registry consumed by Keymaker."""

    return tuple(capability.to_dict() for capability in BACKEND_CAPABILITIES)


def backend_capability_registry() -> dict[str, object]:
    return {
        "schema": QM_CAPABILITY_REGISTRY_SCHEMA,
        "backends": list(backend_capability_records()),
    }


def qm_capability_registry_schema_path() -> Path:
    return Path(
        str(files("matrix_qm").joinpath("schemas/qm-capability-registry-v1.schema.json"))
    )


def qm_selection_schema_path() -> Path:
    return Path(str(files("matrix_qm").joinpath("schemas/qm-selection-v1.schema.json")))


def packaged_basis_set_registry() -> dict[str, object]:
    """Return the versioned registry of non-BSE basis sets shipped with MATRIX."""

    resource = files("matrix_qm").joinpath("basis_sets/registry.json")
    payload = json.loads(resource.read_text(encoding="utf-8"))
    if payload.get("schema") != PACKAGED_BASIS_REGISTRY_SCHEMA:
        raise ValueError("invalid MATRIX packaged-basis registry schema")
    return payload


def packaged_basis_set_path(name: str) -> Path:
    """Resolve a packaged basis by canonical name or alias.

    The returned resource is shared by every MATRIX adapter.  Callers may
    reference it with a backend include directive or embed its exact text in a
    frozen input.
    """

    requested = str(name).strip().casefold()
    registry = packaged_basis_set_registry()
    for record in registry.get("basis_sets", []):
        aliases = {
            str(record.get("name", "")).strip().casefold(),
            *(str(alias).strip().casefold() for alias in record.get("aliases", [])),
        }
        if requested not in aliases:
            continue
        resource = files("matrix_qm").joinpath("basis_sets", str(record["file"]))
        path = Path(str(resource))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record.get("sha256"):
            raise ValueError(f"packaged basis checksum mismatch for {record['name']}")
        return path
    raise ValueError(f"unknown MATRIX packaged basis set: {name}")


def _method_key(method: str) -> str:
    return re.sub(r"[\s_-]+", "", str(method).strip().upper())


def _supports_method(capability: QMBackendCapability, method: str) -> bool:
    key = _method_key(method)
    exact = {_method_key(item) for item in capability.energy_methods}
    return key in exact or method_family(method) in capability.method_families


def _analytic_gradient(capability: QMBackendCapability, method: str) -> bool:
    key = _method_key(method)
    exact = {_method_key(item) for item in capability.analytic_gradient_methods}
    return key in exact or method_family(method) in capability.analytic_gradient_families


def _analytic_hessian(capability: QMBackendCapability, method: str) -> bool:
    key = _method_key(method)
    exact = {_method_key(item) for item in capability.analytic_hessian_methods}
    return key in exact or method_family(method) in capability.analytic_hessian_families


def _derivative_support(
    capability: QMBackendCapability, method: str, derivative: str
) -> str:
    if not _supports_method(capability, method):
        return "unavailable"
    if derivative == "energy":
        return "analytic"
    if derivative == "gradient":
        if _analytic_gradient(capability, method):
            return "analytic"
        return "numerical" if capability.numerical_gradients else "unavailable"
    if _analytic_hessian(capability, method):
        return "analytic"
    return "numerical" if capability.numerical_hessians else "unavailable"


def _support_allowed(support: str, policy: str) -> bool:
    if support == "unavailable":
        return False
    return support == "analytic" if policy == "analytic" else True


def _available_locations(environment: MatrixEnvironment) -> dict[str, tuple[str, ...]]:
    inventory = discover_execution_combinations(environment, include_resident=False)
    return {
        backend: tuple(
            sorted(
                "local" if item.kind == "local" else f"remote:{item.machine}"
                for item in inventory.combinations_for(backend)
            )
        )
        for backend in inventory.backends
    }


def _profile_for_method(method: str) -> QMMethodProfile | None:
    key = _method_key(method)
    return next((item for item in METHOD_PROFILES if _method_key(item.key) == key), None)


def _method_alternatives(
    *,
    method: str,
    derivative: str,
    policy: str,
    requested_backend: str,
    require_ecp: bool,
    locations: dict[str, tuple[str, ...]],
) -> tuple[MethodAlternative, ...]:
    requested_profile = _profile_for_method(method)
    requested_rank = requested_profile.accuracy_rank if requested_profile else 0
    requested_family = method_family(method)
    records: list[tuple[tuple[int, int, int, int, int, str], MethodAlternative]] = []
    methods = (QMMethodProfile(method, requested_family, requested_rank, method), *METHOD_PROFILES)
    seen: set[tuple[str, str]] = set()
    for profile in methods:
        for capability in BACKEND_CAPABILITIES:
            key = (profile.key, capability.key)
            if key in seen or not capability.suggestible or capability.key not in locations:
                continue
            seen.add(key)
            if require_ecp and capability.ecp_contract != "native":
                continue
            support = _derivative_support(capability, profile.key, derivative)
            if not _support_allowed(support, policy):
                continue
            same_method = _method_key(profile.key) == _method_key(method)
            same_backend = capability.key == requested_backend
            family_penalty = 0 if profile.family == requested_family else 1
            distance = abs(profile.accuracy_rank - requested_rank)
            if requested_profile is None and not same_method:
                distance = profile.accuracy_rank
            if same_method:
                reason = (
                    f"{capability.label} preserves {method} and supplies the requested "
                    f"{derivative} as {support}."
                )
            else:
                reason = (
                    f"{profile.label} on {capability.label} supplies the requested {derivative} "
                    f"as {support}; it changes the method and therefore requires confirmation."
                )
            sort_key = (
                0 if same_method else 1,
                0 if same_backend else 1,
                family_penalty,
                distance,
                capability.efficiency_rank,
                capability.key,
            )
            records.append(
                (
                    sort_key,
                    MethodAlternative(profile.key, capability.key, support, distance, reason),
                )
            )
    records.sort(key=lambda item: item[0])
    return tuple(item[1] for item in records[:5])


def recommend_qm_backend(
    environment: MatrixEnvironment,
    *,
    method: str,
    derivative: str = "gradient",
    requested_backend: str = "auto",
    derivative_policy: str = "prefer-analytic",
    require_ecp: bool = False,
) -> BackendRecommendation:
    """Resolve a verified method/backend/derivative combination.

    Exact combinations are selected automatically.  If the request cannot be
    fulfilled, the return value contains ranked alternatives and sets
    ``requires_confirmation``; callers must never execute that proposal as if
    it were the original request.
    """

    family = method_family(method)
    derivative_key = str(derivative).strip().casefold()
    if derivative_key not in {"energy", "gradient", "hessian"}:
        raise ValueError("QM derivative must be energy, gradient or hessian")
    policy = str(derivative_policy).strip().casefold()
    if policy not in DERIVATIVE_POLICIES:
        raise ValueError(f"unknown derivative policy: {derivative_policy}")
    requested = str(requested_backend).strip().casefold() or "auto"
    if requested != "auto":
        backend_capability(requested)
    locations = _available_locations(environment)
    candidates: list[BackendCandidate] = []
    for capability in BACKEND_CAPABILITIES:
        compatible = _supports_method(capability, method)
        analytic_gradient = _analytic_gradient(capability, method)
        analytic_hessian = _analytic_hessian(capability, method)
        support = _derivative_support(capability, method, derivative_key)
        is_available = capability.key in locations
        ecp_ok = not require_ecp or capability.ecp_contract == "native"
        reasons: list[str] = []
        reasons.append("configured" if is_available else "not configured on an enabled machine")
        reasons.append(
            "method supported by the verified MATRIX adapter"
            if compatible else "method not present in the verified MATRIX adapter contract"
        )
        reasons.append(f"{derivative_key} support: {support}")
        if require_ecp:
            reasons.append("ECP contract verified" if ecp_ok else "no verified ECP contract")
        numerical_penalty = 100 if support == "numerical" and policy == "prefer-analytic" else 0
        candidates.append(
            BackendCandidate(
                capability.key,
                is_available,
                compatible,
                analytic_gradient,
                capability.efficiency_rank + numerical_penalty,
                "; ".join(reasons),
                analytic_hessian=analytic_hessian,
                derivative_support=support,
                execution_locations=locations.get(capability.key, ()),
                basis_contract=capability.basis_contract,
                ecp_contract=capability.ecp_contract,
                license_model=capability.license_model,
                platforms=capability.platforms,
                accelerators=capability.accelerators,
                suggestible=capability.suggestible,
            )
        )
    eligible = [
        item for item in candidates
        if item.available
        and item.compatible
        and _support_allowed(item.derivative_support, policy)
        and (not require_ecp or item.ecp_contract == "native")
        and backend_capability(item.backend).suggestible
    ]
    if requested != "auto":
        eligible = [item for item in eligible if item.backend == requested]
    if not eligible:
        alternatives = _method_alternatives(
            method=method,
            derivative=derivative_key,
            policy=policy,
            requested_backend=requested,
            require_ecp=require_ecp,
            locations=locations,
        )
        if not alternatives:
            raise RuntimeError(
                f"no configured MATRIX QM backend can supply {method}/{derivative_key} "
                f"under the {policy!r} derivative policy"
            )
        proposal = alternatives[0]
        reason = (
            f"The requested combination {method}/{derivative_key}"
            f"{f' on {requested}' if requested != 'auto' else ''} is unavailable. "
            f"Keymaker proposes {proposal.method} on {proposal.backend}; accepting a different "
            "method or backend requires explicit scientific confirmation."
        )
        return BackendRecommendation(
            proposal.backend, proposal.method, method_family(proposal.method), derivative_key,
            requested != "auto", reason, tuple(candidates), status="alternative",
            requested_method=method, derivative_policy=policy, ecp_required=require_ecp,
            alternatives=alternatives, requires_confirmation=True,
        )
    selected = min(eligible, key=lambda item: (item.efficiency_rank, item.backend))
    derivative_alternatives: tuple[MethodAlternative, ...] = ()
    if selected.derivative_support == "numerical" and policy == "prefer-analytic":
        derivative_alternatives = _method_alternatives(
            method=method,
            derivative=derivative_key,
            policy="analytic",
            requested_backend=requested,
            require_ecp=require_ecp,
            locations=locations,
        )
    if requested != "auto":
        reason = (
            f"The user selected {requested}; Keymaker verified the complete "
            f"{method}/{derivative_key} combination ({selected.derivative_support})."
        )
    else:
        reason = (
            f"Keymaker selected {selected.backend}: it is configured, supports {method}, "
            f"supplies the requested {derivative_key} as {selected.derivative_support}, and "
            "has the best current MATRIX rank among exact eligible combinations."
        )
    if derivative_alternatives:
        first = derivative_alternatives[0]
        reason += (
            f" An analytical alternative is {first.method} on {first.backend}; "
            "changing to it would require explicit confirmation."
        )
    return BackendRecommendation(
        selected.backend, method, family, derivative_key, requested != "auto", reason,
        tuple(candidates), requested_method=method, derivative_policy=policy,
        ecp_required=require_ecp, alternatives=derivative_alternatives,
    )


def resolve_open_hf_sto3g_provider(
    environment: MatrixEnvironment,
    *,
    requested_backend: str = "auto",
) -> InitializationHFProvider:
    """Resolve the frozen no-xTB HF/STO-3G Hessian provider locally.

    Only a configured, enabled, open-source adapter with a verified analytic
    HF Hessian is eligible.  The complete initialization workflow is already
    running on the selected machine, so remote-only inventory entries are not
    silently selected here.
    """

    requested = str(requested_backend).strip().casefold() or "auto"
    candidates: list[tuple[int, str, str]] = []
    for capability in BACKEND_CAPABILITIES:
        if capability.license_model != "open-source" or not capability.suggestible:
            continue
        if not _supports_method(capability, "HF") or not _analytic_hessian(
            capability, "HF"
        ):
            continue
        if requested != "auto" and capability.key != requested:
            continue
        program = environment.program(capability.key)
        if program is None or not program.enabled or not program.available:
            continue
        candidates.append(
            (capability.efficiency_rank, capability.key, str(program.executable))
        )
    if not candidates:
        qualifier = "" if requested == "auto" else f" {requested!r}"
        raise RuntimeError(
            "no configured local open-source MATRIX backend"
            f"{qualifier} provides a verified analytic HF/STO-3G Hessian"
        )
    _rank, backend, executable = min(candidates)
    return InitializationHFProvider(backend=backend, executable=executable)


def basis_format_for_backend(backend: str) -> str:
    capability = backend_capability(backend)
    if capability.bse_format is None:
        raise ValueError(f"{capability.label} has no automatic Basis Set Exchange serializer: {capability.note}")
    return capability.bse_format


def basis_set_url(request: BasisSetRequest, *, base_url: str = BSE_BASE_URL) -> str:
    fmt = basis_format_for_backend(request.backend)
    query: dict[str, str] = {"header": "true" if request.header else "false"}
    if request.elements:
        query["elements"] = ",".join(_normalize_elements(request.elements))
    if request.version is not None:
        query["version"] = str(request.version)
    return (
        f"{base_url.rstrip('/')}/api/basis/{quote(request.name.strip(), safe='')}/"
        f"format/{quote(fmt, safe='')}/?{urlencode(query)}"
    )


def basis_set_availability(request: BasisSetRequest, directory: Path | str) -> dict[str, object]:
    """Describe local/cache availability without contacting Basis Set Exchange."""

    backend = request.backend.strip().casefold()
    directory_path = Path(directory).expanduser().resolve()
    try:
        fmt = basis_format_for_backend(backend)
    except ValueError as exc:
        return {
            "schema": BASIS_ARTIFACT_SCHEMA,
            "backend": backend,
            "basis": request.name.strip(),
            "available": False,
            "download_required": False,
            "source": "native_backend",
            "message": str(exc),
        }
    packaged = None
    try:
        packaged = packaged_basis_set_path(request.name)
    except ValueError:
        pass
    stem = _safe_name(request.name)
    suffix = "g94" if backend == "et" else {"gaussian94": "gbs", "orca": "orca", "molpro": "molpro", "cfour": "genbas", "psi4": "psi4"}.get(fmt, fmt)
    content = directory_path / f"{stem}.{suffix}"
    manifest = directory_path / f"{stem}.{_safe_name(backend)}.{fmt}.json"
    available = packaged is not None or (content.is_file() and manifest.is_file())
    return {
        "schema": BASIS_ARTIFACT_SCHEMA,
        "backend": backend,
        "basis": request.name.strip(),
        "format": fmt,
        "available": available,
        "download_required": not available,
        "download_confirmation_required": not available,
        "source": "packaged" if packaged is not None else "cache" if available else "missing",
        "packaged_path": str(packaged) if packaged is not None else "",
        "content_path": str(content),
        "manifest_path": str(manifest),
        "source_url": basis_set_url(request),
    }


def acquire_basis_set(
    request: BasisSetRequest,
    directory: Path | str,
    *,
    fetcher: Callable[[str], bytes] | None = None,
    base_url: str = BSE_BASE_URL,
    confirmed: bool = False,
) -> BasisSetArtifact:
    """Download once, verify, and preserve a backend-native BSE artifact."""

    if not request.name.strip():
        raise ValueError("basis-set name cannot be empty")
    fmt = basis_format_for_backend(request.backend)
    elements = _normalize_elements(request.elements)
    url = basis_set_url(request, base_url=base_url)
    target_dir = Path(directory).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_name(request.name)
    suffix = (
        "g94"
        if request.backend.strip().casefold() == "et"
        else {"gaussian94": "gbs", "orca": "orca", "molpro": "molpro", "cfour": "genbas", "psi4": "psi4"}.get(fmt, fmt)
    )
    content_path = target_dir / f"{stem}.{suffix}"
    backend_key = _safe_name(request.backend.strip().casefold())
    manifest_path = target_dir / f"{stem}.{backend_key}.{fmt}.json"
    if content_path.is_file() and manifest_path.is_file():
        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(content_path.read_bytes()).hexdigest()
        if stored.get("source_url") == url and stored.get("sha256") == digest:
            stored["reused"] = True
            stored.setdefault("normalization", "none")
            stored.setdefault("schema", BASIS_ARTIFACT_SCHEMA)
            return BasisSetArtifact(
                **{key: stored[key] for key in BasisSetArtifact.__dataclass_fields__}
            )
    if not confirmed:
        raise BasisSetConfirmationRequired(
            f"Basis {request.name!r} is not cached for {request.backend}; "
            f"confirm Basis Set Exchange download: {url}"
        )
    payload = (fetcher or _fetch_url)(url)
    if not payload.strip():
        raise RuntimeError("Basis Set Exchange returned an empty basis")
    payload, normalization = _prepare_backend_basis(payload, request.backend)
    digest = hashlib.sha256(payload).hexdigest()
    retrieved = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    atomic_write_bytes(content_path, payload)
    artifact = BasisSetArtifact(
        request.name.strip(),
        request.backend.strip().casefold(),
        fmt,
        elements,
        request.version,
        url,
        str(content_path),
        str(manifest_path),
        digest,
        retrieved,
        normalization=normalization,
    )
    manifest_path.write_text(json.dumps(artifact.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _fetch_url(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "MATRIX-Keymaker/1"})
    with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed HTTPS service by default
        status = getattr(response, "status", 200)
        if status != 200:
            raise RuntimeError(f"Basis Set Exchange returned HTTP {status}")
        return response.read()


def _prepare_backend_basis(payload: bytes, backend: str) -> tuple[bytes, str]:
    """Apply only transformations required by a documented backend parser."""

    if str(backend).strip().casefold() != "et":
        return payload, "none"
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("eT Gaussian94 basis data must be UTF-8 text") from exc
    lines = [line.rstrip() for line in text.splitlines()]
    first_data = next(
        (
            index
            for index, line in enumerate(lines)
            if line.strip() and not line.lstrip().startswith(("!", "#"))
        ),
        None,
    )
    if first_data is None:
        raise ValueError("eT basis data contains no Gaussian94 element blocks")
    if lines[first_data].strip() != "****":
        lines.insert(first_data, "****")
    if not any(line.strip() == "****" for line in lines):
        raise ValueError("eT basis data lacks Gaussian94 element separators")
    while lines and not lines[-1]:
        lines.pop()
    if not lines or lines[-1].strip() != "****":
        raise ValueError("eT basis data must end with a complete Gaussian94 element block")
    # eT/Libint has historically rejected trailing whitespace in .g94 files.
    return ("\n".join(lines) + "\n").encode("utf-8"), "et-libint-g94-v1"


def _normalize_elements(elements: Iterable[str]) -> tuple[str, ...]:
    values = {str(value).strip() for value in elements if str(value).strip()}
    return tuple(sorted(values, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value.casefold())))


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "-", value.strip()).strip("-") or "basis"
