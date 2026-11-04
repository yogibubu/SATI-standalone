"""Immutable execution contracts shared by MATRIX and Keymaker."""

from __future__ import annotations

from copy import deepcopy
from importlib import resources
import json
from typing import Any, Mapping


EXECUTION_PROTOCOL_SCHEMA = "matrix.qm.backend_execution_protocol.v2"
EXECUTION_PROTOCOL_ID = "matrix-qm-backend-execution-v2"
EXECUTION_PROTOCOL_VERSION = "2.0.0"
_BACKENDS = (
    "gaussian",
    "orca",
    "molpro",
    "mrcc",
    "cfour",
    "xtb",
    "pyscf",
    "et",
    "psi4",
    "zaff",
    "external",
)


def load_backend_execution_protocol() -> dict[str, Any]:
    """Return the sole approved backend execution protocol.

    The returned mapping is a copy. Callers cannot mutate the packaged
    authority or silently alter a backend's resource/scientific contract.
    """

    resource = resources.files("matrix_qm").joinpath("data/backend_execution_protocol_v2.json")
    try:
        payload = json.loads(resource.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("backend execution protocol is unreadable") from exc
    validate_backend_execution_protocol(payload)
    return deepcopy(payload)


def validate_backend_execution_protocol(payload: Mapping[str, Any]) -> None:
    """Fail closed if the frozen execution contract has been altered."""

    if payload.get("schema") != EXECUTION_PROTOCOL_SCHEMA:
        raise RuntimeError("unsupported MATRIX backend execution protocol schema")
    if payload.get("protocol_id") != EXECUTION_PROTOCOL_ID:
        raise RuntimeError("unexpected MATRIX backend execution protocol")
    if payload.get("manifest_version") != EXECUTION_PROTOCOL_VERSION:
        raise RuntimeError("unsupported MATRIX backend execution protocol version")
    if payload.get("status") != "approved":
        raise RuntimeError("backend execution protocol is not approved")
    if payload.get("change_policy") != "new_manifest_version_and_explicit_approval":
        raise RuntimeError("backend execution protocol change policy was weakened")
    resource = payload.get("resource_contract")
    if not isinstance(resource, Mapping):
        raise RuntimeError("backend execution protocol lacks resource contract")
    if resource.get("authorization_required") is not True:
        raise RuntimeError("backend launches must require explicit authorization")
    if resource.get("implicit_defaults_allowed") is not False:
        raise RuntimeError("backend launches may not use implicit resource defaults")
    contracts = payload.get("backend_contracts")
    if not isinstance(contracts, Mapping) or tuple(contracts) != _BACKENDS:
        raise RuntimeError("backend execution protocol does not cover the approved backends")
    names_and_aliases: set[str] = set()
    for backend in _BACKENDS:
        record = contracts[backend]
        if not isinstance(record, Mapping):
            raise RuntimeError(f"invalid execution contract for {backend}")
        if record.get("resource_owner") != "MATRIX_launch_authorization":
            raise RuntimeError(f"resource owner changed for {backend}")
        aliases = record.get("aliases")
        if not isinstance(aliases, list):
            raise RuntimeError(f"backend aliases are invalid for {backend}")
        for name in (backend, *(str(alias) for alias in aliases)):
            normalized = name.casefold().replace("-", "_")
            if not normalized or normalized in names_and_aliases:
                raise RuntimeError("backend names and aliases must be unique")
            names_and_aliases.add(normalized)
        if record.get("resource_parallelism") not in {"processes", "threads"}:
            raise RuntimeError(f"backend parallelism is invalid for {backend}")
    pyscf = contracts["pyscf"]
    if pyscf.get("symmetry") is not True:
        raise RuntimeError("PySCF symmetry=True is mandatory")
    if pyscf.get("density_fitting") != "required_for_supported_HF_DFT_SCF_requests":
        raise RuntimeError("PySCF density-fitting contract changed")
    if pyscf.get("tight_scf") is not False or pyscf.get("nosymm") is not False:
        raise RuntimeError("PySCF tight/nosymm guard changed")


def canonical_backend_name(
    backend: str,
    protocol: Mapping[str, Any] | None = None,
) -> str:
    """Resolve one backend alias through the versioned execution authority."""

    normalized = str(backend).strip().casefold().replace("-", "_")
    authority = load_backend_execution_protocol() if protocol is None else protocol
    matches = tuple(
        name
        for name, record in authority["backend_contracts"].items()
        if normalized
        in {
            name.casefold().replace("-", "_"),
            *(str(alias).casefold().replace("-", "_") for alias in record.get("aliases", ())),
        }
    )
    if len(matches) != 1:
        raise KeyError(f"unknown or ambiguous QM backend: {backend}")
    return matches[0]


def backend_execution_contract(
    backend: str,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return an isolated copy of the canonical backend execution record."""

    authority = load_backend_execution_protocol() if protocol is None else protocol
    canonical = canonical_backend_name(backend, authority)
    return deepcopy(authority["backend_contracts"][canonical])
