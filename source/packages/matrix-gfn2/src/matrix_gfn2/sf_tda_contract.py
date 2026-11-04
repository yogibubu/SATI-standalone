"""Versioned scientific contract for the MATRIX spin-flip TDA project."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.resources import files
import json
from types import MappingProxyType
from typing import Any, Mapping


_CONTRACT_RESOURCE = "sf-tda-contract-v1.json"
# Updated only through a deliberate scientific-contract revision.
EXPECTED_SF_TDA_CONTRACT_SHA256 = (
    "9b8337d1740325ae2cb1330a93272c1f8db379577336c4dde317264d8d133032"
)


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class SFTDAContract:
    schema_version: int
    model_id: str
    ground_state_model: str
    reference: Mapping[str, Any]
    excitation_space: Mapping[str, Any]
    solver: Mapping[str, Any]
    electronic_properties: Mapping[str, Any]
    kernel_profiles: Mapping[str, Any]
    gradient: Mapping[str, Any]
    production_dependencies: Mapping[str, Any]
    sha256: str


def load_sf_tda_contract() -> SFTDAContract:
    payload = files("matrix_gfn2").joinpath("data", _CONTRACT_RESOURCE).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != EXPECTED_SF_TDA_CONTRACT_SHA256:
        raise RuntimeError("SF-TDA scientific contract differs from its frozen fingerprint")
    document = json.loads(payload)
    required = {
        "schema_version",
        "model_id",
        "ground_state_model",
        "reference",
        "excitation_space",
        "solver",
        "electronic_properties",
        "kernel_profiles",
        "gradient",
        "production_dependencies",
    }
    missing = required.difference(document)
    if missing:
        raise ValueError(f"SF-TDA contract lacks required fields: {sorted(missing)}")
    if document["model_id"] != "SF-TDA-MATRIX-GFN2-R.v1":
        raise ValueError("unsupported SF-TDA model identifier")
    if document["reference"].get("kind") != "restricted-open-shell":
        raise ValueError("version 1 requires a restricted-open-shell reference")
    if document["production_dependencies"].get("external_xtb") is not False:
        raise ValueError("the resident SF-TDA contract cannot depend on external xTB")
    return SFTDAContract(
        schema_version=int(document["schema_version"]),
        model_id=str(document["model_id"]),
        ground_state_model=str(document["ground_state_model"]),
        reference=_freeze(document["reference"]),
        excitation_space=_freeze(document["excitation_space"]),
        solver=_freeze(document["solver"]),
        electronic_properties=_freeze(document["electronic_properties"]),
        kernel_profiles=_freeze(document["kernel_profiles"]),
        gradient=_freeze(document["gradient"]),
        production_dependencies=_freeze(document["production_dependencies"]),
        sha256=digest,
    )


__all__ = [
    "EXPECTED_SF_TDA_CONTRACT_SHA256",
    "SFTDAContract",
    "load_sf_tda_contract",
]
