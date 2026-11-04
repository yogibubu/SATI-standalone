"""Pinned scientific reference for the ARCHITECT ZAFF hierarchy in KeyMaker."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from importlib import resources
import json


ARCHITECT_ZAFF_REFERENCE_PROTOCOL_ID = "architect-zaff-hierarchy.v7"
ARCHITECT_ZAFF_REFERENCE_SHA256 = (
    "37d45228b8777e6fbc533bbbbf9ccb10ffedc7bb3535151683ec22000a01d8f2"
)
ARCHITECT_ZAFF_REFERENCE_VERSIONS = {
    "architect-zaff-hierarchy.v1": (
        "architect_zaff_hierarchy_v1.json",
        "e85c0649da8ba812522b69434b5a0aeffe2b0714103c416cec63d354e1679482",
    ),
    "architect-zaff-hierarchy.v2": (
        "architect_zaff_hierarchy_v2.json",
        "5f32f3ed72e5973de1df9cba85368155a3114b0d1af279c4473eb81aaffa03e2",
    ),
    "architect-zaff-hierarchy.v3": (
        "architect_zaff_hierarchy_v3.json",
        "0b23e7f25ec1b621d267d47235dfe4351c84708b83bc330fb00b5bba729bb7a3",
    ),
    "architect-zaff-hierarchy.v4": (
        "architect_zaff_hierarchy_v4.json",
        "db10f53d18d874ab4ab2e41c863cedce59d5f9dde7938f12bdf6870813e67f21",
    ),
    "architect-zaff-hierarchy.v5": (
        "architect_zaff_hierarchy_v5.json",
        "4bf8ab37d51e2f65cbd2f40553d08fcf8daf8edd27e62595d37623237fbb7ea8",
    ),
    "architect-zaff-hierarchy.v6": (
        "architect_zaff_hierarchy_v6.json", "bbfecbdb7c607c4afb300b6ea31b00a6175a566cc8aa39cd943bce327086d874",
    ),
    ARCHITECT_ZAFF_REFERENCE_PROTOCOL_ID: (
        "architect_zaff_hierarchy_v7.json", ARCHITECT_ZAFF_REFERENCE_SHA256,
    ),
}


def load_architect_zaff_reference_protocol(
    protocol_id: str = ARCHITECT_ZAFF_REFERENCE_PROTOCOL_ID,
) -> dict[str, object]:
    """Reject any in-place change to the approved protocol resource."""

    try:
        filename, expected_digest = ARCHITECT_ZAFF_REFERENCE_VERSIONS[protocol_id]
    except KeyError as exc:
        raise ValueError(f"unknown ARCHITECT ZAFF protocol version: {protocol_id}") from exc
    raw = resources.files("matrix_core").joinpath("data", filename).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_digest:
        raise RuntimeError(
            "ARCHITECT ZAFF reference changed in place; a new version and "
            "explicit user approval are required"
        )
    payload = json.loads(raw)
    if (
        payload.get("protocol_id") != protocol_id
        or payload.get("status") != "approved_reference"
        or payload.get("change_policy") != "new_version_and_explicit_user_approval"
    ):
        raise RuntimeError("ARCHITECT ZAFF reference identity is invalid")
    return deepcopy(payload)


__all__ = [
    "ARCHITECT_ZAFF_REFERENCE_PROTOCOL_ID",
    "ARCHITECT_ZAFF_REFERENCE_SHA256",
    "ARCHITECT_ZAFF_REFERENCE_VERSIONS",
    "load_architect_zaff_reference_protocol",
]
