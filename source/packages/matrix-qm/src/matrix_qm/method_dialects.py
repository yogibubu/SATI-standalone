"""Backend spellings for canonical electronic-structure methods."""

from __future__ import annotations

from copy import deepcopy
from importlib import resources
import json
import re
from typing import Any, Mapping

from .execution_protocol import canonical_backend_name


BACKEND_METHOD_DIALECTS_SCHEMA = "matrix.qm.backend_method_dialects.v1"
BACKEND_METHOD_DIALECTS_ID = "matrix-qm-backend-method-dialects-v1"
BACKEND_METHOD_DIALECTS_VERSION = "1.0.0"
_CHANGE_POLICY = "new_manifest_version_and_explicit_approval"
_FALLBACK_POLICY = "preserve_explicit_engine_spelling_when_no_override_exists"


def _method_key(method: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(method).strip().upper())


def load_backend_method_dialects() -> dict[str, Any]:
    """Return the validated, versioned method-spelling atlas."""

    resource = resources.files("matrix_qm").joinpath("data/backend_method_dialects_v1.json")
    try:
        payload = json.loads(resource.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("backend method-dialect atlas is unreadable") from exc
    validate_backend_method_dialects(payload)
    return deepcopy(payload)


def validate_backend_method_dialects(payload: Mapping[str, Any]) -> None:
    """Reject ambiguous or scientifically unqualified spelling overrides."""

    if payload.get("schema") != BACKEND_METHOD_DIALECTS_SCHEMA:
        raise RuntimeError("unsupported MATRIX backend method-dialect schema")
    if payload.get("atlas_id") != BACKEND_METHOD_DIALECTS_ID:
        raise RuntimeError("unexpected MATRIX backend method-dialect atlas")
    if payload.get("manifest_version") != BACKEND_METHOD_DIALECTS_VERSION:
        raise RuntimeError("unsupported MATRIX backend method-dialect version")
    if payload.get("status") != "approved":
        raise RuntimeError("backend method-dialect atlas is not approved")
    if payload.get("change_policy") != _CHANGE_POLICY:
        raise RuntimeError("backend method-dialect change policy was weakened")
    if payload.get("fallback_policy") != _FALLBACK_POLICY:
        raise RuntimeError("backend method-dialect fallback policy changed")
    records = payload.get("records")
    if not isinstance(records, list):
        raise RuntimeError("backend method-dialect atlas lacks records")
    observed: set[tuple[str, str]] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise RuntimeError("invalid backend method-dialect record")
        canonical = _method_key(str(record.get("canonical_method", "")))
        engine = str(record.get("engine_method", "")).strip()
        equivalence = str(record.get("scientific_equivalence", "")).strip()
        try:
            backend = canonical_backend_name(str(record.get("backend", "")))
        except KeyError as exc:
            raise RuntimeError("method-dialect record names an unknown backend") from exc
        if not canonical or not engine or not equivalence:
            raise RuntimeError("incomplete backend method-dialect record")
        key = (canonical, backend)
        if key in observed:
            raise RuntimeError("duplicate backend method-dialect record")
        observed.add(key)


def backend_method_name(
    backend: str,
    method: str,
    atlas: Mapping[str, Any] | None = None,
) -> str:
    """Translate a canonical method only when the backend requires another spelling."""

    requested = str(method).strip()
    if not requested:
        raise ValueError("QM method cannot be empty")
    authority = load_backend_method_dialects() if atlas is None else atlas
    validate_backend_method_dialects(authority)
    canonical_backend = canonical_backend_name(backend)
    key = _method_key(requested)
    matches = tuple(
        str(record["engine_method"]).strip()
        for record in authority["records"]
        if _method_key(str(record["canonical_method"])) == key
        and canonical_backend_name(str(record["backend"])) == canonical_backend
    )
    if len(matches) > 1:
        raise RuntimeError("ambiguous backend method-dialect resolution")
    return matches[0] if matches else requested
