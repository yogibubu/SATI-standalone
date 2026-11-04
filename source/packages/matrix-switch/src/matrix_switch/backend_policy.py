"""Canonical SWITCH-primary backend-selection policy."""

from __future__ import annotations

import os
from typing import Literal


SWITCH_BACKEND_POLICY_SCHEMA = "matrix.switch.backend-policy.v1"
STRICT_SWITCH = "strict-switch"
ALLOW_RDKIT_FALLBACK = "allow-rdkit-fallback"
SwitchBackendPolicy = Literal["strict-switch", "allow-rdkit-fallback"]
_POLICIES = frozenset((STRICT_SWITCH, ALLOW_RDKIT_FALLBACK))


def resolve_switch_backend_policy(value: str | None = None) -> SwitchBackendPolicy:
    """Resolve one policy; normal execution permits only an explicit fallback."""

    raw = value
    if raw is None:
        raw = os.environ.get("MATRIX_SWITCH_BACKEND_POLICY", ALLOW_RDKIT_FALLBACK)
    normalized = str(raw).strip().casefold().replace("_", "-")
    if normalized not in _POLICIES:
        choices = ", ".join(sorted(_POLICIES))
        raise ValueError(f"SWITCH backend policy must be one of: {choices}")
    return normalized  # type: ignore[return-value]


__all__ = [
    "ALLOW_RDKIT_FALLBACK",
    "STRICT_SWITCH",
    "SWITCH_BACKEND_POLICY_SCHEMA",
    "SwitchBackendPolicy",
    "resolve_switch_backend_policy",
]
