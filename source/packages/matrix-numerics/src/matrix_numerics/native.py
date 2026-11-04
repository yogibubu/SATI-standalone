"""Portable policy for optional ahead-of-time compiled numerical kernels."""

from __future__ import annotations

from dataclasses import dataclass
import os
import platform
from typing import Any


NATIVE_BACKEND_SCHEMA = "matrix.native.backend.v1"
DEFAULT_NATIVE_MIN_WORK = 2


@dataclass(frozen=True)
class NativeBackend:
    """Resolution of one optional native extension against the Python reference."""

    name: str
    available: bool
    accelerated: bool
    requested: str
    reason: str = ""
    implementation: str = "python"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": NATIVE_BACKEND_SCHEMA,
            "requested": self.requested,
            "backend": self.name,
            "implementation": self.implementation,
            "available": self.available,
            "accelerated": self.accelerated,
            "machine": platform.machine(),
            "system": platform.system(),
            "reason": self.reason,
        }


def resolve_native_backend(
    *,
    extension_available: bool,
    extension_name: str,
    workload_size: int,
    requested: str | None = None,
    minimum_work: int | None = None,
) -> NativeBackend:
    """Choose a compiled kernel or the invariant Python reference.

    The selector is deliberately independent of any particular extension.
    Scientific packages retain ownership of their kernels and pass their
    availability here.  ``auto`` uses native code only above a configurable
    threshold; ``python`` and ``compiled`` make parity tests reproducible.
    """

    choice = (
        requested
        if requested is not None
        else os.environ.get("MATRIX_NUMERICAL_BACKEND", "auto")
    )
    normalized = str(choice).strip().casefold() or "auto"
    normalized = {
        "native": "compiled",
        "serial": "compiled",
        "reference": "python",
        "numpy": "python",
    }.get(normalized, normalized)
    if normalized not in {"auto", "python", "compiled"}:
        raise ValueError(
            "MATRIX numerical backend must be auto, python, or compiled"
        )
    threshold = (
        _positive_env_int("MATRIX_NATIVE_MIN_WORK", DEFAULT_NATIVE_MIN_WORK)
        if minimum_work is None
        else int(minimum_work)
    )
    if threshold < 1:
        raise ValueError("MATRIX native minimum work must be positive")

    if normalized == "python":
        return NativeBackend(
            name="python",
            available=True,
            accelerated=False,
            requested=normalized,
            reason="Python reference explicitly requested",
        )
    if extension_available and (
        normalized == "compiled" or int(workload_size) >= threshold
    ):
        return NativeBackend(
            name=extension_name,
            available=True,
            accelerated=True,
            requested=normalized,
            implementation="compiled",
        )
    if normalized == "compiled" and _truthy_env(
        "MATRIX_NUMERICAL_BACKEND_STRICT", False
    ):
        raise RuntimeError(
            f"requested MATRIX compiled backend {extension_name} is unavailable"
        )
    reason = (
        f"native threshold not reached ({workload_size} < {threshold})"
        if extension_available
        else f"optional extension {extension_name} is unavailable"
    )
    return NativeBackend(
        name="python",
        available=True,
        accelerated=False,
        requested=normalized,
        reason=reason,
    )


def _positive_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    return value


def _truthy_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().casefold() not in {"", "0", "false", "no", "off"}


__all__ = [
    "DEFAULT_NATIVE_MIN_WORK",
    "NATIVE_BACKEND_SCHEMA",
    "NativeBackend",
    "resolve_native_backend",
]
