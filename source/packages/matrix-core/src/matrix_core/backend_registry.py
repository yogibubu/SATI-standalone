"""Declarative backend registry for host capability routing."""
from __future__ import annotations
from typing import Mapping, Sequence

def available_backend(host: Mapping[str, object], backend: str, *, architecture: str | None = None) -> bool:
    if architecture and str(host.get("architecture", "")) != architecture: return False
    return str(backend) in {str(value) for value in host.get("backends", ())}

def select_backend(hosts: Sequence[Mapping[str, object]], backend: str, *, architecture: str | None = None) -> dict[str, object]:
    matches = [host for host in hosts if available_backend(host, backend, architecture=architecture)]
    if not matches: raise ValueError(f"backend {backend!r} unavailable for requested architecture")
    return dict(sorted(matches, key=lambda host: (int(host.get("priority", 100)), str(host.get("name", ""))))[0])
