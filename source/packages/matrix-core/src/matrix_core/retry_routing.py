"""Keymaker retry/fallback policy helpers."""
from __future__ import annotations
from collections.abc import Callable, Sequence
from typing import TypeVar

T = TypeVar("T")

def run_with_host_fallback(hosts: Sequence[str], operation: Callable[[str], T]) -> tuple[str, T, tuple[str, ...]]:
    errors: list[str] = []
    for host in hosts:
        try: return str(host), operation(str(host)), tuple(errors)
        except Exception as exc: errors.append(f"{host}:{type(exc).__name__}:{exc}")
    raise RuntimeError("all Keymaker hosts failed: " + "; ".join(errors))
