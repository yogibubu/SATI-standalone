from __future__ import annotations
from .capabilities import smith_capabilities

def minimal_capabilities() -> dict[str, object]:
    return smith_capabilities()
