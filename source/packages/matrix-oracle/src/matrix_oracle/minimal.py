"""Minimal diagnostics that avoid optional QM/GUI imports."""
from __future__ import annotations
import platform
from .dependencies import dependency_status
from .remote import local_qm_capabilities

def minimal_capabilities() -> dict[str, object]:
    return {"schema": "matrix.oracle.minimal_capabilities.v1",
            "platform": platform.platform(), "machine": platform.machine(),
            "dependencies": dependency_status(), "local_qm": local_qm_capabilities()}
