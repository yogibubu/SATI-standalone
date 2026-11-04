"""Compatibility namespace for the staged SURVIBFIT migration."""
from __future__ import annotations
from importlib import import_module

_MODULES = ("primitives", "pipeline", "geometry", "transforms", "fit", "weights", "terms", "io", "gaussian_log")

def load(module: str):
    if module not in _MODULES:
        raise ValueError(f"unsupported SURVIBFIT module: {module}")
    if module in {"weights", "terms", "fit", "io"}:
        return import_module(f"matrix_trinity.fitting_kernel.{module}")
    return import_module(f"matrix_smith.coordinate_kernel.{module}")

def capability():
    return {"schema":"matrix.trinity.survibfit_boundary.v2", "modules":list(_MODULES), "implementation":"legacy-compatible adapter"}
