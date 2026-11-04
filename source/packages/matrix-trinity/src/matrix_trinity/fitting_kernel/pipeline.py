"""Compatibility facade; implementation remains in matrix_smith.coordinate_kernel.pipeline."""
from importlib import import_module

_owner = import_module('matrix_smith.coordinate_kernel.pipeline')
__all__ = [name for name in vars(_owner) if not name.startswith("_")]

def __getattr__(name):
    return getattr(_owner, name)

def __dir__():
    return sorted(set(globals()) | set(dir(_owner)))
