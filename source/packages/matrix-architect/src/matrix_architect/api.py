"""Small, stable public API for application and library consumers."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {'ZaffBuildConfig': ('.compiler', 'ZaffBuildConfig'), 'ArchitectForceField': ('.model', 'ArchitectForceField'), 'build_zaff_force_field': ('.compiler', 'build_zaff_force_field'), 'evaluate_force_field': ('.model', 'evaluate_force_field'), 'load_force_field': ('.model', 'load_force_field'), 'save_force_field': ('.model', 'save_force_field'), 'fit_diagonal_harmonic_zaff': ('.diagonal_harmonic', 'fit_diagonal_harmonic_zaff'), 'fit_sparse_zaff_couplings': ('.sparse_coupling', 'fit_sparse_zaff_couplings'), 'validate_force_field_derivatives': ('.model', 'validate_force_field_derivatives')}

__all__ = list(_EXPORTS)

def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    module = import_module(module_name, 'matrix_architect') if module_name.startswith(".") else import_module(module_name)
    value = getattr(module, attribute)
    globals()[name] = value
    return value
