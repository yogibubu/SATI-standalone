"""Small, stable public API for application and library consumers."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {'GFResult': ('.harmonic', 'GFResult'), 'solve_wilson_gf': ('.harmonic', 'solve_wilson_gf'), 'HarmonicFieldTransform': ('.harmonic_field', 'HarmonicFieldTransform'), 'transform_cartesian_hessian': ('.harmonic_field', 'transform_cartesian_hessian'), 'gf_from_hessian_input_and_xyzin': ('.internal', 'gf_from_hessian_input_and_xyzin'), 'read_gf_ped_section': ('.sections', 'read_gf_ped_section'), 'write_gf_ped_section': ('.sections', 'write_gf_ped_section'), 'AdaptiveEffectiveSurface': ('.adaptive_surface', 'AdaptiveEffectiveSurface'), 'fit_adaptive_effective_surface': ('.adaptive_surface', 'fit_adaptive_effective_surface'), 'adaptive_surface_oscillator_matrices': ('.adaptive_surface', 'adaptive_surface_oscillator_matrices'), 'contract_nmode_vscf_to_adaptive_surface': ('.reduced_anharmonic', 'contract_nmode_vscf_to_adaptive_surface')}

_EXPORTS['plan_tabulated_refinement'] = ('.tabulated_sampling', 'plan_tabulated_refinement')
__all__ = list(_EXPORTS)

def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    module = import_module(module_name, 'matrix_gf') if module_name.startswith(".") else import_module(module_name)
    value = getattr(module, attribute)
    globals()[name] = value
    return value
