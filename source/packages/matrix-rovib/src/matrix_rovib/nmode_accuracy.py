"""Separate transverse VSCF and native rovibrational convergence estimators."""
from __future__ import annotations
from copy import deepcopy
import numpy as np


def evaluate_nmode_accuracy(base_surface,request,specification,blocks,policy,budget,*,reference=None,
                            observable_budgets=None):
    """Qualify one supplied n-mode model, preserving assigned state identities.

    Grid refinement interpolates the supplied representation. It does not add
    QC samples. VSCF and native frequency/intensity errors are added as
    absolute estimates on measured lines and their component strengths.
    """
    from .adaptive_couplings import build_nmode_lam_surface
    from .lam_accuracy import qualify_lam_accuracy,_carry_assignments
    from .lam_qualification import _evaluate,refine_lam_grid
    local=deepcopy(specification)
    from inspect import signature
    from matrix_gf.reduced_anharmonic import contract_nmode_vscf_to_active_surface
    default_quanta=signature(contract_nmode_vscf_to_active_surface).parameters['maximum_quanta'].default
    declared=local.get('maximum_quanta',default_quanta)
    if isinstance(declared,bool) or not isinstance(declared,(int,np.integer)) or declared<1:
        raise ValueError('a positive integer VSCF maximum_quanta is required')
    quanta=int(declared)
    step=policy.get('vscf_quanta_step',2)
    maximum=policy.get('maximum_vscf_quanta')
    if (isinstance(step,bool) or not isinstance(step,int) or step<1
        or isinstance(maximum,bool) or not isinstance(maximum,int) or maximum<quanta+step):
        raise ValueError('VSCF qualification requires an integer maximum allowing an independent enlargement')
    native=deepcopy(policy.get('native',{}))
    property_budgets = {}
    for name, raw in (observable_budgets or {}).items():
        if name != 'intensity':
            raise ValueError('unsupported numerical property observable')
        bound = np.asarray(raw, float)
        if not np.all(np.isfinite(bound)) or np.any(bound <= 0):
            raise ValueError('positive finite intensity budgets required')
        property_budgets[name] = bound
        native['intensity_budget'] = bound/2
    native['frequency_tolerance_MHz']=min(float(native.get('frequency_tolerance_MHz',np.inf)),float(min(budget))/2)
    # The physical compression tolerance is independent of experimental residuals.
    native['fraction_of_experimental_sigma']=float(np.max(np.asarray(budget)/np.array(
        [o['uncertainty_MHz']/np.sqrt(o.get('weight',1.)) for o in request['observations']])))
    history=[]
    def represented(surface,shape):
        return refine_lam_grid(surface,shape) if surface.grid_shape!=shape else surface
    while True:
        local['maximum_quanta']=quanta
        surface=build_nmode_lam_surface(base_surface,local,blocks)
        initial_request=deepcopy(reference[2].get("_nmode_numerical_request",request) if reference else request)
        if reference is not None:
            shape=tuple(max(a,b) for a,b in zip(surface.grid_shape,reference[1].grid_shape,strict=True))
            surface=represented(surface,shape)
            before=_evaluate(surface,initial_request,reference=reference)
            initial_request=_carry_assignments(initial_request,before)
        qualified=qualify_lam_accuracy(surface,initial_request,native)
        surface=qualified.surface
        value=_evaluate(surface,qualified.request,reference=reference)
        value['context']['_nmode_numerical_request']=deepcopy(qualified.request)
        entry=dict(maximum_quanta=quanta,native=qualified.report)
        if not qualified.report['converged']:
            history.append(entry)
            return surface,value,dict(converged=False,envelope_MHz=None,
                observable_envelopes={name:None for name in property_budgets},
                properties_converged=False,history=history,reason='native representation did not converge')
        native_error=np.maximum(np.asarray(qualified.report['independent_L1_estimate_MHz']),
            abs(np.asarray(qualified.report['joint_frequency_change_MHz'])))
        enlarged=deepcopy(local);enlarged['maximum_quanta']=min(quanta+step,maximum)
        other=represented(build_nmode_lam_surface(base_surface,enlarged,blocks),surface.grid_shape)
        probe=_evaluate(other,qualified.request,reference=(value['solved'],surface,value['context']))
        transverse=abs(probe['frequency']-value['frequency'])
        envelope=native_error+transverse
        property_envelopes = {}
        for name, bound in property_budgets.items():
            if value.get(name) is None or probe.get(name) is None:
                raise ValueError('intensity qualification requires native dipole observables')
            bound = np.broadcast_to(bound, np.shape(value[name])).copy()
            property_budgets[name] = bound
            native_property_error = np.maximum(
                qualified.report['independent_L1_intensity_estimate'],
                abs(np.asarray(qualified.report['joint_intensity_change'])))
            transverse_property_error = abs(probe[name]-value[name])
            property_envelopes[name] = native_property_error+transverse_property_error
            entry.update(native_intensity_envelope=native_property_error.tolist(),
                vscf_intensity_change=transverse_property_error.tolist(),
                intensity_envelope=property_envelopes[name].tolist())
        entry.update(vscf_probe_quanta=enlarged['maximum_quanta'],vscf_frequency_change_MHz=transverse.tolist(),
            native_envelope_MHz=native_error.tolist(),envelope_MHz=envelope.tolist())
        history.append(entry)
        properties_passed = all(np.all(property_envelopes[name] <= bound)
                                for name, bound in property_budgets.items())
        passed=bool(np.all(envelope<=budget) and properties_passed)
        if passed or enlarged['maximum_quanta']>=maximum:
            return surface,value,dict(converged=passed,envelope_MHz=envelope.tolist(),history=history,
                observable_envelopes={name:array.tolist() for name,array in property_envelopes.items()},
                properties_converged=bool(property_budgets and properties_passed),
                observable_budgets={name:bound.tolist() for name,bound in property_budgets.items()},
                line_budget_MHz=np.asarray(budget).tolist(),
                scope='tested VSCF enlargement and requested native axes; convergence estimates, not rigorous bounds',
                reason='passed' if passed else 'VSCF representation budget exhausted')
        quanta=enlarged['maximum_quanta']
