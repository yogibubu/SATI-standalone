"""Correlated QC coefficient uncertainty through VSCF and native spectra."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict, replace
import numpy as np


def _tree(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,dict):return {key:_tree(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [_tree(item) for item in value]
    return deepcopy(value)


def _coefficient_target(path):
    """Accept physical coefficients, never solver controls, powers or indices."""
    if len(path)<3:return False
    if path[0]=='specification':
        return (path[1] in {'one_mode_coefficients_cm','frequencies_cm'}
            or (path[1]=='tensor_surfaces' and len(path)>=5 and path[3]=='coefficients'))
    if path[0]=='blocks' and len(path)>=6 and path[2] in {'active_pairs','contracted_pairs'}:
        return path[4]==(1 if path[2]=='active_pairs' else 2)
    return False


def _leaf(tree,path):
    parent=tree
    for segment in path[:-1]:parent=parent[segment]
    value=parent[path[-1]]
    if isinstance(value,bool) or not np.isscalar(value) or not np.isfinite(float(value)):
        raise ValueError('QM uncertainty targets must be finite scalar physical coefficients')
    return parent,path[-1],float(value)


def _read_output(surface,item):
    field=item['field']
    if field not in {'F_MHz','rho','R_MHz','dipole_au','geometric_potential_MHz','potential_MHz'} and not field.startswith('spin_tensor_surfaces_MHz.'):
        raise ValueError('unsupported contracted uncertainty output field')
    if field.startswith('spin_tensor_surfaces_MHz.'):
        value=surface.spin_tensor_surfaces_MHz[field.split('.',1)[1]]
    else:value=getattr(surface,field)
    if value is None:raise ValueError('requested contracted field is absent')
    value=np.asarray(value,float)
    # A supplied linear functional can implement quadrature, expansion fitting,
    # unit conversion or a field sample; weights are never inferred.
    weights=np.asarray(item['weights'],float)
    offset=float(item.get('offset',0.))
    if weights.shape!=value.shape or not np.all(np.isfinite(weights)) or not np.isfinite(offset):
        raise ValueError('one finite explicit weight per contracted field entry is required')
    return float(np.sum(weights*value)+offset)


def propagate_nmode_qm_uncertainty(base_surface,request,specification,blocks,parameters,covariance,*,
                                    contracted_outputs=(),relative_tolerance=1e-3,absolute_tolerance=1e-5,
                                    qm_covariance_evidence=None):
    """Delta-method covariance, including the response of the VSCF states.

    Parameters are zero-centered physical coefficient increments with explicit
    target paths, factors, units and finite-difference steps. Two symmetric
    step sizes check derivatives; stability boundaries use a second-order
    one-sided stencil and are reported explicitly. Every evaluation recontracts the physical
    model and tracks the assigned native states. Invalid models fail explicitly.
    Numerical truncation and missing-physics errors are not random QM variance.
    """
    from .adaptive_couplings import NModeCouplingBlock,build_nmode_lam_surface,NModeStabilityError
    from .lam_qualification import _evaluate
    from .prediction_budget import _covariance
    if request.get('variables') or request.get('predicates'):
        raise ValueError('uncertainty propagation requires a fixed physical model')
    parameters=tuple(parameters);outputs=tuple(contracted_outputs)
    names=[str(p['name']) for p in parameters]
    output_names=[str(item['name']) for item in outputs]
    if not names or len(set(names))!=len(names) or len(set(output_names))!=len(output_names):
        raise ValueError('unique nonempty QM parameter and contracted-output names required')
    if (not np.all(np.isfinite([relative_tolerance,absolute_tolerance]))
        or relative_tolerance<=0 or absolute_tolerance<=0):
        raise ValueError('positive finite derivative tolerances required')
    if not np.isrealobj(covariance):raise ValueError('QM coefficient covariance must be real')
    c=_covariance(covariance,len(parameters),'QM coefficient')
    template=_tree(dict(specification=specification,blocks=[asdict(b) for b in blocks]))
    registry=[]
    for p in parameters:
        step=float(p['step'])
        if not p.get('units') or not np.isfinite(step) or step<=0 or not p.get('targets'):
            raise ValueError('QM parameters require units, positive steps and explicit coefficient targets')
        targets=[];seen=set()
        for target in p['targets']:
            path=tuple(target['path']);factor=float(target.get('factor',1.))
            if not _coefficient_target(path) or path in seen or not np.isfinite(factor) or factor==0:
                raise ValueError('invalid or duplicate physical QM coefficient target')
            seen.add(path)
            _,_,initial=_leaf(template,path)
            targets.append(dict(path=path,factor=factor,reference_value=initial))
        registry.append(dict(name=p['name'],units=p['units'],step=step,targets=targets))
    evaluated={};rejected=[]
    def evaluate(displacements,reference=None):
        key=tuple(displacements)
        if key in evaluated:return evaluated[key]
        model=deepcopy(template)
        for increment,p in zip(displacements,registry,strict=True):
            for target in p['targets']:
                parent,index,value=_leaf(model,target['path'])
                parent[index]=value+increment*target['factor']
        # The reference base includes the central harmonic transverse ZPE.
        # Move that reference with perturbed frequencies before subtracting it
        # in the anharmonic adapter, so harmonic-frequency uncertainty survives.
        delta_zpe=.5*np.sum(np.asarray(model['specification']['frequencies_cm'])-
            np.asarray(specification['frequencies_cm']),axis=1)
        delta_zpe=delta_zpe.reshape((-1,)+(1,)*(base_surface.dimension-1))*29979.2458
        displaced_base=replace(base_surface,potential_MHz=np.asarray(base_surface.potential_MHz)+delta_zpe)
        try:
            surface=build_nmode_lam_surface(displaced_base,model['specification'],
                [NModeCouplingBlock(**b) for b in model['blocks']])
        except NModeStabilityError:
            rejected.append(list(displacements));raise
        spectrum=_evaluate(surface,request,reference=reference)
        result=np.r_[[ _read_output(surface,item) for item in outputs],spectrum['frequency']]
        if not np.all(np.isfinite(result)):raise RuntimeError('nonfinite propagated QM observable')
        evaluated[key]=result,(spectrum['solved'],surface,spectrum['context'])
        return evaluated[key]
    centre,reference=evaluate(np.zeros(len(parameters)))
    coarse=np.empty((len(centre),len(parameters)));fine=np.empty_like(coarse)
    stencils=[]
    for index,p in enumerate(registry):
        full=np.zeros(len(parameters));full[index]=p['step']
        endpoints=[]
        for direction in (1,-1):
            try:endpoints.append(evaluate(direction*full,reference)[0])
            except NModeStabilityError:endpoints.append(None)
        direction=0 if all(x is not None for x in endpoints) else (1 if endpoints[0] is not None else -1)
        if all(x is None for x in endpoints):
            raise NModeStabilityError('both derivative directions leave the certified model; revise the physical parameterization')
        stencils.append('central' if direction==0 else ('forward' if direction==1 else 'backward'))
        for divisor,jacobian in ((1,coarse),(2,fine)):
            step=p['step']/divisor;delta=np.zeros(len(parameters));delta[index]=step
            if direction==0:
                plus,_=evaluate(delta,reference);minus,_=evaluate(-delta,reference)
                jacobian[:,index]=(plus-minus)/(2*step)
            else:
                first,_=evaluate(direction*delta,reference);second,_=evaluate(2*direction*delta,reference)
                jacobian[:,index]=direction*(-3*centre+4*first-second)/(2*step)
    # Test in each declared parameter increment, so units do not bias the gate.
    steps=np.array([p['step'] for p in registry])
    change=abs((fine-coarse)*steps)
    tolerance=absolute_tolerance+relative_tolerance*abs(fine*steps)
    stable=bool(np.all(change<=tolerance))
    joint=fine@c@fine.T;joint=.5*(joint+joint.T)
    count=len(outputs)
    sigma=np.sqrt(np.maximum(np.diag(joint),0.))
    predicate_indices=[i for i,item in enumerate(outputs) if item.get('fit_path') is not None]
    if any(sigma[i]<=0 for i in predicate_indices):
        raise ValueError('a deterministic contracted output cannot become a finite-sigma predicate')
    paths=[outputs[i]['fit_path'] for i in predicate_indices]
    from .lam_fit import _validate_lam_path
    for path in paths:_validate_lam_path(str(path),reference[1])
    if len(set(paths))!=len(paths):raise ValueError('predicate fit paths must be unique')
    predicates=[dict(path=outputs[i]['fit_path'],value=float(centre[i]),sigma=float(sigma[i]))
        for i in predicate_indices]
    predicate_covariance=joint[np.ix_(predicate_indices,predicate_indices)]
    if predicate_indices:
        # The existing native whitening accepts SPD predicates. Dependent
        # outputs remain available as propagated covariance without fit_path;
        # never add artificial diagonal noise to make them invertible.
        from matrix_numerics.least_squares import CovarianceWhitening
        CovarianceWhitening(sigma[predicate_indices],predicate_covariance)
    from .qm_covariance_qualification import qm_covariance_identity
    evidence=deepcopy(qm_covariance_evidence)
    if evidence is not None and evidence.get('covariance_identity')!=qm_covariance_identity(
            names,[p['units'] for p in registry],c):
        raise ValueError('QM covariance qualification does not match the supplied coefficients, units and covariance')
    return dict(schema='matrix.lock.nmode_qm_uncertainty.v1',qualified=stable,
        qm_covariance_calibrated=bool(evidence is not None and evidence.get('qualified') is True),
        qm_covariance_evidence=evidence,
        method='local delta method; h and h/2 response with certified central or second-order one-sided stencils',
        derivative_stencils=stencils,rejected_stability_probes=rejected,
        parameter_registry=registry,parameter_covariance=c.tolist(),output_names=output_names,
        contracted_values=centre[:count].tolist(),contracted_covariance=joint[:count,:count].tolist(),
        frequencies_MHz=centre[count:].tolist(),frequency_covariance_MHz2=joint[count:,count:].tolist(),
        contracted_frequency_cross_covariance=joint[:count,count:].tolist(),
        joint_covariance=joint.tolist(),jacobian=fine.tolist(),
        derivative_step_comparison=change.tolist(),derivative_step_tolerance=tolerance.tolist(),
        predicates=predicates,predicate_covariance=predicate_covariance.tolist(),
        evaluation_count=len(evaluated),
        scope='conditional on supplied QC covariance and local linear response; no numerical convergence, nonlinear uncertainty or missing-physics claim')


__all__=['propagate_nmode_qm_uncertainty']
