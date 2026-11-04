"""Progressive variable release through the existing LOCK/MORPHEUS minimizer."""
from __future__ import annotations

import numpy as np


def fit_assignment_stage(context, observations, variables, initial, active_paths, request, *, max_nfev):
    from matrix_numerics.least_squares import CovarianceWhitening
    from .fit_convergence import statistical_trust_region,assess_statistical_convergence

    initial=np.asarray(initial,float)
    active=np.array([i for i,v in enumerate(variables) if v['path'] in active_paths],int)
    lower=np.array([v['lower'] for v in variables])[active]
    upper=np.array([v['upper'] for v in variables])[active]
    scales=np.array([v['scale'] for v in variables])[active]
    observed=np.array([r['frequency_MHz'] for r in observations],float)
    weights=np.sqrt([r.get('weight',1.) for r in observations])
    sigma=np.array([r['uncertainty_MHz'] for r in observations])/weights
    covariance=request.get('covariance_MHz2')
    if covariance is not None:covariance=np.asarray(covariance)/weights[:,None]/weights[None,:]
    whitening=CovarianceWhitening(sigma,covariance)
    penalties=[]
    prior=[(i,v['prior']['value'],v['prior']['sigma']) for i,v in enumerate(variables) if v.get('prior')]
    names=[v['name'] for v in variables]
    predicates=[]
    for p in request.get('predicates',()):
        if p['name'] not in names:raise ValueError('QM predicate refers to an unknown variable')
        predicates.append((names.index(p['name']),p['value'],p['sigma']))
    for rows,key in ((prior,'prior_covariance'),(predicates,'predicate_covariance')):
        if not rows:
            if request.get(key) is not None:raise ValueError('QM covariance without corresponding predicate rows')
            continue
        ids=np.array([r[0] for r in rows],int);means=np.array([r[1] for r in rows],float)
        w=CovarianceWhitening([r[2] for r in rows],request.get(key))
        penalties.append((ids,means,w))
    def expand(x):
        values=initial.copy();values[active]=x;return values
    def residual(x):
        values=expand(x);data=context.evaluate(values,observations)
        return np.concatenate([whitening.apply(data['values']-observed)]+
            [w.apply(values[ids]-means) for ids,means,w in penalties])
    def jacobian(x):
        data=context.evaluate(expand(x),observations)
        rows=[whitening.apply(data['jacobian'][:,active])]
        for ids,_,w in penalties:
            rows.append(w.apply(np.eye(len(initial))[ids][:,active]))
        return np.vstack(rows)
    convergence=request.get('options',{}).get('convergence',{})
    settings={k:v for k,v in convergence.items() if k!='mode'}
    def assess(x):
        return assess_statistical_convergence(residual(x),jacobian(x),x,lower,upper,
            data_count=len(observations),data_correction_transform=whitening.standardized_transform,**settings)
    class Converged(Exception):
        def __init__(self,x):self.x=np.asarray(x).copy()
    def checked(x):
        if assess(x)['converged']:raise Converged(x)
        return jacobian(x)
    trace=[]
    try:
        optimized=statistical_trust_region(residual,checked,initial[active],lower,upper,scales,
            max_nfev=max_nfev,trace=trace,
            state_callback=lambda s:context.promote_accepted(expand(s['values'])))
        x=optimized.x
    except Converged as stop:x=stop.x
    report=assess(x);values=expand(x);data=context.evaluate(values,observations)
    return dict(success=bool(report['converged']),convergence=report,values=values.tolist(),
        rank=report['rank'],active_paths=list(active_paths),iteration_trace=trace,
        chi_square=float(residual(x)@residual(x)),
        data_chi_square=float(whitening.apply(data['values']-observed)@whitening.apply(data['values']-observed)),
        rms_residual_MHz=float(np.sqrt(np.mean((data['values']-observed)**2))),
        frequencies_MHz=data['values'].tolist(),jacobian=data['jacobian'].tolist(),
        covariance_scope='full local correlated QM penalties retained under affine variable freezing')
