"""Correlated profile objective, bounded elimination and joint uncertainty."""
from __future__ import annotations

import numpy as np
from matrix_numerics.least_squares import CovarianceWhitening,svd_covariance


def normalize_profile_predicates(variables,request):
    """Expand native path/pattern selectors, retaining the legacy exact-name form."""
    from copy import deepcopy
    from .lam_fit import _lam_predicates
    result=deepcopy(request);raw=deepcopy(result.get('predicates',[]))
    by_name={v['name']:v['path'] for v in variables}
    for p in raw:
        if 'path' not in p and 'path_pattern' not in p:
            if p.get('name') not in by_name:raise ValueError('QM predicate refers to an unknown profile variable')
            p['path']=by_name[p['name']]
    parsed=_lam_predicates(raw,variables)
    result['predicates']=[dict(name=variables[p['index']]['name'],value=p['value'],sigma=p['sigma']) for p in parsed]
    return result


class ProfileObjective:
    def __init__(self,context,train,observed,sigma,variables,request):
        self.context=context;self.train=np.asarray(train,int);self.observed=np.asarray(observed,float)
        self.variables=variables;self.rank_rcond=request.get('options',{}).get('convergence',{}).get('rank_rcond',1.e-10)
        self.whitening=CovarianceWhitening(sigma,request.get('covariance_MHz2'))
        self.penalties=[];names=[v['name'] for v in variables]
        prior=[(i,v['prior']['value'],v['prior']['sigma']) for i,v in enumerate(variables) if v.get('prior')]
        predicates=[]
        for p in request.get('predicates',[]):
            if p['name'] not in names:raise ValueError('QM predicate refers to an unknown profile variable')
            predicates.append((names.index(p['name']),p['value'],p['sigma']))
        for rows,key in ((prior,'prior_covariance'),(predicates,'predicate_covariance')):
            if not rows:
                if request.get(key) is not None:raise ValueError('QM covariance without corresponding predicate rows')
                continue
            ids=np.array([r[0] for r in rows],int);mean=np.array([r[1] for r in rows],float)
            if not np.all(np.isfinite(mean)):raise ValueError('finite QM predicate means required')
            white=CovarianceWhitening([r[2] for r in rows],request.get(key))
            self.penalties.append((white.apply(np.eye(len(variables))[ids]),white.apply(mean)))

    def linearize(self,values,linear=(),nonlinear=()):
        data=self.context.evaluate(values);raw=data['jacobian'][self.train]
        residual=[self.whitening.apply(data['values'][self.train]-self.observed)]
        rows=[self.whitening.apply(raw)]
        for p,mean in self.penalties:residual.append(p@values-mean);rows.append(p)
        mixed=np.zeros((sum(len(r) for r in residual),len(linear),len(nonlinear)))
        # Only the amplitude column depends on nonlinear profile coordinates.
        # All physical, linewidth and offset responses are proportional to it.
        amplitude=next((i for i,v in enumerate(self.variables) if v['path']=='profile.amplitude'),None)
        if amplitude is not None and amplitude in linear and len(nonlinear):
            column=list(linear).index(amplitude)
            mixed[:len(self.train),column,:]=self.whitening.apply(raw[:,nonlinear]/values[amplitude])
        return np.concatenate(residual),np.vstack(rows),mixed

    def covariance(self,values,lower,upper):
        _,jac,_=self.linearize(values)
        covariance,rank,singular=svd_covariance(jac,rcond=self.rank_rcond)
        bound=np.array([np.isfinite(lo) and abs(x-lo)<=1e-10*max(abs(x),abs(lo),1.)
                       or np.isfinite(hi) and abs(x-hi)<=1e-10*max(abs(x),abs(hi),1.)
                       for x,lo,hi in zip(values,lower,upper,strict=True)])
        valid=rank==len(values) and not np.any(bound)
        return dict(parameter_paths=[v['path'] for v in self.variables],rank=rank,
            matrix=covariance.tolist() if valid else None,
            standard_errors=np.sqrt(np.maximum(np.diag(covariance),0.)).tolist() if valid else None,
            at_bounds=bound.tolist(),singular_values=singular.tolist(),
            scope='joint local Gaussian augmented covariance, including eliminated/nonlinear cross-correlations; supplied noise scale unchanged; unavailable for rank deficiency or active bounds; not calibrated model-error intervals')


def fit_projected_profile(objective,variables,initial,request,*,max_nfev):
    from matrix_numerics.variable_projection import VariableProjection
    from .fit_convergence import statistical_trust_region,assess_statistical_convergence
    paths=[v['path'] for v in variables]
    linear=[i for i,p in enumerate(paths) if p in ('profile.amplitude','profile.baseline','profile.slope_per_MHz')]
    lo=np.array([v['lower'] for v in variables]);hi=np.array([v['upper'] for v in variables])
    settings={k:v for k,v in request.get('options',{}).get('convergence',{}).items() if k!='mode'}
    projection=VariableProjection(objective.linearize,initial,linear,lo,hi,rcond=settings.get('rank_rcond',1.e-10))
    nonlinear=projection.nonlinear
    def assess(theta):
        state=projection.evaluate(theta)
        return assess_statistical_convergence(state['residual'],state['joint_jacobian'],state['values'],lo,hi,
            data_count=len(objective.train),data_correction_transform=objective.whitening.standardized_transform,**settings)
    class Converged(Exception):
        def __init__(self,x):self.x=np.asarray(x).copy()
    def checked(theta):
        if assess(theta)['converged']:raise Converged(theta)
        return projection.evaluate(theta)['jacobian']
    trace=[];theta=np.asarray(initial)[nonlinear]
    if len(nonlinear):
        try:
            result=statistical_trust_region(lambda x:projection.evaluate(x)['residual'],checked,
                theta,lo[nonlinear],hi[nonlinear],np.array([v['scale'] for v in variables])[nonlinear],
                max_nfev=max_nfev,trace=trace,
                state_callback=lambda s:objective.context.promote_accepted(projection.evaluate(s['values'])['values']))
            theta=result.x
        except Converged as stop:theta=stop.x
    state=projection.evaluate(theta);values=state['values'];report=assess(theta)
    data=objective.context.evaluate(values);error=data['values'][objective.train]-objective.observed
    residual=state['residual'];weighted=objective.whitening.apply(error)
    return dict(success=bool(report['converged']),convergence=report,values=values.tolist(),rank=report['rank'],
        active_paths=paths,iteration_trace=trace,chi_square=float(residual@residual),data_chi_square=float(weighted@weighted),
        rms_residual_MHz=float(np.sqrt(np.mean(error**2))),frequencies_MHz=data['values'][objective.train].tolist(),
        jacobian=data['jacobian'][objective.train].tolist(),
        covariance_scope='full correlated QM penalties retained, including linear/nonlinear cross-correlation',
        variable_projection=dict(enabled=True,linear_paths=[paths[i] for i in linear],
            nonlinear_paths=[paths[i] for i in nonlinear],nonlinear_dimension=len(nonlinear),
            full_dimension=len(variables),linear_rank=state['linear_rank'],
            linear_active_bounds=state['active_linear_bounds'].tolist(),
            profile_evaluations=projection.evaluations,bounded_linear_solves=projection.bounded_solves,
            jacobian='exact constant-active-set response including the nonzero-residual term',
            scope='piecewise smooth at bound changes; no automatic rank truncation'))
