"""Analytic-Jacobian profiles and QM-weight sensitivity for any least-squares fit."""
from __future__ import annotations
import numpy as np
from .fit_convergence import assess_statistical_convergence, statistical_trust_region


def _covariance(jacobian, scales):
    a=np.asarray(jacobian)*scales
    _,s,vh=np.linalg.svd(a,full_matrices=False)
    keep=s>(s[0]*1.e-10 if s.size else 0.)
    inverse=vh[keep].T/s[keep]
    return scales[:,None]*(inverse@inverse.T)*scales[None,:],int(np.sum(keep))


def profile_and_predicate_sensitivity(fun,jac,point,lower,upper,scales,*,data_count,
        predicate_count,parameter_names,profile_count=2,max_nfev=24,initial_radius=.1,
        progress=None, resume=None, data_correction_transform=None):
    """Profile the most QM-sensitive coefficients and vary all QM widths.

    Nuisance coefficients are reoptimized with analytic derivatives and exact
    fixed-coordinate elimination. No molecular labels, finite differences,
    artificially tight bounds or data exclusions enter the algorithm. Costs
    include declared penalties; profiles are not automatically confidence limits.
    """
    x=np.asarray(point,float);lo=np.asarray(lower,float);hi=np.asarray(upper,float);scale=np.asarray(scales,float)
    base=np.asarray(fun(x));j=np.asarray(jac(x));n=int(data_count);nq=int(predicate_count)
    if len(parameter_names)!=len(x) or max_nfev<1 or profile_count<0:
        raise ValueError('invalid profile registry or evaluation budget')
    if not 0<=nq<=len(base)-n or n<1:raise ValueError('invalid training/predicate row inventory')
    baseline_assessment=assess_statistical_convergence(base,j,x,lo,hi,data_count=n,data_correction_transform=data_correction_transform)
    if not baseline_assessment['converged']:
        raise ValueError('profiles require a statistically stationary baseline')
    cov,rank=_covariance(j,scale);data_cov,data_rank=_covariance(j[:n],scale)
    if rank!=len(x):raise ValueError('nonlinear profiles require full augmented rank')
    nonqm=j[:-nq] if nq else j
    nonqm_cov,nonqm_rank=_covariance(nonqm,scale)
    dependence=[]
    for i,name in enumerate(parameter_names):
        error=np.sqrt(max(cov[i,i],0.))
        dependence.append(dict(index=i,parameter=name,augmented_error=float(error),
            data_error=float(np.sqrt(max(data_cov[i,i],0.))) if data_rank==len(x) else None,
            without_qm_error=float(np.sqrt(max(nonqm_cov[i,i],0.))) if nonqm_rank==len(x) else None,
            data_to_augmented_error_ratio=float(np.sqrt(max(data_cov[i,i],0.))/error) if data_rank==len(x) and error>0 else None))
    candidates=[d for d in dependence if nq and np.any(j[-nq:,d['index']]!=0)]
    if data_rank==len(x):candidates.sort(key=lambda d:-(d['data_to_augmented_error_ratio'] or 0.))
    else:candidates.sort(key=lambda d:-d['augmented_error']/scale[d['index']])
    report=dict(schema='matrix.lock.penalized_profile_sensitivity.v1',baseline_chi_square=float(base@base),
        data_rank=data_rank,without_qm_rank=nonqm_rank,augmented_rank=rank,parameter_dependence=dependence,
        selection='QM-constrained coordinates with greatest data-only/augmented standard-error ratio; if data rank is deficient, largest scaled augmented error.',
        profiles=[],qm_width_sensitivity=[],complete=False,
        interpretation='Conditional penalized-objective profiles, not calibrated likelihood intervals. Nuisance parameters are reoptimized; validation rows never enter these objectives. Numerical rank alone does not establish individual physical identifiability.')
    class Stop(Exception):
        def __init__(self,values,assessment):self.values=values.copy();self.assessment=assessment
    def optimize(start,weights,fixed=None):
        mask=np.ones(len(x),bool)
        if fixed is not None:mask[fixed[0]]=False
        template=np.array(start,copy=True)
        if fixed is not None:template[fixed[0]]=fixed[1]
        def assemble(y):
            trial=template.copy();trial[mask]=y;return trial
        def residual(y):return np.asarray(fun(assemble(y)))*weights
        last={}
        def checked(y):
            a=np.asarray(jac(assemble(y)))[:,mask]*weights[:,None]
            d=assess_statistical_convergence(residual(y),a,y,lo[mask],hi[mask],data_count=n,data_correction_transform=data_correction_transform)
            last.update(d)
            if d['converged']:raise Stop(np.asarray(y),d)
            return a
        try:
            if not np.any(mask):
                raise Stop(np.empty(0),dict(converged=True,classification='all_parameters_fixed'))
            fit=statistical_trust_region(residual,checked,template[mask],lo[mask],hi[mask],scale[mask],
                                         max_nfev=max_nfev,initial_radius=initial_radius)
            values=assemble(fit.x);converged=False;termination=str(fit.message)
        except Stop as stopped:
            values=assemble(stopped.values);converged=True;termination='statistical_stationarity'
        raw=np.asarray(fun(values));cost=float((raw*weights)@(raw*weights))
        return dict(values=values.tolist(),converged=converged,termination=termination,
                    assessment=last,chi_square=cost,training_chi_square=float(raw[:n]@raw[:n]),
                    maximum_training_prediction_shift_sigma=float(np.max(abs(raw[:n]-base[:n]))))
    for d in candidates[:profile_count]:
        i=d['index'];error=d['augmented_error']
        for offset in (-1.,1.):
            target=x[i]+offset*error
            if not lo[i]<=target<=hi[i]:
                row=dict(parameter=parameter_names[i],sigma_offset=offset,status='outside_bounds')
            else:
                # Conditional linearized optimum supplies a cheap warm start.
                start=np.clip(x+cov[:,i]/cov[i,i]*(target-x[i]),lo,hi)
                previous=next((p for p in (resume or {}).get('profiles',[]) if p.get('parameter')==parameter_names[i] and p.get('target')==float(target) and p.get('converged')),None)
                if previous is not None:
                    report['profiles'].append(previous)
                    if progress is not None:progress(report)
                    continue
                result=optimize(start,np.ones(len(base)),fixed=(i,target))
                row=dict(parameter=parameter_names[i],target=float(target),sigma_offset=offset,**result)
                row['delta_chi_square']=row['chi_square']-report['baseline_chi_square']
            report['profiles'].append(row)
            if progress is not None:progress(report)
    if nq:
        for factor in (.5,2.):
            previous=next((p for p in (resume or {}).get('qm_width_sensitivity',[]) if p.get('qm_sigma_factor')==factor and p.get('converged')),None)
            if previous is not None:
                report['qm_width_sensitivity'].append(previous)
                if progress is not None:progress(report)
                continue
            weights=np.ones(len(base));weights[-nq:]/=factor
            # The reweighted linear problem supplies the nonlinear warm start.
            step=assess_statistical_convergence(base*weights,j*weights[:,None],x,lo,hi,data_count=n,data_correction_transform=data_correction_transform)['parameter_step']
            start=np.clip(x+np.asarray(step),lo,hi)
            result=optimize(start,weights)
            result['original_penalized_chi_square']=float(np.asarray(fun(result['values']))@np.asarray(fun(result['values'])))
            report['qm_width_sensitivity'].append(dict(qm_sigma_factor=factor,**result))
            if progress is not None:progress(report)
    report['complete']=all(p.get('converged',False) for p in report['profiles']+report['qm_width_sensitivity'])
    if progress is not None:progress(report)
    return report
