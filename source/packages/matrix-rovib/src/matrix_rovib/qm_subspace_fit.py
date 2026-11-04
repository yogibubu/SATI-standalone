"""Nonlinear refinement of experimental and QM-supported combinations."""
from __future__ import annotations

import numpy as np
from scipy.optimize import OptimizeResult
from .fit_convergence import assess_statistical_convergence, statistical_trust_region
from .parameter_selection import select_spectral_parameters


def fit_identifiable_qm_subspaces(fun, jac, values, lower, upper, scales, *,
                                  data_count, max_nfev=100, rank_rcond=1e-10,
                                  minimum_singular_value=1., joint_tolerance=.1,
                                  line_tolerance=.01, data_correction_transform=None,
                                  accepted_callback=None, parameter_names=None):
    """Refine all combinations with the shared MORPHEUS bounded SVD LM kernel.

    Leading rows are covariance-whitened experimental residuals; remaining
    rows are explicit Gaussian priors/QM predicates. The experimental SVD and
    its orthogonal complement span every scaled physical parameter. Weak
    combinations must have declared prior support. The default information
    floor is one experimental sigma per physical scale increment; zero selects
    algebraic rank only. No physical parameter or objective term is discarded.

    An orthogonal rotation of scaled coordinates leaves the physical trust
    metric invariant. Solve in physical coordinates to impose the original
    box bounds exactly, and report steps in the experimental/weak basis. This
    reuses the existing compiled LM algebra and avoids inverse-information
    scaling that can impede convergence in curved spectroscopic valleys.
    """
    x0=np.asarray(values,float);scale=np.asarray(scales,float)
    lo,hi=np.asarray(lower,float),np.asarray(upper,float)
    if (x0.ndim!=1 or not len(x0) or scale.shape!=x0.shape or lo.shape!=x0.shape or hi.shape!=x0.shape
        or not np.all(np.isfinite(x0)) or not np.all(np.isfinite(scale)) or np.any(scale<=0)
        or np.any(np.isnan(lo)) or np.any(np.isnan(hi)) or np.any(lo>=hi) or np.any(x0<lo) or np.any(x0>hi)
        or isinstance(max_nfev,bool) or not isinstance(max_nfev,int) or max_nfev<1
        or not np.isfinite(rank_rcond) or not 0<rank_rcond<1
        or not np.isfinite(minimum_singular_value) or minimum_singular_value<0):
        raise ValueError('invalid QM subspace fit geometry, scales or budgets')
    n=len(x0)
    names=tuple(f'parameter-{k}' for k in range(n)) if parameter_names is None else tuple(parameter_names)
    if len(names)!=n or len(set(names))!=n:
        raise ValueError('one unique physical name is required per parameter')
    counts=dict(nfev=0,njev=0);cache={};history=[];accepted=x0.copy();accepted_r=None;accepted_j=None
    def evaluate(x,derivatives=False):
        key=np.ascontiguousarray(x,dtype=float).tobytes()
        if cache.get('key')!=key:
            residual=np.asarray(fun(x),float)
            if residual.ndim!=1 or not np.all(np.isfinite(residual)):
                raise ValueError('nonfinite or inconsistent nonlinear residual')
            counts['nfev']+=1
            cache.clear();cache.update(key=key,residual=residual,jacobian=None)
        if derivatives and cache['jacobian'] is None:
            j=np.asarray(jac(x),float)
            if j.shape!=(len(cache['residual']),n) or not np.all(np.isfinite(j)):
                raise ValueError('nonfinite or inconsistent nonlinear Jacobian')
            cache['jacobian']=j;counts['njev']+=1
        return cache['residual'],cache['jacobian']
    initial_r,initial_j=evaluate(x0,True)
    accepted_r,accepted_j=initial_r.copy(),initial_j.copy()
    if isinstance(data_count,bool) or not isinstance(data_count,(int,np.integer)) or not 0<data_count<=len(initial_r):
        raise ValueError('data_count must select the leading experimental rows')
    data=initial_j[:data_count]*scale
    _,singular,vh=np.linalg.svd(data,full_matrices=data.shape[0]<n)
    relative_cutoff=rank_rcond*singular[0] if len(singular) else 0.
    rank=int(np.count_nonzero((singular>relative_cutoff)&(singular>=minimum_singular_value)))
    if not rank:raise ValueError('no experimentally supported spectral combination')
    observed=vh[:rank].T;weak=vh[rank:].T
    if weak.shape[1]:
        penalty=initial_j[data_count:]*scale
        prior_s=np.linalg.svd(penalty@weak,compute_uv=False)
        if len(prior_s)<weak.shape[1] or np.any(prior_s<=rank_rcond*prior_s[0]):
            raise ValueError('QM predicates do not resolve every weak experimental combination')
    else:prior_s=np.empty(0)
    rotation=np.column_stack((observed,weak))
    transform=scale[:,None]*rotation
    if not np.all(np.isfinite(transform)):
        raise ValueError('physical parameter scales exceed numerical range')
    def assess(x,r,j):
        return assess_statistical_convergence(r,j,x,lo,hi,data_count=data_count,
            rank_rcond=rank_rcond,joint_tolerance=joint_tolerance,line_tolerance=line_tolerance,
            data_correction_transform=data_correction_transform)
    class Stationary(Exception):
        def __init__(self,x,r,j):self.x,self.r,self.j=x.copy(),r.copy(),j.copy()
    def checked_jacobian(x):
        nonlocal accepted_r,accepted_j
        r,j=evaluate(x,True)
        accepted_r,accepted_j=r.copy(),j.copy()
        if assess(x,r,j)['converged']:raise Stationary(x,r,j)
        return j
    def save(state):
        nonlocal accepted,accepted_r,accepted_j
        x=np.asarray(state['values'],float)
        step=rotation.T@((x-accepted)/scale)
        same=np.array_equal(x,accepted)
        accepted=x.copy()
        # The kernel calls save only for accepted steps, immediately after fun.
        accepted_r=cache['residual'].copy()
        if not same:accepted_j=None
        if accepted_callback is not None:accepted_callback(accepted)
        history.append(dict(iteration=len(history),chi_square=float(accepted_r@accepted_r),
            observable_step_norm=float(np.linalg.norm(step[:rank])),
            qm_supported_step_norm=float(np.linalg.norm(step[rank:]))))
    try:
        result=statistical_trust_region(lambda x:evaluate(x)[0],checked_jacobian,
            x0,lo,hi,scale,max_nfev=max_nfev,state_callback=save)
        accepted=np.asarray(result.x,float)
        status,message=int(result.status),str(result.message)
    except Stationary as stopped:
        accepted,accepted_r,accepted_j=stopped.x,stopped.r,stopped.j
        status,message=5,'Statistical stationarity reached.'
    # Rejected trials cannot displace the accepted linearization. At a newly
    # accepted budget endpoint only its Jacobian is missing, not a new solve.
    if accepted_j is None:
        accepted_j=np.asarray(jac(accepted),float);counts['njev']+=1
    assessment=assess(accepted,accepted_r,accepted_j)
    if assessment['converged']:
        status,message=5,'Statistical stationarity reached.'
    info=select_spectral_parameters(accepted_j[:data_count],names,scale,
        augmented_jacobian=accepted_j,rcond=rank_rcond)
    return OptimizeResult(x=accepted,success=bool(assessment['converged']),status=status,message=message,
        nfev=counts['nfev'],njev=counts['njev'],cost=.5*float(accepted_r@accepted_r),
        optimality=float(np.linalg.norm(accepted_j.T@accepted_r,ord=np.inf)),
        convergence=assessment,subspace_report=dict(schema='matrix.lock.qm_subspace_fit.v1',
            experimental_rank=rank,weak_dimension=n-rank,
            experimental_absolute_singular_floor=minimum_singular_value,
            experimental_scaled_directions=observed.T.tolist(),weak_scaled_directions=weak.T.tolist(),
            experimental_singular_values=singular.tolist(),qm_weak_singular_values=prior_s.tolist(),
            physical_transform=transform.tolist(),final_information=info,
            trust_metric='declared physical scales; orthogonal combination rotation is invariant',
            optimizer_kernel='shared MORPHEUS bounded SVD Levenberg-Marquardt',
            all_physical_parameters_refined=True,objective_changed=False,physical_bounds_preserved=True,
            accepted_linearization_retained=True,history=history,
            restart_scope='native accepted-workspace replay; inner LM state is not serialized for this method'))


__all__=['fit_identifiable_qm_subspaces']
