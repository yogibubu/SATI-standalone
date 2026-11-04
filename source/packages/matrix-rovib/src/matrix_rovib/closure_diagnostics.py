"""Separate numerical stationarity, conditional rank, and predictive residuals."""
from __future__ import annotations
import numpy as np
from .fit_convergence import assess_statistical_convergence


def diagnose_spectral_fit(residual, jacobian, values, lower, upper, scales, *,
                          data_count, parameter_names=None, rank_rcond=1.e-10):
    """Inputs are already whitened; theoretical rows follow experimental rows.

    Bound distances and sensitivity columns use the same declared parameter
    scales, so changing coefficient units cannot change the diagnosis. This
    routine never changes bounds, convergence tolerances, or observations.
    """
    r,j,x,lo,hi,scale=map(lambda a:np.asarray(a,dtype=float),
                        (residual,jacobian,values,lower,upper,scales))
    if scale.shape!=x.shape or np.any(scale<=0) or not np.all(np.isfinite(scale)):
        raise ValueError('positive finite scales are required')
    names=list(parameter_names) if parameter_names is not None else [str(i) for i in range(len(x))]
    if len(names)!=len(x) or len(set(names))!=len(names):
        raise ValueError('one unique name per parameter is required')
    stationarity=assess_statistical_convergence(r,j,x,lo,hi,data_count=data_count,rank_rcond=rank_rcond)
    def spectrum(a):
        _,s,vh=np.linalg.svd(a*scale,full_matrices=True)
        cutoff=(s[0]*rank_rcond if len(s) else 0.)
        rank=int(np.count_nonzero(s>cutoff))
        return dict(rank=rank,singular_values=s.tolist(),
                    unsupported_scaled_directions=vh[rank:].tolist())
    active=[]
    for name,value,l,u,s in zip(names,x,lo,hi,scale,strict=True):
        distances=((value-l)/s,(u-value)/s)
        active.append(dict(name=name,lower_distance_in_scales=None if not np.isfinite(l) else float(distances[0]),
                           upper_distance_in_scales=None if not np.isfinite(u) else float(distances[1]),
                           near_lower=bool(distances[0]<=1.e-6),near_upper=bool(distances[1]<=1.e-6)))
    step=np.asarray(stationarity['parameter_step'])
    return dict(schema='matrix.lock.fit_closure_diagnostics.v1',stationarity=stationarity,
                experimental=spectrum(j[:data_count]),augmented=spectrum(j),bounds=active,
                whitened_data_rms=float(np.sqrt(np.mean(r[:data_count]**2))),
                remaining_local_data_rms=float(np.sqrt(np.mean((r[:data_count]+j[:data_count]@step)**2))),
                scope='local linearized diagnosis; no global optimum or physical adequacy certification')


def grouped_spectral_residuals(observed, predicted, uncertainty, roles, groups):
    """Summarize user-supplied transition families without fitting holdout rows."""
    y,p,sigma=map(lambda a:np.asarray(a,float),(observed,predicted,uncertainty))
    roles,groups=list(roles),list(groups)
    if (y.ndim!=1 or not len(y) or p.shape!=y.shape or sigma.shape!=y.shape
        or len(roles)!=len(y) or len(groups)!=len(y)
        or not np.all(np.isfinite(np.r_[y,p,sigma])) or np.any(sigma<=0)):
        raise ValueError('inconsistent finite spectral rows and uncertainties')
    out=[]
    for role,group in sorted(set(zip(roles,groups))):
        indices=np.array([i for i in range(len(y)) if roles[i]==role and groups[i]==group])
        error=p[indices]-y[indices]
        out.append(dict(role=role,group=group,count=len(indices),mean_MHz=float(np.mean(error)),
                        rms_MHz=float(np.sqrt(np.mean(error**2))),
                        weighted_rms=float(np.sqrt(np.mean((error/sigma[indices])**2))),
                        maximum_absolute_MHz=float(np.max(abs(error)))))
    return out


def retained_space_continuity(reference_vectors, current_vectors, *, minimum_overlap=0.8):
    """Check loss of the original orthonormal LAM span on a common grid.

    A rectangular current span may contain additional states; it may not lose
    an original direction and still be described as the same differentiable
    contraction. This is a local numerical gate, not an assignment rule.
    """
    a,b=map(lambda v:np.asarray(v,complex),(reference_vectors,current_vectors))
    if (a.ndim!=2 or b.ndim!=2 or a.shape[0]!=b.shape[0] or not a.shape[1] or not b.shape[1]
        or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b))
        or not np.isfinite(minimum_overlap) or not 0<=minimum_overlap<=1):
        raise ValueError('finite common-grid spans and a valid overlap threshold are required')
    for v in (a,b):
        if not np.allclose(v.conj().T@v,np.eye(v.shape[1]),atol=1.e-8,rtol=1.e-8):
            raise ValueError('orthonormal retained spans are required')
    singular=np.linalg.svd(a.conj().T@b,compute_uv=False)
    minimum=0. if b.shape[1]<a.shape[1] else float(np.min(singular))
    return dict(continuous=minimum>=minimum_overlap,minimum_principal_overlap=minimum,
        threshold=minimum_overlap,reference_dimension=a.shape[1],current_dimension=b.shape[1],
        classification='compatible_retained_span' if minimum>=minimum_overlap else 'retained_space_changed')
