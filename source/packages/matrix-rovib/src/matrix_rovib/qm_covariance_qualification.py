"""Independent-reference qualification of a declared QM covariance model."""
from __future__ import annotations
import hashlib
import json
import numpy as np
from scipy.linalg import solve_triangular
from scipy.stats import beta,chi2
from .prediction_budget import _covariance


def qm_covariance_identity(names,units,covariance):
    matrix=_covariance(covariance,len(names),'QM coefficient')
    if len(units)!=len(names):raise ValueError('one unit per QM coefficient required')
    payload=dict(names=list(names),units=list(units),covariance=matrix.tolist())
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def qualify_qm_covariance(coefficient_names,coefficient_units,coefficient_covariance,*,
        reference_errors,reference_jacobians,independent_group_ids,predictive_covariances=None,
        training_group_ids=(),reference_covariances=None,reference_kind,
        reference_source_sha256,reference_surrogate_error_included=False,
        confidence=.95,significance=.05,minimum_groups=20):
    """Check a fixed Gaussian error model on independent held-out QC groups.

    Each row is one independent group and may hold multiple jointly correlated
    observables. Repeated geometries or isotopologues from the same family must
    be aggregated with their full covariance, not counted as independent rows.
    Covariance parameters must be frozen before validation. No scale or bias is
    fitted here; no experimental fit residual is used to manufacture a QM prior.
    """
    names=tuple(coefficient_names);units=tuple(coefficient_units)
    identity=qm_covariance_identity(names,units,coefficient_covariance)
    if not names or len(set(names))!=len(names) or not all(str(n).strip() for n in names) or not all(str(u).strip() for u in units):
        raise ValueError('unique coefficient names and explicit units required')
    if not np.isrealobj(reference_errors) or not np.isrealobj(reference_jacobians):
        raise ValueError('validation residuals and responses must be real')
    errors=np.asarray(reference_errors,float)
    groups=tuple(independent_group_ids)
    if errors.ndim!=2 or errors.shape[1]<1 or len(groups)!=len(errors) or not np.all(np.isfinite(errors)):
        raise ValueError('finite validation vectors and one independent group per row required')
    n,dimension=errors.shape
    jacobians=np.asarray(reference_jacobians,float)
    if jacobians.shape!=(n,dimension,len(names)) or not np.all(np.isfinite(jacobians)):
        raise ValueError('one finite response matrix from the declared QM coefficients per validation group required')
    coefficient_c=_covariance(coefficient_covariance,len(names),'QM coefficient')
    covariances=jacobians@coefficient_c@jacobians.swapaxes(-1,-2)
    if predictive_covariances is not None:
        supplied=np.asarray(predictive_covariances,float)
        if supplied.shape!=covariances.shape or not np.allclose(supplied,covariances,rtol=1e-10,atol=1e-14):
            raise ValueError('predictive covariance must derive from the declared coefficient covariance and responses')
    if len(set(groups))!=n or set(groups)&set(training_group_ids):
        raise ValueError('validation groups must be independent and disjoint from training')
    if (isinstance(minimum_groups,bool) or not isinstance(minimum_groups,int) or minimum_groups<3
        or not 0<confidence<1 or not 0<significance<1):
        raise ValueError('invalid calibration confidence, significance or group budget')
    if reference_kind not in {'ab_initio','validated_surrogate'}:
        raise ValueError('independent ab initio or explicitly validated surrogate references required')
    if not reference_source_sha256 or any(len(str(h))!=64 or any(c not in '0123456789abcdef' for c in str(h)) for h in reference_source_sha256.values()):
        raise ValueError('reference source SHA-256 identities required')
    known=np.zeros_like(covariances) if reference_covariances is None else np.asarray(reference_covariances,float)
    if known.shape!=covariances.shape:raise ValueError('reference covariance shape differs')
    standardized=[]
    for i in range(n):
        total=_covariance(covariances[i],dimension,'predictive')+_covariance(known[i],dimension,'reference')
        try:factor=np.linalg.cholesky(total)
        except np.linalg.LinAlgError as exc:
            raise ValueError('total validation covariance must be positive definite on the declared independent coordinates') from exc
        standardized.append(solve_triangular(factor,errors[i],lower=True))
    reasons=[]
    if n<minimum_groups:reasons.append('insufficient independent validation groups')
    if reference_kind=='validated_surrogate' and not reference_surrogate_error_included:
        reasons.append('surrogate reference error has not been included')
    result=dict(schema='matrix.lock.qm_covariance_qualification.v1',coefficient_names=list(names),
        coefficient_units=list(units),covariance_identity=identity,independent_validation_groups=n,
        minimum_groups=minimum_groups,reference_kind=reference_kind,reference_source_sha256=dict(reference_source_sha256),
        confidence=confidence,significance=significance,
        scope='empirical validation of a fixed Gaussian QM error model on declared independent held-out reference groups; conditional on reference quality and applicability')
    if not n:return dict(result,qualified=False,reasons=reasons or ['no independent validation groups'])
    z=np.asarray(standardized);distance=np.sum(z*z,axis=1)
    covered=int(sum(distance<=chi2.ppf(confidence,dimension)))
    # Bonferroni allocation for coverage, aggregate dispersion and aggregate bias.
    alpha=significance/3
    interval=[0. if covered==0 else float(beta.ppf(alpha/2,covered,n-covered+1)),
              1. if covered==n else float(beta.ppf(1-alpha/2,covered+1,n-covered))]
    dispersion=float(np.sum(distance));bias=float(n*np.sum(np.mean(z,axis=0)**2))
    dispersion_p=float(min(1.,2*min(chi2.cdf(dispersion,n*dimension),chi2.sf(dispersion,n*dimension))))
    bias_p=float(chi2.sf(bias,dimension))
    if not interval[0]<=confidence<=interval[1]:reasons.append('declared coverage outside held-out binomial interval')
    if dispersion_p<alpha:reasons.append('held-out dispersion incompatible with declared covariance')
    if bias_p<alpha:reasons.append('held-out bias incompatible with zero-mean QM error')
    return dict(result,qualified=not reasons,reasons=reasons,empirical_coverage=covered/n,
        coverage_confidence_interval=interval,standardized_rms=float(np.sqrt(dispersion/(n*dimension))),
        dispersion_p_value=dispersion_p,bias_p_value=bias_p,mahalanobis_squared=distance.tolist())


__all__=['qualify_qm_covariance','qm_covariance_identity']
