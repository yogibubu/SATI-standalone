"""Correlated experimental/QM information and prediction error decomposition."""
from __future__ import annotations

import numpy as np
from .prediction_budget import _covariance


def spin_fit_information(experimental_jacobian,qm_jacobian,*,parameter_scales=None):
    """Local Gaussian covariance with source decomposition, without double counting.

    Jacobians are whitened by their full respective observation covariance.
    Scales define the metric for rank and null-space tests. Fixed externally
    chosen QM means produce a conditional covariance; varying independent
    Gaussian QM targets gives the displayed QM source component. Experimental
    and QM rows are assumed independent after whitening their own blocks.
    """
    j=np.asarray(experimental_jacobian,float);q=np.asarray(qm_jacobian,float)
    if j.ndim!=2 or q.ndim!=2 or j.shape[1]!=q.shape[1] or not j.shape[1] or not all(np.all(np.isfinite(a)) for a in (j,q)):
        raise ValueError('finite experimental and QM Jacobians with common parameters required')
    n=j.shape[1];scale=np.ones(n) if parameter_scales is None else np.asarray(parameter_scales,float)
    if scale.shape!=(n,) or not np.all(np.isfinite(scale)) or np.any(scale<=0):raise ValueError('positive finite parameter scales required')
    js=j*scale;qs=q*scale;a=np.vstack((js,qs))
    _,s,vh=np.linalg.svd(a,full_matrices=True)
    tolerance=np.finfo(float).eps*max(a.shape)*max(s,default=0.)
    rank=int(np.sum(s>tolerance));vectors=vh[:rank].T
    c=(vectors/(s[:rank]**2))@vectors.T
    experimental=c@js.T@js@c;qm=c@qs.T@qs@c
    null=scale[:,None]*vh[rank:].T
    identified=np.linalg.norm(vh[rank:],axis=0)<=1e-10
    physical=lambda m:scale[:,None]*m*scale[None,:]
    diagonal=np.diag(c)
    fraction=np.divide(np.diag(qm),diagonal,out=np.zeros(n),where=diagonal>0)
    return dict(schema='matrix.lock.spin_fit_information.v1',rank=rank,parameter_count=n,
        covariance=physical(c).tolist(),experimental_covariance=physical(experimental).tolist(),
        qm_covariance=physical(qm).tolist(),null_space=null.tolist(),
        identifiable_parameters=identified.tolist(),
        qm_variance_fraction=[float(v) if good else None for v,good in zip(np.clip(fraction,0,1),identified,strict=True)],
        parameter_scales=scale.tolist(),absolute_observation_uncertainties=True,
        interpretation='local linear Gaussian decomposition; experimental/QM row blocks independent; no residual variance rescaling or QC model-bias claim')


def spin_prediction_uncertainty(predictions,jacobian,information,*,numerical_comparisons=None,
                                observable_units=None,external_qm_jacobian=None,
                                external_qm_covariance=None,parameter_qm_cross_covariance=None):
    """Propagate all covariance and add numerical differences as separate envelopes.

    Extra QM inputs must be distinct from fitted parameters/QM predicate rows.
    Cross covariance is propagated explicitly and the joint covariance must
    be PSD. Numerical reduction errors are not Gaussian standard deviations.
    Missing numerical checks remain unavailable, never silently zero.
    """
    y=np.asarray(predictions,float);g=np.asarray(jacobian,float)
    n=information['parameter_count']
    if y.ndim!=1 or g.shape!=(len(y),n) or not np.all(np.isfinite(y)) or not np.all(np.isfinite(g)):
        raise ValueError('finite predictions and matching parameter sensitivities required')
    c=_covariance(information['covariance'],n,'fitted parameter')
    ce=_covariance(information['experimental_covariance'],n,'experimental source')
    cq=_covariance(information['qm_covariance'],n,'QM source')
    null=np.asarray(information['null_space'],float).reshape(n,-1)
    scales=np.asarray(information['parameter_scales'])
    identified=np.linalg.norm(g@null,axis=1)<=1e-10*np.maximum(np.linalg.norm(g*scales,axis=1),np.finfo(float).tiny)
    exp=g@ce@g.T;qm=g@cq@g.T;total=g@c@g.T
    external=np.zeros_like(total);cross=np.zeros_like(total)
    if external_qm_jacobian is not None:
        k=np.asarray(external_qm_jacobian,float)
        if k.ndim!=2 or k.shape[0]!=len(y) or not np.all(np.isfinite(k)):raise ValueError('invalid external QM sensitivities')
        qc=_covariance(external_qm_covariance,k.shape[1],'external QM')
        pc=np.zeros((n,k.shape[1])) if parameter_qm_cross_covariance is None else np.asarray(parameter_qm_cross_covariance,float)
        if pc.shape!=(n,k.shape[1]) or not np.all(np.isfinite(pc)):raise ValueError('invalid parameter/external QM cross covariance')
        joint=np.block([[c,pc],[pc.T,qc]])
        _covariance(joint,len(joint),'joint parameter/external QM')
        external=k@qc@k.T;cross=g@pc@k.T+k@pc.T@g.T
        total+=external+cross
    elif external_qm_covariance is not None or parameter_qm_cross_covariance is not None:
        raise ValueError('external QM covariance requires its sensitivities')
    envelope=None;checks={}
    if numerical_comparisons is not None:
        if not numerical_comparisons:raise ValueError('provide nonempty numerical refinement comparisons')
        envelope=np.zeros(len(y))
        for name,delta in numerical_comparisons.items():
            values=np.asarray(delta,float)
            if values.shape!=y.shape or not np.all(np.isfinite(values)):raise ValueError('invalid numerical refinement differences')
            checks[name]=abs(values).tolist();envelope+=abs(values)
    sigma=np.sqrt(np.maximum(np.diag(total),0.))
    units=['unspecified']*len(y) if observable_units is None else list(observable_units)
    if len(units)!=len(y):raise ValueError('one unit per observable required')
    intervals=None if envelope is None else [[float(v-1.959963984540054*s-e),float(v+1.959963984540054*s+e)]
        if good else None for v,s,e,good in zip(y,sigma,envelope,identified,strict=True)]
    return dict(schema='matrix.lock.spin_prediction_uncertainty.v1',predictions=y.tolist(),observable_units=units,
        identifiable_predictions=identified.tolist(),experimental_covariance=exp.tolist(),qm_predicate_covariance=qm.tolist(),
        external_qm_covariance=external.tolist(),parameter_external_cross_covariance=cross.tolist(),
        statistical_covariance=total.tolist(),standard_uncertainty=[float(v) if good else None for v,good in zip(sigma,identified,strict=True)],
        numerical_envelope=None if envelope is None else envelope.tolist(),numerical_comparisons=checks,
        conditional_95_percent_intervals=intervals,complete=bool(envelope is not None and all(identified)),
        interpretation='local Gaussian prediction covariance plus additive refinement-difference envelope; conditional on supplied QC model and numerical checks; no calibrated model discrepancy or rigorous bound')
