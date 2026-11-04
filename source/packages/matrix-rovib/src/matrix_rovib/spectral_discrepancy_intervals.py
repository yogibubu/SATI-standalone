"""Spectroscopic discrepancy intervals with disjoint physical blocks and support checks."""
from __future__ import annotations

import math
from collections.abc import Mapping
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import linprog
from scipy.stats import norm
from .prediction_budget import _covariance
from .structured_discrepancy import _kernel,structured_model_discrepancy


def spectroscopic_discrepancy_features(observations):
    """Describe arbitrary assigned spectra; no molecule or parameter-name rules.

    Common assigned J/N, K/Ka/Kc and LAM/v labels become feature columns.
    Missing optional labels are reported rather than assigned fictitious zeros.
    Default physical blocks share max(J_lower,J_upper). Custom experimental
    blocks/features can replace these descriptors in the interval API.
    """
    observations=tuple(observations)
    if not observations:raise ValueError('nonempty assigned observations required')
    if any(not isinstance(row.get(side),Mapping) for row in observations for side in ('lower','upper')):
        raise ValueError('explicit discrepancy features are required for blends or unassigned states')
    def get(row,side,aliases):
        selector=row[side]
        return next((selector[k] for k in aliases if k in selector),None)
    fields=[('J',('J','N')),('Ka',('Ka','K')),('Kc',('Kc',)),('v',('torsional_state','v'))]
    columns=[];names=[];missing=[]
    for field,aliases in fields:
        for side in ('lower','upper'):
            values=[get(row,side,aliases) for row in observations]
            name=side+'.'+field
            if any(v is None for v in values):
                if field=='J':raise ValueError('assigned J or N is needed; provide explicit features otherwise')
                missing.append(name);continue
            column=np.asarray(values,float)
            if not np.all(np.isfinite(column)):raise ValueError('nonfinite spectral quantum labels')
            columns.append(column);names.append(name)
    x=np.column_stack(columns)
    jl=names.index('lower.J');ju=names.index('upper.J')
    families=[];blocks=[]
    for i,row in enumerate(observations):
        parts=[]
        for side in ('lower','upper'):
            v=get(row,side,('torsional_state','v'))
            symmetry=row[side].get('lam_symmetry',row[side].get('symmetry','unspecified'))
            parts.append(f'{v if v is not None else "unspecified"}:{symmetry}')
        system=row.get('system',row.get('dataset'))
        prefix='' if system is None else str(system)+'|'
        families.append(prefix+'->'.join(parts))
        blocks.append(prefix+str(row.get('source_block',f'Jmax={max(x[i,jl],x[i,ju]):g}')))
    return dict(features=x,feature_names=names,groups=np.asarray(families),block_keys=np.asarray(blocks),
        unavailable_features=missing,default_block_rule='explicit source_block else shared max(J/N)')


def _indices(values,n,name,minimum=0):
    raw=list(values)
    if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) for v in raw):
        raise ValueError(f'{name} must contain integer row indices')
    result=np.asarray(raw,int)
    if len(result)<minimum or len(set(raw))!=len(raw) or np.any(result<0) or np.any(result>=n):
        raise ValueError(f'invalid or insufficient {name}')
    return result


def _domain_support(x,groups,cal,lengths):
    supported=[];reasons=[];distance=[]
    for row,group in zip(x,groups,strict=True):
        same=cal[groups[cal]==group]
        if not len(same):
            supported.append(False);reasons.append('unseen_transition_family');distance.append(None);continue
        points=x[same]/lengths;target=row/lengths
        distance.append(float(np.min(np.linalg.norm(points-target,axis=1))))
        # Convex support includes correlations between quantum labels. A box
        # alone would falsely admit a point off a sampled affine manifold.
        origin=points.mean(axis=0);points=points-origin;target=target-origin
        equality=np.vstack((points.T,np.ones(len(same))))
        feasible=linprog(np.zeros(len(same)),A_eq=equality,b_eq=np.r_[target,1.],bounds=(0,None),method='highs')
        good=bool(feasible.success and np.max(abs(equality@feasible.x-np.r_[target,1.]))<=1e-7)
        supported.append(good);reasons.append('interpolation' if good else 'outside_calibration_convex_support')
    return np.asarray(supported),reasons,distance


def spectral_discrepancy_intervals(features,residuals,known_covariance,predictions,*,
        calibration_indices,validation_indices=(),training_indices=(),block_keys,
        groups=None,lengthscales=None,feature_names=None,numerical_envelope_MHz=None,
        confidence=.95,validation_status='retrospective',return_covariance=False,
        memory_budget_bytes=512*1024**2):
    """Calibrate only on rows excluded from the physical fit and validation.

    known_covariance is the *residual* covariance, including experimental noise,
    fitted-parameter uncertainty and their cross terms where relevant. Calibration
    conditions this covariance jointly with the discrepancy; no independent
    quadrature addition or second use of parameter uncertainty is made.
    All rows may include future queries with NaN residuals. Only declared
    calibration and validation residuals are inspected. Intervals target measured
    frequencies; the Hamiltonian, fitted parameters and physical centers are fixed.
    """
    x=np.asarray(features,float);r=np.asarray(residuals,float);y=np.asarray(predictions,float)
    if (x.ndim!=2 or not len(x) or not x.shape[1] or r.shape!=(len(x),) or y.shape!=r.shape
        or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or not np.isfinite(confidence) or not 0<confidence<1):
        raise ValueError('finite spectral features/predictions and a valid confidence required')
    n=len(x)
    if 8*n*n*20>memory_budget_bytes:raise MemoryError('spectral discrepancy exceeds dense covariance workspace budget')
    cal=_indices(calibration_indices,n,'calibration rows',3);val=_indices(validation_indices,n,'validation rows')
    train=_indices(training_indices,n,'physical-fit rows')
    if set(cal)&set(val) or set(train)&(set(cal)|set(val)):
        raise ValueError('physical fit, calibration and validation rows must be disjoint')
    keys=np.asarray(block_keys).astype(str)
    if keys.shape!=(n,):raise ValueError('one physical block key per row required')
    if (set(keys[cal])&set(keys[val]) or set(keys[train])&(set(keys[cal])|set(keys[val]))):
        raise ValueError('physical blocks must be disjoint across fit/calibration/validation')
    if not np.all(np.isfinite(r[np.r_[cal,val]])):raise ValueError('finite calibration/validation residuals required')
    if validation_status not in ('prospective','retrospective'):
        raise ValueError('validation status must record prospective or retrospective evidence')
    group=np.zeros(n,dtype=str) if groups is None else np.asarray(groups).astype(str)
    if group.shape!=(n,):raise ValueError('one transition family per row required')
    names=[f'feature_{i}' for i in range(x.shape[1])] if feature_names is None else list(feature_names)
    if len(names)!=x.shape[1]:raise ValueError('one feature name per column required')
    if lengthscales is None:
        lengths=np.maximum(np.ptp(x[cal],axis=0),1.)
        length_rule='calibration feature ranges with one quantum-number unit floor; no validation residuals'
    else:
        lengths=np.asarray(lengthscales,float);length_rule='declared before validation'
    if lengths.shape!=(x.shape[1],) or not np.all(np.isfinite(lengths)) or np.any(lengths<=0):
        raise ValueError('positive finite kernel lengthscales required')
    known=_covariance(known_covariance,n,'known spectral residual')
    clean=np.zeros(n);clean[cal]=r[cal]
    # Reuse the existing GP amplitude estimation; evaluation rows are excluded.
    fitted=structured_model_discrepancy(x,clean,known,x,lengthscales=lengths,groups=group,
        prediction_groups=group,calibration_indices=cal,block_keys=keys)
    variance=fitted['variance_MHz2']
    kernel=variance*_kernel(x,x,lengths,group,group)
    total=known+kernel;factor=cho_factor(total[np.ix_(cal,cal)],lower=True)
    cross=total[:,cal];weights=cho_solve(factor,r[cal])
    mean=cross@weights
    gain=cross@cho_solve(factor,cross.T)
    conditional=.5*(total+total.T-gain-gain.T)
    sigma=np.sqrt(np.maximum(np.diag(conditional),0.))
    supported,reasons,distances=_domain_support(x,group,cal,lengths)
    supported[train]=False
    for i in train:reasons[i]='used_in_physical_fit'
    envelope=np.zeros(n) if numerical_envelope_MHz is None else np.asarray(numerical_envelope_MHz,float)
    if envelope.shape!=(n,) or not np.all(np.isfinite(envelope)) or np.any(envelope<0):
        raise ValueError('finite nonnegative numerical envelopes required')
    z=float(norm.ppf((1+confidence)/2));width=abs(mean)+z*sigma+envelope
    intervals=[[float(a-b),float(a+b)] if good else None for a,b,good in zip(y,width,supported,strict=True)]
    conditional_intervals=[[float(a-b),float(a+b)] for a,b in zip(y,width,strict=True)]
    # Descriptive trends use calibration rows only and do not select a kernel.
    centered=(x[cal]-x[cal].mean(axis=0))/lengths
    design=np.column_stack((np.ones(len(cal)),centered))
    solved=cho_solve(factor,design);coef=np.linalg.pinv(design.T@solved)@(design.T@weights)
    families=[]
    for label in sorted(set(group[cal])):
        ids=cal[group[cal]==label]
        families.append(dict(family=str(label),count=len(ids),mean_MHz=float(np.mean(r[ids])),rms_MHz=float(np.sqrt(np.mean(r[ids]**2)))))
    # Raw block-max scores do not reuse a GP fitted on these same scores.
    scale=np.sqrt(np.diag(known)[cal])
    if np.any(scale<=0):raise ValueError('positive marginal known calibration uncertainty required')
    scores=abs(r[cal])/scale
    blocks=sorted(set(keys[cal]));block_scores=[float(np.max(scores[keys[cal]==b])) for b in blocks]
    order=math.ceil((len(blocks)+1)*confidence)
    finite=order<=len(blocks)
    quantile=float(np.sort(block_scores)[order-1]) if finite else None
    calibration=dict(block_count=len(blocks),order_statistic=order,confidence=confidence,
        finite_interval_available=finite,quantile=quantile,
        reason='finite order statistic' if finite else 'insufficient independent calibration blocks for a finite requested-level interval',
        minimum_blocks_for_finite_interval=math.ceil(confidence/(1-confidence)-1e-12),
        block_labels=blocks,raw_block_scores=block_scores,
        assumption='exchangeable whole-block max scores; not established by quantum labels or retrospective coverage',
        uses_GP_fitted_scores=False)
    validation=None
    if len(val):
        good=val[supported[val]];all_coverage=abs(r[val])<=width[val]
        validation=dict(status=validation_status,indices=val.tolist(),count=len(val),supported_count=len(good),
            physical_centered_conditional_coverage_all=float(np.mean(all_coverage)),
            supported_coverage=None if not len(good) else float(np.mean(abs(r[good])<=width[good])),
            raw_rms_MHz=float(np.sqrt(np.mean(r[val]**2))),
            conditional_error_rms_MHz=float(np.sqrt(np.mean((r[val]-mean[val])**2))),
            interval_interpretation='GP conditional diagnostic including unsupported extrapolations in all-row coverage; not a new coverage guarantee')
    report=dict(schema='matrix.lock.spectral_discrepancy_intervals.v1',calibration_indices=cal.tolist(),
        training_indices=train.tolist(),validation_indices=val.tolist(),confidence=confidence,
        physical_predictions_MHz=y.tolist(),physical_predictions_corrected=False,
        conditional_residual_mean_MHz=mean.tolist(),model_error_conditional_mean_MHz=fitted['model_error_mean_MHz'],
        known_error_conditional_mean_MHz=(known[:,cal]@weights).tolist(),conditional_standard_uncertainty_MHz=sigma.tolist(),
        conditional_intervals_MHz=conditional_intervals,supported_intervals_MHz=intervals,
        supported_predictions=supported.tolist(),support_reasons=reasons,nearest_calibration_distance=distances,
        model_discrepancy_variance_MHz2=variance,lengthscales=lengths.tolist(),lengthscale_rule=length_rule,
        numerical_envelope_MHz=None if numerical_envelope_MHz is None else envelope.tolist(),
        calibration_diagnostics=dict(feature_names=names,GLS_coefficients_scaled=coef.tolist(),
            design_rank=int(np.linalg.matrix_rank(design)),individual_trends_identified=bool(np.linalg.matrix_rank(design)==design.shape[1]),family_residuals=families,
            interpretation='descriptive calibration-only trends; no multiple-testing significance or Hamiltonian modification'),
        block_calibration=calibration,validation=validation,
        block_score_intervals_MHz=None if not finite else [[float(a-quantile*s-e),float(a+quantile*s+e)]
            if i not in set(train) else None for i,(a,s,e) in enumerate(zip(y,np.sqrt(np.diag(known)),envelope,strict=True))],
        interval_scope='physical-centered conditional GP intervals; abs conditional bias plus Gaussian width plus supplied numerical envelope; hyperparameter uncertainty excluded',
        coverage_guaranteed=False,complete=bool(numerical_envelope_MHz is not None and np.all(supported[val]) and len(val)>0 and finite))
    if return_covariance:
        kc=known[:,cal];dc=kernel[:,cal]
        a=kc@cho_solve(factor,kc.T);b=dc@cho_solve(factor,dc.T)
        mixed=-kc@cho_solve(factor,dc.T)-dc@cho_solve(factor,kc.T)
        report['conditional_covariance_MHz2']=conditional.tolist()
        report['conditional_known_covariance_MHz2']=(known-a).tolist()
        report['conditional_model_covariance_MHz2']=(kernel-b).tolist()
        report['conditional_known_model_cross_covariance_MHz2']=mixed.tolist()
    return report
