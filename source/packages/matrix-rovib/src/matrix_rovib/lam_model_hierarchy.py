"""Nonlinear model selection through existing LAM qualification and fit engines."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Mapping, Sequence
import numpy as np
from matrix_numerics.least_squares import CovarianceWhitening
from .lam_fit import LAMFitResult, fit_lam_assigned_spectrum
from .lam_rotation import LAMRotationSurface
from .model_selection import ModelCandidateEvaluation, select_model_hierarchy


@dataclass(frozen=True)
class LAMModelCandidate:
    name: str
    surface: LAMRotationSurface
    request: Mapping


@dataclass(frozen=True)
class LAMModelHierarchyResult:
    selection: Mapping
    fits: Mapping[str, LAMFitResult]

    @property
    def selected_result(self):
        return self.fits.get(self.selection['selected'])

    def to_dict(self):
        return dict(schema='matrix.lock.lam_model_hierarchy.v1',
            selection=dict(self.selection), fits={k:v.to_dict() for k,v in self.fits.items()})


def fit_lam_model_hierarchy(candidates: Sequence[LAMModelCandidate], *,
                            validation_tolerance=0.0, allow_predicate_supported=False,
                            rank_rcond=1e-10, correlation_limit=.98):
    """Qualify and refit each declared physical model, then select the smallest adequate one.

    Candidate surfaces may come from TRINITY contractions or different kinetic
    and operator approximations. No surfaces or missing QM couplings are
    invented. Every model uses the same labeled measurements and uncertainties;
    its own numerical hierarchy must pass before and after fitting. This driver
    currently accepts assigned, resolved lines and linear Gaussian objectives.
    It does not share mutable optimizers or reinterpret a model's checkpoint.
    """
    candidates=tuple(candidates)
    if not candidates or len({c.name for c in candidates}) != len(candidates) or any(not c.name.strip() for c in candidates):
        raise ValueError('provide uniquely named physical model candidates')
    signature=None
    for c in candidates:
        options=c.request.get('options',{})
        if not options.get('basis_convergence') or options.get('loss','linear') != 'linear':
            raise ValueError('model hierarchy requires explicit basis convergence and linear Gaussian objectives')
        observations=c.request.get('observations',())
        if any('components' in row for row in observations):
            raise ValueError('nonlinear model hierarchy currently requires resolved assigned lines')
        labels=[row.get('label') for row in observations]
        if not labels or any(not isinstance(label,str) or not label for label in labels) or len(set(labels)) != len(labels):
            raise ValueError('model comparisons require unique common measurement labels')
        identity=tuple((row['label'],float(row['frequency_MHz']),float(row['uncertainty_MHz']),
            float(row.get('weight',1.)),row.get('role','training')) for row in observations)
        covariance=c.request.get('covariance_MHz2')
        contract=(identity,None if covariance is None else np.asarray(covariance,float).tolist())
        if signature is not None and contract != signature:
            raise ValueError('model candidates must use identical labeled observations, roles and covariance')
        signature=contract
    fits={};evaluations=[]
    for c in candidates:
        captured={}
        def capture(residual,jacobian,x,lower,upper,scales,*,data_count,**_):
            r=np.asarray(residual(x));j=np.asarray(jacobian(x))
            captured.update(residual=r[:data_count].copy(),data=j[:data_count].copy(),
                penalties=j[data_count:].copy(),scales=tuple(scales))
            return dict(schema='matrix.lock.model_selection_linearization.v1',training_count=data_count,
                penalty_count=len(r)-data_count)
        fit=fit_lam_assigned_spectrum(c.surface,deepcopy(dict(c.request)),postfit_diagnostic=capture)
        fits[c.name]=fit
        payload=fit.payload
        report=payload['basis_qualification'][-1]
        budget=report['numerical_budget']
        ratio=budget.get('maximum_budget_ratio')
        if ratio is None:
            last=max((row['round'] for row in report['history']),default=-1)
            ratio=max((row.get('maximum_frequency_change_MHz',0.)/report['frequency_tolerance_MHz']
                for row in report['history'] if row['round']==last),default=0.)
        if not report['converged']:
            ratio=max(2.,ratio)
        observations=c.request['observations']
        validation=np.array([row.get('role','training')=='validation' for row in observations])
        residual=np.array([row['calculated_minus_observed_MHz'] for row in payload['comparisons']])
        weights=np.array([row.get('weight',1.) for row in observations],float)
        sigma=np.array([row['uncertainty_MHz'] for row in observations])/np.sqrt(weights)
        covariance=c.request.get('covariance_MHz2')
        validation_covariance=(None if covariance is None else
            (np.asarray(covariance,float)/np.sqrt(weights[:,None]*weights[None,:]))[np.ix_(validation,validation)])
        validation_residual=(CovarianceWhitening(sigma[validation],validation_covariance).apply(residual[validation])
            if np.any(validation) else ())
        evaluations.append(ModelCandidateEvaluation(c.name,
            tuple(row['name'] for row in payload['variables']),tuple(captured['residual']),captured['data'],
            validation_residuals=tuple(validation_residual),parameter_scales=captured['scales'],
            predicate_jacobian=captured['penalties'],numerical_error_ratio=float(ratio),
            fit_converged=bool(payload['success'] and payload['convergence']['converged'])))
    selection=select_model_hierarchy(evaluations,require_qualification=True,
        validation_tolerance=validation_tolerance,allow_predicate_supported=allow_predicate_supported,
        rank_rcond=rank_rcond,correlation_limit=correlation_limit)
    return LAMModelHierarchyResult(selection,fits)


__all__=['LAMModelCandidate','LAMModelHierarchyResult','fit_lam_model_hierarchy']
