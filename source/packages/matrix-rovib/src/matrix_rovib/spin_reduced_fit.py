"""Analytic fixed-Galerkin coupled-spin spectra and shared nonlinear fitting."""
from __future__ import annotations

from dataclasses import replace
import numpy as np
from scipy.optimize import linear_sum_assignment

from .spin_effective_response import canonical_spin_response,SpinReductionGapError


def native_spin_parameter_matrices(surface,lam_result,model,*,two_f_values,parameters,
                                   memory_budget_bytes=512*1024**2):
    """Exact linear physical parameter operators, not finite differences.

    A parameter is a native spectral-fit spin path, or {surface_key, field}
    for an explicit coordinate-field direction. Existing constant/surface
    override semantics are preserved. Spin-free energies are omitted during
    unit-operator assembly, avoiding subtraction of large energy matrices.
    The native LAM/Galerkin basis is held fixed for this inner fit.
    """
    from .spin_hyperfine import ElectronSpin,NuclearSpin,NuclearSpinSpinCoupling,SpinHyperfineModel
    from .spectral_fit import _set_parameter
    from .native_spin_effective import build_native_lam_spin_operator
    zero=SpinHyperfineModel(electron=ElectronSpin(two_S=model.electron.two_S),
        nuclei=tuple(NuclearSpin(n.label,n.two_I) for n in model.nuclei),
        nuclear_spin_spin=tuple(NuclearSpinSpinCoupling(p.left,p.right) for p in model.nuclear_spin_spin))
    empty=replace(surface,spin_tensor_surfaces_MHz={key:np.zeros_like(value)
        for key,value in (surface.spin_tensor_surfaces_MHz or {}).items()})
    fs=tuple(two_f_values);output={f:[] for f in fs}
    for parameter in parameters:
        if isinstance(parameter,str):
            if parameter.startswith('rotor.'):
                raise ValueError('supply complete physical matrix directions for rotor/PES changes')
            if parameter.startswith('electron.spin_lam_momentum_vectors_MHz.'):
                parts=parameter.split('.')
                if len(parts)!=4 or parts[3] not in ('a','b','c'):
                    raise ValueError('spin-LAM path requires coordinate index and a/b/c axis')
                vectors=np.zeros((surface.dimension,3));mu=int(parts[2])
                if not 0<=mu<surface.dimension:raise ValueError('invalid spin-LAM coordinate index')
                vectors[mu,('a','b','c').index(parts[3])]=1.
                unit=replace(zero,electron=replace(zero.electron,spin_lam_momentum_vectors_MHz=tuple(map(tuple,vectors))))
            else:
                _,unit=_set_parameter(None,zero,parameter,1.)
            unit_surface=empty
        else:
            key=parameter['surface_key'];field=np.asarray(parameter['field'])
            unit=zero;unit_surface=replace(surface,spin_tensor_surfaces_MHz={key:field})
        for f in fs:
            output[f].append(build_native_lam_spin_operator(unit_surface,lam_result,unit,two_F=f,
                memory_budget_bytes=memory_budget_bytes,_include_spin_free=False).hamiltonian_MHz)
            if sum(v.nbytes for values in output.values() for v in values)>memory_budget_bytes/4:
                raise MemoryError('native spin parameter operator registry exceeds memory budget')
    return {f:np.stack(values) for f,values in output.items()}


def _selected_group(energies,indices):
    selected={int(indices)} if np.isscalar(indices) else set(map(int,indices))
    if not selected or min(selected)<0 or max(selected)>=len(energies):
        raise ValueError('measurement selector outside the tracked spectrum')
    tol=128*np.finfo(float).eps*max(1.,np.max(abs(energies),initial=0.))
    # A degenerate state has a projector observable; choosing one arbitrary
    # partner would give gauge-dependent strengths and derivatives.
    return np.array(sorted({j for i in selected for j,e in enumerate(energies)
                            if abs(e-energies[i])<=tol}),int)


def _projector_response(e,u,dh,ids):
    n=len(e);outside=np.array([i for i in range(n) if i not in set(ids)],int)
    selected=u[:,ids]
    response=np.zeros((len(dh),len(u),len(ids)),complex)
    if len(outside):
        gap=e[ids][None,:]-e[outside][:,None]
        coupling=u[:,outside].conj().T@dh@selected
        response=u[:,outside]@(coupling/gap)
    centroid_derivative=np.trace(selected.conj().T@dh@selected,axis1=-2,axis2=-1).real/len(ids)
    return selected,response,centroid_derivative


class NativeSpinReducedEvaluator:
    """Reusable analytic spectra in qualified, frozen model projectors.

    producer(x) returns (H,dH,mu,dmu), dictionaries keyed by F or (F,F').
    Complete physical derivatives must be in the same fixed native basis.
    Models and angular unit operators can be reused across evaluations.
    On a gap/residual failure the evaluator switches permanently to the
    full represented solver, with explicit status. It never changes P/Q
    silently within an inner fit. Outputs include centroid frequencies and
    projector-summed strengths and their analytic Jacobians.
    """
    def __init__(self,models,producer,*,order,initial_values,residual_budget_MHz,
                 memory_budget_bytes=512*1024**2):
        self.models=dict(models);self.producer=producer;self.order=order
        if not self.models or not np.isfinite(residual_budget_MHz) or residual_budget_MHz<=0:
            raise ValueError('models and a positive residual budget required')
        self.residual_budget=float(residual_budget_MHz);self.memory_budget=memory_budget_bytes
        self.full_mode=False;self.fallback_reason=None;self.cache={};self.references={};self.definition_energies={}
        for f,m in self.models.items():
            e,u=np.linalg.eigh(m.hamiltonian(order=order))
            self.references[f]=(e,m.embedding(order=order)@u)
            self.definition_energies[f]=e.copy()
        self._initialize_response(np.asarray(initial_values,float))
        self.cache.clear()

    def _initialize_response(self,x):
        self._responses(x)

    def _responses(self,x):
        h,dh,mu,dmu=self.producer(x)
        self._last_physical=(np.asarray(x,float).tobytes(),h,dh,mu,dmu)
        if set(h)!=set(self.models) or set(dh)!=set(h):
            raise ValueError('producer must span the frozen F registry')
        responses={}
        try:
            for f,m in self.models.items():
                if self.full_mode:
                    response=canonical_spin_response(h[f],dh[f],order=0,memory_budget_bytes=self.memory_budget)
                else:
                    p=m.reduction.active_states;q=m.reduction.external_states
                    if q and len(p)!=len(m.requested_basis_indices):
                        raise SpinReductionGapError('discrete near-degeneracy promotion needs full reference')
                    frame=m.reference_vectors[:,list(p)+list(q)]
                    response=canonical_spin_response(h[f],dh[f],frame=frame,
                        retained_dimension=len(p),order=self.order,memory_budget_bytes=self.memory_budget)
                    e,u=np.linalg.eigh(response.hamiltonian_MHz);wave=response.embedding@u
                    residual=np.max(np.linalg.norm(h[f]@wave-wave*e,axis=0),initial=0.)
                    if residual>self.residual_budget:
                        raise SpinReductionGapError('Ritz residual exceeds the frozen reduction budget')
                responses[f]=response
        except (SpinReductionGapError,FloatingPointError) as error:
            self._reduction_failure(error)
            responses={f:canonical_spin_response(h[f],dh[f],order=0,memory_budget_bytes=self.memory_budget) for f in h}
        return responses,mu,dmu

    def _reduction_failure(self,error):
        self.full_mode=True;self.fallback_reason=str(error);self.cache.clear()

    def promote_accepted(self,x):
        """Update tracking only at optimizer-accepted points; keep P/Q fixed."""
        key=np.asarray(x,float).tobytes()
        responses=self.cache[key][0] if key in self.cache else self._responses(np.asarray(x,float))[0]
        for f,response in responses.items():
            e,u=np.linalg.eigh(response.hamiltonian_MHz)
            w=response.embedding@u
            rows,cols=linear_sum_assignment(-abs(self.references[f][1].conj().T@w)**2)
            mapping=cols[np.argsort(rows)]
            self.references[f]=(e[mapping],w[:,mapping])

    def evaluate(self,x,observations):
        x=np.asarray(x,float)
        if x.ndim!=1 or not np.all(np.isfinite(x)):raise ValueError('finite parameter vector required')
        key=x.tobytes()
        if key not in self.cache:
            self.cache.clear();self.cache[key]=self._responses(x)
        responses,mu,dmu=self.cache[key]
        return self._evaluate_responses(responses,mu,dmu,x,observations)

    def _evaluate_responses(self,responses,mu,dmu,x,observations):
        eig={}
        for f,response in responses.items():
            e,u=np.linalg.eigh(response.hamiltonian_MHz)
            reference=self.references[f][1]
            rows,cols=linear_sum_assignment(-abs(reference.conj().T@response.embedding@u)**2)
            mapping=cols[np.argsort(rows)]
            eig[f]=(e,u,mapping)
        values=[];jac=[];selectors=[]
        for row in observations:
            f,left=row['lower'];g,right=row['upper']
            el,ul,ml=eig[f];er,ur,mr=eig[g]
            initial_left=_selected_group(self.definition_energies[f],left)
            initial_right=_selected_group(self.definition_energies[g],right)
            il=_selected_group(el,ml[initial_left]);ir=_selected_group(er,mr[initial_right])
            l,dl,del_= _projector_response(el,ul,responses[f].hamiltonian_derivatives,il)
            r,dr,der_= _projector_response(er,ur,responses[g].hamiltonian_derivatives,ir)
            kind=row.get('kind','frequency')
            if kind=='frequency':
                values.append(float(np.mean(er[ir])-np.mean(el[il])));jac.append(der_-del_)
            elif kind=='strength':
                from .spin_effective_response import effective_spin_dipole_response
                pair=(f,g)
                if pair not in mu:raise ValueError('physical reduced dipole for this directed F pair is missing')
                op,dop=effective_spin_dipole_response(responses[f],responses[g],mu[pair],dmu[pair])
                amplitude=l.conj().T@op@r
                da=dl.conj().swapaxes(-1,-2)@op@r+l.conj().T@dop@r+l.conj().T@op@dr
                values.append(float(np.sum(abs(amplitude)**2)))
                jac.append(2*np.real(np.einsum('ij,pij->p',amplitude.conj(),da)))
            else:raise ValueError('measurement kind must be frequency or strength')
            selectors.append(dict(lower=il.tolist(),upper=ir.tolist()))
        return dict(values=np.asarray(values),jacobian=np.asarray(jac).reshape(len(values),len(x)),
            selectors=selectors,mode='full_fallback' if self.full_mode else 'reduced',fallback_reason=self.fallback_reason)


def fit_native_spin_reduced(evaluator,observations,initial_values,*,lower,upper,scales,
                            maximum_evaluations=100,covariance=None,priors=(),
                            joint_tolerance=.1,line_tolerance=.01):
    """Use LOCK/MORPHEUS trust-region LM with analytic frequency/strength rows.

    Optional Gaussian QM rows remain separate from experimental rows.
    Discrete parent/basis selection is done by the qualifier before fitting.
    """
    from .fit_convergence import statistical_trust_region,assess_statistical_convergence
    from scipy.optimize import OptimizeResult
    from matrix_numerics.least_squares import CovarianceWhitening
    observed=np.array([r['value'] for r in observations],float)
    sigma=np.array([r['uncertainty'] for r in observations],float)
    whitening=CovarianceWhitening(sigma,covariance)
    x0=np.asarray(initial_values,float);prior_rows=[]
    for prior in priors:
        index=int(prior['index']);width=float(prior['sigma']);mean=float(prior['mean'])
        if not 0<=index<len(x0) or not np.isfinite(width) or width<=0 or not np.isfinite(mean):
            raise ValueError('valid Gaussian QM predicate rows required')
        prior_rows.append((index,mean,width))
    evaluations=0;seen=set();accepted=x0.copy();mode=evaluator.full_mode;restarts=0
    class Converged(Exception):
        def __init__(self,x,report):self.x=x.copy();self.report=report
    from .adaptive_spin_fit import SpinRequalificationRequired
    class RestartFull(Exception):pass
    requalifications=0
    def residual(x):
        nonlocal evaluations,mode
        key=(np.asarray(x,float).tobytes(),evaluator.full_mode,getattr(evaluator,"model_generation",0))
        if key not in seen:evaluations+=1;seen.add(key)
        data=whitening.apply(evaluator.evaluate(x,observations)['values']-observed)
        if mode!=evaluator.full_mode:
            mode=evaluator.full_mode;seen.clear()
            raise RestartFull()
        return np.r_[data,[(x[i]-mean)/width for i,mean,width in prior_rows]]
    def jacobian(x):
        data=whitening.apply(evaluator.evaluate(x,observations)['jacobian'])
        prior=np.zeros((len(prior_rows),len(x)))
        for row,(i,_mean,width) in enumerate(prior_rows):prior[row,i]=1/width
        return np.vstack((data,prior))
    lower=np.asarray(lower,float);upper=np.asarray(upper,float)
    def checked_jacobian(x):
        j=jacobian(x)
        report=assess_statistical_convergence(residual(x),j,x,lower,upper,data_count=len(observations),
            joint_tolerance=joint_tolerance,line_tolerance=line_tolerance,
            data_correction_transform=whitening.standardized_transform)
        if report['converged']:raise Converged(np.asarray(x),report)
        return j
    def save_state(state):
        nonlocal accepted
        accepted=np.asarray(state['values'],float).copy()
        evaluator.promote_accepted(accepted)
    while True:
        try:
            fit=statistical_trust_region(residual,checked_jacobian,accepted,lower,upper,
                np.asarray(scales,float),max_nfev=max(1,int(maximum_evaluations)-evaluations),state_callback=save_state)
            break
        except Converged as stop:
            fit=OptimizeResult(x=stop.x,success=True,message='Local statistical stationarity qualified')
            break
        except SpinRequalificationRequired as request:
            requalifications+=1
            evaluator.requalify(accepted,observations,reason=str(request))
            mode=evaluator.full_mode
            if evaluations>=maximum_evaluations:
                fit=OptimizeResult(x=accepted,success=False,message="Evaluation budget exhausted during model requalification")
                break
        except RestartFull:
            restarts+=1
            if evaluations>=maximum_evaluations:
                fit=OptimizeResult(x=accepted,success=False,message='Evaluation budget exhausted after full-solver switch')
                break
    result=evaluator.evaluate(fit.x,observations)
    result.update(parameters=fit.x,success=bool(fit.success),message=str(fit.message),
        residual_evaluations=evaluations,full_solver_restarts=restarts,model_requalifications=requalifications,
        statistical_convergence=assess_statistical_convergence(residual(fit.x),jacobian(fit.x),fit.x,lower,upper,
            data_count=len(observations),joint_tolerance=joint_tolerance,line_tolerance=line_tolerance,
            data_correction_transform=whitening.standardized_transform),
        chi2_experiment=float(np.sum(residual(fit.x)[:len(observations)]**2)),
        chi2_qm=float(np.sum(residual(fit.x)[len(observations):]**2)))
    from .spin_prediction_uncertainty import spin_fit_information
    final_jacobian=jacobian(fit.x)
    result["information"]=spin_fit_information(final_jacobian[:len(observations)],final_jacobian[len(observations):],parameter_scales=scales)
    from .spin_prediction_uncertainty import spin_prediction_uncertainty
    comparison=result.get('reduction_comparison')
    numerical=({'coupling_order':comparison['differences']} if comparison is not None else
               ({'coupling_elimination':[0.]*len(observations)} if evaluator.full_mode else None))
    result['prediction_uncertainty']=spin_prediction_uncertainty(result['values'],result['jacobian'],
        result['information'],numerical_comparisons=numerical,
        observable_units=['D2' if row.get('kind','frequency')=='strength' else 'MHz' for row in observations])
    result['success']=bool(result['success'] and result['statistical_convergence']['converged'])
    result["model_history"]=getattr(evaluator,"model_history",[])
    result["residual_evaluations"]=evaluations
    return result
