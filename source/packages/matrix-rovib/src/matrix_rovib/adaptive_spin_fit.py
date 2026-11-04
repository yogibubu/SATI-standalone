"""Accepted-point requalification of reduced Hamiltonians during nonlinear fits."""
from __future__ import annotations

from dataclasses import replace
import numpy as np
from scipy.optimize import linear_sum_assignment
from .spin_reduced_fit import NativeSpinReducedEvaluator
from .spin_effective_response import canonical_spin_response
from .native_spin_effective import NativeSpinEffectiveHamiltonian
from .ordered_downfold import OrderedEffectiveHamiltonian


class SpinRequalificationRequired(RuntimeError):
    """A trial needs an outer model update, never a silent inner-basis change."""


class AdaptiveNativeSpinEvaluator(NativeSpinReducedEvaluator):
    """Freeze each inner model; rebuild only at the last accepted point.

    Trial Ritz residuals, external separation and a higher-order frequency/
    strength comparison guard the explored fit region. The order comparison
    is an error estimate, not a rigorous bound. Each outer reconstruction uses
    one full reference per F at the accepted point, closes near degeneracies,
    and preserves tracked observable identities. Updates and dimensions are
    bounded; exhaustion switches permanently to the full represented solver.
    """
    def __init__(self,*args,frequency_budget_MHz,strength_budget,minimum_external_gap_MHz=0.,
                 maximum_rebuilds=3,maximum_retained_dimension=512,comparison_order_increment=2,**kwargs):
        if (not np.all(np.isfinite([frequency_budget_MHz,strength_budget,minimum_external_gap_MHz]))
            or min(frequency_budget_MHz,strength_budget)<=0 or minimum_external_gap_MHz<0
            or any(isinstance(v,bool) or not isinstance(v,(int,np.integer)) or v<1
                   for v in (maximum_rebuilds,maximum_retained_dimension,comparison_order_increment))):
            raise ValueError('positive observable/resource budgets and nonnegative gap required')
        self.frequency_budget=frequency_budget_MHz;self.strength_budget=strength_budget
        self.gap_budget=minimum_external_gap_MHz;self.maximum_rebuilds=maximum_rebuilds
        self.dimension_budget=maximum_retained_dimension;self.increment=comparison_order_increment
        self.model_generation=0;self.model_history=[]
        super().__init__(*args,**kwargs)

    def _initialize_response(self,x):
        # The first fit evaluation qualifies or reconstructs the initial model
        # through the same accepted-anchor protocol as later iterations.
        pass

    def _reduction_failure(self,error):
        raise SpinRequalificationRequired(str(error)) from error

    def evaluate(self,x,observations):
        result=super().evaluate(x,observations)
        if self.full_mode:return result
        data=self.cache[np.asarray(x,float).tobytes()];responses,mu,dmu=data
        if any(r.minimum_external_gap_MHz is not None and r.minimum_external_gap_MHz<self.gap_budget for r in responses.values()):
            raise SpinRequalificationRequired('external separation falls below declared gap budget')
        _,h,dh,_,_=self._last_physical
        higher={}
        for f,m in self.models.items():
            p=m.reduction.active_states;q=m.reduction.external_states
            higher[f]=canonical_spin_response(h[f],dh[f][:0],frame=m.reference_vectors[:,list(p)+list(q)],
                retained_dimension=len(p),order=self.order+self.increment,memory_budget_bytes=self.memory_budget)
        check=self._evaluate_responses(higher,mu,{pair:array[:0] for pair,array in dmu.items()},np.empty(0),observations)
        differences=abs(result['values']-check['values'])
        budgets=np.array([self.strength_budget if r.get('kind','frequency')=='strength' else self.frequency_budget for r in observations])
        if np.any(differences>budgets):
            raise SpinRequalificationRequired('nested-order frequency/strength comparison exceeds budget')
        result['reduction_comparison']=dict(differences=differences.tolist(),budgets=budgets.tolist(),
            higher_order=self.order+self.increment,interpretation='local refinement-difference estimate; not a rigorous bound')
        result['model_generation']=self.model_generation
        return result

    def requalify(self,accepted_values,observations,*,reason):
        """Refresh the projectors at an accepted point, with explicit full checks."""
        x=np.asarray(accepted_values,float);h,dh,mu,dmu=self.producer(x)
        self.model_generation+=1;self.cache.clear()
        record=dict(generation=self.model_generation,accepted_parameters=x.tolist(),reason=reason)
        if self.model_generation>self.maximum_rebuilds:
            self.full_mode=True;self.fallback_reason='outer reconstruction budget exhausted'
            record['status']='full_fallback';self.model_history.append(record);return
        models={};full={};dimensions={};fallback=False
        for f,old in self.models.items():
            e,u=np.linalg.eigh(h[f]);full[f]=canonical_spin_response(h[f],dh[f],order=0,memory_budget_bytes=self.memory_budget)
            rows,cols=linear_sum_assignment(-abs(self.references[f][1].conj().T@u)**2)
            selected=set(map(int,cols[np.argsort(rows)]))
            # Grow beyond the previous size after a failed earlier reconstruction.
            goal=min(len(e),max(len(selected),len(old.reduction.active_states)+self.model_generation-1))
            while len(selected)<goal:
                candidates=[i for i in range(len(e)) if i not in selected]
                chosen=min(candidates,key=lambda j:min(abs(e[j]-e[list(selected)])))
                selected.add(chosen)
            tolerance=max(self.gap_budget,128*np.finfo(float).eps*max(1.,np.max(abs(e))))
            while True:
                additions={j for j in range(len(e)) if j not in selected
                           and min(abs(e[j]-e[list(selected)]))<=tolerance}
                if not additions:break
                selected.update(additions)
            p=tuple(sorted(selected));q=tuple(i for i in range(len(e)) if i not in selected)
            dimensions[str(f)]=len(p)
            if len(p)>self.dimension_budget:fallback=True
            reduction=OrderedEffectiveHamiltonian({((),()):np.diag(e[list(p)])},p,q,self.order,None,{}, {})
            models[f]=NativeSpinEffectiveHamiltonian(replace(old.block,hamiltonian_MHz=h[f]),reduction,u,p)
        self.models=models
        if fallback or all(not m.reduction.external_states for m in models.values()):
            self.full_mode=True;self.fallback_reason='retained dimension limit or complete represented space'
            record.update(status='full_fallback',dimensions=dimensions)
        else:
            # At the accepted anchor, retained exact eigenspaces must reproduce
            # the same tracked full-reference observables within declared budgets.
            responses,_,_=self._responses(x)
            actual=self._evaluate_responses(responses,mu,dmu,x,observations)
            exact=self._evaluate_responses(full,mu,dmu,x,observations)
            delta=abs(actual['values']-exact['values'])
            budgets=np.array([self.strength_budget if r.get('kind','frequency')=='strength' else self.frequency_budget for r in observations])
            if np.any(delta>budgets):
                self.full_mode=True;self.fallback_reason='accepted-anchor full-reference comparison failed'
            record.update(status='full_fallback' if self.full_mode else 'requalified',dimensions=dimensions,
                full_reference_diagonalizations=len(full),maximum_anchor_difference=float(np.max(delta,initial=0.)))
        self.model_history.append(record)
