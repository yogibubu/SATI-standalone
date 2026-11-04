"""Occupation-indexed GVPT2 operator with one fixed perturbative partition.

Diagonal and pair elements never depend on the currently selected support.
The legacy solver and target-directed refinement share these contractions.
"""
from __future__ import annotations

from itertools import combinations, combinations_with_replacement

import numpy as np

from .curvilinear_vpt2 import (
    _canonical_cubic_terms, _canonical_quartic_terms, _compiled_operator_couplings,
    _compiled_operator_matrix_element, _fermi_reduced_coupling,
    _gdv_overtone_dd_coupling, _gdv_combination_overtone_dd_coupling,
    _gdv_fundamental_dd_coupling, _modalize_sonic_field,
    _operator_terms_by_changed_modes, _quartic_expectation, _resonance_kind,
    _semidiagonal_quartic_edge_available,
)
from .resonance_screening import IntensityResonanceCriteria, directed_cubic_numerators, pair_decision


class GVPT2EffectiveOperator:
    def __init__(self, field, *, workspace=None, resonance_policy='gdv',
            fermi_window_cm=200.,dd_window_cm=100.,martin_threshold_cm=1.,
            numerical_zero_cm=.01,denominator_tolerance_cm=1e-6,
            polyad_excluded_modes=(),resonance_criteria=None):
        if resonance_policy not in {'auto','all','fermi','fermi_auto','gdv','intensity','none'}:
            raise ValueError('unsupported GVPT2 operator resonance policy')
        if resonance_criteria is not None and resonance_policy!='intensity':
            raise ValueError('resonance_criteria requires the intensity resonance policy')
        limits=(fermi_window_cm,dd_window_cm,martin_threshold_cm,numerical_zero_cm,
                denominator_tolerance_cm)
        if any(not np.isfinite(v) or v<0 for v in limits) or denominator_tolerance_cm==0:
            raise ValueError('GVPT2 operator thresholds must be finite and nonnegative')
        self.field=field
        self.workspace={} if workspace is None else workspace
        self.policy=resonance_policy
        self.fermi_window=float(fermi_window_cm)
        self.dd_window=float(dd_window_cm)
        self.martin=float(martin_threshold_cm)
        self.zero=float(numerical_zero_cm)
        self.denominator_tolerance=float(denominator_tolerance_cm)
        self.excluded=tuple(polyad_excluded_modes)
        self.criteria=IntensityResonanceCriteria.from_options(resonance_criteria)
        profile=(resonance_policy,self.fermi_window,self.dd_window,self.martin,self.zero,
                 self.denominator_tolerance,self.excluded,tuple(sorted((resonance_criteria or {}).items())))
        if self.workspace.get('effective_operator_profile',profile)!=profile:
            raise ValueError('GVPT2 workspace cannot mix different operator profiles')
        self.workspace['effective_operator_profile']=profile
        if 'modalized' not in self.workspace:
            self.workspace['modalized']=_modalize_sonic_field(field)
        self.frequencies,_,self.f3,self.f4,self.g1,self.g2=self.workspace['modalized']
        if 'cubic_terms' not in self.workspace:
            self.workspace['cubic_terms']=_canonical_cubic_terms(self.f3,self.g1,0.)
        self.terms3=self.workspace['cubic_terms']
        self.cubic_cache=self.workspace.setdefault('cubic_couplings',{})
        self.energy_cache=self.workspace.setdefault('occupation_harmonic_energies',{})
        self.screen_cache=self.workspace.setdefault('fermi_operator_patterns',{})
        self.dd_cache=self.workspace.setdefault('effective_dd_elements',{})
        self.diagonal_cache=self.workspace.setdefault('effective_diagonal_elements',{})
        self.pool={}

    def harmonic(self,state):
        if state not in self.energy_cache:
            self.energy_cache[state]=float(np.dot(self.frequencies,state))
        return self.energy_cache[state]

    def cubic(self,state,*,retain=True):
        if state in self.cubic_cache:
            return self.cubic_cache[state]
        value=_compiled_operator_couplings(state,self.terms3)
        if retain:
            self.cubic_cache[state]=value
        return value

    def resonant_fermi(self,a,b,coupling):
        if abs(coupling)<self.zero:
            return False
        pattern=tuple((i,x-y) for i,(x,y) in enumerate(zip(a,b)) if x!=y)
        left=sum(max(v,0) for _,v in pattern)
        right=sum(max(-v,0) for _,v in pattern)
        if {left,right}!={1,2}:
            return False
        if self.policy in {'all','fermi'}:
            return True
        if self.policy=='none':
            return False
        if self.policy=='intensity':
            if any(a[i] or b[i] for i in self.excluded):
                return False
            gap=abs(self.harmonic(a)-self.harmonic(b))
            return gap<=self.fermi_window and pair_decision(kind='1-2',gap_cm=gap,
                coupling_cm=abs(coupling),numerators_cm=None,criteria=self.criteria,
                denominator_tolerance_cm=self.denominator_tolerance,numerical_zero_cm=self.zero)['decision']=='accepted'
        if left!=1:
            pattern=tuple((i,-v) for i,v in pattern)
        if pattern not in self.screen_cache:
            gap=abs(sum(self.frequencies[i]*v for i,v in pattern))
            reduced=abs(_fermi_reduced_coupling(a,b,self.f3,self.g1))
            martin=reduced**4/gap**3 if gap else (np.inf if reduced else 0.)
            accepted=(gap<=self.fermi_window and (martin>=self.martin or
                self.policy=='gdv' and reduced/max(gap,self.denominator_tolerance)>=.1))
            self.screen_cache[pattern]=bool(accepted)
        return self.screen_cache[pattern]

    def directed_second(self,a,b):
        cache=self.workspace.setdefault('directed_second_order',{})
        if (a,b) not in cache:
            cache[(a,b)]=directed_cubic_numerators(a,b,self.frequencies,
                self.cubic(a,retain=False),self.cubic(b,retain=False),self.resonant_fermi,
                denominator_tolerance_cm=self.denominator_tolerance)
        return cache[(a,b)]

    def second_order(self,a,b):
        if self.policy=='intensity':
            forward,reverse=self.directed_second(a,b)
            return (np.conjugate(forward)+reverse)/2
        cache=self.workspace.setdefault('second_order',{})
        if (a,b) in cache:
            return cache[(a,b)]
        if (b,a) in cache:
            return np.conjugate(cache[(b,a)])
        ca=self.cubic(a,retain=False)
        cb=ca if a==b else self.cubic(b,retain=False)
        value=0j
        for intermediate in ca.keys() & cb.keys():
            va,vb=ca[intermediate],cb[intermediate]
            if self.resonant_fermi(a,intermediate,va) or self.resonant_fermi(b,intermediate,vb):
                continue
            er=self.harmonic(intermediate)
            da,db=self.harmonic(a)-er,self.harmonic(b)-er
            if abs(da)<=self.denominator_tolerance or abs(db)<=self.denominator_tolerance:
                if abs(va*vb)>self.zero**2:
                    raise ValueError(f'unresolved SONIC GVPT2 denominator for {a}, {b}, {intermediate}')
                continue
            value+=.5*np.conjugate(va)*vb*(1/da+1/db)
        cache[(a,b)]=value
        return value

    def diagonal(self,state):
        if state not in self.diagonal_cache:
            quartic=self.workspace.setdefault('quartic_diagonal',{})
            if state not in quartic:
                quartic[state]=_quartic_expectation(state,self.f4,self.g2,self.field.pseudopotential_cm,0.)
            self.diagonal_cache[state]=self.harmonic(state)+quartic[state]+self.second_order(state,state).real
        return self.diagonal_cache[state]

    def dd_coupling(self,a,b,kind):
        key=(a,b,kind)
        if key in self.dd_cache:
            return self.dd_cache[key]
        if 'quartic_terms' not in self.workspace:
            self.workspace['quartic_terms']=_canonical_quartic_terms(self.f4,self.g2)
        if 'quartic_term_index' not in self.workspace:
            self.workspace['quartic_term_index']=_operator_terms_by_changed_modes(self.workspace['quartic_terms'])
        value=_compiled_operator_matrix_element(b,a,self.workspace['quartic_terms'],
            self.workspace['quartic_term_index'])+self.second_order(a,b)
        if abs(value.imag)>1e-6:
            raise ValueError('SONIC GVPT2 effective Hamiltonian is not real')
        coupling=value.real
        if self.policy=='gdv' and kind=='2-2':
            overtone=_gdv_overtone_dd_coupling(a,b,self.frequencies,self.f3,self.f4,self.g1,self.g2)
            if overtone is not None:
                coupling=overtone
            mixed=_gdv_combination_overtone_dd_coupling(a,b,self.frequencies,self.f3,self.f4,self.g1,self.g2)
            if mixed is not None:
                coupling=mixed
        if self.policy=='gdv' and kind=='1-1':
            fundamental=_gdv_fundamental_dd_coupling(a,b,self.frequencies,self.f3,self.f4,self.g1,self.g2)
            if fundamental is not None:
                coupling=fundamental
        self.dd_cache[key]=float(coupling)
        self.dd_cache[(b,a,kind)]=float(coupling)
        return float(coupling)

    def coupling(self,a,b,*,reduced_polynomial=False):
        if a==b:
            return self.diagonal(a)
        if any(a[i] or b[i] for i in self.excluded):
            return 0.
        kind=_resonance_kind(a,b)
        if kind=='1-2':
            value=self.cubic(a,retain=False).get(b,0j)
            return value.conjugate() if self.resonant_fermi(a,b,value) else 0.
        if self.policy in {'none','fermi','fermi_auto'} or kind not in {'1-1','2-2','1-3'}:
            return 0.
        if abs(self.harmonic(a)-self.harmonic(b))>self.dd_window:
            return 0.
        if self.field.quartic_coverage!='complete' and not reduced_polynomial and not _semidiagonal_quartic_edge_available(a,b,kind):
            return 0.
        value=self.dd_coupling(a,b,kind)
        return value if abs(value)>=self.zero else 0.

    def _pool(self,order):
        if order not in self.pool:
            indices=list(combinations_with_replacement(range(len(self.frequencies)),order))
            energies=np.array([sum(self.frequencies[i] for i in item) for item in indices])
            sort=np.argsort(energies,kind='stable')
            self.pool[order]=(energies[sort],[indices[i] for i in sort])
        return self.pool[order]

    def neighbors(self,state):
        result=set()
        for target,value in self.cubic(state).items():
            if self.resonant_fermi(state,target,value):
                result.add(target)
        if self.policy in {'none','fermi','fermi_auto'}:
            return result
        occupied=tuple(i for i,count in enumerate(state) for _ in range(count))
        for removed_count,added_count in ((1,1),(2,2),(1,3),(3,1)):
            energies,additions=self._pool(added_count)
            for removed in sorted(set(combinations(occupied,removed_count))):
                energy=sum(self.frequencies[i] for i in removed)
                first=np.searchsorted(energies,energy-self.dd_window,side='left')
                stop=np.searchsorted(energies,energy+self.dd_window,side='right')
                for added in additions[first:stop]:
                    trial=list(state)
                    for i in removed: trial[i]-=1
                    for i in added: trial[i]+=1
                    target=tuple(trial)
                    if target!=state and _resonance_kind(state,target) in {'1-1','2-2','1-3'}:
                        result.add(target)
        return result
