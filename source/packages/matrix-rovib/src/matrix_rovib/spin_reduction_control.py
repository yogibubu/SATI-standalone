"""Bounded selection of native spin model space and intermanifold order."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment

from .native_spin_effective import (
    build_native_lam_spin_operator, reduce_native_spin_operator, NativeSpinEffectiveHamiltonian,
)
from .ordered_downfold import OrderedEffectiveHamiltonian
from .spin_effective_observables import native_spin_dipole_operators


@dataclass(frozen=True)
class NativeSpinReductionQualification:
    models: dict
    order: int
    report: dict


def _clusters(energies):
    tolerance = 128*np.finfo(float).eps*max(1.,np.max(abs(energies),initial=0.))
    groups=[]
    for i,e in enumerate(energies):
        if groups and abs(e-energies[groups[-1][0]])<=tolerance:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def _full_model(block):
    n=len(block.basis)
    reduction=OrderedEffectiveHamiltonian({((),()):block.hamiltonian_MHz},tuple(range(n)),(),0,
                                         None,{}, {})
    return NativeSpinEffectiveHamiltonian(block,reduction,np.eye(n,dtype=complex),tuple(range(n)))


def qualify_native_spin_reduction(surface,lam_result,model,*,two_f_values,active_lam_states=(0,),
        orders=(2,4,6),frequency_budget_MHz,strength_budget,maximum_evaluations=12,
        maximum_reduced_dimension=512,memory_budget_bytes=512*1024**2,
        include_gap_MHz=0.,fallback_full=True,dipole_au=None,target_indices=None):
    """Qualify against a cached full reference, then freeze the chosen fit space.

    Reference F diagonalizations and raw dipoles are made once. Each trial
    uses order-matched wave operators. Exact-degenerate transition strengths
    are compared as projector sums, never as arbitrary partner intensities.
    Space enlargement adds a whole LAM parent (and degenerate partners) over
    all N and spin paths. Targets may be explicit full-reference root indices;
    otherwise they are chosen once by projection onto the initial space.
    Full fallback is an explicitly labelled exact represented-space result.
    """
    from .spin_hyperfine import _lam_spin_basis_registry
    fs=tuple(sorted(set(two_f_values)));orders=tuple(orders)
    if (not fs or any(isinstance(f,bool) or not isinstance(f,(int,np.integer)) or f<0 for f in fs)
        or not orders or any(isinstance(k,bool) or not isinstance(k,int) or k<1 for k in orders)
        or tuple(sorted(set(orders)))!=orders or maximum_evaluations<1
        or isinstance(maximum_evaluations,bool) or int(maximum_evaluations)!=maximum_evaluations
        or isinstance(maximum_reduced_dimension,bool) or int(maximum_reduced_dimension)!=maximum_reduced_dimension
        or maximum_reduced_dimension<1 or not np.all(np.isfinite([frequency_budget_MHz,strength_budget]))
        or min(frequency_budget_MHz,strength_budget)<=0):
        raise ValueError('valid F blocks, increasing orders and positive numerical/resource budgets required')
    _,registry,_=_lam_spin_basis_registry(lam_result,model)
    if any(f not in registry for f in fs):
        raise ValueError('requested F outside supplied native N range')
    dimensions={f:len(registry[f]) for f in fs}
    if sum(16*n*n*(16+4*max(orders)) for n in dimensions.values())>memory_budget_bytes:
        raise MemoryError('joint native spin qualification exceeds memory budget')
    blocks={f:build_native_lam_spin_operator(surface,lam_result,model,two_F=f,
        memory_budget_bytes=memory_budget_bytes) for f in fs}
    raw=native_spin_dipole_operators(surface,lam_result,blocks.values(),dipole_au=dipole_au,
        memory_budget_bytes=memory_budget_bytes)
    parents={f:np.array([lam_result.blocks[b.N].levels[b.k_index].torsional_state for b in block.basis])
             for f,block in blocks.items()}
    energies=np.asarray(lam_result.basis.energies_MHz)
    selected=set(active_lam_states)
    if not selected or any(isinstance(i,bool) or not isinstance(i,(int,np.integer)) or not 0<=i<len(energies) for i in selected):
        raise ValueError('valid nonempty initial LAM parents required')
    groups=_clusters(energies)
    def close(values):
        return set().union(*(set(g) for g in groups if set(g)&set(values)))
    selected=close(selected)
    full={f:np.linalg.eigh(block.hamiltonian_MHz) for f,block in blocks.items()}
    targets={}
    if target_indices is not None and set(target_indices)!=set(fs):
        raise ValueError('target root registry must span exactly the requested F blocks')
    for f,(e,u) in full.items():
        initial=np.flatnonzero(np.isin(parents[f],list(selected)))
        if not len(initial):
            raise ValueError('initial LAM parents do not intersect every requested F')
        if target_indices is None:
            weights=np.sum(abs(u[initial])**2,axis=0)
            chosen=set(np.lexsort((e,-weights))[:len(initial)])
        else:
            chosen=set(target_indices[f])
            if not chosen or any(isinstance(i,bool) or not isinstance(i,(int,np.integer)) or not 0<=i<len(e) for i in chosen):
                raise ValueError('valid full-reference target indices required')
        targets[f]=np.array(sorted(set().union(*(set(g) for g in _clusters(e) if set(g)&chosen))),int)
    target_groups={f:_clusters(full[f][0][targets[f]]) for f in fs}
    history=[];evaluations=0;models={};last_order=orders[0];passed=False
    while evaluations<maximum_evaluations-(1 if fallback_full else 0):
        indices={f:np.flatnonzero(np.isin(parents[f],list(selected))) for f in fs}
        if max(map(len,indices.values()))>maximum_reduced_dimension:
            break
        models={f:reduce_native_spin_operator(blocks[f],indices[f],maximum_order=max(orders),
            include_gap_MHz=include_gap_MHz,memory_budget_bytes=memory_budget_bytes) for f in fs}
        if max(len(m.reduction.active_states) for m in models.values())>maximum_reduced_dimension:
            break
        scores=np.zeros(len(energies))
        for order in orders:
            if evaluations>=maximum_evaluations-(1 if fallback_full else 0):break
            evaluations+=1;last_order=order
            waves={};ritz={};residual=0.;energy_error=0.;overlaps=[];enough=True
            for f,m in models.items():
                e,v=np.linalg.eigh(m.hamiltonian(order=order));w=m.embedding(order=order)@v
                ids=targets[f]
                if len(e)<len(ids):enough=False;break
                weights=abs(full[f][1][:,ids].conj().T@w)**2
                rows,cols=linear_sum_assignment(-weights)
                cols=cols[np.argsort(rows)];waves[f]=w[:,cols];ritz[f]=e[cols]
                overlaps.extend(weights[rows,cols].tolist())
                r=blocks[f].hamiltonian_MHz@waves[f]-waves[f]*ritz[f]
                residual=max(residual,float(np.max(np.linalg.norm(r,axis=0),initial=0.)))
                energy_error=max(energy_error,float(np.max(abs(ritz[f]-full[f][0][ids]),initial=0.)))
                for parent in range(len(energies)):
                    scores[parent]+=float(np.sum(abs(r[parents[f]==parent])**2))
            frequency_error=0.;strength_error=0.
            if enough:
                for (f,g),mu in raw.items():
                    ef,eg=full[f][0][targets[f]],full[g][0][targets[g]]
                    frequency_error=max(frequency_error,float(np.max(abs(
                        (ritz[g][None,:]-ritz[f][:,None])-(eg[None,:]-ef[:,None])),initial=0.)))
                    a=waves[f].conj().T@mu@waves[g]
                    b=full[f][1][:,targets[f]].conj().T@mu@full[g][1][:,targets[g]]
                    for left in target_groups[f]:
                        for right in target_groups[g]:
                            strength_error=max(strength_error,abs(float(np.sum(abs(a[np.ix_(left,right)])**2)
                                -np.sum(abs(b[np.ix_(left,right)])**2))))
            passed=bool(enough and residual<=frequency_budget_MHz/2 and frequency_error<=frequency_budget_MHz
                        and energy_error<=frequency_budget_MHz/2 and strength_error<=strength_budget)
            history.append(dict(order=order,lam_parents=sorted(map(int,selected)),dimensions={str(f):len(m.reduction.active_states)
                for f,m in models.items()},qualified=passed,enough_states=enough,
                maximum_ritz_residual_MHz=residual if enough else None,
                maximum_frequency_error_MHz=frequency_error if enough else None,
                maximum_strength_error=strength_error if enough else None,
                minimum_individual_matching_weight=min(overlaps,default=None)))
            if passed:break
        if passed:break
        available=[g for g in groups if not set(g)&selected]
        if not available:break
        chosen=min(available,key=lambda g:(-float(np.sum(scores[g])),energies[g[0]],g[0]))
        selected.update(chosen)
    reduction_passed=passed and any(m.reduction.external_states for m in models.values())
    status=("reduced" if reduction_passed else "full_space") if passed else "budget_exhausted"
    if not passed and fallback_full and evaluations<maximum_evaluations:
        models={f:_full_model(block) for f,block in blocks.items()}
        evaluations+=1;last_order=0
        full_residual=max(float(np.max(np.linalg.norm(blocks[f].hamiltonian_MHz@u-u*e,axis=0)[targets[f]],initial=0.))
                          for f,(e,u) in full.items())
        passed=full_residual<=frequency_budget_MHz/2
        status='full_fallback' if passed else 'full_precision_limit'
    report=dict(schema='matrix.lock.native_spin_reduction_qualification.v1',qualified=passed,
        reduction_qualified=reduction_passed,status=status,chosen_order=last_order,
        target_indices={str(f):ids.tolist() for f,ids in targets.items()},
        strength_convention='Frobenius sums over exactly degenerate target projectors',
        frequency_budget_MHz=float(frequency_budget_MHz),strength_budget=float(strength_budget),
        evaluations=evaluations,maximum_evaluations=maximum_evaluations,history=history,
        full_reference_diagonalizations=len(full),experimental_data_used=False,
        scope='cached full-reference qualification in supplied native LAM/N spaces; no primitive-grid or QC accuracy claim')
    return NativeSpinReductionQualification(models,last_order,report)
