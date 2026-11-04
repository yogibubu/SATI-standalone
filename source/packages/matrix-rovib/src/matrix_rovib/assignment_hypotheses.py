"""Bounded comparison of complete, competing assignment maps on identical data."""
from __future__ import annotations

from copy import deepcopy
import numpy as np


def compare_assignment_hypotheses(surface,fit_request,assignment_report,catalogue,reference,stages,*,
        maximum_hypotheses=8,maximum_nodes=512,maximum_total_evaluations=120,
        minimum_objective_gap=9.,refinement_options=None,_resume=None,_checkpoint_callback=None):
    """Retain alternatives unless every enumerated fit is comparable and stationary.

    Scores are penalized least-squares objectives, not posterior probabilities.
    Null alternatives, truncated search or failed fits prevent automatic choice.
    A selection is proposed separately; the original assignments are immutable.
    """
    from .assignment_refinement import refine_exploratory_assignments
    from .assignment_search import assignment_components
    from .lam_assignment import solve_transition_term_values
    from .assignment_checkpoint import assignment_fingerprint
    for value in (maximum_hypotheses,maximum_nodes,maximum_total_evaluations):
        if isinstance(value,bool) or not isinstance(value,int) or value<1:raise ValueError('positive integer hypothesis budgets required')
    if not np.isfinite(minimum_objective_gap) or minimum_objective_gap<=0:raise ValueError('positive declared objective gap required')
    if not stages:raise ValueError('hypothesis comparison requires explicit bounded fit stages')
    budgets=[s.get('max_nfev',20) for s in stages]
    if any(isinstance(b,bool) or not isinstance(b,int) or b<1 for b in budgets):raise ValueError('positive integer stage budgets required')
    per_fit=sum(budgets)
    if per_fit<1 or per_fit>maximum_total_evaluations:raise ValueError('single hypothesis exceeds total evaluation budget')
    lookup={str(r['id']):r for r in catalogue}
    if len(lookup)!=len(catalogue):raise ValueError('unique catalogue IDs required')
    rows=deepcopy(assignment_report['rows']);choices=[];has_null=False
    cutoff=assignment_report['search']['ambiguity_cost']
    for index,row in enumerate(rows):
        if row['peak'].get('role','training')!='training':continue
        if row['status']=='accepted':options=[str(row['catalogue_id'])]
        elif row['status']=='ambiguous':
            candidates=[c for c in row['candidates'] if c['global_cost_increase']<cutoff]
            has_null|=any(c['catalogue_id'] is None for c in candidates)
            options=sorted({str(c['catalogue_id']) for c in candidates if c['catalogue_id'] is not None})
        else:continue
        if any(o not in lookup for o in options):raise ValueError('candidate missing from catalogue')
        if options:choices.append((index,options))
    if not choices:raise ValueError('no comparable assigned measurements')
    # Search independent of input row order; all hypotheses retain identical rows.
    choices.sort(key=lambda c:(len(c[1]),str(rows[c[0]]['peak']['id'])))
    maps=[];nodes=0;truncated=False
    def visit(k,selected,used):
        nonlocal nodes,truncated
        if nodes>=maximum_nodes:truncated=True;return
        nodes+=1
        if k==len(choices):
            if len(maps)>=maximum_hypotheses:truncated=True;return
            maps.append(dict(selected));return
        index,options=choices[k]
        for cid in options:
            line=lookup[cid];occupied=set(map(str,line.get('constituent_ids',[cid])))
            if used & occupied:continue
            selected[index]=cid;visit(k+1,selected,used|occupied);selected.pop(index)
            if truncated:return
    visit(0,{},set())
    contract=assignment_fingerprint(surface,fit_request,assignment_report,catalogue,reference,stages,
        maximum_hypotheses,maximum_nodes,maximum_total_evaluations,minimum_objective_gap,refinement_options)
    previous={} if _resume is None else deepcopy(_resume)
    if previous and previous.get('contract')!=contract:raise ValueError('hypothesis checkpoint contract mismatch')
    completed=previous.get('hypotheses',[]);results=[]
    for h,mapping in enumerate(maps):
        ids={str(rows[i]['peak']['id']):cid for i,cid in mapping.items()}
        identity=assignment_fingerprint(ids)
        cached=next((r for r in completed if r['identity']==identity),None)
        if cached is not None:results.append(cached);continue
        if (len(results)+1)*per_fit>maximum_total_evaluations:truncated=True;break
        report=deepcopy(assignment_report)
        for i,row in enumerate(report['rows']):
            if i in mapping:row.update(status='accepted',catalogue_id=mapping[i],transition=deepcopy(lookup[mapping[i]]))
            elif row['status']=='accepted':raise ValueError('inconsistent common measurement set')
        report['accepted_count']=len(mapping)
        network=[]
        for i,cid in mapping.items():
            t=lookup[cid];peak=rows[i]['peak']
            if 'components' in t:continue
            for c in assignment_components(t):
                network.append(dict(lower_id=c['lower']['state_id'],upper_id=c['upper']['state_id'],
                    frequency_MHz=peak['frequency_MHz'],uncertainty_MHz=peak['uncertainty_MHz']))
        item=dict(identity=identity,assignments=ids,status='pending',refinement=None,objective=None)
        try:
            # The existing sparse term-value solver has diagonal weights.
            # Correlated rows are compared by the full-covariance native fit,
            # never excluded using a diagonal approximation to their noise.
            if network and fit_request.get('covariance_MHz2') is None:
                solved=solve_transition_term_values(network)
                if any(abs(r)>assignment_report['search'].get('cycle_tolerance_sigma',4.)*o['uncertainty_MHz']
                       for r,o in zip(solved.transition_residuals_MHz,network,strict=True)):
                    item['status']='inconsistent_combination_differences';results.append(item)
                    if _checkpoint_callback:_checkpoint_callback(dict(contract=contract,hypotheses=results))
                    continue
            from collections import Counter
            families=Counter(lookup[cid]['family'] for cid in mapping.values())
            if any(count<assignment_report['search'].get('minimum_family_lines',1) for count in families.values()):
                raise ValueError('hypothesis lacks the declared family support')
            options=dict(refinement_options or {});options['maximum_total_evaluations']=per_fit
            result=refine_exploratory_assignments(surface,fit_request,report,reference,stages,**options)
            item.update(status=result['status'],refinement=result)
            if result['completed']:
                fit=result['stages'][-1]['fit'];score=float(fit['chi_square'])
                if np.isfinite(score):item['objective']=score
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:item.update(status='blocked',message=str(exc))
        results.append(item)
        if _checkpoint_callback:_checkpoint_callback(dict(contract=contract,hypotheses=results))
    ranked=sorted((r for r in results if r['objective'] is not None),key=lambda r:(r['objective'],r['identity']))
    # A cycle-inconsistent map is excluded by measured energy consistency;
    # an unconverged fit cannot be excluded simply because of its last residual.
    qualified=all(r['objective'] is not None or r['status']=='inconsistent_combination_differences' for r in results)
    gap=None if len(ranked)<2 else ranked[1]['objective']-ranked[0]['objective']
    select=bool(ranked and qualified and not truncated and not has_null and (len(ranked)==1 or gap>=minimum_objective_gap))
    return dict(schema='matrix.lock.assignment_hypotheses.v1',hypotheses=results,
        selected_identity=ranked[0]['identity'] if select else None,
        selected_assignments=ranked[0]['assignments'] if select else None,
        selected_refinement=ranked[0]['refinement'] if select else None,
        status='discriminated' if select else 'alternatives_retained',objective_gap=gap,
        search_complete=not truncated,null_alternative_retained=has_null,all_fits_qualified=qualified,
        common_measurement_ids=sorted(str(rows[i]['peak']['id']) for i,_ in choices),
        budget=dict(nodes=nodes,maximum_nodes=maximum_nodes,maximum_hypotheses=maximum_hypotheses,
                    declared_evaluations_per_fit=per_fit,maximum_total_evaluations=maximum_total_evaluations),
        criterion=dict(minimum_objective_gap=minimum_objective_gap,score='same-data Gaussian chi-square plus unchanged QM penalties; no posterior probability'),
        scope='conditional on retained catalogue alternatives and declared model; does not eliminate unsearched physical assignments')


__all__=['compare_assignment_hypotheses']
