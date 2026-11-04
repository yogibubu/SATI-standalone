"""Serializable review state shared by LOCK Studio and batch workflows."""
from __future__ import annotations
from copy import deepcopy


def build_assignment_review(result,*,configuration=None,profile_result=None):
    from .assignment_checkpoint import assignment_fingerprint
    if result.get('schema')!='matrix.lock.assignment_workflow_result.v1':raise ValueError('assignment workflow result required')
    rows=deepcopy(result['assignments']['rows']);catalogue={str(c['id']):c for c in result['catalogue']}
    if len(catalogue)!=len(result['catalogue']):raise ValueError('unique catalogue IDs required')
    refinement=result.get('refinement') or {};stages=refinement.get('stages',[])
    fit=stages[-1].get('fit',{}) if stages else {}
    predicted={}
    if 'frequencies_MHz' in fit:
        predicted=dict(zip([o['label'] for o in refinement.get('observations',[])],fit['frequencies_MHz'],strict=True))
    for c in fit.get('comparisons',[]):
        label=c.get('label');value=c.get('calculated_MHz',c.get('calculated_frequency_MHz'))
        if label is not None and value is not None:predicted[str(label)]=value
    for row in rows:
        peak=row['peak'];value=predicted.get(str(peak['id']))
        row['calculated_MHz']=value
        row['residual_MHz']=None if value is None else float(value-peak['frequency_MHz'])
    values=refinement.get('last_accepted_values',{})
    parameter_rows=[]
    for p in (configuration or {}).get('fit_request',{}).get('variables',[]):
        constraints=([p['prior']] if p.get('prior') else [])
        constraints.extend(q for q in (configuration or {}).get('fit_request',{}).get('predicates',[]) if q['name']==p.get('name',p['path']))
        for prior in constraints or [{}]:
            v=values.get(p['path']);mean=prior.get('value');sigma=prior.get('sigma')
            if sigma is not None and sigma<=0:raise ValueError('positive QM predicate sigma required')
            parameter_rows.append(dict(path=p['path'],initial=p.get('initial'),fitted=v,qm_value=mean,qm_sigma=sigma,
                marginal_shift_over_sigma=None if v is None or mean is None or sigma is None else (v-mean)/sigma))
    if profile_result is not None and profile_result.get('schema')!='matrix.lock.spectral_profile_result.v1':
        raise ValueError('native profile result required')
    content=dict(schema='matrix.lock.assignment_review.v1',source_sha256=assignment_fingerprint(result,configuration,profile_result),
        rows=rows,catalogue=list(catalogue.values()),parameters=parameter_rows,
        parameter_diagnostic_scope='marginal QM shifts only; correlated direction diagnostics remain in the fit report',
        alternatives=deepcopy(result.get('alternatives')),profile=deepcopy(profile_result),
        events=[],decisions={},scope='human review; decisions do not certify assignment correctness or overwrite source fits')
    content['review_sha256']=assignment_fingerprint(content)
    return content


def record_assignment_decision(review,*,peak_id,catalogue_id,reason,author='user'):
    """Append a hash-linked decision, rejecting withheld rows and line reuse."""
    from .assignment_checkpoint import assignment_fingerprint
    if review.get('schema')!='matrix.lock.assignment_review.v1':raise ValueError('assignment review required')
    original=deepcopy(review);stored=original.pop('review_sha256',None)
    if assignment_fingerprint(original)!=stored:raise ValueError('review content hash mismatch')
    peak_id=str(peak_id);reason=str(reason).strip();author=str(author).strip()
    if not reason or not author:raise ValueError('a reason and author are required for every decision')
    rows={str(r['peak']['id']):r for r in review['rows']};lookup={str(c['id']):c for c in review['catalogue']}
    if peak_id not in rows or rows[peak_id]['peak'].get('role','training')!='training':raise ValueError('only declared training peaks may be reassigned')
    cid=None if catalogue_id is None else str(catalogue_id)
    allowed={str(c['catalogue_id']) for c in rows[peak_id]['candidates'] if c.get('catalogue_id') is not None}
    if cid is not None and (cid not in lookup or cid not in allowed):raise ValueError('decision must select a recorded candidate')
    decisions=deepcopy(review['decisions']);decisions[peak_id]=cid
    used=set()
    for pid,row in rows.items():
        selected=decisions.get(pid,row.get('catalogue_id') if row['status']=='accepted' else None)
        if selected is None:continue
        line=lookup[selected];occupied=set(map(str,line.get('constituent_ids',[selected])))
        if used&occupied:raise ValueError('decision would assign one physical transition to multiple measured peaks')
        used|=occupied
    result=deepcopy(review)
    event=dict(sequence=len(result['events'])+1,peak_id=peak_id,catalogue_id=cid,reason=reason,
               author=author,previous_review_sha256=stored)
    result['events'].append(event);result['decisions']=decisions
    result.pop('review_sha256');result['review_sha256']=assignment_fingerprint(result)
    return result


def reviewed_assignment_report(review):
    """Export reviewed choices separately for normal scientific qualification."""
    from .assignment_checkpoint import assignment_fingerprint
    from .assignment_search import assignments_to_observations
    content=deepcopy(review);stored=content.pop('review_sha256',None)
    if stored!=assignment_fingerprint(content):raise ValueError('review content hash mismatch')
    lookup={str(c['id']):c for c in review['catalogue']};rows=deepcopy(review['rows'])
    for row in rows:
        pid=str(row['peak']['id'])
        if pid not in review['decisions']:continue
        cid=review['decisions'][pid]
        row.update(status='unassigned' if cid is None else 'accepted',catalogue_id=cid,
                   transition=None if cid is None else deepcopy(lookup[cid]))
    report=dict(schema='matrix.lock.assignment_candidates.v1',rows=rows,
        accepted_count=sum(r['status']=='accepted' for r in rows),review_sha256=stored,
        scope='reviewed proposals; must pass normal covariance, overlap, cycle and fit checks')
    return dict(schema='matrix.lock.reviewed_assignments.v1',report=report,
                observations=assignments_to_observations(report),review_sha256=stored)


__all__=['build_assignment_review','record_assignment_decision','reviewed_assignment_report']
