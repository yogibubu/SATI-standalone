"""Sequential, provenance-bound QC acquisition and native spectral refitting."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import hashlib,json,shutil
import numpy as np
from scipy.linalg import block_diag,cho_factor,cho_solve
from .qc_cycle import (_digest,_write,prepare_qc_cycle,collect_qc_cycle,
    refine_qc_pes_cycle,apply_qc_prior,_physical_qc_design)


def _design(specification,parameters,candidates,reference,names):
    return _physical_qc_design(specification,names,candidates,reference)


def conditional_qc_noise(original_covariance, observed_design, observed_noise,
                         new_noise, observed_new_noise, spectral_covariance,
                         spectral_parameter_indices):
    """Joint (theta, new QC noise) after prior QC and local spectral updates.

    New noise shares reference-energy errors with past QC observations.
    Conditioning must retain the induced parameter/noise correlation; simply
    appending the original new-noise block to a spectral posterior is wrong.
    The spectral update is a local Gaussian approximation, not an extra prior.
    """
    c0=np.asarray(original_covariance,float);h=np.asarray(observed_design,float)
    ca=np.asarray(observed_noise,float);rn=np.asarray(new_noise,float)
    ran=np.asarray(observed_new_noise,float);post=np.asarray(spectral_covariance,float)
    indices=np.asarray(spectral_parameter_indices,int)
    n=len(c0);m=len(rn)
    if (c0.shape!=(n,n) or h.ndim!=2 or h.shape[1]!=n or ca.shape!=(len(h),len(h))
        or ran.shape!=(len(h),m) or rn.shape!=(m,m) or post.ndim!=2 or post.shape[0]!=post.shape[1]
        or indices.shape!=(n,) or len(set(indices))!=n or np.any(indices<0) or np.any(indices>=len(post))
        or not all(np.all(np.isfinite(a)) for a in (c0,h,ca,rn,ran,post))):
        raise ValueError("incompatible sequential QC covariance blocks")
    np.linalg.cholesky(c0);np.linalg.cholesky(post)
    if len(h):
        factor=cho_factor(h@c0@h.T+ca,lower=True)
        cross_q=-c0@h.T@cho_solve(factor,ran)
        gain=cho_solve(factor,(c0@h.T).T).T
        r=np.eye(n)-gain@h
        cq=r@c0@r.T+gain@ca@gain.T
        noise_q=rn-ran.T@cho_solve(factor,ran)
    else:
        cq=c0;cross_q=np.zeros((n,m));noise_q=rn
    precision_cross=cho_solve(cho_factor(cq,lower=True),cross_q)
    cross=post[:,indices]@precision_cross
    noise=noise_q-cross_q.T@precision_cross+precision_cross.T@post[np.ix_(indices,indices)]@precision_cross
    noise=.5*(noise+noise.T)
    # Return actual conditional noise, including the update by experimental
    # information. select_qc_candidates validates its full joint covariance.
    return cross,noise


def _advance_qc_campaign(directory,configuration):
    """Advance one campaign to completion, or prepare the next GDV batch.

    configuration owns original fit_request, candidates, qc_request (the
    keyword arguments of prepare_qc_cycle, with original QM prior), and an
    acquisition policy. Files are imported only; no QC executable is launched.
    Each round is a *cumulative* refit of the original QM information and all
    distinct QC observations. Previously collected files are copied with
    hashes; no spectral posterior is reused as independent QM information.
    """
    from .structure_spectrum import fit_structure_pes_spectra
    from .qc_acquisition import select_qc_candidates
    root=Path(directory);path=root/'campaign.json';config=deepcopy(configuration)
    digest=_digest(config)
    original=deepcopy(config['fit_request']);qc=deepcopy(config['qc_request'])
    candidates=deepcopy(config['candidates']);policy=dict(config.get('acquisition',{}))
    ids=[str(r['id']) for r in candidates]
    if not ids or len(set(ids))!=len(ids):raise ValueError('unique nonempty candidate IDs required')
    names=list(qc['qm_prior_values']);all_names=[v['name'] for v in original['variables']]
    if set(names)-set(all_names):raise ValueError('QC parameter registry differs from spectral fit')
    if any(v['kind']!='pes' for v in original['variables'] if v['name'] in names):
        raise ValueError('sequential fixed-geometry QC updates require PES coefficient variables')
    indices=[all_names.index(n) for n in names]
    # Install precisely the original QM block also in the first spectral fit.
    # This binds acquisition's conditional covariance to the actual restraints.
    source=qc.setdefault('qm_source_id','QM_PES')
    original_initials={v['name']:v['initial'] for v in original['variables']}
    original=apply_qc_prior(original,dict(verified=True,parameter_names=names,
        qm_source_id=source,updated_values=qc['qm_prior_values'],
        updated_qm_covariance=qc['qm_prior_covariance'],refinement_id='original:'+digest))
    for variable in original['variables']:
        variable['initial']=original_initials[variable['name']]
    if original.get('options',{}).get('parameter_selection',{}).get('mode','report')!='report':
        raise ValueError('QC campaigns require a full identifiable spectral parameter covariance')
    sizes=[len(r['job']['coordinate'])+1 for r in candidates]
    offsets=np.r_[0,np.cumsum(sizes)]
    noise=np.asarray(config.get('noise_covariance',block_diag(*(r['noise_covariance'] for r in candidates))),float)
    if noise.shape!=(offsets[-1],offsets[-1]) or not np.all(np.isfinite(noise)):
        raise ValueError('campaign noise covariance does not span all candidates')
    reference_sigma=float(qc.pop('reference_energy_sigma_MHz',0.))
    if not np.isfinite(reference_sigma) or reference_sigma<0:raise ValueError('invalid reference-energy noise')
    reference_rows=np.zeros(offsets[-1]);reference_rows[offsets[:-1]]=1.
    noise=noise+reference_sigma**2*np.outer(reference_rows,reference_rows)
    np.linalg.cholesky(noise)
    if 'noise_covariance' in qc:raise ValueError('declare joint campaign noise on the campaign, not on an individual round')
    if 'candidates' in qc or 'selected_ids' in qc:raise ValueError('campaign owns cumulative candidate selection')
    maximum=int(policy.pop('maximum_rounds',5))
    target=policy.pop('target_maximum_sigma_MHz',None)
    target_rows=policy.pop('target_rows',None)
    if maximum<1 or (target is not None and (not np.isfinite(target) or target<=0)):
        raise ValueError('invalid acquisition stopping policy')
    state=json.loads(path.read_text()) if path.exists() else dict(schema='matrix.lock.qc_campaign.v1',
        configuration_sha256=digest,configuration=config,rounds=[],acquired_ids=[],status='initial')
    if state['configuration_sha256']!=digest or _digest(state['configuration'])!=digest:
        raise ValueError('campaign physical configuration changed')
    previous_ids=[]
    for i,entry in enumerate(state['rounds'],start=1):
        if (entry['cycle']!=f'round-{i:03d}/cycle.json'
            or entry['cumulative_ids']!=previous_ids+entry['selected_new_ids']
            or len(set(entry['cumulative_ids']))!=len(entry['cumulative_ids'])
            or set(entry['cumulative_ids'])-set(ids)):
            raise ValueError('campaign round registry changed or contains an invalid path')
        previous_ids=entry['cumulative_ids']
    if state.get('status') in {'target_reached','budget_exhausted','no_informative_candidates'}:
        return state
    root.mkdir(parents=True,exist_ok=True)
    _write(path,state)
    def prepare_round(entry, previous_entry=None):
        chosen=entry['cumulative_ids']
        selected_indices=[ids.index(i) for i in chosen]
        selected_rows=[j for i in selected_indices for j in range(offsets[i],offsets[i+1])]
        round_dir=root/Path(entry['cycle']).parent
        cycle=prepare_qc_cycle(round_dir,candidates,chosen,
            noise_covariance=noise[np.ix_(selected_rows,selected_rows)],**qc)
        ledger=json.loads(cycle.read_text())
        if previous_entry is not None:
            old_path=root/previous_entry['cycle'];old=json.loads(old_path.read_text())
            for oldjob,newjob in zip(old['jobs'],ledger['jobs'],strict=False):
                if oldjob['id']!=newjob['id']:
                    raise ValueError('cumulative QC job registry changed')
                for kind in ('log','fchk'):
                    source=old_path.parent/oldjob[kind];destination=round_dir/newjob[kind]
                    expected=old['results'][oldjob['id']][kind+'_sha256']
                    if hashlib.sha256(source.read_bytes()).hexdigest()!=expected:
                        raise ValueError('previous QC source changed during cumulative acquisition')
                    if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest()!=expected:
                        raise ValueError('cumulative QC result conflicts with previous round')
                    if not destination.exists():shutil.copyfile(source,destination)
        return ledger

    refinement=None
    if state['rounds']:
        current=state['rounds'][-1];cycle_path=root/current['cycle']
        ledger=prepare_round(current,state['rounds'][-2] if len(state['rounds'])>1 else None)
        if ledger['status'] not in {'collected','refined'}:
            ledger=collect_qc_cycle(cycle_path)
            if ledger['status'] not in {'collected','refined'}:
                state['status']='awaiting_QC';_write(path,state);return state
        refinement=refine_qc_pes_cycle(cycle_path)
        state['acquired_ids']=list(ledger['contract']['selected_ids'])
        # Source-identical cumulative update: replace, never append, the prior.
        current['refinement_id']=refinement['refinement_id']
    fit_request=deepcopy(original)
    if state.get('accepted_fit'):
        accepted=root/state['accepted_fit']['path']
        if hashlib.sha256(accepted.read_bytes()).hexdigest()!=state['accepted_fit']['sha256']:
            raise ValueError('accepted spectral checkpoint changed')
        previous=json.loads(accepted.read_text())
        fit_request=deepcopy(previous['restart_request'])
    if refinement is not None:fit_request=apply_qc_prior(fit_request,refinement)
    if state.get('accepted_fit'):
        for variable,value in zip(fit_request['variables'],previous['variables'],strict=True):
            if variable['name']!=value['name']:raise ValueError('accepted parameter order changed')
            variable['initial']=value['value']
        if 'optimizer_state' in fit_request.get('options',{}):
            fit_request['options']['optimizer_state']['values']=[v['value'] for v in previous['variables']]
    fit_request.setdefault('options',{})['return_spectral_sensitivities']=True
    fit=fit_structure_pes_spectra(fit_request)
    fit_path=root/f'fit-{len(state["rounds"]):03d}.json';_write(fit_path,fit)
    state['accepted_fit']=dict(path=fit_path.name,sha256=hashlib.sha256(fit_path.read_bytes()).hexdigest())
    if not fit['success'] or fit['augmented_rank']!=len(all_names):
        state['status']='fit_not_qualified';_write(path,state);return state
    for variable,fitted in zip(fit_request['variables'],fit['variables'],strict=True):
        value=float(fitted['value']);step=float(variable['scale'])
        if any(abs(value-float(variable[side]))<=1.e-8*step for side in ('lower','upper') if side in variable):
            state['status']='fit_not_qualified'
            state['reason']='bound-active posterior requires a constrained acquisition distribution'
            _write(path,state);return state
    covariance=np.asarray(fit['covariance'],float);np.linalg.cholesky(covariance)
    sensitivities=np.asarray(fit['spectral_sensitivities']['jacobian_MHz_per_parameter'])
    rows=np.arange(len(sensitivities)) if target_rows is None else np.asarray(target_rows,int)
    if rows.ndim!=1 or not len(rows) or len(set(rows))!=len(rows) or np.any(rows<0) or np.any(rows>=len(sensitivities)):
        raise ValueError('invalid acquisition target rows')
    g=sensitivities[rows]
    uncertainty=np.sqrt(np.maximum(0.,np.einsum('ij,jk,ik->i',g,covariance,g)))
    state['target_parameter_sigmas_MHz']=uncertainty.tolist()
    if target is not None and np.max(uncertainty)<=target:
        state['status']='target_reached';_write(path,state);return state
    if len(state['rounds'])>=maximum:
        state['status']='budget_exhausted';_write(path,state);return state
    acquired=[ids.index(i) for i in state['acquired_ids']]
    remaining=[i for i in range(len(ids)) if i not in acquired]
    if not remaining:
        state['status']='no_informative_candidates';_write(path,state);return state
    # Exact affine designs at fixed electronic geometries. Rebuild them from
    # the physical producer; input candidate designs cannot dictate the update.
    parameters={v['name']:v['value'] for v in fit['variables']}
    h=_design(qc['producer_specification'],parameters,candidates,qc['reference'],names)
    past_rows=[j for i in acquired for j in range(offsets[i],offsets[i+1])]
    new_rows=[j for i in remaining for j in range(offsets[i],offsets[i+1])]
    cross,conditional_noise=conditional_qc_noise(qc['qm_prior_covariance'],h[past_rows],
        noise[np.ix_(past_rows,past_rows)],noise[np.ix_(new_rows,new_rows)],
        noise[np.ix_(past_rows,new_rows)],covariance,indices)
    available=[]
    for i in remaining:
        record=deepcopy(candidates[i]);design=np.zeros((sizes[i],len(all_names)))
        design[:,indices]=h[offsets[i]:offsets[i+1]];record['design']=design.tolist()
        available.append(record)
    selection=select_qc_candidates(covariance,g,available,noise_covariance=conditional_noise,
        parameter_observation_cross_covariance=cross,**policy)
    if not selection['selected_ids']:
        state['status']='no_informative_candidates';state['last_selection']=selection
        _write(path,state);return state
    chosen=[*state['acquired_ids'],*selection['selected_ids']]
    round_dir=root/f'round-{len(state["rounds"])+1:03d}'
    entry=dict(cycle=str((round_dir/'cycle.json').relative_to(root)),
        cumulative_ids=chosen,selected_new_ids=selection['selected_ids'],selection=selection,
        starting_fit_sha256=state['accepted_fit']['sha256'])
    previous_entry=state['rounds'][-1] if state['rounds'] else None
    state['rounds'].append(entry)
    state['status']='awaiting_QC';_write(path,state)
    prepare_round(entry,previous_entry)
    return state


def advance_qc_campaign(directory,configuration):
    """Advance an imported-results campaign under an exclusive state lock."""
    import fcntl
    root=Path(directory)
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.campaign.lock').open('a+') as lock:
        try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another process is advancing this QC campaign") from exc
        try:return _advance_qc_campaign(root,configuration)
        finally:fcntl.flock(lock.fileno(),fcntl.LOCK_UN)
