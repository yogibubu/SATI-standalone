"""A bounded exploration--assignment--refinement workflow for LOCK."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path


def run_assignment_workflow(surface, configuration, *, checkpoint_path=None,resume_checkpoint=None,stage_limit=None):
    """Use one source surface, one declared search, and optional finite fit stages."""
    from .assignment_exploration import build_assignment_exploration, exploration_catalog
    from .assignment_search import propose_spectral_assignments,group_assignment_blends
    from .assignment_refinement import refine_exploratory_assignments
    from .assignment_checkpoint import assignment_fingerprint,load_assignment_checkpoint,save_assignment_checkpoint

    config = deepcopy(configuration)
    if config.pop('schema',None) != 'matrix.lock.assignment_workflow.v1':
        raise ValueError('unsupported assignment workflow schema')
    unknown = set(config)-{'exploration','peaks','search','fit_request','stages','refinement','blends','alternatives'}
    if unknown:
        raise ValueError(f'unknown assignment workflow fields: {sorted(unknown)}')
    contract=assignment_fingerprint(surface,configuration)
    state={}
    if resume_checkpoint is None and checkpoint_path is not None and Path(checkpoint_path).exists():
        resume_checkpoint=checkpoint_path
    if resume_checkpoint is not None:
        state=load_assignment_checkpoint(resume_checkpoint,contract=contract)
        if checkpoint_path is None:checkpoint_path=resume_checkpoint
        if state.get('terminal_result') is not None:return deepcopy(state['terminal_result'])
    def save():
        if checkpoint_path is not None:save_assignment_checkpoint(checkpoint_path,state,contract=contract)
    exploration = config['exploration']
    fit = config.get('fit_request')
    if fit is not None:
        for key in ('temperature_K','dipole_au'):
            if key in exploration:fit.setdefault(key,deepcopy(exploration[key]))
        roles={str(r.get('label',r.get('id'))):r.get('role','training') for r in fit.get('observations',())}
        for peak in config['peaks']:
            if str(peak['id']) in roles:
                role=roles[str(peak['id'])]
                if 'role' in peak and peak['role']!=role:
                    raise ValueError('peak and source measurement roles disagree')
                peak['role']=role
    model = build_assignment_exploration(surface,
        rotational_order=exploration.get('rotational_order',2),
        operator_manifest=None if fit is None else fit.get('operator_manifest'),
        essential_parameters=exploration.get('essential_parameters',()))
    if 'reference' not in state:
        reference,catalogue=exploration_catalog(model,
            spin_model=None if fit is None else fit.get('spin_model'),
            **{k:v for k,v in exploration.items() if k not in ('rotational_order','essential_parameters')})
        if config.get('blends') is not None:
            catalogue=group_assignment_blends(catalogue,**config['blends'])
        state.update(reference=reference,catalogue=catalogue,phase='catalogue');save()
    reference=state['reference'];catalogue=state['catalogue']
    if 'assignments' not in state:
        state.update(assignments=propose_spectral_assignments(config['peaks'],catalogue,
                     **config.get('search',{})),phase='assigned');save()
    assignments=state['assignments']
    refinement = None
    alternatives = None
    if config.get('alternatives') is not None:
        from .assignment_hypotheses import compare_assignment_hypotheses
        if stage_limit is not None:
            raise ValueError('stage-limit applies to single-map refinement, not hypothesis comparison')
        if fit is None:raise ValueError('fit request required for assignment hypotheses')
        def save_alternatives(progress):
            state.update(alternative_progress=progress,phase='comparing');save()
        alternatives=compare_assignment_hypotheses(surface,fit,assignments,catalogue,reference,
            config.get('stages',[]),refinement_options=config.get('refinement'),
            _resume=state.get('alternative_progress'),_checkpoint_callback=save_alternatives,
            **config['alternatives'])
        refinement=alternatives['selected_refinement']
    elif config.get('stages'):
        if fit is None:
            raise ValueError('fit request required for explicit refinement stages')
        def save_progress(progress):
            state.update(progress=progress,phase='refining');save()
        refinement = refine_exploratory_assignments(surface,fit,assignments,reference,
            config['stages'],stage_limit=stage_limit,_resume_state=state.get('progress'),
            _checkpoint_callback=save_progress,**config.get('refinement',{}))
    status = (alternatives['status'] if alternatives is not None else
              'assignment_proposals' if refinement is None else refinement['status'])
    result=dict(schema='matrix.lock.assignment_workflow_result.v1',status=status,
        exploration=model.audit,catalogue=catalogue,assignments=assignments,
        refinement=refinement,alternatives=alternatives,scope='exploratory assignments; precision qualification remains a separate stage')
    if refinement is None or refinement['status']!='checkpointed':
        state.update(phase='terminal',terminal_result=result);save()
    return result


__all__=['run_assignment_workflow']
