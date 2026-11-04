"""Conservative handoff from exploratory assignments to native progressive fits."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import numpy as np


def transport_assignment_observations(reference, current, observations, *,
        minimum_principal_overlap=.9, minimum_state_overlap=.8,
        degeneracy_tolerance_MHz=1.e-7):
    """Preserve IDs through native overlap transport on a common primitive grid.

    Individual assignments within unresolved degenerate spans are blocked;
    this interface never interprets a rotated projector basis as eigenstates.
    """
    from .lam_fit import track_lam_rotation_states
    from .assignment_search import assignment_components

    if (not 0 < minimum_principal_overlap <= 1 or not 0 < minimum_state_overlap <= 1
            or not np.isfinite(degeneracy_tolerance_MHz) or degeneracy_tolerance_MHz < 0):
        raise ValueError('valid overlap and degeneracy thresholds required')
    if reference.representation != current.representation:
        raise ValueError('assignment transport requires the same axis representation')
    # Results retain a basis grid but not its original coordinate arrays.
    # The high-level driver additionally verifies surface identity by retaining
    # a single original surface throughout all stages.
    selected = {j:set() for j in reference.blocks}
    ids = {}
    for row in [c for r in observations for c in assignment_components(r)]:
        for side in ('lower', 'upper'):
            state = row[side]; j = state['J']; index = state['reference_index']
            if j not in reference.blocks or j not in current.blocks:
                raise ValueError('assigned J block missing during promotion')
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(reference.blocks[j].levels):
                raise ValueError('assigned reference index outside reference block')
            if not state.get('state_id'):
                raise ValueError('persistent state IDs required for promotion')
            key = (j,index)
            if state['state_id'] in ids and ids[state['state_id']] != key:
                raise ValueError('one state ID refers to inconsistent reference levels')
            ids[state['state_id']] = key; selected[j].add(index)
    if not ids:
        raise ValueError('no accepted assigned states to promote')
    try:
        mapping, diagnostic = track_lam_rotation_states(reference, current,
            diagnostic_indices=selected, subspace_gap_MHz=degeneracy_tolerance_MHz)
    except ValueError as exc:
        return dict(ready=False,reason='incomplete_or_incompatible_state_span',
                    diagnostic=dict(message=str(exc)), observations=None,registry=[])
    if (diagnostic['minimum_principal_overlap'] < minimum_principal_overlap
            or diagnostic['minimum_state_overlap'] < minimum_state_overlap):
        return dict(ready=False, reason='insufficient_state_overlap', diagnostic=diagnostic,
                    observations=None, registry=[])
    registry = []
    for state_id,(j,index) in sorted(ids.items()):
        target = mapping[j][index]
        before = reference.blocks[j].levels[index]; after = current.blocks[j].levels[target]
        if (before.symmetry_sector != after.symmetry_sector
                or before.rovibrational_irrep != after.rovibrational_irrep):
            return dict(ready=False, reason='symmetry_changed', diagnostic=diagnostic,
                        observations=None, registry=[])
        for block,k in ((reference.blocks[j],index),(current.blocks[j],target)):
            level = block.levels[k]
            partners = [p for p in block.levels if p.symmetry_sector == level.symmetry_sector
                and p.rovibrational_irrep == level.rovibrational_irrep
                and abs(p.energy_MHz-level.energy_MHz) <= degeneracy_tolerance_MHz]
            if len(partners) > 1:
                return dict(ready=False, reason='degenerate_assignment_requires_component_model',
                            diagnostic=diagnostic, observations=None, registry=[])
        registry.append(dict(state_id=state_id, J=j, reference_index=index,
            current_index=target, current_energy_MHz=float(after.energy_MHz)))
    result = deepcopy(observations)
    for row in [c for r in result for c in assignment_components(r)]:
        for side in ('lower','upper'):
            state = row[side]; j = state['J']; target = mapping[j][state['reference_index']]
            state['reference_index'] = int(target)
            state['reference_energy_MHz'] = float(current.blocks[j].energies_MHz[target])
    return dict(ready=True, reason='overlap_and_symmetry_preserved', diagnostic=diagnostic,
                observations=result, registry=registry)


def refine_exploratory_assignments(surface, fit_request, assignment_report, reference,
        stages, *, minimum_principal_overlap=.9, minimum_state_overlap=.8,
        degeneracy_tolerance_MHz=1.e-7, maximum_total_evaluations=60,
        checkpoint_path=None, resume_checkpoint=None, stage_limit=None,
        _resume_state=None, _checkpoint_callback=None):
    """Release nested variable sets, warm starting only accepted native stages.

    The complete supplied Hamiltonian is restored at the first fit stage;
    inactive parameters remain at QM/last accepted values. The existing global
    affine-target fitter freezes directions without rebasing scale/offset
    operators. All correlated priors and predicates remain in local space.
    Model enlargement and assignment transport are separate from fit success.
    This is the resolved spin-free LAM adapter; the candidate matcher itself
    also accepts catalogues supplied by other native spin/rigid backends.
    """
    if fit_request.get('spin_model') is not None:
        from .assignment_spin import refine_spin_assignments
        return refine_spin_assignments(surface,fit_request,assignment_report,reference,stages,
            minimum_principal_overlap=minimum_principal_overlap,minimum_state_overlap=minimum_state_overlap,
            degeneracy_tolerance_MHz=degeneracy_tolerance_MHz,maximum_total_evaluations=maximum_total_evaluations,
            checkpoint_path=checkpoint_path,resume_checkpoint=resume_checkpoint,stage_limit=stage_limit,
            _resume_state=_resume_state,_checkpoint_callback=_checkpoint_callback)
    from .assignment_search import assignments_to_observations,assignment_components
    from .general_lam_operator import normalize_general_lam_operator_manifest
    from .lam_fit import _lam_variable, _apply_lam_variables, _operator_parameter_values
    from .lam_global_fit import _backend, fit_global_lam_spectra
    from .lam_rotation import solve_lam_rotation
    from .assignment_checkpoint import assignment_fingerprint,load_assignment_checkpoint,save_assignment_checkpoint
    from .assignment_covariance import assignment_fit_covariance

    template = deepcopy(fit_request); stages = deepcopy(list(stages))
    if template.get('schema') != 'matrix.lock.lam_spectral_fit.v1':
        raise ValueError('native LAM fit template required')
    if reference.representation != 'Ir':
        raise ValueError('the native LAM fit adapter currently requires the Ir representation')
    if 'structured_discrepancy' in template:
        raise ValueError('discrepancy calibration follows the assignment stage on separate declared rows')
    if (isinstance(maximum_total_evaluations,bool) or not isinstance(maximum_total_evaluations,int)
            or maximum_total_evaluations < 1 or not stages):
        raise ValueError('positive total evaluation budget and nonempty stages required')
    paths = [v['path'] for v in template.get('variables',())]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError('unique native fit variable paths required')
    active = set(); count = len(reference.basis.energies_MHz); total = 0; names = set()
    for stage in stages:
        chosen = set(stage['active_paths']); name = stage['name']; states = stage.get('lam_states',count)
        budget = stage.get('max_nfev',20)
        if (not name or name in names or not chosen or chosen-set(paths) or not active<=chosen
                or len(chosen)!=len(stage['active_paths'])):
            raise ValueError('stages need unique names and nested nonempty declared variable paths')
        if (isinstance(states,bool) or not isinstance(states,int) or states<count
                or isinstance(budget,bool) or not isinstance(budget,int) or budget<1):
            raise ValueError('stage LAM spaces must grow and fit budgets must be positive integers')
        active=chosen; count=states; total+=budget; names.add(name)
    if total > maximum_total_evaluations:
        raise ValueError('declared stage budgets exceed total budget')
    if stage_limit is not None and (isinstance(stage_limit,bool) or not isinstance(stage_limit,int) or stage_limit<1):
        raise ValueError('stage_limit must be a positive integer')
    contract=assignment_fingerprint(surface,template,assignment_report,stages,
        dict(minimum_principal_overlap=minimum_principal_overlap,minimum_state_overlap=minimum_state_overlap,
             degeneracy_tolerance_MHz=degeneracy_tolerance_MHz,maximum_total_evaluations=maximum_total_evaluations))
    if resume_checkpoint is None and checkpoint_path is not None and Path(checkpoint_path).exists():
        resume_checkpoint=checkpoint_path
    if resume_checkpoint is not None:
        if _resume_state is not None:raise ValueError('duplicate assignment resume states')
        _resume_state=load_assignment_checkpoint(resume_checkpoint,contract=contract)
        if checkpoint_path is None:checkpoint_path=resume_checkpoint
    observations = assignments_to_observations(assignment_report)
    if not observations:
        return dict(schema='matrix.lock.assignment_refinement.v1',status='no_accepted_assignments',
                    completed=False,stages=[],observations=[],last_accepted_values={})
    template,covariance_audit=assignment_fit_covariance(template,assignment_report,observations)
    options = template.setdefault('options',{})
    if options.get('tracking','overlap') != 'overlap':
        raise ValueError('progressive assignment refinement requires overlap tracking')
    if options.get('basis_convergence') is not None:
        raise ValueError('assignment refinement uses declared fixed grids; use normal basis qualification afterwards')
    if options.get('lam_contraction','moving') != 'moving':
        raise ValueError('progressive assignment refinement requires moving LAM contraction')
    if any(k in options for k in ('resume_workspace','resume_checkpoint','optimizer_state',
            '_workspace_resume_state','workspace_checkpoint_path','checkpoint_path',
            'linearization_path','evaluation_log_path','k_contraction_states',
            'k_contraction_reference_values')):
        raise ValueError('stage-local checkpoint/contraction options must not be reused across distinct models')
    if surface.spin_tensor_surfaces_MHz and any(np.any(v) for v in surface.spin_tensor_surfaces_MHz.values()):
        raise ValueError('spin-active refinement requires its native spin adapter')
    manifest = template.get('operator_manifest')
    normalized = None if manifest is None else normalize_general_lam_operator_manifest(manifest)
    defaults = {} if normalized is None else normalized.parameters_MHz
    parsed = [_lam_variable(v,surface,i,defaults) for i,v in enumerate(template['variables'])]
    values = np.array([v['initial'] for v in parsed]); accepted = dict(zip(paths,values.tolist(),strict=True))
    controls = {k:options[k] for k in ('primitive_basis','fourier_cutoffs','assembly_backend',
        'diagonalizer_backend','workers') if k in options}
    records = []; completed = False; start_stage=0
    if _resume_state is not None:
        if _resume_state.get('contract')!=contract:
            raise ValueError('assignment refinement checkpoint contract mismatch')
        if _resume_state.get('terminal_report') is not None:
            return deepcopy(_resume_state['terminal_report'])
        reference=_resume_state['reference']; observations=deepcopy(_resume_state['observations'])
        values=np.asarray(_resume_state['values'],float); records=deepcopy(_resume_state['records'])
        start_stage=int(_resume_state['next_stage']);accepted=dict(zip(paths,values.tolist(),strict=True))
    js=sorted(reference.blocks)
    def snapshot(next_stage,terminal_report=None):
        state=dict(contract=contract,reference=reference,observations=observations,
            values=values.copy(),records=records,next_stage=next_stage,terminal_report=terminal_report)
        if checkpoint_path is not None:save_assignment_checkpoint(checkpoint_path,state,contract=contract)
        if _checkpoint_callback is not None:_checkpoint_callback(state)
    snapshot(start_stage)
    executed=0
    for stage_index in range(start_stage,len(stages)):
        if stage_limit is not None and executed>=stage_limit:break
        stage=stages[stage_index];executed+=1
        states = stage.get('lam_states',len(reference.basis.energies_MHz))
        current = solve_lam_rotation(_apply_lam_variables(surface,parsed,values),
            j_values=js, lam_state_count=states, representation=reference.representation,
            operator_manifest=normalized,
            operator_parameters_MHz=_operator_parameter_values(normalized,parsed,values), **controls)
        handoff = transport_assignment_observations(reference,current,observations,
            minimum_principal_overlap=minimum_principal_overlap,minimum_state_overlap=minimum_state_overlap,
            degeneracy_tolerance_MHz=degeneracy_tolerance_MHz)
        entry = dict(name=stage['name'],active_paths=stage['active_paths'],lam_states=states,
                     initial_values=dict(zip(paths,values.tolist(),strict=True)),handoff=handoff)
        records.append(entry)
        if not handoff['ready']:
            entry['status']='assignment_transport_blocked'; break
        local = deepcopy(template); local['observations']=handoff['observations']; local['lam_states']=states
        local['representation']=reference.representation
        for variable,value in zip(local['variables'],values,strict=True): variable['initial']=float(value)
        backend = _backend(surface,local)
        # Preserve the original local prior/predicate covariance, including
        # active--inactive cross terms. Global targets release only chosen paths.
        variables=[]
        for i,path in enumerate(paths):
            if path in stage['active_paths']:
                p=parsed[i]
                variables.append(dict(name=p['name'],initial=float(values[i]),scale=p['scale'],
                    lower=p['lower'],upper=p['upper'],targets=[dict(system='molecule',path=path)]))
        global_request=dict(schema='matrix.lock.global_lam_spectral_fit.v1',
            datasets=[dict(system='molecule',request=local)], variables=variables,
            options={**options,'max_nfev':stage.get('max_nfev',20)})
        try:
            fit=fit_global_lam_spectra({'molecule':surface},global_request,_backends={'molecule':backend})
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            entry.update(status='fit_failed',message=str(exc)); break
        entry['fit']=fit
        if not fit['success']:
            entry['status']='fit_not_converged'; break
        trial=np.array([r['value'] for r in fit['local_parameters']])
        predicted=backend['predict'](trial)
        updated=transport_assignment_observations(current,predicted[2],handoff['observations'],
            minimum_principal_overlap=minimum_principal_overlap,minimum_state_overlap=minimum_state_overlap,
            degeneracy_tolerance_MHz=degeneracy_tolerance_MHz)
        entry['postfit_handoff']=updated
        if not updated['ready']:
            entry['status']='postfit_assignment_transport_blocked'; break
        native_states=fit['component_models']['molecule']
        flattened=[c for r in updated['observations'] for c in assignment_components(r)]
        if any(row[side]['reference_index'] != native[side]['reference_index']
               for row,native in zip(flattened,native_states,strict=True)
               for side in ('lower','upper')):
            entry['status']='native_continuation_disagrees_with_handoff'; break
        entry['status']='accepted'; values=trial; reference=predicted[2]; observations=updated['observations']
        accepted=dict(zip(paths,values.tolist(),strict=True))
        snapshot(stage_index+1)
    else:
        completed=True
    paused=not completed and records and records[-1]['status']=='accepted'
    report=dict(schema='matrix.lock.assignment_refinement.v1',
        status='checkpointed' if paused else 'completed' if completed else records[-1]['status'],completed=completed,
        stages=records,observations=observations,last_accepted_values=accepted,
        measurement_covariance=covariance_audit,
        resumed_from_stage=start_stage,executed_stages=executed,
        declared_evaluation_budget=total,scope='assignment-assisted refinement; not precision basis qualification')
    snapshot(sum(r['status']=='accepted' for r in records),None if paused else report)
    return report


__all__=['transport_assignment_observations','refine_exploratory_assignments']
