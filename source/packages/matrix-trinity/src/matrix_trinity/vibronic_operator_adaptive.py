"""Adaptive complex operator banks with positive-norm omission certificates."""
import numpy as np
from time import perf_counter
from .vibronic_storage import ArrayBlocks
from .vibronic_harmonic import harmonic_thermal_factors
from .vibronic_selection import ConditionalOccupationTree, PackedHermiteCache
from .vibronic_kernels import parallel_rows


def adaptive_operator_ti(model, direction, c, h, w, g, constant, linear, common, *,
        temperature_K, maximum_initial_quanta, maximum_final_quanta, maximum_class,
        maximum_states, maximum_transitions, maximum_integrals, workers, target_fraction,
        selection_batch_size, selection_maximum_per_mode_quanta, checkpoint_path=None,
        resume=False, checkpoint_interval_batches=10, checkpoint_interval_seconds=60.,selection_proposal='gaussian_c2',
        native_library=None,selection_energy_bias_per_cm=0.,selection_channel_weights=None,spectral_target=None,assignment_report_path=None):
    started=perf_counter()
    timings=dict(selection=0.,recurrence=0.,strengths=0.,checkpoint=0.)
    if selection_proposal not in ('gaussian_c2','operator_c2'):
        raise ValueError('selection_proposal must be gaussian_c2 or operator_c2')
    from .vibronic_observables import _operator_rows
    from .vibronic_checkpoint import AdaptiveCheckpoint
    n = len(model.initial_frequencies_cm)
    maximum_class = n if maximum_class is None else maximum_class
    for value in (maximum_initial_quanta, maximum_final_quanta, maximum_class):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError('nonnegative integer quanta/class limits required')
    for value in (maximum_states, maximum_transitions, maximum_integrals,
                  selection_batch_size, selection_maximum_per_mode_quanta, checkpoint_interval_batches):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('positive integer selection/cache/checkpoint budgets required')
    if not np.isfinite(target_fraction) or not 0 < target_fraction <= 1:
        raise ValueError('target fraction must be in (0,1]')
    if g['log_ground'] < np.log(np.finfo(float).tiny)+4:
        raise FloatingPointError('operator bank TI overlap underflows; use TD')
    q = harmonic_thermal_factors(model.initial_frequencies_cm, temperature_K)
    budget = min(maximum_states, maximum_transitions)
    cap = max(maximum_initial_quanta, maximum_final_quanta, selection_maximum_per_mode_quanta)
    identity = dict(engine='operator-bank', temperature_K=temperature_K, direction=direction,
        qi=maximum_initial_quanta, qf=maximum_final_quanta, maximum_class=maximum_class,
        per_mode=selection_maximum_per_mode_quanta, packing_cap=cap)
    if selection_proposal!='gaussian_c2':
        identity['selection_proposal']='operator_c2_v2'
    if spectral_target is not None:spectral_target.requested_fraction=target_fraction
    from .vibronic_window_selection import ranking_bias,WindowOccupationForest
    selection_energy_bias_per_cm=ranking_bias(spectral_target,model,g,temperature_K,direction,selection_energy_bias_per_cm)
    use_forest=np.ndim(selection_energy_bias_per_cm)>0
    if use_forest:identity['window_rankings_v1']=selection_energy_bias_per_cm.tolist()
    elif selection_energy_bias_per_cm:identity['selection_energy_bias_per_cm']=float(selection_energy_bias_per_cm)
    if spectral_target is not None:spectral_target.observable_norms=np.max(abs(np.linalg.eigvalsh(w)),axis=1)
    if selection_channel_weights is not None:
        selection_channel_weights=np.asarray(selection_channel_weights,float)
        if (selection_proposal!='operator_c2' or selection_channel_weights.shape!=constant.shape
                or not np.all(np.isfinite(selection_channel_weights)) or np.any(selection_channel_weights<=0)):
            raise ValueError('positive finite per-channel weights require operator_c2')
        identity['selection_channel_weights']=selection_channel_weights.tolist()
    checkpoint = AdaptiveCheckpoint(checkpoint_path, resume, identity,
        dict(wi=model.initial_frequencies_cm, wf=model.final_frequencies_cm,
             mode_map=model.mode_map, shift=model.final_shift_dimensionless,
             origin=np.asarray(model.origin_cm), c=c, h=h, w=w), checkpoint_interval_seconds)
    cache = PackedHermiteCache(g, linear, maximum_integrals, cap,native_library=native_library)
    tree, payload = checkpoint.restore(cache, budget)
    if tree is None:
        controls=dict(qi=0 if temperature_K==0 else maximum_initial_quanta,qf=maximum_final_quanta,
            maximum_class=maximum_class,operator_constant=constant,operator_linear=linear,proposal=selection_proposal,
            gaussian=g,native_library=native_library,channel_weights=selection_channel_weights)
        if use_forest:
            tree=WindowOccupationForest(model,temperature_K,selection_maximum_per_mode_quanta,budget,
                biases=selection_energy_bias_per_cm,target=spectral_target,fraction=target_fraction,packing_cap=cap,**controls)
        else:
            tree=ConditionalOccupationTree(model,temperature_K,selection_maximum_per_mode_quanta,budget,
                energy_bias_per_cm=selection_energy_bias_per_cm,**controls)
    elif native_library is not None:tree.enable_native(native_library)
    if getattr(tree,'is_window_forest',False):tree.attach(spectral_target,target_fraction)
    states_kept = [payload['states']] if 'states' in payload else []
    strengths_kept = [payload['strengths']] if 'strengths' in payload else []
    norm_parts = [payload['norms']] if 'norms' in payload else []
    pending = payload.get('pending', np.empty((0, 2*n), np.int64))
    if spectral_target is not None and 'states' in payload:
        for start in range(0,len(payload['states']),8192):
            states=payload['states'][start:start+8192]
            energies=direction*(model.origin_cm+states[:,n:]@model.final_frequencies_cm
                -states[:,:n]@model.initial_frequencies_cm)
            spectral_target.add(energies,payload['norms'][start:start+8192],payload['strengths'][:,start:start+8192])
    norm = float(sum(part.sum() for part in norm_parts))
    rule = common['bank_norm_sum_rule']; exhausted = False; batches = 0
    def save():
        if checkpoint.path is None: return
        before=perf_counter()
        checkpoint.save(tree, cache, dict(states=ArrayBlocks(states_kept,empty=np.empty((0,2*n),np.int64)),
            strengths=ArrayBlocks(strengths_kept,axis=1,empty=np.empty((len(w),0))),
            norms=ArrayBlocks(norm_parts,empty=np.empty(0)),pending=pending))
        timings['checkpoint']+=perf_counter()-before
    timings['preparation']=perf_counter()-started
    def reached_target():
        return spectral_target.converged(target_fraction) if spectral_target is not None else norm>=target_fraction*rule
    while (len(pending) or (tree.heap and tree.visited < budget)) and not reached_target():
        before=perf_counter()
        states = pending if len(pending) else tree.batch(selection_batch_size)
        timings['selection']+=perf_counter()-before
        if not len(states):
            continue
        before=perf_counter();ids, exhausted = cache.evaluate(states)
        timings['recurrence']+=perf_counter()-before
        pending = states[len(ids):].copy()
        states = states[:len(ids)]
        if len(states):
            before=perf_counter()
            if native_library is not None:
                strengths,norms=native_library.strengths(states,ids,cache,constant,linear,w,q)
            else:
                amplitudes = np.empty((len(states), len(c)), complex)
                parallel_rows(_operator_rows, len(states), workers, states, ids, cache.values, constant, linear, amplitudes)
                populations = np.prod(1-q)*np.prod(q[None, :]**states[:, :n], axis=1)
                strengths = np.einsum('ta,jab,tb->jt', amplitudes, w, amplitudes.conj(), optimize=True).real*populations
                norms = np.sum(abs(amplitudes)**2, axis=1)*populations
            timings['strengths']+=perf_counter()-before
            if spectral_target is not None:
                energies=direction*(model.origin_cm+states[:,n:]@model.final_frequencies_cm
                    -states[:,:n]@model.initial_frequencies_cm)
                spectral_target.add(energies,norms,strengths)
            norm += float(norms.sum())
            states_kept.append(states.astype(np.min_scalar_type(cap))); strengths_kept.append(strengths); norm_parts.append(norms)
        if norm > rule+1e-8*max(rule, 1.):
            raise RuntimeError('operator bank violates positive total-strength sum rule')
        batches += 1
        if batches % checkpoint_interval_batches == 0 and checkpoint.due():
            save()
        if exhausted:
            break
    save()
    before=perf_counter()
    states = np.concatenate(states_kept) if states_kept else np.empty((0, 2*n), np.int64)
    strengths = np.concatenate(strengths_kept, axis=1) if strengths_kept else np.empty((len(w), 0))
    missing = max(0., rule-norm)
    fraction = norm/rule if rule else 1.
    # The cold selector admits only the initial vacuum. Sorting millions of
    # identical 75-column initial occupations adds no information; the finite-
    # temperature path still computes its exact set of distinct occupations.
    if temperature_K==0:
        population=float(bool(len(states)))
    else:
        unique = np.unique(states[:, :n], axis=0)
        population = float(np.sum(np.prod(1-q)*np.prod(q[None, :]**unique, axis=1)))
    energies=model.origin_cm+states[:, n:] @ model.final_frequencies_cm
    if temperature_K!=0:energies-=states[:, :n] @ model.initial_frequencies_cm
    targeted=None if spectral_target is None else spectral_target.report(direction*energies,strengths,
        np.max(abs(np.linalg.eigvalsh(w)),axis=1))
    assignment_report=None
    if spectral_target is not None:
        from .vibronic_assignment_report import targeted_assignment_report,save_targeted_assignment_report
        def components(indices):
            selected=states[indices].astype(np.int64);ids,failed=cache.evaluate(selected)
            if failed:raise RuntimeError('reported assignments lost their cached integrals')
            values=[]
            for base,insertions in [(c,np.zeros_like(linear)),(constant-c,linear)]:
                if native_library is not None:
                    value,_=native_library.strengths(selected,ids,cache,base,insertions,w,q)
                else:
                    amplitudes=np.empty((len(selected),len(c)),complex)
                    parallel_rows(_operator_rows,len(selected),workers,selected,ids,cache.values,base,insertions,amplitudes)
                    populations=np.prod(1-q)*np.prod(q[None,:]**selected[:,:n],axis=1)
                    value=np.einsum('ta,jab,tb->jt',amplitudes,w,amplitudes.conj(),optimize=True).real*populations
                values.append(value)
            return dict(FC=values[0],HT=values[1],interference=strengths[:,indices]-values[0]-values[1])
        assignment_report=targeted_assignment_report(spectral_target,direction*energies,strengths,np.concatenate(norm_parts) if norm_parts else np.empty(0),
            states[:,:n],states[:,n:],components=components)
        if assignment_report_path is not None:save_targeted_assignment_report(assignment_report,assignment_report_path)
    timings['materialization']=perf_counter()-before;timings['total']=perf_counter()-started
    return dict(**common, assignment_report=assignment_report,wavenumbers_cm=direction*energies,
        strengths=strengths, initial_quanta=states[:, :n], final_quanta=states[:, n:],
        diagnostics=dict(timing_seconds=timings,cached_integrals=cache.count, recurrence_dependency_edges=cache.edge_count,
            evaluated_transitions=len(states), visited_proposal_states=tree.visited,
            queued_proposal_states=len(tree.heap), pending_transitions=len(pending),
            initial_population_retained=population, computed_bank_norm=norm, missing_bank_norm=missing,
            observable_omission_bounds=np.max(abs(np.linalg.eigvalsh(w)), axis=1)*missing,
            captured_fraction=fraction, target_fraction=target_fraction, converged=reached_target(),global_norm_converged=bool(fraction>=target_fraction),
            convergence_scope='global' if spectral_target is None else 'requested_spectral_windows',targeted_selection=targeted,
            transition_budget_exhausted=bool(tree.visited >= budget and not reached_target()),
            proposal_domain_exhausted=bool(not tree.heap and not len(pending) and not reached_target()),
            cache_budget_exhausted=exhausted, workers=1 if native_library is not None else workers, finite_state_sum_used=True,
            numerical_backend='C++' if native_library is not None else 'compiled Python',
            selection_backend=('conditional operator-block C2 tree; positive operator-bank norm stopping'
                if selection_proposal=='operator_c2' else 'conditional Gaussian C2 tree; positive operator-bank norm stopping'),
            selection_proposal=selection_proposal,
            selection_energy_bias_per_cm=np.asarray(selection_energy_bias_per_cm).tolist(),
            window_ranking_statistics=tree.statistics() if use_forest else None,
            selection_channel_weights=(None if selection_channel_weights is None else selection_channel_weights.tolist()),
            physical_strengths_reweighted=False,
            selection_proposal_version=2 if selection_proposal=='operator_c2' else 1,
            proposal_is_physical_probability=False, signed_strengths_clipped=False,
            maximum_class_evaluated=int(np.max(np.count_nonzero(states[:, n:], axis=1), initial=0)),
            checkpoint_resumed=checkpoint.resumed, checkpoint_path=checkpoint.path,
            no_renormalization=True))
