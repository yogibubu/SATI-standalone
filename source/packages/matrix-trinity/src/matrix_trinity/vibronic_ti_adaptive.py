"""High-class TI with a compiled priority tree and exact completeness audit."""
from pathlib import Path
from time import perf_counter
import numpy as np
from .vibronic_storage import ArrayBlocks,LineBlockStore
from .vibronic_preparation import preparation_session
from .vibronic_harmonic import (orient_harmonic_model, harmonic_gaussian_generator,
    harmonic_strength_sum_rule, harmonic_thermal_factors)
from .vibronic_selection import ConditionalOccupationTree, PackedHermiteCache
from .vibronic_kernels import worker_count, parallel_rows, transition_amplitudes


@preparation_session
def harmonic_vibronic_ti_adaptive(model, *, temperature_K=0., kind='absorption', operator='FCHT',
        maximum_initial_quanta=128, maximum_final_quanta=128, maximum_class=None,
        maximum_states=100000, maximum_transitions=100000, maximum_integrals=500000,
        minimum_strength_au2=0., target_fraction=.999, workers=0,
        selection_batch_size=None, selection_maximum_per_mode_quanta=128, line_format="records", progress_callback=None,
        checkpoint_path=None, resume=False, checkpoint_interval_batches=10, checkpoint_interval_seconds=60.,
        selection_proposal='gaussian_c2',native_library=None,line_storage_path=None,preparation_cache=None,checkpoint_async=True,
        spectral_target=None,selection_energy_bias_per_cm=0.,assignment_report_path=None):
    started=perf_counter();timings=dict(preparation=0.,selection=0.,recurrence=0.,amplitudes_and_reductions=0.,
        line_management=0.,checkpoint=0.,materialization=0.)
    if line_storage_path is not None and line_format!='columns':raise ValueError('Disk-backed line storage requires columns')
    if selection_proposal not in ('gaussian_c2','operator_c2'):
        raise ValueError('selection_proposal must be gaussian_c2 or operator_c2')
    if line_format not in {"records", "columns"}:
        raise ValueError("line_format must be records or columns")
    if not isinstance(checkpoint_async,bool):raise ValueError('checkpoint_async must be boolean')
    automatic_batch=selection_batch_size is None
    if automatic_batch:
        selection_batch_size=256 if native_library is None else (8192 if native_library.recurrence_mode=='levels' else 4096)
    source = model; model, direction = orient_harmonic_model(model, kind)
    rule = harmonic_strength_sum_rule(source, temperature_K, kind=kind, operator=operator)
    n = len(model.initial_frequencies_cm)
    maximum_class = n if maximum_class is None else maximum_class
    for value in (maximum_initial_quanta, maximum_final_quanta, maximum_class):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError('nonnegative integer quanta/class limits required')
    for value in (maximum_states, maximum_transitions, maximum_integrals, selection_batch_size, selection_maximum_per_mode_quanta, checkpoint_interval_batches):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('positive integer selection/cache budgets required')
    if automatic_batch:selection_batch_size=min(selection_batch_size,maximum_states,maximum_transitions)
    if not np.isfinite(minimum_strength_au2) or minimum_strength_au2 < 0 or not 0 < target_fraction <= 1:
        raise ValueError('nonnegative strength floor and target fraction in (0,1] required')
    workers = worker_count(workers); q = harmonic_thermal_factors(model.initial_frequencies_cm, temperature_K)
    if preparation_cache is not None and native_library is None:raise ValueError('Preparation reuse requires native kernels')
    g=(harmonic_gaussian_generator(model) if preparation_cache is None else
        preparation_cache.checkpoint_gaussian(native_library,model,temperature_K))
    if g['log_ground'] < np.log(np.finfo(float).tiny)+4:
        raise FloatingPointError('adaptive TI ground overlap underflows; use the analytic TD engine')
    budget = min(maximum_states, maximum_transitions)
    from .vibronic_checkpoint import AdaptiveCheckpoint
    cap = max(maximum_initial_quanta, maximum_final_quanta, selection_maximum_per_mode_quanta)
    identity=dict(engine='dipole', temperature_K=temperature_K, kind=kind, operator=operator,
            qi=maximum_initial_quanta, qf=maximum_final_quanta, maximum_class=maximum_class,
            per_mode=selection_maximum_per_mode_quanta, packing_cap=cap,
            minimum_strength=minimum_strength_au2, line_format=line_format)
    if selection_proposal!='gaussian_c2':
        identity['selection_proposal']='operator_c2_v2'
    if spectral_target is not None:spectral_target.requested_fraction=target_fraction
    from .vibronic_window_selection import ranking_bias,WindowOccupationForest
    selection_energy_bias_per_cm=ranking_bias(spectral_target,model,g,temperature_K,direction,selection_energy_bias_per_cm)
    use_forest=np.ndim(selection_energy_bias_per_cm)>0
    if use_forest:identity['window_rankings_v1']=selection_energy_bias_per_cm.tolist()
    elif selection_energy_bias_per_cm:identity['selection_energy_bias_per_cm']=float(selection_energy_bias_per_cm)
    checkpoint = AdaptiveCheckpoint(checkpoint_path, resume,identity,
        dict(matrix=g['matrix'], linear=g['linear'], log_ground=np.asarray(g['log_ground']), ht_constant=g['ht_constant'],
            ht_linear=g['ht_linear'], dipole=model.transition_dipole_au,
            wi=model.initial_frequencies_cm, wf=model.final_frequencies_cm, origin=np.asarray(model.origin_cm)), checkpoint_interval_seconds,asynchronous=checkpoint_async)
    # The established Gaussian fingerprint remains the checkpoint identity.
    # Native preparation supplies the working arrays, with the same schema.
    if native_library is not None:
        g=(native_library.prepare_dipole(model,temperature_K) if preparation_cache is None else
            preparation_cache.gaussian(native_library,model,temperature_K))
    cache = PackedHermiteCache(g, g['ht_linear'], maximum_integrals, cap,native_library=native_library)
    tree, payload = checkpoint.restore(cache, budget)
    if tree is None:
        guide={}
        if selection_proposal=='operator_c2':
            guide=dict(operator_constant=(model.transition_dipole_au if operator=='FC' else
                g['ht_constant'] if operator=='HT' else model.transition_dipole_au+g['ht_constant']),
                operator_linear=np.zeros_like(g['ht_linear']) if operator=='FC' else g['ht_linear'])
        controls=dict(qi=0 if temperature_K==0 else maximum_initial_quanta,qf=maximum_final_quanta,
            maximum_class=maximum_class,proposal=selection_proposal,gaussian=g,native_library=native_library,**guide)
        if use_forest:
            tree=WindowOccupationForest(model,temperature_K,selection_maximum_per_mode_quanta,budget,
                biases=selection_energy_bias_per_cm,target=spectral_target,fraction=target_fraction,packing_cap=cap,**controls)
        else:
            tree=ConditionalOccupationTree(model,temperature_K,selection_maximum_per_mode_quanta,budget,
                energy_bias_per_cm=selection_energy_bias_per_cm,**controls)
    elif native_library is not None:tree.enable_native(native_library)
    if getattr(tree,'is_window_forest',False):tree.attach(spectral_target,target_fraction)
    store=None if line_storage_path is None else LineBlockStore(line_storage_path)
    rows = []; column_batches = []; total = 0.; reported = 0.; components = np.zeros(3)
    by_class = {}; initial_states = {}; cache_exhausted = False; evaluated = 0; batches = 0
    pending = payload.get('pending', np.empty((0, 2*n), np.int64))
    def records(batch):
        states, energies, populations, amplitudes, values, weighted = batch
        return [dict(initial_quanta=state[:n].tolist(), final_quanta=state[n:].tolist(),
            wavenumber_cm=float(energies[i]), population=float(populations[i]), FC_overlap=float(amplitudes[i, 0]),
            FC_strength_au2=float(amplitudes[i, 1]), HT_strength_au2=float(amplitudes[i, 2]),
            interference_au2=float(amplitudes[i, 3]), strength_au2=float(values[i]), weighted_strength_au2=float(weighted[i]))
            for i, state in enumerate(states)]
    if payload:
        total, reported, evaluated = float(payload['total']), float(payload['reported']), int(payload['evaluated'])
        components = payload['components']
        by_class = dict(zip(map(int, payload['classes']), map(float, payload['class_values']), strict=True))
        initial_states = {tuple(state): float(pop) for state, pop in zip(payload['initial_states'], payload['initial_populations'], strict=True)}
        batch = tuple(payload['column_'+str(k)] for k in range(6))
        if spectral_target is not None:spectral_target.add(batch[1],batch[5],batch[5][None])
        if store is None:column_batches.append(batch)
        else:store.append(batch)
        if line_format == 'records': rows.extend(records(batch))
    def save(final=False):
        if checkpoint.path is None: return
        empty = (np.empty((0, 2*n), np.int64), np.empty(0), np.empty(0), np.empty((0, 5)), np.empty(0), np.empty(0))
        columns = (store.arrays() if store is not None and store.rows else
            tuple(ArrayBlocks([b[k] for b in column_batches],empty=empty[k]) for k in range(6)))
        data = dict(total=total, reported=reported, evaluated=evaluated, components=components,
            classes=np.array(list(by_class), np.int64), class_values=np.array(list(by_class.values())),
            initial_states=np.array(list(initial_states), np.int64).reshape(-1, n) if temperature_K != 0 else np.empty((len(initial_states), 0), np.int64),
            initial_populations=np.array(list(initial_states.values())), pending=pending)
        data.update({'column_'+str(k): value for k, value in enumerate(columns)})
        before=perf_counter();checkpoint.save(tree, cache, data,final=final,append_only=True);timings['checkpoint']+=perf_counter()-before
    timings['preparation']=perf_counter()-started
    def reached_target():
        return spectral_target.converged(target_fraction) if spectral_target is not None else reported>=target_fraction*rule['total']
    try:
        while (len(pending) or (tree.heap and tree.visited < budget)) and (rule['total'] > 0 and not reached_target()):
            checkpoint.poll()
            if native_library is not None:
                native_tree=tree
                if use_forest:
                    if not len(pending):pending=tree.batch(selection_batch_size)
                    if not len(pending):continue
                    native_tree=tree.children[tree.last_window]
                (states,pending,amplitudes,columns,summary,class_weights,class_counts,cache_exhausted,costs)=native_library.dipole_ti_batch(
                    native_tree,cache,pending,selection_batch_size,model,g,q,direction,minimum_strength_au2,operator)
                for name,value in zip(('selection','recurrence','amplitudes_and_reductions'),costs):timings[name]+=float(value)
                energies,populations,values,weighted=columns.T
                total+=float(summary[0]);components+=summary[2:]
            else:
                before=perf_counter();states=pending if len(pending) else tree.batch(selection_batch_size)
                timings['selection']+=perf_counter()-before
                if not len(states):continue
                before=perf_counter();ids,cache_exhausted=cache.evaluate(states)
                timings['recurrence']+=perf_counter()-before
                pending=states[len(ids):].copy();states=states[:len(ids)]
                before=perf_counter();amplitudes=np.empty((len(states),5))
                parallel_rows(transition_amplitudes,len(states),workers,states,ids,cache.values,
                    model.transition_dipole_au,g['ht_constant'],g['ht_linear'],amplitudes)
                populations=np.prod(1-q)*np.prod(q[None,:]**states[:,:n],axis=1)
                values=amplitudes[:,1 if operator=='FC' else 2 if operator=='HT' else 4]
                weighted=populations*values;total+=float(weighted.sum())
                components+=(amplitudes[:,1:4]*populations[:,None]).sum(axis=0)
                energies=direction*(model.origin_cm+states[:,n:]@model.final_frequencies_cm-states[:,:n]@model.initial_frequencies_cm)
                timings['amplitudes_and_reductions']+=perf_counter()-before
            managed_at=perf_counter()
            if len(states):
                keep = weighted > minimum_strength_au2
                if spectral_target is not None:spectral_target.add(energies[keep],weighted[keep],weighted[None,keep])
                if line_format == 'columns' or checkpoint.path is not None:
                    batch=(states[keep].astype(np.min_scalar_type(max(maximum_initial_quanta, maximum_final_quanta))),
                        energies[keep], populations[keep], amplitudes[keep], values[keep], weighted[keep])
                    if store is None:column_batches.append(batch)
                    else:store.append(batch)
                    if line_format == 'columns': reported += float(weighted[keep].sum())
                if temperature_K == 0:
                    initial_states[()] = 1.
                else:
                    unique, first = np.unique(states[:, :n], axis=0, return_index=True)
                    for state, index in zip(unique, first, strict=True):
                        initial_states[tuple(state)] = float(populations[index])
                if native_library is None:
                    classes=np.count_nonzero(states[:,n:],axis=1)
                    class_weights=np.bincount(classes,weights=weighted,minlength=n+1)
                    class_counts=np.bincount(classes,minlength=n+1)
                for cls in np.flatnonzero(class_counts):
                    by_class[int(cls)]=by_class.get(int(cls),0.)+float(class_weights[cls])
                if line_format == 'records':
                    reported += float(weighted[keep].sum())
                    rows.extend(records((states[keep], energies[keep], populations[keep], amplitudes[keep], values[keep], weighted[keep])))
                evaluated += len(states)
            timings['line_management']+=perf_counter()-managed_at
            if total > rule['total']+1e-7*max(rule['total'], 1.):
                raise RuntimeError('adaptive TI violates the exact positive total-strength sum rule')
            batches += 1
            if batches % checkpoint_interval_batches == 0 and checkpoint.due(): save()
            if progress_callback is not None:
                progress_callback(dict(evaluated_transitions=evaluated, captured_fraction=(1. if rule['total']==0 else reported/rule['total']), cached_integrals=cache.count))
            if rule['total'] == 0 or reached_target() or cache_exhausted:
                break
        save(final=True)
    finally:
        before=perf_counter();checkpoint.close();timings['checkpoint']+=perf_counter()-before
    missing = max(0., rule['total']-reported)
    fraction = 1. if rule['total'] == 0 else reported/rule['total']
    columns = {}
    if line_format == 'columns' and (column_batches or (store is not None and store.rows)):
        before=perf_counter()
        parts=(store.materialize() if store is not None else
            [ArrayBlocks([batch[k] for batch in column_batches]).materialize() for k in range(6)])
        state,energy,population,amplitude,strength,weighted=parts
        columns=dict(initial_quanta=state[:,:n],final_quanta=state[:,n:],wavenumber_cm=energy,
            population=population,FC_overlap=amplitude[:,0],FC_strength_au2=amplitude[:,1],
            HT_strength_au2=amplitude[:,2],interference_au2=amplitude[:,3],strength_au2=strength,weighted_strength_au2=weighted)
        timings['materialization']=perf_counter()-before
    if store is not None:store.close()
    targeted=None;assignment_report=None
    if spectral_target is not None:
        energies=columns['wavenumber_cm'] if line_format=='columns' and columns else np.array([row['wavenumber_cm'] for row in rows])
        strengths=columns['weighted_strength_au2'] if line_format=='columns' and columns else np.array([row['weighted_strength_au2'] for row in rows])
        targeted=spectral_target.report(energies,strengths[None],[1.])
        from .vibronic_assignment_report import targeted_assignment_report,save_targeted_assignment_report
        if line_format=='columns' and columns:
            initial=columns['initial_quanta'];final=columns['final_quanta'];population=columns['population']
            component_columns={name:columns[name+'_strength_au2'] for name in ('FC','HT')}
            component_columns['interference']=columns['interference_au2']
        else:
            initial=np.array([row['initial_quanta'] for row in rows]);final=np.array([row['final_quanta'] for row in rows])
            population=np.array([row['population'] for row in rows])
            component_columns={name:np.array([row[name+'_strength_au2'] for row in rows]) for name in ('FC','HT')}
            component_columns['interference']=np.array([row['interference_au2'] for row in rows])
        def assignment_components(indices):
            return {name:((column[indices]*population[indices])[None] if operator=='FCHT' or operator==name
                else np.zeros((1,len(indices)))) for name,column in component_columns.items()}
        assignment_report=targeted_assignment_report(spectral_target,energies,strengths[None],strengths,initial,final,components=assignment_components)
        assignment_report['observable_names']=[operator]
        if assignment_report_path is not None:save_targeted_assignment_report(assignment_report,assignment_report_path)
    timings['total']=perf_counter()-started
    return dict(schema='matrix.trinity.harmonic_vibronic_ti.v1',assignment_report=assignment_report,kind=kind, operator=operator,
        origin_cm=float(direction*model.origin_cm), temperature_K=float(temperature_K), lines=rows, line_columns=columns, line_format=line_format,
        diagnostics=dict(selection_batch_size=selection_batch_size,checkpoint_async=checkpoint_async,checkpoint_writer=dict(checkpoint.statistics),recurrence_workspace=(None if native_library is None else native_library.workspace_stats(cache)),preparation_cache=None if preparation_cache is None else preparation_cache.snapshot(),timing_seconds=timings,line_storage='memory' if line_storage_path is None else 'read-only NumPy memmaps',
            line_storage_path=None if line_storage_path is None else str(Path(line_storage_path).resolve()),strength_sum_rule=rule, computed_strength_au2=total, reported_strength_au2=reported,
            component_totals_au2=dict(zip(('FC', 'HT', 'interference'), map(float, components), strict=True)),
            captured_fraction=fraction, target_fraction=target_fraction, converged=reached_target(),global_norm_converged=bool(fraction>=target_fraction),
            convergence_scope='global' if spectral_target is None else 'requested_spectral_windows',targeted_selection=targeted,
            selection_energy_bias_per_cm=np.asarray(selection_energy_bias_per_cm).tolist(),
            window_ranking_statistics=tree.statistics() if use_forest else None,
            omitted_strength_bound_au2=missing, initial_population_retained=float(sum(initial_states.values())),
            evaluated_transitions=evaluated, visited_proposal_states=tree.visited, queued_proposal_states=len(tree.heap),
            checkpoint_resumed=checkpoint.resumed, checkpoint_path=checkpoint.path, pending_transitions=len(pending),
            transition_budget_exhausted=bool(tree.visited >= budget and not reached_target()),
            proposal_domain_exhausted=bool(not tree.heap and not len(pending) and not reached_target()),
            cache_budget_exhausted=cache_exhausted, cached_integrals=cache.count,
            recurrence_dependency_edges=cache.edge_count, recurrence_cache='collision-checked packed multiword keys; compiled shared recurrence without retained edges',
            cached_FC_probability_at_zero_temperature=(float(np.dot(cache.values[:cache.count], cache.values[:cache.count])) if temperature_K==0 else None),
            workers=workers if native_library is None else 1, numerical_backend='python' if native_library is None else 'cpp',
            amplitude_backend='numba nogil shared DAG' if native_library is None else 'C++ shared recurrence',
            selection_backend=('compiled conditional operator-block C2 tree; exact positive-strength stopping'
                if selection_proposal=='operator_c2' else
                'compiled conditional Gaussian C2 tree with max-product completion; exact positive-strength stopping'),
            selection_proposal=selection_proposal,
            selection_proposal_version=2 if selection_proposal=='operator_c2' else 1,
            proposal_block_operator_interference_retained=selection_proposal=='operator_c2',
            proposal_is_physical_probability=False, local_branch_bound_used=False,
            no_parent_amplitude_pruning=True, class_strengths_au2=by_class,
            maximum_class=maximum_class, maximum_class_evaluated=max(by_class, default=0),
            maximum_initial_quanta=maximum_initial_quanta, maximum_final_quanta=maximum_final_quanta,
            selection_maximum_per_mode_quanta=selection_maximum_per_mode_quanta,
            scope=('proposal orders candidates only; global completeness uses the exact harmonic strength sum rule; no renormalization'
                if spectral_target is None else 'requested windows use numerical TD-reference recovery; the exact global omitted-strength bound is retained; no renormalization')),
        provenance=model.provenance, scope='adaptive high-class harmonic FC/HT strengths; incomplete budgets remain explicitly unconverged')


__all__ = ['harmonic_vibronic_ti_adaptive']
