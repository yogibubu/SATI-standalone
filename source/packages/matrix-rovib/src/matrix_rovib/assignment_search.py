"""Bounded global candidate matching with explicit assignment ambiguity."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import numpy as np
from scipy.optimize import linear_sum_assignment


def assignment_components(row):
    return row.get('components',[row])


def group_assignment_blends(catalogue, *, resolution_MHz, weight_mode='native'):
    """Group predicted components by a declared instrumental resolution.

    Every cluster has total frequency span <= resolution; no unbounded
    single-link chaining. Groups are hypotheses, not measured resolutions.
    """
    if not np.isfinite(resolution_MHz) or resolution_MHz<=0 or weight_mode not in ('fixed','native'):
        raise ValueError('positive declared blend resolution and fixed/native weight mode required')
    ordered=sorted(deepcopy(list(catalogue)),key=lambda r:(r['frequency_MHz'],str(r['id'])))
    groups=[]
    for row in ordered:
        if 'components' in row:raise ValueError('blend grouping requires elementary catalogue transitions')
        if not groups or row['frequency_MHz']-groups[-1][0]['frequency_MHz']>resolution_MHz:
            groups.append([row])
        else:groups[-1].append(row)
    output=[]
    for group in groups:
        if len(group)==1:output.append(group[0]);continue
        intensities=np.array([r['relative_intensity'] for r in group],float)
        if not np.all(np.isfinite(intensities)) or np.any(intensities<=0):
            raise ValueError('positive component intensities required for predicted blend centroids')
        weights=intensities/intensities.sum()
        output.append(dict(id='blend:'+ '|'.join(sorted(str(r['id']) for r in group)),
            family='blend:'+ '|'.join(sorted(set(r['family'] for r in group))),
            frequency_MHz=float(weights@np.array([r['frequency_MHz'] for r in group])),
            uncertainty_MHz=float(weights@np.array([r['uncertainty_MHz'] for r in group])),
            relative_intensity=float(intensities.sum()),weight_mode=weight_mode,
            components=[dict(lower=r['lower'],upper=r['upper'],
                             weight=1. if weight_mode=='native' else float(w))
                        for r,w in zip(group,weights,strict=True)],
            constituent_ids=[r['id'] for r in group],resolution_MHz=float(resolution_MHz)))
    return output


def propose_spectral_assignments(peaks, catalogue, *, frequency_window_sigma=3.,
        ambiguity_cost=4., minimum_family_lines=2, cycle_tolerance_sigma=4.,
        log_intensity_scale=None, maximum_candidates=100000,
        maximum_component_rows=32, maximum_component_candidates=64,
        maximum_assignment_solves=4096):
    """Match resolved peak candidates; never silently force a blended line.

    Gaussian matching costs combine declared experimental and QM uncertainty.
    All competing peaks and transitions in a connected candidate component
    share a one-to-one assignment with private unassigned slots. Alternatives
    are retained using exact constrained re-solves, not a nearest-line gap.
    This discrete hypothesis score is not a posterior probability.
    """
    from .lam_assignment import solve_transition_term_values

    peaks = deepcopy(list(peaks)); catalogue = deepcopy(list(catalogue))
    for name, rows in (('peaks', peaks), ('catalogue', catalogue)):
        if len({str(r['id']) for r in rows}) != len(rows):
            raise ValueError(f'{name} IDs must be unique')
        for r in rows:
            if (not np.isfinite(r['frequency_MHz']) or r['frequency_MHz'] <= 0
                    or not np.isfinite(r['uncertainty_MHz']) or r['uncertainty_MHz'] <= 0):
                raise ValueError('positive finite frequencies and uncertainties required')
    if (not np.isfinite(frequency_window_sigma) or frequency_window_sigma <= 0
            or not np.isfinite(ambiguity_cost) or ambiguity_cost <= 0
            or not np.isfinite(cycle_tolerance_sigma) or cycle_tolerance_sigma <= 0
            or isinstance(minimum_family_lines, bool) or not isinstance(minimum_family_lines, int)
            or minimum_family_lines < 1):
        raise ValueError('positive matching thresholds and family size required')
    if log_intensity_scale is not None and not np.isfinite(log_intensity_scale):
        raise ValueError('log intensity scale must be finite and declared in advance')
    for r in catalogue:
        if not r.get('family') or not assignment_components(r) or any(
                not c.get(s,{}).get('state_id') for c in assignment_components(r) for s in ('lower','upper')):
            raise ValueError('catalogue needs physical families and persistent lower/upper state IDs')
    for limit in (maximum_candidates, maximum_component_rows, maximum_component_candidates,
                  maximum_assignment_solves):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError('positive integer search budgets required')
    order = np.argsort([r['frequency_MHz'] for r in catalogue], kind='stable')
    frequencies = np.array([catalogue[i]['frequency_MHz'] for i in order])
    largest = max((r['uncertainty_MHz'] for r in catalogue), default=0.)
    edges = {}; by_peak = defaultdict(list); by_line = defaultdict(list)
    skipped = {}
    for i, peak in enumerate(peaks):
        if peak.get('role', 'training') != 'training':
            skipped[i] = 'withheld_from_assignment'; continue
        if peak.get('blended', False) and not any('components' in r for r in catalogue):
            skipped[i] = 'blend_requires_component_model'; continue
        radius = frequency_window_sigma * np.hypot(peak['uncertainty_MHz'], largest)
        lo = np.searchsorted(frequencies, peak['frequency_MHz'] - radius, side='left')
        hi = np.searchsorted(frequencies, peak['frequency_MHz'] + radius, side='right')
        for j in order[lo:hi]:
            row = catalogue[j]
            if peak.get('blended',False) and 'components' not in row:continue
            if peak.get('family') is not None and peak['family'] != row['family']:
                continue
            sigma = np.hypot(peak['uncertainty_MHz'], row['uncertainty_MHz'])
            delta = (peak['frequency_MHz'] - row['frequency_MHz']) / sigma
            if abs(delta) > frequency_window_sigma:
                continue
            cost = delta * delta
            if log_intensity_scale is not None and 'relative_intensity' in peak:
                measured = float(peak['relative_intensity']); predicted = float(row['relative_intensity'])
                uncertainty = float(peak['log_intensity_sigma'])
                if not all(np.isfinite(v) and v > 0 for v in (measured, predicted, uncertainty)):
                    raise ValueError('positive relative intensities and log intensity sigma required')
                cost += ((np.log(measured / predicted) - log_intensity_scale) / uncertainty) ** 2
            edges[i, int(j)] = float(cost)
            by_peak[i].append(int(j)); by_line[int(j)].append(i)
            if len(edges) > maximum_candidates:
                raise MemoryError('candidate edge budget exceeded; narrow the declared search range')
    records = {}; unseen = set(by_peak); components = 0; solves = 0
    while unseen:
        first = min(unseen); pis = {first}; js = set(); frontier = [first]
        while frontier:
            i = frontier.pop()
            for j in by_peak[i]:
                if j in js: continue
                js.add(j)
                for other in by_line[j]:
                    if other not in pis: pis.add(other); frontier.append(other)
        unseen -= pis; pi = sorted(pis); cj = sorted(js); components += 1
        required_solves = 1 + sum(len(by_peak[i])+1 for i in pi)
        if (len(pi) > maximum_component_rows or len(cj) > maximum_component_candidates
                or solves+required_solves > maximum_assignment_solves):
            for i in pi:
                records[i] = dict(status='search_budget_exceeded', candidates=[
                    dict(catalogue_id=catalogue[j]['id'], cost=edges[i,j]) for j in by_peak[i]])
            continue
        solves += required_solves
        cost = np.full((len(pi), len(cj) + len(pi)), np.inf)
        columns = {j:k for k,j in enumerate(cj)}
        for a,i in enumerate(pi):
            cost[a,len(cj)+a] = frequency_window_sigma**2 + ambiguity_cost
            for j in by_peak[i]: cost[a,columns[j]] = edges[i,j]
        rr, cc = linear_sum_assignment(cost); best = float(cost[rr,cc].sum())
        chosen = dict(zip(rr,cc,strict=True))
        for a,i in enumerate(pi):
            alternatives = []
            for b in [columns[j] for j in by_peak[i]] + [len(cj)+a]:
                constrained = cost.copy(); constrained[a,:] = np.inf; constrained[a,b] = cost[a,b]
                r,c = linear_sum_assignment(constrained)
                increase = max(0., float(constrained[r,c].sum()) - best)
                alternatives.append(dict(catalogue_id=None if b >= len(cj) else catalogue[cj[b]]['id'],
                    cost=float(cost[a,b]), global_cost_increase=increase))
            alternatives.sort(key=lambda x:(x['global_cost_increase'], str(x['catalogue_id'])))
            near = [x for x in alternatives if x['global_cost_increase'] < ambiguity_cost]
            j = None if chosen[a] >= len(cj) else cj[chosen[a]]
            status = 'unassigned' if j is None else 'ambiguous' if len(near)>1 else 'provisional'
            records[i] = dict(status=status, catalogue_id=None if j is None else catalogue[j]['id'],
                candidates=alternatives, component=components,
                transition=None if j is None else deepcopy(catalogue[j]))
    family_counts = defaultdict(int)
    for r in records.values():
        if r['status'] == 'provisional': family_counts[r['transition']['family']] += 1
    for r in records.values():
        if r['status'] == 'provisional':
            r['status'] = ('accepted' if family_counts[r['transition']['family']] >= minimum_family_lines
                           else 'isolated_candidate')
    # Cycle closure uses measurement sigmas alone. It tests consistency of
    # the proposed IDs without fitting a Hamiltonian or widening QM priors.
    network_rows = []; selected = []
    for i,r in records.items():
        if r['status'] != 'accepted': continue
        t = r['transition']; selected.append(i)
        if 'components' in t:
            selected.pop();continue
        network_rows.append(dict(lower_id=t['lower']['state_id'], upper_id=t['upper']['state_id'],
            frequency_MHz=peaks[i]['frequency_MHz'], uncertainty_MHz=peaks[i]['uncertainty_MHz']))
    network = None
    if network_rows:
        solved = solve_transition_term_values(network_rows)
        bad = {solved.component_by_state[network_rows[k]['lower_id']] for k,residual in
            enumerate(solved.transition_residuals_MHz)
            if abs(residual) > cycle_tolerance_sigma * network_rows[k]['uncertainty_MHz']}
        for i,row in zip(selected, network_rows, strict=True):
            if solved.component_by_state[row['lower_id']] in bad:
                records[i]['status'] = 'inconsistent_combination_differences'
        vertex_count = len(solved.energies_MHz); component_count = len(set(solved.component_by_state.values()))
        network = dict(weighted_rms=solved.weighted_rms,
            independent_cycles=len(network_rows)-vertex_count+component_count,
            inconsistent_components=sorted(bad),
            interpretation='zero cycle count gives no independent combination-difference evidence')
    remaining = defaultdict(int)
    for r in records.values():
        if r['status'] == 'accepted': remaining[r['transition']['family']] += 1
    for r in records.values():
        if r['status'] == 'accepted' and remaining[r['transition']['family']] < minimum_family_lines:
            r['status'] = 'isolated_candidate'
    output = [dict(peak=peaks[i], **records.get(i, dict(status=skipped.get(i,'unassigned'), candidates=[])))
              for i in range(len(peaks))]
    return dict(schema='matrix.lock.assignment_candidates.v1', rows=output,
        accepted_count=sum(r['status']=='accepted' for r in output), network=network,
        search=dict(candidate_edges=len(edges), components=components,
            frequency_window_sigma=frequency_window_sigma, ambiguity_cost=ambiguity_cost,
            minimum_family_lines=minimum_family_lines, cycle_tolerance_sigma=cycle_tolerance_sigma, maximum_component_rows=maximum_component_rows,
            maximum_component_candidates=maximum_component_candidates,
            assignment_solves=solves, maximum_assignment_solves=maximum_assignment_solves,
            intensity_used=log_intensity_scale is not None),
        scope='provisional resolved-line hypotheses; ambiguity retained; no automatic blend resolution')


def assignments_to_observations(report):
    """Export only accepted hypotheses, preserving experimental uncertainties."""
    rows = []
    for record in report['rows']:
        if record['status'] != 'accepted': continue
        peak = record['peak']; line = record['transition']
        row=dict(label=str(peak['id']),frequency_MHz=peak['frequency_MHz'],
                 uncertainty_MHz=peak['uncertainty_MHz'],role='training',weight=peak.get('weight',1.))
        if 'components' in line:
            row.update(components=deepcopy(line['components']),weight_mode=line.get('weight_mode','fixed'))
        else:row.update(lower=deepcopy(line['lower']),upper=deepcopy(line['upper']))
        rows.append(row)
    return rows


__all__ = ['propose_spectral_assignments','assignments_to_observations','group_assignment_blends']
