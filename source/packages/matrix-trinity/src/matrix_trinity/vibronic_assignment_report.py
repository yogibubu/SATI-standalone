"""Compact state-to-state assignment tables with complete window/tail totals."""
import json
import os
from pathlib import Path
import tempfile
import numpy as np


def _top(values,limit):
    count=min(limit,int(np.count_nonzero(values>0)))
    if not count:return np.empty(0,np.int64)
    cut=np.partition(values,len(values)-count)[len(values)-count]
    ids=np.flatnonzero(values>cut)
    ids=np.r_[ids,np.flatnonzero(values==cut)[:count-len(ids)]]
    return ids[np.lexsort((ids,-values[ids]))]


def targeted_assignment_report(target,energies,strengths,norms,initial,final,*,components=None):
    """Summaries include ALL selected lines; displayed lists have a named cap.

    components(ids) returns physical, thermally weighted FC/HT/interference
    strengths for the displayed transition IDs, never ranking-proposal weights.
    """
    windows=[];energies=np.asarray(energies);strengths=np.asarray(strengths)
    for window,(lo,hi) in enumerate(target.windows):
        weight=target._cdf(hi-energies)-target._cdf(lo-energies)
        inside=(energies>=lo)&(energies<=hi)
        bank_rank=_top(weight*norms,target.assignment_limit)
        ranks=[_top(abs(row)*weight,target.assignment_limit) for row in strengths]
        ids=np.unique(np.concatenate([bank_rank,*ranks]))
        parts={} if components is None or not len(ids) else components(ids)
        records=[]
        def label(values):
            return ', '.join(f'v{i+1}={int(value)}' for i,value in enumerate(values) if value) or 'ground'
        for position,index in enumerate(ids):
            record=dict(transition_index=int(index),initial_quanta=np.asarray(initial[index]).tolist(),
                final_quanta=np.asarray(final[index]).tolist(),
                initial_state_label=label(initial[index]),final_state_label=label(final[index]),wavenumber_cm=float(energies[index]),
                location='inside' if inside[index] else 'lower_tail' if energies[index]<lo else 'upper_tail',
                line_strengths=strengths[:,index].tolist(),window_line_fraction=float(weight[index]),
                window_strengths=(weight[index]*strengths[:,index]).tolist(),
                window_positive_bank_strength=float(weight[index]*norms[index]))
            if parts:record['components']={name:np.asarray(value)[:,position].tolist() for name,value in parts.items()}
            records.append(record)
        windows.append(dict(window_cm=[float(lo),float(hi)],evaluated_transition_count=len(energies),
            center_inside_count=int(inside.sum()),center_outside_count=int((~inside).sum()),
            strength_from_centers_inside=strengths[:,inside]@weight[inside],
            strength_from_external_tails=strengths[:,~inside]@weight[~inside],
            positive_bank_from_centers_inside=float(norms[inside]@weight[inside]),
            positive_bank_from_external_tails=float(norms[~inside]@weight[~inside]),
            reference_bank_area=float(target.reference_bank[window]),
            computed_bank_area=float(target.computed_bank[window]),
            profile_criteria_met=target.profile_status()[0][window],
            top_positive_bank_transition_indices=bank_rank,
            top_observable_transition_indices=ranks,listed_transition_count=len(ids),assignments=records))
    return dict(schema='sati.targeted_state_assignments.v1',windows=windows,
        observable_names=[f'observable_{i}' for i in range(len(strengths))],
        target_fraction=getattr(target,'requested_fraction',None),
        TD_comparison=target.report(energies,strengths,target.observable_norms),
        displayed_limit_per_observable=target.assignment_limit,mode_label_indices_are_one_based=True,
        component_scope='listed transitions; physical weighted strengths',
        full_assignments='all state labels, frequencies and strengths remain in the returned transition columns',
        window_summaries_include_all_selected_lines=True,external_line_tails_retained=True,
        physical_strengths_reweighted=False)


def save_targeted_assignment_report(report,path):
    """Atomically export a portable report, including signed contributions."""
    from .vibronic_workflow import vibronic_json_value
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=None
    try:
        with tempfile.NamedTemporaryFile('w',dir=path.parent,prefix=path.name+'.',delete=False) as stream:
            temporary=Path(stream.name);json.dump(vibronic_json_value(report),stream,indent=2,allow_nan=False)
            stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if temporary is not None and temporary.exists():temporary.unlink()


__all__=['targeted_assignment_report','save_targeted_assignment_report']
