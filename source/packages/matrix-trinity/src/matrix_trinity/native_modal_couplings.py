"""Connected finite-displacement potential cuts for active-space selection."""
from __future__ import annotations

import numpy as np
from itertools import combinations,product


def sample_connected_mode_hierarchy(potential_evaluator,*,mode_count,potential_identifier,
        pair_displacements=(-4.,-2.,-1.,1.,2.,4.),triple_amplitudes=(1.,2.,4.),
        batch_size=128,maximum_points=200000,progress=None):
    """All pairs and triples on declared finite probes, without a rate bound.

    Pair probes use the Cartesian product of supplied displacements. Triple
    probes use all signs at each common absolute amplitude. Inclusion-exclusion
    cancels lower-order cuts exactly algebraically. Points shared by different
    supports are evaluated once; no support is pruned by its measured size.
    """
    if (not isinstance(mode_count,int) or mode_count<3 or not callable(potential_evaluator)
            or not isinstance(potential_identifier,str) or not potential_identifier
            or not isinstance(batch_size,int) or batch_size<1
            or not isinstance(maximum_points,int) or maximum_points<1):
        raise ValueError('Identified real potential, at least three modes and positive budgets required')
    shifts=np.asarray(pair_displacements,float);amplitudes=np.asarray(triple_amplitudes,float)
    if (shifts.ndim!=1 or not len(shifts) or not np.all(np.isfinite(shifts))
            or amplitudes.ndim!=1 or not len(amplitudes)
            or not np.all(np.isfinite(amplitudes)) or np.any(amplitudes<=0)):
        raise ValueError('Finite pair displacements and positive triple amplitudes required')
    points=[];lookup={};tables={}
    def index(q):
        q[q==0]=0.;key=q.tobytes()
        if key not in lookup:
            if len(points)>=maximum_points:raise MemoryError('Connected hierarchy point budget exceeded before acquisition')
            lookup[key]=len(points);points.append(q)
        return lookup[key]
    designs={2:np.asarray(list(product(shifts,repeat=2))),
             3:np.asarray([a*np.asarray(sign) for a in amplitudes for sign in product((-1.,1.),repeat=3)])}
    for order in (2,3):
        supports=np.asarray(list(combinations(range(mode_count),order)),int)
        ids=np.empty((len(supports),len(designs[order]),2**order),int)
        signs=np.array([(-1.)**(order-mask.bit_count()) for mask in range(2**order)])
        for i,support in enumerate(supports):
            for j,displacement in enumerate(designs[order]):
                for mask in range(2**order):
                    q=np.zeros(mode_count)
                    for axis in range(order):
                        if mask&(1<<axis):q[support[axis]]=displacement[axis]
                    ids[i,j,mask]=index(q)
        tables[order]=(supports,ids,signs)
    coordinates=np.asarray(points);energies=np.empty(len(points))
    for start in range(0,len(points),batch_size):
        stop=min(start+batch_size,len(points));values=np.asarray(potential_evaluator(coordinates[start:stop]))
        if np.iscomplexobj(values) or values.shape!=(stop-start,) or not np.all(np.isfinite(values)):
            raise ValueError('One finite real energy per hierarchy probe required')
        energies[start:stop]=values
        if progress:progress(dict(acquired_points=stop,total_points=len(points)))
    result={}
    for order,label in ((2,'pair'),(3,'triple')):
        supports,ids,signs=tables[order]
        result[label+'_modes']=supports
        result[label+'_displacements']=designs[order]
        result[label+'_connected_cm']=np.sum(energies[ids]*signs,axis=-1)
    result['diagnostics']=dict(potential_identifier=potential_identifier,mode_count=mode_count,
        acquired_points=len(points),scope='all supports on finite probes; no global potential or rate bound',
        support_pruning_used=False,observable_convergence_certified=False)
    return result


def sample_connected_modal_cuts(potential_evaluator, *, mode_count, core_modes,
                                core_configurations, candidate_modes,
                                candidate_displacements, potential_identifier,
                                batch_size=128, maximum_points=100000,
                                progress=None):
    """Sample V(A,b)-V(A,0)-V(0,b)+V(0,0) without changing the PES.

    The callable takes dimensionless full modal coordinates and returns energies
    in cm-1. Mode indices are zero based. Core configurations may contain mixed
    displacements. Pure cuts are evaluated once and reused. Results describe
    only the supplied finite cuts: they neither bound unsampled couplings nor
    certify convergence of a molecular observable.
    """
    if not isinstance(mode_count, int) or mode_count < 2:
        raise ValueError('mode_count must be an integer >= 2')
    def indices(values):
        raw = np.asarray(values)
        if (raw.ndim != 1 or not len(raw) or raw.dtype.kind not in 'iu'
                or len(np.unique(raw)) != len(raw)
                or np.any(raw < 0) or np.any(raw >= mode_count)):
            raise ValueError('unique integer mode indices in range required')
        return raw.astype(int)
    core = indices(core_modes); candidates = indices(candidate_modes)
    if np.iscomplexobj(core_configurations) or np.iscomplexobj(candidate_displacements):
        raise ValueError('real modal coordinates required')
    anchors = np.asarray(core_configurations, float)
    shifts = np.asarray(candidate_displacements, float)
    if (np.intersect1d(core, candidates).size
            or anchors.ndim != 2 or anchors.shape[1] != len(core)
            or not len(anchors) or not np.all(np.isfinite(anchors))
            or shifts.ndim != 1 or not len(shifts) or not np.all(np.isfinite(shifts))
            or not isinstance(batch_size, int) or batch_size < 1
            or not isinstance(maximum_points, int) or maximum_points < 1
            or not isinstance(potential_identifier, str) or not potential_identifier):
        raise ValueError('finite disjoint modal cuts, identifier and positive budgets required')
    # Deduplicate zero anchors/displacements and any repeated user inputs.
    unique_anchors, anchor_inverse = np.unique(anchors, axis=0, return_inverse=True)
    unique_shifts, shift_inverse = np.unique(shifts, return_inverse=True)
    points = [np.zeros(mode_count)]; lookup = {points[0].tobytes(): 0}
    def index(q):
        q[q == 0] = 0.0  # canonicalize negative zero
        key = q.tobytes()
        if key not in lookup:
            if len(points) >= maximum_points:
                raise ValueError('connected-cut point budget exceeded before PES acquisition')
            lookup[key] = len(points); points.append(q)
        return lookup[key]
    ai = []
    for anchor in unique_anchors:
        q = np.zeros(mode_count); q[core] = anchor; ai.append(index(q))
    bi = np.empty((len(candidates), len(unique_shifts)), int)
    mixed = np.empty((len(candidates), len(unique_anchors), len(unique_shifts)), int)
    for m, candidate in enumerate(candidates):
        for b, shift in enumerate(unique_shifts):
            q = np.zeros(mode_count); q[candidate] = shift; bi[m, b] = index(q)
            for a, anchor in enumerate(unique_anchors):
                q = np.zeros(mode_count); q[core] = anchor; q[candidate] = shift
                mixed[m, a, b] = index(q)
    coordinates = np.asarray(points)
    energies = np.empty(len(points))
    for start in range(0, len(points), batch_size):
        stop = min(start+batch_size, len(points))
        values = np.asarray(potential_evaluator(coordinates[start:stop]))
        if (np.iscomplexobj(values) or values.shape != (stop-start,)
                or not np.all(np.isfinite(values))):
            raise ValueError('PES must return one finite energy per geometry')
        energies[start:stop] = values
        if progress is not None:
            progress(dict(acquired_points=stop, total_points=len(points)))
    pure_core = energies[np.asarray(ai)][anchor_inverse]
    pure_candidate = energies[bi][:, shift_inverse]
    full = energies[mixed][:, anchor_inverse][:, :, shift_inverse]
    connected = full-pure_core[None, :, None]-pure_candidate[:, None, :]+energies[0]
    return dict(core_modes=core, candidate_modes=candidates,
                core_configurations=anchors, candidate_displacements=shifts,
                connected_cm=connected, mixed_energies_cm=full,
                core_energies_cm=pure_core, candidate_energies_cm=pure_candidate,
                origin_energy_cm=float(energies[0]),
                diagnostics=dict(potential_identifier=potential_identifier,
                    acquired_points=len(points), mode_count=mode_count,
                    scope='finite connected cuts; no observable convergence certificate'))
