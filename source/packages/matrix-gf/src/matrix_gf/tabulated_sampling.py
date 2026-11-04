"""Budgeted acquisition priorities for finite-support vibrational cuts.

Interpolation differences are acquisition indicators, never uncertainty bounds.
No extrapolated energy is used to qualify a state or select a spectral fit.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

from .tabulated_vscf import solve_tabulated_pair_vscf


def plan_tabulated_refinement(cuts, *, max_points, domain_factors=(4/3, 5/3, 2.),
                              shape_points=2, points=127):
    """Select complete cut refinements within a strict new-sample budget.

    A cut has an immutable ``id``, coordinates, potential, kinetic coefficient,
    qualification report, and optional family and observable sensitivity weight.
    Families receive turns in descending local priority, preventing one family
    from consuming the entire pilot. Missing outer nodes and density-weighted
    interior midpoint probes form one atomic proposal per cut. Skipped proposals
    remain explicitly unresolved. Symmetry equivalence is not presumed.
    """
    if isinstance(max_points, bool) or not isinstance(max_points, (int, np.integer)) or max_points < 0:
        raise ValueError('max_points must be a nonnegative integer')
    if isinstance(shape_points, bool) or not isinstance(shape_points, (int, np.integer)) or shape_points < 0:
        raise ValueError('shape_points must be a nonnegative integer')
    factors=np.asarray(domain_factors, float)
    if factors.ndim != 1 or len(factors) < 1 or not np.all(np.isfinite(factors)) or np.any(factors <= 1) or np.any(np.diff(factors) <= 0):
        raise ValueError('increasing finite domain factors above one required')
    groups={}; identities=set(); qualified=[]
    for cut in cuts:
        identity=str(cut['id'])
        if identity in identities:
            raise ValueError('cut identities must be unique')
        identities.add(identity)
        report=cut['qualification']
        if report.get('qualified'):
            qualified.append(identity);continue
        if not np.isrealobj(cut['coordinates']) or not np.isrealobj(cut['potential']):
            raise ValueError('real sampled coordinates and potentials required')
        q=np.asarray(cut['coordinates'], float);v=np.asarray(cut['potential'], float)
        state=solve_tabulated_pair_vscf([q], [v], [cut['kinetic_coefficient']], points=points)
        weight=float(cut.get('weight', 1.))
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError('positive finite sensitivity weights required')
        changes=report['domain_changes'][-2:]
        if not changes:
            raise ValueError('domain comparisons are required for acquisition priorities')
        tolerance=float(report['energy_tolerance'])
        observable_tol=np.asarray(report['observable_tolerances'], float)
        indicators=[max(c['energy']/tolerance, c['moments']/report['moment_tolerance'],
            float(np.max(np.asarray(c['observables'])/observable_tol, initial=0.))) for c in changes]
        severity=weight*max(*indicators, max(state.boundary_probabilities)/report['boundary_tolerance'])
        requested=[]
        if not (report.get('domain_passed') and report.get('boundary_passed') and report.get('independent_domain_passed')):
            if not q[0] < 0 < q[-1]:
                raise ValueError('origin-straddling cuts required for domain extension')
            requested.extend(float(endpoint*factor) for factor in factors for endpoint in (q[0], q[-1]))
        x=state.coordinates[0];rho=abs(state.solution.modal_coefficients[0][:, 0])**2
        difference=abs(CubicSpline(q,v)(x)-PchipInterpolator(q,v)(x))
        bins=np.clip(np.searchsorted(q,x)-1,0,len(q)-2)
        interval_weight=np.bincount(bins,weights=rho*difference,minlength=len(q)-1)
        order=np.argsort(-interval_weight,kind='stable')[:shape_points]
        requested.extend(float((q[i]+q[i+1])/2) for i in order)
        requested=sorted(set(requested)-set(map(float,q)))
        if not requested:
            continue
        record=dict(id=identity,family=str(cut.get('family',identity)),priority=severity,
            coordinates=requested,new_points=len(requested),
            interpolation_indicator=float(rho@difference),
            weight=weight,weight_source='supplied observable sensitivity' if 'weight' in cut else 'unit local weight; no spectral sensitivity supplied')
        groups.setdefault(record['family'],[]).append(record)
    for group in groups.values():
        group.sort(key=lambda row:(-row['priority'],row['id']))
    selected=[];deferred=[];used=0
    while any(groups.values()):
        heads=sorted((group.pop(0) for group in groups.values() if group),key=lambda row:(-row['priority'],row['id']))
        for row in heads:
            if used+row['new_points'] <= max_points:
                selected.append(row);used+=row['new_points']
            else:
                deferred.append(row)
    return dict(schema='matrix.gf.budgeted_tabulated_acquisition.v1',max_points=int(max_points),
        planned_new_points=used,selected=selected,deferred=deferred,already_qualified=qualified,
        domain_factors=factors.tolist(),shape_points=int(shape_points),
        scope='local numerical acquisition indicators; no electronic-model or spectral accuracy certificate')


__all__=['plan_tabulated_refinement']
