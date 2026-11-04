"""Derivative coverage and cost for Hessian/gradient anharmonic acquisitions.

The primary/auxiliary/spectator hierarchy follows Barone, Lazzari and
Mendolicchio, JCTC 2026, 22, 2453-2466, doi:10.1021/acs.jctc.5c02123.
The observable reduction below is a stricter harmonic-spectator trial,
not a reproduction of all retained perturbative spectator contributions
in that work. All callbacks refer to one common chart and consistent units.
Unknown derivatives are NaN, never an implicit zero. A reference field can
complete them explicitly, representing a declared constant-coupling model.
"""
from collections import Counter,defaultdict
from itertools import permutations,combinations_with_replacement
import numpy as np


def _weights(points,order):
    points=np.asarray(points,float);scale=max(abs(points));x=points/scale
    rhs=np.zeros(len(x));rhs[order]=1 if order==1 else 2
    return np.linalg.solve(np.array([x**i for i in range(len(x))]),rhs)/scale**order


def acquire_tiered_anharmonic(energy,gradient,hessian,origin,*,reaction_mode,
                             primary_modes=(),intermediate_modes=(),steps=.06,
                             reflection_actions=None,secondary_modes=None,auxiliary_modes=None):
    """3 reactive Hessians, 2 Hessians/primary and 2 gradients/auxiliary.

    Hessian scans supply C_ija and Q_ijaa. Gradient scans supply C_iaa/Q_iaaa;
    energies and the reference gradient/Hessian supply Q_aaaa. Steps can be
    scalar or (n,2) positive (minus,plus) distances. Explicit involutive
    orthogonal reflection actions may reconstruct minus packets only for
    equal steps; equal step sizes alone never imply a symmetry saving.
    """
    if auxiliary_modes is not None:
        if secondary_modes is not None or tuple(intermediate_modes):
            raise ValueError('use auxiliary_modes or a legacy alias, not both')
        intermediate_modes=auxiliary_modes
    elif secondary_modes is not None:
        if tuple(intermediate_modes):raise ValueError('use secondary_modes or the legacy intermediate_modes alias, not both')
        intermediate_modes=secondary_modes
    x=np.asarray(origin,float)
    if x.ndim!=1 or not len(x) or not np.all(np.isfinite(x)):raise ValueError('finite origin vector required')
    n=len(x);r=reaction_mode;p=tuple(primary_modes);m=tuple(intermediate_modes)
    indices=(r,)+p+m
    if (any(isinstance(i,(bool,np.bool_)) or not isinstance(i,(int,np.integer)) or not 0<=i<n for i in indices)
            or len(set(indices))!=len(indices)):
        raise ValueError('disjoint valid reaction, primary and auxiliary indices required')
    step=np.asarray(steps,float)
    if step.ndim==0:step=np.full((n,2),float(step))
    if step.shape!=(n,2) or np.any(step<=0) or not np.all(np.isfinite(step)):raise ValueError('positive finite scalar or (n,2) steps required')
    e0=float(energy(x));g0=np.asarray(gradient(x),float);h0=np.asarray(hessian(x),float)
    if g0.shape!=(n,) or h0.shape!=(n,n) or not np.isfinite(e0) or not np.all(np.isfinite(g0)) or not np.all(np.isfinite(h0)) or not np.allclose(h0,h0.T):
        raise ValueError('matching finite reference derivatives required')
    estimates={3:defaultdict(list),4:defaultdict(list)};counts=dict(hessian=1,gradient=1,energy=1);saved=[]
    def add(order,key,value):estimates[order][tuple(sorted(key))].append(float(value))
    actions={} if reflection_actions is None else dict(reflection_actions)
    for a in (r,)+p+m:
        hm,hp=step[a];direction=np.eye(n)[a];plus=x+hp*direction
        is_hessian=a in (r,)+p
        ep=float(energy(plus));gp=np.asarray(gradient(plus),float);counts['energy']+=1;counts['gradient']+=1
        Hplus=np.asarray(hessian(plus),float) if is_hessian else None
        if is_hessian:counts['hessian']+=1
        S=actions.get(a)
        if S is not None:
            S=np.asarray(S,float)
            if (hm!=hp or S.shape!=(n,n) or not np.all(np.isfinite(S))
                    or not np.allclose(S.T@S,np.eye(n),atol=1e-10)
                    or not np.allclose(S@S,np.eye(n),atol=1e-10)
                    or not np.allclose(S@direction,-direction,atol=1e-10)
                    or not np.allclose(S.T@g0,g0,atol=1e-8)
                    or not np.allclose(S.T@h0@S,h0,atol=1e-7)):
                raise ValueError('valid coordinate reflection, reference invariance and equal steps required')
            em=ep;gm=S.T@gp;Hminus=None if Hplus is None else S.T@Hplus@S;saved.append(a)
        else:
            minus=x-hm*direction;em=float(energy(minus));gm=np.asarray(gradient(minus),float)
            counts['energy']+=1;counts['gradient']+=1
            Hminus=np.asarray(hessian(minus),float) if is_hessian else None
            if is_hessian:counts['hessian']+=1
        if gp.shape!=(n,) or gm.shape!=(n,) or not np.all(np.isfinite([ep,em])) or not np.all(np.isfinite([gp,gm])):
            raise ValueError('finite matching displaced gradients/energies required')
        w1=_weights([-hm,0,hp],1);w2=_weights([-hm,0,hp],2)
        if is_hessian:
            if Hplus.shape!=(n,n) or Hminus.shape!=(n,n) or not np.all(np.isfinite([Hplus,Hminus])):
                raise ValueError('finite matching displaced Hessians required')
            third=np.einsum('k,kij->ij',w1,[Hminus,h0,Hplus])
            fourth=np.einsum('k,kij->ij',w2,[Hminus,h0,Hplus])
            for i in range(n):
                for j in range(i,n):add(3,(i,j,a),third[i,j]);add(4,(i,j,a,a),fourth[i,j])
        else:
            t=np.array([-hm,hp])
            # Use known dg_i/dq_a=H_ia so unequal steps do not contaminate
            # the cubic estimate with the quartic odd-gradient term.
            residual=np.stack([gm-g0+hm*h0[:,a],gp-g0-hp*h0[:,a]])
            gradient_fit=np.linalg.solve(np.column_stack((t*t/2,t**3/6)),residual)
            for i in range(n):
                add(3,(i,a,a),gradient_fit[0,i])
                if i!=a:add(4,(i,a,a,a),gradient_fit[1,i])
            # Two energies plus F0/F1/F2 determine diagonal F3/F4, also with
            # unequal steps. Pair Q_aabb still requires mixed energies or Hessians.
            t=np.array([-hm,hp]);res=np.array([em,ep])-e0-g0[a]*t-.5*h0[a,a]*t*t
            fit=np.linalg.solve(np.column_stack((t**3/6,t**4/24)),res)
            add(4,(a,)*4,fit[1])
    tensors={};masks={}
    for order in [3,4]:
        tensor=np.full((n,)*order,np.nan);mask=np.zeros((n,)*order,bool)
        for key,values in estimates[order].items():
            for index in set(permutations(key)):tensor[index]=np.mean(values);mask[index]=True
        tensors[order]=tensor;masks[order]=mask
    return dict(cubic=tensors[3],quartic=tensors[4],cubic_observed=masks[3],quartic_observed=masks[4],
        reference_energy=e0,reference_gradient=g0,reference_hessian=h0,
        callback_counts=counts,symmetry_reconstructed_minus_modes=saved,
        electronic_jobs=dict(hessian=counts['hessian'],gradient=2*len(m)-sum(a in m for a in saved),
                             separate_energy=0),
        coverage='Hessian: C_ija,Q_ijaa; gradient: C_iaa,Q_iaaa; energy axes: Q_aaaa; missing values are NaN')


def complete_tiered_from_reference(acquired,reference_cubic,reference_quartic):
    """Explicitly hold all unacquired derivatives at a same-chart reference."""
    result=[]
    for name,reference in [('cubic',reference_cubic),('quartic',reference_quartic)]:
        base=np.asarray(reference,float);value=acquired[name]
        if base.shape!=value.shape or not np.all(np.isfinite(base)):raise ValueError('finite same-shape reference tensors required')
        result.append(np.where(acquired[name+'_observed'],value,base))
    return tuple(result)


def classify_transverse_variation(reference_cubic,reference_quartic,samples,*,reaction_mode,
                                  cubic_absolute_tolerance,quartic_absolute_tolerance,
                                  relative_tolerance=.2):
    """Legacy variation screen, independent of absolute magnitude.

    Here the legacy secondary_modes output means reference-held modes,
    NOT harmonic spectator modes. Use classify_anharmonic_modes for the
    primary/auxiliary/spectator accuracy-based definition.

    Samples must be expressed in the same transported basis. Three-distinct
    cubics require at least one primary Hessian axis. Quartic iijj terms
    require i or j as a primary axis. Remaining varying C_iaa/Q_aaaa require
    gradient axis a unless a primary already supplies them. Other quartic
    families are not certified by this screening and remain explicit controls.
    """
    C=np.asarray(reference_cubic,float);Q=np.asarray(reference_quartic,float);n=C.shape[0]
    if C.shape!=(n,)*3 or Q.shape!=(n,)*4 or not np.all(np.isfinite(C)) or not np.all(np.isfinite(Q)):
        raise ValueError('finite complete reference tensors required')
    if not 0<=reaction_mode<n or min(cubic_absolute_tolerance,quartic_absolute_tolerance)<=0 or relative_tolerance<0:
        raise ValueError('valid reaction index and positive variation tolerances required')
    observations=list(samples)
    if not observations:raise ValueError('at least one path control required')
    arrays=[]
    for c,q in observations:
        c=np.asarray(c,float);q=np.asarray(q,float)
        if c.shape!=C.shape or q.shape!=Q.shape or not np.all(np.isfinite(c)) or not np.all(np.isfinite(q)):
            raise ValueError('finite same-basis path tensors required')
        arrays.append((c,q))
    varying={3:[],4:[]};ratios={3:{},4:{}}
    stable=[a for a in range(n) if a!=reaction_mode]
    for order,base,tol in [(3,C,cubic_absolute_tolerance),(4,Q,quartic_absolute_tolerance)]:
        for key in combinations_with_replacement(stable,order):
            if order==4 and sorted(Counter(key).values()) not in ([4],[2,2]):continue
            ratio=max(abs(pair[order-3][key]-base[key])/(tol+relative_tolerance*max(abs(base[key]),abs(pair[order-3][key]))) for pair in arrays)
            ratios[order][key]=float(ratio)
            if ratio>1:varying[order].append(key)
    hard=[set(k) for k in varying[3] if len(set(k))==3]
    hard += [set(k) for k in varying[4] if len(set(k))==2]
    primary=[]
    while hard:
        chosen=max(stable,key=lambda a:(sum(a in term for term in hard),-a))
        primary.append(chosen);hard=[term for term in hard if chosen not in term]
    intermediate=set()
    for key in varying[3]:
        if set(key)&set(primary):continue
        counts=Counter(key);intermediate.add(max(counts,key=counts.get))
    for key in varying[4]:
        if len(set(key))==1 and key[0] not in primary:intermediate.add(key[0])
    secondary=set(stable)-set(primary)-intermediate
    return dict(primary_modes=sorted(primary),intermediate_modes=sorted(intermediate),secondary_modes=sorted(secondary),
        varying_cubic_keys=[list(k) for k in varying[3]],varying_quartic_keys=[list(k) for k in varying[4]],
        maximum_variation_ratio={str(order):max(ratios[order].values(),default=0.) for order in [3,4]},
        cubic_absolute_tolerance=cubic_absolute_tolerance,quartic_absolute_tolerance=quartic_absolute_tolerance,
        relative_tolerance=relative_tolerance,selection='greedy derivative-coverage of varying terms; no global cost optimality claim',
        other_quartic_families_certified=False)


def acquire_energy_semidiagonal_quartics(energy, origin, pairs, *, steps=.12):
    """Acquire selected Q_aabb from a symmetric nine-point energy stencil.

    Four corner energies per pair, plus cached origin/axis energies. These
    are potential derivatives in one common chart; no kinetic derivatives
    or three-distinct cubic terms are inferred. Compare step sizes because
    fourth differences amplify electronic-energy and chart-solver noise.
    """
    x=np.asarray(origin,float)
    if x.ndim!=1 or not len(x) or not np.all(np.isfinite(x)):
        raise ValueError('finite origin vector required')
    n=len(x);h=np.asarray(steps,float)
    if h.ndim==0:h=np.full(n,float(h))
    if h.shape!=(n,) or np.any(h<=0) or not np.all(np.isfinite(h)):
        raise ValueError('positive finite scalar or n-vector steps required')
    selected=[]
    for pair in pairs:
        if len(pair)!=2 or any(isinstance(i,(bool,np.bool_)) or not isinstance(i,(int,np.integer)) or not 0<=i<n for i in pair) or pair[0]==pair[1]:
            raise ValueError('distinct valid index pairs required')
        key=tuple(sorted(pair))
        if key not in selected:selected.append(key)
    cache={}
    def value(offset):
        key=tuple(offset)
        if key not in cache:
            cache[key]=float(energy(x+offset))
            if not np.isfinite(cache[key]):raise ValueError('finite displaced energies required')
        return cache[key]
    zero=np.zeros(n);e0=value(zero);Q=np.full((n,)*4,np.nan);mask=np.zeros_like(Q,dtype=bool)
    for a,b in selected:
        da=np.eye(n)[a]*h[a];db=np.eye(n)[b]*h[b]
        # Subtract E0 before summation to avoid cancellation of a large
        # constant electronic-energy origin where possible.
        corners=sum(value(sa*da+sb*db)-e0 for sa in [-1,1] for sb in [-1,1])
        axes=sum(value(v)-e0 for v in [da,-da,db,-db])
        derivative=(corners-2*axes)/(h[a]**2*h[b]**2)
        for index in set(permutations((a,a,b,b))):Q[index]=derivative;mask[index]=True
    return dict(quartic=Q,quartic_observed=mask,energy_evaluations=len(cache),
                pairs=[list(p) for p in selected],steps=h,
                coverage='selected Q_aabb only; unacquired entries are NaN')


def transverse_tier_model(cubic,quartic,*,reaction_mode,primary_modes,auxiliary_modes):
    """Project purely transverse anharmonicity onto the specified tiers.

    Spectators are harmonic: all purely transverse cubic/quartic terms
    containing them are zero. Auxiliary axes provide all one/two-distinct
    families, including energy-acquired Q_aabb. Three-distinct cubics need
    a primary index; three-distinct quartics need a repeated primary index.
    Four-distinct quartics are outside Hessian-scan coverage. Reaction-
    containing terms are held fixed except cubic transverse dependence
    involving spectators, which is removed. These terms require separate qualification.
    This is an explicit model projection, not a measurement mask.
    """
    C=np.asarray(cubic,float);Q=np.asarray(quartic,float)
    if C.ndim!=3 or C.shape!=(C.shape[0],)*3 or Q.shape!=(C.shape[0],)*4 or not np.all(np.isfinite(C)) or not np.all(np.isfinite(Q)):
        raise ValueError('finite matching cubic/quartic tensors required')
    n=len(C);p=set(primary_modes);s=set(auxiliary_modes)
    indices=[reaction_mode,*p,*s]
    if any(isinstance(i,(bool,np.bool_)) or not isinstance(i,(int,np.integer)) or not 0<=i<n for i in indices) or p&s or reaction_mode in p|s:
        raise ValueError('valid disjoint reaction, primary and auxiliary modes required')
    active=p|s;result=[]
    for order,source in [(3,C),(4,Q)]:
        tensor=source.copy()
        for key in combinations_with_replacement([i for i in range(n) if i!=reaction_mode],order):
            counts=Counter(key);distinct=set(key)
            keep=distinct<=active and (len(distinct)<=2 or
                (order==3 and bool(distinct&p)) or
                (order==4 and any(a in p and count>=2 for a,count in counts.items())))
            if not keep:
                for index in set(permutations(key)):tensor[index]=0.
        # Harmonic along the IRC permits r*t*t and r*r*t*t but not
        # r*t*t*t for a spectator mode.
        spectator=set(range(n))-{reaction_mode}-active
        if order==4:
            for key in combinations_with_replacement(range(n),4):
                if key.count(reaction_mode)==1 and set(key)&spectator:
                    for index in set(permutations(key)):tensor[index]=0.
        result.append(tensor)
    return tuple(result)


def classify_anharmonic_modes(samples,evaluate_model,*,reaction_mode,
                              absolute_tolerance,relative_tolerance=0.,projection_model='strict'):
    """Greedy P -> A -> S reductions qualified against observable errors.

    samples: same-chart (C,Q) references at TS/path/temperature controls.
    evaluate_model(index,C,Q,tiers): consistent observable vector; tiers is
    None for the full reference and a P/A/S dictionary for trial models.
    projection_model='pas_coverage' retains all accessible P/S and A/S
    potential couplings; its coordinate owner supplies consistent kinetic
    derivatives. For 'strict', the evaluator must also remove nonquadratic
    spectator kinetic terms; potential projection alone is insufficient.
    use log rates/partitions for multiplicative accuracy, retaining kinetic
    terms and resonances in the evaluator. All reductions are checked
    globally against the ORIGINAL reference, not against the previous step.
    The deterministic greedy sequence minimizes derivative tier, not global
    electronic cost. Independent holdout validation is still required.
    """
    if projection_model not in {'strict','pas_coverage'}:raise ValueError('strict or pas_coverage projection required')
    def project(c,q,primary,auxiliary):
        if projection_model=='strict':
            return transverse_tier_model(c,q,reaction_mode=reaction_mode,primary_modes=primary,auxiliary_modes=auxiliary)
        from .pas_reduction import pas_coverage_projection
        model=pas_coverage_projection(c,q,reaction_mode=reaction_mode,primary_modes=primary,auxiliary_modes=auxiliary)
        return model['cubic'],model['quartic']
    fields=[(np.asarray(c,float),np.asarray(q,float)) for c,q in samples]
    if not fields:raise ValueError('at least one reference sample required')
    n=fields[0][0].shape[0] if fields[0][0].ndim==3 else 0
    stable=[a for a in range(n) if a!=reaction_mode]
    # Validate all tensors and the reaction index before evaluating anything.
    for c,q in fields:
        transverse_tier_model(c,q,reaction_mode=reaction_mode,primary_modes=stable,auxiliary_modes=[])
        if len(c)!=n:raise ValueError('all samples must share one dimension and transported gauge')
    if np.ndim(relative_tolerance)!=0 or not np.isfinite(relative_tolerance) or relative_tolerance<0:
        raise ValueError('finite nonnegative relative tolerance required')
    baseline=[np.atleast_1d(np.asarray(evaluate_model(i,c,q,None),float)) for i,(c,q) in enumerate(fields)]
    shape=baseline[0].shape
    if any(b.shape!=shape or not b.size or not np.all(np.isfinite(b)) for b in baseline):raise ValueError('matching finite observable vectors required')
    try:tol=np.broadcast_to(np.asarray(absolute_tolerance,float),shape)
    except ValueError as exc:raise ValueError('matching positive observable tolerances required') from exc
    if not np.all(np.isfinite(tol)) or np.any(tol<=0):raise ValueError('positive finite observable tolerances required')
    def assess(primary,auxiliary):
        maximum=0.;values=[]
        for i,(c,q) in enumerate(fields):
            cc,qq=project(c,q,primary,auxiliary)
            tiers=dict(primary_modes=sorted(primary),auxiliary_modes=sorted(auxiliary),
                       spectator_modes=sorted(set(stable)-primary-auxiliary),projection_model=projection_model)
            try:value=np.atleast_1d(np.asarray(evaluate_model(i,cc,qq,tiers),float))
            except (ValueError,FloatingPointError,np.linalg.LinAlgError):return float('inf'),[]
            if value.shape!=shape or not np.all(np.isfinite(value)):return float('inf'),[]
            bound=tol+relative_tolerance*np.maximum(abs(value),abs(baseline[i]))
            maximum=max(maximum,float(np.max(abs(value-baseline[i])/bound)));values.append(value.tolist())
        return maximum,values
    primary=set(stable);auxiliary=set();trace=[]
    initial,_=assess(primary,auxiliary)
    if initial>1:raise ValueError('Hessian-scan transverse coverage fails the observable tolerance; qualify unsupported families first')
    while True:
        candidates=[]
        for a in sorted(primary|auxiliary):
            p=primary.copy();s=auxiliary.copy()
            if a in p:p.remove(a);s.add(a);transition='primary_to_auxiliary'
            else:s.remove(a);transition='auxiliary_to_spectator'
            error,_=assess(p,s)
            if error<=1:candidates.append((error,a,transition,p,s))
        if not candidates:break
        error,a,transition,primary,auxiliary=min(candidates,key=lambda item:(item[0],item[1]))
        trace.append(dict(mode=a,transition=transition,maximum_normalized_error=error))
    error,values=assess(primary,auxiliary)
    return dict(primary_modes=sorted(primary),primary_modes_total=sorted({reaction_mode}|primary),
        reactive_primary_mode=reaction_mode,auxiliary_modes=sorted(auxiliary),
        spectator_modes=sorted(set(stable)-primary-auxiliary),maximum_normalized_error=error,
        reference_observables=[b.tolist() for b in baseline],model_observables=values,
        reductions=trace,independent_holdout_required=True,
        reaction_terms='held fixed; qualify separately; harmonic transverse Hessian may vary along IRC',
        projection_model=projection_model,
        selection='greedy observable-controlled reduction; no global cost optimality claim')
