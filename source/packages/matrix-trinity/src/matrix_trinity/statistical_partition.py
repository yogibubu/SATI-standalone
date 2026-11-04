"""Shared finite explicit sum and matched harmonic-tail thermochemistry.

Extracted from SONIC's resident matched_spt implementation. Inputs are levels,
occupations and boundary transitions, not a molecule or a stationary-field
class. Thus the same code applies to fixed-action SCTST providers and to
ordinary vibrational fields. It does not infer physical tail accuracy.
"""
from math import comb
import numpy as np


def matched_partition_moments(levels_cm, shells, occupations, inverse_temperatures,
                              boundary_transitions):
    """Sum finite levels and disjoint harmonic cones matched at their boundary.

    boundary_transitions(anchor, first_active) returns the positive frequencies
    for increasing modes first_active..M-1. For the first active mode use the
    last backward gap; higher unoccupied modes use their forward gaps. This
    disjoint cone partition is the existing SONIC convention, with no repeated
    anchor and no double counting of tails.
    """
    levels=np.asarray(levels_cm,float);shells=np.asarray(shells);states=np.asarray(occupations)
    beta=np.asarray(inverse_temperatures,float)
    if (levels.ndim!=1 or not len(levels) or not np.all(np.isfinite(levels))
            or shells.shape!=levels.shape or states.ndim!=2 or len(states)!=len(levels)
            or states.dtype.kind not in 'iu' or np.any(states<0)
            or len(np.unique(states,axis=0))!=len(states)
            or np.any(shells!=states.sum(axis=1)) or not np.any(states.sum(axis=1)==0)
            or beta.ndim!=1 or not len(beta) or np.any(beta<=0) or not np.all(np.isfinite(beta))):
        raise ValueError('finite levels, distinct integer states, shells and positive inverse temperatures required')
    maximum=int(max(shells));modes=states.shape[1]
    if len(states)!=comb(modes+maximum,maximum):raise ValueError('matched tail requires complete explicit occupation shells')
    # Shift all energies together; absolute origins are restored in log Z/U.
    origin=float(min(levels));excitation=levels-origin
    weights=np.exp(-beta[:,None]*excitation)
    sums=weights.sum(axis=1);first=weights@excitation;second=weights@(excitation**2)
    ids=np.flatnonzero(shells==maximum);anchors=states[ids].astype(int)
    frequencies=np.full((len(ids),modes),np.nan);tail=np.zeros(len(beta))
    for row,index in enumerate(ids):
        anchor=states[index];active=int(np.max(np.flatnonzero(anchor))) if maximum else 0
        gaps=np.asarray(boundary_transitions(anchor.copy(),active),float)
        if gaps.shape!=(modes-active,) or np.any(gaps<=0) or not np.all(np.isfinite(gaps)):
            raise ValueError('matched SPT tail requires positive boundary transitions')
        frequencies[row,active:]=gaps
        x=beta[:,None]*gaps;occupation=1/np.expm1(x)
        log_product=-np.sum(np.log(-np.expm1(-x)),axis=1)
        product=np.exp(log_product);residual=np.expm1(log_product)
        mean=np.sum(gaps*occupation,axis=1)
        variance=np.sum(gaps**2*occupation*(1+occupation),axis=1)
        energy=excitation[index];factor=weights[:,index]
        contribution=factor*residual;tail+=contribution;sums+=contribution
        first+=factor*(energy*residual+product*mean)
        second+=factor*(energy**2*residual+2*energy*product*mean+product*(mean**2+variance))
    mean=first/sums;variance=second/sums-mean**2
    return dict(log_partition=np.log(sums)-beta*origin,
        shifted_partition_sums=sums,energy_origin_cm=origin,
        mean_energy_cm=mean+origin,energy_variance_cm2=np.maximum(variance,0.),
        tail_fraction=tail/sums,tail_frequencies_cm=frequencies,
        tail_anchor_states=anchors,explicit_state_count=len(states),maximum_explicit_quanta=maximum,
        disjoint_tails=True,tail_accuracy_qualified=False)


def matched_occupation_partition(energy_model, *, inverse_temperatures,
                                 max_total_quanta, max_partition_states=30000):
    """Reuse the matched tail with any explicit occupation-energy provider.

    The provider exposes fundamentals_cm and energies(integer_rows). It may
    include regularized cubic/higher occupations. No harmonic coefficient is
    guessed: all tail gaps are actual differences of the supplied energies.
    This small explicit sector is bounded by max_partition_states. Large state
    counts continue to use the existing Wang--Landau owners.
    """
    from .mixed_statistical import occupation_shells
    modes=len(energy_model.fundamentals_cm)
    if isinstance(max_total_quanta,bool) or not isinstance(max_total_quanta,int) or max_total_quanta<0:
        raise ValueError('nonnegative explicit cutoff required')
    if isinstance(max_partition_states,bool) or not isinstance(max_partition_states,int) or max_partition_states<1:
        raise ValueError('positive explicit state budget required')
    maximum=max_total_quanta
    while maximum and comb(modes+maximum,maximum)>max_partition_states:maximum-=1
    if comb(modes+maximum,maximum)>max_partition_states:raise MemoryError('explicit state budget cannot contain the ground state')
    states=np.asarray(list(occupation_shells(modes,maximum)),int)
    levels=energy_model.energies(states)
    def transitions(anchor,active):
        values=[]
        for i in range(active,modes):
            other=anchor.copy()
            if i==active and maximum:
                other[i]-=1;pair=np.stack([anchor,other]);sign=1
            else:
                other[i]+=1;pair=np.stack([other,anchor]);sign=1
            e=energy_model.energies(pair);values.append(sign*(e[0]-e[1]))
        return values
    result=matched_partition_moments(levels,states.sum(axis=1),states,inverse_temperatures,transitions)
    result['requested_maximum_explicit_quanta']=max_total_quanta
    result['explicit_budget_limited']=maximum!=max_total_quanta
    return result


def matched_occupation_energies(energy_model, occupations, matched_partition):
    """Evaluate the SAME disjoint harmonic cones used by the partition owner.

    Complete low shells retain their supplied occupation energy. Each higher
    state has a unique prefix anchor on the boundary shell; subsequent quanta
    use the native last-backward / forward gaps from matched_partition.
    No state is discarded and no negative level is clipped. This exposes the
    resident thermochemical completion as a reusable path-energy provider.
    """
    states=np.asarray(occupations)
    n=len(energy_model.fundamentals_cm)
    if states.ndim!=2 or states.shape[1]!=n or states.dtype.kind not in 'iu' or np.any(states<0):
        raise ValueError('nonnegative integer occupation rows required')
    maximum=matched_partition['maximum_explicit_quanta']
    anchors=np.asarray(matched_partition['tail_anchor_states'],dtype=int)
    frequencies=np.asarray(matched_partition['tail_frequencies_cm'],float)
    if anchors.ndim!=2 or anchors.shape[1]!=n or frequencies.shape!=anchors.shape:
        raise ValueError('matched native anchor/frequency metadata required')
    lookup={tuple(a):i for i,a in enumerate(anchors)}
    values=np.empty(len(states),float)
    low=states.sum(axis=1)<=maximum
    values[low]=energy_model.energies(states[low])
    for row in np.flatnonzero(~low):
        state=states[row];anchor=np.zeros(n,dtype=int);remaining=maximum
        for i in range(n):
            anchor[i]=min(int(state[i]),remaining);remaining-=anchor[i]
            if remaining==0:break
        index=lookup[tuple(anchor)];extra=state-anchor;active=extra>0
        gaps=frequencies[index,active]
        if not np.all(np.isfinite(gaps)) or np.any(gaps<=0):
            raise ValueError('native matched cone has no positive required gap')
        values[row]=energy_model.energies(anchor[None,:])[0]+extra[active]@gaps
    if not np.all(np.isfinite(values)):raise FloatingPointError('matched occupation energy exceeds floating range')
    return values
