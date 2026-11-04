"""Positive occupation ladders attached to resident disjoint tail cones.

The finite field and first boundary gap are immutable. Exponential or
Gaussian relaxation connects that gap to a declared positive asymptotic
scale. These are model completions, not independently validated spectra.
"""
import numpy as np


def positive_gap_ladder(boundary_gap_cm, asymptotic_gap_cm, *, inverse_temperatures,
                        form, width_quanta=2., relative_tolerance=1e-11,
                        maximum_levels=100000):
    """Return excitation levels and rigorous geometric partition-tail bounds.

    gap(j)=w+(g-w)*exp(-((j-1)/width)**p), p=1 or 2. Every gap
    is between positive g and w, with gap(1)=g. No sign clipping occurs.
    """
    beta=np.asarray(inverse_temperatures,float)
    g,w=float(boundary_gap_cm),float(asymptotic_gap_cm)
    if (not np.all(np.isfinite([g,w,width_quanta,relative_tolerance])) or min(g,w,width_quanta)<=0
            or not 0<relative_tolerance<1 or beta.ndim!=1 or not len(beta)
            or np.any(beta<=0) or not np.all(np.isfinite(beta))
            or form not in ('exponential','gaussian') or isinstance(maximum_levels,bool)
            or not isinstance(maximum_levels,int) or maximum_levels<2):
        raise ValueError('positive gaps, temperatures, width and explicit damping form required')
    lower=min(g,w);power=1 if form=='exponential' else 2
    # This sufficient count uses only the certified positive lower gap.
    target=-np.log(relative_tolerance*(-np.expm1(-beta.min()*lower)))/beta.min()
    count=max(2,int(np.ceil(target/lower)))
    if count>maximum_levels:raise ValueError('declared positive-ladder numerical level budget exhausted')
    j=np.arange(count,dtype=float)
    gaps=w+(g-w)*np.exp(-(j/width_quanta)**power)
    levels=np.r_[0.,np.cumsum(gaps)]
    z=np.exp(-beta[:,None]*levels).sum(axis=1)
    bound=np.exp(-beta*(levels[-1]+lower))/(-np.expm1(-beta*lower))
    if np.any(bound/z>relative_tolerance):raise RuntimeError('positive-ladder partition bound failed')
    return dict(levels_cm=levels,partition=z,omitted_partition_upper_bound=bound,
        minimum_gap_lower_bound_cm=lower,first_gap_cm=float(gaps[0]),
        asymptotic_gap_cm=w,width_quanta=float(width_quanta),form=form,
        positive_gaps_proved=True,physical_spectral_accuracy_qualified=False)


def relaxed_matched_tail_factors(matched_partition, asymptotic_frequencies_cm, *,
                                 inverse_temperatures, form, width_quanta=2.,
                                 relative_tolerance=1e-11):
    """Use the SAME anchors and disjoint cones as statistical_partition.

    residual_products[a,t] excludes the cone's zero increment, as in
    the resident matched harmonic owner. Finite occupation energies and
    all anchors remain unchanged. No Wang--Landau resampling is required.
    """
    beta=np.asarray(inverse_temperatures,float)
    anchors=np.asarray(matched_partition['tail_anchor_states'])
    gaps=np.asarray(matched_partition['tail_frequencies_cm'],float)
    asym=np.asarray(asymptotic_frequencies_cm,float)
    if (anchors.ndim!=2 or gaps.shape!=anchors.shape or asym.shape!=(anchors.shape[1],)
            or not np.all(np.isfinite(asym)) or np.any(asym<=0)
            or not matched_partition.get('disjoint_tails',False)):
        raise ValueError('resident disjoint anchors and positive asymptotic frequencies required')
    products=[];maximum_bound=0.;maximum_levels=0
    cache={}
    for row in gaps:
        log_product=np.zeros(len(beta))
        for i in np.flatnonzero(np.isfinite(row)):
            key=(float(row[i]),float(asym[i]))
            if key not in cache:
                cache[key]=positive_gap_ladder(*key,inverse_temperatures=beta,form=form,
                    width_quanta=width_quanta,relative_tolerance=relative_tolerance/len(asym))
            ladder=cache[key];log_product+=np.log(ladder['partition'])
            maximum_bound=max(maximum_bound,float(max(ladder['omitted_partition_upper_bound']/ladder['partition'])))
            maximum_levels=max(maximum_levels,len(ladder['levels_cm']))
        products.append(np.expm1(log_product))
    return dict(residual_products=np.asarray(products),form=form,width_quanta=float(width_quanta),
        maximum_single_ladder_relative_omitted_bound=maximum_bound,
        maximum_ladder_levels=maximum_levels,finite_anchors_preserved=True,
        first_boundary_gaps_preserved=True,disjoint_tails=True,
        asymptotic_frequencies_cm=asym,physical_spectral_accuracy_qualified=False)
