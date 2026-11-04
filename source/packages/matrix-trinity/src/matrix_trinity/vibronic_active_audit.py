"""Describe omitted polynomial terms without assigning a rate-error certificate."""
from itertools import combinations_with_replacement
from math import factorial

import numpy as np

from .vibronic_vpt2 import SAMVPT2Field


def gaussian_spectator_tail_bounds(*, norm_squared, exponential_number_moment,
        number_tilt, maximum_total_quanta, energy_edge_cm,
        vibronic_origin_upper_cm, gaussian_sigma_cm, rate_prefactor):
    """Bound omitted occupation and energy sectors of a fixed Gaussian rate.

    ``exponential_number_moment`` is the unnormalized strength sum weighted
    by exp(number_tilt*N) over the actual spectator subset. The origin upper
    bound must use a lower spectral bound for the actual finite active H.
    The union bound excludes active-space, basis and potential-model errors.
    """
    values=(norm_squared,exponential_number_moment,number_tilt,energy_edge_cm,
            vibronic_origin_upper_cm,gaussian_sigma_cm,rate_prefactor)
    if (not all(np.isfinite(v) for v in values) or norm_squared<0
            or exponential_number_moment<0 or number_tilt<=0
            or gaussian_sigma_cm<=0 or rate_prefactor<0 or energy_edge_cm<=0
            or isinstance(maximum_total_quanta,bool)
            or not isinstance(maximum_total_quanta,int) or maximum_total_quanta<0):
        raise ValueError('finite positive domain/width/tilt and nonnegative strengths required')
    peak=1/(np.sqrt(2*np.pi)*gaussian_sigma_cm)
    number_strength=min(norm_squared,exponential_number_moment*
                        np.exp(-(maximum_total_quanta+1)*number_tilt))
    distance=max(0.,energy_edge_cm-vibronic_origin_upper_cm)
    number_bound=rate_prefactor*peak*number_strength
    energy_bound=rate_prefactor*peak*norm_squared*np.exp(-.5*(distance/gaussian_sigma_cm)**2)
    return dict(occupation_tail_bound_s_1=float(number_bound),
        energy_tail_bound_s_1=float(energy_bound),
        combined_omitted_rate_bound_s_1=float(number_bound+energy_bound),
        energy_edge_exceeds_origin=energy_edge_cm>vibronic_origin_upper_cm,
        bounds_include_roundoff_enclosure=False,
        finite_basis_convergence_certified=False)


def restrict_modal_field(field, modes):
    """Restrict the unchanged potential to a declared coordinate subspace.

    Omitted coordinates are set to zero, not averaged over their vacuum.
    All mixed terms among retained modes survive, including transported
    four-distinct-index quartics. This operation makes no confinement claim.
    """
    if not isinstance(field, SAMVPT2Field):
        raise TypeError('SAMVPT2Field required')
    indices = np.asarray(modes)
    n = len(field.frequencies_cm)
    if (indices.ndim != 1 or indices.dtype.kind not in 'iu' or not len(indices)
            or len(set(map(int, indices))) != len(indices)
            or np.any(indices < 0) or np.any(indices >= n)):
        raise ValueError('unique retained modal indices required')
    if np.any(field.metric_first_cm) or np.any(field.metric_second_cm):
        raise ValueError('restriction currently supports potential-only modal fields')
    return SAMVPT2Field(field.frequencies_cm[indices],
        field.cubic_cm[np.ix_(indices, indices, indices)],
        field.quartic_cm[np.ix_(indices, indices, indices, indices)],
        constant_shift_cm=field.constant_shift_cm,
        quartic_coverage=field.quartic_coverage)


def retain_anharmonic_modes(field, modes):
    """Keep a common full harmonic frame while switching off omitted V3/V4.

    The finite basis can therefore be identical in paired mode-inclusion
    calculations. Terms are retained only when every index is in ``modes``;
    no spectator vacuum contraction or compensating frequency shift is used.
    """
    sub = restrict_modal_field(field, modes)
    indices = np.asarray(modes)
    cubic = np.zeros_like(field.cubic_cm)
    quartic = np.zeros_like(field.quartic_cm)
    cubic[np.ix_(indices, indices, indices)] = sub.cubic_cm
    quartic[np.ix_(indices, indices, indices, indices)] = sub.quartic_cm
    return SAMVPT2Field(field.frequencies_cm, cubic, quartic,
        constant_shift_cm=field.constant_shift_cm, quartic_coverage=field.quartic_coverage)


def audit_active_polynomial(field, active_modes, *, coefficient_tolerance_cm=1e-8):
    """Partition canonical cubic/quartic coefficients in dimensionless modes.

    Tensor entries follow V3=F3*q^3/3! and V4=F4*q^4/4!. Reported coefficients
    include the permutation multiplicity. Their norms are descriptive ranking
    scores in this fixed modal frame, not errors in energies, states or rates.
    Terms at or below the requested reporting tolerance are counted separately;
    the field and all downstream numerical calculations remain unchanged.
    """
    if not isinstance(field, SAMVPT2Field):
        raise TypeError('SAMVPT2Field required')
    n=len(field.frequencies_cm)
    active=np.asarray(active_modes)
    if (active.ndim!=1 or active.dtype.kind not in 'iu' or not len(active)
            or len(np.unique(active))!=len(active) or np.any(active<0)
            or np.any(active>=n) or not np.isfinite(coefficient_tolerance_cm)
            or coefficient_tolerance_cm<0):
        raise ValueError('unique active mode indices and nonnegative reporting tolerance required')
    aset=set(map(int,active));omitted=sorted(set(range(n))-aset)
    summaries={};scores={i:0. for i in omitted}
    for degree,tensor in ((3,field.cubic_cm),(4,field.quartic_cm)):
        groups={name:[] for name in ('retained','active_spectator','spectator_only')}
        below=0
        for key in combinations_with_replacement(range(n),degree):
            divisor=1
            for i in set(key):divisor*=factorial(key.count(i))
            coefficient=float(tensor[key])/divisor
            if abs(coefficient)<=coefficient_tolerance_cm:
                below+=1;continue
            modes=set(key)
            group='retained' if modes<=aset else 'active_spectator' if modes&aset else 'spectator_only'
            groups[group].append(coefficient)
            if group=='active_spectator':
                for i in modes-aset:scores[i]+=coefficient**2
        summaries[str(degree)]={name:dict(term_count=len(values),coefficient_l2_cm=float(np.linalg.norm(values)),maximum_absolute_coefficient_cm=max(map(abs,values),default=0.)) for name,values in groups.items()}
        summaries[str(degree)]['terms_at_or_below_reporting_tolerance']=below
    return dict(schema='matrix.trinity.active_polynomial_audit.v1',active_modes=active.tolist(),
        omitted_modes=omitted,degrees=summaries,coefficient_reporting_tolerance_cm=coefficient_tolerance_cm,
        omitted_mode_coupling_ranking=[dict(mode=i,coefficient_l2_cm=float(np.sqrt(scores[i]))) for i in sorted(omitted,key=lambda i:(-scores[i],i))],
        ranking_is_rate_error_bound=False,physical_field_modified=False,active_space_convergence_certified=False)


def quartic_descent_witness(field, *, seed=481731, random_starts=16,
                            maximum_iterations=300):
    """Find, but never assume absence of, an unbounded ray of a bare quartic.

    A negative leading coefficient along any real normalized direction proves
    polynomial descent to minus infinity. Failure of the finite search is not
    a confinement certificate. The radial barrier is that of this ray only.
    """
    from scipy.optimize import minimize
    if not isinstance(field, SAMVPT2Field):
        raise TypeError('SAMVPT2Field required')
    if (isinstance(random_starts, bool) or not isinstance(random_starts, int)
            or random_starts < 0 or isinstance(maximum_iterations, bool)
            or not isinstance(maximum_iterations, int) or maximum_iterations < 1):
        raise ValueError('nonnegative start count and positive iteration budget required')
    n=len(field.frequencies_cm);f=field.quartic_cm
    rng=np.random.default_rng(seed)
    starts=list(np.eye(n))+[rng.normal(size=n) for _ in range(random_starts)]
    scale=max(float(np.linalg.norm(f)),1.)
    def evaluate(x):
        radius=float(x@x)
        if radius<=np.finfo(float).tiny:raise FloatingPointError('zero descent direction')
        numerator=float(np.einsum('ijkl,i,j,k,l',f,x,x,x,x,optimize=True))/24
        gradient=np.einsum('ijkl,j,k,l->i',f,x,x,x,optimize=True)/6
        return numerator/radius**2/scale,(gradient/radius**2-4*numerator*x/radius**3)/scale
    best=None
    for start in starts:
        start=start/np.linalg.norm(start)
        fit=minimize(evaluate,start,jac=True,method='BFGS',options=dict(maxiter=maximum_iterations,gtol=1e-10))
        direction=fit.x/np.linalg.norm(fit.x)
        q=float(np.einsum('ijkl,i,j,k,l',f,direction,direction,direction,direction,optimize=True))/24
        if best is None or q<best[0]:best=(q,direction,bool(fit.success))
    q,u,success=best
    negative=q < -1e-10*scale
    result=dict(schema='matrix.trinity.quartic_descent_witness.v1',
        negative_quartic_ray_found=bool(negative),leading_coefficient_cm=q,
        direction=u.tolist(),seed=seed,search_start_count=len(starts),
        best_optimizer_converged=success,search_certifies_confinement=False,
        potential_modified=False)
    if negative:
        k=float(field.frequencies_cm@(u*u))
        c=float(np.einsum('ijk,i,j,k',field.cubic_cm,u,u,u,optimize=True))/6
        rays=[]
        for sign in (1.,-1.):
            cubic=sign*c
            roots=np.roots([4*q,3*cubic,k])
            positive=sorted(float(z.real) for z in roots if abs(z.imag)<1e-9 and z.real>0)
            radius=positive[0]
            height=.5*k*radius**2+cubic*radius**3+q*radius**4
            rays.append(dict(direction_sign=sign,quadratic_coefficient_cm=.5*k,
                cubic_coefficient_cm=cubic,quartic_coefficient_cm=q,
                barrier_radius_dimensionless=radius,barrier_height_above_origin_cm=height))
        result['radial_barriers']=rays
    return result


def harmonic_basis_count_enclosure(frequencies_cm, maximum_energy_cm, *,
                                    maximum_total_quanta=40, bin_width_cm=10.,
                                    count_cap=10**12):
    """Bound a full oscillator basis count without enumerating its rows.

    Rounding each frequency up/down gives lower/upper integer-coin counts.
    Count saturation is explicit and can never be mistaken for an exact count.
    """
    w=np.asarray(frequencies_cm,float)
    if (w.ndim!=1 or not len(w) or np.any(w<=0) or not np.all(np.isfinite(w))
            or not np.isfinite(maximum_energy_cm) or maximum_energy_cm<=0
            or not np.isfinite(bin_width_cm) or bin_width_cm<=0
            or isinstance(maximum_total_quanta,bool) or not isinstance(maximum_total_quanta,int)
            or maximum_total_quanta<0 or isinstance(count_cap,bool) or not isinstance(count_cap,int)
            or not 1<=count_cap<np.iinfo(np.int64).max//4):
        raise ValueError('positive frequencies/energy/grid and bounded integer controls required')
    if bin_width_cm>=min(w):raise ValueError('count grid must resolve every positive frequency')
    bins=int(np.floor(maximum_energy_cm/bin_width_cm))
    if (maximum_total_quanta+1)*(bins+1)>2000000:
        raise MemoryError('count enclosure grid exceeds two million cells')
    totals=[]
    for rounded in (np.ceil(w/bin_width_cm),np.floor(w/bin_width_cm)):
        table=np.zeros((maximum_total_quanta+1,bins+1),dtype=np.int64);table[0,0]=1
        for cost in rounded.astype(int):
            if cost>bins:continue
            for quanta in range(1,maximum_total_quanta+1):
                table[quanta,cost:]=np.minimum(count_cap,
                    table[quanta,cost:]+table[quanta-1,:bins-cost+1])
        total=sum(map(int,table.ravel()))
        totals.append(min(count_cap,total))
    return dict(schema='matrix.trinity.harmonic_basis_count_enclosure.v1',
        lower_bound=totals[0],upper_bound=totals[1] if totals[1]<count_cap else None,
        lower_count_saturated=totals[0]==count_cap,upper_count_saturated=totals[1]==count_cap,
        count_cap=count_cap,maximum_energy_cm=float(maximum_energy_cm),
        maximum_total_quanta=maximum_total_quanta,bin_width_cm=bin_width_cm,
        frequency_rounding='ceil for lower count, floor for upper count')
