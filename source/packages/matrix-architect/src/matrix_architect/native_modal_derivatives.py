"""Match sparse measured modal derivatives to native ZAFF corrections.

Only supplied derivative entries are observations. Missing and unprinted entries
are never replaced with zero-energy or zero-derivative pseudo-measurements.
Native corrections start at degree three, preserving reference E/G/H exactly.
"""
from itertools import permutations,combinations
import numpy as np
from scipy.linalg import lstsq
from matrix_gf.harmonic import CM_PER_HARTREE
from types import SimpleNamespace
from matrix_chem.primitive_coordinates import primitive_b_matrix,eval_primitives
from matrix_zaff.basic_primitive_derivative import analytic_basic_primitive_second_derivative
from .asymptotic_pic import AsymptoticPicCorrection


def native_factor_modal_two_jet(primitives,reference_angstrom,displacements_angstrom):
    """First/second jets of the catalog's unit-normalized native factors."""
    xyz=np.asarray(reference_angstrom,float);dx=np.asarray(displacements_angstrom,float).reshape(len(displacements_angstrom),-1).T
    q=eval_primitives(primitives,xyz);first=primitive_b_matrix(primitives,xyz)@dx
    second=np.zeros((len(primitives),dx.shape[1],dx.shape[1]))
    for i,p in enumerate(primitives):
        item=SimpleNamespace(function={'bond':'R','angle':'A','dihedral':'D','out_of_plane':'U'}[p.kind],atoms=tuple(a+1 for a in p.atoms))
        h=np.zeros((xyz.size,xyz.size))
        for a,b,v in analytic_basic_primitive_second_derivative(item,xyz).entries:h[a,b]+=v
        second[i]=dx.T@h@dx
        curvature=-2/q[i] if p.kind=='bond' else 1/np.tan(q[i]) if p.kind=='angle' else 0.
        second[i]+=curvature*np.outer(first[i],first[i])
    return first,second


def native_modal_derivative_design(first,second,factors,indices,*,maximum_matrix_bytes=8*1024**3):
    """Exact third/fourth product rule for existing native cubic/quartic factors.

    Every support-bond switch starts at degree four, so cannot change the
    third/fourth jet of a correction whose leading degree is at least three.
    This does not replace those switches in the production evaluator.
    """
    rows=tuple(tuple(k) for k in indices)
    if not rows:raise ValueError('at least one measured derivative required')
    degree=len(rows[0]);factors=tuple(factors)
    if degree not in (3,4) or any(len(k)!=degree for k in rows):raise ValueError('one derivative order per design required')
    if len(rows)*len(factors)*8>maximum_matrix_bytes:raise MemoryError('native derivative design exceeds declared budget')
    result=np.zeros((len(rows),len(factors)))
    for leading in (3,4):
        positions=[];products=[]
        for j,term in enumerate(factors):
            if any(t.form not in ('native','inverse_distance','sin') or t.harmonic!=1 for t in term):raise ValueError('unsupported local factor jet')
            expanded=tuple(t.primitive_index for t in term for _ in range(t.power))
            if len(expanded)==leading:positions.append(j);products.append(expanded)
        if not positions or leading>degree:continue
        product=np.array(products,int);block=np.zeros((len(rows),len(product)))
        for row,key in enumerate(rows):
            if degree==leading:
                for order in permutations(range(degree)):
                    v=np.ones(len(product))
                    for factor,axis in enumerate(order):v*=first[product[:,factor],key[axis]]
                    block[row]+=v
            else:
                for pair in combinations(range(4),2):
                    singles=[i for i in range(4) if i not in pair]
                    for double in range(3):
                        others=[i for i in range(3) if i!=double]
                        for order in permutations(singles):
                            block[row]+=second[product[:,double],key[pair[0]],key[pair[1]]]*first[product[:,others[0]],key[order[0]]]*first[product[:,others[1]],key[order[1]]]
        result[:,positions]=block
    return result


def fit_native_modal_derivatives(first,second,factors,observed_cubic,observed_quartic,baseline_cubic,baseline_quartic,*,relative_rank_cutoff=1e-9,maximum_matrix_bytes=8*1024**3):
    """Rank-revealing local fit, with no global-PES acceptance implied.

    Coefficients use the catalog's angstrom/radian native convention. The
    minimum Euclidean norm resolves unobserved coefficient freedoms; its
    extrapolation must subsequently be checked with independent QM points.
    """
    factors=tuple(factors);degrees=np.array([sum(t.power for t in f) for f in factors])
    if np.any((degrees!=3)&(degrees!=4)):raise ValueError('corrections must start at cubic/quartic order')
    coefficients=np.zeros(len(factors));report={}
    for order,observed,baseline in [(3,observed_cubic,baseline_cubic),(4,observed_quartic,baseline_quartic)]:
        keys=tuple(sorted(observed));active=np.flatnonzero(degrees==order)
        design=native_modal_derivative_design(first,second,tuple(factors[i] for i in active),keys,maximum_matrix_bytes=maximum_matrix_bytes)
        target=np.array([observed[k]-baseline[k] for k in keys],float)
        if order==4:
            cubic=np.flatnonzero((degrees==3)&(coefficients!=0))
            previous=native_modal_derivative_design(first,second,tuple(factors[i] for i in cubic),keys,maximum_matrix_bytes=maximum_matrix_bytes)
            target-=previous@coefficients[cubic]
        values,_,rank,_=lstsq(design,target,cond=relative_rank_cutoff,lapack_driver='gelsy')
        coefficients[active]=values
        residual=design@values-target
        report[str(order)]=dict(observed_entries=len(keys),candidate_count=len(active),resolved_rank=int(rank),maximum_residual_cm=float(np.max(abs(residual))),rms_residual_cm=float(np.sqrt(np.mean(residual**2))))
    report.update(unprinted_derivatives_used_as_zero=False,off_reference_quartic_targets_used=False,reference_EGH_changed=False,physical_field_qualified=False)
    return tuple(AsymptoticPicCorrection(float(c)/CM_PER_HARTREE,f) for c,f in zip(coefficients,factors) if abs(c)>1e-14),report


def complete_native_quartic_catalog(primitives,first,existing_factors,*,relative_rank_cutoff=1e-9,complete_cubic=False):
    """Complete the quartic response with a rank-certified native coordinate view.

    A local rank failure is resolved by adding native products, including four
    distinct primitives. This adds candidate functions, not zero observations
    for unmeasured four-distinct modal derivatives. Every product retains its
    production support-bond switches.
    """
    from itertools import combinations_with_replacement
    from scipy.linalg import qr
    from .asymptotic_pic import AsymptoticPicFactor
    a=np.asarray(first,float)
    _,r,pivots=qr(a.T,mode='economic',pivoting=True)
    diagonal=abs(np.diag(r));rank=int(np.count_nonzero(diagonal>diagonal[0]*relative_rank_cutoff))
    if rank!=a.shape[1]:raise ValueError('native factors do not span the declared modal space')
    selected=tuple(sorted(int(i) for i in pivots[:rank]))
    extra=[]
    rows=(row for degree in ((3,4) if complete_cubic else (4,)) for row in combinations_with_replacement(selected,degree))
    for row in rows:
        extra.append(tuple(AsymptoticPicFactor(i,'inverse_distance' if primitives[i].kind=='bond' else 'native' if primitives[i].kind=='angle' else 'sin',row.count(i)) for i in sorted(set(row))))
    combined=tuple(dict.fromkeys(tuple(existing_factors)+tuple(extra)))
    return combined,dict(native_coordinate_rank=rank,selected_primitive_indices_zero_based=selected,added_native_functions=len(combined)-len(existing_factors),cubic_completion=bool(complete_cubic),unmeasured_derivative_observations_added=0)
