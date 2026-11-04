"""Disjoint occupation reservoirs from acquired-field diagonal spectra.

The low occupation field is unchanged. Higher diagonal levels come from a
declared variational spectrum, rather than freezing the last backward gap.
Intermode couplings to the cone anchor are retained. A selected positive
pair can be counted jointly; remaining added-quanta couplings are separable.
"""
import numpy as np


def matched_diagonal_quartic(frequencies_cm,cubic_cm,quartic_cm,metric_first_cm,metric_second_cm,target_diagonal_x_cm):
    """Effective 1D potentials retaining the full-field HDC diagonal X.

    The diagonal slice alone omits virtual transitions involving other modes.
    Its quartic coefficient is shifted by 16*(X_full-X_slice), because the
    q^4/24 diagonal contribution to the occupation coefficient is F4/16.
    Native metric terms are included in X_slice before this matching.
    This is a spectroscopic potential, not a replacement of the acquired PES.
    """
    from .vibronic_vpt2 import SAMVPT2Field
    from .hdc_polynomial import sam_hdc_polynomial_energy
    values=[np.asarray(a,float) for a in (frequencies_cm,cubic_cm,quartic_cm,metric_first_cm,metric_second_cm,target_diagonal_x_cm)]
    w,c,q,g1,g2,target=values
    if w.ndim!=1 or not len(w) or any(a.shape!=w.shape or not np.all(np.isfinite(a)) for a in values) or np.any(w<=0):
        raise ValueError('matching finite diagonal force/metric arrays and positive frequencies required')
    source=[]
    for i in range(len(w)):
        field=SAMVPT2Field(w[i:i+1],np.full((1,)*3,c[i]),np.full((1,)*4,q[i]),
            np.full((1,)*3,g1[i]),np.full((1,)*4,g2[i]),0.,'complete','effective_diagonal_spectral_tail')
        source.append(sam_hdc_polynomial_energy(field).x_matrix_cm[0,0])
    source=np.asarray(source)
    return dict(effective_quartic_cm=q+16*(target-source),slice_diagonal_x_cm=source,
        target_diagonal_x_cm=target,acquired_force_field_changed=False,
        interpretation='effective one-mode potential matched to full-field HDC diagonal anharmonicity')


def anchor_mode_spectra(levels_cm, fundamentals_cm, diagonal_x_cm):
    """Preserve n=0,1,2 HDC energies; use variational gaps above n=2.

    An affine correction matches the fundamental. A constant offset for all
    n>=2 matches the overtone without an unbounded quadratic correction.
    No level is sorted, clipped, discarded, or assigned an absolute DVR ZPE.
    """
    raw=np.asarray(levels_cm,float);nu=np.asarray(fundamentals_cm,float);x=np.asarray(diagonal_x_cm,float)
    if (raw.ndim!=2 or raw.shape[1]<4 or nu.shape!=(len(raw),) or x.shape!=nu.shape
            or not all(np.all(np.isfinite(a)) for a in (raw,nu,x))
            or np.any(nu<=0) or np.any(np.diff(raw,axis=1)<=0)):
        raise ValueError('positive ordered variational spectra and matching HDC anchors required')
    raw=raw-raw[:,:1];n=np.arange(raw.shape[1])
    linear=nu-raw[:,1];result=raw+linear[:,None]*n
    offset=2*nu+2*x-result[:,2];result[:,2:]+=offset[:,None]
    if np.any(np.diff(result,axis=1)<=0):
        raise ValueError('HDC-anchored variational spectrum is not monotone')
    return dict(levels_cm=result,linear_shift_cm=linear,overtone_offset_cm=offset,
        ground_fundamental_overtone_preserved=True,perturbative_absolute_ZPE_replaced=False)


def spectral_matched_tail_factors(energy_model, matched_partition, mode_levels_cm, *,
                                  inverse_temperatures, tolerance=1e-8,coupled_pair=None,
                                  replacement_modes=None):
    """Replace cone gap guesses by forward variational excitation spectra.

    The finite table is closed with its positive final gap. The corresponding
    geometric remainder is returned separately, never hidden in the sum. Its
    magnitude controls the table truncation of this declared completion;
    it is not a physical error bound on the unknown remote PES.
    Optional replacement_modes limits the spectral substitution to supplied
    modes; every other mode retains its exact native geometric cone.
    """
    beta=np.asarray(inverse_temperatures,float);levels=np.asarray(mode_levels_cm,float)
    anchors=np.asarray(matched_partition['tail_anchor_states']);nu=np.asarray(energy_model.fundamentals_cm)
    x=np.asarray(energy_model.x_matrix_cm)
    if (beta.ndim!=1 or not len(beta) or np.any(beta<=0) or not np.all(np.isfinite(beta))
            or levels.ndim!=2 or levels.shape[0]!=len(nu) or levels.shape[1]<5
            or not np.all(np.isfinite(levels)) or np.any(np.diff(levels,axis=1)<=0)
            or anchors.ndim!=2 or anchors.shape[1]!=len(nu)
            or not matched_partition.get('disjoint_tails',False)
            or np.any(levels[:,0]!=0) or not np.allclose(levels[:,1],nu,rtol=0,atol=1e-7)
            or energy_model.y_tensor_cm is not None or not 0<tolerance<1):
        raise ValueError('quadratic HDC model, monotone anchored spectra and disjoint cones required')
    selected=_replacement_modes(replacement_modes,len(nu))
    if coupled_pair is not None:
        pair=tuple(coupled_pair)
        if (len(pair)!=2 or len(set(pair))!=2 or any(not isinstance(i,(int,np.integer)) or not 0<=i<len(nu) for i in pair)
                or x[pair]<0 or any(i not in selected for i in pair)):
            raise ValueError('two distinct valid modes with nonnegative cross coupling required')
    else:pair=()
    off=x-np.diag(np.diag(x));products=[];bounds=[];ladders=[]
    for anchor_index,anchor in enumerate(anchors):
        active=int(np.max(np.flatnonzero(anchor))) if np.any(anchor) else 0
        logs=np.zeros(len(beta));remainder_log=np.zeros(len(beta));row=[];summed={}
        cross=off@anchor
        for i in range(active,len(nu)):
            if i not in selected:
                gap=float(matched_partition['tail_frequencies_cm'][anchor_index,i])
                if not np.isfinite(gap) or gap<=0:
                    raise ValueError('positive native cone spacing required')
                logs-=np.log(-np.expm1(-beta*gap))
                row.append(dict(mode=i,native_geometric_frequency_cm=gap,
                                exact_geometric_spectrum=True))
                continue
            a=int(anchor[i]);j=np.arange(levels.shape[1]-a)
            e=levels[i,a:]-levels[i,a]+cross[i]*j
            gaps=np.diff(e)
            if len(gaps)<2 or np.any(gaps<=0):
                raise ValueError('conditional spectral cone has a nonpositive forward gap')
            finite=np.exp(-beta[:,None]*e).sum(axis=1)
            residual=np.exp(-beta*(e[-1]+gaps[-1]))/(-np.expm1(-beta*gaps[-1]))
            logs+=np.log(finite)+np.log1p(residual/finite);remainder_log+=np.log1p(residual/finite)
            summed[i]=(e,finite+residual,finite)
            row.append(dict(mode=i,levels_cm=e,final_gap_cm=float(gaps[-1])))
        if pair and all(i in summed for i in pair):
            i,j=pair;ei,zi,fi=summed[i];ej,zj,fj=summed[j]
            joint=(ei[:,None]+ej[None,:]+x[i,j]*np.arange(len(ei))[:,None]*np.arange(len(ej))[None,:]).ravel()
            zjoint=np.exp(-beta[:,None]*joint).sum(axis=1)
            # Positive cross coupling bounds omitted joint levels by the
            # product of uncoupled positive tails, independently of sorting.
            pair_bound=(zi*zj)*(-np.expm1(-np.log(zi/fi)-np.log(zj/fj)))/zjoint
            logs+=np.log(zjoint)-np.log(zi)-np.log(zj)
            remainder_log+=np.log1p(pair_bound)
            row=[r for r in row if r['mode'] not in pair]
            row.append(dict(modes=pair,levels_cm=np.sort(joint),cross_cm=float(x[i,j]),finite_joint_spectrum=True))
        products.append(np.expm1(logs));bounds.append(np.expm1(remainder_log));ladders.append(row)
    bound=np.asarray(bounds)
    if bound.size and np.max(bound)>tolerance:
        raise ValueError('spectral table remainder exceeds declared numerical tolerance')
    return dict(residual_products=np.asarray(products),relative_table_completion=bound,
        maximum_relative_table_completion=float(np.max(bound)) if bound.size else 0.,ladders=ladders,
        disjoint_tails=True,finite_occupation_model_preserved=True,
        boundary_policy='forward variational gaps with anchor intermode shifts',
        coupled_pair=pair,cross_couplings_between_added_quanta=bool(pair),
        replaced_mode_indices=tuple(sorted(selected)),
        remaining_added_quanta_cross_terms=False,physical_spectral_accuracy_qualified=False)


def _replacement_modes(indices,modes):
    if indices is None:return set(range(modes))
    indices=tuple(indices)
    if (len(set(indices))!=len(indices) or any(isinstance(i,(bool,np.bool_))
            or not isinstance(i,(int,np.integer)) or not 0<=i<modes for i in indices)):
        raise ValueError('distinct valid spectral replacement mode indices required')
    return set(indices)


def spectral_matched_energies(energy_model,occupations,matched_partition,mode_levels_cm,*,coupled_pair=None,
                              replacement_modes=None):
    """Expose the same completed energies to counting/sampling consumers.

    Above the supplied table use its declared final positive spacing. This
    is the exact geometric completion integrated by the partition owner.
    """
    n=np.asarray(occupations);L=np.asarray(mode_levels_cm,float);m=len(energy_model.fundamentals_cm)
    if (n.ndim!=2 or n.shape[1]!=m or n.dtype.kind not in 'iu' or np.any(n<0)
            or L.ndim!=2 or L.shape[0]!=m or L.shape[1]<3 or not np.all(np.isfinite(L))
            or np.any(np.diff(L,axis=1)<=0) or not matched_partition.get('disjoint_tails',False)
            or energy_model.y_tensor_cm is not None):
        raise ValueError('integer occupations, positive spectra and shared disjoint domain required')
    selected=_replacement_modes(replacement_modes,m)
    cutoff=matched_partition['maximum_explicit_quanta'];lookup={tuple(a):i for i,a in enumerate(matched_partition['tail_anchor_states'])}
    if cutoff>=L.shape[1]-1:raise ValueError('spectra must extend beyond the finite occupation domain')
    off=energy_model.x_matrix_cm-np.diag(np.diag(energy_model.x_matrix_cm));last=L.shape[1]-1
    result=np.empty(len(n));low=n.sum(axis=1)<=cutoff;result[low]=energy_model.energies(n[low])
    checked=set()
    for r in np.flatnonzero(~low):
        anchor=np.zeros(m,int);remaining=cutoff
        for i in range(m):anchor[i]=min(int(n[r,i]),remaining);remaining-=anchor[i]
        if tuple(anchor) not in lookup:raise ValueError('occupation has no shared cone anchor')
        extra=n[r]-anchor;cross=off@anchor
        if tuple(anchor) not in checked:
            active=int(np.max(np.flatnonzero(anchor))) if np.any(anchor) else 0
            for i in range(active,m):
                if i in selected and np.any(np.diff(L[i,anchor[i]:])+cross[i]<=0):
                    raise ValueError('conditional spectral cone has a nonpositive forward gap')
            checked.add(tuple(anchor))
        high=L[np.arange(m),np.minimum(n[r],last)]+np.maximum(n[r]-last,0)*(L[:,-1]-L[:,-2])
        base=L[np.arange(m),anchor]
        use=np.array([i in selected for i in range(m)])
        if np.any((L[:,-1]-L[:,-2]+cross)[(extra>0)&use]<=0):raise ValueError('conditional spectral final gap is nonpositive')
        native=~use&(extra>0)
        gaps=np.asarray(matched_partition['tail_frequencies_cm'])[lookup[tuple(anchor)]]
        if np.any(~np.isfinite(gaps[native])) or np.any(gaps[native]<=0):raise ValueError('positive native cone spacing required')
        value=(energy_model.energies(anchor[None,:])[0]+np.sum((high-base)[use])
               +extra[use]@cross[use]+extra[native]@gaps[native])
        if coupled_pair is not None:
            i,j=coupled_pair
            if i==j or not 0<=i<m or not 0<=j<m or off[i,j]<0 or i not in selected or j not in selected:raise ValueError('valid nonnegative coupled pair required')
            value+=off[i,j]*extra[i]*extra[j]
        result[r]=value
    if not np.all(np.isfinite(result)):raise FloatingPointError('completed spectral energy exceeds floating-point range')
    return result
