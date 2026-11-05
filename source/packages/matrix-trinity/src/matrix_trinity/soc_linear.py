"""Symmetry-projected linear SOC from explicitly orthogonal spin channels."""
from __future__ import annotations
import numpy as np


def coherent_soc_pairs(reference, pairs, *, mode_count, minimum_overlap=.8):
    """Signed central derivatives after independently tracked singlet/triplet phases.

    Each endpoint supplies ``singlet_tracking`` and ``triplet_tracking`` with
    phase, score and source from the electronic-state tracking owner (APOC).
    Phases multiply the candidate kets into the reference convention. SOC
    itself is never used to infer a phase, particularly at a symmetry zero.
    A triplet spin-frame rotation must already have been transported by the
    caller; a scalar phase cannot repair a rotated degenerate spin manifold.
    """
    reference=np.asarray(reference,complex)
    if (reference.shape!=(3,) or not np.all(np.isfinite(reference))
            or not isinstance(mode_count,int) or mode_count<1
            or not np.isfinite(minimum_overlap) or not 0<minimum_overlap<=1):
        raise ValueError('Finite SOC reference, positive mode count and overlap threshold required')
    gradient=np.zeros((3,mode_count),complex);rows=[];seen=set()
    def aligned(endpoint):
        value=np.asarray(endpoint['soc'],complex)
        if value.shape!=(3,) or not np.all(np.isfinite(value)):
            raise ValueError('Finite three-component SOC vector required')
        phases=[];scores=[]
        for name in ('triplet_tracking','singlet_tracking'):
            record=endpoint[name];phase=complex(record['phase']);score=float(record['score'])
            if (not record.get('source') or not np.isfinite(phase) or abs(abs(phase)-1)>1e-10
                    or not np.isfinite(score) or not minimum_overlap<=score<=1):
                raise ValueError('Unambiguous state-tracking evidence and unit phase required')
            phases.append(phase);scores.append(score)
        if endpoint.get('spin_frame')!='reference':
            raise ValueError('SOC components must be transported to the reference spin frame')
        return phases[0].conjugate()*phases[1]*value,min(scores)
    for pair in pairs:
        mode=int(pair['mode_index']);step=float(pair['step'])
        if mode in seen or not 0<=mode<mode_count or not np.isfinite(step) or step<=0:
            raise ValueError('Distinct valid modes and positive steps required')
        seen.add(mode);plus,sp=aligned(pair['plus']);minus,sm=aligned(pair['minus'])
        gradient[:,mode]=(plus-minus)/(2*step)
        even=(plus+minus)/2-reference
        rows.append(dict(mode_index=mode,minimum_tracking_overlap=min(sp,sm),
            even_residual_hartree=float(np.linalg.norm(even)),
            derivative_norm_hartree=float(np.linalg.norm(gradient[:,mode])),
            tracking={sign:{name:dict(pair[sign][name]) for name in
                ('singlet_tracking','triplet_tracking')} for sign in ('plus','minus')}))
    return dict(gradient_dimensionless=gradient,constant_hartree=reference,pairs=rows,
        phase_convention='Independently tracked ket phases in a common spin frame',
        scope='Signed coherent derivatives; finite-step convergence requires a separate check')


def rank_soc_promoting_modes(gradient, response_gram, *, relative_rate_tolerance=.01):
    """Rank modes using a supplied nuclear response, retaining interference.

    ``response_gram[j,k]`` is the common-model broadened nuclear response for
    unit Q_j/Q_k insertions at the requested energy, in the same normal basis.
    It must be Hermitian positive semidefinite. No QM points are launched.
    The omission estimate uses Cauchy--Schwarz and includes interference;
    derivative magnitudes alone never certify a negligible rate contribution.
    Channel/spin population weights must already be included in the gradient.
    This concerns the supplied linear operator. A nonzero constant coupling and
    its interference require an additional constant form in the response basis.
    """
    g=np.asarray(gradient,complex);r=np.asarray(response_gram,complex)
    if (g.ndim!=2 or r.shape!=(g.shape[1],g.shape[1])
            or not np.all(np.isfinite(g)) or not np.all(np.isfinite(r))
            or not np.isfinite(relative_rate_tolerance) or not 0<relative_rate_tolerance<1):
        raise ValueError('Finite gradient, compatible response and tolerance in (0,1) required')
    scale=max(float(np.max(abs(r),initial=0)),np.finfo(float).tiny)
    if np.max(abs(r-r.conj().T),initial=0)>1e-10*scale:
        raise ValueError('Hermitian nuclear response required')
    eigen=np.linalg.eigvalsh(r)
    if eigen.min(initial=0)<-1e-10*scale:
        raise ValueError('Positive semidefinite nuclear response required')
    # Roundoff-sized negative eigenvalues are accounted for in every bound.
    rounding=max(0.,-float(eigen.min(initial=0)))
    def response(a):return max(0.,float(np.einsum('aj,jk,ak->',a.conj(),r,a).real))
    total=response(g)
    diagonal=np.maximum(0.,r.diagonal().real)*np.sum(abs(g)**2,axis=0)
    order=np.argsort(-diagonal,kind='stable');retained=np.zeros_like(g);rows=[];selected=[]
    if total==0:
        return dict(total_response=0.,selected_modes=[],ranking=order.tolist(),
            status='zero_response_no_relative_certificate',history=[])
    for mode in order:
        selected.append(int(mode));retained[:,mode]=g[:,mode];omitted=g-retained
        a=response(retained)+rounding*float(np.vdot(retained,retained).real)
        b=response(omitted)+rounding*float(np.vdot(omitted,omitted).real)
        cross=abs(np.einsum('aj,jk,ak->',retained.conj(),r,omitted))
        round_cross=rounding*np.linalg.norm(retained)*np.linalg.norm(omitted)
        bound=b+2*(min(np.sqrt(a*b),float(cross))+round_cross)
        rows.append(dict(added_mode=int(mode),retained_response=response(retained),
            omission_absolute_bound=float(bound),omission_relative_bound=float(bound/total)))
        if bound<=relative_rate_tolerance*total:break
    return dict(total_response=total,selected_modes=selected,ranking=order.tolist(),history=rows,
        status='response_based_selection_complete',
        scope='Conditional on the supplied full nuclear response and electronic derivatives; no uncomputed mode is certified')


def symmetry_projected_soc_pairs(reference, pairs, *, spin_basis, mode_count):
    """Contract paired SOC vectors without differencing unrelated root phases.

    Each pair supplies a distinct one-dimensional spin channel, a mode index
    and a positive dimensionless step. Their projectors must be orthonormal.
    For a symmetry-forbidden constant, |H(+h)| and |H(-h)| give the magnitude
    of its odd derivative. Independent phases between orthogonal channels
    cannot affect an unpolarized rate. Multiple modes in the same channel
    require explicit electronic gauge transport and are rejected here.
    """
    reference = np.asarray(reference, complex)
    basis = np.asarray(spin_basis, complex)
    if reference.shape != (3,) or basis.shape != (3, 3) or not np.allclose(basis.conj().T@basis, np.eye(3), atol=1e-12):
        raise ValueError('Three finite SOC components and an orthonormal spin basis required')
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(basis)):
        raise ValueError('Nonfinite SOC reference or basis')
    gradient = np.zeros((3, mode_count), complex)
    lower = gradient.copy(); upper = gradient.copy(); rows = []; used = set(); modes = set()
    reference_channels = basis.conj().T@reference
    for pair in pairs:
        mode, channel, step = int(pair['mode_index']), int(pair['channel_index']), float(pair['step'])
        if not 0 <= mode < mode_count or not 0 <= channel < 3 or step <= 0 or not np.isfinite(step):
            raise ValueError('Explicit valid mode, spin channel and positive step required')
        if channel in used or mode in modes:
            raise ValueError('Repeated channels/modes require explicit coherent electronic gauge transport')
        used.add(channel); modes.add(mode)
        plus = np.asarray(pair['plus'], complex); minus = np.asarray(pair['minus'], complex)
        if plus.shape != (3,) or minus.shape != (3,) or not np.all(np.isfinite([plus, minus])):
            raise ValueError('Finite three-component paired SOC vectors required')
        ap, am = (basis.conj().T@v for v in (plus, minus))
        values = np.abs([ap[channel], am[channel]])/step
        gradient[channel, mode] = values.mean()
        lower[channel, mode] = values.min(); upper[channel, mode] = values.max()
        rows.append(dict(mode_index=mode, channel_index=channel, step=step,
            derivative_magnitude_hartree_per_dimensionless=float(values.mean()),
            paired_relative_difference=float(abs(values[0]-values[1])/max(values.mean(), np.finfo(float).tiny)),
            reference_projected_magnitude_hartree=float(abs(reference_channels[channel])),
            plus_projected_magnitude_hartree=float(abs(ap[channel])),
            minus_projected_magnitude_hartree=float(abs(am[channel])),
            plus_forbidden_norm_hartree=float(np.linalg.norm(np.delete(ap, channel))),
            minus_forbidden_norm_hartree=float(np.linalg.norm(np.delete(am, channel)))))
    return dict(gradient_dimensionless=gradient, endpoint_lower_gradient=lower,
        endpoint_upper_gradient=upper, pairs=rows, constant_by_exact_symmetry=np.zeros(3),
        phase_convention='Magnitudes in distinct symmetry-orthogonal spin channels; no same-channel interference is inferred',
        scope='Restricted promoting-mode linear operator; endpoint spread is a sensitivity diagnostic, not a rigorous error bound')
