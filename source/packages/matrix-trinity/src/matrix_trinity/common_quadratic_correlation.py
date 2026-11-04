"""Exact harmonic limit of common-mode VCI, directly from both F/G operators.

Only a single Hamiltonian is spectrally factorized at a time. No intersurface
mode map or Duschinsky overlap transformation is constructed. The resident
Gaussian number-trace owner evaluates the infinite quadratic-space limit.
"""
from dataclasses import replace
import numpy as np
from scipy.sparse.csgraph import connected_components

from .common_quadratic_modes import common_quadratic_coefficients
from .gaussian_partial_trace import (gaussian_vacuum_trace_batch,
    gaussian_vacuum_operator_trace_batch)
from .quantum_dynamics import PHASE_PER_CM_FS


def _positive_root(matrix):
    eigen,vectors=np.linalg.eigh(matrix)
    if np.any(eigen<=0):
        raise ValueError('positive quadratic potential and kinetic matrices required')
    return (vectors*np.sqrt(eigen))@vectors.T


def _oscillators(a,b):
    root=_positive_root(b)
    eigen,vectors=np.linalg.eigh(root@a@root)
    if np.any(eigen<=0):
        raise ValueError('retained quadratic model is unstable')
    omega=np.sqrt(eigen)
    return omega,(root@vectors)/np.sqrt(omega)


def _quadratic_vacuum_source(ai, bi, af, bf, li, lf):
    """Normalized initial vacuum and coordinate insertions in the final trace."""
    n = len(li)
    di = -np.linalg.solve(ai, li); df = -np.linalg.solve(af, lf)
    wi, initial_transform = _oscillators(ai, bi)
    wf, final_transform = _oscillators(af, bf)
    inverse = np.linalg.solve(initial_transform, np.eye(n))
    precision = inverse.T@inverse
    w = final_transform.T@precision@final_transform
    displacement = np.linalg.solve(final_transform, di-df)
    summed = np.eye(n)+w; force = w@displacement
    solved = np.linalg.solve(summed, np.column_stack([np.eye(n), force]))
    generator = 2*solved[:, :n]-np.eye(n)
    log_ground = (n*np.log(2)/2+.25*np.linalg.slogdet(w)[1]
        -.5*np.linalg.slogdet(summed)[1]-.5*displacement@force+.5*force@solved[:, -1])
    return dict(matrix=(generator+generator.T)/2, linear=np.sqrt(2)*solved[:, -1],
        log_ground=float(log_ground), frequencies_cm=wf,
        initial_energy_cm=.5*wi.sum()+.5*li@di,
        final_energy_cm=.5*wf.sum()+.5*lf@df,
        coordinate_constant=df+final_transform@solved[:, -1],
        coordinate_linear=np.sqrt(2)*final_transform@solved[:, :n],
        initial_center=di, initial_transform=initial_transform)


def retain_common_quadratic_pairs(states, pairs):
    """Mask mixed F and G pairs, preserving expansion points and gradients.

    Each requested pair applies to both surfaces and both q-q/p-p operators.
    Diagonals, equilibrium coordinates and electronic reference energies stay
    fixed; linear/scalar translated coefficients are recomputed by their owner.
    """
    states=tuple(states);n=len(states[0].reference_esalc)
    mask=np.eye(n,dtype=bool);seen=set()
    for pair in pairs:
        if (len(pair)!=2 or any(isinstance(i,(bool,np.bool_)) or not isinstance(i,(int,np.integer)) for i in pair)
                or not 0<=pair[0]<pair[1]<n or tuple(pair) in seen):
            raise ValueError('distinct ordered common-mode pairs required')
        seen.add(tuple(pair));i,j=pair;mask[i,j]=mask[j,i]=True
    return tuple(replace(s,force_constants_hartree=np.where(mask,s.force_constants_hartree,0.),
        metric_inverse_amu=np.where(mask,s.metric_inverse_amu,0.)) for s in states)


def common_quadratic_correlation(initial, final, reference_frequencies_cm, times_fs, *,
        electronic_gap_cm, numerical_zero_cm=0., chunk_size=128,
        maximum_workspace_bytes=256*1024**2):
    """T=0 nuclear correlation for two complete common-coordinate quadratic H.

    electronic_gap_cm = E_final_reference - E_initial_reference. Return phase
    exp[+i(E_initial - E_final - gap)t], compatible with the shared FGR owner.
    All retained quadratic couplings are treated exactly, without perturbative
    denominators or a finite oscillator occupation cutoff. This qualifies the
    harmonic Hamiltonian, not the accuracy of a later finite dressed VCI.
    """
    times=np.asarray(times_fs,float)
    if (initial.coordinate_identity!=final.coordinate_identity or times.ndim!=1
            or not len(times) or times[0]!=0 or not np.all(np.isfinite(times))
            or np.any(np.diff(times)<0) or not np.isfinite(electronic_gap_cm)
            or not np.isfinite(numerical_zero_cm) or numerical_zero_cm<0):
        raise ValueError('one identified common chart, times from zero and finite gap/threshold required')
    coefficients=[common_quadratic_coefficients(s,reference_frequencies_cm) for s in (initial,final)]
    matrices=[np.array(c[:2]) for c in coefficients]
    n=len(reference_frequencies_cm);adjacency=np.zeros((n,n),bool)
    omitted=[]
    for state_matrices in matrices:
        for matrix in state_matrices:
            off=~np.eye(n,dtype=bool)
            small=off&(abs(matrix)<=numerical_zero_cm)
            omitted.extend(abs(matrix[small]))
            matrix[small]=0.
            adjacency|=off&(matrix!=0)
    count,labels=connected_components(adjacency,directed=False)
    correlation=np.ones(len(times),complex);tau=PHASE_PER_CM_FS*times
    group_sizes=[]
    total_initial=float(coefficients[0][3]);total_final=float(coefficients[1][3])
    # If numerical zeros are removed, preserve each relaxed center: use exact
    # coefficients by default. Nonzero-threshold qualification is explicit.
    for label in range(count):
        ids=np.flatnonzero(labels==label);group_sizes.append(len(ids))
        ai,bi=(m[np.ix_(ids,ids)] for m in matrices[0])
        af,bf=(m[np.ix_(ids,ids)] for m in matrices[1])
        li,lf=(c[2][ids] for c in coefficients)
        source=_quadratic_vacuum_source(ai,bi,af,bf,li,lf)
        trace=gaussian_vacuum_trace_batch(source['matrix'],source['linear'],source['log_ground'],
            np.exp(-1j*tau[:,None]*source['frequencies_cm']),chunk_size=chunk_size,
            maximum_workspace_bytes=maximum_workspace_bytes)
        ei=source['initial_energy_cm'];ef=source['final_energy_cm']
        total_initial+=ei;total_final+=ef
        correlation*=trace*np.exp(1j*(ei-ef)*tau)
    correlation*=np.exp(1j*(coefficients[0][3]-coefficients[1][3]-electronic_gap_cm)*tau)
    if abs(correlation[0]-1)>1e-9 or np.max(abs(correlation))>1+1e-8:
        raise ArithmeticError('common quadratic correlation violates vacuum normalization/unitarity')
    return dict(times_fs=times,correlation=correlation,diagnostics=dict(
        Duschinsky_overlap_used=False,intersurface_mode_map_constructed=False,
        quadratic_treatment='exact Gaussian limit of common-mode Hamiltonians',
        finite_occupation_cutoff_used=False,connected_quadratic_block_sizes=group_sizes,
        numerical_zero_cm=float(numerical_zero_cm),largest_removed_matrix_element_cm=float(max(omitted,default=0.)),
        initial_ground_energy_cm=total_initial,final_ground_energy_cm=total_final,
        relative_propagator_error_bound=0.,gaussian_time_tail_bound_available=True))


def common_quadratic_operator_correlation(initial, final, reference_frequencies_cm,
        times_fs, *, electronic_gap_cm, transition_dipole_au,
        transition_gradient_au, transition_reference=None, kind='absorption',
        chunk_size=64, maximum_workspace_bytes=256*1024**2):
    """Exact T=0 common-coordinate FC/HT correlation with a linear dipole.

    Coordinates, property reference and gradient all belong to the supplied
    common state chart, before dimensionless oscillator scaling. ``initial``
    is the populated surface; gap is final minus initial reference energy.
    Emission uses positive photon-energy Fourier phases. No intersurface mode
    map, finite occupation cutoff or finite-difference property is constructed.
    This is the infinite quadratic limit, not a finite dressed-VCI benchmark.
    """
    times=np.asarray(times_fs,float);n=len(reference_frequencies_cm)
    mu=np.asarray(transition_dipole_au,complex)
    gradient=np.asarray(transition_gradient_au,complex)
    reference=np.zeros(n) if transition_reference is None else np.asarray(transition_reference,float)
    if (initial.coordinate_identity!=final.coordinate_identity or times.ndim!=1
            or not len(times) or times[0]!=0 or not np.all(np.isfinite(times))
            or np.any(np.diff(times)<0) or not np.isfinite(electronic_gap_cm)
            or kind not in ('absorption','emission') or mu.shape!=(3,)
            or gradient.shape!=(3,n) or reference.shape!=(n,)
            or not all(np.all(np.isfinite(a)) for a in (mu,gradient,reference))):
        raise ValueError('matched common states, finite property/chart and times from zero required')
    ci=common_quadratic_coefficients(initial,reference_frequencies_cm)
    cf=common_quadratic_coefficients(final,reference_frequencies_cm)
    source=_quadratic_vacuum_source(*ci[:2],*cf[:2],ci[2],cf[2])
    length=ci[-1]
    ht_constant=gradient@(length*source['coordinate_constant']-reference)
    ht_linear=gradient@(length[:,None]*source['coordinate_linear'])
    constants=np.r_[mu,ht_constant]
    insertions=np.vstack([np.zeros((3,n)),ht_linear])
    tau=(1 if kind=='absorption' else -1)*PHASE_PER_CM_FS*times
    contracted=gaussian_vacuum_operator_trace_batch(source['matrix'],source['linear'],
        source['log_ground'],np.exp(-1j*tau[:,None]*source['frequencies_cm']),
        constants,insertions,chunk_size=chunk_size,maximum_workspace_bytes=maximum_workspace_bytes)
    phase=np.exp(1j*(source['initial_energy_cm']+ci[3]
        -source['final_energy_cm']-cf[3]-electronic_gap_cm)*tau)
    kernel=contracted['operator_kernel']*phase[:,None,None]
    components=dict(FC=sum(kernel[:,a,a] for a in range(3)),
        HT=sum(kernel[:,a,a] for a in range(3,6)),
        interference=sum(kernel[:,a,a+3]+kernel[:,a+3,a] for a in range(3)))
    center=length*source['initial_center']-reference
    covariance=.5*(length[:,None]*source['initial_transform'])@(length[:,None]*source['initial_transform']).T
    ht_mean=gradient@center
    rule=dict(FC=float(np.vdot(mu,mu).real),
        HT=float(np.vdot(ht_mean,ht_mean).real+np.einsum('ai,ij,aj->',gradient,covariance,gradient.conj()).real),
        interference=float(2*np.vdot(mu,ht_mean).real))
    error=max(abs(components[key][0]-value) for key,value in rule.items())
    if error>1e-9*max(sum(rule.values()),np.finfo(float).tiny):
        raise ArithmeticError('common FC/HT correlation violates the independent zero-time property moment')
    return dict(times_fs=times,correlation=sum(components.values()),components=components,
        diagnostics=dict(Duschinsky_overlap_used=False,intersurface_mode_map_constructed=False,
            finite_occupation_cutoff_used=False,quadratic_treatment='exact Gaussian common-Hamiltonian limit',
            kind=kind,zero_time_sum_rule=rule,zero_time_error=float(error),
            initial_ground_energy_cm=float(source['initial_energy_cm']+ci[3]),
            final_ground_energy_cm=float(source['final_energy_cm']+cf[3]),
            signed_interference_preserved=True,finite_vci_convergence_certified=False))


def screen_common_quadratic_pairs(state, reference_frequencies_cm, *,
        excitation_ceiling_cm, maximum_total_quanta=None, frequency_tolerance_cm=1., mixing_weight_tolerance=.01,
        numerical_zero_cm=1e-6):
    """State-specific conservative quadratic screening over a local-HO domain.

    The resident two-level/polyad owner makes each resonance decision. Here
    the two q-q/p-p channels are converted to exchange and pair-creation
    couplings, including their largest occupation factors in the declared
    excitation domain. Mode-pair screening is not a proof of convergence of
    FGR or of an entire many-state polyad; that requires the correlation test.
    Virtual pair-created states may lie above the source-domain ceiling.
    """
    from .local_quadratic import build_local_quadratic_model
    if not np.isfinite(excitation_ceiling_cm) or excitation_ceiling_cm<0:
        raise ValueError('finite nonnegative source excitation ceiling required')
    build_local_quadratic_model([0.,1.],{},{},frequency_tolerance_cm=frequency_tolerance_cm,
        mixing_weight_tolerance=mixing_weight_tolerance,numerical_zero_cm=numerical_zero_cm)
    if maximum_total_quanta is not None and (isinstance(maximum_total_quanta,bool) or not isinstance(maximum_total_quanta,int) or maximum_total_quanta<0):
        raise ValueError('nonnegative integer occupation ceiling required')
    a,b,*_=common_quadratic_coefficients(state,reference_frequencies_cm)
    omega=np.sqrt(np.diag(a)*np.diag(b));scale=(np.diag(b)/np.diag(a))**.25
    rows=[]
    for i in range(len(omega)):
        for j in range(i+1,len(omega)):
            q=a[i,j]*scale[i]*scale[j];p=b[i,j]/(scale[i]*scale[j])
            if max(abs(q),abs(p))<=numerical_zero_cm:
                continue
            cap=int(excitation_ceiling_cm/omega[i])
            if maximum_total_quanta is not None:cap=min(cap,maximum_total_quanta)
            ni=np.arange(cap+1)
            nj=np.floor((excitation_ceiling_cm-ni*omega[i])/omega[j]+1e-12).astype(int)
            if maximum_total_quanta is not None:nj=np.minimum(nj,maximum_total_quanta-ni)
            created=np.sqrt((ni+1)*(nj+1));exchanged=np.sqrt((ni+1)*nj)
            channels=[]
            for name,value,gap,factors in (('exchange',.5*(q+p),abs(omega[i]-omega[j]),exchanged),
                                          ('pair_creation',.5*(q-p),omega[i]+omega[j],created)):
                maximum=float(max(factors,default=0));coupling=value*maximum
                if abs(coupling)<=numerical_zero_cm:
                    continue
                model=build_local_quadratic_model([0.,gap],{},{(0,1):coupling},
                    frequency_tolerance_cm=frequency_tolerance_cm,
                    mixing_weight_tolerance=mixing_weight_tolerance,numerical_zero_cm=numerical_zero_cm)
                edge=model.selection[0]
                pt2_shift=None if gap<=model.denominator_tolerance_cm else coupling**2/gap
                channels.append(dict(channel=name,maximum_occupation_factor=maximum,
                    coupling_cm=float(coupling),gap_cm=float(gap),
                    two_level_shift_cm=edge.two_level_shift_cm,mixing_weight=edge.mixing_weight,
                    pt2_shift_cm=pt2_shift,variational=edge.variational))
            if channels:
                rows.append(dict(modes=(i,j),q_q_cm=float(q),p_p_cm=float(p),channels=channels,
                    variational=any(c['variational'] for c in channels),
                    ranking_score=max(c['two_level_shift_cm'] for c in channels)))
    rows.sort(key=lambda row:row['ranking_score'],reverse=True)
    return dict(pairs=rows,variational_pairs=[row['modes'] for row in rows if row['variational']],
        perturbative_pairs=[row['modes'] for row in rows if not row['variational']],
        diagnostics=dict(excitation_ceiling_cm=float(excitation_ceiling_cm),maximum_total_quanta=maximum_total_quanta,
            frequency_tolerance_cm=frequency_tolerance_cm,mixing_weight_tolerance=mixing_weight_tolerance,
            numerical_zero_cm=numerical_zero_cm,occupation_domain='independent local oscillator excitation sum',
            state_specific_screening=True,global_observable_convergence_certified=False))


def common_quadratic_dressed_correlation(initial, final, reference_frequencies_cm, times_fs, *,
        variational_pairs_initial, variational_pairs_final, electronic_gap_cm,
        parameter_step=.0625, optical_properties=None, **controls):
    """Order-two response in residual K around separately retained exact K.

    This is the harmonic benchmark of the two-state dressing: both normalized
    source and accepting propagator depend on the same response parameter.
    It includes their cross terms, rather than changing energies alone.
    The exact quadratic owner evaluates symmetric parameter points; comparing
    h and h/2 is mandatory to qualify numerical response derivatives. This
    analytic-Gaussian limit does not replace finite dressed-VCI polyads.
    """
    if not np.isfinite(parameter_step) or not 0<parameter_step<.25:
        raise ValueError('explicit small positive quadratic-response parameter step required')
    originals=(initial,final)
    references=tuple(retain_common_quadratic_pairs((state,),pairs)[0] for state,pairs in
                     zip(originals,(variational_pairs_initial,variational_pairs_final)))
    def evaluate(parameter):
        states=tuple(replace(reference,
            force_constants_hartree=reference.force_constants_hartree+parameter*(state.force_constants_hartree-reference.force_constants_hartree),
            metric_inverse_amu=reference.metric_inverse_amu+parameter*(state.metric_inverse_amu-reference.metric_inverse_amu))
            for state,reference in zip(originals,references))
        if optical_properties is None:
            return common_quadratic_correlation(*states,reference_frequencies_cm,times_fs,
                electronic_gap_cm=electronic_gap_cm,**controls)
        return common_quadratic_operator_correlation(*states,reference_frequencies_cm,times_fs,
            electronic_gap_cm=electronic_gap_cm,**optical_properties,**controls)
    zero=evaluate(0.);plus=evaluate(parameter_step);minus=evaluate(-parameter_step)
    c0=zero['correlation'];c1=(plus['correlation']-minus['correlation'])/(2*parameter_step)
    c2=(plus['correlation']-2*c0+minus['correlation'])/(2*parameter_step**2)
    energy_orders={}
    for name in ('initial_ground_energy_cm','final_ground_energy_cm'):
        e0=zero['diagnostics'][name];ep=plus['diagnostics'][name];em=minus['diagnostics'][name]
        energy_orders[name]=(e0,(ep-em)/(2*parameter_step),(ep-2*e0+em)/(2*parameter_step**2))
    result=dict(times_fs=zero['times_fs'],correlation=c0+c1+c2,correlation_orders=(c0,c1,c2),
        ground_energy_orders_cm=energy_orders,
        diagnostics=dict(zero['diagnostics'],quadratic_treatment='exact retained terms plus order-two residual response',
            response_parameter_step=parameter_step,numerical_response_step_converged=False,
            initial_variational_pairs=list(variational_pairs_initial),final_variational_pairs=list(variational_pairs_final),
            state_specific_quadratic_partition=True,source_and_acceptor_response_both_included=True,
            mixed_response_terms_included=True,gaussian_time_tail_bound_available=False))
    if optical_properties is not None:
        result['component_orders']={}
        result['components']={}
        for name,value in zero['components'].items():
            first=(plus['components'][name]-minus['components'][name])/(2*parameter_step)
            second=(plus['components'][name]-2*value+minus['components'][name])/(2*parameter_step**2)
            result['component_orders'][name]=(value,first,second)
            result['components'][name]=value+first+second
        result['diagnostics']['optical_FC_HT_response_included']=True
    return result
