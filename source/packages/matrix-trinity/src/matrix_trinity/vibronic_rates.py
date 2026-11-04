"""Vacuum Einstein rates, harmonic golden-rule rates and electronic SOC mixing.

Coupling coefficients for nonradiative transitions are energies (Hartree),
not dipoles. Momentum is p=-i d/dx in the initial dimensionless normal modes.
The input operator must include all terms of the chosen nonadiabatic model;
no automatic conversion of an incomplete derivative-coupling vector is implied.
"""
from __future__ import annotations
from dataclasses import replace
import numpy as np
from numba import njit,prange
from scipy.constants import alpha, physical_constants
from scipy.special import voigt_profile
from matrix_gf.harmonic import CM_PER_HARTREE
from .vibronic_ti import harmonic_vibronic_ti
from .vibronic_td import harmonic_vibronic_td
from .vibronic_vpt2 import vpt2_vibronic_ti

ATOMIC_TIME_S=physical_constants['atomic unit of time'][0]
EINSTEIN_A_FACTOR=4*alpha**3/(3*ATOMIC_TIME_S*CM_PER_HARTREE**3)
GOLDEN_RULE_FACTOR=2*np.pi*CM_PER_HARTREE/ATOMIC_TIME_S


def _harmonic_ti_lines(result):
    if result.get('line_format')!='columns':return result['lines']
    columns=result['line_columns']
    if not columns:return []
    return [dict(initial_quanta=i.tolist(),final_quanta=f.tolist(),wavenumber_cm=float(nu),
        weighted_strength_au2=float(strength)) for i,f,nu,strength in zip(columns['initial_quanta'],
        columns['final_quanta'],columns['wavenumber_cm'],columns['weighted_strength_au2'])]


def harmonic_radiative_emission_rate(model,*,temperature_K=0.,**ti_controls):
    """Sum vacuum spontaneous-emission Einstein A over positive-energy TI lines.

    This is a thermally averaged rate per emitting molecule, not an oscillator
    strength or a normalized spectral shape. No spin average is implicit.
    Truncation diagnostics qualify dipole-strength completeness, not convergence
    of its energy-cubed moment. No broadened-spectrum cubic moment is used.
    """
    result=harmonic_vibronic_ti(model,temperature_K=temperature_K,kind='emission',**ti_controls)
    lines=_harmonic_ti_lines(result);rates=[];excluded=0
    for line in lines:
        nu=line['wavenumber_cm']
        if nu<=0:excluded+=1;continue
        rates.append(dict(initial_quanta=line['initial_quanta'],final_quanta=line['final_quanta'],
            wavenumber_cm=nu,rate_s_1=float(EINSTEIN_A_FACTOR*nu**3*line['weighted_strength_au2'])))
    if ti_controls.get('backend')=='cpp':
        from .vibronic_native_kernels import NativeKernels
        moment,_,_=NativeKernels(str(ti_controls['native_executable'])+'.kernels.so').rate_moments(
            [row['wavenumber_cm'] for row in lines],[row['weighted_strength_au2'] for row in lines],1.,0.)
        rate=float(EINSTEIN_A_FACTOR*moment)
    else:rate=float(sum(row['rate_s_1'] for row in rates))
    return dict(schema='matrix.trinity.harmonic_radiative_rate.v1',rate_s_1=rate,
        radiative_lifetime_s=None if rate==0 else 1/rate,lines=rates,
        diagnostics=dict(TI=result['diagnostics'],excluded_nonpositive_photon_lines=excluded,
            energy_cubed_moment_convergence_requires_larger_progression=True,
            spin_population_average_applied=False,environment='vacuum electric-dipole spontaneous emission'),
        provenance=model.provenance)


def harmonic_nonradiative_rate(model,coupling_hartree,*,gradient_hartree=None,
        momentum_hartree=None,channel_weights=None,temperature_K=0.,kind='emission',
        method='TD',gaussian_sigma_cm=50.,lorentz_hwhm_cm=0.,**controls):
    """Golden rule for explicit constant + linear position/momentum couplings.

    Three orthogonal channels may represent SOC spin components or independently
    incoherent final channels. Coherent pathways to the same channel must be
    summed in the complex input amplitudes first. Explicit channel weights may
    describe spin populations; they are never inferred or normalized.

    A linewidth is mandatory for an isolated discrete harmonic spectrum to
    define a finite energy-conserving rate. It is a physical/model input, not
    a predicted bath or density-of-states parameter. TI evaluates a Voigt delta
    directly; TD samples the corresponding Fourier density at zero energy.
    """
    n=len(model.initial_frequencies_cm)
    constant=np.asarray(coupling_hartree,complex)
    gradient=np.zeros((3,n),complex) if gradient_hartree is None else np.asarray(gradient_hartree,complex)
    momentum=np.zeros((3,n),complex) if momentum_hartree is None else np.asarray(momentum_hartree,complex)
    weights=np.ones(3) if channel_weights is None else np.asarray(channel_weights,float)
    if constant.shape!=(3,) or gradient.shape!=(3,n) or momentum.shape!=(3,n) or weights.shape!=(3,) or not all(np.all(np.isfinite(x)) for x in (constant,gradient,momentum,weights)) or np.any(weights<0):
        raise ValueError('three finite coupling channels and nonnegative explicit channel weights required')
    if not np.isfinite(gaussian_sigma_cm+lorentz_hwhm_cm) or min(gaussian_sigma_cm,lorentz_hwhm_cm)<0 or gaussian_sigma_cm+lorentz_hwhm_cm<=0:
        raise ValueError('nonzero finite nonnegative linewidths required for a discrete harmonic rate')
    scale=np.sqrt(weights)
    coupling_model=replace(model,transition_dipole_au=constant*scale,
        transition_gradient_dimensionless_au=gradient*scale[:,None],
        transition_momentum_dimensionless_au=momentum*scale[:,None])
    if method=='TI':
        result=harmonic_vibronic_ti(coupling_model,temperature_K=temperature_K,kind=kind,**controls)
        lines=_harmonic_ti_lines(result)
        if controls.get('backend')=='cpp':
            from .vibronic_native_kernels import NativeKernels
            _,density,peak=NativeKernels(str(controls['native_executable'])+'.kernels.so').rate_moments(
                [row['wavenumber_cm'] for row in lines],
                [row['weighted_strength_au2'] for row in lines],gaussian_sigma_cm,lorentz_hwhm_cm)
        else:
            density=float(sum(row['weighted_strength_au2']*voigt_profile(row['wavenumber_cm'],gaussian_sigma_cm,lorentz_hwhm_cm) for row in lines))
            peak=float(voigt_profile(0.,gaussian_sigma_cm,lorentz_hwhm_cm))
        diagnostics=dict(TI=result['diagnostics'],energy_conservation_sampling='direct Voigt sum',
            missing_rate_upper_bound_s_1=float(GOLDEN_RULE_FACTOR*peak*max(0.,result['diagnostics']['strength_sum_rule']['total']-result['diagnostics']['reported_strength_au2'])))
    elif method=='TD':
        result=harmonic_vibronic_td(coupling_model,temperature_K=temperature_K,kind=kind,
            gaussian_sigma_cm=gaussian_sigma_cm,lorentz_hwhm_cm=lorentz_hwhm_cm,**controls)
        axis=result['wavenumbers_cm']
        if not axis[0]<=0<=axis[-1]:raise ValueError('zero energy is outside the TD Nyquist interval; decrease time_step_fs')
        density=float(np.interp(0.,axis,result['density_au2_per_cm']))
        # Interpolation is explicitly qualified; grid refinement remains necessary.
        width=max(gaussian_sigma_cm,lorentz_hwhm_cm)
        diagnostics=dict(TD=result['diagnostics'],energy_conservation_sampling='linear interpolation at zero energy',
            grid_spacing_to_linewidth=float(result['diagnostics']['frequency_step_cm']/width),
            time_step_window_and_linewidth_convergence_required=True)
        if density<0:raise FloatingPointError('negative energy-conserving TD density; refine time step/window before using a rate')
    else:raise ValueError('method must be TI or TD')
    return dict(schema='matrix.trinity.harmonic_nonradiative_rate.v1',method=method,
        rate_s_1=float(GOLDEN_RULE_FACTOR*density),coupling_density_hartree2_per_cm=density,
        gaussian_sigma_cm=gaussian_sigma_cm,lorentz_hwhm_cm=lorentz_hwhm_cm,
        diagnostics=diagnostics,channel_weights=weights.tolist(),
        scope='harmonic broadened golden-rule rate for the supplied weak-coupling operator; not a bath model or strong-coupling population dynamics',provenance=model.provenance)


def sam_vpt2_nonradiative_rate(model,initial_field,final_field,coupling_hartree,*,
        gradient_hartree=None,momentum_hartree=None,channel_weights=None,
        gaussian_sigma_cm=50.,lorentz_hwhm_cm=0.,kind='emission',operator='FCHT',**ti_controls):
    """Broadened FGR sum over explicitly requested nonresonant SAM VPT2 states.

    Couplings are electronic nonadiabatic matrix elements in Hartree, not
    electric dipoles. Three channels are incoherent orthogonal channels; paths
    within a channel must be combined coherently by the caller. Initial-state
    populations and target states follow :func:`vpt2_vibronic_ti` and are not
    inferred from a truncated partition function. The rate is a selected-state
    contribution unless the caller independently establishes target coverage.

    The nonnegative rate uses the square of the order-two corrected amplitude,
    so it contains terms above strict mechanical order two. The signed,
    strictly order-two strength remains available in the nested TI result.
    A Voigt width defines the broadened energy-conservation delta; it is an
    explicit model input, not a predicted bath property.
    """
    n=len(model.initial_frequencies_cm)
    constant=np.asarray(coupling_hartree,complex)
    gradient=np.zeros((3,n),complex) if gradient_hartree is None else np.asarray(gradient_hartree,complex)
    momentum=np.zeros((3,n),complex) if momentum_hartree is None else np.asarray(momentum_hartree,complex)
    weights=np.ones(3) if channel_weights is None else np.asarray(channel_weights,float)
    if (constant.shape!=(3,) or gradient.shape!=(3,n) or momentum.shape!=(3,n)
            or weights.shape!=(3,) or not all(np.all(np.isfinite(x)) for x in (constant,gradient,momentum,weights))
            or np.any(weights<0)):
        raise ValueError('three finite coupling channels and nonnegative explicit channel weights required')
    if (not np.isfinite(gaussian_sigma_cm) or not np.isfinite(lorentz_hwhm_cm)
            or min(gaussian_sigma_cm,lorentz_hwhm_cm)<0
            or gaussian_sigma_cm+lorentz_hwhm_cm<=0):
        raise ValueError('nonzero finite nonnegative linewidths required for a discrete rate')
    if kind not in {'absorption','emission'}:
        raise ValueError('kind must be absorption or emission')
    scale=np.sqrt(weights)
    coupling_model=replace(model,transition_dipole_au=constant*scale,
        transition_gradient_dimensionless_au=gradient*scale[:,None],
        transition_momentum_dimensionless_au=momentum*scale[:,None])
    transitions=vpt2_vibronic_ti(coupling_model,initial_field,final_field,
        kind=kind,operator=operator,**ti_controls)
    profile=voigt_profile(np.asarray(transitions['wavenumbers_cm'],float),
                          gaussian_sigma_cm,lorentz_hwhm_cm)
    weighted_strength=np.asarray(transitions['weighted_positive_completed_strengths_au2'],float)
    with np.errstate(over='raise',invalid='raise'):
        contributions=GOLDEN_RULE_FACTOR*weighted_strength*profile
        rate=float(np.sum(contributions))
    if not np.all(np.isfinite(contributions)) or np.any(contributions<0):
        raise FloatingPointError('SAM VPT2 Golden Rule contributions must be finite and nonnegative')
    if not np.isfinite(rate):
        raise FloatingPointError('SAM VPT2 Golden Rule total exceeds numerical range')
    lines=[]
    for i,contribution in enumerate(contributions):
        lines.append(dict(
            initial_quanta=transitions['initial_quanta'][i].copy(),
            final_quanta=transitions['final_quanta'][i].copy(),
            population=float(transitions['population'][i]),
            energy_mismatch_cm=float(transitions['wavenumbers_cm'][i]),
            positive_completed_strength_au2=float(transitions['positive_completed_strengths_au2'][i]),
            weighted_strength_au2=float(weighted_strength[i]),
            voigt_density_per_cm=float(profile[i]),
            rate_contribution_s_1=float(contribution)))
    return dict(schema='matrix.sati.sam_vpt2_nonradiative_rate.v1',method='SAM VPT2 explicit-state TI',
        rate_s_1=rate,coupling_density_hartree2_per_cm=float(np.sum(weighted_strength*profile)),
        gaussian_sigma_cm=float(gaussian_sigma_cm),lorentz_hwhm_cm=float(lorentz_hwhm_cm),
        channel_weights=weights.tolist(),kind=kind,lines=lines,transitions=transitions,
        diagnostics=dict(target_space_complete=False,selected_transition_count=len(lines),
            rate_strength_convention='squared order-two corrected amplitudes; includes higher mechanical orders',
            strict_order_two_strengths_may_be_negative=True,
            missing_rate_upper_bound_s_1=None,
            broadening_is_physical_input=True,
            scope='selected-state broadened nonadiabatic Golden Rule contribution; no complete-rate or bath claim',
            vpt2=transitions['diagnostics']),provenance=model.provenance)


def _accumulate_coherent_filter(phase,packets,carrier,rules):
    output=np.zeros((3,packets.shape[1],packets.shape[2]),np.complex128)
    for level in prange(packets.shape[1]):
        # Small node batches benefit from register-local sums. Larger batches
        # retain the time-major loop for contiguous packet reads.
        if packets.shape[2] <= 4:
            for node in range(packets.shape[2]):
                fine=0j;coarse=0j;shortened=0j
                for ti in range(packets.shape[0]):
                    value=phase[ti,level]*carrier[ti]*packets[ti,level,node]
                    fine+=rules[0,ti]*value
                    coarse+=rules[1,ti]*value
                    shortened+=rules[2,ti]*value
                output[0,level,node]=fine
                output[1,level,node]=coarse
                output[2,level,node]=shortened
            continue
        for ti in range(packets.shape[0]):
            rotation=phase[ti,level]*carrier[ti]
            for node in range(packets.shape[2]):
                value=rotation*packets[ti,level,node]
                output[0,level,node]+=rules[0,ti]*value
                output[1,level,node]+=rules[1,ti]*value
                output[2,level,node]+=rules[2,ti]*value
    return output


_accumulate_coherent_serial=njit(cache=False,nogil=True)(_accumulate_coherent_filter)
_accumulate_coherent_parallel=njit(cache=False,nogil=True,parallel=True)(_accumulate_coherent_filter)


def coherent_bath_filtered_rate(dynamics,provider,nodes,weights,coupling_hartree,times_fs,*,
        source_coordinate_id,gaussian_sigma_cm,maximum_workspace_bytes=256*1024**2):
    """Positive coherent estimator: square the Gaussian-filtered amplitude.

    The existing diagonal dynamics must represent the supplied active spectral
    embedding. Nodes/weights integrate the standard complex Gaussian measure;
    their selection and convergence belong to the caller. Both signs of time
    are integrated. Shared source contraction and propagation are unchanged.
    No finite-time, quadrature, omitted-spectrum or full-molecule bound is
    inferred. Step doubling and a shorter window reuse the same evaluations.
    """
    from .gaussian_partial_trace import GaussianCoherentBathPackets
    from .quantum_dynamics import QuantumDynamics,PHASE_PER_CM_FS,_positive_integer
    from threading import current_thread,main_thread
    if (not isinstance(dynamics,QuantumDynamics) or dynamics._diagonal_energies is None
            or not isinstance(provider,GaussianCoherentBathPackets)):
        raise TypeError('existing diagonal QuantumDynamics and coherent packet provider required')
    nodes,weights=np.asarray(nodes,complex),np.asarray(weights,float)
    times,coupling=np.asarray(times_fs,float),np.asarray(coupling_hartree,complex)
    if (source_coordinate_id!=dynamics.coordinate_id
            or dynamics.hamiltonian.shape!=(provider.size,provider.size)):
        raise ValueError('coherent source and active spectral identities/spaces differ')
    if (nodes.ndim!=2 or nodes.shape[1]!=len(provider.bath_frequencies_cm) or not len(nodes)
            or weights.shape!=(len(nodes),) or not np.all(np.isfinite(nodes))
            or not np.all(np.isfinite(weights)) or np.any(weights<0)
            or coupling.ndim!=1 or not len(coupling) or not np.all(np.isfinite(coupling))
            or times.ndim!=1 or len(times)<5 or times[0]!=0 or (len(times)-1)%4
            or not np.all(np.isfinite(times)) or times[1]<=0
            or not np.allclose(np.diff(times),times[1],rtol=1e-10,atol=0)
            or not np.isfinite(gaussian_sigma_cm) or gaussian_sigma_cm<=0):
        raise ValueError('Gaussian nodes/weights, finite coupling, positive linewidth and uniform 4m+1 times required')
    _positive_integer(maximum_workspace_bytes,'maximum_workspace_bytes')
    n,ho=provider.size,provider.diagnostics['active_ho_states']
    reserved=(provider.numeric_storage_bytes+provider.evaluation_workspace_bytes
              +nodes.nbytes+weights.nbytes+64*len(times)+112*n*len(nodes))
    available=min(maximum_workspace_bytes-reserved,
        provider.maximum_workspace_bytes-provider.numeric_storage_bytes-provider.evaluation_workspace_bytes,
        dynamics.maximum_workspace_bytes)
    per_node=16*(2*ho+8*n)
    node_batch=min(len(nodes),available//per_node)
    if node_batch<1:
        raise MemoryError('positive coherent filter exceeds numeric workspace budget')
    time_batch=min(len(times),max(1,available//(node_batch*per_node)))
    tau=times*PHASE_PER_CM_FS
    step=tau[1]
    sigma=gaussian_sigma_cm
    prefactor=np.sqrt(1/(sigma*np.sqrt(2*np.pi)))*sigma/np.sqrt(np.pi)
    envelope=prefactor*np.exp(-(sigma*tau)**2)
    quadrature=np.full(len(times),step);quadrature[[0,-1]]*=.5
    coarse=np.zeros(len(times));coarse[::2]=2*step;coarse[[0,-1]]*=.5
    end=3*(len(times)-1)//4
    shortened=np.zeros(len(times));shortened[:end+1]=step;shortened[[0,end]]*=.5
    rules=np.stack([quadrature,coarse,shortened])*envelope
    energy=provider.initial_energy_cm-provider.bath_zero_point_cm
    filtered=np.zeros((3,n,len(nodes)),complex)
    accumulate=(_accumulate_coherent_parallel if n>=32 and current_thread() is main_thread()
                else _accumulate_coherent_serial)
    for first in range(0,len(nodes),node_batch):
        last=min(first+node_batch,len(nodes))
        for start in range(0,len(times),time_batch):
            stop=min(start+time_batch,len(times))
            phase=dynamics.propagate(np.ones(n,complex),times[start:stop])['values']
            carrier=np.exp(1j*energy*tau[start:stop])
            for sign in (1,-1):
                packets=provider.evaluate_grid(nodes[first:last],-sign*times[start:stop])
                spectral_phase=phase if sign==1 else phase.conj()
                electronic_phase=carrier if sign==1 else carrier.conj()
                filtered[:,:,first:last]+=accumulate(spectral_phase,packets,electronic_phase,
                                                      rules[:,start:stop])
                del packets
    node_rates=GOLDEN_RULE_FACTOR*sum(abs(coupling)**2)*np.sum(abs(filtered)**2,axis=1)
    estimates=node_rates@weights
    if not np.all(np.isfinite(estimates)):
        raise FloatingPointError('positive coherent rate exceeds numerical range')
    return dict(schema='matrix.trinity.coherent_gaussian_filtered_rate.v1',
        rate_s_1=float(estimates[0]),node_rates_s_1=node_rates[0],
        coarse_grid_rate_s_1=float(estimates[1]),shorter_window_rate_s_1=float(estimates[2]),
        coarse_grid_node_rates_s_1=node_rates[1],shorter_window_node_rates_s_1=node_rates[2],
        gaussian_sigma_cm=float(sigma),provenance=provider.provenance,
        diagnostics=dict(provider.diagnostics,coherent_node_count=len(nodes),
            coherent_node_batch_size=int(node_batch),time_batch_size=int(time_batch),
            full_time_fs=float(times[-1]),shorter_time_fs=float(times[end]),time_step_fs=float(times[1]),
            negative_time_branch_included=True,gaussian_filter_squared_after_integration=True,
            temporal_controls_fused=True,
            coherent_weights_normalized_by_engine=False,coherent_weight_sum=float(weights.sum()),
            active_pair_table_constructed=False,operator_vector_products=0,
            numeric_workspace_estimate_bytes=int(reserved+time_batch*node_batch*per_node),
            time_quadrature_and_window_controls_are_empirical=True,
            sampling_error_included=False,omitted_spectral_error_included=False,
            full_molecular_convergence_certified=False))


def analytic_bath_nonradiative_rate(dynamics,kernel,coupling_hartree,times_fs,*,
        source_coordinate_id,gaussian_sigma_cm,channel_weights=None,
        tolerance=1e-10,fft_workers=1,maximum_workspace_bytes=256*1024**2):
    """FGR with the shared coupled Gaussian bath and native active dynamics.

    Bath occupations are summed analytically; the finite active basis and any
    supplied contraction still require independent convergence controls.
    """
    from .vibronic_td_rates import native_gaussian_bath_correlation,nonradiative_rate_from_correlation
    correlation=native_gaussian_bath_correlation(dynamics,kernel,coupling_hartree,times_fs,
        source_coordinate_id=source_coordinate_id,tolerance=tolerance,
        maximum_workspace_bytes=maximum_workspace_bytes)
    result=nonradiative_rate_from_correlation(correlation,gaussian_sigma_cm=gaussian_sigma_cm,
        channel_weights=channel_weights,fft_workers=fft_workers)
    return dict(result,correlation=correlation,
        scope='analytic harmonic bath and supplied finite native active space; active-basis convergence not inferred')


def time_dependent_nonradiative_rate(dynamics,source_hartree,times_fs,*,
        initial_energy_cm,source_coordinate_id,gaussian_sigma_cm,
        channel_weights=None,tolerance=1e-10,fft_workers=1,provenance=None,
        spectator_energies_cm=None,spectator_weights=None):
    """FGR facade over the common quantum propagator and strength transform.

    The supplied doorway is already in the final Hamiltonian's coordinates.
    Its provider owns Duschinsky mapping, spectator integration and source-space
    convergence. This adapter does not infer those operations or change ZAFF.
    """
    from .vibronic_td_rates import (nonradiative_doorway_correlation,
        nonradiative_rate_from_correlation,spectator_doorway_correlation)
    if spectator_energies_cm is None:
        if spectator_weights is not None:
            raise ValueError('spectator energies required with spectator weights')
        correlation=nonradiative_doorway_correlation(dynamics,source_hartree,times_fs,
            initial_energy_cm=initial_energy_cm,source_coordinate_id=source_coordinate_id,
            tolerance=tolerance,provenance=provenance)
    else:
        correlation=spectator_doorway_correlation(dynamics,source_hartree,times_fs,
            spectator_energies_cm=spectator_energies_cm,spectator_weights=spectator_weights,
            initial_energy_cm=initial_energy_cm,source_coordinate_id=source_coordinate_id,
            tolerance=tolerance,provenance=provenance)
    result=nonradiative_rate_from_correlation(correlation,gaussian_sigma_cm=gaussian_sigma_cm,
        channel_weights=channel_weights,fft_workers=fft_workers)
    return dict(result,correlation=correlation,
        scope='time-dependent FGR on the supplied common finite source/operator space; no inferred bath or source completeness')


def spin_orbit_mixed_electronic_states(energies_hartree,spin_orbit_hartree,dipole_au):
    """Diagonalize a supplied electronic SOC manifold, retaining complex phases.

    Triplet sublevels are explicit electronic basis states. Electric-dipole
    matrices are Hermitian, shape (3,states,states). All coherent intermediate
    pathways are retained by a single unitary transformation. No energy-
    denominator regularization or empirical spin mixing is applied. Derivatives
    of SOC and of the mixing eigenvectors require separate supplied fields.
    """
    energy=np.asarray(energies_hartree,float);soc=np.asarray(spin_orbit_hartree,complex)
    mu=np.asarray(dipole_au,complex);n=len(energy)
    if energy.ndim!=1 or n<2 or soc.shape!=(n,n) or mu.shape!=(3,n,n) or not all(np.all(np.isfinite(x)) for x in (energy,soc,mu)) or not np.allclose(soc,soc.conj().T,rtol=1e-12,atol=1e-14) or not np.allclose(mu,mu.conj().swapaxes(-1,-2),rtol=1e-12,atol=1e-14):
        raise ValueError('finite energies, Hermitian SOC and three Hermitian dipole matrices required')
    values,vectors=np.linalg.eigh(np.diag(energy)+soc)
    for column in range(n):
        pivot=np.argmax(abs(vectors[:,column]));vectors[:,column]*=np.exp(-1j*np.angle(vectors[pivot,column]))
    mixed=np.einsum('pi,apq,qj->aij',vectors.conj(),mu,vectors,optimize=True)
    return dict(schema='matrix.trinity.spin_orbit_electronic_states.v1',energies_hartree=values,
        coefficients=vectors,parent_state_weights=abs(vectors)**2,dipole_au=mixed,
        scope='electronic SOC mixing at one geometry; choose emitting eigenstates and populations explicitly; vibronic derivatives of the mixing are not computed')


__all__=['harmonic_radiative_emission_rate','harmonic_nonradiative_rate',
    'sam_vpt2_nonradiative_rate','time_dependent_nonradiative_rate',
    'analytic_bath_nonradiative_rate','coherent_bath_filtered_rate','spin_orbit_mixed_electronic_states']
