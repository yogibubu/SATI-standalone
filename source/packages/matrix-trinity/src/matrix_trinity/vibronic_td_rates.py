"""Golden Rule adapters for the shared TRINITY quantum-dynamics engine.

Sources are already mapped into the final operator's declared coordinates.
This service neither changes a force field nor substitutes dipoles for SOCs.
"""
import numpy as np
from scipy.special import erfc

from .quantum_dynamics import QuantumDynamics, correlation_emission_spectrum, PHASE_PER_CM_FS
from .vibronic_rates import GOLDEN_RULE_FACTOR


def coherent_bath_correlation(dynamics, provider, nodes, weights, coupling_hartree, times_fs, *,
        source_coordinate_id, tolerance=1e-10, maximum_workspace_bytes=256*1024**2):
    """Integrate native vector propagation over supplied coherent bath nodes.

    Nodes/weights approximate the standard complex Gaussian measure
    pi**(-nb) exp(-|z|**2) dRe(z) dIm(z). They are neither selected nor
    renormalized here. In particular Monte Carlo weights are 1/sample_count.
    Propagation and source contraction reuse the existing common engines.
    Finite quadrature/sampling, time-grid and active-space errors are separate.
    """
    from .gaussian_partial_trace import GaussianCoherentBathPackets
    from .quantum_dynamics import _positive_integer
    if not isinstance(dynamics, QuantumDynamics) or not isinstance(provider, GaussianCoherentBathPackets):
        raise TypeError('existing QuantumDynamics and GaussianCoherentBathPackets required')
    nodes, weights = np.asarray(nodes, complex), np.asarray(weights, float)
    times, coupling = np.asarray(times_fs, float), np.asarray(coupling_hartree, complex)
    if (source_coordinate_id != dynamics.coordinate_id
            or dynamics.hamiltonian.shape != (provider.size, provider.size)):
        raise ValueError('coherent source and final Hamiltonian coordinate identities/spaces differ')
    if (nodes.ndim != 2 or nodes.shape[1] != len(provider.bath_frequencies_cm) or not len(nodes)
            or weights.shape != (len(nodes),) or np.any(weights < 0)
            or not np.all(np.isfinite(nodes)) or not np.all(np.isfinite(weights))
            or coupling.ndim != 1 or not len(coupling) or not np.all(np.isfinite(coupling))
            or times.ndim != 1 or not len(times) or times[0] != 0
            or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0)):
        raise ValueError('finite coherent nodes, nonnegative weights, channels and increasing times from zero required')
    _positive_integer(maximum_workspace_bytes, 'maximum_workspace_bytes')
    reserved = (provider.numeric_storage_bytes+provider.evaluation_workspace_bytes
                +nodes.nbytes+weights.nbytes+64*len(times)*(len(coupling)+1)
                +16*len(times)*len(nodes))
    per_node = 16*(provider.size*(len(times)+8)+4*provider.diagnostics['active_ho_states'])
    batch = min(len(nodes), (maximum_workspace_bytes-reserved)//per_node,
                (dynamics.maximum_workspace_bytes-64*len(times))//per_node)
    if batch < 1:
        raise MemoryError('coherent bath/native vector propagation exceeds numeric workspace budget')
    values = np.zeros(len(times), complex)
    node_values = np.empty((len(times), len(nodes)), complex)
    products, maximum_bound = 0, 0.
    for first in range(0, len(nodes), batch):
        last = min(first+batch, len(nodes))
        source = provider.evaluate_nodes(nodes[first:last])
        propagated = dynamics.propagate(source, times, tolerance=tolerance)
        products += propagated['diagnostics']['operator_vector_products']
        maximum_bound = max(maximum_bound, propagated['diagnostics']['relative_propagator_error_bound'])
        per_time = 16*(last-first)*(provider.diagnostics['active_ho_states']+3*provider.size)
        available = min(maximum_workspace_bytes-reserved-propagated['values'].nbytes,
            provider.maximum_workspace_bytes-provider.numeric_storage_bytes-provider.evaluation_workspace_bytes)
        time_batch = min(len(times), max(1, available//per_time))
        for ti in range(0, len(times), time_batch):
            end = min(ti+time_batch,len(times))
            bra = provider.evaluate_grid(nodes[first:last],times[ti:end])
            node_values[ti:end, first:last] = np.sum(bra.conj()*propagated['values'][ti:end], axis=1)
        values += node_values[:, first:last]@weights[first:last]
        del propagated
    energy = provider.initial_energy_cm-provider.bath_zero_point_cm
    values *= np.exp(1j*energy*PHASE_PER_CM_FS*times)
    node_values *= np.exp(1j*energy*PHASE_PER_CM_FS*times[:, None])
    diagnostics = dict(provider.diagnostics, coordinate_id=dynamics.coordinate_id,
        coherent_node_count=len(nodes), coherent_node_batch_size=int(batch),
        coherent_weights_normalized_by_engine=False, coherent_weight_sum=float(weights.sum()),
        operator_vector_products=products, maximum_node_relative_propagator_error_bound=maximum_bound,
        relative_propagator_error_bound=0. if maximum_bound == 0. else None,
        gaussian_time_tail_bound_available=False,
        quadrature_or_sampling_error_included=False, nodes_and_weights_owned_by_caller=True,
        active_pair_table_constructed=False, active_propagator_storage_entries=int(batch)*len(times)*provider.size,
        numeric_workspace_estimate_bytes=int(reserved+batch*per_node),
        bath_zero_point_cm=provider.bath_zero_point_cm,
        hamiltonian_partition='native active plus independent harmonic spectators',
        full_molecular_convergence_certified=False)
    return dict(schema='matrix.trinity.coherent_bath_correlation.v1', times_fs=times,
        values=values[:, None]*abs(coupling[None, :])**2,
        node_correlations=node_values, node_correlation_units='dimensionless',
        correlation_units='hartree_squared', diagnostics=diagnostics, provenance=provider.provenance)


def native_gaussian_bath_correlation(dynamics, kernel, coupling_hartree, times_fs, *,
        source_coordinate_id, tolerance=1e-10, maximum_workspace_bytes=256*1024**2):
    """Contract the exact analytic bath kernel with shared native dynamics.

    The active propagator may be a native tensor action or an already prepared
    diagonal representation with the matching kernel embedding. The source
    provider owns the basis/coordinate identity. All harmonic bath occupations
    are summed analytically, with no independent active/bath source
    factorization. The model Hamiltonian is H_active + H_bath; anharmonic
    potential couplings to bath modes are not inferred by this contraction.

    This deterministic reference stores the finite active propagators (only
    phases for the engine's existing diagonal backend), not a spectator packet
    inventory. A preflight rejects oversized time/active
    tables. It is not a claim of practical scaling for an arbitrary large
    active space; reduced modals and independent active-basis controls remain
    the caller's responsibility.
    """
    from .gaussian_partial_trace import GaussianBathKernel
    from .quantum_dynamics import _positive_integer
    if not isinstance(dynamics, QuantumDynamics) or not isinstance(kernel, GaussianBathKernel):
        raise TypeError('existing QuantumDynamics and GaussianBathKernel required')
    if source_coordinate_id != dynamics.coordinate_id:
        raise ValueError('analytic source and final Hamiltonian coordinate identities differ')
    times = np.asarray(times_fs, float)
    coupling = np.asarray(coupling_hartree, complex)
    if (dynamics.hamiltonian.shape != (kernel.size, kernel.size)
            or coupling.ndim != 1 or not len(coupling) or not np.all(np.isfinite(coupling))
            or times.ndim != 1 or not len(times) or times[0] != 0
            or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0)):
        raise ValueError('matching active space, finite coupling channels and time grid from zero required')
    _positive_integer(maximum_workspace_bytes, 'maximum_workspace_bytes')
    n = kernel.size
    diagonal = dynamics._diagonal_energies is not None
    columns = 1 if diagonal else n
    estimated = (kernel.numeric_storage_bytes+16*len(times)*n*columns+96*n*columns+64*len(times)
        +kernel.evaluation_workspace_bytes+32*len(times)*len(coupling))
    if estimated > maximum_workspace_bytes:
        raise MemoryError('analytic bath/native propagator exceeds numeric workspace budget')
    zero = kernel.evaluate(0.)
    scale = max(float(np.max(abs(zero))), np.finfo(float).tiny)
    strength = np.trace(zero)
    if (not np.allclose(zero, zero.conj().T, rtol=1e-10, atol=1e-12*scale)
            or abs(strength.imag) > 1e-10*scale or strength.real < 0):
        raise ArithmeticError('zero-time analytic bath kernel violates the source norm contract')
    sources = np.ones(n, complex) if diagonal else np.eye(n, dtype=complex)
    propagated = dynamics.propagate(sources, times, tolerance=tolerance)
    values = np.empty((len(times), len(coupling)), complex)
    energy = kernel.initial_energy_cm-kernel.bath_zero_point_cm
    for index, time in enumerate(times):
        reduced = zero if index == 0 else kernel.evaluate(time)
        value = np.sum((np.diag(reduced) if diagonal else reduced)*propagated['values'][index])
        values[index] = value*np.exp(1j*energy*PHASE_PER_CM_FS*time)*abs(coupling)**2
    diagnostics = dict(kernel.diagnostics)
    diagnostics.update(propagated['diagnostics'])
    diagnostics.update(spectator_occupation_sum_analytic=True,
        spectator_sampling_used=False, spectator_occupation_cutoff_used=False,
        active_spectator_factorization_assumed=False,
        projected_source_norm=float(strength.real),
        numeric_workspace_estimate_bytes=estimated,
        bath_zero_point_cm=kernel.bath_zero_point_cm,
        propagator_pair_table_constructed=propagated['diagnostics']['pair_table_constructed'],
        pair_table_constructed=(kernel.diagnostics['initial_active_pair_plan_constructed']
                                or propagated['diagnostics']['pair_table_constructed']),
        active_propagator_storage_entries=len(times)*n*columns,
        diagonal_phases_reused=diagonal,
        hamiltonian_partition='native active plus independent harmonic spectators',
        coupling_operator='constant incoherent nonradiative electronic channels',
        source_mapping_owned_by_caller=True, stores_active_propagator_table=True,
        propagation_bound_excludes_kernel_roundoff=True,
        full_molecular_convergence_certified=False)
    return dict(schema='matrix.trinity.native_gaussian_bath_correlation.v1',
        times_fs=times, values=values, correlation_units='hartree_squared',
        diagnostics=diagnostics, provenance=kernel.provenance)


def finite_initial_harmonic_correlation(model, initial, coupling_hartree, times_fs, *,
        kind='emission', initial_level=0, generator=None, maximum_integrals=500000):
    """Infinite harmonic final-state sum for a finite, possibly anharmonic source.

    This is the harmonic-final reference, not a replacement for a native ZAFF
    final Hamiltonian. It reuses the Raman Mehler contraction and complex
    Hermite recurrence, retaining the full Duschinsky map and all supplied
    initial coefficients. ``generator`` may be the existing thresholded kernel
    definition; the routine does not introduce a Gaussian matrix threshold.
    """
    from .vibronic_harmonic import orient_harmonic_model, harmonic_gaussian_generator
    from .vibronic_variational import FiniteVibrationalStates, gaussian_number_moment
    from .gaussian_partial_trace import gaussian_partial_number_trace
    from .gaussian_hermite import hermite_recurrence_plan, hermite_coefficients
    oriented, _ = orient_harmonic_model(model, kind)
    if (not isinstance(initial, FiniteVibrationalStates)
            or not np.array_equal(initial.frequencies_cm, oriented.initial_frequencies_cm)):
        raise ValueError('finite initial state must match the oriented emitting/absorbing surface')
    coupling = np.asarray(coupling_hartree, complex)
    times = np.asarray(times_fs, float)
    if (coupling.ndim != 1 or not len(coupling) or not np.all(np.isfinite(coupling))
            or times.ndim != 1 or not len(times) or times[0] != 0
            or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0)
            or isinstance(initial_level, bool) or not isinstance(initial_level, (int, np.integer))
            or not 0 <= initial_level < len(initial.energies_cm)
            or not isinstance(maximum_integrals, int) or maximum_integrals < 1):
        raise ValueError('finite coupling channels, increasing time grid from zero and valid source level required')
    n = len(initial.frequencies_cm)
    g = harmonic_gaussian_generator(oriented) if generator is None else generator
    if np.shape(g['matrix']) != (2*n, 2*n) or np.shape(g['linear']) != (2*n,):
        raise ValueError('supplied Gaussian generator must use the same two-surface coordinate dimensions')
    active = np.flatnonzero(initial.basis.max(axis=0))
    rows = initial.basis[:, active]
    count = len(rows)
    if count**2 > maximum_integrals:
        raise MemoryError('finite initial-state Hermite pair plan exceeds the declared budget')
    pairs = np.column_stack([np.repeat(rows, count, axis=0), np.tile(rows, (count, 1))])
    plan = hermite_recurrence_plan(pairs, 0, maximum_integrals)
    coefficients = initial.eigenvectors[:, initial_level]
    products = np.outer(coefficients.conj(), coefficients).ravel()
    # Initial coordinates that are vacuum in every supplied row have Bargmann
    # variables exactly zero. Select them out before the existing Gaussian
    # contraction, rather than constructing and discarding their kernel blocks.
    selected = np.r_[active, np.arange(n, 2*n)]
    matrix = g['matrix'][np.ix_(selected, selected)]
    linear = g['linear'][selected]
    contracted = np.arange(len(active), len(selected))
    values = np.empty((len(times), len(coupling)), complex)
    phase_energy = (initial.energies_cm[initial_level]-oriented.electronic_gap_cm
                    -.5*oriented.final_frequencies_cm.sum())
    for k, time in enumerate(times):
        tau = time*PHASE_PER_CM_FS
        traced = gaussian_partial_number_trace(matrix, linear, g['log_ground'],
            contracted, np.exp(-1j*oriented.final_frequencies_cm*tau))
        overlap = hermite_coefficients(traced['matrix'], traced['linear'], traced['constant'], plan)
        values[k] = (products@overlap)*np.exp(1j*phase_energy*tau)*abs(coupling)**2
    moment = gaussian_number_moment(g['matrix'], g['linear'], g['log_ground'], initial,
        initial_level=int(initial_level), maximum_grid_entries=maximum_integrals)
    expected = moment['moment']*abs(coupling)**2
    error = float(np.max(abs(values[0]-expected)))
    if error > 1e-10*max(float(np.max(expected)), np.finfo(float).tiny):
        raise ArithmeticError('finite-source correlation violates the existing Mehler norm sum')
    return dict(schema='matrix.trinity.finite_initial_harmonic_correlation.v1',
        times_fs=times, values=values, correlation_units='hartree_squared',
        diagnostics=dict(zero_time_error_hartree_squared=error,
            final_occupation_sum_analytic=True, final_state_cutoff_used=False,
            initial_coefficients_pruned=False, full_duschinsky_retained=True,
            final_model='harmonic reference', initial_states=count,
            hermite_recurrence_entries=len(plan[0]), spectator_sampling_used=False,
            full_molecular_convergence_certified=False), provenance=model.provenance)


def nonradiative_doorway_correlation(dynamics, source_hartree, times_fs, *,
        initial_energy_cm, source_coordinate_id, tolerance=1e-10, provenance=None):
    """Correlate V_fi|i> in the same finite coordinates as the final operator.

    Each column is an incoherent electronic coupling channel. The source
    provider owns the Duschinsky transformation and any finite-basis projection.
    The energy origin must be common to initial_energy_cm and the Hamiltonian.
    """
    if not isinstance(dynamics, QuantumDynamics):
        raise TypeError('existing QuantumDynamics engine required')
    if source_coordinate_id != dynamics.coordinate_id:
        raise ValueError('doorway and final Hamiltonian must use identical coordinate references')
    times = np.asarray(times_fs, float)
    if times.ndim != 1 or not len(times) or times[0] != 0:
        raise ValueError('time grid must start at zero')
    result = dynamics.correlate(source_hartree, times,
        initial_energy_cm=initial_energy_cm, tolerance=tolerance)
    result.update(schema='matrix.trinity.nonradiative_doorway_correlation.v1',
        correlation_units='hartree_squared', provenance={} if provenance is None else dict(provenance))
    result['diagnostics'].update(operator='supplied nonradiative electronic coupling',
        source_mapping_owned_by_caller=True, spectator_trace_inferred=False,
        full_molecular_convergence_certified=False)
    return result


def spectator_doorway_correlation(dynamics, source_hartree, times_fs, *,
        spectator_energies_cm, initial_energy_cm, source_coordinate_id,
        spectator_weights=None, tolerance=1e-10, provenance=None, reduce_spectators=True):
    """Contract correlated conditional packets with a diagonal harmonic bath.

    Column s is the *conditional* coupled active packet <s|V|i>, not a common
    packet times an independent bath probability. Columns label orthogonal
    spectator states (or weighted samples of them). Their energies include the
    spectator zero-point energy. The active Hamiltonian must exclude that
    energy. Initial_energy_cm includes the electronic energy offset in the
    same convention as the active Hamiltonian.

    The existing QuantumDynamics recurrence propagates all supplied columns;
    only scalar correlations are stored. Importance/quadrature weights remain
    external: this deterministic contraction neither selects states nor
    certifies statistical or omitted-domain errors.
    """
    packets = np.asarray(source_hartree, complex)
    energies = np.asarray(spectator_energies_cm, float)
    if packets.ndim != 2 or packets.shape[1] == 0:
        raise ValueError('one conditional source column per spectator state required')
    weights = (np.ones(packets.shape[1]) if spectator_weights is None
               else np.asarray(spectator_weights, float))
    if (energies.shape != (packets.shape[1],) or weights.shape != energies.shape
            or not np.all(np.isfinite(energies)) or not np.all(np.isfinite(weights))
            or np.any(weights < 0)):
        raise ValueError('finite spectator energies and nonnegative weights required')
    result = nonradiative_doorway_correlation(dynamics, packets, times_fs,
        initial_energy_cm=initial_energy_cm, source_coordinate_id=source_coordinate_id,
        tolerance=tolerance, provenance=provenance)
    phase = PHASE_PER_CM_FS*result['times_fs']
    # Reduce in place to avoid a second time-by-source complex workspace.
    for column, energy in enumerate(energies):
        result['values'][:, column] *= np.exp(-1j*phase*energy)
    if not isinstance(reduce_spectators, (bool, np.bool_)):
        raise ValueError('boolean spectator reduction choice required')
    if reduce_spectators:
        result['values'] = (result['values']@weights)[:, None]
    else:
        result['values'] *= weights[None, :]
    result['schema'] = 'matrix.trinity.spectator_doorway_correlation.v1'
    result['diagnostics'].update(conditional_source_count=len(energies),
        active_spectator_factorization_assumed=False,
        spectator_energies_include_zero_point=True,
        spectator_weights_normalized=False, spectator_selection_owned_by_caller=True,
        spectator_columns_reduced=bool(reduce_spectators),
        statistical_and_omitted_domain_errors_included=False)
    return result


def prepared_spectator_correlation(spectrum, source_hartree, times_fs, *,
        spectator_energies_cm, initial_energy_cm, source_coordinate_id,
        spectator_weights=None, maximum_workspace_bytes=256*1024**2, provenance=None,
        reduce_spectators=True):
    """Reuse an existing native eigenwindow, with an omitted-strength bound.

    Exact disjoint block spectra are supported without materializing a global
    eigenvector matrix. Missing eigenlevels remain missing; their nonnegative
    packet norm bounds the correlation error uniformly in time. This bound
    excludes eigensolver/orthogonality errors and floating-point roundoff.
    It says nothing about omitted spectator occupations or the native basis.
    """
    from scipy.sparse import diags
    from .vibronic_spectral_cache import PreparedGaussianSpectrum, PreparedBlockGaussianSpectrum
    if not isinstance(spectrum, (PreparedGaussianSpectrum, PreparedBlockGaussianSpectrum)):
        raise TypeError('existing prepared Gaussian or exact block spectrum required')
    if source_coordinate_id != spectrum.hamiltonian_sha256:
        raise ValueError('prepared spectrum and doorway Hamiltonian identities differ')
    packets = np.asarray(source_hartree, complex)
    parts = (spectrum.parts if isinstance(spectrum, PreparedBlockGaussianSpectrum)
             else ((slice(None), spectrum),))
    size = spectrum.basis_size if isinstance(spectrum, PreparedBlockGaussianSpectrum) else len(spectrum.eigenvectors)
    levels = sum(len(part.energies_cm) for _, part in parts)
    if (packets.ndim != 2 or packets.shape[0] != size or not packets.shape[1]
            or not np.all(np.isfinite(packets))
            or isinstance(maximum_workspace_bytes, bool)
            or not isinstance(maximum_workspace_bytes, int) or maximum_workspace_bytes < 1):
        raise ValueError('finite compatible conditional packets and positive workspace required')
    projected_bytes = levels*packets.shape[1]*16
    if 4*packets.nbytes+3*projected_bytes+8*levels > maximum_workspace_bytes:
        raise MemoryError('prepared doorway projection exceeds declared workspace budget')
    projected = np.empty((levels, packets.shape[1]), complex)
    energies = np.empty(levels)
    start = 0
    for rows, part in parts:
        stop = start+len(part.energies_cm)
        local = packets[rows]
        projected[start:stop] = ((part.eigenvectors.T@local.conj()).conj()
                                 if np.iscomplexobj(part.eigenvectors) else
                                 part.eigenvectors.T@local)
        energies[start:stop] = part.energies_cm
        start = stop
    norms = np.sum(abs(packets)**2, axis=0)
    kept = np.sum(abs(projected)**2, axis=0)
    missing = norms-kept
    if np.any(missing < -1e-10*np.maximum(norms, np.finfo(float).tiny)):
        raise ArithmeticError('prepared eigenvectors violate the conditional source norm')
    # Small negative subtraction errors are reported; no spectral density or
    # physical rate is clipped. Positive missing strength is never discarded.
    negative_roundoff = np.minimum(missing, 0.)
    missing = np.maximum(missing, 0.)
    weights = (np.ones(packets.shape[1]) if spectator_weights is None
               else np.asarray(spectator_weights, float))
    engine = QuantumDynamics(diags(energies, format='dia'),
        energy_bounds_cm=(float(energies.min()), float(energies.max())),
        coordinate_id=source_coordinate_id,
        maximum_workspace_bytes=maximum_workspace_bytes-packets.nbytes-projected.nbytes)
    result = spectator_doorway_correlation(engine, projected, times_fs,
        spectator_energies_cm=spectator_energies_cm, initial_energy_cm=initial_energy_cm,
        source_coordinate_id=source_coordinate_id, spectator_weights=weights, provenance=provenance,
        reduce_spectators=reduce_spectators)
    result['diagnostics'].update(prepared_spectrum_reused=True,
        projected_levels=levels, full_native_basis_size=size,
        omitted_spectral_norm_hartree_squared=float(missing@weights),
        spectral_norm_negative_roundoff_hartree_squared=float(negative_roundoff@weights),
        weighted_full_packet_norm_hartree_squared=float(norms@weights),
        omitted_spectral_norm_by_source_hartree_squared=missing.tolist(),
        spectral_omitted_lower_bound_cm=spectrum.omitted_lower_bound_cm,
        spectator_target_energies_cm=(initial_energy_cm-np.asarray(spectator_energies_cm)).tolist(),
        spectator_weights=weights.tolist(),
        eigenvector_matrix_materialized=False,
        omission_bound_excludes_eigensolver_and_roundoff=True)
    return result


def prepared_spectator_filtered_rate(spectrum, source_hartree, times_fs, *,
        target_energies_cm, source_coordinate_id, gaussian_sigma_cm,
        spectator_weights=None, maximum_workspace_bytes=256*1024**2):
    """Positive TD filter for conditional packets in a prepared finite spectrum.

    Reuses diagonal QuantumDynamics and the coherent filter's accumulation
    kernel for packets with fixed spectator number. Both time branches are
    integrated before taking the squared norm. Source columns already include
    their electronic couplings; weights are supplied without renormalization.
    Fine/coarse/short controls concern the same finite projected sources.
    """
    from scipy.sparse import diags
    from .vibronic_spectral_cache import PreparedGaussianSpectrum,gaussian_omitted_tail_profile
    from .vibronic_rates import _accumulate_coherent_serial
    if not isinstance(spectrum,PreparedGaussianSpectrum):
        raise TypeError('existing prepared finite Gaussian spectrum required')
    source=np.asarray(source_hartree,complex);times=np.asarray(times_fs,float)
    targets=np.asarray(target_energies_cm,float);sigma=float(gaussian_sigma_cm)
    if (source_coordinate_id!=spectrum.hamiltonian_sha256
            or source.ndim!=2 or source.shape[0]!=len(spectrum.eigenvectors) or not source.shape[1]
            or not np.all(np.isfinite(source)) or targets.shape!=(source.shape[1],)
            or not np.all(np.isfinite(targets)) or not np.isfinite(sigma) or sigma<=0
            or times.ndim!=1 or len(times)<5 or (len(times)-1)%4 or times[0]!=0
            or not np.all(np.isfinite(times)) or times[1]<=0
            or not np.allclose(np.diff(times),times[1],rtol=1e-10,atol=0)
            or isinstance(maximum_workspace_bytes,bool) or not isinstance(maximum_workspace_bytes,int)
            or maximum_workspace_bytes<1):
        raise ValueError('matching finite packets/targets, identity, positive linewidth and uniform 4m+1 times required')
    weights=np.ones(source.shape[1]) if spectator_weights is None else np.asarray(spectator_weights,float)
    if weights.shape!=targets.shape or not np.all(np.isfinite(weights)) or np.any(weights<0):
        raise ValueError('one finite nonnegative weight per conditional packet required')
    n=len(spectrum.energies_cm);columns=source.shape[1]
    reserved=4*source.nbytes+16*n*columns*8+96*len(times)+64*n
    batch=min(len(times),256,(maximum_workspace_bytes-reserved)//(96*n+128*columns+64))
    if batch<1:raise MemoryError('positive prepared filter exceeds numeric workspace budget')
    vectors=spectrum.eigenvectors
    projected=((vectors.T@source.conj()).conj() if np.iscomplexobj(vectors) else vectors.T@source)
    norms=np.sum(abs(source)**2,axis=0);kept=np.sum(abs(projected)**2,axis=0)
    missing=norms-kept
    if np.any(missing < -1e-10*np.maximum(norms,np.finfo(float).tiny)):
        raise ArithmeticError('prepared eigenvectors violate conditional source norm')
    missing=np.maximum(missing,0.)
    engine=QuantumDynamics(diags(spectrum.energies_cm,format='dia'),
        energy_bounds_cm=(float(spectrum.energies_cm[0]),float(spectrum.energies_cm[-1])),
        coordinate_id=source_coordinate_id,maximum_workspace_bytes=maximum_workspace_bytes)
    tau=times*PHASE_PER_CM_FS;step=tau[1]
    prefactor=np.sqrt(1/(sigma*np.sqrt(2*np.pi)))*sigma/np.sqrt(np.pi)
    fine=np.full(len(times),step);fine[[0,-1]]*=.5
    coarse=np.zeros(len(times));coarse[::2]=2*step;coarse[[0,-1]]*=.5
    end=3*(len(times)-1)//4
    short=np.zeros(len(times));short[:end+1]=step;short[[0,end]]*=.5
    rules=np.stack([fine,coarse,short])*(prefactor*np.exp(-(sigma*tau)**2))
    filtered=np.zeros((3,n,columns),complex)
    for first in range(0,len(times),batch):
        last=min(first+batch,len(times))
        phase=engine.propagate(np.ones(n,complex),times[first:last])['values']
        for column,target in enumerate(targets):
            packet=np.broadcast_to(projected[None,:,column,None],(last-first,n,1))
            carrier=np.exp(1j*target*tau[first:last])
            for sign in (1,-1):
                filtered[:,:,column:column+1]+=_accumulate_coherent_serial(
                    phase if sign==1 else phase.conj(),packet,
                    carrier if sign==1 else carrier.conj(),rules[:,first:last])
    rates=GOLDEN_RULE_FACTOR*np.sum(abs(filtered)**2,axis=1)
    estimates=rates@weights
    if not np.all(np.isfinite(estimates)):raise FloatingPointError('positive prepared rate exceeds numerical range')
    profile=gaussian_omitted_tail_profile(spectrum.omitted_lower_bound_cm,targets,sigma)
    return dict(schema='matrix.trinity.prepared_spectator_filtered_rate.v1',
        rate_s_1=float(estimates[0]),rates_by_source_s_1=rates[0],
        coarse_grid_rate_s_1=float(estimates[1]),shorter_window_rate_s_1=float(estimates[2]),
        diagnostics=dict(gaussian_filter_squared_after_integration=True,
            negative_time_branch_included=True,prepared_spectrum_reused=True,
            conditional_source_columns=columns,weights_renormalized=False,
            time_step_fs=float(times[1]),full_time_fs=float(times[-1]),shorter_time_fs=float(times[end]),
            time_batch_size=int(batch),numeric_workspace_estimate_bytes=int(reserved+batch*(96*n+128*columns+64)),
            omitted_spectral_gaussian_tail_bound_s_1=GOLDEN_RULE_FACTOR*float((missing*profile)@weights),
            temporal_controls_are_empirical=True,bounds_exclude_roundoff=True,
            full_molecular_convergence_certified=False))


def nonradiative_rate_from_correlation(correlation, *, gaussian_sigma_cm,
        channel_weights=None, fft_workers=1):
    """Reuse the shared FFT strength transform at zero energy for FGR.

    Only the unweighted strength density is used; the radiative helper's photon
    frequency factors never enter this rate. Gaussian tails and propagation
    errors are bounded separately. Aliasing, time discretization, source-space
    truncation and physical model convergence require independent controls.
    """
    if correlation.get('correlation_units') != 'hartree_squared':
        raise ValueError('explicit Hartree-squared nonradiative correlation required')
    spectrum = correlation_emission_spectrum(correlation,
        gaussian_sigma_cm=gaussian_sigma_cm, fft_workers=fft_workers)
    density = spectrum['strength_density_au2_per_cm']
    weights = np.ones(density.shape[1]) if channel_weights is None else np.asarray(channel_weights, float)
    if (weights.shape != (density.shape[1],) or not np.all(np.isfinite(weights))
            or np.any(weights < 0)):
        raise ValueError('one finite nonnegative weight per incoherent channel required')
    zero = np.flatnonzero(spectrum['wavenumbers_cm'] == 0)
    if len(zero) != 1:
        raise ArithmeticError('shared Fourier grid must contain exactly one zero-energy point')
    value = float(density[zero[0]]@weights)
    if value < 0:
        raise FloatingPointError('negative energy-conserving density; refine time grid/window')
    values = np.asarray(correlation['values'])
    if values.ndim == 1:
        values = values[:, None]
    strength = float(values[0].real@weights)
    if (np.max(abs(values[0].imag)) > 1e-12*max(strength, np.finfo(float).tiny)
            or strength < 0):
        raise ValueError('zero-time coupling norm must be real and nonnegative')
    peak = 1/(np.sqrt(2*np.pi)*gaussian_sigma_cm)
    end = correlation['times_fs'][-1]*PHASE_PER_CM_FS
    tail = (strength*peak*erfc(gaussian_sigma_cm*end/np.sqrt(2))
            if correlation['diagnostics'].get('gaussian_time_tail_bound_available', True) else None)
    propagation = correlation['diagnostics'].get('relative_propagator_error_bound')
    propagation_bound = None if propagation is None else float(strength*peak*propagation)
    spectral_bound = None
    omitted_strength_bound = None
    if 'omitted_spectral_norm_by_source_hartree_squared' in correlation['diagnostics']:
        from .vibronic_spectral_cache import gaussian_omitted_tail_profile
        d = correlation['diagnostics']
        profile = gaussian_omitted_tail_profile(d['spectral_omitted_lower_bound_cm'],
            d['spectator_target_energies_cm'], gaussian_sigma_cm)
        source_weights = np.asarray(d['spectator_weights'])*(weights[0] if
            d['spectator_columns_reduced'] else weights)
        omitted = np.asarray(d['omitted_spectral_norm_by_source_hartree_squared'])*source_weights
        spectral_bound = GOLDEN_RULE_FACTOR*float(omitted@profile)
        omitted_strength_bound = GOLDEN_RULE_FACTOR*peak*float(omitted.sum())
    return dict(schema='matrix.trinity.nonradiative_correlation_rate.v1',
        rate_s_1=GOLDEN_RULE_FACTOR*value, coupling_density_hartree2_per_cm=value,
        rates_by_channel_s_1=GOLDEN_RULE_FACTOR*density[zero[0]].copy(),
        gaussian_sigma_cm=gaussian_sigma_cm, channel_weights=weights,
        diagnostics=dict(shared_transform=spectrum['diagnostics'],
            gaussian_time_tail_bound_s_1=None if tail is None else float(GOLDEN_RULE_FACTOR*tail),
            propagation_bound_s_1=None if propagation_bound is None else GOLDEN_RULE_FACTOR*propagation_bound,
            omitted_spectral_strength_bound_s_1=omitted_strength_bound,
            omitted_spectral_gaussian_tail_bound_s_1=spectral_bound,
            bounds_exclude_roundoff=True, frequency_weighting_applied=False,
            zero_energy_sampling='exact shared FFT bin',
            time_grid_aliasing_and_source_convergence_required=True,
            full_molecular_convergence_certified=False),
        provenance=correlation.get('provenance', {}))
