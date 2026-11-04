"""Vibronic lines and FGR from the resident two-surface common-mode VCI.

The identical finite line list feeds TI and the existing TD Fourier owner.
No Duschinsky overlaps, occupation sampler, or new line-broadening engine.
"""
import numpy as np
from scipy.special import voigt_profile

from .common_mode_vci import CommonVCIStates, CommonVCIProperty, common_vci_transition_orders
from .quantum_dynamics import PHASE_PER_CM_FS
from .vibronic_rates import GOLDEN_RULE_FACTOR, EINSTEIN_A_FACTOR
from .vibronic_td_rates import nonradiative_rate_from_correlation


def common_vci_lines(initial, final, *, electronic_gap_cm,
                     initial_populations=None, strength_model='strict_order_two', property_operator=None):
    """Finite normalized nuclear lines, with final-minus-initial electronic gap.

    Populations refer to the supplied initial levels, defaulting to their
    lowest state. No population or intensity renormalization is performed.
    ``completed`` strengths retain additional orders of the dressed amplitude;
    they are never labelled as strict PT2. Negative truncated strengths are
    reported rather than silently clipped.
    """
    if not isinstance(initial, CommonVCIStates) or not isinstance(final, CommonVCIStates):
        raise TypeError('two common-mode VCI eigensystems required')
    if not np.isfinite(electronic_gap_cm):
        raise ValueError('finite final-minus-initial electronic gap required')
    orders = common_vci_transition_orders(initial, final, property_operator)
    if strength_model not in ('strict_order_two', 'completed'):
        raise ValueError('strength model must be strict_order_two or completed')
    strengths = orders['strengths' if strength_model == 'strict_order_two' else 'completed_strengths']
    populations = np.zeros(len(initial.energies_cm)) if initial_populations is None else np.asarray(initial_populations, float)
    if initial_populations is None:
        populations[0] = 1.
    if (populations.shape != initial.energies_cm.shape or np.any(populations < 0)
            or not np.all(np.isfinite(populations)) or not np.isclose(populations.sum(), 1., atol=1e-12, rtol=0.)):
        raise ValueError('explicit finite initial populations must be nonnegative and sum to one')
    gap = initial.energies_cm[None, :] - final.energies_cm[:, None] - electronic_gap_cm
    weights = strengths * populations[None, :]
    return dict(wavenumbers_cm=gap, strengths=strengths, population_weighted_strengths=weights,
        initial_populations=populations, amplitude_orders=orders['amplitude_orders'],
        strength_orders=orders['strength_orders'], diagnostics=dict(
            common_basis_id=initial.reference_id, Duschinsky_overlap_used=False,
            strength_model=strength_model, intensity_renormalized=False,
            finite_strength_sum=float(weights.sum()),
            negative_strength_count=int(np.count_nonzero(weights < 0)),
            finite_basis_convergence_certified=False))



def common_vci_operator_lines(initial, final, property_operators, forms, *,
        electronic_gap_cm, kind='absorption', initial_populations=None,
        strength_model='strict_order_two'):
    """Contract a shared FC/HT bank with finite common-mode wavefunctions.

    Forms use the resident optical convention O_a W_ab O_b*. They may be
    signed (for interference) and complex Hermitian. Nuclear contractions
    scale with the small polynomial bank, not the electronic-channel count.
    Absorption energies are final minus initial; emission reverses that sign.
    This evaluates supplied wavefunctions, not an independent TD propagation.
    """
    operators = tuple(property_operators)
    w = np.asarray(forms, complex)
    if (not operators or any(not isinstance(op, CommonVCIProperty) for op in operators)
            or w.ndim != 3 or w.shape[1:] != (len(operators), len(operators))
            or not len(w) or not np.all(np.isfinite(w))
            or not np.allclose(w, w.conj().swapaxes(1, 2), atol=1e-12, rtol=1e-12)):
        raise ValueError('identified property bank and finite Hermitian forms required')
    if kind not in ('absorption', 'emission'):
        raise ValueError('kind must be absorption or emission')
    lines = common_vci_lines(initial, final, electronic_gap_cm=electronic_gap_cm,
        initial_populations=initial_populations, strength_model=strength_model,
        property_operator=operators[0])
    banks = [lines['amplitude_orders']] + [common_vci_transition_orders(
        initial, final, op)['amplitude_orders'] for op in operators[1:]]
    amplitudes = np.stack(banks)  # operator, perturbation order, final, initial
    orders = np.stack([sum(np.einsum('afi,jab,bfi->jfi', amplitudes[:, r], w,
        amplitudes[:, k-r].conj(), optimize=True).real for r in range(k+1))
        for k in range(3)])
    completed = amplitudes.sum(axis=1)
    strengths = (orders.sum(axis=0) if strength_model == 'strict_order_two' else
        np.einsum('afi,jab,bfi->jfi', completed, w, completed.conj(), optimize=True).real)
    lines.update(schema='matrix.trinity.common_vci_operator_lines.v1', kind=kind,
        wavenumbers_cm=lines['wavenumbers_cm']*(-1 if kind == 'absorption' else 1),
        amplitude_orders=amplitudes, strength_orders=orders, strengths=strengths,
        population_weighted_strengths=strengths*lines['initial_populations'][None, None, :])
    lines['diagnostics'].update(polynomial_operator_count=len(operators),
        observable_count=len(w), signed_observables_preserved=True,
        finite_strength_sums=lines['population_weighted_strengths'].sum(axis=(1, 2)).tolist(),
        negative_strength_count=int(np.count_nonzero(strengths < 0)))
    # A bank has one sum per observable, rather than the first operator's sum.
    lines['diagnostics'].pop('finite_strength_sum')
    return lines


def _channel_weights(raw, count):
    if raw is not None and np.iscomplexobj(raw):
        raise ValueError('real nonnegative channel weights required')
    weights = np.ones(count) if raw is None else np.asarray(raw, float)
    if weights.shape != (count,) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError('one finite nonnegative explicit weight per incoherent channel required')
    return weights


def _operator_channel_lines(initial, final, *, electronic_gap_cm, operators,
                            channel_weights=None, initial_populations=None,
                            strength_model='strict_order_two'):
    operators = tuple(operators)
    if not operators or any(not isinstance(op, CommonVCIProperty) for op in operators):
        raise ValueError('explicit complete CommonVCIProperty transition operators required')
    weights = _channel_weights(channel_weights, len(operators))
    channels = [common_vci_lines(initial, final, electronic_gap_cm=electronic_gap_cm,
        initial_populations=initial_populations, strength_model=strength_model,
        property_operator=op) for op in operators]
    result = dict(channels[0]);result['diagnostics'] = dict(result['diagnostics'])
    for key in ('strengths', 'population_weighted_strengths'):
        result[key] = sum(weight*channel[key] for weight,channel in zip(weights,channels))
    # Separate channels are incoherent. Their complex amplitudes cannot be
    # summed as one path; retain each channel's orders explicitly instead.
    result.pop('amplitude_orders');result.pop('strength_orders')
    result['channels'] = channels
    result['channel_weights'] = weights
    result['diagnostics'].update(incoherent_channel_count=len(channels),
        channel_weights_normalized=False,
        finite_strength_sum=float(result['population_weighted_strengths'].sum()),
        negative_strength_count=int(np.count_nonzero(result['population_weighted_strengths']<0)))
    return result


def common_vci_radiative_emission_rate(initial, final, *, electronic_gap_cm,
        dipole_operators_au, initial_populations=None, strength_model='strict_order_two'):
    """Vacuum spontaneous Sn->S0 emission from shared VCI FC/HT operators.

    Three Cartesian dipole operators are in atomic units. Positive photon
    energies use initial - final - electronic_gap_cm, where electronic_gap_cm
    is final minus initial electronic reference energy. Sum exact line A values;
    do not integrate a broadened spectrum's cubic moment. A finite-space rate
    is not automatically a converged molecular rate.
    """
    operators = tuple(dipole_operators_au)
    if len(operators) != 3:
        raise ValueError('three Cartesian electric-dipole operators in atomic units required')
    lines = _operator_channel_lines(initial, final, electronic_gap_cm=electronic_gap_cm,
        operators=operators, initial_populations=initial_populations, strength_model=strength_model)
    photon = lines['wavenumbers_cm']; strengths = lines['population_weighted_strengths']
    if np.any(strengths < 0):
        raise ValueError('negative truncated dipole strengths; enlarge the variational treatment')
    contributions = EINSTEIN_A_FACTOR*np.maximum(photon,0)**3*strengths
    rate = float(contributions.sum())
    return dict(schema='matrix.trinity.common_vci_radiative_rate.v1',rate_s_1=rate,
        radiative_lifetime_s=None if rate==0 else 1/rate,lines=lines,
        rate_contributions_s_1=contributions,diagnostics=dict(lines['diagnostics'],
            excluded_nonpositive_photon_lines=int(np.count_nonzero(photon<=0)),
            environment='vacuum electric-dipole spontaneous emission',
            strengths_units='atomic_unit_dipole_squared',
            broadened_cubic_moment_used=False,spin_population_average_applied=False,
            energy_cubed_moment_convergence_certified=False))

def common_vci_nonradiative_rate(initial, final, *, electronic_gap_cm,
        coupling_hartree=None, gaussian_sigma_cm, initial_populations=None,
        strength_model='strict_order_two', method='TI', times_fs=None,
        maximum_workspace_bytes=256*1024**2, coupling_operators_hartree=None,
        channel_weights=None):
    """Finite-space FGR for constant channels or complete energy operators.

    Position/momentum coupling operators support Sn->S0 internal conversion.
    Their coefficients are Hartree; this API does not complete an incomplete
    derivative-coupling vector or infer a scalar second-derivative term.

    Each supplied coupling is an independently incoherent channel. No spin
    average is inferred. TI uses the existing Voigt profile with zero
    Lorentzian width; TD reuses the common correlation-to-FGR transform.
    Neither route changes the source VCI Hamiltonian or discards residual K.
    """
    if (not np.isfinite(gaussian_sigma_cm) or gaussian_sigma_cm <= 0
            or isinstance(maximum_workspace_bytes, bool)
            or not isinstance(maximum_workspace_bytes, int) or maximum_workspace_bytes < 1):
        raise ValueError('positive Gaussian width and workspace required')
    if (coupling_hartree is None) == (coupling_operators_hartree is None):
        raise ValueError('supply either constant coupling channels or complete coupling operators')
    if coupling_operators_hartree is None:
        coupling = np.asarray(coupling_hartree, complex)
        if coupling.ndim != 1 or not len(coupling) or not np.all(np.isfinite(coupling)):
            raise ValueError('finite coupling channels required')
        weights = _channel_weights(channel_weights, len(coupling))
        coupling = coupling*np.sqrt(weights)
        lines = common_vci_lines(initial, final, electronic_gap_cm=electronic_gap_cm,
            initial_populations=initial_populations, strength_model=strength_model)
        lines['diagnostics']['coupling_model'] = 'Condon energy channels'
    else:
        lines = _operator_channel_lines(initial, final, electronic_gap_cm=electronic_gap_cm,
            operators=coupling_operators_hartree, channel_weights=channel_weights,
            initial_populations=initial_populations, strength_model=strength_model)
        # Operator-weighted strengths already have Hartree squared units.
        coupling = np.ones(1, complex)
        lines['diagnostics']['coupling_model'] = 'complete supplied position/momentum energy operators'
        lines['diagnostics']['coupling_strength_units'] = 'hartree_squared'
    gap = lines['wavenumbers_cm'].ravel()
    weights = lines['population_weighted_strengths'].ravel()
    density = float(weights @ voigt_profile(gap, gaussian_sigma_cm, 0.))
    coupling_norm = float(np.vdot(coupling, coupling).real)
    direct = GOLDEN_RULE_FACTOR * coupling_norm * density
    if direct < 0:
        raise ValueError('negative truncated FGR; enlarge the variational treatment before using the rate')
    if method == 'TI':
        return dict(schema='matrix.trinity.common_vci_rate.v1', method=method,
            rate_s_1=direct, coupling_density_hartree2_per_cm=coupling_norm*density,
            lines=lines, diagnostics=dict(lines['diagnostics'],
                energy_conservation='direct finite Gaussian line sum',
                spin_population_average_applied=False, statistical_sampling_used=False))
    if method != 'TD':
        raise ValueError('method must be TI or TD')
    times = np.asarray(times_fs, float)
    if (times.ndim != 1 or len(times) < 3 or times[0] != 0
            or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0)
            or not np.allclose(np.diff(times), times[1], rtol=1e-10, atol=0)):
        raise ValueError('explicit uniform TD grid starting at zero required')
    reserved = 64*len(times)*(len(coupling)+1)+gap.nbytes+weights.nbytes
    batch = min(256, (maximum_workspace_bytes-reserved)//(32*len(times)))
    if batch < 1:
        raise MemoryError('common-mode VCI correlation exceeds workspace budget')
    signal = np.zeros(len(times), complex)
    active = np.flatnonzero(weights)
    for first in range(0, len(active), batch):
        ids = active[first:first+batch]
        signal += np.exp(1j*np.outer(times*PHASE_PER_CM_FS, gap[ids])) @ weights[ids]
    correlation = dict(times_fs=times,
        values=signal[:, None]*abs(coupling[None, :])**2,
        correlation_units='hartree_squared', diagnostics=dict(
            lines['diagnostics'], gaussian_time_tail_bound_available=bool(np.all(weights >= 0)),
            relative_propagator_error_bound=0.))
    result = nonradiative_rate_from_correlation(correlation, gaussian_sigma_cm=gaussian_sigma_cm)
    result.update(method=method, lines=lines, correlation=correlation, finite_TI_rate_s_1=direct,
        absolute_TI_TD_difference_s_1=abs(result['rate_s_1']-direct))
    return result
