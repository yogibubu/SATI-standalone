"""Nonresonant SAM wavefunctions and two-surface Duschinsky FC/HT contractions.

H(lambda)=H0+lambda V3+lambda**2 V4. No denominator shifts, resonant term
omissions, amplitude pruning or implicit variational diagonalization are used.
"""
from dataclasses import dataclass
import hashlib
import numpy as np
from numba import njit
from .curvilinear_vpt2 import (_canonical_cubic_terms, _canonical_quartic_terms,
    _compiled_operator_couplings, _validate_field)
from .vibronic_harmonic import orient_harmonic_model, harmonic_gaussian_generator
from .vibronic_kernels import parallel_rows, worker_count
from .vibronic_observables import _operator_rows
from .vibronic_selection import PackedHermiteCache


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def _occupations(states, n, name):
    states = np.asarray(states)
    if (states.ndim != 2 or states.shape[1] != n or not len(states)
            or not np.issubdtype(states.dtype, np.integer) or np.any(states < 0)
            or np.any(states > np.iinfo(np.int32).max)):
        raise ValueError(f'{name} must contain nonnegative integer occupation rows')
    states = np.asarray(states, np.int64)
    if len(np.unique(states, axis=0)) != len(states):
        raise ValueError(f'{name} contains duplicate states')
    return tuple(map(tuple, states.tolist()))


@dataclass(frozen=True)
class SAMVPT2Field:
    """Dense derivatives of a declared modal model in dimensionless q,p and cm^-1.

    Potential coefficients are F3/6 and F4/24. Kinetic coefficients are
    G1*p*q*p/2 and G2*p*q*q*p/4, using TRINITY operator ordering. Frequencies,
    ordering and phases must be those of the corresponding SATI harmonic state.
    ``complete`` denotes a complete supplied quartic field. The explicit
    ``semidiagonal_model`` declares a polynomial whose unacquired four-distinct
    source-coordinate coefficients were set to zero as a model approximation.
    Transform that entire polynomial between bases, without applying the
    truncation again. Bare ``semidiagonal`` remains incomplete input and is
    rejected; it must not silently acquire a complete-field interpretation.
    """
    frequencies_cm: np.ndarray
    cubic_cm: np.ndarray
    quartic_cm: np.ndarray
    metric_first_cm: np.ndarray | None = None
    metric_second_cm: np.ndarray | None = None
    constant_shift_cm: float = 0.
    quartic_coverage: str = 'complete'
    basis_id: str = ''

    def __post_init__(self):
        wi = np.asarray(self.frequencies_cm, float)
        if wi.ndim != 1 or not len(wi) or np.any(wi <= 0) or not np.all(np.isfinite(wi)):
            raise ValueError('SAM VPT2 requires positive finite harmonic frequencies')
        n = len(wi)
        for name, shape in (('frequencies_cm', (n,)), ('cubic_cm', (n,)*3),
                ('quartic_cm', (n,)*4), ('metric_first_cm', (n,)*3), ('metric_second_cm', (n,)*4)):
            value = getattr(self, name)
            value = np.zeros(shape) if value is None and name.startswith('metric_') else np.array(value, copy=True)
            if (value.shape != shape or not np.issubdtype(value.dtype, np.number)
                    or np.iscomplexobj(value) or not np.all(np.isfinite(value))):
                raise ValueError(f'finite real full SAM tensor required: {name}')
            value = np.array(value, float, copy=True); value.setflags(write=False)
            object.__setattr__(self, name, value)
        for name, swaps in (('cubic_cm', ((0, 1), (1, 2))),
                ('quartic_cm', ((0, 1), (1, 2), (2, 3))),
                ('metric_first_cm', ((0, 1),)), ('metric_second_cm', ((0, 1), (2, 3)))):
            tensor = getattr(self, name)
            if any(not np.allclose(tensor, tensor.swapaxes(i, j), rtol=1e-10, atol=1e-12) for i, j in swaps):
                raise ValueError(f'SAM tensor {name} is not symmetric')
        if self.quartic_coverage not in {'complete', 'semidiagonal_model'}:
            raise ValueError('wavefunction VPT2 requires complete quartic coverage or an explicit semidiagonal_model approximation')
        if not np.isfinite(self.constant_shift_cm):
            raise ValueError('finite SAM constant shift required')

    @classmethod
    def from_trinity(cls, field, harmonic_state):
        """Transport native ESALC F3/F4/G1/G2 in the EXACT SATI modal gauge.

        A stationary expansion and a declared closed quartic model are required. Degenerate
        modes and signs come from harmonic_state.modes(), not a new eigensolver.
        """
        from matrix_gf.harmonic import CM_PER_HARTREE, HESSIAN_EIGENVALUE_TO_CM
        native = getattr(field, 'field', field)
        f2, f3, f4, metric, g1, g2 = _validate_field(native)
        if (not np.array_equal(f2, harmonic_state.force_constants_hartree)
                or not np.array_equal(metric, harmonic_state.metric_inverse_amu)):
            raise ValueError('native field and SATI harmonic state must have the same F/G')
        if harmonic_state.gradient_hartree is not None and np.any(np.asarray(harmonic_state.gradient_hartree) != 0):
            raise ValueError('wavefunction VPT2 requires a stationary expansion, without an untransported cubic/quartic shift')
        wi, transform, minimum, _ = harmonic_state.modes()
        displacement = transform/np.sqrt(wi)[None, :]
        momentum = np.linalg.inv(displacement)
        scale = HESSIAN_EIGENVALUE_TO_CM**2/CM_PER_HARTREE
        basis_id = hashlib.sha256(harmonic_state.coordinate_identity.encode()+b''.join(
            np.asarray(a, dtype='<f8').tobytes() for a in (wi, transform, minimum))).hexdigest()
        return cls(wi,
            CM_PER_HARTREE*np.einsum('abc,ai,bj,ck->ijk', f3, displacement, displacement, displacement, optimize=True),
            CM_PER_HARTREE*np.einsum('abcd,ai,bj,ck,dl->ijkl', f4, displacement, displacement, displacement, displacement, optimize=True),
            scale*np.einsum('ia,jb,abc,ck->ijk', momentum, momentum, g1, displacement, optimize=True),
            scale*np.einsum('ia,jb,abcd,ck,dl->ijkl', momentum, momentum, g2, displacement, displacement, optimize=True),
            native.pseudopotential_cm, native.quartic_coverage, basis_id)


class NonresonantVPT2Error(ValueError):
    """A requested state needs treatment outside the nonresonant domain."""
    def __init__(self, message, diagnostic):
        super().__init__(message)
        self.diagnostic = diagnostic


@dataclass(frozen=True)
class SAMVPT2Wavefunction:
    reference: tuple
    orders: tuple  # sparse coefficients, order 0/1/2
    energy_orders_cm: np.ndarray
    diagnostics: dict


class _Wavefunctions:
    def __init__(self, field, denominator_tolerance_cm, maximum_correction_norm, maximum_terms):
        self.field = field
        self.tolerance = denominator_tolerance_cm
        self.norm_limit = maximum_correction_norm
        self.maximum_terms = maximum_terms
        self.cubic_terms = _canonical_cubic_terms(field.cubic_cm, field.metric_first_cm, 0.)
        self.quartic_terms = _canonical_quartic_terms(field.quartic_cm, field.metric_second_cm)
        self.actions = {}; self.states = {}

    def cubic(self, state):
        if state not in self.actions:
            if len(self.actions) >= self.maximum_terms:
                raise MemoryError('SAM action-cache budget exceeded')
            self.actions[state] = {k: v for k, v in _compiled_operator_couplings(state, self.cubic_terms).items() if v != 0}
        return self.actions[state]

    def build(self, reference):
        if reference in self.states: return self.states[reference]
        w = self.field.frequencies_cm; minimum_gap = np.inf
        def divide(numerator, state, order):
            nonlocal minimum_gap
            gap = float(w @ (np.asarray(reference)-state))
            minimum_gap = min(minimum_gap, abs(gap))
            if abs(gap) <= self.tolerance:
                raise NonresonantVPT2Error('SAM VPT2 has a resonant denominator; variational treatment is required',
                    dict(reference=reference, partner=state, order=order, gap_cm=gap,
                         numerator=complex(numerator), denominator_tolerance_cm=self.tolerance))
            return numerator/gap
        v1 = self.cubic(reference)
        e1 = v1.get(reference, 0j)
        first = {state: divide(value, state, 1) for state, value in v1.items() if state != reference}
        numerator = dict(_compiled_operator_couplings(reference, self.quartic_terms))
        for middle, coefficient in first.items():
            for state, value in self.cubic(middle).items():
                numerator[state] = numerator.get(state, 0j)+value*coefficient
                if len(numerator) > self.maximum_terms:
                    raise MemoryError('SAM wavefunction support budget exceeded')
        e2 = numerator.pop(reference, 0j)+self.field.constant_shift_cm
        second = {}
        for state in numerator.keys() | first.keys():
            value = numerator.get(state, 0j)-e1*first.get(state, 0j)
            if value != 0: second[state] = divide(value, state, 2)
        first_norm = float(np.linalg.norm(list(first.values())))
        second[reference] = -.5*first_norm**2
        second = {state: value for state, value in second.items() if value != 0}
        second_norm = float(np.linalg.norm(list(second.values())))
        if max(first_norm, second_norm) >= self.norm_limit:
            raise NonresonantVPT2Error('SAM VPT2 correction exceeds the declared validity limit; variational treatment is required',
                dict(reference=reference, first_order_norm=first_norm, second_order_norm=second_norm,
                     maximum_correction_norm=self.norm_limit))
        if abs(e1.imag)+abs(e2.imag) > 1e-10*max(1., abs(e1), abs(e2)):
            raise ValueError('SAM Hermitian perturbation produced nonreal energies')
        result = SAMVPT2Wavefunction(reference, ({reference: 1.+0j}, first, second),
            np.array([float(w @ (np.asarray(reference)+.5)), e1.real, e2.real]),
            dict(first_order_norm=first_norm, second_order_norm=second_norm,
                 minimum_active_denominator_cm=None if np.isinf(minimum_gap) else minimum_gap,
                 support_sizes=[1, len(first), len(second)],
                 validity='nonresonant and below the user norm limit; not a rigorous perturbative remainder bound'))
        self.states[reference] = result
        return result


def sam_vpt2_wavefunction(field, state, *, denominator_tolerance_cm=1e-8,
        maximum_correction_norm=1., maximum_terms=200000):
    """Normalized-through-order-two Rayleigh--Schrodinger state, without resonances."""
    _validate_controls(denominator_tolerance_cm, maximum_correction_norm, maximum_terms)
    state = _occupations([state], len(field.frequencies_cm), 'state')[0]
    return _Wavefunctions(field, denominator_tolerance_cm, maximum_correction_norm, maximum_terms).build(state)


def _validate_controls(tolerance, norm, terms):
    if not np.isfinite(tolerance) or tolerance <= 0 or not np.isfinite(norm) or norm <= 0:
        raise ValueError('positive finite denominator tolerance and correction-norm limit required')
    _positive_integer(terms, 'maximum_wavefunction_terms')


@njit(cache=True, nogil=True)
def _contract_orders(start, stop, offsets, indices, coefficients, moments, output):
    for line in range(start, stop):
        for order in range(3):
            row = 3*line+order
            for edge in range(offsets[row], offsets[row+1]):
                for channel in range(3):
                    output[line, order, channel] += coefficients[edge]*moments[indices[edge], channel]


def vpt2_vibronic_ti(model, initial_field, final_field, *, final_states, initial_states=None,
        initial_populations=None, kind='absorption', operator='FCHT',
        denominator_tolerance_cm=1e-8, maximum_correction_norm=1.,
        maximum_wavefunction_terms=200000, maximum_transitions=100000,
        maximum_harmonic_pairs=1000000, maximum_contraction_terms=5000000,
        maximum_integrals=1000000, integral_batch_size=256, workers=0,backend='python',native_executable=None):
    """Contract two independently corrected SAM states through harmonic FC/HT.

    Target states are explicit; no harmonic intensity preselection or two-quanta
    cap is imposed. Populations are supplied absolute weights, without implicit
    thermal renormalization. Signed strengths are strictly truncated at order 2;
    the positive square of the corrected moment is returned separately.
    """
    _validate_controls(denominator_tolerance_cm, maximum_correction_norm, maximum_wavefunction_terms)
    for name, value in dict(maximum_transitions=maximum_transitions,
            maximum_harmonic_pairs=maximum_harmonic_pairs, maximum_contraction_terms=maximum_contraction_terms,
            maximum_integrals=maximum_integrals, integral_batch_size=integral_batch_size).items():
        _positive_integer(value, name)
    if operator not in {'FC', 'HT', 'FCHT'}: raise ValueError('operator must be FC, HT or FCHT')
    # Check the fields in the supplied orientation before swapping for emission.
    for field, wi in ((initial_field, model.initial_frequencies_cm), (final_field, model.final_frequencies_cm)):
        if not np.array_equal(field.frequencies_cm, wi):
            raise ValueError('SAM frequencies/order must match the corresponding SATI harmonic basis exactly')
    for name, field in (('initial', initial_field), ('final', final_field)):
        expected = model.provenance.get(name+'_normal_basis_sha256')
        if field.basis_id and expected and field.basis_id != expected:
            raise ValueError(f'{name} SAM modal gauge does not match the SATI basis')
    if kind == 'emission': initial_field, final_field = final_field, initial_field
    model, direction = orient_harmonic_model(model, kind)
    n = len(model.initial_frequencies_cm); workers = worker_count(workers)
    initial = _occupations([np.zeros(n, dtype=int)] if initial_states is None else initial_states, n, 'initial_states')
    final = _occupations(final_states, n, 'final_states')
    count = len(initial)*len(final)
    if count > maximum_transitions: raise MemoryError('SAM transition budget exceeded')
    if initial_populations is None:
        if initial != ((0,)*n,): raise ValueError('non-vacuum initial states require explicit populations')
        population = np.ones(1)
    else:
        population = np.asarray(initial_populations, float)
        if (population.shape != (len(initial),) or not np.all(np.isfinite(population))
                or np.any(population < 0) or population.sum() > 1+1e-12):
            raise ValueError('absolute nonnegative initial populations summing to at most one required')
    sources = _Wavefunctions(initial_field, denominator_tolerance_cm, maximum_correction_norm, maximum_wavefunction_terms)
    targets = _Wavefunctions(final_field, denominator_tolerance_cm, maximum_correction_norm, maximum_wavefunction_terms)
    left = [sources.build(state) for state in initial]; right = [targets.build(state) for state in final]
    pairs = {}; pair_rows = []; indices = []; coefficients = []; offsets = [0]
    for source in left:
        for target in right:
            for order in range(3):
                for left_order in range(order+1):
                    for i, ci in source.orders[left_order].items():
                        for f, cf in target.orders[order-left_order].items():
                            pair = i+f
                            if pair not in pairs:
                                if len(pairs) >= maximum_harmonic_pairs:
                                    raise MemoryError('SAM harmonic-pair budget exceeded')
                                pairs[pair] = len(pair_rows); pair_rows.append(pair)
                            if len(indices) >= maximum_contraction_terms:
                                raise MemoryError('SAM contraction-edge budget exceeded')
                            indices.append(pairs[pair]); coefficients.append(ci*np.conj(cf))
                offsets.append(len(indices))
    states = np.asarray(pair_rows, np.int64)
    if backend not in ('python','cpp'):raise ValueError('backend must be python or cpp')
    library=None
    if backend=='cpp':
        if native_executable is None:raise ValueError('Explicit native_executable required for C++')
        from .vibronic_native_kernels import NativeKernels
        library=NativeKernels(str(native_executable)+'.kernels.so')
    generator = harmonic_gaussian_generator(model) if library is None else library.prepare_dipole(model,0.)
    if generator['log_ground'] < np.log(np.finfo(float).tiny)+4:
        raise FloatingPointError('SAM harmonic overlap underflows; log-scaled recurrences are required')
    constant = (model.transition_dipole_au if operator != 'HT' else np.zeros(3))+(
        generator['ht_constant'] if operator != 'FC' else np.zeros(3))
    linear = generator['ht_linear'] if operator != 'FC' else np.zeros_like(generator['ht_linear'])
    cache = PackedHermiteCache(generator, linear, maximum_integrals, max(1, int(states.max())),native_library=library)
    moments = np.empty((len(states), 3), complex)
    for start in range(0, len(states), integral_batch_size):
        batch = states[start:start+integral_batch_size]
        ids, exhausted = cache.evaluate(batch)
        if exhausted: raise MemoryError('SAM harmonic overlap-cache budget exceeded; no partial spectrum returned')
        if library is None:
            parallel_rows(_operator_rows, len(batch), workers, batch, ids, cache.values, constant, linear,
                          moments[start:start+len(batch)])
        else:moments[start:start+len(batch)]=library.operator_rows(batch,ids,cache.values,constant,linear)
    if library is None:
        amplitudes = np.zeros((count, 3, 3), complex)
        parallel_rows(_contract_orders, count, workers, np.asarray(offsets, np.int64),
                      np.asarray(indices, np.int64), np.asarray(coefficients, complex), moments, amplitudes)
    else:amplitudes=library.contract_orders(count,offsets,indices,coefficients,moments)
    strength_orders = np.column_stack([np.sum(abs(amplitudes[:, 0])**2, axis=1),
        2*np.einsum('ta,ta->t', amplitudes[:, 0].conj(), amplitudes[:, 1]).real,
        np.sum(abs(amplitudes[:, 1])**2, axis=1)+2*np.einsum('ta,ta->t', amplitudes[:, 0].conj(), amplitudes[:, 2]).real])
    energies_i = np.array([state.energy_orders_cm for state in left])
    energies_f = np.array([state.energy_orders_cm for state in right])
    photon_orders = direction*(np.tile(energies_f, (len(initial), 1))-np.repeat(energies_i, len(final), axis=0))
    photon_orders[:, 0] += direction*model.electronic_gap_cm
    weights = np.repeat(population, len(final)); strength = strength_orders.sum(axis=1)
    positive = np.sum(abs(amplitudes.sum(axis=1))**2, axis=1)
    return dict(schema='matrix.sati.vpt2_vibronic_ti.v1', kind=kind, operator=operator,
        initial_quanta=np.repeat(np.asarray(initial), len(final), axis=0),
        final_quanta=np.tile(np.asarray(final), (len(initial), 1)), population=weights,
        photon_energy_orders_cm=photon_orders, wavenumbers_cm=photon_orders.sum(axis=1),
        amplitude_orders_au=amplitudes, strength_orders_au2=strength_orders,
        strengths_au2=strength, weighted_strengths_au2=weights*strength,
        positive_completed_strengths_au2=positive, weighted_positive_completed_strengths_au2=weights*positive,
        initial_state_diagnostics=[state.diagnostics for state in left],
        final_state_diagnostics=[state.diagnostics for state in right],
        diagnostics=dict(numerical_backend=backend,evaluated_transitions=count, harmonic_pairs=len(pairs), contraction_terms=len(indices),
            cached_integrals=cache.count, cubic_action_cache_sizes=[len(sources.actions), len(targets.actions)],
            initial_population_retained=float(population.sum()), maximum_correction_norm=maximum_correction_norm,
            denominator_tolerance_cm=denominator_tolerance_cm, workers=workers,
            maximum_first_order_norm=max(s.diagnostics['first_order_norm'] for s in left+right),
            maximum_second_order_norm=max(s.diagnostics['second_order_norm'] for s in left+right),
            negative_truncated_strengths=int(np.count_nonzero(strength < 0)),
            maximum_positive_completion_difference_au2=float(np.max(abs(positive-strength))),
            harmonic_completeness_certificate_used=False, anharmonic_omission_bound=None,
            target_space_complete=False, resonant_terms_removed=False, denominator_shifts_used=False,
            amplitude_pruning_used=False, no_renormalization=True,
            scope='explicit nonresonant SAM targets; mechanical order two, FC/linear HT; no completeness or remainder certificate'),
        provenance=model.provenance)


__all__ = ['SAMVPT2Field', 'SAMVPT2Wavefunction', 'NonresonantVPT2Error',
           'sam_vpt2_wavefunction', 'vpt2_vibronic_ti']
