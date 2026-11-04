"""Fourth-order quantum normal forms for an index-one molecular saddle.

Energies are cm^-1; canonical coordinates and actions use hbar=1. The
reactive quadratic operator is lambda*(p**2-q**2)/2, never an oscillator.
Ordered TRINITY q/p factors are converted with exact Weyl star products.
"""
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from math import comb, factorial, sqrt

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq
from scipy.special import expit

from .curvilinear_vpt2 import (
    _canonical_cubic_terms, _canonical_quartic_terms,
)
from .vibronic_vpt2 import SAMVPT2Field


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')


def _real(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or np.iscomplexobj(value):
        raise ValueError(f'{name} must be a finite real scalar')
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a finite real scalar') from exc
    if not np.isfinite(value):
        raise ValueError(f'{name} must be a finite real scalar')
    return value


@dataclass(frozen=True)
class TransitionStateField:
    """One negative signed frequency, real F3/F4/G1/G2 in |omega|-scaled q,p.

    Complete or explicitly closed semidiagonal_model quartics are accepted.
    All fields must describe the same stationary saddle and coordinate basis.
    """
    signed_frequencies_cm: np.ndarray
    cubic_cm: np.ndarray
    quartic_cm: np.ndarray
    metric_first_cm: np.ndarray | None = None
    metric_second_cm: np.ndarray | None = None
    constant_shift_cm: float = 0.
    quartic_coverage: str = 'complete'
    barrier_energy_cm: float = 0.
    basis_id: str = ''

    def __post_init__(self):
        w = np.asarray(self.signed_frequencies_cm)
        if (w.ndim != 1 or not len(w) or np.iscomplexobj(w)
                or not np.all(np.isfinite(w)) or np.any(w == 0)
                or np.count_nonzero(w < 0) != 1):
            raise ValueError('exactly one negative signed frequency and no zero modes required')
        checked = SAMVPT2Field(abs(w), self.cubic_cm, self.quartic_cm,
                              self.metric_first_cm, self.metric_second_cm,
                              self.constant_shift_cm, self.quartic_coverage, self.basis_id)
        for name in ('cubic_cm', 'quartic_cm', 'metric_first_cm', 'metric_second_cm'):
            object.__setattr__(self, name, getattr(checked, name))
        w = np.array(w, float, copy=True)
        w.setflags(write=False)
        object.__setattr__(self, 'signed_frequencies_cm', w)
        object.__setattr__(self, 'barrier_energy_cm', _real(self.barrier_energy_cm, 'barrier_energy_cm'))

    @property
    def reaction_mode(self):
        return int(np.flatnonzero(self.signed_frequencies_cm < 0)[0])

    @classmethod
    def from_gaussian_anharmonic(cls, path, *, omitted_terms, quartic_coverage,
                                barrier_energy_cm=0., basis_id=''):
        """Real Gaussian saddle field with explicit print/quartic model closure."""
        from matrix_gaussian.saddle_qff import read_gaussian_saddle_qff
        data = read_gaussian_saddle_qff(path, omitted_terms=omitted_terms,
                                        quartic_coverage=quartic_coverage)
        return cls(data.signed_frequencies_cm, data.cubic_cm, data.quartic_cm,
                   quartic_coverage=quartic_coverage, barrier_energy_cm=barrier_energy_cm,
                   basis_id=basis_id or data.diagnostics['source_sha256'])

    @classmethod
    def from_trinity(cls, source, *, barrier_energy_cm=0., basis_id=''):
        """Native positive Wilson metric and signed saddle Hessian, without reflection."""
        from .transition_state_coordinates import modalize_transition_state
        source = getattr(source, 'field', source)
        w, f3, f4, g1, g2 = modalize_transition_state(source)
        return cls(w, f3, f4, g1, g2, source.pseudopotential_cm,
                   source.quartic_coverage, barrier_energy_cm, basis_id)


class _Algebra:
    def __init__(self, modes, reaction, maximum_terms, maximum_bracket_pairs):
        _positive_integer(maximum_terms, 'maximum_terms')
        _positive_integer(maximum_bracket_pairs, 'maximum_bracket_pairs')
        self.n = modes
        self.reaction = reaction
        self.maximum_terms = maximum_terms
        self.maximum_bracket_pairs = maximum_bracket_pairs
        self.pairs = 0
        self.zero = (0,) * (2 * modes)

    def check(self, poly):
        if len(poly) > self.maximum_terms:
            raise MemoryError('transition-state polynomial term budget exceeded')
        if any(not np.isfinite(value) for value in poly.values()):
            raise FloatingPointError('nonfinite normal-form coefficient')
        return {key: value for key, value in poly.items() if value != 0}

    def add(self, *polys):
        out = {}
        for poly in polys:
            for key, value in poly.items():
                out[key] = out.get(key, 0j) + value
        return self.check(out)

    def multiply(self, left, right, *, bracket=False):
        out = {}
        # A bracket only connects terms sharing conjugate canonical factors.
        index = {}
        if bracket:
            for key in right:
                for j, power in enumerate(key):
                    if power:
                        index.setdefault(j, []).append(key)
        for a, va in left.items():
            if bracket:
                candidates = set()
                for j, power in enumerate(a):
                    if power:
                        candidates.update(index.get(j ^ 1, ()))
                candidates = sorted(candidates)
            else:
                candidates = right
            for b in candidates:
                self.pairs += 1
                if self.pairs > self.maximum_bracket_pairs:
                    raise MemoryError('transition-state contraction budget exceeded')
                vb = right[b]
                options = []
                for mode in range(self.n):
                    j = 2 * mode
                    options.append(_local_star(a[j], a[j+1], b[j], b[j+1],
                                               mode == self.reaction))
                for selections in product(*options):
                    contractions = sum(item[2] for item in selections)
                    if bracket and contractions % 2 == 0:
                        continue
                    key = tuple(power for item in selections for power in item[:2])
                    value = va * vb * (-2j if bracket else 1)
                    for item in selections:
                        value *= item[3]
                    out[key] = out.get(key, 0j) + value
                if len(out) > self.maximum_terms:
                    raise MemoryError('transition-state polynomial term budget exceeded')
        return self.check(out)

    def bracket(self, left, right):
        return self.multiply(left, right, bracket=True)

    def ordered(self, terms):
        out = {}
        for groups, coefficient in terms:
            if not np.isscalar(coefficient) or not np.isfinite(coefficient):
                raise ValueError('finite scalar operator coefficient required')
            poly = {self.zero: complex(coefficient)}
            for mode, operators in groups:
                if (isinstance(mode, (bool, np.bool_)) or not isinstance(mode, (int, np.integer))
                        or not 0 <= mode < self.n or any(op not in {'q', 'p'} for op in operators)):
                    raise ValueError('ordered operator factors must contain valid q/p modes')
                for operator in operators:
                    j = 2 * mode
                    a = list(self.zero); a[j] = 1
                    b = list(self.zero); b[j+1] = 1
                    if mode == self.reaction:
                        # q=(u-v)/sqrt(2), p=(u+v)/sqrt(2), {u,v}=1.
                        factor = 1 / sqrt(2)
                        linear = {tuple(a): factor,
                                  tuple(b): factor if operator == 'p' else -factor}
                    else:
                        # z=(q+ip)/sqrt(2), zbar=(q-ip)/sqrt(2).
                        linear = ({tuple(a): 1/sqrt(2), tuple(b): 1/sqrt(2)}
                                  if operator == 'q' else
                                  {tuple(a): -1j/sqrt(2), tuple(b): 1j/sqrt(2)})
                    poly = self.multiply(poly, linear)
            # Each ordered term has a small canonical support. Accumulate it
            # directly instead of copying and rechecking the entire growing
            # polynomial for every input tensor element.
            for key, value in poly.items():
                updated = out.get(key, 0j) + value
                if not np.isfinite(updated):
                    raise FloatingPointError('nonfinite normal-form coefficient')
                if updated == 0:
                    out.pop(key, None)
                else:
                    out[key] = updated
            if len(out) > self.maximum_terms:
                raise MemoryError('transition-state polynomial term budget exceeded')
        return self.check(out)


@lru_cache(maxsize=16384)
def _local_star(a, b, c, d, reactive):
    """Exact monomial star contractions; {u,v}=1 and {z,zbar}=-i."""
    j = 1 if reactive else -1j
    rows = []
    for forward in range(min(a, d) + 1):
        for backward in range(min(b, c) + 1):
            k = forward + backward
            falling = (factorial(a)//factorial(a-forward)
                       * factorial(d)//factorial(d-forward)
                       * factorial(b)//factorial(b-backward)
                       * factorial(c)//factorial(c-backward))
            value = ((.5j*j)**k * (-1)**backward * falling
                     / (factorial(forward)*factorial(backward)))
            rows.append((a+c-k, b+d-k, k, value))
    return tuple(rows)


def _partition(poly, frequencies, reaction, window):
    retained, generator = {}, {}
    removed = 0.
    for key, coefficient in poly.items():
        imbalance = key[2*reaction] - key[2*reaction+1]
        bath_gap = sum(abs(w)*(key[2*i+1]-key[2*i])
                       for i, w in enumerate(frequencies) if i != reaction)
        divisor = abs(frequencies[reaction])*imbalance + 1j*bath_gap
        if imbalance == 0 and abs(bath_gap) <= window:
            retained[key] = coefficient
        else:
            generator[key] = -coefficient/divisor
            removed = max(removed, abs(coefficient/divisor))
    return retained, generator, removed


@dataclass(frozen=True)
class TransitionStateNormalForm:
    field: TransitionStateField
    polynomial: dict
    generator_cubic: dict
    generator_quartic: dict
    diagnostics: dict
    normal_cubic: dict | None = None
    normal_quartic: dict | None = None


def solve_transition_state_normal_form(field, *, resonance_window_cm=1e-8,
                                      maximum_terms=100000,
                                      maximum_bracket_pairs=5000000):
    """Quantum Weyl normal form through V4 and V3**2, retaining bath resonances.

    This is a saddle normal form, not an HDC analytic continuation. No bound
    state is assigned to the reactive coordinate. A nonzero resonance window
    defines a partial resonant normal form: retained bath couplings are treated
    by the channel matrix, not removed through a small homological divisor.
    """
    if not isinstance(field, TransitionStateField):
        raise TypeError('TransitionStateField required')
    resonance_window_cm = _real(resonance_window_cm, 'resonance_window_cm')
    if resonance_window_cm < 0:
        raise ValueError('nonnegative resonance_window_cm required')
    n = len(field.signed_frequencies_cm); r = field.reaction_mode
    alg = _Algebra(n, r, maximum_terms, maximum_bracket_pairs)
    cubic = alg.ordered(_canonical_cubic_terms(field.cubic_cm, field.metric_first_cm, 0.))
    quartic = alg.ordered(_canonical_quartic_terms(field.quartic_cm, field.metric_second_cm))
    quartic = alg.add(quartic, {alg.zero: field.constant_shift_cm})
    z3, w3, max3 = _partition(cubic, field.signed_frequencies_cm, r, resonance_window_cm)
    bracket = alg.bracket(w3, alg.add(cubic, z3))
    residual = alg.add(quartic, {key: value/2 for key, value in bracket.items()})
    z4, w4, max4 = _partition(residual, field.signed_frequencies_cm, r, resonance_window_cm)
    harmonic = {alg.zero: field.barrier_energy_cm}
    for i, w in enumerate(field.signed_frequencies_cm):
        key = list(alg.zero); key[2*i] = key[2*i+1] = 1
        harmonic[tuple(key)] = abs(w)
    poly = alg.add(harmonic, z3, z4)
    return TransitionStateNormalForm(field, poly, w3, w4, dict(
        schema='matrix.trinity.transition_state_normal_form.v1',
        formulation='quantum_weyl_saddle_normal_form_order_four',
        reaction_mode=r, stable_modes=[i for i in range(n) if i != r],
        quartic_coverage=field.quartic_coverage,
        resonance_window_cm=resonance_window_cm,
        reactive_action_commutes=True,
        retained_bath_resonances=any(any(key[2*i] != key[2*i+1]
                                       for i in range(n) if i != r) for key in poly),
        reactive_coordinate_is_bound=False, global_scattering_claimed=False,
        hbar_convention='dimensionless actions in units of hbar',
        maximum_cubic_generator_coefficient=float(max3),
        maximum_quartic_generator_coefficient=float(max4),
        polynomial_terms=len(poly), contractions=alg.pairs,
    ), normal_cubic=z3, normal_quartic=z4)


def _basis(states, modes, maximum_states, maximum_dense_bytes):
    _positive_integer(maximum_states, 'maximum_states')
    _positive_integer(maximum_dense_bytes, 'maximum_dense_bytes')
    a = np.asarray(states)
    if (a.ndim != 2 or a.shape[1] != modes or not len(a)
            or a.dtype.kind not in 'iu' or np.any(a < 0)
            or np.any(a > np.iinfo(np.int32).max) or len(np.unique(a, axis=0)) != len(a)):
        raise ValueError('unique nonnegative integer bath occupation rows required')
    if len(a) > maximum_states or len(a)**2 * 16 * 6 > maximum_dense_bytes:
        raise MemoryError('transition-state channel matrix budget exceeded')
    return tuple(map(tuple, a.tolist()))


@lru_cache(maxsize=16384)
def _weyl_ladder(a, b, occupation):
    """Weyl(z**a zbar**b) = sum_k C(a,k)C(b,k)k!/2**k adag**(b-k)a**(a-k)."""
    target = occupation + b - a
    if target < 0:
        return target, 0.
    value = 0.
    for k in range(min(a, b)+1):
        lowering = a-k; raising = b-k
        if occupation < lowering:
            continue
        factor = comb(a, k)*comb(b, k)*factorial(k)/2**k
        for j in range(lowering):
            factor *= sqrt(occupation-j)
        for j in range(raising):
            factor *= sqrt(occupation-lowering+j+1)
        value += factor
    return target, value


def _action_polynomial_matrices(result, basis, polynomial):
    r = result.field.reaction_mode
    stable = [i for i in range(len(result.field.signed_frequencies_cm)) if i != r]
    lookup = {state: i for i, state in enumerate(basis)}
    matrices = np.zeros((3, len(basis), len(basis)), complex)
    for key, coefficient in polynomial.items():
        a, b = key[2*r:2*r+2]
        if a != b or a > 2:
            raise ValueError('channel operator must conserve reactive action through degree four')
        # Weyl(u^2 v^2) = I_hat^2 - 1/4, not I_hat^2.
        factors = ((0, 1.),) if a == 0 else (((1, 1.),) if a == 1 else ((2, 1.), (0, -.25)))
        for col, state in enumerate(basis):
            target = list(state); value = coefficient
            for j, mode in enumerate(stable):
                target[j], factor = _weyl_ladder(key[2*mode], key[2*mode+1], state[j])
                value *= factor
            row = lookup.get(tuple(target))
            if row is not None:
                for power, factor in factors:
                    matrices[power, row, col] += value*factor
    for matrix in matrices:
        scale = max(1., float(np.linalg.norm(matrix)))
        if not np.all(np.isfinite(matrix)) or np.linalg.norm(matrix-matrix.conj().T) > 1e-9*scale:
            raise ValueError('non-Hermitian or nonfinite transverse channel operator')
        matrix[:] = (matrix+matrix.conj().T)/2
    return matrices


def transition_state_channel_matrix(result, reactive_action, bath_states, *,
                                    maximum_states=4096, maximum_dense_bytes=512*1024**2):
    """Hermitian transverse operator at a real reactive action; no reaction HO basis."""
    if not isinstance(result, TransitionStateNormalForm):
        raise TypeError('TransitionStateNormalForm required')
    reactive_action = _real(reactive_action, 'reactive_action')
    basis = _basis(bath_states, len(result.field.signed_frequencies_cm)-1,
                   maximum_states, maximum_dense_bytes)
    a, b, c = _action_polynomial_matrices(result, basis, result.polynomial)
    with np.errstate(over='ignore', invalid='ignore'):
        matrix = a + reactive_action*b + reactive_action**2*c
    if not np.all(np.isfinite(matrix)):
        raise FloatingPointError('nonfinite channel matrix at the requested action')
    return matrix


def transition_state_channels(result, reactive_action, bath_states, **controls):
    matrix = transition_state_channel_matrix(result, reactive_action, bath_states, **controls)
    energies, vectors = eigh(matrix, check_finite=False)
    return dict(reactive_action=float(reactive_action), energies_cm=energies,
                eigenvectors=vectors, finite_bath_basis=True,
                reactive_coordinate_is_bound=False)


def transition_state_transmission(result, energy_cm, bath_states, *, channels=None,
                                  action_interval=(-2., 2.), maximum_states=4096,
                                  maximum_dense_bytes=512*1024**2):
    """Local Miller/QNF transmission on certified monotone eigenvalue branches.

    No extrapolation outside the supplied action interval. A selected branch
    must bracket the energy. The sum is over declared finite bath channels.
    """
    energy_cm = _real(energy_cm, 'energy_cm')
    interval = np.asarray(action_interval, float)
    if (interval.shape != (2,) or not np.all(np.isfinite(interval))
            or interval[0] >= interval[1] or not np.isfinite(energy_cm)):
        raise ValueError('finite energy and ordered finite action interval required')
    basis = _basis(bath_states, len(result.field.signed_frequencies_cm)-1,
                   maximum_states, maximum_dense_bytes)
    a, b, c = _action_polynomial_matrices(result, basis, result.polynomial)
    def values(action):
        matrix = a+action*b+action**2*c
        if not np.all(np.isfinite(matrix)):
            raise FloatingPointError('nonfinite channel matrix in the action interval')
        return eigh(matrix, eigvals_only=True, check_finite=False)
    # H'(I) is affine: positivity at both ends certifies the whole interval.
    derivatives = [b+2*t*c for t in interval]
    if any(not np.all(np.isfinite(d)) for d in derivatives):
        raise FloatingPointError('nonfinite channel slope in the action interval')
    margin = 64*np.finfo(float).eps*len(basis)*max(1., *(float(np.linalg.norm(d)) for d in derivatives))
    slopes = [float(np.min(eigh(d, eigvals_only=True, check_finite=False)))-margin for d in derivatives]
    if min(slopes) <= 0:
        raise ValueError('channel monotonicity not certified on this action interval')
    ids = np.arange(len(basis)) if channels is None else np.asarray(channels)
    if (ids.ndim != 1 or not len(ids) or ids.dtype.kind not in 'iu'
            or np.any(ids < 0) or np.any(ids >= len(basis)) or len(np.unique(ids)) != len(ids)):
        raise ValueError('unique included channel indices required')
    lower, upper = values(interval[0]), values(interval[1])
    if any(not lower[k] <= energy_cm <= upper[k] for k in ids):
        raise ValueError('selected channel energy is outside the local action bracket')
    actions = np.array([brentq(lambda x: values(x)[k]-energy_cm, *interval) for k in ids])
    probabilities = expit(2*np.pi*actions)
    return dict(channels=ids, reactive_actions=actions, probabilities=probabilities,
                cumulative_probability=float(probabilities.sum()), action_interval=interval,
                minimum_certified_slope_cm=min(slopes), finite_channel_sum=True,
                global_scattering_claimed=False)


def transition_state_transform_operator(result, ordered_terms, *, maximum_terms=100000,
                                        maximum_bracket_pairs=5000000):
    """Mechanical order-two Weyl operator; retains reactive dependence, not a bound FC factor.

    Return O + {W3,O}_M + {W4,O}_M + {W3,{W3,O}_M}_M/2.
    ``ordered_terms`` use the existing TRINITY canonical q/p factor format.
    """
    alg = _Algebra(len(result.field.signed_frequencies_cm), result.field.reaction_mode,
                   maximum_terms, maximum_bracket_pairs)
    bare = alg.ordered(ordered_terms)
    first = alg.bracket(result.generator_cubic, bare)
    second = alg.add(alg.bracket(result.generator_quartic, bare),
                     {k: v/2 for k, v in alg.bracket(result.generator_cubic, first).items()})
    return dict(bare=bare, first_order=first, second_order=second,
                total=alg.add(bare, first, second), reactive_coordinate_is_bound=False)
