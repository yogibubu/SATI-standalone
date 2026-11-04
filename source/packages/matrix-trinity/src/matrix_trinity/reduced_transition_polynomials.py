"""Algebraically reduced second-order transition-property coefficients.

The keys (u, v) are exponents of auxiliary commuting variables. They are
not intermediate oscillator states. All contractions below have explicit
integer coefficients and denominators (u-v).omega; no virtual-state basis
or oscillator matrix is constructed. See the TRINITY SI reduction.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from math import comb, factorial, sqrt, prod as integer_product

import numpy as np

try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except ImportError:  # portable source fallback
    _NUMBA_AVAILABLE = False
    def njit(*_args, **_kwargs):
        return lambda function: function


Power = bytes
Key = tuple[Power, Power]
Polynomial = dict[Key, complex | np.ndarray]


class ResonantTransitionPolynomialError(ValueError):
    """The global reduction needs a state-selective resonant treatment."""


def _add(output: Polynomial, key: Key, value) -> None:
    output[key] = output.get(key, 0) + value


def _has_nonzero(value) -> bool:
    if isinstance(value, (int, float, complex, np.number)):
        return bool(value != 0)
    return bool(np.asarray(value).any())


def _nonzero(values: Polynomial) -> Polynomial:
    return {key: value for key, value in values.items() if _has_nonzero(value)}


def _python_commuting_factors(n: int, factors: tuple[tuple[str, int], ...]):
    """Coefficients of products of Q_i=x_i+y_i and R_i=x_i-y_i."""
    polynomial = {(bytes(n), bytes(n)): 1}
    for kind, mode in factors:
        following = {}
        for (u, v), coefficient in polynomial.items():
            if kind != "y":
                power = list(u)
                power[mode] += 1
                _add(following, (bytes(power), v), coefficient)
            if kind != "x":
                power = list(v)
                power[mode] += 1
                _add(following, (u, bytes(power)),
                     -coefficient if kind == "R" else coefficient)
        polynomial = following
    return tuple(polynomial.items())


@njit(cache=True)
def _compiled_commuting_factors(n, kinds, modes):
    rank = len(kinds)
    count = 1
    for k in kinds:
        if k >= 2:
            count *= 2
    left = np.zeros((count, n), dtype=np.uint8)
    right = np.zeros((count, n), dtype=np.uint8)
    weights = np.zeros(count, dtype=np.int64)
    used = 0
    for choice in range(count):
        u = np.zeros(n, dtype=np.uint8)
        v = np.zeros(n, dtype=np.uint8)
        sign, bits = 1, choice
        for j in range(rank):
            kind, mode = kinds[j], modes[j]
            if kind == 0:
                u[mode] += 1
            elif kind == 1:
                v[mode] += 1
            else:
                if bits & 1:
                    v[mode] += 1
                    if kind == 3:
                        sign = -sign
                else:
                    u[mode] += 1
                bits >>= 1
        previous = -1
        for row in range(used):
            same = True
            for i in range(n):
                if left[row, i] != u[i] or right[row, i] != v[i]:
                    same = False
                    break
            if same:
                previous = row
                break
        if previous < 0:
            previous = used
            left[used] = u
            right[used] = v
            used += 1
        weights[previous] += sign
    return left[:used], right[:used], weights[:used]


@lru_cache(maxsize=32768)
def _commuting_factors(n: int, factors: tuple[tuple[str, int], ...]):
    if not _NUMBA_AVAILABLE or len(factors) > 4 or any(mode < 0 or mode >= n for _, mode in factors):
        return _python_commuting_factors(n, factors)
    codes = {"x": 0, "y": 1, "Q": 2, "R": 3}
    kinds = np.asarray([codes.get(kind, 2) for kind, _ in factors], dtype=np.uint8)
    modes = np.asarray([mode for _, mode in factors], dtype=np.int64)
    left, right, weights = _compiled_commuting_factors(n, kinds, modes)
    return tuple(((bytes(a), bytes(b)), int(weight))
                 for a, b, weight in zip(left, right, weights))


def _put(output: Polynomial, n: int, coefficient,
         factors: tuple[tuple[str, int], ...]) -> None:
    if not _has_nonzero(coefficient):
        return
    for key, multiplier in _commuting_factors(n, factors):
        _add(output, key, coefficient * multiplier)


@lru_cache(maxsize=32768)
def _integer_contractions(a: Power, b: Power, c: Power, d: Power):
    """The explicit binomial coefficient C_(a,b),(c,d)^(u,v) in the SI."""
    n = len(a)
    output = {}
    for sign, left, right in ((1, b, c), (-1, d, a)):
        active = [i for i in range(n) if min(left[i], right[i])]
        for values in product(*(range(min(left[i], right[i]) + 1) for i in active)):
            if not any(values):
                continue
            contraction = [0] * n
            coefficient = sign
            for i, count in zip(active, values, strict=True):
                contraction[i] = count
                coefficient *= comb(left[i], count) * comb(right[i], count) * factorial(count)
            u = bytes(a[i] + c[i] - contraction[i] for i in range(n))
            v = bytes(b[i] + d[i] - contraction[i] for i in range(n))
            _add(output, (u, v), coefficient)
    return tuple((key, value) for key, value in output.items() if value)


def _contract(left: Polynomial, right: Polynomial) -> Polynomial:
    output = {}
    right_terms = [(c, d, y, _support(c), _support(d)) for (c, d), y in right.items()]
    for (a, b), x in left.items():
        ma, mb = _support(a), _support(b)
        for c, d, y, mc, md in right_terms:
            if not ((mb & mc) or (md & ma)):
                continue
            for key, coefficient in _integer_contractions(a, b, c, d):
                _add(output, key, coefficient * x * y)
    return _nonzero(output)


def _support(power: Power) -> int:
    return sum(1 << i for i, count in enumerate(power) if count)


def _paired_scalar_contraction(left: Polynomial, right: Polynomial) -> Polynomial:
    """Pair antisymmetric scalar terms once for the field-only C(c,s1)."""
    terms = [(a, b, x, right[a, b], _support(a), _support(b))
             for (a, b), x in left.items()]
    output = {}
    for index, (a, b, x, y, ma, mb) in enumerate(terms):
        for c, d, z, w, mc, md in terms[index + 1:]:
            if not ((mb & mc) or (md & ma)):
                continue
            value = x * w - y * z
            if value == 0:
                continue
            for key, coefficient in _integer_contractions(a, b, c, d):
                _add(output, key, coefficient * value)
    return _nonzero(output)


def _frequency_inverse(values: Polynomial, frequencies: np.ndarray,
                       tolerance: float) -> Polynomial:
    output = {}
    for (u, v), coefficient in values.items():
        difference = (np.frombuffer(u, dtype=np.uint8).astype(np.int16)
                      - np.frombuffer(v, dtype=np.uint8).astype(np.int16))
        if not np.any(difference):
            continue
        denominator = float(np.dot(difference, frequencies))
        if abs(denominator) <= tolerance:
            raise ResonantTransitionPolynomialError(
                "resonant monomial requires the selected-polyad contraction")
        output[u, v] = coefficient / denominator
    return output


def _hamiltonian_coefficients(f3: np.ndarray, f4: np.ndarray,
                             g1: np.ndarray, g2: np.ndarray) -> tuple[Polynomial, Polynomial]:
    n = len(f3)
    cubic, quartic = {}, {}
    rho = f3 - g1 + g1.transpose(0, 2, 1) + g1.transpose(2, 0, 1)
    sigma = f3 - g1 - g1.transpose(0, 2, 1) - g1.transpose(2, 0, 1)
    for i, j, k in np.argwhere(sigma != 0):
        value = sigma[i, j, k] / (12 * sqrt(2))
        _put(cubic, n, value, (("x", i), ("x", j), ("x", k)))
        _put(cubic, n, value, (("y", i), ("y", j), ("y", k)))
    for i, j, k in np.argwhere(rho != 0):
        value = rho[i, j, k] / (4 * sqrt(2))
        _put(cubic, n, value, (("x", i), ("x", j), ("y", k)))
        _put(cubic, n, value, (("x", k), ("y", j), ("y", i)))
    for i, j in product(range(n), repeat=2):
        _put(cubic, n, rho[i, j, j] / (4 * sqrt(2)), (("Q", i),))
        _put(quartic, n, np.sum(f4[i, j, np.arange(n), np.arange(n)]) / 16,
             (("Q", i), ("Q", j)))
    # Symmetry-adapted semidiagonal fields are sparse.  Enumerating n**4
    # tensor slots made property transport dominate the electronic work for
    # medium molecules, even though _put discarded almost every coefficient.
    # Visit each populated force or metric coefficient directly; the two
    # contributions remain additive when they share an index tuple.
    # Coordinate factors commute: collect all ordered tensor contributions
    # before normal ordering. This is an exact sum, including nonsymmetric
    # input tensors, rather than a coefficient-size or symmetry approximation.
    indices = np.argwhere(f4 != 0)
    if len(indices):
        canonical = np.sort(indices, axis=1)
        codes = np.ravel_multi_index(canonical.T, f4.shape)
        unique, inverse = np.unique(codes, return_inverse=True)
        coefficients = np.bincount(inverse, weights=f4[tuple(indices.T)])
        indices = np.array(np.unravel_index(unique, f4.shape)).T
    else:
        coefficients = ()
    for (i, j, k, ell), coefficient in zip(indices, coefficients):
        _put(quartic, n, coefficient / 96,
             (("Q", i), ("Q", j), ("Q", k), ("Q", ell)))
    for i, j, k, ell in np.argwhere(g2 != 0):
        coefficient = -g2[i, j, k, ell] / 16
        _put(quartic, n, coefficient,
             (("R", i), ("Q", k), ("Q", ell), ("R", j)))
        for delta, sign, factors in (
            (i == k, -1, (("Q", ell), ("R", j))),
            (i == ell, -1, (("Q", k), ("R", j))),
            (i == j, -1, (("Q", k), ("Q", ell))),
            (k == ell, 1, (("R", i), ("R", j))),
            (k == j, 1, (("R", i), ("Q", ell))),
            (ell == j, 1, (("R", i), ("Q", k))),
        ):
            if delta:
                _put(quartic, n, sign * coefficient, factors)
    # Additive constants in H2 (including V_g at equilibrium) commute with
    # every property and therefore do not enter these transition coefficients.
    return _nonzero(cubic), _nonzero(quartic)


def _property_coefficients(first: np.ndarray, second: np.ndarray,
                           third: np.ndarray, momentum: bool) -> tuple[Polynomial, ...]:
    n = first.shape[1]
    p0, p1, p2 = {}, {}, {}
    for i in range(n):
        _put(p0, n, first[:, i] * (1j if momentum else 1) / sqrt(2),
             (("R" if momentum else "Q", i),))
    for i, j in product(range(n), repeat=2):
        if momentum:
            _put(p1, n, 0.5j * second[:, i, j], (("R", i), ("Q", j)))
        else:
            _put(p1, n, second[:, i, j] / 4, (("Q", i), ("Q", j)))
            if i == j:
                _put(p1, n, second[:, i, j] / 4, ())
    for i, j, k in product(range(n), repeat=3):
        if momentum:
            value = 1j * third[:, i, j, k] / (4 * sqrt(2))
            _put(p2, n, value, (("R", i), ("Q", j), ("Q", k)))
            if j == k:
                _put(p2, n, value, (("R", i),))
        else:
            value = third[:, i, j, k] / (12 * sqrt(2))
            _put(p2, n, value, (("Q", i), ("Q", j), ("Q", k)))
            for delta, mode in ((i == j, k), (i == k, j), (j == k, i)):
                if delta:
                    _put(p2, n, value, (("Q", mode),))
    return _nonzero(p0), _nonzero(p1), _nonzero(p2)


@dataclass(frozen=True)
class ReducedTransitionField:
    """Field-only rational coefficients reusable for any property surface."""

    frequencies_cm: np.ndarray
    first_generator: Polynomial
    second_generator: Polynomial


def prepare_reduced_transition_field(
    frequencies: np.ndarray, f3: np.ndarray, f4: np.ndarray,
    g1: np.ndarray, g2: np.ndarray, *, denominator_tolerance_cm: float = 1e-6,
) -> ReducedTransitionField:
    """Reduce a nonresonant modal field to explicit rational coefficients."""
    cubic, quartic = _hamiltonian_coefficients(f3, f4, g1, g2)
    first = {key: -value for key, value in _frequency_inverse(
        cubic, frequencies, denominator_tolerance_cm).items()}
    effective_second = quartic.copy()
    for key, value in _paired_scalar_contraction(cubic, first).items():
        _add(effective_second, key, value / 2)
    second = {key: -value for key, value in _frequency_inverse(
        _nonzero(effective_second), frequencies, denominator_tolerance_cm).items()}
    return ReducedTransitionField(np.asarray(frequencies).copy(), first, second)


def reduced_property_coefficients(
    field: ReducedTransitionField, first: np.ndarray, second: np.ndarray,
    third: np.ndarray, *, momentum: bool = False,
) -> tuple[Polynomial, ...]:
    """Evaluate the expanded SI coefficient equations through second order."""
    p0, p1, p2 = _property_coefficients(first, second, third, momentum)
    correction0 = _contract(p0, field.first_generator)
    t1 = p1.copy()
    for key, value in correction0.items():
        _add(t1, key, value)
    t2 = p2.copy()
    for scale, values in (
        (1, _contract(p1, field.first_generator)),
        (1, _contract(p0, field.second_generator)),
        (0.5, _contract(correction0, field.first_generator)),
    ):
        for key, value in values.items():
            _add(t2, key, scale * value)
    return p0, _nonzero(t1), _nonzero(t2)


@lru_cache(maxsize=32768)
def _occupation_transition_terms(initial, final):
    """Nonzero normal-ordered monomials for fixed Fock endpoints.

    Leave r quanta untouched: u=final-r, v=initial-r and the exact ladder
    factor is sqrt(initial! final!)/r!. Only modes occupied at both endpoints
    contribute choices. This is an integer identity, not an excitation cutoff.
    """
    shared = [(i, min(a, b)) for i, (a, b) in enumerate(zip(initial, final, strict=True))
              if a and b]
    if integer_product(q + 1 for _, q in shared) > 64:
        # Select the direct coefficient traversal when Fock lookup would be
        # more costly. Both algorithms evaluate the same complete expression.
        return None
    base = sqrt(integer_product(factorial(a) * factorial(b)
                                for a, b in zip(initial, final, strict=True)))
    terms = []
    for retained in product(*(range(q + 1) for _, q in shared)):
        u, v = list(final), list(initial)
        denominator = 1
        for (mode, _), count in zip(shared, retained, strict=True):
            u[mode] -= count
            v[mode] -= count
            denominator *= factorial(count)
        terms.append(((bytes(u), bytes(v)), base / denominator))
    return tuple(terms)


def evaluate_reduced_transition(
    coefficients: Polynomial, initial: tuple[int, ...], final: tuple[int, ...],
    components: int,
) -> np.ndarray:
    """Specialized cold/hot-band formulas, plus arbitrary-occupation factorials."""
    zero = bytes(len(initial))
    initial_power = bytes(initial)
    final_power = bytes(final)
    value = np.zeros(components, complex)
    if sum(initial) == 0:
        factor = sqrt(np.prod([factorial(v) for v in final]))
        return factor * coefficients.get((final_power, zero), value)
    if sum(initial) == 1:
        factor = sqrt(np.prod([factorial(v) for v in final]))
        mode = initial.index(1)
        value += coefficients.get((final_power, initial_power), 0)
        if final[mode]:
            lower = list(final)
            lower[mode] -= 1
            value += coefficients.get((bytes(lower), zero), 0)
        return factor * value
    lookup = _occupation_transition_terms(tuple(initial), tuple(final))
    if lookup is not None:
        for key, factor in lookup:
            coefficient = coefficients.get(key)
            if coefficient is not None:
                value += factor * coefficient
        return value
    for (u, v), coefficient in coefficients.items():
        if any(b > a for a, b in zip(initial, v, strict=True)):
            continue
        if any(a - b + c != d for a, b, c, d in zip(initial, v, u, final, strict=True)):
            continue
        factor = 1.0
        for a, b, c in zip(initial, v, final, strict=True):
            factor *= sqrt(factorial(a) * factorial(c)) / factorial(a - b)
        value += factor * coefficient
    return value
