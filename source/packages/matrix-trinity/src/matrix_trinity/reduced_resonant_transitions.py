"""Pair-selective reduced property formulas, including resonant polyads.

All sums run over normal-polynomial tensor coefficients. Products use the
explicit binomial/factorial coefficients in the SI, with occupation-dependent
pair masks. No oscillator matrix, operator action, or virtual-state basis is
built. The separate full contraction is a test reference only.
"""
from __future__ import annotations

from functools import lru_cache
from operator import add, sub

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - portable source fallback
    def njit(*_args, **_kwargs):
        def decorate(function):
            return function
        return decorate

from .reduced_transition_polynomials import (
    _add, _hamiltonian_coefficients, _nonzero, _property_coefficients, _has_nonzero,
    evaluate_reduced_transition,
)


@njit(cache=True)
def _compiled_product_weight_arrays(a, b, c, d):
    """Compiled mixed-radix enumeration of the exact integer contraction."""
    n = len(a)
    total = 1
    for i in range(n):
        total *= min(b[i], c[i]) + 1
    left = np.empty((total, n), dtype=np.uint8)
    right = np.empty((total, n), dtype=np.uint8)
    weights = np.ones(total, dtype=np.int64)
    for row in range(total):
        remainder = row
        weight = 1
        for i in range(n):
            radix = min(b[i], c[i]) + 1
            count = remainder % radix
            remainder //= radix
            left[row, i] = a[i] + c[i] - count
            right[row, i] = b[i] + d[i] - count
            factorial_value = 1
            for value in range(2, count + 1):
                factorial_value *= value
            choose_b = 1
            choose_c = 1
            for value in range(1, count + 1):
                choose_b = choose_b * (b[i] - value + 1) // value
                choose_c = choose_c * (c[i] - value + 1) // value
            weight *= factorial_value * choose_b * choose_c
        weights[row] = weight
    return left, right, weights


@lru_cache(maxsize=32768)
def _product_weights(a, b, c, d):
    """L_(a,b),(c,d)^(u,v) = t! binom(b,t) binom(c,t), including t=0."""
    arrays = tuple(np.frombuffer(value, dtype=np.uint8) for value in (a, b, c, d))
    left, right, weights = _compiled_product_weight_arrays(*arrays)
    return tuple(
        ((bytes(left[index]), bytes(right[index])), int(weights[index]))
        for index in range(len(weights))
    )


@lru_cache(maxsize=32768)
def _guarded_occupation(n):
    # One guard bit per byte makes all componentwise comparisons independent.
    # A deficit clears that byte's guard without borrowing into another byte.
    if any(v < 0 or v >= 128 for v in n):
        return None
    guard = int.from_bytes(b"\x80" * len(n), "little")
    return int.from_bytes(bytes(n), "little") | guard, guard


@lru_cache(maxsize=32768)
def _packed_annihilation(power):
    return int.from_bytes(power, "little")


def _annihilation_allowed(power, n):
    """Exact simultaneous occupation inequalities; no floating-point cutoff."""
    if len(power) != len(n):
        return all(v <= ni for v, ni in zip(power, n, strict=True))
    packed = _guarded_occupation(n)
    if packed is not None:
        value = _packed_annihilation(power)
        if value == 0:
            return True
        occupation, guard = packed
        if value & guard == 0:
            return ((occupation - value) & guard) == guard
    return all(v <= ni for v, ni in zip(power, n, strict=True))


def _shift(key):
    return tuple(u - v for u, v in zip(*key, strict=True))


def _plus(a, b):
    # ``map`` keeps the mode loop in C.  These two operations dominate the
    # occupation-dependent hot-band path for systems with tens of modes.
    return tuple(map(add, a, b))


def _minus(a, b):
    return tuple(map(sub, a, b))


def _groups(polynomial):
    groups = {}
    for key, value in polynomial.items():
        groups.setdefault(_shift(key), {})[key] = value
    return groups


def _parity_mask(shift):
    return sum((int(value) & 1) << index for index, value in enumerate(shift))


def _gf2_basis(masks):
    """Return a reduced binary basis for symmetry parities inferred from H."""
    basis = {}
    for mask in masks:
        value = int(mask)
        while value:
            pivot = value.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = value
                break
            value ^= basis[pivot]
    return tuple(basis[pivot] for pivot in sorted(basis, reverse=True))


def _gf2_reduce(mask, basis):
    value = int(mask)
    for vector in basis:
        value = min(value, value ^ vector)
    return value


class PairReducedField:
    """Field-only coefficients shared by every property and initial state."""

    def __init__(self, frequencies, f3, f4, g1, g2, exclusions, *, tolerance=1e-6):
        self.frequencies = np.asarray(frequencies)
        self.exclusions = {tuple(n): frozenset(map(tuple, partners))
                           for n, partners in exclusions.items()}
        self.tolerance = tolerance
        cubic, quartic = _hamiltonian_coefficients(f3, f4, g1, g2)
        self.base = {'c': _groups(cubic), 'h': _groups(quartic)}
        self.support = {'c': tuple(self.base['c']), 'h': tuple(self.base['h'])}
        for name in ('a', 'd', 'ad'):
            self.support[name] = self.support['c']
        # Integer codes are additive and unique on every shift reachable at
        # this perturbative order. They screen impossible products before
        # constructing full mode tuples or recursively reducing coefficients.
        degree=max((sum(map(abs,s)) for shifts in self.support.values() for s in shifts),default=0)
        self.shift_bits=max(1,(2*(2*degree+3)+1).bit_length())
        self.shift_codes={}
        self.shift_codes={s:self.encode_shift(s) for shifts in self.support.values() for s in shifts}
        self.support_codes={name:frozenset(self.encode_shift(s) for s in shifts)
                            for name,shifts in self.support.items()}
        self.cache = {}
        self.active_support_cache = {}
        self.coded_support_cache = {}

    def encode_shift(self, shift):
        code=self.shift_codes.get(shift)
        if code is None:
            code=sum(int(v) << (self.shift_bits*i) for i,v in enumerate(shift) if v)
            # Retain tensor supports, not every generated product shift.
            return code
        return code

    def may_have_code(self, name, code):
        support=self.support_codes.get(name)
        return support is None or code in support

    def internal(self, n, shift):
        target = _plus(n, shift)
        return target == n or target in self.exclusions.get(n, ())

    @staticmethod
    def prune(polynomial, n):
        # These terms act on the declared occupation on their right. Terms
        # whose annihilation degree exceeds it contribute exactly zero.
        return {key: value for key, value in polynomial.items()
                if _annihilation_allowed(key[1], n)
                and _has_nonzero(value)}

    def divide(self, polynomial, n, shift):
        polynomial = self.prune(polynomial, n)
        if not polynomial:
            return {}
        amplitude = np.asarray(evaluate_reduced_transition(polynomial, n, _plus(n, shift), 1)).item()
        if abs(amplitude) <= 1e-12:
            return {}
        denominator = -float(np.dot(self.frequencies, shift))
        if abs(denominator) <= self.tolerance:
            raise ValueError(f'unresolved curvilinear transition denominator for {n}, {_plus(n, shift)}')
        return {key: value / denominator for key, value in polynomial.items()}

    def multiply(self, left, right, n, shift, lookup=None, support=None):
        """Explicit coefficient product with the right-shifted left mask."""
        lookup = self.terms if lookup is None else lookup
        support = self.support if support is None else support
        output = {}
        # Query only net tensor differences that can produce this transition.
        # Select the smaller known support; b and normalization need no
        # global degree-six tensor expansion.
        lset, rset = support.get(left), support.get(right)
        owner = getattr(lookup, "__self__", None)
        if owner is not None and hasattr(owner, "active_support"):
            active = owner.active_support(right, n)
            if active is not None:
                rset = active
        if lset is None and rset is None:
            raise ValueError('a reduced product requires one finite tensor support')
        screening=owner if owner is not None and hasattr(owner,'may_have_code') else self
        code=self.encode_shift(shift)
        right_side = rset is not None and (lset is None or len(rset) <= len(lset))
        selected = rset if right_side else lset
        # Retain integer codes with the immutable active support. Avoid millions
        # of repeated Python method calls and big-integer encodings per source.
        cache = screening.coded_support_cache
        key = (right if right_side else left, n if right_side else None)
        coded = cache.get(key)
        if coded is None:
            coded = tuple((s, self.encode_shift(s)) for s in selected)
            cache[key] = coded
        opposite = left if right_side else right
        allowed = screening.support_codes.get(opposite)
        if allowed is not None:
            matching = (s for s, sc in coded if code-sc in allowed)
        else:
            matching = (s for s, sc in coded if screening.may_have_code(opposite, code-sc))
        splits = (((_minus(shift, s), s) if right_side else (s, _minus(shift, s)))
                  for s in matching)
        for dl, dr in splits:
            middle = _plus(n, dr)
            if min(middle) < 0:
                continue
            right_terms = lookup(right, n, dr)
            if not right_terms:
                continue
            left_terms = lookup(left, middle, dl)
            for (a, b), x in left_terms.items():
                for (c, d), y in right_terms.items():
                    for key, weight in _product_weights(a, b, c, d):
                        if _annihilation_allowed(key[1], n):
                            _add(output, key, weight * x * y)
        return _nonzero(output)

    def active_support(self, name, n):
        """Shifts of a right operator that can act on occupation ``n``."""
        key = (name, n)
        if key in self.active_support_cache:
            return self.active_support_cache[key]
        if name in self.base:
            out = tuple(
                shift for shift, polynomial in self.base[name].items()
                if any(_annihilation_allowed(term[1], n)
                       and _has_nonzero(value)
                       for term, value in polynomial.items())
            )
        elif name in {"a", "d"}:
            out = tuple(
                shift for shift in self.active_support("c", n)
                if self.internal(n, shift) == (name == "d")
            )
        else:
            return None
        self.active_support_cache[key] = out
        return out

    def terms(self, name, n, shift):
        key = (name, n, shift)
        if key in self.cache:
            return self.cache[key]
        if min(n) < 0 or min(_plus(n, shift)) < 0:
            return {}
        if name in self.base:
            out = self.prune(self.base[name].get(shift, {}), n)
        elif name == 'a':
            out = {} if self.internal(n, shift) else self.divide(self.terms('c', n, shift), n, shift)
        elif name == 'd':
            out = self.terms('c', n, shift) if self.internal(n, shift) else {}
        elif name in ('ad', 'bd'):
            other = self.terms(name[0], _plus(n, shift), tuple(-x for x in shift))
            out = self.prune({(v, u): np.conjugate(value) for (u, v), value in other.items()}, n)
        elif name == 'b':
            if self.internal(n, shift):
                out = {}
            else:
                out = self.terms('h', n, shift).copy()
                for sign, left, right in ((1, 'c', 'a'), (-1, 'a', 'd')):
                    for k, value in self.multiply(left, right, n, shift).items():
                        _add(out, k, sign * value)
                out = self.divide(out, n, shift)
        elif name == 'norm':
            out = self.multiply('ad', 'a', n, shift) if self.internal(n, shift) else {}
        else:
            raise ValueError(f'unknown reduced field coefficient {name}')
        # Empty lookups dominate large symmetry-adapted systems.  Retaining
        # their full occupation/shift tuple keys costs far more memory than
        # recomputing the cheap absence test.
        if out:
            self.cache[key] = out
        return out


class PairReducedProperty:
    """Explicit order-zero/one/two coefficients for one property surface."""

    def __init__(self, field, first, second, third, *, momentum=False):
        self.field = field
        self.components = len(first)
        self.base = {f'p{rank}': _groups(values) for rank, values in enumerate(
            _property_coefficients(first, second, third, momentum))}
        self.support = dict(field.support)
        self.support.update({name: tuple(groups) for name, groups in self.base.items()})
        self.support_codes=dict(field.support_codes)
        self.support_codes.update({name:frozenset(field.encode_shift(s) for s in shifts)
                                   for name,shifts in self.support.items() if name in self.base})
        self.p0a_support_cache={}
        self.coded_support_cache = {}
        self._p0a_possible = lru_cache(maxsize=262144)(self._p0a_possible)
        self.cache = {}
        self.moment_cache = {}
        # Discover Abelian selection rules directly from the exact sparse
        # tensors.  Scalar-Hamiltonian shifts span the totally symmetric
        # parity subspace; property shifts identify the allowed cosets.  This
        # avoids evaluating identically zero transitions without requiring
        # point-group labels from a particular coordinate provider.
        h_basis = _gf2_basis(
            _parity_mask(shift)
            for name in ("c", "h") for shift in field.support[name]
        )
        self.allowed_property_cosets = frozenset(
            _gf2_reduce(_parity_mask(shift), h_basis)
            for name in ("p0", "p1", "p2") for shift in self.support[name]
        )
        self.h_symmetry_basis = h_basis

    def may_have_code(self, name, code):
        support=self.support_codes.get(name)
        if support is not None:
            return code in support
        if name=='p0a':
            answer=self.p0a_support_cache.get(code)
            if answer is None:
                answer=self._p0a_possible(code)
                if len(self.p0a_support_cache)>=32768:self.p0a_support_cache.clear()
                self.p0a_support_cache[code]=answer
            return answer
        return True

    def _p0a_possible(self, code):
        return any(code-p in self.support_codes['a'] for p in self.support_codes['p0'])

    def active_support(self, name, n):
        key = ("support", name, n)
        if key in self.cache:
            return self.cache[key]
        if name in self.base:
            out = tuple(
                shift for shift, polynomial in self.base[name].items()
                if any(_annihilation_allowed(term[1], n)
                       and _has_nonzero(value)
                       for term, value in polynomial.items())
            )
        else:
            out = self.field.active_support(name, n)
            if out is None:
                return None
        self.cache[key] = out
        return out

    def terms(self, name, n, shift):
        if name in self.base:
            return self.field.prune(self.base[name].get(shift, {}), n)
        if name == 'p0a':
            key = (n, shift)
            if key not in self.cache:
                values = self.multiply('p0', 'a', n, shift)
                if values:
                    self.cache[key] = values
                return values
            return self.cache[key]
        return self.field.terms(name, n, shift)

    def multiply(self, left, right, n, shift):
        return self.field.multiply(left, right, n, shift, self.terms, self.support)

    def moment(self, initial, final):
        cache_key = (tuple(initial), tuple(final))
        cached = self.moment_cache.get(cache_key)
        if cached is not None:
            return cached
        shift = _minus(final, initial)
        # Apply the exact symmetry rule inferred from the sparse Hamiltonian
        # and property tensors before constructing any coefficient products.
        # No quantum-distance cutoff is used: the second-order contact
        # transformation can connect more than three total quanta.
        if (_gf2_reduce(_parity_mask(shift), self.h_symmetry_basis)
                not in self.allowed_property_cosets):
            zero = np.zeros(self.components, complex)
            result = (zero.copy(), zero.copy(), zero.copy())
            self.moment_cache[cache_key] = result
            return result
        out = []
        orders = (
            (),
            ((1, 'ad', 'p0'), (1, 'p0', 'a')),
            ((1, 'ad', 'p1'), (1, 'p1', 'a'),
             (1, 'bd', 'p0'), (1, 'p0', 'b'), (1, 'ad', 'p0a'),
             (-.5, 'norm', 'p0'), (-.5, 'p0', 'norm')),
        )
        for rank, additions in enumerate(orders):
            if (sum(final) - sum(initial) - rank - 1) % 2:
                out.append(np.zeros(self.components, complex))
                continue
            coefficients = self.terms(f'p{rank}', initial, shift).copy()
            for scale, left, right in additions:
                for key, value in self.multiply(left, right, initial, shift).items():
                    _add(coefficients, key, scale * value)
            out.append(evaluate_reduced_transition(coefficients, initial, final, self.components))
        result = tuple(out)
        self.moment_cache[cache_key] = result
        return result
