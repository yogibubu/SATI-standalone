"""Normal-ordered operators and occupation-dependent transition amplitudes.

This algebra evaluates supplied operators; it does not infer an HDC wave
operator from scalar energies. No state-specific eigenvectors are constructed.
"""
from dataclasses import dataclass
from itertools import product
from types import MappingProxyType

import numpy as np


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class LadderPolynomial:
    """Sum C[r,s] product_i (a_i^dagger)^r_i a_i^s_i, with no hidden factorial.

    Coefficients are scalar complex numbers. Use one polynomial per property
    component. All mode axes refer to the explicit coordinate_id.
    """
    mode_count: int
    terms: dict
    coordinate_id: str

    def __post_init__(self):
        n = _integer(self.mode_count, "mode_count", 1)
        if not isinstance(self.coordinate_id, str) or not self.coordinate_id.strip():
            raise ValueError("nonempty coordinate_id required")
        cleaned = {}
        for key, value in self.terms.items():
            if len(key) != 2 or any(len(powers) != n for powers in key):
                raise ValueError("each key must contain creation and annihilation exponent vectors")
            r, s = (tuple(_integer(k, "exponent") for k in powers) for powers in key)
            if max((*r, *s)) > np.iinfo(np.int64).max // 4:
                raise ValueError("exponents exceed safe integer range")
            c = complex(value)
            if not np.isfinite(c):
                raise ValueError("finite scalar coefficients required")
            if c != 0:
                cleaned[r, s] = c
        object.__setattr__(self, "terms", MappingProxyType(cleaned))

    def _compatible(self, other):
        if not isinstance(other, LadderPolynomial):
            raise TypeError("LadderPolynomial required")
        if (self.mode_count, self.coordinate_id) != (other.mode_count, other.coordinate_id):
            raise ValueError("operators must share mode count and coordinate_id")

    def scaled(self, coefficient):
        return LadderPolynomial(self.mode_count,
            {key: coefficient * value for key, value in self.terms.items()}, self.coordinate_id)

    def __add__(self, other):
        self._compatible(other)
        terms = dict(self.terms)
        for key, value in other.terms.items():
            terms[key] = terms.get(key, 0j) + value
        return LadderPolynomial(self.mode_count, terms, self.coordinate_id)

    def adjoint(self):
        return LadderPolynomial(self.mode_count,
            {(s, r): c.conjugate() for (r, s), c in self.terms.items()}, self.coordinate_id)

    def compose(self, other, *, maximum_terms=100000, maximum_contractions=1000000):
        """Exact boson contractions; caps raise instead of dropping terms.

        a^s (a^dagger)^u = sum_k binom(s,k) binom(u,k) k!
                                  (a^dagger)^(u-k) a^(s-k).
        """
        from .algebraic_local_x import _factor
        self._compatible(other)
        _integer(maximum_terms, "maximum_terms", 1)
        _integer(maximum_contractions, "maximum_contractions", 1)
        result = {}; actions = 0
        for (r, s), left in self.terms.items():
            for (u, v), right in other.terms.items():
                ranges = [range(min(a, b) + 1) for a, b in zip(s, u)]
                count = 1
                for choices in ranges:
                    count *= len(choices)
                actions += count
                if actions > maximum_contractions:
                    raise MemoryError("normal-order contraction budget exceeded")
                for k in product(*ranges):
                    factor = _factor(s, u, k)
                    key = (tuple(a+b-c for a, b, c in zip(r, u, k)),
                           tuple(a+b-c for a, b, c in zip(s, v, k)))
                    value = result.get(key, 0j) + left * right * factor
                    if value == 0:
                        result.pop(key, None)
                    else:
                        result[key] = value
                    if len(result) > maximum_terms:
                        raise MemoryError("normal-ordered term budget exceeded")
        return LadderPolynomial(self.mode_count, result, self.coordinate_id)

    def commutator(self, other, **budgets):
        return self.compose(other, **budgets) + other.compose(self, **budgets).scaled(-1)


def ladder_polynomial_from_qp(mode_count, terms, *, coordinate_id,
                              maximum_terms=100000, maximum_contractions=1000000):
    """Convert ordered (coefficient, ((kind, mode), ...)) q/p monomials.

    q=(a^dagger+a)/sqrt(2), p=i(a^dagger-a)/sqrt(2). The order of
    noncommuting q/p factors is preserved. Input coefficients already include
    the Taylor multiplicity factorials, as in HDCPropertyExpansion.terms().
    """
    from .algebraic_local_x import _operator
    n = _integer(mode_count, "mode_count", 1)
    zero = (0,) * n
    result = LadderPolynomial(n, {}, coordinate_id)
    budgets = dict(maximum_terms=maximum_terms, maximum_contractions=maximum_contractions)
    for coefficient, operators in terms:
        term = LadderPolynomial(n, {(zero, zero): coefficient}, coordinate_id)
        for kind, mode in operators:
            mode = _integer(mode, "mode")
            if mode >= n or kind not in ("q", "p"):
                raise ValueError("q/p kind and a valid mode index required")
            factor = LadderPolynomial(n, _operator(n, kind, mode), coordinate_id)
            term = term.compose(factor, **budgets)
        result = result + term
        if len(result.terms) > maximum_terms:
            raise MemoryError("normal-ordered property term budget exceeded")
    return result


def contact_property_orders(orders, generator_first, generator_second, **budgets):
    """Return O0,O1_eff,O2_eff for e^-S O e^S through perturbative order two.

    S=lambda*S1+lambda^2*S2 and O=O0+lambda*O1+lambda^2*O2.
    Anti-Hermitian generators are supplied by their theory owner; no
    regularization, resonance classification, or HDC derivation is inferred.
    """
    o0, o1, o2 = orders
    for value in (o1, o2, generator_first, generator_second):
        o0._compatible(value)
    for s in (generator_first, generator_second):
        residual = s + s.adjoint()
        scale = max((abs(v) for v in s.terms.values()), default=0.)
        if max((abs(v) for v in residual.terms.values()), default=0.) > 1e-12 * max(1., scale):
            raise ValueError("anti-Hermitian contact generators required")
    first = o0.commutator(generator_first, **budgets)
    return (o0, o1 + first,
            o2 + o1.commutator(generator_first, **budgets)
            + o0.commutator(generator_second, **budgets)
            + first.commutator(generator_first, **budgets).scaled(.5))


@dataclass(frozen=True)
class OccupationTransitions:
    initial_rows: np.ndarray
    final_occupations: np.ndarray
    amplitudes: np.ndarray

    @property
    def strengths(self):
        values = np.sum(abs(self.amplitudes)**2, axis=1)
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("transition strengths exceed floating-point range")
        return values


class OccupationTransitionOperator:
    """Compile by occupation change; sum amplitudes before squaring.

    Memory scales with the supplied batch and connected shifts, never with
    the complete oscillator space. Supply Cartesian components for an
    isotropically summed dipole strength; arbitrary component lists are also
    accepted with their contraction left to the caller.
    """
    def __init__(self, components, *, units, name="property"):
        self.components = tuple(components)
        if not self.components:
            raise ValueError("at least one property component required")
        first = self.components[0]
        if not isinstance(first, LadderPolynomial):
            raise TypeError("LadderPolynomial components required")
        for component in self.components:
            first._compatible(component)
        if not isinstance(units, str) or not units.strip():
            raise ValueError("explicit property units required")
        self.mode_count = first.mode_count
        self.coordinate_id = first.coordinate_id
        self.units = units; self.name = name
        groups = {}
        for c, component in enumerate(self.components):
            for (r, s), coefficient in component.terms.items():
                delta = tuple(a-b for a, b in zip(r, s))
                active = tuple((i, a, b) for i, (a, b) in enumerate(zip(r, s)) if a or b)
                groups.setdefault(delta, []).append((c, coefficient, active))
        self._groups = tuple(sorted(groups.items()))

    @classmethod
    def from_property_expansion(cls, expansion, **budgets):
        """Reuse the existing general scalar/vector/tensor Taylor interface."""
        from .hdc_properties import HDCPropertyExpansion
        if not isinstance(expansion, HDCPropertyExpansion):
            raise TypeError("HDCPropertyExpansion required")
        count = int(np.prod(expansion.component_shape)) if expansion.component_shape else 1
        terms = [[] for _ in range(count)]
        for coefficient, operators in expansion.terms():
            for i, value in enumerate(np.asarray(coefficient).reshape(-1)):
                if value:
                    terms[i].append((complex(value), operators))
        components = [ladder_polynomial_from_qp(expansion.mode_count, entries,
            coordinate_id=expansion.coordinate_id, **budgets) for entries in terms]
        return cls(components, units=expansion.units, name=expansion.name)

    def transitions(self, occupations, *, maximum_channels=1000000,
                    maximum_dense_bytes=256*1024**2):
        a = np.asarray(occupations)
        if (a.ndim != 2 or a.shape[1] != self.mode_count or a.dtype.kind not in "iu"
                or np.any(a < 0)):
            raise ValueError("nonnegative integer occupation rows required")
        _integer(maximum_channels, "maximum_channels", 1)
        _integer(maximum_dense_bytes, "maximum_dense_bytes", 1)
        if a.size and np.max(a) > np.iinfo(np.int64).max // 2:
            raise ValueError("occupations exceed safe integer range")
        a = a.astype(np.int64, copy=False)
        if len(a) * len(self._groups) > maximum_channels:
            raise MemoryError("connected-channel budget exceeded; use smaller occupation batches")
        # Includes Python staging objects as well as output and working arrays.
        if len(a) * max(1, len(self._groups)) * (512 + 64*self.mode_count + 128*len(self.components)) > maximum_dense_bytes:
            raise MemoryError("connected-channel working-set budget exceeded; use smaller occupation batches")
        rows = []; states = []; values = []
        for delta, entries in self._groups:
            target = a + np.asarray(delta, dtype=np.int64)
            valid_target = np.all(target >= 0, axis=1)
            amplitudes = np.zeros((len(a), len(self.components)), complex)
            for c, coefficient, active in entries:
                factor = np.ones(len(a))
                for mode, r, s in active:
                    n = a[:, mode]
                    factor[n < s] = 0.
                    # Products of square roots avoid factorial overflow.
                    for k in range(s):
                        factor *= np.sqrt(np.maximum(n-k, 0).astype(float))
                    for k in range(1, r+1):
                        factor *= np.sqrt(np.maximum(n-s+k, 0).astype(float))
                amplitudes[:, c] += coefficient * factor
            if not np.all(np.isfinite(amplitudes)):
                raise FloatingPointError("occupation amplitudes exceed floating-point range")
            selected = valid_target & np.any(amplitudes != 0, axis=1)
            rows.extend(np.flatnonzero(selected))
            states.extend(target[selected]); values.extend(amplitudes[selected])
        return OccupationTransitions(np.asarray(rows, dtype=np.int64),
            np.asarray(states, dtype=np.int64).reshape(-1, self.mode_count),
            np.asarray(values, dtype=complex).reshape(-1, len(self.components)))


def occupation_transition_lines(energy_model, operator, occupations, *,
                                maximum_total_quanta, maximum_channels=1000000,
                                maximum_dense_bytes=256*1024**2):
    """Bind connected channels to the existing X/Y polynomial energy model.

    Positive signed gaps denote emission. The explicit finite domain applies
    to both endpoints. Excluded endpoint strengths are reported, not silently
    included in a renormalized branching fraction. The caller owns the
    physical-domain certificate and statistical weights.
    """
    from .hdc_polynomial import HDCPolynomialEnergy
    if not isinstance(energy_model, HDCPolynomialEnergy):
        raise TypeError("HDCPolynomialEnergy required")
    if not isinstance(operator, OccupationTransitionOperator):
        raise TypeError("OccupationTransitionOperator required")
    if energy_model.coordinate_id != operator.coordinate_id:
        raise ValueError("energy and property coordinate_id must match explicitly")
    cutoff = _integer(maximum_total_quanta, "maximum_total_quanta")
    transitions = operator.transitions(occupations, maximum_channels=maximum_channels,
        maximum_dense_bytes=maximum_dense_bytes)
    initial = np.asarray(occupations)
    if np.any(initial.sum(axis=1, dtype=object) > cutoff):
        raise ValueError("initial occupation outside declared model domain")
    initial_energy = energy_model.energies(initial)
    inside = transitions.final_occupations.sum(axis=1, dtype=object) <= cutoff
    omitted = np.bincount(transitions.initial_rows[~inside],
        weights=transitions.strengths[~inside], minlength=len(initial))
    rows = transitions.initial_rows[inside]
    final = transitions.final_occupations[inside]
    amplitudes = transitions.amplitudes[inside]
    final_energy = energy_model.energies(final)
    return dict(initial_rows=rows, final_occupations=final, amplitudes=amplitudes,
        strengths=np.sum(abs(amplitudes)**2, axis=1),
        initial_energies_cm=initial_energy[rows], final_energies_cm=final_energy,
        signed_gaps_cm=initial_energy[rows]-final_energy,
        excluded_endpoint_strengths=omitted,
        diagnostics=dict(schema="matrix.trinity.occupation_transition_lines.v1",
            coordinate_id=operator.coordinate_id, units=operator.units,
            maximum_total_quanta=cutoff, state_specific_diagonalization=False,
            amplitudes_summed_before_squaring=True,
            supplied_operator_only=True, hdc_generator_derived=False,
            statistical_weights_applied=False, physical_domain_certificate_checked=False))


def occupation_dipole_emission(lines, *, dipole_to_coulomb_metre):
    """Vacuum electric-dipole A rates and conditional resolved-state branches.

    Supply an explicit SI conversion, e.g. scipy.constants.e * Bohr_radius for
    dipoles in atomic units. Omitted endpoints remain a separate strength
    diagnostic; their omission prevents a completeness claim for the rates.
    No nuclear-spin/rotational degeneracy factors are inferred.
    """
    from scipy.constants import c, epsilon_0, hbar
    if (not np.isfinite(dipole_to_coulomb_metre) or dipole_to_coulomb_metre <= 0
            or lines['amplitudes'].ndim != 2 or lines['amplitudes'].shape[1] != 3):
        raise ValueError("three Cartesian dipole components and a positive SI conversion required")
    selected = lines['signed_gaps_cm'] > 0
    rows = lines['initial_rows'][selected]
    omega = 2*np.pi*c*100*lines['signed_gaps_cm'][selected]
    rates = omega**3 * lines['strengths'][selected] * dipole_to_coulomb_metre**2 / (3*np.pi*epsilon_0*hbar*c**3)
    if not np.all(np.isfinite(rates)):
        raise FloatingPointError("radiative rates exceed floating-point range")
    total = np.bincount(rows, weights=rates, minlength=len(lines['excluded_endpoint_strengths']))
    return dict(initial_rows=rows, final_occupations=lines['final_occupations'][selected],
        wavenumbers_cm=lines['signed_gaps_cm'][selected], rates_s_inverse=rates,
        conditional_branching_fractions=np.divide(rates, total[rows], out=np.zeros_like(rates), where=total[rows]>0),
        total_retained_rates_s_inverse=total,
        excluded_endpoint_strengths=lines['excluded_endpoint_strengths'].copy(),
        completeness_certified=False, rotational_resolution_included=False)


def occupation_strength_observable(energy_model, operator, frequency_bin_edges_cm, *,
        maximum_total_quanta, direction="emission", maximum_dense_bytes=256*1024**2):
    """Callback for existing frozen REWL observables, without modifying its walk.

    Output columns are strength bins, omitted-endpoint strength and
    out-of-frequency-range strength. Frequencies are evaluated as exact
    polynomial differences before binning. This returns per-visit strengths,
    not a DOS or canonical spectrum. Independent-ladder errors remain the
    sampler's responsibility. Static zero-gap moments are excluded.
    """
    from .hdc_polynomial import HDCPolynomialEnergy
    if not isinstance(energy_model, HDCPolynomialEnergy) or not isinstance(operator, OccupationTransitionOperator):
        raise TypeError("HDCPolynomialEnergy and OccupationTransitionOperator required")
    if (energy_model.coordinate_id != operator.coordinate_id
            or len(energy_model.fundamentals_cm) != operator.mode_count):
        raise ValueError("energy and operator must share the same modal coordinate identity")
    _integer(maximum_total_quanta, "maximum_total_quanta")
    edges = np.array(frequency_bin_edges_cm, dtype=float, copy=True)
    if (edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges))
            or edges[0] < 0 or np.any(np.diff(edges) <= 0)):
        raise ValueError("finite increasing nonnegative frequency edges required")
    if direction not in ("emission", "absorption"):
        raise ValueError("direction must be emission or absorption")
    _integer(maximum_dense_bytes, "maximum_dense_bytes", 1)
    edges.setflags(write=False)
    channels = max(1, len(operator._groups))
    cost = channels * (512+64*operator.mode_count+128*len(operator.components))
    chunk = max(1, maximum_dense_bytes//(4*cost))

    def measure(occupations):
        initial = np.asarray(occupations)
        if initial.ndim != 2:
            raise ValueError("occupation batch required")
        columns = len(edges)+1
        if 8*len(initial)*columns > maximum_dense_bytes//2:
            raise MemoryError("transition observable output exceeds working-set budget")
        output = np.zeros((len(initial), columns))
        for start in range(0, len(initial), chunk):
            block = initial[start:start+chunk]
            lines = occupation_transition_lines(energy_model, operator, block,
                maximum_total_quanta=maximum_total_quanta,
                maximum_channels=max(1, len(block)*channels),
                maximum_dense_bytes=maximum_dense_bytes//2)
            output[start:start+len(block), -2] = lines['excluded_endpoint_strengths']
            gaps = lines['signed_gaps_cm'] * (1 if direction == "emission" else -1)
            selected = gaps > 0
            bins = np.searchsorted(edges, gaps, side="right")-1
            bins[gaps == edges[-1]] = len(edges)-2
            inside = selected & (bins >= 0) & (bins < len(edges)-1)
            np.add.at(output, (start+lines['initial_rows'][inside], bins[inside]), lines['strengths'][inside])
            omitted = selected & ~inside
            np.add.at(output, (start+lines['initial_rows'][omitted], np.full(np.sum(omitted), columns-1)), lines['strengths'][omitted])
        return output
    return measure
