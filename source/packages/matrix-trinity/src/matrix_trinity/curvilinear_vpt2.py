"""Second-order vibrational perturbation theory in nonredundant SONIC modes.

The operators follow Mendolicchio's curvilinear Hamiltonian (2026, eqs.
11--19). All input derivatives are with respect to the same SONIC coordinates:
F in hartree, and the contravariant Wilson metric G in inverse amu. The
equilibrium Podolsky term is supplied by the full rovibrational metric owner.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
from itertools import combinations, combinations_with_replacement, permutations, product
from math import factorial, sqrt
from time import perf_counter
from typing import Callable, Iterator, Mapping

import numpy as np

from matrix_gf.harmonic import CM_PER_HARTREE
from matrix_qm import HESSIAN_EIGENVALUE_TO_CM
from .local_quadratic import _maximum_overlap_assignment
from .resonance_screening import (
    PROFILE_ID as INTENSITY_RESONANCE_PROFILE, IntensityResonanceCriteria,
    pair_decision,
)


@dataclass(frozen=True)
class SonicVPT2Input:
    harmonic_sonic: np.ndarray
    cubic_sonic: np.ndarray
    quartic_sonic: np.ndarray
    metric_sonic: np.ndarray
    metric_first_sonic: np.ndarray
    metric_second_sonic: np.ndarray
    pseudopotential_cm: float
    quartic_coverage: str = "complete"
    symmetry_actions: tuple[np.ndarray, ...] | None = None
    symmetry_tolerance: float = 1e-7


@dataclass(frozen=True)
class SonicVPT2Result:
    frequencies_cm: np.ndarray
    zero_point_cm: float
    fundamentals_cm: np.ndarray
    harmonic_energies_cm: np.ndarray
    quartic_kinetic_energies_cm: np.ndarray
    cubic_second_order_energies_cm: np.ndarray
    sonic_modes: np.ndarray
    representation: str = "nonredundant-sonic-gf"
    acquisition: dict[str, object] = field(default_factory=dict)
    x_matrix_cm: np.ndarray | None = None


@dataclass(frozen=True)
class SonicGVPT2Result:
    """Resonance polyads and assigned transition energies in SONIC modes."""

    frequencies_cm: np.ndarray
    states: tuple[tuple[int, ...], ...]
    deperturbed_cm: np.ndarray
    energies_cm: np.ndarray
    fundamentals_cm: np.ndarray
    resonances: tuple[dict[str, object], ...]
    blocks: tuple[dict[str, object], ...]
    zero_point_cm: float
    report: str
    acquisition: dict[str, object] = field(default_factory=dict)
    diagnostics: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SonicAcquiredField:
    field: SonicVPT2Input
    acquisition: dict[str, object]


def _validate_field(field: SonicVPT2Input) -> tuple[np.ndarray, ...]:
    arrays = tuple(np.asarray(getattr(field, name), dtype=float) for name in (
        "harmonic_sonic", "cubic_sonic", "quartic_sonic", "metric_sonic",
        "metric_first_sonic", "metric_second_sonic",
    ))
    f2, f3, f4, metric, metric1, metric2 = arrays
    if f2.ndim != 2 or f2.shape[0] != f2.shape[1]:
        raise ValueError("SONIC VPT2 requires a square nonredundant F2")
    n = f2.shape[0]
    if tuple(a.shape for a in arrays) != (
        (n, n), (n, n, n), (n, n, n, n), (n, n),
        (n, n, n), (n, n, n, n),
    ):
        raise ValueError("SONIC VPT2 requires complete F2/F3/F4 and G/G'/G''")
    if not all(np.all(np.isfinite(a)) for a in arrays) or not np.isfinite(field.pseudopotential_cm):
        raise ValueError("SONIC VPT2 derivatives and pseudopotential must be finite")
    for name, tensor, axes in (
        ("F2", f2, ((0, 1),)),
        ("F3", f3, ((0, 1), (1, 2))),
        ("F4", f4, ((0, 1), (1, 2), (2, 3))),
        ("G", metric, ((0, 1),)),
        ("G'", metric1, ((0, 1),)),
        ("G''", metric2, ((0, 1), (2, 3))),
    ):
        if any(not np.allclose(tensor, np.swapaxes(tensor, i, j), rtol=1e-8, atol=1e-10)
               for i, j in axes):
            raise ValueError(f"SONIC VPT2 {name} is not symmetric")
    if np.min(np.linalg.eigvalsh(metric)) <= 0:
        raise ValueError("SONIC VPT2 metric must be positive definite")
    return arrays


def _apply(state: tuple[int, ...], operators: tuple[tuple[str, int], ...]) -> dict[tuple[int, ...], complex]:
    """Apply an ordered q/p product from right to left in the HO basis."""
    states: dict[tuple[int, ...], complex] = {state: 1.0}
    for kind, mode in reversed(operators):
        updated: dict[tuple[int, ...], complex] = {}
        for vector, amplitude in states.items():
            quanta = vector[mode]
            for change, factor in ((1, sqrt(quanta + 1)), (-1, sqrt(quanta))):
                if change < 0 and quanta == 0:
                    continue
                next_state = list(vector)
                next_state[mode] += change
                phase = 1 if kind == "q" else (1j if change > 0 else -1j)
                key = tuple(next_state)
                updated[key] = updated.get(key, 0j) + amplitude * factor * phase / sqrt(2)
        states = updated
    return states


def _add_term(
    destination: dict[tuple[int, ...], complex], state: tuple[int, ...],
    coefficient: float, operators: tuple[tuple[str, int], ...],
) -> None:
    for target, value in _apply(state, operators).items():
        destination[target] = destination.get(target, 0j) + coefficient * value


_OperatorKey = tuple[tuple[int, tuple[str, ...]], ...]
_OperatorTerms = tuple[tuple[_OperatorKey, float], ...]


def _canonical_operator_key(operators: tuple[tuple[str, int], ...]) -> _OperatorKey:
    """Commute distinct modes while retaining the order of p and q on each mode."""
    groups: dict[int, list[str]] = {}
    for kind, mode in operators:
        groups.setdefault(mode, []).append(kind)
    return tuple((mode, tuple(groups[mode])) for mode in sorted(groups))


@lru_cache(maxsize=8192)
def _local_operator_transitions(
    operators: tuple[str, ...], occupation: int,
) -> tuple[tuple[int, complex], ...]:
    """Cached action of one ordered local operator string on a HO state."""
    values: dict[int, complex] = {occupation: 1.0 + 0j}
    for kind in reversed(operators):
        updated: dict[int, complex] = {}
        for quanta, amplitude in values.items():
            for change, factor in ((1, sqrt(quanta + 1)), (-1, sqrt(quanta))):
                if change < 0 and quanta == 0:
                    continue
                next_quanta = quanta + change
                phase = 1 if kind == "q" else (1j if change > 0 else -1j)
                updated[next_quanta] = (
                    updated.get(next_quanta, 0j) + amplitude * factor * phase / sqrt(2)
                )
        values = updated
    return tuple(values.items())


def _compiled_operator_couplings(
    state: tuple[int, ...], terms: _OperatorTerms,
) -> dict[tuple[int, ...], complex]:
    couplings: dict[tuple[int, ...], complex] = {}
    for key, coefficient in terms:
        local = tuple((mode, _local_operator_transitions(operators, state[mode]))
                      for mode, operators in key)
        for choices in product(*(transitions for _mode, transitions in local)):
            target = list(state)
            amplitude = complex(coefficient)
            for (mode, _transitions), (occupation, factor) in zip(local, choices, strict=True):
                target[mode] = occupation
                amplitude *= factor
            result = tuple(target)
            couplings[result] = couplings.get(result, 0j) + amplitude
    return couplings


def _compiled_operator_matrix_element(
    source: tuple[int, ...], target: tuple[int, ...], terms: _OperatorTerms,
    term_index: Mapping[frozenset[int], _OperatorTerms] | None = None,
) -> complex:
    """Evaluate only the requested sparse operator element for a DD edge."""
    changed = {i for i, (a, b) in enumerate(zip(source, target)) if a != b}
    value = 0j
    selected_terms = (terms if term_index is None else
                      term_index.get(frozenset(changed), ()))
    for key, coefficient in selected_terms:
        if term_index is None and not changed.issubset(mode for mode, _operators in key):
            continue
        factor = complex(coefficient)
        for mode, operators in key:
            local = dict(_local_operator_transitions(operators, source[mode]))
            factor *= local.get(target[mode], 0j)
            if factor == 0:
                break
        value += factor
    return value


def _operator_terms_by_changed_modes(
    terms: _OperatorTerms,
) -> dict[frozenset[int], _OperatorTerms]:
    indexed: dict[frozenset[int], list[tuple[_OperatorKey, float]]] = {}
    for term in terms:
        modes = tuple(mode for mode, _operators in term[0])
        for count in range(len(modes) + 1):
            for subset in combinations(modes, count):
                indexed.setdefault(frozenset(subset), []).append(term)
    return {key: tuple(values) for key, values in indexed.items()}


def _canonical_cubic_terms(
    f3: np.ndarray, g1: np.ndarray, threshold: float,
) -> _OperatorTerms:
    terms: dict[_OperatorKey, float] = {}
    for indices in combinations_with_replacement(range(f3.shape[0]), 3):
        coefficient = float(f3[indices])
        if abs(coefficient) <= threshold:
            continue
        multiplicity = factorial(3)
        for mode in set(indices):
            multiplicity //= factorial(indices.count(mode))
        key = _canonical_operator_key(tuple(("q", mode) for mode in indices))
        terms[key] = terms.get(key, 0.0) + coefficient * multiplicity / 6
    for i, j, k in np.argwhere(np.abs(g1) > threshold):
        key = _canonical_operator_key((("p", int(i)), ("q", int(k)), ("p", int(j))))
        terms[key] = terms.get(key, 0.0) + float(g1[i, j, k]) / 2
    return tuple((key, value) for key, value in terms.items() if value)


def _canonical_quartic_terms(f4: np.ndarray, g2: np.ndarray) -> _OperatorTerms:
    terms: dict[_OperatorKey, float] = {}
    for i, j, k, l in np.argwhere(f4 != 0):
        key = _canonical_operator_key((("q", int(i)), ("q", int(j)),
                                       ("q", int(k)), ("q", int(l))))
        terms[key] = terms.get(key, 0.0) + float(f4[i, j, k, l]) / 24
    for i, j, k, l in np.argwhere(g2 != 0):
        key = _canonical_operator_key((("p", int(i)), ("q", int(k)),
                                       ("q", int(l)), ("p", int(j))))
        terms[key] = terms.get(key, 0.0) + float(g2[i, j, k, l]) / 4
    return tuple((key, value) for key, value in terms.items() if value)


def _cubic_couplings(
    state: tuple[int, ...], f3: np.ndarray, g1: np.ndarray,
    threshold: float, *, terms: _OperatorTerms | None = None,
) -> dict[tuple[int, ...], complex]:
    if terms is not None:
        return _compiled_operator_couplings(state, terms)
    n = len(state)
    couplings: dict[tuple[int, ...], complex] = {}
    for indices in combinations_with_replacement(range(n), 3):
        coefficient = float(f3[indices])
        if abs(coefficient) <= threshold:
            continue
        multiplicity = factorial(3)
        for mode in set(indices):
            multiplicity //= factorial(indices.count(mode))
        _add_term(couplings, state, coefficient * multiplicity / 6,
                  tuple(("q", mode) for mode in indices))
    for i, j, k in np.argwhere(np.abs(g1) > threshold):
        _add_term(couplings, state, float(g1[i, j, k]) / 2,
                  (("p", int(i)), ("q", int(k)), ("p", int(j))))
    return couplings


def _quartic_expectation(
    state: tuple[int, ...], f4: np.ndarray, g2: np.ndarray,
    pseudopotential_cm: float, threshold: float,
) -> float:
    n = len(state)
    value = float(pseudopotential_cm)
    for i in range(n):
        if abs(f4[i, i, i, i]) > threshold:
            value += float(f4[i, i, i, i]) * _apply(state, (("q", i),) * 4).get(state, 0j).real / 24
        for j in range(i + 1, n):
            if abs(f4[i, i, j, j]) > threshold:
                value += float(f4[i, i, j, j]) * _apply(
                    state, (("q", i), ("q", i), ("q", j), ("q", j))
                ).get(state, 0j).real / 4
    # Only even occurrences of every mode can return to the same Fock
    # state. Enumerating those index patterns avoids scanning n**4 G''
    # entries for each vibrational state in a large polyad.
    for i in range(n):
        indices = (i, i, i, i)
        if abs(g2[indices]) > threshold:
            value += float(g2[indices]) * _apply(
                state, (("p", i), ("q", i), ("q", i), ("p", i))
            ).get(state, 0j).real / 4
        for j in range(i + 1, n):
            for a, b, c, d in set(permutations((i, i, j, j))):
                if abs(g2[a, b, c, d]) <= threshold:
                    continue
                value += float(g2[a, b, c, d]) * _apply(
                    state, (("p", a), ("q", c), ("q", d), ("p", b))
                ).get(state, 0j).real / 4
    return value


def _modalize_sonic_field(
    field: SonicVPT2Input,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    f2, f3, f4, metric, metric1, metric2 = _validate_field(field)
    values, vectors = np.linalg.eigh(metric)
    ghalf = (vectors * np.sqrt(values)) @ vectors.T
    actions=None
    if field.symmetry_actions is None:
        eigenvalues, eigenvectors = np.linalg.eigh(ghalf @ f2 @ ghalf)
    else:
        from .field_symmetry import canonical_actions,invariant_eigh
        inverse_half=np.linalg.inv(ghalf)
        actions=canonical_actions([inverse_half@a@ghalf for a in field.symmetry_actions],
                                  tolerance=field.symmetry_tolerance)
        eigenvalues,eigenvectors=invariant_eigh(ghalf@f2@ghalf,actions)
    if np.any(eigenvalues <= 0):
        raise ValueError("SONIC perturbation theory requires a stationary vibrational minimum")
    frequencies = np.sqrt(eigenvalues) * HESSIAN_EIGENVALUE_TO_CM
    modes = ghalf @ eigenvectors
    inverse_modes = np.linalg.inv(modes)
    gamma = np.sqrt(frequencies * CM_PER_HARTREE) / HESSIAN_EIGENVALUE_TO_CM
    displacement = modes / gamma[None, :]
    momentum = gamma[:, None] * inverse_modes
    metric_scale = HESSIAN_EIGENVALUE_TO_CM**2 / CM_PER_HARTREE
    f3q = CM_PER_HARTREE * np.einsum("abc,ai,bj,ck->ijk", f3, displacement, displacement, displacement, optimize=True)
    f4q = CM_PER_HARTREE * np.einsum("abcd,ai,bj,ck,dl->ijkl", f4, displacement, displacement, displacement, displacement, optimize=True)
    g1q = metric_scale * np.einsum("ia,jb,abc,ck->ijk", momentum, momentum, metric1, displacement, optimize=True)
    g2q = metric_scale * np.einsum("ia,jb,abcd,ck,dl->ijkl", momentum, momentum, metric2, displacement, displacement, optimize=True)
    if actions is not None:
        from .field_symmetry import canonical_actions,invariant_tensor
        modal_actions=canonical_actions([eigenvectors.T@a@eigenvectors for a in actions],
                                        tolerance=field.symmetry_tolerance)
        f3q=invariant_tensor(f3q,modal_actions)
        f4q=invariant_tensor(f4q,modal_actions)
        g1q=invariant_tensor(g1q,modal_actions)
        g2q=invariant_tensor(g2q,modal_actions)
    return frequencies, modes, f3q, f4q, g1q, g2q


def solve_sonic_vpt2(
    field: SonicVPT2Input, *, denominator_tolerance_cm: float = 1e-6,
    coefficient_tolerance_cm: float = 0.0,
) -> SonicVPT2Result:
    """Calculate nonresonant VPT2 ZPVE and fundamentals in SONIC GF modes.

    A small denominator is rejected: resonant states require a separate
    deperturbed/GVPT2 treatment and are never silently removed.
    """
    if denominator_tolerance_cm <= 0 or coefficient_tolerance_cm < 0:
        raise ValueError("VPT2 tolerances must be nonnegative, with a positive denominator limit")
    frequencies, modes, f3q, f4q, g1q, g2q = _modalize_sonic_field(field)
    return _solve_modal_vpt2(
        frequencies, modes, f3q, f4q, g1q, g2q,
        field.pseudopotential_cm,
        denominator_tolerance_cm=denominator_tolerance_cm,
        coefficient_tolerance_cm=coefficient_tolerance_cm,
    )


def _algebraic_x_matrix(
    frequencies: np.ndarray, f3: np.ndarray, f4: np.ndarray,
    g1: np.ndarray, g2: np.ndarray, *, denominator_tolerance_cm: float,
) -> np.ndarray:
    """Mendolicchio eta/rho/sigma VPT2 x, without virtual-state sums.

    This is the nonresonant diagonal-quadratic reference. A resonant
    denominator belongs to a GVPT2 polyad, not to this x matrix.
    """
    from .x_diagnostics import algebraic_x_matrix
    return algebraic_x_matrix(
        frequencies, f3, f4, g1, g2,
        denominator_tolerance_cm=denominator_tolerance_cm,
    )


def sonic_vpt2_x_matrix(
    field: SonicVPT2Input, *, denominator_tolerance_cm: float = 1e-6,
) -> np.ndarray:
    """Algebraic nonresonant curvilinear VPT2 anharmonicity matrix."""
    if not np.isfinite(denominator_tolerance_cm) or denominator_tolerance_cm <= 0:
        raise ValueError("VPT2 denominator tolerance must be positive and finite")
    frequencies, _modes, f3, f4, g1, g2 = _modalize_sonic_field(field)
    return _algebraic_x_matrix(
        frequencies, f3, f4, g1, g2,
        denominator_tolerance_cm=denominator_tolerance_cm,
    )


def _solve_modal_vpt2(
    frequencies: np.ndarray, modes: np.ndarray, f3q: np.ndarray,
    f4q: np.ndarray, g1q: np.ndarray, g2q: np.ndarray,
    pseudopotential_cm: float, *, denominator_tolerance_cm: float,
    coefficient_tolerance_cm: float,
) -> SonicVPT2Result:
    n = len(frequencies)
    states = (tuple(0 for _ in range(n)),) + tuple(
        tuple(int(j == i) for j in range(n)) for i in range(n)
    )
    harmonic = np.empty(n + 1)
    quartic = np.empty(n + 1)
    cubic = np.empty(n + 1)
    for index, state in enumerate(states):
        harmonic[index] = float(np.dot(frequencies, np.asarray(state) + 0.5))
        quartic[index] = _quartic_expectation(
            state, f4q, g2q, pseudopotential_cm, coefficient_tolerance_cm
        )
        couplings = _cubic_couplings(state, f3q, g1q, coefficient_tolerance_cm)
        correction = 0.0
        for other, coupling in couplings.items():
            if other == state or abs(coupling) <= coefficient_tolerance_cm:
                continue
            denominator = float(np.dot(frequencies, np.asarray(state) - np.asarray(other)))
            if abs(denominator) <= denominator_tolerance_cm:
                raise ValueError(f"SONIC VPT2 resonance between {state} and {other}; use GVPT2")
            correction += abs(coupling)**2 / denominator
        cubic[index] = correction
    energies = harmonic + quartic + cubic
    x_matrix = _algebraic_x_matrix(
        frequencies, f3q, f4q, g1q, g2q,
        denominator_tolerance_cm=denominator_tolerance_cm,
    )
    return SonicVPT2Result(
        frequencies_cm=frequencies, zero_point_cm=float(energies[0]),
        fundamentals_cm=energies[1:] - energies[0],
        harmonic_energies_cm=harmonic,
        quartic_kinetic_energies_cm=quartic,
        cubic_second_order_energies_cm=cubic,
        sonic_modes=modes, x_matrix_cm=x_matrix,
    )


def _quartic_couplings(
    state: tuple[int, ...], f4: np.ndarray, g2: np.ndarray,
    *, terms: _OperatorTerms | None = None,
) -> dict[tuple[int, ...], complex]:
    if terms is not None:
        return _compiled_operator_couplings(state, terms)
    couplings: dict[tuple[int, ...], complex] = {}
    for indices in np.argwhere(f4 != 0):
        i, j, k, l = map(int, indices)
        _add_term(couplings, state, float(f4[i, j, k, l]) / 24,
                  (("q", i), ("q", j), ("q", k), ("q", l)))
    for indices in np.argwhere(g2 != 0):
        i, j, k, l = map(int, indices)
        _add_term(couplings, state, float(g2[i, j, k, l]) / 4,
                  (("p", i), ("q", k), ("q", l), ("p", j)))
    return couplings


@lru_cache(maxsize=4096)
def _local_oscillator_element(
    operators: tuple[str, ...], source: int, target: int,
) -> complex:
    """One-mode ladder contraction, preserving the stated p/q ordering."""
    values: dict[int, complex] = {source: 1.0 + 0j}
    for kind in reversed(operators):
        updated: dict[int, complex] = {}
        for occupation, amplitude in values.items():
            for change in (-1, 1):
                if occupation + change < 0:
                    continue
                factor = sqrt(occupation if change < 0 else occupation + 1) / sqrt(2)
                phase = 1 if kind == "q" else (1j if change > 0 else -1j)
                next_occupation = occupation + change
                updated[next_occupation] = (
                    updated.get(next_occupation, 0j) + amplitude * factor * phase
                )
        values = updated
    return values.get(target, 0j)


def _factorized_operator_element(
    source: tuple[int, ...], target: tuple[int, ...],
    operators: tuple[tuple[str, int], ...],
) -> complex:
    groups: dict[int, list[str]] = {}
    for kind, mode in operators:
        groups.setdefault(mode, []).append(kind)
    if any(source[mode] != target[mode] for mode in range(len(source)) if mode not in groups):
        return 0j
    value = 1.0 + 0j
    for mode, kinds in groups.items():
        value *= _local_oscillator_element(tuple(kinds), source[mode], target[mode])
        if not value:
            break
    return value


def _operator_element(
    source: tuple[int, ...], target: tuple[int, ...],
    f3: np.ndarray, f4: np.ndarray, g1: np.ndarray, g2: np.ndarray,
    *, order: int,
) -> complex:
    """Evaluate one sparse cubic or quartic Fock matrix element directly.

    A degree-d polynomial can connect states only when the sum of absolute
    occupation changes has the parity of d and is at most d. The remaining
    indices occur in equal pairs. This avoids building all virtual-state
    couplings when the property contraction needs only selected targets.
    """
    changes = tuple(abs(a - b) for a, b in zip(source, target))
    distance = sum(changes)
    if (len(source) != len(target) or order not in {3, 4}
            or distance > order or distance % 2 != order % 2):
        return 0j
    required = tuple(i for i, count in enumerate(changes) for _ in range(count))
    if distance == order:
        multisets = (tuple(sorted(required)),)
    elif distance == order - 2:
        multisets = tuple(tuple(sorted((*required, mode, mode)))
                          for mode in range(len(source)))
    else:
        multisets = tuple(
            tuple(sorted((i, i, j, j)))
            for i in range(len(source)) for j in range(i, len(source))
        )
    answer = 0j
    for indices in multisets:
        ordered = set(permutations(indices))
        if order == 3:
            coefficient = float(f3[indices])
            if coefficient:
                answer += coefficient * sum(
                    _factorized_operator_element(
                        source, target, tuple(("q", mode) for mode in item)
                    )
                    for item in ordered
                ) / 6
            for i, j, k in ordered:
                coefficient = float(g1[i, j, k])
                if coefficient:
                    answer += coefficient * _factorized_operator_element(
                        source, target, (("p", i), ("q", k), ("p", j))
                    ) / 2
        else:
            for i, j, k, l in ordered:
                coefficient = float(f4[i, j, k, l])
                if coefficient:
                    answer += coefficient * _factorized_operator_element(
                        source, target, (("q", i), ("q", j), ("q", k), ("q", l))
                    ) / 24
                coefficient = float(g2[i, j, k, l])
                if coefficient:
                    answer += coefficient * _factorized_operator_element(
                        source, target, (("p", i), ("q", k), ("q", l), ("p", j))
                    ) / 4
    return answer


class _SelectedOperatorEvaluator:
    """Reuse ordered local Hamiltonian terms across selected Fock contractions."""

    def __init__(self, f3: np.ndarray, f4: np.ndarray,
                 g1: np.ndarray, g2: np.ndarray) -> None:
        self.tensors = (f3, f4, g1, g2)
        self.terms: dict[tuple[int, tuple[int, ...]], _OperatorTerms] = {}

    def _terms(self, indices: tuple[int, ...], order: int) -> _OperatorTerms:
        cache_key = order, indices
        if cache_key in self.terms:
            return self.terms[cache_key]
        f3, f4, g1, g2 = self.tensors
        ordered = set(permutations(indices))
        terms: dict[_OperatorKey, float] = {}
        if order == 3:
            coefficient = float(f3[indices])
            if coefficient:
                key = _canonical_operator_key(tuple(("q", mode) for mode in indices))
                terms[key] = coefficient * len(ordered) / 6
            for i, j, k in ordered:
                coefficient = float(g1[i, j, k])
                if coefficient:
                    key = _canonical_operator_key((("p", i), ("q", k), ("p", j)))
                    terms[key] = terms.get(key, 0.0) + coefficient / 2
        else:
            for i, j, k, l in ordered:
                coefficient = float(f4[i, j, k, l])
                if coefficient:
                    key = _canonical_operator_key(
                        (("q", i), ("q", j), ("q", k), ("q", l)),
                    )
                    terms[key] = terms.get(key, 0.0) + coefficient / 24
                coefficient = float(g2[i, j, k, l])
                if coefficient:
                    key = _canonical_operator_key(
                        (("p", i), ("q", k), ("q", l), ("p", j)),
                    )
                    terms[key] = terms.get(key, 0.0) + coefficient / 4
        compiled = tuple((key, value) for key, value in terms.items() if value)
        self.terms[cache_key] = compiled
        return compiled

    def element(self, source: tuple[int, ...], target: tuple[int, ...],
                *, order: int) -> complex:
        if len(source) != len(target):
            return 0j
        changes = tuple(abs(a - b) for a, b in zip(source, target))
        distance = sum(changes)
        if distance > order or distance % 2 != order % 2:
            return 0j
        required = tuple(i for i, count in enumerate(changes) for _ in range(count))
        if distance == order:
            multisets = (required,)
        elif distance == order - 2:
            multisets = tuple(tuple(sorted((*required, mode, mode)))
                              for mode in range(len(source)))
        else:
            multisets = tuple((i, i, j, j)
                              for i in range(len(source)) for j in range(i, len(source)))
        answer = 0j
        for indices in multisets:
            for key, coefficient in self._terms(indices, order):
                value = complex(coefficient)
                for mode, operators in key:
                    value *= _local_oscillator_element(
                        operators, source[mode], target[mode],
                    )
                    if not value:
                        break
                answer += value
        return answer


def _gvpt2_states(n: int, max_quanta: int, max_states: int) -> tuple[tuple[int, ...], ...]:
    if max_quanta < 2 or max_states < 1:
        raise ValueError("GVPT2 requires at least two quanta and a positive state cap")
    states: list[tuple[int, ...]] = [tuple(0 for _ in range(n))]
    for quanta in range(1, max_quanta + 1):
        for occupied in combinations_with_replacement(range(n), quanta):
            counts = [0] * n
            for mode in occupied:
                counts[mode] += 1
            states.append(tuple(counts))
            if len(states) > max_states:
                raise ValueError(f"GVPT2 basis exceeds max_states={max_states}")
    return tuple(states)


def _resonance_kind(a: tuple[int, ...], b: tuple[int, ...]) -> str | None:
    left = sum(max(x - y, 0) for x, y in zip(a, b))
    right = sum(max(y - x, 0) for x, y in zip(a, b))
    if {left, right} == {1, 2}:
        return "1-2"
    if (left, right) == (1, 1):
        return "1-1"
    if (left, right) == (2, 2):
        return "2-2"
    if {left, right} == {1, 3}:
        return "1-3"
    return None


def _semidiagonal_quartic_edge_available(
    a: tuple[int, ...], b: tuple[int, ...], kind: str,
) -> bool:
    """Return whether F_ijkk contains the direct quartic matrix element.

    A 1:1 contact-transformation element contracts the two unused quartic
    operators on one spectator mode and is therefore semidiagonal.  For 2:2
    and 1:3 edges, the four changed quanta identify the required direct
    quartic element.  A four-distinct-mode element is absent from a
    single-mode Hessian stencil and its cubic contribution alone is not a
    complete effective coupling.
    """
    if kind == "1-1":
        return True
    changed: list[int] = []
    for mode, (left, right) in enumerate(zip(a, b, strict=True)):
        changed.extend([mode] * abs(left - right))
    return len(changed) < 4 or len(set(changed)) < 4


def _fermi_reduced_coupling(
    a: tuple[int, ...], b: tuple[int, ...], f3: np.ndarray, g1: np.ndarray,
) -> float:
    """Bare 1:2 Fermi coupling, independent of spectator occupations."""
    residual_a = np.maximum(np.subtract(a, b), 0)
    residual_b = np.maximum(np.subtract(b, a), 0)
    single, double = (residual_a, residual_b) if sum(residual_a) == 1 else (residual_b, residual_a)
    i = int(np.flatnonzero(single)[0])
    j, k = (int(mode) for mode in np.repeat(np.arange(len(a)), double))
    rho = f3[j, k, i] - g1[j, k, i] + g1[j, i, k] + g1[k, i, j]
    return float(rho / (4 if j == k else sqrt(8)))


def _fermi_martin(
    a: tuple[int, ...], b: tuple[int, ...], frequencies: np.ndarray,
    f3: np.ndarray, g1: np.ndarray,
) -> float:
    reduced_coupling = _fermi_reduced_coupling(a, b, f3, g1)
    gap = abs(float(np.dot(frequencies, np.subtract(a, b))))
    return abs(reduced_coupling) ** 4 / gap**3 if gap > 0 else (
        np.inf if reduced_coupling else 0.0
    )


def _gdv_overtone_dd_coupling(
    left: tuple[int, ...], right: tuple[int, ...], frequencies: np.ndarray,
    f3: np.ndarray, f4: np.ndarray, g1: np.ndarray, g2: np.ndarray,
) -> float | None:
    """Reduced 2:2 Darling–Dennison coupling for two pure overtones.

    This is the closed contact-transformation expression used by the
    Cartesian GVPT2 reference.  Terms with a small Fermi denominator are
    regularized before polyad diagonalization.
    """
    if sum(left) != 2 or sum(right) != 2 or 2 not in left or 2 not in right:
        return None
    k, l = left.index(2), right.index(2)
    if k == l:
        return None
    wk, wl = float(frequencies[k]), float(frequencies[l])

    def sigma(i: int, j: int, m: int) -> float:
        return float(f3[i, j, m] - g1[i, j, m] - g1[i, m, j] - g1[j, m, i])

    def rho(i: int, j: int, m: int) -> float:
        return float(f3[i, j, m] - g1[i, j, m] + g1[i, m, j] + g1[j, m, i])

    def regularized(i: int, j: int, m: int, numerator: float) -> float:
        gap = float(frequencies[i] + frequencies[j] - frequencies[m])
        half = abs(gap) / 2
        return float(np.sign(numerator * gap) * abs(numerator)
                     / (sqrt(abs(numerator) + half * half) + half))

    eta_dd = (
        f4[k, k, l, l] - g2[k, k, l, l] - g2[l, l, k, k]
        + g2[k, l, k, l] + g2[k, l, k, l]
        + g2[k, l, l, k] + g2[k, l, l, k]
    )
    value = float(eta_dd) / 4
    for r, wr in enumerate(frequencies):
        skkr, sllr = sigma(k, k, r), sigma(l, l, r)
        rkkr, rllr = rho(k, k, r), rho(l, l, r)
        rkrl, rlrk = rho(k, r, l), rho(l, r, k)
        value -= skkr * sllr / 16 * (1 / (2 * wk + wr) + 1 / (2 * wl + wr))
        value += regularized(k, k, r, rkkr * rllr / 16)
        value += regularized(l, l, r, rkkr * rllr / 16)
        value -= regularized(k, r, l, rkrl * rlrk / 4)
        value -= regularized(l, r, k, rkrl * rlrk / 4)
    return value / 2


def _gdv_combination_overtone_dd_coupling(
    left: tuple[int, ...], right: tuple[int, ...], frequencies: np.ndarray,
    f3: np.ndarray, f4: np.ndarray, g1: np.ndarray, g2: np.ndarray,
) -> float | None:
    """Closed 2:2 coupling between a pure overtone and a combination band."""
    if sum(left) != 2 or sum(right) != 2:
        return None
    pure, mixed = (left, right) if 2 in left else (right, left)
    if 2 not in pure or 2 in mixed:
        return None
    k = pure.index(2)
    l, m = (int(i) for i in np.flatnonzero(mixed))
    if k in (l, m):
        return None
    wk, wl, wm = (float(frequencies[i]) for i in (k, l, m))

    def sigma(i: int, j: int, n: int) -> float:
        return float(f3[i, j, n] - g1[i, j, n] - g1[i, n, j] - g1[j, n, i])

    def rho(i: int, j: int, n: int) -> float:
        return float(f3[i, j, n] - g1[i, j, n] + g1[i, n, j] + g1[j, n, i])

    def regularized(i: int, j: int, n: int, numerator: float) -> float:
        gap = float(frequencies[i] + frequencies[j] - frequencies[n])
        half = abs(gap) / 2
        return float(np.sign(numerator * gap) * abs(numerator)
                     / (sqrt(abs(numerator) + half * half) + half))

    eta_dd = (
        f4[k, k, l, m] - g2[k, k, l, m] - g2[l, m, k, k]
        + g2[k, l, k, m] + g2[k, m, l, k]
        + g2[k, m, k, l] + g2[k, l, m, k]
    )
    value = float(eta_dd) / 2
    for r, wr in enumerate(frequencies):
        rkrl, rmrk = rho(k, r, l), rho(m, r, k)
        rlrk, rkrm = rho(l, r, k), rho(k, r, m)
        rkkr, rlmr = rho(k, k, r), rho(l, m, r)
        value -= regularized(k, r, l, rkrl * rmrk / 4)
        value -= regularized(l, r, k, rlrk * rkrm / 4)
        value -= regularized(k, r, m, rlrk * rkrm / 4)
        value -= regularized(m, r, k, rkrl * rmrk / 4)
        value -= sigma(k, k, r) * sigma(l, m, r) / 8 * (
            1 / (2 * wk + wr) + 1 / (wl + wm + wr)
        )
        value += regularized(k, k, r, rkkr * rlmr / 8)
        value += regularized(l, m, r, rkkr * rlmr / 8)
    return value * sqrt(2) / 4


def _gdv_fundamental_dd_coupling(
    left: tuple[int, ...], right: tuple[int, ...], frequencies: np.ndarray,
    f3: np.ndarray, f4: np.ndarray, g1: np.ndarray, g2: np.ndarray,
) -> float | None:
    """Closed 1:1 contact-transformation coupling, including spectator modes."""
    if sum(left) != 1 or sum(right) != 1:
        return None
    k, m = left.index(1), right.index(1)
    if k == m:
        return None
    wk, wm = float(frequencies[k]), float(frequencies[m])

    def sigma(i: int, j: int, l: int) -> float:
        return float(f3[i, j, l] - g1[i, j, l] - g1[i, l, j] - g1[j, l, i])

    def rho(i: int, j: int, l: int) -> float:
        return float(f3[i, j, l] - g1[i, j, l] + g1[i, l, j] + g1[j, l, i])

    def eta(i: int, j: int, l: int, n: int) -> float:
        return float(f4[i, j, l, n] + g2[i, j, l, n] + g2[l, n, i, j])

    def regularized(i: int, j: int, l: int, numerator: float) -> float:
        gap = float(frequencies[i] + frequencies[j] - frequencies[l])
        half = abs(gap) / 2
        return float(np.sign(numerator * gap) * abs(numerator)
                     / (sqrt(abs(numerator) + half * half) + half))

    kkkm = eta(k, k, m, k) / 6
    kmmm = eta(k, m, m, m) / 6
    for r, wr in enumerate(frequencies):
        skmr, rkmr = sigma(k, m, r), rho(k, m, r)
        rkrm, rmrk = rho(k, r, m), rho(m, r, k)
        skkr, rkkr, rrkk = sigma(k, k, r), rho(k, k, r), rho(r, k, k)
        smmr, rmmr, rrmm = sigma(m, m, r), rho(m, m, r), rho(r, m, m)
        kkkm -= skmr * skkr / 24 * (1 / (2 * wk + wr) + 1 / (wm + wk + wr))
        kkkm += regularized(k, m, r, rkmr * rkkr / 24)
        kkkm += regularized(k, k, r, rkmr * rkkr / 24)
        kkkm -= 2 * regularized(k, r, m, rkrm * rrkk / 24)
        kkkm -= 2 * regularized(m, r, k, rmrk * rrkk / 24)
        kkkm -= (rkrm + rmrk) * rrkk / (12 * wr)
        kmmm -= skmr * smmr / 24 * (1 / (2 * wm + wr) + 1 / (wm + wk + wr))
        kmmm += regularized(k, m, r, rkmr * rmmr / 24)
        kmmm += regularized(m, m, r, rkmr * rmmr / 24)
        kmmm -= 2 * regularized(m, r, k, rmrk * rrmm / 24)
        kmmm -= 2 * regularized(k, r, m, rkrm * rrmm / 24)
        kmmm -= (rmrk + rkrm) * rrmm / (12 * wr)
    value = 1.5 * (kkkm + kmmm)
    for r, wr in enumerate(frequencies):
        if r in (k, m):
            continue
        kkr = eta(k, m, r, r) / 2
        for s, ws in enumerate(frequencies):
            skrs, smrs = sigma(k, r, s), sigma(m, r, s)
            rrsk, rrsm = rho(r, s, k), rho(r, s, m)
            rksr, rmsr = rho(k, s, r), rho(m, s, r)
            rkrs, rmrs = rho(k, r, s), rho(m, r, s)
            kkr -= skrs * smrs / 8 * (
                1 / (wk + wr + ws) + 1 / (wm + wr + ws))
            kkr -= regularized(r, s, k, rrsk * rrsm / 8)
            kkr -= regularized(r, s, m, rrsk * rrsm / 8)
            kkr -= regularized(k, s, r, rksr * rmsr / 8)
            kkr -= regularized(m, s, r, rksr * rmsr / 8)
            kkr += regularized(k, r, s, rkrs * rmrs / 8)
            kkr += regularized(m, r, s, rkrs * rmrs / 8)
            rmsk, rksm, rsrr = rho(m, s, k), rho(k, s, m), rho(s, r, r)
            kkr -= regularized(m, s, k, rmsk * rsrr / 8)
            kkr -= regularized(k, s, m, rksm * rsrr / 8)
            kkr -= rsrr * (rmsk + rksm) / (8 * ws)
        value += kkr / 2
    return value / 2


def _tars_auto_fermi_states(
    frequencies: np.ndarray, f3: np.ndarray, g1: np.ndarray,
    *, fermi_window_cm: float, martin_threshold_cm: float,
    numerical_zero_cm: float, max_states: int,
    wavefunction_threshold: float | None = None,
    dd_window_cm: float | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Seed 2Q states by Fermi tests, then close DD candidate windows.

    For a fundamental and a two-quantum state the TARS curvilinear H3
    matrix element is rho/sqrt(8), or rho/4 for an overtone.  This
    preselection is identical to the later Fermi edge test.  When DD is
    enabled, all energy-compatible 2Q neighbors of a selected state are
    retained as candidates, transitively.  The later effective-Hamiltonian
    test decides which of those candidates actually belong to its polyad.
    Virtual perturbative intermediate states remain unrestricted.
    """
    n = len(frequencies)
    states = [tuple(0 for _ in range(n))]
    states.extend(tuple(int(j == i) for j in range(n)) for i in range(n))
    for j, k in combinations_with_replacement(range(n), 2):
        pair_energy = float(frequencies[j] + frequencies[k])
        for i in range(n):
            gap = abs(float(frequencies[i]) - pair_energy)
            if gap > fermi_window_cm:
                continue
            rho = f3[j, k, i] - g1[j, k, i] + g1[j, i, k] + g1[k, i, j]
            coupling = float(rho) / (4 if j == k else sqrt(8))
            martin = abs(coupling)**4 / gap**3 if gap > 0 else (
                np.inf if coupling else 0.0
            )
            wavefunction = abs(coupling) / gap if gap > 0 else (
                np.inf if coupling else 0.0
            )
            if abs(coupling) >= numerical_zero_cm and (
                martin >= martin_threshold_cm
                or wavefunction_threshold is not None
                and wavefunction >= wavefunction_threshold
            ):
                counts = [0] * n
                counts[j] += 1
                counts[k] += 1
                states.append(tuple(counts))
                if len(states) > max_states:
                    raise ValueError(f"TARS selected basis exceeds max_states={max_states}")
                break
    if dd_window_cm is not None:
        if not np.isfinite(dd_window_cm) or dd_window_cm < 0:
            raise ValueError("DD candidate window must be finite and nonnegative")
        selected = set(states)
        two_quanta = []
        for j, k in combinations_with_replacement(range(n), 2):
            counts = [0] * n
            counts[j] += 1
            counts[k] += 1
            two_quanta.append(tuple(counts))
        energies = {state: float(np.dot(frequencies, state)) for state in two_quanta}
        frontier = [state for state in states if sum(state) == 2]
        while frontier:
            current = frontier.pop()
            for candidate in two_quanta:
                if candidate in selected or abs(energies[current] - energies[candidate]) > dd_window_cm:
                    continue
                if _resonance_kind(current, candidate) in {"1-1", "2-2"}:
                    selected.add(candidate)
                    states.append(candidate)
                    frontier.append(candidate)
                    if len(states) > max_states:
                        raise ValueError(f"TARS selected basis exceeds max_states={max_states}")
    return tuple(states)


def _candidate_state_pairs(
    states: tuple[tuple[int, ...], ...], max_quanta: int,
    *, kinds: frozenset[str] | None = None,
) -> Iterator[tuple[int, int, str]]:
    """Stream candidate pairs indexed by their common occupation state."""
    buckets: dict[tuple[int, tuple[int, ...]], list[tuple[int, int]]] = {}
    for index, state in enumerate(states):
        for residual in product(*(range(value + 1) for value in state)):
            order = sum(residual)
            if not 1 <= order <= min(3, max_quanta):
                continue
            base = tuple(value - removed for value, removed in zip(state, residual))
            support = sum(1 << mode for mode, count in enumerate(residual) if count)
            buckets.setdefault((order, base), []).append((index, support))
    classes = ((1, 1, "1-1"), (1, 2, "1-2"), (2, 2, "2-2"))
    if max_quanta > 2:
        classes += ((1, 3, "1-3"),)
    for left_order, right_order, kind in classes:
        if kinds is not None and kind not in kinds:
            continue
        for (order, base), left in buckets.items():
            if order != left_order:
                continue
            right = buckets.get((right_order, base), ())
            iterator = combinations(left, 2) if left_order == right_order else product(left, right)
            for (i, support_i), (j, support_j) in iterator:
                if i != j and not support_i & support_j:
                    yield min(i, j), max(i, j), kind


def _gvpt2_report(
    states: tuple[tuple[int, ...], ...], frequencies: np.ndarray,
    deperturbed: np.ndarray, energies: np.ndarray,
    resonances: list[dict[str, object]], blocks: list[dict[str, object]],
    diagnostics: dict[str, object],
) -> str:
    def label(state: tuple[int, ...]) -> str:
        return "|0>" if not any(state) else "|" + ";".join(
            f"{count}_{index + 1}" for index, count in enumerate(state) if count
        ) + ">"

    lines = [
        "TRINITY SONIC GVPT2",
        f"State basis: {len(states)}; candidate pairs: {diagnostics['candidate_pair_count']}; "
        f"accepted couplings: {diagnostics['accepted_pair_count']}",
        f"Resonant blocks: {diagnostics['resonant_block_count']}; "
        f"largest dimension: {diagnostics['largest_polyad_dimension']}",
        "Harmonic modes [cm^-1]",
    ]
    lines.extend(f"  {i + 1:3d}  {value:13.5f}" for i, value in enumerate(frequencies))
    if "resonance_screening" in diagnostics:
        settings = diagnostics["resonance_screening"]
        lines.append(f"Resonance profile: {settings['profile']}; energy tolerance "
                     f"{settings['energy_tolerance_cm']} cm^-1; C1/C2 limits "
                     f"{settings['first_order_limit']}/{settings['second_order_limit']}")
    lines.extend(("", "Resonance selection [cm^-1]", "  State 1       State 2       Class   Gap       H12       Decision"))
    for item in resonances:
        lines.append(
            f"  {label(states[int(item['i'])]):13s} {label(states[int(item['j'])]):13s} "
            f"{str(item['kind']):5s} {float(item['gap_cm']):9.3f} "
            f"{float(item['coupling_cm']):9.3f}  {item['decision']}"
        )
        if "screening_profile" in item:
            lines.append(f"    energy={item['energy_score_cm']} cm^-1; "
                         f"|C| forward/reverse={item['coefficient_abs_forward']}/"
                         f"{item['coefficient_abs_reverse']}; pole={item['coefficient_pole']}; "
                         f"selected by energy/coefficient={item['energy_selected']}/"
                         f"{item['coefficient_selected']}")
    if not resonances:
        lines.append("  No supported resonance candidates.")
    lines.extend(("", "Polyads and eigenstates [cm^-1]"))
    for number, block in enumerate(blocks, 1):
        lines.append(f"  Polyad {number}: {len(block['indices'])} states")
        vectors = np.asarray(block["eigenvectors"])
        for root, value in enumerate(np.asarray(block["eigenvalues_cm"])):
            components = sorted(
                ((abs(vectors[row, root]) ** 2, states[index])
                 for row, index in enumerate(block["indices"])), reverse=True,
            )
            composition = " + ".join(
                f"{100 * weight:.1f}% {label(state)}"
                for weight, state in components if weight >= 0.01
            )
            lines.append(f"    {value:13.5f}  {composition}")
    if not blocks:
        lines.append("  No selected nonzero polyads.")
    lines.extend(("", "Assigned states [cm^-1]", "  State         Deperturbed    GVPT2"))
    lines.extend(
        f"  {label(state):13s} {deperturbed[i]:13.5f} {energies[i]:13.5f}"
        for i, state in enumerate(states)
    )
    return "\n".join(lines) + "\n"


def solve_sonic_gvpt2(
    field: SonicVPT2Input, *, max_quanta: int = 3, max_states: int = 30000,
    resonance_policy: str = "auto", fermi_window_cm: float = 200.0,
    dd_window_cm: float = 100.0, martin_threshold_cm: float = 1.0,
    dd_coupling_threshold_cm: float = 3.0,
    numerical_zero_cm: float | None = None,
    denominator_tolerance_cm: float = 1e-6,
    fermi_basis_policy: str = "complete",
    candidate_states: tuple[tuple[int, ...], ...] | None = None,
    polyad_excluded_modes: tuple[int, ...] = (),
    max_mode_quanta: int = 4,
    chart_connection: object | None = None,
    symmetry_actions: tuple[np.ndarray, ...] | None = None,
    resonance_criteria: Mapping[str, float] | None = None,
    targeted_polyad_options: Mapping[str, object] | None = None,
    _workspace: dict[str, object] | None = None,
) -> SonicGVPT2Result:
    """Build deperturbed SONIC GVPT2 polyads and diagonalize them.

    ``targeted_polyad_options`` enables the shared target-directed cubic
    refinement, including resource guards and consistent property columns.
    The production route raises rather than returning a partial spectrum.

    This implementation requires declared complete quartic coverage for a
    Darling--Dennison policy: arbitrary four-distinct-index couplings are
    possible. Individual edges with a repeated index need only the matching
    semidiagonal elements; see the versioned VP2TD polyad protocol.
    A supplied chart connection and complete real group action close and
    transport selected polyads before returning the normal PT2D result.
    """
    if (chart_connection is None) != (symmetry_actions is None):
        raise ValueError("common-gauge PT2D requires both chart and group action")
    if resonance_policy not in {"auto", "all", "fermi", "fermi_auto", "gdv", "intensity", "none"}:
        raise ValueError("unsupported SONIC GVPT2 resonance policy")
    if resonance_criteria is not None and resonance_policy != "intensity":
        raise ValueError("resonance_criteria requires the intensity resonance policy")
    criteria = IntensityResonanceCriteria.from_options(resonance_criteria)
    if numerical_zero_cm is None:
        numerical_zero_cm = 1e-12 if resonance_policy == "intensity" else 0.01
    if field.quartic_coverage not in {"complete", "semidiagonal"}:
        raise ValueError("SONIC GVPT2 field has an unknown quartic coverage")
    if field.quartic_coverage != "complete" and resonance_policy == "all":
        raise ValueError("all-resonance policy requires complete quartic coverage")
    thresholds = (fermi_window_cm, dd_window_cm, martin_threshold_cm,
                  dd_coupling_threshold_cm, numerical_zero_cm,
                  denominator_tolerance_cm)
    if any(not np.isfinite(v) or v < 0 for v in thresholds) or denominator_tolerance_cm == 0:
        raise ValueError("SONIC GVPT2 thresholds must be finite and nonnegative")
    start_time = perf_counter()
    workspace = {} if _workspace is None else _workspace
    if "modalized" not in workspace:
        workspace["modalized"] = _modalize_sonic_field(field)
    frequencies, _modes, f3q, f4q, g1q, g2q = workspace["modalized"]
    excluded_modes = frozenset(int(mode) for mode in polyad_excluded_modes)
    if (len(excluded_modes) != len(polyad_excluded_modes)
            or any(mode < 0 or mode >= len(frequencies) for mode in excluded_modes)):
        raise ValueError("invalid polyad-excluded mode indices")
    if max_mode_quanta < 1:
        raise ValueError("per-mode quantum limit must be positive")
    def allowed_state(state: tuple[int, ...]) -> bool:
        return ((not any(state[mode] for mode in excluded_modes)
                 or sum(state) == 1)
                and max(state) <= max_mode_quanta)
    modalization_time = perf_counter() - start_time
    basis_start = perf_counter()
    if candidate_states is not None:
        n = len(frequencies)
        required = {tuple(0 for _ in range(n))}
        required.update(tuple(int(j == i) for j in range(n)) for i in range(n))
        states = tuple(tuple(int(v) for v in state) for state in candidate_states)
        if (len(states) > max_states or len(set(states)) != len(states)
                or any(len(state) != n or any(v < 0 for v in state) for state in states)
                or not required.issubset(states)
                or any(not allowed_state(state) for state in states)
                or any(sum(state) > max_quanta for state in states)):
            raise ValueError("invalid or incomplete explicit GVPT2 candidate basis")
    elif fermi_basis_policy == "complete":
        states = _gvpt2_states(len(frequencies), max_quanta, max_states)
    elif fermi_basis_policy == "tars_auto":
        if resonance_policy not in {"fermi_auto", "gdv"} or max_quanta != 2:
            raise ValueError("TARS Fermi state screening requires fermi_auto or gdv and max_quanta=2")
        states = _tars_auto_fermi_states(
            frequencies, f3q, g1q, fermi_window_cm=fermi_window_cm,
            martin_threshold_cm=martin_threshold_cm,
            numerical_zero_cm=numerical_zero_cm, max_states=max_states,
            wavefunction_threshold=0.1 if resonance_policy == "gdv" else None,
            dd_window_cm=dd_window_cm if resonance_policy == "gdv" else None,
        )
    else:
        raise ValueError("unsupported Fermi basis policy")
    if excluded_modes:
        states = tuple(state for state in states if allowed_state(state))
    indices = {state: i for i, state in enumerate(states)}
    harmonic = np.array([np.dot(frequencies, state) for state in states])
    basis_time = perf_counter() - basis_start
    coupling_start = perf_counter()
    if "cubic_terms" not in workspace:
        workspace["cubic_terms"] = _canonical_cubic_terms(f3q, g1q, 0.0)
    cubic_terms = workspace["cubic_terms"]
    cubic_cache = workspace.setdefault("cubic_couplings", {})
    h3 = []
    for state in states:
        if state not in cubic_cache:
            cubic_cache[state] = _cubic_couplings(state, f3q, g1q, 0.0, terms=cubic_terms)
        h3.append(cubic_cache[state])
    quartic_terms: _OperatorTerms | None = workspace.get("quartic_terms")
    candidates: list[dict[str, object]] = []
    candidate_count = 0

    from .gvpt2_operator import GVPT2EffectiveOperator
    effective_operator=GVPT2EffectiveOperator(field,workspace=workspace,
        resonance_policy=resonance_policy,fermi_window_cm=fermi_window_cm,
        dd_window_cm=dd_window_cm,martin_threshold_cm=martin_threshold_cm,
        numerical_zero_cm=numerical_zero_cm,denominator_tolerance_cm=denominator_tolerance_cm,
        polyad_excluded_modes=polyad_excluded_modes,resonance_criteria=resonance_criteria)
    resonant_fermi=effective_operator.resonant_fermi

    for i, j, kind in _candidate_state_pairs(states, max_quanta, kinds=frozenset(("1-2",))):
            if any(states[i][mode] or states[j][mode] for mode in excluded_modes):
                continue
            candidate_count += 1
            a, b = states[i], states[j]
            gap = abs(harmonic[i] - harmonic[j])
            coupling = float(np.real_if_close(h3[j].get(a, 0j)).real)
            if abs(coupling) < numerical_zero_cm:
                continue
            martin = _fermi_martin(a, b, frequencies, f3q, g1q)
            reduced = _fermi_reduced_coupling(a, b, f3q, g1q)
            mixing = abs(reduced) / max(gap, denominator_tolerance_cm)
            accepted = resonant_fermi(a, b, coupling)
            decision = "accepted" if accepted else (
                "policy" if resonance_policy == "none" else
                "outside window" if gap > fermi_window_cm and resonance_policy in {"auto", "fermi_auto", "gdv", "intensity"} else
                "resonance test" if resonance_policy in {"auto", "fermi_auto", "gdv", "intensity"} else
                "numerical zero"
            )
            screen = (pair_decision(
                kind=kind, gap_cm=gap, coupling_cm=coupling, numerators_cm=None,
                criteria=criteria, denominator_tolerance_cm=denominator_tolerance_cm,
                numerical_zero_cm=numerical_zero_cm,
            ) if resonance_policy == "intensity" else {})
            candidates.append({"i": i, "j": j, "kind": kind, "gap_cm": gap,
                               "coupling_cm": coupling, "reduced_coupling_cm": reduced,
                               "martin_cm": martin, "wavefunction_ratio": mixing,
                               **screen, "decision": decision})

    def directed_second(i: int, j: int) -> tuple[complex,complex]:
        return effective_operator.directed_second(states[i],states[j])

    def second_order(i: int,j: int) -> complex:
        return effective_operator.second_order(states[i],states[j])

    diagonal=np.array([effective_operator.diagonal(state) for state in states])

    def dd_coupling(i: int,j: int,kind: str) -> float:
        nonlocal quartic_terms
        value=effective_operator.dd_coupling(states[i],states[j],kind)
        quartic_terms=workspace.get('quartic_terms')
        return value

    edges: list[tuple[int, int, float]] = []
    for item in candidates:
        if item["decision"] == "accepted":
            edges.append((int(item["i"]), int(item["j"]), float(item["coupling_cm"])))
    dd_kinds = frozenset(("1-1", "2-2", "1-3"))
    dd_pairs = (() if resonance_policy in {"none", "fermi", "fermi_auto"} else
                _candidate_state_pairs(states, max_quanta, kinds=dd_kinds))
    for i, j, kind in dd_pairs:
            if any(states[i][mode] or states[j][mode] for mode in excluded_modes):
                continue
            candidate_count += 1
            a, b = states[i], states[j]
            gap = abs(harmonic[i] - harmonic[j])
            if resonance_policy in {"none", "fermi", "fermi_auto"}:
                continue
            if resonance_policy in {"auto", "fermi", "gdv", "intensity"} and gap > dd_window_cm:
                candidates.append({"i": i, "j": j, "kind": kind, "gap_cm": gap,
                                   "coupling_cm": 0.0, "decision": "outside window"})
                continue
            if (field.quartic_coverage == "semidiagonal"
                    and not _semidiagonal_quartic_edge_available(a, b, kind)):
                candidates.append({
                    "i": i, "j": j, "kind": kind, "gap_cm": gap,
                    "coupling_cm": 0.0, "decision": "not_acquired",
                })
                continue
            coupling = dd_coupling(i, j, kind)
            screening_coupling = abs(coupling) * (
                sqrt(8) if kind == "2-2" and ((2 in a) != (2 in b)) else 1.0
            )
            mixing = abs(coupling) / max(
                abs(diagonal[i] - diagonal[j]), denominator_tolerance_cm,
            )
            accepted = (
                resonance_policy in {"auto", "all", "gdv"}
                and (
                    resonance_policy == "all"
                    or resonance_policy == "gdv" and (
                        screening_coupling >= 10.0
                        or kind == "1-1" and mixing >= 0.3
                    )
                    or resonance_policy == "auto" and abs(coupling) >= dd_coupling_threshold_cm
                )
                and abs(coupling) >= numerical_zero_cm
            )
            screen = {}
            if resonance_policy == "intensity":
                forward, reverse = directed_second(i, j)
                difference = (forward - np.conjugate(reverse)) / 2
                numerators = (coupling + difference, coupling - np.conjugate(difference))
                screen = pair_decision(
                    kind=kind, gap_cm=gap, coupling_cm=coupling,
                    numerators_cm=numerators, criteria=criteria,
                    denominator_tolerance_cm=denominator_tolerance_cm,
                    numerical_zero_cm=numerical_zero_cm,
                )
                screen["numerator_forward_cm"] = float(np.real_if_close(numerators[0]).real)
                screen["numerator_reverse_cm"] = float(np.real_if_close(numerators[1]).real)
                accepted = screen["decision"] == "accepted"
            decision = "accepted" if accepted else (
                "policy" if resonance_policy in {"none", "fermi"} else
                "coupling test" if abs(coupling) < dd_coupling_threshold_cm
                and resonance_policy == "auto" else "numerical zero"
            )
            candidates.append({"i": i, "j": j, "kind": kind,
                               "gap_cm": gap, "coupling_cm": coupling,
                               "screening_coupling_cm": screening_coupling,
                               "wavefunction_ratio": mixing,
                               "decision": decision, **screen})
            if accepted:
                edges.append((i, j, coupling))

    candidates.sort(key=lambda item: (item["kind"] != "1-2", item["i"], item["j"]))
    coupling_time = perf_counter() - coupling_start
    diagonalization_start = perf_counter()

    adjacency = [set() for _ in states]
    edge_values: dict[tuple[int, int], float] = {}
    for i, j, coupling in edges:
        adjacency[i].add(j)
        adjacency[j].add(i)
        edge_values[(min(i, j), max(i, j))] = coupling
    blocks: list[dict[str, object]] = []
    energies = diagonal.copy()
    seen: set[int] = set()
    for start in range(len(states)):
        if start in seen or not adjacency[start]:
            continue
        stack, component = [start], set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency[node] - component)
        seen.update(component)
        members = sorted(component)
        local = np.diag(diagonal[members])
        positions = {index: position for position, index in enumerate(members)}
        for i in members:
            for j in adjacency[i]:
                if j > i:
                    coupling = edge_values[(i, j)]
                    local[positions[i], positions[j]] = coupling
                    local[positions[j], positions[i]] = coupling
        # A weak 1:1 coupling between fundamentals already in the same
        # polyad still belongs to its effective Hamiltonian.  Do not extend
        # this closure to arbitrary two-quantum spectator-state pairs: those
        # are not the selected 2:2 Darling–Dennison resonances.
        for position, i in enumerate(members):
            for j in members[position + 1:]:
                if (i, j) in edge_values:
                    continue
                if (sum(states[i]) != 1 or sum(states[j]) != 1
                        or abs(harmonic[i] - harmonic[j]) > dd_window_cm):
                    continue
                coupling = dd_coupling(i, j, "1-1")
                if abs(coupling) >= numerical_zero_cm:
                    local[positions[i], positions[j]] = coupling
                    local[positions[j], positions[i]] = coupling
        eigenvalues, eigenvectors = np.linalg.eigh(local)
        basis_to_root = _maximum_overlap_assignment(np.abs(eigenvectors) ** 2)
        for position, index in enumerate(members):
            root = basis_to_root[position]
            energies[index] = eigenvalues[root]
        blocks.append({"indices": tuple(members), "hamiltonian_cm": local,
                       "eigenvalues_cm": eigenvalues, "eigenvectors": eigenvectors,
                       "basis_to_root": basis_to_root})
        if resonance_policy == "intensity":
            blocks[-1]["direct_resonance_states"] = tuple(
                (states[i], states[j]) for i, j, _coupling in edges if i in component and j in component
            )

    ground = indices[tuple(0 for _ in frequencies)]
    zero_point = float(energies[ground] + np.sum(frequencies) / 2)
    relative_diagonal = diagonal - diagonal[ground]
    relative_energies = energies - energies[ground]
    fundamentals = np.array([
        relative_energies[indices[tuple(int(j == i) for j in range(len(frequencies)))]]
        for i in range(len(frequencies))
    ])
    relative_blocks = [dict(block, eigenvalues_cm=np.asarray(block["eigenvalues_cm"]) - energies[ground])
                       for block in blocks]
    diagonalization_time = perf_counter() - diagonalization_start
    diagnostics = {
        "effective_operator_profile":dict(resonance_policy=resonance_policy,
            fermi_window_cm=fermi_window_cm,dd_window_cm=dd_window_cm,
            martin_threshold_cm=martin_threshold_cm,numerical_zero_cm=numerical_zero_cm,
            denominator_tolerance_cm=denominator_tolerance_cm,
            polyad_excluded_modes=polyad_excluded_modes,resonance_criteria=resonance_criteria),
        "candidate_pair_count": candidate_count,
        "accepted_pair_count": len(edges),
        "resonant_block_count": len(blocks),
        "largest_polyad_dimension": max((len(block["indices"]) for block in blocks), default=1),
        "polyad_excluded_modes": tuple(sorted(excluded_modes)),
        "max_mode_quanta": max_mode_quanta,
        "canonical_cubic_term_count": len(cubic_terms),
        "canonical_quartic_term_count": 0 if quartic_terms is None else len(quartic_terms),
        "timings_s": {
            "modalization": modalization_time,
            "state_basis": basis_time,
            "coupling_assembly": coupling_time,
            "polyad_diagonalization": diagonalization_time,
        },
    }
    if resonance_policy == "intensity":
        diagnostics["resonance_screening"] = {
            "profile": INTENSITY_RESONANCE_PROFILE,
            "energy_tolerance_cm": criteria.energy_tolerance_cm,
            "first_order_limit": criteria.first_order_limit,
            "second_order_limit": criteria.second_order_limit,
            "fermi_window_cm": fermi_window_cm, "dd_window_cm": dd_window_cm,
            "numerical_zero_cm": numerical_zero_cm,
            "denominator_tolerance_cm": denominator_tolerance_cm,
            "directed_second_order_cache_size": len(workspace.get("directed_second_order", {})),
        }
    report = _gvpt2_report(states, frequencies, relative_diagonal, relative_energies,
                           candidates, relative_blocks, diagnostics)
    result = SonicGVPT2Result(
        frequencies_cm=frequencies, states=states,
        deperturbed_cm=relative_diagonal, energies_cm=relative_energies,
        fundamentals_cm=fundamentals, resonances=tuple(candidates),
        blocks=tuple(relative_blocks), zero_point_cm=zero_point, report=report,
        diagnostics=diagnostics,
    )
    if targeted_polyad_options is not None:
        from .targeted_polyads import refine_sonic_gvpt2_polyads

        refinement_options=dict(targeted_polyad_options)
        if any(key in refinement_options for key in ('require_convergence','return_state')):
            raise ValueError('diagnostic partial results require refine_sonic_gvpt2_polyads directly')
        refinement_targets=refinement_options.pop('target_states',tuple(state for state in states
            if not any(state[mode] for mode in polyad_excluded_modes)))
        refinement_options.setdefault('maximum_states',max_states)
        refinement_options.setdefault('maximum_quanta',max_quanta)
        result=refine_sonic_gvpt2_polyads(field,result,target_states=refinement_targets,
            maximum_mode_quanta=max_mode_quanta,polyad_excluded_modes=polyad_excluded_modes,
            **refinement_options)
    if chart_connection is not None:
        from .coordinate_gauge import transport_gvpt2_polyads

        return transport_gvpt2_polyads(
            result, field, chart_connection, symmetry_actions=symmetry_actions,
        )
    return result


def _resonance_neighbors(
    state: tuple[int, ...], frequencies: np.ndarray, *,
    fermi_window_cm: float, dd_window_cm: float, include_dd: bool,
    max_candidates: int, f3: np.ndarray | None = None,
    g1: np.ndarray | None = None, resonance_policy: str = "gdv",
    martin_threshold_cm: float = 1.0,
    polyad_excluded_modes: tuple[int, ...] = (),
    max_total_quanta: int | None = None,
    max_mode_quanta: int | None = None,
    resonance_criteria: Mapping[str, float] | None = None,
    denominator_tolerance_cm: float = 1e-6,
    numerical_zero_cm: float | None = None,
) -> set[tuple[int, ...]]:
    """Generate one interaction shell without an excitation-order cutoff."""
    n = len(state)
    criteria = IntensityResonanceCriteria.from_options(resonance_criteria)
    if numerical_zero_cm is None:
        numerical_zero_cm = 1e-12 if resonance_policy == "intensity" else 0.01
    neighbors: set[tuple[int, ...]] = set()
    classes = ((1, 2, fermi_window_cm), (2, 1, fermi_window_cm))
    if include_dd:
        classes += ((1, 1, dd_window_cm), (2, 2, dd_window_cm),
                    (1, 3, dd_window_cm), (3, 1, dd_window_cm))
    for removed_count, added_count, window in classes:
        occupied = tuple(i for i, count in enumerate(state) for _ in range(count))
        for removed in set(combinations(occupied, removed_count)):
            base = list(state)
            for mode in removed:
                base[mode] -= 1
            for added in combinations_with_replacement(range(n), added_count):
                trial = base.copy()
                for mode in added:
                    trial[mode] += 1
                candidate = tuple(trial)
                if any(candidate[mode] for mode in polyad_excluded_modes):
                    continue
                if (max_total_quanta is not None and sum(candidate) > max_total_quanta
                        or max_mode_quanta is not None and max(candidate) > max_mode_quanta):
                    continue
                if (candidate == state
                        or _resonance_kind(state, candidate) not in {"1-2", "1-1", "2-2", "1-3"}
                        or abs(float(np.dot(frequencies, np.subtract(state, candidate)))) > window):
                    continue
                if (removed_count, added_count) in {(1, 2), (2, 1)} and f3 is not None:
                    if g1 is None:
                        raise ValueError("Fermi screening requires both F3 and G'")
                    gap = abs(float(np.dot(frequencies, np.subtract(state, candidate))))
                    reduced = abs(_fermi_reduced_coupling(state, candidate, f3, g1))
                    if resonance_policy == "intensity":
                        occupation_factor = sqrt(np.prod([
                            factorial(max(a, b)) / (factorial(min(a, b)) * factorial(abs(a - b)))
                            for a, b in zip(state, candidate, strict=True)
                        ]))
                        decision = pair_decision(
                            kind="1-2", gap_cm=gap, coupling_cm=reduced * occupation_factor,
                            numerators_cm=None, criteria=criteria,
                            denominator_tolerance_cm=denominator_tolerance_cm,
                            numerical_zero_cm=numerical_zero_cm,
                        )
                        if decision["decision"] != "accepted":
                            continue
                    martin = reduced**4 / gap**3 if gap > 0 else (
                        np.inf if reduced else 0.0
                    )
                    if (resonance_policy == "gdv" and martin < martin_threshold_cm
                            and reduced / max(gap, 1e-6) < 0.1):
                        continue
                    if (resonance_policy in {"auto", "fermi_auto"}
                            and martin < martin_threshold_cm):
                        continue
                neighbors.add(candidate)
                if len(neighbors) > max_candidates:
                    raise ValueError("adaptive polyad candidate budget exceeded before convergence")
    return neighbors


def solve_sonic_gvpt2_adaptive(
    field: SonicVPT2Input, *,
    seed_states: tuple[tuple[int, ...], ...] | None = None,
    target_max_quanta: int = 2,
    spectral_window_cm: tuple[float, float] | None = None,
    energy_tolerance_cm: float,
    observable_tolerance: float | None = None,
    observable: Callable[
        [SonicGVPT2Result, tuple[int, ...]], Mapping[tuple[int, ...], float]
    ] | None = None,
    stable_expansions: int = 2,
    max_expansions: int = 12,
    max_states: int = 30000,
    resonance_policy: str = "gdv",
    fermi_window_cm: float = 200.0,
    dd_window_cm: float = 100.0,
    polyad_excluded_modes: tuple[int, ...] = (),
    max_total_quanta: int = 4,
    max_mode_quanta: int = 4,
    targeted_polyad_options: Mapping[str, object] | None = None,
    **solver_options: object,
) -> SonicGVPT2Result:
    """Grow each requested resonance component until its observables converge.

    ``targeted_polyad_options`` applies the same production refinement used
    by the fixed solver after initial graph discovery.

    ``max_states`` and ``max_expansions`` are fail-closed resource guards.
    They never turn an unconverged component into a reported spectrum.
    """
    if (not np.isfinite(energy_tolerance_cm) or energy_tolerance_cm <= 0
            or stable_expansions < 1 or max_expansions < stable_expansions
            or max_states < 1 or target_max_quanta not in {2, 3}
            or max_total_quanta < target_max_quanta or max_mode_quanta < 1):
        raise ValueError("invalid adaptive polyad convergence or resource limit")
    if (observable is None) != (observable_tolerance is None):
        raise ValueError("observable and its convergence tolerance must be supplied together")
    if observable_tolerance is not None and (not np.isfinite(observable_tolerance)
                                              or observable_tolerance <= 0):
        raise ValueError("observable tolerance must be finite and positive")
    workspace: dict[str, object] = {"modalized": _modalize_sonic_field(field)}
    frequencies, _modes, f3q, _f4q, g1q, _g2q = workspace["modalized"]
    n = len(frequencies)
    excluded_modes = frozenset(int(mode) for mode in polyad_excluded_modes)
    if (len(excluded_modes) != len(polyad_excluded_modes)
            or any(mode < 0 or mode >= n for mode in excluded_modes)):
        raise ValueError("invalid polyad-excluded mode indices")
    fundamentals = tuple(tuple(int(i == j) for i in range(n)) for j in range(n))
    ground = tuple(0 for _ in range(n))
    targets: set[tuple[int, ...]] = set()
    if spectral_window_cm is None and seed_states is None:
        targets.update(
            state for state in fundamentals
            if not any(state[mode] for mode in excluded_modes)
        )
        for quanta in range(2, target_max_quanta + 1):
            for occupied in combinations_with_replacement(range(n), quanta):
                counts = [0] * n
                for mode in occupied:
                    counts[mode] += 1
                if (any(counts[mode] for mode in excluded_modes)
                        or max(counts) > max_mode_quanta):
                    continue
                targets.add(tuple(counts))
                if len(targets) > max_states:
                    raise ValueError("adaptive target manifold exceeds max_states")
    if spectral_window_cm is not None:
        low, high = spectral_window_cm
        if (not np.isfinite(low) or not np.isfinite(high)
                or low < 0 or high < low or np.any(frequencies <= 0)):
            raise ValueError("invalid adaptive spectral window")
        # A requested spectral window selects targets in that window; states
        # elsewhere remain possible resonance partners, not convergence goals.
        stack = [(0, ground, 0.0)]
        visited = 0
        while stack:
            first_mode, state, energy = stack.pop()
            visited += 1
            if visited > max_states:
                raise ValueError("adaptive spectral-window enumeration exceeded max_states")
            for mode in range(first_mode, n):
                if mode in excluded_modes:
                    continue
                new_energy = energy + float(frequencies[mode])
                if (new_energy > high or sum(state) + 1 > max_total_quanta
                        or state[mode] + 1 > max_mode_quanta):
                    continue
                counts = list(state)
                counts[mode] += 1
                candidate = tuple(counts)
                if new_energy >= low:
                    targets.add(candidate)
                    if len(targets) > max_states:
                        raise ValueError("adaptive spectral targets exceed max_states")
                stack.append((mode, candidate, new_energy))
    if seed_states is not None:
        if any(any(state[mode] for mode in excluded_modes)
               or sum(state) > max_total_quanta or max(state) > max_mode_quanta
               for state in seed_states):
            raise ValueError("polyad seed violates the mode or quantum limits")
        targets.update(seed_states)
    seeds = tuple(sorted(targets, key=lambda state: (sum(state), state)))
    if not seeds or any(len(state) != n or any(v < 0 for v in state) for state in seeds):
        raise ValueError("adaptive polyad requires valid seed states")
    selected = {ground, *fundamentals, *seeds}
    finalized: set[tuple[int, ...]] = set()
    history: list[dict[str, object]] = []
    converged_values: dict[
        tuple[int, ...], tuple[dict[tuple[int, ...], float], dict[tuple[int, ...], float] | None]
    ] = {}

    def solve(candidate_set: set[tuple[int, ...]]) -> SonicGVPT2Result:
        if len(candidate_set) > max_states:
            raise ValueError("adaptive polyad candidate budget exceeded before convergence")
        states = (ground, *fundamentals, *sorted(candidate_set - {ground, *fundamentals}))
        return solve_sonic_gvpt2(
            field, max_quanta=max_total_quanta, max_states=max_states,
            resonance_policy=resonance_policy, fermi_window_cm=fermi_window_cm,
            dd_window_cm=dd_window_cm, candidate_states=states,
            polyad_excluded_modes=tuple(sorted(excluded_modes)),
            max_mode_quanta=max_mode_quanta,
            _workspace=workspace,
            **solver_options,
        )

    def component(result: SonicGVPT2Result, seed: tuple[int, ...]) -> set[tuple[int, ...]]:
        index = result.states.index(seed)
        for block in result.blocks:
            if index in block["indices"]:
                return {result.states[int(i)] for i in block["indices"]}
        return {seed}

    include_dd = resonance_policy not in {"none", "fermi", "fermi_auto"}
    def reported_observables(
        result: SonicGVPT2Result, values: Mapping[tuple[int, ...], float],
    ) -> dict[tuple[int, ...], float]:
        indices = {state: index for index, state in enumerate(result.states)}
        return {
            state: value for state, value in values.items()
            if state in targets or (
                abs(value) >= observable_tolerance
                and (spectral_window_cm is None
                     or spectral_window_cm[0] <= result.energies_cm[indices[state]]
                     <= spectral_window_cm[1])
            )
        }

    for seed in seeds:
        if seed in finalized:
            continue
        members = {seed}
        previous_energies: dict[tuple[int, ...], float] | None = None
        previous_observable: dict[tuple[int, ...], float] | None = None
        stable = 0
        for iteration in range(1, max_expansions + 1):
            # Solve the active component locally; the global target set is
            # retained for the final spectrum, not rebuilt for every seed.
            candidates = {ground, *fundamentals, *members}
            for state in members:
                candidates.update(_resonance_neighbors(
                    state, frequencies, fermi_window_cm=fermi_window_cm,
                    dd_window_cm=dd_window_cm, include_dd=include_dd,
                    max_candidates=max_states, f3=f3q, g1=g1q,
                    resonance_policy=resonance_policy,
                    martin_threshold_cm=float(solver_options.get("martin_threshold_cm", 1.0)),
                    polyad_excluded_modes=tuple(sorted(excluded_modes)),
                    max_total_quanta=max_total_quanta,
                    max_mode_quanta=max_mode_quanta,
                    resonance_criteria=solver_options.get("resonance_criteria"),
                    denominator_tolerance_cm=float(solver_options.get("denominator_tolerance_cm", 1e-6)),
                    numerical_zero_cm=solver_options.get("numerical_zero_cm"),
                ))
                if len(candidates) > max_states:
                    raise ValueError(
                        f"adaptive polyad candidate budget exceeded before convergence "
                        f"for seed {seed} at expansion {iteration}: {len(candidates)} states "
                        f"from {len(members)} component states"
                    )
            result = solve(candidates)
            updated = component(result, seed)
            energies = {state: float(result.energies_cm[result.states.index(state)])
                        for state in updated}
            values = None if observable is None else dict(observable(result, seed))
            if values is not None and (not set(values).issubset(updated)
                                       or not all(np.isfinite(value) for value in values.values())):
                raise ValueError("adaptive polyad observable is not finite")
            if values is not None:
                values = reported_observables(result, values)
            tracked = (updated & targets) | (
                set() if values is None else
                set(values)
            )
            energy_change = (None if previous_energies is None else
                             max((abs(energies[state] - previous_energies[state])
                                  for state in tracked & previous_energies.keys()),
                                 default=0.0))
            observable_change = (None if values is None or previous_observable is None
                                 else max((abs(values.get(state, 0.0)
                                               - previous_observable.get(state, 0.0))
                                           for state in values.keys() | previous_observable.keys()),
                                          default=0.0))
            history.append({"seed": seed, "iteration": iteration,
                            "members": tuple(sorted(updated)),
                            "energy_change_cm": energy_change,
                            "observable_change": observable_change})
            selected.update(updated)
            if updated == members:
                finalized.update(updated)
                converged_values[seed] = ({state: energies[state] for state in tracked}, values)
                break
            if (values is not None and energy_change is not None
                    and energy_change <= energy_tolerance_cm
                    and (observable_change is None
                         or observable_change <= observable_tolerance)):
                stable += 1
            else:
                stable = 0
            if stable >= stable_expansions:
                finalized.update(updated)
                converged_values[seed] = ({state: energies[state] for state in tracked}, values)
                break
            members = updated
            previous_energies = energies
            previous_observable = values
        else:
            last = history[-1]
            raise ValueError(
                f"adaptive polyad for {seed} did not converge after "
                f"{max_expansions} expansions: {len(last['members'])} members, "
                f"energy change {last['energy_change_cm']}, "
                f"observable change {last['observable_change']}"
            )
    final = solve(selected)
    for seed, (energies, values) in converged_values.items():
        if any(abs(float(final.energies_cm[final.states.index(state)]) - energy)
               > energy_tolerance_cm for state, energy in energies.items()):
            raise ValueError(f"later polyad expansion changed converged energy for {seed}")
        if values is not None and observable is not None:
            final_values = reported_observables(final, dict(observable(final, seed)))
            if max((abs(final_values.get(state, 0.0) - values.get(state, 0.0))
                    for state in final_values.keys() | values.keys()), default=0.0) > observable_tolerance:
                raise ValueError(f"later polyad expansion changed converged observable for {seed}")
    boundary_states = tuple(sorted({
        final.states[int(index)]
        for block in final.blocks for index in block["indices"]
        if (sum(final.states[int(index)]) == max_total_quanta
            or max(final.states[int(index)]) == max_mode_quanta)
    }))
    final = replace(final, diagnostics={**final.diagnostics,
                                       "adaptive_polyad_history": tuple(history),
                                       "adaptive_target_count": len(seeds),
                                       "adaptive_energy_tolerance_cm": energy_tolerance_cm,
                                       "adaptive_observable_tolerance": observable_tolerance,
                                       "max_total_quanta": max_total_quanta,
                                       "max_mode_quanta": max_mode_quanta,
                                       "occupation_boundary_states": boundary_states})
    if targeted_polyad_options is not None:
        from .targeted_polyads import refine_sonic_gvpt2_polyads

        refinement_options=dict(targeted_polyad_options)
        if any(key in refinement_options for key in ('require_convergence','return_state')):
            raise ValueError('diagnostic partial results require refine_sonic_gvpt2_polyads directly')
        refinement_targets=refinement_options.pop('target_states', seeds)
        refinement_options.setdefault('maximum_states',max_states)
        refinement_options.setdefault('maximum_quanta',max_total_quanta)
        final=refine_sonic_gvpt2_polyads(field,final,target_states=refinement_targets,
            maximum_mode_quanta=max_mode_quanta,polyad_excluded_modes=polyad_excluded_modes,
            **refinement_options)
    return final


def acquire_sonic_vpt2_from_hessians(
    *,
    hessian_at_s: Callable[[np.ndarray], np.ndarray],
    gradient_at_s: Callable[[np.ndarray], np.ndarray] | None = None,
    energy_at_s: Callable[[np.ndarray], float | None] | None = None,
    metric_at_s: Callable[[np.ndarray], np.ndarray] | None = None,
    metric_first_at_s: Callable[[np.ndarray], np.ndarray] | None = None,
    b_matrix_at_s: Callable[[np.ndarray], np.ndarray] | None = None,
    b_prime_cartesian_at_s: Callable[[np.ndarray], np.ndarray] | None = None,
    tangent_at_s: Callable[[np.ndarray], np.ndarray] | None = None,
    geometry_at_s: Callable[[np.ndarray], np.ndarray] | None = None,
    masses_amu: np.ndarray | None = None,
    coordinate_count: int,
    step_dimensionless: float = 0.02,
    step_mass_weighted_angstrom: float | None = None,
    cubic_estimator_policy: str = "hessian_gradient_mean",
    force_constant_cutoff_au: float = 0.0,
    final_force_constant_cutoff_au: float | None = None,
    geometry_derivative_step_s: float = 1e-4,
    rotational_dof: int = 3,
    denominator_tolerance_cm: float = 1e-6,
    complete_quartic: bool = False,
    fermi_only: bool = False,
    single_mode_only: bool = False,
    mode_contribution_cache: object | None = None,
    return_field: bool = False,
) -> SonicVPT2Result | SonicAcquiredField:
    """Acquire F3/F4, kinetic derivatives and Vg in one SONIC Hessian run.

    Callbacks receive the displacement in the *frozen nonredundant SONIC*
    coordinates. ``hessian_at_s`` must return the curvilinear Hessian, including
    the gradient-times-second-backtransform term away from equilibrium (see
    ``matrix_smith.curvilinear_hessian``). When ``geometry_at_s`` and masses
    are supplied, TRINITY builds the complete rovibrational metric from the
    body-fixed Cartesian t vectors at every stencil point. The geometry map
    must retain a smooth body-fixed orientation across the stencil. An
    already validated ``metric_at_s`` provider may be used instead.
    Mixed metric differences are inexpensive and require no extra Hessians.
    Alternatively, B and Cartesian B' providers derive vibrational G' by
    the mass-weighted Wilson product rule. B and B' use bohr coordinates.
    If an analytic ``metric_first_at_s`` is provided, its shape is either
    ``(vibrational, vibrational, vibrational)`` or
    ``(vibrational + rotational, vibrational + rotational, vibrational)``;
    G'' is obtained from G' at the same single-mode plus/minus points as the
    Hessians, without pair-mode metric evaluations.
    By default, only the reference and plus/minus one normal-mode
    displacement acquire semidiagonal F4 (F_ijkk) and G'' (G_ij,kk).
    Both independent estimates of each F_iijj are averaged, as in L717.
    When gradients accompany the Hessians, their central second difference
    supplies one additional estimate for each F_ikk; all available estimates
    are averaged. Energies only diagnose diagonal quartic consistency.
    ``complete_quartic`` adds pair-mode Hessian and metric displacements;
    a semidiagonal field returned for GVPT2 requires
    ``fermi_only`` to exclude quartic Darling-Dennison couplings.
    The final symmetric F3/F4 cutoff defaults to the directional cutoff;
    set ``final_force_constant_cutoff_au=0`` to disable only this second
    screen, or supply a different nonnegative threshold in atomic units.
    """
    n = int(coordinate_count)
    step = float(step_dimensionless)
    if n < 1 or rotational_dof not in {2, 3} or not np.isfinite(step) or step <= 0:
        raise ValueError("invalid SONIC VPT2 coordinate count, rotation count or step")
    if cubic_estimator_policy not in {"hessian_mean", "hessian_gradient_mean"}:
        raise ValueError("invalid cubic estimator policy")
    if not np.isfinite(force_constant_cutoff_au) or force_constant_cutoff_au < 0:
        raise ValueError("force-constant cutoff must be finite and nonnegative")
    final_cutoff_au = (float(force_constant_cutoff_au) if final_force_constant_cutoff_au is None
                       else float(final_force_constant_cutoff_au))
    if not np.isfinite(final_cutoff_au) or final_cutoff_au < 0:
        raise ValueError("final force-constant cutoff must be finite and nonnegative")
    if single_mode_only and complete_quartic:
        raise ValueError("single-mode stencil cannot acquire the complete quartic field")
    if return_field and not complete_quartic and not fermi_only:
        raise ValueError("semidiagonal GVPT2 field requires Fermi-only acquisition")
    if (b_matrix_at_s is None) != (b_prime_cartesian_at_s is None):
        raise ValueError("analytic Wilson metric requires both B and B prime providers")
    if b_matrix_at_s is not None:
        if metric_first_at_s is not None:
            raise ValueError("provide either G prime or B/B prime, not both")
        if geometry_at_s is None or masses_amu is None:
            raise ValueError("B/B prime metric requires geometry and masses")
        tangent_step = float(geometry_derivative_step_s)
        if not np.isfinite(tangent_step) or tangent_step <= 0:
            raise ValueError("B/B prime metric requires a positive geometry step")

        def derived_metric_first(displacement: np.ndarray) -> np.ndarray:
            tangent = (np.asarray(tangent_at_s(displacement), dtype=float)
                       if tangent_at_s is not None else np.column_stack([
                           ((np.asarray(geometry_at_s(displacement + tangent_step * np.eye(n)[k]))
                             - np.asarray(geometry_at_s(displacement - tangent_step * np.eye(n)[k])))
                            / (2 * tangent_step)).reshape(-1)
                           for k in range(n)
                       ]))
            return wilson_metric_first_from_b_prime(
                b_matrix_at_s(displacement), b_prime_cartesian_at_s(displacement),
                tangent, masses_amu,
            )

        metric_first_at_s = derived_metric_first
    metric_from_geometry = metric_at_s is None
    if metric_from_geometry:
        if geometry_at_s is None or masses_amu is None:
            raise ValueError("SONIC VPT2 requires a geometry/masses or full metric provider")
        metric_at_s = lambda displacement: sonic_rovibrational_metric(
            geometry_at_s, displacement, masses_amu,
            rotational_dof=rotational_dof, tangent_at_s=tangent_at_s,
        )

    def hessian(displacement: np.ndarray) -> np.ndarray:
        value = np.asarray(hessian_at_s(displacement), dtype=float)
        if value.shape != (n, n) or not np.all(np.isfinite(value)) or not np.allclose(value, value.T, atol=1e-8):
            raise ValueError("SONIC Hessian callback must return a finite symmetric matrix")
        return 0.5 * (value + value.T)

    def metric(displacement: np.ndarray) -> np.ndarray:
        value = np.asarray(metric_at_s(displacement), dtype=float)
        size = n + rotational_dof
        if value.shape != (size, size) or not np.all(np.isfinite(value)) or not np.allclose(value, value.T, atol=1e-8):
            raise ValueError("SONIC kinetic callback must return the full rovibrational metric")
        value = 0.5 * (value + value.T)
        sign, _ = np.linalg.slogdet(value)
        if sign <= 0 or np.min(np.linalg.eigvalsh(value)) <= 0:
            raise ValueError("SONIC rovibrational metric must be positive definite")
        return value

    def metric_first(displacement: np.ndarray) -> np.ndarray:
        if metric_first_at_s is None:
            raise RuntimeError("analytic metric derivative is not available")
        value = np.asarray(metric_first_at_s(displacement), dtype=float)
        size = n + rotational_dof
        if (value.shape not in {(n, n, n), (size, size, n)}
                or not np.all(np.isfinite(value))
                or not np.allclose(value, value.swapaxes(0, 1), atol=1e-8)):
            raise ValueError("SONIC analytic metric first derivative has invalid shape or symmetry")
        return value

    zero = np.zeros(n)
    f2 = hessian(zero)
    gradient0 = (None if gradient_at_s is None else
                 np.asarray(gradient_at_s(zero), dtype=float).reshape(-1))
    if gradient0 is not None and (gradient0.shape != (n,)
                                   or not np.all(np.isfinite(gradient0))):
        raise ValueError("SONIC gradient callback must return a finite vector")
    energy0 = None if energy_at_s is None else energy_at_s(zero)
    if energy0 is not None and not np.isfinite(energy0):
        raise ValueError("SONIC energy callback must return a finite value")
    full_metric = metric(zero)
    vibrational = full_metric[:n, :n]
    values, vectors = np.linalg.eigh(vibrational)
    ghalf = (vectors * np.sqrt(values)) @ vectors.T
    eigenvalues, eigenvectors = np.linalg.eigh(ghalf @ f2 @ ghalf)
    if np.any(eigenvalues <= 0):
        raise ValueError("SONIC VPT2 Hessian stencil requires a minimum")
    frequencies = np.sqrt(eigenvalues) * HESSIAN_EIGENVALUE_TO_CM
    modes = ghalf @ eigenvectors
    inverse_modes = np.linalg.inv(modes)
    gamma = np.sqrt(frequencies * CM_PER_HARTREE) / HESSIAN_EIGENVALUE_TO_CM
    if step_mass_weighted_angstrom is not None:
        from matrix_chem import BOHR_TO_ANGSTROM
        physical_step = float(step_mass_weighted_angstrom)
        if not np.isfinite(physical_step) or physical_step <= 0:
            raise ValueError("mass-weighted stencil step must be positive")
        mode_steps = gamma * physical_step / BOHR_TO_ANGSTROM
    else:
        mode_steps = np.full(n, step)
    displacement_per_q = modes / gamma[None, :]
    momentum = gamma[:, None] * inverse_modes
    metric_scale = HESSIAN_EIGENVALUE_TO_CM**2 / CM_PER_HARTREE

    def modal_metric(full: np.ndarray) -> np.ndarray:
        return metric_scale * momentum @ full[:n, :n] @ momentum.T

    def modal_metric_first(full_first: np.ndarray) -> np.ndarray:
        return metric_scale * np.einsum(
            "ia,jb,abc,ck->ijk", momentum, momentum,
            full_first[:n, :n, :], displacement_per_q, optimize=True,
        )

    modal_hessian0 = CM_PER_HARTREE * displacement_per_q.T @ f2 @ displacement_per_q
    modal_metric0 = modal_metric(full_metric)
    if not np.allclose(modal_metric0, np.diag(frequencies), rtol=1e-8, atol=1e-6):
        raise ValueError("SONIC GF modes do not diagonalize the kinetic metric")
    f3 = np.zeros((n, n, n))
    f4 = np.zeros((n, n, n, n))
    gradient_cubic = np.zeros((n, n)) if gradient0 is not None else None
    energy_gradient_quartic = np.full(n, np.nan) if energy0 is not None else None
    gradient_quartic = np.full(n, np.nan) if gradient0 is not None else None
    g1 = np.zeros((n, n, n))
    g2 = np.zeros((n, n, n, n))
    logdet0 = np.linalg.slogdet(full_metric)[1]
    logdet_first = np.zeros(n)
    determinant_second = np.zeros(n)
    hessian_evaluations = 1
    metric_evaluations = 1
    metric_first_evaluations = 0
    if metric_first_at_s is not None:
        g1 = modal_metric_first(metric_first(zero))
        metric_first_evaluations += 1
    mode_cache_signature = (
        mode_contribution_cache.signature(
            f2, full_metric, displacement_per_q,
            step=(mode_steps if step_mass_weighted_angstrom is not None else step),
            analytic_metric=metric_first_at_s is not None,
        ) + f":quartic-mean-gradient-{gradient0 is not None}-energy-{energy0 is not None}-v1"
        if mode_contribution_cache is not None else None
    )
    cached_modes = 0
    for k in range(n):
        step_k = mode_steps[k]
        saved = (mode_contribution_cache.load(k, mode_cache_signature)
                 if mode_contribution_cache is not None else None)
        if saved is not None:
            f3[:, :, k] = saved["f3"]
            f4[:, :, k, k] = saved["f4"]
            if gradient_cubic is not None:
                gradient_cubic[:, k] = saved["gradient_cubic"]
                gradient_quartic[k] = saved["gradient_quartic"]
            if energy_gradient_quartic is not None:
                energy_gradient_quartic[k] = saved["energy_gradient_quartic"]
            if metric_first_at_s is None:
                g1[:, :, k] = saved["g1"]
                g2[:, :, k, k] = saved["g2"]
            else:
                g2[:, :, :, k] = saved["g2"]
            logdet_first[k] = saved["logdet_first"]
            determinant_second[k] = saved["determinant_second"]
            cached_modes += 1
            continue
        displacement = step_k * displacement_per_q[:, k]
        # Evaluate the negative point first: a certified symmetry operation
        # can then supply the positive Cartesian E/G/H point without QM work.
        hm, hp = hessian(-displacement), hessian(displacement)
        mm, mp = metric(-displacement), metric(displacement)
        hessian_evaluations += 2
        metric_evaluations += 2
        hpq = CM_PER_HARTREE * displacement_per_q.T @ hp @ displacement_per_q
        hmq = CM_PER_HARTREE * displacement_per_q.T @ hm @ displacement_per_q
        f3[:, :, k] = (hpq - hmq) / (2 * step_k)
        f4[:, :, k, k] = (hpq - 2 * modal_hessian0 + hmq) / step_k**2
        if gradient_cubic is not None:
            gp = np.asarray(gradient_at_s(displacement), dtype=float).reshape(-1)
            gm = np.asarray(gradient_at_s(-displacement), dtype=float).reshape(-1)
            if (gp.shape != (n,) or gm.shape != (n,)
                    or not np.all(np.isfinite(gp)) or not np.all(np.isfinite(gm))):
                raise ValueError("SONIC displaced gradients must be finite vectors")
            gradient_cubic[:, k] = (
                CM_PER_HARTREE * displacement_per_q.T @ (gp + gm - 2 * gradient0)
                / step_k**2
            )
            gradient_slope = (
                CM_PER_HARTREE * displacement_per_q[:, k] @ (gp - gm)
                / (2 * step_k)
            )
            gradient_quartic[k] = 6 * (
                gradient_slope - modal_hessian0[k, k]
            ) / step_k**2
            if energy_gradient_quartic is not None:
                ep, em = energy_at_s(displacement), energy_at_s(-displacement)
                if ep is not None and em is not None and np.isfinite(ep) and np.isfinite(em):
                    energy_curvature = CM_PER_HARTREE * (ep + em - 2 * energy0) / step_k**2
                    energy_gradient_quartic[k] = 12 * (
                        gradient_slope - energy_curvature
                    ) / step_k**2
        if metric_first_at_s is None:
            g1[:, :, k] = (modal_metric(mp) - modal_metric(mm)) / (2 * step_k)
            g2[:, :, k, k] = (modal_metric(mp) - 2 * modal_metric0 + modal_metric(mm)) / step_k**2
        else:
            g2[:, :, :, k] = (
                modal_metric_first(metric_first(displacement))
                - modal_metric_first(metric_first(-displacement))
            ) / (2 * step_k)
            metric_first_evaluations += 2
        logp, logm = np.linalg.slogdet(mp)[1], np.linalg.slogdet(mm)[1]
        logdet_first[k] = (logp - logm) / (2 * step_k)
        determinant_second[k] = (
            (logp - 2 * logdet0 + logm) / step_k**2 + logdet_first[k] ** 2
        )
        if mode_contribution_cache is not None:
            mode_contribution_cache.store(k, mode_cache_signature, {
                "f3": f3[:, :, k], "f4": f4[:, :, k, k],
                "gradient_cubic": (gradient_cubic[:, k] if gradient_cubic is not None else None),
                "gradient_quartic": (gradient_quartic[k] if gradient_quartic is not None else None),
                "energy_gradient_quartic": (energy_gradient_quartic[k]
                                            if energy_gradient_quartic is not None else None),
                "g1": g1[:, :, k],
                "g2": (g2[:, :, :, k] if metric_first_at_s is not None
                       else g2[:, :, k, k]),
                "logdet_first": logdet_first[k],
                "determinant_second": determinant_second[k],
            })
    raw_cubic = f3.copy()
    f3 = sum(np.transpose(raw_cubic, axes) for axes in (
        (0, 1, 2), (0, 2, 1), (1, 0, 2),
        (1, 2, 0), (2, 0, 1), (2, 1, 0),
    )) / 6
    # A nominally zero cubic field has no meaningful relative symmetry error.
    # Check before directional thresholding, which can break raw symmetry.
    symmetry_residual = float(np.linalg.norm(raw_cubic - f3)
                              / max(np.linalg.norm(f3), 1e-8))
    if symmetry_residual > 0.01:
        raise ValueError("SONIC Hessian stencil has inconsistent cubic symmetry")
    # Screen directional estimates before their mean.  L717 also screens the
    # final symmetric F3/F4: a surviving direction can average below the
    # cutoff, as in the uracil F(1,1,6,6) checkpoint control.
    if force_constant_cutoff_au:
        cubic_scale = np.einsum("i,j,k->ijk", gamma, gamma, gamma) / CM_PER_HARTREE
        quartic_scale = np.einsum("i,j,k->ijk", gamma, gamma, gamma**2) / CM_PER_HARTREE
        raw_cubic[np.abs(raw_cubic * cubic_scale) < force_constant_cutoff_au] = 0.0
        for k in range(n):
            plane = f4[:, :, k, k]
            plane[np.abs(plane * quartic_scale[:, :, k]) < force_constant_cutoff_au] = 0.0
            if gradient_cubic is not None:
                gradient_scale = gamma * gamma[k] ** 2 / CM_PER_HARTREE
                gradient_cubic[
                    np.abs(gradient_cubic[:, k] * gradient_scale) < force_constant_cutoff_au,
                    k,
                ] = 0.0
        f3 = sum(np.transpose(raw_cubic, axes) for axes in (
            (0, 1, 2), (0, 2, 1), (1, 0, 2),
            (1, 2, 0), (2, 0, 1), (2, 1, 0),
        )) / 6
    for k in range(n):
        for i in range(n):
            estimates = [raw_cubic[i, k, k]]
            if i != k:
                estimates.append(raw_cubic[k, k, i])
            if gradient_cubic is not None and cubic_estimator_policy == "hessian_gradient_mean":
                estimates.append(gradient_cubic[i, k])
            value = float(np.mean(estimates))
            for axes in set(permutations((i, k, k))):
                f3[axes] = value
    if final_cutoff_au:
        final_cubic_scale = np.einsum("i,j,k->ijk", gamma, gamma, gamma) / CM_PER_HARTREE
        f3[np.abs(f3 * final_cubic_scale) < final_cutoff_au] = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            value = 0.5 * (f4[i, i, j, j] + f4[j, j, i, i])
            for axes in set(permutations((i, i, j, j))):
                f4[axes] = value
            if not single_mode_only and metric_first_at_s is None:
                dk = mode_steps[i] * displacement_per_q[:, i]
                dl = mode_steps[j] * displacement_per_q[:, j]
                mixed = (
                    modal_metric(metric(dk + dl)) - modal_metric(metric(dk - dl))
                    - modal_metric(metric(-dk + dl)) + modal_metric(metric(-dk - dl))
                ) / (4 * mode_steps[i] * mode_steps[j])
                metric_evaluations += 4
                g2[:, :, i, j] = g2[:, :, j, i] = mixed
            if complete_quartic:
                hp_pm = [
                    hessian(dk + dl), hessian(dk - dl),
                    hessian(-dk + dl), hessian(-dk - dl),
                ]
                hessian_evaluations += 4
                mixed_hessian = CM_PER_HARTREE * displacement_per_q.T @ (
                    hp_pm[0] - hp_pm[1] - hp_pm[2] + hp_pm[3]
                ) @ displacement_per_q / (4 * mode_steps[i] * mode_steps[j])
                f4[:, :, i, j] = f4[:, :, j, i] = mixed_hessian
    if complete_quartic:
        f4 = sum(np.transpose(f4, axes) for axes in set(permutations(range(4)))) / 24
    else:
        raw_semidiagonal = f4.copy()
        f4.fill(0.0)
        for indices in combinations_with_replacement(range(n), 4):
            repeated = {mode for mode in indices if indices.count(mode) >= 2}
            if not repeated:
                continue
            estimates = []
            for mode in repeated:
                remaining = list(indices)
                remaining.remove(mode)
                remaining.remove(mode)
                estimates.append(raw_semidiagonal[remaining[0], remaining[1], mode, mode])
            value = float(np.mean(estimates))
            for arrangement in set(permutations(indices)):
                f4[arrangement] = value
    if final_cutoff_au:
        pair_scale = np.outer(gamma, gamma) / CM_PER_HARTREE
        for i in range(n):
            for j in range(n):
                plane = f4[i, j]
                plane[np.abs(plane * (gamma[i] * gamma[j] * pair_scale))
                      < final_cutoff_au] = 0.0
    g2 = (g2 + g2.swapaxes(0, 1) + g2.swapaxes(2, 3)
          + g2.swapaxes(0, 1).swapaxes(2, 3)) / 4
    pseudopotential = float(np.sum(
        5 * frequencies * logdet_first**2
        - 4 * frequencies * determinant_second
        - 4 * np.einsum("iji,j->i", g1, logdet_first)
    ) / 32)
    acquisition = {
        "mode_basis": "nonredundant SONIC Wilson GF",
        "sonic_displacements_per_dimensionless_mode": displacement_per_q.tolist(),
        "hessian_evaluations": hessian_evaluations,
        "metric_evaluations": metric_evaluations,
        "metric_first_evaluations": metric_first_evaluations,
        "cached_mode_contributions": cached_modes,
        "gradient_cubic_estimates": gradient_cubic is not None,
        "gradient_diagonal_quartic_cm": (gradient_quartic.tolist()
                                         if gradient_quartic is not None else None),
        "energy_gradient_diagonal_quartic_cm": (
            energy_gradient_quartic.tolist()
            if energy_gradient_quartic is not None else None
        ),
        "step_dimensionless": step,
        "mode_steps_dimensionless": mode_steps.tolist(),
        "force_constant_cutoff_au": float(force_constant_cutoff_au),
        "final_force_constant_cutoff_au": final_cutoff_au,
        "cubic_symmetry_residual": symmetry_residual,
        "quartic_coverage": "complete F_ijkl" if complete_quartic else "semidiagonal F_ijkk",
        "metric_second_coverage": (
            "complete G_ij,kl from G' single-mode differences"
            if metric_first_at_s is not None else
            "single-mode G_ij,kk" if single_mode_only else "complete G_ij,kl"
        ),
        "pseudopotential_cm": pseudopotential,
        "metric_source": "body-fixed geometry t-vectors" if metric_from_geometry else "full metric callback",
    }
    if return_field:
        modal = SonicVPT2Input(
            harmonic_sonic=np.diag((frequencies / HESSIAN_EIGENVALUE_TO_CM) ** 2),
            cubic_sonic=np.einsum("ijk,i,j,k->ijk", f3, gamma, gamma, gamma) / CM_PER_HARTREE,
            quartic_sonic=np.einsum("ijkl,i,j,k,l->ijkl", f4, gamma, gamma, gamma, gamma) / CM_PER_HARTREE,
            metric_sonic=np.eye(n),
            metric_first_sonic=np.einsum("ijk,k,i,j->ijk", g1, gamma, 1 / gamma, 1 / gamma) / metric_scale,
            metric_second_sonic=np.einsum("ijkl,k,l,i,j->ijkl", g2, gamma, gamma, 1 / gamma, 1 / gamma) / metric_scale,
            pseudopotential_cm=pseudopotential,
            quartic_coverage="complete" if complete_quartic else "semidiagonal",
        )
        return SonicAcquiredField(modal, acquisition)
    result = _solve_modal_vpt2(
        frequencies, modes, f3, f4, g1, g2, pseudopotential,
        denominator_tolerance_cm=denominator_tolerance_cm,
        coefficient_tolerance_cm=0.0,
    )
    return replace(result, acquisition=acquisition)


def acquire_sonic_gvpt2_from_hessians(**kwargs: object) -> SonicGVPT2Result:
    """Acquire complete curvilinear F3/F4/G'/G'' and solve SONIC GVPT2."""
    options = dict(kwargs)
    solver_options = {name: options.pop(name) for name in (
        "max_quanta", "max_states", "resonance_policy", "fermi_window_cm",
        "dd_window_cm", "martin_threshold_cm", "dd_coupling_threshold_cm",
        "numerical_zero_cm", "resonance_criteria", "targeted_polyad_options",
    ) if name in options}
    if "denominator_tolerance_cm" in options:
        solver_options["denominator_tolerance_cm"] = options["denominator_tolerance_cm"]
    acquired = acquire_sonic_vpt2_from_hessians(
        **options, complete_quartic=True, return_field=True,
    )
    if not isinstance(acquired, SonicAcquiredField):
        raise RuntimeError("complete SONIC acquisition returned an incomplete field")
    return replace(solve_sonic_gvpt2(acquired.field, **solver_options),
                   acquisition=acquired.acquisition)


def wilson_metric_first_from_b_prime(
    b_matrix: np.ndarray,
    b_prime_cartesian: np.ndarray,
    cartesian_tangent: np.ndarray,
    masses_amu: np.ndarray,
) -> np.ndarray:
    """Return analytic vibrational G' from B and its Cartesian derivative.

    ``b_prime_cartesian[i, a, c]`` is the Cartesian derivative of
    ``B[i, a]`` with respect to Cartesian component ``c``. Cartesian
    coordinates are in bohr, so B and B' must use inverse-bohr and
    inverse-bohr-squared units respectively. The chart tangent
    converts that derivative to frozen ESALC-I coordinates. Only the
    vibrational block is needed for G'/G'' in the perturbative Hamiltonian;
    the complete rovibrational metric is still used for the geometric term.
    """
    b = np.asarray(b_matrix, dtype=float)
    prime = np.asarray(b_prime_cartesian, dtype=float)
    tangent = np.asarray(cartesian_tangent, dtype=float)
    masses = np.asarray(masses_amu, dtype=float).reshape(-1)
    if (b.ndim != 2 or b.shape[1] != 3 * len(masses)
            or prime.shape != (b.shape[0], b.shape[1], b.shape[1])
            or tangent.shape != (b.shape[1], b.shape[0])
            or np.any(masses <= 0)
            or not all(np.all(np.isfinite(value)) for value in (b, prime, tangent, masses))):
        raise ValueError("analytic Wilson G prime requires matching finite B, B prime, tangent and masses")
    prime_s = np.einsum("iac,ck->iak", prime, tangent, optimize=True)
    weighted_b = b * np.repeat(1.0 / masses, 3)[None, :]
    derivative = np.einsum("iak,ja->ijk", prime_s, weighted_b, optimize=True)
    return derivative + derivative.swapaxes(0, 1)


def sonic_rovibrational_metric(
    geometry_at_s: Callable[[np.ndarray], np.ndarray],
    displacement_s: np.ndarray,
    masses_amu: np.ndarray,
    *, rotational_dof: int = 3,
    derivative_step_s: float = 1e-4,
    tangent_at_s: Callable[[np.ndarray], np.ndarray] | None = None,
    constrained_vibrational_dof: int = 0,
) -> np.ndarray:
    """Build the complete contravariant metric from SONIC and rotation t vectors.

    The body-fixed geometry is in bohr. Center-of-mass motion is projected out
    before the covariant vibrational/rotational block is inverted.
    ``constrained_vibrational_dof=1`` permits a complete transverse TS chart
    with its reaction coordinate fixed. Constraint tangents must already be
    excluded by the geometry callback; the induced covariant block is inverted
    afterwards. This does not remove an imaginary mode by taking its absolute value.
    """
    s = np.asarray(displacement_s, dtype=float).reshape(-1)
    masses = np.asarray(masses_amu, dtype=float).reshape(-1)
    h = float(derivative_step_s)
    if s.size < 1 or masses.size < 2 or np.any(masses <= 0) or not np.isfinite(h) or h <= 0:
        raise ValueError("invalid SONIC geometry metric inputs")
    if (isinstance(constrained_vibrational_dof,(bool,np.bool_))
            or not isinstance(constrained_vibrational_dof,(int,np.integer))
            or constrained_vibrational_dof<0):
        raise ValueError('nonnegative integer constrained vibrational rank required')
    if rotational_dof not in {2, 3} or s.size+constrained_vibrational_dof != 3 * masses.size - 3 - rotational_dof:
        raise ValueError("SONIC geometry metric requires the complete nonredundant vibrational rank")

    def centered(at_s: np.ndarray) -> np.ndarray:
        geometry = np.asarray(geometry_at_s(at_s), dtype=float)
        if geometry.shape != (masses.size, 3) or not np.all(np.isfinite(geometry)):
            raise ValueError("SONIC geometry callback must return finite body-fixed bohr coordinates")
        return geometry - np.average(geometry, axis=0, weights=masses)

    xyz = centered(s)
    if tangent_at_s is None:
        tangents = np.empty((s.size, masses.size, 3))
        for index in range(s.size):
            delta = np.zeros_like(s)
            delta[index] = h
            tangents[index] = (centered(s + delta) - centered(s - delta)) / (2 * h)
    else:
        tangent = np.asarray(tangent_at_s(s), dtype=float)
        if tangent.shape != (3 * masses.size, s.size) or not np.all(np.isfinite(tangent)):
            raise ValueError("analytic SONIC tangent has invalid shape")
        tangents = tangent.T.reshape(s.size, masses.size, 3)
    if rotational_dof == 3:
        axes = np.eye(3)
    elif rotational_dof == 2:
        _, _, vectors = np.linalg.svd(xyz * np.sqrt(masses[:, None]), full_matrices=False)
        axis = vectors[0]
        first = np.eye(3)[int(np.argmin(np.abs(axis)))]
        first -= np.dot(first, axis) * axis
        first /= np.linalg.norm(first)
        axes = np.stack((first, np.cross(axis, first)))
    else:
        raise ValueError("SONIC metric supports two or three rotational coordinates")
    rotations = np.stack([np.cross(axis, xyz) for axis in axes])
    vectors = np.concatenate((tangents, rotations), axis=0)
    covariant = np.einsum("aic,i,bic->ab", vectors, masses, vectors)
    if np.min(np.linalg.eigvalsh(covariant)) <= 0:
        raise ValueError("SONIC rovibrational t vectors are rank deficient")
    return np.linalg.inv(covariant)


def rectilinear_coriolis_metric_second(
    geometry_bohr: np.ndarray,
    masses_amu: np.ndarray,
    mass_normal_modes: np.ndarray,
) -> np.ndarray:
    """Exact Watson Coriolis G'' in a mass-orthonormal Cartesian mode gauge.

    ``mass_normal_modes`` has shape (3 * atoms, modes) and contains Cartesian
    displacements per unit mass-weighted normal coordinate. Its signs must
    match the signs used for the force and property derivatives. No displaced
    electronic calculation or finite difference of the metric is needed.
    """
    xyz = np.asarray(geometry_bohr, dtype=float)
    masses = np.asarray(masses_amu, dtype=float).reshape(-1)
    modes = np.asarray(mass_normal_modes, dtype=float)
    if (xyz.shape != (masses.size, 3) or modes.ndim != 2
            or modes.shape[0] != 3 * masses.size or masses.size < 3
            or np.any(masses <= 0) or not all(np.all(np.isfinite(a)) for a in (xyz, masses, modes))):
        raise ValueError("invalid Cartesian normal modes for Coriolis metric")
    if not np.allclose(modes.T @ (np.repeat(masses, 3)[:, None] * modes),
                       np.eye(modes.shape[1]), rtol=0, atol=1e-6):
        raise ValueError("Coriolis metric requires mass-orthonormal modes")
    xyz = xyz - np.average(xyz, axis=0, weights=masses)
    inertia = np.eye(3) * np.einsum("a,ac,ac->", masses, xyz, xyz)
    inertia -= np.einsum("a,ac,ad->cd", masses, xyz, xyz)
    if np.linalg.eigvalsh(inertia)[0] <= 0:
        raise ValueError("Coriolis metric requires a nonlinear geometry")
    tangents = modes.reshape(masses.size, 3, -1)
    coupling = np.empty((modes.shape[1], 3, modes.shape[1]))
    for axis in range(3):
        rotated = np.moveaxis(
            np.cross(np.eye(3)[axis], np.moveaxis(tangents, 1, -1)), -1, 1,
        )
        coupling[:, axis, :] = np.einsum(
            "aci,acj,a->ij", tangents, rotated, masses, optimize=True,
        )
    # G_ij(Q) = delta_ij + C_ia,k I^-1_ab C_jb,l Q_k Q_l + O(Q^3).
    first = np.einsum(
        "iak,ab,jbl->ijkl", coupling, np.linalg.inv(inertia), coupling,
        optimize=True,
    )
    return first + first.swapaxes(2, 3)


def acquire_sonic_vpt2_from_cartesian_evaluator(
    *,
    geometry_at_s: Callable[[np.ndarray], np.ndarray],
    masses_amu: np.ndarray,
    evaluator: Callable[[np.ndarray], object],
    metric_first_at_s: Callable[[np.ndarray], np.ndarray] | None = None,
    b_matrix_at_s: Callable[[np.ndarray], np.ndarray] | None = None,
    b_prime_cartesian_at_s: Callable[[np.ndarray], np.ndarray] | None = None,
    analytic_eckart_chart: bool = False,
    eckart_full_metric_prime: bool = False,
    coordinate_count: int,
    step_dimensionless: float = 0.02,
    step_mass_weighted_angstrom: float | None = None,
    cubic_estimator_policy: str = "hessian_gradient_mean",
    force_constant_cutoff_au: float = 0.0,
    final_force_constant_cutoff_au: float | None = None,
    geometry_derivative_step_s: float = 1e-4,
    rotational_dof: int = 3,
    denominator_tolerance_cm: float = 1e-6,
    complete_quartic: bool = False,
    fermi_only: bool = False,
    single_mode_only: bool = False,
    mode_contribution_cache: object | None = None,
    return_field: bool = False,
) -> SonicVPT2Result | SonicAcquiredField:
    """Run the SONIC VPT2 acquisition from Cartesian E/G/H evaluations.

    ``geometry_at_s`` realizes frozen SONIC displacements as smoothly embedded
    Cartesian bohr coordinates. The evaluator receives Cartesian angstrom and
    returns ``gradient_hartree_per_bohr`` and ``hessian_hartree_per_bohr2``.
    If it also returns ``energy_hartree``, energies are checked against the
    gradient-based diagonal quartic estimates without changing the force field.
    Each displaced Hessian is transformed with SMITH's full curvature term.
    """
    from matrix_chem import BOHR_TO_ANGSTROM
    from matrix_smith import curvilinear_hessian

    n = int(coordinate_count)
    h = float(geometry_derivative_step_s)
    if n < 1 or not np.isfinite(h) or h <= 0:
        raise ValueError("SONIC Cartesian adapter requires a positive geometry step")
    if analytic_eckart_chart and (b_matrix_at_s is None or b_prime_cartesian_at_s is None
                                  or masses_amu is None):
        raise ValueError("analytic Eckart chart requires B, B prime and masses")

    def geometry(displacement: np.ndarray) -> np.ndarray:
        value = np.asarray(geometry_at_s(displacement), dtype=float)
        if value.ndim != 2 or value.shape[1] != 3 or not np.all(np.isfinite(value)):
            raise ValueError("SONIC realization must return finite Cartesian bohr coordinates")
        return value

    reference = geometry(np.zeros(n)) if analytic_eckart_chart else None
    if reference is not None:
        masses = np.asarray(masses_amu, dtype=float).reshape(-1)
        centered = reference - np.average(reference, axis=0, weights=masses)
        constraints = np.zeros((6, reference.size))
        for atom, mass in enumerate(masses):
            constraints[:3, 3 * atom:3 * atom + 3] = mass * np.eye(3)
            for axis in range(3):
                constraints[3 + axis, 3 * atom:3 * atom + 3] = (
                    mass * np.cross(np.eye(3)[axis], centered[atom])
                )

    def chart_derivatives(displacement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        b = np.asarray(b_matrix_at_s(displacement), dtype=float)
        b_prime = np.asarray(b_prime_cartesian_at_s(displacement), dtype=float)
        if b.shape != (n, reference.size) or b_prime.shape != (n, reference.size, reference.size):
            raise ValueError("analytic Eckart B/B prime has invalid shape")
        tangent = np.linalg.solve(np.vstack((b, constraints)),
                                  np.vstack((np.eye(n), np.zeros((6, n)))))
        contracted = np.einsum("iab,aj,bk->ijk", b_prime, tangent, tangent, optimize=True)
        curvature = -np.einsum("ai,ijk->ajk", tangent, contracted, optimize=True)
        return tangent, curvature

    point_derivatives: dict[bytes, tuple[np.ndarray, float | None]] = {}

    def internal_hessian(displacement: np.ndarray) -> np.ndarray:
        s = np.asarray(displacement, dtype=float)
        xyz = geometry(s)
        result = evaluator(xyz * BOHR_TO_ANGSTROM)
        cartesian = np.asarray(result.hessian_hartree_per_bohr2, dtype=float)
        gradient = np.asarray(result.gradient_hartree_per_bohr, dtype=float).reshape(-1)
        size = xyz.size
        if cartesian.shape != (size, size) or gradient.shape != (size,):
            raise ValueError("Cartesian evaluator must return a matching gradient and Hessian")
        if not np.all(np.isfinite(cartesian)):
            raise ValueError("Cartesian evaluator returned a nonfinite Hessian")
        cartesian = 0.5 * (cartesian + cartesian.T)
        if analytic_eckart_chart:
            transform, curvature = chart_derivatives(s)
        else:
            transform = np.empty((size, n))
            curvature = np.empty((size, n, n))
            for i in range(n):
                delta = np.zeros(n)
                delta[i] = h
                plus = geometry(s + delta).reshape(-1)
                minus = geometry(s - delta).reshape(-1)
                transform[:, i] = (plus - minus) / (2 * h)
                curvature[:, i, i] = (plus - 2 * xyz.reshape(-1) + minus) / h**2
                for j in range(i):
                    second = np.zeros(n)
                    second[j] = h
                    mixed = (
                        geometry(s + delta + second).reshape(-1)
                        - geometry(s + delta - second).reshape(-1)
                        - geometry(s - delta + second).reshape(-1)
                        + geometry(s - delta - second).reshape(-1)
                    ) / (4 * h**2)
                    curvature[:, i, j] = curvature[:, j, i] = mixed
        energy = getattr(result, "energy_hartree", None)
        if energy is None:
            energy = getattr(result, "total_energy_hartree", None)
        point_derivatives[s.tobytes()] = (
            transform.T @ gradient,
            None if energy is None else float(energy),
        )
        return curvilinear_hessian(
            cartesian, transform, gradient, second_back_transform=curvature,
        )

    if eckart_full_metric_prime:
        if not analytic_eckart_chart or rotational_dof != 3:
            raise ValueError("full Eckart G prime requires a nonlinear analytic chart")
        if metric_first_at_s is not None:
            raise ValueError("analytic Eckart B/B prime already determine G prime")
        from matrix_smith import eckart_rovibrational_metric_first_from_b_prime

        def eckart_metric_first(displacement: np.ndarray) -> np.ndarray:
            return eckart_rovibrational_metric_first_from_b_prime(
                b_matrix_at_s(displacement),
                b_prime_cartesian_at_s(displacement),
                geometry(displacement), masses_amu, constraints,
            )

        metric_provider = eckart_metric_first
        b_provider = None
        b_prime_provider = None
    else:
        metric_provider = metric_first_at_s
        b_provider = b_matrix_at_s
        b_prime_provider = b_prime_cartesian_at_s

    return acquire_sonic_vpt2_from_hessians(
        hessian_at_s=internal_hessian,
        gradient_at_s=lambda s: point_derivatives[np.asarray(s).tobytes()][0],
        energy_at_s=lambda s: point_derivatives[np.asarray(s).tobytes()][1],
        metric_first_at_s=metric_provider,
        b_matrix_at_s=b_provider,
        b_prime_cartesian_at_s=b_prime_provider,
        tangent_at_s=(lambda s: chart_derivatives(s)[0]) if analytic_eckart_chart else None,
        geometry_at_s=geometry,
        masses_amu=masses_amu,
        coordinate_count=n,
        step_dimensionless=step_dimensionless,
        step_mass_weighted_angstrom=step_mass_weighted_angstrom,
        cubic_estimator_policy=cubic_estimator_policy,
        force_constant_cutoff_au=force_constant_cutoff_au,
        final_force_constant_cutoff_au=final_force_constant_cutoff_au,
        geometry_derivative_step_s=h,
        rotational_dof=rotational_dof,
        denominator_tolerance_cm=denominator_tolerance_cm,
        complete_quartic=complete_quartic,
        fermi_only=fermi_only,
        single_mode_only=single_mode_only,
        mode_contribution_cache=mode_contribution_cache,
        return_field=return_field,
    )


def acquire_sonic_gvpt2_from_cartesian_evaluator(**kwargs: object) -> SonicGVPT2Result:
    """Run the complete SONIC GVPT2 Hessian stencil through a Cartesian E/G/H provider."""
    options = dict(kwargs)
    solver_options = {name: options.pop(name) for name in (
        "max_quanta", "max_states", "resonance_policy", "fermi_window_cm",
        "dd_window_cm", "martin_threshold_cm", "dd_coupling_threshold_cm",
        "numerical_zero_cm", "resonance_criteria", "targeted_polyad_options",
    ) if name in options}
    if "denominator_tolerance_cm" in options:
        solver_options["denominator_tolerance_cm"] = options["denominator_tolerance_cm"]
    acquired = acquire_sonic_vpt2_from_cartesian_evaluator(
        **options, complete_quartic=True, return_field=True,
    )
    if not isinstance(acquired, SonicAcquiredField):
        raise RuntimeError("Cartesian SONIC acquisition returned an incomplete field")
    return replace(solve_sonic_gvpt2(acquired.field, **solver_options),
                   acquisition=acquired.acquisition)


__all__ = [
    "SonicVPT2Input", "SonicVPT2Result", "SonicGVPT2Result", "SonicAcquiredField",
    "solve_sonic_vpt2", "sonic_vpt2_x_matrix", "solve_sonic_gvpt2",
    "solve_sonic_gvpt2_adaptive",
    "acquire_sonic_vpt2_from_hessians", "acquire_sonic_gvpt2_from_hessians",
    "sonic_rovibrational_metric",
    "wilson_metric_first_from_b_prime",
    "acquire_sonic_vpt2_from_cartesian_evaluator",
    "acquire_sonic_gvpt2_from_cartesian_evaluator",
]
