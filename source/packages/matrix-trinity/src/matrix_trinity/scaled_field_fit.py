"""Reusable fixed-support GVPT2 evaluation for a coherently scaled PES.

Independent parameters are log(s_i), K_ii=s_i**2. Native potential tensors
scale by one s per index. The metric remains the geometrical metric. Oscillator
normalization is updated, including every cubic, quartic and kinetic term.
No molecular identity, mode labels or experimental values enter the evaluator.
"""

from itertools import product
import hashlib
from math import sqrt
import numpy as np
from scipy.sparse import csr_matrix
from numba import njit, prange
from .curvilinear_vpt2 import (
    _canonical_cubic_terms,
    _local_operator_transitions,
    _local_oscillator_element,
    _SelectedOperatorEvaluator,
    _resonance_kind,
    _gdv_overtone_dd_coupling,
    _gdv_combination_overtone_dd_coupling,
    _gdv_fundamental_dd_coupling,
    _modalize_sonic_field,
)


def _exponent(key, n):
    # Twice the exponent of s in dimensionless oscillator normalization.
    kinetic = any("p" in ops for _, ops in key)
    result = np.zeros(n, np.int8)
    for mode, ops in key:
        result[mode] += sum(1 if op == "p" or not kinetic else -1 for op in ops)
    return result


def _pack_cubic(terms, n, maximum_occupation):
    groups = {}
    deltas = []
    entry_groups = []
    modes = []
    tables = []
    entry_terms = []
    exponents = []
    for term, (key, coefficient) in enumerate(terms):
        exponents.append(_exponent(key, n))
        choices = []
        for mode, ops in key:
            local = {}
            for occupation in range(maximum_occupation + 1):
                for target, value in _local_operator_transitions(ops, occupation):
                    local.setdefault(
                        target - occupation, np.zeros(maximum_occupation + 1, complex)
                    )[occupation] = value
            choices.append(tuple(local.items()))
        for selection in product(*choices):
            delta = np.zeros(n, np.int8)
            em = np.full(3, -1, np.int16)
            tab = np.ones((3, maximum_occupation + 1), complex)
            for j, ((mode, _), (change, values)) in enumerate(zip(key, selection)):
                delta[mode] = change
                em[j] = mode
                tab[j] = values
            if not np.any(delta):
                continue
            signature = delta.tobytes()
            if signature not in groups:
                groups[signature] = len(deltas)
                deltas.append(delta)
            tab[0] *= coefficient
            entry_groups.append(groups[signature])
            modes.append(em)
            tables.append(tab)
            entry_terms.append(term)
    return (
        np.asarray(deltas, np.int8),
        groups,
        np.asarray(entry_groups, np.int32),
        np.asarray(modes, np.int16),
        np.asarray(tables, complex),
        np.asarray(entry_terms, np.int32),
        np.asarray(exponents, float),
    )


@njit(cache=True, parallel=True, nogil=True)
def _evaluate_cubic(states, ngroups, entry_groups, modes, tables, entry_terms, scales):
    answer = np.zeros((len(states), ngroups))
    for a in prange(len(states)):
        for e in range(len(entry_groups)):
            value = complex(scales[entry_terms[e]])
            for j in range(3):
                m = modes[e, j]
                if m >= 0:
                    value *= tables[e, j, states[a, m]]
            answer[a, entry_groups[e]] += value.real
    return answer


@njit(cache=True, parallel=True, nogil=True)
def _second_order(
    cubic, gaps, resonant, zero, denom_tol, pair_a, pair_b, starts, left, right, active_pairs
):
    diagonals = np.zeros(len(cubic))
    off = np.zeros(len(pair_a))
    for a in prange(len(cubic)):
        value = 0.0
        for p in range(len(gaps)):
            va = cubic[a, p]
            if resonant[p] and abs(va) >= zero:
                continue
            if abs(gaps[p]) <= denom_tol:
                if abs(va) > zero:
                    value = np.nan
                continue
            value += va * va / gaps[p]
        diagonals[a] = value
    for pair in prange(len(pair_a)):
        if not active_pairs[pair]:
            continue
        a, b = pair_a[pair], pair_b[pair]
        value = 0.0
        for k in range(starts[pair], starts[pair + 1]):
            p, q = left[k], right[k]
            va, vb = cubic[a, p], cubic[b, q]
            if (resonant[p] and abs(va) >= zero) or (resonant[q] and abs(vb) >= zero):
                continue
            if abs(gaps[p]) <= denom_tol or abs(gaps[q]) <= denom_tol:
                if abs(va * vb) > zero * zero:
                    value = np.nan
                continue
            value += 0.5 * va * vb * (1 / gaps[p] + 1 / gaps[q])
        off[pair] = value
    return diagonals, off


class ScaledGVPT2Evaluator:
    """Precontract fixed harmonic products, reevaluate the full GVPT2 map.

    Supports the native GDV resonance policy, including its closed-form DD
    overrides. Screening is recomputed at each point; policy switches are
    returned in diagnostics. A fixed support is not a convergence certificate:
    verify the fitted field with the production adaptive solver afterwards.
    """

    def __init__(
        self,
        field,
        states,
        *,
        fermi_window_cm=200.0,
        dd_window_cm=100.0,
        martin_threshold_cm=1.0,
        numerical_zero_cm=0.01,
        denominator_tolerance_cm=1e-6,
    ):
        self.field = field
        self.states = tuple(tuple(map(int, s)) for s in states)
        self.occupations = np.asarray(self.states, np.int16)
        self.freq, self.modes, self.f3, self.f4, self.g1, self.g2 = _modalize_sonic_field(field)
        n = len(self.freq)
        if (
            self.occupations.ndim != 2
            or self.occupations.shape[1] != n
            or np.any(self.occupations < 0)
        ):
            raise ValueError("nonnegative occupation products matching the field required")
        self.n = n
        self.modal_actions = []
        signatures = None
        if field.symmetry_actions is not None:
            inv = np.linalg.inv(self.modes)
            self.modal_actions = [inv @ action @ self.modes for action in field.symmetry_actions]
            if all(
                np.max(abs(action - np.diag(np.diag(action)))) < field.symmetry_tolerance
                and np.max(abs(abs(np.diag(action)) - 1)) < field.symmetry_tolerance
                for action in self.modal_actions
            ):
                characters = np.array([np.sign(np.diag(action)) for action in self.modal_actions])
                signatures = np.prod(characters[None, :, :] ** self.occupations[:, None, :], axis=2)
        self.fermi_window = fermi_window_cm
        self.dd_window = dd_window_cm
        self.martin = martin_threshold_cm
        self.zero = numerical_zero_cm
        self.denom_tol = denominator_tolerance_cm
        pack = _pack_cubic(
            _canonical_cubic_terms(self.f3, self.g1, 0.0), n, int(self.occupations.max())
        )
        (
            self.deltas,
            self.group_lookup,
            self.entry_groups,
            self.entry_modes,
            self.tables,
            self.entry_terms,
            self.cubic_exponents,
        ) = pack
        self.norms = np.sum(abs(self.deltas), axis=1)
        self.fermi_modes = np.full((len(self.deltas), 3), -1, int)
        for p, d in enumerate(self.deltas):
            pos = np.repeat(np.arange(n), np.maximum(d, 0))
            neg = np.repeat(np.arange(n), np.maximum(-d, 0))
            if {len(pos), len(neg)} == {1, 2}:
                single, double = (pos, neg) if len(pos) == 1 else (neg, pos)
                self.fermi_modes[p] = single[0], double[0], double[1]
        self.pairs = []
        paira = []
        pairb = []
        starts = [0]
        left = []
        right = []
        self.fermi_groups = []
        self.overrides = []
        self.quartic_entries = []
        for a, sa in enumerate(self.states):
            for b, sb in enumerate(self.states[:a]):
                if signatures is not None and not np.array_equal(signatures[a], signatures[b]):
                    continue
                kind = _resonance_kind(sa, sb)
                if kind is None:
                    continue
                pair = len(self.pairs)
                self.pairs.append((a, b, kind))
                if kind == "1-2":
                    self.fermi_groups.append(
                        (
                            pair,
                            self.group_lookup.get(
                                (self.occupations[b] - self.occupations[a])
                                .astype(np.int8)
                                .tobytes(),
                                -1,
                            ),
                        )
                    )
                    continue
                paira.append(a)
                pairb.append(b)
                change = self.occupations[a] - self.occupations[b]
                nz = np.flatnonzero(change)
                reachable = self.norms.copy()
                for m in nz:
                    reachable += abs(self.deltas[:, m] + change[m]) - abs(self.deltas[:, m])
                for p in np.flatnonzero(reachable <= 3):
                    q = self.group_lookup.get((self.deltas[p] + change).astype(np.int8).tobytes())
                    if q is not None:
                        left.append(p)
                        right.append(q)
                starts.append(len(left))
                self.quartic_entries.append((pair, a, b))
                funcs = (
                    [_gdv_overtone_dd_coupling, _gdv_combination_overtone_dd_coupling]
                    if kind == "2-2"
                    else [_gdv_fundamental_dd_coupling]
                    if kind == "1-1"
                    else []
                )
                for function in funcs:
                    if function(sa, sb, self.freq, self.f3, self.f4, self.g1, self.g2) is not None:
                        self.overrides.append((pair, a, b, function))
                        break
        self.pair_a = np.asarray(paira, int)
        self.pair_b = np.asarray(pairb, int)
        self.starts = np.asarray(starts, int)
        self.left = np.asarray(left, np.int32)
        self.right = np.asarray(right, np.int32)
        self.dd_pair_indices = np.array(
            [p for p, (_, _, kind) in enumerate(self.pairs) if kind != "1-2"], int
        )
        # Quartic matrix elements grouped by their exact scaling exponent.
        operator = _SelectedOperatorEvaluator(self.f3, self.f4, self.g1, self.g2)
        exp_lookup = {}
        exponents = []
        rows = []
        cols = []
        values = []
        exp_cache = {}
        qentries = [(-1, a, a) for a in range(len(self.states))] + self.quartic_entries
        self.qentries = qentries
        for row, (_, a, b) in enumerate(qentries):
            sa, sb = self.states[a], self.states[b]
            distance = sum(abs(x - y) for x, y in zip(sa, sb))
            required = tuple(i for i, (x, y) in enumerate(zip(sa, sb)) for _ in range(abs(x - y)))
            multisets = (
                [required]
                if distance == 4
                else [tuple(sorted((*required, m, m))) for m in range(n)]
                if distance == 2
                else [(i, i, j, j) for i in range(n) for j in range(i, n)]
            )
            sums = {}
            for indices in multisets:
                for key, coefficient in operator._terms(indices, 4):
                    value = complex(coefficient)
                    for mode, ops in key:
                        value *= _local_oscillator_element(ops, sa[mode], sb[mode])
                        if value == 0:
                            break
                    if not value:
                        continue
                    if abs(value.imag) > 1e-8:
                        raise ValueError("real vibrational operator required")
                    if key not in exp_cache:
                        exponent = tuple(_exponent(key, n))
                        if exponent not in exp_lookup:
                            exp_lookup[exponent] = len(exponents)
                            exponents.append(exponent)
                        exp_cache[key] = exp_lookup[exponent]
                    col = exp_cache[key]
                    sums[col] = sums.get(col, 0.0) + value.real
            for col, value in sums.items():
                if value:
                    rows.append(row)
                    cols.append(col)
                    values.append(value)
        self.quartic_exponents = np.asarray(exponents, float)
        self.quartic_design = csr_matrix(
            (values, (rows, cols)), shape=(len(qentries), len(exponents))
        )
        self.initial_resonant = None
        self.evaluations = 0

    def modal_tensors(self, theta):
        theta = np.asarray(theta, float)
        if theta.shape != (self.n,) or not np.all(np.isfinite(theta)):
            raise ValueError("finite log-frequency scales required")
        s = np.exp(theta)
        if any(
            np.max(abs(action * s[None, :] - s[:, None] * action)) > self.field.symmetry_tolerance
            for action in self.modal_actions
        ):
            raise ValueError("scaling must preserve each molecular symmetry multiplet")
        r = np.sqrt(s)
        f3 = self.f3 * r[:, None, None] * r[None, :, None] * r[None, None, :]
        f4 = (
            self.f4
            * r[:, None, None, None]
            * r[None, :, None, None]
            * r[None, None, :, None]
            * r[None, None, None, :]
        )
        g1 = self.g1 * r[:, None, None] * r[None, :, None] / r[None, None, :]
        g2 = (
            self.g2
            * r[:, None, None, None]
            * r[None, :, None, None]
            / r[None, None, :, None]
            / r[None, None, None, :]
        )
        return self.freq * s, f3, f4, g1, g2

    def native_field(self, theta):
        """Apply the same PES scaling in the fixed native ESALC-I chart."""
        from dataclasses import replace
        from matrix_qm import HESSIAN_EIGENVALUE_TO_CM

        theta = np.asarray(theta, float)
        s = np.exp(theta)
        m = self.modes
        inverse = np.linalg.inv(m)
        transform = (m * s) @ inverse
        f3 = np.einsum(
            "abc,ai,bj,ck->ijk",
            self.field.cubic_sonic,
            transform,
            transform,
            transform,
            optimize=True,
        )
        f4 = np.einsum(
            "abcd,ai,bj,ck,dl->ijkl",
            self.field.quartic_sonic,
            transform,
            transform,
            transform,
            transform,
            optimize=True,
        )
        f2 = inverse.T @ np.diag((self.freq * s / HESSIAN_EIGENVALUE_TO_CM) ** 2) @ inverse
        return replace(self.field, harmonic_sonic=f2, cubic_sonic=f3, quartic_sonic=f4)

    def matrix(self, theta):
        theta = np.asarray(theta, float)
        freq, f3, f4, g1, g2 = self.modal_tensors(theta)
        cubic = _evaluate_cubic(
            self.occupations,
            len(self.deltas),
            self.entry_groups,
            self.entry_modes,
            self.tables,
            self.entry_terms,
            np.exp(0.5 * self.cubic_exponents @ theta),
        )
        gaps = -self.deltas @ freq
        resonant = np.zeros(len(gaps), bool)
        for p in np.flatnonzero(self.fermi_modes[:, 0] >= 0):
            i, j, k = self.fermi_modes[p]
            u = abs(
                (f3[j, k, i] - g1[j, k, i] + g1[j, i, k] + g1[k, i, j]) / (4 if j == k else sqrt(8))
            )
            gap = abs(gaps[p])
            resonant[p] = gap <= self.fermi_window and (
                u**4 / max(gap, 1e-300) ** 3 >= self.martin if gap > 0 else u > 0
            )
            if gap <= self.fermi_window and u / max(gap, self.denom_tol) >= 0.1:
                resonant[p] = True
        if self.initial_resonant is None:
            self.initial_resonant = resonant.copy()
        d2, off2 = _second_order(
            cubic,
            gaps,
            resonant,
            self.zero,
            self.denom_tol,
            self.pair_a,
            self.pair_b,
            self.starts,
            self.left,
            self.right,
            np.abs((self.occupations[self.pair_a] - self.occupations[self.pair_b]) @ freq)
            <= self.dd_window,
        )
        q = self.quartic_design @ np.exp(0.5 * self.quartic_exponents @ theta)
        h = np.diag(
            self.occupations @ freq + d2 + q[: len(self.states)] + self.field.pseudopotential_cm
        )
        pair_values = np.zeros(len(self.pairs))
        pair_values[self.dd_pair_indices] = off2 + q[len(self.states) :]
        for pair, a, b, function in self.overrides:
            if abs(np.dot(freq, self.occupations[a] - self.occupations[b])) > self.dd_window:
                continue
            pair_values[pair] = function(self.states[a], self.states[b], freq, f3, f4, g1, g2)
        for pair, p in self.fermi_groups:
            if p >= 0:
                a, b, _ = self.pairs[pair]
                v = cubic[a, p]
                pair_values[pair] = v if resonant[p] and abs(v) >= self.zero else 0.0
        for pair, (a, b, kind) in enumerate(self.pairs):
            v = pair_values[pair]
            if kind != "1-2" and (
                abs(np.dot(freq, self.occupations[a] - self.occupations[b])) > self.dd_window
                or abs(v) < self.zero
            ):
                v = 0.0
            h[a, b] = h[b, a] = v
        if not np.all(np.isfinite(h)):
            raise ValueError("unresolved GVPT2 denominator in fitted field")
        self.evaluations += 1
        return h, dict(
            resonance_pattern_switches=int(np.count_nonzero(resonant != self.initial_resonant)),
            evaluations=self.evaluations,
            resonance_signature=hashlib.sha256(resonant.tobytes()).hexdigest(),
        )
