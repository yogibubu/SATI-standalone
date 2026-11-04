"""Finite normal-ordered KQ/KCC corrections to a local-mode VPT2 x matrix."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import comb, factorial, sqrt

import numpy as np

from .curvilinear_vpt2 import _algebraic_x_matrix

Monomial = tuple[tuple[int, ...], tuple[int, ...]]
Polynomial = dict[Monomial, complex]


@dataclass(frozen=True)
class LocalAlgebraicXResult:
    reference_x_cm: np.ndarray
    kq_x_cm: np.ndarray
    kcc_x_cm: np.ndarray
    total_x_cm: np.ndarray


def _shift(monomial: Monomial) -> tuple[int, ...]:
    return tuple(a - b for a, b in zip(*monomial))


def _factor(
    annihilators: tuple[int, ...], creators: tuple[int, ...],
    contraction: tuple[int, ...],
) -> int:
    value = 1
    for b, c, h in zip(annihilators, creators, contraction):
        if h < 0 or h > min(b, c):
            return 0
        value *= comb(b, h) * comb(c, h) * factorial(h)
    return value


def _multiply(left: Polynomial, right: Polynomial) -> Polynomial:
    output: Polynomial = {}
    for (alpha, beta), first in left.items():
        for (gamma, delta), second in right.items():
            for contraction in product(*(range(min(b, c) + 1)
                                         for b, c in zip(beta, gamma))):
                factor = _factor(beta, gamma, contraction)
                key = (
                    tuple(a + c - h for a, c, h in zip(alpha, gamma, contraction)),
                    tuple(b + d - h for b, d, h in zip(beta, delta, contraction)),
                )
                output[key] = output.get(key, 0j) + first * second * factor
    return output


def _operator(n: int, kind: str, mode: int) -> Polynomial:
    unit = tuple(int(i == mode) for i in range(n))
    zero = (0,) * n
    if kind == "q":
        return {(unit, zero): 1 / sqrt(2), (zero, unit): 1 / sqrt(2)}
    return {(unit, zero): 1j / sqrt(2), (zero, unit): -1j / sqrt(2)}


def _ordered_product(n: int, operators: tuple[tuple[str, int], ...]) -> Polynomial:
    zero = (0,) * n
    value: Polynomial = {(zero, zero): 1.0}
    for kind, mode in operators:
        value = _multiply(value, _operator(n, kind, mode))
    return value


def _add_operator(
    destination: Polynomial, n: int, coefficient: float,
    operators: tuple[tuple[str, int], ...],
) -> None:
    if coefficient == 0:
        return
    for key, value in _ordered_product(n, operators).items():
        destination[key] = destination.get(key, 0j) + coefficient * value


def _polynomials(
    kq: np.ndarray, kp: np.ndarray, f3: np.ndarray, f4: np.ndarray,
    g1: np.ndarray, g2: np.ndarray,
) -> tuple[Polynomial, Polynomial, Polynomial]:
    n = len(kq)
    k: Polynomial = {}
    c: Polynomial = {}
    q: Polynomial = {}
    for i in range(n):
        for j in range(i + 1, n):
            _add_operator(k, n, float(kq[i, j]), (("q", i), ("q", j)))
            _add_operator(k, n, float(kp[i, j]), (("p", i), ("p", j)))
    for i, j, l in np.argwhere(f3 != 0):
        _add_operator(c, n, float(f3[i, j, l]) / 6,
                      (("q", int(i)), ("q", int(j)), ("q", int(l))))
    for i, j, l in np.argwhere(g1 != 0):
        _add_operator(c, n, float(g1[i, j, l]) / 2,
                      (("p", int(i)), ("q", int(l)), ("p", int(j))))
    for i, j, l, m in np.argwhere(f4 != 0):
        _add_operator(q, n, float(f4[i, j, l, m]) / 24,
                      (("q", int(i)), ("q", int(j)),
                       ("q", int(l)), ("q", int(m))))
    for i, j, l, m in np.argwhere(g2 != 0):
        _add_operator(q, n, float(g2[i, j, l, m]) / 4,
                      (("p", int(i)), ("q", int(l)),
                       ("q", int(m)), ("p", int(j))))
    return k, c, q


def _by_shift(polynomial: Polynomial) -> dict[tuple[int, ...], list[tuple[Monomial, complex]]]:
    groups: dict[tuple[int, ...], list[tuple[Monomial, complex]]] = {}
    for key, coefficient in polynomial.items():
        if coefficient:
            groups.setdefault(_shift(key), []).append((key, coefficient))
    return groups


def _pair_coefficient(left: Monomial, right: Monomial, m: tuple[int, ...]) -> int:
    alpha, beta = left
    gamma, delta = right
    if tuple(a + c for a, c in zip(alpha, gamma)) != tuple(
        b + d for b, d in zip(beta, delta)
    ):
        return 0
    contraction = tuple(a + c - x for a, c, x in zip(alpha, gamma, m))
    return _factor(beta, gamma, contraction)


def _triple_coefficient(
    left: Monomial, middle: Monomial, right: Monomial, m: tuple[int, ...],
) -> int:
    alpha, beta = left
    gamma, delta = middle
    epsilon, zeta = right
    if tuple(a + c + e for a, c, e in zip(alpha, gamma, epsilon)) != tuple(
        b + d + z for b, d, z in zip(beta, delta, zeta)
    ):
        return 0
    total = tuple(a + c + e - x for a, c, e, x in zip(alpha, gamma, epsilon, m))
    value = 0
    for contraction in product(*(range(min(b, c) + 1)
                                 for b, c in zip(beta, gamma))):
        first = _factor(beta, gamma, contraction)
        second = _factor(
            tuple(b + d - h for b, d, h in zip(beta, delta, contraction)),
            epsilon, tuple(h - t for h, t in zip(total, contraction)),
        )
        value += first * second
    return value


def _k_correction(
    frequencies: np.ndarray, k: Polynomial, c: Polynomial, q: Polynomial,
    m: tuple[int, ...], tolerance: float,
) -> tuple[float, float]:
    k_groups, c_groups, q_groups = (_by_shift(value) for value in (k, c, q))
    kq = 0j
    for shift, left in k_groups.items():
        opposite = tuple(-value for value in shift)
        for ka, kval in left:
            for qa, qval in q_groups.get(opposite, ()):
                factor = _pair_coefficient(ka, qa, m) - _pair_coefficient(qa, ka, m)
                if factor == 0:
                    continue
                gap = float(np.dot(frequencies, shift))
                if abs(gap) <= tolerance:
                    raise ValueError("local VPT2 KQ has a resonant denominator; use a polyad")
                kq += kval * qval * factor / gap

    kcc = 0j
    for left, middle, right in ((k_groups, c_groups, c_groups),
                                (c_groups, k_groups, c_groups),
                                (c_groups, c_groups, k_groups)):
        for left_shift, left_terms in left.items():
            for middle_shift, middle_terms in middle.items():
                right_shift = tuple(-a - b for a, b in zip(left_shift, middle_shift))
                for la, lval in left_terms:
                    for ma, mval in middle_terms:
                        for ra, rval in right.get(right_shift, ()):
                            factor = _triple_coefficient(la, ma, ra, m)
                            if factor == 0:
                                continue
                            first = float(np.dot(frequencies, right_shift))
                            second = float(np.dot(frequencies, tuple(
                                a + b for a, b in zip(middle_shift, right_shift)
                            )))
                            if abs(first) <= tolerance or abs(second) <= tolerance:
                                raise ValueError("local VPT2 KCC has a resonant denominator; use a polyad")
                            kcc += lval * mval * rval * factor / (first * second)
    if abs(kq.imag) > 1e-8 or abs(kcc.imag) > 1e-8:
        raise ValueError("algebraic local VPT2 x has a nonreal coefficient")
    return float(kq.real), float(kcc.real)


def solve_local_algebraic_x(
    frequencies_cm: np.ndarray, kq_cm: np.ndarray, kp_cm: np.ndarray,
    f3_cm: np.ndarray, f4_cm: np.ndarray, g1_cm: np.ndarray, g2_cm: np.ndarray,
    *, denominator_tolerance_cm: float = 1e-6,
) -> LocalAlgebraicXResult:
    """Algebraic x = curvilinear chi + KQ + KCC in local dimensionless modes.

    All tensors are in cm^-1 and use p_i g_ij,k q_k p_j ordering. The
    quadratic matrices contain only off-diagonal local couplings.
    Resonant terms must be handled by a variational polyad.
    """
    frequencies = np.asarray(frequencies_cm, dtype=float)
    n = len(frequencies)
    kq, kp, f3, f4, g1, g2 = (np.asarray(value, dtype=float) for value in (
        kq_cm, kp_cm, f3_cm, f4_cm, g1_cm, g2_cm,
    ))
    if (frequencies.ndim != 1 or n == 0 or np.any(frequencies <= 0)
            or not np.all(np.isfinite(frequencies))):
        raise ValueError("local frequencies must be positive and finite")
    if tuple(value.shape for value in (kq, kp, f3, f4, g1, g2)) != (
        (n, n), (n, n), (n, n, n), (n, n, n, n),
        (n, n, n), (n, n, n, n),
    ) or not all(np.all(np.isfinite(value)) for value in (kq, kp, f3, f4, g1, g2)):
        raise ValueError("local VPT2 tensors have invalid shapes or nonfinite values")
    if (not np.allclose(kq, kq.T) or not np.allclose(kp, kp.T)
            or np.any(np.diag(kq) != 0) or np.any(np.diag(kp) != 0)):
        raise ValueError("local quadratic couplings must be symmetric and off-diagonal")
    for name, tensor, pairs in (
        ("F3", f3, ((0, 1), (1, 2))),
        ("F4", f4, ((0, 1), (1, 2), (2, 3))),
        ("G'", g1, ((0, 1),)),
        ("G''", g2, ((0, 1), (2, 3))),
    ):
        if any(not np.allclose(tensor, np.swapaxes(tensor, a, b), rtol=1e-8, atol=1e-10)
               for a, b in pairs):
            raise ValueError(f"local VPT2 {name} is not symmetric")
    if not np.isfinite(denominator_tolerance_cm) or denominator_tolerance_cm <= 0:
        raise ValueError("local VPT2 denominator tolerance must be positive and finite")
    reference = _algebraic_x_matrix(
        frequencies, f3, f4, g1, g2,
        denominator_tolerance_cm=denominator_tolerance_cm,
    )
    k, c, q = _polynomials(kq, kp, f3, f4, g1, g2)
    delta_kq = np.zeros((n, n))
    delta_kcc = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            m = tuple(int(mode == i) + int(mode == j) for mode in range(n))
            kq_value, kcc_value = _k_correction(
                frequencies, k, c, q, m, denominator_tolerance_cm,
            )
            delta_kq[i, j] = delta_kq[j, i] = kq_value
            delta_kcc[i, j] = delta_kcc[j, i] = kcc_value
    return LocalAlgebraicXResult(
        reference_x_cm=reference,
        kq_x_cm=delta_kq,
        kcc_x_cm=delta_kcc,
        total_x_cm=reference + delta_kq + delta_kcc,
    )


__all__ = ["LocalAlgebraicXResult", "solve_local_algebraic_x"]
