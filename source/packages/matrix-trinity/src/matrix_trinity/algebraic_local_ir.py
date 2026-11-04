"""Nonresonant local-mode IR moments with residual quadratic couplings.

The finite Fock contractions implement the displayed K/eta/rho/sigma and
dipole-derivative equations without a virtual-state basis cutoff. Near
resonances belong to a separate variational polyad and are rejected here.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import factorial

import numpy as np

from .algebraic_local_x import Polynomial, _add_operator, _polynomials
from .gvpt2_spectroscopy import IR_TRANSITION_FACTOR


@dataclass(frozen=True)
class LocalAlgebraicIRResult:
    states: tuple[tuple[int, ...], ...]
    moment_orders_au: np.ndarray  # order, state, Cartesian component
    intensity_orders_km_mol: np.ndarray  # order, state; fixed input transition energy
    dipole_moments_au: np.ndarray  # state, Cartesian component
    ir_intensities_km_mol: np.ndarray  # positive, squared total moment


def _apply_polynomial(
    state: tuple[int, ...], polynomial: Polynomial,
) -> dict[tuple[int, ...], complex]:
    output: dict[tuple[int, ...], complex] = {}
    for (alpha, beta), coefficient in polynomial.items():
        if not coefficient or any(n < b for n, b in zip(state, beta)):
            continue
        target = tuple(n - b + a for n, a, b in zip(state, alpha, beta))
        factor = 1.0
        for n, m, b in zip(state, target, beta):
            factor *= np.sqrt(factorial(n) * factorial(m)) / factorial(n - b)
        output[target] = output.get(target, 0j) + coefficient * factor
    return output


def _dipole_polynomials(
    first: np.ndarray, second: np.ndarray, third: np.ndarray,
) -> tuple[tuple[Polynomial, ...], ...]:
    n = first.shape[1]
    operators: list[tuple[Polynomial, ...]] = []
    for rank, tensor in enumerate((first, second, third), 1):
        components: list[Polynomial] = []
        for component in range(3):
            polynomial: Polynomial = {}
            for indices in product(range(n), repeat=rank):
                coefficient = float(tensor[(component, *indices)]) / factorial(rank)
                if coefficient:
                    _add_operator(polynomial, n, coefficient,
                                  tuple(("q", mode) for mode in indices))
            components.append(polynomial)
        operators.append(tuple(components))
    return tuple(operators)


def solve_local_algebraic_ir(
    frequencies_cm: np.ndarray, kq_cm: np.ndarray, kp_cm: np.ndarray,
    f3_cm: np.ndarray, f4_cm: np.ndarray, g1_cm: np.ndarray, g2_cm: np.ndarray,
    dipole_first_au: np.ndarray, dipole_second_au: np.ndarray,
    dipole_third_au: np.ndarray, states: tuple[tuple[int, ...], ...],
    *, transition_energies_cm: np.ndarray | None = None,
    denominator_tolerance_cm: float = 1e-6,
) -> LocalAlgebraicIRResult:
    """Return ground-to-state transition moments through second order.

    All Hamiltonian tensors are in dimensionless local modes and cm^-1,
    as for ``solve_local_algebraic_x``. Dipole derivatives are in atomic
    units per dimensionless coordinate. The output intensity orders are
    the coefficients at orders zero, one, and two using a fixed
    transition-energy prefactor (harmonic unless supplied explicitly).
    The states must be nonresonant.
    """
    frequencies = np.asarray(frequencies_cm, dtype=float)
    n = len(frequencies)
    if frequencies.ndim != 1 or n == 0 or np.any(frequencies <= 0) or not np.all(np.isfinite(frequencies)):
        raise ValueError("local IR frequencies must be positive and finite")
    tensors = tuple(np.asarray(value, dtype=float) for value in (
        kq_cm, kp_cm, f3_cm, f4_cm, g1_cm, g2_cm,
        dipole_first_au, dipole_second_au, dipole_third_au,
    ))
    expected = ((n, n), (n, n), (n, n, n), (n, n, n, n),
                (n, n, n), (n, n, n, n), (3, n), (3, n, n), (3, n, n, n))
    if any(value.shape != shape or not np.all(np.isfinite(value))
           for value, shape in zip(tensors, expected)):
        raise ValueError("local IR Hamiltonian or dipole derivatives have invalid shape")
    kq, kp, f3, f4, g1, g2, mu1, mu2, mu3 = tensors
    if (not np.allclose(kq, kq.T) or not np.allclose(kp, kp.T)
            or np.any(np.diag(kq) != 0) or np.any(np.diag(kp) != 0)):
        raise ValueError("local IR quadratic couplings must be symmetric and off-diagonal")
    for name, tensor, pairs in (
        ("F3", f3, ((0, 1), (1, 2))),
        ("F4", f4, ((0, 1), (1, 2), (2, 3))),
        ("G'", g1, ((0, 1),)),
        ("G''", g2, ((0, 1), (2, 3))),
        ("mu2", mu2, ((1, 2),)),
        ("mu3", mu3, ((1, 2), (2, 3))),
    ):
        if any(not np.allclose(tensor, np.swapaxes(tensor, a, b), rtol=1e-8, atol=1e-10)
               for a, b in pairs):
            raise ValueError(f"local IR {name} tensor is not symmetric")
    if denominator_tolerance_cm <= 0 or not np.isfinite(denominator_tolerance_cm):
        raise ValueError("local IR denominator tolerance must be positive")
    if not states or len(set(states)) != len(states) or any(
        len(state) != n or any(not isinstance(q, int) or q < 0 for q in state)
        for state in states
    ):
        raise ValueError("local IR states must be distinct nonnegative occupations")
    transition_energies = (
        np.array([np.dot(frequencies, state) for state in states], dtype=float)
        if transition_energies_cm is None
        else np.asarray(transition_energies_cm, dtype=float)
    )
    if (transition_energies.shape != (len(states),)
            or np.any(transition_energies <= 0)
            or not np.all(np.isfinite(transition_energies))):
        raise ValueError("local IR transition energies must be positive and finite")
    k, c, q = _polynomials(kq, kp, f3, f4, g1, g2)
    w = dict(c)
    for monomial, coefficient in k.items():
        w[monomial] = w.get(monomial, 0j) + coefficient
    dipole = _dipole_polynomials(mu1, mu2, mu3)
    zero = (0,) * n
    wave_first: dict[tuple[int, ...], dict[tuple[int, ...], complex]] = {}
    wave_second: dict[tuple[int, ...], dict[tuple[int, ...], complex]] = {}

    def energy(state: tuple[int, ...]) -> float:
        return float(np.dot(frequencies, state))

    def gap(source: tuple[int, ...], target: tuple[int, ...]) -> float:
        value = energy(source) - energy(target)
        if abs(value) <= denominator_tolerance_cm:
            raise ValueError("resonant local IR denominator; use a variational polyad")
        return value

    for source in (zero, *states):
        if source in wave_first:
            continue
        first = {
            target: value / gap(source, target)
            for target, value in _apply_polynomial(source, w).items()
            if target != source and abs(value) > 1e-14
        }
        numerator = _apply_polynomial(source, q)
        for intermediate, amplitude in first.items():
            for target, value in _apply_polynomial(intermediate, w).items():
                numerator[target] = numerator.get(target, 0j) + value * amplitude
        second = {
            target: value / gap(source, target)
            for target, value in numerator.items()
            if target != source and abs(value) > 1e-14
        }
        wave_first[source] = first
        wave_second[source] = second

    def element(
        left: dict[tuple[int, ...], complex], polynomial: Polynomial,
        right: dict[tuple[int, ...], complex],
    ) -> complex:
        return sum(
            np.conjugate(left_amplitude) * right_amplitude
            * _apply_polynomial(right_state, polynomial).get(left_state, 0j)
            for left_state, left_amplitude in left.items()
            for right_state, right_amplitude in right.items()
        )

    moments = np.zeros((3, len(states), 3), dtype=complex)
    ground = {zero: 1.0}
    ag, bg = wave_first[zero], wave_second[zero]
    ng = sum(abs(value) ** 2 for value in ag.values())
    for index, final in enumerate(states):
        excited = {final: 1.0}
        af, bf = wave_first[final], wave_second[final]
        nf = sum(abs(value) ** 2 for value in af.values())
        for component in range(3):
            d1, d2, d3 = (rank[component] for rank in dipole)
            m0 = element(excited, d1, ground)
            m1 = (element(excited, d2, ground)
                  + element(af, d1, ground) + element(excited, d1, ag))
            m2 = (element(excited, d3, ground)
                  + element(af, d2, ground) + element(excited, d2, ag)
                  + element(bf, d1, ground) + element(excited, d1, bg)
                  + element(af, d1, ag) - 0.5 * (nf + ng) * m0)
            moments[:, index, component] = (m0, m1, m2)
    if np.max(abs(moments.imag)) > 1e-8:
        raise ValueError("local IR transition moments have a nonreal component")
    real_moments = moments.real
    intensities = np.zeros((3, len(states)))
    for index in range(len(states)):
        prefactor = IR_TRANSITION_FACTOR * transition_energies[index]
        m0, m1, m2 = real_moments[:, index]
        intensities[:, index] = prefactor * np.array((
            np.dot(m0, m0), 2 * np.dot(m0, m1),
            np.dot(m1, m1) + 2 * np.dot(m0, m2),
        ))
    dipole_moments = np.sum(real_moments, axis=0)
    ir_intensities = IR_TRANSITION_FACTOR * transition_energies * np.sum(
        dipole_moments**2, axis=1,
    )
    return LocalAlgebraicIRResult(
        states, real_moments, intensities, dipole_moments, ir_intensities,
    )


__all__ = ["LocalAlgebraicIRResult", "solve_local_algebraic_ir"]
