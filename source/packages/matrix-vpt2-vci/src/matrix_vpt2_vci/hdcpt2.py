"""Resonance-free hybrid degeneracy-corrected vibrational PT2.

The DCPT2 replacement is applied to each cubic-state coupling before the
second-order sum.  The published HDCPT2 switching function recovers ordinary
VPT2 when the product of coupling and energy separation is large.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, tanh

import numpy as np

from .models import AnharmonicInput
from .vci import (
    QuarticForceField,
    VCIOptions,
    _selected_force_field,
    _x_matrix_power,
    force_field_from_anharmonic_input,
)
from .vpt2 import (
    VPT2State,
    _cubic_couplings_from_state,
    _harmonic_energy,
    _term_matrix_element,
)


@dataclass(frozen=True)
class HDCPT2Result:
    harmonic_frequencies_cm: np.ndarray
    fundamentals_cm: np.ndarray
    x_matrix_cm: np.ndarray
    states: tuple[VPT2State, ...]
    alpha: float
    beta_cm2: float
    overtone_residuals_cm: np.ndarray


def _hybrid_cubic_term(coupling_cm: float, separation_cm: float, alpha: float, beta_cm2: float) -> float:
    """Return the HDCPT2 second-order contribution for one state pair."""
    if coupling_cm == 0.0 or separation_cm == 0.0:
        # The two diabatic limits have opposite signs. At exact degeneracy
        # use their symmetric principal value; an arbitrary signed zero would
        # otherwise select one branch through copysign.
        return 0.0
    epsilon = abs(separation_cm) / 2.0
    amplitude = abs(coupling_cm)
    # Scale before squaring: finite pair shifts need not have finite v**2.
    scale = max(epsilon, amplitude)
    a, d = amplitude / scale, epsilon / scale
    dcpt2 = np.copysign(amplitude * (a / (hypot(d, a) + d)), separation_cm)
    weight_vpt2 = (tanh(alpha * (abs(coupling_cm) * epsilon - beta_cm2)) + 1.0) / 2.0
    perturbative = (amplitude * (weight_vpt2 * amplitude / separation_cm)
                    if weight_vpt2 else 0.0)
    result = perturbative + (1.0 - weight_vpt2) * dcpt2
    if not np.isfinite(result):
        raise FloatingPointError("HDCPT2 pair shift exceeds the floating-point range")
    return result


def solve_hdcpt2(
    force_field: QuarticForceField,
    *,
    options: VCIOptions | None = None,
    alpha: float = 1.0,
    beta_cm2: float = 5.0e5,
    harmonic_modes: tuple[int, ...] = (),
    degeneracy_tolerance_cm: float = 1.0e-8,
) -> HDCPT2Result:
    """Compute fundamentals and an effective x matrix without a variational step.

    ``harmonic_modes`` are zero-based indices.  All cubic and quartic terms
    containing them are omitted, so their frequencies remain harmonic.
    The effective diagonal x values reproduce the computed fundamentals;
    overtone residuals disclose the non-quadratic part of the DCPT2 energies.
    """
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be finite and positive")
    if not np.isfinite(beta_cm2) or beta_cm2 < 0.0:
        raise ValueError("beta_cm2 must be finite and non-negative")
    if not np.isfinite(degeneracy_tolerance_cm) or degeneracy_tolerance_cm < 0.0:
        raise ValueError("degeneracy_tolerance_cm must be finite and non-negative")
    opts = options or VCIOptions()
    opts.validate()
    qff = _selected_force_field(force_field, opts)
    frequencies = np.asarray(qff.harmonic_frequencies_cm, dtype=float)
    n_modes = len(frequencies)
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
        raise ValueError("harmonic frequencies must be finite and positive")
    if any(mode < 0 or mode >= n_modes for mode in harmonic_modes):
        raise ValueError("harmonic mode index out of range")
    masked = frozenset(harmonic_modes)
    cubic = tuple((tuple(sorted(key)), float(value)) for key, value in qff.cubic_cm.items() if not masked.intersection(key))
    quartic = tuple((tuple(sorted(key)), float(value)) for key, value in qff.quartic_cm.items() if not masked.intersection(key))
    powers = {power: _x_matrix_power(6, power) for power in range(5)}
    zero = (0,) * n_modes
    singles = []
    overtones = []
    pairs = []
    for i in range(n_modes):
        state = list(zero)
        state[i] = 1
        singles.append(tuple(state))
        state[i] = 2
        overtones.append(tuple(state))
        for j in range(i + 1, n_modes):
            state = list(zero)
            state[i] = state[j] = 1
            pairs.append((i, j, tuple(state)))

    def evaluate(state: tuple[int, ...]) -> VPT2State:
        harmonic = _harmonic_energy(frequencies, state)
        first = sum(value * _term_matrix_element(indices, state, state, powers) for indices, value in quartic)
        second = 0.0
        for other, coupling in _cubic_couplings_from_state(state, cubic, powers).items():
            if other == state:
                continue
            separation = harmonic - _harmonic_energy(frequencies, other)
            if abs(separation) <= degeneracy_tolerance_cm:
                separation = 0.0
            second += _hybrid_cubic_term(coupling, separation, alpha, beta_cm2)
        return VPT2State(state, harmonic, first, second)

    states = tuple(evaluate(state) for state in (zero, *singles, *overtones, *(state for _, _, state in pairs)))
    ground = states[0].energy_cm
    fundamentals = np.array([states[1 + i].energy_cm - ground for i in range(n_modes)])
    x = np.zeros((n_modes, n_modes), dtype=float)
    for (i, j, _), state in zip(pairs, states[1 + 2 * n_modes:]):
        x[i, j] = x[j, i] = state.energy_cm - states[1 + i].energy_cm - states[1 + j].energy_cm + ground
    for i in range(n_modes):
        x[i, i] = (fundamentals[i] - frequencies[i] - 0.5 * np.sum(x[i])) / 2.0
    overtones_cm = np.array([states[1 + n_modes + i].energy_cm - ground for i in range(n_modes)])
    modeled_overtones = 2.0 * frequencies + 6.0 * np.diag(x) + np.sum(x, axis=1) - np.diag(x)
    return HDCPT2Result(frequencies, fundamentals, x, states, alpha, beta_cm2, overtones_cm - modeled_overtones)


def solve_hdcpt2_from_anharmonic_input(input_data: AnharmonicInput, **kwargs: object) -> HDCPT2Result:
    return solve_hdcpt2(force_field_from_anharmonic_input(input_data), **kwargs)
