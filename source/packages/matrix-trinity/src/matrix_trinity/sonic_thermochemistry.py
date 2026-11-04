"""SONIC HDCPT2 energies and vibrational thermochemistry with Podolsky Vg."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from math import comb

import numpy as np

from matrix_chem import Phy, get_physical_constants

from .curvilinear_vpt2 import (
    SonicGVPT2Result, SonicVPT2Input, _cubic_couplings, _modalize_sonic_field,
    _quartic_expectation,
)


@dataclass(frozen=True)
class SonicHDCPT2Result:
    harmonic_frequencies_cm: np.ndarray
    fundamentals_cm: np.ndarray
    zero_point_cm: float
    x_matrix_cm: np.ndarray
    temperatures_K: np.ndarray
    log_partition_vibrational: np.ndarray
    internal_energy_kj_mol: np.ndarray
    entropy_j_mol_K: np.ndarray
    heat_capacity_j_mol_K: np.ndarray
    pseudopotential_cm: float
    alpha: float
    beta_cm2: float
    partition_model: str
    partition_max_quanta: int | None
    partition_state_count: int | None
    partition_tail_fraction: np.ndarray
    partition_tail_frequencies_cm: np.ndarray | None
    partition_tail_anchor_states: np.ndarray | None
    degeneracy_tolerance_cm: float
    degenerate_pair_count: int


@dataclass(frozen=True)
class SonicGVPT2ThermochemistryResult:
    zero_point_cm: float
    temperatures_K: np.ndarray
    log_partition_vibrational: np.ndarray
    internal_energy_kj_mol: np.ndarray
    entropy_j_mol_K: np.ndarray
    heat_capacity_j_mol_K: np.ndarray
    partition_state_count: int
    highest_shell_fraction: np.ndarray
    max_total_quanta: int


def _effective_x_levels(
    fundamentals: np.ndarray, x_matrix: np.ndarray, max_quanta: int,
    max_states: int,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Enumerate local effective-x energies in increasing excitation shells."""
    n = len(fundamentals)
    levels = [0.0]
    shells = [0]
    states = [np.zeros(n)]
    for quanta in range(1, max_quanta + 1):
        if len(levels) + comb(n + quanta - 1, quanta) > max_states:
            break
        for occupied in combinations_with_replacement(range(n), quanta):
            state = np.bincount(occupied, minlength=n).astype(float)
            energy = float(
                np.dot(fundamentals, state)
                + np.dot(np.diag(x_matrix), state * (state - 1))
                + 0.5 * state @ (x_matrix - np.diag(np.diag(x_matrix))) @ state
            )
            if energy <= 0 or not np.isfinite(energy):
                raise ValueError("effective-x level is nonpositive; reduce the partition cutoff")
            levels.append(energy)
            shells.append(quanta)
            states.append(state)
    if len(levels) == 1 and max_quanta > 0:
        raise ValueError("partition state limit cannot include all fundamentals")
    return np.asarray(levels), np.asarray(shells), int(shells[-1]), np.asarray(states)


def solve_sonic_gvpt2_thermochemistry(
    result: SonicGVPT2Result, *,
    temperatures_K: tuple[float, ...] = (298.15, 500.0, 1000.0),
) -> SonicGVPT2ThermochemistryResult:
    """Sum the complete finite GVPT2 level basis, including mixed polyads.

    This is the adiabatic route across a resonance crossing. The highest-shell
    population exposes the finite-state truncation at each temperature.
    """
    temperatures = np.asarray(temperatures_K, dtype=float)
    if temperatures.ndim != 1 or np.any(temperatures <= 0) or not np.all(np.isfinite(temperatures)):
        raise ValueError("thermochemistry requires positive finite temperatures")
    states = result.states
    if not states or any(len(state) != len(states[0]) for state in states):
        raise ValueError("GVPT2 thermochemistry requires a nonempty common mode basis")
    shells = np.asarray([sum(state) for state in states])
    max_quanta = int(np.max(shells))
    expected = sum(comb(len(states[0]) + q - 1, q) for q in range(max_quanta + 1))
    if len(states) != expected or len(set(states)) != expected:
        raise ValueError("GVPT2 thermochemistry requires complete excitation shells")
    energies = np.asarray(result.deperturbed_cm, dtype=float).copy()
    for block in result.blocks:
        members = tuple(block["indices"])
        roots = np.asarray(block["eigenvalues_cm"], dtype=float)
        if roots.shape != (len(members),):
            raise ValueError("GVPT2 thermochemistry found an incomplete polyad spectrum")
        energies[list(members)] = roots
    ground = states.index((0,) * len(states[0]))
    energies = energies - energies[ground]
    if energies.shape != (len(states),) or np.any(energies[np.arange(len(states)) != ground] <= 0):
        raise ValueError("GVPT2 thermochemistry requires positive excitation energies")
    if not np.all(np.isfinite(energies)) or not np.isfinite(result.zero_point_cm):
        raise ValueError("GVPT2 thermochemistry requires finite energies")
    constants = get_physical_constants()
    hc = constants[Phy.PLANCK] * constants[Phy.C_LIGHT]
    inverse_temperature = hc / (constants[Phy.BOLTZMANN] * temperatures[:, None])
    weights = np.exp(-inverse_temperature * energies[None, :])
    sums = np.sum(weights, axis=1)
    probabilities = weights / sums[:, None]
    mean = probabilities @ energies
    variance = probabilities @ (energies**2) - mean**2
    gas_constant = constants[Phy.AVOGADRO] * constants[Phy.BOLTZMANN]
    return SonicGVPT2ThermochemistryResult(
        zero_point_cm=float(result.zero_point_cm), temperatures_K=temperatures,
        log_partition_vibrational=(
            -inverse_temperature[:, 0] * result.zero_point_cm + np.log(sums)
        ),
        internal_energy_kj_mol=hc * constants[Phy.AVOGADRO] * (result.zero_point_cm + mean) / 1000,
        entropy_j_mol_K=gas_constant * (np.log(sums) + inverse_temperature[:, 0] * mean),
        heat_capacity_j_mol_K=(
            gas_constant * inverse_temperature[:, 0] ** 2 * np.maximum(variance, 0)
        ),
        partition_state_count=len(states),
        highest_shell_fraction=np.sum(probabilities[:, shells == max_quanta], axis=1),
        max_total_quanta=max_quanta,
    )


def solve_sonic_hdcpt2(
    field: SonicVPT2Input, *,
    temperatures_K: tuple[float, ...] = (298.15, 500.0, 1000.0),
    alpha: float = 1.0, beta_cm2: float = 5.0e5,
    harmonic_modes: tuple[int, ...] = (),
    partition_model: str = "effective_x", max_total_quanta: int = 8,
    max_partition_states: int = 30000,
    degeneracy_tolerance_cm: float = 1.0e-8,
) -> SonicHDCPT2Result:
    """Apply MATRIX HDCPT2 to the complete curvilinear Hamiltonian.

    All potential and kinetic cubic couplings use the same smooth denominator
    replacement. The full quartic/kinetic expectation and equilibrium Vg enter
    each state; Vg affects ZPVE and the absolute partition function.
    """
    try:
        from matrix_vpt2_vci.hdcpt2 import _hybrid_cubic_term
    except ImportError as exc:
        raise RuntimeError("SONIC HDCPT2 requires matrix-trinity[anharmonic]") from exc

    if not np.isfinite(alpha) or alpha <= 0 or not np.isfinite(beta_cm2) or beta_cm2 < 0:
        raise ValueError("invalid HDCPT2 alpha or beta")
    if not np.isfinite(degeneracy_tolerance_cm) or degeneracy_tolerance_cm < 0:
        raise ValueError("invalid HDCPT2 degeneracy tolerance")
    if partition_model not in {"effective_x", "oscillator", "matched_spt"}:
        raise ValueError("unsupported vibrational partition model")
    if max_total_quanta < (0 if partition_model == "matched_spt" else 1) or max_partition_states < 2:
        raise ValueError("partition cutoff and state limit must be positive")
    temperatures = np.asarray(temperatures_K, dtype=float)
    if temperatures.ndim != 1 or np.any(temperatures <= 0) or not np.all(np.isfinite(temperatures)):
        raise ValueError("thermochemistry requires positive finite temperatures")
    frequencies, _modes, f3, f4, g1, g2 = _modalize_sonic_field(field)
    n = len(frequencies)
    if any(mode < 0 or mode >= n for mode in harmonic_modes):
        raise ValueError("harmonic mode index out of range")
    if harmonic_modes:
        f3, f4, g1, g2 = (array.copy() for array in (f3, f4, g1, g2))
        for mode in harmonic_modes:
            for array in (f3, f4, g1, g2):
                for axis in range(array.ndim):
                    index = [slice(None)] * array.ndim
                    index[axis] = mode
                    array[tuple(index)] = 0
    zero = (0,) * n
    singles = tuple(tuple(int(j == i) for j in range(n)) for i in range(n))
    overtones = tuple(tuple(2 * int(j == i) for j in range(n)) for i in range(n))
    pairs = tuple(tuple(int(j in {i, k}) for j in range(n))
                  for i in range(n) for k in range(i + 1, n))
    degenerate_pairs: set[frozenset[tuple[int, ...]]] = set()

    def energy(state: tuple[int, ...]) -> float:
        value = float(np.dot(frequencies, np.asarray(state) + 0.5))
        value += _quartic_expectation(state, f4, g2, field.pseudopotential_cm, 0.0)
        for other, coupling in _cubic_couplings(state, f3, g1, 0.0).items():
            if other == state or abs(coupling) < 1e-14:
                continue
            separation = float(np.dot(frequencies, np.subtract(state, other)))
            if abs(separation) <= degeneracy_tolerance_cm:
                separation = 0.0
                degenerate_pairs.add(frozenset((state, other)))
            value += _hybrid_cubic_term(abs(coupling), separation, alpha, beta_cm2)
        return value

    ground = energy(zero)
    single_energies = np.array([energy(state) for state in singles])
    overtone_energies = np.array([energy(state) for state in overtones])
    pair_energies = np.array([energy(state) for state in pairs])
    fundamentals = single_energies - ground
    if np.any(fundamentals <= 0):
        raise ValueError("HDCPT2 fundamentals must be positive for thermochemistry")
    x_matrix = np.zeros((n, n))
    for i in range(n):
        x_matrix[i, i] = (overtone_energies[i] - 2 * single_energies[i] + ground) / 2
    for value, (i, j) in zip(pair_energies,
                             ((i, j) for i in range(n) for j in range(i + 1, n))):
        x_matrix[i, j] = x_matrix[j, i] = value - single_energies[i] - single_energies[j] + ground

    constants = get_physical_constants()
    hc = constants[Phy.PLANCK] * constants[Phy.C_LIGHT]
    gas_constant = constants[Phy.AVOGADRO] * constants[Phy.BOLTZMANN]
    if partition_model == "oscillator" or (
        partition_model != "matched_spt" and not np.any(x_matrix)
    ):
        x = hc * fundamentals[None, :] / (constants[Phy.BOLTZMANN] * temperatures[:, None])
        occupation = 1 / np.expm1(x)
        log_q = -hc * ground / (constants[Phy.BOLTZMANN] * temperatures) - np.sum(
            np.log(-np.expm1(-x)), axis=1,
        )
        internal_energy = hc * constants[Phy.AVOGADRO] * (
            ground + np.sum(fundamentals[None, :] * occupation, axis=1)
        ) / 1000
        entropy = gas_constant * np.sum(
            x * occupation - np.log(-np.expm1(-x)), axis=1,
        )
        cv = gas_constant * np.sum(x**2 * occupation * (1 + occupation), axis=1)
        actual_model = "oscillator" if partition_model == "oscillator" else "harmonic-exact"
        actual_quanta = state_count = None
        tail_fraction = np.zeros_like(temperatures)
        tail_frequencies = tail_anchors = None
    else:
        levels, shells, actual_quanta, states = _effective_x_levels(
            fundamentals, x_matrix, max_total_quanta, max_partition_states,
        )
        state_count = len(levels)
        inverse_temperature = hc / (constants[Phy.BOLTZMANN] * temperatures[:, None])
        weights = np.exp(-inverse_temperature * levels[None, :])
        if partition_model == "matched_spt":
            from .statistical_partition import matched_partition_moments
            off_diagonal = x_matrix - np.diag(np.diag(x_matrix))
            def boundary_transitions(anchor, first_active):
                local = fundamentals + 2 * np.diag(x_matrix) * anchor + off_diagonal @ anchor
                if actual_quanta:
                    local[first_active] -= 2 * x_matrix[first_active, first_active]
                return local[first_active:]
            matched = matched_partition_moments(
                levels, shells, states.astype(int), inverse_temperature[:, 0], boundary_transitions,
            )
            log_sum = matched['log_partition']
            mean = matched['mean_energy_cm']
            variance = matched['energy_variance_cm2']
            tail_fraction = matched['tail_fraction']
            tail_frequencies = matched['tail_frequencies_cm']
            tail_anchors = matched['tail_anchor_states']
            actual_model = "matched_spt"
        else:
            sums = np.sum(weights, axis=1)
            probabilities = weights / sums[:, None]
            log_sum = np.log(sums)
            mean = probabilities @ levels
            variance = probabilities @ (levels**2) - mean**2
            tail_fraction = np.sum(probabilities[:, shells == actual_quanta], axis=1)
            tail_frequencies = tail_anchors = None
            actual_model = "effective_x"
        log_q = -inverse_temperature[:, 0] * ground + log_sum
        internal_energy = hc * constants[Phy.AVOGADRO] * (ground + mean) / 1000
        entropy = gas_constant * (log_sum + inverse_temperature[:, 0] * mean)
        cv = gas_constant * inverse_temperature[:, 0] ** 2 * np.maximum(variance, 0)
    return SonicHDCPT2Result(
        harmonic_frequencies_cm=frequencies, fundamentals_cm=fundamentals,
        zero_point_cm=ground, x_matrix_cm=x_matrix,
        temperatures_K=temperatures, log_partition_vibrational=log_q,
        internal_energy_kj_mol=internal_energy,
        entropy_j_mol_K=entropy, heat_capacity_j_mol_K=cv,
        pseudopotential_cm=float(field.pseudopotential_cm),
        alpha=alpha, beta_cm2=beta_cm2,
        partition_model=actual_model, partition_max_quanta=actual_quanta,
        partition_state_count=state_count, partition_tail_fraction=tail_fraction,
        partition_tail_frequencies_cm=tail_frequencies,
        partition_tail_anchor_states=tail_anchors,
        degeneracy_tolerance_cm=degeneracy_tolerance_cm,
        degenerate_pair_count=len(degenerate_pairs),
    )


__all__ = [
    "SonicHDCPT2Result", "SonicGVPT2ThermochemistryResult",
    "solve_sonic_hdcpt2", "solve_sonic_gvpt2_thermochemistry",
]
