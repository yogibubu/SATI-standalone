import numpy as np
import pytest
from itertools import permutations

from matrix_trinity.vibronic_harmonic import HarmonicVibronicModel, harmonic_gaussian_generator
from matrix_trinity.vibronic_ti import _integral_dag
from matrix_trinity.vibronic_vpt2 import SAMVPT2Field
from matrix_trinity.vibronic_variational import (
    embed_variational_states, harmonic_overlap_matrix,
    project_anharmonic_potential, solve_quartic_variational, variational_vibronic_transitions,
)


def field(w=100.0, cubic=15.0, quartic=8.0):
    return SAMVPT2Field(np.array([w]), np.array([[[cubic]]]), np.array([[[[quartic]]]]))


def model(wi, wf, shift, j=None):
    wi, wf = np.asarray(wi), np.asarray(wf)
    return HarmonicVibronicModel(wi, wf, np.eye(len(wi)) if j is None else j,
                                np.asarray(shift), 400.0, np.zeros(3), np.zeros((3, len(wi))))


def test_bare_solver_matches_direct_oscillator_polynomial_and_converges():
    states = solve_quartic_variational(field(), maximum_total_quanta=24, level_count=8)
    # Build q powers in a padded space, then project: powers of a truncated q
    # would incorrectly lose virtual paths at the basis boundary.
    a = np.diag(np.sqrt(np.arange(1, 30)), 1)
    q = (a + a.T)/np.sqrt(2)
    h = np.diag(100*(np.arange(30)+0.5)) + 15*np.linalg.matrix_power(q, 3)/6 + 8*np.linalg.matrix_power(q, 4)/24
    assert np.allclose(states.energies_cm, np.linalg.eigvalsh(h[:25, :25])[:8], atol=1e-9)
    larger = solve_quartic_variational(field(), maximum_total_quanta=32, level_count=8)
    assert np.max(abs(states.energies_cm-larger.energies_cm)) < 0.002
    assert states.diagnostics['hdc_completion_used'] is False


def test_identical_anharmonic_surfaces_give_unit_transition_matrix():
    states = solve_quartic_variational(field(), maximum_total_quanta=24, level_count=10)
    result = variational_vibronic_transitions(
        model([100.0], [100.0], [0.0]), states, states, [0.001, 0, 0],
        initial_levels=np.arange(10), final_levels=np.arange(10), kind='absorption',
    )
    assert np.allclose(result['amplitudes'], np.eye(10), atol=1e-12)
    assert np.allclose(result['strengths_hartree2'].sum(axis=1), 1e-6)


def test_full_duschinsky_grid_matches_existing_integral_recurrence():
    theta = 0.4
    j = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    m = model([100.0, 130.0], [110.0, 140.0], [0.8, -0.2], j)
    initial = np.array([[0, 0], [1, 0], [0, 2]])
    final = np.array([[0, 0], [1, 1], [3, 0]])
    overlap, diagnostics = harmonic_overlap_matrix(m, initial, final)
    pairs = np.array([np.r_[i, f] for f in final for i in initial])
    g = harmonic_gaussian_generator(m)
    ids, values = _integral_dag(pairs, g['matrix'], g['linear'], np.zeros((3, 4)), 10000, np.exp(g['log_ground']))
    assert np.allclose(overlap, values[ids[:, 0]].reshape(3, 3), atol=1e-13)
    assert diagnostics['full_duschinsky_retained']


def test_anharmonic_overlaps_invariant_under_normal_mode_phase_change():
    initial = solve_quartic_variational(field(cubic=15), maximum_total_quanta=24, level_count=1)
    rephased = solve_quartic_variational(field(cubic=-15), maximum_total_quanta=24, level_count=1)
    final = solve_quartic_variational(field(w=120, cubic=-8), maximum_total_quanta=24, level_count=8)
    original = variational_vibronic_transitions(model([100.0], [120.0], [0.7]), initial, final, [1.0], kind='absorption')
    flipped = variational_vibronic_transitions(model([100.0], [120.0], [0.7], np.array([[-1.0]])), rephased, final, [1.0], kind='absorption')
    assert np.allclose(original['strengths_hartree2'], flipped['strengths_hartree2'], atol=1e-12)


def test_active_projection_transports_sign_and_scale_then_embeds_spectator_zpe():
    projected = project_anharmonic_potential(field(), [120.0], np.array([[-2.0]]))
    assert projected.cubic_cm.item() == -120
    assert projected.quartic_cm.item() == 128
    states = solve_quartic_variational(field(), maximum_total_quanta=12, level_count=4)
    embedded = embed_variational_states(states, [80.0, 100.0, 150.0], [1])
    assert np.allclose(embedded.energies_cm-states.energies_cm, 115.0)
    assert not np.any(embedded.basis[:, [0, 2]])


def test_unconfined_potential_and_oversized_overlap_fail_explicitly():
    with pytest.raises(ValueError, match='confinement'):
        solve_quartic_variational(field(quartic=-1.0))
    with pytest.raises(MemoryError):
        harmonic_overlap_matrix(model([100.0], [100.0], [0.0]),
                                np.array([[0], [100]]), np.array([[0], [100]]), maximum_grid_entries=1000)


def test_vacuum_spectators_keep_duschinsky_rotation_in_excited_overlap():
    angle = 0.4
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    overlap, _ = harmonic_overlap_matrix(
        model([100.0, 100.0], [100.0, 100.0], [0.0, 0.0], rotation),
        np.array([[1, 0]]), np.array([[1, 0]]),
    )
    assert np.allclose(overlap, np.cos(angle), atol=1e-14)


def test_coupled_two_mode_solver_matches_direct_product_oscillator_projection():
    cubic = np.zeros((2,)*3)
    quartic = np.zeros((2,)*4)
    for row in set(permutations((0, 0, 1))):
        cubic[row] = 3.0
    for row in set(permutations((0, 0, 1, 1))):
        quartic[row] = 2.0
    quartic[0, 0, 0, 0], quartic[1, 1, 1, 1] = 4.0, 6.0
    f = SAMVPT2Field(np.array([100.0, 130.0]), cubic, quartic)
    states = solve_quartic_variational(f, maximum_total_quanta=12, level_count=12)
    dimension = 17
    annihilation = np.diag(np.sqrt(np.arange(1, dimension)), 1)
    q = (annihilation+annihilation.T)/np.sqrt(2)
    q2, q4 = q@q, np.linalg.matrix_power(q, 4)
    eye = np.eye(dimension)
    energies = (np.arange(dimension)[:, None]+0.5)*100 + (np.arange(dimension)[None, :]+0.5)*130
    matrix = (np.diag(energies.ravel()) + 4*np.kron(q4, eye)/24 + 6*np.kron(eye, q4)/24
              + 3*np.kron(q2, q)/2 + 2*np.kron(q2, q2)/4)
    indices = states.basis[:, 0]*dimension + states.basis[:, 1]
    exact = np.linalg.eigvalsh(matrix[np.ix_(indices, indices)])[:12]
    assert np.allclose(states.energies_cm, exact, atol=1e-8)


def test_emission_swaps_both_wavefunctions_and_duschinsky_consistently():
    initial = solve_quartic_variational(field(), maximum_total_quanta=20, level_count=4)
    final = solve_quartic_variational(field(w=120, cubic=-8), maximum_total_quanta=20, level_count=4)
    m = model([100.0], [120.0], [0.7])
    arguments = dict(initial_levels=np.arange(4), final_levels=np.arange(4))
    absorption = variational_vibronic_transitions(m, initial, final, [0.001], kind='absorption', **arguments)
    emission = variational_vibronic_transitions(m, initial, final, [0.001], kind='emission', **arguments)
    assert np.allclose(emission['amplitudes'], absorption['amplitudes'].conj().T, atol=1e-12)
    assert np.allclose(emission['mismatch_cm'], absorption['mismatch_cm'].T, atol=1e-10)
