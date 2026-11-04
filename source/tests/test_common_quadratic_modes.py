import numpy as np
import pytest

from matrix_trinity import joint_quadratic_modes, common_quadratic_model
from matrix_trinity.vibronic_harmonic import ESALCHarmonicState
from matrix_gf.harmonic import HESSIAN_EIGENVALUE_TO_CM, CM_PER_HARTREE


def state(f, g=None, point=None, gradient=None, identity='shared'):
    n = len(f)
    return ESALCHarmonicState(np.array(f, float), np.eye(n) if g is None else g,
        np.zeros(n) if point is None else point, identity, gradient)


def test_commuting_and_noncommuting_objectives():
    angle = .37
    r = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    a = state(r @ np.diag([1., 3.]) @ r.T)
    b = state(r @ np.diag([2., 5.]) @ r.T)
    frame = joint_quadratic_modes([a, b])
    assert frame.converged
    assert max(frame.relative_offdiagonal_norms) < 1e-12
    b = state([[2., .3], [.3, 4.]])
    frame = joint_quadratic_modes([a, b])
    assert frame.converged
    assert min(frame.relative_offdiagonal_norms) > 1e-4
    assert np.max(np.diff(frame.objective_history)) < 1e-12
    # Exhaustive plane search provides an independent 2D objective check.
    values = []
    for t in np.linspace(0, np.pi / 2, 3001):
        u = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        values.append(sum((u.T @ s.force_constants_hartree @ u)[0, 1] ** 2 for s in (a, b)))
    assert frame.objective_history[-1] <= min(values) + 1e-12


def test_full_affine_transport_preserves_potential_metric_and_frequencies():
    a = state([[2., .2], [.2, 3.]], np.array([[1., .1], [.1, 2.]]),
              np.array([.2, -.3]), np.array([.01, .02]))
    b = state([[4., .4], [.4, 1.]], np.diag([1.1, 1.7]), np.array([-.2, .4]))
    frame = joint_quadratic_modes([a, b], origin=[.1, .2])
    for old, new in zip((a, b), frame.states):
        assert new.modes()[0] == pytest.approx(old.modes()[0])
        x = np.array([.3, -.1])
        ds = frame.origin + frame.transform @ x - old.reference_esalc
        dx = x - new.reference_esalc
        go = np.zeros(2) if old.gradient_hartree is None else old.gradient_hartree
        gn = np.zeros(2) if new.gradient_hartree is None else new.gradient_hartree
        assert .5 * ds @ old.force_constants_hartree @ ds + go @ ds == pytest.approx(
            .5 * dx @ new.force_constants_hartree @ dx + gn @ dx)
    assert not np.allclose(frame.states[0].metric_inverse_amu, frame.states[1].metric_inverse_amu)


def test_shared_ho_projection_displacement_and_kinetic_couplings():
    omega = np.array([500., 800.])
    f = np.diag((omega / HESSIAN_EIGENVALUE_TO_CM) ** 2)
    basis = ((0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2))
    model = common_quadratic_model(state(f), basis, omega)
    assert model.reference_energies_cm == pytest.approx((np.array(basis) + .5) @ omega)
    assert all(abs(value) < 1e-10 for _, _, value in model.external_k_edges)
    # Direct q*p-free analytic <00|H|11> including distinct kinetic metric.
    f[0, 1] = f[1, 0] = 1e-9
    metric = np.array([[1., .002], [.002, 1.]])
    point = np.array([.02, -.03])
    model = common_quadratic_model(state(f, metric, point), basis, omega,
        frequency_tolerance_cm=1e-8, mixing_weight_tolerance=1e-8)
    edges = {(i, j): v for i, j, v in (*model.polyad_k_edges, *model.external_k_edges)}
    expected = .5 * (f[0, 1] * HESSIAN_EIGENVALUE_TO_CM ** 2 / np.sqrt(omega.prod())
                       - metric[0, 1] * np.sqrt(omega.prod()))
    assert edges[0, 3] == pytest.approx(expected)
    length = HESSIAN_EIGENVALUE_TO_CM / np.sqrt(CM_PER_HARTREE * omega)
    assert edges[0, 1] == pytest.approx(-length[0] * (f @ point)[0] * CM_PER_HARTREE / np.sqrt(2))


def test_chart_and_stability_gates():
    with pytest.raises(ValueError, match='same identified'):
        joint_quadratic_modes([state(np.eye(2)), state(np.eye(2), identity='other')])
    with pytest.raises(ValueError):
        joint_quadratic_modes([state(np.eye(2)), state(np.eye(2))], weights=[0., 0.])


def test_noncommuting_many_mode_sweeps_and_perturbative_scaling():
    rng = np.random.default_rng(41)
    matrices = []
    for _ in range(2):
        a = rng.normal(size=(6, 6))
        matrices.append(a.T @ a + np.eye(6))
    frame = joint_quadratic_modes([state(f) for f in matrices])
    assert frame.converged
    assert np.max(np.diff(frame.objective_history)) < 1e-10
    assert frame.transform.T @ frame.transform == pytest.approx(np.eye(6), abs=1e-12)
    # Existing PT2 owner on an explicitly finite two-state projection.
    omega = np.array([500., 800.])
    diagonal = np.diag((omega / HESSIAN_EIGENVALUE_TO_CM) ** 2)
    errors = []
    for scale in (.1, .2):
        f = diagonal.copy()
        f[0, 1] = f[1, 0] = scale * 1e-3
        model = common_quadratic_model(state(f), ((0, 0), (1, 1)), omega,
            frequency_tolerance_cm=100., mixing_weight_tolerance=.49)
        v = model.external_k_edges[0][2]
        exact = np.linalg.eigvalsh(np.diag(model.reference_energies_cm) + [[0, v], [v, 0]])
        errors.append(np.max(abs(model.operative_energies_cm() - exact)))
    assert errors[1] / errors[0] == pytest.approx(16., rel=.02)
