import numpy as np

from matrix_apoc.state_reuse import (
    SCFBackendRequest,
    SCFGuess,
    SCFState,
    SCFStateCapabilities,
    SCFStateContractError,
    SCFStateManager,
    extrapolate_scf_state,
    project_scf_state,
    validate_state_for_request,
)


def request(method="hf"):
    return SCFBackendRequest(
        backend="fake", method=method, basis="minimal", charge=0,
        multiplicity=1, electronic_state=0, coordinates=np.zeros((2, 3)),
        electron_count=2,
    )


def state(method="hf", value=1.0):
    return SCFState(
        backend="fake", method=method, basis="minimal", charge=0,
        multiplicity=1, electronic_state=0, coordinates=np.zeros((2, 3)),
        density_alpha=np.eye(2) * value, fock_alpha=np.eye(2), electron_count=2,
    )


class FakeAdapter:
    def __init__(self):
        self.calls = []

    def capabilities(self):
        return SCFStateCapabilities(
            density_projection=True, density_extrapolation=True,
            fock_projection=True, fock_extrapolation=True,
        )

    def export_state(self, result):
        return result

    def project_state(self, parent, target_geometry, target_basis):
        self.calls.append(("project", parent))
        return SCFGuess(density_alpha=parent.density_alpha, status="projected")

    def extrapolate_state(self, history, target_geometry, target_basis):
        self.calls.append(("extrapolate", tuple(history)))
        return SCFGuess(density_alpha=history[-1].density_alpha, status="extrapolated")

    def evaluate(self, request, guess=None):
        return state()


def test_manager_projects_then_extrapolates_and_fork_is_independent():
    adapter = FakeAdapter()
    manager = SCFStateManager(adapter)
    assert manager.prepare_guess(request()).status == "standard_guess"
    manager.record_converged(state(value=1.0))
    assert manager.prepare_guess(request()).status == "projected"
    manager.record_converged(state(value=2.0))
    assert manager.prepare_guess(request()).status == "extrapolated"
    branch = manager.fork(child="fd-plus")
    manager.record_converged(state(value=3.0), branch=branch)
    assert len(manager._histories["root"]) == 2
    assert len(manager._histories[branch]) == 3
    assert [call[0] for call in adapter.calls] == ["project", "extrapolate"]


def test_evaluate_and_record_uses_guess_then_records_only_converged_result():
    manager = SCFStateManager(FakeAdapter())
    result, guess, recorded = manager.evaluate_and_record(request())
    assert guess.status == "standard_guess"
    assert result is recorded
    assert manager.events[-1]["event"] == "state_recorded"


def test_incompatible_state_is_rejected_before_projection():
    with np.testing.assert_raises(SCFStateContractError):
        validate_state_for_request(state(method="dft"), request(method="hf"))


def test_incompatible_history_falls_back_and_is_audited():
    manager = SCFStateManager(FakeAdapter())
    manager.record_converged(state())
    guess = manager.prepare_guess(request(method="dft"))
    assert guess.status == "standard_guess_fallback"
    assert manager.events[-2]["event"] == "standard_guess_fallback"
    assert "incompatible" in manager.events[-2]["reason"]


def test_negative_density_is_rejected():
    with np.testing.assert_raises(SCFStateContractError):
        state(value=-1.0)


def test_extrapolation_uses_more_than_two_states():
    history = [
        state(value=1.0),
        SCFState(**{**state(value=2.0).__dict__, "coordinates": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])}),
        SCFState(**{**state(value=3.0).__dict__, "coordinates": np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])}),
    ]
    guess = extrapolate_scf_state(
        history, np.array([[3.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), "minimal",
        lambda item, geometry, basis: project_scf_state(
            item, geometry, basis, lambda _state, _geometry, _basis: np.eye(2)
        ),
    )
    assert guess.status == "extrapolated"
    np.testing.assert_allclose(guess.density_alpha, np.eye(2) * 4.0)


def test_projection_uses_dual_congruences_for_density_and_fock():
    transform = np.asarray([[1.1, 0.2], [0.1, 0.9]])
    source = state(value=1.0)
    source = SCFState(
        **{
            **source.__dict__,
            "fock_alpha": np.asarray([[-0.8, 0.15], [0.15, 0.3]]),
        }
    )
    guess = project_scf_state(
        source,
        np.ones((2, 3)),
        "minimal",
        lambda *_args: transform,
    )
    orbitals = np.asarray([[0.7, -0.1], [0.2, 1.0]])
    projected_orbitals = transform @ orbitals
    np.testing.assert_allclose(
        projected_orbitals.T @ guess.fock_alpha @ projected_orbitals,
        orbitals.T @ source.fock_alpha @ orbitals,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        guess.density_alpha,
        transform @ source.density_alpha @ transform.T,
    )
