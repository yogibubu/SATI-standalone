import struct

import numpy as np

from matrix_apoc.scf_adapters import (
    SCF_BACKENDS, ExternalCheckpointSCFAdapter, MRCCDensSCFAdapter,
    MolproMoldenSCFAdapter, PySCFSCFAdapter, adapter_for_backend,
    read_mrcc_scf_densities, scf_state_from_mrcc_scf_densities,
    read_mrcc_ccdensities, scf_state_from_mrcc_ccdensities,
    PyFockSCFAdapter,
)
from matrix_apoc.state_reuse import SCFBackendRequest, SCFStateManager


def _request(backend):
    return SCFBackendRequest(
        backend=backend, method="test", basis="test", charge=0,
        multiplicity=1, electronic_state=0, coordinates=np.zeros((1, 3)),
    )


def test_every_non_pyfock_backend_has_a_fail_closed_adapter():
    for backend in SCF_BACKENDS:
        adapter = adapter_for_backend(backend)
        assert adapter.backend == backend
        assert adapter.capabilities().maximum_history >= 1
        manager = SCFStateManager(adapter)
        guess = manager.prepare_guess(_request(backend))
        assert guess.status == "standard_guess"


def test_callbacks_enable_reuse_only_for_the_declared_backend():
    def export(result):
        return result

    adapter = adapter_for_backend("pyscf", export=export)
    assert adapter.capabilities().in_process
    assert not adapter.capabilities().density_projection


def test_all_adapters_are_connected_to_the_common_lifecycle_with_fallback():
    for backend in SCF_BACKENDS:
        adapter = adapter_for_backend(
            backend, evaluate=lambda request, guess: {"backend": request.backend}
        )
        result, guess, state = adapter.evaluate_with_state(_request(backend))
        assert result["backend"] == backend
        assert guess.status == "standard_guess"
        assert state is None


def test_pyscf_adapter_projects_and_extrapolates_with_a_verified_transform():
    def result(request, guess):
        return {
            "coordinates": request.coordinates,
            "density_alpha": np.eye(2),
            "fock_alpha": np.eye(2),
            "electron_count": 2,
        }

    adapter = PySCFSCFAdapter(
        method="test", basis="test", evaluate=result,
        basis_transform=lambda state, geometry, basis: np.eye(2),
    )
    manager = SCFStateManager(adapter)
    request0 = _request("pyscf")
    request0 = request0.__class__(**{**request0.__dict__, "method": "test", "basis": "test"})
    adapter.evaluate_with_state(request0, manager=manager)
    moved = request0.__class__(**{**request0.__dict__, "coordinates": np.array([[1.0, 0.0, 0.0]])})
    _, guess, state = adapter.evaluate_with_state(moved, manager=manager)
    assert guess.status == "same_basis_reuse"
    assert state is not None


def test_pyscf_native_cross_overlap_is_computed_from_two_geometries():
    atoms = ("H", "H")
    def result(request, guess):
        return {
            "coordinates": request.coordinates,
            "density_alpha": np.eye(2),
            "electron_count": 2,
        }
    adapter = PySCFSCFAdapter(
        method="HF", basis="sto-3g", atoms=atoms, evaluate=result,
    )
    manager = SCFStateManager(adapter)
    first = _request("pyscf").__class__(**{
        **_request("pyscf").__dict__, "method": "HF", "basis": "sto-3g",
        "coordinates": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
    })
    adapter.evaluate_with_state(first, manager=manager)
    second = first.__class__(**{
        **first.__dict__, "basis": "6-31g",
        "coordinates": np.array([[0.1, 0.0, 0.0], [0.0, 0.0, 0.8]]),
    })
    _, guess, _ = adapter.evaluate_with_state(second, manager=manager)
    assert guess.status == "projected"
    assert guess.reason == "backend_basis_projection"


def test_external_checkpoint_adapter_is_conservative_without_cross_overlap():
    adapter = ExternalCheckpointSCFAdapter("orca", method="HF", basis="minimal")
    assert adapter.capabilities().orbital_restart
    assert not adapter.capabilities().density_projection


def test_molpro_and_mrcc_extrapolation_contracts_are_available():
    molpro = MolproMoldenSCFAdapter(
        method="HF", basis="minimal", coordinates=np.zeros((1, 3)),
    )
    assert molpro.backend == "molpro"
    assert molpro.capabilities().density_projection

    def load_state(result):
        return result
    mrcc = MRCCDensSCFAdapter(method="HF", basis="minimal", state_loader=load_state)
    assert mrcc.backend == "mrcc"
    assert mrcc.capabilities().density_projection


def test_mrcc_scf_densities_loader_reads_sparse_fortran_records(tmp_path):
    # Same layout as the real MRCC SCFDENSITIES record: n, spin, (value,i,j)*n.
    entries = ((0.60245568919, 1, 1), (0.125, 2, 1), (0.60245568919, 2, 2))
    payload = struct.pack("<ii", len(entries), 0)
    payload += b"".join(struct.pack("<dii", *entry) for entry in entries)
    artifact = tmp_path / "SCFDENSITIES"
    artifact.write_bytes(
        struct.pack("<i", len(payload)) + payload + struct.pack("<i", len(payload))
    )

    blocks = read_mrcc_scf_densities(artifact)
    np.testing.assert_allclose(
        blocks[0], [[0.60245568919, 0.125], [0.125, 0.60245568919]]
    )
    state = scf_state_from_mrcc_scf_densities(
        artifact, method="HF", basis="STO-3G", coordinates=np.zeros((2, 3))
    )
    assert state.backend == "mrcc"
    assert state.provenance["format"] == "mrcc-sparse-fortran"


def test_mrcc_ccdensities_loader_separates_one_and_two_body_terms(tmp_path):
    artifact = tmp_path / "CCDENSITIES"
    artifact.write_text(
        " 1.0D+00   1   1   1   1\n"
        " 2.0D+00   1   1   0   0\n"
        " 3.0D+00   2   2   0   0\n",
        encoding="ascii",
    )
    densities = read_mrcc_ccdensities(artifact)
    np.testing.assert_allclose(densities.one_body, np.diag([2.0, 3.0]))
    assert densities.two_body[(1, 1, 1, 1)] == 1.0
    state = scf_state_from_mrcc_ccdensities(
        artifact, method="CCSD", basis="STO-3G", coordinates=np.zeros((2, 3)),
        mo_coefficients=np.eye(2),
    )
    np.testing.assert_allclose(state.density_alpha, densities.one_body)


def test_pyfock_adapter_uses_general_same_basis_lifecycle():
    def evaluate(request, guess):
        value = 1.0 + float(request.coordinates[0, 0])
        return {
            "coordinates": request.coordinates,
            "density_alpha": np.eye(2) * value,
            "fock_alpha": np.eye(2) * value,
            "electron_count": 2,
        }

    adapter = PyFockSCFAdapter(
        method="PBE", basis="sto-3g", atoms=("H",), evaluate=evaluate
    )
    manager = SCFStateManager(adapter)
    request0 = _request("pyfock").__class__(
        **{**_request("pyfock").__dict__, "method": "PBE", "basis": "sto-3g"}
    )
    adapter.evaluate_with_state(request0, manager=manager)
    request1 = request0.__class__(
        **{**request0.__dict__, "coordinates": np.array([[0.1, 0.0, 0.0]])}
    )
    _, guess, _ = adapter.evaluate_with_state(request1, manager=manager)
    assert guess.status == "same_basis_reuse"
    assert adapter_for_backend("pyfock", method="PBE").backend == "pyfock"
