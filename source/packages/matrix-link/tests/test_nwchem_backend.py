from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import matrix_link.nwchem as nwchem_module

from matrix_core import (
    CALCULATION_AUTHORIZATION_BUNDLE_ENV,
    CalculationResources,
    authorize_calculation_launch,
    build_calculation_launch_plan,
    write_calculation_authorization_bundle,
)
from matrix_link import (
    NWCHEM_SPEC,
    QMScanBackend,
    ScanPoint,
    read_nwchem_point_result,
    run_qm_scan_points,
    write_nwchem_point_input,
)
from matrix_gui.owner_engines.keymaker_pipeline import KeymakerPipeline


def test_nwchem_point_writer_uses_cartesian_atomic_gradient_contract(tmp_path):
    target = write_nwchem_point_input(
        tmp_path / "nwchem.nw",
        ("H", "H"),
        np.asarray([[0.0, 0.0, -0.35], [0.0, 0.0, 0.35]]),
        method="scf",
        basis="sto-3g",
        memory_gb=2,
    )

    text = target.read_text(encoding="utf-8")
    assert "geometry units angstrom noautoz nocenter noautosym" in text
    assert "* library sto-3g" in text
    assert "memory total 2 gb" in text
    assert "task scf gradient" in text


def test_nwchem_point_parser_reads_last_energy_and_gradient(tmp_path):
    output = tmp_path / "nwchem.out"
    output.write_text(
        """
 Total SCF energy = -1.000000000000D+00
                         RHF ENERGY GRADIENTS

    atom               coordinates                        gradient
                 x          y          z           x          y          z
   1 H       0.000000   0.000000  -0.661404    0.000000   0.000000   0.013743
   2 H       0.000000   0.000000   0.661404    0.000000   0.000000  -0.013743

 Total SCF energy = -1.117349033688D+00
 Total times  cpu:        0.1s     wall:        0.1s
""",
        encoding="utf-8",
    )

    result = read_nwchem_point_result(output)

    assert result.energy_hartree == pytest.approx(-1.117349033688)
    assert result.gradient_hartree_per_bohr == pytest.approx(
        [0.0, 0.0, 0.013743, 0.0, 0.0, -0.013743]
    )
    assert result.normal_termination
    assert np.asarray(result.coordinates_bohr) == pytest.approx(
        np.asarray([[0.0, 0.0, -0.661404], [0.0, 0.0, 0.661404]])
    )
    assert result.method_family == "SCF"


def test_nwchem_dft_writer_has_explicit_functional_and_multiplicity(tmp_path):
    target = write_nwchem_point_input(
        tmp_path / "nwchem.nw",
        ("H",),
        np.zeros((1, 3)),
        method="dft:pbe0",
        multiplicity=2,
    )

    text = target.read_text(encoding="utf-8")
    assert "dft\n  xc pbe0\n  mult 2\nend" in text
    assert "task dft gradient" in text


def test_nwchem_open_shell_mp2_uses_ump2_reference(tmp_path):
    target = write_nwchem_point_input(
        tmp_path / "nwchem.nw",
        ("H",),
        np.zeros((1, 3)),
        method="mp2",
        basis="sto-3g",
        multiplicity=2,
    )

    text = target.read_text(encoding="utf-8")
    assert "scf\n  uhf\n  doublet\n" in text
    assert "task mp2 gradient" in text


@pytest.mark.parametrize(("family", "energy"), (("DFT", -75.1), ("MP2", -75.2)))
def test_nwchem_parser_supports_dft_and_mp2_gradient_headers(tmp_path, family, energy):
    output = tmp_path / f"{family.casefold()}.out"
    output.write_text(
        f"""
 Total {family} energy = {energy: .12E}
                         U{family} ENERGY GRADIENTS

    atom               coordinates                        gradient
                 x          y          z           x          y          z
   1 O       0.000000   0.000000   0.000000    0.000000   0.001000   0.002000

 Total times  cpu:        0.1s     wall:        0.1s
""",
        encoding="utf-8",
    )

    result = read_nwchem_point_result(output)

    assert result.method_family == family
    assert result.energy_hartree == pytest.approx(energy)
    assert result.gradient_hartree_per_bohr == pytest.approx([0.0, 0.001, 0.002])


def test_nwchem_is_a_registered_link_backend():
    assert NWCHEM_SPEC.name == "NWChem"
    assert QMScanBackend("nwchem").name == "nwchem"


def test_structured_nwchem_launcher_is_host_independent(tmp_path):
    micromamba = tmp_path / "micromamba"
    executable = tmp_path / "envs" / "nwchem" / "bin" / "nwchem"
    micromamba.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    micromamba.chmod(0o755)
    executable.chmod(0o755)
    launcher, arguments = nwchem_module._nwchem_launch_command(
        str(executable),
        ("--some-nwchem-option",),
        launcher={
            "executable": str(executable),
            "runner": str(micromamba),
            "arguments": (
                "run",
                "--clean-env",
                "-n",
                "nwchem",
                "mpirun",
                "-np",
                "{processes}",
                "{executable}",
            ),
        },
        resources=CalculationResources(
            process_count=4,
            threads_per_process=1,
            memory_per_job_gb=8,
        ),
    )

    assert launcher == str(micromamba)
    assert arguments[:2] == ("run", "--clean-env")
    assert arguments[-5:] == (
        "mpirun",
        "-np",
        "4",
        str(executable),
        "--some-nwchem-option",
    )


def test_nwchem_restart_writer_and_explicit_scf_failure(tmp_path):
    target = write_nwchem_point_input(
        tmp_path / "nwchem.nw",
        ("H",),
        np.zeros((1, 3)),
        restart=True,
    )
    assert target.read_text(encoding="utf-8").startswith("restart matrix_link_point\n")

    output = tmp_path / "failed.out"
    output.write_text(
        "Total SCF energy = -1.0\nSCF failed to converge\nTotal times  cpu: 1.0s\n",
        encoding="utf-8",
    )
    with pytest.raises(nwchem_module.NWChemConvergenceError, match="failed to converge"):
        read_nwchem_point_result(output, require_gradient=False)


@pytest.mark.external_qm
def test_installed_nwchem_h2_gradient_through_link(
    tmp_path,
    monkeypatch,
):
    xyzin = tmp_path / "h2.xyz"
    xyzin.write_text("2\nH2 LINK/NWChem qualification\nH 0 0 -0.35\nH 0 0 0.35\n")
    pipeline = KeymakerPipeline(tmp_path / "keymaker-session")
    pipeline.prepare_input(
        source_type="XYZ",
        source=str(xyzin),
        molecule_count=1,
        open_in_avogadro=False,
    )
    try:
        state = pipeline.prepare_menu_request(
            "exploitation",
            options={
                "program": "nwchem",
                "qm_program": "nwchem",
                "optimizer": "link",
                "method": "scf",
                "basis": "sto-3g",
                "execution_target": "local",
                "cpu_cores": 2,
                "memory_gb": 1,
            },
        )
    except ValueError as exc:
        if "not available" in str(exc) or "selected target" in str(exc):
            pytest.skip("the qualified local NWChem installation is unavailable")
        raise
    request_path = Path(state["menus"]["exploitation"]["request_path"])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    options = request["options"]
    backend_configuration = options["backend_configuration"]
    executable = Path(str(backend_configuration["executable"]))
    if not executable.is_file():
        pytest.skip("the qualified local NWChem executable is unavailable")
    coordinates = np.asarray([[0.0, 0.0, -0.35], [0.0, 0.0, 0.35]])
    point = ScanPoint(index=0, displacement=0.0, coordinates_angstrom=coordinates)
    resources = CalculationResources(
        process_count=2,
        threads_per_process=1,
        memory_per_job_gb=1.0,
    )
    plan = build_calculation_launch_plan(
        backend="test-parent",
        input_path=xyzin,
        command=("test-parent", str(xyzin)),
        resources=resources,
    )
    authorization = authorize_calculation_launch(plan, approved_plan_sha256=plan.sha256)
    bundle = write_calculation_authorization_bundle(
        tmp_path / "calculation-authorization.json",
        plan,
        authorization,
    )
    monkeypatch.setenv(CALCULATION_AUTHORIZATION_BUNDLE_ENV, str(bundle))

    result = run_qm_scan_points(
        xyzin,
        (point,),
        QMScanBackend(
            "nwchem",
            method="scf",
            basis="sto-3g",
            executable=str(executable),
            launcher=backend_configuration.get("launcher"),
            processors=int(options["cpu_cores"]),
            memory_gb=1,
        ),
        run_dir=tmp_path / "run",
    )[0]

    assert result.status == "completed", result.message
    assert result.energy_hartree == pytest.approx(-1.117349033688, abs=1.0e-10)
    assert result.gradient_hartree_per_bohr is not None
    assert result.gradient_hartree_per_bohr.shape == (6,)
    assert np.linalg.norm(result.gradient_hartree_per_bohr) > 0.0
    assert result.execution["restart_artifact_format"] == "nwchem_db"
    assert result.execution["processes"] == 2
    assert result.execution["threads_per_process"] == 1
    restart_artifact = Path(str(result.execution["restart_artifact"]))
    assert restart_artifact.is_file()

    restarted = run_qm_scan_points(
        xyzin,
        (point,),
        QMScanBackend(
            "nwchem",
            method="scf",
            basis="sto-3g",
            executable=str(executable),
            launcher=backend_configuration.get("launcher"),
            processors=1,
            memory_gb=1,
            restart_artifact=restart_artifact,
        ),
        run_dir=tmp_path / "restart-run",
    )[0]
    assert restarted.status == "completed", restarted.message
    assert restarted.energy_hartree == pytest.approx(result.energy_hartree, abs=1.0e-10)
    assert restarted.execution["restart_source_role"] == "accepted_point"
    assert restarted.execution["processes"] == 1
    assert restarted.execution["threads_per_process"] == 1

    qualifications = (
        ("open-shell-scf", "scf", "1\nH doublet\nH 0 0 0\n", 2, "SCF"),
        ("dft", "PBE", xyzin.read_text(encoding="utf-8"), 1, "DFT"),
        ("mp2", "MP2", xyzin.read_text(encoding="utf-8"), 1, "MP2"),
    )
    for label, method, geometry_text, multiplicity, family in qualifications:
        model_geometry = tmp_path / f"{label}.xyz"
        model_geometry.write_text(geometry_text, encoding="utf-8")
        model_coordinates = (
            np.zeros((1, 3)) if multiplicity == 2 else coordinates
        )
        model_point = ScanPoint(0, 0.0, model_coordinates)
        qualified = run_qm_scan_points(
            model_geometry,
            (model_point,),
            QMScanBackend(
                "nwchem",
                method=method,
                basis="sto-3g",
                executable=str(executable),
                launcher=backend_configuration.get("launcher"),
                processors=1,
                memory_gb=1,
                multiplicity=multiplicity,
            ),
            run_dir=tmp_path / f"{label}-run",
        )[0]
        assert qualified.status == "completed", qualified.message
        assert qualified.execution["method_family"] == family
        assert qualified.gradient_hartree_per_bohr is not None
        assert qualified.backend_coordinates_angstrom == pytest.approx(model_coordinates)
