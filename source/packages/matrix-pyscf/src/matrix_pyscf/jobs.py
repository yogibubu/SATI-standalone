from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from matrix_core import CalculationLaunchAuthorization, CalculationResources

from matrix_engines import (
    ExternalQMJobStatus,
    ExternalQMProgramSpec,
    ExternalQMRunResult,
    external_qm_job_status,
    run_external_qm_job,
)


PYSCF_EXECUTABLE = "pyscf-python"
PYSCF_SPEC = ExternalQMProgramSpec(
    name="PySCF",
    default_executable=PYSCF_EXECUTABLE,
    env_var="MATRIX_PYSCF_EXE",
    input_candidates=("pyscf_job.py", "input.py", "molecule.py"),
    output_suffix=".out",
    normal_markers=("MATRIX PYSCF TERMINATED NORMALLY",),
    error_markers=("Traceback (most recent call last)", "MATRIX PYSCF FAILED"),
    pid_file="pyscf.pid",
)


def pyscf_job_status(
    workdir: Path | str,
    *,
    input_path: Path | str | None = None,
    output_path: Path | str | None = None,
) -> ExternalQMJobStatus:
    return external_qm_job_status(
        workdir,
        PYSCF_SPEC,
        input_path=input_path,
        output_path=output_path,
    )


def run_pyscf_job(
    workdir: Path | str,
    *,
    executable: str | None = None,
    input_path: Path | str | None = None,
    output_path: Path | str | None = None,
    background: bool = False,
    timeout: float | None = None,
    env: Mapping[str, str] | None = None,
    extra_args: Sequence[str] = (),
    resources: CalculationResources | None = None,
    launch_authorization: CalculationLaunchAuthorization | None = None,
) -> ExternalQMRunResult:
    return run_external_qm_job(
        workdir,
        PYSCF_SPEC,
        executable=executable,
        input_path=input_path,
        output_path=output_path,
        background=background,
        timeout=timeout,
        env=env,
        extra_args=extra_args,
        resources=resources,
        launch_authorization=launch_authorization,
    )
