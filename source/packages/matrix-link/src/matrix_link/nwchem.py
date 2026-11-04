"""NWChem energy/gradient adapter used by LINK."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil

import numpy as np

from matrix_core import CalculationLaunchAuthorization, CalculationResources
from matrix_engines import (
    ExternalQMJobStatus,
    ExternalQMProgramSpec,
    ExternalQMRunResult,
    external_qm_job_status,
    run_external_qm_job,
)


_APPLE_SILICON_CONDA_NWCHEM = Path("/opt/miniconda3/envs/nwchem/bin/nwchem")
NWCHEM_EXECUTABLE = (
    os.environ.get("MATRIX_NWCHEM_EXE")
    or shutil.which("nwchem")
    or (
        str(_APPLE_SILICON_CONDA_NWCHEM)
        if _APPLE_SILICON_CONDA_NWCHEM.is_file()
        else "nwchem"
    )
)
NWCHEM_SPEC = ExternalQMProgramSpec(
    name="NWChem",
    default_executable=NWCHEM_EXECUTABLE,
    env_var="MATRIX_NWCHEM_EXE",
    input_candidates=("nwchem.nw", "input.nw", "molecule.nw"),
    output_suffix=".out",
    normal_markers=("Total times  cpu:",),
    error_markers=(
        "There is an error in the input file",
        "failed to open the input file",
        "MPI_ABORT was invoked",
        "SCF failed to converge",
        "convergence failure",
    ),
    pid_file="nwchem.pid",
    pass_input_basename=True,
    extra_args_before_input=True,
)

_ENERGY_RE = re.compile(
    r"Total\s+(SCF|DFT|MP2)\s+energy\s*(?:=\s*)?"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][-+]?\d+)?)",
    re.I,
)
_GRADIENT_ROW_RE = re.compile(
    r"^\s*\d+\s+[A-Za-z][A-Za-z0-9]*\s+"
    r"([-+0-9.DEde]+)\s+([-+0-9.DEde]+)\s+([-+0-9.DEde]+)\s+"
    r"([-+0-9.DEde]+)\s+([-+0-9.DEde]+)\s+([-+0-9.DEde]+)\s*$"
)
_DFT_FUNCTIONALS = {
    "b3lyp",
    "blyp",
    "bp86",
    "hcth",
    "m05",
    "m05-2x",
    "m06",
    "m06-2x",
    "pbe",
    "pbe0",
    "pw91",
    "revpbe",
    "scan",
    "tpss",
}
_DFT_XC_DIRECTIVES = {
    "pbe": "xpbe96 cpbe96",
    "pbe96": "xpbe96 cpbe96",
    "bp86": "becke88 perdew86",
    "blyp": "becke88 lyp",
    "pw91": "xperdew91 perdew91",
}
_NWCHEM_FAILURE_RE = re.compile(
    r"(?:SCF\s+failed\s+to\s+converge|convergence\s+failure|"
    r"There\s+is\s+an\s+error\s+in\s+the\s+input\s+file|"
    r"failed\s+to\s+open\s+the\s+input\s+file|MPI_ABORT\s+was\s+invoked)",
    re.I,
)
BOHR_TO_ANGSTROM = 0.529177210903


class NWChemConvergenceError(RuntimeError):
    """Raised when NWChem explicitly reports a failed electronic solution."""


@dataclass(frozen=True)
class NWChemPointResult:
    """One NWChem energy and Cartesian gradient in atomic units."""

    path: Path
    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray | None
    normal_termination: bool
    coordinates_bohr: np.ndarray | None = None
    method_family: str = ""


def write_nwchem_point_input(
    output: Path | str,
    atoms: Sequence[str],
    coordinates_angstrom,
    *,
    method: str = "scf",
    basis: str = "sto-3g",
    route: str = "",
    charge: int = 0,
    multiplicity: int = 1,
    gradient: bool = True,
    memory_gb: int | None = None,
    restart: bool = False,
) -> Path:
    """Write an NWChem input for a LINK Cartesian energy/gradient point."""

    coords = np.asarray(coordinates_angstrom, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("NWChem point coordinates must have shape natoms x 3")
    if len(atoms) != coords.shape[0]:
        raise ValueError("NWChem atom count does not match coordinate array")
    if int(multiplicity) < 1:
        raise ValueError("NWChem multiplicity must be positive")

    module, model_lines = _nwchem_model(method, int(multiplicity))
    lines = [
        f"{'restart' if restart else 'start'} matrix_link_point",
        "geometry units angstrom noautoz nocenter noautosym",
    ]
    lines.extend(
        f"  {atom:<3s} {float(xyz[0]): .16f} {float(xyz[1]): .16f} {float(xyz[2]): .16f}"
        for atom, xyz in zip(atoms, coords, strict=True)
    )
    lines.extend(
        [
            "end",
            "basis",
            f"  * library {str(basis).strip() or 'sto-3g'}",
            "end",
            f"charge {int(charge)}",
        ]
    )
    if memory_gb is not None:
        memory = int(memory_gb)
        if memory < 1:
            raise ValueError("NWChem memory must be a positive number of GB")
        lines.append(f"memory total {memory} gb")
    lines.extend(model_lines)
    route_lines = [line.rstrip() for line in str(route).splitlines() if line.strip()]
    lines.extend(route_lines)
    lines.append(f"task {module} {'gradient' if gradient else 'energy'}")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def read_nwchem_point_result(
    output: Path | str,
    *,
    require_gradient: bool = True,
) -> NWChemPointResult:
    """Parse the final NWChem energy and analytic Cartesian gradient."""

    target = Path(output)
    text = target.read_text(encoding="utf-8", errors="replace")
    failure = _NWCHEM_FAILURE_RE.search(text)
    if failure is not None:
        raise NWChemConvergenceError(
            f"NWChem electronic calculation failed: {failure.group(0)} ({target})"
        )
    energies = [
        (match.group(1).upper(), _as_float(match.group(2)))
        for match in _ENERGY_RE.finditer(text)
    ]
    if not energies:
        raise ValueError(f"NWChem output contains no parseable point energy: {target}")
    gradient_block = _last_nwchem_gradient_block(text.splitlines())
    coordinates = None if gradient_block is None else gradient_block[0]
    gradient = None if gradient_block is None else gradient_block[1]
    if gradient is None and require_gradient:
        raise ValueError(f"NWChem output contains no Cartesian gradient: {target}")
    return NWChemPointResult(
        path=target,
        energy_hartree=float(energies[-1][1]),
        gradient_hartree_per_bohr=gradient,
        normal_termination="Total times  cpu:" in text,
        coordinates_bohr=coordinates,
        method_family=energies[-1][0],
    )


def nwchem_job_status(
    workdir: Path | str,
    *,
    input_path: Path | str | None = None,
    output_path: Path | str | None = None,
) -> ExternalQMJobStatus:
    return external_qm_job_status(
        workdir,
        NWCHEM_SPEC,
        input_path=input_path,
        output_path=output_path,
    )


def run_nwchem_job(
    workdir: Path | str,
    *,
    executable: str | None = None,
    input_path: Path | str | None = None,
    output_path: Path | str | None = None,
    background: bool = False,
    timeout: float | None = None,
    env: Mapping[str, str] | None = None,
    extra_args: Sequence[str] = (),
    launcher: Mapping[str, object] | None = None,
    resources: CalculationResources | None = None,
    launch_authorization: CalculationLaunchAuthorization | None = None,
) -> ExternalQMRunResult:
    """Launch NWChem, including a portable Conda-prefix library setup."""

    run_env = dict(env or {})
    launcher_environment = (
        launcher.get("environment", {}) if isinstance(launcher, Mapping) else {}
    )
    if not isinstance(launcher_environment, Mapping):
        raise ValueError("NWChem launcher environment must be an object")
    values = {
        "processes": str(1 if resources is None else resources.process_count),
        "threads": str(1 if resources is None else resources.threads_per_process),
    }
    run_env.update(
        {
            str(key): str(value).format_map(values)
            for key, value in launcher_environment.items()
        }
    )
    if resources is not None:
        run_env.setdefault("OMP_NUM_THREADS", str(resources.threads_per_process))
        run_env.setdefault("OPENBLAS_NUM_THREADS", str(resources.threads_per_process))
    resolved = _resolve_nwchem_executable(executable, run_env)
    _populate_nwchem_libraries(resolved, run_env)
    launch_executable, launch_args = _nwchem_launch_command(
        resolved,
        extra_args,
        launcher=launcher,
        resources=resources,
    )
    return run_external_qm_job(
        workdir,
        NWCHEM_SPEC,
        executable=launch_executable,
        input_path=input_path,
        output_path=output_path,
        background=background,
        timeout=timeout,
        env=run_env,
        extra_args=launch_args,
        resources=resources,
        launch_authorization=launch_authorization,
    )


def _nwchem_model(method: str, multiplicity: int) -> tuple[str, list[str]]:
    key = str(method).strip().casefold() or "scf"
    if key in {"hf", "rhf", "rohf", "uhf", "scf"}:
        spin_names = {
            1: "singlet",
            2: "doublet",
            3: "triplet",
            4: "quartet",
            5: "quintet",
            6: "sextet",
        }
        spin = spin_names.get(multiplicity, f"nopen {multiplicity - 1}")
        reference = key if key in {"rhf", "rohf", "uhf"} else ""
        directives = [item for item in (reference, spin, "thresh 1.0e-10") if item]
        return "scf", ["scf", *(f"  {item}" for item in directives), "end"]
    if key == "mp2":
        _, scf_lines = _nwchem_model("uhf" if multiplicity > 1 else "scf", multiplicity)
        return "mp2", [*scf_lines, "mp2", "end"]
    functional = key.removeprefix("dft:").strip()
    if key == "dft":
        functional = "pbe0"
    if key.startswith("dft:") or key in _DFT_FUNCTIONALS:
        xc_directive = _DFT_XC_DIRECTIVES.get(functional, functional)
        return "dft", ["dft", f"  xc {xc_directive}", f"  mult {multiplicity}", "end"]
    raise ValueError(
        "unsupported NWChem LINK method; use SCF/HF, MP2, a known DFT functional, "
        "or the explicit dft:<xc> form"
    )


def _last_nwchem_gradient(lines: Sequence[str]) -> np.ndarray | None:
    block = _last_nwchem_gradient_block(lines)
    return None if block is None else block[1]


def _last_nwchem_gradient_block(
    lines: Sequence[str],
) -> tuple[np.ndarray, np.ndarray] | None:
    blocks: list[np.ndarray] = []
    coordinate_blocks: list[np.ndarray] = []
    current_gradient: list[list[float]] | None = None
    current_coordinates: list[list[float]] | None = None
    for line in lines:
        if "ENERGY GRADIENTS" in line.upper():
            if current_gradient:
                blocks.append(np.asarray(current_gradient, dtype=float).reshape(-1))
                coordinate_blocks.append(np.asarray(current_coordinates, dtype=float))
            current_gradient = []
            current_coordinates = []
            continue
        if current_gradient is None:
            continue
        match = _GRADIENT_ROW_RE.match(line)
        if match is not None:
            assert current_coordinates is not None
            current_coordinates.append(
                [_as_float(match.group(index)) for index in (1, 2, 3)]
            )
            current_gradient.append(
                [_as_float(match.group(index)) for index in (4, 5, 6)]
            )
        elif current_gradient and not line.strip():
            blocks.append(np.asarray(current_gradient, dtype=float).reshape(-1))
            coordinate_blocks.append(np.asarray(current_coordinates, dtype=float))
            current_gradient = None
            current_coordinates = None
    if current_gradient:
        blocks.append(np.asarray(current_gradient, dtype=float).reshape(-1))
        coordinate_blocks.append(np.asarray(current_coordinates, dtype=float))
    return None if not blocks else (coordinate_blocks[-1], blocks[-1])


def _resolve_nwchem_executable(
    executable: str | None,
    environment: Mapping[str, str],
) -> str:
    selected = executable or environment.get(NWCHEM_SPEC.env_var) or os.environ.get(
        NWCHEM_SPEC.env_var
    )
    if selected:
        return str(Path(selected).expanduser()) if os.sep in selected else selected
    return NWCHEM_EXECUTABLE


def _nwchem_launch_command(
    executable: str,
    extra_args: Sequence[str] = (),
    *,
    launcher: Mapping[str, object] | None = None,
    resources: CalculationResources | None = None,
) -> tuple[str, tuple[str, ...]]:
    """Resolve a structured runner without invoking a shell."""

    if launcher:
        contract = dict(launcher)
        runner = str(contract.get("runner", "")).strip()
        if not runner:
            raise ValueError("NWChem structured launcher requires runner")
        processes = 1 if resources is None else int(resources.process_count)
        threads = 1 if resources is None else int(resources.threads_per_process)
        values = {
            "executable": str(contract.get("executable", executable)).strip() or executable,
            "processes": str(processes),
            "threads": str(threads),
        }
        arguments = tuple(
            str(item).format_map(values) for item in contract.get("arguments", ())
        )
        return runner, (*arguments, *map(str, extra_args))

    processes = 1 if resources is None else int(resources.process_count)
    micromamba = _micromamba_environment(executable)
    if micromamba is not None:
        runner, environment_name = micromamba
        clean_environment: tuple[str, ...] = (
            "run",
            "--clean-env",
            "-e",
            f"HOME={Path.home()}",
            "-e",
            f"USER={os.environ.get('USER', Path.home().name)}",
            "-e",
            f"TMPDIR={os.environ.get('TMPDIR', '/tmp')}",
            "-n",
            environment_name,
        )
        parallel = ("mpirun", "-np", str(processes)) if processes > 1 else ()
        return runner, (*clean_environment, *parallel, "nwchem", *map(str, extra_args))
    if processes > 1:
        selected = Path(executable).expanduser()
        mpi = selected.parent / "mpirun" if selected.is_absolute() else None
        mpi_executable = str(mpi) if mpi is not None and mpi.is_file() else shutil.which("mpirun")
        if not mpi_executable:
            raise FileNotFoundError("parallel NWChem launch requires mpirun")
        return mpi_executable, ("-np", str(processes), executable, *map(str, extra_args))
    return executable, tuple(str(item) for item in extra_args)


def _micromamba_environment(executable: str) -> tuple[str, str] | None:
    """Infer a micromamba environment from its executable path, without host names."""

    selected = Path(executable).expanduser()
    parts = selected.parts
    try:
        envs_index = parts.index("envs")
    except ValueError:
        return None
    if envs_index < 1 or envs_index + 1 >= len(parts) or "micromamba" not in parts[:envs_index]:
        return None
    environment_name = parts[envs_index + 1]
    candidates = (
        shutil.which("micromamba"),
        str(Path.home() / ".local" / "bin" / "micromamba"),
    )
    runner = next((item for item in candidates if item and Path(item).is_file()), None)
    return None if runner is None else (str(runner), environment_name)


def _populate_nwchem_libraries(executable: str, environment: dict[str, str]) -> None:
    resolved = Path(executable).expanduser()
    if not resolved.is_absolute():
        found = shutil.which(executable)
        if found is None:
            return
        resolved = Path(found)
    prefix = resolved.parent.parent
    basis = prefix / "share" / "nwchem" / "libraries"
    nwpw = prefix / "share" / "nwchem" / "libraryps"
    if basis.is_dir():
        environment.setdefault("NWCHEM_BASIS_LIBRARY", f"{basis}{os.sep}")
    if nwpw.is_dir():
        environment.setdefault("NWCHEM_NWPW_LIBRARY", f"{nwpw}{os.sep}")


def _as_float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


__all__ = [
    "NWCHEM_EXECUTABLE",
    "NWCHEM_SPEC",
    "NWChemConvergenceError",
    "NWChemPointResult",
    "nwchem_job_status",
    "read_nwchem_point_result",
    "run_nwchem_job",
    "write_nwchem_point_input",
]
