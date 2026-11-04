"""External std2 spectra from an explicitly supplied ground-state Molden file.

std2 owns the excited-state model. This adapter neither recomputes its kernel
nor interprets an excitation energy as a total electronic energy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil

from matrix_core import CalculationResources, load_keymaker_protocol_library
from matrix_engines import ExternalQMProgramSpec, run_external_qm_job
from matrix_core.calculation_launch import build_calculation_launch_plan


STD2_PROTOCOL_ID = "link-std2-spectrum.v1"
STD2_SPEC = ExternalQMProgramSpec(
    name="std2", default_executable="std2", env_var="MATRIX_STD2_EXE",
    input_candidates=("reference.molden",), output_suffix=".out",
    normal_markers=("sTDA done.", "sTD-DFT done."),
    pid_file="std2.pid", pass_input_basename=True, extra_args_before_input=True,
)
_RSH_PRESETS = {"CAMB3LYP", "wB97XD2", "wB97XD3", "wB97MV"}


@dataclass(frozen=True)
class STD2Settings:
    """Molden conventions and model choices must match the producing backend."""

    molden_style: int
    method: str = "stda"
    exact_exchange: float | None = None
    rsh_preset: str | None = None
    energy_window_ev: float = 7.0
    selection_exponent: float = 4.0

    def arguments(self) -> tuple[str, ...]:
        if self.molden_style not in (0, 1, 2, 3):
            raise ValueError("std2 Molden style must be 0, 1, 2 or 3")
        if self.method not in {"stda", "stddft", "xstda", "xstddft"}:
            raise ValueError("unsupported std2 method")
        for value in (self.energy_window_ev, self.selection_exponent):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("std2 energy window and selection exponent must be positive")
        args = ["-sty", str(self.molden_style), "-e", str(self.energy_window_ev),
                "-p", str(self.selection_exponent)]
        if self.method.endswith("ddft"):
            args.append("-rpa")
        if self.rsh_preset is not None:
            if self.rsh_preset not in _RSH_PRESETS or not self.method.startswith("x"):
                raise ValueError("RSH requires an explicit supported XsTD preset")
            if self.exact_exchange is not None:
                raise ValueError("RSH preset and global exact_exchange are mutually exclusive")
            args.append("-" + self.rsh_preset)
        else:
            if self.exact_exchange is None or not math.isfinite(self.exact_exchange):
                raise ValueError("global models require explicit exact_exchange")
            if not 0 <= self.exact_exchange <= 1:
                raise ValueError("exact_exchange must lie between zero and one")
            args.extend(("-ax", str(self.exact_exchange)))
            if self.method.startswith("x"):
                args.append("-XsTD")
        # Zero requests all computed eigenvectors in std2's native TM format.
        return (*args, "-vectm", "0", "-f")


@dataclass(frozen=True)
class STD2Transition:
    state_id: int
    excitation_energy_ev: float
    oscillator_strength_length: float
    oscillator_strength_velocity: float
    rotatory_strength_length: float
    rotatory_strength_velocity: float


def read_std2_spectrum(workdir: Path | str) -> tuple[STD2Transition, ...]:
    """Require completion as Fortran STOP can return a zero exit status."""
    directory = Path(workdir)
    log = (directory / "reference.out").read_text(errors="replace")
    if not any(marker in log for marker in STD2_SPEC.normal_markers):
        raise RuntimeError("std2 did not complete; inspect reference.out")
    data = (directory / "tda.dat").read_text().splitlines()
    try:
        first = next(i for i, line in enumerate(data) if line.strip() == "DATXY") + 1
    except StopIteration as exc:
        raise ValueError("std2 tda.dat has no DATXY section") from exc
    transitions = []
    for line in data[first:]:
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 6:
            raise ValueError("invalid std2 spectral row")
        root = int(fields[0])
        values = tuple(float(item.replace("D", "E")) for item in fields[1:])
        if root <= 0 or not all(math.isfinite(item) for item in values) or values[0] <= 0:
            raise ValueError("invalid std2 transition")
        if transitions and root <= transitions[-1].state_id:
            raise ValueError("std2 roots must be strictly increasing")
        transitions.append(STD2Transition(root, *values))
    if not transitions:
        raise ValueError("std2 produced no transitions in the requested window")
    return tuple(transitions)


def prepare_std2_spectrum(
    molden: Path | str, workdir: Path | str, *, settings: STD2Settings,
    resources: CalculationResources, executable: str | None = None,
):
    """Prepare reproducible input and an exact resource launch plan, without running."""
    args = settings.arguments()
    if resources.process_count != 1 or resources.concurrent_jobs != 1:
        raise ValueError("std2 uses one process per spectrum; set OpenMP threads explicitly")
    source = Path(molden).expanduser().resolve(strict=True)
    raw = source.read_bytes()
    content = raw.decode("utf-8")
    if "[Molden Format]" not in content or "[MO]" not in content:
        raise ValueError("std2 requires a ground-state Molden wavefunction")
    # std2 accepts AU without parentheses; the numerical coordinates are unchanged.
    content = content.replace("[Atoms] (AU)", "[Atoms] AU")
    directory = Path(workdir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if any(path.name not in {"reference.molden", "std2-request.json"} for path in directory.iterdir()):
        raise FileExistsError("std2 requires a fresh work directory for every execution")
    target = directory / "reference.molden"
    if target.exists() and target.read_text() != content:
        raise FileExistsError("std2 reference.molden differs from the requested input")
    target.write_text(content)
    selected = executable or os.environ.get("MATRIX_STD2_EXE") or shutil.which("std2")
    if not selected:
        candidates = (Path("/opt/homebrew/bin/std2"), Path.home() / ".local/bin/std2")
        selected = next((str(path) for path in candidates if path.is_file()), None)
    if not selected:
        raise FileNotFoundError("std2 unavailable; set MATRIX_STD2_EXE")
    selected = shutil.which(selected) or str(Path(selected).expanduser().resolve(strict=True))
    request = {
        "schema": "matrix.link.std2.request.v1", "source": str(source),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "molden_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "settings": asdict(settings),
    }
    (directory / "std2-request.json").write_text(json.dumps(request, indent=2) + "\n")
    return build_calculation_launch_plan(
        backend="std2", input_path=target, workdir=directory,
        command=(selected, *args, target.name), resources=resources,
    )


def evaluate_std2_spectrum(plan, *, launch_authorization, timeout: float = 300.0) -> dict:
    """Execute a prepared Keymaker protocol using the common external QM launcher."""
    directory = Path(plan.workdir)
    if any(path.name not in {"reference.molden", "std2-request.json"} for path in directory.iterdir()):
        raise FileExistsError("std2 execution requires a fresh work directory")
    request = json.loads((directory / "std2-request.json").read_text())
    settings = STD2Settings(**request["settings"])
    expected = (plan.command[0], *settings.arguments(), "reference.molden")
    if plan.backend != "std2" or tuple(plan.command) != expected:
        raise ValueError("std2 launch plan differs from its request")
    if hashlib.sha256(Path(plan.input_path).read_bytes()).hexdigest() != request["molden_sha256"]:
        raise ValueError("std2 Molden input changed after preparation")
    library = load_keymaker_protocol_library()
    protocol = library.resolve(STD2_PROTOCOL_ID)
    if protocol.status != "approved" or (protocol.owner_api, protocol.action) != (
        __name__, "evaluate_std2_spectrum"
    ):
        raise RuntimeError("std2 protocol implementation changed")
    threads = str(plan.resources.threads_per_process)
    run = run_external_qm_job(
        directory, STD2_SPEC, executable=plan.command[0], input_path=plan.input_path,
        output_path=directory / "reference.out", extra_args=settings.arguments(),
        resources=plan.resources, launch_authorization=launch_authorization, timeout=timeout,
        env={"OMP_NUM_THREADS": threads, "OPENBLAS_NUM_THREADS": threads,
             "MKL_NUM_THREADS": threads},
    )
    if run.exit_code != 0:
        raise RuntimeError(run.message)
    log = (directory / "reference.out").read_text(errors="replace")
    marker = "sTD-DFT done." if settings.method.endswith("ddft") else "sTDA done."
    if marker not in log:
        raise RuntimeError("std2 did not complete the requested response model")
    transitions = read_std2_spectrum(directory)
    version = re.search(r"Version\s+([0-9.]+)", log)
    vector_files = [directory / name for name in ("ciss_a", "sing_a", "ucis_a", "unrs_a")
                    if (directory / name).is_file()]
    if len(vector_files) != 1:
        raise RuntimeError("std2 did not export an unambiguous excited-state vector artifact")
    result = {
        "schema": "matrix.link.std2.spectrum.v1", "backend": "std2",
        "executable": plan.command[0],
        "executable_sha256": hashlib.sha256(Path(plan.command[0]).read_bytes()).hexdigest(),
        "reported_version": None if version is None else version.group(1),
        "quantity": "vertical_excitation_energy", "energy_unit": "eV",
        "energy_print_precision_ev": 0.0001,
        "rotatory_strength_unit": "1e-40_cgs", "state_index_base": 1,
        "state_following_owner": "APOC",
        "state_adapter": "matrix_apoc.electronic_state_from_std2",
        "state_vectors": {
            "path": vector_files[0].name,
            "sha256": hashlib.sha256(vector_files[0].read_bytes()).hexdigest(),
            "format": "std2.turbomole.eigenpairs",
        },
        "transitions": [asdict(item) for item in transitions],
        "request": request, "launch_plan_sha256": plan.sha256,
        "keymaker_provenance": {
            "schema": "matrix.keymaker.protocol_binding.v1",
            "protocol_id": protocol.protocol_id, "sha256": protocol.sha256,
            "library_sha256": library.sha256,
        },
    }
    (directory / "spectrum.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


__all__ = ["STD2_SPEC", "STD2_PROTOCOL_ID", "STD2Settings", "STD2Transition",
           "prepare_std2_spectrum", "evaluate_std2_spectrum", "read_std2_spectrum"]
