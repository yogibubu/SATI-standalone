"""Persistent local and remote capability profile used by Keymaker."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import importlib.util
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Mapping

from .atomic_io import atomic_json_write


ENVIRONMENT_SCHEMA = "matrix.environment.v1"
RUNTIME_MACHINE_SCHEMA = "matrix.runtime-machine.v1"
DEFAULT_CONFIG_PATH = Path("~/.config/matrix/environment.json").expanduser()

PROGRAM_DEFINITIONS: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    ("gdv", "Gaussian development version (GDV)", "qm", ("matrix-gdv", "gdv-run")),
    ("g16", "Gaussian 16", "qm", ("g16",)),
    ("orca", "ORCA", "qm", ("orca",)),
    ("turbomole", "TURBOMOLE", "qm", ("ridft", "dscf")),
    ("nwchem", "NWChem", "qm", ("nwchem",)),
    ("molpro", "Molpro", "qm", ("molpro",)),
    ("mrcc", "MRCC", "qm", ("dmrcc",)),
    ("cfour", "CFOUR", "qm", ("xcfour",)),
    ("xtb", "xTB", "qm", ("xtb",)),
    ("pyscf", "PySCF bundled calculator", "qm", ("pyscf-python",)),
    ("pyfock", "PyFock bundled calculator", "qm", ()),
    ("psi4", "Psi4", "qm", ("psi4",)),
    ("et", "eT electronic-structure program", "qm", ("eT_launch.py", "eT")),
    ("avogadro2", "Avogadro 2", "viewer", ("avogadro2",)),
    ("avogadro1", "Avogadro 1", "viewer", ("avogadro",)),
    ("molden", "Molden", "viewer", ("molden",)),
    ("vmd", "VMD", "viewer", ("vmd",)),
)

PROGRAM_ENVIRONMENT = {
    "gdv": "MATRIX_GDV_EXE",
    "g16": "MATRIX_G16_EXE",
    "orca": "MATRIX_ORCA_EXE",
    "turbomole": "MATRIX_TURBOMOLE_EXE",
    "nwchem": "MATRIX_NWCHEM_EXE",
    "molpro": "MATRIX_MOLPRO_EXE",
    "mrcc": "MATRIX_MRCC_EXE",
    "cfour": "MATRIX_CFOUR_EXE",
    "xtb": "MATRIX_XTB_EXE",
    "pyscf": "MATRIX_PYSCF_EXE",
    "psi4": "MATRIX_PSI4_EXE",
    "et": "MATRIX_ET_EXE",
    "avogadro2": "MATRIX_AVOGADRO2_EXE",
    "avogadro1": "MATRIX_AVOGADRO1_EXE",
    "molden": "MATRIX_MOLDEN_EXE",
    "vmd": "MATRIX_VMD_EXE",
}

_MACOS_APPLICATIONS = {
    "avogadro2": Path("/Applications/Avogadro2.app"),
    "avogadro1": Path("/Applications/Avogadro.app"),
    "vmd": Path("/Applications/VMD.app"),
}

_COMMON_PROGRAM_PATHS = {
    "nwchem": (Path("/opt/miniconda3/envs/nwchem/bin/nwchem"),),
}

_LEGACY_KEYMAKER_BASENAME = "matrix-keymaker"


def normalize_operating_system(value: str) -> str:
    normalized = str(value).strip().casefold()
    return {
        "darwin": "macos",
        "mac": "macos",
        "macos": "macos",
        "linux": "linux",
    }.get(normalized, normalized)


def normalize_architecture(value: str) -> str:
    normalized = str(value).strip().casefold().replace("-", "_")
    return {
        "aarch64": "arm64",
        "arm64": "arm64",
        "amd64": "x86_64",
        "x64": "x86_64",
        "x86_64": "x86_64",
    }.get(normalized, normalized)


def _deployment_path_basename(value: str) -> str:
    return value.rstrip("/\\").replace("\\", "/").rsplit("/", 1)[-1].casefold()


def _is_separate_keymaker_deployment(value: str) -> bool:
    return _deployment_path_basename(value) == _LEGACY_KEYMAKER_BASENAME


def _migrate_legacy_deployment_path(value: str) -> str:
    """Map the former Keymaker-only checkout/runtime to the MATRIX deployment."""

    if not _is_separate_keymaker_deployment(value):
        return value
    normalized = value.rstrip("/\\")
    replacement = "MATRIX" if normalized.rsplit("/", 1)[-1][:1].isupper() else "matrix"
    parent, separator, _name = normalized.rpartition("/")
    return f"{parent}{separator}{replacement}" if separator else replacement


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class ProgramLauncher:
    """Structured, shell-free launch contract for one external program."""

    executable: str = ""
    runner: str = ""
    arguments: tuple[str, ...] = ()
    clean_environment: bool = False
    environment: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "executable", str(self.executable).strip())
        object.__setattr__(self, "runner", str(self.runner).strip())
        object.__setattr__(self, "arguments", tuple(str(item) for item in self.arguments))
        object.__setattr__(
            self,
            "environment",
            {str(key): str(value) for key, value in dict(self.environment).items()},
        )
        if self.arguments and not self.runner:
            raise ValueError("program launcher arguments require a runner")


def _program_launcher(value: object) -> ProgramLauncher | None:
    if value is None or value == {}:
        return None
    if isinstance(value, ProgramLauncher):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("program launcher must be an object")
    data = dict(value)
    data["arguments"] = tuple(data.get("arguments", ()))
    return ProgramLauncher(**data)


@dataclass(frozen=True)
class ExternalProgram:
    key: str
    label: str
    category: str
    executable: str = ""
    enabled: bool = False
    launcher: ProgramLauncher | None = None

    def __post_init__(self) -> None:
        if self.category not in {"qm", "viewer"}:
            raise ValueError(f"unsupported external-program category: {self.category}")
        object.__setattr__(self, "launcher", _program_launcher(self.launcher))

    @property
    def available(self) -> bool:
        return bool(resolve_program_path(self.executable))


@dataclass(frozen=True)
class MachineLimits:
    processors: int = 1
    memory_gb: float = 1.0
    gpu_count: int = 0
    neural_engine_count: int = 0
    max_concurrent_jobs: int = 1

    def __post_init__(self) -> None:
        if self.processors < 1:
            raise ValueError("processors must be positive")
        if self.memory_gb <= 0:
            raise ValueError("memory_gb must be positive")
        if self.gpu_count < 0:
            raise ValueError("gpu_count cannot be negative")
        if self.neural_engine_count < 0:
            raise ValueError("neural_engine_count cannot be negative")
        if self.max_concurrent_jobs < 1:
            raise ValueError("max_concurrent_jobs must be positive")


@dataclass(frozen=True)
class RuntimeMachine:
    """Observed identity and hardware facts for the process running MATRIX."""

    node_name: str
    hostname: str
    operating_system: str
    operating_system_release: str
    architecture: str
    processor: str
    python_version: str
    python_abi: str
    limits: MachineLimits
    schema: str = RUNTIME_MACHINE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RUNTIME_MACHINE_SCHEMA:
            raise ValueError(f"unsupported runtime-machine schema: {self.schema}")
        if not self.node_name or not self.hostname:
            raise ValueError("runtime machine requires node and host names")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RemoteMachine:
    name: str
    host: str
    remote_root: str = "~/MATRIX"
    qm_root: str = "~/matrix"
    runtime_venv: str = "~/.venvs/matrix"
    runtime_config_dir: str = "~/.config/matrix"
    expected_operating_system: str = ""
    expected_architecture: str = ""
    expected_python_abi: str = ""
    enabled: bool = True
    limits: MachineLimits = field(default_factory=MachineLimits)
    engines: tuple[str, ...] = ()
    requires_vpn: bool = False
    notes: str = ""
    engine_launchers: Mapping[str, ProgramLauncher] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("remote machine name is required")
        if not self.host.strip():
            raise ValueError("remote machine SSH host is required")
        for label, value in (
            ("remote_root", self.remote_root),
            ("qm_root", self.qm_root),
            ("runtime_venv", self.runtime_venv),
            ("runtime_config_dir", self.runtime_config_dir),
        ):
            if not value.strip():
                raise ValueError(f"{label} is required")
            if _is_separate_keymaker_deployment(value):
                raise ValueError(
                    f"{label} cannot use a Keymaker-only deployment; "
                    "Keymaker must use the MATRIX checkout, virtualenv and configuration"
                )
        engines = {item.strip().casefold() for item in self.engines if item.strip()}
        object.__setattr__(self, "engines", tuple(sorted(engines)))
        launchers = {
            str(key).strip().casefold(): _program_launcher(value)
            for key, value in dict(self.engine_launchers).items()
        }
        if any(key not in engines for key in launchers):
            raise ValueError("remote engine launcher must refer to a declared engine")
        object.__setattr__(
            self,
            "engine_launchers",
            {key: value for key, value in launchers.items() if value is not None},
        )
        operating_system = normalize_operating_system(self.expected_operating_system)
        architecture = normalize_architecture(self.expected_architecture)
        object.__setattr__(self, "expected_operating_system", operating_system)
        object.__setattr__(self, "expected_architecture", architecture)
        object.__setattr__(self, "expected_python_abi", self.expected_python_abi.strip().casefold())


@dataclass(frozen=True)
class MatrixEnvironment:
    matrix_root: str
    projects_root: str
    programs: tuple[ExternalProgram, ...]
    local_limits: MachineLimits
    remote_machines: tuple[RemoteMachine, ...] = ()
    schema: str = ENVIRONMENT_SCHEMA
    created_utc: str = field(default_factory=_utc_now)
    updated_utc: str = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        if self.schema != ENVIRONMENT_SCHEMA:
            raise ValueError(f"unsupported MATRIX environment schema: {self.schema}")
        if not self.matrix_root.strip() or not self.projects_root.strip():
            raise ValueError("MATRIX and project roots are required")
        if _is_separate_keymaker_deployment(self.matrix_root):
            raise ValueError(
                "matrix_root cannot be a Keymaker-only checkout; Keymaker is part of MATRIX"
            )
        program_keys = [program.key for program in self.programs]
        if len(program_keys) != len(set(program_keys)):
            raise ValueError("external program keys must be unique")
        machine_names = [machine.name.strip().casefold() for machine in self.remote_machines]
        machine_hosts = [machine.host.strip().casefold() for machine in self.remote_machines]
        if len(machine_names) != len(set(machine_names)):
            raise ValueError("remote machine names must be unique")
        if len(machine_hosts) != len(set(machine_hosts)):
            raise ValueError("remote machine SSH aliases must be unique")
        identifier_owners: dict[str, int] = {}
        for index, machine in enumerate(self.remote_machines):
            for identifier in {
                machine.name.strip().casefold(),
                machine.host.strip().casefold(),
            }:
                owner = identifier_owners.setdefault(identifier, index)
                if owner != index:
                    raise ValueError(
                        "remote machine names and SSH aliases must not overlap across machines"
                    )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "MatrixEnvironment":
        raw_programs = data.get("programs", ())
        raw_machines = data.get("remote_machines", ())
        raw_local = data.get("local_limits", {})
        if not isinstance(raw_programs, (list, tuple)):
            raise ValueError("programs must be an array")
        if not isinstance(raw_machines, (list, tuple)):
            raise ValueError("remote_machines must be an array")
        if not isinstance(raw_local, Mapping):
            raise ValueError("local_limits must be an object")
        stored_programs = tuple(
            ExternalProgram(
                **{
                    **dict(item),
                    "launcher": _program_launcher(dict(item).get("launcher")),
                }
            )
            for item in raw_programs
        )
        stored_by_key = {program.key: program for program in stored_programs}
        resolved_programs: list[ExternalProgram] = []
        for detected in detect_external_programs():
            stored = stored_by_key.pop(detected.key, None)
            variable = PROGRAM_ENVIRONMENT.get(detected.key, "")
            # A process-local executable declaration is an explicit runtime
            # override.  Persisted capability profiles remain the fallback,
            # but must not hide a path supplied for this Keymaker session.
            explicitly_configured = bool(variable and os.environ.get(variable, "").strip())
            resolved_programs.append(
                detected if explicitly_configured and detected.enabled else (stored or detected)
            )
        programs = tuple(resolved_programs) + tuple(stored_by_key.values())
        machines: list[RemoteMachine] = []
        for raw in raw_machines:
            item = dict(raw)
            limits = item.get("limits", {})
            if not isinstance(limits, Mapping):
                raise ValueError("remote machine limits must be an object")
            item["limits"] = MachineLimits(**dict(limits))
            item["engines"] = tuple(item.get("engines", ()))
            raw_launchers = item.get("engine_launchers", {})
            if not isinstance(raw_launchers, Mapping):
                raise ValueError("remote engine_launchers must be an object")
            item["engine_launchers"] = {
                str(key): _program_launcher(value)
                for key, value in raw_launchers.items()
            }
            for key in ("remote_root", "qm_root", "runtime_venv", "runtime_config_dir"):
                if key in item:
                    item[key] = _migrate_legacy_deployment_path(str(item[key]))
            machines.append(RemoteMachine(**item))
        return cls(
            matrix_root=_migrate_legacy_deployment_path(str(data["matrix_root"])),
            projects_root=str(data["projects_root"]),
            programs=programs,
            local_limits=MachineLimits(**dict(raw_local)),
            remote_machines=tuple(machines),
            schema=str(data.get("schema", "")),
            created_utc=str(data.get("created_utc", "")),
            updated_utc=str(data.get("updated_utc", "")),
        )

    def program(self, key: str) -> ExternalProgram | None:
        normalized = key.strip().casefold()
        return next((item for item in self.programs if item.key == normalized), None)

    def remote_machine(
        self,
        identifier: str = "",
        *,
        engine: str = "",
    ) -> RemoteMachine:
        """Resolve one enabled remote by logical name or its configured SSH alias."""

        normalized_identifier = str(identifier).strip().casefold()
        normalized_engine = str(engine).strip().casefold()
        normalized_engine = {"gdv32": "gdv"}.get(normalized_engine, normalized_engine)
        candidates = tuple(machine for machine in self.remote_machines if machine.enabled)
        if normalized_identifier:
            candidates = tuple(
                machine
                for machine in candidates
                if normalized_identifier
                in {machine.name.strip().casefold(), machine.host.strip().casefold()}
            )
            if not candidates:
                raise ValueError(
                    f"remote machine {identifier!r} is not an enabled logical name or SSH alias "
                    "in the MATRIX environment"
                )
        if normalized_engine:
            candidates = tuple(
                machine for machine in candidates if normalized_engine in machine.engines
            )
            if not candidates:
                qualifier = f" {identifier!r}" if normalized_identifier else ""
                raise ValueError(
                    f"no enabled MATRIX remote{qualifier} provides engine {engine!r}"
                )
        if not candidates:
            raise ValueError("the MATRIX environment has no enabled remote machines")
        return candidates[0]


def resolve_program_path(value: str) -> str:
    text = str(value).strip()
    if not text:
        return ""
    expanded = Path(text).expanduser()
    if expanded.exists():
        return str(expanded.resolve())
    return shutil.which(text) or ""


def detect_external_programs() -> tuple[ExternalProgram, ...]:
    programs: list[ExternalProgram] = []
    for key, label, category, candidates in PROGRAM_DEFINITIONS:
        detected = ""
        variable = PROGRAM_ENVIRONMENT.get(key, "")
        configured = os.environ.get(variable, "").strip() if variable else ""
        if configured:
            detected = resolve_program_path(configured)
        if sys.platform == "darwin" and key in _MACOS_APPLICATIONS:
            application = _MACOS_APPLICATIONS[key]
            if not detected and application.exists():
                detected = str(application)
        if not detected:
            if key in {"pyscf", "pyfock"} and importlib.util.find_spec(key) is not None:
                detected = sys.executable
        if not detected:
            for candidate_path in _COMMON_PROGRAM_PATHS.get(key, ()):
                if candidate_path.is_file():
                    detected = str(candidate_path.resolve())
                    break
        if not detected:
            for candidate in candidates:
                detected = shutil.which(candidate) or ""
                if detected:
                    break
        programs.append(ExternalProgram(key, label, category, detected, bool(detected)))
    return tuple(programs)


def detect_local_limits() -> MachineLimits:
    processors = (
        _positive_int_environment("MATRIX_LOCAL_PROCESSORS") or os.cpu_count() or 1
    )
    memory_bytes = 0
    if sys.platform == "darwin":
        try:
            output = subprocess.check_output(("sysctl", "-n", "hw.memsize"), text=True)
            memory_bytes = int(output.strip())
        except (OSError, subprocess.SubprocessError, ValueError):
            memory_bytes = 0
    elif Path("/proc/meminfo").is_file():
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                memory_bytes = int(line.split()[1]) * 1024
                break
    configured_memory = _positive_float_environment("MATRIX_LOCAL_MEMORY_GB")
    memory_gb = configured_memory or (
        round(memory_bytes / (1024**3), 1) if memory_bytes else 1.0
    )
    gpu_count = _nonnegative_int_environment("MATRIX_LOCAL_GPU_COUNT")
    neural_engine_count = _nonnegative_int_environment(
        "MATRIX_LOCAL_NEURAL_ENGINE_COUNT"
    )
    max_jobs = _positive_int_environment("MATRIX_LOCAL_MAX_JOBS")
    return MachineLimits(
        processors=processors,
        memory_gb=max(1.0, memory_gb),
        gpu_count=_detect_gpu_count() if gpu_count is None else gpu_count,
        neural_engine_count=(
            _detect_apple_neural_engine_count()
            if neural_engine_count is None
            else neural_engine_count
        ),
        max_concurrent_jobs=max_jobs or max(1, min(4, processors // 2)),
    )


def _positive_int_environment(name: str) -> int | None:
    text = os.environ.get(name, "").strip()
    if not text:
        return None
    try:
        value = int(text)
    except ValueError:
        return None
    return value if value > 0 else None


def _nonnegative_int_environment(name: str) -> int | None:
    text = os.environ.get(name, "").strip()
    if not text:
        return None
    try:
        value = int(text)
    except ValueError:
        return None
    return value if value >= 0 else None


def _positive_float_environment(name: str) -> float | None:
    text = os.environ.get(name, "").strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if value > 0 else None


def _detect_gpu_count() -> int:
    if sys.platform.startswith("linux"):
        executable = shutil.which("nvidia-smi")
        if executable:
            try:
                output = subprocess.check_output(
                    (executable, "-L"), text=True, timeout=5
                )
                count = sum(1 for line in output.splitlines() if line.strip())
                if count:
                    return count
            except (OSError, subprocess.SubprocessError):
                pass
        return 1 if Path("/dev/kfd").exists() else 0
    if sys.platform == "darwin":
        try:
            output = subprocess.check_output(
                ("system_profiler", "SPDisplaysDataType", "-json"),
                text=True,
                timeout=5,
            )
            payload = json.loads(output)
            displays = payload.get("SPDisplaysDataType", ())
            return len(displays) if isinstance(displays, list) else 0
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
            return 0
    return 0


def _detect_apple_neural_engine_count() -> int:
    """Return the number of Apple Neural Engine devices, not their core count."""

    if sys.platform != "darwin" or platform.machine().casefold() not in {"arm64", "aarch64"}:
        return 0
    try:
        output = subprocess.check_output(("sysctl", "-n", "hw.optional.arm64"), text=True)
    except (OSError, subprocess.SubprocessError):
        return 0
    return 1 if output.strip() == "1" else 0


def detect_runtime_machine(
    *,
    node_name: str | None = None,
    hostname: str | None = None,
    operating_system: str | None = None,
    operating_system_release: str | None = None,
    architecture: str | None = None,
    processor: str | None = None,
    limits: MachineLimits | None = None,
) -> RuntimeMachine:
    """Inspect this node without assigning host-specific scientific behavior."""

    observed_hostname = str(hostname or platform.node() or "localhost").strip()
    observed_node = str(
        node_name or os.environ.get("MATRIX_NODE_NAME") or observed_hostname
    ).strip()
    return RuntimeMachine(
        node_name=observed_node,
        hostname=observed_hostname,
        operating_system=normalize_operating_system(
            operating_system or platform.system()
        ),
        operating_system_release=str(
            operating_system_release or platform.release()
        ).strip(),
        architecture=normalize_architecture(architecture or platform.machine()),
        processor=str(processor or platform.processor()).strip(),
        python_version=platform.python_version(),
        python_abi=sys.implementation.cache_tag or "",
        limits=limits or detect_local_limits(),
    )


def refresh_runtime_environment(
    environment: MatrixEnvironment,
    *,
    machine: RuntimeMachine | None = None,
    detected_programs: tuple[ExternalProgram, ...] | None = None,
) -> MatrixEnvironment:
    """Combine persistent policy with facts observed on the current node."""

    observed = machine or detect_runtime_machine()
    detected = {
        item.key: item for item in (detected_programs or detect_external_programs())
    }
    programs: list[ExternalProgram] = []
    for stored in environment.programs:
        current = detected.pop(stored.key, None)
        explicit = resolve_program_path(stored.executable)
        executable = explicit or (current.executable if current is not None else "")
        programs.append(
            ExternalProgram(
                stored.key,
                stored.label,
                stored.category,
                executable,
                stored.enabled and bool(executable),
                stored.launcher or (None if current is None else current.launcher),
            )
        )
    programs.extend(detected.values())
    limits = MachineLimits(
        processors=observed.limits.processors,
        memory_gb=observed.limits.memory_gb,
        gpu_count=observed.limits.gpu_count,
        neural_engine_count=observed.limits.neural_engine_count,
        max_concurrent_jobs=min(
            observed.limits.max_concurrent_jobs,
            environment.local_limits.max_concurrent_jobs,
        ),
    )
    return MatrixEnvironment(
        matrix_root=environment.matrix_root,
        projects_root=environment.projects_root,
        programs=tuple(programs),
        local_limits=limits,
        remote_machines=environment.remote_machines,
        schema=environment.schema,
        created_utc=environment.created_utc,
        updated_utc=environment.updated_utc,
    )


def default_environment(
    *, matrix_root: Path | str | None = None, projects_root: Path | str | None = None
) -> MatrixEnvironment:
    root = Path(matrix_root or os.environ.get("MATRIX_HOME") or Path.cwd()).expanduser().resolve()
    projects = Path(
        projects_root
        or os.environ.get("MATRIX_PROJECTS_DIR")
        or (Path.home() / "Documents" / "MATRIX-projects")
    ).expanduser().resolve()
    return MatrixEnvironment(
        matrix_root=str(root),
        projects_root=str(projects),
        programs=detect_external_programs(),
        local_limits=detect_local_limits(),
    )


def load_environment(
    path: Path | str = DEFAULT_CONFIG_PATH, *, missing_ok: bool = False
) -> MatrixEnvironment:
    target = Path(path).expanduser()
    if missing_ok and not target.is_file():
        return default_environment()
    return MatrixEnvironment.from_dict(json.loads(target.read_text(encoding="utf-8")))


def load_runtime_environment(
    path: Path | str = DEFAULT_CONFIG_PATH,
    *,
    missing_ok: bool = False,
) -> MatrixEnvironment:
    """Load persistent policy and refresh all local machine facts."""

    return refresh_runtime_environment(load_environment(path, missing_ok=missing_ok))


def configured_remote_machine(
    identifier: str = "",
    *,
    engine: str = "",
    path: Path | str | None = None,
) -> RemoteMachine:
    """Resolve a remote solely through the canonical MATRIX environment profile."""

    config_path = path or os.environ.get("MATRIX_CONFIG") or DEFAULT_CONFIG_PATH
    requested = str(identifier).strip() or os.environ.get("MATRIX_REMOTE_MACHINE", "").strip()
    if not requested:
        requested = os.environ.get("MATRIX_REMOTE_HOST", "").strip()
    return load_environment(config_path).remote_machine(requested, engine=engine)


def runtime_machine_schema_path() -> Path:
    return Path(__file__).resolve().parent / "schemas" / "runtime-machine-v1.schema.json"


def write_environment(
    environment: MatrixEnvironment, path: Path | str = DEFAULT_CONFIG_PATH
) -> Path:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    data = environment.to_dict()
    data["updated_utc"] = _utc_now()
    atomic_json_write(target, data)
    return target


def environment_exports(
    environment: MatrixEnvironment, *, config_path: Path | str = DEFAULT_CONFIG_PATH
) -> dict[str, str]:
    exports = {
        "MATRIX_CONFIG": str(Path(config_path).expanduser().resolve()),
        "MATRIX_HOME": str(Path(environment.matrix_root).expanduser().resolve()),
        "MATRIX_PROJECTS_DIR": str(Path(environment.projects_root).expanduser().resolve()),
        "MATRIX_NODE_NAME": detect_runtime_machine(limits=environment.local_limits).node_name,
        "MATRIX_OPERATING_SYSTEM": normalize_operating_system(platform.system()),
        "MATRIX_ARCHITECTURE": normalize_architecture(platform.machine()),
        "MATRIX_LOCAL_PROCESSORS": str(environment.local_limits.processors),
        "MATRIX_LOCAL_MEMORY_GB": str(environment.local_limits.memory_gb),
        "MATRIX_LOCAL_GPU_COUNT": str(environment.local_limits.gpu_count),
        "MATRIX_LOCAL_NEURAL_ENGINE_COUNT": str(
            environment.local_limits.neural_engine_count
        ),
        "MATRIX_LOCAL_MAX_JOBS": str(environment.local_limits.max_concurrent_jobs),
    }
    for program in environment.programs:
        variable = PROGRAM_ENVIRONMENT.get(program.key)
        resolved = resolve_program_path(program.executable) if program.enabled else ""
        if variable and resolved:
            exports[variable] = resolved
    gdv = exports.get("MATRIX_GDV_EXE")
    g16 = exports.get("MATRIX_G16_EXE")
    if gdv or g16:
        exports["ORACLE_GAUSSIAN_EXE"] = gdv or g16 or ""
        exports["MATRIX_GAUSSIAN_BACKEND"] = "gdv" if gdv else "g16"
    enabled_remotes = [machine for machine in environment.remote_machines if machine.enabled]
    if enabled_remotes:
        primary = enabled_remotes[0]
        exports.update(
            {
                "MATRIX_REMOTE_MACHINE": primary.name,
                "MATRIX_REMOTE_HOST": primary.host,
                "MATRIX_REMOTE_ROOT": primary.remote_root,
                "MATRIX_REMOTE_QM_ROOT": primary.qm_root,
                "MATRIX_REMOTE_VENV": primary.runtime_venv,
                "MATRIX_REMOTE_CONFIG_DIR": primary.runtime_config_dir,
                "MATRIX_REMOTE_EXPECTED_OS": primary.expected_operating_system,
                "MATRIX_REMOTE_EXPECTED_ARCHITECTURE": primary.expected_architecture,
                "MATRIX_REMOTE_EXPECTED_PYTHON_ABI": primary.expected_python_abi,
                "MATRIX_REMOTE_PROCESSORS": str(primary.limits.processors),
                "MATRIX_REMOTE_MEMORY_GB": str(primary.limits.memory_gb),
                "MATRIX_REMOTE_GPU_COUNT": str(primary.limits.gpu_count),
                "MATRIX_REMOTE_NEURAL_ENGINE_COUNT": str(
                    primary.limits.neural_engine_count
                ),
                "MATRIX_REMOTE_MAX_JOBS": str(primary.limits.max_concurrent_jobs),
                "MATRIX_REMOTE_REQUIRES_VPN": "1" if primary.requires_vpn else "0",
            }
        )
    return exports
