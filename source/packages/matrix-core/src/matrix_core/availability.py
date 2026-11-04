"""Single source of truth for executable, machine and resource combinations."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .environment import MachineLimits, MatrixEnvironment, ProgramLauncher, resolve_program_path


AVAILABILITY_SCHEMA = "matrix.availability.v1"
BACKEND_ALIASES: dict[str, tuple[str, ...]] = {
    "gaussian": ("gdv", "g16"),
    # G16.C02 is deployed through the stable matrix-gdv launcher on hosts
    # that do not expose a separate ``g16`` executable.  Keep ``g16`` as the
    # public protocol/backend identity while retaining the qualified launcher
    # as its provider.  A native g16 row always takes precedence.
    "g16": ("gdv",),
}
RESIDENT_BACKENDS = ("zaff", "matrix_gfn2", "pyscf", "pyfock")


@dataclass(frozen=True)
class ExecutionCombination:
    backend: str
    provider: str
    machine: str
    kind: str
    limits: MachineLimits
    executable: str = ""
    host: str = ""
    schema: str = AVAILABILITY_SCHEMA
    launcher: ProgramLauncher | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"local", "remote", "resident"}:
            raise ValueError(f"unsupported execution-combination kind: {self.kind}")

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.machine}:{self.backend}"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AvailabilityInventory:
    combinations: tuple[ExecutionCombination, ...]
    schema: str = AVAILABILITY_SCHEMA

    @property
    def backends(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.backend for item in self.combinations))

    def combinations_for(self, backend: str) -> tuple[ExecutionCombination, ...]:
        key = str(backend).strip().casefold()
        return tuple(item for item in self.combinations if item.backend == key)

    def remote_machines_for(self, backend: str = "") -> tuple[str, ...]:
        key = str(backend).strip().casefold()
        return tuple(
            dict.fromkeys(
                item.machine
                for item in self.combinations
                if item.kind == "remote" and (not key or item.backend == key)
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "combinations": [item.to_dict() for item in self.combinations],
        }


def discover_execution_combinations(
    environment: MatrixEnvironment,
    *,
    include_resident: bool = True,
) -> AvailabilityInventory:
    """Return only executable combinations admitted by the environment profile.

    Local programs are re-resolved at discovery time.  Remote combinations are
    admitted only when both the machine and engine are explicitly enabled in
    the persistent capability profile.  Every row carries the applicable
    resource limits, avoiding separate program/machine/resource filtering.
    """

    combinations: list[ExecutionCombination] = []
    for program in environment.programs:
        executable = resolve_program_path(program.executable) if program.enabled else ""
        if program.category != "qm" or not executable:
            continue
        combinations.append(
            ExecutionCombination(
                backend=program.key,
                provider=program.key,
                machine="local",
                kind="local",
                executable=executable,
                launcher=program.launcher,
                limits=environment.local_limits,
            )
        )
    for machine in environment.remote_machines:
        if not machine.enabled:
            continue
        for engine in machine.engines:
            launcher = machine.engine_launchers.get(engine)
            combinations.append(
                ExecutionCombination(
                    backend=engine,
                    provider=engine,
                    machine=machine.name,
                    kind="remote",
                    host=machine.host,
                    executable="" if launcher is None else launcher.executable,
                    launcher=launcher,
                    limits=machine.limits,
                )
            )

    native = tuple(combinations)
    for alias, providers in BACKEND_ALIASES.items():
        for item in native:
            if item.provider not in providers:
                continue
            if any(
                existing.backend == alias
                and existing.machine == item.machine
                and existing.kind == item.kind
                for existing in combinations
            ):
                continue
            combinations.append(
                ExecutionCombination(
                    backend=alias,
                    provider=item.provider,
                    machine=item.machine,
                    kind=item.kind,
                    limits=item.limits,
                    executable=item.executable,
                    launcher=item.launcher,
                    host=item.host,
                )
            )
    if include_resident:
        for backend in RESIDENT_BACKENDS:
            combinations.append(
                ExecutionCombination(
                    backend=backend,
                    provider=backend,
                    machine="local",
                    kind="resident",
                    limits=environment.local_limits,
                )
            )
    return AvailabilityInventory(tuple(combinations))
