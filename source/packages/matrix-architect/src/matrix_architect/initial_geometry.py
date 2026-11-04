"""ARCHITECT provider boundary for mandatory initial-geometry refinement."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
import shlex
import sys


ARCHITECT_INITIAL_GEOMETRY_PROVIDER_SCHEMA = (
    "matrix.architect.initial_geometry_provider.v1"
)


@dataclass(frozen=True)
class ArchitectInitialGeometryProvider:
    provider: str
    runtime_backend: str
    method: str
    route: str
    executable: str | None
    schema: str = ARCHITECT_INITIAL_GEOMETRY_PROVIDER_SCHEMA
    owner: str = "ARCHITECT"

    def __post_init__(self) -> None:
        if self.schema != ARCHITECT_INITIAL_GEOMETRY_PROVIDER_SCHEMA:
            raise ValueError("unsupported ARCHITECT initial-geometry provider schema")
        if self.owner != "ARCHITECT":
            raise ValueError("initial-geometry refinement provider must be owned by ARCHITECT")

    def to_dict(self) -> dict[str, str | None]:
        """Serialize the public provider contract for process-boundary clients."""

        return asdict(self)


def resolve_initial_geometry_provider(
    provider: str,
    *,
    xtb_executable: str | None = None,
) -> ArchitectInitialGeometryProvider:
    """Resolve the frozen low-layer provider without implementing a second engine."""

    normalized = str(provider).strip().upper()
    if normalized == "GFN-FF":
        return ArchitectInitialGeometryProvider(
            provider="GFN-FF",
            runtime_backend="xtb",
            method="GFN-FF",
            route="--gfnff",
            executable=xtb_executable,
        )
    if normalized == "GFN-XTB":
        return ArchitectInitialGeometryProvider(
            provider="GFN-xTB",
            runtime_backend="xtb",
            method="GFN2-xTB",
            route="--gfn 2",
            executable=xtb_executable,
        )
    if normalized == "UFF":
        return ArchitectInitialGeometryProvider(
            provider="UFF",
            runtime_backend="architect_uff",
            method="Universal Force Field",
            route="resident",
            executable=None,
        )
    raise ValueError(f"unsupported ARCHITECT initial-geometry provider: {provider}")


def uff_point_engine_command(
    model: Path | str,
    *,
    python_executable: str | None = None,
) -> str:
    """Return the one canonical LINK command for the resident UFF provider."""

    executable = str(python_executable or sys.executable)
    return " ".join(
        (
            shlex.quote(executable),
            "-m",
            "matrix_architect.cli",
            "uff-point",
            shlex.quote(str(Path(model).expanduser().resolve())),
            "{xyz}",
            "--result",
            "{result}",
            "--point-index",
            "{index}",
            "--properties",
            "'{properties}'",
        )
    )


__all__ = [
    "ARCHITECT_INITIAL_GEOMETRY_PROVIDER_SCHEMA",
    "ArchitectInitialGeometryProvider",
    "resolve_initial_geometry_provider",
    "uff_point_engine_command",
]
