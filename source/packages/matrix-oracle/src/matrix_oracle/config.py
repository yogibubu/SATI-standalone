"""Portable configuration for the public ORACLE interface."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class OraclePaths:
    """User-selected directories; no checkout-specific paths are assumed."""

    data_dir: Path | None = None
    cache_dir: Path | None = None
    work_dir: Path | None = None


@dataclass(frozen=True)
class OracleSymmetryConfig:
    distance_angstrom: float = 1.0e-3
    inertia_relative: float = 1.0e-3
    max_rotation_order: int = 6


@dataclass(frozen=True)
class OracleConfig:
    paths: OraclePaths = OraclePaths()
    symmetry: OracleSymmetryConfig = OracleSymmetryConfig()
    source: Path | None = None


def load_oracle_config(path: Path | None = None) -> OracleConfig:
    """Load TOML configuration with environment variables taking precedence."""
    selected = _config_path(path)
    payload: dict[str, object] = {}
    if selected is not None:
        if not selected.is_file():
            raise FileNotFoundError(f"ORACLE configuration does not exist: {selected}")
        payload = tomllib.loads(selected.read_text(encoding="utf-8"))

    path_values = _mapping(payload.get("paths"))
    symmetry_values = _mapping(payload.get("symmetry"))
    paths = OraclePaths(
        data_dir=_configured_path("ORACLE_DATA_DIR", path_values.get("data_dir")),
        cache_dir=_configured_path("ORACLE_CACHE_DIR", path_values.get("cache_dir")),
        work_dir=_configured_path("ORACLE_WORK_DIR", path_values.get("work_dir")),
    )
    symmetry = OracleSymmetryConfig(
        distance_angstrom=_positive_float(
            "symmetry.distance_angstrom",
            symmetry_values.get("distance_angstrom", 1.0e-3),
        ),
        inertia_relative=_positive_float(
            "symmetry.inertia_relative",
            symmetry_values.get("inertia_relative", 1.0e-3),
        ),
        max_rotation_order=_positive_int(
            "symmetry.max_rotation_order",
            symmetry_values.get("max_rotation_order", 6),
        ),
    )
    return OracleConfig(paths=paths, symmetry=symmetry, source=selected)


def oracle_config_template() -> str:
    return """# ORACLE paths are selected by the installer or user.
[paths]
data_dir = "/path/to/oracle-data"
cache_dir = "/path/to/oracle-cache"
work_dir = "/path/to/oracle-work"

[symmetry]
distance_angstrom = 0.001
inertia_relative = 0.001
max_rotation_order = 6
"""


def write_oracle_config_template(path: Path, *, overwrite: bool = False) -> Path:
    target = Path(path).expanduser()
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing configuration: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(oracle_config_template(), encoding="utf-8")
    return target


def _config_path(path: Path | None) -> Path | None:
    if path is not None:
        return Path(path).expanduser().resolve()
    value = os.environ.get("ORACLE_CONFIG", "").strip()
    return Path(value).expanduser().resolve() if value else None


def _configured_path(environment_name: str, value: object) -> Path | None:
    selected = os.environ.get(environment_name, "").strip() or str(value or "").strip()
    return Path(selected).expanduser().resolve() if selected else None


def _mapping(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, dict) else {}


def _positive_float(name: str, value: object) -> float:
    converted = float(value)
    if converted <= 0.0:
        raise ValueError(f"{name} must be positive")
    return converted


def _positive_int(name: str, value: object) -> int:
    converted = int(value)
    if converted < 1:
        raise ValueError(f"{name} must be at least one")
    return converted
