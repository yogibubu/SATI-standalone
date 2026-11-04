from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .sectioned_xyz import is_section_header_line


ORACLE_MANIFEST_SCHEMA = "oracle.run.v1"
MATRIX_MANIFEST_FRAMEWORK = "MATRIX"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(path: Path, data: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def file_checksums(paths: Mapping[str, Path]) -> dict[str, str]:
    return {name: sha256_file(path) for name, path in sorted(paths.items()) if Path(path).is_file()}


@dataclass(frozen=True)
class RunManifest:
    workflow: str
    status: str
    run_dir: Path
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    input_sha256: dict[str, str] = field(default_factory=dict)
    output_sha256: dict[str, str] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    backend: dict[str, Any] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)
    schema_version: str = ORACLE_MANIFEST_SCHEMA
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    framework: str = MATRIX_MANIFEST_FRAMEWORK
    matrix_version: str = field(default_factory=lambda: matrix_version())
    python_version: str = field(default_factory=platform.python_version)
    python_executable: str = field(default_factory=lambda: sys.executable)
    python_abi: str = field(default_factory=lambda: sys.implementation.cache_tag or "")
    architecture: str = field(default_factory=platform.machine)
    operating_system: str = field(default_factory=platform.system)
    environment_prefix: str = field(default_factory=lambda: sys.prefix)
    command: tuple[str, ...] = field(default_factory=lambda: tuple(sys.argv))
    xyzin_sections: dict[str, tuple[str, ...]] = field(default_factory=dict)
    external_backends: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["run_dir"] = str(self.run_dir)
        data["command"] = list(self.command)
        data["xyzin_sections"] = {
            name: list(sections) for name, sections in self.xyzin_sections.items()
        }
        return data

    def write(self, path: Path | None = None) -> Path:
        target = Path(path) if path is not None else self.run_dir / "manifest.json"
        return write_manifest(target, self.to_dict())


def build_run_manifest(
    *,
    workflow: str,
    status: str,
    run_dir: Path,
    inputs: Mapping[str, Path] | None = None,
    outputs: Mapping[str, Path] | None = None,
    parameters: Mapping[str, Any] | None = None,
    backend: Mapping[str, Any] | None = None,
    messages: list[str] | None = None,
    command: Sequence[str] | str | None = None,
    xyzin_sections: Mapping[str, Sequence[str]] | None = None,
    external_backends: Mapping[str, Any] | None = None,
) -> RunManifest:
    input_paths = dict(inputs or {})
    output_paths = dict(outputs or {})
    backend_data = dict(backend or {})
    detected_sections = _detected_xyzin_sections(input_paths | output_paths)
    if xyzin_sections is not None:
        detected_sections.update(
            {
                name: tuple(str(section) for section in sections)
                for name, sections in xyzin_sections.items()
            }
        )
    return RunManifest(
        workflow=workflow,
        status=status,
        run_dir=Path(run_dir),
        inputs={name: str(path) for name, path in sorted(input_paths.items())},
        outputs={name: str(path) for name, path in sorted(output_paths.items())},
        input_sha256=file_checksums(input_paths),
        output_sha256=file_checksums(output_paths),
        parameters=dict(parameters or {}),
        backend=backend_data,
        messages=list(messages or []),
        command=_normalize_command(command),
        xyzin_sections=dict(sorted(detected_sections.items())),
        external_backends=dict(external_backends or _external_backend_summary(backend_data)),
    )


def matrix_version() -> str:
    from .version import MATRIX_VERSION

    return MATRIX_VERSION


def _normalize_command(command: Sequence[str] | str | None) -> tuple[str, ...]:
    if command is None:
        return tuple(str(item) for item in sys.argv)
    if isinstance(command, str):
        return (command,)
    return tuple(str(item) for item in command)


def _detected_xyzin_sections(paths: Mapping[str, Path]) -> dict[str, tuple[str, ...]]:
    sections: dict[str, tuple[str, ...]] = {}
    for name, path in sorted(paths.items()):
        target = Path(path)
        if target.suffix.lower() != ".xyzin" or not target.is_file():
            continue
        detected = _section_names(target)
        if detected:
            sections[name] = detected
    return sections


def _section_names(path: Path) -> tuple[str, ...]:
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return ()
    names: list[str] = []
    for line in lines:
        text = line.strip()
        if is_section_header_line(text):
            names.append(text[1:].strip().upper())
    return tuple(names)


def _external_backend_summary(backend: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "name",
        "solver",
        "executable",
        "executable_sha256",
        "engine_command",
        "external_protocol",
        "python_executable",
        "fortran77_source",
        "vci_method",
    )
    return {key: backend[key] for key in keys if key in backend}
