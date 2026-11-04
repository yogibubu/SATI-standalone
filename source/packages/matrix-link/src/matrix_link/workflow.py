from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from matrix_core import (
    build_run_manifest,
    key_value_section_lines,
    parse_key_value_section,
    read_sectioned_lines,
    replace_section,
    section_content,
)


ORACLE_XYZ_LINK_SCHEMA = "oracle.xyz.link.v1"
LINK_LONG_NAME = "Level-aware Internal-coordinate Network for Kinetics and optimization"


@dataclass(frozen=True)
class LinkSection:
    """Persistent LINK optimization/PES request stored in a MATRIX xyzin container."""

    source_kind: str = "xyzin"
    source_path: Path | None = None
    run_dir: Path | None = None
    manifest_path: Path | None = None
    engine_command: str = ""
    external_protocol: str = "xyz-energy-gradient-v1"
    coordinate_model: str = "gic"
    active_space: str = "total_symmetric"
    energy_unit: str = "hartree"
    gradient_unit: str = "hartree/bohr"
    max_steps: int = 50
    trust_radius: float = 0.2
    gradient_tolerance: float = 1.0e-5
    step_tolerance: float = 1.0e-5
    energy_tolerance: float = 1.0e-8
    trajectory_path: Path | None = None
    final_geometry_path: Path | None = None
    energy_gradient_log_path: Path | None = None
    outputs: Mapping[str, Path] | None = None
    status: str = "prepared"
    schema: str = ORACLE_XYZ_LINK_SCHEMA

    def __post_init__(self) -> None:
        for attr in (
            "source_path",
            "run_dir",
            "manifest_path",
            "trajectory_path",
            "final_geometry_path",
            "energy_gradient_log_path",
        ):
            value = getattr(self, attr)
            if value is not None:
                object.__setattr__(self, attr, Path(value))
        outputs = {
            _normalize_output_key(name): Path(path)
            for name, path in dict(self.outputs or {}).items()
            if path is not None
        }
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "max_steps", int(self.max_steps))
        object.__setattr__(self, "trust_radius", float(self.trust_radius))
        object.__setattr__(self, "gradient_tolerance", float(self.gradient_tolerance))
        object.__setattr__(self, "step_tolerance", float(self.step_tolerance))
        object.__setattr__(self, "energy_tolerance", float(self.energy_tolerance))

    def with_manifest(self, manifest_path: Path | None) -> "LinkSection":
        return LinkSection(
            source_kind=self.source_kind,
            source_path=self.source_path,
            run_dir=self.run_dir,
            manifest_path=manifest_path,
            engine_command=self.engine_command,
            external_protocol=self.external_protocol,
            coordinate_model=self.coordinate_model,
            active_space=self.active_space,
            energy_unit=self.energy_unit,
            gradient_unit=self.gradient_unit,
            max_steps=self.max_steps,
            trust_radius=self.trust_radius,
            gradient_tolerance=self.gradient_tolerance,
            step_tolerance=self.step_tolerance,
            energy_tolerance=self.energy_tolerance,
            trajectory_path=self.trajectory_path,
            final_geometry_path=self.final_geometry_path,
            energy_gradient_log_path=self.energy_gradient_log_path,
            outputs=self.outputs,
            status=self.status,
            schema=self.schema,
        )


def link_section_from_request(
    *,
    xyzin_path: Path | str,
    run_dir: Path | str,
    engine_command: str,
    coordinate_model: str = "gic",
    active_space: str = "total_symmetric",
    max_steps: int = 50,
    trust_radius: float = 0.2,
    gradient_tolerance: float = 1.0e-5,
    step_tolerance: float = 1.0e-5,
    energy_tolerance: float = 1.0e-8,
    energy_unit: str = "hartree",
    gradient_unit: str = "hartree/bohr",
    external_protocol: str = "xyz-energy-gradient-v1",
    status: str = "prepared",
) -> LinkSection:
    target_run_dir = Path(run_dir)
    if not engine_command.strip():
        raise ValueError("LINK needs an external engine command")
    if coordinate_model not in {"gic", "cartesian"}:
        raise ValueError(f"unsupported LINK coordinate model: {coordinate_model}")
    return LinkSection(
        source_path=Path(xyzin_path),
        run_dir=target_run_dir,
        manifest_path=target_run_dir / "link_manifest.json",
        engine_command=engine_command.strip(),
        external_protocol=external_protocol,
        coordinate_model=coordinate_model,
        active_space=active_space,
        energy_unit=energy_unit,
        gradient_unit=gradient_unit,
        max_steps=max_steps,
        trust_radius=trust_radius,
        gradient_tolerance=gradient_tolerance,
        step_tolerance=step_tolerance,
        energy_tolerance=energy_tolerance,
        trajectory_path=target_run_dir / "link_trajectory.xyz",
        final_geometry_path=target_run_dir / "link_final.xyz",
        energy_gradient_log_path=target_run_dir / "link_energy_gradient.jsonl",
        status=status,
    )


def prepare_link_section(
    xyzin_path: Path | str,
    *,
    run_dir: Path | str,
    engine_command: str,
    coordinate_model: str = "gic",
    active_space: str = "total_symmetric",
    max_steps: int = 50,
    trust_radius: float = 0.2,
    gradient_tolerance: float = 1.0e-5,
    step_tolerance: float = 1.0e-5,
    energy_tolerance: float = 1.0e-8,
    energy_unit: str = "hartree",
    gradient_unit: str = "hartree/bohr",
    external_protocol: str = "xyz-energy-gradient-v1",
) -> LinkSection:
    """Write a prepared LINK request in the compatibility #LINK section.

    This intentionally does not execute an optimization yet. The section is the
    autonomous input contract consumed by the LINK runner. The section name is
    retained only for compatibility during the package split.
    """
    section = link_section_from_request(
        xyzin_path=xyzin_path,
        run_dir=run_dir,
        engine_command=engine_command,
        coordinate_model=coordinate_model,
        active_space=active_space,
        max_steps=max_steps,
        trust_radius=trust_radius,
        gradient_tolerance=gradient_tolerance,
        step_tolerance=step_tolerance,
        energy_tolerance=energy_tolerance,
        energy_unit=energy_unit,
        gradient_unit=gradient_unit,
        external_protocol=external_protocol,
    )
    if section.run_dir is not None:
        section.run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = write_link_manifest(section)
    section = section.with_manifest(manifest_path)
    write_link_section(xyzin_path, section)
    return section


def write_link_manifest(section: LinkSection) -> Path:
    if section.run_dir is None:
        raise ValueError("LINK manifest needs a run directory")
    outputs = _default_outputs(section)
    manifest = build_run_manifest(
        workflow="link",
        status=section.status,
        run_dir=section.run_dir,
        inputs={"xyzin": section.source_path} if section.source_path is not None else {},
        outputs=outputs,
        parameters={
            "coordinate_model": section.coordinate_model,
            "active_space": section.active_space,
            "max_steps": section.max_steps,
            "trust_radius": section.trust_radius,
            "gradient_tolerance": section.gradient_tolerance,
            "step_tolerance": section.step_tolerance,
            "energy_tolerance": section.energy_tolerance,
        },
        backend={
            "engine_command": section.engine_command,
            "external_protocol": section.external_protocol,
            "energy_unit": section.energy_unit,
            "gradient_unit": section.gradient_unit,
        },
        messages=[
            "LINK state prepared; use link optimize for the energy/gradient optimizer."
        ],
    )
    return manifest.write(section.run_dir / "link_manifest.json")


def link_section_lines(section: LinkSection) -> list[str]:
    values = {
        "SOURCE_KIND": section.source_kind,
        "SOURCE_PATH": section.source_path,
        "STATUS": section.status,
        "RUN_DIR": section.run_dir,
        "MANIFEST": section.manifest_path,
        "ENGINE_COMMAND": section.engine_command,
        "EXTERNAL_PROTOCOL": section.external_protocol,
        "COORDINATE_MODEL": section.coordinate_model,
        "ACTIVE_SPACE": section.active_space,
        "ENERGY_UNIT": section.energy_unit,
        "GRADIENT_UNIT": section.gradient_unit,
        "MAX_STEPS": section.max_steps,
        "TRUST_RADIUS": _format_float(section.trust_radius),
        "GRADIENT_TOLERANCE": _format_float(section.gradient_tolerance),
        "STEP_TOLERANCE": _format_float(section.step_tolerance),
        "ENERGY_TOLERANCE": _format_float(section.energy_tolerance),
        "TRAJECTORY": section.trajectory_path,
        "FINAL_GEOMETRY": section.final_geometry_path,
        "ENERGY_GRADIENT_LOG": section.energy_gradient_log_path,
    }
    values.update({f"OUTPUT_{name.upper()}": path for name, path in section.outputs.items()})
    return key_value_section_lines(
        ORACLE_XYZ_LINK_SCHEMA,
        values,
        key_order=(
            "SOURCE_KIND",
            "SOURCE_PATH",
            "STATUS",
            "RUN_DIR",
            "MANIFEST",
            "ENGINE_COMMAND",
            "EXTERNAL_PROTOCOL",
            "COORDINATE_MODEL",
            "ACTIVE_SPACE",
            "ENERGY_UNIT",
            "GRADIENT_UNIT",
            "MAX_STEPS",
            "TRUST_RADIUS",
            "GRADIENT_TOLERANCE",
            "STEP_TOLERANCE",
            "ENERGY_TOLERANCE",
            "TRAJECTORY",
            "FINAL_GEOMETRY",
            "ENERGY_GRADIENT_LOG",
        ),
    )


def parse_link_section(lines: list[str] | tuple[str, ...]) -> LinkSection:
    values = parse_key_value_section(lines)
    schema = values.get("SCHEMA", ORACLE_XYZ_LINK_SCHEMA)
    if schema != ORACLE_XYZ_LINK_SCHEMA:
        raise ValueError(f"unsupported LINK schema: {schema}")
    outputs = {
        _normalize_output_key(key[len("OUTPUT_") :]): Path(raw)
        for key, raw in values.items()
        if key.startswith("OUTPUT_") and raw.strip()
    }
    return LinkSection(
        source_kind=values.get("SOURCE_KIND", "xyzin"),
        source_path=_optional_path(values.get("SOURCE_PATH")),
        run_dir=_optional_path(values.get("RUN_DIR")),
        manifest_path=_optional_path(values.get("MANIFEST")),
        engine_command=values.get("ENGINE_COMMAND", ""),
        external_protocol=values.get("EXTERNAL_PROTOCOL", "xyz-energy-gradient-v1"),
        coordinate_model=values.get("COORDINATE_MODEL", "gic"),
        active_space=values.get("ACTIVE_SPACE", "total_symmetric"),
        energy_unit=values.get("ENERGY_UNIT", "hartree"),
        gradient_unit=values.get("GRADIENT_UNIT", "hartree/bohr"),
        max_steps=_int_value(values, "MAX_STEPS", 50),
        trust_radius=_float_value(values, "TRUST_RADIUS", 0.2),
        gradient_tolerance=_float_value(values, "GRADIENT_TOLERANCE", 1.0e-5),
        step_tolerance=_float_value(values, "STEP_TOLERANCE", 1.0e-5),
        energy_tolerance=_float_value(values, "ENERGY_TOLERANCE", 1.0e-8),
        trajectory_path=_optional_path(values.get("TRAJECTORY")),
        final_geometry_path=_optional_path(values.get("FINAL_GEOMETRY")),
        energy_gradient_log_path=_optional_path(values.get("ENERGY_GRADIENT_LOG")),
        outputs=outputs,
        status=values.get("STATUS", "prepared"),
        schema=schema,
    )


def read_link_section(path: Path | str) -> LinkSection:
    content = section_content(read_sectioned_lines(Path(path)), "LINK")
    if not content:
        raise ValueError("missing #LINK section")
    return parse_link_section(content)


def write_link_section(path: Path | str, section: LinkSection) -> None:
    replace_section(Path(path), "LINK", link_section_lines(section))


def link_section_summary_lines(section: LinkSection) -> list[str]:
    return [
        f"status: {section.status}",
        f"source: {section.source_kind}",
        f"source path: {_path_text(section.source_path)}",
        f"run dir: {_path_text(section.run_dir)}",
        f"manifest: {_path_text(section.manifest_path)}",
        f"engine command: {section.engine_command}",
        f"external protocol: {section.external_protocol}",
        f"coordinate model: {section.coordinate_model}",
        f"active space: {section.active_space}",
        f"energy unit: {section.energy_unit}",
        f"gradient unit: {section.gradient_unit}",
        f"max steps: {section.max_steps}",
        f"trust radius: {_format_float(section.trust_radius)}",
        f"gradient tolerance: {_format_float(section.gradient_tolerance)}",
        f"step tolerance: {_format_float(section.step_tolerance)}",
        f"energy tolerance: {_format_float(section.energy_tolerance)}",
        f"trajectory: {_path_text(section.trajectory_path)}",
        f"final geometry: {_path_text(section.final_geometry_path)}",
        f"energy/gradient log: {_path_text(section.energy_gradient_log_path)}",
    ]


def _default_outputs(section: LinkSection) -> dict[str, Path]:
    outputs = dict(section.outputs)
    if section.trajectory_path is not None:
        outputs.setdefault("trajectory", section.trajectory_path)
    if section.final_geometry_path is not None:
        outputs.setdefault("final_geometry", section.final_geometry_path)
    if section.energy_gradient_log_path is not None:
        outputs.setdefault("energy_gradient_log", section.energy_gradient_log_path)
    return outputs


def _optional_path(raw: str | None) -> Path | None:
    if raw is None or not raw.strip():
        return None
    return Path(raw)


def _int_value(values: Mapping[str, str], key: str, default: int) -> int:
    raw = values.get(key)
    if raw is None:
        return default
    return int(float(raw.replace("D", "E").replace("d", "e")))


def _float_value(values: Mapping[str, str], key: str, default: float) -> float:
    raw = values.get(key)
    if raw is None:
        return default
    return float(raw.replace("D", "E").replace("d", "e"))


def _format_float(value: float) -> str:
    return f"{float(value):.12g}"


def _path_text(path: Path | None) -> str:
    return "" if path is None else str(path)


def _normalize_output_key(value: str) -> str:
    return "_".join(part for part in str(value).lower().replace("-", "_").split("_") if part)
