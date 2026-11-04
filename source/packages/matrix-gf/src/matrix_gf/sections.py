from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from matrix_core import (
    key_value_section_lines,
    parse_key_value_section,
    read_sectioned_lines,
    replace_section,
    section_content,
)


ORACLE_XYZ_GF_PED_SCHEMA = "oracle.xyz.gf_ped.v1"


@dataclass(frozen=True)
class GFModeRow:
    index: int
    frequency_cm: float
    block: str = ""


@dataclass(frozen=True)
class GFGICRow:
    identifier: str
    name: str
    irrep: str
    label: str
    ped: tuple[float, ...] = ()
    scaling_factor: float | None = None
    family: str = ""
    anharmonic_class: str = "PENDING_TOPOLOGY_OR_ANHARMONIC_DATA"
    zeroth_order_model: str = "REQUIRES_TOPOLOGY_OR_SCAN_OR_QFF"
    cross_coupling_policy: str = "REQUIRES_SUBSPACE_COUPLING_CHECK"
    anharmonic_reason: str = ""


@dataclass(frozen=True)
class GFLargeAmplitudeCoordinateRow:
    identifier: str
    name: str
    irrep: str
    family: str
    label: str
    frequency_cm: float | None = None
    active: bool = True
    status: str = "ACTIVE"
    anharmonic_class: str = "PENDING_TOPOLOGY_OR_ANHARMONIC_DATA"
    zeroth_order_model: str = "REQUIRES_TOPOLOGY_OR_SCAN_OR_QFF"
    cross_coupling_policy: str = "REQUIRES_SUBSPACE_COUPLING_CHECK"
    anharmonic_reason: str = ""


@dataclass(frozen=True)
class GFLargeAmplitudeBlockRow:
    label: str
    family: str
    gics: tuple[str, ...]
    frequencies_cm: tuple[float, ...]
    frequency_model: str = "full"
    g_inverse_block: tuple[tuple[float, ...], ...] = ()
    g_inverse_source: str = ""
    metric_role: str = "EQUILIBRIUM_REFERENCE_ONLY"
    kinetic_operator_status: str = "REQUIRES_PODOLSKY_GRID_METRIC"
    anharmonic_class: str = "PENDING_TOPOLOGY_OR_ANHARMONIC_DATA"
    zeroth_order_model: str = "REQUIRES_TOPOLOGY_OR_SCAN_OR_QFF"
    cross_coupling_policy: str = "REQUIRES_SUBSPACE_COUPLING_CHECK"
    anharmonic_reason: str = ""
    max_f_coupling_to_rest: float = 0.0
    relative_f_coupling_to_rest: float = 0.0
    max_g_coupling_to_rest: float = 0.0
    relative_g_coupling_to_rest: float = 0.0

    @property
    def relative_fg_coupling_to_rest(self) -> float:
        return max(self.relative_f_coupling_to_rest, self.relative_g_coupling_to_rest)


@dataclass(frozen=True)
class GFLargeAmplitudeModeRow:
    index: int
    frequency_cm: float
    ped_percent: float


@dataclass(frozen=True)
class GFLargeAmplitudeDVRPlanRow:
    identifier: str
    name: str
    irrep: str
    family: str
    status: str
    frequency_cm: float | None = None
    fg_coupling_to_rest: float = 0.0
    central_bond: tuple[int, int] | None = None
    periodicity: int | None = None
    minimum_rad: float | None = None
    force_constant_hartree: float | None = None
    fourier_amplitude_cm: float | None = None
    barrier_cm: float | None = None
    barrier_kcal_mol: float | None = None
    rotor_symmetry_number: int | None = None
    rotor_multiplicity: int | None = None
    hindered_rotor_status: str = ""
    hindered_rotor_source: str = ""
    g_inverse_diagonal: float | None = None
    g_inverse_source: str = ""
    reason: str = ""


@dataclass(frozen=True)
class GFPEDSection:
    source_kind: str = "xyzin"
    source_path: Path | None = None
    hessian_source: str = ""
    coordinate_source: str = ""
    report_path: Path | None = None
    csv_dir: Path | None = None
    status: str = "complete"
    point_group: str = "UNKNOWN"
    symmetrized_gics: bool = False
    matrix_model: str = "FULL"
    hessian_correction: str = "NONE"
    force_threshold: float | None = None
    modes: tuple[GFModeRow, ...] = ()
    gics: tuple[GFGICRow, ...] = ()
    large_amplitude_coordinates: tuple[GFLargeAmplitudeCoordinateRow, ...] = ()
    large_amplitude_blocks: tuple[GFLargeAmplitudeBlockRow, ...] = ()
    large_amplitude_modes: tuple[GFLargeAmplitudeModeRow, ...] = ()
    large_amplitude_dvr_plan: tuple[GFLargeAmplitudeDVRPlanRow, ...] = ()
    schema: str = ORACLE_XYZ_GF_PED_SCHEMA

    def __post_init__(self) -> None:
        for attr in ("source_path", "report_path", "csv_dir"):
            value = getattr(self, attr)
            if value is not None:
                object.__setattr__(self, attr, Path(value))
        object.__setattr__(self, "modes", tuple(self.modes))
        object.__setattr__(self, "gics", tuple(self.gics))
        object.__setattr__(
            self,
            "large_amplitude_coordinates",
            tuple(self.large_amplitude_coordinates),
        )
        object.__setattr__(self, "large_amplitude_blocks", tuple(self.large_amplitude_blocks))
        object.__setattr__(self, "large_amplitude_modes", tuple(self.large_amplitude_modes))
        object.__setattr__(self, "large_amplitude_dvr_plan", tuple(self.large_amplitude_dvr_plan))


def gf_ped_section_from_report(
    report,
    *,
    source_kind: str | None = None,
    source_path: Path | str | None = None,
    report_path: Path | str | None = None,
    csv_dir: Path | str | None = None,
    status: str = "complete",
) -> GFPEDSection:
    """Build a normalized #GF_PED section from an ORACLE GF report object."""
    result = report.result
    frequencies = np.asarray(result.frequencies_cm, dtype=float).reshape(-1)
    block_labels = tuple(getattr(result, "block_labels", ()))
    modes = tuple(
        GFModeRow(
            index=idx,
            frequency_cm=float(frequency),
            block=block_labels[idx - 1] if idx - 1 < len(block_labels) else "",
        )
        for idx, frequency in enumerate(frequencies, start=1)
    )

    ped_values = np.asarray(result.ped.values, dtype=float)
    gic_labels = tuple(getattr(result, "gic_labels", ()))
    gic_names = tuple(getattr(result, "gic_names", ()))
    gic_irreps = tuple(getattr(result, "gic_irreps", ()))
    gic_families = tuple(getattr(result, "gic_families", ()))
    gic_anharmonic_classes = tuple(getattr(result, "gic_anharmonic_classes", ()))
    gic_zeroth_order_models = tuple(getattr(result, "gic_zeroth_order_models", ()))
    gic_cross_coupling_policies = tuple(getattr(result, "gic_cross_coupling_policies", ()))
    gic_anharmonic_reasons = tuple(getattr(result, "gic_anharmonic_reasons", ()))
    scaling = getattr(result, "scaling_factors", None)
    scaling_values = None if scaling is None else np.asarray(scaling, dtype=float).reshape(-1)
    gics: list[GFGICRow] = []
    for index, label in enumerate(gic_labels):
        ped = (
            tuple(float(value) for value in ped_values[index, :])
            if ped_values.ndim == 2 and index < ped_values.shape[0]
            else ()
        )
        scale = (
            float(scaling_values[index])
            if scaling_values is not None and index < scaling_values.size
            else None
        )
        gics.append(
            GFGICRow(
                identifier=f"GIC{index + 1:03d}",
                name=gic_names[index] if index < len(gic_names) else f"GIC{index + 1:03d}",
                irrep=gic_irreps[index] if index < len(gic_irreps) else "UNK",
                label=str(label),
                ped=ped,
                scaling_factor=scale,
                family=gic_families[index] if index < len(gic_families) else "",
                anharmonic_class=(
                    gic_anharmonic_classes[index]
                    if index < len(gic_anharmonic_classes)
                    else "PENDING_TOPOLOGY_OR_ANHARMONIC_DATA"
                ),
                zeroth_order_model=(
                    gic_zeroth_order_models[index]
                    if index < len(gic_zeroth_order_models)
                    else "REQUIRES_TOPOLOGY_OR_SCAN_OR_QFF"
                ),
                cross_coupling_policy=(
                    gic_cross_coupling_policies[index]
                    if index < len(gic_cross_coupling_policies)
                    else "REQUIRES_SUBSPACE_COUPLING_CHECK"
                ),
                anharmonic_reason=(
                    gic_anharmonic_reasons[index]
                    if index < len(gic_anharmonic_reasons)
                    else ""
                ),
            )
        )

    large = getattr(result, "large_amplitude", None)
    large_coordinates = ()
    large_blocks = ()
    large_modes = ()
    large_dvr_plan = ()
    if large is not None:
        large_coordinates = tuple(
            GFLargeAmplitudeCoordinateRow(
                identifier=f"GIC{coordinate.index:03d}",
                name=coordinate.name,
                irrep=coordinate.irrep,
                family=coordinate.family,
                label=coordinate.label,
                frequency_cm=_section_float(coordinate.local_frequency_cm),
                active=coordinate.active,
                status=coordinate.status,
                anharmonic_class=coordinate.anharmonic_class,
                zeroth_order_model=coordinate.zeroth_order_model,
                cross_coupling_policy=coordinate.cross_coupling_policy,
                anharmonic_reason=coordinate.anharmonic_reason,
            )
            for coordinate in large.coordinates
        )
        large_blocks = tuple(
            GFLargeAmplitudeBlockRow(
                label=block.label,
                family=block.family,
                gics=tuple(f"GIC{index:03d}" for index in block.indices),
                frequencies_cm=block.frequencies_cm,
                frequency_model=block.frequency_model,
                g_inverse_block=_section_matrix(block.g_inverse_block),
                g_inverse_source=block.g_inverse_source,
                metric_role=block.metric_role,
                kinetic_operator_status=block.kinetic_operator_status,
                anharmonic_class=block.anharmonic_class,
                zeroth_order_model=block.zeroth_order_model,
                cross_coupling_policy=block.cross_coupling_policy,
                anharmonic_reason=block.anharmonic_reason,
                max_f_coupling_to_rest=block.max_f_coupling_to_rest,
                relative_f_coupling_to_rest=block.relative_f_coupling_to_rest,
                max_g_coupling_to_rest=block.max_g_coupling_to_rest,
                relative_g_coupling_to_rest=block.relative_g_coupling_to_rest,
            )
            for block in large.blocks
        )
        large_modes = tuple(
            GFLargeAmplitudeModeRow(
                index=item.mode,
                frequency_cm=item.frequency_cm,
                ped_percent=item.ped_percent,
            )
            for item in large.mode_contributions
        )
        large_dvr_plan = tuple(
            GFLargeAmplitudeDVRPlanRow(
                identifier=f"GIC{item.index:03d}",
                name=item.name,
                irrep=item.irrep,
                family=item.family,
                status=item.status,
                frequency_cm=_section_float(item.frequency_cm),
                fg_coupling_to_rest=_section_float(item.fg_coupling_to_rest) or 0.0,
                central_bond=item.central_bond,
                periodicity=item.periodicity,
                minimum_rad=_section_float(item.minimum_rad),
                force_constant_hartree=_section_float(item.force_constant_hartree),
                fourier_amplitude_cm=_section_float(item.fourier_amplitude_cm),
                barrier_cm=_section_float(item.barrier_cm),
                barrier_kcal_mol=_section_float(item.barrier_kcal_mol),
                rotor_symmetry_number=item.rotor_symmetry_number,
                rotor_multiplicity=item.rotor_multiplicity,
                hindered_rotor_status=item.hindered_rotor_status,
                hindered_rotor_source=item.hindered_rotor_source,
                g_inverse_diagonal=_section_float(item.g_inverse_diagonal),
                g_inverse_source=item.g_inverse_source,
                reason=item.reason,
            )
            for item in large.dvr_candidates
        )

    resolved_source_path = Path(source_path) if source_path is not None else Path(report.fchk_path)
    resolved_source_kind = source_kind or _infer_source_kind(report, resolved_source_path)
    return GFPEDSection(
        source_kind=resolved_source_kind,
        source_path=resolved_source_path,
        hessian_source=getattr(report, "hessian_source", "") or str(resolved_source_path),
        coordinate_source=getattr(result, "coordinate_source", ""),
        report_path=None if report_path is None else Path(report_path),
        csv_dir=None if csv_dir is None else Path(csv_dir),
        status=status,
        point_group=getattr(result, "point_group", "UNKNOWN"),
        symmetrized_gics=bool(getattr(result, "symmetrized_gics", False)),
        matrix_model=getattr(result, "matrix_model", "FULL"),
        hessian_correction=getattr(result, "hessian_correction", "NONE"),
        force_threshold=getattr(result, "force_threshold", None),
        modes=modes,
        gics=tuple(gics),
        large_amplitude_coordinates=large_coordinates,
        large_amplitude_blocks=large_blocks,
        large_amplitude_modes=large_modes,
        large_amplitude_dvr_plan=large_dvr_plan,
    )


def gf_ped_section_lines(section: GFPEDSection) -> list[str]:
    values = {
        "STATUS": section.status,
        "SOURCE_KIND": section.source_kind,
        "SOURCE_PATH": section.source_path,
        "HESSIAN_SOURCE": section.hessian_source,
        "COORDINATE_SOURCE": section.coordinate_source,
        "REPORT": section.report_path,
        "CSV_DIR": section.csv_dir,
        "POINT_GROUP": section.point_group,
        "SYMMETRIZED_GICS": int(section.symmetrized_gics),
        "MATRIX_MODEL": section.matrix_model,
        "HESSIAN_CORRECTION": section.hessian_correction,
        "FORCE_THRESHOLD": None
        if section.force_threshold is None
        else _format_float(section.force_threshold),
        "MODE_COUNT": len(section.modes),
        "GIC_COUNT": len(section.gics),
    }
    lines = key_value_section_lines(
        ORACLE_XYZ_GF_PED_SCHEMA,
        values,
        key_order=(
            "STATUS",
            "SOURCE_KIND",
            "SOURCE_PATH",
            "HESSIAN_SOURCE",
            "COORDINATE_SOURCE",
            "REPORT",
            "CSV_DIR",
            "POINT_GROUP",
            "SYMMETRIZED_GICS",
            "MATRIX_MODEL",
            "HESSIAN_CORRECTION",
            "FORCE_THRESHOLD",
            "MODE_COUNT",
            "GIC_COUNT",
        ),
    )
    lines.append("[MODES]")
    if section.modes:
        for mode in section.modes:
            block = f" BLOCK={mode.block}" if mode.block else ""
            lines.append(f"{mode.index} FREQUENCY_CM-1={_format_float(mode.frequency_cm)}{block}")
    else:
        lines.append("NONE")

    lines.append("[GICS]")
    if section.gics:
        for gic in section.gics:
            scale = (
                "" if gic.scaling_factor is None else f" SCALE={_format_float(gic.scaling_factor)}"
            )
            lines.append(
                f"{gic.identifier} NAME={gic.name} IRREP={gic.irrep} "
                f"FAMILY={gic.family or 'NA'} "
                f"ANHARMONIC_CLASS={gic.anharmonic_class or 'NA'} "
                f"ZEROTH_ORDER_MODEL={gic.zeroth_order_model or 'NA'} "
                f"CROSS_COUPLING_POLICY={gic.cross_coupling_policy or 'NA'} "
                f"ANH_REASON={gic.anharmonic_reason or 'NA'}{scale} LABEL={gic.label}"
            )
    else:
        lines.append("NONE")

    lines.append("[PED]")
    if section.gics and any(gic.ped for gic in section.gics):
        for gic in section.gics:
            lines.append(
                f"{gic.identifier} VALUES={','.join(_format_float(value) for value in gic.ped)}"
            )
    else:
        lines.append("NONE")
    lines.append("[LARGE_AMPLITUDE_COORDINATES]")
    if section.large_amplitude_coordinates:
        for coordinate in section.large_amplitude_coordinates:
            lines.append(
                f"{coordinate.identifier} NAME={coordinate.name} IRREP={coordinate.irrep} "
                f"FAMILY={coordinate.family} "
                f"FREQUENCY_CM-1={_format_optional_float(coordinate.frequency_cm)} "
                f"ACTIVE={int(coordinate.active)} STATUS={coordinate.status} "
                f"ANHARMONIC_CLASS={coordinate.anharmonic_class or 'NA'} "
                f"ZEROTH_ORDER_MODEL={coordinate.zeroth_order_model or 'NA'} "
                f"CROSS_COUPLING_POLICY={coordinate.cross_coupling_policy or 'NA'} "
                f"ANH_REASON={coordinate.anharmonic_reason or 'NA'} "
                f"LABEL={coordinate.label}"
            )
    else:
        lines.append("NONE")

    lines.append("[LARGE_AMPLITUDE_BLOCKS]")
    if section.large_amplitude_blocks:
        for block in section.large_amplitude_blocks:
            lines.append(
                f"{block.label} FAMILY={block.family} DIM={len(block.gics)} "
                f"GICS={','.join(block.gics)} "
                f"FREQUENCIES_CM-1={','.join(_format_float(value) for value in block.frequencies_cm)} "
                f"FREQUENCY_MODEL={block.frequency_model or 'full'} "
                f"F_COUPLE_MAX={_format_float(block.max_f_coupling_to_rest)} "
                f"F_COUPLE_REL={_format_float(block.relative_f_coupling_to_rest)} "
                f"G_COUPLE_MAX={_format_float(block.max_g_coupling_to_rest)} "
                f"G_COUPLE_REL={_format_float(block.relative_g_coupling_to_rest)} "
                f"FG_COUPLE_REL={_format_float(block.relative_fg_coupling_to_rest)} "
                f"G_INV_SOURCE={block.g_inverse_source or 'NA'} "
                f"METRIC_ROLE={block.metric_role or 'NA'} "
                f"KINETIC_OPERATOR_STATUS={block.kinetic_operator_status or 'NA'} "
                f"ANHARMONIC_CLASS={block.anharmonic_class or 'NA'} "
                f"ZEROTH_ORDER_MODEL={block.zeroth_order_model or 'NA'} "
                f"CROSS_COUPLING_POLICY={block.cross_coupling_policy or 'NA'} "
                f"ANH_REASON={block.anharmonic_reason or 'NA'} "
                f"G_INV_BLOCK={_format_matrix(block.g_inverse_block)}"
            )
    else:
        lines.append("NONE")

    lines.append("[LARGE_AMPLITUDE_MODE_PED]")
    if section.large_amplitude_modes:
        for mode in section.large_amplitude_modes:
            lines.append(
                f"{mode.index} FREQUENCY_CM-1={_format_float(mode.frequency_cm)} "
                f"PED_PERCENT={_format_float(mode.ped_percent)}"
            )
    else:
        lines.append("NONE")
    lines.append("[LARGE_AMPLITUDE_DVR_PLAN]")
    if section.large_amplitude_dvr_plan:
        for row in section.large_amplitude_dvr_plan:
            lines.append(
                f"{row.identifier} NAME={row.name} IRREP={row.irrep} FAMILY={row.family} "
                f"STATUS={row.status} "
                f"FREQUENCY_CM-1={_format_optional_float(row.frequency_cm)} "
                f"FG_COUPLE_REL={_format_float(row.fg_coupling_to_rest)} "
                f"CENTRAL_BOND={_format_pair(row.central_bond)} "
                f"PERIODICITY={row.periodicity if row.periodicity is not None else 'NA'} "
                f"MINIMUM_RAD={_format_optional_float(row.minimum_rad)} "
                f"F_HARTREE={_format_optional_float(row.force_constant_hartree)} "
                f"FOURIER_AMPLITUDE_CM-1={_format_optional_float(row.fourier_amplitude_cm)} "
                f"BARRIER_CM-1={_format_optional_float(row.barrier_cm)} "
                f"BARRIER_KCAL_MOL={_format_optional_float(row.barrier_kcal_mol)} "
                f"ROTOR_SYMMETRY={row.rotor_symmetry_number if row.rotor_symmetry_number is not None else 'NA'} "
                f"ROTOR_MULTIPLICITY={row.rotor_multiplicity if row.rotor_multiplicity is not None else 'NA'} "
                f"HINDERED_ROTOR_STATUS={row.hindered_rotor_status or 'NA'} "
                f"HINDERED_ROTOR_SOURCE={row.hindered_rotor_source or 'NA'} "
                f"G_INV_DIAG={_format_optional_float(row.g_inverse_diagonal)} "
                f"G_INV_SOURCE={row.g_inverse_source or 'NA'} "
                f"REASON={row.reason or 'NA'}"
            )
    else:
        lines.append("NONE")
    return lines


def parse_gf_ped_section(lines: Iterable[str]) -> GFPEDSection:
    raw_lines = list(lines)
    values = parse_key_value_section(_header_lines(raw_lines))
    schema = values.get("SCHEMA", ORACLE_XYZ_GF_PED_SCHEMA)
    if schema != ORACLE_XYZ_GF_PED_SCHEMA:
        raise ValueError(f"unsupported GF_PED schema: {schema}")
    modes = tuple(
        _parse_mode_line(line) for line in _subsection(raw_lines, "MODES") if _data_line(line)
    )
    ped_by_gic = {
        identifier: values
        for identifier, values in (
            _parse_ped_line(line) for line in _subsection(raw_lines, "PED") if _data_line(line)
        )
    }
    gics = tuple(
        _parse_gic_line(line, ped_by_gic=ped_by_gic)
        for line in _subsection(raw_lines, "GICS")
        if _data_line(line)
    )
    large_coordinates = tuple(
        _parse_large_amplitude_coordinate_line(line)
        for line in _subsection(raw_lines, "LARGE_AMPLITUDE_COORDINATES")
        if _data_line(line)
    )
    large_blocks = tuple(
        _parse_large_amplitude_block_line(line)
        for line in _subsection(raw_lines, "LARGE_AMPLITUDE_BLOCKS")
        if _data_line(line)
    )
    large_modes = tuple(
        _parse_large_amplitude_mode_line(line)
        for line in _subsection(raw_lines, "LARGE_AMPLITUDE_MODE_PED")
        if _data_line(line)
    )
    large_dvr_plan = tuple(
        _parse_large_amplitude_dvr_plan_line(line)
        for line in _subsection(raw_lines, "LARGE_AMPLITUDE_DVR_PLAN")
        if _data_line(line)
    )
    return GFPEDSection(
        source_kind=values.get("SOURCE_KIND", "xyzin"),
        source_path=_optional_path(values.get("SOURCE_PATH")),
        hessian_source=values.get("HESSIAN_SOURCE", ""),
        coordinate_source=values.get("COORDINATE_SOURCE", ""),
        report_path=_optional_path(values.get("REPORT")),
        csv_dir=_optional_path(values.get("CSV_DIR")),
        status=values.get("STATUS", "complete"),
        point_group=values.get("POINT_GROUP", "UNKNOWN"),
        symmetrized_gics=_bool_value(values.get("SYMMETRIZED_GICS"), default=False),
        matrix_model=values.get("MATRIX_MODEL", "FULL"),
        hessian_correction=values.get("HESSIAN_CORRECTION", "NONE"),
        force_threshold=_optional_float(values.get("FORCE_THRESHOLD")),
        modes=modes,
        gics=gics,
        large_amplitude_coordinates=large_coordinates,
        large_amplitude_blocks=large_blocks,
        large_amplitude_modes=large_modes,
        large_amplitude_dvr_plan=large_dvr_plan,
        schema=schema,
    )


def read_gf_ped_section(path: Path | str) -> GFPEDSection:
    content = section_content(read_sectioned_lines(Path(path)), "GF_PED")
    if not content:
        raise ValueError("missing #GF_PED section")
    return parse_gf_ped_section(content)


def write_gf_ped_section(path: Path | str, section: GFPEDSection) -> None:
    replace_section(Path(path), "GF_PED", gf_ped_section_lines(section))


def write_gf_ped_section_from_report(
    path: Path | str,
    report,
    *,
    source_kind: str | None = None,
    source_path: Path | str | None = None,
    report_path: Path | str | None = None,
    csv_dir: Path | str | None = None,
    status: str = "complete",
) -> GFPEDSection:
    section = gf_ped_section_from_report(
        report,
        source_kind=source_kind,
        source_path=source_path,
        report_path=report_path,
        csv_dir=csv_dir,
        status=status,
    )
    write_gf_ped_section(path, section)
    return section


def _infer_source_kind(report, source_path: Path) -> str:
    xyzin_path = getattr(report, "xyzin_path", None)
    if xyzin_path is not None and Path(xyzin_path) == source_path:
        return "xyzin"
    return "fchk"


def _header_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            break
        out.append(line)
    return out


def _subsection(section_lines: list[str], name: str) -> list[str]:
    header = f"[{name.upper()}]"
    start = None
    for idx, line in enumerate(section_lines):
        if line.strip().upper() == header:
            start = idx + 1
            break
    if start is None:
        return []
    end = len(section_lines)
    for idx in range(start, len(section_lines)):
        text = section_lines[idx].strip()
        if text.startswith("[") and text.endswith("]"):
            end = idx
            break
    return list(section_lines[start:end])


def _parse_mode_line(line: str) -> GFModeRow:
    parts = line.split()
    if not parts:
        raise ValueError("empty GF_PED mode line")
    fields = _key_values(parts[1:])
    return GFModeRow(
        index=int(parts[0]),
        frequency_cm=float(fields["FREQUENCY_CM-1"]),
        block=fields.get("BLOCK", ""),
    )


def _parse_gic_line(line: str, *, ped_by_gic: Mapping[str, tuple[float, ...]]) -> GFGICRow:
    if " LABEL=" in line:
        prefix, label = line.split(" LABEL=", 1)
    else:
        prefix, label = line, ""
    parts = prefix.split()
    if not parts:
        raise ValueError("empty GF_PED GIC line")
    fields = _key_values(parts[1:])
    identifier = parts[0]
    return GFGICRow(
        identifier=identifier,
        name=fields.get("NAME", identifier),
        irrep=fields.get("IRREP", "UNK"),
        label=label,
        ped=ped_by_gic.get(identifier, ()),
        scaling_factor=_optional_float(fields.get("SCALE")),
        family=_optional_text(fields.get("FAMILY")),
        anharmonic_class=(
            _optional_text(fields.get("ANHARMONIC_CLASS"))
            or "PENDING_TOPOLOGY_OR_ANHARMONIC_DATA"
        ),
        zeroth_order_model=(
            _optional_text(fields.get("ZEROTH_ORDER_MODEL"))
            or "REQUIRES_TOPOLOGY_OR_SCAN_OR_QFF"
        ),
        cross_coupling_policy=(
            _optional_text(fields.get("CROSS_COUPLING_POLICY"))
            or "REQUIRES_SUBSPACE_COUPLING_CHECK"
        ),
        anharmonic_reason=_optional_text(fields.get("ANH_REASON")),
    )


def _parse_large_amplitude_coordinate_line(line: str) -> GFLargeAmplitudeCoordinateRow:
    if " LABEL=" in line:
        prefix, label = line.split(" LABEL=", 1)
    else:
        prefix, label = line, ""
    parts = prefix.split()
    if not parts:
        raise ValueError("empty GF_PED large-amplitude coordinate line")
    fields = _key_values(parts[1:])
    identifier = parts[0]
    return GFLargeAmplitudeCoordinateRow(
        identifier=identifier,
        name=fields.get("NAME", identifier),
        irrep=fields.get("IRREP", "UNK"),
        family=fields.get("FAMILY", ""),
        label=label,
        frequency_cm=_optional_float(fields.get("FREQUENCY_CM-1")),
        active=_bool_value(fields.get("ACTIVE"), default=True),
        status=fields.get("STATUS", "ACTIVE"),
        anharmonic_class=(
            _optional_text(fields.get("ANHARMONIC_CLASS"))
            or "PENDING_TOPOLOGY_OR_ANHARMONIC_DATA"
        ),
        zeroth_order_model=(
            _optional_text(fields.get("ZEROTH_ORDER_MODEL"))
            or "REQUIRES_TOPOLOGY_OR_SCAN_OR_QFF"
        ),
        cross_coupling_policy=(
            _optional_text(fields.get("CROSS_COUPLING_POLICY"))
            or "REQUIRES_SUBSPACE_COUPLING_CHECK"
        ),
        anharmonic_reason=_optional_text(fields.get("ANH_REASON")),
    )


def _parse_large_amplitude_block_line(line: str) -> GFLargeAmplitudeBlockRow:
    parts = line.split()
    if not parts:
        raise ValueError("empty GF_PED large-amplitude block line")
    fields = _key_values(parts[1:])
    return GFLargeAmplitudeBlockRow(
        label=parts[0],
        family=fields.get("FAMILY", ""),
        gics=tuple(item for item in fields.get("GICS", "").split(",") if item),
        frequencies_cm=_float_tuple(fields.get("FREQUENCIES_CM-1", "")),
        frequency_model=fields.get("FREQUENCY_MODEL", "full"),
        g_inverse_block=_float_matrix(fields.get("G_INV_BLOCK", "")),
        g_inverse_source=_optional_text(fields.get("G_INV_SOURCE")),
        metric_role=_optional_text(fields.get("METRIC_ROLE")) or "EQUILIBRIUM_REFERENCE_ONLY",
        kinetic_operator_status=_optional_text(fields.get("KINETIC_OPERATOR_STATUS"))
        or "REQUIRES_PODOLSKY_GRID_METRIC",
        anharmonic_class=(
            _optional_text(fields.get("ANHARMONIC_CLASS"))
            or "PENDING_TOPOLOGY_OR_ANHARMONIC_DATA"
        ),
        zeroth_order_model=(
            _optional_text(fields.get("ZEROTH_ORDER_MODEL"))
            or "REQUIRES_TOPOLOGY_OR_SCAN_OR_QFF"
        ),
        cross_coupling_policy=(
            _optional_text(fields.get("CROSS_COUPLING_POLICY"))
            or "REQUIRES_SUBSPACE_COUPLING_CHECK"
        ),
        anharmonic_reason=_optional_text(fields.get("ANH_REASON")),
        max_f_coupling_to_rest=_optional_float(fields.get("F_COUPLE_MAX")) or 0.0,
        relative_f_coupling_to_rest=_optional_float(fields.get("F_COUPLE_REL")) or 0.0,
        max_g_coupling_to_rest=_optional_float(fields.get("G_COUPLE_MAX")) or 0.0,
        relative_g_coupling_to_rest=_optional_float(fields.get("G_COUPLE_REL")) or 0.0,
    )


def _parse_large_amplitude_mode_line(line: str) -> GFLargeAmplitudeModeRow:
    parts = line.split()
    if not parts:
        raise ValueError("empty GF_PED large-amplitude mode line")
    fields = _key_values(parts[1:])
    return GFLargeAmplitudeModeRow(
        index=int(parts[0]),
        frequency_cm=float(fields["FREQUENCY_CM-1"]),
        ped_percent=float(fields["PED_PERCENT"]),
    )


def _parse_large_amplitude_dvr_plan_line(line: str) -> GFLargeAmplitudeDVRPlanRow:
    parts = line.split()
    if not parts:
        raise ValueError("empty GF_PED large-amplitude DVR plan line")
    fields = _key_values(parts[1:])
    return GFLargeAmplitudeDVRPlanRow(
        identifier=parts[0],
        name=fields.get("NAME", parts[0]),
        irrep=fields.get("IRREP", "UNK"),
        family=fields.get("FAMILY", ""),
        status=fields.get("STATUS", ""),
        frequency_cm=_optional_float(fields.get("FREQUENCY_CM-1")),
        fg_coupling_to_rest=_optional_float(fields.get("FG_COUPLE_REL")) or 0.0,
        central_bond=_optional_pair(fields.get("CENTRAL_BOND")),
        periodicity=_optional_int(fields.get("PERIODICITY")),
        minimum_rad=_optional_float(fields.get("MINIMUM_RAD")),
        force_constant_hartree=_optional_float(fields.get("F_HARTREE")),
        fourier_amplitude_cm=_optional_float(fields.get("FOURIER_AMPLITUDE_CM-1")),
        barrier_cm=_optional_float(fields.get("BARRIER_CM-1")),
        barrier_kcal_mol=_optional_float(fields.get("BARRIER_KCAL_MOL")),
        rotor_symmetry_number=_optional_int(fields.get("ROTOR_SYMMETRY")),
        rotor_multiplicity=_optional_int(fields.get("ROTOR_MULTIPLICITY")),
        hindered_rotor_status=_optional_text(fields.get("HINDERED_ROTOR_STATUS")),
        hindered_rotor_source=_optional_text(fields.get("HINDERED_ROTOR_SOURCE")),
        g_inverse_diagonal=_optional_float(fields.get("G_INV_DIAG")),
        g_inverse_source=_optional_text(fields.get("G_INV_SOURCE")),
        reason=fields.get("REASON", ""),
    )


def _parse_ped_line(line: str) -> tuple[str, tuple[float, ...]]:
    parts = line.split(maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"invalid GF_PED PED line: {line}")
    fields = _key_values([parts[1]])
    return parts[0], _float_tuple(fields.get("VALUES", ""))


def _key_values(parts: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for part in parts:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        result[key.upper()] = value
    return result


def _float_tuple(text: str) -> tuple[float, ...]:
    if not text or text.upper() == "NONE":
        return ()
    return tuple(float(item) for item in text.split(",") if item)


def _float_matrix(text: str) -> tuple[tuple[float, ...], ...]:
    if not text or text.upper() in {"NA", "NONE"}:
        return ()
    return tuple(
        tuple(float(item.replace("D", "E").replace("d", "e")) for item in row.split(",") if item)
        for row in text.split(";")
        if row
    )


def _optional_path(raw: str | None) -> Path | None:
    if raw is None or not raw.strip():
        return None
    return Path(raw)


def _optional_text(raw: str | None) -> str:
    if raw is None:
        return ""
    text = str(raw).strip()
    return "" if not text or text.upper() == "NA" else text


def _optional_float(raw: str | None) -> float | None:
    if raw is None or not str(raw).strip() or str(raw).strip().upper() == "NA":
        return None
    return float(str(raw).replace("D", "E").replace("d", "e"))


def _optional_int(raw: str | None) -> int | None:
    if raw is None or not str(raw).strip() or str(raw).strip().upper() == "NA":
        return None
    return int(raw)


def _optional_pair(raw: str | None) -> tuple[int, int] | None:
    if raw is None or not str(raw).strip() or str(raw).strip().upper() == "NA":
        return None
    left, right = str(raw).split(",", 1)
    return int(left), int(right)


def _bool_value(raw: str | None, *, default: bool) -> bool:
    if raw is None:
        return default
    return str(raw).strip().upper() in {"1", "TRUE", "YES", "Y"}


def _data_line(line: str) -> bool:
    text = line.strip()
    return bool(text and text.upper() != "NONE")


def _format_float(value: float) -> str:
    return f"{float(value):.12g}"


def _section_float(value: float | None) -> float | None:
    if value is None:
        return None
    return float(_format_float(value))


def _section_matrix(rows: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(_format_float(value)) for value in row) for row in rows)


def _format_optional_float(value: float | None) -> str:
    return "NA" if value is None else _format_float(value)


def _format_matrix(rows: tuple[tuple[float, ...], ...]) -> str:
    if not rows:
        return "NA"
    return ";".join(",".join(_format_float(value) for value in row) for row in rows)


def _format_pair(value: tuple[int, int] | None) -> str:
    return "NA" if value is None else f"{int(value[0])},{int(value[1])}"
