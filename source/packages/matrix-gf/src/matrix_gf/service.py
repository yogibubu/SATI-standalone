from __future__ import annotations

from dataclasses import dataclass
import csv
from io import StringIO
from pathlib import Path

import numpy as np

from matrix_smith import (
    CV_RADIAL_SIGMA_SCALE,
    CV_RADIAL_WEIGHT_THRESHOLD,
    read_gic_definition_from_xyzin,
)
from matrix_gaussian import hessian_input_from_gaussian_fchk
from matrix_qm import hessian_input_from_engine, hessian_input_from_xyzin

from .internal import (
    BOHR_TO_ANGSTROM,
    GFLocalOptions,
    InternalGFResult,
    gf_from_hessian_input_and_xyzin,
    gf_from_hessian_input_with_matrix_gics,
    topology_bonds_from_xyzin,
)
from .large_amplitude import gic_coordinate_family
from .nonbonded import nonbonded_cartesian_hessian_correction, synthon_charges_from_xyzin


@dataclass(frozen=True)
class GFFrequencyComparison:
    reference_source: str
    rows: tuple[tuple[int, float, float, float], ...]
    rms_delta_cm: float
    max_abs_delta_cm: float
    warning: str = ""


@dataclass(frozen=True)
class GFGeometryComparison:
    raw_rms_angstrom: float
    raw_max_angstrom: float
    aligned_rms_angstrom: float
    aligned_max_angstrom: float
    warning: str = ""


@dataclass(frozen=True)
class GFReport:
    fchk_path: Path
    result: InternalGFResult
    text: str
    xyzin_path: Path | None = None
    scale_path: Path | None = None
    hessian_source: str = ""
    frequency_comparison: GFFrequencyComparison | None = None
    geometry_comparison: GFGeometryComparison | None = None


@dataclass(frozen=True)
class GFScalingClass:
    """Named class of GICs receiving the same diagonal Pulay factor."""

    name: str
    factor: float
    patterns: tuple[str, ...]

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("Pulay scaling class name cannot be empty")
        if not self.patterns or any(not pattern.strip() for pattern in self.patterns):
            raise ValueError(f"Pulay scaling class {self.name!r} needs at least one pattern")
        if self.factor < 0.0:
            raise ValueError("Pulay scaling factors must be non-negative")


@dataclass(frozen=True)
class GFScalingRulePreview:
    order: int
    kind: str
    name: str
    factor: float
    patterns: tuple[str, ...]
    matches: tuple[int, ...]
    family: str = ""


@dataclass(frozen=True)
class GFScalingAssignment:
    index: int
    identifier: str
    name: str
    family: str
    factor: float
    source: str
    label: str


@dataclass(frozen=True)
class GFScalingPreview:
    xyzin_path: Path | None
    assignments: tuple[GFScalingAssignment, ...]
    rules: tuple[GFScalingRulePreview, ...]

    @property
    def changed_count(self) -> int:
        return sum(1 for item in self.assignments if abs(item.factor - 1.0) > 1.0e-12)


@dataclass(frozen=True)
class _ScalingEvent:
    kind: str
    selector: str = ""
    factor: float = 1.0
    scaling_class: GFScalingClass | None = None


def run_gf_report_from_fchk(fchk_path: Path) -> GFReport:
    """Read an FCHK adapter and return a formatted quick TRINITY harmonic report."""
    path = Path(fchk_path)
    hessian_input = hessian_input_from_gaussian_fchk(path)
    result = gf_from_hessian_input_with_matrix_gics(hessian_input)
    frequency_comparison = _frequency_comparison(
        result.frequencies_cm,
        hessian_input.harmonic_frequencies_cm,
        hessian_input.source or f"FCHK {path}",
    )
    return GFReport(
        path,
        result,
        format_gf_report(path, result, frequency_comparison=frequency_comparison),
        frequency_comparison=frequency_comparison,
    )


def run_xyzin_gf_report_from_fchk(
    fchk_path: Path,
    xyzin_path: Path,
    *,
    scale_path: Path | None = None,
    scale_records: tuple[str, ...] = (),
    scale_class_records: tuple[str, ...] = (),
    local: bool = False,
    force_threshold: float | None = None,
    block_by_irrep: bool = False,
    subtract_electrostatic: bool = False,
    subtract_uff_vdw: bool = False,
    nonbonded_14_scale: float = 0.5,
    large_amplitude_frequency_cutoff_cm: float | None = 250.0,
    large_amplitude_frequency_model: str = "projected",
    cv_correction: bool = False,
    cv_correction_sigma_scale: float = CV_RADIAL_SIGMA_SCALE,
    cv_correction_threshold: float = CV_RADIAL_WEIGHT_THRESHOLD,
) -> GFReport:
    """Run the frozen-xyzin GF branch from a Cartesian Hessian FCHK adapter."""
    path = Path(fchk_path)
    hessian_input = hessian_input_from_gaussian_fchk(path)
    return _run_xyzin_gf_report_from_hessian_input(
        hessian_input,
        path,
        Path(xyzin_path),
        hessian_source=f"FCHK {path}",
        scale_path=scale_path,
        scale_records=scale_records,
        scale_class_records=scale_class_records,
        local=local,
        force_threshold=force_threshold,
        block_by_irrep=block_by_irrep,
        subtract_electrostatic=subtract_electrostatic,
        subtract_uff_vdw=subtract_uff_vdw,
        nonbonded_14_scale=nonbonded_14_scale,
        large_amplitude_frequency_cutoff_cm=large_amplitude_frequency_cutoff_cm,
        large_amplitude_frequency_model=large_amplitude_frequency_model,
        cv_correction=cv_correction,
        cv_correction_sigma_scale=cv_correction_sigma_scale,
        cv_correction_threshold=cv_correction_threshold,
    )


def run_xyzin_gf_report_from_engine_hessian(
    engine: str,
    hessian_path: Path,
    xyzin_path: Path,
    *,
    cfour_grd: Path | None = None,
    cfour_output: Path | None = None,
    xtb_geometry: Path | None = None,
    xtb_spectrum: Path | None = None,
    xtb_output: Path | None = None,
    pyscf_input: Path | None = None,
    scale_path: Path | None = None,
    scale_records: tuple[str, ...] = (),
    scale_class_records: tuple[str, ...] = (),
    local: bool = False,
    force_threshold: float | None = None,
    block_by_irrep: bool = False,
    subtract_electrostatic: bool = False,
    subtract_uff_vdw: bool = False,
    nonbonded_14_scale: float = 0.5,
    large_amplitude_frequency_cutoff_cm: float | None = 250.0,
    large_amplitude_frequency_model: str = "projected",
    cv_correction: bool = False,
    cv_correction_sigma_scale: float = CV_RADIAL_SIGMA_SCALE,
    cv_correction_threshold: float = CV_RADIAL_WEIGHT_THRESHOLD,
) -> GFReport:
    """Run frozen-SONIC GF from any supported MATRIX Cartesian-Hessian adapter."""

    path = Path(hessian_path)
    hessian_input = hessian_input_from_engine(
        engine,
        path,
        grd=cfour_grd,
        output=xtb_output if str(engine).strip().lower() == "xtb" else cfour_output,
        geometry=xtb_geometry,
        spectrum=xtb_spectrum,
        input_path=pyscf_input,
    )
    return _run_xyzin_gf_report_from_hessian_input(
        hessian_input,
        path,
        Path(xyzin_path),
        hessian_source=f"{engine}-hessian {path}",
        scale_path=scale_path,
        scale_records=scale_records,
        scale_class_records=scale_class_records,
        local=local,
        force_threshold=force_threshold,
        block_by_irrep=block_by_irrep,
        subtract_electrostatic=subtract_electrostatic,
        subtract_uff_vdw=subtract_uff_vdw,
        nonbonded_14_scale=nonbonded_14_scale,
        large_amplitude_frequency_cutoff_cm=large_amplitude_frequency_cutoff_cm,
        large_amplitude_frequency_model=large_amplitude_frequency_model,
        cv_correction=cv_correction,
        cv_correction_sigma_scale=cv_correction_sigma_scale,
        cv_correction_threshold=cv_correction_threshold,
    )


def run_xyzin_gf_report_from_hessian_input(
    hessian_input,
    source_path: Path,
    xyzin_path: Path,
    *,
    hessian_source: str,
    scale_path: Path | None = None,
    scale_records: tuple[str, ...] = (),
    scale_class_records: tuple[str, ...] = (),
    local: bool = False,
    force_threshold: float | None = None,
    block_by_irrep: bool = False,
    subtract_electrostatic: bool = False,
    subtract_uff_vdw: bool = False,
    nonbonded_14_scale: float = 0.5,
    large_amplitude_frequency_cutoff_cm: float | None = 250.0,
    large_amplitude_frequency_model: str = "projected",
    cv_correction: bool = False,
    cv_correction_sigma_scale: float = CV_RADIAL_SIGMA_SCALE,
    cv_correction_threshold: float = CV_RADIAL_WEIGHT_THRESHOLD,
) -> GFReport:
    """Run frozen-SONIC GF from an already validated, immutable Hessian input."""

    hessian_input.validate()
    return _run_xyzin_gf_report_from_hessian_input(
        hessian_input,
        Path(source_path),
        Path(xyzin_path),
        hessian_source=hessian_source,
        scale_path=scale_path,
        scale_records=scale_records,
        scale_class_records=scale_class_records,
        local=local,
        force_threshold=force_threshold,
        block_by_irrep=block_by_irrep,
        subtract_electrostatic=subtract_electrostatic,
        subtract_uff_vdw=subtract_uff_vdw,
        nonbonded_14_scale=nonbonded_14_scale,
        large_amplitude_frequency_cutoff_cm=large_amplitude_frequency_cutoff_cm,
        large_amplitude_frequency_model=large_amplitude_frequency_model,
        cv_correction=cv_correction,
        cv_correction_sigma_scale=cv_correction_sigma_scale,
        cv_correction_threshold=cv_correction_threshold,
    )


def run_xyzin_gf_report_from_xyzin(
    xyzin_path: Path,
    *,
    scale_path: Path | None = None,
    scale_records: tuple[str, ...] = (),
    scale_class_records: tuple[str, ...] = (),
    local: bool = False,
    force_threshold: float | None = None,
    block_by_irrep: bool = False,
    subtract_electrostatic: bool = False,
    subtract_uff_vdw: bool = False,
    nonbonded_14_scale: float = 0.5,
    large_amplitude_frequency_cutoff_cm: float | None = 250.0,
    large_amplitude_frequency_model: str = "projected",
    cv_correction: bool = False,
    cv_correction_sigma_scale: float = CV_RADIAL_SIGMA_SCALE,
    cv_correction_threshold: float = CV_RADIAL_WEIGHT_THRESHOLD,
) -> GFReport:
    """Run GF from frozen #GIC and #CARTESIAN_HESSIAN sections in one xyzin."""
    xyzin = Path(xyzin_path)
    hessian_input = hessian_input_from_xyzin(xyzin)
    return _run_xyzin_gf_report_from_hessian_input(
        hessian_input,
        xyzin,
        xyzin,
        hessian_source=f"#CARTESIAN_HESSIAN in {xyzin}",
        scale_path=scale_path,
        scale_records=scale_records,
        scale_class_records=scale_class_records,
        local=local,
        force_threshold=force_threshold,
        block_by_irrep=block_by_irrep,
        subtract_electrostatic=subtract_electrostatic,
        subtract_uff_vdw=subtract_uff_vdw,
        nonbonded_14_scale=nonbonded_14_scale,
        large_amplitude_frequency_cutoff_cm=large_amplitude_frequency_cutoff_cm,
        large_amplitude_frequency_model=large_amplitude_frequency_model,
        cv_correction=cv_correction,
        cv_correction_sigma_scale=cv_correction_sigma_scale,
        cv_correction_threshold=cv_correction_threshold,
    )


def _run_xyzin_gf_report_from_hessian_input(
    hessian_input,
    source_path: Path,
    xyzin: Path,
    *,
    hessian_source: str,
    scale_path: Path | None = None,
    scale_records: tuple[str, ...] = (),
    scale_class_records: tuple[str, ...] = (),
    local: bool = False,
    force_threshold: float | None = None,
    block_by_irrep: bool = False,
    subtract_electrostatic: bool = False,
    subtract_uff_vdw: bool = False,
    nonbonded_14_scale: float = 0.5,
    large_amplitude_frequency_cutoff_cm: float | None = 250.0,
    large_amplitude_frequency_model: str = "projected",
    cv_correction: bool = False,
    cv_correction_sigma_scale: float = CV_RADIAL_SIGMA_SCALE,
    cv_correction_threshold: float = CV_RADIAL_WEIGHT_THRESHOLD,
) -> GFReport:
    xyzin = Path(xyzin)
    definition = read_gic_definition_from_xyzin(xyzin)
    correction = None
    correction_label = "NONE"
    if subtract_electrostatic or subtract_uff_vdw:
        topology_bonds = topology_bonds_from_xyzin(xyzin)
        charges = None
        labels: list[str] = []
        if subtract_electrostatic:
            charges, charge_source = synthon_charges_from_xyzin(
                xyzin, len(hessian_input.atomic_numbers)
            )
            labels.append(f"ELECTROSTATIC({charge_source})")
        if subtract_uff_vdw:
            labels.append("UFF_VDW")
        correction = nonbonded_cartesian_hessian_correction(
            hessian_input.cartesian_coordinates_bohr,
            hessian_input.atomic_numbers,
            topology_bonds,
            charges=charges,
            electrostatic=subtract_electrostatic,
            uff_vdw=subtract_uff_vdw,
            one_four_scale=nonbonded_14_scale,
        )
        correction_label = (
            "+".join(labels)
            + f"; 1-4 scale={float(nonbonded_14_scale):g}"
            + "; analytic radial chain rule; target=SONIC"
        )
    names = tuple(gic.name for gic in definition.gics)
    labels = tuple(
        _gic_display_label(gic.identifier, gic.gaussian_expression) for gic in definition.gics
    )
    scaling = pulay_scaling_factors(
        len(definition.gics),
        labels=labels,
        names=names,
        scale_path=scale_path,
        scale_records=scale_records,
        scale_class_records=scale_class_records,
    )
    result = gf_from_hessian_input_and_xyzin(
        hessian_input,
        xyzin,
        scaling_factors=scaling,
        local_options=GFLocalOptions(enabled=local),
        force_threshold=force_threshold,
        block_by_irrep=block_by_irrep,
        cartesian_hessian_correction=correction,
        cartesian_hessian_correction_label=correction_label,
        large_amplitude_frequency_cutoff_cm=large_amplitude_frequency_cutoff_cm,
        large_amplitude_frequency_model=large_amplitude_frequency_model,
        cv_correction=cv_correction,
        cv_correction_sigma_scale=cv_correction_sigma_scale,
        cv_correction_threshold=cv_correction_threshold,
    )
    frequency_comparison = _frequency_comparison(
        result.frequencies_cm,
        hessian_input.harmonic_frequencies_cm,
        hessian_source,
    )
    geometry_comparison = _geometry_comparison(
        definition.reference_coordinates_angstrom,
        hessian_input.cartesian_coordinates_bohr,
    )
    return GFReport(
        Path(source_path),
        result,
        format_gf_report(
            Path(source_path),
            result,
            xyzin_path=xyzin,
            scale_path=scale_path,
            hessian_source=hessian_source,
            frequency_comparison=frequency_comparison,
            geometry_comparison=geometry_comparison,
        ),
        xyzin_path=xyzin,
        scale_path=scale_path,
        hessian_source=hessian_source,
        frequency_comparison=frequency_comparison,
        geometry_comparison=geometry_comparison,
    )


def format_gf_report(
    fchk_path: Path,
    result: InternalGFResult,
    *,
    xyzin_path: Path | None = None,
    scale_path: Path | None = None,
    hessian_source: str | None = None,
    frequency_comparison: GFFrequencyComparison | None = None,
    geometry_comparison: GFGeometryComparison | None = None,
) -> str:
    lines = [
        "TRINITY harmonic analysis from SMITH-generated SONIC coordinates",
        f"Hessian source: {hessian_source or f'FCHK {Path(fchk_path)}'}",
        f"Coordinate source: {result.coordinate_source}",
        f"Point group: {result.point_group}",
        f"Symmetrized GICs: {result.symmetrized_gics}",
        f"Matrix model: {result.matrix_model}",
        f"Cartesian Hessian correction: {result.hessian_correction}",
        f"GIC count: {len(result.gic_labels)}",
    ]
    if xyzin_path is not None:
        lines.insert(2, f"Frozen xyzin: {Path(xyzin_path)}")
    if result.scaling_factors is not None:
        changed = int(np.sum(np.abs(result.scaling_factors - 1.0) > 1.0e-12))
        lines.insert(
            3 if xyzin_path is not None else 2,
            "Pulay Hessian scaling: "
            f"applied ({changed} factors != 1; file={Path(scale_path) if scale_path else 'inline/default'})",
        )
    if result.force_threshold is not None:
        lines.insert(
            3 if xyzin_path is not None else 2,
            f"Force-constant threshold: {result.force_threshold:g}",
        )
    if result.block_labels:
        counts = ", ".join(
            f"{label}:{result.block_labels.count(label)}"
            for label in dict.fromkeys(result.block_labels)
        )
        lines.insert(3 if xyzin_path is not None else 2, f"GF blocks: {counts}")
    if result.large_amplitude is not None and result.large_amplitude.coordinate_count:
        large = result.large_amplitude
        family_counts = ", ".join(
            f"{family}:{sum(1 for item in large.coordinates if item.family == family)}"
            for family in dict.fromkeys(item.family for item in large.coordinates)
        )
        lines.append(
            "Large-amplitude GIC subspace: "
            f"coordinates={large.coordinate_count} active={large.active_coordinate_count} "
            f"frequency_cutoff_cm={large.frequency_cutoff_cm} families={family_counts}"
        )
        lines.append("Large-amplitude block frequencies (cm-1):")
        for block in large.blocks:
            frequencies = ", ".join(f"{value:.3f}" for value in block.frequencies_cm)
            indices = ",".join(f"GIC{index:03d}" for index in block.indices)
            lines.append(
                f"  {block.label}: dim={len(block.indices)} [{indices}] "
                f"model={block.frequency_model} "
                f"freqs=[{frequencies}] "
                f"Fcouple={block.max_f_coupling_to_rest:.6g} "
                f"Gcouple={block.max_g_coupling_to_rest:.6g} "
                f"FGrel={block.relative_fg_coupling_to_rest:.6g} "
                f"Ginv_block={len(block.g_inverse_block)}x{len(block.g_inverse_block)} "
                f"metric={block.metric_role} kinetic={block.kinetic_operator_status}"
            )
        dvr_ready = [
            item
            for item in large.dvr_candidates
            if item.status in {"ACTIVE_TORSION_1D", "ACTIVE_COUPLED_DVR", "ACTIVE_BLOCK_DVR"}
        ]
        if dvr_ready:
            lines.append("Large-amplitude DVR plan:")
            for item in dvr_ready:
                fields = [
                    f"GIC{item.index:03d}",
                    item.family,
                    item.status,
                    f"freq={item.frequency_cm:.3f}" if item.frequency_cm is not None else "freq=NA",
                    f"FGrel={item.fg_coupling_to_rest:.6g}",
                ]
                if item.periodicity is not None:
                    fields.append(f"period={item.periodicity}")
                if item.barrier_cm is not None:
                    fields.append(f"barrier={item.barrier_cm:.3f} cm-1")
                if item.barrier_kcal_mol is not None:
                    fields.append(f"barrier={item.barrier_kcal_mol:.3f} kcal/mol")
                if item.rotor_symmetry_number is not None:
                    fields.append(f"sigma={item.rotor_symmetry_number}")
                if item.rotor_multiplicity is not None:
                    fields.append(f"mult={item.rotor_multiplicity}")
                if item.hindered_rotor_status:
                    fields.append(f"hr={item.hindered_rotor_status}")
                if item.g_inverse_diagonal is not None:
                    fields.append(f"Ginvii={item.g_inverse_diagonal:.6g}")
                lines.append("  " + " ".join(fields))
        dominant = [item for item in large.mode_contributions if item.ped_percent >= 50.0]
        if dominant:
            lines.append("Large-amplitude dominated GF modes (PED >= 50%):")
            for item in dominant:
                lines.append(
                    f"  mode {item.mode:3d}: {item.frequency_cm:12.3f} cm-1 "
                    f"large-amplitude PED={item.ped_percent:7.2f}%"
                )
    if geometry_comparison is not None:
        lines.append(
            "Geometry check: "
            f"raw RMS={geometry_comparison.raw_rms_angstrom:.6g} Angstrom "
            f"max={geometry_comparison.raw_max_angstrom:.6g}; "
            f"aligned RMS={geometry_comparison.aligned_rms_angstrom:.6g} Angstrom "
            f"max={geometry_comparison.aligned_max_angstrom:.6g}"
        )
        if geometry_comparison.warning:
            lines.append(f"Geometry warning: {geometry_comparison.warning}")
    if frequency_comparison is not None:
        lines.append(
            f"Frequency check vs {frequency_comparison.reference_source}: "
            f"compared={len(frequency_comparison.rows)} "
            f"RMS delta={frequency_comparison.rms_delta_cm:.6g} cm-1 "
            f"max |delta|={frequency_comparison.max_abs_delta_cm:.6g} cm-1"
        )
        if frequency_comparison.warning:
            lines.append(f"Frequency warning: {frequency_comparison.warning}")
        lines.append("Frequency comparison (cm-1, sorted):")
        lines.append("  mode        GF       source        delta")
        for mode, gf_freq, ref_freq, delta in frequency_comparison.rows:
            lines.append(f"  {mode:4d} {gf_freq:10.3f} {ref_freq:12.3f} {delta:12.3f}")
    lines.extend(["", "Frequencies (cm-1):"])
    for idx, freq in enumerate(result.frequencies_cm, start=1):
        lines.append(f"  mode {idx:3d}: {freq:12.3f}")

    lines.extend(["", "GIC labels:"])
    for idx, label in enumerate(result.gic_labels, start=1):
        name = result.gic_names[idx - 1] if idx <= len(result.gic_names) else f"GIC{idx:03d}"
        irrep = result.gic_irreps[idx - 1] if idx <= len(result.gic_irreps) else "UNK"
        anh_class = (
            result.gic_anharmonic_classes[idx - 1]
            if idx <= len(result.gic_anharmonic_classes)
            else "PENDING_TOPOLOGY_OR_ANHARMONIC_DATA"
        )
        zero_model = (
            result.gic_zeroth_order_models[idx - 1]
            if idx <= len(result.gic_zeroth_order_models)
            else "REQUIRES_TOPOLOGY_OR_SCAN_OR_QFF"
        )
        lines.append(
            f"  GIC{idx:03d}: {name:12s} irrep={irrep:6s} "
            f"anh={anh_class} zero={zero_model} {label}"
        )

    lines.extend(["", "PED (%) rows=GIC cols=modes:"])
    header = "          " + " ".join(
        f"M{idx:02d}" for idx in range(1, len(result.frequencies_cm) + 1)
    )
    lines.append(header)
    for idx, row in enumerate(result.ped.values, start=1):
        values = " ".join(f"{value:7.2f}" for value in row)
        lines.append(f"  GIC{idx:03d} {values}")
    return "\n".join(lines)


def _frequency_comparison(
    gf_frequencies,
    reference_frequencies,
    reference_source: str,
) -> GFFrequencyComparison | None:
    gf_values = np.asarray(gf_frequencies, dtype=float).reshape(-1)
    ref_values = np.asarray(reference_frequencies, dtype=float).reshape(-1)
    gf_values = gf_values[np.isfinite(gf_values)]
    ref_values = ref_values[np.isfinite(ref_values)]
    if gf_values.size == 0 or ref_values.size == 0:
        return None
    warning = ""
    if gf_values.size != ref_values.size:
        warning = (
            f"frequency count mismatch: GF has {gf_values.size}, "
            f"reference has {ref_values.size}; compared sorted overlap"
        )
    count = int(min(gf_values.size, ref_values.size))
    gf_sorted = np.sort(gf_values)[:count]
    ref_sorted = np.sort(ref_values)[:count]
    deltas = gf_sorted - ref_sorted
    rows = tuple(
        (index + 1, float(gf_sorted[index]), float(ref_sorted[index]), float(deltas[index]))
        for index in range(count)
    )
    rms = float(np.sqrt(np.mean(deltas * deltas))) if count else 0.0
    max_abs = float(np.max(np.abs(deltas))) if count else 0.0
    return GFFrequencyComparison(
        reference_source=reference_source,
        rows=rows,
        rms_delta_cm=rms,
        max_abs_delta_cm=max_abs,
        warning=warning,
    )


def _geometry_comparison(
    definition_coordinates_angstrom,
    hessian_coordinates_bohr,
) -> GFGeometryComparison | None:
    definition = np.asarray(definition_coordinates_angstrom, dtype=float)
    hessian = np.asarray(hessian_coordinates_bohr, dtype=float) * BOHR_TO_ANGSTROM
    if definition.shape != hessian.shape or definition.ndim != 2 or definition.shape[1] != 3:
        return None
    raw_delta = definition - hessian
    raw_distances = np.linalg.norm(raw_delta, axis=1)
    raw_rms = float(np.sqrt(np.mean(raw_distances * raw_distances)))
    raw_max = float(np.max(raw_distances)) if raw_distances.size else 0.0
    definition_centered = definition - np.mean(definition, axis=0)
    hessian_centered = hessian - np.mean(hessian, axis=0)
    try:
        u, _singular, vt = np.linalg.svd(definition_centered.T @ hessian_centered)
        rotation = u @ vt
        aligned_delta = definition_centered @ rotation - hessian_centered
    except np.linalg.LinAlgError:
        aligned_delta = definition_centered - hessian_centered
    aligned_distances = np.linalg.norm(aligned_delta, axis=1)
    aligned_rms = float(np.sqrt(np.mean(aligned_distances * aligned_distances)))
    aligned_max = float(np.max(aligned_distances)) if aligned_distances.size else 0.0
    warning = ""
    if aligned_max > 1.0e-3:
        warning = (
            "Hessian geometry differs from frozen #GIC reference geometry; "
            "the B matrix is evaluated on the Hessian geometry."
        )
    elif raw_max > 1.0e-3:
        warning = "Coordinate frames differ, but the aligned geometry is consistent."
    return GFGeometryComparison(
        raw_rms_angstrom=raw_rms,
        raw_max_angstrom=raw_max,
        aligned_rms_angstrom=aligned_rms,
        aligned_max_angstrom=aligned_max,
        warning=warning,
    )


def gf_csv_tables(report: GFReport) -> dict[str, str]:
    """Return CSV tables for GF frequencies, GIC labels, matrices, normal modes and PED."""
    freq_rows = [["mode", "frequency_cm-1"]]
    freq_rows.extend(
        [[idx, f"{freq:.10g}"] for idx, freq in enumerate(report.result.frequencies_cm, start=1)]
    )

    label_rows = [
        [
            "gic",
            "name",
            "irrep",
            "family",
            "anharmonic_class",
            "zeroth_order_model",
            "cross_coupling_policy",
            "anharmonic_reason",
            "label",
        ]
    ]
    for idx, label in enumerate(report.result.gic_labels, start=1):
        name = (
            report.result.gic_names[idx - 1]
            if idx <= len(report.result.gic_names)
            else f"GIC{idx:03d}"
        )
        irrep = report.result.gic_irreps[idx - 1] if idx <= len(report.result.gic_irreps) else "UNK"
        family = (
            report.result.gic_families[idx - 1]
            if idx <= len(report.result.gic_families)
            else ""
        )
        anh_class = (
            report.result.gic_anharmonic_classes[idx - 1]
            if idx <= len(report.result.gic_anharmonic_classes)
            else "PENDING_TOPOLOGY_OR_ANHARMONIC_DATA"
        )
        zero_model = (
            report.result.gic_zeroth_order_models[idx - 1]
            if idx <= len(report.result.gic_zeroth_order_models)
            else "REQUIRES_TOPOLOGY_OR_SCAN_OR_QFF"
        )
        cross_policy = (
            report.result.gic_cross_coupling_policies[idx - 1]
            if idx <= len(report.result.gic_cross_coupling_policies)
            else "REQUIRES_SUBSPACE_COUPLING_CHECK"
        )
        reason = (
            report.result.gic_anharmonic_reasons[idx - 1]
            if idx <= len(report.result.gic_anharmonic_reasons)
            else ""
        )
        label_rows.append(
            [
                f"GIC{idx:03d}",
                name,
                irrep,
                family,
                anh_class,
                zero_model,
                cross_policy,
                reason,
                label,
            ]
        )

    ped_rows = [
        [
            "gic",
            "name",
            "irrep",
            *[f"mode_{idx}" for idx in range(1, len(report.result.frequencies_cm) + 1)],
        ]
    ]
    for idx, row in enumerate(report.result.ped.values, start=1):
        name = (
            report.result.gic_names[idx - 1]
            if idx <= len(report.result.gic_names)
            else f"GIC{idx:03d}"
        )
        irrep = report.result.gic_irreps[idx - 1] if idx <= len(report.result.gic_irreps) else "UNK"
        ped_rows.append([f"GIC{idx:03d}", name, irrep, *[f"{value:.10g}" for value in row]])

    tables = {
        "frequencies.csv": _csv_text(freq_rows),
        "gic_labels.csv": _csv_text(label_rows),
        "ped.csv": _csv_text(ped_rows),
        "normal_modes.csv": _csv_text(
            _gic_mode_table(report.result.modes_internal, "mode", report.result)
        ),
        "force_constants.csv": _csv_text(
            _square_gic_table(report.result.force_constants, report.result)
        ),
        "g_matrix.csv": _csv_text(_square_gic_table(report.result.g_matrix, report.result)),
    }
    if report.result.large_amplitude is not None:
        g_inverse = np.asarray(report.result.large_amplitude.g_inverse, dtype=float)
        tables["g_inverse.csv"] = _csv_text(_square_gic_table(g_inverse, report.result))
        tables.update(_large_amplitude_csv_tables(report.result))
    if report.frequency_comparison is not None:
        tables["frequency_comparison.csv"] = _csv_text(
            [
                ["mode", "gf_frequency_cm-1", "source_frequency_cm-1", "delta_cm-1"],
                *[
                    [mode, f"{gf_freq:.10g}", f"{ref_freq:.10g}", f"{delta:.10g}"]
                    for mode, gf_freq, ref_freq, delta in report.frequency_comparison.rows
                ],
            ]
        )
    return tables


def write_csv_tables(report: GFReport, outdir: Path, *, prefix: str = "gf") -> dict[str, Path]:
    target_dir = Path(outdir)
    target_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, text in gf_csv_tables(report).items():
        path = target_dir / f"{prefix}_{name}"
        path.write_text(text, encoding="utf-8")
        written[name] = path
    return written


def pulay_scaling_factors(
    n_gics: int,
    *,
    labels: tuple[str, ...],
    names: tuple[str, ...] = (),
    scale_path: Path | None = None,
    scale_records: tuple[str, ...] = (),
    scale_class_records: tuple[str, ...] = (),
) -> np.ndarray | None:
    """Build diagonal Pulay scaling factors from selector and class records."""
    events = _collect_scaling_events(
        scale_path=scale_path,
        scale_records=scale_records,
        scale_class_records=scale_class_records,
    )
    if not events:
        return None
    factors = np.ones(n_gics, dtype=float)
    for event in events:
        if event.kind == "selector":
            if event.factor < 0.0:
                raise ValueError("Pulay scaling factors must be non-negative")
            for index in _resolve_scaling_selector(event.selector, labels=labels, names=names):
                factors[index] = event.factor
            continue
        if event.kind == "class" and event.scaling_class is not None:
            event.scaling_class.validate()
            for index in _resolve_scaling_class(event.scaling_class, labels=labels, names=names):
                factors[index] = event.scaling_class.factor
            continue
        raise ValueError(f"Unsupported Pulay scaling event: {event!r}")
    return factors


def gf_scaling_preview_from_xyzin(
    xyzin_path: Path,
    *,
    scale_path: Path | None = None,
    scale_records: tuple[str, ...] = (),
    scale_class_records: tuple[str, ...] = (),
) -> GFScalingPreview:
    """Return a dry-run Pulay scaling assignment preview for frozen xyzin GICs."""
    xyzin = Path(xyzin_path)
    definition = read_gic_definition_from_xyzin(xyzin)
    names = tuple(gic.name for gic in definition.gics)
    labels = tuple(
        _gic_display_label(gic.identifier, gic.gaussian_expression) for gic in definition.gics
    )
    return pulay_scaling_preview(
        len(definition.gics),
        labels=labels,
        names=names,
        xyzin_path=xyzin,
        scale_path=scale_path,
        scale_records=scale_records,
        scale_class_records=scale_class_records,
    )


def pulay_scaling_preview(
    n_gics: int,
    *,
    labels: tuple[str, ...],
    names: tuple[str, ...] = (),
    xyzin_path: Path | None = None,
    scale_path: Path | None = None,
    scale_records: tuple[str, ...] = (),
    scale_class_records: tuple[str, ...] = (),
) -> GFScalingPreview:
    """Resolve Pulay scaling records without running GF."""
    events = _collect_scaling_events(
        scale_path=scale_path,
        scale_records=scale_records,
        scale_class_records=scale_class_records,
    )
    factors = np.ones(n_gics, dtype=float)
    sources = ["default" for _ in range(n_gics)]
    padded_names = _padded_names(names, n_gics)
    rule_previews: list[GFScalingRulePreview] = []
    for order, event in enumerate(events, start=1):
        if event.kind == "selector":
            if event.factor < 0.0:
                raise ValueError("Pulay scaling factors must be non-negative")
            matches = _resolve_scaling_selector(event.selector, labels=labels, names=names)
            source = f"selector {event.selector}"
            for index in matches:
                factors[index] = event.factor
                sources[index] = source
            rule_previews.append(
                GFScalingRulePreview(
                    order=order,
                    kind="selector",
                    name=event.selector,
                    factor=event.factor,
                    patterns=(event.selector,),
                    matches=tuple(index + 1 for index in matches),
                    family=_common_family(matches, labels=labels, names=padded_names),
                )
            )
            continue
        if event.kind == "class" and event.scaling_class is not None:
            scaling_class = event.scaling_class
            scaling_class.validate()
            matches = _resolve_scaling_class(scaling_class, labels=labels, names=names)
            source = f"class {scaling_class.name}"
            for index in matches:
                factors[index] = scaling_class.factor
                sources[index] = source
            rule_previews.append(
                GFScalingRulePreview(
                    order=order,
                    kind="class",
                    name=scaling_class.name,
                    factor=scaling_class.factor,
                    patterns=scaling_class.patterns,
                    matches=tuple(index + 1 for index in matches),
                    family=_common_family(matches, labels=labels, names=padded_names),
                )
            )
            continue
        raise ValueError(f"Unsupported Pulay scaling event: {event!r}")
    assignments = tuple(
        GFScalingAssignment(
            index=index + 1,
            identifier=f"GIC{index + 1:03d}",
            name=padded_names[index] or f"GIC{index + 1:03d}",
            family=_gic_coordinate_family(padded_names[index], labels[index]) or "unknown",
            factor=float(factors[index]),
            source=sources[index],
            label=labels[index],
        )
        for index in range(n_gics)
    )
    return GFScalingPreview(
        None if xyzin_path is None else Path(xyzin_path), assignments, tuple(rule_previews)
    )


def format_gf_scaling_preview(preview: GFScalingPreview) -> str:
    """Format a Pulay scaling dry-run preview for CLI/GUI logs."""
    lines = ["TRINITY Pulay scaling preview"]
    if preview.xyzin_path is not None:
        lines.append(f"Frozen xyzin: {preview.xyzin_path}")
    lines.append(f"GIC count: {len(preview.assignments)}")
    lines.append(f"Changed factors: {preview.changed_count}")
    lines.append("")
    lines.append("Rules:")
    if not preview.rules:
        lines.append("  none; all factors remain 1.0")
    for rule in preview.rules:
        pattern_text = "|".join(rule.patterns)
        family = rule.family or "mixed/unknown"
        matches = ",".join(f"GIC{index:03d}" for index in rule.matches)
        lines.append(
            f"  {rule.order:02d} {rule.kind:8s} {rule.name:20s} "
            f"factor={rule.factor:.8g} family={family:10s} "
            f"matches={len(rule.matches):3d} [{matches}] patterns={pattern_text}"
        )
    lines.append("")
    lines.append("Assignments:")
    lines.append("  GIC     name             family     factor       source               label")
    for item in preview.assignments:
        lines.append(
            f"  {item.identifier:7s} {item.name[:14]:14s} {item.family[:10]:10s} "
            f"{item.factor:11.8g} {item.source[:20]:20s} {item.label}"
        )
    return "\n".join(lines)


def _collect_scaling_events(
    *,
    scale_path: Path | None = None,
    scale_records: tuple[str, ...] = (),
    scale_class_records: tuple[str, ...] = (),
) -> list[_ScalingEvent]:
    events: list[_ScalingEvent] = []
    if scale_path is not None:
        events.extend(_read_scaling_events(Path(scale_path)))
    for record in scale_records:
        events.append(_selector_event(*_parse_scaling_record(record)))
    for record in scale_class_records:
        events.append(_class_event(_parse_scaling_class_record(record)))
    return events




def _read_scaling_events(path: Path) -> list[_ScalingEvent]:
    events: list[_ScalingEvent] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.split("#", 1)[0].split("!", 1)[0].strip()
        if not line:
            continue
        lower = line.lower().replace(",", " ").split()
        if lower[:2] in (["selector", "factor"], ["gic", "factor"], ["name", "factor"]):
            continue
        events.append(_parse_scaling_event(line))
    return events


def _parse_scaling_event(record: str) -> _ScalingEvent:
    text = record.strip()
    if text.lower().startswith("class "):
        return _class_event(_parse_scaling_class_record(text))
    return _selector_event(*_parse_scaling_record(text))


def _selector_event(selector: str, factor: float) -> _ScalingEvent:
    return _ScalingEvent("selector", selector=selector, factor=factor)


def _class_event(scaling_class: GFScalingClass) -> _ScalingEvent:
    return _ScalingEvent("class", scaling_class=scaling_class)


def _parse_scaling_record(record: str) -> tuple[str, float]:
    text = record.strip()
    if "=" in text:
        selector, value = text.split("=", 1)
    elif "," in text:
        selector, value = text.split(",", 1)
    else:
        parts = text.split()
        if len(parts) != 2:
            raise ValueError(f"Invalid Pulay scaling record: {record!r}")
        selector, value = parts
    return selector.strip(), float(value.strip())


def _parse_scaling_class_record(record: str) -> GFScalingClass:
    text = record.strip()
    if text.lower().startswith("class "):
        text = text.split(None, 1)[1].strip()
    if ":" in text:
        parts = text.split(":", 2)
        if len(parts) != 3:
            raise ValueError(f"Invalid Pulay scaling class record: {record!r}")
        name, factor_text, patterns_text = parts
    else:
        parts = text.split(None, 2)
        if len(parts) != 3:
            raise ValueError(f"Invalid Pulay scaling class record: {record!r}")
        name, factor_text, patterns_text = parts
    factor_text = factor_text.strip()
    if "=" in factor_text:
        _key, factor_text = factor_text.split("=", 1)
    patterns = tuple(pattern.strip() for pattern in patterns_text.split("|") if pattern.strip())
    scaling_class = GFScalingClass(
        name=name.strip(), factor=float(factor_text.strip()), patterns=patterns
    )
    scaling_class.validate()
    return scaling_class


def _resolve_scaling_selector(
    selector: str,
    *,
    labels: tuple[str, ...],
    names: tuple[str, ...],
) -> tuple[int, ...]:
    token = selector.strip()
    lower = token.lower()
    n_gics = len(labels)
    if lower in {"*", "all", "default"}:
        return tuple(range(n_gics))
    if lower.startswith("gic") and lower[3:].isdigit():
        index = int(lower[3:]) - 1
        if 0 <= index < n_gics:
            return (index,)
    if token.isdigit():
        index = int(token) - 1
        if 0 <= index < n_gics:
            return (index,)
    exact = [idx for idx, name in enumerate(names) if name == token]
    exact.extend(idx for idx, label in enumerate(labels) if label == token)
    if exact:
        return tuple(dict.fromkeys(exact))
    matches = [
        idx
        for idx, (name, label) in enumerate(zip(_padded_names(names, n_gics), labels))
        if token in name or token in label
    ]
    if len(matches) == 1:
        return (matches[0],)
    if matches:
        raise ValueError(f"Pulay scaling selector {token!r} is ambiguous")
    raise ValueError(f"Pulay scaling selector {token!r} did not match any GIC")


def _resolve_scaling_class(
    scaling_class: GFScalingClass,
    *,
    labels: tuple[str, ...],
    names: tuple[str, ...],
) -> tuple[int, ...]:
    matches: list[int] = []
    for pattern in scaling_class.patterns:
        matches.extend(_resolve_scaling_class_pattern(pattern, labels=labels, names=names))
    unique = tuple(dict.fromkeys(matches))
    if not unique:
        raise ValueError(f"Pulay scaling class {scaling_class.name!r} did not match any GIC")
    padded_names = _padded_names(names, len(labels))
    families = {_gic_coordinate_family(padded_names[index], labels[index]) for index in unique}
    known_families = {family for family in families if family}
    if len(known_families) > 1:
        raise ValueError(
            f"Pulay scaling class {scaling_class.name!r} mixes coordinate types: "
            f"{', '.join(sorted(known_families))}"
        )
    return unique


def _resolve_scaling_class_pattern(
    pattern: str,
    *,
    labels: tuple[str, ...],
    names: tuple[str, ...],
) -> tuple[int, ...]:
    token = pattern.strip()
    lower = token.lower()
    n_gics = len(labels)
    if lower in {"*", "all", "default"}:
        return tuple(range(n_gics))
    if lower.startswith("gic") and lower[3:].isdigit():
        index = int(lower[3:]) - 1
        if 0 <= index < n_gics:
            return (index,)
    if token.isdigit():
        index = int(token) - 1
        if 0 <= index < n_gics:
            return (index,)
    padded_names = _padded_names(names, n_gics)
    exact = [idx for idx, name in enumerate(padded_names) if name.lower() == lower]
    exact.extend(idx for idx, label in enumerate(labels) if label.lower() == lower)
    if exact:
        return tuple(dict.fromkeys(exact))
    matches = [
        idx
        for idx, (name, label) in enumerate(zip(padded_names, labels))
        if lower in name.lower() or lower in label.lower()
    ]
    if matches:
        return tuple(dict.fromkeys(matches))
    raise ValueError(f"Pulay scaling class pattern {token!r} did not match any GIC")


def _gic_coordinate_family(name: str, label: str) -> str:
    return gic_coordinate_family(name, label)


def _common_family(
    indices: tuple[int, ...],
    *,
    labels: tuple[str, ...],
    names: tuple[str, ...],
) -> str:
    families = {_gic_coordinate_family(names[index], labels[index]) for index in indices}
    known = {family for family in families if family}
    if len(known) == 1:
        return next(iter(known))
    if len(known) > 1:
        return "mixed"
    return ""


def _padded_names(names: tuple[str, ...], n_gics: int) -> tuple[str, ...]:
    if len(names) >= n_gics:
        return names[:n_gics]
    return names + tuple("" for _ in range(n_gics - len(names)))


def _gic_display_label(identifier: str, expression: str) -> str:
    return f"{identifier} {expression}" if expression and expression != "NONE" else identifier


def _csv_text(rows: list[list[object]]) -> str:
    stream = StringIO()
    writer = csv.writer(stream)
    writer.writerows(rows)
    return stream.getvalue()


def _gic_mode_table(matrix: np.ndarray, label: str, result: InternalGFResult) -> list[list[object]]:
    values = np.asarray(matrix, dtype=float)
    rows: list[list[object]] = [
        ["gic", "name", "irrep", *[f"{label}_{idx}" for idx in range(1, values.shape[1] + 1)]]
    ]
    for idx, row in enumerate(values, start=1):
        name = result.gic_names[idx - 1] if idx <= len(result.gic_names) else f"GIC{idx:03d}"
        irrep = result.gic_irreps[idx - 1] if idx <= len(result.gic_irreps) else "UNK"
        rows.append([f"GIC{idx:03d}", name, irrep, *[f"{value:.10g}" for value in row]])
    return rows


def _square_gic_table(matrix: np.ndarray, result: InternalGFResult) -> list[list[object]]:
    values = np.asarray(matrix, dtype=float)
    rows: list[list[object]] = [
        ["gic", "name", "irrep", *[f"GIC{idx:03d}" for idx in range(1, values.shape[1] + 1)]]
    ]
    for idx, row in enumerate(values, start=1):
        name = result.gic_names[idx - 1] if idx <= len(result.gic_names) else f"GIC{idx:03d}"
        irrep = result.gic_irreps[idx - 1] if idx <= len(result.gic_irreps) else "UNK"
        rows.append([f"GIC{idx:03d}", name, irrep, *[f"{value:.10g}" for value in row]])
    return rows


def _large_amplitude_csv_tables(result: InternalGFResult) -> dict[str, str]:
    large = result.large_amplitude
    if large is None:
        return {}
    coordinate_rows = [
        [
            "gic",
            "name",
            "irrep",
            "family",
            "local_frequency_cm-1",
            "active",
            "status",
            "anharmonic_class",
            "zeroth_order_model",
            "cross_coupling_policy",
            "anharmonic_reason",
            "label",
        ]
    ]
    coordinate_rows.extend(
        [
            f"GIC{coordinate.index:03d}",
            coordinate.name,
            coordinate.irrep,
            coordinate.family,
            "" if coordinate.local_frequency_cm is None else f"{coordinate.local_frequency_cm:.10g}",
            int(coordinate.active),
            coordinate.status,
            coordinate.anharmonic_class,
            coordinate.zeroth_order_model,
            coordinate.cross_coupling_policy,
            coordinate.anharmonic_reason,
            coordinate.label,
        ]
        for coordinate in large.coordinates
    )
    block_rows = [
        [
            "label",
            "family",
            "dimension",
            "gics",
            "frequencies_cm-1",
            "max_f_coupling_to_rest",
            "relative_f_coupling_to_rest",
            "max_g_coupling_to_rest",
            "relative_g_coupling_to_rest",
            "relative_fg_coupling_to_rest",
            "g_inverse_source",
            "metric_role",
            "kinetic_operator_status",
            "anharmonic_class",
            "zeroth_order_model",
            "cross_coupling_policy",
            "anharmonic_reason",
            "g_inverse_block",
        ]
    ]
    block_rows.extend(
        [
            block.label,
            block.family,
            len(block.indices),
            ",".join(f"GIC{index:03d}" for index in block.indices),
            ",".join(f"{value:.10g}" for value in block.frequencies_cm),
            f"{block.max_f_coupling_to_rest:.10g}",
            f"{block.relative_f_coupling_to_rest:.10g}",
            f"{block.max_g_coupling_to_rest:.10g}",
            f"{block.relative_g_coupling_to_rest:.10g}",
            f"{block.relative_fg_coupling_to_rest:.10g}",
            block.g_inverse_source,
            block.metric_role,
            block.kinetic_operator_status,
            block.anharmonic_class,
            block.zeroth_order_model,
            block.cross_coupling_policy,
            block.anharmonic_reason,
            _compact_matrix_csv(block.g_inverse_block),
        ]
        for block in large.blocks
    )
    g_inverse_block_rows = [
        ["block", "family", "row_gic", "col_gic", "g_inverse_value", "g_inverse_source"]
    ]
    for block in large.blocks:
        for row_idx, row in enumerate(block.g_inverse_block):
            row_gic = block.indices[row_idx]
            for col_idx, value in enumerate(row):
                col_gic = block.indices[col_idx]
                g_inverse_block_rows.append(
                    [
                        block.label,
                        block.family,
                        f"GIC{row_gic:03d}",
                        f"GIC{col_gic:03d}",
                        f"{value:.10g}",
                        block.g_inverse_source,
                    ]
                )
    mode_rows = [["mode", "frequency_cm-1", "large_amplitude_ped_percent"]]
    mode_rows.extend(
        [item.mode, f"{item.frequency_cm:.10g}", f"{item.ped_percent:.10g}"]
        for item in large.mode_contributions
    )
    dvr_rows = [
        [
            "gic",
            "name",
            "irrep",
            "family",
            "status",
            "frequency_cm-1",
            "fg_coupling_to_rest",
            "central_bond",
            "periodicity",
            "minimum_rad",
            "force_constant_hartree",
            "fourier_amplitude_cm-1",
            "barrier_cm-1",
            "barrier_kcal_mol",
            "rotor_symmetry_number",
            "rotor_multiplicity",
            "hindered_rotor_status",
            "hindered_rotor_source",
            "g_inverse_diagonal",
            "g_inverse_source",
            "reason",
        ]
    ]
    dvr_rows.extend(
        [
            f"GIC{item.index:03d}",
            item.name,
            item.irrep,
            item.family,
            item.status,
            "" if item.frequency_cm is None else f"{item.frequency_cm:.10g}",
            f"{item.fg_coupling_to_rest:.10g}",
            "" if item.central_bond is None else f"{item.central_bond[0]}-{item.central_bond[1]}",
            "" if item.periodicity is None else item.periodicity,
            "" if item.minimum_rad is None else f"{item.minimum_rad:.10g}",
            "" if item.force_constant_hartree is None else f"{item.force_constant_hartree:.10g}",
            "" if item.fourier_amplitude_cm is None else f"{item.fourier_amplitude_cm:.10g}",
            "" if item.barrier_cm is None else f"{item.barrier_cm:.10g}",
            "" if item.barrier_kcal_mol is None else f"{item.barrier_kcal_mol:.10g}",
            "" if item.rotor_symmetry_number is None else item.rotor_symmetry_number,
            "" if item.rotor_multiplicity is None else item.rotor_multiplicity,
            item.hindered_rotor_status,
            item.hindered_rotor_source,
            "" if item.g_inverse_diagonal is None else f"{item.g_inverse_diagonal:.10g}",
            item.g_inverse_source,
            item.reason,
        ]
        for item in large.dvr_candidates
    )
    return {
        "large_amplitude_coordinates.csv": _csv_text(coordinate_rows),
        "large_amplitude_blocks.csv": _csv_text(block_rows),
        "large_amplitude_g_inverse_blocks.csv": _csv_text(g_inverse_block_rows),
        "large_amplitude_mode_ped.csv": _csv_text(mode_rows),
        "large_amplitude_dvr_plan.csv": _csv_text(dvr_rows),
    }


def _compact_matrix_csv(rows: tuple[tuple[float, ...], ...]) -> str:
    return ";".join(",".join(f"{value:.10g}" for value in row) for row in rows)
