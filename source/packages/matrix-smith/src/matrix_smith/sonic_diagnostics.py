"""Human-readable definitions and Cartesian motion diagnostics for SMITH SONICs."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
from typing import Callable, Sequence

import numpy as np

from matrix_core import atomic_json_write
from matrix_chem import (
    Structure,
    atomic_mass,
    build_topology_objects,
    get_default_isotope,
    get_isotope,
    read_enriched_xyz,
    topology_bonds_from_xyzin,
)
from matrix_chem.topology.elements import atomic_number
from matrix_chem import (
    kabsch_rotation,
    read_xyzin_isotopologue_records,
)
from matrix_qm import read_cartesian_hessian_section

from .definition import read_gic_definition_from_xyzin
from .evaluation import (
    build_gic_b_matrix,
    evaluate_gic_value,
    evaluate_gic_values,
)
from .symmetry_labels import is_total_symmetric_irrep

SONIC_DIAGNOSTICS_SCHEMA_V1 = "matrix.smith.sonic_diagnostics.v1"
SONIC_DIAGNOSTICS_SCHEMA = "matrix.smith.sonic_diagnostics.v2"
SUPPORTED_SONIC_DIAGNOSTICS_SCHEMAS = (
    SONIC_DIAGNOSTICS_SCHEMA_V1,
    SONIC_DIAGNOSTICS_SCHEMA,
)
DEFAULT_DISTANCE_STEP_ANGSTROM = 0.05
DEFAULT_ANGLE_STEP_RADIAN = float(np.deg2rad(5.0))
DEFAULT_RING_PUCKERING_STEP_RADIAN = 0.10
DEFAULT_MAX_ATOM_DISPLACEMENT_ANGSTROM = 0.10
DEFAULT_MAX_BOND_RELATIVE_CHANGE = 0.15
CONSTANT_B_PINV_RCOND = 1.0e-8
CARTESIAN_METRICS = ("euclidean", "mass-weighted")
DEFAULT_CARTESIAN_METRIC = "euclidean"
MASS_SOURCES = ("auto", "average", "default-isotope", "hessian", "isotopologue")
DEFAULT_MASS_SOURCE = "auto"
SIGNIFICANT_ATOM_RELATIVE_THRESHOLD = 0.05
SONIC_TOPOLOGY_CACHE_SCHEMA = "matrix.smith.oracle_perception_cache.v1"
SONIC_TOPOLOGY_CACHE_FILENAME = "oracle_perception_cache.json"
AUTO_TOPOLOGY_WORKER_TASKS_PER_PROCESS = 12
AUTO_TOPOLOGY_WORKER_LIMIT = 8


_MotionWorkerContext = tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    frozenset[tuple[int, int]],
    tuple[int, ...],
]
_MOTION_WORKER_CONTEXT: _MotionWorkerContext | None = None


@dataclass(frozen=True)
class SonicCoordinateMotion:
    index: int
    identifier: str
    name: str
    family: str
    irrep: str
    unit: str
    step: float
    status: str
    minus_coordinates_angstrom: np.ndarray
    reference_coordinates_angstrom: np.ndarray
    plus_coordinates_angstrom: np.ndarray
    cartesian_vector_angstrom: np.ndarray
    minus_realized_displacement: float
    plus_realized_displacement: float
    minus_residual_norm: float
    plus_residual_norm: float
    max_atom_displacement_angstrom: float
    max_bond_relative_change: float
    connectivity_preserved: bool
    orientation_preserved: bool
    minimum_alignment_determinant: float
    cartesian_metric: str
    normalized_b_condition_number: float
    coordinate_condition_indicator: float
    conditioning_status: str
    cartesian_amplification_angstrom_per_unit: float
    participation_ratio: float
    significant_atom_count: int
    significant_atoms: tuple[int, ...]
    maximum_atom_index: int
    maximum_atom_symbol: str
    highlighted_atoms: tuple[int, ...]
    ring_atoms: tuple[int, ...]
    component_terms: tuple[str, ...]
    local_domain: str
    local_group: str
    local_irrep: str
    locally_totally_symmetric: bool
    globally_totally_symmetric: bool
    full_topology_checks: int
    trajectory_path: Path
    vector_path: Path


@dataclass(frozen=True)
class SonicDiagnosticsResult:
    output_directory: Path
    coordinate_report: Path
    aggregate_trajectory: Path
    manifest: Path
    motions: tuple[SonicCoordinateMotion, ...]


@dataclass(frozen=True)
class _ConstantBMotionResult:
    coordinates_angstrom: np.ndarray
    residual: np.ndarray


_ConstantBMotionTuple = tuple[
    _ConstantBMotionResult,
    _ConstantBMotionResult,
    float,
    str,
    bool,
    float,
    int,
]


@dataclass(frozen=True)
class _MassContract:
    requested: str
    resolved: str
    detail: str
    masses_amu: tuple[float, ...] = ()
    isotope_mass_numbers: tuple[int, ...] = ()


def default_sonic_diagnostics_directory(xyzin: Path | str) -> Path:
    source = Path(xyzin)
    return source.parent / f"{source.stem}_sonic"


def available_topology_workers() -> int:
    """Return the processors available to independent ORACLE perceptions."""
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        try:
            return max(1, len(affinity(0)))
        except OSError:
            pass
    return max(1, int(os.cpu_count() or 1))


def topology_process_start_method() -> str:
    """Return the platform process-start policy used by Python workers."""

    return multiprocessing.get_context().get_start_method()


def recommended_topology_workers(
    task_count: int,
    *,
    start_method: str | None = None,
) -> int:
    """Choose workers without imposing spawn serialization on every task."""

    tasks = max(0, int(task_count))
    if tasks <= 8:
        return 1
    method = str(start_method or topology_process_start_method()).strip().casefold()
    if method != "fork":
        # These topology probes are short and share large immutable NumPy
        # arrays. On spawn/forkserver platforms each process serializes that
        # context; the measured cost exceeds the complete 114-coordinate
        # pagodane audit. Explicit worker requests remain available for a
        # caller with a separately benchmarked large workload.
        return 1
    desired = max(
        2,
        (tasks + AUTO_TOPOLOGY_WORKER_TASKS_PER_PROCESS - 1)
        // AUTO_TOPOLOGY_WORKER_TASKS_PER_PROCESS,
    )
    return min(available_topology_workers(), AUTO_TOPOLOGY_WORKER_LIMIT, desired, tasks)


def sonic_diagnostics_schema_path() -> Path:
    """Return the installed JSON Schema for the current diagnostics contract."""
    return Path(__file__).with_name("schemas") / "sonic_diagnostics_v2.schema.json"


def load_sonic_diagnostics_manifest(path: Path | str) -> dict[str, object]:
    """Load and validate either a legacy-v1 or current-v2 diagnostics manifest."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("SMITH diagnostics manifest must contain a JSON object")
    validate_sonic_diagnostics_payload(payload, allow_v1=True)
    return payload


def validate_sonic_diagnostics_payload(
    payload: dict[str, object],
    *,
    allow_v1: bool = True,
) -> None:
    """Validate stable fields needed by CLI and GUI consumers.

    The bundled JSON Schema is the normative v2 machine contract.  This small
    dependency-free validator keeps clean installations self-contained and
    deliberately accepts v1 manifests for GUI/restart compatibility.
    """
    schema = payload.get("schema")
    supported = {SONIC_DIAGNOSTICS_SCHEMA}
    if allow_v1:
        supported.add(SONIC_DIAGNOSTICS_SCHEMA_V1)
    if schema not in supported:
        raise ValueError(f"unsupported SMITH diagnostics schema: {schema!r}")
    coordinates = payload.get("coordinates")
    if not isinstance(coordinates, list):
        raise ValueError("SMITH diagnostics coordinates must be a JSON array")
    if payload.get("coordinate_count") != len(coordinates):
        raise ValueError("SMITH diagnostics coordinate_count does not match coordinates")
    required_coordinate_fields = {
        "index": int,
        "identifier": str,
        "name": str,
        "family": str,
        "unit": str,
        "step": (int, float),
        "trajectory": str,
        "vector": str,
    }
    for position, record in enumerate(coordinates, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"SMITH diagnostics coordinate {position} must be an object")
        for field, field_type in required_coordinate_fields.items():
            if field not in record or not isinstance(record[field], field_type):
                raise ValueError(f"SMITH diagnostics coordinate {position} needs {field}")
    if schema == SONIC_DIAGNOSTICS_SCHEMA_V1:
        return
    required_v2 = (
        "mass_source_requested",
        "mass_source_resolved",
        "mass_source_detail",
        "masses_amu",
        "isotope_mass_numbers",
        "topology_validation",
        "full_topology_check_count",
    )
    missing = [field for field in required_v2 if field not in payload]
    if missing:
        raise ValueError(f"SMITH diagnostics v2 is missing: {', '.join(missing)}")
    masses = payload["masses_amu"]
    isotopes = payload["isotope_mass_numbers"]
    if not isinstance(masses, list) or not isinstance(isotopes, list):
        raise ValueError("SMITH diagnostics v2 mass provenance must use JSON arrays")
    if masses and (not all(isinstance(value, (int, float)) for value in masses)):
        raise ValueError("SMITH diagnostics v2 masses must be numeric")
    if isotopes and len(isotopes) != len(masses):
        raise ValueError("SMITH diagnostics isotope and mass arrays must have equal length")


def _resolve_mass_contract(
    source: Path,
    atoms: tuple[str, ...],
    coordinates: np.ndarray,
    atomic_numbers: tuple[int, ...],
    *,
    metric: str,
    requested_source: str,
    isotopologue_label: str | None,
) -> _MassContract:
    requested = str(requested_source).strip().lower()
    if requested not in MASS_SOURCES:
        raise ValueError(
            f"unsupported SONIC mass source {requested_source!r}; "
            f"choose one of {', '.join(MASS_SOURCES)}"
        )
    if metric == "euclidean":
        return _MassContract(requested, "not-used", "Euclidean Cartesian metric")
    if requested == "auto":
        try:
            hessian = read_cartesian_hessian_section(source)
        except ValueError as exc:
            if "missing #CARTESIAN_HESSIAN" not in str(exc):
                raise
        else:
            return _mass_contract_from_hessian(hessian, atomic_numbers, requested)
        requested = "isotopologue" if isotopologue_label else "average"
    if requested == "average":
        return _MassContract(
            str(requested_source).strip().lower(),
            "average",
            "standard average atomic masses from matrix_chem",
            tuple(float(atomic_mass(number)) for number in atomic_numbers),
        )
    if requested == "default-isotope":
        structure = Structure(list(atoms), [tuple(row) for row in coordinates])
        return _MassContract(
            str(requested_source).strip().lower(),
            "default-isotope",
            "default spectroscopic isotopes from matrix_chem",
            tuple(float(value) for value in structure.mass_isotope),
            tuple(int(value) for value in structure.isotopes),
        )
    if requested == "hessian":
        hessian = read_cartesian_hessian_section(source)
        return _mass_contract_from_hessian(
            hessian,
            atomic_numbers,
            str(requested_source).strip().lower(),
        )
    records = read_xyzin_isotopologue_records(source)
    label = (isotopologue_label or "parent").strip()
    record = next((item for item in records if item.label.casefold() == label.casefold()), None)
    if record is None:
        available = ", ".join(item.label for item in records)
        raise ValueError(f"unknown isotopologue {label!r}; available labels: {available}")
    isotopes: list[int] = []
    masses: list[float] = []
    for index, number in enumerate(atomic_numbers, start=1):
        default = get_default_isotope(number)
        if default is None:
            raise ValueError(f"no default isotope is available for atom {index}")
        mass_number = int(record.substitutions.get(index, default.A))
        isotope = get_isotope(number, mass_number)
        if isotope is None:
            raise ValueError(f"isotope {mass_number} is unavailable for atom {index} (Z={number})")
        isotopes.append(mass_number)
        masses.append(float(isotope.mass))
    return _MassContract(
        str(requested_source).strip().lower(),
        "isotopologue",
        f"#ISOTOPOLOGUES label={record.label}",
        tuple(masses),
        tuple(isotopes),
    )


def _mass_contract_from_hessian(
    hessian,
    atomic_numbers: tuple[int, ...],
    requested: str,
) -> _MassContract:
    if tuple(hessian.atomic_numbers) != atomic_numbers:
        raise ValueError("#CARTESIAN_HESSIAN atom ordering does not match the geometry")
    masses = tuple(float(value) for value in hessian.masses_amu)
    if not masses or any(not np.isfinite(value) or value <= 0.0 for value in masses):
        raise ValueError("#CARTESIAN_HESSIAN contains invalid masses")
    return _MassContract(
        requested,
        "hessian",
        f"#CARTESIAN_HESSIAN source={hessian.source or 'unspecified'}",
        masses,
    )


def _local_coordinate_metadata(definition) -> dict[str, dict[str, str]]:
    diagnostics = definition.reduction_diagnostics
    if diagnostics is None:
        return {}
    result: dict[str, dict[str, str]] = {}
    for detail in diagnostics.skipped_dependent_details:
        if not detail.startswith("LOCAL_SALC "):
            continue
        fields: dict[str, str] = {}
        for token in detail.split()[1:]:
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            fields[key.upper()] = value
        name = fields.get("GIC", "")
        if not name or fields.get("STATUS") != "KEPT":
            continue
        result[name] = {
            "domain": fields.get("DOMAIN", ""),
            "group": fields.get("GROUP", ""),
            "local_irrep": fields.get("LOCAL_IRREP", ""),
            "kind": fields.get("KIND", ""),
        }
    return result


def write_sonic_diagnostics(
    xyzin: Path | str,
    output_directory: Path | str | None = None,
    *,
    distance_step_angstrom: float = DEFAULT_DISTANCE_STEP_ANGSTROM,
    angle_step_radian: float = DEFAULT_ANGLE_STEP_RADIAN,
    ring_puckering_step_radian: float = DEFAULT_RING_PUCKERING_STEP_RADIAN,
    max_atom_displacement_angstrom: float = DEFAULT_MAX_ATOM_DISPLACEMENT_ANGSTROM,
    cartesian_metric: str = DEFAULT_CARTESIAN_METRIC,
    mass_source: str = DEFAULT_MASS_SOURCE,
    isotopologue_label: str | None = None,
    coordinate_indices: Sequence[int] | None = None,
    topology_workers: int | str = "auto",
    use_topology_cache: bool = True,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> SonicDiagnosticsResult:
    """Write readable SONIC definitions and one Cartesian motion per coordinate."""
    source = Path(xyzin).resolve()
    output = (
        default_sonic_diagnostics_directory(source)
        if output_directory is None
        else Path(output_directory).expanduser().resolve()
    )
    for name, value in (
        ("distance step", distance_step_angstrom),
        ("angle step", angle_step_radian),
        ("ring-puckering step", ring_puckering_step_radian),
        ("maximum atom displacement", max_atom_displacement_angstrom),
    ):
        if not np.isfinite(value) or float(value) <= 0.0:
            raise ValueError(f"SONIC {name} must be finite and positive")
    metric = str(cartesian_metric).strip().lower()
    if metric not in CARTESIAN_METRICS:
        raise ValueError(
            f"unsupported SONIC Cartesian metric {cartesian_metric!r}; "
            f"choose one of {', '.join(CARTESIAN_METRICS)}"
        )

    definition = read_gic_definition_from_xyzin(source)
    geometry = read_enriched_xyz(source)
    reference = np.asarray(geometry.coordinates_angstrom, dtype=float)
    atomic_numbers = tuple(atomic_number(atom) for atom in geometry.atoms)
    if any(number is None for number in atomic_numbers):
        raise ValueError("SMITH motion diagnostics require recognized atomic symbols")
    atomic_numbers = tuple(int(number) for number in atomic_numbers)
    mass_contract = _resolve_mass_contract(
        source,
        geometry.atoms,
        reference,
        atomic_numbers,
        metric=metric,
        requested_source=mass_source,
        isotopologue_label=isotopologue_label,
    )
    reference_topology = frozenset(
        (left - 1, right - 1) for left, right in topology_bonds_from_xyzin(source)
    )
    reference_values = evaluate_gic_values(definition, coordinates_angstrom=reference)
    reference_b = np.asarray(
        build_gic_b_matrix(definition, coordinates_angstrom=reference).rows,
        dtype=float,
    )
    reference_rank = int(np.linalg.matrix_rank(reference_b, tol=1.0e-10))
    if reference_rank != len(definition.gics):
        raise ValueError(
            "SMITH SONIC contract failed the Cartesian rank gate: "
            f"rank={reference_rank}, coordinates={len(definition.gics)}"
        )
    cartesian_from_q = _cartesian_from_q_matrix(
        reference_b,
        metric=metric,
        masses_amu=mass_contract.masses_amu,
    )
    normalized_b_condition, coordinate_condition_indicators = _conditioning_diagnostics(reference_b)
    primitive_by_id = {primitive.identifier: primitive for primitive in definition.primitives}
    local_metadata = _local_coordinate_metadata(definition)

    output.mkdir(parents=True, exist_ok=True)
    motion_directory = output / "motions"
    motion_directory.mkdir(parents=True, exist_ok=True)
    coordinate_report = output / "sonic_coordinates.txt"
    aggregate_trajectory = output / "sonic_motions.xyz"
    manifest_path = output / "sonic_diagnostics.json"
    topology_cache_path = output / SONIC_TOPOLOGY_CACHE_FILENAME

    selected_indices = _selected_coordinate_indices(len(definition.gics), coordinate_indices)
    requested_topology_workers = _normalize_topology_worker_request(topology_workers)
    coordinate_steps = {
        index: _coordinate_step(
            definition.gics[index].family,
            distance_step_angstrom=float(distance_step_angstrom),
            angle_step_radian=float(angle_step_radian),
            ring_puckering_step_radian=float(ring_puckering_step_radian),
        )
        for index in selected_indices
    }
    cache_fingerprint = _topology_cache_fingerprint(
        reference,
        reference_b,
        cartesian_from_q,
        max_atom_displacement_angstrom=float(max_atom_displacement_angstrom),
        reference_topology=reference_topology,
        atomic_numbers=atomic_numbers,
    )
    topology_cache = _load_topology_cache(topology_cache_path, cache_fingerprint)
    motion_results: dict[int, _ConstantBMotionTuple] = {}
    pending_tasks: list[tuple[int, float]] = []
    cache_hits = 0
    for index in selected_indices:
        requested_step = coordinate_steps[index][1]
        cached = (
            _motion_result_from_cache(
                topology_cache.get("entries", {}).get(str(index + 1)),
                reference,
                reference_b,
                cartesian_from_q,
                index,
                requested_step,
            )
            if use_topology_cache
            else None
        )
        if cached is None:
            pending_tasks.append((index, requested_step))
        else:
            motion_results[index] = cached
            cache_hits += 1

    resolved_topology_workers = _resolve_topology_workers(
        requested_topology_workers,
        task_count=len(pending_tasks),
    )
    if progress_callback is not None and cache_hits:
        progress_callback(cache_hits, len(selected_indices), f"reused {cache_hits} cached SONICs")

    completed_new = 0

    def cache_completed(index: int, result: _ConstantBMotionTuple) -> None:
        nonlocal completed_new
        completed_new += 1
        if use_topology_cache:
            entries = topology_cache.setdefault("entries", {})
            entries[str(index + 1)] = _motion_result_cache_record(
                coordinate_steps[index][1],
                result,
            )
            _write_topology_cache(topology_cache_path, topology_cache)
        if progress_callback is not None:
            progress_callback(
                cache_hits + completed_new,
                len(selected_indices),
                f"SONIC {index + 1} topology verified",
            )

    motion_results.update(
        _constant_b_motion_batch(
            reference,
            reference_b,
            cartesian_from_q,
            tuple(pending_tasks),
            max_atom_displacement_angstrom=float(max_atom_displacement_angstrom),
            reference_topology=reference_topology,
            atomic_numbers=atomic_numbers,
            workers=resolved_topology_workers,
            completion_callback=cache_completed,
        )
    )
    pending_indices = {index for index, _step in pending_tasks}
    topology_checks_executed = sum(motion_results[index][6] for index in pending_indices)
    topology_checks_reused = sum(
        motion_results[index][6] for index in selected_indices if index not in pending_indices
    )

    from .report import gic_report_lines

    report_lines = gic_report_lines(definition)
    report_lines.extend(
        (
            "",
            "Cartesian Motion Diagnostics",
            "----------------------------",
            "Each display motion uses the B matrix frozen at the reference geometry.",
            (
                "The displacement is dx(k)=P(B)e(k)Delta and x(+/-)=x0+/-dx(k), "
                f"using the {metric} Cartesian metric."
            ),
            f"Mass source: {mass_contract.resolved} ({mass_contract.detail})",
            (
                "Independent ORACLE topology perceptions: "
                f"{resolved_topology_workers} process"
                f"{'es' if resolved_topology_workers != 1 else ''} "
                f"({available_topology_workers()} available)"
            ),
            (
                f"Topology cache: {cache_hits} reused, {len(pending_tasks)} evaluated "
                f"({topology_cache_path.name})"
            ),
            "These are linearized visualization vectors, not nonlinear back-transformations.",
            "The viewer rescales this mathematical vector independently to a user-selected "
            "maximum atomic displacement.",
            f"Distance step: {float(distance_step_angstrom):.10g} angstrom",
            f"Angular step: {float(angle_step_radian):.10g} radian",
            f"Ring-puckering step: {float(ring_puckering_step_radian):.10g} radian",
            f"Maximum per-atom displacement: {float(max_atom_displacement_angstrom):.10g} angstrom",
            "",
        )
    )

    motions: list[SonicCoordinateMotion] = []
    aggregate_lines: list[str] = []
    for index in selected_indices:
        gic = definition.gics[index]
        unit = coordinate_steps[index][0]
        (
            minus_result,
            plus_result,
            accepted_step,
            status,
            connectivity_preserved,
            maximum_bond_change,
            full_topology_checks,
        ) = motion_results[index]
        raw_direction = np.asarray(cartesian_from_q[:, index], dtype=float).reshape(reference.shape)
        localization = _localization_diagnostics(
            raw_direction,
            geometry.atoms,
            coordinate_condition_indicators[index],
        )
        highlighted_atoms, ring_atoms, component_terms = _coordinate_metadata(
            gic,
            primitive_by_id,
        )
        minus_rotation = kabsch_rotation(minus_result.coordinates_angstrom, reference)
        plus_rotation = kabsch_rotation(plus_result.coordinates_angstrom, reference)
        minus = minus_result.coordinates_angstrom
        plus = plus_result.coordinates_angstrom
        # The contract vector is the analytic constant-B SONIC tangent, not
        # the finite chord between the two nonlinear displaced geometries.
        # The latter is useful for visualization but can lose rank in coupled
        # ring-puckering blocks even when the frozen GIC B matrix is exact.
        vector = (2.0 * accepted_step) * raw_direction
        maximum_displacement = max(
            float(np.max(np.linalg.norm(minus - reference, axis=1), initial=0.0)),
            float(np.max(np.linalg.norm(plus - reference, axis=1), initial=0.0)),
        )
        minimum_determinant = min(
            float(np.linalg.det(minus_rotation)),
            float(np.linalg.det(plus_rotation)),
        )
        realized_minus = evaluate_gic_value(
            definition,
            index,
            coordinates_angstrom=minus,
        )
        realized_plus = evaluate_gic_value(
            definition,
            index,
            coordinates_angstrom=plus,
        )
        coordinate_local = local_metadata.get(gic.name, {})
        stem = f"{index + 1:04d}_{_safe_name(gic.name or gic.identifier)}"
        trajectory_path = motion_directory / f"{stem}.xyz"
        vector_path = motion_directory / f"{stem}.vector.tsv"
        trajectory_lines = _trajectory_lines(
            geometry.atoms,
            minus,
            reference,
            plus,
            label=f"{gic.identifier} {gic.name}",
            step=accepted_step,
            unit=unit,
        )
        trajectory_path.write_text("\n".join(trajectory_lines) + "\n", encoding="utf-8")
        vector_path.write_text(
            "\n".join(_vector_lines(geometry.atoms, minus, reference, plus, vector)) + "\n",
            encoding="utf-8",
        )
        aggregate_lines.extend(trajectory_lines)
        motion = SonicCoordinateMotion(
            index=index + 1,
            identifier=gic.identifier,
            name=gic.name,
            family=gic.family,
            irrep=gic.irrep,
            unit=unit,
            step=accepted_step,
            status=status,
            minus_coordinates_angstrom=minus,
            reference_coordinates_angstrom=reference.copy(),
            plus_coordinates_angstrom=plus,
            cartesian_vector_angstrom=vector,
            minus_realized_displacement=float(realized_minus - reference_values[index]),
            plus_realized_displacement=float(realized_plus - reference_values[index]),
            minus_residual_norm=float(np.linalg.norm(minus_result.residual)),
            plus_residual_norm=float(np.linalg.norm(plus_result.residual)),
            max_atom_displacement_angstrom=maximum_displacement,
            max_bond_relative_change=maximum_bond_change,
            connectivity_preserved=connectivity_preserved,
            orientation_preserved=minimum_determinant > 0.0,
            minimum_alignment_determinant=minimum_determinant,
            cartesian_metric=metric,
            normalized_b_condition_number=normalized_b_condition,
            coordinate_condition_indicator=localization["condition_indicator"],
            conditioning_status=localization["conditioning_status"],
            cartesian_amplification_angstrom_per_unit=localization["amplification"],
            participation_ratio=localization["participation_ratio"],
            significant_atom_count=localization["significant_atom_count"],
            significant_atoms=localization["significant_atoms"],
            maximum_atom_index=localization["maximum_atom_index"],
            maximum_atom_symbol=localization["maximum_atom_symbol"],
            highlighted_atoms=highlighted_atoms,
            ring_atoms=ring_atoms,
            component_terms=component_terms,
            local_domain=str(coordinate_local.get("domain", "")),
            local_group=str(coordinate_local.get("group", "")),
            local_irrep=str(coordinate_local.get("local_irrep", "")),
            locally_totally_symmetric=(coordinate_local.get("local_irrep") == "A1"),
            globally_totally_symmetric=is_total_symmetric_irrep(
                definition.point_group,
                gic.irrep,
            ),
            full_topology_checks=full_topology_checks,
            trajectory_path=trajectory_path,
            vector_path=vector_path,
        )
        motions.append(motion)
        report_lines.append(
            f"{index + 1:4d} {gic.identifier} {gic.name} status={status} "
            f"step={accepted_step:.10g} {unit} "
            f"max_atom={maximum_displacement:.6g} connectivity="
            f"{'PRESERVED' if connectivity_preserved else 'CHANGED'} "
            f"orientation={'PRESERVED' if minimum_determinant > 0.0 else 'CHANGED'} "
            f"conditioning={localization['conditioning_status']} "
            f"indicator={localization['condition_indicator']:.5g} "
            f"participation={localization['participation_ratio']:.5g} "
            f"moved_atoms={localization['significant_atom_count']} "
            f"largest_atom={localization['maximum_atom_index']}"
            f"({localization['maximum_atom_symbol']}) "
            f"local_domain={coordinate_local.get('domain', 'NONE')} "
            f"local_irrep={coordinate_local.get('local_irrep', 'NONE')} "
            f"trajectory={trajectory_path.relative_to(output)} "
            f"vector={vector_path.relative_to(output)}"
        )

    coordinate_report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    aggregate_trajectory.write_text("\n".join(aggregate_lines) + "\n", encoding="utf-8")
    payload = {
        "schema": SONIC_DIAGNOSTICS_SCHEMA,
        "source_xyzin": str(source),
        "coordinate_report": str(coordinate_report),
        "aggregate_trajectory": str(aggregate_trajectory),
        "coordinate_count": len(motions),
        "total_coordinate_count": len(definition.gics),
        "motion_model": "CONSTANT_REFERENCE_B_PSEUDOINVERSE",
        "cartesian_metric": metric,
        "mass_source_requested": mass_contract.requested,
        "mass_source_resolved": mass_contract.resolved,
        "mass_source_detail": mass_contract.detail,
        "masses_amu": list(mass_contract.masses_amu),
        "isotope_mass_numbers": list(mass_contract.isotope_mass_numbers),
        "b_pseudoinverse_rcond": CONSTANT_B_PINV_RCOND,
        "normalized_b_condition_number": normalized_b_condition,
        "significant_atom_relative_threshold": SIGNIFICANT_ATOM_RELATIVE_THRESHOLD,
        "topology_validation": "FULL_ORACLE_PER_ACCEPTED_FRAME",
        "topology_workers_requested": requested_topology_workers,
        "topology_workers_used": resolved_topology_workers,
        "topology_parallel_backend": (
            "CACHE_ONLY"
            if not pending_tasks
            else "PROCESS_POOL"
            if resolved_topology_workers > 1
            else "SERIAL"
        ),
        "topology_cache_enabled": bool(use_topology_cache),
        "topology_cache_path": str(topology_cache_path),
        "topology_cache_fingerprint": cache_fingerprint,
        "topology_cache_hits": cache_hits,
        "topology_cache_misses": len(pending_tasks),
        "topology_checks_executed": topology_checks_executed,
        "topology_checks_reused": topology_checks_reused,
        "full_topology_check_count": sum(motion.full_topology_checks for motion in motions),
        "coordinates": [_motion_record(motion, output) for motion in motions],
    }
    validate_sonic_diagnostics_payload(payload)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return SonicDiagnosticsResult(
        output_directory=output,
        coordinate_report=coordinate_report,
        aggregate_trajectory=aggregate_trajectory,
        manifest=manifest_path,
        motions=tuple(motions),
    )


def _cartesian_from_q_matrix(
    reference_b: np.ndarray,
    *,
    metric: str,
    masses_amu: tuple[float, ...],
) -> np.ndarray:
    """Return the right inverse of B in the requested Cartesian metric."""
    if metric == "euclidean":
        return np.linalg.pinv(reference_b, rcond=CONSTANT_B_PINV_RCOND)
    if not masses_amu or len(masses_amu) * 3 != reference_b.shape[1]:
        raise ValueError("mass-weighted SONIC motion needs one mass per atom")
    if not np.all(np.isfinite(masses_amu)) or any(mass <= 0.0 for mass in masses_amu):
        raise ValueError("mass-weighted SONIC motion needs finite positive masses")
    masses = np.repeat(np.asarray(masses_amu, dtype=float), 3)
    inverse_sqrt_mass = 1.0 / np.sqrt(masses)
    mass_scaled_b = reference_b * inverse_sqrt_mass[np.newaxis, :]
    return inverse_sqrt_mass[:, np.newaxis] * np.linalg.pinv(
        mass_scaled_b,
        rcond=CONSTANT_B_PINV_RCOND,
    )


def _conditioning_diagnostics(reference_b: np.ndarray) -> tuple[float, np.ndarray]:
    """Condition B after row normalization and estimate each inverse column."""
    row_norms = np.linalg.norm(reference_b, axis=1)
    if np.any(row_norms <= 1.0e-14):
        raise ValueError("SMITH B matrix contains a null SONIC row")
    normalized = reference_b / row_norms[:, np.newaxis]
    singular_values = np.linalg.svd(normalized, compute_uv=False)
    retained = singular_values[singular_values > CONSTANT_B_PINV_RCOND * singular_values[0]]
    condition = float(singular_values[0] / retained[-1]) if retained.size else float("inf")
    inverse = np.linalg.pinv(normalized, rcond=CONSTANT_B_PINV_RCOND)
    indicators = singular_values[0] * np.linalg.norm(inverse, axis=0)
    return condition, np.asarray(indicators, dtype=float)


def _localization_diagnostics(
    direction: np.ndarray,
    atoms: tuple[str, ...],
    condition_indicator: float,
) -> dict[str, object]:
    amplitudes = np.linalg.norm(direction, axis=1)
    maximum_index = int(np.argmax(amplitudes))
    maximum = float(amplitudes[maximum_index])
    significant = tuple(
        index + 1
        for index, amplitude in enumerate(amplitudes)
        if amplitude >= SIGNIFICANT_ATOM_RELATIVE_THRESHOLD * maximum
    )
    weights = amplitudes**2
    denominator = float(np.sum(weights**2))
    participation = float(np.sum(weights) ** 2 / denominator) if denominator > 1.0e-30 else 0.0
    if condition_indicator < 10.0:
        conditioning_status = "GOOD"
    elif condition_indicator < 100.0:
        conditioning_status = "MODERATE"
    else:
        conditioning_status = "ILL_CONDITIONED"
    return {
        "condition_indicator": float(condition_indicator),
        "conditioning_status": conditioning_status,
        "amplification": float(np.linalg.norm(direction)),
        "participation_ratio": participation,
        "significant_atom_count": len(significant),
        "significant_atoms": significant,
        "maximum_atom_index": maximum_index + 1,
        "maximum_atom_symbol": atoms[maximum_index],
    }


def _coordinate_metadata(gic, primitive_by_id: dict[str, object]):
    components = gic.coefficients or ((gic.primitive_id, 1.0),)
    highlighted: list[int] = []
    ring_atoms: tuple[int, ...] = ()
    terms: list[str] = []
    component_primitives: list[object] = []
    for primitive_id, coefficient in components:
        primitive = primitive_by_id.get(primitive_id)
        if primitive is None:
            terms.append(f"{float(coefficient):+.8g} * {primitive_id}: UNKNOWN")
            continue
        component_primitives.append(primitive)
        for atom in primitive.atoms:
            if atom not in highlighted:
                highlighted.append(atom)
        candidate_ring = _ring_atoms_from_refs(primitive.refs)
        if not candidate_ring and str(primitive.function).upper() == "RPCK":
            candidate_ring = tuple(primitive.atoms)
        if candidate_ring and not ring_atoms:
            ring_atoms = candidate_ring
        atoms = ",".join(str(atom) for atom in primitive.atoms)
        terms.append(
            f"{float(coefficient):+.8g} * {primitive_id}: "
            f"{str(primitive.function).upper()}({atoms})"
        )
    ring_family = any(
        token in str(gic.family).upper()
        for token in ("RING", "CYCLIC", "BUTTERFLY", "PSEUDO_CYCLE")
    )
    if not ring_atoms and ring_family and component_primitives:
        if all(
            str(primitive.function).upper() == "A" and len(primitive.atoms) == 3
            for primitive in component_primitives
        ):
            ring_atoms = tuple(
                dict.fromkeys(primitive.atoms[1] for primitive in component_primitives)
            )
        else:
            ring_atoms = tuple(
                dict.fromkeys(
                    atom for primitive in component_primitives for atom in primitive.atoms
                )
            )
    return tuple(highlighted), ring_atoms, tuple(terms)


def _ring_atoms_from_refs(refs: tuple[str, ...]) -> tuple[int, ...]:
    for ref in refs:
        if not str(ref).upper().startswith("RING:"):
            continue
        try:
            return tuple(int(atom) for atom in str(ref).split(":", 1)[1].split("-") if atom)
        except ValueError:
            return ()
    return ()


def _normalize_topology_worker_request(requested: int | str) -> int | str:
    if isinstance(requested, str):
        normalized = requested.strip().lower()
        if normalized == "auto":
            return "auto"
        try:
            return int(normalized)
        except ValueError as exc:
            raise ValueError(
                "SONIC topology workers must be 'auto' or a non-negative integer"
            ) from exc
    return int(requested)


def _resolve_topology_workers(requested: int | str, *, task_count: int) -> int:
    if task_count <= 0:
        return 0
    if requested == "auto":
        return recommended_topology_workers(task_count)
    requested_count = int(requested)
    if requested_count < 0:
        raise ValueError("SONIC topology workers must be zero or positive")
    available = available_topology_workers()
    desired = available if requested_count == 0 else requested_count
    return max(1, min(desired, available, task_count))


def _topology_cache_fingerprint(
    reference: np.ndarray,
    reference_b: np.ndarray,
    cartesian_from_q: np.ndarray,
    *,
    max_atom_displacement_angstrom: float,
    reference_topology: frozenset[tuple[int, int]],
    atomic_numbers: tuple[int, ...],
) -> str:
    digest = hashlib.sha256()
    digest.update(SONIC_TOPOLOGY_CACHE_SCHEMA.encode("ascii"))
    for array in (reference, reference_b, cartesian_from_q):
        canonical = np.ascontiguousarray(array, dtype="<f8")
        digest.update(str(canonical.shape).encode("ascii"))
        digest.update(canonical.tobytes())
    settings = {
        "atomic_numbers": list(atomic_numbers),
        "reference_topology": [list(pair) for pair in sorted(reference_topology)],
        "max_atom_displacement_angstrom": float(max_atom_displacement_angstrom),
        "max_bond_relative_change": DEFAULT_MAX_BOND_RELATIVE_CHANGE,
        "b_pseudoinverse_rcond": CONSTANT_B_PINV_RCOND,
    }
    digest.update(json.dumps(settings, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def _load_topology_cache(path: Path, fingerprint: str) -> dict[str, object]:
    empty: dict[str, object] = {
        "schema": SONIC_TOPOLOGY_CACHE_SCHEMA,
        "fingerprint": fingerprint,
        "entries": {},
    }
    if not path.is_file():
        return empty
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty
    if not isinstance(payload, dict):
        return empty
    if payload.get("schema") != SONIC_TOPOLOGY_CACHE_SCHEMA:
        return empty
    if payload.get("fingerprint") != fingerprint or not isinstance(payload.get("entries"), dict):
        return empty
    return payload


def _write_topology_cache(path: Path, payload: dict[str, object]) -> None:
    atomic_json_write(path, payload)


def _motion_result_cache_record(
    requested_step: float,
    result: _ConstantBMotionTuple,
) -> dict[str, object]:
    return {
        "requested_step": float(requested_step),
        "accepted_step": float(result[2]),
        "status": result[3],
        "connectivity_preserved": bool(result[4]),
        "max_bond_relative_change": float(result[5]),
        "full_topology_checks": int(result[6]),
    }


def _motion_result_from_cache(
    record: object,
    reference: np.ndarray,
    reference_b: np.ndarray,
    cartesian_from_q: np.ndarray,
    coordinate_index: int,
    requested_step: float,
) -> _ConstantBMotionTuple | None:
    if not isinstance(record, dict):
        return None
    try:
        cached_requested = float(record["requested_step"])
        step = float(record["accepted_step"])
        status = str(record["status"])
        connectivity_value = record["connectivity_preserved"]
        bond_change = float(record["max_bond_relative_change"])
        full_topology_checks = int(record["full_topology_checks"])
    except (KeyError, TypeError, ValueError):
        return None
    if cached_requested != float(requested_step):
        return None
    if not isinstance(connectivity_value, bool):
        return None
    connectivity_preserved = connectivity_value
    if (
        not np.isfinite(step)
        or step <= 0.0
        or not np.isfinite(bond_change)
        or full_topology_checks < 2
    ):
        return None
    if status not in {"CONSTANT_B", "CONSTANT_B_REDUCED_STEP"}:
        return None
    direction = np.asarray(cartesian_from_q[:, coordinate_index], dtype=float).reshape(
        reference.shape
    )
    reconstructed: list[_ConstantBMotionResult] = []
    for sign in (-1.0, 1.0):
        coordinates = reference + sign * direction * step
        target_delta = np.zeros(reference_b.shape[0], dtype=float)
        target_delta[coordinate_index] = sign * step
        linearized_delta = reference_b @ (coordinates - reference).reshape(-1)
        reconstructed.append(
            _ConstantBMotionResult(
                coordinates_angstrom=coordinates,
                residual=target_delta - linearized_delta,
            )
        )
    return (
        reconstructed[0],
        reconstructed[1],
        step,
        status,
        connectivity_preserved,
        bond_change,
        full_topology_checks,
    )


def _initialize_motion_worker(
    reference: np.ndarray,
    reference_b: np.ndarray,
    cartesian_from_q: np.ndarray,
    max_atom_displacement_angstrom: float,
    reference_topology: frozenset[tuple[int, int]],
    atomic_numbers: tuple[int, ...],
) -> None:
    global _MOTION_WORKER_CONTEXT
    _MOTION_WORKER_CONTEXT = (
        reference,
        reference_b,
        cartesian_from_q,
        max_atom_displacement_angstrom,
        reference_topology,
        atomic_numbers,
    )


def _constant_b_motion_worker(task: tuple[int, float]) -> _ConstantBMotionTuple:
    if _MOTION_WORKER_CONTEXT is None:
        raise RuntimeError("SMITH topology worker was not initialized")
    (
        reference,
        reference_b,
        cartesian_from_q,
        max_atom_displacement_angstrom,
        reference_topology,
        atomic_numbers,
    ) = _MOTION_WORKER_CONTEXT
    coordinate_index, requested_step = task
    return _constant_b_symmetric_motion(
        reference,
        reference_b,
        cartesian_from_q,
        coordinate_index,
        requested_step,
        max_atom_displacement_angstrom=max_atom_displacement_angstrom,
        reference_topology=reference_topology,
        atomic_numbers=atomic_numbers,
    )


def _constant_b_motion_batch(
    reference: np.ndarray,
    reference_b: np.ndarray,
    cartesian_from_q: np.ndarray,
    tasks: tuple[tuple[int, float], ...],
    *,
    max_atom_displacement_angstrom: float,
    reference_topology: frozenset[tuple[int, int]],
    atomic_numbers: tuple[int, ...],
    workers: int,
    completion_callback: Callable[[int, _ConstantBMotionTuple], None] | None = None,
) -> dict[int, _ConstantBMotionTuple]:
    if not tasks:
        return {}
    if workers == 1:
        results: dict[int, _ConstantBMotionTuple] = {}
        for coordinate_index, requested_step in tasks:
            result = _constant_b_symmetric_motion(
                reference,
                reference_b,
                cartesian_from_q,
                coordinate_index,
                requested_step,
                max_atom_displacement_angstrom=max_atom_displacement_angstrom,
                reference_topology=reference_topology,
                atomic_numbers=atomic_numbers,
            )
            results[coordinate_index] = result
            if completion_callback is not None:
                completion_callback(coordinate_index, result)
        return results

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_motion_worker,
        initargs=(
            reference,
            reference_b,
            cartesian_from_q,
            max_atom_displacement_angstrom,
            reference_topology,
            atomic_numbers,
        ),
    ) as executor:
        futures = {executor.submit(_constant_b_motion_worker, task): task[0] for task in tasks}
        results = {}
        for future in as_completed(futures):
            coordinate_index = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                raise RuntimeError(
                    f"ORACLE worker failed while validating SONIC {coordinate_index + 1}: {exc}"
                ) from exc
            results[coordinate_index] = result
            if completion_callback is not None:
                completion_callback(coordinate_index, result)
    return results


def _constant_b_symmetric_motion(
    reference: np.ndarray,
    reference_b: np.ndarray,
    cartesian_from_q: np.ndarray,
    coordinate_index: int,
    requested_step: float,
    *,
    max_atom_displacement_angstrom: float,
    reference_topology: frozenset[tuple[int, int]],
    atomic_numbers: tuple[int, ...],
) -> _ConstantBMotionTuple:
    direction = np.asarray(cartesian_from_q[:, coordinate_index], dtype=float).reshape(
        reference.shape
    )
    direction_scale = float(np.max(np.linalg.norm(direction, axis=1), initial=0.0))
    if direction_scale <= 1.0e-14:
        raise ValueError(f"SONIC coordinate {coordinate_index + 1} has a null B-pseudoinverse row")

    step = min(
        float(requested_step),
        0.95 * float(max_atom_displacement_angstrom) / direction_scale,
    )
    reduced = step < float(requested_step) * (1.0 - 1.0e-12)
    full_topology_checks = 0

    for attempt in range(13):
        minus_coordinates = reference - direction * step
        plus_coordinates = reference + direction * step
        try:
            minus_topology = _topology_signature(minus_coordinates, atomic_numbers)
        except Exception as exc:
            raise RuntimeError(
                f"ORACLE perception failed for SONIC {coordinate_index + 1}, "
                f"frame -Delta, attempt {attempt + 1}"
            ) from exc
        try:
            plus_topology = _topology_signature(plus_coordinates, atomic_numbers)
        except Exception as exc:
            raise RuntimeError(
                f"ORACLE perception failed for SONIC {coordinate_index + 1}, "
                f"frame +Delta, attempt {attempt + 1}"
            ) from exc
        full_topology_checks += 2
        connectivity_preserved = (
            minus_topology == reference_topology and plus_topology == reference_topology
        )
        bond_change = _max_bond_relative_change(
            reference,
            minus_coordinates,
            plus_coordinates,
            reference_topology,
        )
        if connectivity_preserved and bond_change < DEFAULT_MAX_BOND_RELATIVE_CHANGE:
            break
        if attempt == 12:
            raise RuntimeError(
                f"SONIC coordinate {coordinate_index + 1} cannot produce a safe constant-B motion"
            )
        step *= 0.5
        reduced = True

    results = []
    for sign, coordinates in ((-1.0, minus_coordinates), (1.0, plus_coordinates)):
        target_delta = np.zeros(reference_b.shape[0], dtype=float)
        target_delta[coordinate_index] = sign * step
        linearized_delta = reference_b @ (coordinates - reference).reshape(-1)
        results.append(
            _ConstantBMotionResult(
                coordinates_angstrom=coordinates,
                residual=target_delta - linearized_delta,
            )
        )
    return (
        results[0],
        results[1],
        step,
        "CONSTANT_B_REDUCED_STEP" if reduced else "CONSTANT_B",
        connectivity_preserved,
        bond_change,
        full_topology_checks,
    )


def _coordinate_step(
    family: str,
    *,
    distance_step_angstrom: float,
    angle_step_radian: float,
    ring_puckering_step_radian: float,
) -> tuple[str, float]:
    normalized = str(family).strip().upper()
    if normalized in {
        "STRETCH",
        "H_BOND_DISTANCE",
        "FRAGMENT_CENTER_DISTANCE",
        "FRAGMENT_CENTER_ATOM_DISTANCE",
        "CENTER_ATOM_DISTANCE",
        "FRAG_TRANSLATION",
    }:
        return "angstrom", distance_step_angstrom
    if normalized in {"RING_PUCKERING", "RING_PUCKER_COMPONENT", "PSEUDO_CYCLE_TORSION"}:
        return "radian", ring_puckering_step_radian
    return "radian", angle_step_radian


def _trajectory_lines(
    atoms: tuple[str, ...],
    minus: np.ndarray,
    reference: np.ndarray,
    plus: np.ndarray,
    *,
    label: str,
    step: float,
    unit: str,
) -> list[str]:
    lines: list[str] = []
    for phase, coordinates in (("minus", minus), ("reference", reference), ("plus", plus)):
        lines.append(str(len(atoms)))
        lines.append(f"SMITH SONIC {label} phase={phase} step={step:.10g} unit={unit}")
        lines.extend(
            f"{atom:2s} {x:16.10f} {y:16.10f} {z:16.10f}"
            for atom, (x, y, z) in zip(atoms, coordinates, strict=True)
        )
    return lines


def _vector_lines(
    atoms: tuple[str, ...],
    minus: np.ndarray,
    reference: np.ndarray,
    plus: np.ndarray,
    vector: np.ndarray,
) -> list[str]:
    lines = [
        "atom\tsymbol\tminus_x\tminus_y\tminus_z\treference_x\treference_y\treference_z"
        "\tplus_x\tplus_y\tplus_z\tdelta_x\tdelta_y\tdelta_z"
    ]
    for index, (atom, xyz_minus, xyz_reference, xyz_plus, delta) in enumerate(
        zip(atoms, minus, reference, plus, vector, strict=True),
        start=1,
    ):
        values = (*xyz_minus, *xyz_reference, *xyz_plus, *delta)
        lines.append(f"{index}\t{atom}\t" + "\t".join(f"{float(value):.12g}" for value in values))
    return lines


def _motion_record(motion: SonicCoordinateMotion, output: Path) -> dict[str, object]:
    return {
        "index": motion.index,
        "identifier": motion.identifier,
        "name": motion.name,
        "family": motion.family,
        "irrep": motion.irrep,
        "unit": motion.unit,
        "step": motion.step,
        "status": motion.status,
        "minus_realized_displacement": motion.minus_realized_displacement,
        "plus_realized_displacement": motion.plus_realized_displacement,
        "minus_residual_norm": motion.minus_residual_norm,
        "plus_residual_norm": motion.plus_residual_norm,
        "max_atom_displacement_angstrom": motion.max_atom_displacement_angstrom,
        "max_bond_relative_change": motion.max_bond_relative_change,
        "connectivity_preserved": motion.connectivity_preserved,
        "orientation_preserved": motion.orientation_preserved,
        "minimum_alignment_determinant": motion.minimum_alignment_determinant,
        "cartesian_metric": motion.cartesian_metric,
        "normalized_b_condition_number": motion.normalized_b_condition_number,
        "coordinate_condition_indicator": motion.coordinate_condition_indicator,
        "conditioning_status": motion.conditioning_status,
        "cartesian_amplification_angstrom_per_unit": (
            motion.cartesian_amplification_angstrom_per_unit
        ),
        "participation_ratio": motion.participation_ratio,
        "significant_atom_count": motion.significant_atom_count,
        "significant_atoms": list(motion.significant_atoms),
        "maximum_atom_index": motion.maximum_atom_index,
        "maximum_atom_symbol": motion.maximum_atom_symbol,
        "highlighted_atoms": list(motion.highlighted_atoms),
        "ring_atoms": list(motion.ring_atoms),
        "ring_phase": motion.irrep,
        "component_terms": list(motion.component_terms),
        "local_domain": motion.local_domain,
        "local_group": motion.local_group,
        "local_irrep": motion.local_irrep,
        "locally_totally_symmetric": motion.locally_totally_symmetric,
        "globally_totally_symmetric": motion.globally_totally_symmetric,
        "full_topology_checks": motion.full_topology_checks,
        "cartesian_vector_norm_angstrom": float(np.linalg.norm(motion.cartesian_vector_angstrom)),
        "cartesian_vector_angstrom": motion.cartesian_vector_angstrom.tolist(),
        "trajectory": str(motion.trajectory_path.relative_to(output)),
        "vector": str(motion.vector_path.relative_to(output)),
    }


def _selected_coordinate_indices(
    coordinate_count: int,
    requested: Sequence[int] | None,
) -> tuple[int, ...]:
    if requested is None:
        return tuple(range(coordinate_count))
    selected: list[int] = []
    for value in requested:
        index = int(value) - 1
        if index < 0 or index >= coordinate_count:
            raise ValueError(f"SONIC coordinate index {value} is outside 1..{coordinate_count}")
        if index not in selected:
            selected.append(index)
    if not selected:
        raise ValueError("at least one SONIC coordinate index must be selected")
    return tuple(selected)


def _topology_signature(
    coordinates: np.ndarray,
    atomic_numbers: tuple[int, ...],
) -> frozenset[tuple[int, int]]:
    try:
        _continuous, discrete, _rings, _synthons, _aromaticity = build_topology_objects(
            coordinates,
            atomic_numbers,
        )
    except ValueError:
        return frozenset()
    return frozenset(tuple(sorted((int(left), int(right)))) for left, right in discrete.bonds)


def _max_bond_relative_change(
    reference: np.ndarray,
    minus: np.ndarray,
    plus: np.ndarray,
    bonds: frozenset[tuple[int, int]],
) -> float:
    maximum = 0.0
    for left, right in bonds:
        base = float(np.linalg.norm(reference[left] - reference[right]))
        if base <= 1.0e-12:
            continue
        for frame in (minus, plus):
            distance = float(np.linalg.norm(frame[left] - frame[right]))
            maximum = max(maximum, abs(distance - base) / base)
    return maximum


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return normalized.strip("._") or "SONIC"
