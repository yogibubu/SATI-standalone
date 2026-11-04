"""Stable public API for ORACLE molecular perception."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import json
import os
import platform
import sys
import subprocess
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from matrix_chem import (
    PrimitiveCoordinateContract,
    SymmetryThresholds,
    preprocess_to_enriched_xyz,
    read_enriched_xyz,
    read_molecular_symmetry,
    read_primitive_contract,
    read_symmetry_thresholds,
    topology_snapshot_from_xyzin,
    validate_enriched_molecule,
    write_topology_snapshot,
    write_validation_section,
)
from matrix_core import read_sectioned_lines, section_content, sha256_file
from .config import OracleConfig, load_oracle_config
from ._version import __version__
from .atom_classes import classify_synthon_atoms
from .atom_typing import gaff_translation_from_snapshot
from .scope import oracle_scope_contract
from .synthon_fingerprint import build_synthon_fingerprint
from .cache import cache_key, write_cached


ORACLE_REPORT_SCHEMA = "matrix.oracle.analysis.v2"
ORACLE_BATCH_SCHEMA = "matrix.oracle.batch.v1"
SUPPORTED_INPUT_FORMATS = (
    ("xyz", ".xyz or extensionless XYZ", "built-in"),
    ("enriched_xyz", "MATRIX enriched XYZ / xyzin", "built-in"),
    ("mol/sdf", "MDL MOL or SDF", "built-in"),
    ("mol2", "Tripos MOL2", "built-in"),
    ("zmatrix", ".zmat or .zmt", "built-in"),
    ("smiles", ".smi or .smiles", "matrix-oracle[smiles]"),
    ("gaussian", ".gjf, .com, .log, .out, .fchk", "matrix-oracle[formats]"),
    ("molpro", "Molpro output", "matrix-oracle[formats]"),
    ("mrcc", "MRCC output", "matrix-oracle[formats]"),
    ("orca", "ORCA output", "matrix-oracle[formats]"),
)


@dataclass(frozen=True)
class OracleAnalysis:
    output: Path
    report: Path | None
    human_report: Path | None
    topology_snapshot: Path | None
    status: str
    atom_count: int
    point_group: str
    symmetry_operation_count: int
    bond_count: int
    ring_count: int
    aromatic_atom_count: int
    synthon_count: int
    topology_sha256: str
    primitive_count: int
    primitive_b_matrix_rank: int
    primitive_b_matrix_sha256: str
    cartesian_symmetry_status: str
    symmetrization_required_threshold_angstrom: float | None
    cartesian_symmetrization_decision: str
    proposed_point_group: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("output", "report", "human_report", "topology_snapshot"):
            if payload[key] is not None:
                payload[key] = str(payload[key])
        return payload


@dataclass(frozen=True)
class OracleAnalysisRequest:
    """One deterministic, process-safe ORACLE perception request."""

    source: Path
    output: Path
    source_kind: str = "auto"
    report: Path | None = None
    human_report: Path | None = None
    topology_snapshot: Path | None = None
    config: Path | None = None
    validate: bool = True
    cartesian_symmetrization: str = "inspect"


def analyze_structure(
    source: Path,
    output: Path,
    *,
    source_kind: str = "auto",
    config: OracleConfig | Path | None = None,
    symmetry_distance: float | None = None,
    symmetry_inertia: float | None = None,
    max_rotation_order: int | None = None,
    report: Path | None = None,
    human_report: Path | None = None,
    topology_snapshot: Path | None = None,
    validate: bool = True,
    cartesian_symmetrization: str = "inspect",
    bond_order_model: str = "auto",
) -> OracleAnalysis:
    """Run ORACLE perception and write a versioned enriched molecular state."""
    settings = _coerce_config(config)
    thresholds = SymmetryThresholds(
        distance_angstrom=(
            settings.symmetry.distance_angstrom
            if symmetry_distance is None
            else _positive("symmetry_distance", symmetry_distance)
        ),
        inertia_relative=(
            settings.symmetry.inertia_relative
            if symmetry_inertia is None
            else _positive("symmetry_inertia", symmetry_inertia)
        ),
        max_rotation_order=(
            settings.symmetry.max_rotation_order
            if max_rotation_order is None
            else _positive_integer("max_rotation_order", max_rotation_order)
        ),
    )
    source_path = Path(source).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    population_options: dict[str, object] = {}
    normalized_bond_order_model = str(bond_order_model).strip().casefold()
    if normalized_bond_order_model not in {
        "auto",
        "pauling",
        "matrix-gfn2",
        "integer-connectivity",
    }:
        raise ValueError(f"unsupported ORACLE bond-order model: {bond_order_model}")
    if normalized_bond_order_model == "matrix-gfn2":
        from matrix_chem import read_geometry_with_kind
        from matrix_gfn2 import GFN2Calculator
        from .gfn2_symmetry import freeze_matrix_gfn2_symmetry

        population_geometry = read_geometry_with_kind(source_path, source_kind)
        calculator = GFN2Calculator(
            tuple(population_geometry.atoms),
            charge=0 if population_geometry.charge is None else int(population_geometry.charge),
            multiplicity=(
                1 if population_geometry.multiplicity is None else int(population_geometry.multiplicity)
            ),
            symmetry_contract=freeze_matrix_gfn2_symmetry(
                population_geometry.atoms,
                population_geometry.coordinates_angstrom,
                thresholds=settings.symmetry,
            ),
        )
        population = calculator.evaluate(
            population_geometry.coordinates_angstrom, compute_cm5=True
        )
        charges = (
            population.cm5_charges_electron
            if population.cm5_charges_electron is not None
            else population.charges_electron
        )
        population_options = {
            "external_charges": {
                index: float(value) for index, value in enumerate(charges)
            },
            "bond_order_overrides": {
                (left, right): float(population.bond_orders[left, right])
                for left in range(population_geometry.natoms)
                for right in range(left + 1, population_geometry.natoms)
            },
            "charge_source": "ORACLE autonomous MATRIX-GFN2 CM5 charges",
            "bond_order_source": "ORACLE autonomous MATRIX-GFN2 bond orders",
            "electronic_connectivity": True,
        }
    elif normalized_bond_order_model == "integer-connectivity":
        from matrix_chem import build_topology_objects, read_geometry_with_kind
        from matrix_chem.topology.elements import atomic_number

        population_geometry = read_geometry_with_kind(source_path, source_kind)
        numbers = tuple(int(atomic_number(atom) or 0) for atom in population_geometry.atoms)
        _continuous, graph, _rings, synthons, _aromaticity = build_topology_objects(
            population_geometry.coordinates_angstrom,
            numbers,
        )
        edges = set(graph.bonds) | set(graph.transitional_contacts)
        coordinates = np.asarray(population_geometry.coordinates_angstrom, dtype=float)
        heavy = tuple(index for index, number in enumerate(numbers) if number != 1)
        for hydrogen, number in enumerate(numbers):
            if number != 1 or any(hydrogen in edge for edge in edges) or not heavy:
                continue
            partner = min(
                heavy,
                key=lambda index: (float(np.linalg.norm(coordinates[hydrogen] - coordinates[index])), index),
            )
            edges.add(tuple(sorted((hydrogen, partner))))
        orders = _integer_orders_from_connectivity(
            tuple(sorted(edges)),
            numbers,
            coordinates,
        )
        population_options = {
            "external_charges": {
                index: float(synthons.charge(index)) for index in range(len(numbers))
            },
            "bond_order_overrides": orders,
            "charge_source": "ORACLE electronegativity estimate",
            "bond_order_source": "ORACLE integer orders from frozen connectivity",
            "electronic_connectivity": True,
        }
    result = preprocess_to_enriched_xyz(
        source_path,
        output_path,
        source_kind=source_kind,
        symmetry_thresholds=thresholds,
        cartesian_symmetrization=cartesian_symmetrization,
        **population_options,
    )
    from matrix_fragments import write_fragment_build_section

    write_fragment_build_section(output_path)
    # Keep special ORACLE primitives (metal--ring eta_n, ring centres and
    # related atom--centre coordinates) in the serialized input consumed by
    # SMITH.  Without this section the SONIC builder sees only ordinary
    # valence primitives even when ORACLE has recognized an eta interaction.
    from matrix_fragments import write_interaction_center_section

    write_interaction_center_section(output_path)
    validation = (
        write_validation_section(output_path)
        if validate
        else validate_enriched_molecule(output_path)
    )
    symmetry = read_molecular_symmetry(output_path)
    primitive_contract: PrimitiveCoordinateContract = read_primitive_contract(output_path)
    snapshot = topology_snapshot_from_xyzin(
        output_path,
        source=str(source_path),
    )
    synthon_data = _synthon_data(output_path)
    atom_classes = classify_synthon_atoms(synthon_data["atoms"])
    gaff_translation = gaff_translation_from_snapshot(snapshot, synthon_data["atoms"])
    symmetry_metadata = _section_metadata(output_path, "SYMMETRY")

    snapshot_path = None
    if topology_snapshot is not None:
        snapshot_path = Path(topology_snapshot).expanduser().resolve()
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        write_topology_snapshot(snapshot_path, output_path)

    report_path = None
    human_report_path = None
    if report is not None or human_report is not None:
        report_payload = _analysis_report_payload(
            output_path,
            source=source_path,
            source_kind=source_kind,
            thresholds=thresholds,
            validation=validation,
            symmetry=symmetry,
            primitive_contract=primitive_contract,
            snapshot=snapshot,
            synthon_data=synthon_data,
            atom_classes=atom_classes.to_dict(),
            gaff_translation=gaff_translation.to_dict(),
            symmetry_metadata=symmetry_metadata,
            input_geometry=result.input_geometry,
            accepted_geometry=result.geometry,
        )
        if settings.paths.cache_dir is not None:
            write_cached(settings.paths.cache_dir, str(report_payload["cache_key"]), report_payload)
    if report is not None:
        report_path = Path(report).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if human_report is not None:
        human_report_path = Path(human_report).expanduser().resolve()
        human_report_path.parent.mkdir(parents=True, exist_ok=True)
        human_report_path.write_text(
            "\n".join(oracle_human_report_lines(report_payload)) + "\n",
            encoding="utf-8",
        )

    analysis_status = (
        "AWAITING_CARTESIAN_SYMMETRY_CONFIRMATION"
        if result.cartesian_symmetry_status == "QUASI_SYMMETRY"
        and str(cartesian_symmetrization).strip().casefold() == "inspect"
        else validation.status
    )
    return OracleAnalysis(
        output=output_path,
        report=report_path,
        human_report=human_report_path,
        topology_snapshot=snapshot_path,
        status=analysis_status,
        atom_count=result.geometry.natoms,
        point_group=symmetry.point_group,
        symmetry_operation_count=len(symmetry.operations),
        bond_count=int(snapshot["bond_count"]),
        ring_count=int(snapshot["ring_count"]),
        aromatic_atom_count=len(snapshot["aromatic_atoms"]),
        synthon_count=len(synthon_data["atoms"]),
        topology_sha256=str(snapshot["topology_sha256"]),
        primitive_count=len(primitive_contract.primitives),
        primitive_b_matrix_rank=primitive_contract.b_matrix_rank,
        primitive_b_matrix_sha256=primitive_contract.b_matrix_sha256,
        cartesian_symmetry_status=result.cartesian_symmetry_status,
        symmetrization_required_threshold_angstrom=(
            result.symmetrization_required_threshold_angstrom
        ),
        cartesian_symmetrization_decision=result.cartesian_symmetrization_decision,
        proposed_point_group=result.proposed_point_group,
    )


def _integer_orders_from_connectivity(
    bonds: Sequence[tuple[int, int]],
    atomic_numbers: Sequence[int],
    coordinates_angstrom: np.ndarray,
) -> dict[tuple[int, int], float]:
    """Assign deterministic 1/2/3 orders from connectivity, never bond lengths alone."""

    targets = {1: 1, 5: 3, 6: 4, 7: 3, 8: 2, 9: 1, 14: 4, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}
    canonical = tuple(sorted({tuple(sorted(map(int, pair))) for pair in bonds}))
    orders = {pair: 1.0 for pair in canonical}
    degree = [0] * len(atomic_numbers)
    for left, right in canonical:
        degree[left] += 1
        degree[right] += 1
    deficit = [max(0, targets.get(int(number), degree[index]) - degree[index]) for index, number in enumerate(atomic_numbers)]
    distances = {
        pair: float(np.linalg.norm(coordinates_angstrom[pair[0]] - coordinates_angstrom[pair[1]]))
        for pair in canonical
    }
    while True:
        candidates = [
            pair
            for pair in canonical
            if deficit[pair[0]] > 0 and deficit[pair[1]] > 0 and orders[pair] < 3.0
        ]
        if not candidates:
            break
        pair = min(candidates, key=lambda item: (distances[item], item))
        orders[pair] += 1.0
        deficit[pair[0]] -= 1
        deficit[pair[1]] -= 1
    return orders


def analyze_structures(
    requests: Sequence[OracleAnalysisRequest],
    *,
    workers: int = 0,
) -> tuple[OracleAnalysis, ...]:
    """Run independent perceptions in processes while preserving input order."""

    jobs = tuple(requests)
    if not jobs:
        return ()
    outputs = [Path(job.output).expanduser().resolve() for job in jobs]
    if len(outputs) != len(set(outputs)):
        raise ValueError("parallel ORACLE requests must have distinct output files")
    resolved = _resolved_workers(workers, len(jobs))
    if resolved == 1:
        return tuple(_run_analysis_request(job) for job in jobs)
    with ProcessPoolExecutor(max_workers=resolved) as executor:
        return tuple(executor.map(_run_analysis_request, jobs))


def write_oracle_analysis_reports(
    path: Path,
    *,
    json_output: Path | None = None,
    human_output: Path | None = None,
) -> dict[str, Any]:
    """Report an existing frozen ORACLE state without reperceiving it."""

    target = Path(path).expanduser().resolve()
    geometry = read_enriched_xyz(target)
    thresholds = read_symmetry_thresholds(target)
    validation = validate_enriched_molecule(target)
    symmetry = read_molecular_symmetry(target)
    primitive_contract = read_primitive_contract(target)
    snapshot = topology_snapshot_from_xyzin(target, source=str(target))
    synthon_data = _synthon_data(target)
    source_metadata = _section_metadata(target, "SOURCE")
    source_value = source_metadata.get("PATH", "")
    source = Path(source_value).expanduser() if source_value else target
    payload = _analysis_report_payload(
        target,
        source=source,
        source_kind=source_metadata.get("KIND", "enriched_xyz"),
        thresholds=thresholds,
        validation=validation,
        symmetry=symmetry,
        primitive_contract=primitive_contract,
        snapshot=snapshot,
        synthon_data=synthon_data,
        atom_classes=classify_synthon_atoms(synthon_data["atoms"]).to_dict(),
        gaff_translation=gaff_translation_from_snapshot(
            snapshot, synthon_data["atoms"]
        ).to_dict(),
        symmetry_metadata=_section_metadata(target, "SYMMETRY"),
        atom_count=geometry.natoms,
        accepted_geometry=geometry,
    )
    if json_output is not None:
        output = Path(json_output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if human_output is not None:
        output = Path(human_output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(oracle_human_report_lines(payload)) + "\n", encoding="utf-8")
    return payload


def oracle_human_report_lines(payload: dict[str, Any]) -> list[str]:
    """Render the complete ORACLE contract for a human reader."""

    geometry = payload["geometry"]
    topology = payload["topology"]
    primitives = payload["primitive_coordinates"]
    descriptors = payload["continuous_descriptors"]["atoms"]
    classes = payload["synthon_atom_classes"]["classes"]
    pi = payload.get("permutation_inversion")
    lines = [
        "ORACLE MOLECULAR PERCEPTION REPORT",
        "==================================",
        f"Schema: {payload['schema']}",
        f"ORACLE version: {payload['oracle_version']}",
        f"Source: {payload['source']}",
        f"Enriched state: {payload['output']}",
        f"Validation: {payload['validation']['status']}",
        "",
        "OWNERSHIP BOUNDARY",
        "------------------",
        "ORACLE owns: perception, symmetry, synthons/classes, redundant PIC/B, L1-to-PL1.",
        "SMITH owns SONIC; LINK owns optimization/scans; MORPHEUS owns vibrational symmetry;",
        "ARCHITECT owns Hessian reduction and ZAFF; TRINITY owns multilevel derivatives.",
        "",
        "GEOMETRY AND SYMMETRY",
        "---------------------",
        f"Atoms: {geometry['atom_count']}",
        f"Point group: {geometry['point_group']}",
        f"Proposed point group: {geometry['proposed_point_group']}",
        f"Operations: {geometry['symmetry_operation_count']}",
        "PI group order: " + (
            str(pi["order"]) if pi and pi["order"] is not None
            else "UNEXPANDED (exact generators retained)" if pi else "NOT_AVAILABLE"
        ),
        f"Certified rigid cyclic rotor generators: {len(pi['rotors']) if pi else 0}",
        f"Maximum operation residual / A: {geometry['symmetry_max_deviation_angstrom']:.6e}",
        f"Input maximum operation residual / A: {geometry['input_symmetry_max_deviation_angstrom']:.6e}",
        f"Projection: {geometry['projection_status']}",
        f"Cartesian decision: {geometry['cartesian_symmetrization_decision']}",
        f"Required symmetrization threshold / A: {geometry['symmetrization_required_threshold_angstrom'] if geometry['symmetrization_required_threshold_angstrom'] is not None else 'NOT_REQUIRED'}",
        "",
        "TOPOLOGY",
        "--------",
        f"Bonds: {topology['bond_count']}    Fragments: {topology['fragment_count']}    Rings: {topology['ring_count']}",
        f"Aromatic atoms: {_csv(topology['aromatic_atoms'])}",
        f"Topology SHA256: {topology['topology_sha256']}",
        "Bond components (atoms, total, sigma, pi, pi-pi):",
    ]
    for row in topology["bond_order_components"]:
        lines.append(
                f"  {_csv(row['atoms']):>9s} {row['bond_order']:10.6f} "
            f"{row['sigma']:10.6f} {row['pi']:10.6f} {row['pi_pi']:10.6f}"
        )
    lines.extend(("", "RINGS", "-----"))
    lines.extend(
        f"  {ring['index']:3d} size={ring['size']} atoms={_csv(ring['atoms'])}"
        for ring in topology["rings"]
    )
    if not topology["rings"]:
        lines.append("  NONE")
    lines.extend(
        (
            "",
            "CONTINUOUS ATOM DESCRIPTORS",
            "---------------------------",
            " atom element       Zeff     charge  covalency      deloc     strain     sigma        pi     pi-pi",
        )
    )
    for atom in descriptors:
        lines.append(
            f" {atom['atom']:4d} {str(atom['element']):>7s} "
            f"{atom['z_eff']:10.6f} {atom['charge']:10.6f} {atom['covalency']:10.6f} "
            f"{atom['delocalization']:10.6f} {atom['strain']:10.6f} "
            f"{atom['sigma_index']:10.6f} {atom['pi_index']:9.6f} {atom['pi_pi_index']:9.6f}"
        )
    lines.extend(("", "SYNTHON ATOM CLASSES", "--------------------"))
    for item in classes:
        lines.append(
            f"  {item['identifier']:<8s} element={item['element']:<3s} atoms={_csv(item['atoms'])}"
        )
    lines.extend(
        (
            "",
            "REDUNDANT PRIMITIVE COORDINATES",
            "--------------------------------",
            f"Count: {primitives['count']}    B rank: {primitives['b_matrix_rank']}/{primitives['b_matrix_columns']}",
            f"B SHA256: {primitives['b_matrix_sha256']}",
            " id     kind             coordinate                 reference",
        )
    )
    for item in primitives["definitions"]:
        lines.append(
            f" {item['identifier']:<6s} {item['kind']:<16.16s} "
            f"{item['label']:<26.26s} {item['reference_value']:13.7f}"
        )
    return lines


def _analysis_report_payload(
    output: Path,
    *,
    source: Path,
    source_kind: str,
    thresholds: SymmetryThresholds,
    validation: Any,
    symmetry: Any,
    primitive_contract: PrimitiveCoordinateContract,
    snapshot: dict[str, Any],
    synthon_data: dict[str, Any],
    atom_classes: dict[str, Any],
    gaff_translation: dict[str, Any],
    symmetry_metadata: dict[str, str],
    atom_count: int | None = None,
    input_geometry: Any | None = None,
    accepted_geometry: Any | None = None,
) -> dict[str, Any]:
    source_path = Path(source)
    bond_order_by_atoms = {
        tuple(item["atoms"]): float(item["value"])
        for item in snapshot.get("bond_orders", ())
    }
    fingerprint = build_synthon_fingerprint(
        synthon_data["atoms"],
        bonds=tuple(
            (
                int(left),
                int(right),
                bond_order_by_atoms.get((left, right), 1.0),
            )
            for left, right in snapshot.get("bonds", ())
        ),
    )
    try:
        git_commit = subprocess.check_output(("git", "rev-parse", "HEAD"), text=True, stderr=subprocess.DEVNULL).strip()
        git_branch = subprocess.check_output(("git", "branch", "--show-current"), text=True, stderr=subprocess.DEVNULL).strip()
        git_dirty = bool(subprocess.run(("git", "diff", "--quiet"), check=False).returncode)
    except (OSError, subprocess.CalledProcessError):
        git_commit, git_branch, git_dirty = None, None, None
    pi_payload = None
    pi_metadata = _section_metadata(output, "PERMUTATION_INVERSION")
    if pi_metadata:
        from matrix_chem import read_permutation_inversion_symmetry

        pi = read_permutation_inversion_symmetry(output)
        pi_payload = {
            "schema": pi_metadata["SCHEMA"],
            "scope": pi_metadata["SCOPE"],
            "indexing": "ZERO_BASED",
            "order": pi.order,
            "generator_sources": list(pi.generator_sources),
            "nuclear_labels": list(pi.nuclear_labels),
            "rotors": [asdict(rotor) for rotor in pi.rotors],
            "diagnostics": list(pi.rotor_diagnostics),
            "tolerance_angstrom": pi.tolerance_angstrom,
        }
    return {
        "schema": ORACLE_REPORT_SCHEMA,
        "oracle_version": oracle_version(),
        "scope": oracle_scope_contract(),
        "source": str(source_path),
        "source_sha256": sha256_file(source_path) if source_path.is_file() else None,
        "output": str(output),
        "output_sha256": sha256_file(output),
        "source_kind": source_kind,
        "cache_key": cache_key(str(source_path), source_kind, thresholds.distance_angstrom,
                                thresholds.inertia_relative, thresholds.max_rotation_order),
        "provenance": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "implementation": platform.python_implementation(),
            "backend": "matrix-chem-reference",
            "git_commit": git_commit, "git_branch": git_branch, "git_dirty": git_dirty,
        },
        "thresholds": {
            "distance_angstrom": thresholds.distance_angstrom,
            "inertia_relative": thresholds.inertia_relative,
            "max_rotation_order": thresholds.max_rotation_order,
        },
        "geometry": {
            "atom_count": int(atom_count if atom_count is not None else len(synthon_data["atoms"])),
            "point_group": symmetry.point_group,
            "proposed_point_group": symmetry_metadata.get(
                "PROPOSED_POINT_GROUP", symmetry.point_group
            ),
            "symmetry_operation_count": len(symmetry.operations),
            "symmetry_max_deviation_angstrom": symmetry.max_deviation,
            "symmetry_mean_deviation_angstrom": symmetry.mean_deviation,
            "projection_status": symmetry_metadata.get("CARTESIAN_PROJECTION_STATUS", "UNKNOWN"),
            "projection_max_displacement_angstrom": _optional_float(
                symmetry_metadata.get("CARTESIAN_PROJECTION_MAX_DISPLACEMENT_ANGSTROM")
            ),
            "projection_rms_displacement_angstrom": _optional_float(
                symmetry_metadata.get("CARTESIAN_PROJECTION_RMS_DISPLACEMENT_ANGSTROM")
            ),
            "cartesian_symmetry_status": symmetry_metadata.get(
                "CARTESIAN_SYMMETRY_STATUS", "UNKNOWN"
            ),
            "symmetrization_required_threshold_angstrom": _optional_float(
                symmetry_metadata.get(
                    "CARTESIAN_SYMMETRIZATION_REQUIRED_THRESHOLD_ANGSTROM"
                )
            ),
            "input_symmetry_max_deviation_angstrom": _optional_float(
                symmetry_metadata.get("INPUT_MAX_OPERATION_DEVIATION_ANGSTROM")
            )
            if symmetry_metadata.get("INPUT_MAX_OPERATION_DEVIATION_ANGSTROM")
            is not None
            else float(symmetry.max_deviation),
            "cartesian_symmetrization_decision": symmetry_metadata.get(
                "CARTESIAN_SYMMETRIZATION_DECISION", "UNKNOWN"
            ),
            "input_cartesian_geometry": _cartesian_geometry_payload(input_geometry),
            "accepted_cartesian_geometry": _cartesian_geometry_payload(
                accepted_geometry
            ),
        },
        "topology": snapshot,
        "permutation_inversion": pi_payload,
        "primitive_coordinates": {
            "schema": primitive_contract.schema,
            "owner": "ORACLE",
            "count": len(primitive_contract.primitives),
            "b_matrix_rank": primitive_contract.b_matrix_rank,
            "b_matrix_columns": primitive_contract.b_matrix_columns,
            "b_matrix_sha256": primitive_contract.b_matrix_sha256,
            "definitions": [
                {
                    "identifier": f"P{index:04d}",
                    "kind": primitive.kind,
                    "label": primitive.label,
                    "atoms": [atom + 1 for atom in primitive.atoms],
                    "mode": primitive.mode,
                    "reference_value": float(reference),
                }
                for index, (primitive, reference) in enumerate(
                    zip(
                        primitive_contract.primitives,
                        primitive_contract.reference_values,
                        strict=True,
                    ),
                    start=1,
                )
            ],
        },
        "continuous_descriptors": synthon_data,
        "synthon_atom_classes": atom_classes,
        "synthon_fingerprint": fingerprint.to_dict(),
        "gaff_atom_types": gaff_translation,
        "validation": {
            "status": validation.status,
            "messages": [asdict(message) for message in validation.messages],
        },
    }


def _run_analysis_request(request: OracleAnalysisRequest) -> OracleAnalysis:
    return analyze_structure(
        request.source,
        request.output,
        source_kind=request.source_kind,
        config=request.config,
        report=request.report,
        human_report=request.human_report,
        topology_snapshot=request.topology_snapshot,
        validate=request.validate,
        cartesian_symmetrization=request.cartesian_symmetrization,
    )


def _resolved_workers(requested: int, count: int) -> int:
    if requested < 0:
        raise ValueError("ORACLE worker count must be non-negative")
    available = max(1, int(os.cpu_count() or 1))
    cap = max(0, int(os.environ.get("ORACLE_MAX_WORKERS", "0") or 0))
    limit = cap or available
    return min(count, limit if requested == 0 else max(1, min(int(requested), limit)))


def _csv(values: Sequence[Any]) -> str:
    return ",".join(str(value) for value in values) if values else "NONE"


def oracle_version() -> str:
    return __version__


def _coerce_config(config: OracleConfig | Path | None) -> OracleConfig:
    if isinstance(config, OracleConfig):
        return config
    return load_oracle_config(config)


def _positive(name: str, value: float) -> float:
    converted = float(value)
    if converted <= 0.0:
        raise ValueError(f"{name} must be positive")
    return converted


def _positive_integer(name: str, value: int) -> int:
    converted = int(value)
    if converted < 1:
        raise ValueError(f"{name} must be at least one")
    return converted


def _section_metadata(path: Path, name: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in section_content(read_sectioned_lines(path), name):
        text = line.strip()
        if not text or text.startswith("["):
            continue
        fields = text.split(maxsplit=1)
        if len(fields) == 2:
            values[fields[0].upper()] = fields[1]
    return values


def _synthon_data(path: Path) -> dict[str, Any]:
    content = section_content(read_sectioned_lines(path), "SYNTHONS")
    metadata = _section_metadata(path, "SYNTHONS")
    columns = _section_columns(content)
    positions = {name: index for index, name in enumerate(columns)}
    atoms: list[dict[str, Any]] = []
    for line in content:
        fields = line.split()
        if not fields or not fields[0].isdigit():
            continue
        if not columns:
            columns = (
                "ATOM",
                "Z",
                "ZEFF",
                "CHARGE",
                "COVALENCY",
                "DELOCALIZATION",
                "STRAIN",
                "SIGNATURE",
            )
            positions = {name: index for index, name in enumerate(columns)}
        if len(fields) < len(columns):
            continue

        def value(name: str, default: str = "0") -> str:
            index = positions.get(name)
            return fields[index] if index is not None and index < len(fields) else default

        atoms.append(
            {
                "atom": int(value("ATOM")),
                "element": value("Z"),
                "z_eff": float(value("ZEFF")),
                "charge": float(value("CHARGE")),
                "covalency": float(value("COVALENCY")),
                "delocalization": float(value("DELOCALIZATION")),
                "radial_dispersion": float(value("RADIAL_DISPERSION")),
                "strain": float(value("STRAIN")),
                "sigma_index": float(value("SIGMA_INDEX")),
                "pi_index": float(value("PI_INDEX")),
                "pi_pi_index": float(value("PI_PI_INDEX")),
                "signature": [
                    _signature_value(item)
                    for item in value("SIGNATURE", "").split(",")
                    if item
                ],
            }
        )
    return {
        "schema": metadata.get("SCHEMA", ""),
        "charge_source": metadata.get("CHARGE_SOURCE", ""),
        "bond_order_source": metadata.get("BOND_ORDER_SOURCE", ""),
        "atoms": atoms,
    }


def _section_columns(content: list[str]) -> tuple[str, ...]:
    for line in content:
        fields = line.strip().split()
        if fields and fields[0].upper() == "COLUMNS":
            return tuple(field.upper() for field in fields[1:])
    return ()


def _optional_float(value: str | None) -> float | None:
    return float(value) if value not in {None, ""} else None


def _cartesian_geometry_payload(geometry: Any | None) -> dict[str, Any] | None:
    if geometry is None:
        return None
    return {
        "atoms": [str(atom) for atom in geometry.atoms],
        "coordinates_angstrom": [
            [float(component) for component in row]
            for row in geometry.coordinates_angstrom
        ],
        "charge": geometry.charge,
        "multiplicity": geometry.multiplicity,
    }


def _signature_value(value: str) -> int | float | bool | str:
    if value == "True":
        return True
    if value == "False":
        return False
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value
