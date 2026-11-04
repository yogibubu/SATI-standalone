"""Deterministic overlap-constrained assembly from LCB25 geometries."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from matrix_chem import (
    MolecularGeometry,
    build_topology_objects,
    complete_valence_hydrogens,
    read_xyz,
)
from matrix_chem.topology.elements import atomic_number, atomic_symbol
from matrix_switch import (
    SwitchMolecularGraph,
    find_substructure_matches,
    graph_from_topology,
    maximum_common_connected_subgraphs,
    parse_smiles,
    perceive_aromaticity,
    smiles_to_cartesian,
)

from .contracts import ORACLE_XYZ_ASSEMBLY_SCHEMA, geometric_parameter_source_binding


class OverlapAssemblyError(ValueError):
    """Raised when fragments cannot define a complete, unambiguous assembly."""


@dataclass(frozen=True)
class AssemblyStep:
    source_path: Path
    match_index: int
    target_atoms: tuple[int, ...]
    overlap_atoms: tuple[int, ...]
    new_atoms: tuple[int, ...]
    source_atom_count: int
    source_heavy_atom_count: int
    overlap_rmsd_angstrom: float
    source_to_target_atoms: tuple[tuple[int, int], ...] = ()
    extraction: str = "full_molecule"


@dataclass(frozen=True)
class OverlapAssemblyResult:
    smiles: str
    atoms: tuple[str, ...]
    coordinates_angstrom: np.ndarray
    heavy_atom_coordinates_angstrom: np.ndarray
    steps: tuple[AssemblyStep, ...]
    max_fragment_atoms: int
    min_overlap_atoms: int
    covered_heavy_atoms: int
    target_heavy_atoms: int
    uff_status: int | None
    search_method: str = "beam-search"
    beam_width: int = 64
    fragment_workers: int = 1
    pose_iterations: int = 0
    pose_converged: bool = False
    max_overlap_rmsd_angstrom: float = 0.0
    atom_provenance: tuple[tuple[str, ...], ...] = ()
    quality_warnings: tuple[str, ...] = ()
    relaxation_method: str = "NONE"
    uff_message: str = ""


@dataclass(frozen=True)
class _PlacementCandidate:
    source_path: Path
    query: SwitchMolecularGraph
    match: tuple[int, ...]
    match_index: int
    source_atom_count: int
    coordinates_angstrom: np.ndarray
    source_atom_indices: tuple[int, ...]
    extraction: str = "full_molecule"

    @property
    def key(self) -> tuple[str, str, tuple[int, ...], int, tuple[int, ...]]:
        return (
            str(self.source_path),
            self.extraction,
            self.source_atom_indices,
            self.match_index,
            self.match,
        )


@dataclass(frozen=True)
class _CoverState:
    selected: tuple[int, ...]
    covered: frozenset[int]
    total_overlap: int


def assemble_overlapping_fragments(
    smiles: str,
    fragment_paths: Iterable[Path],
    *,
    max_fragment_atoms: int = 20,
    min_overlap_atoms: int = 3,
    relax: bool = True,
    beam_width: int = 64,
    pose_max_iterations: int = 50,
    pose_tolerance_angstrom: float = 1.0e-5,
    max_overlap_rmsd_angstrom: float = 0.35,
    strict_quality: bool = False,
    workers: int | None = None,
    extract_common_subgraphs: bool = False,
    preferred_seed_paths: Iterable[Path] = (),
    strict_hydrogen_interfaces: bool = False,
    weighted_internal_closure: bool = True,
) -> OverlapAssemblyResult:
    """Assemble a target from complete fragment substructures with real overlap.

    Fragment size is counted literally, including hydrogens.  Each placement
    after the seed must add at least one target heavy atom and share at least
    ``min_overlap_atoms`` already placed heavy atoms.  Three non-collinear
    overlap atoms are required by default so that the rigid placement is not
    underdetermined.
    """
    if max_fragment_atoms < 1:
        raise OverlapAssemblyError("max_fragment_atoms must be positive")
    if min_overlap_atoms < 3:
        raise OverlapAssemblyError(
            "min_overlap_atoms must be at least 3 for an unambiguous 3D placement"
        )
    if beam_width < 1 or pose_max_iterations < 1:
        raise OverlapAssemblyError("beam_width and pose_max_iterations must be positive")
    if workers is not None and workers < 1:
        raise OverlapAssemblyError("workers must be positive")
    if pose_tolerance_angstrom <= 0.0 or max_overlap_rmsd_angstrom <= 0.0:
        raise OverlapAssemblyError("pose and overlap tolerances must be positive")
    try:
        target = _heavy_graph(perceive_aromaticity(parse_smiles(smiles)))
    except ValueError as exc:
        raise OverlapAssemblyError(f"SWITCH could not parse target SMILES: {smiles}") from exc
    target_count = len(target.atoms)
    paths = tuple(Path(path) for path in fragment_paths)
    if not paths:
        raise OverlapAssemblyError("at least one fragment geometry is required")
    worker_count = min(len(paths), workers or max(1, os.cpu_count() or 1))

    def load(path: Path) -> tuple[list[_PlacementCandidate], tuple[Path, int] | None]:
        return _load_fragment_candidates(
            path,
            target=target,
            max_fragment_atoms=max_fragment_atoms,
            extract_common_subgraphs=extract_common_subgraphs,
            # The global donor is selected before MCS.  Do not discard it by
            # imposing a larger, unrelated MCS size threshold here: the
            # placement stage already enforces a non-collinear overlap of
            # ``min_overlap_atoms`` atoms.
            minimum_common_atoms=min_overlap_atoms,
            strict_hydrogen_interfaces=strict_hydrogen_interfaces,
            use_chirality=True,
        )

    if worker_count == 1:
        loaded = [load(path) for path in paths]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            loaded = list(executor.map(load, paths))
    candidates: list[_PlacementCandidate] = []
    rejected_oversize: list[tuple[Path, int]] = []
    for fragment_candidates, rejected in loaded:
        candidates.extend(fragment_candidates)
        if rejected is not None:
            rejected_oversize.append(rejected)
    if not candidates:
        detail = ""
        if rejected_oversize:
            detail = "; all matching candidates may exceed the literal atom limit"
        raise OverlapAssemblyError(f"no usable LCB25 fragment matches the target{detail}")

    candidates.sort(key=lambda candidate: candidate.key)
    selected_indices = _beam_cover(
        candidates,
        target_count=target_count,
        min_overlap_atoms=min_overlap_atoms,
        beam_width=beam_width,
        preferred_seed_paths=tuple(str(Path(path)) for path in preferred_seed_paths),
    )
    selected_candidates = tuple(candidates[index] for index in selected_indices)
    initial_coordinates, overlap_sets, new_sets, transformed = _initial_placement(
        selected_candidates,
        target_count=target_count,
    )
    (
        coordinates,
        transformed,
        pose_iterations,
        pose_converged,
    ) = _optimize_fragment_poses(
        selected_candidates,
        transformed,
        initial_coordinates,
        max_iterations=pose_max_iterations,
        tolerance=pose_tolerance_angstrom,
    )
    if weighted_internal_closure:
        coordinates = _weighted_internal_closure(
            coordinates,
            target,
            selected_candidates,
            transformed,
        )
    selected = _assembly_steps(
        selected_candidates,
        transformed,
        coordinates,
        overlap_sets,
        new_sets,
    )
    maximum_overlap = max(
        (step.overlap_rmsd_angstrom for step in selected),
        default=0.0,
    )
    quality_warnings = []
    if maximum_overlap > max_overlap_rmsd_angstrom:
        quality_warnings.append(
            f"maximum overlap RMSD {maximum_overlap:.6f} exceeds "
            f"{max_overlap_rmsd_angstrom:.6f} angstrom"
        )
    covered = set().union(*(set(candidate.match) for candidate in selected_candidates))

    perturbed = _steric_release_seed(coordinates, target)
    heavy_geometry = MolecularGeometry(
        atoms=tuple(atom.symbol for atom in target.atoms),
        coordinates_angstrom=perturbed,
        comment=smiles,
        source_format="switch_overlap_assembly",
        charge=target.total_formal_charge,
    )
    completion = complete_valence_hydrogens(
        heavy_geometry,
        tuple(bond.key for bond in target.bonds),
        bond_orders={bond.key: bond.order for bond in target.bonds},
        requested_counts=tuple(atom.hydrogen_count for atom in target.atoms),
    )
    final_coords = np.asarray(completion.geometry.coordinates_angstrom, dtype=float)
    if not strict_hydrogen_interfaces:
        _transfer_fragment_hydrogen_positions(
            final_coords,
            completion.additions,
            selected_candidates,
            coordinates,
        )
    atom_symbols = completion.geometry.atoms
    assembled_bonds = completion.bonds
    if strict_hydrogen_interfaces:
        _validate_target_topology_and_hydrogens(
            atom_symbols,
            assembled_bonds,
            target,
            target_count=target_count,
        )
    uff_status: int | None = None
    relaxation_method = "NONE"
    uff_message = "not requested"
    if relax:
        final_coords, converged = _steric_relaxation(
            atom_symbols,
            final_coords,
            assembled_bonds,
        )
        relaxation_method = "SWITCH_ANALYTIC_STERIC"
        uff_status = 0 if converged else 1
        uff_message = "converged" if converged else "iteration limit reached"
    final_heavy_coordinates = final_coords[:target_count].copy()
    chemical_warnings = _chemical_quality_warnings(
        atom_symbols,
        assembled_bonds,
        target,
        final_coords,
        target_count=target_count,
    )
    quality_warnings.extend(chemical_warnings)
    if strict_quality and quality_warnings:
        raise OverlapAssemblyError("assembly quality gate failed: " + "; ".join(quality_warnings))
    provenance = _atom_provenance(selected_candidates, target_count=target_count)
    return OverlapAssemblyResult(
        smiles=smiles,
        atoms=atom_symbols,
        coordinates_angstrom=final_coords,
        heavy_atom_coordinates_angstrom=final_heavy_coordinates,
        steps=tuple(selected),
        max_fragment_atoms=max_fragment_atoms,
        min_overlap_atoms=min_overlap_atoms,
        covered_heavy_atoms=len(covered),
        target_heavy_atoms=target_count,
        uff_status=uff_status,
        beam_width=beam_width,
        fragment_workers=worker_count,
        pose_iterations=pose_iterations,
        pose_converged=pose_converged,
        max_overlap_rmsd_angstrom=maximum_overlap,
        atom_provenance=provenance,
        quality_warnings=tuple(quality_warnings),
        relaxation_method=relaxation_method,
        uff_message=uff_message,
    )


def assemble_l1_parameter_geometry(
    smiles: str,
    donor_paths: Iterable[Path],
    *,
    donor_weights: Mapping[Path, float] | None = None,
    donor_labels: Mapping[Path, str] | None = None,
    electronic_records: Mapping[Path, Mapping[str, object]] | None = None,
    parameter_source: str = "L1_geometry_only",
    selection_priority: str = "composition",
    max_local_signature_distance: float = 2.0,
    max_fragment_atoms: int = 10_000,
    min_overlap_atoms: int = 3,
    strict_parameters: bool = True,
    strict_hydrogen_interfaces: bool = True,
    allow_relaxed_hydrogen_fallback: bool = False,
    max_donors: int | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Build a heavy-atom seed from L1 bond/valence-angle observations.

    This is deliberately different from rigid overlap assembly.  Each donor
    contributes only the target primitives that it maps exactly: bond lengths
    and valence angles.  Exocyclic dihedrals and Cartesian donor poses are not
    transferred.  The observations are combined with the existing weighted
    Wilson-B closure routine.
    """
    if min_overlap_atoms < 3:
        raise OverlapAssemblyError("min_overlap_atoms must be at least 3")
    if selection_priority not in {"composition", "similarity"}:
        raise OverlapAssemblyError("selection_priority must be composition or similarity")
    if max_local_signature_distance < 0.0:
        raise OverlapAssemblyError("max_local_signature_distance must be non-negative")
    parameter_binding = geometric_parameter_source_binding(parameter_source)
    if parameter_binding["cm5_mayer_allowed"] and not electronic_records:
        raise OverlapAssemblyError("L2/PL2 parameter assembly requires enriched CM5/Mayer records")
    if not parameter_binding["cm5_mayer_allowed"] and electronic_records:
        raise OverlapAssemblyError("L1 parameter assembly cannot consume CM5/Mayer records")
    target = _heavy_graph(perceive_aromaticity(parse_smiles(smiles)))
    target_element_counts: dict[str, int] = {}
    for atom in target.atoms:
        target_element_counts[atom.symbol] = target_element_counts.get(atom.symbol, 0) + 1
    paths = tuple(Path(path).resolve() for path in donor_paths)
    if max_donors is not None:
        if max_donors < 1:
            raise OverlapAssemblyError("max_donors must be positive")
        paths = paths[:max_donors]
    if not paths:
        raise OverlapAssemblyError("at least one L1 donor geometry is required")

    target_geometry = smiles_to_cartesian(smiles, title="MATRIX L1 parameter seed")
    coordinates = np.asarray(target_geometry.coordinates_angstrom[: len(target.atoms)], dtype=float)
    target_numbers = tuple(int(atomic_number(atom)) for atom in target_geometry.atoms)
    _target_continuous, _target_discrete, _target_rings, target_synthons, _target_aromaticity = (
        build_topology_objects(target_geometry.coordinates_angstrom, target_numbers)
    )
    bonds = [tuple(bond.key) for bond in target.bonds]
    adjacency = [set() for _ in target.atoms]
    for left, right in bonds:
        adjacency[left].add(right)
        adjacency[right].add(left)
    angles = [
        (left, center, right)
        for center, neighbors in enumerate(adjacency)
        for left in neighbors
        for right in neighbors
        if left < right
    ]
    ring_bond_indices = {
        index for index, (left, right) in enumerate(bonds)
        if _is_ring_edge(left, right, adjacency)
    }
    ring_angle_indices = {
        len(bonds) + index
        for index, (left, center, right) in enumerate(angles)
        if _is_ring_edge(left, center, adjacency)
        and _is_ring_edge(center, right, adjacency)
    }
    protected_ring_indices = tuple(sorted(ring_bond_indices | ring_angle_indices))

    bond_observations: list[list[tuple[float, float]]] = [[] for _ in bonds]
    angle_observations: list[list[tuple[float, float]]] = [[] for _ in angles]
    candidate_records: list[dict[str, object]] = []
    missing_by_donor: dict[str, list[str]] = {}

    def primitive_value(fragment: np.ndarray, primitive: tuple[int, ...]) -> float:
        if len(primitive) == 2:
            left, right = primitive
            return float(np.linalg.norm(fragment[left] - fragment[right]))
        left, center, right = primitive
        u = fragment[left] - fragment[center]
        v = fragment[right] - fragment[center]
        denominator = max(np.linalg.norm(u) * np.linalg.norm(v), 1.0e-15)
        # The Wilson-B realization contract uses radians for angles.  Reports
        # and validation convert the resulting geometry to degrees.
        return float(np.arccos(np.clip(np.dot(u, v) / denominator, -1.0, 1.0)))

    for path in paths:
        electronic_record = None if electronic_records is None else electronic_records.get(path)
        if parameter_binding["cm5_mayer_allowed"]:
            if not isinstance(electronic_record, Mapping):
                missing_by_donor[str(path)] = ["missing_enriched_electronic_record"]
                continue
            if not electronic_record.get("cm5_charges_e") or not (
                electronic_record.get("mayer_bond_orders")
                or electronic_record.get("mayer_bond_components")
            ):
                missing_by_donor[str(path)] = ["missing_cm5_or_mayer_observables"]
                continue
        candidates, _rejected = _load_fragment_candidates(
            path,
            target=target,
            max_fragment_atoms=max_fragment_atoms,
            extract_common_subgraphs=True,
            minimum_common_atoms=min_overlap_atoms,
            strict_hydrogen_interfaces=strict_hydrogen_interfaces,
            use_chirality=False,
        )
        hydrogen_interface_relaxed = False
        if not candidates and allow_relaxed_hydrogen_fallback and strict_hydrogen_interfaces:
            candidates, _rejected = _load_fragment_candidates(
                path,
                target=target,
                max_fragment_atoms=max_fragment_atoms,
                extract_common_subgraphs=True,
                minimum_common_atoms=min_overlap_atoms,
                strict_hydrogen_interfaces=False,
                use_chirality=False,
            )
            hydrogen_interface_relaxed = bool(candidates)
        if not candidates:
            missing_by_donor[str(path)] = ["no_connected_target_match"]
            continue
        source_graph, _source_coordinates, _source_heavy, _source_count = _read_xyz_molecule(
            path,
            include_hydrogen_counts=True,
        )
        source_geometry = read_xyz(path)
        source_numbers = tuple(int(atomic_number(atom)) for atom in source_geometry.atoms)
        _source_continuous, _source_discrete, _source_rings, source_synthons, _source_aromaticity = (
            build_topology_objects(source_geometry.coordinates_angstrom, source_numbers)
        )
        source_adjacency = [set() for _ in source_graph.atoms]
        for bond in source_graph.bonds:
            left, right = bond.key
            source_adjacency[left].add(right)
            source_adjacency[right].add(left)
        components = 0
        unseen = set(range(len(source_graph.atoms)))
        while unseen:
            components += 1
            stack = [unseen.pop()]
            while stack:
                atom = stack.pop()
                for neighbor in source_adjacency[atom] & unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        if components != 1 and len(target.atoms) > 1:
            missing_by_donor[str(path)] = ["disconnected_source_molecule"]
            continue
        candidate = max(
            candidates,
            key=lambda item: (
                len(item.match),
                item.extraction == "full_molecule",
                tuple(-value for value in item.source_atom_indices),
                item.key,
            ),
        )
        target_to_source = {
            target_index: source_index
            for source_index, target_index in enumerate(candidate.match)
        }
        source_counts: dict[str, int] = {}
        for atom in source_graph.atoms:
            source_counts[atom.symbol] = source_counts.get(atom.symbol, 0) + 1
        composition_distance = sum(
            abs(source_counts.get(symbol, 0) - target_element_counts.get(symbol, 0))
            for symbol in set(source_counts) | set(target_element_counts)
        )
        source_heavy_indices = tuple(index for index, atom in enumerate(source_geometry.atoms) if atom != "H")
        target_heavy_indices = tuple(index for index, atom in enumerate(target_geometry.atoms) if atom != "H")
        target_full_by_heavy = {heavy: full for heavy, full in enumerate(target_heavy_indices)}
        source_full_by_heavy = {heavy: full for heavy, full in enumerate(source_heavy_indices)}
        synthon_distance = 0.0
        primitive_synthon_distances: dict[str, float] = {}
        primitive_local_signature_distances: dict[str, float] = {}
        def local_signature_distance(target_index: int, source_index: int) -> float:
            target_atom = target.atoms[target_index]
            source_atom = source_graph.atoms[source_index]
            distance = 0.0 if target_atom.symbol == source_atom.symbol else 100.0
            distance += 2.0 * abs(
                int(target_atom.hydrogen_count or 0) - int(source_atom.hydrogen_count or 0)
            )
            target_neighbors = sorted(
                (target.atoms[neighbor].symbol, round(float(bond.order), 1))
                for bond in target.bonds
                for neighbor in ([bond.key[1]] if bond.key[0] == target_index else [bond.key[0]] if bond.key[1] == target_index else [])
            )
            source_neighbors = sorted(
                (source_graph.atoms[neighbor].symbol, round(float(bond.order), 1))
                for bond in source_graph.bonds
                for neighbor in ([bond.key[1]] if bond.key[0] == source_index else [bond.key[0]] if bond.key[1] == source_index else [])
            )
            distance += sum(
                1.0 for left, right in zip(target_neighbors, source_neighbors)
                if left != right
            ) + abs(len(target_neighbors) - len(source_neighbors))
            return distance
        def atom_synthon_distance(target_index: int, source_index: int) -> float:
            target_full = target_full_by_heavy[target_index]
            source_full = source_full_by_heavy[source_index]
            return (
                abs(float(target_synthons.cna(target_full)) - float(source_synthons.cna(source_full)))
                + abs(float(target_synthons.Zeff(target_full)) - float(source_synthons.Zeff(source_full))) / 2.0
                + abs(float(target_synthons.charge(target_full)) - float(source_synthons.charge(source_full)))
                + abs(float(target_synthons.electron_domains(target_full)) - float(source_synthons.electron_domains(source_full)))
                + abs(float(target_synthons.pi_index(target_full)) - float(source_synthons.pi_index(source_full)))
                + abs(float(target_synthons.pi_pi_index(target_full)) - float(source_synthons.pi_pi_index(source_full)))
            )
        for source_index, target_index in enumerate(candidate.match):
            synthon_distance += atom_synthon_distance(target_index, source_index)
        synthon_distance /= max(len(candidate.match), 1)
        for left, right in bonds:
            if left in target_to_source and right in target_to_source:
                primitive_synthon_distances[f"bond:{left + 1}-{right + 1}"] = float(max([
                    atom_synthon_distance(left, target_to_source[left]),
                    atom_synthon_distance(right, target_to_source[right]),
                ]))
                primitive_local_signature_distances[f"bond:{left + 1}-{right + 1}"] = float(max([
                    local_signature_distance(left, target_to_source[left]),
                    local_signature_distance(right, target_to_source[right]),
                ]))
        for left, center, right in angles:
            if left in target_to_source and center in target_to_source and right in target_to_source:
                primitive_synthon_distances[f"angle:{left + 1}-{center + 1}-{right + 1}"] = float(max([
                    atom_synthon_distance(left, target_to_source[left]),
                    atom_synthon_distance(center, target_to_source[center]),
                    atom_synthon_distance(right, target_to_source[right]),
                ]))
                primitive_local_signature_distances[f"angle:{left + 1}-{center + 1}-{right + 1}"] = float(max([
                    local_signature_distance(left, target_to_source[left]),
                    local_signature_distance(center, target_to_source[center]),
                    local_signature_distance(right, target_to_source[right]),
                ]))
        source_edges = {tuple(sorted(bond.key)) for bond in candidate.query.bonds}
        weight = 1.0 if donor_weights is None else float(donor_weights.get(path, 0.0))
        label = str(path) if donor_labels is None else str(donor_labels.get(path, path.stem))
        if not np.isfinite(weight) or weight <= 0.0:
            missing_by_donor[str(path)] = ["nonpositive_donor_weight"]
            continue
        missing: list[str] = []
        mapped_bond_values: dict[int, float] = {}
        mapped_angle_values: dict[int, float] = {}
        mapped_bonds = 0
        mapped_angles = 0
        for index, bond in enumerate(bonds):
            if all(atom in target_to_source for atom in bond):
                source_bond = tuple(sorted(target_to_source[atom] for atom in bond))
                if source_bond in source_edges:
                    mapped_bond_values[index] = primitive_value(candidate.coordinates_angstrom, source_bond)
                    mapped_bonds += 1
                    continue
            missing.append(f"bond:{bond[0] + 1}-{bond[1] + 1}")
        for index, angle in enumerate(angles):
            if all(atom in target_to_source for atom in angle):
                source_angle = tuple(target_to_source[atom] for atom in angle)
                if (
                    tuple(sorted((source_angle[0], source_angle[1]))) in source_edges
                    and tuple(sorted((source_angle[1], source_angle[2]))) in source_edges
                ):
                    mapped_angle_values[index] = primitive_value(candidate.coordinates_angstrom, source_angle)
                    mapped_angles += 1
                    continue
            missing.append(f"angle:{angle[0] + 1}-{angle[1] + 1}-{angle[2] + 1}")
        candidate_records.append({
            "source": str(path),
            "identifier": label,
            "match_atoms": len(candidate.match),
            "composition_distance": composition_distance,
            "synthon_distance": synthon_distance,
            "primitive_synthon_distances": primitive_synthon_distances,
            "primitive_local_signature_distances": primitive_local_signature_distances,
            "hydrogen_interface_relaxed": hydrogen_interface_relaxed,
            "mapped_bonds": mapped_bonds,
            "mapped_angles": mapped_angles,
            "weight": weight,
            "bond_values": mapped_bond_values,
            "angle_values": mapped_angle_values,
        })
        if missing:
            missing_by_donor[str(path)] = missing

    # Select a small, auditable set of fragments.  First take the best global
    # donor that covers each primitive, then add the best additional donor for
    # each primitive.  This is a weighted primitive-cover problem: unrelated
    # records that happen to contain a three-atom match cannot dilute all
    # parameters, while a lower-ranked donor is retained when it is the only
    # admissible source for a missing bond or angle.
    def record_key(record: dict[str, object], primitive_coverage: int) -> tuple[float, int, int, str]:
        return (
            float(record["weight"]),
            primitive_coverage,
            int(record["match_atoms"]),
            str(record["source"]),
        )

    selected_records: list[dict[str, object]] = []
    selected_sources: set[str] = set()
    selected_by_primitive: dict[str, dict[str, object]] = {}
    primitive_count = len(bonds) + len(angles)
    for primitive in range(primitive_count):
        is_bond = primitive < len(bonds)
        values_key = "bond_values" if is_bond else "angle_values"
        local_index = primitive if is_bond else primitive - len(bonds)
        primitive_label = (
            f"bond:{bonds[local_index][0] + 1}-{bonds[local_index][1] + 1}"
            if is_bond
            else f"angle:{angles[local_index][0] + 1}-{angles[local_index][1] + 1}-{angles[local_index][2] + 1}"
        )
        eligible = [
            record for record in candidate_records
            if local_index in record[values_key]
            and float(record["primitive_synthon_distances"].get(primitive_label, float("inf")))
            <= max_local_signature_distance
        ]
        eligible.sort(
            key=lambda record: record_key(record, len(record["bond_values"]) + len(record["angle_values"])),
            reverse=True,
        )
        if not eligible:
            continue
        # Exactly one donor is selected for this primitive.  Chemical local
        # admissibility is established by SWITCH/MCS first; among admissible
        # candidates prefer the largest local environment, then similarity.
        if selection_priority == "similarity":
            eligible.sort(
                key=lambda record: (
                    float(record["weight"]),
                    -float(record["primitive_local_signature_distances"].get(primitive_label, float("inf"))),
                    -int(record["composition_distance"]),
                    -float(record["primitive_synthon_distances"].get(primitive_label, float("inf"))),
                    int(record["match_atoms"]),
                    str(record["identifier"]),
                ),
                reverse=True,
            )
        else:
            eligible.sort(
                key=lambda record: (
                    -int(record["composition_distance"]),
                    -float(record["primitive_local_signature_distances"].get(primitive_label, float("inf"))),
                    -float(record["primitive_synthon_distances"].get(primitive_label, float("inf"))),
                    int(record["match_atoms"]),
                    float(record["weight"]),
                    str(record["identifier"]),
                ),
                reverse=True,
            )
        record = eligible[0]
        selected_by_primitive[primitive_label] = record
        source = str(record["source"])
        if source not in selected_sources:
            selected_records.append(record)
            selected_sources.add(source)

    for primitive_index, record in selected_by_primitive.items():
        weight = float(record["weight"])
        if primitive_index.startswith("bond:"):
            index = next(
                index for index, (left, right) in enumerate(bonds)
                if primitive_index == f"bond:{left + 1}-{right + 1}"
            )
            bond_observations[index].append((float(record["bond_values"][index]), weight))
        else:
            index = next(
                index for index, (left, center, right) in enumerate(angles)
                if primitive_index == f"angle:{left + 1}-{center + 1}-{right + 1}"
            )
            angle_observations[index].append((float(record["angle_values"][index]), weight))

    donors_by_primitive = {
        **{
            f"bond:{left + 1}-{right + 1}": [str(selected_by_primitive[f"bond:{left + 1}-{right + 1}"]["identifier"])]
            for index, (left, right) in enumerate(bonds)
            if f"bond:{left + 1}-{right + 1}" in selected_by_primitive
        },
        **{
            f"angle:{left + 1}-{center + 1}-{right + 1}": [str(selected_by_primitive[f"angle:{left + 1}-{center + 1}-{right + 1}"]["identifier"])]
            for index, (left, center, right) in enumerate(angles)
            if f"angle:{left + 1}-{center + 1}-{right + 1}" in selected_by_primitive
        },
    }
    synthon_distance_by_primitive = {
        primitive: float(record["primitive_synthon_distances"].get(primitive, float("inf")))
        for primitive, record in selected_by_primitive.items()
    }

    missing_bonds = [
        f"bond:{left + 1}-{right + 1}"
        for (left, right), observations in zip(bonds, bond_observations, strict=True)
        if not observations
    ]
    missing_angles = [
        f"angle:{left + 1}-{center + 1}-{right + 1}"
        for (left, center, right), observations in zip(angles, angle_observations, strict=True)
        if not observations
    ]
    if strict_parameters and (missing_bonds or missing_angles):
        raise OverlapAssemblyError(
            "L1 parameter coverage is incomplete: "
            + ", ".join(missing_bonds + missing_angles)
        )
    from matrix_oracle import weighted_l1_internal_closure

    # Redundant internal coordinates can be mutually incompatible.  Keep the
    # least-squares projection instead of rejecting the molecule, and refine
    # its Cartesian seed iteratively until the residual stops improving.
    closure = None
    refinement_history: list[dict[str, object]] = []
    for iteration in range(8):
        current = weighted_l1_internal_closure(
            coordinates,
            bonds,
            angles,
            bond_observations,
            angle_observations,
            protected_ring_indices=protected_ring_indices,
            max_iterations=500,
            tolerance=1.0e-8,
        )
        refinement_history.append({
            "iteration": iteration + 1,
            "converged": bool(current.converged),
            "maximum_residual": float(current.maximum_residual),
            "backtransform_iterations": int(current.iterations),
        })
        if closure is None or current.maximum_residual < closure.maximum_residual:
            closure = current
        updated = np.asarray(current.coordinates_angstrom, dtype=float)
        if np.allclose(updated, coordinates, rtol=0.0, atol=1.0e-9):
            break
        coordinates = updated
    assert closure is not None
    audit = {
        "parameter_source": parameter_source,
        "donors_considered": len(paths),
        "donors_used": [
            {key: value for key, value in record.items() if key not in {"bond_values", "angle_values"}}
            for record in selected_records
        ],
        "selection": {
            "method": "one chemically admissible donor per primitive",
            "priority": selection_priority,
            "max_local_signature_distance": max_local_signature_distance,
            "candidate_count": len(candidate_records),
            "selected_count": len(selected_records),
            "donors_by_primitive": donors_by_primitive,
            "synthon_distance_by_primitive": synthon_distance_by_primitive,
        },
        "parameter_source_binding": parameter_binding,
        "electronic_records_validated": len(electronic_records or {}),
        "synthon_provenance": {
            "engine": "ORACLE build_topology_objects",
            "charges": "electronegativity_estimated_from_geometry",
            "bond_orders": "Pauling_orders_estimated_from_geometry",
            "cm5_mayer_used": False,
        },
        "bond_count": len(bonds),
        "angle_count": len(angles),
        "ring_bond_indices": sorted(ring_bond_indices),
        "ring_angle_indices": sorted(index - len(bonds) for index in ring_angle_indices),
        "missing_bonds": missing_bonds,
        "missing_angles": missing_angles,
        "donor_mapping_gaps": missing_by_donor,
        "closure_converged": bool(closure.converged),
        "closure_max_residual": float(closure.maximum_residual),
        "closure_refinement": refinement_history,
        "exocyclic_dihedrals_used": False,
    }
    return np.asarray(closure.coordinates_angstrom, dtype=float), audit


def _beam_cover(
    candidates: list[_PlacementCandidate],
    *,
    target_count: int,
    min_overlap_atoms: int,
    beam_width: int,
    preferred_seed_paths: tuple[str, ...] = (),
) -> tuple[int, ...]:
    target_atoms = frozenset(range(target_count))
    preferred = [
        (index, candidate)
        for index, candidate in enumerate(candidates)
        if preferred_seed_paths
        and str(candidate.source_path) == preferred_seed_paths[0]
    ]
    seed_candidates = preferred or list(enumerate(candidates))
    beam = [
        _CoverState((index,), frozenset(candidate.match), 0)
        for index, candidate in seed_candidates
    ]
    beam = _rank_cover_states(beam, candidates)[:beam_width]
    while beam:
        completed = [state for state in beam if state.covered == target_atoms]
        if completed:
            return min(
                completed,
                key=lambda state: (
                    len(state.selected),
                    -state.total_overlap,
                    tuple(candidates[index].key for index in state.selected),
                ),
            ).selected
        expanded = []
        seen = set()
        for state in beam:
            for index, candidate in enumerate(candidates):
                if index in state.selected:
                    continue
                match = frozenset(candidate.match)
                overlap = state.covered.intersection(match)
                new = match.difference(state.covered)
                if (
                    not new
                    or len(overlap) < min_overlap_atoms
                    or not _candidate_overlap_is_noncollinear(candidate, overlap)
                ):
                    continue
                next_state = _CoverState(
                    state.selected + (index,),
                    state.covered.union(match),
                    state.total_overlap + len(overlap),
                )
                identity = (next_state.selected, next_state.covered)
                if identity not in seen:
                    expanded.append(next_state)
                    seen.add(identity)
        if not expanded:
            best = max(beam, key=lambda state: len(state.covered))
            missing = ",".join(
                str(index + 1) for index in sorted(target_atoms.difference(best.covered))
            )
            raise OverlapAssemblyError(
                "overlap-constrained beam search stalled; uncovered target heavy atoms: " + missing
            )
        beam = _rank_cover_states(expanded, candidates)[:beam_width]
    raise OverlapAssemblyError("overlap-constrained beam search found no assembly")


def _rank_cover_states(
    states: list[_CoverState],
    candidates: list[_PlacementCandidate],
) -> list[_CoverState]:
    return sorted(
        states,
        key=lambda state: (
            -len(state.covered),
            len(state.selected),
            -state.total_overlap,
            -sum(len(candidates[index].match) for index in state.selected),
            tuple(candidates[index].key for index in state.selected),
        ),
    )
def _candidate_overlap_is_noncollinear(
    candidate: _PlacementCandidate,
    overlap: frozenset[int],
) -> bool:
    target_to_source = {
        target_index: source_index for source_index, target_index in enumerate(candidate.match)
    }
    coordinates = np.asarray(
        [
            candidate.coordinates_angstrom[target_to_source[target_index]]
            for target_index in sorted(overlap)
        ],
        dtype=float,
    )
    return bool(
        len(coordinates) >= 3
        and np.linalg.matrix_rank(coordinates - coordinates.mean(axis=0), tol=1.0e-8) >= 2
    )


def _initial_placement(
    candidates: tuple[_PlacementCandidate, ...],
    *,
    target_count: int,
) -> tuple[
    np.ndarray,
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
    tuple[np.ndarray, ...],
]:
    coordinates = np.full((target_count, 3), np.nan, dtype=float)
    covered: set[int] = set()
    overlaps = []
    new_atoms = []
    transformed_fragments = []
    for candidate in candidates:
        match_set = set(candidate.match)
        overlap = tuple(sorted(match_set.intersection(covered)))
        new = tuple(sorted(match_set.difference(covered)))
        target_to_source = {
            target_index: source_index for source_index, target_index in enumerate(candidate.match)
        }
        if transformed_fragments:
            moving = candidate.coordinates_angstrom[[target_to_source[index] for index in overlap]]
            fixed = coordinates[list(overlap)]
            rotation, translation = _rigid_alignment(moving, fixed)
            transformed = candidate.coordinates_angstrom @ rotation + translation
        else:
            transformed = candidate.coordinates_angstrom.copy()
        for target_index in overlap:
            source_index = target_to_source[target_index]
            coordinates[target_index] = 0.5 * (
                coordinates[target_index] + transformed[source_index]
            )
        for target_index in new:
            coordinates[target_index] = transformed[target_to_source[target_index]]
        overlaps.append(overlap)
        new_atoms.append(new)
        transformed_fragments.append(transformed)
        covered.update(candidate.match)
    if np.any(~np.isfinite(coordinates)):
        raise OverlapAssemblyError("selected fragment cover left undefined coordinates")
    return (
        coordinates,
        tuple(overlaps),
        tuple(new_atoms),
        tuple(transformed_fragments),
    )


def _optimize_fragment_poses(
    candidates: tuple[_PlacementCandidate, ...],
    transformed: tuple[np.ndarray, ...],
    initial_coordinates: np.ndarray,
    *,
    max_iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, tuple[np.ndarray, ...], int, bool]:
    consensus = np.asarray(initial_coordinates, dtype=float).copy()
    current = tuple(np.asarray(fragment, dtype=float).copy() for fragment in transformed)
    for iteration in range(1, max_iterations + 1):
        updated = []
        for candidate in candidates:
            fixed = consensus[list(candidate.match)]
            rotation, translation = _rigid_alignment(
                candidate.coordinates_angstrom,
                fixed,
            )
            updated.append(candidate.coordinates_angstrom @ rotation + translation)
        next_consensus = np.zeros_like(consensus)
        counts = np.zeros(len(consensus), dtype=int)
        for candidate, fragment in zip(candidates, updated, strict=True):
            for source_index, target_index in enumerate(candidate.match):
                next_consensus[target_index] += fragment[source_index]
                counts[target_index] += 1
        if np.any(counts == 0):
            raise OverlapAssemblyError("global pose optimization lost target coverage")
        next_consensus /= counts[:, None]
        frame_rotation, frame_translation = _rigid_alignment(next_consensus, initial_coordinates)
        next_consensus = next_consensus @ frame_rotation + frame_translation
        updated = [fragment @ frame_rotation + frame_translation for fragment in updated]
        change = float(np.sqrt(np.mean(np.sum((next_consensus - consensus) ** 2, axis=1))))
        consensus = next_consensus
        current = tuple(updated)
        if change <= tolerance:
            return consensus, current, iteration, True
    return consensus, current, max_iterations, False


def _weighted_internal_closure(
    coordinates: np.ndarray,
    target: SwitchMolecularGraph,
    candidates: tuple[_PlacementCandidate, ...],
    transformed: tuple[np.ndarray, ...],
    *,
    max_iterations: int = 30,
    tolerance: float = 1.0e-7,
) -> np.ndarray:
    """Close transferred bonds/angles with a weighted Wilson-B pseudoinverse.

    Observations are taken only from the selected fragment geometries.  No
    withheld target coordinates enter this projection.  Non-bridge edges and
    their angles are protected so ring geometry remains fragment-owned.
    """
    bonds = [bond.key for bond in target.bonds]
    adjacency = [set() for _ in target.atoms]
    for left, right in bonds:
        adjacency[left].add(right)
        adjacency[right].add(left)
    angles = [
        (left, center, right)
        for center, neighbors in enumerate(adjacency)
        for left in neighbors
        for right in neighbors
        if left < right
    ]

    def value(coords, primitive):
        if len(primitive) == 2:
            left, right = primitive
            return float(np.linalg.norm(coords[left] - coords[right]))
        left, center, right = primitive
        u, v = coords[left] - coords[center], coords[right] - coords[center]
        denominator = max(np.linalg.norm(u) * np.linalg.norm(v), 1.0e-15)
        return float(np.arccos(np.clip(np.dot(u, v) / denominator, -1.0, 1.0)))

    observations: dict[tuple[int, ...], list[float]] = {item: [] for item in bonds + angles}
    for candidate, fragment in zip(candidates, transformed, strict=True):
        mapped = {source: target_index for source, target_index in enumerate(candidate.match)}
        for bond in candidate.query.bonds:
            if bond.left in mapped and bond.right in mapped:
                key = tuple(sorted((mapped[bond.left], mapped[bond.right])))
                if key in observations:
                    observations[key].append(value(fragment, (bond.left, bond.right)))
        for center in range(len(candidate.query.atoms)):
            neighbors = [n for n in range(len(candidate.query.atoms)) if any(
                bond.key == tuple(sorted((center, n))) for bond in candidate.query.bonds
            )]
            for left_index, left in enumerate(neighbors):
                for right in neighbors[left_index + 1:]:
                    if left not in mapped or center not in mapped or right not in mapped:
                        continue
                    key = (
                        min(mapped[left], mapped[right]),
                        mapped[center],
                        max(mapped[left], mapped[right]),
                    )
                    if key in observations:
                        observations[key].append(value(fragment, (left, center, right)))
    primitives = bonds + angles
    protected = {index for index, (left, right) in enumerate(bonds) if _is_ring_edge(left, right, adjacency)}
    protected.update(len(bonds) + index for index, angle in enumerate(angles)
                     if _is_ring_edge(angle[0], angle[1], adjacency)
                     and _is_ring_edge(angle[1], angle[2], adjacency))
    movable = [index for index in range(len(primitives)) if index not in protected and observations[primitives[index]]]
    if not movable:
        return coordinates
    from matrix_oracle import weighted_l1_internal_closure

    bond_observations = [[(float(item), 1.0) for item in observations[bond]] for bond in bonds]
    angle_observations = [[(float(item), 1.0) for item in observations[angle]] for angle in angles]
    result = weighted_l1_internal_closure(
        coordinates,
        bonds,
        angles,
        bond_observations,
        angle_observations,
        protected_ring_indices=tuple(protected),
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    if not result.converged:
        raise OverlapAssemblyError(
            "L1 Wilson-B closure did not converge: "
            f"residual={result.maximum_residual:.8g}"
        )
    return np.asarray(result.coordinates_angstrom, dtype=float)


def _is_ring_edge(left: int, right: int, adjacency: list[set[int]]) -> bool:
    """Return whether an edge remains connected after its removal."""
    seen = {left}
    stack = [left]
    while stack:
        atom = stack.pop()
        for neighbor in adjacency[atom]:
            if (atom == left and neighbor == right) or (atom == right and neighbor == left):
                continue
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return right in seen


def _assembly_steps(
    candidates: tuple[_PlacementCandidate, ...],
    transformed: tuple[np.ndarray, ...],
    consensus: np.ndarray,
    overlaps: tuple[tuple[int, ...], ...],
    new_atoms: tuple[tuple[int, ...], ...],
) -> tuple[AssemblyStep, ...]:
    steps = []
    for candidate, fragment, overlap, new in zip(
        candidates, transformed, overlaps, new_atoms, strict=True
    ):
        target_to_source = {
            target_index: source_index for source_index, target_index in enumerate(candidate.match)
        }
        if overlap:
            fitted = fragment[[target_to_source[index] for index in overlap]]
            fixed = consensus[list(overlap)]
            rmsd = float(np.sqrt(np.mean(np.sum((fitted - fixed) ** 2, axis=1))))
        else:
            rmsd = 0.0
        steps.append(
            AssemblyStep(
                source_path=candidate.source_path,
                match_index=candidate.match_index,
                target_atoms=tuple(index + 1 for index in candidate.match),
                overlap_atoms=tuple(index + 1 for index in overlap),
                new_atoms=tuple(index + 1 for index in new),
                source_atom_count=candidate.source_atom_count,
                source_heavy_atom_count=len(candidate.query.atoms),
                overlap_rmsd_angstrom=rmsd,
                source_to_target_atoms=tuple(
                    (source_index + 1, target_index + 1)
                    for source_index, target_index in zip(
                        candidate.source_atom_indices,
                        candidate.match,
                        strict=True,
                    )
                ),
                extraction=candidate.extraction,
            )
        )
    return tuple(steps)


def _atom_provenance(
    candidates: tuple[_PlacementCandidate, ...],
    *,
    target_count: int,
) -> tuple[tuple[str, ...], ...]:
    provenance: list[list[str]] = [[] for _ in range(target_count)]
    for candidate in candidates:
        for source_index, target_index in enumerate(candidate.match):
            provenance[target_index].append(
                f"{candidate.source_path}#match={candidate.match_index}"
                f":source_atom={source_index + 1}"
            )
    return tuple(tuple(items) for items in provenance)


def _chemical_quality_warnings(
    atoms: tuple[str, ...],
    bonds: tuple[tuple[int, int], ...],
    target: SwitchMolecularGraph,
    coordinates: np.ndarray,
    *,
    target_count: int,
) -> list[str]:
    from matrix_chem.topology.covalent_radii import covalent_radius
    from matrix_chem.topology.elements import atomic_number

    warnings = []
    numbers = tuple(int(atomic_number(symbol) or 0) for symbol in atoms)
    adjacency = [set() for _ in atoms]
    for left, right in bonds:
        adjacency[left].add(right)
        adjacency[right].add(left)
        radius_sum = float(covalent_radius(numbers[left]) or 0.75) + float(
            covalent_radius(numbers[right]) or 0.75
        )
        distance = float(np.linalg.norm(coordinates[left] - coordinates[right]))
        ratio = distance / max(radius_sum, 1.0e-12)
        if not 0.55 <= ratio <= 1.65:
            warnings.append(f"bond {left + 1}-{right + 1} has covalent-radius ratio {ratio:.3f}")
    excluded = {
        tuple(sorted((left, right)))
        for left, neighbors in enumerate(adjacency)
        for right in neighbors
    }
    for neighbors in adjacency:
        for left in neighbors:
            for right in neighbors:
                if left < right:
                    excluded.add((left, right))
    minimum_nonbonded_scale = 0.60
    for left in range(len(atoms)):
        for right in range(left + 1, len(atoms)):
            if (left, right) in excluded:
                continue
            radius_sum = float(covalent_radius(numbers[left]) or 0.75) + float(
                covalent_radius(numbers[right]) or 0.75
            )
            distance = float(np.linalg.norm(coordinates[left] - coordinates[right]))
            if distance < minimum_nonbonded_scale * radius_sum:
                warnings.append(
                    f"nonbonded collision {left + 1}-{right + 1} at {distance:.3f} angstrom"
                )
    for atom in target.atoms:
        if atom.chirality and atom.index >= target_count:
            warnings.append(
                f"declared stereocenter {atom.index + 1} is outside the heavy-atom target"
            )
    return warnings


def _validate_target_topology_and_hydrogens(
    atoms: tuple[str, ...],
    bonds: tuple[tuple[int, int], ...],
    target: SwitchMolecularGraph,
    *,
    target_count: int,
) -> None:
    """Validate the immutable target graph without distance-based perception."""
    heavy_bonds = {
        tuple(sorted((left, right)))
        for left, right in bonds
        if left < target_count and right < target_count
    }
    target_bonds = {tuple(sorted(bond.key)) for bond in target.bonds}
    if heavy_bonds != target_bonds:
        raise OverlapAssemblyError(
            "assembled heavy topology differs from the immutable SWITCH target graph"
        )
    hydrogen_bonds = {}
    for left, right in bonds:
        left_h = left >= target_count
        right_h = right >= target_count
        if left_h == right_h:
            continue
        hydrogen = left if left_h else right
        heavy = right if left_h else left
        hydrogen_bonds[hydrogen] = hydrogen_bonds.get(hydrogen, 0) + 1
        if atoms[heavy] == "H" or atoms[hydrogen] != "H":
            raise OverlapAssemblyError("invalid hydrogen/heavy-atom bond assignment")
    if any(count != 1 for count in hydrogen_bonds.values()):
        raise OverlapAssemblyError("a hydrogen is assigned to more than one heavy atom")
    expected_hydrogens = sum(1 for symbol in atoms if symbol == "H")
    if len(hydrogen_bonds) != expected_hydrogens:
        raise OverlapAssemblyError("not every hydrogen has exactly one target attachment")


def write_overlap_assembly(
    result: OverlapAssemblyResult,
    xyz_path: Path,
    *,
    manifest_path: Path | None = None,
    embed_assembly_section: bool = False,
) -> tuple[Path, Path]:
    """Write a strict XYZ plus a machine-readable assembly manifest.

    ``embed_assembly_section`` retains support for MATRIX enriched XYZ files,
    while the strict default is directly readable by Avogadro and other XYZ
    consumers that reject trailing records.
    """
    xyz_target = Path(xyz_path)
    manifest_target = (
        Path(manifest_path)
        if manifest_path is not None
        else xyz_target.with_suffix(xyz_target.suffix + ".assembly.json")
    )
    xyz_target.parent.mkdir(parents=True, exist_ok=True)
    manifest_target.parent.mkdir(parents=True, exist_ok=True)
    xyz_lines = [
        str(len(result.atoms)),
        "ORACLE overlap-constrained LCB25 assembly",
    ]
    xyz_lines.extend(
        f"{symbol:<2s} {x: .10f} {y: .10f} {z: .10f}"
        for symbol, (x, y, z) in zip(result.atoms, result.coordinates_angstrom, strict=True)
    )
    if embed_assembly_section:
        xyz_lines.extend(["", "#ASSEMBLY", *_assembly_section_lines(result)])
    xyz_target.write_text("\n".join(xyz_lines) + "\n", encoding="utf-8")
    manifest = {
        "schema": ORACLE_XYZ_ASSEMBLY_SCHEMA,
        "target_smiles": result.smiles,
        "atom_count": len(result.atoms),
        "target_heavy_atoms": result.target_heavy_atoms,
        "covered_heavy_atoms": result.covered_heavy_atoms,
        "max_fragment_atoms": result.max_fragment_atoms,
        "min_overlap_atoms": result.min_overlap_atoms,
        "uff_status": result.uff_status,
        "uff_message": result.uff_message,
        "relaxation_method": result.relaxation_method,
        "search": {
            "method": result.search_method,
            "beam_width": result.beam_width,
            "fragment_workers": result.fragment_workers,
        },
        "pose_optimization": {
            "iterations": result.pose_iterations,
            "converged": result.pose_converged,
            "max_overlap_rmsd_angstrom": result.max_overlap_rmsd_angstrom,
        },
        "quality": {
            "status": "PASS" if not result.quality_warnings else "WARNING",
            "warnings": list(result.quality_warnings),
        },
        "atom_provenance": [
            {
                "target_atom": target_atom,
                "sources": list(sources),
            }
            for target_atom, sources in enumerate(result.atom_provenance, start=1)
        ],
        "steps": [_step_mapping(step) for step in result.steps],
    }
    manifest_target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return xyz_target, manifest_target


def _assembly_section_lines(result: OverlapAssemblyResult) -> list[str]:
    lines = [
        f"SCHEMA {ORACLE_XYZ_ASSEMBLY_SCHEMA}",
        "STATUS BUILT",
        "INDEXING ATOMS=ONE_BASED",
        f"TARGET_SMILES {result.smiles}",
        f"ATOM_COUNT {len(result.atoms)}",
        f"TARGET_HEAVY_ATOMS {result.target_heavy_atoms}",
        f"COVERED_HEAVY_ATOMS {result.covered_heavy_atoms}",
        f"MAX_FRAGMENT_ATOMS {result.max_fragment_atoms}",
        f"MIN_OVERLAP_ATOMS {result.min_overlap_atoms}",
        f"UFF_STATUS {result.uff_status if result.uff_status is not None else 'UNAVAILABLE'}",
        f"RELAXATION_METHOD {result.relaxation_method}",
        f"SEARCH_METHOD {result.search_method}",
        f"BEAM_WIDTH {result.beam_width}",
        f"FRAGMENT_WORKERS {result.fragment_workers}",
        f"POSE_ITERATIONS {result.pose_iterations}",
        f"POSE_CONVERGED {'YES' if result.pose_converged else 'NO'}",
        f"MAX_OVERLAP_RMSD {result.max_overlap_rmsd_angstrom:.8g}",
        f"QUALITY_STATUS {'PASS' if not result.quality_warnings else 'WARNING'}",
        f"QUALITY_WARNING_COUNT {len(result.quality_warnings)}",
        f"STEP_COUNT {len(result.steps)}",
        "[STEPS]",
    ]
    for index, step in enumerate(result.steps, start=1):
        targets = ",".join(str(value) for value in step.target_atoms)
        overlap = ",".join(str(value) for value in step.overlap_atoms) or "NONE"
        new = ",".join(str(value) for value in step.new_atoms)
        lines.append(
            f"S{index:03d} SOURCE={step.source_path} MATCH={step.match_index} "
            f"ATOMS={targets} OVERLAP={overlap} NEW={new} "
            f"SOURCE_ATOMS={step.source_atom_count} "
            f"SOURCE_HEAVY_ATOMS={step.source_heavy_atom_count} "
            f"OVERLAP_RMSD={step.overlap_rmsd_angstrom:.8g}"
        )
    lines.append("[ATOM_PROVENANCE]")
    for target_atom, sources in enumerate(result.atom_provenance, start=1):
        lines.append(f"A{target_atom:04d} SOURCES={';'.join(sources)}")
    if result.quality_warnings:
        lines.append("[QUALITY_WARNINGS]")
        lines.extend(
            f"W{index:03d} {warning}"
            for index, warning in enumerate(result.quality_warnings, start=1)
        )
    return lines


def _step_mapping(step: AssemblyStep) -> dict[str, object]:
    return {
        "source": str(step.source_path),
        "match_index": step.match_index,
        "target_atoms": list(step.target_atoms),
        "overlap_atoms": list(step.overlap_atoms),
        "new_atoms": list(step.new_atoms),
        "source_atom_count": step.source_atom_count,
        "source_heavy_atom_count": step.source_heavy_atom_count,
        "overlap_rmsd_angstrom": step.overlap_rmsd_angstrom,
        "source_to_target_atoms": [
            {
                "source_atom": source_atom,
                "target_atom": target_atom,
            }
            for source_atom, target_atom in step.source_to_target_atoms
        ],
        "extraction": step.extraction,
    }


def _load_fragment_candidates(
    path: Path,
    *,
    target: SwitchMolecularGraph,
    max_fragment_atoms: int,
    extract_common_subgraphs: bool,
    minimum_common_atoms: int,
    strict_hydrogen_interfaces: bool = False,
    use_chirality: bool = True,
) -> tuple[list[_PlacementCandidate], tuple[Path, int] | None]:
    source, source_coordinates, heavy_source_indices, source_atom_count = (
        _read_xyz_molecule(
            path,
            include_hydrogen_counts=strict_hydrogen_interfaces,
        )
    )
    if source_atom_count > max_fragment_atoms:
        return [], (path, source_atom_count)
    query = source
    query_coordinates = source_coordinates
    matches = find_substructure_matches(
        target,
        query,
        uniquify=True,
        use_chirality=use_chirality,
    )
    candidates = [
        _PlacementCandidate(
            source_path=path,
            query=query,
            match=tuple(int(index) for index in match),
            match_index=match_index,
            source_atom_count=source_atom_count,
            coordinates_angstrom=query_coordinates,
            source_atom_indices=heavy_source_indices,
        )
        for match_index, match in enumerate(matches)
    ]
    if candidates or not extract_common_subgraphs:
        return candidates, None

    common = maximum_common_connected_subgraphs(
        query,
        target,
        minimum_atoms=minimum_common_atoms,
        timeout_seconds=5.0,
        max_matches=256,
        hydrogen_mode="interface" if strict_hydrogen_interfaces else "ignore",
        induced=strict_hydrogen_interfaces,
    )
    if not common:
        return [], None
    candidates = []
    for match in common:
        coordinates = query_coordinates[list(match.source_atoms)]
        source_indices = tuple(
            heavy_source_indices[index] for index in match.source_atoms
        )
        extracted = _induced_graph(query, match.source_atoms)
        candidates.append(
            _PlacementCandidate(
                source_path=path,
                query=extracted,
                match=tuple(int(index) for index in match.target_atoms),
                match_index=len(candidates),
                source_atom_count=source_atom_count,
                coordinates_angstrom=coordinates,
                source_atom_indices=source_indices,
                extraction="maximum_common_subgraph",
            )
        )
    return candidates, None


def _read_xyz_molecule(
    path: Path,
    *,
    include_hydrogen_counts: bool = False,
) -> tuple[SwitchMolecularGraph, np.ndarray, tuple[int, ...], int]:
    try:
        geometry = read_xyz(path)
    except (OSError, ValueError) as exc:
        raise OverlapAssemblyError(f"cannot read fragment geometry: {path}") from exc
    try:
        numbers = tuple(int(atomic_number(symbol) or 0) for symbol in geometry.atoms)
        _continuous, discrete, _rings, synthons, aromaticity = build_topology_objects(
            geometry.coordinates_angstrom,
            numbers,
        )
    except Exception as exc:
        raise OverlapAssemblyError(
            f"ORACLE could not determine fragment bonds: {path}"
        ) from exc
    heavy_source_indices = tuple(
        index for index, number in enumerate(numbers) if number > 1
    )
    hydrogen_counts = None
    if include_hydrogen_counts:
        hydrogen_counts = tuple(
            sum(
                1
                for neighbor in range(len(numbers))
                if numbers[neighbor] == 1
                and any(
                    (left == index and right == neighbor)
                    or (left == neighbor and right == index)
                    for left, right in discrete.bonds
                )
            )
            for index in heavy_source_indices
        )
    old_to_new = {
        old: new for new, old in enumerate(heavy_source_indices)
    }
    aromatic_atoms = {
        old_to_new[index]
        for index in aromaticity.aromatic_atoms
        if index in old_to_new
    }
    heavy_bonds = [
        (old_to_new[left], old_to_new[right])
        for left, right in discrete.bonds
        if left in old_to_new and right in old_to_new
    ]
    bond_orders = {}
    for left, right in discrete.bonds:
        if left not in old_to_new or right not in old_to_new:
            continue
        key = tuple(sorted((old_to_new[left], old_to_new[right])))
        if left in aromaticity.aromatic_atoms and right in aromaticity.aromatic_atoms:
            bond_orders[key] = 1.5
        else:
            estimate = float(synthons.bond_order(left, right))
            bond_orders[key] = 3.0 if estimate >= 2.4 else 2.0 if estimate >= 1.65 else 1.0
    graph = graph_from_topology(
        tuple(atomic_symbol(numbers[index]) for index in heavy_source_indices),
        heavy_bonds,
        bond_orders=bond_orders,
        hydrogen_counts=hydrogen_counts,
        aromatic_atoms=tuple(sorted(aromatic_atoms)),
    )
    return (
        graph,
        np.asarray(geometry.coordinates_angstrom[list(heavy_source_indices)], dtype=float),
        heavy_source_indices,
        len(geometry.atoms),
    )


def read_xyz_switch_graph(path: Path) -> SwitchMolecularGraph:
    """Read an XYZ fragment as a SWITCH graph with inferred H counts."""

    return _read_xyz_molecule(Path(path), include_hydrogen_counts=True)[0]


def _transfer_fragment_hydrogen_positions(
    coordinates: np.ndarray,
    additions,
    candidates: tuple[_PlacementCandidate, ...],
    target_heavy_coordinates: np.ndarray,
) -> None:
    """Reuse donor X--H orientations when the matched fragment defines them."""

    available: dict[int, list[np.ndarray]] = {}
    for candidate in candidates:
        geometry = read_xyz(candidate.source_path)
        numbers = tuple(int(atomic_number(symbol) or 0) for symbol in geometry.atoms)
        try:
            _continuous, discrete, _rings, _synthons, _aromaticity = (
                build_topology_objects(geometry.coordinates_angstrom, numbers)
            )
        except Exception:
            continue
        rotation, translation = _rigid_alignment(
            candidate.coordinates_angstrom,
            target_heavy_coordinates[list(candidate.match)],
        )
        source_to_target = dict(
            zip(candidate.source_atom_indices, candidate.match, strict=True)
        )
        adjacency = [set() for _ in geometry.atoms]
        for left, right in discrete.bonds:
            adjacency[left].add(right)
            adjacency[right].add(left)
        for left, right in discrete.bonds:
            if numbers[left] == 1 and right in source_to_target:
                hydrogen, parent = left, right
            elif numbers[right] == 1 and left in source_to_target:
                hydrogen, parent = right, left
            else:
                continue
            point = (
                np.asarray(geometry.coordinates_angstrom[hydrogen], dtype=float)
                @ rotation
                + translation
            )
            local_point = _local_hydrogen_transfer_point(
                geometry.coordinates_angstrom,
                hydrogen=hydrogen,
                parent=parent,
                adjacency=adjacency,
                source_to_target=source_to_target,
                target_coordinates=target_heavy_coordinates,
                atomic_numbers=numbers,
            )
            if local_point is not None:
                point = local_point
            available.setdefault(source_to_target[parent], []).append(point)

    additions_by_parent: dict[int, list[int]] = {}
    for addition in additions:
        additions_by_parent.setdefault(int(addition.parent), []).append(int(addition.atom))
    for parent, hydrogen_atoms in additions_by_parent.items():
        points = available.get(parent, ())
        unique: list[np.ndarray] = []
        for point in points:
            if not any(float(np.linalg.norm(point - old)) < 0.15 for old in unique):
                unique.append(point)
        if len(unique) < len(hydrogen_atoms):
            continue
        unique.sort(
            key=lambda point: (
                float(point[0]),
                float(point[1]),
                float(point[2]),
            )
        )
        for atom, point in zip(hydrogen_atoms, unique, strict=False):
            coordinates[atom] = point


def _local_hydrogen_transfer_point(
    source_coordinates: np.ndarray,
    *,
    hydrogen: int,
    parent: int,
    adjacency: list[set[int]],
    source_to_target: dict[int, int],
    target_coordinates: np.ndarray,
    atomic_numbers: tuple[int, ...],
) -> np.ndarray | None:
    """Transfer an X--H vector in a mapped two-bond local frame."""

    mapped_neighbors = sorted(
        neighbor
        for neighbor in adjacency[parent]
        if atomic_numbers[neighbor] > 1 and neighbor in source_to_target
    )
    if not mapped_neighbors:
        return None
    first = mapped_neighbors[0]
    if len(mapped_neighbors) >= 2:
        second = mapped_neighbors[1]
    else:
        second_shell = sorted(
            neighbor
            for neighbor in adjacency[first]
            if (
                neighbor != parent
                and atomic_numbers[neighbor] > 1
                and neighbor in source_to_target
            )
        )
        if not second_shell:
            return None
        second = second_shell[0]

    source = np.asarray(source_coordinates, dtype=float)
    source_origin = source[parent]
    target_origin = target_coordinates[source_to_target[parent]]
    source_frame = _local_frame(
        source[first] - source_origin,
        source[second] - source_origin,
    )
    target_frame = _local_frame(
        target_coordinates[source_to_target[first]] - target_origin,
        target_coordinates[source_to_target[second]] - target_origin,
    )
    if source_frame is None or target_frame is None:
        return None
    components = source_frame @ (source[hydrogen] - source_origin)
    return target_origin + target_frame.T @ components


def _local_frame(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray | None:
    first_norm = float(np.linalg.norm(primary))
    if first_norm < 1.0e-10:
        return None
    first = primary / first_norm
    second = secondary - float(np.dot(secondary, first)) * first
    second_norm = float(np.linalg.norm(second))
    if second_norm < 1.0e-10:
        return None
    second /= second_norm
    third = np.cross(first, second)
    return np.vstack((first, second, third))


def _rigid_alignment(moving: np.ndarray, fixed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    moving_center = moving.mean(axis=0)
    fixed_center = fixed.mean(axis=0)
    moving_zero = moving - moving_center
    if np.linalg.matrix_rank(moving_zero, tol=1.0e-8) < 2:
        raise OverlapAssemblyError("overlap atoms are collinear; 3D placement is underdetermined")
    covariance = moving_zero.T @ (fixed - fixed_center)
    left, _, right_t = np.linalg.svd(covariance)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
    translation = fixed_center - moving_center @ rotation
    return rotation, translation


def _steric_release_seed(
    coordinates: np.ndarray,
    target: SwitchMolecularGraph,
) -> np.ndarray:
    """Break the exactly planar fused-ring saddle without supplying target geometry."""
    result = np.asarray(coordinates, dtype=float).copy()
    centered = result - result.mean(axis=0)
    _, _, right_t = np.linalg.svd(centered, full_matrices=False)
    normal = right_t[-1]
    graph_distance = _graph_distances(target, 0)
    midpoint = 0.5 * (min(graph_distance) + max(graph_distance))
    for index, distance in enumerate(graph_distance):
        result[index] += normal * (0.035 * (distance - midpoint))
    return result


def _graph_distances(molecule: SwitchMolecularGraph, start: int) -> list[int]:
    distances = [-1] * len(molecule.atoms)
    distances[start] = 0
    queue = [start]
    for atom_index in queue:
        for neighbor_index in molecule.neighbors(atom_index):
            if distances[neighbor_index] >= 0:
                continue
            distances[neighbor_index] = distances[atom_index] + 1
            queue.append(neighbor_index)
    return distances


def _heavy_graph(graph: SwitchMolecularGraph) -> SwitchMolecularGraph:
    indices = tuple(
        atom.index for atom in graph.atoms if atom.symbol != "H"
    )
    return _induced_graph(graph, indices)


def _induced_graph(
    graph: SwitchMolecularGraph,
    indices: tuple[int, ...],
) -> SwitchMolecularGraph:
    lookup = {old: new for new, old in enumerate(indices)}
    atoms = tuple(graph.atoms[index].symbol for index in indices)
    charges = tuple(graph.atoms[index].formal_charge for index in indices)
    aromatic = tuple(
        lookup[index] for index in indices if graph.atoms[index].aromatic
    )
    bonds = []
    orders = {}
    for bond in graph.bonds:
        if bond.left not in lookup or bond.right not in lookup:
            continue
        left, right = lookup[bond.left], lookup[bond.right]
        bonds.append((left, right))
        orders[tuple(sorted((left, right)))] = bond.order
    result = graph_from_topology(
        atoms,
        bonds,
        bond_orders=orders,
        formal_charges=charges,
        aromatic_atoms=aromatic,
        source_smiles=graph.source_smiles,
    )
    # Preserve bracket hydrogen requests and declared atom stereochemistry.
    remapped_atoms = tuple(
        replace(
            result.atoms[new],
            hydrogen_count=graph.atoms[old].hydrogen_count,
            chirality=graph.atoms[old].chirality,
            bracketed=graph.atoms[old].bracketed,
        )
        for new, old in enumerate(indices)
    )
    return SwitchMolecularGraph(
        atoms=remapped_atoms,
        bonds=result.bonds,
        components=result.components,
        source_smiles=result.source_smiles,
        total_formal_charge=result.total_formal_charge,
    )


def _steric_relaxation(
    atoms: tuple[str, ...],
    coordinates: np.ndarray,
    bonds: tuple[tuple[int, int], ...],
    *,
    maximum_iterations: int = 400,
    tolerance: float = 1.0e-6,
) -> tuple[np.ndarray, bool]:
    """Resolve severe nonbonded contacts with an analytic local gradient."""

    from matrix_chem.topology.covalent_radii import covalent_radius

    xyz = np.asarray(coordinates, dtype=float).copy()
    adjacency = [set() for _ in atoms]
    for left, right in bonds:
        adjacency[left].add(right)
        adjacency[right].add(left)
    excluded = {tuple(sorted(bond)) for bond in bonds}
    for center, neighbors in enumerate(adjacency):
        _ = center
        ordered = sorted(neighbors)
        excluded.update(
            (ordered[left], ordered[right])
            for left in range(len(ordered))
            for right in range(left + 1, len(ordered))
        )
    numbers = tuple(int(atomic_number(atom) or 0) for atom in atoms)
    radii = np.asarray(
        [float(covalent_radius(number) or 0.75) for number in numbers],
        dtype=float,
    )
    step = 0.025
    for _iteration in range(maximum_iterations):
        gradient = np.zeros_like(xyz)
        maximum_overlap = 0.0
        for left in range(len(atoms)):
            for right in range(left + 1, len(atoms)):
                if (left, right) in excluded:
                    continue
                vector = xyz[right] - xyz[left]
                distance = float(np.linalg.norm(vector))
                target = 0.72 * (radii[left] + radii[right])
                overlap = target - distance
                if overlap <= 0.0:
                    continue
                maximum_overlap = max(maximum_overlap, overlap)
                if distance < 1.0e-10:
                    direction = np.asarray((1.0, 0.0, 0.0))
                else:
                    direction = vector / distance
                force = overlap * direction
                gradient[left] -= force
                gradient[right] += force
        if maximum_overlap <= tolerance:
            return xyz, True
        displacement = step * gradient
        norms = np.linalg.norm(displacement, axis=1)
        scale = np.minimum(1.0, 0.04 / np.maximum(norms, 1.0e-15))
        xyz += displacement * scale[:, None]
    return xyz, False
