"""LCB26-backed construction of amino acids and peptide starting geometries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence

import numpy as np

from matrix_chem import (
    build_topology_objects,
    read_xyz,
    write_xyz,
)
from matrix_chem.geometry import MolecularGeometry
from matrix_chem.primitive_coordinates import dihedral
from matrix_chem.topology.elements import atomic_number
from matrix_switch import (
    build_cartesian_seed,
    complete_graph_hydrogens,
    graph_from_topology,
    maximum_common_connected_subgraphs,
    parse_smiles,
)

from .api import analyze_structure
from .initial_structure import prepare_initial_structure
from .lcb26 import load_lcb26_reference, query_lcb26


PEPTIDE_LIBRARY_SCHEMA = "matrix.lcb26.amino_acid_library.v1"
PEPTIDE_BUILD_SCHEMA = "matrix.oracle.peptide_build.v1"


class PeptideBuildError(ValueError):
    """Raised when a peptide request is chemically incomplete or inconsistent."""


@dataclass(frozen=True)
class AminoAcidDefinition:
    one_letter: str
    three_letter: str
    name: str
    sidechain_variants: dict[str, str | None]
    monomer_smiles: str
    ring_source: str | None = None


@dataclass(frozen=True)
class PeptideBuild:
    schema: str
    sequence_one_letter: str
    sequence_three_letter: tuple[str, ...]
    residue_states: tuple[str, ...]
    n_terminus: str
    c_terminus: str
    conformation: str
    smiles: str
    charge: int
    multiplicity: int
    output_xyz: str
    output_xyzin: str
    report: str
    backbone: tuple[dict[str, Any], ...]
    lcb26_sources: tuple[str, ...]
    constrained_backbone_torsions: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_amino_acid_library(lcb26_root: Path | str) -> dict[str, Any]:
    path = Path(lcb26_root).expanduser().resolve() / "amino_acids.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PeptideBuildError(f"invalid LCB26 amino-acid library: {path}") from exc
    if payload.get("schema") != PEPTIDE_LIBRARY_SCHEMA:
        raise PeptideBuildError(f"unsupported amino-acid library schema: {path}")
    residues = payload.get("residues", ())
    if len(residues) != 20:
        raise PeptideBuildError("the LCB26 amino-acid library must contain 20 residues")
    return payload


def amino_acid_definitions(
    lcb26_root: Path | str,
) -> tuple[AminoAcidDefinition, ...]:
    payload = load_amino_acid_library(lcb26_root)
    definitions = tuple(AminoAcidDefinition(**row) for row in payload["residues"])
    one_letter = {item.one_letter for item in definitions}
    three_letter = {item.three_letter for item in definitions}
    if len(one_letter) != 20 or len(three_letter) != 20:
        raise PeptideBuildError("amino-acid identifiers are not unique")
    return definitions


def query_amino_acid(
    lcb26_root: Path | str,
    selector: str,
) -> AminoAcidDefinition:
    needle = str(selector).strip().casefold()
    matches = [
        item
        for item in amino_acid_definitions(lcb26_root)
        if needle
        in {
            item.one_letter.casefold(),
            item.three_letter.casefold(),
            item.name.casefold(),
        }
    ]
    if len(matches) != 1:
        raise PeptideBuildError(f"unknown or ambiguous amino acid: {selector!r}")
    return matches[0]


def parse_peptide_sequence(
    sequence: str | Sequence[str],
    lcb26_root: Path | str,
) -> tuple[AminoAcidDefinition, ...]:
    definitions = amino_acid_definitions(lcb26_root)
    by_one = {item.one_letter: item for item in definitions}
    by_three = {item.three_letter: item for item in definitions}
    if isinstance(sequence, str):
        raw = sequence.strip().upper()
        if not raw:
            raise PeptideBuildError("peptide sequence cannot be empty")
        if any(separator in raw for separator in ("-", " ", ",")):
            tokens = tuple(
                token for token in raw.replace(",", " ").replace("-", " ").split() if token
            )
        else:
            tokens = tuple(raw)
    else:
        tokens = tuple(str(token).strip().upper() for token in sequence)
    resolved = []
    for token in tokens:
        definition = by_one.get(token) if len(token) == 1 else by_three.get(token)
        if definition is None:
            raise PeptideBuildError(f"unknown amino-acid code: {token!r}")
        resolved.append(definition)
    return tuple(resolved)


def build_peptide(
    sequence: str | Sequence[str],
    output: Path | str,
    *,
    lcb26_root: Path | str,
    n_terminus: str = "amine",
    c_terminus: str = "carboxylic_acid",
    conformation: str = "fully_extended",
    residue_states: Mapping[int, str] | None = None,
    backbone_angles: Mapping[int, Mapping[str, float]] | None = None,
    sidechain_angles: Mapping[int, Sequence[float]] | None = None,
    multiplicity: int = 1,
    terminal_oh_conformation: str | None = None,
) -> PeptideBuild:
    """Construct a peptide N-to-C using LCB26 donors and explicit backbone targets."""

    if int(multiplicity) < 1:
        raise PeptideBuildError("multiplicity must be positive")
    root = Path(lcb26_root).expanduser().resolve()
    library = load_amino_acid_library(root)
    residues = parse_peptide_sequence(sequence, root)
    n_label = str(n_terminus).strip().casefold()
    c_label = str(c_terminus).strip().casefold()
    if n_label not in library["termini"]["n"]:
        raise PeptideBuildError(f"unsupported N terminus: {n_terminus!r}")
    if c_label not in library["termini"]["c"]:
        raise PeptideBuildError(f"unsupported C terminus: {c_terminus!r}")
    oh_label = None if terminal_oh_conformation is None else str(terminal_oh_conformation).strip().casefold()
    if oh_label not in {None, "syn", "anti"}:
        raise PeptideBuildError("terminal_oh_conformation must be syn, anti, or None")
    if oh_label is not None and c_label != "carboxylic_acid":
        raise PeptideBuildError("terminal OH conformation requires a carboxylic_acid C terminus")
    preset_label = str(conformation).strip().casefold()
    presets = library["backbone_presets_degrees"]
    preset_key = next((key for key in presets if str(key).casefold() == preset_label), None)
    if preset_key is None and preset_label != "custom":
        raise PeptideBuildError(f"unsupported peptide conformation: {conformation!r}")
    state_by_position = {int(key): str(value) for key, value in (residue_states or {}).items()}
    invalid_positions = sorted(
        position for position in state_by_position if position < 1 or position > len(residues)
    )
    if invalid_positions:
        raise PeptideBuildError(f"invalid residue-state positions: {invalid_positions}")
    smiles, backbone, states = _peptide_smiles(
        residues,
        n_terminus=n_label,
        c_terminus=c_label,
        state_by_position=state_by_position,
    )
    graph = parse_smiles(smiles)
    heavy_atom_count = len(graph.atoms)
    target_rows = _backbone_targets(
        backbone,
        graph=graph,
        n_terminus=n_label,
        c_terminus=c_label,
        preset=presets.get(preset_key, {}) if preset_key is not None else {},
        residue_overrides=next(
            (
                value
                for key, value in library.get("residue_backbone_overrides_degrees", {}).items()
                if str(key).casefold() == preset_label
            ),
            {},
        ),
        overrides=backbone_angles or {},
    )
    adjacency = _constitutional_adjacency(graph)
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="oracle-peptide-build-") as scratch:
        shaped_seed = (
            _build_sequential_peptide_seed(
                graph,
                backbone,
                target_rows,
                root,
                library,
                multiplicity=int(multiplicity),
                title=smiles,
            )
            if len(residues) > 1
            else build_cartesian_seed(
                graph,
                title=smiles,
                multiplicity=int(multiplicity),
                complete_hydrogens=False,
            )
        )
        seed_coordinates = np.asarray(shaped_seed.coordinates_angstrom, dtype=float).copy()
        _apply_peptide_targets(seed_coordinates, adjacency, target_rows)
        seed_with_hydrogens = _complete_peptide_hydrogens(
            graph,
            seed_coordinates,
            multiplicity=int(multiplicity),
            comment=f"MATRIX pre-shaped peptide {''.join(item.one_letter for item in residues)}",
        )
        preliminary = Path(scratch) / "preliminary_seed.xyz"
        write_xyz(
            preliminary,
            seed_with_hydrogens.atoms,
            seed_with_hydrogens.coordinates_angstrom,
            comment=seed_with_hydrogens.comment,
        )
        prepared = Path(scratch) / "prepared.xyz"
        preparation = prepare_initial_structure(
            preliminary,
            prepared,
            lcb26_root=root,
            source_kind="xyz",
            constitutional_smiles=smiles,
            preserved_dihedrals=tuple(
                tuple(int(index) for index in target["atoms_zero_based"]) for target in target_rows
            ),
        )
        geometry = read_xyz(preparation.output_xyz)
        donor_audit = json.loads(Path(preparation.report).read_text(encoding="utf-8"))[
            "donor_audit"
        ]
    if tuple(atom.symbol for atom in graph.atoms) != geometry.atoms[: len(graph.atoms)]:
        raise PeptideBuildError("SWITCH/ORACLE peptide atom ordering changed unexpectedly")
    coordinates = np.asarray(geometry.coordinates_angstrom[:heavy_atom_count], dtype=float).copy()
    constraints = _apply_peptide_targets(coordinates, adjacency, target_rows)
    requested_chi1 = {int(key): tuple(float(item) for item in value) for key, value in (sidechain_angles or {}).items()}
    if requested_chi1:
        for position, targets in requested_chi1.items():
            if position < 1 or position > len(backbone):
                raise PeptideBuildError(f"invalid sidechain-angle position: {position}")
            atoms = backbone[position - 1]["atoms_zero_based"]
            alpha = int(atoms["CA"])
            candidates = [
                int(neighbour) for neighbour in adjacency[alpha]
                if int(neighbour) not in {int(atoms["N"]), int(atoms["C"])}
                and graph.atoms[int(neighbour)].symbol != "H"
            ]
            if not candidates:
                raise PeptideBuildError(f"residue {position} has no rotatable chi1 bond")
            path = [alpha, candidates[0]]
            while len(path) < len(targets) + 2:
                next_candidates = [
                    int(neighbour) for neighbour in adjacency[path[-1]]
                    if int(neighbour) not in path and graph.atoms[int(neighbour)].symbol != "H"
                ]
                if not next_candidates:
                    raise PeptideBuildError(f"residue {position} has no complete chi{len(path) - 1} dihedral")
                path.append(next_candidates[0])
            for chi_index, target in enumerate(targets):
                quartet = (int(atoms["N"]), alpha, path[1], path[2]) if chi_index == 0 else (path[chi_index - 1], path[chi_index], path[chi_index + 1], path[chi_index + 2])
                _set_peptide_dihedral(coordinates, adjacency, quartet, target)
    completed = _complete_peptide_hydrogens(
        graph,
        coordinates,
        multiplicity=int(multiplicity),
        comment=(f"MATRIX peptide {''.join(item.one_letter for item in residues)} {preset_label}"),
    )
    if c_label == "amide":
        completed = _planarize_terminal_amide(completed, graph, backbone)
    if oh_label is not None:
        completed = _set_terminal_carboxyl_oh_conformation(
            completed,
            graph,
            backbone,
            conformation=oh_label,
        )
    write_xyz(
        output_path,
        completed.atoms,
        completed.coordinates_angstrom,
        comment=completed.comment,
    )
    output_xyzin = output_path.with_suffix(".xyzin")
    audit = analyze_structure(output_path, output_xyzin, source_kind="xyz", validate=True)
    if audit.status != "PASS":
        raise PeptideBuildError(f"ORACLE rejected the constructed peptide: {audit.status}")
    sources = _lcb26_sources(donor_audit)
    report_path = output_path.with_suffix(".peptide.json")
    result = PeptideBuild(
        schema=PEPTIDE_BUILD_SCHEMA,
        sequence_one_letter="".join(item.one_letter for item in residues),
        sequence_three_letter=tuple(item.three_letter for item in residues),
        residue_states=states,
        n_terminus=n_label,
        c_terminus=c_label,
        conformation=preset_label,
        smiles=smiles,
        charge=int(graph.total_formal_charge),
        multiplicity=int(multiplicity),
        output_xyz=str(output_path),
        output_xyzin=str(output_xyzin),
        report=str(report_path),
        backbone=tuple(backbone),
        lcb26_sources=sources,
        constrained_backbone_torsions=tuple(constraints),
    )
    payload = result.to_dict()
    payload["protocol"] = {
        "sequence_direction": "N_TO_C",
        "geometry_owner": "ORACLE",
        "constitution_owner": "SWITCH",
        "backbone_sources": library["backbone_sources"],
        "peptide_linkage_sources": library["peptide_linkage_sources"],
        "local_parameter_selection": "LCB26_SYNTHON_CM5_MAYER_NEAREST_NEIGHBORS",
        "cartesian_internal_reconciliation": "WEIGHTED_WILSON_B_PSEUDOINVERSE",
        "backbone_torsion_policy": "REQUESTED_REGULAR_STRUCTURE_FROZEN",
        "sidechain_relaxation": "ZAFF_FAST_SOFT_EXOCYCLIC_TORSIONS_ONLY",
        "ring_policy": "LCB26_COMPLETE_RING_DONORS_PRESERVED",
        "implicit_ph_policy": "NONE_RESIDUE_MICROSTATES_EXPLICIT",
        "terminal_oh_conformation": oh_label,
    }
    payload["oracle_audit"] = {
        "status": audit.status,
        "atom_count": audit.atom_count,
        "bond_count": audit.bond_count,
        "ring_count": audit.ring_count,
        "topology_sha256": audit.topology_sha256,
    }
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _constitutional_adjacency(graph: Any) -> dict[int, set[int]]:
    adjacency = {index: set() for index in range(len(graph.atoms))}
    for bond in graph.bonds:
        adjacency[int(bond.left)].add(int(bond.right))
        adjacency[int(bond.right)].add(int(bond.left))
    return adjacency


def _apply_peptide_targets(
    coordinates: np.ndarray,
    adjacency: Mapping[int, set[int]],
    targets: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    constraints = []
    for target in targets:
        atoms = tuple(int(index) for index in target["atoms_zero_based"])
        status = _set_peptide_dihedral(
            coordinates,
            adjacency,
            atoms,
            float(target["target_degrees"]),
        )
        constraints.append({**target, "status": status})
    for item in constraints:
        atoms = tuple(int(index) for index in item["atoms_zero_based"])
        item["actual_degrees"] = _peptide_dihedral_degrees(coordinates, atoms)
        item["periodic_error_degrees"] = _periodic_degrees(
            item["actual_degrees"] - float(item["target_degrees"])
        )
        if item["status"].startswith("SET") and abs(item["periodic_error_degrees"]) > 1.0e-6:
            raise PeptideBuildError("a peptide backbone target was not reproduced")
    return constraints


def _complete_peptide_hydrogens(
    graph: Any,
    coordinates: np.ndarray,
    *,
    multiplicity: int,
    comment: str,
) -> MolecularGeometry:
    heavy_geometry = MolecularGeometry(
        atoms=tuple(atom.symbol for atom in graph.atoms),
        coordinates_angstrom=coordinates,
        comment=comment,
        source_format="SMILES",
        charge=int(graph.total_formal_charge),
        multiplicity=int(multiplicity),
    )
    return complete_graph_hydrogens(graph, heavy_geometry).geometry


def _set_terminal_carboxyl_oh_conformation(
    geometry: MolecularGeometry,
    graph: Any,
    backbone: Sequence[Mapping[str, Any]],
    *,
    conformation: str,
) -> MolecularGeometry:
    """Set the terminal C(alpha)-C(=O)-O-H syn/anti torsion exactly."""

    terminal = backbone[-1]["atoms_zero_based"]
    carbonyl = int(terminal["C"])
    alpha = int(terminal["CA"])
    oxygens = [
        int(neighbour)
        for neighbour in graph.neighbors(carbonyl)
        if graph.atoms[int(neighbour)].symbol == "O"
    ]
    if len(oxygens) != 2:
        raise PeptideBuildError("carboxylic-acid terminal carbon must have two oxygens")
    coordinates = np.asarray(geometry.coordinates_angstrom, dtype=float).copy()
    hydroxyl = next(
        (
            oxygen
            for oxygen in oxygens
            if any(
                symbol == "H" and float(np.linalg.norm(coordinates[index] - coordinates[oxygen])) < 1.25
                for index, symbol in enumerate(geometry.atoms)
            )
        ),
        None,
    )
    if hydroxyl is None:
        raise PeptideBuildError("terminal carboxylic acid has no identifiable OH hydrogen")
    hydrogen = next(
        index
        for index, symbol in enumerate(geometry.atoms)
        if symbol == "H" and float(np.linalg.norm(coordinates[index] - coordinates[hydroxyl])) < 1.25
    )
    current = dihedral(alpha, carbonyl, hydroxyl, hydrogen, coordinates)
    target = 0.0 if conformation == "syn" else math.pi
    delta = _periodic_radians(target - current)
    axis = coordinates[hydroxyl] - coordinates[carbonyl]
    axis /= np.linalg.norm(axis)
    vector = coordinates[hydrogen] - coordinates[carbonyl]
    coordinates[hydrogen] = (
        coordinates[carbonyl]
        + vector * math.cos(delta)
        + np.cross(axis, vector) * math.sin(delta)
        + axis * float(np.dot(axis, vector)) * (1.0 - math.cos(delta))
    )
    actual = dihedral(alpha, carbonyl, hydroxyl, hydrogen, coordinates)
    if abs(_periodic_radians(actual - target)) > 1.0e-6:
        raise PeptideBuildError("terminal carboxyl OH conformation was not reproduced")
    return MolecularGeometry(
        atoms=geometry.atoms,
        coordinates_angstrom=coordinates,
        comment=geometry.comment,
        source_format=geometry.source_format,
        source_path=geometry.source_path,
        charge=geometry.charge,
        multiplicity=geometry.multiplicity,
        fixed_parameters=geometry.fixed_parameters,
        metadata={
            **dict(geometry.metadata),
            "terminal_oh_conformation": conformation,
            "terminal_oh_dihedral_degrees": float(math.degrees(actual)),
        },
    )


def _planarize_terminal_amide(
    geometry: MolecularGeometry,
    graph: Any,
    backbone: Sequence[Mapping[str, Any]],
) -> MolecularGeometry:
    """Place terminal -CONH2 in the peptide carbonyl plane."""
    terminal = backbone[-1]["atoms_zero_based"]
    carbonyl = int(terminal["C"])
    alpha = int(terminal["CA"])
    amide_n = next(
        (int(neighbour) for neighbour in graph.neighbors(carbonyl) if graph.atoms[int(neighbour)].symbol == "N"),
        None,
    )
    if amide_n is None:
        raise PeptideBuildError("terminal amide carbon has no nitrogen")
    coordinates = np.asarray(geometry.coordinates_angstrom, dtype=float).copy()
    hydrogens = [
        index for index, symbol in enumerate(geometry.atoms)
        if symbol == "H" and float(np.linalg.norm(coordinates[index] - coordinates[amide_n])) < 1.25
    ]
    if len(hydrogens) != 2:
        raise PeptideBuildError("terminal amide must have exactly two N-H hydrogens")
    axis = coordinates[carbonyl] - coordinates[amide_n]
    axis /= np.linalg.norm(axis)
    in_plane = coordinates[alpha] - coordinates[amide_n]
    in_plane -= axis * float(np.dot(in_plane, axis))
    if np.linalg.norm(in_plane) < 1.0e-10:
        oxygens = [int(neighbour) for neighbour in graph.neighbors(carbonyl) if graph.atoms[int(neighbour)].symbol == "O"]
        if not oxygens:
            raise PeptideBuildError("cannot define terminal amide plane")
        in_plane = coordinates[oxygens[0]] - coordinates[amide_n]
        in_plane -= axis * float(np.dot(in_plane, axis))
    in_plane /= np.linalg.norm(in_plane)
    distance = float(np.mean([np.linalg.norm(coordinates[index] - coordinates[amide_n]) for index in hydrogens]))
    normal = np.cross(axis, in_plane)
    normal /= np.linalg.norm(normal)
    # The two H vectors have 120 degrees to the N-C(O) bond and are mirrored
    # in the C-alpha/C(O)/N plane, as required for a planar primary amide.
    for sign, index in zip((-1.0, 1.0), hydrogens, strict=True):
        vector = -0.5 * axis + sign * (math.sqrt(3.0) / 2.0) * in_plane
        coordinates[index] = coordinates[amide_n] + distance * vector
    return MolecularGeometry(
        atoms=geometry.atoms,
        coordinates_angstrom=coordinates,
        comment=geometry.comment,
        source_format=geometry.source_format,
        source_path=geometry.source_path,
        charge=geometry.charge,
        multiplicity=geometry.multiplicity,
        fixed_parameters=geometry.fixed_parameters,
        metadata={**dict(geometry.metadata), "terminal_amide_nh2_planar": True},
    )


def _periodic_radians(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _build_sequential_peptide_seed(
    graph: Any,
    backbone: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    lcb26_root: Path,
    library: Mapping[str, Any],
    *,
    multiplicity: int,
    title: str,
) -> MolecularGeometry:
    """Assemble a polypeptide residue-by-residue on an LCB26 backbone."""

    metrics = _peptide_linkage_metrics(lcb26_root, library)
    requested = {
        (int(row["residue_position"]), str(row["label"])): float(row["target_degrees"])
        for row in targets
    }
    backbone_coordinates: list[dict[str, np.ndarray]] = []
    first_n = np.asarray((0.0, 0.0, 0.0))
    first_ca = np.asarray((metrics["n_ca"], 0.0, 0.0))
    theta = math.radians(metrics["n_ca_c_angle_degrees"])
    first_c = first_ca + metrics["ca_c"] * np.asarray((-math.cos(theta), math.sin(theta), 0.0))
    backbone_coordinates.append({"N": first_n, "CA": first_ca, "C": first_c})
    for residue_index in range(1, len(backbone)):
        previous = backbone_coordinates[-1]
        previous_position = residue_index
        current_position = residue_index + 1
        next_n = _place_from_internal_coordinates(
            previous["C"],
            previous["CA"],
            previous["N"],
            length=metrics["c_n"],
            angle_degrees=metrics["ca_c_n_angle_degrees"],
            raw_dihedral_degrees=(requested[(previous_position, "psi")] + 180.0),
        )
        next_ca = _place_from_internal_coordinates(
            next_n,
            previous["C"],
            previous["CA"],
            length=metrics["n_ca"],
            angle_degrees=metrics["c_n_ca_angle_degrees"],
            raw_dihedral_degrees=(requested[(previous_position, "omega")] + 180.0),
        )
        next_c = _place_from_internal_coordinates(
            next_ca,
            next_n,
            previous["C"],
            length=metrics["ca_c"],
            angle_degrees=metrics["n_ca_c_angle_degrees"],
            raw_dihedral_degrees=(requested[(current_position, "phi")] + 180.0),
        )
        backbone_coordinates.append({"N": next_n, "CA": next_ca, "C": next_c})

    source = build_cartesian_seed(
        graph,
        title=title,
        multiplicity=multiplicity,
        complete_hydrogens=False,
    )
    coordinates = np.asarray(source.coordinates_angstrom, dtype=float).copy()
    residue_definitions = {str(item["three_letter"]): item for item in library["residues"]}
    cut_edges = {
        tuple(
            sorted(
                (
                    int(backbone[index]["atoms_zero_based"]["C"]),
                    int(backbone[index + 1]["atoms_zero_based"]["N"]),
                )
            )
        )
        for index in range(len(backbone) - 1)
    }
    adjacency = _constitutional_adjacency(graph)
    for left, right in cut_edges:
        adjacency[left].discard(right)
        adjacency[right].discard(left)
    assigned: set[int] = set()
    for residue, target_points in zip(backbone, backbone_coordinates, strict=True):
        atoms = residue["atoms_zero_based"]
        component = _graph_component(adjacency, int(atoms["CA"]))
        if assigned & component:
            raise PeptideBuildError("peptide residue components overlap")
        assigned.update(component)
        anchor_indices = np.asarray([int(atoms[label]) for label in ("N", "CA", "C")], dtype=int)
        target_anchor = np.vstack([target_points[label] for label in ("N", "CA", "C")])
        component_indices = np.asarray(sorted(component), dtype=int)
        coordinates[component_indices] = _kabsch_transform(
            coordinates[component_indices],
            coordinates[anchor_indices],
            target_anchor,
        )
        definition = residue_definitions[str(residue["three_letter"])]
        template_smiles = _residue_template_smiles(
            definition,
            state=str(residue["state"]),
            n_formal_charge=int(graph.atoms[int(atoms["N"])].formal_charge),
        )
        _install_refined_residue_template(
            coordinates,
            graph,
            component,
            atoms,
            target_anchor,
            template_smiles,
            lcb26_root,
        )
        coordinates[anchor_indices] = target_anchor
    if assigned != set(range(len(graph.atoms))):
        raise PeptideBuildError("peptide assembly left unassigned constitutional atoms")
    for index, residue in enumerate(backbone[:-1]):
        atoms = residue["atoms_zero_based"]
        target_psi = requested[(index + 1, "psi")]
        coordinates[int(atoms["O"])] = _place_from_internal_coordinates(
            coordinates[int(atoms["C"])],
            coordinates[int(atoms["CA"])],
            coordinates[int(atoms["N"])],
            length=metrics["c_o"],
            angle_degrees=metrics["ca_c_o_angle_degrees"],
            raw_dihedral_degrees=target_psi,
        )
    coordinates, rotamer_adjustments = _release_sidechain_topology_clashes(
        coordinates,
        graph,
        backbone,
    )
    return MolecularGeometry(
        atoms=tuple(atom.symbol for atom in graph.atoms),
        coordinates_angstrom=coordinates,
        comment=title,
        source_format="SMILES_SEQUENTIAL_LCB26_PEPTIDE",
        charge=int(graph.total_formal_charge),
        multiplicity=int(multiplicity),
        metadata={
            "assembly": "LCB26_C5_C7_SEQUENTIAL_RESIDUE_BACKBONE",
            "backbone_metrics": metrics,
            "sidechain_rotamer_preconditioning_count": rotamer_adjustments,
        },
    )


def _peptide_linkage_metrics(
    lcb26_root: Path,
    library: Mapping[str, Any],
) -> dict[str, float]:
    mapping = {
        key: int(value)
        for key, value in dict(library["peptide_linkage_template_atom_indices_zero_based"]).items()
    }
    observations = []
    for identifier in library["peptide_linkage_sources"]:
        rows = query_lcb26(lcb26_root, identifier=str(identifier), limit=1)
        if len(rows) != 1:
            raise PeptideBuildError(f"missing unique LCB26 peptide-linkage source: {identifier}")
        record = load_lcb26_reference(lcb26_root, rows[0])
        coordinates = np.asarray(record["coordinates_angstrom"], dtype=float)
        n_atom = mapping["N"]
        ca_atom = mapping["CA"]
        c_atom = mapping["C"]
        o_atom = mapping["O"]
        next_n = mapping["NEXT_N"]
        previous_c = mapping["PREVIOUS_C"]
        observations.append(
            {
                "n_ca": float(np.linalg.norm(coordinates[n_atom] - coordinates[ca_atom])),
                "ca_c": float(np.linalg.norm(coordinates[ca_atom] - coordinates[c_atom])),
                "c_n": float(np.linalg.norm(coordinates[c_atom] - coordinates[next_n])),
                "c_o": float(np.linalg.norm(coordinates[c_atom] - coordinates[o_atom])),
                "n_ca_c_angle_degrees": math.degrees(
                    _cartesian_angle(coordinates[n_atom], coordinates[ca_atom], coordinates[c_atom])
                ),
                "ca_c_n_angle_degrees": math.degrees(
                    _cartesian_angle(coordinates[ca_atom], coordinates[c_atom], coordinates[next_n])
                ),
                "ca_c_o_angle_degrees": math.degrees(
                    _cartesian_angle(coordinates[ca_atom], coordinates[c_atom], coordinates[o_atom])
                ),
                "c_n_ca_angle_degrees": math.degrees(
                    _cartesian_angle(
                        coordinates[previous_c], coordinates[n_atom], coordinates[ca_atom]
                    )
                ),
            }
        )
    return {key: float(np.mean([row[key] for row in observations])) for key in observations[0]}


def _residue_template_smiles(
    definition: Mapping[str, Any],
    *,
    state: str,
    n_formal_charge: int,
) -> str:
    three_letter = str(definition["three_letter"])
    if three_letter == "PRO":
        return "[NH2+]1CCC[C@@H]1C(=O)O" if n_formal_charge > 0 else "N1CCC[C@@H]1C(=O)O"
    n_token = "[NH3+]" if n_formal_charge > 0 else "N"
    if three_letter == "GLY":
        return f"{n_token}CC(=O)O"
    branch = dict(definition["sidechain_variants"]).get(state)
    if branch is None:
        raise PeptideBuildError(f"missing {three_letter} residue template for state {state!r}")
    return f"{n_token}[C@@H]({branch})C(=O)O"


def _install_refined_residue_template(
    coordinates: np.ndarray,
    peptide_graph: Any,
    component: set[int],
    backbone_atoms: Mapping[str, int],
    target_anchor: np.ndarray,
    monomer_smiles: str,
    lcb26_root: Path,
) -> None:
    template_graph, template_coordinates = _refined_residue_template(lcb26_root, monomer_smiles)
    ordered = tuple(sorted(int(atom) for atom in component))
    local = {atom: index for index, atom in enumerate(ordered)}
    component_bonds = tuple(
        (local[int(bond.left)], local[int(bond.right)])
        for bond in peptide_graph.bonds
        if int(bond.left) in component and int(bond.right) in component
    )
    component_orders = {
        tuple(sorted((local[int(bond.left)], local[int(bond.right)]))): float(bond.order)
        for bond in peptide_graph.bonds
        if int(bond.left) in component and int(bond.right) in component
    }
    component_graph = graph_from_topology(
        tuple(peptide_graph.atoms[atom].symbol for atom in ordered),
        component_bonds,
        bond_orders=component_orders,
        formal_charges=tuple(int(peptide_graph.atoms[atom].formal_charge) for atom in ordered),
        aromatic_atoms=tuple(local[atom] for atom in ordered if peptide_graph.atoms[atom].aromatic),
        source_smiles="PEPTIDE_RESIDUE_COMPONENT",
    )
    matches = maximum_common_connected_subgraphs(
        template_graph,
        component_graph,
        minimum_atoms=3,
        timeout_seconds=1.0,
        max_matches=64,
    )
    target_anchor_local = tuple(local[int(backbone_atoms[label])] for label in ("N", "CA", "C"))
    candidates = []
    for match in matches:
        reverse = {
            int(target): int(source)
            for source, target in zip(match.source_atoms, match.target_atoms, strict=True)
        }
        if all(index in reverse for index in target_anchor_local):
            source_anchor = tuple(reverse[index] for index in target_anchor_local)
            candidates.append((match.atom_count, source_anchor, match))
    if not candidates:
        raise PeptideBuildError(
            f"cannot map refined residue template {monomer_smiles!r} onto peptide"
        )
    _size, source_anchor, match = max(
        candidates,
        key=lambda item: (item[0], tuple(-index for index in item[1])),
    )
    aligned = _kabsch_transform(
        template_coordinates,
        template_coordinates[np.asarray(source_anchor, dtype=int)],
        target_anchor,
    )
    for source_atom, target_atom in zip(match.source_atoms, match.target_atoms, strict=True):
        coordinates[ordered[int(target_atom)]] = aligned[int(source_atom)]


def _refined_residue_template(
    lcb26_root: Path,
    monomer_smiles: str,
) -> tuple[Any, np.ndarray]:
    index = lcb26_root / "enriched" / "index.json"
    stat = index.stat()
    graph, coordinates = _cached_refined_residue_template(
        str(lcb26_root.resolve()),
        int(stat.st_mtime_ns),
        int(stat.st_size),
        str(monomer_smiles),
    )
    return graph, np.asarray(coordinates, dtype=float).copy()


@lru_cache(maxsize=64)
def _cached_refined_residue_template(
    lcb26_root_text: str,
    _index_mtime_ns: int,
    _index_size: int,
    monomer_smiles: str,
) -> tuple[Any, np.ndarray]:
    root = Path(lcb26_root_text)
    graph = parse_smiles(monomer_smiles)
    with TemporaryDirectory(prefix="oracle-residue-template-") as scratch:
        result = prepare_initial_structure(
            monomer_smiles,
            Path(scratch) / "residue.xyz",
            lcb26_root=root,
            source_kind="smiles",
        )
        geometry = read_xyz(result.output_xyz)
    coordinates = np.asarray(geometry.coordinates_angstrom[: len(graph.atoms)], dtype=float).copy()
    coordinates.setflags(write=False)
    return graph, coordinates


def _release_sidechain_topology_clashes(
    coordinates: np.ndarray,
    graph: Any,
    backbone: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, int]:
    """Scan only bridge-connected sidechain roots to preserve constitution."""

    xyz = np.asarray(coordinates, dtype=float).copy()
    adjacency = _constitutional_adjacency(graph)
    expected = {tuple(bond.key) for bond in graph.bonds}
    adjustments = 0
    for _ in range(3):
        changed = False
        for residue in backbone:
            atoms = residue["atoms_zero_based"]
            ca_atom = int(atoms["CA"])
            backbone_neighbors = {
                int(atoms["N"]),
                int(atoms["C"]),
            }
            for root in sorted(adjacency[ca_atom] - backbone_neighbors):
                edge = tuple(sorted((ca_atom, int(root))))
                component = _graph_component_after_cut(adjacency, int(root), edge)
                if ca_atom in component:
                    continue
                indices = np.asarray(sorted(component), dtype=int)
                baseline = xyz[indices].copy()
                axis = xyz[root] - xyz[ca_atom]
                norm = float(np.linalg.norm(axis))
                if norm <= 1.0e-12:
                    continue
                axis /= norm
                origin = xyz[ca_atom].copy()
                best_coordinates = baseline
                best_score = _peptide_topology_score(xyz, graph, expected)
                best_angle = 0
                for angle_degrees in (-180, -120, -60, 60, 120, 180):
                    angle = math.radians(float(angle_degrees))
                    vectors = baseline - origin
                    trial_component = origin + (
                        vectors * math.cos(angle)
                        + np.cross(axis, vectors) * math.sin(angle)
                        + np.outer(vectors @ axis, axis) * (1.0 - math.cos(angle))
                    )
                    trial = xyz.copy()
                    trial[indices] = trial_component
                    score = _peptide_topology_score(trial, graph, expected)
                    if score < best_score:
                        best_score = score
                        best_coordinates = trial_component
                        best_angle = angle_degrees
                if best_angle:
                    xyz[indices] = best_coordinates
                    adjustments += 1
                    changed = True
        if not changed or _peptide_topology_score(xyz, graph, expected)[0] == 0:
            break
    return xyz, adjustments


def _graph_component_after_cut(
    adjacency: Mapping[int, set[int]],
    seed: int,
    blocked_edge: tuple[int, int],
) -> set[int]:
    component = {int(seed)}
    queue = [int(seed)]
    for atom in queue:
        for neighbor in adjacency.get(atom, set()):
            if tuple(sorted((int(atom), int(neighbor)))) == blocked_edge:
                continue
            if int(neighbor) not in component:
                component.add(int(neighbor))
                queue.append(int(neighbor))
    return component


def _peptide_topology_score(
    coordinates: np.ndarray,
    graph: Any,
    expected: set[tuple[int, int]],
) -> tuple[int, float]:
    numbers = tuple(int(atomic_number(atom.symbol) or 0) for atom in graph.atoms)
    try:
        _continuous, realized_graph, _rings, _synthons, _aromaticity = build_topology_objects(
            coordinates, numbers
        )
        realized = {tuple(sorted((int(left), int(right)))) for left, right in realized_graph.bonds}
    except (RuntimeError, ValueError):
        return (10**6, float("inf"))
    topology_errors = len(expected - realized) + len(realized - expected)
    nonbonded = []
    for left in range(len(coordinates)):
        for right in range(left):
            if (right, left) in expected:
                continue
            nonbonded.append(float(np.linalg.norm(coordinates[left] - coordinates[right])))
    clearance = min(nonbonded, default=float("inf"))
    return topology_errors, -clearance


def _place_from_internal_coordinates(
    bond_reference: np.ndarray,
    angle_reference: np.ndarray,
    dihedral_reference: np.ndarray,
    *,
    length: float,
    angle_degrees: float,
    raw_dihedral_degrees: float,
) -> np.ndarray:
    p1 = np.asarray(bond_reference, dtype=float)
    p2 = np.asarray(angle_reference, dtype=float)
    p3 = np.asarray(dihedral_reference, dtype=float)
    ez = p1 - p2
    ez /= np.linalg.norm(ez)
    reference = p3 - p2
    axis_normal = np.cross(reference, ez)
    if float(np.linalg.norm(axis_normal)) <= 1.0e-12:
        reference = np.asarray((0.0, 0.0, 1.0))
        if abs(float(np.dot(reference, ez))) > 0.9:
            reference = np.asarray((0.0, 1.0, 0.0))
        axis_normal = np.cross(reference, ez)
    ex = axis_normal / np.linalg.norm(axis_normal)
    theta = math.radians(float(angle_degrees))
    candidate = p1 - float(length) * math.cos(theta) * ez + float(length) * math.sin(theta) * ex
    points = np.vstack((p3, p2, p1, candidate))
    current = math.degrees(float(dihedral(0, 1, 2, 3, points)))
    target = _periodic_degrees(raw_dihedral_degrees)
    magnitude = math.radians(abs(_periodic_degrees(target - current)))
    rotation_axis = p1 - p2
    rotation_axis /= np.linalg.norm(rotation_axis)
    best = candidate
    best_error = abs(_periodic_degrees(current - target))
    for signed in (-magnitude, magnitude):
        vector = candidate - p1
        trial = p1 + (
            vector * math.cos(signed)
            + np.cross(rotation_axis, vector) * math.sin(signed)
            + rotation_axis * float(np.dot(rotation_axis, vector)) * (1.0 - math.cos(signed))
        )
        trial_points = np.vstack((p3, p2, p1, trial))
        trial_dihedral = math.degrees(float(dihedral(0, 1, 2, 3, trial_points)))
        error = abs(_periodic_degrees(trial_dihedral - target))
        if error < best_error:
            best = trial
            best_error = error
    if best_error > 1.0e-7:
        raise PeptideBuildError("sequential peptide internal placement failed")
    return best


def _cartesian_angle(left: np.ndarray, center: np.ndarray, right: np.ndarray) -> float:
    first = np.asarray(left, dtype=float) - np.asarray(center, dtype=float)
    second = np.asarray(right, dtype=float) - np.asarray(center, dtype=float)
    cosine = float(np.dot(first, second)) / max(
        float(np.linalg.norm(first) * np.linalg.norm(second)), 1.0e-15
    )
    return float(math.acos(np.clip(cosine, -1.0, 1.0)))


def _graph_component(adjacency: Mapping[int, set[int]], seed: int) -> set[int]:
    component = {int(seed)}
    queue = [int(seed)]
    for atom in queue:
        for neighbor in adjacency.get(atom, set()):
            if int(neighbor) not in component:
                component.add(int(neighbor))
                queue.append(int(neighbor))
    return component


def _kabsch_transform(
    points: np.ndarray,
    source_anchor: np.ndarray,
    target_anchor: np.ndarray,
) -> np.ndarray:
    source = np.asarray(source_anchor, dtype=float)
    target = np.asarray(target_anchor, dtype=float)
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _singular, right = np.linalg.svd(covariance)
    rotation = left @ right
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right
    return target_center + (np.asarray(points, dtype=float) - source_center) @ rotation


def _peptide_smiles(
    residues: Sequence[AminoAcidDefinition],
    *,
    n_terminus: str,
    c_terminus: str,
    state_by_position: Mapping[int, str],
) -> tuple[str, tuple[dict[str, Any], ...], tuple[str, ...]]:
    chunks: list[str] = []
    length = 0

    def append(value: str) -> int:
        nonlocal length
        start = length
        chunks.append(value)
        length += len(value)
        return start

    if n_terminus == "formyl":
        append("C(=O)")
    elif n_terminus == "acetyl":
        append("CC(=O)")
    backbone = []
    states = []
    for position, residue in enumerate(residues, start=1):
        state = state_by_position.get(position, "neutral")
        if state not in residue.sidechain_variants:
            raise PeptideBuildError(
                f"unsupported {residue.three_letter} sidechain state: {state!r}"
            )
        states.append(state)
        if residue.three_letter == "PRO":
            if state != "neutral":
                raise PeptideBuildError("proline currently has only the neutral ring state")
            n_token = "[NH2+]1" if position == 1 and n_terminus == "ammonium" else "N1"
            n_offset = append(n_token)
            append("CCC")
            ca_offset = append("[C@@H]1")
        else:
            n_token = "[NH3+]" if position == 1 and n_terminus == "ammonium" else "N"
            n_offset = append(n_token)
            if residue.three_letter == "GLY":
                ca_offset = append("C")
            else:
                ca_offset = append("[C@@H]")
                branch = residue.sidechain_variants[state]
                if branch is None:
                    raise PeptideBuildError(f"missing sidechain for {residue.three_letter}")
                append(f"({branch})")
        carbonyl_offset = append("C")
        append("(=")
        oxygen_offset = append("O")
        append(")")
        backbone.append(
            {
                "position": position,
                "one_letter": residue.one_letter,
                "three_letter": residue.three_letter,
                "state": state,
                "source_offsets": {
                    "N": n_offset,
                    "CA": ca_offset,
                    "C": carbonyl_offset,
                    "O": oxygen_offset,
                },
            }
        )
    append(
        {
            "carboxylic_acid": "O",
            "carboxylate": "[O-]",
            "amide": "N",
            "n_methylamide": "NC",
        }[c_terminus]
    )
    smiles = "".join(chunks)
    graph = parse_smiles(smiles)
    by_offset = {atom.source_span[0]: atom.index for atom in graph.atoms}
    for residue in backbone:
        residue["atoms_zero_based"] = {
            label: by_offset[offset] for label, offset in residue.pop("source_offsets").items()
        }
        residue["atoms_one_based"] = {
            label: index + 1 for label, index in residue["atoms_zero_based"].items()
        }
    return smiles, tuple(backbone), tuple(states)


def _backbone_targets(
    backbone: Sequence[Mapping[str, Any]],
    *,
    graph: Any,
    n_terminus: str,
    c_terminus: str,
    preset: Mapping[str, float],
    overrides: Mapping[int, Mapping[str, float]],
    residue_overrides: Mapping[str, Mapping[str, float]] | None = None,
) -> tuple[dict[str, Any], ...]:
    invalid = sorted(
        int(position)
        for position in overrides
        if int(position) < 1 or int(position) > len(backbone)
    )
    if invalid:
        raise PeptideBuildError(f"invalid backbone-angle positions: {invalid}")
    rows = []
    for index, residue in enumerate(backbone):
        current = residue["atoms_zero_based"]
        position = index + 1
        requested = {
            **preset,
            **dict((residue_overrides or {}).get(str(residue["three_letter"]), {})),
            **dict(overrides.get(position, {})),
        }
        unknown = sorted(set(requested) - {"phi", "psi", "omega"})
        if unknown:
            raise PeptideBuildError(f"unknown backbone angles at residue {position}: {unknown}")
        if index == 0 and n_terminus == "formyl" and "phi" in requested:
            # A formyl-capped residue has a real N-terminal phi-like torsion:
            # C(formyl)-N-C(alpha)-C.  This is the phi component of the
            # all-trans C5/C7 cap-to-backbone definition.
            formyl_carbon = next(
                int(atom.index)
                for atom in graph.atoms
                if atom.symbol == "C"
                and int(atom.index) not in {
                    int(current["CA"]),
                    int(current["C"]),
                }
                and int(current["N"]) in {
                    int(neighbor) for neighbor in graph.neighbors(atom.index)
                }
            )
            rows.append(
                _target_row(
                    position,
                    "phi_formyl_cap",
                    (formyl_carbon, current["N"], current["CA"], current["C"]),
                    requested["phi"],
                )
            )
        elif index > 0 and "phi" in requested:
            previous = backbone[index - 1]["atoms_zero_based"]
            rows.append(
                _target_row(
                    position,
                    "phi",
                    (previous["C"], current["N"], current["CA"], current["C"]),
                    requested["phi"],
                )
            )
        if index + 1 < len(backbone):
            following = backbone[index + 1]["atoms_zero_based"]
            if "psi" in requested:
                rows.append(
                    _target_row(
                        position,
                        "psi",
                        (current["N"], current["CA"], current["C"], following["N"]),
                        requested["psi"],
                    )
                )
            if "omega" in requested:
                rows.append(
                    _target_row(
                        position,
                        "omega",
                        (current["CA"], current["C"], following["N"], following["CA"]),
                        requested["omega"],
                    )
                )
        elif "psi" in requested:
            if n_terminus != "formyl" or c_terminus not in {"amide", "n_methylamide"}:
                rows.append(
                    _target_row(
                        position,
                        "psi_terminal_carbonyl",
                        (current["N"], current["CA"], current["C"], current["O"]),
                        float(requested["psi"]) - 180.0,
                    )
                )
                continue
            amide_nitrogen = next(
                int(atom.index)
                for atom in graph.atoms
                if atom.symbol == "N"
                and int(atom.index) not in {
                    int(current["N"]),
                }
                and int(current["C"]) in {
                    int(neighbor) for neighbor in graph.neighbors(atom.index)
                }
            )
            rows.append(
                _target_row(
                    position,
                    "psi_c_terminal_cap",
                    (current["N"], current["CA"], current["C"], amide_nitrogen),
                    requested["psi"],
                )
            )
    return tuple(rows)


def _target_row(
    position: int,
    label: str,
    atoms: tuple[int, int, int, int],
    value: float,
) -> dict[str, Any]:
    target = float(value)
    if not math.isfinite(target):
        raise PeptideBuildError(f"non-finite {label} target at residue {position}")
    return {
        "residue_position": int(position),
        "label": label,
        "atoms_zero_based": list(atoms),
        "atoms_one_based": [atom + 1 for atom in atoms],
        "target_degrees": _periodic_degrees(target),
    }


def _set_peptide_dihedral(
    coordinates: np.ndarray,
    adjacency: Mapping[int, set[int]],
    atoms: tuple[int, int, int, int],
    target_degrees: float,
) -> str:
    left, center_left, center_right, right = atoms
    current_raw = math.degrees(float(dihedral(left, center_left, center_right, right, coordinates)))
    target_raw = _periodic_degrees(float(target_degrees) + 180.0)
    delta = math.radians(_periodic_degrees(target_raw - current_raw))
    origin = coordinates[center_left].copy()
    axis = coordinates[center_right] - origin
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-12:
        raise PeptideBuildError("cannot rotate about a zero-length backbone bond")
    axis /= norm
    component = _bond_side_component(adjacency, right, blocked_edge={center_left, center_right})
    status = "SET"
    if center_left in component or left in component:
        # Proline's N-CA bond belongs to the pyrrolidine ring.  The peptide
        # substituent on N is nevertheless exocyclic, so rotate that complete
        # upstream branch about the same axis without moving the ring.
        component = _bond_side_component(adjacency, left, blocked_edge={left, center_left})
        if center_left in component or center_right in component or right in component:
            return "RING_CONSTRAINED"
        delta = -delta
        status = "SET_UPSTREAM_RING_SAFE"
    for atom in sorted(component):
        vector = coordinates[atom] - origin
        coordinates[atom] = origin + (
            vector * math.cos(delta)
            + np.cross(axis, vector) * math.sin(delta)
            + axis * float(np.dot(axis, vector)) * (1.0 - math.cos(delta))
        )
    return status


def _bond_side_component(
    adjacency: Mapping[int, set[int]],
    seed: int,
    *,
    blocked_edge: set[int],
) -> set[int]:
    component = {int(seed)}
    queue = [int(seed)]
    for atom in queue:
        for neighbor in adjacency.get(atom, set()):
            if {atom, neighbor} == blocked_edge:
                continue
            if neighbor not in component:
                component.add(neighbor)
                queue.append(neighbor)
    return component


def _peptide_dihedral_degrees(
    coordinates: np.ndarray,
    atoms: tuple[int, int, int, int],
) -> float:
    raw = math.degrees(float(dihedral(*atoms, coordinates)))
    return _periodic_degrees(raw - 180.0)


def _periodic_degrees(value: float) -> float:
    return float((float(value) + 180.0) % 360.0 - 180.0)


def _lcb26_sources(payload: Any) -> tuple[str, ...]:
    sources: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key in {"identifier", "source_identifier"} and str(item).startswith("LCB26:"):
                    sources.add(str(item))
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(payload)
    return tuple(sorted(sources))


__all__ = [
    "PEPTIDE_BUILD_SCHEMA",
    "PEPTIDE_LIBRARY_SCHEMA",
    "AminoAcidDefinition",
    "PeptideBuild",
    "PeptideBuildError",
    "amino_acid_definitions",
    "build_peptide",
    "load_amino_acid_library",
    "parse_peptide_sequence",
    "query_amino_acid",
]
