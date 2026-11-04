"""Frozen initial-structure preparation protocol for MATRIX workflows."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
import json
from pathlib import Path
import math
from tempfile import TemporaryDirectory
from typing import Any, Sequence

import numpy as np

from matrix_chem import (
    ValenceLevel,
    apply_accuracy_ladder_plan,
    build_accuracy_ladder_plan,
    build_topology_objects,
    read_geometry,
    read_primitive_contract,
    read_xyz,
    write_xyz,
    Primitive,
    backtransform_primitive_targets,
    release_completed_hydrogen_clashes,
)
from matrix_chem.geometry import MolecularGeometry
from matrix_chem.topology.elements import atomic_number
from matrix_switch import (
    SwitchMolecularGraph,
    build_cartesian_seed,
    complete_graph_hydrogens,
    find_substructure_matches,
    graph_from_topology,
    parse_smiles,
)

from .api import analyze_structure
from .lcb26 import load_lcb26_reference, query_lcb26


INITIAL_STRUCTURE_SCHEMA = "matrix.oracle.initial_structure_protocol.v1"
INITIAL_STRUCTURE_PROTOCOL_REVISION = (
    "weighted-b-lcb26-audited-ring-fallback-zaff-exocyclic-r5-2026-08"
)
RING_BOND_MICROCLOSURE_LIMIT_ANGSTROM = 0.002
RING_ANGLE_MICROCLOSURE_LIMIT_DEGREES = 0.2
RING_DIHEDRAL_MICROCLOSURE_LIMIT_DEGREES = 0.2
RING_PRIMITIVE_WEIGHT = 1.0e6
LOCAL_NEIGHBOR_COUNT = 8
LOCAL_GAUSSIAN_BANDWIDTH = 1.0
LOCAL_RELIABILITY_MIN_EFFECTIVE_DONORS = 2.0
LOCAL_RELIABILITY_MAX_DESCRIPTOR_DISTANCE = LOCAL_GAUSSIAN_BANDWIDTH
LOCAL_RELIABILITY_MAX_ATOM_STD = (0.10, 0.10)
LOCAL_RELIABILITY_MAX_BOND_STD_ANGSTROM = 0.025
LOCAL_RELIABILITY_MAX_ANGLE_STD_DEGREES = 2.0
LOCAL_ROBUST_REWEIGHT_ITERATIONS = 3
LOCAL_ROBUST_TUNING = 2.5
SUPPORTED_DECLARED_LEVELS = frozenset(
    {"AUTO", "L0", "L1", "L2", "PL2", "L2_VALENCE_ONLY", "PL2_VALENCE_ONLY"}
)


def weighted_l1_internal_closure(
    coordinates_angstrom: np.ndarray,
    bonds: Sequence[tuple[int, int]],
    angles: Sequence[tuple[int, int, int]],
    bond_observations: Sequence[Sequence[tuple[float, float]]],
    angle_observations: Sequence[Sequence[tuple[float, float]]],
    *,
    protected_ring_indices: Sequence[int] = (),
    max_iterations: int = 50,
    tolerance: float = 1.0e-8,
):
    """Reconstruct L1 Cartesians from weighted fragment bond/angle data.

    Each observation is ``(value, weight)``.  The target for a primitive is
    the weighted mean of its L1 observations; the aggregate observation weight
    controls the Wilson-B projection.  No LCB26 catalog, withheld target
    geometry, or torsion target is consulted.
    """
    if len(bond_observations) != len(bonds) or len(angle_observations) != len(angles):
        raise ValueError("L1 observations must have one entry per bond and angle")
    primitives = [Primitive("bond", tuple(pair)) for pair in bonds]
    primitives.extend(Primitive("angle", tuple(triple)) for triple in angles)
    observations = list(bond_observations) + list(angle_observations)
    targets: dict[int, float] = {}
    weights: dict[int, float] = {}
    for index, values in enumerate(observations):
        if not values:
            continue
        total_weight = sum(float(weight) for _value, weight in values)
        if total_weight <= 0.0:
            raise ValueError("L1 observation weights must be positive")
        targets[index] = sum(float(value) * float(weight) for value, weight in values) / total_weight
        weights[index] = total_weight
    weights.update({int(index): RING_PRIMITIVE_WEIGHT for index in protected_ring_indices})
    result = backtransform_primitive_targets(
        primitives,
        np.asarray(coordinates_angstrom, dtype=float),
        targets,
        primitive_weights=weights,
        deformation_weights={"bond": 1000.0, "angle": 100.0},
        tolerance=tolerance,
        max_iterations=max_iterations,
        maximum_cartesian_step=0.15,
        allow_least_squares_projection=True,
        objective_tolerance=1.0e-9,
    )
    return result


class InitialStructureError(ValueError):
    """Raised when the frozen initial-structure protocol cannot be completed."""


@dataclass(frozen=True)
class InitialStructurePreparation:
    schema: str
    source: str
    source_kind: str
    declared_level: str
    correction_mode: str
    output_xyz: str
    output_xyzin: str
    report: str
    donor_count: int
    corrected_bond_count: int
    corrected_angle_count: int
    closure_converged: bool
    closure_iterations: int
    closure_max_residual: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _InitialStructureInput:
    """Resolved input plus its optional immutable SWITCH constitution."""

    source: str
    kind: str
    geometry: MolecularGeometry
    graph: SwitchMolecularGraph | None

    @property
    def constitutional_bonds(self) -> tuple[tuple[int, int], ...] | None:
        if self.graph is None:
            return None
        return tuple(tuple(sorted(bond.key)) for bond in self.graph.bonds)

    @property
    def constitutional_bond_orders(self) -> dict[tuple[int, int], float] | None:
        if self.graph is None:
            return None
        return {tuple(sorted(bond.key)): float(bond.order) for bond in self.graph.bonds}

    @property
    def constitutional_hydrogens(self) -> tuple[int | None, ...] | None:
        if self.graph is None:
            return None
        return tuple(atom.hydrogen_count for atom in self.graph.atoms)


@dataclass(frozen=True)
class _LCB26LocalTopology:
    """One compiled view of an LCB26 electronic record."""

    atoms: tuple[str, ...]
    coordinates_angstrom: np.ndarray
    atomic_descriptors: tuple[tuple[float, float, int] | None, ...]
    adjacency: dict[int, frozenset[int]]
    orders: dict[tuple[int, int], float]
    ring_edges: frozenset[tuple[int, int]]
    ring_class_by_edge: dict[tuple[int, int], str]
    edge_cycle_sizes: dict[tuple[int, int], int]
    atom_types: dict[int, tuple[int, int, tuple[int, ...]]]
    atom_ring_classes: dict[int, tuple[str, ...]]


@dataclass(frozen=True)
class _LCB26RingSystemTopology:
    """Precompiled exact ring-system graph and its substitution interface."""

    atoms: frozenset[int]
    edges: frozenset[tuple[int, int]]
    element_signature: tuple[str, ...]
    graph: SwitchMolecularGraph
    interface_signatures: dict[int, tuple[tuple[str, str], ...]]


@dataclass(frozen=True)
class _LCB26CompiledRingTopology:
    """Heavy-atom cyclic view compiled once with the LCB26 catalog."""

    atom_indices: tuple[int, ...]
    atoms: tuple[str, ...]
    bonds: frozenset[tuple[int, int]]
    orders: dict[tuple[int, int], float]
    systems: tuple[_LCB26RingSystemTopology, ...]


@dataclass(frozen=True)
class _TargetRingSystemTopology:
    """One target ring system prepared for exact donor comparison."""

    atoms: frozenset[int]
    ordered_atoms: tuple[int, ...]
    edges: frozenset[tuple[int, int]]
    graph: SwitchMolecularGraph
    interface_signatures: dict[int, tuple[tuple[str, str], ...]]


@dataclass(frozen=True)
class _RingTransferCandidate:
    """Fully scored LCB26 ring-system transfer candidate."""

    score: tuple[float, ...]
    row: dict[str, Any]
    record: dict[str, Any]
    donor_atom_indices: tuple[int, ...]
    mapping: dict[int, int]
    order_distance: float
    order_class_mismatches: int
    target_unsaturation_density: float
    donor_unsaturation_density: float
    electronic_state_mismatch: int


@dataclass(frozen=True)
class _LCB26DonorCatalog:
    """Revision-bound local donor indexes shared by molecule preparations."""

    records: tuple[
        tuple[dict[str, Any], dict[str, Any], _LCB26CompiledRingTopology], ...
    ]
    atoms_by_element: dict[str, tuple[tuple[dict[str, Any], _LCB26LocalTopology, int], ...]]
    bonds_by_elements: dict[
        tuple[str, str],
        tuple[tuple[dict[str, Any], _LCB26LocalTopology, tuple[int, int]], ...],
    ]
    angles_by_elements: dict[
        tuple[str, str, str],
        tuple[
            tuple[
                dict[str, Any],
                _LCB26LocalTopology,
                tuple[int, int, int],
            ],
            ...,
        ],
    ]


def prepare_initial_structure(
    source: Path | str,
    output: Path | str,
    *,
    lcb26_root: Path | str,
    declared_level: str = "AUTO",
    source_kind: str = "auto",
    constitutional_smiles: str | None = None,
    preserved_dihedrals: Sequence[tuple[int, int, int, int]] = (),
    excluded_lcb26_identifiers: Sequence[str] = (),
    max_iterations: int = 30,
    closure_tolerance: float = 1.0e-6,
) -> InitialStructurePreparation:
    """Prepare the structure consumed by all subsequent MATRIX tools.

    SMILES are converted through SWITCH.  XYZ and other supported structure
    files are read as supplied. ``constitutional_smiles`` may accompany an
    externally pre-shaped Cartesian seed and remains the immutable source of
    atoms, bonds, bond orders, and hydrogen counts. Complete LCB26 ring systems and local
    bond/angle donors are reconciled with the established weighted Wilson-B
    pseudoinverse. Declared L1 input follows the
    ``declared_level="L1"`` marks the result as an ``INITIAL_L1`` seed for a
    subsequent QM optimization; it does not apply PL1. The separate
    ``refine-l1`` workflow applies L1→PL1 to an already optimized L1 geometry.
    Declared L2/PL2 valence-only input receives only the CV layer.
    """

    level = _normalize_level(declared_level)
    resolved = _resolve_initial_input(
        source,
        source_kind=source_kind,
        constitutional_smiles=constitutional_smiles,
    )
    source_text = resolved.source
    kind = resolved.kind
    geometry = resolved.geometry
    numbers = tuple(_atomic_number(symbol) for symbol in geometry.atoms)
    lcb26_path = Path(lcb26_root).expanduser().resolve()
    if not (lcb26_path / "enriched" / "index.json").is_file():
        raise InitialStructureError(f"LCB26 query index is missing: {lcb26_path}")

    corrected, donor_audit = _improve_from_lcb26(
        geometry,
        numbers,
        lcb26_path,
        max_iterations=max_iterations,
        tolerance=closure_tolerance,
        constitutional_graph=resolved.graph,
        preserved_dihedrals=preserved_dihedrals,
        excluded_lcb26_identifiers=excluded_lcb26_identifiers,
    )
    output_xyz = Path(output).expanduser().resolve()
    output_xyz.parent.mkdir(parents=True, exist_ok=True)
    write_xyz(
        output_xyz,
        corrected.atoms,
        corrected.coordinates_angstrom,
        comment="MATRIX initial structure",
    )

    correction_mode = "LCB26_GEOMETRIC_TRANSFER"
    output_xyzin = output_xyz.with_suffix(".xyzin")
    report_path = output_xyz.with_suffix(".initial_structure.json")
    if level == "L1":
        correction_mode = "INITIAL_L1_SEED"
        analyze_structure(output_xyz, output_xyzin, source_kind="xyz")
    elif level in {"L2_VALENCE_ONLY", "PL2_VALENCE_ONLY"}:
        correction_mode = "CV_ONLY"
        with TemporaryDirectory(prefix="oracle-initial-cv-") as scratch:
            source_xyzin = Path(scratch) / "input.xyzin"
            analyze_structure(output_xyz, source_xyzin, source_kind="xyz")
            cv_result = _apply_cv_only(source_xyzin, output_xyzin)
        write_xyz(
            output_xyz, corrected.atoms, cv_result, comment="MATRIX initial structure CV-only"
        )
    else:
        analyze_structure(output_xyz, output_xyzin, source_kind="xyz")

    payload = {
        "schema": INITIAL_STRUCTURE_SCHEMA,
        "source": source_text,
        "source_kind": kind,
        "constitutional_smiles": (
            resolved.graph.source_smiles if resolved.graph is not None else None
        ),
        "declared_level": level,
        "correction_mode": correction_mode,
        "output_xyz": str(output_xyz),
        "output_xyzin": str(output_xyzin),
        "donor_audit": donor_audit,
        "protocol": {
            "smiles_to_xyz": "SWITCH_BUILD_CARTESIAN_SEED",
            "ordered_stages": [
                "SWITCH_CONSTITUTION_OR_SUPPLIED_XYZ",
                "ORACLE_LCB26_COMPLETE_RING_TRANSFER",
                "ORACLE_LCB26_LOCAL_BOND_ANGLE_SELECTION",
                "MATRIX_CHEM_WEIGHTED_WILSON_B_CLOSURE",
                "ORACLE_CONSTITUTION_AND_RING_GATES",
                "SMITH_SONIC_CONSTRUCTION",
                "ARCHITECT_ZAFF_FAST_SOFT_EXOCYCLIC_RELAXATION",
            ],
            "geometry_perception": "ORACLE",
            "local_donor_library": "LCB26",
            "ring_geometry_authority": "ORACLE_CYCLE_PERCEPTION_WITH_LCB26_COMPLETE_SYSTEM_DONORS",
            "torsion_classification_authority": "SMITH_SONIC_FAMILY",
            "cartesian_internal_closure": "L1_PL1_WEIGHTED_WILSON_B_PSEUDOINVERSE",
            "ring_microclosure_limits": {
                "bond_angstrom": RING_BOND_MICROCLOSURE_LIMIT_ANGSTROM,
                "angle_degrees": RING_ANGLE_MICROCLOSURE_LIMIT_DEGREES,
                "endocyclic_dihedral_degrees": RING_DIHEDRAL_MICROCLOSURE_LIMIT_DEGREES,
            },
            "zaff_fast_active_space": "SOFT_EXOCYCLIC_TORSIONS_ONLY",
            "l1_exception": "L1_TO_PL1",
            "l2_pl2_valence_only_exception": "CV_ONLY",
        },
    }
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = InitialStructurePreparation(
        schema=INITIAL_STRUCTURE_SCHEMA,
        source=source_text,
        source_kind=kind,
        declared_level=level,
        correction_mode=correction_mode,
        output_xyz=str(output_xyz),
        output_xyzin=str(output_xyzin),
        report=str(report_path),
        donor_count=int(donor_audit["donor_count"]),
        corrected_bond_count=int(donor_audit["corrected_bond_count"]),
        corrected_angle_count=int(donor_audit["corrected_angle_count"]),
        closure_converged=bool(donor_audit["closure_converged"]),
        closure_iterations=int(donor_audit["closure_iterations"]),
        closure_max_residual=float(donor_audit["closure_max_residual"]),
    )
    payload["result"] = result.to_dict()
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _normalize_level(value: str) -> str:
    level = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    if level not in SUPPORTED_DECLARED_LEVELS:
        raise InitialStructureError(f"unsupported declared structure level: {value!r}")
    if level == "PL2":
        return "PL2_VALENCE_ONLY"
    return level


def _read_initial_geometry(source_text: str, path: Path, kind: str) -> MolecularGeometry:
    if kind == "smiles":
        graph = parse_smiles(source_text)
        return build_cartesian_seed(graph, title=source_text, complete_hydrogens=True)
    if kind not in {"xyz", "geometry", "enriched_xyz"} or not path.is_file():
        raise InitialStructureError(f"unsupported or missing initial structure: {source_text}")
    return read_geometry(path)


def _resolve_initial_input(
    source: Path | str,
    *,
    source_kind: str,
    constitutional_smiles: str | None,
) -> _InitialStructureInput:
    source_text = str(source)
    source_path = Path(source_text).expanduser()
    kind = source_kind.strip().casefold()
    if kind == "auto":
        kind = "xyz" if source_path.is_file() else "smiles"
    geometry = _read_initial_geometry(source_text, source_path, kind)
    graph = None
    if kind == "smiles" or constitutional_smiles is not None:
        graph = parse_smiles(source_text if kind == "smiles" else str(constitutional_smiles))
        graph_atoms = tuple(atom.symbol for atom in graph.atoms)
        if graph_atoms != tuple(geometry.atoms[: len(graph_atoms)]):
            raise InitialStructureError(
                "constitutional SMILES atoms do not match the Cartesian seed ordering"
            )
        if int(geometry.charge or 0) != int(graph.total_formal_charge):
            geometry = MolecularGeometry(
                atoms=geometry.atoms,
                coordinates_angstrom=geometry.coordinates_angstrom,
                comment=geometry.comment,
                source_format=geometry.source_format,
                source_path=geometry.source_path,
                charge=graph.total_formal_charge,
                multiplicity=geometry.multiplicity,
                fixed_parameters=geometry.fixed_parameters,
                metadata={
                    **dict(geometry.metadata),
                    "constitutional_charge_source": "SWITCH_GRAPH",
                },
            )
    return _InitialStructureInput(
        source=source_text,
        kind=kind,
        geometry=geometry,
        graph=graph,
    )


def _atomic_number(symbol: str) -> int:
    number = atomic_number(symbol)
    if number is None or number <= 0:
        raise InitialStructureError(f"unknown atom symbol: {symbol!r}")
    return int(number)


def _improve_from_lcb26(
    geometry: MolecularGeometry,
    numbers: tuple[int, ...],
    lcb26_root: Path,
    *,
    max_iterations: int,
    tolerance: float,
    constitutional_graph: SwitchMolecularGraph | None = None,
    preserved_dihedrals: Sequence[tuple[int, int, int, int]] = (),
    excluded_lcb26_identifiers: Sequence[str] = (),
) -> tuple[MolecularGeometry, dict[str, Any]]:
    constitutional_bonds = (
        None
        if constitutional_graph is None
        else tuple(tuple(sorted(bond.key)) for bond in constitutional_graph.bonds)
    )
    constitutional_bond_orders = (
        None
        if constitutional_graph is None
        else {tuple(sorted(bond.key)): float(bond.order) for bond in constitutional_graph.bonds}
    )
    constitutional_hydrogens = (
        None
        if constitutional_graph is None
        else tuple(atom.hydrogen_count for atom in constitutional_graph.atoms)
    )
    # Global molecular shortlist precedes local bond/angle transfer.  This
    # prevents a chemically unrelated record from contributing merely because
    # it shares one element pair.
    catalog = _lcb26_donor_catalog(lcb26_root)
    catalog_total_count = len(catalog.records)
    excluded = {str(value) for value in excluded_lcb26_identifiers}
    if excluded:
        catalog = _exclude_catalog_records(catalog, excluded)
    # Complete ring-system donors are selected from the complete library.  A
    # whole-molecule composition shortlist is appropriate for local primitive
    # interpolation, but it can exclude a smaller exact ring scaffold (for
    # example benzene inside ethylbenzene or indole inside tryptophan).
    records = catalog.records
    ring_geometry, ring_atoms, ring_audit = _transfer_lcb26_ring_systems(
        geometry, constitutional_bonds, constitutional_bond_orders, records
    )
    try:
        _c, _g, _r, target_synthons, _a = build_topology_objects(
            ring_geometry.coordinates_angstrom, numbers
        )
    except Exception:
        target_synthons = None
    catalog = _shortlist_catalog(catalog, numbers, target_synthons=target_synthons, limit=32)
    records = catalog.records
    if constitutional_bonds is not None and constitutional_hydrogens is not None:
        heavy_count = len(constitutional_hydrogens)
        heavy_geometry = MolecularGeometry(
            atoms=ring_geometry.atoms[:heavy_count],
            coordinates_angstrom=ring_geometry.coordinates_angstrom[:heavy_count],
            comment=ring_geometry.comment,
            charge=ring_geometry.charge,
            multiplicity=ring_geometry.multiplicity,
            source_format=ring_geometry.source_format,
            metadata=ring_geometry.metadata,
        )
        ring_geometry = complete_graph_hydrogens(
            constitutional_graph,
            heavy_geometry,
        ).geometry
    topology_coordinates = ring_geometry.coordinates_angstrom
    topology_numbers = numbers
    if constitutional_bonds is not None and constitutional_hydrogens is not None:
        # Donor matching at this stage needs only the immutable heavy-atom
        # constitution. Hydrogen directions are regenerated after the heavy
        # weighted-B projection and must not contaminate topology perception.
        heavy_count = len(constitutional_hydrogens)
        topology_coordinates = topology_coordinates[:heavy_count]
        topology_numbers = numbers[:heavy_count]
    _continuous, graph, _rings, synthons, _aromaticity = build_topology_objects(
        topology_coordinates, topology_numbers
    )
    if constitutional_bonds is not None and constitutional_hydrogens is not None:
        # Keep the SWITCH constitution authoritative for peptide inputs.  A
        # geometry-only re-perception of the completed structure can create a
        # spurious X--H bond when an explicitly placed hydrogen lies close to
        # a neighbouring heavy atom (notably the indole N/H region of a
        # tryptophan capped analogue).  All descriptors consumed below refer
        # to constitutional heavy-atom bonds, so the heavy-only synthon graph
        # built above is the correct graph here.
        pass
    inferred_bonds = tuple(tuple(sorted((int(left), int(right)))) for left, right in graph.bonds)
    if constitutional_bonds is None:
        bonds = inferred_bonds
    else:
        heavy = {tuple(sorted(pair)) for pair in constitutional_bonds}
        # The weighted-B assembly is performed on the heavy-atom constitution.
        # Hydrogens are regenerated only after the heavy skeleton converges.
        bonds = tuple(sorted(heavy))
    adjacency = {index: [] for index in range(len(numbers))}
    for left, right in bonds:
        adjacency[left].append(right)
        adjacency[right].append(left)
    angles = tuple(
        (left, center, right)
        for center, neighbours in adjacency.items()
        for left in neighbours
        for right in neighbours
        if left < right
    )
    ring_edges = (
        {tuple(sorted(map(int, pair))) for pair in constitutional_bonds}
        - _bridge_edges(
            len(constitutional_hydrogens),
            {tuple(sorted(map(int, pair))) for pair in constitutional_bonds},
        )
        if constitutional_bonds is not None and constitutional_hydrogens is not None
        else set()
    )
    all_ring_bond_indices = {
        index for index, pair in enumerate(bonds) if tuple(sorted(pair)) in ring_edges
    }
    all_ring_angle_indices = {
        len(bonds) + index
        for index, (left, center, right) in enumerate(angles)
        if tuple(sorted((left, center))) in ring_edges
        and tuple(sorted((center, right))) in ring_edges
    }
    dihedrals = _all_dihedrals(bonds)
    endocyclic_dihedrals = set(_endocyclic_dihedrals(bonds, ring_edges))
    ring_dihedral_offset = len(bonds) + len(angles)
    all_ring_dihedral_indices = {
        ring_dihedral_offset + index
        for index, item in enumerate(dihedrals)
        if item in endocyclic_dihedrals
    }
    # Only a complete, electronically compatible LCB26 ring-system donor earns
    # the immutable micro-closure contract.  A fallback ring block must be
    # allowed to move on the weighted-B manifold while its local LCB26 bond
    # and angle targets are installed.
    strict_ring_atoms = set(int(index) for index in ring_atoms)
    ring_bond_indices = {
        index for index in all_ring_bond_indices if set(bonds[index]) <= strict_ring_atoms
    }
    ring_angle_indices = {
        len(bonds) + index
        for index, angle in enumerate(angles)
        if len(bonds) + index in all_ring_angle_indices and set(angle) <= strict_ring_atoms
    }
    ring_dihedral_indices = {
        ring_dihedral_offset + index
        for index, item in enumerate(dihedrals)
        if item in endocyclic_dihedrals and set(item) <= strict_ring_atoms
    }
    protected_ring_indices = ring_bond_indices | ring_angle_indices | ring_dihedral_indices
    target_values = _internal_values(ring_geometry.coordinates_angstrom, bonds, angles, dihedrals)
    donor_values = target_values.copy()
    selected_target_indices = set(protected_ring_indices)
    dihedral_offset = len(bonds) + len(angles)
    dihedral_lookup = {item: index for index, item in enumerate(dihedrals)}
    preserved_dihedral_indices = set()
    for raw in preserved_dihedrals:
        item = tuple(int(atom) for atom in raw)
        canonical = min(item, tuple(reversed(item)))
        if canonical not in dihedral_lookup:
            raise InitialStructureError(
                f"preserved dihedral is not a constitutional proper torsion: {item}"
            )
        internal_index = dihedral_offset + dihedral_lookup[canonical]
        selected_target_indices.add(internal_index)
        preserved_dihedral_indices.add(internal_index)
    bond_support = 0
    angle_support = 0
    donor_trace: list[dict[str, Any]] = []
    target_edge_cycle_sizes = _edge_cycle_sizes(
        {index: set(neighbours) for index, neighbours in adjacency.items()},
        ring_edges,
    )
    target_atom_types = _local_atom_types(
        tuple(_symbol(number) for number in numbers),
        {index: set(neighbours) for index, neighbours in adjacency.items()},
        target_edge_cycle_sizes,
    )
    target_ring_classes = _ring_classes_for_edges(
        ring_edges,
        {
            edge: float(
                (constitutional_bond_orders or {}).get(edge, 1.0)
                if constitutional_bond_orders is not None
                else synthons.bond_order(*edge)
            )
            for edge in ring_edges
        },
    )
    target_atom_ring_classes = {
        atom: tuple(
            sorted(
                {
                    target_ring_classes[edge]
                    for edge in ring_edges
                    if atom in edge and edge in target_ring_classes
                }
            )
        )
        for atom in target_atom_types
    }
    target_atomic_descriptors, atom_trace = _nearest_atomic_types(
        numbers,
        synthons,
        sorted({atom for bond in bonds for atom in bond}),
        catalog=catalog,
        target_atom_types=target_atom_types,
        target_atom_ring_classes=target_atom_ring_classes,
    )
    for index, (left, right) in enumerate(bonds):
        if index in ring_bond_indices:
            continue
        candidates = _bond_donors(
            numbers,
            target_atomic_descriptors,
            left,
            right,
            catalog=catalog,
            trace=donor_trace,
            target_index=index,
            target_order=(
                (constitutional_bond_orders or {}).get(tuple(sorted((left, right))))
                if constitutional_bond_orders is not None
                else float(synthons.bond_order(left, right))
            ),
            target_distance=float(np.linalg.norm(ring_geometry.coordinates_angstrom[left] - ring_geometry.coordinates_angstrom[right])),
            require_ring=index in all_ring_bond_indices,
            target_atom_types=target_atom_types,
            target_cycle_size=target_edge_cycle_sizes.get(tuple(sorted((left, right)))),
            target_ring_class=target_ring_classes.get(tuple(sorted((left, right)))),
        )
        if candidates:
            donor_values[index] = _weighted_mean(candidates)
            selected_target_indices.add(index)
            bond_support += 1
    angle_offset = len(bonds)
    for local, (left, center, right) in enumerate(angles):
        if angle_offset + local in ring_angle_indices:
            continue
        candidates = _angle_donors(
            numbers,
            target_atomic_descriptors,
            left,
            center,
            right,
            catalog=catalog,
            trace=donor_trace,
            target_index=angle_offset + local,
            target_orders=(
                (
                    (constitutional_bond_orders or {}).get(tuple(sorted((left, center))))
                    if constitutional_bond_orders is not None
                    else float(synthons.bond_order(left, center))
                ),
                (
                    (constitutional_bond_orders or {}).get(tuple(sorted((center, right))))
                    if constitutional_bond_orders is not None
                    else float(synthons.bond_order(center, right))
                ),
            ),
            require_ring=angle_offset + local in all_ring_angle_indices,
            target_atom_types=target_atom_types,
            target_cycle_sizes=(
                target_edge_cycle_sizes.get(tuple(sorted((left, center)))),
                target_edge_cycle_sizes.get(tuple(sorted((center, right)))),
            ),
            target_ring_classes=(
                target_ring_classes.get(tuple(sorted((left, center)))),
                target_ring_classes.get(tuple(sorted((center, right)))),
            ),
        )
        if candidates:
            donor_values[angle_offset + local] = _weighted_mean(candidates)
            selected_target_indices.add(angle_offset + local)
            angle_support += 1
    preconditioned, preconditioned_angles = _precondition_large_angle_targets(
        ring_geometry.coordinates_angstrom,
        bonds,
        angles,
        donor_values,
        selected_target_indices,
        ring_edges,
    )
    corrected, converged, iterations, residual = _close_internal_coordinates(
        preconditioned,
        bonds,
        angles,
        dihedrals,
        donor_values,
        max_iterations,
        tolerance,
        protected_ring_indices=protected_ring_indices,
        target_indices=selected_target_indices,
    )
    corrected, restored_dihedrals = _restore_preserved_dihedrals(
        corrected,
        bonds,
        preserved_dihedrals,
        ring_geometry.coordinates_angstrom,
    )
    topology_transfer_scale = 1.0
    if constitutional_bonds is not None and constitutional_hydrogens is not None:
        heavy_count = len(constitutional_hydrogens)
        candidate_heavy = np.asarray(corrected[:heavy_count], dtype=float)
        reference_heavy = np.asarray(ring_geometry.coordinates_angstrom[:heavy_count], dtype=float)
        corrected = _complete_constitution_coordinates(
            candidate_heavy,
            ring_geometry,
            constitutional_graph,
        )
    topology_gate = _constitutional_topology_gate(
        corrected,
        numbers,
        constitutional_bonds,
        allow_geometric_spurious=constitutional_graph is not None,
        constitutional_atom_count=(
            len(constitutional_graph.atoms) if constitutional_graph is not None else None
        ),
    )
    if not topology_gate["valid"]:
        if constitutional_bonds is None or constitutional_hydrogens is None:
            raise InitialStructureError(
                "weighted-B reconstruction changed the perceived constitution: "
                f"missing={topology_gate['missing_bonds']} "
                f"spurious={topology_gate['spurious_bonds']}"
            )
        reference_completed = _complete_constitution_coordinates(
            reference_heavy,
            ring_geometry,
            constitutional_graph,
        )
        reference_gate = _constitutional_topology_gate(
            reference_completed,
            numbers,
            constitutional_bonds,
            allow_geometric_spurious=True,
            constitutional_atom_count=len(constitutional_graph.atoms),
        )
        if not reference_gate["valid"]:
            raise InitialStructureError(
                "the pre-shaped SWITCH constitution is invalid before LCB26 transfer: "
                f"missing={reference_gate['missing_bonds']} "
                f"spurious={reference_gate['spurious_bonds']}"
            )
        lower, upper = 0.0, 1.0
        accepted = reference_completed
        accepted_gate = reference_gate
        for _ in range(24):
            scale = 0.5 * (lower + upper)
            trial_heavy = reference_heavy + scale * (candidate_heavy - reference_heavy)
            trial_heavy, _ = _restore_preserved_dihedrals(
                trial_heavy,
                bonds,
                preserved_dihedrals,
                reference_heavy,
            )
            trial = _complete_constitution_coordinates(
                trial_heavy,
                ring_geometry,
                constitutional_graph,
            )
            trial_gate = _constitutional_topology_gate(
                trial,
                numbers,
                constitutional_bonds,
                allow_geometric_spurious=True,
                constitutional_atom_count=len(constitutional_graph.atoms),
            )
            if trial_gate["valid"]:
                lower = scale
                accepted = trial
                accepted_gate = trial_gate
            else:
                upper = scale
        corrected = accepted
        topology_gate = accepted_gate
        topology_transfer_scale = lower
    if constitutional_bonds is not None and constitutional_hydrogens is not None:
        # The heavy-atom projection can bring a newly regenerated X--H bond
        # close to an unrelated heavy atom.  Reorient only those generated H
        # atoms, preserving the heavy skeleton and atom order, before the
        # final LINK/ORACLE validation.  This is especially important for
        # fused heteroaromatics and other compact capped residues.
        heavy_count = len(constitutional_hydrogens)
        completed = complete_graph_hydrogens(
            constitutional_graph,
            MolecularGeometry(
                atoms=tuple(ring_geometry.atoms[:heavy_count]),
                coordinates_angstrom=np.asarray(corrected[:heavy_count], dtype=float),
                comment=ring_geometry.comment,
                source_format=ring_geometry.source_format,
                charge=ring_geometry.charge,
                multiplicity=ring_geometry.multiplicity,
                metadata=ring_geometry.metadata,
            ),
        )
        corrected = release_completed_hydrogen_clashes(
            completed,
            minimum_separation_angstrom=1.35,
        ).geometry.coordinates_angstrom
    final_values = _internal_values(corrected, bonds, angles, dihedrals)
    internal_residual = donor_values - final_values
    _attach_fallback_primitive_donors(
        ring_audit,
        donor_trace,
        bonds,
        angles,
    )
    ring_microclosure = _ring_microclosure_audit(
        target_values,
        final_values,
        ring_bond_indices,
        ring_angle_indices,
        ring_dihedral_indices,
    )
    if not ring_microclosure["valid"]:
        raise InitialStructureError(
            "LCB26 ring micro-closure exceeded its immutable limits: "
            f"bond={ring_microclosure['maximum_bond_change_angstrom']:.6g} A, "
            f"angle={ring_microclosure['maximum_angle_change_degrees']:.6g} deg, "
            f"dihedral={ring_microclosure['maximum_dihedral_change_degrees']:.6g} deg"
        )
    fallback_ring_atoms = {
        int(atom)
        for system in ring_audit.get("systems", ())
        if system.get("fallback_used", False)
        for atom in system.get("atoms", ())
    }
    fallback_ring_audit = _ring_fallback_audit(
        target_values,
        final_values,
        bonds,
        angles,
        dihedrals,
        all_ring_bond_indices,
        all_ring_angle_indices,
        all_ring_dihedral_indices,
        fallback_ring_atoms,
        donor_trace,
        topology_gate,
    )
    transfer_reliability = _summarize_transfer_reliability(atom_trace, donor_trace)
    return (
        MolecularGeometry(
            atoms=geometry.atoms,
            coordinates_angstrom=corrected,
            comment=geometry.comment,
            charge=geometry.charge,
            multiplicity=geometry.multiplicity,
            source_format=geometry.source_format,
            metadata={**geometry.metadata, "initial_structure_protocol": INITIAL_STRUCTURE_SCHEMA},
        ),
        {
            "donor_count": catalog_total_count,
            "shortlisted_donor_count": len(records),
            "corrected_bond_count": bond_support,
            "corrected_angle_count": angle_support,
            "preconditioned_large_angle_count": preconditioned_angles,
            "preserved_dihedral_count": len(preserved_dihedral_indices),
            "restored_preserved_dihedral_count": restored_dihedrals,
            "topology_safe_transfer_scale": topology_transfer_scale,
            "closure_converged": converged,
            "closure_iterations": iterations,
            "closure_max_residual": residual,
            "closure_target_projection": "LEAST_SQUARES_INTERNAL_MANIFOLD",
            "bond_count": len(bonds),
            "angle_count": len(angles),
            "donor_trace": donor_trace,
            "atom_type_trace": atom_trace,
            "transfer_reliability": transfer_reliability,
            "target_synthon_observable_sources": {
                "atom_selector": "ORACLE_SYNTHON_ESTIMATE_THEN_LCB26_NEAREST_CM5_ZEFF",
                "donor_atomic_observables": "LCB26_L0_CM5_ZEFF",
                "bond_order": (
                    "SMILES_FORMAL_EXPLICIT_FALLBACK"
                    if constitutional_bond_orders is not None
                    else "ORACLE_ESTIMATED_MAYER_EXPLICIT_FALLBACK"
                ),
                "donor_bond_orders": "LCB26_L0_MAYER",
            },
            "population_iteration_status": "PROVISIONAL_BOOTSTRAP_REQUIRES_FINAL_L0_OR_FRAGMENT_POPULATION",
            "ring_transfer": ring_audit,
            "ring_microclosure": ring_microclosure,
            "ring_fallback_reconstruction": fallback_ring_audit,
            "ring_atom_count": len(ring_atoms),
            "constitutional_topology_gate": topology_gate,
            "stage_gates": {
                "constitution": "PASS" if constitutional_graph is not None else "PERCEIVED_XYZ",
                "ring_transfer": str(ring_audit.get("status", "PASS")),
                "local_donor_selection": "PASS" if bond_support + angle_support else "NO_MATCH",
                "weighted_b_closure": "PASS" if converged else "MAX_ITERATIONS",
                "constitutional_topology": str(topology_gate["status"]),
                "ring_microclosure": str(ring_microclosure["status"]),
            },
            "protocol_revision": INITIAL_STRUCTURE_PROTOCOL_REVISION,
            "closure_rms_residual": float(np.sqrt(np.mean(internal_residual**2)))
            if internal_residual.size
            else 0.0,
            "bond_residual_max": float(np.max(np.abs(internal_residual[: len(bonds)])))
            if bonds
            else 0.0,
            "angle_residual_max": float(
                np.max(np.abs(internal_residual[len(bonds) : len(bonds) + len(angles)]))
            )
            if angles
            else 0.0,
        },
    )


def _descriptor(synthons, index: int, numbers: tuple[int, ...]) -> tuple[float, float]:
    z = int(numbers[index])
    zeff = float(synthons.Zeff(index))
    return float(synthons.charge(index)), (zeff - (z - 0.5)) / 5.5


def _lcb26_atomic_descriptor(descriptor, symbol):
    """Return one complete L0 CM5/Zeff descriptor or reject the row."""

    if "cm5_charge_e" not in descriptor or "zeff_normalized" not in descriptor:
        return None
    values = np.asarray(
        [descriptor["cm5_charge_e"], descriptor["zeff_normalized"]],
        dtype=float,
    )
    number = atomic_number(symbol)
    if number is None or not np.all(np.isfinite(values)):
        return None
    return float(values[0]), float(values[1]), int(number)


def _select_local_neighbors(candidates, *, value_unit):
    """Apply the frozen ARCHITECT nearest-row Gaussian interpolation rule."""

    if not candidates:
        return []
    # Symmetry-equivalent occurrences in one molecule represent one local
    # type and must not acquire extra statistical weight merely by multiplicity.
    unique = {}
    for item in candidates:
        value = np.asarray(item["value"], dtype=float).reshape(-1)
        key = (
            str(item["identifier"]),
            round(float(item["distance"]), 12),
            tuple(round(float(component), 12) for component in value),
        )
        previous = unique.get(key)
        if previous is None or tuple(item.get("source_atoms", ())) < tuple(
            previous.get("source_atoms", ())
        ):
            unique[key] = item
    ranked = sorted(
        unique.values(),
        key=lambda item: (
            float(item["distance"]),
            str(item["identifier"]),
            tuple(item.get("source_atoms", ())),
        ),
    )[:LOCAL_NEIGHBOR_COUNT]
    distances = np.asarray([item["distance"] for item in ranked], dtype=float)
    weights = np.exp(-0.5 * (distances / float(LOCAL_GAUSSIAN_BANDWIDTH)) ** 2)
    if not np.all(np.isfinite(weights)) or float(np.sum(weights)) <= 0.0:
        return []
    weights /= float(np.sum(weights))
    values = np.asarray([item["value"] for item in ranked], dtype=float)
    weights, robust_audit = _robust_kernel_weights(
        values,
        weights,
        max_iterations=LOCAL_ROBUST_REWEIGHT_ITERATIONS,
        tuning=LOCAL_ROBUST_TUNING,
    )
    selected = []
    for item, weight in zip(ranked, weights, strict=True):
        selected.append(
            {
                **item,
                "weight": float(weight),
                "value_unit": value_unit,
                "robust_reweight": robust_audit,
            }
        )
    return selected


def _robust_kernel_weights(
    values,
    weights,
    *,
    max_iterations: int,
    tuning: float,
):
    """Huber-reweight Gaussian neighbours without changing the kernel contract.

    The Gaussian kernel remains the primary selector.  This second, bounded
    stage only suppresses a donor that is inconsistent with the selected local
    population; it never introduces a donor outside the existing compatibility
    tier or descriptor bandwidth.  For a single donor, identical donors, or a
    zero-dispersion set the original weights are returned exactly.
    """

    observations = np.asarray(values, dtype=float)
    current = np.asarray(weights, dtype=float).reshape(-1)
    if observations.ndim == 1:
        observations = observations[:, None]
    if observations.shape[0] != current.size or current.size == 0:
        return current, {"applied": False, "iterations": 0, "outlier_count": 0}
    if current.size < 3 or not np.all(np.isfinite(observations)):
        return current, {"applied": False, "iterations": 0, "outlier_count": 0}
    if not np.isfinite(float(tuning)) or float(tuning) <= 0.0:
        raise ValueError("robust Gaussian tuning must be positive")
    current = current / max(float(np.sum(current)), 1.0e-15)
    applied = False
    iterations = 0
    outliers = np.zeros(current.size, dtype=bool)
    for iteration in range(max(0, int(max_iterations))):
        center = np.median(observations, axis=0)
        residual = np.linalg.norm(observations - center, axis=1)
        median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median)))
        # A zero MAD is common for two nearly identical donors plus one
        # outlier; use the median residual as a conservative floor instead of
        # incorrectly flagging the two coherent donors on the next pass.
        scale = max(1.4826 * mad, median, 1.0e-12)
        standardized = residual / scale
        outliers = standardized > float(tuning)
        if not np.any(outliers):
            break
        factors = np.ones_like(current)
        factors[outliers] = float(tuning) / np.maximum(standardized[outliers], 1.0e-12)
        updated = current * factors
        updated /= max(float(np.sum(updated)), 1.0e-15)
        if np.allclose(updated, current, rtol=0.0, atol=1.0e-12):
            break
        current = updated
        applied = True
        iterations = iteration + 1
    return current, {
        "applied": bool(applied),
        "iterations": int(iterations),
        "outlier_count": int(np.count_nonzero(outliers)),
    }


def _local_transfer_reliability(
    role,
    selected,
    weighted_std,
    compatibility_tier,
):
    """Classify one transfer without hiding sparse or distant support."""

    if not selected:
        return {
            "class": "EXTRAPOLATIVE",
            "acceptance_state": "REQUIRES_QM_VALIDATION",
            "reasons": ["NO_LCB26_DONOR"],
            "effective_donor_count": 0.0,
            "nearest_descriptor_distance": None,
        }
    weights = np.asarray([item["weight"] for item in selected], dtype=float)
    effective = float(1.0 / np.sum(weights**2))
    nearest = float(min(float(item["distance"]) for item in selected))
    reasons = []
    exact_tiers = {
        "EXACT_LOCAL_TOPOLOGY",
        "EXACT_LOCAL_TOPOLOGY_AND_RING_CLASS",
    }
    if str(compatibility_tier) not in exact_tiers:
        reasons.append(f"RELAXED_COMPATIBILITY_TIER:{compatibility_tier}")
    if effective < LOCAL_RELIABILITY_MIN_EFFECTIVE_DONORS:
        reasons.append("INSUFFICIENT_EFFECTIVE_DONORS")
    if nearest > LOCAL_RELIABILITY_MAX_DESCRIPTOR_DISTANCE:
        reasons.append("OUTSIDE_PRIMARY_DESCRIPTOR_BANDWIDTH")

    spread = np.asarray(weighted_std, dtype=float).reshape(-1)
    if role == "atom":
        limits = np.asarray(LOCAL_RELIABILITY_MAX_ATOM_STD, dtype=float)
        if spread.shape != limits.shape or np.any(spread > limits):
            reasons.append("ATOM_DESCRIPTOR_DISPERSION_EXCEEDS_LIMIT")
    elif role == "bond":
        if spread.size != 1 or float(spread[0]) > LOCAL_RELIABILITY_MAX_BOND_STD_ANGSTROM:
            reasons.append("BOND_LENGTH_DISPERSION_EXCEEDS_LIMIT")
    elif role == "angle":
        limit = math.radians(LOCAL_RELIABILITY_MAX_ANGLE_STD_DEGREES)
        if spread.size != 1 or float(spread[0]) > limit:
            reasons.append("ANGLE_DISPERSION_EXCEEDS_LIMIT")

    extrapolative = str(compatibility_tier) == "ELEMENT_ONLY"
    reliability_class = (
        "EXTRAPOLATIVE" if extrapolative else "PROVISIONAL" if reasons else "RELIABLE"
    )
    acceptance_state = {
        "RELIABLE": "ACCEPTED",
        "PROVISIONAL": "REQUIRES_REVIEW",
        "EXTRAPOLATIVE": "REQUIRES_QM_VALIDATION",
    }[reliability_class]
    return {
        "class": reliability_class,
        "acceptance_state": acceptance_state,
        "reasons": reasons,
        "compatibility_tier": str(compatibility_tier),
        "effective_donor_count": effective,
        "nearest_descriptor_distance": nearest,
        "weighted_std": spread.tolist(),
    }


def _summarize_transfer_reliability(atom_trace, donor_trace):
    entries = [
        item["reliability"]
        for item in (*atom_trace, *donor_trace)
        if isinstance(item.get("reliability"), dict)
    ]
    counts = {
        name: sum(item.get("class") == name for item in entries)
        for name in ("RELIABLE", "PROVISIONAL", "EXTRAPOLATIVE")
    }
    if counts["EXTRAPOLATIVE"]:
        status = "EXTRAPOLATIVE_REQUIRES_QM_VALIDATION"
    elif counts["PROVISIONAL"]:
        status = "PROVISIONAL_REQUIRES_REVIEW"
    else:
        status = "ACCEPTED"
    return {
        "status": status,
        "transfer_count": len(entries),
        "counts": counts,
        "requires_explicit_acceptance": status != "ACCEPTED",
        "thresholds": {
            "minimum_effective_donors": LOCAL_RELIABILITY_MIN_EFFECTIVE_DONORS,
            "maximum_descriptor_distance": LOCAL_RELIABILITY_MAX_DESCRIPTOR_DISTANCE,
            "maximum_atom_std_cm5_zeff_norm": list(LOCAL_RELIABILITY_MAX_ATOM_STD),
            "maximum_bond_std_angstrom": LOCAL_RELIABILITY_MAX_BOND_STD_ANGSTROM,
            "maximum_angle_std_degrees": LOCAL_RELIABILITY_MAX_ANGLE_STD_DEGREES,
        },
    }


def _nearest_atomic_types(
    numbers,
    synthons,
    target_atoms,
    *,
    catalog,
    target_atom_types,
    target_atom_ring_classes,
):
    """Bootstrap target local types from nearest LCB26 CM5/Zeff atom rows."""

    resolved: dict[int, tuple[float, float]] = {}
    trace: list[dict[str, Any]] = []
    for target_index in target_atoms:
        target = np.asarray(_descriptor(synthons, target_index, numbers), dtype=float)
        target_symbol = _symbol(numbers[target_index])
        candidates_by_tier = {
            "EXACT_LOCAL_TOPOLOGY_AND_RING_CLASS": [],
            "MATCHED_RING_CHEMISTRY": [],
            "MATCHED_COORDINATION": [],
            "ELEMENT_ONLY": [],
        }
        for row, topology, source_index in catalog.atoms_by_element.get(target_symbol, ()):
            if len(topology.atoms) != len(topology.atomic_descriptors):
                continue
            donor = topology.atomic_descriptors[source_index]
            if donor is None:
                continue
            donor_vector = np.asarray(donor[:2], dtype=float)
            target_type = target_atom_types[target_index]
            donor_type = topology.atom_types[source_index]
            target_classes = target_atom_ring_classes[target_index]
            donor_classes = topology.atom_ring_classes[source_index]
            tier = (
                "EXACT_LOCAL_TOPOLOGY_AND_RING_CLASS"
                if donor_type == target_type and donor_classes == target_classes
                else "MATCHED_RING_CHEMISTRY"
                if (set(donor_type[2]) == set(target_type[2]) and donor_classes == target_classes)
                else "MATCHED_COORDINATION"
                if donor_type[:2] == target_type[:2]
                else "ELEMENT_ONLY"
            )
            candidates_by_tier[tier].append(
                {
                    "identifier": row.get("identifier"),
                    "source_atoms": (source_index + 1,),
                    "distance": float(np.linalg.norm(donor_vector - target)),
                    "value": donor_vector,
                    "descriptor": {
                        "cm5_charge_e": float(donor_vector[0]),
                        "zeff_normalized": float(donor_vector[1]),
                    },
                    "local_topology_type": list(donor_type),
                }
            )
        selected_tier = next(
            (
                tier
                for tier in (
                    "EXACT_LOCAL_TOPOLOGY_AND_RING_CLASS",
                    "MATCHED_RING_CHEMISTRY",
                    "MATCHED_COORDINATION",
                    "ELEMENT_ONLY",
                )
                if candidates_by_tier[tier]
            ),
            "NONE",
        )
        selected = _select_local_neighbors(
            candidates_by_tier.get(selected_tier, ()), value_unit="CM5_ZEFF"
        )
        if not selected:
            # This remains an explicit rule-based fallback and is never
            # represented as an LCB26 CM5 observation.
            resolved[target_index] = (float(target[0]), float(target[1]))
            trace.append(
                {
                    "target_atom": int(target_index),
                    "atomic_number": int(numbers[target_index]),
                    "selector_descriptor": target.tolist(),
                    "target_local_topology_type": list(target_atom_types[target_index]),
                    "target_ring_classes": list(target_atom_ring_classes[target_index]),
                    "status": "RULE_BASED_EXTRAPOLATION_NO_LCB26_ATOM_TYPE",
                    "reliability": _local_transfer_reliability("atom", (), (), "NONE"),
                    "donors": [],
                }
            )
            continue
        weights = np.asarray([item["weight"] for item in selected], dtype=float)
        values = np.asarray([item["value"] for item in selected], dtype=float)
        interpolated = weights @ values
        mean = interpolated
        variance = weights @ (values - interpolated) ** 2
        weighted_std = np.sqrt(np.maximum(variance, 0.0))
        resolved[target_index] = (float(mean[0]), float(mean[1]))
        trace.append(
            {
                "target_atom": int(target_index),
                "atomic_number": int(numbers[target_index]),
                "selector_descriptor": target.tolist(),
                "target_local_topology_type": list(target_atom_types[target_index]),
                "target_ring_classes": list(target_atom_ring_classes[target_index]),
                "resolved_descriptor": mean.tolist(),
                "interpolated_lcb26_descriptor": interpolated.tolist(),
                "weighted_std": weighted_std.tolist(),
                "effective_donor_count": float(1.0 / np.sum(weights**2)),
                "compatibility_tier": selected_tier,
                "reliability": _local_transfer_reliability(
                    "atom", selected, weighted_std, selected_tier
                ),
                "status": "PROVISIONAL_BOOTSTRAP_FROM_LCB26_L0_ATOM_TYPES",
                "donors": [
                    {key: value for key, value in item.items() if key not in {"value"}}
                    for item in selected
                ],
            }
        )
    return resolved, trace


def _edge_cycle_sizes(adjacency, ring_edges):
    """Return the smallest constitutional cycle containing each ring edge."""

    result = {}
    for edge in ring_edges:
        left, right = edge
        queue = [(left, 0)]
        visited = {left}
        path_length = None
        while queue:
            atom, depth = queue.pop(0)
            for neighbor in adjacency.get(atom, ()):
                candidate = tuple(sorted((atom, neighbor)))
                if candidate == edge:
                    continue
                if neighbor == right:
                    path_length = depth + 1
                    queue.clear()
                    break
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
        if path_length is not None:
            result[edge] = int(path_length + 1)
    return result


def _local_atom_types(atoms, adjacency, edge_cycle_sizes):
    """Build graph-exact local atom types used before continuous ranking."""

    types = {}
    for atom, symbol in enumerate(atoms):
        heavy_neighbors = {
            neighbor for neighbor in adjacency.get(atom, ()) if str(atoms[neighbor]) != "H"
        }
        # Keep multiplicity: a fused junction incident to three six-membered
        # ring edges is not the same local atom type as a substituted atom in
        # one six-membered ring.
        ring_sizes = tuple(
            sorted(int(size) for edge, size in edge_cycle_sizes.items() if atom in edge)
        )
        types[atom] = (
            int(atomic_number(symbol) or 0),
            len(heavy_neighbors),
            ring_sizes,
        )
    return types


def _ring_classes_for_edges(ring_edges, orders):
    classes = {}
    for system in _edge_components(ring_edges):
        system_edges = {edge for edge in ring_edges if edge[0] in system and edge[1] in system}
        density = sum(max(float(orders.get(edge, 1.0)) - 1.0, 0.0) for edge in system_edges) / max(
            1, len(system_edges)
        )
        ring_class = _ring_unsaturation_class(density)
        for edge in system_edges:
            classes[edge] = ring_class
    return classes


def _record_local_topology(record) -> _LCB26LocalTopology:
    """Return chemically significant record bonds and their ring classes."""

    cached = record.get("_matrix_local_topology_cache")
    if cached is not None:
        return cached

    atoms = tuple(str(atom) for atom in record.get("atoms", ()))
    coordinates = np.asarray(record.get("coordinates_angstrom", ()), dtype=float)
    if coordinates.shape != (len(atoms), 3):
        coordinates = np.empty((0, 3), dtype=float)
    else:
        coordinates = coordinates.copy()
        coordinates.setflags(write=False)
    descriptors = tuple(
        _lcb26_atomic_descriptor(descriptor, symbol)
        for symbol, descriptor in zip(
            atoms,
            record.get("synthon_descriptors", ()),
            strict=False,
        )
    )
    if len(descriptors) != len(atoms):
        descriptors = tuple(None for _ in atoms)
    adjacency = {index: set() for index in range(len(atoms))}
    orders: dict[tuple[int, int], float] = {}
    for component in record.get("mayer_bond_components", ()):
        pair = component.get("atoms", ())
        order = float(component.get("total", 0.0))
        if len(pair) != 2 or order < 0.30:
            continue
        left, right = int(pair[0]) - 1, int(pair[1]) - 1
        if left not in adjacency or right not in adjacency:
            continue
        edge = tuple(sorted((left, right)))
        adjacency[left].add(right)
        adjacency[right].add(left)
        orders[edge] = order
    edges = set(orders)
    ring_edges = edges - _bridge_edges(len(atoms), edges)
    ring_class_by_edge = _ring_classes_for_edges(ring_edges, orders)
    edge_cycle_sizes = _edge_cycle_sizes(adjacency, ring_edges)
    atom_types = _local_atom_types(atoms, adjacency, edge_cycle_sizes)
    frozen_adjacency = {atom: frozenset(neighbors) for atom, neighbors in adjacency.items()}
    atom_ring_classes = {
        atom: tuple(
            sorted(
                {
                    ring_class_by_edge[edge]
                    for edge in ring_edges
                    if atom in edge and edge in ring_class_by_edge
                }
            )
        )
        for atom in atom_types
    }
    result = _LCB26LocalTopology(
        atoms=atoms,
        coordinates_angstrom=coordinates,
        atomic_descriptors=descriptors,
        adjacency=frozen_adjacency,
        orders=orders,
        ring_edges=frozenset(ring_edges),
        ring_class_by_edge=ring_class_by_edge,
        edge_cycle_sizes=edge_cycle_sizes,
        atom_types=atom_types,
        atom_ring_classes=atom_ring_classes,
    )
    # This cache lives only in the private record loaded for one preparation;
    # it is never written back to the LCB26 library.
    record["_matrix_local_topology_cache"] = result
    return result


def _record_ring_topology(record: dict[str, Any]) -> _LCB26CompiledRingTopology:
    """Compile the immutable heavy-atom ring view used by donor matching."""

    cached = record.get("_matrix_ring_topology_cache")
    if cached is not None:
        return cached
    topology = _record_local_topology(record)
    atom_indices = tuple(
        index for index, symbol in enumerate(topology.atoms) if symbol != "H"
    )
    remap = {atom: local for local, atom in enumerate(atom_indices)}
    atoms = tuple(topology.atoms[index] for index in atom_indices)
    orders = {
        tuple(sorted((remap[left], remap[right]))): float(order)
        for (left, right), order in topology.orders.items()
        if left in remap and right in remap
    }
    bonds = frozenset(orders)
    ring_edges = set(bonds) - _bridge_edges(len(atoms), set(bonds))
    systems = []
    for system in _edge_components(ring_edges):
        system_atoms = tuple(sorted(system))
        local = {atom: index for index, atom in enumerate(system_atoms)}
        edges = frozenset(
            pair for pair in ring_edges if pair[0] in system and pair[1] in system
        )
        graph = graph_from_topology(
            [atoms[index] for index in system_atoms],
            [(local[left], local[right]) for left, right in sorted(edges)],
            bond_orders={
                tuple(sorted((local[left], local[right]))): 1.0
                for left, right in edges
            },
        )
        systems.append(
            _LCB26RingSystemTopology(
                atoms=frozenset(system),
                edges=edges,
                element_signature=tuple(sorted(atoms[index] for index in system)),
                graph=graph,
                interface_signatures=_ring_interface_signatures(
                    set(system), set(bonds), atoms, orders
                ),
            )
        )
    result = _LCB26CompiledRingTopology(
        atom_indices=atom_indices,
        atoms=atoms,
        bonds=bonds,
        orders=orders,
        systems=tuple(systems),
    )
    # This remains private to the in-memory catalog record and is never
    # serialized into the LCB26 library.
    record["_matrix_ring_topology_cache"] = result
    return result


def _catalog_revision(lcb26_root: Path) -> tuple[int, int]:
    index = Path(lcb26_root).expanduser().resolve() / "enriched" / "index.json"
    try:
        stat = index.stat()
    except OSError as exc:
        raise InitialStructureError(f"missing LCB26 query index: {index}") from exc
    return int(stat.st_mtime_ns), int(stat.st_size)


def _lcb26_donor_catalog(lcb26_root: Path) -> _LCB26DonorCatalog:
    root = Path(lcb26_root).expanduser().resolve()
    return _build_lcb26_donor_catalog(str(root), *_catalog_revision(root))


def _exclude_catalog_records(
    catalog: _LCB26DonorCatalog,
    excluded_identifiers: set[str],
) -> _LCB26DonorCatalog:
    """Remove withheld identities from every compiled donor index."""
    keep = lambda item: str(item[0].get("identifier", "")) not in excluded_identifiers
    return _LCB26DonorCatalog(
        records=tuple(item for item in catalog.records if keep(item)),
        atoms_by_element={key: tuple(item for item in values if keep(item)) for key, values in catalog.atoms_by_element.items()},
        bonds_by_elements={key: tuple(item for item in values if keep(item)) for key, values in catalog.bonds_by_elements.items()},
        angles_by_elements={key: tuple(item for item in values if keep(item)) for key, values in catalog.angles_by_elements.items()},
    )


def _shortlist_catalog(catalog: _LCB26DonorCatalog, numbers: Sequence[int], *, target_synthons=None, limit: int) -> _LCB26DonorCatalog:
    """Restrict local donors to globally closest molecular compositions."""
    target = {}
    for z in numbers:
        target[str(_symbol(int(z)))] = target.get(str(_symbol(int(z))), 0) + 1
    target_signatures = {}
    if target_synthons is not None:
        for i, z in enumerate(numbers):
            signature = tuple(target_synthons.canonical_signature(i))
            target_signatures[signature] = target_signatures.get(signature, 0) + 1
    scored = []
    for row, record, _ring_topology in catalog.records:
        counts = {}
        for atom in record.get("atoms", ()):
            counts[str(atom)] = counts.get(str(atom), 0) + 1
        distance = abs(sum(counts.values()) - len(numbers))
        distance += 2.0 * sum(abs(counts.get(k, 0) - v) for k, v in target.items())
        if target_signatures:
            donor_signatures = {}
            for item in record.get("synthon_descriptors", ()):
                signature = tuple(item.get("canonical_signature", ()))
                donor_signatures[signature] = donor_signatures.get(signature, 0) + 1
            distance += 3.0 * sum(abs(donor_signatures.get(k, 0) - v) for k, v in target_signatures.items())
        scored.append((float(distance), str(row.get("identifier", ""))))
    ranked = sorted(scored)
    if not ranked:
        raise InitialStructureError("LCB26 has no molecular donor records")
    allowed = {identifier for _, identifier in ranked[: max(1, int(limit))]}
    records = tuple(item for item in catalog.records if str(item[0].get("identifier", "")) in allowed)
    def keep(items):
        return tuple(item for item in items if str(item[0].get("identifier", "")) in allowed)
    return _LCB26DonorCatalog(
        records=records,
        atoms_by_element={k: keep(v) for k, v in catalog.atoms_by_element.items()},
        bonds_by_elements={k: keep(v) for k, v in catalog.bonds_by_elements.items()},
        angles_by_elements={k: keep(v) for k, v in catalog.angles_by_elements.items()},
    )


@lru_cache(maxsize=8)
def _build_lcb26_donor_catalog(
    root_text: str,
    _index_mtime_ns: int,
    _index_size: int,
) -> _LCB26DonorCatalog:
    """Compile element-indexed LCB26 donors for one exact index revision."""

    root = Path(root_text)
    records: list[
        tuple[dict[str, Any], dict[str, Any], _LCB26CompiledRingTopology]
    ] = []
    atoms: dict[str, list[tuple[dict[str, Any], _LCB26LocalTopology, int]]] = {}
    bonds: dict[
        tuple[str, str],
        list[tuple[dict[str, Any], _LCB26LocalTopology, tuple[int, int]]],
    ] = {}
    angles: dict[
        tuple[str, str, str],
        list[
            tuple[
                dict[str, Any],
                _LCB26LocalTopology,
                tuple[int, int, int],
            ]
        ],
    ] = {}
    for row in query_lcb26(root, limit=None):
        try:
            record = load_lcb26_reference(root, row)
            topology = _record_local_topology(record)
        except Exception:
            geometry_path = root / str(row.get("geometry_path", row.get("geometry", "")))
            try:
                geometry = read_xyz(geometry_path)
                numbers = tuple(_atomic_number(symbol) for symbol in geometry.atoms)
                _continuous, discrete, _rings, _synthons, _aromaticity = build_topology_objects(
                    geometry.coordinates_angstrom, numbers
                )
                record = {
                    "atoms": list(geometry.atoms),
                    "coordinates_angstrom": np.asarray(geometry.coordinates_angstrom).tolist(),
                    "mayer_bond_components": [
                        {"atoms": [int(bond[0]) + 1, int(bond[1]) + 1], "total": 1.0}
                        for bond in discrete.bonds
                    ],
                    "synthon_descriptors": [],
                }
                topology = _record_local_topology(record)
            except Exception:
                continue
        if not topology.atoms or topology.coordinates_angstrom.shape != (
            len(topology.atoms),
            3,
        ):
            continue
        ring_topology = _record_ring_topology(record)
        records.append((row, record, ring_topology))
        for atom, symbol in enumerate(topology.atoms):
            atoms.setdefault(symbol, []).append((row, topology, atom))
        for edge in topology.orders:
            left, right = edge
            key = tuple(sorted((topology.atoms[left], topology.atoms[right])))
            bonds.setdefault(key, []).append((row, topology, edge))
        for center, neighbors in topology.adjacency.items():
            for left in neighbors:
                for right in neighbors:
                    if left >= right:
                        continue
                    terminals = sorted((topology.atoms[left], topology.atoms[right]))
                    key = (topology.atoms[center], terminals[0], terminals[1])
                    angles.setdefault(key, []).append((row, topology, (left, center, right)))
    return _LCB26DonorCatalog(
        records=tuple(records),
        atoms_by_element={key: tuple(value) for key, value in atoms.items()},
        bonds_by_elements={key: tuple(value) for key, value in bonds.items()},
        angles_by_elements={key: tuple(value) for key, value in angles.items()},
    )


def _bond_donors(
    numbers,
    target_atomic_descriptors,
    left,
    right,
    *,
    catalog,
    trace=None,
    target_index=None,
    target_order=None,
    target_distance=None,
    require_ring=False,
    target_atom_types=None,
    target_cycle_size=None,
    target_ring_class=None,
):
    ql, zl = target_atomic_descriptors[left]
    qr, zr = target_atomic_descriptors[right]
    order_target = float(1.0 if target_order is None else target_order)
    target_direct = np.asarray([ql, zl, qr, zr, order_target], dtype=float)
    candidate_groups = {
        "EXACT_LOCAL_TOPOLOGY": [],
        "MATCHED_RING_CHEMISTRY": [],
        "MATCHED_COORDINATION": [],
        "ELEMENT_ONLY": [],
    }
    element_key = tuple(sorted((str(_symbol(numbers[left])), str(_symbol(numbers[right])))))
    for _row, topology, selected_edge in catalog.bonds_by_elements.get(element_key, ()):
        atoms = topology.atoms
        for edge, donor_order in ((selected_edge, topology.orders[selected_edge]),):
            i, j = edge
            if (
                i >= len(atoms)
                or j >= len(atoms)
                or {atoms[i], atoms[j]}
                != {str(_symbol(numbers[left])), str(_symbol(numbers[right]))}
            ):
                continue
            if i >= len(topology.atomic_descriptors) or j >= len(topology.atomic_descriptors):
                continue
            if require_ring and edge not in topology.ring_edges:
                continue
            if not require_ring and target_cycle_size is None and edge in topology.ring_edges:
                continue
            if target_cycle_size is not None and topology.edge_cycle_sizes.get(edge) != int(
                target_cycle_size
            ):
                continue
            if target_ring_class is not None and topology.ring_class_by_edge.get(edge) != str(
                target_ring_class
            ):
                continue
            donor_i = topology.atomic_descriptors[i]
            donor_j = topology.atomic_descriptors[j]
            if donor_i is None or donor_j is None:
                continue
            donor_direct = np.asarray(
                [donor_i[0], donor_i[1], donor_j[0], donor_j[1], donor_order],
                dtype=float,
            )
            donor_reverse = np.asarray(
                [donor_j[0], donor_j[1], donor_i[0], donor_i[1], donor_order],
                dtype=float,
            )

            def compatibility(source_left, source_right):
                if target_atom_types is None:
                    return "EXACT_LOCAL_TOPOLOGY"
                donor_left = topology.atom_types[source_left]
                donor_right = topology.atom_types[source_right]
                if (
                    donor_left == target_atom_types[left]
                    and donor_right == target_atom_types[right]
                ):
                    return "EXACT_LOCAL_TOPOLOGY"
                if require_ring:
                    return "MATCHED_RING_CHEMISTRY"
                if (
                    donor_left[:2] == target_atom_types[left][:2]
                    and donor_right[:2] == target_atom_types[right][:2]
                ):
                    return "MATCHED_COORDINATION"
                return "ELEMENT_ONLY"

            direct_tier = compatibility(i, j)
            reverse_tier = compatibility(j, i)
            tier_order = {
                "EXACT_LOCAL_TOPOLOGY": 0,
                "MATCHED_RING_CHEMISTRY": 1,
                "MATCHED_COORDINATION": 2,
                "ELEMENT_ONLY": 3,
            }
            direct_distance = float(np.linalg.norm(donor_direct - target_direct))
            reverse_distance = float(np.linalg.norm(donor_reverse - target_direct))
            reverse_mapping = (tier_order[reverse_tier], reverse_distance) < (
                tier_order[direct_tier],
                direct_distance,
            )
            distance = reverse_distance if reverse_mapping else direct_distance
            selected_tier = reverse_tier if reverse_mapping else direct_tier
            coords = topology.coordinates_angstrom
            candidate_groups[selected_tier].append(
                {
                    "value": float(np.linalg.norm(coords[i] - coords[j])),
                    "distance": distance,
                    "identifier": _row.get("identifier"),
                    "source_atoms": ((j + 1, i + 1) if reverse_mapping else (i + 1, j + 1)),
                    "mayer_bond_orders": [float(donor_order)],
                    "target_mayer_bond_orders": [float(order_target)],
                    "compatibility_tier": selected_tier,
                }
            )
    if target_distance is not None:
        for tier in candidate_groups:
            candidate_groups[tier] = [
                item for item in candidate_groups[tier]
                if abs(float(item["value"]) - float(target_distance)) <= 0.05
            ]
    candidates = next(
        (
            candidate_groups[tier]
            for tier in (
                "EXACT_LOCAL_TOPOLOGY",
                "MATCHED_RING_CHEMISTRY",
                "MATCHED_COORDINATION",
                "ELEMENT_ONLY",
            )
            if candidate_groups[tier]
        ),
        [],
    )
    return _top_weighted(
        candidates,
        trace=trace,
        target_index=target_index,
        role="bond",
        value_unit="angstrom",
    )


def _angle_donors(
    numbers,
    target_atomic_descriptors,
    left,
    center,
    right,
    *,
    catalog,
    trace=None,
    target_index=None,
    target_orders=(None, None),
    require_ring=False,
    target_atom_types=None,
    target_cycle_sizes=(None, None),
    target_ring_classes=(None, None),
):
    target_left = target_atomic_descriptors[left]
    target_center = target_atomic_descriptors[center]
    target_right = target_atomic_descriptors[right]
    requested_orders = tuple(1.0 if value is None else float(value) for value in target_orders)
    target_vector = np.asarray(
        [
            target_left[0],
            target_left[1],
            target_center[0],
            target_center[1],
            target_right[0],
            target_right[1],
            requested_orders[0],
            requested_orders[1],
        ],
        dtype=float,
    )
    candidate_groups = {
        "EXACT_LOCAL_TOPOLOGY": [],
        "MATCHED_RING_CHEMISTRY": [],
        "MATCHED_COORDINATION": [],
        "ELEMENT_ONLY": [],
    }
    symbols = tuple(_symbol(number) for number in numbers)
    terminal_symbols = sorted((symbols[left], symbols[right]))
    element_key = (symbols[center], terminal_symbols[0], terminal_symbols[1])
    for _row, topology, selected_angle in catalog.angles_by_elements.get(element_key, ()):
        atoms = topology.atoms
        coords = topology.coordinates_angstrom
        if len(topology.atomic_descriptors) != len(atoms):
            continue
        selected_left, selected_center, selected_right = selected_angle
        for c, neighbours in ((selected_center, topology.adjacency[selected_center]),):
            for i in (selected_left,):
                for k in (selected_right,):
                    if (
                        i >= k
                        or {atoms[i], atoms[k]} != {symbols[left], symbols[right]}
                        or atoms[c] != symbols[center]
                    ):
                        continue
                    first_edge = tuple(sorted((i, c)))
                    second_edge = tuple(sorted((c, k)))
                    if (
                        require_ring
                        and not {
                            first_edge,
                            second_edge,
                        }
                        <= topology.ring_edges
                    ):
                        continue
                    if (
                        not require_ring
                        and target_cycle_sizes == (None, None)
                        and (
                            first_edge in topology.ring_edges or second_edge in topology.ring_edges
                        )
                    ):
                        continue
                    if target_cycle_sizes[0] is not None and (
                        topology.edge_cycle_sizes.get(first_edge) != int(target_cycle_sizes[0])
                        and topology.edge_cycle_sizes.get(second_edge) != int(target_cycle_sizes[0])
                    ):
                        continue
                    donor_classes = (
                        topology.ring_class_by_edge.get(first_edge),
                        topology.ring_class_by_edge.get(second_edge),
                    )
                    donor_orders = (
                        float(topology.orders[first_edge]),
                        float(topology.orders[second_edge]),
                    )
                    donor_i = topology.atomic_descriptors[i]
                    donor_c = topology.atomic_descriptors[c]
                    donor_k = topology.atomic_descriptors[k]
                    if donor_i is None or donor_c is None or donor_k is None:
                        continue
                    direct_vector = np.asarray(
                        [
                            donor_i[0],
                            donor_i[1],
                            donor_c[0],
                            donor_c[1],
                            donor_k[0],
                            donor_k[1],
                            donor_orders[0],
                            donor_orders[1],
                        ],
                        dtype=float,
                    )
                    reverse_vector = np.asarray(
                        [
                            donor_k[0],
                            donor_k[1],
                            donor_c[0],
                            donor_c[1],
                            donor_i[0],
                            donor_i[1],
                            donor_orders[1],
                            donor_orders[0],
                        ],
                        dtype=float,
                    )

                    def compatibility(source_left, source_center, source_right):
                        if target_atom_types is None:
                            return "EXACT_LOCAL_TOPOLOGY"
                        donor_types = (
                            topology.atom_types[source_left],
                            topology.atom_types[source_center],
                            topology.atom_types[source_right],
                        )
                        target_types = (
                            target_atom_types[left],
                            target_atom_types[center],
                            target_atom_types[right],
                        )
                        if donor_types == target_types:
                            return "EXACT_LOCAL_TOPOLOGY"
                        if require_ring:
                            return "MATCHED_RING_CHEMISTRY"
                        if all(
                            donor[:2] == target[:2]
                            for donor, target in zip(donor_types, target_types, strict=True)
                        ):
                            return "MATCHED_COORDINATION"
                        return "ELEMENT_ONLY"

                    direct_tier = compatibility(i, c, k)
                    reverse_tier = compatibility(k, c, i)

                    def edge_compatible(donor_edge, donor_class, size, ring_class):
                        donor_is_ring = donor_edge in topology.ring_edges
                        size_matches = (
                            not donor_is_ring
                            if size is None
                            else topology.edge_cycle_sizes.get(donor_edge) == int(size)
                        )
                        class_matches = (
                            donor_class is None if ring_class is None else donor_class == ring_class
                        )
                        return size_matches and class_matches

                    direct_compatible = edge_compatible(
                        first_edge,
                        donor_classes[0],
                        target_cycle_sizes[0],
                        target_ring_classes[0],
                    ) and edge_compatible(
                        second_edge,
                        donor_classes[1],
                        target_cycle_sizes[1],
                        target_ring_classes[1],
                    )
                    reverse_compatible = edge_compatible(
                        second_edge,
                        donor_classes[1],
                        target_cycle_sizes[0],
                        target_ring_classes[0],
                    ) and edge_compatible(
                        first_edge,
                        donor_classes[0],
                        target_cycle_sizes[1],
                        target_ring_classes[1],
                    )
                    mixed_ring_target = sum(size is not None for size in target_cycle_sizes) == 1
                    acyclic_donor = (
                        first_edge not in topology.ring_edges
                        and second_edge not in topology.ring_edges
                    )
                    mixed_ring_fallback = (
                        not direct_compatible
                        and not reverse_compatible
                        and mixed_ring_target
                        and acyclic_donor
                    )
                    if mixed_ring_fallback:
                        # An exocyclic angle at a ring atom may lack an exact
                        # ring-substituent donor.  Acyclic donors of the same
                        # electronic/local type are a safe fallback; fully
                        # endocyclic, potentially strained angles are not.
                        direct_compatible = True
                        reverse_compatible = True
                        if direct_tier == "EXACT_LOCAL_TOPOLOGY":
                            direct_tier = "MATCHED_COORDINATION"
                        if reverse_tier == "EXACT_LOCAL_TOPOLOGY":
                            reverse_tier = "MATCHED_COORDINATION"
                    if not direct_compatible and not reverse_compatible:
                        continue
                    tier_order = {
                        "EXACT_LOCAL_TOPOLOGY": 0,
                        "MATCHED_RING_CHEMISTRY": 1,
                        "MATCHED_COORDINATION": 2,
                        "ELEMENT_ONLY": 3,
                    }
                    direct_distance = float(np.linalg.norm(direct_vector - target_vector))
                    reverse_distance = float(np.linalg.norm(reverse_vector - target_vector))
                    direct_key = (
                        (tier_order[direct_tier], direct_distance)
                        if direct_compatible
                        else (99, float("inf"))
                    )
                    reverse_key = (
                        (tier_order[reverse_tier], reverse_distance)
                        if reverse_compatible
                        else (99, float("inf"))
                    )
                    reverse_mapping = reverse_key < direct_key
                    selected_tier = reverse_tier if reverse_mapping else direct_tier
                    candidate_groups[selected_tier].append(
                        {
                            "value": _angle(coords[i], coords[c], coords[k]),
                            "distance": min(direct_distance, reverse_distance),
                            "identifier": _row.get("identifier"),
                            "source_atoms": (
                                (k + 1, c + 1, i + 1) if reverse_mapping else (i + 1, c + 1, k + 1)
                            ),
                            "mayer_bond_orders": list(
                                reversed(donor_orders) if reverse_mapping else donor_orders
                            ),
                            "target_mayer_bond_orders": list(requested_orders),
                            "compatibility_tier": selected_tier,
                        }
                    )
    candidates = next(
        (
            candidate_groups[tier]
            for tier in (
                "EXACT_LOCAL_TOPOLOGY",
                "MATCHED_RING_CHEMISTRY",
                "MATCHED_COORDINATION",
                "ELEMENT_ONLY",
            )
            if candidate_groups[tier]
        ),
        [],
    )
    return _top_weighted(
        candidates,
        trace=trace,
        target_index=target_index,
        role="angle",
        value_unit="radian",
    )


def _symbol(number: int) -> str:
    from matrix_chem.topology.elements import atomic_symbol

    return str(atomic_symbol(number))


def _top_weighted(
    candidates,
    *,
    trace=None,
    target_index=None,
    role="",
    value_unit="",
):
    selected = _select_local_neighbors(candidates, value_unit=value_unit)
    if not selected:
        return []
    if trace is not None and target_index is not None:
        weights = np.asarray([item["weight"] for item in selected], dtype=float)
        values = np.asarray([item["value"] for item in selected], dtype=float)
        mean = float(weights @ values)
        variance = float(weights @ (values - mean) ** 2)
        weighted_std = math.sqrt(max(0.0, variance))
        compatibility_tier = str(selected[0].get("compatibility_tier", "UNKNOWN"))
        trace.append(
            {
                "role": role,
                "internal_index": int(target_index),
                "status": "LCB26_NEAREST_LOCAL_GAUSSIAN_TRANSFER",
                "weighted_value": mean,
                "weighted_std": weighted_std,
                "robust_reweight": selected[0].get("robust_reweight", {}),
                "effective_donor_count": float(1.0 / np.sum(weights**2)),
                "reliability": _local_transfer_reliability(
                    role, selected, (weighted_std,), compatibility_tier
                ),
                "donors": [
                    {key: value for key, value in item.items() if key != "value"}
                    for item in selected
                ],
            }
        )
    return selected


def _weighted_mean(values):
    weights = np.asarray([item["weight"] for item in values], dtype=float)
    observations = np.asarray([item["value"] for item in values], dtype=float)
    return float(weights @ observations)


def _angle(a, b, c):
    first = np.asarray(a) - np.asarray(b)
    second = np.asarray(c) - np.asarray(b)
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return float(math.acos(np.clip(float(first @ second) / max(denominator, 1.0e-15), -1.0, 1.0)))


def _internal_values(coords, bonds, angles, dihedrals=()):
    values = [float(np.linalg.norm(coords[i] - coords[j])) for i, j in bonds]
    values.extend(_angle(coords[i], coords[j], coords[k]) for i, j, k in angles)
    if dihedrals:
        from matrix_chem.primitive_coordinates import dihedral

        values.extend(
            float(dihedral(i, j, k, ell, np.asarray(coords, dtype=float)))
            for i, j, k, ell in dihedrals
        )
    return np.asarray(values, dtype=float)


def _endocyclic_dihedrals(bonds, ring_edges):
    """Return ring-internal torsions used only to protect LCB26 micro-closure."""
    adjacency: dict[int, set[int]] = {}
    for left, right in bonds:
        adjacency.setdefault(int(left), set()).add(int(right))
        adjacency.setdefault(int(right), set()).add(int(left))
    result: set[tuple[int, int, int, int]] = set()
    for center_left, center_right in sorted(ring_edges):
        left_ring_neighbours = {
            atom
            for atom in adjacency.get(center_left, ())
            if atom != center_right and tuple(sorted((atom, center_left))) in ring_edges
        }
        right_ring_neighbours = {
            atom
            for atom in adjacency.get(center_right, ())
            if atom != center_left and tuple(sorted((center_right, atom))) in ring_edges
        }
        for left in left_ring_neighbours:
            for right in right_ring_neighbours:
                if len({left, center_left, center_right, right}) < 4:
                    continue
                forward = (left, center_left, center_right, right)
                reverse = tuple(reversed(forward))
                result.add(min(forward, reverse))
    return tuple(sorted(result))


def _all_dihedrals(bonds):
    """Return every proper torsion so weighted-B closure cannot drift freely."""

    adjacency: dict[int, set[int]] = {}
    for left, right in bonds:
        adjacency.setdefault(int(left), set()).add(int(right))
        adjacency.setdefault(int(right), set()).add(int(left))
    result: set[tuple[int, int, int, int]] = set()
    for center_left, center_right in bonds:
        for left in adjacency.get(center_left, set()) - {center_right}:
            for right in adjacency.get(center_right, set()) - {center_left}:
                if len({left, center_left, center_right, right}) < 4:
                    continue
                forward = (left, center_left, center_right, right)
                result.add(min(forward, tuple(reversed(forward))))
    return tuple(sorted(result))


def _periodic_difference(value: float) -> float:
    return float((float(value) + math.pi) % (2.0 * math.pi) - math.pi)


def _ring_microclosure_audit(
    reference,
    final,
    bond_indices,
    angle_indices,
    dihedral_indices,
):
    reference = np.asarray(reference, dtype=float)
    final = np.asarray(final, dtype=float)

    def maximum(indices, *, periodic=False):
        if not indices:
            return 0.0
        differences = [
            _periodic_difference(final[index] - reference[index])
            if periodic
            else float(final[index] - reference[index])
            for index in sorted(indices)
        ]
        return float(np.max(np.abs(np.asarray(differences, dtype=float))))

    bond = maximum(bond_indices)
    angle = math.degrees(maximum(angle_indices))
    dihedral = math.degrees(maximum(dihedral_indices, periodic=True))
    valid = (
        bond <= RING_BOND_MICROCLOSURE_LIMIT_ANGSTROM + 1.0e-12
        and angle <= RING_ANGLE_MICROCLOSURE_LIMIT_DEGREES + 1.0e-12
        and dihedral <= RING_DIHEDRAL_MICROCLOSURE_LIMIT_DEGREES + 1.0e-12
    )
    return {
        "valid": bool(valid),
        "status": "PASS" if valid else "FAIL",
        "maximum_bond_change_angstrom": bond,
        "maximum_angle_change_degrees": angle,
        "maximum_dihedral_change_degrees": dihedral,
        "bond_limit_angstrom": RING_BOND_MICROCLOSURE_LIMIT_ANGSTROM,
        "angle_limit_degrees": RING_ANGLE_MICROCLOSURE_LIMIT_DEGREES,
        "dihedral_limit_degrees": RING_DIHEDRAL_MICROCLOSURE_LIMIT_DEGREES,
        "protected_bond_count": len(bond_indices),
        "protected_angle_count": len(angle_indices),
        "protected_dihedral_count": len(dihedral_indices),
    }


def _attach_fallback_primitive_donors(ring_audit, donor_trace, bonds, angles) -> None:
    """Attach the actual local LCB26 support to each fallback ring block."""

    trace_by_index = {}
    for item in donor_trace:
        if "internal_index" not in item:
            continue
        identifiers = []
        for donor in item.get("donors", ()):
            identifier = donor.get("identifier") if isinstance(donor, dict) else donor
            if identifier:
                identifiers.append(str(identifier))
        trace_by_index[int(item["internal_index"])] = tuple(identifiers)
    angle_offset = len(bonds)
    used_fallback = False
    for system in ring_audit.get("systems", ()):
        if not system.get("fallback_used", False):
            continue
        used_fallback = True
        atoms = {int(atom) for atom in system.get("atoms", ())}
        indices = {index for index, pair in enumerate(bonds) if set(pair) <= atoms}
        indices.update(
            angle_offset + index for index, angle in enumerate(angles) if set(angle) <= atoms
        )
        donors = sorted({donor for index in indices for donor in trace_by_index.get(index, ())})
        supported = sum(bool(trace_by_index.get(index)) for index in indices)
        system["local_primitive_donors"] = donors
        system["local_primitive_target_count"] = len(indices)
        system["local_primitive_supported_count"] = int(supported)
        system["fallback_tier"] = (
            "LCB26_LOCAL_PRIMITIVES_WEIGHTED_B"
            if supported
            else "SWITCH_TOPOLOGICAL_SEED_PERIODIC_ZAFF"
        )
    if used_fallback:
        ring_audit["status"] = "PASS_WITH_EXPLICIT_FALLBACK"
        ring_audit["fallback_used"] = True


def _ring_fallback_audit(
    reference,
    final,
    bonds,
    angles,
    dihedrals,
    all_bond_indices,
    all_angle_indices,
    all_dihedral_indices,
    fallback_atoms,
    donor_trace,
    topology_gate,
):
    """Report, without pretending micro-closure, an accepted fallback block."""

    if not fallback_atoms:
        return {
            "valid": True,
            "status": "NOT_APPLICABLE",
            "fallback_used": False,
        }
    bond_indices = {index for index in all_bond_indices if set(bonds[index]) <= fallback_atoms}
    angle_offset = len(bonds)
    angle_indices = {
        angle_offset + index
        for index, angle in enumerate(angles)
        if angle_offset + index in all_angle_indices and set(angle) <= fallback_atoms
    }
    dihedral_offset = len(bonds) + len(angles)
    dihedral_indices = {
        dihedral_offset + index
        for index, dihedral in enumerate(dihedrals)
        if dihedral_offset + index in all_dihedral_indices and set(dihedral) <= fallback_atoms
    }
    reference = np.asarray(reference, dtype=float)
    final = np.asarray(final, dtype=float)

    def maximum(indices, *, periodic=False, degrees=False):
        if not indices:
            return 0.0
        values = []
        for index in sorted(indices):
            difference = float(final[index] - reference[index])
            values.append(_periodic_difference(difference) if periodic else difference)
        result = float(np.max(np.abs(np.asarray(values, dtype=float))))
        return math.degrees(result) if degrees else result

    supported_indices = {int(item["internal_index"]) for item in donor_trace if item.get("donors")}
    target_indices = bond_indices | angle_indices
    supported = len(target_indices & supported_indices)
    valid = bool(topology_gate.get("valid", False)) and bool(np.all(np.isfinite(final)))
    return {
        "valid": valid,
        "status": "PASS_WITH_EXPLICIT_FALLBACK" if valid else "FAIL",
        "fallback_used": True,
        "acceptance_basis": ("SWITCH_CONSTITUTION_PLUS_WEIGHTED_B_MANIFOLD_PLUS_TOPOLOGY_GATE"),
        "local_primitive_supported_count": int(supported),
        "local_primitive_target_count": len(target_indices),
        "periodic_zaff_required": bool(supported < len(target_indices)),
        "maximum_bond_change_angstrom": maximum(bond_indices),
        "maximum_angle_change_degrees": maximum(angle_indices, degrees=True),
        "maximum_endocyclic_dihedral_change_degrees": maximum(
            dihedral_indices, periodic=True, degrees=True
        ),
        "microclosure_limits_apply": False,
        "topology_gate_status": topology_gate.get("status"),
    }


def _target_ring_system_topology(
    system: set[int],
    *,
    ring_edges: set[tuple[int, int]],
    heavy_bonds: set[tuple[int, int]],
    atoms: Sequence[str],
    bond_orders: dict[tuple[int, int], float],
) -> _TargetRingSystemTopology:
    ordered_atoms = tuple(sorted(system))
    remap = {atom: local for local, atom in enumerate(ordered_atoms)}
    edges = frozenset(
        pair for pair in ring_edges if pair[0] in system and pair[1] in system
    )
    graph = graph_from_topology(
        [atoms[index] for index in ordered_atoms],
        [(remap[left], remap[right]) for left, right in sorted(edges)],
        bond_orders={
            tuple(sorted((remap[left], remap[right]))): 1.0
            for left, right in edges
        },
    )
    return _TargetRingSystemTopology(
        atoms=frozenset(system),
        ordered_atoms=ordered_atoms,
        edges=edges,
        graph=graph,
        interface_signatures=_ring_interface_signatures(
            system,
            heavy_bonds,
            atoms,
            bond_orders,
        ),
    )


def _select_lcb26_ring_transfer(
    target: _TargetRingSystemTopology,
    records: Sequence[
        tuple[dict[str, Any], dict[str, Any], _LCB26CompiledRingTopology]
    ],
    *,
    geometry: MolecularGeometry,
    bond_orders: dict[tuple[int, int], float] | None,
) -> tuple[_RingTransferCandidate | None, _RingTransferCandidate | None]:
    """Return the best admissible and best topology-only ring candidates."""

    best = None
    best_topological = None
    target_elements = tuple(sorted(geometry.atoms[index] for index in target.atoms))
    for row, record, compiled_ring in records:
        donor_heavy = compiled_ring.atom_indices
        if len(donor_heavy) < len(target.atoms) or not compiled_ring.bonds:
            continue
        compatible_systems = [
            candidate
            for candidate in compiled_ring.systems
            if len(candidate.atoms) == len(target.atoms)
            and candidate.element_signature == target_elements
        ]
        for donor_topology in compatible_systems:
            donor_atoms = tuple(sorted(donor_topology.atoms))
            donor_edges = set(donor_topology.edges)
            matches = find_substructure_matches(
                target.graph,
                donor_topology.graph,
                use_chirality=False,
                uniquify=False,
                max_matches=32,
                allow_attachment_hydrogen_mismatch=True,
            )
            for match in matches:
                mapping = {
                    target.ordered_atoms[target_local]: donor_atoms[donor_local]
                    for donor_local, target_local in enumerate(match)
                }
                if target.atoms - mapping.keys():
                    continue
                if {mapping[index] for index in target.atoms} != donor_topology.atoms:
                    continue
                mapped_edges = {
                    tuple(sorted((mapping[left], mapping[right])))
                    for left, right in target.edges
                }
                if mapped_edges != donor_edges:
                    continue
                order_distance = 0.0
                order_class_mismatches = 0
                target_density = 0.0
                donor_density = 0.0
                if bond_orders:
                    for target_edge in target.edges:
                        donor_edge = tuple(
                            sorted((mapping[target_edge[0]], mapping[target_edge[1]]))
                        )
                        target_order = float(bond_orders.get(target_edge, 1.0))
                        donor_order = float(compiled_ring.orders.get(donor_edge, 1.0))
                        order_distance += abs(target_order - donor_order)
                        target_density += max(target_order - 1.0, 0.0)
                        donor_density += max(donor_order - 1.0, 0.0)
                        order_class_mismatches += int(
                            _bond_order_class(target_order)
                            != _bond_order_class(donor_order)
                        )
                    edge_count = max(1, len(target.edges))
                    order_distance /= edge_count
                    target_density /= edge_count
                    donor_density /= edge_count
                interface_matches, interface_mismatches = _ring_interface_score(
                    mapping,
                    target_signatures=target.interface_signatures,
                    donor_signatures=donor_topology.interface_signatures,
                )
                target_charge = int(geometry.charge or 0)
                target_multiplicity = int(geometry.multiplicity or 1)
                state_mismatch = int(
                    int(row.get("charge", 0)) != target_charge
                ) + int(int(row.get("multiplicity", 1)) != target_multiplicity)
                candidate = _RingTransferCandidate(
                    score=(
                        -float(order_class_mismatches),
                        -float(state_mismatch),
                        -float(len(donor_heavy) - len(target.atoms)),
                        -order_distance,
                        float(interface_matches),
                        -float(interface_mismatches),
                        -float(int(row.get("atom_count", 0))),
                    ),
                    row=row,
                    record=record,
                    donor_atom_indices=donor_heavy,
                    mapping=mapping,
                    order_distance=order_distance,
                    order_class_mismatches=order_class_mismatches,
                    target_unsaturation_density=target_density,
                    donor_unsaturation_density=donor_density,
                    electronic_state_mismatch=state_mismatch,
                )
                if best_topological is None or candidate.score > best_topological.score:
                    best_topological = candidate
                unsaturation_compatible = _ring_unsaturation_class(
                    target_density
                ) == _ring_unsaturation_class(donor_density)
                if unsaturation_compatible and state_mismatch == 0:
                    if best is None or candidate.score > best.score:
                        best = candidate
    return best, best_topological


def _ring_transfer_fallback_record(
    target: _TargetRingSystemTopology,
    *,
    bond_orders: dict[tuple[int, int], float] | None,
    rejected_candidate: _RingTransferCandidate | None,
) -> dict[str, Any]:
    target_density = 0.0
    if target.edges and bond_orders:
        target_density = sum(
            max(float(bond_orders.get(edge, 1.0)) - 1.0, 0.0)
            for edge in target.edges
        ) / len(target.edges)
    rejected = None
    if rejected_candidate is not None:
        rejected = {
            "identifier": rejected_candidate.row.get("identifier"),
            "name": rejected_candidate.row.get("name"),
            "reason": "RING_UNSATURATION_OR_ELECTRONIC_STATE_MISMATCH",
            "mean_ring_bond_order_distance": rejected_candidate.order_distance,
            "ring_bond_order_class_mismatches": (
                rejected_candidate.order_class_mismatches
            ),
            "donor_ring_unsaturation_density": (
                rejected_candidate.donor_unsaturation_density
            ),
            "electronic_state_mismatch": (
                rejected_candidate.electronic_state_mismatch
            ),
            "used_for_coordinates": False,
        }
    return {
        "atoms": [int(index) for index in target.ordered_atoms],
        "donor": None,
        "donor_name": None,
        "target_ring_unsaturation_density": float(target_density),
        "ring_unsaturation_class": _ring_unsaturation_class(target_density),
        "electronic_state_mismatch": 0,
        "fallback_used": True,
        "fallback_reason": "NO_COMPLETE_COMPATIBLE_LCB26_RING_SYSTEM",
        "topological_candidate_rejected": rejected,
        "coordinate_seed": "SWITCH_CONSTITUTIONAL_EMBEDDING",
        "status": "LCB26_RING_FALLBACK_PENDING_LOCAL_PRIMITIVES",
    }


def _apply_ring_transfer_candidate(
    candidate: _RingTransferCandidate,
    target: _TargetRingSystemTopology,
    coordinates: np.ndarray,
    *,
    heavy_bonds: set[tuple[int, int]],
    heavy_count: int,
) -> dict[str, Any]:
    target_indices = np.asarray(target.ordered_atoms, dtype=int)
    source_indices = np.asarray(
        [
            candidate.donor_atom_indices[candidate.mapping[index]]
            for index in target_indices
        ],
        dtype=int,
    )
    donor_xyz = np.asarray(candidate.record["coordinates_angstrom"], dtype=float)[
        source_indices
    ]
    previous = coordinates.copy()
    coordinates[target_indices] = _kabsch_place(
        donor_xyz,
        coordinates[target_indices],
    )
    _transport_single_anchor_components(
        coordinates,
        previous,
        set(int(index) for index in target_indices),
        heavy_bonds,
        heavy_count,
    )
    return {
        "atoms": [int(index) for index in target_indices],
        "donor": candidate.row.get("identifier"),
        "donor_name": candidate.row.get("name"),
        "mean_ring_bond_order_distance": candidate.order_distance,
        "ring_bond_order_class_mismatches": candidate.order_class_mismatches,
        "target_ring_unsaturation_density": candidate.target_unsaturation_density,
        "donor_ring_unsaturation_density": candidate.donor_unsaturation_density,
        "ring_unsaturation_class": _ring_unsaturation_class(
            candidate.target_unsaturation_density
        ),
        "electronic_state_mismatch": candidate.electronic_state_mismatch,
        "fallback_used": False,
        "fallback_tier": "NONE",
        "status": "LCB26_COMPLETE_RING_SYSTEM_MICROCLOSURE_REFERENCE",
    }


def _transfer_lcb26_ring_systems(
    geometry,
    constitutional_bonds,
    constitutional_bond_orders,
    records,
):
    """Transplant complete ring systems from the largest compatible LCB26 match."""
    if constitutional_bonds is None:
        return geometry, (), {"status": "NOT_APPLICABLE", "systems": []}
    heavy_count = sum(1 for atom in geometry.atoms if atom != "H")
    heavy_bonds = {tuple(sorted(map(int, pair))) for pair in constitutional_bonds}
    ring_edges = heavy_bonds - _bridge_edges(heavy_count, heavy_bonds)
    systems = _edge_components(ring_edges)
    if not systems:
        return geometry, (), {"status": "NO_RINGS", "systems": []}
    coordinates = np.asarray(geometry.coordinates_angstrom, dtype=float).copy()
    frozen: set[int] = set()
    audit = []
    for system in systems:
        target = _target_ring_system_topology(
            system,
            ring_edges=ring_edges,
            heavy_bonds=heavy_bonds,
            atoms=geometry.atoms,
            bond_orders=constitutional_bond_orders or {},
        )
        best, best_topological = _select_lcb26_ring_transfer(
            target,
            records,
            geometry=geometry,
            bond_orders=constitutional_bond_orders,
        )
        if best is None:
            audit.append(
                _ring_transfer_fallback_record(
                    target,
                    bond_orders=constitutional_bond_orders,
                    rejected_candidate=best_topological,
                )
            )
            continue
        audit.append(
            _apply_ring_transfer_candidate(
                best,
                target,
                coordinates,
                heavy_bonds=heavy_bonds,
                heavy_count=heavy_count,
            )
        )
        frozen.update(target.atoms)
    return (
        MolecularGeometry(
            atoms=geometry.atoms,
            coordinates_angstrom=coordinates,
            comment=geometry.comment,
            charge=geometry.charge,
            multiplicity=geometry.multiplicity,
            source_format=geometry.source_format,
            metadata=geometry.metadata,
        ),
        tuple(sorted(frozen)),
        {
            "status": (
                "PASS_WITH_EXPLICIT_FALLBACK"
                if any(item.get("fallback_used", False) for item in audit)
                else "PASS"
            ),
            "fallback_used": any(item.get("fallback_used", False) for item in audit),
            "systems": audit,
        },
    )


def _ring_interface_signatures(
    system: set[int],
    bonds: set[tuple[int, int]],
    symbols: Sequence[str],
    orders: dict[tuple[int, int], float],
) -> dict[int, tuple[tuple[str, str], ...]]:
    """Return deterministic one-bond heavy-atom signatures for a ring system."""

    signatures = {}
    for atom in system:
        labels = []
        for edge in bonds:
            if atom not in edge:
                continue
            neighbor = edge[1] if edge[0] == atom else edge[0]
            if neighbor in system:
                continue
            labels.append(
                (str(symbols[neighbor]), _bond_order_class(orders.get(edge, 1.0)))
            )
        signatures[atom] = tuple(sorted(labels))
    return signatures


def _ring_interface_score(
    mapping: dict[int, int],
    *,
    target_signatures: dict[int, tuple[tuple[str, str], ...]],
    donor_signatures: dict[int, tuple[tuple[str, str], ...]],
) -> tuple[int, int]:
    """Rank ring isomorphisms by their directly attached heavy-atom context.

    Exact ring topology alone leaves symmetry-equivalent atom permutations.
    The complete-molecule MCS previously broke those ties by extending into
    substituents.  Comparing the same one-bond interface explicitly preserves
    that chemical criterion while keeping the graph search local to the ring.
    """

    matches = 0
    mismatches = 0
    for target_atom, donor_atom in mapping.items():
        target = Counter(target_signatures.get(target_atom, ()))
        donor = Counter(donor_signatures.get(donor_atom, ()))
        common = sum((target & donor).values())
        matches += common
        mismatches += sum(target.values()) + sum(donor.values()) - 2 * common
    return matches, mismatches


def _bond_order_class(value: float) -> str:
    order = float(value)
    if order < 1.25:
        return "SINGLE"
    if order < 1.75:
        return "AROMATIC_OR_PARTIAL_DOUBLE"
    return "MULTIPLE"


def _ring_unsaturation_class(density: float) -> str:
    value = float(density)
    if value < 0.08:
        return "SATURATED"
    if value < 0.28:
        return "LOCALLY_UNSATURATED"
    return "DELOCALIZED_OR_MULTIPLY_UNSATURATED"


def _bridge_edges(count: int, edges: set[tuple[int, int]]) -> set[tuple[int, int]]:
    adjacency = {index: set() for index in range(count)}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    discovery = [-1] * count
    low = [0] * count
    parent = [-1] * count
    bridges = set()
    clock = 0

    def visit(atom):
        nonlocal clock
        discovery[atom] = low[atom] = clock
        clock += 1
        for neighbor in adjacency[atom]:
            if discovery[neighbor] < 0:
                parent[neighbor] = atom
                visit(neighbor)
                low[atom] = min(low[atom], low[neighbor])
                if low[neighbor] > discovery[atom]:
                    bridges.add(tuple(sorted((atom, neighbor))))
            elif neighbor != parent[atom]:
                low[atom] = min(low[atom], discovery[neighbor])

    for atom in range(count):
        if discovery[atom] < 0:
            visit(atom)
    return bridges


def _edge_components(edges):
    adjacency = {}
    for left, right in edges:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    result = []
    remaining = set(adjacency)
    while remaining:
        seed = min(remaining)
        stack = [seed]
        component = set()
        while stack:
            atom = stack.pop()
            if atom not in remaining:
                continue
            remaining.remove(atom)
            component.add(atom)
            stack.extend(adjacency[atom] & remaining)
        result.append(frozenset(component))
    return tuple(result)


def _kabsch_place(source, target):
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _singular, right = np.linalg.svd(covariance)
    rotation = left @ right
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right
    return target_center + (source - source_center) @ rotation


def _complete_constitution_coordinates(
    heavy_coordinates,
    template_geometry,
    constitutional_graph,
):
    heavy_count = len(constitutional_graph.atoms)
    completed = complete_graph_hydrogens(
        constitutional_graph,
        MolecularGeometry(
            atoms=template_geometry.atoms[:heavy_count],
            coordinates_angstrom=np.asarray(heavy_coordinates, dtype=float),
            comment=template_geometry.comment,
            charge=template_geometry.charge,
            multiplicity=template_geometry.multiplicity,
            source_format=template_geometry.source_format,
            metadata=template_geometry.metadata,
        ),
    )
    return completed.geometry.coordinates_angstrom


def _constitutional_topology_gate(
    coordinates,
    numbers,
    constitutional_bonds,
    *,
    allow_geometric_spurious: bool = False,
    constitutional_atom_count=None,
):
    if constitutional_bonds is None:
        return {
            "valid": True,
            "status": "NOT_APPLICABLE",
            "missing_bonds": [],
            "spurious_bonds": [],
        }
    expected = {tuple(sorted(map(int, pair))) for pair in constitutional_bonds}
    heavy_count = (
        int(constitutional_atom_count)
        if constitutional_atom_count is not None
        else 1 + max((atom for pair in expected for atom in pair), default=-1)
    )
    if heavy_count < 1 or heavy_count > len(coordinates):
        return {
            "valid": False,
            "status": "FAIL",
            "missing_bonds": [list(pair) for pair in sorted(expected)],
            "spurious_bonds": [],
            "reason": "INVALID_CONSTITUTIONAL_ATOM_COUNT",
        }
    # This gate owns only the immutable heavy-atom constitution supplied by
    # SWITCH. Hydrogens are regenerated after the weighted-B projection and
    # validated separately by the final ORACLE analysis.
    _continuous, graph, _rings, _synthons, _aromaticity = build_topology_objects(
        np.asarray(coordinates, dtype=float)[:heavy_count], numbers[:heavy_count]
    )
    realized = {tuple(sorted((int(left), int(right)))) for left, right in graph.bonds}
    missing = sorted(expected - realized)
    spurious = sorted(realized - expected)
    valid = not missing and (allow_geometric_spurious or not spurious)
    return {
        "valid": valid,
        "status": "PASS" if valid else "FAIL",
        "missing_bonds": [list(pair) for pair in missing],
        "spurious_bonds": [list(pair) for pair in spurious],
        "spurious_bonds_policy": (
            "DIAGNOSTIC_ONLY_SWITCH_CONSTITUTION_AUTHORITATIVE"
            if allow_geometric_spurious
            else "REJECT"
        ),
    }


def _transport_single_anchor_components(
    coordinates, reference, ring_atoms, heavy_bonds, heavy_count
):
    """Rigidly translate every one-anchor substituent with its moved ring anchor."""
    count = len(coordinates)
    bonds = set(heavy_bonds)
    for atom in range(heavy_count, count):
        distances = np.linalg.norm(reference[:heavy_count] - reference[atom], axis=1)
        parent = int(np.argmin(distances))
        bonds.add(tuple(sorted((parent, atom))))
    adjacency = {index: set() for index in range(count)}
    for left, right in bonds:
        adjacency[left].add(right)
        adjacency[right].add(left)
    remaining = set(range(count)) - set(ring_atoms)
    while remaining:
        seed = min(remaining)
        stack = [seed]
        component = set()
        while stack:
            atom = stack.pop()
            if atom not in remaining:
                continue
            remaining.remove(atom)
            component.add(atom)
            stack.extend(adjacency[atom] & remaining)
        anchors = {
            neighbor for atom in component for neighbor in adjacency[atom] if neighbor in ring_atoms
        }
        if len(anchors) != 1:
            continue
        anchor = next(iter(anchors))
        shift = coordinates[anchor] - reference[anchor]
        indices = np.asarray(sorted(component), dtype=int)
        coordinates[indices] = reference[indices] + shift


def _precondition_large_angle_targets(
    coordinates,
    bonds,
    angles,
    targets,
    target_indices,
    ring_edges,
    *,
    threshold_degrees=20.0,
):
    """Move a non-ring branch near a distant LCB26 angle before Wilson closure."""

    xyz = np.asarray(coordinates, dtype=float).copy()
    adjacency: dict[int, set[int]] = {}
    for left, right in bonds:
        adjacency.setdefault(int(left), set()).add(int(right))
        adjacency.setdefault(int(right), set()).add(int(left))
    ring = {tuple(sorted(map(int, edge))) for edge in ring_edges}
    offset = len(bonds)
    changed = 0
    for local, (left, center, right) in enumerate(angles):
        index = offset + local
        if index not in target_indices:
            continue
        incident_ring_edges = sum(
            tuple(sorted((int(endpoint), int(center)))) in ring for endpoint in (left, right)
        )
        if incident_ring_edges != 1:
            continue
        current = _angle(xyz[left], xyz[center], xyz[right])
        desired = float(targets[index])
        if abs(math.degrees(desired - current)) < float(threshold_degrees):
            continue
        candidate_sides = []
        for endpoint in (left, right):
            edge = tuple(sorted((int(endpoint), int(center))))
            if edge in ring:
                continue
            component = _component_after_edge_cut(adjacency, int(endpoint), edge)
            if int(center) not in component:
                candidate_sides.append((len(component), int(endpoint), component))
        if not candidate_sides:
            continue
        _, movable, component = min(candidate_sides, key=lambda item: item[0])
        fixed = int(right) if movable == int(left) else int(left)
        moving_vector = xyz[movable] - xyz[center]
        fixed_vector = xyz[fixed] - xyz[center]
        axis = np.cross(moving_vector, fixed_vector)
        norm = float(np.linalg.norm(axis))
        if norm <= 1.0e-10:
            reference = np.asarray((1.0, 0.0, 0.0))
            if abs(float(np.dot(reference, moving_vector))) > 0.9 * float(
                np.linalg.norm(moving_vector)
            ):
                reference = np.asarray((0.0, 1.0, 0.0))
            axis = np.cross(moving_vector, reference)
            norm = float(np.linalg.norm(axis))
        if norm <= 1.0e-12:
            continue
        axis /= norm
        magnitude = abs(desired - current)
        original = xyz[np.asarray(sorted(component), dtype=int)].copy()
        selected_coordinates = None
        selected_error = float("inf")
        atom_indices = np.asarray(sorted(component), dtype=int)
        for signed in (-magnitude, magnitude):
            trial = xyz.copy()
            vectors = original - xyz[center]
            rotated = (
                vectors * math.cos(signed)
                + np.cross(axis, vectors) * math.sin(signed)
                + np.outer(vectors @ axis, axis) * (1.0 - math.cos(signed))
            )
            trial[atom_indices] = xyz[center] + rotated
            error = abs(_angle(trial[left], trial[center], trial[right]) - desired)
            if error < selected_error:
                selected_error = error
                selected_coordinates = trial[atom_indices].copy()
        if selected_coordinates is not None:
            xyz[atom_indices] = selected_coordinates
            changed += 1
    return xyz, changed


def _component_after_edge_cut(adjacency, seed, blocked_edge):
    component = {int(seed)}
    queue = [int(seed)]
    for atom in queue:
        for neighbor in adjacency.get(atom, set()):
            if tuple(sorted((int(atom), int(neighbor)))) == tuple(blocked_edge):
                continue
            if int(neighbor) not in component:
                component.add(int(neighbor))
                queue.append(int(neighbor))
    return component


def _restore_preserved_dihedrals(
    coordinates,
    bonds,
    preserved_dihedrals,
    reference_coordinates,
):
    """Restore caller-declared proper torsions without changing ring interiors."""

    from matrix_chem.primitive_coordinates import dihedral

    requested = tuple(tuple(int(atom) for atom in item) for item in preserved_dihedrals)
    if not requested:
        return np.asarray(coordinates, dtype=float), 0
    xyz = np.asarray(coordinates, dtype=float).copy()
    reference = np.asarray(reference_coordinates, dtype=float)
    adjacency: dict[int, set[int]] = {}
    for left, right in bonds:
        adjacency.setdefault(int(left), set()).add(int(right))
        adjacency.setdefault(int(right), set()).add(int(left))
    restored = set()
    for _ in range(6):
        for atoms in requested:
            left, center_left, center_right, right = atoms
            desired = float(dihedral(*atoms, reference))
            current = float(dihedral(*atoms, xyz))
            delta = _periodic_difference(desired - current)
            if abs(delta) <= 1.0e-10:
                restored.add(atoms)
                continue
            origin = xyz[center_left].copy()
            axis = xyz[center_right] - origin
            norm = float(np.linalg.norm(axis))
            if norm <= 1.0e-12:
                continue
            axis /= norm
            central_edge = tuple(sorted((center_left, center_right)))
            component = _component_after_edge_cut(adjacency, right, central_edge)
            if center_left in component or left in component:
                upstream_edge = tuple(sorted((left, center_left)))
                component = _component_after_edge_cut(adjacency, left, upstream_edge)
                if center_left in component or center_right in component or right in component:
                    continue
                delta = -delta
            indices = np.asarray(sorted(component), dtype=int)
            vectors = xyz[indices] - origin
            xyz[indices] = origin + (
                vectors * math.cos(delta)
                + np.cross(axis, vectors) * math.sin(delta)
                + np.outer(vectors @ axis, axis) * (1.0 - math.cos(delta))
            )
            restored.add(atoms)
        if all(
            abs(
                _periodic_difference(
                    float(dihedral(*atoms, reference)) - float(dihedral(*atoms, xyz))
                )
            )
            <= 1.0e-8
            for atoms in requested
        ):
            break
    return xyz, len(restored)


def _close_internal_coordinates(
    coords,
    bonds,
    angles,
    dihedrals,
    target,
    max_iterations,
    tolerance,
    *,
    protected_ring_indices=(),
    target_indices=None,
):
    primitives = [Primitive("bond", tuple(pair)) for pair in bonds]
    primitives.extend(Primitive("angle", tuple(triple)) for triple in angles)
    primitives.extend(Primitive("dihedral", tuple(quadruple)) for quadruple in dihedrals)
    values = np.asarray(target, dtype=float)
    valence_count = len(bonds) + len(angles)
    # Bonds and angles define the transferred local geometry.  Exocyclic
    # torsions belong to the subsequent ZAFF-fast conformational stage and
    # therefore enter the Wilson metric as deformation penalties, not as an
    # incompatible redundant set of exact targets.  Only protected
    # endocyclic torsions remain explicit targets.
    selected = (
        set(range(valence_count))
        if target_indices is None
        else {int(index) for index in target_indices}
    )
    selected.update(int(index) for index in protected_ring_indices if int(index) >= valence_count)
    targets = {index: float(values[index]) for index in sorted(selected)}
    primitive_weights = {int(index): RING_PRIMITIVE_WEIGHT for index in protected_ring_indices}
    result = backtransform_primitive_targets(
        primitives,
        np.asarray(coords, dtype=float),
        targets,
        primitive_weights=primitive_weights,
        deformation_weights={"bond": 1000.0, "angle": 100.0, "dihedral": 100.0},
        tolerance=tolerance,
        max_iterations=max_iterations,
        maximum_cartesian_step=0.15,
        allow_least_squares_projection=True,
        objective_tolerance=1.0e-9,
    )
    return result.coordinates_angstrom, result.converged, result.iterations, result.maximum_residual


def _apply_cv_only(source_xyzin: Path, output_xyzin: Path) -> np.ndarray:
    geometry = read_xyz(source_xyzin)
    contract = read_primitive_contract(source_xyzin)
    numbers = tuple(_atomic_number(symbol) for symbol in geometry.atoms)
    plan = build_accuracy_ladder_plan(
        contract.primitives,
        numbers,
        valence_level=ValenceLevel.PL2,
        include_core_valence=True,
        coordinates_angstrom=geometry.coordinates_angstrom,
        include_bl1_conjugation=False,
        include_pl1_hydrogen_bonds=False,
    )
    result = apply_accuracy_ladder_plan(plan, contract.primitives, geometry.coordinates_angstrom)
    if not result.converged:
        raise InitialStructureError("CV-only back-transformation did not converge")
    corrected = np.asarray(result.coordinates_angstrom, dtype=float)
    with TemporaryDirectory(prefix="oracle-cv-output-") as scratch:
        corrected_xyz = Path(scratch) / "cv.xyz"
        write_xyz(
            corrected_xyz, geometry.atoms, corrected, comment="MATRIX CV-only initial structure"
        )
        analyze_structure(corrected_xyz, output_xyzin, source_kind="xyz")
    return corrected


__all__ = [
    "INITIAL_STRUCTURE_SCHEMA",
    "InitialStructureError",
    "InitialStructurePreparation",
    "prepare_initial_structure",
    "weighted_l1_internal_closure",
]
