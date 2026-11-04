"""Primitive-coordinate generation for SMITH/SONIC definitions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path
import re

import numpy as np

from matrix_chem import linear_bend_reference_atom

from .contracts import GICForgeContractError
from .fallback_ledger import make_fallback_event
from .analytic_salc import (
    cyclic_out_of_plane_atom_orders,
    cyclic_out_of_plane_coefficients,
    cyclic_out_of_plane_source_terms,
)
from .coordinate_registry import CoordinateSignature
from .generators import generate_stretch_coordinates
from .models import FallbackEvent, FrozenGIC, GICPointGroupOperation, GICPrimitive
from .numerics import (
    _analytic_b_row,
    _angle_value,
    _fragment_frame_anchor_atoms,
    _fragment_frame_rank,
    _fragment_linear_anchor_atom,
)
from .policy import (
    AROMATIC_LOCAL_MODEL_DIAGNOSTIC,
    LINEAR_ANGLE_DEGREES,
    PRIMITIVE_FAMILY_ORDER,
    RANK_TOLERANCE,
    SONIC_CONSTRUCTION_POLICY,
    XH_STRETCH_POLICY_SYMMETRIZE,
    primitive_prefix,
)
from .semantic import (
    USER_PROVENANCE,
    SemanticContract,
    SemanticCoordinate,
    semantic_signature_for_primitive_like,
)


@dataclass(frozen=True)
class _CoordinateFragmentRecord:
    """Internal fragment view after replacing haptic spokes by one center."""

    identifier: str
    atoms: tuple[int, ...]


def _primitive_candidates(
    bonds: tuple[tuple[int, int], ...],
    *,
    rings: tuple[tuple[int, tuple[int, ...]], ...] = (),
    coords: np.ndarray,
    natoms: int,
    atom_symbols: tuple[str, ...] = (),
    xh_stretch_policy: str = XH_STRETCH_POLICY_SYMMETRIZE,
    local_xh_bonds: tuple[tuple[int, int], ...] = (),
    local_xh_classes: tuple[str, ...] = (),
    improper_dihedrals: bool = False,
    fragment_records: tuple[object, ...] = (),
    body_prescriptions: tuple[object, ...] = (),
    fragment_contacts: tuple[tuple[int, int, str], ...] = (),
    interaction_centers: object | None = None,
    pseudo_bonds: tuple[tuple[int, int], ...] = (),
    pseudo_bond_kinds: tuple[str, ...] = (),
    pseudobond_out_of_plane_support: bool = False,
    reactive_support_edges: tuple[tuple[int, int], ...] = (),
    bond_orders: dict[tuple[int, int], float] | None = None,
    aromatic_atoms: frozenset[int] = frozenset(),
    ring_puckering_model: str = "triangular_flap",
    include_fragment_pose_candidates: bool = True,
) -> tuple[GICPrimitive, ...]:
    coordinate_bonds, haptic_fragment_records = _haptic_coordinate_bonds_and_fragments(
        bonds,
        interaction_centers=interaction_centers,
        atom_symbols=atom_symbols,
        natoms=natoms,
    )
    adjacency = _adjacency(coordinate_bonds, natoms=natoms)
    rings = _complete_single_missing_fullerene_face(
        coordinate_bonds,
        rings=rings,
        adjacency=adjacency,
        atom_symbols=atom_symbols,
        natoms=natoms,
    )
    pseudo_bond_set = {tuple(sorted(pair)) for pair in pseudo_bonds}
    kind_by_pseudo_bond = {
        tuple(sorted(pair)): str(kind).strip().upper()
        for pair, kind in zip(pseudo_bonds, pseudo_bond_kinds, strict=False)
    }
    pseudo_cycles = _pseudo_cycles_from_pseudo_bonds(
        coordinate_bonds,
        pseudo_bonds=tuple(pseudo_bond_set),
        natoms=natoms,
    )
    pseudo_cycle_angle_set = _cycle_angle_triplet_set(pseudo_cycles)
    counters: dict[str, int] = {family: 0 for family in PRIMITIVE_FAMILY_ORDER}
    candidates = list(
        _radial_and_pose_candidates(
            coordinate_bonds,
            pseudo_bond_set=pseudo_bond_set,
            kind_by_pseudo_bond=kind_by_pseudo_bond,
            haptic_fragment_records=haptic_fragment_records,
            fragment_records=fragment_records,
            body_prescriptions=body_prescriptions,
            fragment_contacts=fragment_contacts,
            interaction_centers=interaction_centers,
            coords=coords,
            counters=counters,
            atom_symbols=atom_symbols,
            xh_stretch_policy=xh_stretch_policy,
            local_xh_bonds=local_xh_bonds,
            local_xh_classes=local_xh_classes,
            pseudobond_out_of_plane_support=pseudobond_out_of_plane_support,
            include_fragment_pose_candidates=include_fragment_pose_candidates,
            reactive_support_edges=reactive_support_edges,
        )
    )
    fullerene_cage = _is_fullerene_cage_topology(
        coordinate_bonds,
        rings=rings,
        adjacency=adjacency,
        atom_symbols=atom_symbols,
        natoms=natoms,
    )
    if fullerene_cage:
        # One stretch per cage edge plus n-3 intrinsic diagonals for every
        # face triangulates the closed polyhedral surface.  Euler's relation
        # gives exactly 3V-6 homogeneous, non-periodic coordinates.
        for _ring_index, ring_atoms in rings:
            for left, right in _balanced_face_diagonals(ring_atoms):
                candidates.append(
                    _make_primitive(
                        "FULLERENE_FACE_DIAGONAL",
                        "R",
                        (left, right),
                        counters,
                        refs=("RING:" + "-".join(str(atom) for atom in ring_atoms),),
                    )
                )
        return tuple(candidates)
    angle_candidates, linear_angle_keys, linear_angles = _angle_candidates(
        adjacency,
        coords=coords,
        natoms=natoms,
        rings=rings,
        pseudo_cycle_angle_set=pseudo_cycle_angle_set,
        counters=counters,
    )
    candidates.extend(angle_candidates)
    candidates.extend(
        _torsion_and_ring_candidates(
            coordinate_bonds,
            adjacency=adjacency,
            pseudo_bond_set=pseudo_bond_set,
            reactive_support_edges=set(reactive_support_edges),
            linear_angle_keys=linear_angle_keys,
            linear_angles=linear_angles,
            rings=rings,
            counters=counters,
            bond_orders=bond_orders or {},
            coords=coords,
            aromatic_atoms=aromatic_atoms,
            ring_puckering_model=ring_puckering_model,
        )
    )
    candidates.extend(
        _out_of_plane_candidates(
            adjacency,
            coords=coords,
            natoms=natoms,
            rings=rings,
            counters=counters,
            atom_symbols=atom_symbols,
            bond_orders=bond_orders or {},
            improper_dihedrals=improper_dihedrals,
        )
    )
    # ORACLE owns the chart status of every primitive.  Only proper torsions
    # across exocyclic, single-bond-like edges are wrapped; ring torsions,
    # improper dihedrals, and torsions across multiple bonds remain principal
    # (SMITH may later choose Morse or Gaussian for their SONIC combinations).
    ring_edges = {
        tuple(sorted((ring_atoms[index], ring_atoms[(index + 1) % len(ring_atoms)])))
        for _ring_index, ring_atoms in rings
        if len(ring_atoms) >= 3
        for index in range(len(ring_atoms))
    }
    periodic_bond_order_threshold = 1.5
    annotated: list[GICPrimitive] = []
    for primitive in candidates:
        periodic = False
        if (
            primitive.function == "D"
            and primitive.family in {"TORSION", "PSEUDO_CYCLE_TORSION"}
            and len(primitive.atoms) == 4
        ):
            central = tuple(sorted((primitive.atoms[1], primitive.atoms[2])))
            periodic = (
                central not in ring_edges
                and central in coordinate_bonds
                and float((bond_orders or {}).get(central, 0.0))
                < periodic_bond_order_threshold
            )
        annotated.append(replace(primitive, chart="PERIODIC" if periodic else "PRINCIPAL"))
    return tuple(annotated)


def _is_fullerene_cage_topology(
    bonds: tuple[tuple[int, int], ...],
    *,
    rings: tuple[tuple[int, tuple[int, ...]], ...],
    adjacency: dict[int, set[int]],
    atom_symbols: tuple[str, ...],
    natoms: int,
) -> bool:
    """Recognize a closed carbon fullerene from its face-resolved graph."""

    if natoms < 20 or len(atom_symbols) != natoms:
        return False
    if any(str(symbol).strip().upper() != "C" for symbol in atom_symbols):
        return False
    edge_set = {tuple(sorted((int(left), int(right)))) for left, right in bonds}
    if len(edge_set) != len(bonds) or any(len(adjacency[atom]) != 3 for atom in adjacency):
        return False
    face_cycles = tuple(tuple(int(atom) for atom in ring) for _index, ring in rings)
    if not face_cycles or any(len(face) not in {5, 6} for face in face_cycles):
        return False
    if natoms - len(edge_set) + len(face_cycles) != 2:
        return False
    edge_face_counts = {edge: 0 for edge in edge_set}
    for face in face_cycles:
        if len(set(face)) != len(face):
            return False
        for left, right in zip(face, face[1:] + face[:1]):
            edge = tuple(sorted((left, right)))
            if edge not in edge_face_counts:
                return False
            edge_face_counts[edge] += 1
    if any(count != 2 for count in edge_face_counts.values()):
        return False
    coordinate_count = len(edge_set) + sum(len(face) - 3 for face in face_cycles)
    return coordinate_count == 3 * natoms - 6


def _balanced_face_diagonals(ring: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    """Return a deterministic balanced triangulation of one polygonal face."""

    ncycle = len(ring)
    if ncycle <= 3:
        return ()
    quotient, remainder = divmod(ncycle, 3)
    gaps = tuple(quotient + (1 if index < remainder else 0) for index in range(3))
    anchors = (0, gaps[0], gaps[0] + gaps[1])
    diagonals: list[tuple[int, int]] = []
    for arc_index in range(3):
        start = anchors[arc_index]
        end = anchors[(arc_index + 1) % 3]
        interior: list[int] = []
        index = (start + 1) % ncycle
        while index != end:
            interior.append(index)
            index = (index + 1) % ncycle
        if not interior:
            continue
        diagonals.append(tuple(sorted((ring[start], ring[end]))))
        for position in range(len(interior) - 1, 0, -1):
            diagonals.append(tuple(sorted((ring[start], ring[interior[position]]))))
    if len(diagonals) != ncycle - 3 or len(set(diagonals)) != len(diagonals):
        raise RuntimeError("balanced face triangulation did not produce n-3 unique diagonals")
    return tuple(diagonals)


def _complete_single_missing_fullerene_face(
    bonds: tuple[tuple[int, int], ...],
    *,
    rings: tuple[tuple[int, tuple[int, ...]], ...],
    adjacency: dict[int, set[int]],
    atom_symbols: tuple[str, ...],
    natoms: int,
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Close one omitted fullerene face from its singly incident boundary.

    ORACLE's smallest-ring traversal can omit the final face of a planar cage
    because that face is not independent in the cycle basis.  Completion is
    accepted only when the graph is trivalent carbon, exactly one Euler face
    is absent, and the undercovered edges form one simple pentagon or hexagon.
    """

    if natoms < 20 or len(atom_symbols) != natoms:
        return rings
    if any(str(symbol).strip().upper() != "C" for symbol in atom_symbols):
        return rings
    if any(len(adjacency[atom]) != 3 for atom in adjacency):
        return rings
    edge_set = {tuple(sorted((int(left), int(right)))) for left, right in bonds}
    expected_faces = len(edge_set) - natoms + 2
    if len(rings) != expected_faces - 1:
        return rings
    incidence = {edge: 0 for edge in edge_set}
    for _index, face in rings:
        if len(face) not in {5, 6} or len(set(face)) != len(face):
            return rings
        for left, right in zip(face, face[1:] + face[:1]):
            edge = tuple(sorted((int(left), int(right))))
            if edge not in incidence:
                return rings
            incidence[edge] += 1
    boundary_edges = tuple(edge for edge, count in incidence.items() if count == 1)
    if any(count not in {1, 2} for count in incidence.values()):
        return rings
    boundary: dict[int, set[int]] = {}
    for left, right in boundary_edges:
        boundary.setdefault(left, set()).add(right)
        boundary.setdefault(right, set()).add(left)
    if len(boundary) not in {5, 6} or any(len(neighbors) != 2 for neighbors in boundary.values()):
        return rings
    start = min(boundary)
    cycle = [start]
    previous: int | None = None
    current = start
    while True:
        choices = sorted(boundary[current] - ({previous} if previous is not None else set()))
        if not choices:
            return rings
        following = choices[0]
        if following == start:
            break
        if following in cycle or len(cycle) >= len(boundary):
            return rings
        cycle.append(following)
        previous, current = current, following
    if len(cycle) != len(boundary):
        return rings
    next_index = max((index for index, _face in rings), default=0) + 1
    return rings + ((next_index, tuple(cycle)),)


def _radial_and_pose_candidates(
    coordinate_bonds: tuple[tuple[int, int], ...],
    *,
    pseudo_bond_set: set[tuple[int, int]],
    kind_by_pseudo_bond: dict[tuple[int, int], str],
    haptic_fragment_records: tuple[_CoordinateFragmentRecord, ...],
    fragment_records: tuple[object, ...],
    body_prescriptions: tuple[object, ...],
    fragment_contacts: tuple[tuple[int, int, str], ...],
    interaction_centers: object | None,
    coords: np.ndarray,
    counters: dict[str, int],
    atom_symbols: tuple[str, ...],
    xh_stretch_policy: str,
    local_xh_bonds: tuple[tuple[int, int], ...],
    local_xh_classes: tuple[str, ...],
    pseudobond_out_of_plane_support: bool,
    include_fragment_pose_candidates: bool,
    reactive_support_edges: tuple[tuple[int, int], ...] = (),
) -> tuple[GICPrimitive, ...]:
    candidates: list[GICPrimitive] = []
    stretches = generate_stretch_coordinates(
        tuple(pair for pair in coordinate_bonds if tuple(sorted(pair)) not in pseudo_bond_set),
        atom_symbols=atom_symbols,
        xh_stretch_policy=xh_stretch_policy,
        local_xh_bonds=local_xh_bonds,
        local_xh_classes=local_xh_classes,
    )
    for stretch in stretches:
        candidates.append(
            _make_primitive(stretch.family, stretch.function, stretch.atoms, counters)
        )
    candidates.extend(
        _pseudo_bond_distance_primitives(
            tuple(
                (pair, kind_by_pseudo_bond.get(pair, "INTERFRAGMENT_CLOSEST"))
                for pair in sorted(pseudo_bond_set)
            ),
            counters=counters,
        )
    )

    candidates.extend(
        _interaction_center_primitive_candidates(interaction_centers, counters=counters)
    )
    if pseudo_bond_set and fragment_records:
        # Classified contacts and rigid-body pose primitives are complementary.
        # The former describe the local intermolecular contact; the latter span
        # the remaining relative translations and rotations of the fragments.
        # Dropping the pose block merely because a pseudobond exists leaves
        # multi-fragment clusters rank deficient (one contact cannot constrain
        # all six relative rigid-body degrees of freedom).
        candidates.extend(
            _pseudobond_contact_primitive_candidates(
                fragment_records,
                pseudo_bond_set=pseudo_bond_set,
                coordinate_bonds=coordinate_bonds,
                coords=coords,
                counters=counters,
                include_out_of_plane=pseudobond_out_of_plane_support,
                reactive_support_edges=reactive_support_edges,
            )
        )
        if include_fragment_pose_candidates:
            candidates.extend(
                _fragment_primitive_candidates(
                    fragment_records,
                    coords=coords,
                    counters=counters,
                    body_prescriptions=body_prescriptions,
                )
            )
    elif haptic_fragment_records:
        candidates.extend(
            _haptic_component_primitive_candidates(
                haptic_fragment_records,
                coords=coords,
                counters=counters,
            )
        )
    else:
        candidates.extend(
            _fragment_primitive_candidates(
                fragment_records,
                coords=coords,
                counters=counters,
                body_prescriptions=body_prescriptions,
            )
        )
    candidates.extend(
        _fragment_contact_distance_candidates(fragment_contacts, counters=counters)
    )
    return tuple(candidates)


def _pseudobond_contact_primitive_candidates(
    fragment_records: tuple[object, ...],
    *,
    pseudo_bond_set: set[tuple[int, int]],
    coordinate_bonds: tuple[tuple[int, int], ...],
    coords: np.ndarray,
    counters: dict[str, int],
    include_out_of_plane: bool = False,
    reactive_support_edges: tuple[tuple[int, int], ...] = (),
) -> tuple[GICPrimitive, ...]:
    """Complete classified contacts with local natural-coordinate supports.

    ORACLE has already decided that the OPEN classified contact is a
    pseudobond.  SMITH only supplies a deterministic rank-capable natural
    chart around that edge.  No rigid-body quaternion or exponential-map
    primitive is admitted in this branch.
    """

    records = tuple(sorted(fragment_records, key=lambda item: getattr(item, "identifier")))
    members_by_record = {
        id(record): frozenset(int(atom) for atom in getattr(record, "atoms"))
        for record in records
    }
    record_by_atom = {
        int(atom): record for record in records for atom in getattr(record, "atoms")
    }
    adjacency = _adjacency(coordinate_bonds, natoms=len(coords))
    supports: list[GICPrimitive] = []
    seen = set(pseudo_bond_set)
    for left, right in sorted(pseudo_bond_set):
        left_record = record_by_atom.get(left)
        right_record = record_by_atom.get(right)
        if left_record is None or right_record is None or left_record is right_record:
            raise GICForgeContractError(
                "ORACLE pseudobond does not join two frozen fragments"
            )
        left_anchors = _local_contact_frame_atoms(
            members_by_record[id(left_record)],
            endpoint=left,
            adjacency=adjacency,
            coords=coords,
        )
        right_anchors = _local_contact_frame_atoms(
            members_by_record[id(right_record)],
            endpoint=right,
            adjacency=adjacency,
            coords=coords,
        )
        angular_left, angular_right = left_anchors, right_anchors
        if reactive_support_edges:
            # Frozen fragments can be atomic/linear at a transfer saddle.
            # Borrow only local anchors reached through the declared reactive
            # graph, never an inferred covalent edge or a laboratory axis.
            support_graph = _adjacency(
                tuple(sorted(set(coordinate_bonds) | set(reactive_support_edges))),
                natoms=len(coords),
            )
            angular_left = _reactive_contact_frame_atoms(
                left, right, left_anchors, support_graph, coords,
            )
            angular_right = _reactive_contact_frame_atoms(
                right, left, right_anchors, support_graph, coords,
            )
        refs = (
            "PSEUDOBOND_CONTACT_SUPPORT",
            str(getattr(left_record, "identifier")),
            str(getattr(right_record, "identifier")),
        )
        for left_atom in left_anchors:
            for right_atom in right_anchors:
                pair = tuple(sorted((left_atom, right_atom)))
                if pair in seen:
                    continue
                seen.add(pair)
                supports.append(
                    _make_primitive(
                        "FRAG_DISTANCE",
                        "R",
                        pair,
                        counters,
                        refs=refs,
                    )
                )
        supports.extend(
            _pseudobond_contact_angular_candidates(
                left,
                right,
                left_anchors=angular_left,
                right_anchors=angular_right,
                coords=coords,
                counters=counters,
                refs=refs,
                additional_frame_torsions=(angular_left != left_anchors or angular_right != right_anchors),
            )
        )
        if include_out_of_plane:
            supports.extend(
                _pseudobond_contact_out_of_plane_candidates(
                    left,
                    right,
                    left_anchors=left_anchors,
                    right_anchors=right_anchors,
                    coords=coords,
                    counters=counters,
                    refs=refs,
                )
            )
    return tuple(supports)


def _pseudobond_contact_angular_candidates(
    left: int,
    right: int,
    *,
    left_anchors: tuple[int, ...],
    right_anchors: tuple[int, ...],
    coords: np.ndarray,
    counters: dict[str, int],
    refs: tuple[str, ...],
    additional_frame_torsions: bool = False,
) -> tuple[GICPrimitive, ...]:
    """Return local A/L/D supports spanning a pseudobond contact chart."""

    candidates: list[GICPrimitive] = []
    for outer, center, other in (
        *((atom, left, right) for atom in left_anchors if atom != left),
        *((left, right, atom) for atom in right_anchors if atom != right),
    ):
        angle_atoms = (outer, center, other)
        angle = float(np.degrees(_angle_value(coords, angle_atoms)))
        if _angle_is_collinear_degrees(angle):
            reference = linear_bend_reference_atom(
                tuple(atom - 1 for atom in angle_atoms),
                coords,
                candidate_indices=sorted({atom - 1 for atom in (*left_anchors, *right_anchors)}),
            )
            ref_atoms = () if reference is None else (reference + 1,)
            for mode in (-1, -2):
                candidates.append(
                    _make_primitive(
                        "PSEUDO_BOND_BEND",
                        "L",
                        angle_atoms,
                        counters,
                        mode=mode,
                        ref_atoms=ref_atoms,
                        refs=refs,
                    )
                )
        else:
            candidates.append(
                _make_primitive(
                    "PSEUDO_BOND_BEND",
                    "A",
                    angle_atoms,
                    counters,
                    refs=refs,
                )
            )

    left_outer = tuple(atom for atom in left_anchors if atom != left)
    right_outer = tuple(atom for atom in right_anchors if atom != right)
    torsions = {
        (left_atom, left, right, right_atom)
        for left_atom in left_outer
        for right_atom in right_outer
    }
    if len(left_outer) >= 2:
        torsions.add((left_outer[1], left_outer[0], left, right))
    if len(right_outer) >= 2:
        torsions.add((left, right, right_outer[0], right_outer[1]))
    # A linear contact axis has no regular contact-centered dihedral. Its
    # local side frame can still supply regular internal torsions with the
    # contact endpoints at the two ends, without an L or legacy U coordinate.
    if additional_frame_torsions:
        for outer in (left_outer, right_outer):
            torsions.update((left, a, b, right) for a in outer for b in outer if a != b)
    for atoms in sorted(torsions):
        if len(set(atoms)) != 4 or not _contact_torsion_is_regular(coords, atoms):
            continue
        candidates.append(
            _make_primitive(
                "PSEUDO_BOND_TORSION",
                "D",
                atoms,
                counters,
                refs=refs,
            )
        )
    return tuple(candidates)


def _contact_torsion_is_regular(
    coords: np.ndarray,
    atoms: tuple[int, int, int, int],
) -> bool:
    """Reject torsions whose defining adjacent angle is locally linear."""

    angles = tuple(
        float(np.degrees(_angle_value(coords, triplet)))
        for triplet in (atoms[:3], atoms[1:])
    )
    margin = 180.0 - LINEAR_ANGLE_DEGREES
    return all(margin < angle < LINEAR_ANGLE_DEGREES for angle in angles)


def _pseudobond_contact_out_of_plane_candidates(
    left: int,
    right: int,
    *,
    left_anchors: tuple[int, ...],
    right_anchors: tuple[int, ...],
    coords: np.ndarray,
    counters: dict[str, int],
    refs: tuple[str, ...],
) -> tuple[GICPrimitive, ...]:
    """Return finite U supports for a locally linear pseudobond frame.

    The contact endpoints and two anchors on either frozen fragment define a
    signed out-of-plane direction without using a linear bend.  All finite
    local gauges enter the candidate pool; the common exact-rank selector
    chooses only the minimum number needed by the chart.
    """

    candidates: list[GICPrimitive] = []
    for endpoint, opposite, anchors in (
        (left, right, left_anchors),
        (right, left, right_anchors),
    ):
        outer = tuple(atom for atom in anchors if atom != endpoint)
        for center in outer:
            for out in outer:
                if center == out:
                    continue
                atoms = (center, endpoint, opposite, out)
                probe = GICPrimitive(
                    identifier="PROBE",
                    name="PROBE",
                    family="OUT_OF_PLANE",
                    function="U",
                    atoms=atoms,
                    refs=refs,
                )
                try:
                    row = _analytic_b_row(probe, coords)
                except (ArithmeticError, FloatingPointError):
                    continue
                if np.any(~np.isfinite(row)) or float(np.linalg.norm(row)) <= RANK_TOLERANCE:
                    continue
                candidates.append(
                    _make_primitive(
                        "OUT_OF_PLANE",
                        "U",
                        atoms,
                        counters,
                        refs=refs,
                    )
                )
    return tuple(candidates)


from matrix_chem.contact_primitives import _local_contact_frame_atoms, _reactive_contact_frame_atoms


def _angle_candidates(
    adjacency: dict[int, set[int]],
    *,
    coords: np.ndarray,
    natoms: int,
    rings: tuple[tuple[int, tuple[int, ...]], ...],
    pseudo_cycle_angle_set: set[tuple[int, int, int]],
    counters: dict[str, int],
) -> tuple[tuple[GICPrimitive, ...], set[tuple[int, int, int]], tuple[tuple[int, int, int], ...]]:
    candidates: list[GICPrimitive] = []
    linear_angle_keys: set[tuple[int, int, int]] = set()
    linear_angles: list[tuple[int, int, int]] = []
    for center in range(1, natoms + 1):
        for i, k in combinations(sorted(adjacency[center]), 2):
            if (i, center, k) in pseudo_cycle_angle_set:
                continue
            angle = _angle_value(coords, (i, center, k))
            if _angle_is_collinear_degrees(float(np.degrees(angle))):
                linear_angle_keys.add(_angle_key((i, center, k)))
                linear_angles.append((i, center, k))
                reference = linear_bend_reference_atom(
                    (i - 1, center - 1, k - 1),
                    coords,
                )
                ref_atoms = (reference + 1,) if reference is not None else ()
                candidates.append(
                    _make_primitive(
                        "LINEAR_BEND",
                        "L",
                        (i, center, k),
                        counters,
                        mode=-1,
                        ref_atoms=ref_atoms,
                    )
                )
                candidates.append(
                    _make_primitive(
                        "LINEAR_BEND",
                        "L",
                        (i, center, k),
                        counters,
                        mode=-2,
                        ref_atoms=ref_atoms,
                    )
                )
            else:
                family = (
                    "CYCLIC_BEND"
                    if _ring_index_for_atoms((i, center, k), rings) is not None
                    else "BEND"
                )
                candidates.append(_make_primitive(family, "A", (i, center, k), counters))
    return tuple(candidates), linear_angle_keys, tuple(linear_angles)


def _angle_is_collinear_degrees(angle_degrees: float) -> bool:
    """Classify both parallel and antiparallel singular angle limits."""

    angle = float(angle_degrees)
    margin = 180.0 - float(LINEAR_ANGLE_DEGREES)
    return angle <= margin or angle >= float(LINEAR_ANGLE_DEGREES)


def _torsion_and_ring_candidates(
    coordinate_bonds: tuple[tuple[int, int], ...],
    *,
    adjacency: dict[int, set[int]],
    pseudo_bond_set: set[tuple[int, int]],
    reactive_support_edges: set[tuple[int, int]],
    linear_angle_keys: set[tuple[int, int, int]],
    linear_angles: tuple[tuple[int, int, int], ...],
    rings: tuple[tuple[int, tuple[int, ...]], ...],
    counters: dict[str, int],
    bond_orders: dict[tuple[int, int], float],
    coords: np.ndarray,
    aromatic_atoms: frozenset[int],
    ring_puckering_model: str,
) -> tuple[GICPrimitive, ...]:
    seen_torsions: set[tuple[int, int, int, int]] = set()
    torsion_candidate = tuple[str, tuple[int, int, int, int], tuple[str, ...]]
    butterfly_torsions: list[torsion_candidate] = []
    condensed_torsions: list[torsion_candidate] = []
    ordinary_torsions: list[torsion_candidate] = []
    for j, k in coordinate_bonds:
        # A closure/contact edge has no covalent bond order and therefore no
        # ordinary valence torsional curvature.  Fragment-pose torsions are
        # generated explicitly by their dedicated coordinate family.
        reactive = tuple(sorted((j, k))) in reactive_support_edges
        if tuple(sorted((j, k))) in pseudo_bond_set and not reactive:
            continue
        for i in sorted(adjacency[j] - {k}):
            for ell in sorted(adjacency[k] - {j}):
                torsion = (i, j, k, ell)
                # Three-membered covalent or interaction-closure paths can
                # return to the same endpoint.  D(i,j,k,i) is not a
                # dihedral and its analytic derivative is singular.
                if len(set(torsion)) != 4:
                    continue
                canonical = min(torsion, tuple(reversed(torsion)))
                if canonical in seen_torsions:
                    continue
                seen_torsions.add(canonical)
                if (
                    _angle_key((i, j, k)) in linear_angle_keys
                    or _angle_key((j, k, ell)) in linear_angle_keys
                ):
                    continue
                family = _torsion_family(canonical, rings)
                if family == "CYCLIC_TORSION":
                    continue
                if family == "BUTTERFLY":
                    butterfly_torsions.append((family, canonical, ()))
                elif family == "CONDENSED_RING_TORSION":
                    condensed_torsions.append((family, canonical, ()))
                else:
                    ordinary_torsions.append((family, canonical, ()))

    # A dihedral containing a linear triple has a singular derivative,
    # independently of whether either edge is covalent or a pseudobond.
    # Collapse the linear center and define the equivalent torsion across the
    # two endpoints, as in Gaussian DefRed's special linear-case construction.
    for left_endpoint, linear_center, right_endpoint in linear_angles:
        for left in sorted(adjacency[left_endpoint] - {linear_center}):
            for right in sorted(adjacency[right_endpoint] - {linear_center}):
                if len({left, left_endpoint, right_endpoint, right}) != 4:
                    continue
                torsion = (left, left_endpoint, right_endpoint, right)
                canonical = min(torsion, tuple(reversed(torsion)))
                if canonical in seen_torsions:
                    continue
                seen_torsions.add(canonical)
                ordinary_torsions.append(("TORSION", canonical, ()))

    candidates = [
        _make_primitive(family, "D", torsion, counters, refs=refs)
        for family, torsion, refs in butterfly_torsions
    ]
    candidates.extend(
        _ring_pucker_component_candidates(
            rings,
            counters=counters,
            bond_orders=bond_orders,
            coords=coords,
            aromatic_atoms=aromatic_atoms,
            ring_puckering_model=ring_puckering_model,
        )
    )
    for family, torsion, refs in condensed_torsions:
        candidates.append(_make_primitive(family, "D", torsion, counters, refs=refs))
    for family, torsion, refs in ordinary_torsions:
        function = "RPCK" if refs else "D"
        candidates.append(_make_primitive(family, function, torsion, counters, refs=refs))
    # These remain geometric torsions for symmetry/combinations. Their
    # frozen reactive provenance is distinct from a covalent bond assignment;
    # SH uses no covalent typing when the central bond order is absent.
    return tuple(replace(p, provenance="ORACLE_REACTIVE_SUPPORT")
                 if len(p.atoms) == 4 and tuple(sorted(p.atoms[1:3])) in reactive_support_edges
                 else p for p in candidates)


def _out_of_plane_candidates(
    adjacency: dict[int, set[int]],
    *,
    coords: np.ndarray,
    natoms: int,
    rings: tuple[tuple[int, tuple[int, ...]], ...],
    counters: dict[str, int],
    atom_symbols: tuple[str, ...],
    bond_orders: dict[tuple[int, int], float],
    improper_dihedrals: bool,
) -> tuple[GICPrimitive, ...]:
    candidates: list[GICPrimitive] = []
    cyclic_atoms = _cyclic_atom_set(rings)
    out_of_plane_family = "IMPROPER_DIHEDRAL" if improper_dihedrals else "OUT_OF_PLANE"
    out_of_plane_function = "IMPD" if improper_dihedrals else "U"
    for center in range(1, natoms + 1):
        neighbors = sorted(adjacency[center])
        if len(neighbors) < 3:
            continue
        for n1, n2, n3 in combinations(neighbors, 3):
            if {center, n1, n2, n3}.issubset(cyclic_atoms):
                continue
            atom_orders = _out_of_plane_atom_orders(
                center,
                (n1, n2, n3),
                adjacency=adjacency,
                atom_symbols=atom_symbols,
                coords=coords,
                bond_orders=bond_orders,
            )
            atom_orders = _regular_out_of_plane_atom_orders(atom_orders, coords=coords)
            if not atom_orders:
                continue
            candidates.append(
                _make_primitive(
                    out_of_plane_family,
                    out_of_plane_function,
                    atom_orders[0],
                    counters,
                )
            )

    return tuple(candidates)


def _regular_out_of_plane_atom_orders(
    orders: tuple[tuple[int, int, int, int], ...],
    *,
    coords: np.ndarray,
) -> tuple[tuple[int, int, int, int], ...]:
    """Remove U representations whose two plane vectors are collinear."""

    regular = []
    for atoms in orders:
        center, plane1, plane2, _out = (atom - 1 for atom in atoms)
        first = coords[plane1] - coords[center]
        second = coords[plane2] - coords[center]
        scale = float(np.linalg.norm(first) * np.linalg.norm(second))
        sine = float(np.linalg.norm(np.cross(first, second)) / max(scale, 1.0e-30))
        if scale > RANK_TOLERANCE and sine > RANK_TOLERANCE:
            regular.append(atoms)
    return tuple(regular)


def _freeze_balanced_out_of_plane_coordinates(
    primitives: tuple[GICPrimitive, ...],
    gics: tuple[FrozenGIC, ...],
    *,
    selected: tuple[GICPrimitive, ...],
    bonds: tuple[tuple[int, int], ...],
    coords: np.ndarray,
    atom_symbols: tuple[str, ...],
    bond_orders: dict[tuple[int, int], float],
) -> tuple[tuple[GICPrimitive, ...], tuple[FrozenGIC, ...]]:
    """Freeze heterogeneous XY3 inversion coordinates as an oriented mean.

    Primitive rank selection still sees one local inversion direction. The
    frozen GIC expands that direction into the regular cyclic Gaussian-U terms
    only when no ligand pair defines a clear local C2v-like domain. Collinear
    plane definitions are excluded before the fixed normalized combination is
    frozen, so no geometry-dependent switching occurs later in a path.
    """

    adjacency = _adjacency(bonds, natoms=len(atom_symbols))
    output_primitives = list(primitives)
    primitive_by_signature = {
        (primitive.family, primitive.function, primitive.atoms): primitive
        for primitive in output_primitives
    }
    next_identifier = 1 + max(
        (
            int(match.group(1))
            for primitive in output_primitives
            if (match := re.fullmatch(r"P(\d+)", primitive.identifier)) is not None
        ),
        default=0,
    )
    family_counts: dict[str, int] = {}
    for primitive in output_primitives:
        family_counts[primitive.family] = family_counts.get(primitive.family, 0) + 1

    output_gics: list[FrozenGIC] = []
    for gic, primitive in zip(gics, selected):
        if (
            primitive.function not in {"U", "IMPD"}
            or len(primitive.atoms) != 4
            or primitive.family == "RING_PUCKER_COMPONENT"
        ):
            output_gics.append(gic)
            continue
        center = primitive.atoms[0]
        orders = _out_of_plane_atom_orders(
            center,
            tuple(sorted(primitive.atoms[1:])),
            adjacency=adjacency,
            atom_symbols=atom_symbols,
            coords=coords,
            bond_orders=bond_orders,
        )
        orders = _regular_out_of_plane_atom_orders(orders, coords=coords)
        if not orders:
            raise GICForgeContractError(
                f"out-of-plane primitive {primitive.identifier} has no regular local frame"
            )
        if len(orders) == 1:
            output_gics.append(gic)
            continue
        coefficient = 1.0 / np.sqrt(float(len(orders)))
        terms: list[tuple[str, float]] = []
        for atoms in orders:
            signature = (primitive.family, primitive.function, atoms)
            component = primitive_by_signature.get(signature)
            if component is None:
                family_counts[primitive.family] += 1
                component = replace(
                    primitive,
                    identifier=f"P{next_identifier:03d}",
                    name=(
                        f"{primitive.name.rstrip('0123456789')}"
                        f"{family_counts[primitive.family]:04d}"
                    ),
                    atoms=atoms,
                )
                next_identifier += 1
                output_primitives.append(component)
                primitive_by_signature[signature] = component
            terms.append((component.identifier, coefficient))
        output_gics.append(
            replace(
                gic,
                primitive_id=terms[0][0],
                gaussian_expression="LINEAR_COMBINATION",
                coefficients=tuple(terms),
            )
        )
    return tuple(output_primitives), tuple(output_gics)


def _out_of_plane_atom_orders(
    center: int,
    substituents: tuple[int, int, int],
    *,
    adjacency: dict[int, set[int]],
    atom_symbols: tuple[str, ...],
    coords: np.ndarray,
    bond_orders: dict[tuple[int, int], float],
) -> tuple[tuple[int, int, int, int], ...]:
    """Return oriented ``U(center,plane1,plane2,out)`` atom orders.

    A local XY2Z domain must put the equivalent pair in the defining plane;
    choosing one member of that pair as ``out`` produces a Wilson row that is
    not covariant under the pair-exchange operation away from planarity.  A
    locally C3v-like domain has no privileged ligand.  For a fully
    heterogeneous domain, the three cyclic orders define the balanced mean
    coordinate used by the frozen GIC layer.
    """

    ordered = tuple(sorted(int(atom) for atom in substituents))
    classes = _substituent_equivalence_classes(
        center,
        ordered,
        adjacency=adjacency,
        atom_symbols=atom_symbols,
        bond_orders=bond_orders,
    )
    pair = next((group for group in classes if len(group) == 2), None)
    if pair is not None:
        unique = next(atom for atom in ordered if atom not in pair)
        first, second = sorted(pair)
        return ((center, first, second, unique),)
    if len(classes) == 1:
        first, second, third = ordered
        return (
            (center, first, second, third),
            (center, second, third, first),
            (center, third, first, second),
        )

    c2v_order = _c2v_like_out_of_plane_order(center, ordered, coords=coords)
    if c2v_order is not None:
        return (c2v_order,)
    first, second, third = ordered
    return (
        (center, first, second, third),
        (center, second, third, first),
        (center, third, first, second),
    )


def _substituent_equivalence_classes(
    center: int,
    substituents: tuple[int, int, int],
    *,
    adjacency: dict[int, set[int]],
    atom_symbols: tuple[str, ...],
    bond_orders: dict[tuple[int, int], float],
) -> tuple[tuple[int, ...], ...]:
    """Classify ligand roots by an atom-number-invariant rooted graph label."""

    active = tuple(atom for atom in sorted(adjacency) if atom != center)

    def symbol(atom: int) -> str:
        return atom_symbols[atom - 1].strip().upper() if len(atom_symbols) >= atom else "X"

    def order(left: int, right: int) -> float:
        return float(bond_orders.get(tuple(sorted((left, right))), 1.0))

    labels: dict[int, object] = {
        atom: (
            symbol(atom),
            round(order(center, atom), 6) if atom in adjacency[center] else 0.0,
            len(adjacency[atom] - {center}),
        )
        for atom in active
    }
    for _iteration in range(len(active)):
        raw = {
            atom: (
                labels[atom],
                tuple(
                    sorted(
                        (round(order(atom, other), 6), labels[other])
                        for other in adjacency[atom]
                        if other != center
                    )
                ),
            )
            for atom in active
        }
        unique = {value: index for index, value in enumerate(sorted(set(raw.values()), key=repr))}
        refined = {atom: unique[value] for atom, value in raw.items()}
        if all(refined[atom] == labels[atom] for atom in active):
            break
        labels = refined
    grouped: dict[object, list[int]] = {}
    for atom in substituents:
        grouped.setdefault(labels[atom], []).append(atom)
    return tuple(tuple(sorted(group)) for _label, group in sorted(grouped.items(), key=repr))


def _c2v_like_out_of_plane_order(
    center: int,
    substituents: tuple[int, int, int],
    *,
    coords: np.ndarray,
    pair_margin: float = 0.45,
    absolute_tolerance: float = 2.0e-2,
) -> tuple[int, int, int, int] | None:
    """Recognize a clear geometric XY2Z pattern among three distinct branches.

    Pair scores combine relative bond-length mismatch and the mismatch of the
    two angles made with the remaining ligand.  A pair is accepted only when
    it is both locally close and clearly separated from the next candidate;
    ambiguous domains are represented by the cyclic mean instead.
    """

    vectors = {
        atom: np.asarray(coords[atom - 1] - coords[center - 1], dtype=float)
        for atom in substituents
    }
    lengths = {atom: float(np.linalg.norm(vector)) for atom, vector in vectors.items()}
    if any(length <= 1.0e-12 for length in lengths.values()):
        return None

    def angle(first: int, second: int) -> float:
        cosine = float(np.dot(vectors[first], vectors[second]) / (lengths[first] * lengths[second]))
        return float(np.arccos(np.clip(cosine, -1.0, 1.0)))

    scored: list[tuple[float, tuple[int, int], int]] = []
    for first, second in combinations(substituents, 2):
        unique = next(atom for atom in substituents if atom not in {first, second})
        length_scale = 0.5 * (lengths[first] + lengths[second])
        length_mismatch = abs(lengths[first] - lengths[second]) / length_scale
        angle_mismatch = abs(angle(first, unique) - angle(second, unique))
        scored.append((float(np.hypot(length_mismatch, angle_mismatch)), (first, second), unique))
    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    best, runner_up = scored[0], scored[1]
    if best[0] > absolute_tolerance or best[0] > pair_margin * max(runner_up[0], 1.0e-12):
        return None
    first, second = sorted(best[1])
    return (center, first, second, best[2])


def _torsion_family(
    atoms: tuple[int, int, int, int],
    rings: tuple[tuple[int, tuple[int, ...]], ...],
) -> str:
    if _butterfly_torsion(atoms, rings):
        return "BUTTERFLY"
    if _ring_index_for_atoms(atoms, rings) is not None:
        return "CYCLIC_TORSION"
    component = _ring_component_for_atoms(atoms, rings)
    if component is not None and len(component) > 1:
        return "CONDENSED_RING_TORSION"
    return "TORSION"


def _pseudo_cycles_from_pseudo_bonds(
    bonds: tuple[tuple[int, int], ...],
    *,
    pseudo_bonds: tuple[tuple[int, int], ...],
    natoms: int,
) -> tuple[tuple[int, ...], ...]:
    pseudo_set = {tuple(sorted(pair)) for pair in pseudo_bonds}
    if not pseudo_set:
        return ()
    graph_bonds = tuple(tuple(sorted(pair)) for pair in bonds)
    cycles: list[tuple[int, ...]] = []
    seen: set[frozenset[int]] = set()
    for left, right in sorted(pseudo_set):
        path_bonds = tuple(pair for pair in graph_bonds if pair != (left, right))
        path = _shortest_graph_path(left, right, _adjacency(path_bonds, natoms=natoms))
        if len(path) < 4:
            continue
        key = frozenset(path)
        if key in seen:
            continue
        seen.add(key)
        cycles.append(_canonical_cycle_order(path))
    return tuple(cycles)


def _canonical_cycle_order(cycle: tuple[int, ...]) -> tuple[int, ...]:
    if not cycle:
        return ()
    variants: list[tuple[int, ...]] = []
    ncycle = len(cycle)
    for base in (cycle, tuple(reversed(cycle))):
        for shift in range(ncycle):
            variants.append(base[shift:] + base[:shift])
    return min(variants)


def _cycle_angle_triplet_set(cycles: tuple[tuple[int, ...], ...]) -> set[tuple[int, int, int]]:
    triplets: set[tuple[int, int, int]] = set()
    for cycle in cycles:
        ncycle = len(cycle)
        for index, center in enumerate(cycle):
            left = cycle[(index - 1) % ncycle]
            right = cycle[(index + 1) % ncycle]
            triplets.add((left, center, right))
            triplets.add((right, center, left))
    return triplets


def _shortest_graph_path(
    start: int,
    stop: int,
    adjacency: dict[int, set[int]],
) -> tuple[int, ...]:
    queue: list[tuple[int, ...]] = [(int(start),)]
    seen = {int(start)}
    while queue:
        path = queue.pop(0)
        node = path[-1]
        if node == stop:
            return path
        for neighbor in sorted(adjacency.get(node, ())):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            queue.append(path + (neighbor,))
    return ()










def _ring_pucker_component_candidates(
    rings: tuple[tuple[int, tuple[int, ...]], ...],
    *,
    counters: dict[str, int],
    bond_orders: dict[tuple[int, int], float],
    coords: np.ndarray,
    aromatic_atoms: frozenset[int] = frozenset(),
    ring_puckering_model: str = "triangular_flap",
) -> tuple[GICPrimitive, ...]:
    candidates: list[GICPrimitive] = []
    for _ring_index, ring_atoms in rings:
        effective_model = SONIC_CONSTRUCTION_POLICY.discrete_minimum_ring_model(
            aromatic=bool(ring_atoms) and set(ring_atoms).issubset(aromatic_atoms)
        )
        if effective_model == "local_out_of_plane":
            candidates.extend(
                _ring_out_of_plane_candidates(
                    ring_atoms,
                    counters=counters,
                    coords=coords,
                )
            )
            continue
        for terms in _ring_pucker_component_terms(ring_atoms, bond_orders=bond_orders):
            candidates.append(
                _make_primitive(
                    "RING_PUCKER_COMPONENT",
                    "RPCK",
                    tuple(ring_atoms),
                    counters,
                    refs=tuple(
                        _encode_ring_pucker_term(coefficient, atoms) for coefficient, atoms in terms
                    ),
                )
            )
    return tuple(candidates)


def _ring_out_of_plane_candidates(
    ring_atoms: tuple[int, ...],
    *,
    counters: dict[str, int],
    coords: np.ndarray,
) -> tuple[GICPrimitive, ...]:
    """Build the ``n-3`` analytic SALCs of one native local-U ring block."""

    ncycle = len(ring_atoms)
    if ncycle <= 3:
        return ()
    source_terms = cyclic_out_of_plane_source_terms(ring_atoms)

    source_rows = np.vstack(
        [
            sum(
                coefficient
                * _analytic_b_row(
                    GICPrimitive(
                        identifier=f"RING_U_{source_index + 1}_{term_index + 1}",
                        name=f"RingU{source_index + 1:04d}_{term_index + 1}",
                        family="RING_PUCKER_COMPONENT",
                        function="U",
                        atoms=atoms,
                    ),
                    coords,
                )
                for term_index, (coefficient, atoms) in enumerate(terms)
            )
            for source_index, terms in enumerate(source_terms)
        ]
    )
    norms = np.linalg.norm(source_rows, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= RANK_TOLERANCE):
        raise GICForgeContractError(
            f"local-U ring block {ring_atoms} contains a singular primitive"
        )

    coefficients = cyclic_out_of_plane_coefficients(ncycle)
    mode_rows = coefficients.T @ source_rows
    singular = np.linalg.svd(mode_rows, compute_uv=False)
    expected_rank = ncycle - 3
    rank = int(np.sum(singular > RANK_TOLERANCE))
    if rank != expected_rank:
        raise GICForgeContractError(
            f"local-U SALC block {ring_atoms} has rank {rank}, expected {expected_rank}"
        )
    condition = float(singular[0] / singular[expected_rank - 1])
    if not np.isfinite(condition) or condition > 1.0e10:
        raise GICForgeContractError(
            f"local-U SALC block {ring_atoms} is ill-conditioned ({condition:.6g})"
        )
    candidates = tuple(
        _make_primitive(
            "RING_PUCKER_COMPONENT",
            "RPU",
            tuple(ring_atoms),
            counters,
            refs=tuple(
                _encode_ring_pucker_term(
                    float(mode_coefficient) * float(source_coefficient), atoms
                )
                for mode_coefficient, source in zip(coefficients[:, mode], source_terms)
                if abs(float(mode_coefficient)) > 1.0e-12
                for source_coefficient, atoms in source
            ),
        )
        for mode in range(expected_rank)
    )
    # Reserve the complete paired-U source block.  The Fourier reduction keeps
    # n-3 modes, while native materialization needs both orientations at every
    # vertex; reserving their slots prevents collisions with later families.
    native_source_count = len(cyclic_out_of_plane_atom_orders(ring_atoms))
    counters["RING_PUCKER_COMPONENT"] += native_source_count - expected_rank
    return candidates


def _ring_pucker_component_terms(
    ring_atoms: tuple[int, ...],
    *,
    bond_orders: dict[tuple[int, int], float] | None = None,
) -> tuple[tuple[tuple[float, tuple[int, int, int, int]], ...], ...]:
    """Return ORACLE RPck linear combinations for one ordered ring."""
    ncyc = len(ring_atoms)
    if ncyc <= 3:
        return ()
    vnorm = float(np.sqrt(2.0 / float(ncyc)))
    vnorm1 = float(np.sqrt(1.0 / float(ncyc)))
    istart = ncyc
    components: list[tuple[tuple[float, tuple[int, int, int, int]], ...]] = []
    for ivar in range(1, ncyc - 2):
        even = ivar == 2 * (ivar // 2)
        terms: list[tuple[float, tuple[int, int, int, int]]] = []
        for iterm in range(1, ncyc + 1):
            iang1 = _cyclic_index(iterm + istart - 1, ncyc)
            iang2 = _cyclic_index(iterm + istart, ncyc)
            iang3 = _cyclic_index(iterm + istart + 1, ncyc)
            iang4 = _cyclic_index(iterm + istart + 2, ncyc)
            harmonic = (ivar + 1) // 2 + 1
            value = np.pi * (2.0 * float(harmonic) * float(iterm - 1)) / float(ncyc)
            if even:
                coefficient = vnorm * float(np.sin(value))
            elif ivar < ncyc - 3:
                coefficient = vnorm * float(np.cos(value))
            else:
                coefficient = vnorm1 * float(np.cos(float(iterm - 1) * np.pi))
            if abs(coefficient) <= 1.0e-14:
                coefficient = 0.0
            terms.append(
                (
                    coefficient
                    * _ring_dihedral_flexibilities_one_based(
                        ring_atoms,
                        bond_orders=bond_orders or {},
                    )[iterm - 1],
                    (
                        ring_atoms[iang1],
                        ring_atoms[iang2],
                        ring_atoms[iang3],
                        ring_atoms[iang4],
                    ),
                )
            )
        components.append(_normalize_ring_pucker_terms(terms))
    return tuple(components)


def _ring_dihedral_flexibilities_one_based(
    ring_atoms: tuple[int, ...],
    *,
    bond_orders: dict[tuple[int, int], float],
    contrast_tolerance: float = 0.50,
) -> tuple[float, ...]:
    orders = _ring_dihedral_bond_orders_one_based(ring_atoms, bond_orders=bond_orders)
    finite = [order for order in orders if order is not None and order > 1.0e-12]
    if len(finite) != len(orders):
        return tuple(1.0 for _order in orders)
    reference = min(float(order) for order in finite)
    maximum = max(float(order) for order in finite)
    if reference <= 0.0 or maximum / reference <= 1.0 + float(contrast_tolerance):
        return tuple(1.0 for _order in orders)
    return tuple(float(np.sqrt(reference / float(order))) for order in finite)


def _ring_dihedral_bond_orders_one_based(
    ring_atoms: tuple[int, ...],
    *,
    bond_orders: dict[tuple[int, int], float],
) -> tuple[float | None, ...]:
    ncyc = len(ring_atoms)
    orders: list[float | None] = []
    for iterm in range(1, ncyc + 1):
        iang2 = _cyclic_index(iterm + ncyc, ncyc)
        iang3 = _cyclic_index(iterm + ncyc + 1, ncyc)
        orders.append(bond_orders.get(tuple(sorted((ring_atoms[iang2], ring_atoms[iang3])))))
    return tuple(orders)


def _ring_puckering_diagnostics(
    rings: tuple[tuple[int, tuple[int, ...]], ...],
    *,
    bond_orders: dict[tuple[int, int], float],
    bond_order_components: dict[tuple[int, int], tuple[float, float, float]] | None = None,
    ring_puckering_model: str = "triangular_flap",
    aromatic_atoms: frozenset[int] = frozenset(),
) -> tuple[str, ...]:
    lines: list[str] = []
    for ring_index, ring_atoms in rings:
        orders = _ring_dihedral_bond_orders_one_based(ring_atoms, bond_orders=bond_orders)
        flex = _ring_dihedral_flexibilities_one_based(ring_atoms, bond_orders=bond_orders)
        bonds = []
        for iterm in range(1, len(ring_atoms) + 1):
            iang2 = _cyclic_index(iterm + len(ring_atoms), len(ring_atoms))
            iang3 = _cyclic_index(iterm + len(ring_atoms) + 1, len(ring_atoms))
            bonds.append(f"{ring_atoms[iang2]}-{ring_atoms[iang3]}")
        order_text = ",".join("NA" if order is None else f"{float(order):.8g}" for order in orders)
        flex_text = ",".join(f"{value:.8g}" for value in flex)
        components = bond_order_components or {}
        component_values = [
            components.get(tuple(sorted(tuple(int(value) for value in bond.split("-")))))
            for bond in bonds
        ]
        pi_text = ",".join(
            "NA" if value is None else f"{value[1]:.8g}" for value in component_values
        )
        pi_pi_text = ",".join(
            "NA" if value is None else f"{value[2]:.8g}" for value in component_values
        )
        prefix = (
            f"RING {ring_index} ATOMS={','.join(str(atom) for atom in ring_atoms)} "
            f"CENTRAL_BONDS={','.join(bonds)} BOND_ORDERS={order_text} "
            f"PI_INDICES={pi_text} PI_PI_INDICES={pi_pi_text}"
        )
        aromatic = bool(ring_atoms) and set(ring_atoms).issubset(aromatic_atoms)
        effective_model = SONIC_CONSTRUCTION_POLICY.discrete_minimum_ring_model(
            aromatic=aromatic
        )
        if effective_model == "triangular_flap":
            lines.append(f"{prefix} MODEL=TRIANGULAR_FLAP WEIGHTING=NONE")
        elif effective_model == "charm":
            lines.append(f"{prefix} MODEL=CHARM PRIMITIVE=GAUSSIAN_H WEIGHTING=CYCLIC_BALANCED")
        elif effective_model == "local_out_of_plane":
            model = AROMATIC_LOCAL_MODEL_DIAGNOSTIC if aromatic else "MODEL=LOCAL_OUT_OF_PLANE"
            lines.append(f"{prefix} {model} PRIMITIVE=GAUSSIAN_U FLEX={flex_text}")
        else:
            lines.append(
                f"{prefix} MODEL=ENDOCYCLIC_DIHEDRAL STATUS=LEGACY_DEPRECATED FLEX={flex_text}"
            )
    return tuple(lines)


def _normalize_ring_pucker_terms(
    terms: list[tuple[float, tuple[int, int, int, int]]],
) -> tuple[tuple[float, tuple[int, int, int, int]], ...]:
    norm = float(np.sqrt(sum(float(coefficient) ** 2 for coefficient, _atoms in terms)))
    if norm <= 1.0e-14:
        return tuple(terms)
    return tuple((float(coefficient) / norm, atoms) for coefficient, atoms in terms)


def _cyclic_index(index_1based: int, ncyc: int) -> int:
    while index_1based > ncyc:
        index_1based -= ncyc
    while index_1based <= 0:
        index_1based += ncyc
    return index_1based - 1


def _encode_ring_pucker_term(
    coefficient: float,
    atoms: tuple[int, int, int, int],
) -> str:
    atom_text = "-".join(str(atom) for atom in atoms)
    return f"{float(coefficient):.17g}:{atom_text}"




def _ring_index_for_atoms(
    atoms: tuple[int, ...],
    rings: tuple[tuple[int, tuple[int, ...]], ...],
) -> int | None:
    atom_set = set(atoms)
    best_index: int | None = None
    best_size: int | None = None
    for ring_index, ring_atoms in rings:
        ring_set = set(ring_atoms)
        if not atom_set.issubset(ring_set):
            continue
        ring_size = len(ring_atoms)
        if (
            best_size is None
            or ring_size < best_size
            or (ring_size == best_size and ring_index < (best_index or ring_index))
        ):
            best_index = ring_index
            best_size = ring_size
    return best_index


def _ring_component_for_atoms(
    atoms: tuple[int, ...],
    rings: tuple[tuple[int, tuple[int, ...]], ...],
) -> tuple[int, ...] | None:
    atom_set = set(atoms)
    for component in _ring_components(rings):
        component_atoms: set[int] = set()
        for ring_index in component:
            component_atoms.update(dict(rings)[ring_index])
        if atom_set.issubset(component_atoms):
            return component
    return None


def _ring_components(
    rings: tuple[tuple[int, tuple[int, ...]], ...],
) -> tuple[tuple[int, ...], ...]:
    if not rings:
        return ()
    ring_ids = tuple(ring_index for ring_index, _atoms in rings)
    bond_to_rings = _ring_bond_to_rings(rings)
    neighbors: dict[int, set[int]] = {ring_index: set() for ring_index in ring_ids}
    for ring_indices in bond_to_rings.values():
        if len(ring_indices) < 2:
            continue
        for left, right in combinations(ring_indices, 2):
            neighbors[left].add(right)
            neighbors[right].add(left)
    components: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for ring_index in ring_ids:
        if ring_index in seen:
            continue
        stack = [ring_index]
        seen.add(ring_index)
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(neighbors[current]):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                stack.append(neighbor)
        components.append(tuple(sorted(component)))
    return tuple(components)


def _butterfly_torsion(
    atoms: tuple[int, int, int, int],
    rings: tuple[tuple[int, tuple[int, ...]], ...],
) -> bool:
    if not rings:
        return False
    i, j, k, ell = atoms
    central_bond = tuple(sorted((j, k)))
    ring_by_index = {ring_index: set(ring_atoms) for ring_index, ring_atoms in rings}
    sharing_rings = _ring_bond_to_rings(rings).get(central_bond, ())
    if len(sharing_rings) < 2:
        return False
    for left_index, right_index in combinations(sharing_rings, 2):
        left_only = ring_by_index[left_index] - ring_by_index[right_index]
        right_only = ring_by_index[right_index] - ring_by_index[left_index]
        if (i in left_only and ell in right_only) or (i in right_only and ell in left_only):
            return True
    return False


def _ring_bond_to_rings(
    rings: tuple[tuple[int, tuple[int, ...]], ...],
) -> dict[tuple[int, int], tuple[int, ...]]:
    mapping: dict[tuple[int, int], list[int]] = {}
    for ring_index, ring_atoms in rings:
        if len(ring_atoms) < 2:
            continue
        for left, right in zip(ring_atoms, ring_atoms[1:] + ring_atoms[:1]):
            bond = tuple(sorted((left, right)))
            mapping.setdefault(bond, []).append(ring_index)
    return {bond: tuple(indices) for bond, indices in mapping.items()}


def _make_primitive(
    family: str,
    function: str,
    atoms: tuple[int, ...],
    counters: dict[str, int],
    *,
    mode: int = 0,
    ref_atoms: tuple[int, ...] = (),
    refs: tuple[str, ...] = (),
    frame_atoms: tuple[int, ...] = (),
    ref_frame_atoms: tuple[int, ...] = (),
) -> GICPrimitive:
    counters[family] += 1
    prefix = primitive_prefix(family)
    serial = sum(counters.values())
    return GICPrimitive(
        identifier=f"P{serial:03d}",
        name=f"{prefix}{counters[family]:04d}",
        family=family,
        function=function,
        atoms=tuple(int(atom) for atom in atoms),
        mode=int(mode),
        ref_atoms=tuple(int(atom) for atom in ref_atoms),
        refs=tuple(str(ref) for ref in refs),
        frame_atoms=tuple(int(atom) for atom in frame_atoms),
        ref_frame_atoms=tuple(int(atom) for atom in ref_frame_atoms),
    )


def _apply_semantic_contract_to_candidates(
    candidates: tuple[GICPrimitive, ...],
    contract: SemanticContract,
    *,
    symmetry_operations: tuple[GICPointGroupOperation, ...],
) -> tuple[
    tuple[GICPrimitive, ...],
    SemanticContract,
    tuple[str, ...],
    tuple[FallbackEvent, ...],
]:
    diagnostics = list(contract.diagnostics)
    if not contract.protect_coordinates:
        return candidates, contract, tuple(diagnostics), ()

    generated: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    user_primitives: list[GICPrimitive] = []
    primitive_substitutions: dict[CoordinateSignature, SemanticCoordinate] = {}
    auto_semantic_requests = [
        coordinate
        for coordinate in contract.protect_coordinates
        if coordinate.semantic_type not in _SEMANTIC_PRIMITIVE_PROTECT_FUNCTIONS
    ]

    for serial, coordinate in enumerate(contract.protect_coordinates, start=1):
        primitive = _semantic_primitive_from_coordinate(coordinate, serial=serial)
        if primitive is None:
            diagnostics.append(
                f"SEMANTIC_STORED_ONLY ID={coordinate.identifier} TYPE={coordinate.semantic_type}"
            )
            continue
        primitive_substitutions[coordinate.signature] = coordinate
        user_primitives.append(primitive)
        generated[coordinate.identifier] = ((primitive.identifier,), ())
        diagnostics.append(
            f"SEMANTIC_USER_PROTECT ID={coordinate.identifier} PRIMITIVE={primitive.identifier}"
        )

    filtered_candidates: list[GICPrimitive] = []
    matched_auto_requests: set[str] = set()
    for primitive in candidates:
        signature = _semantic_signature_for_gic_primitive(primitive)
        substitute = primitive_substitutions.get(signature)
        if substitute is not None:
            diagnostics.append(
                "SEMANTIC_SUBSTITUTION "
                f"USER={substitute.identifier} REPLACES={primitive.identifier}"
            )
            continue
        matched_coordinate = _semantic_auto_request_for_primitive(
            primitive,
            auto_semantic_requests,
        )
        if matched_coordinate is not None:
            primitive = replace(
                primitive,
                provenance=USER_PROVENANCE,
                semantic_id=matched_coordinate.identifier,
                semantic_type=matched_coordinate.semantic_type,
            )
            matched_auto_requests.add(matched_coordinate.identifier)
            generated[matched_coordinate.identifier] = (
                generated.get(matched_coordinate.identifier, ((), ()))[0] + (primitive.identifier,),
                (),
            )
            diagnostics.append(
                "SEMANTIC_AUTO_PROTECT "
                f"ID={matched_coordinate.identifier} PRIMITIVE={primitive.identifier}"
            )
        filtered_candidates.append(primitive)

    for coordinate in auto_semantic_requests:
        if coordinate.identifier not in matched_auto_requests:
            diagnostics.append(
                "SEMANTIC_PENDING_GENERATOR "
                f"ID={coordinate.identifier} TYPE={coordinate.semantic_type}"
            )

    partial_diagnostics, fallback_events = _semantic_partial_orbit_records(
        contract.protect_coordinates,
        symmetry_operations=symmetry_operations,
    )
    diagnostics.extend(partial_diagnostics)
    return (
        tuple(user_primitives + filtered_candidates),
        contract.with_coordinate_generation(generated),
        tuple(diagnostics),
        fallback_events,
    )


_SEMANTIC_PRIMITIVE_PROTECT_FUNCTIONS = {
    "DISTANCE": ("STRETCH", "R"),
    "ANGLE": ("BEND", "A"),
    "TORSION": ("TORSION", "D"),
    "OUT_OF_PLANE": ("OUT_OF_PLANE", "U"),
}


def _semantic_primitive_from_coordinate(
    coordinate: SemanticCoordinate,
    *,
    serial: int,
) -> GICPrimitive | None:
    mapping = _SEMANTIC_PRIMITIVE_PROTECT_FUNCTIONS.get(coordinate.semantic_type)
    if mapping is None:
        return None
    family, function = mapping
    try:
        atoms = tuple(int(argument) for argument in coordinate.arguments)
    except ValueError as exc:
        raise GICForgeContractError(
            f"semantic {coordinate.semantic_type} needs atom-index arguments"
        ) from exc
    return GICPrimitive(
        identifier=f"PU{serial:03d}",
        name=_semantic_user_coordinate_name(coordinate.identifier, serial=serial),
        family=family,
        function=function,
        atoms=atoms,
        provenance=USER_PROVENANCE,
        semantic_id=coordinate.identifier,
        semantic_type=coordinate.semantic_type,
    )


def _semantic_user_coordinate_name(identifier: str, *, serial: int) -> str:
    text = re.sub(r"[^A-Za-z0-9]", "", identifier)
    if not text:
        text = "Coord"
    return f"Usr{text[:18]}{serial:03d}"


def _semantic_signature_for_gic_primitive(primitive: GICPrimitive) -> CoordinateSignature:
    return semantic_signature_for_primitive_like(
        primitive.function,
        primitive.atoms,
        mode=primitive.mode,
        ref_atoms=primitive.ref_atoms,
    )


def _semantic_auto_request_for_primitive(
    primitive: GICPrimitive,
    coordinates: list[SemanticCoordinate],
) -> SemanticCoordinate | None:
    for coordinate in coordinates:
        if coordinate.semantic_type == "RING_PUCKERING":
            if primitive.family == "RING_PUCKER_COMPONENT" and set(primitive.atoms) == {
                int(atom) for atom in coordinate.arguments
            }:
                return coordinate
        elif coordinate.semantic_type == "BUTTERFLY":
            if primitive.family == "BUTTERFLY" and set(primitive.atoms) == {
                int(atom) for atom in coordinate.arguments
            }:
                return coordinate
    return None


def _semantic_partial_orbit_records(
    coordinates: tuple[SemanticCoordinate, ...],
    *,
    symmetry_operations: tuple[GICPointGroupOperation, ...],
) -> tuple[tuple[str, ...], tuple[FallbackEvent, ...]]:
    if len(symmetry_operations) <= 1:
        return (), ()
    user_signatures = {coordinate.signature for coordinate in coordinates}
    diagnostics: list[str] = []
    events: list[FallbackEvent] = []
    for coordinate in coordinates:
        mapping = _SEMANTIC_PRIMITIVE_PROTECT_FUNCTIONS.get(coordinate.semantic_type)
        if mapping is None:
            continue
        _family, function = mapping
        atoms = tuple(int(argument) for argument in coordinate.arguments)
        orbit: set[CoordinateSignature] = set()
        for operation in symmetry_operations:
            mapped = tuple(operation.permutation[atom - 1] for atom in atoms)
            orbit.add(semantic_signature_for_primitive_like(function, mapped))
        if len(orbit) > 1 and not orbit.issubset(user_signatures):
            source = (
                "SEMANTIC_PARTIAL_ORBIT "
                f"ID={coordinate.identifier} SIZE={len(orbit)} "
                "FALLBACK=LOCAL_SALC"
            )
            diagnostics.append(source)
            events.append(
                make_fallback_event(
                    stage="SMITH_SEMANTIC_PROTECTION",
                    algorithm_id="LOCAL_SALC",
                    trigger="USER_PROTECTION_DECLARES_PARTIAL_SYMMETRY_ORBIT",
                    domain=f"SEMANTIC:{coordinate.identifier}",
                    macrofamily=mapping[0],
                    source=source,
                )
            )
    return tuple(diagnostics), tuple(events)


def _fragment_primitive_candidates(
    fragment_records: tuple[object, ...],
    *,
    coords: np.ndarray,
    counters: dict[str, int],
    body_prescriptions: tuple[object, ...],
) -> tuple[GICPrimitive, ...]:
    if len(fragment_records) <= 1:
        return ()
    records = tuple(sorted(fragment_records, key=lambda item: getattr(item, "identifier")))
    bodies = {
        str(getattr(body, "body_id")): body
        for body in body_prescriptions
    }
    if set(bodies) != {str(getattr(record, "identifier")) for record in records}:
        raise GICForgeContractError(
            "ORACLE atlas body membership does not match the frozen fragment contract"
        )
    reference_ids = tuple(
        body_id
        for body_id, body in bodies.items()
        if str(getattr(body, "pose_role")) == "REFERENCE"
    )
    if len(reference_ids) != 1:
        raise GICForgeContractError("ORACLE atlas must prescribe exactly one reference body")
    reference = next(
        record
        for record in records
        if str(getattr(record, "identifier")) == reference_ids[0]
    )
    for record in records:
        body = bodies[str(getattr(record, "identifier"))]
        if tuple(int(atom) for atom in getattr(record, "atoms")) != tuple(
            int(atom) for atom in getattr(body, "atoms")
        ):
            raise GICForgeContractError("ORACLE atlas body atom membership is stale")
    candidates: list[GICPrimitive] = []
    for record in records:
        if getattr(record, "identifier") == getattr(reference, "identifier"):
            continue
        candidates.extend(
            _body_pair_pose_candidates(
                record,
                reference,
                coords=coords,
                counters=counters,
            )
        )
    return tuple(candidates)


def _body_pair_pose_candidates(
    record: object,
    reference: object,
    *,
    coords: np.ndarray,
    counters: dict[str, int],
) -> tuple[GICPrimitive, ...]:
    atoms = tuple(getattr(record, "atoms"))
    ref_atoms = tuple(getattr(reference, "atoms"))
    refs = (getattr(record, "identifier"), getattr(reference, "identifier"))
    moving_frame_rank = _fragment_frame_rank(coords, atoms)
    reference_frame_rank = _fragment_frame_rank(coords, ref_atoms)
    ref_frame_atoms = (
        _fragment_frame_anchor_atoms(ref_atoms, coords=coords)
        if reference_frame_rank >= 2
        else ()
    )
    linear_reference_anchor = (
        (_fragment_linear_anchor_atom(ref_atoms, coords=coords),)
        if reference_frame_rank == 1
        else ref_frame_atoms
    )
    candidates = [
        _make_primitive(
            "FRAG_DISTANCE",
            "FC_DIST",
            atoms,
            counters,
            ref_atoms=ref_atoms,
            refs=refs,
        )
    ]
    translation_function = "FLIN_TRANS" if reference_frame_rank == 1 else "FTRANS"
    for axis in range(2 if reference_frame_rank == 1 else 3):
        candidates.append(
            _make_primitive(
                "FRAG_TRANSLATION",
                translation_function,
                atoms,
                counters,
                mode=axis,
                ref_atoms=ref_atoms,
                refs=refs,
                ref_frame_atoms=linear_reference_anchor,
            )
        )
    if moving_frame_rank == 1 and reference_frame_rank == 1:
        moving_anchor = (_fragment_linear_anchor_atom(atoms, coords=coords),)
        for axis in range(2):
            candidates.append(
                _make_primitive(
                    "FRAG_ORIENTATION",
                    "FAXIS",
                    atoms,
                    counters,
                    mode=axis,
                    ref_atoms=ref_atoms,
                    refs=refs,
                    frame_atoms=moving_anchor,
                    ref_frame_atoms=linear_reference_anchor,
                )
            )
    if moving_frame_rank >= 1 and reference_frame_rank >= 2:
        frame_atoms = (
            _fragment_frame_anchor_atoms(atoms, coords=coords)
            if moving_frame_rank >= 2
            else (_fragment_linear_anchor_atom(atoms, coords=coords),)
        )
        for axis in range(2 if moving_frame_rank == 1 else 3):
            candidates.append(
                _make_primitive(
                    "FRAG_ORIENTATION",
                    "FROT",
                    atoms,
                    counters,
                    mode=axis,
                    ref_atoms=ref_atoms,
                    refs=refs,
                    frame_atoms=frame_atoms,
                    ref_frame_atoms=ref_frame_atoms,
                )
            )
    return tuple(candidates)


def _haptic_component_primitive_candidates(
    component_records: tuple[_CoordinateFragmentRecord, ...],
    *,
    coords: np.ndarray,
    counters: dict[str, int],
) -> tuple[GICPrimitive, ...]:
    """Return a symmetry-closed pool built from existing fragment coordinates."""

    records = tuple(sorted(component_records, key=lambda item: item.identifier))
    candidates: list[GICPrimitive] = []
    for moving in records:
        moving_atoms = tuple(moving.atoms)
        for reference in records:
            if moving.identifier == reference.identifier:
                continue
            reference_atoms = tuple(reference.atoms)
            refs = (moving.identifier, reference.identifier)
            for axis in range(3):
                candidates.append(
                    _make_primitive(
                        "FRAG_TRANSLATION",
                        "FTRANS",
                        moving_atoms,
                        counters,
                        mode=axis,
                        ref_atoms=reference_atoms,
                        refs=refs,
                    )
                )
            if (
                _fragment_frame_rank(coords, moving_atoms) >= 2
                and _fragment_frame_rank(coords, reference_atoms) >= 2
            ):
                moving_frame = _fragment_frame_anchor_atoms(moving_atoms, coords=coords)
                reference_frame = _fragment_frame_anchor_atoms(reference_atoms, coords=coords)
                for axis in range(3):
                    candidates.append(
                        _make_primitive(
                            "FRAG_ORIENTATION",
                            "FROT",
                            moving_atoms,
                            counters,
                            mode=axis,
                            ref_atoms=reference_atoms,
                            refs=refs,
                            frame_atoms=moving_frame,
                            ref_frame_atoms=reference_frame,
                        )
                    )
    return tuple(candidates)


def _fragment_contact_distance_candidates(
    contacts: tuple[tuple[int, int, str], ...],
    *,
    counters: dict[str, int],
) -> tuple[GICPrimitive, ...]:
    """Materialize physical interfragment contacts without changing adjacency."""

    candidates: list[GICPrimitive] = []
    seen: set[tuple[int, int]] = set()
    for left, right, kind in contacts:
        pair = tuple(sorted((int(left), int(right))))
        if pair[0] == pair[1] or pair in seen:
            continue
        seen.add(pair)
        candidates.append(
            _make_primitive(
                "FRAG_DISTANCE",
                "R",
                pair,
                counters,
                refs=(str(kind).strip().upper(),),
            )
        )
    return tuple(candidates)


def _pseudo_bond_distance_primitives(
    contacts: tuple[tuple[tuple[int, int], str], ...],
    *,
    counters: dict[str, int],
) -> tuple[GICPrimitive, ...]:
    """Keep graph-completion contacts separate from covalent stretches."""
    return tuple(
        _make_primitive(
            "PSEUDO_BOND_DISTANCE",
            "R",
            tuple(sorted(int(atom) for atom in pair)),
            counters,
            refs=(str(source).strip().upper(),),
        )
        for pair, source in contacts
    )


def _normalize_pseudo_contact(contact) -> tuple[int, int, str]:
    """Normalize an atlas contact record without interpreting its semantics."""

    if len(contact) == 3:
        left, right, kind = contact
        return int(left), int(right), str(kind)
    if len(contact) == 2:
        pair, kind = contact
        left, right = pair
        return int(left), int(right), str(kind)
    raise ValueError(f"invalid pseudo-contact record: {contact!r}")


def _angle_key(atoms: tuple[int, int, int]) -> tuple[int, int, int]:
    left, center, right = (int(atom) for atom in atoms)
    ends = tuple(sorted((left, right)))
    return (ends[0], center, ends[1])


def _interaction_center_primitive_candidates(
    definition: object | None,
    *,
    counters: dict[str, int],
) -> tuple[GICPrimitive, ...]:
    if definition is None:
        return ()
    centers = {
        getattr(center, "identifier"): center for center in getattr(definition, "centers", ())
    }
    candidates: list[GICPrimitive] = []
    for interaction in getattr(definition, "interactions", ()):
        center = centers.get(getattr(interaction, "center_id"))
        if center is None:
            continue
        atom = int(getattr(interaction, "atom"))
        atoms = tuple(int(item) for item in getattr(center, "atoms"))
        candidates.append(
            _make_primitive(
                "CENTER_ATOM_DISTANCE",
                "CENTER_ATOM_DIST",
                atoms,
                counters,
                ref_atoms=(atom,),
                refs=(getattr(center, "identifier"), f"A{atom}"),
            )
        )
    return tuple(candidates)


from matrix_chem.haptic_primitives import _haptic_coordinate_bonds_and_fragments


def _fragment_records(path: Path) -> tuple[object, ...]:
    try:
        from matrix_fragments import read_fragment_records
    except ImportError:
        return ()
    return tuple(read_fragment_records(Path(path)))


def _fragment_index_by_atom(records: tuple[object, ...]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for index, record in enumerate(records):
        for atom in getattr(record, "atoms", ()):
            mapping[int(atom)] = index
    return mapping


def _interaction_center_definition(path: Path) -> object | None:
    try:
        from matrix_fragments import read_interaction_center_definition
    except ImportError:
        return None
    return read_interaction_center_definition(Path(path))








def _vibrational_rank(coords: np.ndarray) -> int:
    natoms = int(coords.shape[0])
    if natoms <= 1:
        return 0
    return max(0, 3 * natoms - (5 if _is_linear(coords) else 6))


def _is_linear(coords: np.ndarray) -> bool:
    if coords.shape[0] <= 2:
        return True
    centered = coords - np.mean(coords, axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    if singular_values.size < 2 or singular_values[0] <= RANK_TOLERANCE:
        return False
    return bool(singular_values[1] / singular_values[0] <= 1.0e-6)


def _adjacency(bonds: tuple[tuple[int, int], ...], *, natoms: int) -> dict[int, set[int]]:
    graph = {idx: set() for idx in range(1, natoms + 1)}
    for i, j in bonds:
        graph[i].add(j)
        graph[j].add(i)
    return graph


def _cyclic_atom_set(rings: tuple[tuple[int, tuple[int, ...]], ...]) -> set[int]:
    return {atom for _ring_index, atoms in rings for atom in atoms}
