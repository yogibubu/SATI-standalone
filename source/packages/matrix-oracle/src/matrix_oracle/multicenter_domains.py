"""ORACLE providers for primary shared-ligand multicenter domains."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Iterable, Sequence

import numpy as np

from matrix_chem import (
    MulticenterDomain,
    Primitive,
    PrimitiveCandidate,
    perceive_hydrogen_bonds,
    perceive_proton_transfer_bridges,
)
from matrix_chem.primitive_coordinates import sparse_grad_primitive_entries
from matrix_chem.topology.bonding_roles import (
    is_bridging_ligand,
    is_electronegative_lone_pair_donor,
    is_structural_center,
)
from matrix_chem.topology.periodic_properties import periodic_atomic_properties


MULTICENTER_PROVIDER_SCHEMA = "matrix.oracle.multicenter_domains.v1"
SHARED_PROTON_PROVIDER = "ORACLE_SHARED_PROTON_LONE_PAIR_CENTERS"
TRANSFERABLE_HYDROGEN_PROVIDER = "ORACLE_TRANSFERABLE_HYDROGEN_CENTERS"
STRUCTURAL_LIGAND_BRIDGE_PROVIDER = "ORACLE_STRUCTURAL_CENTER_LIGAND_BRIDGE"
BRIDGE_PLANE_PROVIDER = "ORACLE_MULTIBRIDGE_PLANE_COUPLING"
PROVIDER_VERSION = "1"
MAXIMUM_NORMALIZED_STRUCTURAL_BRIDGE_DISTANCE = 1.55
MINIMUM_STRUCTURAL_BRIDGE_ANGLE_DEGREES = 60.0
MAXIMUM_TRANSFERABLE_HYDROGEN_COVALENT_EXTENSION_ANGSTROM = 0.80


def perceive_multicenter_domains(
    atomic_numbers: Sequence[int],
    coordinates_angstrom: np.ndarray,
    bonded_pairs: Iterable[tuple[int, int]],
    *,
    explicit_multicenter_bridges: Iterable[tuple[int, int, int]] = (),
) -> tuple[tuple[MulticenterDomain, ...], tuple[PrimitiveCandidate, ...]]:
    """Perceive primary X--L--X' hyperedges and their analytic primitives.

    Input and explicit bridge indices are zero based.  The returned frozen
    contract is one based. Shared protons reuse MATRIX's established bridge
    kernel. H/halide bridges between structural centers use periodic bonding
    roles and one covalent-radius-normalized applicability rule; no element
    pair, center equivalence, or raw X--L distance equality is hard-coded.
    """

    numbers = tuple(int(value) for value in atomic_numbers)
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    if xyz.shape != (len(numbers), 3) or np.any(~np.isfinite(xyz)):
        raise ValueError("multicenter coordinates must be finite (natoms, 3)")
    bonds = {tuple(sorted((int(left), int(right)))) for left, right in bonded_pairs}
    if any(left == right or left < 0 or right >= len(numbers) for left, right in bonds):
        raise ValueError("multicenter primary topology contains an invalid bond")

    records: list[tuple[str, int, int, int, str, tuple[tuple[str, str], ...]]] = []
    shared_proton_records = perceive_proton_transfer_bridges(numbers, xyz, bonds)
    for hydrogen, left, right in shared_proton_records:
        if all(
            is_electronegative_lone_pair_donor(numbers[atom])
            for atom in (left, right)
        ) and not any(is_structural_center(numbers[atom]) for atom in (left, right)):
            records.append(
                (
                    "SHARED_PROTON",
                    hydrogen,
                    left,
                    right,
                    SHARED_PROTON_PROVIDER,
                    (
                        (
                            "recognition",
                            "near-linear electronegative lone-pair-center shared proton",
                        ),
                    ),
                )
            )

    # A donor-bound acidic hydrogen is promoted to a transferable,
    # bicoordinated ZAFF center whenever ORACLE's common contact kernel finds
    # a nearby acceptor.  This is a covalent-coordinate classification, not a
    # second hydrogen-bond energy term.  Keep both distances in the redundant
    # field catalog; SMITH adds their difference as the proton-transfer
    # coordinate and reduces the chart only for GF/displacement work.
    existing_proton_domains = {
        (hydrogen, min(left, right), max(left, right))
        for hydrogen, left, right in shared_proton_records
    }
    for contact in perceive_hydrogen_bonds(numbers, xyz, bonds):
        hydrogen = int(contact.hydrogen)
        left = int(contact.donor)
        right = int(contact.acceptor)
        transfer_cutoff = (
            periodic_atomic_properties(numbers[hydrogen]).covalent_radius_angstrom
            + periodic_atomic_properties(numbers[right]).covalent_radius_angstrom
            + MAXIMUM_TRANSFERABLE_HYDROGEN_COVALENT_EXTENSION_ANGSTROM
        )
        if float(contact.distance_angstrom) > transfer_cutoff:
            continue
        key = (hydrogen, min(left, right), max(left, right))
        if key in existing_proton_domains:
            continue
        records.append(
            (
                "TRANSFERABLE_HYDROGEN",
                hydrogen,
                left,
                right,
                TRANSFERABLE_HYDROGEN_PROVIDER,
                (
                    (
                        "recognition",
                        "transferable acidic hydrogen with a nearby lone-pair acceptor",
                    ),
                    ("coordination_weight", f"{contact.coordination_weight:.12g}"),
                    ("transfer_cutoff_angstrom", f"{transfer_cutoff:.12g}"),
                ),
            )
        )

    explicit = {
        (int(bridge), min(int(left), int(right)), max(int(left), int(right)))
        for bridge, left, right in explicit_multicenter_bridges
    }
    perceived_structural = set(
        _perceive_structural_ligand_bridges(numbers, xyz, bonds)
    )
    for bridge, left, right in sorted(perceived_structural | explicit):
        if min(bridge, left, right) < 0 or max(bridge, left, right) >= len(numbers):
            raise ValueError("explicit structural bridge lies outside the geometry")
        if not is_bridging_ligand(numbers[bridge]):
            raise ValueError("structural bridge atom must be hydrogen or a halogen")
        if not all(is_structural_center(numbers[atom]) for atom in (left, right)):
            raise ValueError("structural bridge endpoints must be structural centers")
        left_ratio = _normalized_bridge_distance(numbers, xyz, bridge, left)
        right_ratio = _normalized_bridge_distance(numbers, xyz, bridge, right)
        records.append(
            (
                "STRUCTURAL_LIGAND_BRIDGE",
                bridge,
                left,
                right,
                STRUCTURAL_LIGAND_BRIDGE_PROVIDER,
                (
                    ("bridge_element", str(numbers[bridge])),
                    ("left_element", str(numbers[left])),
                    ("right_element", str(numbers[right])),
                    ("left_normalized_xl", f"{left_ratio:.12g}"),
                    ("right_normalized_xl", f"{right_ratio:.12g}"),
                    (
                        "normalization",
                        "d(X,L)/(Rcov(X)+Rcov(L)); periodic structural-center/bridge-ligand roles",
                    ),
                ),
            )
        )

    domains: list[MulticenterDomain] = []
    candidates: list[PrimitiveCandidate] = []
    bridges_by_centers: dict[tuple[int, int], list[tuple[str, int]]] = defaultdict(list)
    seen = set()
    for kind, bridge, left, right, provider, descriptors in records:
        key = (kind, bridge, min(left, right), max(left, right))
        if key in seen:
            continue
        seen.add(key)
        domain_id = f"D{len(domains) + 1:04d}"
        primitive_ids = []
        for heavy in (left, right):
            candidate = _primitive_candidate(
                Primitive("bond", (bridge, heavy)),
                xyz,
                candidate_id=f"P{len(candidates) + 1:05d}",
                family="MULTICENTER_DISTANCE",
                units="ANGSTROM",
                owner_id=domain_id,
                domain_id=domain_id,
                provenance=f"{provider}@{PROVIDER_VERSION}",
            )
            candidates.append(candidate)
            primitive_ids.append(candidate.candidate_id)
        bend = _primitive_candidate(
            Primitive("angle", (left, bridge, right)),
            xyz,
            candidate_id=f"P{len(candidates) + 1:05d}",
            family="MULTICENTER_BEND",
            units="RADIAN",
            owner_id=domain_id,
            domain_id=domain_id,
            provenance=f"{provider}@{PROVIDER_VERSION}",
        )
        candidates.append(bend)
        primitive_ids.append(bend.candidate_id)
        domains.append(
            MulticenterDomain(
                domain_id=domain_id,
                kind="MULTICENTER_BRIDGE(X,L,X')",
                atoms=(left + 1, bridge + 1, right + 1),
                provider=provider,
                provider_version=PROVIDER_VERSION,
                primitive_candidate_ids=tuple(primitive_ids),
                provenance=f"{MULTICENTER_PROVIDER_SCHEMA}:{provider}@{PROVIDER_VERSION}",
                descriptors=descriptors,
            )
        )
        bridges_by_centers[(min(left, right), max(left, right))].append((domain_id, bridge))

    for (left, right), bridges in sorted(bridges_by_centers.items()):
        if len(bridges) < 2:
            continue
        for index in range(len(bridges) - 1):
            owner_id, first_bridge = bridges[index]
            _second_owner, second_bridge = bridges[index + 1]
            coupling = _primitive_candidate(
                Primitive("dihedral", (first_bridge, left, second_bridge, right)),
                xyz,
                candidate_id=f"P{len(candidates) + 1:05d}",
                family="MULTICENTER_BRIDGE_PLANE_COUPLING",
                units="RADIAN",
                owner_id=owner_id,
                domain_id=owner_id,
                provenance=f"{BRIDGE_PLANE_PROVIDER}@{PROVIDER_VERSION}",
            )
            wilson_values = (
                coupling.analytic_wilson_values
                if coupling.analytic_wilson_values
                else coupling.analytic_wilson_row
            )
            if float(np.linalg.norm(wilson_values)) <= 1.0e-10:
                raise ValueError("multibridge plane coupling has a null Wilson row")
            candidates.append(coupling)
            domain_index = next(
                position for position, domain in enumerate(domains) if domain.domain_id == owner_id
            )
            domains[domain_index] = replace(
                domains[domain_index],
                primitive_candidate_ids=(
                    *domains[domain_index].primitive_candidate_ids,
                    coupling.candidate_id,
                ),
            )
    return tuple(domains), tuple(candidates)


def _normalized_bridge_distance(
    numbers: tuple[int, ...], xyz: np.ndarray, bridge: int, center: int
) -> float:
    bridge_radius = periodic_atomic_properties(
        numbers[bridge]
    ).covalent_radius_angstrom
    center_radius = periodic_atomic_properties(
        numbers[center]
    ).covalent_radius_angstrom
    distance = float(np.linalg.norm(xyz[bridge] - xyz[center]))
    return distance / (bridge_radius + center_radius)


def _perceive_structural_ligand_bridges(
    numbers: tuple[int, ...],
    xyz: np.ndarray,
    bonds: set[tuple[int, int]],
) -> tuple[tuple[int, int, int], ...]:
    """Recognize H/halide bridges using only periodic roles and normalized geometry."""

    adjacency = [set() for _ in numbers]
    for left, right in bonds:
        adjacency[left].add(right)
        adjacency[right].add(left)
    records = []
    for bridge, number in enumerate(numbers):
        if not is_bridging_ligand(number):
            continue
        if any(not is_structural_center(numbers[atom]) for atom in adjacency[bridge]):
            continue
        candidates = sorted(
            (
                _normalized_bridge_distance(numbers, xyz, bridge, center),
                center,
            )
            for center, center_number in enumerate(numbers)
            if center != bridge
            and is_structural_center(center_number)
            and 1.0e-12
            < float(np.linalg.norm(xyz[bridge] - xyz[center]))
        )
        candidates = [
            item
            for item in candidates
            if item[0] <= MAXIMUM_NORMALIZED_STRUCTURAL_BRIDGE_DISTANCE
        ]
        if len(candidates) < 2:
            continue
        (_left_ratio, left), (_right_ratio, right) = candidates[:2]
        left_vector = xyz[left] - xyz[bridge]
        right_vector = xyz[right] - xyz[bridge]
        denominator = float(np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
        if denominator <= 1.0e-12:
            continue
        angle = float(
            np.degrees(
                np.arccos(
                    np.clip(
                        np.dot(left_vector, right_vector) / denominator,
                        -1.0,
                        1.0,
                    )
                )
            )
        )
        if angle < MINIMUM_STRUCTURAL_BRIDGE_ANGLE_DEGREES:
            continue
        records.append((bridge, min(left, right), max(left, right)))
    return tuple(records)


def _primitive_candidate(
    primitive: Primitive,
    xyz: np.ndarray,
    *,
    candidate_id: str,
    family: str,
    units: str,
    owner_id: str,
    domain_id: str,
    provenance: str,
) -> PrimitiveCandidate:
    entries = sparse_grad_primitive_entries(primitive, xyz)
    values = tuple(value for _column, value in entries)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"non-finite analytic Wilson row for {candidate_id}")
    large_sparse = xyz.shape[0] >= 256
    dense_row = np.zeros(xyz.size, dtype=float)
    if not large_sparse:
        for column, value in entries:
            dense_row[column] = value
    return PrimitiveCandidate(
        candidate_id=candidate_id,
        function=primitive.function,
        atoms=tuple(atom + 1 for atom in primitive.atoms),
        mode=primitive.mode,
        ref_atoms=tuple(atom + 1 for atom in primitive.ref),
        refs=(),
        family=family,
        units=units,
        owner_id=owner_id,
        domain_id=domain_id,
        provenance=provenance,
        analytic_wilson_row=tuple(float(value) for value in dense_row),
        analytic_wilson_indices=(
            tuple(column for column, _value in entries) if large_sparse else ()
        ),
        analytic_wilson_values=values if large_sparse else (),
        wilson_column_count=int(xyz.size) if large_sparse else 0,
    )


__all__ = [
    "BRIDGE_PLANE_PROVIDER",
    "MAXIMUM_NORMALIZED_STRUCTURAL_BRIDGE_DISTANCE",
    "MAXIMUM_TRANSFERABLE_HYDROGEN_COVALENT_EXTENSION_ANGSTROM",
    "MINIMUM_STRUCTURAL_BRIDGE_ANGLE_DEGREES",
    "MULTICENTER_PROVIDER_SCHEMA",
    "STRUCTURAL_LIGAND_BRIDGE_PROVIDER",
    "SHARED_PROTON_PROVIDER",
    "TRANSFERABLE_HYDROGEN_PROVIDER",
    "perceive_multicenter_domains",
]
