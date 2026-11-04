"""ORACLE-owned transition-state catalog for one starting geometry.

The catalog classifies a frozen reaction kernel and resolves the complete
chart prescription.  Category identifiers are intentionally open-ended;
SMITH consumes only the stable execution policy and exact pseudobond records.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from matrix_chem import (
    ORACLE_TS_GEOMETRY_CONTRACT_OWNER,
    ORACLE_TS_GEOMETRY_CONTRACT_SCHEMA,
    FragmentMembership,
    OracleTransitionStateGeometryContract,
    TS_CHART_MINIMUM_LIKE,
    TS_CHART_REACTIVE_DISTANCE,
    TS_CHART_REACTIVE_PSEUDOBOND,
    TS_ENDPOINT_ROUTE_STATUS,
    TS_SOURCE_SINGLE_GEOMETRY,
    TransitionStateKernelEdge,
    TransitionStatePseudobond,
    is_bridging_ligand,
    pseudo_bond_pairs,
    primary_topology_hash,
    read_enriched_xyz,
    read_molecular_symmetry,
    topology_snapshot_from_xyzin,
    validate_oracle_transition_state_geometry_contract,
    write_oracle_transition_state_geometry_contract,
)
from matrix_chem.topology.elements import atomic_number
from matrix_chem.topology.covalent_radii import covalent_radius
from matrix_chem.topology.bonding_roles import periodic_group
from matrix_chem.primitive_coordinates import LINEAR_ANGLE_DEGREES


ORACLE_TS_SINGLE_GEOMETRY_CATALOG = "ORACLE_TS_SINGLE_GEOMETRY_CATALOG"
ORACLE_TS_SINGLE_GEOMETRY_CATALOG_VERSION = "2"
ORACLE_TS_SINGLE_GEOMETRY_BUILDER = "ORACLE_TS_SINGLE_GEOMETRY_BUILDER@2"
ORACLE_TS_LIFECYCLE_KERNEL_ANCHOR = "ORACLE_TS_LIFECYCLE_KERNEL_ANCHOR@2"
TS_EXTENDED_COVALENT_DISTANCE_SCALE = 1.65
TS_LOCAL_TRANSFER_DISTANCE_SCALE = 1.50
TS_SINGLETON_TRANSFER_DISTANCE_SCALE = 2.05
TS_KERNEL_ALTERNATIVE_DISTANCE_MARGIN = 0.18
TS_DISTANCE_ONLY_KERNEL_DESCRIPTOR = "DISTANCE_ONLY_KERNEL_EDGES"


@dataclass(frozen=True)
class _TransitionStateContactCandidate:
    atoms: tuple[int, int]
    role: str
    kind: str
    priority: int
    distance_angstrom: float
    distance_ratio: float
    coordination_gain: int
    creates_linear_angle: bool
    component_pair: tuple[int, int]
    singleton_transfer_center: int = 0


@dataclass(frozen=True)
class TransitionStateGeometryFeatures:
    natoms: int
    atomic_numbers: tuple[int, ...]
    bonds: tuple[tuple[int, int], ...]
    breaking: tuple[tuple[int, int], ...]
    forming: tuple[tuple[int, int], ...]
    stable_degree: tuple[int, ...]
    symmetry_permutations: tuple[tuple[int, ...], ...]

    @property
    def kernel(self) -> tuple[tuple[int, int], ...]:
        return tuple(sorted(set(self.breaking + self.forming)))


@dataclass(frozen=True)
class TransitionStateCatalogRule:
    category_id: str
    chart_policy: str
    priority: int
    predicate: Callable[[TransitionStateGeometryFeatures], bool]


def _symmetric_monovalent_bridge(features: TransitionStateGeometryFeatures) -> bool:
    adjacency = _adjacency(features.natoms, features.bonds)
    return any(
        is_bridging_ligand(features.atomic_numbers[center - 1])
        and len(adjacency[center]) == 2
        and permutation[center - 1] == center
        and permutation[left - 1] == right
        and permutation[right - 1] == left
        for center in range(1, features.natoms + 1)
        for left, right in _one_pair(adjacency[center])
        for permutation in features.symmetry_permutations
    )


def _empty_kernel(features: TransitionStateGeometryFeatures) -> bool:
    return not features.kernel


def _symmetric_hypervalent_substitution(
    features: TransitionStateGeometryFeatures,
) -> bool:
    changed_neighbors: dict[int, set[int]] = {}
    for left, right in features.kernel:
        changed_neighbors.setdefault(left, set()).add(right)
        changed_neighbors.setdefault(right, set()).add(left)
    return any(
        features.stable_degree[center - 1] >= 2
        and permutation[center - 1] == center
        and permutation[left - 1] == right
        and permutation[right - 1] == left
        for center, neighbors in changed_neighbors.items()
        for left, right in _pairs(neighbors)
        for permutation in features.symmetry_permutations
    )


def _atom_transfer(features: TransitionStateGeometryFeatures) -> bool:
    changed_neighbors: dict[int, set[int]] = {}
    for left, right in features.kernel:
        changed_neighbors.setdefault(left, set()).add(right)
        changed_neighbors.setdefault(right, set()).add(left)
    return any(
        features.stable_degree[atom - 1] <= 1 and len(neighbors) >= 2
        for atom, neighbors in changed_neighbors.items()
    )


def _insertion_extrusion(features: TransitionStateGeometryFeatures) -> bool:
    return (len(features.forming), len(features.breaking)) in {(2, 1), (1, 2)}


def _cyclic_multibond(features: TransitionStateGeometryFeatures) -> bool:
    if len(features.kernel) < 2:
        return False
    return _cycle_rank(features.natoms, features.bonds + features.kernel) > _cycle_rank(
        features.natoms,
        features.bonds,
    )


def _ring_closure(features: TransitionStateGeometryFeatures) -> bool:
    if len(features.forming) != 1:
        return False
    return _cycle_rank(features.natoms, features.bonds + features.forming) > _cycle_rank(
        features.natoms,
        features.bonds,
    )


def _association_dissociation(features: TransitionStateGeometryFeatures) -> bool:
    return bool(features.kernel) and (not features.breaking or not features.forming)


def _reactive_fallback(features: TransitionStateGeometryFeatures) -> bool:
    return bool(features.kernel)


def _substitution_rearrangement(features: TransitionStateGeometryFeatures) -> bool:
    return bool(features.breaking and features.forming)


# New situations are appended here as rules.  The transport schema and SMITH
# consumer do not change when a new chemical category is introduced.
TRANSITION_STATE_SINGLE_GEOMETRY_CATALOG = tuple(
    sorted(
        (
            TransitionStateCatalogRule(
                "SYMMETRIC_MONOVALENT_BRIDGE_MINIMUM",
                TS_CHART_MINIMUM_LIKE,
                10,
                lambda features: _empty_kernel(features) and _symmetric_monovalent_bridge(features),
            ),
            TransitionStateCatalogRule(
                "TOPOLOGY_PRESERVING_MINIMUM_LIKE",
                TS_CHART_MINIMUM_LIKE,
                20,
                _empty_kernel,
            ),
            TransitionStateCatalogRule(
                "SYMMETRIC_HYPERVALENT_SUBSTITUTION",
                TS_CHART_REACTIVE_PSEUDOBOND,
                30,
                _symmetric_hypervalent_substitution,
            ),
            TransitionStateCatalogRule(
                "ATOM_TRANSFER",
                TS_CHART_REACTIVE_PSEUDOBOND,
                40,
                _atom_transfer,
            ),
            TransitionStateCatalogRule(
                "INSERTION_EXTRUSION",
                TS_CHART_REACTIVE_PSEUDOBOND,
                50,
                _insertion_extrusion,
            ),
            TransitionStateCatalogRule(
                "CYCLIC_MULTIBOND",
                TS_CHART_REACTIVE_PSEUDOBOND,
                60,
                _cyclic_multibond,
            ),
            TransitionStateCatalogRule(
                "RING_CLOSURE",
                TS_CHART_REACTIVE_PSEUDOBOND,
                65,
                _ring_closure,
            ),
            TransitionStateCatalogRule(
                "ASSOCIATION_DISSOCIATION",
                TS_CHART_REACTIVE_PSEUDOBOND,
                70,
                _association_dissociation,
            ),
            TransitionStateCatalogRule(
                "SUBSTITUTION_REARRANGEMENT",
                TS_CHART_REACTIVE_PSEUDOBOND,
                80,
                _substitution_rearrangement,
            ),
            TransitionStateCatalogRule(
                "UNCLASSIFIED_REACTIVE_KERNEL",
                TS_CHART_REACTIVE_PSEUDOBOND,
                1000,
                _reactive_fallback,
            ),
        ),
        key=lambda rule: (rule.priority, rule.category_id),
    )
)


def classify_transition_state_geometry_features(
    features: TransitionStateGeometryFeatures,
) -> TransitionStateCatalogRule:
    """Apply the ordered catalog and return its complete chart decision."""

    rule = next(
        (item for item in TRANSITION_STATE_SINGLE_GEOMETRY_CATALOG if item.predicate(features)),
        None,
    )
    if rule is None:
        raise ValueError("ORACLE TS catalog has no deterministic fallback rule")
    return rule


def build_oracle_transition_state_geometry_contract_from_xyzin(
    path: Path,
    *,
    lifecycle_anchor: OracleTransitionStateGeometryContract | None = None,
) -> OracleTransitionStateGeometryContract:
    """Classify and prescribe one explicitly requested single-geometry TS."""

    target = Path(path)
    geometry = read_enriched_xyz(target)
    coordinates = np.asarray(geometry.coordinates_angstrom, dtype=float)
    numbers = tuple(atomic_number(symbol) for symbol in geometry.atoms)
    snapshot = topology_snapshot_from_xyzin(target)
    bonds = tuple(tuple(int(atom) for atom in pair) for pair in snapshot["bonds"])
    evidence = tuple(snapshot.get("transitional_contacts", ()))
    candidates = _transition_state_contact_candidates(
        numbers,
        coordinates,
        bonds,
        evidence,
    )
    perceived_kernel_candidates = _select_reaction_kernel_candidates(
        candidates,
        natoms=len(numbers),
        bonds=bonds,
    )
    outside_anchor_pairs = _lifecycle_outside_anchor_pairs(
        lifecycle_anchor,
        perceived_kernel_candidates=perceived_kernel_candidates,
    )
    kernel_candidates = _lifecycle_kernel_candidates(
        lifecycle_anchor,
        perceived_kernel_candidates=perceived_kernel_candidates,
        numbers=numbers,
        coordinates=coordinates,
        bonds=bonds,
    )
    breaking = tuple(
        candidate.atoms for candidate in kernel_candidates if candidate.role == "BREAKING"
    )
    forming = tuple(
        candidate.atoms for candidate in kernel_candidates if candidate.role == "FORMING"
    )
    degree = [0] * len(numbers)
    for left, right in bonds:
        degree[left - 1] += 1
        degree[right - 1] += 1
    symmetry = read_molecular_symmetry(target)
    permutations = tuple(
        dict.fromkeys(operation.permutation for operation in symmetry.operations)
    ) or (tuple(range(1, len(numbers) + 1)),)
    features = TransitionStateGeometryFeatures(
        natoms=len(numbers),
        atomic_numbers=numbers,
        bonds=bonds,
        breaking=breaking,
        forming=forming,
        stable_degree=tuple(degree),
        symmetry_permutations=permutations,
    )
    perceived_rule = classify_transition_state_geometry_features(features)

    kernel = tuple(
        TransitionStateKernelEdge(
            atoms=candidate.atoms,
            role=candidate.role,
            kind=candidate.kind,
            priority=candidate.priority,
            provenance=(
                ORACLE_TS_LIFECYCLE_KERNEL_ANCHOR
                if lifecycle_anchor is not None
                else ORACLE_TS_SINGLE_GEOMETRY_BUILDER
            ),
        )
        for candidate in kernel_candidates
    )
    contact_candidates = _merge_kernel_contact_candidates(candidates, kernel_candidates)
    if lifecycle_anchor is None:
        category_id = perceived_rule.category_id
        chart_policy = perceived_rule.chart_policy
        rule_priority = str(perceived_rule.priority)
        pseudobonds = _prescribed_pseudobonds(
            chart_policy,
            numbers=numbers,
            coordinates=coordinates,
            bonds=bonds,
            kernel=kernel,
            candidates=contact_candidates,
        )
        anchor_descriptors: dict[str, str] = {}
    else:
        anchor_descriptors = dict(lifecycle_anchor.descriptors)
        category_id = lifecycle_anchor.category_id
        rule_priority = anchor_descriptors.get(
            "RULE_PRIORITY",
            str(perceived_rule.priority),
        )
        anchor_pairs = {edge.atoms for edge in kernel}
        anchored_contact_candidates = tuple(
            candidate
            for candidate in contact_candidates
            if candidate.atoms in anchor_pairs
        )
        pseudobonds = _prescribed_pseudobonds(
            lifecycle_anchor.chart_policy,
            numbers=numbers,
            coordinates=coordinates,
            bonds=bonds,
            kernel=kernel,
            candidates=anchored_contact_candidates,
            allow_auxiliary_contacts=False,
            require_interfragment_kernel=True,
            provenance=ORACLE_TS_LIFECYCLE_KERNEL_ANCHOR,
        )
        chart_policy = (
            TS_CHART_MINIMUM_LIKE
            if lifecycle_anchor.chart_policy == TS_CHART_MINIMUM_LIKE
            else TS_CHART_REACTIVE_PSEUDOBOND
            if pseudobonds
            else TS_CHART_REACTIVE_DISTANCE
        )
    pseudobond_pairs = {record.atoms for record in pseudobonds}
    # A multibond cyclic TS needs every forming arm in its coordinate chart,
    # but only a spanning subset may alter valence adjacency.  The remaining
    # kernel arms are prescribed as distance-only coordinates so SMITH can
    # retain the complete reaction subspace without manufacturing rings,
    # angles or torsions around artificial closure edges.
    distance_only_kernel_edges = (
        tuple(edge.atoms for edge in kernel if edge.role == "FORMING")
        if category_id == "RING_CLOSURE"
        else tuple(edge.atoms for edge in kernel if edge.atoms not in pseudobond_pairs)
        if category_id == "CYCLIC_MULTIBOND"
        else ()
    )
    if lifecycle_anchor is None:
        distance_only_text = _pair_list_text(distance_only_kernel_edges)
    else:
        # Preserve every reaction-distance identity from the frozen chart.
        # When an anchored pseudobond becomes current constitutional adjacency,
        # its ordinary stretch does not carry the same reaction semantics; keep
        # that pair as a distance-only TS coordinate in the rebuilt chart.
        anchor_distance_only = anchor_descriptors["DISTANCE_ONLY_KERNEL_EDGES"]
        current_pseudobond_pairs = {record.atoms for record in pseudobonds}
        lost_anchor_pseudobond_pairs = {
            record.atoms for record in lifecycle_anchor.prescribed_pseudobonds
        } - current_pseudobond_pairs
        anchored_distance_only_pairs = set(_pair_list_pairs(anchor_distance_only))
        preserved_distance_only_pairs = tuple(
            sorted(anchored_distance_only_pairs | lost_anchor_pseudobond_pairs)
        )
        if preserved_distance_only_pairs or pseudobonds or not kernel:
            distance_only_text = _pair_list_text(preserved_distance_only_pairs)
        else:
            selected_edge = min(
                kernel,
                key=lambda edge: (
                    _fallback_contact_candidate(
                        edge,
                        numbers=numbers,
                        coordinates=coordinates,
                        bonds=bonds,
                    ).creates_linear_angle,
                    edge.priority,
                    edge.atoms,
                ),
            )
            distance_only_text = _pair_list_text((selected_edge.atoms,))
    separate_exocyclic_torsions = (
        "TRUE"
        if category_id == "TOPOLOGY_PRESERVING_MINIMUM_LIKE"
        else "FALSE"
        if lifecycle_anchor is None
        else anchor_descriptors["SEPARATE_EXOCYCLIC_TORSIONS"]
    )
    execution_descriptors = (
        ("SEPARATE_EXOCYCLIC_TORSIONS", separate_exocyclic_torsions),
        ("DISTANCE_ONLY_KERNEL_EDGES", distance_only_text),
        *(
            (
                (
                    "LIFECYCLE_REACTION_KERNEL_ANCHOR",
                    _pair_list_text(tuple(edge.atoms for edge in kernel)),
                ),
                (
                    "LIFECYCLE_PERCEIVED_OUTSIDE_ANCHOR",
                    _pair_list_text(outside_anchor_pairs),
                ),
            )
            if lifecycle_anchor is not None
            else ()
        ),
    )
    contract = OracleTransitionStateGeometryContract(
        schema=ORACLE_TS_GEOMETRY_CONTRACT_SCHEMA,
        owner=ORACLE_TS_GEOMETRY_CONTRACT_OWNER,
        source=TS_SOURCE_SINGLE_GEOMETRY,
        catalog_id=ORACLE_TS_SINGLE_GEOMETRY_CATALOG,
        catalog_version=ORACLE_TS_SINGLE_GEOMETRY_CATALOG_VERSION,
        category_id=category_id,
        chart_policy=chart_policy,
        natoms=len(numbers),
        topology_hash=primary_topology_hash(
            numbers,
            bonds,
            tuple(tuple(int(atom) for atom in ring["atoms"]) for ring in snapshot["rings"]),
            tuple(
                FragmentMembership(
                    f"F{index:03d}",
                    tuple(int(atom) for atom in atoms),
                )
                for index, atoms in enumerate(snapshot["fragments"], start=1)
            ),
        ),
        reaction_kernel=kernel,
        prescribed_pseudobonds=pseudobonds,
        descriptors=(
            ("BREAKING_EDGE_COUNT", str(len(breaking))),
            ("FORMING_EDGE_COUNT", str(len(forming))),
            ("POINT_GROUP", str(symmetry.point_group)),
            ("RULE_PRIORITY", rule_priority),
            *execution_descriptors,
        ),
        endpoints_route_status=TS_ENDPOINT_ROUTE_STATUS,
        provenance=(
            ORACLE_TS_LIFECYCLE_KERNEL_ANCHOR
            if lifecycle_anchor is not None
            else ORACLE_TS_SINGLE_GEOMETRY_BUILDER
        ),
    )
    validate_oracle_transition_state_geometry_contract(contract)
    return contract


def _pair_list_text(pairs: tuple[tuple[int, int], ...]) -> str:
    return ",".join(f"{left}-{right}" for left, right in pairs) or "NONE"


def _pair_list_pairs(text: str) -> tuple[tuple[int, int], ...]:
    """Parse one validated internal pair-list descriptor."""

    value = str(text).strip().upper()
    if value == "NONE":
        return ()
    pairs: list[tuple[int, int]] = []
    for token in value.split(","):
        fields = token.split("-")
        if len(fields) != 2:
            raise ValueError(f"invalid transition-state pair-list descriptor: {text}")
        left, right = sorted((int(fields[0]), int(fields[1])))
        if left < 1 or left == right:
            raise ValueError(f"invalid transition-state pair-list descriptor: {text}")
        pairs.append((left, right))
    if len(set(pairs)) != len(pairs):
        raise ValueError(f"duplicate transition-state pair-list descriptor: {text}")
    return tuple(pairs)


def write_oracle_transition_state_geometry_contract_from_xyzin(
    path: Path,
    *,
    lifecycle_anchor: OracleTransitionStateGeometryContract | None = None,
) -> OracleTransitionStateGeometryContract:
    contract = build_oracle_transition_state_geometry_contract_from_xyzin(
        Path(path),
        lifecycle_anchor=lifecycle_anchor,
    )
    write_oracle_transition_state_geometry_contract(Path(path), contract)
    # The TS atlas replaces the previously written MINIMUM atlas, if present,
    # while preserving the separate frozen SONIC candidate pool.
    from matrix_chem import read_oracle_sonic_contract

    from .coordinate_atlas import write_transition_state_coordinate_atlas_contract

    try:
        sonic_contract = read_oracle_sonic_contract(Path(path))
    except (OSError, ValueError):
        sonic_contract = None
    write_transition_state_coordinate_atlas_contract(
        Path(path),
        contract,
        sonic_contract,
    )
    return contract


def _prescribed_pseudobonds(
    chart_policy: str,
    *,
    numbers: tuple[int, ...],
    coordinates: np.ndarray,
    bonds: tuple[tuple[int, int], ...],
    kernel: tuple[TransitionStateKernelEdge, ...],
    candidates: tuple[_TransitionStateContactCandidate, ...],
    allow_auxiliary_contacts: bool = True,
    require_interfragment_kernel: bool = False,
    provenance: str = ORACLE_TS_SINGLE_GEOMETRY_BUILDER,
) -> tuple[TransitionStatePseudobond, ...]:
    if chart_policy in {TS_CHART_MINIMUM_LIKE, TS_CHART_REACTIVE_DISTANCE}:
        return ()
    selected: dict[tuple[int, int], TransitionStatePseudobond] = {}
    candidate_by_pair = {candidate.atoms: candidate for candidate in candidates}
    required_pairs = _minimum_conditioned_component_contacts(
        len(numbers),
        bonds,
        tuple(candidate_by_pair[pair] for pair in candidate_by_pair),
    )
    kernel_by_pair = {edge.atoms: edge for edge in kernel}
    component_by_atom = _component_by_atom(len(numbers), bonds)
    crossing_kernel = tuple(
        edge
        for edge in kernel
        if component_by_atom[edge.atoms[0]]
        != component_by_atom[edge.atoms[1]]
    )
    fallback_kernel = crossing_kernel if require_interfragment_kernel else kernel
    if not required_pairs and fallback_kernel:
        required_pairs = (
            min(
                fallback_kernel,
                key=lambda edge: (
                    candidate_by_pair.get(
                        edge.atoms,
                        _fallback_contact_candidate(
                            edge,
                            numbers=numbers,
                            coordinates=coordinates,
                            bonds=bonds,
                        ),
                    ).creates_linear_angle,
                    edge.priority,
                    edge.atoms,
                ),
            ).atoms,
        )
    for pair in required_pairs:
        edge = kernel_by_pair.get(pair)
        candidate = candidate_by_pair.get(pair)
        role = edge.role if edge is not None else candidate.role
        kind = edge.kind if edge is not None else candidate.kind
        priority = edge.priority if edge is not None else candidate.priority
        selected[pair] = TransitionStatePseudobond(
            atoms=pair,
            kind=f"ORACLE_TS_{role}_{kind}",
            priority=priority,
            mandatory=True,
            provenance=provenance,
        )
    auxiliary_pairs = (
        pseudo_bond_pairs(
            numbers,
            coordinates,
            tuple(
                (left - 1, right - 1)
                for left, right in (
                    *bonds,
                    *(edge.atoms for edge in kernel),
                )
            ),
        )
        if allow_auxiliary_contacts
        else ()
    )
    for zero_pair, kind in auxiliary_pairs:
        pair = tuple(sorted((zero_pair[0] + 1, zero_pair[1] + 1)))
        selected.setdefault(
            pair,
            TransitionStatePseudobond(
                atoms=pair,
                kind=f"ORACLE_{kind}",
                priority=100,
                mandatory=True,
                provenance=provenance,
            ),
        )
    return tuple(sorted(selected.values(), key=lambda item: (item.priority, item.atoms)))


def _transition_state_contact_candidates(
    numbers: tuple[int, ...],
    coordinates: np.ndarray,
    bonds: tuple[tuple[int, int], ...],
    evidence: tuple[dict[str, object], ...],
) -> tuple[_TransitionStateContactCandidate, ...]:
    """Build a TS-only contact pool without changing the constitutional graph."""

    stable = {tuple(sorted(pair)) for pair in bonds}
    evidence_by_pair = {
        tuple(sorted(int(atom) for atom in item["atoms"])): str(item["kind"]).upper()
        for item in evidence
    }
    evidence_atoms = {
        atom
        for pair in evidence_by_pair
        for atom in pair
    }
    adjacency = _adjacency(len(numbers), bonds)
    component_by_atom = _component_by_atom(len(numbers), bonds)
    result: dict[tuple[int, int], _TransitionStateContactCandidate] = {}

    for left in range(1, len(numbers) + 1):
        for right in range(left + 1, len(numbers) + 1):
            pair = (left, right)
            evidence_kind = evidence_by_pair.get(pair)
            if pair in stable and evidence_kind is None:
                continue
            radius_sum = covalent_radius(numbers[left - 1]) + covalent_radius(numbers[right - 1])
            if radius_sum <= 0.0:
                continue
            ratio = float(
                np.linalg.norm(coordinates[left - 1] - coordinates[right - 1]) / radius_sum
            )
            component_pair = tuple(sorted((component_by_atom[left], component_by_atom[right])))
            singleton_transfer_center = 0
            if evidence_kind == "WEAK_ACYCLIC":
                # A weak edge inside one stable component is a stretched bond.
                # Across disconnected components it is instead evidence for
                # association, except when topology has deliberately left a
                # monovalent H/halogen transfer center as a singleton.  That
                # latter edge is the elongated donor bond and remains breaking.
                singleton_transfer = any(
                    is_bridging_ligand(numbers[atom - 1]) and not adjacency[atom]
                    for atom in pair
                )
                role = (
                    "BREAKING"
                    if component_pair[0] == component_pair[1] or singleton_transfer
                    else "FORMING"
                )
                kind, priority = evidence_kind, 10
            elif evidence_kind == "NEAR_COVALENT":
                role, kind, priority = "FORMING", evidence_kind, 20
            else:
                separation = _graph_separation(left, right, adjacency)
                bonded_hydrogen_cross_component = (
                    any(numbers[atom - 1] == 1 and adjacency[atom] for atom in pair)
                    and component_pair[0] != component_pair[1]
                )
                bonded_monovalent_transfer_contact = (
                    bonded_hydrogen_cross_component
                    and ratio <= TS_LOCAL_TRANSFER_DISTANCE_SCALE
                )
                is_local_monovalent_transfer = (
                    1 in {numbers[left - 1], numbers[right - 1]}
                    and separation == 2
                    and ratio <= TS_LOCAL_TRANSFER_DISTANCE_SCALE
                )
                singleton_centers = tuple(
                    atom
                    for atom in pair
                    if is_bridging_ligand(numbers[atom - 1])
                    and not adjacency[atom]
                    and atom not in evidence_atoms
                )
                singleton_transfer_center = (
                    singleton_centers[0] if len(singleton_centers) == 1 else 0
                )
                singleton_monovalent_transfer = (
                    component_pair[0] != component_pair[1]
                    and bool(singleton_transfer_center)
                    and ratio > TS_EXTENDED_COVALENT_DISTANCE_SCALE
                    and ratio <= TS_SINGLETON_TRANSFER_DISTANCE_SCALE
                )
                if (
                    (
                        ratio > TS_EXTENDED_COVALENT_DISTANCE_SCALE
                        and not singleton_monovalent_transfer
                    )
                    or (
                        bonded_hydrogen_cross_component
                        and not bonded_monovalent_transfer_contact
                    )
                    or (
                        component_pair[0] == component_pair[1]
                        and separation <= 2
                        and not is_local_monovalent_transfer
                    )
                ):
                    continue
                role, kind, priority = (
                    ("FORMING", "SINGLETON_MONOVALENT_TRANSFER", 30)
                    if singleton_monovalent_transfer
                    else ("FORMING", "EXTENDED_COVALENT", 30)
                )
            result[pair] = _TransitionStateContactCandidate(
                atoms=pair,
                role=role,
                kind=kind,
                priority=priority,
                distance_angstrom=float(
                    np.linalg.norm(coordinates[left - 1] - coordinates[right - 1])
                ),
                distance_ratio=ratio,
                coordination_gain=_contact_coordination_gain(
                    pair,
                    numbers,
                    adjacency,
                ),
                creates_linear_angle=_contact_creates_linear_angle(
                    pair,
                    coordinates,
                    adjacency,
                ),
                component_pair=component_pair,
                singleton_transfer_center=(
                    singleton_transfer_center
                    if evidence_kind is None
                    else 0
                ),
            )

    # A local atom-transfer saddle may already place the monovalent center
    # inside the constitutional graph through its shorter arm.  If that atom
    # also has a near-covalent alternative arm, the existing stable edge is
    # the breaking half of the same reaction kernel; omitting it would
    # misclassify the TS as a one-sided association after optimization.
    for center in range(1, len(numbers) + 1):
        if not is_bridging_ligand(numbers[center - 1]) or len(adjacency[center]) != 1:
            continue
        alternatives = tuple(
            candidate
            for candidate in result.values()
            if center in candidate.atoms
            and candidate.role == "FORMING"
            and candidate.distance_ratio <= TS_LOCAL_TRANSFER_DISTANCE_SCALE
        )
        if not alternatives:
            continue
        anchor = next(iter(adjacency[center]))
        pair = tuple(sorted((center, anchor)))
        if pair in result:
            continue
        radius_sum = covalent_radius(numbers[center - 1]) + covalent_radius(
            numbers[anchor - 1]
        )
        distance = float(
            np.linalg.norm(coordinates[center - 1] - coordinates[anchor - 1])
        )
        result[pair] = _TransitionStateContactCandidate(
            atoms=pair,
            role="BREAKING",
            kind="LOCAL_MONOVALENT_TRANSFER",
            priority=15,
            distance_angstrom=distance,
            distance_ratio=distance / radius_sum,
            coordination_gain=_contact_coordination_gain(pair, numbers, adjacency),
            creates_linear_angle=_contact_creates_linear_angle(
                pair, coordinates, adjacency
            ),
            component_pair=tuple(
                sorted((component_by_atom[center], component_by_atom[anchor]))
            ),
        )
    for left, right in _closest_intercomponent_pairs(coordinates, component_by_atom):
        pair = (left, right)
        if pair in result:
            continue
        radius_sum = covalent_radius(numbers[left - 1]) + covalent_radius(numbers[right - 1])
        distance = float(np.linalg.norm(coordinates[left - 1] - coordinates[right - 1]))
        result[pair] = _TransitionStateContactCandidate(
            atoms=pair,
            role="SUPPORT",
            kind="INTERCOMPONENT_CLOSEST",
            priority=100,
            distance_angstrom=distance,
            distance_ratio=distance / radius_sum,
            coordination_gain=_contact_coordination_gain(
                pair,
                numbers,
                adjacency,
            ),
            creates_linear_angle=_contact_creates_linear_angle(
                pair,
                coordinates,
                adjacency,
            ),
            component_pair=tuple(sorted((component_by_atom[left], component_by_atom[right]))),
        )
    return tuple(sorted(result.values(), key=_contact_order_key))


def _select_reaction_kernel_candidates(
    candidates: tuple[_TransitionStateContactCandidate, ...],
    *,
    natoms: int,
    bonds: tuple[tuple[int, int], ...],
) -> tuple[_TransitionStateContactCandidate, ...]:
    chemical_candidates = tuple(
        candidate for candidate in candidates if candidate.role in {"BREAKING", "FORMING"}
    )
    breaking = tuple(candidate for candidate in chemical_candidates if candidate.role == "BREAKING")
    selected = list(breaking)
    selected_forming = list(
        _minimum_conditioned_component_candidates(natoms, bonds, chemical_candidates)
    )
    if not selected_forming:
        local_breaking = tuple(
            candidate
            for candidate in breaking
            if candidate.kind == "LOCAL_MONOVALENT_TRANSFER"
        )
        local_forming = tuple(
            candidate
            for candidate in chemical_candidates
            if candidate.role == "FORMING"
            and any(
                set(candidate.atoms) & set(reference.atoms)
                for reference in local_breaking
            )
        )
        if local_forming:
            selected_forming.append(min(local_forming, key=_contact_order_key))
    if not selected_forming and not breaking:
        internal = tuple(
            candidate
            for candidate in chemical_candidates
            if candidate.role == "FORMING"
            and candidate.component_pair[0] == candidate.component_pair[1]
        )
        if internal:
            selected_forming.append(min(internal, key=_contact_order_key))
    selected.extend(selected_forming)

    # If a monovalent transfer center is deliberately left as a singleton by
    # the constitutional graph, one Cartesian TS geometry contains no donor/
    # acceptor direction.  Retain its two nearest chemically plausible arms;
    # choosing only the shortest one would collapse an atom-transfer saddle
    # into an ordinary association coordinate.
    singleton_arms: dict[int, list[_TransitionStateContactCandidate]] = {}
    for candidate in chemical_candidates:
        if not candidate.singleton_transfer_center:
            continue
        singleton_arms.setdefault(candidate.singleton_transfer_center, []).append(
            candidate
        )
    for arms in singleton_arms.values():
        if len(arms) >= 2:
            selected.extend(
                sorted(
                    arms,
                    key=lambda candidate: (
                        candidate.distance_angstrom,
                        candidate.distance_ratio,
                        candidate.atoms,
                    ),
                )[:2]
            )

    # Retain near-degenerate alternatives in the chemical kernel (for example
    # the two arms of a transfer or cycloaddition), but not in the minimum
    # pseudobond chart passed to SMITH.
    for reference in tuple(selected_forming):
        for candidate in chemical_candidates:
            if candidate.role != "FORMING" or candidate.creates_linear_angle:
                continue
            if candidate.component_pair != reference.component_pair:
                continue
            if (
                candidate.distance_ratio
                <= reference.distance_ratio + TS_KERNEL_ALTERNATIVE_DISTANCE_MARGIN
            ):
                selected.append(candidate)
    return tuple(
        sorted(
            {candidate.atoms: candidate for candidate in selected}.values(),
            key=lambda candidate: (candidate.priority, candidate.atoms, candidate.role),
        )
    )


def _lifecycle_kernel_candidates(
    lifecycle_anchor: OracleTransitionStateGeometryContract | None,
    *,
    perceived_kernel_candidates: tuple[_TransitionStateContactCandidate, ...],
    numbers: tuple[int, ...],
    coordinates: np.ndarray,
    bonds: tuple[tuple[int, int], ...],
) -> tuple[_TransitionStateContactCandidate, ...]:
    """Keep one TS reaction identity while re-evaluating its local geometry.

    Candidates outside the anchor are observations, not permission to mutate
    the frozen reaction identity.  The caller records them in the lifecycle
    descriptors while this function reconstructs only the anchored edges at
    the accepted geometry.
    """

    if lifecycle_anchor is None:
        return perceived_kernel_candidates
    validate_oracle_transition_state_geometry_contract(lifecycle_anchor)
    if lifecycle_anchor.natoms != len(numbers):
        raise ValueError("TS lifecycle anchor changed the atom count")
    anchor_by_pair = {
        tuple(sorted(edge.atoms)): edge for edge in lifecycle_anchor.reaction_kernel
    }
    return tuple(
        _fallback_contact_candidate(
            edge,
            numbers=numbers,
            coordinates=coordinates,
            bonds=bonds,
        )
        for edge in sorted(
            anchor_by_pair.values(),
            key=lambda item: (item.priority, item.atoms, item.role),
        )
    )


def _lifecycle_outside_anchor_pairs(
    lifecycle_anchor: OracleTransitionStateGeometryContract | None,
    *,
    perceived_kernel_candidates: tuple[_TransitionStateContactCandidate, ...],
) -> tuple[tuple[int, int], ...]:
    """Return current TS candidates excluded by the frozen reaction anchor."""

    if lifecycle_anchor is None:
        return ()
    validate_oracle_transition_state_geometry_contract(lifecycle_anchor)
    anchor_pairs = {
        tuple(sorted(edge.atoms)) for edge in lifecycle_anchor.reaction_kernel
    }
    perceived_pairs = {
        tuple(sorted(candidate.atoms)) for candidate in perceived_kernel_candidates
    }
    return tuple(sorted(perceived_pairs - anchor_pairs))


def _merge_kernel_contact_candidates(
    candidates: tuple[_TransitionStateContactCandidate, ...],
    kernel_candidates: tuple[_TransitionStateContactCandidate, ...],
) -> tuple[_TransitionStateContactCandidate, ...]:
    """Expose every frozen kernel edge to current-geometry support selection."""

    merged = {candidate.atoms: candidate for candidate in candidates}
    merged.update({candidate.atoms: candidate for candidate in kernel_candidates})
    return tuple(sorted(merged.values(), key=_contact_order_key))


def _minimum_conditioned_component_contacts(
    natoms: int,
    bonds: tuple[tuple[int, int], ...],
    candidates: tuple[_TransitionStateContactCandidate, ...],
) -> tuple[tuple[int, int], ...]:
    return tuple(
        candidate.atoms
        for candidate in _minimum_conditioned_component_candidates(
            natoms,
            bonds,
            candidates,
        )
    )


def _minimum_conditioned_component_candidates(
    natoms: int,
    bonds: tuple[tuple[int, int], ...],
    candidates: tuple[_TransitionStateContactCandidate, ...],
) -> tuple[_TransitionStateContactCandidate, ...]:
    component_by_atom = _component_by_atom(natoms, bonds)
    component_count = len(set(component_by_atom.values()))
    if component_count <= 1:
        return ()
    parent = list(range(component_count))

    def find(component: int) -> int:
        while parent[component] != component:
            parent[component] = parent[parent[component]]
            component = parent[component]
        return component

    selected = []
    # A reactive TS chart must contain a reaction-kernel direction, not merely
    # the shortest accidental contact between fragments.  Seed exactly one
    # well-conditioned chemical edge; the remaining component links may then
    # use the existing closest-contact support policy.
    chemical = tuple(
        candidate
        for candidate in candidates
        if candidate.role in {"FORMING", "BREAKING"}
        and candidate.component_pair[0] != candidate.component_pair[1]
    )
    if chemical:
        seed = min(chemical, key=_contact_order_key)
        left, right = seed.component_pair
        parent[find(right)] = find(left)
        selected.append(seed)
    for candidate in sorted(candidates, key=_support_completion_order_key):
        if candidate in selected:
            continue
        left, right = candidate.component_pair
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            continue
        parent[root_right] = root_left
        selected.append(candidate)
        if len(selected) == component_count - 1:
            break
    return tuple(selected)


def _contact_order_key(candidate: _TransitionStateContactCandidate):
    return (
        candidate.creates_linear_angle,
        candidate.role == "BREAKING",
        candidate.priority,
        -candidate.coordination_gain,
        candidate.distance_angstrom,
        candidate.distance_ratio,
        candidate.atoms,
    )


def _support_completion_order_key(candidate: _TransitionStateContactCandidate):
    return (
        candidate.creates_linear_angle,
        candidate.role == "BREAKING",
        candidate.priority,
        candidate.distance_angstrom,
        candidate.atoms,
    )


def _fallback_contact_candidate(
    edge: TransitionStateKernelEdge,
    *,
    numbers: tuple[int, ...],
    coordinates: np.ndarray,
    bonds: tuple[tuple[int, int], ...],
) -> _TransitionStateContactCandidate:
    left, right = edge.atoms
    radius_sum = covalent_radius(numbers[left - 1]) + covalent_radius(numbers[right - 1])
    components = _component_by_atom(len(numbers), bonds)
    return _TransitionStateContactCandidate(
        atoms=edge.atoms,
        role=edge.role,
        kind=edge.kind,
        priority=edge.priority,
        distance_angstrom=float(np.linalg.norm(coordinates[left - 1] - coordinates[right - 1])),
        distance_ratio=float(
            np.linalg.norm(coordinates[left - 1] - coordinates[right - 1]) / radius_sum
        ),
        coordination_gain=_contact_coordination_gain(
            edge.atoms,
            numbers,
            _adjacency(len(numbers), bonds),
        ),
        creates_linear_angle=_contact_creates_linear_angle(
            edge.atoms,
            coordinates,
            _adjacency(len(numbers), bonds),
        ),
        component_pair=tuple(sorted((components[left], components[right]))),
    )


def _contact_creates_linear_angle(
    pair: tuple[int, int],
    coordinates: np.ndarray,
    adjacency: tuple[set[int], ...],
) -> bool:
    cosine_limit = float(np.cos(np.deg2rad(LINEAR_ANGLE_DEGREES)))
    for center, other in (pair, pair[::-1]):
        vector = coordinates[other - 1] - coordinates[center - 1]
        norm = float(np.linalg.norm(vector))
        if norm <= 1.0e-12:
            return True
        direction = vector / norm
        for neighbor in adjacency[center]:
            neighbor_vector = coordinates[neighbor - 1] - coordinates[center - 1]
            neighbor_norm = float(np.linalg.norm(neighbor_vector))
            if neighbor_norm <= 1.0e-12:
                return True
            if float(np.dot(direction, neighbor_vector / neighbor_norm)) <= cosine_limit:
                return True
    return False


def _contact_coordination_gain(
    pair: tuple[int, int],
    numbers: tuple[int, ...],
    adjacency: tuple[set[int], ...],
) -> int:
    return sum(
        max(0, _ordinary_coordination_target(numbers[atom - 1]) - len(adjacency[atom]))
        for atom in pair
    )


def _ordinary_coordination_target(atomic_number_value: int) -> int:
    number = int(atomic_number_value)
    group = periodic_group(number)
    if number == 1 or group == 17:
        return 1
    if group == 16:
        return 2
    if group in {13, 15}:
        return 3
    if group == 14:
        return 4
    if group in {1, 2}:
        return group
    return 0


def _component_by_atom(
    natoms: int,
    bonds: tuple[tuple[int, int], ...],
) -> dict[int, int]:
    adjacency = _adjacency(natoms, bonds)
    remaining = set(range(1, natoms + 1))
    result: dict[int, int] = {}
    component = 0
    while remaining:
        stack = [min(remaining)]
        while stack:
            atom = stack.pop()
            if atom not in remaining:
                continue
            remaining.remove(atom)
            result[atom] = component
            stack.extend(sorted(adjacency[atom] & remaining, reverse=True))
        component += 1
    return result


def _closest_intercomponent_pairs(
    coordinates: np.ndarray,
    component_by_atom: dict[int, int],
) -> tuple[tuple[int, int], ...]:
    components: dict[int, list[int]] = {}
    for atom, component in component_by_atom.items():
        components.setdefault(component, []).append(atom)
    result = []
    for left_component in sorted(components):
        for right_component in range(left_component + 1, len(components)):
            pair = min(
                (
                    (
                        float(np.linalg.norm(coordinates[left - 1] - coordinates[right - 1])),
                        (min(left, right), max(left, right)),
                    )
                    for left in components[left_component]
                    for right in components[right_component]
                ),
                key=lambda item: (item[0], item[1]),
            )[1]
            result.append(pair)
    return tuple(result)


def _graph_separation(
    left: int,
    right: int,
    adjacency: tuple[set[int], ...],
) -> int | None:
    frontier = {left}
    visited = {left}
    for separation in range(1, len(adjacency)):
        frontier = {
            neighbor for atom in frontier for neighbor in adjacency[atom] if neighbor not in visited
        }
        if right in frontier:
            return separation
        if not frontier:
            return None
        visited.update(frontier)
    return None


def _adjacency(natoms: int, bonds: tuple[tuple[int, int], ...]) -> tuple[set[int], ...]:
    result = tuple(set() for _ in range(natoms + 1))
    for left, right in bonds:
        result[left].add(right)
        result[right].add(left)
    return result


def _pairs(values: set[int]):
    ordered = sorted(values)
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            yield left, right


def _one_pair(values: set[int]):
    ordered = sorted(values)
    if len(ordered) == 2:
        yield ordered[0], ordered[1]




def _cycle_rank(natoms: int, edges: tuple[tuple[int, int], ...]) -> int:
    canonical = tuple(sorted({tuple(sorted(pair)) for pair in edges}))
    components = len(set(_component_by_atom(natoms, canonical).values()))
    return len(canonical) - natoms + components


__all__ = [
    "ORACLE_TS_SINGLE_GEOMETRY_CATALOG",
    "ORACLE_TS_SINGLE_GEOMETRY_CATALOG_VERSION",
    "TRANSITION_STATE_SINGLE_GEOMETRY_CATALOG",
    "TransitionStateCatalogRule",
    "TransitionStateGeometryFeatures",
    "build_oracle_transition_state_geometry_contract_from_xyzin",
    "classify_transition_state_geometry_features",
    "write_oracle_transition_state_geometry_contract_from_xyzin",
]
