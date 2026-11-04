"""ORACLE compatibility frontend for the frozen ONIC/SONIC contract.

The builder is the only stage that converts ORACLE's frozen topology,
symmetry, multicenter, structural-site, and auxiliary-contact evidence into a
typed primitive pool.  SMITH receives the serialized result and performs no
chemical reperception.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np

from matrix_chem import (
    FragmentMembership,
    OracleSonicContract,
    ORACLE_SONIC_CONTRACT_OWNER,
    ORACLE_SONIC_CONTRACT_SCHEMA,
    PrimaryTopology,
    Primitive,
    PrimitiveCandidate,
    StructuralSite,
    build_primitives,
    graph_cycle_rank,
    geometry_identity_payload_sha256,
    admits_dative_pair,
    is_bridging_ligand,
    is_chalcogen_linkage,
    is_structural_center,
    primary_topology_hash,
    read_enriched_xyz,
    read_geometry_identity_certificate,
    read_molecular_symmetry,
    topology_snapshot_from_xyzin,
    validate_geometry_identity_certificate,
    validate_oracle_sonic_contract,
    write_oracle_sonic_contract,
)
from matrix_chem.primitive_coordinates import sparse_grad_primitive_entries
from matrix_chem.soop_contract import (
    expected_soop_support_keys, frozen_cyclic_partition_manifest,
    soop_support_key,
)
from matrix_chem.topology.elements import atomic_number
from matrix_fragments import (
    fragment_frame_anchor_atoms,
    fragment_local_frame,
    read_interaction_center_definition,
)

from .auxiliary_contacts import (
    StructuralSiteContactRequest,
    perceive_auxiliary_contact_evidence,
    qualified_vdw_radius,
)
from .contact_graph import complete_and_classify_contact_orbits
from .multicenter_domains import perceive_multicenter_domains
from .local_perception import (
    perceive_local_perception_domains,
    read_frozen_effective_atomic_numbers,
)
from .perception_policy import chemical_perception_policy_manifest


ORACLE_SONIC_CONTRACT_BUILDER = "ORACLE_SONIC_CONTRACT_BUILDER"
ORACLE_SONIC_CONTRACT_BUILDER_VERSION = "2"


def extend_contact_primitive_pool(contract, atlas, coordinates):
    """Publish natural contact supports also for imported frozen contracts."""
    from .special_primitives import contact_chart_primitives, _contact_topology
    candidates = list(contract.primitive_candidates)
    keys = {(c.function, c.atoms, c.mode, c.ref_atoms) for c in candidates}
    contacts = list(contract.auxiliary_contacts)
    owner_by_pair = {}
    additions_by_owner = {}
    for record in contacts:
        if record.endpoint_a.kind == record.endpoint_b.kind == "ATOM":
            owner_by_pair.setdefault(
                frozenset((record.endpoint_a.identifier, record.endpoint_b.identifier)),
                record.contact_id,
            )
    fragments = (tuple(FragmentMembership(b.body_id, b.atoms) for b in atlas.bodies)
                 or contract.primary_topology.fragments)
    contact_topology = _contact_topology(fragments, contract.primary_topology.bonds)
    reactive_support_edges = tuple(
        tuple(sorted((int(item.endpoint_a[1]), int(item.endpoint_b[1]))))
        for item in atlas.interactions
        if getattr(item, "graph_role", None) == "REACTIVE_SUPPORT"
        and getattr(item, "coordinate_role", None) != "REACTION_DISTANCE_ONLY"
        and item.endpoint_a[0] == item.endpoint_b[0] == "ATOM"
    )
    for item in atlas.interactions:
        if item.pseudobond_policy != "REQUIRED" or item.endpoint_a[0] != "ATOM" or item.endpoint_b[0] != "ATOM":
            continue
        contact = SimpleNamespace(endpoint_a=SimpleNamespace(kind="ATOM", identifier=item.endpoint_a[1]),
                                  endpoint_b=SimpleNamespace(kind="ATOM", identifier=item.endpoint_b[1]))
        pair = frozenset((item.endpoint_a[1], item.endpoint_b[1]))
        owner = owner_by_pair.get(pair, "PRIMARY_TOPOLOGY")
        additions = additions_by_owner.setdefault(owner, [])
        for primitive in contact_chart_primitives(
            contact, fragments, contract.primary_topology.bonds, coordinates,
            _topology=contact_topology,
            reactive_support_edges=reactive_support_edges,
        ):
            key = (primitive.function, tuple(a+1 for a in primitive.atoms), primitive.mode,
                   tuple(a+1 for a in primitive.ref))
            if key in keys:
                continue
            candidate = _finite_primary_candidate(primitive, coordinates,
                candidate_id=_next_candidate_id(candidates), function=primitive.function,
                family="CONTACT_SUPPORT", units=_primitive_units(primitive.function),
                owner_id=owner, domain_id="AUXILIARY_CONTACTS" if owner != "PRIMARY_TOPOLOGY" else "PRIMARY_TOPOLOGY::CONTACT_SUPPORT",
                provenance="ORACLE_ATLAS_CONTACT_SUPPORT@2:LOCAL_RADIUS3", refs=())
            if candidate is not None:
                candidates.append(candidate)
                additions.append(candidate.candidate_id)
                keys.add(key)
    contacts = [replace(c, primitive_candidate_ids=c.primitive_candidate_ids+
                        tuple(additions_by_owner[c.contact_id]))
                if c.contact_id in additions_by_owner else c for c in contacts]
    return replace(contract, primitive_candidates=tuple(candidates), auxiliary_contacts=tuple(contacts))


def extend_reactive_primitive_pool(contract, atlas, coordinates):
    """Freeze local TS support measurements without changing covalent topology.

    Distance-only arms do not seed valence neighborhoods. Symmetry images of
    allowed supports are nevertheless retained for SMITH's projection, without
    changing the graph role of an arm. The pool contains no derived SONICs.
    """
    from collections import defaultdict
    from .special_primitives import ring_and_local_plane_primitives, linear_axis_primitives
    from matrix_chem.coordinate_atlas_contract import GRAPH_ROLE_REACTIVE_SUPPORT
    from matrix_chem.primitive_coordinates import angle

    contract = extend_contact_primitive_pool(contract, atlas, coordinates)
    pairs = set()
    support = set(contract.primary_topology.bonds)
    for item in atlas.interactions:
        if item.endpoint_a[0] != "ATOM" or item.endpoint_b[0] != "ATOM":
            continue
        pair = tuple(sorted((int(item.endpoint_a[1]), int(item.endpoint_b[1]))))
        if item.coordinate_role in {"REACTION_DISTANCE", "REACTION_DISTANCE_ONLY", "TS_SUPPORT"}:
            pairs.add(pair)
        if item.graph_role == GRAPH_ROLE_REACTIVE_SUPPORT and item.coordinate_role != "REACTION_DISTANCE_ONLY":
            support.add(pair)
    adjacency = defaultdict(set)
    for a, b in support:
        adjacency[a-1].add(b-1)
        adjacency[b-1].add(a-1)
    primitives = [Primitive("bond", tuple(a-1 for a in pair)) for pair in sorted(pairs)]
    for center, neighbors in sorted(adjacency.items()):
        for a, b in combinations(sorted(neighbors), 2):
            if 1.e-6 < angle(a, center, b, coordinates) < np.pi-1.e-6:
                primitives.append(Primitive("angle", (a, center, b)))
    for a, b in sorted(support):
        for left in sorted(adjacency[a-1]-{b-1}):
            for right in sorted(adjacency[b-1]-{a-1, left}):
                if all(1.e-6 < angle(*triple, coordinates) < np.pi-1.e-6
                       for triple in ((left, a-1, b-1), (a-1, b-1, right))):
                    primitives.append(Primitive("dihedral", (left, a-1, b-1, right)))
    primitives.extend(ring_and_local_plane_primitives(contract.primary_topology.rings, tuple(sorted(support))))
    primitives.extend(linear_axis_primitives(tuple(sorted(support)), coordinates))
    # A distance-only arm can be the symmetry partner of a spanning arm.
    # Closing measurements is necessary; promoting that arm to a graph edge
    # is not. The atlas still controls which combinations SMITH may select.
    primitives = [replace(p, atoms=tuple(permutation[a]-1 for a in p.atoms),
                          ref=tuple(permutation[a]-1 for a in p.ref))
                  for p in primitives
                  for permutation in (contract.primary_topology.symmetry_permutations
                                      or (tuple(range(1, len(coordinates)+1)),))]
    candidates = list(contract.primitive_candidates)
    keys = {(c.function, c.atoms, c.mode, c.ref_atoms) for c in candidates}
    for primitive in dict.fromkeys(primitives):
        key = (primitive.function, tuple(a+1 for a in primitive.atoms), primitive.mode,
               tuple(a+1 for a in primitive.ref))
        if key in keys:
            continue
        candidate = _finite_primary_candidate(primitive, coordinates,
            candidate_id=_next_candidate_id(candidates), function=primitive.function,
            family=_primary_family(primitive), units=_primitive_units(primitive.function),
            owner_id="PRIMARY_TOPOLOGY", domain_id="PRIMARY_TOPOLOGY::REACTIVE_SUPPORT",
            provenance="ORACLE_REACTIVE_PRIMITIVE_SUPPORT@1", refs=())
        if candidate is not None:
            candidates.append(candidate)
            keys.add(key)
    reactive_out_of_plane_orders = tuple(
        candidate.atoms
        for candidate in candidates
        if candidate.function == "U"
        and candidate.family == "OUT_OF_PLANE"
        and candidate.domain_id == "PRIMARY_TOPOLOGY::REACTIVE_SUPPORT"
    )
    from .selective_out_of_plane import extend_reactive_out_of_plane_primitives
    return replace(
        contract,
        primitive_candidates=tuple(candidates),
        selective_out_of_plane_primitives=extend_reactive_out_of_plane_primitives(
            contract.selective_out_of_plane_primitives,
            contract.primary_topology,
            coordinates,
            reactive_out_of_plane_orders,
        ),
    )


def _exclude_promoted_transfer_contacts(evidence, domains):
    """Remove H-bond contacts replaced by a covalent proton-transfer domain."""

    transferred_hydrogen_pairs = {
        tuple(sorted((int(domain.atoms[1]), int(endpoint))))
        for domain in domains
        if domain.provider == "ORACLE_TRANSFERABLE_HYDROGEN_CENTERS"
        for endpoint in (domain.atoms[0], domain.atoms[2])
    }
    return tuple(
        item
        for item in evidence
        if not (
            item.kind == "HYDROGEN_BOND"
            and item.endpoint_a.kind == item.endpoint_b.kind == "ATOM"
            and tuple(
                sorted(
                    (
                        int(item.endpoint_a.identifier),
                        int(item.endpoint_b.identifier),
                    )
                )
            )
            in transferred_hydrogen_pairs
        )
    )

def build_oracle_sonic_contract_from_xyzin(
    path: Path,
    *,
    configured_site_contacts: Iterable[StructuralSiteContactRequest] = (),
    explicit_multicenter_bridges: Iterable[tuple[int, int, int]] = (),
    special_primitives: Iterable[Primitive] = (),
    primitive_mode: str = "CHEMICAL_PRIMITIVES",
) -> OracleSonicContract:
    """Build a complete frozen contract from existing ORACLE xyzin sections.

    Explicit bridge indices follow the zero-based provider API. Every
    serialized atom index remains one based. Fragment-pose candidates are
    original finite distance functions (center--center, center--anchor, and
    anchor--anchor), so their Wilson rows are analytic and no projected row is
    serialized as a nonlinear coordinate.
    """

    primitive_mode = str(primitive_mode).strip().upper()
    if primitive_mode not in {"SIMPLE_PRIMITIVES", "CHEMICAL_PRIMITIVES"}:
        raise ValueError("primitive_mode must be SIMPLE_PRIMITIVES or CHEMICAL_PRIMITIVES")
    frozen_cyclic_partition_manifest()
    target = Path(path)
    geometry = read_enriched_xyz(target)
    geometry_identity = read_geometry_identity_certificate(target)
    validate_geometry_identity_certificate(
        geometry_identity,
        canonical_atoms=geometry.atoms,
        canonical_coordinates_angstrom=geometry.coordinates_angstrom,
    )
    coordinates = np.asarray(geometry.coordinates_angstrom, dtype=float)
    numbers = tuple(int(atomic_number(symbol)) for symbol in geometry.atoms)
    snapshot = topology_snapshot_from_xyzin(target)
    bonds = tuple(tuple(int(atom) for atom in pair) for pair in snapshot["bonds"])
    rings = tuple(tuple(int(atom) for atom in item["atoms"]) for item in snapshot["rings"])
    fragments = tuple(
        FragmentMembership(f"F{index:03d}", tuple(int(atom) for atom in atoms))
        for index, atoms in enumerate(snapshot["fragments"], start=1)
    )
    symmetry = read_molecular_symmetry(target)
    cycle_rank = graph_cycle_rank(len(numbers), bonds)
    topology = PrimaryTopology(
        natoms=len(numbers),
        atomic_numbers=numbers,
        bonds=bonds,
        rings=rings,
        fragments=fragments,
        topology_hash=primary_topology_hash(numbers, bonds, rings, fragments),
        cycle_rank=cycle_rank,
        symmetry_permutations=tuple(
            dict.fromkeys(operation.permutation for operation in symmetry.operations)
        ),
    )

    zero_bonds = tuple((left - 1, right - 1) for left, right in bonds)
    zero_rings = tuple(tuple(atom - 1 for atom in ring) for ring in rings)
    effective_atomic_numbers = read_frozen_effective_atomic_numbers(
        target,
        natoms=len(numbers),
    )
    local_perception_domains = perceive_local_perception_domains(
        numbers,
        coordinates,
        zero_bonds,
        zero_rings,
        effective_atomic_numbers=effective_atomic_numbers,
    )
    domains, domain_candidates = perceive_multicenter_domains(
        numbers,
        coordinates,
        zero_bonds,
        explicit_multicenter_bridges=explicit_multicenter_bridges,
    )
    domain_signatures = {
        _candidate_signature(candidate.function, candidate.atoms)
        for candidate in domain_candidates
    }
    candidates: list[PrimitiveCandidate] = []

    graph = _frozen_graph(numbers, coordinates, zero_bonds)
    ring_index = _ring_membership_index(zero_rings)
    ring_edges = _ring_edge_index(zero_rings)
    for primitive in build_primitives(
        graph,
        coordinates,
        include_pseudo_bonds=False,
    ):
        one_based_atoms = tuple(atom + 1 for atom in primitive.atoms)
        if (
            primitive.function != "R"
            and _candidate_signature(primitive.function, one_based_atoms)
            in domain_signatures
        ):
            continue
        candidate = _finite_primary_candidate(
            primitive,
            coordinates,
            candidate_id=_next_candidate_id(candidates),
            function=primitive.function,
            family=_primary_family(primitive),
            units=_primitive_units(primitive.function),
            owner_id="PRIMARY_TOPOLOGY",
            domain_id=_primary_coordinate_domain(
                primitive, zero_rings, ring_index=ring_index,
                ring_edges=ring_edges,
            ),
            provenance=(
                f"{ORACLE_SONIC_CONTRACT_BUILDER}@"
                f"{ORACLE_SONIC_CONTRACT_BUILDER_VERSION}"
            ),
            refs=_primary_chemical_role_refs(primitive, numbers),
        )
        if candidate is not None:
            candidates.append(candidate)
    simple_candidate_count = len(candidates)

    # Fullerene face diagonals are ORACLE primitives (not SONICs): SMITH
    # later combines them with edge stretches and local flap coordinates.
    # The redundant pool contains all diagonals within each five/six-membered
    # face (not all atom pairs). SMITH selects/composes a triangulation; the
    # pool is closed under face automorphisms and remains linear in cage size.
    from .special_primitives import ring_and_local_plane_primitives, linear_axis_primitives
    from itertools import chain
    existing = {(p.function, p.atoms, p.mode, p.ref_atoms) for p in candidates}
    for primitive in chain(ring_and_local_plane_primitives(rings, bonds),
                           linear_axis_primitives(bonds, coordinates)):
        key = (primitive.function, tuple(a + 1 for a in primitive.atoms), primitive.mode, ())
        if key in existing:
            continue
        candidate = _finite_primary_candidate(
            primitive, coordinates, candidate_id=_next_candidate_id(candidates),
            function=primitive.function, family=_primary_family(primitive),
            units=_primitive_units(primitive.function), owner_id="PRIMARY_TOPOLOGY",
            domain_id="PRIMARY_TOPOLOGY::PLANE_SUPPORT", provenance=ORACLE_SONIC_CONTRACT_BUILDER,
            refs=(),
        )
        if candidate is not None:
            candidates.append(candidate)
            existing.add(key)

    for left, right, ring_index in _fullerene_face_diagonal_specs(
        numbers, bonds, rings
    ):
        primitive = Primitive("bond", (left - 1, right - 1))
        candidates.append(
            _candidate_from_primitive(
                primitive,
                coordinates,
                candidate_id=_next_candidate_id(candidates),
                function="R",
                family="FULLERENE_FACE_DIAGONAL",
                units="ANGSTROM",
                owner_id="PRIMARY_TOPOLOGY",
                domain_id=f"PRIMARY_TOPOLOGY::FULLERENE_FACE:{ring_index}",
                provenance=f"{ORACLE_SONIC_CONTRACT_BUILDER}@{ORACLE_SONIC_CONTRACT_BUILDER_VERSION}:FACE_TRIANGULATION",
                refs=(f"FULLERENE_FACE:{ring_index}",),
            )
        )

    remapped_domains = []
    for domain in domains:
        id_map: dict[str, str] = {}
        for candidate in domain_candidates:
            if candidate.domain_id != domain.domain_id:
                continue
            new_id = _next_candidate_id(candidates)
            id_map[candidate.candidate_id] = new_id
            candidates.append(replace(candidate, candidate_id=new_id))
        remapped_domains.append(
            replace(
                domain,
                primitive_candidate_ids=tuple(
                    id_map[candidate_id]
                    for candidate_id in domain.primitive_candidate_ids
                ),
            )
        )

    sites, site_candidates = _structural_sites_and_candidates(
        target,
        coordinates,
        topology,
        candidate_offset=len(candidates),
    )
    candidates.extend(site_candidates)
    from matrix_chem.face_primitives import FACE_OPERATORS

    for primitive in special_primitives:
        if primitive.kind not in FACE_OPERATORS:
            raise ValueError(f"unsupported special primitive {primitive.kind}")
        candidates.append(_candidate_from_primitive(
            primitive, coordinates, candidate_id=_next_candidate_id(candidates),
            function=primitive.function, family=primitive.function,
            units="ANGSTROM" if primitive.kind in {"signed_face_flap", "center_face_distance"} else "RADIAN",
            owner_id="PRIMARY_TOPOLOGY", domain_id="PRIMARY_TOPOLOGY::SPECIAL",
            provenance="ORACLE_EXPLICIT_SPECIAL_PRIMITIVE@1",
        ))

    evidence = perceive_auxiliary_contact_evidence(
        numbers,
        coordinates,
        zero_bonds,
        configured_site_contacts=configured_site_contacts,
    )
    evidence = _exclude_promoted_transfer_contacts(evidence, domains)
    classified = complete_and_classify_contact_orbits(
        evidence,
        topology,
        structural_sites=sites,
    )


    contacts = []
    from .special_primitives import topology_special_primitives
    from matrix_chem.face_primitives import compiled_face_value_gradient
    face_rings = _fullerene_faces(numbers, bonds) or rings
    for primitive in topology_special_primitives(face_rings, bonds, read_interaction_center_definition(target), classified.contacts):
        try:
            _value, entries = compiled_face_value_gradient(primitive, coordinates)
        except FloatingPointError:
            # Undefined face normals and collinear angles are not usable charts.
            continue
        candidates.append(PrimitiveCandidate(
            candidate_id=_next_candidate_id(candidates), function=primitive.function,
            atoms=tuple(a+1 for a in primitive.atoms), mode=0,
            ref_atoms=tuple(a+1 for a in primitive.ref), refs=(), family=primitive.function,
            units="ANGSTROM" if primitive.kind in {"signed_face_flap", "center_face_distance"} else "RADIAN",
            owner_id="PRIMARY_TOPOLOGY", domain_id="PRIMARY_TOPOLOGY::SPECIAL",
            provenance="ORACLE_TOPOLOGY_SPECIAL@1",
            frame_atoms=tuple(a+1 for a in primitive.frame_atoms),
            analytic_wilson_indices=tuple(i for i,v in entries),
            analytic_wilson_values=tuple(v for i,v in entries), wilson_column_count=coordinates.size,
        ))
    site_by_id = {site.site_id: site for site in sites}
    from .special_primitives import contact_chart_primitives, _contact_topology
    contact_topology = _contact_topology(fragments, bonds)
    for contact in classified.contacts:
        candidate = _contact_distance_candidate(
            contact,
            coordinates,
            site_by_id,
            candidate_id=_next_candidate_id(candidates),
        )
        if candidate is None:
            contacts.append(contact)
            continue
        candidates.append(candidate)
        contact_ids = [candidate.candidate_id]
        contact_support = list(contact_chart_primitives(
            contact, fragments, bonds, coordinates, _topology=contact_topology,
        ))
        if contact.endpoint_a.kind == contact.endpoint_b.kind == "ATOM":
            endpoints = {int(contact.endpoint_a.identifier), int(contact.endpoint_b.identifier)}
            orbit = []
            for permutation in topology.symmetry_permutations:
                if {permutation[a-1] for a in endpoints} != endpoints:
                    continue
                for primitive in contact_support:
                    atoms = tuple(permutation[a]-1 for a in primitive.atoms)
                    if primitive.kind == "linear_bend":
                        atoms = min(atoms, atoms[::-1])
                    orbit.append(replace(primitive, atoms=atoms,
                                         ref=tuple(permutation[a]-1 for a in primitive.ref)))
            contact_support.extend(orbit)
        for primitive in dict.fromkeys(contact_support):
            support = _finite_primary_candidate(
                primitive, coordinates, candidate_id=_next_candidate_id(candidates),
                function=primitive.function, family="CONTACT_SUPPORT",
                units=_primitive_units(primitive.function), owner_id=contact.contact_id,
                domain_id="AUXILIARY_CONTACTS", provenance="ORACLE_CONTACT_SUPPORT@2:LOCAL_RADIUS3", refs=(),
            )
            if support is not None:
                candidates.append(support)
                contact_ids.append(support.candidate_id)
        contacts.append(
            replace(contact, primitive_candidate_ids=tuple(contact_ids))
        )

    candidates.extend(
        _fragment_pose_candidates(
            coordinates,
            fragments,
            candidate_offset=len(candidates),
            symmetry_permutations=topology.symmetry_permutations,
        )
    )
    candidates.extend(_haptic_pose_candidates(target, coordinates, bonds,
                                             tuple(geometry.atoms), topology.symmetry_permutations,
                                             candidate_offset=len(candidates)))
    if primitive_mode == "SIMPLE_PRIMITIVES":
        # SIMPLE_PRIMITIVES is deliberately topology-local: conventional
        # bonds, bends, linear bends, torsions and out-of-plane rows only.
        # No inferred multicentre/contact/fragment chart may leak into this
        # recovery contract.
        candidates = candidates[:simple_candidate_count]
        remapped_domains = []
        contacts = []
    from .selective_out_of_plane import build_selective_out_of_plane_primitives
    contract = OracleSonicContract(
        schema=ORACLE_SONIC_CONTRACT_SCHEMA,
        owner=ORACLE_SONIC_CONTRACT_OWNER,
        primary_topology=topology,
        multicenter_domains=tuple(remapped_domains),
        structural_sites=sites,
        auxiliary_contacts=tuple(contacts),
        primitive_candidates=tuple(candidates),
        primary_cycle_rank=cycle_rank,
        auxiliary_cycle_rank=(cycle_rank if primitive_mode == "SIMPLE_PRIMITIVES" else classified.auxiliary_cycle_rank),
        provenance=(
            f"{ORACLE_SONIC_CONTRACT_BUILDER}@"
            f"{ORACLE_SONIC_CONTRACT_BUILDER_VERSION}"
        ),
        local_perception_domains=local_perception_domains,
        chemical_policy_sha256=chemical_perception_policy_manifest()["sha256"],
        reference_geometry_sha256=geometry_identity.canonical_geometry_sha256,
        geometry_identity_payload_sha256=geometry_identity_payload_sha256(
            geometry_identity
        ),
        primitive_mode=primitive_mode,
        # SOOP is geometry/topology evidence owned by ORACLE, not an optional
        # member of the richer chemical-candidate catalogue.  Recovery charts
        # built from SIMPLE_PRIMITIVES need the same oriented support so SMITH
        # never has to resurrect a legacy U row during a TS rebuild.
        selective_out_of_plane_primitives=(
            build_selective_out_of_plane_primitives(topology, coordinates)
        ),
    )
    validate_oracle_sonic_contract(contract)
    _validate_constructed_primitive_pool(contract, zero_rings)
    return contract


def _validate_constructed_primitive_pool(
    contract: OracleSonicContract,
    zero_rings: tuple[tuple[int, ...], ...],
) -> None:
    """Certify ORACLE's elementary pool before SMITH selects any SONICs."""

    topology = contract.primary_topology
    stretches = {
        tuple(sorted(candidate.atoms))
        for candidate in contract.primitive_candidates
        if candidate.owner_id == "PRIMARY_TOPOLOGY"
        and candidate.function == "R" and candidate.family == "STRETCH"
    }
    missing_bonds = set(topology.bonds) - stretches
    if missing_bonds:
        raise ValueError(
            f"ORACLE primitive pool lacks covalent stretches for {sorted(missing_bonds)}"
        )

    ring_index = _ring_membership_index(zero_rings)
    ring_edges = _ring_edge_index(zero_rings)
    for candidate in contract.primitive_candidates:
        if candidate.owner_id != "PRIMARY_TOPOLOGY" or candidate.function not in {"D", "U", "H"}:
            continue
        if not candidate.domain_id.startswith(("PRIMARY_TOPOLOGY::BOND:", "PRIMARY_TOPOLOGY::CENTER:")):
            continue
        kind = {"D": "dihedral", "U": "out_of_plane", "H": "out_of_plane_height"}[candidate.function]
        expected = _primary_coordinate_domain(
            Primitive(kind, tuple(atom - 1 for atom in candidate.atoms)),
            zero_rings, ring_index=ring_index, ring_edges=ring_edges,
        )
        if candidate.domain_id != expected:
            raise ValueError(
                f"ORACLE primitive {candidate.candidate_id} has inconsistent cyclic role"
            )

    expected_soop = expected_soop_support_keys(topology)
    actual_soop = {soop_support_key(primitive)
                   for primitive in contract.selective_out_of_plane_primitives}
    if actual_soop != expected_soop:
        raise ValueError(
            "ORACLE primitive pool has incomplete oriented SOOP support"
        )


def write_oracle_sonic_contract_from_xyzin(
    path: Path,
    *,
    configured_site_contacts: Iterable[StructuralSiteContactRequest] = (),
    explicit_multicenter_bridges: Iterable[tuple[int, int, int]] = (),
    minimum_fragment_representation: str = "automatic",
    special_primitives: Iterable[Primitive] = (),
    primitive_mode: str = "CHEMICAL_PRIMITIVES",
) -> OracleSonicContract:
    """Build, validate, and serialize the ORACLE-owned frozen contract."""

    contract = build_oracle_sonic_contract_from_xyzin(
        Path(path),
        configured_site_contacts=configured_site_contacts,
        explicit_multicenter_bridges=explicit_multicenter_bridges,
        special_primitives=special_primitives,
        primitive_mode=primitive_mode,
    )
    write_oracle_sonic_contract(Path(path), contract)
    # Freeze the scientific MINIMUM prescription in the same transaction.
    # Downstream SMITH must consume this atlas instead of inferring fragment
    # contacts or chart modes from geometry.
    from .coordinate_atlas import write_minimum_coordinate_atlas_contract

    write_minimum_coordinate_atlas_contract(
        Path(path),
        contract,
        fragment_representation=minimum_fragment_representation,
    )
    return contract


def _frozen_graph(
    numbers: tuple[int, ...],
    coordinates: np.ndarray,
    bonds: tuple[tuple[int, int], ...],
) -> SimpleNamespace:
    adjacency = [set() for _ in numbers]
    for left, right in bonds:
        adjacency[left].add(right)
        adjacency[right].add(left)
    return SimpleNamespace(
        natoms=len(numbers),
        Z=np.asarray(numbers, dtype=int),
        coords=coordinates,
        bonds=list(bonds),
        adjacency=adjacency,
    )


def _haptic_pose_candidates(path, coords, bonds, symbols, permutations, *, candidate_offset):
    from matrix_chem.haptic_primitives import _haptic_coordinate_bonds_and_fragments
    coordinate_bonds, records = _haptic_coordinate_bonds_and_fragments(
        bonds, interaction_centers=read_interaction_center_definition(path),
        atom_symbols=symbols, natoms=len(symbols))
    if not records:
        return ()
    # Only components joined by a haptic spoke, or sharing its metal, form a
    # local pose domain. Do not create a global all-component-pair pool.
    by_atom = {a: i for i, record in enumerate(records) for a in record.atoms}
    adjacency = {i: set() for i in range(len(records))}
    retained = set(coordinate_bonds)
    for left, right in bonds:
        if (left, right) not in retained and by_atom[left] != by_atom[right]:
            adjacency[by_atom[left]].add(by_atom[right])
            adjacency[by_atom[right]].add(by_atom[left])
    candidates = []
    seen = set()
    for i, moving in enumerate(records):
        nearby = adjacency[i] | {k for j in adjacency[i] for k in adjacency[j]}
        for j in sorted(nearby - {i}):
            reference = records[j]
            specifications = [Primitive("frag_trans", tuple(a-1 for a in moving.atoms), mode,
                                        tuple(a-1 for a in reference.atoms)) for mode in range(3)]
            moving_centered = coords[np.asarray(moving.atoms)-1]
            reference_centered = coords[np.asarray(reference.atoms)-1]
            if (np.linalg.matrix_rank(moving_centered-moving_centered.mean(axis=0)) >= 2
                    and np.linalg.matrix_rank(reference_centered-reference_centered.mean(axis=0)) >= 2):
                moving_frame = fragment_frame_anchor_atoms(coords, moving.atoms)
                ref_frame = fragment_frame_anchor_atoms(coords, reference.atoms)
                for permutation in permutations or (tuple(range(1,len(coords)+1)),):
                    atoms = tuple(sorted(permutation[a-1]-1 for a in moving.atoms))
                    ref = tuple(sorted(permutation[a-1]-1 for a in reference.atoms))
                    frame = tuple(permutation[a-1]-1 for a in moving_frame)
                    ref_frame_mapped = tuple(permutation[a-1]-1 for a in ref_frame)
                    support = sorted(set(atoms+ref))
                    for mode in range(3):
                        specifications.append(Primitive("frag_rot", atoms, mode, ref, frame, ref_frame_mapped,
                            tuple(tuple(float(v) for v in coords[a]) for a in support)))
            for primitive in specifications:
                if primitive in seen:
                    continue
                seen.add(primitive)
                candidates.append(_candidate_from_primitive(
                    primitive, coords, candidate_id=f"P{candidate_offset+len(candidates)+1:05d}",
                    function=primitive.function,
                    family="FRAG_ORIENTATION" if primitive.kind == "frag_rot" else "FRAG_TRANSLATION",
                    units="RADIAN" if primitive.kind == "frag_rot" else "ANGSTROM",
                    owner_id=f"FRAGMENT_PAIR:{moving.identifier}:{reference.identifier}",
                    domain_id="FRAGMENT_POSE", provenance="ORACLE_HAPTIC_POSE@1",
                    refs=(moving.identifier, reference.identifier)))
    return tuple(candidates)


def _fullerene_face_diagonal_specs(
    numbers: tuple[int, ...],
    bonds: tuple[tuple[int, int], ...],
    rings: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, int, int], ...]:
    """Return a face-local, symmetry-closed redundant distance pool."""

    if not numbers or any(number != 6 for number in numbers):
        return ()
    rings = _fullerene_faces(numbers, bonds)
    if not rings:
        return ()
    edge_set = {tuple(sorted(pair)) for pair in bonds}
    # A closed trivalent cage has E=3V/2, not E>=2V. Certify the
    # supplied face complex as well: edge count alone accepts open cages.
    adjacency = {atom: set() for atom in range(1, len(numbers) + 1)}
    for left, right in edge_set:
        if left not in adjacency or right not in adjacency or left == right:
            return ()
        adjacency[left].add(right)
        adjacency[right].add(left)
    if any(len(neighbors) != 3 for neighbors in adjacency.values()):
        return ()
    visited = set()
    pending = [1]
    while pending:
        atom = pending.pop()
        if atom not in visited:
            visited.add(atom)
            pending.extend(adjacency[atom] - visited)
    if len(visited) != len(numbers):
        return ()
    incidence = {edge: 0 for edge in edge_set}
    face_keys = set()
    for ring in rings:
        if len(set(ring)) != len(ring):
            return ()
        edges = frozenset(tuple(sorted((ring[i], ring[(i + 1) % len(ring)])))
                          for i in range(len(ring)))
        if edges in face_keys or not edges.issubset(edge_set):
            return ()
        face_keys.add(edges)
        for edge in edges:
            incidence[edge] += 1
    if any(count != 2 for count in incidence.values()):
        return ()
    if len(numbers) - len(edge_set) + len(rings) != 2:
        return ()
    specs: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int]] = set(edge_set)
    for ring_index, ring in enumerate(rings, start=1):
        from itertools import combinations
        for pair in combinations(ring, 2):
            left, right = sorted(pair)
            if (left, right) in seen:
                continue
            seen.add((left, right))
            specs.append((left, right, ring_index))
    return tuple(specs)


def _fullerene_faces(numbers, bonds):
    """Recover the complete face set, not just a cycle basis, in ORACLE.

    The planar embedding of a three-connected cage is unique up to reversal.
    Face orientation is canonicalized for reproducible signed coordinates.
    """
    import networkx as nx

    if not numbers or any(number != 6 for number in numbers):
        return ()
    graph = nx.Graph()
    graph.add_nodes_from(range(1, len(numbers) + 1))
    graph.add_edges_from(bonds)
    if set(graph) != set(range(1, len(numbers) + 1)):
        return ()
    if any(degree != 3 for _, degree in graph.degree()) or not nx.is_connected(graph):
        return ()
    planar, embedding = nx.check_planarity(graph)
    if not planar:
        return ()
    seen = set()
    faces = []
    for left in sorted(embedding):
        for right in embedding.neighbors_cw_order(left):
            if (left, right) in seen:
                continue
            face = tuple(embedding.traverse_face(left, right, seen))
            if len(face) not in {5, 6} or len(set(face)) != len(face):
                return ()
            variants = [order[i:] + order[:i]
                        for order in (face, face[::-1]) for i in range(len(face))]
            faces.append(min(variants))
    if len(numbers) - graph.number_of_edges() + len(faces) != 2:
        return ()
    return tuple(sorted(faces))


def _candidate_from_primitive(
    primitive: Primitive,
    coordinates: np.ndarray,
    *,
    candidate_id: str,
    function: str,
    family: str,
    units: str,
    owner_id: str,
    domain_id: str,
    provenance: str,
    refs: tuple[str, ...] = (),
) -> PrimitiveCandidate:
    entries = sparse_grad_primitive_entries(primitive, coordinates)
    values = tuple(value for _column, value in entries)
    if not values or not np.all(np.isfinite(values)) or float(np.linalg.norm(values)) <= 1.0e-12:
        raise ValueError(f"ORACLE primitive {candidate_id} has no finite Wilson row")
    large_sparse = coordinates.shape[0] >= 256
    dense_row = np.zeros(0 if large_sparse else coordinates.size, dtype=float)
    if not large_sparse:
        for column, value in entries:
            dense_row[column] = value
    return PrimitiveCandidate(
        candidate_id=candidate_id,
        function=function,
        atoms=tuple(atom + 1 for atom in primitive.atoms),
        mode=primitive.mode,
        ref_atoms=tuple(atom + 1 for atom in primitive.ref),
        refs=tuple(str(value) for value in refs),
        family=family,
        units=units,
        owner_id=owner_id,
        domain_id=domain_id,
        provenance=provenance,
        frame_atoms=tuple(atom + 1 for atom in primitive.frame_atoms),
        ref_frame_atoms=tuple(atom + 1 for atom in primitive.ref_frame_atoms),
        analytic_wilson_row=tuple(float(value) for value in dense_row),
        analytic_wilson_indices=(
            tuple(column for column, _value in entries) if large_sparse else ()
        ),
        analytic_wilson_values=values if large_sparse else (),
        wilson_column_count=int(coordinates.size) if large_sparse else 0,
    )


def _finite_primary_candidate(
    primitive: Primitive,
    coordinates: np.ndarray,
    **metadata,
) -> PrimitiveCandidate | None:
    """Discard only mathematically undefined generated valence candidates."""

    try:
        return _candidate_from_primitive(primitive, coordinates, **metadata)
    except ValueError as exc:
        if "has no finite Wilson row" not in str(exc):
            raise
        return None


def _structural_sites_and_candidates(
    path: Path,
    coordinates: np.ndarray,
    topology: PrimaryTopology,
    *,
    candidate_offset: int,
) -> tuple[tuple[StructuralSite, ...], tuple[PrimitiveCandidate, ...]]:
    definition = read_interaction_center_definition(Path(path))
    fragment_by_atom = {
        atom: fragment.fragment_id
        for fragment in topology.fragments
        for atom in fragment.atoms
    }
    sites = []
    for record in definition.centers:
        radius = float(
            np.mean(
                [
                    qualified_vdw_radius(topology.atomic_numbers[atom - 1])[0]
                    for atom in record.atoms
                ]
            )
        )
        sites.append(
            StructuralSite(
                site_id=record.identifier,
                kind=record.kind,
                members=record.atoms,
                fragment_ids=tuple(
                    dict.fromkeys(fragment_by_atom[atom] for atom in record.atoms)
                ),
                center_angstrom=tuple(float(value) for value in record.center),
                frame=fragment_local_frame(coordinates, record.atoms),
                exposed=True,
                effective_radius_angstrom=radius,
                provider="ORACLE_INTERACTION_CENTERS",
                provider_version="2",
                provenance=f"{record.source}:MEMBERWISE_QUALIFIED_VDW_MEAN",
            )
        )
    site_by_id = {site.site_id: site for site in sites}
    candidates = []
    for interaction in definition.interactions:
        site = site_by_id[interaction.center_id]
        primitive = Primitive(
            "frag_atom_dist",
            tuple(atom - 1 for atom in site.members),
            ref=(interaction.atom - 1,),
        )
        candidates.append(
            _candidate_from_primitive(
                primitive,
                coordinates,
                candidate_id=f"P{candidate_offset + len(candidates) + 1:05d}",
                function="CENTER_ATOM_DIST",
                family="CENTER_ATOM_DISTANCE",
                units="ANGSTROM",
                owner_id=site.site_id,
                domain_id="STRUCTURAL_SITES",
                provenance=f"ORACLE_INTERACTION_CENTERS@2:{interaction.source}",
                refs=(site.site_id,),
            )
        )
    return tuple(sites), tuple(candidates)


def _contact_distance_candidate(
    contact,
    coordinates: np.ndarray,
    sites: dict[str, StructuralSite],
    *,
    candidate_id: str,
) -> PrimitiveCandidate | None:
    left, right = contact.endpoint_a, contact.endpoint_b
    if left.kind == right.kind == "ATOM":
        primitive = Primitive(
            "hbond_dist",
            (int(left.identifier) - 1, int(right.identifier) - 1),
        )
        function = "R"
    elif left.kind == "ATOM" and right.kind == "STRUCTURAL_SITE":
        site = sites[right.identifier]
        primitive = Primitive(
            "frag_atom_dist",
            tuple(atom - 1 for atom in site.members),
            ref=(int(left.identifier) - 1,),
        )
        function = "FCA_DIST"
    elif left.kind == "STRUCTURAL_SITE" and right.kind == "ATOM":
        site = sites[left.identifier]
        primitive = Primitive(
            "frag_atom_dist",
            tuple(atom - 1 for atom in site.members),
            ref=(int(right.identifier) - 1,),
        )
        function = "FCA_DIST"
    else:
        return None
    return _candidate_from_primitive(
        primitive,
        coordinates,
        candidate_id=candidate_id,
        function=function,
        family=f"{contact.kind}_DISTANCE",
        units="ANGSTROM",
        owner_id=contact.contact_id,
        domain_id="AUXILIARY_CONTACTS",
        provenance=f"{contact.provider}@{contact.provider_version}:{contact.provenance}",
        refs=(site.site_id,) if function == "FCA_DIST" else (),
    )


def _fragment_pose_candidates(
    coordinates: np.ndarray,
    fragments: tuple[FragmentMembership, ...],
    *,
    candidate_offset: int,
    symmetry_permutations: tuple[tuple[int, ...], ...] = (),
) -> tuple[PrimitiveCandidate, ...]:
    candidates = []
    if len(fragments) < 2:
        return ()
    from .coordinate_atlas import _body_prescriptions
    bodies = {body.body_id: body for body in _body_prescriptions(fragments, coordinates)}
    reference = next(f for f in fragments if bodies[f.fragment_id].pose_role == "REFERENCE")
    memberships = {frozenset(f.atoms): f.fragment_id for f in fragments}
    reference_dimension = bodies[reference.fragment_id].dimension
    reference_anchors = fragment_frame_anchor_atoms(coordinates, reference.atoms)
    if reference_dimension == "LINEAR":
        reference_anchors = reference_anchors[:1]
    for moving in fragments:
        if moving.fragment_id == reference.fragment_id:
            continue
        owner = f"FRAGMENT_PAIR:{reference.fragment_id}|{moving.fragment_id}"
        provenance = (
            f"{ORACLE_SONIC_CONTRACT_BUILDER}@{ORACLE_SONIC_CONTRACT_BUILDER_VERSION}:"
            "ANALYTIC_FINITE_DISTANCE_POSE_POOL"
        )
        specifications: list[tuple[Primitive, str, str, tuple[str, ...]]] = [
            (
                Primitive(
                    "frag_dist",
                    tuple(atom - 1 for atom in moving.atoms),
                    ref=tuple(atom - 1 for atom in reference.atoms),
                ),
                "FC_DIST",
                "FRAG_DISTANCE",
                (moving.fragment_id, reference.fragment_id),
            )
        ]
        moving_anchors = fragment_frame_anchor_atoms(coordinates, moving.atoms)
        moving_dimension = bodies[moving.fragment_id].dimension
        if moving_dimension == "LINEAR":
            moving_anchors = moving_anchors[:1]
        translation_kind = "frag_linear_trans" if reference_dimension == "LINEAR" else "frag_trans"
        for axis in range(2 if reference_dimension == "LINEAR" else 3):
            specifications.append((Primitive(translation_kind, tuple(a-1 for a in moving.atoms), axis,
                tuple(a-1 for a in reference.atoms), ref_frame_atoms=tuple(a-1 for a in reference_anchors)
                if reference_dimension != "POINT" else ()),
                "FLIN_TRANS" if reference_dimension == "LINEAR" else "FTRANS", "FRAG_TRANSLATION",
                (moving.fragment_id, reference.fragment_id)))
        if reference_dimension == moving_dimension == "LINEAR":
            for axis in range(2):
                specifications.append((Primitive("frag_axis", tuple(a-1 for a in moving.atoms), axis,
                    tuple(a-1 for a in reference.atoms), tuple(a-1 for a in moving_anchors),
                    tuple(a-1 for a in reference_anchors),
                    tuple(tuple(float(v) for v in coordinates[a-1])
                          for a in sorted(set(reference.atoms + moving.atoms)))), "FAXIS", "FRAG_ORIENTATION",
                    (moving.fragment_id, reference.fragment_id)))
        if reference_dimension == "NONLINEAR" and moving_dimension != "POINT":
            support = sorted(set(reference.atoms + moving.atoms))
            for mode in range(2 if moving_dimension == "LINEAR" else 3):
                specifications.append((
                    Primitive(
                        "frag_rot", tuple(a - 1 for a in moving.atoms), mode,
                        tuple(a - 1 for a in reference.atoms),
                        tuple(a - 1 for a in moving_anchors),
                        tuple(a - 1 for a in reference_anchors),
                        tuple(tuple(float(v) for v in coordinates[a - 1]) for a in support),
                    ),
                    "FROT", "FRAG_ORIENTATION", (moving.fragment_id, reference.fragment_id),
                ))
        specifications.extend(
            (
                Primitive(
                    "frag_atom_dist",
                    tuple(atom - 1 for atom in moving.atoms),
                    ref=(atom - 1,),
                ),
                "FCA_DIST",
                "FRAG_CENTER_ATOM_DISTANCE",
                (moving.fragment_id,),
            )
            for atom in reference_anchors
        )
        specifications.extend(
            (
                Primitive(
                    "frag_atom_dist",
                    tuple(atom - 1 for atom in reference.atoms),
                    ref=(atom - 1,),
                ),
                "FCA_DIST",
                "FRAG_CENTER_ATOM_DISTANCE",
                (reference.fragment_id,),
            )
            for atom in moving_anchors
        )
        specifications.extend(
            (
                Primitive("bond", (left - 1, right - 1)),
                "R",
                "FRAGMENT_POSE_DISTANCE",
                (),
            )
            for left in reference_anchors
            for right in moving_anchors
        )
        pose_specs = []
        for primitive, function, family, refs in specifications:
            if function not in {"FROT", "FTRANS", "FLIN_TRANS", "FAXIS"}:
                continue
            for permutation in symmetry_permutations:
                mapped_atoms = tuple(permutation[a]-1 for a in primitive.atoms)
                mapped_ref = tuple(permutation[a]-1 for a in primitive.ref)
                mapped_support = sorted(set(mapped_atoms+mapped_ref))
                mapped = replace(primitive, atoms=mapped_atoms, ref=mapped_ref,
                    frame_atoms=tuple(permutation[a]-1 for a in primitive.frame_atoms),
                    ref_frame_atoms=tuple(permutation[a]-1 for a in primitive.ref_frame_atoms),
                    rotation_reference=tuple(tuple(float(v) for v in coordinates[a]) for a in mapped_support)
                    if function in {"FROT", "FAXIS"} else ())
                mapped_refs = (memberships.get(frozenset(a+1 for a in mapped_atoms), refs[0]),
                               memberships.get(frozenset(a+1 for a in mapped_ref), refs[1]))
                pose_specs.append((mapped,function,family,mapped_refs))
        specifications.extend(pose_specs)
        seen = set()
        for primitive, function, family, refs in specifications:
            signature = (function, tuple(primitive.atoms), tuple(primitive.ref), primitive.mode,
                         primitive.frame_atoms, primitive.ref_frame_atoms)
            if signature in seen:
                continue
            seen.add(signature)
            candidates.append(
                _candidate_from_primitive(
                    primitive,
                    coordinates,
                    candidate_id=f"P{candidate_offset + len(candidates) + 1:05d}",
                    function=function,
                    family=family,
                    units=("DIMENSIONLESS" if function == "FAXIS" or
                           (function == "FROT" and len(primitive.frame_atoms) == 1)
                           else "RADIAN" if function == "FROT" else "ANGSTROM"),
                    owner_id=owner,
                    domain_id="FRAGMENT_POSE",
                    provenance=provenance,
                    refs=refs,
                )
            )
    return tuple(candidates)


def _primary_family(primitive: Primitive) -> str:
    return {
        "bond": "STRETCH",
        "angle": "BEND",
        "linear_bend": "LINEAR_BEND",
        "dihedral": "TORSION",
        "out_of_plane": "OUT_OF_PLANE",
        "out_of_plane_height": "OUT_OF_PLANE_HEIGHT",
    }.get(primitive.kind, primitive.kind.upper())


def _ring_membership_index(rings):
    by_atom = {}
    for index, ring in enumerate(rings, start=1):
        for atom in ring:
            by_atom.setdefault(atom, set()).add(index)
    return by_atom


def _ring_edge_index(rings):
    """Identify cyclic bonds, not merely atoms that happen to belong to rings."""
    by_edge = {}
    for index, ring in enumerate(rings, start=1):
        for left, right in zip(ring, ring[1:] + ring[:1]):
            by_edge.setdefault(tuple(sorted((left, right))), set()).add(index)
    return by_edge


def _primary_coordinate_domain(
    primitive: Primitive,
    rings: tuple[tuple[int, ...], ...],
    *,
    ring_index=None,
    ring_edges=None,
) -> str:
    """Return a topology-local compatibility domain for a valence primitive."""

    atoms = tuple(atom + 1 for atom in primitive.atoms)
    if ring_index is None:
        ring_index = _ring_membership_index(rings)
    if ring_edges is None:
        ring_edges = _ring_edge_index(rings)
    if primitive.kind == "bond" and len(atoms) == 2:
        return f"PRIMARY_TOPOLOGY::BOND:{min(atoms)}-{max(atoms)}"
    if primitive.kind in {"angle", "linear_bend"} and len(atoms) == 3:
        memberships = [ring_index.get(atom, set()) for atom in primitive.atoms]
        ring_owners = sorted(set.intersection(*memberships)) if memberships else []
        if len(ring_owners) == 1:
            return f"PRIMARY_TOPOLOGY::RING:{ring_owners[0]}::BEND"
        return f"PRIMARY_TOPOLOGY::CENTER:{atoms[1]}::BEND"
    if primitive.kind == "dihedral" and len(atoms) == 4:
        center = tuple(sorted(primitive.atoms[1:3]))
        owners = sorted(ring_edges.get(center, ()))
        role = "ENDO_TORSION" if owners else "EXO_TORSION" if (
            ring_index.get(center[0]) or ring_index.get(center[1])
        ) else "TORSION"
        return f"PRIMARY_TOPOLOGY::BOND:{center[0]+1}-{center[1]+1}::{role}"
    if primitive.kind in {"out_of_plane", "out_of_plane_height"} and atoms:
        center, moving = primitive.atoms[0], primitive.atoms[3]
        owners = sorted(ring_edges.get(tuple(sorted((center, moving))), ()))
        role = "ENDO_OUT_OF_PLANE" if owners else "EXO_OUT_OF_PLANE" if ring_index.get(center) else "OUT_OF_PLANE"
        return f"PRIMARY_TOPOLOGY::CENTER:{center+1}::{role}"
    return "PRIMARY_TOPOLOGY::GLOBAL"


def _primary_chemical_role_refs(
    primitive: Primitive,
    numbers: tuple[int, ...],
) -> tuple[str, ...]:
    """Attach periodic-role semantics without changing primitive families."""

    if primitive.kind != "bond" or len(primitive.atoms) != 2:
        return ()
    left, right = primitive.atoms
    left_number, right_number = numbers[left], numbers[right]
    if is_chalcogen_linkage(left_number, right_number):
        return ("CHEMICAL_ROLE=CHALCOGEN_LINKAGE",)
    if (
        is_structural_center(left_number) and is_bridging_ligand(right_number)
    ) or (
        is_structural_center(right_number) and is_bridging_ligand(left_number)
    ):
        return ("CHEMICAL_ROLE=STRUCTURAL_CENTER_LIGAND",)
    if admits_dative_pair(left_number, right_number) or admits_dative_pair(
        right_number, left_number
    ):
        return ("CHEMICAL_ROLE=DATIVE_DONOR_ACCEPTOR",)
    return ()


def _primitive_units(function: str) -> str:
    return (
        "ANGSTROM"
        if function in {"R", "H", "FC_DIST", "FCA_DIST", "CENTER_ATOM_DIST"}
        else "RADIAN"
    )


def _candidate_signature(
    function: str,
    atoms: tuple[int, ...],
) -> tuple[str, tuple[int, ...]]:
    values = tuple(int(atom) for atom in atoms)
    if function == "R":
        values = tuple(sorted(values))
    elif function == "A" and len(values) == 3:
        values = min(values, tuple(reversed(values)))
    elif function == "D" and len(values) == 4:
        values = min(values, tuple(reversed(values)))
    return function, values


def _next_candidate_id(candidates: list[PrimitiveCandidate]) -> str:
    return f"P{len(candidates) + 1:05d}"


__all__ = [
    "ORACLE_SONIC_CONTRACT_BUILDER",
    "ORACLE_SONIC_CONTRACT_BUILDER_VERSION",
    "build_oracle_sonic_contract_from_xyzin",
    "write_oracle_sonic_contract_from_xyzin",
]
