"""Symmetry-complete auxiliary-contact orbits and closure classification."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

from matrix_chem import (
    AuxiliaryContact,
    ContactEndpoint,
    FragmentMembership,
    PrimaryTopology,
    StructuralSite,
)

from .auxiliary_contacts import AuxiliaryContactEvidence


ORACLE_CONTACT_GRAPH_SCHEMA = "matrix.oracle.auxiliary_contact_graph.v1"


@dataclass(frozen=True)
class ClassifiedAuxiliaryContacts:
    contacts: tuple[AuxiliaryContact, ...]
    primary_cycle_rank: int
    auxiliary_cycle_rank: int
    orbit_deltas: tuple[tuple[str, int], ...]


def complete_and_classify_contact_orbits(
    evidence: Iterable[AuxiliaryContactEvidence],
    primary_topology: PrimaryTopology,
    *,
    structural_sites: tuple[StructuralSite, ...] = (),
) -> ClassifiedAuxiliaryContacts:
    """Complete molecular-symmetry orbits and classify whole orbits as open/closing.

    Equivalent contacts are never reduced to a spanning tree.  Orbits are
    processed deterministically by provider, family, decreasing confidence,
    and their canonical endpoint orbit; an entire orbit is added at once.
    """

    fragment_by_atom = _fragment_by_atom(primary_topology.fragments, primary_topology.natoms)
    sites = {site.site_id: site for site in structural_sites}
    permutations = primary_topology.symmetry_permutations or (
        tuple(range(1, primary_topology.natoms + 1)),
    )
    site_lookup = {
        (site.kind, tuple(sorted(site.members))): site.site_id for site in structural_sites
    }
    grouped: dict[
        tuple[str, str, str, tuple[tuple[tuple[str, str], ...], ...]],
        list[AuxiliaryContactEvidence],
    ] = {}
    for item in evidence:
        endpoint_orbit = _endpoint_pair_orbit(item, permutations, sites, site_lookup)
        key = (item.kind, item.provider, item.provider_version, endpoint_orbit)
        grouped.setdefault(key, []).append(item)

    ordered_groups = sorted(
        grouped.items(),
        key=lambda item: (
            item[0][1],
            item[0][0],
            -max(record.confidence for record in item[1]),
            item[0][3],
        ),
    )
    connectivity = _IncrementalContactGraph(primary_topology.fragments)
    contacts: list[AuxiliaryContact] = []
    orbit_deltas: list[tuple[str, int]] = []
    for orbit_index, ((_kind, _provider, _version, endpoint_orbit), records) in enumerate(
        ordered_groups, start=1
    ):
        orbit_id = f"OC{orbit_index:04d}"
        complete = _complete_evidence_orbit(
            endpoint_orbit,
            tuple(records),
            permutations,
            sites,
            site_lookup,
        )
        orbit_edges = [
            _fragment_edge(record.endpoint_a, record.endpoint_b, fragment_by_atom, sites)
            for record in complete
        ]
        delta = sum(connectivity.add_edge(left, right) for left, right in orbit_edges)
        policy = "OPEN" if delta == 0 else "CLOSING"
        orbit_deltas.append((orbit_id, delta))
        for record in complete:
            contact_id = f"C{len(contacts) + 1:04d}"
            fragment_ids = tuple(
                dict.fromkeys(
                    (
                        _endpoint_fragment(record.endpoint_a, fragment_by_atom, sites),
                        _endpoint_fragment(record.endpoint_b, fragment_by_atom, sites),
                    )
                )
            )
            contacts.append(
                AuxiliaryContact(
                    contact_id=contact_id,
                    kind=record.kind,
                    endpoint_a=record.endpoint_a,
                    endpoint_b=record.endpoint_b,
                    rho_vdw=record.rho_vdw,
                    distance_angstrom=record.distance_angstrom,
                    directional_descriptors=record.directional_descriptors,
                    confidence=record.confidence,
                    persistence=record.persistence,
                    provider=record.provider,
                    provider_version=record.provider_version,
                    fragment_ids=fragment_ids,
                    symmetry_orbit_id=orbit_id,
                    delta_beta1_if_added=delta,
                    open_or_closing=policy,
                    primitive_candidate_ids=(),
                    provenance=(
                        f"{record.provenance}:{ORACLE_CONTACT_GRAPH_SCHEMA}:"
                        f"{record.applicability_range}"
                    ),
                )
            )
    auxiliary_rank = primary_topology.cycle_rank + sum(delta for _orbit, delta in orbit_deltas)
    return ClassifiedAuxiliaryContacts(
        contacts=tuple(contacts),
        primary_cycle_rank=primary_topology.cycle_rank,
        auxiliary_cycle_rank=auxiliary_rank,
        orbit_deltas=tuple(orbit_deltas),
    )


def _endpoint_pair_orbit(
    record: AuxiliaryContactEvidence,
    permutations: tuple[tuple[int, ...], ...],
    sites: dict[str, StructuralSite],
    site_lookup: dict[tuple[str, tuple[int, ...]], str],
) -> tuple[tuple[tuple[str, str], ...], ...]:
    return tuple(
        sorted(
            {
                _contact_key(
                    _mapped_endpoint(record.endpoint_a, permutation, sites, site_lookup),
                    _mapped_endpoint(record.endpoint_b, permutation, sites, site_lookup),
                )
                for permutation in permutations
            }
        )
    )


def _complete_evidence_orbit(
    endpoint_orbit: tuple[tuple[tuple[str, str], ...], ...],
    records: tuple[AuxiliaryContactEvidence, ...],
    permutations: tuple[tuple[int, ...], ...],
    sites: dict[str, StructuralSite],
    site_lookup: dict[tuple[str, tuple[int, ...]], str],
) -> tuple[AuxiliaryContactEvidence, ...]:
    by_key = {_contact_key(item.endpoint_a, item.endpoint_b): item for item in records}
    representative = max(records, key=lambda item: (item.confidence, item.persistence))
    result = []
    for key in endpoint_orbit:
        existing = by_key.get(key)
        if existing is not None:
            result.append(existing)
            continue
        mapped = None
        for permutation in permutations:
            left = _mapped_endpoint(representative.endpoint_a, permutation, sites, site_lookup)
            right = _mapped_endpoint(representative.endpoint_b, permutation, sites, site_lookup)
            if _contact_key(left, right) == key:
                mapped = replace(
                    representative,
                    endpoint_a=left,
                    endpoint_b=right,
                    directional_descriptors=_mapped_directional_descriptors(
                        representative.directional_descriptors, permutation
                    ),
                    provenance=f"{representative.provenance}:SYMMETRY_ORBIT_COMPLETION",
                )
                break
        if mapped is None:
            raise ValueError("failed to complete an auxiliary-contact symmetry orbit")
        result.append(mapped)
    return tuple(result)


def _mapped_directional_descriptors(
    descriptors: tuple[tuple[str, float], ...], permutation: tuple[int, ...]
) -> tuple[tuple[str, float], ...]:
    result = []
    for key, value in descriptors:
        if "ATOM" in key and float(value).is_integer() and 1 <= int(value) <= len(permutation):
            result.append((key, float(permutation[int(value) - 1])))
        else:
            result.append((key, value))
    return tuple(result)


def _mapped_endpoint(
    endpoint: ContactEndpoint,
    permutation: tuple[int, ...],
    sites: dict[str, StructuralSite],
    site_lookup: dict[tuple[str, tuple[int, ...]], str],
) -> ContactEndpoint:
    if endpoint.kind == "ATOM":
        return ContactEndpoint("ATOM", str(permutation[int(endpoint.identifier) - 1]))
    site = sites.get(endpoint.identifier)
    if site is None:
        raise ValueError(f"unknown structural site in contact orbit: {endpoint.identifier}")
    members = tuple(sorted(permutation[atom - 1] for atom in site.members))
    mapped_id = site_lookup.get((site.kind, members))
    if mapped_id is None:
        raise ValueError(f"structural site {endpoint.identifier} has an incomplete symmetry orbit")
    return ContactEndpoint("STRUCTURAL_SITE", mapped_id)


def _fragment_by_atom(
    fragments: tuple[FragmentMembership, ...], natoms: int
) -> dict[int, str]:
    result = {
        atom: fragment.fragment_id for fragment in fragments for atom in fragment.atoms
    }
    if set(result) != set(range(1, natoms + 1)):
        raise ValueError("primary fragments must partition all atoms before contact classification")
    return result


def _endpoint_fragment(
    endpoint: ContactEndpoint,
    fragment_by_atom: dict[int, str],
    sites: dict[str, StructuralSite],
) -> str:
    if endpoint.kind == "ATOM":
        return fragment_by_atom[int(endpoint.identifier)]
    site = sites[endpoint.identifier]
    if len(site.fragment_ids) != 1:
        raise ValueError(
            f"structural site {site.site_id} must have one primary fragment for cycle classification"
        )
    return site.fragment_ids[0]


def _fragment_edge(
    left: ContactEndpoint,
    right: ContactEndpoint,
    fragment_by_atom: dict[int, str],
    sites: dict[str, StructuralSite],
) -> tuple[str, str]:
    return (
        _endpoint_fragment(left, fragment_by_atom, sites),
        _endpoint_fragment(right, fragment_by_atom, sites),
    )


class _IncrementalContactGraph:
    """Union by size/path compression; retain parallel edges and self loops."""

    def __init__(self, fragments):
        self.parent = {fragment.fragment_id: fragment.fragment_id for fragment in fragments}
        self.size = {node: 1 for node in self.parent}

    def find(self, node):
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def add_edge(self, left, right):
        if left not in self.parent or right not in self.parent:
            raise ValueError("auxiliary contact references an unknown primary fragment")
        left, right = self.find(left), self.find(right)
        if left == right:
            return 1
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]
        return 0


def _multigraph_cycle_rank(
    fragments: tuple[FragmentMembership, ...], edges: Iterable[tuple[str, str]]
) -> int:
    nodes = tuple(fragment.fragment_id for fragment in fragments)
    parent = {node: node for node in nodes}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    edge_list = []
    for left, right in edges:
        if left not in parent or right not in parent:
            raise ValueError("auxiliary contact references an unknown primary fragment")
        edge_list.append((left, right))
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left
    components = len({find(node) for node in nodes})
    return len(edge_list) - len(nodes) + components


def _contact_key(
    left: ContactEndpoint, right: ContactEndpoint
) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(((left.kind, left.identifier), (right.kind, right.identifier))))


__all__ = [
    "ClassifiedAuxiliaryContacts",
    "ORACLE_CONTACT_GRAPH_SCHEMA",
    "complete_and_classify_contact_orbits",
]
