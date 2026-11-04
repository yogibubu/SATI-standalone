"""Frozen ORACLE-to-SMITH contract for topology-backed SONIC construction.

ORACLE is the semantic owner and sole producer of this contract.  The data
classes live in :mod:`matrix_chem` because both ORACLE and SMITH depend on that
shared package; placing the transport schema here avoids a reverse SMITH ->
ORACLE package dependency without transferring chemical ownership to SMITH.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from matrix_core import read_sectioned_lines, replace_section, section_content

from .coordinate_library import CoordinateComponent, coordinate_selection_units
from .soop_contract import SelectiveOutOfPlanePrimitive, soop_from_dict, validate_soop_primitives
from .perception_contract import (
    FrozenPerceptionHandoff,
    PerceptionHistory,
    PerceptionNoiseAudit,
    frozen_perception_handoff_from_dict,
    perception_history_from_dict,
    perception_noise_audit_from_dict,
)
from .geometry_identity import (
    GeometryIdentityError,
    geometry_identity_payload_sha256,
    read_geometry_identity_certificate,
)


ORACLE_SONIC_CONTRACT_SCHEMA_V1 = "matrix.oracle_sonic_contract.v1"
ORACLE_SONIC_CONTRACT_SCHEMA = "matrix.oracle_sonic_contract.v2"
SUPPORTED_ORACLE_SONIC_CONTRACT_SCHEMAS = (
    ORACLE_SONIC_CONTRACT_SCHEMA_V1,
    ORACLE_SONIC_CONTRACT_SCHEMA,
)
ORACLE_SONIC_CONTRACT_SECTION = "ORACLE_SONIC_CONTRACT"
ORACLE_SONIC_CONTRACT_OWNER = "ORACLE"
ORACLE_PRIMITIVE_MODES = frozenset({"SIMPLE_PRIMITIVES", "CHEMICAL_PRIMITIVES"})


class OracleSonicContractError(ValueError):
    """Raised when the frozen ORACLE-to-SMITH contract is inconsistent."""


@dataclass(frozen=True)
class FragmentMembership:
    fragment_id: str
    atoms: tuple[int, ...]


@dataclass(frozen=True)
class PrimaryTopology:
    natoms: int
    atomic_numbers: tuple[int, ...]
    bonds: tuple[tuple[int, int], ...]
    rings: tuple[tuple[int, ...], ...]
    fragments: tuple[FragmentMembership, ...]
    topology_hash: str
    cycle_rank: int
    symmetry_permutations: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True)
class MulticenterDomain:
    domain_id: str
    kind: str
    atoms: tuple[int, ...]
    provider: str
    provider_version: str
    primitive_candidate_ids: tuple[str, ...]
    provenance: str
    descriptors: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class StructuralSite:
    site_id: str
    kind: str
    members: tuple[int, ...]
    fragment_ids: tuple[str, ...]
    center_angstrom: tuple[float, float, float]
    frame: tuple[tuple[float, float, float], ...]
    exposed: bool
    effective_radius_angstrom: float | None
    provider: str
    provider_version: str
    provenance: str


@dataclass(frozen=True)
class ContactEndpoint:
    kind: str
    identifier: str


@dataclass(frozen=True)
class AuxiliaryContact:
    contact_id: str
    kind: str
    endpoint_a: ContactEndpoint
    endpoint_b: ContactEndpoint
    rho_vdw: float
    distance_angstrom: float
    directional_descriptors: tuple[tuple[str, float], ...]
    confidence: float
    persistence: float
    provider: str
    provider_version: str
    fragment_ids: tuple[str, ...]
    symmetry_orbit_id: str
    delta_beta1_if_added: int
    open_or_closing: str
    primitive_candidate_ids: tuple[str, ...]
    provenance: str


@dataclass(frozen=True)
class PrimitiveCandidate:
    candidate_id: str
    function: str
    atoms: tuple[int, ...]
    mode: int
    ref_atoms: tuple[int, ...]
    refs: tuple[str, ...]
    family: str
    units: str
    owner_id: str
    domain_id: str
    provenance: str
    analytic_wilson_row: tuple[float, ...] = ()
    analytic_wilson_indices: tuple[int, ...] = ()
    analytic_wilson_values: tuple[float, ...] = ()
    wilson_column_count: int = 0
    frame_atoms: tuple[int, ...] = ()
    ref_frame_atoms: tuple[int, ...] = ()


def primitive_candidate_wilson_row(
    candidate: PrimitiveCandidate,
    natoms: int,
) -> np.ndarray:
    """Return one candidate row, expanding sparse storage only on demand."""

    expected = 3 * int(natoms)
    if candidate.analytic_wilson_row:
        row = np.asarray(candidate.analytic_wilson_row, dtype=float)
        if row.shape != (expected,):
            raise OracleSonicContractError(
                f"candidate {candidate.candidate_id} Wilson row has the wrong size"
            )
        return row
    indices = np.asarray(candidate.analytic_wilson_indices, dtype=int)
    values = np.asarray(candidate.analytic_wilson_values, dtype=float)
    columns = int(candidate.wilson_column_count or expected)
    if columns != expected or indices.shape != values.shape:
        raise OracleSonicContractError(
            f"candidate {candidate.candidate_id} sparse Wilson row has the wrong size"
        )
    if np.any(indices < 0) or np.any(indices >= expected) or len(set(indices.tolist())) != len(indices):
        raise OracleSonicContractError(
            f"candidate {candidate.candidate_id} sparse Wilson indices are invalid"
        )
    row = np.zeros(expected, dtype=float)
    row[indices] = values
    return row


@dataclass(frozen=True)
class LocalEquivalenceClass:
    """One ORACLE-owned local ligand/atom equivalence class."""

    class_id: str
    members: tuple[int, ...]
    centroid_effective_atomic_number: float
    centroid_distance_angstrom: float
    maximum_zeff_spread: float
    maximum_distance_spread_angstrom: float


@dataclass(frozen=True)
class LocalTemplateDecision:
    """Complete best/competitor record for one local coordination decision."""

    selected_template: str | None
    best_template: str | None
    competing_template: str | None
    score: float | None
    competing_score: float | None
    margin: float | None
    status: str
    rms_headroom: float | None
    margin_headroom: float | None
    threshold_sensitivity: str


@dataclass(frozen=True)
class LocalPerceptionDomain:
    """Frozen local equivalence, pseudosymmetry, and template assignment."""

    domain_id: str
    kind: str
    center_atom: int | None
    members: tuple[int, ...]
    equivalence_classes: tuple[LocalEquivalenceClass, ...]
    proposed_group: str
    confidence: str
    operation_count: int
    template_decision: LocalTemplateDecision | None
    thresholds: tuple[tuple[str, float, str], ...]
    provider: str
    provider_version: str
    provenance: str


@dataclass(frozen=True)
class OracleSonicContract:
    schema: str
    owner: str
    primary_topology: PrimaryTopology
    multicenter_domains: tuple[MulticenterDomain, ...]
    structural_sites: tuple[StructuralSite, ...]
    auxiliary_contacts: tuple[AuxiliaryContact, ...]
    primitive_candidates: tuple[PrimitiveCandidate, ...]
    primary_cycle_rank: int
    auxiliary_cycle_rank: int
    provenance: str
    local_perception_domains: tuple[LocalPerceptionDomain, ...] = ()
    robustness_audit: PerceptionNoiseAudit | None = None
    perception_history: PerceptionHistory | None = None
    perception_handoff: FrozenPerceptionHandoff | None = None
    chemical_policy_sha256: str = ""
    reference_geometry_sha256: str = ""
    geometry_identity_payload_sha256: str = ""
    primitive_mode: str = "CHEMICAL_PRIMITIVES"
    selective_out_of_plane_primitives: tuple[SelectiveOutOfPlanePrimitive, ...] = ()

    @property
    def topology_certified(self) -> bool:
        """Whether ORACLE has supplied a stable topology for SMITH consumption."""
        if self.primitive_mode not in ORACLE_PRIMITIVE_MODES:
            return False
        if self.robustness_audit is not None and getattr(self.robustness_audit, "audit_status", "PASS") not in {"PASS", "CERTIFIED", "NOT_RUN"}:
            return False
        uncertain = {"LOW", "UNCERTAIN", "FAIL", "REJECTED", "AMBIGUOUS"}
        return all(str(domain.confidence).upper() not in uncertain for domain in self.local_perception_domains)


def primary_topology_hash(
    atomic_numbers: Iterable[int],
    bonds: Iterable[tuple[int, int]],
    rings: Iterable[tuple[int, ...]],
    fragments: Iterable[FragmentMembership],
) -> str:
    """Return the canonical topology fingerprint shared by ORACLE and SMITH."""

    payload = {
        "atomic_numbers": [int(value) for value in atomic_numbers],
        "bonds": [list(pair) for pair in sorted(_canonical_pairs(bonds))],
        "rings": [list(ring) for ring in sorted(_canonical_rings(rings))],
        "fragments": [
            {"fragment_id": item.fragment_id, "atoms": list(sorted(item.atoms))}
            for item in sorted(fragments, key=lambda value: value.fragment_id)
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def graph_cycle_rank(natoms: int, edges: Iterable[tuple[int, int]]) -> int:
    """Compute ``|E| - |V| + components`` for a one-based undirected graph."""

    count = int(natoms)
    if count < 1:
        raise OracleSonicContractError("primary topology must contain at least one atom")
    pairs = _canonical_pairs(edges)
    parent = list(range(count + 1))

    def find(atom: int) -> int:
        while parent[atom] != atom:
            parent[atom] = parent[parent[atom]]
            atom = parent[atom]
        return atom

    for left, right in pairs:
        if left < 1 or right > count or left == right:
            raise OracleSonicContractError(f"invalid one-based graph edge: {(left, right)}")
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left
    components = len({find(atom) for atom in range(1, count + 1)})
    return len(pairs) - count + components


def validate_oracle_sonic_contract(contract: OracleSonicContract) -> None:
    """Validate ownership, references, ranks, rows, and complete contact orbits."""

    if contract.schema not in SUPPORTED_ORACLE_SONIC_CONTRACT_SCHEMAS:
        raise OracleSonicContractError(f"unsupported contract schema: {contract.schema}")
    if contract.owner != ORACLE_SONIC_CONTRACT_OWNER:
        raise OracleSonicContractError("ORACLE must be the sole contract owner")
    if not contract.provenance:
        raise OracleSonicContractError("contract provenance is required")
    if contract.primitive_mode not in ORACLE_PRIMITIVE_MODES:
        raise OracleSonicContractError(f"unsupported ORACLE primitive mode: {contract.primitive_mode}")
    identity_fingerprints = (
        contract.reference_geometry_sha256,
        contract.geometry_identity_payload_sha256,
    )
    if any(identity_fingerprints) and not all(len(value) == 64 for value in identity_fingerprints):
        raise OracleSonicContractError(
            "geometry identity requires complete 64-character fingerprints"
        )

    topology = contract.primary_topology
    if topology.natoms != len(topology.atomic_numbers) or topology.natoms < 1:
        raise OracleSonicContractError("PRIMARY_TOPOLOGY atom count is inconsistent")
    if any(number < 1 for number in topology.atomic_numbers):
        raise OracleSonicContractError("PRIMARY_TOPOLOGY contains invalid atomic numbers")
    bonds = _canonical_pairs(topology.bonds)
    reactive_soop_bonds = {
        tuple(sorted((candidate.atoms[0], atom)))
        for candidate in contract.primitive_candidates
        if candidate.function == "U"
        and candidate.family == "OUT_OF_PLANE"
        and candidate.domain_id == "PRIMARY_TOPOLOGY::REACTIVE_SUPPORT"
        and len(candidate.atoms) == 4
        for atom in candidate.atoms[1:]
    }
    validate_soop_primitives(
        contract.selective_out_of_plane_primitives,
        topology,
        auxiliary_bonds=reactive_soop_bonds,
    )
    if len(bonds) != len(topology.bonds):
        raise OracleSonicContractError("PRIMARY_TOPOLOGY contains duplicate bonds")
    rank = graph_cycle_rank(topology.natoms, bonds)
    if topology.cycle_rank != rank or contract.primary_cycle_rank != rank:
        raise OracleSonicContractError("PRIMARY_TOPOLOGY cycle rank is inconsistent")
    expected_hash = primary_topology_hash(
        topology.atomic_numbers, bonds, topology.rings, topology.fragments
    )
    if topology.topology_hash != expected_hash:
        raise OracleSonicContractError("PRIMARY_TOPOLOGY hash is inconsistent")

    fragment_ids = _validate_fragments(topology.fragments, topology.natoms)
    permutations = _validate_symmetry_permutations(topology, bonds)
    sites = _validate_sites(contract.structural_sites, topology.natoms, fragment_ids)
    candidates = _validate_primitive_candidates(contract.primitive_candidates, topology.natoms)
    domains = _unique_by_id(contract.multicenter_domains, "domain_id", "multicenter domain")
    contacts = _unique_by_id(contract.auxiliary_contacts, "contact_id", "auxiliary contact")
    _validate_local_perception_domains(
        contract.local_perception_domains,
        topology.natoms,
        schema=contract.schema,
    )
    _validate_perception_robustness_contract(contract)

    for domain in domains.values():
        if not domain.kind or not domain.provider or not domain.provider_version or not domain.provenance:
            raise OracleSonicContractError(f"incomplete multicenter domain {domain.domain_id}")
        _validate_atom_indices(domain.atoms, topology.natoms, f"domain {domain.domain_id}")
        for candidate_id in domain.primitive_candidate_ids:
            candidate = candidates.get(candidate_id)
            if candidate is None or candidate.domain_id != domain.domain_id:
                raise OracleSonicContractError(
                    f"domain {domain.domain_id} has inconsistent primitive ownership"
                )

    primary_bonds = set(bonds)
    orbit_members: dict[str, list[AuxiliaryContact]] = {}
    for contact in contacts.values():
        _validate_contact(
            contact,
            topology=topology,
            sites=sites,
            fragment_ids=fragment_ids,
            candidates=candidates,
            primary_bonds=primary_bonds,
        )
        orbit_members.setdefault(contact.symmetry_orbit_id, []).append(contact)
    for orbit_id, members in orbit_members.items():
        _validate_contact_orbit(orbit_id, tuple(members), permutations, sites)

    orbit_deltas: dict[str, int] = {}
    for contact in contract.auxiliary_contacts:
        previous = orbit_deltas.setdefault(
            contact.symmetry_orbit_id, contact.delta_beta1_if_added
        )
        if previous != contact.delta_beta1_if_added:
            raise OracleSonicContractError(
                f"contact orbit {contact.symmetry_orbit_id} has inconsistent cycle deltas"
            )
    expected_auxiliary_rank = contract.primary_cycle_rank + sum(orbit_deltas.values())
    if contract.auxiliary_cycle_rank != expected_auxiliary_rank:
        raise OracleSonicContractError("auxiliary cycle rank is inconsistent with contact deltas")


def oracle_sonic_contract_to_dict(contract: OracleSonicContract) -> dict[str, Any]:
    validate_oracle_sonic_contract(contract)
    return asdict(contract)


def oracle_sonic_contract_from_dict(payload: dict[str, Any]) -> OracleSonicContract:
    """Materialize and validate a contract from its canonical JSON representation."""

    try:
        primary_data = payload["primary_topology"]
        primary = PrimaryTopology(
            natoms=int(primary_data["natoms"]),
            atomic_numbers=_ints(primary_data["atomic_numbers"]),
            bonds=_pairs(primary_data["bonds"]),
            rings=tuple(_ints(item) for item in primary_data["rings"]),
            fragments=tuple(
                FragmentMembership(str(item["fragment_id"]), _ints(item["atoms"]))
                for item in primary_data["fragments"]
            ),
            topology_hash=str(primary_data["topology_hash"]),
            cycle_rank=int(primary_data["cycle_rank"]),
            symmetry_permutations=tuple(
                _ints(item) for item in primary_data.get("symmetry_permutations", ())
            ),
        )
        contract = OracleSonicContract(
            schema=str(payload["schema"]),
            owner=str(payload["owner"]),
            primary_topology=primary,
            multicenter_domains=tuple(_domain_from_dict(item) for item in payload["multicenter_domains"]),
            structural_sites=tuple(_site_from_dict(item) for item in payload["structural_sites"]),
            auxiliary_contacts=tuple(_contact_from_dict(item) for item in payload["auxiliary_contacts"]),
            primitive_candidates=tuple(
                _candidate_from_dict(item) for item in payload["primitive_candidates"]
            ),
            primary_cycle_rank=int(payload["primary_cycle_rank"]),
            auxiliary_cycle_rank=int(payload["auxiliary_cycle_rank"]),
            provenance=str(payload["provenance"]),
            local_perception_domains=tuple(
                _local_domain_from_dict(item)
                for item in payload.get("local_perception_domains", ())
            ),
            robustness_audit=(
                None
                if payload.get("robustness_audit") is None
                else perception_noise_audit_from_dict(payload["robustness_audit"])
            ),
            perception_history=(
                None
                if payload.get("perception_history") is None
                else perception_history_from_dict(payload["perception_history"])
            ),
            perception_handoff=(
                None
                if payload.get("perception_handoff") is None
                else frozen_perception_handoff_from_dict(payload["perception_handoff"])
            ),
            chemical_policy_sha256=str(payload.get("chemical_policy_sha256", "")),
            reference_geometry_sha256=str(payload.get("reference_geometry_sha256", "")),
            geometry_identity_payload_sha256=str(
                payload.get("geometry_identity_payload_sha256", "")
            ),
            primitive_mode=str(payload.get("primitive_mode", "CHEMICAL_PRIMITIVES")),
            selective_out_of_plane_primitives=tuple(
                soop_from_dict(item) for item in payload.get("selective_out_of_plane_primitives", ())
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise OracleSonicContractError("invalid ORACLE SONIC contract payload") from exc
    validate_oracle_sonic_contract(contract)
    return contract


def migrate_oracle_sonic_contract_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return deterministic structural transport to the current schema.

    A v1 artifact contains no local-perception evidence, so migration records
    an empty tuple rather than reconstructing chemistry from coordinates.
    ORACLE must explicitly rebuild a v2 contract when local decisions are
    required by a new workflow.
    """

    result = json.loads(json.dumps(payload))
    schema = str(result.get("schema", ""))
    if schema == ORACLE_SONIC_CONTRACT_SCHEMA_V1:
        result.setdefault("local_perception_domains", [])
        result.setdefault("robustness_audit", None)
        result.setdefault("perception_history", None)
        result.setdefault("perception_handoff", None)
        result.setdefault("chemical_policy_sha256", "")
        result.setdefault("reference_geometry_sha256", "")
        result.setdefault("geometry_identity_payload_sha256", "")
        provenance = str(result.get("provenance", ""))
        result["provenance"] = (
            f"{provenance}:MIGRATED_V1_TO_V2_NO_LOCAL_RECONSTRUCTION"
            if provenance
            else "MIGRATED_V1_TO_V2_NO_LOCAL_RECONSTRUCTION"
        )
    if schema == ORACLE_SONIC_CONTRACT_SCHEMA_V1:
        result["schema"] = ORACLE_SONIC_CONTRACT_SCHEMA
        for candidate in result.get("primitive_candidates", []):
            candidate.setdefault("analytic_wilson_indices", [])
            candidate.setdefault("analytic_wilson_values", [])
            candidate.setdefault("wilson_column_count", 0)
    elif schema != ORACLE_SONIC_CONTRACT_SCHEMA:
        raise OracleSonicContractError(f"unsupported contract schema: {schema}")
    return result


def oracle_sonic_contract_section_lines(contract: OracleSonicContract) -> list[str]:
    payload = json.dumps(
        oracle_sonic_contract_to_dict(contract), sort_keys=True, separators=(",", ":")
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    chunks = [payload[index : index + 4096] for index in range(0, len(payload), 4096)]
    return [
        f"SCHEMA {contract.schema}",
        f"OWNER {ORACLE_SONIC_CONTRACT_OWNER}",
        "ENCODING CANONICAL_JSON_UTF8",
        f"PAYLOAD_SHA256 {digest}",
        "[PAYLOAD]",
        *chunks,
    ]


def write_oracle_sonic_contract(path: Path, contract: OracleSonicContract) -> None:
    target = Path(path)
    try:
        identity = read_geometry_identity_certificate(target)
    except GeometryIdentityError as exc:
        if "missing #" not in str(exc):
            raise OracleSonicContractError(str(exc)) from exc
    else:
        identity_digest = geometry_identity_payload_sha256(identity)
        if contract.reference_geometry_sha256 not in {
            "",
            identity.canonical_geometry_sha256,
        }:
            raise OracleSonicContractError(
                "SONIC reference geometry contradicts ORACLE geometry provenance"
            )
        if contract.geometry_identity_payload_sha256 not in {"", identity_digest}:
            raise OracleSonicContractError(
                "SONIC geometry-identity fingerprint contradicts ORACLE provenance"
            )
        contract = replace(
            contract,
            reference_geometry_sha256=identity.canonical_geometry_sha256,
            geometry_identity_payload_sha256=identity_digest,
        )
    replace_section(
        target,
        ORACLE_SONIC_CONTRACT_SECTION,
        oracle_sonic_contract_section_lines(contract),
    )


def read_oracle_sonic_contract(path: Path) -> OracleSonicContract:
    content = section_content(read_sectioned_lines(Path(path)), ORACLE_SONIC_CONTRACT_SECTION)
    if not content:
        raise OracleSonicContractError(f"missing #{ORACLE_SONIC_CONTRACT_SECTION} section")
    metadata: dict[str, str] = {}
    chunks: list[str] = []
    in_payload = False
    for raw in content:
        if in_payload:
            # Payload chunks are fixed-width slices of canonical JSON.  A
            # slice may begin or end with whitespace that belongs to a JSON
            # string, so reassembly must preserve every character exactly.
            chunks.append(raw)
            continue
        text = raw.strip()
        if text == "[PAYLOAD]":
            in_payload = True
            continue
        if text:
            fields = text.split(maxsplit=1)
            if len(fields) == 2:
                metadata[fields[0]] = fields[1]
    if metadata.get("SCHEMA") not in SUPPORTED_ORACLE_SONIC_CONTRACT_SCHEMAS:
        raise OracleSonicContractError("unsupported serialized ORACLE SONIC schema")
    if metadata.get("OWNER") != ORACLE_SONIC_CONTRACT_OWNER:
        raise OracleSonicContractError("serialized SONIC contract is not ORACLE-owned")
    payload_text = "".join(chunks)
    digest = hashlib.sha256(payload_text.encode("utf-8")).hexdigest()
    if digest != metadata.get("PAYLOAD_SHA256"):
        raise OracleSonicContractError("ORACLE SONIC payload fingerprint mismatch")
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise OracleSonicContractError("invalid ORACLE SONIC JSON payload") from exc
    return oracle_sonic_contract_from_dict(payload)


def _validate_fragments(
    fragments: tuple[FragmentMembership, ...], natoms: int
) -> set[str]:
    records = _unique_by_id(fragments, "fragment_id", "fragment")
    assigned: set[int] = set()
    for fragment in records.values():
        _validate_atom_indices(fragment.atoms, natoms, f"fragment {fragment.fragment_id}")
        overlap = assigned.intersection(fragment.atoms)
        if overlap:
            raise OracleSonicContractError("PRIMARY_TOPOLOGY fragments overlap")
        assigned.update(fragment.atoms)
    if assigned != set(range(1, natoms + 1)):
        raise OracleSonicContractError("PRIMARY_TOPOLOGY fragments must partition all atoms")
    return set(records)


def _validate_symmetry_permutations(
    topology: PrimaryTopology, bonds: tuple[tuple[int, int], ...]
) -> tuple[tuple[int, ...], ...]:
    identity = tuple(range(1, topology.natoms + 1))
    permutations = topology.symmetry_permutations or (identity,)
    if identity not in permutations or len(set(permutations)) != len(permutations):
        raise OracleSonicContractError("symmetry permutations must be unique and contain identity")
    expected = list(identity)
    bond_set = set(bonds)
    for permutation in permutations:
        if sorted(permutation) != expected:
            raise OracleSonicContractError("invalid PRIMARY_TOPOLOGY symmetry permutation")
        if any(
            topology.atomic_numbers[index - 1]
            != topology.atomic_numbers[permutation[index - 1] - 1]
            for index in identity
        ):
            raise OracleSonicContractError("symmetry permutation changes atomic numbers")
        mapped = {
            tuple(sorted((permutation[left - 1], permutation[right - 1])))
            for left, right in bonds
        }
        if mapped != bond_set:
            raise OracleSonicContractError("symmetry permutation does not preserve topology")
    return permutations


def _validate_sites(
    records: tuple[StructuralSite, ...], natoms: int, fragment_ids: set[str]
) -> dict[str, StructuralSite]:
    sites = _unique_by_id(records, "site_id", "structural site")
    for site in sites.values():
        if not site.kind or not site.provider or not site.provider_version or not site.provenance:
            raise OracleSonicContractError(f"incomplete structural site {site.site_id}")
        _validate_atom_indices(site.members, natoms, f"site {site.site_id}")
        if not set(site.fragment_ids).issubset(fragment_ids):
            raise OracleSonicContractError(f"site {site.site_id} references an unknown fragment")
        if len(site.center_angstrom) != 3 or not _all_finite(site.center_angstrom):
            raise OracleSonicContractError(f"site {site.site_id} has an invalid center")
        if len(site.frame) != 3 or any(len(row) != 3 for row in site.frame):
            raise OracleSonicContractError(f"site {site.site_id} has an invalid local frame")
        if not _all_finite(value for row in site.frame for value in row):
            raise OracleSonicContractError(f"site {site.site_id} has a non-finite frame")
        if site.effective_radius_angstrom is not None and (
            not math.isfinite(site.effective_radius_angstrom)
            or site.effective_radius_angstrom <= 0.0
        ):
            raise OracleSonicContractError(f"site {site.site_id} has an invalid effective radius")
    return sites


def _validate_primitive_candidates(
    records: tuple[PrimitiveCandidate, ...], natoms: int
) -> dict[str, PrimitiveCandidate]:
    candidates = _unique_by_id(records, "candidate_id", "primitive candidate")
    for candidate in candidates.values():
        if candidate.function == "FROT":
            # Older frozen contracts encode the same anchors in refs.
            metadata = dict(item.split("=", 1) for item in candidate.refs if "=" in item)
            frame = candidate.frame_atoms or tuple(int(a) for a in metadata.get("FRAME_ATOMS", "").split(",") if a)
            ref_frame = candidate.ref_frame_atoms or tuple(int(a) for a in metadata.get("REF_FRAME_ATOMS", "").split(",") if a)
            if len(frame) not in (1, 2) or len(ref_frame) != 2:
                raise OracleSonicContractError("FROT requires frozen frame anchors")
            if not set(frame) <= set(candidate.atoms) or not set(ref_frame) <= set(candidate.ref_atoms):
                raise OracleSonicContractError("FROT anchors must belong to their bodies")
            if candidate.mode not in (0, 1, 2):
                raise OracleSonicContractError("invalid FROT component")
        if not all(
            (
                candidate.function,
                candidate.family,
                candidate.units,
                candidate.owner_id,
                candidate.provenance,
            )
        ):
            raise OracleSonicContractError(f"incomplete primitive candidate {candidate.candidate_id}")
        _validate_atom_indices(candidate.atoms, natoms, f"candidate {candidate.candidate_id}")
        _validate_atom_indices(
            candidate.ref_atoms, natoms, f"candidate {candidate.candidate_id} references"
        )
        if candidate.analytic_wilson_row:
            row_valid = bool(
                len(candidate.analytic_wilson_row) == 3 * natoms
                and _all_finite(candidate.analytic_wilson_row)
            )
        else:
            indices = candidate.analytic_wilson_indices
            values = candidate.analytic_wilson_values
            row_valid = bool(
                int(candidate.wilson_column_count or 3 * natoms) == 3 * natoms
                and len(indices) == len(values)
                and tuple(sorted(set(indices))) == indices
                and all(0 <= int(index) < 3 * natoms for index in indices)
                and _all_finite(values)
            )
        if not row_valid:
            raise OracleSonicContractError(
                f"candidate {candidate.candidate_id} Wilson row is invalid"
            )
        if candidate.function in {"FC_DIST", "FTRANS", "FROT"} and len(candidate.refs) < 2:
            raise OracleSonicContractError(
                f"candidate {candidate.candidate_id} requires two typed fragment references"
            )
        if candidate.function in {"FCA_DIST", "CENTER_ATOM_DIST"} and not candidate.refs:
            raise OracleSonicContractError(
                f"candidate {candidate.candidate_id} requires a typed center reference"
            )
        if candidate.function in {"FCA_DIST", "CENTER_ATOM_DIST"} and len(candidate.ref_atoms) != 1:
            raise OracleSonicContractError(
                f"candidate {candidate.candidate_id} requires one reference atom"
            )
    try:
        coordinate_selection_units(
            tuple(
                CoordinateComponent(
                    operator=candidate.function,
                    atoms=candidate.atoms,
                    mode=candidate.mode,
                    ref_atoms=candidate.ref_atoms,
                    context=(
                        candidate.family,
                        candidate.owner_id,
                        candidate.domain_id,
                        *candidate.refs,
                    ),
                )
                for candidate in records
            )
        )
    except ValueError as exc:
        raise OracleSonicContractError(
            f"primitive candidate component invariant failed: {exc}"
        ) from exc
    return candidates


def _validate_local_perception_domains(
    records: tuple[LocalPerceptionDomain, ...],
    natoms: int,
    *,
    schema: str,
) -> None:
    if schema == ORACLE_SONIC_CONTRACT_SCHEMA_V1:
        if records:
            raise OracleSonicContractError("v1 ORACLE contracts cannot contain local perception")
        return
    domains = _unique_by_id(records, "domain_id", "local perception domain")
    for domain in domains.values():
        if domain.kind not in {"ATOM_CENTER", "RING"}:
            raise OracleSonicContractError(
                f"unsupported local perception domain kind: {domain.kind}"
            )
        if not all((domain.proposed_group, domain.confidence, domain.provider,
                    domain.provider_version, domain.provenance)):
            raise OracleSonicContractError(
                f"incomplete local perception domain {domain.domain_id}"
            )
        _validate_atom_indices(domain.members, natoms, f"local domain {domain.domain_id}")
        if domain.kind == "ATOM_CENTER":
            if domain.center_atom is None:
                raise OracleSonicContractError(
                    f"local atom-center domain {domain.domain_id} has no center"
                )
            _validate_atom_indices(
                (domain.center_atom,), natoms, f"local domain {domain.domain_id} center"
            )
            if domain.center_atom in domain.members:
                raise OracleSonicContractError(
                    f"local domain {domain.domain_id} repeats its center as a member"
                )
        elif domain.center_atom is not None:
            raise OracleSonicContractError(
                f"ring domain {domain.domain_id} must not define an atom center"
            )
        if domain.operation_count < 1:
            raise OracleSonicContractError(
                f"local domain {domain.domain_id} has no admitted operation"
            )
        classes = _unique_by_id(
            domain.equivalence_classes, "class_id", "local equivalence class"
        )
        assigned: set[int] = set()
        for equivalence in classes.values():
            _validate_atom_indices(
                equivalence.members,
                natoms,
                f"local equivalence class {equivalence.class_id}",
            )
            if not set(equivalence.members).issubset(domain.members):
                raise OracleSonicContractError(
                    f"local equivalence class {equivalence.class_id} escapes its domain"
                )
            if assigned.intersection(equivalence.members):
                raise OracleSonicContractError(
                    f"local domain {domain.domain_id} has overlapping equivalence classes"
                )
            assigned.update(equivalence.members)
            values = (
                equivalence.centroid_effective_atomic_number,
                equivalence.centroid_distance_angstrom,
                equivalence.maximum_zeff_spread,
                equivalence.maximum_distance_spread_angstrom,
            )
            if not _all_finite(values) or min(values[1:]) < 0.0:
                raise OracleSonicContractError(
                    f"local equivalence class {equivalence.class_id} has invalid evidence"
                )
        if assigned != set(domain.members):
            raise OracleSonicContractError(
                f"local domain {domain.domain_id} equivalence classes do not partition members"
            )
        threshold_names: set[str] = set()
        for name, value, unit in domain.thresholds:
            if not name or name in threshold_names or not unit or not math.isfinite(value):
                raise OracleSonicContractError(
                    f"local domain {domain.domain_id} has invalid threshold metadata"
                )
            threshold_names.add(name)
        decision = domain.template_decision
        if decision is not None:
            if decision.status not in {"FROZEN", "AMBIGUOUS", "OUT_OF_RANGE", "GENERIC"}:
                raise OracleSonicContractError(
                    f"local domain {domain.domain_id} has invalid template status"
                )
            finite_or_none = (
                decision.score,
                decision.competing_score,
                decision.margin,
                decision.rms_headroom,
                decision.margin_headroom,
            )
            if any(value is not None and not math.isfinite(value) for value in finite_or_none):
                raise OracleSonicContractError(
                    f"local domain {domain.domain_id} has non-finite template evidence"
                )
            if decision.status == "FROZEN" and not decision.selected_template:
                raise OracleSonicContractError(
                    f"local domain {domain.domain_id} froze no template"
                )


def _validate_perception_robustness_contract(contract: OracleSonicContract) -> None:
    records = (
        contract.robustness_audit,
        contract.perception_history,
        contract.perception_handoff,
    )
    if contract.schema == ORACLE_SONIC_CONTRACT_SCHEMA_V1:
        if any(item is not None for item in records) or contract.chemical_policy_sha256:
            raise OracleSonicContractError(
                "v1 ORACLE contracts cannot contain robustness or handoff records"
            )
        return
    if contract.chemical_policy_sha256 and (
        len(contract.chemical_policy_sha256) != 64
        or any(value not in "0123456789abcdef" for value in contract.chemical_policy_sha256)
    ):
        raise OracleSonicContractError("chemical perception policy hash is invalid")
    audit = contract.robustness_audit
    history = contract.perception_history
    handoff = contract.perception_handoff
    if history is not None and audit is not None and audit.history_fingerprint not in {
        "",
        history.fingerprint,
    }:
        raise OracleSonicContractError("audit and temporal-history fingerprints disagree")
    if handoff is not None:
        if audit is None:
            raise OracleSonicContractError("frozen exploitation handoff requires an audit")
        if handoff.state_hash != audit.reference_state_hash:
            raise OracleSonicContractError("handoff and audit reference different states")
        if handoff.symmetry_decision != audit.symmetry_decision:
            raise OracleSonicContractError("handoff and audit symmetry decisions disagree")
        if audit.status not in {"ROBUST", "REQUIRES_DECISION"}:
            raise OracleSonicContractError("unstable perception cannot be frozen for exploitation")


def _validate_contact(
    contact: AuxiliaryContact,
    *,
    topology: PrimaryTopology,
    sites: dict[str, StructuralSite],
    fragment_ids: set[str],
    candidates: dict[str, PrimitiveCandidate],
    primary_bonds: set[tuple[int, int]],
) -> None:
    if not all(
        (
            contact.kind,
            contact.provider,
            contact.provider_version,
            contact.symmetry_orbit_id,
            contact.provenance,
        )
    ):
        raise OracleSonicContractError(f"incomplete auxiliary contact {contact.contact_id}")
    _validate_endpoint(contact.endpoint_a, topology.natoms, sites)
    _validate_endpoint(contact.endpoint_b, topology.natoms, sites)
    if contact.endpoint_a == contact.endpoint_b:
        raise OracleSonicContractError(f"contact {contact.contact_id} repeats its endpoint")
    if contact.endpoint_a.kind == contact.endpoint_b.kind == "ATOM":
        pair = tuple(
            sorted((int(contact.endpoint_a.identifier), int(contact.endpoint_b.identifier)))
        )
        if pair in primary_bonds:
            raise OracleSonicContractError(
                f"contact {contact.contact_id} duplicates a PRIMARY_TOPOLOGY bond"
            )
    if not math.isfinite(contact.rho_vdw) or contact.rho_vdw <= 0.0:
        raise OracleSonicContractError(f"contact {contact.contact_id} has invalid rho_vdw")
    if not math.isfinite(contact.distance_angstrom) or contact.distance_angstrom <= 0.0:
        raise OracleSonicContractError(f"contact {contact.contact_id} has invalid distance")
    if not 0.0 <= contact.confidence <= 1.0 or not 0.0 <= contact.persistence <= 1.0:
        raise OracleSonicContractError(f"contact {contact.contact_id} has invalid confidence")
    if not _all_finite(value for _name, value in contact.directional_descriptors):
        raise OracleSonicContractError(f"contact {contact.contact_id} has invalid directionality")
    if not set(contact.fragment_ids).issubset(fragment_ids):
        raise OracleSonicContractError(f"contact {contact.contact_id} references an unknown fragment")
    expected_policy = "OPEN" if contact.delta_beta1_if_added == 0 else "CLOSING"
    if contact.delta_beta1_if_added < 0 or contact.open_or_closing != expected_policy:
        raise OracleSonicContractError(f"contact {contact.contact_id} has inconsistent cycle metadata")
    for candidate_id in contact.primitive_candidate_ids:
        candidate = candidates.get(candidate_id)
        if candidate is None or candidate.owner_id != contact.contact_id:
            raise OracleSonicContractError(
                f"contact {contact.contact_id} has inconsistent primitive ownership"
            )


def _validate_contact_orbit(
    orbit_id: str,
    members: tuple[AuxiliaryContact, ...],
    permutations: tuple[tuple[int, ...], ...],
    sites: dict[str, StructuralSite],
) -> None:
    signatures = {
        (
            item.kind,
            item.provider,
            item.provider_version,
            item.delta_beta1_if_added,
            item.open_or_closing,
        )
        for item in members
    }
    if len(signatures) != 1:
        raise OracleSonicContractError(
            f"contact orbit {orbit_id} mixes typing, provenance, or cycle policy"
        )
    keys = {_contact_key(item.endpoint_a, item.endpoint_b) for item in members}
    site_lookup = {(site.kind, tuple(sorted(site.members))): site.site_id for site in sites.values()}
    for member in members:
        for permutation in permutations:
            mapped_a = _mapped_endpoint(member.endpoint_a, permutation, sites, site_lookup)
            mapped_b = _mapped_endpoint(member.endpoint_b, permutation, sites, site_lookup)
            if _contact_key(mapped_a, mapped_b) not in keys:
                raise OracleSonicContractError(
                    f"contact orbit {orbit_id} is incomplete under molecular symmetry"
                )


def _validate_endpoint(
    endpoint: ContactEndpoint, natoms: int, sites: dict[str, StructuralSite]
) -> None:
    if endpoint.kind == "ATOM":
        try:
            atom = int(endpoint.identifier)
        except ValueError as exc:
            raise OracleSonicContractError("ATOM endpoint identifier must be an integer") from exc
        _validate_atom_indices((atom,), natoms, "contact endpoint")
    elif endpoint.kind == "STRUCTURAL_SITE":
        if endpoint.identifier not in sites:
            raise OracleSonicContractError("contact references an unknown structural site")
    else:
        raise OracleSonicContractError(f"unsupported contact endpoint kind: {endpoint.kind}")


def _mapped_endpoint(
    endpoint: ContactEndpoint,
    permutation: tuple[int, ...],
    sites: dict[str, StructuralSite],
    site_lookup: dict[tuple[str, tuple[int, ...]], str],
) -> ContactEndpoint:
    if endpoint.kind == "ATOM":
        return ContactEndpoint("ATOM", str(permutation[int(endpoint.identifier) - 1]))
    site = sites[endpoint.identifier]
    mapped_members = tuple(sorted(permutation[atom - 1] for atom in site.members))
    mapped_id = site_lookup.get((site.kind, mapped_members))
    if mapped_id is None:
        raise OracleSonicContractError(
            f"structural-site orbit for {endpoint.identifier} is incomplete"
        )
    return ContactEndpoint("STRUCTURAL_SITE", mapped_id)


def _contact_key(left: ContactEndpoint, right: ContactEndpoint) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(((left.kind, left.identifier), (right.kind, right.identifier))))


def _validate_atom_indices(atoms: Iterable[int], natoms: int, label: str) -> None:
    values = tuple(int(atom) for atom in atoms)
    if len(set(values)) != len(values) or any(atom < 1 or atom > natoms for atom in values):
        raise OracleSonicContractError(f"{label} contains invalid atom indices")


def _unique_by_id(records: Iterable[Any], field: str, label: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for record in records:
        identifier = str(getattr(record, field))
        if not identifier or identifier in result:
            raise OracleSonicContractError(f"duplicate or empty {label} identifier: {identifier}")
        result[identifier] = record
    return result


def _canonical_pairs(edges: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted({tuple(sorted((int(left), int(right)))) for left, right in edges}))


def _canonical_rings(rings: Iterable[tuple[int, ...]]) -> tuple[tuple[int, ...], ...]:
    return tuple(sorted(tuple(int(atom) for atom in ring) for ring in rings))


def _all_finite(values: Iterable[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _ints(values: Iterable[Any]) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def _pairs(values: Iterable[Iterable[Any]]) -> tuple[tuple[int, int], ...]:
    return tuple(tuple(_ints(value)) for value in values)  # type: ignore[return-value]


def _domain_from_dict(data: dict[str, Any]) -> MulticenterDomain:
    return MulticenterDomain(
        domain_id=str(data["domain_id"]),
        kind=str(data["kind"]),
        atoms=_ints(data["atoms"]),
        provider=str(data["provider"]),
        provider_version=str(data["provider_version"]),
        primitive_candidate_ids=tuple(str(value) for value in data["primitive_candidate_ids"]),
        provenance=str(data["provenance"]),
        descriptors=tuple((str(key), str(value)) for key, value in data.get("descriptors", ())),
    )


def _site_from_dict(data: dict[str, Any]) -> StructuralSite:
    return StructuralSite(
        site_id=str(data["site_id"]),
        kind=str(data["kind"]),
        members=_ints(data["members"]),
        fragment_ids=tuple(str(value) for value in data["fragment_ids"]),
        center_angstrom=tuple(float(value) for value in data["center_angstrom"]),  # type: ignore[arg-type]
        frame=tuple(tuple(float(value) for value in row) for row in data["frame"]),
        exposed=bool(data["exposed"]),
        effective_radius_angstrom=(
            None
            if data.get("effective_radius_angstrom") is None
            else float(data["effective_radius_angstrom"])
        ),
        provider=str(data["provider"]),
        provider_version=str(data["provider_version"]),
        provenance=str(data["provenance"]),
    )


def _endpoint_from_dict(data: dict[str, Any]) -> ContactEndpoint:
    return ContactEndpoint(str(data["kind"]), str(data["identifier"]))


def _contact_from_dict(data: dict[str, Any]) -> AuxiliaryContact:
    return AuxiliaryContact(
        contact_id=str(data["contact_id"]),
        kind=str(data["kind"]),
        endpoint_a=_endpoint_from_dict(data["endpoint_a"]),
        endpoint_b=_endpoint_from_dict(data["endpoint_b"]),
        rho_vdw=float(data["rho_vdw"]),
        distance_angstrom=float(data["distance_angstrom"]),
        directional_descriptors=tuple(
            (str(key), float(value)) for key, value in data["directional_descriptors"]
        ),
        confidence=float(data["confidence"]),
        persistence=float(data["persistence"]),
        provider=str(data["provider"]),
        provider_version=str(data["provider_version"]),
        fragment_ids=tuple(str(value) for value in data["fragment_ids"]),
        symmetry_orbit_id=str(data["symmetry_orbit_id"]),
        delta_beta1_if_added=int(data["delta_beta1_if_added"]),
        open_or_closing=str(data["open_or_closing"]),
        primitive_candidate_ids=tuple(str(value) for value in data["primitive_candidate_ids"]),
        provenance=str(data["provenance"]),
    )


def _candidate_from_dict(data: dict[str, Any]) -> PrimitiveCandidate:
    return PrimitiveCandidate(
        candidate_id=str(data["candidate_id"]),
        function=str(data["function"]),
        atoms=_ints(data["atoms"]),
        mode=int(data["mode"]),
        ref_atoms=_ints(data["ref_atoms"]),
        refs=tuple(str(value) for value in data["refs"]),
        family=str(data["family"]),
        units=str(data["units"]),
        owner_id=str(data["owner_id"]),
        domain_id=str(data["domain_id"]),
        provenance=str(data["provenance"]),
        analytic_wilson_row=tuple(float(value) for value in data.get("analytic_wilson_row", ())),
        analytic_wilson_indices=_ints(data.get("analytic_wilson_indices", ())),
        analytic_wilson_values=tuple(
            float(value) for value in data.get("analytic_wilson_values", ())
        ),
        wilson_column_count=int(data.get("wilson_column_count", 0)),
        frame_atoms=_ints(data.get("frame_atoms", ())),
        ref_frame_atoms=_ints(data.get("ref_frame_atoms", ())),
    )


def _local_equivalence_from_dict(data: dict[str, Any]) -> LocalEquivalenceClass:
    return LocalEquivalenceClass(
        class_id=str(data["class_id"]),
        members=_ints(data["members"]),
        centroid_effective_atomic_number=float(data["centroid_effective_atomic_number"]),
        centroid_distance_angstrom=float(data["centroid_distance_angstrom"]),
        maximum_zeff_spread=float(data["maximum_zeff_spread"]),
        maximum_distance_spread_angstrom=float(data["maximum_distance_spread_angstrom"]),
    )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _local_template_from_dict(data: dict[str, Any]) -> LocalTemplateDecision:
    return LocalTemplateDecision(
        selected_template=(
            None if data.get("selected_template") is None else str(data["selected_template"])
        ),
        best_template=None if data.get("best_template") is None else str(data["best_template"]),
        competing_template=(
            None if data.get("competing_template") is None else str(data["competing_template"])
        ),
        score=_optional_float(data.get("score")),
        competing_score=_optional_float(data.get("competing_score")),
        margin=_optional_float(data.get("margin")),
        status=str(data["status"]),
        rms_headroom=_optional_float(data.get("rms_headroom")),
        margin_headroom=_optional_float(data.get("margin_headroom")),
        threshold_sensitivity=str(data["threshold_sensitivity"]),
    )


def _local_domain_from_dict(data: dict[str, Any]) -> LocalPerceptionDomain:
    template = data.get("template_decision")
    return LocalPerceptionDomain(
        domain_id=str(data["domain_id"]),
        kind=str(data["kind"]),
        center_atom=None if data.get("center_atom") is None else int(data["center_atom"]),
        members=_ints(data["members"]),
        equivalence_classes=tuple(
            _local_equivalence_from_dict(item) for item in data["equivalence_classes"]
        ),
        proposed_group=str(data["proposed_group"]),
        confidence=str(data["confidence"]),
        operation_count=int(data["operation_count"]),
        template_decision=(None if template is None else _local_template_from_dict(template)),
        thresholds=tuple(
            (str(name), float(value), str(unit)) for name, value, unit in data["thresholds"]
        ),
        provider=str(data["provider"]),
        provider_version=str(data["provider_version"]),
        provenance=str(data["provenance"]),
    )


__all__ = [
    "AuxiliaryContact",
    "ContactEndpoint",
    "FragmentMembership",
    "LocalEquivalenceClass",
    "LocalPerceptionDomain",
    "LocalTemplateDecision",
    "MulticenterDomain",
    "ORACLE_SONIC_CONTRACT_OWNER",
    "ORACLE_PRIMITIVE_MODES",
    "ORACLE_SONIC_CONTRACT_SCHEMA",
    "ORACLE_SONIC_CONTRACT_SCHEMA_V1",
    "ORACLE_SONIC_CONTRACT_SECTION",
    "OracleSonicContract",
    "OracleSonicContractError",
    "PrimaryTopology",
    "PrimitiveCandidate",
    "StructuralSite",
    "SUPPORTED_ORACLE_SONIC_CONTRACT_SCHEMAS",
    "graph_cycle_rank",
    "migrate_oracle_sonic_contract_payload",
    "oracle_sonic_contract_from_dict",
    "oracle_sonic_contract_section_lines",
    "oracle_sonic_contract_to_dict",
    "primitive_candidate_wilson_row",
    "primary_topology_hash",
    "read_oracle_sonic_contract",
    "validate_oracle_sonic_contract",
    "write_oracle_sonic_contract",
]
