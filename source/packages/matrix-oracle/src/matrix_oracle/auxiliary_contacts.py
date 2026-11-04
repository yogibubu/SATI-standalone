"""ORACLE-owned perception providers for typed auxiliary contacts.

The providers reuse MATRIX's established hydrogen-bond, directional-contact,
periodic-property, and metal-classification kernels.  They produce immutable
chemical evidence only; cycle policy and symmetry-orbit completion are separate
ORACLE stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Iterable, Sequence

import numpy as np

from matrix_chem import (
    ContactEndpoint,
    perceive_directional_contacts,
    perceive_hydrogen_bonds,
)
from matrix_chem.topology.bonding_roles import (
    admits_dative_pair,
    is_bridging_ligand,
    is_structural_center,
)
from matrix_chem.topology.periodic_properties import periodic_atomic_properties


AUXILIARY_CONTACT_PROVIDER_SCHEMA = "matrix.oracle.auxiliary_contact_providers.v1"
HBOND_PROVIDER = "ORACLE_HYDROGEN_BOND_VDW_DIRECTIONAL"
HBOND_PROVIDER_VERSION = "1"
DIRECTIONAL_PROVIDER = "ORACLE_SIGMA_HOLE_VDW_DIRECTIONAL"
DIRECTIONAL_PROVIDER_VERSION = "1"
DATIVE_PROVIDER = "ORACLE_DATIVE_VDW_DIRECTIONAL"
DATIVE_PROVIDER_VERSION = "1"
STRUCTURAL_LIGAND_PROVIDER = "ORACLE_STRUCTURAL_LIGAND_VDW_DIRECTIONAL"
STRUCTURAL_LIGAND_PROVIDER_VERSION = "1"
CONFIGURED_SITE_PROVIDER = "ORACLE_CONFIGURED_STRUCTURAL_SITE_CONTACT"
CONFIGURED_SITE_PROVIDER_VERSION = "1"


@dataclass(frozen=True)
class AuxiliaryContactEvidence:
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
    applicability_range: str
    provenance: str


@dataclass(frozen=True)
class StructuralSiteContactRequest:
    """Explicit ORACLE provider request for an atom--structural-site contact."""

    kind: str
    atom: int
    site_id: str
    distance_angstrom: float
    site_effective_radius_angstrom: float | None
    member_rho_vdw: tuple[float, ...] = ()
    directional_descriptors: tuple[tuple[str, float], ...] = ()
    confidence: float = 1.0
    persistence: float = 1.0
    provenance: str = "ORACLE_CONFIGURED_STRUCTURAL_SITE"


@dataclass(frozen=True)
class AuxiliaryContactProviderSettings:
    maximum_rho_vdw: float = 1.0
    persistence_width: float = 0.08
    minimum_confidence: float = 0.20
    family_maximum_rho_vdw: tuple[tuple[str, float], ...] = (
        ("HYDROGEN_BOND", 1.0),
        ("DATIVE_CONTACT", 1.0),
        ("STRUCTURAL_LIGAND_CONTACT", 1.0),
        ("TETREL_BOND", 1.0),
        ("PNICTOGEN_BOND", 1.0),
        ("CHALCOGEN_BOND", 1.0),
        ("HALOGEN_BOND", 1.0),
    )
    family_minimum_confidence: tuple[tuple[str, float], ...] = (
        ("HYDROGEN_BOND", 0.20),
        ("DATIVE_CONTACT", 0.20),
        ("STRUCTURAL_LIGAND_CONTACT", 0.20),
        ("TETREL_BOND", 0.20),
        ("PNICTOGEN_BOND", 0.20),
        ("CHALCOGEN_BOND", 0.20),
        ("HALOGEN_BOND", 0.20),
    )

    def __post_init__(self) -> None:
        if self.maximum_rho_vdw <= 0.0 or self.persistence_width <= 0.0:
            raise ValueError("auxiliary-contact radial settings must be positive")
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("auxiliary-contact confidence threshold must lie in [0, 1]")
        _validate_family_thresholds(
            self.family_maximum_rho_vdw,
            label="maximum rho_vdw",
            lower=0.0,
            upper=None,
        )
        _validate_family_thresholds(
            self.family_minimum_confidence,
            label="minimum confidence",
            lower=0.0,
            upper=1.0,
        )

    def maximum_rho_for(self, kind: str) -> float:
        return dict(self.family_maximum_rho_vdw).get(
            str(kind).strip().upper(), self.maximum_rho_vdw
        )

    def minimum_confidence_for(self, kind: str) -> float:
        return dict(self.family_minimum_confidence).get(
            str(kind).strip().upper(), self.minimum_confidence
        )


def qualified_vdw_radius(atomic_number: int) -> tuple[float, str]:
    """Return ORACLE's finite provenance-bearing vdW radius, never a covalent radius."""

    properties = periodic_atomic_properties(int(atomic_number))
    sources = "+".join(source for source in properties.sources if "VDW" in source or "UFF" in source)
    return float(properties.vdw_radius_angstrom), sources or "PERIODIC_PROPERTIES_VDW"


def perceive_auxiliary_contact_evidence(
    atomic_numbers: Sequence[int],
    coordinates_angstrom: np.ndarray,
    bonded_pairs: Iterable[tuple[int, int]],
    *,
    settings: AuxiliaryContactProviderSettings | None = None,
    configured_site_contacts: Iterable[StructuralSiteContactRequest] = (),
) -> tuple[AuxiliaryContactEvidence, ...]:
    """Run all configured ORACLE contact providers on frozen primary topology.

    Atom indices in ``bonded_pairs`` are zero based, matching MATRIX topology
    kernels.  Serialized endpoints are one based.  Existing primary bonds are
    excluded before provider evidence is returned.
    """

    options = settings or AuxiliaryContactProviderSettings()
    numbers = tuple(int(value) for value in atomic_numbers)
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    if xyz.shape != (len(numbers), 3) or np.any(~np.isfinite(xyz)):
        raise ValueError("auxiliary-contact coordinates must be finite (natoms, 3)")
    bonds = {tuple(sorted((int(left), int(right)))) for left, right in bonded_pairs}
    _validate_bonds(bonds, len(numbers))
    adjacency = _adjacency(len(numbers), bonds)

    evidence = [
        *_hydrogen_bond_evidence(numbers, xyz, bonds, options),
        *_directional_contact_evidence(numbers, xyz, bonds, options),
        *_center_ligand_contact_evidence(numbers, xyz, bonds, adjacency, options),
        *_configured_site_contact_evidence(numbers, options, configured_site_contacts),
    ]
    primary_one_based = {tuple(sorted((left + 1, right + 1))) for left, right in bonds}
    filtered = []
    for item in evidence:
        if item.endpoint_a.kind == item.endpoint_b.kind == "ATOM":
            pair = tuple(
                sorted((int(item.endpoint_a.identifier), int(item.endpoint_b.identifier)))
            )
            if pair in primary_one_based:
                continue
        if (
            item.rho_vdw > options.maximum_rho_for(item.kind)
            or item.confidence < options.minimum_confidence_for(item.kind)
        ):
            continue
        filtered.append(item)
    return _deduplicate_contact_evidence(filtered)


def _hydrogen_bond_evidence(
    numbers: tuple[int, ...],
    xyz: np.ndarray,
    bonds: set[tuple[int, int]],
    settings: AuxiliaryContactProviderSettings,
) -> tuple[AuxiliaryContactEvidence, ...]:
    contacts = perceive_hydrogen_bonds(numbers, xyz, bonds)
    result = []
    for contact in contacts:
        pair = tuple(sorted((int(contact.hydrogen), int(contact.acceptor))))
        if pair in bonds:
            continue
        rho, radius_source = _atom_pair_rho(
            numbers[contact.hydrogen],
            numbers[contact.acceptor],
            contact.distance_angstrom,
        )
        angular = max(0.0, min(1.0, 0.5 * (1.0 - np.cos(contact.angle_radians))))
        confidence = float(max(0.0, min(1.0, contact.coordination_weight * angular)))
        result.append(
            AuxiliaryContactEvidence(
                kind="HYDROGEN_BOND",
                endpoint_a=ContactEndpoint("ATOM", str(contact.hydrogen + 1)),
                endpoint_b=ContactEndpoint("ATOM", str(contact.acceptor + 1)),
                rho_vdw=rho,
                distance_angstrom=float(contact.distance_angstrom),
                directional_descriptors=(
                    ("DONOR_ATOM", float(contact.donor + 1)),
                    ("D_H_A_ANGLE_DEGREES", float(np.degrees(contact.angle_radians))),
                    ("ANGULAR_SCORE", angular),
                    ("INTRAMOLECULAR", float(contact.intramolecular)),
                ),
                confidence=confidence,
                persistence=_persistence(rho, settings, "HYDROGEN_BOND"),
                provider=HBOND_PROVIDER,
                provider_version=HBOND_PROVIDER_VERSION,
                applicability_range=(
                    "electronegative lone-pair-center hydrogen bonds; "
                    "rho_vdw<=configured maximum"
                ),
                provenance=f"{AUXILIARY_CONTACT_PROVIDER_SCHEMA}:{radius_source}",
            )
        )
    return tuple(result)


def _directional_contact_evidence(
    numbers: tuple[int, ...],
    xyz: np.ndarray,
    bonds: set[tuple[int, int]],
    settings: AuxiliaryContactProviderSettings,
) -> tuple[AuxiliaryContactEvidence, ...]:
    result = []
    for contact in perceive_directional_contacts(numbers, xyz, bonds):
        pair = tuple(sorted((contact.center, contact.acceptor)))
        if pair in bonds:
            continue
        rho, radius_source = _atom_pair_rho(
            numbers[contact.center], numbers[contact.acceptor], contact.distance_angstrom
        )
        result.append(
            AuxiliaryContactEvidence(
                kind=contact.kind.upper().replace("-", "_"),
                endpoint_a=ContactEndpoint("ATOM", str(contact.center + 1)),
                endpoint_b=ContactEndpoint("ATOM", str(contact.acceptor + 1)),
                rho_vdw=rho,
                distance_angstrom=float(contact.distance_angstrom),
                directional_descriptors=(
                    ("ANCHOR_ATOM", float(contact.anchor + 1)),
                    ("ANCHOR_CENTER_ACCEPTOR_ANGLE_DEGREES", float(contact.angle_degrees)),
                    ("DIRECTIONAL_STRENGTH", float(contact.strength)),
                ),
                confidence=float(max(0.0, min(1.0, contact.strength))),
                persistence=_persistence(rho, settings, contact.kind),
                provider=DIRECTIONAL_PROVIDER,
                provider_version=DIRECTIONAL_PROVIDER_VERSION,
                applicability_range="conservative topology-assigned sigma/pi-hole centers",
                provenance=(
                    f"{AUXILIARY_CONTACT_PROVIDER_SCHEMA}:{contact.schema}:{radius_source}"
                ),
            )
        )
    return tuple(result)


def _center_ligand_contact_evidence(
    numbers: tuple[int, ...],
    xyz: np.ndarray,
    bonds: set[tuple[int, int]],
    adjacency: tuple[frozenset[int], ...],
    settings: AuxiliaryContactProviderSettings,
) -> tuple[AuxiliaryContactEvidence, ...]:
    result = []
    for center, center_number in enumerate(numbers):
        center_properties = periodic_atomic_properties(center_number)
        if not is_structural_center(center_number):
            continue
        for ligand, ligand_number in enumerate(numbers):
            pair = tuple(sorted((center, ligand)))
            if ligand == center or pair in bonds:
                continue
            structural_ligand = is_bridging_ligand(ligand_number)
            dative = admits_dative_pair(ligand_number, center_number)
            if not structural_ligand and not dative:
                continue
            kind = "STRUCTURAL_LIGAND_CONTACT" if structural_ligand else "DATIVE_CONTACT"
            provider = STRUCTURAL_LIGAND_PROVIDER if structural_ligand else DATIVE_PROVIDER
            provider_version = (
                STRUCTURAL_LIGAND_PROVIDER_VERSION
                if structural_ligand
                else DATIVE_PROVIDER_VERSION
            )
            distance = float(np.linalg.norm(xyz[ligand] - xyz[center]))
            if distance <= 1.0e-12:
                continue
            rho, radius_source = _atom_pair_rho(
                center_number, ligand_number, distance
            )
            ligand_exposure = 1.0 / (1.0 + len(adjacency[ligand]))
            center_exposure = 1.0 / (1.0 + len(adjacency[center]))
            approach = _open_direction_score(ligand, center, xyz, adjacency)
            confidence = float(
                min(1.0, 2.0 * ligand_exposure * center_exposure * approach)
            )
            result.append(
                AuxiliaryContactEvidence(
                    kind=kind,
                    endpoint_a=ContactEndpoint("ATOM", str(center + 1)),
                    endpoint_b=ContactEndpoint("ATOM", str(ligand + 1)),
                    rho_vdw=rho,
                    distance_angstrom=distance,
                    directional_descriptors=(
                        ("CENTER_GROUP", float(center_properties.group)),
                        (
                            "LIGAND_GROUP",
                            float(periodic_atomic_properties(ligand_number).group),
                        ),
                        ("LIGAND_EXPOSURE", ligand_exposure),
                        ("CENTER_EXPOSURE", center_exposure),
                        ("OPEN_DIRECTION_SCORE", approach),
                    ),
                    confidence=confidence,
                    persistence=_persistence(rho, settings, kind),
                    provider=provider,
                    provider_version=provider_version,
                    applicability_range=(
                        "periodic structural centers with H/halide ligands"
                        if structural_ligand
                        else "electronegative lone-pair donors with structural Lewis-acid centers"
                    ),
                    provenance=f"{AUXILIARY_CONTACT_PROVIDER_SCHEMA}:{radius_source}",
                )
            )
    return tuple(result)


def _configured_site_contact_evidence(
    numbers: tuple[int, ...],
    settings: AuxiliaryContactProviderSettings,
    requests: Iterable[StructuralSiteContactRequest],
) -> tuple[AuxiliaryContactEvidence, ...]:
    result = []
    for request in requests:
        atom = int(request.atom)
        if atom < 0 or atom >= len(numbers):
            raise ValueError("configured structural-site contact atom is outside the system")
        atom_radius, radius_source = qualified_vdw_radius(numbers[atom])
        if request.site_effective_radius_angstrom is not None:
            site_radius = float(request.site_effective_radius_angstrom)
            if not np.isfinite(site_radius) or site_radius <= 0.0:
                raise ValueError("structural-site effective radius must be finite and positive")
            rho = float(request.distance_angstrom) / (atom_radius + site_radius)
            scoring = "PROVIDER_EFFECTIVE_RADIUS"
        elif request.member_rho_vdw:
            rho = min(float(value) for value in request.member_rho_vdw)
            scoring = "DOCUMENTED_MEMBERWISE_MINIMUM_RHO"
        else:
            raise ValueError(
                "structural-center contact requires an effective radius or memberwise vdW scores"
            )
        result.append(
            AuxiliaryContactEvidence(
                kind=str(request.kind).strip().upper(),
                endpoint_a=ContactEndpoint("ATOM", str(atom + 1)),
                endpoint_b=ContactEndpoint("STRUCTURAL_SITE", request.site_id),
                rho_vdw=rho,
                distance_angstrom=float(request.distance_angstrom),
                directional_descriptors=(
                    *tuple(request.directional_descriptors),
                    ("STRUCTURAL_CENTER_SCORING", 1.0),
                ),
                confidence=float(request.confidence),
                persistence=float(request.persistence),
                provider=CONFIGURED_SITE_PROVIDER,
                provider_version=CONFIGURED_SITE_PROVIDER_VERSION,
                applicability_range=scoring,
                provenance=f"{request.provenance}:{radius_source}",
            )
        )
    return tuple(result)


def _atom_pair_rho(left_z: int, right_z: int, distance: float) -> tuple[float, str]:
    left_radius, left_source = qualified_vdw_radius(left_z)
    right_radius, right_source = qualified_vdw_radius(right_z)
    denominator = left_radius + right_radius
    if denominator <= 0.0 or not np.isfinite(distance) or distance <= 0.0:
        raise ValueError("vdW-normalized contact distance is invalid")
    return float(distance / denominator), f"{left_source}|{right_source}"


def _persistence(
    rho: float,
    settings: AuxiliaryContactProviderSettings,
    kind: str,
) -> float:
    return float(
        1.0
        / (
            1.0
            + exp(
                (float(rho) - settings.maximum_rho_for(kind))
                / settings.persistence_width
            )
        )
    )


def _validate_family_thresholds(records, *, label: str, lower: float, upper: float | None) -> None:
    names = [str(name).strip().upper() for name, _value in records]
    if len(names) != len(set(names)) or any(not name for name in names):
        raise ValueError(f"contact-family {label} records must have unique names")
    for _name, raw_value in records:
        value = float(raw_value)
        if not np.isfinite(value) or value < lower or (upper is not None and value > upper):
            raise ValueError(f"contact-family {label} is outside its valid range")


def _open_direction_score(
    donor: int,
    target: int,
    xyz: np.ndarray,
    adjacency: tuple[frozenset[int], ...],
) -> float:
    neighbors = adjacency[donor]
    if not neighbors:
        return 1.0
    outward = np.zeros(3, dtype=float)
    for neighbor in neighbors:
        vector = xyz[donor] - xyz[neighbor]
        norm = float(np.linalg.norm(vector))
        if norm > 1.0e-12:
            outward += vector / norm
    target_vector = xyz[target] - xyz[donor]
    denominator = float(np.linalg.norm(outward) * np.linalg.norm(target_vector))
    if denominator <= 1.0e-12:
        return 0.5
    cosine = float(np.clip(np.dot(outward, target_vector) / denominator, -1.0, 1.0))
    return 0.5 * (1.0 + cosine)


def _adjacency(
    natoms: int, bonds: set[tuple[int, int]]
) -> tuple[frozenset[int], ...]:
    result = [set() for _ in range(natoms)]
    for left, right in bonds:
        result[left].add(right)
        result[right].add(left)
    return tuple(frozenset(items) for items in result)


def _validate_bonds(bonds: set[tuple[int, int]], natoms: int) -> None:
    if any(left == right or left < 0 or right >= natoms for left, right in bonds):
        raise ValueError("primary topology contains an invalid bond")


def _deduplicate_contact_evidence(
    records: Iterable[AuxiliaryContactEvidence],
) -> tuple[AuxiliaryContactEvidence, ...]:
    priority = {
        "HYDROGEN_BOND": 0,
        "STRUCTURAL_LIGAND_CONTACT": 1,
        "DATIVE_CONTACT": 2,
        "HALOGEN_BOND": 3,
        "CHALCOGEN_BOND": 4,
        "PNICTOGEN_BOND": 5,
        "TETREL_BOND": 6,
    }
    selected: dict[tuple[tuple[str, str], ...], AuxiliaryContactEvidence] = {}
    for record in records:
        key = tuple(
            sorted(
                (
                    (record.endpoint_a.kind, record.endpoint_a.identifier),
                    (record.endpoint_b.kind, record.endpoint_b.identifier),
                )
            )
        )
        previous = selected.get(key)
        if previous is None or (
            priority.get(record.kind, 99), -record.confidence, record.provider
        ) < (
            priority.get(previous.kind, 99), -previous.confidence, previous.provider
        ):
            selected[key] = record
    return tuple(
        sorted(
            selected.values(),
            key=lambda item: (
                item.kind,
                item.endpoint_a.kind,
                item.endpoint_a.identifier,
                item.endpoint_b.kind,
                item.endpoint_b.identifier,
            ),
        )
    )


__all__ = [
    "AUXILIARY_CONTACT_PROVIDER_SCHEMA",
    "AuxiliaryContactEvidence",
    "AuxiliaryContactProviderSettings",
    "DATIVE_PROVIDER",
    "STRUCTURAL_LIGAND_PROVIDER",
    "StructuralSiteContactRequest",
    "perceive_auxiliary_contact_evidence",
    "qualified_vdw_radius",
]
