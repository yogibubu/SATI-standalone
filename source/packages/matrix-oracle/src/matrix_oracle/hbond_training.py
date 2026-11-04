"""Reproducible paired-CM5 training set for hydrogen-bond charge response.

The catalogue is deliberately backend-neutral.  It defines small neutral
monomers, their chemically distinct donor/acceptor sites, and deterministic
geometries for the oriented dimers.  Intramolecular coordinates remain fixed:
only the controlled intermolecular H...A distance and D--H...A angle vary.
This prevents monomer deformation from being absorbed into the fitted
hydrogen-bond charge response.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from itertools import permutations, product
import json
from typing import Mapping, Sequence

import numpy as np

from .hbond_charge_response import (
    HydrogenBondChargeContact,
    HydrogenBondResponseCalibration,
    fit_cm5_hydrogen_bond_response,
)
from matrix_chem import perceive_hydrogen_bonds
from matrix_chem.topology.elements import atomic_number
from matrix_chem.topology.vdw_radii import uff_vdw_radius


@dataclass(frozen=True)
class HydrogenBondTrainingMolecule:
    """One monomer and the sites used by the minimal response training set."""

    name: str
    atoms: tuple[str, ...]
    coordinates_angstrom: np.ndarray
    bonds: tuple[tuple[int, int], ...]
    donor_sites: tuple[tuple[int, int, str], ...] = ()
    acceptor_sites: tuple[tuple[int, int, str], ...] = ()

    def __post_init__(self) -> None:
        xyz = np.asarray(self.coordinates_angstrom, dtype=float)
        if xyz.shape != (len(self.atoms), 3) or np.any(~np.isfinite(xyz)):
            raise ValueError("training-molecule coordinates must have shape (natoms, 3)")
        if not self.name.strip() or not self.atoms:
            raise ValueError("a training molecule needs a name and atoms")
        for donor, hydrogen, label in self.donor_sites:
            if not (0 <= donor < len(self.atoms) and 0 <= hydrogen < len(self.atoms)):
                raise ValueError("training donor indices are outside the molecule")
            if tuple(sorted((donor, hydrogen))) not in {
                tuple(sorted(pair)) for pair in self.bonds
            }:
                raise ValueError("a training donor hydrogen must be bonded to its donor")
            if not label.strip():
                raise ValueError("a training donor site needs a synthon label")
        for acceptor, anchor, label in self.acceptor_sites:
            if not (0 <= acceptor < len(self.atoms) and 0 <= anchor < len(self.atoms)):
                raise ValueError("training acceptor indices are outside the molecule")
            if acceptor == anchor or not label.strip():
                raise ValueError("a training acceptor needs an anchor and synthon label")
        object.__setattr__(self, "coordinates_angstrom", xyz.copy())


@dataclass(frozen=True)
class HydrogenBondTrainingComplex:
    """One oriented donor--acceptor dimer in the training set."""

    identifier: str
    donor_molecule: str
    acceptor_molecule: str
    donor_site: int = 0
    acceptor_site: int = 0
    role: str = "PRIMARY_XH_HBOND"
    reference_distance_angstrom: float | None = None


@dataclass(frozen=True)
class HydrogenBondTrainingGeometry:
    """Assembled dimer with the mapping needed by frozen-fragment analysis."""

    specification: HydrogenBondTrainingComplex
    atoms: tuple[str, ...]
    coordinates_angstrom: np.ndarray
    donor_atoms: tuple[int, ...]
    acceptor_atoms: tuple[int, ...]
    donor: int
    hydrogen: int
    acceptor: int
    donor_synthon: str
    acceptor_synthon: str
    hydrogen_acceptor_distance_angstrom: float
    dha_angle_degrees: float


@dataclass(frozen=True)
class HydrogenBondResponseTemplate:
    """Index-independent response rule tied to donor/acceptor synthons."""

    donor_molecule: str
    acceptor_molecule: str
    donor_synthon: str
    acceptor_synthon: str
    donor_polarization_response_e: tuple[tuple[int, float], ...]
    acceptor_polarization_response_e: tuple[tuple[int, float], ...]
    charge_transfer_e: float
    reference_distance_angstrom: float
    reference_mayer_bond_order: float | None
    vdw_radius_sum_angstrom: float
    population_level: str

    def instantiate(
        self,
        *,
        donor_mapping: Sequence[int],
        acceptor_mapping: Sequence[int],
        donor: int,
        hydrogen: int,
        acceptor: int,
    ) -> HydrogenBondChargeContact:
        donor_map = tuple(int(atom) for atom in donor_mapping)
        acceptor_map = tuple(int(atom) for atom in acceptor_mapping)
        return HydrogenBondChargeContact(
            donor=int(donor),
            hydrogen=int(hydrogen),
            acceptor=int(acceptor),
            donor_polarization_response_e=tuple(
                (donor_map[index], float(value))
                for index, value in self.donor_polarization_response_e
            ),
            acceptor_polarization_response_e=tuple(
                (acceptor_map[index], float(value))
                for index, value in self.acceptor_polarization_response_e
            ),
            charge_transfer_e=float(self.charge_transfer_e),
            label=f"{self.donor_synthon}...{self.acceptor_synthon}",
            reference_distance_angstrom=float(self.reference_distance_angstrom),
            reference_mayer_bond_order=self.reference_mayer_bond_order,
            vdw_radius_sum_angstrom=float(self.vdw_radius_sum_angstrom),
        )


@dataclass(frozen=True)
class HydrogenBondResponseLibrary:
    """Resident ORACLE rules fitted from paired fixed-geometry CM5 data."""

    templates: tuple[HydrogenBondResponseTemplate, ...]
    population_level: str
    source: str

    def template(
        self,
        donor_synthon: str,
        acceptor_synthon: str,
    ) -> HydrogenBondResponseTemplate | None:
        matches = tuple(
            item
            for item in self.templates
            if item.donor_synthon == str(donor_synthon)
            and item.acceptor_synthon == str(acceptor_synthon)
        )
        if len(matches) > 1:
            raise ValueError("the H-bond library contains duplicate synthon rules")
        return matches[0] if matches else None


@dataclass(frozen=True)
class WaterCoordinationCluster:
    """Rigid water cluster indexed by the central donor/acceptor counts."""

    identifier: str
    donor_count: int
    acceptor_count: int
    atoms: tuple[str, ...]
    coordinates_angstrom: np.ndarray
    waters: tuple[tuple[int, int, int], ...]
    central_water: tuple[int, int, int] = (0, 1, 2)


@dataclass(frozen=True)
class HydrogenBondGeometryAudit:
    """Chemical preflight for one rigid training dimer."""

    identifier: str
    intended_contact_count: int
    unexpected_contact_count: int
    minimum_bond_radius_ratio: float
    maximum_bond_radius_ratio: float
    minimum_unintended_vdw_ratio: float
    passed: bool


def standard_hbond_training_molecules() -> Mapping[str, HydrogenBondTrainingMolecule]:
    """Return the immutable N/O/P/S training catalogue."""

    # These fixed intramolecular references define the training convention.
    # Production calculations vary only the relative pose of two rigid
    # monomers.
    molecules = (
        HydrogenBondTrainingMolecule(
            name="water",
            atoms=("O", "H", "H"),
            coordinates_angstrom=np.asarray(
                ((0.000000, 0.000000, 0.000000),
                 (0.957200, 0.000000, 0.000000),
                 (-0.239987, 0.927297, 0.000000))
            ),
            bonds=((0, 1), (0, 2)),
            donor_sites=((0, 1, "water_OH"),),
            acceptor_sites=((0, 1, "water_O"),),
        ),
        HydrogenBondTrainingMolecule(
            name="ammonia",
            atoms=("N", "H", "H", "H"),
            coordinates_angstrom=np.asarray(
                ((0.000000, 0.000000, 0.117000),
                 (0.000000, 0.938000, -0.273000),
                 (0.812000, -0.469000, -0.273000),
                 (-0.812000, -0.469000, -0.273000))
            ),
            bonds=((0, 1), (0, 2), (0, 3)),
            donor_sites=((0, 1, "ammonia_NH"),),
            acceptor_sites=((0, 1, "ammonia_N"),),
        ),
        HydrogenBondTrainingMolecule(
            name="formaldehyde",
            atoms=("C", "O", "H", "H"),
            coordinates_angstrom=np.asarray(
                ((0.000000, 0.000000, 0.000000),
                 (1.208000, 0.000000, 0.000000),
                 (-0.583000, 0.935000, 0.000000),
                 (-0.583000, -0.935000, 0.000000))
            ),
            bonds=((0, 1), (0, 2), (0, 3)),
            acceptor_sites=((1, 0, "formaldehyde_carbonyl_O"),),
        ),
        HydrogenBondTrainingMolecule(
            name="hydrogen_sulfide",
            atoms=("S", "H", "H"),
            coordinates_angstrom=np.asarray(
                ((0.000000, 0.000000, 0.000000),
                 (1.335000, 0.000000, 0.000000),
                 (-0.048900, 1.334104, 0.000000))
            ),
            bonds=((0, 1), (0, 2)),
            donor_sites=((0, 1, "hydrogen_sulfide_SH"),),
            acceptor_sites=((0, 1, "hydrogen_sulfide_S"),),
        ),
        HydrogenBondTrainingMolecule(
            name="phosphine",
            atoms=("P", "H", "H", "H"),
            coordinates_angstrom=np.asarray(
                ((0.000000, 0.000000, 0.000000),
                 (1.194000, 0.000000, -0.768000),
                 (-0.597000, 1.034034, -0.768000),
                 (-0.597000, -1.034034, -0.768000))
            ),
            bonds=((0, 1), (0, 2), (0, 3)),
            donor_sites=((0, 1, "phosphine_PH"),),
            acceptor_sites=((0, 1, "phosphine_P"),),
        ),
    )
    return {molecule.name: molecule for molecule in molecules}


def extended_hbond_training_molecules() -> Mapping[str, HydrogenBondTrainingMolecule]:
    """Return primary references plus substituted-environment transfer probes."""

    extra = (
        HydrogenBondTrainingMolecule(
            name="methanol",
            atoms=("O", "H", "C", "H", "H", "H"),
            coordinates_angstrom=np.asarray(
                ((1.4300, 0.0000, 0.0000), (1.8500, 0.8700, 0.0000),
                 (0.0000, 0.0000, 0.0000), (-0.3900, 1.0300, 0.0000),
                 (-0.3900, -0.5150, 0.8920), (-0.3900, -0.5150, -0.8920))
            ),
            bonds=((0, 1), (0, 2), (2, 3), (2, 4), (2, 5)),
            donor_sites=((0, 1, "methanol_OH"),),
            acceptor_sites=((0, 2, "methanol_O"),),
        ),
        HydrogenBondTrainingMolecule(
            name="methylamine",
            atoms=("N", "H", "H", "C", "H", "H", "H"),
            coordinates_angstrom=np.asarray(
                ((1.4700, 0.0000, 0.0000), (1.8200, 0.9400, 0.0000),
                 (1.8200, -0.4700, 0.8140), (0.0000, 0.0000, 0.0000),
                 (-0.3900, 1.0300, 0.0000), (-0.3900, -0.5150, 0.8920),
                 (-0.3900, -0.5150, -0.8920))
            ),
            bonds=((0, 1), (0, 2), (0, 3), (3, 4), (3, 5), (3, 6)),
            donor_sites=((0, 1, "methylamine_NH"),),
            acceptor_sites=((0, 3, "methylamine_N"),),
        ),
        HydrogenBondTrainingMolecule(
            name="methanethiol",
            atoms=("S", "H", "C", "H", "H", "H"),
            coordinates_angstrom=np.asarray(
                ((1.8200, 0.0000, 0.0000), (2.3400, 1.2300, 0.0000),
                 (0.0000, 0.0000, 0.0000), (-0.3900, 1.0300, 0.0000),
                 (-0.3900, -0.5150, 0.8920), (-0.3900, -0.5150, -0.8920))
            ),
            bonds=((0, 1), (0, 2), (2, 3), (2, 4), (2, 5)),
            donor_sites=((0, 1, "methanethiol_SH"),),
            acceptor_sites=((0, 2, "methanethiol_S"),),
        ),
        HydrogenBondTrainingMolecule(
            name="methylphosphine",
            atoms=("P", "H", "H", "C", "H", "H", "H"),
            coordinates_angstrom=np.asarray(
                ((1.8600, 0.0000, 0.0000), (2.3300, 1.3100, 0.0000),
                 (2.3300, -0.6550, 1.1340), (0.0000, 0.0000, 0.0000),
                 (-0.3900, 1.0300, 0.0000), (-0.3900, -0.5150, 0.8920),
                 (-0.3900, -0.5150, -0.8920))
            ),
            bonds=((0, 1), (0, 2), (0, 3), (3, 4), (3, 5), (3, 6)),
            donor_sites=((0, 1, "methylphosphine_PH"),),
            acceptor_sites=((0, 3, "methylphosphine_P"),),
        ),
        HydrogenBondTrainingMolecule(
            name="formamide",
            atoms=("C", "O", "N", "H", "H", "H"),
            coordinates_angstrom=np.asarray(
                ((-1.188593, 0.122535, 0.008805),
                 (-0.509687, 1.137137, -0.006838),
                 (-0.700259, -1.129375, -0.015521),
                 (-2.285040, 0.166214, 0.043616),
                 (-1.322817, -1.912841, 0.014402),
                 (0.301093, -1.266127, -0.044664))
            ),
            bonds=((0, 1), (0, 2), (0, 3), (2, 4), (2, 5)),
            donor_sites=((2, 4, "formamide_NH"),),
            acceptor_sites=((1, 0, "formamide_carbonyl_O"),),
        ),
    )
    return {**standard_hbond_training_molecules(), **{item.name: item for item in extra}}


def standard_hbond_training_complexes() -> tuple[HydrogenBondTrainingComplex, ...]:
    """Return the 4 donor classes x 5 acceptor classes (20 directions)."""

    donors = (
        "water",
        "ammonia",
        "hydrogen_sulfide",
        "phosphine",
    )
    acceptors = (
        "water",
        "ammonia",
        "formaldehyde",
        "hydrogen_sulfide",
        "phosphine",
    )
    catalogue = extended_hbond_training_molecules()
    radius_h = uff_vdw_radius(1)
    radius_o = uff_vdw_radius(8)
    if radius_h is None or radius_o is None:
        raise RuntimeError("resident UFF radii do not cover H and O")
    return tuple(
        HydrogenBondTrainingComplex(
            identifier=f"{donor}__to__{acceptor}",
            donor_molecule=donor,
            acceptor_molecule=acceptor,
            reference_distance_angstrom=(
                1.85
                if acceptor not in {"hydrogen_sulfide", "phosphine"}
                else 1.85
                * (
                    radius_h
                    + float(
                        uff_vdw_radius(
                            int(
                                atomic_number(
                                    catalogue[acceptor].atoms[
                                        catalogue[acceptor].acceptor_sites[0][0]
                                    ]
                                )
                                or 0
                            )
                        )
                    )
                )
                / (radius_h + radius_o)
            ),
        )
        for donor in donors
        for acceptor in acceptors
    )


def extended_hbond_transfer_complexes() -> tuple[HydrogenBondTrainingComplex, ...]:
    """Return a balanced rigid transfer set outside the primary 4x5 fit.

    Every substituted donor is crossed with the five primary acceptors, every
    primary donor is crossed with the five substituted acceptors, and each
    substituted molecule contributes its chemically allowed self pair.  With
    the primary grid this supplies at least three contexts for every one of
    the 20 XH...A classes while retaining a fixed 4 x 5 parameter table.
    """

    primary_donors = ("water", "ammonia", "hydrogen_sulfide", "phosphine")
    primary_acceptors = (
        "water",
        "ammonia",
        "formaldehyde",
        "hydrogen_sulfide",
        "phosphine",
    )
    substituted = (
        "methanol",
        "methylamine",
        "formamide",
        "methanethiol",
        "methylphosphine",
    )
    rows = [
        HydrogenBondTrainingComplex(
            identifier=f"{donor}__to__{acceptor}",
            donor_molecule=donor,
            acceptor_molecule=acceptor,
            role="TRANSFER_VALIDATION",
        )
        for donor in substituted
        for acceptor in primary_acceptors
    ]
    rows.extend(
        HydrogenBondTrainingComplex(
            identifier=f"{donor}__to__{acceptor}",
            donor_molecule=donor,
            acceptor_molecule=acceptor,
            role="TRANSFER_VALIDATION",
        )
        for donor in primary_donors
        for acceptor in substituted
    )
    rows.extend(
        HydrogenBondTrainingComplex(
            identifier=f"{molecule}__to__{molecule}",
            donor_molecule=molecule,
            acceptor_molecule=molecule,
            role="TRANSFER_VALIDATION",
        )
        for molecule in substituted
    )
    identifiers = tuple(item.identifier for item in rows)
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError(
            "the balanced H-bond transfer set contains duplicate dimers"
        )
    return tuple(rows)


def water_coordination_response_clusters() -> tuple[WaterCoordinationCluster, ...]:
    """Build the complete central-water ``(n_D,n_A)`` grid for 0--2 contacts.

    The nine states are monomer; both directed dimers; the three trimers; the
    two tetramers; and the symmetric four-coordinate pentamer.  Intramolecular
    water coordinates and all H...O reference distances remain fixed.
    """

    water = standard_hbond_training_molecules()["water"]
    central = np.asarray(water.coordinates_angstrom, dtype=float)
    oxygen = central[0]
    donor_directions = tuple(_unit(central[index] - oxygen) for index in (1, 2))
    donor_sum = donor_directions[0] + donor_directions[1]
    normal = _unit(np.cross(donor_directions[0], donor_directions[1]))
    lateral = float(np.sqrt(max(0.0, 1.0 - np.dot(donor_sum, donor_sum) / 4.0)))
    acceptor_directions = (
        _unit(-0.5 * donor_sum + lateral * normal),
        _unit(-0.5 * donor_sum - lateral * normal),
    )

    clusters = []
    for donor_count in range(3):
        for acceptor_count in range(3):
            atoms = list(water.atoms)
            coordinates = [row.copy() for row in central]
            waters = [(0, 1, 2)]
            # The central water donates through its first n_D O--H bonds.
            for direction, hydrogen_index in zip(
                donor_directions[:donor_count],
                (1, 2)[:donor_count],
                strict=True,
            ):
                partner = _orient_acceptor(
                    central,
                    acceptor=0,
                    anchor=1,
                    target_outward=-direction,
                )
                partner += (
                    central[hydrogen_index] + 1.85 * direction - partner[0]
                )
                offset = len(atoms)
                atoms.extend(water.atoms)
                coordinates.extend(partner)
                waters.append((offset, offset + 1, offset + 2))
            # The remaining partners donate toward the two central lone-pair
            # directions, so the central water accepts n_A contacts.
            for direction in acceptor_directions[:acceptor_count]:
                partner = _orient_acceptor(
                    central,
                    acceptor=1,
                    anchor=0,
                    target_outward=-direction,
                )
                partner += oxygen + 1.85 * direction - partner[1]
                offset = len(atoms)
                atoms.extend(water.atoms)
                coordinates.extend(partner)
                waters.append((offset, offset + 1, offset + 2))
            clusters.append(
                WaterCoordinationCluster(
                    identifier=f"water_central_D{donor_count}_A{acceptor_count}",
                    donor_count=donor_count,
                    acceptor_count=acceptor_count,
                    atoms=tuple(atoms),
                    coordinates_angstrom=np.asarray(coordinates, dtype=float),
                    waters=tuple(waters),
                )
            )
    return tuple(clusters)


def formaldehyde_homodimer_control() -> HydrogenBondTrainingComplex:
    """Return the weak C--H...O control excluded from the primary fit."""

    return HydrogenBondTrainingComplex(
        identifier="formaldehyde__ch_to_o__formaldehyde",
        donor_molecule="formaldehyde",
        acceptor_molecule="formaldehyde",
        role="WEAK_CH_O_CONTROL_NOT_IN_PRIMARY_FIT",
    )


def build_hbond_training_geometry(
    specification: HydrogenBondTrainingComplex,
    *,
    hydrogen_acceptor_distance_angstrom: float | None = None,
    dha_angle_degrees: float = 180.0,
) -> HydrogenBondTrainingGeometry:
    """Build a deterministic rigid-monomer D--H...A scan geometry."""

    if specification.role == "WEAK_CH_O_CONTROL_NOT_IN_PRIMARY_FIT":
        return _build_formaldehyde_control(
            specification,
            hydrogen_acceptor_distance_angstrom=hydrogen_acceptor_distance_angstrom,
            dha_angle_degrees=dha_angle_degrees,
        )
    catalogue = extended_hbond_training_molecules()
    acceptor_molecule = catalogue[specification.acceptor_molecule]
    acceptor_index = acceptor_molecule.acceptor_sites[specification.acceptor_site][0]
    acceptor_number = int(atomic_number(acceptor_molecule.atoms[acceptor_index]) or 0)
    radius_h = uff_vdw_radius(1)
    radius_o = uff_vdw_radius(8)
    radius_a = uff_vdw_radius(acceptor_number)
    if radius_h is None or radius_o is None or radius_a is None:
        raise ValueError("the training acceptor lacks a resident van der Waals radius")
    default_distance = 1.85 * (radius_h + radius_a) / (radius_h + radius_o)
    distance = float(
        hydrogen_acceptor_distance_angstrom
        if hydrogen_acceptor_distance_angstrom is not None
        else (
            specification.reference_distance_angstrom
            if specification.reference_distance_angstrom is not None
            else default_distance
        )
    )
    angle = float(dha_angle_degrees)
    if not np.isfinite(distance) or distance <= 0.0:
        raise ValueError("the H...A distance must be finite and positive")
    if not np.isfinite(angle) or not 90.0 <= angle <= 180.0:
        raise ValueError("the D--H...A angle must lie between 90 and 180 degrees")
    donor_molecule = catalogue[specification.donor_molecule]
    try:
        donor, hydrogen, donor_synthon = donor_molecule.donor_sites[
            specification.donor_site
        ]
        acceptor, anchor, acceptor_synthon = acceptor_molecule.acceptor_sites[
            specification.acceptor_site
        ]
    except IndexError as exc:
        raise ValueError("training-complex site index is outside the catalogue") from exc
    donor_xyz = donor_molecule.coordinates_angstrom.copy()
    direction = _unit(donor_xyz[hydrogen] - donor_xyz[donor])
    approach = _angled_approach(direction, 180.0 - angle)
    acceptor_xyz = _orient_acceptor(
        acceptor_molecule.coordinates_angstrom,
        acceptor=acceptor,
        anchor=anchor,
        target_outward=-approach,
    )
    target_acceptor = (
        donor_xyz[hydrogen]
        + distance * approach
    )
    acceptor_xyz += target_acceptor - acceptor_xyz[acceptor]
    acceptor_xyz = _optimize_acceptor_torsion(
        donor_xyz,
        donor_molecule.atoms,
        acceptor_xyz,
        acceptor_molecule.atoms,
        hydrogen=hydrogen,
        acceptor=acceptor,
        axis=approach,
    )
    offset = len(donor_molecule.atoms)
    return HydrogenBondTrainingGeometry(
        specification=specification,
        atoms=donor_molecule.atoms + acceptor_molecule.atoms,
        coordinates_angstrom=np.vstack((donor_xyz, acceptor_xyz)),
        donor_atoms=tuple(range(offset)),
        acceptor_atoms=tuple(range(offset, offset + len(acceptor_molecule.atoms))),
        donor=donor,
        hydrogen=hydrogen,
        acceptor=offset + acceptor,
        donor_synthon=donor_synthon,
        acceptor_synthon=acceptor_synthon,
        hydrogen_acceptor_distance_angstrom=distance,
        dha_angle_degrees=angle,
    )


def audit_hbond_training_geometry(
    geometry: HydrogenBondTrainingGeometry,
) -> HydrogenBondGeometryAudit:
    """Reject strained monomers, steric clashes and undeclared strong contacts."""

    from matrix_chem.topology.pykko_radii import covalent_radius

    catalogue = extended_hbond_training_molecules()
    specification = geometry.specification
    donor = catalogue[specification.donor_molecule]
    acceptor = catalogue[specification.acceptor_molecule]
    offset = len(donor.atoms)
    bonds = tuple(donor.bonds) + tuple(
        (left + offset, right + offset) for left, right in acceptor.bonds
    )
    numbers = tuple(int(atomic_number(symbol) or 0) for symbol in geometry.atoms)
    bond_ratios = []
    for left, right in bonds:
        radius_left = covalent_radius(numbers[left])
        radius_right = covalent_radius(numbers[right])
        if radius_left is None or radius_right is None:
            continue
        distance = float(
            np.linalg.norm(
                geometry.coordinates_angstrom[left]
                - geometry.coordinates_angstrom[right]
            )
        )
        bond_ratios.append(distance / float(radius_left + radius_right))
    contacts = tuple(
        contact
        for contact in perceive_hydrogen_bonds(
            numbers,
            geometry.coordinates_angstrom,
            bonds,
            selector_threshold=0.05,
            minimum_angle_degrees=110.0,
        )
        if not contact.intramolecular
    )
    intended = tuple(
        contact
        for contact in contacts
        if (contact.hydrogen, contact.acceptor)
        == (geometry.hydrogen, geometry.acceptor)
    )
    unexpected = tuple(contact for contact in contacts if contact not in intended)
    normalized = []
    donor_count = len(donor.atoms)
    for left in range(donor_count):
        radius_left = uff_vdw_radius(numbers[left])
        for right in range(donor_count, len(numbers)):
            if (left, right) == (geometry.hydrogen, geometry.acceptor):
                continue
            radius_right = uff_vdw_radius(numbers[right])
            if radius_left is None or radius_right is None:
                continue
            distance = float(
                np.linalg.norm(
                    geometry.coordinates_angstrom[left]
                    - geometry.coordinates_angstrom[right]
                )
            )
            normalized.append(distance / float(radius_left + radius_right))
    minimum_bond = min(bond_ratios, default=1.0)
    maximum_bond = max(bond_ratios, default=1.0)
    minimum_vdw = min(normalized, default=1.0)
    passed = (
        len(intended) == 1
        and not unexpected
        and minimum_bond >= 0.65
        and maximum_bond <= 1.45
        and minimum_vdw >= 0.45
    )
    return HydrogenBondGeometryAudit(
        identifier=specification.identifier,
        intended_contact_count=len(intended),
        unexpected_contact_count=len(unexpected),
        minimum_bond_radius_ratio=float(minimum_bond),
        maximum_bond_radius_ratio=float(maximum_bond),
        minimum_unintended_vdw_ratio=float(minimum_vdw),
        passed=bool(passed),
    )


def fit_training_geometry_cm5(
    geometry: HydrogenBondTrainingGeometry,
    dimer_cm5_e: Sequence[float],
    frozen_donor_cm5_e: Sequence[float],
    frozen_acceptor_cm5_e: Sequence[float],
    *,
    reference_mayer_bond_order: float | None = None,
    population_level: str = "L0:PBE0/def2-TZVP",
) -> HydrogenBondResponseCalibration:
    """Fit one rigid reference dimer from isolated-monomer CM5 values."""

    donor_name = geometry.specification.donor_molecule
    acceptor_name = geometry.specification.acceptor_molecule
    acceptor_number = int(atomic_number(geometry.atoms[geometry.acceptor]) or 0)
    radius_h = uff_vdw_radius(1)
    radius_a = uff_vdw_radius(acceptor_number)
    if radius_h is None or radius_a is None:
        raise ValueError("the training H...A pair lacks resident van der Waals radii")
    vdw_radius_sum = float(radius_h + radius_a)
    donor_local = _training_polarization_atoms(
        donor_name,
        role="donor",
        selected_hydrogen=geometry.hydrogen,
    )
    acceptor_offset = len(geometry.donor_atoms)
    acceptor_local = tuple(
        acceptor_offset + atom
        for atom in _training_polarization_atoms(
            acceptor_name,
            role="acceptor",
        )
    )
    donor_reference = np.asarray(frozen_donor_cm5_e, dtype=float).reshape(-1)
    acceptor_reference = np.asarray(frozen_acceptor_cm5_e, dtype=float).reshape(-1)
    if len(donor_reference) != len(geometry.donor_atoms):
        raise ValueError("frozen donor CM5 vector has the wrong size")
    if len(acceptor_reference) != len(geometry.acceptor_atoms):
        raise ValueError("frozen acceptor CM5 vector has the wrong size")
    reference = np.concatenate((donor_reference, acceptor_reference))
    target = np.asarray(dimer_cm5_e, dtype=float).reshape(-1)
    if len(target) != len(geometry.atoms):
        raise ValueError("dimer CM5 vector has the wrong size")
    return fit_cm5_hydrogen_bond_response(
        reference,
        target,
        donor=geometry.donor,
        hydrogen=geometry.hydrogen,
        acceptor=geometry.acceptor,
        donor_fragment_atoms=geometry.donor_atoms,
        acceptor_fragment_atoms=geometry.acceptor_atoms,
        donor_polarization_atoms=donor_local,
        acceptor_polarization_atoms=acceptor_local,
        donor_synthon=geometry.donor_synthon,
        acceptor_synthon=geometry.acceptor_synthon,
        reference_strength=1.0,
        reference_distance_angstrom=(
            geometry.hydrogen_acceptor_distance_angstrom
        ),
        reference_mayer_bond_order=reference_mayer_bond_order,
        vdw_radius_sum_angstrom=vdw_radius_sum,
        reference_source=f"{population_level} isolated fixed-geometry monomers",
        target_source=(
            f"{population_level} rigid dimer "
            f"rHA={geometry.hydrogen_acceptor_distance_angstrom:.3f} angstrom "
            f"DHA={geometry.dha_angle_degrees:.1f} degree"
        ),
        teacher_model="PAIRED_CM5",
        boundary_contract="ISOLATED_GAS_PHASE_CLUSTER",
    )


@lru_cache(maxsize=1)
def load_standard_hbond_response_library() -> HydrogenBondResponseLibrary:
    """Load the compact paired-CM5 library shared by all MATRIX tools."""

    resource = files("matrix_chem").joinpath(
        "data/hbond_charge_response_l0.json"
    )
    payload = json.loads(resource.read_text(encoding="utf-8"))
    if payload.get("schema") not in {
        "matrix.oracle.hbond_training.v1",
        "matrix.oracle.hbond_training.v2",
    }:
        raise ValueError("unsupported resident ORACLE H-bond training schema")
    catalogue = standard_hbond_training_molecules()
    templates: list[HydrogenBondResponseTemplate] = []
    for record in payload.get("calibrations", ()):
        donor_name = str(record["donor_molecule"])
        acceptor_name = str(record["acceptor_molecule"])
        donor_size = len(catalogue[donor_name].atoms)
        contact = record["contact"]
        donor_response = tuple(
            (int(index), float(value))
            for index, value in contact["donor_polarization_response_e"]
        )
        acceptor_response = tuple(
            (int(index) - donor_size, float(value))
            for index, value in contact["acceptor_polarization_response_e"]
        )
        if (
            donor_response
            and max(index for index, _value in donor_response) >= donor_size
        ):
            raise ValueError("resident donor response is outside its template")
        if acceptor_response and (
            min(index for index, _value in acceptor_response) < 0
            or max(index for index, _value in acceptor_response)
            >= len(catalogue[acceptor_name].atoms)
        ):
            raise ValueError("resident acceptor response is outside its template")
        templates.append(
            HydrogenBondResponseTemplate(
                donor_molecule=donor_name,
                acceptor_molecule=acceptor_name,
                donor_synthon=str(record["donor_synthon"]),
                acceptor_synthon=str(record["acceptor_synthon"]),
                donor_polarization_response_e=donor_response,
                acceptor_polarization_response_e=acceptor_response,
                charge_transfer_e=float(contact["charge_transfer_e"]),
                reference_distance_angstrom=float(
                    contact["reference_distance_angstrom"]
                ),
                reference_mayer_bond_order=(
                    None
                    if contact.get("reference_mayer_bond_order") is None
                    else float(contact["reference_mayer_bond_order"])
                ),
                vdw_radius_sum_angstrom=float(
                    contact["vdw_radius_sum_angstrom"]
                ),
                population_level=str(payload["population_level"]),
            )
        )
    expected = len(standard_hbond_training_complexes())
    if len(templates) != expected:
        raise ValueError(
            f"the resident ORACLE H-bond library must contain {expected} rules"
        )
    return HydrogenBondResponseLibrary(
        templates=tuple(templates),
        population_level=str(payload["population_level"]),
        source=str(resource),
    )


def resident_hbond_response_contacts(
    atomic_numbers: Sequence[int],
    coordinates_angstrom: np.ndarray,
    bonded_pairs: Sequence[tuple[int, int]],
    *,
    library: HydrogenBondResponseLibrary | None = None,
) -> tuple[HydrogenBondChargeContact, ...]:
    """Instantiate every resident rule that exactly matches two fragments.

    The resident library intentionally recognizes only the five training
    fragments.  Unknown synthons remain unresolved rather than receiving an
    invented parameter; later synthon interpolation can extend this contract.
    """

    numbers = tuple(int(value) for value in atomic_numbers)
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    bonds = tuple(tuple(sorted((int(i), int(j)))) for i, j in bonded_pairs)
    selected = library or load_standard_hbond_response_library()
    components = _bond_components(len(numbers), bonds)
    component_by_atom = {
        atom: component
        for component in components
        for atom in component
    }
    contacts: list[HydrogenBondChargeContact] = []
    for perceived in perceive_hydrogen_bonds(numbers, xyz, bonds):
        donor_component = component_by_atom[perceived.donor]
        acceptor_component = component_by_atom[perceived.acceptor]
        if donor_component == acceptor_component:
            continue
        donor_match = _match_training_fragment(
            numbers,
            bonds,
            donor_component,
            required_actual={
                "donor": perceived.donor,
                "hydrogen": perceived.hydrogen,
            },
        )
        acceptor_match = _match_training_fragment(
            numbers,
            bonds,
            acceptor_component,
            required_actual={"acceptor": perceived.acceptor},
        )
        if donor_match is None or acceptor_match is None:
            continue
        donor_name, donor_mapping, donor_synthon = donor_match
        acceptor_name, acceptor_mapping, acceptor_synthon = acceptor_match
        template = selected.template(donor_synthon, acceptor_synthon)
        if (
            template is None
            or template.donor_molecule != donor_name
            or template.acceptor_molecule != acceptor_name
        ):
            continue
        contacts.append(
            template.instantiate(
                donor_mapping=donor_mapping,
                acceptor_mapping=acceptor_mapping,
                donor=perceived.donor,
                hydrogen=perceived.hydrogen,
                acceptor=perceived.acceptor,
            )
        )
    return tuple(contacts)


def _build_formaldehyde_control(
    specification: HydrogenBondTrainingComplex,
    *,
    hydrogen_acceptor_distance_angstrom: float,
    dha_angle_degrees: float,
) -> HydrogenBondTrainingGeometry:
    catalogue = standard_hbond_training_molecules()
    molecule = catalogue["formaldehyde"]
    donor, hydrogen = 0, 2
    acceptor, anchor = 1, 0
    donor_xyz = molecule.coordinates_angstrom.copy()
    direction = _unit(donor_xyz[hydrogen] - donor_xyz[donor])
    approach = _angled_approach(direction, 180.0 - float(dha_angle_degrees))
    acceptor_xyz = _orient_acceptor(
        molecule.coordinates_angstrom,
        acceptor=acceptor,
        anchor=anchor,
        target_outward=-approach,
    )
    acceptor_xyz += (
        donor_xyz[hydrogen]
        + float(hydrogen_acceptor_distance_angstrom) * approach
        - acceptor_xyz[acceptor]
    )
    offset = len(molecule.atoms)
    return HydrogenBondTrainingGeometry(
        specification=specification,
        atoms=molecule.atoms + molecule.atoms,
        coordinates_angstrom=np.vstack((donor_xyz, acceptor_xyz)),
        donor_atoms=tuple(range(offset)),
        acceptor_atoms=tuple(range(offset, 2 * offset)),
        donor=donor,
        hydrogen=hydrogen,
        acceptor=offset + acceptor,
        donor_synthon="formaldehyde_CH_weak_control",
        acceptor_synthon="formaldehyde_carbonyl_O",
        hydrogen_acceptor_distance_angstrom=float(
            hydrogen_acceptor_distance_angstrom
        ),
        dha_angle_degrees=float(dha_angle_degrees),
    )


def _orient_acceptor(
    coordinates: np.ndarray,
    *,
    acceptor: int,
    anchor: int,
    target_outward: np.ndarray,
) -> np.ndarray:
    xyz = np.asarray(coordinates, dtype=float).copy()
    origin = xyz[acceptor].copy()
    outward = _unit(xyz[acceptor] - xyz[anchor])
    target = _unit(np.asarray(target_outward, dtype=float))
    cross = np.cross(outward, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(outward, target), -1.0, 1.0))
    if sine < 1.0e-12:
        if cosine > 0.0:
            rotation = np.eye(3)
        else:
            trial = np.asarray((1.0, 0.0, 0.0))
            if abs(float(np.dot(outward, trial))) > 0.9:
                trial = np.asarray((0.0, 1.0, 0.0))
            axis = _unit(np.cross(outward, trial))
            rotation = 2.0 * np.outer(axis, axis) - np.eye(3)
    else:
        axis = cross / sine
        skew = np.asarray(
            ((0.0, -axis[2], axis[1]),
             (axis[2], 0.0, -axis[0]),
             (-axis[1], axis[0], 0.0))
        )
        rotation = np.eye(3) + sine * skew + (1.0 - cosine) * (skew @ skew)
    return (xyz - origin) @ rotation.T + origin


def _optimize_acceptor_torsion(
    donor_xyz: np.ndarray,
    donor_atoms: Sequence[str],
    acceptor_xyz: np.ndarray,
    acceptor_atoms: Sequence[str],
    *,
    hydrogen: int,
    acceptor: int,
    axis: np.ndarray,
) -> np.ndarray:
    """Choose the least crowded rigid rotation about the intended H...A axis."""

    origin = np.asarray(acceptor_xyz[acceptor], dtype=float)
    unit = _unit(axis)
    best_score: tuple[float, float] | None = None
    best = None
    for step in range(72):
        angle = 2.0 * np.pi * step / 72.0
        skew = np.asarray(
            (
                (0.0, -unit[2], unit[1]),
                (unit[2], 0.0, -unit[0]),
                (-unit[1], unit[0], 0.0),
            )
        )
        rotation = (
            np.eye(3)
            + np.sin(angle) * skew
            + (1.0 - np.cos(angle)) * (skew @ skew)
        )
        candidate = (acceptor_xyz - origin) @ rotation.T + origin
        normalized = []
        for left, left_symbol in enumerate(donor_atoms):
            left_radius = uff_vdw_radius(int(atomic_number(left_symbol) or 0))
            for right, right_symbol in enumerate(acceptor_atoms):
                if left == hydrogen and right == acceptor:
                    continue
                right_radius = uff_vdw_radius(int(atomic_number(right_symbol) or 0))
                if left_radius is None or right_radius is None:
                    continue
                distance = float(np.linalg.norm(donor_xyz[left] - candidate[right]))
                normalized.append(distance / float(left_radius + right_radius))
        if not normalized:
            return candidate
        score = (min(normalized), float(sum(min(value, 1.5) for value in normalized)))
        if best_score is None or score > best_score:
            best_score = score
            best = candidate
    if best is None:
        raise RuntimeError("failed to orient the H-bond training partner")
    return best


def _unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("cannot orient a training site from a zero vector")
    return value / norm


def _angled_approach(direction: np.ndarray, deviation_degrees: float) -> np.ndarray:
    axis = _unit(np.asarray(direction, dtype=float))
    trial = np.asarray((0.0, 0.0, 1.0))
    if abs(float(np.dot(axis, trial))) > 0.9:
        trial = np.asarray((0.0, 1.0, 0.0))
    perpendicular = _unit(np.cross(axis, trial))
    angle = np.deg2rad(float(deviation_degrees))
    return _unit(np.cos(angle) * axis + np.sin(angle) * perpendicular)


def _training_polarization_atoms(
    molecule: str,
    *,
    role: str,
    selected_hydrogen: int | None = None,
) -> tuple[int, ...]:
    """Minimal X--H/CO/NCO support for the initial response library."""

    name = str(molecule)
    if role == "donor":
        if selected_hydrogen is None:
            raise ValueError("donor locality requires the selected hydrogen")
        if name in {
            "water", "ammonia", "hydrogen_sulfide", "phosphine",
            "methanol", "methylamine", "methanethiol", "methylphosphine",
        }:
            return (0, int(selected_hydrogen))
        if name == "formamide":
            return tuple(sorted((0, 1, 2, int(selected_hydrogen))))
    elif role == "acceptor":
        if name == "water":
            return (0, 1, 2)
        if name == "ammonia":
            return (0, 1, 2, 3)
        if name == "formaldehyde":
            return (0, 1)
        if name == "formamide":
            return (0, 1, 2)
        if name == "hydrogen_sulfide":
            return (0, 1, 2)
        if name == "phosphine":
            return (0, 1, 2, 3)
        if name in {"methanol", "methanethiol"}:
            return (0, 1, 2)
        if name in {"methylamine", "methylphosphine"}:
            return (0, 1, 2, 3)
    raise ValueError(f"no minimal {role} polarization support for {name}")


def _bond_components(
    natoms: int,
    bonds: Sequence[tuple[int, int]],
) -> tuple[tuple[int, ...], ...]:
    adjacency = [set() for _ in range(natoms)]
    for left, right in bonds:
        adjacency[left].add(right)
        adjacency[right].add(left)
    unseen = set(range(natoms))
    components = []
    while unseen:
        stack = [min(unseen)]
        component = []
        while stack:
            atom = stack.pop()
            if atom not in unseen:
                continue
            unseen.remove(atom)
            component.append(atom)
            stack.extend(adjacency[atom] & unseen)
        components.append(tuple(sorted(component)))
    return tuple(components)


def _match_training_fragment(
    numbers: tuple[int, ...],
    bonds: Sequence[tuple[int, int]],
    component: tuple[int, ...],
    *,
    required_actual: Mapping[str, int],
) -> tuple[str, tuple[int, ...], str] | None:
    actual_bonds = {
        tuple(sorted((component.index(left), component.index(right))))
        for left, right in bonds
        if left in component and right in component
    }
    actual_numbers = tuple(numbers[atom] for atom in component)
    for name, molecule in standard_hbond_training_molecules().items():
        if len(molecule.atoms) != len(component):
            continue
        template_numbers = tuple(int(atomic_number(atom) or 0) for atom in molecule.atoms)
        if sorted(template_numbers) != sorted(actual_numbers):
            continue
        choices = [
            tuple(index for index, value in enumerate(actual_numbers) if value == z)
            for z in template_numbers
        ]
        for mapping in product(*(permutations(choice, 1) for choice in choices)):
            local_mapping = tuple(item[0] for item in mapping)
            if len(set(local_mapping)) != len(local_mapping):
                continue
            mapped_bonds = {
                tuple(sorted((local_mapping[left], local_mapping[right])))
                for left, right in molecule.bonds
            }
            if mapped_bonds != actual_bonds:
                continue
            global_mapping = tuple(component[index] for index in local_mapping)
            if "donor" in required_actual and not any(
                global_mapping[donor] == required_actual["donor"]
                and global_mapping[hydrogen] == required_actual["hydrogen"]
                for donor, hydrogen, _label in molecule.donor_sites
            ):
                continue
            if "acceptor" in required_actual and not any(
                global_mapping[acceptor] == required_actual["acceptor"]
                for acceptor, _anchor, _label in molecule.acceptor_sites
            ):
                continue
            if "donor" in required_actual:
                synthon = next(
                    label
                    for donor, hydrogen, label in molecule.donor_sites
                    if global_mapping[donor] == required_actual["donor"]
                    and global_mapping[hydrogen] == required_actual["hydrogen"]
                )
            else:
                synthon = next(
                    label
                    for acceptor, _anchor, label in molecule.acceptor_sites
                    if global_mapping[acceptor] == required_actual["acceptor"]
                )
            return name, global_mapping, synthon
    return None


__all__ = [
    "HydrogenBondTrainingComplex",
    "HydrogenBondTrainingGeometry",
    "HydrogenBondTrainingMolecule",
    "HydrogenBondResponseLibrary",
    "HydrogenBondResponseTemplate",
    "HydrogenBondGeometryAudit",
    "WaterCoordinationCluster",
    "build_hbond_training_geometry",
    "audit_hbond_training_geometry",
    "fit_training_geometry_cm5",
    "extended_hbond_transfer_complexes",
    "extended_hbond_training_molecules",
    "formaldehyde_homodimer_control",
    "load_standard_hbond_response_library",
    "resident_hbond_response_contacts",
    "standard_hbond_training_complexes",
    "standard_hbond_training_molecules",
    "water_coordination_response_clusters",
]
