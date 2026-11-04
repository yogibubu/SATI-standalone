"""Unified ORACLE electrostatic perception and FMM charge grouping.

The contract deliberately separates three quantities:

``source_charges``
    the indivisible CM5/Mayer or electronegativity/Pauling observation at a
    declared reference geometry;
``intrinsic_charges``
    the reference charges with every perceived hydrogen-bond polarization and
    charge-transfer contribution removed;
``physical_charges``
    the intrinsic charges plus the hydrogen-bond response at the current
    geometry.

For an environment-polarized CM5 observation, subtraction and reconstruction
use the same contacts, response vectors, and continuous strengths.  The
reference geometry is therefore an exact round trip before any explicitly
audited equivalence/group projection.

Intrinsic charges are typed from charge-independent continuous synthons and
projected onto local charge groups by a minimum-displacement constrained
least-squares solve.  Neutral groups are directly suitable for multipole
evaluation.  A charged fragment retains exactly one explicitly non-neutral
group; net molecular charge is never silently removed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import acos, degrees, exp, sqrt

from matrix_numerics import numerical_matrix_rank
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np

from matrix_chem import (
    DirectionalContact,
    HALOGEN_ATOMIC_NUMBERS,
    XB_ACCEPTOR_ATOMIC_NUMBERS,
    build_topology_objects,
    perceive_directional_contacts,
    perceive_hydrogen_bonds,
    perceive_proton_transfer_bridges,
)

from .hbond_charge_response import (
    ANGSTROM_TO_BOHR,
    HydrogenBondChargeContact,
    evaluate_hydrogen_bond_charge_response,
)


ORACLE_ELECTROSTATICS_SCHEMA = "matrix.oracle.electrostatics.v1"
ORACLE_FRAGMENT_RECONSTRUCTION_SCHEMA = "matrix.oracle.fragment_reconstruction.v2"
ORACLE_REFERENCE_CHARGE_FLUCTUATION_SCHEMA = (
    "matrix.oracle.reference_charge_fluctuation.v1"
)


def _perceive_oracle_hydrogen_bonds(
    atomic_numbers: Sequence[int],
    coordinates_angstrom: np.ndarray,
    bonded_pairs: Sequence[tuple[int, int]],
) -> tuple[object, ...]:
    """Use the shared H-bond kernel and retain symmetric proton bridges.

    A proton nearly equidistant from two N/O/P/S centres is a valid
    proton-transfer contact even when it has no conventional covalent donor.
    The shared kernel already creates the corresponding two-sided contact;
    this explicit audit prevents ORACLE from silently dropping it during
    future charge-response changes.
    """
    contacts = tuple(perceive_hydrogen_bonds(atomic_numbers, coordinates_angstrom, bonded_pairs))
    bridges = perceive_proton_transfer_bridges(
        atomic_numbers, coordinates_angstrom, bonded_pairs
    )
    observed = {(item.donor, item.hydrogen, item.acceptor) for item in contacts}
    missing = tuple(
        bridge for bridge in bridges if (bridge[1], bridge[0], bridge[2]) not in observed
    )
    if missing:
        raise RuntimeError(
            "shared hydrogen-bond kernel dropped central proton-transfer contacts: "
            f"{missing}"
        )
    return contacts
ORACLE_XB_DESCRIPTOR_SCHEMA = "matrix.oracle.xb_descriptor.v1"
DEFAULT_XB_CM5_TRANSFER_E = 0.04
DEFAULT_POPULATION_HALO_DEPTH = 2
ChargeObservation = Literal["auto", "intrinsic", "environment_polarized"]
EquivalenceMode = Literal["synthon", "zeff"]


@dataclass(frozen=True)
class ElectrostaticEquivalenceThresholds:
    """Complete-link thresholds for charge-independent synthon equivalence."""

    z_eff: float = 0.08
    coordination: float = 0.08
    covalency: float = 0.04
    delocalization: float = 0.04
    strain: float = 0.05
    pi_index: float = 0.10
    pi_pi_index: float = 0.10

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not np.isfinite(value) or float(value) <= 0.0:
                raise ValueError(f"{name} equivalence threshold must be finite and positive")


@dataclass(frozen=True)
class ElectrostaticAtomType:
    """One element-preserving class of atoms constrained to one intrinsic charge."""

    identifier: str
    atomic_number: int
    atoms: tuple[int, ...]
    maximum_normalized_spread: float


@dataclass(frozen=True)
class NeutralChargeGroup:
    """A local FMM source group, normally with zero net charge."""

    identifier: str
    atoms: tuple[int, ...]
    target_charge_e: float
    source_charge_e: float
    final_charge_e: float
    neutral: bool


@dataclass(frozen=True)
class ChargeProjectionAudit:
    """Numerical audit of the minimum-displacement charge projection."""

    rms_displacement_e: float
    maximum_displacement_e: float
    maximum_constraint_residual_e: float
    rank: int


@dataclass(frozen=True)
class PerceivedHydrogenBond:
    """Geometry-perceived contact and whether a response rule was available."""

    donor: int
    hydrogen: int
    acceptor: int
    distance_angstrom: float
    angle_degrees: float
    response_resolved: bool
    response_strength: float | None


@dataclass(frozen=True)
class PerceivedHalogenBond:
    """Geometry-only sigma-hole contact used as an ORACLE descriptor.

    The descriptor is deliberately independent of CM5/H-bond charge response:
    it records a covalently bound halogen (``anchor--halogen``) pointing at a
    Lewis-basic atom.  Atom indices are zero based and refer to the ORACLE
    input geometry.
    """

    anchor: int
    halogen: int
    acceptor: int
    distance_angstrom: float
    angle_degrees: float
    strength: float
    schema: str = ORACLE_XB_DESCRIPTOR_SCHEMA


def evaluate_halogen_bond_cm5_response(
    natoms: int,
    contacts: Sequence[PerceivedHalogenBond],
    *,
    transfer_e: float = DEFAULT_XB_CM5_TRANSFER_E,
) -> np.ndarray:
    """Return a simple charge-conserving CM5/XB response vector.

    Unlike the calibrated H-bond map, XB uses one continuous sigma-hole
    transfer amplitude.  It is applied only to the physical charges; the
    intrinsic CM5/synthon state is never modified.
    """

    if natoms < 1 or not np.isfinite(transfer_e) or transfer_e < 0.0:
        raise ValueError("XB CM5 response dimensions or amplitude are invalid")
    delta = np.zeros(natoms, dtype=float)
    for contact in contacts:
        if max(contact.anchor, contact.halogen, contact.acceptor) >= natoms:
            raise IndexError("halogen-bond atom is outside the system")
        amount = float(transfer_e) * float(contact.strength)
        delta[contact.halogen] += amount
        delta[contact.acceptor] -= amount
    return delta


def perceive_halogen_bonds(
    atomic_numbers: Sequence[int],
    coordinates_angstrom: np.ndarray,
    bonded_pairs: Iterable[tuple[int, int]],
    *,
    distance_cutoff_angstrom: float = 3.5,
    distance_width_angstrom: float = 0.25,
    angle_cutoff_degrees: float = 140.0,
    angle_width_degrees: float = 20.0,
    strength_cutoff: float = 0.20,
) -> tuple[PerceivedHalogenBond, ...]:
    """Perceive directional X···A contacts without changing atom typing.

    Halogens are F/Cl/Br/I; acceptors are N/O/S.  The covalent neighbour of a
    halogen supplies the sigma-hole axis.  The returned continuous strength is
    a product of distance and angular factors and is suitable for Pareto
    scoring.  No contact is reported for a directly bonded pair.
    """

    numbers = tuple(int(value) for value in atomic_numbers)
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    if xyz.shape != (len(numbers), 3) or np.any(~np.isfinite(xyz)):
        raise ValueError("coordinates must be a finite (natoms, 3) array")
    if distance_cutoff_angstrom <= 0.0 or distance_width_angstrom <= 0.0:
        raise ValueError("XB distance cutoff and width must be positive")
    if not 0.0 < angle_cutoff_degrees < 180.0 or angle_width_degrees <= 0.0:
        raise ValueError("XB angle parameters are invalid")
    if not 0.0 <= strength_cutoff <= 1.0:
        raise ValueError("XB strength cutoff must lie in [0, 1]")
    bonds = {tuple(sorted((int(left), int(right)))) for left, right in bonded_pairs}
    neighbours: dict[int, list[int]] = {atom: [] for atom in range(len(numbers))}
    for left, right in bonds:
        if left < 0 or right >= len(numbers) or left == right:
            raise ValueError("bonded pairs contain an invalid atom index")
        neighbours[left].append(right)
        neighbours[right].append(left)
    halogens = HALOGEN_ATOMIC_NUMBERS
    acceptors = XB_ACCEPTOR_ATOMIC_NUMBERS
    result: list[PerceivedHalogenBond] = []
    for halogen, atomic_number in enumerate(numbers):
        if atomic_number not in halogens:
            continue
        for anchor in sorted(neighbours[halogen]):
            axis = xyz[anchor] - xyz[halogen]
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm <= 1.0e-12:
                continue
            for acceptor, acceptor_number in enumerate(numbers):
                if acceptor == halogen or acceptor == anchor or acceptor_number not in acceptors:
                    continue
                if tuple(sorted((halogen, acceptor))) in bonds:
                    continue
                target = xyz[acceptor] - xyz[halogen]
                distance = float(np.linalg.norm(target))
                if distance > distance_cutoff_angstrom or distance <= 1.0e-12:
                    continue
                cosine = float(np.dot(axis, target) / (axis_norm * distance))
                angle = degrees(acos(float(np.clip(cosine, -1.0, 1.0))))
                if angle < angle_cutoff_degrees:
                    continue
                radial = exp(-((distance - distance_cutoff_angstrom) / distance_width_angstrom) ** 2)
                angular = exp(-((180.0 - angle) / angle_width_degrees) ** 2)
                strength = float(radial * angular)
                if strength < strength_cutoff:
                    continue
                result.append(
                    PerceivedHalogenBond(
                        anchor=anchor,
                        halogen=halogen,
                        acceptor=acceptor,
                        distance_angstrom=distance,
                        angle_degrees=angle,
                        strength=strength,
                    )
                )
    return tuple(result)


@dataclass(frozen=True)
class PopulationFragmentResult:
    """CM5/Mayer result for one overlapping core-plus-halo fragment.

    ``atoms`` maps every real atom retained in the QM fragment.  ``core_atoms``
    identifies the atoms owned by that fragment and therefore eligible for
    molecular charge reconstruction.  Halo populations are retained only as
    boundary-consistency observations.  An empty ``core_atoms`` preserves the
    legacy contract in which every mapped atom is owned.
    """

    identifier: str
    atoms: tuple[int, ...]
    cm5_charges_e: tuple[float, ...]
    mayer_bond_orders: tuple[tuple[int, int, float], ...]
    core_atoms: tuple[int, ...] = ()
    weight: float = 1.0
    level: str = "L0:PBE0/def2-TZVP"

    def __post_init__(self) -> None:
        if not self.identifier.strip():
            raise ValueError("population fragment identifier cannot be empty")
        if len(self.atoms) != len(self.cm5_charges_e) or not self.atoms:
            raise ValueError("fragment atoms and CM5 charges must be complete and nonempty")
        if len(set(self.atoms)) != len(self.atoms) or min(self.atoms) < 0:
            raise ValueError("fragment target atoms must be unique and nonnegative")
        if (
            len(set(self.core_atoms)) != len(self.core_atoms)
            or any(atom not in set(self.atoms) for atom in self.core_atoms)
        ):
            raise ValueError("fragment core atoms must be unique mapped atoms")
        if not all(np.isfinite(self.cm5_charges_e)):
            raise ValueError("fragment CM5 charges must be finite")
        if not np.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError("fragment population weight must be finite and positive")


@dataclass(frozen=True)
class PopulationFragmentPlan:
    """Graph-derived ownership and halo definition for one QM fragment."""

    identifier: str
    core_atoms: tuple[int, ...]
    atoms: tuple[int, ...]
    graph_depth: tuple[tuple[int, int], ...]
    boundary_bonds: tuple[tuple[int, int], ...]
    schema: str = ORACLE_FRAGMENT_RECONSTRUCTION_SCHEMA


@dataclass(frozen=True)
class PopulationAssembly:
    """Complete molecular CM5/Mayer observables synthesized from overlaps."""

    cm5_charges_e: np.ndarray
    mayer_bond_orders: Mapping[tuple[int, int], float]
    charge_standard_error_e: np.ndarray
    charge_observation_count: np.ndarray
    overlap_observation_count: np.ndarray
    bond_order_observation_count: Mapping[tuple[int, int], int]
    fragment_ids: tuple[str, ...]
    total_charge_e: float
    maximum_total_charge_correction_e: float
    level: str


@dataclass(frozen=True)
class ReferenceChargeFluctuationContract:
    """Versioned CM5-reference/no-H-bond inversion used by every consumer."""

    observation: Literal["intrinsic", "environment_polarized"]
    source_charges_e: np.ndarray
    no_hbond_charges_e: np.ndarray
    reference_response_e: np.ndarray
    contact_triplets: tuple[tuple[int, int, int], ...]
    contact_strengths: tuple[float, ...]
    schema: str = ORACLE_REFERENCE_CHARGE_FLUCTUATION_SCHEMA

    def __post_init__(self) -> None:
        source = np.asarray(self.source_charges_e, dtype=float).reshape(-1)
        baseline = np.asarray(self.no_hbond_charges_e, dtype=float).reshape(-1)
        response = np.asarray(self.reference_response_e, dtype=float).reshape(-1)
        triplets = tuple(
            tuple(int(value) for value in item) for item in self.contact_triplets
        )
        strengths = tuple(float(value) for value in self.contact_strengths)
        if self.observation not in {"intrinsic", "environment_polarized"}:
            raise ValueError("reference charge observation is not resolved")
        if source.shape != baseline.shape or source.shape != response.shape:
            raise ValueError(
                "reference charge fluctuation vectors have inconsistent dimensions"
            )
        if any(np.any(~np.isfinite(item)) for item in (source, baseline, response)):
            raise ValueError("reference charge fluctuation vectors must be finite")
        if len(triplets) != len(strengths):
            raise ValueError(
                "reference contacts and strengths have inconsistent lengths"
            )
        if any(len(item) != 3 or min(item) < 0 for item in triplets):
            raise ValueError(
                "reference contact triplets must contain valid atom indices"
            )
        if len(triplets) != len(set(triplets)):
            raise ValueError("reference contact triplets must be unique")
        if any(
            not np.isfinite(value) or not 0.0 <= value <= 1.0
            for value in strengths
        ):
            raise ValueError(
                "reference contact strengths must lie between zero and one"
            )
        expected_baseline = (
            source - response
            if self.observation == "environment_polarized"
            else source
        )
        if not np.allclose(
            baseline, expected_baseline, atol=2.0e-14, rtol=0.0
        ):
            raise ValueError(
                "reference CM5/no-hydrogen-bond inversion is algebraically "
                "inconsistent"
            )
        object.__setattr__(self, "source_charges_e", source)
        object.__setattr__(self, "no_hbond_charges_e", baseline)
        object.__setattr__(self, "reference_response_e", response)
        object.__setattr__(self, "contact_triplets", triplets)
        object.__setattr__(self, "contact_strengths", strengths)

    @property
    def reference_target_charges_e(self) -> np.ndarray:
        """Charges that the runtime must reproduce at the reference geometry."""

        return self.no_hbond_charges_e + self.reference_response_e

    @property
    def roundtrip_residual_e(self) -> float:
        """Residual of the environment-polarized CM5 inverse/forward pair."""

        target = (
            self.source_charges_e
            if self.observation == "environment_polarized"
            else self.reference_target_charges_e
        )
        return float(np.max(np.abs(self.reference_target_charges_e - target)))

    def to_record(self) -> dict[str, object]:
        """Return a JSON-ready provenance record for serialized ORACLE states."""

        return {
            "schema": self.schema,
            "observation": self.observation,
            "source_charges_e": self.source_charges_e.tolist(),
            "no_hbond_charges_e": self.no_hbond_charges_e.tolist(),
            "reference_response_e": self.reference_response_e.tolist(),
            "reference_target_charges_e": self.reference_target_charges_e.tolist(),
            "contact_triplets": [list(item) for item in self.contact_triplets],
            "contact_strengths": list(self.contact_strengths),
            "roundtrip_residual_e": self.roundtrip_residual_e,
        }


def plan_overlapping_population_fragments(
    natoms: int,
    bonded_pairs: Iterable[tuple[int, int]],
    core_regions: Mapping[str, Sequence[int]],
    *,
    halo_depth: int = DEFAULT_POPULATION_HALO_DEPTH,
    protected_bonds: Iterable[tuple[int, int]] = (),
) -> tuple[PopulationFragmentPlan, ...]:
    """Dilate disjoint atom cores into overlapping, graph-local QM fragments.

    Every molecular atom has exactly one core owner.  The retained halo extends
    by ``halo_depth`` covalent-graph edges; caps, when required by a backend,
    are placed only on bonds crossing the outer halo boundary.
    """

    if natoms < 1:
        raise ValueError("population fragment planning needs at least one atom")
    if halo_depth < 1:
        raise ValueError("population fragment halos require depth >= 1")
    if not core_regions:
        raise ValueError("population fragment planning needs core regions")
    adjacency = [set() for _ in range(natoms)]
    normalized_bonds: set[tuple[int, int]] = set()
    for raw_left, raw_right in bonded_pairs:
        left, right = int(raw_left), int(raw_right)
        if left == right or min(left, right) < 0 or max(left, right) >= natoms:
            raise ValueError(f"invalid population-fragment bond: {(left, right)}")
        pair = tuple(sorted((left, right)))
        normalized_bonds.add(pair)
        adjacency[left].add(right)
        adjacency[right].add(left)
    protected = {
        tuple(sorted((int(left), int(right))))
        for left, right in protected_bonds
    }
    unknown_protected = protected - normalized_bonds
    if unknown_protected:
        raise ValueError("protected fragment bonds must be accepted covalent bonds")
    owner = np.full(natoms, -1, dtype=int)
    normalized_cores: list[tuple[str, tuple[int, ...]]] = []
    for region_index, (identifier, raw_core) in enumerate(core_regions.items()):
        core = tuple(sorted({int(atom) for atom in raw_core}))
        if not str(identifier).strip() or not core:
            raise ValueError("every population fragment needs an identifier and core")
        if min(core) < 0 or max(core) >= natoms:
            raise ValueError(f"fragment core {identifier} contains an out-of-range atom")
        for atom in core:
            if owner[atom] >= 0:
                raise ValueError(f"atom {atom} has more than one fragment core owner")
            owner[atom] = region_index
        normalized_cores.append((str(identifier), core))
    missing = tuple(int(atom) for atom in np.flatnonzero(owner < 0))
    if missing:
        raise ValueError(
            "population fragment cores leave atoms unowned: "
            + ", ".join(str(atom) for atom in missing)
        )
    plans = []
    for identifier, core in normalized_cores:
        depths = {atom: 0 for atom in core}
        frontier = set(core)
        depth = 0
        while frontier:
            depth += 1
            frontier = {
                neighbor
                for atom in frontier
                for neighbor in adjacency[atom]
                if neighbor not in depths
            }
            for atom in frontier:
                depths[atom] = depth
        atom_set = {atom for atom, distance in depths.items() if distance <= halo_depth}
        changed = True
        while changed:
            changed = False
            for left, right in protected:
                if (left in atom_set) == (right in atom_set):
                    continue
                atom_set.update((left, right))
                changed = True
        atoms = tuple(sorted(atom_set))
        boundary = tuple(
            pair
            for pair in sorted(normalized_bonds)
            if (pair[0] in atom_set) != (pair[1] in atom_set)
        )
        plans.append(
            PopulationFragmentPlan(
                identifier=identifier,
                core_atoms=core,
                atoms=atoms,
                graph_depth=tuple(
                    sorted((atom, depths[atom]) for atom in atom_set)
                ),
                boundary_bonds=boundary,
            )
        )
    if len(plans) > 1 and not _overlap_graph_connected(
        [set(plan.atoms) for plan in plans]
    ):
        raise ValueError("planned population fragments do not form a connected overlap graph")
    return tuple(plans)


@dataclass(frozen=True)
class OracleElectrostatics:
    """Backend-neutral electrostatic state consumed by ZAFF and other tools."""

    source_charges_e: np.ndarray
    intrinsic_charges_e: np.ndarray
    physical_charges_e: np.ndarray
    hydrogen_bond_polarization_e: np.ndarray
    hydrogen_bond_charge_transfer_e: np.ndarray
    halogen_bond_cm5_response_e: np.ndarray
    atom_type_ids: tuple[str, ...]
    atom_types: tuple[ElectrostaticAtomType, ...]
    charge_groups: tuple[NeutralChargeGroup, ...]
    perceived_hydrogen_bonds: tuple[PerceivedHydrogenBond, ...]
    perceived_halogen_bonds: tuple[PerceivedHalogenBond, ...]
    projection: ChargeProjectionAudit
    reference_fluctuation: ReferenceChargeFluctuationContract
    charge_source: str
    bond_order_source: str
    charge_observation: ChargeObservation
    directional_bond_cm5_response_e: np.ndarray | None = None
    perceived_directional_bonds: tuple[DirectionalContact, ...] = ()
    schema: str = ORACLE_ELECTROSTATICS_SCHEMA

    @property
    def unresolved_hydrogen_bonds(self) -> tuple[PerceivedHydrogenBond, ...]:
        return tuple(
            contact
            for contact in self.perceived_hydrogen_bonds
            if not contact.response_resolved
        )

    @property
    def reference_reconstruction_residual_e(self) -> float:
        """Largest source-minus-reconstructed charge at the reference geometry."""

        return self.reference_fluctuation.roundtrip_residual_e

    @property
    def projected_reference_residual_e(self) -> float:
        """Additional residual introduced by the audited type/group projection."""

        return float(
            np.max(
                np.abs(
                    self.physical_charges_e
                    - self.reference_fluctuation.reference_target_charges_e
                )
            )
        )


def standard_cm5_mayer_request(
    *,
    accuracy_level: Literal["L0", "L1"] = "L0",
    method: str | None = None,
    atomic_numbers: Sequence[int] = (),
    basis: str | None = None,
) -> dict[str, object]:
    """Return the portable population request understood by MATRIX QM adapters.

    ORACLE does not execute a particular electronic backend.  This record asks
    the workflow layer for one single-point density followed by APOC CM5 and
    Mayer analysis.  Both observables are required as one indivisible level.
    """

    numbers = tuple(int(value) for value in atomic_numbers)
    if numbers and (min(numbers) < 1 or max(numbers) > 86):
        raise ValueError(
            "the def2 CM5/Mayer profile supports H through Rn; "
            "elements beyond Rn require an explicit future profile"
        )
    level = str(accuracy_level).strip().upper()
    if level not in {"L0", "L1"}:
        raise ValueError("the population accuracy level must be L0 or L1")
    selected_method = (
        ("PBE0" if level == "L0" else "MP2")
        if method is None
        else str(method).strip().upper()
    )
    if level == "L0" and selected_method != "PBE0":
        raise ValueError("the standard L0 population method is PBE0")
    if level == "L1" and selected_method != "MP2":
        raise ValueError("the standard L1 population method is MP2")
    if level == "L1" and basis is None:
        raise ValueError(
            "the L1 MP2 orbital basis has not been standardized; "
            "supply it explicitly until the L1 contract is frozen"
        )
    maximum_z = max(numbers, default=36)
    default_basis = "def2-TZVP"
    hamiltonian = "ECP_SCALAR_RELATIVISTIC" if maximum_z > 36 else "NONRELATIVISTIC"
    basis_domain = "H-Rn"
    pseudopotential = "def2-ECP_Rb-Rn" if maximum_z > 36 else "NONE_ALL_ELECTRON"
    selected_basis = default_basis if basis is None else str(basis)
    if not selected_basis.strip():
        raise ValueError("the standard CM5/Mayer request needs a method and basis")
    return {
        "schema": "matrix.qm.population_request.v1",
        "task": "single_point_population",
        "accuracy_level": level,
        "method": selected_method,
        "basis": selected_basis,
        "basis_domain": basis_domain,
        "hamiltonian": hamiltonian,
        "pseudopotential": pseudopotential,
        "relativistic_scope": (
            "HEAVY_ATOMS_NATIVE_DEF2_ECP"
            if hamiltonian != "NONRELATIVISTIC"
            else "NOT_APPLICABLE"
        ),
        "geometry": "FIXED_ORACLE_REFERENCE",
        "environment": "GAS_PHASE",
        "scf": "TIGHT",
        "integration_grid": "ULTRAFINE_OR_BACKEND_EQUIVALENT",
        "required_observables": ["CM5_CHARGES", "MAYER_BOND_ORDERS"],
        "required_electronic_state": [
            "AO_OVERLAP",
            "AO_DENSITY",
            "MO_COEFFICIENTS",
            "MO_OCCUPATIONS",
            "OCCUPIED_VIRTUAL_PARTITION",
            "ONE_PARTICLE_DENSITY",
            "NATURAL_ORBITALS",
            "NATURAL_OCCUPATIONS",
        ],
        "natural_orbital_contract": (
            "ONE_RDM_NATURAL_ORBITALS_PNO_REQUIRES_PAIR_CONSTRUCTION"
            if level == "L0"
            else "MP2_ONE_RDM_NATURAL_ORBITALS_AND_PAIR_DENSITIES_PNO_READY"
        ),
        "pair_correlation_artifacts": (
            []
            if level == "L0"
            else ["MP2_T2_AMPLITUDES", "MP2_PAIR_DENSITIES"]
        ),
        "basis_convergence_contract": (
            "SINGLE_L0_REFERENCE_NOT_FOR_SYSTEMATIC_EXTRAPOLATION"
            if level == "L0"
            else "MP2_CARDINAL_FAMILY_AND_EXTRAPOLATION_TO_BE_STANDARDIZED"
        ),
        "extrapolation_role": (
            "REFERENCE_ONLY"
            if level == "L0"
            else "PRIMARY_CORRELATED_EXTRAPOLATION_LEVEL"
        ),
        "population_analyzer": "APOC",
        "fallback": "ORACLE_ELECTRONEGATIVITY_PAULING",
        "density_contract": (
            "PBE0_KOHN_SHAM_DENSITY"
            if level == "L0"
            else "MP2_ONE_PARTICLE_DENSITY"
        ),
        "hirshfeld_quadrature": "NUMERICAL_FOR_BOTH_PBE0_AND_MP2",
        "spin_orbit_policy": "ENCODED_IN_DEF2_ECP_AVERAGE_NOT_EXPLICIT",
        "pl1_compatibility": (
            "NOT_APPLICABLE_L0"
            if level == "L0"
            else "REQUIRES_RECALIBRATION_FROM_LEGACY_DPCS3_L1"
        ),
    }


def assemble_overlapping_cm5_mayer(
    natoms: int,
    bonded_pairs: Iterable[tuple[int, int]],
    fragments: Sequence[PopulationFragmentResult],
    *,
    total_charge_e: float,
    required_level: str = "L0:PBE0/def2-TZVP",
) -> PopulationAssembly:
    """Synthesize a complete molecular population state from overlaps.

    Every atom has a core owner and every perceived covalent bond occurs in at
    least one retained fragment.  Core observations reconstruct molecular
    charges; halo observations quantify boundary consistency without
    contaminating the reconstruction.  Mayer orders use the most interior
    available observations.  Only after assembly is the minimum weighted
    displacement applied to recover the declared molecular charge.
    """

    if natoms < 1:
        raise ValueError("population assembly needs at least one atom")
    if not fragments:
        raise ValueError("population assembly needs fragment results")
    if len(fragments) > 1 and any(not fragment.core_atoms for fragment in fragments):
        raise ValueError(
            "multi-fragment population assembly requires explicit core ownership; "
            "legacy boundary averaging is read-only"
        )
    if not np.isfinite(total_charge_e):
        raise ValueError("molecular charge must be finite")
    normalized_level = str(required_level).strip().upper()
    if not normalized_level:
        raise ValueError("required population level cannot be empty")
    charge_weight = np.zeros(natoms)
    core_owner_count = np.zeros(natoms, dtype=int)
    charge_sum = np.zeros(natoms)
    overlap_weight = np.zeros(natoms)
    overlap_sum = np.zeros(natoms)
    overlap_square_sum = np.zeros(natoms)
    order_values: dict[tuple[int, int], list[tuple[int, float, float]]] = {}
    identifiers = []
    atom_sets = []
    for fragment in fragments:
        if fragment.level.strip().upper() != normalized_level:
            raise ValueError(
                f"fragment {fragment.identifier} uses {fragment.level}, "
                f"expected the uniform level {required_level}"
            )
        if max(fragment.atoms) >= natoms:
            raise ValueError(f"fragment {fragment.identifier} contains an out-of-range atom")
        identifiers.append(fragment.identifier)
        atom_sets.append(set(fragment.atoms))
        core = set(fragment.core_atoms or fragment.atoms)
        for atom, charge in zip(fragment.atoms, fragment.cm5_charges_e, strict=True):
            weight = float(fragment.weight)
            overlap_weight[atom] += weight
            overlap_sum[atom] += weight * float(charge)
            overlap_square_sum[atom] += weight * float(charge) ** 2
            if atom in core:
                core_owner_count[atom] += 1
                charge_weight[atom] += weight
                charge_sum[atom] += weight * float(charge)
        allowed = set(fragment.atoms)
        for left, right, value in fragment.mayer_bond_orders:
            pair = tuple(sorted((int(left), int(right))))
            if pair[0] == pair[1] or not set(pair).issubset(allowed):
                raise ValueError(
                    f"fragment {fragment.identifier} has a Mayer pair outside its target atoms"
                )
            if not np.isfinite(value):
                raise ValueError("fragment Mayer bond orders must be finite")
            interior_rank = int(pair[0] in core) + int(pair[1] in core)
            order_values.setdefault(pair, []).append(
                (interior_rank, float(fragment.weight), float(value))
            )
    missing_atoms = tuple(int(atom) for atom in np.flatnonzero(charge_weight == 0.0))
    if missing_atoms:
        raise ValueError(
            "overlapping population fragments leave atoms undefined: "
            + ", ".join(str(atom) for atom in missing_atoms)
        )
    multiply_owned = tuple(int(atom) for atom in np.flatnonzero(core_owner_count > 1))
    if multiply_owned:
        raise ValueError(
            "population fragment cores overlap; every atom needs one owner: "
            + ", ".join(str(atom) for atom in multiply_owned)
        )
    if len(fragments) > 1 and not _overlap_graph_connected(atom_sets):
        raise ValueError(
            "population fragments must form one connected overlap graph; "
            "disjoint tiling cannot control boundary consistency"
        )
    required_bonds = {
        tuple(sorted((int(left), int(right)))) for left, right in bonded_pairs
    }
    missing_bonds = sorted(required_bonds - set(order_values))
    if missing_bonds:
        raise ValueError(
            "overlapping population fragments leave Mayer bonds undefined: "
            + ", ".join(f"{left}-{right}" for left, right in missing_bonds)
        )
    charges = charge_sum / charge_weight
    overlap_mean = overlap_sum / overlap_weight
    variance = np.maximum(
        0.0,
        overlap_square_sum / overlap_weight - overlap_mean**2,
    )
    count = np.zeros(natoms, dtype=int)
    overlap_count = np.zeros(natoms, dtype=int)
    for fragment, atoms in zip(fragments, atom_sets, strict=True):
        overlap_count[list(atoms)] += 1
        count[list(fragment.core_atoms or fragment.atoms)] += 1
    standard_error = np.sqrt(variance / np.maximum(1, overlap_count))
    residual = float(np.sum(charges) - total_charge_e)
    inverse_weight = 1.0 / charge_weight
    correction = residual * inverse_weight / float(np.sum(inverse_weight))
    charges -= correction
    orders = {}
    for pair, values in order_values.items():
        if pair not in required_bonds:
            continue
        best_rank = max(rank for rank, _weight, _value in values)
        selected = [
            (weight, value)
            for rank, weight, value in values
            if rank == best_rank
        ]
        orders[pair] = float(
            sum(weight * value for weight, value in selected)
            / sum(weight for weight, _value in selected)
        )
    return PopulationAssembly(
        cm5_charges_e=charges,
        mayer_bond_orders=orders,
        charge_standard_error_e=standard_error,
        charge_observation_count=count,
        overlap_observation_count=overlap_count,
        bond_order_observation_count={
            pair: len(order_values[pair]) for pair in sorted(required_bonds)
        },
        fragment_ids=tuple(identifiers),
        total_charge_e=float(total_charge_e),
        maximum_total_charge_correction_e=float(np.max(np.abs(correction))),
        level=str(required_level),
    )


def prepare_oracle_electrostatics(
    atomic_numbers: Sequence[int],
    coordinates_angstrom: np.ndarray,
    *,
    cm5_charges_e: Sequence[float] | Mapping[int, float] | None = None,
    mayer_bond_orders: Mapping[tuple[int, int], float] | None = None,
    charge_observation: ChargeObservation = "auto",
    hydrogen_bond_responses: Sequence[HydrogenBondChargeContact] | None = None,
    directional_transfer_by_kind_e: Mapping[str, float] | None = None,
    equivalence_mode: EquivalenceMode = "synthon",
    equivalence_thresholds: ElectrostaticEquivalenceThresholds | None = None,
    maximum_group_charge_shift_e: float = 0.12,
    total_charge_e: float | None = None,
) -> OracleElectrostatics:
    """Build the complete ORACLE electrostatic state.

    A complete CM5 vector is accepted only together with Mayer values for all
    perceived covalent bonds. Otherwise the whole structure uses ORACLE
    electronegativity charges and Pauling bond orders.

    In ``auto`` mode, supplied CM5 values are observations at
    ``coordinates_angstrom``. If that reference contains hydrogen bonds,
    ORACLE evaluates their continuous strengths and subtracts the
    corresponding calibrated polarization and charge-transfer vectors to
    recover the no-hydrogen-bond baseline. Runtime charges are reconstructed
    from that baseline with the same response map, so the reference geometry
    returns the supplied CM5 vector. When no CM5 vector is supplied, ``auto``
    treats the electronegativity estimate as intrinsic. Pass
    ``charge_observation="intrinsic"`` only when supplied charges already
    represent the no-hydrogen-bond baseline.

    Response rules must refer to contacts in the current atom indexing. When
    no explicit sequence is supplied, ORACLE instantiates every matching rule
    from its resident paired-CM5 library. Passing an empty tuple explicitly
    disables the correction.

    ``directional_transfer_by_kind_e`` enables additional conservative
    chalcogen-, pnictogen- or tetrel-contact CM5 responses.  Every amplitude
    must be supplied explicitly; ``None`` preserves the resident halogen-bond
    response and an empty mapping disables every non-H-bond response.
    """

    numbers = tuple(int(value) for value in atomic_numbers)
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    if xyz.shape != (len(numbers), 3) or np.any(~np.isfinite(xyz)):
        raise ValueError("coordinates must be a finite (natoms, 3) array")
    if not numbers or min(numbers) <= 0:
        raise ValueError("atomic numbers must be positive")
    if charge_observation not in {"auto", "intrinsic", "environment_polarized"}:
        raise ValueError(
            "charge_observation must be auto, intrinsic or environment_polarized"
        )
    effective_charge_observation: ChargeObservation = (
        "environment_polarized"
        if charge_observation == "auto" and cm5_charges_e is not None
        else "intrinsic"
        if charge_observation == "auto"
        else charge_observation
    )
    if equivalence_mode not in {"synthon", "zeff"}:
        raise ValueError("equivalence_mode must be synthon or zeff")
    if (
        not np.isfinite(maximum_group_charge_shift_e)
        or maximum_group_charge_shift_e <= 0.0
    ):
        raise ValueError("maximum group charge shift must be finite and positive")

    external_charges = _complete_charge_mapping(cm5_charges_e, len(numbers))
    external_orders = {
        tuple(sorted((int(left), int(right)))): float(value)
        for (left, right), value in dict(mayer_bond_orders or {}).items()
    }
    _continuous, graph, _rings, synthons, _aromaticity = build_topology_objects(
        xyz,
        numbers,
        external_charges=external_charges,
        bond_order_overrides=external_orders,
        charge_source="APOC CM5",
        bond_order_source="APOC Mayer",
    )
    source = np.asarray([synthons.charge(atom) for atom in range(len(numbers))])
    bonds = tuple(tuple(sorted(pair)) for pair in graph.bonds)
    perceived_raw = _perceive_oracle_hydrogen_bonds(numbers, xyz, bonds)
    perceived_xb = perceive_halogen_bonds(numbers, xyz, bonds)
    directional_amplitudes = (
        {"halogen-bond": DEFAULT_XB_CM5_TRANSFER_E}
        if directional_transfer_by_kind_e is None
        else {
            str(kind).strip().lower().replace("_", "-"): float(value)
            for kind, value in directional_transfer_by_kind_e.items()
        }
    )
    if any(
        not np.isfinite(value) or value < 0.0
        for value in directional_amplitudes.values()
    ):
        raise ValueError("directional CM5 transfer amplitudes must be finite and non-negative")
    xb_delta = (
        evaluate_halogen_bond_cm5_response(
            len(numbers),
            perceived_xb,
            transfer_e=directional_amplitudes["halogen-bond"],
        )
        if "halogen-bond" in directional_amplitudes
        else np.zeros(len(numbers))
    )
    extra_kinds = tuple(
        kind for kind in directional_amplitudes if kind != "halogen-bond"
    )
    perceived_directional = (
        perceive_directional_contacts(numbers, xyz, bonds, kinds=extra_kinds)
        if extra_kinds
        else ()
    )
    directional_delta = xb_delta.copy()
    for contact in perceived_directional:
        amount = directional_amplitudes[contact.kind] * contact.strength
        directional_delta[contact.center] += amount
        directional_delta[contact.acceptor] -= amount
    if hydrogen_bond_responses is None:
        from .hbond_training import resident_hbond_response_contacts

        selected_responses: Sequence[HydrogenBondChargeContact] = (
            resident_hbond_response_contacts(numbers, xyz, bonds)
        )
    else:
        selected_responses = hydrogen_bond_responses
    response_by_triplet = {
        (item.donor, item.hydrogen, item.acceptor): item
        for item in selected_responses
    }
    if len(response_by_triplet) != len(selected_responses):
        raise ValueError("hydrogen-bond response triplets must be unique")
    perceived_triplets = {
        (item.donor, item.hydrogen, item.acceptor) for item in perceived_raw
    }
    extra = sorted(set(response_by_triplet) - perceived_triplets)
    if extra:
        raise ValueError(
            "response rules do not match perceived hydrogen bonds: "
            + ", ".join(f"{donor}-{hydrogen}...{acceptor}" for donor, hydrogen, acceptor in extra)
        )
    resolved_responses = tuple(
        response_by_triplet[triplet]
        for triplet in sorted(response_by_triplet)
    )
    if resolved_responses:
        response = evaluate_hydrogen_bond_charge_response(
            xyz * ANGSTROM_TO_BOHR,
            np.zeros(len(numbers)),
            resolved_responses,
        )
        polarization = response.polarization_delta_e.copy()
        transfer = response.charge_transfer_delta_e.copy()
        strength_by_triplet = {
            (item.donor, item.hydrogen, item.acceptor): float(strength)
            for item, strength in zip(
                resolved_responses,
                response.contact_strengths,
                strict=True,
            )
        }
    else:
        polarization = np.zeros(len(numbers))
        transfer = np.zeros(len(numbers))
        strength_by_triplet = {}
    hbond_delta = polarization + transfer
    raw_intrinsic = (
        source.copy()
        if effective_charge_observation == "intrinsic"
        else source - hbond_delta
    )
    reference_fluctuation = ReferenceChargeFluctuationContract(
        observation=effective_charge_observation,
        source_charges_e=source,
        no_hbond_charges_e=raw_intrinsic,
        reference_response_e=hbond_delta,
        contact_triplets=tuple(
            (item.donor, item.hydrogen, item.acceptor)
            for item in resolved_responses
        ),
        contact_strengths=tuple(
            strength_by_triplet[
                (item.donor, item.hydrogen, item.acceptor)
            ]
            for item in resolved_responses
        ),
    )

    policy = equivalence_thresholds or ElectrostaticEquivalenceThresholds()
    atom_types = _electrostatic_atom_types(numbers, synthons, policy, equivalence_mode)
    groups = _build_local_charge_groups(
        numbers,
        graph.adjacency,
        raw_intrinsic,
        maximum_group_charge_shift_e=float(maximum_group_charge_shift_e),
        total_charge_e=total_charge_e,
    )
    projected, audit = _minimum_displacement_projection(
        raw_intrinsic,
        tuple(item.atoms for item in atom_types),
        tuple((item[0], item[1]) for item in groups),
    )
    physical = projected + hbond_delta + directional_delta
    type_ids = [""] * len(numbers)
    for item in atom_types:
        for atom in item.atoms:
            type_ids[atom] = item.identifier
    charge_groups = tuple(
        NeutralChargeGroup(
            identifier=f"G{index + 1}",
            atoms=atoms,
            target_charge_e=target,
            source_charge_e=float(np.sum(raw_intrinsic[list(atoms)])),
            final_charge_e=float(np.sum(projected[list(atoms)])),
            neutral=bool(abs(target) <= 1.0e-12),
        )
        for index, (atoms, target) in enumerate(groups)
    )
    perceived = tuple(
        PerceivedHydrogenBond(
            donor=item.donor,
            hydrogen=item.hydrogen,
            acceptor=item.acceptor,
            distance_angstrom=float(item.distance_angstrom),
            angle_degrees=float(np.degrees(item.angle_radians)),
            response_resolved=(
                item.donor,
                item.hydrogen,
                item.acceptor,
            )
            in response_by_triplet,
            response_strength=strength_by_triplet.get(
                (item.donor, item.hydrogen, item.acceptor)
            ),
        )
        for item in perceived_raw
    )
    return OracleElectrostatics(
        source_charges_e=source,
        intrinsic_charges_e=projected,
        physical_charges_e=physical,
        hydrogen_bond_polarization_e=polarization,
        hydrogen_bond_charge_transfer_e=transfer,
        halogen_bond_cm5_response_e=xb_delta,
        atom_type_ids=tuple(type_ids),
        atom_types=atom_types,
        charge_groups=charge_groups,
        perceived_hydrogen_bonds=perceived,
        perceived_halogen_bonds=perceived_xb,
        projection=audit,
        reference_fluctuation=reference_fluctuation,
        charge_source=str(getattr(synthons, "_charge_source", "")),
        bond_order_source=str(getattr(synthons, "_bond_order_source", "")),
        charge_observation=effective_charge_observation,
        directional_bond_cm5_response_e=directional_delta,
        perceived_directional_bonds=perceived_directional,
    )


def _complete_charge_mapping(
    values: Sequence[float] | Mapping[int, float] | None,
    natoms: int,
) -> dict[int, float]:
    if values is None:
        return {}
    if isinstance(values, Mapping):
        result = {int(atom): float(value) for atom, value in values.items()}
    else:
        array = np.asarray(values, dtype=float).reshape(-1)
        result = {atom: float(value) for atom, value in enumerate(array)}
    if set(result) != set(range(natoms)) or any(not np.isfinite(value) for value in result.values()):
        raise ValueError("CM5 requires one finite charge for every atom")
    return result


def _electrostatic_atom_types(
    numbers: tuple[int, ...],
    synthons: object,
    thresholds: ElectrostaticEquivalenceThresholds,
    mode: EquivalenceMode,
) -> tuple[ElectrostaticAtomType, ...]:
    descriptor_names = (
        ("z_eff",)
        if mode == "zeff"
        else (
            "coordination",
            "covalency",
            "delocalization",
            "strain",
            "pi_index",
            "pi_pi_index",
        )
    )
    descriptors = []
    for atom in range(len(numbers)):
        descriptors.append(
            {
                "z_eff": float(synthons.Zeff(atom)),
                "coordination": float(synthons.cna(atom)),
                "covalency": float(synthons.covalency(atom)),
                "delocalization": float(synthons.delocalization(atom)),
                "radial_dispersion": float(synthons.radial_dispersion(atom)),
                "strain": float(synthons.strain(atom)),
                "pi_index": float(synthons.pi_index(atom)),
                "pi_pi_index": float(synthons.pi_pi_index(atom)),
            }
        )
    clusters: list[list[int]] = [[atom] for atom in range(len(numbers))]
    while True:
        candidates: list[tuple[float, tuple[int, ...], int, int]] = []
        for left in range(len(clusters)):
            for right in range(left + 1, len(clusters)):
                combined = sorted(clusters[left] + clusters[right])
                if len({numbers[atom] for atom in combined}) != 1:
                    continue
                spreads = []
                admissible = True
                for name in descriptor_names:
                    values = [descriptors[atom][name] for atom in combined]
                    spread = (max(values) - min(values)) / float(getattr(thresholds, name))
                    if spread > 1.0 + 1.0e-12:
                        admissible = False
                        break
                    spreads.append(spread)
                if admissible:
                    candidates.append((max(spreads, default=0.0), tuple(combined), left, right))
        if not candidates:
            break
        _score, _atoms, left, right = min(candidates)
        clusters[left] = sorted(clusters[left] + clusters[right])
        del clusters[right]
    clusters.sort(key=min)
    counts: dict[int, int] = {}
    result = []
    for cluster in clusters:
        atomic_number = numbers[cluster[0]]
        counts[atomic_number] = counts.get(atomic_number, 0) + 1
        spreads = []
        for name in descriptor_names:
            values = [descriptors[atom][name] for atom in cluster]
            spreads.append(
                (max(values) - min(values)) / float(getattr(thresholds, name))
            )
        result.append(
            ElectrostaticAtomType(
                identifier=f"Z{atomic_number}_{counts[atomic_number]}",
                atomic_number=atomic_number,
                atoms=tuple(cluster),
                maximum_normalized_spread=max(spreads, default=0.0),
            )
        )
    return tuple(result)


def _build_local_charge_groups(
    numbers: tuple[int, ...],
    adjacency: Sequence[Iterable[int]],
    charges: np.ndarray,
    *,
    maximum_group_charge_shift_e: float,
    total_charge_e: float | None,
) -> tuple[tuple[tuple[int, ...], float], ...]:
    neighbors = [set(int(item) for item in row) for row in adjacency]
    fragments = _connected_components(neighbors)
    molecular_charge = (
        float(np.rint(np.sum(charges)))
        if total_charge_e is None
        else float(total_charge_e)
    )
    if not np.isfinite(molecular_charge):
        raise ValueError("total charge must be finite")
    fragment_sums = np.asarray([np.sum(charges[list(fragment)]) for fragment in fragments])
    fragment_targets = np.rint(fragment_sums)
    mismatch = molecular_charge - float(np.sum(fragment_targets))
    if abs(mismatch) > 1.0e-10:
        order = np.argsort(-np.abs(fragment_sums - fragment_targets))
        step = 1.0 if mismatch > 0.0 else -1.0
        for index in order:
            if abs(mismatch) <= 1.0e-10:
                break
            fragment_targets[index] += step
            mismatch -= step
    if abs(mismatch) > 1.0e-8:
        raise ValueError("fragment charges cannot be reconciled with the molecular charge")

    result: list[tuple[tuple[int, ...], float]] = []
    for fragment, fragment_target in zip(fragments, fragment_targets, strict=True):
        unassigned = set(fragment)
        local: list[set[int]] = []
        for atom in fragment:
            if numbers[atom] == 1 or atom not in unassigned:
                continue
            group = {atom} | {
                neighbor
                for neighbor in neighbors[atom]
                if neighbor in unassigned and numbers[neighbor] == 1
            }
            local.append(group)
            unassigned -= group
        local.extend({atom} for atom in sorted(unassigned))
        group_target = [0.0] * len(local)
        if abs(fragment_target) > 1.0e-12:
            best = min(
                range(len(local)),
                key=lambda index: abs(
                    (float(np.sum(charges[list(local[index])])) - fragment_target)
                    / sqrt(len(local[index]))
                ),
            )
            group_target[best] = float(fragment_target)
        while len(local) > 1:
            shifts = [
                abs(float(np.sum(charges[list(group)])) - target) / sqrt(len(group))
                for group, target in zip(local, group_target, strict=True)
            ]
            worst = int(np.argmax(shifts))
            if shifts[worst] <= maximum_group_charge_shift_e:
                break
            adjacent = [
                index
                for index, group in enumerate(local)
                if index != worst
                and any(neighbor in group for atom in local[worst] for neighbor in neighbors[atom])
            ]
            if not adjacent:
                adjacent = [index for index in range(len(local)) if index != worst]
            partner = min(
                adjacent,
                key=lambda index: (
                    abs(
                        float(np.sum(charges[list(local[worst] | local[index])]))
                        - group_target[worst]
                        - group_target[index]
                    )
                    / sqrt(len(local[worst] | local[index])),
                    min(local[index]),
                ),
            )
            keep, drop = sorted((worst, partner))
            local[keep] |= local[drop]
            group_target[keep] += group_target[drop]
            del local[drop]
            del group_target[drop]
        result.extend(
            (tuple(sorted(group)), float(target))
            for group, target in zip(local, group_target, strict=True)
        )
    result.sort(key=lambda item: min(item[0]))
    return tuple(result)


def _minimum_displacement_projection(
    charges: np.ndarray,
    equivalent_classes: Sequence[Sequence[int]],
    group_constraints: Sequence[tuple[Sequence[int], float]],
) -> tuple[np.ndarray, ChargeProjectionAudit]:
    q0 = np.asarray(charges, dtype=float).reshape(-1)
    rows: list[np.ndarray] = []
    targets: list[float] = []
    for atoms in equivalent_classes:
        ordered = tuple(sorted(int(atom) for atom in atoms))
        for atom in ordered[1:]:
            row = np.zeros(len(q0))
            row[ordered[0]] = 1.0
            row[atom] = -1.0
            rows.append(row)
            targets.append(0.0)
    for atoms, target in group_constraints:
        row = np.zeros(len(q0))
        row[list(atoms)] = 1.0
        rows.append(row)
        targets.append(float(target))
    if not rows:
        return q0.copy(), ChargeProjectionAudit(0.0, 0.0, 0.0, 0)
    matrix = np.vstack(rows)
    rhs = np.asarray(targets)
    gram = matrix @ matrix.T
    correction = matrix.T @ (np.linalg.pinv(gram, rcond=1.0e-12) @ (matrix @ q0 - rhs))
    projected = q0 - correction
    residual = matrix @ projected - rhs
    delta = projected - q0
    return projected, ChargeProjectionAudit(
        rms_displacement_e=float(np.sqrt(np.mean(delta**2))),
        maximum_displacement_e=float(np.max(np.abs(delta))),
        maximum_constraint_residual_e=float(np.max(np.abs(residual))),
        rank=numerical_matrix_rank(
            matrix,
            relative_tolerance=np.finfo(float).eps * max(matrix.shape),
        ),
    )


def _connected_components(adjacency: Sequence[set[int]]) -> tuple[tuple[int, ...], ...]:
    unseen = set(range(len(adjacency)))
    result = []
    while unseen:
        root = min(unseen)
        stack = [root]
        unseen.remove(root)
        component = []
        while stack:
            atom = stack.pop()
            component.append(atom)
            for neighbor in sorted(adjacency[atom], reverse=True):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        result.append(tuple(sorted(component)))
    return tuple(result)


def _overlap_graph_connected(atom_sets: Sequence[set[int]]) -> bool:
    pending = {0}
    reached = set()
    while pending:
        index = pending.pop()
        if index in reached:
            continue
        reached.add(index)
        pending.update(
            other
            for other in range(len(atom_sets))
            if other not in reached and atom_sets[index] & atom_sets[other]
        )
    return len(reached) == len(atom_sets)


__all__ = [
    "DEFAULT_POPULATION_HALO_DEPTH",
    "DEFAULT_XB_CM5_TRANSFER_E",
    "ORACLE_ELECTROSTATICS_SCHEMA",
    "ORACLE_FRAGMENT_RECONSTRUCTION_SCHEMA",
    "ORACLE_XB_DESCRIPTOR_SCHEMA",
    "ChargeProjectionAudit",
    "ElectrostaticAtomType",
    "ElectrostaticEquivalenceThresholds",
    "NeutralChargeGroup",
    "OracleElectrostatics",
    "PerceivedHydrogenBond",
    "PerceivedHalogenBond",
    "PopulationAssembly",
    "PopulationFragmentPlan",
    "PopulationFragmentResult",
    "assemble_overlapping_cm5_mayer",
    "plan_overlapping_population_fragments",
    "prepare_oracle_electrostatics",
    "perceive_halogen_bonds",
    "evaluate_halogen_bond_cm5_response",
    "standard_cm5_mayer_request",
]
