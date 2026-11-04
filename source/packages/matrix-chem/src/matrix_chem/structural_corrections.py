"""Continuous ORACLE structural corrections on primitive atom pairs.

The numerical models in this module follow the final working equations in
``CV_radial`` and ``JCP_IS1``.  Distances are expressed in angstrom.  The
core--valence exponential field returns electronic derivatives in atomic
units so that LINK can add them directly to a backend result.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, erf, exp, pi
from typing import Iterable, Sequence

import numpy as np

from .spatial_regions import SpatialRegions

BOHR_TO_ANGSTROM = 0.529177210903
CV_RADIAL_SIGMA_SCALE = 1.3
CV_RADIAL_WEIGHT_THRESHOLD = 0.9

# Radii and valence metadata used by the published CV_radial calibration.
CV_RADIAL_COVALENT_RADII_ANGSTROM = {
    1: 0.31,
    5: 0.84,
    6: 0.76,
    7: 0.71,
    8: 0.66,
    9: 0.57,
    13: 1.21,
    14: 1.11,
    15: 1.07,
    16: 1.05,
    17: 1.02,
}
CV_RADIAL_PERIOD = {
    1: 1,
    5: 2,
    6: 2,
    7: 2,
    8: 2,
    9: 2,
    13: 3,
    14: 3,
    15: 3,
    16: 3,
    17: 3,
}
CV_RADIAL_VALENCE_ELECTRONS = {
    1: 1,
    5: 3,
    6: 4,
    7: 5,
    8: 6,
    9: 7,
    13: 3,
    14: 4,
    15: 5,
    16: 6,
    17: 7,
}

# Class-weighted four-parameter fit from CV_radial.  The result of
# Rcov*(intercept+slope*nval) is in milliangstrom.
CV_RADIAL_RADIUS_AWARE_PERIOD_LINE = {
    2: (-4.1889094413, 0.5212770805),
    3: (-7.2988006002, 0.8169634536),
}
# Joint structural reduction of B and C used for geometry optimization.
# B_XY = -exp(b_X+b_Y) Eh and C_XY is the negative harmonic mean of the
# endpoint period decay constants.
CV_EXPONENTIAL_ATOMIC_LOG_AMPLITUDE = {
    1: -3.810340191539109,
    5: -0.9994686069660973,
    6: -1.3018727451049243,
    7: -1.4848560102102037,
    8: -1.6341504672362506,
    9: -1.9864199158286915,
    13: 0.0045993717286958585,
    14: -0.4757271225882106,
    15: -0.7307785200402656,
    16: -0.935596238766313,
    17: -1.148208616468822,
}
CV_EXPONENTIAL_PERIOD_DECAY_INV_ANGSTROM = {
    1: 0.928290388526001,
    2: 2.564105205901616,
    3: 2.1900014878409646,
}

# JCP_IS1 working PL1 parameterization.  Only OH...O(H) is calibrated.
PL1_HBOND_AMPLITUDE_ANGSTROM = 0.030
PL1_HBOND_REFERENCE_ANGSTROM = 2.15
PL1_HBOND_REFERENCE_UFF_SCALE = 0.81
PL1_HBOND_SIGMA_ANGSTROM = 0.50
PL1_PAIR_SWITCH_SIGMA_ANGSTROM = 0.50
PL1_COVALENT_SWITCH_SIGMA_ANGSTROM = 0.057


@dataclass(frozen=True)
class PairCorrection:
    atoms: tuple[int, int]
    delta_angstrom: float
    layer: str
    weight: float
    donor: int | None = None
    hydrogen: int | None = None
    acceptor: int | None = None


@dataclass(frozen=True)
class HydrogenBondContact:
    donor: int
    hydrogen: int
    acceptor: int
    distance_angstrom: float
    angle_radians: float
    coordination_weight: float
    intramolecular: bool
    pl1_calibrated: bool


def perceive_proton_transfer_bridges(
    atomic_numbers: Sequence[int],
    coordinates_angstrom: np.ndarray,
    bonded_pairs: Iterable[tuple[int, int]],
    *,
    distance_tolerance_angstrom: float = 0.15,
    relative_tolerance: float = 0.10,
    minimum_angle_degrees: float = 140.0,
    minimum_structural_bridge_angle_degrees: float = 60.0,
    covalent_extension_angstrom: float = 0.60,
) -> tuple[tuple[int, int, int], ...]:
    """Identify hydrogens shared by two equivalent heavy-atom neighbours.

    Indices are zero based and records are ``(hydrogen, left, right)``.  This
    is a distinct structural motif rather than an ordinary hydrogen bond.
    Electronegative lone-pair centers use the near-linear proton-transfer
    geometry. Electron-deficient group-13 hydride bridges use periodic-group
    metadata, admit bent X--H--X' domains, and do not require equivalent
    centers or equal raw distances. Candidate neighbours are limited by the
    existing element-specific covalent radii plus one common extension. The
    structural-bridge angle threshold is common to the complete family.
    """

    numbers = tuple(int(value) for value in atomic_numbers)
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    if xyz.shape != (len(numbers), 3) or np.any(~np.isfinite(xyz)):
        raise ValueError("proton-transfer coordinates must have shape (natoms, 3)")
    if distance_tolerance_angstrom <= 0.0 or relative_tolerance <= 0.0:
        raise ValueError("proton-transfer distance tolerances must be positive")
    if not 0.0 < minimum_angle_degrees <= 180.0:
        raise ValueError("proton-transfer angle threshold is invalid")
    if not 0.0 < minimum_structural_bridge_angle_degrees <= 180.0:
        raise ValueError("structural-bridge angle threshold is invalid")
    if covalent_extension_angstrom <= 0.0:
        raise ValueError("shared-hydrogen covalent extension must be positive")
    bonds = {tuple(sorted((int(left), int(right)))) for left, right in bonded_pairs}
    adjacency = [set() for _ in numbers]
    for left, right in bonds:
        if left == right or min(left, right) < 0 or max(left, right) >= len(numbers):
            raise ValueError(f"invalid proton-transfer bond: {(left, right)}")
        adjacency[left].add(right)
        adjacency[right].add(left)
    from .topology.covalent_radii import covalent_radius

    from .topology.bonding_roles import (
        is_electron_deficient_center,
        is_electronegative_lone_pair_donor,
    )

    def bridge_heavy(number: int) -> bool:
        return is_electronegative_lone_pair_donor(
            number
        ) or is_electron_deficient_center(number)

    hydrogen_radius = covalent_radius(1)
    if hydrogen_radius is None:
        raise RuntimeError("ORACLE covalent radius for hydrogen is unavailable")
    eligible_heavy = tuple(
        bridge_heavy(number) and covalent_radius(number) is not None
        for number in numbers
    )
    nearby_heavy: dict[int, list[tuple[float, int]]] | None = None
    if len(numbers) > 128 and any(eligible_heavy):
        maximum_heavy_radius = max(
            float(covalent_radius(numbers[atom]))
            for atom, eligible in enumerate(eligible_heavy) if eligible
        )
        maximum_cutoff = float(
            hydrogen_radius + maximum_heavy_radius + covalent_extension_angstrom
        )
        nearby_heavy = {}
        regions = SpatialRegions.build(xyz, cell_size=maximum_cutoff)
        for left, right in regions.candidate_pairs(maximum_cutoff):
            if numbers[left] == 1 and eligible_heavy[right]:
                hydrogen, atom = left, right
            elif numbers[right] == 1 and eligible_heavy[left]:
                hydrogen, atom = right, left
            else:
                continue
            distance = float(np.linalg.norm(xyz[atom] - xyz[hydrogen]))
            heavy_radius = covalent_radius(numbers[atom])
            cutoff = float(hydrogen_radius + heavy_radius + covalent_extension_angstrom)
            if 1.0e-12 < distance <= cutoff:
                nearby_heavy.setdefault(hydrogen, []).append((distance, atom))
    result: list[tuple[int, int, int]] = []
    for hydrogen, number in enumerate(numbers):
        if number != 1:
            continue
        if any(not bridge_heavy(numbers[atom]) for atom in adjacency[hydrogen]):
            continue
        candidates: list[tuple[float, int]] = []
        if nearby_heavy is not None:
            candidates = nearby_heavy.get(hydrogen, []).copy()
        else:
            for atom, atom_number in enumerate(numbers):
                if atom == hydrogen or not eligible_heavy[atom]:
                    continue
                heavy_radius = covalent_radius(atom_number)
                cutoff = float(hydrogen_radius + heavy_radius + covalent_extension_angstrom)
                distance = float(np.linalg.norm(xyz[atom] - xyz[hydrogen]))
                if distance <= 1.0e-12 or distance > cutoff:
                    continue
                candidates.append((distance, atom))
        candidates.sort()
        if len(candidates) < 2:
            continue
        (left_distance, left), (right_distance, right) = candidates[:2]
        endpoint_numbers = (numbers[left], numbers[right])
        proton_domain = all(
            is_electronegative_lone_pair_donor(number)
            for number in endpoint_numbers
        )
        electron_deficient_domain = all(
            is_electron_deficient_center(number) for number in endpoint_numbers
        )
        if not proton_domain and not electron_deficient_domain:
            continue
        if proton_domain:
            tolerance = max(
                float(distance_tolerance_angstrom),
                float(relative_tolerance) * min(left_distance, right_distance),
            )
            if abs(left_distance - right_distance) > tolerance:
                continue
        left_vector = xyz[left] - xyz[hydrogen]
        right_vector = xyz[right] - xyz[hydrogen]
        denominator = float(np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
        if denominator <= 1.0e-12:
            continue
        angle = float(
            np.degrees(
                np.arccos(
                    np.clip(np.dot(left_vector, right_vector) / denominator, -1.0, 1.0)
                )
            )
        )
        required_angle = (
            minimum_structural_bridge_angle_degrees
            if electron_deficient_domain
            else minimum_angle_degrees
        )
        if angle < required_angle:
            continue
        result.append((hydrogen, min(left, right), max(left, right)))
    return tuple(result)


@dataclass(frozen=True)
class HydrogenBondRecognitionPlan:
    """Immutable H-bond chemistry compiled once for repeated evaluations."""

    atomic_numbers: tuple[int, ...]
    bonded_pairs: tuple[tuple[int, int], ...]
    adjacency: tuple[frozenset[int], ...]
    donor_hydrogens: tuple[tuple[int, int], ...]
    acceptors: tuple[int, ...]
    component_by_atom: tuple[int, ...]
    cutoff_angstrom: float
    selector_threshold: float
    minimum_angle_radians: float

    def new_pair_list(self, *, skin_angstrom: float = 0.5) -> "HydrogenBondPairList":
        return HydrogenBondPairList(self, skin_angstrom=skin_angstrom)


@dataclass
class HydrogenBondPairList:
    """Reusable H...A neighbor list for MC, MD, GA and geometry workflows."""

    plan: HydrogenBondRecognitionPlan
    skin_angstrom: float = 0.5
    reference_coordinates_angstrom: np.ndarray | None = None
    acceptors_by_hydrogen: dict[int, tuple[int, ...]] | None = None
    rebuild_count: int = 0

    def __post_init__(self) -> None:
        if not np.isfinite(self.skin_angstrom) or self.skin_angstrom < 0.0:
            raise ValueError("hydrogen-bond pair-list skin must be finite and nonnegative")

    def perceive(self, coordinates_angstrom: np.ndarray) -> tuple[HydrogenBondContact, ...]:
        xyz = np.asarray(coordinates_angstrom, dtype=float)
        if xyz.shape != (len(self.plan.atomic_numbers), 3):
            raise ValueError(
                "hydrogen-bond runtime coordinates must have shape (natoms, 3)"
            )
        if np.any(~np.isfinite(xyz)):
            raise ValueError("hydrogen-bond runtime coordinates must be finite")
        rebuild = self.reference_coordinates_angstrom is None
        if not rebuild and self.skin_angstrom > 0.0:
            displacement = np.linalg.norm(
                xyz - self.reference_coordinates_angstrom,
                axis=1,
            )
            rebuild = float(np.max(displacement, initial=0.0)) > 0.5 * self.skin_angstrom
        if rebuild or self.acceptors_by_hydrogen is None:
            self.acceptors_by_hydrogen = _hydrogen_bond_candidate_pairs(
                self.plan,
                xyz,
                extra_cutoff=self.skin_angstrom,
            )
            self.reference_coordinates_angstrom = xyz.copy()
            self.rebuild_count += 1
        contacts = _evaluate_hydrogen_bond_candidates(
            self.plan,
            xyz,
            self.acceptors_by_hydrogen,
        )
        central = perceive_proton_transfer_bridges(
            self.plan.atomic_numbers,
            xyz,
            self.plan.bonded_pairs,
        )
        if not central:
            return contacts
        result = list(contacts)
        for hydrogen, left, right in central:
            vector_left = xyz[left] - xyz[hydrogen]
            vector_right = xyz[right] - xyz[hydrogen]
            angle = float(
                np.arccos(
                    np.clip(
                        np.dot(vector_left, vector_right)
                        / (np.linalg.norm(vector_left) * np.linalg.norm(vector_right)),
                        -1.0,
                        1.0,
                    )
                )
            )
            result.append(
                HydrogenBondContact(
                    donor=left,
                    hydrogen=hydrogen,
                    acceptor=right,
                    distance_angstrom=float(np.linalg.norm(vector_right)),
                    angle_radians=angle,
                    coordination_weight=1.0,
                    intramolecular=(
                        self.plan.component_by_atom[left]
                        == self.plan.component_by_atom[right]
                    ),
                    pl1_calibrated=False,
                )
            )
        return tuple(result)


@dataclass(frozen=True)
class CVExponentialPair:
    atoms: tuple[int, int]
    amplitude_hartree: float
    decay_inv_angstrom: float


@dataclass(frozen=True)
class CVExponentialFieldResult:
    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray
    hessian_hartree_per_bohr2: np.ndarray
    pairs: tuple[CVExponentialPair, ...]


def cv_radial_radius_aware_alpha_milliangstrom(atomic_number: int) -> float | None:
    z = int(atomic_number)
    if z == 1:
        return 0.0
    period = CV_RADIAL_PERIOD.get(z)
    valence = CV_RADIAL_VALENCE_ELECTRONS.get(z)
    radius = CV_RADIAL_COVALENT_RADII_ANGSTROM.get(z)
    line = CV_RADIAL_RADIUS_AWARE_PERIOD_LINE.get(period)
    if period is None or valence is None or radius is None or line is None:
        return None
    intercept, slope = line
    return radius * (intercept + slope * valence)


def cv_radial_posterior_amplitude_milliangstrom(
    atomic_number_a: int,
    atomic_number_b: int,
) -> float | None:
    """Return the unique radius-aware CV posterior amplitude."""
    left = cv_radial_radius_aware_alpha_milliangstrom(atomic_number_a)
    right = cv_radial_radius_aware_alpha_milliangstrom(atomic_number_b)
    return None if left is None or right is None else left + right


def cv_radial_bond_delta_angstrom(
    atomic_number_a: int,
    atomic_number_b: int,
    distance_angstrom: float,
    *,
    sigma_scale: float = CV_RADIAL_SIGMA_SCALE,
    weight_threshold: float = CV_RADIAL_WEIGHT_THRESHOLD,
) -> float | None:
    radius_a = CV_RADIAL_COVALENT_RADII_ANGSTROM.get(int(atomic_number_a))
    radius_b = CV_RADIAL_COVALENT_RADII_ANGSTROM.get(int(atomic_number_b))
    if radius_a is None or radius_b is None:
        return None
    if sigma_scale <= 0.0:
        raise ValueError("CV-radial sigma scale must be positive")
    if not 0.0 <= weight_threshold <= 1.0:
        raise ValueError("CV-radial weight threshold must lie between zero and one")
    r0 = radius_a + radius_b
    weight = exp(-((float(distance_angstrom) - r0) / (sigma_scale * r0)) ** 2)
    if weight < weight_threshold:
        return None
    amplitude = cv_radial_posterior_amplitude_milliangstrom(atomic_number_a, atomic_number_b)
    return None if amplitude is None else weight * amplitude / 1000.0


def cv_exponential_pair_parameters(
    atomic_number_a: int, atomic_number_b: int
) -> tuple[float, float] | None:
    za, zb = int(atomic_number_a), int(atomic_number_b)
    log_a = CV_EXPONENTIAL_ATOMIC_LOG_AMPLITUDE.get(za)
    log_b = CV_EXPONENTIAL_ATOMIC_LOG_AMPLITUDE.get(zb)
    period_a, period_b = CV_RADIAL_PERIOD.get(za), CV_RADIAL_PERIOD.get(zb)
    if log_a is None or log_b is None or period_a is None or period_b is None:
        return None
    lambda_a = CV_EXPONENTIAL_PERIOD_DECAY_INV_ANGSTROM[period_a]
    lambda_b = CV_EXPONENTIAL_PERIOD_DECAY_INV_ANGSTROM[period_b]
    amplitude = -exp(log_a + log_b)
    decay = -2.0 * lambda_a * lambda_b / (lambda_a + lambda_b)
    return amplitude, decay


def evaluate_cv_exponential_field(
    atomic_numbers: Sequence[int],
    coordinates_angstrom: np.ndarray,
    bonded_pairs: Iterable[tuple[int, int]],
) -> CVExponentialFieldResult:
    """Evaluate the LINK CV field, including exact radial E/G/H terms."""
    numbers = tuple(int(value) for value in atomic_numbers)
    unsupported = sorted(
        set(numbers).difference(CV_EXPONENTIAL_ATOMIC_LOG_AMPLITUDE)
    )
    if unsupported:
        labels = ", ".join(str(value) for value in unsupported)
        raise ValueError(
            "ORACLE CV exponential field is not parametrized for atomic "
            f"number(s): {labels}"
        )
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    if xyz.shape != (len(numbers), 3):
        raise ValueError("CV field coordinates must have shape (natoms, 3)")
    gradient_a = np.zeros_like(xyz)
    hessian_a = np.zeros((3 * len(numbers), 3 * len(numbers)), dtype=float)
    energy = 0.0
    records: list[CVExponentialPair] = []
    for raw_i, raw_j in sorted({tuple(sorted((int(i), int(j)))) for i, j in bonded_pairs}):
        if raw_i == raw_j or raw_i < 0 or raw_j >= len(numbers):
            raise IndexError(f"invalid CV field atom pair: {(raw_i, raw_j)}")
        parameters = cv_exponential_pair_parameters(numbers[raw_i], numbers[raw_j])
        if parameters is None:
            raise RuntimeError("validated CV elements produced no pair parameters")
        amplitude, decay = parameters
        vector = xyz[raw_i] - xyz[raw_j]
        distance = float(np.linalg.norm(vector))
        if distance <= 1.0e-12:
            raise ValueError("CV field is undefined for coincident atoms")
        unit = vector / distance
        value = amplitude * exp(decay * distance)
        first = decay * value
        second = decay * first
        energy += value
        gradient_a[raw_i] += first * unit
        gradient_a[raw_j] -= first * unit
        radial_hessian = second * np.outer(unit, unit) + (first / distance) * (
            np.eye(3) - np.outer(unit, unit)
        )
        si, sj = slice(3 * raw_i, 3 * raw_i + 3), slice(3 * raw_j, 3 * raw_j + 3)
        hessian_a[si, si] += radial_hessian
        hessian_a[sj, sj] += radial_hessian
        hessian_a[si, sj] -= radial_hessian
        hessian_a[sj, si] -= radial_hessian
        records.append(CVExponentialPair((raw_i, raw_j), amplitude, decay))
    return CVExponentialFieldResult(
        energy_hartree=float(energy),
        gradient_hartree_per_bohr=gradient_a.reshape(-1) * BOHR_TO_ANGSTROM,
        hessian_hartree_per_bohr2=hessian_a * BOHR_TO_ANGSTROM**2,
        pairs=tuple(records),
    )


def coordination_switch(distance: float, reference: float, scale: float, width: float) -> float:
    if width <= 0.0:
        raise ValueError("coordination switching width must be positive")
    return 0.5 * (1.0 - erf((float(distance) - scale * float(reference)) / width))


def hbond_angular_factor(angle_radians: float) -> float:
    return 0.5 * (1.0 + cos(pi - float(angle_radians)))


def pl1_hbond_delta_angstrom(distance_angstrom: float, dha_angle_radians: float) -> float:
    gaussian = exp(
        -((float(distance_angstrom) - PL1_HBOND_REFERENCE_ANGSTROM) / PL1_HBOND_SIGMA_ANGSTROM)
        ** 2
    )
    return PL1_HBOND_AMPLITUDE_ANGSTROM * gaussian * hbond_angular_factor(dha_angle_radians)


def perceive_hydrogen_bonds(
    atomic_numbers: Sequence[int],
    coordinates_angstrom: np.ndarray,
    bonded_pairs: Iterable[tuple[int, int]],
    *,
    selector_threshold: float = 1.0e-3,
    minimum_angle_degrees: float = 110.0,
    donor_hydrogens: Iterable[tuple[int, int]] | None = None,
    acceptor_atoms: Iterable[int] | None = None,
) -> tuple[HydrogenBondContact, ...]:
    """Perceive continuous X--H...A contacts through the shared MATRIX kernel.

    General perception accepts electronegative lone-pair families as donors
    and acceptors. ``pl1_calibrated`` is deliberately narrower: the current
    PL1 parameters are valid only for O--H...O--H contacts.
    """
    plan = prepare_hydrogen_bond_recognition(
        atomic_numbers,
        bonded_pairs,
        selector_threshold=selector_threshold,
        minimum_angle_degrees=minimum_angle_degrees,
        donor_hydrogens=donor_hydrogens,
        acceptor_atoms=acceptor_atoms,
    )
    return plan.new_pair_list(skin_angstrom=0.0).perceive(coordinates_angstrom)


def continuous_hydrogen_bond_coordination(
    plan: HydrogenBondRecognitionPlan,
    coordinates_angstrom: np.ndarray,
    *,
    intermolecular_only: bool = True,
) -> float:
    """Return the all-pair continuous effective number of hydrogen bonds.

    Donor and acceptor identities come from one immutable component topology.
    No distance, angle, or neighbour-list threshold changes the observable.
    """

    from .topology.vdw_radii import uff_vdw_radius

    xyz = np.asarray(coordinates_angstrom, dtype=float)
    if xyz.shape != (len(plan.atomic_numbers), 3) or np.any(~np.isfinite(xyz)):
        raise ValueError("hydrogen-bond coordinates have an invalid shape")
    total = 0.0
    radius_h = uff_vdw_radius(1)
    if radius_h is None:
        return total
    for donor, hydrogen in plan.donor_hydrogens:
        donor_vector = xyz[donor] - xyz[hydrogen]
        donor_norm = float(np.linalg.norm(donor_vector))
        if donor_norm <= 1.0e-12:
            continue
        for acceptor in plan.acceptors:
            if acceptor in {donor, hydrogen}:
                continue
            if (
                intermolecular_only
                and plan.component_by_atom[hydrogen]
                == plan.component_by_atom[acceptor]
            ):
                continue
            radius_a = uff_vdw_radius(plan.atomic_numbers[acceptor])
            if radius_a is None:
                continue
            acceptor_vector = xyz[acceptor] - xyz[hydrogen]
            distance = float(np.linalg.norm(acceptor_vector))
            if distance <= 1.0e-12:
                continue
            angle = float(
                np.arccos(
                    np.clip(
                        np.dot(donor_vector, acceptor_vector)
                        / (donor_norm * distance),
                        -1.0,
                        1.0,
                    )
                )
            )
            total += coordination_switch(
                distance,
                float(radius_h + radius_a),
                1.0,
                PL1_PAIR_SWITCH_SIGMA_ANGSTROM,
            ) * hbond_angular_factor(angle)
    return float(total)


def prepare_hydrogen_bond_recognition(
    atomic_numbers: Sequence[int],
    bonded_pairs: Iterable[tuple[int, int]],
    *,
    selector_threshold: float = 1.0e-3,
    minimum_angle_degrees: float = 110.0,
    donor_hydrogens: Iterable[tuple[int, int]] | None = None,
    acceptor_atoms: Iterable[int] | None = None,
) -> HydrogenBondRecognitionPlan:
    """Compile invariant chemistry for a reusable, ORACLE-free runtime."""

    from .topology.vdw_radii import uff_vdw_radius

    numbers = tuple(int(value) for value in atomic_numbers)
    if not 0.0 <= selector_threshold <= 1.0:
        raise ValueError("hydrogen-bond selector threshold must lie between zero and one")
    minimum_angle = float(minimum_angle_degrees)
    if not np.isfinite(minimum_angle) or not 0.0 <= minimum_angle <= 180.0:
        raise ValueError("minimum hydrogen-bond angle must lie between 0 and 180 degrees")
    bonds = {tuple(sorted((int(i), int(j)))) for i, j in bonded_pairs}
    adjacency = [set() for _ in numbers]
    for left, right in bonds:
        if left == right or min(left, right) < 0 or max(left, right) >= len(numbers):
            raise ValueError(f"invalid hydrogen-bond runtime bond: {(left, right)}")
        adjacency[left].add(right)
        adjacency[right].add(left)
    components = _connected_components(adjacency)
    component_lookup = {
        atom: component for component, atoms in enumerate(components) for atom in atoms
    }
    from .topology.bonding_roles import is_electronegative_lone_pair_donor

    if acceptor_atoms is None:
        acceptors = tuple(
            atom
            for atom, value in enumerate(numbers)
            if is_electronegative_lone_pair_donor(value)
        )
    else:
        acceptors = tuple(sorted({int(atom) for atom in acceptor_atoms}))
        if any(atom < 0 or atom >= len(numbers) for atom in acceptors):
            raise ValueError("hydrogen-bond acceptor site is outside the system")
    if donor_hydrogens is None:
        donors = tuple(
            (next(iter(adjacency[hydrogen])), hydrogen)
            for hydrogen, value in enumerate(numbers)
            if value == 1
            and len(adjacency[hydrogen]) == 1
            and is_electronegative_lone_pair_donor(
                numbers[next(iter(adjacency[hydrogen]))]
            )
        )
    else:
        donors = tuple(
            sorted({(int(donor), int(hydrogen)) for donor, hydrogen in donor_hydrogens})
        )
        for donor, hydrogen in donors:
            if (
                min(donor, hydrogen) < 0
                or max(donor, hydrogen) >= len(numbers)
                or numbers[hydrogen] != 1
                or tuple(sorted((donor, hydrogen))) not in bonds
            ):
                raise ValueError(
                    f"invalid configured hydrogen-bond donor site: {(donor, hydrogen)}"
                )
    radius_h = uff_vdw_radius(1)
    acceptor_numbers = tuple(numbers[atom] for atom in acceptors)
    maximum_acceptor_radius = max(
        (
            float(radius)
            for radius in (uff_vdw_radius(value) for value in acceptor_numbers)
            if radius is not None
        ),
        default=2.1,
    )
    hbond_cutoff = float(radius_h or 1.5) + maximum_acceptor_radius + 0.5
    return HydrogenBondRecognitionPlan(
        atomic_numbers=numbers,
        bonded_pairs=tuple(sorted(bonds)),
        adjacency=tuple(frozenset(items) for items in adjacency),
        donor_hydrogens=donors,
        acceptors=acceptors,
        component_by_atom=tuple(component_lookup[index] for index in range(len(numbers))),
        cutoff_angstrom=hbond_cutoff,
        selector_threshold=float(selector_threshold),
        minimum_angle_radians=float(np.deg2rad(minimum_angle)),
    )


def _hydrogen_bond_candidate_pairs(
    plan: HydrogenBondRecognitionPlan,
    xyz: np.ndarray,
    *,
    extra_cutoff: float,
) -> dict[int, tuple[int, ...]]:
    cutoff = plan.cutoff_angstrom + float(extra_cutoff)
    regions = SpatialRegions.build(xyz, cell_size=cutoff)
    eligible_acceptors = set(plan.acceptors)
    acceptors_by_hydrogen: dict[int, list[int]] = {}
    donor_hydrogens = {hydrogen for _donor, hydrogen in plan.donor_hydrogens}
    for left, right in regions.candidate_pairs(cutoff):
        if left in donor_hydrogens and right in eligible_acceptors:
            acceptors_by_hydrogen.setdefault(left, []).append(right)
        if right in donor_hydrogens and left in eligible_acceptors:
            acceptors_by_hydrogen.setdefault(right, []).append(left)
    return {
        hydrogen: tuple(sorted(acceptors))
        for hydrogen, acceptors in acceptors_by_hydrogen.items()
    }


def _evaluate_hydrogen_bond_candidates(
    plan: HydrogenBondRecognitionPlan,
    xyz: np.ndarray,
    acceptors_by_hydrogen: dict[int, tuple[int, ...]],
) -> tuple[HydrogenBondContact, ...]:
    from .topology.vdw_radii import uff_vdw_radius

    numbers = plan.atomic_numbers
    adjacency = plan.adjacency
    bonds = set(plan.bonded_pairs)
    radius_h = uff_vdw_radius(1)
    contacts: list[HydrogenBondContact] = []
    for donor, hydrogen in plan.donor_hydrogens:
        dh = xyz[donor] - xyz[hydrogen]
        for acceptor in sorted(acceptors_by_hydrogen.get(hydrogen, ())):
            z_acceptor = numbers[acceptor]
            if acceptor in {hydrogen, donor}:
                continue
            if tuple(sorted((hydrogen, acceptor))) in bonds:
                continue
            radius_a = uff_vdw_radius(z_acceptor)
            if radius_h is None or radius_a is None:
                continue
            ha = xyz[acceptor] - xyz[hydrogen]
            distance = float(np.linalg.norm(ha))
            reference = float(radius_h + radius_a)
            weight = coordination_switch(
                distance, reference, 1.0, PL1_PAIR_SWITCH_SIGMA_ANGSTROM
            )
            if weight < plan.selector_threshold:
                continue
            denominator = float(np.linalg.norm(dh) * np.linalg.norm(ha))
            if denominator <= 1.0e-12:
                continue
            angle = float(np.arccos(np.clip(np.dot(dh, ha) / denominator, -1.0, 1.0)))
            if angle < plan.minimum_angle_radians:
                continue
            if hbond_angular_factor(angle) < plan.selector_threshold:
                continue
            acceptor_bears_hydrogen = any(numbers[item] == 1 for item in adjacency[acceptor])
            contacts.append(
                HydrogenBondContact(
                    donor=donor,
                    hydrogen=hydrogen,
                    acceptor=acceptor,
                    distance_angstrom=distance,
                    angle_radians=angle,
                    coordination_weight=weight,
                    intramolecular=(
                        plan.component_by_atom[hydrogen]
                        == plan.component_by_atom[acceptor]
                    ),
                    pl1_calibrated=(
                        numbers[donor] == 8
                        and numbers[acceptor] == 8
                        and acceptor_bears_hydrogen
                    ),
                )
            )
    return tuple(
        sorted(
            contacts,
            key=lambda item: (item.hydrogen, item.distance_angstrom, item.acceptor),
        )
    )


def pseudo_bond_pairs(
    atomic_numbers: Sequence[int],
    coordinates_angstrom: np.ndarray,
    bonded_pairs: Iterable[tuple[int, int]],
    *,
    selector_threshold: float = 1.0e-3,
) -> tuple[tuple[tuple[int, int], str], ...]:
    """Return H-bond pseudo-pairs and an MST joining remaining fragments."""
    numbers = tuple(int(value) for value in atomic_numbers)
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    bonds = {tuple(sorted((int(i), int(j)))) for i, j in bonded_pairs}
    adjacency = [set() for _ in numbers]
    for left, right in bonds:
        adjacency[left].add(right)
        adjacency[right].add(left)
    components = _connected_components(adjacency)
    central_bridges = tuple(
        (pair, kind)
        for hydrogen, left, right in perceive_proton_transfer_bridges(
            numbers, xyz, bonds
        )
        for kind in (
            (
                "INTRAMOLECULAR_BORANE_BRIDGE"
                if 5 in {numbers[left], numbers[right]}
                else "INTRAMOLECULAR_PROTON_BRIDGE"
            ),
        )
        for pair in (tuple(sorted((hydrogen, left))), tuple(sorted((hydrogen, right))))
        if pair not in bonds
    )
    if len(components) <= 1:
        return central_bridges
    component_by_atom = {
        atom: component for component, atoms in enumerate(components) for atom in atoms
    }
    selected: list[tuple[tuple[int, int], str]] = list(central_bridges)
    parent = list(range(len(components)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for (atom_left, atom_right), _kind in central_bridges:
        left_component = component_by_atom[atom_left]
        right_component = component_by_atom[atom_right]
        root_left, root_right = find(left_component), find(right_component)
        if root_left != root_right:
            parent[root_right] = root_left

    for contact in perceive_hydrogen_bonds(
        numbers, xyz, bonds, selector_threshold=selector_threshold
    ):
        if contact.intramolecular:
            continue
        pair = tuple(sorted((contact.hydrogen, contact.acceptor)))
        if pair in {item[0] for item in central_bridges}:
            continue
        selected.append(
            (pair, "INTRAMOLECULAR_HBOND" if contact.intramolecular else "INTERFRAGMENT_HBOND")
        )
        left, right = component_by_atom[pair[0]], component_by_atom[pair[1]]
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    candidates: list[tuple[float, int, int, tuple[int, int]]] = []
    if len(components) > 32:
        # A complete component graph is quadratic and dominated ORACLE for
        # solvent baths.  A geometrically local k-nearest graph is enlarged
        # until it contains a spanning tree; pseudo bonds require a stable
        # sparse connection, not the complete interfragment distance matrix.
        from scipy.spatial import cKDTree

        tree = cKDTree(xyz)
        component_index = np.asarray(
            [component_by_atom[atom] for atom in range(len(numbers))], dtype=int
        )
        candidate_by_pair: dict[tuple[int, int], tuple[float, int, int, tuple[int, int]]] = {}
        neighbor_count = min(len(numbers), 8)
        while True:
            distances, neighbors = tree.query(xyz, k=neighbor_count, workers=1)
            if neighbor_count == 1:
                distances = distances[:, None]; neighbors = neighbors[:, None]
            for atom in range(len(numbers)):
                left_component = int(component_index[atom])
                for distance, other in zip(distances[atom, 1:], neighbors[atom, 1:], strict=True):
                    other = int(other)
                    right_component = int(component_index[other])
                    if left_component == right_component:
                        continue
                    pair = tuple(sorted((atom, other)))
                    record = (
                        float(distance),
                        min(left_component, right_component),
                        max(left_component, right_component),
                        pair,
                    )
                    previous = candidate_by_pair.get(pair)
                    if previous is None or record < previous:
                        candidate_by_pair[pair] = record
            trial_parent = parent.copy()

            def trial_find(index: int) -> int:
                while trial_parent[index] != index:
                    trial_parent[index] = trial_parent[trial_parent[index]]
                    index = trial_parent[index]
                return index

            for _distance, left, right, _pair in sorted(candidate_by_pair.values()):
                root_left, root_right = trial_find(left), trial_find(right)
                if root_left != root_right:
                    trial_parent[root_right] = root_left
            if len({trial_find(index) for index in range(len(components))}) == 1:
                candidates = list(candidate_by_pair.values())
                break
            if neighbor_count >= len(numbers):
                raise RuntimeError("local interfragment graph could not be connected")
            neighbor_count = min(len(numbers), 2 * neighbor_count)
    else:
        for left in range(len(components)):
            for right in range(left + 1, len(components)):
                pair = min(
                    (
                        (float(np.linalg.norm(xyz[i] - xyz[j])), tuple(sorted((i, j))))
                        for i in components[left]
                        for j in components[right]
                    ),
                    key=lambda item: (item[0], item[1]),
                )
                candidates.append((pair[0], left, right, pair[1]))
    for _distance, left, right, pair in sorted(candidates):
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            continue
        parent[root_right] = root_left
        selected.append((pair, "INTERFRAGMENT_CLOSEST"))
    return tuple(dict.fromkeys(selected))


def _connected_components(adjacency: Sequence[set[int]]) -> tuple[tuple[int, ...], ...]:
    remaining = set(range(len(adjacency)))
    result: list[tuple[int, ...]] = []
    while remaining:
        start = min(remaining)
        stack, component = [start], set()
        while stack:
            atom = stack.pop()
            if atom in component:
                continue
            component.add(atom)
            stack.extend(sorted(adjacency[atom] - component, reverse=True))
        remaining -= component
        result.append(tuple(sorted(component)))
    return tuple(result)
