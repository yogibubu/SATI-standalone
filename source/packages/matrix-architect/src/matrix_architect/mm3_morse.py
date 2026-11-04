"""MM3 Morse bond priors and continuous ORACLE-synthon interpolation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha256
from importlib.resources import files
from io import StringIO
from math import exp, isfinite, log, sqrt
import re
from typing import Any, Sequence

import numpy as np

from matrix_chem import build_topology_objects


MDYNE_ANGSTROM_TO_HARTREE = 1.0e-18 / 4.3597447222071e-18
MM3_MORSE_REFERENCE = (
    "Shannon et al., J. Phys. Chem. A 123, 2991-2999 (2019), DOI:10.1021/acs.jpca.8b12006"
)
MM3_MORSE_CATALOG_SHA256 = "a120aec5a5d238f8a2253b97cd682ad942c258dea017ab57bce9aa75066b2784"


def mix_heteronuclear_morse_alpha(
    alpha_aa_per_angstrom: float,
    alpha_bb_per_angstrom: float,
    *,
    rule: str = "geometric",
) -> float:
    """Infer ``alpha_AB`` from homonuclear limits when no AB datum exists.

    The geometric rule is the production default because it is symmetric,
    positive and scale-covariant.  The harmonic rule is retained as an
    explicit validation alternative for future BDE benchmarks; an available
    heteronuclear observation must always take precedence over either rule.
    """

    left = float(alpha_aa_per_angstrom)
    right = float(alpha_bb_per_angstrom)
    if not all(isfinite(value) and value > 0.0 for value in (left, right)):
        raise ValueError("homonuclear Morse alpha values must be finite and positive")
    normalized = str(rule).strip().lower()
    if normalized == "geometric":
        return sqrt(left * right)
    if normalized == "harmonic":
        return 2.0 * left * right / (left + right)
    raise ValueError("Morse alpha mixing rule must be 'geometric' or 'harmonic'")


@dataclass(frozen=True)
class MM3MorseBondEntry:
    type_i: int
    label_i: str
    type_j: int
    label_j: str
    k_h_mdyne_per_angstrom: float
    r_h_angstrom: float
    d_e_mdyne_angstrom: float
    k_alpha_mdyne_per_angstrom: float
    alpha_per_angstrom: float

    @property
    def pair(self) -> tuple[int, int]:
        return tuple(sorted((self.type_i, self.type_j)))

    @property
    def k_h_hartree_per_angstrom2(self) -> float:
        return self.k_h_mdyne_per_angstrom * MDYNE_ANGSTROM_TO_HARTREE

    @property
    def d_e_hartree(self) -> float:
        return self.d_e_mdyne_angstrom * MDYNE_ANGSTROM_TO_HARTREE


@dataclass(frozen=True)
class MM3MorseContribution:
    type_i: int
    type_j: int
    label_i: str
    label_j: str
    weight: float
    descriptor_distance: float


@dataclass(frozen=True)
class MM3MorsePrior:
    """One continuously interpolated MM3 Morse prior for a bond."""

    d_e_hartree: float
    r0_angstrom: float
    alpha_per_angstrom: float
    harmonic_curvature_hartree_per_angstrom2: float
    contributions: tuple[MM3MorseContribution, ...]
    source: str = MM3_MORSE_REFERENCE

    def potential(self) -> "MorseBondPotential":
        return MorseBondPotential(
            self.d_e_hartree,
            self.r0_angstrom,
            self.alpha_per_angstrom,
            parameter_source="MM3_MORSE_SYNTHON_INTERPOLATION",
        )

    def match_qm_derivatives(
        self,
        reference_distance_angstrom: float,
        curvature_hartree_per_angstrom2: float,
        *,
        gradient_hartree_per_angstrom: float = 0.0,
    ) -> "MorseBondPotential":
        """Keep interpolated ``D_e`` and match the local QM gradient/Hessian."""
        return morse_with_fixed_depth_from_local_derivatives(
            self.d_e_hartree,
            reference_distance_angstrom,
            gradient_hartree_per_angstrom,
            curvature_hartree_per_angstrom2,
            parameter_source="MM3_D_E_WITH_EXACT_QM_LOCAL_GRADIENT_AND_CURVATURE",
        )


@dataclass(frozen=True)
class MorseBondPotential:
    """A dimensional Morse bond potential in hartree and Angstrom."""

    d_e_hartree: float
    r0_angstrom: float
    alpha_per_angstrom: float
    parameter_source: str

    def __post_init__(self) -> None:
        values = (self.d_e_hartree, self.r0_angstrom, self.alpha_per_angstrom)
        if not all(isfinite(float(value)) and float(value) > 0.0 for value in values):
            raise ValueError("Morse bond parameters must be finite and positive")

    def derivatives(self, distance_angstrom: float) -> tuple[float, float, float]:
        distance = float(distance_angstrom)
        if not isfinite(distance) or distance <= 0.0:
            raise ValueError("Morse bond distance must be finite and positive")
        exponent = exp(-self.alpha_per_angstrom * (distance - self.r0_angstrom))
        energy = self.d_e_hartree * (1.0 - exponent) ** 2
        gradient = 2.0 * self.d_e_hartree * self.alpha_per_angstrom * exponent * (1.0 - exponent)
        curvature = (
            2.0 * self.d_e_hartree * self.alpha_per_angstrom**2 * exponent * (2.0 * exponent - 1.0)
        )
        return energy, gradient, curvature


@dataclass(frozen=True)
class _MM3TypePrototype:
    number: int
    label: str
    atomic_number: int
    coordination: float
    pi_character: float
    pi_pi_character: float
    charge: float


class MM3MorseCatalog:
    """Frozen 180-entry MM3 Morse table with continuous synthon lookup."""

    def __init__(self, entries: Sequence[MM3MorseBondEntry]):
        self.entries = tuple(entries)
        if len(self.entries) != 180:
            raise ValueError("the MM3 Morse catalog must contain 180 fitted bond pairs")
        if len({entry.pair for entry in self.entries}) != len(self.entries):
            raise ValueError("MM3 Morse type pairs must be unique")
        labels: dict[int, str] = {}
        for entry in self.entries:
            labels.setdefault(entry.type_i, entry.label_i)
            labels.setdefault(entry.type_j, entry.label_j)
        self.prototypes = {
            number: _prototype_from_label(number, label) for number, label in labels.items()
        }

    def exact(self, type_i: int, type_j: int) -> MM3MorseBondEntry:
        pair = tuple(sorted((int(type_i), int(type_j))))
        for entry in self.entries:
            if entry.pair == pair:
                return entry
        raise KeyError(f"MM3 Morse parameters are unavailable for pair {pair}")

    def interpolate_from_synthons(
        self,
        synthons: Any,
        atomic_numbers: Sequence[int],
        atom_i: int,
        atom_j: int,
        reference_distance_angstrom: float,
        *,
        maximum_contributors: int | None = None,
        distance_scale_angstrom: float = 0.20,
    ) -> MM3MorsePrior:
        """Interpolate compatible MM3 pairs from continuous atom descriptors.

        Production interpolation uses every element-compatible prototype.  A
        hard top-k truncation would make parameters jump when two contributors
        exchange rank, so ``maximum_contributors`` is only an explicit
        diagnostic approximation.
        """
        numbers = tuple(int(value) for value in atomic_numbers)
        left = _synthon_descriptor(synthons, numbers, int(atom_i))
        right = _synthon_descriptor(synthons, numbers, int(atom_j))
        reference = float(reference_distance_angstrom)
        if not isfinite(reference) or reference <= 0.0:
            raise ValueError("reference bond distance must be finite and positive")
        scored: list[tuple[float, MM3MorseBondEntry, bool]] = []
        for entry in self.entries:
            prototype_i = self.prototypes[entry.type_i]
            prototype_j = self.prototypes[entry.type_j]
            for swapped in (False, True):
                first, second = (
                    (prototype_j, prototype_i) if swapped else (prototype_i, prototype_j)
                )
                if left[0] != first.atomic_number or right[0] != second.atomic_number:
                    continue
                distance = (
                    _prototype_distance(left, first)
                    + _prototype_distance(right, second)
                    + ((reference - entry.r_h_angstrom) / distance_scale_angstrom) ** 2
                )
                scored.append((distance, entry, swapped))
        if not scored:
            raise ValueError(
                f"MM3 Morse catalog has no pair for elements Z={left[0]} and Z={right[0]}"
            )
        scored.sort(key=lambda item: (item[0], item[1].pair, item[2]))
        selected = (
            scored if maximum_contributors is None else scored[: max(1, int(maximum_contributors))]
        )
        minimum_distance = selected[0][0]
        raw_weights = [exp(-0.5 * (item[0] - minimum_distance)) for item in selected]
        total = sum(raw_weights)
        weights = [value / total for value in raw_weights]
        k_h = sum(
            weight * entry.k_h_hartree_per_angstrom2
            for weight, (_distance, entry, _swapped) in zip(weights, selected, strict=True)
        )
        r_h = sum(
            weight * entry.r_h_angstrom
            for weight, (_distance, entry, _swapped) in zip(weights, selected, strict=True)
        )
        d_e = sum(
            weight * entry.d_e_hartree
            for weight, (_distance, entry, _swapped) in zip(weights, selected, strict=True)
        )
        alpha = sqrt(k_h / (2.0 * d_e))
        contributions = tuple(
            MM3MorseContribution(
                entry.type_i,
                entry.type_j,
                entry.label_i,
                entry.label_j,
                weight,
                distance,
            )
            for weight, (distance, entry, _swapped) in zip(weights, selected, strict=True)
        )
        return MM3MorsePrior(d_e, r_h, alpha, k_h, contributions)

    def interpolate_for_geometry(
        self,
        coordinates_angstrom: np.ndarray,
        atomic_numbers: Sequence[int],
        atom_i: int,
        atom_j: int,
    ) -> MM3MorsePrior:
        """Rerun ORACLE perception and obtain the geometry-current bond prior."""

        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        numbers = tuple(int(value) for value in atomic_numbers)
        if coordinates.shape != (len(numbers), 3):
            raise ValueError("coordinates must have shape (natoms, 3)")
        _continuous, _discrete, _rings, synthons, _aromaticity = build_topology_objects(
            coordinates,
            numbers,
        )
        distance = float(np.linalg.norm(coordinates[int(atom_i)] - coordinates[int(atom_j)]))
        return self.interpolate_from_synthons(
            synthons,
            numbers,
            atom_i,
            atom_j,
            distance,
        )


def load_mm3_morse_catalog() -> MM3MorseCatalog:
    resource = files("matrix_architect").joinpath("data/mm3_morse_bonds.csv")
    raw = resource.read_bytes()
    if sha256(raw).hexdigest() != MM3_MORSE_CATALOG_SHA256:
        raise RuntimeError("ZAFF Morse catalog differs from its frozen fingerprint")
    entries = []
    with StringIO(raw.decode("utf-8"), newline="") as handle:
        for row in csv.DictReader(handle):
            entries.append(
                MM3MorseBondEntry(
                    int(row["type_i"]),
                    row["label_i"],
                    int(row["type_j"]),
                    row["label_j"],
                    float(row["k_h_mdyne_per_angstrom"]),
                    float(row["r_h_angstrom"]),
                    float(row["d_e_mdyne_angstrom"]),
                    float(row["k_alpha_mdyne_per_angstrom"]),
                    float(row["alpha_per_angstrom"]),
                )
            )
    return MM3MorseCatalog(entries)


def morse_with_fixed_depth_from_local_derivatives(
    d_e_hartree: float,
    reference_distance_angstrom: float,
    gradient_hartree_per_angstrom: float,
    curvature_hartree_per_angstrom2: float,
    *,
    parameter_source: str = "FIXED_D_E_EXACT_LOCAL_GRADIENT_AND_CURVATURE",
) -> MorseBondPotential:
    """Fit ``alpha`` and ``r0`` while retaining ``D_e`` exactly."""
    depth = float(d_e_hartree)
    q_ref = float(reference_distance_angstrom)
    gradient = float(gradient_hartree_per_angstrom)
    curvature = float(curvature_hartree_per_angstrom2)
    if not isfinite(depth) or depth <= 0.0:
        raise ValueError("Morse depth must be finite and positive")
    if not isfinite(q_ref) or q_ref <= 0.0:
        raise ValueError("Morse reference distance must be finite and positive")
    if not isfinite(curvature) or curvature <= 0.0 or not isfinite(gradient):
        raise ValueError("local Morse gradient/curvature must be finite with positive curvature")
    if abs(gradient) <= 1.0e-12 * max(1.0, curvature):
        return MorseBondPotential(depth, q_ref, sqrt(curvature / (2.0 * depth)), parameter_source)
    reduced_gradient = gradient / (2.0 * depth)
    reduced_curvature = curvature / (2.0 * depth)
    target = reduced_gradient**2 / reduced_curvature

    def equation(exponent: float) -> float:
        return exponent * (1.0 - exponent) ** 2 / (2.0 * exponent - 1.0)

    if gradient > 0.0:
        lower = 0.5 + 1.0e-12
        upper = 1.0 - 1.0e-12
        for _ in range(160):
            middle = 0.5 * (lower + upper)
            if equation(middle) > target:
                lower = middle
            else:
                upper = middle
    else:
        lower = 1.0 + 1.0e-12
        upper = 2.0
        while equation(upper) < target:
            upper *= 2.0
        for _ in range(160):
            middle = 0.5 * (lower + upper)
            if equation(middle) < target:
                lower = middle
            else:
                upper = middle
    exponent = 0.5 * (lower + upper)
    alpha = reduced_gradient / (exponent * (1.0 - exponent))
    r0 = q_ref + log(exponent) / alpha
    return MorseBondPotential(depth, r0, alpha, parameter_source)


def _synthon_descriptor(
    synthons: Any,
    atomic_numbers: tuple[int, ...],
    atom: int,
) -> tuple[int, float, float, float, float]:
    pi_character = sum(
        float(synthons.bond_order_pi(atom, neighbor))
        for neighbor in synthons.neighbors[atom]
    )
    pi_pi_character = sum(
        float(synthons.bond_order_pi_pi(atom, neighbor))
        for neighbor in synthons.neighbors[atom]
    )
    return (
        atomic_numbers[atom],
        float(synthons.cna(atom)),
        pi_character,
        pi_pi_character,
        float(synthons.charge(atom)),
    )


def _prototype_distance(
    descriptor: tuple[int, float, float, float, float],
    prototype: _MM3TypePrototype,
) -> float:
    _atomic_number, coordination, pi_character, pi_pi_character, charge = descriptor
    return (
        ((coordination - prototype.coordination) / 0.75) ** 2
        + ((pi_character - prototype.pi_character) / 0.60) ** 2
        + ((pi_pi_character - prototype.pi_pi_character) / 0.60) ** 2
        + ((charge - prototype.charge) / 0.40) ** 2
    )


def _prototype_from_label(number: int, label: str) -> _MM3TypePrototype:
    upper = label.upper()
    match = re.match(r"(Cl|Br|Si|Ge|Se|Sn|Te|Pb|[A-Za-z])", label)
    symbol = match.group(1).capitalize() if match else ""
    atomic_number = _ELEMENT_NUMBERS.get(symbol)
    if atomic_number is None:
        raise ValueError(f"cannot infer the element of MM3 type {number}: {label}")
    default_coordination = {
        1: 1.0,
        5: 3.0,
        6: 4.0,
        7: 3.0,
        8: 2.0,
        9: 1.0,
        14: 4.0,
        15: 3.0,
        16: 2.0,
        17: 1.0,
        32: 4.0,
        34: 2.0,
        35: 1.0,
        50: 4.0,
        52: 2.0,
        53: 1.0,
        82: 4.0,
    }.get(atomic_number, 2.0)
    coordination = default_coordination
    if "AMMONIUM" in upper:
        coordination = 4.0
    elif "SP3" in upper:
        coordination = 4.0 if atomic_number == 6 else default_coordination
    elif "SP2" in upper or "TRIGONAL" in upper:
        coordination = 3.0 if atomic_number in {5, 6} else 2.0
    elif "ALKYNE" in upper or " NSP " in f" {upper} ":
        coordination = 2.0 if atomic_number == 6 else 1.0
    elif atomic_number == 8 and ("CARBONYL" in upper or "O=C" in upper):
        coordination = 1.0
    pi_character = 0.0
    pi_pi_character = 0.0
    if "ALKYNE" in upper or "ACETYLENE" in upper or " NSP " in f" {upper} ":
        pi_character = 1.0
        pi_pi_character = 1.0
    elif any(
        token in upper
        for token in (
            "SP2",
            "CARBONYL",
            "BENZENE",
            "FURAN",
            "THIOPHENE",
            "IMINE",
            "AZOXY",
            "NITRO",
            "OXIME",
            "KETENE",
            "PYR",
        )
    ):
        pi_character = 1.0
    charge = 0.0
    if any(token in upper for token in ("AMMONIUM", "SULFONIUM", "IMMINIUM", "PYRIDINIUM")):
        charge = 1.0
    elif "CARBOXYLATE ION" in upper or "AMINE OXIDE OXYGEN" in upper:
        charge = -1.0
    return _MM3TypePrototype(
        number,
        label,
        atomic_number,
        coordination,
        pi_character,
        pi_pi_character,
        charge,
    )


_ELEMENT_NUMBERS = {
    "H": 1,
    "D": 1,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ge": 32,
    "Se": 34,
    "Br": 35,
    "Sn": 50,
    "Te": 52,
    "I": 53,
    "Pb": 82,
}
