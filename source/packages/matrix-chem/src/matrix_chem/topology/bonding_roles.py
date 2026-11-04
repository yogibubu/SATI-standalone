"""Periodic-table bonding roles shared by ORACLE perception providers.

The predicates in this module classify element families, never individual
molecules.  Geometry, topology, exposure, and directionality remain separate
requirements at the provider that recognizes a concrete interaction.
"""

from __future__ import annotations

from .metals import is_metal_atomic_number
from .periodic_properties import periodic_atomic_properties


ELECTRON_DEFICIENT_GROUPS = frozenset({13})
ELECTRONEGATIVE_LONE_PAIR_GROUPS = frozenset({15, 16, 17})
CHALCOGEN_GROUP = 16
HALOGEN_GROUP = 17


def periodic_group(atomic_number: int) -> int:
    """Return the IUPAC group used by MATRIX's completed periodic table."""

    return int(periodic_atomic_properties(int(atomic_number)).group)


def is_chalcogen(atomic_number: int) -> bool:
    return periodic_group(atomic_number) == CHALCOGEN_GROUP


def is_chalcogen_linkage(left_atomic_number: int, right_atomic_number: int) -> bool:
    """Recognize every homo- or heteronuclear group-16 covalent linkage."""

    return is_chalcogen(left_atomic_number) and is_chalcogen(right_atomic_number)


def is_halogen(atomic_number: int) -> bool:
    return periodic_group(atomic_number) == HALOGEN_GROUP


def is_electron_deficient_center(atomic_number: int) -> bool:
    """Return whether an element belongs to an electron-deficient main-group family."""

    return periodic_group(atomic_number) in ELECTRON_DEFICIENT_GROUPS


def is_structural_center(atomic_number: int) -> bool:
    """Return whether an atom can own a metal/acceptor coordination domain."""

    number = int(atomic_number)
    return is_metal_atomic_number(number) or is_electron_deficient_center(number)


def is_bridging_ligand(atomic_number: int) -> bool:
    """Return whether an element can bridge structural centers as H or halide."""

    number = int(atomic_number)
    return number == 1 or is_halogen(number)


def is_electronegative_lone_pair_donor(atomic_number: int) -> bool:
    """Return whether an element belongs to a lone-pair donor main-group family."""

    return periodic_group(atomic_number) in ELECTRONEGATIVE_LONE_PAIR_GROUPS


def admits_dative_pair(donor_atomic_number: int, acceptor_atomic_number: int) -> bool:
    """Classify a donor--acceptor element pair before geometric filtering.

    The periodic roles establish applicability.  The electronegativity ordering
    prevents reversing the donor and acceptor labels for unusual metal pairs.
    """

    donor = periodic_atomic_properties(int(donor_atomic_number))
    acceptor = periodic_atomic_properties(int(acceptor_atomic_number))
    return (
        is_electronegative_lone_pair_donor(donor.atomic_number)
        and is_structural_center(acceptor.atomic_number)
        and donor.electronegativity > acceptor.electronegativity
    )


__all__ = [
    "CHALCOGEN_GROUP",
    "ELECTRONEGATIVE_LONE_PAIR_GROUPS",
    "ELECTRON_DEFICIENT_GROUPS",
    "HALOGEN_GROUP",
    "admits_dative_pair",
    "is_bridging_ligand",
    "is_chalcogen",
    "is_chalcogen_linkage",
    "is_electron_deficient_center",
    "is_electronegative_lone_pair_donor",
    "is_halogen",
    "is_structural_center",
    "periodic_group",
]
