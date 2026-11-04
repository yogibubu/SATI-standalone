"""Shared CM5 charge response for hydrogen-bonded MATRIX workflows.

The resident 4 x 5 table is a runtime resource of :mod:`matrix_chem`, which is
already a dependency of exploration, optimization and simulation tools.  This
module deliberately has no dependency on ORACLE, SENTINEL, ZAFF or an
electronic-structure backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
import json
from typing import Literal, Mapping, Sequence

import numpy as np

from .structural_corrections import (
    HydrogenBondContact,
    HydrogenBondPairList,
    prepare_hydrogen_bond_recognition,
)


MATRIX_CM5_HBOND_CHARGE_SCHEMA = "matrix.chem.cm5_hbond_charges.v1"
CM5ReferenceKind = Literal["intrinsic", "environment_polarized"]


@dataclass(frozen=True)
class CM5HydrogenBondChargeRule:
    """One statistically averaged donor--acceptor response rule."""

    donor_class: str
    acceptor_class: str
    donor_polarization_e: float
    acceptor_polarization_e: float
    charge_transfer_e: float
    reference_distance_angstrom: float
    vdw_radius_sum_angstrom: float
    observation_count: int
    maximum_bootstrap_half_width_e: float
    maximum_leave_one_environment_out_shift_e: float


@dataclass(frozen=True)
class CM5HydrogenBondChargeTable:
    """Versioned 4 x 5 CM5 response table shared by MATRIX tools."""

    rules: tuple[CM5HydrogenBondChargeRule, ...]
    population_level: str
    total_observations: int
    source: str

    def rule(
        self,
        donor_class: str,
        acceptor_class: str,
    ) -> CM5HydrogenBondChargeRule | None:
        matches = tuple(
            item
            for item in self.rules
            if item.donor_class == donor_class
            and item.acceptor_class == acceptor_class
        )
        if len(matches) > 1:
            raise ValueError("duplicate donor--acceptor rule in the resident table")
        return matches[0] if matches else None


@dataclass(frozen=True)
class CM5HydrogenBondChargeResult:
    """Charges and a chemically resolved audit for one runtime geometry."""

    charges_e: np.ndarray
    intrinsic_charges_e: np.ndarray
    polarization_delta_e: np.ndarray
    charge_transfer_delta_e: np.ndarray
    contacts: tuple[HydrogenBondContact, ...]
    contact_strengths: tuple[float, ...]
    contact_classes: tuple[tuple[str, str], ...]
    unresolved_contacts: tuple[tuple[int, int, int], ...]
    table_source: str
    schema: str = MATRIX_CM5_HBOND_CHARGE_SCHEMA


@dataclass
class CM5HydrogenBondChargeModel:
    """Reusable charge model for optimization, sampling and simulation.

    ``intrinsic_charges_e`` is geometry independent.  A reusable skinned pair
    list recognizes contacts at each geometry, after which the same resident
    response table supplies polarization and charge transfer.
    """

    atomic_numbers: tuple[int, ...]
    bonded_pairs: tuple[tuple[int, int], ...]
    intrinsic_charges_e: np.ndarray
    pair_list: HydrogenBondPairList
    table: CM5HydrogenBondChargeTable
    bond_orders: Mapping[tuple[int, int], float]
    acceptor_class_overrides: Mapping[int, str]

    def evaluate(
        self,
        coordinates_angstrom: np.ndarray,
    ) -> CM5HydrogenBondChargeResult:
        xyz = np.asarray(coordinates_angstrom, dtype=float)
        if xyz.shape != (len(self.atomic_numbers), 3) or np.any(~np.isfinite(xyz)):
            raise ValueError("runtime coordinates must be a finite (natoms, 3) array")
        contacts = self.pair_list.perceive(xyz)
        polarization = np.zeros(len(self.atomic_numbers))
        transfer = np.zeros(len(self.atomic_numbers))
        strengths: list[float] = []
        classes: list[tuple[str, str]] = []
        resolved: list[HydrogenBondContact] = []
        unresolved: list[tuple[int, int, int]] = []
        adjacency = _adjacency(len(self.atomic_numbers), self.bonded_pairs)

        for contact in contacts:
            donor_class = _donor_class(self.atomic_numbers[contact.donor])
            acceptor_class = self.acceptor_class_overrides.get(
                contact.acceptor,
                _acceptor_class(
                    contact.acceptor,
                    self.atomic_numbers,
                    adjacency,
                    self.bond_orders,
                    xyz,
                ),
            )
            rule = (
                None
                if donor_class is None or acceptor_class is None
                else self.table.rule(donor_class, acceptor_class)
            )
            if rule is None:
                unresolved.append(
                    (contact.donor, contact.hydrogen, contact.acceptor)
                )
                continue

            strength = float(
                np.exp(
                    (
                        rule.reference_distance_angstrom
                        - contact.distance_angstrom
                    )
                    / rule.vdw_radius_sum_angstrom
                )
            )
            donor_increment = strength * rule.donor_polarization_e
            acceptor_increment = strength * rule.acceptor_polarization_e
            transfer_increment = strength * rule.charge_transfer_e

            polarization[contact.donor] += donor_increment
            polarization[contact.hydrogen] -= donor_increment
            polarization[contact.acceptor] -= acceptor_increment
            acceptor_neighbors = tuple(sorted(adjacency[contact.acceptor]))
            if not acceptor_neighbors:
                raise ValueError("a hydrogen-bond acceptor needs a covalent neighbor")
            neighbor_share = acceptor_increment / len(acceptor_neighbors)
            polarization[list(acceptor_neighbors)] += neighbor_share

            transfer[contact.donor] += transfer_increment
            transfer[contact.acceptor] -= transfer_increment
            resolved.append(contact)
            strengths.append(strength)
            classes.append((donor_class, acceptor_class))

        if abs(float(np.sum(polarization))) > 1.0e-12:
            raise ArithmeticError("hydrogen-bond polarization does not conserve charge")
        if abs(float(np.sum(transfer))) > 1.0e-12:
            raise ArithmeticError("hydrogen-bond charge transfer does not conserve charge")
        charges = self.intrinsic_charges_e + polarization + transfer
        return CM5HydrogenBondChargeResult(
            charges_e=charges,
            intrinsic_charges_e=self.intrinsic_charges_e.copy(),
            polarization_delta_e=polarization,
            charge_transfer_delta_e=transfer,
            contacts=tuple(resolved),
            contact_strengths=tuple(strengths),
            contact_classes=tuple(classes),
            unresolved_contacts=tuple(unresolved),
            table_source=self.table.source,
        )


@lru_cache(maxsize=1)
def load_cm5_hydrogen_bond_charge_table() -> CM5HydrogenBondChargeTable:
    """Load the single resident response table distributed by ``matrix-chem``."""

    resource = files("matrix_chem").joinpath("data/hbond_charge_response_l0.json")
    payload = json.loads(resource.read_text(encoding="utf-8"))
    if payload.get("schema") != "matrix.oracle.hbond_training.v2":
        raise ValueError("unsupported resident CM5 hydrogen-bond table")
    aggregation = payload["aggregation_contract"]
    if aggregation.get("parameter_table_shape") != [4, 5]:
        raise ValueError("the resident CM5 table must have shape 4 x 5")

    rules = []
    for record in payload["calibrations"]:
        statistics = record["aggregation"]["statistics"]
        uncertainty_records = (
            statistics["donor_polarization_displaced_charge_e"],
            statistics["acceptor_polarization_displaced_charge_e"],
            statistics["charge_transfer_e"],
        )
        contact = record["contact"]
        rules.append(
            CM5HydrogenBondChargeRule(
                donor_class=str(record["donor_class"]),
                acceptor_class=str(record["acceptor_class"]),
                donor_polarization_e=float(
                    record["donor_polarization_displaced_charge_e"]
                ),
                acceptor_polarization_e=float(
                    record["acceptor_polarization_displaced_charge_e"]
                ),
                charge_transfer_e=float(contact["charge_transfer_e"]),
                reference_distance_angstrom=float(
                    contact["reference_distance_angstrom"]
                ),
                vdw_radius_sum_angstrom=float(
                    contact["vdw_radius_sum_angstrom"]
                ),
                observation_count=int(record["observation_count"]),
                maximum_bootstrap_half_width_e=max(
                    float(item["bootstrap_95_half_width"])
                    for item in uncertainty_records
                ),
                maximum_leave_one_environment_out_shift_e=max(
                    float(
                        item["leave_one_environment_out"][
                            "maximum_absolute_mean_shift"
                        ]
                    )
                    for item in uncertainty_records
                ),
            )
        )
    if len(rules) != 20:
        raise ValueError("the resident CM5 table must contain 20 rules")
    return CM5HydrogenBondChargeTable(
        rules=tuple(rules),
        population_level=str(payload["population_level"]),
        total_observations=int(aggregation["total_observations"]),
        source=str(resource),
    )


def prepare_cm5_hydrogen_bond_charge_model(
    atomic_numbers: Sequence[int],
    reference_coordinates_angstrom: np.ndarray,
    bonded_pairs: Sequence[tuple[int, int]],
    cm5_charges_e: Sequence[float],
    *,
    reference_kind: CM5ReferenceKind = "intrinsic",
    bond_orders: Mapping[tuple[int, int], float] | None = None,
    acceptor_class_overrides: Mapping[int, str] | None = None,
    skin_angstrom: float = 0.5,
    table: CM5HydrogenBondChargeTable | None = None,
) -> CM5HydrogenBondChargeModel:
    """Prepare the common MATRIX CM5/H-bond charge model.

    ``reference_kind="intrinsic"`` treats the supplied CM5 vector as the
    hydrogen-bond-free baseline and adds the response at each geometry.
    ``"environment_polarized"`` removes the response at the reference
    geometry first; reevaluation at that geometry is therefore an exact
    depolarize/reconstruct round trip.
    """

    numbers = tuple(int(value) for value in atomic_numbers)
    xyz = np.asarray(reference_coordinates_angstrom, dtype=float)
    charges = np.asarray(cm5_charges_e, dtype=float).reshape(-1)
    bonds = tuple(
        sorted(
            {
                tuple(sorted((int(left), int(right))))
                for left, right in bonded_pairs
            }
        )
    )
    if xyz.shape != (len(numbers), 3) or np.any(~np.isfinite(xyz)):
        raise ValueError("reference coordinates must be a finite (natoms, 3) array")
    if charges.shape != (len(numbers),) or np.any(~np.isfinite(charges)):
        raise ValueError("CM5 charges must be one finite value per atom")
    if reference_kind not in {"intrinsic", "environment_polarized"}:
        raise ValueError("reference_kind must be intrinsic or environment_polarized")
    normalized_orders = {
        tuple(sorted((int(left), int(right)))): float(value)
        for (left, right), value in dict(bond_orders or {}).items()
    }
    overrides = {
        int(atom): str(value)
        for atom, value in dict(acceptor_class_overrides or {}).items()
    }
    invalid_classes = set(overrides.values()) - {"O", "N", "C=O", "P", "S"}
    if invalid_classes:
        raise ValueError(
            "unsupported acceptor-class override: "
            + ", ".join(sorted(invalid_classes))
        )
    plan = prepare_hydrogen_bond_recognition(numbers, bonds)
    model = CM5HydrogenBondChargeModel(
        atomic_numbers=numbers,
        bonded_pairs=bonds,
        intrinsic_charges_e=charges.copy(),
        pair_list=plan.new_pair_list(skin_angstrom=skin_angstrom),
        table=table or load_cm5_hydrogen_bond_charge_table(),
        bond_orders=normalized_orders,
        acceptor_class_overrides=overrides,
    )
    if reference_kind == "environment_polarized":
        reference = model.evaluate(xyz)
        model.intrinsic_charges_e = (
            charges
            - reference.polarization_delta_e
            - reference.charge_transfer_delta_e
        )
        reconstructed = model.evaluate(xyz)
        if not np.allclose(reconstructed.charges_e, charges, atol=2.0e-12):
            raise ArithmeticError(
                "CM5 hydrogen-bond depolarize/reconstruct round trip failed"
            )
    return model


def tune_cm5_hydrogen_bond_charges(
    atomic_numbers: Sequence[int],
    coordinates_angstrom: np.ndarray,
    bonded_pairs: Sequence[tuple[int, int]],
    cm5_charges_e: Sequence[float],
    *,
    reference_kind: CM5ReferenceKind = "intrinsic",
    bond_orders: Mapping[tuple[int, int], float] | None = None,
    acceptor_class_overrides: Mapping[int, str] | None = None,
) -> CM5HydrogenBondChargeResult:
    """One-shot convenience API used by any MATRIX tool."""

    model = prepare_cm5_hydrogen_bond_charge_model(
        atomic_numbers,
        coordinates_angstrom,
        bonded_pairs,
        cm5_charges_e,
        reference_kind=reference_kind,
        bond_orders=bond_orders,
        acceptor_class_overrides=acceptor_class_overrides,
    )
    return model.evaluate(coordinates_angstrom)


def _adjacency(
    natoms: int,
    bonded_pairs: Sequence[tuple[int, int]],
) -> tuple[frozenset[int], ...]:
    adjacency = [set() for _ in range(natoms)]
    for left, right in bonded_pairs:
        adjacency[left].add(right)
        adjacency[right].add(left)
    return tuple(frozenset(neighbors) for neighbors in adjacency)


def _donor_class(atomic_number: int) -> str | None:
    return {8: "OH", 7: "NH", 15: "PH", 16: "SH"}.get(atomic_number)


def _acceptor_class(
    atom: int,
    atomic_numbers: Sequence[int],
    adjacency: Sequence[frozenset[int]],
    bond_orders: Mapping[tuple[int, int], float],
    coordinates_angstrom: np.ndarray,
) -> str | None:
    number = atomic_numbers[atom]
    if number != 8:
        return {7: "N", 15: "P", 16: "S"}.get(number)
    carbon_neighbors = tuple(
        neighbor for neighbor in adjacency[atom] if atomic_numbers[neighbor] == 6
    )
    for carbon in carbon_neighbors:
        pair = tuple(sorted((atom, carbon)))
        order = bond_orders.get(pair)
        if order is not None and order >= 1.25:
            return "C=O"
        distance = float(
            np.linalg.norm(coordinates_angstrom[atom] - coordinates_angstrom[carbon])
        )
        if order is None and distance <= 1.32:
            return "C=O"
    return "O"


__all__ = [
    "MATRIX_CM5_HBOND_CHARGE_SCHEMA",
    "CM5HydrogenBondChargeModel",
    "CM5HydrogenBondChargeResult",
    "CM5HydrogenBondChargeRule",
    "CM5HydrogenBondChargeTable",
    "load_cm5_hydrogen_bond_charge_table",
    "prepare_cm5_hydrogen_bond_charge_model",
    "tune_cm5_hydrogen_bond_charges",
]
