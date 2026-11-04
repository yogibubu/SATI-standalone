"""ORACLE contract for topology-preserving Cartesian preoptimization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Sequence

from matrix_chem import MolecularGeometry, build_topology_objects
from matrix_chem.topology.elements import atomic_number

from .atom_typing import assign_gaff_atom_types


ORACLE_PREOPTIMIZATION_TOPOLOGY_SCHEMA = "matrix.oracle.preoptimization_topology.v1"


@dataclass(frozen=True)
class PreoptimizationTopologyContract:
    """Frozen topology that a classical force field is allowed to preserve."""

    bonds: tuple[tuple[int, int], ...]
    gaff_atom_types: tuple[str, ...]
    atom_charges: tuple[float, ...]
    topology_hash: str
    topology_fixed: bool
    gaff_eligible: bool
    reason: str
    schema: str = ORACLE_PREOPTIMIZATION_TOPOLOGY_SCHEMA
    owner: str = "ORACLE"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bonds"] = [list(pair) for pair in self.bonds]
        payload["gaff_atom_types"] = list(self.gaff_atom_types)
        payload["atom_charges"] = list(self.atom_charges)
        return payload


def build_preoptimization_topology_contract(
    geometry: MolecularGeometry,
    *,
    expected_bonds: Sequence[tuple[int, int]] | None = None,
    atom_charges: Sequence[float] = (),
) -> PreoptimizationTopologyContract:
    """Perceive topology once and freeze it only when a trusted graph agrees.

    A force field is eligible only when ORACLE's geometry-first connectivity
    exactly matches the supplied constitutional graph and every atom receives
    a GAFF type.  Otherwise the caller must use a topology-flexible method.
    """

    numbers = tuple(int(atomic_number(atom) or 0) for atom in geometry.atoms)
    if any(number <= 0 for number in numbers):
        raise ValueError("ORACLE topology freezing requires recognized elements")
    continuous, discrete, _rings, synthons, aromaticity = build_topology_objects(
        geometry.coordinates_angstrom,
        numbers,
    )
    del continuous
    bonds = tuple(sorted(tuple(sorted(map(int, pair))) for pair in discrete.bonds))
    reference = (
        None
        if expected_bonds is None
        else tuple(sorted(tuple(sorted(map(int, pair))) for pair in expected_bonds))
    )
    graph_agrees = reference is None or (bool(reference) and bonds == reference)
    translation = assign_gaff_atom_types(
        numbers,
        discrete,
        synthons,
        aromaticity,
        strict=False,
    )
    typed = translation.complete
    supplied_charges = tuple(float(value) for value in atom_charges)
    if supplied_charges and len(supplied_charges) != len(numbers):
        raise ValueError("ORACLE force-field contract needs one charge per atom")
    charges = supplied_charges or tuple(float(synthons.charge(i)) for i in range(len(numbers)))
    topology_fixed = graph_agrees
    gaff_eligible = graph_agrees and typed
    if not graph_agrees:
        reason = "ORACLE_INITIAL_TOPOLOGY_DISAGREES_WITH_CONSTITUTIONAL_GRAPH"
    elif not typed:
        reason = "ORACLE_GAFF_ASSIGNMENT_INCOMPLETE"
    elif reference is None:
        reason = "ORACLE_POST_PREOPTIMIZATION_TOPOLOGY_FROZEN_FOR_SONIC"
    else:
        reason = "ORACLE_INITIAL_TOPOLOGY_FROZEN_FOR_FORCE_FIELD_AND_SONIC"
    identity = json.dumps(
        {"atomic_numbers": numbers, "bonds": bonds},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return PreoptimizationTopologyContract(
        bonds=bonds,
        gaff_atom_types=(tuple(str(label) for label in translation.labels) if typed else ()),
        atom_charges=charges,
        topology_hash=hashlib.sha256(identity).hexdigest(),
        topology_fixed=topology_fixed,
        gaff_eligible=gaff_eligible,
        reason=reason,
    )


__all__ = [
    "ORACLE_PREOPTIMIZATION_TOPOLOGY_SCHEMA",
    "PreoptimizationTopologyContract",
    "build_preoptimization_topology_contract",
]
