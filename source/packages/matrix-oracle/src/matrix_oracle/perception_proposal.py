"""LCB26 perception proposals owned by ORACLE and exposed through TANK.

The proposal is deliberately distinct from a QM population.  LCB26 records
can provide transferable CM5/Mayer priors for a new constitution, but only a
declared QM calculation may promote those priors to an electronic result.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from matrix_chem import MolecularGeometry, build_topology_objects
from matrix_chem.topology.elements import atomic_number

from .initial_structure import (
    _bond_donors,
    _bridge_edges,
    _edge_cycle_sizes,
    _lcb26_donor_catalog,
    _local_atom_types,
    _nearest_atomic_types,
    _ring_classes_for_edges,
    _shortlist_catalog,
)


TANK_PERCEPTION_SCHEMA = "matrix.tank.perception_proposal.v1"
TANK_ELECTRONIC_STATUS = "LCB26_TRANSFERRED_PROVISIONAL_REQUIRES_QM_VALIDATION"


def propose_lcb26_perception(
    geometry: MolecularGeometry,
    *,
    lcb26_root: Path | str,
    molecular_charge: int | None = None,
) -> dict[str, Any]:
    """Return charge and Mayer proposals for one complete molecular geometry.

    The target topology is obtained from ORACLE's continuous perception.  Each
    target atom receives a weighted local CM5 proposal and each covalent bond
    receives a weighted Mayer proposal from compatible LCB26 records.  Formal
    connectivity is never replaced by a transferred fractional order.
    """

    atoms = tuple(str(atom) for atom in geometry.atoms)
    coordinates = np.asarray(geometry.coordinates_angstrom, dtype=float)
    numbers = tuple(int(atomic_number(atom) or 0) for atom in atoms)
    if not atoms or coordinates.shape != (len(atoms), 3):
        raise ValueError("TANK perception needs a complete Cartesian geometry")
    if min(numbers) <= 0:
        raise ValueError("TANK perception contains an unknown element")
    root = Path(lcb26_root).expanduser().resolve()
    catalog = _lcb26_donor_catalog(root)
    continuous, graph, rings, synthons, _aromaticity = build_topology_objects(
        coordinates,
        numbers,
    )
    del continuous, rings
    catalog = _shortlist_catalog(
        catalog,
        numbers,
        target_synthons=synthons,
        limit=max(32, min(128, len(catalog.records))),
    )
    bonds = tuple(tuple(sorted((int(left), int(right)))) for left, right in graph.bonds)
    adjacency = {index: set() for index in range(len(atoms))}
    for left, right in bonds:
        adjacency[left].add(right)
        adjacency[right].add(left)
    ring_edges = set(bonds) - _bridge_edges(len(atoms), set(bonds))
    cycle_sizes = _edge_cycle_sizes(adjacency, ring_edges)
    bond_orders = {edge: float(synthons.bond_order(*edge)) for edge in bonds}
    ring_classes = _ring_classes_for_edges(ring_edges, bond_orders)
    atom_types = _local_atom_types(atoms, adjacency, cycle_sizes)
    atom_ring_classes = {
        atom: tuple(
            sorted(
                {
                    ring_classes[edge]
                    for edge in ring_edges
                    if atom in edge and edge in ring_classes
                }
            )
        )
        for atom in range(len(atoms))
    }
    target_descriptors, atom_trace = _nearest_atomic_types(
        numbers,
        synthons,
        range(len(atoms)),
        catalog=catalog,
        target_atom_types=atom_types,
        target_atom_ring_classes=atom_ring_classes,
    )
    raw_charges = np.asarray(
        [float(target_descriptors[index][0]) for index in range(len(atoms))],
        dtype=float,
    )
    target_charge = int(
        geometry.charge if molecular_charge is None and geometry.charge is not None
        else molecular_charge if molecular_charge is not None
        else 0
    )
    raw_residual = float(np.sum(raw_charges) - target_charge)
    confidence = np.asarray(
        [
            max(
                1.0e-6,
                float(item.get("effective_donor_count", 1.0)),
            )
            for item in atom_trace
        ],
        dtype=float,
    )
    inverse_confidence = 1.0 / confidence
    correction = raw_residual * inverse_confidence / float(np.sum(inverse_confidence))
    constrained_charges = raw_charges - correction
    bond_trace: list[dict[str, Any]] = []
    bond_proposals: list[dict[str, Any]] = []
    for internal_index, (left, right) in enumerate(bonds):
        target_order = bond_orders[(left, right)]
        target_distance = float(np.linalg.norm(coordinates[left] - coordinates[right]))
        trace: list[dict[str, Any]] = []
        weighted_length = _bond_donors(
            numbers,
            target_descriptors,
            left,
            right,
            catalog=catalog,
            trace=trace,
            target_index=internal_index,
            target_order=target_order,
            target_distance=target_distance,
            require_ring=(left, right) in ring_edges,
            target_atom_types=atom_types,
            target_cycle_size=cycle_sizes.get((left, right)),
            target_ring_class=ring_classes.get((left, right)),
        )
        item = deepcopy(trace[-1]) if trace else {
            "role": "bond",
            "internal_index": internal_index,
            "status": "NO_LCB26_BOND_DONOR",
            "reliability": {
                "class": "EXTRAPOLATIVE",
                "acceptance_state": "REQUIRES_QM_VALIDATION",
            },
            "donors": [],
        }
        donors = item.get("donors", ())
        total_weight = sum(float(donor.get("weight", 0.0)) for donor in donors)
        if total_weight > 0.0:
            values = np.asarray(
                [float(donor["mayer_bond_orders"][0]) for donor in donors],
                dtype=float,
            )
            weights = np.asarray(
                [float(donor.get("weight", 0.0)) for donor in donors],
                dtype=float,
            )
            proposed_order = float(np.dot(weights, values) / total_weight)
            weighted_std = float(
                np.sqrt(max(0.0, np.dot(weights, (values - proposed_order) ** 2) / total_weight))
            )
        else:
            proposed_order = None
            weighted_std = None
        bond_proposals.append(
            {
                "atoms": [left, right],
                "formal_order": target_order,
                "mayer_order": proposed_order,
                "mayer_weighted_std": weighted_std,
                "geometry_distance_angstrom": target_distance,
                "source": "LCB26_CM5_MAYER_TRANSFER" if proposed_order is not None else "UNRESOLVED",
                "reliability": item.get("reliability", {}),
                "donors": donors,
            }
        )
        bond_trace.append(item)
        del weighted_length
    atom_proposals = []
    for index, item in enumerate(atom_trace):
        atom_proposals.append(
            {
                "atom": index,
                "element": atoms[index],
                "cm5_charge_e": float(raw_charges[index]),
                "charge_conserving_cm5_charge_e": float(constrained_charges[index]),
                "selector_descriptor": item.get("selector_descriptor"),
                "resolved_descriptor": item.get("resolved_descriptor"),
                "reliability": item.get("reliability", {}),
                "donors": item.get("donors", []),
                "status": item.get("status", "UNKNOWN"),
            }
        )
    reliability = _proposal_reliability(atom_trace, bond_trace)
    return {
        "schema": TANK_PERCEPTION_SCHEMA,
        "tool": "TANK",
        "owner": "ORACLE",
        "status": TANK_ELECTRONIC_STATUS,
        "molecular_charge": target_charge,
        "atoms": list(atoms),
        "bond_count": len(bonds),
        "raw_cm5_charge_sum_e": float(np.sum(raw_charges)),
        "charge_residual_before_projection_e": raw_residual,
        "cm5_charges_e": constrained_charges.tolist(),
        "mayer_bond_orders": bond_proposals,
        "atom_proposals": atom_proposals,
        "bond_proposals": bond_proposals,
        "reliability": reliability,
        "provenance": {
            "library": "LCB26",
            "electronic_source": "LCB26 enriched CM5/Mayer records",
            "geometry_source": "input geometry",
            "promotion_rule": "QM_REQUIRED_FOR_ELECTRONIC_RESULT",
            "formal_connectivity_owner": "ORACLE/SWITCH constitution",
        },
    }


def _proposal_reliability(atom_trace: list[dict[str, Any]], bond_trace: list[dict[str, Any]]) -> dict[str, Any]:
    entries = [
        item.get("reliability", {})
        for item in (*atom_trace, *bond_trace)
        if isinstance(item.get("reliability"), dict)
    ]
    counts = {
        name: sum(item.get("class") == name for item in entries)
        for name in ("RELIABLE", "PROVISIONAL", "EXTRAPOLATIVE")
    }
    status = "ACCEPTED" if not counts["EXTRAPOLATIVE"] and not counts["PROVISIONAL"] else "PROVISIONAL"
    if counts["EXTRAPOLATIVE"]:
        status = "REQUIRES_QM_VALIDATION"
    return {
        "status": status,
        "counts": counts,
        "transfer_count": len(entries),
        "requires_qm_validation": status != "ACCEPTED",
    }


__all__ = ["TANK_ELECTRONIC_STATUS", "TANK_PERCEPTION_SCHEMA", "propose_lcb26_perception"]
