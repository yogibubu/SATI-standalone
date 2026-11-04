"""ORACLE-owned electronic-region plan for localized orbital preparation.

This is perception, not a quantum solver. The plan retains ORACLE's graph and
aromaticity decisions; an electronic executor may consume its atom supports
without importing ORACLE's implementation or re-perceiving chemistry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from matrix_chem.topology.pipeline import build_topology_objects
from matrix_chem.topology.discrete_graph import _graph_bridge_edges


@dataclass(frozen=True)
class OrbitalRegion:
    kind: str
    atoms: tuple[int, ...]


@dataclass(frozen=True)
class FragmentCutCandidate:
    """A chemically conservative prospective cut, not a capped fragment."""

    bond: tuple[int, int]
    pauling_order: float
    charge_left: float
    charge_right: float
    fragment_charge_left: float
    fragment_charge_right: float


@dataclass(frozen=True)
class QuasiNeutralFragmentProposal:
    """Bounded domain requiring link-atom treatment before fragment SCF."""

    atoms: tuple[int, ...]
    orbitals: int
    estimated_charge: float
    boundary_bonds: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class OrbitalRegionPlan:
    atomic_numbers: tuple[int, ...]
    bonds: tuple[tuple[int, int], ...]
    regions: tuple[OrbitalRegion, ...]
    electronegativity_charges: tuple[float, ...] = ()
    pauling_bond_orders: tuple[tuple[int, int, float], ...] = ()
    cut_candidates: tuple[FragmentCutCandidate, ...] = ()
    provider: str = "ORACLE_GEOMETRY_TOPOLOGY"
    topology_backend: str = "unknown"
    topology_candidate_pairs: int = 0


def perceive_orbital_regions(
    coordinates_angstrom: np.ndarray,
    atomic_numbers: np.ndarray,
) -> OrbitalRegionPlan:
    """Perceive compact AO supports from ORACLE's covalent topology.

    Lone-pair sites are *candidates* inferred from element and coordination;
    no lone-pair electron count is asserted here. The full global SCF remains
    responsible for occupations and charge transfer.
    """
    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    numbers = np.asarray(atomic_numbers, dtype=int)
    if coordinates.shape != (len(numbers), 3) or not np.all(np.isfinite(coordinates)):
        raise ValueError("ORACLE orbital-region geometry is invalid")
    if len(numbers) == 0 or np.any(numbers < 1):
        raise ValueError("ORACLE orbital-region atomic numbers are invalid")
    continuous_graph, graph, _, synthons, aromaticity = build_topology_objects(
        coordinates, numbers
    )
    bonds = tuple(sorted(tuple(sorted((int(a), int(b)))) for a, b in graph.bonds))
    charges = tuple(float(synthons.charge(atom)) for atom in range(len(numbers)))
    orders = tuple((left, right, float(synthons.bond_order(left, right)))
                   for left, right in bonds)
    bridges = _graph_bridge_edges(len(numbers), bonds)
    # One DFS gives the two prospective fragment charges of every graph
    # bridge. This is linear in graph size; no repeated graph traversal per cut.
    parent = [-1] * len(numbers)
    component = [-1] * len(numbers)
    traversal: list[int] = []
    for start in range(len(numbers)):
        if component[start] >= 0:
            continue
        component[start] = start
        stack = [start]
        while stack:
            atom = stack.pop()
            traversal.append(atom)
            for neighbor in sorted(graph.adjacency[atom], reverse=True):
                if component[neighbor] < 0:
                    component[neighbor] = start
                    parent[neighbor] = atom
                    stack.append(neighbor)
    component_charge: dict[int, float] = {}
    subtree_charge = list(charges)
    for atom in traversal:
        key = component[atom]
        component_charge[key] = component_charge.get(key, 0.0) + charges[atom]
    for atom in reversed(traversal):
        if parent[atom] >= 0:
            subtree_charge[parent[atom]] += subtree_charge[atom]

    def prospective_charges(left: int, right: int) -> tuple[float, float]:
        if parent[right] == left:
            right_charge = subtree_charge[right]
            return component_charge[component[left]] - right_charge, right_charge
        if parent[left] == right:
            left_charge = subtree_charge[left]
            return left_charge, component_charge[component[right]] - left_charge
        raise ValueError("ORACLE bridge is not a DFS tree edge")
    regions: list[OrbitalRegion] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()

    def add(kind: str, atoms: tuple[int, ...]) -> None:
        key = (kind, tuple(sorted(set(atoms))))
        if key[1] and key not in seen:
            regions.append(OrbitalRegion(*key))
            seen.add(key)

    aromatic_bonds = {tuple(sorted(pair)) for pair in aromaticity.aromatic_bonds}
    cut_candidates: list[FragmentCutCandidate] = []
    for left, right, order in orders:
        if (
            (left, right) not in bridges
            or numbers[left] == 1 or numbers[right] == 1
            or (left, right) in aromatic_bonds
            or not 0.75 <= order <= 1.15
            or synthons.bond_order_total_pi(left, right) >= 0.10
            or not np.isfinite(charges[left]) or not np.isfinite(charges[right])
            or max(abs(charges[left]), abs(charges[right])) > 0.40
            or abs(charges[left] - charges[right]) > 0.45
        ):
            continue
        left_charge, right_charge = prospective_charges(left, right)
        if max(abs(left_charge), abs(right_charge)) > 0.50:
            continue
        cut_candidates.append(FragmentCutCandidate(
            (left, right), order, charges[left], charges[right],
            left_charge, right_charge,
        ))
    for assignment in aromaticity.assignments:
        add("DELOCALIZED_AROMATIC", tuple(int(atom) for atom in assignment.atoms))
    for left, right in bonds:
        if (left, right) in aromatic_bonds:
            continue
        if synthons.bond_order_pi(left, right) > 0.25:
            add("DELOCALIZED_PI", (left, right))
        else:
            add("BOND_SIGMA", (left, right))
    for atom, number in enumerate(numbers):
        coordination = len(graph.adjacency[atom])
        if (
            (number == 7 and coordination <= 3)
            or (number in (8, 16) and coordination <= 2)
            or (number in (9, 17, 35, 53) and coordination <= 1)
        ):
            add("LONE_PAIR_CANDIDATE", (atom,))
        # Every AO is covered, including nonstandard or uncertain chemistry.
        add("ATOMIC_FALLBACK", (atom,))
    return OrbitalRegionPlan(
        atomic_numbers=tuple(int(z) for z in numbers),
        bonds=bonds,
        regions=tuple(regions),
        electronegativity_charges=charges,
        pauling_bond_orders=orders,
        cut_candidates=tuple(cut_candidates),
        topology_backend=str(continuous_graph.pair_screening_backend),
        topology_candidate_pairs=len(continuous_graph.candidate_pairs),
    )


def assemble_quasi_neutral_fragments(
    plan: OrbitalRegionPlan,
    orbital_atoms: np.ndarray,
    *,
    maximum_fragment_orbitals: int = 64,
    maximum_absolute_charge: float = 0.50,
) -> tuple[QuasiNeutralFragmentProposal, ...]:
    """Join ORACLE subunits into bounded near-neutral *proposals*.

    Only ORACLE-approved single acyclic bonds may remain cut. Each proposal
    records every dangling boundary; no independent fragment SCF is valid
    until those boundaries receive a certified link-atom/embedding treatment.
    """
    orbital_atoms = np.asarray(orbital_atoms, dtype=np.intp)
    natoms = len(plan.atomic_numbers)
    if (
        maximum_fragment_orbitals < 2 or maximum_absolute_charge < 0.0
        or len(plan.electronegativity_charges) != natoms
        or np.any(orbital_atoms < 0) or np.any(orbital_atoms >= natoms)
    ):
        raise ValueError("invalid ORACLE quasi-neutral fragment inputs")
    widths = np.bincount(orbital_atoms, minlength=natoms).astype(int)
    charges = np.asarray(plan.electronegativity_charges, dtype=float)
    if not np.all(np.isfinite(charges)):
        raise ValueError("nonfinite ORACLE fragment charge estimate")
    parent = list(range(natoms))
    group_width = widths.tolist()
    group_charge = charges.tolist()

    def root(atom: int) -> int:
        while parent[atom] != atom:
            parent[atom] = parent[parent[atom]]
            atom = parent[atom]
        return atom

    def join(left: int, right: int) -> None:
        a, b = root(left), root(right)
        if a == b:
            return
        if a > b:
            a, b = b, a
        parent[b] = a
        group_width[a] += group_width[b]
        group_charge[a] += group_charge[b]

    candidate_bonds = {item.bond for item in plan.cut_candidates}
    for left, right in plan.bonds:
        if (left, right) not in candidate_bonds:
            join(left, right)
    if any(group_width[atom] > maximum_fragment_orbitals
           for atom in range(natoms) if root(atom) == atom):
        raise ValueError("an uncuttable ORACLE domain exceeds the AO cap")
    # Stronger prospective cuts are rejoined first. Only the weak remaining
    # bridge bonds become actual proposed boundaries.
    for item in sorted(
        plan.cut_candidates,
        key=lambda candidate: (-candidate.pauling_order, candidate.bond),
    ):
        left, right = item.bond
        a, b = root(left), root(right)
        if a == b:
            continue
        if (
            group_width[a] + group_width[b] <= maximum_fragment_orbitals
            and abs(group_charge[a] + group_charge[b]) <= maximum_absolute_charge
        ):
            join(a, b)
    groups: dict[int, list[int]] = {}
    for atom in range(natoms):
        groups.setdefault(root(atom), []).append(atom)
    if any(abs(group_charge[key]) > maximum_absolute_charge for key in groups):
        raise ValueError("an ORACLE fragment exceeds the near-neutral charge cap")
    boundaries: dict[int, list[tuple[int, int]]] = {key: [] for key in groups}
    for item in plan.cut_candidates:
        left, right = item.bond
        a, b = root(left), root(right)
        if a != b:
            boundaries[a].append(item.bond)
            boundaries[b].append(item.bond)
    return tuple(
        QuasiNeutralFragmentProposal(
            tuple(atoms), int(group_width[key]), float(group_charge[key]),
            tuple(sorted(boundaries[key])),
        )
        for key, atoms in sorted(groups.items())
    )
