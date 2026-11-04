"""Deterministic aromatic normalization for Kekulé and aromatic SMILES."""

from __future__ import annotations

from dataclasses import replace

from .model import SwitchMolecularGraph


SWITCH_AROMATICITY_SCHEMA = "matrix.switch.aromaticity.v1"
_AROMATIC_ELEMENTS = {"B", "C", "N", "O", "P", "S", "As", "Se"}


def perceive_aromaticity(
    graph: SwitchMolecularGraph,
    *,
    maximum_ring_size: int = 8,
) -> SwitchMolecularGraph:
    """Return a graph with simple Hückel-like Kekulé rings normalized.

    This deliberately conservative pass recognizes already aromatic rings and
    ordinary five- and six-membered conjugated rings. ORACLE remains the owner
    of geometry-aware aromaticity for final molecular states.
    """

    cycles = _simple_cycles(graph, maximum_size=maximum_ring_size)
    aromatic_atoms = {atom.index for atom in graph.atoms if atom.aromatic}
    aromatic_edges = {bond.key for bond in graph.bonds if bond.aromatic}
    bond_lookup = {bond.key: bond for bond in graph.bonds}
    for cycle in cycles:
        if len(cycle) not in {5, 6}:
            continue
        if any(graph.atoms[index].symbol not in _AROMATIC_ELEMENTS for index in cycle):
            continue
        edges = [
            tuple(sorted((cycle[index], cycle[(index + 1) % len(cycle)])))
            for index in range(len(cycle))
        ]
        bonds = [bond_lookup[edge] for edge in edges]
        if all(bond.aromatic for bond in bonds):
            aromatic_atoms.update(cycle)
            aromatic_edges.update(edges)
            continue
        double_count = sum(1.75 <= bond.order < 2.5 for bond in bonds)
        conjugated = all(0.8 <= bond.order < 2.5 for bond in bonds)
        globally_conjugated = all(
            any(
                candidate.aromatic or 1.75 <= candidate.order < 2.5
                for candidate in graph.bonds
                if candidate.left == atom_index or candidate.right == atom_index
            )
            for atom_index in cycle
        )
        hetero_lone_pair = any(
            graph.atoms[index].symbol in {"N", "O", "P", "S", "As", "Se"}
            and (
                (graph.atoms[index].hydrogen_count or 0) > 0
                or graph.atoms[index].formal_charge <= 0
            )
            for index in cycle
        )
        if conjugated and (
            (
                len(cycle) == 6
                and (double_count == 3 or globally_conjugated)
            )
            or (len(cycle) == 5 and double_count == 2 and hetero_lone_pair)
        ):
            aromatic_atoms.update(cycle)
            aromatic_edges.update(edges)
    atoms = tuple(
        replace(atom, aromatic=True)
        if atom.index in aromatic_atoms
        else atom
        for atom in graph.atoms
    )
    bonds = tuple(
        replace(bond, order=1.5, aromatic=True)
        if bond.key in aromatic_edges
        else bond
        for bond in graph.bonds
    )
    return SwitchMolecularGraph(
        atoms=atoms,
        bonds=bonds,
        components=graph.components,
        source_smiles=graph.source_smiles,
        total_formal_charge=graph.total_formal_charge,
    )


def _simple_cycles(
    graph: SwitchMolecularGraph,
    *,
    maximum_size: int,
) -> tuple[tuple[int, ...], ...]:
    adjacency = [set() for _ in graph.atoms]
    for bond in graph.bonds:
        adjacency[bond.left].add(bond.right)
        adjacency[bond.right].add(bond.left)
    found: set[tuple[int, ...]] = set()
    for root in range(len(graph.atoms)):
        stack = [(root, (root,), {root})]
        while stack:
            current, path, visited = stack.pop()
            if len(path) > maximum_size:
                continue
            for neighbor in adjacency[current]:
                if neighbor == root and len(path) >= 3:
                    found.add(_canonical_cycle(path))
                elif neighbor > root and neighbor not in visited:
                    stack.append((neighbor, path + (neighbor,), visited | {neighbor}))
    return tuple(sorted(found, key=lambda cycle: (len(cycle), cycle)))


def _canonical_cycle(cycle: tuple[int, ...]) -> tuple[int, ...]:
    variants = []
    for orientation in (cycle, tuple(reversed(cycle))):
        for shift in range(len(cycle)):
            variants.append(orientation[shift:] + orientation[:shift])
    return min(variants)


__all__ = ["SWITCH_AROMATICITY_SCHEMA", "perceive_aromaticity"]
