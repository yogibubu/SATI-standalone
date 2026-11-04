"""Bounded enumeration of unspecified tetrahedral and alkene stereochemistry."""

from __future__ import annotations

from dataclasses import replace
from itertools import product

from .canonical import canonical_smiles
from .model import SwitchMolecularGraph


SWITCH_STEREO_ENUMERATION_SCHEMA = "matrix.switch.stereoisomers.v1"


def enumerate_stereoisomers(
    graph: SwitchMolecularGraph,
    *,
    max_isomers: int = 256,
) -> tuple[SwitchMolecularGraph, ...]:
    """Enumerate locally distinguishable unspecified tetrahedral/E--Z choices."""

    if max_isomers < 1:
        raise ValueError("max_isomers must be positive")
    tetrahedral = _unspecified_tetrahedral_centers(graph)
    double_bonds = _unspecified_directional_double_bonds(graph)
    variable_count = len(tetrahedral) + len(double_bonds)
    if variable_count == 0:
        return (graph,)
    if 2**variable_count > max_isomers:
        raise ValueError(
            f"stereoisomer enumeration needs {2**variable_count} states, "
            f"above max_isomers={max_isomers}"
        )
    results = []
    identities: set[str] = set()
    for choices in product((0, 1), repeat=variable_count):
        atoms = list(graph.atoms)
        bonds = list(graph.bonds)
        for atom_index, choice in zip(tetrahedral, choices, strict=False):
            atoms[atom_index] = replace(
                atoms[atom_index],
                chirality="@" if choice == 0 else "@@",
            )
        offset = len(tetrahedral)
        for (double_index, left_single, right_single), choice in zip(
            double_bonds,
            choices[offset:],
            strict=True,
        ):
            _ = double_index
            bonds[left_single] = replace(bonds[left_single], direction="/")
            bonds[right_single] = replace(
                bonds[right_single],
                direction="/" if choice == 0 else "\\",
            )
        isomer = SwitchMolecularGraph(
            atoms=tuple(atoms),
            bonds=tuple(bonds),
            components=graph.components,
            source_smiles=graph.source_smiles,
            total_formal_charge=graph.total_formal_charge,
        )
        identity = canonical_smiles(isomer)
        if identity not in identities:
            identities.add(identity)
            results.append(isomer)
    return tuple(results)


def _unspecified_tetrahedral_centers(graph: SwitchMolecularGraph) -> tuple[int, ...]:
    adjacency = _adjacency(graph)
    centers = []
    for atom in graph.atoms:
        if atom.chirality is not None or atom.symbol not in {"C", "N", "P", "S"}:
            continue
        if atom.symbol == "N" and atom.formal_charge != 1:
            continue
        neighbors = adjacency[atom.index]
        valence = sum(bond.order for _, bond in neighbors)
        implicit_hydrogens = 0
        if atom.hydrogen_count is None and atom.symbol == "C":
            implicit_hydrogens = max(0, int(round(4.0 - valence)))
        elif atom.hydrogen_count:
            implicit_hydrogens = atom.hydrogen_count
        if len(neighbors) + implicit_hydrogens != 4:
            continue
        signatures = [
            _rooted_signature(graph, neighbor, blocked=atom.index, depth=4)
            for neighbor, _ in neighbors
        ]
        signatures.extend(("H",) for _ in range(implicit_hydrogens))
        if len(set(signatures)) == 4:
            centers.append(atom.index)
    return tuple(centers)


def _unspecified_directional_double_bonds(
    graph: SwitchMolecularGraph,
) -> tuple[tuple[int, int, int], ...]:
    adjacency = _adjacency(graph)
    results = []
    for index, bond in enumerate(graph.bonds):
        if not 1.75 <= bond.order < 2.5:
            continue
        left = _alkene_substituents(graph, bond.left, bond.right, adjacency)
        right = _alkene_substituents(graph, bond.right, bond.left, adjacency)
        if (
            len(left) != 2
            or len(right) != 2
            or left[0][1] == left[1][1]
            or right[0][1] == right[1][1]
        ):
            continue
        left_atoms = [atom for atom, _signature in left if atom is not None]
        right_atoms = [atom for atom, _signature in right if atom is not None]
        if not left_atoms or not right_atoms:
            continue
        left_bond = next(
            candidate_index
            for candidate_index, candidate in enumerate(graph.bonds)
            if {candidate.left, candidate.right} == {bond.left, left_atoms[0]}
        )
        right_bond = next(
            candidate_index
            for candidate_index, candidate in enumerate(graph.bonds)
            if {candidate.left, candidate.right} == {bond.right, right_atoms[0]}
        )
        if graph.bonds[left_bond].direction or graph.bonds[right_bond].direction:
            continue
        results.append((index, left_bond, right_bond))
    return tuple(results)


def _alkene_substituents(graph, center, opposite, adjacency):
    substituents = [
        (
            other,
            _rooted_signature(graph, other, blocked=center, depth=4),
        )
        for other, _bond in adjacency[center]
        if other != opposite
    ]
    atom = graph.atoms[center]
    valence = sum(candidate.order for _other, candidate in adjacency[center])
    explicit_hydrogens = atom.hydrogen_count or 0
    implicit_hydrogens = explicit_hydrogens
    if atom.hydrogen_count is None and atom.symbol == "C":
        implicit_hydrogens = max(0, int(round(4.0 - valence)))
    substituents.extend((None, ("H",)) for _ in range(implicit_hydrogens))
    return substituents


def _adjacency(graph: SwitchMolecularGraph):
    adjacency = [[] for _ in graph.atoms]
    for bond in graph.bonds:
        adjacency[bond.left].append((bond.right, bond))
        adjacency[bond.right].append((bond.left, bond))
    return adjacency


def _rooted_signature(
    graph: SwitchMolecularGraph,
    atom: int,
    *,
    blocked: int,
    depth: int,
):
    if depth == 0:
        return (graph.atoms[atom].symbol,)
    branches = []
    for neighbor in graph.neighbors(atom):
        if neighbor == blocked:
            continue
        branches.append(
            _rooted_signature(
                graph,
                neighbor,
                blocked=atom,
                depth=depth - 1,
            )
        )
    return (
        graph.atoms[atom].symbol,
        graph.atoms[atom].formal_charge,
        graph.atoms[atom].aromatic,
        tuple(sorted(branches)),
    )


__all__ = ["SWITCH_STEREO_ENUMERATION_SCHEMA", "enumerate_stereoisomers"]
