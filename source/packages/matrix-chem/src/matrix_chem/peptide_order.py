"""Canonical atom ordering for capped dipeptide-analogue records.

The order is defined by chemical role, never by the order emitted by a
generator or a quantum-chemistry backend.  This is deliberately independent
of RDKit so that SWITCH and imported XYZ records use the same contract.
"""

from __future__ import annotations

from collections import deque
from math import hypot
from typing import Iterable, Sequence


_COVALENT_RADII = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05, "P": 1.07}


def infer_covalent_adjacency(atoms: Sequence[str], coordinates: Sequence[Sequence[float]]) -> tuple[tuple[int, ...], ...]:
    """Infer a conservative constitutional graph from an XYZ geometry."""
    if len(atoms) != len(coordinates):
        raise ValueError("atom and coordinate counts differ")
    adjacency = [set() for _ in atoms]
    for i, left in enumerate(atoms):
        for j in range(i):
            right = atoms[j]
            dx = float(coordinates[i][0]) - float(coordinates[j][0])
            dy = float(coordinates[i][1]) - float(coordinates[j][1])
            dz = float(coordinates[i][2]) - float(coordinates[j][2])
            if hypot(hypot(dx, dy), dz) <= _COVALENT_RADII[left] + _COVALENT_RADII[right] + 0.28:
                adjacency[i].add(j)
                adjacency[j].add(i)
    return tuple(tuple(sorted(neighbours)) for neighbours in adjacency)


def _heavy_neighbours(adjacency, atoms, index):
    return [neighbour for neighbour in adjacency[index] if atoms[neighbour] != "H"]


def _single_or_fail(values: Iterable[int], label: str) -> int:
    values = list(values)
    if len(values) != 1:
        raise ValueError(f"expected one {label}, found {values}")
    return values[0]


def canonical_dipeptide_order(atoms: Sequence[str], coordinates: Sequence[Sequence[float]]) -> tuple[int, ...]:
    """Return one canonical order for ``CH3-C(O)-N-Ca(R)-C(O)-NH2``.

    The fixed sequence is terminal amide N/H, terminal carbonyl, C-alpha/H,
    peptide N/H, acetyl carbonyl, acetyl methyl/H; the side chain follows in
    deterministic breadth-first order.  Thus every residue uses one common
    backbone prefix and only the suffix varies.
    """
    adjacency = infer_covalent_adjacency(atoms, coordinates)
    nitrogens = [i for i, atom in enumerate(atoms) if atom == "N"]

    def is_carbonyl_carbon(index: int) -> bool:
        return atoms[index] == "C" and any(atoms[n] == "O" for n in adjacency[index])

    terminal_candidates = []
    for candidate in nitrogens:
        if len(_heavy_neighbours(adjacency, atoms, candidate)) != 1:
            continue
        for carbonyl in _heavy_neighbours(adjacency, atoms, candidate):
            if not is_carbonyl_carbon(carbonyl):
                continue
            for alpha_candidate in _heavy_neighbours(adjacency, atoms, carbonyl):
                if alpha_candidate == candidate or atoms[alpha_candidate] != "C" or is_carbonyl_carbon(alpha_candidate):
                    continue
                if any(atoms[n] == "N" for n in _heavy_neighbours(adjacency, atoms, alpha_candidate) if n != carbonyl):
                    terminal_candidates.append(candidate)
    terminal_n = _single_or_fail(terminal_candidates, "terminal amide N")
    terminal_carbonyl = _single_or_fail((n for n in _heavy_neighbours(adjacency, atoms, terminal_n) if is_carbonyl_carbon(n)), "terminal carbonyl C")
    alpha = _single_or_fail((n for n in _heavy_neighbours(adjacency, atoms, terminal_carbonyl) if n != terminal_n and atoms[n] == "C" and not is_carbonyl_carbon(n)), "C-alpha")
    peptide_n = _single_or_fail((n for n in _heavy_neighbours(adjacency, atoms, alpha) if n != terminal_carbonyl and atoms[n] == "N"), "peptide N")
    acetyl_carbonyl = _single_or_fail((n for n in _heavy_neighbours(adjacency, atoms, peptide_n) if n != alpha and is_carbonyl_carbon(n)), "N-terminal carbonyl C")
    acetyl_methyl = _single_or_fail((n for n in _heavy_neighbours(adjacency, atoms, acetyl_carbonyl) if n != peptide_n and atoms[n] == "C"), "N-terminal methyl C")

    order = []

    def add_atom(index: int) -> None:
        if index not in order:
            order.append(index)

    def add_heavy_with_hydrogens(index: int) -> None:
        if index not in order:
            order.append(index)
        for neighbour in adjacency[index]:
            if atoms[neighbour] == "H" and neighbour not in order:
                order.append(neighbour)

    def add_carbonyl(index: int) -> None:
        add_atom(index)
        oxygens = [neighbour for neighbour in adjacency[index] if atoms[neighbour] == "O"]
        if len(oxygens) != 1:
            raise ValueError(f"expected one carbonyl oxygen for atom {index}, found {oxygens}")
        add_atom(oxygens[0])

    add_heavy_with_hydrogens(terminal_n)
    add_carbonyl(terminal_carbonyl)
    add_heavy_with_hydrogens(alpha)
    add_heavy_with_hydrogens(peptide_n)
    add_carbonyl(acetyl_carbonyl)
    add_heavy_with_hydrogens(acetyl_methyl)

    queue = deque(sorted(n for n in _heavy_neighbours(adjacency, atoms, alpha) if n not in {terminal_carbonyl, peptide_n} and n not in order))
    seen = set(order)
    while queue:
        heavy = queue.popleft()
        if heavy in seen or atoms[heavy] == "H":
            continue
        seen.add(heavy)
        add_heavy_with_hydrogens(heavy)
        for neighbour in sorted(_heavy_neighbours(adjacency, atoms, heavy)):
            if neighbour not in seen:
                queue.append(neighbour)
    if len(order) != len(atoms):
        raise ValueError(f"unassigned atoms in dipeptide canonicalization: {sorted(set(range(len(atoms))) - set(order))}")
    return tuple(order)


def reorder_population_payload(payload: dict, order: Sequence[int]) -> dict:
    """Reorder atom-wise fields and remap one-based Mayer pairs."""
    if len(order) != len(payload["atoms"]):
        raise ValueError("population/order atom counts differ")
    reordered = dict(payload)
    for key in ("atoms", "coordinates_angstrom", "cm5_charges", "hirshfeld_charges"):
        if key in payload:
            reordered[key] = [payload[key][index] for index in order]
    inverse = {old: new + 1 for new, old in enumerate(order)}
    if "mayer_bond_orders" in payload:
        reordered["mayer_bond_orders"] = [[inverse[int(left) - 1], inverse[int(right) - 1], value] for left, right, value in payload["mayer_bond_orders"]]
    return reordered
