"""ORACLE atom typing and conservative GAFF/GAFF2 interoperability."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from matrix_chem.topology.elements import atomic_number


ORACLE_GAFF_TRANSLATION_SCHEMA = "matrix.oracle.gaff_atom_types.v1"


@dataclass(frozen=True)
class GaffAtomTypeTranslation:
    """Auditable translation from ORACLE environments to GAFF labels."""

    labels: tuple[str | None, ...]
    reasons: tuple[str | None, ...]
    schema: str = ORACLE_GAFF_TRANSLATION_SCHEMA

    @property
    def complete(self) -> bool:
        return all(label is not None for label in self.labels)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "complete": self.complete,
            "assigned_count": sum(label is not None for label in self.labels),
            "atom_count": len(self.labels),
            "atoms": [
                {
                    "atom": index,
                    "gaff_type": label,
                    "status": "ASSIGNED" if label is not None else "UNSUPPORTED",
                    "reason": reason,
                }
                for index, (label, reason) in enumerate(
                    zip(self.labels, self.reasons, strict=True), start=1
                )
            ],
        }


def assign_gaff_atom_types(
    atomic_numbers: Sequence[int],
    graph: Any,
    synthons: Any,
    aromaticity: Any,
    *,
    strict: bool = True,
) -> GaffAtomTypeTranslation:
    """Translate perceived ORACLE environments into conservative GAFF labels.

    The translation covers hydrocarbons, ordinary amino-acid functional
    groups and neutral sugars.  It consumes accepted connectivity,
    aromaticity and continuous bond-order components; GAFF labels never feed
    back into ORACLE perception.  With ``strict=False``, unsupported
    environments remain explicit instead of aborting the complete analysis.
    """

    numbers = tuple(int(value) for value in atomic_numbers)
    if len(graph.adjacency) != len(numbers):
        raise ValueError("GAFF typing needs one adjacency row per atom")
    aromatic_atoms = set(int(value) for value in aromaticity.aromatic_atoms)
    labels: list[str | None] = []
    reasons: list[str | None] = []
    for atom, number in enumerate(numbers):
        try:
            label = _gaff_label(atom, number, numbers, graph, synthons, aromatic_atoms)
        except ValueError as exc:
            if strict:
                raise
            label = None
            reasons.append(str(exc))
        else:
            reasons.append(None)
        labels.append(label)
    return GaffAtomTypeTranslation(tuple(labels), tuple(reasons))


def gaff_translation_from_snapshot(
    snapshot: Mapping[str, Any],
    synthon_atoms: Sequence[Mapping[str, Any]],
) -> GaffAtomTypeTranslation:
    """Translate a serialized ORACLE state without reperceiving its geometry."""

    numbers = tuple(_atomic_number(atom["element"]) for atom in synthon_atoms)
    adjacency: list[list[int]] = [[] for _ in numbers]
    for left, right in snapshot.get("bonds", ()):
        i, j = int(left) - 1, int(right) - 1
        adjacency[i].append(j)
        adjacency[j].append(i)
    bond_orders = {
        tuple(int(atom) - 1 for atom in row["atoms"]): float(row["value"])
        for row in snapshot.get("bond_orders", ())
    }
    pi_orders = {
        tuple(int(atom) - 1 for atom in row["atoms"]): float(row["pi"])
        for row in snapshot.get("bond_order_components", ())
    }

    class _SerializedSynthons:
        @staticmethod
        def bond_order(left: int, right: int) -> float:
            return bond_orders.get(tuple(sorted((left, right))), 1.0)

        @staticmethod
        def bond_order_pi(left: int, right: int) -> float:
            return pi_orders.get(tuple(sorted((left, right))), 0.0)

    graph = SimpleNamespace(adjacency=tuple(tuple(sorted(row)) for row in adjacency))
    aromaticity = SimpleNamespace(
        aromatic_atoms=tuple(int(atom) - 1 for atom in snapshot.get("aromatic_atoms", ()))
    )
    return assign_gaff_atom_types(
        numbers,
        graph,
        _SerializedSynthons(),
        aromaticity,
        strict=False,
    )


def _gaff_label(
    atom: int,
    number: int,
    numbers: tuple[int, ...],
    graph: Any,
    synthons: Any,
    aromatic_atoms: set[int],
) -> str:
    neighbors = tuple(int(value) for value in graph.adjacency[atom])
    neighbor_numbers = tuple(numbers[value] for value in neighbors)
    if number == 1:
        if len(neighbors) != 1:
            raise ValueError("hydrogen requires exactly one accepted covalent neighbor")
        parent = neighbor_numbers[0]
        return {7: "hn", 8: "ho", 15: "hp", 16: "hs"}.get(
            parent, "ha" if neighbors[0] in aromatic_atoms else "hc"
        )
    if number == 6:
        if atom in aromatic_atoms:
            return "ca"
        pi = sum(float(synthons.bond_order_pi(atom, other)) for other in neighbors)
        carbonyl = any(
            numbers[other] == 8 and float(synthons.bond_order(atom, other)) >= 1.35
            for other in neighbors
        )
        return "c" if carbonyl else ("c1" if pi >= 1.5 else "c2" if pi >= 0.35 else "c3")
    if number == 7:
        if not 2 <= len(neighbors) <= 4:
            raise ValueError("nitrogen translation covers coordination two to four")
        amide = any(
            numbers[other] == 6
            and any(
                numbers[third] == 8
                and float(synthons.bond_order(other, third)) >= 1.35
                for third in graph.adjacency[other]
                if third != atom
            )
            for other in neighbors
        )
        if amide:
            return "n"
        if atom in aromatic_atoms:
            return "na" if len(neighbors) == 3 else "nb"
        if len(neighbors) == 4:
            return "n4"
        pi = sum(float(synthons.bond_order_pi(atom, other)) for other in neighbors)
        return "n2" if pi >= 0.35 else "n3"
    if number == 8:
        if len(neighbors) == 1 and float(synthons.bond_order(atom, neighbors[0])) >= 1.35:
            return "o"
        if len(neighbors) == 1 or 1 in neighbor_numbers:
            return "oh"
        if len(neighbors) == 2:
            return "os"
        raise ValueError("oxygen translation covers carbonyl, hydroxyl and ether")
    if number == 15:
        if atom in aromatic_atoms:
            return "pb"
        if len(neighbors) == 2:
            return "p2"
        if len(neighbors) == 3:
            multiple = any(
                float(synthons.bond_order(atom, other)) >= 1.35 for other in neighbors
            )
            return "p4" if multiple else "p3"
        if len(neighbors) == 4:
            return "p5"
        raise ValueError("phosphorus translation covers coordination two to four")
    if number == 16:
        if len(neighbors) == 1:
            return "s"
        if len(neighbors) == 2:
            if 1 in neighbor_numbers:
                return "sh"
            multiple = any(
                float(synthons.bond_order(atom, other)) >= 1.35 for other in neighbors
            )
            return "s2" if multiple else "ss"
        if len(neighbors) == 3:
            return "s4"
        if len(neighbors) == 4:
            return "s6"
        raise ValueError("sulfur translation covers coordination one to four")
    raise ValueError(f"GAFF translation is not defined for atomic number {number}")


def _atomic_number(label: Any) -> int:
    value = atomic_number(str(label))
    if value is None or value < 1:
        raise ValueError(f"invalid atomic label in serialized synthon data: {label!r}")
    return int(value)


__all__ = [
    "ORACLE_GAFF_TRANSLATION_SCHEMA",
    "GaffAtomTypeTranslation",
    "assign_gaff_atom_types",
    "gaff_translation_from_snapshot",
]
