"""Finite molecular-symmetry groups for LOCK rovibronic calculations.

The implementation works with characters tabulated for every operation rather
than hard-coding point groups.  It therefore covers non-Abelian molecular
permutation-inversion groups, complex irreducible representations, and double
groups, provided the caller supplies a complete finite character table.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import itertools
import math
from typing import Mapping, Sequence

import numpy as np


def _complex_tuple(values: Sequence[complex | float | Sequence[float]]) -> tuple[complex, ...]:
    answer = []
    for value in values:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            pair = tuple(value)
            if len(pair) != 2:
                raise ValueError("complex characters must be [real, imaginary] pairs")
            answer.append(complex(float(pair[0]), float(pair[1])))
        else:
            answer.append(complex(value))
    return tuple(answer)


def _encode_complex(value: complex) -> float | list[float]:
    return float(value.real) if abs(value.imag) < 1.0e-14 else [float(value.real), float(value.imag)]


@dataclass(frozen=True)
class MolecularSymmetryIrrep:
    """One irreducible representation through its operation characters."""

    label: str
    dimension: int
    characters: tuple[complex, ...]

    def __post_init__(self) -> None:
        if not self.label.strip() or self.dimension < 1:
            raise ValueError("an irrep needs a label and a positive dimension")
        if not all(math.isfinite(value.real) and math.isfinite(value.imag) for value in self.characters):
            raise ValueError("irrep characters must be finite")

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "dimension": self.dimension,
            "characters": [_encode_complex(value) for value in self.characters],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "MolecularSymmetryIrrep":
        return cls(
            label=str(payload["label"]),
            dimension=int(payload["dimension"]),
            characters=_complex_tuple(payload["characters"]),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class EquivalentNuclearSpinSet:
    """Equivalent nuclei and their permutation under every group operation.

    ``permutations[g][i]`` is the image of nucleus ``i`` under operation
    ``g``.  Inversion has no extra action on space-fixed nuclear spins.
    """

    label: str
    two_I: int
    permutations: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not self.label.strip() or self.two_I < 0:
            raise ValueError("a nuclear-spin set needs a label and nonnegative two_I")
        if not self.permutations or not self.permutations[0]:
            raise ValueError("a nuclear-spin set needs nonempty permutations")
        size = len(self.permutations[0])
        wanted = tuple(range(size))
        if any(len(row) != size or tuple(sorted(row)) != wanted for row in self.permutations):
            raise ValueError("each nuclear operation must be a permutation of the same nuclei")

    def characters(self) -> np.ndarray:
        base = self.two_I + 1
        return np.asarray([base ** _cycle_count(row) for row in self.permutations], dtype=float)

    def representation(self) -> tuple[np.ndarray, ...]:
        """Return product-spin permutation matrices for every operation."""

        return _nuclear_spin_permutation_representation(
            self.two_I, self.permutations
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "two_I": self.two_I,
            "permutations": [list(row) for row in self.permutations],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "EquivalentNuclearSpinSet":
        return cls(
            label=str(payload["label"]),
            two_I=int(payload["two_I"]),
            permutations=tuple(
                tuple(int(value) for value in row)
                for row in payload["permutations"]  # type: ignore[arg-type]
            ),
        )


@dataclass(frozen=True)
class FiniteMolecularSymmetryGroup:
    """Validated finite PI group with character algebra and projectors."""

    name: str
    operations: tuple[str, ...]
    multiplication_table: tuple[tuple[int, ...], ...]
    irreps: tuple[MolecularSymmetryIrrep, ...]
    nuclear_spin_sets: tuple[EquivalentNuclearSpinSet, ...] = ()
    identity_irrep: str | None = None

    def __post_init__(self) -> None:
        order = len(self.operations)
        table = np.asarray(self.multiplication_table, dtype=int)
        if not self.name.strip() or order < 1 or len(set(self.operations)) != order:
            raise ValueError("a molecular-symmetry group needs unique operations")
        if table.shape != (order, order) or np.any(table < 0) or np.any(table >= order):
            raise ValueError("the operation multiplication table must be square and closed")
        if not np.array_equal(table[0], np.arange(order)) or not np.array_equal(table[:, 0], np.arange(order)):
            raise ValueError("operation zero must be the identity")
        for left in range(order):
            if np.count_nonzero(table[left] == 0) != 1:
                raise ValueError("every molecular-symmetry operation needs a unique inverse")
            if not np.array_equal(table[table[left], :], table[left, table]):
                raise ValueError("the operation multiplication table is not associative")
        if not self.irreps or len({item.label for item in self.irreps}) != len(self.irreps):
            raise ValueError("provide a nonempty character table with unique irrep labels")
        characters = np.asarray([item.characters for item in self.irreps], dtype=complex)
        if characters.shape != (len(self.irreps), order):
            raise ValueError("provide one character per operation for every irrep")
        dimensions = np.asarray([item.dimension for item in self.irreps])
        if not np.allclose(characters[:, 0], dimensions, atol=1.0e-9):
            raise ValueError("an irrep identity character must equal its dimension")
        gram = characters.conj() @ characters.T / order
        if not np.allclose(gram, np.eye(len(self.irreps)), atol=1.0e-8):
            raise ValueError("irreducible characters are not orthonormal")
        if int(np.dot(dimensions, dimensions)) != order:
            raise ValueError("irrep dimensions do not exhaust the finite group")
        if any(len(item.permutations) != order for item in self.nuclear_spin_sets):
            raise ValueError("map every group operation for each equivalent-nucleus set")
        identity = self.identity_irrep or self.irreps[0].label
        if identity not in self.labels:
            raise ValueError("identity_irrep is absent from the character table")
        identity_row = self.irrep(identity).characters
        if not np.allclose(identity_row, np.ones(order), atol=1.0e-9):
            raise ValueError("identity_irrep must be totally symmetric")
        object.__setattr__(self, "identity_irrep", identity)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(item.label for item in self.irreps)

    def irrep(self, label: str) -> MolecularSymmetryIrrep:
        try:
            return self.irreps[self.labels.index(label)]
        except ValueError as exc:
            raise ValueError(f"unknown {self.name} irrep: {label}") from exc

    def decompose_characters(self, characters: Sequence[complex]) -> dict[str, int]:
        values = np.asarray(characters, dtype=complex)
        if values.shape != (len(self.operations),):
            raise ValueError("a representation needs one character per group operation")
        answer: dict[str, int] = {}
        for irrep in self.irreps:
            raw = np.vdot(np.asarray(irrep.characters), values) / len(self.operations)
            multiplicity = int(np.rint(raw.real))
            if abs(raw.imag) > 1.0e-8 or abs(raw.real - multiplicity) > 1.0e-8 or multiplicity < 0:
                raise ValueError("characters do not form a nonnegative integral representation")
            if multiplicity:
                answer[irrep.label] = multiplicity
        return answer

    def direct_product(self, *labels: str) -> dict[str, int]:
        characters = np.ones(len(self.operations), dtype=complex)
        for label in labels:
            characters *= np.asarray(self.irrep(label).characters)
        return self.decompose_characters(characters)

    def contains_scalar(self, lower: str, operator: str, upper: str) -> bool:
        """Return the exact finite-group transition selection rule."""
        identity = self.identity_irrep or self.irreps[0].label
        return self.direct_product(lower, operator, upper).get(identity, 0) > 0

    def projector(self, representation: Sequence[np.ndarray], irrep: str) -> np.ndarray:
        """Build the character projector in an arbitrary operation representation."""
        matrices = tuple(np.asarray(value, dtype=complex) for value in representation)
        if len(matrices) != len(self.operations) or not matrices:
            raise ValueError("provide one representation matrix per group operation")
        shape = matrices[0].shape
        if len(shape) != 2 or shape[0] != shape[1] or any(matrix.shape != shape for matrix in matrices):
            raise ValueError("group representation matrices must be square and equally sized")
        item = self.irrep(irrep)
        result = sum(
            character.conjugate() * matrix
            for character, matrix in zip(item.characters, matrices, strict=True)
        ) * (item.dimension / len(self.operations))
        result = 0.5 * (result + result.conj().T)
        if not np.allclose(result @ result, result, atol=1.0e-8):
            raise ValueError("operation matrices do not yield an idempotent symmetry projector")
        return result

    def symmetry_basis(
        self, representation: Sequence[np.ndarray], irrep: str, *, tolerance: float = 1.0e-9
    ) -> np.ndarray:
        """Return orthonormal columns spanning one full isotypic component."""
        projector = self.projector(representation, irrep)
        values, vectors = np.linalg.eigh(projector)
        return vectors[:, values > 1.0 - tolerance]

    def nuclear_spin_decomposition(self) -> dict[str, int]:
        characters = np.ones(len(self.operations), dtype=complex)
        for spin_set in self.nuclear_spin_sets:
            characters *= spin_set.characters()
        return self.decompose_characters(characters)

    def nuclear_spin_representation(self) -> tuple[np.ndarray, ...]:
        """Return the explicit representation for all equivalent-spin sets.

        Separate sets are combined by Kronecker products.  A one-member set
        therefore represents an inequivalent spin without special handling.
        """

        matrices = [np.ones((1, 1), dtype=complex) for _ in self.operations]
        for spin_set in self.nuclear_spin_sets:
            for operation, local in enumerate(spin_set.representation()):
                matrices[operation] = np.kron(matrices[operation], local)
        for matrix in matrices:
            matrix.setflags(write=False)
        return tuple(matrices)

    def nuclear_spin_symmetry_basis(self, irrep: str) -> np.ndarray:
        """Return explicit orthonormal spin functions for one group irrep."""

        return self.symmetry_basis(self.nuclear_spin_representation(), irrep)

    def nuclear_spin_statistical_weight(self, rovibronic_irrep: str, required_total: str) -> int:
        """Return spin functions attached to a complete rovibronic irrep multiplet.

        The returned convention includes the dimension of the rovibronic irrep;
        dividing by that dimension gives the weight per symmetry component.
        """
        self.irrep(required_total)
        decomposition = self.nuclear_spin_decomposition()
        compatible = sum(
            multiplicity * self.direct_product(rovibronic_irrep, spin_irrep).get(required_total, 0)
            for spin_irrep, multiplicity in decomposition.items()
        )
        return self.irrep(rovibronic_irrep).dimension * compatible

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "matrix.lock.molecular_symmetry_group.v1",
            "name": self.name,
            "operations": list(self.operations),
            "multiplication_table": [list(row) for row in self.multiplication_table],
            "irreps": [item.to_dict() for item in self.irreps],
            "nuclear_spin_sets": [item.to_dict() for item in self.nuclear_spin_sets],
            "identity_irrep": self.identity_irrep or self.irreps[0].label,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "FiniteMolecularSymmetryGroup":
        if payload.get("schema") != "matrix.lock.molecular_symmetry_group.v1":
            raise ValueError(f"unsupported molecular-symmetry schema: {payload.get('schema')!r}")
        return cls(
            name=str(payload["name"]),
            operations=tuple(str(value) for value in payload["operations"]),  # type: ignore[arg-type]
            multiplication_table=tuple(
                tuple(int(value) for value in row)
                for row in payload["multiplication_table"]  # type: ignore[arg-type]
            ),
            irreps=tuple(
                MolecularSymmetryIrrep.from_dict(item)
                for item in payload["irreps"]  # type: ignore[arg-type]
            ),
            nuclear_spin_sets=tuple(
                EquivalentNuclearSpinSet.from_dict(item)
                for item in payload.get("nuclear_spin_sets", ())  # type: ignore[arg-type]
            ),
            identity_irrep=str(payload.get("identity_irrep")) if payload.get("identity_irrep") else None,
        )


@lru_cache(maxsize=128)
def _nuclear_spin_permutation_representation(
    two_I: int, permutations: tuple[tuple[int, ...], ...]
) -> tuple[np.ndarray, ...]:
    projections = tuple(range(-two_I, two_I + 1, 2))
    states = tuple(itertools.product(projections, repeat=len(permutations[0])))
    lookup = {state: index for index, state in enumerate(states)}
    matrices = []
    for permutation in permutations:
        matrix = np.zeros((len(states), len(states)), dtype=complex)
        for column, state in enumerate(states):
            transformed = [0] * len(state)
            for source, target in enumerate(permutation):
                transformed[target] = state[source]
            matrix[lookup[tuple(transformed)], column] = 1.0
        matrix.setflags(write=False)
        matrices.append(matrix)
    return tuple(matrices)


def _cycle_count(permutation: Sequence[int]) -> int:
    visited = [False] * len(permutation)
    cycles = 0
    for start in range(len(permutation)):
        if visited[start]:
            continue
        cycles += 1
        current = start
        while not visited[current]:
            visited[current] = True
            current = permutation[current]
    return cycles


__all__ = [
    "EquivalentNuclearSpinSet",
    "FiniteMolecularSymmetryGroup",
    "MolecularSymmetryIrrep",
]
