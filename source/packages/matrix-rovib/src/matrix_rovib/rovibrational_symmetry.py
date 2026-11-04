"""Generic finite-group actions on rovibrational basis states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .molecular_symmetry import FiniteMolecularSymmetryGroup


@dataclass(frozen=True)
class RovibrationalSymmetryBlock:
    """One isotypic Hamiltonian block in a rovibrational basis."""

    irrep: str
    basis: np.ndarray
    hamiltonian: np.ndarray
    energies: np.ndarray
    eigenvectors: np.ndarray


@dataclass(frozen=True)
class RovibrationalSymmetryAction:
    """A validated finite-group representation on rovibrational states.

    ``operations[g]`` is the matrix for operation ``g`` in the same order as
    ``group.operations``.  This interface does not assume a particular
    molecular group, rotor count, or choice of internal/overall coordinates.
    Specialized builders can construct the matrices and pass them here.
    """

    group: FiniteMolecularSymmetryGroup
    operations: tuple[np.ndarray, ...]
    generator_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        matrices = tuple(np.asarray(item, dtype=complex) for item in self.operations)
        order = len(self.group.operations)
        if len(matrices) != order or not matrices:
            raise ValueError("provide one state-action matrix per group operation")
        shape = matrices[0].shape
        if len(shape) != 2 or shape[0] != shape[1] or any(item.shape != shape for item in matrices):
            raise ValueError("state-action matrices must be square and equally sized")
        if any(not np.all(np.isfinite(item)) for item in matrices):
            raise ValueError("state-action matrices must be finite")
        identity = np.eye(shape[0], dtype=complex)
        if not np.allclose(matrices[0], identity, atol=1.0e-9):
            raise ValueError("operation zero must act as the identity")
        for index, matrix in enumerate(matrices):
            if not np.allclose(matrix.conj().T @ matrix, identity, atol=1.0e-9):
                raise ValueError(f"state-action matrix for operation {index} is not unitary")
        table = self.group.multiplication_table
        if any(type(i) is not int or i < 0 or i >= order for i in self.generator_indices):
            raise ValueError("invalid symmetry generator indices")
        if self.generator_indices:
            reached = {0}
            frontier = [0]
            for left in frontier:
                for right in self.generator_indices:
                    product = table[left][right]
                    if product not in reached:
                        reached.add(product)
                        frontier.append(product)
            if len(reached) != order:
                raise ValueError("supplied generators do not generate the entire symmetry group")
        for left, row in enumerate(table):
            for right in self.generator_indices or range(order):
                product = row[right]
                if not np.allclose(matrices[left] @ matrices[right], matrices[product], atol=1.0e-9):
                    raise ValueError(
                        "state-action matrices violate the group multiplication table "
                        f"for operations {left} and {right}"
                    )
        object.__setattr__(self, "operations", matrices)

    @classmethod
    def tensor_product(
        cls,
        group: FiniteMolecularSymmetryGroup,
        *representations: Sequence[np.ndarray],
    ) -> "RovibrationalSymmetryAction":
        """Build a product-basis action from one representation per degree of freedom."""
        if not representations:
            raise ValueError("provide at least one degree-of-freedom representation")
        if any(len(rep) != len(group.operations) for rep in representations):
            raise ValueError("each degree-of-freedom representation must cover the whole group")
        combined = []
        for operation in range(len(group.operations)):
            matrix = np.asarray(representations[0][operation], dtype=complex)
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError("degree-of-freedom representation matrices must be square")
            for representation in representations[1:]:
                factor = np.asarray(representation[operation], dtype=complex)
                if factor.ndim != 2 or factor.shape[0] != factor.shape[1]:
                    raise ValueError("degree-of-freedom representation matrices must be square")
                matrix = np.kron(matrix, factor)
            combined.append(matrix)
        return cls(group, tuple(combined))

    def symmetry_basis(self, irrep: str) -> np.ndarray:
        """Return the full isotypic basis for an irrep."""
        return self.group.symmetry_basis(self.operations, irrep)

    def diagonalize(
        self,
        hamiltonian: np.ndarray,
        *,
        tolerance: float = 1.0e-9,
    ) -> tuple[RovibrationalSymmetryBlock, ...]:
        """Validate invariance and diagonalize a Hermitian matrix by isotypic blocks."""
        matrix = np.asarray(hamiltonian, dtype=complex)
        dimension = self.operations[0].shape[0]
        if matrix.shape != (dimension, dimension) or not np.all(np.isfinite(matrix)):
            raise ValueError(f"Hamiltonian must be finite with shape {(dimension, dimension)}")
        if not np.allclose(matrix, matrix.conj().T, atol=tolerance):
            raise ValueError("rovibrational Hamiltonian must be Hermitian")
        for index in self.generator_indices or range(len(self.operations)):
            operation = self.operations[index]
            residual = np.linalg.norm(matrix @ operation - operation @ matrix)
            if residual > tolerance * max(1.0, float(np.linalg.norm(matrix))):
                raise ValueError(f"Hamiltonian is not invariant under operation {self.group.operations[index]}")
        blocks = []
        for label in self.group.labels:
            basis = self.symmetry_basis(label)
            if basis.shape[1] == 0:
                continue
            projected = basis.conj().T @ matrix @ basis
            projected = 0.5 * (projected + projected.conj().T)
            energies, coefficients = np.linalg.eigh(projected)
            blocks.append(
                RovibrationalSymmetryBlock(
                    label, basis, projected, energies.real, basis @ coefficients
                )
            )
        complete = np.column_stack([block.basis for block in blocks])
        if complete.shape != (dimension, dimension) or not np.allclose(
            complete.conj().T @ complete, np.eye(dimension), atol=10.0 * tolerance
        ):
            raise ValueError("isotypic projectors do not span the rovibrational basis")
        return tuple(blocks)


@dataclass(frozen=True)
class LAMRotationSymmetry:
    """Molecular-group actions for the production contracted LAM--rotation solver.

    ``primitive_lam_actions[g]`` acts on the flattened DVR grid and
    ``rotational_actions[J][g]`` acts on the solver's body-fixed ``|J,K>``
    basis.  The solver contracts the former with its computed LAM eigenvectors
    and forms the tensor-product rovibrational action in the exact basis order
    used by its Hamiltonian.
    """

    group: FiniteMolecularSymmetryGroup
    primitive_lam_actions: tuple[np.ndarray, ...]
    rotational_actions: Mapping[int, Sequence[np.ndarray]]

    def __post_init__(self) -> None:
        count = len(self.group.operations)
        matrices = tuple(np.asarray(item, dtype=complex) for item in self.primitive_lam_actions)
        if len(matrices) != count or not matrices:
            raise ValueError("provide one primitive LAM action per group operation")
        shape = matrices[0].shape
        if len(shape) != 2 or shape[0] != shape[1] or any(item.shape != shape for item in matrices):
            raise ValueError("primitive LAM actions must be square and equally sized")
        object.__setattr__(self, "primitive_lam_actions", matrices)
        rotations = {}
        for J, representation in self.rotational_actions.items():
            if isinstance(J, bool) or int(J) != J or J < 0:
                raise ValueError("rotational action keys must be nonnegative integer J values")
            values = tuple(np.asarray(item, dtype=complex) for item in representation)
            if len(values) != count or any(item.shape != (2 * int(J) + 1,) * 2 for item in values):
                raise ValueError(f"J={J} needs one rotational action matrix of dimension {2 * int(J) + 1}")
            rotations[int(J)] = values
        object.__setattr__(self, "rotational_actions", rotations)

    def contracted_action(self, lam_vectors: np.ndarray, J: int) -> RovibrationalSymmetryAction:
        """Project primitive DVR actions into the retained LAM basis and combine rotation."""
        vectors = np.asarray(lam_vectors, dtype=complex)
        primitive_dimension = self.primitive_lam_actions[0].shape[0]
        if vectors.ndim != 2 or vectors.shape[0] != primitive_dimension:
            raise ValueError("LAM eigenvectors do not match the primitive symmetry-action dimension")
        try:
            rotation = self.rotational_actions[int(J)]
        except KeyError as exc:
            raise ValueError(f"no rotational symmetry action supplied for J={J}") from exc
        projected = tuple(vectors.conj().T @ operation @ vectors for operation in self.primitive_lam_actions)
        lam_identity = np.eye(vectors.shape[1], dtype=complex)
        for index, matrix in enumerate(projected):
            if not np.allclose(matrix.conj().T @ matrix, lam_identity, atol=1.0e-9):
                raise ValueError(
                    "retained LAM contraction is not closed under group operation "
                    f"{self.group.operations[index]}; increase lam_state_count or use a symmetry-adapted contraction"
                )
        return RovibrationalSymmetryAction.tensor_product(self.group, projected, rotation)


__all__ = [
    "LAMRotationSymmetry",
    "RovibrationalSymmetryAction",
    "RovibrationalSymmetryBlock",
]
