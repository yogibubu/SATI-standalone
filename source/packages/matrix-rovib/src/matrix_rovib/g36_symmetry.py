"""Representations of the two-equivalent-top molecular symmetry group G36.

The construction is molecule-independent: callers provide the nuclear
permutations for the two intrinsic C3 rotations, the operation P that exchanges
the tops, and the operation R that reverses both torsions.  P and R commute;
their rotational actions are respectively b- and c-axis pi rotations.  The
remaining frame operation is Q = P R (an a-axis pi rotation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.linalg import expm

from .molecular_symmetry import (
    EquivalentNuclearSpinSet,
    FiniteMolecularSymmetryGroup,
    MolecularSymmetryIrrep,
)
from .native_rotor import angular_momentum_operators
from .rovibrational_symmetry import RovibrationalSymmetryAction


G36Operation = tuple[int, int, int, int]


@dataclass(frozen=True)
class G36SymmetryBlock:
    """One isotypic Hamiltonian block in the full rovibrational basis."""

    irrep: str
    basis: np.ndarray
    hamiltonian: np.ndarray
    energies: np.ndarray
    eigenvectors: np.ndarray


@dataclass(frozen=True)
class G36TwoTopAction:
    """Finite G36 group and its action on two-top torsion--rotation states.

    Operations are stored in normal form ``a**i b**j P**p R**r`` with
    ``i,j in {0,1,2}`` and ``p,r in {0,1}``.  The torsional basis consists of
    integer Fourier mode pairs ``(m1,m2)`` and must be closed under top exchange
    and simultaneous mode reversal.  The rotational basis is the raw body-fixed
    ``K`` basis used by LOCK's selected Watson representation.
    """

    group: FiniteMolecularSymmetryGroup
    operations: tuple[G36Operation, ...]
    nuclear_permutations: tuple[tuple[int, ...], ...]

    @classmethod
    def from_generators(
        cls,
        *,
        rotor_a_cycle: Sequence[int],
        rotor_b_cycle: Sequence[int],
        top_exchange: Sequence[int],
        simultaneous_torsion_reversal: Sequence[int],
        two_I: int = 1,
        spin_label: str = "G36 permuted nuclei",
    ) -> "G36TwoTopAction":
        """Build and validate G36 from its action on a common nuclear set.

        Each input is a permutation tuple in image notation: ``p[i]`` is the
        image of nucleus ``i``.  The supplied generators must obey the G36
        relations, including exchange of the two C3 cycles by ``P`` and
        inversion of both cycles by ``R``.
        """
        generators = tuple(
            tuple(int(index) for index in values)
            for values in (
                rotor_a_cycle,
                rotor_b_cycle,
                top_exchange,
                simultaneous_torsion_reversal,
            )
        )
        if not generators or len({len(item) for item in generators}) != 1:
            raise ValueError("all G36 nuclear permutations must have the same order")
        order = len(generators[0])
        identity_permutation = tuple(range(order))
        if order < 1 or any(tuple(sorted(item)) != identity_permutation for item in generators):
            raise ValueError("G36 generators must be permutations of the same nuclei")
        a, b, p_swap, r_reverse = generators

        def compose(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
            return tuple(left[right[index]] for index in range(order))

        def power(permutation: tuple[int, ...], exponent: int) -> tuple[int, ...]:
            result = identity_permutation
            for _ in range(exponent):
                result = compose(permutation, result)
            return result

        def inverse(permutation: tuple[int, ...]) -> tuple[int, ...]:
            result = [0] * order
            for source, target in enumerate(permutation):
                result[target] = source
            return tuple(result)

        def require_relation(
            condition: bool, detail: str
        ) -> None:
            if not condition:
                raise ValueError(f"nuclear permutations do not satisfy G36 relation: {detail}")

        require_relation(power(a, 3) == identity_permutation, "a^3 = e")
        require_relation(power(b, 3) == identity_permutation, "b^3 = e")
        require_relation(power(p_swap, 2) == identity_permutation, "P^2 = e")
        require_relation(power(r_reverse, 2) == identity_permutation, "R^2 = e")
        require_relation(compose(a, b) == compose(b, a), "ab = ba")
        require_relation(compose(p_swap, r_reverse) == compose(r_reverse, p_swap), "PR = RP")
        require_relation(
            compose(compose(p_swap, a), inverse(p_swap)) == b
            and compose(compose(p_swap, b), inverse(p_swap)) == a,
            "P exchanges a and b",
        )
        require_relation(
            compose(compose(r_reverse, a), inverse(r_reverse)) == inverse(a)
            and compose(compose(r_reverse, b), inverse(r_reverse)) == inverse(b),
            "R inverts a and b",
        )

        operations: tuple[G36Operation, ...] = tuple(
            (i, j, p_bit, r_bit)
            for i in range(3)
            for j in range(3)
            for p_bit in range(2)
            for r_bit in range(2)
        )
        operation_index = {operation: index for index, operation in enumerate(operations)}
        permutations = tuple(
            compose(
                compose(
                    compose(power(a, i), power(b, j)),
                    power(p_swap, p_bit),
                ),
                power(r_reverse, r_bit),
            )
            for i, j, p_bit, r_bit in operations
        )
        if len(set(permutations)) != 36:
            raise ValueError("G36 generators must generate 36 distinct nuclear permutations")

        def multiply(left: G36Operation, right: G36Operation) -> G36Operation:
            i, j, p_bit, r_bit = left
            k, ell, q_bit, s_bit = right
            if r_bit:
                k, ell = -k, -ell
            if p_bit:
                k, ell = ell, k
            return (
                (i + k) % 3,
                (j + ell) % 3,
                p_bit ^ q_bit,
                r_bit ^ s_bit,
            )

        multiplication = tuple(
            tuple(operation_index[multiply(left, right)] for right in operations)
            for left in operations
        )
        # Character rows are products of the two S3 irreps after the
        # C3 coordinates are changed to c = ab and d = ab^-1.  P reverses d;
        # R reverses both c and d.  The conventional G36 factor ordering used
        # here lists d first and c second, matching the published C3v- x C3v+
        # labels.  This yields the nine irreps without a character table keyed
        # to molecule-specific operation names.
        factor_irreps = (("A1", 1), ("A2", 1), ("E", 2))

        def s3_character(exponent: int, reflection: int, label: str) -> int:
            if label == "A1":
                return 1
            if label == "A2":
                return -1 if reflection else 1
            if reflection:
                return 0
            return 2 if exponent % 3 == 0 else -1

        irreps = tuple(
            MolecularSymmetryIrrep(
                label=f"{left_label}{right_label}",
                dimension=left_dimension * right_dimension,
                characters=tuple(
                    complex(
                        s3_character((2 * (i - j)) % 3, p_bit ^ r_bit, left_label)
                        * s3_character((2 * (i + j)) % 3, r_bit, right_label)
                    )
                    for i, j, p_bit, r_bit in operations
                ),
            )
            for left_label, left_dimension in factor_irreps
            for right_label, right_dimension in factor_irreps
        )
        spin = EquivalentNuclearSpinSet(
            spin_label,
            two_I=two_I,
            permutations=permutations,
        )
        group = FiniteMolecularSymmetryGroup(
            name="G36 = (C3 x C3) semidirect C2v",
            operations=tuple(
                f"a^{i}b^{j}P^{p_bit}R^{r_bit}"
                for i, j, p_bit, r_bit in operations
            ),
            multiplication_table=multiplication,
            irreps=irreps,
            nuclear_spin_sets=(spin,),
            identity_irrep="A1A1",
        )
        # The group table built abstractly must match the supplied physical
        # nuclear permutations operation by operation.
        permutation_index = {item: index for index, item in enumerate(permutations)}
        for left_index, row in enumerate(multiplication):
            for right_index, product_index in enumerate(row):
                composed = compose(permutations[left_index], permutations[right_index])
                if permutation_index.get(composed) != product_index:
                    raise ValueError("nuclear permutations disagree with G36 operation multiplication")
        return cls(group, operations, permutations)

    def torsion_representation(
        self, modes: Sequence[tuple[int, int]]
    ) -> tuple[np.ndarray, ...]:
        """Return the unitary G36 action on a finite Fourier mode-pair basis."""
        mode_pairs = tuple((int(left), int(right)) for left, right in modes)
        if not mode_pairs or len(set(mode_pairs)) != len(mode_pairs):
            raise ValueError("provide a nonempty unique set of torsional Fourier modes")
        mode_index = {mode: index for index, mode in enumerate(mode_pairs)}
        matrices = []
        for i, j, p_bit, r_bit in self.operations:
            transform = np.eye(2, dtype=int)
            if p_bit:
                transform = np.asarray(((0, 1), (1, 0)), dtype=int) @ transform
            if r_bit:
                transform = -transform
            matrix = np.zeros((len(mode_pairs), len(mode_pairs)), dtype=complex)
            translation = (2.0 * np.pi / 3.0) * np.asarray((i, j), dtype=float)
            for source, mode in enumerate(mode_pairs):
                transformed = tuple(int(value) for value in transform @ np.asarray(mode))
                if transformed not in mode_index:
                    raise ValueError(
                        "torsional Fourier basis is not closed under G36; "
                        f"missing mode {transformed}"
                    )
                phase = np.exp(-1j * float(np.dot(transformed, translation)))
                matrix[mode_index[transformed], source] = phase
            matrices.append(matrix)
        return tuple(matrices)

    def rotational_representation(
        self, J: int, representation: str = "Ir"
    ) -> tuple[np.ndarray, ...]:
        """Return body-fixed rotational matrices for the G36 frame operations."""
        if isinstance(J, bool) or int(J) != J or J < 0:
            raise ValueError("J must be a nonnegative integer")
        _Ja, Jb, Jc = angular_momentum_operators(int(J), representation)
        rotations = {
            (0, 0): np.eye(2 * int(J) + 1, dtype=complex),
            (1, 0): expm(-1j * np.pi * Jb),
            (0, 1): expm(-1j * np.pi * Jc),
            (1, 1): expm(-1j * np.pi * Jb) @ expm(-1j * np.pi * Jc),
        }
        return tuple(rotations[(p_bit, r_bit)] for _i, _j, p_bit, r_bit in self.operations)

    def rovibrational_representation(
        self,
        modes: Sequence[tuple[int, int]],
        J: int,
        representation: str = "Ir",
    ) -> tuple[np.ndarray, ...]:
        """Return operation matrices on ``Fourier(torsion1,torsion2) x |J,K>``."""
        torsion = self.torsion_representation(modes)
        rotation = self.rotational_representation(J, representation)
        combined = tuple(
            np.kron(torsion_matrix, rotation_matrix)
            for torsion_matrix, rotation_matrix in zip(torsion, rotation, strict=True)
        )
        self.validate_representation(combined)
        return combined

    def symmetry_basis(
        self,
        modes: Sequence[tuple[int, int]],
        J: int,
        irrep: str,
        representation: str = "Ir",
    ) -> np.ndarray:
        """Return the complete isotypic rovibrational basis for one G36 irrep."""
        return self.rovibrational_action(modes, J, representation).symmetry_basis(irrep)

    def rovibrational_action(
        self,
        modes: Sequence[tuple[int, int]],
        J: int,
        representation: str = "Ir",
    ) -> RovibrationalSymmetryAction:
        """Build the generic finite-group action for this G36 state basis."""
        matrices = self.rovibrational_representation(modes, J, representation)
        return RovibrationalSymmetryAction(self.group, matrices)

    def dvr_lam_rotation_symmetry(
        self,
        grid_sizes: Sequence[int],
        j_values: Sequence[int],
        representation: str = "Ir",
    ):
        """Return the exact G36 action used by the production product-DVR solver.

        The construction is a unitary Fourier-to-grid transformation of the
        validated mode-pair representation.  It therefore avoids a separate
        grid-index convention and works for every pair of equal, uniform grids
        whose size is divisible by three.
        """

        from .rovibrational_symmetry import LAMRotationSymmetry

        sizes = tuple(int(value) for value in grid_sizes)
        if (
            len(sizes) != 2
            or sizes[0] != sizes[1]
            or sizes[0] < 3
            or sizes[0] % 2 == 0
            or sizes[0] % 3
        ):
            raise ValueError("G36 DVR requires two equal odd grid sizes divisible by three")
        size = sizes[0]
        mode_axis = tuple(int(value) for value in np.rint(np.fft.fftfreq(size) * size))
        modes = tuple((left, right) for left in mode_axis for right in mode_axis)
        points = 2.0 * np.pi * np.arange(size, dtype=float) / size
        fourier = np.exp(1j * np.outer(points, mode_axis)) / np.sqrt(size)
        product_fourier = np.kron(fourier, fourier)
        primitive = tuple(
            product_fourier @ matrix @ product_fourier.conj().T
            for matrix in self.torsion_representation(modes)
        )
        primitive = tuple(
            np.real_if_close(matrix, tol=1000).astype(complex) for matrix in primitive
        )
        rotations = {
            int(J): self.rotational_representation(int(J), representation)
            for J in sorted(set(j_values))
        }
        return LAMRotationSymmetry(self.group, primitive, rotations)

    def dipole_component_irreps(self) -> Mapping[str, str]:
        """Return the G36 irreps of the body-fixed electric-dipole components."""

        characters = {
            "a": tuple(complex((-1) ** (p_bit + r_bit)) for _, _, p_bit, r_bit in self.operations),
            "b": tuple(complex((-1) ** r_bit) for _, _, p_bit, r_bit in self.operations),
            "c": tuple(complex((-1) ** p_bit) for _, _, p_bit, r_bit in self.operations),
        }
        result: dict[str, str] = {}
        for axis, values in characters.items():
            matches = [
                irrep.label for irrep in self.group.irreps
                if irrep.dimension == 1 and np.allclose(irrep.characters, values, atol=1.0e-12)
            ]
            if len(matches) != 1:
                raise RuntimeError(f"G36 dipole component {axis} has no unique irrep")
            result[axis] = matches[0]
        return result

    def electric_dipole_allowed(self, lower: str, upper: str, component: str) -> bool:
        """Apply the exact finite-group selection rule for one dipole component."""

        axis = str(component).lower()
        try:
            dipole = self.group.irrep(self.dipole_component_irreps()[axis])
        except KeyError as exc:
            raise ValueError("G36 dipole component must be a, b, or c") from exc
        lower_irrep = self.group.irrep(lower)
        upper_irrep = self.group.irrep(upper)
        multiplicity = sum(
            np.conj(left) * middle * right
            for left, middle, right in zip(
                lower_irrep.characters,
                dipole.characters,
                upper_irrep.characters,
                strict=True,
            )
        ) / len(self.operations)
        return bool(abs(multiplicity) > 0.5)

    def diagonalize_rovibrational_blocks(
        self,
        hamiltonian: np.ndarray,
        modes: Sequence[tuple[int, int]],
        J: int,
        representation: str = "Ir",
        *,
        tolerance: float = 1.0e-9,
    ) -> tuple[G36SymmetryBlock, ...]:
        """Project and diagonalize a G36-invariant torsion--rotation matrix.

        The input matrix is expressed in the same ``Fourier x |J,K>`` basis as
        :meth:`rovibrational_representation`.  Generator commutation is checked
        before reduction, so a model without G36 symmetry is never silently
        forced into symmetry blocks.
        """
        generic_blocks = self.rovibrational_action(modes, J, representation).diagonalize(
            hamiltonian, tolerance=tolerance
        )
        return tuple(
            G36SymmetryBlock(
                block.irrep,
                block.basis,
                block.hamiltonian,
                block.energies,
                block.eigenvectors,
            )
            for block in generic_blocks
        )

    def validate_representation(self, matrices: Sequence[np.ndarray], *, tolerance: float = 1.0e-9) -> None:
        """Validate the four generator matrices and the 36 normal forms.

        Validation uses the group presentation rather than checking all 1,296
        operation products.  This keeps the cost cubic in the state dimension
        with a small fixed factor, even when the Fourier basis is enlarged.
        """
        values = tuple(np.asarray(matrix, dtype=complex) for matrix in matrices)
        if len(values) != 36 or not values:
            raise ValueError("a G36 representation needs one matrix per operation")
        shape = values[0].shape
        if len(shape) != 2 or shape[0] != shape[1] or any(item.shape != shape for item in values):
            raise ValueError("G36 representation matrices must be square and equally sized")
        identity = np.eye(shape[0], dtype=complex)
        generator_indices = tuple(
            self.operations.index(operation)
            for operation in ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
        )
        a, b, p_swap, r_reverse = (values[index] for index in generator_indices)
        for index, matrix in zip(generator_indices, (a, b, p_swap, r_reverse), strict=True):
            if not np.allclose(matrix.conj().T @ matrix, identity, atol=tolerance):
                raise ValueError(f"G36 operation {index} is not unitary")
        checks = (
            (a @ a @ a, identity),
            (b @ b @ b, identity),
            (a @ b, b @ a),
            (p_swap @ p_swap, identity),
            (r_reverse @ r_reverse, identity),
            (p_swap @ r_reverse, r_reverse @ p_swap),
            (p_swap @ a @ p_swap, b),
            (p_swap @ b @ p_swap, a),
            (r_reverse @ a @ r_reverse, a.conj().T),
            (r_reverse @ b @ r_reverse, b.conj().T),
        )
        if any(not np.allclose(left, right, atol=tolerance) for left, right in checks):
            raise ValueError("G36 representation generators violate the defining relations")
        for index, (i, j, p_bit, r_bit) in enumerate(self.operations):
            expected = np.linalg.matrix_power(a, i) @ np.linalg.matrix_power(b, j)
            if p_bit:
                expected = expected @ p_swap
            if r_bit:
                expected = expected @ r_reverse
            if not np.allclose(values[index], expected, atol=tolerance):
                raise ValueError(f"G36 operation {index} disagrees with its normal form")


__all__ = ["G36SymmetryBlock", "G36TwoTopAction"]
