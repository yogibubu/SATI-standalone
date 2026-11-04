"""ORACLE PI groups and geometry-dependent actions in LOCK's coupled basis."""

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.spatial.transform import Rotation

from matrix_chem.permutation_inversion_grid import PermutationInversionGridAction

from .molecular_symmetry import (
    EquivalentNuclearSpinSet,
    FiniteMolecularSymmetryGroup,
    MolecularSymmetryIrrep,
)
from .native_rotor import angular_momentum_operators, normalize_representation
from .rovibrational_symmetry import RovibrationalSymmetryAction


@lru_cache(maxsize=32)
def molecular_group_from_oracle_pi(contract) -> FiniteMolecularSymmetryGroup:
    """Derive complex irreducible characters from the finite class algebra.

    No group name or molecular character table is supplied. Class indicators
    normalized in the regular representation give commuting normal matrices;
    simultaneous Hermitian diagonalization separates their primitive central
    idempotents, whose coefficients are conjugate irreducible characters.
    """
    if contract.order is None or contract.order > 256:
        raise ValueError("LOCK PI character construction currently requires an expanded group of at most 256 operations")
    table = np.asarray(contract.multiplication_table(), dtype=int)
    n = len(table)
    inverse = np.argmin(table, axis=1)
    classes = []
    remaining = set(range(n))
    while remaining:
        seed = min(remaining)
        orbit = sorted({int(table[table[g, seed], inverse[g]]) for g in range(n)})
        classes.append(orbit)
        remaining.difference_update(orbit)
    nc = len(classes)
    class_of = np.empty(n, dtype=int)
    for c, members in enumerate(classes):
        class_of[members] = c
    sizes = np.asarray([len(c) for c in classes])
    spaces = [np.eye(nc, dtype=complex)]
    central_matrices = []
    for members in classes:
        matrix = np.zeros((nc, nc), dtype=complex)
        for j, domain in enumerate(classes):
            counts = np.bincount(class_of[table[np.ix_(members, domain)]].ravel(), minlength=nc)
            matrix[:, j] = counts / np.sqrt(sizes * sizes[j])
        central_matrices.append(matrix)
        for hermitian in ((matrix + matrix.conj().T) / 2, (matrix - matrix.conj().T) / (2j)):
            refined = []
            for space in spaces:
                values, vectors = np.linalg.eigh(space.conj().T @ hermitian @ space)
                tolerance = 1e-10 * max(1., float(np.max(np.abs(values))))
                boundaries = np.r_[0, np.flatnonzero(np.diff(values) > tolerance) + 1, len(values)]
                refined.extend(space @ vectors[:, a:b] for a, b in zip(boundaries[:-1], boundaries[1:], strict=True))
            spaces = refined
    if len(spaces) != nc or any(space.shape[1] != 1 for space in spaces):
        raise ValueError("PI class algebra did not resolve all irreducible characters")
    rows = []
    for space in spaces:
        vector = space[:, 0]
        if abs(vector[0]) < 1e-12:
            raise ValueError("invalid central idempotent: zero identity coefficient")
        vector = vector * np.exp(-1j * np.angle(vector[0]))
        for matrix in central_matrices:
            eigenvalue = np.vdot(vector, matrix @ vector)
            if not np.allclose(matrix @ vector, eigenvalue * vector, atol=1e-8, rtol=0):
                raise ValueError("unresolved PI central character eigenspace")
        characters = np.sqrt(n) * vector.conj()[class_of] / np.sqrt(sizes[class_of])
        dimension = int(round(characters[0].real))
        if dimension < 1 or abs(characters[0] - dimension) > 1e-8:
            raise ValueError("nonintegral PI irrep dimension")
        rows.append((dimension, tuple(complex(value) for value in characters)))
    rows.sort(key=lambda row: (
        not np.allclose(row[1], 1, atol=1e-8, rtol=0), row[0],
        tuple((round(x.real, 9), round(x.imag, 9)) for x in row[1]),
    ))
    irreps = tuple(MolecularSymmetryIrrep("A" if i == 0 else f"PI{i}", dim, chars)
                   for i, (dim, chars) in enumerate(rows))
    return FiniteMolecularSymmetryGroup(
        f"ORACLE-PI[{n}]", tuple("E" if i == 0 else f"p{i}{'*' if op.inverted else ''}"
                                for i, op in enumerate(contract.operations)),
        tuple(tuple(int(x) for x in row) for row in table), irreps,
        nuclear_spin_sets=_nuclear_spin_sets_from_oracle_pi(contract),
        identity_irrep="A",
    )


def _nuclear_spin_sets_from_oracle_pi(contract) -> tuple[EquivalentNuclearSpinSet, ...]:
    """Build equivalent-spin orbits directly from isotope-labelled PI actions."""
    from matrix_chem.isotopes_table import get_isotope
    from matrix_chem.topology.elements import atomic_number

    remaining = set(range(len(contract.atoms)))
    sets = []
    while remaining:
        seed = min(remaining)
        orbit = sorted({operation.permutation[seed] for operation in contract.operations})
        # Closure guarantees that the orbit of one point is invariant.
        remaining.difference_update(orbit)
        element = contract.atoms[seed]
        label = contract.nuclear_labels[seed]
        labelled_element, separator, mass_number = label.partition(":")
        isotope = (
            get_isotope(atomic_number(element) or 0, int(mass_number))
            if separator and mass_number.isdigit() and labelled_element == element
            else None
        )
        if isotope is None:
            continue
        local = {atom: index for index, atom in enumerate(orbit)}
        permutations = tuple(
            tuple(local[operation.permutation[atom]] for atom in orbit)
            for operation in contract.operations
        )
        sets.append(
            EquivalentNuclearSpinSet(
                label=f"{label}[{','.join(str(atom) for atom in orbit)}]",
                two_I=isotope.spin2,
                permutations=permutations,
            )
        )
    return tuple(sets)


@lru_cache(maxsize=128)
def _rotation_eigensystems(J, representation):
    return tuple(np.linalg.eigh(axis) for axis in angular_momentum_operators(J, representation))


def _body_rotation_matrices(rotations, J, representation):
    """Anti-representation U(C1 C2)=U(C2)U(C1) in the native |J,K> basis.

    Euler elementary-axis exponentials reuse three eigensystems per J. No
    per-grid matrix exponential, inferred phase, or J-dependent inversion sign
    is needed: the starred operation was already applied to the geometry.
    """
    if J == 0:
        return np.ones((len(rotations), 1, 1), dtype=complex)
    # Extrinsic xyz: C=Rz(gamma) Ry(beta) Rx(alpha). The anti-representation
    # reverses these factors. SciPy's gimbal gauge does not change their product.
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected.*", category=UserWarning)
        angles = Rotation.from_matrix(rotations).as_euler("xyz")
    sign = 1 if representation.endswith("l") else -1
    output = np.broadcast_to(np.eye(2 * J + 1, dtype=complex), (len(rotations), 2 * J + 1, 2 * J + 1)).copy()
    for axis, (values, vectors) in enumerate(_rotation_eigensystems(J, representation)):
        factors = np.exp(sign * 1j * angles[:, axis, None] * values)
        matrix = (vectors[None] * factors[:, None, :]) @ vectors.conj().T
        output = output @ matrix
    return output


@dataclass(frozen=True)
class PIGridLAMRotationSymmetry:
    """Coupled grid x rotation action; rotations may vary with the LAM point."""

    grid_action: PermutationInversionGridAction
    representation: str = "Ir"
    group: FiniteMolecularSymmetryGroup = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "representation", normalize_representation(self.representation))
        object.__setattr__(self, "group", molecular_group_from_oracle_pi(self.grid_action.contract))

    def contracted_action(self, lam_vectors, J):
        if isinstance(J, bool) or int(J) != J or J < 0:
            raise ValueError("PI rotational action requires nonnegative integer J")
        J = int(J)
        vectors = np.asarray(lam_vectors, dtype=complex)
        grid = self.grid_action
        if vectors.ndim != 2 or vectors.shape[0] != grid.targets.shape[1] or not np.all(np.isfinite(vectors)):
            raise ValueError("contracted LAM basis does not match the PI grid")
        if not np.allclose(vectors.conj().T @ vectors, np.eye(vectors.shape[1]), atol=1e-9, rtol=0):
            raise ValueError("PI contraction requires an orthonormal LAM basis")
        dimension = vectors.shape[1] * (2 * J + 1)
        identity = np.eye(dimension, dtype=complex)
        lookup = {op: i for i, op in enumerate(grid.contract.operations)}
        generators = []
        for op in grid.contract.generators:
            g = lookup[op]
            matrix = np.zeros((vectors.shape[1], 2*J+1, vectors.shape[1], 2*J+1), dtype=complex)
            # Bound temporary Wigner storage even for large DVR grids.
            for start in range(0, len(vectors), 64):
                stop = min(start + 64, len(vectors))
                rotations = _body_rotation_matrices(grid.compensating_rotations[g, start:stop], J, self.representation)
                matrix += np.einsum("qi,qab,qj->iajb", vectors[grid.targets[g, start:stop]].conj(),
                                    rotations, vectors[start:stop], optimize=True)
            matrix = matrix.reshape(dimension, dimension)
            if not np.allclose(matrix.conj().T @ matrix, identity, atol=1e-8, rtol=0):
                raise ValueError("retained LAM x rotation space is not closed under PI; increase lam_state_count")
            generators.append(matrix)
        actions, word_index = [identity], {(): 0}
        for word in grid.contract.words[1:]:
            actions.append(actions[word_index[word[:-1]]] @ generators[word[-1]])
            word_index[word] = len(actions) - 1
        return RovibrationalSymmetryAction(
            self.group, tuple(actions), tuple(lookup[op] for op in grid.contract.generators),
        )
