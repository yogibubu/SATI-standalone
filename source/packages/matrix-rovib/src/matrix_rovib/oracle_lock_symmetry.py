"""Derive LOCK's LAM--rotation symmetry actions from ORACLE/SMITH artifacts.

Each frozen ORACLE Cartesian operation and atom permutation acts on every DVR
geometry. The transformed geometry is matched to the sampled grid modulo a
proper rigid rotation. The derivation refuses incomplete grids and operations
that do not preserve the represented DVR subspace.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class OracleLAMGridActions:
    """Actions in the flattened DVR grid order used to build the LOCK surface."""

    operation_labels: tuple[str, ...]
    actions: tuple[np.ndarray, ...]
    maximum_rmsd_angstrom: tuple[float, ...]
    group: object
    cartesian_rotations: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        if not self.operation_labels or len(self.operation_labels) != len(self.actions):
            raise ValueError("provide one grid action for every ORACLE operation")
        if len(self.maximum_rmsd_angstrom) != len(self.actions):
            raise ValueError("provide one matching residual for every ORACLE operation")
        if len(self.cartesian_rotations) != len(self.actions):
            raise ValueError("provide one Cartesian matrix for every ORACLE operation")
        matrices = tuple(np.asarray(item, dtype=complex) for item in self.actions)
        shape = matrices[0].shape
        if len(shape) != 2 or shape[0] != shape[1] or any(item.shape != shape for item in matrices):
            raise ValueError("LAM grid actions must be equally sized square matrices")
        identity = np.eye(shape[0], dtype=complex)
        if not np.allclose(matrices[0], identity, atol=1.0e-12):
            raise ValueError("the first ORACLE operation must be the identity")
        for label, matrix in zip(self.operation_labels, matrices, strict=True):
            if not np.allclose(matrix.conj().T @ matrix, identity, atol=1.0e-12):
                raise ValueError(f"ORACLE operation {label!r} does not permute the DVR grid")
        rotations = tuple(np.asarray(item, dtype=float) for item in self.cartesian_rotations)
        if any(
            item.shape != (3, 3)
            or not np.allclose(item.T @ item, np.eye(3), atol=1.0e-8)
            or not np.isclose(abs(np.linalg.det(item)), 1.0, atol=1.0e-8)
            for item in rotations
        ):
            raise ValueError("ORACLE Cartesian operation matrices must be orthogonal")
        object.__setattr__(self, "actions", matrices)
        object.__setattr__(self, "cartesian_rotations", rotations)


def derive_oracle_lam_grid_actions(
    oracle_geometry_xyz: Path | str,
    dvr_geometry_xyz: Path | str,
    *,
    masses_amu: Sequence[float],
    body_axes: Sequence[Sequence[float]] | None = None,
    tolerance_angstrom: float = 2.0e-4,
) -> OracleLAMGridActions:
    """Derive DVR-grid permutation matrices from ORACLE symmetry and scan XYZ.

    ``oracle_geometry_xyz`` must contain ORACLE's frozen ``[SYMMETRY]``
    section. ``dvr_geometry_xyz`` is the ordered multi-frame trajectory used
    for the LAM grid. Grid points are matched by mass-weighted proper-Kabsch
    RMSD after the ORACLE operation's atom mapping and Cartesian matrix are
    applied. A non-closed or undersampled grid raises instead of silently
    assigning approximate symmetry.
    """
    from matrix_chem import read_molecular_symmetry

    symmetry = read_molecular_symmetry(Path(oracle_geometry_xyz))
    atom_names, frames = _read_xyz_trajectory(Path(dvr_geometry_xyz))
    masses = np.asarray(masses_amu, dtype=float)
    natoms = len(atom_names)
    if masses.shape != (natoms,) or np.any(masses <= 0.0) or not np.all(np.isfinite(masses)):
        raise ValueError("masses_amu must contain one positive finite mass per DVR atom")
    if not np.isfinite(tolerance_angstrom) or tolerance_angstrom <= 0.0:
        raise ValueError("tolerance_angstrom must be positive and finite")
    if any(len(operation.permutation) != natoms for operation in symmetry.operations):
        raise ValueError("ORACLE operation atom count differs from the DVR trajectory")
    axes = np.eye(3) if body_axes is None else np.asarray(body_axes, dtype=float)
    if (
        axes.shape != (3, 3)
        or not np.allclose(axes.T @ axes, np.eye(3), atol=1.0e-8)
        or not np.isclose(np.linalg.det(axes), 1.0, atol=1.0e-8)
    ):
        raise ValueError("body_axes must be a right-handed 3-by-3 orthogonal matrix")
    centered = tuple(_center(frame, masses) @ axes for frame in frames)
    operations: list[np.ndarray] = []
    residuals: list[float] = []
    rotations: list[np.ndarray] = []
    for operation_index, operation in enumerate(symmetry.operations):
        rotation = axes.T @ np.asarray(operation.rotation, dtype=float) @ axes
        permutation = np.asarray(operation.permutation, dtype=int) - 1
        if (
            rotation.shape != (3, 3)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-8)
            or not np.isclose(abs(np.linalg.det(rotation)), 1.0, atol=1.0e-8)
            or sorted(permutation.tolist()) != list(range(natoms))
        ):
            raise ValueError(f"invalid ORACLE operation {operation.label!r}")
        if not np.allclose(masses, masses[permutation], rtol=0.0, atol=1.0e-10):
            raise ValueError(
                f"ORACLE operation {operation.label!r} exchanges nuclei with different masses"
            )
        inverse = np.argsort(permutation)
        action = np.zeros((len(frames), len(frames)), dtype=complex)
        operation_residual = 0.0
        for source_index, frame in enumerate(centered):
            transformed = frame[inverse] @ rotation
            distances = [
                _proper_kabsch_rmsd(transformed, candidate, masses)
                for candidate in centered
            ]
            target_index = int(np.argmin(distances))
            rmsd = float(distances[target_index])
            if rmsd > tolerance_angstrom:
                raise ValueError(
                    f"ORACLE operation {operation.label!r} maps DVR point {source_index} "
                    f"outside the sampled grid (best mass-weighted RMSD {rmsd:.6g} Å)"
                )
            if abs(action[target_index, source_index]) > 0.0:
                raise ValueError(
                    f"ORACLE operation {operation.label!r} maps multiple DVR points "
                    "to the same grid point"
                )
            action[target_index, source_index] = 1.0
            operation_residual = max(operation_residual, rmsd)
        operations.append(action)
        residuals.append(operation_residual)
        rotations.append(rotation)
    group = _finite_group_from_oracle(symmetry.point_group, symmetry.operations, rotations)
    return OracleLAMGridActions(
        tuple(operation.label for operation in symmetry.operations),
        tuple(operations),
        tuple(residuals),
        group,
        tuple(rotations),
    )


def derive_oracle_lam_rotation_symmetry(
    oracle_geometry_xyz: Path | str,
    dvr_geometry_xyz: Path | str,
    *,
    j_values: Sequence[int],
    body_axes: Sequence[Sequence[float]] | None = None,
    representation: str = "Ir",
    tolerance_angstrom: float = 2.0e-4,
):
    """Build LOCK's full finite-group LAM--rotation action from ORACLE.

    The group, primitive-grid actions and rotational Wigner matrices all come
    from the same ORACLE operation ordering. Every representation is checked
    against the multiplication table before it is returned to the solver.
    """
    from matrix_chem import atomic_mass
    from matrix_chem.topology.elements import atomic_number
    from matrix_rovib import LAMRotationSymmetry

    from matrix_core import read_sectioned_lines, section_content
    if section_content(read_sectioned_lines(Path(oracle_geometry_xyz)), "PERMUTATION_INVERSION"):
        from matrix_chem import read_permutation_inversion_symmetry, derive_permutation_inversion_grid_action
        from matrix_rovib import PIGridLAMRotationSymmetry

        contract = read_permutation_inversion_symmetry(oracle_geometry_xyz)
        atoms, frames = _read_xyz_trajectory(Path(dvr_geometry_xyz))
        if atoms != contract.atoms:
            raise ValueError("DVR atom order differs from the ORACLE PI contract")
        axes, masses = _principal_body_frame(oracle_geometry_xyz, masses_amu=contract.nuclear_masses_amu())
        selected_axes = axes if body_axes is None else np.asarray(body_axes, dtype=float)
        if selected_axes.shape != (3, 3) or not np.allclose(selected_axes.T @ selected_axes, np.eye(3), atol=1e-9) or not np.isclose(np.linalg.det(selected_axes), 1):
            raise ValueError("body_axes must be a proper orthogonal frame")
        grid = derive_permutation_inversion_grid_action(
            contract, np.asarray(frames) @ selected_axes, masses,
            tolerance_angstrom=tolerance_angstrom,
        )
        return PIGridLAMRotationSymmetry(grid, representation)

    axes, masses = _principal_body_frame(oracle_geometry_xyz)
    selected_axes = axes if body_axes is None else np.asarray(body_axes, dtype=float)
    # The caller may pass the exact frame used to create the surface. This
    # avoids recomputing an eigenvector gauge from a different geometry.
    if body_axes is not None:
        from matrix_chem import read_enriched_xyz

        geometry = read_enriched_xyz(Path(oracle_geometry_xyz))
        masses = np.asarray(
            [atomic_mass(atomic_number(atom) or 0) for atom in geometry.atoms], dtype=float
        )
    grid = derive_oracle_lam_grid_actions(
        oracle_geometry_xyz,
        dvr_geometry_xyz,
        masses_amu=masses,
        body_axes=selected_axes,
        tolerance_angstrom=tolerance_angstrom,
    )
    rotations = _rotational_actions(
        grid.group, grid.cartesian_rotations, tuple(j_values), representation
    )
    return LAMRotationSymmetry(
        grid.group,
        grid.actions,
        rotations,
    )


def gf_torsion_periodicities(
    gf_ped_xyz: Path | str,
    coordinates: Sequence[str],
) -> tuple[int, ...]:
    """Read periodicities for selected LAM coordinates from ORACLE/GF PED.

    Coordinate names must match a GF DVR-plan ``name`` or ``identifier``.
    Ambiguous or unsupported coordinates are errors; periodicity is never
    guessed from a point-group label.
    """
    from importlib import import_module
    read_gf_ped_section = import_module("matrix_gf").read_gf_ped_section

    section = read_gf_ped_section(Path(gf_ped_xyz))
    plan = tuple(row for row in section.large_amplitude_dvr_plan if row.periodicity)
    answer: list[int] = []
    for coordinate in coordinates:
        matches = [
            row for row in plan
            if coordinate.casefold() in {row.name.casefold(), row.identifier.casefold()}
        ]
        if len(matches) != 1:
            raise ValueError(
                f"GF DVR plan must identify exactly one torsion for coordinate {coordinate!r}"
            )
        answer.append(int(matches[0].periodicity))
    return tuple(answer)


def _finite_group_from_oracle(point_group: str, operations, rotations):
    from matrix_rovib import FiniteMolecularSymmetryGroup, MolecularSymmetryIrrep
    from importlib import import_module
    irrep_characters_for_operations = import_module(
        "matrix_smith"
    ).irrep_characters_for_operations

    labels = tuple(str(operation.label) for operation in operations)
    permutations = tuple(tuple(int(value) - 1 for value in operation.permutation) for operation in operations)
    matrices = tuple(np.asarray(value, dtype=float) for value in rotations)
    if not labels or labels[0].casefold() != "e":
        raise ValueError("ORACLE symmetry operations must place identity E first")
    raw_irreps = irrep_characters_for_operations(
        labels,
        point_group=point_group,
        operation_matrices=tuple(tuple(tuple(float(v) for v in row) for row in item) for item in matrices),
    )
    if not raw_irreps:
        raise ValueError(f"SMITH has no character table for ORACLE point group {point_group!r}")
    irreps = tuple(
        MolecularSymmetryIrrep(
            label=name,
            dimension=int(round(float(characters[0]))),
            characters=tuple(complex(value) for value in characters),
        )
        for name, characters in raw_irreps
    )

    # ORACLE stores an operation as a Cartesian matrix plus a source-to-target
    # atom map. Probe both standard composition conventions and accept only a
    # table that the finite-group contract proves associative and closed.
    for matrix_order in (0, 1):
        for permutation_order in (0, 1):
            table: list[list[int]] = []
            valid = True
            for left in range(len(labels)):
                row = []
                for right in range(len(labels)):
                    matrix_product = (
                        matrices[left] @ matrices[right]
                        if matrix_order == 0
                        else matrices[right] @ matrices[left]
                    )
                    permutation_product = tuple(
                        permutations[left][permutations[right][atom]]
                        if permutation_order == 0
                        else permutations[right][permutations[left][atom]]
                        for atom in range(len(permutations[0]))
                    )
                    matches = [
                        index for index, (candidate_matrix, candidate_permutation) in enumerate(
                            zip(matrices, permutations, strict=True)
                        )
                        if candidate_permutation == permutation_product
                        and np.allclose(candidate_matrix, matrix_product, atol=1.0e-7)
                    ]
                    if len(matches) != 1:
                        valid = False
                        break
                    row.append(matches[0])
                if not valid:
                    break
                table.append(row)
            if not valid:
                continue
            try:
                identity_label = next(
                    (
                        irrep.label
                        for irrep in irreps
                        if np.allclose(irrep.characters, np.ones(len(labels)))
                    ),
                    None,
                )
                if identity_label is None:
                    continue
                return FiniteMolecularSymmetryGroup(
                    name=f"ORACLE:{point_group}",
                    operations=labels,
                    multiplication_table=tuple(tuple(row) for row in table),
                    irreps=irreps,
                    identity_irrep=identity_label,
                )
            except ValueError:
                continue
    raise ValueError("ORACLE operation matrices and atom permutations do not form a finite group")


def _rotational_actions(group, cartesian_rotations, j_values, representation):
    from scipy.linalg import expm
    from scipy.spatial.transform import Rotation

    from matrix_rovib.native_rotor import angular_momentum_operators

    table = group.multiplication_table
    result: dict[int, tuple[np.ndarray, ...]] = {}
    for J in j_values:
        if isinstance(J, bool) or int(J) != J or J < 0:
            raise ValueError("J values must be nonnegative integers")
        Ja, Jb, Jc = angular_momentum_operators(int(J), representation)
        op_matrices = []
        for operation in cartesian_rotations:
            determinant = float(np.linalg.det(operation))
            proper = determinant * np.asarray(operation, dtype=float)
            rotation = Rotation.from_matrix(proper)
            vector = rotation.as_rotvec()
            angle = float(np.linalg.norm(vector))
            if angle <= 1.0e-14:
                op_matrices.append(
                    (determinant**int(J))
                    * np.eye(2 * int(J) + 1, dtype=complex)
                )
                continue
            axis = vector / angle
            generator = axis[0] * Ja + axis[1] * Jb + axis[2] * Jc
            op_matrices.append((determinant**int(J)) * expm(-1j * angle * generator))
        if _is_group_representation(op_matrices, table):
            result[int(J)] = tuple(op_matrices)
            continue
        # Body-fixed angular momentum has the reverse commutator convention.
        # The inverse action is the equivalent state-space convention; accept
        # it only if the complete ORACLE multiplication table validates.
        inverse_matrices = [matrix.conj().T for matrix in op_matrices]
        if _is_group_representation(inverse_matrices, table):
            result[int(J)] = tuple(inverse_matrices)
            continue
        raise ValueError(
            f"ORACLE Cartesian operations do not induce a validated J={J} rotational representation"
        )
    return result


def _is_group_representation(matrices, table) -> bool:
    identity = np.eye(matrices[0].shape[0], dtype=complex)
    if not np.allclose(matrices[0], identity, atol=1.0e-8):
        return False
    return all(
        np.allclose(matrices[left] @ matrices[right], matrices[product], atol=1.0e-8)
        for left, row in enumerate(table)
        for right, product in enumerate(row)
    )


def _principal_body_frame(oracle_geometry_xyz: Path | str, *, masses_amu=None) -> tuple[np.ndarray, np.ndarray]:
    from matrix_chem import atomic_mass, read_enriched_xyz
    from matrix_chem.topology.elements import atomic_number

    geometry = read_enriched_xyz(Path(oracle_geometry_xyz))
    masses = np.asarray(
        [atomic_mass(atomic_number(atom) or 0) for atom in geometry.atoms]
        if masses_amu is None else masses_amu, dtype=float
    )
    if masses.shape != (len(geometry.atoms),) or not np.all(np.isfinite(masses)) or np.any(masses <= 0):
        raise ValueError("one positive finite mass is required per body-frame atom")
    coordinates = np.asarray(geometry.coordinates_angstrom, dtype=float)
    centered = coordinates - np.average(coordinates, axis=0, weights=masses)
    inertia = np.zeros((3, 3), dtype=float)
    for mass, vector in zip(masses, centered, strict=True):
        inertia += mass * (float(vector @ vector) * np.eye(3) - np.outer(vector, vector))
    _moments, axes = np.linalg.eigh(inertia)
    for column in range(2):
        pivot = int(np.argmax(np.abs(axes[:, column])))
        if axes[pivot, column] < 0.0:
            axes[:, column] *= -1.0
    if np.linalg.det(axes) < 0.0:
        axes[:, 2] *= -1.0
    return axes, masses


def _center(frame: np.ndarray, masses: np.ndarray) -> np.ndarray:
    return frame - np.average(frame, axis=0, weights=masses)


def _proper_kabsch_rmsd(left: np.ndarray, right: np.ndarray, masses: np.ndarray) -> float:
    covariance = (masses[:, None] * left).T @ right
    u, _singular, vh = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vh)
    rotation = u @ correction @ vh
    residual = left @ rotation - right
    return float(np.sqrt(np.sum(masses[:, None] * residual**2) / np.sum(masses)))


def _read_xyz_trajectory(path: Path) -> tuple[tuple[str, ...], tuple[np.ndarray, ...]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    cursor = 0
    atoms: tuple[str, ...] | None = None
    frames: list[np.ndarray] = []
    while cursor < len(lines):
        if not lines[cursor].strip():
            cursor += 1
            continue
        try:
            count = int(lines[cursor].strip())
        except ValueError as exc:
            raise ValueError(f"invalid XYZ atom count at line {cursor + 1}") from exc
        if count < 1 or cursor + count + 1 >= len(lines):
            raise ValueError("truncated or empty multi-frame XYZ trajectory")
        symbols: list[str] = []
        coordinates: list[list[float]] = []
        for line in lines[cursor + 2 : cursor + count + 2]:
            fields = line.split()
            if len(fields) < 4:
                raise ValueError("invalid atom row in multi-frame XYZ trajectory")
            symbols.append(fields[0])
            coordinates.append([float(value) for value in fields[1:4]])
        current_atoms = tuple(symbols)
        if atoms is None:
            atoms = current_atoms
        elif current_atoms != atoms:
            raise ValueError("atom order changes between DVR frames")
        frames.append(np.asarray(coordinates, dtype=float))
        cursor += count + 2
    if atoms is None or not frames:
        raise ValueError("empty multi-frame XYZ trajectory")
    return atoms, tuple(frames)


__all__ = [
    "OracleLAMGridActions",
    "derive_oracle_lam_grid_actions",
    "derive_oracle_lam_rotation_symmetry",
    "gf_torsion_periodicities",
]
