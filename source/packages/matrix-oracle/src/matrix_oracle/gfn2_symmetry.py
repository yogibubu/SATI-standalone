"""Frozen ORACLE point-group contracts for autonomous MATRIX-GFN2."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from matrix_chem import (
    MolecularGeometry,
    MolecularSymmetry,
    analyze_molecular_symmetry,
    restrict_molecular_symmetry,
)
from matrix_gfn2 import FrozenPointGroup, FrozenSymmetryOperation

from .config import OracleSymmetryConfig


def freeze_matrix_gfn2_symmetry(
    atoms: Sequence[str],
    coordinates_angstrom: np.ndarray,
    *,
    symmetry: MolecularSymmetry | None = None,
    requested_point_group: str | None = None,
    thresholds: OracleSymmetryConfig = OracleSymmetryConfig(),
) -> FrozenPointGroup:
    """Freeze ORACLE's certified group for one MATRIX-GFN2 atom frame."""

    symbols = tuple(str(atom) for atom in atoms)
    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    if coordinates.shape != (len(symbols), 3) or not np.all(np.isfinite(coordinates)):
        raise ValueError("ORACLE-to-MATRIX-GFN2 coordinates must have shape natoms x 3")
    assigned = symmetry
    if assigned is None:
        assigned = analyze_molecular_symmetry(
            MolecularGeometry(atoms=symbols, coordinates_angstrom=coordinates),
            distance_tolerance=thresholds.distance_angstrom,
            inertia_tolerance=thresholds.inertia_relative,
            max_rotation_order=thresholds.max_rotation_order,
        )
    if requested_point_group:
        assigned = restrict_molecular_symmetry(assigned, requested_point_group)
    operations = tuple(
        FrozenSymmetryOperation(
            str(operation.label),
            tuple(tuple(float(value) for value in row) for row in operation.rotation),
            tuple(int(atom) - 1 for atom in operation.permutation),
        )
        for operation in assigned.operations
    )
    atom_orbits = tuple(
        tuple(int(atom) - 1 for atom in orbit) for orbit in assigned.atom_classes
    )
    tolerance = max(
        float(thresholds.distance_angstrom),
        2.0 * float(assigned.max_deviation),
        1.0e-7,
    )
    contract = FrozenPointGroup.create(
        point_group=assigned.point_group,
        atom_symbols=symbols,
        operations=operations,
        atom_orbits=atom_orbits,
        geometry_tolerance_angstrom=tolerance,
    )
    contract.validate()
    return contract


__all__ = ["freeze_matrix_gfn2_symmetry"]
