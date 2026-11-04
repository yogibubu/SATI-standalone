"""Nonlinear SMITH coordinate realization in a fixed Eckart frame.

Frozen-frame SOOP rows change under a subsequent rigid alignment.  The
coordinate and Eckart constraints therefore have to be solved together.
"""

from __future__ import annotations

import numpy as np

from .evaluation import build_gic_b_matrix, evaluate_gic_values
from .models import GICDefinition


def eckart_rigid_constraint_rows(coordinates_angstrom, masses_amu):
    """Six mass-weighted translation/rotation constraints at a nonlinear geometry.

    This geometry-only operation is also valid when the reactive internal
    coordinate set loses rank at separated fragments.
    """
    xyz=np.asarray(coordinates_angstrom,float);masses=np.asarray(masses_amu,float)
    if (xyz.ndim!=2 or xyz.shape[1]!=3 or masses.shape!=(len(xyz),)
            or np.any(masses<=0) or not np.all(np.isfinite(xyz))
            or not np.all(np.isfinite(masses))):
        raise ValueError('finite geometry and matching positive masses required')
    relative=xyz-np.average(xyz,axis=0,weights=masses)
    translation=np.empty((3,xyz.size));rotation=np.empty_like(translation)
    axes=np.eye(3)
    for atom,mass in enumerate(masses):
        translation[:,3*atom:3*atom+3]=mass*axes
        for axis in range(3):
            rotation[:,3*atom+axis]=mass*np.cross(relative[atom],axes[axis])
    return np.vstack((translation,rotation))


def eckart_projected_b_matrix(
    b_matrix: np.ndarray,
    masses_amu: np.ndarray,
    rigid_constraints: np.ndarray,
) -> np.ndarray:
    """Remove rigid components from B in the mass metric."""
    b = np.asarray(b_matrix, dtype=float)
    masses = np.asarray(masses_amu, dtype=float)
    rigid = np.asarray(rigid_constraints, dtype=float)
    weights = np.repeat(masses, 3)
    if b.ndim != 2 or b.shape[1] != len(weights):
        raise ValueError("B matrix and masses have inconsistent dimensions")
    if rigid.shape != (6, len(weights)):
        raise ValueError("Eckart constraints must have six Cartesian rows")
    inverse_mass_rigid = rigid.T / weights[:, None]
    return b - (b @ inverse_mass_rigid) @ np.linalg.solve(
        rigid @ inverse_mass_rigid, rigid,
    )


def eckart_rovibrational_metric_first_from_b_prime(
    b_matrix: np.ndarray,
    b_prime_cartesian: np.ndarray,
    coordinates: np.ndarray,
    masses_amu: np.ndarray,
    rigid_constraints: np.ndarray,
) -> np.ndarray:
    """Differentiate the complete Eckart metric using analytic B prime.

    All Cartesian quantities use the same length unit.  The fixed six-row
    constraint matrix defines the reference Eckart frame.  Moving rotational
    tangents are differentiated along with the vibrational tangents.
    """
    b = np.asarray(b_matrix, dtype=float)
    prime = np.asarray(b_prime_cartesian, dtype=float)
    xyz = np.asarray(coordinates, dtype=float)
    masses = np.asarray(masses_amu, dtype=float)
    rigid = np.asarray(rigid_constraints, dtype=float)
    n = xyz.size - 6
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or masses.shape != (len(xyz),)
            or b.shape != (n, xyz.size) or prime.shape != (n, xyz.size, xyz.size)
            or rigid.shape != (6, xyz.size) or np.any(masses <= 0)):
        raise ValueError("Eckart metric derivative has inconsistent dimensions")
    augmented = np.vstack((b, rigid))
    tangent = np.linalg.solve(
        augmented, np.vstack((np.eye(n), np.zeros((6, n)))),
    )
    contracted = np.einsum("iac,aj,ck->ijk", prime, tangent, tangent, optimize=True)
    second = -np.linalg.solve(
        augmented, np.concatenate((
            contracted.reshape(n, n * n), np.zeros((6, n * n)),
        ), axis=0),
    ).reshape(xyz.size, n, n)
    centered = xyz - np.average(xyz, axis=0, weights=masses)
    rotations = np.column_stack([
        np.cross(axis, centered).reshape(-1) for axis in np.eye(3)
    ])
    vectors = np.column_stack((tangent, rotations))
    weights = np.repeat(masses, 3)
    metric = np.linalg.inv(vectors.T @ (weights[:, None] * vectors))
    derivative = np.empty((n + 3, n + 3, n))
    for mode in range(n):
        rotation_prime = np.column_stack([
            np.cross(axis, tangent[:, mode].reshape(-1, 3)).reshape(-1)
            for axis in np.eye(3)
        ])
        vectors_prime = np.column_stack((second[:, :, mode], rotation_prime))
        covariant_prime = (
            vectors_prime.T @ (weights[:, None] * vectors)
            + vectors.T @ (weights[:, None] * vectors_prime)
        )
        derivative[:, :, mode] = -metric @ covariant_prime @ metric
    return derivative


class EckartConstrainedGICChart:
    """Realize frozen GIC displacements while retaining six Eckart constraints."""

    def __init__(
        self,
        definition: GICDefinition,
        coordinates_angstrom: np.ndarray,
        masses_amu: np.ndarray,
        *,
        tolerance: float = 1e-10,
        max_iterations: int = 12,
    ) -> None:
        reference = np.asarray(coordinates_angstrom, dtype=float)
        masses = np.asarray(masses_amu, dtype=float)
        if reference.ndim != 2 or reference.shape[1] != 3:
            raise ValueError("reference coordinates must have shape (natoms, 3)")
        if masses.shape != (len(reference),) or np.any(masses <= 0):
            raise ValueError("masses must be positive and match the atoms")
        if definition.rank != reference.size - 6:
            raise ValueError("Eckart chart requires a non-linear full-rank GIC basis")
        if not np.isfinite(tolerance) or tolerance <= 0 or max_iterations < 1:
            raise ValueError("invalid chart convergence controls")
        self.definition = definition
        self.reference_angstrom = reference.copy()
        self.masses_amu = masses.copy()
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        self.reference_values = evaluate_gic_values(
            definition, coordinates_angstrom=reference,
        )
        self.frozen_b_per_angstrom = np.asarray(
            build_gic_b_matrix(definition, coordinates_angstrom=reference).rows,
            dtype=float,
        )
        self._rigid_rows = eckart_rigid_constraint_rows(reference, masses)
        self.rigid_constraint_rows = self._rigid_rows.copy()
        self.eckart_projected_b_per_angstrom = eckart_projected_b_matrix(
            self.frozen_b_per_angstrom, masses, self._rigid_rows,
        )
        augmented = np.vstack((self.frozen_b_per_angstrom, self._rigid_rows))
        if np.linalg.matrix_rank(augmented) != reference.size:
            raise ValueError("GIC and Eckart constraints do not span Cartesian space")
        self._cache: dict[bytes, np.ndarray] = {}

    def geometry_at_displacement_angstrom(self, displacement: np.ndarray) -> np.ndarray:
        """Return geometry for a displacement in the frozen GIC units."""
        step = np.asarray(displacement, dtype=float)
        if step.shape != (self.definition.rank,) or not np.all(np.isfinite(step)):
            raise ValueError("GIC displacement has the wrong shape or non-finite values")
        key = np.ascontiguousarray(step).tobytes()
        if key not in self._cache:
            position = self.reference_angstrom.copy()
            for _ in range(self.max_iterations):
                residual = np.concatenate((
                    evaluate_gic_values(
                        self.definition, coordinates_angstrom=position,
                    ) - self.reference_values - step,
                    self._rigid_rows @ (
                        position - self.reference_angstrom
                    ).reshape(-1),
                ))
                if np.max(np.abs(residual)) < self.tolerance:
                    break
                jacobian = np.vstack((
                    build_gic_b_matrix(
                        self.definition, coordinates_angstrom=position,
                    ).rows,
                    self._rigid_rows,
                ))
                position -= np.linalg.solve(jacobian, residual).reshape(-1, 3)
            else:
                raise ValueError("joint GIC/Eckart chart did not converge")
            self._cache[key] = position
        return self._cache[key].copy()


def build_eckart_constrained_gic_chart(
    definition: GICDefinition,
    coordinates_angstrom: np.ndarray,
    masses_amu: np.ndarray,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 12,
) -> EckartConstrainedGICChart:
    """Build the general SMITH chart for curvilinear Eckart calculations."""
    return EckartConstrainedGICChart(
        definition, coordinates_angstrom, masses_amu,
        tolerance=tolerance, max_iterations=max_iterations,
    )
