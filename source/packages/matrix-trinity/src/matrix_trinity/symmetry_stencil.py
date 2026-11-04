"""Reuse Cartesian electronic stencil points related by exact molecular symmetry."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Callable, Sequence

import numpy as np


def _polarizability_action(rotation: np.ndarray) -> np.ndarray:
    pairs = ((0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2))
    columns = []
    for i, j in pairs:
        basis = np.zeros((3, 3))
        basis[i, j] = basis[j, i] = 1.0
        transformed = rotation @ basis @ rotation.T
        columns.append([transformed[a, b] for a, b in pairs])
    return np.asarray(columns).T


class SymmetryStencilEvaluator:
    """Transform E, gradient, Hessian and Cartesian property Jacobians.

    Operations use ORACLE's one-based atom permutation and row-vector
    Cartesian rotation. Every operation is checked at the reference geometry,
    and every reused point is checked against the requested geometry. This
    also works for a C1 chart embedded in a symmetric molecule.
    """

    def __init__(
        self, evaluator: Callable[[np.ndarray], object],
        reference_angstrom: np.ndarray, masses_amu: np.ndarray,
        operations: Sequence[object], *, tolerance_angstrom: float = 1e-8,
        gradient_offset: np.ndarray | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.reference = np.asarray(reference_angstrom, dtype=float)
        masses = np.asarray(masses_amu, dtype=float).reshape(-1)
        if (self.reference.shape != (len(masses), 3) or np.any(masses <= 0)
                or not np.isfinite(tolerance_angstrom) or tolerance_angstrom <= 0):
            raise ValueError("invalid symmetry stencil reference or tolerance")
        self.center = np.average(self.reference, axis=0, weights=masses)
        self.gradient_offset = (
            np.zeros(self.reference.size) if gradient_offset is None
            else np.asarray(gradient_offset, dtype=float).reshape(self.reference.size)
        )
        self.tolerance = float(tolerance_angstrom)
        self.operations: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for operation in operations:
            rotation = np.asarray(operation.rotation, dtype=float)
            permutation = np.asarray(operation.permutation, dtype=int) - 1
            if (rotation.shape != (3, 3) or permutation.shape != (len(masses),)
                    or sorted(permutation.tolist()) != list(range(len(masses)))
                    or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-10)):
                raise ValueError("invalid molecular symmetry operation")
            transformed = self._transform_geometry(self.reference, rotation, permutation)
            if np.max(abs(transformed - self.reference)) > self.tolerance:
                raise ValueError("symmetry operation does not preserve the reference geometry")
            if np.allclose(rotation, np.eye(3), atol=1e-12) and np.array_equal(
                permutation, np.arange(len(masses))
            ):
                continue
            cartesian = np.zeros((3 * len(masses), 3 * len(masses)))
            for source, target in enumerate(permutation):
                cartesian[3 * target:3 * target + 3, 3 * source:3 * source + 3] = rotation.T
            self.operations.append((rotation, permutation, cartesian))
        self.points: list[tuple[np.ndarray, object]] = []
        self.reused = 0

    def _transform_geometry(
        self, xyz: np.ndarray, rotation: np.ndarray, permutation: np.ndarray,
    ) -> np.ndarray:
        result = np.empty_like(xyz)
        result[permutation] = (xyz - self.center) @ rotation + self.center
        return result

    def _transform_result(self, source: object, rotation: np.ndarray, cartesian: np.ndarray) -> object:
        gradient = np.asarray(source.gradient_hartree_per_bohr, dtype=float).reshape(-1)
        hessian = np.asarray(source.hessian_hartree_per_bohr2, dtype=float)
        hessian = 0.5 * (hessian + hessian.T)
        properties = dict(
            gradient_hartree_per_bohr=(cartesian @ (gradient + self.gradient_offset)
                                       - self.gradient_offset),
            hessian_hartree_per_bohr2=cartesian @ hessian @ cartesian.T,
        )
        if hasattr(source, "dipole_derivatives_au_per_bohr"):
            derivative = np.asarray(source.dipole_derivatives_au_per_bohr, dtype=float)
            properties["dipole_derivatives_au_per_bohr"] = (
                rotation.T @ derivative @ cartesian.T
            )
        if hasattr(source, "polarizability_derivatives_au_per_bohr"):
            derivative = np.asarray(source.polarizability_derivatives_au_per_bohr, dtype=float)
            if derivative.shape != (6, cartesian.shape[0]):
                raise ValueError("Cartesian polarizability Jacobian has invalid shape")
            properties["polarizability_derivatives_au_per_bohr"] = (
                _polarizability_action(rotation).T @ derivative @ cartesian.T
            )
        for name in ("energy_hartree", "total_energy_hartree"):
            if hasattr(source, name):
                properties[name] = getattr(source, name)
        return SimpleNamespace(**properties)

    def __call__(self, xyz_angstrom: np.ndarray) -> object:
        xyz = np.asarray(xyz_angstrom, dtype=float).reshape(self.reference.shape)
        for stored_xyz, stored in self.points:
            if np.max(abs(stored_xyz - xyz)) <= self.tolerance:
                return stored
            for rotation, permutation, cartesian in self.operations:
                mapped = self._transform_geometry(stored_xyz, rotation, permutation)
                if np.max(abs(mapped - xyz)) <= self.tolerance:
                    self.reused += 1
                    result = self._transform_result(stored, rotation, cartesian)
                    self.points.append((xyz.copy(), result))
                    return result
        source = self.evaluator(xyz)
        size = xyz.size
        identity = self._transform_result(source, np.eye(3), np.eye(size))
        stabilizer = [
            (rotation, cartesian)
            for rotation, permutation, cartesian in self.operations
            if np.max(abs(self._transform_geometry(xyz, rotation, permutation) - xyz))
            <= self.tolerance
        ]
        if stabilizer:
            images = [identity] + [
                self._transform_result(source, rotation, cartesian)
                for rotation, cartesian in stabilizer
            ]
            result = SimpleNamespace(
                gradient_hartree_per_bohr=np.mean(
                    [item.gradient_hartree_per_bohr for item in images], axis=0),
                hessian_hartree_per_bohr2=np.mean(
                    [item.hessian_hartree_per_bohr2 for item in images], axis=0),
            )
            if hasattr(identity, "dipole_derivatives_au_per_bohr"):
                result.dipole_derivatives_au_per_bohr = np.mean(
                    [item.dipole_derivatives_au_per_bohr for item in images], axis=0,
                )
            if hasattr(identity, "polarizability_derivatives_au_per_bohr"):
                result.polarizability_derivatives_au_per_bohr = np.mean(
                    [item.polarizability_derivatives_au_per_bohr for item in images], axis=0,
                )
            for name in ("energy_hartree", "total_energy_hartree"):
                if hasattr(source, name):
                    setattr(result, name, getattr(source, name))
        else:
            result = identity
        self.points.append((xyz.copy(), result))
        return result
