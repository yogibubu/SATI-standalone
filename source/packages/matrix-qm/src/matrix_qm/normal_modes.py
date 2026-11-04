"""Backend-independent Cartesian normal modes from a Cartesian Hessian.

This is the shared numerical boundary used by QM adapters and MATRIX-GF.  It
depends only on the canonical MATRIX-QM Hessian units and the common numerical
diagonalizer, so program adapters never need to import the higher-level GF
package.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from matrix_numerics import eigh_arrays


HESSIAN_EIGENVALUE_TO_CM = 5140.487143715055


@dataclass(frozen=True)
class CartesianNormalModeSet:
    """Vibrational modes in mass-weighted and display-ready Cartesian forms."""

    frequencies_cm: np.ndarray
    eigenvalues: np.ndarray
    mass_weighted_modes: np.ndarray
    cartesian_directions: np.ndarray
    external_rank: int
    source: str = ""


def mass_weighted_cartesian_hessian(
    hessian: np.ndarray,
    masses_amu: np.ndarray,
) -> np.ndarray:
    """Return a Cartesian Hessian mass-weighted by positive atomic masses."""

    hess = np.asarray(hessian, dtype=float)
    masses = np.asarray(masses_amu, dtype=float)
    expected = 3 * len(masses)
    if hess.shape != (expected, expected):
        raise ValueError(f"Hessian shape must be {(expected, expected)}, got {hess.shape}")
    if not np.all(np.isfinite(hess)):
        raise ValueError("Cartesian Hessian must be finite")
    hess = 0.5 * (hess + hess.T)
    if np.any(~np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("normal-mode masses must be finite and positive")
    weights = 1.0 / np.sqrt(np.repeat(masses, 3))
    return hess * weights[:, None] * weights[None, :]


def cartesian_normal_modes_from_hessian(
    cartesian_hessian_au: np.ndarray,
    masses_amu: np.ndarray,
    coordinates_bohr: np.ndarray,
    *,
    project_external: bool = True,
    source: str = "Cartesian Hessian",
) -> CartesianNormalModeSet:
    """Diagonalize a Cartesian Hessian after mass weighting and T/R projection."""

    masses = np.asarray(masses_amu, dtype=float)
    coordinates = np.asarray(coordinates_bohr, dtype=float)
    if coordinates.shape != (len(masses), 3):
        raise ValueError("normal-mode geometry must have shape (natoms, 3)")
    if np.any(~np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("normal-mode masses must be finite and positive")
    weighted = mass_weighted_cartesian_hessian(cartesian_hessian_au, masses)
    weighted = 0.5 * (weighted + weighted.T)
    external = _orthonormal_columns(_translation_rotation_basis(coordinates, masses))
    if project_external:
        vibrational_basis = _orthogonal_complement(external, weighted.shape[0])
    else:
        vibrational_basis = np.eye(weighted.shape[0], dtype=float)
        external = external[:, :0]
    if vibrational_basis.shape[1] == 0:
        raise ValueError("molecule has no vibrational Cartesian subspace")
    reduced = vibrational_basis.T @ weighted @ vibrational_basis
    eigenvalues, reduced_modes = eigh_arrays(0.5 * (reduced + reduced.T))
    order = np.argsort(eigenvalues)
    eigenvalues = np.asarray(eigenvalues[order], dtype=float)
    mass_weighted = (vibrational_basis @ reduced_modes[:, order]).T.reshape(
        (-1, len(masses), 3)
    )
    cartesian = mass_weighted / np.sqrt(masses)[None, :, None]
    cartesian = _normalize_display_directions(cartesian)
    frequencies = (
        np.sign(eigenvalues)
        * np.sqrt(np.abs(eigenvalues))
        * HESSIAN_EIGENVALUE_TO_CM
    )
    return CartesianNormalModeSet(
        frequencies_cm=frequencies,
        eigenvalues=eigenvalues,
        mass_weighted_modes=mass_weighted,
        cartesian_directions=cartesian,
        external_rank=external.shape[1],
        source=str(source),
    )


def _translation_rotation_basis(
    coordinates_bohr: np.ndarray,
    masses_amu: np.ndarray,
) -> np.ndarray:
    natoms = len(masses_amu)
    dimension = 3 * natoms
    sqrt_mass = np.sqrt(masses_amu)
    center = np.average(coordinates_bohr, axis=0, weights=masses_amu)
    shifted = coordinates_bohr - center
    vectors: list[np.ndarray] = []
    for axis in range(3):
        translation = np.zeros((natoms, 3), dtype=float)
        translation[:, axis] = sqrt_mass
        vectors.append(translation.reshape(dimension))
    for axis in np.eye(3):
        vectors.append((np.cross(axis, shifted) * sqrt_mass[:, None]).reshape(dimension))
    return np.column_stack(vectors)


def _orthonormal_columns(matrix: np.ndarray, rtol: float = 1.0e-12) -> np.ndarray:
    if matrix.size == 0:
        return np.asarray(matrix, dtype=float)
    u_matrix, singular_values, _vh = np.linalg.svd(matrix, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] == 0.0:
        return u_matrix[:, :0]
    rank = int(np.sum(singular_values > rtol * singular_values[0]))
    return u_matrix[:, :rank]


def _orthogonal_complement(basis: np.ndarray, dimension: int) -> np.ndarray:
    if basis.shape[1] == 0:
        return np.eye(dimension, dtype=float)
    _u, _s, vh = np.linalg.svd(basis.T, full_matrices=True)
    return vh[basis.shape[1] :, :].T


def _normalize_display_directions(modes: np.ndarray) -> np.ndarray:
    normalized = np.asarray(modes, dtype=float).copy()
    for index, mode in enumerate(normalized):
        scale = float(np.max(np.linalg.norm(mode, axis=1)))
        if scale > 0.0:
            normalized[index] = mode / scale
    return normalized


__all__ = [
    "HESSIAN_EIGENVALUE_TO_CM",
    "CartesianNormalModeSet",
    "cartesian_normal_modes_from_hessian",
    "mass_weighted_cartesian_hessian",
]
