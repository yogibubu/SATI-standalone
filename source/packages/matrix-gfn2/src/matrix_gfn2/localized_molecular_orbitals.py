"""Density-preserving Pipek--Mezey localization in a nonorthogonal AO basis.

Atomic populations use symmetric Lowdin or Cholesky metric projectors. Rotations
are confined to orbitals with equal occupations, so localization changes
neither the density nor the energy of a fixed-density GFN2 calculation.
This is a localization primitive, not a truncated-SCF approximation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cholesky, eigh


_COMPILED_KERNEL = None


def _compiled_jacobi_kernel(orthogonal, coefficients, populations, atoms,
                            occupations, maximum_sweeps, objective_tolerance,
                            occupation_tolerance):
    """Nopython-compatible optional accelerator for the Jacobi sweeps."""
    size = len(occupations)
    natoms = len(populations)
    cross = np.zeros(natoms)
    rotations = 0
    sweeps = 0
    for sweep in range(maximum_sweeps):
        gain_this_sweep = 0.0
        for left in range(size):
            for right in range(left + 1, size):
                if abs(occupations[left] - occupations[right]) > occupation_tolerance:
                    continue
                cross[:] = 0.0
                for row in range(size):
                    cross[atoms[row]] += orthogonal[row, left] * orthogonal[row, right]
                a = 0.0
                b = 0.0
                d = 0.0
                for atom in range(natoms):
                    difference = 0.5 * (populations[atom, left] - populations[atom, right])
                    a += difference * difference
                    b += difference * cross[atom]
                    d += cross[atom] * cross[atom]
                gain = a + d + np.hypot(a - d, 2.0 * b) - 2.0 * a
                if gain <= objective_tolerance:
                    continue
                angle = 0.25 * np.arctan2(2.0 * b, a - d)
                if abs(angle) <= 1.0e-12:
                    continue
                cosine = np.cos(angle)
                sine = np.sin(angle)
                for row in range(size):
                    old_left = orthogonal[row, left]
                    old_right = orthogonal[row, right]
                    orthogonal[row, left] = cosine * old_left + sine * old_right
                    orthogonal[row, right] = -sine * old_left + cosine * old_right
                    old_left = coefficients[row, left]
                    old_right = coefficients[row, right]
                    coefficients[row, left] = cosine * old_left + sine * old_right
                    coefficients[row, right] = -sine * old_left + cosine * old_right
                populations[:, left] = 0.0
                populations[:, right] = 0.0
                for row in range(size):
                    populations[atoms[row], left] += orthogonal[row, left] ** 2
                    populations[atoms[row], right] += orthogonal[row, right] ** 2
                rotations += 1
                gain_this_sweep += gain
        sweeps = sweep + 1
        if gain_this_sweep <= objective_tolerance:
            break
    return sweeps, rotations


@dataclass(frozen=True)
class LocalizedMolecularOrbitals:
    coefficients: np.ndarray
    atom_populations: np.ndarray
    objective_before: float
    objective_after: float
    sweeps: int
    rotations: int
    density_error: float
    coefficient_derivative: np.ndarray | None = None
    transform: np.ndarray | None = None
    transform_derivative: np.ndarray | None = None
    atom_population_derivative: np.ndarray | None = None

    def fragment_labels(self, atom_fragments: np.ndarray) -> np.ndarray:
        """Assign each MO to its largest projector-population fragment."""
        labels = np.asarray(atom_fragments, dtype=np.intp)
        if (labels.shape != (len(self.atom_populations),)
                or np.any(labels < 0)):
            raise ValueError("invalid atom-to-fragment assignment")
        weights = np.zeros((int(np.max(labels)) + 1, self.coefficients.shape[1]))
        np.add.at(weights, labels, self.atom_populations)
        return np.argmax(weights, axis=0)


def _atomic_populations(
    orthogonal_coefficients: np.ndarray, orbital_atoms: np.ndarray, natoms: int,
) -> np.ndarray:
    weights = np.square(orthogonal_coefficients)
    result = np.zeros((natoms, weights.shape[1]))
    np.add.at(result, orbital_atoms, weights)
    return result


def _projector_seed_frame(
    coefficients: np.ndarray,
    overlap: np.ndarray,
    coefficient_derivative: np.ndarray | None,
    overlap_derivative: np.ndarray | None,
    *,
    symmetric_orthogonalization: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Build a deterministic frame from a subspace projector and its response."""
    c = np.asarray(coefficients, dtype=float)
    s = np.asarray(overlap, dtype=float)
    nao, nmo = c.shape
    projector = c @ (c.T @ s)
    projected_ao = projector
    selected: list[int] = []
    orthogonal_candidates = np.zeros((nao, nmo), dtype=float)
    while len(selected) < nmo:
        residuals = projected_ao.copy()
        if selected:
            q = orthogonal_candidates[:, :len(selected)]
            residuals -= q @ (q.T @ s @ residuals)
        norms = np.einsum("ij,ij->j", residuals, s @ residuals)
        norms[np.asarray(selected, dtype=int)] = -np.inf
        maximum = float(np.max(norms))
        if maximum <= 1.0e-8:
            raise ValueError("occupation subspace has no stable projected-AO frame")
        tie_window = max(1.0e-8, 1.0e-5 * maximum)
        candidate = int(np.flatnonzero(norms >= maximum - tie_window)[0])
        residual = residuals[:, candidate]
        norm = float(norms[candidate])
        selected.append(candidate)
        orthogonal_candidates[:, len(selected) - 1] = residual / np.sqrt(norm)
        if len(selected) == nmo:
            break
    if len(selected) != nmo:
        raise ValueError("occupation subspace has no stable projected-AO frame")
    seeds = projected_ao[:, selected]
    metric = seeds.T @ s @ seeds
    if not symmetric_orthogonalization:
        lower = np.linalg.cholesky(metric)
        frame = seeds @ np.linalg.inv(lower.T)
        return frame, None

    values, vectors = eigh(metric)
    if values[0] <= 1.0e-12:
        raise ValueError("projected-AO seed metric is singular")
    inverse_root = (vectors * (1.0 / np.sqrt(values))) @ vectors.T
    frame = seeds @ inverse_root
    if coefficient_derivative is None:
        return frame, None

    dc = np.asarray(coefficient_derivative, dtype=float)
    ds = np.asarray(overlap_derivative, dtype=float)
    coordinate_count = dc.shape[0]
    dprojected = np.empty((coordinate_count, nao, nao), dtype=float)
    for coordinate in range(coordinate_count):
        dprojected[coordinate] = (
            dc[coordinate] @ c.T @ s
            + c @ dc[coordinate].T @ s
            + c @ c.T @ ds[coordinate]
        )
    dseeds = dprojected[:, :, selected]
    dmetric = np.empty((coordinate_count, nmo, nmo), dtype=float)
    for coordinate in range(coordinate_count):
        dmetric[coordinate] = (
            dseeds[coordinate].T @ s @ seeds
            + seeds.T @ ds[coordinate] @ seeds
            + seeds.T @ s @ dseeds[coordinate]
        )
    inverse_roots = 1.0 / np.sqrt(values)
    divided = np.empty((nmo, nmo), dtype=float)
    for left in range(nmo):
        for right in range(nmo):
            if left == right:
                divided[left, right] = -0.5 * values[left] ** -1.5
            elif abs(values[left] - values[right]) <= 1.0e-12 * max(
                values[left], values[right], 1.0
            ):
                divided[left, right] = -0.5 * (0.5 * (values[left] + values[right])) ** -1.5
            else:
                divided[left, right] = (
                    inverse_roots[left] - inverse_roots[right]
                ) / (values[left] - values[right])
    dinverse_root = np.empty_like(dmetric)
    for coordinate in range(coordinate_count):
        in_eigenbasis = vectors.T @ dmetric[coordinate] @ vectors
        dinverse_root[coordinate] = vectors @ (divided * in_eigenbasis) @ vectors.T
    dframe = (
        np.einsum("pij,jk->pik", dseeds, inverse_root, optimize=True)
        + np.einsum("ij,pjk->pik", seeds, dinverse_root, optimize=True)
    )
    return frame, dframe


def localize_molecular_orbitals(
    coefficients: np.ndarray,
    overlap: np.ndarray,
    orbital_atoms: np.ndarray,
    occupations: np.ndarray,
    *,
    maximum_sweeps: int = 12,
    objective_tolerance: float = 1.0e-10,
    occupation_tolerance: float = 1.0e-8,
    backend: str = "python",
    population_scheme: str = "lowdin",
    coefficient_derivative: np.ndarray | None = None,
    overlap_derivative: np.ndarray | None = None,
) -> LocalizedMolecularOrbitals:
    """Maximize squared atom populations by occupation-preserving Jacobi sweeps.

    Columns with unequal occupations are never mixed. Consequently, a
    finite-temperature calculation can have small unlocalized fractional-
    occupation groups; the returned density remains exactly unchanged.
    """
    c = np.asarray(coefficients, dtype=float).copy()
    s = np.asarray(overlap, dtype=float)
    atoms = np.asarray(orbital_atoms, dtype=np.intp)
    occ = np.asarray(occupations, dtype=float)
    size = len(c)
    if (
        c.shape != (size, size) or s.shape != c.shape
        or atoms.shape != (size,) or occ.shape != (size,)
        or np.any(atoms < 0) or np.any(occ < 0.0)
        or not np.all(np.isfinite(c)) or not np.all(np.isfinite(s))
        or not np.all(np.isfinite(occ))
        or maximum_sweeps < 0 or objective_tolerance <= 0.0
        or occupation_tolerance < 0.0
        or backend not in ("python", "numba")
        or population_scheme not in ("lowdin", "cholesky")
        or not np.allclose(c.T @ s @ c, np.eye(size), atol=1.0e-8)
    ):
        raise ValueError("invalid molecular-orbital localization model")
    if population_scheme == "lowdin":
        eigenvalues, eigenvectors = eigh(s)
        if eigenvalues[0] <= 1.0e-10:
            raise ValueError("overlap must be positive definite")
        square_root = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    else:
        # S = U.T U. U C is Euclidean-orthonormal without a global
        # eigenproblem; the atomic projector is AO-order dependent, unlike
        # symmetric Lowdin partitioning. Report this scheme explicitly.
        square_root = cholesky(s, lower=False)
    orthogonal = square_root @ c
    natoms = int(np.max(atoms)) + 1
    populations = _atomic_populations(orthogonal, atoms, natoms)
    objective_before = float(np.sum(populations**2))
    derivative_requested = coefficient_derivative is not None or overlap_derivative is not None
    if derivative_requested:
        if backend != "python" or population_scheme != "lowdin":
            raise ValueError("analytic localization response currently requires the python backend")
        if coefficient_derivative is None or overlap_derivative is None:
            raise ValueError("coefficient and overlap derivatives must be supplied together")
        canonical_dc = np.asarray(coefficient_derivative, dtype=float).copy()
        dc = canonical_dc.copy()
        ds = np.asarray(overlap_derivative, dtype=float)
        if (dc.ndim != 3 or dc.shape[1:] != c.shape or ds.shape != (dc.shape[0], *s.shape)
                or not np.all(np.isfinite(dc)) or not np.all(np.isfinite(ds))):
            raise ValueError("orbital and overlap derivatives must have shapes ncoord by nao by nmo and ncoord by nao by nao")
        coordinate_count = dc.shape[0]
        # Frechet derivative of the symmetric Lowdin square root.
        eigenvalues, eigenvectors = eigh(s)
        roots = np.sqrt(eigenvalues)
        divided_difference = 1.0 / (roots[:, None] + roots[None, :])
        droot = np.empty_like(ds)
        for coordinate in range(coordinate_count):
            in_eigenbasis = eigenvectors.T @ ds[coordinate] @ eigenvectors
            droot[coordinate] = eigenvectors @ (divided_difference * in_eigenbasis) @ eigenvectors.T
        dorthogonal = np.einsum("pij,jk->pik", droot, c, optimize=True) + np.einsum(
            "ij,pjk->pik", square_root, dc, optimize=True
        )
        dpopulations = np.zeros((natoms, size, coordinate_count), dtype=float)
        for atom in range(natoms):
            rows = np.flatnonzero(atoms == atom)
            dpopulations[atom] = 2.0 * np.einsum(
                "ki,pki->ip", orthogonal[rows], dorthogonal[:, rows, :], optimize=True
            )
    else:
        dc = ds = dorthogonal = dpopulations = None
    # Start each equal-occupation subspace from projected AO vectors.  This
    # frame depends on its projector rather than arbitrary canonical MO
    # rotations, which keeps LMO identities reproducible near degeneracies.
    for occupation in np.unique(occ):
        group = np.flatnonzero(np.abs(occ - occupation) <= occupation_tolerance)
        if not len(group):
            continue
        seed, seed_derivative = _projector_seed_frame(
            c[:, group], s,
            None if not derivative_requested else dc[:, :, group],
            None if not derivative_requested else ds,
            symmetric_orthogonalization=population_scheme == "lowdin",
        )
        c[:, group] = seed
        if derivative_requested:
            assert seed_derivative is not None
            dc[:, :, group] = seed_derivative
    orthogonal = square_root @ c
    populations = _atomic_populations(orthogonal, atoms, natoms)
    seed_objective = float(np.sum(populations**2))
    if derivative_requested:
        assert dc is not None and ds is not None and droot is not None
        dorthogonal = np.einsum("pij,jk->pik", droot, c, optimize=True) + np.einsum(
            "ij,pjk->pik", square_root, dc, optimize=True
        )
        dpopulations = np.zeros((natoms, size, coordinate_count), dtype=float)
        for atom in range(natoms):
            rows = np.flatnonzero(atoms == atom)
            dpopulations[atom] = 2.0 * np.einsum(
                "ki,pki->ip", orthogonal[rows], dorthogonal[:, rows, :], optimize=True
            )
    density_before = (c * occ) @ c.T
    rotations = 0
    sweeps = 0
    if maximum_sweeps == 0:
        sweeps = 0
    elif backend == "numba":
        global _COMPILED_KERNEL
        try:
            from numba import njit
        except ImportError as exc:
            raise RuntimeError("numba backend requested but not installed") from exc
        if _COMPILED_KERNEL is None:
            _COMPILED_KERNEL = njit(cache=True)(_compiled_jacobi_kernel)
        sweeps, rotations = _COMPILED_KERNEL(
            orthogonal, c, populations, atoms, occ, maximum_sweeps,
            objective_tolerance, occupation_tolerance,
        )
    else:
        for sweep in range(maximum_sweeps):
            gain_this_sweep = 0.0
            for left in range(size):
                for right in range(left + 1, size):
                    if abs(occ[left] - occ[right]) > occupation_tolerance:
                        continue
                    difference = 0.5 * (populations[:, left] - populations[:, right])
                    cross = np.bincount(
                        atoms,
                        weights=orthogonal[:, left] * orthogonal[:, right],
                        minlength=natoms,
                    )
                    a = float(difference @ difference)
                    b = float(difference @ cross)
                    d = float(cross @ cross)
                    # The pair objective is constant + 2 ||difference*cos(2t)
                    # + cross*sin(2t)||^2. Its principal eigenvector gives t.
                    eigenvalue = 0.5 * (a + d + np.hypot(a - d, 2.0 * b))
                    gain = 2.0 * (eigenvalue - a)
                    if gain <= objective_tolerance:
                        continue
                    angle = 0.25 * np.arctan2(2.0 * b, a - d)
                    if abs(angle) <= 1.0e-12:
                        continue
                    cosine, sine = np.cos(angle), np.sin(angle)
                    left_column = orthogonal[:, left].copy()
                    right_column = orthogonal[:, right].copy()
                    if derivative_requested:
                        difference_derivative = 0.5 * (
                            dpopulations[:, left, :] - dpopulations[:, right, :]
                        )
                        cross_derivative = np.zeros((natoms, coordinate_count), dtype=float)
                        np.add.at(
                            cross_derivative, atoms,
                            (dorthogonal[:, :, left] * orthogonal[:, right][None, :]
                             + dorthogonal[:, :, right] * orthogonal[:, left][None, :]).T,
                        )
                        da = 2.0 * difference @ difference_derivative
                        db = difference_derivative.T @ cross + cross_derivative.T @ difference
                        dd = 2.0 * cross @ cross_derivative
                        atan_x = a - d
                        atan_y = 2.0 * b
                        denominator = atan_x * atan_x + atan_y * atan_y
                        dangle = np.zeros(coordinate_count, dtype=float)
                        if denominator > 1.0e-30:
                            dangle = 0.25 * (atan_x * 2.0 * db - atan_y * (da - dd)) / denominator
                        dleft = dorthogonal[:, :, left].copy()
                        dright = dorthogonal[:, :, right].copy()
                        dorthogonal[:, :, left] = (
                            cosine * dleft + sine * dright
                            + dangle[:, None] * (-sine * left_column + cosine * right_column)[None, :]
                        )
                        dorthogonal[:, :, right] = (
                            -sine * dleft + cosine * dright
                            + dangle[:, None] * (-cosine * left_column - sine * right_column)[None, :]
                        )
                        dleft = dc[:, :, left].copy()
                        dright = dc[:, :, right].copy()
                        dc[:, :, left] = (
                            cosine * dleft + sine * dright
                            + dangle[:, None] * (-sine * c[:, left] + cosine * c[:, right])[None, :]
                        )
                        dc[:, :, right] = (
                            -sine * dleft + cosine * dright
                            + dangle[:, None] * (-cosine * c[:, left] - sine * c[:, right])[None, :]
                        )
                    orthogonal[:, left] = cosine * left_column + sine * right_column
                    orthogonal[:, right] = -sine * left_column + cosine * right_column
                    left_column = c[:, left].copy()
                    right_column = c[:, right].copy()
                    c[:, left] = cosine * left_column + sine * right_column
                    c[:, right] = -sine * left_column + cosine * right_column
                    populations[:, (left, right)] = _atomic_populations(
                        orthogonal[:, (left, right)], atoms, natoms,
                    )
                    if derivative_requested:
                        for atom in range(natoms):
                            rows = np.flatnonzero(atoms == atom)
                            dpopulations[atom, left] = 2.0 * np.einsum(
                                "i,pi->p", orthogonal[rows, left], dorthogonal[:, rows, left], optimize=True
                            )
                            dpopulations[atom, right] = 2.0 * np.einsum(
                                "i,pi->p", orthogonal[rows, right], dorthogonal[:, rows, right], optimize=True
                            )
                    rotations += 1
                    gain_this_sweep += gain
            sweeps = sweep + 1
            if gain_this_sweep <= objective_tolerance:
                break
    populations = _atomic_populations(orthogonal, atoms, natoms)
    objective_after = float(np.sum(populations**2))
    density_after = (c * occ) @ c.T
    density_error = float(np.max(np.abs(density_after - density_before)))
    if objective_after + 1.0e-8 < seed_objective or density_error > 1.0e-7:
        raise RuntimeError(
            f"orbital localization violated a conserved quantity "
            f"(objective {objective_before:.6g}->{objective_after:.6g}; density {density_error:.3g})"
        )
    if derivative_requested:
        assert dc is not None and ds is not None and dorthogonal is not None
        transform = np.asarray(coefficients, dtype=float).T @ s @ c
        dtransform = (
            np.einsum("pia,ij,jk->pak", canonical_dc, s, c, optimize=True)
            + np.einsum("ij,pjk,kl->pil", np.asarray(coefficients, dtype=float).T, ds, c, optimize=True)
            + np.einsum("ij,jk,pkl->pil", np.asarray(coefficients, dtype=float).T, s, dc, optimize=True)
        )
        return LocalizedMolecularOrbitals(
            c, populations, objective_before, objective_after,
            sweeps, rotations, density_error, dc.copy(), transform,
            dtransform, dpopulations,
        )
    return LocalizedMolecularOrbitals(
        c, populations, objective_before, objective_after,
        sweeps, rotations, density_error,
    )
