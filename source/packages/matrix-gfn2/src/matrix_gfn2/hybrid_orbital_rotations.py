"""Experimental self-consistent occupied-virtual rotations by fragment range.

The Fock callback must evaluate the *same* energy functional as production
GFN2. This module does not itself approximate interfragment electrostatics or
replace the current production SCF. Fixed orbital-to-fragment labels are only
appropriate for the initial local stages; validation is required before use
on delocalized systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.linalg import solve

from .persistent_orbital_support import SparseOrbitalSupport, tidy_orbital_support


@dataclass(frozen=True)
class LocalRotationResult:
    coefficients: np.ndarray
    density: np.ndarray
    energy: float
    max_residual: float
    cycles: int
    reached_radius: float
    converged: bool


@dataclass(frozen=True)
class BlockLocalRotationResult:
    coefficients: np.ndarray
    density: np.ndarray
    energy: float
    max_residual: float
    macro_iterations: int
    accepted_pair_rotations: int
    stage_rotations: tuple[int, ...]
    unique_active_pairs: int
    stage_unique_pairs: tuple[int, ...]
    stage_allowed_pairs: tuple[int, ...]
    stage_residuals: tuple[float, ...]
    stage_energies: tuple[float, ...]
    support_pruning_attempts: int
    support_pruning_accepted: int
    minimum_candidate_orthogonality_error: float
    minimum_candidate_density_error: float
    final_coefficient_nnz: int
    final_orbital_atom_blocks: int
    sparse_fock_projections: int
    converged: bool


def solve_block_local_orbital_rotations(
    coefficients: np.ndarray,
    overlap: np.ndarray,
    orbital_fragments: np.ndarray,
    fragment_centers: np.ndarray,
    occupied: int,
    fock_and_energy: Callable[[np.ndarray], tuple[np.ndarray, float]],
    *,
    radii_angstrom: tuple[float, ...] = (0.0, 3.0, 6.0, np.inf),
    max_macro_iterations_per_radius: int = 40,
    residual_tolerance: float = 1.0e-7,
    maximum_column_rotation: float = 0.25,
    fragment_distances_angstrom: np.ndarray | None = None,
    orbital_atoms: np.ndarray | None = None,
    support_policy=None,
    rotation_scheme: str = "cayley",
) -> BlockLocalRotationResult:
    """Relax simultaneous occupied--virtual rotations without Fock eigensolves.

    A Cayley transform preserves S-orthonormality. Every trial evaluates the
    full supplied energy, and a global occupied--virtual residual is reported
    even if the final radius is finite. Pair counts are accepted nonzero
    amplitudes across all macro steps, not unique orbital pairs.
    """
    c = np.asarray(coefficients, dtype=float).copy()
    s = np.asarray(overlap, dtype=float)
    labels = np.asarray(orbital_fragments, dtype=np.intp)
    centers = np.asarray(fragment_centers, dtype=float)
    size = len(c)
    if (
        c.shape != (size, size) or s.shape != c.shape
        or labels.shape != (size,) or centers.ndim != 2 or centers.shape[1] != 3
        or np.any(labels < 0) or np.any(labels >= len(centers))
        or not 0 < occupied < size or max_macro_iterations_per_radius < 1
        or residual_tolerance <= 0.0 or maximum_column_rotation <= 0.0
        or not radii_angstrom or radii_angstrom[0] != 0.0
        or any(left >= right for left, right in zip(radii_angstrom, radii_angstrom[1:]))
        or not np.allclose(c.T @ s @ c, np.eye(size), atol=1.0e-8)
        or (support_policy is not None and orbital_atoms is None)
        or rotation_scheme not in ("cayley", "disjoint-pairs")
    ):
        raise ValueError("invalid block orbital-rotation model")
    if fragment_distances_angstrom is None:
        distance = np.linalg.norm(
            centers[labels[occupied:]][:, None]
            - centers[labels[:occupied]][None, :], axis=-1,
        )
    else:
        fragment_distance = np.asarray(fragment_distances_angstrom, dtype=float)
        if (fragment_distance.shape != (len(centers), len(centers))
                or np.any(fragment_distance < 0.0)
                or not np.all(np.isfinite(fragment_distance))):
            raise ValueError("invalid fragment distance matrix")
        distance = fragment_distance[
            labels[occupied:][:, None], labels[:occupied][None, :]
        ]
    density = _density(c, occupied)
    fock, energy = fock_and_energy(density)
    energy = float(energy)
    macro_iterations = 0
    stage_rotations = []
    stage_unique_pairs = []
    stage_allowed_pairs = []
    stage_residuals = []
    stage_energies = []
    pruning_attempts = 0
    pruning_accepted = 0
    sparse_fock_projections = 0
    sparse_support = None
    if support_policy is not None and np.count_nonzero(c) * 2 <= c.size:
        sparse_support = SparseOrbitalSupport.from_dense(c, occupied)
    minimum_candidate_orth_error = np.inf
    minimum_candidate_density_error = np.inf
    total_unique = np.zeros_like(distance, dtype=bool)
    identity = np.eye(size)
    for radius in radii_angstrom:
        allowed = distance <= radius + 1.0e-12
        stage_allowed_pairs.append(int(np.count_nonzero(allowed)))
        unique_in_stage = np.zeros_like(allowed, dtype=bool)
        count = 0
        for _ in range(max_macro_iterations_per_radius):
            if sparse_support is None:
                f_mo = c.T @ fock @ c
            else:
                f_mo = sparse_support.project_fock(fock)
                sparse_fock_projections += 1
            residual = f_mo[occupied:, :occupied]
            if not np.any(allowed) or float(np.max(abs(residual[allowed]))) < residual_tolerance:
                break
            gaps = np.diag(f_mo)[occupied:, None] - np.diag(f_mo)[None, :occupied]
            amplitude = np.where(
                allowed & (abs(residual) >= residual_tolerance),
                -residual / np.maximum(gaps, 0.02), 0.0,
            )
            if not np.any(amplitude):
                break
            # Many individually small steps can otherwise generate a large
            # collective occupied-to-virtual rotation.
            maximum_norm = max(
                float(np.max(np.linalg.norm(amplitude, axis=0))),
                float(np.max(np.linalg.norm(amplitude, axis=1))),
            )
            amplitude *= min(1.0, maximum_column_rotation / maximum_norm)
            if rotation_scheme == "disjoint-pairs":
                selected = np.zeros_like(amplitude, dtype=bool)
                used_occupied = np.zeros(occupied, dtype=bool)
                used_virtual = np.zeros(size - occupied, dtype=bool)
                for flat in np.argsort(-np.abs(amplitude), axis=None):
                    virtual, orbital = np.unravel_index(flat, amplitude.shape)
                    if amplitude[virtual, orbital] == 0.0:
                        break
                    if not used_occupied[orbital] and not used_virtual[virtual]:
                        selected[virtual, orbital] = True
                        used_occupied[orbital] = True
                        used_virtual[virtual] = True
                active = selected
            else:
                active = amplitude != 0.0
                generator = np.zeros((size, size))
                generator[occupied:, :occupied] = amplitude
                generator[:occupied, occupied:] = -amplitude.T
            accepted = False
            for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625):
                if rotation_scheme == "disjoint-pairs":
                    trial = c.copy()
                    virtual, orbital = np.nonzero(selected)
                    atom_virtual = occupied + virtual
                    angle = fraction * amplitude[virtual, orbital]
                    cosine = np.cos(angle)[None, :]
                    sine = np.sin(angle)[None, :]
                    old_occupied = c[:, orbital]
                    old_virtual = c[:, atom_virtual]
                    new_occupied = cosine * old_occupied + sine * old_virtual
                    trial[:, orbital] = new_occupied
                    trial[:, atom_virtual] = (
                        -sine * old_occupied + cosine * old_virtual
                    )
                    trial_density = density + 2.0 * (
                        new_occupied @ new_occupied.T
                        - old_occupied @ old_occupied.T
                    )
                else:
                    half = 0.5 * fraction * generator
                    rotation = solve(identity - half, identity + half, assume_a="gen")
                    trial = c @ rotation
                    trial_density = _density(trial, occupied)
                trial_fock, trial_energy = fock_and_energy(trial_density)
                if trial_energy < energy - 1.0e-13:
                    if support_policy is not None:
                        support = tidy_orbital_support(
                            trial, s, orbital_atoms, occupied, fock_and_energy,
                            reference_fock=trial_fock,
                            reference_energy=float(trial_energy),
                            policy=support_policy,
                        )
                        pruning_attempts += int(support.changed)
                        if support.changed:
                            minimum_candidate_orth_error = min(
                                minimum_candidate_orth_error,
                                support.maximum_orthogonality_error,
                            )
                            minimum_candidate_density_error = min(
                                minimum_candidate_density_error,
                                support.maximum_density_error,
                            )
                        if support.accepted and support.energy < energy - 1.0e-13:
                            trial, trial_density, trial_fock, trial_energy = (
                                support.coefficients, support.density,
                                support.fock, support.energy,
                            )
                            pruning_accepted += 1
                    c, density, fock, energy = (
                        trial, trial_density, trial_fock, float(trial_energy)
                    )
                    sparse_support = None
                    if (support_policy is not None
                            and np.count_nonzero(c) * 2 <= c.size):
                        sparse_support = SparseOrbitalSupport.from_dense(c, occupied)
                    macro_iterations += 1
                    count += int(np.count_nonzero(active))
                    unique_in_stage |= active
                    total_unique |= active
                    accepted = True
                    break
            if not accepted:
                break
        stage_rotations.append(count)
        stage_unique_pairs.append(int(np.count_nonzero(unique_in_stage)))
        if sparse_support is None:
            final_residual = c.T @ fock @ c
        else:
            final_residual = sparse_support.project_fock(fock)
            sparse_fock_projections += 1
        stage_residuals.append(float(np.max(abs(final_residual[occupied:, :occupied]))))
        stage_energies.append(energy)
    max_residual = stage_residuals[-1]
    return BlockLocalRotationResult(
        c, density, energy, max_residual, macro_iterations,
        sum(stage_rotations), tuple(stage_rotations),
        int(np.count_nonzero(total_unique)), tuple(stage_unique_pairs),
        tuple(stage_allowed_pairs), tuple(stage_residuals), tuple(stage_energies),
        pruning_attempts, pruning_accepted,
        minimum_candidate_orth_error,
        minimum_candidate_density_error,
        int(np.count_nonzero(c)),
        int(sum(len(np.unique(np.asarray(orbital_atoms)[np.flatnonzero(column)]))
                for column in c.T)) if orbital_atoms is not None else 0,
        sparse_fock_projections,
        bool(max_residual < residual_tolerance),
    )


def solve_from_localized_molecular_orbitals(
    coefficients: np.ndarray,
    overlap: np.ndarray,
    orbital_atoms: np.ndarray,
    atom_fragments: np.ndarray,
    fragment_centers: np.ndarray,
    occupied: int,
    fock_and_energy: Callable[[np.ndarray], tuple[np.ndarray, float]],
    *,
    localization_sweeps: int = 8,
    localization_backend: str = "python",
    radii_angstrom: tuple[float, ...] = (0.0, 3.0, 6.0, np.inf),
    max_cycles_per_radius: int = 8,
    residual_tolerance: float = 1.0e-7,
) -> tuple[object, LocalRotationResult]:
    """Localize equal-occupation MOs, then relax progressively wider pairs.

    This opt-in prototype retains the *complete* Fock/energy callback. It
    makes no claim that fixed orbital-to-fragment labels remain valid after
    extensive interfragment mixing or that the algorithm beats dense SCF.
    """
    from .localized_molecular_orbitals import localize_molecular_orbitals

    size = len(coefficients)
    if not 0 < occupied < size:
        raise ValueError("invalid occupied orbital count")
    occupations = np.zeros(size)
    occupations[:occupied] = 2.0
    localized = localize_molecular_orbitals(
        coefficients, overlap, orbital_atoms, occupations,
        maximum_sweeps=localization_sweeps, backend=localization_backend,
    )
    result = solve_local_orbital_rotations(
        localized.coefficients, overlap,
        localized.fragment_labels(atom_fragments), fragment_centers,
        occupied, fock_and_energy,
        radii_angstrom=radii_angstrom,
        max_cycles_per_radius=max_cycles_per_radius,
        residual_tolerance=residual_tolerance,
    )
    return localized, result


def _density(coefficients: np.ndarray, occupied: int) -> np.ndarray:
    block = coefficients[:, :occupied]
    return 2.0 * block @ block.T


def solve_local_orbital_rotations(
    coefficients: np.ndarray,
    overlap: np.ndarray,
    orbital_fragments: np.ndarray,
    fragment_centers: np.ndarray,
    occupied: int,
    fock_and_energy: Callable[[np.ndarray], tuple[np.ndarray, float]],
    *,
    radii_angstrom: tuple[float, ...] = (0.0, 3.0, 6.0, np.inf),
    max_cycles_per_radius: int = 8,
    residual_tolerance: float = 1.0e-7,
    maximum_rotation: float = 0.25,
) -> LocalRotationResult:
    """Relax local singles, then extend the allowed interfragment distance.

    A pairwise Givens rotation keeps C.T S C = I exactly to roundoff. Every
    accepted trial rebuilds the density and Fock and is energy-checked. The
    final residual is over *all* occupied-virtual pairs, not only the last
    truncated domain. A finite last radius therefore cannot certify an SCF.
    """
    c = np.asarray(coefficients, dtype=float).copy()
    s = np.asarray(overlap, dtype=float)
    fragments = np.asarray(orbital_fragments, dtype=np.intp)
    centers = np.asarray(fragment_centers, dtype=float)
    size = len(c)
    if (
        c.shape != (size, size) or s.shape != c.shape
        or fragments.shape != (size,) or centers.ndim != 2 or centers.shape[1] != 3
        or np.any(fragments < 0) or np.any(fragments >= len(centers))
        or not 0 < occupied < size or max_cycles_per_radius < 1
        or residual_tolerance <= 0.0 or maximum_rotation <= 0.0
        or not radii_angstrom or radii_angstrom[0] != 0.0
        or any(left >= right for left, right in zip(radii_angstrom, radii_angstrom[1:]))
        or not np.allclose(c.T @ s @ c, np.eye(size), atol=1.0e-8)
    ):
        raise ValueError("invalid local orbital-rotation model")
    distances = np.linalg.norm(
        centers[fragments[:occupied], None] - centers[fragments[occupied:]][None, :],
        axis=-1,
    )
    density = _density(c, occupied)
    fock, energy = fock_and_energy(density)
    energy = float(energy)
    cycles = 0
    for radius in radii_angstrom:
        allowed = np.argwhere(distances <= radius + 1.0e-12)
        for _ in range(max_cycles_per_radius):
            f_mo = c.T @ fock @ c
            residual = f_mo[occupied:, :occupied]
            if not len(allowed) or max(abs(residual[a, i]) for i, a in allowed) < residual_tolerance:
                break
            # Largest commutator element first; later pairs see the updated C.
            ranked = sorted(allowed, key=lambda pair: -abs(residual[pair[1], pair[0]]))
            accepted = False
            for i, a_local in ranked:
                a = occupied + a_local
                numerator = float(f_mo[a, i])
                if abs(numerator) < residual_tolerance:
                    continue
                gap = max(float(f_mo[a, a] - f_mo[i, i]), 1.0e-3)
                angle = -np.clip(numerator / gap, -maximum_rotation, maximum_rotation)
                for fraction in (1.0, 0.5, 0.25, 0.125):
                    trial = c.copy()
                    cosine, sine = np.cos(fraction * angle), np.sin(fraction * angle)
                    trial[:, i] = cosine * c[:, i] + sine * c[:, a]
                    trial[:, a] = -sine * c[:, i] + cosine * c[:, a]
                    trial_density = _density(trial, occupied)
                    trial_fock, trial_energy = fock_and_energy(trial_density)
                    if trial_energy < energy - 1.0e-13:
                        c, density, fock, energy = trial, trial_density, trial_fock, float(trial_energy)
                        accepted = True
                        break
                if accepted:
                    break
            if not accepted:
                break
            cycles += 1
    final_fock_mo = c.T @ fock @ c
    max_residual = float(np.max(abs(final_fock_mo[occupied:, :occupied])))
    return LocalRotationResult(
        c, density, energy, max_residual, cycles,
        float(radii_angstrom[-1]),
        bool(max_residual < residual_tolerance),
    )
