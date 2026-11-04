"""Experimental atom-local storage of the occupied GFN2 subspace.

Only occupied columns are retained.  A proposed atom-block truncation is
accepted against the *same* Fock/energy functional and a supplied complete
gradient evaluator.  No performance or SCF convergence claim follows from a
successful truncation certificate alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.optimize import minimize
from scipy.sparse import csc_matrix, identity, issparse


@dataclass(frozen=True)
class OccupiedSubspace:
    coefficients: csc_matrix
    orbital_atoms: np.ndarray

    @classmethod
    def from_dense(cls, coefficients: np.ndarray, orbital_atoms: np.ndarray):
        c = np.asarray(coefficients, dtype=float)
        atoms = np.asarray(orbital_atoms, dtype=np.intp)
        if (c.ndim != 2 or not 0 < c.shape[1] <= c.shape[0]
                or atoms.shape != (c.shape[0],) or np.any(atoms < 0)
                or not np.all(np.isfinite(c))):
            raise ValueError("invalid occupied subspace")
        return cls(csc_matrix(c), atoms.copy())

    @property
    def shape(self) -> tuple[int, int]:
        return self.coefficients.shape

    @property
    def fill_fraction(self) -> float:
        n, occupied = self.shape
        return self.coefficients.nnz / (n * occupied)

    def density(self) -> np.ndarray:
        c = self.coefficients
        return (2.0 * (c @ c.T)).toarray()

    def weighted_density(self, fock: np.ndarray) -> np.ndarray:
        """Pulay matrix 2 C (C^T F C) C^T for noncanonical occupied MOs."""
        f = np.asarray(fock, dtype=float)
        if f.shape != (self.shape[0],) * 2:
            raise ValueError("Fock shape does not match the occupied subspace")
        c = self.coefficients
        projected = np.asarray(c.T @ (f @ c))
        return np.asarray(2.0 * c @ projected @ c.T)

    def orthogonality_error(self, overlap: np.ndarray) -> float:
        s = np.asarray(overlap, dtype=float)
        if s.shape != (self.shape[0],) * 2:
            raise ValueError("overlap shape does not match the occupied subspace")
        gram = np.asarray(self.coefficients.T @ (s @ self.coefficients))
        return float(np.max(abs(gram - np.eye(self.shape[1]))))


@dataclass(frozen=True)
class MetricOccupiedSubspace:
    """Local occupied columns with their exact, generally nonidentity metric.

    Keeping the metric avoids spreading each column across distant atoms to
    enforce global S-orthogonality. The metric inverse is still dense, so this
    representation alone makes no linear-scaling or timing claim.
    """

    coefficients: csc_matrix
    orbital_atoms: np.ndarray
    metric: np.ndarray
    inverse_metric: np.ndarray

    @classmethod
    def from_dense(
        cls, coefficients: np.ndarray, overlap: np.ndarray,
        orbital_atoms: np.ndarray,
    ) -> MetricOccupiedSubspace:
        c = np.asarray(coefficients, dtype=float)
        s = np.asarray(overlap, dtype=float)
        atoms = np.asarray(orbital_atoms, dtype=np.intp)
        if (c.ndim != 2 or not 0 < c.shape[1] <= c.shape[0]
                or s.shape != (c.shape[0], c.shape[0])
                or atoms.shape != (c.shape[0],) or np.any(atoms < 0)
                or not np.all(np.isfinite(c)) or not np.all(np.isfinite(s))):
            raise ValueError("invalid metric occupied subspace")
        gram = c.T @ s @ c
        metric = 0.5 * (gram + gram.T)
        try:
            factor = cho_factor(metric, check_finite=False)
            inverse = cho_solve(factor, np.eye(c.shape[1]), check_finite=False)
        except np.linalg.LinAlgError as exc:
            raise ValueError("local occupied columns lost metric rank") from exc
        if not np.all(np.isfinite(inverse)) or np.max(abs(
            metric @ inverse - np.eye(c.shape[1])
        )) > 1.0e-8:
            raise ValueError("local occupied metric inverse did not certify")
        sparse = csc_matrix(c)
        sparse.eliminate_zeros()
        return cls(sparse, atoms.copy(), metric, inverse)

    @property
    def shape(self) -> tuple[int, int]:
        return self.coefficients.shape

    @property
    def fill_fraction(self) -> float:
        n, occupied = self.shape
        return self.coefficients.nnz / (n * occupied)

    def metric_error(self, overlap: np.ndarray) -> float:
        s = np.asarray(overlap, dtype=float)
        if s.shape != (self.shape[0],) * 2:
            raise ValueError("overlap shape does not match the occupied subspace")
        gram = np.asarray(self.coefficients.T @ (s @ self.coefficients))
        return float(max(
            np.max(abs(gram - self.metric)),
            np.max(abs(gram @ self.inverse_metric - np.eye(self.shape[1]))),
        ))

    def density(self) -> np.ndarray:
        c = self.coefficients
        return np.asarray(2.0 * c @ (self.inverse_metric @ c.T))

    def weighted_density(self, fock: np.ndarray) -> np.ndarray:
        f = np.asarray(fock, dtype=float)
        if f.shape != (self.shape[0],) * 2:
            raise ValueError("Fock shape does not match the occupied subspace")
        c = self.coefficients
        projected = np.asarray(c.T @ (f @ c))
        inverse = self.inverse_metric
        return np.asarray(2.0 * c @ (inverse @ projected @ inverse) @ c.T)


@dataclass(frozen=True)
class LocalMetricOrthogonalization:
    subspace: OccupiedSubspace
    polynomial_order: int
    maximum_orthogonality_error: float
    metric_fill_fraction: float


def orthogonalize_local_occupied_subspace(
    coefficients: csc_matrix,
    overlap: np.ndarray | csc_matrix,
    orbital_atoms: np.ndarray,
    *,
    truncation_threshold: float = 1.0e-10,
    maximum_order: int = 6,
    orthogonality_tolerance: float = 1.0e-8,
) -> LocalMetricOrthogonalization:
    """Restore local occupied orthogonality with a sparse inverse-root series.

    The series acts on the sparse occupied metric near the identity. It never
    diagonalizes or inverts that metric. The returned factor is certified
    against the supplied overlap; energy and gradient still need an external
    accuracy check before accepting any support truncation.
    """
    c = csc_matrix(coefficients, dtype=float)
    s = overlap.tocsc() if issparse(overlap) else csc_matrix(overlap)
    atoms = np.asarray(orbital_atoms, dtype=np.intp)
    if (c.ndim != 2 or not 0 < c.shape[1] <= c.shape[0]
            or s.shape != (c.shape[0], c.shape[0])
            or atoms.shape != (c.shape[0],) or np.any(atoms < 0)
            or not np.all(np.isfinite(c.data)) or not np.all(np.isfinite(s.data))
            or truncation_threshold < 0.0 or maximum_order < 1
            or orthogonality_tolerance <= 0.0):
        raise ValueError("invalid local metric orthogonalization")

    occupied = c.shape[1]
    unit = identity(occupied, format="csc")
    metric_error = (c.T @ (s @ c)).tocsc() - unit

    def prune(matrix: csc_matrix) -> csc_matrix:
        matrix = matrix.tocsc()
        if truncation_threshold:
            matrix.data[abs(matrix.data) < truncation_threshold] = 0.0
            matrix.eliminate_zeros()
        return matrix

    metric_error = prune(metric_error)
    metric_fill = metric_error.nnz / (occupied * occupied)
    if float(np.max(np.asarray(abs(metric_error).sum(axis=1)))) >= 1.0:
        raise ValueError("local occupied metric is outside the inverse-root series")
    power = unit
    correction = unit
    binomial = 1.0
    for order in range(1, maximum_order + 1):
        power = prune(power @ metric_error)
        binomial *= (-0.5 - order + 1.0) / order
        correction = prune(correction + binomial * power)
        transformed = prune(c @ correction)
        residual = transformed.T @ (s @ transformed) - unit
        error = float(np.max(abs(residual.data), initial=0.0))
        if error <= orthogonality_tolerance:
            return LocalMetricOrthogonalization(
                OccupiedSubspace(transformed, atoms.copy()),
                order, error, metric_fill,
            )
    raise ValueError(
        "local inverse-root series did not certify occupied orthogonality"
    )


@dataclass(frozen=True)
class OccupiedSupportCertificate:
    subspace: OccupiedSubspace
    accepted: bool
    threshold: float
    orthogonality_error: float
    energy_error_hartree: float
    maximum_gradient_error_hartree_per_bohr: float
    rms_gradient_error_hartree_per_bohr: float
    density_error: float
    weighted_density_error: float
    largest_orthogonalization_cluster: int
    attempts: tuple[dict[str, float | str], ...]


@dataclass(frozen=True)
class OccupiedRelaxationResult:
    subspace: OccupiedSubspace
    energy_hartree: float
    maximum_stationarity_residual: float
    cycles: int
    converged: bool
    initial_fill_fraction: float
    final_fill_fraction: float


@dataclass(frozen=True)
class MetricOccupiedRelaxationResult:
    subspace: MetricOccupiedSubspace
    energy_hartree: float
    maximum_stationarity_residual: float
    cycles: int
    converged: bool
    initial_fill_fraction: float
    final_fill_fraction: float


def _components(adjacency: np.ndarray) -> list[np.ndarray]:
    pending = set(range(len(adjacency)))
    result = []
    while pending:
        root = pending.pop()
        component = {root}
        frontier = [root]
        while frontier:
            for neighbor in np.flatnonzero(adjacency[frontier.pop()]):
                if int(neighbor) in pending:
                    pending.remove(int(neighbor))
                    component.add(int(neighbor))
                    frontier.append(int(neighbor))
        result.append(np.asarray(sorted(component), dtype=np.intp))
    return result


def _pruned_columns(
    original: OccupiedSubspace,
    overlap: np.ndarray,
    threshold: float,
    maximum_cluster_orbitals: int,
    orthogonality_target: float = 1.0e-8,
    support_hint: OccupiedSubspace | None = None,
) -> tuple[OccupiedSubspace, int]:
    c = original.coefficients.toarray()
    atoms = original.orbital_atoms
    natoms = int(np.max(atoms)) + 1
    if support_hint is not None and support_hint.shape != original.shape:
        raise ValueError("occupied support hint has the wrong shape")
    hinted = (
        None if support_hint is None else support_hint.coefficients.toarray() != 0.0
    )
    for column in range(c.shape[1]):
        population = np.bincount(
            atoms, weights=c[:, column] ** 2, minlength=natoms,
        )
        keep = population >= threshold * float(np.sum(population))
        if hinted is not None:
            keep[np.unique(atoms[hinted[:, column]])] = True
        keep[int(np.argmax(population))] = True
        c[~keep[atoms], column] = 0.0
    mask = c != 0.0
    gram = c.T @ overlap @ c
    adjacency = abs(gram - np.diag(np.diag(gram))) > 1.0e-12
    components = _components(adjacency)
    largest = max(map(len, components))
    if largest > maximum_cluster_orbitals:
        raise ValueError("occupied orthogonalization cluster exceeds local budget")
    # Newton's symmetric orthogonalization correction is projected back onto
    # each column's atom support.  Unlike an unconstrained inverse square
    # root, this cannot turn every occupied column dense.  It may fail to
    # reach the metric tolerance; the caller then rejects the candidate.
    identity = np.eye(c.shape[1])
    for _ in range(8 * c.shape[1] + 1):
        for _ in range(20):
            error = c.T @ overlap @ c - identity
            if float(np.max(abs(error))) <= orthogonality_target:
                break
            c -= 0.5 * (c @ error) * mask
        error = c.T @ overlap @ c - identity
        if float(np.max(abs(error))) <= orthogonality_target:
            break
        left, right = np.unravel_index(np.argmax(abs(error)), error.shape)
        candidates = []
        for column, other in ((right, left), (left, right)):
            missing = original.coefficients[:, column].toarray().ravel()
            missing[mask[:, column]] = 0.0
            overlap_with_other = overlap @ c[:, other]
            contribution = np.bincount(
                atoms, weights=missing * overlap_with_other,
                minlength=natoms,
            )
            for atom in np.flatnonzero(contribution):
                candidates.append((abs(contribution[atom]), column, int(atom)))
        if not candidates:
            break
        _, column, atom = max(candidates)
        rows = np.flatnonzero(atoms == atom)
        mask[rows, column] = True
        c[rows, column] = original.coefficients[rows, column].toarray().ravel()
    return OccupiedSubspace.from_dense(c, atoms), largest


def certify_local_occupied_support(
    occupied_coefficients: np.ndarray,
    overlap: np.ndarray,
    orbital_atoms: np.ndarray,
    fock_and_energy: Callable[[np.ndarray], tuple[np.ndarray, float]],
    energy_gradient: Callable[[np.ndarray, np.ndarray], tuple[float, np.ndarray]],
    *,
    thresholds: tuple[float, ...] = (1.0e-8, 1.0e-10, 1.0e-12),
    maximum_cluster_orbitals: int = 32,
    maximum_orthogonality_error: float = 1.0e-8,
    maximum_energy_error_hartree: float = 1.0e-7,
    maximum_gradient_error_hartree_per_bohr: float = 1.0e-6,
) -> OccupiedSupportCertificate:
    """Keep the most compressed occupied state passing energy and gradient.

    The gradient callback must evaluate the complete model gradient for its
    density and energy-weighted density.  Candidates that cannot be checked
    are rejected, so the uncompressed subspace is returned as a fallback.
    """
    original = OccupiedSubspace.from_dense(occupied_coefficients, orbital_atoms)
    s = np.asarray(overlap, dtype=float)
    if (s.shape != (original.shape[0],) * 2 or not thresholds
            or any(not np.isfinite(x) or x <= 0.0 for x in thresholds)
            or maximum_cluster_orbitals < 1
            or maximum_orthogonality_error <= 0.0
            or maximum_energy_error_hartree < 0.0
            or maximum_gradient_error_hartree_per_bohr < 0.0):
        raise ValueError("invalid occupied support certificate")
    original_orthogonality = original.orthogonality_error(s)
    if original_orthogonality > maximum_orthogonality_error:
        raise ValueError("input occupied orbitals are not S-orthonormal")
    reference_density = original.density()
    reference_fock, _ = fock_and_energy(reference_density)
    reference_weighted = original.weighted_density(reference_fock)
    reference_energy, reference_gradient = energy_gradient(
        reference_density, reference_weighted,
    )
    reference_gradient = np.asarray(reference_gradient, dtype=float)
    if not np.isfinite(reference_energy) or not np.all(np.isfinite(reference_gradient)):
        raise ValueError("invalid reference energy or gradient")
    attempts: list[dict[str, float | str]] = []
    fallback = OccupiedSupportCertificate(
        original, False, 0.0, original_orthogonality, 0.0, 0.0, 0.0,
        0.0, 0.0, original.shape[1], (),
    )
    for threshold in thresholds:
        try:
            candidate, largest = _pruned_columns(
                original, s, threshold, maximum_cluster_orbitals,
                orthogonality_target=maximum_orthogonality_error,
            )
        except ValueError as exc:
            attempts.append({"threshold": threshold, "reason": str(exc)})
            continue
        orthogonality = candidate.orthogonality_error(s)
        attempt: dict[str, float | str] = {
            "threshold": threshold,
            "fill_fraction": candidate.fill_fraction,
            "orthogonality_error": orthogonality,
        }
        if candidate.coefficients.nnz >= original.coefficients.nnz:
            attempt["reason"] = "no compression"
            attempts.append(attempt)
            continue
        if orthogonality > maximum_orthogonality_error:
            attempt["reason"] = "orthogonality"
            attempts.append(attempt)
            continue
        density = candidate.density()
        fock, _ = fock_and_energy(density)
        weighted = candidate.weighted_density(fock)
        energy, gradient = energy_gradient(density, weighted)
        gradient = np.asarray(gradient, dtype=float)
        if gradient.shape != reference_gradient.shape:
            raise ValueError("candidate and reference gradient shapes differ")
        difference = gradient - reference_gradient
        energy_error = abs(float(energy) - float(reference_energy))
        gradient_error = float(np.max(abs(difference)))
        attempt["energy_error_hartree"] = energy_error
        attempt["gradient_error_hartree_per_bohr"] = gradient_error
        if (not np.isfinite(energy_error) or not np.isfinite(gradient_error)
                or energy_error > maximum_energy_error_hartree
                or gradient_error > maximum_gradient_error_hartree_per_bohr):
            attempt["reason"] = "energy or gradient"
            attempts.append(attempt)
            continue
        attempt["reason"] = "accepted"
        attempts.append(attempt)
        return OccupiedSupportCertificate(
            candidate, True, threshold, orthogonality, energy_error,
            gradient_error, float(np.sqrt(np.mean(difference**2))),
            float(np.max(abs(density - reference_density))),
            float(np.max(abs(weighted - reference_weighted))), largest,
            tuple(attempts),
        )
    return OccupiedSupportCertificate(
        fallback.subspace, fallback.accepted, fallback.threshold,
        fallback.orthogonality_error, fallback.energy_error_hartree,
        fallback.maximum_gradient_error_hartree_per_bohr,
        fallback.rms_gradient_error_hartree_per_bohr,
        fallback.density_error, fallback.weighted_density_error,
        fallback.largest_orthogonalization_cluster, tuple(attempts),
    )


def relax_local_occupied_subspace(
    initial: OccupiedSubspace,
    overlap: np.ndarray,
    atom_centers_angstrom: np.ndarray,
    fock_and_energy: Callable[[np.ndarray], tuple[np.ndarray, float]],
    *,
    radius_angstrom: float,
    support_threshold: float = 1.0e-12,
    maximum_cycles: int = 80,
    residual_tolerance: float = 1.0e-7,
) -> OccupiedRelaxationResult:
    """Experimental occupied-only local tangent relaxation.

    Each occupied orbital receives a metric-preconditioned Fock residual only
    on nearby atom blocks.  A line search accepts full-functional energy
    descent only after support-restricted reorthogonalization.  This avoids a
    virtual-orbital basis and full Fock eigensolve.  A finite radius can leave
    an unrelaxed global residual; callers must inspect ``converged``.
    """
    s = np.asarray(overlap, dtype=float)
    centers = np.asarray(atom_centers_angstrom, dtype=float)
    n, occupied = initial.shape
    if (s.shape != (n, n) or centers.ndim != 2 or centers.shape[1] != 3
            or len(centers) <= int(np.max(initial.orbital_atoms))
            or radius_angstrom < 0.0 or support_threshold <= 0.0
            or maximum_cycles < 1 or residual_tolerance <= 0.0
            or initial.orthogonality_error(s) > 1.0e-8):
        raise ValueError("invalid local occupied relaxation model")
    state = initial
    fock, energy = fock_and_energy(state.density())
    energy = float(energy)
    accepted = 0
    residual_max = np.inf
    atom_distance = np.linalg.norm(
        centers[:, None, :] - centers[None, :, :], axis=2,
    )
    orbital_atoms = initial.orbital_atoms
    preconditioners: dict[bytes, tuple[np.ndarray, tuple[np.ndarray, bool]]] = {}
    for _ in range(maximum_cycles):
        c = state.coefficients.toarray()
        f = np.asarray(fock, dtype=float)
        projected = c.T @ f @ c
        residual = f @ c - s @ c @ projected
        residual_max = float(np.max(abs(residual)))
        if residual_max <= residual_tolerance:
            break
        direction = np.zeros_like(c)
        for column in range(occupied):
            active_atoms = np.unique(orbital_atoms[np.flatnonzero(c[:, column])])
            allowed_atoms = np.any(
                atom_distance[:, active_atoms] <= radius_angstrom, axis=1,
            )
            rows = np.flatnonzero(allowed_atoms[orbital_atoms])
            key = rows.tobytes()
            if key not in preconditioners:
                preconditioners[key] = (
                    rows, cho_factor(s[np.ix_(rows, rows)], check_finite=False),
                )
            selected, factor = preconditioners[key]
            direction[selected, column] = cho_solve(
                factor, -residual[selected, column], check_finite=False,
            )
        step_accepted = False
        for step in (0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625,
                     0.0078125):
            trial = c + step * direction
            gram = trial.T @ s @ trial
            eigenvalues, eigenvectors = eigh(0.5 * (gram + gram.T))
            if eigenvalues[0] <= 1.0e-10:
                continue
            inverse_root = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
            trial = trial @ inverse_root
            trial_state = OccupiedSubspace.from_dense(trial, initial.orbital_atoms)
            try:
                local, _ = _pruned_columns(
                    trial_state, s, support_threshold, occupied,
                    orthogonality_target=1.0e-10,
                    support_hint=state,
                )
            except ValueError:
                continue
            if local.orthogonality_error(s) > 1.0e-8:
                continue
            trial_fock, trial_energy = fock_and_energy(local.density())
            if float(trial_energy) < energy - 1.0e-15:
                state, fock, energy = local, trial_fock, float(trial_energy)
                accepted += 1
                step_accepted = True
                break
        if not step_accepted:
            break
    c = state.coefficients.toarray()
    projected = c.T @ np.asarray(fock) @ c
    residual_max = float(np.max(abs(
        np.asarray(fock) @ c - s @ c @ projected
    )))
    return OccupiedRelaxationResult(
        state, energy, residual_max, accepted,
        residual_max <= residual_tolerance,
        initial.fill_fraction, state.fill_fraction,
    )


def relax_metric_occupied_subspace(
    initial: MetricOccupiedSubspace,
    overlap: np.ndarray,
    atom_centers_angstrom: np.ndarray,
    fock_and_energy: Callable[[np.ndarray], tuple[np.ndarray, float]],
    *,
    radius_angstrom: float,
    support_threshold: float = 1.0e-10,
    maximum_cycles: int = 80,
    residual_tolerance: float = 1.0e-7,
) -> MetricOccupiedRelaxationResult:
    """Optimize local occupied columns with an exact nonidentity metric.

    This experimental route retains a dense occupied metric and dense GFN2
    callback. It never globally whitens the coefficient columns; trial steps
    change only atom supports near currently occupied atoms. Convergence is
    judged by the full AO occupied-subspace residual.
    """
    s = np.asarray(overlap, dtype=float)
    centers = np.asarray(atom_centers_angstrom, dtype=float)
    n, occupied = initial.shape
    if (s.shape != (n, n) or centers.ndim != 2 or centers.shape[1] != 3
            or len(centers) <= int(np.max(initial.orbital_atoms))
            or radius_angstrom < 0.0 or support_threshold <= 0.0
            or maximum_cycles < 1 or residual_tolerance <= 0.0
            or initial.metric_error(s) > 1.0e-8):
        raise ValueError("invalid metric occupied relaxation model")
    state = initial
    fock, energy = fock_and_energy(state.density())
    energy = float(energy)
    accepted = 0
    atom_distance = np.linalg.norm(
        centers[:, None, :] - centers[None, :, :], axis=2,
    )
    orbital_atoms = initial.orbital_atoms
    preconditioners: dict[bytes, tuple[np.ndarray, tuple[np.ndarray, bool]]] = {}
    for _ in range(maximum_cycles):
        c = state.coefficients.toarray()
        f = np.asarray(fock, dtype=float)
        projected = c.T @ f @ c
        residual = f @ c - s @ c @ (state.inverse_metric @ projected)
        residual_max = float(np.max(abs(residual)))
        if residual_max <= residual_tolerance:
            break
        direction = np.zeros_like(c)
        for column in range(occupied):
            active_atoms = np.unique(orbital_atoms[np.flatnonzero(c[:, column])])
            allowed_atoms = np.any(
                atom_distance[:, active_atoms] <= radius_angstrom, axis=1,
            )
            rows = np.flatnonzero(allowed_atoms[orbital_atoms])
            key = rows.tobytes()
            if key not in preconditioners:
                preconditioners[key] = (
                    rows, cho_factor(s[np.ix_(rows, rows)], check_finite=False),
                )
            selected, factor = preconditioners[key]
            direction[selected, column] = cho_solve(
                factor, -residual[selected, column], check_finite=False,
            )
        step_accepted = False
        for step in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625,
                     0.0078125):
            trial = c + step * direction
            for column in range(occupied):
                population = np.bincount(
                    orbital_atoms, weights=trial[:, column] ** 2,
                    minlength=len(centers),
                )
                keep = population >= support_threshold * float(np.sum(population))
                keep[int(np.argmax(population))] = True
                trial[~keep[orbital_atoms], column] = 0.0
            try:
                local = MetricOccupiedSubspace.from_dense(
                    trial, s, orbital_atoms,
                )
            except ValueError:
                continue
            trial_fock, trial_energy = fock_and_energy(local.density())
            if float(trial_energy) < energy - 1.0e-15:
                state, fock, energy = local, trial_fock, float(trial_energy)
                accepted += 1
                step_accepted = True
                break
        if not step_accepted:
            break
    c = state.coefficients.toarray()
    projected = c.T @ np.asarray(fock) @ c
    residual_max = float(np.max(abs(
        np.asarray(fock) @ c
        - s @ c @ (state.inverse_metric @ projected)
    )))
    return MetricOccupiedRelaxationResult(
        state, energy, residual_max, accepted,
        residual_max <= residual_tolerance,
        initial.fill_fraction, state.fill_fraction,
    )


def relax_metric_occupied_subspace_lbfgs(
    initial: MetricOccupiedSubspace,
    overlap: np.ndarray,
    atom_centers_angstrom: np.ndarray,
    fock_and_energy: Callable[[np.ndarray], tuple[np.ndarray, float]],
    *,
    radius_angstrom: float,
    support_threshold: float = 1.0e-10,
    maximum_cycles: int = 40,
    residual_tolerance: float = 1.0e-6,
) -> MetricOccupiedRelaxationResult:
    """Optimize an atom-local occupied support with the exact metric gradient.

    The support is fixed during each L-BFGS solve. Whole-atom pruning follows
    the solve and is accepted only when it does not increase the full energy.
    Dense Fock, overlap and occupied-metric operations remain in this opt-in
    experiment; convergence uses the *full* AO residual after pruning.
    """
    s = np.asarray(overlap, dtype=float)
    centers = np.asarray(atom_centers_angstrom, dtype=float)
    n, occupied = initial.shape
    if (s.shape != (n, n) or centers.ndim != 2 or centers.shape[1] != 3
            or len(centers) <= int(np.max(initial.orbital_atoms))
            or radius_angstrom < 0.0 or support_threshold <= 0.0
            or maximum_cycles < 1 or residual_tolerance <= 0.0
            or initial.metric_error(s) > 1.0e-8):
        raise ValueError("invalid metric occupied relaxation model")
    atoms = initial.orbital_atoms
    c0 = initial.coefficients.toarray()
    distances = np.linalg.norm(
        centers[:, None, :] - centers[None, :, :], axis=2,
    )
    mask = np.zeros_like(c0, dtype=bool)
    for column in range(occupied):
        active = np.unique(atoms[np.flatnonzero(c0[:, column])])
        allowed = np.any(distances[:, active] <= radius_angstrom, axis=1)
        mask[:, column] = allowed[atoms]

    def objective(values: np.ndarray):
        coefficients = np.zeros_like(c0)
        coefficients[mask] = values
        try:
            state = MetricOccupiedSubspace.from_dense(coefficients, s, atoms)
        except ValueError:
            return 1.0e20, np.zeros_like(values)
        fock, energy = fock_and_energy(state.density())
        c = state.coefficients.toarray()
        projected = c.T @ fock @ c
        residual = fock @ c - s @ c @ (state.inverse_metric @ projected)
        derivative = 4.0 * residual @ state.inverse_metric
        return float(energy), np.asarray(derivative[mask], dtype=float)

    optimized = minimize(
        objective, c0[mask], jac=True, method="L-BFGS-B",
        options={"maxiter": maximum_cycles, "maxls": 20,
                 "gtol": residual_tolerance, "ftol": 1.0e-14},
    )
    coefficients = np.zeros_like(c0)
    coefficients[mask] = optimized.x
    unpruned = MetricOccupiedSubspace.from_dense(coefficients, s, atoms)
    fock, energy = fock_and_energy(unpruned.density())
    pruned = coefficients.copy()
    for column in range(occupied):
        population = np.bincount(
            atoms, weights=pruned[:, column] ** 2, minlength=len(centers),
        )
        keep = population >= support_threshold * float(np.sum(population))
        keep[int(np.argmax(population))] = True
        pruned[~keep[atoms], column] = 0.0
    try:
        candidate = MetricOccupiedSubspace.from_dense(pruned, s, atoms)
        candidate_fock, candidate_energy = fock_and_energy(candidate.density())
        if float(candidate_energy) <= float(energy) + 1.0e-8:
            state, fock, energy = candidate, candidate_fock, candidate_energy
        else:
            state = unpruned
    except ValueError:
        state = unpruned
    c = state.coefficients.toarray()
    projected = c.T @ fock @ c
    residual = fock @ c - s @ c @ (state.inverse_metric @ projected)
    maximum_residual = float(np.max(abs(residual)))
    return MetricOccupiedRelaxationResult(
        state, float(energy), maximum_residual, int(optimized.nfev),
        maximum_residual <= residual_tolerance,
        initial.fill_fraction, state.fill_fraction,
    )
