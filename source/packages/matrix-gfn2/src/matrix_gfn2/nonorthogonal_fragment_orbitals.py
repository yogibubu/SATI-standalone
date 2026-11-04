"""Nonorthogonal occupied fragment subspace without global MO whitening.

Rank selection is confined to disjoint real-AO fragment blocks. The exact
occupied projector uses solves with the global fragment-overlap metric; no
inverse is formed explicitly. This is an opt-in starting subspace, not yet a
replacement for the converged GFN2 SCF or a linear-scaling metric solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.linalg import (
    cho_factor, cho_solve, cholesky, qr, solve, solve_triangular,
)
from scipy.optimize import minimize
from scipy.sparse import csc_matrix, csr_matrix, eye as sparse_eye
from scipy.sparse.linalg import cg, LinearOperator


@dataclass(frozen=True)
class NonorthogonalFragmentSubspace:
    coefficients: csc_matrix
    overlap_metric: np.ndarray
    density: np.ndarray
    occupied: int
    fragment_labels: np.ndarray
    local_candidate_counts: tuple[int, ...]
    local_selected_counts: tuple[int, ...]
    metric_condition: float
    projector_error: float

    def project_fock(self, fock: np.ndarray) -> np.ndarray:
        matrix = np.asarray(fock, dtype=float)
        if matrix.shape != (self.coefficients.shape[0],) * 2:
            raise ValueError("incompatible Fock matrix")
        return np.asarray(self.coefficients.T @ (matrix @ self.coefficients))

    def ao_stationarity_residual(self, fock: np.ndarray, overlap: np.ndarray) -> np.ndarray:
        """Return the occupied-subspace residual in the original AO basis."""
        matrix = np.asarray(fock, dtype=float)
        s = np.asarray(overlap, dtype=float)
        c = self.coefficients.toarray()
        projected = self.project_fock(matrix)
        connection = solve(self.overlap_metric, projected, assume_a="pos")
        return matrix @ c - s @ c @ connection


@dataclass(frozen=True)
class NonorthogonalRelaxationResult:
    coefficients: np.ndarray
    density: np.ndarray
    energy: float
    maximum_ao_residual: float
    stage_energies: tuple[float, ...]
    stage_residuals: tuple[float, ...]
    stage_coefficient_nnz: tuple[int, ...]
    stage_callback_calls: tuple[int, ...]
    stage_seconds: tuple[float, ...]
    callback_calls: int
    converged: bool
    metric_inverse_nnz: int = 0
    metric_krylov_fallbacks: int = 0


@dataclass(frozen=True)
class FragmentVirtualBasis:
    coefficients: csc_matrix
    fragment_labels: np.ndarray


def _certified_local_metric_inverse(
    metric: np.ndarray, labels: np.ndarray,
    *, tolerance: float = 1.0e-12,
) -> tuple[np.ndarray, int, bool]:
    """Invert a fragment metric by local block-preconditioned iteration.

    The exact dense metric certifies the final inverse. No global factorization
    is used; Krylov refinement is a conservative fallback when the local
    Neumann expansion does not converge.
    """
    size = len(metric)
    blocks = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        block = np.linalg.inv(metric[np.ix_(indices, indices)])
        blocks.append((indices, block))
    rows = np.concatenate([np.repeat(indices, len(indices)) for indices, _ in blocks])
    columns = np.concatenate([np.tile(indices, len(indices)) for indices, _ in blocks])
    values = np.concatenate([block.ravel() for _, block in blocks])
    preconditioner = csr_matrix((values, (rows, columns)), shape=(size, size))
    identity = sparse_eye(size, format="csr")
    for threshold in (1.0e-10, 1.0e-12, 0.0):
        sparse_metric = csr_matrix(np.where(abs(metric) >= threshold, metric, 0.0))
        correction = identity - preconditioner @ sparse_metric
        term = preconditioner.copy()
        inverse = term.copy()
        for _ in range(32):
            term = correction @ term
            if term.nnz:
                term.data[abs(term.data) < max(threshold * 0.01, 1.0e-15)] = 0.0
                term.eliminate_zeros()
            inverse = inverse + term
            candidate = inverse.toarray()
            if np.max(abs(metric @ candidate - np.eye(size))) <= tolerance:
                return candidate, inverse.nnz, False
    operator = LinearOperator((size, size), matvec=lambda vector: metric @ vector)
    precondition = LinearOperator(
        (size, size), matvec=lambda vector: preconditioner @ vector,
    )
    inverse = np.empty_like(metric)
    for column in range(size):
        right = np.zeros(size)
        right[column] = 1.0
        solution, status = cg(
            operator, right, M=precondition, rtol=1.0e-13,
            atol=0.0, maxiter=200,
        )
        if status != 0:
            raise RuntimeError("fragment metric Krylov inverse did not converge")
        inverse[:, column] = solution
    if np.max(abs(metric @ inverse - np.eye(size))) > tolerance:
        raise RuntimeError("fragment metric inverse failed certification")
    return inverse, int(np.count_nonzero(inverse)), True


def fragment_virtual_basis(
    states: tuple[object, ...], seed: NonorthogonalFragmentSubspace,
    overlap: np.ndarray,
) -> FragmentVirtualBasis:
    """Complete each fragment's occupied columns with local virtual MOs.

    No global orbital orthogonalization is used. The virtual columns are
    S-orthonormal within each disjoint real-AO fragment only.
    """
    s = np.asarray(overlap, dtype=float)
    n = seed.coefficients.shape[0]
    if s.shape != (n, n) or len(states) != len(seed.local_selected_counts):
        raise ValueError("incompatible fragment virtual basis")
    occupied = seed.coefficients.toarray()
    rows_parts = []
    columns_parts = []
    values_parts = []
    labels = []
    offset = 0
    for fragment, state in enumerate(states):
        rows = np.asarray(state.global_orbitals, dtype=np.intp)
        columns = np.flatnonzero(seed.fragment_labels == fragment)
        local = occupied[np.ix_(rows, columns)]
        factor = cholesky(s[np.ix_(rows, rows)], lower=True)
        orthogonal, upper = qr(factor.T @ local, mode="full")
        if len(columns) and np.min(abs(np.diag(upper))) < 1.0e-10:
            raise ValueError("rank-deficient occupied fragment basis")
        virtual = solve_triangular(
            factor.T, orthogonal[:, len(columns):], lower=False,
        )
        for local_column in range(virtual.shape[1]):
            values = virtual[:, local_column]
            nonzero = np.flatnonzero(values)
            rows_parts.append(rows[nonzero])
            columns_parts.append(np.full(len(nonzero), offset, dtype=np.intp))
            values_parts.append(values[nonzero])
            labels.append(fragment)
            offset += 1
    if offset != n - seed.occupied:
        raise RuntimeError("fragment virtual basis has wrong dimension")
    coefficients = csc_matrix((
        np.concatenate(values_parts),
        (np.concatenate(rows_parts), np.concatenate(columns_parts)),
    ), shape=(n, offset))
    return FragmentVirtualBasis(coefficients, np.asarray(labels, dtype=np.intp))


def assemble_nonorthogonal_fragment_subspace(
    states: tuple[object, ...], overlap: np.ndarray, electrons: int,
    *, local_rank_tolerance: float = 1.0e-8,
) -> NonorthogonalFragmentSubspace:
    """Select occupied fragment columns locally and solve the overlap metric."""
    s = np.asarray(overlap, dtype=float)
    size = len(s)
    occupied = electrons // 2
    if (s.shape != (size, size) or electrons % 2 or not 0 < occupied < size
            or not states or local_rank_tolerance <= 0.0
            or not np.all(np.isfinite(s))):
        raise ValueError("invalid nonorthogonal fragment model")
    covered = np.zeros(size, dtype=bool)
    candidates = []
    local_counts = []
    for fragment, state in enumerate(states):
        rows = np.asarray(state.global_orbitals, dtype=np.intp)
        block = np.asarray(state.coefficients_on_real_atoms, dtype=float)
        occupation = (
            np.asarray(state.alpha_occupations, dtype=float)
            + np.asarray(state.beta_occupations, dtype=float)
        )
        if (block.shape != (len(rows), len(occupation))
                or len(rows) == 0 or np.any(rows < 0)
                or np.any(rows >= size) or np.any(covered[rows])
                or len(np.unique(rows)) != len(rows)
                or not np.all(np.isfinite(block))):
            raise ValueError("invalid disjoint fragment columns")
        covered[rows] = True
        occupied_columns = np.flatnonzero(occupation > 1.0)
        local_counts.append(len(occupied_columns))
        if not len(occupied_columns):
            continue
        local_overlap = s[np.ix_(rows, rows)]
        factor = cholesky(local_overlap, lower=True)
        whitened = factor.T @ block[:, occupied_columns]
        _, upper, pivots = qr(whitened, mode="economic", pivoting=True)
        diagonal = abs(np.diag(upper))
        minimum = local_rank_tolerance * max(float(diagonal[0]), 1.0e-30)
        for position, score in enumerate(diagonal):
            if score <= minimum:
                break
            column = occupied_columns[pivots[position]]
            candidates.append((float(score), fragment, rows, block[:, column]))
    if not np.all(covered) or len(candidates) < occupied:
        raise ValueError("fragment occupied span has insufficient local rank")
    candidates.sort(key=lambda item: -item[0])
    selected = candidates[:occupied]
    row_parts = []
    column_parts = []
    value_parts = []
    labels = []
    for column, (_, fragment, rows, values) in enumerate(selected):
        nonzero = np.flatnonzero(values)
        row_parts.append(rows[nonzero])
        column_parts.append(np.full(len(nonzero), column, dtype=np.intp))
        value_parts.append(values[nonzero])
        labels.append(fragment)
    coefficients = csc_matrix((
        np.concatenate(value_parts),
        (np.concatenate(row_parts), np.concatenate(column_parts)),
    ), shape=(size, occupied))
    c = coefficients.toarray()
    metric = c.T @ s @ c
    metric = 0.5 * (metric + metric.T)
    condition = float(np.linalg.cond(metric))
    if not np.isfinite(condition) or condition > 1.0e12:
        raise ValueError("fragment overlap metric is rank deficient")
    dual = solve(metric, c.T, assume_a="pos")
    density = 2.0 * c @ dual
    density = 0.5 * (density + density.T)
    projector_error = float(np.max(abs(density @ s @ density - 2.0 * density)))
    if projector_error > 1.0e-7:
        raise RuntimeError("nonorthogonal occupied projector is inaccurate")
    return NonorthogonalFragmentSubspace(
        coefficients, metric, density, occupied,
        np.asarray(labels, dtype=np.intp), tuple(local_counts),
        tuple(int(np.count_nonzero(np.asarray(labels) == fragment))
              for fragment in range(len(states))),
        condition, projector_error,
    )


def relax_nonorthogonal_fragment_subspace(
    seed: NonorthogonalFragmentSubspace,
    overlap: np.ndarray,
    ao_fragments: np.ndarray,
    fragment_centers_angstrom: np.ndarray,
    fock_and_energy,
    *,
    fock_action_and_energy=None,
    metric_solver: str = "cholesky",
    radii_angstrom: tuple[float, ...] | None = None,
    maximum_iterations_per_radius: int | None = None,
    residual_tolerance: float = 1.0e-6,
    fragment_distances_angstrom: np.ndarray | None = None,
    adaptive_support: bool = False,
    ao_atoms: np.ndarray | None = None,
    maximum_adaptive_stages: int = 24,
    atoms_per_orbital_per_stage: int = 6,
    virtual_basis: FragmentVirtualBasis | None = None,
    orbital_coordinates: str = "ao",
) -> NonorthogonalRelaxationResult:
    """Variationally optimize local nonorthogonal occupied columns.

    The support grows by fragment radius and optionally by the largest
    missing atom-block residuals. The occupied projector always uses the
    exact metric. The full AO residual remains the final convergence gate.
    """
    if radii_angstrom is None:
        radii_angstrom = ((3.0,) if adaptive_support
                            else (0.0, 3.0, 6.0, np.inf))
    if maximum_iterations_per_radius is None:
        maximum_iterations_per_radius = 7 if adaptive_support else 40
    s = np.asarray(overlap, dtype=float)
    centers = np.asarray(fragment_centers_angstrom, dtype=float)
    ao_labels = np.asarray(ao_fragments, dtype=np.intp)
    c = seed.coefficients.toarray()
    n, occupied = c.shape
    if (s.shape != (n, n) or ao_labels.shape != (n,)
            or centers.ndim != 2 or centers.shape[1] != 3
            or np.any(ao_labels < 0) or np.any(ao_labels >= len(centers))
            or np.any(seed.fragment_labels < 0)
            or np.any(seed.fragment_labels >= len(centers))
            or not radii_angstrom or radii_angstrom[0] < 0.0
            or any(left >= right for left, right in zip(radii_angstrom, radii_angstrom[1:]))
            or maximum_iterations_per_radius < 1 or residual_tolerance <= 0.0
            or maximum_adaptive_stages < 1 or atoms_per_orbital_per_stage < 1
            or orbital_coordinates not in ("ao", "fragment-ov")
            or metric_solver not in ("cholesky", "local-iterative")
            or (fock_action_and_energy is not None
                and orbital_coordinates != "ao")):
        raise ValueError("invalid nonorthogonal relaxation model")
    if orbital_coordinates == "fragment-ov":
        if (virtual_basis is None
                or virtual_basis.coefficients.shape != (n, n - occupied)):
            raise ValueError("occupied-virtual update requires local virtual MOs")
        virtual = virtual_basis.coefficients.toarray()
    else:
        virtual = None
    atom_labels = (np.arange(n, dtype=np.intp) if ao_atoms is None
                   else np.asarray(ao_atoms, dtype=np.intp))
    if atom_labels.shape != (n,) or np.any(atom_labels < 0):
        raise ValueError("invalid AO atom labels")
    if fragment_distances_angstrom is None:
        distance = np.linalg.norm(
            centers[ao_labels, None, :] - centers[None, seed.fragment_labels, :],
            axis=-1,
        )
    else:
        fragment_distance = np.asarray(fragment_distances_angstrom, dtype=float)
        if (fragment_distance.shape != (len(centers), len(centers))
                or np.any(fragment_distance < 0.0)
                or not np.all(np.isfinite(fragment_distance))):
            raise ValueError("invalid fragment distance matrix")
        distance = fragment_distance[
            ao_labels[:, None], seed.fragment_labels[None, :]
        ]
    callback_calls = 0
    metric_inverse_nnz = 0
    metric_krylov_fallbacks = 0

    def evaluate(trial):
        nonlocal callback_calls, metric_inverse_nnz, metric_krylov_fallbacks
        overlap_columns = s @ trial
        metric = trial.T @ overlap_columns
        metric = 0.5 * (metric + metric.T)
        if metric_solver == "cholesky":
            factor = cho_factor(metric, lower=True, check_finite=False)
            dual = cho_solve(factor, trial.T, check_finite=False)
            inverse = None
        else:
            inverse, nnz, fallback = _certified_local_metric_inverse(
                metric, seed.fragment_labels,
            )
            metric_inverse_nnz = max(metric_inverse_nnz, nnz)
            metric_krylov_fallbacks += int(fallback)
            dual = inverse @ trial.T
        density = 2.0 * trial @ dual
        density = 0.5 * (density + density.T)
        if fock_action_and_energy is None:
            fock, energy = fock_and_energy(density)
            fock = np.asarray(fock, dtype=float)
            fock_columns = fock @ trial
        else:
            fock_columns, energy = fock_action_and_energy(density, trial)
            fock_columns = np.asarray(fock_columns, dtype=float)
            if fock_columns.shape != trial.shape:
                raise ValueError("native Fock action has incompatible dimensions")
            fock = None
        callback_calls += 1
        connection = dual @ fock_columns
        residual = fock_columns - overlap_columns @ connection
        gradient = (
            4.0 * cho_solve(factor, residual.T, check_finite=False).T
            if inverse is None else 4.0 * residual @ inverse
        )
        return float(energy), density, residual, gradient, fock

    energy, density, residual, gradient, fock = evaluate(c)
    stages_energy = []
    stages_residual = []
    stages_nnz = []
    stages_calls = []
    stages_seconds = []
    def optimize_support(allowed):
        nonlocal c, density, residual, gradient, fock, energy
        started = perf_counter()
        calls_before = callback_calls
        if np.any(c[~allowed] != 0.0):
            raise RuntimeError("fragment support shrank during relaxation")
        last_evaluation = None
        if orbital_coordinates == "fragment-ov":
            base = c.copy()
            # The local fragment virtuals span the complement of the seed
            # occupied space. Projecting their updates through the AO support
            # mask retains strict locality at each stage.
            active = (abs(virtual).T @ allowed.astype(np.int8)) > 0
            local_occupied_energy = np.sum(base * (fock @ base), axis=0) / np.sum(
                base * (s @ base), axis=0,
            )
            local_virtual_energy = np.sum(virtual * (fock @ virtual), axis=0)
            gap = np.maximum(
                abs(local_virtual_energy[:, None] - local_occupied_energy[None, :]),
                0.10,
            )
            scale = 1.0 / np.sqrt(gap)
            initial = np.zeros(int(np.count_nonzero(active)), dtype=float)
        else:
            initial = c[allowed].copy()
        def coefficients(values):
            if orbital_coordinates == "fragment-ov":
                rotations = np.zeros((virtual.shape[1], occupied))
                rotations[active] = values * scale[active]
                trial = base + virtual @ rotations * allowed
            else:
                trial = np.zeros_like(c)
                trial[allowed] = values
            return trial

        def objective(values):
            nonlocal last_evaluation
            trial = coefficients(values)
            try:
                result = evaluate(trial)
            except (ValueError, np.linalg.LinAlgError):
                return 1.0e30, np.zeros_like(values)
            last_evaluation = (values.copy(), result)
            if orbital_coordinates == "fragment-ov":
                derivative = virtual.T @ (result[3] * allowed)
                return result[0], (derivative * scale)[active]
            return result[0], result[3][allowed]

        optimized = minimize(
            objective, initial, method="L-BFGS-B", jac=True,
            options={"maxiter": maximum_iterations_per_radius,
                     "gtol": residual_tolerance * 0.1,
                     "ftol": 1.0e-14, "maxls": 20},
        )
        trial = coefficients(optimized.x)
        if (last_evaluation is not None
                and np.array_equal(optimized.x, last_evaluation[0])):
            trial_energy, trial_density, trial_residual, trial_gradient, trial_fock = last_evaluation[1]
        else:
            trial_energy, trial_density, trial_residual, trial_gradient, trial_fock = (
                evaluate(trial)
            )
        if trial_energy <= energy + 1.0e-10:
            c, density, residual, gradient, fock, energy = (
                trial, trial_density, trial_residual, trial_gradient,
                trial_fock, trial_energy,
            )
        stages_energy.append(energy)
        stages_residual.append(float(np.max(abs(residual))))
        stages_nnz.append(int(np.count_nonzero(c)))
        stages_calls.append(callback_calls - calls_before)
        stages_seconds.append(perf_counter() - started)
    atom_rows = tuple(np.flatnonzero(atom_labels == atom)
                      for atom in range(int(atom_labels.max()) + 1))
    def expand_support(allowed, maximum_atoms):
        additions = 0
        for column in range(occupied):
            outside = ~allowed[:, column]
            if not np.any(outside):
                continue
            scores = np.zeros(len(atom_rows), dtype=float)
            np.maximum.at(
                scores, atom_labels[outside], abs(residual[outside, column]),
            )
            strongest = np.argsort(scores)[::-1][:maximum_atoms]
            for atom in strongest:
                if scores[atom] <= residual_tolerance:
                    continue
                rows = atom_rows[atom]
                additions += int(np.count_nonzero(outside[rows]))
                allowed[rows, column] = True
        return additions
    allowed = np.zeros_like(c, dtype=bool)
    for radius in radii_angstrom:
        allowed |= distance <= radius + 1.0e-12
        optimize_support(allowed)
        if stages_residual[-1] <= residual_tolerance:
            break
    if adaptive_support and stages_residual[-1] > residual_tolerance:
        for _ in range(maximum_adaptive_stages):
            additions = expand_support(allowed, atoms_per_orbital_per_stage)
            if not additions and np.max(abs(residual)) <= residual_tolerance:
                break
            # With no missing support, keep iterating the same variational
            # problem: an optimizer iteration cap is not a convergence test.
            optimize_support(allowed)
            if stages_residual[-1] <= residual_tolerance:
                break
    maximum_residual = float(np.max(abs(residual)))
    return NonorthogonalRelaxationResult(
        c, density, energy, maximum_residual,
        tuple(stages_energy), tuple(stages_residual), tuple(stages_nnz),
        tuple(stages_calls), tuple(stages_seconds), callback_calls,
        bool(maximum_residual <= residual_tolerance),
        metric_inverse_nnz, metric_krylov_fallbacks,
    )
