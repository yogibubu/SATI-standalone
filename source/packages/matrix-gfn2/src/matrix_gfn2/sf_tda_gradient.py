"""Frozen-orbital analytic derivatives for resident spin-flip TDA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .electronic_state import GFN2OrbitalState
from .sf_tda import (
    AtomicTransitionChargeResponse,
    GRIMME_XTB_SF_STDA,
    SFTDAResult,
    SFTDASelectionPolicy,
    SelectedSpinFlipSpace,
    SpinFlipExcitationSpace,
    grimme_stda_atomic_hardnesses,
    select_grimme_spin_flip_configurations,
)


@dataclass(frozen=True)
class FrozenOrbitalDerivativeInputs:
    """First derivatives of the AO metric and frozen-density Fock matrix.

    Both arrays use ``(natom, 3, nao, nao)`` ordering and derivatives with
    respect to nuclear coordinates in bohr.  Orbital relaxation and the SCC
    density response are deliberately excluded from this contract.
    """

    overlap_derivative: np.ndarray
    fock_derivative: np.ndarray

    def validated(self, state: GFN2OrbitalState) -> tuple[np.ndarray, np.ndarray]:
        expected = (len(state.atomic_numbers), 3, state.nao, state.nao)
        overlap = np.asarray(self.overlap_derivative, dtype=float)
        fock = np.asarray(self.fock_derivative, dtype=float)
        if overlap.shape != expected or fock.shape != expected:
            raise ValueError("frozen-orbital AO derivatives have incompatible dimensions")
        if not np.all(np.isfinite(overlap)) or not np.all(np.isfinite(fock)):
            raise ValueError("frozen-orbital AO derivatives must be finite")
        if not np.allclose(overlap, overlap.swapaxes(-1, -2), atol=2.0e-12, rtol=0.0):
            raise ValueError("AO overlap derivatives must be symmetric")
        if not np.allclose(fock, fock.swapaxes(-1, -2), atol=2.0e-12, rtol=0.0):
            raise ValueError("AO Fock derivatives must be symmetric")
        return overlap, fock


@dataclass(frozen=True)
class FrozenOrbitalSFTDAGradient:
    root_gradients_hartree_per_bohr: np.ndarray
    matrix_derivative_hartree_per_bohr: np.ndarray
    minimum_root_gap_hartree: np.ndarray

    def __post_init__(self) -> None:
        gradients = np.array(self.root_gradients_hartree_per_bohr, dtype=float, copy=True)
        matrices = np.array(self.matrix_derivative_hartree_per_bohr, dtype=float, copy=True)
        gaps = np.array(self.minimum_root_gap_hartree, dtype=float, copy=True)
        if gradients.ndim != 3 or gradients.shape[2] != 3:
            raise ValueError("SF-TDA root gradients must have shape nroot x natom x 3")
        if matrices.shape[:2] != gradients.shape[1:] or matrices.ndim != 4:
            raise ValueError("SF-TDA matrix derivatives have incompatible dimensions")
        if matrices.shape[2] != matrices.shape[3] or gaps.shape != (gradients.shape[0],):
            raise ValueError("SF-TDA gradient result dimensions are inconsistent")
        if not np.all(np.isfinite(gradients)) or not np.all(np.isfinite(matrices)):
            raise ValueError("SF-TDA gradient result must be finite")
        if np.any(np.isnan(gaps)) or np.any(gaps < 0.0):
            raise ValueError("SF-TDA root gaps must be nonnegative")
        for item in (gradients, matrices, gaps):
            item.setflags(write=False)
        object.__setattr__(self, "root_gradients_hartree_per_bohr", gradients)
        object.__setattr__(self, "matrix_derivative_hartree_per_bohr", matrices)
        object.__setattr__(self, "minimum_root_gap_hartree", gaps)


@dataclass(frozen=True)
class CoupledOrbitalDerivativeInputs:
    """Analytic fixed-density derivatives and the linear SCF response kernel.

    ``fock_density_response`` maps one AO density perturbation to the
    corresponding AO Fock perturbation.  It must contain the complete linear
    SCC/multipole response but no explicit nuclear derivative.
    """

    overlap_derivative: np.ndarray
    explicit_fock_derivative: np.ndarray
    fock_density_response: Callable[[np.ndarray], np.ndarray]
    degeneracy_tolerance_hartree: float = 1.0e-8

    def validated(self, state: GFN2OrbitalState) -> tuple[np.ndarray, np.ndarray]:
        frozen = FrozenOrbitalDerivativeInputs(
            self.overlap_derivative, self.explicit_fock_derivative
        )
        overlap, fock = frozen.validated(state)
        if not callable(self.fock_density_response):
            raise TypeError("coupled orbital response requires a callable Fock kernel")
        tolerance = float(self.degeneracy_tolerance_hartree)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("orbital degeneracy tolerance must be positive and finite")
        return overlap, fock


@dataclass(frozen=True)
class CoupledOrbitalResponse:
    coefficient_derivative: np.ndarray
    density_derivative: np.ndarray
    total_fock_derivative: np.ndarray
    orbital_energy_derivative: np.ndarray
    residual_norms: np.ndarray

    def __post_init__(self) -> None:
        arrays = tuple(
            np.array(value, dtype=float, copy=True)
            for value in (
                self.coefficient_derivative,
                self.density_derivative,
                self.total_fock_derivative,
                self.orbital_energy_derivative,
                self.residual_norms,
            )
        )
        if not all(np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("coupled orbital response must be finite")
        for value in arrays:
            value.setflags(write=False)
        for name, value in zip(
            (
                "coefficient_derivative",
                "density_derivative",
                "total_fock_derivative",
                "orbital_energy_derivative",
                "residual_norms",
            ),
            arrays,
            strict=True,
        ):
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class RelaxedOrbitalSFTDAGradient:
    excitation_gradients_hartree_per_bohr: np.ndarray
    matrix_derivative_hartree_per_bohr: np.ndarray
    minimum_root_gap_hartree: np.ndarray
    orbital_response: CoupledOrbitalResponse

    def __post_init__(self) -> None:
        frozen = FrozenOrbitalSFTDAGradient(
            self.excitation_gradients_hartree_per_bohr,
            self.matrix_derivative_hartree_per_bohr,
            self.minimum_root_gap_hartree,
        )
        object.__setattr__(
            self,
            "excitation_gradients_hartree_per_bohr",
            frozen.root_gradients_hartree_per_bohr,
        )
        object.__setattr__(
            self,
            "matrix_derivative_hartree_per_bohr",
            frozen.matrix_derivative_hartree_per_bohr,
        )
        object.__setattr__(
            self, "minimum_root_gap_hartree", frozen.minimum_root_gap_hartree
        )



@dataclass(frozen=True)
class MatrixFreeZVectorResult:
    """Adjoint SCF response represented by AO density contractions.

    ``response_density`` and ``response_weighted_density`` are the two
    arguments of the common pair-packed variational-gradient contraction,
    ``P_Z:F^R - W_Z:S^R``.  No coupled-perturbed matrix is stored.
    """

    orbital_source: np.ndarray
    orbital_z_vector: np.ndarray
    response_density: np.ndarray
    response_weighted_density: np.ndarray
    residual_norms: np.ndarray
    iterations: np.ndarray

    def __post_init__(self) -> None:
        arrays = tuple(
            np.array(value, dtype=float, copy=True)
            for value in (
                self.orbital_source,
                self.orbital_z_vector,
                self.response_density,
                self.response_weighted_density,
                self.residual_norms,
            )
        )
        iterations = np.array(self.iterations, dtype=np.int64, copy=True)
        if arrays[0].shape != arrays[1].shape or arrays[0].ndim != 2:
            raise ValueError("Z-vector source and solution dimensions are inconsistent")
        nroot = arrays[0].shape[0]
        if arrays[2].shape[0] != nroot or arrays[3].shape != arrays[2].shape:
            raise ValueError("Z-vector AO density dimensions are inconsistent")
        if arrays[4].shape != (nroot,) or iterations.shape != (nroot,):
            raise ValueError("Z-vector diagnostics must have one value per root")
        if not all(np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("Z-vector result must be finite")
        if np.any(iterations < 0):
            raise ValueError("Z-vector iteration counts must be nonnegative")
        for value in (*arrays, iterations):
            value.setflags(write=False)
        for name, value in zip(
            (
                "orbital_source",
                "orbital_z_vector",
                "response_density",
                "response_weighted_density",
                "residual_norms",
            ),
            arrays,
            strict=True,
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "iterations", iterations)


@dataclass(frozen=True)
class FrozenOrbitalSFTDALagrangian:
    """AO contractions and explicit kernel force for frozen SF-TDA roots."""

    response_density: np.ndarray
    response_weighted_density: np.ndarray
    explicit_kernel_gradient_hartree_per_bohr: np.ndarray

    def __post_init__(self) -> None:
        density = np.array(self.response_density, dtype=float, copy=True)
        weighted = np.array(self.response_weighted_density, dtype=float, copy=True)
        explicit = np.array(
            self.explicit_kernel_gradient_hartree_per_bohr, dtype=float, copy=True
        )
        if density.ndim != 3 or weighted.shape != density.shape:
            raise ValueError("frozen SF-TDA Lagrangian densities are inconsistent")
        if explicit.ndim != 3 or explicit.shape[0] != density.shape[0] or explicit.shape[2] != 3:
            raise ValueError("frozen SF-TDA explicit gradients are inconsistent")
        if not all(np.all(np.isfinite(item)) for item in (density, weighted, explicit)):
            raise ValueError("frozen SF-TDA Lagrangian must be finite")
        for item in (density, weighted, explicit):
            item.setflags(write=False)
        object.__setattr__(self, "response_density", density)
        object.__setattr__(self, "response_weighted_density", weighted)
        object.__setattr__(self, "explicit_kernel_gradient_hartree_per_bohr", explicit)

def _symmetric_square_root(overlap: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(overlap)
    if float(np.min(values)) <= 0.0:
        raise ValueError("SF-TDA Lowdin derivative requires positive-definite overlap")
    root_values = np.sqrt(values)
    root = (vectors * root_values) @ vectors.T
    return root, values, vectors


def _square_root_frechet_derivatives(
    overlap: np.ndarray,
    derivatives: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    root, values, vectors = _symmetric_square_root(overlap)
    root_values = np.sqrt(values)
    denominator = root_values[:, None] + root_values[None, :]
    transformed = np.einsum(
        "pi,...pq,qj->...ij", vectors, derivatives, vectors, optimize=True
    )
    derivative_root_eigenbasis = transformed / denominator
    derivative_root = np.einsum(
        "pi,...ij,qj->...pq",
        vectors,
        derivative_root_eigenbasis,
        vectors,
        optimize=True,
    )
    return root, derivative_root


def _pair_populations(
    lowdin_coefficients: np.ndarray,
    ao_atom: np.ndarray,
    orbitals: np.ndarray,
    derivative_lowdin_coefficients: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    selected = lowdin_coefficients[:, orbitals]
    natoms = int(np.max(ao_atom)) + 1
    populations = np.empty((natoms, len(orbitals), len(orbitals)), dtype=float)
    derivatives = None
    if derivative_lowdin_coefficients is not None:
        derivatives = np.empty(
            (*derivative_lowdin_coefficients.shape[:2], natoms, len(orbitals), len(orbitals)),
            dtype=float,
        )
    for atom in range(natoms):
        mask = ao_atom == atom
        block = selected[mask]
        populations[atom] = block.T @ block
        if derivatives is not None:
            derivative_block = derivative_lowdin_coefficients[:, :, mask][:, :, :, orbitals]
            derivatives[:, :, atom] = (
                np.einsum("...mi,mj->...ij", derivative_block, block, optimize=True)
                + np.einsum("mi,...mj->...ij", block, derivative_block, optimize=True)
            )
    return populations, derivatives


def _gamma_and_derivative(
    state: GFN2OrbitalState,
    coordinates_bohr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    numbers = np.asarray(state.atomic_numbers, dtype=np.intp)
    hardness = grimme_stda_atomic_hardnesses()[numbers - 1]
    scale = GRIMME_XTB_SF_STDA.hardness_scale
    exponent = GRIMME_XTB_SF_STDA.distance_exponent
    screened = scale * 0.5 * (hardness[:, None] + hardness[None, :])
    displacement = coordinates_bohr[:, None, :] - coordinates_bohr[None, :, :]
    distance = np.linalg.norm(displacement, axis=2)
    gamma = 1.0 / (distance**exponent + screened ** (-exponent)) ** (1.0 / exponent)
    natoms = len(numbers)
    derivative = np.zeros((natoms, 3, natoms, natoms), dtype=float)
    for left in range(natoms):
        for right in range(left):
            radius = distance[left, right]
            if radius <= np.finfo(float).tiny:
                continue
            radial = -(radius ** (exponent - 1.0)) * gamma[left, right] ** (exponent + 1.0)
            vector = radial * displacement[left, right] / radius
            derivative[left, :, left, right] += vector
            derivative[left, :, right, left] += vector
            derivative[right, :, left, right] -= vector
            derivative[right, :, right, left] -= vector
    return gamma, derivative


def _response_matrix(
    occupied_pairs: np.ndarray,
    virtual_pairs: np.ndarray,
    gamma: np.ndarray,
) -> np.ndarray:
    tensor = -np.einsum(
        "Aij,AB,Bab->iajb", occupied_pairs, gamma, virtual_pairs, optimize=True
    )
    return tensor.reshape(
        occupied_pairs.shape[1] * virtual_pairs.shape[1],
        occupied_pairs.shape[1] * virtual_pairs.shape[1],
    )


def _response_matrix_derivative(
    occupied_pairs: np.ndarray,
    occupied_pair_derivative: np.ndarray,
    virtual_pairs: np.ndarray,
    virtual_pair_derivative: np.ndarray,
    gamma: np.ndarray,
    gamma_derivative: np.ndarray,
) -> np.ndarray:
    first = -np.einsum(
        "...Aij,AB,Bab->...iajb",
        occupied_pair_derivative,
        gamma,
        virtual_pairs,
        optimize=True,
    )
    first -= np.einsum(
        "Aij,...AB,Bab->...iajb",
        occupied_pairs,
        gamma_derivative,
        virtual_pairs,
        optimize=True,
    )
    first -= np.einsum(
        "Aij,AB,...Bab->...iajb",
        occupied_pairs,
        gamma,
        virtual_pair_derivative,
        optimize=True,
    )
    size = occupied_pairs.shape[1] * virtual_pairs.shape[1]
    return first.reshape((*first.shape[:2], size, size))


def _fixed_selection_matrix(
    full_diagonal: np.ndarray,
    response_matrix: np.ndarray,
    selected: SelectedSpinFlipSpace,
    policy: SFTDASelectionPolicy,
) -> np.ndarray:
    primary = selected.parent_indices[: selected.primary_count]
    included = selected.parent_indices[selected.primary_count :]
    candidates = np.flatnonzero(
        (full_diagonal > policy.spectral_range_hartree)
        & (full_diagonal < policy.maximum_configuration_energy_hartree)
    )
    if policy.frozen_selection is not None:
        candidates = np.asarray(policy.frozen_selection.candidates, dtype=int)
    rejected = candidates[~np.isin(candidates, included)]
    primary_diagonal = full_diagonal[primary].copy()
    if len(rejected):
        coupling = response_matrix[np.ix_(rejected, primary)]
        denominator = full_diagonal[rejected, None] - full_diagonal[primary][None, :] + 1.0e-10
        primary_diagonal -= np.sum(coupling * coupling / denominator, axis=0)
    indices = selected.parent_indices
    matrix = response_matrix[np.ix_(indices, indices)].copy()
    diagonal = np.concatenate((primary_diagonal, full_diagonal[included]))
    np.fill_diagonal(matrix, diagonal)
    return matrix


def frozen_orbital_selected_matrix(
    state: GFN2OrbitalState,
    selected: SelectedSpinFlipSpace,
    policy: SFTDASelectionPolicy,
    *,
    coordinates_bohr: np.ndarray,
    overlap: np.ndarray,
    fock: np.ndarray,
) -> np.ndarray:
    """Evaluate the fixed-CSF frozen-orbital matrix for derivative validation."""

    coordinates = np.asarray(coordinates_bohr, dtype=float)
    metric = np.asarray(overlap, dtype=float)
    hamiltonian = np.asarray(fock, dtype=float)
    coefficients = state.coefficients
    norms = np.diag(coefficients.T @ metric @ coefficients)
    energies = np.diag(coefficients.T @ hamiltonian @ coefficients) / norms
    parent = selected.parent
    gaps = (
        energies[parent.virtual_beta][None, :]
        - energies[parent.occupied_alpha][:, None]
    ).reshape(-1)
    square_root, _, _ = _symmetric_square_root(metric)
    lowdin = square_root @ coefficients
    occupied_pairs, _ = _pair_populations(lowdin, state.ao_atom, parent.occupied_alpha)
    virtual_pairs, _ = _pair_populations(lowdin, state.ao_atom, parent.virtual_beta)
    gamma, _ = _gamma_and_derivative(state, coordinates)
    response = _response_matrix(occupied_pairs, virtual_pairs, gamma)
    return _fixed_selection_matrix(gaps + np.diag(response), response, selected, policy)


def solve_coupled_orbital_response(
    state: GFN2OrbitalState,
    derivatives: CoupledOrbitalDerivativeInputs,
) -> CoupledOrbitalResponse:
    """Solve the linearized generalized SCF equations in canonical orbitals.

    The response is formed in the anti-Hermitian orbital-rotation space.  The
    metric (Pulay) density response is included on the right-hand side and the
    supplied SCC/multipole Fock kernel is contracted exactly into the response
    matrix.  This dense reference implementation is the qualification oracle
    for the later matrix-free Z-vector solver.
    """

    overlap_derivative, explicit_fock_derivative = derivatives.validated(state)
    coefficients = state.coefficients
    energies = state.orbital_energies_hartree
    occupations = state.alpha_occupations + state.beta_occupations
    nmo = state.nmo
    left, right = np.triu_indices(nmo, 1)
    gaps = energies[right] - energies[left]
    tolerance = float(derivatives.degeneracy_tolerance_hartree)
    singular = np.abs(gaps) <= tolerance
    if np.any(singular):
        pairs = tuple((int(p), int(q)) for p, q in zip(left[singular], right[singular]))
        raise ValueError(
            "coupled orbital response has an unresolved canonical degeneracy "
            f"for pairs {pairs[:4]}"
        )

    pair_count = len(left)
    response_matrix = np.eye(pair_count, dtype=float)
    for column, (p, q) in enumerate(zip(left, right, strict=True)):
        density_mo = np.zeros((nmo, nmo), dtype=float)
        value = occupations[q] - occupations[p]
        density_mo[p, q] = value
        density_mo[q, p] = value
        density_ao = coefficients @ density_mo @ coefficients.T
        fock_ao = np.asarray(derivatives.fock_density_response(density_ao), dtype=float)
        if fock_ao.shape != state.fock.shape or not np.all(np.isfinite(fock_ao)):
            raise ValueError("Fock density-response kernel returned an invalid matrix")
        fock_mo = coefficients.T @ fock_ao @ coefficients
        response_matrix[:, column] -= fock_mo[left, right] / gaps

    axes = overlap_derivative.shape[:2]
    coefficient_response = np.empty((*axes, state.nao, nmo), dtype=float)
    density_response = np.empty((*axes, state.nao, state.nao), dtype=float)
    total_fock_response = np.empty_like(explicit_fock_derivative)
    energy_response = np.empty((*axes, nmo), dtype=float)
    residuals = np.empty(axes, dtype=float)
    for atom in range(axes[0]):
        for component in range(axes[1]):
            ds_mo = coefficients.T @ overlap_derivative[atom, component] @ coefficients
            metric_density_mo = -0.5 * (
                ds_mo * occupations[None, :] + occupations[:, None] * ds_mo
            )
            metric_density_ao = coefficients @ metric_density_mo @ coefficients.T
            metric_fock = np.asarray(
                derivatives.fock_density_response(metric_density_ao), dtype=float
            )
            if metric_fock.shape != state.fock.shape or not np.all(
                np.isfinite(metric_fock)
            ):
                raise ValueError("Fock density-response kernel returned an invalid matrix")
            explicit_total = explicit_fock_derivative[atom, component] + metric_fock
            explicit_mo = coefficients.T @ explicit_total @ coefficients
            rhs = (
                explicit_mo[left, right]
                - 0.5 * (energies[left] + energies[right]) * ds_mo[left, right]
            ) / gaps
            rotations = np.linalg.solve(response_matrix, rhs)
            coupled_residual = np.linalg.norm(response_matrix @ rotations - rhs)

            kappa = np.zeros((nmo, nmo), dtype=float)
            kappa[left, right] = rotations
            kappa[right, left] = -rotations
            orbital_connection = -0.5 * ds_mo + kappa
            dc = coefficients @ orbital_connection
            density_mo = metric_density_mo.copy()
            density_mo[left, right] += (occupations[right] - occupations[left]) * rotations
            density_mo[right, left] = density_mo[left, right]
            dp = coefficients @ density_mo @ coefficients.T
            df = explicit_fock_derivative[atom, component] + np.asarray(
                derivatives.fock_density_response(dp), dtype=float
            )
            de = np.diag(
                coefficients.T @ df @ coefficients - ds_mo * energies[None, :]
            )
            eigen_residual = (
                df @ coefficients
                + state.fock @ dc
                - (overlap_derivative[atom, component] @ coefficients) * energies
                - (state.overlap @ dc) * energies
                - (state.overlap @ coefficients) * de
            )
            residuals[atom, component] = max(
                coupled_residual, float(np.linalg.norm(eigen_residual))
            )
            coefficient_response[atom, component] = dc
            density_response[atom, component] = dp
            total_fock_response[atom, component] = df
            energy_response[atom, component] = de

    return CoupledOrbitalResponse(
        coefficient_response,
        density_response,
        total_fock_response,
        energy_response,
        residuals,
    )



def _matrix_free_gmres(
    apply: Callable[[np.ndarray], np.ndarray],
    right_hand_side: np.ndarray,
    *,
    tolerance: float,
    maximum_iterations: int,
    right_preconditioner: Callable[[np.ndarray], np.ndarray] | None = None,
    single_pass: bool = False,
) -> tuple[np.ndarray, float, int]:
    """Small deterministic unrestarted GMRES for an implicit SCF adjoint.

    An optional right preconditioner solves ``A M^-1 y = b`` and returns
    ``x = M^-1 y``.  The reported residual is always certified against the
    original, unpreconditioned equation.
    """

    rhs = np.asarray(right_hand_side, dtype=float)
    if rhs.ndim != 1:
        raise ValueError("matrix-free GMRES right-hand side must be a vector")
    rhs_norm = float(np.linalg.norm(rhs))
    if rhs_norm == 0.0:
        return np.zeros_like(rhs), 0.0, 0
    threshold = tolerance * max(1.0, rhs_norm)
    if rhs_norm <= threshold:
        return np.zeros_like(rhs), rhs_norm, 0
    limit = min(len(rhs), int(maximum_iterations))
    if limit <= 0 or tolerance <= 0.0:
        raise ValueError("matrix-free GMRES limits must be positive")

    def precondition(vector: np.ndarray) -> np.ndarray:
        if right_preconditioner is None:
            return vector
        transformed = np.asarray(right_preconditioner(vector), dtype=float)
        if transformed.shape != rhs.shape or not np.all(np.isfinite(transformed)):
            raise ValueError("matrix-free GMRES preconditioner returned an invalid vector")
        return transformed

    def krylov_apply(vector: np.ndarray) -> np.ndarray:
        return apply(precondition(vector))

    basis = np.zeros((len(rhs), limit + 1), dtype=float)
    hessenberg = np.zeros((limit + 1, limit), dtype=float)
    basis[:, 0] = rhs / rhs_norm
    # Maintain the QR factorization of the growing Hessenberg matrix with
    # Givens rotations.  Re-solving the full least-squares problem at every
    # Arnoldi step makes a long response solve unnecessarily cubic in the
    # iteration count and dominated complete analytic-Hessian timings.
    cosines = np.zeros(limit, dtype=float)
    sines = np.zeros(limit, dtype=float)
    rotated_target = np.zeros(limit + 1, dtype=float)
    rotated_target[0] = rhs_norm
    residual = rhs_norm
    for column in range(limit):
        vector = np.asarray(krylov_apply(basis[:, column]), dtype=float)
        if vector.shape != rhs.shape or not np.all(np.isfinite(vector)):
            raise ValueError("matrix-free SCF adjoint returned an invalid vector")
        active_basis = basis[:, : column + 1]
        projections = active_basis.T @ vector
        hessenberg[: column + 1, column] = projections
        vector -= active_basis @ projections
        if not single_pass:
            # The certified fallback retains full reorthogonalization near
            # SCF instabilities and for the other response callers.
            correction = active_basis.T @ vector
            hessenberg[: column + 1, column] += correction
            vector -= active_basis @ correction
        hessenberg[column + 1, column] = np.linalg.norm(vector)
        if hessenberg[column + 1, column] > np.finfo(float).eps:
            basis[:, column + 1] = vector / hessenberg[column + 1, column]

        for row in range(column):
            upper = hessenberg[row, column]
            lower = hessenberg[row + 1, column]
            hessenberg[row, column] = cosines[row] * upper + sines[row] * lower
            hessenberg[row + 1, column] = -sines[row] * upper + cosines[row] * lower

        upper = hessenberg[column, column]
        lower = hessenberg[column + 1, column]
        radius = float(np.hypot(upper, lower))
        if radius > np.finfo(float).eps:
            cosines[column] = upper / radius
            sines[column] = lower / radius
        else:
            cosines[column] = 1.0
            sines[column] = 0.0
        hessenberg[column, column] = radius
        hessenberg[column + 1, column] = 0.0

        target_value = rotated_target[column]
        rotated_target[column] = cosines[column] * target_value
        rotated_target[column + 1] = -sines[column] * target_value
        residual = abs(float(rotated_target[column + 1]))
        converged = residual <= threshold
        breakdown = lower <= np.finfo(float).eps
        if converged or breakdown:
            triangular = hessenberg[: column + 1, : column + 1]
            target = rotated_target[: column + 1]
            if float(np.min(np.abs(np.diag(triangular)))) > np.finfo(float).eps:
                coefficients = np.linalg.solve(triangular, target)
            else:
                coefficients = np.linalg.lstsq(triangular, target, rcond=None)[0]
            solution = precondition(basis[:, : column + 1] @ coefficients)
            residual = float(np.linalg.norm(apply(solution) - rhs))
            if residual <= threshold:
                return solution, residual, column + 1
        if breakdown:
            break
    if single_pass:
        return _matrix_free_gmres(
            apply, rhs, tolerance=tolerance,
            maximum_iterations=maximum_iterations,
            right_preconditioner=right_preconditioner,
        )
    raise RuntimeError(
        f"matrix-free SCF adjoint did not converge: residual={residual:.3e}, iterations={limit}"
    )


def matrix_free_sf_tda_z_vector(
    state: GFN2OrbitalState,
    result: SFTDAResult,
    fock_density_response: Callable[[np.ndarray], np.ndarray],
    *,
    selection_policy: SFTDASelectionPolicy | None = None,
    tolerance: float = 1.0e-10,
    maximum_iterations: int = 200,
    degeneracy_tolerance_hartree: float = 1.0e-8,
) -> MatrixFreeZVectorResult:
    """Solve the SF-TDA SCF adjoint without forming the CP-SCF matrix.

    The SF-TDA orbital Lagrangian supplies the right-hand side.  Applications
    of the transposed SCF Jacobian use only AO density-response products.  The
    returned AO densities can consequently be passed directly to the common
    pair-packed ground-state derivative assembler.
    """

    if not callable(fock_density_response):
        raise TypeError("matrix-free Z-vector requires a callable Fock kernel")
    coefficients = state.coefficients
    energies = state.orbital_energies_hartree
    occupations = state.alpha_occupations + state.beta_occupations
    left, right = np.triu_indices(state.nmo, 1)
    gaps = energies[right] - energies[left]
    if np.any(np.abs(gaps) <= float(degeneracy_tolerance_hartree)):
        raise ValueError("matrix-free Z-vector encountered a canonical degeneracy")
    occupation_difference = occupations[right] - occupations[left]

    nroot = len(result.excitation_energies_hartree)
    frozen, coefficient_adjoint = _frozen_orbital_sf_tda_reverse(
        state, result, selection_policy=selection_policy
    )
    source = np.empty((nroot, len(left)), dtype=float)
    metric_weighted_mo = np.zeros((nroot, state.nmo, state.nmo), dtype=float)
    for root_index in range(nroot):
        # The density-response Jacobian is the Hessian of the GFN2 energy and
        # therefore self-adjoint.  One application to the root AO adjoint
        # replaces one application for every orbital-pair direction.
        kernel_ao = np.asarray(
            fock_density_response(frozen.response_density[root_index]), dtype=float
        )
        if kernel_ao.shape != state.fock.shape or not np.all(np.isfinite(kernel_ao)):
            raise ValueError("Fock density-response kernel returned an invalid matrix")
        kernel_mo = coefficients.T @ kernel_ao @ coefficients
        coefficient_mo = coefficient_adjoint[root_index].T @ coefficients
        source[root_index] = (
            coefficient_mo[right, left] - coefficient_mo[left, right]
            + 2.0 * occupation_difference * kernel_mo[left, right]
        )
        metric_source = -0.5 * (coefficient_mo + coefficient_mo.T)
        kernel_metric = -0.5 * (
            occupations[:, None] + occupations[None, :]
        ) * kernel_mo
        kernel_metric *= 2.0
        metric_source += kernel_metric
        metric_weighted_mo[root_index] = -0.5 * metric_source

    def apply_transpose(vector: np.ndarray) -> np.ndarray:
        scaled = np.asarray(vector, dtype=float) / gaps
        trial_mo = np.zeros((state.nmo, state.nmo), dtype=float)
        trial_mo[left, right] = 0.5 * scaled
        trial_mo[right, left] = 0.5 * scaled
        trial_ao = coefficients @ trial_mo @ coefficients.T
        response_ao = np.asarray(fock_density_response(trial_ao), dtype=float)
        response_mo = coefficients.T @ response_ao @ coefficients
        return np.asarray(vector, dtype=float) - (
            2.0 * occupation_difference * response_mo[left, right]
        )

    z_vectors = np.empty_like(source)
    residuals = np.empty(len(source), dtype=float)
    iterations = np.empty(len(source), dtype=np.int64)
    response_density = np.empty((len(source), state.nao, state.nao), dtype=float)
    response_weighted = np.empty_like(response_density)
    energy_sum = energies[left] + energies[right]
    for root_index, root_source in enumerate(source):
        z_vector, residual, count = _matrix_free_gmres(
            apply_transpose,
            root_source,
            tolerance=float(tolerance),
            maximum_iterations=int(maximum_iterations),
        )
        z_vectors[root_index] = z_vector
        residuals[root_index] = residual
        iterations[root_index] = count
        scaled = z_vector / gaps
        density_mo = np.zeros((state.nmo, state.nmo), dtype=float)
        density_mo[left, right] = 0.5 * scaled
        density_mo[right, left] = 0.5 * scaled
        density_ao = coefficients @ density_mo @ coefficients.T
        response_density[root_index] = density_ao

        kernel_ao = np.asarray(fock_density_response(density_ao), dtype=float)
        kernel_mo = coefficients.T @ kernel_ao @ coefficients
        weighted_mo = metric_weighted_mo[root_index].copy()
        weighted_mo += 0.5 * kernel_mo * (occupations[:, None] + occupations[None, :])
        weighted_mo[left, right] += 0.25 * scaled * energy_sum
        weighted_mo[right, left] += 0.25 * scaled * energy_sum
        response_weighted[root_index] = coefficients @ weighted_mo @ coefficients.T

    return MatrixFreeZVectorResult(
        source,
        z_vectors,
        response_density,
        response_weighted,
        residuals,
        iterations,
    )

def _sf_tda_gradient_with_coefficient_response(
    state: GFN2OrbitalState,
    result: SFTDAResult,
    overlap_derivative: np.ndarray,
    fock_derivative: np.ndarray,
    coefficient_derivative: np.ndarray,
    *,
    selection_policy: SFTDASelectionPolicy | None = None,
) -> FrozenOrbitalSFTDAGradient:
    policy = selection_policy or SFTDASelectionPolicy()
    parent = SpinFlipExcitationSpace.from_grimme_window(state, policy)
    response = AtomicTransitionChargeResponse.grimme_xtb_spin_flip(state, parent)
    selected = select_grimme_spin_flip_configurations(parent, response, policy)
    if result.amplitudes.shape[1] != selected.size:
        raise ValueError("SF-TDA result and frozen selected space are inconsistent")

    root, derivative_root = _square_root_frechet_derivatives(
        state.overlap, overlap_derivative
    )
    lowdin = root @ state.coefficients
    derivative_lowdin = np.einsum(
        "...pq,qr->...pr", derivative_root, state.coefficients, optimize=True
    )
    derivative_lowdin += np.einsum(
        "pq,...qr->...pr", root, coefficient_derivative, optimize=True
    )
    occupied_pairs, occupied_pair_derivative = _pair_populations(
        lowdin,
        state.ao_atom,
        parent.occupied_alpha,
        derivative_lowdin,
    )
    virtual_pairs, virtual_pair_derivative = _pair_populations(
        lowdin,
        state.ao_atom,
        parent.virtual_beta,
        derivative_lowdin,
    )
    gamma, gamma_derivative = _gamma_and_derivative(state, state.coordinates_bohr)
    response_matrix = _response_matrix(occupied_pairs, virtual_pairs, gamma)
    response_derivative = _response_matrix_derivative(
        occupied_pairs,
        occupied_pair_derivative,
        virtual_pairs,
        virtual_pair_derivative,
        gamma,
        gamma_derivative,
    )

    coefficients = state.coefficients
    energy_derivative = np.einsum(
        "pi,...pq,qi->...i", coefficients, fock_derivative, coefficients, optimize=True
    )
    energy_derivative -= np.einsum(
        "pi,...pq,qi,i->...i",
        coefficients,
        overlap_derivative,
        coefficients,
        state.orbital_energies_hartree,
        optimize=True,
    )
    gap_derivative = (
        energy_derivative[:, :, parent.virtual_beta][:, :, None, :]
        - energy_derivative[:, :, parent.occupied_alpha][:, :, :, None]
    ).reshape((*energy_derivative.shape[:2], parent.size))
    full_diagonal = parent.diagonal_hartree + np.diag(response_matrix)
    full_diagonal_derivative = gap_derivative + np.diagonal(
        response_derivative, axis1=-2, axis2=-1
    )

    primary = selected.parent_indices[: selected.primary_count]
    included = selected.parent_indices[selected.primary_count :]
    candidates = np.flatnonzero(
        (full_diagonal > policy.spectral_range_hartree)
        & (full_diagonal < policy.maximum_configuration_energy_hartree)
    )
    if policy.frozen_selection is not None:
        candidates = np.asarray(policy.frozen_selection.candidates, dtype=int)
    rejected = candidates[~np.isin(candidates, included)]
    primary_derivative = full_diagonal_derivative[:, :, primary].copy()
    if len(rejected):
        coupling = response_matrix[np.ix_(rejected, primary)]
        coupling_derivative = response_derivative[:, :, rejected][:, :, :, primary]
        denominator = full_diagonal[rejected, None] - full_diagonal[primary][None, :] + 1.0e-10
        denominator_derivative = (
            full_diagonal_derivative[:, :, rejected, None]
            - full_diagonal_derivative[:, :, None, primary]
        )
        contribution_derivative = (
            2.0 * coupling[None, None, :, :] * coupling_derivative / denominator
            - coupling[None, None, :, :] ** 2
            * denominator_derivative
            / denominator**2
        )
        primary_derivative -= np.sum(contribution_derivative, axis=2)

    indices = selected.parent_indices
    matrix_derivative = response_derivative[:, :, indices][:, :, :, indices].copy()
    selected_diagonal_derivative = np.concatenate(
        (primary_derivative, full_diagonal_derivative[:, :, included]), axis=2
    )
    diagonal_indices = np.arange(selected.size)
    matrix_derivative[:, :, diagonal_indices, diagonal_indices] = selected_diagonal_derivative
    root_gradients = np.einsum(
        "ri,...ij,rj->r...",
        result.amplitudes,
        matrix_derivative,
        result.amplitudes,
        optimize=True,
    )
    energies = result.excitation_energies_hartree
    gaps = np.abs(energies[:, None] - energies[None, :])
    np.fill_diagonal(gaps, np.inf)
    minimum_gap = np.min(gaps, axis=1) if len(energies) > 1 else np.asarray((np.inf,))
    return FrozenOrbitalSFTDAGradient(root_gradients, matrix_derivative, minimum_gap)


def _frozen_orbital_sf_tda_reverse(
    state: GFN2OrbitalState,
    result: SFTDAResult,
    *,
    selection_policy: SFTDASelectionPolicy | None = None,
) -> tuple[FrozenOrbitalSFTDALagrangian, np.ndarray]:
    """Build the frozen-root AO VJP consumed by pair-packed derivatives.

    This is the exact reverse contraction of the selected SF-TDA eigenvalue
    with respect to the AO Fock and overlap matrices.  Nuclear derivatives are
    not materialized.  The remaining explicit term is the derivative of the
    atom-pair SF-TDA interaction kernel.
    """

    policy = selection_policy or SFTDASelectionPolicy()
    parent = SpinFlipExcitationSpace.from_grimme_window(state, policy)
    transition_response = AtomicTransitionChargeResponse.grimme_xtb_spin_flip(
        state, parent
    )
    selected = select_grimme_spin_flip_configurations(
        parent, transition_response, policy
    )
    if result.amplitudes.shape[1] != selected.size:
        raise ValueError("SF-TDA result and frozen selected space are inconsistent")

    # Reverse differentiation replaces the former AO-by-AO directional
    # probing.  Its working set is independent of 3*Natom*NAO**2 and the
    # expensive electronic intermediates are built exactly once.
    square_root, overlap_values, overlap_vectors = _symmetric_square_root(
        state.overlap
    )
    lowdin = square_root @ state.coefficients
    occupied_pairs, _ = _pair_populations(
        lowdin, state.ao_atom, parent.occupied_alpha
    )
    virtual_pairs, _ = _pair_populations(
        lowdin, state.ao_atom, parent.virtual_beta
    )
    gamma, _ = _gamma_and_derivative(state, state.coordinates_bohr)
    response_matrix = _response_matrix(occupied_pairs, virtual_pairs, gamma)
    full_diagonal = parent.diagonal_hartree + np.diag(response_matrix)

    nroot = len(result.excitation_energies_hartree)
    density = np.empty((nroot, state.nao, state.nao), dtype=float)
    weighted = np.empty_like(density)
    explicit = np.empty((nroot, len(state.atomic_numbers), 3), dtype=float)
    coefficient_adjoint = np.empty((nroot, state.nao, state.nmo), dtype=float)
    root_values = np.sqrt(overlap_values)
    frechet_denominator = root_values[:, None] + root_values[None, :]
    occupied = parent.occupied_alpha
    virtual = parent.virtual_beta

    for root_index, amplitude in enumerate(result.amplitudes):
        response_adjoint, gap_adjoint = _selected_root_adjoint(
            amplitude,
            full_diagonal,
            response_matrix,
            selected,
            policy,
        )

        orbital_adjoint = np.zeros(state.nmo, dtype=float)
        excitation_adjoint = gap_adjoint.reshape(
            len(parent.occupied_alpha), len(parent.virtual_beta)
        )
        np.add.at(orbital_adjoint, virtual, np.sum(excitation_adjoint, axis=0))
        np.add.at(orbital_adjoint, occupied, -np.sum(excitation_adjoint, axis=1))
        density[root_index] = (
            state.coefficients * orbital_adjoint
        ) @ state.coefficients.T

        response_tensor = response_adjoint.reshape(
            len(occupied), len(virtual), len(occupied), len(virtual)
        )
        occupied_adjoint = -np.einsum(
            "iajb,AB,Bab->Aij",
            response_tensor,
            gamma,
            virtual_pairs,
            optimize=True,
        )
        virtual_adjoint = -np.einsum(
            "iajb,Aij,AB->Bab",
            response_tensor,
            occupied_pairs,
            gamma,
            optimize=True,
        )
        gamma_adjoint = -np.einsum(
            "iajb,Aij,Bab->AB",
            response_tensor,
            occupied_pairs,
            virtual_pairs,
            optimize=True,
        )

        lowdin_adjoint = np.zeros_like(lowdin)
        for atom in range(len(state.atomic_numbers)):
            mask = state.ao_atom == atom
            occupied_block = lowdin[np.ix_(mask, occupied)]
            virtual_block = lowdin[np.ix_(mask, virtual)]
            lowdin_adjoint[np.ix_(mask, occupied)] += occupied_block @ (
                occupied_adjoint[atom] + occupied_adjoint[atom].T
            )
            lowdin_adjoint[np.ix_(mask, virtual)] += virtual_block @ (
                virtual_adjoint[atom] + virtual_adjoint[atom].T
            )

        square_root_adjoint = lowdin_adjoint @ state.coefficients.T
        coefficient_adjoint[root_index] = square_root @ lowdin_adjoint
        square_root_adjoint = 0.5 * (
            square_root_adjoint + square_root_adjoint.T
        )
        eigenbasis_adjoint = (
            overlap_vectors.T @ square_root_adjoint @ overlap_vectors
        ) / frechet_denominator
        overlap_adjoint = (
            overlap_vectors @ eigenbasis_adjoint @ overlap_vectors.T
        )
        overlap_adjoint -= (
            state.coefficients
            * (orbital_adjoint * state.orbital_energies_hartree)
        ) @ state.coefficients.T
        weighted[root_index] = -0.5 * (
            overlap_adjoint + overlap_adjoint.T
        )
        explicit[root_index] = _gamma_adjoint_gradient(
            state, gamma, gamma_adjoint
        )

    return (
        FrozenOrbitalSFTDALagrangian(density, weighted, explicit),
        coefficient_adjoint,
    )


def frozen_orbital_sf_tda_lagrangian(
    state: GFN2OrbitalState,
    result: SFTDAResult,
    *,
    selection_policy: SFTDASelectionPolicy | None = None,
) -> FrozenOrbitalSFTDALagrangian:
    """Build the frozen-root AO VJP with one analytic reverse contraction."""

    lagrangian, _ = _frozen_orbital_sf_tda_reverse(
        state, result, selection_policy=selection_policy
    )
    return lagrangian


def _selected_root_adjoint(
    amplitude: np.ndarray,
    full_diagonal: np.ndarray,
    response_matrix: np.ndarray,
    selected: SelectedSpinFlipSpace,
    policy: SFTDASelectionPolicy,
) -> tuple[np.ndarray, np.ndarray]:
    """Reverse contraction of one selected SF-TDA eigenvalue.

    The routine differentiates the fixed-selection/downfolding expression
    directly.  It is linear in the retained matrix working set and avoids one
    complete SF-TDA rebuild for every AO matrix element.
    """

    vector = np.asarray(amplitude, dtype=float)
    matrix_adjoint = np.outer(vector, vector)
    response_adjoint = np.zeros_like(response_matrix)
    diagonal_adjoint = np.zeros_like(full_diagonal)
    indices = selected.parent_indices
    off_diagonal = matrix_adjoint.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    response_adjoint[np.ix_(indices, indices)] += off_diagonal

    primary = indices[: selected.primary_count]
    included = indices[selected.primary_count :]
    selected_diagonal_adjoint = np.diag(matrix_adjoint)
    if len(included):
        diagonal_adjoint[included] += selected_diagonal_adjoint[
            selected.primary_count :
        ]
    candidates = np.flatnonzero(
        (full_diagonal > policy.spectral_range_hartree)
        & (full_diagonal < policy.maximum_configuration_energy_hartree)
    )
    if policy.frozen_selection is not None:
        candidates = np.asarray(policy.frozen_selection.candidates, dtype=int)
    rejected = candidates[~np.isin(candidates, included)]
    for position, primary_index in enumerate(primary):
        weight = selected_diagonal_adjoint[position]
        diagonal_adjoint[primary_index] += weight
        if not len(rejected):
            continue
        coupling = response_matrix[rejected, primary_index]
        denominator = (
            full_diagonal[rejected] - full_diagonal[primary_index] + 1.0e-10
        )
        diagonal_adjoint[rejected] += weight * coupling**2 / denominator**2
        diagonal_adjoint[primary_index] -= np.sum(
            weight * coupling**2 / denominator**2
        )
        coupling_adjoint = -2.0 * weight * coupling / denominator
        response_adjoint[rejected, primary_index] += 0.5 * coupling_adjoint
        response_adjoint[primary_index, rejected] += 0.5 * coupling_adjoint

    diagonal = np.diag_indices_from(response_adjoint)
    response_adjoint[diagonal] += diagonal_adjoint
    return response_adjoint, diagonal_adjoint


def _gamma_adjoint_gradient(
    state: GFN2OrbitalState,
    gamma: np.ndarray,
    gamma_adjoint: np.ndarray,
) -> np.ndarray:
    """Contract the screened atom-pair kernel derivative without a 4-D tensor."""

    coordinates = np.asarray(state.coordinates_bohr, dtype=float)
    exponent = GRIMME_XTB_SF_STDA.distance_exponent
    gradient = np.zeros_like(coordinates)
    for left in range(len(coordinates)):
        displacement = coordinates[left] - coordinates[:left]
        distance = np.linalg.norm(displacement, axis=1)
        valid = distance > np.finfo(float).tiny
        if not np.any(valid):
            continue
        radial = np.zeros(left, dtype=float)
        radial[valid] = -(
            distance[valid] ** (exponent - 1.0)
            * gamma[left, :left][valid] ** (exponent + 1.0)
        )
        vector = np.zeros_like(displacement)
        vector[valid] = (
            radial[valid, None]
            * displacement[valid]
            / distance[valid, None]
        )
        weight = gamma_adjoint[left, :left] + gamma_adjoint[:left, left]
        contribution = weight[:, None] * vector
        gradient[left] += np.sum(contribution, axis=0)
        gradient[:left] -= contribution
    return gradient


def frozen_orbital_sf_tda_gradient(
    state: GFN2OrbitalState,
    result: SFTDAResult,
    derivatives: FrozenOrbitalDerivativeInputs,
    *,
    selection_policy: SFTDASelectionPolicy | None = None,
) -> FrozenOrbitalSFTDAGradient:
    """Differentiate selected SF-TDA roots at fixed AO MO coefficients."""

    overlap_derivative, fock_derivative = derivatives.validated(state)
    coefficient_derivative = np.zeros(
        (*overlap_derivative.shape[:2], state.nao, state.nmo), dtype=float
    )
    return _sf_tda_gradient_with_coefficient_response(
        state,
        result,
        overlap_derivative,
        fock_derivative,
        coefficient_derivative,
        selection_policy=selection_policy,
    )


def relaxed_orbital_sf_tda_gradient(
    state: GFN2OrbitalState,
    result: SFTDAResult,
    derivatives: CoupledOrbitalDerivativeInputs,
    *,
    selection_policy: SFTDASelectionPolicy | None = None,
) -> RelaxedOrbitalSFTDAGradient:
    """Return the analytic excitation gradient with orbital/SCC relaxation.

    The active space and PT inclusion decisions remain frozen at the reference
    geometry.  Continuous canonical-orbital, density, metric, and supplied
    SCC/multipole Fock responses are fully relaxed.
    """

    overlap_derivative, _explicit_fock = derivatives.validated(state)
    response = solve_coupled_orbital_response(state, derivatives)
    gradient = _sf_tda_gradient_with_coefficient_response(
        state,
        result,
        overlap_derivative,
        response.total_fock_derivative,
        response.coefficient_derivative,
        selection_policy=selection_policy,
    )
    return RelaxedOrbitalSFTDAGradient(
        gradient.root_gradients_hartree_per_bohr,
        gradient.matrix_derivative_hartree_per_bohr,
        gradient.minimum_root_gap_hartree,
        response,
    )


__all__ = [
    "CoupledOrbitalDerivativeInputs",
    "CoupledOrbitalResponse",
    "FrozenOrbitalDerivativeInputs",
    "FrozenOrbitalSFTDALagrangian",
    "FrozenOrbitalSFTDAGradient",
    "MatrixFreeZVectorResult",
    "RelaxedOrbitalSFTDAGradient",
    "frozen_orbital_selected_matrix",
    "frozen_orbital_sf_tda_lagrangian",
    "frozen_orbital_sf_tda_gradient",
    "matrix_free_sf_tda_z_vector",
    "relaxed_orbital_sf_tda_gradient",
    "solve_coupled_orbital_response",
]
