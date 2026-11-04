"""Analytic coupled response of the resident MATRIX-GFN2 ground state.

This module deliberately contains no excited-state objects.  It provides the
first-order density and energy-weighted-density response needed by an analytic
ground-state Hessian.  The explicit, fixed-density second nuclear derivative is
a separate kernel and is not approximated here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .electronic_state import GFN2OrbitalState


@dataclass(frozen=True)
class GroundStateDerivativeMatrices:
    """Fixed-density AO derivatives entering the generalized CPSCF equations."""

    overlap_per_bohr: np.ndarray
    fock_hartree_per_bohr: np.ndarray

    def validated(self, state: GFN2OrbitalState) -> tuple[np.ndarray, np.ndarray]:
        expected = (len(state.atomic_numbers), 3, state.nao, state.nao)
        overlap = np.asarray(self.overlap_per_bohr, dtype=float)
        fock = np.asarray(self.fock_hartree_per_bohr, dtype=float)
        if overlap.shape != expected or fock.shape != expected:
            raise ValueError("ground-state AO derivatives have incompatible dimensions")
        if not np.all(np.isfinite(overlap)) or not np.all(np.isfinite(fock)):
            raise ValueError("ground-state AO derivatives must be finite")
        if not np.allclose(overlap, overlap.swapaxes(-1, -2), atol=2.0e-12, rtol=0.0):
            raise ValueError("overlap derivatives must be symmetric")
        if not np.allclose(fock, fock.swapaxes(-1, -2), atol=2.0e-12, rtol=0.0):
            raise ValueError("fixed-density Fock derivatives must be symmetric")
        return overlap, fock


@dataclass(frozen=True)
class GroundStateCoupledResponse:
    """First nuclear derivatives of a stationary GFN2 electronic state."""

    coefficient_derivative: np.ndarray
    density_derivative: np.ndarray
    weighted_density_derivative: np.ndarray
    fock_derivative: np.ndarray
    orbital_energy_derivative: np.ndarray
    residual_norms: np.ndarray

    def __post_init__(self) -> None:
        names = (
            "coefficient_derivative",
            "density_derivative",
            "weighted_density_derivative",
            "fock_derivative",
            "orbital_energy_derivative",
            "residual_norms",
        )
        for name in names:
            value = np.array(getattr(self, name), dtype=float, copy=True, order="C")
            if not np.all(np.isfinite(value)):
                raise ValueError("ground-state coupled response must be finite")
            value.setflags(write=False)
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class GroundStateDirectionalResponse:
    """CPSCF response to one nuclear displacement direction."""

    coefficient_derivative: np.ndarray
    density_derivative: np.ndarray
    weighted_density_derivative: np.ndarray
    fock_derivative: np.ndarray
    orbital_energy_derivative: np.ndarray
    residual_norm: float
    iterations: int

    def __post_init__(self) -> None:
        coefficient = np.array(self.coefficient_derivative, dtype=float, copy=True)
        density = np.array(self.density_derivative, dtype=float, copy=True)
        weighted = np.array(
            self.weighted_density_derivative, dtype=float, copy=True
        )
        fock = np.array(self.fock_derivative, dtype=float, copy=True)
        energies = np.array(self.orbital_energy_derivative, dtype=float, copy=True)
        if density.ndim != 2 or density.shape != weighted.shape:
            raise ValueError("directional response matrices have inconsistent shapes")
        if coefficient.ndim != 2 or coefficient.shape[0] != density.shape[0]:
            raise ValueError("directional coefficient response has incompatible shape")
        if fock.shape != density.shape or energies.shape != (coefficient.shape[1],):
            raise ValueError("directional Fock/orbital response has incompatible shape")
        arrays = (coefficient, density, weighted, fock, energies)
        if any(not np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("directional response matrices must be finite")
        for value in arrays:
            value.setflags(write=False)
        object.__setattr__(self, "coefficient_derivative", coefficient)
        object.__setattr__(self, "density_derivative", density)
        object.__setattr__(self, "weighted_density_derivative", weighted)
        object.__setattr__(self, "fock_derivative", fock)
        object.__setattr__(self, "orbital_energy_derivative", energies)
        residual = float(self.residual_norm)
        if not np.isfinite(residual) or residual < 0.0 or int(self.iterations) < 0:
            raise ValueError("directional CPSCF diagnostics are invalid")
        object.__setattr__(self, "residual_norm", residual)
        object.__setattr__(self, "iterations", int(self.iterations))


def solve_ground_state_directional_response(
    state: GFN2OrbitalState,
    overlap_direction: np.ndarray,
    fock_direction: np.ndarray,
    fock_density_response: Callable[[np.ndarray], np.ndarray],
    *,
    tolerance: float = 1.0e-10,
    maximum_iterations: int = 200,
    degeneracy_tolerance_hartree: float = 1.0e-8,
) -> GroundStateDirectionalResponse:
    """Solve one CPSCF direction without forming the orbital Hessian."""

    from .sf_tda_gradient import _matrix_free_gmres

    ds = np.asarray(overlap_direction, dtype=float)
    explicit = np.asarray(fock_direction, dtype=float)
    if ds.shape != state.fock.shape or explicit.shape != state.fock.shape:
        raise ValueError("directional AO derivatives have incompatible dimensions")
    coefficients = state.coefficients
    energies = state.orbital_energies_hartree
    occupations = state.alpha_occupations + state.beta_occupations
    occupation_roundoff = np.max(
        np.abs(occupations - np.rint(occupations)), initial=0.0
    )
    if occupation_roundoff > 1.0e-10:
        raise ValueError("directional CPSCF requires integer final occupations")
    if occupation_roundoff <= 1.0e-11:
        occupations = np.rint(occupations)
    all_left, all_right = np.triu_indices(state.nmo, 1)
    all_gaps = energies[all_right] - energies[all_left]
    active = np.abs(occupations[all_right] - occupations[all_left]) > 1.0e-12
    left, right = all_left[active], all_right[active]
    gaps = all_gaps[active]
    threshold = float(degeneracy_tolerance_hartree)
    if np.any(np.abs(gaps) <= threshold):
        raise ValueError("directional CPSCF encountered an occupied/virtual degeneracy")
    occupation_difference = occupations[right] - occupations[left]
    occupied_count = int(np.count_nonzero(occupations > 1.0))
    virtual_count = state.nmo - occupied_count
    closed_shell_block = (
        occupied_count > 0
        and virtual_count > 0
        and np.all(np.abs(occupations[:occupied_count] - 2.0) <= 1.0e-10)
        and np.all(np.abs(occupations[occupied_count:]) <= 1.0e-10)
        and np.array_equal(
            left, np.repeat(np.arange(occupied_count), virtual_count)
        )
        and np.array_equal(
            right, np.tile(
                np.arange(occupied_count, state.nmo), occupied_count
            )
        )
    )
    if closed_shell_block:
        occupied_coefficients = coefficients[:, :occupied_count]
        virtual_coefficients = coefficients[:, occupied_count:]
        virtual_coefficients_t = virtual_coefficients.T
    ds_mo = coefficients.T @ ds @ coefficients
    metric_density_mo = -0.5 * (
        ds_mo * occupations[None, :] + occupations[:, None] * ds_mo
    )
    metric_density = coefficients @ metric_density_mo @ coefficients.T
    metric_kernel = np.asarray(fock_density_response(metric_density), dtype=float)
    explicit_total_mo = coefficients.T @ (explicit + metric_kernel) @ coefficients
    rhs = (
        explicit_total_mo[left, right]
        - 0.5 * (energies[left] + energies[right]) * ds_mo[left, right]
    ) / gaps

    def apply(rotations: np.ndarray) -> np.ndarray:
        if closed_shell_block:
            ov_density = (
                occupied_coefficients
                @ rotations.reshape(occupied_count, virtual_count)
            ) @ virtual_coefficients_t
            density = -2.0 * (ov_density + ov_density.T)
        else:
            density_mo = np.zeros((state.nmo, state.nmo), dtype=float)
            density_mo[left, right] = occupation_difference * rotations
            density_mo[right, left] = density_mo[left, right]
            density = coefficients @ density_mo @ coefficients.T
        kernel = np.asarray(fock_density_response(density), dtype=float)
        if closed_shell_block:
            kernel_ov = occupied_coefficients.T @ (
                kernel @ virtual_coefficients
            )
            return rotations - kernel_ov.reshape(-1) / gaps
        kernel_mo = coefficients.T @ kernel @ coefficients
        return rotations - kernel_mo[left, right] / gaps

    rotations, residual, iterations = _matrix_free_gmres(
        apply,
        rhs,
        tolerance=float(tolerance),
        maximum_iterations=int(maximum_iterations),
        single_pass=True,
    )
    density_mo = metric_density_mo.copy()
    density_mo[left, right] += occupation_difference * rotations
    density_mo[right, left] = density_mo[left, right]
    density = coefficients @ density_mo @ coefficients.T
    total_fock = explicit + np.asarray(fock_density_response(density), dtype=float)
    total_fock_mo = coefficients.T @ total_fock @ coefficients
    kappa_values = np.zeros_like(all_gaps)
    nondegenerate = np.abs(all_gaps) > threshold
    kappa_values[nondegenerate] = (
        total_fock_mo[all_left[nondegenerate], all_right[nondegenerate]]
        - 0.5
        * (energies[all_left[nondegenerate]] + energies[all_right[nondegenerate]])
        * ds_mo[all_left[nondegenerate], all_right[nondegenerate]]
    ) / all_gaps[nondegenerate]
    kappa = np.zeros((state.nmo, state.nmo), dtype=float)
    kappa[all_left, all_right] = kappa_values
    kappa[all_right, all_left] = -kappa_values
    connection = -0.5 * ds_mo + kappa
    coefficient_derivative = coefficients @ connection
    orbital_energy_derivative = np.diag(
        total_fock_mo - ds_mo * energies[None, :]
    )
    weighted_occupations = occupations * energies
    weighted = (
        (coefficient_derivative * weighted_occupations) @ coefficients.T
        + coefficients @ (coefficient_derivative * weighted_occupations).T
        + (coefficients * (occupations * orbital_energy_derivative))
        @ coefficients.T
    )
    return GroundStateDirectionalResponse(
        coefficient_derivative,
        density,
        0.5 * (weighted + weighted.T),
        0.5 * (total_fock + total_fock.T),
        orbital_energy_derivative,
        residual,
        iterations,
    )


@dataclass(frozen=True)
class GroundStateResponseHessian:
    """Orbital-relaxation/Pulay block of the Cartesian ground-state Hessian.

    This is not the complete Hessian: the fixed-density explicit nuclear
    second derivative must be added before vibrational analysis.
    """

    response_hessian_hartree_per_bohr2: np.ndarray
    reciprocity_error_hartree_per_bohr2: float
    maximum_cpscf_residual: float

    def __post_init__(self) -> None:
        hessian = np.array(
            self.response_hessian_hartree_per_bohr2,
            dtype=float,
            copy=True,
            order="C",
        )
        if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
            raise ValueError("ground-state response Hessian must be square")
        if not np.all(np.isfinite(hessian)):
            raise ValueError("ground-state response Hessian must be finite")
        hessian.setflags(write=False)
        object.__setattr__(self, "response_hessian_hartree_per_bohr2", hessian)
        for name in (
            "reciprocity_error_hartree_per_bohr2",
            "maximum_cpscf_residual",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError("ground-state response diagnostics must be nonnegative")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class GroundStateExplicitPairHVP:
    """Analytic components of the explicit fixed-density R,R product."""

    ao_h0_variational: np.ndarray
    repulsion: np.ndarray
    frozen_scc: np.ndarray
    frozen_d4_radial: np.ndarray
    frozen_d4_coordination: np.ndarray
    anisotropic: np.ndarray | None = None
    d4_coefficient_response: np.ndarray | None = None
    atm: np.ndarray | None = None

    def __post_init__(self) -> None:
        shape = None
        for name in (
            "ao_h0_variational",
            "repulsion",
            "frozen_scc",
            "frozen_d4_radial",
            "frozen_d4_coordination",
            "anisotropic",
            "d4_coefficient_response",
            "atm",
        ):
            raw = getattr(self, name)
            if raw is None:
                if shape is None:
                    continue
                raw = np.zeros(shape, dtype=float)
            value = np.array(raw, dtype=float, copy=True, order="C")
            if value.ndim != 2 or value.shape[1] != 3 or not np.all(np.isfinite(value)):
                raise ValueError("explicit pair Hessian products must be natoms x 3")
            if shape is not None and value.shape != shape:
                raise ValueError("explicit pair Hessian products have different shapes")
            shape = value.shape
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def total(self) -> np.ndarray:
        return (
            self.ao_h0_variational
            + self.repulsion
            + self.frozen_scc
            + self.frozen_d4_radial
            + self.frozen_d4_coordination
            + self.anisotropic
            + self.d4_coefficient_response
            + self.atm
        )


@dataclass(frozen=True)
class GroundStateHessian:
    """Complete assembled Cartesian ground-state Hessian and diagnostics."""

    hessian_hartree_per_bohr2: np.ndarray
    explicit_hessian_hartree_per_bohr2: np.ndarray
    response_hessian_hartree_per_bohr2: np.ndarray
    reciprocity_error_hartree_per_bohr2: float
    translational_residual_hartree_per_bohr2: float
    maximum_cpscf_residual: float

    def __post_init__(self) -> None:
        total = np.array(self.hessian_hartree_per_bohr2, dtype=float, copy=True)
        explicit = np.array(
            self.explicit_hessian_hartree_per_bohr2, dtype=float, copy=True
        )
        response = np.array(
            self.response_hessian_hartree_per_bohr2, dtype=float, copy=True
        )
        if total.ndim != 2 or total.shape[0] != total.shape[1]:
            raise ValueError("ground-state Hessian must be square")
        if explicit.shape != total.shape or response.shape != total.shape:
            raise ValueError("ground-state Hessian blocks have inconsistent dimensions")
        if not all(np.all(np.isfinite(item)) for item in (total, explicit, response)):
            raise ValueError("ground-state Hessian blocks must be finite")
        for item in (total, explicit, response):
            item.setflags(write=False)
        object.__setattr__(self, "hessian_hartree_per_bohr2", total)
        object.__setattr__(self, "explicit_hessian_hartree_per_bohr2", explicit)
        object.__setattr__(self, "response_hessian_hartree_per_bohr2", response)
        for name in (
            "reciprocity_error_hartree_per_bohr2",
            "translational_residual_hartree_per_bohr2",
            "maximum_cpscf_residual",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError("ground-state Hessian diagnostics must be nonnegative")
            object.__setattr__(self, name, value)


class GroundStateHessianOperator:
    """Matrix-free analytic Cartesian Hessian action."""

    def __init__(
        self,
        calculator: Any,
        state: GFN2OrbitalState,
        context: dict[str, object],
        *,
        cpscf_tolerance: float = 1.0e-10,
        cpscf_maximum_iterations: int = 200,
        degeneracy_tolerance_hartree: float = 1.0e-8,
    ) -> None:
        self.calculator = calculator
        self.state = state
        self.context = context
        self.cpscf_tolerance = float(cpscf_tolerance)
        self.cpscf_maximum_iterations = int(cpscf_maximum_iterations)
        self.degeneracy_tolerance_hartree = float(degeneracy_tolerance_hartree)
        self.dimension = 3 * len(state.atomic_numbers)
        self.fock_density_response = calculator.density_response_kernel(state)
        if self.cpscf_tolerance <= 0.0 or self.cpscf_maximum_iterations < 1:
            raise ValueError("matrix-free CPSCF controls must be positive")

    @property
    def shape(self) -> tuple[int, int]:
        return self.dimension, self.dimension

    def directional_response(
        self, vector: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, GroundStateDirectionalResponse]:
        """Return ``S'[v]``, fixed-density ``F'[v]`` and coupled response."""

        value = np.asarray(vector, dtype=float)
        if value.shape not in {(self.dimension,), (self.dimension // 3, 3)}:
            raise ValueError("nuclear trial vector has the wrong dimension")
        direction = value.reshape(-1, 3)
        overlap_direction, fock_direction = (
            self.calculator._ground_state_directional_derivative_matrices(
                self.state,
                direction,
                context=self.context,
            )
        )
        response = solve_ground_state_directional_response(
            self.state,
            overlap_direction,
            fock_direction,
            self.fock_density_response,
            tolerance=self.cpscf_tolerance,
            maximum_iterations=self.cpscf_maximum_iterations,
            degeneracy_tolerance_hartree=self.degeneracy_tolerance_hartree,
        )
        return overlap_direction, fock_direction, response

    def components_matvec(
        self,
        vector: np.ndarray,
        *,
        integral_directional_seconds: tuple[np.ndarray, np.ndarray] | None = None,
        alpb_geometry_hvp: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Return explicit and CPSCF Hessian actions without a dense pair matrix."""

        value = np.asarray(vector, dtype=float)
        _overlap_direction, _fock_direction, response = self.directional_response(
            value
        )
        direction = value.reshape(-1, 3)
        response_product = self.calculator.density_response_coordinate_vjp(
            self.state,
            response.density_derivative,
            response.weighted_density_derivative,
            _context=self.context,
        )
        explicit = self.calculator.ground_state_explicit_pair_hvp(
            self.state,
            direction,
            _context=self.context,
            _integral_directional_seconds=integral_directional_seconds,
            _alpb_geometry_hvp=alpb_geometry_hvp,
        ).total
        return (
            explicit.reshape(-1),
            response_product.reshape(-1),
            response.residual_norm,
        )

    def matvec(self, vector: np.ndarray) -> np.ndarray:
        explicit, response, _residual = self.components_matvec(vector)
        return explicit + response

    def apply_block(self, vectors: np.ndarray) -> np.ndarray:
        values = np.asarray(vectors, dtype=float)
        if values.ndim != 2 or values.shape[0] != self.dimension:
            raise ValueError("Hessian trial block has the wrong dimension")
        return np.column_stack(
            tuple(self.matvec(values[:, column]) for column in range(values.shape[1]))
        )

    def materialize(self, *, symmetrize: bool = True) -> np.ndarray:
        """Materialize only when explicitly requested for a small-system audit."""

        matrix = self.apply_block(np.eye(self.dimension))
        return 0.5 * (matrix + matrix.T) if symmetrize else matrix

    def exact_diagonal(self) -> np.ndarray:
        """Return an optional exact Davidson preconditioner without retaining H."""

        diagonal = np.empty(self.dimension, dtype=float)
        for coordinate in range(self.dimension):
            vector = np.zeros(self.dimension)
            vector[coordinate] = 1.0
            diagonal[coordinate] = self.matvec(vector)[coordinate]
        return diagonal

    def project(self, basis_cartesian_from_reduced: np.ndarray):
        return ProjectedGroundStateHessianOperator(
            self, basis_cartesian_from_reduced
        )

    def project_sycart(self, plan: Any):
        return self.project(np.asarray(plan.basis, dtype=float))


class ProjectedGroundStateHessianOperator:
    """Reduced operator ``B.T H B`` for SYCART or another orthonormal chart."""

    def __init__(
        self,
        cartesian_operator: GroundStateHessianOperator,
        basis_cartesian_from_reduced: np.ndarray,
    ) -> None:
        basis = np.array(basis_cartesian_from_reduced, dtype=float, copy=True)
        if basis.ndim != 2 or basis.shape[0] != cartesian_operator.dimension:
            raise ValueError("projected Hessian basis has the wrong shape")
        if not np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=2.0e-12):
            raise ValueError("projected Hessian basis must be orthonormal")
        basis.setflags(write=False)
        self.cartesian_operator = cartesian_operator
        self.basis = basis
        self.dimension = basis.shape[1]

    @property
    def shape(self) -> tuple[int, int]:
        return self.dimension, self.dimension

    def matvec(self, vector: np.ndarray) -> np.ndarray:
        value = np.asarray(vector, dtype=float)
        if value.shape != (self.dimension,):
            raise ValueError("projected Hessian vector has the wrong dimension")
        return self.basis.T @ self.cartesian_operator.matvec(self.basis @ value)

    def apply_block(self, vectors: np.ndarray) -> np.ndarray:
        values = np.asarray(vectors, dtype=float)
        if values.ndim != 2 or values.shape[0] != self.dimension:
            raise ValueError("projected Hessian block has the wrong dimension")
        cartesian = self.basis @ values
        return self.basis.T @ self.cartesian_operator.apply_block(cartesian)

    def materialize(self, *, symmetrize: bool = True) -> np.ndarray:
        matrix = self.apply_block(np.eye(self.dimension))
        return 0.5 * (matrix + matrix.T) if symmetrize else matrix

    def exact_diagonal(self) -> np.ndarray:
        diagonal = np.empty(self.dimension, dtype=float)
        for coordinate in range(self.dimension):
            vector = np.zeros(self.dimension)
            vector[coordinate] = 1.0
            diagonal[coordinate] = self.matvec(vector)[coordinate]
        return diagonal

    def lowest_eigenpairs(
        self,
        roots: int,
        *,
        diagonal: np.ndarray | None = None,
        backend: str = "auto",
        tolerance: float = 1.0e-7,
        maximum_iterations: int = 100,
        maximum_subspace: int | None = None,
    ):
        from matrix_numerics import solve_symmetric_matrix_free

        preconditioner = (
            self.exact_diagonal()
            if diagonal is None
            else np.asarray(diagonal, dtype=float)
        )
        if preconditioner.shape != (self.dimension,):
            raise ValueError("Hessian preconditioner has the wrong dimension")
        return solve_symmetric_matrix_free(
            self.matvec,
            preconditioner,
            roots=int(roots),
            block_matvec=self.apply_block,
            backend=backend,
            tolerance=float(tolerance),
            maximum_iterations=int(maximum_iterations),
            maximum_subspace=maximum_subspace,
        )


def solve_ground_state_coupled_response(
    state: GFN2OrbitalState,
    derivatives: GroundStateDerivativeMatrices,
    fock_density_response: Callable[[np.ndarray], np.ndarray],
    *,
    degeneracy_tolerance_hartree: float = 1.0e-8,
) -> GroundStateCoupledResponse:
    """Solve the dense canonical CPSCF reference equations.

    Occupations are held fixed.  This is the zero-temperature stationary-state
    derivative; a genuinely fractional final state requires an additional
    occupation/entropy response and is rejected.
    """

    overlap_derivative, explicit_fock_derivative = derivatives.validated(state)
    if not callable(fock_density_response):
        raise TypeError("ground-state CPSCF requires a callable Fock Jacobian")
    tolerance = float(degeneracy_tolerance_hartree)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("CPSCF degeneracy tolerance must be positive and finite")

    coefficients = state.coefficients
    energies = state.orbital_energies_hartree
    occupations = state.alpha_occupations + state.beta_occupations
    if np.max(np.abs(occupations - np.rint(occupations)), initial=0.0) > 1.0e-10:
        raise ValueError(
            "analytic ground-state response requires integer final occupations"
        )
    nmo = state.nmo
    left, right = np.triu_indices(nmo, 1)
    gaps = energies[right] - energies[left]
    active = np.abs(occupations[right] - occupations[left]) > 1.0e-12
    left = left[active]
    right = right[active]
    gaps = gaps[active]
    if np.any(np.abs(gaps) <= tolerance):
        pairs = tuple(
            (int(p), int(q))
            for p, q in zip(left[np.abs(gaps) <= tolerance], right[np.abs(gaps) <= tolerance])
        )
        raise ValueError(
            "ground-state CPSCF encountered an occupied/virtual degeneracy "
            f"for pairs {pairs[:4]}"
        )

    pair_count = len(left)
    response_matrix = np.eye(pair_count, dtype=float)
    occupation_difference = occupations[right] - occupations[left]
    for column, (p, q) in enumerate(zip(left, right, strict=True)):
        density_mo = np.zeros((nmo, nmo), dtype=float)
        density_mo[p, q] = occupation_difference[column]
        density_mo[q, p] = occupation_difference[column]
        density_ao = coefficients @ density_mo @ coefficients.T
        kernel_ao = np.asarray(fock_density_response(density_ao), dtype=float)
        if kernel_ao.shape != state.fock.shape or not np.all(np.isfinite(kernel_ao)):
            raise ValueError("ground-state Fock Jacobian returned an invalid matrix")
        kernel_mo = coefficients.T @ kernel_ao @ coefficients
        response_matrix[:, column] -= kernel_mo[left, right] / gaps

    axes = overlap_derivative.shape[:2]
    dc_all = np.empty((*axes, state.nao, nmo), dtype=float)
    dp_all = np.empty((*axes, state.nao, state.nao), dtype=float)
    dw_all = np.empty_like(dp_all)
    df_all = np.empty_like(explicit_fock_derivative)
    de_all = np.empty((*axes, nmo), dtype=float)
    residuals = np.empty(axes, dtype=float)
    weighted_occupations = occupations * energies

    for atom in range(axes[0]):
        for component in range(axes[1]):
            ds = overlap_derivative[atom, component]
            ds_mo = coefficients.T @ ds @ coefficients
            metric_density_mo = -0.5 * (
                ds_mo * occupations[None, :] + occupations[:, None] * ds_mo
            )
            metric_density_ao = coefficients @ metric_density_mo @ coefficients.T
            metric_kernel = np.asarray(
                fock_density_response(metric_density_ao), dtype=float
            )
            explicit_total = explicit_fock_derivative[atom, component] + metric_kernel
            explicit_mo = coefficients.T @ explicit_total @ coefficients
            rhs = (
                explicit_mo[left, right]
                - 0.5 * (energies[left] + energies[right]) * ds_mo[left, right]
            ) / gaps
            rotations = np.linalg.solve(response_matrix, rhs)

            density_mo = metric_density_mo.copy()
            density_mo[left, right] += occupation_difference * rotations
            density_mo[right, left] = density_mo[left, right]
            dp = coefficients @ density_mo @ coefficients.T
            df = explicit_fock_derivative[atom, component] + np.asarray(
                fock_density_response(dp), dtype=float
            )
            df_mo = coefficients.T @ df @ coefficients
            # Equal-occupation rotations do not enter P', hence they are not
            # CPSCF unknowns.  They are nevertheless required by C' and W'.
            # Recover their nondegenerate gauge from the differentiated
            # generalized eigenproblem after the physical density response
            # has converged.
            all_left, all_right = np.triu_indices(nmo, 1)
            all_gaps = energies[all_right] - energies[all_left]
            all_kappa = np.zeros_like(all_gaps)
            nondegenerate = np.abs(all_gaps) > tolerance
            all_kappa[nondegenerate] = (
                df_mo[all_left[nondegenerate], all_right[nondegenerate]]
                - 0.5
                * (energies[all_left[nondegenerate]] + energies[all_right[nondegenerate]])
                * ds_mo[all_left[nondegenerate], all_right[nondegenerate]]
            ) / all_gaps[nondegenerate]
            kappa = np.zeros((nmo, nmo), dtype=float)
            kappa[all_left, all_right] = all_kappa
            kappa[all_right, all_left] = -all_kappa
            connection = -0.5 * ds_mo + kappa
            dc = coefficients @ connection
            de = np.diag(df_mo - ds_mo * energies[None, :])
            # W = C diag(n epsilon) C^T.  Differentiate the stationary
            # energy-weighted density without introducing orbital gauges.
            dw = (
                (dc * weighted_occupations) @ coefficients.T
                + coefficients @ (dc * weighted_occupations).T
                + (coefficients * (occupations * de)) @ coefficients.T
            )
            eigen_residual = (
                df @ coefficients
                + state.fock @ dc
                - (ds @ coefficients) * energies
                - (state.overlap @ dc) * energies
                - (state.overlap @ coefficients) * de
            )
            residuals[atom, component] = max(
                float(np.linalg.norm(response_matrix @ rotations - rhs)),
                float(np.linalg.norm(eigen_residual)),
            )
            dc_all[atom, component] = dc
            dp_all[atom, component] = dp
            dw_all[atom, component] = 0.5 * (dw + dw.T)
            df_all[atom, component] = df
            de_all[atom, component] = de

    return GroundStateCoupledResponse(
        dc_all, dp_all, dw_all, df_all, de_all, residuals
    )


__all__ = [
    "GroundStateCoupledResponse",
    "GroundStateDerivativeMatrices",
    "GroundStateDirectionalResponse",
    "GroundStateExplicitPairHVP",
    "GroundStateHessian",
    "GroundStateHessianOperator",
    "GroundStateResponseHessian",
    "ProjectedGroundStateHessianOperator",
    "solve_ground_state_coupled_response",
    "solve_ground_state_directional_response",
]
