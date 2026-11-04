"""Matrix-free analytic second derivatives of variational SF-TDA roots.

The nuclear Hessian of a TDA root is evaluated as an operator.  For one
nuclear direction ``v`` the implementation solves the projected eigenvector
response equation

``Q (A - omega I) Q X[v] = -Q A[v] X``

and contracts

``omega[q,v] = X.T A[q,v] X + 2 X.T A[q] X[v]``.

This module owns the response and contraction layer.  The supplied first and
second matrix derivatives must already include the AO metric, orbital/SCC
response, selected-space downfolding, and any external-charge contribution.
Keeping the second derivative as a directional callback avoids a
``(3N)^2 * ncsf^2`` tensor and is the contract used by the resident sparse
TDA-xTB derivative kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np

from .sf_tda import SFTDAResult
from .sf_tda_gradient import _matrix_free_gmres


class SymmetricTDAOperator(Protocol):
    """Minimal matrix-free TDA operator required by the Hessian layer."""

    @property
    def size(self) -> int: ...

    def apply(self, vector: np.ndarray) -> np.ndarray: ...


class SFTDADerivativeKernel(Protocol):
    """Matrix-free nuclear derivative contract for a TDA operator."""

    dimension: int
    size: int

    def matrix_direction(self, direction: np.ndarray) -> np.ndarray: ...

    def gradient_vjp(self, left: np.ndarray, right: np.ndarray) -> np.ndarray: ...

    def explicit_hessian_action(
        self, root: np.ndarray, direction: np.ndarray
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class PairPackedSFTDAContribution:
    """One pair-packed term in the differentiated TDA Lagrangian.

    The four required production terms are named ``ao``, ``scc_multipole``,
    ``embedding``, and ``orbital_adjoint``.  An absent physical contribution
    (for example embedding without point charges) is represented by a zero
    callback, never by removing the term from the contract.
    """

    name: str
    matrix_direction: Callable[[np.ndarray], np.ndarray]
    gradient_vjp: Callable[[np.ndarray, np.ndarray], np.ndarray]
    explicit_hessian_action: Callable[[np.ndarray, np.ndarray], np.ndarray]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("SF-TDA derivative contribution requires a name")
        if not all(
            callable(value)
            for value in (
                self.matrix_direction,
                self.gradient_vjp,
                self.explicit_hessian_action,
            )
        ):
            raise TypeError("SF-TDA derivative contribution callbacks must be callable")


@dataclass(frozen=True)
class PairPackedSFTDAFusedActions:
    """Single-pass actions over all differentiated physical contributions.

    Production kernels use these callbacks to traverse the persistent pair
    pattern once.  The named contribution callbacks remain available as a
    reference/fallback path and document that AO, SCC/multipole, embedding,
    and orbital-adjoint terms are all present.
    """

    matrix_direction: Callable[[np.ndarray], np.ndarray]
    hessian_terms: Callable[
        [np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]
    ]
    symmetric_by_construction: bool = False

    def __post_init__(self) -> None:
        if not callable(self.matrix_direction) or not callable(self.hessian_terms):
            raise TypeError("fused SF-TDA derivative actions must be callable")


class PairPackedSFTDADerivativeKernel:
    """Fuse AO, SCC/multipole, embedding, and orbital-adjoint actions.

    Every callback consumes or produces only a CSF vector/matrix, a nuclear
    vector, or values on the persistent AO pair pattern.  The composite never
    constructs an AO-square derivative or a nuclear second-derivative tensor.
    """

    REQUIRED_CONTRIBUTIONS = frozenset(
        ("ao", "scc_multipole", "embedding", "orbital_adjoint")
    )

    def __init__(
        self,
        dimension: int,
        size: int,
        contributions: Sequence[PairPackedSFTDAContribution],
        *,
        fused_actions: PairPackedSFTDAFusedActions | None = None,
    ) -> None:
        self.dimension = int(dimension)
        self.size = int(size)
        if self.dimension < 1 or self.size < 1:
            raise ValueError("pair-packed SF-TDA derivative dimensions must be positive")
        items = tuple(contributions)
        names = tuple(item.name for item in items)
        if len(set(names)) != len(names):
            raise ValueError("SF-TDA derivative contribution names must be unique")
        missing = self.REQUIRED_CONTRIBUTIONS.difference(names)
        if missing:
            raise ValueError(
                "pair-packed SF-TDA derivative kernel is incomplete: "
                + ", ".join(sorted(missing))
            )
        self.contributions = items
        self.fused_actions = fused_actions

    def matrix_direction(self, direction: np.ndarray) -> np.ndarray:
        value = np.asarray(direction, dtype=float)
        if value.shape != (self.dimension,):
            raise ValueError("SF-TDA nuclear direction has the wrong dimension")
        if self.fused_actions is not None:
            result = np.asarray(
                self.fused_actions.matrix_direction(value), dtype=float
            )
            if result.shape != (self.size, self.size) or not np.all(
                np.isfinite(result)
            ):
                raise ValueError("fused kernel returned an invalid TDA matrix direction")
        else:
            result = np.zeros((self.size, self.size), dtype=float)
            for contribution in self.contributions:
                block = np.asarray(contribution.matrix_direction(value), dtype=float)
                if block.shape != result.shape or not np.all(np.isfinite(block)):
                    raise ValueError(
                        f"{contribution.name} returned an invalid TDA matrix direction"
                    )
                result += block
        certified_symmetric = (
            self.fused_actions is not None
            and self.fused_actions.symmetric_by_construction
        )
        if not certified_symmetric and float(
            np.max(np.abs(result - result.T))
        ) > 2.0e-11:
            raise ValueError("assembled TDA matrix direction is not symmetric")
        return result

    def hessian_terms(
        self,
        root: np.ndarray,
        response: np.ndarray,
        direction: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return explicit and relaxation terms in one pair-pattern traversal."""

        amplitude = np.asarray(root, dtype=float)
        response_value = np.asarray(response, dtype=float)
        nuclear_direction = np.asarray(direction, dtype=float)
        if (
            amplitude.shape != (self.size,)
            or response_value.shape != (self.size,)
            or nuclear_direction.shape != (self.dimension,)
        ):
            raise ValueError("fused SF-TDA Hessian action has incompatible dimensions")
        if self.fused_actions is None:
            return (
                self.explicit_hessian_action(amplitude, nuclear_direction),
                self.gradient_vjp(amplitude, response_value),
            )
        terms = self.fused_actions.hessian_terms(
            amplitude, response_value, nuclear_direction
        )
        if not isinstance(terms, tuple) or len(terms) != 2:
            raise ValueError("fused SF-TDA Hessian callback must return two terms")
        explicit = np.asarray(terms[0], dtype=float)
        relaxation = np.asarray(terms[1], dtype=float)
        expected = (self.dimension,)
        if (
            explicit.shape != expected
            or relaxation.shape != expected
            or not np.all(np.isfinite(explicit))
            or not np.all(np.isfinite(relaxation))
        ):
            raise ValueError("fused kernel returned invalid SF-TDA Hessian terms")
        return explicit, relaxation

    def gradient_vjp(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        lhs = np.asarray(left, dtype=float)
        rhs = np.asarray(right, dtype=float)
        if lhs.shape != (self.size,) or rhs.shape != (self.size,):
            raise ValueError("SF-TDA derivative VJP amplitudes have the wrong dimension")
        result = np.zeros(self.dimension, dtype=float)
        for contribution in self.contributions:
            value = np.asarray(contribution.gradient_vjp(lhs, rhs), dtype=float)
            if value.shape != result.shape or not np.all(np.isfinite(value)):
                raise ValueError(
                    f"{contribution.name} returned an invalid nuclear derivative VJP"
                )
            result += value
        return result

    def explicit_hessian_action(
        self, root: np.ndarray, direction: np.ndarray
    ) -> np.ndarray:
        amplitude = np.asarray(root, dtype=float)
        value = np.asarray(direction, dtype=float)
        if amplitude.shape != (self.size,) or value.shape != (self.dimension,):
            raise ValueError("SF-TDA explicit Hessian action has incompatible dimensions")
        result = np.zeros(self.dimension, dtype=float)
        for contribution in self.contributions:
            action = np.asarray(
                contribution.explicit_hessian_action(amplitude, value), dtype=float
            )
            if action.shape != result.shape or not np.all(np.isfinite(action)):
                raise ValueError(
                    f"{contribution.name} returned an invalid Hessian action"
                )
            result += action
        return result


@dataclass(frozen=True)
class SFTDADirectionalHessianResult:
    """One analytic excited-state Hessian-vector product and diagnostics."""

    excitation_hessian_vector_hartree_per_bohr2: np.ndarray
    total_hessian_vector_hartree_per_bohr2: np.ndarray
    amplitude_response: np.ndarray
    response_residual_norm: float
    response_iterations: int

    def __post_init__(self) -> None:
        excitation = np.array(
            self.excitation_hessian_vector_hartree_per_bohr2,
            dtype=float,
            copy=True,
        )
        total = np.array(
            self.total_hessian_vector_hartree_per_bohr2, dtype=float, copy=True
        )
        response = np.array(self.amplitude_response, dtype=float, copy=True)
        if excitation.ndim != 1 or total.shape != excitation.shape:
            raise ValueError("SF-TDA Hessian-vector dimensions are inconsistent")
        if response.ndim != 1:
            raise ValueError("SF-TDA amplitude response must be a vector")
        if not all(np.all(np.isfinite(value)) for value in (excitation, total, response)):
            raise ValueError("SF-TDA Hessian-vector result must be finite")
        residual = float(self.response_residual_norm)
        iterations = int(self.response_iterations)
        if not np.isfinite(residual) or residual < 0.0 or iterations < 0:
            raise ValueError("SF-TDA response diagnostics are invalid")
        for value in (excitation, total, response):
            value.setflags(write=False)
        object.__setattr__(
            self, "excitation_hessian_vector_hartree_per_bohr2", excitation
        )
        object.__setattr__(self, "total_hessian_vector_hartree_per_bohr2", total)
        object.__setattr__(self, "amplitude_response", response)
        object.__setattr__(self, "response_residual_norm", residual)
        object.__setattr__(self, "response_iterations", iterations)


class SFTDAHessianOperator:
    """Analytic matrix-free Hessian of one variational SF-TDA state.

    ``matrix_first_derivative`` has shape ``(3N,ncsf,ncsf)`` or
    ``(N,3,ncsf,ncsf)``.  ``matrix_second_derivative_action(v)`` returns the
    derivative along ``v`` of every first-derivative matrix with the same
    leading nuclear dimension.  Thus only ``O(3N*ncsf^2)`` derivative storage
    is required, and pair-packed implementations may return the contraction
    from a transient buffer.
    """

    def __init__(
        self,
        tda_operator: SymmetricTDAOperator,
        result: SFTDAResult,
        matrix_first_derivative: np.ndarray | None,
        matrix_second_derivative_action: Callable[[np.ndarray], np.ndarray] | None,
        *,
        derivative_kernel: SFTDADerivativeKernel | None = None,
        root: int = 0,
        ground_state_hessian_operator: object | None = None,
        response_tolerance: float = 1.0e-10,
        maximum_response_iterations: int = 200,
        root_gap_tolerance_hartree: float = 1.0e-7,
        response_deflation_roots: int = 16,
        response_diagonal_preconditioning: bool = True,
    ) -> None:
        if not callable(getattr(tda_operator, "apply", None)):
            raise TypeError("SF-TDA Hessian requires a matrix-free TDA operator")
        if derivative_kernel is None and not callable(matrix_second_derivative_action):
            raise TypeError("SF-TDA Hessian requires an analytic second-derivative action")
        root_index = int(root)
        if root_index < 0 or root_index >= len(result.excitation_energies_hartree):
            raise IndexError("SF-TDA Hessian root index is out of range")
        amplitudes = np.asarray(result.amplitudes, dtype=float)
        size = int(tda_operator.size)
        if amplitudes.ndim != 2 or amplitudes.shape[1] != size:
            raise ValueError("SF-TDA result and Hessian operator spaces differ")
        if derivative_kernel is None:
            if matrix_first_derivative is None:
                raise TypeError("SF-TDA Hessian requires first matrix derivatives")
            first = np.asarray(matrix_first_derivative, dtype=float)
            if first.ndim == 4 and first.shape[1] == 3:
                first = first.reshape((-1, size, size))
            if first.ndim != 3 or first.shape[1:] != (size, size):
                raise ValueError("SF-TDA first matrix derivative has incompatible dimensions")
            if not np.all(np.isfinite(first)):
                raise ValueError("SF-TDA first matrix derivative must be finite")
            if not np.allclose(first, first.swapaxes(-1, -2), atol=2.0e-11, rtol=0.0):
                raise ValueError("SF-TDA first matrix derivatives must be symmetric")
            dimension = int(first.shape[0])
        else:
            if int(derivative_kernel.size) != size:
                raise ValueError("SF-TDA derivative kernel and operator spaces differ")
            dimension = int(derivative_kernel.dimension)
            first = np.empty((0, size, size), dtype=float)
        if response_tolerance <= 0.0 or maximum_response_iterations < 1:
            raise ValueError("SF-TDA response controls must be positive")
        if root_gap_tolerance_hartree <= 0.0:
            raise ValueError("SF-TDA root-gap tolerance must be positive")
        if int(response_deflation_roots) < 0:
            raise ValueError("SF-TDA response deflation count must be nonnegative")

        vector = amplitudes[root_index].copy()
        norm = float(np.linalg.norm(vector))
        if norm <= np.finfo(float).tiny:
            raise ValueError("SF-TDA root amplitude has zero norm")
        vector /= norm
        energies = np.asarray(result.excitation_energies_hartree, dtype=float)
        if len(energies) > 1:
            gaps = np.abs(energies - energies[root_index])
            gaps[root_index] = np.inf
            if float(np.min(gaps)) <= float(root_gap_tolerance_hartree):
                raise ValueError(
                    "state-specific SF-TDA Hessian is undefined at an unresolved root degeneracy"
                )

        first = np.array(first, copy=True, order="C")
        first.setflags(write=False)
        vector.setflags(write=False)
        self.tda_operator = tda_operator
        self.result = result
        self.root = root_index
        self.root_energy_hartree = float(energies[root_index])
        self.root_amplitude = vector
        self.matrix_first_derivative = first
        self.matrix_second_derivative_action = matrix_second_derivative_action
        self.derivative_kernel = derivative_kernel
        self.ground_state_hessian_operator = ground_state_hessian_operator
        self.response_tolerance = float(response_tolerance)
        self.maximum_response_iterations = int(maximum_response_iterations)
        self.root_gap_tolerance_hartree = float(root_gap_tolerance_hartree)
        self.response_deflation_roots = int(response_deflation_roots)
        self.response_diagonal_preconditioning = bool(
            response_diagonal_preconditioning
        )
        self.dimension = dimension

        gaps = energies - self.root_energy_hartree
        residual_limit = max(10.0 * self.response_tolerance, 1.0e-12)
        eligible = [
            index
            for index in np.argsort(np.abs(gaps), kind="stable")
            if index != self.root
            and abs(float(gaps[index])) > self.root_gap_tolerance_hartree
            and float(result.residual_norms[index]) <= residual_limit
        ][: self.response_deflation_roots]
        deflation_vectors = np.empty((size, 0), dtype=float)
        deflation_gaps = np.empty(0, dtype=float)
        if eligible:
            candidate_vectors = amplitudes[eligible].T.copy()
            candidate_vectors -= vector[:, None] * (vector @ candidate_vectors)[None, :]
            norms = np.linalg.norm(candidate_vectors, axis=0)
            if np.all(norms > 1.0e-10):
                candidate_vectors /= norms[None, :]
                if np.allclose(
                    candidate_vectors.T @ candidate_vectors,
                    np.eye(len(eligible)),
                    atol=2.0e-8,
                    rtol=0.0,
                ):
                    deflation_vectors = candidate_vectors
                    deflation_gaps = gaps[eligible].copy()
        deflation_vectors.setflags(write=False)
        deflation_gaps.setflags(write=False)
        self.response_deflation_vectors = deflation_vectors
        self.response_deflation_gaps_hartree = deflation_gaps

        diagonal = getattr(tda_operator, "diagonal_hartree", None)
        if self.response_diagonal_preconditioning and diagonal is not None:
            response_diagonal = np.asarray(diagonal, dtype=float) - self.root_energy_hartree
            if response_diagonal.shape != (size,) or not np.all(
                np.isfinite(response_diagonal)
            ):
                raise ValueError("SF-TDA response diagonal is invalid")
            response_diagonal = response_diagonal.copy()
        else:
            response_diagonal = np.empty(0, dtype=float)
        response_diagonal.setflags(write=False)
        self.response_diagonal_hartree = response_diagonal
        if ground_state_hessian_operator is not None:
            shape = getattr(ground_state_hessian_operator, "shape", None)
            if shape != (self.dimension, self.dimension):
                raise ValueError("ground- and excited-state Hessian dimensions differ")

    @property
    def shape(self) -> tuple[int, int]:
        return self.dimension, self.dimension

    def _project(self, vector: np.ndarray) -> np.ndarray:
        root = self.root_amplitude
        return vector - root * float(np.dot(root, vector))

    def amplitude_directional_response(
        self, direction: np.ndarray
    ) -> tuple[np.ndarray, float, int]:
        """Solve the state-following TDA eigenvector response for one direction."""

        value = np.asarray(direction, dtype=float)
        if value.shape != (self.dimension,):
            raise ValueError("SF-TDA nuclear direction has the wrong dimension")
        first_direction = (
            np.einsum(
                "q,qij->ij", value, self.matrix_first_derivative, optimize=True
            )
            if self.derivative_kernel is None
            else np.asarray(self.derivative_kernel.matrix_direction(value), dtype=float)
        )
        right_hand_side = -self._project(first_direction @ self.root_amplitude)

        def apply(trial: np.ndarray) -> np.ndarray:
            projected = self._project(np.asarray(trial, dtype=float))
            image = np.asarray(self.tda_operator.apply(projected), dtype=float)
            if image.shape != projected.shape or not np.all(np.isfinite(image)):
                raise ValueError("SF-TDA operator returned an invalid response vector")
            return self._project(image - self.root_energy_hartree * projected)

        deflation = self.response_deflation_vectors

        def project_response_space(trial: np.ndarray) -> np.ndarray:
            projected = self._project(np.asarray(trial, dtype=float))
            if deflation.shape[1]:
                projected -= deflation @ (deflation.T @ projected)
            return projected

        if deflation.shape[1]:
            coefficients = (deflation.T @ right_hand_side) / (
                self.response_deflation_gaps_hartree
            )
            locked_response = deflation @ coefficients
            iterative_right_hand_side = project_response_space(
                right_hand_side - apply(locked_response)
            )
        else:
            locked_response = np.zeros_like(right_hand_side)
            iterative_right_hand_side = right_hand_side

        def iterative_apply(trial: np.ndarray) -> np.ndarray:
            return project_response_space(apply(project_response_space(trial)))

        right_preconditioner = None
        full_right_preconditioner = None
        if self.response_diagonal_hartree.size:
            floor = max(self.root_gap_tolerance_hartree, 1.0e-10)
            diagonal = self.response_diagonal_hartree
            safe_diagonal = np.where(
                np.abs(diagonal) < floor,
                np.copysign(floor, diagonal + 1.0e-300),
                diagonal,
            )

            def right_preconditioner(trial: np.ndarray) -> np.ndarray:
                return project_response_space(
                    project_response_space(trial) / safe_diagonal
                )

            def full_right_preconditioner(trial: np.ndarray) -> np.ndarray:
                projected = self._project(np.asarray(trial, dtype=float))
                return self._project(projected / safe_diagonal)

        def solve_without_deflation() -> tuple[np.ndarray, float, int]:
            try:
                return _matrix_free_gmres(
                    apply,
                    right_hand_side,
                    tolerance=self.response_tolerance,
                    maximum_iterations=self.maximum_response_iterations,
                    right_preconditioner=full_right_preconditioner,
                )
            except RuntimeError:
                if full_right_preconditioner is None:
                    raise
                return _matrix_free_gmres(
                    apply,
                    right_hand_side,
                    tolerance=self.response_tolerance,
                    maximum_iterations=self.maximum_response_iterations,
                )

        try:
            iterative_response, _, iterations = _matrix_free_gmres(
                iterative_apply,
                iterative_right_hand_side,
                tolerance=self.response_tolerance,
                maximum_iterations=self.maximum_response_iterations,
                right_preconditioner=right_preconditioner,
            )
        except RuntimeError:
            response, residual, iterations = solve_without_deflation()
            return self._project(response), residual, iterations
        response = self._project(locked_response + iterative_response)
        residual = float(np.linalg.norm(apply(response) - right_hand_side))
        if residual > self.response_tolerance * max(
            1.0, float(np.linalg.norm(right_hand_side))
        ):
            response, residual, iterations = solve_without_deflation()
        return response, residual, iterations

    def directional_result(self, direction: np.ndarray) -> SFTDADirectionalHessianResult:
        """Return excitation and total state-specific Hessian actions."""

        value = np.asarray(direction, dtype=float)
        if value.shape != (self.dimension,):
            raise ValueError("SF-TDA nuclear direction has the wrong dimension")
        response, residual, iterations = self.amplitude_directional_response(value)
        root = self.root_amplitude
        if self.derivative_kernel is None:
            second = np.asarray(self.matrix_second_derivative_action(value), dtype=float)
            if second.ndim == 4 and second.shape[1] == 3:
                second = second.reshape(self.matrix_first_derivative.shape)
            if second.shape != self.matrix_first_derivative.shape:
                raise ValueError("SF-TDA second matrix derivative action has the wrong shape")
            if not np.all(np.isfinite(second)) or not np.allclose(
                second, second.swapaxes(-1, -2), atol=2.0e-11, rtol=0.0
            ):
                raise ValueError("SF-TDA second matrix derivative action must be finite and symmetric")
            explicit = np.einsum("i,qij,j->q", root, second, root, optimize=True)
            relaxation = 2.0 * np.einsum(
                "i,qij,j->q", root, self.matrix_first_derivative, response, optimize=True
            )
        else:
            fused_terms = getattr(self.derivative_kernel, "hessian_terms", None)
            if callable(fused_terms):
                explicit, relaxation_vjp = fused_terms(root, response, value)
                relaxation = 2.0 * relaxation_vjp
            else:
                explicit = self.derivative_kernel.explicit_hessian_action(root, value)
                relaxation = 2.0 * self.derivative_kernel.gradient_vjp(root, response)
        excitation = explicit + relaxation
        if self.ground_state_hessian_operator is None:
            total = excitation.copy()
        else:
            total = excitation + np.asarray(
                self.ground_state_hessian_operator.matvec(value), dtype=float
            )
        return SFTDADirectionalHessianResult(
            excitation, total, response, residual, iterations
        )

    def excitation_matvec(self, direction: np.ndarray) -> np.ndarray:
        return self.directional_result(direction).excitation_hessian_vector_hartree_per_bohr2

    def matvec(self, direction: np.ndarray) -> np.ndarray:
        return self.directional_result(direction).total_hessian_vector_hartree_per_bohr2

    def apply_block(self, vectors: np.ndarray) -> np.ndarray:
        values = np.asarray(vectors, dtype=float)
        if values.ndim != 2 or values.shape[0] != self.dimension:
            raise ValueError("SF-TDA Hessian trial block has the wrong dimension")
        return np.column_stack(
            tuple(self.matvec(values[:, column]) for column in range(values.shape[1]))
        )

    def materialize(self, *, symmetrize: bool = True) -> np.ndarray:
        """Materialize the Hessian only for small-system analysis or export."""

        matrix = self.apply_block(np.eye(self.dimension))
        return 0.5 * (matrix + matrix.T) if symmetrize else matrix

    def project(self, basis_cartesian_from_reduced: np.ndarray):
        """Project the state-specific Hessian into an orthonormal chart."""

        return ProjectedSFTDAHessianOperator(self, basis_cartesian_from_reduced)

    def project_sycart(self, plan: object):
        """Project into the immutable symmetry-adapted Cartesian chart."""

        return self.project(np.asarray(getattr(plan, "basis"), dtype=float))


class ProjectedSFTDAHessianOperator:
    """Reduced state-specific operator ``B.T H B`` for SYCART/SONIC work."""

    def __init__(
        self,
        cartesian_operator: SFTDAHessianOperator,
        basis_cartesian_from_reduced: np.ndarray,
    ) -> None:
        basis = np.array(basis_cartesian_from_reduced, dtype=float, copy=True)
        if basis.ndim != 2 or basis.shape[0] != cartesian_operator.dimension:
            raise ValueError("projected SF-TDA Hessian basis has the wrong shape")
        if not np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=2.0e-12):
            raise ValueError("projected SF-TDA Hessian basis must be orthonormal")
        basis.setflags(write=False)
        self.cartesian_operator = cartesian_operator
        self.basis = basis
        self.dimension = basis.shape[1]

    @property
    def shape(self) -> tuple[int, int]:
        return self.dimension, self.dimension

    def matvec(self, direction: np.ndarray) -> np.ndarray:
        value = np.asarray(direction, dtype=float)
        if value.shape != (self.dimension,):
            raise ValueError("projected SF-TDA Hessian vector has the wrong dimension")
        return self.basis.T @ self.cartesian_operator.matvec(self.basis @ value)

    def apply_block(self, vectors: np.ndarray) -> np.ndarray:
        values = np.asarray(vectors, dtype=float)
        if values.ndim != 2 or values.shape[0] != self.dimension:
            raise ValueError("projected SF-TDA Hessian block has the wrong dimension")
        return self.basis.T @ self.cartesian_operator.apply_block(self.basis @ values)

    def materialize(self, *, symmetrize: bool = True) -> np.ndarray:
        matrix = self.apply_block(np.eye(self.dimension))
        return 0.5 * (matrix + matrix.T) if symmetrize else matrix


__all__ = [
    "PairPackedSFTDAContribution",
    "PairPackedSFTDADerivativeKernel",
    "PairPackedSFTDAFusedActions",
    "ProjectedSFTDAHessianOperator",
    "SFTDADirectionalHessianResult",
    "SFTDAHessianOperator",
    "SFTDADerivativeKernel",
    "SymmetricTDAOperator",
]
