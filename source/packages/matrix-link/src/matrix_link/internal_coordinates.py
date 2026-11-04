"""LINK-owned internal-to-Cartesian realization services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import splu

try:
    from . import _optimizer_native
except ImportError:  # pragma: no cover - portable pure-Python installation
    _optimizer_native = None

from matrix_smith import (
    SparseBDirectSumCertificate,
    SparseBMatrix,
    verify_sparse_b_direct_sum,
)


@dataclass(frozen=True)
class InternalCoordinateBackTransform:
    coordinates_angstrom: np.ndarray
    values: np.ndarray
    residual: np.ndarray
    iterations: int
    converged: bool
    cartesian_from_q: np.ndarray | None


@dataclass(frozen=True)
class ProjectorSecantUpdate:
    cartesian_from_q: np.ndarray | None
    relative_error: float
    accepted: bool


class CertifiedDirectSumInverseError(ValueError):
    """Raised when fixed columns destroy a certified full-rank block solve."""


@dataclass(frozen=True)
class SparseGaugeSymbolicPlan:
    """Geometry-independent sparse gauge/factorization structure."""

    support_indices: np.ndarray
    support_indptr: np.ndarray
    gauge_columns: tuple[int, ...]
    gauge_free_columns: np.ndarray
    column_permutation: np.ndarray
    csc_indices: np.ndarray
    csc_indptr: np.ndarray
    raw_positions: np.ndarray


class _PermutedSparseLU:
    """Expose an LU of ``A[:, permutation]`` as an LU of ``A``."""

    def __init__(self, factor, permutation: np.ndarray):
        self._factor = factor
        self._permutation = np.asarray(permutation, dtype=int)
        self.L = factor.L
        self.U = factor.U

    def solve(self, rhs: np.ndarray, trans: str = "N") -> np.ndarray:
        vector = np.asarray(rhs, dtype=float)
        if trans == "N":
            permuted = self._factor.solve(vector)
            result = np.empty_like(permuted)
            result[self._permutation] = permuted
            return result
        if trans == "T":
            return self._factor.solve(vector[self._permutation], trans="T")
        raise ValueError("sparse gauge solve supports only N and T")


def native_sonic_backtransform_available() -> bool:
    """Return whether the compiled sparse SONIC controller is loaded."""

    return bool(
        _optimizer_native is not None
        and hasattr(_optimizer_native, "sonic_sparse_backtransform")
    )


def _rigid_rotation_axes(
    centered_coordinates: np.ndarray,
    gauge_count: int,
) -> tuple[np.ndarray, ...]:
    """Return the independent rotational axes of a linear/nonlinear geometry."""

    centered = np.asarray(centered_coordinates, dtype=float)
    inertia = (
        float(np.sum(centered * centered)) * np.eye(3, dtype=float)
        - centered.T @ centered
    )
    values, vectors = np.linalg.eigh(inertia)
    rotation_count = max(0, int(gauge_count) - 3)
    order = np.argsort(values)[::-1][:rotation_count]
    return tuple(np.asarray(vectors[:, index], dtype=float) for index in order)


def _rigid_mode_vector(
    centered_coordinates: np.ndarray,
    mode: int,
    rotation_axes: tuple[np.ndarray, ...],
) -> np.ndarray:
    """Return one rigid displacement mode without building the full mode matrix."""

    centered = np.asarray(centered_coordinates, dtype=float)
    values = np.zeros_like(centered)
    if mode < 3:
        values[:, int(mode)] = 1.0
    else:
        values = np.cross(rotation_axes[int(mode) - 3], centered)
    return values.reshape(-1)


def _independent_rigid_gauge_columns(
    centered_coordinates: np.ndarray,
    rotation_axes: tuple[np.ndarray, ...],
) -> tuple[int, ...]:
    """Choose an independent Cartesian gauge in O(N) storage and work.

    Each Cartesian column is represented by only its at-most-six coefficients
    in the rigid-motion rows.  Pivoted modified Gram--Schmidt is therefore
    performed in the fixed external-mode dimension, never on a ``6 x 3N``
    materialization.
    """

    centered = np.asarray(centered_coordinates, dtype=float)
    count = 3 + len(rotation_axes)
    basis: list[np.ndarray] = []
    selected: list[int] = []
    def coefficients(position: np.ndarray, component: int) -> np.ndarray:
        result = np.zeros(count, dtype=float)
        result[component] = 1.0
        x, y, z = (float(value) for value in position)
        for axis_index, axis in enumerate(rotation_axes):
            ax, ay, az = (float(value) for value in axis)
            result[3 + axis_index] = (
                (ay * z - az * y)
                if component == 0
                else (az * x - ax * z)
                if component == 1
                else (ax * y - ay * x)
            )
        return result

    first = int(np.argmax(np.sum(centered * centered, axis=1)))
    distance_from_first = np.sum((centered - centered[first]) ** 2, axis=1)
    second = int(np.argmax(distance_from_first))
    anchors = [first, second]
    if len(rotation_axes) == 3:
        line = centered[second] - centered[first]
        line_norm = float(np.linalg.norm(line))
        if line_norm > np.finfo(float).eps:
            transverse = np.cross(centered - centered[first], line / line_norm)
            anchors.append(int(np.argmax(np.sum(transverse * transverse, axis=1))))
    anchors.append(0)

    candidates: list[tuple[int, np.ndarray]] = []
    for atom in dict.fromkeys(anchors):
        position = centered[atom]
        offset = 3 * atom
        for component in range(3):
            candidates.append((offset + component, coefficients(position, component)))

    def select(candidate_rows: list[tuple[int, np.ndarray]]) -> None:
        while len(selected) < count:
            best_column = -1
            best_residual: np.ndarray | None = None
            best_norm = -1.0
            for column, candidate in candidate_rows:
                if column in selected:
                    continue
                residual = np.asarray(candidate, dtype=float).copy()
                for vector in basis:
                    residual -= vector * float(vector @ residual)
                norm = float(np.linalg.norm(residual))
                if norm > best_norm:
                    best_column, best_residual, best_norm = column, residual, norm
            if best_residual is None or best_norm <= 1.0e-12:
                break
            selected.append(best_column)
            basis.append(best_residual / best_norm)

    select(candidates)
    if len(selected) < count:
        fallback = [
            (3 * atom + component, coefficients(position, component))
            for atom, position in enumerate(centered)
            for component in range(3)
        ]
        select(fallback)
    return tuple(selected)


def _rigid_gauge_columns_remain_independent(
    centered_coordinates: np.ndarray,
    rotation_axes: tuple[np.ndarray, ...],
    columns: tuple[int, ...],
) -> bool:
    """Certify a cached Cartesian gauge in the fixed 5/6-D rigid space."""

    count = 3 + len(rotation_axes)
    if len(columns) != count or len(set(columns)) != count:
        return False
    centered = np.asarray(centered_coordinates, dtype=float)
    if _optimizer_native is not None and hasattr(
        _optimizer_native, "rigid_gauge_columns_valid"
    ):
        axes = np.asarray(rotation_axes, dtype=float).reshape((-1, 3))
        return bool(
            _optimizer_native.rigid_gauge_columns_valid(
                centered,
                axes,
                np.asarray(columns, dtype=np.intp),
                1.0e10,
            )
        )
    coefficients = np.zeros((count, count), dtype=float)
    for local_column, column in enumerate(columns):
        atom, component = divmod(int(column), 3)
        if atom < 0 or atom >= centered.shape[0]:
            return False
        coefficients[component, local_column] = 1.0
        position = centered[atom]
        for axis_index, axis in enumerate(rotation_axes):
            coefficients[3 + axis_index, local_column] = np.cross(axis, position)[component]
    singular_values = np.linalg.svd(coefficients, compute_uv=False)
    if singular_values.size != count or singular_values[-1] <= 1.0e-12:
        return False
    return bool(singular_values[0] / singular_values[-1] <= 1.0e10)


@dataclass(frozen=True)
class CertifiedSparseInverseOperator:
    """Matrix-free action of a SMITH-certified sparse Wilson inverse.

    The operator represents ``J = pinv(W B_free) W`` without constructing
    either ``J`` or ``J.T @ J``.  ``W`` is LINK's established row
    equilibration and fixed Cartesian columns are removed from the iterative
    least-squares problem.  SMITH's direct-sum certificate is verified before
    any numerical action; the sparse Krylov solve then keeps storage
    proportional to the Wilson nonzeros.
    """

    b_matrix: SparseBMatrix
    certificate: SparseBDirectSumCertificate
    fixed_cartesian_columns: tuple[int, ...] = ()
    coordinates_angstrom: np.ndarray | None = None
    relative_tolerance: float = 1.0e-11
    maximum_iterations: int | None = None
    canonical_certificate_from_current_matrix: bool = False
    preferred_rigid_gauge_columns: tuple[int, ...] = ()
    preferred_symbolic_plan: SparseGaugeSymbolicPlan | None = None

    def __post_init__(self) -> None:
        # Public callers remain fail-closed.  The SONIC runtime may set the
        # provenance flag only when it has just obtained this certificate by
        # calling certify_sparse_b_direct_sum on this exact immutable matrix;
        # recomputing the same graph and hash in the constructor is then pure
        # duplicate O(nnz) work.
        if not self.canonical_certificate_from_current_matrix:
            verify_sparse_b_direct_sum(self.b_matrix, self.certificate)
        fixed = np.asarray(self.fixed_cartesian_columns, dtype=int).reshape(-1)
        if np.any(fixed < 0) or np.any(fixed >= self.b_matrix.column_count):
            raise ValueError("fixed Cartesian column is outside the Wilson B matrix")
        if len(set(int(value) for value in fixed)) != fixed.size:
            raise ValueError("fixed Cartesian columns must be unique")
        tolerance = float(self.relative_tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("matrix-free inverse tolerance must be finite and positive")

        fixed_set = set(int(value) for value in fixed)
        free = np.asarray(
            [column for column in range(self.b_matrix.column_count) if column not in fixed_set],
            dtype=int,
        )
        data, indices, indptr = self.b_matrix.csr_arrays()
        native_b_arrays = (
            np.asarray(data, dtype=float),
            np.asarray(indices, dtype=np.intp),
            np.asarray(indptr, dtype=np.intp),
            int(self.b_matrix.column_count),
        )
        for array in native_b_arrays[:3]:
            array.flags.writeable = False
        object.__setattr__(self, "_native_b_csr", native_b_arrays)
        raw = csr_matrix(
            (data, indices, indptr),
            shape=(self.b_matrix.row_count, self.b_matrix.column_count),
            copy=False,
        )
        row_norms = np.sqrt(np.asarray(raw.multiply(raw).sum(axis=1)).reshape(-1))
        row_scale = np.ones(self.b_matrix.row_count, dtype=float)
        nonzero = row_norms > np.finfo(float).eps
        row_scale[nonzero] = 1.0 / row_norms[nonzero]
        scaled = raw[:, free].multiply(row_scale[:, None]).tocsr()
        scaled.sort_indices()
        object.__setattr__(self, "fixed_cartesian_columns", tuple(int(v) for v in fixed))
        object.__setattr__(self, "relative_tolerance", tolerance)
        object.__setattr__(self, "_free_columns", free)
        object.__setattr__(self, "_row_scale", row_scale)
        object.__setattr__(self, "_scaled_b", scaled)
        object.__setattr__(self, "_gauge_factor", None)
        object.__setattr__(self, "_gauge_free_columns", None)
        object.__setattr__(self, "_rigid_centered_coordinates", None)
        object.__setattr__(self, "_rigid_rotation_inverse", None)
        object.__setattr__(self, "_rigid_null_residual", None)
        object.__setattr__(self, "_symbolic_plan", None)
        coordinates = self.coordinates_angstrom
        if coordinates is not None:
            xyz = np.asarray(coordinates, dtype=float)
            expected = (self.b_matrix.column_count // 3, 3)
            if xyz.shape != expected or not np.all(np.isfinite(xyz)):
                raise ValueError(
                    "matrix-free inverse coordinates do not match Wilson columns"
                )
            object.__setattr__(self, "coordinates_angstrom", xyz.copy())
            if fixed.size == 0:
                self._install_exact_rigid_gauge_solver(xyz)
        factor = self._gauge_factor
        if factor is not None:
            native_inverse_arrays = (
                factor.solve,
                self._row_scale,
                self._gauge_free_columns,
                self._rigid_centered_coordinates,
                self._rigid_rotation_inverse,
                int(self.b_matrix.column_count),
            )
            for array in native_inverse_arrays[1:5]:
                if isinstance(array, np.ndarray):
                    array.flags.writeable = False
            object.__setattr__(self, "_native_inverse_plan", native_inverse_arrays)
        else:
            object.__setattr__(self, "_native_inverse_plan", None)

    def _install_exact_rigid_gauge_solver(self, coordinates: np.ndarray) -> None:
        """Factor one square rigid-gauge section of a full-row-rank Wilson map."""

        gauge_count = self.b_matrix.column_count - self.b_matrix.row_count
        if gauge_count not in (5, 6):
            return
        centered = coordinates - np.mean(coordinates, axis=0)
        rotation_axes = _rigid_rotation_axes(centered, gauge_count)
        null_residual = max(
            (
                float(
                    np.max(
                        np.abs(
                            self._scaled_b
                            @ _rigid_mode_vector(centered, mode, rotation_axes)
                        ),
                        initial=0.0,
                    )
                )
                for mode in range(gauge_count)
            ),
            default=0.0,
        )
        if null_residual > max(1.0e-9, 100.0 * self.relative_tolerance):
            return
        preferred = tuple(int(value) for value in self.preferred_rigid_gauge_columns)
        gauge_columns = (
            preferred
            if _rigid_gauge_columns_remain_independent(
                centered, rotation_axes, preferred
            )
            else _independent_rigid_gauge_columns(centered, rotation_axes)
        )
        if len(gauge_columns) != gauge_count:
            return
        removed = set(gauge_columns)
        free = np.asarray(
            [
                column
                for column in range(self.b_matrix.column_count)
                if column not in removed
            ],
            dtype=int,
        )
        if free.size != self.b_matrix.row_count:
            return
        factor = None
        symbolic_plan = self.preferred_symbolic_plan
        raw_data, raw_indices, raw_indptr = self.b_matrix.csr_arrays()
        if (
            symbolic_plan is not None
            and symbolic_plan.gauge_columns == tuple(sorted(gauge_columns))
            and np.array_equal(raw_indices, symbolic_plan.support_indices)
            and np.array_equal(raw_indptr, symbolic_plan.support_indptr)
            and np.array_equal(free, symbolic_plan.gauge_free_columns)
        ):
            numeric = (
                raw_data[symbolic_plan.raw_positions]
                * self._row_scale[symbolic_plan.csc_indices]
            )
            matrix = csc_matrix(
                (numeric, symbolic_plan.csc_indices, symbolic_plan.csc_indptr),
                shape=(self.b_matrix.row_count, self.b_matrix.row_count),
                copy=False,
            )
            try:
                factor = _PermutedSparseLU(
                    splu(matrix, permc_spec="NATURAL"),
                    symbolic_plan.column_permutation,
                )
            except RuntimeError:
                factor = None
        if factor is None:
            gauge_matrix = self._scaled_b[:, free].tocsc()
            try:
                factor = splu(gauge_matrix, permc_spec="COLAMD")
            except RuntimeError:
                return
            permutation = np.argsort(np.asarray(factor.perm_c, dtype=int))
            markers = np.arange(raw_data.size, dtype=float) + 1.0
            marker_matrix = csr_matrix(
                (markers, raw_indices, raw_indptr),
                shape=(self.b_matrix.row_count, self.b_matrix.column_count),
            )[:, free][:, permutation].tocsc()
            raw_positions = np.asarray(marker_matrix.data, dtype=np.intp) - 1
            plan_arrays = (
                np.asarray(raw_indices, dtype=np.intp).copy(),
                np.asarray(raw_indptr, dtype=np.intp).copy(),
                np.asarray(free, dtype=int).copy(),
                np.asarray(permutation, dtype=int).copy(),
                np.asarray(marker_matrix.indices, dtype=np.intp).copy(),
                np.asarray(marker_matrix.indptr, dtype=np.intp).copy(),
                raw_positions,
            )
            for array in plan_arrays:
                array.flags.writeable = False
            symbolic_plan = SparseGaugeSymbolicPlan(
                support_indices=plan_arrays[0],
                support_indptr=plan_arrays[1],
                gauge_columns=tuple(sorted(gauge_columns)),
                gauge_free_columns=plan_arrays[2],
                column_permutation=plan_arrays[3],
                csc_indices=plan_arrays[4],
                csc_indptr=plan_arrays[5],
                raw_positions=plan_arrays[6],
            )
        inertia = (
            float(np.sum(centered * centered)) * np.eye(3, dtype=float)
            - centered.T @ centered
        )
        rotation_inverse = np.linalg.pinv(inertia, rcond=self.relative_tolerance)
        object.__setattr__(self, "_gauge_factor", factor)
        object.__setattr__(self, "_gauge_free_columns", free)
        object.__setattr__(self, "_rigid_centered_coordinates", centered.copy())
        object.__setattr__(self, "_rigid_rotation_inverse", rotation_inverse)
        object.__setattr__(self, "_rigid_null_residual", null_residual)
        object.__setattr__(self, "_symbolic_plan", symbolic_plan)

    @property
    def rigid_gauge_columns(self) -> tuple[int, ...]:
        free = self._gauge_free_columns
        if free is None:
            return ()
        retained = set(int(value) for value in free)
        return tuple(
            column
            for column in range(self.b_matrix.column_count)
            if column not in retained
        )

    @property
    def symbolic_plan(self) -> SparseGaugeSymbolicPlan | None:
        return self._symbolic_plan

    @property
    def shape(self) -> tuple[int, int]:
        return self.b_matrix.column_count, self.b_matrix.row_count

    @property
    def stored_value_count(self) -> int:
        """Number of floating-point values retained by the sparse map."""

        factor_values = 0
        if self._gauge_factor is not None:
            factor_values = int(self._gauge_factor.L.nnz + self._gauge_factor.U.nnz)
        rigid_values = (
            0
            if self._rigid_centered_coordinates is None
            else int(self._rigid_centered_coordinates.size + 9)
        )
        return int(
            self._scaled_b.nnz
            + self._row_scale.size
            + factor_values
            + rigid_values
        )

    @property
    def solver_method(self) -> str:
        return (
            "SPARSE_RIGID_GAUGE_SUPERLU_MINIMUM_NORM"
            if self._gauge_factor is not None
            else "SPARSE_LSMR_MINIMUM_NORM"
        )

    def full_row_rank_diagnostics(
        self,
        *,
        relative_tolerance: float = 1.0e-10,
        absolute_tolerance: float = 1.0e-10,
    ) -> dict[str, float | int | str] | None:
        """Return a rank audit from the already factorized sparse gauge section.

        This deliberately reuses the solve factor assembled for Cartesian
        realization.  It neither rebuilds Wilson rows nor materializes a
        dense singular-value problem merely to audit the accepted geometry.
        """

        factor = self._gauge_factor
        if factor is None:
            return None
        pivots = np.abs(np.asarray(factor.U.diagonal(), dtype=float))
        maximum = float(np.max(pivots, initial=0.0))
        minimum = float(np.min(pivots, initial=np.inf))
        cutoff = max(float(absolute_tolerance), float(relative_tolerance) * maximum)
        rank = int(np.count_nonzero(pivots > cutoff))
        condition = float(maximum / minimum) if minimum > 0.0 else float("inf")
        return {
            "rank": rank,
            "maximum_rank": self.b_matrix.row_count,
            "condition_number": condition,
            "minimum_pivot": minimum,
            "maximum_pivot": maximum,
            "pivot_cutoff": cutoff,
            "rank_method": "REUSED_SPARSE_RIGID_GAUGE_SUPERLU",
            "rigid_mode_residual": float(self._rigid_null_residual or 0.0),
        }

    def matvec(self, internal_displacement: np.ndarray, *, damping: float = 0.0) -> np.ndarray:
        """Apply ``dx = B+ dq`` with a sparse minimum-norm solve."""

        rhs = np.asarray(internal_displacement, dtype=float).reshape(-1)
        if rhs.shape != (self.b_matrix.row_count,) or not np.all(np.isfinite(rhs)):
            raise ValueError("internal displacement does not match sparse inverse rows")
        scaled_rhs = self._row_scale * rhs
        result = np.zeros(self.b_matrix.column_count, dtype=float)
        if self._gauge_factor is not None and float(damping) == 0.0:
            result[self._gauge_free_columns] = self._gauge_factor.solve(scaled_rhs)
            result = self._remove_rigid_motion(result)
        else:
            from scipy.sparse.linalg import lsmr

            free_solution = lsmr(
                self._scaled_b,
                scaled_rhs,
                damp=float(damping),
                atol=self.relative_tolerance,
                btol=self.relative_tolerance,
                maxiter=self.maximum_iterations,
            )[0]
            result[self._free_columns] = free_solution
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("matrix-free sparse inverse returned non-finite values")
        return result

    def rmatvec(self, cartesian_covector: np.ndarray) -> np.ndarray:
        """Apply ``dq_covector = B+T dx_covector`` without storing ``B+``."""

        vector = np.asarray(cartesian_covector, dtype=float).reshape(-1)
        if vector.shape != (self.b_matrix.column_count,) or not np.all(np.isfinite(vector)):
            raise ValueError("Cartesian covector does not match sparse inverse columns")
        if self._gauge_factor is not None:
            projected = self._remove_rigid_motion(vector)
            solution = self._gauge_factor.solve(
                projected[self._gauge_free_columns],
                trans="T",
            )
        else:
            from scipy.sparse.linalg import lsmr

            solution = lsmr(
                self._scaled_b.T,
                vector[self._free_columns],
                atol=self.relative_tolerance,
                btol=self.relative_tolerance,
                maxiter=self.maximum_iterations,
            )[0]
        result = self._row_scale * solution
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("matrix-free transpose inverse returned non-finite values")
        return result

    def _remove_rigid_motion(self, vector: np.ndarray) -> np.ndarray:
        """Remove translations and rotations without storing a ``6 x 3N`` basis."""

        centered = self._rigid_centered_coordinates
        rotation_inverse = self._rigid_rotation_inverse
        if centered is None or rotation_inverse is None:
            return np.asarray(vector, dtype=float).copy()
        if _optimizer_native is not None and hasattr(
            _optimizer_native, "remove_rigid_motion"
        ):
            return np.asarray(
                _optimizer_native.remove_rigid_motion(
                    np.asarray(vector, dtype=float).reshape(-1),
                    centered,
                    rotation_inverse,
                ),
                dtype=float,
            )
        velocities = np.asarray(vector, dtype=float).reshape(centered.shape)
        translation = np.mean(velocities, axis=0)
        relative = velocities - translation
        angular_rhs = np.sum(np.cross(centered, relative), axis=0)
        angular_velocity = rotation_inverse @ angular_rhs
        rigid = translation + np.cross(angular_velocity, centered)
        return (velocities - rigid).reshape(-1)

    def b_matvec(self, cartesian_vector: np.ndarray) -> np.ndarray:
        """Apply the original sparse Wilson matrix to a Cartesian vector."""

        vector = np.asarray(cartesian_vector, dtype=float).reshape(-1)
        if vector.shape != (self.b_matrix.column_count,) or not np.all(np.isfinite(vector)):
            raise ValueError("Cartesian vector does not match Wilson columns")
        free_values = self._scaled_b @ vector[self._free_columns]
        return np.asarray(free_values, dtype=float) / self._row_scale

    def b_rmatvec(self, internal_covector: np.ndarray) -> np.ndarray:
        """Apply the original sparse Wilson transpose."""

        vector = np.asarray(internal_covector, dtype=float).reshape(-1)
        if vector.shape != (self.b_matrix.row_count,) or not np.all(np.isfinite(vector)):
            raise ValueError("internal covector does not match Wilson rows")
        free_values = self._scaled_b.T @ (vector / self._row_scale)
        result = np.zeros(self.b_matrix.column_count, dtype=float)
        result[self._free_columns] = np.asarray(free_values, dtype=float)
        return result

    def metric_matvec(self, internal_displacement: np.ndarray) -> np.ndarray:
        """Apply the Wilson displacement metric ``J.T J`` matrix-free."""

        return self.rmatvec(self.matvec(internal_displacement))

    def project_cartesian(self, cartesian_vector: np.ndarray) -> np.ndarray:
        """Project a Cartesian vector onto the certified active tangent."""

        return self.matvec(self.b_matvec(cartesian_vector))


def cartesian_from_internal_jacobian(
    b_matrix: np.ndarray,
    *,
    rcond: float = 1.0e-8,
    fixed_cartesian_columns: np.ndarray | None = None,
) -> np.ndarray:
    """Return LINK's local internal-to-Cartesian Jacobian from a Wilson B matrix.

    Coordinate providers supply ``B``; LINK owns its generalized inverse and
    the policies that realize finite internal-coordinate changes.
    """

    b = np.asarray(b_matrix, dtype=float)
    if b.ndim != 2 or not np.all(np.isfinite(b)):
        raise ValueError("Wilson B matrix must be a finite two-dimensional array")
    # Wilson rows can have very different numerical scales.  This is common
    # for nearly linear bends, whose analytic derivatives can be orders of
    # magnitude larger than stretches or fragment coordinates.  Applying the
    # cutoff directly to the raw SVD then discards valid coordinate directions
    # in mixed-scale internal-coordinate models.
    # Equilibrate rows before the SVD and undo that scaling on the right:
    # pinv(W B) W is the minimum-norm inverse of B in the original internal
    # displacement units.
    row_norms = np.linalg.norm(b, axis=1)
    row_scale = np.ones(b.shape[0], dtype=float)
    nonzero = row_norms > np.finfo(float).eps
    row_scale[nonzero] = 1.0 / row_norms[nonzero]
    scaled = row_scale[:, None] * b
    fixed = (
        np.asarray((), dtype=int)
        if fixed_cartesian_columns is None
        else np.asarray(fixed_cartesian_columns, dtype=int).reshape(-1)
    )
    row_unscale = np.diag(row_scale)
    if fixed.size == 0:
        return np.linalg.pinv(scaled, rcond=float(rcond)) @ row_unscale
    if np.any(fixed < 0) or np.any(fixed >= b.shape[1]):
        raise ValueError("fixed Cartesian column is outside the Wilson B matrix")
    free = np.ones(b.shape[1], dtype=bool)
    free[fixed] = False
    projector = np.zeros((b.shape[1], b.shape[0]), dtype=float)
    projector[free, :] = np.linalg.pinv(scaled[:, free], rcond=float(rcond)) @ row_unscale
    return projector


def cartesian_from_internal_jacobian_direct_sum(
    b_matrix: SparseBMatrix,
    certificate: SparseBDirectSumCertificate,
    *,
    rcond: float = 1.0e-8,
    fixed_cartesian_columns: np.ndarray | None = None,
) -> np.ndarray:
    """Return the exact blockwise realization certified by SMITH."""

    verify_sparse_b_direct_sum(b_matrix, certificate)
    fixed = (
        np.asarray((), dtype=int)
        if fixed_cartesian_columns is None
        else np.asarray(fixed_cartesian_columns, dtype=int).reshape(-1)
    )
    if np.any(fixed < 0) or np.any(fixed >= b_matrix.column_count):
        raise ValueError("fixed Cartesian column is outside the Wilson B matrix")
    fixed_set = set(int(index) for index in fixed)
    free_columns = tuple(
        index for index in range(b_matrix.column_count) if index not in fixed_set
    )
    free_position = {column: index for index, column in enumerate(free_columns)}
    result_free = np.zeros((len(free_columns), b_matrix.row_count), dtype=float)
    row_norms = np.asarray([row.norm() for row in b_matrix.rows], dtype=float)
    row_scale = np.ones(b_matrix.row_count, dtype=float)
    nonzero = row_norms > np.finfo(float).eps
    row_scale[nonzero] = 1.0 / row_norms[nonzero]
    separator_rows = certificate.separator_row_indices
    separator_b = np.zeros((len(separator_rows), len(free_columns)), dtype=float)
    for local_row, global_row in enumerate(separator_rows):
        for global_column, value in b_matrix.rows[global_row].entries:
            if global_column in free_position:
                separator_b[local_row, free_position[global_column]] = (
                    row_scale[global_row] * value
                )
    separator_gram = separator_b @ separator_b.T
    separator_cross_sum = np.zeros_like(separator_gram)
    leaf_cartesian_times_u = np.zeros(
        (len(free_columns), len(separator_rows)), dtype=float
    )
    leaf_records: list[
        tuple[tuple[int, ...], np.ndarray, np.ndarray, np.ndarray]
    ] = []
    for component in certificate.components:
        rows = component.row_indices
        columns = tuple(
            column for column in component.column_indices if column in free_position
        )
        local_column_by_global = {column: index for index, column in enumerate(columns)}
        scaled_b = np.zeros((len(rows), len(columns)), dtype=float)
        for local_row, global_row in enumerate(rows):
            for global_column, value in b_matrix.rows[global_row].entries:
                if global_column in local_column_by_global:
                    scaled_b[local_row, local_column_by_global[global_column]] = (
                        row_scale[global_row] * value
                    )
        gram_inverse = _full_rank_symmetric_inverse(
            scaled_b @ scaled_b.T,
            rcond=rcond,
            block_name=f"leaf rows {rows}",
        )
        separator_on_columns = separator_b[
            :, [free_position[column] for column in columns]
        ]
        cross = scaled_b @ separator_on_columns.T
        u_matrix = gram_inverse @ cross
        base = scaled_b.T @ gram_inverse
        separator_cross_sum += cross.T @ u_matrix
        mapped = np.asarray([free_position[column] for column in columns], dtype=int)
        leaf_cartesian_times_u[mapped, :] += scaled_b.T @ u_matrix
        leaf_records.append((rows, mapped, base, u_matrix))

    if separator_rows:
        schur_inverse = _full_rank_symmetric_inverse(
            separator_gram - separator_cross_sum,
            rcond=rcond,
            block_name="separator Schur complement",
        )
        q_matrix = separator_b.T - leaf_cartesian_times_u
        q_times_schur = q_matrix @ schur_inverse
        result_free[:, separator_rows] = q_times_schur
        for rows, mapped, base, u_matrix in leaf_records:
            result_free[np.ix_(mapped, rows)] += base
            result_free[:, rows] -= q_times_schur @ u_matrix.T
    else:
        for rows, mapped, base, _u_matrix in leaf_records:
            result_free[np.ix_(mapped, rows)] = base

    result_free *= row_scale[None, :]
    result = np.zeros((b_matrix.column_count, b_matrix.row_count), dtype=float)
    result[np.asarray(free_columns, dtype=int), :] = result_free
    return result


def _full_rank_symmetric_inverse(
    matrix: np.ndarray,
    *,
    rcond: float,
    block_name: str,
) -> np.ndarray:
    symmetric = np.asarray(matrix, dtype=float)
    symmetric = 0.5 * (symmetric + symmetric.T)
    values = np.linalg.eigvalsh(symmetric)
    if values.size == 0:
        return np.zeros_like(symmetric, dtype=float)
    scale = float(np.max(np.abs(values)))
    cutoff = (float(rcond) ** 2) * max(scale, np.finfo(float).eps)
    if float(np.min(np.abs(values))) <= cutoff:
        raise CertifiedDirectSumInverseError(
            f"{block_name} is rank deficient after applying fixed Cartesian columns"
        )
    return np.linalg.inv(symmetric)


def internal_from_cartesian_jacobian(
    cartesian_from_internal: np.ndarray,
    *,
    rcond: float = 1.0e-8,
) -> np.ndarray:
    """Return the minimum-norm Cartesian-to-internal linear map.

    This is LINK's projection service for user or SENTINEL variables supplied
    as Cartesian directions. It does not define new coordinates; it projects
    them into the frozen SONIC tangent space supplied by SMITH.
    """

    matrix = np.asarray(cartesian_from_internal, dtype=float)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("Cartesian-from-internal Jacobian must be a finite 2-D array")
    return np.linalg.pinv(matrix, rcond=float(rcond))


def nonlinear_internal_coordinate_step(
    coordinates_angstrom: np.ndarray,
    target_values: np.ndarray,
    evaluate: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    *,
    evaluate_values: Callable[[np.ndarray], np.ndarray] | None = None,
    cartesian_from_q: np.ndarray | None = None,
    max_iterations: int = 60,
    tolerance: float = 1.0e-9,
    max_cartesian_step_angstrom: float = 0.25,
    fixed_atom_indices: tuple[int, ...] = (),
    project_coordinates: Callable[[np.ndarray], np.ndarray] | None = None,
) -> InternalCoordinateBackTransform:
    """Solve ``q(x) = target_values`` by damped least squares.

    ``evaluate`` returns the active internal-coordinate values and their Wilson
    matrix at a Cartesian geometry.  The accepted geometry is always evaluated
    explicitly, so callers never need to identify a requested internal step
    with the step actually realized by a nonlinear back-transformation.
    """

    coords = np.asarray(coordinates_angstrom, dtype=float).copy()
    fixed_columns = _fixed_cartesian_columns(fixed_atom_indices, coords.shape[0])
    target = np.asarray(target_values, dtype=float).reshape(-1)
    projector = None if cartesian_from_q is None else np.asarray(cartesian_from_q, dtype=float)
    values, b_matrix = _validated_evaluation(evaluate, coords, target.size)
    residual = target - values
    initial_norm = float(np.linalg.norm(residual))
    if initial_norm <= tolerance:
        projector = _cartesian_projector(b_matrix, fixed_columns)
        return InternalCoordinateBackTransform(coords, values, residual, 0, True, projector)

    for iteration in range(1, max_iterations + 1):
        if projector is None or projector.shape != (coords.size, target.size):
            projector = _cartesian_projector(b_matrix, fixed_columns)
        dx = projector @ residual
        dx_norm = float(np.linalg.norm(dx))
        if max_cartesian_step_angstrom > 0.0 and dx_norm > max_cartesian_step_angstrom:
            dx *= max_cartesian_step_angstrom / dx_norm

        current_norm = float(np.linalg.norm(residual))
        accepted = False
        scale = 1.0
        for _ in range(10):
            trial_coords = coords + (scale * dx).reshape(coords.shape)
            if project_coordinates is not None:
                trial_coords = np.asarray(project_coordinates(trial_coords), dtype=float)
            if evaluate_values is None:
                trial_values, trial_b = _validated_evaluation(evaluate, trial_coords, target.size)
            else:
                trial_values = _validated_values(evaluate_values, trial_coords, target.size)
                trial_b = None
            trial_residual = target - trial_values
            if float(np.linalg.norm(trial_residual)) < current_norm:
                if trial_b is None:
                    trial_values, trial_b = _validated_evaluation(
                        evaluate, trial_coords, target.size
                    )
                coords = trial_coords
                values = trial_values
                b_matrix = trial_b
                residual = trial_residual
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            projector = _cartesian_projector(b_matrix, fixed_columns)
            fallback = _damped_internal_step(
                b_matrix, residual, coords.shape, fixed_columns=fixed_columns
            )
            if fallback is None:
                return InternalCoordinateBackTransform(
                    coords, values, residual, iteration, False, projector
                )
            for fallback_dx in fallback:
                fallback_norm = float(np.linalg.norm(fallback_dx))
                if (
                    max_cartesian_step_angstrom > 0.0
                    and fallback_norm > max_cartesian_step_angstrom
                ):
                    fallback_dx *= max_cartesian_step_angstrom / fallback_norm
                trial_coords = coords + fallback_dx.reshape(coords.shape)
                if project_coordinates is not None:
                    trial_coords = np.asarray(project_coordinates(trial_coords), dtype=float)
                if evaluate_values is None:
                    trial_values, trial_b = _validated_evaluation(
                        evaluate, trial_coords, target.size
                    )
                else:
                    trial_values = _validated_values(evaluate_values, trial_coords, target.size)
                    trial_b = None
                trial_residual = target - trial_values
                if float(np.linalg.norm(trial_residual)) < current_norm:
                    if trial_b is None:
                        trial_values, trial_b = _validated_evaluation(
                            evaluate, trial_coords, target.size
                        )
                        trial_residual = target - trial_values
                    coords, values, b_matrix, residual = (
                        trial_coords,
                        trial_values,
                        trial_b,
                        trial_residual,
                    )
                    accepted = True
                    break
            if not accepted:
                return InternalCoordinateBackTransform(
                    coords, values, residual, iteration, False, projector
                )
        projector = _cartesian_projector(b_matrix, fixed_columns)
        if float(np.linalg.norm(residual)) <= tolerance * max(1.0, initial_norm):
            return InternalCoordinateBackTransform(
                coords, values, residual, iteration, True, projector
            )
    return InternalCoordinateBackTransform(
        coords, values, residual, max_iterations, False, projector
    )


def nonlinear_sparse_internal_coordinate_step(
    coordinates_angstrom: np.ndarray,
    target_values: np.ndarray,
    evaluate: Callable[
        [np.ndarray],
        tuple[np.ndarray, SparseBMatrix, SparseBDirectSumCertificate],
    ],
    *,
    evaluate_values: Callable[[np.ndarray], np.ndarray] | None = None,
    fixed_atom_indices: tuple[int, ...] = (),
    project_coordinates: Callable[[np.ndarray], np.ndarray] | None = None,
    tolerance: float = 1.0e-9,
    max_iterations: int = 60,
    max_cartesian_step_angstrom: float = 0.25,
    maximum_operator_age: int = 4,
    maximum_linearization_error: float = 0.15,
    initial_operator: CertifiedSparseInverseOperator | None = None,
    initial_values: np.ndarray | None = None,
    evaluate_operator: Callable[
        [np.ndarray], tuple[np.ndarray, CertifiedSparseInverseOperator]
    ] | None = None,
) -> InternalCoordinateBackTransform:
    """Realize an internal target without dense ``B``, ``B+`` or metric arrays.

    The sparse Wilson operator is reused across accepted micro-iterations.  A
    refresh is triggered by either a bounded age or the observed secant defect
    ``||dq - B dx|| / ||dq||``.  Coordinate values are still evaluated at
    every trial and provide the authoritative final residual.
    """

    coords = np.asarray(coordinates_angstrom, dtype=float).copy()
    target = np.asarray(target_values, dtype=float).reshape(-1)
    fixed_columns = tuple(
        int(value) for value in _fixed_cartesian_columns(fixed_atom_indices, coords.shape[0])
    )
    operator_age_limit = int(maximum_operator_age)
    linearization_limit = float(maximum_linearization_error)
    if operator_age_limit < 1:
        raise ValueError("maximum sparse-Wilson operator age must be positive")
    if not np.isfinite(linearization_limit) or linearization_limit <= 0.0:
        raise ValueError("maximum sparse-Wilson linearization error must be positive")

    if (
        native_sonic_backtransform_available()
        and initial_operator is not None
        and evaluate_operator is not None
    ):
        native_coordinates, native_values, native_residual, iterations, converged = (
            _optimizer_native.sonic_sparse_backtransform(
                coords,
                target,
                evaluate_values if evaluate_values is not None else lambda point: evaluate(point)[0],
                evaluate_operator,
                project_coordinates,
                initial_operator,
                initial_values,
                tolerance=float(tolerance),
                max_iterations=int(max_iterations),
                max_cartesian_step_angstrom=float(max_cartesian_step_angstrom),
                maximum_operator_age=operator_age_limit,
                maximum_linearization_error=linearization_limit,
            )
        )
        return InternalCoordinateBackTransform(
            np.asarray(native_coordinates, dtype=float),
            np.asarray(native_values, dtype=float),
            np.asarray(native_residual, dtype=float),
            int(iterations),
            bool(converged),
            None,
        )

    def evaluated(point: np.ndarray):
        if evaluate_operator is not None:
            array, operator = evaluate_operator(point)
            array = np.asarray(array, dtype=float).reshape(-1)
            if array.shape != target.shape or not np.all(np.isfinite(array)):
                raise ValueError("sparse internal evaluator returned invalid values")
            if operator.shape != (coords.size, target.size):
                raise ValueError("sparse internal evaluator returned an invalid operator")
            return array, operator
        values, sparse_b, certificate = evaluate(point)
        array = np.asarray(values, dtype=float).reshape(-1)
        if array.shape != target.shape or not np.all(np.isfinite(array)):
            raise ValueError("sparse internal evaluator returned invalid values")
        if sparse_b.row_count != target.size or sparse_b.column_count != coords.size:
            raise ValueError("sparse internal evaluator returned an invalid Wilson shape")
        operator = CertifiedSparseInverseOperator(
            sparse_b,
            certificate,
            fixed_cartesian_columns=fixed_columns,
            coordinates_angstrom=point,
        )
        return array, operator

    if initial_operator is None:
        values, operator = evaluated(coords)
    else:
        if initial_operator.shape != (coords.size, target.size):
            raise ValueError("initial sparse-Wilson operator has the wrong shape")
        operator = initial_operator
        if initial_values is not None:
            values = np.asarray(initial_values, dtype=float).reshape(-1)
            if values.shape != target.shape or not np.all(np.isfinite(values)):
                raise ValueError("initial internal-coordinate values are invalid")
        else:
            values = (
                _validated_values(evaluate_values, coords, target.size)
                if evaluate_values is not None
                else evaluated(coords)[0]
            )
    operator_age = 0
    residual = target - values
    initial_norm = float(np.linalg.norm(residual))
    if initial_norm <= tolerance:
        return InternalCoordinateBackTransform(coords, values, residual, 0, True, None)

    for iteration in range(1, int(max_iterations) + 1):
        dx = operator.matvec(residual)
        dx_norm = float(np.linalg.norm(dx))
        if max_cartesian_step_angstrom > 0.0 and dx_norm > max_cartesian_step_angstrom:
            dx *= max_cartesian_step_angstrom / dx_norm
        current_norm = float(np.linalg.norm(residual))
        accepted = False
        for scale in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625):
            trial_coords = coords + (scale * dx).reshape(coords.shape)
            if project_coordinates is not None:
                trial_coords = np.asarray(project_coordinates(trial_coords), dtype=float)
            trial_values = (
                np.asarray(evaluate_values(trial_coords), dtype=float).reshape(-1)
                if evaluate_values is not None
                else evaluated(trial_coords)[0]
            )
            trial_residual = target - trial_values
            if float(np.linalg.norm(trial_residual)) < current_norm:
                cartesian_change = (trial_coords - coords).reshape(-1)
                internal_change = trial_values - values
                predicted_change = operator.b_matvec(cartesian_change)
                secant_scale = max(
                    float(np.linalg.norm(internal_change)),
                    float(tolerance),
                    np.finfo(float).eps,
                )
                linearization_error = float(
                    np.linalg.norm(internal_change - predicted_change) / secant_scale
                )
                coords = trial_coords
                values = trial_values
                residual = trial_residual
                operator_age += 1
                if (
                    operator_age >= operator_age_limit
                    or linearization_error > linearization_limit
                ) and float(np.linalg.norm(residual)) > tolerance * max(1.0, initial_norm):
                    values, operator = evaluated(coords)
                    residual = target - values
                    operator_age = 0
                accepted = True
                break
        if not accepted:
            if operator_age:
                values, operator = evaluated(coords)
                residual = target - values
                operator_age = 0
                continue
            for damping in (1.0e-6, 1.0e-4, 1.0e-2, 1.0):
                fallback = operator.matvec(residual, damping=damping)
                fallback_norm = float(np.linalg.norm(fallback))
                if (
                    max_cartesian_step_angstrom > 0.0
                    and fallback_norm > max_cartesian_step_angstrom
                ):
                    fallback *= max_cartesian_step_angstrom / fallback_norm
                trial_coords = coords + fallback.reshape(coords.shape)
                if project_coordinates is not None:
                    trial_coords = np.asarray(project_coordinates(trial_coords), dtype=float)
                trial_values = (
                    np.asarray(evaluate_values(trial_coords), dtype=float).reshape(-1)
                    if evaluate_values is not None
                    else evaluated(trial_coords)[0]
                )
                if float(np.linalg.norm(target - trial_values)) < current_norm:
                    cartesian_change = (trial_coords - coords).reshape(-1)
                    internal_change = trial_values - values
                    predicted_change = operator.b_matvec(cartesian_change)
                    secant_scale = max(
                        float(np.linalg.norm(internal_change)),
                        float(tolerance),
                        np.finfo(float).eps,
                    )
                    linearization_error = float(
                        np.linalg.norm(internal_change - predicted_change) / secant_scale
                    )
                    coords = trial_coords
                    values = trial_values
                    residual = target - values
                    operator_age += 1
                    if (
                        operator_age >= operator_age_limit
                        or linearization_error > linearization_limit
                    ) and float(np.linalg.norm(residual)) > tolerance * max(1.0, initial_norm):
                        values, operator = evaluated(coords)
                        residual = target - values
                        operator_age = 0
                    accepted = True
                    break
        if not accepted:
            return InternalCoordinateBackTransform(
                coords, values, residual, iteration, False, None
            )
        if float(np.linalg.norm(residual)) <= tolerance * max(1.0, initial_norm):
            return InternalCoordinateBackTransform(
                coords, values, residual, iteration, True, None
            )
    return InternalCoordinateBackTransform(
        coords, values, residual, int(max_iterations), False, None
    )


def gdv_redq2x_internal_coordinate_step(
    coordinates_angstrom: np.ndarray,
    target_values: np.ndarray,
    evaluate: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    *,
    coordinate_unit_scales: np.ndarray,
    max_iterations: int = 200,
    fixed_atom_indices: tuple[int, ...] = (),
) -> InternalCoordinateBackTransform:
    """Replicate GDV ``RedQ2X`` for generic redundant coordinates.

    The implementation follows ``gdv.j32+/utilnz.F:RedQ2X`` in GDV's
    bohr/radian units: each residual component is initially limited to 0.2,
    the Wilson least-squares displacement is rescaled by the agreement
    between its linear and curvilinear internal changes, and convergence is
    based on the Cartesian correction becoming smaller than 1e-6 bohr RMS.
    As in GDV, a small residual in a redundant coordinate is not by itself a
    back-transformation failure; the caller must retain the internal values
    actually realized by the returned Cartesian geometry.
    """

    coords_angstrom = np.asarray(coordinates_angstrom, dtype=float)
    target_native = np.asarray(target_values, dtype=float).reshape(-1)
    scales = np.asarray(coordinate_unit_scales, dtype=float).reshape(-1)
    if scales.shape != target_native.shape:
        raise ValueError("GDV coordinate-unit scales do not match back-transform target")
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("GDV coordinate-unit scales must be positive and finite")
    if max_iterations < 1:
        raise ValueError("GDV RedQ2X max_iterations must be positive")

    # Keep the linear algebra in the same units as RedQ2X.  The evaluator's
    # native convention is q(angstrom/radian) and dq/dx(angstrom).
    from .scan import ANGSTROM_TO_BOHR

    coords_bohr = coords_angstrom.copy() * ANGSTROM_TO_BOHR
    fixed_columns = _fixed_cartesian_columns(
        fixed_atom_indices, coords_angstrom.shape[0]
    )
    free_columns = np.setdiff1d(
        np.arange(coords_bohr.size, dtype=int), fixed_columns, assume_unique=True
    )

    def evaluate_gdv(cartesian_bohr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        native_values, native_b = _validated_evaluation(
            evaluate,
            np.asarray(cartesian_bohr, dtype=float) / ANGSTROM_TO_BOHR,
            target_native.size,
        )
        values = scales * native_values
        b_matrix = (
            scales[:, None]
            * np.asarray(native_b, dtype=float)
            / ANGSTROM_TO_BOHR
        )
        return values, b_matrix

    values0, b_matrix = evaluate_gdv(coords_bohr)
    target = scales * target_native
    requested = target - values0
    values = values0.copy()
    residual = requested.copy()
    dq_limit = 0.2
    cartesian_rms = np.inf
    projector = np.zeros((coords_bohr.size, target.size), dtype=float)
    active_b = b_matrix[:, free_columns]
    inverse_btb = _gdv_frmbtb_diagonal_inverse(active_b)

    for iteration in range(1, int(max_iterations) + 1):
        limited = np.clip(residual, -dq_limit, dq_limit)
        dx = np.zeros(coords_bohr.size, dtype=float)
        if free_columns.size:
            active_b = b_matrix[:, free_columns]
            btb_operator = _gdv_mlpxx1_operator(
                active_b,
                coords_bohr,
                free_columns,
            )
            # SLEqS3 solves (B^T B) dx = B^T dq by iterative refinement of
            # the diagonal inverse guess built once by FrmBTB.  MlPXX1 adds
            # GDV's translation and FormRB rotation terms to B^T B; the
            # refined inverse persists between RedQ2X iterations.
            free_dx, inverse_btb, solve_status = _gdv_sleqs3(
                btb_operator,
                active_b.T @ limited,
                inverse_btb,
            )
            if solve_status > 0:
                inverse_btb = _gdv_frmbtb_diagonal_inverse(active_b)
                free_dx, inverse_btb, solve_status = _gdv_sleqs3(
                    btb_operator,
                    active_b.T @ limited,
                    inverse_btb,
                )
            dx[free_columns] = free_dx
            dx = _gdv_remove_translation_rotation(coords_bohr, dx)
            dx[fixed_columns] = 0.0
            projector[free_columns, :] = np.linalg.pinv(active_b, rcond=1.0e-10)

        # CrdBMl(IOp=3) overwrites DQTmp0 with B*dx before RedQ2X compares
        # the linear and curvilinear changes.  In redundant coordinates this
        # is the representable projection of the requested residual, not the
        # residual itself.
        linear = b_matrix @ dx
        trial_values, _trial_b = evaluate_gdv(coords_bohr + dx.reshape(coords_bohr.shape))
        curvilinear = trial_values - values
        linear_norm = float(np.linalg.norm(linear))
        curvilinear_norm = float(np.linalg.norm(curvilinear))
        scalar_product = float(curvilinear @ linear)
        small = np.finfo(float).tiny
        if (
            curvilinear_norm >= small
            and linear_norm > small
            and abs(scalar_product) > small
        ):
            cosine = scalar_product / (curvilinear_norm * linear_norm)
            if abs(cosine) <= small:
                return InternalCoordinateBackTransform(
                    coords_bohr / ANGSTROM_TO_BOHR,
                    values / scales,
                    residual / scales,
                    iteration,
                    False,
                    projector / ANGSTROM_TO_BOHR * scales[None, :],
                )
            scale = abs(scalar_product) / (curvilinear_norm * curvilinear_norm)
            minimum_scale = 0.1 if iteration == 1 else 0.0
            dx *= max(min(scale, 1.0), minimum_scale)

        cartesian_rms = float(np.sqrt(np.mean(dx * dx)))
        coords_bohr += dx.reshape(coords_bohr.shape)
        values, b_matrix = evaluate_gdv(coords_bohr)
        realized = values - values0
        residual = requested - realized

        if cartesian_rms < 1.0e-6:
            native_values = values / scales
            native_residual = residual / scales
            native_projector = projector / ANGSTROM_TO_BOHR * scales[None, :]
            return InternalCoordinateBackTransform(
                coords_bohr / ANGSTROM_TO_BOHR,
                native_values,
                native_residual,
                iteration,
                True,
                native_projector,
            )
        if iteration > 100 and cartesian_rms > 1.0e-4:
            break

    return InternalCoordinateBackTransform(
        coords_bohr / ANGSTROM_TO_BOHR,
        values / scales,
        residual / scales,
        min(int(max_iterations), iteration),
        False,
        projector / ANGSTROM_TO_BOHR * scales[None, :],
    )


def _gdv_frmbtb_diagonal_inverse(b_matrix: np.ndarray) -> np.ndarray:
    """Return the literal diagonal ``FrmBTB`` inverse guess."""

    b = np.asarray(b_matrix, dtype=float)
    diagonal = np.sum(b * b, axis=0)
    result = np.zeros((b.shape[1], b.shape[1]), dtype=float)
    usable = diagonal > 1.0e-20
    indices = np.flatnonzero(usable)
    result[indices, indices] = 1.0 / diagonal[usable]
    return result


def _gdv_mlpxx1_operator(
    b_matrix: np.ndarray,
    coordinates_bohr: np.ndarray,
    cartesian_columns: np.ndarray,
) -> np.ndarray:
    """Return the literal translation/rotation-augmented ``MlPXX1`` matrix."""

    b = np.asarray(b_matrix, dtype=float)
    coordinates = np.asarray(coordinates_bohr, dtype=float)
    columns = np.asarray(cartesian_columns, dtype=int).reshape(-1)
    atom_count = coordinates.shape[0]
    operator = b.T @ b
    if atom_count == 0:
        return operator

    translation_scale = 1.0 / float(atom_count * atom_count)
    for axis in range(3):
        basis = np.zeros_like(coordinates)
        basis[:, axis] = 1.0
        vector = basis.reshape(-1)[columns]
        operator += translation_scale * np.outer(vector, vector)

    centered = coordinates - np.mean(coordinates, axis=0)
    values, axes = np.linalg.eigh(centered.T @ centered)
    value_sum = float(np.sum(values))
    for index in range(3):
        basis = np.cross(centered, axes[:, index])
        scale = value_sum - float(values[index])
        if scale > 1.0e-15:
            basis /= scale
        else:
            basis *= scale
        vector = basis.reshape(-1)[columns]
        operator += np.outer(vector, vector)
    return operator


def _gdv_sleqs3(
    matrix: np.ndarray,
    right_hand_side: np.ndarray,
    inverse_guess: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Translate GDV ``SLEqS3`` for the non-projected, zero-guess path."""

    operator = np.asarray(matrix, dtype=float)
    target = np.asarray(right_hand_side, dtype=float).reshape(-1)
    inverse = np.asarray(inverse_guess, dtype=float).copy()
    solution = np.zeros_like(target)
    residual = target.copy()
    error = float(np.max(np.abs(target), initial=0.0))
    if error == 0.0:
        return solution, inverse, 0

    convergence = error * 1.0e-10
    convergence_squared = convergence * convergence
    best_solution = solution.copy()
    best_error = error
    previous_error = error
    error_change = error
    update_count = 0
    small_count = 0
    small_cosine = 0.2
    maximum_small_count = 13
    iteration = 0
    max_iterations = 10 * target.size

    def direction_scale(
        mapped_direction: np.ndarray,
        current_residual: np.ndarray,
    ) -> tuple[float, float] | None:
        mapped_norm2 = float(mapped_direction @ mapped_direction)
        residual_norm2 = float(current_residual @ current_residual)
        if mapped_norm2 == 0.0 or residual_norm2 == 0.0:
            return None
        scalar_product = float(mapped_direction @ current_residual)
        cosine = scalar_product / np.sqrt(mapped_norm2 * residual_norm2)
        return scalar_product / mapped_norm2, cosine

    while error > convergence and iteration <= max_iterations:
        direction = inverse @ residual
        mapped = operator @ direction
        update_status = _gdv_sr1_inverse_update(inverse, mapped, direction)
        if update_status == 0:
            update_count += 1

        scaling = direction_scale(mapped, residual)
        if scaling is None:
            iteration = max_iterations + 1
            break
        step_scale, cosine = scaling

        # SLEqS3 first accepts the current step and then tries the *raw*
        # residual when an unsuccessful SR1 update leaves a nearly orthogonal
        # search direction.  In particular, this is not MInv * residual.
        if abs(cosine) <= small_cosine and update_status != 0:
            solution += step_scale * direction
            mapped_solution = operator @ solution
            if update_status * update_count != 0:
                update_status = _gdv_sr1_inverse_update(
                    inverse, mapped_solution, solution
                )
                if update_status == 0:
                    update_count += 1
            residual = target - mapped_solution
            error = float(np.max(np.abs(residual), initial=0.0))
            direction = residual.copy()
            mapped = operator @ direction
            scaling = direction_scale(mapped, residual)
            if scaling is None:
                iteration = max_iterations + 1
                break
            step_scale, cosine = scaling

        if abs(cosine) <= small_cosine and error > convergence:
            small_count += 1
            if small_count > maximum_small_count and update_count != 0:
                # Literal MSmall recovery block: finish the current direction,
                # retain the best solution, reset X, and rebuild a direction
                # from MInv * Y before returning to the main cycle.
                update_count = 0
                small_cosine = 0.2
                small_count = 0
                solution += step_scale * direction
                mapped_solution = operator @ solution
                residual = target - mapped_solution
                error = float(np.max(np.abs(residual), initial=0.0))
                if error > best_error:
                    solution = best_solution.copy()
                    mapped_solution = operator @ solution
                    _gdv_sr1_inverse_update(inverse, mapped_solution, solution)
                    residual = target - mapped_solution
                    error = float(np.max(np.abs(residual), initial=0.0))
                elif error < best_error:
                    best_solution = solution.copy()
                    best_error = error

                solution.fill(0.0)
                direction = inverse @ target
                mapped = operator @ direction
                update_status = _gdv_sr1_inverse_update(inverse, mapped, direction)
                if update_status == 0:
                    update_count += 1
                else:
                    update_count = 0
                scaling = direction_scale(mapped, residual)
                if scaling is None:
                    iteration = max_iterations + 1
                    break
                step_scale, _cosine = scaling
                solution += step_scale * direction
                residual = target - operator @ solution
                error = float(np.max(np.abs(residual), initial=0.0))
                iteration += 1
                if error < best_error:
                    best_solution = solution.copy()
                    best_error = error
                continue
        else:
            small_count = 0

        if small_count > maximum_small_count and update_count == 0:
            small_cosine = 0.05
            small_count = 0

        solution += step_scale * direction
        if not (
            update_count == 0 and error_change <= convergence_squared
        ) or small_count != 0:
            residual = target - operator @ solution
            error = float(np.max(np.abs(residual), initial=0.0))
            error_change = abs(previous_error - error)
            previous_error = error
            iteration += 1
            if error < best_error:
                best_solution = solution.copy()
                best_error = error
            continue

        if best_error < error:
            solution = best_solution.copy()
            error = best_error
        if error > convergence:
            break

    status = 1 if iteration > max_iterations else 0
    return solution, inverse, status


def _gdv_sr1_inverse_update(
    inverse: np.ndarray,
    displacement: np.ndarray,
    response: np.ndarray,
) -> int:
    """Apply ``utilnz.F:SR1`` to the mutable SLEqS3 inverse guess."""

    dx = np.asarray(displacement, dtype=float).reshape(-1)
    dg = np.asarray(response, dtype=float).reshape(-1)
    dx_norm = float(np.linalg.norm(dx))
    dg_norm = float(np.linalg.norm(dg))
    if dx_norm == 0.0 or dg_norm == 0.0:
        return -1
    epsilon = (dg_norm / dx_norm) * 1.0e-10
    update_limit = (dg_norm / dx_norm) / 1.0e-10
    difference = dg - inverse @ dx
    difference_norm = float(np.linalg.norm(difference))
    if difference_norm / dg_norm <= 1.0e-10:
        return -1
    denominator = float(difference @ dx)
    if abs(denominator) <= epsilon:
        if abs(difference_norm / dg_norm) <= 1.0e-10:
            return 0
        if abs(denominator / (dx_norm * dg_norm)) <= epsilon:
            return 3
        if abs(difference_norm * difference_norm / denominator) > update_limit:
            return 4
    inverse_denominator = 1.0 / denominator
    if np.sqrt(abs(difference_norm * difference_norm * inverse_denominator)) >= update_limit:
        return 5
    inverse += np.outer(difference, difference) * inverse_denominator
    return 0


def _gdv_remove_translation_rotation(
    coordinates_bohr: np.ndarray,
    displacement_bohr: np.ndarray,
) -> np.ndarray:
    """Remove the six equal-weight null vectors used by GDV ``RemTR``."""

    coordinates = np.asarray(coordinates_bohr, dtype=float)
    displacement = np.asarray(displacement_bohr, dtype=float).reshape(
        coordinates.shape
    ).copy()
    centered = coordinates - np.mean(coordinates, axis=0)
    _values, axes = np.linalg.eigh(centered.T @ centered)
    bases = []
    for axis in np.eye(3):
        bases.append(np.broadcast_to(axis, coordinates.shape))
    for index in range(3):
        bases.append(np.cross(centered, axes[:, index]))
    flat = displacement.reshape(-1)
    for basis in bases:
        vector = np.asarray(basis, dtype=float).reshape(-1)
        norm2 = float(vector @ vector)
        if norm2 > 1.0e-15:
            flat -= vector * float(flat @ vector) / norm2
    return flat


def constrained_internal_coordinate_step(
    coordinates_angstrom: np.ndarray,
    target_values: np.ndarray,
    evaluate: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    *,
    evaluate_values: Callable[[np.ndarray], np.ndarray] | None = None,
    solve_indices: tuple[int, ...],
    protected_indices: tuple[int, ...] = (),
    protected_combinations: np.ndarray | None = None,
    max_iterations: int = 30,
    tolerance: float = 1.0e-9,
    max_cartesian_step_angstrom: float = 0.20,
    fixed_atom_indices: tuple[int, ...] = (),
    project_coordinates: Callable[[np.ndarray], np.ndarray] | None = None,
) -> InternalCoordinateBackTransform:
    """Correct selected coordinates without stepping along protected rows.

    At every iteration the Cartesian correction is restricted to the numerical
    null space of the protected Wilson rows.  ``protected_combinations`` may
    additionally contain covectors in the internal-coordinate basis; their
    Cartesian constraint rows are ``C @ B``.  This permits LINK to protect a
    delocalized normal coordinate such as the SONIC Qim mode rather than only
    one named internal coordinate.  A damped least-squares solution is then
    computed for ``B_h Z y = r_h``.  This is the corrector half of the finite-
    predictor/linear-corrector SONIC back-transform.
    """

    coords = np.asarray(coordinates_angstrom, dtype=float).copy()
    target = np.asarray(target_values, dtype=float).reshape(-1)
    solve = np.asarray(tuple(dict.fromkeys(int(item) for item in solve_indices)), dtype=int)
    protected = np.asarray(tuple(dict.fromkeys(int(item) for item in protected_indices)), dtype=int)
    if np.intersect1d(solve, protected).size:
        raise ValueError("solve and protected internal-coordinate sets must be disjoint")
    if solve.size and (np.min(solve) < 0 or np.max(solve) >= target.size):
        raise ValueError("solve internal-coordinate index is outside target vector")
    if protected.size and (np.min(protected) < 0 or np.max(protected) >= target.size):
        raise ValueError("protected internal-coordinate index is outside target vector")
    combinations = _validated_constraint_combinations(protected_combinations, target.size)
    fixed_columns = _fixed_cartesian_columns(fixed_atom_indices, coords.shape[0])
    values, b_matrix = _validated_evaluation(evaluate, coords, target.size)
    residual = target - values
    initial_solve_norm = float(np.linalg.norm(residual[solve])) if solve.size else 0.0
    initial_total_norm = float(np.linalg.norm(residual))
    projector = _cartesian_projector(b_matrix, fixed_columns)
    if initial_solve_norm <= tolerance:
        return InternalCoordinateBackTransform(coords, values, residual, 0, True, projector)

    for iteration in range(1, max_iterations + 1):
        candidates = _constrained_damped_steps(
            b_matrix,
            residual,
            solve,
            _protected_constraint_rows(b_matrix, protected, combinations),
            coords.shape,
            fixed_columns=fixed_columns,
        )
        if not candidates:
            return InternalCoordinateBackTransform(
                coords, values, residual, iteration, False, projector
            )
        current_hard = float(np.linalg.norm(residual[solve]))
        current_soft = float(
            np.linalg.norm(_protected_constraint_residual(residual, protected, combinations))
        )
        current_score = current_hard**2 + 100.0 * current_soft**2
        accepted = False
        for candidate in candidates:
            step = np.asarray(candidate, dtype=float)
            step_norm = float(np.linalg.norm(step))
            if max_cartesian_step_angstrom > 0.0 and step_norm > max_cartesian_step_angstrom:
                step *= max_cartesian_step_angstrom / step_norm
            for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
                trial_coords = coords + (scale * step).reshape(coords.shape)
                if project_coordinates is not None:
                    trial_coords = np.asarray(project_coordinates(trial_coords), dtype=float)
                if evaluate_values is None:
                    trial_values, trial_b = _validated_evaluation(
                        evaluate, trial_coords, target.size
                    )
                else:
                    trial_values = _validated_values(evaluate_values, trial_coords, target.size)
                    trial_b = None
                trial_residual = target - trial_values
                trial_hard = float(np.linalg.norm(trial_residual[solve]))
                trial_soft = float(
                    np.linalg.norm(
                        _protected_constraint_residual(trial_residual, protected, combinations)
                    )
                )
                score = trial_hard**2 + 100.0 * trial_soft**2
                if trial_hard < current_hard and score < current_score:
                    if trial_b is None:
                        trial_values, trial_b = _validated_evaluation(
                            evaluate, trial_coords, target.size
                        )
                        trial_residual = target - trial_values
                    coords, values, b_matrix, residual = (
                        trial_coords,
                        trial_values,
                        trial_b,
                        trial_residual,
                    )
                    accepted = True
                    break
            if accepted:
                break
        projector = _cartesian_projector(b_matrix, fixed_columns)
        if not accepted:
            return InternalCoordinateBackTransform(
                coords, values, residual, iteration, False, projector
            )
        hard_norm = float(np.linalg.norm(residual[solve]))
        if hard_norm <= tolerance * max(1.0, initial_solve_norm, initial_total_norm):
            return InternalCoordinateBackTransform(
                coords, values, residual, iteration, True, projector
            )
    return InternalCoordinateBackTransform(
        coords, values, residual, max_iterations, False, projector
    )


def secant_projector_update(
    cartesian_from_q: np.ndarray,
    previous_q: np.ndarray,
    previous_coordinates: np.ndarray,
    current_q: np.ndarray,
    current_coordinates: np.ndarray,
) -> ProjectorSecantUpdate:
    """Apply the shared rank-one secant update to an internal-coordinate projector."""

    matrix = np.asarray(cartesian_from_q, dtype=float)
    q_delta = np.asarray(current_q, dtype=float).reshape(-1) - np.asarray(
        previous_q, dtype=float
    ).reshape(-1)
    x_delta = np.asarray(current_coordinates, dtype=float).reshape(-1) - np.asarray(
        previous_coordinates, dtype=float
    ).reshape(-1)
    if matrix.shape != (x_delta.size, q_delta.size):
        return ProjectorSecantUpdate(None, float("inf"), False)
    denominator = float(q_delta @ q_delta)
    if denominator <= 1.0e-24 or not np.isfinite(denominator):
        return ProjectorSecantUpdate(None, float("inf"), False)
    residual = x_delta - matrix @ q_delta
    relative_error = float(np.linalg.norm(residual) / max(np.linalg.norm(x_delta), 1.0e-12))
    correction = np.outer(residual, q_delta) / denominator
    updated = matrix + correction
    if not np.all(np.isfinite(updated)):
        return ProjectorSecantUpdate(None, relative_error, False)
    if relative_error > 0.75 or np.linalg.norm(correction) > max(
        0.75 * np.linalg.norm(matrix), 1.0e-8
    ):
        return ProjectorSecantUpdate(None, relative_error, False)
    return ProjectorSecantUpdate(updated, relative_error, True)


def should_refresh_coordinate_model(
    *,
    model_age: int,
    line_search_scale: float,
    trust_ratio: float,
    secant_relative_error: float,
) -> bool:
    """Shared MORPHEUS/TRINITY policy for refreshing a nonlinear coordinate model."""

    return bool(
        model_age >= 3
        or line_search_scale < 0.5
        or not np.isfinite(secant_relative_error)
        or secant_relative_error > 0.5
        or trust_ratio < 0.25
        or trust_ratio > 2.5
    )


def transport_internal_hessian(
    hessian: np.ndarray,
    old_b_matrix: np.ndarray,
    new_cartesian_from_q: np.ndarray,
) -> np.ndarray:
    """Transport an internal Hessian to the tangent space at a new geometry."""

    h = np.asarray(hessian, dtype=float)
    tangent_map = np.asarray(old_b_matrix, dtype=float) @ np.asarray(
        new_cartesian_from_q, dtype=float
    )
    transported = tangent_map.T @ h @ tangent_map
    return 0.5 * (transported + transported.T)


def _cartesian_projector(
    b_matrix: np.ndarray,
    fixed_columns: np.ndarray | None = None,
) -> np.ndarray:
    return cartesian_from_internal_jacobian(
        b_matrix,
        fixed_cartesian_columns=fixed_columns,
    )


def _damped_internal_step(
    b_matrix: np.ndarray,
    residual: np.ndarray,
    coordinate_shape: tuple[int, ...],
    fixed_columns: np.ndarray | None = None,
) -> tuple[np.ndarray, ...] | None:
    b = np.asarray(b_matrix, dtype=float)
    free = np.ones(b.shape[1], dtype=bool)
    if fixed_columns is not None and fixed_columns.size:
        free[fixed_columns] = False
    b_free = b[:, free]
    rhs = np.asarray(residual, dtype=float)
    metric = b_free @ b_free.T
    identity = np.eye(metric.shape[0], dtype=float)
    candidates: list[np.ndarray] = []
    for damping in (1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2, 1.0, 100.0):
        try:
            free_step = b_free.T @ np.linalg.solve(metric + damping * identity, rhs)
        except np.linalg.LinAlgError:
            continue
        step = np.zeros(b.shape[1], dtype=float)
        step[free] = free_step
        if step.shape == (int(np.prod(coordinate_shape)),) and np.all(np.isfinite(step)):
            candidates.append(step)
    return tuple(candidates) if candidates else None


def _constrained_damped_steps(
    b_matrix: np.ndarray,
    residual: np.ndarray,
    solve_indices: np.ndarray,
    protected_rows: np.ndarray,
    coordinate_shape: tuple[int, ...],
    *,
    fixed_columns: np.ndarray | None = None,
) -> tuple[np.ndarray, ...]:
    b = np.asarray(b_matrix, dtype=float)
    free = np.ones(b.shape[1], dtype=bool)
    if fixed_columns is not None and fixed_columns.size:
        free[fixed_columns] = False
    free_count = int(np.count_nonzero(free))
    if free_count == 0 or solve_indices.size == 0:
        return ()
    hard = b[solve_indices, :][:, free]
    if protected_rows.size:
        soft = np.asarray(protected_rows, dtype=float)[:, free]
        _u, singular, vh = np.linalg.svd(soft, full_matrices=True)
        cutoff = 1.0e-10 * max(float(singular[0]) if singular.size else 0.0, 1.0)
        rank = int(np.count_nonzero(singular > cutoff))
        null_basis = vh[rank:, :].T
    else:
        null_basis = np.eye(free_count, dtype=float)
    if null_basis.shape[1] == 0:
        return ()
    reduced = hard @ null_basis
    rhs = np.asarray(residual, dtype=float)[solve_indices]
    metric = reduced @ reduced.T
    identity = np.eye(metric.shape[0], dtype=float)
    candidates: list[np.ndarray] = []
    for damping in (1.0e-10, 1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2, 1.0):
        try:
            reduced_step = reduced.T @ np.linalg.solve(metric + damping * identity, rhs)
        except np.linalg.LinAlgError:
            continue
        free_step = null_basis @ reduced_step
        step = np.zeros(b.shape[1], dtype=float)
        step[free] = free_step
        if step.shape == (int(np.prod(coordinate_shape)),) and np.all(np.isfinite(step)):
            candidates.append(step)
    return tuple(candidates)


def _validated_constraint_combinations(
    combinations: np.ndarray | None, coordinate_count: int
) -> np.ndarray:
    if combinations is None:
        return np.empty((0, coordinate_count), dtype=float)
    matrix = np.asarray(combinations, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.ndim != 2 or matrix.shape[1] != coordinate_count:
        raise ValueError(
            "protected internal-coordinate combinations must have one column per coordinate"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("protected internal-coordinate combinations must be finite")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms <= 1.0e-14):
        raise ValueError("protected internal-coordinate combination has zero norm")
    return matrix / norms[:, None]


def _protected_constraint_rows(
    b_matrix: np.ndarray,
    protected_indices: np.ndarray,
    combinations: np.ndarray,
) -> np.ndarray:
    rows = []
    if protected_indices.size:
        rows.append(np.asarray(b_matrix, dtype=float)[protected_indices, :])
    if combinations.size:
        rows.append(combinations @ np.asarray(b_matrix, dtype=float))
    if not rows:
        return np.empty((0, np.asarray(b_matrix).shape[1]), dtype=float)
    return np.vstack(rows)


def _protected_constraint_residual(
    residual: np.ndarray,
    protected_indices: np.ndarray,
    combinations: np.ndarray,
) -> np.ndarray:
    values = []
    if protected_indices.size:
        values.append(np.asarray(residual, dtype=float)[protected_indices])
    if combinations.size:
        values.append(combinations @ np.asarray(residual, dtype=float))
    return np.concatenate(values) if values else np.empty(0, dtype=float)


def _fixed_cartesian_columns(atom_indices: tuple[int, ...], natoms: int) -> np.ndarray:
    atoms = tuple(int(index) for index in atom_indices)
    if not atoms:
        return np.asarray([], dtype=int)
    if len(set(atoms)) != len(atoms) or min(atoms) < 0 or max(atoms) >= natoms:
        raise ValueError("fixed atom indices are invalid")
    return np.asarray(
        [3 * atom + component for atom in atoms for component in range(3)],
        dtype=int,
    )


def _validated_evaluation(
    evaluate: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    coordinates: np.ndarray,
    coordinate_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    values, b_matrix = evaluate(coordinates)
    values = np.asarray(values, dtype=float).reshape(-1)
    b_matrix = np.asarray(b_matrix, dtype=float)
    if values.shape != (coordinate_count,) or b_matrix.shape != (
        coordinate_count,
        coordinates.size,
    ):
        raise ValueError("internal-coordinate evaluator returned incompatible dimensions")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(b_matrix)):
        raise ValueError("internal-coordinate evaluator returned non-finite values")
    return values, b_matrix


def _validated_values(
    evaluate_values: Callable[[np.ndarray], np.ndarray],
    coordinates: np.ndarray,
    coordinate_count: int,
) -> np.ndarray:
    values = np.asarray(evaluate_values(coordinates), dtype=float).reshape(-1)
    if values.shape != (coordinate_count,):
        raise ValueError("internal-coordinate value evaluator returned incompatible dimensions")
    if not np.all(np.isfinite(values)):
        raise ValueError("internal-coordinate value evaluator returned non-finite values")
    return values
