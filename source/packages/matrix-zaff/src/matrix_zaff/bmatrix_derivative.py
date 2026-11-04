"""Sparse analytic first Cartesian derivative of a frozen SONIC Wilson B matrix."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import os
from typing import Callable, Sequence

import numpy as np

from matrix_smith.bmatrix import SparseBMatrix, SparseBRow
from matrix_smith.evaluation import build_gic_b_matrix
from matrix_smith.models import FrozenGIC, GICDefinition, GICPrimitive
from matrix_smith.numerics import _sparse_analytic_b_row
from matrix_smith.derivative_capabilities import primitive_b_prime_capability
from .primitive_second_derivative import analytic_primitive_second_derivative


B_MATRIX_DERIVATIVE_BACKEND = "architect-sparse-analytic-bmatrix-first-derivative.v1"
NUMERICAL_B_MATRIX_DERIVATIVE_BACKEND = "architect-sparse-numerical-bmatrix-first-derivative.v1"
GENERAL_NUMERICAL_B_MATRIX_DERIVATIVE_BACKEND = (
    "architect-sparse-general-bmatrix-first-derivative.v1"
)
BOHR_TO_ANGSTROM = 0.529177210903


@dataclass(frozen=True)
class SparseBMatrixDerivativeSlice:
    """Sparse matrix ``dB/dx_k`` for one Cartesian displacement column."""

    derivative_column: int
    rows: tuple[tuple[int, SparseBRow], ...]

    def __post_init__(self) -> None:
        column = int(self.derivative_column)
        canonical = tuple(sorted((int(index), row) for index, row in self.rows))
        if column < 0 or len({index for index, _row in canonical}) != len(canonical):
            raise ValueError("invalid sparse Wilson-B derivative slice")
        if any(index < 0 for index, _row in canonical):
            raise ValueError("Wilson-B derivative row indices must be non-negative")
        object.__setattr__(self, "derivative_column", column)
        object.__setattr__(self, "rows", canonical)

    @property
    def nnz(self) -> int:
        return sum(row.nnz for _index, row in self.rows)

    def to_dense(self, row_count: int, column_count: int) -> np.ndarray:
        matrix = np.zeros((int(row_count), int(column_count)), dtype=float)
        for row_index, row in self.rows:
            if row_index >= int(row_count) or row.size != int(column_count):
                raise ValueError("sparse Wilson-B derivative slice has incompatible dimensions")
            matrix[row_index] = row.to_dense()
        return matrix

    def to_dict(self) -> dict[str, object]:
        return {
            "derivative_column_zero_based": self.derivative_column,
            "rows": [
                {
                    "row_zero_based": row_index,
                    "entries": [
                        {"b_column_zero_based": column, "value_per_angstrom2": value}
                        for column, value in row.entries
                    ],
                }
                for row_index, row in self.rows
            ],
        }


@dataclass(frozen=True)
class SparseBMatrixDerivative:
    r"""Sparse tensor :math:`D_{iab}=\partial B_{ia}/\partial x_b`.

    The last tensor index is the displaced Cartesian component.  This layout
    makes the nonstationary curvature term
    :math:`C_{ab}=\sum_i g_i D_{iab}` directly available without materializing
    the complete dense three-index tensor.
    """

    row_count: int
    column_count: int
    slices: tuple[SparseBMatrixDerivativeSlice, ...]
    coordinate_labels: tuple[str, ...]
    cartesian_columns: tuple[str, ...]
    step_angstrom: float
    zero_tolerance_per_angstrom2: float
    workers: int
    analytic_primitive_count: int = 0
    numerical_fallback_primitives: tuple[str, ...] = ()
    stencil: str = "ANALYTIC_SECOND_ORDER_FORWARD_AD"
    backend: str = B_MATRIX_DERIVATIVE_BACKEND

    def __post_init__(self) -> None:
        row_count = int(self.row_count)
        column_count = int(self.column_count)
        if row_count < 0 or column_count < 0:
            raise ValueError("Wilson-B derivative dimensions must be non-negative")
        if len(self.coordinate_labels) != row_count:
            raise ValueError("Wilson-B derivative coordinate labels do not match its rows")
        if len(self.cartesian_columns) != column_count:
            raise ValueError("Wilson-B derivative Cartesian labels do not match its columns")
        columns = tuple(item.derivative_column for item in self.slices)
        if columns != tuple(range(column_count)):
            raise ValueError("Wilson-B derivative slices must cover Cartesian columns in order")
        if not np.isfinite(self.step_angstrom) or self.step_angstrom <= 0.0:
            raise ValueError("Wilson-B derivative step must be positive and finite")
        if self.zero_tolerance_per_angstrom2 < 0.0 or self.workers < 1:
            raise ValueError("Wilson-B derivative tolerance/workers are invalid")
        if self.analytic_primitive_count < 0:
            raise ValueError("analytic primitive count must be non-negative")
        object.__setattr__(self, "row_count", row_count)
        object.__setattr__(self, "column_count", column_count)
        object.__setattr__(
            self,
            "numerical_fallback_primitives",
            tuple(str(item) for item in self.numerical_fallback_primitives),
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.row_count, self.column_count, self.column_count

    @property
    def nnz(self) -> int:
        return sum(item.nnz for item in self.slices)

    @property
    def density(self) -> float:
        total = self.row_count * self.column_count * self.column_count
        return 0.0 if total == 0 else float(self.nnz) / float(total)

    def to_dense(self) -> np.ndarray:
        tensor = np.zeros(self.shape, dtype=float)
        for item in self.slices:
            tensor[:, :, item.derivative_column] = item.to_dense(
                self.row_count, self.column_count
            )
        return tensor

    def contract_internal_gradient(
        self,
        internal_gradient: np.ndarray,
        *,
        symmetrize: bool = True,
    ) -> np.ndarray:
        """Return ``sum_i g_i dB_i/dx`` for an off-equilibrium Hessian chain rule."""

        gradient = np.asarray(internal_gradient, dtype=float).reshape(-1)
        if gradient.shape != (self.row_count,):
            raise ValueError("internal gradient does not match Wilson-B derivative rows")
        curvature = np.zeros((self.column_count, self.column_count), dtype=float)
        for item in self.slices:
            column = item.derivative_column
            for row_index, row in item.rows:
                coefficient = float(gradient[row_index])
                if coefficient == 0.0:
                    continue
                for b_column, value in row.entries:
                    curvature[b_column, column] += coefficient * value
        if symmetrize:
            curvature = 0.5 * (curvature + curvature.T)
        return curvature

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "matrix.architect.sparse_bmatrix_first_derivative.v1",
            "backend": self.backend,
            "tensor_convention": "D[i,a,b]=dB[i,a]/dx[b]",
            "units": "coordinate_unit_per_angstrom2",
            "shape": list(self.shape),
            "coordinate_labels": list(self.coordinate_labels),
            "cartesian_columns": list(self.cartesian_columns),
            "step_angstrom": self.step_angstrom,
            "step_role": (
                "NUMERICAL_AUDIT_OR_FALLBACK_ONLY"
                if self.stencil == "ANALYTIC_SECOND_ORDER_FORWARD_AD"
                else "PRODUCTION_STENCIL"
            ),
            "stencil": self.stencil,
            "zero_tolerance_per_angstrom2": self.zero_tolerance_per_angstrom2,
            "workers": self.workers,
            "analytic_primitive_count": self.analytic_primitive_count,
            "numerical_fallback_primitives": list(self.numerical_fallback_primitives),
            "nnz": self.nnz,
            "density": self.density,
            "slices": [item.to_dict() for item in self.slices],
        }


@dataclass(frozen=True)
class CurvilinearHessianTransform:
    """Auditable Cartesian-to-internal Hessian transformation using B-prime."""

    hessian_internal: np.ndarray
    linear_hessian_internal: np.ndarray
    cartesian_coordinate_curvature_hartree_per_bohr2: np.ndarray
    internal_gradient: np.ndarray
    b_prime_backend: str
    b_prime_nnz: int
    numerical_fallback_primitives: tuple[str, ...]

    @property
    def correction_norm(self) -> float:
        return float(np.linalg.norm(self.hessian_internal - self.linear_hessian_internal))


def curvilinear_internal_hessian_from_cartesian(
    definition: GICDefinition,
    cartesian_hessian_hartree_per_bohr2: np.ndarray,
    cartesian_gradient_hartree_per_bohr: np.ndarray,
    *,
    coordinates_angstrom: np.ndarray | None = None,
    cartesian_from_internal_bohr: np.ndarray | None = None,
    coordinate_indices: Sequence[int] | None = None,
    parallel_workers: int = 0,
) -> CurvilinearHessianTransform:
    r"""Transform a Cartesian Hessian with the exact off-equilibrium chain rule.

    With :math:`B=\partial q/\partial x`, the Cartesian Hessian satisfies

    .. math:: H_x = B^T H_q B + \sum_i g_i^q B'_i.

    ARCHITECT constructs the sparse analytic :math:`B'`; callers such as LINK
    supply the frozen coordinate contract and consume the transformed Hessian.
    """

    working_definition = _selected_gic_definition(definition, coordinate_indices)
    coordinates = np.asarray(
        definition.reference_coordinates_angstrom
        if coordinates_angstrom is None
        else coordinates_angstrom,
        dtype=float,
    )
    b_matrix = np.asarray(
        build_gic_b_matrix(
            working_definition,
            coordinates_angstrom=coordinates,
            parallel_workers=max(1, int(parallel_workers)),
        ).rows,
        dtype=float,
    )
    ncart = b_matrix.shape[1]
    hessian = np.asarray(cartesian_hessian_hartree_per_bohr2, dtype=float)
    gradient_bohr = np.asarray(cartesian_gradient_hartree_per_bohr, dtype=float).reshape(-1)
    if hessian.shape != (ncart, ncart) or gradient_bohr.shape != (ncart,):
        raise ValueError("Cartesian Hessian/gradient do not match the frozen coordinate contract")
    if not np.all(np.isfinite(hessian)) or not np.all(np.isfinite(gradient_bohr)):
        raise ValueError("Cartesian Hessian/gradient contain non-finite values")

    derivative = build_sparse_gic_b_matrix_derivative(
        working_definition,
        coordinates_angstrom=coordinates,
        parallel_workers=parallel_workers,
    )
    # B is expressed per angstrom, whereas electronic gradients use bohr.
    gradient_angstrom = gradient_bohr / BOHR_TO_ANGSTROM
    internal_gradient, *_ = np.linalg.lstsq(
        b_matrix.T,
        gradient_angstrom,
        rcond=1.0e-10,
    )
    coordinate_curvature_angstrom2 = derivative.contract_internal_gradient(
        internal_gradient
    )
    coordinate_curvature_bohr2 = (
        coordinate_curvature_angstrom2 * BOHR_TO_ANGSTROM**2
    )
    corrected_cartesian = 0.5 * (
        hessian + hessian.T - 2.0 * coordinate_curvature_bohr2
    )
    if cartesian_from_internal_bohr is None:
        cartesian_from_internal = (
            np.linalg.pinv(b_matrix, rcond=1.0e-8) / BOHR_TO_ANGSTROM
        )
    else:
        cartesian_from_internal = np.asarray(
            cartesian_from_internal_bohr, dtype=float
        )
    if cartesian_from_internal.ndim != 2 or cartesian_from_internal.shape[0] != ncart:
        raise ValueError("Cartesian-from-internal Jacobian has incompatible dimensions")
    linear = cartesian_from_internal.T @ hessian @ cartesian_from_internal
    transformed = (
        cartesian_from_internal.T @ corrected_cartesian @ cartesian_from_internal
    )
    return CurvilinearHessianTransform(
        hessian_internal=0.5 * (transformed + transformed.T),
        linear_hessian_internal=0.5 * (linear + linear.T),
        cartesian_coordinate_curvature_hartree_per_bohr2=(
            coordinate_curvature_bohr2
        ),
        internal_gradient=np.asarray(internal_gradient, dtype=float),
        b_prime_backend=derivative.backend,
        b_prime_nnz=derivative.nnz,
        numerical_fallback_primitives=derivative.numerical_fallback_primitives,
    )


def build_sparse_gic_b_matrix_derivative(
    definition: GICDefinition,
    *,
    coordinates_angstrom: np.ndarray | None = None,
    coordinate_indices: Sequence[int] | None = None,
    step_angstrom: float = 1.0e-4,
    zero_tolerance_per_angstrom2: float = 1.0e-10,
    parallel_workers: int = 0,
) -> SparseBMatrixDerivative:
    """Build analytic sparse ``dB/dx`` from primitive second derivatives.

    Each required primitive is evaluated by second-order forward automatic
    differentiation on its local Cartesian support.  When ``coordinate_indices``
    is provided, primitives that do not contribute to those SONIC/GIC rows are
    never evaluated.  The selected Hessian rows are assembled by the same
    frozen linear coefficients used for B.  A fourth-order numerical derivative
    is used only for a singular or currently unsupported primitive and is
    reported explicitly in the returned contract.
    """

    coordinates = np.asarray(
        definition.reference_coordinates_angstrom
        if coordinates_angstrom is None
        else coordinates_angstrom,
        dtype=float,
    )
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("Wilson-B derivative coordinates must have shape (natoms, 3)")
    step = float(step_angstrom)
    threshold = float(zero_tolerance_per_angstrom2)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("Wilson-B derivative step must be positive and finite")
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("Wilson-B derivative zero tolerance must be finite and non-negative")

    selected_definition = _selected_gic_definition(definition, coordinate_indices)
    selected_gics = selected_definition.gics
    selected_primitives = selected_definition.primitives
    reference_coordinates = np.asarray(definition.reference_coordinates_angstrom, dtype=float)
    workers = _resolved_workers(parallel_workers, len(selected_primitives))
    reference = build_gic_b_matrix(
        selected_definition,
        coordinates_angstrom=coordinates,
        parallel_workers=workers,
    )
    row_count = len(reference.rows)
    column_count = coordinates.size
    oracle_primitive_backend = definition.backend == "matrix-smith-gicforge.v1"

    def differentiate_primitive(
        primitive: GICPrimitive,
    ) -> tuple[str, tuple[tuple[int, int, float], ...], bool]:
        capability = primitive_b_prime_capability(primitive)
        if not capability.analytic:
            entries = _numerical_primitive_second_derivative(
                primitive,
                coordinates,
                reference_coordinates=reference_coordinates,
                step_angstrom=step,
                zero_tolerance_per_angstrom2=threshold,
                oracle_primitive_backend=oracle_primitive_backend,
            )
            return primitive.identifier, entries, False
        try:
            result = analytic_primitive_second_derivative(
                primitive,
                coordinates,
                reference_coordinates=reference_coordinates,
                oracle_primitive_convention=oracle_primitive_backend,
                zero_tolerance=threshold,
            )
            return primitive.identifier, result.entries, True
        except (FloatingPointError, NotImplementedError, ZeroDivisionError):
            entries = _numerical_primitive_second_derivative(
                primitive,
                coordinates,
                reference_coordinates=reference_coordinates,
                step_angstrom=step,
                zero_tolerance_per_angstrom2=threshold,
                oracle_primitive_backend=oracle_primitive_backend,
            )
            return primitive.identifier, entries, False

    if workers == 1:
        primitive_results = tuple(
            differentiate_primitive(primitive) for primitive in selected_primitives
        )
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            primitive_results = tuple(pool.map(differentiate_primitive, selected_primitives))

    derivative_by_primitive = {
        identifier: entries for identifier, entries, _analytic in primitive_results
    }
    fallback = tuple(
        identifier for identifier, _entries, analytic in primitive_results if not analytic
    )
    row_entries_by_slice: list[list[tuple[int, SparseBRow]]] = [
        [] for _column in range(column_count)
    ]
    for row_index, gic in enumerate(selected_gics):
        coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
        accumulator: dict[tuple[int, int], float] = {}
        for primitive_id, coefficient in coefficients:
            if primitive_id not in derivative_by_primitive:
                raise ValueError(
                    f"unknown primitive {primitive_id!r} in frozen GIC {gic.identifier}"
                )
            for b_column, derivative_column, value in derivative_by_primitive[primitive_id]:
                key = (b_column, derivative_column)
                accumulator[key] = accumulator.get(key, 0.0) + float(coefficient) * value
        by_derivative_column: dict[int, list[tuple[int, float]]] = {}
        for (b_column, derivative_column), value in sorted(accumulator.items()):
            if abs(value) > threshold:
                by_derivative_column.setdefault(derivative_column, []).append((b_column, value))
        for derivative_column, entries in by_derivative_column.items():
            row_entries_by_slice[derivative_column].append(
                (row_index, SparseBRow(column_count, tuple(entries)))
            )

    slices = tuple(
        SparseBMatrixDerivativeSlice(column, tuple(rows))
        for column, rows in enumerate(row_entries_by_slice)
    )
    return SparseBMatrixDerivative(
        row_count=row_count,
        column_count=column_count,
        slices=slices,
        coordinate_labels=reference.coordinate_labels,
        cartesian_columns=reference.cartesian_columns,
        step_angstrom=step,
        zero_tolerance_per_angstrom2=threshold,
        workers=workers,
        analytic_primitive_count=len(primitive_results) - len(fallback),
        numerical_fallback_primitives=fallback,
    )


def build_sparse_gic_b_matrix_derivative_numerical(
    definition: GICDefinition,
    *,
    coordinates_angstrom: np.ndarray | None = None,
    step_angstrom: float = 1.0e-4,
    zero_tolerance_per_angstrom2: float = 1.0e-10,
    parallel_workers: int = 0,
) -> SparseBMatrixDerivative:
    """Audit ``dB/dx`` by a parallel fourth-order centered stencil."""

    coordinates = np.asarray(
        definition.reference_coordinates_angstrom
        if coordinates_angstrom is None
        else coordinates_angstrom,
        dtype=float,
    )
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("Wilson-B derivative coordinates must have shape (natoms, 3)")
    step = float(step_angstrom)
    threshold = float(zero_tolerance_per_angstrom2)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("Wilson-B derivative step must be positive and finite")
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("Wilson-B derivative zero tolerance must be finite and non-negative")

    reference = build_gic_b_matrix(definition, coordinates_angstrom=coordinates)
    return build_sparse_b_matrix_derivative_numerical(
        lambda point: build_gic_b_matrix(
            definition,
            coordinates_angstrom=point,
        ).sparse_matrix(),
        coordinates,
        coordinate_labels=reference.coordinate_labels,
        cartesian_columns=reference.cartesian_columns,
        step_angstrom=step,
        zero_tolerance_per_angstrom2=threshold,
        parallel_workers=parallel_workers,
        numerical_fallback_primitives=tuple(
            primitive.identifier for primitive in definition.primitives
        ),
        backend=NUMERICAL_B_MATRIX_DERIVATIVE_BACKEND,
    )


def build_sparse_b_matrix_derivative_numerical(
    evaluate_b_matrix: Callable[[np.ndarray], SparseBMatrix],
    coordinates_angstrom: np.ndarray,
    *,
    coordinate_labels: Sequence[str] | None = None,
    cartesian_columns: Sequence[str] | None = None,
    step_angstrom: float = 1.0e-4,
    zero_tolerance_per_angstrom2: float = 1.0e-10,
    parallel_workers: int = 0,
    numerical_fallback_primitives: Sequence[str] = (),
    backend: str = GENERAL_NUMERICAL_B_MATRIX_DERIVATIVE_BACKEND,
) -> SparseBMatrixDerivative:
    """Differentiate any canonical sparse Wilson B evaluator.

    This is the representation-independent form of the existing fourth-order
    audit stencil.  It is used for composite typed-ONIC charts, where the
    direct sum may contain Cartesian, inverse-distance, natural-internal and
    exponential-map rows in one frozen order.
    """

    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or not np.all(np.isfinite(coordinates)):
        raise ValueError("Wilson-B derivative coordinates must be finite with shape (natoms, 3)")
    step = float(step_angstrom)
    threshold = float(zero_tolerance_per_angstrom2)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("Wilson-B derivative step must be positive and finite")
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("Wilson-B derivative zero tolerance must be finite and non-negative")
    reference = evaluate_b_matrix(coordinates)
    if not isinstance(reference, SparseBMatrix):
        raise TypeError("general B-prime evaluator must return SparseBMatrix")
    row_count = reference.row_count
    column_count = coordinates.size
    if reference.column_count != column_count:
        raise ValueError("general B-prime evaluator has an incompatible Cartesian dimension")
    labels = tuple(str(item) for item in (coordinate_labels or reference.row_labels))
    if len(labels) != row_count:
        raise ValueError("general B-prime coordinate labels do not match B rows")
    columns = tuple(str(item) for item in (cartesian_columns or _cartesian_labels(len(coordinates))))
    if len(columns) != column_count:
        raise ValueError("general B-prime Cartesian labels do not match B columns")
    workers = _resolved_workers(parallel_workers, column_count)

    def differentiate(column: int) -> SparseBMatrixDerivativeSlice:
        displaced: list[SparseBMatrix] = []
        for multiplier in (-2, -1, 1, 2):
            point = coordinates.reshape(-1).copy()
            point[column] += multiplier * step
            matrix = evaluate_b_matrix(point.reshape(coordinates.shape))
            if (
                not isinstance(matrix, SparseBMatrix)
                or matrix.row_count != row_count
                or matrix.column_count != column_count
                or (matrix.row_labels and matrix.row_labels != reference.row_labels)
            ):
                raise ValueError("general B-prime evaluator changed its frozen matrix contract")
            displaced.append(matrix)
        minus_two, minus_one, plus_one, plus_two = displaced
        scale = 1.0 / (12.0 * step)
        rows: list[tuple[int, SparseBRow]] = []
        for row_index in range(row_count):
            derivative = SparseBRow.combine(
                (scale, minus_two.rows[row_index]),
                (-8.0 * scale, minus_one.rows[row_index]),
                (8.0 * scale, plus_one.rows[row_index]),
                (-scale, plus_two.rows[row_index]),
            )
            filtered = SparseBRow(
                derivative.size,
                tuple(
                    (index, value)
                    for index, value in derivative.entries
                    if abs(value) > threshold
                ),
            )
            if filtered.nnz:
                rows.append((row_index, filtered))
        return SparseBMatrixDerivativeSlice(column, tuple(rows))

    if workers == 1:
        slices = tuple(differentiate(column) for column in range(column_count))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            slices = tuple(pool.map(differentiate, range(column_count)))
    return SparseBMatrixDerivative(
        row_count=row_count,
        column_count=column_count,
        slices=slices,
        coordinate_labels=labels,
        cartesian_columns=columns,
        step_angstrom=step,
        zero_tolerance_per_angstrom2=threshold,
        workers=workers,
        analytic_primitive_count=0,
        numerical_fallback_primitives=tuple(str(item) for item in numerical_fallback_primitives),
        stencil="FIVE_POINT_CENTERED_O4",
        backend=str(backend),
    )


def _cartesian_labels(natoms: int) -> tuple[str, ...]:
    return tuple(
        f"{axis}{atom}"
        for atom in range(1, int(natoms) + 1)
        for axis in ("X", "Y", "Z")
    )


def _numerical_primitive_second_derivative(
    primitive: GICPrimitive,
    coordinates: np.ndarray,
    *,
    reference_coordinates: np.ndarray,
    step_angstrom: float,
    zero_tolerance_per_angstrom2: float,
    oracle_primitive_backend: bool,
) -> tuple[tuple[int, int, float], ...]:
    """Numerical fallback for one primitive, never silently used as the default."""

    column_count = coordinates.size
    scale = 1.0 / (12.0 * step_angstrom)
    result: list[tuple[int, int, float]] = []
    for derivative_column in range(column_count):
        rows: list[SparseBRow] = []
        for multiplier in (-2, -1, 1, 2):
            point = coordinates.reshape(-1).copy()
            point[derivative_column] += multiplier * step_angstrom
            displaced = point.reshape(coordinates.shape)
            if oracle_primitive_backend:
                from matrix_chem.primitive_coordinates import grad_primitive

                from matrix_smith.evaluation import _coordinate_kernel_primitive_from_gic_primitive

                oracle_primitive = _coordinate_kernel_primitive_from_gic_primitive(primitive)
                rows.append(
                    SparseBRow.from_dense(grad_primitive(oracle_primitive, displaced).reshape(-1))
                )
            else:
                rows.append(
                    _sparse_analytic_b_row(
                        primitive,
                        displaced,
                        reference_coords=reference_coordinates,
                    )
                )
        derivative = SparseBRow.combine(
            (scale, rows[0]),
            (-8.0 * scale, rows[1]),
            (8.0 * scale, rows[2]),
            (-scale, rows[3]),
        )
        result.extend(
            (b_column, derivative_column, value)
            for b_column, value in derivative.entries
            if abs(value) > zero_tolerance_per_angstrom2
        )
    return tuple(result)


def build_sparse_pic_b_matrix_derivative(
    definition: GICDefinition,
    **kwargs: object,
) -> SparseBMatrixDerivative:
    """Evaluate the same sparse derivative contract in the redundant PIC basis."""

    pic_definition = replace(
        definition,
        gics=tuple(
            FrozenGIC(
                identifier=f"PIC{index:04d}",
                name=primitive.name,
                family=primitive.family,
                irrep="UNPROJECTED",
                primitive_id=primitive.identifier,
                gaussian_expression=primitive.gaussian_expression(),
                coefficients=((primitive.identifier, 1.0),),
            )
            for index, primitive in enumerate(definition.primitives, start=1)
        ),
    )
    return build_sparse_gic_b_matrix_derivative(pic_definition, **kwargs)


def _resolved_workers(requested: int, job_count: int) -> int:
    if job_count < 1:
        return 1
    if requested < 0:
        raise ValueError("parallel_workers must be non-negative")
    available = max(1, int(os.cpu_count() or 1))
    return min(job_count, available if requested == 0 else max(1, int(requested)))


def _selected_gic_definition(
    definition: GICDefinition,
    coordinate_indices: Sequence[int] | None,
) -> GICDefinition:
    if coordinate_indices is None:
        return definition
    indices = tuple(int(index) for index in coordinate_indices)
    if not indices:
        raise ValueError("SONIC derivative row selection must not be empty")
    if len(set(indices)) != len(indices):
        raise ValueError("SONIC derivative row selection contains duplicates")
    if any(index < 0 or index >= len(definition.gics) for index in indices):
        raise IndexError("SONIC derivative row selection is outside the definition")
    selected_gics = tuple(definition.gics[index] for index in indices)
    required_primitive_ids = {
        primitive_id
        for gic in selected_gics
        for primitive_id, _coefficient in (
            gic.coefficients or ((gic.primitive_id, 1.0),)
        )
    }
    selected_primitives = tuple(
        primitive
        for primitive in definition.primitives
        if primitive.identifier in required_primitive_ids
    )
    if {primitive.identifier for primitive in selected_primitives} != required_primitive_ids:
        raise ValueError("selected SONIC rows reference an unknown primitive")
    return replace(
        definition,
        primitives=selected_primitives,
        gics=selected_gics,
        target_rank=len(selected_gics),
        rank=len(selected_gics),
    )
