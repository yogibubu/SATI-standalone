"""Coordinate-value and B-matrix evaluation for frozen SMITH definitions."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np

from .bmatrix import SPARSE_ZERO_TOLERANCE, SparseBMatrix, SparseBRow
from .contracts import GICForgeContractError
from .models import GICBMatrix, GICDefinition, GICPrimitive
from .numerics import (
    _angle_component_terms_from_refs,
    _fragment_relative_frames,
    _fragment_frame_anchor_atoms,
    _ring_pucker_terms_from_refs,
    _linear_fragment_stereographic_values,
    _primitive_value,
    _rotation_vector,
    _sparse_analytic_b_row,
)
from .policy import B_MATRIX_BACKEND
from .parallel import resolve_local_workers


# ABI-stable primitive kinds consumed by the compiled SMITH evaluator.  The
# inverse-distance codes are intentionally 23--25: codes 7--9 belong to the
# smaller ORACLE local-primitive kernel, whereas the complete SMITH atlas has
# its own namespace (see _smith_primitive_native.hpp).  Keeping this table in
# one place prevents the fullerene IR chart from being rejected as an unknown
# function when the two kernels are rebuilt independently.
_COMPILED_GIC_FUNCTION_CODES = {
    "R": 1, "A": 2, "D": 3, "U": 4, "H": 5, "L": 6,
    "IMPD": 7, "FC_DIST": 8, "FCA_DIST": 9,
    "CENTER_ATOM_DIST": 9, "FTRANS": 10, "FLIN_TRANS": 11,
    "RPCB": 12, "RPCK": 13, "RPU": 14, "FAXIS": 15, "FROT": 16,
    "FACE_FLAP": 18, "CENTER_FACE_DIST": 19,
    "FACE_DIRECTION_ANGLE": 20, "FACE_HINGE": 21,
    "CONTACT_CENTER_ANGLE": 22,
    "IR": 23, "IR3": 24, "ER": 25,
}


@lru_cache(maxsize=64)
def _gicforge_reference_primitive_values(
    primitives: tuple[GICPrimitive, ...],
    reference_coordinates_angstrom: tuple[tuple[float, float, float], ...],
) -> tuple[float, ...]:
    """Evaluate an immutable GICForge primitive reference only once."""

    from matrix_smith.coordinate_kernel.primitives import eval_primitives

    basis = tuple(
        _coordinate_kernel_primitive_from_gic_primitive(
            item,
            reference_coordinates_angstrom=reference_coordinates_angstrom,
        )
        for item in primitives
    )
    values = eval_primitives(basis, np.asarray(reference_coordinates_angstrom, dtype=float))
    return tuple(float(value) for value in values)


@lru_cache(maxsize=64)
def _gicforge_value_plan(
    definition_primitives: tuple[GICPrimitive, ...],
    gics: tuple,
    reference_coordinates_angstrom: tuple[tuple[float, float, float], ...],
):
    """Compile the immutable primitive basis and sparse GIC combination."""

    from scipy.sparse import csr_matrix
    from matrix_chem import compile_local_primitive_plan

    primitive_by_id = {
        primitive.identifier: primitive for primitive in definition_primitives
    }
    primitive_ids = tuple(
        dict.fromkeys(
            primitive_id
            for gic in gics
            for primitive_id, _coefficient in (
                gic.coefficients or ((gic.primitive_id, 1.0),)
            )
        )
    )
    primitives = tuple(primitive_by_id[primitive_id] for primitive_id in primitive_ids)
    primitive_index = {
        primitive.identifier: index for index, primitive in enumerate(primitives)
    }
    basis = tuple(
        _coordinate_kernel_primitive_from_gic_primitive(
            item,
            reference_coordinates_angstrom=reference_coordinates_angstrom,
        )
        for item in primitives
    )
    native_plan = compile_local_primitive_plan(basis)
    reference_values = np.asarray(
        _gicforge_reference_primitive_values(primitives, reference_coordinates_angstrom),
        dtype=float,
    )
    rows: list[int] = []
    columns: list[int] = []
    coefficients: list[float] = []
    for row, gic in enumerate(gics):
        for primitive_id, coefficient in (
            gic.coefficients or ((gic.primitive_id, 1.0),)
        ):
            rows.append(row)
            columns.append(primitive_index[primitive_id])
            coefficients.append(float(coefficient))
    combination = csr_matrix(
        (
            np.asarray(coefficients, dtype=float),
            (
                np.asarray(rows, dtype=np.int64),
                np.asarray(columns, dtype=np.int64),
            ),
        ),
        shape=(len(gics), len(primitives)),
    )
    principal_periodic = np.asarray(
        [
            index
            for index, primitive in enumerate(primitives)
            if primitive.function in {"D", "IMPD", "FROT", "RPCK", "FACE_HINGE"}
            and not (
                primitive.function == "D"
                and primitive.chart == "PERIODIC_CONTINUATION"
            )
        ],
        dtype=np.int64,
    )
    continuation_periodic = np.asarray(
        [
            index
            for index, primitive in enumerate(primitives)
            if primitive.function == "D"
            and primitive.chart == "PERIODIC_CONTINUATION"
        ],
        dtype=np.int64,
    )
    return (
        basis,
        native_plan,
        reference_values,
        combination,
        principal_periodic,
        continuation_periodic,
    )


def _evaluate_gicforge_gic_values(
    definition: GICDefinition,
    gics: tuple,
    coords: np.ndarray,
) -> np.ndarray:
    """Evaluate shared GICForge primitives once for a SONIC subset."""

    from matrix_smith.coordinate_kernel.primitives import eval_primitives

    (
        basis,
        native_plan,
        reference_values,
        combination,
        principal_periodic,
        continuation_periodic,
    ) = _gicforge_value_plan(
        definition.primitives,
        gics,
        definition.reference_coordinates_angstrom,
    )
    if native_plan is None:
        values = np.asarray(eval_primitives(basis, coords), dtype=float)
    else:
        from matrix_chem import evaluate_compiled_local_primitive_plan

        values = evaluate_compiled_local_primitive_plan(native_plan, coords)[0]
    if principal_periodic.size:
        delta = (
            values[principal_periodic]
            - reference_values[principal_periodic]
            + np.pi
        ) % (2.0 * np.pi) - np.pi
        values[principal_periodic] = reference_values[principal_periodic] + delta
    if continuation_periodic.size:
        values[continuation_periodic] = (
            values[continuation_periodic]
            - reference_values[continuation_periodic]
            + np.pi
        ) % (2.0 * np.pi) - np.pi
    return np.asarray(combination @ values, dtype=float).reshape(-1)


def _cartesian_column_labels(natoms: int) -> tuple[str, ...]:
    axes = ("X", "Y", "Z")
    return tuple(f"{atom}:{axis}" for atom in range(1, natoms + 1) for axis in axes)


def build_gic_b_matrix(
    definition: GICDefinition,
    *,
    coordinates_angstrom: tuple[tuple[float, float, float], ...] | np.ndarray | None = None,
    rotation_reference_coordinates: np.ndarray | None = None,
    parallel_workers: int = 1,
) -> GICBMatrix:
    """Evaluate the Wilson B matrix for a frozen GIC definition.

    Rows are independent and may be evaluated concurrently.  Thread workers
    share the immutable Cartesian arrays and therefore avoid copying the
    molecular geometry for every coordinate.
    """
    coords = (
        np.asarray(definition.reference_coordinates_angstrom, dtype=float)
        if coordinates_angstrom is None
        else np.asarray(coordinates_angstrom, dtype=float)
    )
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise GICForgeContractError("B-matrix coordinates must have shape (natoms, 3)")
    if definition.backend == "matrix-smith-gicforge.v1":
        return _build_smith_python_b_matrix(definition, coords)
    reference_coords = (
        np.asarray(definition.reference_coordinates_angstrom, dtype=float)
        if rotation_reference_coordinates is None
        else np.asarray(rotation_reference_coordinates, dtype=float)
    )
    workers = int(parallel_workers)
    if workers <= 0:
        raise ValueError("parallel_workers must be positive")

    evaluated_rows = _evaluate_native_sparse_gic_rows(
        definition,
        coords,
        reference_coords=reference_coords,
        coordinate_indices=tuple(range(len(definition.gics))),
        parallel_workers=workers,
    )

    rows: list[tuple[float, ...]] = []
    sparse_rows: list[SparseBRow] = []
    for gic, sparse_row in zip(definition.gics, evaluated_rows, strict=True):
        row = sparse_row.to_dense()
        if not np.all(np.isfinite(row)):
            raise GICForgeContractError(f"non-finite B-matrix row for frozen GIC {gic.identifier}")
        sparse_rows.append(sparse_row)
        rows.append(tuple(float(value) for value in row))
    return GICBMatrix(
        backend=B_MATRIX_BACKEND,
        coordinate_labels=tuple(gic.identifier for gic in definition.gics),
        coordinate_names=tuple(gic.name for gic in definition.gics),
        irreps=tuple(gic.irrep for gic in definition.gics),
        cartesian_columns=_cartesian_column_labels(coords.shape[0]),
        rows=tuple(rows),
        sparse_rows=tuple(sparse_rows),
    )


def build_primitive_b_matrix(
    definition: GICDefinition,
    *,
    coordinates_angstrom: tuple[tuple[float, float, float], ...] | np.ndarray | None = None,
    rotation_reference_coordinates: np.ndarray | None = None,
    parallel_workers: int = 1,
) -> GICBMatrix:
    """Evaluate the primitive Wilson rows underlying a frozen GIC definition.

    This is the authoritative primitive derivative path used by consumers
    that must transform a primitive quadratic model through the *physical*
    Cartesian tangent.  It deliberately reuses the same analytic kernels and
    fragment-rotation gauge as :func:`build_gic_b_matrix`.
    """

    coords = (
        np.asarray(definition.reference_coordinates_angstrom, dtype=float)
        if coordinates_angstrom is None
        else np.asarray(coordinates_angstrom, dtype=float)
    )
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise GICForgeContractError("primitive B-matrix coordinates must have shape (natoms, 3)")
    workers = int(parallel_workers)
    if workers <= 0:
        raise ValueError("parallel_workers must be positive")
    if definition.backend == "matrix-smith-gicforge.v1":
        from matrix_smith.coordinate_kernel.pipeline import b_matrix_analytic

        primitive_basis = tuple(
            _coordinate_kernel_primitive_from_gic_primitive(primitive)
            for primitive in definition.primitives
        )
        dense_rows = np.asarray(b_matrix_analytic(primitive_basis, coords), dtype=float)
        sparse_rows = tuple(SparseBRow.from_dense(row) for row in dense_rows)
    else:
        reference_coords = (
            np.asarray(definition.reference_coordinates_angstrom, dtype=float)
            if rotation_reference_coordinates is None
            else np.asarray(rotation_reference_coordinates, dtype=float)
        )
        sparse_rows = _evaluate_native_sparse_primitive_rows(
            definition,
            coords,
            reference_coords=reference_coords,
            primitive_ids=tuple(primitive.identifier for primitive in definition.primitives),
            parallel_workers=workers,
        )
        dense_rows = np.asarray([row.to_dense() for row in sparse_rows], dtype=float)
    if not np.all(np.isfinite(dense_rows)):
        raise GICForgeContractError("primitive B matrix contains non-finite values")
    return GICBMatrix(
        backend=B_MATRIX_BACKEND,
        coordinate_labels=tuple(primitive.identifier for primitive in definition.primitives),
        coordinate_names=tuple(primitive.identifier for primitive in definition.primitives),
        irreps=tuple("" for _primitive in definition.primitives),
        cartesian_columns=_cartesian_column_labels(coords.shape[0]),
        rows=tuple(tuple(float(value) for value in row) for row in dense_rows),
        sparse_rows=sparse_rows,
    )


def build_gic_and_primitive_b_matrices(
    definition: GICDefinition,
    *,
    coordinates_angstrom: tuple[tuple[float, float, float], ...] | np.ndarray | None = None,
    rotation_reference_coordinates: np.ndarray | None = None,
    parallel_workers: int = 1,
) -> tuple[GICBMatrix, GICBMatrix]:
    """Evaluate frozen GIC and primitive rows through one primitive pass.

    Conditioning needs both representations at the same Cartesian sample.
    Evaluating them independently duplicates every primitive derivative.  This
    paired API preserves the public matrix contracts while sharing the sole
    scientifically meaningful primitive evaluation.
    """

    coords = (
        np.asarray(definition.reference_coordinates_angstrom, dtype=float)
        if coordinates_angstrom is None
        else np.asarray(coordinates_angstrom, dtype=float)
    )
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise GICForgeContractError("B-matrix coordinates must have shape (natoms, 3)")
    workers = int(parallel_workers)
    if workers <= 0:
        raise ValueError("parallel_workers must be positive")
    if definition.backend == "matrix-smith-gicforge.v1":
        return (
            build_gic_b_matrix(
                definition,
                coordinates_angstrom=coords,
                rotation_reference_coordinates=rotation_reference_coordinates,
                parallel_workers=workers,
            ),
            build_primitive_b_matrix(
                definition,
                coordinates_angstrom=coords,
                rotation_reference_coordinates=rotation_reference_coordinates,
                parallel_workers=workers,
            ),
        )

    reference_coords = (
        np.asarray(definition.reference_coordinates_angstrom, dtype=float)
        if rotation_reference_coordinates is None
        else np.asarray(rotation_reference_coordinates, dtype=float)
    )
    primitive_ids = tuple(primitive.identifier for primitive in definition.primitives)
    primitive_rows = _evaluate_native_sparse_primitive_rows(
        definition,
        coords,
        reference_coords=reference_coords,
        primitive_ids=primitive_ids,
        parallel_workers=workers,
    )
    primitive_row_by_id = dict(zip(primitive_ids, primitive_rows, strict=True))
    gic_rows = tuple(
        SparseBRow.combine(
            *tuple(
                (float(coefficient), primitive_row_by_id[primitive_id])
                for primitive_id, coefficient in (gic.coefficients or ((gic.primitive_id, 1.0),))
            )
        )
        for gic in definition.gics
    )
    if any(not np.all(np.isfinite(row.to_dense())) for row in (*gic_rows, *primitive_rows)):
        raise GICForgeContractError("B matrix contains non-finite values")
    columns = _cartesian_column_labels(coords.shape[0])
    gic_matrix = GICBMatrix(
        backend=B_MATRIX_BACKEND,
        coordinate_labels=tuple(gic.identifier for gic in definition.gics),
        coordinate_names=tuple(gic.name for gic in definition.gics),
        irreps=tuple(gic.irrep for gic in definition.gics),
        cartesian_columns=columns,
        rows=tuple(tuple(float(value) for value in row.to_dense()) for row in gic_rows),
        sparse_rows=gic_rows,
    )
    primitive_matrix = GICBMatrix(
        backend=B_MATRIX_BACKEND,
        coordinate_labels=primitive_ids,
        coordinate_names=primitive_ids,
        irreps=tuple("" for _primitive in definition.primitives),
        cartesian_columns=columns,
        rows=tuple(tuple(float(value) for value in row.to_dense()) for row in primitive_rows),
        sparse_rows=primitive_rows,
    )
    return gic_matrix, primitive_matrix


def build_sparse_gic_b_matrix(
    definition: GICDefinition,
    *,
    coordinates_angstrom: tuple[tuple[float, float, float], ...] | np.ndarray | None = None,
    rotation_reference_coordinates: np.ndarray | None = None,
    coordinate_indices: tuple[int, ...] | None = None,
    parallel_workers: int = 1,
) -> SparseBMatrix:
    """Evaluate selected Wilson rows without materializing a dense matrix.

    The legacy :func:`build_gic_b_matrix` keeps dense rows for serialization
    compatibility.  Iterative LINK consumers should use this API when they do
    not require that representation.
    """

    coords = (
        np.asarray(definition.reference_coordinates_angstrom, dtype=float)
        if coordinates_angstrom is None
        else np.asarray(coordinates_angstrom, dtype=float)
    )
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise GICForgeContractError("B-matrix coordinates must have shape (natoms, 3)")
    indices = (
        tuple(range(len(definition.gics)))
        if coordinate_indices is None
        else tuple(int(index) for index in coordinate_indices)
    )
    if any(index < 0 or index >= len(definition.gics) for index in indices):
        raise IndexError("sparse B-matrix subset contains an invalid coordinate index")
    if definition.backend == "matrix-smith-gicforge.v1":
        from matrix_chem import (
            assemble_compiled_gic_csr,
            evaluate_compiled_local_primitive_plan,
        )

        selected_gics = tuple(definition.gics[index] for index in indices)
        (
            basis,
            native_plan,
            _reference_values,
            combination,
            _principal_periodic,
            _continuation_periodic,
        ) = _gicforge_value_plan(
            definition.primitives,
            selected_gics,
            definition.reference_coordinates_angstrom,
        )
        if native_plan is None:
            from matrix_chem.primitive_coordinates import (
                compiled_local_primitive_values_and_sparse_gradients,
                sparse_grad_primitive_entries,
            )

            primitive_ids = tuple(
                dict.fromkeys(
                    primitive_id
                    for gic in selected_gics
                    for primitive_id, _coefficient in (
                        gic.coefficients or ((gic.primitive_id, 1.0),)
                    )
                )
            )
            compiled = compiled_local_primitive_values_and_sparse_gradients(basis, coords)
            primitive_entries = (
                {
                    primitive_id: sparse_grad_primitive_entries(primitive, coords)
                    for primitive_id, primitive in zip(primitive_ids, basis, strict=True)
                }
                if compiled is None
                else dict(zip(primitive_ids, compiled[1], strict=True))
            )
            assembled_rows = []
            for gic in selected_gics:
                accumulator: dict[int, float] = {}
                for primitive_id, coefficient in (
                    gic.coefficients or ((gic.primitive_id, 1.0),)
                ):
                    factor = float(coefficient)
                    for column, value in primitive_entries[primitive_id]:
                        accumulator[column] = accumulator.get(column, 0.0) + factor * value
                assembled_rows.append(
                    SparseBRow(size=coords.size, entries=tuple(accumulator.items()))
                )
        else:
            _values, data, columns, indptr = evaluate_compiled_local_primitive_plan(
                native_plan, coords
            )
            gic_data, gic_indices, gic_indptr = assemble_compiled_gic_csr(
                data,
                columns,
                indptr,
                combination,
                column_count=coords.size,
                zero_tolerance=SPARSE_ZERO_TOLERANCE,
            )
            return SparseBMatrix.from_csr(
                gic_data,
                gic_indices,
                gic_indptr,
                column_count=coords.size,
                row_labels=tuple(definition.gics[index].identifier for index in indices),
                backend=B_MATRIX_BACKEND,
            )
        rows = tuple(assembled_rows)
    else:
        reference_coords = (
            np.asarray(definition.reference_coordinates_angstrom, dtype=float)
            if rotation_reference_coordinates is None
            else np.asarray(rotation_reference_coordinates, dtype=float)
        )
        rows = _evaluate_native_sparse_gic_rows(
            definition,
            coords,
            reference_coords=reference_coords,
            coordinate_indices=indices,
            parallel_workers=int(parallel_workers),
        )
    return SparseBMatrix(
        rows=rows,
        column_count=coords.size,
        row_labels=tuple(definition.gics[index].identifier for index in indices),
        backend=B_MATRIX_BACKEND,
    )


def update_sparse_gic_b_matrix(
    definition: GICDefinition,
    previous: SparseBMatrix,
    *,
    coordinates_angstrom: np.ndarray,
    affected_atoms: tuple[int, ...],
    parallel_workers: int = 1,
) -> tuple[SparseBMatrix, tuple[int, ...]]:
    """Re-evaluate only GIC rows whose Cartesian support meets a changed domain."""

    if previous.row_count != len(definition.gics):
        raise ValueError("incremental Wilson update requires the unchanged GIC row frame")
    indices = previous.rows_touching_atoms(affected_atoms)
    if not indices:
        return previous, ()
    refreshed = build_sparse_gic_b_matrix(
        definition,
        coordinates_angstrom=coordinates_angstrom,
        coordinate_indices=indices,
        parallel_workers=parallel_workers,
    )
    replacements = dict(zip(indices, refreshed.rows, strict=True))
    return previous.replace_rows(replacements), indices


def evaluate_gic_values_and_sparse_b_matrix(
    definition: GICDefinition,
    *,
    coordinates_angstrom: tuple[tuple[float, float, float], ...] | np.ndarray | None = None,
    rotation_reference_coordinates: np.ndarray | None = None,
    coordinate_indices: tuple[int, ...] | None = None,
    parallel_workers: int = 1,
) -> tuple[np.ndarray, SparseBMatrix]:
    """Evaluate GIC values and sparse Wilson rows from one primitive pass.

    GICForge's compiled primitive kernel already returns values and CSR
    derivatives together.  Keeping those products coupled avoids evaluating
    the complete primitive atlas twice at every SONIC geometry.  Unsupported
    backends retain the established independently audited paths.
    """

    coords = (
        np.asarray(definition.reference_coordinates_angstrom, dtype=float)
        if coordinates_angstrom is None
        else np.asarray(coordinates_angstrom, dtype=float)
    )
    gics = getattr(definition, "gics", None)
    if gics is None:
        values = evaluate_gic_values(
            definition,
            coordinates_angstrom=coords,
            rotation_reference_coordinates=rotation_reference_coordinates,
        )
        count = len(values)
        indices = (
            tuple(range(count))
            if coordinate_indices is None
            else tuple(int(index) for index in coordinate_indices)
        )
        if coordinate_indices is not None:
            values = np.asarray(values, dtype=float)[list(indices)]
        return np.asarray(values, dtype=float), build_sparse_gic_b_matrix(
            definition,
            coordinates_angstrom=coords,
            rotation_reference_coordinates=rotation_reference_coordinates,
            coordinate_indices=indices,
            parallel_workers=parallel_workers,
        )
    indices = (
        tuple(range(len(gics)))
        if coordinate_indices is None
        else tuple(int(index) for index in coordinate_indices)
    )
    selected_gics = tuple(gics[index] for index in indices)
    compiled_native = _compiled_native_gic_values_and_sparse_b_matrix(
        definition,
        selected_gics=selected_gics,
        coordinates_angstrom=coords,
        rotation_reference_coordinates=rotation_reference_coordinates,
    )
    if compiled_native is not None:
        return compiled_native
    if definition.backend != "matrix-smith-gicforge.v1":
        values = evaluate_gic_values_subset(
            definition,
            indices,
            coordinates_angstrom=coords,
            rotation_reference_coordinates=rotation_reference_coordinates,
        )
        return values, build_sparse_gic_b_matrix(
            definition,
            coordinates_angstrom=coords,
            rotation_reference_coordinates=rotation_reference_coordinates,
            coordinate_indices=indices,
            parallel_workers=parallel_workers,
        )

    from matrix_chem import (
        assemble_compiled_gic_csr,
        evaluate_compiled_local_primitive_plan,
    )

    (
        basis,
        native_plan,
        reference_values,
        combination,
        principal_periodic,
        continuation_periodic,
    ) = _gicforge_value_plan(
        definition.primitives,
        selected_gics,
        definition.reference_coordinates_angstrom,
    )
    if native_plan is None:
        values = evaluate_gic_values_subset(
            definition,
            indices,
            coordinates_angstrom=coords,
            rotation_reference_coordinates=rotation_reference_coordinates,
        )
        return values, build_sparse_gic_b_matrix(
            definition,
            coordinates_angstrom=coords,
            rotation_reference_coordinates=rotation_reference_coordinates,
            coordinate_indices=indices,
            parallel_workers=parallel_workers,
        )

    primitive_values, data, columns, indptr = evaluate_compiled_local_primitive_plan(
        native_plan, coords
    )
    phase_values = np.asarray(primitive_values, dtype=float).copy()
    if principal_periodic.size:
        delta = (
            phase_values[principal_periodic]
            - reference_values[principal_periodic]
            + np.pi
        ) % (2.0 * np.pi) - np.pi
        phase_values[principal_periodic] = reference_values[principal_periodic] + delta
    if continuation_periodic.size:
        phase_values[continuation_periodic] = (
            phase_values[continuation_periodic]
            - reference_values[continuation_periodic]
            + np.pi
        ) % (2.0 * np.pi) - np.pi
    values = np.asarray(combination @ phase_values, dtype=float).reshape(-1)
    gic_data, gic_indices, gic_indptr = assemble_compiled_gic_csr(
        data,
        columns,
        indptr,
        combination,
        column_count=coords.size,
        zero_tolerance=SPARSE_ZERO_TOLERANCE,
    )
    matrix = SparseBMatrix.from_csr(
        gic_data,
        gic_indices,
        gic_indptr,
        column_count=coords.size,
        row_labels=tuple(definition.gics[index].identifier for index in indices),
        backend=B_MATRIX_BACKEND,
    )
    return values, matrix


def _compiled_native_gic_values_and_sparse_b_matrix(
    definition: GICDefinition,
    *,
    selected_gics: tuple,
    coordinates_angstrom: np.ndarray,
    rotation_reference_coordinates: np.ndarray | None,
) -> tuple[np.ndarray, SparseBMatrix] | None:
    """Evaluate an arbitrary ORACLE/SMITH chart in one compiled CSR pass."""

    from matrix_chem import assemble_compiled_gic_csr

    try:
        (
            _basis,
            _local_plan,
            reference_values,
            combination,
            principal_periodic,
            continuation_periodic,
        ) = _gicforge_value_plan(
            definition.primitives,
            selected_gics,
            definition.reference_coordinates_angstrom,
        )
    except (GICForgeContractError, ValueError):
        return None
    primitive_by_id = {item.identifier: item for item in definition.primitives}
    primitive_ids = tuple(
        dict.fromkeys(
            primitive_id
            for gic in selected_gics
            for primitive_id, _coefficient in (
                gic.coefficients or ((gic.primitive_id, 1.0),)
            )
        )
    )
    primitives = tuple(primitive_by_id[item] for item in primitive_ids)
    reference = (
        np.asarray(definition.reference_coordinates_angstrom, dtype=float)
        if rotation_reference_coordinates is None
        else np.asarray(rotation_reference_coordinates, dtype=float)
    )
    compiled = compiled_gic_primitive_csr(
        primitives,
        coordinates_angstrom,
        reference_coords=reference,
    )
    if compiled is None:
        return None
    primitive_values, data, columns, indptr = compiled
    phase_values = np.asarray(primitive_values, dtype=float).copy()
    if principal_periodic.size:
        delta = (
            phase_values[principal_periodic] - reference_values[principal_periodic] + np.pi
        ) % (2.0 * np.pi) - np.pi
        phase_values[principal_periodic] = reference_values[principal_periodic] + delta
    if continuation_periodic.size:
        phase_values[continuation_periodic] = (
            phase_values[continuation_periodic]
            - reference_values[continuation_periodic]
            + np.pi
        ) % (2.0 * np.pi) - np.pi
    combination = combination.tocsr(copy=False)
    values = np.asarray(combination @ phase_values, dtype=float).reshape(-1)
    gic_data, gic_indices, gic_indptr = assemble_compiled_gic_csr(
        data,
        columns,
        indptr,
        combination,
        column_count=coordinates_angstrom.size,
        zero_tolerance=SPARSE_ZERO_TOLERANCE,
    )
    return values, SparseBMatrix.from_csr(
        gic_data,
        gic_indices,
        gic_indptr,
        column_count=coordinates_angstrom.size,
        row_labels=tuple(gic.identifier for gic in selected_gics),
        backend=B_MATRIX_BACKEND,
    )


def build_sparse_primitive_b_matrix(
    definition: GICDefinition,
    *,
    coordinates_angstrom: tuple[tuple[float, float, float], ...] | np.ndarray | None = None,
    rotation_reference_coordinates: np.ndarray | None = None,
    primitive_indices: tuple[int, ...] | None = None,
    parallel_workers: int = 1,
) -> SparseBMatrix:
    """Evaluate primitive Wilson rows without constructing dense compatibility rows."""

    coords = (
        np.asarray(definition.reference_coordinates_angstrom, dtype=float)
        if coordinates_angstrom is None
        else np.asarray(coordinates_angstrom, dtype=float)
    )
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise GICForgeContractError("primitive B-matrix coordinates must have shape (natoms, 3)")
    indices = (
        tuple(range(len(definition.primitives)))
        if primitive_indices is None
        else tuple(int(index) for index in primitive_indices)
    )
    if any(index < 0 or index >= len(definition.primitives) for index in indices):
        raise IndexError("sparse primitive B-matrix subset contains an invalid index")
    workers = int(parallel_workers)
    if workers <= 0:
        raise ValueError("parallel_workers must be positive")
    if definition.backend == "matrix-smith-gicforge.v1":
        from matrix_chem import compile_local_primitive_plan, evaluate_compiled_local_primitive_plan

        basis = tuple(
            _coordinate_kernel_primitive_from_gic_primitive(item)
            for item in definition.primitives
        )
        plan = compile_local_primitive_plan(basis)
        if plan is None:
            # Unsupported legacy primitives retain their independently audited
            # compatibility path. Native primitive plans never densify here.
            dense = build_primitive_b_matrix(
                definition,
                coordinates_angstrom=coords,
                rotation_reference_coordinates=rotation_reference_coordinates,
                parallel_workers=workers,
            )
            rows = tuple(SparseBRow.from_dense(dense.rows[index]) for index in indices)
        else:
            _values, data, columns, indptr = evaluate_compiled_local_primitive_plan(plan, coords)
            complete = SparseBMatrix.from_csr(
                data,
                columns,
                indptr,
                column_count=coords.size,
                row_labels=tuple(primitive.identifier for primitive in definition.primitives),
                backend=B_MATRIX_BACKEND,
            )
            rows = tuple(complete.rows[index] for index in indices)
    else:
        reference_coords = (
            np.asarray(definition.reference_coordinates_angstrom, dtype=float)
            if rotation_reference_coordinates is None
            else np.asarray(rotation_reference_coordinates, dtype=float)
        )
        primitive_ids = tuple(definition.primitives[index].identifier for index in indices)
        rows = _evaluate_native_sparse_primitive_rows(
            definition,
            coords,
            reference_coords=reference_coords,
            primitive_ids=primitive_ids,
            parallel_workers=workers,
        )
    return SparseBMatrix(
        rows=rows,
        column_count=coords.size,
        row_labels=tuple(definition.primitives[index].identifier for index in indices),
        backend=B_MATRIX_BACKEND,
    )


def build_sparse_gic_and_primitive_b_matrices(
    definition: GICDefinition,
    *,
    coordinates_angstrom: tuple[tuple[float, float, float], ...] | np.ndarray | None = None,
    rotation_reference_coordinates: np.ndarray | None = None,
    parallel_workers: int = 1,
) -> tuple[SparseBMatrix, SparseBMatrix]:
    """Evaluate GIC and primitive Wilson matrices without dense rows.

    The primitive tangent is evaluated exactly once. This is the normal API
    for matrix-free SH construction and iterative conditioning. The similarly
    named dense API remains for old serializers and explicit small-system
    reference audits only.
    """

    primitive_matrix = build_sparse_primitive_b_matrix(
        definition,
        coordinates_angstrom=coordinates_angstrom,
        rotation_reference_coordinates=rotation_reference_coordinates,
        parallel_workers=parallel_workers,
    )
    primitive_matrix.assert_finite()
    rows_by_id = dict(
        zip(
            tuple(primitive.identifier for primitive in definition.primitives),
            primitive_matrix.rows,
            strict=True,
        )
    )
    gic_matrix = SparseBMatrix(
        rows=tuple(
            SparseBRow.combine(
                *tuple(
                    (float(coefficient), rows_by_id[primitive_id])
                    for primitive_id, coefficient in (
                        gic.coefficients or ((gic.primitive_id, 1.0),)
                    )
                )
            )
            for gic in definition.gics
        ),
        column_count=primitive_matrix.column_count,
        row_labels=tuple(gic.identifier for gic in definition.gics),
        backend=B_MATRIX_BACKEND,
    )
    gic_matrix.assert_finite()
    return gic_matrix, primitive_matrix


def recombine_sparse_gic_rows(
    definition: GICDefinition,
    primitive_matrix: SparseBMatrix,
) -> SparseBMatrix:
    """Compile frozen GIC combinations from an already-evaluated CSR pool.

    No primitive is reevaluated and no dense Wilson row is formed.  The
    portable native kernel owns the CSR-by-CSR assembly, including duplicate
    column reduction.
    """

    from scipy.sparse import csr_matrix
    from matrix_chem import assemble_compiled_gic_csr

    primitive_index = {
        primitive.identifier: index
        for index, primitive in enumerate(definition.primitives)
    }
    data: list[float] = []
    indices: list[int] = []
    indptr = [0]
    for gic in definition.gics:
        for primitive_id, coefficient in gic.coefficients or ((gic.primitive_id, 1.0),):
            indices.append(primitive_index[primitive_id])
            data.append(float(coefficient))
        indptr.append(len(data))
    combination = csr_matrix(
        (np.asarray(data, dtype=float), np.asarray(indices, dtype=np.intp), np.asarray(indptr, dtype=np.intp)),
        shape=(len(definition.gics), primitive_matrix.row_count),
    )
    primitive_data, primitive_indices, primitive_indptr = primitive_matrix.csr_arrays()
    gic_data, gic_indices, gic_indptr = assemble_compiled_gic_csr(
        primitive_data,
        primitive_indices,
        primitive_indptr,
        combination,
        column_count=primitive_matrix.column_count,
        zero_tolerance=SPARSE_ZERO_TOLERANCE,
    )
    return SparseBMatrix.from_csr(
        gic_data, gic_indices, gic_indptr,
        column_count=primitive_matrix.column_count,
        row_labels=tuple(gic.identifier for gic in definition.gics),
        backend="smith-native-gic-recombination-csr.v1",
    )


def _evaluate_native_sparse_gic_rows(
    definition: GICDefinition,
    coords: np.ndarray,
    *,
    reference_coords: np.ndarray,
    coordinate_indices: tuple[int, ...],
    parallel_workers: int,
) -> tuple[SparseBRow, ...]:
    """Evaluate each required primitive row once, then form frozen GICs."""

    workers = int(parallel_workers)
    if workers <= 0:
        raise ValueError("parallel_workers must be positive")
    primitive_by_id = {primitive.identifier: primitive for primitive in definition.primitives}
    selected = tuple(definition.gics[index] for index in coordinate_indices)
    primitive_ids = tuple(
        dict.fromkeys(
            primitive_id
            for gic in selected
            for primitive_id, _coefficient in (gic.coefficients or ((gic.primitive_id, 1.0),))
        )
    )
    for primitive_id in primitive_ids:
        if primitive_id not in primitive_by_id:
            raise GICForgeContractError(
                f"unknown primitive {primitive_id!r} in frozen GIC definition"
            )

    row_by_id = dict(
        zip(
            primitive_ids,
            _evaluate_native_sparse_primitive_rows(
                definition,
                coords,
                reference_coords=reference_coords,
                primitive_ids=primitive_ids,
                parallel_workers=workers,
            ),
            strict=True,
        )
    )
    return tuple(
        SparseBRow.combine(
            *tuple(
                (float(coefficient), row_by_id[primitive_id])
                for primitive_id, coefficient in (gic.coefficients or ((gic.primitive_id, 1.0),))
            )
        )
        for gic in selected
    )


def _evaluate_native_sparse_primitive_rows(
    definition: GICDefinition,
    coords: np.ndarray,
    *,
    reference_coords: np.ndarray,
    primitive_ids: tuple[str, ...],
    parallel_workers: int,
) -> tuple[SparseBRow, ...]:
    """Evaluate requested primitive rows once in the frozen rotation gauge."""

    primitive_by_id = {primitive.identifier: primitive for primitive in definition.primitives}
    for primitive_id in primitive_ids:
        if primitive_id not in primitive_by_id:
            raise GICForgeContractError(
                f"unknown primitive {primitive_id!r} in frozen GIC definition"
            )
    requested_primitives = tuple(
        primitive_by_id[primitive_id] for primitive_id in primitive_ids
    )
    # Every ORACLE primitive family now has an analytic native implementation.
    # Keeping collective fragment rows on the old Python path made one Wilson
    # refresh dominate a cheap TB/FF evaluation for solvent clusters.
    compiled_rows = _compiled_local_gic_primitive_rows(
        requested_primitives,
        coords,
        reference_coords=reference_coords,
    )

    def evaluate_primitive(primitive_id: str) -> SparseBRow:
        return _sparse_analytic_b_row(
            primitive_by_id[primitive_id],
            coords,
            reference_coords=reference_coords,
        )

    workers = int(parallel_workers)
    if compiled_rows is not None:
        return compiled_rows
    if workers == 1 or len(primitive_ids) < 2:
        primitive_rows = tuple(evaluate_primitive(primitive_id) for primitive_id in primitive_ids)
    else:
        with ThreadPoolExecutor(
            max_workers=resolve_local_workers(workers, len(primitive_ids)),
            thread_name_prefix="smith-b-primitive",
        ) as executor:
            primitive_rows = tuple(executor.map(evaluate_primitive, primitive_ids))
    return primitive_rows


def _compiled_local_gic_primitive_rows(
    primitives: tuple[GICPrimitive, ...],
    coordinates_angstrom: np.ndarray,
    *,
    reference_coords: np.ndarray | None = None,
) -> tuple[SparseBRow, ...] | None:
    """Evaluate the complete SMITH primitive atlas in one sparse C++ call."""

    compiled = compiled_local_gic_primitive_csr(
        primitives, coordinates_angstrom, reference_coords=reference_coords
    )
    if compiled is None:
        return None
    _values, data, indices, indptr = compiled
    size = int(np.asarray(coordinates_angstrom).size)
    return tuple(
        SparseBRow(
            size=size,
            entries=tuple(
                (int(indices[position]), float(data[position]))
                for position in range(int(indptr[row]), int(indptr[row + 1]))
            ),
        )
        for row in range(len(primitives))
    )


def compiled_gic_primitive_csr(
    primitives: tuple[GICPrimitive, ...],
    coordinates_angstrom: np.ndarray,
    *,
    reference_coords: np.ndarray | None = None,
    topology_plan: tuple[np.ndarray, ...] | None = None,
):
    """Return every frozen SMITH primitive and Wilson row as native CSR."""

    from matrix_chem import compiled_smith_primitive_plan_csr

    coords = np.ascontiguousarray(coordinates_angstrom, dtype=float)
    reference = np.ascontiguousarray(
        coords if reference_coords is None else reference_coords, dtype=float
    )
    if coords.ndim != 2 or coords.shape[1:] != (3,) or reference.shape != coords.shape:
        raise GICForgeContractError(
            "compiled SMITH primitive coordinates and reference must have shape (natoms, 3)"
        )
    if not np.all(np.isfinite(coords)) or not np.all(np.isfinite(reference)):
        raise GICForgeContractError("compiled SMITH primitive coordinates must be finite")
    if any(p.function in {"SOOP", "SYCART"} for p in primitives):
        from matrix_chem.soop_contract import frozen_soop_covector
        ordinary = tuple(p for p in primitives if p.function not in {"SOOP", "SYCART"})
        compiled = compiled_gic_primitive_csr(ordinary, coords, reference_coords=reference) if ordinary else None
        if ordinary and compiled is None:
            return None
        values, data, indices, pointers = [], [], [], [0]
        cursor = 0
        for primitive in primitives:
            if primitive.function in {"SOOP", "SYCART"}:
                row, offset = frozen_soop_covector(primitive, len(coords))
                columns = np.flatnonzero(row)
                values.append(float(row @ coords.ravel() - offset))
                data.extend(row[columns])
                indices.extend(columns)
            else:
                v, d, ix, ip = compiled
                start, stop = int(ip[cursor]), int(ip[cursor+1])
                values.append(v[cursor])
                data.extend(d[start:stop])
                indices.extend(ix[start:stop])
                cursor += 1
            pointers.append(len(data))
        return (np.asarray(values), np.asarray(data), np.asarray(indices,dtype=np.intp),
                np.asarray(pointers,dtype=np.intp))
    # Native capability is optional per primitive family.  An older resident
    # extension must request the established sparse Python evaluator rather
    # than failing chart construction when ORACLE has introduced a new local
    # family (the common case during rolling upgrades).
    if any(str(p.function).upper() not in _COMPILED_GIC_FUNCTION_CODES for p in primitives):
        return None
    # Only implicit fragment frames depend on the reference geometry. All
    # other topology arrays can be reused as the current geometry changes.
    reference_key = (
        tuple(map(tuple, reference))
        if any(p.function.upper() == "FROT" and
               (not p.frame_atoms or not p.ref_frame_atoms) for p in primitives)
        else ()
    )
    plan = topology_plan or _compiled_gic_topology_plan(tuple(primitives), reference_key)
    try:
        return compiled_smith_primitive_plan_csr(coords, reference, *plan)
    except (RuntimeError, ValueError) as exc:
        # Wheels built before a newly added primitive family can still be
        # present in a resident environment.  Treat an explicit native
        # capability miss as a request for the audited Python evaluator; do
        # not turn a fullerene/IR chart into a hard failure merely because the
        # extension and the Python atlas were upgraded at different times.
        message = str(exc).lower()
        if "unsupported" in message or "no implementation" in message:
            return None
        raise


@lru_cache(maxsize=8)
def _compiled_gic_topology_plan(primitives, reference_key):
    """Bounded, immutable native topology plans; never cache values or B rows.

    The complete primitive records key the cache, including ring coefficients
    and fragment gauges. Changed charts therefore cannot reuse a stale plan.
    """
    reference = np.asarray(reference_key, dtype=float)
    codes = _COMPILED_GIC_FUNCTION_CODES
    kinds: list[int] = []
    modes: list[int] = []
    atom_data: list[int] = []; atom_ptr = [0]
    ref_data: list[int] = []; ref_ptr = [0]
    frame_data: list[int] = []; frame_ptr = [0]
    rframe_data: list[int] = []; rframe_ptr = [0]
    term_kinds: list[int] = []; term_atoms: list[tuple[int, int, int, int]] = []
    term_coefficients: list[float] = []; term_ptr = [0]
    for primitive in primitives:
        function = str(primitive.function).upper()
        if function not in codes:
            raise GICForgeContractError(
                f"compiled SMITH atlas has no implementation for {function!r}"
            )
        atoms = tuple(int(x) - 1 for x in primitive.atoms)
        refs = tuple(int(x) - 1 for x in primitive.ref_atoms)
        frames = tuple(int(x) - 1 for x in primitive.frame_atoms)
        rframes = tuple(int(x) - 1 for x in primitive.ref_frame_atoms)
        if function == "IMPD" and len(atoms) == 4:
            center, first, second, third = atoms
            atoms = (first, center, third, second)
        if function == "FROT":
            if not rframes:
                rframes = tuple(
                    x - 1 for x in _fragment_frame_anchor_atoms(
                        primitive.ref_atoms, coords=reference
                    )
                )
            if not frames:
                frames = tuple(
                    x - 1 for x in _fragment_frame_anchor_atoms(
                        primitive.atoms, coords=reference
                    )
                )
        expected_atoms = {
            "R": 2, "IR": 2, "IR3": 2, "ER": 2,
            "A": 3, "L": 3, "D": 4, "IMPD": 4, "U": 4, "H": 4,
        }
        if function in expected_atoms and len(atoms) != expected_atoms[function]:
            raise GICForgeContractError(
                f"compiled SMITH {function} primitive has invalid atom arity"
            )
        if function == "L" and len(refs) not in {0, 1}:
            raise GICForgeContractError("compiled SMITH L primitive has invalid reference arity")
        if function in {"FC_DIST", "FTRANS", "FLIN_TRANS", "FAXIS", "FROT"} and (
            not atoms or not refs
        ):
            raise GICForgeContractError(f"compiled SMITH {function} needs two fragment bodies")
        if function in {"FCA_DIST", "CENTER_ATOM_DIST"} and (
            not atoms or len(refs) != 1
        ):
            raise GICForgeContractError(f"compiled SMITH {function} needs one reference atom")
        if function == "FTRANS" and (int(primitive.mode) not in {0, 1, 2} or len(rframes) not in {0, 2}):
            raise GICForgeContractError("compiled SMITH FTRANS chart is invalid")
        if function == "FLIN_TRANS" and (int(primitive.mode) not in {0, 1} or len(rframes) != 1):
            raise GICForgeContractError("compiled SMITH FLIN_TRANS chart is invalid")
        if function == "FAXIS" and (
            int(primitive.mode) not in {0, 1} or len(frames) != 1 or len(rframes) != 1
        ):
            raise GICForgeContractError("compiled SMITH FAXIS chart is invalid")
        if function == "FROT" and (
            int(primitive.mode) not in {0, 1, 2} or len(frames) not in {1, 2} or len(rframes) != 2
        ):
            raise GICForgeContractError("compiled SMITH FROT chart is invalid")
        native_code = codes[function]
        if function == "FROT" and len(frames) == 1 and int(primitive.mode) == 2:
            native_code = 17
        kinds.append(native_code); modes.append(int(primitive.mode))
        atom_data.extend(atoms); atom_ptr.append(len(atom_data))
        ref_data.extend(refs); ref_ptr.append(len(ref_data))
        frame_data.extend(frames); frame_ptr.append(len(frame_data))
        rframe_data.extend(rframes); rframe_ptr.append(len(rframe_data))
        if function in {"RPCB", "RPCK", "RPU"}:
            parser = _angle_component_terms_from_refs if function == "RPCB" else _ring_pucker_terms_from_refs
            term_code = 2 if function == "RPCB" else (3 if function == "RPCK" else 4)
            for coefficient, term in parser(primitive):
                zero_based_term = tuple(int(x) - 1 for x in term)
                padded = (*zero_based_term, zero_based_term[-1])[:4]
                term_kinds.append(term_code); term_atoms.append(padded)
                term_coefficients.append(float(coefficient))
        term_ptr.append(len(term_kinds))
    plan = (
        np.asarray(kinds, dtype=np.intp),
        np.asarray(modes, dtype=np.intp), np.asarray(atom_data, dtype=np.intp),
        np.asarray(atom_ptr, dtype=np.intp), np.asarray(ref_data, dtype=np.intp),
        np.asarray(ref_ptr, dtype=np.intp), np.asarray(frame_data, dtype=np.intp),
        np.asarray(frame_ptr, dtype=np.intp), np.asarray(rframe_data, dtype=np.intp),
        np.asarray(rframe_ptr, dtype=np.intp), np.asarray(term_kinds, dtype=np.intp),
        np.asarray(term_atoms, dtype=np.intp).reshape((-1, 4)),
        np.asarray(term_coefficients, dtype=float), np.asarray(term_ptr, dtype=np.intp),
    )
    for array in plan:
        array.setflags(write=False)
    return plan


def compiled_smith_plan_payload(definition: GICDefinition) -> dict[str, object]:
    """Serialize the compiled ORACLE topology plan and GIC combination CSR."""

    from scipy.sparse import csr_matrix
    from matrix_chem import compiled_chem_kernel_identity

    reference = tuple(map(tuple, definition.reference_coordinates_angstrom))
    plan = _compiled_gic_topology_plan(tuple(definition.primitives), reference)
    primitive_index = {item.identifier: index for index, item in enumerate(definition.primitives)}
    data = []; indices = []; indptr = [0]
    for gic in definition.gics:
        for primitive_id, coefficient in gic.coefficients or ((gic.primitive_id, 1.0),):
            indices.append(primitive_index[primitive_id]); data.append(float(coefficient))
        indptr.append(len(data))
    combination = csr_matrix((data, indices, indptr), shape=(len(definition.gics), len(definition.primitives)))
    return {
        "schema": "matrix.smith.compiled_oracle_plan.v1",
        "native_kernel": compiled_chem_kernel_identity(),
        "primitive_ids": [item.identifier for item in definition.primitives],
        "gic_ids": [item.identifier for item in definition.gics],
        "plan": [
            {"dtype": "float64" if array.dtype.kind == "f" else "intp",
             "shape": list(array.shape), "values": array.reshape(-1).tolist()}
            for array in plan
        ],
        "combination": {
            "shape": list(combination.shape), "data": combination.data.tolist(),
            "indices": combination.indices.tolist(), "indptr": combination.indptr.tolist(),
        },
    }


def compiled_smith_plan_from_payload(
    payload: dict[str, object], definition: GICDefinition | None = None
):
    """Restore an immutable native topology plan and GIC combination CSR."""

    from scipy.sparse import csr_matrix
    from matrix_chem import compiled_chem_kernel_identity

    if payload.get("schema") != "matrix.smith.compiled_oracle_plan.v1":
        raise ValueError("unsupported compiled ORACLE plan schema")
    if payload.get("native_kernel") != compiled_chem_kernel_identity():
        raise ValueError("compiled ORACLE plan native-kernel ABI does not match")
    if definition is not None and (
        tuple(payload.get("primitive_ids", ()))
        != tuple(item.identifier for item in definition.primitives)
        or tuple(payload.get("gic_ids", ()))
        != tuple(item.identifier for item in definition.gics)
    ):
        raise ValueError("compiled ORACLE plan does not match the frozen SMITH definition")
    plan = []
    for record in payload["plan"]:
        dtype = float if record["dtype"] == "float64" else np.intp
        array = np.asarray(record["values"], dtype=dtype).reshape(tuple(record["shape"]))
        array.setflags(write=False); plan.append(array)
    record = payload["combination"]
    combination = csr_matrix(
        (np.asarray(record["data"], dtype=float), np.asarray(record["indices"], dtype=np.intp),
         np.asarray(record["indptr"], dtype=np.intp)), shape=tuple(record["shape"]),
    )
    return tuple(plan), combination


# Compatibility name retained for callers created while only the bounded
# local subset was compiled.  It now has complete-atlas semantics.
compiled_local_gic_primitive_csr = compiled_gic_primitive_csr


def evaluate_gic_values(
    definition: GICDefinition,
    *,
    coordinates_angstrom: tuple[tuple[float, float, float], ...] | np.ndarray | None = None,
    rotation_reference_coordinates: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate the values of a frozen SONIC/GIC definition."""
    coords = (
        np.asarray(definition.reference_coordinates_angstrom, dtype=float)
        if coordinates_angstrom is None
        else np.asarray(coordinates_angstrom, dtype=float)
    )
    primitive_by_id = {primitive.identifier: primitive for primitive in definition.primitives}
    reference_coords = np.asarray(definition.reference_coordinates_angstrom, dtype=float)
    rotation_reference = (
        reference_coords
        if rotation_reference_coordinates is None
        else np.asarray(rotation_reference_coordinates, dtype=float)
    )
    if definition.backend == "matrix-smith-gicforge.v1":
        return _evaluate_gicforge_gic_values(definition, definition.gics, coords)
    else:
        values_by_id = _evaluate_native_primitive_values(
            tuple(primitive_by_id.values()),
            coords,
            reference_coords=reference_coords,
            rotation_reference=rotation_reference,
        )
    values: list[float] = []
    for gic in definition.gics:
        coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
        values.append(
            sum(
                float(coefficient) * values_by_id[primitive_id]
                for primitive_id, coefficient in coefficients
            )
        )
    return np.asarray(values, dtype=float)


def evaluate_primitive_values(
    definition: GICDefinition,
    *,
    primitive_ids: tuple[str, ...] | None = None,
    coordinates_angstrom: tuple[tuple[float, float, float], ...] | np.ndarray | None = None,
    rotation_reference_coordinates: np.ndarray | None = None,
) -> dict[str, float]:
    """Evaluate selected frozen primitive values in definition order.

    The returned values use SMITH's frozen-reference gauges.  Stateful
    consumers such as LINK may then transport periodic primitive phases to
    their latest accepted geometry before forming collective GICs.
    """

    coords = (
        np.asarray(definition.reference_coordinates_angstrom, dtype=float)
        if coordinates_angstrom is None
        else np.asarray(coordinates_angstrom, dtype=float)
    )
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise GICForgeContractError("primitive-value coordinates must have shape (natoms, 3)")
    reference_coords = np.asarray(definition.reference_coordinates_angstrom, dtype=float)
    rotation_reference = (
        reference_coords
        if rotation_reference_coordinates is None
        else np.asarray(rotation_reference_coordinates, dtype=float)
    )
    primitive_by_id = {primitive.identifier: primitive for primitive in definition.primitives}
    selected_ids = (
        tuple(primitive_by_id)
        if primitive_ids is None
        else tuple(str(identifier) for identifier in primitive_ids)
    )
    missing = tuple(identifier for identifier in selected_ids if identifier not in primitive_by_id)
    if missing:
        raise KeyError(f"unknown frozen primitive identifiers: {','.join(missing)}")
    selected = tuple(primitive_by_id[identifier] for identifier in selected_ids)
    if definition.backend == "matrix-smith-gicforge.v1":
        from matrix_smith.coordinate_kernel.primitives import eval_primitives

        primitive_basis = tuple(
            _coordinate_kernel_primitive_from_gic_primitive(primitive) for primitive in selected
        )
        raw_values = eval_primitives(primitive_basis, coords)
        reference_values = eval_primitives(primitive_basis, reference_coords)
        return {
            primitive.identifier: _continuous_primitive_value(
                primitive,
                float(raw_values[index]),
                float(reference_values[index]),
            )
            for index, primitive in enumerate(selected)
        }
    return _evaluate_native_primitive_values(
        selected,
        coords,
        reference_coords=reference_coords,
        rotation_reference=rotation_reference,
    )


def evaluate_gic_value(
    definition: GICDefinition,
    coordinate_index: int,
    *,
    coordinates_angstrom: tuple[tuple[float, float, float], ...] | np.ndarray | None = None,
    rotation_reference_coordinates: np.ndarray | None = None,
) -> float:
    """Evaluate one frozen SONIC without evaluating unrelated primitive rows."""
    index = int(coordinate_index)
    if index < 0 or index >= len(definition.gics):
        raise IndexError(f"SONIC coordinate index {coordinate_index} is outside the definition")
    coords = (
        np.asarray(definition.reference_coordinates_angstrom, dtype=float)
        if coordinates_angstrom is None
        else np.asarray(coordinates_angstrom, dtype=float)
    )
    reference_coords = np.asarray(definition.reference_coordinates_angstrom, dtype=float)
    rotation_reference = (
        reference_coords
        if rotation_reference_coordinates is None
        else np.asarray(rotation_reference_coordinates, dtype=float)
    )
    gic = definition.gics[index]
    components = gic.coefficients or ((gic.primitive_id, 1.0),)
    primitive_by_id = {primitive.identifier: primitive for primitive in definition.primitives}
    primitives = tuple(primitive_by_id[primitive_id] for primitive_id, _coefficient in components)
    if definition.backend == "matrix-smith-gicforge.v1":
        from matrix_smith.coordinate_kernel.primitives import eval_primitives

        primitive_basis = tuple(
            _coordinate_kernel_primitive_from_gic_primitive(primitive) for primitive in primitives
        )
        raw_values = eval_primitives(primitive_basis, coords)
        reference_values = eval_primitives(primitive_basis, reference_coords)
        values = tuple(
            _continuous_primitive_value(
                primitive,
                float(raw_values[primitive_index]),
                float(reference_values[primitive_index]),
            )
            for primitive_index, primitive in enumerate(primitives)
        )
    else:
        values = tuple(
            _continuous_primitive_value(
                primitive,
                float(
                    _primitive_value(
                        primitive,
                        coords,
                        reference_coords=(
                            rotation_reference if primitive.function == "FROT" else reference_coords
                        ),
                    )
                ),
                float(
                    _primitive_value(
                        primitive,
                        rotation_reference if primitive.function == "FROT" else reference_coords,
                        reference_coords=(
                            rotation_reference if primitive.function == "FROT" else reference_coords
                        ),
                    )
                ),
            )
            for primitive in primitives
        )
    return float(
        sum(
            float(coefficient) * value
            for (_primitive_id, coefficient), value in zip(components, values, strict=True)
        )
    )


def evaluate_gic_subset(
    definition: GICDefinition,
    coordinate_indices: tuple[int, ...],
    *,
    coordinates_angstrom: tuple[tuple[float, float, float], ...] | np.ndarray | None = None,
    rotation_reference_coordinates: np.ndarray | None = None,
    parallel_workers: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate selected SONIC values and B rows without building the full B."""
    workers = int(parallel_workers)
    if workers <= 0:
        raise ValueError("parallel_workers must be positive")
    indices = tuple(int(index) for index in coordinate_indices)
    if any(index < 0 or index >= len(definition.gics) for index in indices):
        raise IndexError("SONIC coordinate subset contains an index outside the definition")
    coords = (
        np.asarray(definition.reference_coordinates_angstrom, dtype=float)
        if coordinates_angstrom is None
        else np.asarray(coordinates_angstrom, dtype=float)
    )
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise GICForgeContractError("B-matrix coordinates must have shape (natoms, 3)")
    reference_coords = np.asarray(definition.reference_coordinates_angstrom, dtype=float)
    rotation_reference = (
        reference_coords
        if rotation_reference_coordinates is None
        else np.asarray(rotation_reference_coordinates, dtype=float)
    )
    primitive_by_id = {primitive.identifier: primitive for primitive in definition.primitives}
    selected_gics = tuple(definition.gics[index] for index in indices)
    if definition.backend == "matrix-smith-gicforge.v1":
        values = _evaluate_gicforge_gic_values(definition, selected_gics, coords)
    else:
        primitive_ids_for_values = tuple(
            dict.fromkeys(
                primitive_id
                for gic in selected_gics
                for primitive_id, _coefficient in (gic.coefficients or ((gic.primitive_id, 1.0),))
            )
        )
        values_by_id = _evaluate_native_primitive_values(
            tuple(primitive_by_id[primitive_id] for primitive_id in primitive_ids_for_values),
            coords,
            reference_coords=reference_coords,
            rotation_reference=rotation_reference,
        )
        values = np.asarray(
            [
                sum(
                    float(coefficient) * values_by_id[primitive_id]
                    for primitive_id, coefficient in (
                        gic.coefficients or ((gic.primitive_id, 1.0),)
                    )
                )
                for gic in selected_gics
            ],
            dtype=float,
        )
    if definition.backend == "matrix-smith-gicforge.v1":
        from matrix_smith.coordinate_kernel.pipeline import b_matrix_analytic

        primitive_ids = tuple(
            dict.fromkeys(
                primitive_id
                for gic in selected_gics
                for primitive_id, _coefficient in (gic.coefficients or ((gic.primitive_id, 1.0),))
            )
        )
        primitive_basis = tuple(
            _coordinate_kernel_primitive_from_gic_primitive(primitive_by_id[primitive_id])
            for primitive_id in primitive_ids
        )
        primitive_rows = np.asarray(b_matrix_analytic(primitive_basis, coords), dtype=float)
        primitive_index = {primitive_id: index for index, primitive_id in enumerate(primitive_ids)}
        rows = np.asarray(
            [
                sum(
                    float(coefficient) * primitive_rows[primitive_index[primitive_id]]
                    for primitive_id, coefficient in (
                        gic.coefficients or ((gic.primitive_id, 1.0),)
                    )
                )
                for gic in selected_gics
            ],
            dtype=float,
        )
    else:
        sparse_rows = _evaluate_native_sparse_gic_rows(
            definition,
            coords,
            reference_coords=rotation_reference,
            coordinate_indices=indices,
            parallel_workers=workers,
        )
        rows = np.asarray([row.to_dense() for row in sparse_rows], dtype=float)
    return values, rows


def evaluate_gic_values_subset(
    definition: GICDefinition,
    coordinate_indices: tuple[int, ...],
    *,
    coordinates_angstrom: tuple[tuple[float, float, float], ...] | np.ndarray | None = None,
    rotation_reference_coordinates: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate selected frozen SONIC values without constructing Wilson rows."""

    indices = tuple(int(index) for index in coordinate_indices)
    if any(index < 0 or index >= len(definition.gics) for index in indices):
        raise IndexError("SONIC coordinate subset contains an index outside the definition")
    coords = (
        np.asarray(definition.reference_coordinates_angstrom, dtype=float)
        if coordinates_angstrom is None
        else np.asarray(coordinates_angstrom, dtype=float)
    )
    reference_coords = np.asarray(definition.reference_coordinates_angstrom, dtype=float)
    rotation_reference = (
        reference_coords
        if rotation_reference_coordinates is None
        else np.asarray(rotation_reference_coordinates, dtype=float)
    )
    if definition.backend == "matrix-smith-gicforge.v1":
        selected = tuple(definition.gics[index] for index in indices)
        return _evaluate_gicforge_gic_values(definition, selected, coords)
    selected = tuple(definition.gics[index] for index in indices)
    primitive_by_id = {primitive.identifier: primitive for primitive in definition.primitives}
    primitive_ids = tuple(
        dict.fromkeys(
            primitive_id
            for gic in selected
            for primitive_id, _coefficient in (gic.coefficients or ((gic.primitive_id, 1.0),))
        )
    )
    primitive_values = _evaluate_native_primitive_values(
        tuple(primitive_by_id[primitive_id] for primitive_id in primitive_ids),
        coords,
        reference_coords=reference_coords,
        rotation_reference=rotation_reference,
    )
    return np.asarray(
        [
            sum(
                float(coefficient) * primitive_values[primitive_id]
                for primitive_id, coefficient in (gic.coefficients or ((gic.primitive_id, 1.0),))
            )
            for gic in selected
        ],
        dtype=float,
    )


def _evaluate_native_primitive_values(
    primitives: tuple[GICPrimitive, ...],
    coords: np.ndarray,
    *,
    reference_coords: np.ndarray,
    rotation_reference: np.ndarray,
) -> dict[str, float]:
    """Evaluate primitive values once, sharing each three-component FROT frame."""

    values: dict[str, float] = {}
    rotation_groups: dict[
        tuple[
            tuple[int, ...],
            tuple[int, ...],
            tuple[int, ...],
            tuple[int, ...],
        ],
        list[GICPrimitive],
    ] = {}
    for primitive in primitives:
        if primitive.function == "FROT":
            key = (
                tuple(primitive.atoms),
                tuple(primitive.ref_atoms),
                tuple(primitive.frame_atoms),
                tuple(primitive.ref_frame_atoms),
            )
            rotation_groups.setdefault(key, []).append(primitive)
            continue
        raw = float(
            _primitive_value(
                primitive,
                coords,
                reference_coords=reference_coords,
            )
        )
        reference = float(
            _primitive_value(
                primitive,
                reference_coords,
                reference_coords=reference_coords,
            )
        )
        values[primitive.identifier] = _continuous_primitive_value(primitive, raw, reference)
    for (
        atoms,
        ref_atoms,
        frame_atoms,
        ref_frame_atoms,
    ), group in rotation_groups.items():
        if len(frame_atoms) == 1:
            current = _linear_fragment_stereographic_values(
                coords,
                atoms,
                frame_atoms[0],
                ref_atoms,
                ref_frame_atoms,
                selection_coords=rotation_reference,
            )
            reference = _linear_fragment_stereographic_values(
                rotation_reference,
                atoms,
                frame_atoms[0],
                ref_atoms,
                ref_frame_atoms,
                selection_coords=rotation_reference,
            )
            for primitive in group:
                values[primitive.identifier] = float(
                    current[primitive.mode] - reference[primitive.mode]
                )
            continue
        current_frag, current_ref = _fragment_relative_frames(
            coords,
            atoms,
            ref_atoms,
            frame_atoms=frame_atoms,
            ref_frame_atoms=ref_frame_atoms,
            gauge_reference_coords=rotation_reference,
        )
        reference_frag, reference_ref = _fragment_relative_frames(
            rotation_reference,
            atoms,
            ref_atoms,
            frame_atoms=frame_atoms,
            ref_frame_atoms=ref_frame_atoms,
            gauge_reference_coords=rotation_reference,
        )
        delta_rotation = (current_frag.T @ current_ref) @ (reference_frag.T @ reference_ref).T
        vector = _rotation_vector(delta_rotation)
        for primitive in group:
            values[primitive.identifier] = float(vector[primitive.mode])
    return values


def _continuous_primitive_value(
    primitive: GICPrimitive,
    value: float,
    reference: float,
) -> float:
    if primitive.function not in {"D", "IMPD", "FROT", "RPCK", "FACE_HINGE"}:
        return float(value)
    delta = (float(value) - float(reference) + np.pi) % (2.0 * np.pi) - np.pi
    if primitive.function == "D" and primitive.chart == "PERIODIC_CONTINUATION":
        return float(delta)
    return float(reference) + delta


def _build_smith_python_b_matrix(
    definition: GICDefinition,
    coords: np.ndarray,
) -> GICBMatrix:
    from matrix_smith.coordinate_kernel.pipeline import b_matrix_analytic

    primitive_by_id = {primitive.identifier: primitive for primitive in definition.primitives}
    primitive_index_by_id = {
        primitive.identifier: index for index, primitive in enumerate(definition.primitives)
    }
    primitive_basis = tuple(
        _coordinate_kernel_primitive_from_gic_primitive(primitive) for primitive in definition.primitives
    )
    primitive_b = b_matrix_analytic(primitive_basis, coords)
    primitive_sparse_b = tuple(SparseBRow.from_dense(row) for row in primitive_b)
    rows: list[tuple[float, ...]] = []
    sparse_rows: list[SparseBRow] = []
    for gic in definition.gics:
        weighted_rows: list[tuple[float, SparseBRow]] = []
        coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
        for primitive_id, coefficient in coefficients:
            primitive = primitive_by_id.get(primitive_id)
            if primitive is None:
                raise GICForgeContractError(
                    f"unknown primitive {primitive_id!r} in frozen GIC {gic.identifier}"
                )
            primitive_index = primitive_index_by_id[primitive.identifier]
            weighted_rows.append((float(coefficient), primitive_sparse_b[primitive_index]))
        sparse_row = SparseBRow.combine(*tuple(weighted_rows))
        row = sparse_row.to_dense()
        if not np.all(np.isfinite(row)):
            raise GICForgeContractError(f"non-finite B-matrix row for frozen GIC {gic.identifier}")
        sparse_rows.append(sparse_row)
        rows.append(tuple(float(value) for value in row))
    return GICBMatrix(
        backend="matrix-smith-analytic-bmatrix.v1",
        coordinate_labels=tuple(gic.identifier for gic in definition.gics),
        coordinate_names=tuple(gic.name for gic in definition.gics),
        irreps=tuple(gic.irrep for gic in definition.gics),
        cartesian_columns=_cartesian_column_labels(coords.shape[0]),
        rows=tuple(rows),
        sparse_rows=tuple(sparse_rows),
    )


def _coordinate_kernel_primitive_from_gic_primitive(
    primitive: GICPrimitive,
    *,
    reference_coordinates_angstrom=None,
):
    from matrix_smith.coordinate_kernel.primitives import Primitive
    from matrix_chem.face_primitives import FACE_OPERATORS, face_from_smith

    if primitive.function in FACE_OPERATORS.values():
        return face_from_smith(primitive)

    atoms = tuple(atom - 1 for atom in primitive.atoms)
    if primitive.function == "R":
        return Primitive("bond", atoms)
    if primitive.function == "IR":
        return Primitive("inverse_bond", atoms)
    if primitive.function == "IR3":
        return Primitive("inverse_cubic_bond", atoms)
    if primitive.function == "ER":
        return Primitive("exponential_bond", atoms)
    if primitive.function == "A":
        return Primitive("angle", atoms)
    if primitive.function == "L":
        return Primitive(
            "linear_bend",
            atoms,
            mode=primitive.mode,
            ref=tuple(atom - 1 for atom in primitive.ref_atoms),
        )
    if primitive.function == "D":
        return Primitive("dihedral", atoms)
    if primitive.function == "U":
        return Primitive("out_of_plane", atoms)
    if primitive.function == "H":
        return Primitive("out_of_plane_height", atoms)
    if primitive.function in {"SOOP", "SYCART"}:
        return Primitive(
            "selective_out_of_plane",
            atoms,
            refs=tuple(str(value) for value in primitive.refs),
            chart=str(primitive.chart),
        )
    fragment_kinds = {
        "FC_DIST": "frag_dist",
        "FCA_DIST": "frag_atom_dist",
        "CENTER_ATOM_DIST": "frag_atom_dist",
        "FTRANS": "frag_trans",
        "FLIN_TRANS": "frag_linear_trans",
        "FAXIS": "frag_axis",
        "FROT": "frag_rot",
    }
    if primitive.function in fragment_kinds:
        return Primitive(
            fragment_kinds[primitive.function],
            atoms,
            mode=primitive.mode,
            ref=tuple(atom - 1 for atom in primitive.ref_atoms),
            frame_atoms=tuple(atom - 1 for atom in primitive.frame_atoms),
            ref_frame_atoms=tuple(atom - 1 for atom in primitive.ref_frame_atoms),
            rotation_reference=(
                tuple(
                    tuple(float(value) for value in reference_coordinates_angstrom[index])
                    for index in sorted(set((*atoms, *(atom - 1 for atom in primitive.ref_atoms))))
                )
                if primitive.function == "FROT" and reference_coordinates_angstrom is not None
                else ()
            ),
        )
    raise GICForgeContractError(
        f"unsupported SMITH primitive function for B matrix: {primitive.function}"
    )
