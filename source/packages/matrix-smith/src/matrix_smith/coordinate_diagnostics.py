"""SONIC coordinate conditioning and singularity diagnostics."""

from __future__ import annotations

import numpy as np

from .bmatrix import SparseBMatrix

from matrix_numerics import singular_spectrum

SONIC_CONDITION_DIAGNOSTICS_SCHEMA = "matrix.smith.sonic_condition_diagnostics.v2"


def _rigid_rotation_axes(
    centered_coordinates: np.ndarray,
    gauge_count: int,
) -> tuple[np.ndarray, ...]:
    centered = np.asarray(centered_coordinates, dtype=float)
    inertia = (
        float(np.sum(centered * centered)) * np.eye(3, dtype=float)
        - centered.T @ centered
    )
    values, vectors = np.linalg.eigh(inertia)
    order = np.argsort(values)[::-1][: max(0, int(gauge_count) - 3)]
    return tuple(np.asarray(vectors[:, index], dtype=float) for index in order)


def _rigid_mode_vector(
    centered_coordinates: np.ndarray,
    mode: int,
    rotation_axes: tuple[np.ndarray, ...],
) -> np.ndarray:
    centered = np.asarray(centered_coordinates, dtype=float)
    values = np.zeros_like(centered)
    if int(mode) < 3:
        values[:, int(mode)] = 1.0
    else:
        values = np.cross(rotation_axes[int(mode) - 3], centered)
    return values.reshape(-1)


def _independent_rigid_gauge_columns(
    centered_coordinates: np.ndarray,
    rotation_axes: tuple[np.ndarray, ...],
) -> tuple[int, ...]:
    """Select a rigid gauge without materializing its ``6 x 3N`` rows."""

    centered = np.asarray(centered_coordinates, dtype=float)
    count = 3 + len(rotation_axes)

    def coefficients(position: np.ndarray, component: int) -> np.ndarray:
        result = np.zeros(count, dtype=float)
        result[component] = 1.0
        basis = np.eye(3, dtype=float)[component]
        for axis_index, axis in enumerate(rotation_axes):
            result[3 + axis_index] = float(np.cross(axis, position) @ basis)
        return result

    first = int(np.argmax(np.sum(centered * centered, axis=1)))
    second = int(np.argmax(np.sum((centered - centered[first]) ** 2, axis=1)))
    anchors = [first, second]
    if len(rotation_axes) == 3:
        line = centered[second] - centered[first]
        line_norm = float(np.linalg.norm(line))
        if line_norm > np.finfo(float).eps:
            transverse = np.cross(centered - centered[first], line / line_norm)
            anchors.append(int(np.argmax(np.sum(transverse * transverse, axis=1))))
    anchors.append(0)
    candidates = [
        (3 * atom + component, coefficients(centered[atom], component))
        for atom in dict.fromkeys(anchors)
        for component in range(3)
    ]
    selected: list[int] = []
    basis: list[np.ndarray] = []

    def select(rows: list[tuple[int, np.ndarray]]) -> None:
        while len(selected) < count:
            best_column = -1
            best_residual: np.ndarray | None = None
            best_norm = -1.0
            for column, candidate in rows:
                if column in selected:
                    continue
                residual = candidate.copy()
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
        select(
            [
                (3 * atom + component, coefficients(position, component))
                for atom, position in enumerate(centered)
                for component in range(3)
            ]
        )
    return tuple(selected)


def sparse_full_row_rank_diagnostics(
    jacobian: SparseBMatrix,
    coordinates_angstrom: np.ndarray,
    *,
    tolerance: float = 1.0e-8,
    absolute_tolerance: float = 1.0e-10,
) -> dict[str, object]:
    """Certify a large internal Jacobian after removing its rigid gauge."""

    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import splu

    if not isinstance(jacobian, SparseBMatrix) or jacobian.row_count == 0:
        raise ValueError("sparse SONIC Jacobian must be nonempty")
    coords = np.asarray(coordinates_angstrom, dtype=float)
    if coords.shape != (jacobian.column_count // 3, 3):
        raise ValueError("sparse SONIC coordinates do not match Jacobian columns")
    gauge_count = jacobian.column_count - jacobian.row_count
    if gauge_count < 0 or gauge_count > 6:
        raise ValueError("sparse SONIC full-row-rank audit requires a rigid-mode nullity")
    row_norms = np.asarray([row.norm() for row in jacobian.rows], dtype=float)
    if np.any(~np.isfinite(row_norms)) or np.any(row_norms <= absolute_tolerance):
        raise ValueError("sparse SONIC Jacobian contains a zero or non-finite row")
    data: list[float] = []
    indices: list[int] = []
    indptr = [0]
    for row_index, row in enumerate(jacobian.rows):
        for column, value in row.entries:
            data.append(float(value) / row_norms[row_index])
            indices.append(int(column))
        indptr.append(len(data))
    matrix = csr_matrix(
        (
            np.asarray(data, dtype=float),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=(jacobian.row_count, jacobian.column_count),
    )
    centered = coords - np.mean(coords, axis=0)
    rotation_axes = _rigid_rotation_axes(centered, gauge_count)
    column_order = _independent_rigid_gauge_columns(centered, rotation_axes)
    if len(column_order) != gauge_count:
        raise ValueError("sparse SONIC rigid gauge is rank deficient")
    fixed = set(column_order)
    free = [column for column in range(jacobian.column_count) if column not in fixed]
    if len(free) != jacobian.row_count:
        raise ValueError("sparse SONIC rigid gauge has the wrong dimension")
    factor = splu(matrix[:, free].tocsc(), permc_spec="COLAMD")
    pivots = np.abs(factor.U.diagonal())
    maximum = float(np.max(pivots, initial=0.0))
    minimum = float(np.min(pivots, initial=np.inf))
    cutoff = max(float(absolute_tolerance), float(tolerance) * maximum)
    rank = jacobian.row_count if minimum > cutoff else int(np.count_nonzero(pivots > cutoff))
    rigid_residual = max(
        (
            float(
                np.max(
                    np.abs(matrix @ _rigid_mode_vector(centered, mode, rotation_axes)),
                    initial=0.0,
                )
            )
            for mode in range(gauge_count)
        ),
        default=0.0,
    )
    condition = float(maximum / minimum) if minimum > 0.0 else float("inf")
    return {
        "schema": SONIC_CONDITION_DIAGNOSTICS_SCHEMA,
        "shape": [jacobian.row_count, jacobian.column_count],
        "rank": rank,
        "maximum_rank": jacobian.row_count,
        "singular_values": [maximum, minimum],
        "singular_value_cutoff": cutoff,
        "relative_tolerance": float(tolerance),
        "absolute_tolerance": float(absolute_tolerance),
        "condition_number": condition,
        "minimum_active_singular_value": minimum,
        "cutoff_margin": minimum / cutoff if cutoff else float("inf"),
        "status": "FULL_RANK" if rank == jacobian.row_count else "RANK_DEFICIENT",
        "near_dependent": rank != jacobian.row_count,
        "rank_method": "SPARSE_RIGID_GAUGE_SUPERLU",
        "rigid_mode_residual": rigid_residual,
        "stored_value_count": jacobian.nnz + row_norms.size,
    }


def sonic_condition_diagnostics(
    jacobian: np.ndarray,
    *,
    tolerance: float = 1.0e-8,
    absolute_tolerance: float = 0.0,
    warning_condition_number: float = 1.0e8,
) -> dict[str, object]:
    """Report scale-aware rank, conditioning and singularity state.

    The rank cutoff is ``max(abs_tol, rel_tol * s_max)``.  The legacy
    ``tolerance`` argument remains the relative tolerance so existing callers
    retain their numerical meaning.
    """

    matrix = np.asarray(jacobian, dtype=float)
    if matrix.ndim != 2 or not matrix.size or not np.isfinite(matrix).all():
        raise ValueError("SONIC Jacobian must be a finite nonempty matrix")
    relative = float(tolerance)
    absolute = float(absolute_tolerance)
    warning = float(warning_condition_number)
    if not np.isfinite(relative) or relative <= 0.0:
        raise ValueError("SONIC relative tolerance must be positive and finite")
    if not np.isfinite(absolute) or absolute < 0.0:
        raise ValueError("SONIC absolute tolerance must be finite and non-negative")
    if not np.isfinite(warning) or warning <= 1.0:
        raise ValueError("SONIC warning condition number must be greater than one")

    spectrum = singular_spectrum(
        matrix,
        absolute_tolerance=absolute,
        relative_tolerance=relative,
    )
    singular = spectrum.singular_values
    cutoff = spectrum.cutoff
    rank = spectrum.rank
    minimum_active = spectrum.minimum_active
    condition = spectrum.condition_number
    if not rank:
        status = "SINGULAR"
    elif rank < min(matrix.shape):
        status = "RANK_DEFICIENT"
    elif condition >= warning:
        status = "ILL_CONDITIONED"
    else:
        status = "FULL_RANK"
    margin = float(minimum_active / cutoff) if cutoff and minimum_active else 0.0
    return {
        "schema": SONIC_CONDITION_DIAGNOSTICS_SCHEMA,
        "shape": list(matrix.shape),
        "rank": rank,
        "maximum_rank": min(matrix.shape),
        "singular_values": singular.tolist(),
        "singular_value_cutoff": cutoff,
        "relative_tolerance": relative,
        "absolute_tolerance": absolute,
        "condition_number": condition,
        "minimum_active_singular_value": minimum_active,
        "cutoff_margin": margin,
        "status": status,
        "near_dependent": bool(status in {"RANK_DEFICIENT", "ILL_CONDITIONED", "SINGULAR"}),
    }


def normalized_sonic_condition_diagnostics(
    jacobian: np.ndarray,
    *,
    tolerance: float = 1.0e-8,
    maximum_condition_number: float,
) -> dict[str, object]:
    """Audit a SONIC basis independently of heterogeneous coordinate units."""

    matrix = np.asarray(jacobian, dtype=float)
    if matrix.ndim != 2 or not matrix.size or not np.isfinite(matrix).all():
        raise ValueError("SONIC Jacobian must be a finite nonempty matrix")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms <= 1.0e-12):
        raise ValueError("SONIC Jacobian contains a singular coordinate row")
    return sonic_condition_diagnostics(
        matrix / norms[:, None],
        tolerance=tolerance,
        warning_condition_number=maximum_condition_number,
    )
