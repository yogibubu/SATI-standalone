"""Matrix-free totally symmetric Cartesian optimization primitives.

The runtime deliberately represents a SYCART displacement in the ambient
Cartesian frame.  Its physical value is obtained by an orthogonal projector
that removes translations/rotations and averages over the frozen finite point
group.  No ``3N x 3N`` projector or symmetry-adapted Cartesian basis is ever
formed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from matrix_core import read_sectioned_lines, section_content


@dataclass(frozen=True)
class MatrixFreeRedundantPrimitiveTransform:
    """Matrix-free Pulay-style transform for a redundant primitive set.

    ``B`` is consumed only through sparse ``matvec``/``transpose_matvec``.
    Cartesian-to-primitive projection is direct; primitive-to-Cartesian
    back-transformation solves the minimum-norm least-squares problem with
    LSQR, so neither ``B`` nor ``B B.T`` is materialized densely.
    """

    b_matrix: object
    cartesian_dimension: int
    atol: float = 1.0e-8
    btol: float = 1.0e-8
    iter_lim: int = 128

    def __post_init__(self) -> None:
        from matrix_smith import GICBMatrix, SparseBMatrix
        if isinstance(self.b_matrix, GICBMatrix):
            object.__setattr__(self, "b_matrix", self.b_matrix.sparse_matrix())
        if not isinstance(self.b_matrix, SparseBMatrix):
            raise TypeError("coordinate transform requires SparseBMatrix or GICBMatrix")
        if self.b_matrix.column_count != int(self.cartesian_dimension):
            raise ValueError("B matrix and Cartesian dimensions disagree")

    @property
    def primitive_dimension(self) -> int:
        return self.b_matrix.row_count

    def primitive_from_cartesian(self, cartesian: Sequence[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(cartesian, dtype=float).reshape(-1)
        if values.size != int(self.cartesian_dimension):
            raise ValueError("Cartesian vector has the wrong dimension")
        return self.b_matrix.matvec(values)

    def cartesian_from_primitive(self, primitive: Sequence[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(primitive, dtype=float).reshape(-1)
        if values.size != self.primitive_dimension:
            raise ValueError("primitive vector has the wrong dimension")
        from scipy.sparse.linalg import LinearOperator, lsqr
        operator = LinearOperator(
            (self.primitive_dimension, int(self.cartesian_dimension)),
            matvec=self.b_matrix.matvec,
            rmatvec=self.b_matrix.transpose_matvec,
            dtype=float,
        )
        result = lsqr(operator, values, atol=self.atol, btol=self.btol, iter_lim=self.iter_lim)
        if result[1] not in (0, 1, 2):
            raise RuntimeError(f"redundant primitive back-transformation did not converge: istop={result[1]}")
        return np.asarray(result[0], dtype=float)


# The same operator applies to a non-redundant delocalized GIC matrix; the
# alias makes that route explicit at call sites and in benchmark reports.
MatrixFreeDelocalizedTransform = MatrixFreeRedundantPrimitiveTransform


@dataclass(frozen=True)
class HessianEigenvectorTransform:
    """Reference modal coordinate transform from an initial Hessian.

    This is intentionally an audit/benchmark alternative: unlike the sparse
    transforms above, its eigenvector basis is generally dense and therefore
    does not claim linear-scaling storage.
    """

    eigenvectors: np.ndarray

    def __post_init__(self) -> None:
        modes = np.asarray(self.eigenvectors, dtype=float)
        if modes.ndim != 2 or modes.shape[0] != modes.shape[1] or not np.all(np.isfinite(modes)):
            raise ValueError("Hessian eigenvectors must be a finite square matrix")
        if not np.allclose(modes.T @ modes, np.eye(modes.shape[0]), atol=1.0e-7):
            raise ValueError("Hessian eigenvectors must be orthonormal")
        object.__setattr__(self, "eigenvectors", modes)

    def primitive_from_cartesian(self, cartesian: Sequence[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(cartesian, dtype=float).reshape(-1)
        return self.eigenvectors.T @ values

    def cartesian_from_primitive(self, modal: Sequence[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(modal, dtype=float).reshape(-1)
        return self.eigenvectors @ values


def _apply_cartesian_operation(
    vector: np.ndarray,
    rotation: np.ndarray,
    permutation: np.ndarray,
) -> np.ndarray:
    values = np.asarray(vector, dtype=float).reshape((-1, 3))
    # This is exactly the action defined by matrix_chem.cartesian_operation_matrix:
    # output[source] = rotation @ input[target].
    return (values[permutation] @ rotation.T).reshape(-1)


@dataclass(frozen=True)
class MatrixFreeCartesianSymmetryProjector:
    """Totally symmetric Cartesian projector with linear storage.

    Finite groups use their exact group average.  Sampled continuous groups
    use cyclic projections onto the fixed space of every sampled operation;
    this avoids both the stacked constraint matrix and the ``3N x 3N``
    projector formerly built by LINK.
    """

    rotations: tuple[np.ndarray, ...]
    permutations: tuple[np.ndarray, ...]
    cartesian_dimension: int
    point_group: str
    continuous_sample: bool = False
    tolerance: float = 1.0e-12
    maximum_sweeps: int = 64

    @property
    def stored_value_count(self) -> int:
        return int(sum(item.size for item in self.rotations + self.permutations))

    def _apply_operation(self, values: np.ndarray, index: int) -> np.ndarray:
        if values.ndim == 1:
            return _apply_cartesian_operation(
                values, self.rotations[index], self.permutations[index]
            )
        shaped = values.reshape((-1, 3, values.shape[1]))
        return np.einsum(
            "ab,ibk->iak",
            self.rotations[index],
            shaped[self.permutations[index]],
            optimize=True,
        ).reshape(values.shape)

    def _operation_project(self, values: np.ndarray, index: int) -> np.ndarray:
        current = values
        accumulated = np.array(values, copy=True)
        for power in range(1, 121):
            current = self._apply_operation(current, index)
            if np.allclose(current, values, rtol=0.0, atol=1.0e-11):
                return accumulated / float(power)
            accumulated += current
        raise ValueError(
            "frozen symmetry operation does not have a certified finite sampled order"
        )

    def project(self, vector: Sequence[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(vector, dtype=float)
        if values.ndim not in {1, 2} or values.shape[0] != self.cartesian_dimension:
            raise ValueError("symmetry projector input must have leading dimension 3N")
        if not np.all(np.isfinite(values)):
            raise ValueError("symmetry projector input contains non-finite values")
        if not self.continuous_sample:
            result = np.zeros_like(values)
            for index in range(len(self.rotations)):
                result += self._apply_operation(values, index)
            return result / float(len(self.rotations))
        result = np.array(values, copy=True)
        for _sweep in range(self.maximum_sweeps):
            previous = result.copy()
            for index in range(len(self.rotations)):
                result = self._operation_project(result, index)
            scale = max(float(np.linalg.norm(result)), 1.0)
            if float(np.linalg.norm(result - previous)) <= self.tolerance * scale:
                return result
        raise ValueError("sampled continuous-group projector did not converge")

    def project_rows(self, rows: np.ndarray) -> np.ndarray:
        matrix = np.asarray(rows, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != self.cartesian_dimension:
            raise ValueError("symmetry row block must have shape (nrow, 3N)")
        return self.project(matrix.T).T


def build_matrix_free_symmetry_projector(
    symmetry: object,
    *,
    natoms: int,
) -> MatrixFreeCartesianSymmetryProjector:
    """Compile a frozen Cartesian group action without a dense matrix."""

    atom_count = int(natoms)
    operations = tuple(getattr(symmetry, "operations", ()))
    if atom_count <= 0 or not operations:
        raise ValueError("matrix-free symmetry projection needs atoms and operations")
    rotations = tuple(np.asarray(item.rotation, dtype=float) for item in operations)
    permutations = tuple(
        np.asarray([int(atom) - 1 for atom in item.permutation], dtype=int)
        for item in operations
    )
    for rotation, permutation in zip(rotations, permutations, strict=True):
        if rotation.shape != (3, 3) or np.linalg.norm(
            rotation.T @ rotation - np.eye(3), ord=2
        ) > 1.0e-8:
            raise ValueError("frozen Cartesian symmetry rotation is not orthogonal")
        if permutation.shape != (atom_count,) or sorted(permutation.tolist()) != list(
            range(atom_count)
        ):
            raise ValueError("frozen Cartesian symmetry permutation is invalid")
    point_group = str(getattr(symmetry, "point_group", "C1")).strip().upper()
    return MatrixFreeCartesianSymmetryProjector(
        rotations=rotations,
        permutations=permutations,
        cartesian_dimension=3 * atom_count,
        point_group=point_group,
        continuous_sample=point_group in {"CINFV", "DINFH"},
    )


def _external_motion_rows(coordinates_angstrom: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("SYCART reference coordinates must have shape (natoms, 3)")
    natoms = coordinates.shape[0]
    centered = coordinates - np.mean(coordinates, axis=0)
    candidates = []
    identity = np.eye(3, dtype=float)
    for axis in identity:
        candidates.append(np.tile(axis, natoms))
    for axis in identity:
        candidates.append(np.asarray([np.cross(axis, row) for row in centered]).reshape(-1))
    candidate_matrix = np.asarray(candidates, dtype=float)
    _left, singular_values, right_transpose = np.linalg.svd(
        candidate_matrix,
        full_matrices=False,
    )
    scale = max(float(singular_values[0]) if singular_values.size else 0.0, 1.0)
    threshold = max(
        1.0e-10 * scale,
        np.finfo(float).eps * max(candidate_matrix.shape) * scale,
    )
    rank = int(np.count_nonzero(singular_values > threshold))
    if rank not in {3, 5, 6}:
        raise ValueError(f"unsupported Cartesian external-motion rank {rank}")
    return np.asarray(right_transpose[:rank], dtype=float)


@dataclass(frozen=True)
class MatrixFreeSYCartRuntime:
    """Orthogonal totally-symmetric vibrational projector in ``O(N)`` storage."""

    rotations: tuple[np.ndarray, ...]
    permutations: tuple[np.ndarray, ...]
    external_rows: np.ndarray
    cartesian_dimension: int
    active_rank: int
    point_group: str

    def __post_init__(self) -> None:
        dimension = int(self.cartesian_dimension)
        if dimension <= 0 or dimension % 3:
            raise ValueError("SYCART Cartesian dimension must be a positive multiple of three")
        if not self.rotations or len(self.rotations) != len(self.permutations):
            raise ValueError("SYCART runtime needs matched frozen symmetry operations")
        natoms = dimension // 3
        rotations = tuple(np.asarray(item, dtype=float).copy() for item in self.rotations)
        permutations = tuple(np.asarray(item, dtype=int).reshape(-1).copy() for item in self.permutations)
        for rotation, permutation in zip(rotations, permutations, strict=True):
            if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
                raise ValueError("SYCART symmetry rotation must be finite and 3 by 3")
            if np.linalg.norm(rotation.T @ rotation - np.eye(3), ord=2) > 1.0e-8:
                raise ValueError("SYCART symmetry rotation must be orthogonal")
            if permutation.shape != (natoms,) or sorted(permutation.tolist()) != list(range(natoms)):
                raise ValueError("SYCART symmetry permutation is invalid")
        external = np.asarray(self.external_rows, dtype=float).copy()
        if external.ndim != 2 or external.shape[1] != dimension:
            raise ValueError("SYCART external modes do not match the Cartesian frame")
        if np.linalg.norm(external @ external.T - np.eye(external.shape[0]), ord=2) > 1.0e-8:
            raise ValueError("SYCART external modes must be orthonormal")
        rank = int(self.active_rank)
        if rank <= 0 or rank > dimension - external.shape[0]:
            raise ValueError("SYCART active rank is invalid")
        object.__setattr__(self, "rotations", rotations)
        object.__setattr__(self, "permutations", permutations)
        object.__setattr__(self, "external_rows", external)
        object.__setattr__(self, "cartesian_dimension", dimension)
        object.__setattr__(self, "active_rank", rank)
        object.__setattr__(self, "point_group", str(self.point_group).strip().upper() or "C1")

    @property
    def stored_value_count(self) -> int:
        return int(
            self.external_rows.size
            + sum(rotation.size + permutation.size for rotation, permutation in zip(
                self.rotations, self.permutations, strict=True
            ))
        )

    def _remove_external(self, vector: np.ndarray) -> np.ndarray:
        values = np.asarray(vector, dtype=float).reshape(-1)
        return values - self.external_rows.T @ (self.external_rows @ values)

    def _symmetry_average(self, vector: np.ndarray) -> np.ndarray:
        values = np.asarray(vector, dtype=float).reshape(-1)
        output = np.zeros_like(values)
        for rotation, permutation in zip(self.rotations, self.permutations, strict=True):
            output += _apply_cartesian_operation(values, rotation, permutation)
        return output / float(len(self.rotations))

    def project(self, vector: Sequence[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(vector, dtype=float).reshape(-1)
        if values.shape != (self.cartesian_dimension,) or not np.all(np.isfinite(values)):
            raise ValueError("SYCART projector vector does not match the Cartesian frame")
        # The rigid-motion space is invariant under the molecular group, so
        # these projectors commute analytically.  Applying the small external
        # projector on both sides suppresses only round-off noncommutation.
        internal = self._remove_external(values)
        symmetric = self._symmetry_average(internal)
        return self._remove_external(symmetric)


def build_matrix_free_sycart_runtime(
    symmetry: object,
    coordinates_angstrom: np.ndarray,
    *,
    expected_active_rank: int | None = None,
) -> MatrixFreeSYCartRuntime:
    """Compile the frozen finite-group SYCART projector without a dense basis."""

    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or not np.all(np.isfinite(coordinates)):
        raise ValueError("SYCART reference coordinates must be finite and natoms by 3")
    natoms = coordinates.shape[0]
    point_group = str(getattr(symmetry, "point_group", "C1")).strip().upper() or "C1"
    if point_group in {"CINFV", "DINFH"}:
        raise ValueError(
            "matrix-free SYCART currently requires a finite frozen point group; "
            "linear molecules retain the certified dense SMITH basis"
        )
    operations = tuple(getattr(symmetry, "operations", ()))
    if not operations:
        if point_group not in {"C1", "UNKNOWN"}:
            raise ValueError("nontrivial matrix-free SYCART group has no frozen operations")
        rotations = (np.eye(3, dtype=float),)
        permutations = (np.arange(natoms, dtype=int),)
    else:
        rotations = tuple(np.asarray(operation.rotation, dtype=float) for operation in operations)
        permutations = tuple(
            np.asarray([int(atom) - 1 for atom in operation.permutation], dtype=int)
            for operation in operations
        )
    external = _external_motion_rows(coordinates)

    def symmetry_average(vector: np.ndarray) -> np.ndarray:
        result = np.zeros(3 * natoms, dtype=float)
        for rotation, permutation in zip(rotations, permutations, strict=True):
            result += _apply_cartesian_operation(vector, rotation, permutation)
        return result / float(len(rotations))

    symmetry_trace = 0.0
    for rotation, permutation in zip(rotations, permutations, strict=True):
        for atom, target in enumerate(permutation):
            if atom == int(target):
                symmetry_trace += float(np.trace(rotation))
    symmetry_trace /= float(len(rotations))
    external_symmetric_trace = sum(
        float(row @ symmetry_average(row)) for row in external
    )
    active_rank = int(round(symmetry_trace - external_symmetric_trace))
    if expected_active_rank is not None and active_rank != int(expected_active_rank):
        raise ValueError(
            "matrix-free SYCART rank disagrees with the frozen SMITH contract: "
            f"runtime={active_rank}, contract={int(expected_active_rank)}"
        )
    runtime = MatrixFreeSYCartRuntime(
        rotations=rotations,
        permutations=permutations,
        external_rows=external,
        cartesian_dimension=3 * natoms,
        active_rank=active_rank,
        point_group=point_group,
    )
    # Deterministic operator audits remain linear-memory.  They catch an
    # incomplete operation set, a convention mismatch, or noncommuting T/R
    # removal without ever materializing P.
    generator = np.random.default_rng(7264381)
    for _index in range(min(3, active_rank)):
        probe = generator.standard_normal(3 * natoms)
        projected = runtime.project(probe)
        idempotency = float(np.linalg.norm(runtime.project(projected) - projected))
        scale = max(float(np.linalg.norm(projected)), 1.0)
        if idempotency > 1.0e-8 * scale:
            raise ValueError(
                "matrix-free SYCART frozen operation average is not an orthogonal projector "
                f"(residual={idempotency / scale:.3e})"
            )
    return runtime


def frozen_sycart_total_symmetric_rank(path: Path | str) -> tuple[int, str]:
    """Read only SYCART rank/irrep metadata, never its serialized vectors."""

    from matrix_smith import is_total_symmetric_irrep

    lines = read_sectioned_lines(Path(path))
    section = section_content(lines, "SYCART")
    if not section:
        raise ValueError("missing frozen #SYCART section")
    fields: dict[str, str] = {}
    coordinate_irreps: list[str] = []
    for raw in section:
        text = raw.strip()
        if not text:
            continue
        parts = text.split()
        if parts[0].upper().startswith("SYC") and parts[0][3:].isdigit():
            encoded = {
                key.upper(): value
                for part in parts[1:]
                if "=" in part
                for key, value in (part.split("=", 1),)
            }
            irrep = encoded.get("IRREP")
            if not irrep:
                raise ValueError("frozen #SYCART coordinate lacks an irrep")
            coordinate_irreps.append(irrep)
        elif len(parts) >= 2:
            fields[parts[0].upper()] = " ".join(parts[1:])
    point_group = fields.get("POINT_GROUP", "C1").strip().upper()
    try:
        coordinate_count = int(fields.get("COORD_COUNT", "0"))
        target_rank = int(fields.get("TARGET_RANK", "0"))
    except ValueError as exc:
        raise ValueError("invalid frozen #SYCART rank metadata") from exc
    if coordinate_count <= 0 or coordinate_count != target_rank:
        raise ValueError("inconsistent frozen #SYCART rank metadata")
    if len(coordinate_irreps) != coordinate_count:
        raise ValueError("frozen #SYCART coordinate count is incomplete")
    active_rank = sum(
        int(is_total_symmetric_irrep(point_group, irrep)) for irrep in coordinate_irreps
    )
    if active_rank <= 0:
        raise ValueError("frozen #SYCART contract has no totally symmetric coordinates")
    return active_rank, point_group


@dataclass(frozen=True)
class DavidsonPreconditioner:
    """Bounded Davidson deflation inside a projected Krylov inverse action.

    The stored eigenvectors accelerate the solve but do not define its
    accuracy.  Consequently the number of Davidson roots can remain bounded
    as the molecule grows; the residual-controlled Krylov iteration recovers
    the remaining Cartesian modes without forming or storing a dense matrix.
    """

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    baseline_inverse: float
    iterations: int
    converged: bool
    operator: Callable[[np.ndarray], np.ndarray] | None = None
    projector: Callable[[np.ndarray], np.ndarray] | None = None
    regularization_shift: float = 0.0
    krylov_relative_tolerance: float = 1.0e-3
    krylov_maximum_iterations: int = 64

    def _apply_deflation(self, vector: np.ndarray) -> np.ndarray:
        """Apply the bounded low-rank Davidson inverse preconditioner."""

        values = np.asarray(vector, dtype=float).reshape(-1)
        vectors = np.asarray(self.eigenvectors, dtype=float)
        if values.shape != (vectors.shape[0],):
            raise ValueError("Davidson preconditioner vector has the wrong dimension")
        result = float(self.baseline_inverse) * values
        if vectors.shape[1]:
            coefficients = vectors.T @ values
            corrections = (1.0 / self.eigenvalues - float(self.baseline_inverse)) * coefficients
            result = result + vectors @ corrections
        return result

    def apply(self, vector: Sequence[float] | np.ndarray) -> np.ndarray:
        """Solve the regularized projected Hessian action to a relative residual."""

        values = np.asarray(vector, dtype=float).reshape(-1)
        vectors = np.asarray(self.eigenvectors, dtype=float)
        if values.shape != (vectors.shape[0],):
            raise ValueError("Davidson preconditioner vector has the wrong dimension")
        if self.operator is None or self.projector is None:
            return self._apply_deflation(values)
        tolerance = float(self.krylov_relative_tolerance)
        maximum_iterations = int(self.krylov_maximum_iterations)
        shift = float(self.regularization_shift)
        if (
            not np.isfinite(tolerance)
            or tolerance <= 0.0
            or maximum_iterations <= 0
            or not np.isfinite(shift)
            or shift < 0.0
        ):
            raise ValueError("invalid projected Krylov inverse controls")

        project = self.projector
        operator = self.operator
        rhs = np.asarray(project(values), dtype=float).reshape(-1)
        rhs_norm = float(np.linalg.norm(rhs))
        if rhs_norm <= np.finfo(float).tiny:
            return np.zeros_like(rhs)
        solution = np.zeros_like(rhs)
        residual = rhs.copy()
        preconditioned = np.asarray(
            project(self._apply_deflation(residual)), dtype=float
        ).reshape(-1)
        direction = preconditioned.copy()
        residual_product = float(residual @ preconditioned)
        target = tolerance * rhs_norm
        for _iteration in range(maximum_iterations):
            action = np.asarray(project(operator(direction)), dtype=float).reshape(-1)
            action += shift * direction
            denominator = float(direction @ action)
            if not np.isfinite(denominator) or denominator <= 0.0:
                break
            step = residual_product / denominator
            solution += step * direction
            residual = np.asarray(project(residual - step * action), dtype=float).reshape(-1)
            if float(np.linalg.norm(residual)) <= target:
                return np.asarray(project(solution), dtype=float).reshape(-1)
            preconditioned = np.asarray(
                project(self._apply_deflation(residual)), dtype=float
            ).reshape(-1)
            next_product = float(residual @ preconditioned)
            if not np.isfinite(next_product) or next_product <= 0.0:
                break
            direction = preconditioned + (next_product / residual_product) * direction
            residual_product = next_product
        # A bounded inexact solve is still a valid positive inverse action.
        # The owning L-BFGS descent gate and Cartesian trust region provide
        # the final fail-closed safeguards.
        return np.asarray(project(solution), dtype=float).reshape(-1)


@dataclass(frozen=True)
class ProjectedPrimitiveHessianOperator:
    """Apply ``P B.T K B P`` from sparse primitive rows."""

    runtime: MatrixFreeSYCartRuntime
    b_matrix: object | None
    curvatures: np.ndarray
    csr_matrix: object | None = None

    def __post_init__(self) -> None:
        from matrix_smith import SparseBMatrix
        from scipy.sparse import csr_matrix

        if self.b_matrix is None and self.csr_matrix is None:
            raise TypeError("projected primitive Hessian requires sparse Wilson rows")
        if self.b_matrix is not None and not isinstance(self.b_matrix, SparseBMatrix):
            raise TypeError("projected primitive Hessian requires a SparseBMatrix")
        values = np.asarray(self.curvatures, dtype=float).reshape(-1).copy()
        if (
            values.ndim != 1
            or not np.all(np.isfinite(values))
            or np.any(values <= 0.0)
        ):
            raise ValueError("primitive Hessian curvatures must be finite and positive")
        object.__setattr__(self, "curvatures", values)
        if self.csr_matrix is None:
            if self.b_matrix.row_count != values.size or self.b_matrix.column_count != self.runtime.cartesian_dimension:
                raise ValueError("primitive Hessian and SYCART runtime dimensions differ")
            object.__setattr__(self, "_csr", None)
            return
        if self.csr_matrix is None:
            row_indices: list[int] = []
            column_indices: list[int] = []
            entries: list[float] = []
            for row_index, row in enumerate(self.b_matrix.rows):
                for column_index, value in row.entries:
                    row_indices.append(row_index)
                    column_indices.append(column_index)
                    entries.append(value)
            csr = csr_matrix(
                (entries, (row_indices, column_indices)),
                shape=(self.b_matrix.row_count, self.b_matrix.column_count),
                dtype=float,
            )
        else:
            csr = csr_matrix(self.csr_matrix, dtype=float, copy=False)
        if csr.shape != (values.size, self.runtime.cartesian_dimension):
            raise ValueError("primitive Hessian and SYCART runtime dimensions differ")
        object.__setattr__(self, "_csr", csr)

    @property
    def stored_value_count(self) -> int:
        if self._csr is None:
            return int(self.runtime.stored_value_count + self.b_matrix.nnz + self.curvatures.size)
        csr = self._csr
        return int(
            self.runtime.stored_value_count
            + self._csr.nnz
            + self.curvatures.size
            + csr.data.size
            + csr.indices.size
            + csr.indptr.size
        )

    @property
    def baseline_curvature(self) -> float:
        if self._csr is None:
            row_norms = np.asarray(
                [sum(value * value for _index, value in row.entries) for row in self.b_matrix.rows],
                dtype=float,
            )
            return max(float(np.dot(self.curvatures, row_norms)) / max(self.runtime.active_rank, 1), 1.0e-4)
        row_norm_contributions = self.curvatures * np.asarray(
            self._csr.multiply(self._csr).sum(axis=1),
            dtype=float,
        ).reshape(-1)
        total = float(np.sum(row_norm_contributions))
        return max(total / max(self.runtime.active_rank, 1), 1.0e-4)

    def matvec(self, vector: Sequence[float] | np.ndarray) -> np.ndarray:
        projected = self.runtime.project(vector)
        if self._csr is None:
            primitive = self.b_matrix.matvec(projected)
            cartesian = self.b_matrix.transpose_matvec(self.curvatures * primitive)
            return self.runtime.project(cartesian)
        primitive = np.asarray(self._csr @ projected, dtype=float).reshape(-1)
        cartesian = np.asarray(
            self._csr.T @ (self.curvatures * primitive), dtype=float
        ).reshape(-1)
        return self.runtime.project(cartesian)


def davidson_inverse_preconditioner(
    operator: Callable[[np.ndarray], np.ndarray],
    projector: Callable[[np.ndarray], np.ndarray],
    *,
    dimension: int,
    active_rank: int,
    roots: int = 12,
    maximum_iterations: int = 64,
    maximum_subspace: int = 96,
    convergence: float = 1.0e-7,
    eigenvalue_floor: float = 1.0e-5,
    baseline_curvature: float = 0.5,
    krylov_relative_tolerance: float = 1.0e-3,
    krylov_maximum_iterations: int = 64,
    initial_vectors: np.ndarray | None = None,
) -> DavidsonPreconditioner:
    """Build a bounded Davidson-deflated, residual-controlled Krylov inverse."""

    n = int(dimension)
    root_count = min(max(int(roots), 1), int(active_rank), n)
    if root_count <= 0:
        raise ValueError("Davidson needs a nonempty active SYCART subspace")
    generator = np.random.default_rng(18273645)

    def append_orthonormal(columns: list[np.ndarray], candidate: np.ndarray) -> bool:
        vector = np.asarray(projector(candidate), dtype=float).reshape(-1)
        for basis in columns:
            vector -= basis * float(basis @ vector)
        # One reorthogonalization is inexpensive and important for the very
        # low-rank totally symmetric spaces of high-symmetry molecules.
        for basis in columns:
            vector -= basis * float(basis @ vector)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 1.0e-10:
            return False
        columns.append(vector / norm)
        return True

    basis_columns: list[np.ndarray] = []
    if initial_vectors is not None:
        seeds = np.asarray(initial_vectors, dtype=float)
        if seeds.ndim != 2 or seeds.shape[0] != n or not np.all(np.isfinite(seeds)):
            raise ValueError("Davidson initial vectors do not match the projected space")
        for column in seeds.T:
            if len(basis_columns) >= root_count:
                break
            append_orthonormal(basis_columns, column)
    attempts = 0
    while len(basis_columns) < root_count and attempts < 8 * root_count + 16:
        append_orthonormal(basis_columns, generator.standard_normal(n))
        attempts += 1
    if len(basis_columns) < root_count:
        raise RuntimeError("Davidson could not seed the projected SYCART subspace")
    action_columns = [np.asarray(operator(column), dtype=float) for column in basis_columns]
    last_values = np.full(root_count, float(baseline_curvature), dtype=float)
    last_vectors = np.column_stack(basis_columns[:root_count])
    converged = False
    completed_iterations = 0
    for iteration in range(1, int(maximum_iterations) + 1):
        completed_iterations = iteration
        basis = np.column_stack(basis_columns)
        actions = np.column_stack(action_columns)
        reduced = 0.5 * (basis.T @ actions + actions.T @ basis)
        values, coefficients = np.linalg.eigh(reduced)
        selected = np.argsort(values)[:root_count]
        last_values = values[selected]
        last_vectors = basis @ coefficients[:, selected]
        residuals = actions @ coefficients[:, selected] - last_vectors * last_values[None, :]
        residual_norms = np.linalg.norm(residuals, axis=0)
        if np.all(residual_norms <= float(convergence)):
            converged = True
            break
        corrections: list[np.ndarray] = []
        scale = max(float(np.max(np.abs(last_values), initial=0.0)), float(baseline_curvature))
        for root, residual in enumerate(residuals.T):
            if residual_norms[root] <= float(convergence):
                continue
            denominator = max(abs(float(last_values[root])), 0.1 * scale, eigenvalue_floor)
            candidate = -residual / denominator
            combined = [*basis_columns, *corrections]
            if append_orthonormal(combined, candidate):
                corrections.append(combined[-1])
        if not corrections:
            break
        if len(basis_columns) + len(corrections) > int(maximum_subspace):
            restarted: list[np.ndarray] = []
            for column in last_vectors.T:
                append_orthonormal(restarted, column)
            basis_columns = restarted
            action_columns = [np.asarray(operator(column), dtype=float) for column in basis_columns]
        else:
            basis_columns.extend(corrections)
            action_columns.extend(np.asarray(operator(column), dtype=float) for column in corrections)
    positive = np.maximum(np.asarray(last_values, dtype=float), float(eigenvalue_floor))
    baseline_inverse = 1.0 / max(float(baseline_curvature), float(eigenvalue_floor))
    return DavidsonPreconditioner(
        eigenvalues=positive,
        eigenvectors=np.asarray(last_vectors, dtype=float),
        baseline_inverse=baseline_inverse,
        iterations=completed_iterations,
        converged=converged,
        operator=operator,
        projector=projector,
        regularization_shift=float(eigenvalue_floor),
        krylov_relative_tolerance=float(krylov_relative_tolerance),
        krylov_maximum_iterations=int(krylov_maximum_iterations),
    )


__all__ = [
    "DavidsonPreconditioner",
    "MatrixFreeSYCartRuntime",
    "ProjectedPrimitiveHessianOperator",
    "build_matrix_free_sycart_runtime",
    "davidson_inverse_preconditioner",
    "frozen_sycart_total_symmetric_rank",
]
