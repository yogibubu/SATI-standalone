"""Minimal local delocalization of ORACLE primitive classes.

Only closed local blocks are diagonalized. Intrafragment/interfragment and
symmetry partitions are immutable. No mass matrix appears in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.sparse import csr_matrix, vstack
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.sparse.linalg import svds

from matrix_oracle.primitive_classes import (
    PRIMITIVE_DOMAIN_INTER,
    PRIMITIVE_DOMAIN_INTRA,
    PrimitiveClassAssignment,
)

from .bmatrix import SparseBMatrix


ADAPTIVE_DELOCALIZATION_SCHEMA = "matrix.smith.adaptive_delocalization.v1"
METRIC_GEOMETRY = "GEOMETRY"
METRIC_GEOMETRY_SH = "GEOMETRY_SH"


@dataclass(frozen=True)
class AdaptiveDelocalizationPolicy:
    metric_mode: str = METRIC_GEOMETRY
    maximum_condition_number: float = 1.0e6
    rank_tolerance: float = 1.0e-10
    maximum_local_block_rows: int = 192

    def __post_init__(self) -> None:
        mode = str(self.metric_mode).strip().upper().replace("-", "_")
        if mode not in {METRIC_GEOMETRY, METRIC_GEOMETRY_SH}:
            raise ValueError("adaptive metric must be GEOMETRY or GEOMETRY_SH")
        if self.maximum_condition_number <= 1.0 or self.rank_tolerance <= 0.0:
            raise ValueError("invalid adaptive conditioning tolerances")
        if self.maximum_local_block_rows <= 0:
            raise ValueError("maximum local block size must be positive")
        object.__setattr__(self, "metric_mode", mode)


@dataclass(frozen=True)
class AdaptiveDelocalizationCertificate:
    status: str
    metric_mode: str
    selected_stage: int
    selected_maximum_class_rank: int | None
    target_rank: int
    realized_rank: int
    sycart_rank: int
    selection_condition_number: float
    geometric_condition_number: float
    block_count: int
    mixed_class_blocks: tuple[str, ...]
    sycart_domains: tuple[str, ...]
    invariant: str = "NO_INTRA_INTER_OR_IRREP_MIXING"
    schema: str = ADAPTIVE_DELOCALIZATION_SCHEMA


@dataclass(frozen=True)
class AdaptiveDelocalizedBasis:
    primitive_transformation: csr_matrix
    b_matrix: SparseBMatrix
    certificate: AdaptiveDelocalizationCertificate


def _as_csr(matrix: SparseBMatrix) -> csr_matrix:
    data, indices, indptr = matrix.csr_arrays()
    return csr_matrix(
        (data, indices, indptr),
        shape=(matrix.row_count, matrix.column_count),
    )


def _sparse_condition(matrix: csr_matrix, tolerance: float) -> tuple[int, float]:
    """Bounded dense audit for tiny blocks, sparse extremal spectrum otherwise."""

    if not matrix.shape[0]:
        return 0, 1.0
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).reshape(-1))
    if np.any(norms <= tolerance):
        return 0, float("inf")
    normalized = matrix.multiply((1.0 / norms)[:, None]).tocsr()
    if min(normalized.shape) <= 96:
        singular = np.linalg.svd(normalized.toarray(), compute_uv=False)
        rank = int(np.count_nonzero(singular > tolerance))
        condition = float("inf") if rank < len(norms) else float(singular[0] / singular[-1])
        return rank, condition
    try:
        largest = float(svds(normalized, k=1, which="LM", return_singular_vectors=False)[0])
        smallest = float(svds(normalized, k=1, which="SM", return_singular_vectors=False)[0])
    except (ValueError, np.linalg.LinAlgError):
        return 0, float("inf")
    if smallest <= tolerance or not np.isfinite(smallest + largest):
        return 0, float("inf")
    return matrix.shape[0], largest / smallest


def _structural_row_indices(matrix: csr_matrix) -> np.ndarray:
    support = matrix.copy()
    support.data = np.ones_like(support.data, dtype=np.int8)
    matching = maximum_bipartite_matching(support, perm_type="column")
    return np.asarray(np.flatnonzero(matching >= 0), dtype=int)


def _structurally_independent_rows(matrix: csr_matrix, count: int) -> np.ndarray:
    selected = _structural_row_indices(matrix)
    if selected.size < count:
        return np.asarray([], dtype=int)
    return np.asarray(selected[:count], dtype=int)


def _least_mixed_partition_transform(
    work: csr_matrix,
    indices: tuple[int, ...],
    assignments: Sequence[PrimitiveClassAssignment],
    partition: tuple[str, str, str],
    settings: AdaptiveDelocalizationPolicy,
) -> tuple[csr_matrix, int, int | None, int, tuple[str, ...], bool]:
    # Structural matching removes redundant local rows without a numerical
    # factorization of the complete molecular matrix.
    local_rank = int(_structural_row_indices(work[np.asarray(indices, dtype=int)]).size)
    by_rank: dict[int, list[int]] = {}
    for index in indices:
        by_rank.setdefault(assignments[index].class_rank, []).append(index)
    ranks = sorted(by_rank)
    last = None
    for stage, cutoff in enumerate((None, *ranks)):
        blocks: list[tuple[int, ...]] = []
        mixed: tuple[str, ...] = ()
        if cutoff is None:
            blocks.extend(tuple(values) for _rank, values in sorted(by_rank.items()))
        else:
            admitted = tuple(
                index
                for rank, values in sorted(by_rank.items())
                if rank <= cutoff
                for index in values
            )
            if admitted:
                blocks.append(admitted)
                if len({assignments[index].class_rank for index in admitted}) > 1:
                    mixed = ("::".join((*partition, f"LE_{cutoff}")),)
            blocks.extend(
                tuple(values) for rank, values in sorted(by_rank.items()) if rank > cutoff
            )
        modes = tuple(
            _block_modes(
                work,
                block,
                tolerance=settings.rank_tolerance,
                maximum_rows=settings.maximum_local_block_rows,
            )
            for block in blocks
        )
        transform = vstack(modes, format="csr") if modes else csr_matrix((0, work.shape[0]))
        candidate = (transform @ work).tocsr()
        selected = _structurally_independent_rows(candidate, local_rank)
        if selected.size != local_rank:
            continue
        transform = transform[selected]
        rank, condition = _sparse_condition(candidate[selected], settings.rank_tolerance)
        last = (transform, stage, cutoff, len(blocks), mixed, rank == local_rank)
        if rank == local_rank and condition <= settings.maximum_condition_number:
            return transform, stage, cutoff, len(blocks), mixed, True
    if last is None:
        return csr_matrix((0, work.shape[0])), 0, None, 0, (), False
    return last


def _block_modes(
    work_rows: csr_matrix,
    row_indices: tuple[int, ...],
    *,
    tolerance: float,
    maximum_rows: int,
) -> csr_matrix:
    if len(row_indices) > maximum_rows:
        raise ValueError(
            "adaptive delocalization block exceeds the local dense budget; "
            "ORACLE must split its domain"
        )
    local = work_rows[np.asarray(row_indices, dtype=int)].toarray()
    gram = local @ local.T
    values, vectors = np.linalg.eigh(gram)
    scale = float(values[-1]) if values.size else 0.0
    keep = values > max(tolerance, tolerance * scale)
    if not np.any(keep):
        return csr_matrix((0, work_rows.shape[0]), dtype=float)
    coefficients = vectors[:, keep].T
    rows = np.repeat(np.arange(coefficients.shape[0]), coefficients.shape[1])
    cols = np.tile(np.asarray(row_indices, dtype=int), coefficients.shape[0])
    return csr_matrix(
        (coefficients.reshape(-1), (rows, cols)),
        shape=(coefficients.shape[0], work_rows.shape[0]),
    )


def build_adaptive_delocalized_basis(
    primitive_b_matrix: SparseBMatrix,
    assignments: Sequence[PrimitiveClassAssignment],
    *,
    irreps: Sequence[str] | None = None,
    topology_certified: bool | Mapping[str, bool] = True,
    target_rank: int | None = None,
    sh_curvatures: Sequence[float] | None = None,
    policy: AdaptiveDelocalizationPolicy | None = None,
) -> AdaptiveDelocalizedBasis:
    """Choose the least class mixing that satisfies the conditioning gate.

    ``GEOMETRY`` uses normalized ``B B.T``. ``GEOMETRY_SH`` additionally
    weights normalized primitive rows by ``sqrt(K_SH)``. The latter requires
    positive SH curvatures; neither route uses atomic masses.
    """

    settings = policy or AdaptiveDelocalizationPolicy()
    if len(assignments) != primitive_b_matrix.row_count:
        raise ValueError("one ORACLE class assignment is required per primitive row")
    labels = tuple(str(value) for value in (irreps or ("A" for _ in assignments)))
    if len(labels) != len(assignments):
        raise ValueError("one irrep label is required per primitive row")
    for assignment in assignments:
        if assignment.domain_kind not in {PRIMITIVE_DOMAIN_INTRA, PRIMITIVE_DOMAIN_INTER}:
            raise ValueError("invalid ORACLE fragment-domain assignment")

    requested_rank = (
        int(target_rank) if target_rank is not None else int(primitive_b_matrix.structural_rank())
    )
    if requested_rank < 0 or requested_rank > primitive_b_matrix.column_count:
        raise ValueError("adaptive target rank is outside the Cartesian dimension")
    if not topology_certified:
        empty = csr_matrix((0, primitive_b_matrix.row_count), dtype=float)
        return AdaptiveDelocalizedBasis(
            primitive_transformation=empty,
            b_matrix=SparseBMatrix(rows=(), column_count=primitive_b_matrix.column_count),
            certificate=AdaptiveDelocalizationCertificate(
                status="SYCART_FALLBACK_UNCERTIFIED_TOPOLOGY",
                metric_mode=settings.metric_mode,
                selected_stage=0,
                selected_maximum_class_rank=None,
                target_rank=requested_rank,
                realized_rank=0,
                sycart_rank=requested_rank,
                selection_condition_number=float("inf"),
                geometric_condition_number=float("inf"),
                block_count=0,
                mixed_class_blocks=(),
                sycart_domains=("GLOBAL",),
            ),
        )

    raw = _as_csr(primitive_b_matrix)
    row_norms = np.sqrt(np.asarray(raw.multiply(raw).sum(axis=1)).reshape(-1))
    if np.any(~np.isfinite(row_norms)) or np.any(row_norms <= settings.rank_tolerance):
        raise ValueError("primitive Wilson matrix contains a zero or non-finite row")
    scale = 1.0 / row_norms
    if settings.metric_mode == METRIC_GEOMETRY_SH:
        if sh_curvatures is None or len(sh_curvatures) != len(assignments):
            raise ValueError("GEOMETRY_SH requires one SH curvature per primitive")
        curvature = np.asarray(sh_curvatures, dtype=float)
        if np.any(~np.isfinite(curvature)) or np.any(curvature <= 0.0):
            raise ValueError("SH curvatures must be finite and positive")
        scale *= np.sqrt(curvature)
    work = raw.multiply(scale[:, None]).tocsr()

    partitions: dict[tuple[str, str, str], list[int]] = {}
    for index, (assignment, irrep) in enumerate(zip(assignments, labels, strict=True)):
        partitions.setdefault((assignment.domain_kind, assignment.domain_id, irrep), []).append(
            index
        )
    certification = (
        None
        if isinstance(topology_certified, bool)
        else {str(key): bool(value) for key, value in topology_certified.items()}
    )
    sycart_domains = tuple(
        sorted(
            {
                partition[1]
                for partition in partitions
                if certification is not None and not certification.get(partition[1], False)
            }
        )
    )
    certified_partitions = {
        partition: indices
        for partition, indices in partitions.items()
        if partition[1] not in sycart_domains
    }
    certified_rows = tuple(index for indices in certified_partitions.values() for index in indices)
    certified_matrix = (
        work[np.asarray(certified_rows, dtype=int)]
        if certified_rows
        else csr_matrix((0, work.shape[1]))
    )
    chemical_target_rank = min(
        requested_rank,
        int(_structural_row_indices(certified_matrix).size),
    )
    if chemical_target_rank == 0:
        empty = csr_matrix((0, primitive_b_matrix.row_count), dtype=float)
        return AdaptiveDelocalizedBasis(
            primitive_transformation=empty,
            b_matrix=SparseBMatrix(rows=(), column_count=primitive_b_matrix.column_count),
            certificate=AdaptiveDelocalizationCertificate(
                status="SYCART_FALLBACK_UNCERTIFIED_TOPOLOGY",
                metric_mode=settings.metric_mode,
                selected_stage=0,
                selected_maximum_class_rank=None,
                target_rank=requested_rank,
                realized_rank=0,
                sycart_rank=requested_rank,
                selection_condition_number=float("inf"),
                geometric_condition_number=float("inf"),
                block_count=0,
                mixed_class_blocks=(),
                sycart_domains=sycart_domains or ("GLOBAL",),
            ),
        )
    transforms = []
    mixed: list[str] = []
    failed_domains: list[str] = []
    block_count = 0
    selected_stage = 0
    selected_cutoffs: list[int] = []
    for partition, indices in sorted(certified_partitions.items()):
        transform, stage, cutoff, blocks, partition_mixed, passed = (
            _least_mixed_partition_transform(
                work,
                tuple(indices),
                assignments,
                partition,
                settings,
            )
        )
        transforms.append(transform)
        mixed.extend(partition_mixed)
        block_count += blocks
        selected_stage = max(selected_stage, stage)
        if cutoff is not None:
            selected_cutoffs.append(cutoff)
        if not passed:
            failed_domains.append(partition[1])
    transform = vstack(transforms, format="csr") if transforms else csr_matrix((0, raw.shape[0]))
    candidate_work = (transform @ work).tocsr()
    selected = _structurally_independent_rows(candidate_work, chemical_target_rank)
    if selected.size != chemical_target_rank:
        raise ValueError("ORACLE primitive domains cannot span the requested rank")
    transform = transform[selected]
    realized_rank, selection_condition = _sparse_condition(
        candidate_work[selected], settings.rank_tolerance
    )
    primitive_transform = transform.multiply(scale[None, :]).tocsr()
    output = (primitive_transform @ raw).tocsr()
    geometric_rank, geometric_condition = _sparse_condition(output, settings.rank_tolerance)
    passed = (
        not failed_domains
        and realized_rank == chemical_target_rank
        and selection_condition <= settings.maximum_condition_number
    )
    status = (
        "PASS_MIXED_SONIC_SYCART"
        if passed and sycart_domains
        else "PASS"
        if passed
        else "SYCART_FALLBACK_CONDITIONING"
    )
    output_matrix = SparseBMatrix.from_csr(
        output.data,
        output.indices,
        output.indptr,
        column_count=output.shape[1],
        row_labels=tuple(f"ADAPTIVE_{index + 1:06d}" for index in range(output.shape[0])),
        backend="matrix-smith-adaptive-delocalized.v1",
    )
    return AdaptiveDelocalizedBasis(
        primitive_transformation=primitive_transform,
        b_matrix=output_matrix,
        certificate=AdaptiveDelocalizationCertificate(
            status=status,
            metric_mode=settings.metric_mode,
            selected_stage=selected_stage,
            selected_maximum_class_rank=max(selected_cutoffs, default=None),
            target_rank=requested_rank,
            realized_rank=geometric_rank,
            sycart_rank=max(0, requested_rank - geometric_rank),
            selection_condition_number=selection_condition,
            geometric_condition_number=geometric_condition,
            block_count=block_count,
            mixed_class_blocks=tuple(mixed),
            sycart_domains=(
                tuple(sorted(set(sycart_domains).union(failed_domains)))
                if passed or failed_domains
                else tuple(sorted({item.domain_id for item in assignments}))
            ),
        ),
    )


def build_adaptive_delocalized_basis_from_definition(
    definition: object,
    coordinates_angstrom: np.ndarray,
    *,
    fragment_by_atom: dict[int, str] | None = None,
    primitive_irreps: Sequence[str] | None = None,
    topology_certified: bool | Mapping[str, bool] = True,
    sh_curvatures: Sequence[float] | None = None,
    policy: AdaptiveDelocalizationPolicy | None = None,
) -> AdaptiveDelocalizedBasis:
    """Run the complete ORACLE-primitive to adaptive-SMITH basis handoff."""

    from matrix_oracle import assign_primitive_classes

    from .evaluation import build_sparse_gic_and_primitive_b_matrices
    from .models import GICDefinition

    if not isinstance(definition, GICDefinition):
        raise TypeError("adaptive SMITH construction requires a frozen GICDefinition")
    _gic_rows, primitive_rows = build_sparse_gic_and_primitive_b_matrices(
        definition,
        coordinates_angstrom=np.asarray(coordinates_angstrom, dtype=float),
    )
    assignments = assign_primitive_classes(
        definition.primitives,
        fragment_by_atom=fragment_by_atom,
    )
    return build_adaptive_delocalized_basis(
        primitive_rows,
        assignments,
        irreps=primitive_irreps,
        topology_certified=topology_certified,
        target_rank=definition.target_rank,
        sh_curvatures=sh_curvatures,
        policy=policy,
    )


__all__ = [
    "ADAPTIVE_DELOCALIZATION_SCHEMA",
    "METRIC_GEOMETRY",
    "METRIC_GEOMETRY_SH",
    "AdaptiveDelocalizationCertificate",
    "AdaptiveDelocalizationPolicy",
    "AdaptiveDelocalizedBasis",
    "build_adaptive_delocalized_basis",
    "build_adaptive_delocalized_basis_from_definition",
]
