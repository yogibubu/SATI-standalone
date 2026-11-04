"""Exact bordered direct-sum certificates for sparse Wilson matrices.

The certificate is structural: connected components of the bipartite graph
between non-separator Wilson rows and Cartesian columns are precisely the
diagonal leaf blocks, while declared separator rows form an explicit border.
No floating-point orthogonality or numerical threshold is inferred here; the
canonical entries already stored in :class:`SparseBMatrix` are authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import struct

from .bmatrix import SparseBMatrix


SPARSE_B_DIRECT_SUM_CERTIFICATE_SCHEMA = "matrix.smith.sparse_b_direct_sum.v1"
SPARSE_B_DIRECT_SUM_CONSTRUCTION = "EXACT_BORDERED_BIPARTITE_SUPPORT_COMPONENTS"
SPARSE_B_DIRECT_SUM_EXACTNESS_SCOPE = "REPRESENTED_CANONICAL_SPARSE_OPERATOR"
SPARSE_B_AUTOMATIC_PARTITION_MINIMUM_ROWS = 512
SPARSE_B_AUTOMATIC_PARTITION_ATOMS = 64


class SparseBDirectSumCertificateError(ValueError):
    """Raised when a certificate does not match its sparse Wilson matrix."""


@dataclass(frozen=True)
class SparseBDirectSumComponent:
    """One row/Cartesian-column component of the certified direct sum."""

    row_indices: tuple[int, ...]
    column_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        rows = tuple(int(index) for index in self.row_indices)
        columns = tuple(int(index) for index in self.column_indices)
        if not rows:
            raise ValueError("a sparse-B direct-sum component must contain a row")
        if rows != tuple(sorted(set(rows))):
            raise ValueError("direct-sum component row indices must be unique and sorted")
        if columns != tuple(sorted(set(columns))):
            raise ValueError("direct-sum component column indices must be unique and sorted")
        object.__setattr__(self, "row_indices", rows)
        object.__setattr__(self, "column_indices", columns)


@dataclass(frozen=True)
class SparseBDirectSumCertificate:
    """Immutable proof record for an exact sparse-B direct-sum partition."""

    matrix_identity_sha256: str
    row_count: int
    column_count: int
    nnz: int
    components: tuple[SparseBDirectSumComponent, ...]
    separator_row_indices: tuple[int, ...]
    isolated_column_indices: tuple[int, ...]
    separator_only_column_indices: tuple[int, ...]
    schema: str = SPARSE_B_DIRECT_SUM_CERTIFICATE_SCHEMA
    construction: str = SPARSE_B_DIRECT_SUM_CONSTRUCTION
    exactness_scope: str = SPARSE_B_DIRECT_SUM_EXACTNESS_SCOPE

    def __post_init__(self) -> None:
        separator = tuple(int(index) for index in self.separator_row_indices)
        if separator != tuple(sorted(set(separator))):
            raise ValueError("certificate separator rows must be unique and sorted")
        object.__setattr__(self, "separator_row_indices", separator)

    @property
    def component_count(self) -> int:
        return len(self.components)

    @property
    def is_nontrivial(self) -> bool:
        """Whether blockwise algebra avoids the single global row block."""

        return self.component_count > 1 or bool(
            self.components and self.separator_row_indices
        )

    @property
    def separator_row_count(self) -> int:
        return len(self.separator_row_indices)

    @property
    def largest_row_block(self) -> int:
        sizes = [len(component.row_indices) for component in self.components]
        if self.separator_row_indices:
            sizes.append(len(self.separator_row_indices))
        return max(sizes, default=0)

    @property
    def largest_column_block(self) -> int:
        return max((len(component.column_indices) for component in self.components), default=0)


def sparse_b_matrix_identity_sha256(matrix: SparseBMatrix) -> str:
    """Return a canonical identity for dimensions and stored IEEE-754 entries."""

    if not isinstance(matrix, SparseBMatrix):
        raise TypeError("sparse-B identity requires a SparseBMatrix")
    digest = hashlib.sha256()
    digest.update(SPARSE_B_DIRECT_SUM_CERTIFICATE_SCHEMA.encode("ascii"))
    digest.update(struct.pack(">QQ", matrix.row_count, matrix.column_count))
    data, indices, indptr = matrix.csr_arrays()
    # Preserve the certificate byte schema exactly while packing complete CSR
    # rows in vectorized native NumPy code instead of looping over nonzeros.
    import numpy as np

    entry_dtype = np.dtype([("column", ">u8"), ("value", ">f8")])
    for row_index in range(matrix.row_count):
        start = int(indptr[row_index])
        stop = int(indptr[row_index + 1])
        digest.update(struct.pack(">QQ", row_index, stop - start))
        if stop == start:
            continue
        packed = np.empty(stop - start, dtype=entry_dtype)
        packed["column"] = indices[start:stop]
        packed["value"] = data[start:stop]
        digest.update(packed.tobytes())
    return digest.hexdigest()


def certify_sparse_b_direct_sum(
    matrix: SparseBMatrix,
    *,
    separator_row_indices: tuple[int, ...] = (),
) -> SparseBDirectSumCertificate:
    """Certify an exact support-connected bordered direct sum of ``matrix``.

    Each stored nonzero in a non-separator row is an edge between its Wilson
    row and Cartesian column.  Different leaf components have disjoint
    supports, while separator rows may couple any leaves.  Consequently the
    row Gram matrix has an exact block-arrowhead form.
    """

    if not isinstance(matrix, SparseBMatrix):
        raise TypeError("direct-sum certification requires a SparseBMatrix")
    separator = tuple(int(index) for index in separator_row_indices)
    if separator != tuple(sorted(set(separator))):
        raise ValueError("separator row indices must be unique and sorted")
    if any(index < 0 or index >= matrix.row_count for index in separator):
        raise ValueError("separator row is outside the sparse Wilson matrix")
    if matrix.row_count >= SPARSE_B_AUTOMATIC_PARTITION_MINIMUM_ROWS:
        separator = tuple(
            sorted(
                set(separator)
                | set(
                    _locality_partition_separator_rows(
                        matrix,
                        atoms_per_partition=SPARSE_B_AUTOMATIC_PARTITION_ATOMS,
                    )
                )
            )
        )
    # The constructor below is itself the canonical certification algorithm:
    # it derives every component and the numeric matrix identity directly
    # from this immutable matrix.  Recomputing the same canonical record here
    # used to traverse and hash the complete Wilson operator a second time;
    # the consumer still verifies the certificate before using it.
    return _uncertified_canonical_certificate(matrix, separator)


def _locality_partition_separator_rows(
    matrix: SparseBMatrix,
    *,
    atoms_per_partition: int,
) -> tuple[int, ...]:
    """Return exact border rows crossing deterministic Cartesian atom tiles."""

    width = int(atoms_per_partition)
    if width < 1:
        raise ValueError("atoms_per_partition must be positive")
    separators = []
    for row_index, entries in enumerate(matrix.iter_csr_entries()):
        partitions = {int(column) // (3 * width) for column, _value in entries}
        if len(partitions) > 1:
            separators.append(row_index)
    return tuple(separators)


def verify_sparse_b_direct_sum(
    matrix: SparseBMatrix,
    certificate: SparseBDirectSumCertificate,
) -> None:
    """Fail closed unless ``certificate`` is the canonical proof for ``matrix``."""

    if not isinstance(matrix, SparseBMatrix):
        raise TypeError("direct-sum verification requires a SparseBMatrix")
    if not isinstance(certificate, SparseBDirectSumCertificate):
        raise TypeError("direct-sum verification requires a certificate")
    if certificate.schema != SPARSE_B_DIRECT_SUM_CERTIFICATE_SCHEMA:
        raise SparseBDirectSumCertificateError("unsupported sparse-B certificate schema")
    if certificate.construction != SPARSE_B_DIRECT_SUM_CONSTRUCTION:
        raise SparseBDirectSumCertificateError("unsupported sparse-B certificate construction")
    if certificate.exactness_scope != SPARSE_B_DIRECT_SUM_EXACTNESS_SCOPE:
        raise SparseBDirectSumCertificateError("unsupported sparse-B exactness scope")
    if any(index < 0 or index >= matrix.row_count for index in certificate.separator_row_indices):
        raise SparseBDirectSumCertificateError("certificate separator row is out of range")
    expected = _uncertified_canonical_certificate(matrix, certificate.separator_row_indices)
    if certificate != expected:
        raise SparseBDirectSumCertificateError(
            "sparse-B direct-sum certificate does not match the represented matrix"
        )


def recertify_sparse_b_identical_support(
    matrix: SparseBMatrix,
    template: SparseBDirectSumCertificate,
    *,
    reference_indices,
    reference_indptr,
) -> SparseBDirectSumCertificate:
    """Bind a proven partition to new values with exactly identical support.

    Direct-sum exactness depends only on the bipartite support graph.  Equality
    of CSR indices and row pointers is therefore a complete proof that the
    existing components remain canonical; only the numeric matrix identity
    must be renewed.
    """

    import numpy as np

    data, indices, indptr = matrix.csr_arrays()
    del data
    if (
        matrix.row_count != template.row_count
        or matrix.column_count != template.column_count
        or matrix.nnz != template.nnz
        or not np.array_equal(indices, np.asarray(reference_indices, dtype=np.intp))
        or not np.array_equal(indptr, np.asarray(reference_indptr, dtype=np.intp))
    ):
        raise SparseBDirectSumCertificateError(
            "sparse-B support changed and requires a fresh direct-sum certification"
        )
    return replace(
        template,
        matrix_identity_sha256=sparse_b_matrix_identity_sha256(matrix),
    )


def _uncertified_canonical_certificate(
    matrix: SparseBMatrix,
    separator_row_indices: tuple[int, ...],
) -> SparseBDirectSumCertificate:
    """Recompute the canonical record without recursively invoking verification."""

    node_count = matrix.row_count + matrix.column_count
    parent = list(range(node_count))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    offset = matrix.row_count
    separator = set(separator_row_indices)
    entries_by_row = tuple(matrix.iter_csr_entries())
    for row_index, entries in enumerate(entries_by_row):
        if row_index in separator:
            continue
        for column_index, _value in entries:
            union(row_index, offset + column_index)
    rows_by_root: dict[int, list[int]] = {}
    columns_by_root: dict[int, list[int]] = {}
    for row_index in range(matrix.row_count):
        if row_index in separator:
            continue
        rows_by_root.setdefault(find(row_index), []).append(row_index)
    for column_index in range(matrix.column_count):
        columns_by_root.setdefault(find(offset + column_index), []).append(column_index)
    components = tuple(
        sorted(
            (
                SparseBDirectSumComponent(tuple(rows), tuple(columns_by_root.get(root, ())))
                for root, rows in rows_by_root.items()
            ),
            key=lambda component: component.row_indices,
        )
    )
    separator_columns = {
        column for row_index in separator for column, _value in entries_by_row[row_index]
    }
    leaf_columns = {
        column for component in components for column in component.column_indices
    }
    all_stored_columns = {
        column for entries in entries_by_row for column, _value in entries
    }
    isolated = tuple(sorted(set(range(matrix.column_count)) - all_stored_columns))
    separator_only = tuple(sorted(separator_columns - leaf_columns))
    return SparseBDirectSumCertificate(
        matrix_identity_sha256=sparse_b_matrix_identity_sha256(matrix),
        row_count=matrix.row_count,
        column_count=matrix.column_count,
        nnz=matrix.nnz,
        components=components,
        separator_row_indices=separator_row_indices,
        isolated_column_indices=isolated,
        separator_only_column_indices=separator_only,
    )


__all__ = [
    "SPARSE_B_DIRECT_SUM_CERTIFICATE_SCHEMA",
    "SPARSE_B_DIRECT_SUM_CONSTRUCTION",
    "SPARSE_B_DIRECT_SUM_EXACTNESS_SCOPE",
    "SPARSE_B_AUTOMATIC_PARTITION_ATOMS",
    "SPARSE_B_AUTOMATIC_PARTITION_MINIMUM_ROWS",
    "SparseBDirectSumCertificate",
    "SparseBDirectSumCertificateError",
    "SparseBDirectSumComponent",
    "certify_sparse_b_direct_sum",
    "recertify_sparse_b_identical_support",
    "sparse_b_matrix_identity_sha256",
    "verify_sparse_b_direct_sum",
]
