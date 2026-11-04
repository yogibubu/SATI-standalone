"""Sparse/local certification for the MINIMUM conditioning pipeline.

The certificate deliberately proves only support and structural rank.  Dense
SVD remains an opt-in reference audit until condition-number equivalence is
demonstrated for each conditioning policy.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .bmatrix import SparseBMatrix
from .bmatrix import SparseBRow


MINIMUM_SPARSE_CERTIFICATE_SCHEMA = "matrix.smith.minimum_sparse_certificate.v1"


@dataclass(frozen=True)
class MinimumSparseCertificate:
    row_count: int
    column_count: int
    structural_rank: int
    nnz: int
    status: str
    unsupported_rows: tuple[int, ...] = ()
    connected_components: int = 0
    first_unmatchable_rows: tuple[int, ...] = ()
    hall_deficient_rows: tuple[int, ...] = ()
    schema: str = MINIMUM_SPARSE_CERTIFICATE_SCHEMA


@dataclass(frozen=True)
class IncrementalSupportMatcher:
    """Persistent bipartite matcher for a growing set of sparse rows.

    Branches share the immutable prefix state.  Extending a branch searches
    only augmenting paths introduced by its new rows, rather than recomputing
    a maximum matching for the complete chart.
    """

    column_count: int
    row_supports: tuple[tuple[int, ...], ...] = ()
    column_to_row: tuple[int, ...] = ()

    @classmethod
    def empty(cls, column_count: int) -> "IncrementalSupportMatcher":
        count = int(column_count)
        return cls(count, (), (-1,) * count)

    @property
    def rank(self) -> int:
        return len(self.row_supports)

    def payload(self) -> dict[str, object]:
        return {
            "schema": "matrix.smith.incremental_support_matching.v1",
            "column_count": self.column_count,
            "row_supports": [list(row) for row in self.row_supports],
            "column_to_row": list(self.column_to_row),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "IncrementalSupportMatcher":
        if payload.get("schema") != "matrix.smith.incremental_support_matching.v1":
            raise ValueError("unsupported support-matching cache schema")
        result = cls(
            int(payload["column_count"]),
            tuple(tuple(int(column) for column in row) for row in payload["row_supports"]),
            tuple(int(owner) for owner in payload["column_to_row"]),
        )
        if len(result.column_to_row) != result.column_count:
            raise ValueError("cached support matching has an invalid column dimension")
        return result

    def invalidate_columns(self, columns) -> "IncrementalSupportMatcher":
        """Drop rows touching changed Cartesian columns and rebuild unaffected prefixes."""

        changed = {int(column) for column in columns}
        retained = tuple(
            SparseBRow(self.column_count, tuple((column, 1.0) for column in support))
            for support in self.row_supports
            if changed.isdisjoint(support)
        )
        rebuilt = IncrementalSupportMatcher.empty(self.column_count).extend(retained)
        if rebuilt is None:
            raise RuntimeError("unaffected matching rows unexpectedly became dependent")
        return rebuilt

    def extend(self, rows: tuple[SparseBRow, ...]) -> "IncrementalSupportMatcher | None":
        supports = list(self.row_supports)
        matching = list(self.column_to_row)
        new_supports = [tuple(column for column, _value in row.entries) for row in rows]
        if any(not support for support in new_supports):
            return None
        complete_supports = tuple(supports) + tuple(new_supports)
        native_nnz = sum(len(support) for support in complete_supports)
        if len(complete_supports) >= 32 or native_nnz >= 256:
            try:
                from matrix_chem import compiled_csr_incremental_matching

                indices = np.asarray(
                    [column for support in complete_supports for column in support], dtype=np.intp
                )
                indptr = np.asarray(
                    [0, *np.cumsum([len(support) for support in complete_supports])], dtype=np.intp
                )
                native = compiled_csr_incremental_matching(
                    indices, indptr, matching,
                    column_count=self.column_count, start_row=len(supports),
                )
                if native is None:
                    return None
                return IncrementalSupportMatcher(
                    self.column_count, complete_supports, tuple(int(value) for value in native)
                )
            except (ImportError, RuntimeError, AttributeError):
                pass
        for sparse_row in rows:
            support = tuple(column for column, _value in sparse_row.entries)
            if not support:
                return None
            row_index = len(supports)
            supports.append(support)

            def augment(candidate: int, seen_rows: set[int], seen_columns: set[int]) -> bool:
                if candidate in seen_rows:
                    return False
                seen_rows.add(candidate)
                for column in supports[candidate]:
                    if column in seen_columns:
                        continue
                    seen_columns.add(column)
                    owner = matching[column]
                    if owner < 0 or augment(owner, seen_rows, seen_columns):
                        matching[column] = candidate
                        return True
                return False

            if not augment(row_index, set(), set()):
                return None
        return IncrementalSupportMatcher(
            self.column_count,
            tuple(supports),
            tuple(matching),
        )


def certify_minimum_sparse_rows(matrix: SparseBMatrix, *, expected_rank: int) -> MinimumSparseCertificate:
    """Fail closed on a missing Wilson support direction without densifying B."""
    if not isinstance(matrix, SparseBMatrix):
        raise TypeError("MINIMUM sparse certificate requires SparseBMatrix")
    matrix.assert_finite()
    data, indices, indptr = matrix.csr_arrays()
    native_certificate = None
    if matrix.row_count >= 32 or matrix.nnz >= 256:
        try:
            from matrix_chem import compiled_csr_support_certificate

            native_certificate = compiled_csr_support_certificate(
                indices, indptr, column_count=matrix.column_count
            )
        except (ImportError, RuntimeError, AttributeError):
            native_certificate = None
    rank = (
        int(native_certificate["rank"])
        if native_certificate is not None
        else matrix.structural_rank()
    )
    expected = int(expected_rank)
    status = "PASS" if rank >= expected and matrix.row_count >= expected else "FAIL"
    unsupported = (
        tuple(native_certificate["unsupported_rows"])
        if native_certificate is not None
        else tuple(
            index
            for index, (start, stop) in enumerate(zip(indptr[:-1], indptr[1:], strict=True))
            if int(start) == int(stop)
        )
    )
    if native_certificate is not None:
        components = int(native_certificate["connected_components"])
    else:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        support = csr_matrix(
            (np.ones(data.size, dtype=np.int8), indices, indptr),
            shape=(matrix.row_count, matrix.column_count),
        )
        gram_pattern = (support @ support.T).astype(bool)
        components = int(connected_components(gram_pattern, directed=False, return_labels=False))
    matcher = IncrementalSupportMatcher.empty(matrix.column_count)
    first_unmatchable: tuple[int, ...] = ()
    if status == "FAIL":
        for row_index, row in enumerate(matrix.rows):
            extended = matcher.extend((row,))
            if extended is None:
                first_unmatchable = tuple(range(row_index + 1))
                break
            matcher = extended
    hall_rows = (
        _polynomial_hall_witness(matrix, first_unmatchable)
        if first_unmatchable
        else ()
    )
    return MinimumSparseCertificate(
        matrix.row_count, matrix.column_count, rank, matrix.nnz, status,
        unsupported_rows=unsupported, connected_components=components,
        first_unmatchable_rows=first_unmatchable,
        hall_deficient_rows=hall_rows,
    )


def _polynomial_hall_witness(
    matrix: SparseBMatrix,
    prefix: tuple[int, ...],
) -> tuple[int, ...]:
    """Return a minimal Hall witness using alternating forests, never subsets.

    A maximum matching orients unmatched row-to-column edges forward and
    matched edges back to their owning row.  The rows reachable from any
    unmatched root form a Hall-deficient set.  We choose the smallest such
    tree and prune redundant rows while preserving ``|N(S)| < |S|``.  The
    runtime is polynomial in the sparse support size.
    """

    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching

    selected = tuple(int(row) for row in prefix)
    supports = tuple(
        tuple(column for column, _value in matrix.rows[row].entries)
        for row in selected
    )
    indptr = np.asarray(
        [0, *np.cumsum([len(support) for support in supports])], dtype=np.intp
    )
    indices = np.asarray(
        [column for support in supports for column in support], dtype=np.intp
    )
    graph = csr_matrix(
        (np.ones(indices.size, dtype=np.int8), indices, indptr),
        shape=(len(selected), matrix.column_count),
    )
    row_to_column = np.asarray(
        maximum_bipartite_matching(graph, perm_type="column"), dtype=np.intp
    )
    column_to_row = np.full(matrix.column_count, -1, dtype=np.intp)
    for row, column in enumerate(row_to_column):
        if column >= 0:
            column_to_row[int(column)] = row

    candidates: list[tuple[int, ...]] = []
    for root in np.flatnonzero(row_to_column < 0):
        reached_rows = {int(root)}
        reached_columns: set[int] = set()
        queue = [int(root)]
        while queue:
            row = queue.pop()
            matched_column = int(row_to_column[row])
            for column in supports[row]:
                if column == matched_column or column in reached_columns:
                    continue
                reached_columns.add(column)
                owner = int(column_to_row[column])
                if owner >= 0 and owner not in reached_rows:
                    reached_rows.add(owner)
                    queue.append(owner)
        if len(reached_columns) < len(reached_rows):
            candidates.append(tuple(sorted(reached_rows)))
    if not candidates:
        return selected
    witness = list(min(candidates, key=lambda rows: (len(rows), rows)))

    def deficient(rows: list[int]) -> bool:
        neighbors = {column for row in rows for column in supports[row]}
        return len(neighbors) < len(rows)

    changed = True
    while changed and len(witness) > 1:
        changed = False
        for row in tuple(witness):
            trial = [item for item in witness if item != row]
            if trial and deficient(trial):
                witness = trial
                changed = True
                break
    return tuple(selected[row] for row in witness)
