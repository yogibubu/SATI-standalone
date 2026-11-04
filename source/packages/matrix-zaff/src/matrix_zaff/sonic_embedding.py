"""Sparse local-before-symmetry SONIC charts and frozen-coordinate constraints.

SMITH coefficients are applied to *natural* local variables here. The new
coordinates are not the old linear combinations of physical distances.
Only bounded independent coefficient blocks are inverted; no Cartesian B
matrix, global dense inverse, or claim of geometric realizability is made.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


class SparseSonicEmbedding:
    """u = u0 + L_active z, with S L = I for the complete declared chart.

    Inactive rows of S are frozen, including for nonorthogonal S. Directions
    outside the row space are also frozen. Every connected coefficient block
    is certified nonsingular before any inverse is constructed. Blocks above
    the declared bound are rejected, never sent to a global dense fallback.
    """

    def __init__(self, analysis, *, local_labels, labels, reference_natural,
                 active_labels=None, maximum_block_size=32, maximum_condition=1e8):
        if not sparse.issparse(analysis):
            raise TypeError("SONIC analysis must be sparse")
        self.analysis = sparse.csr_matrix(analysis, dtype=float, copy=True)
        self.analysis.sum_duplicates()
        self.analysis.eliminate_zeros()
        self.local_labels, self.labels = tuple(local_labels), tuple(labels)
        nrows, nlocal = self.analysis.shape
        if (nrows != len(self.labels) or nlocal != len(self.local_labels)
                or len(set(self.labels)) != nrows or len(set(self.local_labels)) != nlocal):
            raise ValueError("SONIC coefficient labels are inconsistent or duplicated")
        if not np.all(np.isfinite(self.analysis.data)):
            raise ValueError("SONIC coefficients must be finite")
        if (int(maximum_block_size) != maximum_block_size or maximum_block_size < 1
                or not np.isfinite(maximum_condition) or maximum_condition < 1):
            raise ValueError("invalid bounded SONIC inverse policy")
        self.reference = self._vector(reference_natural, nlocal).copy()
        active = self.labels if active_labels is None else tuple(active_labels)
        label_index = {label: i for i, label in enumerate(self.labels)}
        if len(set(active)) != len(active) or any(label not in label_index for label in active):
            raise ValueError("unknown or duplicated active SONIC label")
        self.active_labels = active
        active_indices = np.asarray([label_index[label] for label in active], dtype=int)

        # Bipartite row/column graph has O(nnz) storage even if a column
        # touches many rows. Do not construct S S.T before bounding its blocks.
        incidence = self.analysis.copy()
        incidence.data[:] = 1.0
        graph = sparse.bmat([[None, incidence], [incidence.T, None]], format="csr")
        _, groups = connected_components(graph, directed=False)
        row_groups = {}
        for row, group in enumerate(groups[:nrows]):
            row_groups.setdefault(int(group), []).append(row)
        inverse_rows, inverse_columns, inverse_values = [], [], []
        worst_condition = 1.0
        for rows in row_groups.values():
            if len(rows) > maximum_block_size:
                raise ValueError("SONIC coefficient block exceeds the bounded inverse policy")
            block = self.analysis[rows]
            gram = (block @ block.T).toarray()
            eigenvalues = np.linalg.eigvalsh(gram)
            if eigenvalues[0] <= 0 or eigenvalues[-1] / eigenvalues[0] > maximum_condition**2:
                raise ValueError("SONIC coefficient block is rank deficient or ill conditioned")
            worst_condition = max(worst_condition, float(np.sqrt(eigenvalues[-1] / eigenvalues[0])))
            inverse = np.linalg.solve(gram, np.eye(len(rows)))
            if not np.allclose(gram @ inverse, np.eye(len(rows)), rtol=0, atol=1e-9):
                raise ValueError("SONIC coefficient inverse failed its residual check")
            left, right = np.nonzero(inverse)
            inverse_rows.extend(np.asarray(rows)[left])
            inverse_columns.extend(np.asarray(rows)[right])
            inverse_values.extend(inverse[left, right])
        inverse_gram = sparse.csr_matrix(
            (inverse_values, (inverse_rows, inverse_columns)), shape=(nrows, nrows))
        self.synthesis = (self.analysis.T @ inverse_gram[:, active_indices]).tocsr()
        self.synthesis.eliminate_zeros()
        self.reference.setflags(write=False)
        for matrix in (self.analysis, self.synthesis):
            for buffer in (matrix.data, matrix.indices, matrix.indptr):
                buffer.setflags(write=False)
        self.audit = {
            "coefficient_space": "LOCAL_NATURAL_BEFORE_SYMMETRY",
            "block_count": len(row_groups),
            "maximum_block_size": max(map(len, row_groups.values()), default=0),
            "maximum_coefficient_condition": worst_condition,
            "synthesis_nnz": self.synthesis.nnz,
            "frozen_coordinate_count": nrows - len(active),
            "global_dense_matrices": 0,
        }

    @classmethod
    def from_smith(cls, definition, *, local_labels, reference_natural,
                   active_labels=None, **policy):
        """Import explicit SMITH coefficients; do not infer a new topology.

        The caller selects retained symmetry species through active_labels.
        Other declared coordinates remain fixed. No raw Cartesian atom
        constraint is inferred from this coordinate-space constraint.
        """
        local_labels = tuple(local_labels)
        local_index = {label: i for i, label in enumerate(local_labels)}
        known = {primitive.identifier for primitive in definition.primitives}
        if len(local_index) != len(local_labels) or not set(local_labels) <= known:
            raise ValueError("local SONIC labels do not match the typed SMITH primitives")
        rows, columns, data = [], [], []
        for row, gic in enumerate(definition.gics):
            for label, coefficient in gic.coefficients or ((gic.primitive_id, 1.),):
                if label not in local_index:
                    raise ValueError(f"missing local SONIC component {label}")
                rows.append(row)
                columns.append(local_index[label])
                data.append(float(coefficient))
        return cls(sparse.csr_matrix((data, (rows, columns)),
                                    shape=(len(definition.gics), len(local_labels))),
                   local_labels=local_labels, labels=[g.identifier for g in definition.gics],
                   reference_natural=reference_natural, active_labels=active_labels, **policy)

    @staticmethod
    def _vector(values, count):
        result = np.asarray(values, dtype=float)
        if result.shape != (count,) or not np.all(np.isfinite(result)):
            raise ValueError("invalid SONIC embedding vector")
        return result

    def expand(self, coordinates):
        return self.reference + self.push(coordinates)

    def push(self, direction):
        return np.asarray(self.synthesis @ self._vector(direction, len(self.active_labels)))

    def pull(self, covector):
        return np.asarray(self.synthesis.T @ self._vector(covector, len(self.local_labels)))

    def evaluate(self, runtime, coordinates, *, direction=None):
        """Pull back direct local bonded E/G/Hv to active symmetry coordinates."""
        value = runtime.evaluate(self.expand(coordinates),
                                 direction=None if direction is None else self.push(direction))
        return replace(value, gradient_natural=self.pull(value.gradient_natural),
                       hessian_vector_natural=(None if value.hessian_vector_natural is None
                                               else self.pull(value.hessian_vector_natural)),
                       execution={**value.execution, **self.audit,
                                  "coordinate_space": "ACTIVE_NATURAL_SONIC"})
