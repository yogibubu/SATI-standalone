"""Coordinate charts and discrete frame transport for supplied subspaces.

These operations preserve a supplied subspace. They do not solve a Hamiltonian
or repair a nonunique limit of the collective Gram approximation. Cluster
membership is explicit; no energy threshold or state sorting is performed.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import qr


def _readonly(value):
    value = np.array(value, dtype=complex, copy=True)
    value.setflags(write=False)
    return value


def _compatible(left, right):
    if (left.coordinate_id != right.coordinate_id
            or left.columns.shape[0] != right.columns.shape[0]):
        raise ValueError("frames must share the same ambient coordinate basis")


def _reduced_matrix(value, size):
    value = np.asarray(value, dtype=complex)
    if value.ndim < 2 or value.shape[-2:] != (size, size):
        raise ValueError("reduced operator dimensions do not match the frame")
    if not np.all(np.isfinite(value)):
        raise ValueError("finite reduced operator required")
    return value


@dataclass(frozen=True)
class SubspaceChart:
    """Graph plus anchor retaining the original column gauge exactly."""
    selected_rows: tuple
    remaining_rows: tuple
    graph: np.ndarray
    anchor: np.ndarray
    coordinate_id: str
    anchor_condition: float
    changed_from_preferred: bool

    def to_frame(self):
        rank = len(self.selected_rows)
        size = rank + len(self.remaining_rows)
        indices = self.selected_rows + self.remaining_rows
        if (sorted(indices) != list(range(size))
                or self.anchor.shape != (rank, rank)
                or self.graph.shape != (size - rank, rank)):
            raise ValueError("invalid chart dimensions or row permutation")
        columns = np.empty((size, rank), dtype=complex)
        columns[list(self.selected_rows)] = self.anchor
        columns[list(self.remaining_rows)] = self.graph @ self.anchor
        return CollectiveSubspaceFrame(columns, self.coordinate_id)


@dataclass(frozen=True)
class FrameTransport:
    """Discrete polar alignment, not an exact finite-step parallel propagator.

    frame.columns = original_current.columns @ basis_change.
    Transform reduced energies, properties and populations by R† K R as well.
    A dense change within nondegenerate levels is not just a phase convention.
    """
    frame: object
    basis_change: np.ndarray
    overlap_singular_values: np.ndarray
    alignment_distance: float

    def transform_reduced_operator(self, operator):
        value = _reduced_matrix(operator, self.basis_change.shape[0])
        return self.basis_change.conj().T @ value @ self.basis_change


@dataclass(frozen=True)
class CollectiveSubspaceFrame:
    """Orthonormal columns in one fixed, explicitly identified ambient basis."""
    columns: np.ndarray
    coordinate_id: str

    def __post_init__(self):
        columns = np.asarray(self.columns, dtype=complex)
        if (columns.ndim != 2 or min(columns.shape) < 1
                or columns.shape[1] > columns.shape[0]
                or not np.all(np.isfinite(columns))):
            raise ValueError("finite nonempty tall frame required")
        if not isinstance(self.coordinate_id, str) or not self.coordinate_id.strip():
            raise ValueError("nonempty coordinate_id required")
        rank = columns.shape[1]
        defect = np.linalg.norm(columns.conj().T @ columns - np.eye(rank), "fro")
        tolerance = 256 * np.finfo(float).eps * max(columns.shape) * max(1, rank)
        if defect > tolerance:
            raise ValueError("orthonormal columns required; no implicit orthogonalization is performed")
        object.__setattr__(self, "columns", _readonly(columns))

    @classmethod
    def from_collective_result(cls, result, *, block, coordinate_id):
        """Extract a declared P or Q cluster without re-solving its Gram graph."""
        q, p = result.graph.shape
        if result.unitary.shape != (p + q, p + q) or block not in ("p", "q"):
            raise ValueError("valid collective result and block='p' or 'q' required")
        columns = result.unitary[:, :p] if block == "p" else result.unitary[:, p:]
        return cls(columns, coordinate_id)

    def chart(self, *, preferred_rows=None):
        """Use a nonsingular row chart; pivoted QR supplies an automatic fallback.

        The frame itself never changes. Row changes are coordinate changes only.
        Numerical rank checks use floating-point scale, not molecular energies.
        """
        columns = self.columns
        size, rank = columns.shape
        eps = np.finfo(float).eps

        def usable(rows):
            singular = np.linalg.svd(columns[list(rows)], compute_uv=False)
            return singular[-1] > 64 * eps * max(size, rank) * singular[0]

        preferred = None
        if preferred_rows is not None:
            preferred = tuple(preferred_rows)
            if (len(preferred) != rank or len(set(preferred)) != rank
                    or any(isinstance(i, (bool, np.bool_))
                           or not isinstance(i, (int, np.integer))
                           or i < 0 or i >= size for i in preferred)):
                raise ValueError("preferred_rows must contain rank distinct ambient row indices")
        if preferred is not None and usable(preferred):
            selected = preferred
        else:
            _, _, pivots = qr(columns.T, mode="economic", pivoting=True)
            selected = tuple(int(i) for i in pivots[:rank])
            if not usable(selected):
                raise FloatingPointError("no numerically resolved row chart; increase precision")
        selected_set = set(selected)
        remaining = tuple(i for i in range(size) if i not in selected_set)
        anchor = columns[list(selected)]
        graph = np.linalg.solve(anchor.T, columns[list(remaining)].T).T
        return SubspaceChart(selected, remaining, _readonly(graph), _readonly(anchor),
                             self.coordinate_id, float(np.linalg.cond(anchor)),
                             preferred is not None and selected != preferred)

    def align_to(self, previous):
        """Align the current declared cluster to the previous one by polar SVD.

        This is a discrete nearest-frame gauge. It requires equal ranks and a
        nonsingular overlap. An orthogonal step needs intermediate path data;
        the routine raises rather than selecting an arbitrary polar factor.
        """
        _compatible(self, previous)
        if self.columns.shape != previous.columns.shape:
            raise ValueError("constant cluster rank required for frame transport")
        overlap = previous.columns.conj().T @ self.columns
        left, singular, right_h = np.linalg.svd(overlap, full_matrices=False)
        threshold = 64 * np.finfo(float).eps * max(self.columns.shape)
        if singular[-1] <= threshold:
            raise FloatingPointError("singular cluster overlap; intermediate path frames required")
        rotation = right_h.conj().T @ left.conj().T
        aligned = CollectiveSubspaceFrame(self.columns @ rotation, self.coordinate_id)
        singular = singular.copy()
        singular.setflags(write=False)
        return FrameTransport(aligned, _readonly(rotation), singular,
                              float(np.linalg.norm(aligned.columns - previous.columns, "fro")))

    def transition_amplitudes(self, final, operator):
        """Return final† O initial for scalar or stacked property components."""
        _compatible(self, final)
        value = _reduced_matrix(operator, self.columns.shape[0])
        return final.columns.conj().T @ value @ self.columns

    def transition_strength(self, final, operator, *, initial_density=None):
        """Cluster-summed strengths; leading property-component axes are retained.

        Without density, sum over initial columns (not a degeneracy average).
        A supplied positive density/population matrix need not have unit trace.
        No thermal factors or photon factors are inferred.
        """
        amplitudes = self.transition_amplitudes(final, operator)
        if initial_density is None:
            return np.sum(np.abs(amplitudes) ** 2, axis=(-2, -1))
        rank = self.columns.shape[1]
        density = _reduced_matrix(initial_density, rank)
        if density.ndim != 2:
            raise ValueError("one initial density matrix required")
        tolerance = 256 * np.finfo(float).eps * rank * np.linalg.norm(density, "fro")
        if np.linalg.norm(density - density.conj().T, "fro") > tolerance:
            raise ValueError("initial density must be Hermitian")
        density = (density + density.conj().T) / 2
        if np.min(np.linalg.eigvalsh(density)) < -tolerance:
            raise ValueError("initial density must be positive semidefinite")
        strength = np.einsum("...fi,ij,...fj->...", amplitudes, density,
                             amplitudes.conj()).real
        return np.maximum(strength, 0)
