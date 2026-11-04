"""Virtual valence spaces from one-electron atomic reference overlaps.

The backend supplies cross-overlap with a compact atomic basis and that basis'
overlap. No auxiliary orbital energies or two-electron integrals are used.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import TYPE_CHECKING

import numpy as np

from .electronic_state import _readonly_array
from .sf_tda import HARTREE_TO_EV

if TYPE_CHECKING:
    from .imported_tda import ImportedTDAReference


def _reference_fingerprint(reference: ImportedTDAReference) -> str:
    digest = hashlib.sha256(reference.model_id.encode("utf-8"))
    for name in ("atomic_numbers", "coordinates_bohr", "ao_atom", "overlap", "fock",
                 "coefficients", "orbital_energies_hartree", "occupations"):
        array = np.ascontiguousarray(getattr(reference, name))
        digest.update(name.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ProjectedTDAVirtualSpace:
    """An orthonormal subspace of a particular reference's complete virtuals.

    ``rotation`` maps compressed virtual coefficients to parent canonical MO
    coefficients. ``fock_hartree`` is the entire projected virtual Fock block;
    its diagonal alone does not represent the one-electron response operator.
    """

    reference_fingerprint: str
    parent_virtual_orbitals: np.ndarray
    rotation: np.ndarray
    fock_hartree: np.ndarray
    singular_values: np.ndarray
    valence_dimension: int
    seed_metric_rank: int
    augmented_canonical_orbitals: np.ndarray
    augmentation_window_ev: float | None
    relative_cutoff: float = 1e-8

    def __post_init__(self) -> None:
        for name in ("rotation", "fock_hartree", "singular_values"):
            object.__setattr__(self, name, _readonly_array(getattr(self, name)))
        for name in ("parent_virtual_orbitals", "augmented_canonical_orbitals"):
            raw = np.asarray(getattr(self, name))
            if not np.all(np.isfinite(raw)) or np.any(raw != np.floor(raw)):
                raise ValueError("virtual orbital indices must be integers")
            object.__setattr__(self, name, _readonly_array(raw, dtype=np.int64))
        parents, rotation = self.parent_virtual_orbitals, self.rotation
        if parents.ndim != 1 or not len(parents) or np.any(parents < 0) or len(np.unique(parents)) != len(parents):
            raise ValueError("parent virtual indices must be nonempty, unique and nonnegative")
        if rotation.ndim != 2 or rotation.shape[0] != len(parents) or not rotation.shape[1]:
            raise ValueError("virtual rotation must have shape (nparent, ncompressed)")
        dimension = rotation.shape[1]
        if not np.allclose(rotation.T @ rotation, np.eye(dimension), atol=2e-8, rtol=0):
            raise ValueError("compressed virtual rotation must have orthonormal columns")
        if self.fock_hartree.shape != (dimension, dimension) or not np.allclose(
                self.fock_hartree, self.fock_hartree.T, atol=1e-10, rtol=0):
            raise ValueError("projected virtual Fock must be a symmetric square matrix")
        if not 1 <= self.valence_dimension <= dimension:
            raise ValueError("valence dimension must lie within the retained dimension")
        if self.singular_values.ndim != 1 or len(self.singular_values) < self.valence_dimension:
            raise ValueError("projection singular values must cover the valence space")
        if self.seed_metric_rank < self.valence_dimension:
            raise ValueError("seed metric rank is smaller than the requested valence space")
        if not np.isfinite(self.relative_cutoff) or not 0 < self.relative_cutoff < 1:
            raise ValueError("relative_cutoff must lie strictly between zero and one")
        if self.augmentation_window_ev is not None and (
                not np.isfinite(self.augmentation_window_ev) or self.augmentation_window_ev <= 0):
            raise ValueError("augmentation window must be positive and finite")

    @property
    def dimension(self) -> int:
        return self.rotation.shape[1]

    def validate_reference(self, reference: ImportedTDAReference) -> None:
        if _reference_fingerprint(reference) != self.reference_fingerprint:
            raise ValueError("projected virtual space belongs to a different reference")
        virtual = np.flatnonzero(reference.occupations < 1)
        if not np.array_equal(virtual, self.parent_virtual_orbitals):
            raise ValueError("projected space must refer to the complete parent virtual space")
        expected = self.rotation.T @ (
            reference.orbital_energies_hartree[virtual, None] * self.rotation)
        if not np.allclose(expected, self.fock_hartree, atol=2e-8, rtol=0):
            raise ValueError("virtual Fock block does not equal the projected reference Fock")

    def expand_amplitudes(self, amplitudes: np.ndarray, occupied_count: int) -> np.ndarray:
        """Return (nroot, noccupied, nparent_virtual) amplitudes for root tracking."""
        array = np.asarray(amplitudes, dtype=float)
        if not isinstance(occupied_count, (int, np.integer)) or occupied_count < 1:
            raise ValueError("occupied_count must be a positive integer")
        if array.ndim != 2 or array.shape[1] != occupied_count * self.dimension or not np.all(np.isfinite(array)):
            raise ValueError("compressed amplitudes have inconsistent dimensions")
        return np.einsum("rik,ak->ria", array.reshape(len(array), occupied_count, self.dimension),
                         self.rotation)


def build_projected_tda_virtual_space(
    reference: ImportedTDAReference, cross_overlap: np.ndarray, seed_overlap: np.ndarray,
    *, virtual_count: int, relative_cutoff: float = 1e-8,
    augment_virtual_window_ev: float | None = None,
) -> ProjectedTDAVirtualSpace:
    """Select leading virtual projections of an orthonormal compact atomic basis.

    For a complete minimal atomic basis (including core), a conventional valence
    count is nminimal - noccupied. The caller supplies it explicitly: projecting
    imperfect atomic core/occupied functions can produce small nonzero singular
    values, so the numerical projection rank alone is not a valence count.

    ``cross_overlap`` is <original AO|compact AO> (nao, nseed), ``seed_overlap``
    is <compact AO|compact AO>. Optional low-energy canonical virtuals enrich
    the valence space by taking an orthonormal union, without double counting.
    """
    if isinstance(virtual_count, bool) or not isinstance(virtual_count, (int, np.integer)) or virtual_count < 1:
        raise ValueError("virtual_count must be an explicit positive integer")
    if not np.isfinite(relative_cutoff) or not 0 < relative_cutoff < 1:
        raise ValueError("relative_cutoff must lie strictly between zero and one")
    if np.iscomplexobj(cross_overlap) or np.iscomplexobj(seed_overlap):
        raise ValueError("compact basis overlaps must be real")
    cross, metric = np.asarray(cross_overlap, dtype=float), np.asarray(seed_overlap, dtype=float)
    if metric.ndim != 2 or not len(metric) or metric.shape != (len(metric), len(metric)):
        raise ValueError("compact basis overlap must be a nonempty square matrix")
    if cross.shape != (len(reference.overlap), len(metric)):
        raise ValueError("cross overlap must have shape (nao, nseed)")
    if not np.all(np.isfinite(cross)) or not np.all(np.isfinite(metric)) or not np.allclose(
            metric, metric.T, atol=1e-10, rtol=0):
        raise ValueError("compact overlaps must be finite and the seed metric symmetric")
    values, vectors = np.linalg.eigh(metric)
    if values[-1] <= 0 or values[0] < -1e-10 * values[-1]:
        raise ValueError("compact basis overlap must be positive semidefinite")
    keep = values > relative_cutoff * values[-1]
    orthogonalizer = vectors[:, keep] / np.sqrt(values[keep])
    parent = np.flatnonzero(reference.occupations < 1)
    projection = reference.coefficients[:, parent].T @ cross @ orthogonalizer
    u, singular, _ = np.linalg.svd(projection, full_matrices=False)
    if len(singular) == 0 or singular[0] <= 0 or singular[0] > 1 + 2e-7:
        raise ValueError("compact overlaps are inconsistent with normalized parent orbitals")
    rank = int(np.count_nonzero(singular > relative_cutoff * singular[0]))
    if virtual_count > rank:
        raise ValueError("requested valence count exceeds the resolved virtual projection rank")
    rotation = u[:, :virtual_count]
    added = np.array([], dtype=int)
    if augment_virtual_window_ev is not None:
        if not np.isfinite(augment_virtual_window_ev) or augment_virtual_window_ev <= 0:
            raise ValueError("augmentation window must be positive and finite")
        homo = np.max(reference.orbital_energies_hartree[reference.occupations > 1])
        added_rows = np.flatnonzero(reference.orbital_energies_hartree[parent]
                                   <= homo + augment_virtual_window_ev / HARTREE_TO_EV)
        added = parent[added_rows]
        candidates = np.eye(len(parent))[:, added_rows]
        # Two projections suppress numerical occupied-valence contamination.
        for _ in range(2):
            candidates -= rotation @ (rotation.T @ candidates)
        extra, strengths, _ = np.linalg.svd(candidates, full_matrices=False)
        extra = extra[:, strengths > relative_cutoff]
        rotation = np.column_stack((rotation, extra))
    fock = rotation.T @ (reference.orbital_energies_hartree[parent, None] * rotation)
    return ProjectedTDAVirtualSpace(
        _reference_fingerprint(reference), parent, rotation, (fock + fock.T) / 2,
        singular, virtual_count, int(np.count_nonzero(keep)), added, augment_virtual_window_ev, relative_cutoff)
