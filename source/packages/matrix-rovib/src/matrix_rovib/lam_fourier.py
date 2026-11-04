"""Fourier and hybrid Fourier--DVR primitive spaces for periodic LAMs.

The quadrature and kinetic ordering are exactly those of the supplied surface.
Fourier cutoffs select a variational subspace of that discrete problem; they do
not change the quadrature or silently discard aliased Fourier coefficients.
Translation sectors are constructed before assembling or solving a block.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from matrix_numerics import diagonalize_hermitian


@dataclass(frozen=True)
class FourierLAMSpace:
    """Selected tensor harmonics, with DVR indices on nonperiodic axes."""

    grid_shape: tuple[int, ...]
    periodic_axes: tuple[int, ...]
    primitive_indices: np.ndarray
    sector_labels: np.ndarray
    cutoffs: tuple[int | None, ...]
    wave_numbers: tuple[np.ndarray, ...]

    @classmethod
    def from_surface(cls, surface, cutoffs: Sequence[int | None] | None = None):
        shape = surface.grid_shape
        if not any(surface.periodic):
            raise ValueError("a Fourier LAM space requires a periodic coordinate")
        cutoffs = (None,) * surface.dimension if cutoffs is None else tuple(cutoffs)
        if len(cutoffs) != surface.dimension:
            raise ValueError("provide one Fourier cutoff per LAM coordinate")
        selected, waves, residues = [], [], []
        normalized = []
        for n, q, periodic, order, cutoff in zip(
            shape, surface.coordinates, surface.periodic, surface.symmetry_orders,
            cutoffs, strict=True,
        ):
            modes = np.rint(np.fft.fftfreq(n) * n).astype(int)
            if not periodic:
                if cutoff is not None:
                    raise ValueError("a nonperiodic coordinate cannot have a Fourier cutoff")
                indices = np.arange(n)
                waves.append(np.zeros(n))
            else:
                if cutoff is not None and (
                    isinstance(cutoff, (bool, np.bool_))
                    or not isinstance(cutoff, (int, np.integer))
                    or not 0 <= cutoff <= n // 2
                ):
                    raise ValueError("Fourier cutoffs must be integers between zero and grid_size//2")
                indices = np.arange(n) if cutoff is None else np.flatnonzero(np.abs(modes) <= cutoff)
                waves.append(2 * np.pi * np.fft.fftfreq(n, float(q[1] - q[0])))
            selected.append(indices)
            residues.append(modes[indices] % order if periodic else np.zeros(len(indices), dtype=int))
            normalized.append(None if cutoff is None else int(cutoff))
        index_mesh = np.meshgrid(*selected, indexing="ij")
        indices = np.ravel_multi_index(tuple(item.reshape(-1) for item in index_mesh), shape)
        sector = np.zeros(len(indices), dtype=int)
        for order, residue in zip(surface.symmetry_orders, np.meshgrid(*residues, indexing="ij"), strict=True):
            sector = sector * order + residue.reshape(-1)
        for array in (indices, sector, *waves):
            array.setflags(write=False)
        return cls(shape, tuple(i for i, value in enumerate(surface.periodic) if value),
                   indices, sector, tuple(normalized), tuple(waves))

    @property
    def dimension(self):
        return len(self.primitive_indices)

    def to_grid(self, coefficients, indices=None):
        """Embed selected harmonics and perform unitary inverse FFTs."""
        values = np.asarray(coefficients, dtype=complex)
        vector = values.ndim == 1
        indices = self.primitive_indices if indices is None else np.asarray(indices)
        embedded = np.zeros((math.prod(self.grid_shape), values.reshape(len(indices), -1).shape[1]), complex)
        embedded[indices] = values.reshape(len(indices), -1)
        tensor = embedded.reshape(self.grid_shape + (-1,))
        grid = np.fft.ifftn(tensor, axes=self.periodic_axes, norm="ortho").reshape(embedded.shape)
        return grid[:, 0] if vector else grid

    def from_grid(self, values, indices=None):
        """Project DVR vectors onto selected tensor harmonics."""
        array = np.asarray(values, dtype=complex)
        vector = array.ndim == 1
        tensor = array.reshape(self.grid_shape + (-1,))
        transformed = np.fft.fftn(tensor, axes=self.periodic_axes, norm="ortho").reshape((math.prod(self.grid_shape), -1))
        indices = self.primitive_indices if indices is None else np.asarray(indices)
        projected = transformed[indices]
        return projected[:, 0] if vector else projected


def _validate_translation_fields(surface, potential, *, include_rotation=False):
    """A sector reduction is valid only when every J=0 coefficient commutes."""
    fields = {"potential": potential.reshape(surface.grid_shape), "F_MHz": np.asarray(surface.F_MHz)}
    if surface.lam_quartic_kinetic_MHz is not None:
        fields["lam_quartic_kinetic_MHz"] = np.asarray(surface.lam_quartic_kinetic_MHz)
    if include_rotation:
        for name in ("rho", "R_MHz", "quartic_distortion_MHz", "sextic_distortion_MHz",
                     "p2_rotational_metric_MHz", "p_rotational_cubic_MHz",
                     "p_rotational_anticommutator_MHz"):
            values = getattr(surface, name)
            if values is not None:
                fields[name] = np.asarray(values)
    for name, values in fields.items():
        tolerance = max(1.e-10, 512 * np.finfo(float).eps * float(np.max(np.abs(values), initial=0.)))
        for axis, order in enumerate(surface.symmetry_orders):
            if order > 1 and np.max(np.abs(values - np.roll(values, surface.grid_shape[axis] // order, axis=axis)), initial=0.) > tolerance:
                raise ValueError(f"{name} does not preserve the declared LAM translation symmetry on axis {axis}")


def solve_fourier_lam_basis(surface, *, state_count, potential, cutoffs=None, diagonalizer_backend=None):
    """Solve invariant Fourier blocks; return grid vectors for all consumers.

    Purely periodic blocks use convolution coefficients and diagonal momenta.
    Mixed blocks apply the existing kinetic operator between unitary transforms,
    retaining its boundary flux and mixed metric ordering. Large low-root blocks
    use FFT matvecs rather than allocating dense primitive matrices.
    """
    from .lam_rotation import LAMBasis, _apply_lam_hamiltonian

    space = FourierLAMSpace.from_surface(surface, cutoffs)
    if not 1 <= state_count <= space.dimension:
        raise ValueError("state_count must fit the selected Fourier space")
    _validate_translation_fields(surface, potential)
    size = math.prod(space.grid_shape)
    axes = tuple(range(surface.dimension))
    purely_periodic = len(space.periodic_axes) == surface.dimension
    if purely_periodic:
        vhat = np.fft.fftn(potential.reshape(space.grid_shape)) / size
        fhat = np.fft.fftn(surface.F_MHz, axes=axes) / size
        qhat = (None if surface.lam_quartic_kinetic_MHz is None else
                np.fft.fftn(surface.lam_quartic_kinetic_MHz, axes=axes) / size)
    records = []

    def apply(indices, values):
        grid = space.to_grid(values, indices)
        return space.from_grid(_apply_lam_hamiltonian(surface, grid, potential=potential), indices)

    def dense_block(indices):
        if purely_periodic:
            coordinates = np.stack(np.unravel_index(indices, space.grid_shape), axis=-1)
            difference = tuple((coordinates[:, None, a] - coordinates[None, :, a]) % n
                               for a, n in enumerate(space.grid_shape))
            wave = np.stack([space.wave_numbers[a][coordinates[:, a]] for a in axes], axis=-1)
            block = vhat[difference].copy()
            for mu in axes:
                for nu in axes:
                    block += wave[:, mu, None] * fhat[difference + (mu, nu)] * wave[None, :, nu]
                if qhat is not None:
                    block += wave[:, mu, None]**2 * qhat[difference + (mu,)] * wave[None, :, mu]**2
        else:
            block = np.empty((len(indices), len(indices)), complex)
            # Batched applications avoid a primitive_size squared workspace.
            for start in range(0, len(indices), 32):
                stop = min(start + 32, len(indices))
                columns = np.zeros((len(indices), stop - start))
                columns[np.arange(start, stop), np.arange(stop - start)] = 1.
                block[:, start:stop] = apply(indices, columns)
        return .5 * (block + block.conj().T)

    backend_names = set()
    dense_bytes = 0
    def diagonalize(record, roots):
        nonlocal dense_bytes
        indices = record["indices"]
        n = len(indices)
        iterative = (diagonalizer_backend == "arpack" or
                     (diagonalizer_backend is None and n >= 512 and roots <= n // 4)) and roots < n - 1
        if iterative:
            from scipy.sparse.linalg import LinearOperator, eigsh
            operator = LinearOperator((n, n), matvec=lambda v: apply(indices, v),
                                      matmat=lambda v: apply(indices, v), dtype=np.complex128)
            # A deterministic nonconstant complex seed also covers conjugate sectors.
            start = np.linspace(1., 2., n) + 1j * np.linspace(.3, .7, n)**2
            energies, vectors = eigsh(operator, k=roots, which="SA", tol=1.e-11,
                                     maxiter=max(1000, 20*n), v0=start)
            order = np.argsort(energies, kind="stable")
            energies, vectors = energies[order], vectors[:, order]
            from .lam_rotation import _orthonormalize_degenerate_subspaces
            vectors = _orthonormalize_degenerate_subspaces(energies, vectors)
            from .lam_rotation import _lam_kinetic_norm_bound
            from ._iterative_roots import complete_lowest_ritz_subspace
            energies, vectors = complete_lowest_ritz_subspace(
                lambda values: apply(indices, values), energies, vectors,
                spectral_bound=_lam_kinetic_norm_bound(surface, potential),
            )
            backend_names.add("scipy-arpack")
        else:
            if record.get("matrix") is None:
                record["matrix"] = dense_block(indices)
                dense_bytes += record["matrix"].nbytes
            solved = diagonalize_hermitian(record["matrix"],
                backend=None if diagonalizer_backend == "arpack" else diagonalizer_backend,
                subset_by_index=(0, roots-1) if roots < n else None)
            if solved.eigenvectors is None:
                raise RuntimeError("LOCK Fourier diagonalizer did not return eigenvectors")
            energies, vectors = solved.eigenvalues, solved.eigenvectors
            backend_names.add(solved.backend)
        record["energies"], record["vectors"] = np.asarray(energies).real, vectors

    sectors = np.unique(space.sector_labels)
    for sector in sectors:
        indices = space.primitive_indices[space.sector_labels == sector]
        record = {"indices": indices, "sector": int(sector)}
        roots = min(len(indices), max(2, math.ceil(state_count/len(sectors)) + 2))
        diagonalize(record, roots)
        records.append(record)
    # Every uncomputed root must be above the global cutoff. Grow all sectors
    # touching it, including the complete degeneracy at the retention boundary.
    while True:
        combined = np.sort(np.concatenate([record["energies"] for record in records]))
        cutoff = combined[min(state_count, len(combined))-1]
        tolerance = max(1.e-6, 256 * float(np.spacing(abs(float(cutoff)))))
        grow = [record for record in records if len(record["energies"]) < len(record["indices"])
                and (len(combined) < state_count or record["energies"][-1] <= cutoff + tolerance)]
        if not grow:
            break
        for record in grow:
            diagonalize(record, min(len(record["indices"]), max(len(record["energies"])+1, 2*len(record["energies"]))))
    entries = sorted((float(energy), r, i) for r, record in enumerate(records)
                     for i, energy in enumerate(record["energies"]))
    entries = [entry for entry in entries if entry[0] <= cutoff + tolerance]
    coefficients = np.zeros((space.dimension, len(entries)), complex)
    local_positions = {int(sector): np.flatnonzero(space.sector_labels == sector) for sector in sectors}
    labels = []
    for column, (_, r, i) in enumerate(entries):
        record = records[r]
        coefficients[local_positions[record["sector"]], column] = record["vectors"][:, i]
        labels.append(record["sector"])
    vectors = space.to_grid(coefficients)
    counts = {}
    state_indices = []
    for label in labels:
        state_indices.append(counts.get(label, 0)); counts[label] = counts.get(label, 0) + 1
    diagnostics = {"quadrature_shape": list(space.grid_shape), "selected_primitive_size": space.dimension,
        "sector_dimensions": [len(record["indices"]) for record in records],
        "sector_roots": [len(record["energies"]) for record in records],
        "dense_block_bytes": dense_bytes, "root_cutoff_certified": True}
    return LAMBasis(np.array([entry[0] for entry in entries]), vectors, (), None,
        "+".join(sorted(backend_names)), tuple(labels), tuple(state_indices),
        primitive_representation="fourier" if purely_periodic else "fourier-dvr",
        primitive_size=space.dimension, fourier_cutoffs=space.cutoffs, diagnostics=diagnostics,
        spectral_space=space, spectral_coefficients=coefficients)


__all__ = ["FourierLAMSpace"]
