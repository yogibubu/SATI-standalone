"""Geometry-independent analytic SALC catalogs shared by SMITH builders."""

from __future__ import annotations

import numpy as np


def cyclic_out_of_plane_source_terms(
    ring_atoms: tuple[int, ...],
) -> tuple[tuple[tuple[float, tuple[int, int, int, int]], ...], ...]:
    """Return an automorphism-closed local-U source at every ring vertex.

    A single oriented four-atom stencil is closed under cyclic rotations but
    not under reflections of the ring. Its normalized difference from the
    reverse-oriented stencil makes every vertex source transform into another
    source in the same block and retains the crown mode of even rings. The
    construction is topology-only and valid for every ring size for which
    local out-of-plane coordinates are defined.
    """

    size = len(ring_atoms)
    if size <= 3:
        return ()
    weight = float(2.0**-0.5)
    sources = []
    for index, center in enumerate(ring_atoms):
        forward = (
            center,
            ring_atoms[(index - 1) % size],
            ring_atoms[(index + 2) % size],
            ring_atoms[(index + 1) % size],
        )
        reverse = (
            center,
            ring_atoms[(index + 1) % size],
            ring_atoms[(index - 2) % size],
            ring_atoms[(index - 1) % size],
        )
        if reverse == forward:
            sources.append(((1.0, forward),))
        else:
            sources.append(((weight, forward), (-weight, reverse)))
    return tuple(sources)


def cyclic_out_of_plane_atom_orders(
    ring_atoms: tuple[int, ...],
) -> tuple[tuple[int, int, int, int], ...]:
    """Return every native-U order used by the closed cyclic source block."""

    return tuple(
        atoms
        for source in cyclic_out_of_plane_source_terms(ring_atoms)
        for _coefficient, atoms in source
    )


def cyclic_out_of_plane_coefficients(ncycle: int) -> np.ndarray:
    """Return the canonical ``ncycle - 3`` cyclic puckering SALCs.

    Translation of the local ring plane and its two rigid tilts occupy the
    Fourier wavenumbers zero and one.  The remaining real Fourier modes form
    a topology-only orthonormal basis for local out-of-plane deformation.
    """

    size = int(ncycle)
    coefficients = np.zeros((size, max(0, size - 3)), dtype=float)
    if size <= 3:
        return coefficients
    atom_indices = np.arange(size, dtype=float)
    columns: list[np.ndarray] = []
    for wavenumber in range(2, (size - 1) // 2 + 1):
        phase = 2.0 * np.pi * float(wavenumber) * atom_indices / float(size)
        columns.append(np.sqrt(2.0 / float(size)) * np.cos(phase))
        columns.append(np.sqrt(2.0 / float(size)) * np.sin(phase))
    if size % 2 == 0:
        columns.append(((-1.0) ** atom_indices) / np.sqrt(float(size)))
    coefficients[:, :] = np.column_stack(columns)
    coefficients[np.abs(coefficients) < 1.0e-14] = 0.0
    return coefficients


__all__ = [
    "cyclic_out_of_plane_atom_orders",
    "cyclic_out_of_plane_coefficients",
    "cyclic_out_of_plane_source_terms",
]
