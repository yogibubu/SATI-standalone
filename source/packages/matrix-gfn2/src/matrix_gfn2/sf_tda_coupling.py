"""Analytic interstate and nonadiabatic couplings for SF-TDA roots.

For a real symmetric TDA operator ``A(R)`` represented in its orthonormal
configuration space, the regular interstate derivative is

``h_ij = X_i.T A[R] X_j``.

Away from a degeneracy, the corresponding adiabatic derivative coupling is

``d_ij = h_ij / (omega_j - omega_i)``.

The pair-packed derivative kernel supplies the complete derivative of the
selected TDA operator, including AO-metric, orbital/SCC, multipolar, and
embedding response.  This module only performs vector--Jacobian products, so
its working set is linear in the nuclear dimension and independent of a dense
``3N * ncsf * ncsf`` derivative tensor.

At an unresolved degeneracy an individual adiabatic derivative coupling is
not defined.  The finite interstate numerator and the derivative matrix in
the chosen root subspace remain well defined and are returned explicitly for
branching-plane and state-intersection algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .sf_tda import SFTDAResult
from .sf_tda_hessian import SFTDADerivativeKernel


def _normalised_root(amplitudes: np.ndarray, index: int) -> np.ndarray:
    root = np.array(amplitudes[index], dtype=float, copy=True)
    norm = float(np.linalg.norm(root))
    if norm <= np.finfo(float).tiny:
        raise ValueError("SF-TDA root amplitude has zero norm")
    root /= norm
    return root


def _fix_root_phase(root: np.ndarray, reference: np.ndarray | None) -> tuple[np.ndarray, float]:
    """Return a phase-aligned real root and its signed reference overlap."""

    value = np.array(root, dtype=float, copy=True)
    if reference is None:
        pivot = int(np.argmax(np.abs(value)))
        sign = -1.0 if value[pivot] < 0.0 else 1.0
        value *= sign
        return value, sign

    target = np.asarray(reference, dtype=float)
    if target.shape != value.shape:
        raise ValueError("reference SF-TDA amplitude has the wrong dimension")
    target_norm = float(np.linalg.norm(target))
    if target_norm <= np.finfo(float).tiny:
        raise ValueError("reference SF-TDA amplitude has zero norm")
    overlap = float(np.dot(value, target / target_norm))
    if overlap < 0.0:
        value *= -1.0
        overlap *= -1.0
    return value, overlap


@dataclass(frozen=True)
class SFTDAInterstateCoupling:
    """One analytic SF-TDA interstate derivative-coupling record.

    ``derivative_coupling_per_bohr`` is ``None`` when the energy gap is at or
    below ``gap_tolerance_hartree``.  In that case the regular
    ``interstate_numerator_hartree_per_bohr`` is the quantity to pass to a
    degenerate-subspace or branching-plane treatment.
    """

    state_i: int
    state_j: int
    energy_gap_hartree: float
    interstate_numerator_hartree_per_bohr: np.ndarray
    derivative_coupling_per_bohr: np.ndarray | None
    gap_tolerance_hartree: float
    phase_overlaps: tuple[float, float]

    def __post_init__(self) -> None:
        numerator = np.array(
            self.interstate_numerator_hartree_per_bohr, dtype=float, copy=True
        )
        if numerator.ndim != 1 or not np.all(np.isfinite(numerator)):
            raise ValueError("SF-TDA interstate numerator must be a finite vector")
        coupling = self.derivative_coupling_per_bohr
        if coupling is not None:
            coupling = np.array(coupling, dtype=float, copy=True)
            if coupling.shape != numerator.shape or not np.all(np.isfinite(coupling)):
                raise ValueError("SF-TDA derivative coupling must be a finite vector")
            coupling.setflags(write=False)
        if self.state_i == self.state_j:
            raise ValueError("an interstate coupling requires two distinct roots")
        if self.gap_tolerance_hartree <= 0.0:
            raise ValueError("SF-TDA coupling gap tolerance must be positive")
        numerator.setflags(write=False)
        object.__setattr__(self, "interstate_numerator_hartree_per_bohr", numerator)
        object.__setattr__(self, "derivative_coupling_per_bohr", coupling)

    @property
    def resolved(self) -> bool:
        return self.derivative_coupling_per_bohr is not None


@dataclass(frozen=True)
class SFTDASubspaceDerivative:
    """Finite nuclear derivatives of the TDA operator in a root subspace."""

    states: tuple[int, ...]
    energies_hartree: np.ndarray
    derivative_matrices_hartree_per_bohr: np.ndarray

    def __post_init__(self) -> None:
        energies = np.array(self.energies_hartree, dtype=float, copy=True)
        matrices = np.array(
            self.derivative_matrices_hartree_per_bohr, dtype=float, copy=True
        )
        count = len(self.states)
        if count < 2 or len(set(self.states)) != count:
            raise ValueError("SF-TDA derivative subspace requires distinct roots")
        if energies.shape != (count,):
            raise ValueError("SF-TDA subspace energies have incompatible dimensions")
        if matrices.ndim != 3 or matrices.shape[1:] != (count, count):
            raise ValueError("SF-TDA subspace derivative matrices are incompatible")
        if not np.all(np.isfinite(energies)) or not np.all(np.isfinite(matrices)):
            raise ValueError("SF-TDA subspace derivatives must be finite")
        if not np.allclose(matrices, matrices.swapaxes(1, 2), atol=2.0e-11, rtol=0.0):
            raise ValueError("SF-TDA subspace derivative matrices must be symmetric")
        energies.setflags(write=False)
        matrices.setflags(write=False)
        object.__setattr__(self, "energies_hartree", energies)
        object.__setattr__(self, "derivative_matrices_hartree_per_bohr", matrices)


def sf_tda_interstate_coupling(
    result: SFTDAResult,
    derivative_kernel: SFTDADerivativeKernel,
    state_i: int,
    state_j: int,
    *,
    gap_tolerance_hartree: float = 1.0e-7,
    reference_amplitudes: np.ndarray | None = None,
) -> SFTDAInterstateCoupling:
    """Evaluate one analytic interstate numerator and derivative coupling.

    The sign of a derivative coupling is gauge dependent.  When reference
    amplitudes are supplied, each root is phase matched to the corresponding
    reference root.  Otherwise a deterministic largest-component-positive
    gauge is used.
    """

    first = int(state_i)
    second = int(state_j)
    energies = np.asarray(result.excitation_energies_hartree, dtype=float)
    amplitudes = np.asarray(result.amplitudes, dtype=float)
    if first == second:
        raise ValueError("an interstate coupling requires two distinct roots")
    if min(first, second) < 0 or max(first, second) >= len(energies):
        raise IndexError("SF-TDA coupling root index is out of range")
    if int(derivative_kernel.size) != amplitudes.shape[1]:
        raise ValueError("SF-TDA derivative kernel and result spaces differ")
    tolerance = float(gap_tolerance_hartree)
    if tolerance <= 0.0:
        raise ValueError("SF-TDA coupling gap tolerance must be positive")

    references: tuple[np.ndarray | None, np.ndarray | None]
    if reference_amplitudes is None:
        references = (None, None)
    else:
        reference = np.asarray(reference_amplitudes, dtype=float)
        if reference.shape != amplitudes.shape:
            raise ValueError("reference SF-TDA amplitudes have incompatible dimensions")
        references = (reference[first], reference[second])
    left, overlap_i = _fix_root_phase(_normalised_root(amplitudes, first), references[0])
    right, overlap_j = _fix_root_phase(
        _normalised_root(amplitudes, second), references[1]
    )
    numerator = np.asarray(derivative_kernel.gradient_vjp(left, right), dtype=float)
    expected = (int(derivative_kernel.dimension),)
    if numerator.shape != expected or not np.all(np.isfinite(numerator)):
        raise ValueError("SF-TDA derivative kernel returned an invalid interstate VJP")
    gap = float(energies[second] - energies[first])
    coupling = None if abs(gap) <= tolerance else numerator / gap
    return SFTDAInterstateCoupling(
        first,
        second,
        gap,
        numerator,
        coupling,
        tolerance,
        (float(overlap_i), float(overlap_j)),
    )


def sf_tda_subspace_derivative(
    result: SFTDAResult,
    derivative_kernel: SFTDADerivativeKernel,
    states: Sequence[int],
    *,
    reference_amplitudes: np.ndarray | None = None,
) -> SFTDASubspaceDerivative:
    """Return the regular derivative matrix within a selected root subspace."""

    indices = tuple(int(index) for index in states)
    if len(indices) < 2 or len(set(indices)) != len(indices):
        raise ValueError("SF-TDA derivative subspace requires distinct roots")
    energies = np.asarray(result.excitation_energies_hartree, dtype=float)
    amplitudes = np.asarray(result.amplitudes, dtype=float)
    if min(indices) < 0 or max(indices) >= len(energies):
        raise IndexError("SF-TDA derivative-subspace root index is out of range")
    if int(derivative_kernel.size) != amplitudes.shape[1]:
        raise ValueError("SF-TDA derivative kernel and result spaces differ")
    reference = None
    if reference_amplitudes is not None:
        reference = np.asarray(reference_amplitudes, dtype=float)
        if reference.shape != amplitudes.shape:
            raise ValueError("reference SF-TDA amplitudes have incompatible dimensions")
    roots = tuple(
        _fix_root_phase(
            _normalised_root(amplitudes, index),
            None if reference is None else reference[index],
        )[0]
        for index in indices
    )
    dimension = int(derivative_kernel.dimension)
    matrices = np.empty((dimension, len(indices), len(indices)), dtype=float)
    for row, left in enumerate(roots):
        for column in range(row, len(indices)):
            value = np.asarray(
                derivative_kernel.gradient_vjp(left, roots[column]), dtype=float
            )
            if value.shape != (dimension,) or not np.all(np.isfinite(value)):
                raise ValueError("SF-TDA derivative kernel returned an invalid subspace VJP")
            matrices[:, row, column] = value
            matrices[:, column, row] = value
    return SFTDASubspaceDerivative(indices, energies[list(indices)], matrices)


__all__ = [
    "SFTDAInterstateCoupling",
    "SFTDASubspaceDerivative",
    "sf_tda_interstate_coupling",
    "sf_tda_subspace_derivative",
]
