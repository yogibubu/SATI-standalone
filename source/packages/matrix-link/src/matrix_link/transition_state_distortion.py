"""Explicit symmetry-lowering displacement of an index-two transition structure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from matrix_chem import MolecularGeometry, atomic_mass
from matrix_chem.topology.elements import atomic_number
from matrix_smith import is_total_symmetric_irrep


@dataclass(frozen=True)
class TransitionStateModeDistortion:
    geometry: MolecularGeometry
    mode_index: int
    frequency_cm: float
    irrep: str
    point_group: str
    direction: int
    maximum_atom_displacement_angstrom: float


@dataclass(frozen=True)
class PureModeDisplacementFit:
    amplitude: float
    rms_residual_angstrom: float
    relative_residual: float
    success: bool


def fit_pure_mode_displacement(
    source_coordinates_angstrom: np.ndarray,
    target_coordinates_angstrom: np.ndarray,
    source_cartesian_mode: np.ndarray,
) -> PureModeDisplacementFit:
    """Fit one normal-mode amplitude modulo rigid rotation and translation."""

    from matrix_chem import kabsch_align

    source = np.asarray(source_coordinates_angstrom, dtype=float)
    target = np.asarray(target_coordinates_angstrom, dtype=float)
    mode = np.asarray(source_cartesian_mode, dtype=float)
    if source.ndim != 2 or source.shape[1] != 3 or target.shape != source.shape:
        raise ValueError("pure-mode fit requires matching N by 3 geometries")
    if mode.size != source.size:
        raise ValueError("pure-mode fit mode has the wrong dimension")
    mode = mode.reshape(source.shape)
    if np.any(~np.isfinite(source)) or np.any(~np.isfinite(target)) or np.any(~np.isfinite(mode)):
        raise ValueError("pure-mode fit inputs must be finite")
    aligned_source = kabsch_align(source, target)
    displacement_norm = float(np.linalg.norm(target - aligned_source))
    mode_norm = float(np.linalg.norm(mode))
    if displacement_norm <= 1.0e-12 or mode_norm <= 1.0e-12:
        raise ValueError("pure-mode fit displacement or mode is null")

    def residual(amplitude: float) -> float:
        aligned = kabsch_align(source + float(amplitude) * mode, target)
        return float(np.sqrt(np.mean((aligned - target) ** 2)))

    # Rigid motions preserve every interatomic distance.  Each atom pair
    # therefore supplies a quadratic equation for the one allowed amplitude;
    # testing all real roots avoids the multiple minima that make a direct
    # one-dimensional Cartesian fit unreliable for floppy/rotational modes.
    amplitudes = [0.0]
    for left in range(len(source)):
        for right in range(left):
            delta_source = source[left] - source[right]
            delta_mode = mode[left] - mode[right]
            target_distance2 = float(
                np.dot(target[left] - target[right], target[left] - target[right])
            )
            quadratic = float(np.dot(delta_mode, delta_mode))
            linear = 2.0 * float(np.dot(delta_source, delta_mode))
            constant = float(np.dot(delta_source, delta_source)) - target_distance2
            if quadratic <= 1.0e-20:
                if abs(linear) > 1.0e-20:
                    amplitudes.append(-constant / linear)
                continue
            discriminant = linear * linear - 4.0 * quadratic * constant
            tolerance = 1.0e-12 * max(
                linear * linear, abs(4.0 * quadratic * constant), 1.0
            )
            if discriminant < -tolerance:
                continue
            root = np.sqrt(max(0.0, discriminant))
            amplitudes.extend(
                ((-linear - root) / (2.0 * quadratic),
                 (-linear + root) / (2.0 * quadratic))
            )
    values = np.asarray([residual(value) for value in amplitudes])
    best = int(np.argmin(values))
    amplitude = float(amplitudes[best])
    rms = float(values[best])
    relative = rms / max(displacement_norm / np.sqrt(source.size), 1.0e-15)
    return PureModeDisplacementFit(
        amplitude=amplitude,
        rms_residual_angstrom=rms,
        relative_residual=float(relative),
        success=bool(np.isfinite(amplitude) and np.isfinite(rms)),
    )


def assign_mode_irreps_by_overlap(
    exact_cartesian_modes: np.ndarray,
    printed_cartesian_modes: np.ndarray,
    printed_irreps: Sequence[str],
    *,
    minimum_overlap: float = 0.90,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Transfer printed symmetry labels to full-precision Hessian eigenvectors.

    Gaussian's normal-coordinate table is adequate for assigning irreps but
    its rounded vectors are not a safe displacement source for soft modes.
    A global maximum-overlap assignment keeps those two roles separate.
    """

    exact = np.asarray(exact_cartesian_modes, dtype=float)
    printed = np.asarray(printed_cartesian_modes, dtype=float)
    if exact.ndim < 2 or printed.shape != exact.shape:
        raise ValueError("exact and printed normal modes must have the same shape")
    labels = tuple(str(label).strip() for label in printed_irreps)
    if len(labels) != exact.shape[0] or any(not label for label in labels):
        raise ValueError("printed normal-mode irreps are incomplete")
    exact_rows = exact.reshape((exact.shape[0], -1))
    printed_rows = printed.reshape((printed.shape[0], -1))
    exact_norms = np.linalg.norm(exact_rows, axis=1)
    printed_norms = np.linalg.norm(printed_rows, axis=1)
    if np.any(exact_norms <= 0.0) or np.any(printed_norms <= 0.0):
        raise ValueError("normal-mode assignment requires nonzero vectors")
    overlaps = np.abs(exact_rows @ printed_rows.T) / (
        exact_norms[:, None] * printed_norms[None, :]
    )
    exact_indices, printed_indices = linear_sum_assignment(-overlaps)
    assigned = np.empty(exact.shape[0], dtype=int)
    assigned[exact_indices] = printed_indices
    selected_overlaps = overlaps[np.arange(exact.shape[0]), assigned]
    if np.any(selected_overlaps < float(minimum_overlap)):
        worst = float(np.min(selected_overlaps))
        raise ValueError(
            f"full-precision/printed normal-mode overlap {worst:.6f} is below "
            f"the required {float(minimum_overlap):.6f}"
        )
    return tuple(labels[index] for index in assigned), selected_overlaps


def distort_index_two_transition_state(
    geometry: MolecularGeometry,
    *,
    frequencies_cm: Sequence[float],
    normal_modes: np.ndarray,
    irreps: Sequence[str],
    point_group: str,
    maximum_atom_displacement_angstrom: float,
    direction: int = 1,
    allow_symmetry_lowering: bool = False,
) -> TransitionStateModeDistortion:
    """Displace along the unique non-total imaginary mode, only by explicit consent.

    The source geometry is immutable. The requested amplitude is the largest
    Cartesian displacement of any atom after removing residual center-of-mass
    translation from the printed normal mode.
    """

    if allow_symmetry_lowering is not True:
        raise PermissionError(
            "symmetry lowering requires allow_symmetry_lowering=True from an explicit request"
        )
    amplitude = float(maximum_atom_displacement_angstrom)
    if not np.isfinite(amplitude) or amplitude <= 0.0:
        raise ValueError("symmetry-lowering displacement amplitude must be positive")
    if direction not in {-1, 1}:
        raise ValueError("distortion direction must be +1 or -1")
    frequencies = np.asarray(frequencies_cm, dtype=float).reshape(-1)
    modes = np.asarray(normal_modes, dtype=float)
    labels = tuple(str(label).strip() for label in irreps)
    expected_shape = (frequencies.size, 3 * geometry.natoms)
    if modes.shape != expected_shape or len(labels) != frequencies.size:
        raise ValueError("frequencies, irreps and normal modes have inconsistent dimensions")
    imaginary = np.flatnonzero(frequencies < -1.0e-6)
    if imaginary.size != 2:
        raise ValueError("symmetry-lowering distortion requires exactly two imaginary modes")
    candidates = tuple(
        int(index)
        for index in imaginary
        if not is_total_symmetric_irrep(point_group, labels[index])
    )
    if len(candidates) != 1:
        raise ValueError(
            "index-two distortion requires exactly one non-totally-symmetric imaginary mode"
        )
    index = candidates[0]
    vector = modes[index].reshape((geometry.natoms, 3)).copy()
    masses = np.asarray(
        [atomic_mass(int(atomic_number(atom) or 0)) for atom in geometry.atoms], dtype=float
    )
    vector -= np.sum(masses[:, None] * vector, axis=0) / np.sum(masses)
    maximum = float(np.max(np.linalg.norm(vector, axis=1), initial=0.0))
    if not np.isfinite(maximum) or maximum <= 1.0e-14:
        raise ValueError("selected symmetry-breaking normal mode has zero Cartesian norm")
    displacement = float(direction) * amplitude * vector / maximum
    coordinates = np.asarray(geometry.coordinates_angstrom, dtype=float) + displacement
    distorted = MolecularGeometry(
        atoms=geometry.atoms,
        coordinates_angstrom=coordinates,
        comment=(
            f"symmetry-lowered from {point_group} along mode {index + 1} "
            f"({labels[index]}, {frequencies[index]:.4f} cm^-1); "
            f"max displacement {amplitude:.6f} angstrom"
        ),
    )
    return TransitionStateModeDistortion(
        geometry=distorted,
        mode_index=index + 1,
        frequency_cm=float(frequencies[index]),
        irrep=labels[index],
        point_group=str(point_group),
        direction=int(direction),
        maximum_atom_displacement_angstrom=amplitude,
    )


__all__ = [
    "PureModeDisplacementFit",
    "TransitionStateModeDistortion",
    "assign_mode_irreps_by_overlap",
    "distort_index_two_transition_state",
    "fit_pure_mode_displacement",
]
