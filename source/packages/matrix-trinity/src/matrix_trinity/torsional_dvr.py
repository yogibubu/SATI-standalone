"""Periodic one-dimensional DVR for an isolated internal torsion.

The kinetic metric is the inverse effective moment of inertia along the
specified molecular path, after removal of overall translation and rotation.
The flat-measure Hamiltonian includes the one-dimensional geometric term.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from matrix_gf.harmonic import CM_PER_HARTREE


AMU_TO_ELECTRON_MASS = 1822.888486209


@dataclass(frozen=True)
class PeriodicTorsionDVR:
    energies_cm: np.ndarray
    eigenvectors: np.ndarray
    effective_inertia_amu_bohr2: np.ndarray
    geometric_potential_cm: np.ndarray
    transition_dipoles_au: np.ndarray | None
    ir_intensities_km_mol: np.ndarray | None


@dataclass(frozen=True)
class ConfinedTorsionDVR:
    """Grid DVR levels in one conformational well with zero boundary amplitude."""

    energies_cm: np.ndarray
    angles_rad: np.ndarray
    potential_cm: np.ndarray
    inertia_amu_bohr2: np.ndarray
    eigenvectors: np.ndarray


def _spectral_derivative_grid(count: int) -> tuple[np.ndarray, np.ndarray]:
    wave_numbers = np.fft.fftfreq(count, d=1.0 / count)
    identity = np.eye(count)
    derivative = np.fft.ifft(
        1j * wave_numbers[:, None] * np.fft.fft(identity, axis=0), axis=0,
    )
    return wave_numbers, derivative


def solve_periodic_torsion_dvr(
    potential_cm: np.ndarray,
    inertia_amu_bohr2: np.ndarray,
    *,
    dipole_au: np.ndarray | None = None,
    level_count: int = 6,
) -> PeriodicTorsionDVR:
    """Diagonalize a 2π-periodic torsion on an equally spaced Fourier grid.

    The grid is φ_j = 2πj/N. Potential, inertia, and optional Cartesian
    dipole components must refer to the same body-fixed molecular path.
    """
    potential = np.asarray(potential_cm, dtype=float)
    inertia = np.asarray(inertia_amu_bohr2, dtype=float)
    if (potential.ndim != 1 or len(potential) < 8 or len(potential) % 2
            or inertia.shape != potential.shape or np.any(inertia <= 0)
            or not np.all(np.isfinite(potential)) or not np.all(np.isfinite(inertia))):
        raise ValueError("periodic DVR needs finite even-grid potential and positive inertia")
    count = len(potential)
    if not 1 <= level_count <= count:
        raise ValueError("level_count must fit the periodic grid")
    wave_numbers, derivative = _spectral_derivative_grid(count)
    inverse_inertia = 1.0 / (inertia * AMU_TO_ELECTRON_MASS)
    first = np.fft.ifft(1j * wave_numbers * np.fft.fft(inverse_inertia)).real
    second = np.fft.ifft(-wave_numbers**2 * np.fft.fft(inverse_inertia)).real
    geometric = CM_PER_HARTREE * (
        -second / 8.0 + first**2 / (32.0 * inverse_inertia)
    )
    kinetic = (CM_PER_HARTREE / 2.0) * (
        derivative.conj().T @ (inverse_inertia[:, None] * derivative)
    )
    hamiltonian = kinetic + np.diag(potential + geometric)
    hamiltonian = (hamiltonian + hamiltonian.conj().T) / 2.0
    levels, vectors = np.linalg.eigh(hamiltonian)
    levels = levels[:level_count].real
    vectors = vectors[:, :level_count]
    transition_dipoles = None
    intensities = None
    if dipole_au is not None:
        dipole = np.asarray(dipole_au, dtype=float)
        if dipole.shape != (count, 3) or not np.all(np.isfinite(dipole)):
            raise ValueError("dipole samples must have shape (grid points, 3)")
        from .gvpt2_spectroscopy import IR_TRANSITION_FACTOR

        transition_dipoles = np.einsum(
            "i,ia,ij->ja", vectors[:, 0].conj(), dipole, vectors,
            optimize=True,
        )
        intensities = (IR_TRANSITION_FACTOR * np.maximum(levels - levels[0], 0.0)
                       * np.sum(np.abs(transition_dipoles)**2, axis=1))
    return PeriodicTorsionDVR(
        energies_cm=levels, eigenvectors=vectors,
        effective_inertia_amu_bohr2=inertia,
        geometric_potential_cm=geometric,
        transition_dipoles_au=transition_dipoles,
        ir_intensities_km_mol=intensities,
    )


def compare_periodic_property_perturbation_dvr(
    potential_cm: np.ndarray,
    inertia_amu_bohr2: np.ndarray,
    property_values: np.ndarray,
    *,
    state_index: int = 0,
    interpolation_points: int = 288,
    rotor_symmetry_number: int = 1,
    cosine_harmonics: int | None = None,
    include_sine_terms: bool = True,
) -> dict[str, object]:
    """Compare local diagonal perturbation theory with periodic DVR.

    Potential, path inertia, and property samples share an equally spaced
    2*pi-periodic grid.  Fourier interpolation supplies local derivatives at
    the minimum.  The reported sequence is the leading expectation, the next
    nonzero quartic-Hamiltonian result, an actual local H5/G''' correction,
    a diagonal-Morse H5 estimate, and the DVR expectation on the same surfaces.
    """
    potential = np.asarray(potential_cm, dtype=float)
    inertia = np.asarray(inertia_amu_bohr2, dtype=float)
    properties = np.asarray(property_values, dtype=float)
    if properties.ndim == 1:
        properties = properties[:, None]
    if (potential.ndim != 1 or len(potential) < 8 or len(potential) % 2
            or inertia.shape != potential.shape or properties.shape[0] != len(potential)
            or np.any(inertia <= 0) or not np.all(np.isfinite(potential))
            or not np.all(np.isfinite(inertia)) or not np.all(np.isfinite(properties))
            or interpolation_points < len(potential) or interpolation_points % 2
            or rotor_symmetry_number < 1
            or (cosine_harmonics is not None and cosine_harmonics < 1)):
        raise ValueError("invalid periodic surfaces for perturbative/DVR comparison")
    if not 0 <= state_index < interpolation_points:
        raise ValueError("DVR state index is outside the interpolated grid")

    def interpolate(values: np.ndarray) -> np.ndarray:
        count = len(values)
        harmonics = np.fft.fftfreq(count, d=1.0 / count)
        angles = 2 * np.pi * np.arange(interpolation_points) / interpolation_points
        coefficients = np.fft.fft(values, axis=0) / count
        return np.real(np.tensordot(
            np.exp(1j * angles[:, None] * harmonics[None, :]),
            coefficients, axes=(1, 0),
        ))

    potential = interpolate(potential)
    inertia = interpolate(inertia)
    properties = interpolate(properties)
    cosine_fit = None
    if cosine_harmonics is not None:
        minimum_guess = int(np.argmin(potential))
        phi = 2 * np.pi * np.arange(interpolation_points) / interpolation_points
        theta = rotor_symmetry_number * (phi - phi[minimum_guess])
        columns = [np.ones_like(theta)]
        columns.extend(np.cos(order * theta)
                       for order in range(1, cosine_harmonics + 1))
        if include_sine_terms:
            columns.extend(np.sin(order * theta)
                           for order in range(1, cosine_harmonics + 1))
        design = np.stack(columns, axis=1)

        def fit(values: np.ndarray) -> tuple[np.ndarray, float | list[float]]:
            coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
            fitted = design @ coefficients
            axes = 0
            centered = values - np.mean(values, axis=axes)
            scale = np.sqrt(np.mean(centered**2, axis=axes))
            fallback = np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2, axis=axes))
            scale = np.where(scale > 0, scale, np.where(fallback > 0, fallback, 1.0))
            residual_array = np.sqrt(np.mean((fitted - values) ** 2, axis=axes)) / scale
            residual = (float(residual_array) if np.ndim(residual_array) == 0
                        else np.asarray(residual_array).tolist())
            return fitted, residual

        potential, potential_residual = fit(potential)
        inverse_fit, inertia_residual = fit(1.0 / inertia)
        inertia = 1.0 / inverse_fit
        properties, property_residual = fit(properties)
        if np.any(inertia <= 0):
            raise ValueError("cosine representation gives a nonpositive path inertia")
        cosine_fit = {
            "coordinate": (
                f"real Fourier in theta={rotor_symmetry_number}*(phi-phi0)"
                if include_sine_terms else
                f"cos(k*theta), theta={rotor_symmetry_number}*(phi-phi0)"
            ),
            "harmonics": cosine_harmonics,
            "sine_terms": include_sine_terms,
            "relative_rms_potential": potential_residual,
            "relative_rms_inverse_inertia": inertia_residual,
            "relative_rms_property": property_residual,
        }
    wave_numbers = np.fft.fftfreq(interpolation_points, d=1.0 / interpolation_points)
    inverse_inertia = 1.0 / (inertia * AMU_TO_ELECTRON_MASS)

    def derivative(values: np.ndarray, order: int) -> np.ndarray:
        coefficients = np.fft.fft(values, axis=0)
        factor = (1j * wave_numbers) ** order
        if values.ndim > 1:
            factor = factor[:, None]
        return np.fft.ifft(factor * coefficients, axis=0).real

    first_inverse = derivative(inverse_inertia, 1)
    second_inverse = derivative(inverse_inertia, 2)
    geometric = CM_PER_HARTREE * (
        -second_inverse / 8 + first_inverse**2 / (32 * inverse_inertia)
    )
    effective_potential = potential + geometric
    minimum = int(np.argmin(effective_potential))
    metric_cm = CM_PER_HARTREE * inverse_inertia
    k2 = float(derivative(effective_potential, 2)[minimum])
    if k2 <= 0:
        raise ValueError("periodic reference point is not a local minimum")
    omega = float(np.sqrt(metric_cm[minimum] * k2))
    displacement = float(np.sqrt(metric_cm[minimum] / omega))
    f3 = float(derivative(effective_potential, 3)[minimum] * displacement**3)
    f4 = float(derivative(effective_potential, 4)[minimum] * displacement**4)
    f5 = float(derivative(effective_potential, 5)[minimum] * displacement**5)
    g1 = float(derivative(metric_cm, 1)[minimum] / displacement)
    g2 = float(derivative(metric_cm, 2)[minimum])
    g3 = float(derivative(metric_cm, 3)[minimum] * displacement)
    p = [derivative(properties, rank)[minimum] * displacement**rank
         for rank in range(1, 5)]

    from .gvpt2_spectroscopy import (
        _diagonal_morse_quintic,
        _next_nonzero_coordinate_expectation_qff,
    )
    state = (state_index,)
    tensors = (
        np.array([omega]), np.array([[[f3]]]), np.array([[[[f4]]]]),
        np.array([[[g1]]]), np.array([[[[g2]]]]),
        np.asarray(p[0])[:, None], np.asarray(p[1])[:, None, None],
        np.asarray(p[2])[:, None, None, None],
        np.asarray(p[3])[:, None, None, None, None],
    )
    leading, following = _next_nonzero_coordinate_expectation_qff(*tensors, state)
    _same, actual_following = _next_nonzero_coordinate_expectation_qff(
        *tensors, state,
        fifth=np.array([[[[[f5]]]]]),
        g3=np.array([[[[[g3]]]]]),
    )
    _same, h5_following = _next_nonzero_coordinate_expectation_qff(
        *tensors, state, fifth=np.array([[[[[f5]]]]]),
    )
    _same, g3_following = _next_nonzero_coordinate_expectation_qff(
        *tensors, state, g3=np.array([[[[[g3]]]]]),
    )
    morse_h5, morse_mismatch = _diagonal_morse_quintic(
        tensors[0], tensors[1], tensors[2],
    )
    _same, morse_following = _next_nonzero_coordinate_expectation_qff(
        *tensors, state, fifth=morse_h5,
    )
    dvr = solve_periodic_torsion_dvr(
        potential, inertia, level_count=state_index + 1,
    )
    dvr_property = np.einsum(
        "i,ic,i->c", dvr.eigenvectors[:, state_index].conj(), properties,
        dvr.eigenvectors[:, state_index], optimize=True,
    ).real
    equilibrium = properties[minimum]
    return {
        "state_index": state_index,
        "minimum_angle_rad": float(2 * np.pi * minimum / interpolation_points),
        "harmonic_frequency_cm": omega,
        "standard": (equilibrium + leading).tolist(),
        "complete_qff": (equilibrium + leading + following).tolist(),
        "complete_local_h5": (equilibrium + leading + h5_following).tolist(),
        "complete_local_g3": (equilibrium + leading + g3_following).tolist(),
        "complete_local_h5_g3": (equilibrium + leading + actual_following).tolist(),
        "complete_morse_h5": (equilibrium + leading + morse_following).tolist(),
        "dvr": dvr_property.tolist(),
        "morse_quartic_relative_rms": morse_mismatch,
        "cosine_representation": cosine_fit,
        "local_coefficients": {
            "f3_cm": f3, "f4_cm": f4, "f5_cm": f5,
            "g1_cm": g1, "g2_cm": g2, "g3_cm": g3,
        },
    }


def solve_confined_torsion_dvr(
    angles_rad: np.ndarray,
    potential_cm: np.ndarray,
    inertia_amu_bohr2: np.ndarray,
    *, grid_points: int = 512,
    level_count: int = 24,
) -> ConfinedTorsionDVR:
    """Diagonalize an adiabatic torsion on one continuous, finite well.

    A variable-mass finite-volume kinetic operator is represented on a dense
    coordinate grid.  The wavefunction vanishes at the two supplied well
    boundaries, which must lie in a classically forbidden region for all
    levels used subsequently.  No levels above the boundary potential should
    be interpreted as conformer-localized states.
    """
    from scipy.interpolate import PchipInterpolator
    from scipy.linalg import eigh_tridiagonal

    angles = np.asarray(angles_rad, dtype=float)
    potential = np.asarray(potential_cm, dtype=float)
    inertia = np.asarray(inertia_amu_bohr2, dtype=float)
    if (angles.ndim != 1 or len(angles) < 5 or potential.shape != angles.shape
            or inertia.shape != angles.shape or np.any(np.diff(angles) <= 0)
            or np.any(inertia <= 0) or not np.all(np.isfinite(angles))
            or not np.all(np.isfinite(potential))
            or not np.all(np.isfinite(inertia))
            or grid_points < 16 or not 1 <= level_count <= grid_points):
        raise ValueError("invalid continuous torsional well or DVR grid")
    grid = np.linspace(angles[0], angles[-1], grid_points + 2)[1:-1]
    spacing = (angles[-1] - angles[0]) / (grid_points + 1)
    potential_grid = PchipInterpolator(angles, potential)(grid)
    inverse_inertia_spline = PchipInterpolator(
        angles, 1.0 / (inertia * AMU_TO_ELECTRON_MASS),
    )
    inverse_inertia = inverse_inertia_spline(grid)
    first = inverse_inertia_spline.derivative()(grid)
    second = inverse_inertia_spline.derivative(2)(grid)
    geometric = CM_PER_HARTREE * (
        -second / 8.0 + first**2 / (32.0 * inverse_inertia)
    )
    edges = inverse_inertia_spline(
        np.linspace(angles[0], angles[-1], grid_points + 1),
    )
    factor = CM_PER_HARTREE / (2 * spacing**2)
    diagonal = factor * (edges[:-1] + edges[1:]) + potential_grid + geometric
    off_diagonal = -factor * edges[1:-1]
    energies, vectors = eigh_tridiagonal(
        diagonal, off_diagonal, select="i", select_range=(0, level_count - 1),
    )
    return ConfinedTorsionDVR(
        energies_cm=energies,
        angles_rad=grid,
        potential_cm=potential_grid,
        inertia_amu_bohr2=1.0 / (inverse_inertia * AMU_TO_ELECTRON_MASS),
        eigenvectors=vectors,
    )
