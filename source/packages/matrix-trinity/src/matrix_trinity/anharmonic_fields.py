from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np


def build_normal_mode_anharmonic_field(
    *,
    reference_coordinates_angstrom: np.ndarray,
    frequencies_cm: np.ndarray,
    eigenvalues: np.ndarray,
    mass_weighted_modes: np.ndarray,
    masses_amu: np.ndarray,
    evaluator: Callable[[np.ndarray], tuple[float, np.ndarray]],
    config=None,
):
    """Build a TRINITY QFF from any direct QM or analytic E/G evaluator."""

    from matrix_vpt2_vci import NormalModeQFFFitConfig, fit_normal_mode_qff

    return fit_normal_mode_qff(
        reference_coordinates_angstrom=reference_coordinates_angstrom,
        frequencies_cm=frequencies_cm,
        eigenvalues=eigenvalues,
        mass_weighted_modes=mass_weighted_modes,
        masses_amu=masses_amu,
        evaluator=evaluator,
        config=NormalModeQFFFitConfig() if config is None else config,
        source="TRINITY_DIRECT_ENERGY_GRADIENT",
    )


def build_normal_mode_anharmonic_field_from_architect(
    xyzin: Path | str,
    force_field_path: Path | str,
    *,
    config=None,
):
    """Build the same TRINITY QFF by sampling an ARCHITECT/ZAFF E/G field."""

    from matrix_vpt2_vci import NormalModeQFFFitConfig, fit_normal_mode_qff_from_zaff

    return fit_normal_mode_qff_from_zaff(
        xyzin,
        force_field_path,
        config=NormalModeQFFFitConfig() if config is None else config,
    )


def build_normal_mode_qff_from_calculator(
    calculator,
    coordinates_angstrom: np.ndarray,
    harmonic,
    masses_amu: np.ndarray,
    *,
    config=None,
):
    """TRINITY-owned adapter from any resident E/G calculator to a QFF."""
    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    masses = np.asarray(masses_amu, dtype=float).reshape(-1)
    frequencies = np.asarray(harmonic.frequencies_cm1, dtype=float).reshape(-1)
    modes = np.asarray(harmonic.mass_weighted_modes, dtype=float)
    atom_count = coordinates.shape[0]
    if coordinates.shape != (atom_count, 3):
        raise ValueError("TRINITY bridge coordinates have the wrong shape")
    if masses.shape != (atom_count,) or np.any(masses <= 0.0):
        raise ValueError("TRINITY bridge masses must be positive")
    if modes.shape != (frequencies.size, coordinates.size):
        raise ValueError("TRINITY bridge normal modes have the wrong shape")
    if np.any(frequencies <= 0.0):
        raise ValueError("TRINITY QFF requires a vibrational minimum")
    frequency_factor_cm1 = 5140.487143715828
    eigenvalues = (frequencies / frequency_factor_cm1) ** 2

    def evaluate(geometry: np.ndarray) -> tuple[float, np.ndarray]:
        result = calculator.evaluate(
            np.asarray(geometry, dtype=float), compute_bond_orders=False
        )
        return float(result.energy_hartree), np.asarray(
            result.gradient_hartree_per_bohr, dtype=float
        )

    return build_normal_mode_anharmonic_field(
        reference_coordinates_angstrom=coordinates,
        frequencies_cm=frequencies,
        eigenvalues=eigenvalues,
        mass_weighted_modes=modes.reshape((frequencies.size, atom_count, 3)),
        masses_amu=masses,
        evaluator=evaluate,
        config=config,
    )
