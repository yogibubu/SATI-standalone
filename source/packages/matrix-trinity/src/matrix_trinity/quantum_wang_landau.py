"""TRINITY thermochemistry adapter for SENTINEL integer Wang--Landau DOS."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from matrix_chem import Phy, get_physical_constants
from matrix_sentinel.quantum_wang_landau import (
    QuantumWangLandauDOS,
    sample_integer_quadratic_wang_landau,
)

from .sonic_thermochemistry import SonicHDCPT2Result


@dataclass(frozen=True)
class QuantumWangLandauResult(QuantumWangLandauDOS):
    temperatures_K: np.ndarray
    log_partition_vibrational: np.ndarray
    internal_energy_kj_mol: np.ndarray
    entropy_j_mol_K: np.ndarray
    heat_capacity_j_mol_K: np.ndarray


def sample_hdcpt2_quantum_wang_landau(
    hdcpt2: SonicHDCPT2Result, *,
    max_total_quanta: int,
    mode_levels_cm: np.ndarray | None = None,
    y_tensor_cm: np.ndarray | None = None,
    energy_bins: int = 100,
    steps: int = 500_000,
    temperatures_K: tuple[float, ...] = (298.15, 500.0, 1000.0),
    flatness_fraction: float = 0.8,
    minimum_visits_per_bin: int = 20,
    check_interval: int = 1000,
    seed: int = 61453,
    walkers: int = 1,
    device: str = "cpu",
    precision: str = "auto",
) -> QuantumWangLandauResult:
    """Convert a finite HDCPT2 DOS into vibrational thermochemistry."""
    temperatures = np.asarray(temperatures_K, dtype=float)
    if (
        temperatures.ndim != 1 or temperatures.size == 0
        or np.any(temperatures <= 0) or not np.all(np.isfinite(temperatures))
        or not np.isfinite(hdcpt2.zero_point_cm)
    ):
        raise ValueError("thermochemistry requires finite ZPVE and positive temperatures")
    constants = get_physical_constants()
    hc = constants[Phy.PLANCK] * constants[Phy.C_LIGHT]
    inverse_temperature = hc / (constants[Phy.BOLTZMANN] * temperatures)
    dos = sample_integer_quadratic_wang_landau(
        hdcpt2.fundamentals_cm, hdcpt2.x_matrix_cm,
        max_total_quanta=max_total_quanta,
        mode_levels_cm=mode_levels_cm,
        y_tensor_cm=(getattr(hdcpt2, "y_tensor_cm", None) if y_tensor_cm is None else y_tensor_cm),
        inverse_temperatures=inverse_temperature,
        energy_bins=energy_bins, steps=steps,
        flatness_fraction=flatness_fraction,
        minimum_visits_per_bin=minimum_visits_per_bin,
        check_interval=check_interval, seed=seed,
        walkers=walkers, device=device, precision=precision,
    )
    sums = dos.partition_sums
    mean = dos.first_energy_sums / sums
    variance = np.maximum(dos.second_energy_sums / sums - mean**2, 0)
    gas_constant = constants[Phy.AVOGADRO] * constants[Phy.BOLTZMANN]
    return QuantumWangLandauResult(
        **vars(dos), temperatures_K=temperatures,
        log_partition_vibrational=(
            -inverse_temperature * hdcpt2.zero_point_cm + np.log(sums)
        ),
        internal_energy_kj_mol=(
            hc * constants[Phy.AVOGADRO] * (hdcpt2.zero_point_cm + mean) / 1000
        ),
        entropy_j_mol_K=gas_constant * (np.log(sums) + inverse_temperature * mean),
        heat_capacity_j_mol_K=gas_constant * inverse_temperature**2 * variance,
    )


__all__ = ["QuantumWangLandauResult", "sample_hdcpt2_quantum_wang_landau"]
