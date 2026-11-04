"""Estimate solvent populations required by successive ellipsoidal shells."""

from __future__ import annotations

import math
import numpy as np


def ellipsoid_volume(semiaxes_angstrom: np.ndarray) -> float:
    axes = np.asarray(semiaxes_angstrom, dtype=float)
    if axes.shape != (3,) or np.any(axes <= 0.0):
        raise ValueError("ellipsoid semiaxes must be three positive values")
    return float(4.0 * math.pi * np.prod(axes) / 3.0)


def estimate_next_shell_population(
    base_semiaxes_angstrom: np.ndarray,
    shell_count: int,
    solvent_vdw_radius_angstrom: float,
    number_density_angstrom3: float,
    *,
    delta_vdw_factor: float = 1.0,
    packing_fraction: float = 0.55,
) -> dict[str, float | int]:
    """Estimate molecules needed for shell ``n+1`` from shell ``n``.

    ``base_semiaxes`` are the solute extrema/2.  Each shell adds the solvent
    diameter; ``delta`` is an additional vdW-radius multiple.  The packing
    fraction accounts for inaccessible volume and molecular packing.
    """
    n = int(shell_count)
    radius = float(solvent_vdw_radius_angstrom)
    density = float(number_density_angstrom3)
    eta = float(packing_fraction)
    if n < 1 or radius <= 0.0 or density <= 0.0 or not 0.0 < eta <= 1.0:
        raise ValueError("invalid shell population controls")
    base = np.asarray(base_semiaxes_angstrom, dtype=float)
    if base.shape != (3,) or np.any(base < 0.0) or not np.any(base > 0.0):
        raise ValueError("base semiaxes must be non-negative with one positive axis")
    increment = n * (2.0 * radius) + float(delta_vdw_factor) * radius
    next_increment = (n + 1) * (2.0 * radius) + float(delta_vdw_factor) * radius
    v_n = ellipsoid_volume(base + increment)
    v_next = ellipsoid_volume(base + next_increment)
    shell_volume = max(0.0, v_next - v_n)
    raw = shell_volume * density
    estimated = max(1, int(round(raw * eta)))
    return {
        "shell_count": n,
        "next_shell_count": n + 1,
        "shell_volume_angstrom3": shell_volume,
        "number_density_angstrom3": density,
        "packing_fraction": eta,
        "raw_molecule_estimate": raw,
        "estimated_added_molecules": estimated,
    }
