"""Numerical invariants shared by Cartesian and SONIC Hessian/PED paths."""

from __future__ import annotations

import numpy as np


def validate_hessian_ped_invariants(
    *,
    atom_count: int,
    cartesian_hessian: object,
    sonic_hessian: object,
    frequencies_cartesian_cm1: object,
    frequencies_sonic_cm1: object,
    ped: object | None = None,
    tolerance_cm1: float = 1.0e-6,
) -> dict[str, object]:
    dimension = 3 * int(atom_count)
    cart = np.asarray(cartesian_hessian, dtype=float)
    sonic = np.asarray(sonic_hessian, dtype=float)
    freq_cart = np.asarray(frequencies_cartesian_cm1, dtype=float).reshape(-1)
    freq_sonic = np.asarray(frequencies_sonic_cm1, dtype=float).reshape(-1)
    errors: list[str] = []
    if cart.shape != (dimension, dimension):
        errors.append(f"Cartesian Hessian shape {cart.shape} != {(dimension, dimension)}")
    if sonic.shape != (dimension, dimension):
        errors.append(f"SONIC Hessian shape {sonic.shape} != {(dimension, dimension)}")
    if not np.isfinite(cart).all() or not np.isfinite(sonic).all():
        errors.append("Hessian contains NaN or infinity")
    if len(freq_cart) != dimension or len(freq_sonic) != dimension:
        errors.append("normal-mode collection is incomplete")
    elif not np.allclose(freq_cart, freq_sonic, atol=float(tolerance_cm1), rtol=0.0):
        errors.append("Cartesian and SONIC frequencies differ")
    if ped is not None:
        ped_array = np.asarray(ped, dtype=float)
        if ped_array.shape[0:2] != (dimension, dimension):
            errors.append(f"PED leading dimensions {ped_array.shape} are incompatible with 3N")
        if not np.isfinite(ped_array).all():
            errors.append("PED contains NaN or infinity")
    if errors:
        raise ValueError("; ".join(errors))
    return {"valid": True, "dimension": dimension, "mode_count": dimension, "tolerance_cm1": tolerance_cm1}
