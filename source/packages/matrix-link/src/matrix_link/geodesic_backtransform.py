"""Optional geodesic realization of SONIC displacements.

The implementation follows the Cartesian form of the internal-coordinate
geodesic equation.  The directional derivative of the SONIC Jacobian is
evaluated numerically, while the existing LINK corrector remains responsible
for removing the small endpoint residual.  This keeps the option backend
independent and avoids introducing coordinate-type-specific patches here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class GeodesicBackTransform:
    coordinates_angstrom: np.ndarray
    values: np.ndarray
    residual: np.ndarray
    steps: int
    converged: bool
    method: str = "NUMERICAL_SONIC_GEODESIC"


def geodesic_internal_coordinate_step(
    coordinates_angstrom: np.ndarray,
    target_values: np.ndarray,
    evaluate: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    *,
    fixed_atom_indices: tuple[int, ...] = (),
    tolerance: float = 1.0e-8,
    max_steps: int = 16,
    jacobian_displacement: float = 1.0e-4,
) -> GeodesicBackTransform:
    """Follow one SONIC displacement along the local internal manifold.

    The initial Cartesian tangent is ``B^+ Delta q``.  At every integration
    point the Cartesian acceleration is approximated as

    ``x'' = -B^+ [(dB/dx x') x']``.

    The derivative is a centered directional difference of the already
    available SONIC Jacobian.  The method is intentionally a realization
    step, not a replacement for the physical Hessian or the endpoint
    corrector.
    """

    geometry_shape = np.asarray(coordinates_angstrom, dtype=float).shape
    x = np.asarray(coordinates_angstrom, dtype=float).reshape(-1).copy()
    evaluate_flat = lambda flat: evaluate(np.asarray(flat).reshape(geometry_shape))
    target = np.asarray(target_values, dtype=float).reshape(-1)
    values, jacobian = _validated_evaluation(evaluate_flat, x, target.size)
    residual = target - values
    if np.linalg.norm(residual) <= tolerance:
        return GeodesicBackTransform(x.reshape(geometry_shape), values, residual, 0, True)

    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if jacobian_displacement <= 0.0:
        raise ValueError("jacobian_displacement must be positive")

    velocity = np.linalg.pinv(jacobian, rcond=1.0e-10) @ residual
    fixed = _fixed_cartesian_mask(x.shape, fixed_atom_indices)
    velocity[fixed] = 0.0
    dt = 1.0 / float(max_steps)

    for _ in range(max_steps):
        x, velocity = _rk4_step(
            x,
            velocity,
            dt,
            evaluate_flat,
            fixed,
            jacobian_displacement,
        )

    values, _ = _validated_evaluation(evaluate_flat, x, target.size)
    residual = target - values
    return GeodesicBackTransform(
        coordinates_angstrom=x.reshape(geometry_shape),
        values=values,
        residual=residual,
        steps=max_steps,
        converged=bool(np.linalg.norm(residual) <= tolerance),
    )


def _rk4_step(x, velocity, dt, evaluate, fixed, jacobian_displacement):
    k1x = velocity
    k1v = _acceleration(x, velocity, evaluate, fixed, jacobian_displacement)

    x2 = x + 0.5 * dt * k1x
    v2 = velocity + 0.5 * dt * k1v
    k2x = v2
    k2v = _acceleration(x2, v2, evaluate, fixed, jacobian_displacement)

    x3 = x + 0.5 * dt * k2x
    v3 = velocity + 0.5 * dt * k2v
    k3x = v3
    k3v = _acceleration(x3, v3, evaluate, fixed, jacobian_displacement)

    x4 = x + dt * k3x
    v4 = velocity + dt * k3v
    k4x = v4
    k4v = _acceleration(x4, v4, evaluate, fixed, jacobian_displacement)

    x_new = x + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
    v_new = velocity + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    x_new[fixed] = np.asarray(x)[fixed]
    v_new[fixed] = 0.0
    if not np.all(np.isfinite(x_new)) or not np.all(np.isfinite(v_new)):
        raise FloatingPointError("non-finite SONIC geodesic integration state")
    return x_new, v_new


def _acceleration(x, velocity, evaluate, fixed, jacobian_displacement):
    _, jacobian = _validated_evaluation(evaluate, x, None)
    speed = float(np.linalg.norm(velocity))
    if speed <= 1.0e-14:
        return np.zeros_like(velocity)
    epsilon = min(jacobian_displacement, 1.0e-4 / max(speed, 1.0))
    plus_x = x + epsilon * velocity
    minus_x = x - epsilon * velocity
    _, plus = _validated_evaluation(evaluate, plus_x, None)
    _, minus = _validated_evaluation(evaluate, minus_x, None)
    directional_jacobian = (plus - minus) / (2.0 * epsilon)
    acceleration = -np.linalg.pinv(jacobian, rcond=1.0e-10) @ (
        directional_jacobian @ velocity
    )
    acceleration[fixed] = 0.0
    return acceleration


def _validated_evaluation(evaluate, coordinates, expected_size):
    values, jacobian = evaluate(np.asarray(coordinates, dtype=float))
    values = np.asarray(values, dtype=float).reshape(-1)
    jacobian = np.asarray(jacobian, dtype=float)
    if expected_size is not None and values.shape != (expected_size,):
        raise ValueError("SONIC evaluator returned an unexpected coordinate count")
    if jacobian.ndim != 2 or jacobian.shape[0] != values.size:
        raise ValueError("SONIC evaluator returned an invalid Jacobian")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(jacobian)):
        raise FloatingPointError("SONIC evaluator returned non-finite values")
    return values, jacobian


def _fixed_cartesian_mask(shape, fixed_atom_indices):
    mask = np.zeros(int(np.prod(shape)), dtype=bool)
    for atom in fixed_atom_indices:
        start = 3 * int(atom)
        if start < 0 or start + 3 > mask.size:
            raise ValueError("fixed atom index is outside the Cartesian geometry")
        mask[start : start + 3] = True
    return mask
