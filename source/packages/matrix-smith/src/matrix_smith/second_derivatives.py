"""Analytic Cartesian second derivatives for SONIC primitive coordinates."""

from __future__ import annotations

import numpy as np

from .derivative_capabilities import ANALYTIC_B_PRIME_FAMILIES as ANALYTIC_B_PRIME_FAMILIES

from .models import GICPrimitive


def analytic_primitive_hessian(
    primitive: GICPrimitive,
    coordinates_angstrom: np.ndarray,
) -> np.ndarray:
    """Return the analytic Cartesian Hessian of a supported primitive.

    ``R`` uses the exact radial Hessian. Frozen-reference ``SOOP`` and legacy
    laboratory-frame ``FTRANS`` are linear and therefore have zero Hessians.
    Body-fixed ``FTRANS`` is nonlinear through the reference-fragment frame and
    fails closed until its dedicated second-order kernel is available; it must
    never be reported as zero.
    """

    coords = np.asarray(coordinates_angstrom, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3 or not np.all(np.isfinite(coords)):
        raise ValueError("primitive Hessian coordinates must be finite with shape (natoms, 3)")
    if primitive.function in {"SOOP", "SYCART"}:
        from matrix_chem.soop_contract import frozen_soop_covector

        frozen_soop_covector(primitive, len(coords))
        return np.zeros((coords.size, coords.size), dtype=float)
    if primitive.function == "FTRANS":
        if primitive.ref_frame_atoms:
            raise NotImplementedError(
                "body-fixed FTRANS has a nonzero B-prime kernel"
            )
        return np.zeros((coords.size, coords.size), dtype=float)
    if primitive.function != "R" or len(primitive.atoms) != 2:
        raise NotImplementedError(
            f"analytic primitive Hessian is not implemented for {primitive.function!r}"
        )
    first, second = (int(atom) - 1 for atom in primitive.atoms)
    delta = coords[first] - coords[second]
    distance = float(np.linalg.norm(delta))
    if distance <= 1.0e-14:
        raise FloatingPointError("zero-length distance coordinate")
    unit = delta / distance
    block = (np.eye(3) - np.outer(unit, unit)) / distance
    hessian = np.zeros((coords.size, coords.size), dtype=float)
    first_slice = slice(3 * first, 3 * first + 3)
    second_slice = slice(3 * second, 3 * second + 3)
    hessian[first_slice, first_slice] = block
    hessian[second_slice, second_slice] = block
    hessian[first_slice, second_slice] = -block
    hessian[second_slice, first_slice] = -block
    return hessian
