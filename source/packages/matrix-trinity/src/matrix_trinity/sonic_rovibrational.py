"""Second-order rotation--vibration constants from an acquired ESALC field.

The contractions follow Eqs. 47, 50 and 52 of Mendolicchio et al.,
J. Chem. Theory Comput. 20, 8378 (2024).  The input metric is the full
body-fixed contravariant vibrational/rotational metric in the dimensionless
normal coordinates used by the vibrational force field.  No electronic
property points are required beyond the acquired F3 field.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Callable

import numpy as np

from matrix_chem.physical_constants import Phy, get_physical_constants
from matrix_gf.harmonic import CM_PER_HARTREE
from matrix_qm import HESSIAN_EIGENVALUE_TO_CM

from .curvilinear_vpt2 import (
    SonicAcquiredField, _modalize_sonic_field, sonic_rovibrational_metric,
)


_CONSTANTS = get_physical_constants()
_CM_PER_MHZ = 1.0e6 / _CONSTANTS[Phy.C_LIGHT]
_ROTATIONAL_METRIC_CM = (
    _CONSTANTS[Phy.PLANCK]
    / (4 * pi**2 * _CONSTANTS[Phy.TO_KG]
       * _CONSTANTS[Phy.M_PER_B]**2 * _CONSTANTS[Phy.C_LIGHT])
)


@dataclass(frozen=True)
class SonicRovibrationalConstants:
    """Principal-axis constants and quartic rotational tensor.

    ``delta_b_cm`` is B0 - Be; ``alpha_cm`` obeys
    B(v) = Be - sum_i alpha_i (v_i+1/2).  For exactly degenerate
    vibrational groups, only their alpha trace is invariant; the reported
    individual rows share that trace equally in the real partner gauge.
    ``tau_cm`` multiplies J_gamma J_delta J_tau J_eta / 4 before rotational
    operator symmetrization. ``tau_prime_cm`` is the conventional pair-axis
    combination of Eq. 53. Both retain the two transverse axes of a linear
    molecule or all three axes of a nonlinear molecule.
    """

    equilibrium_b_cm: np.ndarray
    ground_b_cm: np.ndarray
    delta_b_cm: np.ndarray
    alpha_cm: np.ndarray
    delta_components_cm: np.ndarray
    tau_cm: np.ndarray
    tau_prime_cm: np.ndarray
    metric_first_cm: np.ndarray
    metric_second_diagonal_cm: np.ndarray
    mode_frequencies_cm: np.ndarray

    @property
    def delta_b_mhz(self) -> np.ndarray:
        return self.delta_b_cm / _CM_PER_MHZ

    @property
    def tau_prime_mhz(self) -> np.ndarray:
        return self.tau_prime_cm / _CM_PER_MHZ


def rovibrational_constants_from_modal_derivatives(
    frequencies_cm: np.ndarray,
    cubic_cm: np.ndarray,
    vibrational_metric_first_cm: np.ndarray,
    full_metric_zero_cm: np.ndarray,
    full_metric_first_cm: np.ndarray,
    full_metric_second_diagonal_cm: np.ndarray,
    *, frequency_tolerance_cm: float = 0.01,
) -> SonicRovibrationalConstants:
    """Contract the complete modal metric into DeltaB and quartic distortion.

    Each term of Eq. 50 is returned separately in ``delta_components_cm``:
    rotational metric curvature, vibration--rotation coupling, and rho.
    The latter uses rho[j,i,i] = F3[j,i,i] + Gv'[i,i,j].
    """
    w = np.asarray(frequencies_cm, dtype=float)
    f3 = np.asarray(cubic_cm, dtype=float)
    vg1 = np.asarray(vibrational_metric_first_cm, dtype=float)
    g0 = np.asarray(full_metric_zero_cm, dtype=float)
    g1 = np.asarray(full_metric_first_cm, dtype=float)
    g2 = np.asarray(full_metric_second_diagonal_cm, dtype=float)
    n = w.size
    d = g0.shape[0] if g0.ndim == 2 else 0
    rcount = d - n
    if (w.ndim != 1 or np.any(w <= 0) or f3.shape != (n, n, n)
            or rcount not in {2, 3} or vg1.shape != (n, n, n)
            or g0.shape != (d, d)
            or g1.shape != (d, d, n) or g2.shape != (d, d, n)
            or not all(np.all(np.isfinite(x)) for x in (w, f3, vg1, g0, g1, g2))
            or frequency_tolerance_cm <= 0):
        raise ValueError("rovibrational contraction requires finite complete modal tensors")
    if (not np.allclose(g0, g0.T, atol=1e-7)
            or not np.allclose(g1, g1.swapaxes(0, 1), atol=1e-7)
            or not np.allclose(g2, g2.swapaxes(0, 1), atol=1e-7)
            or np.max(abs(g0[:n, n:]), initial=0.0) > 1e-4):
        raise ValueError("rovibrational metric must be symmetric and Eckart at equilibrium")
    rot_values, axes = np.linalg.eigh(g0[n:, n:])
    if np.any(rot_values <= 0):
        raise ValueError("rovibrational rotational metric must be positive definite")
    order = np.argsort(rot_values)[::-1]
    axes = axes[:, order]
    values = rot_values[order]
    # Principal axes inside a degenerate rotational block are not fixed by
    # eigh. Choose a reproducible real gauge from the body-frame unit axes.
    start = 0
    while start < rcount:
        stop = start + 1
        while stop < rcount and abs(values[stop] - values[start]) < 1e-7:
            stop += 1
        subspace = axes[:, start:stop]
        chosen: list[np.ndarray] = []
        for anchor in np.eye(rcount):
            vector = subspace @ (subspace.T @ anchor)
            for previous in chosen:
                vector -= previous * np.dot(previous, vector)
            length = np.linalg.norm(vector)
            if length > 1e-8:
                chosen.append(vector / length)
            if len(chosen) == stop - start:
                break
        axes[:, start:stop] = np.column_stack(chosen)
        start = stop
    for axis in range(rcount):
        pivot = int(np.argmax(abs(axes[:, axis])))
        if axes[pivot, axis] < 0:
            axes[:, axis] *= -1
    rotation = np.eye(d)
    rotation[n:, n:] = axes
    g0 = rotation.T @ g0 @ rotation
    g1 = np.einsum("ia,ijk,jb->abk", rotation, g1, rotation)
    g2 = np.einsum("ia,ijk,jb->abk", rotation, g2, rotation)
    beq = np.diag(g0[n:, n:]) / 2
    if np.max(abs(g0[:n, :n] - np.diag(w)), initial=0.0) > 1e-3:
        raise ValueError("rovibrational metric and harmonic field use different normal modes")

    components = np.zeros((n, rcount, 3))
    alpha = np.zeros((n, rcount))
    for i, wi in enumerate(w):
        for axis in range(rcount):
            r = n + axis
            components[i, axis, 0] = g2[r, r, i] / 8
            alpha[i, axis] -= g2[r, r, i] / 4
            for j, wj in enumerate(w):
                a = g1[i, r, j]
                b = g1[j, r, i]
                components[i, axis, 1] -= (a + b)**2 / (8 * (wi + wj))
                rho = f3[j, i, i] + vg1[i, i, j]
                components[i, axis, 2] -= rho * g1[r, r, j] / (8 * wj)
                alpha[i, axis] += rho * g1[r, r, j] / (4 * wj)
                denominator = wi**2 - wj**2
                if abs(wi - wj) > frequency_tolerance_cm:
                    alpha[i, axis] -= (
                        (a*a + b*b) * wj - 2 * a*b*wi
                    ) / (2 * denominator)
    delta = components.sum(axis=(0, 2))
    # Inside a degenerate irrep only the trace of alpha is gauge invariant.
    # Terms with both modes inside the irrep cancel in that trace; distribute
    # the remaining trace equally between real partners.
    groups: list[list[int]] = []
    for i in range(n):
        group = next((group for group in groups
                      if abs(w[i] - w[group[0]]) <= frequency_tolerance_cm), None)
        if group is None:
            groups.append([i])
        else:
            group.append(i)
    if any(len(group) > 1 for group in groups):
        for group in groups:
            if len(group) > 1:
                alpha[group, :] = alpha[group, :].mean(axis=0)
    # The asymmetric-top alpha formula and Eq. 50 must have the same trace.
    # Report Eq. 50 as the authoritative resonance-free ground-state result.
    rotational_derivative = g1[n:, n:, :]
    tau = -0.5 * np.einsum(
        "abi,cdi,i->abcd", rotational_derivative, rotational_derivative,
        1 / w, optimize=True,
    )
    tau_prime = np.empty((rcount, rcount))
    for axis in range(rcount):
        for other in range(rcount):
            tau_prime[axis, other] = (
                tau[axis, axis, other, other]
                + (2 * tau[axis, other, axis, other] if axis == other else 0.0)
            )
    return SonicRovibrationalConstants(
        equilibrium_b_cm=beq, ground_b_cm=beq + delta,
        delta_b_cm=delta, alpha_cm=alpha,
        delta_components_cm=components,
        tau_cm=tau, tau_prime_cm=tau_prime, metric_first_cm=g1,
        metric_second_diagonal_cm=g2, mode_frequencies_cm=w,
    )


def acquire_sonic_rovibrational_constants(
    acquired: SonicAcquiredField,
    *,
    geometry_at_s: Callable[[np.ndarray], np.ndarray],
    masses_amu: np.ndarray,
    metric_at_s: Callable[[np.ndarray], np.ndarray] | None = None,
    tangent_at_s: Callable[[np.ndarray], np.ndarray] | None = None,
    derivative_step_dimensionless: float = 0.02,
    rotational_dof: int = 3,
    constrained_vibrational_dof: int = 0,
) -> SonicRovibrationalConstants:
    """Reuse ESALC and geometry-based full-G points, optionally on a constrained TS.

    A constrained input contains only the stable transverse field and its
    matching induced metric. The frozen reaction coordinate is excluded before
    metric inversion; it is not treated as a stable vibrational occupation.
    """
    if not isinstance(acquired, SonicAcquiredField):
        raise TypeError("rovibrational acquisition requires a saved SONIC field")
    w, _, f3, _, vg1, _ = _modalize_sonic_field(acquired.field)
    n = w.size
    displacement = np.asarray(
        acquired.acquisition.get("sonic_displacements_per_dimensionless_mode"), dtype=float,
    )
    masses = np.asarray(masses_amu, dtype=float)
    h = float(derivative_step_dimensionless)
    if (isinstance(constrained_vibrational_dof,(bool,np.bool_))
            or not isinstance(constrained_vibrational_dof,(int,np.integer))
            or constrained_vibrational_dof<0):
        raise ValueError('nonnegative integer constrained vibrational rank required')
    if (displacement.shape != (n, n) or masses.ndim != 1
            or rotational_dof not in {2, 3}
            or masses.size * 3 - 3 - rotational_dof != n+constrained_vibrational_dof or np.any(masses <= 0)
            or not np.isfinite(h) or h <= 0):
        raise ValueError("rovibrational acquisition requires a complete ESALC chart")
    provider = metric_at_s or (lambda s: sonic_rovibrational_metric(
        geometry_at_s, s, masses, rotational_dof=rotational_dof,
        tangent_at_s=tangent_at_s,
        constrained_vibrational_dof=constrained_vibrational_dof,
    ))
    inverse = np.linalg.inv(displacement)
    transform = np.eye(n + rotational_dof)
    transform[:n, :n] = inverse
    vibrational_scale = HESSIAN_EIGENVALUE_TO_CM**2 / CM_PER_HARTREE
    scale = np.r_[np.full(n, np.sqrt(vibrational_scale)),
                  np.full(rotational_dof, np.sqrt(_ROTATIONAL_METRIC_CM))]

    def modal_metric(q: np.ndarray) -> np.ndarray:
        raw = np.asarray(provider(displacement @ q), dtype=float)
        if raw.shape != (n + rotational_dof, n + rotational_dof) or not np.all(np.isfinite(raw)):
            raise ValueError("full rovibrational metric has invalid dimensions")
        transformed = transform @ raw @ transform.T
        return transformed * scale[:, None] * scale[None, :]

    zero = np.zeros(n)
    g0 = modal_metric(zero)
    g1 = np.zeros((n + rotational_dof, n + rotational_dof, n))
    g2 = np.zeros_like(g1)
    for i in range(n):
        step = h * np.eye(n)[i]
        plus, minus = modal_metric(step), modal_metric(-step)
        g1[:, :, i] = (plus - minus) / (2 * h)
        g2[:, :, i] = (plus - 2 * g0 + minus) / h**2
    return rovibrational_constants_from_modal_derivatives(
        w, f3, vg1, g0, g1, g2,
    )


__all__ = [
    "SonicRovibrationalConstants", "acquire_sonic_rovibrational_constants",
    "rovibrational_constants_from_modal_derivatives",
]
