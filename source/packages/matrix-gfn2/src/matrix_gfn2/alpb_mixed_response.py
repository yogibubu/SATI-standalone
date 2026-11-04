"""Analytic mixed geometry derivatives of the existing ALPB SCC kernel.

The same piecewise Born descreening, p16 electrostatics, electrostatic size,
Lebedev surface switching and charge-quadratic hydrogen-bond term as alpb.py
are differentiated with internal NumPy mixed jets. Charge-independent CDS and
reference shifts belong to the existing complete ground-state Hessian.
"""

from __future__ import annotations
import numpy as np
from ._mixed_jet import MixedJet as Jet, stack, contract
from .alpb import (
    ALPB_ALPHA,
    P16_ZETA,
    ANGSTROM_TO_BOHR,
    KCAL_TO_HARTREE,
    _surface_grid,
    alpb_scc_interaction_matrix_hartree,
)


def alpb_mixed_interaction(r, numbers, parameters):
    n = len(numbers)
    z = lambda x: Jet.constant(x, r.n)
    radius = parameters.radii_bohr[np.asarray(numbers) - 1]
    rho = radius * parameters.descreening[np.asarray(numbers) - 1]
    cavity = radius - parameters.born_offset_bohr
    distances = {}
    for i in range(n):
        for j in range(i):
            distances[i, j] = ((r[i] - r[j]) ** 2).sum() ** 0.5
    psi = z(np.zeros(n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            distance = distances[max(i, j), min(i, j)]
            if distance.v > 35 * ANGSTROM_TO_BOHR:
                continue
            ap = distance + rho[j]
            am = distance - rho[j]
            if distance.v >= radius[i] + rho[j]:
                contribution = rho[j] / (ap * am) + 0.5 * (am / ap).log() / distance
            elif distance.v + rho[j] <= radius[i]:
                contribution = z(0.0)
            else:
                ratio = ap / radius[i]
                contribution = (
                    1 / radius[i]
                    - 1 / ap
                    + 0.5 / distance * (0.5 * am * (1 / ap - ratio / radius[i]) - ratio.log())
                )
            psi[i] = psi[i] + contribution
    scaled = 0.5 * cavity * psi
    argument = scaled * (1 + scaled * (4.85 * scaled - 0.8))
    inverse_born = 1 / cavity - argument.tanh() / radius
    if np.any(inverse_born.v <= 0):
        raise ValueError("nonpositive inverse ALPB Born radius")
    born = parameters.born_scale / inverse_born
    alpha_beta = ALPB_ALPHA / parameters.dielectric
    factor = (1 / parameters.dielectric - 1) / (1 + alpha_beta)
    K = z(np.zeros((n, n)))
    for i in range(n):
        K[i, i] = factor / born[i]
        for j in range(i):
            distance = distances[i, j]
            pair = (born[i] * born[j]) ** 0.5
            p16 = (1 + P16_ZETA * distance / (16 * pair)) ** -16
            value = factor / (distance + pair * p16)
            K[i, j] = value
            K[j, i] = value
    volume = radius**3
    total = volume.sum()
    centered = r - contract("i,ij->j", volume, r) / total
    second = contract("i,ij,ik->jk", volume, centered, centered)
    moment = (volume * ((centered**2).sum(axis=1) + 0.4 * radius**2)).sum()
    inertia = moment * np.eye(3) - second
    determinant = (
        inertia[0, 0] * (inertia[1, 1] * inertia[2, 2] - inertia[1, 2] * inertia[2, 1])
        - inertia[0, 1] * (inertia[1, 0] * inertia[2, 2] - inertia[1, 2] * inertia[2, 0])
        + inertia[0, 2] * (inertia[1, 0] * inertia[2, 1] - inertia[1, 1] * inertia[2, 0])
    )
    size = (determinant ** (1 / 3) / (0.4 * total)) ** 0.5
    K = K + factor * alpha_beta / size
    angular, weight = _surface_grid()
    sr = radius + parameters.probe_bohr
    smoothing = 0.3 * ANGSTROM_TO_BOHR
    cubic = -1 / (4 * smoothing**3)

    def primitive(value, index):
        return (
            0.25 / smoothing
            + 3 * cubic * (0.2 * value**2 - 0.5 * value * sr[index] + sr[index] ** 2 / 3)
        ) * value**3

    areas = []
    for i in range(n):
        radial_weight = primitive(sr[i] + smoothing, i) - primitive(sr[i] - smoothing, i)
        exposure = z(np.ones(len(weight)))
        points = r[i] + sr[i] * angular
        for j in range(n):
            if i == j or distances[max(i, j), min(i, j)].v >= sr[i] + sr[j] + smoothing:
                continue
            separation = ((points - r[j]) ** 2).sum(axis=1) ** 0.5
            reduced = separation - sr[j]
            normalized = reduced / smoothing
            switch = 0.5 + 0.75 * normalized - 0.25 * normalized**3
            switch[reduced.v <= -smoothing] = 0.0
            switch[reduced.v >= smoothing] = 1.0
            exposure = exposure * switch
        exposure[exposure.v <= 1e-6] = 0.0
        areas.append(radial_weight * (weight * exposure).sum())
    surface = stack(areas)
    hbond = (
        -KCAL_TO_HARTREE
        * parameters.hydrogen_bond_sqrt_kcal[np.asarray(numbers) - 1] ** 2
        / (4 * np.pi * sr**2)
    )
    for i in range(n):
        K[i, i] = K[i, i] + 2 * hbond[i] * surface[i]
    reference, _ = alpb_scc_interaction_matrix_hartree(r.v, np.asarray(numbers), parameters)
    error = float(np.max(abs(K.v - reference)))
    if not np.isfinite(error) or error > 2e-12:
        raise RuntimeError(f"ALPB mixed kernel differs from the ground-state functional: {error:g}")
    return K
