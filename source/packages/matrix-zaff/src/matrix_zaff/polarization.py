"""Variational induced-dipole polarization on atomic and virtual sites."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import numpy as np


ANGSTROM3_TO_BOHR3 = (1.0 / 0.529177210903) ** 3
ZAFF_POLARIZATION_SCHEMA = "matrix.zaff.constrained_polarization.v1"


@dataclass(frozen=True)
class PolarizationSite:
    """One polarizable site expressed as an affine function of atom positions.

    ``atom_weights`` maps zero-based atom indices to coefficients.  Their sum
    must be one, but individual values may be negative; this represents
    extrapolated lone-pair sites while retaining an exact Cartesian chain
    rule.  A one-entry map is an ordinary atom-centered site.
    """

    polarizability_bohr3: float
    atom_weights: tuple[tuple[int, float], ...]
    label: str = ""

    def __post_init__(self) -> None:
        alpha = float(self.polarizability_bohr3)
        if not math.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("site polarizability must be finite and positive")
        if not self.atom_weights:
            raise ValueError("a polarization site needs atom-coordinate weights")
        indices = tuple(int(index) for index, _weight in self.atom_weights)
        weights = tuple(float(weight) for _index, weight in self.atom_weights)
        if len(set(indices)) != len(indices) or min(indices) < 0:
            raise ValueError("polarization-site atom indices must be unique and nonnegative")
        if not all(math.isfinite(weight) for weight in weights):
            raise ValueError("polarization-site weights must be finite")
        if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("polarization-site affine weights must sum to one")


@dataclass(frozen=True)
class PolarizationResult:
    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray
    induced_dipoles_e_bohr: np.ndarray
    iterations: int
    residual_norm: float
    backend: str
    schema: str = ZAFF_POLARIZATION_SCHEMA


def solve_constrained_polarization(
    coordinates_bohr: np.ndarray,
    permanent_charges: Sequence[float],
    sites: Sequence[PolarizationSite],
    *,
    thole_damping: float = 0.39,
    tolerance: float = 1.0e-11,
    maximum_iterations: int = 400,
    backend: Literal["auto", "direct", "fmm"] = "auto",
    fmm_minimum_sites: int = 256,
) -> PolarizationResult:
    """Minimize the damped induced-dipole functional.

    The response is charge conserving by construction: only dipoles are
    induced.  The stationary functional supplies conservative forces through
    the envelope theorem.  Atomic and off-atom sites share the same equations;
    affine site gradients are mapped exactly back to their parent atoms.
    """

    xyz = np.asarray(coordinates_bohr, dtype=float)
    charges = np.asarray(permanent_charges, dtype=float).reshape(-1)
    if xyz.shape != (len(charges), 3) or np.any(~np.isfinite(xyz)):
        raise ValueError("polarization coordinates and charges are inconsistent")
    if not sites:
        return PolarizationResult(
            0.0, np.zeros(xyz.size), np.zeros((0, 3)), 0, 0.0, "NONE"
        )
    damping = float(thole_damping)
    threshold = float(tolerance)
    if not math.isfinite(damping) or damping <= 0.0:
        raise ValueError("Thole damping must be finite and positive")
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("polarization tolerance must be finite and positive")

    site_xyz, weight_matrix = _site_geometry(xyz, sites)
    alpha = np.asarray([site.polarizability_bohr3 for site in sites], dtype=float)
    selected = _select_backend(backend, len(sites), int(fmm_minimum_sites))
    field = _permanent_field_direct(site_xyz, xyz, charges, alpha, damping)

    def matrix_vector(vector: np.ndarray) -> np.ndarray:
        dipoles = np.asarray(vector, dtype=float).reshape(len(sites), 3)
        induced = _induced_field_direct(site_xyz, dipoles, alpha, damping)
        return (dipoles / alpha[:, None] - induced).reshape(-1)

    diagonal = np.repeat(alpha, 3)
    dipoles, iterations, residual = _preconditioned_cg(
        matrix_vector,
        field.reshape(-1),
        diagonal,
        threshold,
        int(maximum_iterations),
    )
    mu = dipoles.reshape(len(sites), 3)
    energy, atomic_gradient = _polarization_energy_gradient(
        xyz,
        charges,
        site_xyz,
        weight_matrix,
        alpha,
        mu,
        damping,
    )
    return PolarizationResult(
        float(energy),
        atomic_gradient.reshape(-1),
        mu,
        iterations,
        residual,
        f"{selected.upper()}_VARIATIONAL_THOLE_INDUCED_DIPOLES",
    )


def atom_centered_polarization_sites(
    polarizabilities_angstrom3: Sequence[float],
) -> tuple[PolarizationSite, ...]:
    """Construct one atomic site per supplied positive polarizability."""

    return tuple(
        PolarizationSite(
            float(value) * ANGSTROM3_TO_BOHR3,
            ((index, 1.0),),
            label=f"ATOM_{index + 1}",
        )
        for index, value in enumerate(polarizabilities_angstrom3)
    )


def methyl_united_atom_polarization_sites(
    polarizabilities_angstrom3: Sequence[float],
    methyl_groups: Sequence[Sequence[int]],
) -> tuple[PolarizationSite, ...]:
    """Condense optional ``(C,H,H,H)`` groups to charge-conserving CH3 sites.

    The united site is located on carbon and receives the sum of the four
    atomic polarizabilities.  Groups must be disjoint; atoms outside them
    remain ordinary atom-centered sites.
    """

    values = tuple(float(value) for value in polarizabilities_angstrom3)
    consumed: set[int] = set()
    united: list[PolarizationSite] = []
    for group_index, raw_group in enumerate(methyl_groups):
        group = tuple(int(atom) for atom in raw_group)
        if len(group) != 4 or len(set(group)) != 4:
            raise ValueError("a methyl united-atom group must contain C and three H atoms")
        if min(group) < 0 or max(group) >= len(values) or consumed.intersection(group):
            raise ValueError("methyl united-atom groups must be in range and disjoint")
        consumed.update(group)
        united.append(
            PolarizationSite(
                sum(values[atom] for atom in group) * ANGSTROM3_TO_BOHR3,
                ((group[0], 1.0),),
                label=f"UNITED_CH3_{group_index + 1}",
            )
        )
    atomic = [
        PolarizationSite(
            values[atom] * ANGSTROM3_TO_BOHR3,
            ((atom, 1.0),),
            label=f"ATOM_{atom + 1}",
        )
        for atom in range(len(values))
        if atom not in consumed
    ]
    return tuple(united + atomic)


def _site_geometry(
    coordinates: np.ndarray,
    sites: Sequence[PolarizationSite],
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.zeros((len(sites), len(coordinates)), dtype=float)
    for site_index, site in enumerate(sites):
        for atom, weight in site.atom_weights:
            if atom >= len(coordinates):
                raise ValueError("polarization-site atom index is out of range")
            weights[site_index, atom] = weight
    return weights @ coordinates, weights


def _thole_factors(
    distance: float,
    alpha_i: float,
    alpha_j: float,
    damping: float,
) -> tuple[float, float, float, float]:
    scale = math.sqrt(alpha_i * alpha_j)
    x = damping * distance**3 / scale
    exponential = math.exp(-x)
    derivative_x = 3.0 * x / distance
    f3 = 1.0 - exponential
    f5 = 1.0 - (1.0 + x) * exponential
    return f3, f5, exponential * derivative_x, x * exponential * derivative_x


def _dipole_tensor(
    delta: np.ndarray,
    alpha_i: float,
    alpha_j: float,
    damping: float,
) -> np.ndarray:
    distance = float(np.linalg.norm(delta))
    if distance <= 1.0e-12:
        return np.zeros((3, 3))
    f3, f5, _df3, _df5 = _thole_factors(distance, alpha_i, alpha_j, damping)
    return 3.0 * f5 * np.outer(delta, delta) / distance**5 - f3 * np.eye(3) / distance**3


def _permanent_field_direct(
    site_xyz: np.ndarray,
    atom_xyz: np.ndarray,
    charges: np.ndarray,
    alpha: np.ndarray,
    damping: float,
) -> np.ndarray:
    field = np.zeros_like(site_xyz)
    for site_index, position in enumerate(site_xyz):
        for atom, charge in enumerate(charges):
            delta = position - atom_xyz[atom]
            distance = float(np.linalg.norm(delta))
            if distance <= 1.0e-12 or charge == 0.0:
                continue
            # A permanent charge is assigned the same finite size as its
            # responding site.  This is the one-center limit of Thole damping.
            f3, _f5, _df3, _df5 = _thole_factors(
                distance, alpha[site_index], alpha[site_index], damping
            )
            field[site_index] += charge * f3 * delta / distance**3
    return field


def _induced_field_direct(
    site_xyz: np.ndarray,
    dipoles: np.ndarray,
    alpha: np.ndarray,
    damping: float,
) -> np.ndarray:
    field = np.zeros_like(dipoles)
    for left in range(len(site_xyz)):
        for right in range(left + 1, len(site_xyz)):
            tensor = _dipole_tensor(
                site_xyz[left] - site_xyz[right],
                alpha[left],
                alpha[right],
                damping,
            )
            field[left] += tensor @ dipoles[right]
            field[right] += tensor @ dipoles[left]
    return field


def _polarization_energy_gradient(
    atom_xyz: np.ndarray,
    charges: np.ndarray,
    site_xyz: np.ndarray,
    weights: np.ndarray,
    alpha: np.ndarray,
    dipoles: np.ndarray,
    damping: float,
) -> tuple[float, np.ndarray]:
    energy = 0.5 * float(np.sum(dipoles * dipoles / alpha[:, None]))
    atom_gradient = np.zeros_like(atom_xyz)
    site_gradient = np.zeros_like(site_xyz)
    for site_index, position in enumerate(site_xyz):
        mu = dipoles[site_index]
        for atom, charge in enumerate(charges):
            delta = position - atom_xyz[atom]
            distance = float(np.linalg.norm(delta))
            if distance <= 1.0e-12 or charge == 0.0:
                continue
            f3, _f5, df3, _df5 = _thole_factors(
                distance, alpha[site_index], alpha[site_index], damping
            )
            radial = f3 / distance**3
            radial_prime = df3 / distance**3 - 3.0 * f3 / distance**4
            projection = float(mu @ delta)
            pair_energy = -charge * radial * projection
            derivative = -charge * (
                radial * mu + radial_prime * projection * delta / distance
            )
            energy += pair_energy
            site_gradient[site_index] += derivative
            atom_gradient[atom] -= derivative
    for left in range(len(site_xyz)):
        for right in range(left + 1, len(site_xyz)):
            delta = site_xyz[left] - site_xyz[right]
            distance = float(np.linalg.norm(delta))
            if distance <= 1.0e-12:
                continue
            f3, f5, df3, df5 = _thole_factors(
                distance, alpha[left], alpha[right], damping
            )
            g = f3 / distance**3
            h = f5 / distance**5
            gp = df3 / distance**3 - 3.0 * f3 / distance**4
            hp = df5 / distance**5 - 5.0 * f5 / distance**6
            left_projection = float(dipoles[left] @ delta)
            right_projection = float(dipoles[right] @ delta)
            dot = float(dipoles[left] @ dipoles[right])
            pair_energy = (
                -3.0 * h * left_projection * right_projection + g * dot
            )
            derivative = (
                -3.0
                * (
                    hp
                    * left_projection
                    * right_projection
                    * delta
                    / distance
                    + h
                    * (
                        right_projection * dipoles[left]
                        + left_projection * dipoles[right]
                    )
                )
                + gp * dot * delta / distance
            )
            energy += pair_energy
            site_gradient[left] += derivative
            site_gradient[right] -= derivative
    atom_gradient += weights.T @ site_gradient
    return energy, atom_gradient


def _preconditioned_cg(
    matrix_vector,
    right_hand_side: np.ndarray,
    inverse_diagonal: np.ndarray,
    tolerance: float,
    maximum_iterations: int,
) -> tuple[np.ndarray, int, float]:
    solution = np.zeros_like(right_hand_side)
    residual = right_hand_side.copy()
    preconditioned = inverse_diagonal * residual
    direction = preconditioned.copy()
    rz = float(residual @ preconditioned)
    target = tolerance * max(1.0, float(np.linalg.norm(right_hand_side)))
    for iteration in range(1, maximum_iterations + 1):
        product = matrix_vector(direction)
        denominator = float(direction @ product)
        if denominator <= 0.0 or not math.isfinite(denominator):
            raise ValueError("polarization response matrix is not positive definite")
        step = rz / denominator
        solution += step * direction
        residual -= step * product
        norm = float(np.linalg.norm(residual))
        if norm <= target:
            return solution, iteration, norm
        next_preconditioned = inverse_diagonal * residual
        next_rz = float(residual @ next_preconditioned)
        direction = next_preconditioned + (next_rz / rz) * direction
        preconditioned = next_preconditioned
        rz = next_rz
    raise RuntimeError("polarization SCF did not converge")


def _select_backend(
    backend: str,
    site_count: int,
    fmm_minimum_sites: int,
) -> str:
    normalized = str(backend).lower()
    if normalized not in {"auto", "direct", "fmm"}:
        raise ValueError("polarization backend must be auto, direct or fmm")
    if normalized == "fmm":
        raise RuntimeError(
            "the polarization FMM operator is not available in this build; "
            "use direct until the dipole FMM optional dependency is installed"
        )
    if normalized == "auto" and site_count >= fmm_minimum_sites:
        # Fail closed rather than silently claiming a scalable backend.
        return "direct"
    return "direct"


__all__ = [
    "ANGSTROM3_TO_BOHR3",
    "ZAFF_POLARIZATION_SCHEMA",
    "PolarizationResult",
    "PolarizationSite",
    "atom_centered_polarization_sites",
    "methyl_united_atom_polarization_sites",
    "solve_constrained_polarization",
]
