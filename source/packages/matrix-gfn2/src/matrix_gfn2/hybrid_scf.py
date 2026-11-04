"""Dense standard SCF in a persistent full-rank hybrid basis.

The callback supplies the complete density-dependent Fock and energy in that
basis. The module makes no AO conversion inside the iterative solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.linalg import eigh

from .scc import fermi_occupations
from .scc import monopole_energy
from .multipole import anisotropic_energy_and_moment_potentials
from .dispersion import d4_two_body


@dataclass(frozen=True)
class HybridSCFResult:
    density: np.ndarray
    coefficients: np.ndarray
    orbital_energies: np.ndarray
    occupations: np.ndarray
    energy: float
    iterations: int
    density_residual: float
    energy_residual: float
    converged: bool


def full_gfn2_hybrid_fock_energy(
    core: np.ndarray,
    hybrid_scc,
    interaction: np.ndarray,
    partition,
    coordinates_bohr: np.ndarray,
    anisotropic_parameters,
    anisotropic_geometry,
    d4_prepared,
    d4_parameters,
    d4_coordination,
    d4_models,
) -> Callable[[np.ndarray], tuple[np.ndarray, float]]:
    """Bind all density-dependent dense GFN2 terms in hybrid coordinates."""
    third = np.asarray(partition.third_order_hartree, dtype=float)

    def evaluate(density: np.ndarray) -> tuple[np.ndarray, float]:
        charges, multipoles = hybrid_scc.observables(density)
        if hasattr(interaction, "evaluate"):
            evaluation = interaction.evaluate(charges, compute_gradient=False)
            quadratic_potential = evaluation.potential
            quadratic_energy = evaluation.energy_hartree
        else:
            quadratic_potential = interaction @ charges
            quadratic_energy = 0.5 * float(charges @ quadratic_potential)
        shell_potential = quadratic_potential + third * charges**2
        anisotropic_energy, charge_potential, dipole_potential, quadrupole_potential = (
            anisotropic_energy_and_moment_potentials(
                multipoles, coordinates_bohr, anisotropic_parameters,
                geometry=anisotropic_geometry,
            )
        )
        d4 = d4_two_body(
            coordinates_bohr, multipoles.charges, d4_coordination,
            d4_models, d4_parameters, prepared=d4_prepared,
        )
        fock = hybrid_scc.fock(
            core, shell_potential, charge_potential, dipole_potential,
            quadrupole_potential, d4.charge_derivative_hartree,
        )
        energy = (
            float(np.sum(density * core))
            + monopole_energy(
                charges, interaction, partition,
                quadratic_energy_override=quadratic_energy,
            )
            + anisotropic_energy + d4.energy_hartree
        )
        return fock, float(energy)

    return evaluate


def run_hybrid_standard_scf(
    overlap: np.ndarray,
    initial_density: np.ndarray,
    electrons: int,
    fock_and_energy: Callable[[np.ndarray], tuple[np.ndarray, float]],
    *,
    multiplicity: int = 1,
    temperature_kelvin: float = 300.0,
    density_tolerance: float = 1.0e-7,
    energy_tolerance: float = 1.0e-9,
    maximum_iterations: int = 100,
    mixing: float = 0.5,
    diis: bool = True,
) -> HybridSCFResult:
    """Solve a generalized Roothaan fixed point without AO cycling.

    A production caller must provide the *complete* GFN2 Fock and energy;
    testing only the monopole model does not certify the full method.
    """
    s = np.asarray(overlap, dtype=float)
    density = np.asarray(initial_density, dtype=float).copy()
    if (
        s.ndim != 2 or s.shape[0] != s.shape[1] or density.shape != s.shape
        or not np.all(np.isfinite(s)) or not np.all(np.isfinite(density))
        or electrons <= 0 or maximum_iterations < 1
        or density_tolerance <= 0.0 or energy_tolerance <= 0.0
        or not 0.0 < mixing <= 1.0
    ):
        raise ValueError("invalid hybrid dense SCF configuration")
    population = float(np.sum(density * s))
    if population <= 0:
        raise ValueError("initial hybrid density has no electrons")
    density *= electrons / population
    previous_energy = None
    converged = False
    residual = np.inf
    energy_residual = np.inf
    fock_history: list[np.ndarray] = []
    error_history: list[np.ndarray] = []
    for iteration in range(1, maximum_iterations + 1):
        fock, energy = fock_and_energy(density)
        fock = np.asarray(fock, dtype=float)
        effective_fock = fock
        if diis:
            error = fock @ density @ s - s @ density @ fock
            fock_history.append(fock)
            error_history.append(error.ravel())
            if len(fock_history) > 6:
                del fock_history[0]
                del error_history[0]
            if len(fock_history) >= 2:
                count = len(fock_history)
                gram = np.asarray([
                    [float(left @ right) for right in error_history]
                    for left in error_history
                ])
                system = np.empty((count + 1, count + 1))
                system[:count, :count] = gram
                system[:count, count] = -1.0
                system[count, :count] = -1.0
                system[count, count] = 0.0
                rhs = np.zeros(count + 1)
                rhs[count] = -1.0
                try:
                    weights = np.linalg.solve(system, rhs)[:count]
                    if np.all(np.isfinite(weights)) and np.max(np.abs(weights)) < 10.0:
                        effective_fock = sum(
                            weight * item for weight, item in zip(
                                weights, fock_history, strict=True,
                            )
                        )
                except np.linalg.LinAlgError:
                    pass
        eigenvalues, coefficients = eigh(effective_fock, s)
        occupations = fermi_occupations(
            eigenvalues, electrons, multiplicity=multiplicity,
            temperature_kelvin=temperature_kelvin,
        )
        output_density = (coefficients * occupations) @ coefficients.T
        residual = float(np.max(np.abs(output_density - density)))
        energy_residual = (
            np.inf if previous_energy is None else abs(float(energy) - previous_energy)
        )
        if residual < density_tolerance and energy_residual < energy_tolerance:
            density = output_density
            converged = True
            break
        damping = mixing if iteration < 3 or not diis else 1.0
        density = (1.0 - damping) * density + damping * output_density
        previous_energy = float(energy)
    # Report the energy and canonical spectrum of the actual returned density.
    final_fock, final_energy = fock_and_energy(density)
    eigenvalues, coefficients = eigh(np.asarray(final_fock, dtype=float), s)
    occupations = fermi_occupations(
        eigenvalues, electrons, multiplicity=multiplicity,
        temperature_kelvin=temperature_kelvin,
    )
    return HybridSCFResult(
        density, coefficients, eigenvalues, occupations, float(final_energy),
        iteration, residual, energy_residual, converged,
    )
