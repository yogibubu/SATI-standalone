"""Analytic resident MATRIX-GFN2 density-response operators."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .dispersion import (
    D4AtomModel,
    D4Parameters,
    D4PreparedGeometry,
    SparseD4PreparedGeometry,
    d4_charge_hessian,
    d4_charge_hessian_matvec,
    d4_two_body,
    d4_two_body_sparse,
)
from .localized_solver import SparseSymmetricMatrix
from .multipole import (
    AnisotropicGeometry,
    AnisotropicParameters,
    CumulativeMultipoleOperators,
    AtomicMultipoles,
    anisotropic_energy_and_moment_potentials,
    dense_scc_fock,
    dense_scc_observables,
    anisotropic_fock_sparse,
    cumulative_atomic_multipoles_sparse,
    SparseCumulativeMultipoleOperators,
)
from .scc import ShellPartition, shell_charges_sparse


@dataclass(frozen=True)
class GFN2DensityResponseKernel:
    """Jacobian of the complete density-dependent resident GFN2 Fock matrix."""

    overlap: np.ndarray
    operators: CumulativeMultipoleOperators
    partition: ShellPartition
    shell_interaction: object
    reference_shell_charges: np.ndarray
    reference_multipoles: AtomicMultipoles
    coordinates_bohr: np.ndarray
    anisotropic_parameters: AnisotropicParameters
    anisotropic_geometry: AnisotropicGeometry
    d4_prepared: D4PreparedGeometry | SparseD4PreparedGeometry
    d4_parameters: D4Parameters
    d4_models: tuple[D4AtomModel, ...]
    d4_coordination: np.ndarray
    atm_charge_hessian: np.ndarray | None = None

    @cached_property
    def _d4_charge_response_hessian(self) -> np.ndarray:
        """The fixed-state D4 charge Jacobian is shared by every CPSCF action."""

        return d4_charge_hessian(
            self.reference_multipoles.charges, self.d4_prepared
        )

    def _observables(self, density: np.ndarray):
        return dense_scc_observables(
            density,
            self.operators,
            self.partition.orbital_shell,
            np.zeros(self.partition.nshells, dtype=float),
            np.zeros(len(self.reference_multipoles.charges), dtype=float),
        )

    def _shell_potential(self, shell_charges: np.ndarray) -> np.ndarray:
        if hasattr(self.shell_interaction, "evaluate"):
            quadratic = self.shell_interaction.evaluate(
                shell_charges, compute_gradient=False
            ).potential
        else:
            quadratic = np.asarray(self.shell_interaction) @ shell_charges
        return quadratic + np.asarray(self.partition.third_order_hartree) * shell_charges**2

    def _linear_shell_potential(self, perturbation: np.ndarray) -> np.ndarray:
        if hasattr(self.shell_interaction, "evaluate"):
            quadratic = self.shell_interaction.evaluate(
                perturbation, compute_gradient=False
            ).potential
        else:
            quadratic = np.asarray(self.shell_interaction) @ perturbation
        return quadratic + (
            2.0
            * np.asarray(self.partition.third_order_hartree)
            * self.reference_shell_charges
            * perturbation
        )

    def _d4(self, charges: np.ndarray):
        if isinstance(self.d4_prepared, SparseD4PreparedGeometry):
            return d4_two_body_sparse(charges, self.d4_prepared, self.d4_parameters)
        return d4_two_body(
            self.coordinates_bohr,
            charges,
            self.d4_coordination,
            self.d4_models,
            self.d4_parameters,
            prepared=self.d4_prepared,
        )

    def __call__(self, density_perturbation: np.ndarray) -> np.ndarray:
        """Apply the exact fixed-geometry Fock Jacobian to an AO density."""

        density = np.asarray(density_perturbation, dtype=float)
        if density.shape != self.overlap.shape or not np.all(np.isfinite(density)):
            raise ValueError("GFN2 density perturbation has incompatible dimensions")
        if np.max(np.abs(density - density.T), initial=0.0) > 2.0e-12:
            raise ValueError("GFN2 density perturbation must be symmetric")
        shell_delta, multipole_delta = self._observables(density)
        shell_potential = self._linear_shell_potential(shell_delta)
        _, qpot, mupot, thetapot = anisotropic_energy_and_moment_potentials(
            multipole_delta,
            self.coordinates_bohr,
            self.anisotropic_parameters,
            geometry=self.anisotropic_geometry,
        )
        d4_potential = self._d4_charge_response_hessian @ multipole_delta.charges
        if self.atm_charge_hessian is not None:
            d4_potential += self.atm_charge_hessian @ multipole_delta.charges
        return dense_scc_fock(
            np.zeros_like(self.overlap),
            self.overlap,
            self.operators,
            self.partition.orbital_shell,
            shell_potential,
            qpot,
            mupot,
            thetapot,
            d4_potential,
        )

    def density_dependent_fock(self, density: np.ndarray) -> np.ndarray:
        """Evaluate the nonlinear Fock at a density displacement from the reference."""

        values = np.asarray(density, dtype=float)
        shell_delta, multipole_delta = self._observables(values)
        shell_charges = self.reference_shell_charges + shell_delta
        multipoles = AtomicMultipoles(
            self.reference_multipoles.charges + multipole_delta.charges,
            self.reference_multipoles.dipoles_bohr + multipole_delta.dipoles_bohr,
            self.reference_multipoles.quadrupoles_bohr2
            + multipole_delta.quadrupoles_bohr2,
        )
        shell_potential = self._shell_potential(shell_charges)
        _, qpot, mupot, thetapot = anisotropic_energy_and_moment_potentials(
            multipoles,
            self.coordinates_bohr,
            self.anisotropic_parameters,
            geometry=self.anisotropic_geometry,
        )
        d4 = self._d4(multipoles.charges)
        return dense_scc_fock(
            np.zeros_like(self.overlap),
            self.overlap,
            self.operators,
            self.partition.orbital_shell,
            shell_potential,
            qpot,
            mupot,
            thetapot,
            d4.charge_derivative_hartree,
        )


@dataclass(frozen=True)
class SparseGFN2DensityResponseKernel:
    """Pair-packed Jacobian of all density-dependent GFN2 Fock terms."""

    overlap: SparseSymmetricMatrix
    operators: SparseCumulativeMultipoleOperators
    partition: ShellPartition
    shell_interaction: object
    reference_shell_charges: np.ndarray
    reference_multipoles: AtomicMultipoles
    coordinates_bohr: np.ndarray
    anisotropic_parameters: AnisotropicParameters
    anisotropic_geometry: AnisotropicGeometry
    d4_prepared: D4PreparedGeometry | SparseD4PreparedGeometry
    orbital_atom: np.ndarray

    def __call__(
        self, density_perturbation: SparseSymmetricMatrix
    ) -> SparseSymmetricMatrix:
        density = density_perturbation
        if not isinstance(density, SparseSymmetricMatrix):
            raise TypeError("sparse GFN2 response requires SparseSymmetricMatrix")
        if density.size != self.overlap.size:
            raise ValueError("sparse GFN2 density response has incompatible size")
        shell_delta = shell_charges_sparse(
            density, self.overlap, self.partition
        ) - np.asarray(self.partition.reference_population)
        multipole_delta = cumulative_atomic_multipoles_sparse(
            density,
            self.operators,
            np.zeros(len(self.reference_multipoles.charges)),
        )
        if hasattr(self.shell_interaction, "evaluate"):
            quadratic = self.shell_interaction.evaluate(
                shell_delta, compute_gradient=False
            ).potential
        else:
            quadratic = np.asarray(self.shell_interaction) @ shell_delta
        shell_potential = quadratic + (
            2.0
            * np.asarray(self.partition.third_order_hartree)
            * self.reference_shell_charges
            * shell_delta
        )
        _, qpot, mupot, thetapot = anisotropic_energy_and_moment_potentials(
            multipole_delta,
            self.coordinates_bohr,
            self.anisotropic_parameters,
            geometry=self.anisotropic_geometry,
        )
        d4_potential = d4_charge_hessian_matvec(
            self.reference_multipoles.charges,
            multipole_delta.charges,
            self.d4_prepared,
        )
        shell_index = np.asarray(self.partition.orbital_shell, dtype=np.int64)
        atom_index = np.asarray(self.orbital_atom, dtype=np.int64)
        scalar = shell_potential[shell_index] + d4_potential[atom_index]
        monopole_values = -0.5 * self.overlap.values * (
            scalar[self.overlap.rows] + scalar[self.overlap.columns]
        )
        monopole = SparseSymmetricMatrix(
            self.overlap.size,
            self.overlap.rows,
            self.overlap.columns,
            monopole_values,
        )
        anisotropic = anisotropic_fock_sparse(
            self.operators,
            qpot,
            mupot,
            thetapot,
            size=self.overlap.size,
        )
        combined, discarded = monopole.scaled_add(
            anisotropic, threshold=0.0
        )
        if discarded != 0.0:
            raise RuntimeError("exact sparse density response discarded AO pairs")
        return combined

__all__ = ["GFN2DensityResponseKernel", "SparseGFN2DensityResponseKernel"]
