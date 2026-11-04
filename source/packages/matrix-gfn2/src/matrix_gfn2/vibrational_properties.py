"""Analytic ground-state IR and Raman properties for MATRIX-GFN2.

Nuclear directions are consumed one at a time.  The implementation therefore
shares the matrix-free CPSCF path used by the analytic Hessian and never needs
to retain a ``3N`` stack of density, dipole, or polarizability derivatives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .basis import orbital_angular_momentum_matrices
from .ground_state_response import (
    GroundStateDirectionalResponse,
    GroundStateHessianOperator,
)
from .localized_solver import SparseSymmetricMatrix
from .multipole import (
    cumulative_atomic_multipole_directional_derivative_sparse,
    cumulative_atomic_multipoles_sparse,
)


BOHR_TO_ANGSTROM = 0.529177210903
IR_INTENSITY_KM_MOL_PER_AU = 974.880111835
FINE_STRUCTURE_CONSTANT = 7.2973525693e-3


@dataclass(frozen=True)
class DirectionalVibrationalProperties:
    """Analytic property derivatives along one Cartesian displacement."""

    dipole_derivative_au: np.ndarray
    polarizability_derivative_au: np.ndarray
    magnetic_dipole_derivative_au: np.ndarray
    cpscf_residual_norm: float

    def __post_init__(self) -> None:
        dipole = np.array(self.dipole_derivative_au, dtype=float, copy=True)
        polarizability = np.array(
            self.polarizability_derivative_au, dtype=float, copy=True
        )
        magnetic = np.array(
            self.magnetic_dipole_derivative_au, dtype=float, copy=True
        )
        if (
            dipole.shape != (3,)
            or polarizability.shape != (3, 3)
            or magnetic.shape != (3,)
        ):
            raise ValueError("directional vibrational-property dimensions are invalid")
        if not np.allclose(polarizability, polarizability.T, atol=2.0e-11):
            raise ValueError("polarizability derivative must be symmetric")
        if (
            not np.all(np.isfinite(dipole))
            or not np.all(np.isfinite(polarizability))
            or not np.all(np.isfinite(magnetic))
        ):
            raise ValueError("directional vibrational properties must be finite")
        dipole.setflags(write=False)
        polarizability.setflags(write=False)
        magnetic.setflags(write=False)
        object.__setattr__(self, "dipole_derivative_au", dipole)
        object.__setattr__(self, "polarizability_derivative_au", polarizability)
        object.__setattr__(self, "magnetic_dipole_derivative_au", magnetic)
        object.__setattr__(self, "cpscf_residual_norm", float(self.cpscf_residual_norm))


@dataclass(frozen=True)
class VibrationalIntensityResult:
    """IR intensities and static Raman activities in normal coordinates."""

    dipole_derivatives_au: np.ndarray
    polarizability_derivatives_au: np.ndarray
    magnetic_dipole_derivatives_au: np.ndarray
    ir_intensities_km_mol: np.ndarray
    raman_activities_angstrom4_amu: np.ndarray
    raman_depolarization_ratios: np.ndarray
    vcd_rotational_strengths_au: np.ndarray
    maximum_cpscf_residual: float

    def __post_init__(self) -> None:
        dipole = np.array(self.dipole_derivatives_au, dtype=float, copy=True)
        polarizability = np.array(
            self.polarizability_derivatives_au, dtype=float, copy=True
        )
        magnetic = np.array(
            self.magnetic_dipole_derivatives_au, dtype=float, copy=True
        )
        ir = np.array(self.ir_intensities_km_mol, dtype=float, copy=True)
        raman = np.array(self.raman_activities_angstrom4_amu, dtype=float, copy=True)
        depolarization = np.array(
            self.raman_depolarization_ratios, dtype=float, copy=True
        )
        rotational = np.array(self.vcd_rotational_strengths_au, dtype=float, copy=True)
        nmodes = len(ir)
        if (
            dipole.shape != (nmodes, 3)
            or polarizability.shape != (nmodes, 3, 3)
            or magnetic.shape != (nmodes, 3)
            or raman.shape != (nmodes,)
            or depolarization.shape != (nmodes,)
            or rotational.shape != (nmodes,)
        ):
            raise ValueError("normal-mode property arrays have incompatible dimensions")
        arrays = (dipole, polarizability, magnetic, ir, raman, depolarization, rotational)
        if any(not np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("normal-mode properties must be finite")
        if np.any(ir < 0.0) or np.any(raman < 0.0):
            raise ValueError("IR intensities and Raman activities must be nonnegative")
        for value in arrays:
            value.setflags(write=False)
        object.__setattr__(self, "dipole_derivatives_au", dipole)
        object.__setattr__(self, "polarizability_derivatives_au", polarizability)
        object.__setattr__(self, "magnetic_dipole_derivatives_au", magnetic)
        object.__setattr__(self, "ir_intensities_km_mol", ir)
        object.__setattr__(self, "raman_activities_angstrom4_amu", raman)
        object.__setattr__(self, "raman_depolarization_ratios", depolarization)
        object.__setattr__(self, "vcd_rotational_strengths_au", rotational)
        object.__setattr__(
            self, "maximum_cpscf_residual", float(self.maximum_cpscf_residual)
        )


class GroundStateVibrationalPropertyOperator:
    """Directional analytic electric properties of one resident GFN2 state.

    The static polarizability is the variational independent-particle response
    of the converged GFN2 orbitals.  Its nuclear derivative includes analytic
    AO-moment, orbital-coefficient, and orbital-energy response terms.
    """

    def __init__(
        self,
        calculator: Any,
        state: Any,
        hessian_operator: GroundStateHessianOperator | None = None,
        magnetic_origin_bohr: np.ndarray | None = None,
    ) -> None:
        self.calculator = calculator
        self.state = state
        self.hessian_operator = (
            calculator.ground_state_hessian_operator(state)
            if hessian_operator is None
            else hessian_operator
        )
        self.context = self.hessian_operator.context
        self.integrals = self.context["integrals"]
        self.coordinates = np.asarray(self.context["coordinates"], dtype=float)
        self.atom_index = np.asarray(
            self.context["basis"].spherical_atom, dtype=np.intp
        )
        self.dimension = 3 * len(self.coordinates)
        valence = np.asarray(calculator._valence, dtype=float)
        self._canonical_magnetic_origin_bohr = (
            np.sum(valence[:, None] * self.coordinates, axis=0) / np.sum(valence)
        )
        self.magnetic_origin_bohr = (
            self._canonical_magnetic_origin_bohr.copy()
            if magnetic_origin_bohr is None
            else np.asarray(magnetic_origin_bohr, dtype=float)
        )
        if self.magnetic_origin_bohr.shape != (3,):
            raise ValueError("magnetic origin must contain three coordinates")
        self._dipole_ao = self._dense_pair_tensor(self.integrals.dipole)
        self._dipole_mo = np.einsum(
            "pi,apq,qj->aij",
            state.coefficients,
            self._dipole_ao,
            state.coefficients,
            optimize=True,
        )
        basis = self.context["basis"]
        self._angular_momentum_ao = orbital_angular_momentum_matrices(
            basis.cartesian_orbitals,
            basis.cartesian_to_spherical,
            origin_bohr=self._canonical_magnetic_origin_bohr,
        )
        self._angular_momentum_mo = np.einsum(
            "pi,apq,qj->aij",
            state.coefficients,
            self._angular_momentum_ao,
            state.coefficients,
            optimize=True,
        )
        occupations = state.alpha_occupations + state.beta_occupations
        left, right = np.triu_indices(state.nmo, 1)
        occupation_difference = occupations[left] - occupations[right]
        active = occupation_difference > 1.0e-12
        self._left = left[active]
        self._right = right[active]
        self._occupation_difference = occupation_difference[active]
        self._gaps = (
            state.orbital_energies_hartree[self._right]
            - state.orbital_energies_hartree[self._left]
        )
        if np.any(self._gaps <= self.hessian_operator.degeneracy_tolerance_hartree):
            raise ValueError("polarizability response encountered a zero orbital gap")

    def _dense_pair_tensor(self, values: np.ndarray) -> np.ndarray:
        pair_values = np.asarray(values, dtype=float)
        tensor = np.zeros((pair_values.shape[1], self.state.nao, self.state.nao))
        tensor[:, self.integrals.rows, self.integrals.columns] = pair_values.T
        off = self.integrals.rows != self.integrals.columns
        tensor[
            :, self.integrals.columns[off], self.integrals.rows[off]
        ] = pair_values[off].T
        return tensor

    def dipole_au(self) -> np.ndarray:
        """Return the origin-consistent molecular electric dipole in atomic units."""

        multipoles = self.context["kernel"].reference_multipoles
        return np.sum(
            multipoles.charges[:, None] * self.coordinates
            + multipoles.dipoles_bohr,
            axis=0,
        )

    def polarizability_au(self) -> np.ndarray:
        """Return the static independent-particle GFN2 polarizability."""

        transitions = self._dipole_mo[:, self._left, self._right].T
        weights = 2.0 * self._occupation_difference / self._gaps
        result = np.einsum(
            "i,ia,ib->ab", weights, transitions, transitions, optimize=True
        )
        return 0.5 * (result + result.T)

    def _dipole_integral_direction(self, direction: np.ndarray) -> np.ndarray:
        left_atom = self.atom_index[self.integrals.rows]
        right_atom = self.atom_index[self.integrals.columns]
        values = np.einsum(
            "pck,pk->pc",
            self.integrals.left_derivatives[:, 1:4],
            direction[left_atom],
            optimize=True,
        ) + np.einsum(
            "pck,pk->pc",
            self.integrals.right_derivatives[:, 1:4],
            direction[right_atom],
            optimize=True,
        )
        return self._dense_pair_tensor(values)

    def _dipole_derivative(
        self,
        direction: np.ndarray,
        response: GroundStateDirectionalResponse,
    ) -> np.ndarray:
        fixed = cumulative_atomic_multipole_directional_derivative_sparse(
            self.context["reference_density"],
            self.integrals,
            self.coordinates,
            direction,
            self.atom_index,
        )
        density = response.density_derivative
        response_density = SparseSymmetricMatrix(
            self.state.nao,
            self.integrals.rows,
            self.integrals.columns,
            density[self.integrals.rows, self.integrals.columns],
        )
        relaxed = cumulative_atomic_multipoles_sparse(
            response_density,
            self.context["sparse_operators"],
            np.zeros(len(self.coordinates)),
        )
        reference = self.context["kernel"].reference_multipoles
        return np.sum(
            fixed.charges[:, None] * self.coordinates
            + reference.charges[:, None] * direction
            + fixed.dipoles_bohr
            + relaxed.charges[:, None] * self.coordinates
            + relaxed.dipoles_bohr,
            axis=0,
        )

    def _polarizability_derivative(
        self,
        direction: np.ndarray,
        response: GroundStateDirectionalResponse,
    ) -> np.ndarray:
        coefficients = self.state.coefficients
        coefficient_direction = response.coefficient_derivative
        dipole_direction = self._dipole_integral_direction(direction)
        moment_direction = (
            np.einsum(
                "pi,apq,qj->aij",
                coefficient_direction,
                self._dipole_ao,
                coefficients,
                optimize=True,
            )
            + np.einsum(
                "pi,apq,qj->aij",
                coefficients,
                dipole_direction,
                coefficients,
                optimize=True,
            )
            + np.einsum(
                "pi,apq,qj->aij",
                coefficients,
                self._dipole_ao,
                coefficient_direction,
                optimize=True,
            )
        )
        transitions = self._dipole_mo[:, self._left, self._right].T
        transition_direction = moment_direction[:, self._left, self._right].T
        gap_direction = (
            response.orbital_energy_derivative[self._right]
            - response.orbital_energy_derivative[self._left]
        )
        weights = 2.0 * self._occupation_difference
        numerator = np.einsum(
            "i,ia,ib->ab",
            weights / self._gaps,
            transition_direction,
            transitions,
            optimize=True,
        )
        result = numerator + numerator.T - np.einsum(
            "i,ia,ib->ab",
            weights * gap_direction / self._gaps**2,
            transitions,
            transitions,
            optimize=True,
        )
        return 0.5 * (result + result.T)

    def _magnetic_dipole_derivative(
        self,
        direction: np.ndarray,
        response: GroundStateDirectionalResponse,
        dipole_derivative: np.ndarray,
    ) -> np.ndarray:
        """Return the reduced AAT contraction for one nuclear velocity.

        The electronic term is the occupied--virtual overlap between analytic
        nuclear CPSCF response and the orbital-Zeeman response.  The
        convective term translates the reference cumulative atomic charges
        and intratomic dipoles, so core and valence currents use exactly the
        same partition as the GFN2 electric properties.  Together these are
        the electric-dipole/orbital-magnetic terms required for harmonic VCD
        in the fixed-origin atomic axial tensor convention.
        """

        coefficients = self.state.coefficients
        connection = (
            coefficients.T
            @ self.state.overlap
            @ response.coefficient_derivative
        )
        rotations = connection[self._left, self._right]
        angular = self._angular_momentum_mo[
            :, self._left, self._right
        ].T
        # The nuclear orbital response in ``rotations`` is contracted with the
        # independent orbital-Zeeman response, which contributes one further
        # occupied--virtual energy denominator.  The real antisymmetric AO
        # generator represents L/i; conversion to the orbital magnetic moment
        # contributes alpha/2 in atomic units.
        electronic = -0.5 * FINE_STRUCTURE_CONSTANT * np.einsum(
            "i,ia->a",
            self._occupation_difference * rotations / self._gaps,
            angular,
            optimize=True,
        )
        relative = self.coordinates - self._canonical_magnetic_origin_bohr
        reference = self.context["kernel"].reference_multipoles
        convective = 0.5 * FINE_STRUCTURE_CONSTANT * np.sum(
            np.cross(
                reference.charges[:, None] * relative
                + reference.dipoles_bohr,
                direction,
            ),
            axis=0,
        )
        canonical = convective + electronic
        origin_shift = (
            self.magnetic_origin_bohr - self._canonical_magnetic_origin_bohr
        )
        # Transform the fixed-origin AAT consistently with the atomic polar
        # tensor.  The added cross product is rigorously perpendicular to the
        # electric transition derivative, so the rotational strength is
        # independent of the user-selected reporting origin.
        return canonical + 0.5 * FINE_STRUCTURE_CONSTANT * np.cross(
            origin_shift, dipole_derivative
        )

    def directional(self, vector: np.ndarray) -> DirectionalVibrationalProperties:
        value = np.asarray(vector, dtype=float)
        if value.shape not in {(self.dimension,), self.coordinates.shape}:
            raise ValueError("property trial vector has the wrong dimension")
        direction = value.reshape(self.coordinates.shape)
        _ds, _df, response = self.hessian_operator.directional_response(value)
        dipole = self._dipole_derivative(direction, response)
        return DirectionalVibrationalProperties(
            dipole,
            self._polarizability_derivative(direction, response),
            self._magnetic_dipole_derivative(direction, response, dipole),
            response.residual_norm,
        )

    def normal_modes(
        self,
        mass_weighted_modes: np.ndarray,
        masses_amu: np.ndarray,
    ) -> VibrationalIntensityResult:
        modes = np.asarray(mass_weighted_modes, dtype=float)
        masses = np.asarray(masses_amu, dtype=float)
        if modes.ndim != 2 or modes.shape[1] != self.dimension:
            raise ValueError("normal modes have incompatible dimensions")
        if masses.shape != (len(self.coordinates),) or np.any(masses <= 0.0):
            raise ValueError("normal-mode masses are invalid")
        inverse_sqrt_mass = 1.0 / np.repeat(np.sqrt(masses), 3)
        dipole = np.empty((len(modes), 3))
        polarizability = np.empty((len(modes), 3, 3))
        magnetic = np.empty((len(modes), 3))
        residual = 0.0
        for index, mode in enumerate(modes):
            properties = self.directional(mode * inverse_sqrt_mass)
            dipole[index] = properties.dipole_derivative_au
            polarizability[index] = properties.polarizability_derivative_au
            magnetic[index] = properties.magnetic_dipole_derivative_au
            residual = max(residual, properties.cpscf_residual_norm)
        ir = IR_INTENSITY_KM_MOL_PER_AU * np.einsum(
            "ia,ia->i", dipole, dipole
        )
        isotropic = np.trace(polarizability, axis1=1, axis2=2) / 3.0
        axx, ayy, azz = (
            polarizability[:, 0, 0],
            polarizability[:, 1, 1],
            polarizability[:, 2, 2],
        )
        anisotropy2 = 0.5 * (
            (axx - ayy) ** 2
            + (ayy - azz) ** 2
            + (azz - axx) ** 2
            + 6.0
            * (
                polarizability[:, 0, 1] ** 2
                + polarizability[:, 0, 2] ** 2
                + polarizability[:, 1, 2] ** 2
            )
        )
        raman_au = 45.0 * isotropic**2 + 7.0 * anisotropy2
        denominator = 45.0 * isotropic**2 + 4.0 * anisotropy2
        depolarization = np.divide(
            3.0 * anisotropy2,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0.0,
        )
        return VibrationalIntensityResult(
            dipole,
            polarizability,
            magnetic,
            ir,
            raman_au * BOHR_TO_ANGSTROM**4,
            depolarization,
            np.einsum("ia,ia->i", dipole, magnetic),
            residual,
        )


__all__ = [
    "DirectionalVibrationalProperties",
    "GroundStateVibrationalPropertyOperator",
    "VibrationalIntensityResult",
]
