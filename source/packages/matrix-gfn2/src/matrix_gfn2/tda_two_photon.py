"""Matrix-free second-order electric-dipole response for TDA states.

The response equation is solved directly in the selected excitation space,
without constructing either an excited-state dipole tensor or a dense inverse.
For a target root ``f`` and photon frequency ``omega`` the Cartesian tensor is

``M_ab = <f|mu_a (A-omega I)^-1 mu_b|0> + (a <-> b)``.

The one-electron dipole action uses sparse configuration adjacencies implied by
the Slater--Condon rules in the spin-adapted ``std2`` normalization.  In that
convention the excited-state contraction is one half of the raw configuration
adjacency, whereas the ground-to-singlet vector retains its ``sqrt(2)`` factor.
Consequently a localized excitation space with a bounded number of
configurations per occupied and virtual orbital retains linear work and storage
at fixed response tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from .basis import (
    ContractedGaussian,
    GaussianPrimitive,
    atomic_multipole_integral_derivatives,
    atomic_multipole_integrals,
)
from .electronic_state import GFN2OrbitalState, GFN2SparseProjectorState


VectorOperator = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class SparseConfigurationDipole:
    """Ground and excited-space dipoles in a selected TDA configuration basis."""

    occupied_orbitals: np.ndarray
    virtual_orbitals: np.ndarray
    ground_vectors: np.ndarray
    rows: np.ndarray
    columns: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        occupied = np.asarray(self.occupied_orbitals, dtype=np.int64)
        virtual = np.asarray(self.virtual_orbitals, dtype=np.int64)
        ground = np.asarray(self.ground_vectors, dtype=float)
        rows = np.asarray(self.rows, dtype=np.int64)
        columns = np.asarray(self.columns, dtype=np.int64)
        values = np.asarray(self.values, dtype=float)
        size = len(occupied)
        if virtual.shape != (size,) or len(set(zip(occupied, virtual, strict=True))) != size:
            raise ValueError("TDA dipole configurations must be unique occupied--virtual pairs")
        if ground.shape != (3, size):
            raise ValueError("TDA ground-transition dipoles must have shape 3 x nconfig")
        if rows.shape != columns.shape or values.shape != (len(rows), 3):
            raise ValueError("sparse TDA dipole adjacency arrays are inconsistent")
        if np.any(rows < 0) or np.any(rows >= size) or np.any(columns < 0) or np.any(columns >= size):
            raise ValueError("sparse TDA dipole adjacency lies outside the configuration space")
        if not np.all(np.isfinite(ground)) or not np.all(np.isfinite(values)):
            raise ValueError("TDA dipole data must be finite")
        for value in (occupied, virtual, ground, rows, columns, values):
            value.setflags(write=False)
        object.__setattr__(self, "occupied_orbitals", occupied)
        object.__setattr__(self, "virtual_orbitals", virtual)
        object.__setattr__(self, "ground_vectors", ground)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "values", values)

    @property
    def size(self) -> int:
        return len(self.occupied_orbitals)

    @property
    def nonzero_count(self) -> int:
        return len(self.rows)

    def apply(self, component: int, vector: np.ndarray) -> np.ndarray:
        axis = int(component)
        operand = np.asarray(vector)
        if axis not in (0, 1, 2) or operand.shape != (self.size,):
            raise ValueError("TDA dipole application has an invalid component or vector")
        result = np.zeros(self.size, dtype=np.result_type(operand.dtype, self.values.dtype))
        np.add.at(result, self.rows, self.values[:, axis] * operand[self.columns])
        return result


@dataclass(frozen=True)
class TDATwoPhotonResult:
    target_state: int
    photon_energy_hartree: float
    transition_tensor_au: np.ndarray
    isotropic_strength_au: float
    response_residual_norms: np.ndarray
    response_iterations: np.ndarray
    resonance_margin_hartree: float
    damping_hartree: float = 0.0

    def __post_init__(self) -> None:
        tensor = np.asarray(self.transition_tensor_au)
        residuals = np.asarray(self.response_residual_norms, dtype=float)
        iterations = np.asarray(self.response_iterations, dtype=np.int64)
        if tensor.shape != (3, 3) or residuals.shape != (3,) or iterations.shape != (3,):
            raise ValueError("two-photon response dimensions are inconsistent")
        if not all(np.all(np.isfinite(value)) for value in (tensor, residuals)):
            raise ValueError("two-photon response must be finite")
        if not np.allclose(tensor, tensor.T, atol=2.0e-10, rtol=0.0):
            raise ValueError("degenerate two-photon transition tensor must be symmetric")
        if (
            self.isotropic_strength_au < -1.0e-14
            or self.resonance_margin_hartree <= 0.0
            or self.damping_hartree < 0.0
        ):
            raise ValueError("two-photon strength and resonance margin are invalid")
        for value in (tensor, residuals, iterations):
            value.setflags(write=False)
        object.__setattr__(self, "transition_tensor_au", tensor)
        object.__setattr__(self, "response_residual_norms", residuals)
        object.__setattr__(self, "response_iterations", iterations)

    @property
    def linewidth_fwhm_hartree(self) -> float:
        """Lorentzian full width at half maximum used by the response."""

        return 2.0 * float(self.damping_hartree)


@dataclass(frozen=True)
class TDATwoPhotonDirectionalDerivative:
    """Analytic directional derivative of one TDA two-photon observable."""

    response: TDATwoPhotonResult
    transition_tensor_derivative_au: np.ndarray
    isotropic_strength_derivative_au: float
    target_energy_derivative_hartree: float
    photon_energy_derivative_hartree: float
    target_response_residual: float
    target_response_iterations: int
    response_derivative_residual_norms: np.ndarray
    response_derivative_iterations: np.ndarray

    def __post_init__(self) -> None:
        tensor = np.asarray(self.transition_tensor_derivative_au)
        residuals = np.asarray(self.response_derivative_residual_norms, dtype=float)
        iterations = np.asarray(self.response_derivative_iterations, dtype=np.int64)
        if tensor.shape != (3, 3) or residuals.shape != (3,) or iterations.shape != (3,):
            raise ValueError("two-photon derivative dimensions are inconsistent")
        if not np.all(np.isfinite(tensor)) or not np.all(np.isfinite(residuals)):
            raise ValueError("two-photon derivative must be finite")
        if not np.allclose(tensor, tensor.T, atol=2.0e-9, rtol=0.0):
            raise ValueError("two-photon tensor derivative must be symmetric")
        for value in (tensor, residuals, iterations):
            value.setflags(write=False)
        object.__setattr__(self, "transition_tensor_derivative_au", tensor)
        object.__setattr__(self, "response_derivative_residual_norms", residuals)
        object.__setattr__(self, "response_derivative_iterations", iterations)


@dataclass(frozen=True)
class MolecularTDATwoPhotonGradient:
    """Complete Cartesian derivative of a molecular TDA two-photon tensor."""

    response: TDATwoPhotonResult
    tensor_gradient_au_per_bohr: np.ndarray
    strength_gradient_au_per_bohr: np.ndarray
    maximum_response_residual: float
    maximum_cpscf_residual: float

    def __post_init__(self) -> None:
        tensor = np.asarray(self.tensor_gradient_au_per_bohr)
        strength = np.asarray(self.strength_gradient_au_per_bohr, dtype=float)
        if tensor.ndim != 4 or tensor.shape[1:] != (3, 3, 3):
            raise ValueError("molecular two-photon tensor gradient has the wrong shape")
        if strength.shape != tensor.shape[:2]:
            raise ValueError("molecular two-photon strength gradient has the wrong shape")
        if not np.all(np.isfinite(tensor)) or not np.all(np.isfinite(strength)):
            raise ValueError("molecular two-photon gradient must be finite")
        tensor.setflags(write=False)
        strength.setflags(write=False)
        object.__setattr__(self, "tensor_gradient_au_per_bohr", tensor)
        object.__setattr__(self, "strength_gradient_au_per_bohr", strength)


def sparse_configuration_dipole(
    occupied_orbitals: np.ndarray,
    virtual_orbitals: np.ndarray,
    dipole_mo: np.ndarray,
    *,
    singlet_factor: float = np.sqrt(2.0),
    threshold: float = 0.0,
) -> SparseConfigurationDipole:
    """Build the selected-space one-body dipole without a dense CSF tensor.

    The factor ``1/2`` on the excited--excited block is part of the sTDA
    spin-adapted eigenvector convention used by the reference implementation;
    it must not be folded into the ground--excited singlet factor.
    """

    occupied = np.asarray(occupied_orbitals, dtype=np.int64)
    virtual = np.asarray(virtual_orbitals, dtype=np.int64)
    dipole = np.asarray(dipole_mo, dtype=float)
    if occupied.ndim != 1 or virtual.shape != occupied.shape or not len(occupied):
        raise ValueError("TDA dipole construction requires selected configurations")
    if dipole.ndim != 3 or dipole.shape[0] != 3 or dipole.shape[1] != dipole.shape[2]:
        raise ValueError("MO dipole integrals must have shape 3 x nmo x nmo")
    if min(np.min(occupied), np.min(virtual)) < 0 or max(np.max(occupied), np.max(virtual)) >= dipole.shape[1]:
        raise ValueError("TDA configurations lie outside the MO dipole space")
    if threshold < 0.0 or singlet_factor <= 0.0:
        raise ValueError("TDA dipole threshold and singlet factor are invalid")
    ground = float(singlet_factor) * dipole[:, occupied, virtual]
    entries: dict[tuple[int, int], np.ndarray] = {}
    occupied_groups: dict[int, list[int]] = {}
    virtual_groups: dict[int, list[int]] = {}
    for index, (i, a) in enumerate(zip(occupied, virtual, strict=True)):
        occupied_groups.setdefault(int(i), []).append(index)
        virtual_groups.setdefault(int(a), []).append(index)
    for indices in occupied_groups.values():
        for row in indices:
            for column in indices:
                key = (row, column)
                entries[key] = entries.get(key, np.zeros(3)) + dipole[
                    :, virtual[row], virtual[column]
                ]
    for indices in virtual_groups.values():
        for row in indices:
            for column in indices:
                key = (row, column)
                entries[key] = entries.get(key, np.zeros(3)) - dipole[
                    :, occupied[column], occupied[row]
                ]
    retained = tuple(
        (row, column, value)
        for (row, column), value in sorted(entries.items())
        if threshold == 0.0 or float(np.max(np.abs(value))) > float(threshold)
    )
    return SparseConfigurationDipole(
        occupied,
        virtual,
        ground,
        np.asarray([item[0] for item in retained], dtype=np.int64),
        np.asarray([item[1] for item in retained], dtype=np.int64),
        0.5 * np.asarray([item[2] for item in retained], dtype=float).reshape(-1, 3),
    )


def sparse_configuration_dipole_derivative(
    dipole: SparseConfigurationDipole,
    dipole_mo_derivative: np.ndarray,
) -> SparseConfigurationDipole:
    """Differentiate a selected-space dipole on its immutable structural graph."""

    derivative = np.asarray(dipole_mo_derivative, dtype=float)
    if (
        derivative.ndim != 3
        or derivative.shape[0] != 3
        or derivative.shape[1] != derivative.shape[2]
    ):
        raise ValueError("MO dipole derivative must have shape 3 x nmo x nmo")
    occupied = dipole.occupied_orbitals
    virtual = dipole.virtual_orbitals
    if max(np.max(occupied), np.max(virtual)) >= derivative.shape[1]:
        raise ValueError("TDA configurations lie outside the MO dipole derivative")
    ground = np.sqrt(2.0) * derivative[:, occupied, virtual]
    values = np.zeros((dipole.nonzero_count, 3), dtype=float)
    for index, (row, column) in enumerate(
        zip(dipole.rows, dipole.columns, strict=True)
    ):
        i = int(occupied[row])
        a = int(virtual[row])
        j = int(occupied[column])
        b = int(virtual[column])
        if i == j:
            values[index] += derivative[:, a, b]
        if a == b:
            values[index] -= derivative[:, j, i]
    return SparseConfigurationDipole(
        occupied,
        virtual,
        ground,
        dipole.rows,
        dipole.columns,
        0.5 * values,
    )


def _local_orbital_position_element(
    first_offsets: np.ndarray,
    first_rows: np.ndarray,
    first_values: np.ndarray,
    first: int,
    second_offsets: np.ndarray,
    second_rows: np.ndarray,
    second_values: np.ndarray,
    second: int,
    ao_positions: np.ndarray,
) -> np.ndarray:
    first_start, first_stop = first_offsets[first : first + 2]
    second_start, second_stop = second_offsets[second : second + 2]
    left = {
        int(row): float(value)
        for row, value in zip(
            first_rows[first_start:first_stop],
            first_values[first_start:first_stop],
            strict=True,
        )
    }
    value = np.zeros(3)
    for row, coefficient in zip(
        second_rows[second_start:second_stop],
        second_values[second_start:second_stop],
        strict=True,
    ):
        value += left.get(int(row), 0.0) * float(coefficient) * ao_positions[int(row)]
    return -value


def sparse_local_configuration_dipole(
    state: GFN2SparseProjectorState,
    basis: object,
    space: object,
    *,
    threshold: float = 1.0e-12,
) -> SparseConfigurationDipole:
    """Build a point-multipole dipole on the localized sparse sTDA graph."""

    occupied_basis = basis.occupied
    virtual_basis = basis.virtual
    occupied = np.asarray(space.occupied_orbitals, dtype=np.int64)
    virtual = np.asarray(space.virtual_orbitals, dtype=np.int64)
    ao_positions = np.asarray(state.coordinates_bohr, dtype=float)[state.ao_atom]
    ground = np.empty((3, len(occupied)))
    for index, (i, a) in enumerate(zip(occupied, virtual, strict=True)):
        ground[:, index] = np.sqrt(2.0) * _local_orbital_position_element(
            occupied_basis.offsets,
            occupied_basis.rows,
            occupied_basis.values,
            int(i),
            virtual_basis.offsets,
            virtual_basis.rows,
            virtual_basis.values,
            int(a),
            ao_positions,
        )
    entries: dict[tuple[int, int], np.ndarray] = {}
    occupied_groups: dict[int, list[int]] = {}
    virtual_groups: dict[int, list[int]] = {}
    for index, (i, a) in enumerate(zip(occupied, virtual, strict=True)):
        occupied_groups.setdefault(int(i), []).append(index)
        virtual_groups.setdefault(int(a), []).append(index)
    for indices in occupied_groups.values():
        for row in indices:
            for column in indices:
                key = (row, column)
                entries[key] = entries.get(key, np.zeros(3)) + _local_orbital_position_element(
                    virtual_basis.offsets,
                    virtual_basis.rows,
                    virtual_basis.values,
                    int(virtual[row]),
                    virtual_basis.offsets,
                    virtual_basis.rows,
                    virtual_basis.values,
                    int(virtual[column]),
                    ao_positions,
                )
    for indices in virtual_groups.values():
        for row in indices:
            for column in indices:
                key = (row, column)
                entries[key] = entries.get(key, np.zeros(3)) - _local_orbital_position_element(
                    occupied_basis.offsets,
                    occupied_basis.rows,
                    occupied_basis.values,
                    int(occupied[column]),
                    occupied_basis.offsets,
                    occupied_basis.rows,
                    occupied_basis.values,
                    int(occupied[row]),
                    ao_positions,
                )
    retained = tuple(
        (row, column, value)
        for (row, column), value in sorted(entries.items())
        if float(np.max(np.abs(value))) > float(threshold)
    )
    return SparseConfigurationDipole(
        occupied,
        virtual,
        ground,
        np.asarray([item[0] for item in retained], dtype=np.int64),
        np.asarray([item[1] for item in retained], dtype=np.int64),
        0.5 * np.asarray([item[2] for item in retained], dtype=float).reshape(-1, 3),
    )


def molecular_dipole_mo(state: GFN2OrbitalState) -> np.ndarray:
    """Evaluate electronic dipole integrals in the resident canonical MO basis."""

    orbitals = _cartesian_basis_functions(state)
    _, dipole_cartesian, _ = atomic_multipole_integrals(orbitals)
    transform = state.basis.spherical_to_cartesian
    dipole_ao = np.einsum(
        "pi,xpq,qj->xij", transform, -dipole_cartesian, transform, optimize=True
    )
    return np.einsum(
        "pi,xpq,qj->xij",
        state.coefficients,
        dipole_ao,
        state.coefficients,
        optimize=True,
    )


def _cartesian_basis_functions(
    state: GFN2OrbitalState,
) -> tuple[ContractedGaussian, ...]:
    angular = {
        1: (0, 0, 0),
        2: (1, 0, 0),
        3: (0, 1, 0),
        4: (0, 0, 1),
        5: (2, 0, 0),
        6: (0, 2, 0),
        7: (0, 0, 2),
        8: (1, 1, 0),
        9: (1, 0, 1),
        10: (0, 1, 1),
    }
    basis = state.basis
    orbitals = []
    for index, (atom, component) in enumerate(
        zip(basis.cartesian_atom, basis.cartesian_type, strict=True)
    ):
        start = int(basis.primitive_offsets[index])
        stop = int(basis.primitive_offsets[index + 1])
        primitives = tuple(
            GaussianPrimitive(float(exponent), float(coefficient))
            for exponent, coefficient in zip(
                basis.primitive_exponents[start:stop],
                basis.primitive_coefficients[start:stop],
                strict=True,
            )
        )
        orbitals.append(
            ContractedGaussian(
                tuple(state.coordinates_bohr[int(atom)]),
                angular[int(component)],
                primitives,
                int(atom),
                "aux",
                1.0,
            )
        )
    return tuple(orbitals)


def molecular_dipole_mo_derivative(
    state: GFN2OrbitalState,
    coefficient_derivative: np.ndarray,
) -> np.ndarray:
    """Return analytic Cartesian derivatives of canonical-MO dipole integrals."""

    dc = np.asarray(coefficient_derivative, dtype=float)
    expected = (len(state.atomic_numbers), 3, state.nao, state.nmo)
    if dc.shape != expected:
        raise ValueError("MO dipole coefficient derivative has the wrong shape")
    orbitals = _cartesian_basis_functions(state)
    _, dipole_cartesian, _ = atomic_multipole_integrals(orbitals)
    _, dipole_cartesian_derivative, _ = atomic_multipole_integral_derivatives(
        orbitals
    )
    transform = state.basis.spherical_to_cartesian
    dipole_ao = np.einsum(
        "pi,xpq,qj->xij", transform, -dipole_cartesian, transform, optimize=True
    )
    dipole_ao_derivative = np.einsum(
        "pi,xakpq,qj->xakij",
        transform,
        -dipole_cartesian_derivative,
        transform,
        optimize=True,
    )
    coefficients = state.coefficients
    derivative = np.einsum(
        "akpi,xpq,qj->akxij", dc, dipole_ao, coefficients, optimize=True
    )
    derivative += np.einsum(
        "pi,xakpq,qj->akxij",
        coefficients,
        dipole_ao_derivative,
        coefficients,
        optimize=True,
    )
    derivative += np.einsum(
        "pi,xpq,akqj->akxij", coefficients, dipole_ao, dc, optimize=True
    )
    return np.moveaxis(derivative, 2, 0)


def _gmres(
    apply: VectorOperator,
    right_hand_side: np.ndarray,
    *,
    tolerance: float,
    maximum_iterations: int,
    diagonal: np.ndarray | None,
) -> tuple[np.ndarray, float, int]:
    rhs = np.asarray(right_hand_side)
    norm = float(np.linalg.norm(rhs))
    if norm == 0.0:
        return np.zeros_like(rhs), 0.0, 0
    limit = min(len(rhs), int(maximum_iterations))
    if limit < 1 or tolerance <= 0.0:
        raise ValueError("two-photon response solver controls must be positive")
    if diagonal is None:
        preconditioner = np.ones_like(rhs)
    else:
        values = np.asarray(diagonal)
        if values.shape != rhs.shape:
            raise ValueError("two-photon response diagonal has the wrong dimension")
        floor = max(float(tolerance), 1.0e-12)
        if np.iscomplexobj(values):
            replacement = floor * np.exp(1j * np.angle(values))
        else:
            replacement = np.copysign(floor, values + floor)
        preconditioner = 1.0 / np.where(
            np.abs(values) > floor, values, replacement
        )

    def transformed(vector: np.ndarray) -> np.ndarray:
        return np.asarray(apply(preconditioner * vector))

    dtype = np.result_type(rhs.dtype, preconditioner.dtype)
    basis = np.zeros((len(rhs), limit + 1), dtype=dtype)
    hessenberg = np.zeros((limit + 1, limit), dtype=dtype)
    basis[:, 0] = rhs / norm
    for column in range(limit):
        vector = transformed(basis[:, column])
        active = basis[:, : column + 1]
        projection = active.conj().T @ vector
        hessenberg[: column + 1, column] = projection
        vector -= active @ projection
        correction = active.conj().T @ vector
        hessenberg[: column + 1, column] += correction
        vector -= active @ correction
        hessenberg[column + 1, column] = np.linalg.norm(vector)
        if hessenberg[column + 1, column] > np.finfo(float).eps:
            basis[:, column + 1] = vector / hessenberg[column + 1, column]
        target = np.zeros(column + 2, dtype=dtype)
        target[0] = norm
        coefficients = np.linalg.lstsq(
            hessenberg[: column + 2, : column + 1], target, rcond=None
        )[0]
        solution = preconditioner * (basis[:, : column + 1] @ coefficients)
        residual = float(np.linalg.norm(apply(solution) - rhs))
        if residual <= tolerance * max(1.0, norm):
            return solution, residual, column + 1
    raise RuntimeError(
        f"matrix-free two-photon response did not converge in {limit} iterations"
    )


def tda_two_photon_response(
    hamiltonian_apply: VectorOperator,
    dipole: SparseConfigurationDipole,
    target_amplitude: np.ndarray,
    target_energy_hartree: float,
    *,
    photon_energy_hartree: float | None = None,
    hamiltonian_diagonal_hartree: np.ndarray | None = None,
    tolerance: float = 1.0e-10,
    maximum_iterations: int = 200,
    resonance_tolerance_hartree: float = 1.0e-7,
    damping_hartree: float = 0.0,
    target_state: int = 0,
) -> TDATwoPhotonResult:
    """Solve the degenerate two-photon response for one TDA target state."""

    target = np.asarray(target_amplitude, dtype=float)
    if target.shape != (dipole.size,) or not np.all(np.isfinite(target)):
        raise ValueError("two-photon target amplitude has the wrong dimension")
    target_norm = float(np.linalg.norm(target))
    if target_norm <= np.finfo(float).tiny:
        raise ValueError("two-photon target amplitude has zero norm")
    target = target / target_norm
    energy = float(target_energy_hartree)
    photon = 0.5 * energy if photon_energy_hartree is None else float(photon_energy_hartree)
    damping = float(damping_hartree)
    if (
        energy <= 0.0
        or photon <= 0.0
        or resonance_tolerance_hartree <= 0.0
        or damping < 0.0
    ):
        raise ValueError("two-photon energies and resonance tolerance must be positive")
    diagonal = None
    if hamiltonian_diagonal_hartree is not None:
        diagonal = np.asarray(hamiltonian_diagonal_hartree, dtype=float) - photon
        if damping > 0.0:
            diagonal = diagonal.astype(complex) - 1j * damping
        margin = float(np.min(np.abs(diagonal)))
    else:
        margin = float("inf")
    if damping == 0.0 and margin <= resonance_tolerance_hartree:
        raise ValueError("two-photon frequency is resonant with an intermediate state")

    def shifted(vector: np.ndarray) -> np.ndarray:
        shift = photon + 1j * damping if damping > 0.0 else photon
        value = np.asarray(hamiltonian_apply(vector)) - shift * vector
        if value.shape != (dipole.size,) or not np.all(np.isfinite(value)):
            raise ValueError("TDA Hamiltonian returned an invalid response vector")
        return value

    response = []
    residuals = []
    iterations = []
    for component in range(3):
        solution, residual, count = _gmres(
            shifted,
            dipole.ground_vectors[component],
            tolerance=float(tolerance),
            maximum_iterations=int(maximum_iterations),
            diagonal=diagonal,
        )
        response.append(solution)
        residuals.append(residual)
        iterations.append(count)
    tensor = np.empty((3, 3), dtype=np.result_type(*response))
    for first in range(3):
        for second in range(3):
            tensor[first, second] = (
                target @ dipole.apply(first, response[second])
                + target @ dipole.apply(second, response[first])
            )
    tensor = 0.5 * (tensor + tensor.T)
    trace = np.trace(tensor)
    strength = float(
        (abs(trace) ** 2 + 2.0 * np.sum(np.abs(tensor) ** 2)) / 15.0
    )
    return TDATwoPhotonResult(
        int(target_state),
        photon,
        tensor,
        strength,
        np.asarray(residuals),
        np.asarray(iterations),
        margin,
        damping,
    )


def tda_two_photon_directional_derivative(
    hamiltonian_apply: VectorOperator,
    hamiltonian_derivative_apply: VectorOperator,
    dipole: SparseConfigurationDipole,
    dipole_derivative: SparseConfigurationDipole,
    target_amplitude: np.ndarray,
    target_energy_hartree: float,
    *,
    photon_energy_hartree: float | None = None,
    photon_energy_derivative_hartree: float | None = None,
    hamiltonian_diagonal_hartree: np.ndarray | None = None,
    tolerance: float = 1.0e-10,
    maximum_iterations: int = 200,
    damping_hartree: float = 0.0,
    target_state: int = 0,
) -> TDATwoPhotonDirectionalDerivative:
    """Differentiate the sparse TDA two-photon response in one direction.

    The derivative contains the target-root response, dipole derivative, and
    differentiated shifted resolvent.  Molecular use supplies the directional
    derivatives of the selected TDA Hamiltonian and dipole after AO, SCC,
    multipolar, and orbital-response contraction; no dense inverse is formed.
    """

    if dipole_derivative.size != dipole.size:
        raise ValueError("two-photon dipole derivative has the wrong dimension")
    if not (
        np.array_equal(dipole_derivative.rows, dipole.rows)
        and np.array_equal(dipole_derivative.columns, dipole.columns)
    ):
        raise ValueError("two-photon dipole derivative must preserve the sparse graph")
    target = np.array(target_amplitude, dtype=float, copy=True)
    if target.shape != (dipole.size,):
        raise ValueError("two-photon target amplitude has the wrong dimension")
    target_norm = float(np.linalg.norm(target))
    if target_norm <= np.finfo(float).tiny:
        raise ValueError("two-photon target amplitude has zero norm")
    target /= target_norm
    energy = float(target_energy_hartree)
    h_target = np.asarray(hamiltonian_derivative_apply(target), dtype=float)
    if h_target.shape != target.shape:
        raise ValueError("TDA Hamiltonian derivative returned an invalid vector")
    energy_derivative = float(target @ h_target)
    photon = 0.5 * energy if photon_energy_hartree is None else float(photon_energy_hartree)
    if photon_energy_derivative_hartree is None:
        photon_derivative = 0.5 * energy_derivative if photon_energy_hartree is None else 0.0
    else:
        photon_derivative = float(photon_energy_derivative_hartree)
    damping = float(damping_hartree)
    shift = photon + 1j * damping if damping > 0.0 else photon
    diagonal = None
    if hamiltonian_diagonal_hartree is not None:
        diagonal = np.asarray(hamiltonian_diagonal_hartree, dtype=float) - photon
        if damping > 0.0:
            diagonal = diagonal.astype(complex) - 1j * damping

    def shifted(vector: np.ndarray) -> np.ndarray:
        return np.asarray(hamiltonian_apply(vector)) - shift * vector

    def project(vector: np.ndarray) -> np.ndarray:
        value = np.asarray(vector, dtype=float)
        return value - target * float(target @ value)

    def target_operator(vector: np.ndarray) -> np.ndarray:
        parallel = target * float(target @ vector)
        orthogonal = project(vector)
        return project(np.asarray(hamiltonian_apply(orthogonal)) - energy * orthogonal) + parallel

    target_rhs = -project(h_target - energy_derivative * target)
    target_derivative, target_residual, target_iterations = _gmres(
        target_operator,
        target_rhs,
        tolerance=float(tolerance),
        maximum_iterations=int(maximum_iterations),
        diagonal=None,
    )
    target_derivative = project(np.real_if_close(target_derivative))

    responses = []
    response_derivatives = []
    base_residuals = []
    base_iterations = []
    derivative_residuals = []
    derivative_iterations = []
    for component in range(3):
        response, residual, iterations = _gmres(
            shifted,
            dipole.ground_vectors[component],
            tolerance=float(tolerance),
            maximum_iterations=int(maximum_iterations),
            diagonal=diagonal,
        )
        derivative_rhs = (
            dipole_derivative.ground_vectors[component]
            - np.asarray(hamiltonian_derivative_apply(response))
            + photon_derivative * response
        )
        response_derivative, derivative_residual, derivative_count = _gmres(
            shifted,
            derivative_rhs,
            tolerance=float(tolerance),
            maximum_iterations=int(maximum_iterations),
            diagonal=diagonal,
        )
        responses.append(response)
        response_derivatives.append(response_derivative)
        base_residuals.append(residual)
        base_iterations.append(iterations)
        derivative_residuals.append(derivative_residual)
        derivative_iterations.append(derivative_count)

    tensor = np.empty((3, 3), dtype=np.result_type(*responses))
    tensor_derivative = np.empty_like(tensor)
    for first in range(3):
        for second in range(3):
            first_action = dipole.apply(first, responses[second])
            second_action = dipole.apply(second, responses[first])
            tensor[first, second] = target @ (first_action + second_action)
            tensor_derivative[first, second] = (
                target_derivative @ (first_action + second_action)
                + target
                @ (
                    dipole_derivative.apply(first, responses[second])
                    + dipole.apply(first, response_derivatives[second])
                    + dipole_derivative.apply(second, responses[first])
                    + dipole.apply(second, response_derivatives[first])
                )
            )
    tensor = 0.5 * (tensor + tensor.T)
    tensor_derivative = 0.5 * (tensor_derivative + tensor_derivative.T)
    trace = np.trace(tensor)
    trace_derivative = np.trace(tensor_derivative)
    strength = float(
        (abs(trace) ** 2 + 2.0 * np.sum(np.abs(tensor) ** 2)) / 15.0
    )
    strength_derivative = float(
        (
            2.0 * np.real(np.conj(trace) * trace_derivative)
            + 4.0 * np.real(np.sum(np.conj(tensor) * tensor_derivative))
        )
        / 15.0
    )
    margin = float("inf") if diagonal is None else float(np.min(np.abs(diagonal)))
    response = TDATwoPhotonResult(
        int(target_state),
        photon,
        tensor,
        strength,
        np.asarray(base_residuals),
        np.asarray(base_iterations),
        margin,
        damping,
    )
    return TDATwoPhotonDirectionalDerivative(
        response,
        tensor_derivative,
        strength_derivative,
        energy_derivative,
        photon_derivative,
        target_residual,
        target_iterations,
        np.asarray(derivative_residuals),
        np.asarray(derivative_iterations),
    )


def molecular_tda_two_photon_gradient(
    calculator: object,
    state: GFN2OrbitalState,
    excited: object,
    target_state: int,
    *,
    spectral_range_ev: float = 7.0,
    maximum_configuration_energy_ev: float | None = None,
    tolerance: float = 1.0e-9,
    maximum_iterations: int = 200,
    damping_hartree: float = 0.0,
) -> MolecularTDATwoPhotonGradient:
    """Assemble the analytic Cartesian 2PA gradient for resident TDA-xTB.

    AO dipole derivatives, Pulay terms, and the fully coupled orbital/SCC
    response are differentiated once.  The selected configuration set is held
    fixed, exactly as for analytic excited-state gradients and Hessians.
    """

    from .stda import closed_shell_stda_matrix_derivative

    root = int(target_state)
    energies = np.asarray(excited.excitation_energies_hartree, dtype=float)
    amplitudes = np.asarray(excited.amplitudes, dtype=float)
    if root < 0 or root >= len(energies) or amplitudes.ndim != 2:
        raise ValueError("molecular two-photon target state is unavailable")
    derivative_matrices = calculator.ground_state_derivative_matrices(state)
    overlap_derivative, _ = derivative_matrices.validated(state)
    coupled = calculator.ground_state_coupled_response(state)
    matrix_data = closed_shell_stda_matrix_derivative(
        state,
        excited,
        overlap_derivative,
        coupled.coefficient_derivative,
        coupled.orbital_energy_derivative,
        spectral_range_ev=float(spectral_range_ev),
        maximum_configuration_energy_ev=maximum_configuration_energy_ev,
    )
    if matrix_data.matrix_hartree.shape != (amplitudes.shape[1],) * 2:
        raise ValueError("molecular TDA derivative and root spaces differ")
    eigen_residual = np.linalg.norm(
        matrix_data.matrix_hartree @ amplitudes[root] - energies[root] * amplitudes[root]
    )
    if eigen_residual > 2.0e-7:
        raise ValueError("reconstructed TDA matrix does not reproduce the target root")

    dipole_mo = molecular_dipole_mo(state)
    dipole = sparse_configuration_dipole(
        excited.occupied_orbitals,
        excited.virtual_orbitals,
        dipole_mo,
    )
    dipole_derivative_mo = molecular_dipole_mo_derivative(
        state, coupled.coefficient_derivative
    )
    tensor_gradient = np.empty(
        (len(state.atomic_numbers), 3, 3, 3), dtype=complex if damping_hartree else float
    )
    strength_gradient = np.empty((len(state.atomic_numbers), 3), dtype=float)

    def apply(vector: np.ndarray) -> np.ndarray:
        return matrix_data.matrix_hartree @ vector

    target = np.array(amplitudes[root], dtype=float, copy=True)
    target /= np.linalg.norm(target)
    energy = float(energies[root])
    photon = 0.5 * energy
    damping = float(damping_hartree)
    shift = photon + 1j * damping if damping > 0.0 else photon
    diagonal = np.diag(matrix_data.matrix_hartree) - photon
    if damping > 0.0:
        diagonal = diagonal.astype(complex) - 1j * damping
    margin = float(np.min(np.abs(diagonal)))
    if damping == 0.0 and margin <= 1.0e-7:
        raise ValueError("two-photon frequency is resonant with an intermediate state")

    def shifted(vector: np.ndarray) -> np.ndarray:
        return apply(vector) - shift * vector

    def project(vector: np.ndarray) -> np.ndarray:
        value = np.asarray(vector)
        return value - target * (target @ value)

    def target_operator(vector: np.ndarray) -> np.ndarray:
        parallel = target * (target @ vector)
        orthogonal = project(vector)
        return project(apply(orthogonal) - energy * orthogonal) + parallel

    # Three physical response vectors, three shifted adjoints, and six
    # symmetric root adjoints are independent of nuclear direction.  Their
    # reuse replaces seven Krylov solves per Cartesian component by a fixed
    # twelve-solve setup followed only by sparse derivative contractions.
    responses: list[np.ndarray] = []
    response_residuals: list[float] = []
    response_iterations: list[int] = []
    shifted_adjoints: list[np.ndarray] = []
    adjoint_residuals: list[float] = []
    for component in range(3):
        value, residual, iterations = _gmres(
            shifted,
            dipole.ground_vectors[component],
            tolerance=float(tolerance),
            maximum_iterations=int(maximum_iterations),
            diagonal=diagonal,
        )
        responses.append(value)
        response_residuals.append(residual)
        response_iterations.append(iterations)
        adjoint, adjoint_residual, _ = _gmres(
            shifted,
            dipole.apply(component, target),
            tolerance=float(tolerance),
            maximum_iterations=int(maximum_iterations),
            diagonal=diagonal,
        )
        shifted_adjoints.append(adjoint)
        adjoint_residuals.append(adjoint_residual)

    tensor = np.empty((3, 3), dtype=np.result_type(*responses))
    root_adjoints: dict[tuple[int, int], np.ndarray] = {}
    root_adjoint_residuals: list[float] = []
    for first in range(3):
        for second in range(first, 3):
            contraction = dipole.apply(first, responses[second]) + dipole.apply(
                second, responses[first]
            )
            tensor[first, second] = tensor[second, first] = target @ contraction
            adjoint, residual, _ = _gmres(
                target_operator,
                project(contraction),
                tolerance=float(tolerance),
                maximum_iterations=int(maximum_iterations),
                diagonal=None,
            )
            root_adjoints[(first, second)] = project(adjoint)
            root_adjoint_residuals.append(residual)

    trace = np.trace(tensor)
    strength = float(
        (abs(trace) ** 2 + 2.0 * np.sum(np.abs(tensor) ** 2)) / 15.0
    )
    response = TDATwoPhotonResult(
        root,
        photon,
        tensor,
        strength,
        np.asarray(response_residuals),
        np.asarray(response_iterations),
        margin,
        damping,
    )
    maximum_residual = max(
        response_residuals + adjoint_residuals + root_adjoint_residuals
    )

    for atom in range(len(state.atomic_numbers)):
        for component in range(3):
            dipole_derivative = sparse_configuration_dipole_derivative(
                dipole, dipole_derivative_mo[:, atom, component]
            )
            matrix_target = matrix_data.apply_derivative(atom, component, target)
            matrix_responses = tuple(
                matrix_data.apply_derivative(atom, component, response_vector)
                for response_vector in responses
            )
            energy_derivative = float(target @ matrix_target)
            photon_derivative = 0.5 * energy_derivative
            derivative = np.empty_like(tensor)
            for first in range(3):
                for second in range(first, 3):
                    explicit = target @ (
                        dipole_derivative.apply(first, responses[second])
                        + dipole_derivative.apply(second, responses[first])
                    )
                    first_rhs = (
                        dipole_derivative.ground_vectors[second]
                        - matrix_responses[second]
                        + photon_derivative * responses[second]
                    )
                    second_rhs = (
                        dipole_derivative.ground_vectors[first]
                        - matrix_responses[first]
                        + photon_derivative * responses[first]
                    )
                    response_term = (
                        shifted_adjoints[first] @ first_rhs
                        + shifted_adjoints[second] @ second_rhs
                    )
                    root_term = -(root_adjoints[(first, second)] @ matrix_target)
                    value = explicit + response_term + root_term
                    derivative[first, second] = derivative[second, first] = value
            tensor_gradient[atom, component] = derivative
            trace_derivative = np.trace(derivative)
            strength_gradient[atom, component] = float(
                (
                    2.0 * np.real(np.conj(trace) * trace_derivative)
                    + 4.0 * np.real(np.sum(np.conj(tensor) * derivative))
                )
                / 15.0
            )
    return MolecularTDATwoPhotonGradient(
        response,
        tensor_gradient,
        strength_gradient,
        maximum_residual,
        float(np.max(coupled.residual_norms)),
    )


def linear_scaling_tda_two_photon(
    state: GFN2SparseProjectorState,
    basis: object,
    space: object,
    operator: object,
    result: object,
    target_state: int,
    *,
    tolerance: float = 1.0e-10,
    maximum_iterations: int = 200,
    damping_hartree: float = 0.0,
) -> TDATwoPhotonResult:
    """Evaluate two-photon response on the pair-packed linear-scaling path."""

    root = int(target_state)
    energies = np.asarray(result.excitation_energies_hartree, dtype=float)
    amplitudes = np.asarray(result.amplitudes, dtype=float)
    if root < 0 or root >= len(energies) or amplitudes.shape[1] != int(space.size):
        raise ValueError("linear-scaling two-photon target is outside the TDA space")
    if int(operator.size) != int(space.size):
        raise ValueError("linear-scaling TDA operator and excitation space differ")
    dipole = sparse_local_configuration_dipole(state, basis, space)
    return tda_two_photon_response(
        operator.apply,
        dipole,
        amplitudes[root],
        energies[root],
        hamiltonian_diagonal_hartree=operator.diagonal_hartree,
        tolerance=float(tolerance),
        maximum_iterations=int(maximum_iterations),
        damping_hartree=float(damping_hartree),
        target_state=root,
    )


def qualify_external_two_photon(
    qualification: Mapping[str, object],
) -> dict[str, object]:
    """Fail closed on an incomplete or non-reproducible external 2PA record."""

    payload = dict(qualification)
    if payload.get("schema") != (
        "matrix.fast_tda.two_photon_external_qualification.v1"
    ):
        raise ValueError("unsupported FAST_TDA external two-photon qualification")
    keymaker = payload.get("keymaker")
    oracle = payload.get("external_oracle")
    states = payload.get("states")
    if not isinstance(keymaker, Mapping) or keymaker.get("protocol_id") != (
        "fast-tda-two-photon-qualification.v1"
    ):
        raise ValueError("external two-photon qualification lacks KeyMaker provenance")
    if not isinstance(oracle, Mapping) or not isinstance(states, list) or not states:
        raise ValueError("external two-photon qualification is incomplete")
    for field in (
        "sos_repeat_max_absolute_residual",
        "response_repeat_max_absolute_residual",
    ):
        if float(oracle.get(field, float("inf"))) > 1.0e-12:
            raise ValueError("external two-photon oracle is not reproducible")
    if float(payload.get("maximum_tda_energy_residual_ev", float("inf"))) > 6.0e-5:
        raise ValueError("ordinary TDA roots fail the same-reference parity gate")
    return payload


def qualify_external_two_photon_corpus(
    qualification: Mapping[str, object],
) -> dict[str, object]:
    """Fail closed on an incomplete pure-TDA molecular corpus."""

    payload = dict(qualification)
    if payload.get("schema") != "matrix.fast_tda.two_photon_corpus.v1":
        raise ValueError("unsupported FAST_TDA two-photon corpus")
    systems = payload.get("systems")
    if not isinstance(systems, list) or not systems:
        raise ValueError("FAST_TDA two-photon corpus is empty")
    if float(payload.get("maximum_energy_residual_ev", float("inf"))) > 6.0e-5:
        raise ValueError("FAST_TDA corpus fails its energy gate")
    if float(payload.get("maximum_tensor_relative_residual", float("inf"))) > 6.0e-5:
        raise ValueError("FAST_TDA corpus fails its tensor gate")
    return payload


__all__ = [
    "MolecularTDATwoPhotonGradient",
    "SparseConfigurationDipole",
    "TDATwoPhotonDirectionalDerivative",
    "TDATwoPhotonResult",
    "molecular_dipole_mo",
    "molecular_dipole_mo_derivative",
    "molecular_tda_two_photon_gradient",
    "sparse_configuration_dipole",
    "sparse_configuration_dipole_derivative",
    "sparse_local_configuration_dipole",
    "linear_scaling_tda_two_photon",
    "qualify_external_two_photon",
    "qualify_external_two_photon_corpus",
    "tda_two_photon_directional_derivative",
    "tda_two_photon_response",
]
