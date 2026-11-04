"""Cartesian Gaussian integral kernel for MATRIX-GFN2.

The kernel is implemented directly from Gaussian product recurrences.  It has
no dependency on an external semiempirical, SCF, or integral engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from math import comb
from math import exp, pi, sqrt
from typing import Sequence

import numpy as np
from ._array_cache import triangular_indices


@dataclass(frozen=True)
class SparseAtomicMultipoleIntegrals:
    """Pair-packed AO multipoles and centre derivatives.

    No dense ``nao x nao`` allocation is made.  ``left_derivatives`` and
    ``right_derivatives`` store the 13 integral components by three Cartesian
    centre derivatives, matching the fused native kernel.
    """

    size: int
    rows: np.ndarray
    columns: np.ndarray
    overlap: np.ndarray
    dipole: np.ndarray
    second: np.ndarray
    left_derivatives: np.ndarray
    right_derivatives: np.ndarray
    threshold_discarded_frobenius_norm: float = 0.0
    domain_complete: bool = False

    def sparse_overlap(self):
        from .localized_solver import SparseSymmetricMatrix

        return SparseSymmetricMatrix(
            self.size, self.rows, self.columns, self.overlap
        )

try:
    from . import _gfn2_native
except ImportError:  # pragma: no cover - source tree without compiled extension
    _gfn2_native = None

# At 20 bohr even the most diffuse GFN2 primitive-pair overlap envelope is
# below 4e-15.  Screening these numerical zeros avoids quadratic zero-work
# without introducing an empirical energy cutoff.
_AO_PAIR_CUTOFF_BOHR = 20.0

# Numba is only a fallback for source-tree installations without the native
# kernel.  Importing it when the extension is present adds a large cold-start
# and JIT cost without executing any of its kernels.
if _gfn2_native is None:
    try:
        from numba import njit
    except ImportError:  # pragma: no cover - portable pure-Python fallback
        njit = None
else:
    njit = None


if njit is not None:
    @njit(cache=True)
    def _overlap_1d_compiled(lp, rp, lc, rc, la, ra):
        total = la + ra
        reduced = la * ra / total
        product = (la * lc + ra * rc) / total
        table = np.zeros((8, 8))
        table[0, 0] = np.sqrt(np.pi / total) * np.exp(-reduced * (lc - rc) ** 2)
        for left_power in range(1, lp + 1):
            table[left_power, 0] = (product - lc) * table[left_power - 1, 0]
            if left_power > 1:
                table[left_power, 0] += (left_power - 1) * table[left_power - 2, 0] / (2.0 * total)
        for right_power in range(1, rp + 1):
            table[0, right_power] = (product - rc) * table[0, right_power - 1]
            if right_power > 1:
                table[0, right_power] += (right_power - 1) * table[0, right_power - 2] / (2.0 * total)
        for left_power in range(1, lp + 1):
            for right_power in range(1, rp + 1):
                table[left_power, right_power] = (
                    (product - rc) * table[left_power, right_power - 1]
                    + left_power * table[left_power - 1, right_power - 1] / (2.0 * total)
                )
                if right_power > 1:
                    table[left_power, right_power] += (
                        (right_power - 1)
                        * table[left_power, right_power - 2]
                        / (2.0 * total)
                    )
        return table[lp, rp]

    @njit(cache=True)
    def _primitive_normalization_compiled(exponent, angular):
        angular_sum = angular[0] + angular[1] + angular[2]
        denominator_product = 1.0
        for axis in range(3):
            value = 2 * angular[axis] - 1
            factor = 1
            while value > 0:
                factor *= value
                value -= 2
            denominator_product *= factor
        return (
            (2.0 * exponent / np.pi) ** 0.75
            * (4.0 * exponent) ** (angular_sum / 2.0)
            / np.sqrt(denominator_product)
        )

    @njit(cache=True)
    def _axis_moments_compiled(lp, rp, lc, rc, la, ra, relative):
        overlap0 = _overlap_1d_compiled(lp, rp, lc, rc, la, ra)
        overlap1 = _overlap_1d_compiled(lp + 1, rp, lc, rc, la, ra)
        overlap2 = _overlap_1d_compiled(lp + 2, rp, lc, rc, la, ra)
        return (
            overlap0,
            relative * overlap0 + overlap1,
            relative * relative * overlap0 + 2.0 * relative * overlap1 + overlap2,
        )

    @njit(cache=True)
    def _contracted_multipole_derivatives_compiled(
        left_center,
        right_center,
        left_angular,
        right_angular,
        left_exponents,
        left_coefficients,
        right_exponents,
        right_coefficients,
        origin,
        denominator,
    ):
        moments = np.array(
            (
                (0, 0, 0),
                (1, 0, 0), (0, 1, 0), (0, 0, 1),
                (2, 0, 0), (1, 1, 0), (1, 0, 1),
                (1, 1, 0), (0, 2, 0), (0, 1, 1),
                (1, 0, 1), (0, 1, 1), (0, 0, 2),
            ),
            dtype=np.int64,
        )
        values = np.zeros(13)
        left_derivatives = np.zeros((13, 3))
        for left_index in range(len(left_exponents)):
            la = left_exponents[left_index]
            left_norm = _primitive_normalization_compiled(la, left_angular)
            for right_index in range(len(right_exponents)):
                ra = right_exponents[right_index]
                prefactor = (
                    left_coefficients[left_index]
                    * right_coefficients[right_index]
                    * left_norm
                    * _primitive_normalization_compiled(ra, right_angular)
                )
                base = np.empty((3, 3))
                raised = np.empty((3, 3))
                lowered = np.zeros((3, 3))
                for axis in range(3):
                    relative = left_center[axis] - origin[axis]
                    base[axis] = _axis_moments_compiled(
                        left_angular[axis], right_angular[axis],
                        left_center[axis], right_center[axis], la, ra, relative,
                    )
                    raised[axis] = _axis_moments_compiled(
                        left_angular[axis] + 1, right_angular[axis],
                        left_center[axis], right_center[axis], la, ra, relative,
                    )
                    if left_angular[axis] > 0:
                        lowered[axis] = _axis_moments_compiled(
                            left_angular[axis] - 1, right_angular[axis],
                            left_center[axis], right_center[axis], la, ra, relative,
                        )
                for moment_index in range(13):
                    m0, m1, m2 = moments[moment_index]
                    values[moment_index] += prefactor * base[0, m0] * base[1, m1] * base[2, m2]
                    for axis in range(3):
                        if axis == 0:
                            other = base[1, m1] * base[2, m2]
                            mp = m0
                        elif axis == 1:
                            other = base[0, m0] * base[2, m2]
                            mp = m1
                        else:
                            other = base[0, m0] * base[1, m1]
                            mp = m2
                        left_derivatives[moment_index, axis] += prefactor * other * (
                            2.0 * la * raised[axis, mp]
                            - left_angular[axis] * lowered[axis, mp]
                        )
        values /= denominator
        left_derivatives /= denominator
        right_derivatives = np.empty((13, 3))
        right_derivatives[0] = -left_derivatives[0]
        for component in range(3):
            for axis in range(3):
                right_derivatives[1 + component, axis] = (
                    (values[0] if component == axis else 0.0)
                    - left_derivatives[1 + component, axis]
                )
        for first in range(3):
            for second_component in range(3):
                index = 4 + 3 * first + second_component
                for axis in range(3):
                    right_derivatives[index, axis] = (
                        (values[1 + second_component] if first == axis else 0.0)
                        + (values[1 + first] if second_component == axis else 0.0)
                        - left_derivatives[index, axis]
                    )
        return values, left_derivatives, right_derivatives

    @njit(cache=True)
    def _multipole_pair_batch_compiled(
        centers,
        angular,
        exponents,
        coefficients,
        primitive_counts,
        pair_left,
        pair_right,
        norms,
        origin,
    ):
        pair_count = len(pair_left)
        values = np.empty((pair_count, 13))
        left_derivatives = np.empty((pair_count, 13, 3))
        right_derivatives = np.empty((pair_count, 13, 3))
        for pair in range(pair_count):
            left = pair_left[pair]
            right = pair_right[pair]
            left_count = primitive_counts[left]
            right_count = primitive_counts[right]
            pair_values, pair_left_derivatives, pair_right_derivatives = (
                _contracted_multipole_derivatives_compiled(
                    centers[left],
                    centers[right],
                    angular[left],
                    angular[right],
                    exponents[left, :left_count],
                    coefficients[left, :left_count],
                    exponents[right, :right_count],
                    coefficients[right, :right_count],
                    origin,
                    np.sqrt(norms[left] * norms[right]),
                )
            )
            values[pair] = pair_values
            left_derivatives[pair] = pair_left_derivatives
            right_derivatives[pair] = pair_right_derivatives
        return values, left_derivatives, right_derivatives
else:
    _contracted_multipole_derivatives_compiled = None
    _multipole_pair_batch_compiled = None


def _odd_double_factorial(value: int) -> int:
    if value <= 0:
        return 1
    result = 1
    for item in range(value, 0, -2):
        result *= item
    return result


@lru_cache(maxsize=None)
def _primitive_normalization(exponent: float, angular: tuple[int, int, int]) -> float:
    lx, ly, lz = angular
    angular_factor = (4.0 * exponent) ** ((lx + ly + lz) / 2.0)
    denominator = sqrt(
        _odd_double_factorial(2 * lx - 1)
        * _odd_double_factorial(2 * ly - 1)
        * _odd_double_factorial(2 * lz - 1)
    )
    return (2.0 * exponent / pi) ** 0.75 * angular_factor / denominator


@dataclass(frozen=True)
class GaussianPrimitive:
    exponent: float
    coefficient: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.exponent) or self.exponent <= 0.0:
            raise ValueError("Gaussian exponent must be finite and positive")
        if not np.isfinite(self.coefficient):
            raise ValueError("Gaussian coefficient must be finite")


@dataclass(frozen=True)
class ContractedGaussian:
    center_bohr: tuple[float, float, float]
    angular: tuple[int, int, int]
    primitives: tuple[GaussianPrimitive, ...]
    atom_index: int
    shell: str
    slater_exponent: float

    def __post_init__(self) -> None:
        if len(self.center_bohr) != 3 or not np.all(np.isfinite(self.center_bohr)):
            raise ValueError("Gaussian center must contain three finite coordinates")
        if len(self.angular) != 3 or any(value < 0 for value in self.angular):
            raise ValueError("Gaussian angular powers must be three nonnegative integers")
        if not self.primitives:
            raise ValueError("contracted Gaussian requires at least one primitive")
        if self.atom_index < 0:
            raise ValueError("Gaussian atom_index must be nonnegative")
        if not np.isfinite(self.slater_exponent) or self.slater_exponent <= 0.0:
            raise ValueError("Slater exponent must be finite and positive")


@dataclass(frozen=True)
class NativeIntegralContext:
    """Opaque persistent native basis and AO-pair context."""

    capsule: object
    size: int
    pair_left: np.ndarray
    pair_right: np.ndarray
    _spherical_configuration_identity: tuple[int, ...] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def update_dense_model(
        self,
        core: np.ndarray,
        overlap: np.ndarray,
        operators,
        orbital_shell: Sequence[int],
        reference_population: Sequence[float],
        valence_nuclear_charges: Sequence[float],
    ) -> None:
        _gfn2_native.context_update_dense_model(
            self.capsule,
            core,
            overlap,
            operators.pair_overlap,
            operators.pair_dipole,
            operators.pair_second,
            operators.centers,
            operators.atom_index,
            np.asarray(orbital_shell, dtype=np.int64),
            np.asarray(reference_population, dtype=float),
            np.asarray(valence_nuclear_charges, dtype=float),
        )

    def dense_observables(self, density: np.ndarray):
        return _gfn2_native.context_dense_observables(self.capsule, density)

    def configure_core(
        self,
        orbitals,
        atomic_numbers: Sequence[int],
        atom_parameters,
        orbital_parameters,
    ) -> None:
        from .hamiltonian import _orbital_value

        atom = np.asarray([item.atom_index for item in orbitals], dtype=np.int64)
        level = np.empty(len(orbitals), dtype=float)
        shift = np.empty(len(orbitals), dtype=float)
        polynomial = np.empty(len(orbitals), dtype=float)
        slater = np.asarray([item.slater_exponent for item in orbitals], dtype=float)
        angular_lookup = {"s": 0, "p": 1, "d": 2}
        angular = np.asarray(
            [angular_lookup.get(str(item.shell)[-1], 3) for item in orbitals],
            dtype=np.int64,
        )
        for index, item in enumerate(orbitals):
            parameters = _orbital_value(
                atomic_numbers, item.atom_index, item.shell, orbital_parameters,
            )
            level[index] = parameters.level_hartree
            shift[index] = parameters.coordination_shift_hartree
            polynomial[index] = parameters.polynomial
        radii = np.asarray(
            [atom_parameters[number].d4_covalent_radius_bohr for number in atomic_numbers],
            dtype=float,
        )
        electronegativity = np.asarray(
            [atom_parameters[number].electronegativity for number in atomic_numbers],
            dtype=float,
        )
        _gfn2_native.context_configure_core(
            self.capsule, atom, level, shift, polynomial, slater, angular,
            radii, electronegativity,
        )

    def configure_spherical(
        self,
        offsets: np.ndarray,
        cartesian_indices: np.ndarray,
        coefficients: np.ndarray,
        pair_left: np.ndarray,
        pair_right: np.ndarray,
    ) -> None:
        identity = tuple(
            id(item)
            for item in (
                offsets,
                cartesian_indices,
                coefficients,
                pair_left,
                pair_right,
            )
        )
        if identity == self._spherical_configuration_identity:
            return
        _gfn2_native.context_configure_spherical(
            self.capsule,
            offsets,
            cartesian_indices,
            coefficients,
            pair_left,
            pair_right,
        )
        object.__setattr__(self, "_spherical_configuration_identity", identity)

    def evaluate_spherical(
        self,
        centers: np.ndarray,
        origin: np.ndarray,
        native_threads: int,
    ):
        return _gfn2_native.context_evaluate_spherical(
            self.capsule, centers, origin, int(native_threads)
        )

    def spherical_directional_seconds(
        self,
        centers: np.ndarray,
        origin: np.ndarray,
        center_direction: np.ndarray,
        native_threads: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return directional derivatives of pair endpoint AO gradients.

        ``center_direction`` may contain one direction or a leading batch.
        The native kernel evaluates primitive second derivatives once for the
        whole batch.
        """

        left, right = _gfn2_native.context_spherical_directional_seconds(
            self.capsule,
            centers,
            origin,
            center_direction,
            int(native_threads),
        )
        left = np.asarray(left)
        right = np.asarray(right)
        if np.asarray(center_direction).ndim == 2:
            return left[0], right[0]
        return left, right

    def build_core(
        self,
        overlap: np.ndarray,
        coordinates: np.ndarray,
        coordination: np.ndarray,
    ) -> np.ndarray:
        return np.asarray(
            _gfn2_native.context_build_core(
                self.capsule, overlap, coordinates, coordination
            )
        )

    def prepare_dense_geometry(
        self,
        cartesian_centers: np.ndarray,
        coordinates: np.ndarray,
        coordination_radii: np.ndarray,
        atom: np.ndarray,
        shell: np.ndarray,
        reference: np.ndarray,
        valence: np.ndarray,
        native_threads: int,
    ):
        """Build and install the complete small dense model in one native call."""

        return tuple(
            np.asarray(item)
            for item in _gfn2_native.context_prepare_dense_geometry(
                self.capsule,
                cartesian_centers,
                np.zeros(3, dtype=float),
                coordinates,
                coordination_radii,
                atom,
                shell,
                reference,
                valence,
                int(native_threads),
            )
        )

    def h0_derivatives(
        self,
        rows: np.ndarray,
        columns: np.ndarray,
        overlap: np.ndarray,
        left_overlap_derivatives: np.ndarray,
        right_overlap_derivatives: np.ndarray,
        coordinates: np.ndarray,
        coordination: np.ndarray,
    ):
        return tuple(
            np.asarray(item)
            for item in _gfn2_native.context_h0_derivatives(
                self.capsule,
                rows,
                columns,
                overlap,
                left_overlap_derivatives,
                right_overlap_derivatives,
                coordinates,
                coordination,
            )
        )

    def repulsion_energy_gradient(
        self,
        coordinates: np.ndarray,
        atomic_numbers: np.ndarray,
        alpha: np.ndarray,
        effective_charge: np.ndarray,
        gradient_cutoff_bohr: float,
    ) -> tuple[float, np.ndarray]:
        energy, gradient = _gfn2_native.repulsion_energy_gradient(
            coordinates,
            atomic_numbers,
            alpha,
            effective_charge,
            float(gradient_cutoff_bohr),
        )
        return float(energy), np.asarray(gradient)

    def anisotropic_geometry(
        self,
        coordinates: np.ndarray,
        radii: np.ndarray,
        a3: float,
        a5: float,
    ):
        return tuple(
            np.asarray(item)
            for item in _gfn2_native.anisotropic_geometry(
                coordinates, radii, float(a3), float(a5)
            )
        )

    def post_scf_explicit_gradient(
        self, coordinates, shell_charges, shell_atoms, shell_hardness,
        charges, dipoles, quadrupoles, anisotropic_radii,
        coordination_radii, radius_coordination_derivative,
        c6, c8, d4_coordination_derivative, d4_radii,
        electronegativities, *, anisotropic_a3, anisotropic_a5,
        s6, s8, d4_a1, d4_a2,
    ) -> np.ndarray:
        return np.asarray(_gfn2_native.post_scf_explicit_gradient(
            coordinates, shell_charges, shell_atoms, shell_hardness,
            charges, dipoles, quadrupoles, anisotropic_radii,
            coordination_radii, radius_coordination_derivative,
            c6, c8, d4_coordination_derivative, d4_radii,
            electronegativities, float(anisotropic_a3), float(anisotropic_a5),
            float(s6), float(s8), float(d4_a1), float(d4_a2),
        ))

    def atm_energy_gradient(
        self,
        coordinates: np.ndarray,
        c6: np.ndarray,
        c8: np.ndarray,
        c6_coordination_derivative: np.ndarray,
        coordination_radii: np.ndarray,
        electronegativities: np.ndarray,
        *,
        s9: float,
        a1: float,
        a2: float,
        alpha: float = 16.0,
    ) -> tuple[float, np.ndarray]:
        energy, gradient = _gfn2_native.atm_three_body_energy_gradient(
            coordinates,
            c6,
            c8,
            c6_coordination_derivative,
            coordination_radii,
            electronegativities,
            float(s9),
            float(a1),
            float(a2),
            float(alpha),
        )
        return float(energy), np.asarray(gradient)

    def dense_fock(
        self,
        shell_potential: np.ndarray,
        charge_potential: np.ndarray,
        dipole_potential: np.ndarray,
        quadrupole_potential: np.ndarray,
        d4_charge_derivative: np.ndarray,
    ) -> np.ndarray:
        return np.asarray(
            _gfn2_native.context_dense_fock(
                self.capsule,
                shell_potential,
                charge_potential,
                dipole_potential,
                quadrupole_potential,
                d4_charge_derivative,
            )
        )

    def generalized_eigh(self, fock: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values, vectors = _gfn2_native.context_generalized_eigh(
            self.capsule, np.asarray(fock, dtype=float)
        )
        return np.asarray(values), np.asarray(vectors)

    def configure_scf(self, configuration: dict[str, np.ndarray]) -> None:
        _gfn2_native.context_configure_scf(self.capsule, configuration)

    def prepare_small_scf(
        self,
        coordinates: np.ndarray,
        coordination: np.ndarray,
        configuration: dict[str, object],
    ):
        return _gfn2_native.context_prepare_small_scf(
            self.capsule, coordinates, coordination, configuration
        )

    def geometry_within_displacement(
        self,
        coordinates: np.ndarray,
        previous: np.ndarray,
        threshold: float,
    ) -> bool:
        return bool(
            _gfn2_native.geometry_within_displacement(
                coordinates, previous, float(threshold)
            )
        )

    def scf_state_fock(self, density: np.ndarray) -> tuple[np.ndarray, float]:
        fock, energy = _gfn2_native.context_scf_state_fock(
            self.capsule, np.asarray(density, dtype=float)
        )
        return np.asarray(fock), float(energy)

    def resident_iscf(
        self,
        density: np.ndarray,
        *,
        electrons: int,
        multiplicity: int,
        temperature_kelvin: float,
        wavefunction_tolerance: float,
        energy_tolerance: float,
    ):
        return _gfn2_native.context_resident_iscf(
            self.capsule,
            np.asarray(density, dtype=float),
            int(electrons),
            int(multiplicity),
            float(temperature_kelvin),
            float(wavefunction_tolerance),
            float(energy_tolerance),
        )

    def resident_diis(
        self,
        density: np.ndarray,
        *,
        electrons: int,
        multiplicity: int,
        temperature_kelvin: float,
        wavefunction_tolerance: float,
        energy_tolerance: float,
        maximum_cycles: int,
        mixer_tolerance: float = -1.0,
    ):
        return _gfn2_native.context_resident_diis(
            self.capsule,
            np.asarray(density, dtype=float),
            int(electrons),
            int(multiplicity),
            float(temperature_kelvin),
            float(wavefunction_tolerance),
            float(energy_tolerance),
            int(maximum_cycles),
            float(mixer_tolerance),
        )

    def last_scf_state(self):
        return _gfn2_native.context_last_scf_state(self.capsule)

    def extreme_singlepoint(
        self,
        cartesian_centers: np.ndarray,
        coordinates: np.ndarray,
        configuration: dict[str, object],
        density: np.ndarray,
        *,
        electrons: int,
        multiplicity: int,
        temperature_kelvin: float,
        wavefunction_tolerance: float,
        energy_tolerance: float,
        maximum_cycles: int,
        mixer_tolerance: float = -1.0,
    ):
        """Run the complete certified tiny-system energy/gradient kernel."""

        return _gfn2_native.context_extreme_singlepoint(
            self.capsule,
            cartesian_centers,
            coordinates,
            configuration,
            density,
            int(electrons),
            int(multiplicity),
            float(temperature_kelvin),
            float(wavefunction_tolerance),
            float(energy_tolerance),
            int(maximum_cycles),
            float(mixer_tolerance),
        )

    def variational_gradient_sparse(
        self,
        density,
        weighted_density,
        rows,
        columns,
        left_atoms,
        right_atoms,
        h0_left,
        h0_right,
        camm_left,
        camm_right,
        pair_scalar,
        overlap_left,
        overlap_right,
        coordination_left,
        coordination_right,
        natoms: int,
        native_threads: int,
    ):
        return _gfn2_native.context_variational_gradient_sparse(
            self.capsule,
            int(density.size),
            density.rows,
            density.columns,
            density.values,
            weighted_density.rows,
            weighted_density.columns,
            weighted_density.values,
            rows,
            columns,
            np.asarray(left_atoms, dtype=np.int32),
            np.asarray(right_atoms, dtype=np.int32),
            h0_left,
            h0_right,
            camm_left,
            camm_right,
            pair_scalar,
            overlap_left,
            overlap_right,
            coordination_left,
            coordination_right,
            int(natoms),
            int(native_threads),
        )

    def variational_gradient_dense(
        self,
        density,
        weighted_density,
        rows,
        columns,
        left_atoms,
        right_atoms,
        h0_left,
        h0_right,
        camm_left,
        camm_right,
        pair_scalar,
        overlap_left,
        overlap_right,
        coordination_left,
        coordination_right,
        natoms: int,
        native_threads: int,
    ):
        return _gfn2_native.context_variational_gradient_dense(
            self.capsule,
            density,
            weighted_density,
            rows,
            columns,
            np.asarray(left_atoms, dtype=np.int32),
            np.asarray(right_atoms, dtype=np.int32),
            h0_left,
            h0_right,
            camm_left,
            camm_right,
            pair_scalar,
            overlap_left,
            overlap_right,
            coordination_left,
            coordination_right,
            int(natoms),
            int(native_threads),
        )


@lru_cache(maxsize=262_144)
def _overlap_1d(
    left_power: int,
    right_power: int,
    left_center: float,
    right_center: float,
    left_exponent: float,
    right_exponent: float,
) -> float:
    total = left_exponent + right_exponent
    reduced = left_exponent * right_exponent / total
    product_center = (
        left_exponent * left_center + right_exponent * right_center
    ) / total
    table = np.zeros((left_power + 1, right_power + 1), dtype=float)
    table[0, 0] = sqrt(pi / total) * exp(
        -reduced * (left_center - right_center) ** 2
    )
    for left in range(1, left_power + 1):
        table[left, 0] = (product_center - left_center) * table[left - 1, 0]
        if left > 1:
            table[left, 0] += (left - 1) * table[left - 2, 0] / (2.0 * total)
    for right in range(1, right_power + 1):
        table[0, right] = (product_center - right_center) * table[0, right - 1]
        if right > 1:
            table[0, right] += (right - 1) * table[0, right - 2] / (2.0 * total)
    for left in range(1, left_power + 1):
        for right in range(1, right_power + 1):
            table[left, right] = (
                (product_center - right_center) * table[left, right - 1]
                + left * table[left - 1, right - 1] / (2.0 * total)
            )
            if right > 1:
                table[left, right] += (
                    (right - 1) * table[left, right - 2] / (2.0 * total)
                )
    return float(table[left_power, right_power])


def _primitive_overlap(
    left: ContractedGaussian,
    right: ContractedGaussian,
    left_primitive: GaussianPrimitive,
    right_primitive: GaussianPrimitive,
    *,
    left_shift: tuple[int, int, int] = (0, 0, 0),
    right_shift: tuple[int, int, int] = (0, 0, 0),
) -> float:
    value = 1.0
    for axis in range(3):
        value *= _overlap_1d(
            left.angular[axis] + left_shift[axis],
            right.angular[axis] + right_shift[axis],
            left.center_bohr[axis],
            right.center_bohr[axis],
            left_primitive.exponent,
            right_primitive.exponent,
        )
    return value * _primitive_normalization(
        left_primitive.exponent, left.angular
    ) * _primitive_normalization(right_primitive.exponent, right.angular)


@lru_cache(maxsize=65_536)
def _raw_contracted_overlap(
    left: ContractedGaussian,
    right: ContractedGaussian,
    *,
    left_shift: tuple[int, int, int] = (0, 0, 0),
    right_shift: tuple[int, int, int] = (0, 0, 0),
) -> float:
    return sum(
        left_primitive.coefficient
        * right_primitive.coefficient
        * _primitive_overlap(
            left,
            right,
            left_primitive,
            right_primitive,
            left_shift=left_shift,
            right_shift=right_shift,
        )
        for left_primitive in left.primitives
        for right_primitive in right.primitives
    )


def contracted_overlap(left: ContractedGaussian, right: ContractedGaussian) -> float:
    numerator = _raw_contracted_overlap(left, right)
    left_norm = _raw_contracted_overlap(left, left)
    right_norm = _raw_contracted_overlap(right, right)
    return numerator / sqrt(left_norm * right_norm)


def contracted_overlap_gradient(
    left: ContractedGaussian, right: ContractedGaussian
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact derivatives of an overlap with respect to both centres."""

    left_norm = _raw_contracted_overlap(left, left)
    right_norm = _raw_contracted_overlap(right, right)
    denominator = sqrt(left_norm * right_norm)
    derivative = np.zeros(3, dtype=float)
    for left_primitive in left.primitives:
        for right_primitive in right.primitives:
            weight = left_primitive.coefficient * right_primitive.coefficient
            for axis in range(3):
                raised = tuple(int(index == axis) for index in range(3))
                value = 2.0 * left_primitive.exponent * _primitive_overlap(
                    left,
                    right,
                    left_primitive,
                    right_primitive,
                    left_shift=raised,
                )
                if left.angular[axis] > 0:
                    lowered = tuple(-int(index == axis) for index in range(3))
                    value -= left.angular[axis] * _primitive_overlap(
                        left,
                        right,
                        left_primitive,
                        right_primitive,
                        left_shift=lowered,
                    )
                derivative[axis] += weight * value
    derivative /= denominator
    return derivative, -derivative


def contracted_orbital_angular_momentum(
    left: ContractedGaussian,
    right: ContractedGaussian,
    origin_bohr: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Return the real generator ``<left|(r-O) x grad|right>``.

    The physical orbital-angular-momentum matrix is ``-1j`` times this real
    antisymmetric generator.  Keeping the real generator avoids complex AO
    storage in the magnetic response used by VCD.
    """

    origin = np.asarray(origin_bohr, dtype=float)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("angular-momentum origin must contain three finite values")
    left_norm = _raw_contracted_overlap(left, left)
    right_norm = _raw_contracted_overlap(right, right)
    denominator = sqrt(left_norm * right_norm)
    position_gradient = np.zeros((3, 3), dtype=float)
    for left_primitive in left.primitives:
        for right_primitive in right.primitives:
            weight = left_primitive.coefficient * right_primitive.coefficient
            for position_axis in range(3):
                raised_left = tuple(
                    int(axis == position_axis) for axis in range(3)
                )
                shift = left.center_bohr[position_axis] - origin[position_axis]
                for gradient_axis in range(3):
                    raised_right = tuple(
                        int(axis == gradient_axis) for axis in range(3)
                    )
                    base = -2.0 * right_primitive.exponent * _primitive_overlap(
                        left,
                        right,
                        left_primitive,
                        right_primitive,
                        right_shift=raised_right,
                    )
                    raised = -2.0 * right_primitive.exponent * _primitive_overlap(
                        left,
                        right,
                        left_primitive,
                        right_primitive,
                        left_shift=raised_left,
                        right_shift=raised_right,
                    )
                    if right.angular[gradient_axis] > 0:
                        lowered_right = tuple(
                            -int(axis == gradient_axis) for axis in range(3)
                        )
                        base += right.angular[gradient_axis] * _primitive_overlap(
                            left,
                            right,
                            left_primitive,
                            right_primitive,
                            right_shift=lowered_right,
                        )
                        raised += right.angular[gradient_axis] * _primitive_overlap(
                            left,
                            right,
                            left_primitive,
                            right_primitive,
                            left_shift=raised_left,
                            right_shift=lowered_right,
                        )
                    position_gradient[position_axis, gradient_axis] += weight * (
                        raised + shift * base
                    )
    position_gradient /= denominator
    return np.asarray(
        (
            position_gradient[1, 2] - position_gradient[2, 1],
            position_gradient[2, 0] - position_gradient[0, 2],
            position_gradient[0, 1] - position_gradient[1, 0],
        )
    )


def orbital_angular_momentum_matrices(
    orbitals: Sequence[ContractedGaussian],
    transformation: np.ndarray | None = None,
    *,
    origin_bohr: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Build real AO generators for the three angular-momentum components."""

    size = len(orbitals)
    cartesian = np.zeros((3, size, size), dtype=float)
    for left in range(size):
        for right in range(left):
            values = contracted_orbital_angular_momentum(
                orbitals[left], orbitals[right], origin_bohr
            )
            cartesian[:, left, right] = values
            cartesian[:, right, left] = -values
    if transformation is None:
        return cartesian
    transform = np.asarray(transformation, dtype=float)
    if transform.ndim != 2 or transform.shape[0] != size:
        raise ValueError("angular-momentum AO transformation is incompatible")
    return np.einsum(
        "pi,apq,qj->aij", transform, cartesian, transform, optimize=True
    )


def overlap_matrix(orbitals: Sequence[ContractedGaussian]) -> np.ndarray:
    size = len(orbitals)
    result = np.eye(size, dtype=float)
    for left in range(size):
        for right in range(left):
            value = contracted_overlap(orbitals[left], orbitals[right])
            result[left, right] = result[right, left] = value
    return result


def contracted_gaussian_values(
    orbitals: Sequence[ContractedGaussian], points_bohr: np.ndarray
) -> np.ndarray:
    """Evaluate normalized contracted Cartesian Gaussians on a point batch."""

    points = np.asarray(points_bohr, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Gaussian evaluation points must have shape npoints x 3")
    values = np.empty((len(points), len(orbitals)))
    for index, orbital in enumerate(orbitals):
        relative = points - np.asarray(orbital.center_bohr)
        radius_squared = np.sum(relative**2, axis=1)
        polynomial = np.prod(
            relative ** np.asarray(orbital.angular)[None, :], axis=1
        )
        radial = np.zeros(len(points))
        for primitive in orbital.primitives:
            radial += (
                primitive.coefficient
                * _primitive_normalization(primitive.exponent, orbital.angular)
                * np.exp(-primitive.exponent * radius_squared)
            )
        values[:, index] = (
            polynomial * radial / sqrt(_raw_contracted_overlap(orbital, orbital))
        )
    return values


@lru_cache(maxsize=65_536)
def _shifted_overlap(
    left: ContractedGaussian,
    right: ContractedGaussian,
    shift: tuple[int, int, int],
) -> float:
    numerator = _raw_contracted_overlap(left, right, left_shift=shift)
    left_norm = _raw_contracted_overlap(left, left)
    right_norm = _raw_contracted_overlap(right, right)
    return numerator / sqrt(left_norm * right_norm)


@lru_cache(maxsize=2)
def _compiled_multipole_pairs_cached(
    orbitals: tuple[ContractedGaussian, ...],
    origin_tuple: tuple[float, float, float],
    norms: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if _gfn2_native is None and _multipole_pair_batch_compiled is None:
        return None
    origin = np.asarray(origin_tuple, dtype=float)
    signature = tuple((item.angular, item.primitives) for item in orbitals)
    pair_left, pair_right, angular, exponents, coefficients, primitive_counts = (
        _compiled_multipole_static(signature)
    )
    centers = np.asarray([item.center_bohr for item in orbitals], dtype=float)
    displacement = centers[pair_left] - centers[pair_right]
    keep = np.einsum("pi,pi->p", displacement, displacement) <= (
        _AO_PAIR_CUTOFF_BOHR**2
    )
    pair_left = pair_left[keep]
    pair_right = pair_right[keep]
    kernel = (
        _gfn2_native.multipole_pair_batch
        if _gfn2_native is not None
        else _multipole_pair_batch_compiled
    )
    values, left_derivatives, right_derivatives = kernel(
        centers, angular, exponents, coefficients, primitive_counts,
        pair_left, pair_right, np.asarray(norms, dtype=float), origin,
    )
    return pair_left, pair_right, values, left_derivatives, right_derivatives


@lru_cache(maxsize=64)
def _compiled_multipole_static(signature):
    """Pack topology/basis data once for a resident calculator."""

    size = len(signature)
    pair_left, pair_right = triangular_indices(size)
    pair_left = np.asarray(pair_left, dtype=np.int32)
    pair_right = np.asarray(pair_right, dtype=np.int32)
    maximum_primitives = max((len(primitives) for _angular, primitives in signature), default=0)
    angular = np.asarray([item[0] for item in signature], dtype=np.int64)
    exponents = np.zeros((size, maximum_primitives), dtype=float)
    coefficients = np.zeros_like(exponents)
    primitive_counts = np.empty(size, dtype=np.int64)
    for orbital_index, (_angular, primitives) in enumerate(signature):
        primitive_counts[orbital_index] = len(primitives)
        exponents[orbital_index, : primitive_counts[orbital_index]] = [
            item.exponent for item in primitives
        ]
        coefficients[orbital_index, : primitive_counts[orbital_index]] = [
            item.coefficient for item in primitives
        ]
    return pair_left, pair_right, angular, exponents, coefficients, primitive_counts


def _compiled_multipole_pairs(
    orbitals: Sequence[ContractedGaussian],
    origin: np.ndarray,
    norms: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Reuse the fused native values/derivatives within one geometry."""

    return _compiled_multipole_pairs_cached(
        tuple(orbitals),
        tuple(float(value) for value in np.asarray(origin, dtype=float)),
        tuple(float(value) for value in norms),
    )


@lru_cache(maxsize=64)
def _compiled_multipole_basis_static(signature):
    size = len(signature)
    maximum_primitives = max(
        (len(primitives) for _angular, primitives in signature), default=0
    )
    angular = np.asarray([item[0] for item in signature], dtype=np.int64)
    exponents = np.zeros((size, maximum_primitives), dtype=float)
    coefficients = np.zeros_like(exponents)
    primitive_counts = np.empty(size, dtype=np.int64)
    for orbital_index, (_angular, primitives) in enumerate(signature):
        primitive_counts[orbital_index] = len(primitives)
        exponents[orbital_index, : primitive_counts[orbital_index]] = [
            item.exponent for item in primitives
        ]
        coefficients[orbital_index, : primitive_counts[orbital_index]] = [
            item.coefficient for item in primitives
        ]
    return angular, exponents, coefficients, primitive_counts


@lru_cache(maxsize=64)
def _compiled_multipole_basis_native_static(signature):
    """Pack primitive coefficients with normalization folded in once."""

    angular, exponents, coefficients, primitive_counts = (
        _compiled_multipole_basis_static(signature)
    )
    normalized = coefficients.copy()
    for orbital_index, (powers, primitives) in enumerate(signature):
        count = primitive_counts[orbital_index]
        normalized[orbital_index, :count] *= [
            _primitive_normalization(item.exponent, powers) for item in primitives
        ]
    normalized.setflags(write=False)
    return angular, exponents, normalized, primitive_counts


def prepare_native_integral_context(
    orbitals: Sequence[ContractedGaussian],
    pair_left: Sequence[int],
    pair_right: Sequence[int],
) -> NativeIntegralContext:
    """Move geometry-invariant AO data into one persistent C++ context."""

    if _gfn2_native is None or not hasattr(_gfn2_native, "create_integral_context"):
        raise RuntimeError("persistent AO integral context requires native kernels")
    left = np.asarray(pair_left, dtype=np.int32)
    right = np.asarray(pair_right, dtype=np.int32)
    signature = tuple((item.angular, item.primitives) for item in orbitals)
    angular, exponents, coefficients, primitive_counts = (
        _compiled_multipole_basis_native_static(signature)
    )
    norms = _compiled_multipole_basis_norms(signature)
    capsule = _gfn2_native.create_integral_context(
        angular, exponents, coefficients, primitive_counts,
        left, right, norms,
    )
    return NativeIntegralContext(capsule, len(orbitals), left, right)


@lru_cache(maxsize=64)
def _compiled_multipole_basis_norms(signature) -> np.ndarray:
    """Return translation-invariant contraction norms for one AO signature."""

    norms = np.empty(len(signature), dtype=float)
    for index, (angular, primitives) in enumerate(signature):
        orbital = ContractedGaussian(
            center_bohr=(0.0, 0.0, 0.0),
            angular=angular,
            primitives=primitives,
            atom_index=0,
            shell="normalization",
            slater_exponent=1.0,
        )
        norms[index] = _raw_contracted_overlap(orbital, orbital)
    norms.setflags(write=False)
    return norms


def sparse_atomic_multipole_integrals(
    orbitals: Sequence[ContractedGaussian],
    pair_left: Sequence[int],
    pair_right: Sequence[int],
    *,
    origin_bohr: Sequence[float] = (0.0, 0.0, 0.0),
    threshold: float = 0.0,
    native_threads: int = 1,
    native_context: NativeIntegralContext | None = None,
) -> SparseAtomicMultipoleIntegrals:
    """Evaluate only cell-list-selected AO pairs in the fused native kernel."""

    if _gfn2_native is None:
        raise RuntimeError("sparse AO integral construction requires the native kernel")
    origin = np.asarray(origin_bohr, dtype=float)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("multipole origin must contain three finite coordinates")
    left = np.asarray(pair_left, dtype=np.int32)
    right = np.asarray(pair_right, dtype=np.int32)
    if left.ndim != 1 or right.shape != left.shape:
        raise ValueError("AO pair arrays must be one-dimensional and equally sized")
    size = len(orbitals)
    if np.any(left < 0) or np.any(right < 0) or np.any(left >= size) or np.any(right >= size):
        raise ValueError("AO pair index is outside the basis")
    if np.any(right > left):
        raise ValueError("AO pairs must use lower-triangular ordering")
    pair_keys = left * max(size, 1) + right
    if len(pair_keys) > 1 and np.any(pair_keys[1:] <= pair_keys[:-1]):
        raise ValueError("AO pairs must be unique and sorted by row and column")
    signature = tuple((item.angular, item.primitives) for item in orbitals)
    # Self-overlaps are invariant to a common translation of bra and ket.
    # Cache them by angular/contraction signature instead of hashing every
    # relocated AO centre and recomputing thousands of Python recurrences at
    # every trajectory step.
    norms = _compiled_multipole_basis_norms(signature)
    # The dense small/medium SCF model has already requested the complete
    # lower triangle at this geometry.  Reuse that fused values+derivatives
    # batch for the pair-packed gradient instead of executing the expensive
    # primitive contraction a second time.  This also handles a *subset* of
    # the triangle: the gradient's locality mask may remove distant pairs
    # even though the dense SCF build already evaluated them.  Large systems
    # retain the genuinely sparse batch below by avoiding this cache once the
    # requested set is a small fraction of the full triangle.
    dense_pair_count = size * (size + 1) // 2
    reuse_dense_cache = (
        dense_pair_count <= 20_000
        and len(left) * 4 >= dense_pair_count
    )
    compiled_candidate = (
        _compiled_multipole_pairs(orbitals, origin, norms)
        if reuse_dense_cache
        else None
    )
    compiled = None
    if compiled_candidate is not None:
        cached_left, cached_right = compiled_candidate[:2]
        cached_keys = cached_left * size + cached_right
        requested_keys = left * size + right
        positions = np.searchsorted(cached_keys, requested_keys)
        if (
            np.all(positions < len(cached_keys))
            and np.array_equal(cached_keys[positions], requested_keys)
        ):
            compiled = (
                left,
                right,
                compiled_candidate[2][positions],
                compiled_candidate[3][positions],
                compiled_candidate[4][positions],
            )
    if native_context is not None:
        if native_context.size != size:
            raise ValueError("persistent integral context has the wrong basis size")
        if not np.array_equal(left, native_context.pair_left) or not np.array_equal(
            right, native_context.pair_right
        ):
            raise ValueError("persistent integral context has the wrong AO pair topology")
        centers = np.asarray([item.center_bohr for item in orbitals], dtype=float)
        values, left_derivatives, right_derivatives = (
            _gfn2_native.integral_context_evaluate(
                native_context.capsule, centers, origin, int(native_threads)
            )
        )
    elif compiled is not None:
        _pair_left, _pair_right, values, left_derivatives, right_derivatives = compiled
        left, right = _pair_left, _pair_right
    else:
        angular, exponents, coefficients, primitive_counts = (
            _compiled_multipole_basis_native_static(signature)
        )
        centers = np.asarray([item.center_bohr for item in orbitals], dtype=float)
        values, left_derivatives, right_derivatives = _gfn2_native.multipole_pair_batch(
            centers, angular, exponents, coefficients, primitive_counts,
            left, right, norms, origin, int(native_threads), 1,
        )
    values = np.asarray(values, dtype=float)
    left_derivatives = np.asarray(left_derivatives, dtype=float)
    right_derivatives = np.asarray(right_derivatives, dtype=float)
    diagonal = left == right
    magnitude = np.max(np.abs(values), axis=1) if len(values) else np.empty(0)
    keep = diagonal | (magnitude >= float(threshold))
    omitted_overlap = values[~keep, 0]
    discarded = float(np.sqrt(2.0 * np.sum(omitted_overlap * omitted_overlap)))
    kept_values = values[keep]
    return SparseAtomicMultipoleIntegrals(
        size=size,
        rows=left[keep].astype(np.int32, copy=False),
        columns=right[keep].astype(np.int32, copy=False),
        overlap=kept_values[:, 0],
        dipole=kept_values[:, 1:4],
        second=kept_values[:, 4:].reshape(-1, 3, 3),
        left_derivatives=left_derivatives[keep],
        right_derivatives=right_derivatives[keep],
        threshold_discarded_frobenius_norm=discarded,
        domain_complete=(len(left) == size * (size + 1) // 2),
    )


def _contracted_overlap_moments(
    left: ContractedGaussian,
    right: ContractedGaussian,
    *,
    left_norm: float,
    right_norm: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Contract overlap, first shifts and second shifts in one pair pass.

    The former implementation requested every shifted overlap independently.
    That repeated the same primitive contraction thirteen times per AO pair
    and dominated even small MATRIX-GFN2 calculations.  All required moments
    are products of the same one-dimensional overlaps with left angular power
    raised by zero, one or two, so they can be accumulated together without
    changing the integral definition.
    """

    zeroth = 0.0
    first = np.zeros(3, dtype=float)
    second = np.zeros((3, 3), dtype=float)
    for left_primitive in left.primitives:
        left_normalization = _primitive_normalization(
            left_primitive.exponent, left.angular
        )
        for right_primitive in right.primitives:
            prefactor = (
                left_primitive.coefficient
                * right_primitive.coefficient
                * left_normalization
                * _primitive_normalization(right_primitive.exponent, right.angular)
            )
            axis_overlap = np.empty((3, 3), dtype=float)
            for axis in range(3):
                for raise_power in range(3):
                    axis_overlap[axis, raise_power] = _overlap_1d(
                        left.angular[axis] + raise_power,
                        right.angular[axis],
                        left.center_bohr[axis],
                        right.center_bohr[axis],
                        left_primitive.exponent,
                        right_primitive.exponent,
                    )
            base = float(np.prod(axis_overlap[:, 0]))
            zeroth += prefactor * base
            for axis in range(3):
                other_axes = [item for item in range(3) if item != axis]
                first[axis] += (
                    prefactor
                    * axis_overlap[axis, 1]
                    * axis_overlap[other_axes[0], 0]
                    * axis_overlap[other_axes[1], 0]
                )
                second[axis, axis] += (
                    prefactor
                    * axis_overlap[axis, 2]
                    * axis_overlap[other_axes[0], 0]
                    * axis_overlap[other_axes[1], 0]
                )
                for other in range(axis):
                    remaining = 3 - axis - other
                    value = (
                        prefactor
                        * axis_overlap[axis, 1]
                        * axis_overlap[other, 1]
                        * axis_overlap[remaining, 0]
                    )
                    second[axis, other] += value
                    second[other, axis] += value
    inverse_norm = 1.0 / sqrt(left_norm * right_norm)
    return zeroth * inverse_norm, first * inverse_norm, second * inverse_norm


def atomic_multipole_integrals(
    orbitals: Sequence[ContractedGaussian],
    *,
    origin_bohr: Sequence[float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return overlap, dipole, and Cartesian second-moment AO integrals."""

    origin = np.asarray(origin_bohr, dtype=float)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("multipole origin must contain three finite coordinates")
    size = len(orbitals)
    overlap = np.zeros((size, size), dtype=float)
    dipole = np.zeros((3, size, size), dtype=float)
    second = np.zeros((3, 3, size, size), dtype=float)
    norms = tuple(_raw_contracted_overlap(orbital, orbital) for orbital in orbitals)
    compiled_pairs = _compiled_multipole_pairs(orbitals, origin, norms)
    if compiled_pairs is not None:
        pair_left, pair_right, pair_values, _, _ = compiled_pairs
        overlap[pair_left, pair_right] = pair_values[:, 0]
        dipole[:, pair_left, pair_right] = pair_values[:, 1:4].T
        second[:, :, pair_left, pair_right] = pair_values[:, 4:].reshape(-1, 3, 3).transpose(1, 2, 0)
        off_diagonal = pair_left != pair_right
        upper = pair_left[off_diagonal]
        lower = pair_right[off_diagonal]
        overlap[lower, upper] = overlap[upper, lower]
        dipole[:, lower, upper] = dipole[:, upper, lower]
        second[:, :, lower, upper] = second[:, :, upper, lower]
        return overlap, dipole, second
    for left_index, left in enumerate(orbitals):
        relative_center = np.asarray(left.center_bohr) - origin
        for right_index in range(left_index + 1):
            right = orbitals[right_index]
            if _contracted_multipole_derivatives_compiled is not None:
                values, _, _ = _contracted_multipole_derivatives_compiled(
                    np.asarray(left.center_bohr, dtype=float),
                    np.asarray(right.center_bohr, dtype=float),
                    np.asarray(left.angular, dtype=np.int64),
                    np.asarray(right.angular, dtype=np.int64),
                    np.asarray([item.exponent for item in left.primitives], dtype=float),
                    np.asarray([item.coefficient for item in left.primitives], dtype=float),
                    np.asarray([item.exponent for item in right.primitives], dtype=float),
                    np.asarray([item.coefficient for item in right.primitives], dtype=float),
                    origin,
                    sqrt(norms[left_index] * norms[right_index]),
                )
                s0 = values[0]
                first = values[1:4]
                shifted_second = values[4:].reshape(3, 3)
            else:
                s0, shifted_first, raw_second = _contracted_overlap_moments(
                    left,
                    right,
                    left_norm=norms[left_index],
                    right_norm=norms[right_index],
                )
                first = shifted_first + relative_center * s0
                shifted_second = np.empty((3, 3), dtype=float)
                for axis in range(3):
                    for other in range(3):
                        shifted_second[axis, other] = (
                            raw_second[axis, other]
                            + relative_center[axis] * shifted_first[other]
                            + relative_center[other] * shifted_first[axis]
                            + relative_center[axis] * relative_center[other] * s0
                        )
            overlap[left_index, right_index] = overlap[right_index, left_index] = s0
            dipole[:, left_index, right_index] = first
            second[:, :, left_index, right_index] = shifted_second
            if right_index != left_index:
                dipole[:, right_index, left_index] = dipole[:, left_index, right_index]
                second[:, :, right_index, left_index] = second[
                    :, :, left_index, right_index
                ]
    return overlap, dipole, second


def _primitive_moment(
    left: ContractedGaussian,
    right: ContractedGaussian,
    left_primitive: GaussianPrimitive,
    right_primitive: GaussianPrimitive,
    moment: tuple[int, int, int],
    origin: np.ndarray,
    *,
    left_shift: tuple[int, int, int] = (0, 0, 0),
    right_shift: tuple[int, int, int] = (0, 0, 0),
) -> float:
    value = 1.0
    for axis in range(3):
        axis_value = 0.0
        relative = left.center_bohr[axis] - origin[axis]
        for power in range(moment[axis] + 1):
            axis_value += (
                comb(moment[axis], power)
                * relative ** (moment[axis] - power)
                * _overlap_1d(
                    left.angular[axis] + left_shift[axis] + power,
                    right.angular[axis] + right_shift[axis],
                    left.center_bohr[axis],
                    right.center_bohr[axis],
                    left_primitive.exponent,
                    right_primitive.exponent,
                )
            )
        value *= axis_value
    return value * _primitive_normalization(
        left_primitive.exponent, left.angular
    ) * _primitive_normalization(right_primitive.exponent, right.angular)


def contracted_multipole_exponent_derivatives(
    left: ContractedGaussian,
    right: ContractedGaussian,
    *,
    origin_bohr: Sequence[float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the 13 AO moments and exact bra/ket Slater-exponent derivatives.

    STO-nG primitive exponents scale as zeta squared while contraction
    coefficients remain fixed.  Differentiating each normalized primitive
    gives (3/2+l) phi/zeta - 2 alpha r² phi/zeta.  The contracted-orbital
    normalization is differentiated separately; omitting it makes even the
    derivative of a diagonal overlap nonzero.
    """
    origin = np.asarray(origin_bohr, dtype=float)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("multipole origin must be a finite three-vector")
    moments = ((0, 0, 0),) + tuple(
        tuple(int(axis == component) for axis in range(3))
        for component in range(3)
    ) + tuple(
        tuple(int(axis == first) + int(axis == second) for axis in range(3))
        for first in range(3) for second in range(3)
    )

    def raw_pair(bra, ket, moment):
        value = bra_derivative = ket_derivative = 0.0
        for primitive_bra in bra.primitives:
            for primitive_ket in ket.primitives:
                weight = primitive_bra.coefficient * primitive_ket.coefficient
                base = _primitive_moment(
                    bra, ket, primitive_bra, primitive_ket, moment, origin,
                )
                bra_radius2 = ket_radius2 = 0.0
                for axis in range(3):
                    raised = tuple(2 * int(component == axis) for component in range(3))
                    bra_radius2 += _primitive_moment(
                        bra, ket, primitive_bra, primitive_ket, moment, origin,
                        left_shift=raised,
                    )
                    ket_radius2 += _primitive_moment(
                        bra, ket, primitive_bra, primitive_ket, moment, origin,
                        right_shift=raised,
                    )
                value += weight * base
                bra_derivative += weight * (
                    (1.5 + sum(bra.angular)) * base
                    - 2.0 * primitive_bra.exponent * bra_radius2
                ) / bra.slater_exponent
                ket_derivative += weight * (
                    (1.5 + sum(ket.angular)) * base
                    - 2.0 * primitive_ket.exponent * ket_radius2
                ) / ket.slater_exponent
        return value, bra_derivative, ket_derivative

    left_norm, left_bra, left_ket = raw_pair(left, left, moments[0])
    right_norm, right_bra, right_ket = raw_pair(right, right, moments[0])
    inverse_norm = 1.0 / sqrt(left_norm * right_norm)
    left_log_norm = 0.5 * (left_bra + left_ket) / left_norm
    right_log_norm = 0.5 * (right_bra + right_ket) / right_norm
    values = np.empty(13)
    left_derivatives = np.empty(13)
    right_derivatives = np.empty(13)
    for index, moment in enumerate(moments):
        value, bra, ket = raw_pair(left, right, moment)
        values[index] = value * inverse_norm
        left_derivatives[index] = (bra - value * left_log_norm) * inverse_norm
        right_derivatives[index] = (ket - value * right_log_norm) * inverse_norm
    return values, left_derivatives, right_derivatives


def contracted_multipole_exponent_center_derivatives(
    left: ContractedGaussian,
    right: ContractedGaussian,
    *, origin_bohr: Sequence[float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact normalized AO-moment mixed Slater-exponent/centre derivatives.

    Return ``(zeta_left, centre_left)``, ``(zeta_left, centre_right)``,
    ``(zeta_right, centre_left)``, ``(zeta_right, centre_right)``; each
    array has shape ``(13, 3)``. The same primitive-moment recurrence and
    contracted-orbital normalization as the existing exponent kernel are
    used. No coordinate or exponent probes are involved.
    """
    origin = np.asarray(origin_bohr, dtype=float)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("multipole origin must be a finite three-vector")
    moments = ((0, 0, 0),) + tuple(
        tuple(int(axis == component) for axis in range(3))
        for component in range(3)
    ) + tuple(
        tuple(int(axis == first) + int(axis == second) for axis in range(3))
        for first in range(3) for second in range(3)
    )
    radial = tuple(
        tuple(2 * int(component == axis) for component in range(3))
        for axis in range(3)
    )

    def raw_pair(bra, ket, moment):
        # Components: value, bra-zeta, ket-zeta, then centre derivatives
        # and their two exponent derivatives for each centre.
        result = np.zeros((3, 3, 3))
        zb, zk = bra.slater_exponent, ket.slater_exponent
        cb = 1.5 + sum(bra.angular)
        ck = 1.5 + sum(ket.angular)
        for primitive_bra in bra.primitives:
            for primitive_ket in ket.primitives:
                alpha, beta = primitive_bra.exponent, primitive_ket.exponent
                weight = primitive_bra.coefficient * primitive_ket.coefficient

                def value(left_shift=(0, 0, 0), right_shift=(0, 0, 0)):
                    return _primitive_moment(
                        bra, ket, primitive_bra, primitive_ket, moment, origin,
                        left_shift=left_shift, right_shift=right_shift,
                    )

                def exponent_terms(left_shift=(0, 0, 0), right_shift=(0, 0, 0)):
                    base = value(left_shift, right_shift)
                    bra_radius = sum(value(
                        tuple(left_shift[a] + shift[a] for a in range(3)),
                        right_shift,
                    ) for shift in radial)
                    ket_radius = sum(value(
                        left_shift,
                        tuple(right_shift[a] + shift[a] for a in range(3)),
                    ) for shift in radial)
                    return np.array((
                        base,
                        (cb * base - 2.0 * alpha * bra_radius) / zb,
                        (ck * base - 2.0 * beta * ket_radius) / zk,
                    ))

                base = exponent_terms()
                for exponent in range(3):
                    result[exponent, 0, 0] += weight * base[exponent]
                for axis in range(3):
                    shift = tuple(int(component == axis) for component in range(3))
                    raised_bra = exponent_terms(left_shift=shift)
                    raised_ket = exponent_terms(right_shift=shift)
                    lowered_bra = (
                        exponent_terms(left_shift=tuple(-item for item in shift))
                        if bra.angular[axis] else np.zeros(3)
                    )
                    lowered_ket = (
                        exponent_terms(right_shift=tuple(-item for item in shift))
                        if ket.angular[axis] else np.zeros(3)
                    )
                    left_center = (2.0 * alpha * raised_bra
                                   - bra.angular[axis] * lowered_bra)
                    right_center = (2.0 * beta * raised_ket
                                    - ket.angular[axis] * lowered_ket)
                    left_center[1] += 4.0 * alpha * raised_bra[0] / zb
                    right_center[2] += 4.0 * beta * raised_ket[0] / zk
                    result[:, 1, axis] += weight * left_center
                    result[:, 2, axis] += weight * right_center
        return result

    bra_norm = raw_pair(left, left, moments[0])
    ket_norm = raw_pair(right, right, moments[0])
    bra_log = 0.5 * (bra_norm[1, 0, 0] + bra_norm[2, 0, 0]) / bra_norm[0, 0, 0]
    ket_log = 0.5 * (ket_norm[1, 0, 0] + ket_norm[2, 0, 0]) / ket_norm[0, 0, 0]
    inverse_norm = 1.0 / sqrt(bra_norm[0, 0, 0] * ket_norm[0, 0, 0])
    output = [np.empty((13, 3)) for _ in range(4)]
    for index, moment in enumerate(moments):
        raw = raw_pair(left, right, moment)
        output[0][index] = (raw[1, 1] - bra_log * raw[0, 1]) * inverse_norm
        output[1][index] = (raw[1, 2] - bra_log * raw[0, 2]) * inverse_norm
        output[2][index] = (raw[2, 1] - ket_log * raw[0, 1]) * inverse_norm
        output[3][index] = (raw[2, 2] - ket_log * raw[0, 2]) * inverse_norm
    return tuple(output)


def contracted_multipole_exponent_second_derivatives(
    left: ContractedGaussian,
    right: ContractedGaussian,
    *, origin_bohr: Sequence[float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact normalized AO-moment ``ζζ`` blocks (bra–bra, bra–ket, ket–ket).

    STO-nG primitive exponents scale as ``ζ²``. The second primitive rule
    includes the fourth radial moment; normalization is differentiated twice
    for each *contracted* orbital, not approximated by a finite difference.
    """
    origin = np.asarray(origin_bohr, dtype=float)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("multipole origin must be a finite three-vector")
    moments = ((0, 0, 0),) + tuple(
        tuple(int(axis == component) for axis in range(3))
        for component in range(3)
    ) + tuple(
        tuple(int(axis == first) + int(axis == second) for axis in range(3))
        for first in range(3) for second in range(3)
    )
    radial_shift = tuple(
        tuple(2 * int(component == axis) for component in range(3))
        for axis in range(3)
    )

    def raw_pair(bra, ket, moment):
        components = np.zeros(6)
        zb, zk = bra.slater_exponent, ket.slater_exponent
        cb = 1.5 + sum(bra.angular)
        ck = 1.5 + sum(ket.angular)
        for primitive_bra in bra.primitives:
            for primitive_ket in ket.primitives:
                weight = primitive_bra.coefficient * primitive_ket.coefficient
                alpha = primitive_bra.exponent
                beta = primitive_ket.exponent

                def integral(left_shift=(0, 0, 0), right_shift=(0, 0, 0)):
                    return _primitive_moment(
                        bra, ket, primitive_bra, primitive_ket, moment, origin,
                        left_shift=left_shift, right_shift=right_shift,
                    )

                base = integral()
                bra2 = sum(integral(left_shift=shift) for shift in radial_shift)
                ket2 = sum(integral(right_shift=shift) for shift in radial_shift)
                bra4 = sum(
                    integral(left_shift=tuple(first[a] + second[a] for a in range(3)))
                    for first in radial_shift for second in radial_shift
                )
                ket4 = sum(
                    integral(right_shift=tuple(first[a] + second[a] for a in range(3)))
                    for first in radial_shift for second in radial_shift
                )
                bra2ket2 = sum(
                    integral(left_shift=first, right_shift=second)
                    for first in radial_shift for second in radial_shift
                )
                components += weight * np.array((
                    base,
                    (cb * base - 2.0 * alpha * bra2) / zb,
                    (ck * base - 2.0 * beta * ket2) / zk,
                    (cb * (cb - 1.0) * base
                     - 2.0 * alpha * (2.0 * cb + 1.0) * bra2
                     + 4.0 * alpha**2 * bra4) / zb**2,
                    (cb * ck * base - 2.0 * cb * beta * ket2
                     - 2.0 * ck * alpha * bra2
                     + 4.0 * alpha * beta * bra2ket2) / (zb * zk),
                    (ck * (ck - 1.0) * base
                     - 2.0 * beta * (2.0 * ck + 1.0) * ket2
                     + 4.0 * beta**2 * ket4) / zk**2,
                ))
        return components

    left_norm = raw_pair(left, left, moments[0])
    right_norm = raw_pair(right, right, moments[0])
    nl, nr = left_norm[0], right_norm[0]
    left_first = (left_norm[1] + left_norm[2]) / nl
    right_first = (right_norm[1] + right_norm[2]) / nr
    left_second = (
        left_norm[3] + 2.0 * left_norm[4] + left_norm[5]
    ) / nl
    right_second = (
        right_norm[3] + 2.0 * right_norm[4] + right_norm[5]
    ) / nr
    ll = 0.5 * left_first
    lr = 0.5 * right_first
    dll = 0.5 * (left_second - left_first**2)
    dlr = 0.5 * (right_second - right_first**2)
    inverse_norm = 1.0 / sqrt(nl * nr)
    bb = np.empty(13)
    bk = np.empty(13)
    kk = np.empty(13)
    for index, moment in enumerate(moments):
        value, bra, ket, bra_bra, bra_ket, ket_ket = raw_pair(left, right, moment)
        bb[index] = inverse_norm * (
            bra_bra - 2.0 * ll * bra + (ll**2 - dll) * value
        )
        bk[index] = inverse_norm * (
            bra_ket - lr * bra - ll * ket + ll * lr * value
        )
        kk[index] = inverse_norm * (
            ket_ket - 2.0 * lr * ket + (lr**2 - dlr) * value
        )
    return bb, bk, kk


def _contracted_moment_and_center_gradients(
    left: ContractedGaussian,
    right: ContractedGaussian,
    moment: tuple[int, int, int],
    origin: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    denominator = np.sqrt(
        _raw_contracted_overlap(left, left) * _raw_contracted_overlap(right, right)
    )
    value = 0.0
    left_gradient = np.zeros(3)
    right_gradient = np.zeros(3)
    for left_primitive in left.primitives:
        for right_primitive in right.primitives:
            weight = left_primitive.coefficient * right_primitive.coefficient
            value += weight * _primitive_moment(
                left, right, left_primitive, right_primitive, moment, origin
            )
            for axis in range(3):
                raised = tuple(int(index == axis) for index in range(3))
                left_derivative = 2.0 * left_primitive.exponent * _primitive_moment(
                    left,
                    right,
                    left_primitive,
                    right_primitive,
                    moment,
                    origin,
                    left_shift=raised,
                )
                right_derivative = 2.0 * right_primitive.exponent * _primitive_moment(
                    left,
                    right,
                    left_primitive,
                    right_primitive,
                    moment,
                    origin,
                    right_shift=raised,
                )
                if left.angular[axis] > 0:
                    lowered = tuple(-int(index == axis) for index in range(3))
                    left_derivative -= left.angular[axis] * _primitive_moment(
                        left,
                        right,
                        left_primitive,
                        right_primitive,
                        moment,
                        origin,
                        left_shift=lowered,
                    )
                if right.angular[axis] > 0:
                    lowered = tuple(-int(index == axis) for index in range(3))
                    right_derivative -= right.angular[axis] * _primitive_moment(
                        left,
                        right,
                        left_primitive,
                        right_primitive,
                        moment,
                        origin,
                        right_shift=lowered,
                    )
                left_gradient[axis] += weight * left_derivative
                right_gradient[axis] += weight * right_derivative
    return value / denominator, left_gradient / denominator, right_gradient / denominator


def _contracted_multipole_derivatives(
    left: ContractedGaussian,
    right: ContractedGaussian,
    origin: np.ndarray,
    *,
    left_norm: float,
    right_norm: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Contract all S/dipole/quadrupole centre derivatives together."""

    values0 = np.zeros(1, dtype=float)
    values1 = np.zeros(3, dtype=float)
    values2 = np.zeros((3, 3), dtype=float)
    left0 = np.zeros(3, dtype=float)
    left1 = np.zeros((3, 3), dtype=float)
    left2 = np.zeros((3, 3, 3), dtype=float)

    moments = [(0, 0, 0)]
    moments.extend(tuple(int(axis == item) for axis in range(3)) for item in range(3))
    moments.extend(
        tuple(int(axis == first) + int(axis == second) for axis in range(3))
        for first in range(3)
        for second in range(3)
    )

    for left_primitive in left.primitives:
        left_normalization = _primitive_normalization(
            left_primitive.exponent, left.angular
        )
        for right_primitive in right.primitives:
            prefactor = (
                left_primitive.coefficient
                * right_primitive.coefficient
                * left_normalization
                * _primitive_normalization(right_primitive.exponent, right.angular)
            )
            base = np.empty((3, 3), dtype=float)
            raised_left = np.empty((3, 3), dtype=float)
            lowered_left = np.zeros((3, 3), dtype=float)

            for axis in range(3):
                relative = left.center_bohr[axis] - origin[axis]

                def axis_moments(left_shift: int, right_shift: int) -> tuple[float, float, float]:
                    overlap0 = _overlap_1d(
                        left.angular[axis] + left_shift,
                        right.angular[axis] + right_shift,
                        left.center_bohr[axis],
                        right.center_bohr[axis],
                        left_primitive.exponent,
                        right_primitive.exponent,
                    )
                    overlap1 = _overlap_1d(
                        left.angular[axis] + left_shift + 1,
                        right.angular[axis] + right_shift,
                        left.center_bohr[axis],
                        right.center_bohr[axis],
                        left_primitive.exponent,
                        right_primitive.exponent,
                    )
                    overlap2 = _overlap_1d(
                        left.angular[axis] + left_shift + 2,
                        right.angular[axis] + right_shift,
                        left.center_bohr[axis],
                        right.center_bohr[axis],
                        left_primitive.exponent,
                        right_primitive.exponent,
                    )
                    return (
                        overlap0,
                        relative * overlap0 + overlap1,
                        relative * relative * overlap0 + 2.0 * relative * overlap1 + overlap2,
                    )

                base[axis] = axis_moments(0, 0)
                raised_left[axis] = axis_moments(1, 0)
                if left.angular[axis] > 0:
                    lowered_left[axis] = axis_moments(-1, 0)

            for moment_index, moment in enumerate(moments):
                value = prefactor * (
                    base[0, moment[0]]
                    * base[1, moment[1]]
                    * base[2, moment[2]]
                )
                dl = np.empty(3, dtype=float)
                for derivative_axis in range(3):
                    if derivative_axis == 0:
                        other_product = base[1, moment[1]] * base[2, moment[2]]
                    elif derivative_axis == 1:
                        other_product = base[0, moment[0]] * base[2, moment[2]]
                    else:
                        other_product = base[0, moment[0]] * base[1, moment[1]]
                    dl[derivative_axis] = prefactor * other_product * (
                        2.0
                        * left_primitive.exponent
                        * raised_left[derivative_axis, moment[derivative_axis]]
                        - left.angular[derivative_axis]
                        * lowered_left[derivative_axis, moment[derivative_axis]]
                    )
                if moment_index == 0:
                    values0[0] += value
                    left0 += dl
                elif moment_index < 4:
                    component = moment_index - 1
                    values1[component] += value
                    left1[component] += dl
                else:
                    pair_index = moment_index - 4
                    first, second_component = divmod(pair_index, 3)
                    values2[first, second_component] += value
                    left2[first, second_component] += dl

    inverse_norm = 1.0 / sqrt(left_norm * right_norm)
    values0 *= inverse_norm
    values1 *= inverse_norm
    values2 *= inverse_norm
    left0 *= inverse_norm
    left1 *= inverse_norm
    left2 *= inverse_norm
    # Translating both basis centres leaves S invariant, shifts a dipole by
    # S, and shifts a second moment by the corresponding dipoles.  These
    # exact identities recover the right-centre derivatives and remove half
    # of the primitive derivative contractions.
    right0 = -left0
    right1 = np.empty_like(left1)
    right2 = np.empty_like(left2)
    for component in range(3):
        for derivative_axis in range(3):
            right1[component, derivative_axis] = (
                (values0[0] if component == derivative_axis else 0.0)
                - left1[component, derivative_axis]
            )
            for other in range(3):
                right2[component, other, derivative_axis] = (
                    (values1[other] if component == derivative_axis else 0.0)
                    + (values1[component] if other == derivative_axis else 0.0)
                    - left2[component, other, derivative_axis]
                )
    return (
        values0,
        values1,
        values2,
        np.concatenate((left0[None, :], left1, left2.reshape(9, 3))),
        np.concatenate((right0[None, :], right1, right2.reshape(9, 3))),
    )


def atomic_multipole_integral_derivatives(
    orbitals: Sequence[ContractedGaussian],
    *,
    origin_bohr: Sequence[float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return analytic nuclear derivatives of S, dipole, and second moments.

    Shapes are ``(natoms, 3, nao, nao)``, ``(3, natoms, 3, nao, nao)``,
    and ``(3, 3, natoms, 3, nao, nao)`` respectively.
    """

    origin = np.asarray(origin_bohr, dtype=float)
    natoms = max((orbital.atom_index for orbital in orbitals), default=-1) + 1
    size = len(orbitals)
    overlap = np.zeros((natoms, 3, size, size))
    dipole = np.zeros((3, natoms, 3, size, size))
    second = np.zeros((3, 3, natoms, 3, size, size))
    norms = tuple(_raw_contracted_overlap(orbital, orbital) for orbital in orbitals)
    compiled_pairs = _compiled_multipole_pairs(orbitals, origin, norms)
    if compiled_pairs is not None:
        pair_left, pair_right, _, pair_dl, pair_dr = compiled_pairs
        left_atoms = np.fromiter(
            (orbitals[index].atom_index for index in pair_left),
            dtype=np.intp,
            count=len(pair_left),
        )
        right_atoms = np.fromiter(
            (orbitals[index].atom_index for index in pair_right),
            dtype=np.intp,
            count=len(pair_right),
        )
        axes = np.arange(3, dtype=np.intp)
        overlap_index = (
            left_atoms[:, None], axes[None, :],
            pair_left[:, None], pair_right[:, None],
        )
        np.add.at(overlap, overlap_index, pair_dl[:, 0, :])
        overlap_index = (
            right_atoms[:, None], axes[None, :],
            pair_left[:, None], pair_right[:, None],
        )
        np.add.at(overlap, overlap_index, pair_dr[:, 0, :])

        moments = np.arange(3, dtype=np.intp)
        dipole_index = (
            moments[None, :, None], left_atoms[:, None, None], axes[None, None, :],
            pair_left[:, None, None], pair_right[:, None, None],
        )
        np.add.at(dipole, dipole_index, pair_dl[:, 1:4, :])
        dipole_index = (
            moments[None, :, None], right_atoms[:, None, None], axes[None, None, :],
            pair_left[:, None, None], pair_right[:, None, None],
        )
        np.add.at(dipole, dipole_index, pair_dr[:, 1:4, :])

        first = np.arange(3, dtype=np.intp)
        second_component = np.arange(3, dtype=np.intp)
        second_index = (
            first[None, :, None, None], second_component[None, None, :, None],
            left_atoms[:, None, None, None], axes[None, None, None, :],
            pair_left[:, None, None, None], pair_right[:, None, None, None],
        )
        np.add.at(second, second_index, pair_dl[:, 4:, :].reshape(-1, 3, 3, 3))
        second_index = (
            first[None, :, None, None], second_component[None, None, :, None],
            right_atoms[:, None, None, None], axes[None, None, None, :],
            pair_left[:, None, None, None], pair_right[:, None, None, None],
        )
        np.add.at(second, second_index, pair_dr[:, 4:, :].reshape(-1, 3, 3, 3))

        off_diagonal = pair_left != pair_right
        upper = pair_left[off_diagonal]
        lower = pair_right[off_diagonal]
        overlap[:, :, lower, upper] = overlap[:, :, upper, lower]
        dipole[:, :, :, lower, upper] = dipole[:, :, :, upper, lower]
        second[:, :, :, :, lower, upper] = second[:, :, :, :, upper, lower]
        return overlap, dipole, second
    for left_index, left in enumerate(orbitals):
        for right_index in range(left_index + 1):
            right = orbitals[right_index]
            if _contracted_multipole_derivatives_compiled is not None:
                _, dl_all, dr_all = _contracted_multipole_derivatives_compiled(
                    np.asarray(left.center_bohr, dtype=float),
                    np.asarray(right.center_bohr, dtype=float),
                    np.asarray(left.angular, dtype=np.int64),
                    np.asarray(right.angular, dtype=np.int64),
                    np.asarray([item.exponent for item in left.primitives], dtype=float),
                    np.asarray([item.coefficient for item in left.primitives], dtype=float),
                    np.asarray([item.exponent for item in right.primitives], dtype=float),
                    np.asarray([item.coefficient for item in right.primitives], dtype=float),
                    origin,
                    sqrt(norms[left_index] * norms[right_index]),
                )
            else:
                _, _, _, dl_all, dr_all = _contracted_multipole_derivatives(
                    left,
                    right,
                    origin,
                    left_norm=norms[left_index],
                    right_norm=norms[right_index],
                )
            dl, dr = dl_all[0], dr_all[0]
            overlap[left.atom_index, :, left_index, right_index] += dl
            overlap[right.atom_index, :, left_index, right_index] += dr
            for component in range(3):
                dl, dr = dl_all[1 + component], dr_all[1 + component]
                dipole[component, left.atom_index, :, left_index, right_index] += dl
                dipole[component, right.atom_index, :, left_index, right_index] += dr
                for other in range(3):
                    index = 4 + 3 * component + other
                    dl, dr = dl_all[index], dr_all[index]
                    second[component, other, left.atom_index, :, left_index, right_index] += dl
                    second[component, other, right.atom_index, :, left_index, right_index] += dr
            if right_index != left_index:
                overlap[:, :, right_index, left_index] = overlap[
                    :, :, left_index, right_index
                ]
                dipole[:, :, :, right_index, left_index] = dipole[
                    :, :, :, left_index, right_index
                ]
                second[:, :, :, :, right_index, left_index] = second[
                    :, :, :, :, left_index, right_index
                ]
    return overlap, dipole, second


__all__ = [
    "ContractedGaussian",
    "GaussianPrimitive",
    "SparseAtomicMultipoleIntegrals",
    "atomic_multipole_integrals",
    "atomic_multipole_integral_derivatives",
    "contracted_overlap",
    "contracted_overlap_gradient",
    "contracted_gaussian_values",
    "overlap_matrix",
    "sparse_atomic_multipole_integrals",
]
