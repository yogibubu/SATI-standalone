"""Published zeroth-order GFN2-xTB energy and Hamiltonian terms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from ._array_cache import triangular_indices

from .basis import ContractedGaussian, SparseAtomicMultipoleIntegrals
from .kernels import batched_gfn2_coordination
from .localized_solver import SparseSymmetricMatrix


EV_TO_HARTREE = 1.0 / 27.21138505
# The distributed GFN2 parameter set stores ``enscale = +0.02``.  The minus
# sign printed in Table 2 of the article is inconsistent with the reference
# implementation and numerical corpus; the machine-readable parametrization
# is authoritative for reproducibility.
K_EN = 0.02


@dataclass(frozen=True)
class AtomParameters:
    alpha: float
    effective_nuclear_charge: float
    covalent_radius_bohr: float
    d4_covalent_radius_bohr: float
    electronegativity: float


@dataclass(frozen=True)
class OrbitalParameters:
    level_hartree: float
    coordination_shift_hartree: float
    polynomial: float


def _atom_values(
    numbers: Sequence[int],
    parameters: Mapping[int, AtomParameters] | Sequence[AtomParameters],
) -> tuple[AtomParameters, ...]:
    if isinstance(parameters, Mapping):
        return tuple(parameters[number] for number in numbers)
    if len(parameters) != len(numbers):
        raise ValueError("per-atom Hamiltonian parameter count differs from atom count")
    return tuple(parameters)


def _orbital_value(
    numbers: Sequence[int], atom: int, shell: str,
    parameters: Mapping[tuple[int, str], OrbitalParameters]
    | Sequence[Mapping[str, OrbitalParameters]],
) -> OrbitalParameters:
    if isinstance(parameters, Mapping):
        return parameters[(numbers[atom], shell)]
    if len(parameters) != len(numbers):
        raise ValueError("per-atom orbital parameter count differs from atom count")
    return parameters[atom][shell]


@dataclass(frozen=True)
class SparseHamiltonianDerivatives:
    left: np.ndarray
    right: np.ndarray
    coordination_left: np.ndarray
    coordination_right: np.ndarray


@dataclass(frozen=True)
class SparseHamiltonianDirectionalDerivatives:
    left: np.ndarray
    right: np.ndarray
    coordination_left: np.ndarray
    coordination_right: np.ndarray


def coordination_numbers(
    atomic_numbers: Sequence[int],
    coordinates_bohr: np.ndarray,
    atom_parameters: Mapping[int, AtomParameters] | Sequence[AtomParameters],
) -> np.ndarray:
    """Evaluate the smooth GFN2 coordination number from published Eq. 18."""

    numbers = tuple(int(value) for value in atomic_numbers)
    coordinates = np.asarray(coordinates_bohr, dtype=float)
    if coordinates.shape != (len(numbers), 3):
        raise ValueError("coordinates_bohr must have shape natoms x 3")
    displacement = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.linalg.norm(displacement, axis=2)
    off_diagonal = ~np.eye(len(numbers), dtype=bool)
    if np.any(distances[off_diagonal] <= 0.0):
        raise ValueError("distinct atoms cannot occupy the same point")
    radii = np.asarray([item.covalent_radius_bohr
                        for item in _atom_values(numbers, atom_parameters)])
    return np.asarray(batched_gfn2_coordination(coordinates, radii), dtype=float)


def _angular_scale(left_shell: str, right_shell: str) -> float:
    pair = "".join(sorted((left_shell[-1], right_shell[-1])))
    if pair == "ss":
        return 1.85
    if pair in {"pp", "dd"}:
        return 2.23
    if pair == "ps":
        return 2.04
    return 2.00


_ANGULAR_SCALE = np.asarray(
    (
        (1.85, 2.04, 2.00, 2.00),
        (2.04, 2.23, 2.00, 2.00),
        (2.00, 2.00, 2.23, 2.00),
        (2.00, 2.00, 2.00, 2.00),
    ),
    dtype=float,
)


def _h0_global_scalings(globals: Mapping[str, float] | None):
    if globals is None:
        return _ANGULAR_SCALE, K_EN
    ks = float(globals["ks"])
    kp = float(globals["kp"])
    kd = float(globals["kd"])
    ksd = float(globals["ksd"])
    kpd = float(globals["kpd"])
    angular = np.asarray((
        (ks, 0.5 * (ks + kp), ksd, ksd),
        (0.5 * (ks + kp), kp, kpd, kpd),
        (ksd, kpd, kd, kd),
        (ksd, kpd, kd, kd),
    ), dtype=float)
    enscale = float(globals["enscale"])
    if not np.all(np.isfinite(angular)) or not np.isfinite(enscale):
        raise ValueError("nonfinite NN-xTB H0 global scaling")
    return angular, enscale


def _angular_classes(shells: Sequence[str]) -> np.ndarray:
    """Encode shell angular labels for vectorized pair prefactors."""

    lookup = {"s": 0, "p": 1, "d": 2}
    return np.fromiter(
        (lookup.get(str(shell)[-1], 3) for shell in shells),
        dtype=np.intp,
        count=len(shells),
    )


def _slater_pair_scaling(left, right, global_scalings=None):
    """GFN2 Slater-overlap prefactor with its learned global exponent."""
    exponent = 0.5 if global_scalings is None else global_scalings.get("wexp", 0.5)
    return (2.0 * np.sqrt(left * right) / (left + right)) ** exponent


def extended_hueckel_hamiltonian(
    overlap: np.ndarray,
    orbitals: Sequence[ContractedGaussian],
    atomic_numbers: Sequence[int],
    coordinates_bohr: np.ndarray,
    atom_parameters: Mapping[int, AtomParameters] | Sequence[AtomParameters],
    orbital_parameters: Mapping[tuple[int, str], OrbitalParameters]
    | Sequence[Mapping[str, OrbitalParameters]],
    *, global_scalings: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Build the environment-dependent extended-Hueckel matrix (Eqs. 16-19)."""

    overlap = np.asarray(overlap, dtype=float)
    if overlap.shape != (len(orbitals), len(orbitals)):
        raise ValueError("overlap must be square with one row per orbital")
    numbers = tuple(int(value) for value in atomic_numbers)
    coordinates = np.asarray(coordinates_bohr, dtype=float)
    angular_scale, enscale = _h0_global_scalings(global_scalings)
    coordination = coordination_numbers(numbers, coordinates, atom_parameters)
    levels = np.empty(len(orbitals), dtype=float)
    atom_index = np.fromiter((item.atom_index for item in orbitals), dtype=np.intp)
    slater = np.fromiter((item.slater_exponent for item in orbitals), dtype=float)
    polynomial_parameter = np.empty(len(orbitals), dtype=float)
    angular_label = []
    for index, orbital in enumerate(orbitals):
        parameters = _orbital_value(
            numbers, orbital.atom_index, orbital.shell, orbital_parameters,
        )
        levels[index] = (
            parameters.level_hartree
            - parameters.coordination_shift_hartree * coordination[orbital.atom_index]
        )
        polynomial_parameter[index] = parameters.polynomial
        angular_label.append(orbital.shell)
    result = np.diag(levels)
    left, right = triangular_indices(len(orbitals), diagonal=False)
    distinct = atom_index[left] != atom_index[right]
    left, right = left[distinct], right[distinct]
    if left.size:
        left_atom, right_atom = atom_index[left], atom_index[right]
        displacement = coordinates[left_atom] - coordinates[right_atom]
        distance = np.linalg.norm(displacement, axis=1)
        atom_values = _atom_values(numbers, atom_parameters)
        electronegativity = np.asarray([item.electronegativity for item in atom_values])
        radii = np.asarray([item.d4_covalent_radius_bohr for item in atom_values])
        exponent_factor = _slater_pair_scaling(slater[left], slater[right], global_scalings)
        root = np.sqrt(distance / (0.5 * (radii[left_atom] + radii[right_atom])))
        polynomial = (1.0 + polynomial_parameter[left] * root) * (
            1.0 + polynomial_parameter[right] * root
        )
        angular_class = _angular_classes(angular_label)
        angular = angular_scale[angular_class[left], angular_class[right]]
        delta_en = electronegativity[left_atom] - electronegativity[right_atom]
        value = (
            0.5 * angular * (levels[left] + levels[right]) * overlap[left, right]
            * exponent_factor * (1.0 + enscale * delta_en**2) * polynomial
        )
        result[left, right] = value
        result[right, left] = value
    return result


def extended_hueckel_hamiltonian_sparse(
    overlap: SparseSymmetricMatrix,
    orbitals: Sequence[ContractedGaussian],
    atomic_numbers: Sequence[int],
    coordinates_bohr: np.ndarray,
    atom_parameters: Mapping[int, AtomParameters] | Sequence[AtomParameters],
    orbital_parameters: Mapping[tuple[int, str], OrbitalParameters]
    | Sequence[Mapping[str, OrbitalParameters]],
    *,
    coordination_override: np.ndarray | None = None,
    global_scalings: Mapping[str, float] | None = None,
) -> SparseSymmetricMatrix:
    """Build H0 directly on the overlap pair pattern."""

    if overlap.size != len(orbitals):
        raise ValueError("sparse overlap must contain one row per orbital")
    numbers = tuple(int(value) for value in atomic_numbers)
    coordinates = np.asarray(coordinates_bohr, dtype=float)
    angular_scale, enscale = _h0_global_scalings(global_scalings)
    coordination = (
        coordination_numbers(numbers, coordinates, atom_parameters)
        if coordination_override is None
        else np.asarray(coordination_override, dtype=float)
    )
    if coordination.shape != (len(numbers),):
        raise ValueError("coordination override has an inconsistent atom count")
    atom_index = np.fromiter((item.atom_index for item in orbitals), dtype=np.intp)
    slater = np.fromiter((item.slater_exponent for item in orbitals), dtype=float)
    levels = np.empty(overlap.size, dtype=float)
    polynomial_parameter = np.empty(overlap.size, dtype=float)
    labels: list[str] = []
    for index, orbital in enumerate(orbitals):
        item = _orbital_value(numbers, orbital.atom_index, orbital.shell, orbital_parameters)
        levels[index] = (
            item.level_hartree
            - item.coordination_shift_hartree * coordination[orbital.atom_index]
        )
        polynomial_parameter[index] = item.polynomial
        labels.append(orbital.shell)
    left, right = overlap.rows, overlap.columns
    values = np.zeros_like(overlap.values)
    diagonal = left == right
    values[diagonal] = levels[left[diagonal]]
    distinct = (~diagonal) & (atom_index[left] != atom_index[right])
    if np.any(distinct):
        lp, rp = left[distinct], right[distinct]
        left_atom, right_atom = atom_index[lp], atom_index[rp]
        distance = np.linalg.norm(
            coordinates[left_atom] - coordinates[right_atom], axis=1
        )
        atom_values = _atom_values(numbers, atom_parameters)
        electronegativity = np.asarray([item.electronegativity for item in atom_values])
        radii = np.asarray([item.d4_covalent_radius_bohr for item in atom_values])
        exponent_factor = _slater_pair_scaling(slater[lp], slater[rp], global_scalings)
        root = np.sqrt(distance / (0.5 * (radii[left_atom] + radii[right_atom])))
        polynomial = (1.0 + polynomial_parameter[lp] * root) * (
            1.0 + polynomial_parameter[rp] * root
        )
        angular_class = _angular_classes(labels)
        angular = angular_scale[angular_class[lp], angular_class[rp]]
        delta_en = electronegativity[left_atom] - electronegativity[right_atom]
        values[distinct] = (
            0.5 * angular * (levels[lp] + levels[rp]) * overlap.values[distinct]
            * exponent_factor * (1.0 + enscale * delta_en**2) * polynomial
        )
    return SparseSymmetricMatrix(
        overlap.size, left.copy(), right.copy(), values
    )


def extended_hueckel_hamiltonian_derivatives_sparse(
    integrals: SparseAtomicMultipoleIntegrals,
    orbitals: Sequence[ContractedGaussian],
    atomic_numbers: Sequence[int],
    coordinates_bohr: np.ndarray,
    atom_parameters: Mapping[int, AtomParameters] | Sequence[AtomParameters],
    orbital_parameters: Mapping[tuple[int, str], OrbitalParameters]
    | Sequence[Mapping[str, OrbitalParameters]],
    coordination: np.ndarray,
    *, global_scalings: Mapping[str, float] | None = None,
) -> SparseHamiltonianDerivatives:
    """Pair-endpoint H0 derivatives and derivatives with respect to atomic CN."""

    numbers = tuple(int(value) for value in atomic_numbers)
    coordinates = np.asarray(coordinates_bohr, dtype=float)
    angular_scale, enscale = _h0_global_scalings(global_scalings)
    cn = np.asarray(coordination, dtype=float)
    rows, columns = integrals.rows, integrals.columns
    atom = np.fromiter((item.atom_index for item in orbitals), dtype=np.intp)
    slater = np.fromiter((item.slater_exponent for item in orbitals), dtype=float)
    level = np.empty(len(orbitals))
    shift = np.empty(len(orbitals))
    poly = np.empty(len(orbitals))
    labels: list[str] = []
    for index, orbital in enumerate(orbitals):
        parameter = _orbital_value(numbers, orbital.atom_index, orbital.shell, orbital_parameters)
        level[index] = parameter.level_hartree - parameter.coordination_shift_hartree * cn[orbital.atom_index]
        shift[index] = parameter.coordination_shift_hartree
        poly[index] = parameter.polynomial
        labels.append(orbital.shell)
    left_derivative = np.zeros((len(rows), 3))
    right_derivative = np.zeros_like(left_derivative)
    cn_left = np.zeros(len(rows))
    cn_right = np.zeros(len(rows))
    distinct = (rows != columns) & (atom[rows] != atom[columns])
    if np.any(distinct):
        indices = np.flatnonzero(distinct)
        left, right = rows[indices], columns[indices]
        left_atom, right_atom = atom[left], atom[right]
        displacement = coordinates[left_atom] - coordinates[right_atom]
        distance = np.linalg.norm(displacement, axis=1)
        atom_values = _atom_values(numbers, atom_parameters)
        radii = np.asarray([item.d4_covalent_radius_bohr for item in atom_values])
        electronegativity = np.asarray([item.electronegativity for item in atom_values])
        radius = 0.5 * (radii[left_atom] + radii[right_atom])
        root = np.sqrt(distance / radius)
        left_poly = 1.0 + poly[left] * root
        right_poly = 1.0 + poly[right] * root
        polynomial = left_poly * right_poly
        radial = (poly[left] * right_poly + poly[right] * left_poly) / (2.0 * np.sqrt(distance * radius))
        exponent = _slater_pair_scaling(slater[left], slater[right], global_scalings)
        angular_class = _angular_classes(labels)
        angular = angular_scale[angular_class[left], angular_class[right]]
        delta = electronegativity[left_atom] - electronegativity[right_atom]
        prefactor = 0.5 * angular * exponent * (1.0 + enscale * delta**2)
        level_sum = level[left] + level[right]
        overlap = integrals.overlap[indices]
        radial_vector = radial[:, None] * displacement / distance[:, None]
        common = prefactor * level_sum
        left_derivative[indices] = common[:, None] * (
            polynomial[:, None] * integrals.left_derivatives[indices, 0]
            + overlap[:, None] * radial_vector
        )
        right_derivative[indices] = common[:, None] * (
            polynomial[:, None] * integrals.right_derivatives[indices, 0]
            - overlap[:, None] * radial_vector
        )
        cn_left[indices] = -prefactor * overlap * polynomial * shift[left]
        cn_right[indices] = -prefactor * overlap * polynomial * shift[right]
    diagonal = rows == columns
    cn_left[diagonal] = -shift[rows[diagonal]]
    return SparseHamiltonianDerivatives(
        left_derivative, right_derivative, cn_left, cn_right
    )


def extended_hueckel_hamiltonian_directional_derivatives_sparse(
    integrals: SparseAtomicMultipoleIntegrals,
    overlap_left_directional: np.ndarray,
    overlap_right_directional: np.ndarray,
    orbitals: Sequence[ContractedGaussian],
    atomic_numbers: Sequence[int],
    coordinates_bohr: np.ndarray,
    direction_bohr: np.ndarray,
    atom_parameters: Mapping[int, AtomParameters] | Sequence[AtomParameters],
    orbital_parameters: Mapping[tuple[int, str], OrbitalParameters]
    | Sequence[Mapping[str, OrbitalParameters]],
    coordination: np.ndarray,
    coordination_directional: np.ndarray,
    *, global_scalings: Mapping[str, float] | None = None,
) -> SparseHamiltonianDirectionalDerivatives:
    """Differentiate pair-endpoint H0 gradients along one nuclear direction."""

    numbers = tuple(int(value) for value in atomic_numbers)
    coordinates = np.asarray(coordinates_bohr, dtype=float)
    angular_scale, enscale = _h0_global_scalings(global_scalings)
    direction = np.asarray(direction_bohr, dtype=float)
    cn = np.asarray(coordination, dtype=float)
    cn_direction = np.asarray(coordination_directional, dtype=float)
    rows, columns = integrals.rows, integrals.columns
    left_overlap_directional = np.asarray(overlap_left_directional, dtype=float)
    right_overlap_directional = np.asarray(overlap_right_directional, dtype=float)
    if left_overlap_directional.shape != (len(rows), 3) or (
        right_overlap_directional.shape != (len(rows), 3)
    ):
        raise ValueError("directional overlap gradients must have shape npairs x 3")
    atom = np.fromiter((item.atom_index for item in orbitals), dtype=np.intp)
    slater = np.fromiter((item.slater_exponent for item in orbitals), dtype=float)
    level = np.empty(len(orbitals))
    level_direction = np.empty(len(orbitals))
    shift = np.empty(len(orbitals))
    polynomial_parameter = np.empty(len(orbitals))
    labels: list[str] = []
    for index, orbital in enumerate(orbitals):
        parameter = _orbital_value(numbers, orbital.atom_index, orbital.shell, orbital_parameters)
        level[index] = (
            parameter.level_hartree
            - parameter.coordination_shift_hartree * cn[orbital.atom_index]
        )
        shift[index] = parameter.coordination_shift_hartree
        level_direction[index] = -shift[index] * cn_direction[orbital.atom_index]
        polynomial_parameter[index] = parameter.polynomial
        labels.append(orbital.shell)

    left_result = np.zeros((len(rows), 3), dtype=float)
    right_result = np.zeros_like(left_result)
    cn_left_result = np.zeros(len(rows), dtype=float)
    cn_right_result = np.zeros(len(rows), dtype=float)
    distinct = (rows != columns) & (atom[rows] != atom[columns])
    if np.any(distinct):
        indices = np.flatnonzero(distinct)
        left, right = rows[indices], columns[indices]
        left_atom, right_atom = atom[left], atom[right]
        displacement = coordinates[left_atom] - coordinates[right_atom]
        relative_direction = direction[left_atom] - direction[right_atom]
        distance = np.linalg.norm(displacement, axis=1)
        unit = displacement / distance[:, None]
        distance_direction = np.einsum("ij,ij->i", unit, relative_direction)
        unit_direction = (
            relative_direction - distance_direction[:, None] * unit
        ) / distance[:, None]
        atom_values = _atom_values(numbers, atom_parameters)
        radii = np.asarray([item.d4_covalent_radius_bohr for item in atom_values])
        electronegativity = np.asarray([item.electronegativity for item in atom_values])
        radius = 0.5 * (radii[left_atom] + radii[right_atom])
        root = np.sqrt(distance / radius)
        left_polynomial = 1.0 + polynomial_parameter[left] * root
        right_polynomial = 1.0 + polynomial_parameter[right] * root
        polynomial = left_polynomial * right_polynomial
        radial = (
            polynomial_parameter[left] * right_polynomial
            + polynomial_parameter[right] * left_polynomial
        ) / (2.0 * np.sqrt(distance * radius))
        radial_second = -(
            polynomial_parameter[left] + polynomial_parameter[right]
        ) / (4.0 * np.sqrt(radius) * distance**1.5)
        angular_class = _angular_classes(labels)
        angular = angular_scale[angular_class[left], angular_class[right]]
        exponent_factor = _slater_pair_scaling(slater[left], slater[right], global_scalings)
        delta_en = electronegativity[left_atom] - electronegativity[right_atom]
        prefactor = 0.5 * angular * exponent_factor * (1.0 + enscale * delta_en**2)
        level_sum = level[left] + level[right]
        level_sum_direction = level_direction[left] + level_direction[right]
        overlap = integrals.overlap[indices]
        overlap_left = integrals.left_derivatives[indices, 0]
        overlap_right = integrals.right_derivatives[indices, 0]
        overlap_direction = np.einsum(
            "ij,ij->i", overlap_left, direction[left_atom]
        ) + np.einsum("ij,ij->i", overlap_right, direction[right_atom])
        polynomial_direction = radial * distance_direction
        radial_vector = radial[:, None] * unit
        radial_vector_direction = (
            radial_second[:, None] * distance_direction[:, None] * unit
            + radial[:, None] * unit_direction
        )

        def endpoint_directional(overlap_gradient, overlap_gradient_directional, sign):
            base = polynomial[:, None] * overlap_gradient + (
                sign * overlap[:, None] * radial_vector
            )
            base_direction = (
                polynomial_direction[:, None] * overlap_gradient
                + polynomial[:, None] * overlap_gradient_directional
                + sign
                * (
                    overlap_direction[:, None] * radial_vector
                    + overlap[:, None] * radial_vector_direction
                )
            )
            return prefactor[:, None] * (
                level_sum_direction[:, None] * base
                + level_sum[:, None] * base_direction
            )

        left_result[indices] = endpoint_directional(
            overlap_left, left_overlap_directional[indices], 1.0
        )
        right_result[indices] = endpoint_directional(
            overlap_right, right_overlap_directional[indices], -1.0
        )
        common_direction = overlap_direction * polynomial + overlap * polynomial_direction
        cn_left_result[indices] = -prefactor * shift[left] * common_direction
        cn_right_result[indices] = -prefactor * shift[right] * common_direction
    return SparseHamiltonianDirectionalDerivatives(
        left_result, right_result, cn_left_result, cn_right_result
    )


def extended_hueckel_hamiltonian_derivative(
    overlap: np.ndarray,
    overlap_derivative: np.ndarray,
    orbitals: Sequence[ContractedGaussian],
    atomic_numbers: Sequence[int],
    coordinates_bohr: np.ndarray,
    atom_parameters: Mapping[int, AtomParameters] | Sequence[AtomParameters],
    orbital_parameters: Mapping[tuple[int, str], OrbitalParameters]
    | Sequence[Mapping[str, OrbitalParameters]],
    coordination_jacobian: np.ndarray,
    *, global_scalings: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Return the analytic nuclear derivative of the GFN2 H0 matrix."""

    overlap = np.asarray(overlap, dtype=float)
    derivative_overlap = np.asarray(overlap_derivative, dtype=float)
    coordinates = np.asarray(coordinates_bohr, dtype=float)
    angular_scale, enscale = _h0_global_scalings(global_scalings)
    numbers = tuple(int(value) for value in atomic_numbers)
    natoms = len(numbers)
    size = len(orbitals)
    if derivative_overlap.shape != (natoms, 3, size, size):
        raise ValueError("overlap derivative has an incompatible shape")
    coordination = coordination_numbers(numbers, coordinates, atom_parameters)
    levels = np.empty(size)
    level_derivative = np.zeros((natoms, 3, size))
    for index, orbital in enumerate(orbitals):
        parameters = _orbital_value(
            numbers, orbital.atom_index, orbital.shell, orbital_parameters,
        )
        levels[index] = (
            parameters.level_hartree
            - parameters.coordination_shift_hartree * coordination[orbital.atom_index]
        )
        level_derivative[:, :, index] = (
            -parameters.coordination_shift_hartree
            * coordination_jacobian[orbital.atom_index]
        )
    result = np.zeros_like(derivative_overlap)
    diagonal = np.arange(size)
    result[:, :, diagonal, diagonal] = level_derivative
    atom_index = np.fromiter((item.atom_index for item in orbitals), dtype=np.intp)
    slater = np.fromiter((item.slater_exponent for item in orbitals), dtype=float)
    polynomial_parameter = np.empty(size, dtype=float)
    angular_label = []
    for index, orbital in enumerate(orbitals):
        polynomial_parameter[index] = _orbital_value(
            numbers, orbital.atom_index, orbital.shell, orbital_parameters,
        ).polynomial
        angular_label.append(orbital.shell)
    left, right = triangular_indices(size, diagonal=False)
    distinct = atom_index[left] != atom_index[right]
    left, right = left[distinct], right[distinct]
    if left.size:
        left_atom, right_atom = atom_index[left], atom_index[right]
        displacement = coordinates[left_atom] - coordinates[right_atom]
        distance = np.linalg.norm(displacement, axis=1)
        atom_values = _atom_values(numbers, atom_parameters)
        radii = np.asarray([item.d4_covalent_radius_bohr for item in atom_values])
        electronegativity = np.asarray([item.electronegativity for item in atom_values])
        radius = 0.5 * (radii[left_atom] + radii[right_atom])
        root = np.sqrt(distance / radius)
        left_polynomial = 1.0 + polynomial_parameter[left] * root
        right_polynomial = 1.0 + polynomial_parameter[right] * root
        polynomial = left_polynomial * right_polynomial
        polynomial_radial = (
            polynomial_parameter[left] * right_polynomial
            + polynomial_parameter[right] * left_polynomial
        ) / (2.0 * np.sqrt(distance * radius))
        exponent_factor = _slater_pair_scaling(slater[left], slater[right], global_scalings)
        angular_class = _angular_classes(angular_label)
        angular = angular_scale[angular_class[left], angular_class[right]]
        delta_en = electronegativity[left_atom] - electronegativity[right_atom]
        prefactor = 0.5 * angular * exponent_factor * (1.0 + enscale * delta_en**2)
        level_sum = levels[left] + levels[right]
        value = prefactor[:, None, None] * (
            (level_derivative[:, :, left] + level_derivative[:, :, right]).transpose(2, 0, 1)
            * overlap[left, right, None, None]
            * polynomial[:, None, None]
            + level_sum[:, None, None]
            * derivative_overlap[:, :, left, right].transpose(2, 0, 1)
            * polynomial[:, None, None]
        )
        radial_vector = polynomial_radial[:, None] * displacement / distance[:, None]
        radial_value = (
            prefactor * level_sum * overlap[left, right]
        )[:, None] * radial_vector
        value[np.arange(len(left)), left_atom] += radial_value
        value[np.arange(len(left)), right_atom] -= radial_value
        result[:, :, left, right] = value.transpose(1, 2, 0)
        result[:, :, right, left] = value.transpose(1, 2, 0)
    return result


def repulsion_energy(
    atomic_numbers: Sequence[int],
    coordinates_bohr: np.ndarray,
    atom_parameters: Mapping[int, AtomParameters] | Sequence[AtomParameters],
    *, kexp: float = 1.5, klight: float = 1.0,
) -> float:
    """Evaluate the pairwise classical GFN2 repulsion from published Eq. 9."""

    numbers = tuple(int(value) for value in atomic_numbers)
    coordinates = np.asarray(coordinates_bohr, dtype=float)
    if coordinates.shape != (len(numbers), 3):
        raise ValueError("coordinates_bohr must have shape natoms x 3")
    if len(numbers) < 2:
        return 0.0
    if not np.isfinite(kexp) or kexp <= 0.0:
        raise ValueError("repulsion exponent must be positive and finite")
    number_array = np.asarray(numbers, dtype=np.intp)
    atom_values = _atom_values(numbers, atom_parameters)
    charge = np.asarray([item.effective_nuclear_charge for item in atom_values], dtype=float)
    alpha = np.asarray([item.alpha for item in atom_values], dtype=float)
    left, right = triangular_indices(len(numbers), diagonal=False)
    distance = np.linalg.norm(coordinates[left] - coordinates[right], axis=1)
    if np.any(distance <= 0.0):
        raise ValueError("distinct atoms cannot occupy the same point")
    exponent = np.where(
        (number_array[left] <= 2) & (number_array[right] <= 2), klight, kexp
    )
    values = (
        charge[left]
        * charge[right]
        / distance
        * np.exp(-np.sqrt(alpha[left] * alpha[right]) * distance**exponent)
    )
    return float(np.sum(values))


def repulsion_energy_sparse(
    atomic_numbers: Sequence[int], coordinates_bohr: np.ndarray,
    atom_parameters: Mapping[int, AtomParameters] | Sequence[AtomParameters],
    *, cutoff_bohr: float = 24.0, kexp: float = 1.5, klight: float = 1.0,
) -> float:
    from .kernels import sparse_pair_geometry

    numbers = np.asarray(tuple(atomic_numbers), dtype=np.intp)
    if not np.isfinite(kexp) or kexp <= 0.0:
        raise ValueError("repulsion exponent must be positive and finite")
    pairs = sparse_pair_geometry(coordinates_bohr, cutoff_bohr=cutoff_bohr)
    atom_values = _atom_values(numbers, atom_parameters)
    left_parameters = [atom_values[index] for index in pairs.left]
    right_parameters = [atom_values[index] for index in pairs.right]
    charge = np.asarray([item.effective_nuclear_charge for item in left_parameters]) * np.asarray([item.effective_nuclear_charge for item in right_parameters])
    alpha = np.sqrt(np.asarray([item.alpha for item in left_parameters]) * np.asarray([item.alpha for item in right_parameters]))
    exponent = np.where((numbers[pairs.left] <= 2) & (numbers[pairs.right] <= 2), klight, kexp)
    return float(np.sum(charge / pairs.distance * np.exp(-alpha * pairs.distance**exponent)))


__all__ = [
    "AtomParameters",
    "EV_TO_HARTREE",
    "OrbitalParameters",
    "coordination_numbers",
    "extended_hueckel_hamiltonian",
    "extended_hueckel_hamiltonian_sparse",
    "extended_hueckel_hamiltonian_derivatives_sparse",
    "extended_hueckel_hamiltonian_directional_derivatives_sparse",
    "SparseHamiltonianDerivatives",
    "SparseHamiltonianDirectionalDerivatives",
    "extended_hueckel_hamiltonian_derivative",
    "repulsion_energy",
    "repulsion_energy_sparse",
]
