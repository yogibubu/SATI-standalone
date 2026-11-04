"""Exact nuclear derivatives for individual MATRIX-GFN2 energy kernels."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from functools import lru_cache
from math import pi, sqrt
from typing import Mapping, Sequence

import numpy as np

try:
    from . import _gfn2_native
except ImportError:  # pragma: no cover - portable source-tree fallback
    _gfn2_native = None

from .hamiltonian import AtomParameters, _atom_values, coordination_numbers
from .dispersion import D4CoefficientDirection, SparseATMResult, SparseD4Result
from .locality import LocalizedOrbitalDomains
from .localized_solver import SparseSymmetricMatrix
from .multipole import AnisotropicParameters, AtomicMultipoles
from .scc import ShellPartition, monopole_energy, shell_interaction_matrix


def _pair_antisymmetric_gradient(
    left: np.ndarray,
    right: np.ndarray,
    contribution: np.ndarray,
    natoms: int,
) -> np.ndarray:
    """Scatter +pair/-pair vectors without NumPy's serialized ``add.at``."""

    if _gfn2_native is not None and hasattr(
        _gfn2_native, "pair_antisymmetric_gradient"
    ):
        return np.asarray(
            _gfn2_native.pair_antisymmetric_gradient(
                left, right, contribution, int(natoms)
            ),
            dtype=float,
        )
    result = np.zeros((natoms, 3), dtype=float)
    np.add.at(result, left, contribution)
    np.add.at(result, right, -contribution)
    return result


def _pair_symmetric_scalar(
    left: np.ndarray,
    right: np.ndarray,
    values: np.ndarray,
    natoms: int,
) -> np.ndarray:
    if _gfn2_native is not None and hasattr(_gfn2_native, "pair_symmetric_scalar"):
        return np.asarray(
            _gfn2_native.pair_symmetric_scalar(left, right, values, int(natoms)),
            dtype=float,
        )
    result = np.zeros(natoms, dtype=float)
    np.add.at(result, left, values)
    np.add.at(result, right, values)
    return result


def _radial_pair_hessian_vector_product(
    left: np.ndarray,
    right: np.ndarray,
    displacement: np.ndarray,
    energy_radial_derivative: np.ndarray,
    energy_radial_second_derivative: np.ndarray,
    direction: np.ndarray,
    natoms: int,
) -> np.ndarray:
    """Apply a central pair Hessian without materializing 3x3 pair blocks."""

    if _gfn2_native is not None and hasattr(
        _gfn2_native, "radial_pair_hessian_vector"
    ):
        return np.asarray(
            _gfn2_native.radial_pair_hessian_vector(
                np.asarray(left, dtype=np.int64),
                np.asarray(right, dtype=np.int64),
                np.asarray(displacement, dtype=float),
                np.asarray(energy_radial_derivative, dtype=float),
                np.asarray(energy_radial_second_derivative, dtype=float),
                np.asarray(direction, dtype=float),
                int(natoms),
            ),
            dtype=float,
        )
    distance = np.linalg.norm(displacement, axis=1)
    if np.any(distance <= 0.0):
        raise ValueError("radial Hessian contains coincident pair centres")
    relative = direction[left] - direction[right]
    unit = displacement / distance[:, None]
    longitudinal = np.einsum("ij,ij->i", unit, relative)
    product = (
        energy_radial_second_derivative[:, None]
        * longitudinal[:, None]
        * unit
        + (energy_radial_derivative / distance)[:, None]
        * (relative - longitudinal[:, None] * unit)
    )
    return _pair_antisymmetric_gradient(left, right, product, natoms)


@dataclass(frozen=True)
class VariationalGradientTerms:
    """Derivative tensors entering the stationary GFN2 Lagrangian."""

    hamiltonian_derivative: np.ndarray
    overlap_derivative: np.ndarray
    explicit_gradients: tuple[np.ndarray, ...] = ()
    scc_stationarity_residual: float = 0.0


def energy_weighted_density(
    coefficients: np.ndarray,
    occupations: np.ndarray,
    orbital_energies_hartree: np.ndarray,
) -> np.ndarray:
    coefficients = np.asarray(coefficients, dtype=float)
    weights = np.asarray(occupations, dtype=float) * np.asarray(
        orbital_energies_hartree, dtype=float
    )
    return (coefficients * weights) @ coefficients.T


def energy_weighted_density_from_fock(
    fock: np.ndarray,
    density: np.ndarray,
    orthogonalizer: np.ndarray,
) -> np.ndarray:
    """Recover the energy-weighted density without canonical orbitals.

    At an SCF stationary point, ``F D = S W`` and ``S^-1 = X X^T``.  This
    identity lets an SP2 density builder supply the Pulay matrix without
    retaining eigenvectors.
    """

    fock_values = np.asarray(fock, dtype=float)
    density_values = np.asarray(density, dtype=float)
    inverse_sqrt = np.asarray(orthogonalizer, dtype=float)
    if not (
        fock_values.shape == density_values.shape == inverse_sqrt.shape
        and fock_values.ndim == 2
        and fock_values.shape[0] == fock_values.shape[1]
    ):
        raise ValueError("Fock, density, and orthogonalizer shapes must match")
    weighted = inverse_sqrt @ inverse_sqrt.T @ fock_values @ density_values
    return 0.5 * (weighted + weighted.T)


def energy_weighted_density_from_fock_sparse(
    fock: SparseSymmetricMatrix,
    orthogonal_projector: SparseSymmetricMatrix,
    orthogonalizer: SparseSymmetricMatrix,
    domains: LocalizedOrbitalDomains,
    *,
    threshold: float = 1.0e-12,
) -> tuple[SparseSymmetricMatrix, float]:
    """Return the Pulay matrix through sparse orthogonal-basis products."""

    orthogonal_fock, first = orthogonalizer.congruence(
        fock, domains, threshold=threshold
    )
    weighted_orthogonal, second = orthogonal_fock.product(
        orthogonal_projector, domains, threshold=threshold
    )
    weighted_orthogonal = SparseSymmetricMatrix(
        weighted_orthogonal.size,
        weighted_orthogonal.rows,
        weighted_orthogonal.columns,
        2.0 * weighted_orthogonal.values,
    )
    weighted, third = orthogonalizer.congruence(
        weighted_orthogonal, domains, threshold=threshold
    )
    return weighted, first + second + third


def assemble_variational_gradient(
    density: np.ndarray,
    weighted_density: np.ndarray,
    terms: VariationalGradientTerms,
    *,
    stationarity_tolerance: float = 1.0e-7,
) -> np.ndarray:
    """Assemble P:H' - W:S' plus explicit stationary energy derivatives."""

    if abs(float(terms.scc_stationarity_residual)) > stationarity_tolerance:
        raise ValueError("SCC is not stationary enough for an analytic gradient")
    density = np.asarray(density, dtype=float)
    weighted = np.asarray(weighted_density, dtype=float)
    dh = np.asarray(terms.hamiltonian_derivative, dtype=float)
    ds = np.asarray(terms.overlap_derivative, dtype=float)
    if dh.shape != ds.shape or dh.ndim != 4 or dh.shape[2:] != density.shape:
        raise ValueError("derivative tensors must have shape natoms x 3 x nao x nao")
    if weighted.shape != density.shape:
        raise ValueError("density and energy-weighted density shapes differ")
    explicit = np.zeros(dh.shape[:2], dtype=float)
    for contribution in terms.explicit_gradients:
        values = np.asarray(contribution, dtype=float)
        if values.shape != explicit.shape:
            raise ValueError("explicit gradient shape does not match derivative tensors")
        explicit += values
    if _gfn2_native is not None:
        return _gfn2_native.variational_gradient(
            density, weighted, dh, ds, explicit
        )
    gradient = np.einsum("xkij,ij->xk", dh, density, optimize=False)
    gradient -= np.einsum("xkij,ij->xk", ds, weighted, optimize=False)
    gradient += explicit
    return gradient


def assemble_variational_gradient_sparse(
    density: SparseSymmetricMatrix,
    weighted_density: SparseSymmetricMatrix,
    rows: np.ndarray,
    columns: np.ndarray,
    left_atoms: np.ndarray,
    right_atoms: np.ndarray,
    hamiltonian_left_derivative: np.ndarray,
    hamiltonian_right_derivative: np.ndarray,
    overlap_left_derivative: np.ndarray,
    overlap_right_derivative: np.ndarray,
    explicit_gradient: np.ndarray,
    native_threads: int = 1,
) -> np.ndarray:
    """Contract stationary AO-pair derivatives without dense tensors."""

    rows = np.asarray(rows, dtype=np.int32)
    columns = np.asarray(columns, dtype=np.int32)
    left_atoms = np.asarray(left_atoms, dtype=np.int32)
    right_atoms = np.asarray(right_atoms, dtype=np.int32)
    dh_left = np.asarray(hamiltonian_left_derivative, dtype=float)
    dh_right = np.asarray(hamiltonian_right_derivative, dtype=float)
    ds_left = np.asarray(overlap_left_derivative, dtype=float)
    ds_right = np.asarray(overlap_right_derivative, dtype=float)
    if any(
        item.shape != (len(rows), 3)
        for item in (dh_left, dh_right, ds_left, ds_right)
    ):
        raise ValueError("sparse AO derivatives must have shape npairs x 3")
    gradient = np.asarray(explicit_gradient, dtype=float).copy()
    # Bound the derivative working set independently of the total pair count.
    # Each chunk is accumulated immediately into atomic gradients.
    chunk_size = 65_536
    for start in range(0, len(rows), chunk_size):
        stop = min(start + chunk_size, len(rows))
        chunk = slice(start, stop)
        chunk_rows = rows[chunk]
        chunk_columns = columns[chunk]
        p = density.values_on(chunk_rows, chunk_columns)
        w = weighted_density.values_on(chunk_rows, chunk_columns)
        multiplicity = np.where(chunk_rows == chunk_columns, 1.0, 2.0)
        if _gfn2_native is not None and hasattr(
            _gfn2_native, "sparse_gradient_accumulate"
        ):
            _gfn2_native.sparse_gradient_accumulate(
                p,
                w,
                multiplicity,
                left_atoms[chunk],
                right_atoms[chunk],
                dh_left[chunk],
                dh_right[chunk],
                ds_left[chunk],
                ds_right[chunk],
                gradient,
                int(native_threads),
            )
            continue
        left_contribution = multiplicity[:, None] * (
            p[:, None] * dh_left[chunk] - w[:, None] * ds_left[chunk]
        )
        right_contribution = multiplicity[:, None] * (
            p[:, None] * dh_right[chunk] - w[:, None] * ds_right[chunk]
        )
        np.add.at(gradient, left_atoms[chunk], left_contribution)
        np.add.at(gradient, right_atoms[chunk], right_contribution)
    return gradient


def assemble_variational_hessian_vector_sparse(
    density: SparseSymmetricMatrix,
    weighted_density: SparseSymmetricMatrix,
    rows: np.ndarray,
    columns: np.ndarray,
    left_atoms: np.ndarray,
    right_atoms: np.ndarray,
    hamiltonian_left_directional: np.ndarray,
    hamiltonian_right_directional: np.ndarray,
    overlap_left_directional: np.ndarray,
    overlap_right_directional: np.ndarray,
    explicit_product: np.ndarray,
    *,
    native_threads: int = 1,
) -> np.ndarray:
    """Contract ``P:H''[v] - W:S''[v]`` on a pair-packed AO pattern."""

    return assemble_variational_gradient_sparse(
        density,
        weighted_density,
        rows,
        columns,
        left_atoms,
        right_atoms,
        hamiltonian_left_directional,
        hamiltonian_right_directional,
        overlap_left_directional,
        overlap_right_directional,
        explicit_product,
        native_threads=native_threads,
    )


def coordination_number_jacobian(
    atomic_numbers: Sequence[int],
    coordinates_bohr: np.ndarray,
    atom_parameters: Mapping[int, AtomParameters] | Sequence[AtomParameters],
) -> np.ndarray:
    """Return exact dCN_A/dR_B for the double-damped GFN2 CN."""

    numbers = tuple(int(value) for value in atomic_numbers)
    coordinates = np.asarray(coordinates_bohr, dtype=float)
    coordination_numbers(numbers, coordinates, atom_parameters)
    atom_values = _atom_values(numbers, atom_parameters)
    jacobian = np.zeros((len(numbers), len(numbers), 3), dtype=float)
    for center in range(len(numbers)):
        center_radius = atom_values[center].covalent_radius_bohr
        for neighbor in range(len(numbers)):
            if center == neighbor:
                continue
            displacement = coordinates[center] - coordinates[neighbor]
            distance = float(np.linalg.norm(displacement))
            neighbor_radius = atom_values[neighbor].covalent_radius_bohr
            scale1 = 4.0 * (center_radius + neighbor_radius) / 3.0
            scale2 = scale1 + 2.0
            first = 1.0 / (1.0 + np.exp(-10.0 * (scale1 / distance - 1.0)))
            second = 1.0 / (1.0 + np.exp(-20.0 * (scale2 / distance - 1.0)))
            radial = (
                -10.0 * scale1 * first * (1.0 - first) * second / distance**2
                -20.0 * scale2 * second * (1.0 - second) * first / distance**2
            )
            derivative = radial * displacement / distance
            jacobian[center, center] += derivative
            jacobian[center, neighbor] -= derivative
    return jacobian


def frozen_charge_scc_gradient(
    coordinates_bohr: np.ndarray,
    charges: np.ndarray,
    partition: ShellPartition,
    *,
    interaction: np.ndarray | None = None,
    alphaj: float = 2.0,
) -> np.ndarray:
    """Exact explicit SCC kernel derivative at stationary shell charges."""

    coordinates = np.asarray(coordinates_bohr, dtype=float)
    charges = np.asarray(charges, dtype=float)
    if interaction is not None and hasattr(interaction, "evaluate"):
        return interaction.evaluate(charges).gradient_hartree_per_bohr
    interaction_matrix = (
        shell_interaction_matrix(coordinates, partition, alphaj=alphaj)
        if interaction is None
        else np.asarray(interaction, dtype=float)
    )
    monopole_energy(charges, interaction_matrix, partition)
    hardness = np.asarray(partition.hardness_hartree, dtype=float)
    left, right = np.tril_indices(partition.nshells, k=-1)
    shell_atom = np.asarray(partition.shell_atom, dtype=np.intp)
    distinct = shell_atom[left] != shell_atom[right]
    left, right = left[distinct], right[distinct]
    left_atom, right_atom = shell_atom[left], shell_atom[right]
    displacement = coordinates[left_atom] - coordinates[right_atom]
    distance = np.linalg.norm(displacement, axis=1)
    average = 0.5 * (hardness[left] + hardness[right])
    if not np.isfinite(alphaj) or alphaj <= 0.0:
        raise ValueError("Klopman-Ohno exponent must be positive and finite")
    softened = distance**alphaj + average**-alphaj
    radial = -distance ** (alphaj - 1.0) * softened ** (-1.0 / alphaj - 1.0)
    derivative = (
        charges[left] * charges[right] * radial / distance
    )[:, None] * displacement
    return _pair_antisymmetric_gradient(
        left_atom, right_atom, derivative, len(coordinates)
    )


def frozen_charge_scc_hessian_vector_product(
    coordinates_bohr: np.ndarray,
    direction_bohr: np.ndarray,
    charges: np.ndarray,
    partition: ShellPartition,
    *,
    interaction: np.ndarray | None = None,
    charge_direction: np.ndarray | None = None,
    alphaj: float = 2.0,
) -> np.ndarray:
    """Apply the SCC coordinate Hessian, optionally including ``dq[v]``."""

    coordinates = np.asarray(coordinates_bohr, dtype=float)
    direction = np.asarray(direction_bohr, dtype=float)
    charges = np.asarray(charges, dtype=float)
    if direction.shape != coordinates.shape:
        raise ValueError("SCC Hessian direction must have shape natoms x 3")
    qdot = (
        np.zeros_like(charges)
        if charge_direction is None
        else np.asarray(charge_direction, dtype=float)
    )
    if qdot.shape != charges.shape:
        raise ValueError("SCC charge direction has an inconsistent shell count")
    if (
        interaction is not None
        and hasattr(interaction, "hessian_vector_product")
        and charge_direction is None
    ):
        return interaction.hessian_vector_product(charges, direction)
    if interaction is not None and hasattr(interaction, "hessian_vector_product"):
        fixed = interaction.hessian_vector_product(charges, direction)
        mixed = (
            interaction.evaluate(charges + qdot).gradient_hartree_per_bohr
            - interaction.evaluate(charges).gradient_hartree_per_bohr
            - interaction.evaluate(qdot).gradient_hartree_per_bohr
        )
        return fixed + mixed
    hardness = np.asarray(partition.hardness_hartree, dtype=float)
    if not np.isfinite(alphaj) or alphaj <= 0.0:
        raise ValueError("Klopman-Ohno exponent must be positive and finite")
    left, right = np.tril_indices(partition.nshells, k=-1)
    shell_atom = np.asarray(partition.shell_atom, dtype=np.intp)
    distinct = shell_atom[left] != shell_atom[right]
    left, right = left[distinct], right[distinct]
    left_atom, right_atom = shell_atom[left], shell_atom[right]
    displacement = coordinates[left_atom] - coordinates[right_atom]
    distance = np.linalg.norm(displacement, axis=1)
    average = 0.5 * (hardness[left] + hardness[right])
    softened = distance**alphaj + average**-alphaj
    charge_product = charges[left] * charges[right]
    radial_kernel = -distance ** (alphaj - 1.0) * softened ** (-1.0 / alphaj - 1.0)
    second_kernel = (
        -(alphaj - 1.0) * distance ** (alphaj - 2.0)
        * softened ** (-1.0 / alphaj - 1.0)
        + (alphaj + 1.0) * distance ** (2.0 * alphaj - 2.0)
        * softened ** (-1.0 / alphaj - 2.0)
    )
    radial = charge_product * radial_kernel
    radial_second = charge_product * second_kernel
    result = _radial_pair_hessian_vector_product(
        left_atom,
        right_atom,
        displacement,
        radial,
        radial_second,
        direction,
        len(coordinates),
    )
    charge_product_direction = qdot[left] * charges[right] + charges[left] * qdot[right]
    mixed_radial = charge_product_direction * radial_kernel
    result += _pair_antisymmetric_gradient(
        left_atom,
        right_atom,
        mixed_radial[:, None] * displacement / distance[:, None],
        len(coordinates),
    )
    return result


def shell_charge_derivatives(
    density: np.ndarray,
    overlap_derivative: np.ndarray,
    partition: ShellPartition,
) -> np.ndarray:
    """Derivative of Mulliken shell charges at fixed AO density."""

    density = np.asarray(density, dtype=float)
    ds = np.asarray(overlap_derivative, dtype=float)
    result = np.zeros((partition.nshells, ds.shape[0], 3))
    for orbital, shell in enumerate(partition.orbital_shell):
        result[shell] -= np.einsum(
            "j,xkj->xk", density[orbital], ds[:, :, :, orbital], optimize=False
        )
    return result


def atomic_multipole_derivatives(
    density: np.ndarray,
    overlap: np.ndarray,
    overlap_derivative: np.ndarray,
    dipole: np.ndarray,
    dipole_derivative: np.ndarray,
    second: np.ndarray,
    second_derivative: np.ndarray,
    coordinates_bohr: np.ndarray,
    orbital_atom: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analytic fixed-density derivatives of CAMM q, dipoles and quadrupoles."""

    density = np.asarray(density, dtype=float)
    coordinates = np.asarray(coordinates_bohr, dtype=float)
    assignment = tuple(int(value) for value in orbital_atom)
    natoms = len(coordinates)
    atom_index = np.asarray(assignment, dtype=np.int64)
    if _gfn2_native is not None:
        return _gfn2_native.atomic_multipole_derivatives(
            density,
            np.asarray(overlap, dtype=float),
            np.asarray(overlap_derivative, dtype=float),
            np.asarray(dipole, dtype=float),
            np.asarray(dipole_derivative, dtype=float),
            np.asarray(second_derivative, dtype=float),
            coordinates,
            atom_index,
        )
    centers = coordinates[atom_index]
    ds = np.transpose(np.asarray(overlap_derivative), (3, 0, 1, 2))
    dd = np.transpose(np.asarray(dipole_derivative), (4, 0, 1, 2, 3))
    dsecond = np.transpose(np.asarray(second_derivative), (5, 0, 1, 2, 3, 4))
    base = np.einsum("jk,jtxk->jtx", density, ds, optimize=False)
    dipole_base = np.einsum("jk,jctxk->jctx", density, dd, optimize=False)
    second_base = np.einsum(
        "jk,jcotxk->jcotx", density, dsecond, optimize=False
    )
    overlap_population = np.einsum("jk,jk->j", density, overlap.T, optimize=False)
    dipole_population = np.einsum(
        "jk,jck->jc", density, np.transpose(dipole, (2, 0, 1)), optimize=False
    )
    shift = np.zeros((len(assignment), natoms, 3, 3), dtype=float)
    orbital_indices = np.arange(len(assignment))
    for component in range(3):
        shift[orbital_indices, atom_index, component, component] = 1.0

    orbital_dq = -base
    orbital_dmu = (
        np.transpose(shift, (0, 3, 1, 2))
        * overlap_population[:, None, None, None]
        + centers[:, :, None, None] * base[:, None, :, :]
        - dipole_base
    )
    shift_component = np.transpose(shift, (0, 3, 1, 2))
    orbital_draw = (
        shift_component[:, None, :, :, :]
        * dipole_population[:, :, None, None, None]
        + centers[:, None, :, None, None]
        * dipole_base[:, :, None, :, :]
        + shift_component[:, :, None, :, :]
        * dipole_population[:, None, :, None, None]
        + centers[:, :, None, None, None]
        * dipole_base[:, None, :, :, :]
        - (
            shift_component[:, :, None, :, :]
            * centers[:, None, :, None, None]
            + centers[:, :, None, None, None]
            * shift_component[:, None, :, :, :]
        )
        * overlap_population[:, None, None, None, None]
        - centers[:, :, None, None, None]
        * centers[:, None, :, None, None]
        * base[:, None, None, :, :]
        - second_base
    )
    dq = np.zeros((natoms, natoms, 3))
    dmu = np.zeros((natoms, 3, natoms, 3))
    draw = np.zeros((natoms, 3, 3, natoms, 3))
    np.add.at(dq, atom_index, orbital_dq)
    np.add.at(dmu, atom_index, orbital_dmu)
    np.add.at(draw, atom_index, orbital_draw)
    trace = np.trace(draw, axis1=1, axis2=2)
    dquadrupole = 1.5 * draw
    for component in range(3):
        dquadrupole[:, component, component] -= 0.5 * trace
    return dq, dmu, dquadrupole


def anisotropic_explicit_gradient(
    multipoles: AtomicMultipoles,
    coordinates_bohr: np.ndarray,
    parameters: AnisotropicParameters,
    radius_jacobian: np.ndarray | None,
    *,
    geometry=None,
    coordination_radii_bohr: np.ndarray | None = None,
    radius_cn_derivative: np.ndarray | None = None,
) -> np.ndarray:
    """Explicit geometry derivative of AES/AXC at frozen atomic moments."""

    coordinates = np.asarray(coordinates_bohr, dtype=float)
    charges = np.asarray(multipoles.charges, dtype=float)
    dipoles = np.asarray(multipoles.dipoles_bohr, dtype=float)
    quadrupoles = np.asarray(multipoles.quadrupoles_bohr2, dtype=float)
    radii = np.asarray(parameters.damping_radius_bohr, dtype=float)
    gradient = np.zeros_like(coordinates)
    radius_derivative = np.zeros(len(coordinates))

    def damped_derivatives(distance, radius, power: int, exponent: float):
        ratio = 6.0 * (radius / distance) ** exponent
        value = distance ** (-power) / (1.0 + ratio)
        radial = value * (-power + exponent * ratio / (1.0 + ratio)) / distance
        damping_radius = -value * exponent * ratio / ((1.0 + ratio) * radius)
        return value, radial, damping_radius

    if geometry is None:
        left, right = np.tril_indices(len(coordinates), k=-1)
        displacement = coordinates[right] - coordinates[left]
    else:
        left, right = geometry.left, geometry.right
        displacement = geometry.displacement
    distance = np.linalg.norm(displacement, axis=1)
    radius = 0.5 * (radii[left] + radii[right])
    f3, df3, da3 = damped_derivatives(distance, radius, 3, parameters.a3)
    f5, df5, da5 = damped_derivatives(distance, radius, 5, parameters.a5)
    left_projection = np.sum(dipoles[left] * displacement, axis=1)
    right_projection = np.sum(dipoles[right] * displacement, axis=1)
    left_quadrupole = np.einsum(
        "pi,pij,pj->p", displacement, quadrupoles[left], displacement
    )
    right_quadrupole = np.einsum(
        "pi,pij,pj->p", displacement, quadrupoles[right], displacement
    )
    distance2 = distance * distance
    dipole_dot = np.sum(dipoles[left] * dipoles[right], axis=1)
    angular3 = -charges[left] * right_projection + charges[right] * left_projection
    angular5 = (
        charges[left] * right_quadrupole
        + charges[right] * left_quadrupole
        - 3.0 * left_projection * right_projection
        + dipole_dot * distance2
    )
    angular3_gradient = (
        -charges[left, None] * dipoles[right]
        + charges[right, None] * dipoles[left]
    )
    angular5_gradient = (
        2.0 * charges[left, None]
        * np.einsum("pij,pj->pi", quadrupoles[right], displacement)
        + 2.0 * charges[right, None]
        * np.einsum("pij,pj->pi", quadrupoles[left], displacement)
        - 3.0
        * (
            dipoles[left] * right_projection[:, None]
            + dipoles[right] * left_projection[:, None]
        )
        + 2.0 * dipole_dot[:, None] * displacement
    )
    displacement_gradient = (
        (df3 * angular3 / distance)[:, None] * displacement
        + f3[:, None] * angular3_gradient
        + (df5 * angular5 / distance)[:, None] * displacement
        + f5[:, None] * angular5_gradient
    )
    gradient += _pair_antisymmetric_gradient(
        right, left, displacement_gradient, len(coordinates)
    )
    pair_radius = 0.5 * (da3 * angular3 + da5 * angular5)
    radius_derivative += _pair_symmetric_scalar(
        left, right, pair_radius, len(coordinates)
    )
    if radius_jacobian is not None:
        gradient += np.einsum("a,axk->xk", radius_derivative, radius_jacobian)
    else:
        if coordination_radii_bohr is None or radius_cn_derivative is None:
            raise ValueError("sparse anisotropic radius response requires CN data")
        gradient += coordination_response_gradient_sparse(
            radius_derivative * np.asarray(radius_cn_derivative, dtype=float),
            coordinates,
            np.asarray(coordination_radii_bohr, dtype=float),
        )
    return gradient


def anisotropic_explicit_hessian_vector_product_sparse(
    multipoles: AtomicMultipoles,
    multipole_direction: AtomicMultipoles,
    coordinates_bohr: np.ndarray,
    direction_bohr: np.ndarray,
    parameters: AnisotropicParameters,
    coordination_radii_bohr: np.ndarray,
    radius_cn_derivative: np.ndarray,
    radius_cn_second_derivative: np.ndarray,
    coordination_direction: np.ndarray,
    *,
    geometry=None,
) -> np.ndarray:
    """Differentiate the pair-packed anisotropic explicit gradient analytically."""

    coordinates = np.asarray(coordinates_bohr, dtype=float)
    direction = np.asarray(direction_bohr, dtype=float)
    if direction.shape != coordinates.shape:
        raise ValueError("anisotropic Hessian direction must have shape natoms x 3")
    q = np.asarray(multipoles.charges, dtype=float)
    mu = np.asarray(multipoles.dipoles_bohr, dtype=float)
    theta = np.asarray(multipoles.quadrupoles_bohr2, dtype=float)
    qdot = np.asarray(multipole_direction.charges, dtype=float)
    mudot = np.asarray(multipole_direction.dipoles_bohr, dtype=float)
    thetadot = np.asarray(multipole_direction.quadrupoles_bohr2, dtype=float)
    radii = np.asarray(parameters.damping_radius_bohr, dtype=float)
    if geometry is None:
        left, right = np.tril_indices(len(coordinates), k=-1)
        displacement = coordinates[right] - coordinates[left]
    else:
        left, right = geometry.left, geometry.right
        displacement = geometry.displacement
    displacement_dot = direction[right] - direction[left]
    distance = np.linalg.norm(displacement, axis=1)
    radial_dot = np.einsum(
        "pi,pi->p", displacement, displacement_dot
    ) / distance
    radius = 0.5 * (radii[left] + radii[right])
    radius_atom_direction = (
        np.asarray(radius_cn_derivative) * np.asarray(coordination_direction)
    )
    radius_dot = 0.5 * (
        radius_atom_direction[left] + radius_atom_direction[right]
    )

    def damped_directional(power: int, exponent: float):
        ratio = 6.0 * (radius / distance) ** exponent
        fraction = ratio / (1.0 + ratio)
        value = distance ** (-power) / (1.0 + ratio)
        logarithmic_r = (-power + exponent * fraction) / distance
        logarithmic_radius = -exponent * fraction / radius
        ratio_dot = ratio * exponent * (
            radius_dot / radius - radial_dot / distance
        )
        fraction_dot = ratio_dot / (1.0 + ratio) ** 2
        value_dot = value * (
            logarithmic_r * radial_dot
            + logarithmic_radius * radius_dot
        )
        logarithmic_r_dot = (
            exponent * fraction_dot / distance
            - (-power + exponent * fraction) * radial_dot / distance**2
        )
        logarithmic_radius_dot = -exponent * (
            fraction_dot / radius - fraction * radius_dot / radius**2
        )
        radial = value * logarithmic_r
        radius_derivative = value * logarithmic_radius
        radial_direction = value_dot * logarithmic_r + value * logarithmic_r_dot
        radius_direction = (
            value_dot * logarithmic_radius
            + value * logarithmic_radius_dot
        )
        return value, value_dot, radial, radial_direction, radius_derivative, radius_direction

    f3, f3dot, df3, df3dot, da3, da3dot = damped_directional(
        3, parameters.a3
    )
    f5, f5dot, df5, df5dot, da5, da5dot = damped_directional(
        5, parameters.a5
    )
    lp = np.einsum("pi,pi->p", mu[left], displacement)
    rp = np.einsum("pi,pi->p", mu[right], displacement)
    lpdot = np.einsum("pi,pi->p", mudot[left], displacement) + np.einsum(
        "pi,pi->p", mu[left], displacement_dot
    )
    rpdot = np.einsum("pi,pi->p", mudot[right], displacement) + np.einsum(
        "pi,pi->p", mu[right], displacement_dot
    )
    ltheta_d = np.einsum("pij,pj->pi", theta[left], displacement)
    rtheta_d = np.einsum("pij,pj->pi", theta[right], displacement)
    ltheta = np.einsum("pi,pi->p", displacement, ltheta_d)
    rtheta = np.einsum("pi,pi->p", displacement, rtheta_d)
    ltheta_dot = (
        2.0 * np.einsum("pi,pi->p", displacement_dot, ltheta_d)
        + np.einsum("pi,pij,pj->p", displacement, thetadot[left], displacement)
    )
    rtheta_dot = (
        2.0 * np.einsum("pi,pi->p", displacement_dot, rtheta_d)
        + np.einsum("pi,pij,pj->p", displacement, thetadot[right], displacement)
    )
    mu_dot = np.einsum("pi,pi->p", mu[left], mu[right])
    mu_dot_direction = np.einsum("pi,pi->p", mudot[left], mu[right]) + np.einsum(
        "pi,pi->p", mu[left], mudot[right]
    )
    distance2 = distance**2
    distance2_dot = 2.0 * np.einsum(
        "pi,pi->p", displacement, displacement_dot
    )
    angular3 = -q[left] * rp + q[right] * lp
    angular3_dot = (
        -qdot[left] * rp
        - q[left] * rpdot
        + qdot[right] * lp
        + q[right] * lpdot
    )
    angular5 = (
        q[left] * rtheta
        + q[right] * ltheta
        - 3.0 * lp * rp
        + mu_dot * distance2
    )
    angular5_dot = (
        qdot[left] * rtheta
        + q[left] * rtheta_dot
        + qdot[right] * ltheta
        + q[right] * ltheta_dot
        - 3.0 * (lpdot * rp + lp * rpdot)
        + mu_dot_direction * distance2
        + mu_dot * distance2_dot
    )
    angular3_gradient = -q[left, None] * mu[right] + q[right, None] * mu[left]
    angular3_gradient_dot = (
        -qdot[left, None] * mu[right]
        - q[left, None] * mudot[right]
        + qdot[right, None] * mu[left]
        + q[right, None] * mudot[left]
    )
    ltheta_d_dot = (
        np.einsum("pij,pj->pi", thetadot[left], displacement)
        + np.einsum("pij,pj->pi", theta[left], displacement_dot)
    )
    rtheta_d_dot = (
        np.einsum("pij,pj->pi", thetadot[right], displacement)
        + np.einsum("pij,pj->pi", theta[right], displacement_dot)
    )
    angular5_gradient = (
        2.0 * q[left, None] * rtheta_d
        + 2.0 * q[right, None] * ltheta_d
        - 3.0 * (mu[left] * rp[:, None] + mu[right] * lp[:, None])
        + 2.0 * mu_dot[:, None] * displacement
    )
    angular5_gradient_dot = (
        2.0 * qdot[left, None] * rtheta_d
        + 2.0 * q[left, None] * rtheta_d_dot
        + 2.0 * qdot[right, None] * ltheta_d
        + 2.0 * q[right, None] * ltheta_d_dot
        - 3.0
        * (
            mudot[left] * rp[:, None]
            + mu[left] * rpdot[:, None]
            + mudot[right] * lp[:, None]
            + mu[right] * lpdot[:, None]
        )
        + 2.0 * mu_dot_direction[:, None] * displacement
        + 2.0 * mu_dot[:, None] * displacement_dot
    )

    def directional_gradient(value, value_dot, radial, radial_direction, angular, angular_dot, angular_gradient, angular_gradient_dot):
        coefficient = radial * angular / distance
        coefficient_dot = (
            radial_direction * angular / distance
            + radial * angular_dot / distance
            - radial * angular * radial_dot / distance**2
        )
        return (
            coefficient_dot[:, None] * displacement
            + coefficient[:, None] * displacement_dot
            + value_dot[:, None] * angular_gradient
            + value[:, None] * angular_gradient_dot
        )

    pair_product = directional_gradient(
        f3, f3dot, df3, df3dot, angular3, angular3_dot,
        angular3_gradient, angular3_gradient_dot,
    ) + directional_gradient(
        f5, f5dot, df5, df5dot, angular5, angular5_dot,
        angular5_gradient, angular5_gradient_dot,
    )
    result = _pair_antisymmetric_gradient(
        right, left, pair_product, len(coordinates)
    )
    pair_radius = 0.5 * (da3 * angular3 + da5 * angular5)
    pair_radius_dot = 0.5 * (
        da3dot * angular3
        + da3 * angular3_dot
        + da5dot * angular5
        + da5 * angular5_dot
    )
    radius_coefficient = _pair_symmetric_scalar(
        left, right, pair_radius, len(coordinates)
    )
    radius_coefficient_dot = _pair_symmetric_scalar(
        left, right, pair_radius_dot, len(coordinates)
    )
    cn_coefficient = radius_coefficient * np.asarray(radius_cn_derivative)
    cn_coefficient_dot = (
        radius_coefficient_dot * np.asarray(radius_cn_derivative)
        + radius_coefficient
        * np.asarray(radius_cn_second_derivative)
        * np.asarray(coordination_direction)
    )
    result += coordination_response_hessian_vector_product_sparse(
        cn_coefficient,
        coordinates,
        direction,
        np.asarray(coordination_radii_bohr),
    )
    result += coordination_response_gradient_sparse(
        cn_coefficient_dot,
        coordinates,
        np.asarray(coordination_radii_bohr),
    )
    return result


def repulsion_gradient(
    atomic_numbers: Sequence[int],
    coordinates_bohr: np.ndarray,
    atom_parameters: Mapping[int, AtomParameters] | Sequence[AtomParameters],
    *, kexp: float = 1.5, klight: float = 1.0,
) -> np.ndarray:
    numbers = tuple(int(value) for value in atomic_numbers)
    coordinates = np.asarray(coordinates_bohr, dtype=float)
    if not np.isfinite(kexp) or kexp <= 0.0:
        raise ValueError("repulsion exponent must be positive and finite")
    atom_values = _atom_values(numbers, atom_parameters)
    gradient = np.zeros_like(coordinates)
    for left, left_number in enumerate(numbers):
        left_parameters = atom_values[left]
        for right in range(left):
            right_parameters = atom_values[right]
            displacement = coordinates[left] - coordinates[right]
            distance = float(np.linalg.norm(displacement))
            exponent = klight if left_number <= 2 and numbers[right] <= 2 else kexp
            alpha = np.sqrt(left_parameters.alpha * right_parameters.alpha)
            energy = (
                left_parameters.effective_nuclear_charge
                * right_parameters.effective_nuclear_charge
                / distance
                * np.exp(-alpha * distance**exponent)
            )
            radial = energy * (-1.0 / distance - alpha * exponent * distance ** (exponent - 1.0))
            contribution = radial * displacement / distance
            gradient[left] += contribution
            gradient[right] -= contribution
    return gradient


def repulsion_gradient_sparse(
    atomic_numbers: Sequence[int], coordinates_bohr: np.ndarray,
    atom_parameters: Mapping[int, AtomParameters] | Sequence[AtomParameters],
    *, cutoff_bohr: float = 24.0, kexp: float = 1.5, klight: float = 1.0,
) -> np.ndarray:
    from .kernels import sparse_pair_geometry

    numbers = np.asarray(tuple(atomic_numbers), dtype=np.intp)
    if not np.isfinite(kexp) or kexp <= 0.0:
        raise ValueError("repulsion exponent must be positive and finite")
    coordinates = np.asarray(coordinates_bohr, dtype=float)
    pairs = sparse_pair_geometry(coordinates, cutoff_bohr=cutoff_bohr)
    atom_values = _atom_values(numbers, atom_parameters)
    lp = [atom_values[index] for index in pairs.left]
    rp = [atom_values[index] for index in pairs.right]
    charge = np.asarray([item.effective_nuclear_charge for item in lp]) * np.asarray([item.effective_nuclear_charge for item in rp])
    alpha = np.sqrt(np.asarray([item.alpha for item in lp]) * np.asarray([item.alpha for item in rp]))
    exponent = np.where((numbers[pairs.left] <= 2) & (numbers[pairs.right] <= 2), klight, kexp)
    energy = charge / pairs.distance * np.exp(-alpha * pairs.distance**exponent)
    radial = energy * (-1.0 / pairs.distance - alpha * exponent * pairs.distance ** (exponent - 1.0))
    contribution = radial[:, None] * pairs.displacement / pairs.distance[:, None]
    return _pair_antisymmetric_gradient(
        pairs.left, pairs.right, contribution, len(coordinates)
    )


def repulsion_hessian_vector_product_sparse(
    atomic_numbers: Sequence[int],
    coordinates_bohr: np.ndarray,
    direction_bohr: np.ndarray,
    atom_parameters: Mapping[int, AtomParameters] | Sequence[AtomParameters],
    *,
    cutoff_bohr: float = 24.0,
    kexp: float = 1.5,
    klight: float = 1.0,
) -> np.ndarray:
    """Apply the analytic pair-packed GFN2 repulsion Hessian."""

    from .kernels import sparse_pair_geometry

    numbers = np.asarray(tuple(atomic_numbers), dtype=np.intp)
    if not np.isfinite(kexp) or kexp <= 0.0:
        raise ValueError("repulsion exponent must be positive and finite")
    coordinates = np.asarray(coordinates_bohr, dtype=float)
    direction = np.asarray(direction_bohr, dtype=float)
    if direction.shape != coordinates.shape:
        raise ValueError("repulsion Hessian direction must have shape natoms x 3")
    pairs = sparse_pair_geometry(coordinates, cutoff_bohr=cutoff_bohr)
    atom_values = _atom_values(numbers, atom_parameters)
    left_parameters = [atom_values[index] for index in pairs.left]
    right_parameters = [atom_values[index] for index in pairs.right]
    charge = np.asarray(
        [item.effective_nuclear_charge for item in left_parameters]
    ) * np.asarray([item.effective_nuclear_charge for item in right_parameters])
    alpha = np.sqrt(
        np.asarray([item.alpha for item in left_parameters])
        * np.asarray([item.alpha for item in right_parameters])
    )
    exponent = np.where(
        (numbers[pairs.left] <= 2) & (numbers[pairs.right] <= 2), klight, kexp
    )
    distance = pairs.distance
    energy = charge / distance * np.exp(-alpha * distance**exponent)
    logarithmic = -1.0 / distance - alpha * exponent * distance ** (exponent - 1.0)
    radial = energy * logarithmic
    radial_second = energy * (
        logarithmic**2
        + 1.0 / distance**2
        - alpha * exponent * (exponent - 1.0) * distance ** (exponent - 2.0)
    )
    return _radial_pair_hessian_vector_product(
        pairs.left,
        pairs.right,
        pairs.displacement,
        radial,
        radial_second,
        direction,
        len(coordinates),
    )


def frozen_d4_two_body_gradient(
    coordinates_bohr: np.ndarray,
    c6: np.ndarray,
    c8: np.ndarray,
    *,
    s6: float = 1.0,
    s8: float = 2.7,
    s10: float = 0.0,
    a1: float = 0.52,
    a2: float = 5.0,
) -> np.ndarray:
    """Exact radial D4 derivative for frozen C6/C8 and coordination.

    This kernel is one term of the complete gradient.  The caller must add the
    coordination-number response and charge-response cancellation supplied by
    the variational SCC Lagrangian.
    """

    coordinates = np.asarray(coordinates_bohr, dtype=float)
    c6 = np.asarray(c6, dtype=float)
    c8 = np.asarray(c8, dtype=float)
    left, right = np.tril_indices(len(coordinates), k=-1)
    displacement = coordinates[left] - coordinates[right]
    distance = np.linalg.norm(displacement, axis=1)
    critical = np.sqrt(c8[left, right] / c6[left, right])
    cutoff = a1 * critical + a2
    derivative = (
        s6 * c6[left, right] * 6.0 * distance**5
        / (distance**6 + cutoff**6) ** 2
        + s8 * c8[left, right] * 8.0 * distance**7
        / (distance**8 + cutoff**8) ** 2
        + s10 * (49.0 / 40.0) * c8[left, right]**2 / c6[left, right]
        * 10.0 * distance**9 / (distance**10 + cutoff**10)**2
    )
    contribution = derivative[:, None] * displacement / distance[:, None]
    return _pair_antisymmetric_gradient(
        left, right, contribution, len(coordinates)
    )


def frozen_d4_two_body_gradient_sparse(
    coordinates_bohr: np.ndarray,
    d4: SparseD4Result,
    *,
    s6: float = 1.0,
    s8: float = 2.7,
    s10: float = 0.0,
    a1: float = 0.52,
    a2: float = 5.0,
) -> np.ndarray:
    """Frozen-C6/C8 radial D4 gradient on pair-packed neighbors."""

    coordinates = np.asarray(coordinates_bohr, dtype=float)
    left, right = d4.rows, d4.columns
    displacement = coordinates[left] - coordinates[right]
    distance = np.linalg.norm(displacement, axis=1)
    critical = np.sqrt(d4.c8 / d4.c6)
    cutoff = a1 * critical + a2
    derivative = (
        s6 * d4.c6 * 6.0 * distance**5 / (distance**6 + cutoff**6) ** 2
        + s8 * d4.c8 * 8.0 * distance**7 / (distance**8 + cutoff**8) ** 2
        + s10 * (49.0 / 40.0) * d4.c8**2 / d4.c6
        * 10.0 * distance**9 / (distance**10 + cutoff**10)**2
    )
    contribution = derivative[:, None] * displacement / distance[:, None]
    return _pair_antisymmetric_gradient(
        left, right, contribution, len(coordinates)
    )


def frozen_d4_two_body_hessian_vector_product(
    coordinates_bohr: np.ndarray,
    direction_bohr: np.ndarray,
    c6: np.ndarray,
    c8: np.ndarray,
    *,
    s6: float = 1.0,
    s8: float = 2.7,
    s10: float = 0.0,
    a1: float = 0.52,
    a2: float = 5.0,
) -> np.ndarray:
    """Apply the dense-reference fixed-C6/C8 radial D4 Hessian."""

    coordinates = np.asarray(coordinates_bohr, dtype=float)
    direction = np.asarray(direction_bohr, dtype=float)
    c6 = np.asarray(c6, dtype=float)
    c8 = np.asarray(c8, dtype=float)
    if direction.shape != coordinates.shape:
        raise ValueError("D4 Hessian direction must have shape natoms x 3")
    left, right = np.tril_indices(len(coordinates), k=-1)
    displacement = coordinates[left] - coordinates[right]
    distance = np.linalg.norm(displacement, axis=1)
    pair_c6 = c6[left, right]
    pair_c8 = c8[left, right]
    cutoff = a1 * np.sqrt(pair_c8 / pair_c6) + a2

    def radial_term(order: int, scale: float, coefficient: np.ndarray):
        denominator = distance**order + cutoff**order
        first = scale * coefficient * order * distance ** (order - 1) / denominator**2
        second = scale * coefficient * order * (
            (order - 1) * distance ** (order - 2) / denominator**2
            - 2.0 * order * distance ** (2 * order - 2) / denominator**3
        )
        return first, second

    first6, second6 = radial_term(6, s6, pair_c6)
    first8, second8 = radial_term(8, s8, pair_c8)
    first10, second10 = radial_term(10, s10, (49.0 / 40.0) * pair_c8**2 / pair_c6)
    return _radial_pair_hessian_vector_product(
        left,
        right,
        displacement,
        first6 + first8 + first10,
        second6 + second8 + second10,
        direction,
        len(coordinates),
    )


def frozen_d4_two_body_hessian_vector_product_sparse(
    coordinates_bohr: np.ndarray,
    direction_bohr: np.ndarray,
    d4: SparseD4Result,
    *,
    s6: float = 1.0,
    s8: float = 2.7,
    s10: float = 0.0,
    a1: float = 0.52,
    a2: float = 5.0,
) -> np.ndarray:
    """Apply the fixed-C6/C8 radial D4 Hessian on stored neighbor pairs."""

    coordinates = np.asarray(coordinates_bohr, dtype=float)
    direction = np.asarray(direction_bohr, dtype=float)
    if direction.shape != coordinates.shape:
        raise ValueError("D4 Hessian direction must have shape natoms x 3")
    left, right = d4.rows, d4.columns
    displacement = coordinates[left] - coordinates[right]
    distance = np.linalg.norm(displacement, axis=1)
    cutoff = a1 * np.sqrt(d4.c8 / d4.c6) + a2

    def radial_term(order: int, scale: float, coefficient: np.ndarray):
        denominator = distance**order + cutoff**order
        first = scale * coefficient * order * distance ** (order - 1) / denominator**2
        second = scale * coefficient * order * (
            (order - 1) * distance ** (order - 2) / denominator**2
            - 2.0 * order * distance ** (2 * order - 2) / denominator**3
        )
        return first, second

    first6, second6 = radial_term(6, s6, d4.c6)
    first8, second8 = radial_term(8, s8, d4.c8)
    first10, second10 = radial_term(10, s10, (49.0 / 40.0) * d4.c8**2 / d4.c6)
    return _radial_pair_hessian_vector_product(
        left,
        right,
        displacement,
        first6 + first8 + first10,
        second6 + second8 + second10,
        direction,
        len(coordinates),
    )


def d4_two_body_hessian_vector_product_sparse(
    coordinates_bohr: np.ndarray,
    direction_bohr: np.ndarray,
    d4: SparseD4Result,
    coefficient_direction: D4CoefficientDirection,
    coordination_direction: np.ndarray,
    coordination_radii_bohr: np.ndarray,
    electronegativities: np.ndarray,
    *,
    s6: float = 1.0,
    s8: float = 2.7,
    s10: float = 0.0,
    a1: float = 0.52,
    a2: float = 5.0,
) -> np.ndarray:
    """Apply the complete pair-packed two-body D4 explicit Hessian."""

    coordinates = np.asarray(coordinates_bohr, dtype=float)
    direction = np.asarray(direction_bohr, dtype=float)
    if direction.shape != coordinates.shape:
        raise ValueError("D4 Hessian direction must have shape natoms x 3")
    if not (
        np.array_equal(d4.rows, coefficient_direction.rows)
        and np.array_equal(d4.columns, coefficient_direction.columns)
    ):
        raise ValueError("D4 coefficient direction uses a different pair pattern")
    result = frozen_d4_two_body_hessian_vector_product_sparse(
        coordinates,
        direction,
        d4,
        s6=s6,
        s8=s8,
        s10=s10,
        a1=a1,
        a2=a2,
    )
    left, right = d4.rows, d4.columns
    displacement = coordinates[left] - coordinates[right]
    distance = np.linalg.norm(displacement, axis=1)
    cutoff = a1 * np.sqrt(d4.c8 / d4.c6) + a2
    coefficient_radial = (
        s6
        * coefficient_direction.c6_derivative
        * 6.0
        * distance**5
        / (distance**6 + cutoff**6) ** 2
        + s8
        * coefficient_direction.c8_derivative
        * 8.0
        * distance**7
        / (distance**8 + cutoff**8) ** 2
        + s10 * (49.0 / 40.0) * (d4.c8 / d4.c6)**2
        * coefficient_direction.c6_derivative * 10.0 * distance**9
        / (distance**10 + cutoff**10)**2
    )
    result += _pair_antisymmetric_gradient(
        left,
        right,
        coefficient_radial[:, None] * displacement / distance[:, None],
        len(coordinates),
    )
    result += coordination_response_hessian_vector_product_sparse(
        d4.coordination_derivative_hartree,
        coordinates,
        direction,
        np.asarray(coordination_radii_bohr),
        electronegativities=np.asarray(electronegativities),
    )
    # The CN potential also changes when the pair damping factor changes
    # radially.  This is the geometry derivative of dE/dCN at fixed C6(CN,q).
    relative_direction = direction[left] - direction[right]
    distance_direction = np.einsum(
        "pk,pk->p", displacement, relative_direction
    ) / distance
    radial_factor_direction = distance_direction * (
        s6 * 6.0 * distance**5 / (distance**6 + cutoff**6)**2
        + s8 * (d4.c8 / d4.c6) * 8.0 * distance**7
        / (distance**8 + cutoff**8)**2
        + s10 * (49.0 / 40.0) * (d4.c8 / d4.c6)**2
        * 10.0 * distance**9 / (distance**10 + cutoff**10)**2
    )
    coordination_potential_direction = np.array(
        coefficient_direction.coordination_potential_derivative, copy=True
    )
    np.add.at(
        coordination_potential_direction, left,
        d4.c6_coordination_left * radial_factor_direction,
    )
    np.add.at(
        coordination_potential_direction, right,
        d4.c6_coordination_right * radial_factor_direction,
    )
    result += coordination_response_gradient_sparse(
        coordination_potential_direction,
        coordinates,
        np.asarray(coordination_radii_bohr),
        electronegativities=np.asarray(electronegativities),
    )
    return result


def d4_coordination_number_jacobian(
    atomic_numbers: Sequence[int],
    coordinates_bohr: np.ndarray,
    covalent_radii_bohr: np.ndarray,
    electronegativities: np.ndarray,
) -> np.ndarray:
    """Analytic derivative of the electronegativity-weighted D4 CN."""

    numbers = tuple(int(value) for value in atomic_numbers)
    coordinates = np.asarray(coordinates_bohr, dtype=float)
    radii = np.asarray(covalent_radii_bohr, dtype=float)
    en = np.asarray(electronegativities, dtype=float)
    natoms = len(numbers)
    result = np.zeros((natoms, natoms, 3))
    left, right = np.tril_indices(natoms, k=-1)
    displacement = coordinates[left] - coordinates[right]
    distance = np.linalg.norm(displacement, axis=1)
    number_array = np.asarray(numbers, dtype=np.intp)
    critical = (4.0 / 3.0) * (
        radii[number_array[left] - 1] + radii[number_array[right] - 1]
    )
    argument = -7.5 * (distance - critical) / critical
    count_derivative = -7.5 * np.exp(-(argument**2)) / (sqrt(pi) * critical)
    factor = 4.10451 * np.exp(
        -(np.abs(en[left] - en[right]) + 19.08857) ** 2 / (2.0 * 11.28174**2)
    )
    derivative = (factor * count_derivative / distance)[:, None] * displacement
    np.add.at(result, (left, left), derivative)
    np.add.at(result, (left, right), -derivative)
    np.add.at(result, (right, left), derivative)
    np.add.at(result, (right, right), -derivative)
    return result


def coordination_response_gradient_sparse(
    coefficients: np.ndarray,
    coordinates_bohr: np.ndarray,
    radii_bohr: np.ndarray,
    *,
    electronegativities: np.ndarray | None = None,
    cutoff_bohr: float = 24.0,
) -> np.ndarray:
    """Contract dE/dCN with pair-packed GFN or D4 CN derivatives."""

    from .kernels import sparse_pair_geometry

    coefficients = np.asarray(coefficients, dtype=float)
    coordinates = np.asarray(coordinates_bohr, dtype=float)
    radii = np.asarray(radii_bohr, dtype=float)
    if (
        len(coordinates) <= 128
        and _gfn2_native is not None
        and hasattr(_gfn2_native, "coordination_response_gradient")
    ):
        return np.asarray(
            _gfn2_native.coordination_response_gradient(
                coefficients,
                coordinates,
                radii,
                None
                if electronegativities is None
                else np.asarray(electronegativities, dtype=float),
                float(cutoff_bohr),
            ),
            dtype=float,
        )
    pairs = sparse_pair_geometry(coordinates, cutoff_bohr=cutoff_bohr)
    left, right = pairs.left, pairs.right
    distance = pairs.distance
    displacement = pairs.displacement
    if electronegativities is None:
        scale1 = 4.0 * (radii[left] + radii[right]) / 3.0
        scale2 = scale1 + 2.0
        first = 1.0 / (1.0 + np.exp(-10.0 * (scale1 / distance - 1.0)))
        second = 1.0 / (1.0 + np.exp(-20.0 * (scale2 / distance - 1.0)))
        radial = (
            -10.0 * scale1 * first * (1.0 - first) * second / distance**2
            -20.0 * scale2 * second * (1.0 - second) * first / distance**2
        )
    else:
        en = np.asarray(electronegativities, dtype=float)
        critical = (4.0 / 3.0) * (radii[left] + radii[right])
        argument = -7.5 * (distance - critical) / critical
        radial = (
            -7.5 * np.exp(-(argument**2)) / (sqrt(pi) * critical)
            * 4.10451
            * np.exp(-(np.abs(en[left] - en[right]) + 19.08857) ** 2 / (2.0 * 11.28174**2))
        )
    contribution = (
        (coefficients[left] + coefficients[right]) * radial / distance
    )[:, None] * displacement
    return _pair_antisymmetric_gradient(
        left, right, contribution, len(coordinates)
    )


def coordination_number_directional_derivative_sparse(
    coordinates_bohr: np.ndarray,
    direction_bohr: np.ndarray,
    radii_bohr: np.ndarray,
    *,
    electronegativities: np.ndarray | None = None,
    cutoff_bohr: float = 24.0,
) -> np.ndarray:
    """Differentiate every GFN/D4 coordination number along one direction."""

    from .kernels import sparse_pair_geometry

    coordinates = np.asarray(coordinates_bohr, dtype=float)
    direction = np.asarray(direction_bohr, dtype=float)
    radii = np.asarray(radii_bohr, dtype=float)
    if direction.shape != coordinates.shape:
        raise ValueError("CN direction must have shape natoms x 3")
    pairs = sparse_pair_geometry(coordinates, cutoff_bohr=cutoff_bohr)
    left, right = pairs.left, pairs.right
    distance = pairs.distance
    radial_direction = np.einsum(
        "ij,ij->i",
        pairs.displacement / distance[:, None],
        direction[left] - direction[right],
    )
    if electronegativities is None:
        scale1 = 4.0 * (radii[left] + radii[right]) / 3.0
        scale2 = scale1 + 2.0
        first = 1.0 / (1.0 + np.exp(-10.0 * (scale1 / distance - 1.0)))
        second = 1.0 / (1.0 + np.exp(-20.0 * (scale2 / distance - 1.0)))
        radial = (
            -10.0 * scale1 * first * (1.0 - first) * second / distance**2
            - 20.0 * scale2 * second * (1.0 - second) * first / distance**2
        )
    else:
        en = np.asarray(electronegativities, dtype=float)
        critical = (4.0 / 3.0) * (radii[left] + radii[right])
        argument = -7.5 * (distance - critical) / critical
        radial = (
            -7.5
            * np.exp(-(argument**2))
            / (sqrt(pi) * critical)
            * 4.10451
            * np.exp(
                -(np.abs(en[left] - en[right]) + 19.08857) ** 2
                / (2.0 * 11.28174**2)
            )
        )
    pair_derivative = radial * radial_direction
    result = np.zeros(len(coordinates), dtype=float)
    np.add.at(result, left, pair_derivative)
    np.add.at(result, right, pair_derivative)
    return result


def coordination_response_hessian_vector_product_sparse(
    coefficients: np.ndarray,
    coordinates_bohr: np.ndarray,
    direction_bohr: np.ndarray,
    radii_bohr: np.ndarray,
    *,
    electronegativities: np.ndarray | None = None,
    cutoff_bohr: float = 24.0,
) -> np.ndarray:
    """Apply the fixed-coefficient CN-response Hessian on neighbor pairs."""

    from .kernels import sparse_pair_geometry

    coefficients = np.asarray(coefficients, dtype=float)
    coordinates = np.asarray(coordinates_bohr, dtype=float)
    direction = np.asarray(direction_bohr, dtype=float)
    radii = np.asarray(radii_bohr, dtype=float)
    if direction.shape != coordinates.shape:
        raise ValueError("CN Hessian direction must have shape natoms x 3")
    pairs = sparse_pair_geometry(coordinates, cutoff_bohr=cutoff_bohr)
    left, right = pairs.left, pairs.right
    distance = pairs.distance
    if electronegativities is None:
        scale1 = 4.0 * (radii[left] + radii[right]) / 3.0
        scale2 = scale1 + 2.0

        def logistic(scale: np.ndarray, steepness: float):
            value = 1.0 / (
                1.0 + np.exp(-steepness * (scale / distance - 1.0))
            )
            logarithmic = -steepness * scale / distance**2
            first = value * (1.0 - value) * logarithmic
            second = (
                first * (1.0 - 2.0 * value) * logarithmic
                + value
                * (1.0 - value)
                * (2.0 * steepness * scale / distance**3)
            )
            return value, first, second

        first_value, first_derivative, first_second = logistic(scale1, 10.0)
        second_value, second_derivative, second_second = logistic(scale2, 20.0)
        radial = (
            first_derivative * second_value
            + first_value * second_derivative
        )
        radial_second = (
            first_second * second_value
            + 2.0 * first_derivative * second_derivative
            + first_value * second_second
        )
    else:
        en = np.asarray(electronegativities, dtype=float)
        critical = (4.0 / 3.0) * (radii[left] + radii[right])
        argument = -7.5 * (distance - critical) / critical
        factor = 4.10451 * np.exp(
            -(np.abs(en[left] - en[right]) + 19.08857) ** 2
            / (2.0 * 11.28174**2)
        )
        radial = (
            -7.5
            * np.exp(-(argument**2))
            / (sqrt(pi) * critical)
            * factor
        )
        radial_second = radial * 15.0 * argument / critical
    coefficient = coefficients[left] + coefficients[right]
    return _radial_pair_hessian_vector_product(
        left,
        right,
        pairs.displacement,
        coefficient * radial,
        coefficient * radial_second,
        direction,
        len(coordinates),
    )


def atm_three_body_gradient(
    coordinates_bohr: np.ndarray,
    c6: np.ndarray,
    c8: np.ndarray,
    c6_coordination_derivative: np.ndarray,
    coordination_jacobian: np.ndarray,
    *,
    s9: float = 5.0,
    a1: float = 0.52,
    a2: float = 5.0,
    alpha: float = 16.0,
) -> np.ndarray:
    """Analytic ATM geometry and neutral-reference CN derivative."""

    coordinates = np.asarray(coordinates_bohr, dtype=float)
    c6 = np.asarray(c6, dtype=float)
    dc6 = np.asarray(c6_coordination_derivative, dtype=float)
    if _gfn2_native is not None and hasattr(_gfn2_native, "atm_three_body_gradient"):
        return np.asarray(
            _gfn2_native.atm_three_body_gradient(
                coordinates,
                c6,
                np.asarray(c8, dtype=float),
                dc6,
                np.asarray(coordination_jacobian, dtype=float),
                float(s9),
                float(a1),
                float(a2),
                float(alpha),
            ),
            dtype=float,
        )
    gradient = np.zeros_like(coordinates)
    dedcn = np.zeros(len(coordinates))
    alpha3 = alpha / 3.0
    triples = _atom_triples(len(coordinates))
    if triples.size:
        kat, jat, iat = triples.T
        vij = coordinates[jat] - coordinates[iat]
        vik = coordinates[kat] - coordinates[iat]
        vjk = coordinates[kat] - coordinates[jat]
        r2ij = np.sum(vij * vij, axis=1)
        r2ik = np.sum(vik * vik, axis=1)
        r2jk = np.sum(vjk * vjk, axis=1)
        r0ij = a1 * np.sqrt(c8[iat, jat] / c6[iat, jat]) + a2
        r0ik = a1 * np.sqrt(c8[iat, kat] / c6[iat, kat]) + a2
        r0jk = a1 * np.sqrt(c8[jat, kat] / c6[jat, kat]) + a2
        r0 = r0ij * r0ik * r0jk
        r2 = r2ij * r2ik * r2jk
        r1 = np.sqrt(r2)
        r3 = r2 * r1
        r5 = r3 * r2
        ratio = (r0 / r1) ** alpha3
        fdmp = 1.0 / (1.0 + 6.0 * ratio)
        angular = (
            0.375
            * (r2ij + r2jk - r2ik)
            * (r2ij - r2jk + r2ik)
            * (-r2ij + r2jk + r2ik)
            / r5
            + 1.0 / r3
        )
        c9 = -s9 * np.sqrt(c6[iat, jat] * c6[iat, kat] * c6[jat, kat])
        de = angular * fdmp * c9
        dfdmp = -2.0 * alpha * ratio * fdmp**2

        def pair_gradient(ra, rb, rc, vector):
            dang = -0.375 * (
                ra**3
                + ra**2 * (rb + rc)
                + ra * (3.0 * rb**2 + 2.0 * rb * rc + 3.0 * rc**2)
                - 5.0 * (rb - rc) ** 2 * (rb + rc)
            ) / r5
            return (c9 * (-dang * fdmp + angular * dfdmp) / ra)[:, None] * vector

        dgij = pair_gradient(r2ij, r2jk, r2ik, vij)
        dgik = pair_gradient(r2ik, r2jk, r2ij, vik)
        dgjk = pair_gradient(r2jk, r2ik, r2ij, vjk)
        np.add.at(gradient, iat, -(dgij + dgik))
        np.add.at(gradient, jat, dgij - dgjk)
        np.add.at(gradient, kat, dgik + dgjk)
        logarithmic = (
            dc6[iat, jat] / c6[iat, jat, None]
            + dc6[iat, kat] / c6[iat, kat, None]
            + dc6[jat, kat] / c6[jat, kat, None]
        )
        dedcn = np.sum(-0.5 * de[:, None] * logarithmic, axis=0)
    gradient += np.einsum("a,axk->xk", dedcn, coordination_jacobian)
    return gradient


@lru_cache(maxsize=64)
def _atom_triples(natoms: int) -> np.ndarray:
    """Cache the topology-only ATM triple enumeration."""

    return np.asarray(tuple(combinations(range(natoms), 3)), dtype=np.intp)


def atm_three_body_gradient_sparse(
    coordinates_bohr: np.ndarray,
    d4: SparseD4Result,
    atm: SparseATMResult,
    coordination_jacobian: np.ndarray | None,
    *,
    s9: float = 5.0,
    a1: float = 0.52,
    a2: float = 5.0,
    alpha: float = 16.0,
    coordination_radii_bohr: np.ndarray | None = None,
    electronegativities: np.ndarray | None = None,
) -> np.ndarray:
    """Analytic ATM gradient on sparse neighbor-graph triangles."""

    coordinates = np.asarray(coordinates_bohr, dtype=float)
    gradient = np.zeros_like(coordinates)
    if atm.explicit_gradient_hartree_per_bohr is not None:
        gradient = np.asarray(
            atm.explicit_gradient_hartree_per_bohr, dtype=float
        ).copy()
        coordination_derivative = atm.coordination_derivative_hartree
        if coordination_derivative is None:
            raise ValueError("fused sparse ATM result is missing its CN derivative")
        if coordination_jacobian is not None:
            gradient += np.einsum(
                "a,axk->xk",
                coordination_derivative,
                np.asarray(coordination_jacobian, dtype=float),
                optimize=False,
            )
        else:
            if coordination_radii_bohr is None or electronegativities is None:
                raise ValueError("sparse ATM CN response requires radii and electronegativities")
            gradient += coordination_response_gradient_sparse(
                coordination_derivative,
                coordinates,
                np.asarray(coordination_radii_bohr, dtype=float),
                electronegativities=np.asarray(electronegativities, dtype=float),
            )
        return gradient
    triples = np.asarray(atm.triples, dtype=np.intp)
    if not len(triples):
        return gradient
    low, middle, high = triples.T
    natoms = len(coordinates)
    cached_positions = atm.pair_positions
    if cached_positions is None:
        keys = d4.rows.astype(np.int64) * max(natoms, 1) + d4.columns

        def position(left, right):
            pair_keys = left.astype(np.int64) * max(natoms, 1) + right
            return np.searchsorted(keys, pair_keys)

        lm = position(middle, low)
        lh = position(high, low)
        mh = position(high, middle)
    else:
        lm, lh, mh = np.asarray(cached_positions, dtype=np.intp).T

    def pair(position):
        return (
            d4.c6[position], d4.c8[position],
            d4.c6_coordination_left[position],
            d4.c6_coordination_right[position],
        )

    c6ij, c8ij, dcij_i, dcij_j = pair(mh)
    c6ik, c8ik, dcik_i, dcik_k = pair(lh)
    c6jk, c8jk, dcjk_j, dcjk_k = pair(lm)
    iat, jat, kat = high, middle, low
    vij = coordinates[jat] - coordinates[iat]
    vik = coordinates[kat] - coordinates[iat]
    vjk = coordinates[kat] - coordinates[jat]
    r2ij = np.sum(vij * vij, axis=1)
    r2ik = np.sum(vik * vik, axis=1)
    r2jk = np.sum(vjk * vjk, axis=1)
    r0ij = a1 * np.sqrt(c8ij / c6ij) + a2
    r0ik = a1 * np.sqrt(c8ik / c6ik) + a2
    r0jk = a1 * np.sqrt(c8jk / c6jk) + a2
    r0 = r0ij * r0ik * r0jk
    r2 = r2ij * r2ik * r2jk
    r1 = np.sqrt(r2); r3 = r2 * r1; r5 = r3 * r2
    ratio = (r0 / r1) ** (alpha / 3.0)
    fdmp = 1.0 / (1.0 + 6.0 * ratio)
    angular = (
        0.375 * (r2ij + r2jk - r2ik) * (r2ij - r2jk + r2ik)
        * (-r2ij + r2jk + r2ik) / r5 + 1.0 / r3
    )
    c9 = -s9 * np.sqrt(c6ij * c6ik * c6jk)
    de = angular * fdmp * c9
    dfdmp = -2.0 * alpha * ratio * fdmp**2

    def pair_gradient(ra, rb, rc, vector):
        dang = -0.375 * (
            ra**3 + ra**2 * (rb + rc)
            + ra * (3.0 * rb**2 + 2.0 * rb * rc + 3.0 * rc**2)
            - 5.0 * (rb - rc) ** 2 * (rb + rc)
        ) / r5
        return (c9 * (-dang * fdmp + angular * dfdmp) / ra)[:, None] * vector

    dgij = pair_gradient(r2ij, r2jk, r2ik, vij)
    dgik = pair_gradient(r2ik, r2jk, r2ij, vik)
    dgjk = pair_gradient(r2jk, r2ik, r2ij, vjk)
    np.add.at(gradient, iat, -(dgij + dgik))
    np.add.at(gradient, jat, dgij - dgjk)
    np.add.at(gradient, kat, dgik + dgjk)
    dedcn = np.zeros(natoms, dtype=float)
    coefficient = -0.5 * de
    np.add.at(dedcn, iat, coefficient * dcij_i / c6ij)
    np.add.at(dedcn, jat, coefficient * dcij_j / c6ij)
    np.add.at(dedcn, iat, coefficient * dcik_i / c6ik)
    np.add.at(dedcn, kat, coefficient * dcik_k / c6ik)
    np.add.at(dedcn, jat, coefficient * dcjk_j / c6jk)
    np.add.at(dedcn, kat, coefficient * dcjk_k / c6jk)
    if coordination_jacobian is not None:
        gradient += np.einsum("a,axk->xk", dedcn, coordination_jacobian)
    else:
        if coordination_radii_bohr is None or electronegativities is None:
            raise ValueError("sparse ATM CN response requires radii and electronegativities")
        gradient += coordination_response_gradient_sparse(
            dedcn,
            coordinates,
            np.asarray(coordination_radii_bohr, dtype=float),
            electronegativities=np.asarray(electronegativities, dtype=float),
        )
    return gradient


def atm_three_body_hessian_vector_product_sparse(
    coordinates_bohr: np.ndarray,
    direction_bohr: np.ndarray,
    d4: SparseD4Result,
    coefficient_direction: D4CoefficientDirection,
    atm: SparseATMResult,
    coordination_radii_bohr: np.ndarray,
    electronegativities: np.ndarray,
    *,
    s9: float = 5.0,
    a1: float = 0.52,
    a2: float = 5.0,
    alpha: float = 16.0,
) -> np.ndarray:
    """Apply the analytic sparse ATM Hessian, including D4-CN variation."""

    coordinates = np.asarray(coordinates_bohr, dtype=float)
    direction = np.asarray(direction_bohr, dtype=float)
    if direction.shape != coordinates.shape:
        raise ValueError("ATM Hessian direction must have shape natoms x 3")
    result = np.zeros_like(coordinates)
    triples = np.asarray(atm.triples, dtype=np.intp)
    if not len(triples):
        return result
    if not (
        np.array_equal(d4.rows, coefficient_direction.rows)
        and np.array_equal(d4.columns, coefficient_direction.columns)
    ):
        raise ValueError("ATM coefficient direction uses a different pair pattern")
    low, middle, high = triples.T
    natoms = len(coordinates)
    cached_positions = atm.pair_positions
    if cached_positions is None:
        keys = d4.rows.astype(np.int64) * max(natoms, 1) + d4.columns

        def position(left, right):
            pair_keys = left.astype(np.int64) * max(natoms, 1) + right
            return np.searchsorted(keys, pair_keys)

        lm = position(middle, low)
        lh = position(high, low)
        mh = position(high, middle)
    else:
        lm, lh, mh = np.asarray(cached_positions, dtype=np.intp).T

    if _gfn2_native is not None and hasattr(_gfn2_native, "atm_three_body_hvp_core"):
        positions = np.column_stack((lm, lh, mh))
        result, cn_coefficient, cn_coefficient_dot = (
            _gfn2_native.atm_three_body_hvp_core(
                coordinates, direction, triples, positions,
                d4.c6, d4.c8,
                d4.c6_coordination_left, d4.c6_coordination_right,
                coefficient_direction.c6_derivative,
                coefficient_direction.c6_coordination_left_derivative,
                coefficient_direction.c6_coordination_right_derivative,
                float(s9), float(a1), float(a2), float(alpha),
            )
        )
        result += coordination_response_hessian_vector_product_sparse(
            cn_coefficient, coordinates, direction,
            np.asarray(coordination_radii_bohr),
            electronegativities=np.asarray(electronegativities),
        )
        result += coordination_response_gradient_sparse(
            cn_coefficient_dot, coordinates,
            np.asarray(coordination_radii_bohr),
            electronegativities=np.asarray(electronegativities),
        )
        return result

    def pair(position):
        return (
            d4.c6[position],
            d4.c8[position],
            d4.c6_coordination_left[position],
            d4.c6_coordination_right[position],
            coefficient_direction.c6_derivative[position],
            coefficient_direction.c6_coordination_left_derivative[position],
            coefficient_direction.c6_coordination_right_derivative[position],
        )

    c6ij, c8ij, dcij_i, dcij_j, c6ij_dot, dcij_i_dot, dcij_j_dot = pair(mh)
    c6ik, c8ik, dcik_i, dcik_k, c6ik_dot, dcik_i_dot, dcik_k_dot = pair(lh)
    c6jk, c8jk, dcjk_j, dcjk_k, c6jk_dot, dcjk_j_dot, dcjk_k_dot = pair(lm)
    iat, jat, kat = high, middle, low
    vij = coordinates[jat] - coordinates[iat]
    vik = coordinates[kat] - coordinates[iat]
    vjk = coordinates[kat] - coordinates[jat]
    vij_dot = direction[jat] - direction[iat]
    vik_dot = direction[kat] - direction[iat]
    vjk_dot = direction[kat] - direction[jat]
    a = np.sum(vij * vij, axis=1)
    b = np.sum(vik * vik, axis=1)
    c = np.sum(vjk * vjk, axis=1)
    adot = 2.0 * np.einsum("pi,pi->p", vij, vij_dot)
    bdot = 2.0 * np.einsum("pi,pi->p", vik, vik_dot)
    cdot = 2.0 * np.einsum("pi,pi->p", vjk, vjk_dot)
    r0ij = a1 * np.sqrt(c8ij / c6ij) + a2
    r0ik = a1 * np.sqrt(c8ik / c6ik) + a2
    r0jk = a1 * np.sqrt(c8jk / c6jk) + a2
    r0 = r0ij * r0ik * r0jk
    product = a * b * c
    product_dot = adot * b * c + a * bdot * c + a * b * cdot
    r1 = np.sqrt(product)
    r1dot = 0.5 * product_dot / r1
    r3 = product * r1
    r3dot = product_dot * r1 + product * r1dot
    r5 = r3 * product
    r5dot = r3dot * product + r3 * product_dot
    damping_ratio = (r0 / r1) ** (alpha / 3.0)
    damping_ratio_dot = (
        -alpha / 3.0 * damping_ratio * r1dot / r1
    )
    fdmp = 1.0 / (1.0 + 6.0 * damping_ratio)
    fdmp_dot = -6.0 * damping_ratio_dot * fdmp**2
    x = a + c - b
    y = a - c + b
    z = -a + c + b
    xdot = adot + cdot - bdot
    ydot = adot - cdot + bdot
    zdot = -adot + cdot + bdot
    numerator = 0.375 * x * y * z
    numerator_dot = 0.375 * (
        xdot * y * z + x * ydot * z + x * y * zdot
    )
    angular = numerator / r5 + 1.0 / r3
    angular_dot = (
        numerator_dot / r5
        - numerator * r5dot / r5**2
        - r3dot / r3**2
    )
    c9 = -s9 * np.sqrt(c6ij * c6ik * c6jk)
    c9_log_dot = 0.5 * (
        c6ij_dot / c6ij + c6ik_dot / c6ik + c6jk_dot / c6jk
    )
    c9dot = c9 * c9_log_dot
    energy = angular * fdmp * c9
    energy_dot = (
        angular_dot * fdmp * c9
        + angular * fdmp_dot * c9
        + angular * fdmp * c9dot
    )
    dfdmp = -2.0 * alpha * damping_ratio * fdmp**2
    dfdmp_dot = -2.0 * alpha * (
        damping_ratio_dot * fdmp**2
        + 2.0 * damping_ratio * fdmp * fdmp_dot
    )

    def pair_gradient_and_direction(ra, rb, rc, radot, rbdot, rcdot, vector, vector_dot):
        polynomial = (
            ra**3
            + ra**2 * (rb + rc)
            + ra * (3.0 * rb**2 + 2.0 * rb * rc + 3.0 * rc**2)
            - 5.0 * (rb - rc) ** 2 * (rb + rc)
        )
        polynomial_dot = (
            3.0 * ra**2 * radot
            + 2.0 * ra * radot * (rb + rc)
            + ra**2 * (rbdot + rcdot)
            + radot * (3.0 * rb**2 + 2.0 * rb * rc + 3.0 * rc**2)
            + ra
            * (
                6.0 * rb * rbdot
                + 2.0 * (rbdot * rc + rb * rcdot)
                + 6.0 * rc * rcdot
            )
            - 5.0
            * (
                2.0 * (rb - rc) * (rbdot - rcdot) * (rb + rc)
                + (rb - rc) ** 2 * (rbdot + rcdot)
            )
        )
        dang = -0.375 * polynomial / r5
        dang_dot = -0.375 * (
            polynomial_dot / r5 - polynomial * r5dot / r5**2
        )
        bracket = -dang * fdmp + angular * dfdmp
        bracket_dot = (
            -dang_dot * fdmp
            - dang * fdmp_dot
            + angular_dot * dfdmp
            + angular * dfdmp_dot
        )
        coefficient = c9 * bracket / ra
        coefficient_dot = (
            c9dot * bracket / ra
            + c9 * bracket_dot / ra
            - c9 * bracket * radot / ra**2
        )
        return (
            coefficient[:, None] * vector,
            coefficient_dot[:, None] * vector + coefficient[:, None] * vector_dot,
        )

    _, dgij = pair_gradient_and_direction(a, c, b, adot, cdot, bdot, vij, vij_dot)
    _, dgik = pair_gradient_and_direction(b, c, a, bdot, cdot, adot, vik, vik_dot)
    _, dgjk = pair_gradient_and_direction(c, b, a, cdot, bdot, adot, vjk, vjk_dot)
    np.add.at(result, iat, -(dgij + dgik))
    np.add.at(result, jat, dgij - dgjk)
    np.add.at(result, kat, dgik + dgjk)

    cn_coefficient = np.zeros(natoms, dtype=float)
    cn_coefficient_dot = np.zeros(natoms, dtype=float)
    prefactor = -0.5 * energy
    prefactor_dot = -0.5 * energy_dot

    def add_cn(atom, coefficient, coefficient_dot, c6, c6dot):
        ratio = coefficient / c6
        ratio_dot = coefficient_dot / c6 - coefficient * c6dot / c6**2
        np.add.at(cn_coefficient, atom, prefactor * ratio)
        np.add.at(
            cn_coefficient_dot,
            atom,
            prefactor_dot * ratio + prefactor * ratio_dot,
        )

    add_cn(iat, dcij_i, dcij_i_dot, c6ij, c6ij_dot)
    add_cn(jat, dcij_j, dcij_j_dot, c6ij, c6ij_dot)
    add_cn(iat, dcik_i, dcik_i_dot, c6ik, c6ik_dot)
    add_cn(kat, dcik_k, dcik_k_dot, c6ik, c6ik_dot)
    add_cn(jat, dcjk_j, dcjk_j_dot, c6jk, c6jk_dot)
    add_cn(kat, dcjk_k, dcjk_k_dot, c6jk, c6jk_dot)
    result += coordination_response_hessian_vector_product_sparse(
        cn_coefficient,
        coordinates,
        direction,
        np.asarray(coordination_radii_bohr),
        electronegativities=np.asarray(electronegativities),
    )
    result += coordination_response_gradient_sparse(
        cn_coefficient_dot,
        coordinates,
        np.asarray(coordination_radii_bohr),
        electronegativities=np.asarray(electronegativities),
    )
    return result


def assert_rigid_motion_invariance(
    coordinates_bohr: np.ndarray,
    gradient_hartree_per_bohr: np.ndarray,
    *,
    tolerance: float = 1.0e-9,
) -> None:
    coordinates = np.asarray(coordinates_bohr, dtype=float)
    gradient = np.asarray(gradient_hartree_per_bohr, dtype=float)
    translation = np.sum(gradient, axis=0)
    centered = coordinates - np.mean(coordinates, axis=0)
    torque = np.sum(np.cross(centered, gradient), axis=0)
    if np.max(np.abs(translation)) > tolerance:
        raise ValueError("gradient violates translational invariance")
    if np.max(np.abs(torque)) > tolerance:
        raise ValueError("gradient violates rotational invariance")


__all__ = [
    "VariationalGradientTerms",
    "anisotropic_explicit_gradient",
    "anisotropic_explicit_hessian_vector_product_sparse",
    "atomic_multipole_derivatives",
    "assemble_variational_gradient",
    "assemble_variational_gradient_sparse",
    "assemble_variational_hessian_vector_sparse",
    "assert_rigid_motion_invariance",
    "frozen_d4_two_body_gradient",
    "frozen_d4_two_body_hessian_vector_product",
    "frozen_d4_two_body_hessian_vector_product_sparse",
    "frozen_d4_two_body_gradient_sparse",
    "coordination_number_jacobian",
    "coordination_response_hessian_vector_product_sparse",
    "coordination_response_gradient_sparse",
    "d4_coordination_number_jacobian",
    "d4_two_body_hessian_vector_product_sparse",
    "energy_weighted_density",
    "energy_weighted_density_from_fock",
    "energy_weighted_density_from_fock_sparse",
    "frozen_charge_scc_gradient",
    "frozen_charge_scc_hessian_vector_product",
    "shell_charge_derivatives",
    "atm_three_body_gradient",
    "atm_three_body_gradient_sparse",
    "atm_three_body_hessian_vector_product_sparse",
    "repulsion_gradient",
    "repulsion_hessian_vector_product_sparse",
    "repulsion_gradient_sparse",
]
