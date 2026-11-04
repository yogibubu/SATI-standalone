"""Cumulative atomic multipoles and anisotropic GFN2 energy terms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from ._array_cache import triangular_indices

from .basis import SparseAtomicMultipoleIntegrals
from .localized_solver import SparseSymmetricMatrix
from .scc import shell_charges_sparse

try:
    from . import _gfn2_native
except ImportError:  # pragma: no cover - portable source-only installation
    _gfn2_native = None


@dataclass(frozen=True)
class AtomicMultipoles:
    charges: np.ndarray
    dipoles_bohr: np.ndarray
    quadrupoles_bohr2: np.ndarray


@dataclass(frozen=True)
class AnisotropicParameters:
    damping_radius_bohr: np.ndarray
    dipole_xc: np.ndarray
    quadrupole_xc: np.ndarray
    a3: float = 3.0
    a5: float = 4.0


@dataclass(frozen=True)
class CumulativeMultipoleOperators:
    """Pair-packed AO moments reused throughout one SCC calculation."""

    atom_index: np.ndarray
    centers: np.ndarray
    pair_overlap: np.ndarray
    pair_dipole: np.ndarray
    pair_second: np.ndarray


@dataclass(frozen=True)
class SparseCumulativeMultipoleOperators:
    rows: np.ndarray
    columns: np.ndarray
    left_atom: np.ndarray
    right_atom: np.ndarray
    overlap: np.ndarray
    dmu_left: np.ndarray | None
    dmu_right: np.ndarray | None
    raw_theta_left: np.ndarray | None
    raw_theta_right: np.ndarray | None
    dquadrupole_left: np.ndarray | None
    dquadrupole_right: np.ndarray | None
    dipole: np.ndarray
    second: np.ndarray
    left_center: np.ndarray
    right_center: np.ndarray

    def raw_endpoint_arrays(self):
        """Materialize only dipole and raw-second-moment endpoints."""

        if self.dmu_left is not None:
            return (
                self.dmu_left, self.dmu_right,
                self.raw_theta_left, self.raw_theta_right,
            )

        def endpoint(center: np.ndarray):
            dmu = center * self.overlap[:, None] - self.dipole
            raw = (
                self.dipole[:, :, None] * center[:, None, :]
                + center[:, :, None] * self.dipole[:, None, :]
                - self.overlap[:, None, None] * center[:, :, None] * center[:, None, :]
                - self.second
            )
            return dmu, raw

        left = endpoint(self.left_center)
        right = endpoint(self.right_center)
        return left[0], right[0], left[1], right[1]

    def quadrupole_endpoint_arrays(self):
        """Materialize dipole and traceless-quadrupole endpoints."""

        if self.dmu_left is not None:
            return (
                self.dmu_left, self.dmu_right,
                self.dquadrupole_left, self.dquadrupole_right,
            )
        dmu_left, dmu_right, raw_left, raw_right = self.raw_endpoint_arrays()

        def traceless(raw: np.ndarray) -> np.ndarray:
            trace = np.trace(raw, axis1=1, axis2=2)
            return 1.5 * raw - 0.5 * trace[:, None, None] * np.eye(3)[None, :, :]

        return dmu_left, dmu_right, traceless(raw_left), traceless(raw_right)


@dataclass(frozen=True)
class AnisotropicGeometry:
    left: np.ndarray
    right: np.ndarray
    displacement: np.ndarray
    distance2: np.ndarray
    f3: np.ndarray
    f5: np.ndarray
    outer: np.ndarray
    omitted_pairs: int = 0
    cutoff_bohr: float = float("inf")


def prepare_anisotropic_geometry(
    coordinates_bohr: np.ndarray,
    parameters: AnisotropicParameters,
) -> AnisotropicGeometry:
    coordinates = np.asarray(coordinates_bohr, dtype=float)
    radii = np.asarray(parameters.damping_radius_bohr, dtype=float)
    left, right = triangular_indices(len(coordinates), diagonal=False)
    displacement = coordinates[right] - coordinates[left]
    distance2 = np.sum(displacement * displacement, axis=1)
    distance = np.sqrt(distance2)
    radius = 0.5 * (radii[left] + radii[right])
    f3 = distance**-3 / (1.0 + 6.0 * (radius / distance) ** parameters.a3)
    f5 = distance**-5 / (1.0 + 6.0 * (radius / distance) ** parameters.a5)
    outer = displacement[:, :, None] * displacement[:, None, :]
    return AnisotropicGeometry(left, right, displacement, distance2, f3, f5, outer)


def prepare_anisotropic_geometry_sparse(
    coordinates_bohr: np.ndarray,
    parameters: AnisotropicParameters,
    *,
    cutoff_bohr: float = 24.0,
) -> AnisotropicGeometry:
    from .kernels import sparse_pair_geometry

    coordinates = np.asarray(coordinates_bohr, dtype=float)
    radii = np.asarray(parameters.damping_radius_bohr, dtype=float)
    pairs = sparse_pair_geometry(coordinates, cutoff_bohr=cutoff_bohr)
    # Existing anisotropic convention uses displacement right-left.
    displacement = -pairs.displacement
    distance2 = pairs.distance * pairs.distance
    radius = 0.5 * (radii[pairs.left] + radii[pairs.right])
    f3 = pairs.distance**-3 / (
        1.0 + 6.0 * (radius / pairs.distance) ** parameters.a3
    )
    f5 = pairs.distance**-5 / (
        1.0 + 6.0 * (radius / pairs.distance) ** parameters.a5
    )
    outer = displacement[:, :, None] * displacement[:, None, :]
    return AnisotropicGeometry(
        pairs.left, pairs.right, displacement, distance2, f3, f5, outer,
        pairs.omitted_pairs, pairs.cutoff_bohr,
    )


def prepare_cumulative_multipole_operators(
    overlap: np.ndarray,
    dipole_integrals: np.ndarray,
    second_moment_integrals: np.ndarray,
    coordinates_bohr: np.ndarray,
    orbital_atom: Sequence[int],
) -> CumulativeMultipoleOperators:
    atom_index = np.asarray(tuple(int(value) for value in orbital_atom), dtype=np.int64)
    coordinates = np.asarray(coordinates_bohr, dtype=float)
    size = len(atom_index)
    left, right = triangular_indices(size)
    centers = np.ascontiguousarray(coordinates[atom_index])
    pair_overlap = np.ascontiguousarray(np.asarray(overlap, dtype=float)[left, right])
    pair_dipole = np.ascontiguousarray(
        np.asarray(dipole_integrals, dtype=float)[:, left, right].T
    )
    pair_second = np.ascontiguousarray(
        np.asarray(second_moment_integrals, dtype=float)[:, :, left, right]
        .transpose(2, 0, 1)
    )
    return CumulativeMultipoleOperators(
        atom_index, centers, pair_overlap, pair_dipole, pair_second
    )


def prepare_pair_cumulative_multipole_operators(
    integrals: SparseAtomicMultipoleIntegrals,
    coordinates_bohr: np.ndarray,
    orbital_atom: Sequence[int],
) -> CumulativeMultipoleOperators:
    """Build dense-route packed operators without transient AO tensors.

    The dense SCC kernel uses a complete lower-triangular pair stream, but
    the integral engine already returns the nonzero lower pairs.  Scattering
    directly into that packed stream avoids materializing 3*N*N and 9*N*N
    dipole/second-moment tensors merely to pack them again.
    """

    atom_index = np.asarray(tuple(int(value) for value in orbital_atom), dtype=np.int64)
    if len(atom_index) != integrals.size:
        raise ValueError("pair integrals and orbital assignment differ")
    size = integrals.size
    pair_count = size * (size + 1) // 2
    pair_overlap = np.zeros(pair_count, dtype=float)
    pair_dipole = np.zeros((pair_count, 3), dtype=float)
    pair_second = np.zeros((pair_count, 3, 3), dtype=float)
    rows = np.asarray(integrals.rows, dtype=np.intp)
    columns = np.asarray(integrals.columns, dtype=np.intp)
    if np.any(rows < columns):
        raise ValueError("pair integrals must store the lower triangle")
    packed = rows * (rows + 1) // 2 + columns
    pair_overlap[packed] = integrals.overlap
    pair_dipole[packed] = integrals.dipole
    pair_second[packed] = integrals.second
    centers = np.ascontiguousarray(
        np.asarray(coordinates_bohr, dtype=float)[atom_index]
    )
    return CumulativeMultipoleOperators(
        atom_index, centers, pair_overlap, pair_dipole, pair_second
    )


def _packed_endpoint_operators(
    operators: CumulativeMultipoleOperators,
    *,
    pair_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Materialize endpoint operators, optionally on an exact pair subset."""

    size = len(operators.atom_index)
    left, right = triangular_indices(size)
    overlap = operators.pair_overlap
    dipole = operators.pair_dipole
    second = operators.pair_second
    if pair_indices is not None:
        left, right = left[pair_indices], right[pair_indices]
        overlap, dipole, second = (
            overlap[pair_indices], dipole[pair_indices], second[pair_indices]
        )

    def endpoint(center: np.ndarray):
        dmu = center * overlap[:, None] - dipole
        raw = (
            dipole[:, :, None] * center[:, None, :]
            + center[:, :, None] * dipole[:, None, :]
            - overlap[:, None, None] * center[:, :, None] * center[:, None, :]
            - second
        )
        trace = np.trace(raw, axis1=1, axis2=2)
        dquadrupole = 1.5 * raw
        for component in range(3):
            dquadrupole[:, component, component] -= 0.5 * trace
        return dmu, raw, dquadrupole

    dmu_left, raw_left, dquad_left = endpoint(operators.centers[left])
    dmu_right, raw_right, dquad_right = endpoint(operators.centers[right])
    return dmu_left, raw_left, dquad_left, dmu_right, raw_right, dquad_right


def prepare_sparse_cumulative_multipole_operators(
    integrals: SparseAtomicMultipoleIntegrals,
    coordinates_bohr: np.ndarray,
    orbital_atom: Sequence[int],
    *,
    compact: bool = False,
) -> SparseCumulativeMultipoleOperators:
    assignment = np.asarray(tuple(int(value) for value in orbital_atom), dtype=np.intp)
    coordinates = np.asarray(coordinates_bohr, dtype=float)
    if len(assignment) != integrals.size:
        raise ValueError("sparse integrals and orbital assignment differ")
    left_atom = assignment[integrals.rows]
    right_atom = assignment[integrals.columns]
    left_center = coordinates[left_atom]
    right_center = coordinates[right_atom]
    overlap = integrals.overlap
    dipole = integrals.dipole
    second = integrals.second

    def endpoint(center: np.ndarray):
        dmu = center * overlap[:, None] - dipole
        raw_theta = (
            dipole[:, :, None] * center[:, None, :]
            + center[:, :, None] * dipole[:, None, :]
            - overlap[:, None, None] * center[:, :, None] * center[:, None, :]
            - second
        )
        trace = np.trace(raw_theta, axis1=1, axis2=2)
        dquadrupole = (
            1.5 * raw_theta
            - 0.5 * trace[:, None, None] * np.eye(3)[None, :, :]
        )
        return dmu, raw_theta, dquadrupole

    if compact:
        dmu_left = dmu_right = None
        raw_left = raw_right = None
        quadrupole_left = quadrupole_right = None
    else:
        dmu_left, raw_left, quadrupole_left = endpoint(left_center)
        dmu_right, raw_right, quadrupole_right = endpoint(right_center)
    return SparseCumulativeMultipoleOperators(
        integrals.rows, integrals.columns, left_atom, right_atom, overlap,
        dmu_left, dmu_right, raw_left, raw_right,
        quadrupole_left, quadrupole_right, dipole, second,
        left_center, right_center,
    )


def cumulative_atomic_multipoles_sparse(
    density: SparseSymmetricMatrix,
    operators: SparseCumulativeMultipoleOperators,
    valence_nuclear_charges: Sequence[float],
) -> AtomicMultipoles:
    """Evaluate CAMMs directly from pair-packed density and AO operators."""

    values = density.values_on(operators.rows, operators.columns)
    natoms = len(valence_nuclear_charges)
    charges = np.asarray(valence_nuclear_charges, dtype=float).copy()
    dipoles = np.zeros((natoms, 3), dtype=float)
    raw_quadrupoles = np.zeros((natoms, 3, 3), dtype=float)
    diagonal = operators.rows == operators.columns
    dmu_left, dmu_right, raw_left, raw_right = operators.raw_endpoint_arrays()
    np.add.at(charges, operators.left_atom, -values * operators.overlap)
    np.add.at(dipoles, operators.left_atom, values[:, None] * dmu_left)
    np.add.at(
        raw_quadrupoles,
        operators.left_atom,
        values[:, None, None] * raw_left,
    )
    off = ~diagonal
    np.add.at(
        charges, operators.right_atom[off],
        -values[off] * operators.overlap[off],
    )
    np.add.at(
        dipoles, operators.right_atom[off],
        values[off, None] * dmu_right[off],
    )
    np.add.at(
        raw_quadrupoles, operators.right_atom[off],
        values[off, None, None] * raw_right[off],
    )
    trace = np.trace(raw_quadrupoles, axis1=1, axis2=2)
    quadrupoles = (
        1.5 * raw_quadrupoles
        - 0.5 * trace[:, None, None] * np.eye(3)[None, :, :]
    )
    return AtomicMultipoles(charges, dipoles, quadrupoles)


def cumulative_atomic_multipole_directional_derivative_sparse(
    density: SparseSymmetricMatrix,
    integrals: SparseAtomicMultipoleIntegrals,
    coordinates_bohr: np.ndarray,
    direction_bohr: np.ndarray,
    orbital_atom: Sequence[int],
) -> AtomicMultipoles:
    """Differentiate pair-packed CAMMs at fixed AO density along a direction."""

    assignment = np.asarray(tuple(int(value) for value in orbital_atom), dtype=np.intp)
    coordinates = np.asarray(coordinates_bohr, dtype=float)
    direction = np.asarray(direction_bohr, dtype=float)
    if direction.shape != coordinates.shape or len(assignment) != integrals.size:
        raise ValueError("CAMM direction or AO assignment is inconsistent")
    left_atom_all = assignment[integrals.rows]
    right_atom_all = assignment[integrals.columns]
    moved = np.any(direction != 0.0, axis=1)
    active_pairs = moved[left_atom_all] | moved[right_atom_all]
    natoms = len(coordinates)
    if not np.any(active_pairs):
        return AtomicMultipoles(
            np.zeros(natoms), np.zeros((natoms, 3)), np.zeros((natoms, 3, 3))
        )
    selected = slice(None) if np.all(active_pairs) else np.flatnonzero(active_pairs)
    left_atom = left_atom_all[selected]
    right_atom = right_atom_all[selected]
    left_direction = direction[left_atom]
    right_direction = direction[right_atom]
    derivative = integrals.left_derivatives[selected]
    right_derivative = integrals.right_derivatives[selected]
    overlap_direction = np.einsum(
        "pck,pk->pc", derivative[:, :1], left_direction, optimize=True
    )[:, 0] + np.einsum(
        "pck,pk->pc", right_derivative[:, :1], right_direction, optimize=True
    )[:, 0]
    dipole_direction = np.einsum(
        "pck,pk->pc", derivative[:, 1:4], left_direction, optimize=True
    ) + np.einsum(
        "pck,pk->pc", right_derivative[:, 1:4], right_direction, optimize=True
    )
    second_direction = np.einsum(
        "pck,pk->pc", derivative[:, 4:].reshape(-1, 9, 3), left_direction,
        optimize=True,
    ).reshape(-1, 3, 3) + np.einsum(
        "pck,pk->pc", right_derivative[:, 4:].reshape(-1, 9, 3), right_direction,
        optimize=True,
    ).reshape(-1, 3, 3)
    values = density.values_on(
        integrals.rows[selected], integrals.columns[selected]
    )
    overlap = integrals.overlap[selected]
    dipole_integrals = integrals.dipole[selected]
    charge = np.zeros(natoms, dtype=float)
    dipole = np.zeros((natoms, 3), dtype=float)
    raw = np.zeros((natoms, 3, 3), dtype=float)

    def endpoint(atom, center, center_direction, include):
        ds = overlap_direction[include]
        dd = dipole_direction[include]
        dsecond = second_direction[include]
        s = overlap[include]
        d = dipole_integrals[include]
        weight = values[include]
        dmu = (
            center_direction * s[:, None]
            + center * ds[:, None]
            - dd
        )
        draw = (
            dd[:, :, None] * center[:, None, :]
            + d[:, :, None] * center_direction[:, None, :]
            + center_direction[:, :, None] * d[:, None, :]
            + center[:, :, None] * dd[:, None, :]
            - ds[:, None, None] * center[:, :, None] * center[:, None, :]
            - s[:, None, None]
            * (
                center_direction[:, :, None] * center[:, None, :]
                + center[:, :, None] * center_direction[:, None, :]
            )
            - dsecond
        )
        np.add.at(charge, atom, -weight * ds)
        np.add.at(dipole, atom, weight[:, None] * dmu)
        np.add.at(raw, atom, weight[:, None, None] * draw)

    all_pairs = np.ones(len(left_atom), dtype=bool)
    endpoint(
        left_atom,
        coordinates[left_atom],
        left_direction,
        all_pairs,
    )
    off = integrals.rows[selected] != integrals.columns[selected]
    endpoint(
        right_atom[off],
        coordinates[right_atom[off]],
        right_direction[off],
        off,
    )
    trace = np.trace(raw, axis1=1, axis2=2)
    quadrupole = 1.5 * raw - 0.5 * trace[:, None, None] * np.eye(3)[None]
    return AtomicMultipoles(charge, dipole, quadrupole)


def sparse_orthogonal_observables(
    projector: SparseSymmetricMatrix,
    orthogonalizer: SparseSymmetricMatrix,
    domains,
    partition,
    operators: SparseCumulativeMultipoleOperators,
    valence_nuclear_charges: Sequence[float],
) -> tuple[np.ndarray, AtomicMultipoles]:
    """Accumulate SCC/CAMM observables directly from an orthogonal projector."""

    if _gfn2_native is None or not hasattr(
        _gfn2_native, "sparse_orthogonal_observables"
    ):
        density, _ = orthogonalizer.congruence(
            projector, domains, threshold=0.0
        )
        density = SparseSymmetricMatrix(
            density.size, density.rows, density.columns, 2.0 * density.values
        )
        return (
            shell_charges_sparse(density, SparseSymmetricMatrix(
                density.size, operators.rows, operators.columns, operators.overlap
            ), partition),
            cumulative_atomic_multipoles_sparse(
                density, operators, valence_nuclear_charges
            ),
        )
    allowed_left, allowed_right = domains.allowed_pair_arrays()
    if operators.dmu_left is None:
        endpoint_arguments = (
            operators.dipole, operators.second,
            operators.left_center, operators.right_center,
        )
    else:
        endpoint_arguments = operators.raw_endpoint_arrays()
    shell_q, charges, dipoles, quadrupoles, _ = _gfn2_native.sparse_orthogonal_observables(
        int(projector.size),
        int(partition.nshells),
        int(len(valence_nuclear_charges)),
        orthogonalizer.rows, orthogonalizer.columns, orthogonalizer.values,
        projector.rows, projector.columns, projector.values,
        allowed_left, allowed_right,
        np.asarray(partition.orbital_shell, dtype=np.int64),
        np.asarray(partition.reference_population, dtype=float),
        valence_nuclear_charges,
        operators.rows, operators.columns,
        operators.left_atom, operators.right_atom, operators.overlap,
        *endpoint_arguments,
        int(domains.native_threads),
    )
    return (
        np.asarray(shell_q, dtype=float),
        AtomicMultipoles(
            np.asarray(charges, dtype=float),
            np.asarray(dipoles, dtype=float),
            np.asarray(quadrupoles, dtype=float),
        ),
    )


def cumulative_atomic_multipoles(
    density: np.ndarray,
    overlap: np.ndarray,
    dipole_integrals: np.ndarray,
    second_moment_integrals: np.ndarray,
    coordinates_bohr: np.ndarray,
    orbital_atom: Sequence[int],
    valence_nuclear_charges: Sequence[float],
    *,
    operators: CumulativeMultipoleOperators | None = None,
) -> AtomicMultipoles:
    """Evaluate CAMMs from published GFN2 Eqs. 26-27."""

    density = np.asarray(density, dtype=float)
    overlap = np.asarray(overlap, dtype=float)
    coordinates = np.asarray(coordinates_bohr, dtype=float)
    assignment = tuple(int(value) for value in orbital_atom)
    natoms = len(valence_nuclear_charges)
    norbitals = len(assignment)
    if density.shape != (norbitals, norbitals) or overlap.shape != density.shape:
        raise ValueError("density and overlap must match orbital_atom")
    if coordinates.shape != (natoms, 3):
        raise ValueError("coordinates_bohr must have shape natoms x 3")

    if operators is None:
        dipole = np.asarray(dipole_integrals, dtype=float)
        second = np.asarray(second_moment_integrals, dtype=float)
        if dipole.shape != (3, norbitals, norbitals):
            raise ValueError("dipole integrals must have shape 3 x nao x nao")
        if second.shape != (3, 3, norbitals, norbitals):
            raise ValueError("second moments must have shape 3 x 3 x nao x nao")
        prepared = prepare_cumulative_multipole_operators(
            overlap, dipole, second, coordinates, assignment
        )
    else:
        prepared = operators
    _, multipoles = dense_scc_observables(
        density,
        prepared,
        prepared.atom_index,
        np.zeros(natoms, dtype=float),
        valence_nuclear_charges,
    )
    return multipoles


def dense_scc_observables(
    density: np.ndarray,
    operators: CumulativeMultipoleOperators,
    orbital_shell: Sequence[int],
    reference_population: Sequence[float],
    valence_nuclear_charges: Sequence[float],
    *,
    native_context=None,
) -> tuple[np.ndarray, AtomicMultipoles]:
    """Contract dense shell populations and CAMMs in one native matrix scan."""

    if _gfn2_native is None or not hasattr(_gfn2_native, "dense_scc_observables"):
        size = len(operators.atom_index)
        left, right = triangular_indices(size)
        off = left != right
        values_left = np.asarray(density)[left, right]
        values_right = np.asarray(density)[right, left]
        dmu_l, raw_l, _, dmu_r, raw_r, _ = _packed_endpoint_operators(operators)
        shell_q = np.asarray(reference_population, dtype=float).copy()
        charges = np.asarray(valence_nuclear_charges, dtype=float).copy()
        dipoles = np.zeros((len(charges), 3), dtype=float)
        raw_quadrupoles = np.zeros((len(charges), 3, 3), dtype=float)
        shell = np.asarray(orbital_shell, dtype=np.intp)
        atom = operators.atom_index
        np.add.at(shell_q, shell[left], -values_left * operators.pair_overlap)
        np.add.at(charges, atom[left], -values_left * operators.pair_overlap)
        np.add.at(dipoles, atom[left], values_left[:, None] * dmu_l)
        np.add.at(raw_quadrupoles, atom[left], values_left[:, None, None] * raw_l)
        np.add.at(shell_q, shell[right[off]], -values_right[off] * operators.pair_overlap[off])
        np.add.at(charges, atom[right[off]], -values_right[off] * operators.pair_overlap[off])
        np.add.at(dipoles, atom[right[off]], values_right[off, None] * dmu_r[off])
        np.add.at(
            raw_quadrupoles,
            atom[right[off]],
            values_right[off, None, None] * raw_r[off],
        )
        trace = np.trace(raw_quadrupoles, axis1=1, axis2=2)
        quadrupoles = 1.5 * raw_quadrupoles
        for component in range(3):
            quadrupoles[:, component, component] -= 0.5 * trace
        return shell_q, AtomicMultipoles(charges, dipoles, quadrupoles)
    shell_q, charges, dipoles, quadrupoles = (
        native_context.dense_observables(density)
        if native_context is not None
        else _gfn2_native.dense_scc_observables(
            density,
            operators.pair_overlap,
            operators.pair_dipole,
            operators.pair_second,
            operators.centers,
            operators.atom_index,
            np.asarray(orbital_shell, dtype=np.int64),
            np.asarray(reference_population, dtype=float),
            np.asarray(valence_nuclear_charges, dtype=float),
        )
    )
    return np.asarray(shell_q), AtomicMultipoles(
        np.asarray(charges), np.asarray(dipoles), np.asarray(quadrupoles)
    )


def dense_scc_fock(
    core: np.ndarray,
    overlap: np.ndarray,
    operators: CumulativeMultipoleOperators,
    orbital_shell: Sequence[int],
    shell_potential: np.ndarray,
    charge_potential: np.ndarray,
    dipole_potential: np.ndarray,
    quadrupole_potential: np.ndarray,
    d4_charge_derivative: np.ndarray,
    *,
    native_context=None,
) -> np.ndarray:
    """Assemble all density-dependent dense Fock terms in one native pass."""

    if _gfn2_native is not None and hasattr(_gfn2_native, "dense_scc_fock"):
        if native_context is not None:
            return native_context.dense_fock(
                shell_potential,
                charge_potential,
                dipole_potential,
                quadrupole_potential,
                d4_charge_derivative,
            )
        return np.asarray(
            _gfn2_native.dense_scc_fock(
                core, overlap, operators.pair_overlap, operators.pair_dipole,
                operators.pair_second, operators.centers,
                operators.atom_index,
                np.asarray(orbital_shell, dtype=np.int64), shell_potential,
                charge_potential, dipole_potential, quadrupole_potential,
                d4_charge_derivative,
            )
        )
    assignment = operators.atom_index
    size = len(assignment)
    left, right = triangular_indices(size)
    dmu_l, _, dquad_l, dmu_r, _, dquad_r = _packed_endpoint_operators(operators)
    qpot = np.asarray(charge_potential)
    mupot = np.asarray(dipole_potential)
    thetapot = np.asarray(quadrupole_potential)

    def endpoint(atom, dmu, dquad):
        return (
            -qpot[atom] * operators.pair_overlap
            + np.einsum("pa,pa->p", mupot[atom], dmu, optimize=False)
            + np.einsum("pac,pac->p", thetapot[atom], dquad, optimize=False)
        )

    raw_l = endpoint(assignment[left], dmu_l, dquad_l)
    raw_r = endpoint(assignment[right], dmu_r, dquad_r)
    orbital_potential = np.asarray(shell_potential)[
        np.asarray(orbital_shell, dtype=np.intp)
    ] + np.asarray(d4_charge_derivative)[assignment]
    values = (
        np.asarray(core)[left, right]
        - 0.5 * np.asarray(overlap)[left, right]
        * (orbital_potential[left] + orbital_potential[right])
        + 0.5 * (raw_l + raw_r)
    )
    result = np.empty_like(core)
    result[left, right] = values
    result[right, left] = values
    return result


def _damped_inverse_power(distance: float, radius: float, power: int, damping: float) -> float:
    return distance ** (-power) / (1.0 + 6.0 * (radius / distance) ** damping)


def anisotropic_energy_and_moment_potentials(
    multipoles: AtomicMultipoles,
    coordinates_bohr: np.ndarray,
    parameters: AnisotropicParameters,
    *,
    geometry: AnisotropicGeometry | None = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Return AES+AXC energy and derivatives with respect to q, mu, Theta."""

    coordinates = np.asarray(coordinates_bohr, dtype=float)
    charges = np.asarray(multipoles.charges, dtype=float)
    dipoles = np.asarray(multipoles.dipoles_bohr, dtype=float)
    quadrupoles = np.asarray(multipoles.quadrupoles_bohr2, dtype=float)
    radii = np.asarray(parameters.damping_radius_bohr, dtype=float)
    dipole_xc = np.asarray(parameters.dipole_xc, dtype=float)
    quadrupole_xc = np.asarray(parameters.quadrupole_xc, dtype=float)
    natoms = len(charges)
    if coordinates.shape != (natoms, 3):
        raise ValueError("coordinates and multipoles have inconsistent atom counts")
    if any(values.shape != (natoms,) for values in (radii, dipole_xc, quadrupole_xc)):
        raise ValueError("anisotropic parameter arrays must contain one value per atom")

    charge_potential = np.zeros(natoms, dtype=float)
    dipole_potential = 2.0 * dipole_xc[:, None] * dipoles
    quadrupole_potential = 2.0 * quadrupole_xc[:, None, None] * quadrupoles
    energy = float(
        np.sum(dipole_xc[:, None] * dipoles**2)
        + np.sum(quadrupole_xc[:, None, None] * quadrupoles**2)
    )
    prepared = geometry or prepare_anisotropic_geometry(coordinates, parameters)
    left, right = prepared.left, prepared.right
    displacement = prepared.displacement
    f3, f5 = prepared.f3, prepared.f5
    if _gfn2_native is not None and hasattr(
        _gfn2_native, "anisotropic_energy_potentials"
    ):
        energy, charge_potential, dipole_potential, quadrupole_potential = (
            _gfn2_native.anisotropic_energy_potentials(
                charges,
                dipoles,
                quadrupoles,
                dipole_xc,
                quadrupole_xc,
                left,
                right,
                displacement,
                f3,
                f5,
                prepared.distance2,
            )
        )
        return (
            float(energy),
            np.asarray(charge_potential),
            np.asarray(dipole_potential),
            np.asarray(quadrupole_potential),
        )
    left_projection = np.einsum("pi,pi->p", dipoles[left], displacement)
    right_projection = np.einsum("pi,pi->p", dipoles[right], displacement)
    left_quadrupole = np.einsum(
        "pi,pij,pj->p", displacement, quadrupoles[left], displacement
    )
    right_quadrupole = np.einsum(
        "pi,pij,pj->p", displacement, quadrupoles[right], displacement
    )
    distance2 = prepared.distance2
    energy += float(
        np.sum(
            f3
            * (-charges[left] * right_projection + charges[right] * left_projection)
            + f5
            * (
                charges[left] * right_quadrupole
                + charges[right] * left_quadrupole
                - 3.0 * left_projection * right_projection
                + np.einsum("pi,pi->p", dipoles[left], dipoles[right]) * distance2
            )
        )
    )
    np.add.at(charge_potential, left, -f3 * right_projection + f5 * right_quadrupole)
    np.add.at(charge_potential, right, f3 * left_projection + f5 * left_quadrupole)
    left_dipole = (
        (f3 * charges[right])[:, None] * displacement
        + f5[:, None]
        * (-3.0 * displacement * right_projection[:, None] + dipoles[right] * distance2[:, None])
    )
    right_dipole = (
        (-f3 * charges[left])[:, None] * displacement
        + f5[:, None]
        * (-3.0 * displacement * left_projection[:, None] + dipoles[left] * distance2[:, None])
    )
    np.add.at(dipole_potential, left, left_dipole)
    np.add.at(dipole_potential, right, right_dipole)
    outer = prepared.outer
    np.add.at(
        quadrupole_potential,
        left,
        (f5 * charges[right])[:, None, None] * outer,
    )
    np.add.at(
        quadrupole_potential,
        right,
        (f5 * charges[left])[:, None, None] * outer,
    )
    return energy, charge_potential, dipole_potential, quadrupole_potential


def anisotropic_moment_potential_direction(
    multipoles: AtomicMultipoles,
    multipole_direction: AtomicMultipoles,
    coordinates_bohr: np.ndarray,
    direction_bohr: np.ndarray,
    parameters: AnisotropicParameters,
    radius_direction_bohr: np.ndarray,
    *,
    geometry: AnisotropicGeometry | None = None,
    exponent_direction3: float = 0.0,
    exponent_direction5: float = 0.0,
    dipole_xc_direction: np.ndarray | None = None,
    quadrupole_xc_direction: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Differentiate anisotropic moment potentials along one geometry direction."""

    coordinates = np.asarray(coordinates_bohr, dtype=float)
    direction = np.asarray(direction_bohr, dtype=float)
    q = np.asarray(multipoles.charges, dtype=float)
    mu = np.asarray(multipoles.dipoles_bohr, dtype=float)
    theta = np.asarray(multipoles.quadrupoles_bohr2, dtype=float)
    qdot = np.asarray(multipole_direction.charges, dtype=float)
    mudot = np.asarray(multipole_direction.dipoles_bohr, dtype=float)
    thetadot = np.asarray(multipole_direction.quadrupoles_bohr2, dtype=float)
    if direction.shape != coordinates.shape:
        raise ValueError("anisotropic potential direction must have shape natoms x 3")
    prepared = geometry or prepare_anisotropic_geometry(coordinates, parameters)
    left, right = prepared.left, prepared.right
    displacement = prepared.displacement
    displacement_dot = direction[right] - direction[left]
    distance2 = prepared.distance2
    distance = np.sqrt(distance2)
    distance_dot = np.einsum("pi,pi->p", displacement, displacement_dot) / distance
    radius = 0.5 * (
        np.asarray(parameters.damping_radius_bohr)[left]
        + np.asarray(parameters.damping_radius_bohr)[right]
    )
    radius_dot = 0.5 * (
        np.asarray(radius_direction_bohr)[left]
        + np.asarray(radius_direction_bohr)[right]
    )

    def value_direction(power: int, exponent: float, exponent_dot: float):
        ratio = 6.0 * (radius / distance) ** exponent
        value = distance ** (-power) / (1.0 + ratio)
        value_dot = value * (
            (-power + exponent * ratio / (1.0 + ratio))
            * distance_dot
            / distance
            - exponent
            * ratio
            / (1.0 + ratio)
            * radius_dot
            / radius
            - ratio / (1.0 + ratio) * np.log(radius / distance) * exponent_dot
        )
        return value, value_dot

    f3, f3dot = value_direction(3, parameters.a3, float(exponent_direction3))
    f5, f5dot = value_direction(5, parameters.a5, float(exponent_direction5))
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
    r2dot = 2.0 * np.einsum("pi,pi->p", displacement, displacement_dot)
    qpot = np.zeros(len(coordinates), dtype=float)
    dipole_xc_dot = (np.zeros(len(coordinates)) if dipole_xc_direction is None
                     else np.asarray(dipole_xc_direction, dtype=float))
    quadrupole_xc_dot = (np.zeros(len(coordinates)) if quadrupole_xc_direction is None
                        else np.asarray(quadrupole_xc_direction, dtype=float))
    if dipole_xc_dot.shape != (len(coordinates),) or quadrupole_xc_dot.shape != (len(coordinates),):
        raise ValueError("AES onsite parameter direction has incompatible shape")
    mupot = (2.0 * np.asarray(parameters.dipole_xc)[:, None] * mudot
             + 2.0 * dipole_xc_dot[:, None] * mu)
    thetapot = (
        2.0 * np.asarray(parameters.quadrupole_xc)[:, None, None] * thetadot
        + 2.0 * quadrupole_xc_dot[:, None, None] * theta
    )
    np.add.at(
        qpot,
        left,
        -f3dot * rp - f3 * rpdot + f5dot * rtheta + f5 * rtheta_dot,
    )
    np.add.at(
        qpot,
        right,
        f3dot * lp + f3 * lpdot + f5dot * ltheta + f5 * ltheta_dot,
    )
    left_base = -3.0 * displacement * rp[:, None] + mu[right] * distance2[:, None]
    left_base_dot = (
        -3.0 * (displacement_dot * rp[:, None] + displacement * rpdot[:, None])
        + mudot[right] * distance2[:, None]
        + mu[right] * r2dot[:, None]
    )
    right_base = -3.0 * displacement * lp[:, None] + mu[left] * distance2[:, None]
    right_base_dot = (
        -3.0 * (displacement_dot * lp[:, None] + displacement * lpdot[:, None])
        + mudot[left] * distance2[:, None]
        + mu[left] * r2dot[:, None]
    )
    np.add.at(
        mupot,
        left,
        (f3dot * q[right] + f3 * qdot[right])[:, None] * displacement
        + (f3 * q[right])[:, None] * displacement_dot
        + f5dot[:, None] * left_base
        + f5[:, None] * left_base_dot,
    )
    np.add.at(
        mupot,
        right,
        (-f3dot * q[left] - f3 * qdot[left])[:, None] * displacement
        - (f3 * q[left])[:, None] * displacement_dot
        + f5dot[:, None] * right_base
        + f5[:, None] * right_base_dot,
    )
    outer = displacement[:, :, None] * displacement[:, None, :]
    outer_dot = (
        displacement_dot[:, :, None] * displacement[:, None, :]
        + displacement[:, :, None] * displacement_dot[:, None, :]
    )
    np.add.at(
        thetapot,
        left,
        (f5dot * q[right] + f5 * qdot[right])[:, None, None] * outer
        + (f5 * q[right])[:, None, None] * outer_dot,
    )
    np.add.at(
        thetapot,
        right,
        (f5dot * q[left] + f5 * qdot[left])[:, None, None] * outer
        + (f5 * q[left])[:, None, None] * outer_dot,
    )
    return qpot, mupot, thetapot


def anisotropic_fock(
    overlap: np.ndarray,
    dipole_integrals: np.ndarray,
    second_moment_integrals: np.ndarray,
    coordinates_bohr: np.ndarray,
    orbital_atom: Sequence[int],
    charge_potential: np.ndarray,
    dipole_potential: np.ndarray,
    quadrupole_potential: np.ndarray,
    *,
    operators: CumulativeMultipoleOperators | None = None,
) -> np.ndarray:
    """Contract CAMM moment potentials back to a symmetric AO Fock matrix."""

    overlap = np.asarray(overlap, dtype=float)
    dipole = np.asarray(dipole_integrals, dtype=float)
    second = np.asarray(second_moment_integrals, dtype=float)
    coordinates = np.asarray(coordinates_bohr, dtype=float)
    assignment = tuple(int(value) for value in orbital_atom)
    prepared = operators or prepare_cumulative_multipole_operators(
        overlap, dipole, second, coordinates, assignment
    )
    atom_index = prepared.atom_index
    size = len(atom_index)
    left, right = triangular_indices(size)
    dmu_l, _, dquad_l, dmu_r, _, dquad_r = _packed_endpoint_operators(
        prepared
    )

    def endpoint(atom, dmu, dquad):
        return (
            -np.asarray(charge_potential)[atom] * prepared.pair_overlap
            + np.einsum(
                "pa,pa->p", np.asarray(dipole_potential)[atom], dmu,
                optimize=False,
            )
            + np.einsum(
                "pac,pac->p", np.asarray(quadrupole_potential)[atom], dquad,
                optimize=False,
            )
        )

    values = 0.5 * (
        endpoint(atom_index[left], dmu_l, dquad_l)
        + endpoint(atom_index[right], dmu_r, dquad_r)
    )
    result = np.empty((size, size), dtype=float)
    result[left, right] = values
    result[right, left] = values
    return result


def anisotropic_fock_sparse(
    operators: SparseCumulativeMultipoleOperators,
    charge_potential: np.ndarray,
    dipole_potential: np.ndarray,
    quadrupole_potential: np.ndarray,
    *,
    size: int,
) -> SparseSymmetricMatrix:
    """Contract CAMM potentials on the existing AO pair list."""

    if (
        _gfn2_native is not None
        and hasattr(_gfn2_native, "anisotropic_fock_compact")
        and operators.dmu_left is None
    ):
        values = _gfn2_native.anisotropic_fock_compact(
            operators.rows, operators.columns,
            operators.left_atom, operators.right_atom,
            operators.overlap, operators.dipole, operators.second,
            operators.left_center, operators.right_center,
            charge_potential, dipole_potential, quadrupole_potential,
        )
        return SparseSymmetricMatrix(
            size, operators.rows.copy(), operators.columns.copy(),
            np.asarray(values, dtype=float),
        )

    def endpoint(atom, dmu, dquadrupole):
        return (
            -charge_potential[atom] * operators.overlap
            + np.einsum("pa,pa->p", dipole_potential[atom], dmu, optimize=False)
            + np.einsum(
                "pac,pac->p",
                quadrupole_potential[atom], dquadrupole,
                optimize=False,
            )
        )

    dmu_left, dmu_right, dquad_left, dquad_right = (
        operators.quadrupole_endpoint_arrays()
    )
    left = endpoint(operators.left_atom, dmu_left, dquad_left)
    right = endpoint(operators.right_atom, dmu_right, dquad_right)
    values = 0.5 * (left + right)
    diagonal = operators.rows == operators.columns
    values[diagonal] = left[diagonal]
    return SparseSymmetricMatrix(
        size, operators.rows.copy(), operators.columns.copy(), values
    )


def anisotropic_fock_derivatives_sparse(
    integrals: SparseAtomicMultipoleIntegrals,
    operators: SparseCumulativeMultipoleOperators | None,
    coordinates_bohr: np.ndarray,
    charge_potential: np.ndarray,
    dipole_potential: np.ndarray,
    quadrupole_potential: np.ndarray,
    *,
    orbital_atom: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair-endpoint derivative of the frozen-potential CAMM Fock term."""

    if _gfn2_native is not None and hasattr(
        _gfn2_native, "anisotropic_fock_derivatives"
    ):
        if operators is None:
            if orbital_atom is None:
                raise ValueError("orbital_atom is required without packed operators")
            assignment = np.asarray(orbital_atom, dtype=np.intp)
            left_atom = assignment[integrals.rows]
            right_atom = assignment[integrals.columns]
        else:
            left_atom = operators.left_atom
            right_atom = operators.right_atom
        left, right = _gfn2_native.anisotropic_fock_derivatives(
            integrals.overlap,
            integrals.dipole,
            integrals.left_derivatives,
            integrals.right_derivatives,
            left_atom,
            right_atom,
            np.asarray(coordinates_bohr, dtype=float),
            np.asarray(charge_potential, dtype=float),
            np.asarray(dipole_potential, dtype=float),
            np.asarray(quadrupole_potential, dtype=float),
        )
        return np.asarray(left, dtype=float), np.asarray(right, dtype=float)

    if operators is None:
        raise RuntimeError("packed operators are required by the Python fallback")

    coordinates = np.asarray(coordinates_bohr, dtype=float)
    identity = np.eye(3)

    endpoint_left = operators.left_atom
    endpoint_right = operators.right_atom

    def allocation(atom, endpoint_atom, derivatives):
        center = coordinates[atom]
        ds = derivatives[:, 0, :]
        dd = derivatives[:, 1:4, :]
        dsecond = derivatives[:, 4:, :].reshape(-1, 3, 3, 3)
        center_occurrences = (atom == endpoint_left).astype(float) + (
            atom == endpoint_right
        ).astype(float)
        shift_weight = np.divide(
            (atom == endpoint_atom).astype(float),
            center_occurrences,
            out=np.zeros_like(center_occurrences),
            where=center_occurrences > 0.0,
        )
        shift = shift_weight[:, None, None] * identity[None, :, :]
        dipole = integrals.dipole
        overlap = integrals.overlap
        dmu = (
            shift * overlap[:, None, None]
            + center[:, :, None] * ds[:, None, :]
            - dd
        )
        draw = np.empty_like(dsecond)
        for axis in range(3):
            delta = shift[:, :, axis]
            draw[:, :, :, axis] = (
                dd[:, :, axis, None] * center[:, None, :]
                + dipole[:, :, None] * delta[:, None, :]
                + delta[:, :, None] * dipole[:, None, :]
                + center[:, :, None] * dd[:, None, :, axis]
                - ds[:, axis, None, None] * center[:, :, None] * center[:, None, :]
                - overlap[:, None, None]
                * (delta[:, :, None] * center[:, None, :] + center[:, :, None] * delta[:, None, :])
                - dsecond[:, :, :, axis]
            )
        trace = np.trace(draw, axis1=1, axis2=2)
        dquadrupole = 1.5 * draw - 0.5 * identity[None, :, :, None] * trace[:, None, None, :]
        return (
            -charge_potential[atom, None] * ds
            + np.einsum("pi,pia->pa", dipole_potential[atom], dmu, optimize=False)
            + np.einsum("pij,pija->pa", quadrupole_potential[atom], dquadrupole, optimize=False)
        )

    left_left = allocation(operators.left_atom, operators.left_atom, integrals.left_derivatives)
    right_left = allocation(operators.right_atom, operators.left_atom, integrals.left_derivatives)
    left_right = allocation(operators.left_atom, operators.right_atom, integrals.right_derivatives)
    right_right = allocation(operators.right_atom, operators.right_atom, integrals.right_derivatives)
    left = 0.5 * (left_left + right_left)
    right = 0.5 * (left_right + right_right)
    diagonal = operators.rows == operators.columns
    left[diagonal] = left_left[diagonal]
    right[diagonal] = left_right[diagonal]
    return left, right


def anisotropic_fock_derivatives_directional_sparse(
    integrals: SparseAtomicMultipoleIntegrals,
    operators: SparseCumulativeMultipoleOperators,
    coordinates_bohr: np.ndarray,
    direction_bohr: np.ndarray,
    integral_left_directional_seconds: np.ndarray,
    integral_right_directional_seconds: np.ndarray,
    charge_potential: np.ndarray,
    dipole_potential: np.ndarray,
    quadrupole_potential: np.ndarray,
    charge_potential_direction: np.ndarray,
    dipole_potential_direction: np.ndarray,
    quadrupole_potential_direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Directional derivative of pair-endpoint CAMM Fock gradients."""

    coordinates = np.asarray(coordinates_bohr, dtype=float)
    direction = np.asarray(direction_bohr, dtype=float)
    left_second = np.asarray(integral_left_directional_seconds, dtype=float)
    right_second = np.asarray(integral_right_directional_seconds, dtype=float)
    expected = integrals.left_derivatives.shape
    if left_second.shape != expected or right_second.shape != expected:
        raise ValueError("CAMM directional AO seconds have an inconsistent shape")
    if _gfn2_native is not None and hasattr(
        _gfn2_native, "anisotropic_fock_derivatives_directional"
    ):
        return _gfn2_native.anisotropic_fock_derivatives_directional(
            integrals.overlap,
            integrals.dipole,
            integrals.left_derivatives,
            integrals.right_derivatives,
            operators.left_atom,
            operators.right_atom,
            coordinates,
            direction,
            left_second,
            right_second,
            charge_potential,
            dipole_potential,
            quadrupole_potential,
            charge_potential_direction,
            dipole_potential_direction,
            quadrupole_potential_direction,
        )
    left_atom = operators.left_atom
    right_atom = operators.right_atom
    overlap_dot = (
        np.einsum(
            "pck,pk->pc", integrals.left_derivatives[:, :1], direction[left_atom]
        )[:, 0]
        + np.einsum(
            "pck,pk->pc", integrals.right_derivatives[:, :1], direction[right_atom]
        )[:, 0]
    )
    dipole_dot = np.einsum(
        "pck,pk->pc", integrals.left_derivatives[:, 1:4], direction[left_atom]
    ) + np.einsum(
        "pck,pk->pc", integrals.right_derivatives[:, 1:4], direction[right_atom]
    )
    def allocation(atom, endpoint_atom, derivatives, derivative_dot):
        center = coordinates[atom]
        center_dot = direction[atom]
        ds = derivatives[:, 0, :]
        dsdot = derivative_dot[:, 0, :]
        dd = derivatives[:, 1:4, :]
        dddot = derivative_dot[:, 1:4, :]
        dsecond = derivatives[:, 4:, :].reshape(-1, 3, 3, 3)
        dseconddot = derivative_dot[:, 4:, :].reshape(-1, 3, 3, 3)
        center_occurrences = (atom == left_atom).astype(float) + (
            atom == right_atom
        ).astype(float)
        shift_weight = np.divide(
            (atom == endpoint_atom).astype(float),
            center_occurrences,
            out=np.zeros_like(center_occurrences),
            where=center_occurrences > 0.0,
        )
        overlap = integrals.overlap
        dipole = integrals.dipole
        p = dipole_potential[atom]
        p_dot = dipole_potential_direction[atom]

        def effective_quadrupole(potential):
            trace = np.trace(potential, axis1=1, axis2=2)
            result = 1.5 * potential.copy()
            result[:, range(3), range(3)] -= 0.5 * trace[:, None]
            return result

        def quadrupole_contraction(potential, directional):
            qc = np.einsum("pij,pj->pi", potential, center)
            cq = np.einsum("pi,pij->pj", center, potential)
            sum_center = qc + cq
            if not directional:
                return (
                    np.einsum("pia,pi->pa", dd, sum_center)
                    + shift_weight[:, None] * (
                        np.einsum("pi,pij->pj", dipole, potential)
                        + np.einsum("pij,pj->pi", potential, dipole)
                    )
                    - ds * np.einsum("pi,pi->p", center, qc)[:, None]
                    - overlap[:, None] * shift_weight[:, None] * sum_center
                    - np.einsum("pij,pija->pa", potential, dsecond)
                )
            qc_dot = np.einsum("pij,pj->pi", potential, center_dot)
            cq_dot = np.einsum("pi,pij->pj", center_dot, potential)
            return (
                np.einsum("pia,pi->pa", dddot, sum_center)
                + np.einsum("pia,pi->pa", dd, qc_dot + cq_dot)
                + shift_weight[:, None] * (
                    np.einsum("pi,pij->pj", dipole_dot, potential)
                    + np.einsum("pij,pj->pi", potential, dipole_dot)
                )
                - dsdot * np.einsum("pi,pi->p", center, qc)[:, None]
                - ds * (
                    np.einsum("pi,pi->p", center_dot, qc)
                    + np.einsum("pi,pi->p", center, qc_dot)
                )[:, None]
                - overlap_dot[:, None] * shift_weight[:, None] * sum_center
                - overlap[:, None] * shift_weight[:, None] * (qc_dot + cq_dot)
                - np.einsum("pij,pija->pa", potential, dseconddot)
            )

        q = effective_quadrupole(quadrupole_potential[atom])
        q_dot = effective_quadrupole(quadrupole_potential_direction[atom])
        dipole_part = (
            shift_weight[:, None] * (
                overlap[:, None] * p_dot + overlap_dot[:, None] * p
            )
            + ds * (
                np.einsum("pi,pi->p", p_dot, center)
                + np.einsum("pi,pi->p", p, center_dot)
            )[:, None]
            + dsdot * np.einsum("pi,pi->p", p, center)[:, None]
            - np.einsum("pi,pia->pa", p_dot, dd)
            - np.einsum("pi,pia->pa", p, dddot)
        )
        return (
            -charge_potential_direction[atom, None] * ds
            - charge_potential[atom, None] * dsdot
            + dipole_part
            + quadrupole_contraction(q_dot, False)
            + quadrupole_contraction(q, True)
        )

    left_left = allocation(left_atom, left_atom, integrals.left_derivatives, left_second)
    right_left = allocation(right_atom, left_atom, integrals.left_derivatives, left_second)
    left_right = allocation(left_atom, right_atom, integrals.right_derivatives, right_second)
    right_right = allocation(right_atom, right_atom, integrals.right_derivatives, right_second)
    left = 0.5 * (left_left + right_left)
    right = 0.5 * (left_right + right_right)
    diagonal = integrals.rows == integrals.columns
    left[diagonal] = left_left[diagonal]
    right[diagonal] = left_right[diagonal]
    return left, right


def anisotropic_fock_directional_sparse(
    integrals: SparseAtomicMultipoleIntegrals,
    operators: SparseCumulativeMultipoleOperators,
    coordinates_bohr: np.ndarray,
    direction_bohr: np.ndarray,
    charge_potential: np.ndarray,
    dipole_potential: np.ndarray,
    quadrupole_potential: np.ndarray,
    charge_potential_direction: np.ndarray,
    dipole_potential_direction: np.ndarray,
    quadrupole_potential_direction: np.ndarray,
) -> SparseSymmetricMatrix:
    """Differentiate the pair-packed CAMM Fock matrix in one direction."""

    coordinates = np.asarray(coordinates_bohr, dtype=float)
    direction = np.asarray(direction_bohr, dtype=float)
    left_atom = operators.left_atom
    right_atom = operators.right_atom
    left_direction = direction[left_atom]
    right_direction = direction[right_atom]
    overlap_dot = np.einsum(
        "pk,pk->p", integrals.left_derivatives[:, 0], left_direction
    ) + np.einsum(
        "pk,pk->p", integrals.right_derivatives[:, 0], right_direction
    )
    dipole_dot = np.einsum(
        "pck,pk->pc", integrals.left_derivatives[:, 1:4], left_direction
    ) + np.einsum(
        "pck,pk->pc", integrals.right_derivatives[:, 1:4], right_direction
    )
    second_dot = np.einsum(
        "pck,pk->pc",
        integrals.left_derivatives[:, 4:].reshape(-1, 9, 3),
        left_direction,
    ).reshape(-1, 3, 3) + np.einsum(
        "pck,pk->pc",
        integrals.right_derivatives[:, 4:].reshape(-1, 9, 3),
        right_direction,
    ).reshape(-1, 3, 3)
    def endpoint(atom, center_direction):
        center = coordinates[atom]
        dmu = center * integrals.overlap[:, None] - integrals.dipole
        dmu_dot = (
            center_direction * integrals.overlap[:, None]
            + center * overlap_dot[:, None]
            - dipole_dot
        )
        # Contract the traceless quadrupole with its potential before forming
        # the Cartesian products; no pair-by-3-by-3 raw tensors are needed.
        potential = 1.5 * quadrupole_potential[atom].copy()
        potential_direction = 1.5 * quadrupole_potential_direction[atom].copy()
        diagonal = np.arange(3)
        potential[:, diagonal, diagonal] -= 0.5 * np.trace(
            quadrupole_potential[atom], axis1=1, axis2=2
        )[:, None]
        potential_direction[:, diagonal, diagonal] -= 0.5 * np.trace(
            quadrupole_potential_direction[atom], axis1=1, axis2=2
        )[:, None]
        symmetric = potential + potential.swapaxes(1, 2)
        symmetric_direction = (
            potential_direction + potential_direction.swapaxes(1, 2)
        )
        sym_center = np.einsum("pij,pj->pi", symmetric, center)
        symdot_center = np.einsum("pij,pj->pi", symmetric_direction, center)
        sym_moving_center = np.einsum(
            "pij,pj->pi", symmetric, center_direction
        )
        quadrupole_contraction = (
            np.einsum("pi,pi->p", integrals.dipole, symdot_center)
            + np.einsum("pi,pi->p", dipole_dot, sym_center)
            + np.einsum("pi,pi->p", integrals.dipole, sym_moving_center)
            - 0.5 * integrals.overlap
            * np.einsum("pi,pi->p", center, symdot_center)
            - 0.5 * overlap_dot
            * np.einsum("pi,pi->p", center, sym_center)
            - integrals.overlap
            * np.einsum("pi,pi->p", center_direction, sym_center)
            - np.einsum("pij,pij->p", potential_direction, integrals.second)
            - np.einsum("pij,pij->p", potential, second_dot)
        )
        return (
            -charge_potential_direction[atom] * integrals.overlap
            - charge_potential[atom] * overlap_dot
            + np.einsum("pi,pi->p", dipole_potential_direction[atom], dmu)
            + np.einsum("pi,pi->p", dipole_potential[atom], dmu_dot)
            + quadrupole_contraction
        )

    left = endpoint(left_atom, left_direction)
    right = endpoint(right_atom, right_direction)
    values = 0.5 * (left + right)
    diagonal = integrals.rows == integrals.columns
    values[diagonal] = left[diagonal]
    return SparseSymmetricMatrix(
        integrals.size,
        integrals.rows.copy(),
        integrals.columns.copy(),
        values,
    )


__all__ = [
    "AnisotropicParameters",
    "AnisotropicGeometry",
    "AtomicMultipoles",
    "CumulativeMultipoleOperators",
    "SparseCumulativeMultipoleOperators",
    "anisotropic_energy_and_moment_potentials",
    "anisotropic_moment_potential_direction",
    "anisotropic_fock",
    "anisotropic_fock_sparse",
    "anisotropic_fock_derivatives_sparse",
    "anisotropic_fock_derivatives_directional_sparse",
    "anisotropic_fock_directional_sparse",
    "cumulative_atomic_multipole_directional_derivative_sparse",
    "cumulative_atomic_multipoles",
    "cumulative_atomic_multipoles_sparse",
    "dense_scc_observables",
    "dense_scc_fock",
    "prepare_cumulative_multipole_operators",
    "prepare_sparse_cumulative_multipole_operators",
    "prepare_anisotropic_geometry",
    "prepare_anisotropic_geometry_sparse",
]
