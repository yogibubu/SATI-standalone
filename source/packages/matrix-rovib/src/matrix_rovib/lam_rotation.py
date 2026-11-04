"""Variational coupling of DVR large-amplitude motion to overall rotation.

The module consumes a reduced kinetic surface on a direct-product grid.  The
surface may describe a periodic torsion, a confined inversion/puckering path,
or up to four coupled large-amplitude coordinates. Overall-rotation operators and
electric-dipole matrices are supplied by the native LOCK rotational kernel.

All Hamiltonian coefficients and energies are frequencies in MHz.  Coordinate
momenta are dimensionless operators ``p_mu = -i d/dq_mu`` in the units used by
the supplied grids.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from matrix_numerics import diagonalize_hermitian, resolve_native_backend

from .native_rotor import (
    MHZ_TO_KELVIN,
    NativeRotorParameters,
    _assign_asymmetric_top_block,
    angular_momentum_operators,
    centrifugal_distortion_operators,
    dipole_transition_matrix,
    normalize_representation,
    rotational_hamiltonian,
)
from .rovibrational_symmetry import LAMRotationSymmetry

try:
    from . import _lock_native
except ImportError:  # source-tree and unsupported-platform fallback
    _lock_native = None


LOCK_LAM_ROTATION_SURFACE_SCHEMA_V1 = "matrix.lock.lam_rotation_surface.v1"
LOCK_LAM_ROTATION_SURFACE_SCHEMA_V2 = "matrix.lock.lam_rotation_surface.v2"
LOCK_LAM_ROTATION_SURFACE_SCHEMA_V3 = "matrix.lock.lam_rotation_surface.v3"
LOCK_LAM_ROTATION_SURFACE_SCHEMA_V4 = "matrix.lock.lam_rotation_surface.v4"
LOCK_LAM_ROTATION_SURFACE_SCHEMA = "matrix.lock.lam_rotation_surface.v5"
LOCK_LAM_ROTATION_RESULT_SCHEMA = "matrix.lock.lam_rotation_result.v1"
LAM_KINETIC_CONVENTION = "matrix.lock.primitive_kinetic_flux.v3"
_PRIMITIVE_MATRIX_FREE_MIN_1D = 2048
_PRIMITIVE_MATRIX_FREE_MIN_MULTIDIMENSIONAL = 512
_COUPLED_MATRIX_FREE_MIN_DIMENSION = 1024
# Compatibility name for code written while the LOCK public name was selected.
LAM_ROTATION_SURFACE_SCHEMA = LOCK_LAM_ROTATION_SURFACE_SCHEMA


@dataclass(frozen=True)
class PermutationInversionSymmetry:
    """Abelian molecular-symmetry contract for rovibronic state filtering.

    Species are numbered as listed in ``species`` and species zero must be the
    identity.  The multiplication table permits cyclic and direct-product
    permutation-inversion groups without embedding group-specific rules in
    LOCK.  Rotational species are selected by ``(Ka mod 2, Kc mod 2)``.
    """

    species: tuple[str, ...]
    multiplication_table: tuple[tuple[int, ...], ...]
    nuclear_spin_weights: tuple[float, ...]
    lam_sector_species: tuple[int, ...]
    rotation_species_by_ka_kc_parity: tuple[int, int, int, int]
    required_total_species: int = 0
    filter_forbidden: bool = True

    def __post_init__(self) -> None:
        size = len(self.species)
        table = np.asarray(self.multiplication_table, dtype=int)
        weights = np.asarray(self.nuclear_spin_weights, dtype=float)
        if size < 1 or len(set(self.species)) != size or any(not x.strip() for x in self.species):
            raise ValueError("permutation-inversion species must be nonempty and unique")
        if table.shape != (size, size) or np.any(table < 0) or np.any(table >= size):
            raise ValueError("the species multiplication table must be square and closed")
        if not np.array_equal(table[0], np.arange(size)) or not np.array_equal(
            table[:, 0], np.arange(size)
        ):
            raise ValueError("permutation-inversion species zero must be the identity")
        if not np.array_equal(table, table.T):
            raise ValueError("LOCK currently requires an Abelian permutation-inversion group")
        for a in range(size):
            for b in range(size):
                for c in range(size):
                    if table[table[a, b], c] != table[a, table[b, c]]:
                        raise ValueError("the species multiplication table is not associative")
        if weights.shape != (size,) or np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
            raise ValueError("provide one finite nonnegative nuclear-spin weight per species")
        indices = self.lam_sector_species + self.rotation_species_by_ka_kc_parity
        if any(index < 0 or index >= size for index in indices):
            raise ValueError("a symmetry-species mapping index is outside the group")
        if not 0 <= self.required_total_species < size:
            raise ValueError("required_total_species is outside the group")
        if any(np.count_nonzero(table[a] == 0) != 1 for a in range(size)):
            raise ValueError("every symmetry species must have a unique inverse")

    def rovib_species(self, sector: int, Ka: int, Kc: int) -> int:
        lam = self.lam_sector_species[sector]
        rotation = self.rotation_species_by_ka_kc_parity[2 * (Ka & 1) + (Kc & 1)]
        return self.multiplication_table[lam][rotation]

    def spin_weight(self, rovib_species: int) -> float:
        row = self.multiplication_table[rovib_species]
        spin_species = row.index(self.required_total_species)
        return float(self.nuclear_spin_weights[spin_species])

    def to_dict(self) -> dict[str, object]:
        return {
            "species": list(self.species),
            "multiplication_table": [list(row) for row in self.multiplication_table],
            "nuclear_spin_weights": list(self.nuclear_spin_weights),
            "lam_sector_species": list(self.lam_sector_species),
            "rotation_species_by_ka_kc_parity": list(self.rotation_species_by_ka_kc_parity),
            "required_total_species": self.required_total_species,
            "filter_forbidden": self.filter_forbidden,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PermutationInversionSymmetry":
        return cls(
            species=tuple(str(x) for x in payload["species"]),  # type: ignore[arg-type]
            multiplication_table=tuple(
                tuple(int(x) for x in row)
                for row in payload["multiplication_table"]  # type: ignore[arg-type]
            ),
            nuclear_spin_weights=tuple(float(x) for x in payload["nuclear_spin_weights"]),  # type: ignore[arg-type]
            lam_sector_species=tuple(int(x) for x in payload["lam_sector_species"]),  # type: ignore[arg-type]
            rotation_species_by_ka_kc_parity=tuple(
                int(x)
                for x in payload["rotation_species_by_ka_kc_parity"]  # type: ignore[arg-type]
            ),  # type: ignore[arg-type]
            required_total_species=int(payload.get("required_total_species", 0)),
            filter_forbidden=bool(payload.get("filter_forbidden", True)),
        )


@dataclass(frozen=True)
class LAMRotationSurface:
    """Potential and reduced kinetic tensors on a one- to four-LAM grid.

    Array leading dimensions follow ``grid_shape``.  The trailing dimensions
    are ``(d,d)`` for ``F_MHz``, ``(3,d)`` for ``rho`` and ``(3,3)`` for
    ``R_MHz``.  ``geometric_potential_MHz`` is the optional scalar completion
    associated with the chosen flat DVR measure.
    """

    coordinates: tuple[np.ndarray, ...]
    potential_MHz: np.ndarray
    F_MHz: np.ndarray
    rho: np.ndarray
    R_MHz: np.ndarray
    periodic: tuple[bool, ...]
    labels: tuple[str, ...] = ()
    dipole_au: np.ndarray | None = None
    quartic_distortion_MHz: np.ndarray | None = None
    sextic_distortion_MHz: np.ndarray | None = None
    distortion_reduction: str = "S"
    geometric_potential_MHz: np.ndarray | None = None
    lam_quartic_kinetic_MHz: np.ndarray | None = None
    p2_rotational_metric_MHz: np.ndarray | None = None
    p_rotational_cubic_MHz: np.ndarray | None = None
    p_rotational_anticommutator_MHz: np.ndarray | None = None
    pi2_rotation_MHz: np.ndarray | None = None
    metadata: Mapping[str, object] | None = None
    torsional_symmetry_order: int = 1
    lam_symmetry_orders: tuple[int, ...] = ()
    nuclear_spin_weights: tuple[float, ...] = ()
    permutation_inversion_symmetry: PermutationInversionSymmetry | None = None
    spin_tensor_surfaces_MHz: Mapping[str, np.ndarray] | None = None

    def __post_init__(self) -> None:
        dimension = len(self.coordinates)
        if not 1 <= dimension <= 4:
            raise ValueError("LAM-rotation surfaces support one to four coordinates")
        if len(self.periodic) != dimension:
            raise ValueError("one boundary-condition flag is required per LAM coordinate")
        if self.labels and len(self.labels) != dimension:
            raise ValueError("one label is required per LAM coordinate")
        if (
            int(self.torsional_symmetry_order) != self.torsional_symmetry_order
            or self.torsional_symmetry_order < 1
        ):
            raise ValueError("torsional_symmetry_order must be a positive integer")
        grids = tuple(np.asarray(grid, dtype=float).reshape(-1) for grid in self.coordinates)
        shape = tuple(len(grid) for grid in grids)
        if any(size < 3 for size in shape):
            raise ValueError("each LAM grid needs at least three points")
        for grid, is_periodic in zip(grids, self.periodic, strict=True):
            differences = np.diff(grid)
            if np.any(differences <= 0.0) or not np.allclose(
                differences, differences[0], rtol=1.0e-9, atol=1.0e-12
            ):
                raise ValueError("LAM grids must be strictly increasing and uniform")
        symmetry_orders = self.symmetry_orders
        for axis, order in enumerate(symmetry_orders):
            if order > 1 and (not self.periodic[axis] or shape[axis] % order):
                raise ValueError(
                    "LAM symmetry requires each active periodic grid to be divisible by its order"
                )
        if self.nuclear_spin_weights:
            weights = np.asarray(self.nuclear_spin_weights, dtype=float)
            if (
                weights.shape != (self.symmetry_sector_count,)
                or np.any(weights < 0.0)
                or not np.all(np.isfinite(weights))
                or float(np.sum(weights)) <= 0.0
            ):
                raise ValueError("provide one finite nonnegative spin weight per LAM sector")
        if self.permutation_inversion_symmetry is not None:
            if (
                len(self.permutation_inversion_symmetry.lam_sector_species)
                != self.symmetry_sector_count
            ):
                raise ValueError("map every LAM sector to a permutation-inversion species")
            if self.nuclear_spin_weights:
                raise ValueError(
                    "use either torsional nuclear_spin_weights or permutation_inversion_symmetry"
                )
        if str(self.distortion_reduction).upper() not in {"A", "S"}:
            raise ValueError("Watson reduction must be A or S")
        expected = {
            "potential_MHz": (shape, self.potential_MHz),
            "F_MHz": (shape + (dimension, dimension), self.F_MHz),
            "rho": (shape + (3, dimension), self.rho),
            "R_MHz": (shape + (3, 3), self.R_MHz),
        }
        for name, (wanted, values) in expected.items():
            array = np.asarray(values)
            if array.shape != wanted or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be finite with shape {wanted}")
        if not np.allclose(self.F_MHz, np.swapaxes(self.F_MHz, -1, -2), atol=1.0e-9):
            raise ValueError("F_MHz must be symmetric")
        if not np.allclose(self.R_MHz, np.swapaxes(self.R_MHz, -1, -2), atol=1.0e-9):
            raise ValueError("R_MHz must be symmetric")
        if np.min(np.linalg.eigvalsh(np.asarray(self.F_MHz))) <= 0.0:
            raise ValueError("F_MHz must be positive definite at every grid point")
        if np.min(np.linalg.eigvalsh(np.asarray(self.R_MHz))) <= 0.0:
            raise ValueError("R_MHz must be positive definite at every grid point")
        for name, values, trailing in (
            ("dipole_au", self.dipole_au, (3,)),
            ("quartic_distortion_MHz", self.quartic_distortion_MHz, (5,)),
            ("sextic_distortion_MHz", self.sextic_distortion_MHz, (7,)),
            ("geometric_potential_MHz", self.geometric_potential_MHz, ()),
            ("lam_quartic_kinetic_MHz", self.lam_quartic_kinetic_MHz, (dimension,)),
            (
                "p2_rotational_metric_MHz",
                self.p2_rotational_metric_MHz,
                (dimension, 3, 3),
            ),
            ("p_rotational_cubic_MHz", self.p_rotational_cubic_MHz, (dimension, 3)),
            (
                "p_rotational_anticommutator_MHz",
                self.p_rotational_anticommutator_MHz,
                (dimension, 3, 3),
            ),
        ):
            if values is not None:
                array = np.asarray(values)
                if array.shape != shape + trailing or not np.all(np.isfinite(array)):
                    raise ValueError(f"{name} must be finite with shape {shape + trailing}")
        if self.p2_rotational_metric_MHz is not None and not np.allclose(
            self.p2_rotational_metric_MHz,
            np.swapaxes(self.p2_rotational_metric_MHz, -1, -2),
            atol=1.0e-12,
        ):
            raise ValueError("p2_rotational_metric_MHz must be symmetric in rotational axes")
        if self.pi2_rotation_MHz is not None:
            coefficients = np.asarray(self.pi2_rotation_MHz)
            if (
                coefficients.shape != (3,)
                or not np.isrealobj(coefficients)
                or not np.all(np.isfinite(coefficients))
            ):
                raise ValueError("pi2_rotation_MHz must contain finite J, K, and minus coefficients")
        spin_surfaces = dict(self.spin_tensor_surfaces_MHz or {})
        tensor_suffixes = (
            "anisotropic_spin_rotation_tensor_MHz",
            "zero_field_splitting_tensor_MHz",
            "quadrupole_tensor_MHz",
            "electron_nuclear_dipolar_tensor_MHz",
            "dipolar_tensor_MHz",
        )
        scalar_suffixes = (
            "isotropic_spin_rotation_MHz",
            "fermi_contact_MHz",
            "isotropic_nuclear_spin_rotation_MHz",
            "isotropic_MHz",
        )
        for name, values in spin_surfaces.items():
            if not str(name).strip():
                raise ValueError("spin tensor surface names must be nonempty")
            array = np.asarray(values, dtype=float)
            expected_shape = (
                shape + (6,)
                if str(name).endswith(tensor_suffixes)
                else shape + (dimension, 3)
                if str(name).endswith("spin_lam_momentum_vectors_MHz")
                else shape
                if str(name).endswith(scalar_suffixes)
                else None
            )
            if expected_shape is None:
                raise ValueError(f"unknown spin coupling surface name {name!r}")
            if array.shape != expected_shape or not np.all(np.isfinite(array)):
                raise ValueError(
                    f"spin coupling surface {name!r} must be finite with shape {expected_shape}"
                )
        object.__setattr__(self, "spin_tensor_surfaces_MHz", spin_surfaces)

    @property
    def dimension(self) -> int:
        return len(self.coordinates)

    @property
    def grid_shape(self) -> tuple[int, ...]:
        return tuple(len(np.asarray(grid)) for grid in self.coordinates)

    @property
    def symmetry_orders(self) -> tuple[int, ...]:
        if self.lam_symmetry_orders:
            if any(int(value) != value for value in self.lam_symmetry_orders):
                raise ValueError("LAM symmetry orders must be integers")
            orders = tuple(int(value) for value in self.lam_symmetry_orders)
            if len(orders) != self.dimension or any(value < 1 for value in orders):
                raise ValueError("provide one positive symmetry order per LAM coordinate")
            return orders
        return (int(self.torsional_symmetry_order),) + (1,) * (self.dimension - 1)

    @property
    def symmetry_sector_count(self) -> int:
        return int(np.prod(self.symmetry_orders))


@dataclass(frozen=True)
class LAMBasis:
    energies_MHz: np.ndarray
    vectors: np.ndarray
    momentum_operators: tuple[np.ndarray, ...]
    hamiltonian_MHz: np.ndarray | None
    diagonalizer_backend: str
    symmetry_sectors: tuple[int, ...] = ()
    sector_state_indices: tuple[int, ...] = ()
    primitive_representation: str = "dvr"
    primitive_size: int | None = None
    fourier_cutoffs: tuple[int | None, ...] | None = None
    diagnostics: Mapping[str, object] | None = None
    spectral_space: object | None = None
    spectral_coefficients: np.ndarray | None = None


@dataclass(frozen=True)
class LAMRotationLevel:
    J: int
    index: int
    energy_MHz: float
    energy_above_ground_MHz: float
    lam_parent: int
    Ka: int
    Kc: int
    parent_purity: float
    symmetry_sector: int = 0
    lam_symmetry: str = "A"
    spin_weight: float = 1.0
    torsional_state: int = 0
    sector_index: int = 0
    rovibrational_irrep: str | None = None


@dataclass(frozen=True)
class LAMRotationBlock:
    J: int
    energies_MHz: np.ndarray
    vectors: np.ndarray
    levels: tuple[LAMRotationLevel, ...]


@dataclass(frozen=True)
class LAMRotationResult:
    basis: LAMBasis
    blocks: Mapping[int, LAMRotationBlock]
    representation: str
    ground_energy_MHz: float
    hermiticity_error_MHz: float
    distortion_model: str | None = None
    assembly_backends: tuple[str, ...] = ()
    diagonalizer_backends: tuple[str, ...] = ()
    operator_manifest_digest: str | None = None
    operator_parameters_MHz: Mapping[str, float] | None = None
    k_contractions: Mapping[int, object] | None = None


@dataclass(frozen=True)
class LAMRotationWorkspace:
    """Validated reusable contractions for a fixed geometric DVR core."""

    basis: LAMBasis
    contracted: Mapping[str, np.ndarray]
    operator_contraction: object | None
    surface_digest: str
    operator_manifest_digest: str | None
    requested_state_count: int
    include_geometric_potential: bool
    primitive_basis: str = "auto"
    fourier_cutoffs: tuple[int | None, ...] | None = None


def _lam_contraction_digest(surface):
    import hashlib

    digest = hashlib.sha256()
    for coordinate in surface.coordinates:
        array = np.ascontiguousarray(coordinate)
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    digest.update(repr((surface.periodic, surface.symmetry_orders)).encode())
    for name in (
        "potential_MHz", "F_MHz", "rho", "R_MHz", "geometric_potential_MHz",
        "lam_quartic_kinetic_MHz", "p2_rotational_metric_MHz",
        "p_rotational_cubic_MHz", "p_rotational_anticommutator_MHz",
        "quartic_distortion_MHz", "sextic_distortion_MHz", "pi2_rotation_MHz",
    ):
        digest.update(name.encode())
        value = getattr(surface, name)
        if value is None:
            digest.update(b"none")
        else:
            array = np.ascontiguousarray(value)
            digest.update(str((array.shape, array.dtype)).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def prepare_lam_rotation_workspace(
    surface, *, lam_state_count=12, operator_manifest=None,
    include_geometric_potential=True, diagonalizer_backend=None,
    primitive_basis="auto", fourier_cutoffs=None,
):
    """Contract the fixed DVR core and generic words once for a nonlinear fit."""
    from types import MappingProxyType

    basis = solve_lam_basis(
        surface, state_count=lam_state_count,
        include_geometric_potential=include_geometric_potential,
        diagonalizer_backend=diagonalizer_backend,
        primitive_basis=primitive_basis, fourier_cutoffs=fourier_cutoffs,
    )
    basis.vectors.setflags(write=False)
    basis.energies_MHz.setflags(write=False)
    if basis.spectral_coefficients is not None:
        basis.spectral_coefficients.setflags(write=False)
    for matrix in basis.momentum_operators:
        matrix.setflags(write=False)
    contracted = _contracted_lam_operators(surface, basis)
    for matrix in contracted.values():
        matrix.setflags(write=False)
    operators = None
    if operator_manifest is not None:
        from .general_lam_operator import contract_general_lam_operators

        operators = contract_general_lam_operators(surface, basis, operator_manifest)
    return LAMRotationWorkspace(
        basis, MappingProxyType(contracted), operators,
        _lam_contraction_digest(surface),
        None if operators is None else operators.manifest.digest,
        int(lam_state_count), bool(include_geometric_potential),
        primitive_basis, None if fourier_cutoffs is None else tuple(fourier_cutoffs),
    )


def primitive_momentum_operators(surface: LAMRotationSurface) -> tuple[np.ndarray, ...]:
    """Return full-grid Hermitian momentum operators for the surface."""
    one_dimensional = tuple(
        _fourier_momentum(grid) if periodic else _confined_momentum(grid)
        for grid, periodic in zip(surface.coordinates, surface.periodic, strict=True)
    )
    shape = surface.grid_shape
    operators = []
    for axis, local in enumerate(one_dimensional):
        left = int(np.prod(shape[:axis]))
        right = int(np.prod(shape[axis + 1 :]))
        operators.append(np.kron(np.eye(left), np.kron(local, np.eye(right))))
    return tuple(operators)


def solve_lam_basis(
    surface: LAMRotationSurface,
    *,
    state_count: int,
    include_geometric_potential: bool = True,
    diagonalizer_backend: str | None = None,
    primitive_basis: str = "auto",
    fourier_cutoffs: Sequence[int | None] | None = None,
) -> LAMBasis:
    """Solve J=0 in Fourier/hybrid coordinates, or explicitly in DVR.

    Periodic axes use Fourier harmonics automatically. Independent harmonic
    cutoffs select a subspace without reducing the surface quadrature. Returned
    vectors remain in the grid representation for all downstream consumers.
    """
    if primitive_basis not in {"auto", "fourier", "dvr"}:
        raise ValueError("primitive_basis must be auto, fourier or dvr")
    if diagonalizer_backend == "auto":
        diagonalizer_backend = None
    if fourier_cutoffs is not None and primitive_basis == "dvr":
        raise ValueError("Fourier cutoffs require a Fourier primitive basis")
    grid_size = int(np.prod(surface.grid_shape))
    if not 1 <= state_count <= grid_size:
        raise ValueError("state_count must fit the primitive DVR grid")
    diagonal_count = min(
        grid_size,
        state_count + max(0, surface.symmetry_sector_count - 1),
    )
    potential = np.asarray(surface.potential_MHz, dtype=float).reshape(-1).copy()
    if include_geometric_potential and surface.geometric_potential_MHz is not None:
        potential += np.asarray(surface.geometric_potential_MHz, dtype=float).reshape(-1)
    if primitive_basis == "fourier" or (primitive_basis == "auto" and any(surface.periodic)):
        from .lam_fourier import solve_fourier_lam_basis

        return solve_fourier_lam_basis(
            surface, state_count=state_count, potential=potential,
            cutoffs=fourier_cutoffs, diagonalizer_backend=diagonalizer_backend,
        )
    if fourier_cutoffs is not None:
        raise ValueError("Fourier cutoffs require a periodic coordinate")
    iterative = (
        diagonalizer_backend == "arpack"
        or (
            diagonalizer_backend is None
            and grid_size
            >= (
                _PRIMITIVE_MATRIX_FREE_MIN_1D
                if surface.dimension == 1
                else _PRIMITIVE_MATRIX_FREE_MIN_MULTIDIMENSIONAL
            )
            and diagonal_count <= min(128, grid_size // 4)
        )
    ) and diagonal_count < grid_size - 1
    if iterative:
        from scipy.sparse.linalg import LinearOperator, eigsh

        operator = LinearOperator(
            (grid_size, grid_size),
            matvec=lambda vector: _apply_lam_hamiltonian(surface, vector, potential=potential),
            matmat=lambda vectors: _apply_lam_hamiltonian(surface, vectors, potential=potential),
            dtype=np.complex128,
        )
        energies, vectors = eigsh(
            operator,
            k=diagonal_count,
            which="SA",
            tol=1.0e-11,
            maxiter=max(1000, 20 * grid_size),
            v0=np.linspace(1.0, 2.0, grid_size, dtype=complex),
        )
        order = np.argsort(energies)
        energies = np.asarray(energies[order]).real
        vectors = np.asarray(vectors[:, order])
        vectors = _orthonormalize_degenerate_subspaces(energies, vectors)
        from ._iterative_roots import complete_lowest_ritz_subspace

        energies, vectors = complete_lowest_ritz_subspace(
            lambda values: _apply_lam_hamiltonian(surface, values, potential=potential),
            energies, vectors, spectral_bound=_lam_kinetic_norm_bound(surface, potential),
        )
        momentum: tuple[np.ndarray, ...] = ()
        hamiltonian = None
        selected_diagonalizer = "scipy-arpack"
    else:
        momentum = primitive_momentum_operators(surface)
        # Assemble the same primitive action used by the iterative solver
        # and its coefficient response, including full confined metrics.
        hamiltonian = _apply_lam_hamiltonian(surface, np.eye(grid_size), potential=potential)
        hamiltonian = _hermitian(hamiltonian)
        selected_backend = None if diagonalizer_backend == "arpack" else diagonalizer_backend
        diagonalized = diagonalize_hermitian(
            hamiltonian,
            backend=selected_backend,
            subset_by_index=(0, diagonal_count - 1) if diagonal_count < grid_size else None,
        )
        if diagonalized.eigenvectors is None:
            raise RuntimeError("LOCK diagonalizer did not return eigenvectors")
        energies = diagonalized.eigenvalues[:diagonal_count].real
        vectors = diagonalized.eigenvectors[:, :diagonal_count]
        selected_diagonalizer = diagonalized.backend
    retained = state_count
    energy_scale = float(np.max(np.abs(energies), initial=0.0))
    degeneracy_tolerance = max(1.0e-6, 256.0 * float(np.spacing(energy_scale)))
    while (
        retained < len(energies)
        and abs(energies[retained] - energies[retained - 1]) <= degeneracy_tolerance
    ):
        retained += 1
    energies = energies[:retained]
    vectors = vectors[:, :retained]
    vectors, sectors = _symmetry_adapt_primitive_basis(surface, energies, vectors)
    sector_counts: dict[int, int] = {}
    sector_state_indices = []
    for sector in sectors:
        sector_state_indices.append(sector_counts.get(sector, 0))
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
    return LAMBasis(
        energies_MHz=energies,
        vectors=vectors,
        momentum_operators=momentum,
        hamiltonian_MHz=hamiltonian,
        diagonalizer_backend=selected_diagonalizer,
        symmetry_sectors=sectors,
        sector_state_indices=tuple(sector_state_indices),
        primitive_size=grid_size,
    )


def solve_lam_rotation(
    surface: LAMRotationSurface,
    *,
    j_values: Sequence[int] | range,
    lam_state_count: int = 12,
    representation: str = "Ir",
    include_geometric_potential: bool = True,
    centrifugal_distortion: NativeRotorParameters | None = None,
    level_count: int | None = None,
    assembly_backend: str | None = None,
    diagonalizer_backend: str | None = None,
    rovibrational_symmetry: LAMRotationSymmetry | None = None,
    operator_manifest: Mapping[str, object] | object | None = None,
    operator_parameters_MHz: Mapping[str, float] | None = None,
    lam_basis: LAMBasis | None = None,
    workspace: LAMRotationWorkspace | None = None,
    primitive_basis: str = "auto",
    fourier_cutoffs: Sequence[int | None] | None = None,
    k_contraction_states: int | None = None,
    k_contractions: Mapping[int, object] | None = None,
    workers: int = 1,
    reproject_lam_core: bool = False,
) -> LAMRotationResult:
    """Diagonalize contracted LAM--rotation Hamiltonians for selected J values.

    Independent J blocks are evaluated concurrently when ``workers > 1``.
    The LAM contraction is built once and shared read-only by the workers.
    ``reproject_lam_core`` evaluates the entire current J=0 operator in an
    explicitly supplied fixed variational span, including its off-diagonals.
    The stored LAM energies then label the reference contraction only.
    """
    values = tuple(sorted(set(int(J) for J in j_values)))
    if not values or values[0] < 0:
        raise ValueError("at least one nonnegative J value is required")
    if isinstance(workers, bool) or int(workers) != workers or workers < 1:
        raise ValueError("workers must be a positive integer")
    workers = int(workers)
    if diagonalizer_backend == "auto":
        diagonalizer_backend = None
    if k_contractions is not None and k_contraction_states is None:
        raise ValueError("frozen J,K contractions require k_contraction_states")
    if k_contraction_states is not None and (
        isinstance(k_contraction_states, (bool, np.bool_))
        or not isinstance(k_contraction_states, (int, np.integer))
        or k_contraction_states < 1
    ):
        raise ValueError("k_contraction_states must be a positive integer")
    normalized_operator_manifest = None
    normalized_operator_parameters = None
    if operator_manifest is not None:
        from .general_lam_operator import normalize_general_lam_operator_manifest

        normalized_operator_manifest = normalize_general_lam_operator_manifest(
            operator_manifest
        )
        normalized_operator_parameters = normalized_operator_manifest.parameters_with(
            operator_parameters_MHz
        )
    elif operator_parameters_MHz:
        raise ValueError("operator parameter overrides require an operator manifest")
    if workspace is not None:
        expected = None if normalized_operator_manifest is None else normalized_operator_manifest.digest
        if (
            workspace.surface_digest != _lam_contraction_digest(surface)
            or workspace.operator_manifest_digest != expected
            or workspace.requested_state_count != lam_state_count
            or workspace.include_geometric_potential != include_geometric_potential
            or workspace.primitive_basis != primitive_basis
            or workspace.fourier_cutoffs != (None if fourier_cutoffs is None else tuple(fourier_cutoffs))
            or (lam_basis is not None and lam_basis is not workspace.basis)
        ):
            raise ValueError("LAM workspace does not match the geometric core, manifest or contraction")
        lam_basis = workspace.basis
    if centrifugal_distortion is not None and normalize_representation(
        centrifugal_distortion.representation
    ) != normalize_representation(representation):
        raise ValueError("the centrifugal-distortion and LAM representations must agree")
    from .body_frame import frame_rotations
    transport = frame_rotations(surface)
    pi_payload = dict(surface.metadata or {}).get("oracle_pi_grid_action")
    if pi_payload is not None:
        from matrix_chem import PermutationInversionGridAction
        from .pi_symmetry import PIGridLAMRotationSymmetry

        if rovibrational_symmetry is not None:
            raise ValueError("surface already carries ORACLE PI symmetry; do not override it")
        if surface.symmetry_sector_count != 1 or surface.permutation_inversion_symmetry is not None:
            raise ValueError("ORACLE PI action cannot be combined with legacy torsional-sector symmetry")
        grid_action = PermutationInversionGridAction.from_dict(pi_payload)
        grid_action.validate_scalar_field(surface.potential_MHz)
        if transport is not None:
            # Pull the action back into the transported contracted basis.
            # H'=W H W† and B'=W B imply B'† H' B'=B† H B exactly.
            grid_action = grid_action.reframe(transport.swapaxes(-1, -2))
        rovibrational_symmetry = PIGridLAMRotationSymmetry(grid_action, representation)
    # A fit of rotation-only fields or effective operator coefficients leaves
    # the J=0 LAM eigenproblem unchanged. Its immutable basis may be reused.
    # Callers supplying it must keep potential and kinetic fields unchanged.
    basis = lam_basis
    if basis is None:
        basis = solve_lam_basis(
            surface,
            state_count=lam_state_count,
            include_geometric_potential=include_geometric_potential,
            diagonalizer_backend=diagonalizer_backend,
            primitive_basis=primitive_basis, fourier_cutoffs=fourier_cutoffs,
        )
    elif (
        basis.vectors.shape[0] != math.prod(surface.grid_shape)
        or len(basis.energies_MHz) < lam_state_count
    ):
        raise ValueError("reused LAM basis is incompatible with the requested contraction")
    core_correction = None
    if reproject_lam_core:
        if lam_basis is None or workspace is not None:
            raise ValueError("fixed variational projection requires an explicit basis and no frozen operator workspace")
        potential = np.asarray(surface.potential_MHz).reshape(-1).copy()
        if include_geometric_potential and surface.geometric_potential_MHz is not None:
            potential += np.asarray(surface.geometric_potential_MHz).reshape(-1)
        projected_core = _hermitian(_project_lam_values(basis, _apply_lam_hamiltonian(
            surface, basis.vectors, potential=potential),
            preserve_sectors=surface.symmetry_sector_count > 1))
        core_correction = projected_core - np.diag(basis.energies_MHz)
    contracted = (
        workspace.contracted if workspace is not None
        else _contracted_lam_operators(surface, basis)
    )
    general_operator_contraction = None if workspace is None else workspace.operator_contraction
    if normalized_operator_manifest is not None and general_operator_contraction is None:
        from .general_lam_operator import contract_general_lam_operators

        general_operator_contraction = contract_general_lam_operators(
            surface, basis, normalized_operator_manifest
        )
    pi2_projected = None
    if surface.pi2_rotation_MHz is not None and np.any(surface.pi2_rotation_MHz):
        from .pi2_rotation import project_covariant_momentum_matrices

        pi2_projected = project_covariant_momentum_matrices(
            basis.vectors,
            np.asarray(surface.rho).reshape((-1, 3, surface.dimension)),
            lambda mu, values: (
                basis.momentum_operators[mu] @ values
                if basis.momentum_operators
                else _apply_momentum(surface, mu, values)
            ),
        )
    reference_index = int(np.argmin(np.asarray(surface.potential_MHz).reshape(-1)))
    reference_R = np.asarray(surface.R_MHz).reshape((-1, 3, 3))[reference_index]
    blocks: dict[int, LAMRotationBlock] = {}
    hermiticity_error = 0.0
    raw: dict[
        int,
        tuple[
            np.ndarray,
            np.ndarray,
            list[tuple[int, int, int, float]],
            tuple[int, ...],
            tuple[str | None, ...],
        ],
    ] = {}
    assembly_backends: list[str] = []
    diagonalizer_backends: list[str] = [basis.diagonalizer_backend]

    def solve_j_block(J: int):
        k_contraction = None
        rovib_labels = None
        preassigned_symmetry_sectors = None
        rot = _rotational_operators(J, representation)
        nrot = 2 * J + 1
        distortion = None
        if centrifugal_distortion is not None:
            rigid_reference = sum(
                constant * operator @ operator
                for constant, operator in zip(
                    (
                        centrifugal_distortion.A_MHz,
                        centrifugal_distortion.B_MHz,
                        centrifugal_distortion.C_MHz,
                    ),
                    rot,
                    strict=True,
                )
            )
            distortion = rotational_hamiltonian(centrifugal_distortion, J) - rigid_reference
        coordinate_terms: list[tuple[np.ndarray, np.ndarray]] = []
        if core_correction is not None:
            coordinate_terms.append((core_correction, np.eye(nrot)))
        if normalized_operator_manifest is not None:
            from .general_lam_operator import compile_general_lam_factors

            coordinate_terms.extend(
                compile_general_lam_factors(
                    surface,
                    basis,
                    J,
                    general_operator_contraction,
                    representation=representation,
                    parameters_MHz=normalized_operator_parameters,
                )
            )
        if contracted["distortion"].shape[0]:
            distortion_operators = centrifugal_distortion_operators(
                J, surface.distortion_reduction, representation
            )
            coordinate_terms.extend(
                zip(
                    contracted["distortion"],
                    distortion_operators[:len(contracted["distortion"])],
                    strict=True,
                )
            )
        if contracted["p2_rotational"].shape[0]:
            for mu in range(surface.dimension):
                for a in range(3):
                    for b in range(3):
                        coefficient = contracted["p2_rotational"][mu, a, b]
                        if not np.any(coefficient):
                            continue
                        operator = 0.5 * (rot[a] @ rot[b] + rot[b] @ rot[a])
                        coordinate_terms.append((coefficient, operator))
        if contracted["p_rotational_cubic"].shape[0]:
            total_rotation2 = sum(operator @ operator for operator in rot)
            for mu in range(surface.dimension):
                for a in range(3):
                    coefficient = contracted["p_rotational_cubic"][mu, a]
                    if not np.any(coefficient):
                        continue
                    coordinate_terms.append((coefficient, rot[a] @ total_rotation2))
        if contracted["p_rotational_anticommutator"].shape[0]:
            for mu in range(surface.dimension):
                for squared_axis in range(3):
                    square = rot[squared_axis] @ rot[squared_axis]
                    for linear_axis in range(3):
                        coefficient = contracted["p_rotational_anticommutator"][
                            mu, squared_axis, linear_axis
                        ]
                        if not np.any(coefficient):
                            continue
                        operator = 0.5 * (
                            square @ rot[linear_axis] + rot[linear_axis] @ square
                        )
                        coordinate_terms.append((coefficient, operator))
        if pi2_projected is not None:
            from .pi2_rotation import pi2_rotation_factors

            families, _ = pi2_rotation_factors(*pi2_projected, rot, np.ones(3))
            for coefficient, factors in zip(surface.pi2_rotation_MHz, families, strict=True):
                if coefficient:
                    coordinate_terms.extend(
                        (coefficient * scale * lam, operator)
                        for scale, lam, operator in factors
                    )
        coordinate_distortion = tuple(coordinate_terms) or None
        dimension = len(basis.energies_MHz) * nrot
        if level_count is not None and level_count < 1:
            raise ValueError("level_count must be positive")
        retained = dimension if level_count is None else min(level_count, dimension)
        diagonal_count = min(
            dimension,
            retained + max(0, surface.symmetry_sector_count - 1),
        )
        iterative = (
            rovibrational_symmetry is None
            and surface.symmetry_sector_count == 1
            and (
                diagonalizer_backend == "arpack"
                or (
                    diagonalizer_backend is None
                    and dimension >= _COUPLED_MATRIX_FREE_MIN_DIMENSION
                    and diagonal_count <= dimension // 3
                )
            )
            and diagonal_count < dimension - 1
        )
        if k_contraction_states is not None:
            from .lam_k_contraction import solve_k_contracted

            factors = [(np.diag(basis.energies_MHz), np.eye(nrot))]
            factors.extend((contracted["linear"][a], rot[a]) for a in range(3))
            factors.extend(
                (contracted["quadratic"][a, b], .5 * (rot[a] @ rot[b] + rot[b] @ rot[a]))
                for a in range(3) for b in range(3)
            )
            if distortion is not None:
                factors.append((np.eye(len(basis.energies_MHz)), distortion))
            factors.extend(coordinate_distortion or ())
            state_action = (None if rovibrational_symmetry is None else
                            rovibrational_symmetry.contracted_action(basis.vectors, J))
            if k_contractions is not None and J not in k_contractions:
                raise ValueError(f"no frozen J,K contraction supplied for J={J}")
            (energies, vectors, preassigned_symmetry_sectors, rovib_labels,
             selected_diagonalizer, error, k_contraction) = solve_k_contracted(
                basis, J, factors, states_per_k=k_contraction_states,
                level_count=level_count, backend=diagonalizer_backend,
                contraction=None if k_contractions is None else k_contractions[J],
                state_action=state_action,
            )
            retained = min(retained, len(energies))
            selected_assembly = "factorized-jk-contraction"
        elif iterative:
            energies, vectors = _diagonalize_coupled_matrix_free(
                basis.energies_MHz,
                contracted["linear"],
                contracted["quadratic"],
                rot,
                distortion,
                coordinate_distortion,
                level_count=diagonal_count,
            )
            selected_assembly = "matrix-free"
            selected_diagonalizer = "scipy-arpack"
            error = _coupled_coefficient_hermiticity_error(
                contracted["linear"],
                contracted["quadratic"],
                rot,
                distortion,
                coordinate_distortion,
            )
        elif surface.symmetry_sector_count > 1 and (
            diagonal_count < dimension or rovibrational_symmetry is None
        ):
            # Assemble exact invariant product-symmetry blocks directly.  This
            # avoids ever materializing the much larger dense coupled matrix.
            sector_energies = []
            sector_vectors = []
            sector_labels = []
            sector_ids = []
            sector_matrices = []
            sector_coupled_indices = []
            sector_backends = []
            assembly_backends_for_sectors = []
            error = 0.0
            basis_sectors = np.asarray(basis.symmetry_sectors)
            for sector in range(surface.symmetry_sector_count):
                lam_indices = np.flatnonzero(basis_sectors == sector)
                if not len(lam_indices):
                    continue
                linear = contracted["linear"][:, lam_indices][:, :, lam_indices]
                quadratic = contracted["quadratic"][:, :, lam_indices][:, :, :, lam_indices]
                local_coordinate_distortion = None
                if coordinate_distortion is not None:
                    local_coordinate_distortion = tuple(
                        (coefficient[np.ix_(lam_indices, lam_indices)], operator)
                        for coefficient, operator in coordinate_distortion
                    )
                block, block_assembly = _assemble_coupled_hamiltonian(
                    basis.energies_MHz[lam_indices],
                    linear,
                    quadratic,
                    rot,
                    distortion,
                    local_coordinate_distortion,
                    backend=assembly_backend,
                )
                error = max(
                    error,
                    float(np.max(np.abs(block - block.conj().T), initial=0.0)),
                )
                block = _hermitian(block)
                roots = len(block) if diagonal_count == dimension else min(
                    diagonal_count,
                    len(block),
                    math.ceil(diagonal_count / surface.symmetry_sector_count) + 2,
                )
                selected_backend = (
                    None if diagonalizer_backend == "arpack" else diagonalizer_backend
                )
                diagonalized = diagonalize_hermitian(
                    block,
                    backend=selected_backend,
                    subset_by_index=(0, roots - 1) if roots < len(block) else None,
                )
                if diagonalized.eigenvectors is None:
                    raise RuntimeError("LOCK diagonalizer did not return eigenvectors")
                coupled_indices = np.concatenate(
                    [np.arange(index * nrot, (index + 1) * nrot) for index in lam_indices]
                )
                embedded = np.zeros(
                    (dimension, len(diagonalized.eigenvalues)), dtype=complex
                )
                embedded[coupled_indices] = diagonalized.eigenvectors
                sector_energies.append(diagonalized.eigenvalues)
                sector_vectors.append(embedded)
                sector_ids.append(sector)
                sector_matrices.append(block)
                sector_coupled_indices.append(coupled_indices)
                sector_backends.append(diagonalized.backend)
                assembly_backends_for_sectors.append(block_assembly)
            # Grow only sectors whose last computed root lies below the
            # current global cutoff. This is an exact root-count certificate:
            # an omitted root cannot precede the last computed sector root.
            # It avoids requesting the global root count in every sector.
            while True:
                combined = np.sort(np.concatenate(sector_energies))
                cutoff = combined[min(diagonal_count, len(combined)) - 1]
                tolerance = max(1.0e-6, 256.0 * float(np.spacing(abs(float(cutoff)))))
                grow = [
                    index for index, (energies, matrix) in enumerate(zip(sector_energies, sector_matrices, strict=True))
                    if len(energies) < len(matrix)
                    and (len(combined) < diagonal_count or energies[-1] <= cutoff + tolerance)
                ]
                if not grow:
                    break
                for index in grow:
                    matrix = sector_matrices[index]
                    roots = min(len(matrix), max(len(sector_energies[index]) + 1, 2 * len(sector_energies[index])))
                    diagonalized = diagonalize_hermitian(
                        matrix, backend=selected_backend,
                        subset_by_index=(0, roots - 1) if roots < len(matrix) else None,
                    )
                    if diagonalized.eigenvectors is None:
                        raise RuntimeError("LOCK diagonalizer did not return eigenvectors")
                    embedded = np.zeros((dimension, len(diagonalized.eigenvalues)), dtype=complex)
                    embedded[sector_coupled_indices[index]] = diagonalized.eigenvectors
                    sector_energies[index] = diagonalized.eigenvalues
                    sector_vectors[index] = embedded
            sector_labels = [
                sector for sector, energies in zip(sector_ids, sector_energies, strict=True)
                for _ in energies
            ]
            energies = np.concatenate(sector_energies)
            vectors = np.column_stack(sector_vectors)
            ordering = np.argsort(energies, kind="stable")
            energies = energies[ordering]
            vectors = vectors[:, ordering]
            preassigned_symmetry_sectors = tuple(
                sector_labels[int(index)] for index in ordering
            )
            selected_assembly = "+".join(sorted(set(assembly_backends_for_sectors)))
            selected_diagonalizer = "+".join(sorted(set(sector_backends)))
        else:
            hamiltonian, selected_assembly = _assemble_coupled_hamiltonian(
                basis.energies_MHz,
                contracted["linear"],
                contracted["quadratic"],
                rot,
                distortion,
                coordinate_distortion,
                backend=assembly_backend,
            )
            error = float(np.max(np.abs(hamiltonian - hamiltonian.conj().T), initial=0.0))
            hermitian = _hermitian(hamiltonian)
            if rovibrational_symmetry is not None:
                state_action = rovibrational_symmetry.contracted_action(basis.vectors, J)
                symmetry_blocks = state_action.diagonalize(hermitian)
                energies = np.concatenate([block.energies for block in symmetry_blocks])
                vectors = np.column_stack([block.eigenvectors for block in symmetry_blocks])
                ordering = np.argsort(energies, kind="stable")
                labels = [block.irrep for block in symmetry_blocks for _ in block.energies]
                rovib_labels = tuple(labels[int(index)] for index in ordering)
                energies = energies[ordering]
                vectors = vectors[:, ordering]
                selected_diagonalizer = "symmetry-blocked"
            else:
                selected_backend = None if diagonalizer_backend == "arpack" else diagonalizer_backend
                diagonalized = diagonalize_hermitian(
                    hermitian,
                    backend=selected_backend,
                    subset_by_index=(0, diagonal_count - 1)
                    if diagonal_count < dimension
                    else None,
                )
                if diagonalized.eigenvectors is None:
                    raise RuntimeError("LOCK diagonalizer did not return eigenvectors")
                energies, vectors = diagonalized.eigenvalues, diagonalized.eigenvectors
                selected_diagonalizer = diagonalized.backend
        selected = retained
        energy_scale = float(np.max(np.abs(energies), initial=0.0))
        degeneracy_tolerance = max(1.0e-6, 256.0 * float(np.spacing(energy_scale)))
        while (
            selected < len(energies)
            and abs(energies[selected] - energies[selected - 1]) <= degeneracy_tolerance
        ):
            selected += 1
        energies = np.asarray(energies)[:selected]
        vectors = np.asarray(vectors)[:, :selected]
        rovib_labels = (None,) * selected if rovib_labels is None else rovib_labels[:selected]
        if preassigned_symmetry_sectors is None:
            vectors, symmetry_sectors = _symmetry_adapt_coupled_basis(
                surface, basis, energies, vectors, nrot
            )
        else:
            symmetry_sectors = preassigned_symmetry_sectors[:selected]
        parent_data = _parent_assignments(
            J,
            vectors,
            rot,
            reference_R,
            state_symmetry_sectors=symmetry_sectors,
            lam_symmetry_sectors=basis.symmetry_sectors,
        )
        pi = surface.permutation_inversion_symmetry
        if pi is not None and pi.filter_forbidden:
            keep = np.asarray(
                [
                    pi.spin_weight(pi.rovib_species(sector, parent[1], parent[2])) > 0.0
                    for parent, sector in zip(parent_data, symmetry_sectors, strict=True)
                ],
                dtype=bool,
            )
            energies = np.asarray(energies)[keep]
            vectors = np.asarray(vectors)[:, keep]
            parent_data = [
                parent
                for parent, retained_state in zip(parent_data, keep, strict=True)
                if retained_state
            ]
            symmetry_sectors = tuple(
                sector
                for sector, retained_state in zip(symmetry_sectors, keep, strict=True)
                if retained_state
            )
            rovib_labels = tuple(label for label, retained_state in zip(rovib_labels, keep, strict=True) if retained_state)
        return (
            J,
            energies.real,
            vectors,
            parent_data,
            symmetry_sectors,
            error,
            selected_assembly,
            selected_diagonalizer,
            rovib_labels,
            k_contraction,
        )

    if workers == 1 or len(values) == 1:
        solved_blocks = tuple(solve_j_block(J) for J in values)
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(
            max_workers=min(workers, len(values)), thread_name_prefix="lock-j-block"
        ) as pool:
            solved_blocks = tuple(pool.map(solve_j_block, values))
    for (
        J,
        energies,
        vectors,
        parent_data,
        symmetry_sectors,
        error,
        selected_assembly,
        selected_diagonalizer,
        rovib_labels,
        k_contraction,
    ) in solved_blocks:
        hermiticity_error = max(hermiticity_error, error)
        assembly_backends.append(selected_assembly)
        diagonalizer_backends.append(selected_diagonalizer)
        raw[J] = (energies, vectors, parent_data, symmetry_sectors, rovib_labels)
    computed_k_contractions = {
        block[0]: block[-1] for block in solved_blocks if block[-1] is not None
    }
    nonempty = [
        energies for energies, _vectors, _parents, _sectors, _irreps in raw.values() if len(energies)
    ]
    if not nonempty:
        raise RuntimeError("permutation-inversion filtering removed every LOCK state")
    ground = min(float(energies[0]) for energies in nonempty)
    normalized_weights = _normalized_spin_weights(surface)
    molecular_group = (
        rovibrational_symmetry.group
        if rovibrational_symmetry is not None
        and rovibrational_symmetry.group.nuclear_spin_sets
        else None
    )
    for J, (energies, vectors, parent_data, symmetry_sectors, rovib_labels) in raw.items():
        sector_counts: dict[int, int] = {}
        sector_indices = []
        for sector in symmetry_sectors:
            sector_indices.append(sector_counts.get(sector, 0))
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        levels = tuple(
            LAMRotationLevel(
                J=J,
                index=index,
                energy_MHz=float(energy),
                energy_above_ground_MHz=float(energy - ground),
                lam_parent=parent[0],
                Ka=parent[1],
                Kc=parent[2],
                parent_purity=min(1.0, max(0.0, parent[3])),
                symmetry_sector=symmetry_sector,
                lam_symmetry=_level_symmetry_label(surface, symmetry_sector, parent[1], parent[2]),
                spin_weight=(
                    float(
                        molecular_group.nuclear_spin_statistical_weight(
                            rovib_labels[index], molecular_group.identity_irrep
                        )
                    )
                    if molecular_group is not None and rovib_labels[index] is not None
                    else _level_spin_weight(
                        surface, normalized_weights, symmetry_sector, parent[1], parent[2]
                    )
                ),
                torsional_state=basis.sector_state_indices[parent[0]],
                sector_index=sector_index,
                rovibrational_irrep=rovib_labels[index],
            )
            for index, (energy, parent, symmetry_sector, sector_index) in enumerate(
                zip(energies, parent_data, symmetry_sectors, sector_indices, strict=True)
            )
        )
        blocks[J] = LAMRotationBlock(J=J, energies_MHz=energies, vectors=vectors, levels=levels)
    return LAMRotationResult(
        basis=basis,
        blocks=blocks,
        representation=representation,
        ground_energy_MHz=ground,
        hermiticity_error_MHz=hermiticity_error,
        distortion_model=_distortion_model_name(surface, centrifugal_distortion),
        assembly_backends=tuple(sorted(set(assembly_backends))),
        diagonalizer_backends=tuple(sorted(set(diagonalizer_backends))),
        operator_manifest_digest=(
            None
            if normalized_operator_manifest is None
            else normalized_operator_manifest.digest
        ),
        operator_parameters_MHz=normalized_operator_parameters,
        k_contractions=computed_k_contractions or None,
    )


def lam_rotation_transitions(
    surface: LAMRotationSurface,
    result: LAMRotationResult,
    *,
    dipole_au: Sequence[float] | None = None,
    include_q_branch: bool = True,
    maximum_lam_parent: int | None = None,
    temperature_K: float | None = None,
    include_level_indices: bool = False,
) -> list[dict[str, object]]:
    """Build Pickett-compatible transition rows from coupled eigenstates."""
    primitive_dipole = surface.dipole_au
    if primitive_dipole is None:
        if dipole_au is None or np.asarray(dipole_au).shape != (3,):
            raise ValueError("provide a three-component dipole or a dipole surface")
        primitive_dipole = np.broadcast_to(
            np.asarray(dipole_au, dtype=float), surface.grid_shape + (3,)
        )
    primitive = np.asarray(primitive_dipole, dtype=float).reshape((-1, 3))
    torsion_invariant_dipole = False
    if surface.symmetry_sector_count > 1:
        primitive_tensor = primitive.reshape(surface.grid_shape + (3,))
        torsion_invariant_dipole = all(
            order == 1
            or np.allclose(
                primitive_tensor,
                np.roll(primitive_tensor, surface.grid_shape[axis] // order, axis=axis),
                rtol=1.0e-10,
                atol=1.0e-12,
            )
            for axis, order in enumerate(surface.symmetry_orders)
        )
    U = result.basis.vectors
    contracted_mu = tuple(U.conj().T @ np.diag(primitive[:, a]) @ U for a in range(3))
    if temperature_K is not None and temperature_K <= 0.0:
        raise ValueError("temperature_K must be positive")
    beta = None if temperature_K is None else MHZ_TO_KELVIN / temperature_K
    partition = 1.0
    if beta is not None:
        partition = sum(
            level.spin_weight
            * (2 * J + 1)
            * np.exp(-(level.energy_MHz - result.ground_energy_MHz) * beta)
            for J, block in result.blocks.items()
            for level in block.levels
        )
    rows: list[dict[str, object]] = []
    for Jl, lower in sorted(result.blocks.items()):
        targets = (Jl, Jl + 1) if include_q_branch else (Jl + 1,)
        for Ju in targets:
            if Ju not in result.blocks or Jl == Ju == 0:
                continue
            upper = result.blocks[Ju]
            transition = np.zeros((lower.vectors.shape[0], upper.vectors.shape[0]), dtype=complex)
            for axis, name in enumerate(("a", "b", "c")):
                rotational = dipole_transition_matrix(Jl, Ju, name, result.representation)
                transition += np.kron(contracted_mu[axis], rotational)
            amplitudes = lower.vectors.conj().T @ transition @ upper.vectors
            for il, lower_state in enumerate(lower.levels):
                for iu, upper_state in enumerate(upper.levels):
                    if Jl == Ju and iu <= il:
                        continue
                    if (
                        torsion_invariant_dipole
                        and lower_state.symmetry_sector != upper_state.symmetry_sector
                    ):
                        continue
                    if (
                        maximum_lam_parent is not None
                        and max(lower_state.torsional_state, upper_state.torsional_state)
                        > maximum_lam_parent
                    ):
                        continue
                    if lower_state.energy_MHz <= upper_state.energy_MHz:
                        low, up = lower_state, upper_state
                    else:
                        low, up = upper_state, lower_state
                    frequency = up.energy_MHz - low.energy_MHz
                    if frequency <= 1.0e-9:
                        continue
                    strength = float(abs(amplitudes[il, iu]) ** 2)
                    if strength <= 1.0e-18:
                        continue
                    intensity = low.spin_weight * strength
                    if beta is not None:
                        intensity *= (
                            frequency
                            * (-np.expm1(-frequency * beta))
                            * np.exp(-(low.energy_MHz - result.ground_energy_MHz) * beta)
                            / partition
                        )
                    rows.append(
                        {
                            "Frequency (MHz)": float(frequency),
                            "Intensity": intensity,
                            "Line strength": strength,
                            "Relative intensity": 0.0,
                            "Branch": ("Q" if low.J == up.J else "R" if up.J > low.J else "P"),
                            "Jl": low.J,
                            "Ka_l": low.Ka,
                            "Kc_l": low.Kc,
                            "lam_l": low.lam_parent,
                            "vt_l": low.torsional_state,
                            "alpha_l": low.sector_index,
                            "tors_sym_l": low.lam_symmetry,
                            "rovib_irrep_l": low.rovibrational_irrep,
                            "symmetry_sector_l": low.symmetry_sector,
                            "Ju": up.J,
                            "Ka_u": up.Ka,
                            "Kc_u": up.Kc,
                            "lam_u": up.lam_parent,
                            "vt_u": up.torsional_state,
                            "alpha_u": up.sector_index,
                            "tors_sym_u": up.lam_symmetry,
                            "rovib_irrep_u": up.rovibrational_irrep,
                            "symmetry_sector_u": up.symmetry_sector,
                            "spin_weight": low.spin_weight,
                            "parent_purity": min(low.parent_purity, up.parent_purity),
                        }
                    )
                    if include_level_indices:
                        rows[-1].update(lower_level_index=low.index, upper_level_index=up.index)
    maximum = max((float(row["Intensity"]) for row in rows), default=1.0)
    for row in rows:
        row["Relative intensity"] = 100.0 * float(row["Intensity"]) / maximum
    return sorted(rows, key=lambda row: float(row["Frequency (MHz)"]))


def write_lam_rotation_surface(path: Path | str, surface: LAMRotationSurface) -> Path:
    """Persist a compact, versioned NPZ hand-off from TRINITY/DVR to LOCK."""
    import json

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "schema": np.asarray(LOCK_LAM_ROTATION_SURFACE_SCHEMA),
        "dimension": np.asarray(surface.dimension),
        "periodic": np.asarray(surface.periodic, dtype=bool),
        "labels": np.asarray(
            surface.labels or tuple(f"q{i + 1}" for i in range(surface.dimension))
        ),
        "torsional_symmetry_order": np.asarray(surface.torsional_symmetry_order),
        "lam_symmetry_orders": np.asarray(surface.symmetry_orders, dtype=int),
        "nuclear_spin_weights": np.asarray(surface.nuclear_spin_weights, dtype=float),
        "distortion_reduction": np.asarray(surface.distortion_reduction.upper()),
        "potential_MHz": np.asarray(surface.potential_MHz),
        "F_MHz": np.asarray(surface.F_MHz),
        "rho": np.asarray(surface.rho),
        "R_MHz": np.asarray(surface.R_MHz),
        "metadata_json": np.asarray(
            json.dumps(
                dict(surface.metadata or {}),
                default=lambda value: np.asarray(value).tolist(),
                sort_keys=True,
            )
        ),
        "permutation_inversion_json": np.asarray(
            json.dumps(
                surface.permutation_inversion_symmetry.to_dict()
                if surface.permutation_inversion_symmetry is not None
                else None,
                sort_keys=True,
            )
        ),
    }
    for index, grid in enumerate(surface.coordinates):
        payload[f"q{index + 1}"] = np.asarray(grid)
    if surface.dipole_au is not None:
        payload["dipole_au"] = np.asarray(surface.dipole_au)
    if surface.quartic_distortion_MHz is not None:
        payload["quartic_distortion_MHz"] = np.asarray(surface.quartic_distortion_MHz)
    if surface.sextic_distortion_MHz is not None:
        payload["sextic_distortion_MHz"] = np.asarray(surface.sextic_distortion_MHz)
    if surface.geometric_potential_MHz is not None:
        payload["geometric_potential_MHz"] = np.asarray(surface.geometric_potential_MHz)
    if surface.lam_quartic_kinetic_MHz is not None:
        payload["lam_quartic_kinetic_MHz"] = np.asarray(surface.lam_quartic_kinetic_MHz)
    if surface.p2_rotational_metric_MHz is not None:
        payload["p2_rotational_metric_MHz"] = np.asarray(surface.p2_rotational_metric_MHz)
    if surface.p_rotational_cubic_MHz is not None:
        payload["p_rotational_cubic_MHz"] = np.asarray(surface.p_rotational_cubic_MHz)
    if surface.p_rotational_anticommutator_MHz is not None:
        payload["p_rotational_anticommutator_MHz"] = np.asarray(
            surface.p_rotational_anticommutator_MHz
        )
    if surface.pi2_rotation_MHz is not None:
        payload["pi2_rotation_MHz"] = np.asarray(surface.pi2_rotation_MHz)
    spin_surface_names = tuple(sorted((surface.spin_tensor_surfaces_MHz or {}).keys()))
    payload["spin_tensor_surface_names"] = np.asarray(spin_surface_names)
    for index, name in enumerate(spin_surface_names):
        payload[f"spin_tensor_surface_{index}"] = np.asarray(surface.spin_tensor_surfaces_MHz[name])
    np.savez_compressed(target, **payload)
    return target


def read_lam_rotation_surface(path: Path | str) -> LAMRotationSurface:
    """Read the versioned TRINITY/DVR to LOCK NPZ hand-off."""
    import json

    with np.load(Path(path), allow_pickle=False) as data:
        schema = str(np.asarray(data["schema"]).item())
        if schema not in {
            LOCK_LAM_ROTATION_SURFACE_SCHEMA_V1,
            LOCK_LAM_ROTATION_SURFACE_SCHEMA_V2,
            LOCK_LAM_ROTATION_SURFACE_SCHEMA_V3,
            LOCK_LAM_ROTATION_SURFACE_SCHEMA_V4,
            LOCK_LAM_ROTATION_SURFACE_SCHEMA,
        }:
            raise ValueError(f"unsupported LAM-rotation surface schema {schema!r}")
        dimension = int(np.asarray(data["dimension"]).item())
        return LAMRotationSurface(
            coordinates=tuple(np.asarray(data[f"q{i + 1}"]) for i in range(dimension)),
            potential_MHz=np.asarray(data["potential_MHz"]),
            F_MHz=np.asarray(data["F_MHz"]),
            rho=np.asarray(data["rho"]),
            R_MHz=np.asarray(data["R_MHz"]),
            periodic=tuple(bool(value) for value in np.asarray(data["periodic"])),
            labels=tuple(str(value) for value in np.asarray(data["labels"])),
            dipole_au=np.asarray(data["dipole_au"]) if "dipole_au" in data else None,
            quartic_distortion_MHz=(
                np.asarray(data["quartic_distortion_MHz"])
                if "quartic_distortion_MHz" in data
                else None
            ),
            sextic_distortion_MHz=(
                np.asarray(data["sextic_distortion_MHz"])
                if "sextic_distortion_MHz" in data
                else None
            ),
            distortion_reduction=(
                str(np.asarray(data["distortion_reduction"]).item())
                if "distortion_reduction" in data
                else "S"
            ),
            geometric_potential_MHz=(
                np.asarray(data["geometric_potential_MHz"])
                if "geometric_potential_MHz" in data
                else None
            ),
            lam_quartic_kinetic_MHz=(
                np.asarray(data["lam_quartic_kinetic_MHz"])
                if "lam_quartic_kinetic_MHz" in data
                else None
            ),
            p2_rotational_metric_MHz=(
                np.asarray(data["p2_rotational_metric_MHz"])
                if "p2_rotational_metric_MHz" in data
                else None
            ),
            p_rotational_cubic_MHz=(
                np.asarray(data["p_rotational_cubic_MHz"])
                if "p_rotational_cubic_MHz" in data
                else None
            ),
            p_rotational_anticommutator_MHz=(
                np.asarray(data["p_rotational_anticommutator_MHz"])
                if "p_rotational_anticommutator_MHz" in data
                else None
            ),
            pi2_rotation_MHz=(
                np.asarray(data["pi2_rotation_MHz"])
                if "pi2_rotation_MHz" in data
                else None
            ),
            metadata={
                **(
                    json.loads(str(np.asarray(data["metadata_json"]).item()))
                    if "metadata_json" in data
                    else {}
                ),
                "source": str(Path(path)),
            },
            torsional_symmetry_order=(
                int(np.asarray(data["torsional_symmetry_order"]).item())
                if "torsional_symmetry_order" in data
                else 1
            ),
            lam_symmetry_orders=(
                tuple(int(value) for value in np.asarray(data["lam_symmetry_orders"]))
                if "lam_symmetry_orders" in data
                else ()
            ),
            nuclear_spin_weights=(
                tuple(float(value) for value in np.asarray(data["nuclear_spin_weights"]))
                if "nuclear_spin_weights" in data
                else ()
            ),
            permutation_inversion_symmetry=(
                PermutationInversionSymmetry.from_dict(
                    json.loads(str(np.asarray(data["permutation_inversion_json"]).item()))
                )
                if "permutation_inversion_json" in data
                and json.loads(str(np.asarray(data["permutation_inversion_json"]).item()))
                is not None
                else None
            ),
            spin_tensor_surfaces_MHz={
                str(name): np.asarray(data[f"spin_tensor_surface_{index}"])
                for index, name in enumerate(
                    np.asarray(data["spin_tensor_surface_names"])
                    if "spin_tensor_surface_names" in data
                    else ()
                )
            },
        )


def write_lam_rotation_result(
    path: Path | str,
    result: LAMRotationResult,
    *,
    surface_path: Path | str | None = None,
) -> Path:
    """Write coupled levels and assignment diagnostics as a LOCK artifact."""
    import json

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": LOCK_LAM_ROTATION_RESULT_SCHEMA,
        "kinetic_convention": LAM_KINETIC_CONVENTION,
        "surface": None if surface_path is None else str(surface_path),
        "representation": result.representation,
        "lam_basis_energies_MHz": [float(value) for value in result.basis.energies_MHz],
        "lam_basis": {
            "primitive_representation": result.basis.primitive_representation,
            "primitive_size": result.basis.primitive_size,
            "fourier_cutoffs": result.basis.fourier_cutoffs,
            "diagnostics": result.basis.diagnostics,
        },
        "jk_contraction": (
            None if result.k_contractions is None else
            {str(J): value.diagnostics for J, value in result.k_contractions.items()}
        ),
        "ground_energy_MHz": result.ground_energy_MHz,
        "hermiticity_error_MHz": result.hermiticity_error_MHz,
        "distortion_model": result.distortion_model,
        "assembly_backends": list(result.assembly_backends),
        "diagonalizer_backends": list(result.diagonalizer_backends),
        "operator_manifest_digest": result.operator_manifest_digest,
        "operator_parameters_MHz": (
            None
            if result.operator_parameters_MHz is None
            else dict(result.operator_parameters_MHz)
        ),
        "blocks": {
            str(J): [
                {
                    "index": level.index,
                    "energy_MHz": level.energy_MHz,
                    "energy_above_ground_MHz": level.energy_above_ground_MHz,
                    "lam_parent": level.lam_parent,
                    "Ka": level.Ka,
                    "Kc": level.Kc,
                    "parent_purity": level.parent_purity,
                    "symmetry_sector": level.symmetry_sector,
                    "lam_symmetry": level.lam_symmetry,
                    "rovibrational_irrep": level.rovibrational_irrep,
                    "spin_weight": level.spin_weight,
                    "torsional_state": level.torsional_state,
                    "sector_index": level.sector_index,
                }
                for level in block.levels
            ]
            for J, block in sorted(result.blocks.items())
        },
    }
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def lock_native_available() -> bool:
    """Return whether the compiled C++ LOCK assembly kernel is importable."""
    return _lock_native is not None


def _assemble_coupled_hamiltonian(
    energies: np.ndarray,
    linear: np.ndarray,
    quadratic: np.ndarray,
    rotational: tuple[np.ndarray, np.ndarray, np.ndarray],
    distortion: np.ndarray | None,
    coordinate_distortion: tuple[tuple[np.ndarray, np.ndarray], ...] | None = None,
    *,
    backend: str | None,
) -> tuple[np.ndarray, str]:
    nlam = len(energies)
    nrot = rotational[0].shape[0]
    selected = resolve_native_backend(
        extension_available=_lock_native is not None,
        extension_name="lock-cpp",
        workload_size=(nlam * nrot) ** 2,
        requested=backend,
        minimum_work=4096,
    )
    if selected.accelerated and _lock_native is not None:
        hamiltonian = np.asarray(
            _lock_native.assemble_lam_rotation(
                np.asarray(energies, dtype=float),
                np.asarray(linear, dtype=complex),
                np.asarray(quadratic, dtype=complex),
                np.asarray(rotational, dtype=complex),
                None if distortion is None else np.asarray(distortion, dtype=complex),
            )
        )
    else:
        hamiltonian = np.kron(np.diag(energies), np.eye(nrot)).astype(complex)
        for axis in range(3):
            hamiltonian += np.kron(linear[axis], rotational[axis])
        for a in range(3):
            for b in range(3):
                symrot = 0.5 * (rotational[a] @ rotational[b] + rotational[b] @ rotational[a])
                hamiltonian += np.kron(quadratic[a, b], symrot)
        if distortion is not None:
            hamiltonian += np.kron(np.eye(nlam), distortion)
    # Many declared words share the same rotational monomial. Combine their
    # LAM factors first, then update only nonzero rotational blocks. This
    # avoids allocating one full product-space temporary for every term.
    grouped: dict[tuple[tuple[int, ...], bytes], tuple[np.ndarray, np.ndarray]] = {}
    for coefficient, operator in coordinate_distortion or ():
        key = (operator.shape, operator.tobytes())
        if key in grouped:
            grouped[key][0][:] += coefficient
        else:
            grouped[key] = (np.array(coefficient, copy=True), operator)
    product_blocks = hamiltonian.reshape(nlam, nrot, nlam, nrot)
    for coefficient, operator in grouped.values():
        rows, columns = np.nonzero(operator)
        if len(rows) <= operator.size // 3:
            for row, column in zip(rows, columns, strict=True):
                product_blocks[:, row, :, column] += operator[row, column] * coefficient
        else:
            hamiltonian += np.kron(coefficient, operator)
    return hamiltonian, selected.name


def _diagonalize_coupled_matrix_free(
    energies: np.ndarray,
    linear: np.ndarray,
    quadratic: np.ndarray,
    rotational: tuple[np.ndarray, np.ndarray, np.ndarray],
    distortion: np.ndarray | None,
    coordinate_distortion: tuple[tuple[np.ndarray, np.ndarray], ...] | None = None,
    *,
    level_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.sparse.linalg import LinearOperator, eigsh

    nlam = len(energies)
    nrot = rotational[0].shape[0]
    dimension = nlam * nrot
    symrot = np.empty((3, 3, nrot, nrot), dtype=complex)
    for a in range(3):
        for b in range(3):
            symrot[a, b] = 0.5 * (rotational[a] @ rotational[b] + rotational[b] @ rotational[a])

    def action(values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=complex)
        is_vector = array.ndim == 1
        columns = array.reshape((dimension, -1))
        result = np.empty_like(columns)
        for column in range(columns.shape[1]):
            state = columns[:, column].reshape((nlam, nrot))
            product = energies[:, None] * state
            for axis in range(3):
                product += linear[axis] @ state @ rotational[axis].T
            for a in range(3):
                for b in range(3):
                    product += quadratic[a, b] @ state @ symrot[a, b].T
            if distortion is not None:
                product += state @ distortion.T
            for coefficient, operator in coordinate_distortion or ():
                product += coefficient @ state @ operator.T
            result[:, column] = product.reshape(-1)
        return result[:, 0] if is_vector else result

    operator = LinearOperator(
        (dimension, dimension),
        matvec=action,
        matmat=action,
        dtype=np.complex128,
    )
    values, vectors = eigsh(
        operator,
        k=level_count,
        which="SA",
        tol=1.0e-10,
        maxiter=max(1000, 20 * dimension),
        v0=np.linspace(1.0, 2.0, dimension, dtype=complex),
    )
    order = np.argsort(values)
    from ._iterative_roots import complete_lowest_ritz_subspace

    bound = float(np.max(np.abs(energies), initial=0.))
    for axis in range(3):
        bound += np.linalg.norm(linear[axis], 1) * np.linalg.norm(rotational[axis], 1)
    for a in range(3):
        for b in range(3):
            bound += np.linalg.norm(quadratic[a, b], 1) * np.linalg.norm(symrot[a, b], 1)
    if distortion is not None:
        bound += np.linalg.norm(distortion, 1)
    for coefficient, operator in coordinate_distortion or ():
        bound += np.linalg.norm(coefficient, 1) * np.linalg.norm(operator, 1)
    return complete_lowest_ritz_subspace(
        action, np.asarray(values[order]).real, np.asarray(vectors[:, order]),
        spectral_bound=bound,
    )


def _coupled_coefficient_hermiticity_error(
    linear: np.ndarray,
    quadratic: np.ndarray,
    rotational: tuple[np.ndarray, np.ndarray, np.ndarray],
    distortion: np.ndarray | None,
    coordinate_distortion: tuple[tuple[np.ndarray, np.ndarray], ...] | None = None,
) -> float:
    errors = [
        float(np.max(np.abs(value - value.conj().T), initial=0.0))
        for value in (*linear, *rotational)
    ]
    for a in range(3):
        for b in range(3):
            errors.append(
                float(
                    np.max(
                        np.abs(quadratic[a, b] - quadratic[b, a].conj().T),
                        initial=0.0,
                    )
                )
            )
    if distortion is not None:
        errors.append(float(np.max(np.abs(distortion - distortion.conj().T), initial=0.0)))
    for coefficient, operator in coordinate_distortion or ():
        errors.append(float(np.max(np.abs(coefficient - coefficient.conj().T), initial=0.0)))
        errors.append(float(np.max(np.abs(operator - operator.conj().T), initial=0.0)))
    return max(errors, default=0.0)


def _lam_kinetic_norm_bound(surface, potential):
    """Conservative norm bound for the discretized reduced kinetic operator."""
    bounds = []
    for q, periodic in zip(surface.coordinates, surface.periodic, strict=True):
        spacing = float(q[1]-q[0])
        bounds.append(float(np.max(np.abs(2*np.pi*np.fft.fftfreq(len(q), spacing))))
                      if periodic else 2./spacing)
    result = float(np.max(np.abs(potential), initial=0.))
    for mu in range(surface.dimension):
        for nu in range(surface.dimension):
            result += float(np.max(np.abs(surface.F_MHz[..., mu, nu]))) * bounds[mu]*bounds[nu]
        if surface.lam_quartic_kinetic_MHz is not None:
            result += float(np.max(np.abs(surface.lam_quartic_kinetic_MHz[..., mu]))) * bounds[mu]**4
    return result


def _apply_lam_hamiltonian(
    surface: LAMRotationSurface,
    values: np.ndarray,
    *,
    potential: np.ndarray,
    coefficient_direction: Mapping[str, object] | None = None,
) -> np.ndarray:
    # The primitive kinetic operator is linear in F and the quartic field.
    # Response calculations use this same dispatcher, including the positive
    # Dirichlet flux discretization on confined axes, with signed directions.
    kinetic_field = surface.F_MHz if coefficient_direction is None else coefficient_direction["F_MHz"]
    quartic_field = surface.lam_quartic_kinetic_MHz if coefficient_direction is None else coefficient_direction["lam_quartic_kinetic_MHz"]
    array = np.asarray(values, dtype=complex)
    is_vector = array.ndim == 1
    columns = array.reshape((int(np.prod(surface.grid_shape)), -1))
    result = potential[:, None] * columns
    F = np.asarray(kinetic_field).reshape((len(columns), surface.dimension, surface.dimension))
    confined_metric = surface.dimension > 1 and not any(surface.periodic)
    needs_momentum = ((surface.dimension > 1 and not confined_metric) or any(surface.periodic)
                      or quartic_field is not None)
    momentum_values = ([_apply_momentum(surface, axis, columns) for axis in range(surface.dimension)]
                       if needs_momentum else [])
    if confined_metric:
        result += _apply_confined_metric_kinetic(surface, F, columns)
    else:
        for mu in range(surface.dimension):
            for nu in range(surface.dimension):
                if mu == nu and not surface.periodic[mu]:
                    result += _apply_confined_kinetic(surface, mu, F[:, mu, mu], columns)
                else:
                    result += _apply_momentum(
                        surface, mu, F[:, mu, nu, None] * momentum_values[nu],
                    )
    if quartic_field is not None:
        quartic = np.asarray(quartic_field).reshape(
            (len(columns), surface.dimension)
        )
        for mu in range(surface.dimension):
            momentum2 = _apply_momentum(surface, mu, momentum_values[mu])
            weighted = quartic[:, mu, None] * momentum2
            result += _apply_momentum(surface, mu, _apply_momentum(surface, mu, weighted))
    return result[:, 0] if is_vector else result


def _apply_momentum(surface: LAMRotationSurface, axis: int, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=complex)
    is_vector = array.ndim == 1
    columns = array.reshape((int(np.prod(surface.grid_shape)), -1))
    tensor = columns.reshape(surface.grid_shape + (columns.shape[1],))
    if surface.periodic[axis]:
        grid = np.asarray(surface.coordinates[axis], dtype=float)
        spacing = float(grid[1] - grid[0])
        wave_numbers = 2.0 * np.pi * np.fft.fftfreq(len(grid), d=spacing)
        reshape = [1] * tensor.ndim
        reshape[axis] = len(grid)
        transformed = np.fft.fft(tensor, axis=axis)
        acted = np.fft.ifft(wave_numbers.reshape(reshape) * transformed, axis=axis)
    else:
        local = _confined_momentum(np.asarray(surface.coordinates[axis], dtype=float))
        moved = np.moveaxis(tensor, axis, 0)
        acted = np.moveaxis(np.tensordot(local, moved, axes=(1, 0)), 0, axis)
    flattened = np.asarray(acted).reshape(columns.shape)
    return flattened[:, 0] if is_vector else flattened


def _apply_confined_kinetic(
    surface: LAMRotationSurface, axis: int, coefficient: np.ndarray, values: np.ndarray,
) -> np.ndarray:
    """Positive Dirichlet flux form for one nonperiodic DVR coordinate.

    Edge gradients include both boundary faces. The interior edge coefficient
    is the mean of its adjacent grid values. This avoids the two decoupled
    sublattices produced by squaring a centered first derivative.
    """
    array = np.asarray(values, dtype=complex)
    is_vector = array.ndim == 1
    size = int(np.prod(surface.grid_shape))
    columns = array.reshape((size, -1))
    field = np.asarray(coefficient, dtype=float).reshape(surface.grid_shape)
    tensor = np.moveaxis(columns.reshape(surface.grid_shape + (columns.shape[1],)), axis, 0)
    weights = np.moveaxis(field, axis, 0)
    n = len(tensor)
    spacing = float(surface.coordinates[axis][1] - surface.coordinates[axis][0])
    edge = np.empty((n + 1,) + weights.shape[1:], dtype=float)
    edge[0], edge[-1] = weights[0], weights[-1]
    edge[1:-1] = 0.5 * (weights[:-1] + weights[1:])
    gradient = np.empty((n + 1,) + tensor.shape[1:], dtype=complex)
    gradient[0], gradient[-1] = tensor[0] / spacing, -tensor[-1] / spacing
    gradient[1:-1] = (tensor[1:] - tensor[:-1]) / spacing
    flux = edge[..., None] * gradient
    acted = np.moveaxis((flux[:-1] - flux[1:]) / spacing, 0, axis).reshape(columns.shape)
    return acted[:, 0] if is_vector else acted


def _apply_confined_metric_kinetic(
    surface: LAMRotationSurface,
    coefficient: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Positive multidimensional flux form for a full nonperiodic metric.

    Cross derivatives must use the same discrete gradient as diagonal terms.
    Combining the older face-flux diagonal with centered-momentum cross terms
    breaks the positive quadratic form and produces grid-parity oscillations.
    Averaging the lower- and upper-oriented Dirichlet forms removes directional
    bias while preserving ``D.T @ F @ D`` positivity for every metric tensor.
    Each orientation contains only one exterior face per axis. Complete the
    other half of both exterior diagonal face energies so the diagonal-metric
    limit agrees with the one-axis Dirichlet flux operator.
    """

    columns = np.asarray(values, dtype=complex).reshape(
        (int(np.prod(surface.grid_shape)), -1)
    )
    tensor = columns.reshape(surface.grid_shape + (columns.shape[1],))
    metric = np.asarray(coefficient, dtype=float).reshape(
        surface.grid_shape + (surface.dimension, surface.dimension)
    )
    spacings = tuple(
        float(coordinate[1] - coordinate[0]) for coordinate in surface.coordinates
    )
    result = np.zeros_like(tensor)

    for orientation in (-1, 1):
        gradients = []
        for axis, spacing in enumerate(spacings):
            gradient = np.empty_like(tensor)
            current = [slice(None)] * tensor.ndim
            neighbor = [slice(None)] * tensor.ndim
            if orientation < 0:
                current[axis] = 0
                gradient[tuple(current)] = tensor[tuple(current)] / spacing
                current[axis] = slice(1, None)
                neighbor[axis] = slice(None, -1)
                gradient[tuple(current)] = (
                    tensor[tuple(current)] - tensor[tuple(neighbor)]
                ) / spacing
            else:
                current[axis] = -1
                gradient[tuple(current)] = -tensor[tuple(current)] / spacing
                current[axis] = slice(None, -1)
                neighbor[axis] = slice(1, None)
                gradient[tuple(current)] = (
                    tensor[tuple(neighbor)] - tensor[tuple(current)]
                ) / spacing
            gradients.append(gradient)

        fluxes = [
            sum(
                metric[..., left, right, None] * gradients[right]
                for right in range(surface.dimension)
            )
            for left in range(surface.dimension)
        ]
        oriented = np.zeros_like(tensor)
        for axis, (flux, spacing) in enumerate(zip(fluxes, spacings, strict=True)):
            current = [slice(None)] * tensor.ndim
            neighbor = [slice(None)] * tensor.ndim
            if orientation < 0:
                current[axis] = -1
                oriented[tuple(current)] += flux[tuple(current)] / spacing
                current[axis] = slice(None, -1)
                neighbor[axis] = slice(1, None)
                oriented[tuple(current)] += (
                    flux[tuple(current)] - flux[tuple(neighbor)]
                ) / spacing
            else:
                current[axis] = 0
                oriented[tuple(current)] -= flux[tuple(current)] / spacing
                current[axis] = slice(1, None)
                neighbor[axis] = slice(None, -1)
                oriented[tuple(current)] += (
                    flux[tuple(neighbor)] - flux[tuple(current)]
                ) / spacing
        result += 0.5 * oriented
    # The oriented average counts each exterior face with weight one half.
    # Dirichlet zero ghosts require its full face energy, independently of
    # whether mixed metric terms are present. The missing terms are diagonal,
    # nonnegative for a physical metric, and linear in coefficient responses.
    for axis, spacing in enumerate(spacings):
        for side in (0, -1):
            face = [slice(None)] * surface.dimension
            face[axis] = side
            face = tuple(face)
            result[face] += (
                0.5 * metric[face + (axis, axis)][..., None]
                * tensor[face] / spacing**2
            )
    flattened = result.reshape(columns.shape)
    return flattened[:, 0] if np.asarray(values).ndim == 1 else flattened


def _symmetry_adapt_primitive_basis(
    surface: LAMRotationSurface,
    energies: np.ndarray,
    vectors: np.ndarray,
) -> tuple[np.ndarray, tuple[int, ...]]:
    orders = surface.symmetry_orders
    if surface.symmetry_sector_count == 1:
        return vectors, (0,) * vectors.shape[1]

    def translate(axis: int, block: np.ndarray) -> np.ndarray:
        tensor = block.reshape(surface.grid_shape + (block.shape[1],))
        step = surface.grid_shape[axis] // orders[axis]
        return np.roll(tensor, -step, axis=axis).reshape(block.shape)

    return _adapt_degenerate_product_symmetry(energies, vectors, translate, orders)


def _symmetry_adapt_coupled_basis(
    surface: LAMRotationSurface,
    basis: LAMBasis,
    energies: np.ndarray,
    vectors: np.ndarray,
    nrot: int,
) -> tuple[np.ndarray, tuple[int, ...]]:
    sector_count = surface.symmetry_sector_count
    if sector_count == 1:
        return vectors, (0,) * vectors.shape[1]
    # A mixed-radix product-sector index is only a label; treating it as the
    # exponent of one cyclic translation is incorrect when more than one LAM
    # coordinate has active symmetry.  Use a Hermitian marker with one distinct
    # eigenvalue per product sector.  It commutes with a symmetry-preserving
    # Hamiltonian and therefore gives an unambiguous simultaneous basis inside
    # every numerically degenerate eigenspace.
    repeated = np.repeat(np.asarray(basis.symmetry_sectors, dtype=float), nrot)
    adapted = np.asarray(vectors, dtype=complex).copy()
    energy_scale = float(np.max(np.abs(energies), initial=0.0))
    tolerance = max(1.0e-6, 256.0 * float(np.spacing(energy_scale)))
    start = 0
    while start < len(energies):
        stop = start + 1
        while stop < len(energies) and abs(energies[stop] - energies[start]) <= tolerance:
            stop += 1
        block = adapted[:, start:stop]
        marker = block.conj().T @ (repeated[:, None] * block)
        _values, rotation = np.linalg.eigh(_hermitian(marker))
        adapted[:, start:stop] = block @ rotation
        start = stop

    sector_projectors = np.zeros((sector_count, len(repeated)), dtype=float)
    sector_projectors[repeated.astype(int), np.arange(len(repeated))] = 1.0
    probabilities = np.abs(adapted) ** 2
    weights = sector_projectors @ probabilities
    sectors = tuple(int(value) for value in np.argmax(weights, axis=0))
    purity = np.max(weights, axis=0)
    if float(np.min(purity, initial=1.0)) < 1.0 - 1.0e-4:
        raise RuntimeError(
            "coupled LOCK states are not pure product-symmetry eigenstates "
            f"(minimum purity {float(np.min(purity)):.9f})"
        )
    return adapted, sectors


def _adapt_degenerate_symmetry(
    energies: np.ndarray,
    vectors: np.ndarray,
    translate,
    order: int,
) -> tuple[np.ndarray, tuple[int, ...]]:
    from scipy.linalg import schur

    adapted = np.asarray(vectors, dtype=complex).copy()
    energy_scale = float(np.max(np.abs(energies), initial=0.0))
    tolerance = max(1.0e-6, 256.0 * float(np.spacing(energy_scale)))
    start = 0
    while start < len(energies):
        stop = start + 1
        while stop < len(energies) and abs(energies[stop] - energies[start]) <= tolerance:
            stop += 1
        block = adapted[:, start:stop]
        projected = block.conj().T @ translate(block)
        _triangular, rotation = schur(projected, output="complex")
        adapted[:, start:stop] = block @ rotation
        start = stop
    translated = translate(adapted)
    sectors = tuple(
        _nearest_symmetry_sector(np.vdot(adapted[:, index], translated[:, index]), order)
        for index in range(adapted.shape[1])
    )
    return adapted, sectors


def _adapt_degenerate_product_symmetry(
    energies: np.ndarray,
    vectors: np.ndarray,
    translate,
    orders: tuple[int, ...],
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Simultaneously adapt degenerate states to commuting LAM translations."""

    adapted = np.asarray(vectors, dtype=complex).copy()
    active_axes = tuple(axis for axis, order in enumerate(orders) if order > 1)
    energy_scale = float(np.max(np.abs(energies), initial=0.0))
    tolerance = max(1.0e-6, 256.0 * float(np.spacing(energy_scale)))
    start = 0
    while start < len(energies):
        stop = start + 1
        while stop < len(energies) and abs(energies[stop] - energies[start]) <= tolerance:
            stop += 1
        block = adapted[:, start:stop]
        generator = np.zeros((stop - start, stop - start), dtype=complex)
        for position, axis in enumerate(active_axes):
            projected = block.conj().T @ translate(axis, block)
            cosine = 0.5 * (projected + projected.conj().T)
            sine = (projected - projected.conj().T) / (2.0j)
            generator += (position + 1.0) * cosine + math.sqrt(position + 2.0) * sine
        _values, rotation = np.linalg.eigh(_hermitian(generator))
        adapted[:, start:stop] = block @ rotation
        start = stop
    sectors = []
    for state in range(adapted.shape[1]):
        sector = 0
        for axis, order in enumerate(orders):
            local = (
                0
                if order == 1
                else _nearest_symmetry_sector(
                    np.vdot(
                        adapted[:, state], translate(axis, adapted[:, state : state + 1])[:, 0]
                    ),
                    order,
                )
            )
            sector = sector * order + local
        sectors.append(sector)
    return adapted, tuple(sectors)


def _orthonormalize_degenerate_subspaces(energies: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Repair ARPACK bases when repeated eigenvalues lose orthogonality."""
    orthonormal = np.asarray(vectors, dtype=complex).copy()
    energy_scale = float(np.max(np.abs(energies), initial=0.0))
    tolerance = max(1.0e-6, 256.0 * float(np.spacing(energy_scale)))
    start = 0
    while start < len(energies):
        stop = start + 1
        while stop < len(energies) and abs(energies[stop] - energies[start]) <= tolerance:
            stop += 1
        orthonormal[:, start:stop], _ = np.linalg.qr(orthonormal[:, start:stop], mode="reduced")
        start = stop
    gram = _hermitian(orthonormal.conj().T @ orthonormal)
    values, rotation = np.linalg.eigh(gram)
    if float(np.min(values)) <= 0.0:
        raise RuntimeError("ARPACK returned a linearly dependent LOCK basis")
    return orthonormal @ (rotation @ np.diag(values**-0.5) @ rotation.conj().T)


def _nearest_symmetry_sector(value: complex, order: int) -> int:
    if abs(value) < 0.5:
        return 0
    phase = float(np.angle(value)) % (2.0 * np.pi)
    return int(np.rint(phase * order / (2.0 * np.pi))) % order


def _normalized_spin_weights(surface: LAMRotationSurface) -> np.ndarray:
    if surface.permutation_inversion_symmetry is not None:
        weights = np.asarray(
            surface.permutation_inversion_symmetry.nuclear_spin_weights, dtype=float
        )
        total = float(np.sum(weights))
        return weights / total if total > 0.0 else weights
    if surface.nuclear_spin_weights:
        weights = np.asarray(surface.nuclear_spin_weights, dtype=float)
    else:
        weights = np.ones(surface.symmetry_sector_count, dtype=float)
    return weights / float(np.sum(weights))


def _level_symmetry_label(surface: LAMRotationSurface, sector: int, Ka: int, Kc: int) -> str:
    pi = surface.permutation_inversion_symmetry
    if pi is None:
        return _lam_sector_name(sector, surface.symmetry_orders)
    return pi.species[pi.rovib_species(sector, Ka, Kc)]


def _level_spin_weight(
    surface: LAMRotationSurface,
    normalized_weights: np.ndarray,
    sector: int,
    Ka: int,
    Kc: int,
) -> float:
    pi = surface.permutation_inversion_symmetry
    if pi is None:
        return float(normalized_weights[sector])
    rovib = pi.rovib_species(sector, Ka, Kc)
    row = pi.multiplication_table[rovib]
    spin_species = row.index(pi.required_total_species)
    return float(normalized_weights[spin_species])


def _torsional_sector_name(sector: int, order: int) -> str:
    if sector == 0:
        return "A"
    if order % 2 == 0 and sector == order // 2:
        return "B"
    sign = "+" if sector < order / 2 else "-"
    index = min(sector, order - sector)
    return f"E{sign}" if order <= 4 else f"E{sign}{index}"


def _lam_sector_name(sector: int, orders: tuple[int, ...]) -> str:
    local = []
    remaining = int(sector)
    for order in reversed(orders):
        local.append(remaining % order)
        remaining //= order
    labels = [
        _torsional_sector_name(value, order)
        for value, order in zip(reversed(local), orders, strict=True)
        if order > 1
    ]
    return "/".join(labels) if labels else "A"


def _project_lam_values(basis, acted, *, preserve_sectors=False):
    """Project an acted grid basis using exact Fourier sector support."""
    left = basis.vectors
    right = np.asarray(acted)
    spectral = basis.spectral_space
    if spectral is not None and basis.spectral_coefficients is not None:
        left = basis.spectral_coefficients
        right = spectral.from_grid(right)
    labels = np.asarray(basis.symmetry_sectors)
    if not preserve_sectors or len(np.unique(labels)) <= 1:
        return left.conj().T @ right
    result = np.zeros((left.shape[1], right.shape[1]), complex)
    for sector in np.unique(labels):
        columns = np.flatnonzero(labels == sector)
        rows = (np.flatnonzero(spectral.sector_labels == sector)
                if spectral is not None and basis.spectral_coefficients is not None
                else np.arange(len(left)))
        result[np.ix_(columns, columns)] = left[np.ix_(rows, columns)].conj().T @ right[np.ix_(rows, columns)]
    return result


def _contracted_lam_operators(
    surface: LAMRotationSurface, basis: LAMBasis
) -> dict[str, np.ndarray]:
    if surface.symmetry_sector_count > 1:
        from .lam_fourier import _validate_translation_fields

        _validate_translation_fields(surface, np.asarray(surface.potential_MHz), include_rotation=True)
    grid_size = int(np.prod(surface.grid_shape))
    dimension = surface.dimension
    F = np.asarray(surface.F_MHz).reshape((grid_size, dimension, dimension))
    rho = np.asarray(surface.rho).reshape((grid_size, 3, dimension))
    R = np.asarray(surface.R_MHz).reshape((grid_size, 3, 3))
    U = basis.vectors
    def project(acted):
        return _project_lam_values(basis, acted, preserve_sectors=surface.symmetry_sector_count > 1)
    nlam = U.shape[1]
    momentum_U = tuple(
        basis.momentum_operators[mu] @ U if basis.momentum_operators
        else _apply_momentum(surface, mu, U)
        for mu in range(dimension)
    )
    linear = np.zeros((3, nlam, nlam), dtype=complex)
    quadratic = np.zeros((3, 3, nlam, nlam), dtype=complex)
    for a in range(3):
        primitive_U = np.zeros_like(U, dtype=complex)
        for mu in range(dimension):
            for nu in range(dimension):
                values = F[:, mu, nu] * rho[:, a, nu]
                if basis.momentum_operators:
                    P = basis.momentum_operators
                    primitive_U -= P[mu] @ (values[:, None] * U)
                    primitive_U -= (F[:, mu, nu] * rho[:, a, mu])[:, None] * momentum_U[nu]
                else:
                    primitive_U -= _apply_momentum(surface, mu, values[:, None] * U)
                    primitive_U -= (F[:, mu, nu] * rho[:, a, mu])[:, None] * momentum_U[nu]
        linear[a] = project(primitive_U)
    for a in range(3):
        for b in range(3):
            values = R[:, a, b].copy()
            for mu in range(dimension):
                for nu in range(dimension):
                    values += rho[:, a, mu] * F[:, mu, nu] * rho[:, b, nu]
            quadratic[a, b] = project(values[:, None] * U)
    distortion_fields = []
    if surface.quartic_distortion_MHz is not None:
        distortion_fields.extend(
            np.moveaxis(
                np.asarray(surface.quartic_distortion_MHz).reshape((grid_size, 5)),
                -1,
                0,
            )
        )
    elif surface.sextic_distortion_MHz is not None:
        distortion_fields.extend(np.zeros((5, grid_size)))
    if surface.sextic_distortion_MHz is not None:
        distortion_fields.extend(
            np.moveaxis(
                np.asarray(surface.sextic_distortion_MHz).reshape((grid_size, 7)),
                -1,
                0,
            )
        )
    elif surface.quartic_distortion_MHz is not None:
        distortion_fields.extend(np.zeros((7, grid_size)))
    distortion = np.asarray(
        [project(values[:, None] * U) for values in distortion_fields],
        dtype=complex,
    )
    if not distortion_fields:
        distortion = np.empty((0, nlam, nlam), dtype=complex)
    p2_rotational = np.empty((0, 3, 3, nlam, nlam), dtype=complex)
    if surface.p2_rotational_metric_MHz is not None:
        fields = np.asarray(surface.p2_rotational_metric_MHz).reshape((grid_size, dimension, 3, 3))
        p2_rotational = np.zeros((dimension, 3, 3, nlam, nlam), dtype=complex)
        for mu in range(dimension):
            if basis.momentum_operators:
                momentum2_U = basis.momentum_operators[mu] @ momentum_U[mu]
            else:
                momentum2_U = _apply_momentum(surface, mu, _apply_momentum(surface, mu, U))
            for a in range(3):
                for b in range(3):
                    values = fields[:, mu, a, b]
                    if basis.momentum_operators:
                        weighted_p2 = basis.momentum_operators[mu] @ (
                            basis.momentum_operators[mu] @ (values[:, None] * U)
                        )
                    else:
                        weighted_p2 = _apply_momentum(
                            surface,
                            mu,
                            _apply_momentum(surface, mu, values[:, None] * U),
                        )
                    primitive = 0.5 * (weighted_p2 + values[:, None] * momentum2_U)
                    p2_rotational[mu, a, b] = project(primitive)
    p_rotational_cubic = np.empty((0, 3, nlam, nlam), dtype=complex)
    if surface.p_rotational_cubic_MHz is not None:
        fields = np.asarray(surface.p_rotational_cubic_MHz).reshape((grid_size, dimension, 3))
        p_rotational_cubic = np.zeros((dimension, 3, nlam, nlam), dtype=complex)
        for mu in range(dimension):
            momentum_U = (
                basis.momentum_operators[mu] @ U
                if basis.momentum_operators
                else _apply_momentum(surface, mu, U)
            )
            for a in range(3):
                values = fields[:, mu, a]
                weighted = (
                    basis.momentum_operators[mu] @ (values[:, None] * U)
                    if basis.momentum_operators
                    else _apply_momentum(surface, mu, values[:, None] * U)
                )
                primitive = 0.5 * (weighted + values[:, None] * momentum_U)
                p_rotational_cubic[mu, a] = project(primitive)
    p_rotational_anticommutator = np.empty((0, 3, 3, nlam, nlam), dtype=complex)
    if surface.p_rotational_anticommutator_MHz is not None:
        fields = np.asarray(surface.p_rotational_anticommutator_MHz).reshape(
            (grid_size, dimension, 3, 3)
        )
        p_rotational_anticommutator = np.zeros(
            (dimension, 3, 3, nlam, nlam), dtype=complex
        )
        for mu in range(dimension):
            momentum_U = (
                basis.momentum_operators[mu] @ U
                if basis.momentum_operators
                else _apply_momentum(surface, mu, U)
            )
            for squared_axis in range(3):
                for linear_axis in range(3):
                    values = fields[:, mu, squared_axis, linear_axis]
                    weighted = (
                        basis.momentum_operators[mu] @ (values[:, None] * U)
                        if basis.momentum_operators
                        else _apply_momentum(surface, mu, values[:, None] * U)
                    )
                    primitive = 0.5 * (weighted + values[:, None] * momentum_U)
                    p_rotational_anticommutator[
                        mu, squared_axis, linear_axis
                    ] = project(primitive)
    contracted = {
        "linear": linear,
        "quadratic": quadratic,
        "distortion": distortion,
        "p2_rotational": p2_rotational,
        "p_rotational_cubic": p_rotational_cubic,
        "p_rotational_anticommutator": p_rotational_anticommutator,
    }
    if surface.symmetry_sector_count > 1:
        sectors = np.asarray(basis.symmetry_sectors, dtype=int)
        forbidden = sectors[:, None] != sectors[None, :]
        # FFT translations and floating-point Fourier surfaces leave roundoff
        # matrix elements between exactly distinct product-symmetry sectors.
        # Removing them is the numerical form of the exact symmetry projector
        # and prevents arbitrary mixing of degenerate C_n x C_m partners.
        for values in contracted.values():
            if values.size:
                values[..., forbidden] = 0.0
    return contracted


def _distortion_model_name(
    surface: LAMRotationSurface, parameters: NativeRotorParameters | None
) -> str | None:
    has_surface = (
        surface.quartic_distortion_MHz is not None or surface.sextic_distortion_MHz is not None
    )
    if parameters is not None and not has_surface:
        return f"Watson {parameters.reduction.upper()} quartic+sextic"
    models = []
    if parameters is not None:
        models.append(f"constant Watson {parameters.reduction.upper()} quartic+sextic")
    if has_surface:
        orders = []
        if surface.quartic_distortion_MHz is not None:
            orders.append("quartic")
        if surface.sextic_distortion_MHz is not None:
            orders.append("sextic")
        models.append(
            f"coordinate-dependent Watson {surface.distortion_reduction.upper()} "
            + "+".join(orders)
        )
    if surface.pi2_rotation_MHz is not None and np.any(surface.pi2_rotation_MHz):
        models.append("covariant LAM momentum-squared rotation (J, K, minus)")
    return "; ".join(models) or None


def _parent_assignments(
    J: int,
    vectors: np.ndarray,
    rotational: tuple[np.ndarray, np.ndarray, np.ndarray],
    reference_R: np.ndarray,
    *,
    state_symmetry_sectors: tuple[int, ...] = (),
    lam_symmetry_sectors: tuple[int, ...] = (),
) -> list[tuple[int, int, int, float]]:
    rigid = sum(
        reference_R[a, b] * 0.5 * (rotational[a] @ rotational[b] + rotational[b] @ rotational[a])
        for a in range(3)
        for b in range(3)
    )
    _energies, parents = np.linalg.eigh(_hermitian(rigid))
    if not np.any(reference_R - np.diag(np.diag(reference_R))):
        from .native_rotor import _adiabatic_asymmetric_top_assignments
        # Expectation-value labels can swap Ka families away from the
        # prolate limit. Use the same canonical continuation as the rigid
        # solver before assigning LAM x rotation parent slots.
        parameters = NativeRotorParameters(*np.diag(reference_R))
        labels = [label[:2] for label in _adiabatic_asymmetric_top_assignments(
            parameters,J,parents)]
    else:
        labels = [label[:2] for label in _assign_asymmetric_top_block(
            parents,J,rotational[0]@rotational[0],rotational[2]@rotational[2])]

    overlaps = np.asarray(
        [
            (np.abs(vector.reshape((-1, 2 * J + 1)) @ parents.conj()) ** 2).reshape(-1)
            for vector in vectors.T
        ]
    )
    nlam, nrot = vectors.shape[0] // (2 * J + 1), 2 * J + 1
    selected_slots = np.full(len(vectors.T), -1, dtype=int)
    if state_symmetry_sectors and lam_symmetry_sectors:
        from scipy.optimize import linear_sum_assignment

        if len(state_symmetry_sectors) != len(vectors.T) or len(lam_symmetry_sectors) != nlam:
            raise ValueError("coupled and LAM symmetry-sector dimensions are inconsistent")
        for sector in sorted(set(state_symmetry_sectors)):
            state_indices = np.flatnonzero(np.asarray(state_symmetry_sectors) == sector)
            lam_indices = np.flatnonzero(np.asarray(lam_symmetry_sectors) == sector)
            slots = np.asarray(
                [lam * nrot + rot for lam in lam_indices for rot in range(nrot)], dtype=int
            )
            if len(state_indices) > len(slots):
                raise RuntimeError("more coupled states than reference slots in a symmetry sector")
            rows, columns = linear_sum_assignment(-overlaps[np.ix_(state_indices, slots)])
            selected_slots[state_indices[rows]] = slots[columns]
    else:
        selected_slots = np.argmax(overlaps, axis=1)
    if np.any(selected_slots < 0):
        raise RuntimeError("failed to assign every coupled LAM-rotation state")
    assignments = []
    for state_index, slot in enumerate(selected_slots):
        lam_parent, rot_parent = divmod(int(slot), nrot)
        Ka, Kc = labels[rot_parent]
        assignments.append(
            (
                lam_parent,
                int(Ka),
                int(Kc),
                float(overlaps[state_index, slot]),
            )
        )
    return assignments


def _rotational_operators(J: int, representation: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return angular_momentum_operators(J, representation)


def _fourier_momentum(grid: np.ndarray) -> np.ndarray:
    values = np.asarray(grid, dtype=float)
    spacing = float(values[1] - values[0])
    wave_numbers = 2.0 * np.pi * np.fft.fftfreq(len(values), d=spacing)
    transform = np.fft.fft(np.eye(len(values)), axis=0)
    derivative = np.fft.ifft(1j * wave_numbers[:, None] * transform, axis=0)
    return _hermitian(-1j * derivative)


def _confined_momentum(grid: np.ndarray) -> np.ndarray:
    values = np.asarray(grid, dtype=float)
    spacing = float(values[1] - values[0])
    derivative = np.zeros((len(values), len(values)), dtype=float)
    indices = np.arange(len(values) - 1)
    derivative[indices, indices + 1] = 0.5 / spacing
    derivative[indices + 1, indices] = -0.5 / spacing
    return -1j * derivative


def _hermitian(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix)
    return 0.5 * (values + values.conj().T)


__all__ = [
    "LAMRotationWorkspace",
    "prepare_lam_rotation_workspace",
    "LAM_ROTATION_SURFACE_SCHEMA",
    "LOCK_LAM_ROTATION_RESULT_SCHEMA",
    "LOCK_LAM_ROTATION_SURFACE_SCHEMA",
    "LAMBasis",
    "LAMRotationBlock",
    "LAMRotationLevel",
    "LAMRotationResult",
    "LAMRotationSurface",
    "PermutationInversionSymmetry",
    "lam_rotation_transitions",
    "lock_native_available",
    "primitive_momentum_operators",
    "read_lam_rotation_surface",
    "solve_lam_basis",
    "solve_lam_rotation",
    "write_lam_rotation_surface",
    "write_lam_rotation_result",
]
