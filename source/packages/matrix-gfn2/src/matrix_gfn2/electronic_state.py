"""Immutable resident electronic state for post-SCF MATRIX methods."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _readonly_array(value, *, dtype=float) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True, order="C")
    if not np.all(np.isfinite(array)):
        raise ValueError("electronic-state arrays must be finite")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class GFN2OrbitalBasis:
    """Cartesian primitive export basis and spherical-to-Cartesian map."""

    cartesian_atom: np.ndarray
    cartesian_type: np.ndarray
    primitive_offsets: np.ndarray
    primitive_exponents: np.ndarray
    primitive_coefficients: np.ndarray
    spherical_to_cartesian: np.ndarray

    def __post_init__(self) -> None:
        atom = _readonly_array(self.cartesian_atom, dtype=np.int64)
        ao_type = _readonly_array(self.cartesian_type, dtype=np.int64)
        offsets = _readonly_array(self.primitive_offsets, dtype=np.int64)
        exponents = _readonly_array(self.primitive_exponents)
        coefficients = _readonly_array(self.primitive_coefficients)
        transform = _readonly_array(self.spherical_to_cartesian)
        if atom.ndim != 1 or ao_type.shape != atom.shape:
            raise ValueError("Cartesian AO atoms and types are inconsistent")
        if offsets.shape != (len(atom) + 1,):
            raise ValueError("AO primitive offsets must have length ncartesian + 1")
        if offsets[0] != 0 or np.any(np.diff(offsets) <= 0):
            raise ValueError("every AO must contain at least one primitive")
        if offsets[-1] != len(exponents) or coefficients.shape != exponents.shape:
            raise ValueError("primitive arrays and offsets are inconsistent")
        if np.any(atom < 0) or np.any(ao_type < 1) or np.any(ao_type > 10):
            raise ValueError("invalid Cartesian AO atom or s/p/d component type")
        if np.any(exponents <= 0.0):
            raise ValueError("Gaussian primitive exponents must be positive")
        if transform.ndim != 2 or transform.shape[0] != len(atom):
            raise ValueError("spherical-to-Cartesian transformation is inconsistent")
        object.__setattr__(self, "cartesian_atom", atom)
        object.__setattr__(self, "cartesian_type", ao_type)
        object.__setattr__(self, "primitive_offsets", offsets)
        object.__setattr__(self, "primitive_exponents", exponents)
        object.__setattr__(self, "primitive_coefficients", coefficients)
        object.__setattr__(self, "spherical_to_cartesian", transform)

    @property
    def nao(self) -> int:
        return int(self.spherical_to_cartesian.shape[1])

    @property
    def ncartesian(self) -> int:
        return len(self.cartesian_atom)

    @property
    def nprimitives(self) -> int:
        return len(self.primitive_exponents)


@dataclass(frozen=True)
class GFN2OrbitalState:
    """Canonical restricted-orbital state retained by MATRIX-GFN2.

    Alpha and beta occupations are stored separately even though the current
    GFN2 reference shares one set of spatial orbitals.  This makes the
    reference semantics explicit and prevents a later spin-flip solver from
    silently interpreting the state as UHF.
    """

    model_id: str
    atomic_numbers: np.ndarray
    coordinates_bohr: np.ndarray
    overlap: np.ndarray
    fock: np.ndarray
    coefficients: np.ndarray
    orbital_energies_hartree: np.ndarray
    alpha_occupations: np.ndarray
    beta_occupations: np.ndarray
    ao_atom: np.ndarray
    ao_shell: tuple[str, ...]
    basis: GFN2OrbitalBasis
    total_energy_hartree: float
    electron_count: int
    charge: int
    multiplicity: int

    def __post_init__(self) -> None:
        for name, dtype in (
            ("atomic_numbers", np.int64),
            ("coordinates_bohr", float),
            ("overlap", float),
            ("fock", float),
            ("coefficients", float),
            ("orbital_energies_hartree", float),
            ("alpha_occupations", float),
            ("beta_occupations", float),
            ("ao_atom", np.int64),
        ):
            object.__setattr__(self, name, _readonly_array(getattr(self, name), dtype=dtype))

        numbers = self.atomic_numbers
        coordinates = self.coordinates_bohr
        overlap = self.overlap
        fock = self.fock
        coefficients = self.coefficients
        energies = self.orbital_energies_hartree
        alpha = self.alpha_occupations
        beta = self.beta_occupations
        nao = overlap.shape[0] if overlap.ndim == 2 else -1
        nmo = coefficients.shape[1] if coefficients.ndim == 2 else -1
        if numbers.ndim != 1 or coordinates.shape != (len(numbers), 3):
            raise ValueError("atomic numbers and coordinates are inconsistent")
        if overlap.shape != (nao, nao) or fock.shape != (nao, nao):
            raise ValueError("overlap and Fock matrices must be square and equally sized")
        if coefficients.shape != (nao, nmo):
            raise ValueError("orbital coefficients must have shape nao x nmo")
        if energies.shape != (nmo,) or alpha.shape != (nmo,) or beta.shape != (nmo,):
            raise ValueError("orbital energies and spin occupations must have length nmo")
        if self.ao_atom.shape != (nao,) or len(self.ao_shell) != nao:
            raise ValueError("AO metadata must have length nao")
        if self.basis.nao != nao:
            raise ValueError("primitive basis does not match the AO dimension")
        if np.any(self.ao_atom < 0) or np.any(self.ao_atom >= len(numbers)):
            raise ValueError("AO-to-atom indices are outside the molecule")
        if np.any(alpha < -1.0e-12) or np.any(alpha > 1.0 + 1.0e-12):
            raise ValueError("alpha occupations must lie in [0, 1]")
        if np.any(beta < -1.0e-12) or np.any(beta > 1.0 + 1.0e-12):
            raise ValueError("beta occupations must lie in [0, 1]")
        if int(self.multiplicity) < 1:
            raise ValueError("multiplicity must be positive")
        if not np.allclose(overlap, overlap.T, atol=2.0e-11, rtol=0.0):
            raise ValueError("overlap matrix must be symmetric")
        if not np.allclose(fock, fock.T, atol=2.0e-11, rtol=0.0):
            raise ValueError("Fock matrix must be symmetric")
        metric = coefficients.T @ overlap @ coefficients
        if not np.allclose(metric, np.eye(nmo), atol=2.0e-7, rtol=2.0e-7):
            raise ValueError("canonical orbitals are not orthonormal in the AO metric")
        eigen_residual = fock @ coefficients - (
            overlap @ coefficients
        ) * energies
        if float(np.max(np.abs(eigen_residual), initial=0.0)) > 2.0e-7:
            raise ValueError("orbital state does not satisfy the generalized eigenproblem")
        occupied = float(np.sum(alpha) + np.sum(beta))
        if abs(occupied - int(self.electron_count)) > 2.0e-8:
            raise ValueError("spin occupations do not reproduce the molecular electron count")
        spin_excess = float(np.sum(alpha) - np.sum(beta))
        if abs(spin_excess - (int(self.multiplicity) - 1)) > 2.0e-8:
            raise ValueError("spin occupations do not reproduce the requested multiplicity")

    @property
    def nao(self) -> int:
        return int(self.overlap.shape[0])

    @property
    def nmo(self) -> int:
        return int(self.coefficients.shape[1])

    @property
    def alpha_electrons(self) -> float:
        return float(np.sum(self.alpha_occupations))

    @property
    def beta_electrons(self) -> float:
        return float(np.sum(self.beta_occupations))

    @property
    def reference_kind(self) -> str:
        return "restricted-open-shell" if self.multiplicity > 1 else "restricted-closed-shell"


@dataclass(frozen=True)
class GFN2SparseProjectorState:
    """Pair-packed occupied projector and Fock retained by sparse SCF."""

    size: int
    occupied_orbitals: int
    projector_rows: np.ndarray
    projector_columns: np.ndarray
    projector_values: np.ndarray
    fock_rows: np.ndarray
    fock_columns: np.ndarray
    fock_values: np.ndarray
    domain_rows: np.ndarray
    domain_columns: np.ndarray
    ao_atom: np.ndarray
    atomic_numbers: np.ndarray
    coordinates_bohr: np.ndarray
    truncation_bound: float

    def __post_init__(self) -> None:
        for name, dtype in (
            ("projector_rows", np.int64),
            ("projector_columns", np.int64),
            ("projector_values", float),
            ("fock_rows", np.int64),
            ("fock_columns", np.int64),
            ("fock_values", float),
            ("domain_rows", np.int64),
            ("domain_columns", np.int64),
            ("ao_atom", np.int64),
            ("atomic_numbers", np.int64),
            ("coordinates_bohr", float),
        ):
            object.__setattr__(self, name, _readonly_array(getattr(self, name), dtype=dtype))
        if int(self.size) < 1 or self.ao_atom.shape != (int(self.size),):
            raise ValueError("sparse electronic-state dimension is inconsistent")
        if not 0 < int(self.occupied_orbitals) < int(self.size):
            raise ValueError("sparse occupied dimension must lie inside the AO space")
        if self.projector_rows.shape != self.projector_columns.shape or (
            self.projector_values.shape != self.projector_rows.shape
        ):
            raise ValueError("sparse projector arrays are inconsistent")
        if self.fock_rows.shape != self.fock_columns.shape or (
            self.fock_values.shape != self.fock_rows.shape
        ):
            raise ValueError("sparse Fock arrays are inconsistent")
        if self.domain_rows.shape != self.domain_columns.shape:
            raise ValueError("sparse domain arrays are inconsistent")
        if self.coordinates_bohr.shape != (len(self.atomic_numbers), 3):
            raise ValueError("sparse-state geometry is inconsistent")
        if np.any(self.ao_atom < 0) or np.any(self.ao_atom >= len(self.atomic_numbers)):
            raise ValueError("sparse-state AO assignment is invalid")
        if not np.isfinite(self.truncation_bound) or self.truncation_bound < 0.0:
            raise ValueError("sparse-state truncation bound must be finite and nonnegative")


__all__ = ["GFN2OrbitalBasis", "GFN2OrbitalState", "GFN2SparseProjectorState"]
