"""Parameterized Cartesian basis and real-spherical transformation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .basis import ContractedGaussian, SparseAtomicMultipoleIntegrals
from .locality import LocalizedOrbitalDomains
from .parameters import GFN2ElementParameters, GFN2ParameterSet
from .sto import shell_quantum_numbers, slater_to_gaussians

try:
    from . import _gfn2_native
except ImportError:  # pragma: no cover
    _gfn2_native = None


_CARTESIAN = {
    0: ((0, 0, 0),),
    1: ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    2: ((2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)),
}


@dataclass(frozen=True)
class MolecularBasis:
    cartesian_orbitals: tuple[ContractedGaussian, ...]
    _cartesian_to_spherical: np.ndarray | None
    spherical_atom: tuple[int, ...]
    spherical_shell: tuple[str, ...]
    shell_reference_occupations: tuple[float, ...]
    spherical_cartesian_offsets: np.ndarray
    spherical_cartesian_indices: np.ndarray
    spherical_cartesian_coefficients: np.ndarray

    @property
    def norbitals(self) -> int:
        return len(self.spherical_cartesian_offsets) - 1

    @property
    def cartesian_to_spherical(self) -> np.ndarray:
        """Materialize the legacy dense transform only for a dense caller."""

        cached = self._cartesian_to_spherical
        if cached is None:
            cached = np.zeros((len(self.cartesian_orbitals), self.norbitals))
            for spherical in range(self.norbitals):
                start = self.spherical_cartesian_offsets[spherical]
                stop = self.spherical_cartesian_offsets[spherical + 1]
                cached[
                    self.spherical_cartesian_indices[start:stop], spherical
                ] = self.spherical_cartesian_coefficients[start:stop]
            object.__setattr__(self, "_cartesian_to_spherical", cached)
        return cached

    def transform_matrix(self, cartesian_matrix: np.ndarray) -> np.ndarray:
        transform = self.cartesian_to_spherical
        return transform.T @ np.asarray(cartesian_matrix, dtype=float) @ transform

    def transform_sparse_multipoles(
        self,
        cartesian: SparseAtomicMultipoleIntegrals,
        domains: LocalizedOrbitalDomains,
        *,
        threshold: float = 0.0,
    ) -> SparseAtomicMultipoleIntegrals:
        """Apply the atom-local real-spherical congruence in native sparse form."""

        if _gfn2_native is None or not hasattr(
            _gfn2_native, "sparse_multipole_congruence"
        ):
            raise RuntimeError("sparse spherical transformation requires native kernels")
        if cartesian.size != len(self.cartesian_orbitals):
            raise ValueError("Cartesian integral dimension does not match basis")
        if domains.norbitals != self.norbitals:
            raise ValueError("spherical domains do not match basis")
        rows, columns = domains.allowed_pair_arrays()
        values = np.column_stack(
            (cartesian.overlap, cartesian.dipole, cartesian.second.reshape(-1, 9))
        )
        transformed, left_derivatives, right_derivatives = (
            _gfn2_native.sparse_multipole_congruence(
                int(cartesian.size),
                cartesian.rows,
                cartesian.columns,
                values,
                cartesian.left_derivatives,
                cartesian.right_derivatives,
                self.spherical_cartesian_offsets,
                self.spherical_cartesian_indices,
                self.spherical_cartesian_coefficients,
                rows,
                columns,
            )
        )
        transformed = np.asarray(transformed, dtype=float)
        diagonal = rows == columns
        magnitude = np.max(np.abs(transformed), axis=1) if len(rows) else np.empty(0)
        keep = diagonal | (magnitude >= float(threshold))
        omitted = transformed[~keep, 0]
        discarded = float(np.sqrt(2.0 * np.sum(omitted * omitted)))
        kept = transformed[keep]
        return SparseAtomicMultipoleIntegrals(
            size=self.norbitals,
            rows=np.asarray(rows[keep], dtype=np.int32),
            columns=np.asarray(columns[keep], dtype=np.int32),
            overlap=kept[:, 0],
            dipole=kept[:, 1:4],
            second=kept[:, 4:].reshape(-1, 3, 3),
            left_derivatives=np.asarray(left_derivatives, dtype=float)[keep],
            right_derivatives=np.asarray(right_derivatives, dtype=float)[keep],
            threshold_discarded_frobenius_norm=(
                cartesian.threshold_discarded_frobenius_norm + discarded
            ),
            domain_complete=(
                cartesian.domain_complete
                and len(rows) == self.norbitals * (self.norbitals + 1) // 2
            ),
        )

    def evaluate_sparse_multipoles(
        self,
        native_context,
        domains: LocalizedOrbitalDomains,
        *,
        origin_bohr: Sequence[float] = (0.0, 0.0, 0.0),
        threshold: float = 0.0,
        native_threads: int = 1,
        centers_bohr: np.ndarray | None = None,
    ) -> SparseAtomicMultipoleIntegrals:
        """Evaluate and transform AO multipoles in one persistent native call."""

        if native_context.size != len(self.cartesian_orbitals):
            raise ValueError("persistent integral context does not match basis")
        if domains.norbitals != self.norbitals:
            raise ValueError("spherical domains do not match basis")
        rows, columns = domains.allowed_pair_arrays()
        native_context.configure_spherical(
            self.spherical_cartesian_offsets,
            self.spherical_cartesian_indices,
            self.spherical_cartesian_coefficients,
            rows,
            columns,
        )
        centers = (
            np.asarray(
                [item.center_bohr for item in self.cartesian_orbitals], dtype=float
            )
            if centers_bohr is None
            else np.asarray(centers_bohr, dtype=float)
        )
        if centers.shape != (len(self.cartesian_orbitals), 3):
            raise ValueError("Cartesian centers do not match molecular basis")
        transformed, left_derivatives, right_derivatives = (
            native_context.evaluate_spherical(
                centers,
                np.asarray(origin_bohr, dtype=float),
                native_threads,
            )
        )
        transformed = np.asarray(transformed, dtype=float)
        left_derivatives = np.asarray(left_derivatives, dtype=float)
        right_derivatives = np.asarray(right_derivatives, dtype=float)
        diagonal = rows == columns
        magnitude = np.max(np.abs(transformed), axis=1) if len(rows) else np.empty(0)
        keep = diagonal | (magnitude >= float(threshold))
        omitted = transformed[~keep, 0]
        discarded = float(np.sqrt(2.0 * np.sum(omitted * omitted)))
        kept = transformed[keep]
        return SparseAtomicMultipoleIntegrals(
            size=self.norbitals,
            rows=np.asarray(rows[keep], dtype=np.int32),
            columns=np.asarray(columns[keep], dtype=np.int32),
            overlap=kept[:, 0],
            dipole=kept[:, 1:4],
            second=kept[:, 4:].reshape(-1, 3, 3),
            left_derivatives=left_derivatives[keep],
            right_derivatives=right_derivatives[keep],
            threshold_discarded_frobenius_norm=discarded,
            domain_complete=(
                len(native_context.pair_left)
                == native_context.size * (native_context.size + 1) // 2
                and len(rows) == self.norbitals * (self.norbitals + 1) // 2
            ),
        )


def build_molecular_basis(
    atomic_numbers: Sequence[int],
    coordinates_bohr: np.ndarray,
    parameters: GFN2ParameterSet,
    *,
    per_atom_elements: Sequence[GFN2ElementParameters] | None = None,
) -> MolecularBasis:
    coordinates = np.asarray(coordinates_bohr, dtype=float)
    if per_atom_elements is not None:
        if len(per_atom_elements) != len(atomic_numbers):
            raise ValueError("per-atom element count differs from atom count")
        if any(element.atomic_number != int(number)
               for element, number in zip(per_atom_elements, atomic_numbers, strict=True)):
            raise ValueError("per-atom element atomic numbers differ from molecular atoms")
    cartesian: list[ContractedGaussian] = []
    spherical_atom: list[int] = []
    spherical_shell: list[str] = []
    occupations: list[float] = []
    spherical_cartesian_offsets = [0]
    spherical_cartesian_indices: list[int] = []
    spherical_cartesian_coefficients: list[float] = []
    for atom, atomic_number in enumerate(atomic_numbers):
        element = (
            parameters.elements[int(atomic_number)]
            if per_atom_elements is None else per_atom_elements[atom]
        )
        for shell in element.shells:
            principal, angular = shell_quantum_numbers(shell.label)
            primitives = slater_to_gaussians(
                shell.gaussian_count, principal, angular, shell.slater_exponent
            )
            start = len(cartesian)
            for powers in _CARTESIAN[angular]:
                cartesian.append(
                    ContractedGaussian(
                        center_bohr=tuple(coordinates[atom]),
                        angular=powers,
                        primitives=primitives,
                        atom_index=atom,
                        shell=shell.label,
                        slater_exponent=shell.slater_exponent,
                    )
                )
            block = len(_CARTESIAN[angular])
            if angular < 2:
                local = np.eye(block)
            else:
                # Cartesian order xx, yy, zz, xy, xz, yz; columns are
                # z2, xz, yz, x2-y2, xy real solid harmonics.
                local = np.zeros((6, 5))
                local[:3, 0] = (-0.5, -0.5, 1.0)
                local[4, 1] = 1.0
                local[5, 2] = 1.0
                local[:2, 3] = (np.sqrt(3.0) / 2.0, -np.sqrt(3.0) / 2.0)
                local[3, 4] = 1.0
            for column in range(local.shape[1]):
                spherical_atom.append(atom)
                spherical_shell.append(shell.label)
                nonzero = np.flatnonzero(local[:, column])
                spherical_cartesian_indices.extend(
                    int(start + value) for value in nonzero
                )
                spherical_cartesian_coefficients.extend(
                    float(local[value, column]) for value in nonzero
                )
                spherical_cartesian_offsets.append(
                    len(spherical_cartesian_indices)
                )
            occupations.append(shell.reference_occupation)
    return MolecularBasis(
        tuple(cartesian),
        None,
        tuple(spherical_atom),
        tuple(spherical_shell),
        tuple(occupations),
        np.asarray(spherical_cartesian_offsets, dtype=np.int64),
        np.asarray(spherical_cartesian_indices, dtype=np.int64),
        np.asarray(spherical_cartesian_coefficients, dtype=float),
    )


def relocate_molecular_basis(
    template: MolecularBasis,
    coordinates_bohr: np.ndarray,
) -> MolecularBasis:
    """Move a topology-fixed basis without rebuilding STO contractions."""

    coordinates = np.asarray(coordinates_bohr, dtype=float)
    natoms = max((item.atom_index for item in template.cartesian_orbitals), default=-1) + 1
    if coordinates.shape != (natoms, 3) or not np.all(np.isfinite(coordinates)):
        raise ValueError("coordinates do not match the molecular basis template")
    orbitals = tuple(
        ContractedGaussian(
            center_bohr=tuple(coordinates[item.atom_index]),
            angular=item.angular,
            primitives=item.primitives,
            atom_index=item.atom_index,
            shell=item.shell,
            slater_exponent=item.slater_exponent,
        )
        for item in template.cartesian_orbitals
    )
    return MolecularBasis(
        orbitals,
        template._cartesian_to_spherical,
        template.spherical_atom,
        template.spherical_shell,
        template.shell_reference_occupations,
        template.spherical_cartesian_offsets,
        template.spherical_cartesian_indices,
        template.spherical_cartesian_coefficients,
    )


__all__ = ["MolecularBasis", "build_molecular_basis", "relocate_molecular_basis"]
