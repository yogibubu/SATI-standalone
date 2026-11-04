"""Frozen ORACLE point groups and symmetry-adapted MATRIX-GFN2 AO blocks."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .molecular_basis import MolecularBasis
from .scc import solve_generalized_symmetric

DEFAULT_SYMMETRY_SMALL_SYSTEM_HOST_THRESHOLD = 128
DEFAULT_SYMMETRY_BLOCK_TOLERANCE = 5.0e-10


def _batched_generalized_eigh(
    fock: np.ndarray,
    overlap: np.ndarray,
    backend: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve equal-sized generalized eigenproblems on one accelerator batch."""

    normalized = str(backend).strip().lower()
    if normalized == "cupy":
        import cupy as cp

        xp = cp
    elif normalized in {"cuda", "torch-cuda", "mps", "torch-mps"}:
        import torch

        device = "mps" if normalized in {"mps", "torch-mps"} else "cuda"
        dtype = torch.float32 if device == "mps" else torch.float64
        overlap_device = torch.as_tensor(overlap, dtype=dtype, device=device)
        fock_device = torch.as_tensor(fock, dtype=dtype, device=device)
        overlap_values, overlap_vectors = torch.linalg.eigh(overlap_device)
        if torch.any(overlap_values <= 1.0e-10):
            raise np.linalg.LinAlgError("symmetry overlap block is not positive definite")
        inverse_root = (
            overlap_vectors
            * torch.rsqrt(overlap_values)[..., None, :]
        ) @ torch.swapaxes(overlap_vectors, -1, -2)
        orthogonal_fock = inverse_root @ fock_device @ inverse_root
        energies, vectors = torch.linalg.eigh(orthogonal_fock)
        coefficients = inverse_root @ vectors
        return (
            energies.detach().cpu().numpy().astype(np.float64),
            coefficients.detach().cpu().numpy().astype(np.float64),
        )
    else:
        raise ValueError(f"unsupported batched symmetry backend: {backend}")
    overlap_device = xp.asarray(overlap)
    fock_device = xp.asarray(fock)
    overlap_values, overlap_vectors = xp.linalg.eigh(overlap_device)
    if bool(xp.any(overlap_values <= 1.0e-10)):
        raise np.linalg.LinAlgError("symmetry overlap block is not positive definite")
    inverse_root = (
        overlap_vectors * (1.0 / xp.sqrt(overlap_values))[..., None, :]
    ) @ xp.swapaxes(overlap_vectors, -1, -2)
    orthogonal_fock = inverse_root @ fock_device @ inverse_root
    energies, vectors = xp.linalg.eigh(orthogonal_fock)
    coefficients = inverse_root @ vectors
    return (
        cp.asnumpy(energies).astype(np.float64),
        cp.asnumpy(coefficients).astype(np.float64),
    )


@dataclass(frozen=True)
class FrozenSymmetryOperation:
    """One immutable ORACLE operation in the input Cartesian frame."""

    label: str
    rotation: tuple[tuple[float, float, float], ...]
    permutation: tuple[int, ...]


@dataclass(frozen=True)
class FrozenPointGroup:
    """ORACLE-to-MATRIX-GFN2 symmetry contract.

    Permutations are zero based and map each source atom onto its target atom.
    """

    point_group: str
    atom_symbols: tuple[str, ...]
    operations: tuple[FrozenSymmetryOperation, ...]
    atom_orbits: tuple[tuple[int, ...], ...]
    geometry_tolerance_angstrom: float
    digest: str
    schema: str = "matrix.oracle-gfn2.frozen-point-group.v1"

    @classmethod
    def create(
        cls,
        *,
        point_group: str,
        atom_symbols: Sequence[str],
        operations: Sequence[FrozenSymmetryOperation],
        atom_orbits: Sequence[Sequence[int]],
        geometry_tolerance_angstrom: float,
    ) -> "FrozenPointGroup":
        symbols = tuple(str(value) for value in atom_symbols)
        frozen_operations = tuple(operations)
        frozen_orbits = tuple(tuple(int(index) for index in orbit) for orbit in atom_orbits)
        payload = {
            "point_group": str(point_group).strip().upper() or "C1",
            "atom_symbols": symbols,
            "operations": [
                {
                    "label": operation.label,
                    "rotation": operation.rotation,
                    "permutation": operation.permutation,
                }
                for operation in frozen_operations
            ],
            "atom_orbits": frozen_orbits,
            "geometry_tolerance_angstrom": float(geometry_tolerance_angstrom),
        }
        digest = sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return cls(
            payload["point_group"],
            symbols,
            frozen_operations,
            frozen_orbits,
            float(geometry_tolerance_angstrom),
            digest,
        )

    @property
    def active(self) -> bool:
        return self.point_group not in {"", "C1", "UNKNOWN"} and len(self.operations) > 1

    def validate(self) -> None:
        natoms = len(self.atom_symbols)
        if self.geometry_tolerance_angstrom <= 0.0:
            raise ValueError("frozen symmetry geometry tolerance must be positive")
        covered = sorted(index for orbit in self.atom_orbits for index in orbit)
        if covered != list(range(natoms)):
            raise ValueError("frozen symmetry atom orbits must partition all atoms")
        for operation in self.operations:
            rotation = np.asarray(operation.rotation, dtype=float)
            if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
                raise ValueError("frozen symmetry rotations must be finite 3-by-3 matrices")
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2.0e-10):
                raise ValueError("frozen symmetry rotations must be orthogonal")
            if sorted(operation.permutation) != list(range(natoms)):
                raise ValueError("frozen symmetry operations must contain atom permutations")
            for source, target in enumerate(operation.permutation):
                if self.atom_symbols[source] != self.atom_symbols[target]:
                    raise ValueError("frozen symmetry operation changes an atomic element")


@dataclass(frozen=True)
class SymmetryAdaptedBasis:
    """Compiled AO representation and mutually uncoupled SALC blocks."""

    contract: FrozenPointGroup
    atom_orbits: tuple[tuple[int, ...], ...]
    shell_orbits: tuple[tuple[int, ...], ...]
    representations: tuple[np.ndarray, ...]
    transformation: np.ndarray
    blocks: tuple[np.ndarray, ...]
    block_labels: tuple[str, ...]
    overlap_blocks: tuple[np.ndarray, ...]
    overlap_invariance_error: float
    geometry_error_angstrom: float
    overlap_inverse_roots: tuple[np.ndarray, ...] = ()
    block_slices: tuple[tuple[slice, slice], ...] = ()

    @property
    def block_dimensions(self) -> tuple[int, ...]:
        return tuple(len(block) for block in self.blocks)

    @property
    def block_batches(self) -> tuple[tuple[int, tuple[int, ...]], ...]:
        """Return equal-sized block batches suitable for CPU/GPU execution."""

        grouped: dict[int, list[int]] = {}
        for index, block in enumerate(self.blocks):
            grouped.setdefault(len(block), []).append(index)
        return tuple((size, tuple(indices)) for size, indices in sorted(grouped.items()))

    def symmetrize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix, dtype=float)
        adapted = self.transformation.T @ values @ self.transformation
        retained = np.zeros_like(adapted)
        for block in self.blocks:
            retained[np.ix_(block, block)] = adapted[np.ix_(block, block)]
        averaged = self.transformation @ retained @ self.transformation.T
        return 0.5 * (averaged + averaged.T)

    def to_blocks(self, matrix: np.ndarray, *, symmetrize: bool = True) -> "SymmetryBlockedMatrix":
        """Project an AO matrix directly onto the retained irrep blocks."""

        values = np.asarray(matrix, dtype=float)
        if values.shape != self.transformation.shape:
            raise ValueError("symmetry-blocked matrix has the wrong AO dimension")
        adapted = self.transformation.T @ values @ self.transformation
        if not symmetrize and self.off_block_error(values) > 5.0e-10:
            raise ValueError("unprojected matrix contains nonzero off-block elements")
        values_by_block = tuple(
            adapted[rows, columns]
            for rows, columns in self.block_slices
        ) if self.block_slices else tuple(
            adapted[np.ix_(block, block)] for block in self.blocks
        )
        return SymmetryBlockedMatrix(
            labels=self.block_labels,
            indices=self.blocks,
            values=values_by_block,
        )

    def from_blocks(self, blocked: "SymmetryBlockedMatrix") -> np.ndarray:
        """Expand an irrep-blocked matrix back into the complete AO frame."""

        if blocked.labels != self.block_labels or len(blocked.values) != len(self.blocks):
            raise ValueError("symmetry-block labels do not match the compiled AO basis")
        adapted = np.zeros_like(self.transformation)
        for block_index, (indices, values) in enumerate(
            zip(self.blocks, blocked.values, strict=True)
        ):
            if values.shape != (len(indices), len(indices)):
                raise ValueError("symmetry block has the wrong dimension")
            if self.block_slices:
                rows, columns = self.block_slices[block_index]
                adapted[rows, columns] = values
            else:
                adapted[np.ix_(indices, indices)] = values
        return self.transformation @ adapted @ self.transformation.T

    def commutator_error(self, matrix: np.ndarray) -> float:
        """Maximum normalized point-group commutator residual."""

        values = np.asarray(matrix, dtype=float)
        scale = max(1.0, float(np.max(np.abs(values), initial=0.0)))
        return max(
            float(np.max(np.abs(values @ rep - rep @ values), initial=0.0)) / scale
            for rep in self.representations
        )

    def off_block_error(self, matrix: np.ndarray) -> float:
        adapted = self.transformation.T @ np.asarray(matrix, dtype=float) @ self.transformation
        retained = np.zeros_like(adapted, dtype=bool)
        for block in self.blocks:
            retained[np.ix_(block, block)] = True
        return float(np.max(np.abs(adapted[~retained]), initial=0.0))

    def solve(
        self,
        fock: np.ndarray,
        *,
        diagonalizer_backend: str | None = None,
        small_system_host_threshold: int = DEFAULT_SYMMETRY_SMALL_SYSTEM_HOST_THRESHOLD,
        reference_coefficients: np.ndarray | None = None,
        degeneracy_tolerance: float = 1.0e-9,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve independent generalized eigenproblems in every SALC block."""

        fock_blocks = self.to_blocks(fock).values
        # MPS launch/synchronization latency dominates tiny irreps.  Keep the
        # symmetry-adapted representation and all block algebra, but execute
        # the small eigensolves in the host BLAS path.  This is an execution
        # policy only; it never bypasses orbital assignment or symmetry
        # expansion of the result.
        solve_backend = diagonalizer_backend
        if (
            str(diagonalizer_backend).lower() in {"mps", "torch-mps"}
            and self.transformation.shape[0] < int(small_system_host_threshold)
        ):
            solve_backend = None
        solved: list[tuple[np.ndarray, np.ndarray] | None] = [None] * len(self.blocks)
        for _, batch in self.block_batches:
            if len(batch) > 1:
                stacked_overlap = np.stack([self.overlap_blocks[index] for index in batch])
                stacked_fock = np.stack([fock_blocks[index] for index in batch])
                if solve_backend is None and self.overlap_inverse_roots:
                    inverse_root = np.stack(
                        [self.overlap_inverse_roots[index] for index in batch]
                    )
                    orthogonal_fock = inverse_root @ stacked_fock @ inverse_root
                    batch_energies, batch_vectors = np.linalg.eigh(orthogonal_fock)
                    batch_coefficients = inverse_root @ batch_vectors
                else:
                    try:
                        batch_energies, batch_coefficients = _batched_generalized_eigh(
                            stacked_fock, stacked_overlap, solve_backend
                        )
                    except (
                        ImportError,
                        RuntimeError,
                        TypeError,
                        ValueError,
                        np.linalg.LinAlgError,
                    ):
                        batch_energies = batch_coefficients = None
                if batch_energies is None or batch_coefficients is None:
                    for index in batch:
                        root = self.overlap_inverse_roots[index]
                        values, vectors = np.linalg.eigh(
                            root @ fock_blocks[index] @ root
                        )
                        solved[index] = (values, root @ vectors)
                    continue
                for local, index in enumerate(batch):
                    solved[index] = (batch_energies[local], batch_coefficients[local])
            else:
                for index in batch:
                    root = self.overlap_inverse_roots[index]
                    values, vectors = np.linalg.eigh(root @ fock_blocks[index] @ root)
                    solved[index] = (values, root @ vectors)
        energies: list[np.ndarray] = []
        adapted_coefficients = np.zeros_like(self.transformation)
        for indices, solution in zip(self.blocks, solved, strict=True):
            assert solution is not None
            block_energies, block_coefficients = solution
            energies.append(block_energies)
            adapted_coefficients[np.ix_(indices, indices)] = block_coefficients
        all_energies = np.concatenate(energies)
        all_coefficients = self.transformation @ adapted_coefficients
        order = np.argsort(all_energies, kind="stable")
        ordered_coefficients = all_coefficients[:, order].copy()
        # Eigenvectors are defined up to a sign.  Fixing the phase by the
        # largest-magnitude AO component makes repeated solves and serialized
        # restarts deterministic without changing the density.
        for column in range(ordered_coefficients.shape[1]):
            pivot = int(np.argmax(np.abs(ordered_coefficients[:, column])))
            if ordered_coefficients[pivot, column] < 0.0:
                ordered_coefficients[:, column] *= -1.0
        if reference_coefficients is not None:
            reference = np.asarray(reference_coefficients, dtype=float)
            if reference.shape == ordered_coefficients.shape:
                overlap = self.from_blocks(
                    SymmetryBlockedMatrix(
                        self.block_labels, self.blocks, self.overlap_blocks
                    )
                )
                start = 0
                while start < len(ordered_energies := all_energies[order]):
                    stop = start + 1
                    while (
                        stop < len(ordered_energies)
                        and abs(ordered_energies[stop] - ordered_energies[start])
                        <= degeneracy_tolerance
                    ):
                        stop += 1
                    if stop - start > 1:
                        match = reference[:, start:stop].T @ overlap @ ordered_coefficients[:, start:stop]
                        left, _, right_transpose = np.linalg.svd(match)
                        ordered_coefficients[:, start:stop] = (
                            ordered_coefficients[:, start:stop]
                            @ right_transpose.T
                            @ left.T
                        )
                    start = stop
                for column in range(ordered_coefficients.shape[1]):
                    pivot = int(np.argmax(np.abs(ordered_coefficients[:, column])))
                    if ordered_coefficients[pivot, column] < 0.0:
                        ordered_coefficients[:, column] *= -1.0
        return all_energies[order], ordered_coefficients

    def irrep_degeneracy_error(self, matrix: np.ndarray) -> float:
        """Return the largest forbidden splitting inside multidimensional irreps."""

        dimensions = {"A1": 1, "A2": 1, "B1": 1, "B2": 1, "E": 2, "T1": 3, "T2": 3}
        blocked = self.to_blocks(matrix).values
        error = 0.0
        row_spectra: dict[str, list[np.ndarray]] = {}
        for label, fock_block, overlap_block in zip(
            self.block_labels, blocked, self.overlap_blocks, strict=True
        ):
            base_label = label.split(".", 1)[0]
            energies, _ = solve_generalized_symmetric(fock_block, overlap_block)
            if "." in label:
                row_spectra.setdefault(base_label, []).append(energies)
                continue
            dimension = dimensions.get(base_label, 1)
            if dimension == 1:
                continue
            if len(energies) % dimension:
                return float("inf")
            for start in range(0, len(energies), dimension):
                group = energies[start : start + dimension]
                error = max(error, float(np.max(group) - np.min(group)))
        for spectra in row_spectra.values():
            reference = spectra[0]
            if any(values.shape != reference.shape for values in spectra[1:]):
                return float("inf")
            error = max(
                error,
                max(
                    (float(np.max(np.abs(values - reference), initial=0.0)) for values in spectra[1:]),
                    default=0.0,
                ),
            )
        return error

    def project_geometry(self, coordinates_angstrom: np.ndarray) -> np.ndarray:
        """Reconstruct the closest geometry in the frozen atom-orbit manifold."""

        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        if coordinates.shape != (len(self.atom_orbits_flat), 3):
            raise ValueError("geometry projection requires natoms x 3 coordinates")
        center = np.mean(coordinates, axis=0)
        centered = coordinates - center
        projected = np.zeros_like(centered)
        for operation in self.contract.operations:
            mapping = np.asarray(operation.permutation, dtype=np.intp)
            rotation = np.asarray(operation.rotation, dtype=float)
            projected += centered[np.argsort(mapping)] @ rotation
        return projected / float(len(self.contract.operations)) + center

    @property
    def atom_orbits_flat(self) -> tuple[int, ...]:
        return tuple(index for orbit in self.atom_orbits for index in orbit)

    def cartesian_representations(self) -> tuple[np.ndarray, ...]:
        """Return polar-vector representations in the complete Cartesian frame."""

        natoms = len(self.contract.atom_symbols)
        result = []
        for operation in self.contract.operations:
            matrix = np.zeros((3 * natoms, 3 * natoms))
            rotation = np.asarray(operation.rotation, dtype=float)
            for source, target in enumerate(operation.permutation):
                matrix[3 * target : 3 * target + 3, 3 * source : 3 * source + 3] = rotation.T
            result.append(matrix)
        return tuple(result)

    def symmetrize_hessian(self, hessian: np.ndarray) -> np.ndarray:
        """Project a Cartesian Hessian onto the frozen point group."""

        values = np.asarray(hessian, dtype=float)
        expected = 3 * len(self.contract.atom_symbols)
        if values.shape != (expected, expected):
            raise ValueError("Cartesian Hessian has the wrong dimension")
        averaged = sum(rep.T @ values @ rep for rep in self.cartesian_representations())
        averaged /= float(len(self.contract.operations))
        return 0.5 * (averaged + averaged.T)

    def classify_cartesian_modes(
        self,
        hessian: np.ndarray,
        masses_amu: Sequence[float],
    ) -> tuple["SymmetryMode", ...]:
        """Diagonalize a Hessian and label every mass-weighted mode by irrep."""

        masses = np.repeat(np.asarray(tuple(masses_amu), dtype=float), 3)
        if masses.shape != (3 * len(self.contract.atom_symbols),) or np.any(masses <= 0.0):
            raise ValueError("mode classification requires one positive mass per atom")
        symmetric = self.symmetrize_hessian(hessian)
        inverse_mass = 1.0 / np.sqrt(masses)
        weighted = inverse_mass[:, None] * symmetric * inverse_mass[None, :]
        transformation, blocks, labels = self.cartesian_blocks()
        result = []
        for label, block in zip(labels, blocks, strict=True):
            basis = transformation[:, block]
            eigenvalues, vectors = np.linalg.eigh(basis.T @ weighted @ basis)
            for value, vector in zip(eigenvalues, vectors.T, strict=True):
                result.append(SymmetryMode(0, float(value), label, 1.0, basis @ vector))
        result.sort(key=lambda item: item.eigenvalue)
        return tuple(
            SymmetryMode(index, item.eigenvalue, item.irrep, item.irrep_weight, item.vector)
            for index, item in enumerate(result)
        )

    def cartesian_blocks(
        self,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[str, ...]]:
        """Return the Cartesian SALC transformation, blocks, and irrep labels."""

        representations = self.cartesian_representations()
        characterized = _character_salc_blocks(
            self.contract.point_group, self.contract.operations, representations
        )
        if characterized is None:
            return _salc_blocks(representations)
        return characterized

    def label_cartesian_vectors(
        self, vectors: np.ndarray
    ) -> tuple[tuple[str, float], ...]:
        """Assign normalized Cartesian vectors to frozen-group irreps."""

        values = np.asarray(vectors, dtype=float)
        if values.ndim == 1:
            values = values[None, :]
        expected = 3 * len(self.contract.atom_symbols)
        if values.ndim != 2 or values.shape[1] != expected:
            raise ValueError("Cartesian mode vectors have the wrong dimension")
        transformation, blocks, labels = self.cartesian_blocks()
        projectors = tuple(
            transformation[:, block] @ transformation[:, block].T for block in blocks
        )
        result = []
        for vector in values:
            norm = max(float(vector @ vector), np.finfo(float).tiny)
            weights = tuple(float(vector @ projector @ vector) / norm for projector in projectors)
            selected = int(np.argmax(weights))
            result.append((labels[selected], weights[selected]))
        return tuple(result)

    def geometry_manifold(self, reference_coordinates_angstrom: np.ndarray) -> "FrozenGeometryManifold":
        """Compile minimal totally-symmetric Cartesian coordinates for optimization."""

        reference = self.project_geometry(reference_coordinates_angstrom)
        projector = sum(self.cartesian_representations()) / float(len(self.contract.operations))
        projector = 0.5 * (projector + projector.T)
        eigenvalues, eigenvectors = np.linalg.eigh(projector)
        basis = eigenvectors[:, eigenvalues > 0.5]
        return FrozenGeometryManifold(reference, basis, self.contract.digest)

    def embedding_orbits(
        self,
        coordinates_angstrom: np.ndarray,
        charges_electron: np.ndarray,
        *,
        origin_angstrom: np.ndarray | None = None,
        tolerance_angstrom: float | None = None,
        charge_tolerance: float = 1.0e-10,
    ) -> tuple[tuple[int, ...], ...] | None:
        """Return solvent/embedding orbits, or ``None`` when it breaks symmetry."""

        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        charges = np.asarray(charges_electron, dtype=float)
        if coordinates.shape != (len(charges), 3):
            raise ValueError("embedding coordinates must have shape ncharges x 3")
        if not len(charges):
            return ()
        tolerance = float(tolerance_angstrom or self.contract.geometry_tolerance_angstrom)
        center = (
            np.mean(coordinates, axis=0)
            if origin_angstrom is None
            else np.asarray(origin_angstrom, dtype=float)
        )
        if center.shape != (3,):
            raise ValueError("embedding symmetry origin must be a Cartesian vector")
        centered = coordinates - center
        mappings: list[np.ndarray] = []
        for operation in self.contract.operations:
            rotated = centered @ np.asarray(operation.rotation, dtype=float).T
            mapping = np.full(len(charges), -1, dtype=np.intp)
            available = set(range(len(charges)))
            for source, point in enumerate(rotated):
                candidates = [
                    target for target in available
                    if abs(charges[target] - charges[source]) <= charge_tolerance
                ]
                if not candidates:
                    return None
                target = min(candidates, key=lambda item: float(np.linalg.norm(centered[item] - point)))
                if np.linalg.norm(centered[target] - point) > tolerance:
                    return None
                mapping[source] = target
                available.remove(target)
            mappings.append(mapping)
        remaining = set(range(len(charges)))
        orbits: list[tuple[int, ...]] = []
        while remaining:
            seed = min(remaining)
            orbit = tuple(sorted({int(mapping[seed]) for mapping in mappings}))
            orbits.append(orbit)
            remaining.difference_update(orbit)
        return tuple(orbits)

    def expand_atomic_scalars(self, values: np.ndarray) -> np.ndarray:
        result = np.asarray(values, dtype=float).copy()
        for orbit in self.atom_orbits:
            result[list(orbit)] = float(np.mean(result[list(orbit)]))
        return result

    def expand_atomic_vectors(self, values: np.ndarray) -> np.ndarray:
        vectors = np.asarray(values, dtype=float)
        result = np.zeros_like(vectors)
        for operation in self.contract.operations:
            rotation = np.asarray(operation.rotation, dtype=float)
            mapping = np.asarray(operation.permutation, dtype=np.intp)
            result += vectors[np.argsort(mapping)] @ rotation
        return result / float(len(self.contract.operations))

    def expand_atomic_pairs(self, values: np.ndarray) -> np.ndarray:
        pairs = np.asarray(values, dtype=float)
        result = np.zeros_like(pairs)
        for operation in self.contract.operations:
            mapping = np.asarray(operation.permutation, dtype=np.intp)
            result[np.ix_(mapping, mapping)] += pairs
        result /= float(len(self.contract.operations))
        return 0.5 * (result + result.T)


@dataclass(frozen=True)
class SymmetryBlockedMatrix:
    """Compact symmetry-adapted matrix with no stored off-block elements."""

    labels: tuple[str, ...]
    indices: tuple[np.ndarray, ...]
    values: tuple[np.ndarray, ...]

    @property
    def stored_elements(self) -> int:
        return sum(int(block.size) for block in self.values)


@dataclass(frozen=True)
class SymmetryMode:
    index: int
    eigenvalue: float
    irrep: str
    irrep_weight: float
    vector: np.ndarray


@dataclass(frozen=True)
class FrozenGeometryManifold:
    """Minimal Cartesian displacement chart that preserves the frozen group."""

    reference_coordinates_angstrom: np.ndarray
    basis: np.ndarray
    contract_digest: str

    @property
    def dimension(self) -> int:
        return int(self.basis.shape[1])

    def compress(self, coordinates_angstrom: np.ndarray) -> np.ndarray:
        displacement = (
            np.asarray(coordinates_angstrom, dtype=float)
            - self.reference_coordinates_angstrom
        ).reshape(-1)
        return self.basis.T @ displacement

    def expand(self, coordinates: np.ndarray) -> np.ndarray:
        values = np.asarray(coordinates, dtype=float).reshape(-1)
        if values.shape != (self.dimension,):
            raise ValueError("symmetry geometry coordinate dimension changed")
        displacement = (self.basis @ values).reshape(self.reference_coordinates_angstrom.shape)
        return self.reference_coordinates_angstrom + displacement


@dataclass(frozen=True)
class SymmetryRestart:
    """Portable restart certificate for a frozen symmetry-adapted density."""

    contract_digest: str
    block_labels: tuple[str, ...]
    block_dimensions: tuple[int, ...]
    transformation_digest: str
    density_blocks: tuple[np.ndarray, ...]

    @classmethod
    def create(cls, adapter: SymmetryAdaptedBasis, density: np.ndarray) -> "SymmetryRestart":
        transformation_digest = sha256(
            np.asarray(adapter.transformation, dtype="<f8").tobytes()
        ).hexdigest()
        blocked = adapter.to_blocks(density)
        return cls(
            adapter.contract.digest,
            adapter.block_labels,
            adapter.block_dimensions,
            transformation_digest,
            tuple(value.copy() for value in blocked.values),
        )

    def validate(self, adapter: SymmetryAdaptedBasis) -> None:
        expected = sha256(np.asarray(adapter.transformation, dtype="<f8").tobytes()).hexdigest()
        if self.contract_digest != adapter.contract.digest:
            raise ValueError("symmetry restart point-group contract changed")
        if self.block_labels != adapter.block_labels or self.block_dimensions != adapter.block_dimensions:
            raise ValueError("symmetry restart irrep layout changed")
        if self.transformation_digest != expected:
            raise ValueError("symmetry restart SALC transformation changed")

    def expand_density(self, adapter: SymmetryAdaptedBasis) -> np.ndarray:
        self.validate(adapter)
        return adapter.from_blocks(
            SymmetryBlockedMatrix(adapter.block_labels, adapter.blocks, self.density_blocks)
        )

    def save(self, path: str | Path) -> None:
        payload = {
            "contract_digest": self.contract_digest,
            "block_labels": self.block_labels,
            "block_dimensions": self.block_dimensions,
            "transformation_digest": self.transformation_digest,
        }
        arrays = {f"block_{index}": block for index, block in enumerate(self.density_blocks)}
        np.savez_compressed(Path(path), metadata=np.asarray(json.dumps(payload)), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "SymmetryRestart":
        with np.load(Path(path), allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            blocks = tuple(
                np.asarray(data[f"block_{index}"], dtype=float)
                for index in range(len(metadata["block_labels"]))
            )
        return cls(
            str(metadata["contract_digest"]),
            tuple(str(value) for value in metadata["block_labels"]),
            tuple(int(value) for value in metadata["block_dimensions"]),
            str(metadata["transformation_digest"]),
            blocks,
        )


def _angular_values(angular: int, points: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(points, dtype=float).T
    if angular == 0:
        return np.ones((len(points), 1))
    if angular == 1:
        return np.column_stack((x, y, z))
    if angular == 2:
        root3 = np.sqrt(3.0)
        return np.column_stack(
            (
                z * z - 0.5 * (x * x + y * y),
                root3 * x * z,
                root3 * y * z,
                0.5 * root3 * (x * x - y * y),
                root3 * x * y,
            )
        )
    raise ValueError(f"unsupported angular momentum in symmetry layer: {angular}")


def _angular_representation(rotation: np.ndarray, angular: int) -> np.ndarray:
    if angular == 0:
        return np.ones((1, 1))
    points = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 0.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (1.0, -1.0, 0.5),
            (-0.5, 1.0, 1.0),
        ),
        dtype=float,
    )
    source = _angular_values(angular, points)
    rotated = _angular_values(angular, points @ np.asarray(rotation, dtype=float))
    representation = np.linalg.lstsq(source, rotated, rcond=None)[0]
    left, _, right = np.linalg.svd(representation)
    return left @ right


def _shell_instances(basis: MolecularBasis) -> tuple[tuple[int, str], ...]:
    instances: list[tuple[int, str]] = []
    for atom, shell in zip(basis.spherical_atom, basis.spherical_shell, strict=True):
        key = (int(atom), str(shell))
        if key not in instances:
            instances.append(key)
    return tuple(instances)


def _shell_orbits(
    instances: Sequence[tuple[int, str]],
    operations: Sequence[FrozenSymmetryOperation],
) -> tuple[tuple[int, ...], ...]:
    lookup = {key: index for index, key in enumerate(instances)}
    remaining = set(range(len(instances)))
    result = []
    while remaining:
        seed = min(remaining)
        atom, shell = instances[seed]
        orbit = {
            lookup[(operation.permutation[atom], shell)] for operation in operations
        }
        result.append(tuple(sorted(orbit)))
        remaining.difference_update(orbit)
    return tuple(result)


def _ao_representation(
    basis: MolecularBasis,
    operation: FrozenSymmetryOperation,
    *,
    transpose_rotation: bool,
) -> np.ndarray:
    size = basis.norbitals
    result = np.zeros((size, size))
    groups: dict[tuple[int, str], list[int]] = {}
    for index, key in enumerate(zip(basis.spherical_atom, basis.spherical_shell, strict=True)):
        groups.setdefault((int(key[0]), str(key[1])), []).append(index)
    rotation = np.asarray(operation.rotation, dtype=float)
    if transpose_rotation:
        rotation = rotation.T
    for (source_atom, shell), source_indices in groups.items():
        target_indices = groups[(operation.permutation[source_atom], shell)]
        angular = {1: 0, 3: 1, 5: 2}[len(source_indices)]
        local = _angular_representation(rotation, angular)
        result[np.ix_(target_indices, source_indices)] = local
    return result


def _operation_class(point_group: str, operation: FrozenSymmetryOperation) -> str:
    label = operation.label.upper()
    if label == "E":
        return "E"
    if point_group == "TD":
        if label.startswith("C3"):
            return "C3"
        if label.startswith("C2"):
            return "C2"
        if label.startswith("IC4"):
            return "S4"
        if label.startswith("SIGMA"):
            return "SD"
    if point_group == "C2V":
        rotation = np.asarray(operation.rotation, dtype=float)
        if np.linalg.det(rotation) > 0.0:
            return "C2"
        normal = np.diag(np.eye(3) - rotation)
        return "SV1" if int(np.argmax(normal)) == 0 else "SV2"
    if point_group == "C3V":
        if label.startswith("C3"):
            return "C3"
        if label.startswith("SIGMA"):
            return "SV"
    return label


def _character_salc_blocks(
    point_group: str,
    operations: Sequence[FrozenSymmetryOperation],
    representations: Sequence[np.ndarray],
) -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[str, ...]] | None:
    """Build exact isotypic projectors for common GFN2 molecular groups."""

    tables: dict[str, tuple[tuple[str, int, dict[str, float]], ...]] = {
        "C2V": (
            ("A1", 1, {"E": 1, "C2": 1, "SV1": 1, "SV2": 1}),
            ("A2", 1, {"E": 1, "C2": 1, "SV1": -1, "SV2": -1}),
            ("B1", 1, {"E": 1, "C2": -1, "SV1": 1, "SV2": -1}),
            ("B2", 1, {"E": 1, "C2": -1, "SV1": -1, "SV2": 1}),
        ),
        "TD": (
            ("A1", 1, {"E": 1, "C3": 1, "C2": 1, "S4": 1, "SD": 1}),
            ("A2", 1, {"E": 1, "C3": 1, "C2": 1, "S4": -1, "SD": -1}),
            ("E", 2, {"E": 2, "C3": -1, "C2": 2, "S4": 0, "SD": 0}),
            ("T1", 3, {"E": 3, "C3": 0, "C2": -1, "S4": 1, "SD": -1}),
            ("T2", 3, {"E": 3, "C3": 0, "C2": -1, "S4": -1, "SD": 1}),
        ),
        "C3V": (
            ("A1", 1, {"E": 1, "C3": 1, "SV": 1}),
            ("A2", 1, {"E": 1, "C3": 1, "SV": -1}),
            ("E", 2, {"E": 2, "C3": -1, "SV": 0}),
        ),
    }
    table = tables.get(point_group)
    if table is None:
        return None
    classes = tuple(_operation_class(point_group, operation) for operation in operations)
    known = {name for _, _, characters in table for name in characters}
    if any(name not in known for name in classes):
        return None
    label_operator = np.zeros_like(representations[0])
    for index, representation in enumerate(representations):
        weight = np.sin((index + 1) * np.sqrt(2.0)) + np.cos((index + 1) * np.sqrt(3.0))
        label_operator += weight * (representation + representation.T)
    columns: list[np.ndarray] = []
    blocks: list[np.ndarray] = []
    labels: list[str] = []
    offset = 0
    order = float(len(operations))
    for label, dimension, characters in table:
        projector = sum(
            characters[name] * representation
            for name, representation in zip(classes, representations, strict=True)
        ) * (dimension / order)
        projector = 0.5 * (projector + projector.T)
        eigenvalues, eigenvectors = np.linalg.eigh(projector)
        selected = eigenvectors[:, eigenvalues > 0.5]
        if selected.shape[1] == 0:
            continue
        row_blocks = (np.arange(selected.shape[1], dtype=np.intp),)
        if dimension > 1:
            row_values, row_vectors = np.linalg.eigh(selected.T @ label_operator @ selected)
            row_scale = max(1.0, float(np.max(np.abs(row_values), initial=0.0)))
            grouped: list[list[int]] = []
            for index, value in enumerate(row_values):
                if not grouped or abs(value - row_values[grouped[-1][0]]) > 2.0e-9 * row_scale:
                    grouped.append([index])
                else:
                    grouped[-1].append(index)
            if len(grouped) == dimension and len({len(group) for group in grouped}) == 1:
                selected = selected @ row_vectors
                row_blocks = tuple(np.asarray(group, dtype=np.intp) for group in grouped)
        for row, local_indices in enumerate(row_blocks, start=1):
            columns.append(selected[:, local_indices])
            size = len(local_indices)
            blocks.append(np.arange(offset, offset + size, dtype=np.intp))
            labels.append(label if len(row_blocks) == 1 else f"{label}.{row}")
            offset += size
    if offset != representations[0].shape[0]:
        return None
    transformation = np.column_stack(columns)
    return transformation, tuple(blocks), tuple(labels)


def _salc_blocks(
    representations: Sequence[np.ndarray],
) -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[str, ...]]:
    size = len(representations[0])
    label_operator = np.zeros((size, size))
    for index, representation in enumerate(representations):
        weight = np.sin((index + 1) * np.sqrt(2.0)) + np.cos((index + 1) * np.sqrt(3.0))
        label_operator += weight * (representation + representation.T)
    eigenvalues, transformation = np.linalg.eigh(0.5 * (label_operator + label_operator.T))
    scale = max(1.0, float(np.max(np.abs(eigenvalues), initial=0.0)))
    blocks: list[list[int]] = []
    for index, value in enumerate(eigenvalues):
        if not blocks or abs(value - eigenvalues[blocks[-1][0]]) > 2.0e-9 * scale:
            blocks.append([index])
        else:
            blocks[-1].append(index)
    indices = tuple(np.asarray(block, dtype=np.intp) for block in blocks)
    return transformation, indices, tuple(f"Gamma{index + 1}" for index in range(len(indices)))


def compile_symmetry_adapted_basis(
    contract: FrozenPointGroup,
    basis: MolecularBasis,
    overlap: np.ndarray,
    coordinates_angstrom: np.ndarray,
) -> SymmetryAdaptedBasis | None:
    """Compile and certify ORACLE operations against one molecular AO basis."""

    contract.validate()
    if len(contract.atom_symbols) != len(coordinates_angstrom):
        raise ValueError("frozen symmetry contract does not match the molecular atom frame")
    if not contract.active:
        return None
    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    centered = coordinates - np.mean(coordinates, axis=0)
    geometry_error = 0.0
    for operation in contract.operations:
        mapping = np.asarray(operation.permutation, dtype=np.intp)
        transformed = centered[np.argsort(mapping)] @ np.asarray(operation.rotation, dtype=float)
        geometry_error = max(
            geometry_error,
            float(np.max(np.linalg.norm(transformed - centered, axis=1), initial=0.0)),
        )
    if geometry_error > contract.geometry_tolerance_angstrom:
        raise RuntimeError(
            f"geometry left frozen {contract.point_group} symmetry: "
            f"{geometry_error:.3e} > {contract.geometry_tolerance_angstrom:.3e} angstrom"
        )
    candidates = []
    for transpose in (False, True):
        representations = tuple(
            _ao_representation(basis, operation, transpose_rotation=transpose)
            for operation in contract.operations
        )
        error = max(
            float(np.max(np.abs(rep.T @ overlap @ rep - overlap), initial=0.0))
            for rep in representations
        )
        candidates.append((error, representations))
    invariance_error, representations = min(candidates, key=lambda item: item[0])
    scale = max(1.0, float(np.max(np.abs(overlap), initial=0.0)))
    if invariance_error > 5.0e-5 * scale:
        raise RuntimeError(
            "ORACLE operations are incompatible with the MATRIX-GFN2 AO basis: "
            f"overlap residual {invariance_error:.3e}"
        )
    symmetric_overlap = sum(
        rep.T @ overlap @ rep for rep in representations
    ) / float(len(representations))
    characterized = _character_salc_blocks(
        contract.point_group, contract.operations, representations
    )
    if characterized is None:
        transformation, blocks, block_labels = _salc_blocks(representations)
    else:
        transformation, blocks, block_labels = characterized
    adapted_overlap = transformation.T @ symmetric_overlap @ transformation
    overlap_blocks = tuple(adapted_overlap[np.ix_(block, block)] for block in blocks)
    overlap_inverse_roots = []
    for block_overlap in overlap_blocks:
        values, vectors = np.linalg.eigh(block_overlap)
        if np.any(values <= 1.0e-10):
            raise np.linalg.LinAlgError("symmetry overlap block is not positive definite")
        overlap_inverse_roots.append((vectors / np.sqrt(values)) @ vectors.T)
    block_slices = tuple(
        (slice(int(block[0]), int(block[-1]) + 1), slice(int(block[0]), int(block[-1]) + 1))
        for block in blocks
    )
    return SymmetryAdaptedBasis(
        contract,
        contract.atom_orbits,
        _shell_orbits(_shell_instances(basis), contract.operations),
        representations,
        transformation,
        blocks,
        block_labels,
        overlap_blocks,
        invariance_error,
        geometry_error,
        tuple(overlap_inverse_roots),
        block_slices,
    )


__all__ = [
    "FrozenPointGroup",
    "FrozenGeometryManifold",
    "FrozenSymmetryOperation",
    "SymmetryAdaptedBasis",
    "SymmetryBlockedMatrix",
    "SymmetryMode",
    "SymmetryRestart",
    "compile_symmetry_adapted_basis",
    "DEFAULT_SYMMETRY_SMALL_SYSTEM_HOST_THRESHOLD",
    "DEFAULT_SYMMETRY_BLOCK_TOLERANCE",
]
