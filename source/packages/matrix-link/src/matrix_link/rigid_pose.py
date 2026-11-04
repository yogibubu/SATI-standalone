"""Direct rigid-fragment realization for intermolecular PES exploration.

The general SONIC back-transform remains the authoritative fallback for mixed
or flexible coordinate sets.  This module compiles complete FTRANS/FROT
triplets into independent SE(3) fragment poses so a Monte Carlo or genetic
driver can realize a candidate without constructing a Wilson B matrix.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np

from matrix_chem import (
    kabsch_rotation,
    rotation_composition_jacobian,
    rotation_matrix_from_vector,
    rotation_vector_from_matrix,
)

if TYPE_CHECKING:
    from matrix_smith.models import GICDefinition


@dataclass(frozen=True)
class RigidFragmentPose:
    """Gauge-fixed pose of one partner relative to the reference fragment.

    ``translation_angstrom`` contains partner-center minus reference-center
    components in the reference-fragment frame when that frame is defined;
    legacy frameless contracts retain laboratory components.
    ``quaternion_wxyz`` represents the row-vector rotation applied to the
    partner's centered reference coordinates.
    """

    translation_angstrom: np.ndarray
    quaternion_wxyz: np.ndarray

    def __post_init__(self) -> None:
        translation = np.asarray(self.translation_angstrom, dtype=float).reshape(3)
        quaternion = _normalized_quaternion(self.quaternion_wxyz)
        if not np.all(np.isfinite(translation)):
            raise ValueError("fragment-pose translation must be finite")
        object.__setattr__(self, "translation_angstrom", translation)
        object.__setattr__(self, "quaternion_wxyz", quaternion)


@dataclass(frozen=True)
class RelativeFragmentPose:
    """Pose of a flexible partner in the instantaneous reference-body frame."""

    translation_angstrom: np.ndarray
    quaternion_wxyz: np.ndarray

    def __post_init__(self) -> None:
        translation = np.asarray(self.translation_angstrom, dtype=float).reshape(3)
        quaternion = _normalized_quaternion(self.quaternion_wxyz)
        if not np.all(np.isfinite(translation)):
            raise ValueError("relative fragment-pose translation must be finite")
        object.__setattr__(self, "translation_angstrom", translation)
        object.__setattr__(self, "quaternion_wxyz", quaternion)


@dataclass(frozen=True)
class RigidFragmentBlock:
    """Compiled mapping between one fragment pose and six frozen SONICs."""

    atom_indices: tuple[int, ...]
    reference_atom_indices: tuple[int, ...]
    translation_indices: tuple[int, int, int]
    rotation_indices: tuple[int, int, int]
    reference_center_angstrom: np.ndarray
    centered_reference_coordinates_angstrom: np.ndarray
    reference_orientation_frame: np.ndarray
    reference_fragment_frame: np.ndarray
    translation_body_fixed: bool
    frame_atom_indices: tuple[int, int]
    reference_frame_atom_indices: tuple[int, int]
    reference_relative_frame: np.ndarray

    @property
    def coordinate_indices(self) -> tuple[int, ...]:
        return (*self.translation_indices, *self.rotation_indices)


def _translation_to_lab(
    block: RigidFragmentBlock,
    translation: np.ndarray,
    reference_frame: np.ndarray,
) -> np.ndarray:
    vector = np.asarray(translation, dtype=float).reshape(3)
    if not block.translation_body_fixed:
        return vector
    return vector @ np.asarray(reference_frame, dtype=float).reshape(3, 3).T


def _translation_from_lab(
    block: RigidFragmentBlock,
    translation: np.ndarray,
    reference_frame: np.ndarray,
) -> np.ndarray:
    vector = np.asarray(translation, dtype=float).reshape(3)
    if not block.translation_body_fixed:
        return vector
    return vector @ np.asarray(reference_frame, dtype=float).reshape(3, 3)


@dataclass(frozen=True)
class PoseConstraintResult:
    """Result of a reduced solve in the ``6 * nfragment`` pose space."""

    poses: tuple[RigidFragmentPose, ...]
    coordinates_angstrom: np.ndarray
    values: np.ndarray
    residual: np.ndarray
    iterations: int
    converged: bool


@dataclass(frozen=True)
class RigidComplexModel:
    """B-free direct realization model for a frozen fragmented SONIC contract."""

    reference_coordinates_angstrom: np.ndarray
    blocks: tuple[RigidFragmentBlock, ...]
    coordinate_count: int

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.reference_coordinates_angstrom, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError("rigid-complex reference coordinates must have shape (natom, 3)")
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("rigid-complex reference coordinates must be finite")
        if not self.blocks:
            raise ValueError("rigid-complex model needs at least one movable fragment")
        object.__setattr__(self, "reference_coordinates_angstrom", coordinates)
        object.__setattr__(self, "coordinate_count", int(self.coordinate_count))

    @classmethod
    def from_definition(cls, definition: "GICDefinition") -> "RigidComplexModel":
        """Compile complete, unit FTRANS/FROT triplets from ``definition``."""

        primitives = {primitive.identifier: primitive for primitive in definition.primitives}
        grouped: dict[
            tuple[tuple[int, ...], tuple[int, ...]],
            dict[str, dict[int, int]],
        ] = {}
        for coordinate_index, gic in enumerate(definition.gics):
            coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
            if len(coefficients) != 1 or not np.isclose(float(coefficients[0][1]), 1.0):
                continue
            primitive = primitives.get(coefficients[0][0])
            if primitive is None or primitive.function not in {"FTRANS", "FROT"}:
                continue
            key = (tuple(primitive.atoms), tuple(primitive.ref_atoms))
            grouped.setdefault(key, {"FTRANS": {}, "FROT": {}})
            grouped[key][primitive.function][int(primitive.mode)] = coordinate_index

        reference = np.asarray(definition.reference_coordinates_angstrom, dtype=float)
        blocks: list[RigidFragmentBlock] = []
        occupied_atoms: set[int] = set()
        for (atoms, reference_atoms), functions in grouped.items():
            translations = functions["FTRANS"]
            rotations = functions["FROT"]
            if set(translations) != {0, 1, 2} or set(rotations) != {0, 1, 2}:
                continue
            atom_indices = tuple(atom - 1 for atom in atoms)
            reference_indices = tuple(atom - 1 for atom in reference_atoms)
            if not atom_indices or not reference_indices:
                continue
            if occupied_atoms.intersection(atom_indices):
                raise ValueError("rigid-fragment pose blocks overlap")
            occupied_atoms.update(atom_indices)
            center = np.mean(reference[list(atom_indices), :], axis=0)
            rotation_primitive = primitives[definition.gics[rotations[0]].coefficients[0][0]]
            translation_primitive = primitives[
                definition.gics[translations[0]].coefficients[0][0]
            ]
            orientation_frame = _fragment_frame(
                reference,
                atom_indices,
                tuple(atom - 1 for atom in rotation_primitive.frame_atoms),
            )
            reference_frame = _fragment_frame(
                reference,
                reference_indices,
                tuple(atom - 1 for atom in rotation_primitive.ref_frame_atoms),
            )
            blocks.append(
                RigidFragmentBlock(
                    atom_indices=atom_indices,
                    reference_atom_indices=reference_indices,
                    translation_indices=tuple(translations[axis] for axis in range(3)),
                    rotation_indices=tuple(rotations[axis] for axis in range(3)),
                    reference_center_angstrom=center,
                    centered_reference_coordinates_angstrom=(
                        reference[list(atom_indices), :] - center
                    ),
                    reference_orientation_frame=orientation_frame,
                    reference_fragment_frame=reference_frame,
                    translation_body_fixed=bool(translation_primitive.ref_frame_atoms),
                    frame_atom_indices=tuple(
                        atom - 1 for atom in rotation_primitive.frame_atoms
                    ),
                    reference_frame_atom_indices=tuple(
                        atom - 1 for atom in rotation_primitive.ref_frame_atoms
                    ),
                    reference_relative_frame=(
                        orientation_frame.T @ reference_frame
                    ),
                )
            )
        if not blocks:
            raise ValueError("definition has no complete unit FTRANS/FROT fragment blocks")
        blocks.sort(key=lambda block: block.atom_indices)
        gauge_atoms = blocks[0].reference_atom_indices
        if any(block.reference_atom_indices != gauge_atoms for block in blocks):
            raise ValueError("rigid pose fast path requires one common reference fragment")
        moving_atoms = {atom for block in blocks for atom in block.atom_indices}
        if moving_atoms.intersection(gauge_atoms):
            raise ValueError("rigid pose reference fragment must not be a movable partner")
        return cls(reference, tuple(blocks), len(definition.gics))

    @classmethod
    def try_from_definition(cls, definition: "GICDefinition") -> "RigidComplexModel | None":
        try:
            return cls.from_definition(definition)
        except ValueError:
            return None

    @property
    def coordinate_indices(self) -> tuple[int, ...]:
        return tuple(index for block in self.blocks for index in block.coordinate_indices)

    def supports_coordinate_indices(self, indices: Iterable[int]) -> bool:
        supported = set(self.coordinate_indices)
        return all(int(index) in supported for index in indices)

    def reference_poses(self) -> tuple[RigidFragmentPose, ...]:
        poses = []
        for block in self.blocks:
            reference_center = np.mean(
                self.reference_coordinates_angstrom[list(block.reference_atom_indices), :],
                axis=0,
            )
            poses.append(
                RigidFragmentPose(
                    _translation_from_lab(
                        block,
                        block.reference_center_angstrom - reference_center,
                        block.reference_fragment_frame,
                    ),
                    np.asarray([1.0, 0.0, 0.0, 0.0]),
                )
            )
        return tuple(poses)

    def poses_from_sonic_values(
        self, values: Sequence[float] | np.ndarray
    ) -> tuple[RigidFragmentPose, ...]:
        """Convert absolute frozen SONIC values into independent poses."""

        vector = np.asarray(values, dtype=float).reshape(-1)
        if vector.shape != (self.coordinate_count,):
            raise ValueError("SONIC value count does not match rigid-complex model")
        poses = []
        for block in self.blocks:
            translation = vector[list(block.translation_indices)]
            sonic_rotation = vector[list(block.rotation_indices)]
            # SMITH frames store their axes as columns.  Applying row rotation
            # A to the fragment gives delta = F0.T A F0 in the frozen local
            # frame, hence A = F0 exp(q) F0.T.
            frame = block.reference_orientation_frame
            row_rotation = frame @ rotation_matrix_from_vector(sonic_rotation) @ frame.T
            poses.append(
                RigidFragmentPose(
                    translation,
                    quaternion_from_row_rotation(row_rotation),
                )
            )
        return tuple(poses)

    def sonic_values_from_poses(
        self,
        poses: Sequence[RigidFragmentPose],
        *,
        base_values: Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Return absolute SONIC values represented by ``poses``."""

        if len(poses) != len(self.blocks):
            raise ValueError("pose count does not match rigid-complex model")
        values = (
            np.zeros(self.coordinate_count, dtype=float)
            if base_values is None
            else np.asarray(base_values, dtype=float).reshape(self.coordinate_count).copy()
        )
        for block, pose in zip(self.blocks, poses, strict=True):
            values[list(block.translation_indices)] = pose.translation_angstrom
            row_rotation = row_rotation_from_quaternion(pose.quaternion_wxyz)
            frame = block.reference_orientation_frame
            values[list(block.rotation_indices)] = rotation_vector_from_matrix(
                frame.T @ row_rotation @ frame
            )
        return values

    def realize(self, poses: Sequence[RigidFragmentPose]) -> np.ndarray:
        """Materialize one candidate geometry without evaluating a B matrix."""

        if len(poses) != len(self.blocks):
            raise ValueError("pose count does not match rigid-complex model")
        coordinates = self.reference_coordinates_angstrom.copy()
        for block, pose in zip(self.blocks, poses, strict=True):
            reference_center = np.mean(coordinates[list(block.reference_atom_indices), :], axis=0)
            desired_center = reference_center + _translation_to_lab(
                block,
                pose.translation_angstrom,
                block.reference_fragment_frame,
            )
            rotation = row_rotation_from_quaternion(pose.quaternion_wxyz)
            coordinates[list(block.atom_indices), :] = (
                block.centered_reference_coordinates_angstrom @ rotation + desired_center
            )
        return coordinates

    def realize_sonic(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        vector = np.asarray(values, dtype=float).reshape(-1)
        if vector.shape != (self.coordinate_count,):
            raise ValueError("SONIC value count does not match rigid-complex model")
        coordinates = self.reference_coordinates_angstrom.copy()
        for block in self.blocks:
            reference_center = np.mean(
                coordinates[list(block.reference_atom_indices), :], axis=0
            )
            desired_center = (
                reference_center
                + _translation_to_lab(
                    block,
                    vector[list(block.translation_indices)],
                    block.reference_fragment_frame,
                )
            )
            frame = block.reference_orientation_frame
            rotation = (
                frame
                @ rotation_matrix_from_vector(
                    vector[list(block.rotation_indices)]
                )
                @ frame.T
            )
            coordinates[list(block.atom_indices), :] = (
                block.centered_reference_coordinates_angstrom @ rotation
                + desired_center
            )
        return coordinates

    def sonic_tangent(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        """Return the analytic ``dx/dq`` map of :meth:`realize_sonic`.

        The direct pose realization fixes the reference fragment as its
        Cartesian gauge.  Its tangent must use the same gauge; the symmetric
        fragment predictor used by the general back-transform has an
        equivalent internal-coordinate displacement but a different
        Cartesian translation/rotation null-space representative.
        """

        vector = np.asarray(values, dtype=float).reshape(-1)
        if vector.shape != (self.coordinate_count,):
            raise ValueError("SONIC value count does not match rigid-complex model")
        tangent = np.zeros(
            (self.reference_coordinates_angstrom.size, self.coordinate_count),
            dtype=float,
        )
        for block in self.blocks:
            atoms = np.asarray(block.atom_indices, dtype=int)
            for axis, coordinate_index in enumerate(block.translation_indices):
                displacement = np.zeros_like(self.reference_coordinates_angstrom)
                direction = (
                    block.reference_fragment_frame[:, axis]
                    if block.translation_body_fixed
                    else np.eye(3, dtype=float)[:, axis]
                )
                displacement[atoms, :] = direction
                tangent[:, coordinate_index] = displacement.reshape(-1)

            rotation_vector = vector[list(block.rotation_indices)]
            base_rotation = rotation_matrix_from_vector(rotation_vector)
            local_from_value = np.linalg.solve(
                rotation_composition_jacobian(rotation_vector),
                np.eye(3, dtype=float),
            )
            frame = block.reference_orientation_frame
            centered = block.centered_reference_coordinates_angstrom
            for axis, coordinate_index in enumerate(block.rotation_indices):
                local = local_from_value[:, axis]
                skew = np.asarray(
                    [
                        [0.0, -local[2], local[1]],
                        [local[2], 0.0, -local[0]],
                        [-local[1], local[0], 0.0],
                    ],
                    dtype=float,
                )
                displacement = np.zeros_like(self.reference_coordinates_angstrom)
                displacement[atoms, :] = (
                    centered @ frame @ (-skew) @ base_rotation @ frame.T
                )
                tangent[:, coordinate_index] = displacement.reshape(-1)
        return tangent

    def sonic_tangent_from_base(
        self,
        values: Sequence[float] | np.ndarray,
        base_coordinates_angstrom: np.ndarray,
    ) -> np.ndarray:
        """Return the analytic tangent of :meth:`realize_sonic_from_base`."""

        vector = np.asarray(values, dtype=float).reshape(-1)
        if vector.shape != (self.coordinate_count,):
            raise ValueError("SONIC value count does not match rigid-complex model")
        coordinates = np.asarray(base_coordinates_angstrom, dtype=float)
        if coordinates.shape != self.reference_coordinates_angstrom.shape:
            raise ValueError("rigid-complex base coordinates have invalid shape")
        tangent = np.zeros((coordinates.size, self.coordinate_count), dtype=float)
        for block in self.blocks:
            atoms = np.asarray(block.atom_indices, dtype=int)
            reference_frame = _fragment_frame(
                coordinates,
                block.reference_atom_indices,
                block.reference_frame_atom_indices,
            )
            for axis, coordinate_index in enumerate(block.translation_indices):
                displacement = np.zeros_like(coordinates)
                direction = (
                    reference_frame[:, axis]
                    if block.translation_body_fixed
                    else np.eye(3, dtype=float)[:, axis]
                )
                displacement[atoms, :] = direction
                tangent[:, coordinate_index] = displacement.reshape(-1)

            fragment = coordinates[atoms, :]
            centered = fragment - np.mean(fragment, axis=0)
            fragment_frame = _fragment_frame(
                coordinates,
                block.atom_indices,
                block.frame_atom_indices,
            )
            reference_frame = _fragment_frame(
                coordinates,
                block.reference_atom_indices,
                block.reference_frame_atom_indices,
            )
            target_rotation = vector[list(block.rotation_indices)]
            base_rotation = rotation_matrix_from_vector(target_rotation)
            local_from_value = np.linalg.solve(
                rotation_composition_jacobian(target_rotation),
                np.eye(3, dtype=float),
            )
            target_frame = reference_frame @ block.reference_relative_frame.T
            for axis, coordinate_index in enumerate(block.rotation_indices):
                local = local_from_value[:, axis]
                skew = np.asarray(
                    [
                        [0.0, -local[2], local[1]],
                        [local[2], 0.0, -local[0]],
                        [-local[1], local[0], 0.0],
                    ],
                    dtype=float,
                )
                displacement = np.zeros_like(coordinates)
                displacement[atoms, :] = (
                    centered
                    @ fragment_frame
                    @ (-skew)
                    @ base_rotation
                    @ target_frame.T
                )
                tangent[:, coordinate_index] = displacement.reshape(-1)
        return tangent

    def realize_sonic_from_base(
        self,
        values: Sequence[float] | np.ndarray,
        base_coordinates_angstrom: np.ndarray,
    ) -> np.ndarray:
        """Apply absolute fragment poses to an already torsion-deformed base.

        This preserves exact intrafragment finite rotations while restoring the
        requested intermolecular translation and orientation without a Wilson
        matrix or a nonlinear Cartesian back-transform.
        """

        vector = np.asarray(values, dtype=float).reshape(-1)
        if vector.shape != (self.coordinate_count,):
            raise ValueError("SONIC value count does not match rigid-complex model")
        coordinates = np.asarray(base_coordinates_angstrom, dtype=float).copy()
        if coordinates.shape != self.reference_coordinates_angstrom.shape:
            raise ValueError("rigid-complex base coordinates have invalid shape")
        for block in self.blocks:
            reference_center = np.mean(coordinates[list(block.reference_atom_indices), :], axis=0)
            reference_frame = _fragment_frame(
                coordinates,
                block.reference_atom_indices,
                block.reference_frame_atom_indices,
            )
            desired_center = reference_center + _translation_to_lab(
                block,
                vector[list(block.translation_indices)],
                reference_frame,
            )
            fragment = coordinates[list(block.atom_indices), :]
            fragment_center = np.mean(fragment, axis=0)
            fragment_frame = _fragment_frame(
                coordinates,
                block.atom_indices,
                block.frame_atom_indices,
            )
            reference_frame = _fragment_frame(
                coordinates,
                block.reference_atom_indices,
                block.reference_frame_atom_indices,
            )
            target_delta = rotation_matrix_from_vector(
                vector[list(block.rotation_indices)]
            )
            desired_fragment_frame = (
                reference_frame
                @ block.reference_relative_frame.T
                @ target_delta.T
            )
            rotation = fragment_frame @ desired_fragment_frame.T
            coordinates[list(block.atom_indices), :] = (
                (fragment - fragment_center) @ rotation + desired_center
            )
        return coordinates

    def realize_batch(
        self,
        pose_batch: Sequence[Sequence[RigidFragmentPose]],
        *,
        workers: int = 1,
    ) -> np.ndarray:
        """Materialize a population as ``(ncandidate, natom, 3)``."""

        candidates = tuple(pose_batch)
        if not candidates:
            return np.empty((0, *self.reference_coordinates_angstrom.shape), dtype=float)
        worker_count = _validated_worker_count(workers)
        if worker_count == 1 or len(candidates) == 1:
            realized = [self.realize(poses) for poses in candidates]
        else:
            with ThreadPoolExecutor(max_workers=min(worker_count, len(candidates))) as executor:
                realized = list(executor.map(self.realize, candidates))
        return np.stack(realized, axis=0)

    def realize_sonic_batch(
        self,
        values_batch: Sequence[Sequence[float]] | np.ndarray,
        *,
        workers: int = 1,
    ) -> np.ndarray:
        values = np.asarray(values_batch, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.coordinate_count:
            raise ValueError("SONIC batch must have shape (ncandidate, coordinate_count)")
        worker_count = min(_validated_worker_count(workers), max(values.shape[0], 1))
        if worker_count == 1 or values.shape[0] < 64 * worker_count:
            return self._realize_sonic_batch_serial(values)
        chunks = tuple(chunk for chunk in np.array_split(values, worker_count) if len(chunk))
        with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            realized = tuple(executor.map(self._realize_sonic_batch_serial, chunks))
        return np.concatenate(realized, axis=0)

    def _realize_sonic_batch_serial(self, values: np.ndarray) -> np.ndarray:
        coordinates = np.broadcast_to(
            self.reference_coordinates_angstrom,
            (values.shape[0], *self.reference_coordinates_angstrom.shape),
        ).copy()
        for block in self.blocks:
            reference_center = np.mean(
                self.reference_coordinates_angstrom[list(block.reference_atom_indices), :],
                axis=0,
            )
            translations = values[:, list(block.translation_indices)]
            if block.translation_body_fixed:
                translations = translations @ block.reference_fragment_frame.T
            centers = reference_center + translations
            local_rotations = _row_rotation_matrices_from_vectors(
                values[:, list(block.rotation_indices)]
            )
            frame = block.reference_orientation_frame
            rotations = np.einsum("ij,njk,kl->nil", frame, local_rotations, frame.T)
            coordinates[:, list(block.atom_indices), :] = (
                np.einsum(
                    "aj,njk->nak",
                    block.centered_reference_coordinates_angstrom,
                    rotations,
                )
                + centers[:, None, :]
            )
        return coordinates

    def extract_poses(
        self,
        coordinates_angstrom: np.ndarray,
        *,
        gauge_fix_reference: bool = True,
    ) -> tuple[RigidFragmentPose, ...]:
        """Extract fragment poses, optionally aligning the reference fragment."""

        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        if coordinates.shape != self.reference_coordinates_angstrom.shape:
            raise ValueError("coordinates do not match rigid-complex model")
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("coordinates must be finite")
        working = coordinates.copy()
        reference_indices = self.blocks[0].reference_atom_indices
        if gauge_fix_reference:
            moving = working[list(reference_indices), :]
            reference = self.reference_coordinates_angstrom[list(reference_indices), :]
            moving_center = np.mean(moving, axis=0)
            reference_center = np.mean(reference, axis=0)
            rotation = kabsch_rotation(moving, reference)
            working = (working - moving_center) @ rotation + reference_center
        poses = []
        for block in self.blocks:
            atom_coordinates = working[list(block.atom_indices), :]
            center = np.mean(atom_coordinates, axis=0)
            reference_center = np.mean(working[list(block.reference_atom_indices), :], axis=0)
            reference_frame = _fragment_frame(
                working,
                block.reference_atom_indices,
                block.reference_frame_atom_indices,
            )
            row_rotation = kabsch_rotation(
                block.centered_reference_coordinates_angstrom,
                atom_coordinates - center,
            )
            poses.append(
                RigidFragmentPose(
                    _translation_from_lab(
                        block,
                        center - reference_center,
                        reference_frame,
                    ),
                    quaternion_from_row_rotation(row_rotation),
                )
            )
        return tuple(poses)

    def extract_relative_poses(
        self,
        coordinates_angstrom: np.ndarray,
    ) -> tuple[RelativeFragmentPose, ...]:
        """Extract poses independently of each fragment's internal deformation."""

        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        if coordinates.shape != self.reference_coordinates_angstrom.shape:
            raise ValueError("coordinates do not match rigid-complex model")
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("coordinates must be finite")
        poses: list[RelativeFragmentPose] = []
        for block in self.blocks:
            reference_center = np.mean(
                coordinates[list(block.reference_atom_indices), :], axis=0
            )
            fragment_center = np.mean(
                coordinates[list(block.atom_indices), :], axis=0
            )
            reference_frame = _fragment_frame(
                coordinates,
                block.reference_atom_indices,
                block.reference_frame_atom_indices,
            )
            fragment_frame = _fragment_frame(
                coordinates,
                block.atom_indices,
                block.frame_atom_indices,
            )
            poses.append(
                RelativeFragmentPose(
                    _translation_from_lab(
                        block,
                        fragment_center - reference_center,
                        reference_frame,
                    ),
                    quaternion_from_row_rotation(reference_frame.T @ fragment_frame),
                )
            )
        return tuple(poses)

    def realize_relative_poses_from_base(
        self,
        base_coordinates_angstrom: np.ndarray,
        poses: Sequence[RelativeFragmentPose],
    ) -> np.ndarray:
        """Apply relative poses while preserving the base intrafragment shapes."""

        if len(poses) != len(self.blocks):
            raise ValueError("relative pose count does not match rigid-complex model")
        coordinates = np.asarray(base_coordinates_angstrom, dtype=float).copy()
        if coordinates.shape != self.reference_coordinates_angstrom.shape:
            raise ValueError("rigid-complex base coordinates have invalid shape")
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("rigid-complex base coordinates must be finite")
        for block, pose in zip(self.blocks, poses, strict=True):
            atoms = list(block.atom_indices)
            reference_center = np.mean(
                coordinates[list(block.reference_atom_indices), :], axis=0
            )
            fragment_center = np.mean(coordinates[atoms, :], axis=0)
            reference_frame = _fragment_frame(
                coordinates,
                block.reference_atom_indices,
                block.reference_frame_atom_indices,
            )
            fragment_frame = _fragment_frame(
                coordinates,
                block.atom_indices,
                block.frame_atom_indices,
            )
            desired_fragment_frame = (
                reference_frame
                @ row_rotation_from_quaternion(pose.quaternion_wxyz)
            )
            rotation = fragment_frame @ desired_fragment_frame.T
            desired_center = reference_center + _translation_to_lab(
                block,
                pose.translation_angstrom,
                reference_frame,
            )
            coordinates[atoms, :] = (
                (coordinates[atoms, :] - fragment_center) @ rotation
                + desired_center
            )
        return coordinates

    def mutate_pose(
        self,
        poses: Sequence[RigidFragmentPose],
        fragment_index: int,
        *,
        translation_increment_angstrom: Sequence[float] = (0.0, 0.0, 0.0),
        rotation_increment_radian: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> tuple[RigidFragmentPose, ...]:
        """Compose a local SE(3) Monte Carlo mutation on one partner."""

        index = int(fragment_index)
        if index < 0 or index >= len(self.blocks):
            raise IndexError("fragment index is outside rigid-complex model")
        updated = list(poses)
        current = updated[index]
        increment = rotation_matrix_from_vector(np.asarray(rotation_increment_radian, dtype=float))
        rotation = row_rotation_from_quaternion(current.quaternion_wxyz) @ increment
        updated[index] = RigidFragmentPose(
            current.translation_angstrom
            + np.asarray(translation_increment_angstrom, dtype=float).reshape(3),
            quaternion_from_row_rotation(rotation),
        )
        return tuple(updated)

    def realize_sonic_constraints(
        self,
        definition: "GICDefinition",
        coordinate_indices: Sequence[int],
        target_values: Sequence[float] | np.ndarray,
        *,
        initial_poses: Sequence[RigidFragmentPose] | None = None,
        max_iterations: int = 20,
        tolerance: float = 1.0e-9,
        max_pose_step: float = 0.25,
        parallel_workers: int = 1,
    ) -> PoseConstraintResult:
        """Realize descriptive intermolecular coordinates in pose space.

        Only requested SMITH rows are evaluated.  Their Cartesian derivatives
        are contracted with analytic rigid-body generators to form a small
        ``nconstraint x (6*nfragment)`` Jacobian.
        """

        from matrix_smith import evaluate_gic_subset, evaluate_gic_values_subset

        smith_workers = _validated_worker_count(parallel_workers)
        indices = tuple(int(index) for index in coordinate_indices)
        target = np.asarray(target_values, dtype=float).reshape(-1)
        if target.shape != (len(indices),):
            raise ValueError("constraint target count does not match coordinate indices")
        poses = tuple(initial_poses or self.reference_poses())
        if len(poses) != len(self.blocks):
            raise ValueError("initial pose count does not match rigid-complex model")
        coordinates = self.realize(poses)
        values = evaluate_gic_values_subset(definition, indices, coordinates_angstrom=coordinates)
        residual = target - values
        initial_norm = float(np.linalg.norm(residual))
        if initial_norm <= tolerance:
            return PoseConstraintResult(poses, coordinates, values, residual, 0, True)

        for iteration in range(1, max(int(max_iterations), 1) + 1):
            values, b_matrix = evaluate_gic_subset(
                definition,
                indices,
                coordinates_angstrom=coordinates,
                parallel_workers=smith_workers,
            )
            residual = target - values
            tangent = self._pose_tangent(coordinates)
            jacobian = np.asarray(b_matrix, dtype=float) @ tangent
            metric = jacobian @ jacobian.T
            scale = max(float(np.trace(metric)), 1.0)
            try:
                multipliers = np.linalg.solve(
                    metric + 1.0e-10 * scale * np.eye(len(indices)),
                    residual,
                )
            except np.linalg.LinAlgError:
                break
            step = jacobian.T @ multipliers
            norm = float(np.linalg.norm(step))
            if not np.isfinite(norm) or norm <= 1.0e-14:
                break
            if max_pose_step > 0.0 and norm > max_pose_step:
                step *= max_pose_step / norm
            current_norm = float(np.linalg.norm(residual))
            accepted = False
            for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625):
                trial_poses = poses
                for block_index in range(len(self.blocks)):
                    offset = 6 * block_index
                    trial_poses = self.mutate_pose(
                        trial_poses,
                        block_index,
                        translation_increment_angstrom=(fraction * step[offset : offset + 3]),
                        rotation_increment_radian=(fraction * step[offset + 3 : offset + 6]),
                    )
                trial_coordinates = self.realize(trial_poses)
                trial_values = evaluate_gic_values_subset(
                    definition, indices, coordinates_angstrom=trial_coordinates
                )
                trial_residual = target - trial_values
                if float(np.linalg.norm(trial_residual)) < current_norm:
                    poses = trial_poses
                    coordinates = trial_coordinates
                    values = trial_values
                    residual = trial_residual
                    accepted = True
                    break
            if not accepted:
                break
            if float(np.linalg.norm(residual)) <= tolerance * max(1.0, initial_norm):
                return PoseConstraintResult(poses, coordinates, values, residual, iteration, True)
        return PoseConstraintResult(
            poses,
            coordinates,
            values,
            residual,
            iteration,
            False,
        )

    def realize_sonic_constraints_batch(
        self,
        definition: "GICDefinition",
        coordinate_indices: Sequence[int],
        target_values_batch: Sequence[Sequence[float]] | np.ndarray,
        *,
        initial_pose_batch: Sequence[Sequence[RigidFragmentPose]] | None = None,
        workers: int = 1,
        max_iterations: int = 20,
        tolerance: float = 1.0e-9,
        max_pose_step: float = 0.25,
    ) -> tuple[PoseConstraintResult, ...]:
        """Solve independent descriptive-coordinate candidates concurrently."""

        indices = tuple(int(index) for index in coordinate_indices)
        targets = np.asarray(target_values_batch, dtype=float)
        if targets.ndim != 2 or targets.shape[1] != len(indices):
            raise ValueError("constraint target batch must have shape (ncandidate, nconstraint)")
        if initial_pose_batch is None:
            initial = (None,) * targets.shape[0]
        else:
            initial = tuple(tuple(candidate) for candidate in initial_pose_batch)
            if len(initial) != targets.shape[0]:
                raise ValueError("initial pose batch count does not match target batch")

        def solve(item: tuple[np.ndarray, Sequence[RigidFragmentPose] | None]):
            target, poses = item
            return self.realize_sonic_constraints(
                definition,
                indices,
                target,
                initial_poses=poses,
                max_iterations=max_iterations,
                tolerance=tolerance,
                max_pose_step=max_pose_step,
                parallel_workers=1,
            )

        items = tuple(zip(targets, initial, strict=True))
        worker_count = _validated_worker_count(workers)
        if worker_count == 1 or len(items) <= 1:
            return tuple(solve(item) for item in items)
        with ThreadPoolExecutor(max_workers=min(worker_count, len(items))) as executor:
            return tuple(executor.map(solve, items))

    def _pose_tangent(self, coordinates_angstrom: np.ndarray) -> np.ndarray:
        """Return analytic Cartesian generators for independent fragment poses."""

        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        tangent = np.zeros((coordinates.size, 6 * len(self.blocks)), dtype=float)
        for block_index, block in enumerate(self.blocks):
            atoms = np.asarray(block.atom_indices, dtype=int)
            center = np.mean(coordinates[atoms, :], axis=0)
            centered = coordinates[atoms, :] - center
            reference_frame = _fragment_frame(
                coordinates,
                block.reference_atom_indices,
                block.reference_frame_atom_indices,
            )
            for axis in range(3):
                translation = np.zeros_like(coordinates)
                direction = (
                    reference_frame[:, axis]
                    if block.translation_body_fixed
                    else np.eye(3, dtype=float)[:, axis]
                )
                translation[atoms, :] = direction
                tangent[:, 6 * block_index + axis] = translation.reshape(-1)
                unit = np.zeros(3, dtype=float)
                unit[axis] = 1.0
                skew = np.asarray(
                    [
                        [0.0, -unit[2], unit[1]],
                        [unit[2], 0.0, -unit[0]],
                        [-unit[1], unit[0], 0.0],
                    ],
                    dtype=float,
                )
                rotation = np.zeros_like(coordinates)
                rotation[atoms, :] = centered @ (-skew)
                tangent[:, 6 * block_index + 3 + axis] = rotation.reshape(-1)
        return tangent


def _validated_worker_count(workers: int) -> int:
    count = int(workers)
    if count <= 0:
        raise ValueError("workers must be positive")
    return count


def row_rotation_from_quaternion(quaternion_wxyz: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return a proper row-vector rotation for a unit ``(w, x, y, z)`` quaternion."""

    w, x, y, z = _normalized_quaternion(quaternion_wxyz)
    column_rotation = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )
    return column_rotation.T


def quaternion_from_row_rotation(rotation: np.ndarray) -> np.ndarray:
    """Return canonical ``(w, x, y, z)`` for a proper row-vector rotation."""

    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("rotation must be a finite 3 x 3 matrix")
    u_matrix, _singular, vt_matrix = np.linalg.svd(matrix)
    row_rotation = u_matrix @ vt_matrix
    if np.linalg.det(row_rotation) < 0.0:
        u_matrix[:, -1] *= -1.0
        row_rotation = u_matrix @ vt_matrix
    m = row_rotation.T
    trace = float(np.trace(m))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.asarray(
            [
                0.25 * scale,
                (m[2, 1] - m[1, 2]) / scale,
                (m[0, 2] - m[2, 0]) / scale,
                (m[1, 0] - m[0, 1]) / scale,
            ]
        )
    else:
        pivot = int(np.argmax(np.diag(m)))
        if pivot == 0:
            scale = 2.0 * np.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 0.0))
            quaternion = np.asarray(
                [
                    (m[2, 1] - m[1, 2]) / scale,
                    0.25 * scale,
                    (m[0, 1] + m[1, 0]) / scale,
                    (m[0, 2] + m[2, 0]) / scale,
                ]
            )
        elif pivot == 1:
            scale = 2.0 * np.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 0.0))
            quaternion = np.asarray(
                [
                    (m[0, 2] - m[2, 0]) / scale,
                    (m[0, 1] + m[1, 0]) / scale,
                    0.25 * scale,
                    (m[1, 2] + m[2, 1]) / scale,
                ]
            )
        else:
            scale = 2.0 * np.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 0.0))
            quaternion = np.asarray(
                [
                    (m[1, 0] - m[0, 1]) / scale,
                    (m[0, 2] + m[2, 0]) / scale,
                    (m[1, 2] + m[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    return _normalized_quaternion(quaternion)


def _normalized_quaternion(values: Sequence[float] | np.ndarray) -> np.ndarray:
    quaternion = np.asarray(values, dtype=float).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm <= 1.0e-14:
        raise ValueError("quaternion must be finite and nonzero")
    quaternion = quaternion / norm
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return quaternion


def _row_rotation_matrices_from_vectors(vectors: np.ndarray) -> np.ndarray:
    """Vectorized row-convention Rodrigues map for a population."""

    values = np.asarray(vectors, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("rotation-vector batch must have shape (n, 3)")
    count = values.shape[0]
    theta = np.linalg.norm(values, axis=1)
    skew = np.zeros((count, 3, 3), dtype=float)
    skew[:, 0, 1] = -values[:, 2]
    skew[:, 0, 2] = values[:, 1]
    skew[:, 1, 0] = values[:, 2]
    skew[:, 1, 2] = -values[:, 0]
    skew[:, 2, 0] = -values[:, 1]
    skew[:, 2, 1] = values[:, 0]
    skew2 = np.einsum("nij,njk->nik", skew, skew)
    small = theta < 1.0e-8
    sine_scale = np.empty(count, dtype=float)
    cosine_scale = np.empty(count, dtype=float)
    sine_scale[small] = 1.0 - theta[small] ** 2 / 6.0
    cosine_scale[small] = 0.5 - theta[small] ** 2 / 24.0
    sine_scale[~small] = np.sin(theta[~small]) / theta[~small]
    cosine_scale[~small] = (1.0 - np.cos(theta[~small])) / theta[~small] ** 2
    return (
        np.eye(3, dtype=float)[None, :, :]
        - sine_scale[:, None, None] * skew
        + cosine_scale[:, None, None] * skew2
    )


def _fragment_frame(
    coordinates: np.ndarray,
    atom_indices: tuple[int, ...],
    frame_atom_indices: tuple[int, ...],
) -> np.ndarray:
    if len(frame_atom_indices) != 2:
        raise ValueError("rigid FROT block needs two frozen frame atoms")
    center = np.mean(coordinates[list(atom_indices), :], axis=0)
    p_axis = coordinates[frame_atom_indices[0]] - center
    p_axis /= np.linalg.norm(p_axis)
    q_raw = np.cross(p_axis, coordinates[frame_atom_indices[1]] - center)
    q_axis = q_raw / np.linalg.norm(q_raw)
    s_axis = np.cross(p_axis, q_axis)
    s_axis /= np.linalg.norm(s_axis)
    return np.column_stack((p_axis, q_axis, s_axis))


__all__ = [
    "PoseConstraintResult",
    "RelativeFragmentPose",
    "RigidComplexModel",
    "RigidFragmentBlock",
    "RigidFragmentPose",
    "quaternion_from_row_rotation",
    "row_rotation_from_quaternion",
]
