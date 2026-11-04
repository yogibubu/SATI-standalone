"""General ring reconstruction with detachable exocyclic substituents."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class DetachedSubstituent:
    """One connected exocyclic component and its unique ring anchor."""

    anchor: int
    atoms: tuple[int, ...]
    frame_neighbors: tuple[int, int] = ()


@dataclass(frozen=True)
class RingSubstituentPlan:
    """Frozen information needed to remove and rigidly reattach substituents."""

    ring_atoms: tuple[int, ...]
    protected_atoms: tuple[int, ...]
    core_atoms: tuple[int, ...]
    components: tuple[DetachedSubstituent, ...]
    reference_coordinates: np.ndarray

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.reference_coordinates, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError("reference coordinates must have shape (N, 3)")
        object.__setattr__(self, "reference_coordinates", coordinates.copy())


@dataclass(frozen=True)
class RingReconstructionResult:
    coordinates: np.ndarray
    converged: bool
    evaluations: int
    cp_residual_norm_angstrom: float
    maximum_bond_error_angstrom: float
    angle_rms_degrees: float
    angle_max_degrees: float


@dataclass(frozen=True)
class RingSystemReconstructionResult:
    """Result of a selected-ring target realized in a coupled ring system."""

    coordinates: np.ndarray
    ring_system_atoms: tuple[int, ...]
    converged: bool
    evaluations: int
    cp_residual_norm_angstrom: float
    maximum_bond_error_angstrom: float
    angle_rms_degrees: float
    angle_max_degrees: float


def cremer_pople_cartesian_components(
    coordinates: np.ndarray,
    *,
    reference_normal: np.ndarray | None = None,
) -> np.ndarray:
    """Return the complete ordered ``N-3`` Cartesian CP component vector."""

    xyz = _coordinates(coordinates)
    try:
        normal = _cp_normal(xyz)
    except ValueError:
        if reference_normal is None:
            raise
        normal = _unit(np.asarray(reference_normal, dtype=float))
    if reference_normal is not None and float(normal @ reference_normal) < 0.0:
        normal = -normal
    heights = (xyz - np.mean(xyz, axis=0)) @ normal
    atom = np.arange(len(xyz), dtype=float)
    values: list[float] = []
    for harmonic in range(2, (len(xyz) - 1) // 2 + 1):
        phase = 2.0 * np.pi * harmonic * atom / len(xyz)
        scale = np.sqrt(2.0 / len(xyz))
        values.extend(
            (
                scale * float(heights @ np.cos(phase)),
                scale * float(heights @ np.sin(phase)),
            )
        )
    if len(xyz) % 2 == 0:
        values.append(float(heights @ ((-1.0) ** atom)) / np.sqrt(len(xyz)))
    return np.asarray(values)


def cremer_pople_cartesian_jacobian(
    coordinates: np.ndarray,
    *,
    reference_normal: np.ndarray | None = None,
) -> np.ndarray:
    """Return the analytic Cartesian Jacobian of the complete CP vector."""

    xyz = _coordinates(coordinates)
    ring_size = len(xyz)
    centered = xyz - np.mean(xyz, axis=0)
    phase = 2.0 * np.pi * np.arange(ring_size, dtype=float) / ring_size
    sine = np.sum(centered * np.sin(phase)[:, None], axis=0)
    cosine = np.sum(centered * np.cos(phase)[:, None], axis=0)
    cross = np.cross(sine, cosine)
    cross_norm = float(np.linalg.norm(cross))
    if cross_norm < 1.0e-12:
        raise ValueError("cannot define a CP normal for a degenerate ring")
    normal = cross / cross_norm
    orientation = 1.0
    if reference_normal is not None and float(normal @ reference_normal) < 0.0:
        normal = -normal
        orientation = -1.0
    projector = np.eye(3) - np.outer(cross / cross_norm, cross / cross_norm)
    sine_cross = _cross_matrix(sine)
    cosine_cross = _cross_matrix(cosine)
    normal_derivatives = []
    for atom in range(ring_size):
        cross_derivative = (
            -np.sin(phase[atom]) * cosine_cross
            + np.cos(phase[atom]) * sine_cross
        )
        normal_derivatives.append(
            orientation * projector @ cross_derivative / cross_norm
        )

    fourier_rows = _cp_fourier_rows(ring_size)
    jacobian = np.zeros((ring_size - 3, 3 * ring_size), dtype=float)
    for row_index, coefficients in enumerate(fourier_rows):
        weighted_position = coefficients @ xyz
        for atom, coefficient in enumerate(coefficients):
            gradient = (
                coefficient * normal
                + normal_derivatives[atom].T @ weighted_position
            )
            jacobian[row_index, 3 * atom : 3 * atom + 3] = gradient
    return jacobian


def reconstruct_ring_from_cp(
    reference: np.ndarray,
    target_components: Sequence[float],
    *,
    bond_lengths: Sequence[float] | None = None,
    angle_weights: Sequence[float] | None = None,
    constraint_weight: float = 1.0e5,
) -> RingReconstructionResult:
    """Reconstruct an arbitrary monocycle from its Cartesian CP components."""

    xyz = _coordinates(reference)
    ring_size = len(xyz)
    target = np.asarray(target_components, dtype=float)
    if target.shape != (ring_size - 3,):
        raise ValueError("a monocycle requires exactly N-3 Cartesian CP components")
    lengths = _bond_lengths(xyz) if bond_lengths is None else np.asarray(bond_lengths, dtype=float)
    weights = np.ones(ring_size) if angle_weights is None else np.asarray(angle_weights, dtype=float)
    if lengths.shape != (ring_size,) or weights.shape != (ring_size,):
        raise ValueError("bond lengths and angle weights require one value per ring atom")
    if np.any(lengths <= 0.0) or np.any(weights <= 0.0) or constraint_weight <= 0.0:
        raise ValueError("bond lengths, angle weights and constraint weight must be positive")

    normal = _cp_normal(xyz)
    reference_angles = _valence_angles(xyz)
    reference_centered = xyz - np.mean(xyz, axis=0)

    def residual(vector: np.ndarray) -> np.ndarray:
        coordinates = vector.reshape(ring_size, 3)
        return np.concatenate(
            (
                constraint_weight
                * (cremer_pople_cartesian_components(coordinates, reference_normal=normal) - target),
                constraint_weight * (_bond_lengths(coordinates) - lengths),
                np.sqrt(weights) * (_valence_angles(coordinates) - reference_angles),
                10.0 * np.mean(coordinates, axis=0),
                10.0
                * np.mean(np.cross(reference_centered, coordinates), axis=0),
            )
        )

    def jacobian(vector: np.ndarray):
        coordinates = vector.reshape(ring_size, 3)
        row_count = len(target) + 2 * ring_size + 6
        # These active ring blocks are small enough that the trust-region
        # factorization is dense.  Building the Jacobian directly in its final
        # representation avoids hundreds of thousands of Python-level sparse
        # scalar insertions for large polycycles.
        matrix = np.zeros((row_count, vector.size), dtype=float)
        matrix[: len(target), :] = constraint_weight * cremer_pople_cartesian_jacobian(
            coordinates,
            reference_normal=normal,
        )

        bond_offset = len(target)
        for atom in range(ring_size):
            neighbor = (atom + 1) % ring_size
            delta = coordinates[atom] - coordinates[neighbor]
            unit = delta / np.linalg.norm(delta)
            row = bond_offset + atom
            matrix[row, 3 * atom : 3 * atom + 3] = constraint_weight * unit
            matrix[row, 3 * neighbor : 3 * neighbor + 3] = -constraint_weight * unit

        angle_offset = bond_offset + ring_size
        for atom in range(ring_size):
            indices = ((atom - 1) % ring_size, atom, (atom + 1) % ring_size)
            gradients = _angle_gradients(*(coordinates[index] for index in indices))
            scale = np.sqrt(weights[atom])
            for index, gradient in zip(indices, gradients, strict=True):
                matrix[
                    angle_offset + atom,
                    3 * index : 3 * index + 3,
                ] = scale * gradient

        center_offset = angle_offset + ring_size
        center_scale = 10.0 / ring_size
        for atom in range(ring_size):
            for axis in range(3):
                matrix[center_offset + axis, 3 * atom + axis] = center_scale
            matrix[
                center_offset + 3 : center_offset + 6,
                3 * atom : 3 * atom + 3,
            ] = 10.0 / ring_size * _cross_matrix(reference_centered[atom])
        return matrix

    inverse_guess = _inverse_cp_initial_guess(xyz, target, normal).reshape(-1)
    initial_guess = _best_linearized_initial(
        residual,
        jacobian,
        xyz.reshape(-1),
        inverse_guess,
    )
    result = least_squares(
        residual,
        initial_guess,
        jac=jacobian,
        x_scale="jac",
        max_nfev=1000,
        xtol=1.0e-8,
        ftol=1.0e-8,
        gtol=1.0e-8,
    )
    coordinates = result.x.reshape(ring_size, 3)
    cp_residual = float(
        np.linalg.norm(
            cremer_pople_cartesian_components(coordinates, reference_normal=normal) - target
        )
    )
    bond_error = float(np.max(np.abs(_bond_lengths(coordinates) - lengths)))
    angle_deviation = np.degrees(_valence_angles(coordinates) - reference_angles)
    return RingReconstructionResult(
        coordinates=coordinates,
        converged=bool(cp_residual < 2.0e-8 and bond_error < 2.0e-9),
        evaluations=int(result.nfev),
        cp_residual_norm_angstrom=cp_residual,
        maximum_bond_error_angstrom=bond_error,
        angle_rms_degrees=float(np.sqrt(np.mean(angle_deviation**2))),
        angle_max_degrees=float(np.max(np.abs(angle_deviation))),
    )


def reconstruct_ring_system_from_cp(
    reference_coordinates: np.ndarray,
    selected_ring_atoms: Sequence[int],
    ring_system_atoms: Sequence[int],
    bonds: Iterable[tuple[int, int]],
    target_components: Sequence[float],
    *,
    angle_weights: Sequence[float] | None = None,
    constraint_weight: float = 1.0e5,
) -> RingSystemReconstructionResult:
    """Realize one CP target while solving a fused or bridged ring block.

    Every bond internal to the connected ring system is constrained to its
    reference length. All ring-system valence angles enter the soft objective,
    so shared atoms and closures are optimized simultaneously instead of
    moving one ring against a frozen polycyclic core.
    """

    reference = _coordinates(reference_coordinates)
    selected = tuple(int(atom) for atom in selected_ring_atoms)
    system = tuple(dict.fromkeys(int(atom) for atom in ring_system_atoms))
    if len(selected) < 4 or len(set(selected)) != len(selected):
        raise ValueError("selected_ring_atoms must be an ordered cycle")
    if not set(selected) <= set(system):
        raise ValueError("the selected ring must belong to the ring-system atom set")
    if min(system) < 0 or max(system) >= len(reference):
        raise IndexError("ring-system atom lies outside the molecular geometry")
    target = np.asarray(target_components, dtype=float)
    if target.shape != (len(selected) - 3,):
        raise ValueError("the selected ring requires exactly N-3 CP components")
    weights = (
        np.ones(len(selected), dtype=float)
        if angle_weights is None
        else np.asarray(angle_weights, dtype=float)
    )
    if weights.shape != (len(selected),) or np.any(weights <= 0.0):
        raise ValueError("angle_weights requires one positive value per selected-ring atom")
    if constraint_weight <= 0.0:
        raise ValueError("constraint_weight must be positive")

    system_set = set(system)
    system_bonds = tuple(
        sorted(
            {
                tuple(sorted((int(left), int(right))))
                for left, right in bonds
                if int(left) in system_set and int(right) in system_set
            }
        )
    )
    if not system_bonds:
        raise ValueError("the ring system contains no internal bonds")
    adjacency = {atom: set() for atom in system}
    for left, right in system_bonds:
        adjacency[left].add(right)
        adjacency[right].add(left)
    angle_triplets = tuple(
        (left, center, right)
        for center in system
        for left, right in combinations(sorted(adjacency[center]), 2)
    )
    reference_lengths = _indexed_bond_lengths(reference, system_bonds)
    reference_angles = _indexed_angles(reference, angle_triplets)
    selected_angle_weight = {
        (selected[(index - 1) % len(selected)], atom, selected[(index + 1) % len(selected)]): (
            weights[index]
        )
        for index, atom in enumerate(selected)
    }
    objective_weights = np.asarray(
        [
            selected_angle_weight.get(
                triplet,
                selected_angle_weight.get((triplet[2], triplet[1], triplet[0]), 1.0),
            )
            for triplet in angle_triplets
        ],
        dtype=float,
    )
    selected_indices = np.asarray([system.index(atom) for atom in selected], dtype=int)
    system_index = {atom: index for index, atom in enumerate(system)}
    normal = _cp_normal(reference[np.asarray(selected, dtype=int)])
    reference_center = np.mean(reference[np.asarray(system, dtype=int)], axis=0)
    system_reference = reference[np.asarray(system, dtype=int)]
    reference_centered = system_reference - reference_center

    initial_full = reference.copy()
    initial_full[np.asarray(selected, dtype=int)] = _inverse_cp_initial_guess(
        reference[np.asarray(selected, dtype=int)],
        target,
        normal,
    )
    initial = initial_full[np.asarray(system, dtype=int)]

    def expand(vector: np.ndarray) -> np.ndarray:
        coordinates = reference.copy()
        coordinates[np.asarray(system, dtype=int)] = vector.reshape(len(system), 3)
        return coordinates

    def residual(
        vector: np.ndarray,
        hard_weight: float = constraint_weight,
    ) -> np.ndarray:
        coordinates = expand(vector)
        system_coordinates = vector.reshape(len(system), 3)
        return np.concatenate(
            (
                hard_weight
                * (
                    cremer_pople_cartesian_components(
                        system_coordinates[selected_indices],
                        reference_normal=normal,
                    )
                    - target
                ),
                hard_weight
                * (_indexed_bond_lengths(coordinates, system_bonds) - reference_lengths),
                np.sqrt(objective_weights)
                * (_indexed_angles(coordinates, angle_triplets) - reference_angles),
                10.0 * (np.mean(system_coordinates, axis=0) - reference_center),
                10.0
                * np.mean(np.cross(reference_centered, system_coordinates), axis=0),
            )
        )

    def jacobian(
        vector: np.ndarray,
        hard_weight: float = constraint_weight,
    ):
        system_coordinates = vector.reshape(len(system), 3)
        row_count = len(target) + len(system_bonds) + len(angle_triplets) + 6
        matrix = np.zeros((row_count, vector.size), dtype=float)

        cp_jacobian = cremer_pople_cartesian_jacobian(
            system_coordinates[selected_indices],
            reference_normal=normal,
        )
        for local_atom, block_atom in enumerate(selected_indices):
            matrix[
                : len(target),
                3 * block_atom : 3 * block_atom + 3,
            ] = hard_weight * cp_jacobian[
                :, 3 * local_atom : 3 * local_atom + 3
            ]

        bond_offset = len(target)
        for bond_index, (left, right) in enumerate(system_bonds):
            left_index = system_index[left]
            right_index = system_index[right]
            delta = system_coordinates[left_index] - system_coordinates[right_index]
            unit = delta / np.linalg.norm(delta)
            row = bond_offset + bond_index
            matrix[row, 3 * left_index : 3 * left_index + 3] = (
                hard_weight * unit
            )
            matrix[row, 3 * right_index : 3 * right_index + 3] = (
                -hard_weight * unit
            )

        angle_offset = bond_offset + len(system_bonds)
        for angle_index, (left, center, right) in enumerate(angle_triplets):
            gradients = _angle_gradients(
                system_coordinates[system_index[left]],
                system_coordinates[system_index[center]],
                system_coordinates[system_index[right]],
            )
            row = angle_offset + angle_index
            scale = np.sqrt(objective_weights[angle_index])
            for atom, gradient in zip(
                (left, center, right), gradients, strict=True
            ):
                index = system_index[atom]
                matrix[row, 3 * index : 3 * index + 3] = scale * gradient

        center_offset = angle_offset + len(angle_triplets)
        center_scale = 10.0 / len(system)
        for atom in range(len(system)):
            for axis in range(3):
                matrix[center_offset + axis, 3 * atom + axis] = center_scale
            matrix[
                center_offset + 3 : center_offset + 6,
                3 * atom : 3 * atom + 3,
            ] = 10.0 / len(system) * _cross_matrix(reference_centered[atom])
        return matrix

    reference_vector = reference[np.asarray(system, dtype=int)].reshape(-1)
    initial_vector = _best_linearized_initial(
        residual,
        jacobian,
        reference_vector,
        initial.reshape(-1),
    )
    # A short, moderately weighted phase first locates the angle-preserving
    # manifold without exposing the trust-region factorization immediately to
    # the five-order penalty ratio.  The full-weight phase then tightens the
    # CP and bond constraints to the acceptance tolerance.
    conditioning_weight = min(constraint_weight, 1.0e3)
    conditioning_evaluations = 0
    if conditioning_weight < constraint_weight:
        conditioned = least_squares(
            lambda vector: residual(vector, conditioning_weight),
            initial_vector,
            jac=lambda vector: jacobian(vector, conditioning_weight),
            x_scale="jac",
            max_nfev=100,
            xtol=1.0e-6,
            ftol=1.0e-6,
            gtol=1.0e-6,
        )
        initial_vector = conditioned.x
        conditioning_evaluations = int(conditioned.nfev)
    result = least_squares(
        residual,
        initial_vector,
        jac=jacobian,
        x_scale="jac",
        max_nfev=100,
        xtol=1.0e-8,
        ftol=1.0e-8,
        gtol=1.0e-8,
    )
    coordinates = expand(result.x)
    cp_residual = float(
        np.linalg.norm(
            cremer_pople_cartesian_components(
                coordinates[np.asarray(selected, dtype=int)],
                reference_normal=normal,
            )
            - target
        )
    )
    bond_error = float(
        np.max(
            np.abs(_indexed_bond_lengths(coordinates, system_bonds) - reference_lengths)
        )
    )
    angle_deviation = np.degrees(
        _indexed_angles(coordinates, angle_triplets) - reference_angles
    )
    return RingSystemReconstructionResult(
        coordinates=coordinates,
        ring_system_atoms=system,
        converged=bool(cp_residual < 2.0e-8 and bond_error < 2.0e-9),
        evaluations=conditioning_evaluations + int(result.nfev),
        cp_residual_norm_angstrom=cp_residual,
        maximum_bond_error_angstrom=bond_error,
        angle_rms_degrees=float(np.sqrt(np.mean(angle_deviation**2))),
        angle_max_degrees=float(np.max(np.abs(angle_deviation))),
    )


def detach_ring_substituents(
    coordinates: np.ndarray,
    bonds: Iterable[tuple[int, int]],
    ring_atoms: Sequence[int],
    *,
    protected_atoms: Iterable[int] | None = None,
) -> RingSubstituentPlan:
    """Identify removable single-anchor components outside a selected ring.

    ``protected_atoms`` should contain the union of all atoms belonging to any
    ring in a polycyclic system.  It records ring-system membership for the
    orchestration manifest.  Classification itself follows attachment
    multiplicity after the selected ring is cut: every single-anchor component
    is transported rigidly, including a second ring at a spiro atom, whereas
    fused and bridged components with two or more anchors remain in the core.
    """

    xyz = _coordinates(coordinates)
    ring = tuple(int(atom) for atom in ring_atoms)
    if len(ring) < 4 or len(set(ring)) != len(ring):
        raise ValueError("ring_atoms must be an ordered cycle of at least four unique atoms")
    if min(ring) < 0 or max(ring) >= len(xyz):
        raise IndexError("ring atom lies outside the molecular geometry")
    protected = set(ring)
    if protected_atoms is not None:
        protected.update(int(atom) for atom in protected_atoms)
    if min(protected) < 0 or max(protected) >= len(xyz):
        raise IndexError("protected atom lies outside the molecular geometry")

    adjacency = [set() for _ in range(len(xyz))]
    for left, right in bonds:
        left, right = int(left), int(right)
        if left == right or min(left, right) < 0 or max(left, right) >= len(xyz):
            raise ValueError("invalid molecular bond")
        adjacency[left].add(right)
        adjacency[right].add(left)

    candidates = set(range(len(xyz))) - set(ring)
    unvisited = set(candidates)
    components: list[DetachedSubstituent] = []
    detached_atoms: set[int] = set()
    while unvisited:
        seed = min(unvisited)
        stack = [seed]
        component: set[int] = set()
        while stack:
            atom = stack.pop()
            if atom not in unvisited:
                continue
            unvisited.remove(atom)
            component.add(atom)
            stack.extend(sorted(adjacency[atom] & unvisited, reverse=True))
        attachments = {
            neighbor
            for atom in component
            for neighbor in adjacency[atom]
            if neighbor in ring
        }
        if len(attachments) == 1:
            anchor = next(iter(attachments))
            if anchor in ring:
                atoms = tuple(sorted(component))
                position = ring.index(anchor)
                components.append(
                    DetachedSubstituent(
                        anchor=anchor,
                        atoms=atoms,
                        frame_neighbors=(
                            ring[(position - 1) % len(ring)],
                            ring[(position + 1) % len(ring)],
                        ),
                    )
                )
                detached_atoms.update(atoms)

    return RingSubstituentPlan(
        ring_atoms=ring,
        protected_atoms=tuple(sorted(protected)),
        core_atoms=tuple(atom for atom in range(len(xyz)) if atom not in detached_atoms),
        components=tuple(sorted(components, key=lambda item: (item.anchor, item.atoms))),
        reference_coordinates=xyz,
    )


def detach_ring_system_substituents(
    coordinates: np.ndarray,
    bonds: Iterable[tuple[int, int]],
    rings: Sequence[Sequence[int]],
    *,
    selected_ring_index: int = 0,
) -> RingSubstituentPlan:
    """Detach mono-anchor components from a connected multi-ring system."""

    xyz = _coordinates(coordinates)
    ordered_rings = tuple(tuple(int(atom) for atom in ring) for ring in rings)
    if not ordered_rings or any(len(ring) < 4 for ring in ordered_rings):
        raise ValueError("rings must contain ordered cycles of at least four atoms")
    if selected_ring_index < 0 or selected_ring_index >= len(ordered_rings):
        raise IndexError("selected ring index lies outside the connected ring system")
    protected = set().union(*(set(ring) for ring in ordered_rings))
    if min(protected) < 0 or max(protected) >= len(xyz):
        raise IndexError("ring-system atom lies outside the molecular geometry")

    adjacency = [set() for _ in range(len(xyz))]
    for left, right in bonds:
        left, right = int(left), int(right)
        if left == right or min(left, right) < 0 or max(left, right) >= len(xyz):
            raise ValueError("invalid molecular bond")
        adjacency[left].add(right)
        adjacency[right].add(left)

    unvisited = set(range(len(xyz))) - protected
    components: list[DetachedSubstituent] = []
    detached_atoms: set[int] = set()
    while unvisited:
        seed = min(unvisited)
        stack = [seed]
        component: set[int] = set()
        while stack:
            atom = stack.pop()
            if atom not in unvisited:
                continue
            unvisited.remove(atom)
            component.add(atom)
            stack.extend(sorted(adjacency[atom] & unvisited, reverse=True))
        attachments = {
            neighbor
            for atom in component
            for neighbor in adjacency[atom]
            if neighbor in protected
        }
        if len(attachments) != 1:
            continue
        anchor = next(iter(attachments))
        anchor_ring = next(ring for ring in ordered_rings if anchor in ring)
        position = anchor_ring.index(anchor)
        atoms = tuple(sorted(component))
        components.append(
            DetachedSubstituent(
                anchor=anchor,
                atoms=atoms,
                frame_neighbors=(
                    anchor_ring[(position - 1) % len(anchor_ring)],
                    anchor_ring[(position + 1) % len(anchor_ring)],
                ),
            )
        )
        detached_atoms.update(atoms)

    selected = ordered_rings[selected_ring_index]
    return RingSubstituentPlan(
        ring_atoms=selected,
        protected_atoms=tuple(sorted(protected)),
        core_atoms=tuple(atom for atom in range(len(xyz)) if atom not in detached_atoms),
        components=tuple(sorted(components, key=lambda item: (item.anchor, item.atoms))),
        reference_coordinates=xyz,
    )


def reposition_ring_substituents(
    plan: RingSubstituentPlan,
    updated_coordinates: np.ndarray,
) -> np.ndarray:
    """Rigidly reattach every detached component in its anchor's local frame."""

    updated = _coordinates(updated_coordinates).copy()
    if updated.shape != plan.reference_coordinates.shape:
        raise ValueError("updated coordinates must contain the complete molecular geometry")
    ring = plan.ring_atoms
    for component in plan.components:
        old_frame = _local_anchor_frame(
            plan.reference_coordinates,
            component.anchor,
            component.frame_neighbors,
            fallback_ring=ring,
        )
        new_frame = _local_anchor_frame(
            updated,
            component.anchor,
            component.frame_neighbors,
            fallback_ring=ring,
        )
        rotation = new_frame @ old_frame.T
        old_anchor = plan.reference_coordinates[component.anchor]
        new_anchor = updated[component.anchor]
        indices = np.asarray(component.atoms, dtype=int)
        updated[indices] = new_anchor + (plan.reference_coordinates[indices] - old_anchor) @ rotation.T
    return updated


def substituent_rigidity_error(plan: RingSubstituentPlan, coordinates: np.ndarray) -> float:
    """Maximum distance error inside any anchor-plus-substituent component."""

    xyz = _coordinates(coordinates)
    maximum = 0.0
    for component in plan.components:
        indices = np.asarray((component.anchor, *component.atoms), dtype=int)
        old = plan.reference_coordinates[indices]
        new = xyz[indices]
        maximum = max(maximum, float(np.max(np.abs(_distance_matrix(old) - _distance_matrix(new)))))
    return maximum


def _local_ring_frame(coordinates: np.ndarray, ring: tuple[int, ...], anchor: int) -> np.ndarray:
    position = ring.index(anchor)
    center = coordinates[anchor]
    forward = coordinates[ring[(position + 1) % len(ring)]] - center
    backward = coordinates[ring[(position - 1) % len(ring)]] - center
    x_axis = _unit(forward)
    y_axis = backward - float(backward @ x_axis) * x_axis
    y_axis = _unit(y_axis)
    z_axis = _unit(np.cross(x_axis, y_axis))
    return np.column_stack((x_axis, y_axis, z_axis))


def _local_anchor_frame(
    coordinates: np.ndarray,
    anchor: int,
    neighbors: tuple[int, int],
    *,
    fallback_ring: tuple[int, ...],
) -> np.ndarray:
    if len(neighbors) != 2:
        return _local_ring_frame(coordinates, fallback_ring, anchor)
    center = coordinates[anchor]
    backward = coordinates[neighbors[0]] - center
    forward = coordinates[neighbors[1]] - center
    x_axis = _unit(forward)
    y_axis = _unit(backward - float(backward @ x_axis) * x_axis)
    z_axis = _unit(np.cross(x_axis, y_axis))
    return np.column_stack((x_axis, y_axis, z_axis))


def _inverse_cp_initial_guess(
    reference: np.ndarray,
    target: np.ndarray,
    normal: np.ndarray,
) -> np.ndarray:
    ring_size = len(reference)
    atom = np.arange(ring_size, dtype=float)
    delta = target - cremer_pople_cartesian_components(
        reference,
        reference_normal=normal,
    )
    heights = np.zeros(ring_size, dtype=float)
    offset = 0
    for harmonic in range(2, (ring_size - 1) // 2 + 1):
        phase = 2.0 * np.pi * harmonic * atom / ring_size
        scale = np.sqrt(2.0 / ring_size)
        heights += scale * (
            delta[offset] * np.cos(phase) + delta[offset + 1] * np.sin(phase)
        )
        offset += 2
    if ring_size % 2 == 0:
        heights += delta[offset] / np.sqrt(ring_size) * (-1.0) ** atom
    return reference + heights[:, None] * normal[None, :]


def _cp_fourier_rows(ring_size: int) -> tuple[np.ndarray, ...]:
    atom = np.arange(ring_size, dtype=float)
    rows = []
    for harmonic in range(2, (ring_size - 1) // 2 + 1):
        phase = 2.0 * np.pi * harmonic * atom / ring_size
        scale = np.sqrt(2.0 / ring_size)
        rows.extend((scale * np.cos(phase), scale * np.sin(phase)))
    if ring_size % 2 == 0:
        rows.append(((-1.0) ** atom) / np.sqrt(ring_size))
    return tuple(rows)


def _best_linearized_initial(
    residual,
    jacobian,
    reference: np.ndarray,
    direct_guess: np.ndarray,
) -> np.ndarray:
    """Choose between direct CP inversion and a whole-block linear predictor."""

    reference_vector = np.asarray(reference, dtype=float).reshape(-1)
    direct_vector = np.asarray(direct_guess, dtype=float).reshape(-1)
    reference_residual = np.asarray(residual(reference_vector), dtype=float)
    reference_jacobian = np.asarray(jacobian(reference_vector), dtype=float)
    step, *_ = np.linalg.lstsq(
        reference_jacobian,
        -reference_residual,
        rcond=1.0e-12,
    )
    linearized = reference_vector + step
    candidates = (direct_vector, linearized)
    return min(candidates, key=lambda item: float(np.linalg.norm(residual(item))))


def _cross_matrix(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=float)
    return np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


def _angle_gradients(
    left: np.ndarray,
    center: np.ndarray,
    right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first = left - center
    second = right - center
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= 1.0e-12 or second_norm <= 1.0e-12:
        raise FloatingPointError("zero-length angle arm")
    first_unit = first / first_norm
    second_unit = second / second_norm
    cosine = float(np.clip(first_unit @ second_unit, -1.0, 1.0))
    sine = float(np.sqrt(max(1.0 - cosine * cosine, 0.0)))
    if sine <= 1.0e-12:
        raise FloatingPointError("linear angle has no ordinary derivative")
    left_gradient = (cosine * first_unit - second_unit) / (first_norm * sine)
    right_gradient = (cosine * second_unit - first_unit) / (second_norm * sine)
    return left_gradient, -(left_gradient + right_gradient), right_gradient


def _cp_normal(coordinates: np.ndarray) -> np.ndarray:
    centered = coordinates - np.mean(coordinates, axis=0)
    phase = 2.0 * np.pi * np.arange(len(coordinates), dtype=float) / len(coordinates)
    sine = np.sum(centered * np.sin(phase)[:, None], axis=0)
    cosine = np.sum(centered * np.cos(phase)[:, None], axis=0)
    return _unit(np.cross(sine, cosine))


def _bond_lengths(coordinates: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.roll(coordinates, -1, axis=0) - coordinates, axis=1)


def _valence_angles(coordinates: np.ndarray) -> np.ndarray:
    left = np.roll(coordinates, 1, axis=0) - coordinates
    right = np.roll(coordinates, -1, axis=0) - coordinates
    cosine = np.sum(left * right, axis=1) / (
        np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    )
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def _distance_matrix(coordinates: np.ndarray) -> np.ndarray:
    displacement = coordinates[:, None, :] - coordinates[None, :, :]
    return np.linalg.norm(displacement, axis=2)


def _indexed_bond_lengths(
    coordinates: np.ndarray,
    bonds: Sequence[tuple[int, int]],
) -> np.ndarray:
    return np.asarray(
        [np.linalg.norm(coordinates[left] - coordinates[right]) for left, right in bonds],
        dtype=float,
    )


def _indexed_angles(
    coordinates: np.ndarray,
    triplets: Sequence[tuple[int, int, int]],
) -> np.ndarray:
    values = []
    for left, center, right in triplets:
        first = coordinates[left] - coordinates[center]
        second = coordinates[right] - coordinates[center]
        cosine = float(first @ second) / float(np.linalg.norm(first) * np.linalg.norm(second))
        values.append(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return np.asarray(values, dtype=float)


def _coordinates(value: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(value, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or len(coordinates) < 4:
        raise ValueError("coordinates must have shape (N, 3) with N >= 4")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("coordinates must be finite")
    return coordinates


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-12:
        raise ValueError("cannot construct a local frame from degenerate ring geometry")
    return vector / norm
