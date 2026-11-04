"""Symmetric dual-tree operator for the long-range SCC shell kernel."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HierarchicalSCCEvaluation:
    potential: np.ndarray
    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray
    energy_error_bound_hartree: float
    accepted_cluster_pairs: int
    exact_shell_pairs: int


@dataclass
class _Node:
    indices: np.ndarray
    center: np.ndarray
    radius: float
    children: tuple["_Node", ...]
    type_members: dict[float, np.ndarray]
    type_centers: dict[float, np.ndarray]
    type_radii: dict[float, float]


class HierarchicalShellInteraction:
    """Matrix-free softened-Coulomb operator with a symmetric dual tree.

    The approximation is an energy functional.  Potentials are its charge
    derivatives and forces its coordinate derivatives, so SCC stationarity
    and translational invariance are preserved by construction.
    """

    def __init__(
        self,
        coordinates_bohr: np.ndarray,
        shell_atom: tuple[int, ...],
        hardness_hartree: tuple[float, ...],
        *,
        opening_angle: float = 0.35,
        leaf_size: int = 16,
        energy_tolerance_hartree: float | None = None,
        kernel_mode: str = "softened",
        exact_cutoff_bohr: float = 0.0,
    ) -> None:
        coordinates = np.asarray(coordinates_bohr, dtype=float)
        assignment = np.asarray(shell_atom, dtype=np.intp)
        hardness = np.asarray(hardness_hartree, dtype=float)
        if assignment.shape != hardness.shape or np.any(hardness <= 0.0):
            raise ValueError("shell assignment and positive hardnesses must match")
        if opening_angle < 0.0 or opening_angle >= 1.0:
            raise ValueError("opening_angle must lie in [0, 1)")
        if leaf_size < 1:
            raise ValueError("leaf_size must be positive")
        if kernel_mode not in {"softened", "coulomb-correction"}:
            raise ValueError("unknown hierarchical SCC kernel mode")
        if exact_cutoff_bohr < 0.0:
            raise ValueError("exact_cutoff_bohr must be nonnegative")
        self.atomic_coordinates = coordinates
        self.shell_atom = assignment
        self.coordinates = coordinates[assignment]
        self.hardness = hardness
        self.opening_angle = float(opening_angle)
        self.leaf_size = int(leaf_size)
        self.energy_tolerance_hartree = energy_tolerance_hartree
        self.kernel_mode = kernel_mode
        self.exact_cutoff_bohr = float(exact_cutoff_bohr)
        self._fallback: HierarchicalShellInteraction | None = None
        self.root = self._build(np.arange(len(assignment), dtype=np.intp))
        self._cluster_pairs: list[tuple[_Node, _Node]] = []
        self._exact_pairs: list[tuple[np.ndarray, np.ndarray, bool]] = []
        self._plan(self.root, self.root)
        self._compile_exact_plan()
        self._compile_cluster_plan()

    def _compile_exact_plan(self) -> None:
        """Flatten invariant near-field pairs and kernels once per geometry."""

        left_parts: list[np.ndarray] = []
        right_parts: list[np.ndarray] = []
        for left_indices, right_indices, same in self._exact_pairs:
            if same:
                positions = np.tril_indices(len(left_indices), k=-1)
                left = left_indices[positions[0]]
                right = left_indices[positions[1]]
            else:
                left = np.repeat(left_indices, len(right_indices))
                right = np.tile(right_indices, len(left_indices))
            if self.kernel_mode == "coulomb-correction":
                distinct = self.shell_atom[left] != self.shell_atom[right]
                left = left[distinct]
                right = right[distinct]
            if len(left):
                left_parts.append(left)
                right_parts.append(right)
        self._exact_left = (
            np.concatenate(left_parts) if left_parts else np.empty(0, dtype=np.intp)
        )
        self._exact_right = (
            np.concatenate(right_parts) if right_parts else np.empty(0, dtype=np.intp)
        )
        displacement = (
            self.coordinates[self._exact_left] - self.coordinates[self._exact_right]
        )
        distance2 = np.einsum("ij,ij->i", displacement, displacement)
        average = 0.5 * (
            self.hardness[self._exact_left] + self.hardness[self._exact_right]
        )
        kernel = (distance2 + average**-2) ** -0.5
        radial = -(distance2 + average**-2) ** -1.5
        if self.kernel_mode == "coulomb-correction":
            kernel -= distance2**-0.5
            radial += distance2**-1.5
        self._exact_displacement = displacement
        self._exact_kernel = kernel
        self._exact_radial = radial

    def _compile_cluster_plan(self) -> None:
        """Flatten typed far-field interactions into vectorized incidence lists."""

        left_members: list[np.ndarray] = []
        right_members: list[np.ndarray] = []
        left_ids: list[np.ndarray] = []
        right_ids: list[np.ndarray] = []
        kernels: list[float] = []
        radial: list[float] = []
        radial_derivative: list[float] = []
        displacement: list[np.ndarray] = []
        deviation: list[float] = []
        left_sizes: list[int] = []
        right_sizes: list[int] = []
        interaction = 0
        for left_node, right_node in self._cluster_pairs:
            for left_type, left_group in left_node.type_members.items():
                for right_type, right_group in right_node.type_members.items():
                    vector = (
                        left_node.type_centers[left_type]
                        - right_node.type_centers[right_type]
                    )
                    distance = float(np.linalg.norm(vector))
                    kernel = self._kernel(
                        distance * distance, left_type, right_type
                    )
                    denominator = (
                        distance * distance
                        + (0.5 * (left_type + right_type)) ** -2
                    ) ** 1.5
                    radial_value = -1.0 / denominator
                    radial_derivative_value = 3.0 / denominator ** (5.0 / 3.0)
                    if self.kernel_mode == "coulomb-correction":
                        radial_value += distance**-3
                        radial_derivative_value -= 3.0 * distance**-5
                    spread = (
                        left_node.type_radii[left_type]
                        + right_node.type_radii[right_type]
                    )
                    low = max(distance - spread, 0.0)
                    high = distance + spread
                    error = max(
                        abs(
                            kernel
                            - self._kernel(low * low, left_type, right_type)
                        ),
                        abs(
                            kernel
                            - self._kernel(high * high, left_type, right_type)
                        ),
                    )
                    left_members.append(left_group)
                    right_members.append(right_group)
                    left_ids.append(
                        np.full(len(left_group), interaction, dtype=np.intp)
                    )
                    right_ids.append(
                        np.full(len(right_group), interaction, dtype=np.intp)
                    )
                    kernels.append(kernel)
                    radial.append(radial_value)
                    radial_derivative.append(radial_derivative_value)
                    displacement.append(vector)
                    deviation.append(error)
                    left_sizes.append(len(left_group))
                    right_sizes.append(len(right_group))
                    interaction += 1
        empty_i = np.empty(0, dtype=np.intp)
        self._cluster_left_members = (
            np.concatenate(left_members) if left_members else empty_i
        )
        self._cluster_right_members = (
            np.concatenate(right_members) if right_members else empty_i
        )
        self._cluster_left_ids = np.concatenate(left_ids) if left_ids else empty_i
        self._cluster_right_ids = (
            np.concatenate(right_ids) if right_ids else empty_i
        )
        self._cluster_kernel = np.asarray(kernels, dtype=float)
        self._cluster_radial = np.asarray(radial, dtype=float)
        self._cluster_radial_derivative = np.asarray(
            radial_derivative, dtype=float
        )
        self._cluster_displacement = np.asarray(displacement, dtype=float).reshape(-1, 3)
        self._cluster_deviation = np.asarray(deviation, dtype=float)
        self._cluster_left_sizes = np.asarray(left_sizes, dtype=float)
        self._cluster_right_sizes = np.asarray(right_sizes, dtype=float)

    def _build(self, indices: np.ndarray) -> _Node:
        points = self.coordinates[indices]
        center = np.mean(points, axis=0)
        radius = float(np.max(np.linalg.norm(points - center, axis=1), initial=0.0))
        type_members: dict[float, np.ndarray] = {}
        type_centers: dict[float, np.ndarray] = {}
        type_radii: dict[float, float] = {}
        for value in np.unique(self.hardness[indices]):
            members = indices[self.hardness[indices] == value]
            type_members[float(value)] = members
            type_center = np.mean(self.coordinates[members], axis=0)
            type_centers[float(value)] = type_center
            type_radii[float(value)] = float(
                np.max(
                    np.linalg.norm(self.coordinates[members] - type_center, axis=1),
                    initial=0.0,
                )
            )
        if len(indices) <= self.leaf_size:
            children: tuple[_Node, ...] = ()
        else:
            extent = np.ptp(points, axis=0)
            axis = int(np.argmax(extent))
            order = indices[np.argsort(points[:, axis], kind="stable")]
            middle = len(order) // 2
            children = (self._build(order[:middle]), self._build(order[middle:]))
        return _Node(
            indices, center, radius, children,
            type_members, type_centers, type_radii,
        )

    def _well_separated(self, left: _Node, right: _Node) -> bool:
        if self.opening_angle == 0.0:
            return False
        distance = float(np.linalg.norm(left.center - right.center))
        if distance - left.radius - right.radius <= self.exact_cutoff_bohr:
            return False
        return distance > 0.0 and (
            left.radius + right.radius
        ) / distance <= self.opening_angle

    def _plan(self, left: _Node, right: _Node) -> None:
        if left is right:
            if not left.children:
                self._exact_pairs.append((left.indices, left.indices, True))
                return
            for index, child in enumerate(left.children):
                self._plan(child, child)
                for other in left.children[:index]:
                    self._plan(child, other)
            return
        if self._well_separated(left, right):
            self._cluster_pairs.append((left, right))
            return
        if not left.children and not right.children:
            self._exact_pairs.append((left.indices, right.indices, False))
            return
        if right.children and (not left.children or right.radius > left.radius):
            for child in right.children:
                self._plan(left, child)
        else:
            for child in left.children:
                self._plan(child, right)

    def _kernel(self, distance2: float, left_hardness: float, right_hardness: float) -> float:
        average = 0.5 * (left_hardness + right_hardness)
        softened = float((distance2 + average**-2) ** -0.5)
        if self.kernel_mode == "softened":
            return softened
        if distance2 == 0.0:
            return 0.0
        return softened - distance2**-0.5

    def evaluate(
        self, charges: np.ndarray, *, compute_gradient: bool = True
    ) -> HierarchicalSCCEvaluation:
        if self._fallback is not None:
            return self._fallback.evaluate(charges, compute_gradient=compute_gradient)
        charges = np.asarray(charges, dtype=float)
        if charges.shape != self.hardness.shape:
            raise ValueError("one SCC charge is required per shell")
        potential = np.zeros_like(charges)
        shell_gradient = (
            np.zeros((len(charges), 3), dtype=float) if compute_gradient else None
        )
        error_bound = 0.0
        if self.kernel_mode == "softened":
            potential += self.hardness * charges
        exact_left = self._exact_left
        exact_right = self._exact_right
        if len(exact_left):
            potential += np.bincount(
                exact_left,
                weights=charges[exact_right] * self._exact_kernel,
                minlength=len(charges),
            )
            potential += np.bincount(
                exact_right,
                weights=charges[exact_left] * self._exact_kernel,
                minlength=len(charges),
            )
            if shell_gradient is not None:
                force = (
                    charges[exact_left, None]
                    * charges[exact_right, None]
                    * self._exact_radial[:, None]
                    * self._exact_displacement
                )
                np.add.at(shell_gradient, exact_left, force)
                np.add.at(shell_gradient, exact_right, -force)
        exact_count = len(exact_left)
        cluster_count = len(self._cluster_kernel)
        if cluster_count:
            left_charge = np.bincount(
                self._cluster_left_ids,
                weights=charges[self._cluster_left_members],
                minlength=cluster_count,
            )
            right_charge = np.bincount(
                self._cluster_right_ids,
                weights=charges[self._cluster_right_members],
                minlength=cluster_count,
            )
            left_absolute = np.bincount(
                self._cluster_left_ids,
                weights=np.abs(charges[self._cluster_left_members]),
                minlength=cluster_count,
            )
            right_absolute = np.bincount(
                self._cluster_right_ids,
                weights=np.abs(charges[self._cluster_right_members]),
                minlength=cluster_count,
            )
            potential += np.bincount(
                self._cluster_left_members,
                weights=(
                    right_charge[self._cluster_left_ids]
                    * self._cluster_kernel[self._cluster_left_ids]
                ),
                minlength=len(charges),
            )
            potential += np.bincount(
                self._cluster_right_members,
                weights=(
                    left_charge[self._cluster_right_ids]
                    * self._cluster_kernel[self._cluster_right_ids]
                ),
                minlength=len(charges),
            )
            if shell_gradient is not None:
                force = (
                    left_charge[:, None] * right_charge[:, None]
                    * self._cluster_radial[:, None]
                    * self._cluster_displacement
                )
                for axis in range(3):
                    shell_gradient[:, axis] += np.bincount(
                        self._cluster_left_members,
                        weights=(
                            force[self._cluster_left_ids, axis]
                            / self._cluster_left_sizes[self._cluster_left_ids]
                        ),
                        minlength=len(charges),
                    )
                    shell_gradient[:, axis] -= np.bincount(
                        self._cluster_right_members,
                        weights=(
                            force[self._cluster_right_ids, axis]
                            / self._cluster_right_sizes[self._cluster_right_ids]
                        ),
                        minlength=len(charges),
                    )
            error_bound = float(
                np.sum(
                    left_absolute * right_absolute * self._cluster_deviation
                )
            )
        atomic_gradient = np.zeros_like(self.atomic_coordinates)
        if shell_gradient is not None:
            np.add.at(atomic_gradient, self.shell_atom, shell_gradient)
        result = HierarchicalSCCEvaluation(
            potential,
            0.5 * float(charges @ potential),
            atomic_gradient,
            float(error_bound),
            len(self._cluster_pairs),
            exact_count,
        )
        if (
            self.energy_tolerance_hartree is not None
            and result.energy_error_bound_hartree > self.energy_tolerance_hartree
            and self.opening_angle > 0.0
        ):
            next_angle = self.opening_angle * 0.5
            if next_angle < 0.025:
                next_angle = 0.0
            self._fallback = HierarchicalShellInteraction(
                self.atomic_coordinates,
                tuple(int(value) for value in self.shell_atom),
                tuple(float(value) for value in self.hardness),
                opening_angle=next_angle,
                leaf_size=self.leaf_size,
                energy_tolerance_hartree=self.energy_tolerance_hartree,
                kernel_mode=self.kernel_mode,
                exact_cutoff_bohr=self.exact_cutoff_bohr,
            )
            return self._fallback.evaluate(charges, compute_gradient=compute_gradient)
        return result

    def hessian_vector_product(
        self, charges: np.ndarray, direction_bohr: np.ndarray
    ) -> np.ndarray:
        """Apply the fixed-charge coordinate Hessian of the tree functional.

        The stored near/far plan is reused.  No shell-pair Hessian is
        materialized, and accepted far-field nodes are differentiated as the
        same symmetric cluster functional used for energy and forces.
        """

        if self._fallback is not None:
            return self._fallback.hessian_vector_product(charges, direction_bohr)
        charges = np.asarray(charges, dtype=float)
        direction = np.asarray(direction_bohr, dtype=float)
        if charges.shape != self.hardness.shape:
            raise ValueError("one SCC charge is required per shell")
        if direction.shape != self.atomic_coordinates.shape:
            raise ValueError("SCC Hessian direction must have shape natoms x 3")
        shell_direction = direction[self.shell_atom]
        shell_product = np.zeros((len(charges), 3), dtype=float)

        left, right = self._exact_left, self._exact_right
        if len(left):
            displacement = self._exact_displacement
            relative = shell_direction[left] - shell_direction[right]
            distance2 = np.einsum("ij,ij->i", displacement, displacement)
            average = 0.5 * (self.hardness[left] + self.hardness[right])
            softened = distance2 + average**-2
            radial = -softened**-1.5
            radial_derivative = 3.0 * softened**-2.5
            if self.kernel_mode == "coulomb-correction":
                radial += distance2**-1.5
                radial_derivative -= 3.0 * distance2**-2.5
            projection = np.einsum("ij,ij->i", displacement, relative)
            product = charges[left, None] * charges[right, None] * (
                radial[:, None] * relative
                + radial_derivative[:, None] * projection[:, None] * displacement
            )
            np.add.at(shell_product, left, product)
            np.add.at(shell_product, right, -product)

        cluster_count = len(self._cluster_kernel)
        if cluster_count:
            left_charge = np.bincount(
                self._cluster_left_ids,
                weights=charges[self._cluster_left_members],
                minlength=cluster_count,
            )
            right_charge = np.bincount(
                self._cluster_right_ids,
                weights=charges[self._cluster_right_members],
                minlength=cluster_count,
            )
            left_direction = np.column_stack(
                tuple(
                    np.bincount(
                        self._cluster_left_ids,
                        weights=shell_direction[self._cluster_left_members, axis],
                        minlength=cluster_count,
                    )
                    / self._cluster_left_sizes
                    for axis in range(3)
                )
            )
            right_direction = np.column_stack(
                tuple(
                    np.bincount(
                        self._cluster_right_ids,
                        weights=shell_direction[self._cluster_right_members, axis],
                        minlength=cluster_count,
                    )
                    / self._cluster_right_sizes
                    for axis in range(3)
                )
            )
            relative = left_direction - right_direction
            displacement = self._cluster_displacement
            projection = np.einsum("ij,ij->i", displacement, relative)
            product = left_charge[:, None] * right_charge[:, None] * (
                self._cluster_radial[:, None] * relative
                + self._cluster_radial_derivative[:, None]
                * projection[:, None]
                * displacement
            )
            for axis in range(3):
                shell_product[:, axis] += np.bincount(
                    self._cluster_left_members,
                    weights=(
                        product[self._cluster_left_ids, axis]
                        / self._cluster_left_sizes[self._cluster_left_ids]
                    ),
                    minlength=len(charges),
                )
                shell_product[:, axis] -= np.bincount(
                    self._cluster_right_members,
                    weights=(
                        product[self._cluster_right_ids, axis]
                        / self._cluster_right_sizes[self._cluster_right_ids]
                    ),
                    minlength=len(charges),
                )
        atomic_product = np.zeros_like(self.atomic_coordinates)
        np.add.at(atomic_product, self.shell_atom, shell_product)
        atomic_product -= np.mean(atomic_product, axis=0, keepdims=True)
        return atomic_product

    def potential_direction(
        self,
        charges: np.ndarray,
        charge_direction: np.ndarray,
        direction_bohr: np.ndarray,
    ) -> np.ndarray:
        """Differentiate the hierarchical shell potential matrix-free."""

        if self._fallback is not None:
            return self._fallback.potential_direction(
                charges, charge_direction, direction_bohr
            )
        charges = np.asarray(charges, dtype=float)
        qdot = np.asarray(charge_direction, dtype=float)
        direction = np.asarray(direction_bohr, dtype=float)
        if charges.shape != self.hardness.shape or qdot.shape != charges.shape:
            raise ValueError("one SCC charge direction is required per shell")
        if direction.shape != self.atomic_coordinates.shape:
            raise ValueError("SCC potential direction must have shape natoms x 3")
        potential = self.evaluate(qdot, compute_gradient=False).potential.copy()
        shell_direction = direction[self.shell_atom]
        left, right = self._exact_left, self._exact_right
        if len(left):
            relative = shell_direction[left] - shell_direction[right]
            kernel_dot = self._exact_radial * np.einsum(
                "ij,ij->i", self._exact_displacement, relative
            )
            potential += np.bincount(
                left,
                weights=charges[right] * kernel_dot,
                minlength=len(charges),
            )
            potential += np.bincount(
                right,
                weights=charges[left] * kernel_dot,
                minlength=len(charges),
            )
        cluster_count = len(self._cluster_kernel)
        if cluster_count:
            left_charge = np.bincount(
                self._cluster_left_ids,
                weights=charges[self._cluster_left_members],
                minlength=cluster_count,
            )
            right_charge = np.bincount(
                self._cluster_right_ids,
                weights=charges[self._cluster_right_members],
                minlength=cluster_count,
            )
            left_direction = np.column_stack(
                tuple(
                    np.bincount(
                        self._cluster_left_ids,
                        weights=shell_direction[self._cluster_left_members, axis],
                        minlength=cluster_count,
                    )
                    / self._cluster_left_sizes
                    for axis in range(3)
                )
            )
            right_direction = np.column_stack(
                tuple(
                    np.bincount(
                        self._cluster_right_ids,
                        weights=shell_direction[self._cluster_right_members, axis],
                        minlength=cluster_count,
                    )
                    / self._cluster_right_sizes
                    for axis in range(3)
                )
            )
            kernel_dot = self._cluster_radial * np.einsum(
                "ij,ij->i",
                self._cluster_displacement,
                left_direction - right_direction,
            )
            potential += np.bincount(
                self._cluster_left_members,
                weights=(
                    right_charge[self._cluster_left_ids]
                    * kernel_dot[self._cluster_left_ids]
                ),
                minlength=len(charges),
            )
            potential += np.bincount(
                self._cluster_right_members,
                weights=(
                    left_charge[self._cluster_right_ids]
                    * kernel_dot[self._cluster_right_ids]
                ),
                minlength=len(charges),
            )
        return potential

    def _exact_batch(self, left, right, charges, potential, gradient) -> None:
        if not len(left):
            return
        displacement = self.coordinates[left] - self.coordinates[right]
        distance2 = np.einsum("ij,ij->i", displacement, displacement)
        average = 0.5 * (self.hardness[left] + self.hardness[right])
        kernel = (distance2 + average**-2) ** -0.5
        if self.kernel_mode == "coulomb-correction":
            kernel -= distance2**-0.5
        np.add.at(potential, left, charges[right] * kernel)
        np.add.at(potential, right, charges[left] * kernel)
        if gradient is not None:
            radial = -(distance2 + average**-2) ** -1.5
            if self.kernel_mode == "coulomb-correction":
                radial += distance2**-1.5
            force = charges[left, None] * charges[right, None] * radial[:, None] * displacement
            np.add.at(gradient, left, force)
            np.add.at(gradient, right, -force)

    def matvec(self, charges: np.ndarray) -> np.ndarray:
        return self.evaluate(charges).potential

    def __matmul__(self, charges: np.ndarray) -> np.ndarray:
        return self.matvec(charges)


__all__ = ["HierarchicalSCCEvaluation", "HierarchicalShellInteraction"]


class FMMShellInteraction:
    """Laplace FMM plus a hierarchical softened-kernel correction.

    No electrostatic pair is truncated.  The long-range ``1/r`` contribution
    is applied by FMM, while a symmetric dual tree evaluates ``gamma-1/r``.
    Near pairs are exact and well-separated clusters use a charge-dependent
    monopole expansion with an explicit energy-error bound.
    """

    def __init__(
        self,
        coordinates_bohr: np.ndarray,
        shell_atom: tuple[int, ...],
        hardness_hartree: tuple[float, ...],
        *,
        precision: float = 1.0e-10,
        correction_cutoff_bohr: float = 48.0,
        correction_opening_angle: float = 0.25,
        correction_leaf_size: int = 16,
    ) -> None:
        self.atomic_coordinates = np.asarray(coordinates_bohr, dtype=float)
        self.shell_atom = np.asarray(shell_atom, dtype=np.intp)
        self.hardness = np.asarray(hardness_hartree, dtype=float)
        # Keep the FMM source layout persistent.  Rebuilding a Fortran
        # transpose and repeatedly scattering shell charges into atoms inside
        # every SCF iteration was pure overhead and became visible in the
        # large-system prefactor.
        self._fmm_sources = np.asfortranarray(self.atomic_coordinates.T)
        self._atomic_count = len(self.atomic_coordinates)
        self.precision = float(precision)
        self.correction_cutoff_bohr = float(correction_cutoff_bohr)
        self.shells_by_atom = tuple(
            np.flatnonzero(self.shell_atom == atom)
            for atom in range(len(self.atomic_coordinates))
        )
        self.correction = HierarchicalShellInteraction(
            self.atomic_coordinates,
            tuple(int(value) for value in self.shell_atom),
            tuple(float(value) for value in self.hardness),
            opening_angle=correction_opening_angle,
            leaf_size=correction_leaf_size,
            kernel_mode="coulomb-correction",
            exact_cutoff_bohr=correction_cutoff_bohr,
        )
        self._cached_charges: np.ndarray | None = None
        self._cached_evaluation: HierarchicalSCCEvaluation | None = None
        self._cached_compute_gradient = False
        self._linear_history_charges: list[np.ndarray] = []
        self._linear_history_potentials: list[np.ndarray] = []
        self._linear_history_size = 12
        self._linear_reuse_tolerance = 1.0e-9

    def evaluate(
        self, charges: np.ndarray, *, compute_gradient: bool = True,
        allow_linear_reuse: bool = True,
    ) -> HierarchicalSCCEvaluation:
        try:
            from fmm3dpy import lfmm3d
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("FMM SCC requires fmm3dpy") from exc
        charges = np.asarray(charges, dtype=float)
        if (
            allow_linear_reuse
            and self._cached_charges is not None
            and np.array_equal(charges, self._cached_charges)
            and (not compute_gradient or self._cached_compute_gradient)
        ):
            assert self._cached_evaluation is not None
            return self._cached_evaluation
        # gamma is a fixed linear operator at one geometry.  DIIS-generated
        # charge vectors frequently lie in the span of earlier exact FMM
        # applications, so reconstruct gamma*q from the same coefficients.
        # The tight residual gate makes this an algebraic reuse, not a frozen
        # potential approximation.  Gradient requests always execute FMM.
        if (
            allow_linear_reuse
            and not compute_gradient
            and len(self._linear_history_charges) >= 2
        ):
            charge_basis = np.column_stack(self._linear_history_charges)
            coefficients, *_ = np.linalg.lstsq(charge_basis, charges, rcond=None)
            represented = charge_basis @ coefficients
            relative_residual = np.linalg.norm(charges - represented) / max(
                np.linalg.norm(charges), 1.0e-30
            )
            if relative_residual <= self._linear_reuse_tolerance:
                potential_basis = np.column_stack(self._linear_history_potentials)
                potential = potential_basis @ coefficients
                evaluation = HierarchicalSCCEvaluation(
                    potential,
                    0.5 * float(charges @ potential),
                    np.zeros_like(self.atomic_coordinates),
                    0.0,
                    0,
                    0,
                )
                self._cached_charges = charges.copy()
                self._cached_evaluation = evaluation
                self._cached_compute_gradient = False
                return evaluation
        atomic_charge = np.bincount(
            self.shell_atom, weights=charges, minlength=self._atomic_count
        )
        result = lfmm3d(
            eps=self.precision,
            sources=self._fmm_sources,
            charges=np.asfortranarray(atomic_charge),
            pg=2 if compute_gradient else 1,
        )
        four_pi = 4.0 * np.pi
        atomic_potential = four_pi * np.asarray(result.pot, dtype=float).reshape(-1)
        field = (
            four_pi * np.asarray(result.grad, dtype=float).reshape(3, -1).T
            if compute_gradient
            else None
        )
        potential = atomic_potential[self.shell_atom].copy()
        gradient = (
            atomic_charge[:, None] * field
            if field is not None
            else np.zeros_like(self.atomic_coordinates)
        )
        # Same-centre interactions are finite in gamma and absent from the
        # grouped point-charge FMM.
        for shells in self.shells_by_atom:
            for position, left in enumerate(shells):
                potential[left] += self.hardness[left] * charges[left]
                for right in shells[:position]:
                    average = 0.5 * (self.hardness[left] + self.hardness[right])
                    potential[left] += average * charges[right]
                    potential[right] += average * charges[left]
        correction = self.correction.evaluate(
            charges, compute_gradient=compute_gradient
        )
        potential += correction.potential
        gradient += correction.gradient_hartree_per_bohr
        # FMM roundoff can leave a minute rigid-translation component even
        # though the pair functional is translation invariant.  Remove only
        # that null-space component; internal forces and torques are unchanged.
        if compute_gradient:
            gradient -= np.mean(gradient, axis=0, keepdims=True)
        evaluation = HierarchicalSCCEvaluation(
            potential,
            0.5 * float(charges @ potential),
            gradient,
            correction.energy_error_bound_hartree,
            1 + correction.accepted_cluster_pairs,
            sum(len(items) ** 2 for items in self.shells_by_atom)
            + correction.exact_shell_pairs,
        )
        self._cached_charges = charges.copy()
        self._cached_evaluation = evaluation
        self._cached_compute_gradient = compute_gradient
        if not compute_gradient:
            self._linear_history_charges.append(charges.copy())
            self._linear_history_potentials.append(potential.copy())
            if len(self._linear_history_charges) > self._linear_history_size:
                self._linear_history_charges.pop(0)
                self._linear_history_potentials.pop(0)
        return evaluation

    def evaluate_exact(
        self, charges: np.ndarray, *, compute_gradient: bool = True
    ) -> HierarchicalSCCEvaluation:
        """Evaluate the operator without Krylov-span reconstruction."""

        return self.evaluate(
            charges,
            compute_gradient=compute_gradient,
            allow_linear_reuse=False,
        )

    def hessian_vector_product(
        self, charges: np.ndarray, direction_bohr: np.ndarray
    ) -> np.ndarray:
        """Apply a hierarchical derivative of the full softened SCC kernel."""

        # A separate derivative tree avoids differentiating an FMM library's
        # source layout.  It remains matrix-free and quasi-linear, while using
        # exactly the softened kernel rather than a truncated Coulomb model.
        if not hasattr(self, "_hessian_operator"):
            self._hessian_operator = HierarchicalShellInteraction(
                self.atomic_coordinates,
                tuple(int(value) for value in self.shell_atom),
                tuple(float(value) for value in self.hardness),
                opening_angle=0.20,
                leaf_size=8,
                kernel_mode="softened",
            )
        return self._hessian_operator.hessian_vector_product(
            charges, direction_bohr
        )

    def potential_direction(
        self,
        charges: np.ndarray,
        charge_direction: np.ndarray,
        direction_bohr: np.ndarray,
    ) -> np.ndarray:
        """Apply the softened-kernel directional potential without dense pairs."""

        if not hasattr(self, "_hessian_operator"):
            self._hessian_operator = HierarchicalShellInteraction(
                self.atomic_coordinates,
                tuple(int(value) for value in self.shell_atom),
                tuple(float(value) for value in self.hardness),
                opening_angle=0.20,
                leaf_size=8,
                kernel_mode="softened",
            )
        return self._hessian_operator.potential_direction(
            charges, charge_direction, direction_bohr
        )

    def matvec(self, charges: np.ndarray) -> np.ndarray:
        return self.evaluate(charges).potential

    def __matmul__(self, charges: np.ndarray) -> np.ndarray:
        return self.matvec(charges)


__all__.append("FMMShellInteraction")
