"""Rigid-frame alignment utilities shared by MATRIX geometry consumers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RotationChart:
    """Continuous optimizer coordinates for a locally rebased SO(3) chart."""

    offset: np.ndarray
    tangent: np.ndarray

    def __post_init__(self) -> None:
        offset = np.asarray(self.offset, dtype=float).reshape(3)
        tangent = np.asarray(self.tangent, dtype=float)
        if tangent.shape != (3, 3) or not np.all(np.isfinite(offset)) or not np.all(np.isfinite(tangent)):
            raise ValueError("rotation chart needs a finite 3-vector and 3 x 3 tangent")
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "tangent", tangent)

    @classmethod
    def identity(cls) -> "RotationChart":
        return cls(np.zeros(3, dtype=float), np.eye(3, dtype=float))

    def value(self, local_rotation_vector: np.ndarray) -> np.ndarray:
        return self.offset + self.tangent @ np.asarray(local_rotation_vector, dtype=float).reshape(3)

    def rows(self, local_rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(local_rows, dtype=float)
        if rows.ndim != 2 or rows.shape[0] != 3:
            raise ValueError("local rotation rows must have shape (3, ncart)")
        return self.tangent @ rows

    def rebase(self, local_rotation_vector: np.ndarray) -> "RotationChart":
        local = np.asarray(local_rotation_vector, dtype=float).reshape(3)
        transport = rotation_composition_jacobian(local)
        return RotationChart(self.value(local), self.tangent @ transport)


def rotation_matrix_from_vector(rotation_vector: np.ndarray) -> np.ndarray:
    """Return the proper row-vector rotation represented by an exponential map."""

    vector = np.asarray(rotation_vector, dtype=float).reshape(3)
    theta = float(np.linalg.norm(vector))
    skew = np.asarray(
        [[0.0, -vector[2], vector[1]], [vector[2], 0.0, -vector[0]], [-vector[1], vector[0], 0.0]],
        dtype=float,
    )
    if theta < 1.0e-8:
        return np.eye(3, dtype=float) - skew + 0.5 * (skew @ skew)
    return (
        np.eye(3, dtype=float)
        - (np.sin(theta) / theta) * skew
        + ((1.0 - np.cos(theta)) / (theta * theta)) * (skew @ skew)
    )


def rotation_vector_from_matrix(rotation: np.ndarray) -> np.ndarray:
    """Return MATRIX's row-vector exponential-map coordinates for a rotation."""

    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("rotation must be a finite 3 x 3 matrix")
    u_matrix, _singular, vt_matrix = np.linalg.svd(matrix)
    matrix = u_matrix @ vt_matrix
    if np.linalg.det(matrix) < 0.0:
        u_matrix[:, -1] *= -1.0
        matrix = u_matrix @ vt_matrix
    theta = float(np.arccos(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0)))
    antisymmetric = np.asarray(
        [matrix[1, 2] - matrix[2, 1], matrix[2, 0] - matrix[0, 2], matrix[0, 1] - matrix[1, 0]],
        dtype=float,
    )
    if theta < 1.0e-8:
        return 0.5 * antisymmetric
    if np.pi - theta < 1.0e-6:
        symmetric = 0.5 * (matrix + np.eye(3, dtype=float))
        axis = np.sqrt(np.maximum(np.diag(symmetric), 0.0))
        pivot = int(np.argmax(axis))
        if axis[pivot] <= 1.0e-10:
            raise ValueError("rotation logarithm is singular at 180 degrees")
        for index in range(3):
            if index != pivot:
                axis[index] = symmetric[pivot, index] / axis[pivot]
        axis /= np.linalg.norm(axis)
        if float(np.dot(axis, antisymmetric)) < 0.0:
            axis *= -1.0
        return theta * axis
    return (theta / (2.0 * np.sin(theta))) * antisymmetric


def rotation_composition_jacobian(
    base_rotation_vector: np.ndarray,
) -> np.ndarray:
    """Analytic Jacobian of ``log(exp(local) exp(base))`` at ``local = 0``.

    This is the inverse right Jacobian of SO(3) in MATRIX's row-vector
    convention.  The small-angle branch is its Taylor series, not a numerical
    finite difference.
    """

    base = np.asarray(base_rotation_vector, dtype=float).reshape(3)
    theta2 = float(np.dot(base, base))
    skew = np.asarray(
        [[0.0, -base[2], base[1]], [base[2], 0.0, -base[0]], [-base[1], base[0], 0.0]],
        dtype=float,
    )
    if theta2 < 1.0e-12:
        coefficient = 1.0 / 12.0 + theta2 / 720.0 + theta2 * theta2 / 30240.0
    else:
        theta = float(np.sqrt(theta2))
        if abs(np.sin(theta)) <= 1.0e-12:
            raise ValueError("SO(3) composition Jacobian is singular")
        coefficient = 1.0 / theta2 - (1.0 + np.cos(theta)) / (
            2.0 * theta * np.sin(theta)
        )
    return np.eye(3, dtype=float) + 0.5 * skew + coefficient * (skew @ skew)


def principal_axis_orientation(
    coordinates: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Place a geometry at its weighted center and on principal inertia axes.

    The returned axes are ordered by increasing principal moment.  Axis signs
    are fixed from the most distant weighted atom and the final frame is made
    right handed.  Exactly degenerate principal moments do not define a unique
    physical frame; the eigensolver's orthogonal representative is used there.
    """

    oriented, _rotation = principal_axis_frame(coordinates, weights)
    return oriented


def principal_axis_frame(
    coordinates: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return principal-axis coordinates and their proper row rotation.

    The rotation maps weighted-centered input row vectors into the returned
    canonical frame, so vectors and Cartesian bases can use the same frame.
    """

    coords = _cartesian_coordinates(coordinates, name="coordinates")
    masses = np.asarray(weights, dtype=float).reshape(-1)
    if masses.shape != (coords.shape[0],):
        raise ValueError("weights must contain one value per atom")
    if not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("weights must be finite and positive")
    center = np.average(coords, axis=0, weights=masses)
    centered = coords - center
    inertia = np.zeros((3, 3), dtype=float)
    for weight, vector in zip(masses, centered, strict=True):
        inertia += weight * (
            np.dot(vector, vector) * np.eye(3, dtype=float) - np.outer(vector, vector)
        )
    try:
        _moments, axes = np.linalg.eigh(inertia)
    except np.linalg.LinAlgError as exc:
        raise ValueError("principal-axis orientation failed") from exc
    oriented = centered @ axes
    for axis in range(3):
        scores = masses * np.abs(oriented[:, axis])
        pivot = int(np.argmax(scores))
        if oriented[pivot, axis] < 0.0:
            axes[:, axis] *= -1.0
            oriented[:, axis] *= -1.0
    if np.linalg.det(axes) < 0.0:
        axes[:, -1] *= -1.0
        oriented[:, -1] *= -1.0
    return oriented, axes


def kabsch_align(
    moving_coordinates: np.ndarray,
    reference_coordinates: np.ndarray,
) -> np.ndarray:
    """Return ``moving_coordinates`` rigidly aligned onto ``reference_coordinates``.

    Reflections are excluded.  A single Cartesian point has no meaningful
    rotational frame, so it is returned unchanged to preserve single-particle
    displacement semantics.
    """

    moving = _cartesian_coordinates(moving_coordinates, name="moving_coordinates")
    reference = _cartesian_coordinates(reference_coordinates, name="reference_coordinates")
    if moving.shape != reference.shape:
        raise ValueError("moving and reference coordinates must have the same shape")
    if moving.shape[0] < 2:
        return moving.copy()
    moving_center = np.mean(moving, axis=0)
    reference_center = np.mean(reference, axis=0)
    rotation = kabsch_rotation(moving, reference)
    return (moving - moving_center) @ rotation + reference_center


def kabsch_rotation(
    moving_coordinates: np.ndarray,
    reference_coordinates: np.ndarray,
) -> np.ndarray:
    """Return the proper rotation mapping ``moving`` vectors into ``reference``."""

    moving = _cartesian_coordinates(moving_coordinates, name="moving_coordinates")
    reference = _cartesian_coordinates(reference_coordinates, name="reference_coordinates")
    if moving.shape != reference.shape:
        raise ValueError("moving and reference coordinates must have the same shape")
    if moving.shape[0] < 2:
        return np.eye(3, dtype=float)
    moving_centered = moving - np.mean(moving, axis=0)
    reference_centered = reference - np.mean(reference, axis=0)
    try:
        u_matrix, _singular, vt_matrix = np.linalg.svd(
            moving_centered.T @ reference_centered
        )
    except np.linalg.LinAlgError as exc:
        raise ValueError("Kabsch alignment failed") from exc
    correction = np.eye(3, dtype=float)
    correction[-1, -1] = np.sign(np.linalg.det(u_matrix @ vt_matrix))
    rotation = u_matrix @ correction @ vt_matrix
    if not np.all(np.isfinite(rotation)) or np.linalg.det(rotation) < 0.0:
        raise ValueError("Kabsch alignment produced an invalid rotation")
    return rotation


def rotate_cartesian_derivatives(
    gradient: np.ndarray,
    rotation: np.ndarray,
    hessian: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Rotate Cartesian derivatives into the frame defined by ``rotation``.

    ``rotation`` follows :func:`kabsch_rotation`: row-vector coordinates obey
    ``x_target = x_source @ rotation``.  The gradient and optional Hessian are
    transformed covariantly into the target frame.
    """

    gradient_array = np.asarray(gradient, dtype=float)
    original_shape = gradient_array.shape
    gradient_rows = gradient_array.reshape((-1, 3))
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("rotation must be a finite 3 x 3 matrix")
    rotated_gradient = (gradient_rows @ matrix).reshape(original_shape)
    if hessian is None:
        return rotated_gradient, None
    hessian_array = np.asarray(hessian, dtype=float)
    ncart = gradient_rows.size
    if hessian_array.shape != (ncart, ncart):
        raise ValueError("Cartesian Hessian shape does not match gradient")
    # Coordinates are row vectors: x_target = x_source @ R.  In the
    # flattened column-vector representation the transformation is I ⊗ R.T,
    # so the Hessian must be transformed covariantly with that operator.
    # The opposite order would make the projected Hessian depend on the
    # arbitrary Cartesian orientation of the input geometry.
    transform = np.kron(np.eye(gradient_rows.shape[0]), matrix.T)
    rotated_hessian = transform @ hessian_array @ transform.T
    return rotated_gradient, 0.5 * (rotated_hessian + rotated_hessian.T)


def aligned_cartesian_displacement(
    before_coordinates: np.ndarray,
    after_coordinates: np.ndarray,
) -> np.ndarray:
    """Return the Cartesian displacement after removing overall translation/rotation."""

    before = _cartesian_coordinates(before_coordinates, name="before_coordinates")
    after = _cartesian_coordinates(after_coordinates, name="after_coordinates")
    if before.shape != after.shape:
        raise ValueError("before and after coordinates must have the same shape")
    if before.shape[0] < 2:
        return after - before
    return kabsch_align(after, before) - before


def _cartesian_coordinates(values: np.ndarray, *, name: str) -> np.ndarray:
    coordinates = np.asarray(values, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(f"{name} must have shape (natom, 3)")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError(f"{name} contains non-finite values")
    return coordinates
